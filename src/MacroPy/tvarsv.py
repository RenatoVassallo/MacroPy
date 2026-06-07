"""
Threshold VAR with stochastic volatility (Alessandri & Mumtaz, 2019).

This module implements the nonlinear VAR of Alessandri, P. and Mumtaz, H. (2019),
"Financial regimes and uncertainty shocks", *Journal of Monetary Economics* 101,
31-46. The model combines three ingredients:

1. **Two regimes** (calm / crisis) selected by a *threshold* on a financial
   distress indicator observed with an unknown *delay* ``d``.
2. **Stochastic volatility**: a single scalar factor ``lambda_t`` (AR(1) in logs)
   scales the whole covariance matrix of the structural shocks.
3. **Volatility-in-mean**: contemporaneous and lagged ``ln lambda_t`` enter the
   VAR mean equations, so an uncertainty shock has first-moment effects.

Model (eq. 1-4 of the paper), with ``S_t = 1`` in the calm regime::

    Z_t = ( c_1 + sum_j B_1j Z_{t-j} + sum_j G_1j ln L_{t-j} + Omega_1t^{1/2} e_t ) * S_t
        + ( c_2 + sum_j B_2j Z_{t-j} + sum_j G_2j ln L_{t-j} + Omega_2t^{1/2} e_t ) * (1 - S_t)

    S_t = 1   <=>   F_{t-d} <= Z*                          (threshold, eq. 2)

    Omega_it = A_i^{-1} H_t A_i^{-T},   H_t = L_t * diag(s_1, ..., s_N)   (eq. 3-4)
    ln L_t   = alpha + F ln L_{t-1} + eta_t,   eta_t ~ N(0, Q)

where ``Z_t`` collects ``N`` endogenous variables (the threshold variable ``F_t``
is one of them) and ``L_t == lambda_t``.

Estimation is a Gibbs sampler (the steps mirror the authors' replication code):

* VAR coefficients per regime, drawn from a Normal after a GLS transform that
  removes the heteroskedasticity implied by ``lambda_t`` (Minnesota dummy prior);
* contemporaneous matrices ``A_1, A_2`` (Normal) and loadings ``s_j`` (inverse
  Gamma), conditional on regime-specific residuals;
* the volatility AR parameters ``F, Q`` and unconditional mean;
* the volatility path ``lambda_t`` via an independence Metropolis step
  (Jacquier-Polson-Rossi);
* the threshold ``Z*`` via a random-walk Metropolis step and the delay ``d`` from
  its multinomial full conditional (Chen-Lee).

Generalized impulse responses (Koop-Pesaran-Potter, 1996) are computed by Monte
Carlo and reported separately by regime; an uncertainty shock is a one-standard
deviation innovation ``eta_t`` to the log-volatility.

The public entry point is the :class:`ThresholdVARSV` class, designed to mirror
the look and feel of :class:`MacroPy.BayesianVAR`.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from numpy.linalg import inv, eigvals, lstsq, solve

from .summary import generate_tvarsv_summary

try:  # joblib ships with most scientific stacks; degrade gracefully if absent
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_JOBLIB = False


# ===========================================================================
# Low-level numerical helpers (direct translations of the authors' functions)
# ===========================================================================


def _chofac(n: int, vec: np.ndarray) -> np.ndarray:
    """Build a unit lower-triangular matrix from its free sub-diagonal elements.

    Mirrors Cogley's ``chofac.m``: ``vec`` fills row 2 col 1, then row 3 cols
    1-2, etc. The diagonal is one.
    """
    cf = np.eye(n)
    i = 0
    for j in range(1, n):
        for k in range(j):
            cf[j, k] = vec[i]
            i += 1
    return cf


def _cholx(a: np.ndarray) -> np.ndarray:
    """Robust Cholesky returning a *lower* factor ``L`` with ``L @ L.T == a``.

    Falls back to the symmetric matrix square root when ``a`` is only positive
    semi-definite (the authors' ``cholx.m`` does the same via ``sqrtm``).
    """
    try:
        return np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        from scipy.linalg import sqrtm
        s = np.real(sqrtm(a))
        return s


def _ig(v0: float, d0: float, x: np.ndarray, rng: np.random.Generator) -> float:
    """Inverse-Gamma posterior draw (``IG.m``).

    Prior degrees of freedom ``v0`` and scale ``d0``; ``x`` is the vector of
    innovations. Returns ``(d0 + x'x) / chi2(v0 + len(x))``.
    """
    v1 = int(round(v0 + x.shape[0]))
    d1 = d0 + float(x @ x)
    z = rng.standard_normal(v1)
    return d1 / float(z @ z)


def _getreg(y: np.ndarray, X: np.ndarray, b0: np.ndarray, Sigma0: np.ndarray,
            sigma2: float, rng: np.random.Generator) -> np.ndarray:
    """Bayesian linear regression draw with known scale ``sigma2`` (``getreg.m``)."""
    iS0 = inv(Sigma0)
    V = inv(iS0 + (X.T @ X) / sigma2)
    M = V @ (iS0 @ b0 + (X.T @ y) / sigma2)
    return M + _cholx(V) @ rng.standard_normal(M.shape[0])


def _logdet_spd(a: np.ndarray) -> float:
    """Log-determinant of a symmetric positive-definite matrix via Cholesky."""
    L = np.linalg.cholesky(a)
    return 2.0 * float(np.sum(np.log(np.diag(L))))


# ===========================================================================
# Minnesota dummy-observation prior with volatility-in-mean terms
# ===========================================================================


def _create_dummies(lamda: float, tau: float, delta: np.ndarray, epsilon: float,
                    p: int, mu: np.ndarray, sigma: np.ndarray, n: int, ph: int,
                    epsilonH: float):
    """Banbura-Giannone-Reichlin dummy observations (``create_dummiesSS.m``).

    Adds ``ph`` (= number of volatility-in-mean terms) loose dummies and one
    constant dummy after the standard lag block.
    """
    yd1 = np.vstack([np.diag(sigma * delta) / lamda, np.zeros((n * (p - 1), n))])
    yd1 = np.vstack([yd1, np.zeros((ph, n)), np.zeros((1, n))])

    jp = np.diag(np.arange(1, p + 1).astype(float))
    xd1 = np.kron(jp, np.diag(sigma) / lamda)
    for _ in range(ph):
        r, c = xd1.shape
        xd1 = np.vstack([
            np.hstack([xd1, np.zeros((r, 1))]),
            np.hstack([np.zeros((1, c)), np.array([[1.0 / np.sqrt(epsilonH)]])]),
        ])
    r, c = xd1.shape
    xd1 = np.vstack([
        np.hstack([xd1, np.zeros((r, 1))]),
        np.hstack([np.zeros((1, c)), np.array([[epsilon]])]),
    ])

    if tau > 0:
        yd2 = np.diag(delta * mu) / tau
        xd2 = np.hstack([np.kron(np.ones((1, p)), yd2),
                         np.zeros((n, ph)), np.zeros((n, 1))])
        return np.vstack([yd1, yd2]), np.vstack([xd1, xd2])
    return yd1, xd1


def _get_dummies(lamdaP: float, tauP: float, epsilonP: float, Y: np.ndarray,
                 L: int, ph: int, epsilonH: float, rw: bool):
    """Set up the dummy prior (``getdummiesSS.m``): AR(1) persistence and scale."""
    n = Y.shape[1]
    muP = Y.mean(axis=0)
    sigmaP = np.zeros(n)
    deltaP = np.zeros(n)
    for i in range(n):
        yt = Y[1:, i]
        xt = np.column_stack([Y[:-1, i], np.ones(Y.shape[0] - 1)])
        bt = lstsq(xt, yt, rcond=None)[0]
        et = yt - xt @ bt
        deltaP[i] = min(bt[0], 1.0)
        sigmaP[i] = np.sqrt(et @ et / yt.shape[0])
    if rw:
        deltaP = np.ones(n)
    return _create_dummies(lamdaP, tauP, deltaP, epsilonP, L, muP, sigmaP, n, ph, epsilonH)


# ===========================================================================
# Conditional posteriors used inside the Gibbs sweep
# ===========================================================================


def _stability(beta: np.ndarray, n: int, L: int, ex: int) -> bool:
    """Return True if the VAR companion matrix has an explosive root."""
    FF = np.zeros((n * L, n * L))
    if L > 1:
        FF[n:, : n * (L - 1)] = np.eye(n * (L - 1))
    B = beta.reshape((n * L + ex, n), order="F")
    FF[:n, :] = B[: n * L, :].T
    return np.max(np.abs(eigvals(FF))) > 1.0


def _get_varcoef(y: np.ndarray, x: np.ndarray, yd: np.ndarray, xd: np.ndarray,
                 Sbig: np.ndarray, iamat: np.ndarray, lam: np.ndarray,
                 maxtrys: int, L: int, ex: int, N: int, rng: np.random.Generator):
    """Draw the regime VAR coefficients (``getvarcoef.m``).

    GLS-transforms the data by ``sqrt(lambda_t)`` to remove heteroskedasticity,
    appends the dummy observations and draws from the Normal posterior, retrying
    until a stationary companion form is obtained.
    """
    sl = np.sqrt(lam)
    Y0 = np.vstack([y / sl[:, None], yd])
    X0 = np.vstack([x / sl[:, None], xd])
    XtX = X0.T @ X0
    coef = solve(XtX, X0.T @ Y0)
    mstar = coef.flatten(order="F")
    ixx = inv(XtX)
    sigma = iamat @ np.diag(Sbig) @ iamat.T
    vstar = np.kron(sigma, ixx)
    chol_v = _cholx(vstar)
    k = N * L + ex
    for _ in range(maxtrys):
        beta = mstar + chol_v @ rng.standard_normal(N * k)
        if not _stability(beta, N, L, ex):
            return beta, False
    return mstar, True


def _get_amat(res: np.ndarray, Sbig: np.ndarray, lam: np.ndarray, c0: np.ndarray,
              p00_a: float, N: int, rng: np.random.Generator):
    """Draw the contemporaneous matrix elements for one regime (``getamatonly.m``)."""
    free: List[float] = []
    for j in range(1, N):
        sc = np.sqrt(lam * Sbig[j])
        yt = res[:, j] / sc
        xt = (-res[:, :j]) / sc[:, None]
        a0 = c0[j, :j].copy()
        v00 = np.diag(np.abs(a0) * p00_a + 1e-12)
        a = _getreg(yt, xt, a0, v00, 1.0, rng)
        free.extend(a.tolist())
    free_arr = np.array(free)
    A = _chofac(N, free_arr)
    return inv(A), free_arr


def _get_sbig(res: np.ndarray, Sbig: np.ndarray, lam: np.ndarray, vs0: float,
              sx0: np.ndarray, N: int, free1: np.ndarray, free2: np.ndarray,
              e1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw the volatility loadings ``s_j`` (``getsbigonly.m``); ``s_1`` is fixed at 1."""
    Sbig = Sbig.copy()
    A1 = _chofac(N, free1)
    A2 = _chofac(N, free2)
    sl = np.sqrt(lam)
    for j in range(1, N):
        yt = res[:, j] / sl
        xt = (-res[:, :j]) / sl[:, None]
        a1 = A1[j, :j]
        a2 = A2[j, :j]
        resx = (yt - xt @ a1) * e1 + (yt - xt @ a2) * (1.0 - e1)
        Sbig[j] = _ig(vs0, sx0[j], resx, rng)
    return Sbig


def _get_vol_transition(lnh: np.ndarray, meanvol: float, F0: float, S0f: float,
                        Q_cur: float, g0: float, vg0: float, MUF0: float,
                        SMUF0: float, rng: np.random.Generator):
    """Draw ``F``, ``Q`` and the intercept of the log-volatility AR(1)."""
    yd = lnh - meanvol
    y = yd[1:]
    x = yd[:-1]
    iS = 1.0 / S0f
    V = 1.0 / (iS + (x @ x) / Q_cur)
    M = V * (iS * F0 + (x @ y) / Q_cur)
    while True:
        F = M + np.sqrt(V) * rng.standard_normal()
        if abs(F) <= 1.0:
            break
    res = y - x * F
    Q = _ig(vg0, g0, res, rng)

    yt = lnh[1:] - lnh[:-1] * F
    xt = np.ones(yt.shape[0]) * (1.0 - F)
    yt = yt / np.sqrt(Q)
    xt = xt / np.sqrt(Q)
    MUJ = _getreg(yt, xt[:, None], np.array([MUF0]),
                  np.array([[SMUF0]]), 1.0, rng)[0]
    mubig = (1.0 - F) * MUJ
    return F, Q, mubig, MUJ


def _sample_vol_path(lam: np.ndarray, F: float, Q: float, mubig: float,
                     h0_prior: float, is0_00: float, y: np.ndarray, x: np.ndarray,
                     B1: np.ndarray, B2: np.ndarray, A1: np.ndarray, A2: np.ndarray,
                     Sbig: np.ndarray, e1: np.ndarray, NL: int, LH: int,
                     rng: np.random.Generator):
    """Single-move independence-Metropolis update of the log-volatility path.

    Translates ``svol1C`` / ``svolttCSSNOTREND`` / ``svoltCSSNOTREND``. The
    proposal is the AR(1) full conditional; acceptance uses the (contemporaneous)
    VAR likelihood. ``lam`` is updated in place and the acceptance count returned.
    """
    T = y.shape[0]
    N = y.shape[1]
    lam = lam.copy()
    sumlogS = float(np.sum(np.log(Sbig)))
    denom = 1.0 + F * F
    vv = Q / denom
    svv = np.sqrt(vv)
    naccept = 0

    # period 0 (initial condition, no observation)
    F2Q = F * F / Q
    vv0 = 1.0 / (is0_00 + F2Q)
    mm0 = vv0 * (is0_00 * np.log(h0_prior) + (np.log(lam[1]) - mubig) * F / Q)
    lam[0] = np.exp(mm0 + np.sqrt(vv0) * rng.standard_normal())

    randu = rng.random(T)
    randn = rng.standard_normal(T)

    for tau in range(1, T + 1):
        i = tau - 1  # observation index governed by period tau
        ln_lag = np.log(lam[tau - 1])
        if tau < T:  # interior period: smoothing conditional uses the lead
            ln_lead = np.log(lam[tau + 1])
            mm = (mubig * (1.0 - F) + F * (ln_lag + ln_lead)) / denom
        else:        # terminal period: no lead term
            mm = (mubig * (1.0 - F) + F * ln_lag) / denom
        cand = np.exp(mm + svv * randn[i])
        if not np.isfinite(cand) or cand <= 0:
            continue

        bb = B1 if e1[i] else B2
        A = A1 if e1[i] else A2

        # volatility-in-mean regressor row: [lags, ln vol terms, constant]
        v0 = lam[tau]
        # lag block of ln-volatility (already-updated values, clipped at period 0)
        vol_lags = [np.log(lam[max(tau - c, 0)]) for c in range(1, LH + 1)]
        xrow_old = np.concatenate([x[i, :NL], [np.log(v0)], vol_lags, [1.0]])
        xrow_new = xrow_old.copy()
        xrow_new[NL] = np.log(cand)

        res_old = y[i] - xrow_old @ bb
        res_new = y[i] - xrow_new @ bb
        a_old = A @ res_old
        a_new = A @ res_new
        lp_old = -0.5 * (N * np.log(v0) + sumlogS) - 0.5 * float(np.sum(a_old * a_old / (v0 * Sbig)))
        lp_new = -0.5 * (N * np.log(cand) + sumlogS) - 0.5 * float(np.sum(a_new * a_new / (cand * Sbig)))

        if np.log(randu[i] + 1e-300) < (lp_new - lp_old):
            lam[tau] = cand
            naccept += 1

    return lam, naccept


def _loglik_var(beta: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                x: np.ndarray) -> float:
    """Gaussian VAR log-likelihood up to an additive constant (``loglik.m``)."""
    v = y - x @ beta
    isigma = inv(sigma)
    dsig = -_logdet_spd(sigma)
    T = y.shape[0]
    sterm = float(np.einsum("ti,ij,tj->", v, isigma, v))
    return (T / 2.0) * dsig - 0.5 * sterm


def _logmvn(x: float, mu: float, var: float) -> float:
    """Univariate Normal log density (used for the threshold prior)."""
    return -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var) - 0.5 * (x - mu) ** 2 / var


def _getvarpost(y: np.ndarray, xvar: np.ndarray, B1: np.ndarray, B2: np.ndarray,
                Sbig: np.ndarray, iamat1: np.ndarray, iamat2: np.ndarray,
                tar: float, tarmean: float, tarvar: float, Ystar: np.ndarray,
                ncrit: int, lam: np.ndarray) -> float:
    """Joint log-posterior of (threshold, delay) given everything else (``getvarpost.m``)."""
    e1 = Ystar <= tar
    n1 = int(e1.sum())
    n2 = e1.shape[0] - n1
    if n1 < ncrit or n2 < ncrit:
        return -np.inf
    sl = np.sqrt(lam)
    Y = y / sl[:, None]
    X = xvar / sl[:, None]
    s1 = iamat1 @ np.diag(Sbig) @ iamat1.T
    s2 = iamat2 @ np.diag(Sbig) @ iamat2.T
    lik1 = _loglik_var(B1, s1, Y[e1], X[e1])
    lik2 = _loglik_var(B2, s2, Y[~e1], X[~e1])
    prior = _logmvn(tar, tarmean, tarvar)
    post = lik1 + lik2 + prior
    if not np.isfinite(post):
        return -np.inf
    return post


# ===========================================================================
# Data preparation
# ===========================================================================


def _build_regressors(data: np.ndarray, L: int):
    """Lag matrix ``[Z_{t-1} ... Z_{t-L}, 1]`` aligned with ``data`` rows >= L."""
    T_full, N = data.shape
    X = np.full((T_full, N * L + 1), np.nan)
    for j in range(1, L + 1):
        X[j:, (j - 1) * N:j * N] = data[:-j, :]
    X[:, -1] = 1.0
    return X


def _threshold_series(data: np.ndarray, tarvar: int, d: int, start: int):
    """Threshold variable lagged by ``d`` aligned to the estimation sample."""
    f = data[:, tarvar]
    return f[start - d: data.shape[0] - d]


# ===========================================================================
# Volatility initialisation (lightweight version of getinitialvolc.m)
# ===========================================================================


def _getsvol(hlast: np.ndarray, g: float, mubar: float, sigmabar: float,
             errors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Univariate stochastic-volatility MH sweep (Jacquier-Polson-Rossi; ``getsvol.m``)."""
    T = errors.shape[0]
    hnew = np.zeros(T + 1)
    hlead = hlast[1]
    ss = sigmabar * g / (g + sigmabar)
    mu = ss * (mubar / sigmabar + np.log(hlead) / g)
    hnew[0] = np.exp(mu + np.sqrt(ss) * rng.standard_normal())
    for i in range(1, T):
        hlead = hlast[i + 1]
        hlag = hnew[i - 1]
        yt = errors[i - 1]
        mu = (np.log(hlead) + np.log(hlag)) / 2.0
        ss = g / 2.0
        htrial = np.exp(mu + np.sqrt(ss) * rng.standard_normal())
        lp1 = -0.5 * np.log(htrial) - yt ** 2 / (2 * htrial)
        lp0 = -0.5 * np.log(hlast[i]) - yt ** 2 / (2 * hlast[i])
        hnew[i] = htrial if np.log(rng.random() + 1e-300) <= (lp1 - lp0) else hlast[i]
    # terminal
    yt = errors[T - 1]
    hlag = hnew[T - 1]
    htrial = np.exp(np.log(hlag) + np.sqrt(g) * rng.standard_normal())
    lp1 = -0.5 * np.log(htrial) - yt ** 2 / (2 * htrial)
    lp0 = -0.5 * np.log(hlast[T]) - yt ** 2 / (2 * hlast[T])
    hnew[T] = htrial if np.log(rng.random() + 1e-300) <= (lp1 - lp0) else hlast[T]
    return hnew


def _initialize_volatility(y: np.ndarray, N: int, reps: int, burn: int,
                           rng: np.random.Generator):
    """Starting values and a few priors for the main sampler.

    Faithful but compact version of ``getinitialvolc.m``: it uses the
    first-differenced data as a rough residual proxy, draws the contemporaneous
    matrix and per-equation stochastic volatilities, and aggregates them into a
    common factor (first principal component) with equation-specific loadings.

    Returns a dict with ``lam`` (common volatility path, length T+1), ``Sbig``
    (loadings), ``Qbig`` / ``g0`` (volatility-innovation scale), ``c0`` (prior
    mean of the contemporaneous matrix) and ``sx0`` (loading prior scale).
    """
    T = y.shape[0]
    errors = np.diff(y, axis=0)
    errors = np.vstack([errors[0:1, :], errors])  # length T

    s0 = (errors.T @ errors) / T
    C0 = np.linalg.cholesky(s0).T          # upper, matches MATLAB chol
    C0 = inv(C0 / np.diag(C0)[None, :]).T  # normalised -> prior mean for A
    SC0 = 10.0
    MU0 = np.log(np.diag(s0))
    SV0 = 1.0

    hlast = np.vstack([(np.diff(y, axis=0) ** 2 + 1e-4)[0:1, :],
                       np.diff(y, axis=0) ** 2 + 1e-4])
    hlast = np.vstack([hlast[0:1, :], hlast])  # length T+1
    g = np.ones(N)
    g0_ig, Tg0 = 1e-4, 1.0

    # burn-in for the per-equation volatilities (matches the 50-step warm start)
    h_cur = hlast.copy()
    for _ in range(50):
        for j in range(N):
            h_cur[:, j] = _getsvol(h_cur[:, j], g[j], np.log(s0[j, j]), 10.0,
                                   errors[:, j], rng)
    hlast = h_cur

    B0h = np.array([0.7, 0.0])
    S0h = np.diag([0.5, 5e-5])
    Fsv = np.zeros(N)
    Musv = np.zeros(N)

    keep_h = np.zeros((T + 1, N))
    keep_g = np.zeros(N)
    keep_A = None
    nkeep = 0
    total = burn + reps
    for it in range(total):
        # contemporaneous matrix from the (fixed) residual proxy
        amatx: List[float] = []
        for j in range(1, N):
            sc = np.sqrt(hlast[1:, j])
            yt = errors[:, j] / sc
            xt = (-errors[:, :j]) / sc[:, None]
            a0 = C0[j, :j].copy()
            v00 = np.diag(np.abs(a0) * SC0 + 1e-12)
            amatx.extend(_getreg(yt, xt, a0, v00, 1.0, rng).tolist())
        amatx = np.array(amatx)
        A = _chofac(N, amatx)
        eps = errors @ A.T

        for j in range(N):
            hlast[:, j] = _getsvol(hlast[:, j], g[j], MU0[j], SV0, eps[:, j], rng)

        gerr = np.zeros((T - 1, N))
        for j in range(N):
            lh = np.log(hlast[:, j])
            yt = lh[1:]
            xt = np.column_stack([lh[:-1], np.ones(T)])
            iS = inv(S0h)
            V = inv(iS + (xt.T @ xt) / g[j])
            M = V @ (iS @ B0h + (xt.T @ yt) / g[j])
            while True:
                bb = M + _cholx(V) @ rng.standard_normal(2)
                if abs(bb[0]) <= 1:
                    break
            Fsv[j], Musv[j] = bb[0], bb[1]
            gerr[:, j] = yt[1:] - xt[1:] @ bb
        for j in range(N):
            g[j] = _ig(Tg0, g0_ig, gerr[:, j], rng)

        if it >= burn:
            keep_h += hlast
            keep_g += g
            keep_A = amatx
            nkeep += 1

    outh = keep_h / nkeep
    outg = keep_g / nkeep
    outA = keep_A

    # common factor: first principal component of standardised log-volatilities
    logh = np.log(outh)
    z = (logh - logh.mean(0)) / logh.std(0)
    C = z.T @ z
    evals, evecs = np.linalg.eigh(C)
    pc = evecs[:, -1] * np.sqrt(N)
    fac = z @ pc / N
    h0 = np.exp(fac)
    load = lstsq(h0[:, None], outh, rcond=None)[0].ravel()
    Sbig = load / load[0]
    Qbig = float(np.mean(outg))

    c0 = _chofac(N, outA)
    return {
        "lam": h0,
        "Sbig": Sbig,
        "Qbig": Qbig,
        "g0": Qbig,
        "c0": c0,
        "sx0": Sbig * 1.0,
        "h0_prior": float(h0[0]),
    }


# ===========================================================================
# One Gibbs chain (module-level so it can be parallelised with joblib)
# ===========================================================================


def _run_chain(seed: int, cfg: dict, show_progress: bool = False) -> dict:
    """Run a single Gibbs chain and return its stored draws."""
    rng = np.random.default_rng(seed)

    data = cfg["data"]
    L = cfg["L"]
    LH = cfg["LH"]
    N = cfg["N"]
    EX = cfg["EX"]
    T0 = cfg["T0"]
    tarvar = cfg["tarvar"]
    max_delay = cfg["max_delay"]
    n_total = cfg["n_total"]    # total sweeps (post_draws, BVAR convention)
    n_burn = cfg["n_burn"]      # burn-in count
    maxdraws = cfg["maxdraws"]
    tarvariance = cfg["tarvariance"]
    p00_a = cfg["p00_a"]
    vs0 = cfg["vs0"]
    vg0 = cfg["vg0"]
    F0 = cfg["F0"]
    S0f = cfg["S0f"]
    MUF0 = cfg["MUF0"]
    SMUF0 = cfg["SMUF0"]

    NL = N * L
    k = NL + EX
    ncrit = NL + EX

    # --- estimation sample: drop L lags and T0 training observations ---------
    start = L + T0
    X_full = _build_regressors(data, L)
    y = data[start:, :]
    x = X_full[start:, :]
    T = y.shape[0]

    Ystar_all = {d: _threshold_series(data, tarvar, d, start) for d in range(1, max_delay + 1)}
    tar_prior_mean = cfg.get("tar_prior_mean", None)
    tarmean = float(np.mean(Ystar_all[1])) if tar_prior_mean is None else float(tar_prior_mean)

    # --- prior: Minnesota dummies -------------------------------------------
    yd, xd = _get_dummies(cfg["lamdaP"], cfg["tauP"], cfg["epsilonP"], y, L,
                          LH + 1, cfg["epsilonH"], cfg["RW"])

    # --- starting values from the volatility initialiser ---------------------
    init = _initialize_volatility(y, N, cfg["init_reps"], cfg["init_burn"], rng)
    lam = np.empty(T + 1)
    lam[:] = init["lam"][: T + 1] if init["lam"].shape[0] >= T + 1 else np.r_[
        init["lam"], np.repeat(init["lam"][-1], T + 1 - init["lam"].shape[0])]
    Sbig = init["Sbig"].copy()
    Qbig = init["Qbig"]
    g0 = init["g0"]
    c0 = init["c0"]
    sx0 = init["sx0"]
    h0_prior = init["h0_prior"]
    is0_00 = 1.0 / 0.1

    F = F0
    mubig = 0.0
    meanvol = 0.0

    def build_xvar(lam_path):
        """Insert the LH+1 log-volatility-in-mean terms before the constant."""
        vm = np.zeros((T, LH + 1))
        for c in range(LH + 1):
            idx = np.clip(np.arange(T) + 1 - c, 0, T)
            vm[:, c] = np.log(lam_path[idx])
        return np.hstack([x[:, :NL], vm, x[:, NL:NL + 1]])

    # initial coefficients / contemporaneous matrices
    xvar = build_xvar(lam)
    coef0 = lstsq(xvar, y, rcond=None)[0]
    B1 = coef0.copy()
    B2 = coef0.copy()
    iamat1 = inv(c0)
    iamat2 = inv(c0)
    free1 = np.zeros(N * (N - 1) // 2)
    free2 = np.zeros(N * (N - 1) // 2)

    cur_delay = 1
    tar = tarmean

    # --- storage -------------------------------------------------------------
    keep = n_total - n_burn
    st_B1 = np.zeros((keep, k, N))
    st_B2 = np.zeros((keep, k, N))
    st_ia1 = np.zeros((keep, N, N))
    st_ia2 = np.zeros((keep, N, N))
    st_Sbig = np.zeros((keep, N))
    st_F = np.zeros(keep)
    st_Q = np.zeros(keep)
    st_mu = np.zeros(keep)
    st_tar = np.zeros(keep)
    st_delay = np.zeros(keep, dtype=int)
    st_lam = np.zeros((keep, T + 1))
    st_e1 = np.zeros((keep, T), dtype=bool)

    tarscale = cfg["tarscale"]
    n_acc_tar = 0
    total = n_total
    stored = 0
    igibbs = 0

    it = tqdm(range(total), desc="Sampling Posterior", disable=not show_progress)
    for sweep in it:
        igibbs += 1
        Ystar = Ystar_all[cur_delay]
        e1 = Ystar <= tar
        lam_obs = lam[1:]

        # split sample by regime
        m1 = e1
        m2 = ~e1
        lam1 = lam_obs[m1]
        lam2 = lam_obs[m2]

        # step 1-2: VAR coefficients per regime
        b1, p1 = _get_varcoef(y[m1], xvar[m1], yd, xd, Sbig, iamat1, lam1,
                              maxdraws, L, EX, N, rng)
        if not p1:
            B1 = b1.reshape((k, N), order="F")
        b2, p2 = _get_varcoef(y[m2], xvar[m2], yd, xd, Sbig, iamat2, lam2,
                              maxdraws, L, EX, N, rng)
        if not p2:
            B2 = b2.reshape((k, N), order="F")

        res1 = y[m1] - xvar[m1] @ B1
        res2 = y[m2] - xvar[m2] @ B2

        # step 3-4: contemporaneous matrices per regime
        iamat1, free1 = _get_amat(res1, Sbig, lam1, c0, p00_a, N, rng)
        iamat2, free2 = _get_amat(res2, Sbig, lam2, c0, p00_a, N, rng)

        # loadings s_j (shared across regimes)
        res_full = (y - xvar @ B1) * m1[:, None] + (y - xvar @ B2) * m2[:, None]
        Sbig = _get_sbig(res_full, Sbig, lam_obs, vs0, sx0, N, free1, free2, e1, rng)

        # step 5: volatility AR parameters
        lnh = np.log(lam)
        F, Qbig, mubig, meanvol = _get_vol_transition(
            lnh, meanvol, F0, S0f, Qbig, g0, vg0, MUF0, SMUF0, rng)

        # step 5: stochastic-volatility path
        lam, _ = _sample_vol_path(lam, F, Qbig, mubig, h0_prior, is0_00, y, x,
                                  B1, B2, iamat1, iamat2, Sbig, e1, NL, LH, rng)
        xvar = build_xvar(lam)
        lam_obs = lam[1:]

        # threshold: random-walk Metropolis
        tar_new = tar + np.sqrt(tarscale) * rng.standard_normal()
        post_new = _getvarpost(y, xvar, B1, B2, Sbig, iamat1, iamat2, tar_new,
                               tarmean, tarvariance, Ystar, ncrit, lam_obs)
        post_old = _getvarpost(y, xvar, B1, B2, Sbig, iamat1, iamat2, tar,
                               tarmean, tarvariance, Ystar, ncrit, lam_obs)
        if np.log(rng.random() + 1e-300) < (post_new - post_old):
            tar = tar_new
            n_acc_tar += 1
        arate = n_acc_tar / igibbs
        if 100 < igibbs < 1000:
            if arate < 0.19:
                tarscale *= 0.9999999
            elif arate > 0.55:
                tarscale *= 1.01

        # delay: multinomial full conditional
        logp = np.array([
            _getvarpost(y, xvar, B1, B2, Sbig, iamat1, iamat2, tar, tarmean,
                        tarvariance, Ystar_all[d], ncrit, lam_obs)
            for d in range(1, max_delay + 1)
        ])
        problem3 = False
        if np.all(~np.isfinite(logp)):
            cur_delay = 1
            problem3 = True
        else:
            w = np.exp(logp - np.nanmax(logp[np.isfinite(logp)]))
            w[~np.isfinite(w)] = 0.0
            s = w.sum()
            if s <= 0 or not np.isfinite(s):
                cur_delay = 1
                problem3 = True
            else:
                cur_delay = int(rng.choice(np.arange(1, max_delay + 1), p=w / s))

        # store
        if sweep >= n_burn and not (p1 or p2 or problem3) and stored < keep:
            st_B1[stored] = B1
            st_B2[stored] = B2
            st_ia1[stored] = iamat1
            st_ia2[stored] = iamat2
            st_Sbig[stored] = Sbig
            st_F[stored] = F
            st_Q[stored] = Qbig
            st_mu[stored] = mubig
            st_tar[stored] = tar
            st_delay[stored] = cur_delay
            st_lam[stored] = lam
            st_e1[stored] = e1
            stored += 1
        if show_progress:
            it.set_postfix(tar=f"{tar:.3f}", d=cur_delay, acc=f"{arate:.2f}")

    sl = slice(0, stored)
    return {
        "B1": st_B1[sl], "B2": st_B2[sl], "iamat1": st_ia1[sl], "iamat2": st_ia2[sl],
        "Sbig": st_Sbig[sl], "F": st_F[sl], "Q": st_Q[sl], "mu": st_mu[sl],
        "tar": st_tar[sl], "delay": st_delay[sl], "lam": st_lam[sl], "e1": st_e1[sl],
        "y": y, "x": x, "T": T,
    }


# ===========================================================================
# Generalized impulse responses (Koop-Pesaran-Potter, 1996)
# ===========================================================================


def _simulate_scenario(Y0: np.ndarray, h0log: np.ndarray, shock: str,
                       shock_idx: int, scale: float, hor: int, L: int, LH: int,
                       N: int, tarvar: int, delay: int, tar: float,
                       B1: np.ndarray, B2: np.ndarray, iamat1: np.ndarray,
                       iamat2: np.ndarray, Sbig: np.ndarray, F: float, Q: float,
                       mubig: float, base_seed: int, irf_1std: int = 1,
                       pre: Optional[int] = None):
    """Simulate one scenario forward and return mean paths of ``y`` and ``ln lambda``.

    ``shock`` is ``"none"`` (baseline), ``"level"`` (a structural shock to variable
    ``shock_idx`` at impact) or ``"vol"`` (a one-SD uncertainty shock to ``eta_0``).
    For a level shock, ``irf_1std=1`` applies a one-standard-deviation structural
    shock, while ``irf_1std=0`` normalises it so the shocked variable moves by
    ``scale`` units on impact (the analogue of ``BayesianVAR``'s ``irf_1std``).
    The RNG is reseeded with ``base_seed`` for every scenario so all scenarios share
    the same future innovations (common random numbers), a variance reduction that
    leaves the expected GIRF unchanged.

    ``pre`` is the size of the pre-sample buffer; it must be at least
    ``max(L, LH, delay)`` so the VAR lags, the volatility-in-mean lags and the
    threshold delay all have enough history. ``Y0`` and ``h0log`` carry ``pre``
    pre-sample values each.
    """
    if pre is None:
        pre = max(L, LH, delay)
    B = Y0.shape[0]
    cQ = np.sqrt(Q)
    sqrtS = np.sqrt(Sbig)
    yhat = np.zeros((B, hor + pre, N))
    yhat[:, :pre, :] = Y0
    hlog = np.zeros((B, hor + pre))
    hlog[:, :pre] = h0log
    rng = np.random.default_rng(base_seed)

    for h in range(hor):
        fi = pre + h
        eta = rng.standard_normal(B)          # drawn every step to keep streams aligned
        z = rng.standard_normal((B, N))
        if h == 0:
            innov = scale * cQ if shock == "vol" else 0.0
            hlog[:, fi] = mubig + F * hlog[:, fi - 1] + innov
        else:
            hlog[:, fi] = mubig + F * hlog[:, fi - 1] + eta * cQ
        lam_fi = np.exp(hlog[:, fi])

        # structural shock vector
        u = z
        if h == 0:
            u = np.zeros((B, N))
            if shock == "level":
                if irf_1std == 1:               # one-standard-deviation shock
                    u[:, shock_idx] = scale
                else:                           # unit shock to the variable itself
                    u[:, shock_idx] = scale / (sqrtS[shock_idx] * np.sqrt(lam_fi))

        # regressor row: [lags Z_{t-1..t-L}, ln-vol terms (0..LH), constant]
        lags = np.concatenate([yhat[:, fi - 1 - j, :] for j in range(L)], axis=1)
        vols = np.stack([hlog[:, fi - c] for c in range(LH + 1)], axis=1)
        xrow = np.concatenate([lags, vols, np.ones((B, 1))], axis=1)

        ystar = yhat[:, fi - delay, tarvar]
        mask1 = ystar <= tar
        scaled = sqrtS[None, :] * np.sqrt(lam_fi)[:, None] * u
        y1 = xrow @ B1 + scaled @ iamat1.T
        y2 = xrow @ B2 + scaled @ iamat2.T
        yhat[:, fi, :] = np.where(mask1[:, None], y1, y2)

    return yhat[:, pre:, :].mean(axis=0), hlog[:, pre:].mean(axis=0)


def _girf_draw(draw: dict, shocks: List[str], reps: int, scale: float, hor: int,
               L: int, LH: int, N: int, tarvar: int, EX: int, seed: int,
               irf_1std: int = 1) -> dict:
    """Generalized IRFs for one posterior draw, averaged by regime of origin.

    ``regime1`` collects histories with ``F_{t-d} <= Z*`` (the low/calm regime)
    and ``regime2`` the high regime. For each, responses are averaged over all
    sample histories in that regime, integrating future shocks by Monte Carlo
    (``reps`` paths per history) with common random numbers across scenarios.
    """
    B1 = draw["B1"]; B2 = draw["B2"]
    iamat1 = draw["iamat1"]; iamat2 = draw["iamat2"]
    Sbig = draw["Sbig"]; F = float(draw["F"]); Q = float(draw["Q"])
    mubig = float(draw["mu"]); tar = float(draw["tar"]); delay = int(draw["delay"])
    lam = draw["lam"]; e1 = draw["e1"]; y = draw["y"]
    lam_obs = lam[1:]
    lnlam = np.log(lam_obs)

    # pre-sample buffer must cover the VAR lags, the vol-in-mean lags and the delay
    pre = max(L, LH, delay)

    out: dict = {}
    for r, (reg, in_regime) in enumerate((("regime1", e1), ("regime2", ~e1))):
        idx = np.where(in_regime)[0]
        idx = idx[idx >= pre - 1]                # need `pre` values of history
        if idx.shape[0] == 0:
            out[reg] = None
            continue
        # build histories: `pre` past observations / log-volatilities ending at t
        Y0 = np.stack([y[t - pre + 1:t + 1, :] for t in idx], axis=0)
        h0 = np.stack([lnlam[t - pre + 1:t + 1] for t in idx], axis=0)
        Y0 = np.repeat(Y0, reps, axis=0)
        h0 = np.repeat(h0, reps, axis=0)
        base_seed = seed + r * 10 ** 6

        base_y, base_h = _simulate_scenario(
            Y0, h0, "none", -1, scale, hor, L, LH, N, tarvar, delay, tar,
            B1, B2, iamat1, iamat2, Sbig, F, Q, mubig, base_seed, irf_1std, pre)

        res = {}
        if "uncertainty" in shocks:
            vy, vh = _simulate_scenario(
                Y0, h0, "vol", -1, scale, hor, L, LH, N, tarvar, delay, tar,
                B1, B2, iamat1, iamat2, Sbig, F, Q, mubig, base_seed, irf_1std, pre)
            res["uncertainty"] = vy - base_y
            res["uncertainty_vol"] = vh - base_h
        if "level" in shocks:
            lev = np.zeros((hor, N, N))           # (horizon, response, shock)
            for s in range(N):
                sy, _ = _simulate_scenario(
                    Y0, h0, "level", s, scale, hor, L, LH, N, tarvar, delay, tar,
                    B1, B2, iamat1, iamat2, Sbig, F, Q, mubig, base_seed, irf_1std, pre)
                lev[:, :, s] = sy - base_y
            res["level"] = lev
        out[reg] = res
    return out


# ===========================================================================
# Public class
# ===========================================================================


class ThresholdVARSV:
    """Threshold VAR with stochastic volatility (Alessandri & Mumtaz, 2019).

    Parameters
    ----------
    y : pd.DataFrame
        Endogenous variables with a datetime index, one column each. The
        threshold variable (financial distress indicator) must be one of them.
    threshold_var : str or int, default last column
        Name or positional index of the variable that drives the regime switch.
    lags : int, default 13
        VAR lags ``P`` of the endogenous variables (monthly benchmark).
    vol_lags : int, default 3
        Lags ``J`` of ``ln lambda_t`` entering the mean (volatility-in-mean).
    max_delay : int, default 12
        Upper bound for the threshold delay ``d``, sampled from its posterior.
    training : int, default 20
        Observations reserved to initialise the stochastic volatility.
    post_draws : int, default 5000
        Total number of Gibbs draws *including* burn-in (same convention as
        ``BayesianVAR``).
    burnin : float, default 0.5
        Proportion of ``post_draws`` to discard as burn-in (e.g. ``0.5`` keeps
        the second half), as in ``BayesianVAR``.
    hor : int, default 48
        Horizon of the generalized impulse responses.
    irf_1std : int, default 1
        IRF shock scaling for *level* (structural) shocks, mirroring
        ``BayesianVAR``: ``1`` = one-standard-deviation shock, ``0`` = unit shock
        (the shocked variable moves by one unit on impact). The uncertainty shock
        is always a one-standard-deviation innovation to ``eta_t``.
    threshold_prior_mean : float, optional
        Prior mean of the threshold ``Z*`` and its starting value. ``None`` (the
        default) uses the sample mean of the threshold variable. Set it together
        with a small ``tarvariance`` in ``prior_params`` for an *informative*
        prior centred on a known cut-off (e.g. ``50`` for a confidence index).
    prior_params : dict, optional
        Overrides for the (Minnesota / volatility) hyperparameters. Notable keys:
        ``tarvariance`` (prior variance of ``Z*``; small = informative) and
        ``tarscale`` (random-walk Metropolis proposal variance for ``Z*``).
    seed : int, optional
        Base random seed (each chain gets ``seed + chain index``).

    Examples
    --------
    >>> from MacroPy import ThresholdVARSV
    >>> model = ThresholdVARSV(df, threshold_var="fci", post_draws=10000, burnin=0.5)
    >>> draws = model.sample_posterior(n_chains=4)
    >>> model.plot_regimes()
    >>> irf = model.compute_irfs(shock="uncertainty")
    >>> model.plot_irfs(shock="uncertainty")
    """

    def __init__(
        self,
        y: pd.DataFrame,
        threshold_var: Optional[Union[str, int]] = None,
        lags: int = 13,
        vol_lags: int = 3,
        max_delay: int = 12,
        training: int = 20,
        post_draws: int = 5000,
        burnin: float = 0.5,
        hor: int = 48,
        irf_1std: int = 1,
        threshold_prior_mean: Optional[float] = None,
        prior_params: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        if not isinstance(y, pd.DataFrame):
            raise ValueError("`y` must be a pandas DataFrame with a datetime index.")
        if y.isna().any().any():
            raise ValueError("`y` cannot contain missing values.")

        self.names = list(y.columns)
        self.dates = y.index
        self.y_df = y
        self.data = y.to_numpy(dtype=float)
        self.N = self.data.shape[1]
        self.lags = lags
        self.vol_lags = vol_lags
        self.max_delay = max_delay
        self.training = training
        self.post_draws = post_draws                 # total draws (incl. burn-in)
        self.burnin = int(burnin * post_draws)       # burn-in count
        self.n_draws = self.post_draws - self.burnin  # retained draws (per chain)
        self.hor = hor
        self.irf_1std = irf_1std
        # Prior mean of the threshold. If None, the sample mean of the threshold
        # variable is used (uninformative location); set it (e.g. 50 for a
        # business-confidence index) to centre an informative prior on Z*.
        self.threshold_prior_mean = threshold_prior_mean
        self.seed = seed

        if threshold_var is None:
            self.tarvar = self.N - 1
        elif isinstance(threshold_var, str):
            if threshold_var not in self.names:
                raise ValueError(f"threshold_var '{threshold_var}' not in columns {self.names}.")
            self.tarvar = self.names.index(threshold_var)
        else:
            self.tarvar = int(threshold_var)
        self.threshold_name = self.names[self.tarvar]

        self.EX = (vol_lags + 1) + 1  # volatility-in-mean terms + constant

        defaults = dict(
            lamdaP=0.1, tauP=1.0, epsilonP=1.0, epsilonH=1.0, RW=False,
            maxdraws=1000, tarvariance=10.0, tarscale=0.005,
            p00_a=1.0, vs0=1.0, vg0=5.0, F0=0.9, S0f=1.0, MUF0=0.0, SMUF0=10.0,
            init_reps=200, init_burn=100,
        )
        if prior_params:
            defaults.update(prior_params)
        self.prior_params = defaults

        # sample-size bookkeeping (number of usable observations)
        self.T = self.data.shape[0] - lags - training
        self.k = self.N * lags + self.EX
        self.draws: Optional[dict] = None

    # ------------------------------------------------------------------
    def _cfg(self) -> dict:
        cfg = dict(
            data=self.data, L=self.lags, LH=self.vol_lags, N=self.N, EX=self.EX,
            T0=self.training, tarvar=self.tarvar, max_delay=self.max_delay,
            n_total=self.post_draws, n_burn=self.burnin,
            tar_prior_mean=self.threshold_prior_mean,
        )
        cfg.update(self.prior_params)
        return cfg

    def model_summary(self):
        """Print a summary of the Threshold VAR-SV model."""
        display(generate_tvarsv_summary(self))

    def sample_posterior(self, n_chains: int = 1, n_jobs: int = -1,
                         show_progress: bool = True) -> dict:
        """Run the Gibbs sampler (optionally several chains in parallel).

        Parameters
        ----------
        n_chains : int, default 1
            Number of independent chains. Draws from all chains are pooled.
        n_jobs : int, default -1
            Worker processes for the chains (joblib). ``-1`` uses all cores.
        show_progress : bool, default True
            Display a progress bar: a per-draw bar for a single chain, or a
            chain-completion bar when several chains run in parallel.

        Returns
        -------
        dict
            Pooled posterior draws (also stored on ``self.draws``).
        """
        cfg = self._cfg()
        base = self.seed if self.seed is not None else np.random.SeedSequence().entropy
        seeds = [int((base + i) % (2 ** 32)) for i in range(n_chains)]

        if n_chains == 1:
            chains = [_run_chain(seeds[0], cfg, show_progress=show_progress)]
        elif _HAS_JOBLIB:
            try:  # stream results so a bar can advance as each chain finishes
                gen = Parallel(n_jobs=n_jobs, return_as="generator")(
                    delayed(_run_chain)(s, cfg, False) for s in seeds)
                chains = list(tqdm(gen, total=n_chains, desc="Sampling Posterior",
                                   disable=not show_progress))
            except TypeError:  # joblib < 1.3 has no return_as
                chains = Parallel(n_jobs=n_jobs)(
                    delayed(_run_chain)(s, cfg, False) for s in seeds)
        else:  # pragma: no cover
            chains = [_run_chain(s, cfg, show_progress and i == 0)
                      for i, s in enumerate(seeds)]

        self._chains = chains
        pooled = self._pool_chains(chains)
        self.draws = pooled
        self.n_draws = pooled["tar"].shape[0]
        return pooled

    @staticmethod
    def _pool_chains(chains: List[dict]) -> dict:
        keys_stack = ["B1", "B2", "iamat1", "iamat2", "Sbig", "lam", "e1"]
        keys_cat = ["F", "Q", "mu", "tar", "delay"]
        out = {}
        for kk in keys_stack:
            out[kk] = np.concatenate([c[kk] for c in chains], axis=0)
        for kk in keys_cat:
            out[kk] = np.concatenate([c[kk] for c in chains], axis=0)
        out["y"] = chains[0]["y"]
        out["x"] = chains[0]["x"]
        out["T"] = chains[0]["T"]
        return out

    # ------------------------------------------------------------------
    def threshold_summary(self) -> pd.DataFrame:
        """Posterior summary of the threshold value and delay."""
        self._require_draws()
        tar = self.draws["tar"]
        delay = self.draws["delay"]
        q = np.percentile(tar, [5, 50, 95])
        rows = {
            "threshold (Z*)": [tar.mean(), q[0], q[1], q[2]],
            "delay (d)": [delay.mean(), np.percentile(delay, 5),
                          np.median(delay), np.percentile(delay, 95)],
        }
        return pd.DataFrame(rows, index=["mean", "p5", "p50", "p95"]).T

    def regime_probability(self) -> pd.Series:
        """Posterior probability of the *high* regime (regime 2) per sample date.

        Regime 2 is the state where the threshold variable exceeds the threshold,
        ``F_{t-d} > Z*`` (the "crisis" regime in the financial application).
        """
        self._require_draws()
        prob = 1.0 - self.draws["e1"].mean(axis=0)  # P(regime 2) = P(F_{t-d} > Z*)
        idx = self.dates[self.lags + self.training:]
        return pd.Series(prob, index=idx, name="P(regime 2)")

    def volatility_path(self, cred: float = 0.68) -> pd.DataFrame:
        """Posterior median and band of the common volatility factor ``lambda_t``."""
        self._require_draws()
        lam = self.draws["lam"][:, 1:]  # drop the period-0 initial condition
        lo = (1 - cred) / 2
        med = np.median(lam, axis=0)
        band = np.percentile(lam, [100 * lo, 100 * (1 - lo)], axis=0)
        idx = self.dates[self.lags + self.training:]
        return pd.DataFrame({"median": med, "lower": band[0], "upper": band[1]}, index=idx)

    def _require_draws(self):
        if self.draws is None:
            raise RuntimeError("Call sample_posterior() before requesting results.")

    # ------------------------------------------------------------------
    def _draw_dict(self, i: int) -> dict:
        d = self.draws
        return dict(
            B1=d["B1"][i], B2=d["B2"][i], iamat1=d["iamat1"][i],
            iamat2=d["iamat2"][i], Sbig=d["Sbig"][i], F=d["F"][i], Q=d["Q"][i],
            mu=d["mu"][i], tar=d["tar"][i], delay=d["delay"][i], lam=d["lam"][i],
            e1=d["e1"][i], y=d["y"],
        )

    def compute_irfs(self, shock: Union[str, List[str]] = "uncertainty",
                     reps: int = 25, n_draws: Optional[int] = 200,
                     scale: float = 1.0, n_jobs: int = -1,
                     show_progress: bool = True) -> dict:
        """Generalized impulse responses by regime (regime 1 vs. regime 2).

        Responses are computed as the Monte-Carlo difference between a shocked and
        a baseline simulation (Koop, Pesaran & Potter, 1996), averaged over the
        sample histories that belong to each regime. The headline experiment is a
        one-standard-deviation ``uncertainty`` shock (an innovation to the
        log-volatility ``eta_t``); pass ``shock="level"`` (or a list) to also get
        Cholesky-identified structural shocks to each variable (scaled per the
        model's ``irf_1std`` setting).

        Parameters
        ----------
        shock : {"uncertainty", "level"} or list, default "uncertainty"
            Which experiment(s) to run.
        reps : int, default 25
            Monte-Carlo paths per history (integrates out future shocks).
        n_draws : int, optional, default 200
            Number of posterior draws used (evenly thinned). ``None`` uses all.
        scale : float, default 1.0
            Shock size (in standard deviations when ``irf_1std=1``).
        n_jobs : int, default -1
            Parallel workers over draws (joblib).

        Returns
        -------
        dict
            ``{"regime1": {shock: array}, "regime2": {...}}`` where ``regime1`` is
            the low regime (``F_{t-d} <= Z*``) and ``regime2`` the high regime.
            Response arrays are ``(n_draws, hor, N)`` for the uncertainty shock
            (plus ``"uncertainty_vol"`` of shape ``(n_draws, hor)`` for the
            volatility's own response) and ``(n_draws, hor, N, N)`` for level
            shocks (horizon, response, shock). Also stored on ``self.irf``.
        """
        self._require_draws()
        shocks = [shock] if isinstance(shock, str) else list(shock)
        total = self.draws["tar"].shape[0]
        if n_draws is None or n_draws >= total:
            sel = np.arange(total)
        else:
            sel = np.linspace(0, total - 1, n_draws).astype(int)

        base = self.seed if self.seed is not None else 12345
        jobs = [(self._draw_dict(i), shocks, reps, scale, self.hor, self.lags,
                 self.vol_lags, self.N, self.tarvar, self.EX, int(base + 1 + j),
                 self.irf_1std)
                for j, i in enumerate(sel)]

        if _HAS_JOBLIB and n_jobs != 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_girf_draw)(*job) for job in
                tqdm(jobs, desc="Computing IRFs", disable=not show_progress))
        else:
            results = [_girf_draw(*job) for job in
                       tqdm(jobs, desc="Computing IRFs", disable=not show_progress)]

        irf: dict = {"regime1": {}, "regime2": {}}
        for reg in ("regime1", "regime2"):
            valid = [r[reg] for r in results if r[reg] is not None]
            if not valid:
                continue
            if "uncertainty" in shocks:
                irf[reg]["uncertainty"] = np.stack([v["uncertainty"] for v in valid])
                irf[reg]["uncertainty_vol"] = np.stack([v["uncertainty_vol"] for v in valid])
            if "level" in shocks:
                irf[reg]["level"] = np.stack([v["level"] for v in valid])
        self.irf = irf
        return irf

    # ------------------------------------------------------------------
    # Plotting (thin wrappers around plots_tvarsv; imported lazily)
    # ------------------------------------------------------------------
    def plot_regimes(self, **kwargs):
        """Threshold variable with the high regime shaded (general; paper Fig. 1).

        Accepts ``regime_label``, ``prob_label``, ``var_label``, ``title``,
        ``prob_threshold`` to relabel for any application.
        """
        from .plots_tvarsv import plot_regimes
        return plot_regimes(self, **kwargs)

    def plot_volatility(self, **kwargs):
        """Posterior common stochastic-volatility factor (paper Fig. 2)."""
        from .plots_tvarsv import plot_volatility
        return plot_volatility(self, **kwargs)

    def plot_irfs(self, **kwargs):
        """GIRFs, regime 1 vs. regime 2 (paper Fig. 3).

        Accepts ``regime_labels=(str, str)``, ``series_titles``, ``title`` to
        relabel for any application.
        """
        from .plots_tvarsv import plot_irfs
        return plot_irfs(self, **kwargs)


__all__ = ["ThresholdVARSV"]
