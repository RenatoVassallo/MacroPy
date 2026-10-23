"""
Time-varying parameter VARs with stochastic volatility (Chan & Eisenstat, 2018).

Port of the replication code of Chan, J.C.C. and Eisenstat, E. (2018),
"Bayesian model comparison for time-varying parameter VARs with stochastic
volatility", *Journal of Applied Econometrics* 33(4), 509-532. One class,
:class:`TimeVaryingVAR`, covers the seven specifications of the paper's
comparison::

    model       time-varying              constant                 variances
    ---------   -----------------------   ----------------------   ---------
    C           nothing                   B, A                     constant
    C-SV        nothing                   B, A                     SV
    TVP         B_t, A_t                  nothing                  constant
    TVP-SV      B_t, A_t                  nothing                  SV
    TVP-R1-SV   A_t                       B (with intercepts)      SV
    TVP-R2-SV   B_t (with intercepts)     A                        SV
    TVP-R3-SV   intercepts mu_t           lags, A                  SV

Structural form, with ``A_t`` unit lower triangular::

    A_t y_t = mu_t + B_1t y_{t-1} + ... + B_pt y_{t-p} + eps_t,
    eps_t ~ N(0, Sigma_t),   Sigma_t = diag(exp(h_1t), ..., exp(h_nt)).

Stacking the free elements, ``y_t = X_t beta_t + W_t gamma_t + eps_t`` with
``X_t = I_n kron (1, y_{t-1}', ..., y_{t-p}')`` and row ``i`` of ``W_t``
holding ``-y_{jt}``, ``j < i`` (the elements of ``A_t`` below the diagonal).
Time-varying blocks follow random walks ``theta_t = theta_{t-1} + eta_t``,
``eta_t ~ N(0, diag(sigma2_theta))``, started at ``theta_0 ~ N(0, V)``; the
log-volatilities follow ``h_t = h_{t-1} + zeta_t``, ``zeta_t ~ N(0,
diag(sigma2_h))``, started at ``h_0 ~ N(0, V_h)``. The reduced form has
``Omega_t = A_t^{-1} Sigma_t A_t^{-1}'``, so ``A_t^{-1} Sigma_t^{1/2}`` is its
recursive (Cholesky) impact matrix: IRFs, FEVD and the historical
decomposition use that identification, date by date.

Estimation is the authors' Gibbs sampler: the states are drawn jointly by the
precision-based sampler (Chan & Jeliazkov, 2009) and the log-volatilities by
the auxiliary mixture sampler (Kim, Shephard & Chib, 1998). The joint
precision of the ``T k`` states is block tridiagonal, so it is assembled in
LAPACK band storage and factorised in ``O(T k^3)`` without ever forming a
``Tk x Tk`` matrix.

Model comparison follows the paper. The marginal likelihood is estimated by
importance sampling over the hyperparameters, with the states integrated out:
the coefficients analytically and the log-volatilities by a second importance
sampler centred at the mode of ``p(h | y)`` (found by EM). The DIC is built on
the same integrated likelihood. Both are optional and expensive: see
:meth:`TimeVaryingVAR.log_marginal_likelihood` and :meth:`TimeVaryingVAR.dic`.

References
----------
Chan, J.C.C. & Eisenstat, E. (2018). Bayesian model comparison for time-varying
parameter VARs with stochastic volatility. JAE 33(4), 509-532.
Chan, J.C.C. & Jeliazkov, I. (2009). Efficient simulation and integrated
likelihood estimation in state space models. IJMMNO 1, 101-120.
Kim, S., Shephard, N. & Chib, S. (1998). Stochastic volatility: likelihood
inference and comparison with ARCH models. REStud 65, 361-393.
Primiceri, G.E. (2005). Time varying structural vector autoregressions and
monetary policy. REStud 72, 821-852.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from IPython.display import display
from scipy.linalg import cho_solve, solve_triangular
from scipy.linalg.lapack import dpbtrf, dpbtrs, dpotrf, dpotri, dtbtrs
from scipy.special import digamma, gammaln, polygamma
from tqdm import tqdm

from .summary import generate_tvpvar_summary

try:  # joblib ships with most scientific stacks; degrade gracefully if absent
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_JOBLIB = False

_LOG2PI = math.log(2.0 * math.pi)

#: Specifications of Chan & Eisenstat (2018): which block drifts, and SV or not.
MODELS = {
    "C": (None, False),
    "C-SV": (None, True),
    "TVP": ("all", False),
    "TVP-SV": ("all", True),
    "TVP-R1-SV": ("impact", True),
    "TVP-R2-SV": ("coef", True),
    "TVP-R3-SV": ("intercept", True),
}
_ALIASES = {"VAR": "C", "VAR-SV": "C-SV"}

#: Prior hyperparameters (the paper's defaults). Inverse-gamma priors IG(nu, S)
#: are set through their degrees of freedom ``nu`` and their prior mean
#: ``S / (nu - 1)``; standard deviations are given for the state innovations.
TVP_PRIOR_DEFAULTS = {
    "coef_var": 10.0,            # V: constant coefficients and initial states theta_0
    "state_df": 5.0,             # nu of the priors on the state-innovation variances
    "state_sd": 0.01,            # prior-mean sd of lag and A_t innovations
    "intercept_state_sd": 0.1,   # prior-mean sd of intercept innovations
    "h0_var": 10.0,              # V_h: initial log-volatilities h_0
    "vol_df": 5.0,               # nu of the priors on the log-volatility variances
    "vol_var": 0.01,             # prior mean of the log-volatility variances
    "sigma_df": 5.0,             # nu of the priors on constant variances (C, TVP)
    "sigma_var": 1.0,            # prior mean of constant variances (C, TVP)
}

# Kim, Shephard & Chib (1998) seven-component approximation of log chi2(1),
# means already shifted by -1.2704 (as in the authors' SVRW.m).
_KSC_P = np.array([0.0073, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.2575])
_KSC_M = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518,
                   -1.08819]) - 1.2704
_KSC_V = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])
_KSC_LOGC = np.log(_KSC_P) - 0.5 * np.log(_KSC_V)

_RNG_STAGES = {"posterior": 0, "forecast": 1, "ml": 2, "dic": 3}


# ===========================================================================
# Banded linear algebra
# ===========================================================================


class _Band:
    """Index cache for the block-tridiagonal precision of ``k``-dim states.

    Upper band storage follows LAPACK / scipy: ``ab[k + i - j, j] = K[i, j]``
    for ``i <= j``. Within-period blocks occupy rows ``1..k`` and the
    random-walk coupling between consecutive periods row ``0``.
    """

    def __init__(self, T: int, k: int):
        r, c = np.triu_indices(k)
        self.r, self.c = r, c
        self.rows = (k + r - c)[None, :]
        self.cols = (np.arange(T) * k)[:, None] + c[None, :]


def _state_precision(X: np.ndarray, w: np.ndarray, sig2: np.ndarray,
                     band: _Band, G: Optional[np.ndarray] = None) -> np.ndarray:
    """``K = H'S^{-1}H + X'Sigma^{-1}X`` in upper band storage.

    ``X`` is the ``(T, n, k)`` design, ``w`` the ``(T, n)`` inverse variances
    and ``sig2`` the ``k`` state-innovation variances; ``G`` optionally holds
    the data blocks ``X_t' diag(w_t) X_t`` already computed.
    """
    T, _, k = X.shape
    if G is None:
        G = np.matmul(np.swapaxes(X * w[:, :, None], 1, 2), X)      # (T, k, k)
    inv = 1.0 / sig2
    ab = np.zeros((k + 1, T * k))
    ab[k] = np.tile(2.0 * inv, T)
    ab[k, -k:] = inv
    ab[0, k:] = -np.tile(inv, T - 1)
    ab[band.rows, band.cols] += G[:, band.r, band.c]
    return ab


def _inverse_blocks(G: np.ndarray, sig2: np.ndarray, X: Optional[np.ndarray] = None):
    """Blocks of ``K^{-1}`` for the block-tridiagonal state precision (data
    blocks ``G``, random-walk coupling ``-diag(1/sig2)``).

    Forward Schur complements ``S_t`` and the backward recursions
    ``Sigma_tt = S_t^{-1} + A_t Sigma_{t+1,t+1} A_t'`` and
    ``Sigma_ts = A_t Sigma_{t+1,s}`` (``s > t``), with
    ``A_t = S_t^{-1} diag(1/sig2)``. Returns the diagonal blocks ``(T, k, k)``
    in ``O(T k^3)``; with the design ``X`` it also returns the dense
    ``X K^{-1} X'`` (``T n x T n``) in ``O(T^2 k^2 n)`` BLAS work, never
    forming ``K^{-1}`` or ``X'``.
    """
    T, k, _ = G.shape
    e = 1.0 / sig2
    Sinv = np.empty_like(G)
    for t in range(T):
        D = G[t] + np.diag((2.0 if t < T - 1 else 1.0) * e)
        if t:
            D -= e[:, None] * Sinv[t - 1] * e[None, :]
        c, info = dpotrf(D, lower=0, clean=0)
        if info == 0:
            c, info = dpotri(c, lower=0)
        if info != 0:
            raise np.linalg.LinAlgError("state precision is not positive definite")
        Sinv[t] = np.triu(c) + np.triu(c, 1).T
    diag = np.empty_like(G)
    diag[-1] = Sinv[-1]
    if X is None:
        for t in range(T - 2, -1, -1):
            A = Sinv[t] * e[None, :]
            diag[t] = Sinv[t] + A @ diag[t + 1] @ A.T
        return diag
    n = X.shape[1]
    M = np.empty((T * n, T * n))
    R = np.empty((k, T * n))            # R[:, block s] = Sigma_ts X_s' for s >= t
    for t in range(T - 1, -1, -1):
        if t < T - 1:
            A = Sinv[t] * e[None, :]
            diag[t] = Sinv[t] + A @ diag[t + 1] @ A.T
            R[:, (t + 1) * n:] = A @ R[:, (t + 1) * n:]
        R[:, t * n:(t + 1) * n] = diag[t] @ X[t].T
        M[t * n:(t + 1) * n, t * n:] = X[t] @ R[:, t * n:]
    iu = np.triu_indices(T * n, 1)
    M[iu[1], iu[0]] = M[iu]
    return diag, M


def _chol_band(ab: np.ndarray) -> np.ndarray:
    U, info = dpbtrf(ab)
    if info != 0:
        raise np.linalg.LinAlgError("band precision is not positive definite")
    return U


def _rw_quad(x: np.ndarray, x0: np.ndarray, sig2: np.ndarray) -> float:
    """``sum_t (x_t - x_{t-1})' diag(1/sig2) (x_t - x_{t-1})`` with ``x_0 = x0``."""
    d = np.diff(x, axis=0, prepend=x0[None, :])
    return float(np.sum(d * d / sig2))


def _rw_grad(x: np.ndarray, x0: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """``H'S^{-1}H (x - alpha)`` for a random walk started at ``x0``."""
    d = np.diff(x, axis=0, prepend=x0[None, :]) / sig2
    out = d.copy()
    out[:-1] -= d[1:]
    return out


def _tri_band(sig2: float, extra: np.ndarray) -> np.ndarray:
    """Band storage of ``H'H / sig2 + diag(extra)`` (random-walk precision)."""
    ab = np.empty((2, extra.shape[0]))
    ab[1] = 2.0 / sig2 + extra
    ab[1, -1] = 1.0 / sig2 + extra[-1]
    ab[0, 0] = 0.0
    ab[0, 1:] = -1.0 / sig2
    return ab


# ===========================================================================
# Gibbs steps
# ===========================================================================


def _draw_states(ytil, X, w, sig2, theta0, band, rng):
    """Precision-based draw of the random-walk states (``T, k``)."""
    T, _, k = X.shape
    U = _chol_band(_state_precision(X, w, sig2, band))
    d = np.einsum("tik,ti->tk", X, w * ytil)
    d[0] += theta0 / sig2
    mean, _ = dpbtrs(U, d.reshape(-1, 1))
    z, _ = dtbtrs(U, rng.standard_normal((T * k, 1)))
    return (mean + z).reshape(T, k)


def _draw_const(ytil, Z, w, V, rng):
    """Constant coefficients: Normal prior ``N(0, V I)``, known variances."""
    Zw = Z * w[:, :, None]
    K = np.tensordot(Zw, Z, axes=([0, 1], [0, 1]))
    K[np.diag_indices_from(K)] += 1.0 / V
    b = np.tensordot(Zw, ytil, axes=([0, 1], [0, 1]))
    L = np.linalg.cholesky(K)
    mean = cho_solve((L, True), b)
    return mean + solve_triangular(L.T, rng.standard_normal(K.shape[0]), lower=False)


def _draw_logvol(e, h, sig2h, h0, rng):
    """Auxiliary mixture sampler for every log-volatility series (``SVRW.m``)."""
    T, n = e.shape
    ystar = np.log(e * e + 1e-4)
    dev = ystar[:, :, None] - h[:, :, None] - _KSC_M
    logq = _KSC_LOGC - 0.5 * dev * dev / _KSC_V
    cum = np.cumsum(np.exp(logq - logq.max(axis=2, keepdims=True)), axis=2)
    u = rng.random((T, n, 1)) * cum[:, :, -1:]
    S = np.minimum((u > cum).sum(axis=2), 6)
    m, v = _KSC_M[S], _KSC_V[S]
    out = np.empty_like(h)
    for i in range(n):
        U = _chol_band(_tri_band(sig2h[i], 1.0 / v[:, i]))
        b = (ystar[:, i] - m[:, i]) / v[:, i]
        b[0] += h0[i] / sig2h[i]
        mean, _ = dpbtrs(U, b.reshape(-1, 1))
        z, _ = dtbtrs(U, rng.standard_normal((T, 1)))
        out[:, i] = (mean + z).ravel()
    return out


def _draw_initial(x1, sig2, V, rng):
    """Initial state ``x_0`` given ``x_1``: prior ``N(0, V)``."""
    prec = 1.0 / sig2 + 1.0 / V
    return (x1 / sig2) / prec + rng.standard_normal(x1.shape[0]) / np.sqrt(prec)


def _draw_ig_rw(x, x0, df, S, rng):
    """Random-walk innovation variances: ``IG(df + T/2, S + sum d^2 / 2)``."""
    d = np.diff(x, axis=0, prepend=x0[None, :])
    rate = S + 0.5 * np.sum(d * d, axis=0)
    return rate / rng.gamma(df + 0.5 * x.shape[0], 1.0, size=rate.shape)


# ===========================================================================
# Integrated likelihoods (states integrated out)
# ===========================================================================


def _ll_states(ytil, X, w, logvar_sum, sig2, theta0, band):
    """``log p(ytil | variances, sig2_theta, theta_0)``, states integrated out.

    The quadratic form ``y'S^{-1}y + a'Pa - d'K^{-1}d`` is evaluated as the
    weighted residual sum of squares at the conditional mode (all terms
    positive), which avoids the cancellation of two large quadratic forms.
    """
    T, n, k = X.shape
    try:
        U = _chol_band(_state_precision(X, w, sig2, band))
    except np.linalg.LinAlgError:
        return -np.inf
    d = np.einsum("tik,ti->tk", X, w * ytil)
    d[0] += theta0 / sig2
    th, _ = dpbtrs(U, d.reshape(-1, 1))
    th = th.reshape(T, k)
    resid = ytil - np.einsum("tik,tk->ti", X, th)
    quad = float(np.sum(w * resid * resid)) + _rw_quad(th, theta0, sig2)
    return (-0.5 * T * n * _LOG2PI - 0.5 * T * float(np.sum(np.log(sig2)))
            - 0.5 * logvar_sum - float(np.sum(np.log(U[k]))) - 0.5 * quad)


def _logmeanexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    if not np.isfinite(m):
        return m
    return m + math.log(float(np.mean(np.exp(x - m))))


def _adaptive_is(draw_fn, R, adaptive, max_draws):
    """``R`` importance draws; if the variance of their mean exceeds one, add
    ``R * floor(var)`` more (the authors' rule), capped at ``max_draws``."""
    ll = draw_fn(R)
    if adaptive and np.all(np.isfinite(ll)):
        v = float(np.var(ll, ddof=1)) / R
        if v > 1.0:
            extra = min(R * int(v), max_draws - R)
            if extra > 0:
                ll = np.concatenate([ll, draw_fn(extra)])
    return _logmeanexp(ll)


def _logvol_objective(h, z, sig2h, h0):
    """``-1/2 sum h - 1/2 sum exp(-h) z + log p(h)`` up to a constant, per series."""
    d = np.diff(h, axis=0, prepend=h0[None, :])
    return (-0.5 * h.sum(axis=0) - 0.5 * np.sum(np.exp(-h) * z, axis=0)
            - 0.5 * np.sum(d * d, axis=0) / sig2h)


def _newton_logvol(z, sig2h, h0, start, tol, max_iter):
    """Mode of ``-1/2 sum h - 1/2 sum exp(-h) z + log p(h)`` (random-walk prior),
    one tridiagonal Newton system per series. Returns ``(mode, converged)``.

    The objective is strictly concave, so a backtracking line search (step
    halving, series by series, until the objective rises) makes Newton converge
    from any start. Without it, the full steps of the authors' code overshoot
    when a few squared residuals are orders of magnitude above the rest (the
    2020 pandemic quarters) and the search fails in a large share of draws. When
    the full step is accepted, as in well-behaved samples, the iterates are the
    authors'. On failure the best iterate is returned, never a worse point."""
    h = start.copy()
    n = h.shape[1]
    with np.errstate(over="ignore", invalid="ignore"):
        f = _logvol_objective(h, z, sig2h, h0)
        for _ in range(max_iter):
            ez = np.exp(-h) * z
            g = -_rw_grad(h, h0, sig2h) - 0.5 * (1.0 - ez)
            if not (np.all(np.isfinite(ez)) and np.all(np.isfinite(g))):
                return h, False
            step = np.empty_like(h)
            for i in range(n):
                try:
                    U = _chol_band(_tri_band(sig2h[i], 0.5 * ez[:, i]))
                except np.linalg.LinAlgError:
                    return h, False
                s, _ = dpbtrs(U, g[:, i:i + 1])
                step[:, i] = s.ravel()
            if float(np.max(np.abs(step))) < tol:
                return h + step, True
            t = np.ones(n)
            new = h + step
            fn = _logvol_objective(new, z, sig2h, h0)
            for _ in range(60):
                bad = ~(fn >= f)                            # also catches nan
                if not bad.any():
                    break
                t[bad] *= 0.5
                new = h + step * t
                fn = _logvol_objective(new, z, sig2h, h0)
            else:
                return h, False
            h, f = new, fn
    return h, False


def _ll_csv(e2, sig2h, h0, rng, R=10, adaptive=True, max_draws=1000):
    """C-SV: ``log p(y | coefficients, sigma2_h, h_0)``, ``h`` integrated by IS
    around the mode of ``p(h | y, coefficients)`` (``intlike_varsv.m``)."""
    T, n = e2.shape
    start = np.tile(h0, (T, 1))
    ht, _ = _newton_logvol(e2, sig2h, h0, start, 0.01, 100)   # best iterate if not converged
    ez = np.exp(-ht) * e2
    Us = [_chol_band(_tri_band(sig2h[i], 0.5 * ez[:, i])) for i in range(n)]
    half_logdet = float(sum(np.sum(np.log(U[1])) for U in Us))
    c_pri = -0.5 * T * n * _LOG2PI - 0.5 * T * float(np.sum(np.log(sig2h)))
    c_is = -0.5 * T * n * _LOG2PI + half_logdet

    def draws(r):
        Z = rng.standard_normal((n, T, r))
        hc = np.empty((r, T, n))
        for i in range(n):
            x, _ = dtbtrs(Us[i], Z[i])
            hc[:, :, i] = (ht[:, i:i + 1] + x).T
        dh = np.diff(hc, axis=1, prepend=np.broadcast_to(h0, (r, 1, n)))
        ll = (-0.5 * T * n * _LOG2PI - 0.5 * hc.sum(axis=(1, 2))
              - 0.5 * np.sum(np.exp(-hc) * e2, axis=(1, 2)))
        pri = c_pri - 0.5 * np.sum(dh * dh / sig2h, axis=(1, 2))
        isd = c_is - 0.5 * np.sum(Z * Z, axis=(0, 1))
        return ll + pri - isd

    return _adaptive_is(draws, R, adaptive, max_draws)


def _ll_tvpsv(ytil, X, sig2, sig2h, h0, theta0, band, rng, R=10,
              adaptive=True, max_draws=1000):
    """``log p(y | sigma2_theta, sigma2_h, theta_0, h_0)``: the states theta
    integrated analytically and ``h`` by importance sampling around the mode of
    the marginal ``p(h | y)``, found by EM (``intlike_tvpsv.m``).

    The EM steps need only the diagonal blocks of ``K^{-1}``; the Hessian
    correction needs the dense ``(T n, T n)`` matrix ``X K^{-1} X'``. Both come
    from the block-tridiagonal recursions of :func:`_inverse_blocks`.
    """
    T, n, k = X.shape
    Tn = T * n

    def e_step(h):
        w = np.exp(-h)
        G = np.matmul(np.swapaxes(X * w[:, :, None], 1, 2), X)
        U = _chol_band(_state_precision(X, w, sig2, band, G))
        d = np.einsum("tik,ti->tk", X, w * ytil)
        d[0] += theta0 / sig2
        th, _ = dpbtrs(U, d.reshape(-1, 1))
        resid = ytil - np.einsum("tik,tk->ti", X, th.reshape(T, k))
        blocks = _inverse_blocks(G, sig2)               # diag of X K^{-1} X'
        zhat = np.einsum("tik,tkl,til->ti", X, blocks, X) + resid * resid
        return G, zhat

    # EM for the mode; every M-step raises its objective (damped Newton), so the
    # last iterate whose E-step worked is kept even without convergence
    ht = prev = np.tile(h0, (T, 1))
    for _ in range(100):
        try:
            _, zhat = e_step(ht)
        except np.linalg.LinAlgError:
            ht = prev
            break
        new, _ = _newton_logvol(zhat, sig2h, h0, ht, 0.1, 1000)
        err = float(np.max(np.abs(new - ht)))
        prev, ht = ht, new
        if err < 0.1:
            break
    G, zhat = e_step(ht)                                # Hessian at the final mode

    # Negative Hessian of log p(h | y): random-walk prior and expected
    # information (banded part) plus the correction for integrating theta out.
    inv = np.tile(1.0 / sig2h, T)
    Kband = np.zeros((Tn, Tn))
    diag = 2.0 * inv + 0.5 * (np.exp(-ht) * zhat).ravel()
    diag[-n:] -= 1.0 / sig2h
    Kband[np.diag_indices(Tn)] = diag
    Kband[np.arange(Tn - n), np.arange(n, Tn)] = -inv[n:]
    Kband[np.arange(n, Tn), np.arange(Tn - n)] = -inv[n:]
    Z = np.exp(-ht).ravel()[:, None] * _inverse_blocks(G, sig2, X)[1]   # Sigma^{-1} X K^{-1} X'
    Kh = Kband + 0.5 * Z.T * (np.eye(Tn) - Z)
    del Z
    try:
        C = np.linalg.cholesky(Kh)
    except np.linalg.LinAlgError:                       # keep the banded part only
        C = np.linalg.cholesky(Kband)
    del Kh, Kband
    half_logdet = float(np.sum(np.log(np.diag(C))))
    c_pri = -0.5 * Tn * _LOG2PI - 0.5 * T * float(np.sum(np.log(sig2h)))
    c_is = -0.5 * Tn * _LOG2PI + half_logdet
    hflat = ht.ravel()

    def draws(r):
        Zs = rng.standard_normal((Tn, r))
        Hc = hflat[:, None] + solve_triangular(C.T, Zs, lower=False)
        out = np.empty(r)
        for j in range(r):
            hc = Hc[:, j].reshape(T, n)
            out[j] = (_ll_states(ytil, X, np.exp(-hc), float(hc.sum()), sig2, theta0, band)
                      + c_pri - 0.5 * _rw_quad(hc, h0, sig2h)
                      - c_is + 0.5 * float(Zs[:, j] @ Zs[:, j]))
        return out

    return _adaptive_is(draws, R, adaptive, max_draws)


def _integrated_loglik(cfg: dict, par: dict, rng, R=10, adaptive=True) -> float:
    """Integrated log-likelihood of one hyperparameter vector (any model)."""
    y, n = cfg["y"], cfg["n"]
    T = y.shape[0]
    ytil = y - np.einsum("tik,k->ti", cfg["Z"], par["const"]) if cfg["kc"] else y
    try:
        if cfg["ks"] == 0:
            if cfg["sv"]:
                return _ll_csv(ytil * ytil, par["vol_var"], par["h0"], rng, R, adaptive)
            s2 = par["sig2"]
            return float(-0.5 * T * n * _LOG2PI - 0.5 * T * np.sum(np.log(s2))
                         - 0.5 * np.sum(ytil * ytil / s2))
        if cfg["sv"]:
            return _ll_tvpsv(ytil, cfg["X"], par["state_var"], par["vol_var"],
                             par["h0"], par["theta0"], cfg["band"], rng, R, adaptive)
        s2 = par["sig2"]
        w = np.broadcast_to(1.0 / s2, (T, n))
        return _ll_states(ytil, cfg["X"], w, T * float(np.sum(np.log(s2))),
                          par["state_var"], par["theta0"], cfg["band"])
    except np.linalg.LinAlgError:
        return -np.inf


def _loglik_chunk(cfg: dict, pars: dict, seed: int, R: int, adaptive: bool) -> np.ndarray:
    """Integrated log-likelihoods for a chunk of hyperparameter draws (one worker)."""
    rng = np.random.default_rng(seed)
    m = next(iter(pars.values())).shape[0]
    return np.array([_integrated_loglik(cfg, {key: v[j] for key, v in pars.items()},
                                        rng, R, adaptive) for j in range(m)])


# ===========================================================================
# Densities for the importance sampler of the marginal likelihood
# ===========================================================================


def _ig_logpdf(x, df, S):
    """Inverse-gamma log density, summed over the last axis."""
    return np.sum(df * np.log(S) - gammaln(df) - (df + 1.0) * np.log(x) - S / x, axis=-1)


def _normal_logpdf_diag(x, var):
    return np.sum(-0.5 * (_LOG2PI + np.log(var)) - 0.5 * x * x / var, axis=-1)


def _fit_gamma(x: np.ndarray):
    """Maximum-likelihood gamma (shape, rate) fit (Minka's Newton iteration)."""
    m = float(np.mean(x))
    s = math.log(m) - float(np.mean(np.log(x)))
    if s <= 1e-12:                                      # degenerate (constant) draws
        return 1e6, 1e6 / m
    a = (3.0 - s + math.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
    for _ in range(100):
        step = (math.log(a) - float(digamma(a)) - s) / (1.0 / a - float(polygamma(1, a)))
        new = a - step
        if new <= 0.0:
            new = 0.5 * a
        done = abs(new - a) < 1e-10 * a
        a = new
        if done:
            break
    return a, a / m


class _GaussBlock:
    """Gaussian importance density fitted to posterior draws."""

    def __init__(self, draws: np.ndarray):
        self.mean = draws.mean(axis=0)
        cov = np.atleast_2d(np.cov(draws, rowvar=False))
        self.L = np.linalg.cholesky(cov)
        self.k = self.mean.shape[0]

    def draw(self, M, rng):
        return self.mean + rng.standard_normal((M, self.k)) @ self.L.T

    def logpdf(self, x):
        z = solve_triangular(self.L, (x - self.mean).T, lower=True)
        return (-0.5 * self.k * _LOG2PI - float(np.sum(np.log(np.diag(self.L))))
                - 0.5 * np.sum(z * z, axis=0))


class _IGBlock:
    """Independent inverse-gamma importance densities (gamma fits to precisions)."""

    def __init__(self, draws: np.ndarray):
        fits = [_fit_gamma(1.0 / draws[:, j]) for j in range(draws.shape[1])]
        self.df = np.array([f[0] for f in fits])
        self.S = np.array([f[1] for f in fits])

    def draw(self, M, rng):
        return self.S / rng.gamma(self.df, 1.0, size=(M, self.df.shape[0]))

    def logpdf(self, x):
        return _ig_logpdf(x, self.df, self.S)


# ===========================================================================
# Public class
# ===========================================================================


class TimeVaryingVAR:
    """
    Time-varying parameter VARs with stochastic volatility (Chan & Eisenstat, 2018).

    Parameters
    ----------
    y : pd.DataFrame
        Endogenous variables with a datetime index, one column each. The
        column order is the recursive ordering of the structural shocks.
    model : str, default "TVP-SV"
        One of ``"C"``, ``"C-SV"``, ``"TVP"``, ``"TVP-SV"``, ``"TVP-R1-SV"``,
        ``"TVP-R2-SV"``, ``"TVP-R3-SV"`` (``"VAR"`` and ``"VAR-SV"`` are
        aliases of the constant-coefficient models). R1: only the
        contemporaneous matrix ``A_t`` drifts; R2: only the lag coefficients
        and intercepts; R3: only the intercepts.
    lags : int, default 2
        VAR lags ``p``.
    presample : int, optional
        Initial observations held out as initial conditions (default
        ``lags``). Set it to the largest lag length you compare (the paper
        uses 4) so that every specification is estimated on the same sample:
        marginal likelihoods and DICs are only comparable on a common sample.
    prior_params : dict, optional
        Overrides for :data:`TVP_PRIOR_DEFAULTS` (unknown keys raise). The
        defaults are the paper's and they are scale-dependent: the state
        innovations have prior-mean standard deviations of 0.01 for lags and
        ``A_t`` and 0.1 for intercepts, in the units of the data.
    post_draws : int, default 10000
        Total Gibbs sweeps, including burn-in (BayesianVAR convention).
    burnin : float, default 0.5
        Fraction of ``post_draws`` discarded as burn-in.
    hor : int, default 20
        IRF / FEVD horizon (number of steps, impact included).
    fhor : int, default 8
        Forecast horizon.
    irf_1std : int, default 1
        ``1`` = one-standard-deviation structural shocks (date-specific size);
        ``0`` = unit shocks (the shocked variable moves by one on impact).
    seed : int, optional
        Seed for reproducibility; each stage (posterior, forecast, marginal
        likelihood, DIC) draws from its own generator.
    progress : bool, default True
        Show progress bars and the cost notices of the expensive methods.
    b_exo : np.ndarray, optional
        Block exogeneity mask (``n x n`` array of 0/1, the convention of
        ``BayesianVAR``): ``b_exo[i, j] = 1`` excludes the lags of variable
        ``j`` from equation ``i``. Excluded coefficients are fixed at zero
        (dropped from the drifting and the constant blocks), not shrunk by a
        tight prior. Contemporaneous exclusions come from the recursive
        ordering: with the external variables ordered first and a block-
        recursive mask (e.g. ``b_exo[:4, 4:] = 1``), domestic shocks have no
        effect on the external block at any horizon, in every date and model.

    Examples
    --------
    >>> from MacroPy import TimeVaryingVAR
    >>> m = TimeVaryingVAR(df, model="TVP-SV", lags=2, presample=4, seed=1)
    >>> m.sample_posterior()
    >>> irf = m.compute_irfs(dates=["2008Q4", "2019Q4"])
    >>> m.plot_irfs(shock=2)
    >>> m.log_marginal_likelihood(n_draws=2000, n_jobs=-1)   # optional, slow
    """

    def __init__(
        self,
        y: pd.DataFrame,
        model: str = "TVP-SV",
        lags: int = 2,
        presample: Optional[int] = None,
        prior_params: Optional[dict] = None,
        post_draws: int = 10000,
        burnin: float = 0.5,
        hor: int = 20,
        fhor: int = 8,
        irf_1std: int = 1,
        seed: Optional[int] = None,
        progress: bool = True,
        b_exo: Optional[np.ndarray] = None,
    ):
        if not isinstance(y, pd.DataFrame):
            raise ValueError("`y` must be a pandas DataFrame with a datetime index.")
        if y.isna().any().any():
            raise ValueError("`y` cannot contain missing values.")
        key = _ALIASES.get(str(model).upper(), str(model).upper())
        if key not in MODELS:
            raise ValueError(f"Unknown model {model!r}; choose one of {list(MODELS)}.")
        if lags < 1:
            raise ValueError("`lags` must be at least 1.")
        presample = lags if presample is None else int(presample)
        if presample < lags:
            raise ValueError("`presample` must be at least `lags`.")
        if irf_1std not in (0, 1):
            raise ValueError("irf_1std must be 0 or 1")

        self.model = key
        self.drift, self.sv = MODELS[key]
        self.names = [str(c) for c in y.columns]
        self.dates = pd.DatetimeIndex(y.index)
        self.y_df = y
        self.data = y.to_numpy(dtype=float)
        self.n_endo = n = self.data.shape[1]
        self.lags = p = int(lags)
        self.presample = presample
        self.post_draws = int(post_draws)
        self.burnin = int(burnin * post_draws)
        self.n_keep = self.post_draws - self.burnin
        if self.n_keep < 2:
            raise ValueError("`post_draws` and `burnin` leave fewer than two retained draws.")
        self.hor = int(hor)
        self.fhor = int(fhor)
        self.irf_1std = irf_1std
        self.seed = seed
        self.progress = progress
        self.figures: dict = {}
        self.draws: Optional[dict] = None
        self.ml: Optional[dict] = None
        self.dic_result: Optional[dict] = None

        unknown = set(prior_params or {}) - set(TVP_PRIOR_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown prior_params keys {sorted(unknown)}; "
                             f"valid keys: {sorted(TVP_PRIOR_DEFAULTS)}.")
        self.prior_params = {**TVP_PRIOR_DEFAULTS, **(prior_params or {})}

        # --- estimation sample and regressors -------------------------------
        self.yy = self.data[presample:]
        self.yy_dates = self.dates[presample:]
        self.T = T = self.yy.shape[0]
        if T < 10:
            raise ValueError("Too few observations after the presample.")
        xlag = np.ones((T, 1 + n * p))
        for j in range(1, p + 1):
            xlag[:, 1 + (j - 1) * n:1 + j * n] = self.data[presample - j:presample - j + T]
        self.xlag = xlag
        self.m = m = n * (n - 1) // 2
        self.kb = kb = n * (1 + n * p)                     # beta: intercepts and lags
        self.kf = kf = kb + m                              # beta and gamma
        Xf = np.zeros((T, n, kf))
        for i in range(n):
            Xf[:, i, i * (1 + n * p):(i + 1) * (1 + n * p)] = xlag
        pair = 0
        for i in range(1, n):                              # gamma order (2,1), (3,1), (3,2), ...
            for j in range(i):
                Xf[:, i, kb + pair] = -self.yy[:, j]
                pair += 1
        intercept = np.arange(n) * (1 + n * p)
        state_idx = {None: np.array([], dtype=int), "all": np.arange(kf),
                     "impact": kb + np.arange(m), "coef": np.arange(kb),
                     "intercept": intercept}[self.drift]
        # block exogeneity: lag j of variable v in equation i sits at
        # i (1 + n p) + 1 + (j - 1) n + v; those coefficients are fixed at zero
        self.b_exo = None
        restricted = np.array([], dtype=int)
        if b_exo is not None:
            mask = np.asarray(b_exo)
            if mask.shape != (n, n) or not np.isin(mask, (0, 1)).all():
                raise ValueError(f"`b_exo` must be an {n} x {n} array of 0/1.")
            self.b_exo = mask.astype(int)
            eq, var = np.nonzero(self.b_exo)
            restricted = np.sort(np.concatenate(
                [eq * (1 + n * p) + 1 + j * n + var for j in range(p)]))
        self._restricted = restricted
        state_idx = np.setdiff1d(state_idx, restricted)
        self.state_idx = state_idx
        self.const_idx = np.setdiff1d(np.arange(kf), np.union1d(state_idx, restricted))
        self.ks, self.kc = state_idx.shape[0], self.const_idx.shape[0]
        self.X = np.ascontiguousarray(Xf[:, :, state_idx])
        self.Z = np.ascontiguousarray(Xf[:, :, self.const_idx])
        self._intercept_idx = intercept
        self._lag_idx = np.setdiff1d(np.arange(kb), intercept)
        self._a_rows = np.concatenate([np.full(i, i) for i in range(1, n)]).astype(int) if n > 1 \
            else np.array([], dtype=int)
        self._a_cols = np.concatenate([np.arange(i) for i in range(1, n)]).astype(int) if n > 1 \
            else np.array([], dtype=int)

        pp = self.prior_params
        sd = np.where(np.isin(state_idx, intercept), pp["intercept_state_sd"], pp["state_sd"])
        self._state_S = (pp["state_df"] - 1.0) * sd * sd
        self._vol_S = (pp["vol_df"] - 1.0) * pp["vol_var"]
        self._sig_S = (pp["sigma_df"] - 1.0) * pp["sigma_var"]
        self._period_freq = self._infer_period_freq()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _rng(self, stage: str) -> np.random.Generator:
        """Deterministic per-stage generator (fresh entropy without a seed)."""
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng([int(self.seed), _RNG_STAGES[stage]])

    def _seed_sequence(self, stage: str) -> np.random.SeedSequence:
        if self.seed is None:
            return np.random.SeedSequence()
        return np.random.SeedSequence([int(self.seed), _RNG_STAGES[stage], 7])

    def _require_draws(self):
        if self.draws is None:
            raise RuntimeError("Call sample_posterior() before requesting results.")

    def _infer_period_freq(self) -> Optional[str]:
        idx = self.yy_dates
        days = float(np.median(np.diff(idx.values).astype("timedelta64[D]").astype(float)))
        if 80 <= days <= 100:
            return "Q"
        if 27 <= days <= 32:
            return "M"
        if 360 <= days <= 370:
            return "Y"
        return None

    def _date_label(self, pos: int) -> str:
        if self._period_freq:
            return str(self.yy_dates[pos].to_period(self._period_freq))
        return str(self.yy_dates[pos].date())

    def _date_position(self, date) -> int:
        """Position in the estimation sample of a date ('2008Q4', Timestamp, int)."""
        if isinstance(date, (int, np.integer)):
            if not -self.T <= int(date) < self.T:
                raise ValueError(f"Date position {date} outside 0..{self.T - 1}.")
            return int(date) % self.T
        if self._period_freq:
            periods = self.yy_dates.to_period(self._period_freq)
            hit = np.flatnonzero(periods == pd.Period(date, freq=self._period_freq))
        else:
            hit = np.flatnonzero(self.yy_dates == pd.Timestamp(date))
        if hit.size == 0:
            raise ValueError(f"{date!r} is not in the estimation sample "
                             f"({self._date_label(0)} to {self._date_label(self.T - 1)}).")
        return int(hit[0])

    def model_summary(self):
        """Print a summary of the model specification."""
        display(generate_tvpvar_summary(self))

    # ------------------------------------------------------------------
    # Posterior simulation
    # ------------------------------------------------------------------
    def _auto_thin(self, thin: Optional[int]) -> int:
        if thin is None:
            return max(1, math.ceil(self.n_keep / 1000))
        return max(1, int(thin))

    def storage_mb(self, thin: Optional[int] = None) -> float:
        """Memory (MB) of the draws that ``sample_posterior(thin)`` keeps."""
        thin = self._auto_thin(thin)
        n_save = len(range(0, self.n_keep, thin))
        light = self.n_keep * (self.kc + 2 * self.ks + 2 * self.n_endo)
        heavy = n_save * self.T * (self.ks + (self.n_endo if self.sv else 0))
        return 8.0 * (light + heavy) / 2 ** 20

    def sample_posterior(self, thin: Optional[int] = None) -> dict:
        """
        Gibbs sampler of Chan & Eisenstat (2018) for the chosen specification.

        Each sweep draws (1) the random-walk states, jointly over all dates, by
        the banded precision sampler; (2) the constant coefficients from their
        Normal full conditional; (3) the initial states; (4) the
        log-volatilities by the auxiliary mixture sampler, or the constant
        variances; (5) the innovation variances of the states and of the
        log-volatilities from inverse-gamma full conditionals.

        Storage has two layers. The hyperparameters (constant coefficients,
        initial states, innovation variances) are kept at every retained
        sweep, because the marginal likelihood and the DIC use them. The state
        paths (``T x k`` per draw, the bulk of the memory) are kept every
        ``thin``-th retained sweep; ``thin=None`` keeps at most 1000 of them.
        Thinning does not add precision at a given number of sweeps: it
        bounds storage and the cost of every later computation.

        Returns
        -------
        dict
            Posterior draws (also on ``self.draws``): ``const`` (n_keep, kc),
            ``state_var`` and ``theta0`` (n_keep, ks), ``vol_var`` and ``h0``
            (SV) or ``sig2`` (n_keep, n), ``states`` (n_save, T, ks), ``h``
            (n_save, T, n; SV) and ``save_idx`` (the retained sweeps the state
            paths come from).
        """
        thin = self._auto_thin(thin)
        rng = self._rng("posterior")
        pp = self.prior_params
        y, X, Z = self.yy, self.X, self.Z
        T, n, ks, kc = self.T, self.n_endo, self.ks, self.kc
        V, Vh = pp["coef_var"], pp["h0_var"]
        band = _Band(T, ks) if ks else None
        save_idx = np.arange(0, self.n_keep, thin)
        n_save = save_idx.shape[0]
        if self.progress:
            print(f"{self.model}: {self.post_draws} sweeps ({self.n_keep} retained), "
                  f"{n_save} state paths stored (thin={thin}), "
                  f"about {self.storage_mb(thin):.1f} MB.")

        # --- starting values (the authors' choices) -----------------------
        theta0 = np.zeros(ks)
        state_var = np.full(ks, 0.1 if self.drift == "intercept" else 0.01)
        states = np.zeros((T, ks))
        if self.drift == "intercept":
            theta0 = y.mean(axis=0)
            states = np.tile(theta0, (T, 1))
        fit_s = np.einsum("tik,tk->ti", X, states) if ks else np.zeros((T, n))
        const = np.zeros(kc)
        if kc:
            const = np.linalg.lstsq(Z.reshape(T * n, kc), (y - fit_s).ravel(), rcond=None)[0]
        fit_c = np.einsum("tik,k->ti", Z, const) if kc else np.zeros((T, n))
        if self.sv:
            if self.drift == "intercept":
                h0 = np.log(np.mean((y - fit_s - fit_c) ** 2, axis=0))
            else:
                h0 = np.log(np.var(y, axis=0, ddof=1))
            h = np.tile(h0, (T, 1))
            vol_var = np.full(n, {None: 0.05, "intercept": 0.1}.get(self.drift, 0.01))
            w = np.exp(-h)
        else:
            sig2 = np.ones(n)
            w = np.ones((T, n))

        st = {"const": np.zeros((self.n_keep, kc)),
              "state_var": np.zeros((self.n_keep, ks)),
              "theta0": np.zeros((self.n_keep, ks)),
              "states": np.zeros((n_save, T, ks))}
        if self.sv:
            st.update(vol_var=np.zeros((self.n_keep, n)), h0=np.zeros((self.n_keep, n)),
                      h=np.zeros((n_save, T, n)))
        else:
            st["sig2"] = np.zeros((self.n_keep, n))

        j_save = 0
        for sweep in tqdm(range(self.post_draws), desc="Sampling Posterior",
                          disable=not self.progress):
            if ks:
                states = _draw_states(y - fit_c, X, w, state_var, theta0, band, rng)
                fit_s = np.einsum("tik,tk->ti", X, states)
            if kc:
                const = _draw_const(y - fit_s, Z, w, V, rng)
                fit_c = np.einsum("tik,k->ti", Z, const)
            if ks:
                theta0 = _draw_initial(states[0], state_var, V, rng)
            e = y - fit_s - fit_c
            if self.sv:
                h = _draw_logvol(e, h, vol_var, h0, rng)
                h0 = _draw_initial(h[0], vol_var, Vh, rng)
                vol_var = _draw_ig_rw(h, h0, pp["vol_df"], self._vol_S, rng)
                w = np.exp(-h)
            else:
                rate = self._sig_S + 0.5 * np.sum(e * e, axis=0)
                sig2 = rate / rng.gamma(pp["sigma_df"] + 0.5 * T, 1.0, size=n)
                w = np.broadcast_to(1.0 / sig2, (T, n))
            if ks:
                state_var = _draw_ig_rw(states, theta0, pp["state_df"], self._state_S, rng)

            i = sweep - self.burnin
            if i < 0:
                continue
            st["const"][i] = const
            st["state_var"][i] = state_var
            st["theta0"][i] = theta0
            if self.sv:
                st["vol_var"][i] = vol_var
                st["h0"][i] = h0
            else:
                st["sig2"][i] = sig2
            if j_save < n_save and i == save_idx[j_save]:
                st["states"][j_save] = states
                if self.sv:
                    st["h"][j_save] = h
                j_save += 1

        st["save_idx"] = save_idx
        self.thin = thin
        self.draws = st
        return st

    # ------------------------------------------------------------------
    # Posterior objects date by date
    # ------------------------------------------------------------------
    def _saved(self, key: str) -> np.ndarray:
        """Hyperparameter draws aligned with the stored state paths."""
        return self.draws[key][self.draws["save_idx"]]

    def _structural(self, coef: np.ndarray):
        """``(A, B)`` from full coefficient vectors (draws, kf): ``A`` unit lower
        triangular (draws, n, n); ``B`` (draws, n, 1 + n p), intercept first."""
        n = self.n_endo
        ns = coef.shape[0]
        B = coef[:, :self.kb].reshape(ns, n, 1 + n * self.lags)
        A = np.broadcast_to(np.eye(n), (ns, n, n)).copy()
        A[:, self._a_rows, self._a_cols] = coef[:, self.kb:]
        return A, B

    def _coef_at(self, t: int):
        """``(A, B, logvar)`` of every stored draw at date ``t``."""
        ns = self.draws["save_idx"].shape[0]
        coef = np.zeros((ns, self.kf))                      # zeros where b_exo excludes
        if self.ks:
            coef[:, self.state_idx] = self.draws["states"][:, t, :]
        if self.kc:
            coef[:, self.const_idx] = self._saved("const")
        A, B = self._structural(coef)
        logvar = self.draws["h"][:, t, :] if self.sv else np.log(self._saved("sig2"))
        return A, B, logvar

    def _irf_at(self, t: int, hor: int, *, one_sd: Optional[bool] = None) -> np.ndarray:
        """Recursive IRFs (n_save, hor, variable, shock), parameters fixed at ``t``.

        ``one_sd`` sets the impact: ``A_t^{-1} Sigma_t^{1/2}`` (one-standard-
        deviation shocks) when True, ``A_t^{-1}`` (unit shocks) when False, and
        ``self.irf_1std`` when None.
        """
        n, p = self.n_endo, self.lags
        A, B, logvar = self._coef_at(t)
        Ainv = np.linalg.inv(A)
        Phi = np.matmul(Ainv, B[:, :, 1:])                  # (ns, n, n p)
        lagmats = [Phi[:, :, j * n:(j + 1) * n] for j in range(p)]
        one_sd = self.irf_1std == 1 if one_sd is None else bool(one_sd)
        P = Ainv * np.exp(0.5 * logvar)[:, None, :] if one_sd else Ainv
        ns = A.shape[0]
        psi = [np.broadcast_to(np.eye(n), (ns, n, n))]
        out = np.empty((ns, hor, n, n))
        out[:, 0] = P
        for h in range(1, hor):
            acc = np.zeros((ns, n, n))
            for j in range(1, min(p, h) + 1):
                acc += np.matmul(lagmats[j - 1], psi[h - j])
            psi.append(acc)
            out[:, h] = np.matmul(acc, P)
        return out

    def _time_invariant(self) -> bool:
        return self.ks == 0 and not self.sv

    def compute_irfs(self, dates=None, cred_interval: Union[float, List[float]] = 0.68,
                     hor: Optional[int] = None, plot_irfs: bool = False) -> dict:
        """
        Time-varying impulse responses: at each date ``t`` the response to a
        recursive structural shock with the parameters held at their
        date-``t`` values along the horizon (Primiceri, 2005).

        The posterior median and the credible bands are computed for every
        date (they feed the 3D and over-time figures); the full posterior
        draws are kept only for the requested ``dates``, which bounds memory.

        Parameters
        ----------
        dates : list, optional
            Dates whose draws are stored, e.g. ``["2008Q4", "2020Q2"]``
            (period strings, timestamps or integer positions). Default: the
            last date.
        cred_interval : float or list, default 0.68
            Central credible level(s) of the bands.
        hor : int, optional
            Horizon (defaults to the model's ``hor``).
        plot_irfs : bool, default False
            Build :func:`MacroPy.plots_tvpvar.plot_tvp_irfs` for the stored
            dates (``figures["irfs"]``).

        Returns
        -------
        dict
            ``"median"`` (T, hor, variable, shock), ``"bands"`` {level:
            (lower, upper)} of the same shape, ``"draws"`` {date label:
            (n_save, hor, variable, shock)}, ``"dates"`` and ``"levels"``.
            Also on ``self.irf``.
        """
        self._require_draws()
        hor = self.hor if hor is None else int(hor)
        levels = [cred_interval] if isinstance(cred_interval, (int, float)) else list(cred_interval)
        qs = [50.0] + [q for lv in levels for q in (50 - 50 * lv, 50 + 50 * lv)]
        positions = [self.T - 1] if dates is None else [self._date_position(d) for d in dates]
        quant = np.empty((len(qs), self.T, hor, self.n_endo, self.n_endo))
        keep: Dict[str, np.ndarray] = {}
        if self._time_invariant():
            th = self._irf_at(0, hor)
            quant[:] = np.percentile(th, qs, axis=0)[:, None]
            for pos in positions:
                keep[self._date_label(pos)] = th
        else:
            for t in tqdm(range(self.T), desc="Computing IRFs", disable=not self.progress):
                th = self._irf_at(t, hor)
                quant[:, t] = np.percentile(th, qs, axis=0)
                if t in positions:
                    keep[self._date_label(t)] = th
        bands = {lv: (quant[1 + 2 * j], quant[2 + 2 * j]) for j, lv in enumerate(levels)}
        self.irf = {"median": quant[0], "bands": bands, "draws": keep,
                    "dates": self.yy_dates, "levels": levels}
        if plot_irfs:
            from .plots_tvpvar import plot_tvp_irfs
            self.figures["irfs"] = plot_tvp_irfs(self, dates=list(keep))
        return self.irf

    def compute_fevd(self, hor: Optional[int] = None, plot_fevd: bool = False,
                     horizon: Optional[int] = None, series_titles=None, shock_titles=None,
                     title=None) -> dict:
        """
        Time-varying forecast error variance decomposition (posterior mean, %).

        At date ``t`` the share of shock ``j`` in the ``h``-step forecast
        error variance of variable ``i`` uses the date-``t`` IRFs:
        ``sum_{l<h} theta_l[i, j]^2 / sum_{l<h} sum_m theta_l[i, m]^2``.

        The decomposition is only defined for one-standard-deviation shocks,
        ``theta_l = Psi_l A_t^{-1} Sigma_t^{1/2}``, so it always uses them and
        does not depend on ``irf_1std`` (which only sets the scale of
        ``compute_irfs``). Unit shocks would split ``Psi_l A_t^{-1}`` instead,
        whose columns have different variances, and give wrong shares.

        Returns
        -------
        dict
            ``"fevd"`` (T, hor, shock, variable) in percent (for each date,
            the layout of ``BayesianVAR.compute_fevd``) and ``"dates"``. Also
            on ``self.fevd``. ``plot_fevd`` stacks the shares at ``horizon``
            steps over time (``figures["fevd"]``; default 20 steps, or ``hor``
            if shorter). Entry ``[:, k]`` is the ``(k + 1)``-step share.
        """
        self._require_draws()
        hor = self.hor if hor is None else int(hor)
        n = self.n_endo
        out = np.empty((self.T, hor, n, n))
        span = [0] if self._time_invariant() else range(self.T)
        for t in tqdm(span, desc="Computing FEVD", disable=not self.progress):
            th = self._irf_at(t, hor, one_sd=True)          # FEVD needs 1-s.d. shocks
            cum = np.cumsum(th * th, axis=1)                 # (ns, hor, var, shock)
            share = 100.0 * cum / cum.sum(axis=3, keepdims=True)
            out[t] = np.swapaxes(share.mean(axis=0), 1, 2)
        if self._time_invariant():
            out[:] = out[0]
        self.fevd = out
        if plot_fevd:
            from .plots_tvpvar import plot_fevd_over_time
            self.figures["fevd"] = plot_fevd_over_time(
                self, horizon=horizon, series_titles=series_titles,
                shock_titles=shock_titles, title=title)
        return {"fevd": out, "dates": self.yy_dates}

    def compute_hd(self, plot_hd: bool = False, series_titles=None, shock_titles=None,
                   title=None, store_draws: bool = False) -> dict:
        """
        Historical decomposition with date-specific parameters (posterior mean).

        For each stored draw the companion recursion runs with the date-``t``
        reduced form: the baseline propagates the initial conditions and the
        intercepts, and each shock's contribution starts at zero and is fed by
        ``A_t^{-1} e_t``, where ``e_t = A_t y_t - B_t x_t``. The identity
        ``baseline + shock.sum(axis=1) == y`` holds draw by draw, hence also
        for the posterior mean.

        Returns
        -------
        dict
            ``"shock"`` (T, n_shocks, n_endo), ``"baseline"`` (T, n_endo),
            ``"endo"`` and ``"dates"`` (the layout of ``BayesianVAR``). Also
            on ``self.HD``; ``store_draws`` keeps every draw in ``hd_draws``.
        """
        self._require_draws()
        n, p, T = self.n_endo, self.lags, self.T
        ns = self.draws["save_idx"].shape[0]
        # state[d, J, l] = lag-(l+1) value of component J (0 = baseline, 1..n = shocks)
        state = np.zeros((ns, n + 1, p, n))
        for j in range(1, p + 1):
            state[:, 0, j - 1] = self.data[self.presample - j]
        shock_sum = np.zeros((T, n, n))
        base_sum = np.zeros((T, n))
        draws = np.zeros((ns, T, n, n)) if store_draws else None
        for t in tqdm(range(T), desc="Computing HD", disable=not self.progress):
            A, B, _ = self._coef_at(t)
            Ainv = np.linalg.inv(A)
            Phi = np.matmul(Ainv, B[:, :, 1:]).reshape(ns, n, p, n).transpose(0, 2, 1, 3)
            eps = np.einsum("dij,j->di", A, self.yy[t]) - np.einsum("dik,k->di", B, self.xlag[t])
            impulse = np.empty((ns, n + 1, n))
            impulse[:, 0] = np.einsum("dij,dj->di", Ainv, B[:, :, 0])
            impulse[:, 1:] = np.swapaxes(Ainv * eps[:, None, :], 1, 2)   # column j times eps_j
            top = np.einsum("dlij,dJlj->dJi", Phi, state) + impulse
            state[:, :, 1:] = state[:, :, :-1]
            state[:, :, 0] = top
            base_sum[t] = top[:, 0].sum(axis=0)
            shock_sum[t] = top[:, 1:].sum(axis=0)
            if store_draws:
                draws[:, t] = top[:, 1:]
        self.HD = {"shock": shock_sum / ns, "baseline": base_sum / ns,
                   "endo": (base_sum + shock_sum.sum(axis=1)) / ns,
                   "dates": pd.DatetimeIndex(self.yy_dates)}
        if store_draws:
            self.hd_draws = draws
        if plot_hd:
            from .plots import generate_hd_plot
            self.figures["hd"] = generate_hd_plot(self, series_titles, shock_titles, title)
        return self.HD

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------
    def forecast(self, fhor: Optional[int] = None, drift: bool = True, reps: int = 1,
                 plot_forecast: bool = False, cred_interval=(0.68, 0.95),
                 last_k: Optional[int] = None, n_breaks: int = 10,
                 zero_line: bool = False) -> dict:
        """
        Predictive simulation: each stored draw is iterated forward from the
        last observation, feeding its own forecasts back as lags.

        With ``drift=True`` (the predictive density of the model) the
        time-varying coefficients and the log-volatilities keep following
        their random walks over the horizon; ``drift=False`` freezes them at
        their last in-sample values, so only future shocks spread the paths.

        Parameters
        ----------
        fhor : int, optional
            Horizon (defaults to the model's ``fhor``).
        drift : bool, default True
            Let the states evolve over the forecast horizon.
        reps : int, default 1
            Predictive paths per stored draw (smoother bands, more memory).
        plot_forecast : bool, default False
            Build the ``BayesianVAR``-style figure (``figures["forecast"]``);
            :meth:`plot_fan_chart` draws BCRP-style fans.

        Returns
        -------
        dict
            ``"forecast_draws"`` (n_save * reps, fhor, n_endo). Also on
            ``self.forecasts``.
        """
        self._require_draws()
        fhor = self.fhor if fhor is None else int(fhor)
        rng = self._rng("forecast")
        n, p = self.n_endo, self.lags
        ns = self.draws["save_idx"].shape[0] * reps

        def rep(a):
            return np.repeat(a, reps, axis=0)

        if self.ks:
            states = rep(self.draws["states"][:, -1, :])
            state_sd = np.sqrt(rep(self._saved("state_var")))
        const = rep(self._saved("const")) if self.kc else None
        if self.sv:
            logvar = rep(self.draws["h"][:, -1, :])
            vol_sd = np.sqrt(rep(self._saved("vol_var")))
        else:
            logvar = np.log(rep(self._saved("sig2")))
        hist = np.tile(self.yy[::-1][:p][None], (ns, 1, 1))  # (ns, p, n), latest first
        out = np.empty((ns, fhor, n))
        coef = np.zeros((ns, self.kf))
        if self.kc:
            coef[:, self.const_idx] = const
        for h in range(fhor):
            if self.ks:
                if drift:
                    states = states + state_sd * rng.standard_normal(states.shape)
                coef[:, self.state_idx] = states
            if self.sv and drift:
                logvar = logvar + vol_sd * rng.standard_normal(logvar.shape)
            A, B = self._structural(coef)
            x = np.concatenate([np.ones((ns, 1)), hist.reshape(ns, n * p)], axis=1)
            rhs = (np.einsum("dik,dk->di", B, x)
                   + np.exp(0.5 * logvar) * rng.standard_normal((ns, n)))
            ynew = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
            out[:, h] = ynew
            hist = np.concatenate([ynew[:, None], hist[:, :-1]], axis=1)
        self.forecasts = out
        if plot_forecast:
            from .plots import generate_forecast_plots
            self.figures["forecast"] = generate_forecast_plots(
                self, out, cred_interval, last_k, n_breaks, zero_line,
                forecast_type="Unconditional")
        return {"forecast_draws": out}

    def _forecast_index(self, fhor: int):
        step = {"Q": pd.DateOffset(months=3), "M": pd.DateOffset(months=1),
                "Y": pd.DateOffset(years=1)}.get(self._period_freq)
        if step is None:
            return None
        last = self.yy_dates[-1]
        return pd.DatetimeIndex([last + step * h for h in range(1, fhor + 1)])

    def forecast_frame(self, fhor: Optional[int] = None, quantiles=(0.05, 0.16, 0.84, 0.95),
                       drift: bool = True, reps: int = 1) -> pd.DataFrame:
        """Unconditional forecast as a tidy (variable, horizon) DataFrame with
        ``date``, ``mean``, ``median`` and the requested quantiles (the layout
        of ``BayesianVAR.forecast_frame``)."""
        fhor = self.fhor if fhor is None else int(fhor)
        qs = sorted(float(q) for q in quantiles)
        if any(not 0.0 < q < 1.0 for q in qs):
            raise ValueError("Quantiles must lie strictly between 0 and 1.")
        draws = self.forecast(fhor=fhor, drift=drift, reps=reps)["forecast_draws"]
        index = pd.MultiIndex.from_product([self.names, range(1, fhor + 1)],
                                           names=["variable", "horizon"])
        data = {}
        future = self._forecast_index(fhor)
        if future is not None:
            data["date"] = np.tile(future, self.n_endo)
        data["mean"] = draws.mean(axis=0).T.ravel()
        data["median"] = np.median(draws, axis=0).T.ravel()
        for q in qs:
            data[f"q{round(q * 100):02d}"] = np.quantile(draws, q, axis=0).T.ravel()
        return pd.DataFrame(data, index=index)

    # ------------------------------------------------------------------
    # Summaries of the time-varying objects
    # ------------------------------------------------------------------
    def volatility_frame(self, cred: float = 0.68) -> pd.DataFrame:
        """Posterior median and band of the structural shock standard
        deviations ``exp(h_t / 2)`` (constant in the models without SV)."""
        self._require_draws()
        lo = 50 - 50 * cred
        ns = self.draws["save_idx"].shape[0]
        sd = (np.exp(0.5 * self.draws["h"]) if self.sv
              else np.broadcast_to(np.sqrt(self._saved("sig2"))[:, None, :], (ns, self.T, self.n_endo)))
        q = np.percentile(sd, [50, lo, 100 - lo], axis=0)    # (3, T, n)
        cols = pd.MultiIndex.from_product([self.names, ["median", "lower", "upper"]])
        return pd.DataFrame(np.stack(q, axis=2).reshape(self.T, -1),
                            index=self.yy_dates, columns=cols)

    def coefficient_frame(self, block: str = "intercept", cred: float = 0.68) -> pd.DataFrame:
        """Posterior median and band of the structural coefficients over time.

        ``block`` is ``"intercept"`` (``mu_t``), ``"impact"`` (the free
        elements of ``A_t``) or ``"lags"`` (``B_t``, ``eq<-var(-l)``).
        Constant blocks appear as flat paths.
        """
        self._require_draws()
        n, p = self.n_endo, self.lags
        if block == "intercept":
            pos = self._intercept_idx
            labels = [f"mu[{v}]" for v in self.names]
        elif block == "impact":
            pos = self.kb + np.arange(self.m)
            labels = [f"a[{self.names[i]},{self.names[j]}]"
                      for i, j in zip(self._a_rows, self._a_cols)]
        elif block == "lags":
            pos = self._lag_idx
            labels = [f"{self.names[i]}<-{self.names[j]}(-{lag})"
                      for i in range(n) for lag in range(1, p + 1) for j in range(n)]
        else:
            raise ValueError("block must be 'intercept', 'impact' or 'lags'.")
        ns = self.draws["save_idx"].shape[0]
        paths = np.empty((ns, self.T, pos.shape[0]))
        const = self._saved("const") if self.kc else None
        for c, k in enumerate(pos):
            hit = np.flatnonzero(self.state_idx == k)
            fixed = np.flatnonzero(self.const_idx == k)
            if hit.size:
                paths[:, :, c] = self.draws["states"][:, :, hit[0]]
            elif fixed.size:
                paths[:, :, c] = const[:, fixed[0], None]
            else:                                           # excluded by b_exo
                paths[:, :, c] = 0.0
        lo = 50 - 50 * cred
        q = np.percentile(paths, [50, lo, 100 - lo], axis=0)
        cols = pd.MultiIndex.from_product([labels, ["median", "lower", "upper"]])
        return pd.DataFrame(np.stack(q, axis=2).reshape(self.T, -1),
                            index=self.yy_dates, columns=cols)

    # ------------------------------------------------------------------
    # Model comparison (optional, expensive)
    # ------------------------------------------------------------------
    def _lik_cfg(self) -> dict:
        return {"y": self.yy, "n": self.n_endo, "ks": self.ks, "kc": self.kc,
                "sv": self.sv, "Z": self.Z, "X": self.X,
                "band": _Band(self.T, self.ks) if self.ks else None}

    def _blocks(self) -> List[str]:
        keys = ["const"] if self.kc else []
        if self.ks:
            keys += ["state_var", "theta0"]
        keys += ["vol_var", "h0"] if self.sv else ["sig2"]
        return keys

    def _log_prior(self, par: dict) -> np.ndarray:
        pp = self.prior_params
        lp = 0.0
        if self.kc:
            lp = lp + _normal_logpdf_diag(par["const"], pp["coef_var"])
        if self.ks:
            lp = lp + _ig_logpdf(par["state_var"], pp["state_df"], self._state_S)
            lp = lp + _normal_logpdf_diag(par["theta0"], pp["coef_var"])
        if self.sv:
            lp = lp + _ig_logpdf(par["vol_var"], pp["vol_df"], self._vol_S)
            lp = lp + _normal_logpdf_diag(par["h0"], pp["h0_var"])
        else:
            lp = lp + _ig_logpdf(par["sig2"], pp["sigma_df"], self._sig_S)
        return lp

    def _cost_notice(self, label: str, n_eval: int, per_eval: float, n_jobs: int):
        workers = n_jobs if n_jobs > 0 else max(1, (os.cpu_count() or 1) + 1 + n_jobs)
        secs = n_eval * per_eval / max(1, workers)
        mem = ""
        if self.ks and self.sv:
            T, n, k = self.X.shape
            mb = 8.0 * (3 * (T * n) ** 2 + 3 * T * k * k) / 2 ** 20
            mem = f", ~{mb:.0f} MB of dense matrices per worker"
        span = f"{secs:.0f} s" if secs < 90 else f"{secs / 60:.1f} min"
        print(f"{label} ({self.model}): {n_eval} evaluations at ~{per_eval:.3f} s each; "
              f"about {span} with {workers} worker(s){mem}.")

    def _run_logliks(self, pars: dict, stage: str, n_jobs: int, desc: str,
                     label: str) -> np.ndarray:
        """Integrated log-likelihood at every row of ``pars``, in seeded chunks
        of 25 (the result does not depend on ``n_jobs``)."""
        cfg = self._lik_cfg()
        M = next(iter(pars.values())).shape[0]
        chunk = 25
        starts = list(range(0, M, chunk))
        seeds = [int(s.generate_state(1)[0])
                 for s in self._seed_sequence(stage).spawn(len(starts))]

        def part(a):
            return {k: v[a:a + chunk] for k, v in pars.items()}

        t0 = time.perf_counter()
        first = _loglik_chunk(cfg, part(0), seeds[0], 10, True)
        if self.progress:
            self._cost_notice(label, M, (time.perf_counter() - t0) / first.shape[0], n_jobs)
        jobs = list(zip(starts[1:], seeds[1:]))
        if not jobs:
            return first
        it = tqdm(jobs, desc=desc, disable=not self.progress)
        if _HAS_JOBLIB and n_jobs != 1:
            res = Parallel(n_jobs=n_jobs)(
                delayed(_loglik_chunk)(cfg, part(a), s, 10, True) for a, s in it)
        else:
            res = [_loglik_chunk(cfg, part(a), s, 10, True) for a, s in it]
        return np.concatenate([first] + list(res))

    def log_marginal_likelihood(self, n_draws: int = 2000, n_batches: int = 20,
                                n_jobs: int = 1) -> dict:
        """
        Marginal likelihood by importance sampling (Chan & Eisenstat, 2018).

        The importance density is fitted to the retained posterior draws of
        the hyperparameters: Normals for the constant coefficients and the
        initial states, inverse-gammas (gamma fits to the precisions) for
        every variance. For each importance draw the states are integrated
        out: the coefficients analytically and, with SV, the log-volatilities
        by an inner importance sampler centred at the mode of ``p(h | y)``.

        **Cost.** With SV and time-varying coefficients each evaluation runs
        an EM search for the mode of ``p(h | y)`` and builds a dense
        ``(Tn x Tn)`` Hessian, so time and memory grow as ``T^2`` (about 0.03 s
        with ``T = 90``, ``n = 3``, ``p = 2``; 0.14 s with ``T = 240``). The
        other models are one to two orders of magnitude cheaper. A notice with
        the estimated time and memory is printed after the first evaluations;
        ``n_jobs=-1`` uses every core. The paper uses 10000 draws.

        **Caveat.** The numerical standard error measures the importance
        sampling noise given the posterior draws the proposal was fitted to.
        When the posterior is multimodal (in TVP models, several coefficients
        can compete to absorb the same drift) a chain stuck in one mode gives
        a proposal that misses the others and a downward-biased estimate with
        a small NSE. Re-estimate with two or three seeds and compare.

        Parameters
        ----------
        n_draws : int, default 2000
            Importance draws (rounded up to a multiple of ``n_batches``).
        n_batches : int, default 20
            Batches for the numerical standard error (the paper's 20).
        n_jobs : int, default 1
            Parallel workers (joblib); ``-1`` uses all cores.

        **Estimator.** ``"logml"`` is the log of the mean of all importance
        weights; the ``n_batches`` batches give its numerical standard error.
        The paper reports instead the mean of the batch log-means
        (``"logml_batch_mean"``), which is biased downward by about half the
        variance of a batch estimate: with 400 draws (batches of 20) the gap
        reaches 1-1.5 log points on the paper's data, with 2000 about 0.4.

        Returns
        -------
        dict
            ``"logml"``, its numerical standard error ``"nse"``, the paper's
            ``"logml_batch_mean"``, ``"n_draws"`` and the ``"log_weights"``.
            Also on ``self.ml``.
        """
        self._require_draws()
        rng = self._rng("ml")
        M = n_batches * math.ceil(n_draws / n_batches)
        d = self.draws
        blocks = {}
        if self.kc:
            blocks["const"] = _GaussBlock(d["const"])
        if self.ks:
            blocks["state_var"] = _IGBlock(d["state_var"])
            blocks["theta0"] = _GaussBlock(d["theta0"])
        if self.sv:
            blocks["vol_var"] = _IGBlock(d["vol_var"])
            blocks["h0"] = _GaussBlock(d["h0"])
        else:
            blocks["sig2"] = _IGBlock(d["sig2"])
        pars = {k: b.draw(M, rng) for k, b in blocks.items()}
        log_g = sum(b.logpdf(pars[k]) for k, b in blocks.items())
        ll = self._run_logliks(pars, "ml", n_jobs, "Marginal likelihood", "Marginal likelihood")
        w = ll + self._log_prior(pars) - log_g
        per_batch = np.array([_logmeanexp(b) for b in w.reshape(n_batches, -1)])
        self.ml = {"logml": _logmeanexp(w),
                   "nse": float(per_batch.std(ddof=1) / math.sqrt(n_batches)),
                   "logml_batch_mean": float(per_batch.mean()),
                   "n_draws": M, "log_weights": w}
        return self.ml

    def dic(self, simstep: int = 20, n_mean: int = 500, n_batches: int = 10,
            n_jobs: int = 1) -> dict:
        """
        Deviance information criterion on the integrated likelihood (states
        integrated out), as in Chan & Eisenstat (2018): ``DIC = -4 E[log L] +
        2 log L(posterior mean)`` and ``pD = -2 E[log L] + 2 log L(mean)``.

        The expectation uses every ``simstep``-th retained draw (inner
        importance sampling with 10 draws, increased when noisy); the plug-in
        term uses ``n_mean`` inner draws. Each evaluation costs as much as one
        of :meth:`log_marginal_likelihood`; a notice with the estimated time
        is printed. Lower is better.

        Returns
        -------
        dict
            ``"dic"``, ``"pD"``, ``"mean_loglik"``, ``"loglik_at_mean"`` and
            ``"nse"`` (batch-means error of the DIC from its first term). Also
            on ``self.dic_result``.
        """
        self._require_draws()
        idx = np.arange(0, self.n_keep, int(simstep))
        pars = {k: self.draws[k][idx] for k in self._blocks()}
        ll = self._run_logliks(pars, "dic", n_jobs, "Computing DIC", "DIC")
        mean_par = {k: self.draws[k].mean(axis=0) for k in self._blocks()}
        rng = np.random.default_rng(self._seed_sequence("dic").spawn(1)[0])
        ll_bar = _integrated_loglik(self._lik_cfg(), mean_par, rng, R=int(n_mean),
                                    adaptive=False)
        mean_ll = float(np.mean(ll))
        nb = max(2, min(int(n_batches), ll.shape[0] // 2))
        batches = np.array([b.mean() for b in np.array_split(ll, nb)])
        self.dic_result = {"dic": -4.0 * mean_ll + 2.0 * ll_bar,
                           "pD": -2.0 * mean_ll + 2.0 * ll_bar,
                           "mean_loglik": mean_ll, "loglik_at_mean": ll_bar,
                           "nse": float(4.0 * batches.std(ddof=1) / math.sqrt(nb))}
        return self.dic_result

    # ------------------------------------------------------------------
    # Plotting (thin wrappers; implementation in plots_tvpvar)
    # ------------------------------------------------------------------
    def plot_volatility(self, **kwargs):
        """Posterior structural shock volatilities over time."""
        from .plots_tvpvar import plot_tvp_volatility
        return plot_tvp_volatility(self, **kwargs)

    def plot_coefficients(self, **kwargs):
        """Structural coefficients over time (``block="intercept"|"impact"|"lags"``)."""
        from .plots_tvpvar import plot_tvp_coefficients
        return plot_tvp_coefficients(self, **kwargs)

    def plot_irfs(self, **kwargs):
        """IRFs at selected dates (median and bands)."""
        from .plots_tvpvar import plot_tvp_irfs
        return plot_tvp_irfs(self, **kwargs)

    def plot_irf_surface(self, **kwargs):
        """3D surface of the median IRF over dates and horizons."""
        from .plots_tvpvar import plot_irf_surface
        return plot_irf_surface(self, **kwargs)

    def plot_fevd(self, **kwargs):
        """FEVD shares at a given horizon over time."""
        from .plots_tvpvar import plot_fevd_over_time
        return plot_fevd_over_time(self, **kwargs)

    def plot_fan_chart(self, **kwargs):
        """BCRP-style fan charts of the predictive density."""
        from .plots_tvpvar import plot_tvp_fan_chart
        return plot_tvp_fan_chart(self, **kwargs)


def compare_tvpvar(models: Dict[str, TimeVaryingVAR]) -> pd.DataFrame:
    """
    Comparison table of fitted :class:`TimeVaryingVAR` models: log marginal
    likelihood (higher = better) and DIC (lower = better), with numerical
    standard errors, for whichever of the two each model has computed. Models
    must share the estimation sample (a common ``presample``).
    """
    samples = {(m.yy_dates[0], m.yy_dates[-1]) for m in models.values()}
    if len(samples) > 1:
        raise ValueError("Models are estimated on different samples; set a common `presample`.")
    rows = []
    for name, m in models.items():
        row = {"model": name}
        if m.ml:
            row.update(logml=m.ml["logml"], logml_nse=m.ml["nse"])
        if m.dic_result:
            row.update(dic=m.dic_result["dic"], dic_nse=m.dic_result["nse"],
                       pD=m.dic_result["pD"])
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("model")
    if "logml" in tab:
        tab["rank_logml"] = tab["logml"].rank(ascending=False).astype("Int64")
    if "dic" in tab:
        tab["rank_dic"] = tab["dic"].rank(ascending=True).astype("Int64")
    return tab


__all__ = ["TimeVaryingVAR", "compare_tvpvar", "MODELS", "TVP_PRIOR_DEFAULTS"]
