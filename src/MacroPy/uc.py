"""
Unobserved-components trend-cycle decomposition for MacroPy.

One model family covers the whole ladder, from the Hodrick-Prescott filter to
a Bayesian UC model with pandemic-specific variance scales::

    y_t   = tau_t + c_t                                  (observation)
    tau_t = tau_{t-1} + g_{t-1} + eta_t                  (trend level)
    g_t   = rho_g g_{t-1} + (1 - rho_g) gbar + zeta_t    (trend growth / drift)
    c_t   = phi_1 c_{t-1} + phi_2 c_{t-2} + eps_t        (AR(2) cycle)

with independent shocks whose variances are ``q_{i,t} = lambda_{i,t} sigma_i^2``,
where ``lambda_{i,t}`` is a deterministic Lenza-Primiceri (2022) COVID scale
path (identically one when the treatment is off). Special cases:

* **HP filter**: ``rho_g = 1``, no level noise, white-noise cycle and a fixed
  variance ratio ``sigma_c^2 / sigma_zeta^2 = lambda`` reproduce the HP trend
  (Harvey & Jaeger, 1993; Grant & Chan, 2017). :func:`hp_filter` also computes
  it directly from the penalized-regression formula.
* **Clark (1987)**: ``rho_g = 1`` (random-walk drift) with an AR(2) cycle,
  estimated by maximum likelihood (:meth:`UCTrendCycle.fit_mle`).
* **Bayesian UC**: the full model with an AR(1) drift and calibrated priors,
  estimated by Gibbs sampling (:meth:`UCTrendCycle.sample_posterior`).
* **COVID scales**: per-component variance scale paths sampled by griddy
  Gibbs, so the pandemic loads on variances instead of distorting the trend.

Estimation uses the precision-based simulation smoother of Chan / McCausland
(as in Grant & Chan, 2017): the joint conditional of ``x = [tau; g]`` is
Gaussian with a sparse-banded precision, so states are drawn by one Cholesky
solve, and the same (K, b) object yields the exact integrated likelihood. A
textbook Kalman filter provides an independent cross-check and the per-period
predictive contributions used by WAIC.

This module is a Python port of the author's MATLAB ``trend_cycle_toolbox``
(itself building on Grant & Chan, 2017, JEDC, and Chan, 2018), keeping the
same conventions: pre-sample cycle zero (``eps_1 = c_1``), fixed initial-state
priors for ``tau_1`` and ``g_1``, and prior-calibration rules that map data
moments into the trend-shock scales.

References
----------
Clark, P.K. (1987). The cyclical component of U.S. economic activity. QJE.
Grant, A.L. & Chan, J.C.C. (2017). Reconciling output gaps. JEDC 75, 114-121.
Harvey, A.C. & Jaeger, A. (1993). Detrending, stylized facts and the business
cycle. Journal of Applied Econometrics 8, 231-247.
Lenza, M. & Primiceri, G.E. (2022). How to estimate a VAR after March 2020.
Journal of Applied Econometrics 37, 688-699.
Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and
WAIC. JMLR 11, 3571-3594.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from IPython.display import display
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
from scipy.stats import truncnorm
from tqdm import tqdm

from .summary import generate_uc_summary


# ===========================================================================
# Rung 1: the Hodrick-Prescott filter (direct penalized-regression solve)
# ===========================================================================


def hp_filter(y: Union[pd.Series, np.ndarray], lam: float = 1600.0) -> pd.DataFrame:
    """
    Two-sided Hodrick-Prescott filter.

    Solves ``min_tau sum (y_t - tau_t)^2 + lam * sum (Delta^2 tau_t)^2``
    exactly: ``tau = (I + lam D'D)^{-1} y`` with ``D`` the second-difference
    operator. ``lam = 1600`` is the standard quarterly value.

    Equivalent to the smoothed trend of a local-linear-trend state-space model
    with variance ratio ``sigma_c^2 / sigma_zeta^2 = lam`` and diffuse initial
    conditions (Harvey & Jaeger, 1993) — see ``UCTrendCycle.hp`` for that
    representation.

    Returns
    -------
    pd.DataFrame with columns ``trend`` and ``cycle`` (indexed like ``y`` when
    it is a Series).
    """
    arr = np.asarray(y, dtype=float).ravel()
    T = arr.shape[0]
    if T < 4:
        raise ValueError("hp_filter needs at least 4 observations.")
    D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(T - 2, T))
    A = identity(T, format="csc") + lam * (D.T @ D)
    trend = spsolve(A.tocsc(), arr)
    index = y.index if isinstance(y, pd.Series) else pd.RangeIndex(T)
    return pd.DataFrame({"trend": trend, "cycle": arr - trend}, index=index)


# ===========================================================================
# Model structure: difference operators, precision assembly, likelihoods
# ===========================================================================


def _lag_matrix(T: int, k: int) -> np.ndarray:
    """Dense lag operator: (L_k x)_t = x_{t-k} (zeros in the first k rows)."""
    L = np.zeros((T, T))
    idx = np.arange(k, T)
    L[idx, idx - k] = 1.0
    return L


def _build_ops(params: dict, T: int) -> dict:
    """
    Linear structure of the UC model (port of build_state_space_matrices.m).

    Returns the difference operators and variance diagonals of the three
    independent innovation blocks:
        eps  = Hphi @ (y - tau)          ~ N(0, diag(Sc))
        eta  = H1 @ tau - L1 @ g - m1    ~ N(0, diag(Se))
        zeta = Hrho @ g - m_g            ~ N(0, diag(Sz))
    Index 1 of Se/Sz carries the initial-state prior variances. Pre-sample
    cycle is zero (eps_1 = c_1), as in Grant & Chan (2017).
    """
    phi1, phi2 = params["phi"]
    rho_g = params["rho_g"]
    L1 = _lag_matrix(T, 1)
    L2 = _lag_matrix(T, 2)
    I = np.eye(T)

    alpha = (1.0 - rho_g) * params["gbar"]
    m1 = np.zeros(T)
    m1[0] = params["tau0_mean"]
    m_g = np.full(T, alpha)
    m_g[0] = params["g0_mean"]

    ops = {
        "Hphi": I - phi1 * L1 - phi2 * L2,
        "H1": I - L1,
        "L1": L1,
        "Hrho": I - rho_g * L1,
        "m1": m1,
        "m_g": m_g,
        "Sc": np.asarray(params["q_c"], dtype=float).copy(),
        "Se": np.concatenate([[params["tau0_var"]], params["q_eta"][1:]]),
        "Sz": np.concatenate([[params["g0_var"]], params["q_zeta"][1:]]),
        "alpha": alpha,
    }
    return ops


def _assemble_precision(y: np.ndarray, params: dict):
    """
    Joint Gaussian precision of x = [tau; g] (port of assemble_precision.m).

    Returns (K, b, c0, logdet_S): the exponent is x'Kx - 2b'x + c0 and
    logdet_S collects the innovation log-variances, so both the simulation
    smoother and the integrated likelihood refer to the same object.
    """
    T = y.shape[0]
    ops = _build_ops(params, T)
    iSc = 1.0 / ops["Sc"]
    iSe = 1.0 / ops["Se"]
    iSz = 1.0 / ops["Sz"]

    Hphi, H1, L1, Hrho = ops["Hphi"], ops["H1"], ops["L1"], ops["Hrho"]
    m1, m_g = ops["m1"], ops["m_g"]

    HphiSc = Hphi.T * iSc          # Hphi' @ diag(iSc)
    H1Se = H1.T * iSe
    L1Se = L1.T * iSe
    HrhoSz = Hrho.T * iSz

    Hphi_y = Hphi @ y
    K_tt = HphiSc @ Hphi + H1Se @ H1
    K_tg = -(H1Se @ L1)
    K_gg = L1Se @ L1 + HrhoSz @ Hrho
    b_t = HphiSc @ Hphi_y + H1Se @ m1
    b_g = -(L1Se @ m1) + HrhoSz @ m_g

    K = np.block([[K_tt, K_tg], [K_tg.T, K_gg]])
    K = 0.5 * (K + K.T)
    b = np.concatenate([b_t, b_g])
    c0 = Hphi_y @ (iSc * Hphi_y) + m1 @ (iSe * m1) + m_g @ (iSz * m_g)
    logdet_S = float(np.sum(np.log(ops["Sc"])) + np.sum(np.log(ops["Se"]))
                     + np.sum(np.log(ops["Sz"])))
    return K, b, c0, logdet_S


def _banded_system(y: np.ndarray, params: dict):
    """
    Banded assembly of the joint precision of the INTERLEAVED state vector
    x = (tau_1, g_1, tau_2, g_2, ...), algebraically identical to
    :func:`_assemble_precision` but O(T): the three innovation blocks are
    accumulated stencil by stencil into upper-banded storage (bandwidth 4),
    exactly the sparse-Cholesky strategy of the MATLAB toolbox. Returns
    ``(ab, b, c0, logdet_S)`` with ``ab`` in the scipy upper-banded layout
    ``ab[4 + i - j, j] = K[i, j]``.
    """
    T = y.shape[0]
    n = 2 * T
    phi1, phi2 = params["phi"]
    rho = params["rho_g"]
    alpha = (1.0 - rho) * params["gbar"]
    q_c = np.asarray(params["q_c"], dtype=float)
    q_eta = np.asarray(params["q_eta"], dtype=float)
    q_zeta = np.asarray(params["q_zeta"], dtype=float)

    ab = np.zeros((5, n))
    b = np.zeros(n)
    c0 = 0.0

    # ---- cycle block: eps_t = hy_t - (tau_t - phi1 tau_{t-1} - phi2 tau_{t-2})
    hy = np.empty(T)
    hy[0] = y[0]
    if T > 1:
        hy[1] = y[1] - phi1 * y[0]
    if T > 2:
        hy[2:] = y[2:] - phi1 * y[1:-1] - phi2 * y[:-2]
    w = 1.0 / q_c
    c0 += float(hy @ (w * hy))
    it = 2 * np.arange(2, T)               # tau_t columns, t >= 2
    wt = w[2:]
    ab[4, it] += wt                        # tau_t, tau_t
    ab[4, it - 2] += phi1 * phi1 * wt      # tau_{t-1}
    ab[4, it - 4] += phi2 * phi2 * wt      # tau_{t-2}
    ab[2, it] += -phi1 * wt                # (tau_{t-1}, tau_t)
    ab[0, it] += -phi2 * wt                # (tau_{t-2}, tau_t)
    ab[2, it - 2] += phi1 * phi2 * wt      # (tau_{t-2}, tau_{t-1})
    np.add.at(b, it, hy[2:] * wt)
    np.add.at(b, it - 2, -phi1 * hy[2:] * wt)
    np.add.at(b, it - 4, -phi2 * hy[2:] * wt)
    ab[4, 0] += w[0]
    b[0] += hy[0] * w[0]
    if T > 1:
        ab[4, 2] += w[1]
        ab[4, 0] += phi1 * phi1 * w[1]
        ab[2, 2] += -phi1 * w[1]
        b[2] += hy[1] * w[1]
        b[0] += -phi1 * hy[1] * w[1]

    # ---- trend-level block: eta_1 = tau_1 - tau0_mean; then
    #      eta_t = tau_t - tau_{t-1} - g_{t-1}
    w0 = 1.0 / params["tau0_var"]
    ab[4, 0] += w0
    b[0] += params["tau0_mean"] * w0
    c0 += params["tau0_mean"] ** 2 * w0
    we = 1.0 / q_eta[1:]
    it = 2 * np.arange(1, T)               # tau_t columns, t >= 1
    ab[4, it] += we                        # tau_t
    ab[4, it - 2] += we                    # tau_{t-1}
    ab[4, it - 1] += we                    # g_{t-1}
    ab[2, it] += -we                       # (tau_{t-1}, tau_t)
    ab[3, it] += -we                       # (g_{t-1},  tau_t)
    ab[3, it - 1] += we                    # (tau_{t-1}, g_{t-1})

    # ---- trend-growth block: zeta_1 = g_1 - g0_mean; then
    #      zeta_t = g_t - rho g_{t-1} - alpha
    w0 = 1.0 / params["g0_var"]
    ab[4, 1] += w0
    b[1] += params["g0_mean"] * w0
    c0 += params["g0_mean"] ** 2 * w0
    wz = 1.0 / q_zeta[1:]
    ig = 2 * np.arange(1, T) + 1           # g_t columns, t >= 1
    ab[4, ig] += wz
    ab[4, ig - 2] += rho * rho * wz
    ab[2, ig] += -rho * wz                 # (g_{t-1}, g_t)
    np.add.at(b, ig, alpha * wz)
    np.add.at(b, ig - 2, -rho * alpha * wz)
    c0 += float(alpha * alpha * wz.sum())

    logdet_S = float(np.log(q_c).sum() + np.log(params["tau0_var"])
                     + np.log(q_eta[1:]).sum() + np.log(params["g0_var"])
                     + np.log(q_zeta[1:]).sum())
    return ab, b, c0, logdet_S


def _draw_states(y: np.ndarray, params: dict, rng: np.random.Generator) -> dict:
    """
    Precision-based simulation smoother (port of sample_states.m): draw
    (tau, g) jointly from their full conditional via one banded Cholesky,
    return implied cycle and structural innovations.
    """
    from scipy.linalg import cholesky_banded, cho_solve_banded, solve_banded
    T = y.shape[0]
    ab, b, _, _ = _banded_system(y, params)
    cb = cholesky_banded(ab, lower=False)          # K = U'U, U upper banded
    xhat = cho_solve_banded((cb, False), b)
    x = xhat + solve_banded((0, 4), cb, rng.standard_normal(2 * T))

    tau = x[0::2]
    g = x[1::2]
    c = y - tau

    phi1, phi2 = params["phi"]
    eps = np.empty(T)
    eps[0] = c[0]
    if T > 1:
        eps[1] = c[1] - phi1 * c[0]
    if T > 2:
        eps[2:] = c[2:] - phi1 * c[1:-1] - phi2 * c[:-2]
    eta = np.empty(T)
    eta[0] = tau[0] - params["tau0_mean"]
    eta[1:] = tau[1:] - tau[:-1] - g[:-1]
    zeta = np.empty(T)
    alpha = (1.0 - params["rho_g"]) * params["gbar"]
    zeta[0] = g[0] - params["g0_mean"]
    zeta[1:] = g[1:] - params["rho_g"] * g[:-1] - alpha
    return {"trend": tau, "drift": g, "cycle": c,
            "eps": eps, "eta": eta, "zeta": zeta}


def integrated_loglik(y: np.ndarray, params: dict) -> float:
    """
    Exact integrated log-likelihood log p(y | params) with the states
    marginalised analytically (port of compute_loglik_kalman.m, which uses
    the sparse-precision identity rather than a filter):

        log p(y) = -T/2 log(2 pi) - 1/2 (logdet_S + log|K|)
                   + 1/2 b'K^{-1}b - 1/2 c0.
    """
    from scipy.linalg import cholesky_banded, cho_solve_banded
    y = np.asarray(y, dtype=float).ravel()
    T = y.shape[0]
    ab, b, _, logdet_S = _banded_system(y, params)
    cb = cholesky_banded(ab, lower=False)          # K = U'U
    logdetK = 2.0 * float(np.log(cb[4]).sum())
    # c0 - b'K^{-1}b equals the weighted residual sum of squares at the
    # posterior mode: evaluating it that way (all terms positive) avoids the
    # catastrophic cancellation of the two large quadratic forms
    xhat = cho_solve_banded((cb, False), b)
    tau = xhat[0::2]
    g = xhat[1::2]
    phi1, phi2 = params["phi"]
    c = y - tau
    eps = np.empty(T)
    eps[0] = c[0]
    if T > 1:
        eps[1] = c[1] - phi1 * c[0]
    if T > 2:
        eps[2:] = c[2:] - phi1 * c[1:-1] - phi2 * c[:-2]
    alpha = (1.0 - params["rho_g"]) * params["gbar"]
    qmin = float(np.sum(eps ** 2 / params["q_c"])
                 + (tau[0] - params["tau0_mean"]) ** 2 / params["tau0_var"]
                 + np.sum((tau[1:] - tau[:-1] - g[:-1]) ** 2
                          / params["q_eta"][1:])
                 + (g[0] - params["g0_mean"]) ** 2 / params["g0_var"]
                 + np.sum((g[1:] - params["rho_g"] * g[:-1] - alpha) ** 2
                          / params["q_zeta"][1:]))
    return float(-0.5 * T * np.log(2 * np.pi) - 0.5 * (logdet_S + logdetK)
                 - 0.5 * qmin)


def kalman_loglik(y: np.ndarray, params: dict):
    """
    Textbook Kalman filter for the 4-state form s_t = [tau, g, c, c_lag]
    (port of kalman_filter_uc.m). Returns (loglik, loglik_t).

    Two roles: the per-period contributions log p(y_t | y_{1:t-1}) feed WAIC,
    and the total is an independent cross-check of :func:`integrated_loglik`
    (one uses a sparse-precision identity, the other a sequential filter, so
    agreement at ~1e-8 is strong evidence both are correct).
    """
    import math
    y = np.asarray(y, dtype=float).ravel()
    T = y.shape[0]
    phi1, phi2 = params["phi"]
    rho_g = params["rho_g"]
    alpha = (1.0 - rho_g) * params["gbar"]
    Tm = np.array([[1.0, 1.0, 0.0, 0.0],
                   [0.0, rho_g, 0.0, 0.0],
                   [0.0, 0.0, phi1, phi2],
                   [0.0, 0.0, 1.0, 0.0]])
    cvec = np.array([0.0, alpha, 0.0, 0.0])
    q_eta, q_zeta, q_c = params["q_eta"], params["q_zeta"], params["q_c"]

    a = np.array([params["tau0_mean"], params["g0_mean"], 0.0, 0.0])
    P = np.diag([params["tau0_var"], params["g0_var"], q_c[0], 0.0])

    # Z = [1, 0, 1, 0]: the Z products reduce to index arithmetic
    loglik_t = np.zeros(T)
    log2pi = math.log(2 * math.pi)
    for t in range(T):
        v = y[t] - a[0] - a[2]
        F = P[0, 0] + 2.0 * P[0, 2] + P[2, 2]
        loglik_t[t] = -0.5 * (log2pi + math.log(F) + v * v / F)
        PZ = P[:, 0] + P[:, 2]
        a = a + PZ * (v / F)
        P = P - np.outer(PZ, PZ) / F
        if t < T - 1:
            a = Tm @ a + cvec
            P = Tm @ P @ Tm.T
            P[0, 0] += q_eta[t + 1]
            P[1, 1] += q_zeta[t + 1]
            P[2, 2] += q_c[t + 1]
            P = 0.5 * (P + P.T)
    return float(loglik_t.sum()), loglik_t


# ===========================================================================
# COVID variance-scale machinery (Lenza-Primiceri, discrete-jump variant)
# ===========================================================================


def covid_lambda_path(T: int, t0: Optional[int], lambda1: float, lambda2: float,
                      lambda3: float, rho_decay: float) -> np.ndarray:
    """
    Deterministic variance-scale path (port of covid_lambda_path.m)::

        lambda_t = 1                                        t <  t0
        lambda_t = lambda1                                  t =  t0
        lambda_t = lambda2                                  t =  t0 + 1
        lambda_t = 1 + (lambda3 - 1) rho_decay^(t-t0-2)     t >= t0 + 2

    Three free scales cover the extreme pandemic quarters; the geometric tail
    reverts to one. ``lambda`` multiplies the *variance*. ``t0`` is 0-based.
    """
    lam = np.ones(T)
    if t0 is None or t0 < 0 or t0 >= T:
        return lam
    lam[t0] = lambda1
    if t0 + 1 < T:
        lam[t0 + 1] = lambda2
    if t0 + 2 < T:
        k = np.arange(T - t0 - 2)
        lam[t0 + 2:] = 1.0 + (lambda3 - 1.0) * rho_decay ** k
    return np.maximum(lam, np.finfo(float).eps)


def _grid_draw(grid: np.ndarray, logp: np.ndarray, rng: np.random.Generator) -> float:
    p = np.exp(logp - logp.max())
    p = p / p.sum()
    return float(rng.choice(grid, p=p))


def _sample_covid_block(comp: dict, resid: np.ndarray, sigma2: float,
                        pri_cov: dict, t0: int, rng: np.random.Generator) -> dict:
    """
    Griddy-Gibbs update of one component's COVID scales (port of
    sample_covid_scale_factors.m): each of (lambda1, lambda2, lambda3,
    rho_decay) is drawn from its 1-D full conditional on a grid, with an
    optional log-normal shrinkage of the lambdas toward one. Only t >= t0
    enters the conditional. ``nfree = 2`` ties lambda3 = 1 + (lambda2-1) rho.
    """
    T = resid.shape[0]
    s = resid[t0:] ** 2 / sigma2           # only t >= t0 enters the conditional
    n_s = s.shape[0]
    tau = pri_cov["tau"]
    nfree = pri_cov.get("nfree", 3)

    lam_grid = np.exp(np.linspace(np.log(pri_cov["lambda_min"]),
                                  np.log(pri_cov["lambda_max"]), pri_cov["ngrid"]))
    rho_grid = np.linspace(pri_cov["rho_min"], pri_cov["rho_max"], pri_cov["ngrid"])
    logprior = (np.zeros_like(lam_grid) if np.isinf(tau)
                else -0.5 * (np.log(lam_grid) / tau) ** 2)
    k = np.arange(max(n_s - 2, 0))         # tail exponents (t >= t0 + 2)
    s_tail = s[2:]

    l1, l2, l3, rho = comp["lambda1"], comp["lambda2"], comp["lambda3"], comp["rho_decay"]

    # lambda1 enters only at t0: closed-form conditional on the grid
    logp = -0.5 * (np.log(lam_grid) + s[0] / lam_grid) + logprior
    l1 = _grid_draw(lam_grid, logp, rng)

    # lambda2 enters at t0+1 (and, when nfree = 2, the tied tail as well)
    logp = logprior.copy()
    if n_s >= 2:
        logp = logp - 0.5 * (np.log(lam_grid) + s[1] / lam_grid)
    if nfree < 3 and n_s >= 3:
        w = 1.0 + (lam_grid[:, None] - 1.0) * rho ** (k + 1)
        w = np.maximum(w, np.finfo(float).eps)
        logp = logp - 0.5 * np.sum(np.log(w) + s_tail / w, axis=1)
    l2 = _grid_draw(lam_grid, logp, rng)

    if nfree >= 3:
        logp = logprior.copy()
        if n_s >= 3:
            w = 1.0 + (lam_grid[:, None] - 1.0) * rho ** k
            w = np.maximum(w, np.finfo(float).eps)
            logp = logp - 0.5 * np.sum(np.log(w) + s_tail / w, axis=1)
        l3 = _grid_draw(lam_grid, logp, rng)
    else:
        l3 = 1.0 + (l2 - 1.0) * rho

    # rho enters only through the tail (flat prior on the grid)
    logp = np.zeros_like(rho_grid)
    if n_s >= 3:
        if nfree >= 3:
            w = 1.0 + (l3 - 1.0) * rho_grid[:, None] ** k
        else:
            w = 1.0 + (l2 - 1.0) * rho_grid[:, None] ** (k + 1)
        w = np.maximum(w, np.finfo(float).eps)
        logp = -0.5 * np.sum(np.log(w) + s_tail / w, axis=1)
    rho = _grid_draw(rho_grid, logp, rng)
    if nfree < 3:
        l3 = 1.0 + (l2 - 1.0) * rho

    return {"lambda1": l1, "lambda2": l2, "lambda3": l3, "rho_decay": rho,
            "lambda": covid_lambda_path(T, t0, l1, l2, l3, rho)}


# ===========================================================================
# Priors: defaults and the data-relative calibration rules
# ===========================================================================


def _movmean(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average with shrinking edge windows."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    half = w // 2
    out = np.empty(n)
    for i in range(n):
        out[i] = x[max(0, i - half): min(n, i + half) + 1].mean()
    return out


def _ig_from_mean(m: float, nu: float) -> Tuple[float, float]:
    """Inverse-gamma (shape nu, scale S) with prior mean m = S / (nu - 1)."""
    return nu, m * (nu - 1.0)


def _default_rules() -> dict:
    """Frozen constants of the prior-calibration rules (default_prior_rules.m)."""
    return {
        "kappa_zeta": 0.35,        # sd(zeta) = kappa * sd(13q MA of dy)
        "eta_sd_ratio": 0.10,      # sd(eta)  = ratio * sd(zeta)
        "lam_mult": 50.0,          # covid grid = mult * standardized outlier
        "lam_floor": 500.0,        # ... floored here
        "lam_eta_ratio": 0.40,     # level grid, share of the cycle grid
        "lam_zeta_ratio": 0.02,    # growth grid, likewise
        "phi_max_modulus": 0.95,   # dominant AR(2) root cap (calibrated value)
    }


def make_priors(y: np.ndarray, covid_t0: Optional[int] = None,
                calibrate: bool = True, overrides: Optional[dict] = None) -> dict:
    """
    Prior hyperparameters for the UC family (port of make_priors.m +
    calibrate_priors.m).

    Defaults follow the modelling goals: a *small* trend-level shock, a
    *moderate* trend-growth shock, a flexible cycle, an AR(2) prior centred on
    persistent complex roots, and a drift persistence centred high. With
    ``calibrate=True`` (recommended; see the tutorial's exhibit) the trend
    shock scales are replaced by data-relative rules computed on the
    pre-COVID sample:

        sd(zeta) = 0.35 * sd(13-quarter centred MA of dy)
        sd(eta)  = 0.10 * sd(zeta)

    which keeps the trend from chasing the data — the classic UC pathology.
    COVID grid bounds scale with the series' own standardized 2020 outlier.
    ``overrides`` merges recursively on top (any field, including nested
    covid blocks).
    """
    y = np.asarray(y, dtype=float).ravel()
    dy = np.diff(y)
    g_scale = float(np.median(dy))
    cyc_scale = float(max(np.var(y - _movmean(y, 13), ddof=1), 1e-8))

    pri: dict = {
        "phi_mean": np.array([1.5, -0.6]),
        "phi_prec": 16.0 * np.eye(2),
        "phi_margin": 0.01,
        "phi_max_modulus": 0.85,
        "gbar_mean": g_scale, "gbar_var": 0.50 ** 2,
        "rho_g_mean": 0.90, "rho_g_var": 0.20 ** 2,
        "rho_g_min": -0.95, "rho_g_max": 0.995,
        "tau0_mean": float(y[0]), "tau0_var": 3.0 ** 2,
        "g0_mean": g_scale, "g0_var": 0.50 ** 2,
    }
    pri["sig2_c_nu"], pri["sig2_c_S"] = _ig_from_mean(max(0.5, cyc_scale), 3)
    pri["sig2_eta_nu"], pri["sig2_eta_S"] = _ig_from_mean(0.05 ** 2, 12)
    pri["sig2_zeta_nu"], pri["sig2_zeta_S"] = _ig_from_mean(0.50 ** 2, 8)

    def cov_block(lmax, ngrid, tau):
        return {"lambda_min": 1.0, "lambda_max": lmax, "ngrid": ngrid,
                "rho_min": 0.02, "rho_max": 0.98, "tau": tau, "nfree": 3}

    pri["cov_c"] = cov_block(5000.0, 250, np.inf)
    pri["cov_eta"] = cov_block(2000.0, 250, 2.5)
    pri["cov_zeta"] = cov_block(100.0, 200, 1.0)

    if calibrate:
        rules = _default_rules()
        t_pre = len(y) if (covid_t0 is None or covid_t0 < 10) else covid_t0
        ypre = y[:t_pre]
        dypre = np.diff(ypre)
        sd_glf = float(np.std(_movmean(dypre, 13), ddof=1))
        cyc_pre = float(max(np.var(ypre - _movmean(ypre, 13), ddof=1), 1e-8))

        sd_zeta = rules["kappa_zeta"] * sd_glf
        sd_eta = rules["eta_sd_ratio"] * sd_zeta
        pri["sig2_zeta_nu"], pri["sig2_zeta_S"] = _ig_from_mean(sd_zeta ** 2, 8)
        pri["sig2_eta_nu"], pri["sig2_eta_S"] = _ig_from_mean(sd_eta ** 2, 12)
        pri["phi_max_modulus"] = rules["phi_max_modulus"]
        pri["_calibration"] = {"sd_zeta": sd_zeta, "sd_eta": sd_eta,
                               "sd_glf": sd_glf, "cyc_pre": cyc_pre}

        if covid_t0 is not None and 1 <= covid_t0 < len(y):
            # r2020: the largest of the FIRST TWO pandemic growth surprises,
            # in units of the pre-COVID cycle variance (calibrate_priors.m)
            g_med = float(np.median(dypre))
            j0, j1 = covid_t0 - 1, min(covid_t0 + 1, len(dy))
            r2020 = float(np.max((dy[j0:j1] - g_med) ** 2) / cyc_pre)

            def ceil_1sig(x):
                if not np.isfinite(x) or x <= 0:
                    return x
                mag = 10.0 ** np.floor(np.log10(x))
                return float(np.ceil(x / mag) * mag)

            lam_c = max(rules["lam_floor"], ceil_1sig(rules["lam_mult"] * r2020))
            pri["cov_c"]["lambda_max"] = lam_c
            pri["cov_eta"]["lambda_max"] = float(round(rules["lam_eta_ratio"] * lam_c))
            pri["cov_zeta"]["lambda_max"] = float(round(rules["lam_zeta_ratio"] * lam_c))
            pri["_calibration"]["r2020"] = r2020
            pri["_calibration"]["lambda_max"] = (
                pri["cov_c"]["lambda_max"], pri["cov_eta"]["lambda_max"],
                pri["cov_zeta"]["lambda_max"])

    if overrides:
        def merge(base, ov):
            for k, v in ov.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    merge(base[k], v)
                else:
                    base[k] = v
        merge(pri, dict(overrides))
    return pri


def _stationary_ar2(phi: np.ndarray, margin: float, max_modulus: float) -> bool:
    """Stationarity triangle with margin plus a cap on the dominant root."""
    p1, p2 = float(phi[0]), float(phi[1])
    lim = 1.0 - margin
    if not (p1 + p2 < lim and p2 - p1 < lim and abs(p2) < lim):
        return False
    roots = np.roots([1.0, -p1, -p2])
    return bool(np.max(np.abs(roots)) <= max_modulus)


def _solve_constrained_u(A: np.ndarray, b: np.ndarray,
                         rng: Optional[np.random.Generator]) -> np.ndarray:
    """
    Draw standardized future shocks u ~ N(0, I) subject to A u = b, as in
    Waggoner & Zha (1999): the conditional is N(A+ b, I - A+ A), sampled via
    the SVD as u = A+ b + V2 z with V2 an orthonormal null-space basis
    (rank-tolerant). With ``rng=None`` the minimum-norm point A+ b is
    returned (no shock uncertainty).
    """
    m = A.shape[1]
    if A.shape[0] == 0:
        return (rng.standard_normal(m) if rng is not None
                else np.zeros(m))
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    r = int(np.sum(s > tol))
    u = Vt[:r].T @ ((U[:, :r].T @ b) / s[:r])
    if rng is not None and r < m:
        u = u + Vt[r:].T @ rng.standard_normal(m - r)
    return u


# ===========================================================================
# The engine
# ===========================================================================


class UCTrendCycle:
    """
    Bayesian unobserved-components trend-cycle decomposition (one engine for
    the HP filter, Clark's UC by MLE, the Bayesian UC and its COVID-scaled
    version — see the module docstring for the family and references).

    Parameters
    ----------
    y : pd.Series or single-column pd.DataFrame
        The series to decompose, already transformed (e.g. ``100*log(GDP)``),
        with a datetime index.
    drift : {"ar1", "rw"}, default "ar1"
        ``"ar1"`` estimates the drift persistence ``rho_g`` (mean-reverting
        potential growth; ``rho_g -> 1`` nests the alternative); ``"rw"``
        fixes ``rho_g = 1`` (Clark, 1987).
    level_noise : bool, default True
        Include the trend-level shock ``eta``. ``False`` (smooth I(2) trend,
        Grant-Chan style) fixes its variance at a negligible value.
    fix_phi : tuple, optional
        Fix the AR(2) cycle coefficients (e.g. ``(0, 0)`` for the white-noise
        cycle of the HP representation) instead of estimating them.
    covid_t0 : str, pd.Timestamp or None, default None
        First pandemic observation (``"2020Q1"`` is the convention of the
        reference MATLAB toolbox: the two free scales then cover 2020Q1-Q2
        and the geometric tail starts at the rebound). Activates the
        Lenza-Primiceri variance-scale paths, sampled by griddy Gibbs.
    covid_components : tuple, default ("cycle", "trend_level", "trend_growth")
        Which shocks receive a COVID scale path (each with its own
        parameters, so the model stays agnostic about where the pandemic
        loaded; the priors regularise the trend channels).
    covid_nfree : {3, 2}, default 3
        Free scale factors before the geometric tail (3 = Lenza-Primiceri).
    calibrate : bool, default True
        Apply the data-relative prior rules (kappa rules) on the pre-COVID
        sample. Strongly recommended: the generic defaults let the trend
        chase the data on calm series.
    prior_params : dict, optional
        Recursive overrides of any prior hyperparameter (see `make_priors`).
    post_draws : int, default 10000
        Total Gibbs draws including burn-in (``BayesianVAR`` convention).
    burnin : float, default 0.5
        Fraction of draws discarded.
    seed : int, optional
        Random seed; per-stage generators make results exactly reproducible.

    Examples
    --------
    >>> uc = UCTrendCycle(100*np.log(gdp), covid_t0="2020Q1", seed=1)
    >>> uc.model_summary()
    >>> uc.sample_posterior()
    >>> uc.decomposition_frame().head()
    >>> uc.waic()
    """

    _RNG_STAGES = {"posterior": 0, "forecast": 1, "conditional": 2}

    def __init__(
        self,
        y: Union[pd.Series, pd.DataFrame],
        drift: str = "ar1",
        level_noise: bool = True,
        fix_phi: Optional[Tuple[float, float]] = None,
        covid_t0: Optional[Union[str, pd.Timestamp]] = None,
        covid_components: Tuple[str, ...] = ("cycle", "trend_level", "trend_growth"),
        covid_nfree: int = 3,
        calibrate: bool = True,
        prior_params: Optional[dict] = None,
        post_draws: int = 10000,
        burnin: float = 0.5,
        seed: Optional[int] = None,
    ):
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("`y` must be a Series or a single-column DataFrame.")
            y = y.iloc[:, 0]
        if not isinstance(y, pd.Series):
            raise ValueError("`y` must be a pandas Series with a datetime index.")
        if y.isna().any():
            raise ValueError("`y` cannot contain missing values.")
        self.name = str(y.name) if y.name is not None else "y"
        self.dates = y.index
        self.y = y.to_numpy(dtype=float)
        self.T = self.y.shape[0]

        if drift not in ("ar1", "rw"):
            raise ValueError("drift must be 'ar1' or 'rw'.")
        self.drift = drift
        self.level_noise = bool(level_noise)
        self.fix_phi = None if fix_phi is None else np.asarray(fix_phi, float)

        bad = [c for c in covid_components
               if c not in ("cycle", "trend_level", "trend_growth")]
        if bad:
            raise ValueError(f"Unknown covid_components {bad}.")
        self.covid_components = tuple(covid_components) if covid_t0 is not None else ()
        self.covid_t0 = self._parse_t0(covid_t0)
        self.covid_nfree = int(covid_nfree)

        self.calibrate = bool(calibrate)
        self.prior_params = dict(prior_params) if prior_params else {}
        self.priors = make_priors(self.y, covid_t0=self.covid_t0,
                                  calibrate=self.calibrate,
                                  overrides=self.prior_params)
        for comp in self.covid_components:
            key = {"cycle": "cov_c", "trend_level": "cov_eta",
                   "trend_growth": "cov_zeta"}[comp]
            self.priors[key]["nfree"] = self.covid_nfree

        self.post_draws = int(post_draws)
        self.burnin = int(burnin * post_draws)
        self.n_draws = self.post_draws - self.burnin
        self.seed = seed

        self._fixed_variances: Dict[str, float] = {}
        self.draws: Optional[dict] = None
        self.mle_: Optional[dict] = None

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def hp(cls, y, lam: float = 1600.0) -> "UCTrendCycle":
        """
        The HP filter as a state-space model: random-walk drift, no level
        noise, white-noise cycle, fixed variance ratio
        ``sigma_c^2 / sigma_zeta^2 = lam``, diffuse initial states. The
        smoothed trend of ``fit_mle()`` equals :func:`hp_filter` (up to the
        diffuse-limit approximation), which is the point of rung 1: a filter
        *is* a model.
        """
        m = cls(y, drift="rw", level_noise=False, fix_phi=(0.0, 0.0),
                calibrate=False,
                prior_params={"tau0_var": 1e12, "g0_var": 1e12})
        m._fixed_variances.update({"sig2_c": float(lam), "sig2_zeta": 1.0})
        m._hp_lambda = float(lam)
        return m

    @classmethod
    def clark(cls, y, **kwargs) -> "UCTrendCycle":
        """Clark (1987): random-walk drift, AR(2) cycle. Use `fit_mle()` for
        the classical rung or `sample_posterior()` for its Bayesian twin."""
        kwargs.setdefault("drift", "rw")
        return cls(y, **kwargs)

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _parse_t0(self, covid_t0) -> Optional[int]:
        if covid_t0 is None:
            return None
        target = pd.Period(str(covid_t0)) if not isinstance(covid_t0, pd.Timestamp) \
            else pd.Period(covid_t0, freq=pd.infer_freq(self.dates) or "Q")
        periods = pd.PeriodIndex(self.dates, freq=target.freq)
        pos = np.flatnonzero(periods == target)
        if pos.size == 0:
            raise ValueError(f"covid_t0={covid_t0!r} not found in the sample index.")
        return int(pos[0])

    def _rng(self, stage: str) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng([int(self.seed), self._RNG_STAGES[stage]])

    def _params_from_state(self, st: dict) -> dict:
        return {
            "phi": st["phi"], "rho_g": st["rho_g"], "gbar": st["gbar"],
            "q_c": st["lam_c"] * st["sig2_c"],
            "q_eta": st["lam_eta"] * st["sig2_eta"],
            "q_zeta": st["lam_zeta"] * st["sig2_zeta"],
            "tau0_mean": self.priors["tau0_mean"], "tau0_var": self.priors["tau0_var"],
            "g0_mean": self.priors["g0_mean"], "g0_var": self.priors["g0_var"],
        }

    def _variance_is_free(self, key: str) -> bool:
        if key in self._fixed_variances:
            return False
        return not (key == "sig2_eta" and not self.level_noise)

    def _apply_tied(self, st: dict) -> None:
        # level_noise=False suppresses eta with a variance tied to a negligible
        # share of the cycle variance; a hard zero (or an absolute constant far
        # from sig2_c's scale) makes the joint precision ill conditioned
        if not self.level_noise:
            st["sig2_eta"] = 1e-8 * st["sig2_c"]

    def _init_state(self) -> dict:
        pri = self.priors
        ones = np.ones(self.T)
        st = {
            "phi": (self.fix_phi.copy() if self.fix_phi is not None
                    else pri["phi_mean"].copy()),
            "rho_g": 1.0 if self.drift == "rw" else float(
                np.clip(pri["rho_g_mean"], pri["rho_g_min"], pri["rho_g_max"])),
            "gbar": 0.0 if self.drift == "rw" else pri["gbar_mean"],
            "sig2_c": self._fixed_variances.get(
                "sig2_c", pri["sig2_c_S"] / (pri["sig2_c_nu"] - 1)),
            "sig2_eta": self._fixed_variances.get(
                "sig2_eta", pri["sig2_eta_S"] / (pri["sig2_eta_nu"] - 1)),
            "sig2_zeta": self._fixed_variances.get(
                "sig2_zeta", pri["sig2_zeta_S"] / (pri["sig2_zeta_nu"] - 1)),
            "lam_c": ones.copy(), "lam_eta": ones.copy(), "lam_zeta": ones.copy(),
        }
        for comp in self.covid_components:
            st[f"cov_{comp}"] = {"lambda1": 1.0, "lambda2": 1.0, "lambda3": 1.0,
                                 "rho_decay": 0.5,
                                 "lambda": np.ones(self.T)}
        self._apply_tied(st)
        return st

    # ------------------------------------------------------------------
    # Rung 2: maximum likelihood
    # ------------------------------------------------------------------

    def fit_mle(self, verbose: bool = False, progress: bool = True,
                polish_budget: int = 200) -> dict:
        """
        Classical estimation: maximize the exact integrated likelihood over
        the free parameters, then smooth the states at the optimum.

        Free parameters depend on the specification: ``phi`` (unless fixed),
        the non-fixed shock variances (in logs), and for ``drift="ar1"`` also
        ``rho_g`` and ``gbar``. Smoothed states and their marginal standard
        errors come from the same precision object used by the Bayesian
        sampler. Results are stored on ``self.mle_`` and returned.

        ``progress`` shows a tqdm bar counting likelihood evaluations.
        ``polish_budget`` caps each derivative-free polish stage at
        ``polish_budget * n_params`` evaluations (the optimum is typically
        reached within a fraction of the default; raise it only if
        ``success`` comes back False on a pathological surface).
        """
        pri = self.priors

        # data-informed starting values from a rough HP pre-decomposition:
        # good basins matter more than the optimizer here (the likelihood has
        # the classic UC ridge between cycle and trend variances)
        hp0 = hp_filter(pd.Series(self.y), 1600.0)
        c0 = hp0["cycle"].to_numpy()
        X0 = np.column_stack([c0[1:-1], c0[:-2]])
        phi0, *_ = np.linalg.lstsq(X0, c0[2:], rcond=None)
        if not _stationary_ar2(phi0, 0.05, 0.97):
            phi0 = pri["phi_mean"].copy()
        sig2_c0 = float(np.var(c0[2:] - X0 @ phi0))

        names = []
        x0, lb, ub = [], [], []
        if self.fix_phi is None:
            names += ["phi1", "phi2"]
            x0 += [phi0[0], phi0[1]]
            lb += [-1.99, -0.99]
            ub += [1.99, 0.99]
        for key, start in (("sig2_c", sig2_c0),
                           ("sig2_eta", pri["sig2_eta_S"] / (pri["sig2_eta_nu"] - 1)),
                           ("sig2_zeta", pri["sig2_zeta_S"] / (pri["sig2_zeta_nu"] - 1))):
            if self._variance_is_free(key):
                names.append("log_" + key)
                x0.append(np.log(start))
                lb.append(-25.0)
                ub.append(10.0)
        if self.drift == "ar1":
            names += ["rho_g", "gbar"]
            x0 += [pri["rho_g_mean"], pri["gbar_mean"]]
            lb += [pri["rho_g_min"], pri["gbar_mean"] - 5.0]
            ub += [pri["rho_g_max"], pri["gbar_mean"] + 5.0]

        st0 = self._init_state()

        def unpack(x):
            st = dict(st0)
            i = 0
            if self.fix_phi is None:
                st["phi"] = np.array([x[i], x[i + 1]])
                i += 2
            for key in ("sig2_c", "sig2_eta", "sig2_zeta"):
                if self._variance_is_free(key):
                    st[key] = float(np.exp(x[i]))
                    i += 1
            if self.drift == "ar1":
                st["rho_g"] = float(x[i])
                st["gbar"] = float(x[i + 1])
            self._apply_tied(st)
            return st

        bar = tqdm(desc="Optimizing likelihood", unit=" eval",
                   disable=not (progress and names), leave=False)

        def negll(x):
            bar.update(1)
            st = unpack(x)
            if self.fix_phi is None and not _stationary_ar2(
                    st["phi"], 0.0, 1.0 - 1e-6):
                return 1e10
            try:
                return -integrated_loglik(self.y, self._params_from_state(st))
            except np.linalg.LinAlgError:
                return 1e10

        if names:
            bounds = list(zip(lb, ub))
            cap = max(200, int(polish_budget) * len(names))
            nm_opts = {"xatol": 1e-5, "fatol": 1e-8,
                       "maxiter": cap, "maxfev": cap}
            res = minimize(negll, np.array(x0), method="L-BFGS-B", bounds=bounds)
            # derivative-free polish: rescues line-search stalls when a
            # variance piles up at its bound (Stock-Watson pile-up)
            res2 = minimize(negll, res.x, method="Nelder-Mead", bounds=bounds,
                            options=nm_opts)
            best = res2 if res2.fun <= res.fun else res
            # one more short polish from the improved point; when it can no
            # longer move the objective, the point is a numerical optimum
            # regardless of the line-search flags (variances piled up at a
            # bound routinely trip those)
            res3 = minimize(negll, best.x, method="Nelder-Mead", bounds=bounds,
                            options=nm_opts)
            if res3.fun <= best.fun:
                best = res3
            st_hat = unpack(best.x)
            loglik = -float(best.fun)
            success = bool(res.success or res2.success or res3.success
                           or abs(res3.fun - res2.fun)
                           <= 1e-6 * (1.0 + abs(res2.fun)))
        else:  # fully fixed specification (e.g. the HP preset): smooth only
            st_hat = dict(st0)
            loglik = integrated_loglik(self.y, self._params_from_state(st_hat))
            success = True
        bar.close()

        params = self._params_from_state(st_hat)
        K, b, _, _ = _assemble_precision(self.y, params)
        xhat = np.linalg.solve(K, b)
        Kinv_diag = np.diag(np.linalg.inv(K))
        T = self.T
        est = {}
        if self.fix_phi is None:
            est["phi1"], est["phi2"] = float(st_hat["phi"][0]), float(st_hat["phi"][1])
        for key in ("sig2_c", "sig2_eta", "sig2_zeta"):
            if self._variance_is_free(key):
                est[key] = float(st_hat[key])
        if self.drift == "ar1":
            est["rho_g"] = float(st_hat["rho_g"])
            est["gbar"] = float(st_hat["gbar"])
        self.mle_ = {
            "params": est,
            "state": st_hat,
            "loglik": loglik,
            "success": success,
            "trend": xhat[:T], "drift": xhat[T:],
            "cycle": self.y - xhat[:T],
            "trend_se": np.sqrt(np.maximum(Kinv_diag[:T], 0.0)),
            "drift_se": np.sqrt(np.maximum(Kinv_diag[T:], 0.0)),
        }
        if verbose and names:
            shown = ", ".join(f"{n}={v:.4g}" for n, v in self.mle_["params"].items())
            print(f"MLE: loglik = {loglik:.2f}; {shown}")
        return self.mle_

    # ------------------------------------------------------------------
    # Rungs 3 and 6: Gibbs sampler
    # ------------------------------------------------------------------

    def sample_posterior(self, store_loglik: bool = True, thin: int = 1) -> dict:
        """
        Gibbs sampler for the UC model (port of uc_gibbs_core.m).

        Sweep: (1) states (tau, g) by the precision-based simulation smoother;
        (2) phi / rho_g / gbar by conjugate (truncated) normals and the shock
        variances by inverse-gamma, deflating residuals by the COVID scales;
        (3) the COVID scale parameters of each flagged component by griddy
        Gibbs. Per-period predictive log-densities (for WAIC/DIC) are stored
        at each saved draw when ``store_loglik``.

        ``thin`` stores every thin-th post burn-in sweep (the MATLAB toolbox
        convention). Thinning does not add statistical precision at a given
        number of total sweeps; it trades storage and post-processing cost
        for wall time, so the way to reduce Monte Carlo noise is to raise
        ``post_draws`` and use ``thin`` to keep the stored sample compact.

        Returns (and stores on ``self.draws``) a dict of posterior draws.
        """
        rng = self._rng("posterior")
        pri = self.priors
        T = self.T
        y = self.y
        st = self._init_state()
        thin = max(int(thin), 1)
        self.thin = thin
        keep = len(range(self.burnin, self.post_draws, thin))
        self.n_draws = keep

        out = {
            "trend": np.zeros((keep, T)), "drift": np.zeros((keep, T)),
            "cycle": np.zeros((keep, T)),
            "phi": np.zeros((keep, 2)), "rho_g": np.zeros(keep),
            "gbar": np.zeros(keep),
            "sig2_c": np.zeros(keep), "sig2_eta": np.zeros(keep),
            "sig2_zeta": np.zeros(keep),
            "loglik": np.zeros(keep),
            "loglik_t": np.zeros((keep, T)) if store_loglik else None,
        }
        for comp in self.covid_components:
            out[f"cov_{comp}"] = np.zeros((keep, 4))
            out[f"lambda_{comp}"] = np.zeros((keep, T))

        comp_map = {"cycle": ("eps", "lam_c", "sig2_c", "cov_c"),
                    "trend_level": ("eta", "lam_eta", "sig2_eta", "cov_eta"),
                    "trend_growth": ("zeta", "lam_zeta", "sig2_zeta", "cov_zeta")}

        s = 0
        for sweep in tqdm(range(self.post_draws), desc="Sampling Posterior"):
            params = self._params_from_state(st)

            # ---- 1) states -------------------------------------------------
            draw = _draw_states(y, params, rng)
            c, g = draw["cycle"], draw["drift"]

            # ---- 2a) AR(2) cycle coefficients ------------------------------
            if self.fix_phi is None:
                w = 1.0 / (st["lam_c"] * st["sig2_c"])
                Xc = np.zeros((T, 2))
                Xc[1, 0] = c[0]
                if T >= 3:
                    Xc[2:, 0] = c[1:-1]
                    Xc[2:, 1] = c[:-2]
                Kphi = pri["phi_prec"] + Xc.T @ (Xc * w[:, None])
                rhs = pri["phi_prec"] @ pri["phi_mean"] + Xc.T @ (w * c)
                phihat = np.linalg.solve(Kphi, rhs)
                Cphi = np.linalg.cholesky(Kphi)
                for _ in range(200):
                    cand = phihat + solve_triangular(
                        Cphi.T, rng.standard_normal(2), lower=False)
                    if _stationary_ar2(cand, pri["phi_margin"],
                                       pri["phi_max_modulus"]):
                        st["phi"] = cand
                        break

            # ---- 2b) drift parameters (ar1 only) ---------------------------
            if self.drift == "ar1":
                wz = 1.0 / (st["lam_zeta"] * st["sig2_zeta"])
                gt, gtm1, wzt = g[1:], g[:-1], wz[1:]
                a = 1.0 - st["rho_g"]
                resid_g = gt - st["rho_g"] * gtm1
                prec = 1.0 / pri["gbar_var"] + a * a * wzt.sum()
                mean = (pri["gbar_mean"] / pri["gbar_var"]
                        + a * float(wzt @ resid_g)) / prec
                st["gbar"] = mean + rng.standard_normal() / np.sqrt(prec)

                u, v = gt - st["gbar"], gtm1 - st["gbar"]
                prec = 1.0 / pri["rho_g_var"] + float(wzt @ (v * v))
                mean = (pri["rho_g_mean"] / pri["rho_g_var"]
                        + float(wzt @ (v * u))) / prec
                sd = 1.0 / np.sqrt(prec)
                lo = (pri["rho_g_min"] - mean) / sd
                hi = (pri["rho_g_max"] - mean) / sd
                st["rho_g"] = float(truncnorm.rvs(lo, hi, loc=mean, scale=sd,
                                                  random_state=rng))

            # ---- 2c) innovations for the just-updated dynamics -------------
            phi1, phi2 = st["phi"]
            eps = np.empty(T)
            eps[0] = c[0]
            eps[1] = c[1] - phi1 * c[0]
            eps[2:] = c[2:] - phi1 * c[1:-1] - phi2 * c[:-2]
            eta = draw["eta"]
            alpha = (1.0 - st["rho_g"]) * st["gbar"]
            zeta = np.empty(T)
            zeta[0] = g[0] - self.priors["g0_mean"]
            zeta[1:] = g[1:] - st["rho_g"] * g[:-1] - alpha
            innov = {"eps": eps, "eta": eta, "zeta": zeta}

            # ---- 2d) constant shock variances (inverse-gamma) --------------
            for key, resid, lam, nu_k, S_k, skip0 in (
                    ("sig2_c", eps, st["lam_c"], pri["sig2_c_nu"],
                     pri["sig2_c_S"], False),
                    ("sig2_eta", eta, st["lam_eta"], pri["sig2_eta_nu"],
                     pri["sig2_eta_S"], True),
                    ("sig2_zeta", zeta, st["lam_zeta"], pri["sig2_zeta_nu"],
                     pri["sig2_zeta_S"], True)):
                if not self._variance_is_free(key):
                    if key in self._fixed_variances:
                        st[key] = self._fixed_variances[key]
                    continue
                r = (resid[1:] / np.sqrt(lam[1:])) if skip0 \
                    else (resid / np.sqrt(lam))
                shape = nu_k + r.shape[0] / 2.0
                scale = S_k + 0.5 * float(r @ r)
                st[key] = float(scale / rng.gamma(shape, 1.0))
            self._apply_tied(st)

            # ---- 3) COVID scale parameters (griddy Gibbs) ------------------
            for comp in self.covid_components:
                rname, lname, sname, cname = comp_map[comp]
                block = _sample_covid_block(
                    st[f"cov_{comp}"], innov[rname], st[sname],
                    pri[cname], self.covid_t0, rng)
                st[f"cov_{comp}"] = block
                st[lname] = block["lambda"]

            # ---- store -----------------------------------------------------
            if sweep >= self.burnin and (sweep - self.burnin) % thin == 0:
                out["trend"][s] = draw["trend"]
                out["drift"][s] = g
                out["cycle"][s] = c
                out["phi"][s] = st["phi"]
                out["rho_g"][s] = st["rho_g"]
                out["gbar"][s] = st["gbar"]
                out["sig2_c"][s] = st["sig2_c"]
                out["sig2_eta"][s] = st["sig2_eta"]
                out["sig2_zeta"][s] = st["sig2_zeta"]
                params = self._params_from_state(st)
                if store_loglik:
                    ll, ll_t = kalman_loglik(y, params)
                    out["loglik"][s] = ll
                    out["loglik_t"][s] = ll_t
                else:
                    out["loglik"][s] = integrated_loglik(y, params)
                for comp in self.covid_components:
                    blk = st[f"cov_{comp}"]
                    out[f"cov_{comp}"][s] = [blk["lambda1"], blk["lambda2"],
                                             blk["lambda3"], blk["rho_decay"]]
                    out[f"lambda_{comp}"][s] = blk["lambda"]
                s += 1

        self.draws = out
        return out

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def _require_draws(self):
        if self.draws is None:
            raise RuntimeError("Call sample_posterior() first.")

    def decomposition_frame(self, cred: float = 0.68) -> pd.DataFrame:
        """
        Tidy decomposition, date-indexed. Bayesian (after `sample_posterior`):
        posterior median and central ``cred`` band of trend / cycle / drift.
        Classical (after `fit_mle` only): smoothed states with +/- z * se
        bands from the same precision object.

        The default band is the central 68% interval (one posterior standard
        deviation for a symmetric posterior), the convention used across the
        plotting helpers; pass ``cred=0.90`` for the wider interval.

        Note that ``c_t = y_t - tau_t`` holds draw by draw, so the trend and
        cycle bands have identical width by construction: uncertainty about
        potential output *is* uncertainty about the gap.
        """
        lo_q = 100 * (1 - cred) / 2
        hi_q = 100 - lo_q
        if self.draws is not None:
            d = self.draws
            data = {}
            for comp in ("trend", "cycle", "drift"):
                arr = d[comp]
                data[comp] = np.median(arr, axis=0)
                data[f"{comp}_lower"] = np.percentile(arr, lo_q, axis=0)
                data[f"{comp}_upper"] = np.percentile(arr, hi_q, axis=0)
            return pd.DataFrame(data, index=self.dates)
        if self.mle_ is not None:
            from scipy.stats import norm
            z = norm.ppf(1 - (1 - cred) / 2)
            m = self.mle_
            return pd.DataFrame({
                "trend": m["trend"],
                "trend_lower": m["trend"] - z * m["trend_se"],
                "trend_upper": m["trend"] + z * m["trend_se"],
                "cycle": m["cycle"],
                "cycle_lower": m["cycle"] - z * m["trend_se"],
                "cycle_upper": m["cycle"] + z * m["trend_se"],
                "drift": m["drift"],
                "drift_lower": m["drift"] - z * m["drift_se"],
                "drift_upper": m["drift"] + z * m["drift_se"],
            }, index=self.dates)
        raise RuntimeError("Call sample_posterior() or fit_mle() first.")

    def forecast_frame(self, fhor: int = 8,
                       quantiles=(0.05, 0.16, 0.84, 0.95)) -> pd.DataFrame:
        """
        Predictive draws of the observable and its trend, iterated from the
        final states of each posterior draw (COVID scale paths extended by
        their geometric tails). Tidy frame indexed by (variable, horizon)
        with ``[date?, mean, median, qXX...]`` columns, as in `BayesianVAR`.
        """
        self._require_draws()
        rng = self._rng("forecast")
        d = self.draws
        n = d["trend"].shape[0]
        sims_y = np.zeros((n, fhor))
        sims_tau = np.zeros((n, fhor))
        lam_ext = self._extended_lambdas(fhor)

        for i in range(n):
            tau, g = d["trend"][i, -1], d["drift"][i, -1]
            c1, c2 = d["cycle"][i, -1], d["cycle"][i, -2]
            phi1, phi2 = d["phi"][i]
            rho, gbar = d["rho_g"][i], d["gbar"][i]
            alpha = (1 - rho) * gbar
            for h in range(fhor):
                tau_new = tau + g + rng.standard_normal() * np.sqrt(
                    d["sig2_eta"][i] * lam_ext["trend_level"][i, h])
                g = rho * g + alpha + rng.standard_normal() * np.sqrt(
                    d["sig2_zeta"][i] * lam_ext["trend_growth"][i, h])
                c_new = phi1 * c1 + phi2 * c2 + rng.standard_normal() * np.sqrt(
                    d["sig2_c"][i] * lam_ext["cycle"][i, h])
                c2, c1, tau = c1, c_new, tau_new
                sims_tau[i, h] = tau_new
                sims_y[i, h] = tau_new + c_new

        return self._frame_from_sims(sims_y, sims_tau, fhor, quantiles)

    # -- shared forecasting plumbing -----------------------------------

    def _forecast_dates(self, fhor: int) -> Optional[pd.DatetimeIndex]:
        try:
            freq = pd.infer_freq(self.dates)
            offset = pd.tseries.frequencies.to_offset(freq)
            return pd.DatetimeIndex(
                [self.dates[-1] + offset * (h + 1) for h in range(fhor)])
        except (ValueError, TypeError):
            return None

    def _extended_lambdas(self, fhor: int) -> Dict[str, np.ndarray]:
        """COVID scale paths over the forecast horizon, per posterior draw
        (the geometric tails continue their decay toward one)."""
        d = self.draws
        n = d["trend"].shape[0]
        T = self.T
        lam_ext = {}
        for comp in ("cycle", "trend_level", "trend_growth"):
            if comp in self.covid_components:
                blk = d[f"cov_{comp}"]
                lam_ext[comp] = np.stack([
                    covid_lambda_path(T + fhor, self.covid_t0, *blk[i])
                    for i in range(n)])[:, T:]
            else:
                lam_ext[comp] = np.ones((n, fhor))
        return lam_ext

    def _frame_from_sims(self, sims_y, sims_tau, fhor, quantiles) -> pd.DataFrame:
        qs = sorted(float(q) for q in quantiles)
        fdates = self._forecast_dates(fhor)
        index = pd.MultiIndex.from_product(
            [[self.name, "trend"], range(1, fhor + 1)],
            names=["variable", "horizon"])
        data = {}
        if fdates is not None:
            data["date"] = np.tile(fdates, 2)
        sims = np.concatenate([sims_y, sims_tau], axis=1)  # (n, 2*fhor)
        data["mean"] = sims.mean(axis=0)
        data["median"] = np.median(sims, axis=0)
        for q in qs:
            data[f"q{round(q * 100):02d}"] = np.quantile(sims, q, axis=0)
        return pd.DataFrame(data, index=index)

    def conditional_forecast(self, fhor: int = 8,
                             annual_growth: Optional[Dict[int, float]] = None,
                             conditions: Optional[dict] = None,
                             shock_uncertainty: bool = True,
                             anchor_states: bool = False) -> dict:
        """
        Conditional predictive draws in the spirit of Waggoner & Zha (1999):
        future shocks are drawn from their Gaussian distribution conditional
        on linear restrictions holding exactly along each path.

        Parameters
        ----------
        fhor : int
            Forecast horizon (periods after the sample end). Must reach the
            last period referenced by the restrictions. With no restrictions
            at all, the method returns unconditional predictive draws (the
            same distribution `forecast_frame` summarizes).
        annual_growth : dict {year: growth}, optional
            Average year-over-year growth for calendar years, in percent
            (log-approximation: the mean over the year's quarters of
            ``y_t - y_{t-4}`` when ``y`` is 100 x log). Quarters already
            observed enter the restriction as data, so a partially observed
            year constrains only its remaining quarters. Quarterly data only.
        conditions : dict {period: value}, optional
            Hard values for future observations of ``y`` (same scale as the
            input series), keyed by anything ``pd.Period`` parses
            (e.g. ``"2027Q1"``).
        shock_uncertainty : bool, default True
            Draw from the full conditional shock distribution
            N(R+ r, I - R+ R); ``False`` keeps the minimum-norm shocks only
            (band widths then understate uncertainty).
        anchor_states : bool, default False
            Start every path from the POSTERIOR MEDIAN of the final state
            (trend and drift medians; the cycle anchors follow from the
            observed data, ``c_T = y_T - median(tau_T)``) instead of each
            draw's own final state. This conditions the predictive on the
            current state estimate, so the uncertainty of the trend/cycle
            split at time T is removed and fan charts of the UNOBSERVED
            components start narrow and widen with the horizon, the
            central-bank convention for gap and potential-growth fans.
            Parameter, COVID-scale and future-shock uncertainty remain.

        Returns
        -------
        dict with draws ``y``, ``trend``, ``drift``, ``cycle`` (each
        ``(n_draws, fhor)``) and the forecast ``dates``.

        Notes
        -----
        Per posterior draw, future observations are linear in the
        standardized shocks, ``y = d + M u`` with ``u ~ N(0, I)``; the
        restrictions become ``A u = b`` and shocks are drawn from the exact
        conditional via the SVD (minimum norm plus null-space noise), the
        same construction used by ``BayesianVAR.conditional_forecast``.
        Parameter and COVID-scale uncertainty is integrated by conditioning
        draw by draw.
        """
        self._require_draws()
        d = self.draws
        n = d["trend"].shape[0]
        T = self.T
        H = int(fhor)
        fdates = self._forecast_dates(H)
        if fdates is None:
            raise ValueError("conditional_forecast needs a regular "
                             "datetime index.")
        freq = pd.infer_freq(self.dates)
        pfreq = {"Q": "Q", "M": "M", "A": "Y", "Y": "Y"}.get(
            str(freq)[0].upper() if freq else "", None)
        if pfreq is None:
            raise ValueError(f"Unsupported frequency {freq!r} for "
                             "conditional_forecast.")
        periods = self.dates.append(fdates).to_period(pfreq)

        def position(p: pd.Period) -> int:
            pos = periods.get_indexer([p])[0]
            if pos < 0:
                raise ValueError(f"{p} outside the sample + forecast window; "
                                 f"increase fhor.")
            return int(pos)

        # with no restrictions the machinery reduces exactly to unconditional
        # predictive draws (A empty, u ~ N(0, I)): the hands-free forecast
        rows, rhs = [], []
        for key, val in (conditions or {}).items():
            pos = position(pd.Period(str(key), freq=pfreq))
            if pos < T:
                raise ValueError(f"conditions key {key!r} is already observed.")
            row = np.zeros(H)
            row[pos - T] = 1.0
            rows.append(row)
            rhs.append(float(val))
        if annual_growth:
            if pfreq != "Q":
                raise ValueError("annual_growth requires quarterly data.")
            for year, g in annual_growth.items():
                coef = np.zeros(H)
                const = 0.0
                any_future = False
                for q in range(1, 5):
                    pos_t = position(pd.Period(f"{int(year)}Q{q}", freq=pfreq))
                    pos_l = pos_t - 4
                    if pos_l < 0:
                        raise ValueError(f"Year {year} needs {year - 1} data "
                                         "for its year-over-year base.")
                    for pos, sign in ((pos_t, 0.25), (pos_l, -0.25)):
                        if pos >= T:
                            coef[pos - T] += sign
                            any_future = True
                        else:
                            const += sign * self.y[pos]
                if not any_future:
                    raise ValueError(f"Year {year} is fully observed; nothing "
                                     "to condition on.")
                rows.append(coef)
                rhs.append(float(g) - const)
        A_y = np.asarray(rows).reshape(len(rows), H)
        b_y = np.asarray(rhs)

        rng = self._rng("conditional")
        if anchor_states:
            tau_T = float(np.median(d["trend"][:, -1]))
            tau_Tm1 = float(np.median(d["trend"][:, -2]))
            s_anchor = np.array([tau_T, float(np.median(d["drift"][:, -1])),
                                 self.y[-1] - tau_T, self.y[-2] - tau_Tm1])
        lam = self._extended_lambdas(H)
        Z = np.array([1.0, 0.0, 1.0, 0.0])
        R = np.zeros((4, 3))
        R[0, 0] = R[1, 1] = R[2, 2] = 1.0
        sims = {k: np.zeros((n, H)) for k in ("y", "trend", "drift", "cycle")}

        for i in range(n):
            phi1, phi2 = d["phi"][i]
            rho, gbar = d["rho_g"][i], d["gbar"][i]
            Tm = np.array([[1.0, 1.0, 0.0, 0.0],
                           [0.0, rho, 0.0, 0.0],
                           [0.0, 0.0, phi1, phi2],
                           [0.0, 0.0, 1.0, 0.0]])
            cvec = np.array([0.0, (1.0 - rho) * gbar, 0.0, 0.0])
            sds = np.stack([
                np.sqrt(d["sig2_eta"][i] * lam["trend_level"][i]),
                np.sqrt(d["sig2_zeta"][i] * lam["trend_growth"][i]),
                np.sqrt(d["sig2_c"][i] * lam["cycle"][i])])       # (3, H)

            if anchor_states:
                s = s_anchor.copy()
            else:
                s = np.array([d["trend"][i, -1], d["drift"][i, -1],
                              d["cycle"][i, -1], d["cycle"][i, -2]])
            det = np.zeros((H, 4))
            for h in range(H):
                s = Tm @ s + cvec
                det[h] = s

            M = np.zeros((H, 4, 3 * H))    # loadings of u on the state path
            for j in range(H):
                B = R * sds[:, j]          # R @ diag(sd_j)
                for h in range(j, H):
                    M[h, :, 3 * j:3 * j + 3] = B
                    B = Tm @ B
            My = np.einsum("k,hkm->hm", Z, M)
            d_y = det @ Z

            A = A_y @ My
            b = b_y - A_y @ d_y
            u = _solve_constrained_u(A, b, rng if shock_uncertainty else None)
            sims["y"][i] = d_y + My @ u
            sims["trend"][i] = det[:, 0] + M[:, 0, :] @ u
            sims["drift"][i] = det[:, 1] + M[:, 1, :] @ u
            sims["cycle"][i] = det[:, 2] + M[:, 2, :] @ u

        sims["dates"] = fdates
        return sims

    def conditional_forecast_frame(self, fhor: int = 8,
                                   annual_growth: Optional[Dict[int, float]] = None,
                                   conditions: Optional[dict] = None,
                                   quantiles=(0.05, 0.16, 0.84, 0.95),
                                   shock_uncertainty: bool = True,
                                   anchor_states: bool = False) -> pd.DataFrame:
        """Tidy version of :meth:`conditional_forecast`, in the layout of
        :meth:`forecast_frame`."""
        sims = self.conditional_forecast(fhor, annual_growth=annual_growth,
                                         conditions=conditions,
                                         shock_uncertainty=shock_uncertainty,
                                         anchor_states=anchor_states)
        return self._frame_from_sims(sims["y"], sims["trend"], fhor, quantiles)

    def waic(self) -> dict:
        """
        In-sample comparison criteria from the one-step-ahead predictive
        decomposition (Watanabe 2010; Spiegelhalter et al. 2002), conditional
        on the sampled variance paths — identical across specifications, so
        the ladder comparison is internally consistent.

        Returns dict with ``waic``, ``pwaic``, ``elpd_waic``, ``lppd``,
        ``dic``, ``pD``. Lower WAIC/DIC = better.
        """
        self._require_draws()
        d = self.draws
        if d["loglik_t"] is None:
            raise RuntimeError("sample_posterior(store_loglik=True) is required.")
        ll_t = d["loglik_t"]                        # (nsave, T)
        m = ll_t.max(axis=0)
        lppd = float(np.sum(m + np.log(np.mean(np.exp(ll_t - m), axis=0))))
        pwaic = float(np.sum(ll_t.var(axis=0, ddof=1)))
        elpd = lppd - pwaic

        # DIC at posterior-mean parameters (variance paths averaged too)
        params_bar = {
            "phi": d["phi"].mean(axis=0),
            "rho_g": float(d["rho_g"].mean()),
            "gbar": float(d["gbar"].mean()),
            "q_c": (d["sig2_c"].mean() * self._mean_lambda("cycle")),
            "q_eta": (d["sig2_eta"].mean() * self._mean_lambda("trend_level")),
            "q_zeta": (d["sig2_zeta"].mean() * self._mean_lambda("trend_growth")),
            "tau0_mean": self.priors["tau0_mean"], "tau0_var": self.priors["tau0_var"],
            "g0_mean": self.priors["g0_mean"], "g0_var": self.priors["g0_var"],
        }
        Dbar = -2.0 * float(d["loglik"].mean())
        Dhat = -2.0 * integrated_loglik(self.y, params_bar)
        pD = Dbar - Dhat
        return {"waic": -2.0 * elpd, "pwaic": pwaic, "elpd_waic": elpd,
                "lppd": lppd, "dic": Dhat + 2.0 * pD, "pD": pD}

    def _mean_lambda(self, comp: str) -> np.ndarray:
        if comp in self.covid_components:
            return self.draws[f"lambda_{comp}"].mean(axis=0)
        return np.ones(self.T)

    def model_summary(self):
        """Print a summary of the UC model specification."""
        display(generate_uc_summary(self))

    # ------------------------------------------------------------------
    # Plotting (thin wrappers; implementation in plots_uc)
    # ------------------------------------------------------------------

    def plot_decomposition(self, **kwargs):
        """Trend + cycle decomposition with credible/confidence bands."""
        from .plots_uc import plot_uc_decomposition
        return plot_uc_decomposition(self, **kwargs)

    def plot_covid_scales(self, **kwargs):
        """Posterior COVID variance-scale paths per component."""
        from .plots_uc import plot_covid_scales
        return plot_covid_scales(self, **kwargs)


def compare_uc(models: Dict[str, UCTrendCycle]) -> pd.DataFrame:
    """
    WAIC/DIC comparison table for fitted UC models (lower = better). Rows are
    model labels; the last column flags the WAIC winner.
    """
    rows = []
    for name, m in models.items():
        crit = m.waic()
        rows.append({"model": name, **{k: round(v, 2) for k, v in crit.items()}})
    tab = pd.DataFrame(rows).set_index("model")
    best = tab["waic"].idxmin()
    tab["best_waic"] = ["*" if ix == best else "" for ix in tab.index]
    return tab


__all__ = ["hp_filter", "UCTrendCycle", "compare_uc", "make_priors",
           "covid_lambda_path", "integrated_loglik", "kalman_loglik"]
