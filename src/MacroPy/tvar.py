"""
Threshold VAR with regime-specific covariances (Alessandri & Mumtaz, 2017).

Two-regime VAR in which each regime has its own coefficients and its own
covariance matrix; the regime is set by a threshold on a lagged endogenous
variable observed with an unknown delay ``d``::

    Z_t = ( c_1 + sum_j B_1j Z_{t-j} + u_1t ) * S_t
        + ( c_2 + sum_j B_2j Z_{t-j} + u_2t ) * (1 - S_t),    u_it ~ N(0, Sigma_i)

    S_t = 1   <=>   F_{t-d} <= Z*

This is the specification of Alessandri, P. and Mumtaz, H. (2017), "Financial
conditions and density forecasts for US output and inflation", *Review of
Economic Dynamics* 24, 66-78, and the homoskedastic-within-regime counterpart
of :class:`MacroPy.ThresholdVARSV` (Alessandri & Mumtaz, 2019): the same
regimes, threshold, delay and coefficient prior, but no common volatility
factor and no volatility-in-mean terms. Volatility can only change when the
regime changes. Both classes share their API, so the two specifications can be
estimated on the same sample and compared side by side
(:func:`compare_threshold_models`, :func:`MacroPy.plot_irf_comparison`,
:func:`MacroPy.plot_regime_comparison`, :func:`MacroPy.plot_volatility_comparison`).

Prior: Banbura-Giannone-Reichlin dummy observations for each regime's
``(B_i, Sigma_i)`` (a proper Normal-inverse-Wishart), a Normal prior on ``Z*``
restricted to splits that leave at least ``k`` observations per regime, a
uniform prior on ``d`` and stable coefficients in both regimes.

Posterior simulation is exact. The prior is conjugate given the split, so the
marginal likelihood of every admissible split ``(d, interval of Z*)`` is known
in closed form and the posterior of ``(Z*, d)`` is computed on the full grid.
Each draw takes ``(d, Z*)`` from it, then ``Sigma_i ~ IW`` and ``B_i | Sigma_i``
matrix normal in each regime, and rejects the whole draw when a regime is
unstable; accepted draws are independent draws from the truncated posterior.
This targets the same posterior as the authors' Gibbs sampler (coefficients,
covariances, random-walk Metropolis for ``Z*``, multinomial ``d``) but cannot
get stuck: the threshold likelihood is a step function with sharp, separated
modes, which a random-walk chain may never leave.

Generalized impulse responses (Koop, Pesaran & Potter, 1996) are computed by
Monte Carlo with common random numbers and reported by regime of origin, with
the same layout as :meth:`ThresholdVARSV.compute_irfs`.
"""
from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from IPython.display import display
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.special import gammaln, log_ndtr, logsumexp, ndtr, ndtri
from scipy.stats import norm
from tqdm import tqdm

from .summary import generate_tvar_summary
from .tvarsv import (
    _HAS_JOBLIB, _ar1_moments, _build_regressors, _gaussian_log_scores, _log_predictive_score,
    _sample_start, _stability,
)

if _HAS_JOBLIB:
    from joblib import Parallel, delayed

__all__ = ["ThresholdVAR", "compare_threshold_models", "recursive_log_scores",
           "compare_log_scores", "TVAR_PRIOR_DEFAULTS"]

#: Default hyperparameters of :class:`ThresholdVAR` (the coefficient and
#: threshold settings are those of :class:`ThresholdVARSV`, so both models
#: share the prior).
TVAR_PRIOR_DEFAULTS = dict(
    lamdaP=0.1,         # overall tightness of the dummy prior
    tauP=1.0,           # sum-of-coefficients dummies (0 switches them off)
    epsilonP=1.0,       # constant dummy (smaller is looser; 1e-3 is nearly flat)
    RW=False,           # True: prior mean of own first lag at 1 for every variable
    maxdraws=1000,      # consecutive unstable draws before repeating the previous one
    tarvariance=10.0,   # prior variance of the threshold Z*
    sigma_scale="sd",   # dummy-prior scale: AR(1) residual "sd" (Banbura et al.) or "variance"
)

_LOGPI = float(np.log(np.pi))


# ===========================================================================
# Prior, marginal likelihoods and conditional draws
# ===========================================================================


def _create_dummies(lamda: float, tau: float, delta: np.ndarray, epsilon: float,
                    p: int, mu: np.ndarray, sigma: np.ndarray, n: int):
    """Banbura-Giannone-Reichlin dummy observations (``create_dummies.m``).

    Rows: the lag block (tightness ``lamda``, prior mean ``delta`` on the own
    first lag), ``diag(sigma)`` for the covariance, the constant (``epsilon``)
    and, when ``tau > 0``, the sum-of-coefficients block. Regressor order is
    ``[lag 1, ..., lag p, constant]``.
    """
    np_ = n * p
    yd1 = np.vstack([np.diag(sigma * delta) / lamda, np.zeros((n * (p - 1), n)),
                     np.diag(sigma), np.zeros((1, n))])
    xd1 = np.zeros((np_ + n + 1, np_ + 1))
    xd1[:np_, :np_] = np.kron(np.diag(np.arange(1.0, p + 1)), np.diag(sigma) / lamda)
    xd1[-1, -1] = epsilon
    if tau <= 0:
        return yd1, xd1
    yd2 = np.diag(delta * mu) / tau
    xd2 = np.hstack([np.kron(np.ones((1, p)), yd2), np.zeros((n, 1))])
    return np.vstack([yd1, yd2]), np.vstack([xd1, xd2])


def _draw_iw(S: np.ndarray, df: float, rng: np.random.Generator) -> np.ndarray:
    """``Sigma ~ IW(S, df)`` (mean ``S / (df - n - 1)``) by the Bartlett decomposition."""
    n = S.shape[0]
    A = np.tril(rng.standard_normal((n, n)), -1)
    A[np.diag_indices(n)] = np.sqrt(rng.chisquare(df - np.arange(n)))
    K = solve_triangular(A, cholesky(S).T, lower=True)     # A^-1 L_S'
    return K.T @ K


def _log_mvgamma(a, n: int):
    """Log multivariate gamma function ``log Gamma_n(a)`` (``a`` may be an array)."""
    return 0.25 * n * (n - 1) * _LOGPI + sum(gammaln(a - 0.5 * j) for j in range(n))


def _niw_kernel(XX: np.ndarray, XY: np.ndarray, YY: np.ndarray, nu):
    """``log`` of the Normal-inverse-Wishart normalising constant, up to terms
    that cancel in a marginal likelihood (batched over leading axes).

    With ``XX = X'X``, ``XY = X'Y``, ``YY = Y'Y`` and ``nu = rows - k``:
    ``-N/2 log|XX| - nu/2 log|S| + log Gamma_N(nu/2)``, ``S = YY - XY' XX^-1 XY``.
    """
    N = YY.shape[-1]
    L = cholesky(XX)
    W = np.linalg.solve(L, XY)                               # L^-1 X'Y
    S = YY - np.swapaxes(W, -1, -2) @ W
    logdet_xx = 2.0 * np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1)
    logdet_s = np.linalg.slogdet(S)[1]
    return -0.5 * N * logdet_xx - 0.5 * nu * logdet_s + _log_mvgamma(0.5 * nu, N)


class _DummyPrior:
    """Moments of the dummy observations and the prior's own kernel."""

    def __init__(self, yd: np.ndarray, xd: np.ndarray):
        self.yd, self.xd = yd, xd
        self.XX, self.XY, self.YY = xd.T @ xd, xd.T @ yd, yd.T @ yd
        self.rows, self.k = xd.shape
        self.kernel = _niw_kernel(self.XX, self.XY, self.YY, self.rows - self.k)

    def log_ml(self, XX, XY, YY, n):
        """Log marginal likelihood of ``n`` observations with moments ``XX, XY, YY``
        (data only; the dummies are added here). Batched over leading axes."""
        N = YY.shape[-1]
        post = _niw_kernel(XX + self.XX, XY + self.XY, YY + self.YY, n + self.rows - self.k)
        return -0.5 * n * N * _LOGPI + post - self.kernel


def _log_normal_mass(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``log(Phi(b) - Phi(a))`` for ``a < b``, accurate in both tails."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    flip = a > 0                                   # use the upper tail by symmetry
    lo, hi = np.where(flip, -b, a), np.where(flip, -a, b)
    lhi, llo = log_ndtr(hi), log_ndtr(lo)
    with np.errstate(divide="ignore"):
        return lhi + np.log1p(-np.exp(llo - lhi))


def _split_posterior(y: np.ndarray, x: np.ndarray, Fstar: np.ndarray, prior: _DummyPrior,
                     ncrit: int, tar_mean: float, tar_var: float) -> pd.DataFrame:
    """Posterior of every admissible regime split.

    For delay ``d``, sorting ``F_{t-d}`` turns the threshold into a split
    point: ``Z*`` in ``[v_(n), v_(n+1))`` puts the ``n`` smallest values in
    regime 1. Cumulative sums of ``x x'``, ``x y'``, ``y y'`` in that order
    give both regimes' moments for every ``n`` at once. Each split's weight is
    its prior mass (Normal prior on ``Z*``, uniform on ``d``) times the two
    regimes' marginal likelihoods.
    """
    T, N = y.shape
    xx = x[:, :, None] * x[:, None, :]
    xy = x[:, :, None] * y[:, None, :]
    yy = y[:, :, None] * y[:, None, :]
    sd = np.sqrt(tar_var)
    rows = []
    for d, F in enumerate(Fstar, start=1):
        order = np.argsort(F, kind="stable")
        v = F[order]
        n = np.arange(ncrit, T - ncrit + 1)
        n = n[v[n - 1] < v[n]]                       # split points between distinct values
        if n.size == 0:
            continue
        cxx = np.cumsum(xx[order], axis=0)
        cxy = np.cumsum(xy[order], axis=0)
        cyy = np.cumsum(yy[order], axis=0)
        low = prior.log_ml(cxx[n - 1], cxy[n - 1], cyy[n - 1], n)
        high = prior.log_ml(cxx[-1] - cxx[n - 1], cxy[-1] - cxy[n - 1], cyy[-1] - cyy[n - 1],
                            T - n)
        lower, upper = v[n - 1], v[n]
        mass = _log_normal_mass((lower - tar_mean) / sd, (upper - tar_mean) / sd)
        rows.append(pd.DataFrame({"delay": d, "n_regime1": n, "lower": lower, "upper": upper,
                                  "log_prior_mass": mass, "log_ml": low + high}))
    if not rows:
        raise ValueError("No threshold leaves enough observations in both regimes; "
                         "reduce `lags` or lengthen the sample.")
    out = pd.concat(rows, ignore_index=True)
    logw = out["log_prior_mass"] + out["log_ml"]
    if not np.isfinite(logw).any():
        raise ValueError("The threshold prior puts no mass on admissible splits; "
                         "check `threshold_prior_mean` and `tarvariance`.")
    out["probability"] = np.exp(logw - logsumexp(logw))
    return out


def _parse_window(window) -> tuple:
    """Inclusive ``(start, end)`` timestamps; bare periods such as ``"2020Q1"`` cover their span."""
    start, end = window
    start_ts = start if isinstance(start, pd.Timestamp) else pd.Period(str(start)).start_time
    end_ts = end if isinstance(end, pd.Timestamp) else pd.Period(str(end)).end_time
    if end_ts < start_ts:
        raise ValueError("covid_window end precedes its start.")
    return start_ts, end_ts


def _draw_threshold(lower: float, upper: float, mean: float, sd: float,
                    rng: np.random.Generator) -> float:
    """``Z* ~ N(mean, sd^2)`` truncated to ``[lower, upper)`` (inverse CDF, tail safe)."""
    a, b = (lower - mean) / sd, (upper - mean) / sd
    u = rng.random()
    if a > 0:                                        # upper tail: sample -Z on [-b, -a)
        pa, pb = ndtr(-b), ndtr(-a)
        z = -ndtri(pa + u * (pb - pa))
    else:
        pa, pb = ndtr(a), ndtr(b)
        z = ndtri(pa + u * (pb - pa))
    tar = mean + sd * z if np.isfinite(z) else lower
    return float(min(max(tar, lower), np.nextafter(upper, lower)))


def _draw_niw(y: np.ndarray, x: np.ndarray, prior: _DummyPrior, rng: np.random.Generator):
    """One regime's ``(B, Sigma)`` from its Normal-inverse-Wishart posterior.

    ``Sigma ~ IW(S*, T* - k)`` and ``B = B* + R^-T Z chol(Sigma)'`` with
    ``X*'X* = R R'``: the ``Sigma (x) (X*'X*)^-1`` covariance without forming
    the Kronecker product (``getcoef.m``, ``iwpQ.m``).
    """
    XX = x.T @ x + prior.XX
    R = cholesky(XX)
    W = solve_triangular(R, x.T @ y + prior.XY, lower=True)          # R^-1 X*'Y*
    Bstar = solve_triangular(R.T, W, lower=False)
    S = y.T @ y + prior.YY - W.T @ W
    sigma = _draw_iw(S, y.shape[0] + prior.rows - prior.k, rng)
    B = Bstar + solve_triangular(R.T, rng.standard_normal(Bstar.shape), lower=False) \
        @ cholesky(sigma).T
    return B, sigma


def _sample_draws(seed: int, n_keep: int, split: dict, y: np.ndarray, x: np.ndarray,
                  Fstar: np.ndarray, prior: _DummyPrior, L: int, maxdraws: int,
                  show_progress: bool = False) -> dict:
    """``n_keep`` independent posterior draws (one block of :meth:`ThresholdVAR.sample_posterior`)."""
    rng = np.random.default_rng(seed)
    T, N = y.shape
    k = x.shape[1]
    st = {"B1": np.empty((n_keep, k, N)), "B2": np.empty((n_keep, k, N)),
          "S1": np.empty((n_keep, N, N)), "S2": np.empty((n_keep, N, N)),
          "tar": np.empty(n_keep), "delay": np.empty(n_keep, dtype=int),
          "e1": np.empty((n_keep, T), dtype=bool)}
    prob, sd = split["probability"], np.sqrt(split["tar_var"])
    fixed_tar = split.get("fixed_tar")
    n_proposed = n_failed = 0
    for j in tqdm(range(n_keep), desc="Sampling Posterior", disable=not show_progress):
        for attempt in range(maxdraws):
            n_proposed += 1
            i = rng.choice(prob.shape[0], p=prob)
            delay = int(split["delay"][i])
            tar = (fixed_tar if fixed_tar is not None else
                   _draw_threshold(split["lower"][i], split["upper"][i], split["tar_mean"], sd, rng))
            low = Fstar[delay - 1] <= tar
            B1, S1 = _draw_niw(y[low], x[low], prior, rng)
            B2, S2 = _draw_niw(y[~low], x[~low], prior, rng)
            if not (_stability(B1.ravel(order="F"), N, L, 1)
                    or _stability(B2.ravel(order="F"), N, L, 1)):
                break
        else:                                        # no stable draw: repeat the previous one
            n_failed += 1
            if j > 0:
                for key in st:
                    st[key][j] = st[key][j - 1]
                continue
        st["B1"][j], st["B2"][j], st["S1"][j], st["S2"][j] = B1, B2, S1, S2
        st["tar"][j], st["delay"][j], st["e1"][j] = tar, delay, low
    st.update(n_proposed=n_proposed, n_failed=n_failed)
    return st


# ===========================================================================
# Generalized impulse responses
# ===========================================================================


def _girf_draw(draw: dict, reps: int, scale: float, hor: int, L: int, N: int,
               tarvar: int, seed: int, irf_1std: int = 1, groups=None) -> dict:
    """GIRFs to every Cholesky shock for one draw, averaged by regime of origin.

    Histories are the sample periods ``t`` in each regime (``F_{t-d}`` at or
    below ``Z*`` for ``regime1``); paths start at ``t + 1``. The baseline and
    the ``N`` shocked scenarios are simulated together and share their future
    innovations (common random numbers): the impact period has no other
    shock, later periods draw ``z ~ N(0, I)`` scaled by the Cholesky factor of
    the regime each path is in.
    """
    B = (draw["B1"], draw["B2"])
    C = (cholesky(draw["S1"]), cholesky(draw["S2"]))
    tar, delay, y = float(draw["tar"]), int(draw["delay"]), draw["y"]
    pos = draw.get("pos", np.arange(y.shape[0]))      # likelihood rows -> rows of `y`
    kept = draw.get("kept", np.ones(y.shape[0], dtype=bool))
    NL = N * L
    pre = max(L, delay)
    # impact of shock s (row s): column s of the Cholesky factor, unit-normalised if asked
    impact = []
    for Ci in C:
        cols = Ci.T * scale
        if irf_1std != 1:
            cols = cols / np.diag(Ci)[:, None]
        impact.append(cols)

    out: dict = {}
    if groups is None:                               # histories by regime of origin
        groups = (("regime1", draw["e1"]), ("regime2", ~draw["e1"]))
    for r, (reg, mask) in enumerate(groups):
        idx = pos[np.where(mask)[0]]
        idx = idx[idx >= pre - 1]
        idx = idx[[kept[t - pre + 1:t + 1].all() for t in idx]]   # histories inside a hole are skipped
        if idx.size == 0:
            out[reg] = None
            continue
        rng = np.random.default_rng([seed, r])
        hist = np.stack([y[t - pre + 1:t + 1] for t in idx])            # (H, pre, N)
        hist = np.repeat(hist, reps, axis=0)                             # (P, pre, N)
        P = hist.shape[0]
        buf = np.broadcast_to(hist, (N + 1, P, pre, N)).copy()          # circular buffer
        means = np.empty((N + 1, hor, N))
        for h in range(hor):
            tau = pre + h
            xrow = np.concatenate([buf[:, :, (tau - j) % pre, :] for j in range(1, L + 1)],
                                  axis=2)
            low = buf[:, :, (tau - delay) % pre, tarvar] <= tar
            if h == 0:
                shock = [np.vstack([np.zeros((1, N)), imp])[:, None, :] for imp in impact]
            else:
                z = rng.standard_normal((P, N))
                shock = [(z @ Ci.T)[None] for Ci in C]
            y1 = xrow @ B[0][:NL] + B[0][NL] + shock[0]
            y2 = xrow @ B[1][:NL] + B[1][NL] + shock[1]
            new = np.where(low[..., None], y1, y2)
            buf[:, :, tau % pre, :] = new
            means[:, h, :] = new.mean(axis=1)
        out[reg] = {"level": np.transpose(means[1:] - means[0], (1, 2, 0))}  # (hor, resp, shock)
    return out


def _tvar_path_scores(d: dict, sel: np.ndarray, target: np.ndarray, horizons: list,
                      reps: int, L: int, tarvar: int, rng: np.random.Generator) -> dict:
    """Simulated predictive log densities for :class:`ThresholdVAR` (see
    ``tvarsv._tvarsv_path_scores``): Gaussian at step ``h`` given the path."""
    idx = np.repeat(sel, reps)
    P, (H, N) = idx.size, target.shape
    B1, B2 = d["B1"][idx], d["B2"][idx]
    C1, C2 = cholesky(d["S1"][idx]), cholesky(d["S2"][idx])
    tar, delay = d["tar"][idx], d["delay"][idx].astype(int)
    pre = max(L, int(delay.max()))
    rows = np.arange(P)
    buf = np.repeat(d["y"][None, -pre:], P, axis=0)                  # circular buffer
    out = {}
    for h in range(1, H + 1):
        tau = pre + h - 1
        xrow = np.concatenate([buf[:, (tau - j) % pre] for j in range(1, L + 1)]
                              + [np.ones((P, 1))], axis=1)
        low = buf[rows, (tau - delay) % pre, tarvar] <= tar
        mean = np.where(low[:, None], np.einsum("pk,pkn->pn", xrow, B1),
                        np.einsum("pk,pkn->pn", xrow, B2))
        C = np.where(low[:, None, None], C1, C2)
        if h in horizons and np.isfinite(target[h - 1]).all():
            out[h] = _gaussian_log_scores(target[h - 1] - mean, C)
        buf[:, tau % pre] = mean + np.einsum("pij,pj->pi", C, rng.standard_normal((P, N)))
    return out


# ===========================================================================
# Public class
# ===========================================================================


class ThresholdVAR:
    """Threshold VAR with regime-specific covariances (Alessandri & Mumtaz, 2017).

    The API mirrors :class:`ThresholdVARSV` (same arguments, attributes and
    output layouts), so both specifications can be estimated on the same
    sample and compared with :func:`compare_threshold_models` and the
    ``plot_*_comparison`` functions.

    Parameters
    ----------
    y : pd.DataFrame
        Endogenous variables with a datetime index; the threshold variable must
        be one of them.
    threshold_var : str or int, default last column
        Name or position of the variable that drives the regime switch.
    lags : int, default 13
        VAR lags ``P``.
    max_delay : int, default 12
        Upper bound for the threshold delay ``d`` (uniform prior on ``1..max_delay``).
    training : int, default 0
        Observations dropped at the start of the sample, after the lags. Set it
        to a :class:`ThresholdVARSV`'s ``training`` (20 by default there) to
        estimate both models on the same sample.
    post_draws, burnin : int, float, default 5000, 0.5
        As in :class:`ThresholdVARSV`: ``post_draws * (1 - burnin)`` draws are
        retained. The sampler is exact, so it needs no burn-in and simply
        generates that many independent draws, which keeps the two classes'
        output sizes equal for the same settings.
    hor : int, default 48
        Horizon of the generalized impulse responses.
    irf_1std : int, default 1
        ``1``: one-standard-deviation Cholesky shocks; ``0``: unit shocks (the
        shocked variable moves by ``scale`` units on impact), the comparable
        choice across models whose shock sizes differ.
    threshold_prior_mean : float, optional
        Prior mean of ``Z*`` (default: the sample mean of the threshold
        variable). Combine with a small ``tarvariance`` for an informative
        prior on a known cut-off.
    prior_params : dict, optional
        Overrides for :data:`TVAR_PRIOR_DEFAULTS`; unknown keys raise.
    seed : int, optional
        Base seed (block ``i`` of draws uses ``seed + i``).
    covid_mode : {"lenza-primiceri", "dummies"}, optional
        Treatment of the observations in ``covid_window``, as in
        :class:`BayesianVAR`. ``"lenza-primiceri"``: the rows are divided by
        volatility scales ``s_t >= 1`` before estimation (exact GLS under the
        conjugate prior; ``log_marginal_likelihood`` includes the Jacobian),
        so they stay in the sample with a smaller weight. ``"dummies"``: the
        rows leave the likelihood altogether, which is what one dummy per
        observation with a diffuse prior does; the lags are still built on
        the full data, so the sample keeps its timing and the threshold
        variable its delays.
    covid_window : tuple, optional
        Inclusive ``(start, end)``; periods such as ``("2020Q1", "2021Q4")``
        or timestamps.
    covid_scales : float or array-like, optional
        Fixed scales for ``"lenza-primiceri"`` (a scalar or one value per
        window observation). If None they are estimated by empirical Bayes
        on the linear VAR with the same prior: the first ``covid_n_free``
        free, the rest decaying geometrically toward one
        (``s_{m+j} = 1 + (s_m - 1) rho^j``). Stored as ``covid_scales``.
    covid_n_free : int, default 3
        Freely estimated scales at the start of the window; set it to the
        window length to free them all.
    covid_drop_lags : bool, default True
        Under ``"dummies"``, also drop the rows whose lagged regressors fall
        in the window (the ``lags`` observations after it), so that the
        outliers never enter the right-hand side either.
    delay : int, optional
        Fix the threshold delay ``d`` instead of estimating it (it must lie in
        ``1..max_delay``; the sample still starts at ``max_delay``).
    threshold : float, optional
        Fix the threshold ``Z*`` (a point-mass prior) instead of estimating
        it, e.g. 50 for a diffusion index. The split posterior then has one
        split per delay and ``log_marginal_likelihood`` compares the TVAR at
        that threshold with the linear VAR.

    Notes
    -----
    The estimation sample starts at ``max(lags + training, max_delay)``, so
    every delay is evaluated on the same observations. Under
    ``covid_mode="dummies"``, ``sample_dates`` lists the rows that remain,
    ``regime_probability`` is ``NaN`` on the dropped dates and the GIRF
    histories never straddle a hole.

    The posterior of ``(Z*, d)`` is computed exactly on all admissible splits
    (``split_posterior`` after sampling) and the draws are independent; see
    the module docstring. It is the posterior targeted by the authors' Gibbs
    sampler, which on threshold models of this kind can stay in whichever
    mode it starts near.

    Examples
    --------
    >>> from MacroPy import ThresholdVAR
    >>> model = ThresholdVAR(df, threshold_var="fci", post_draws=10000, seed=1)
    >>> model.sample_posterior()
    >>> model.threshold_summary()
    >>> model.compute_irfs(shock="level")
    >>> model.plot_irfs(shock="fci")
    """

    #: Label used by the comparison helpers.
    specification = "TVAR"

    def __init__(
        self,
        y: pd.DataFrame,
        threshold_var: Optional[Union[str, int]] = None,
        lags: int = 13,
        max_delay: int = 12,
        training: int = 0,
        post_draws: int = 5000,
        burnin: float = 0.5,
        hor: int = 48,
        irf_1std: int = 1,
        threshold_prior_mean: Optional[float] = None,
        prior_params: Optional[dict] = None,
        seed: Optional[int] = None,
        covid_mode: Optional[str] = None,
        covid_window: Optional[tuple] = None,
        covid_scales: Optional[Union[float, Sequence[float]]] = None,
        covid_n_free: int = 3,
        covid_drop_lags: bool = True,
        delay: Optional[int] = None,
        threshold: Optional[float] = None,
    ):
        if not isinstance(y, pd.DataFrame):
            raise ValueError("`y` must be a pandas DataFrame with a datetime index.")
        if y.isna().any().any():
            raise ValueError("`y` cannot contain missing values.")
        if lags < 1 or max_delay < 1:
            raise ValueError("`lags` and `max_delay` must be at least 1.")
        if delay is not None and not 1 <= int(delay) <= max_delay:
            raise ValueError(f"`delay` must be between 1 and max_delay ({max_delay}).")
        if covid_mode not in (None, "lenza-primiceri", "dummies"):
            raise ValueError("covid_mode must be None, 'lenza-primiceri' or 'dummies'.")
        if (covid_mode is None) != (covid_window is None):
            raise ValueError("`covid_mode` and `covid_window` go together.")

        self.names = list(y.columns)
        self.dates = y.index
        self.y_df = y
        self.data = y.to_numpy(dtype=float)
        self.N = self.data.shape[1]
        self.lags = lags
        self.max_delay = max_delay
        self.training = training
        self.post_draws = post_draws
        self.burnin = int(burnin * post_draws)
        self.n_draws = self.post_draws - self.burnin
        self.hor = hor
        self.irf_1std = irf_1std
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

        unknown = set(prior_params or {}) - set(TVAR_PRIOR_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown prior_params {sorted(unknown)}; "
                             f"valid keys are {sorted(TVAR_PRIOR_DEFAULTS)}.")
        self.prior_params = {**TVAR_PRIOR_DEFAULTS, **(prior_params or {})}
        if self.prior_params["sigma_scale"] not in ("sd", "variance"):
            raise ValueError("prior_params['sigma_scale'] must be 'sd' or 'variance'.")
        self.delay = None if delay is None else int(delay)
        self.threshold = None if threshold is None else float(threshold)

        self.EX = 1                                   # constant
        self.k = self.N * lags + self.EX
        self.start = _sample_start(lags, training, max_delay)
        self.covid_mode, self.covid_window = covid_mode, covid_window
        self.covid_dates = pd.DatetimeIndex([])
        self.covid_scales: Optional[np.ndarray] = None
        in_window = np.zeros(len(self.dates), dtype=bool)
        if covid_mode is not None:
            a, b = _parse_window(covid_window)
            in_window = (self.dates >= a) & (self.dates <= b)
            if not in_window[self.start:].any():
                raise ValueError("covid_window contains no observations of the estimation sample.")
            self.covid_dates = self.dates[self.start:][in_window[self.start:]]
        keep = np.ones(len(self.dates), dtype=bool)
        if covid_mode == "dummies":                   # rows out of the likelihood
            keep &= ~in_window
            if covid_drop_lags:                       # and the rows whose lags fall in the window
                for j in range(1, lags + 1):
                    keep[j:] &= ~in_window[:-j]
        self.keep = keep[self.start:]
        self.T = int(self.keep.sum())
        if self.T <= 2 * self.k:
            raise ValueError(f"{self.T} observations are too few for two regimes with "
                             f"{self.k} coefficients per equation.")
        self.weights = np.ones(self.T)                # 1 / s_t on the rows of the likelihood
        if covid_mode == "lenza-primiceri":
            widx = np.flatnonzero(in_window[self.start:])
            self.covid_scales = self._resolve_covid_scales(widx, covid_scales, covid_n_free)
            self.weights[widx] = 1.0 / self.covid_scales
        self.draws: Optional[dict] = None

    # ------------------------------------------------------------------
    def _rows(self):
        """Unweighted rows of the likelihood: ``y``, the regressors and ``F_{t-d}``."""
        y = self.data[self.start:][self.keep]
        x = _build_regressors(self.data, self.lags)[self.start:][self.keep]
        f = self.data[:, self.tarvar]
        Fstar = np.stack([f[self.start - d:self.data.shape[0] - d]
                          for d in range(1, self.max_delay + 1)])[:, self.keep]
        return y, x, Fstar

    def _prior(self, y: np.ndarray) -> _DummyPrior:
        """The dummy prior scaled with the moments of ``y`` (weighted rows under scales)."""
        pp = self.prior_params
        mu, sigma, delta = _ar1_moments(y, pp["RW"])
        if pp["sigma_scale"] == "variance":
            sigma = sigma ** 2
        return _DummyPrior(*_create_dummies(pp["lamdaP"], pp["tauP"], delta, pp["epsilonP"],
                                            self.lags, mu, sigma, self.N))

    def _resolve_covid_scales(self, widx: np.ndarray, covid_scales, n_free: int) -> np.ndarray:
        """Fixed scales if given, otherwise Lenza-Primiceri estimates."""
        W = widx.size
        if covid_scales is not None:
            scales = np.asarray(covid_scales, dtype=float).ravel()
            if scales.size == 1:
                scales = np.repeat(scales, W)
            if scales.size != W:
                raise ValueError(f"covid_scales must be a scalar or have one value per window "
                                 f"observation ({W}); got {scales.size}.")
            if np.any(scales < 1.0):
                raise ValueError("covid_scales must be >= 1.")
            return scales
        return self._estimate_covid_scales(widx, n_free)

    def _estimate_covid_scales(self, widx: np.ndarray, n_free: int) -> np.ndarray:
        """Lenza-Primiceri scales by empirical Bayes on the linear VAR with this prior.

        The first ``n_free`` window observations get free scales and the rest
        decay geometrically toward one, ``s_{m+j} = 1 + (s_m - 1) rho^j``; the
        objective is the exact marginal likelihood of the (GLS-weighted) linear
        VAR under the dummy prior, Jacobian included, so the scales are chosen
        before any regime is looked at.
        """
        from scipy.optimize import minimize

        y, x, _ = self._rows()
        T, N = y.shape
        W = widx.size
        m = int(min(max(n_free, 1), W))

        def scales_from(theta):
            s = np.empty(W)
            s[:m] = np.exp(theta[:m])
            if W > m:
                s[m:] = 1.0 + (s[m - 1] - 1.0) * theta[m] ** np.arange(1, W - m + 1)
            return s

        def neg_logml(theta):
            w = np.ones(T)
            w[widx] = 1.0 / scales_from(theta)
            yw, xw = y * w[:, None], x * w[:, None]
            prior = self._prior(yw)
            val = prior.log_ml(xw.T @ xw, xw.T @ yw, yw.T @ yw, T) + N * np.log(w).sum()
            return 1e12 if not np.isfinite(val) else -val

        resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]       # OLS heuristic to start
        base = np.median(np.abs(resid), axis=0) + 1e-12
        s0 = np.clip(np.median(np.abs(resid[widx[:m]]) / base[None, :], axis=1), 1.5, 50.0)
        x0 = np.concatenate([np.log(s0), [0.7]]) if W > m else np.log(s0)
        bounds = [(0.0, np.log(500.0))] * m + ([(0.01, 0.99)] if W > m else [])
        res = minimize(neg_logml, x0, method="L-BFGS-B", bounds=bounds)
        self._covid_opt = res
        return scales_from(res.x)

    # ------------------------------------------------------------------
    def _setup(self):
        """Estimation sample, threshold variable by delay and the dummy prior."""
        y, x, Fstar = self._rows()
        w = self.weights[:, None]                     # GLS rows: y_t / s_t, x_t / s_t
        y, x = y * w, x * w
        prior = self._prior(y)
        tar_mean = (float(Fstar[0].mean()) if self.threshold_prior_mean is None
                    else float(self.threshold_prior_mean))
        return y, x, Fstar, prior, tar_mean

    def model_summary(self):
        """Display a Markdown summary of the specification."""
        display(generate_tvar_summary(self))

    def _restrict_splits(self, sp: pd.DataFrame) -> pd.DataFrame:
        """Keep the splits allowed by a fixed ``delay`` and/or ``threshold``.

        A fixed threshold is a point-mass prior: one split per delay, with
        equal prior mass across delays (the uniform prior on ``d``).
        """
        if self.delay is not None:
            sp = sp[sp["delay"] == self.delay]
        if self.threshold is not None:
            sp = sp[(sp["lower"] <= self.threshold) & (self.threshold < sp["upper"])].copy()
            sp["log_prior_mass"] = 0.0
        if sp.empty:
            raise ValueError("The fixed `delay`/`threshold` leaves no admissible split: each "
                             f"regime needs at least {self.k} observations.")
        sp = sp.reset_index(drop=True)
        logw = sp["log_prior_mass"] + sp["log_ml"]
        sp["probability"] = np.exp(logw - logsumexp(logw))
        return sp

    def sample_posterior(self, n_chains: int = 1, n_jobs: int = -1,
                         show_progress: bool = True) -> dict:
        """Draw from the exact posterior.

        First computes the posterior of every admissible ``(d, Z*)`` split
        (stored as ``self.split_posterior``), then draws ``(d, Z*)``, the
        covariances and the coefficients independently, rejecting draws with
        an unstable regime.

        Parameters
        ----------
        n_chains : int, default 1
            Independent blocks of ``post_draws * (1 - burnin)`` draws each,
            pooled; kept for symmetry with :meth:`ThresholdVARSV.sample_posterior`.
        n_jobs : int, default -1
            Worker processes for the blocks (joblib); ``-1`` uses all cores.
        show_progress : bool, default True
            Per-draw bar for one block, block-completion bar for several.

        Returns
        -------
        dict
            Pooled draws (also on ``self.draws``): ``B1``, ``B2`` ``(n, k, N)``
            with rows ``[lag 1, ..., lag P, constant]``; ``S1``, ``S2``
            ``(n, N, N)``; ``tar``, ``delay`` ``(n,)``; ``e1`` ``(n, T)``, True
            in regime 1. ``self.stable_share`` is the share of proposals with
            stable coefficients in both regimes, and
            ``self.failed_stability_draws`` counts draws that repeated the
            previous one after ``maxdraws`` unstable proposals.
        """
        y, x, Fstar, prior, tar_mean = self._setup()
        pp = self.prior_params
        self.split_posterior = self._restrict_splits(
            _split_posterior(y, x, Fstar, prior, self.k, tar_mean, pp["tarvariance"]))
        split = {col: self.split_posterior[col].to_numpy()
                 for col in ("delay", "lower", "upper", "probability")}
        split.update(tar_mean=tar_mean, tar_var=pp["tarvariance"], fixed_tar=self.threshold)
        n_keep = self.post_draws - self.burnin

        base = self.seed if self.seed is not None else np.random.SeedSequence().entropy
        seeds = [int((base + i) % (2 ** 32)) for i in range(n_chains)]
        args = (n_keep, split, y, x, Fstar, prior, self.lags, pp["maxdraws"])
        if n_chains == 1:
            blocks = [_sample_draws(seeds[0], *args, show_progress=show_progress)]
        elif _HAS_JOBLIB:
            gen = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_sample_draws)(s, *args) for s in seeds)
            blocks = list(tqdm(gen, total=n_chains, desc="Sampling Posterior",
                               disable=not show_progress))
        else:  # pragma: no cover
            blocks = [_sample_draws(s, *args, show_progress=show_progress and i == 0)
                      for i, s in enumerate(seeds)]

        pooled = {key: np.concatenate([b[key] for b in blocks])
                  for key in ("B1", "B2", "S1", "S2", "tar", "delay", "e1")}
        # histories and forecasts use the unweighted, contiguous sample; `pos` maps
        # the rows of the likelihood (and of `e1`) into it, `kept` flags them
        pooled.update(y=self.data[self.start:], x=x, T=y.shape[0],
                      pos=np.flatnonzero(self.keep), kept=self.keep.copy())
        self.draws = pooled
        self.n_draws = pooled["tar"].shape[0]
        self.stable_share = n_chains * n_keep / sum(b["n_proposed"] for b in blocks)
        self.failed_stability_draws = int(sum(b["n_failed"] for b in blocks))
        if self.failed_stability_draws:
            warnings.warn(f"{self.failed_stability_draws} draws found no stable coefficients "
                          f"in {pp['maxdraws']} proposals and repeat the previous draw.",
                          RuntimeWarning, stacklevel=2)
        return pooled

    def log_marginal_likelihood(self) -> pd.Series:
        """Exact log marginal likelihoods: this TVAR and the linear VAR with the same prior.

        The TVAR's value integrates the threshold and the delay over their
        prior (Normal on ``Z*`` restricted to admissible splits, uniform on
        ``d``); the linear VAR uses the same dummy prior on one regime. Their
        difference is the log Bayes factor of the threshold model. Both ignore
        the stability restriction (the coefficients' prior is not truncated),
        and the dummy prior is scaled with sample moments, as usual for this
        prior.
        """
        if not hasattr(self, "split_posterior"):
            raise RuntimeError("Call sample_posterior() first.")
        y, x, _, prior, _ = self._setup()
        sp = self.split_posterior
        jacobian = self.N * float(np.log(self.weights).sum())     # scaled rows: y_t = s_t y~_t
        tvar = float(logsumexp(sp["log_prior_mass"] + sp["log_ml"])
                     - logsumexp(sp["log_prior_mass"])) + jacobian
        linear = float(prior.log_ml(x.T @ x, x.T @ y, y.T @ y, y.shape[0])) + jacobian
        return pd.Series({"TVAR": tvar, "linear VAR": linear,
                          "log Bayes factor": tvar - linear}, name="log marginal likelihood")

    def _require_draws(self):
        if self.draws is None:
            raise RuntimeError("Call sample_posterior() before requesting results.")

    @property
    def sample_dates(self) -> pd.DatetimeIndex:
        """Dates of the estimation sample (after `exclude`, the rows that remain)."""
        return self.dates[self.start:][self.keep]

    # ------------------------------------------------------------------
    def threshold_summary(self) -> pd.DataFrame:
        """Posterior mean and 5/50/95 percentiles of the threshold and the delay."""
        self._require_draws()
        rows = {}
        for label, v in (("threshold (Z*)", self.draws["tar"]), ("delay (d)", self.draws["delay"])):
            q = np.percentile(v, [5, 50, 95])
            rows[label] = [v.mean(), q[0], q[1], q[2]]
        return pd.DataFrame(rows, index=["mean", "p5", "p50", "p95"]).T

    def regime_probability(self) -> pd.Series:
        """Posterior probability of regime 2 (``F_{t-d} > Z*``) at each date of the
        estimation span; dates dropped by `exclude` are ``NaN``."""
        self._require_draws()
        p = np.full(len(self.keep), np.nan)
        p[self.keep] = 1.0 - self.draws["e1"].mean(axis=0)
        return pd.Series(p, index=self.dates[self.start:], name="P(regime 2)")

    def residual_volatility(self, cred: float = 0.68) -> pd.DataFrame:
        """Posterior of the reduced-form shock standard deviations over time.

        ``sqrt(diag Sigma_{S_t})``: it only moves when the regime changes. The
        same method on :class:`ThresholdVARSV` returns ``sqrt(diag Omega_{S_t,t})``,
        which also moves with the volatility factor, so the two can be
        compared directly (:func:`MacroPy.plot_volatility_comparison`).

        Returns
        -------
        pd.DataFrame
            Indexed by date, columns ``(variable, "median" | "lower" | "upper")``.
        """
        self._require_draws()
        d = self.draws
        sd1 = np.sqrt(np.diagonal(d["S1"], axis1=1, axis2=2))[:, None, :]
        sd2 = np.sqrt(np.diagonal(d["S2"], axis1=1, axis2=2))[:, None, :]
        sd = np.where(d["e1"][:, :, None], sd1, sd2)               # (n, T, N)
        return _band_frame(sd, self.names, self.sample_dates, cred)

    # ------------------------------------------------------------------
    def log_predictive_score(self, future: pd.DataFrame, horizons=(1,), n_draws: int = 500,
                             reps: int = 10) -> pd.DataFrame:
        """Log predictive density of realised data after the end of the sample.

        Same method as :meth:`ThresholdVARSV.log_predictive_score`: paths of
        the regimes and the data are simulated to ``T + h - 1`` for each
        posterior draw, ``y_{T+h}`` is Gaussian given the path, and its
        density at the realised value is averaged over paths.

        Parameters
        ----------
        future : pd.DataFrame
            Observations after the last date of the model's sample (first row
            ``T + 1``), with the model's columns.
        horizons : sequence of int, default (1,)
        n_draws : int, default 500
            Posterior draws used (evenly spaced); ``None`` uses all.
        reps : int, default 10
            Simulated paths per draw (only matters for ``h > 1``).

        Returns
        -------
        pd.DataFrame
            Index ``horizon``; columns ``joint`` and one per variable.
        """
        def simulate(sel, target, horizons, reps, rng):
            return _tvar_path_scores(self.draws, sel, target, horizons, reps, self.lags,
                                     self.tarvar, rng)
        return _log_predictive_score(self, simulate, future, horizons, n_draws, reps)

    def _draw_dict(self, i: int) -> dict:
        d = self.draws
        return dict(B1=d["B1"][i], B2=d["B2"][i], S1=d["S1"][i], S2=d["S2"][i],
                    tar=d["tar"][i], delay=d["delay"][i], e1=d["e1"][i], y=d["y"],
                    pos=d["pos"], kept=d["kept"])

    def compute_irfs(self, shock: Union[str, List[str]] = "level", reps: int = 25,
                     n_draws: Optional[int] = 200, scale: float = 1.0, n_jobs: int = -1,
                     show_progress: bool = True,
                     histories: Optional[Dict[str, object]] = None) -> dict:
        """Generalized impulse responses by regime of origin (Koop, Pesaran & Potter, 1996).

        For each posterior draw, the response is the Monte Carlo mean of a
        shocked minus a baseline simulation, averaged over the sample histories
        in each regime. Shocks are recursive (Cholesky, in column order) and
        regime specific: the impact uses the Cholesky factor of the regime the
        path is in.

        Parameters
        ----------
        shock : "level", default "level"
            Kept for symmetry with :meth:`ThresholdVARSV.compute_irfs`; a model
            without stochastic volatility has no uncertainty shock.
        reps : int, default 25
            Monte Carlo paths per history.
        n_draws : int, optional, default 200
            Posterior draws used (evenly spaced); ``None`` uses all.
        scale : float, default 1.0
            Shock size: standard deviations (``irf_1std=1``) or units of the
            shocked variable (``irf_1std=0``).
        n_jobs : int, default -1
            Parallel workers over draws (joblib).
        histories : dict, optional
            Average over chosen histories instead of the regimes: ``{label:
            selection}`` with a boolean mask aligned with ``sample_dates`` or a
            collection of dates, e.g. the months in which the threshold variable
            was below 45. The paths still switch regime when the simulated
            threshold variable crosses ``Z*``.

        Returns
        -------
        dict
            ``{"regime1": {"level": array}, "regime2": {...}}`` with arrays of
            shape ``(n_draws, hor, N, N)`` (draw, horizon, response, shock), or
            one entry per label of ``histories``. Stored on ``self.irf`` only
            for the regime split.
        """
        self._require_draws()
        shocks = [shock] if isinstance(shock, str) else list(shock)
        if any(s != "level" for s in shocks):
            raise ValueError("ThresholdVAR has no stochastic volatility, hence no "
                             "uncertainty shock; use shock='level'.")
        total = self.draws["tar"].shape[0]
        sel = (np.arange(total) if n_draws is None or n_draws >= total
               else np.linspace(0, total - 1, n_draws).astype(int))
        base = self.seed if self.seed is not None else 12345
        groups = None if histories is None else self._history_groups(histories)
        jobs = [(self._draw_dict(i), reps, scale, self.hor, self.lags, self.N, self.tarvar,
                 int(base + 1 + j), self.irf_1std, groups) for j, i in enumerate(sel)]
        it = tqdm(jobs, desc="Computing IRFs", disable=not show_progress)
        if _HAS_JOBLIB and n_jobs != 1:
            results = Parallel(n_jobs=n_jobs)(delayed(_girf_draw)(*job) for job in it)
        else:
            results = [_girf_draw(*job) for job in it]

        labels = ("regime1", "regime2") if groups is None else [g[0] for g in groups]
        irf: dict = {reg: {} for reg in labels}
        for reg in irf:
            valid = [r[reg]["level"] for r in results if r[reg] is not None]
            if valid:
                irf[reg]["level"] = np.stack(valid)
        if groups is None:
            self.irf = irf
        return irf

    def _history_groups(self, histories: Dict[str, object]) -> list:
        """``[(label, mask over the likelihood rows)]`` from masks or date collections."""
        dates = self.sample_dates
        groups = []
        for label, sel in histories.items():
            if isinstance(sel, pd.Series) and sel.dtype == bool:
                mask = sel.reindex(dates, fill_value=False).to_numpy()
            elif np.asarray(sel).dtype == bool:
                mask = np.asarray(sel)
                if mask.shape != (len(dates),):
                    raise ValueError(f"histories['{label}'] must have one entry per sample date "
                                     f"({len(dates)}); got {mask.shape}.")
            else:
                mask = dates.isin(pd.DatetimeIndex(sel))
            if not mask.any():
                raise ValueError(f"histories['{label}'] selects no sample date.")
            groups.append((str(label), mask))
        return groups

    # ------------------------------------------------------------------
    def plot_regimes(self, **kwargs):
        """Threshold variable with regime 2 shaded (see :func:`MacroPy.plot_regimes`)."""
        from .plots_tvarsv import plot_regimes
        return plot_regimes(self, **kwargs)

    def plot_irfs(self, **kwargs):
        """GIRFs, regime 1 vs regime 2 (see :func:`MacroPy.plot_irfs`)."""
        from .plots_tvarsv import plot_irfs
        return plot_irfs(self, **kwargs)


# ===========================================================================
# Comparison helpers
# ===========================================================================


def _band_frame(draws: np.ndarray, names: List[str], index, cred: float) -> pd.DataFrame:
    """Median and central ``cred`` band of ``(n, T, N)`` draws as a wide frame."""
    lo = 50 * (1 - cred)
    q = np.percentile(draws, [50, lo, 100 - lo], axis=0)            # (3, T, N)
    cols = pd.MultiIndex.from_product([names, ["median", "lower", "upper"]])
    return pd.DataFrame(np.transpose(q, (1, 2, 0)).reshape(q.shape[1], -1), index=index,
                        columns=cols)


def compare_threshold_models(models: Dict[str, object]) -> pd.DataFrame:
    """Side-by-side posterior summary of estimated threshold models.

    Parameters
    ----------
    models : dict
        ``{label: model}`` with :class:`ThresholdVAR` and/or
        :class:`ThresholdVARSV` instances, all sampled.

    Returns
    -------
    pd.DataFrame
        One row per model: specification, estimation sample, threshold median
        and 90% interval, modal delay and its probability, the average
        posterior share of periods in regime 2, the number of dates where the
        posterior favours regime 2, and the retained draws. The table is
        descriptive; it does not rank the models by fit.
    """
    rows = {}
    for label, m in models.items():
        m._require_draws()
        tar, delay = m.draws["tar"], m.draws["delay"]
        values, counts = np.unique(delay, return_counts=True)
        p2 = m.regime_probability()
        q = np.percentile(tar, [5, 50, 95])
        rows[label] = {
            "specification": getattr(m, "specification", type(m).__name__),
            "sample": f"{p2.index[0]:%Y-%m} to {p2.index[-1]:%Y-%m}",
            "T": len(p2),
            "threshold (median)": q[1],
            "threshold p5": q[0],
            "threshold p95": q[2],
            "delay (mode)": int(values[counts.argmax()]),
            "P(delay = mode)": counts.max() / counts.sum(),
            "share in regime 2": float(p2.mean()),
            "dates with P(regime 2) > 0.5": int((p2 > 0.5).sum()),
            "draws": int(tar.shape[0]),
        }
    return pd.DataFrame(rows).T


# ===========================================================================
# Out-of-sample density forecasts
# ===========================================================================


def _score_origin(make_model: Callable, data: pd.DataFrame, origin, horizons, n_draws: int,
                  reps: int) -> pd.DataFrame:
    model = make_model(data.loc[:origin])
    model.sample_posterior(show_progress=False)
    scores = model.log_predictive_score(data.loc[data.index > origin], horizons, n_draws, reps)
    return pd.concat({origin: scores}, names=["origin"])


def recursive_log_scores(make_model: Callable[[pd.DataFrame], object], data: pd.DataFrame,
                         origins: Sequence, horizons=(1,), n_draws: int = 500, reps: int = 10,
                         n_jobs: int = -1, show_progress: bool = True) -> pd.DataFrame:
    """Recursive out-of-sample log predictive scores (expanding window).

    At each origin the model is re-estimated on the data up to that date and
    its predictive density is evaluated at the realised values ``h`` periods
    later. Re-estimating at every period is the cleanest design but the most
    expensive; spacing the origins (e.g. every quarter of monthly data) keeps
    the exercise affordable with :class:`ThresholdVARSV`.

    Parameters
    ----------
    make_model : callable
        ``make_model(data_up_to_origin)`` returns an unsampled model
        (:class:`ThresholdVAR` or :class:`ThresholdVARSV`), e.g.
        ``lambda d: ThresholdVAR(d, threshold_var="fci", seed=1)``.
    data : pd.DataFrame
        Full sample, datetime index.
    origins : sequence of dates
        Last observation of each estimation sample.
    horizons : sequence of int, default (1,)
    n_draws, reps : int
        Passed to ``log_predictive_score``.
    n_jobs : int, default -1
        Origins estimated in parallel (joblib).

    Returns
    -------
    pd.DataFrame
        Index ``(origin, horizon)``; columns ``joint`` and one per variable.
        NaN where the target date is beyond the data. ``attrs["spacing"]``
        holds the median number of periods between origins, which
        :func:`compare_log_scores` uses to account for overlapping forecasts.
    """
    origins = pd.DatetimeIndex(origins)
    pos = data.index.get_indexer(origins)
    if (pos < 0).any():
        raise ValueError("Every origin must be a date of `data`.")
    jobs = tqdm(origins, desc="Recursive forecasts", disable=not show_progress)
    if _HAS_JOBLIB and n_jobs != 1:
        parts = Parallel(n_jobs=n_jobs)(
            delayed(_score_origin)(make_model, data, o, horizons, n_draws, reps) for o in jobs)
    else:
        parts = [_score_origin(make_model, data, o, horizons, n_draws, reps) for o in jobs]
    out = pd.concat(parts)
    out.attrs["spacing"] = int(np.median(np.diff(pos))) if len(pos) > 1 else 1
    return out


def compare_log_scores(scores: Dict[str, pd.DataFrame], benchmark: str,
                       nw_lags: Optional[int] = None) -> pd.DataFrame:
    """Average log scores against a benchmark, with Amisano-Giacomini tests.

    For every model, horizon and variable: the average log score of the model
    and of the benchmark over the common origins, their difference (positive
    favours the model) and the t statistic of the difference with a
    Newey-West variance (Amisano & Giacomini, 2007). The default lag
    truncation covers the overlap of ``h``-step forecasts made from origins
    ``spacing`` periods apart: ``ceil(h / spacing) - 1``.

    Parameters
    ----------
    scores : dict
        ``{label: frame}`` from :func:`recursive_log_scores`.
    benchmark : str
        Label of the reference model.
    nw_lags : int, optional
        Overrides the Newey-West truncation for every horizon.

    Returns
    -------
    pd.DataFrame
        Index ``(model, horizon, variable)``; columns ``model``, ``benchmark``,
        ``difference``, ``t``, ``p-value`` (two-sided) and ``n``.
    """
    base = scores[benchmark]
    spacing = max(1, int(base.attrs.get("spacing", 1)))
    rows = {}
    for label, frame in scores.items():
        if label == benchmark:
            continue
        common = frame.index.intersection(base.index)
        for h in sorted(set(common.get_level_values("horizon"))):
            at = common[common.get_level_values("horizon") == h]
            lags = nw_lags if nw_lags is not None else max(0, math.ceil(h / spacing) - 1)
            for var in frame.columns:
                pair = pd.concat([frame.loc[at, var], base.loc[at, var]], axis=1).dropna()
                diff = (pair.iloc[:, 0] - pair.iloc[:, 1]).to_numpy()
                n = diff.size
                if n < 2:
                    continue
                u = diff - diff.mean()
                lrv = u @ u / n
                for lag in range(1, min(lags, n - 1) + 1):
                    lrv += 2.0 * (1.0 - lag / (lags + 1)) * (u[lag:] @ u[:-lag]) / n
                t = diff.mean() / np.sqrt(lrv / n) if lrv > 0 else np.nan
                rows[(label, h, var)] = {
                    "model": pair.iloc[:, 0].mean(), "benchmark": pair.iloc[:, 1].mean(),
                    "difference": diff.mean(), "t": t, "p-value": 2.0 * norm.sf(abs(t)), "n": n,
                }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.names = ["model", "horizon", "variable"]
    return out
