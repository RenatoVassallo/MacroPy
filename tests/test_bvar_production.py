"""
Production-readiness tests for BayesianVAR: posterior recovery on synthetic
data, seed reproducibility, dict-based conditional forecasts, the
Lenza-Primiceri COVID volatility treatment, tidy forecast frames, and the
analytic Normal-Wishart marginal likelihood (validated by Monte Carlo).

Run with: pytest tests/test_bvar_production.py
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import invwishart

from MacroPy import BayesianVAR
from MacroPy.priors import nw_log_marginal_likelihood

LOOSE_PRIOR = {"mn_mean": 0.0, "lamda1": 5.0, "lamda2": 1.0, "lamda3": 1.0,
               "lamda4": 1e5}


def simulate_var(T=400, seed=0, start="2000-01-01", covid_scale=None,
                 covid_start=None, covid_len=0):
    """
    Simulate a stable trivariate VAR(1) with intercept.

    Returns (df, A1, c, Sigma). If `covid_scale` is given, residuals in the
    `covid_len` observations starting at `covid_start` are multiplied by it.
    """
    rng = np.random.default_rng(seed)
    A1 = np.array([[0.5, 0.1, 0.0],
                   [0.0, 0.4, 0.15],
                   [0.1, 0.0, 0.3]])
    c = np.array([0.3, -0.2, 0.1])
    Sigma = np.array([[1.0, 0.3, 0.1],
                      [0.3, 1.2, 0.2],
                      [0.1, 0.2, 0.8]])
    L = np.linalg.cholesky(Sigma)

    dates = pd.date_range(start, periods=T, freq="MS")
    scale = np.ones(T)
    if covid_scale is not None:
        i0 = dates.get_loc(pd.Timestamp(covid_start))
        scale[i0:i0 + covid_len] = covid_scale

    burn = 100
    y = np.zeros((T + burn, 3))
    eps = rng.standard_normal((T + burn, 3)) @ L.T
    eps[burn:] *= scale[:, None]
    for t in range(1, T + burn):
        y[t] = c + A1 @ y[t - 1] + eps[t]
    df = pd.DataFrame(y[burn:], index=dates, columns=["y1", "y2", "y3"])
    return df, A1, c, Sigma


def lag_matrix(model):
    """Posterior-mean VAR(1) companion block, oriented as y_t = A1 y_{t-1}."""
    B = model.reshape_beta(model.beta_draws.mean(axis=0),
                           model.ncoeff_eq, model.n_endo)
    return B[: model.n_endo, :].T


# ---------------------------------------------------------------------------
# Reference ports of the Canova-Ferroni BVAR_ toolbox (bvartools/forecasts.m
# and bvartools/cforecasts.m), used as an external check of the forecast code.
# ---------------------------------------------------------------------------

def canova_forecasts(initval, xdata, Phi, Sigma, fhor, lags, EPS):
    ny = Sigma.shape[0]
    out = np.zeros((fhor, ny))
    ld = initval.copy()                      # rows oldest -> newest
    for t in range(fhor):
        X = np.concatenate([ld[::-1].ravel(), xdata[t]])
        y = X @ Phi + EPS[t]
        ld = np.vstack([ld[1:], y])
        out[t] = y
    return out


def canova_cforecast(endo_path, endo_index, initval, xdata, Phi, Sigma, lags,
                     z=None):
    """Waggoner-Zha conditional forecast; z is the V2*randn null-space draw."""
    fhor, ncond = endo_path.shape
    ny = Sigma.shape[0]
    Nres = ncond * fhor
    C = np.linalg.cholesky(Sigma)
    F = np.zeros((ny * lags, ny * lags))
    F[:ny, :] = Phi[: ny * lags].T
    if lags > 1:
        F[ny:, : ny * (lags - 1)] = np.eye(ny * (lags - 1))

    no_shock = canova_forecasts(initval, xdata, Phi, Sigma, fhor, lags,
                                np.zeros((fhor, ny)))
    err = (endo_path - no_shock[:, endo_index]).reshape(-1)

    R = np.zeros((ny * fhor, ny * fhor))
    tmp0 = np.zeros((ny, 0))
    rows = []
    for ff in range(1, fhor + 1):
        tmp = np.linalg.matrix_power(F, ff - 1)[:ny, :ny] @ C
        tmp0 = np.hstack([tmp, tmp0])
        R[(ff - 1) * ny: ff * ny, : tmp0.shape[1]] = tmp0
        rows.append((ff - 1) * ny + np.asarray(endo_index))
    Rt = R[np.concatenate(rows)]

    U, D, Vt = np.linalg.svd(Rt)
    V = Vt.T
    eps = V[:, :Nres] @ np.diag(1.0 / D[:Nres]) @ U.T @ err
    if z is not None:
        eps = eps + V[:, Nres:] @ z
    EPS = eps.reshape(fhor, ny) @ C.T
    return canova_forecasts(initval, xdata, Phi, Sigma, fhor, lags, EPS)


def test_posterior_recovery_synthetic_var():
    df, A1, c, Sigma = simulate_var(T=400, seed=1)
    m = BayesianVAR(df, lags=1, prior_type=2, prior_params=LOOSE_PRIOR,
                    post_draws=600, burnin=0.5, seed=11)
    m.sample_posterior()

    assert np.max(np.abs(lag_matrix(m) - A1)) < 0.12
    Sigma_hat = m.Sigma_draws.mean(axis=0)
    assert np.max(np.abs(Sigma_hat - Sigma)) < 0.35


def test_seed_reproducibility():
    df, *_ = simulate_var(T=200, seed=2)

    def run(seed):
        m = BayesianVAR(df, lags=1, prior_type=2, prior_params=LOOSE_PRIOR,
                        post_draws=200, burnin=0.5, seed=seed)
        m.sample_posterior()
        fr = m.forecast_frame(fhor=6)
        cfr = m.conditional_forecast_frame({"y1": [0.3, 0.2]}, fhor=6)
        return m, fr, cfr

    m_a, fr_a, cfr_a = run(7)
    m_b, fr_b, cfr_b = run(7)
    assert np.array_equal(m_a.beta_draws, m_b.beta_draws)
    assert np.array_equal(m_a.Sigma_draws, m_b.Sigma_draws)
    assert np.array_equal(m_a.forecasts, m_b.forecasts)
    pd.testing.assert_frame_equal(fr_a, fr_b)
    pd.testing.assert_frame_equal(cfr_a, cfr_b)   # WZ null-space draws seeded too

    # Idempotence: re-running the same method on the same model matches too.
    beta_first = m_a.beta_draws.copy()
    m_a.sample_posterior()
    assert np.array_equal(beta_first, m_a.beta_draws)

    m_c, _, _ = run(8)
    assert not np.array_equal(m_a.beta_draws, m_c.beta_draws)


def test_forecasts_match_canova_reference():
    """Draw-by-draw equivalence with the Canova-Ferroni BVAR_ algorithms."""
    df, *_ = simulate_var(T=250, seed=3)
    fhor, lags, n = 6, 2, 3
    m = BayesianVAR(df, lags=lags, prior_type=2, prior_params=LOOSE_PRIOR,
                    post_draws=150, burnin=0.5, hor=fhor, seed=21)
    m.sample_posterior()
    initval = m.yy[-lags:, :]
    xdata = np.ones((fhor, 1))
    path = np.linspace(0.5, 0.0, fhor)
    cond = np.full((fhor, n), np.nan)
    cond[:, 0] = path

    # Unconditional no-shock paths must equal frcst_no_shock exactly.
    m.forecast(fhor=fhor, plot_forecast=False)
    for i in range(len(m.beta_draws)):
        B = m.reshape_beta(m.beta_draws[i], m.ncoeff_eq, n)
        ref = canova_forecasts(initval, xdata, B, m.Sigma_draws[i], fhor, lags,
                               np.zeros((fhor, n)))
        assert np.allclose(m.mean_forecasts[i], ref, atol=1e-10)

    # Conditional mean paths must equal the Waggoner-Zha minimum-norm solution.
    m.conditional_forecast(cond, fhor=fhor, plot_forecast=False,
                           shock_uncertainty=False)
    mean_paths = m.cond_forecasts.copy()
    for i in range(len(m.beta_draws)):
        B = m.reshape_beta(m.beta_draws[i], m.ncoeff_eq, n)
        ref = canova_cforecast(path[:, None], [0], initval, xdata, B,
                               m.Sigma_draws[i], lags)
        assert np.allclose(mean_paths[i], ref, atol=1e-8)

    # Full WZ distribution: conditions still exact, unconstrained variables
    # strictly more dispersed than under parameter uncertainty alone.
    m.conditional_forecast(cond, fhor=fhor, plot_forecast=False)
    assert np.allclose(m.cond_forecasts[:, :, 0], path[None, :], atol=1e-8)
    assert m.cond_forecasts[:, -1, 1].std() > 1.5 * mean_paths[:, -1, 1].std()

    # The IRF display convention must not affect the conditional distribution.
    m0 = BayesianVAR(df, lags=lags, prior_type=2, prior_params=LOOSE_PRIOR,
                     post_draws=150, burnin=0.5, hor=fhor, irf_1std=0, seed=21)
    m0.sample_posterior()
    m0.conditional_forecast(cond, fhor=fhor, plot_forecast=False,
                            shock_uncertainty=False)
    assert np.allclose(m0.cond_forecasts, mean_paths, atol=1e-8)


def test_conditional_forecast_call_order_independent():
    """
    Regression: conditional_forecast must not depend on whether forecast() ran
    first. The constructor pre-allocates `mean_forecasts` as zeros with shape
    (n_draws, fhor, n_endo); the old shape-based staleness check mistook that
    for a valid baseline whenever the requested fhor equaled the constructor's,
    so the solver conditioned against zeros and produced explosive paths.
    """
    df, *_ = simulate_var(T=250, seed=3)
    fhor = 6
    kw = dict(lags=2, prior_type=2, prior_params=LOOSE_PRIOR, post_draws=200,
              burnin=0.5, fhor=fhor, hor=fhor, seed=13)
    cond = {"y3": [1.0] * 4}

    # Path A: conditional_forecast directly after sampling (fhor == self.fhor).
    m_a = BayesianVAR(df, **kw)
    m_a.sample_posterior()
    cf_a, _ = m_a.conditional_forecast(cond, fhor=fhor, plot_forecast=False)

    # Path B: identical model, but an unconditional forecast runs in between.
    m_b = BayesianVAR(df, **kw)
    m_b.sample_posterior()
    m_b.forecast(fhor=fhor, plot_forecast=False)
    cf_b, _ = m_b.conditional_forecast(cond, fhor=fhor, plot_forecast=False)

    assert np.allclose(cf_a, cf_b, atol=1e-10)

    # Falsification of the baseline: conditioning a variable on its own
    # unconditional mean (no-shock) path must leave the others nearly
    # unchanged. Null-space draws are off to isolate the systematic effect
    # from Monte-Carlo noise; what remains is the small Jensen-type term from
    # imposing a common path draw by draw.
    m_b.forecast(fhor=fhor, plot_forecast=False)
    unc_det = m_b.mean_forecasts.mean(axis=0)         # (fhor, n)
    own_path = list(unc_det[:4, 2])
    cf_own, _ = m_b.conditional_forecast({"y3": own_path}, fhor=fhor,
                                         plot_forecast=False,
                                         shock_uncertainty=False)
    gap = np.abs(cf_own.mean(axis=0)[:, :2] - unc_det[:, :2]).max()
    disp = m_b.forecasts.std(axis=0)[:, :2].max()
    assert gap < 0.25 * disp

    # Calling before sample_posterior must fail loudly, not condition on zeros.
    m_c = BayesianVAR(df, **kw)
    with pytest.raises(RuntimeError, match="sample_posterior"):
        m_c.conditional_forecast(cond, fhor=fhor, plot_forecast=False)
    with pytest.raises(RuntimeError, match="sample_posterior"):
        m_c.forecast(fhor=fhor, plot_forecast=False)


def test_conditional_forecast_dict_conditions():
    df, *_ = simulate_var(T=250, seed=3)
    m = BayesianVAR(df, lags=1, prior_type=1, prior_params=LOOSE_PRIOR,
                    post_draws=200, burnin=0.5, hor=10, seed=5)
    m.sample_posterior()

    path = [0.5, 0.7]
    frame = m.conditional_forecast_frame({"y1": path}, fhor=4)

    # The imposed path must hold exactly, draw by draw.
    assert np.allclose(m.cond_forecasts[:, 0, 0], 0.5, atol=1e-6)
    assert np.allclose(m.cond_forecasts[:, 1, 0], 0.7, atol=1e-6)
    # Later horizons are unconstrained: draws should actually disperse.
    assert np.std(m.cond_forecasts[:, 3, 0]) > 1e-3

    assert frame.loc[("y1", 1), "median"] == pytest.approx(0.5, abs=1e-6)
    assert frame.loc[("y1", 2), "mean"] == pytest.approx(0.7, abs=1e-6)

    # {horizon: value} form and unknown-variable validation.
    m.conditional_forecast({"y2": {3: -0.25}}, fhor=4, plot_forecast=False)
    assert np.allclose(m.cond_forecasts[:, 2, 1], -0.25, atol=1e-6)
    with pytest.raises(ValueError, match="Unknown variable"):
        m.conditional_forecast({"nope": [1.0]}, fhor=4, plot_forecast=False)


def test_covid_lp_absorbs_outliers():
    common = dict(T=252, seed=4, start="2002-01-01")
    df_dirty, A1, *_ = simulate_var(covid_scale=15.0, covid_start="2020-03-01",
                                    covid_len=16, **common)
    df_clean = df_dirty.loc[: "2019-12-01"]

    kw = dict(lags=1, prior_type=1, prior_params=LOOSE_PRIOR,
              post_draws=300, burnin=0.5, seed=9)
    m_clean = BayesianVAR(df_clean, **kw)
    m_naive = BayesianVAR(df_dirty, **kw)
    m_lp = BayesianVAR(df_dirty, covid_window=("2020-03", "2021-06"),
                       covid_mode="lenza-primiceri", **kw)
    for m in (m_clean, m_naive, m_lp):
        m.sample_posterior()

    d_clean = np.max(np.abs(lag_matrix(m_clean) - A1))
    d_naive = np.max(np.abs(lag_matrix(m_naive) - A1))
    d_lp = np.max(np.abs(lag_matrix(m_lp) - A1))

    # The scaling must absorb the outliers: close to the clean-sample fit and
    # strictly better than ignoring the window.
    assert d_lp < d_naive
    assert d_lp < d_clean + 0.05

    # Estimated scales should flag the injected volatility (x15).
    assert m_lp.covid_scales is not None and len(m_lp.covid_scales) == 16
    assert m_lp.covid_scales.max() > 5.0

    # Sigma must keep its ordinary-times scale, not the outlier-inflated one.
    assert np.max(np.diag(m_lp.Sigma_ols)) < 3.0
    assert np.max(np.diag(m_naive.Sigma_ols)) > 10.0

    # Fixed scales and the dummies fallback also run end to end.
    m_fix = BayesianVAR(df_dirty, covid_window=("2020-03", "2021-06"),
                        covid_mode="lenza-primiceri", covid_scales=15.0, **kw)
    m_fix.sample_posterior()
    assert np.max(np.abs(lag_matrix(m_fix) - A1)) < d_naive

    m_dum = BayesianVAR(df_dirty, covid_window=("2020-03", "2021-06"),
                        covid_mode="dummies", **kw)
    assert sum(name.startswith("covid_") for name in m_dum.exog_names) == 16
    m_dum.sample_posterior()
    frame = m_dum.forecast_frame(fhor=4)
    assert np.isfinite(frame[["mean", "median"]].to_numpy()).all()


def test_forecast_frame_layout():
    df, *_ = simulate_var(T=200, seed=6)
    m = BayesianVAR(df, lags=2, prior_type=1, prior_params=LOOSE_PRIOR,
                    post_draws=200, burnin=0.5, seed=3)
    m.sample_posterior()
    fhor = 8
    frame = m.forecast_frame(fhor=fhor, quantiles=(0.05, 0.16, 0.84, 0.95))

    assert list(frame.columns) == ["date", "mean", "median", "q05", "q16",
                                   "q84", "q95"]
    assert frame.index.names == ["variable", "horizon"]
    assert len(frame) == 3 * fhor
    assert list(frame.loc["y1"].index) == list(range(1, fhor + 1))

    q = frame[["q05", "q16", "median", "q84", "q95"]].to_numpy()
    assert (np.diff(q, axis=1) >= -1e-12).all()

    step = pd.tseries.frequencies.to_offset("MS")
    assert frame.loc[("y2", 1), "date"] == df.index[-1] + step
    assert frame.loc[("y2", fhor), "date"] == df.index[-1] + step * fhor


def test_nw_logml_matches_monte_carlo():
    """Validate the analytic marginal likelihood against brute-force MC."""
    rng = np.random.default_rng(12)
    T, n, k = 8, 2, 3
    X = rng.standard_normal((T, k))
    Y = rng.standard_normal((T, n))
    moments = {
        "B0": np.vstack([np.eye(n) * 0.5, np.zeros((k - n, n))]),
        "omega_diag": np.array([0.3, 0.6, 2.0]),
        "S0": np.diag([1.0, 0.6]),
        "nu0": n + 2,
    }
    analytic = nw_log_marginal_likelihood(Y, X, moments)

    M = 60000
    Sig = invwishart.rvs(df=moments["nu0"], scale=moments["S0"], size=M,
                         random_state=rng)
    Ls = np.linalg.cholesky(Sig)                        # (M, n, n)
    Lo = np.sqrt(moments["omega_diag"])[:, None]        # (k, 1)
    Z = rng.standard_normal((M, k, n))
    B = moments["B0"][None] + Lo[None] * Z @ np.transpose(Ls, (0, 2, 1))
    E = Y[None] - X[None] @ B                           # (M, T, n)

    Sig_inv = np.linalg.inv(Sig)
    quad = np.einsum("mtn,mnp,mtp->m", E, Sig_inv, E)
    _, logdet = np.linalg.slogdet(Sig)
    loglik = -0.5 * T * n * np.log(2 * np.pi) - 0.5 * T * logdet - 0.5 * quad

    mc = np.logaddexp.reduce(loglik) - np.log(M)
    assert mc == pytest.approx(analytic, abs=0.25)


def test_covid_lp_logml_prefers_true_scales():
    """The LP objective (with Jacobian) must rank scaled windows above s=1."""
    df, *_ = simulate_var(T=252, seed=10, start="2002-01-01", covid_scale=15.0,
                          covid_start="2020-03-01", covid_len=16)
    m = BayesianVAR(df, lags=1, prior_type=1, prior_params=LOOSE_PRIOR,
                    covid_window=("2020-03", "2021-06"),
                    covid_mode="lenza-primiceri", covid_scales=15.0,
                    post_draws=10, seed=1)
    with_scales = m.log_marginal_likelihood()
    m_flat = BayesianVAR(df, lags=1, prior_type=1, prior_params=LOOSE_PRIOR,
                         post_draws=10, seed=1)
    without = m_flat.log_marginal_likelihood()
    assert with_scales > without


def test_select_hyperparameters_glp():
    df, *_ = simulate_var(T=300, seed=8)
    m = BayesianVAR(df, lags=2, prior_type=2, prior_params=dict(LOOSE_PRIOR),
                    post_draws=100, burnin=0.5, seed=2)
    res = m.select_hyperparameters(verbose=False)

    assert res["success"]
    assert res["log_ml"] >= res["initial_log_ml"] - 1e-6
    assert m.prior_params["lamda1"] == pytest.approx(res["params"]["lamda1"])

    # lamda2 has no analytic marginal likelihood: must refuse, not silently fit.
    with pytest.raises(ValueError, match="lamda2"):
        m.select_hyperparameters(select=("lamda1", "lamda2"))


# ---------------------------------------------------------------------------
# Sum-of-coefficients (SOC) and dummy-initial-observation (DIO) priors
# (Doan-Litterman-Sims 1984; Sims 1993; BGR 2010; GLP 2015)
# ---------------------------------------------------------------------------

def test_soc_tight_imposes_unit_sum():
    """A. mu -> 0 shrinks A_1 + ... + A_p toward the identity."""
    df, *_ = simulate_var(T=300, seed=5)
    df = df + 5.0                       # nonzero levels so ybar0 anchors bite
    n, lags = 3, 2
    m = BayesianVAR(df, lags=lags, prior_type=2,
                    prior_params=dict(LOOSE_PRIOR, soc=0.05),
                    post_draws=300, burnin=0.5, seed=6)
    m.sample_posterior()
    B = m.reshape_beta(m.beta_draws.mean(axis=0), m.ncoeff_eq, n)
    sumA = sum(B[l * n:(l + 1) * n, :].T for l in range(lags))
    assert np.max(np.abs(sumA - np.eye(n))) < 0.10


def test_soc_dio_anchor_to_current_level():
    """B. With a mean that falls late in the sample (China-GDP style), SOC/DIO
    anchor long-horizon forecasts to the last observed level while the plain
    model reverts toward the estimation-sample mean.

    The GLP-default tightness (soc = dio = 1) improves the anchoring
    directionally; tight dummies (0.1) impose it almost fully. The default
    cannot flip the path by itself here because with levels around 8 the
    data's information about the coefficient sum outweighs a mu = 1 dummy by
    roughly T / p^2.
    """
    rng = np.random.default_rng(11)
    T, n_fall = 150, 25
    local_mean = np.concatenate([np.full(T - n_fall, 8.0),
                                 np.linspace(8.0, 3.0, n_fall)])
    u = np.zeros((T, 2))
    for t in range(1, T):
        u[t] = 0.5 * u[t - 1] + rng.standard_normal(2) * 0.8
    y = local_mean[:, None] + u + np.array([0.0, 1.0])
    df = pd.DataFrame(y, index=pd.date_range("2005-01-01", periods=T, freq="MS"),
                      columns=["y1", "y2"])
    fhor = 20
    pp = {"mn_mean": 0.0, "lamda1": 0.4, "lamda2": 1.0, "lamda3": 1.0,
          "lamda4": 1e5}
    kw = dict(lags=2, prior_type=2, post_draws=300, burnin=0.5, seed=12)
    y_last = df.iloc[-1].to_numpy()

    def endpoint_dist(params):
        m = BayesianVAR(df, prior_params=params, **kw)
        m.sample_posterior()
        m.forecast(fhor=fhor, plot_forecast=False)
        fc = m.mean_forecasts.mean(axis=0)[-1]
        return fc, np.abs(fc - y_last).max()

    fc_plain, d_plain = endpoint_dist(pp)
    _, d_glp = endpoint_dist(dict(pp, soc=1.0, dio=1.0))
    _, d_tight = endpoint_dist(dict(pp, soc=0.1, dio=0.1))

    assert (fc_plain - y_last).min() > 1.0   # plain reverts up toward the mean
    assert d_glp < d_plain                   # GLP default improves the anchor
    assert d_tight < 0.4 * d_plain           # tight dummies anchor to level


def test_soc_dio_marginal_likelihood_correction():
    """C. Off-threshold reproduces the no-dummy ML; finite values change it;
    selection over soc works."""
    df, *_ = simulate_var(T=300, seed=8)
    m = BayesianVAR(df, lags=2, prior_type=2, prior_params=dict(LOOSE_PRIOR),
                    post_draws=50, seed=2)
    base = m.log_marginal_likelihood()
    off = m.log_marginal_likelihood(dict(LOOSE_PRIOR, soc=1e5, dio=1e5))
    assert off == pytest.approx(base, abs=1e-6)
    on = m.log_marginal_likelihood(dict(LOOSE_PRIOR, soc=1.0, dio=1.0))
    assert abs(on - base) > 1.0

    res = m.select_hyperparameters(select=("lamda1", "soc"), verbose=False)
    assert res["success"] and np.isfinite(res["log_ml"])
    assert 0.1 <= res["params"]["soc"] <= 10.0
    assert m.prior_params["soc"] == pytest.approx(res["params"]["soc"])
    assert m.n_soc_dio > 0                       # fit matrices rebuilt


def test_soc_dio_backward_compatibility():
    """D. Absent keys leave everything untouched; None behaves like absent."""
    df, *_ = simulate_var(T=250, seed=3)
    kw = dict(lags=2, prior_type=2, post_draws=200, burnin=0.5, seed=42)

    m_absent = BayesianVAR(df, prior_params=dict(LOOSE_PRIOR), **kw)
    m_none = BayesianVAR(df, prior_params=dict(LOOSE_PRIOR, soc=None, dio=None),
                         **kw)
    assert m_absent.n_soc_dio == 0 and m_none.n_soc_dio == 0
    assert m_absent.yy_fit is m_absent.yy_w      # no augmentation at all
    m_absent.sample_posterior()
    m_none.sample_posterior()
    assert np.array_equal(m_absent.beta_draws, m_none.beta_draws)
    pd.testing.assert_frame_equal(m_absent.forecast_frame(fhor=6),
                                  m_none.forecast_frame(fhor=6))


def test_soc_dio_combinations():
    """E. SOC+DIO with COVID scaling, conditional frames and b_exo run clean."""
    common = dict(T=252, seed=4, start="2002-01-01")
    df, *_ = simulate_var(covid_scale=15.0, covid_start="2020-03-01",
                          covid_len=16, **common)
    pp = dict(LOOSE_PRIOR, soc=1.0, dio=1.0)
    m = BayesianVAR(df, lags=2, prior_type=2, prior_params=pp,
                    covid_window=("2020-03", "2021-06"),
                    covid_mode="lenza-primiceri",
                    post_draws=200, burnin=0.5, seed=9)
    m.sample_posterior()
    assert m.yy_fit.shape[0] == m.yy.shape[0] + 4
    frame = m.conditional_forecast_frame({"y1": [0.4, 0.3]}, fhor=6,
                                         shock_uncertainty=False)
    assert np.isfinite(frame[["mean", "median"]].to_numpy()).all()
    assert frame.loc[("y1", 1), "median"] == pytest.approx(0.4, abs=1e-6)

    block = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    m_b = BayesianVAR(df, lags=2, prior_type=2, prior_params=pp, b_exo=block,
                      post_draws=150, burnin=0.5, seed=9)
    m_b.sample_posterior()
    assert np.isfinite(m_b.beta_draws).all()


# ---------------------------------------------------------------------------
# Steady-state (mean-adjusted) prior (Villani, 2009)
# ---------------------------------------------------------------------------

def _late_decline_df(T=200, n_fall=30, hi=8.0, lo=4.0, seed=13):
    """Mean at `hi` for most of the sample, declining to `lo` at the end, so
    the sample mean (~8) overstates the terminal level (the China-GDP case)."""
    rng = np.random.default_rng(seed)
    mpath = np.concatenate([np.full(T - n_fall, hi), np.linspace(hi, lo, n_fall)])
    u = np.zeros((T, 2))
    for t in range(1, T):
        u[t] = 0.5 * u[t - 1] + rng.standard_normal(2) * 0.7
    y = mpath[:, None] + u + np.array([0.0, 1.0])
    return pd.DataFrame(y, index=pd.date_range("2004-01-01", periods=T, freq="MS"),
                        columns=["y1", "y2"])


SS_BASE = {"mn_mean": 0.0, "lamda1": 0.5, "lamda2": 1.0, "lamda3": 1.0,
           "lamda4": 1e5}


def test_ss_prior_anchors_to_judgment():
    """A. Tight ss anchors h=40 forecasts at m0; loose reverts to sample mean."""
    df = _late_decline_df()
    target = np.array([4.0, 5.0])

    def run(ss_sd):
        pp = dict(SS_BASE, ss_mean={"y1": 4.0, "y2": 5.0}, ss_sd=ss_sd)
        m = BayesianVAR(df, lags=2, prior_type=2, prior_params=pp,
                        post_draws=400, burnin=0.5, seed=5)
        m.sample_posterior()
        m.forecast(fhor=40, plot_forecast=False)
        return m

    m_tight = run(0.1)
    assert np.abs(m_tight.mu_draws.mean(axis=0) - target).max() < 0.15
    fc40 = m_tight.mean_forecasts.mean(axis=0)[-1]
    assert np.abs(fc40 - target).max() < 0.6
    # Forecasts converge to the draw's own mu by construction.
    drawwise = np.abs(m_tight.mean_forecasts[:, -1, :] - m_tight.mu_draws)
    assert np.median(drawwise) < 0.5

    m_loose = run(1e3)
    sample_mean = df.mean().to_numpy()
    assert np.abs(m_loose.mu_draws.mean(axis=0) - sample_mean).max() < 0.5
    fc40_loose = m_loose.mean_forecasts.mean(axis=0)[-1]
    assert np.abs(fc40_loose - target).max() > 1.5


def test_ss_backward_compatibility():
    """B. Without ss keys nothing changes: absent and ss_mean=None give
    bit-identical draws through untouched code paths (verified against a
    pre-change 0.1.9 snapshot during development)."""
    df, *_ = simulate_var(T=250, seed=3)
    m = BayesianVAR(df, lags=2, prior_type=2, prior_params=LOOSE_PRIOR,
                    post_draws=300, burnin=0.5, seed=42)
    assert not m.ss_active
    m.sample_posterior()

    # ss_mean=None behaves exactly like absent.
    m2 = BayesianVAR(df, lags=2, prior_type=2,
                     prior_params=dict(LOOSE_PRIOR, ss_mean=None),
                     post_draws=300, burnin=0.5, seed=42)
    assert not m2.ss_active
    m2.sample_posterior()
    assert np.array_equal(m.beta_draws, m2.beta_draws)

    # When available (development machine), also pin against the frozen
    # 0.1.9 reference draws.
    import pathlib
    ref = pathlib.Path("/private/tmp/claude-501/-Users-rvs-GitRepos-MacroPy/"
                       "fb3d4244-d0db-4ef8-a29c-92c9979bcfe6/scratchpad/ref019_beta.npy")
    if ref.exists():
        assert np.array_equal(np.load(ref), m.beta_draws)


def test_ss_conditional_forecast():
    """C. Conditions hold exactly under ss; paths converge toward mu beyond."""
    df = _late_decline_df()
    pp = dict(SS_BASE, ss_mean={"y1": 4.0, "y2": 5.0}, ss_sd=0.1)
    m = BayesianVAR(df, lags=2, prior_type=2, prior_params=pp,
                    post_draws=400, burnin=0.5, seed=5)
    m.sample_posterior()
    cf, _ = m.conditional_forecast({"y1": [4.5, 4.8]}, fhor=40,
                                   plot_forecast=False)
    assert np.allclose(cf[:, 0, 0], 4.5, atol=1e-6)
    assert np.allclose(cf[:, 1, 0], 4.8, atol=1e-6)
    mu_post = m.mu_draws.mean(axis=0)
    assert np.abs(cf.mean(axis=0)[-1] - mu_post).max() < 0.6

    frame = m.conditional_forecast_frame({"y1": [4.5]}, fhor=8)
    assert frame.loc[("y1", 1), "median"] == pytest.approx(4.5, abs=1e-6)


def test_ss_combinations():
    """D. ss + Lenza-Primiceri COVID and ss + soc/dio run clean."""
    common = dict(T=252, seed=4, start="2002-01-01")
    df, *_ = simulate_var(covid_scale=15.0, covid_start="2020-03-01",
                          covid_len=16, **common)
    df = df + 6.0
    pp = dict(SS_BASE, ss_mean=6.0, ss_sd=0.5)
    m = BayesianVAR(df, lags=2, prior_type=2, prior_params=pp,
                    covid_window=("2020-03", "2021-06"),
                    covid_mode="lenza-primiceri",
                    post_draws=200, burnin=0.5, seed=9)
    m.sample_posterior()
    assert np.isfinite(m.mu_draws).all() and np.isfinite(m.beta_draws).all()
    assert np.isfinite(m.forecast_frame(fhor=6)[["mean", "median"]].to_numpy()).all()

    # Deliberate combination: soc/dio (unit roots) vs ss (reversion to mu).
    m2 = BayesianVAR(df, lags=2, prior_type=1,
                     prior_params=dict(pp, soc=1.0, dio=1.0),
                     post_draws=150, burnin=0.5, seed=9)
    m2.sample_posterior()
    assert np.isfinite(m2.mu_draws).all()


def test_ss_guards():
    """E. Unsupported combinations fail loudly."""
    df = _late_decline_df(T=120)
    pp = dict(SS_BASE, ss_mean=4.0, ss_sd=0.5)

    with pytest.raises(ValueError, match="timetrend"):
        BayesianVAR(df, lags=2, prior_params=pp, timetrend=True)
    with pytest.raises(ValueError, match="exogenous"):
        exog = pd.DataFrame({"d": np.zeros(len(df))}, index=df.index)
        BayesianVAR(df, lags=2, prior_params=pp, exog=exog)
    with pytest.raises(ValueError, match="dummies"):
        BayesianVAR(df, lags=2, prior_params=pp,
                    covid_window=("2010Q1", "2010Q4"), covid_mode="dummies")
    with pytest.raises(ValueError, match="Unknown variable"):
        BayesianVAR(df, lags=2, prior_params=dict(SS_BASE, ss_mean={"nope": 1.0}))

    m = BayesianVAR(df, lags=2, prior_type=2, prior_params=pp,
                    post_draws=50, seed=1)
    with pytest.raises(NotImplementedError, match="steady-state"):
        m.select_hyperparameters()
    with pytest.raises(NotImplementedError, match="steady-state"):
        m.log_marginal_likelihood()
