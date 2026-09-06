"""
Tests for the UC trend-cycle module (MacroPy.uc).

Layout mirrors the design invariants of the MATLAB trend_cycle_toolbox this
module ports:
  A. HP filter: exact normal equations + state-space equivalence.
  B. Integrated likelihood: precision identity vs. Kalman filter agreement.
  C. MLE parameter recovery (Clark rung).
  D. Gibbs parameter recovery + seed reproducibility (Bayesian rung).
  E. Prior-calibration kappa rules on US GDP.
  F. COVID variance scales absorb an injected pandemic outlier.
  G. Output frames, comparison table, summary.
  H. Guard rails.
"""

import numpy as np
import pandas as pd
import pytest

from MacroPy.uc import (
    UCTrendCycle, compare_uc, covid_lambda_path, hp_filter,
    integrated_loglik, kalman_loglik, make_priors,
)


# ---------------------------------------------------------------------------
# Simulated UC data shared across tests
# ---------------------------------------------------------------------------

TRUE = {"phi": (1.35, -0.5), "rho_g": 0.9, "gbar": 0.75,
        "sd_c": 0.8, "sd_eta": 0.05, "sd_zeta": 0.10}


def simulate_uc(T=260, seed=0, start="1958-03-01"):
    rng = np.random.default_rng(seed)
    phi1, phi2 = TRUE["phi"]
    g = np.full(T, TRUE["gbar"])
    tau = np.zeros(T)
    c = np.zeros(T)
    tau[0] = 700.0
    for t in range(1, T):
        g[t] = (TRUE["rho_g"] * g[t - 1] + (1 - TRUE["rho_g"]) * TRUE["gbar"]
                + TRUE["sd_zeta"] * rng.standard_normal())
        tau[t] = tau[t - 1] + g[t - 1] + TRUE["sd_eta"] * rng.standard_normal()
        c[t] = phi1 * c[t - 1] + phi2 * c[t - 2] + TRUE["sd_c"] * rng.standard_normal()
    dates = pd.date_range(start, periods=T, freq="QS-DEC")
    return pd.Series(tau + c, index=dates, name="y"), tau, c


@pytest.fixture(scope="module")
def sim():
    y, tau, c = simulate_uc()
    return {"y": y, "tau": tau, "c": c}


@pytest.fixture(scope="module")
def us_gdp():
    df = pd.read_csv("datasets/USGDP_Data.csv", parse_dates=["date"],
                     index_col="date")
    return 100 * np.log(df["gdp"])


# ---------------------------------------------------------------------------
# A. HP filter
# ---------------------------------------------------------------------------

def test_hp_filter_solves_normal_equations(sim):
    from scipy.sparse import diags, identity
    y = sim["y"].to_numpy()
    T = y.shape[0]
    out = hp_filter(sim["y"], 1600.0)
    D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(T - 2, T))
    resid = (identity(T) + 1600.0 * (D.T @ D)) @ out["trend"].to_numpy() - y
    assert np.abs(resid).max() < 1e-8
    assert np.allclose(out["trend"] + out["cycle"], y)
    assert out.index.equals(sim["y"].index)


def test_hp_equals_state_space_representation(sim):
    """Rung 1's punchline: the HP filter is a UC model (Harvey-Jaeger)."""
    direct = hp_filter(sim["y"], 1600.0)["trend"].to_numpy()
    ssm = UCTrendCycle.hp(sim["y"], 1600.0)
    ssm.fit_mle()
    assert np.abs(ssm.mle_["trend"] - direct).max() < 1e-4


# ---------------------------------------------------------------------------
# B. Likelihood: precision identity vs Kalman filter
# ---------------------------------------------------------------------------

def test_precision_and_kalman_likelihoods_agree(sim):
    y = sim["y"].to_numpy()
    T = y.shape[0]
    rng = np.random.default_rng(42)
    for _ in range(3):
        q_c = np.full(T, 0.6) * np.exp(0.2 * rng.standard_normal(T))
        q_eta = np.full(T, 0.05 ** 2)
        q_zeta = np.full(T, 0.1 ** 2)
        params = {"phi": np.array([1.2, -0.4]), "rho_g": 0.85, "gbar": 0.7,
                  "q_c": q_c, "q_eta": q_eta, "q_zeta": q_zeta,
                  "tau0_mean": float(y[0]), "tau0_var": 9.0,
                  "g0_mean": 0.7, "g0_var": 0.25}
        ll_prec = integrated_loglik(y, params)
        ll_kf, ll_t = kalman_loglik(y, params)
        assert abs(ll_prec - ll_kf) < 1e-6
        assert abs(ll_t.sum() - ll_kf) < 1e-10


# ---------------------------------------------------------------------------
# C. MLE recovery (Clark rung)
# ---------------------------------------------------------------------------

def test_mle_recovers_cycle_dynamics(sim):
    uc = UCTrendCycle(sim["y"], drift="ar1")
    fit = uc.fit_mle()
    assert fit["success"]
    est = fit["params"]
    assert abs(est["phi1"] - TRUE["phi"][0]) < 0.25
    assert abs(est["phi2"] - TRUE["phi"][1]) < 0.25
    assert abs(np.sqrt(est["sig2_c"]) - TRUE["sd_c"]) < 0.35
    # the smoothed cycle tracks the true cycle
    corr = np.corrcoef(fit["cycle"], sim["c"])[0, 1]
    assert corr > 0.7
    # decomposition_frame works in MLE-only mode
    frame = uc.decomposition_frame()
    assert {"trend", "cycle", "drift"} <= set(frame.columns)


def test_clark_preset_is_random_walk_drift(sim):
    uc = UCTrendCycle.clark(sim["y"])
    assert uc.drift == "rw"
    fit = uc.fit_mle()
    assert fit["success"]
    assert "rho_g" not in fit["params"]


# ---------------------------------------------------------------------------
# D. Gibbs recovery and reproducibility
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gibbs_fit(sim):
    uc = UCTrendCycle(sim["y"], drift="ar1", post_draws=1200, burnin=0.5, seed=7)
    uc.sample_posterior()
    return uc


def test_gibbs_recovers_parameters(sim, gibbs_fit):
    d = gibbs_fit.draws
    assert abs(d["phi"][:, 0].mean() - TRUE["phi"][0]) < 0.2
    assert abs(d["phi"][:, 1].mean() - TRUE["phi"][1]) < 0.2
    assert abs(d["rho_g"].mean() - TRUE["rho_g"]) < 0.15
    assert abs(np.sqrt(d["sig2_c"].mean()) - TRUE["sd_c"]) < 0.15
    corr = np.corrcoef(np.median(d["cycle"], axis=0), sim["c"])[0, 1]
    assert corr > 0.7


def test_seed_reproducibility(sim):
    kw = dict(drift="ar1", post_draws=200, burnin=0.5, seed=11)
    d1 = UCTrendCycle(sim["y"], **kw).sample_posterior(store_loglik=False)
    d2 = UCTrendCycle(sim["y"], **kw).sample_posterior(store_loglik=False)
    assert np.array_equal(d1["trend"], d2["trend"])
    assert np.array_equal(d1["phi"], d2["phi"])
    d3 = UCTrendCycle(sim["y"], drift="ar1", post_draws=200, burnin=0.5,
                      seed=12).sample_posterior(store_loglik=False)
    assert not np.array_equal(d1["phi"], d3["phi"])


def test_thinning_subsamples_the_same_chain(sim):
    kw = dict(drift="ar1", post_draws=400, burnin=0.5, seed=11)
    uc_thin = UCTrendCycle(sim["y"], **kw)
    d_thin = uc_thin.sample_posterior(store_loglik=False, thin=4)
    assert d_thin["trend"].shape[0] == 50
    assert uc_thin.n_draws == 50
    uc_full = UCTrendCycle(sim["y"], **kw)
    d_full = uc_full.sample_posterior(store_loglik=False)
    # storage does not touch the rng stream: thinned = strided subsample
    assert np.array_equal(d_thin["trend"], d_full["trend"][::4])
    assert np.array_equal(d_thin["phi"], d_full["phi"][::4])


def test_forecast_seed_independent_of_call_order(sim, gibbs_fit):
    f1 = gibbs_fit.forecast_frame(fhor=4)
    f2 = gibbs_fit.forecast_frame(fhor=4)
    pd.testing.assert_frame_equal(f1, f2)


# ---------------------------------------------------------------------------
# E. Prior-calibration kappa rules
# ---------------------------------------------------------------------------

def test_kappa_rules_on_us_gdp(us_gdp):
    """The rules should reproduce the documented post-war US magnitude:
    prior sd(zeta) about 0.12, an order of magnitude below the generic
    default (0.5), and sd(eta) a tenth of that."""
    y = us_gdp[us_gdp.index < "2015-01-01"].to_numpy()
    pri = make_priors(y, calibrate=True)
    cal = pri["_calibration"]
    assert 0.06 < cal["sd_zeta"] < 0.18
    assert np.isclose(cal["sd_eta"], 0.1 * cal["sd_zeta"])
    # IG hyperparameters consistent with the implied means
    assert np.isclose(pri["sig2_zeta_S"] / (pri["sig2_zeta_nu"] - 1),
                      cal["sd_zeta"] ** 2)
    # uncalibrated priors keep the generic default
    pri0 = make_priors(y, calibrate=False)
    assert np.isclose(pri0["sig2_zeta_S"] / (pri0["sig2_zeta_nu"] - 1), 0.25)


def test_prior_overrides_merge_recursively(sim):
    y = sim["y"].to_numpy()
    pri = make_priors(y, covid_t0=249, calibrate=True,
                      overrides={"rho_g_mean": 0.8,
                                 "cov_c": {"ngrid": 100}})
    assert pri["rho_g_mean"] == 0.8
    assert pri["cov_c"]["ngrid"] == 100
    assert pri["cov_c"]["rho_max"] == 0.98  # untouched sibling key


# ---------------------------------------------------------------------------
# F. COVID variance scales
# ---------------------------------------------------------------------------

def test_covid_lambda_path_shape():
    lam = covid_lambda_path(12, 4, 100.0, 50.0, 5.0, 0.5)
    assert np.allclose(lam[:4], 1.0)
    assert lam[4] == 100.0 and lam[5] == 50.0
    assert np.allclose(lam[6:9], [5.0, 3.0, 2.0])
    assert np.allclose(covid_lambda_path(12, None, 100, 50, 5, 0.5), 1.0)


@pytest.fixture(scope="module")
def covid_fits():
    y_clean, tau, c = simulate_uc(seed=0)
    y = y_clean.copy()
    t0 = int(np.flatnonzero(y.index == "2020-06-01")[0])
    y.iloc[t0] -= 9.0
    y.iloc[t0 + 1] += 7.0
    y.iloc[t0 + 2] += 1.5
    kw = dict(post_draws=800, burnin=0.5, seed=3)
    plain_clean = UCTrendCycle(y_clean, **kw)
    plain_clean.sample_posterior()
    plain_dirty = UCTrendCycle(y, **kw)
    plain_dirty.sample_posterior()
    covid = UCTrendCycle(y, covid_t0="2020Q2", **kw)
    covid.sample_posterior()
    return {"t0": t0, "clean": plain_clean, "dirty": plain_dirty,
            "covid": covid, "c": c}


def test_covid_scales_absorb_outlier(covid_fits):
    t0 = covid_fits["t0"]
    lam = np.median(covid_fits["covid"].draws["lambda_cycle"], axis=0)
    assert lam[t0] > 20.0            # pandemic quarters heavily rescaled
    assert np.allclose(lam[:t0], 1.0)
    assert lam[-1] < 5.0             # the path reverts toward one


def test_covid_model_protects_pre2020_decomposition(covid_fits):
    """The scaled model on contaminated data should reproduce the plain
    model on clean data before 2020; the plain model on contaminated data
    should not (its cycle variance is inflated sample-wide)."""
    t0 = covid_fits["t0"]
    pre = slice(0, t0 - 4)
    tr_clean = np.median(covid_fits["clean"].draws["trend"], axis=0)[pre]
    tr_covid = np.median(covid_fits["covid"].draws["trend"], axis=0)[pre]
    assert np.abs(tr_covid - tr_clean).max() < 1.0
    sd_c_clean = np.sqrt(covid_fits["clean"].draws["sig2_c"].mean())
    sd_c_dirty = np.sqrt(covid_fits["dirty"].draws["sig2_c"].mean())
    sd_c_covid = np.sqrt(covid_fits["covid"].draws["sig2_c"].mean())
    assert sd_c_dirty > 1.15 * sd_c_clean     # contamination inflates variance
    assert abs(sd_c_covid - sd_c_clean) < 0.15  # scales absorb it


def test_waic_prefers_covid_model_on_contaminated_data(covid_fits):
    tab = compare_uc({"plain": covid_fits["dirty"], "covid": covid_fits["covid"]})
    assert tab.loc["covid", "waic"] < tab.loc["plain", "waic"]
    assert tab.loc["covid", "best_waic"] == "*"
    assert (tab["pwaic"] > 0).all()


# ---------------------------------------------------------------------------
# F2. Conditional forecasts (Waggoner-Zha style)
# ---------------------------------------------------------------------------

def _stacked_growth(model, cf, year):
    """Average YoY growth of `year` per draw, combining data and forecasts."""
    n = cf["y"].shape[0]
    yy = np.concatenate([np.tile(model.y, (n, 1)), cf["y"]], axis=1)
    allp = pd.PeriodIndex(list(model.dates.to_period("Q"))
                          + list(pd.DatetimeIndex(cf["dates"]).to_period("Q")))
    idx = [int(np.flatnonzero(allp == pd.Period(f"{year}Q{q}"))[0])
           for q in range(1, 5)]
    return np.mean([yy[:, i] - yy[:, i - 4] for i in idx], axis=0)


def test_conditional_forecast_annual_growth(gibbs_fit):
    """Annual growth restrictions hold exactly along every draw, with the
    remaining predictive uncertainty preserved."""
    cf = gibbs_fit.conditional_forecast(fhor=8,
                                        annual_growth={2023: 3.0, 2024: 2.0})
    assert np.abs(_stacked_growth(gibbs_fit, cf, 2023) - 3.0).max() < 1e-8
    assert np.abs(_stacked_growth(gibbs_fit, cf, 2024) - 2.0).max() < 1e-8
    assert cf["y"].std(axis=0).min() > 0.05          # WZ null-space noise
    assert np.abs(cf["y"] - cf["trend"] - cf["cycle"]).max() < 1e-8
    # reproducible under the per-stage seed
    cf2 = gibbs_fit.conditional_forecast(fhor=8,
                                         annual_growth={2023: 3.0, 2024: 2.0})
    assert np.array_equal(cf["y"], cf2["y"])


def test_conditional_forecast_hard_pin(gibbs_fit):
    pin = float(gibbs_fit.y[-1] + 2.0)
    cf = gibbs_fit.conditional_forecast(fhor=4, conditions={"2023Q2": pin})
    assert np.abs(cf["y"][:, 1] - pin).max() < 1e-8
    assert cf["y"][:, 3].std() > 0.05                # unpinned horizons vary
    mn = gibbs_fit.conditional_forecast(fhor=4, conditions={"2023Q2": pin},
                                        shock_uncertainty=False)
    assert mn["y"][:, 3].std() < cf["y"][:, 3].std()  # min-norm narrower
    frame = gibbs_fit.conditional_forecast_frame(fhor=4,
                                                 conditions={"2023Q2": pin})
    assert frame.shape[0] == 8
    assert {"mean", "median"} <= set(frame.columns)


def test_conditional_forecast_anchor_states(gibbs_fit):
    """Anchoring on the median final state removes the trend/cycle split
    uncertainty at T: unobserved-component fans start narrow and widen."""
    kw = dict(fhor=8, annual_growth={2023: 3.0})
    cf_full = gibbs_fit.conditional_forecast(**kw)
    cf_anc = gibbs_fit.conditional_forecast(anchor_states=True, **kw)
    assert cf_anc["cycle"][:, 0].std() < 0.9 * cf_full["cycle"][:, 0].std()
    assert cf_anc["cycle"][:, 7].std() > 1.2 * cf_anc["cycle"][:, 0].std()
    assert cf_anc["trend"][:, 7].std() > cf_anc["trend"][:, 0].std()
    # the growth restriction still holds exactly along every anchored path
    assert np.abs(_stacked_growth(gibbs_fit, cf_anc, 2023) - 3.0).max() < 1e-8
    # adding-up at the anchor: y = trend + cycle path by path
    assert np.abs(cf_anc["y"] - cf_anc["trend"] - cf_anc["cycle"]).max() < 1e-8


def test_unconditional_draws_via_conditional_machinery(gibbs_fit):
    """With no restrictions, conditional_forecast returns unconditional
    predictive draws whose quantiles match forecast_frame."""
    cf = gibbs_fit.conditional_forecast(fhor=6)
    frame = gibbs_fit.forecast_frame(fhor=6)
    med_frame = frame.loc["y"]["median"].to_numpy()
    med_cf = np.median(cf["y"], axis=0)
    assert np.abs(med_cf - med_frame).max() < 0.5     # same distribution, MC error
    assert cf["y"].std(axis=0).min() > 0.05


def test_conditional_forecast_guards(gibbs_fit):
    with pytest.raises(ValueError):
        gibbs_fit.conditional_forecast(fhor=4, conditions={"2010Q1": 1.0})
    with pytest.raises(ValueError):
        gibbs_fit.conditional_forecast(fhor=4, annual_growth={2020: 2.0})
    with pytest.raises(ValueError):
        gibbs_fit.conditional_forecast(fhor=2, annual_growth={2024: 2.0})


# ---------------------------------------------------------------------------
# G. Frames, comparison, summary
# ---------------------------------------------------------------------------

def test_decomposition_frame_bayesian(gibbs_fit):
    frame = gibbs_fit.decomposition_frame(cred=0.9)
    assert frame.shape == (gibbs_fit.T, 9)
    assert (frame["trend_lower"] <= frame["trend"]).all()
    assert (frame["trend"] <= frame["trend_upper"]).all()
    assert frame.index.equals(gibbs_fit.dates)


def test_band_width_scales_with_coverage(gibbs_fit):
    """68% is the default band; wider coverage nests it, and the cycle band
    has the same width as the trend band because c_t = y_t - tau_t."""
    inner = gibbs_fit.decomposition_frame()
    outer = gibbs_fit.decomposition_frame(cred=0.90)
    w_in = (inner["trend_upper"] - inner["trend_lower"]).to_numpy()
    w_out = (outer["trend_upper"] - outer["trend_lower"]).to_numpy()
    assert (w_out > w_in).all()
    # ratio of normal quantiles, 1.645 / 0.9945 = 1.85, with sampling slack
    assert 1.5 < np.median(w_out / w_in) < 2.2
    w_cycle = (inner["cycle_upper"] - inner["cycle_lower"]).to_numpy()
    assert np.allclose(w_cycle, w_in)


def test_forecast_frame_layout(gibbs_fit):
    f = gibbs_fit.forecast_frame(fhor=6, quantiles=(0.16, 0.84))
    assert list(f.index.names) == ["variable", "horizon"]
    assert set(f.index.get_level_values("variable")) == {"y", "trend"}
    assert f.shape[0] == 12
    assert {"date", "mean", "median", "q16", "q84"} <= set(f.columns)
    assert (f["q16"] <= f["median"]).all() and (f["median"] <= f["q84"]).all()
    # forecast dates continue the sample at the same frequency
    assert f.loc[("y", 1), "date"] == gibbs_fit.dates[-1] + pd.DateOffset(months=3)


def test_model_summary_renders(gibbs_fit):
    from MacroPy.summary import generate_uc_summary
    md = generate_uc_summary(gibbs_fit)
    text = md.data
    assert "Unobserved-Components" in text
    assert "Gibbs Sampling" in text


# ---------------------------------------------------------------------------
# H. Guard rails
# ---------------------------------------------------------------------------

def test_guards(sim):
    with pytest.raises(ValueError):
        UCTrendCycle(pd.DataFrame({"a": sim["y"], "b": sim["y"]}))
    with pytest.raises(ValueError):
        UCTrendCycle(sim["y"], drift="rw2")
    with pytest.raises(ValueError):
        UCTrendCycle(sim["y"], covid_t0="1900Q1")
    with pytest.raises(ValueError):
        UCTrendCycle(sim["y"], covid_t0="2020Q2", covid_components=("cycle", "x"))
    uc = UCTrendCycle(sim["y"])
    with pytest.raises(RuntimeError):
        uc.decomposition_frame()
    with pytest.raises(RuntimeError):
        uc.forecast_frame()
    with pytest.raises(RuntimeError):
        uc.waic()
