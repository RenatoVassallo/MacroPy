"""Tests for ThresholdVAR (Alessandri & Mumtaz, 2017) and the threshold-model comparison tools."""
import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import invwishart, matrix_normal, multivariate_normal

from MacroPy import (
    ThresholdVAR, ThresholdVARSV, compare_log_scores, compare_threshold_models,
    plot_irf_comparison, plot_irfs, plot_log_scores, plot_regime_comparison, plot_regimes,
    plot_volatility_comparison, recursive_log_scores,
)
from MacroPy import tvar as tv

matplotlib.use("Agg")

S1 = np.array([[1.0, 0.3], [0.3, 0.5]])
S2 = np.array([[4.0, -0.5], [-0.5, 1.0]])


def _data(T=400, seed=0, d=2, tar=0.0):
    """Two-regime VAR(1); regime 1 when the second variable, ``d`` periods back, is <= ``tar``."""
    rng = np.random.default_rng(seed)
    A1, c1 = np.array([[0.5, 0.1], [0.0, 0.8]]), np.array([0.5, -0.3])
    A2, c2 = np.array([[0.2, -0.3], [0.1, 0.6]]), np.array([-0.5, 0.4])
    C1, C2 = np.linalg.cholesky(S1), np.linalg.cholesky(S2)
    y = np.zeros((T + 50, 2))
    low = np.zeros(T + 50, dtype=bool)
    for t in range(d, T + 50):
        low[t] = y[t - d, 1] <= tar
        if low[t]:
            y[t] = c1 + A1 @ y[t - 1] + C1 @ rng.standard_normal(2)
        else:
            y[t] = c2 + A2 @ y[t - 1] + C2 @ rng.standard_normal(2)
    idx = pd.date_range("1980-01-01", periods=T, freq="MS")
    return pd.DataFrame(y[50:], index=idx, columns=["a", "f"]), low[50:]


@pytest.fixture(scope="module")
def fitted():
    df, low = _data()
    m = ThresholdVAR(df, threshold_var="f", lags=1, max_delay=4, post_draws=2000, hor=8,
                     seed=7)
    m.sample_posterior(show_progress=False)
    return m, low


def test_inverse_wishart_draw_has_the_right_mean():
    rng = np.random.default_rng(0)
    S, df = np.array([[2.0, 0.4], [0.4, 1.0]]), 12.0
    mean = np.mean([tv._draw_iw(S, df, rng) for _ in range(40000)], axis=0)
    np.testing.assert_allclose(mean, S / (df - 3), rtol=0.03)


def _small_prior():
    yd, xd = tv._create_dummies(0.3, 1.0, np.array([0.8, 0.5]), 0.5, 1,
                                np.array([0.2, -0.1]), np.array([1.0, 0.7]), 2)
    return tv._DummyPrior(yd, xd)


def test_niw_draws_match_posterior_moments():
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.standard_normal((300, 2)), np.ones(300)])
    y = x @ np.array([[0.3, 0.1], [0.0, 0.2], [0.5, -0.4]]) + rng.standard_normal((300, 2))
    prior = _small_prior()
    draws = [tv._draw_niw(y, x, prior, rng) for _ in range(20000)]
    B = np.array([b.ravel(order="F") for b, _ in draws])
    X0, Y0 = np.vstack([x, prior.xd]), np.vstack([y, prior.yd])
    XX = X0.T @ X0
    Bstar = np.linalg.solve(XX, X0.T @ Y0)
    E = Y0 - X0 @ Bstar
    nu = X0.shape[0] - 3
    Esigma = E.T @ E / (nu - 3)
    np.testing.assert_allclose(np.mean([s for _, s in draws], axis=0), Esigma, rtol=0.03)
    np.testing.assert_allclose(B.mean(0), Bstar.ravel(order="F"), atol=0.003)
    # B is matric-t: its covariance is E[Sigma] (x) (X*'X*)^-1
    np.testing.assert_allclose(np.cov(B.T), np.kron(Esigma, np.linalg.inv(XX)),
                               rtol=0.05, atol=1e-4)


def test_marginal_likelihood_matches_chib_identity():
    """log p(Y) = log p(Y | B, S) + log p(B, S) - log p(B, S | Y) at any (B, S)."""
    rng = np.random.default_rng(2)
    x = np.column_stack([rng.standard_normal((40, 2)), np.ones(40)])
    y = rng.standard_normal((40, 2))
    prior = _small_prior()

    def niw_logpdf(B, S, XX, XY, YY, nu):
        Bhat = np.linalg.solve(XX, XY)
        Shat = YY - XY.T @ Bhat
        return (invwishart(nu, Shat).logpdf(S)
                + matrix_normal(Bhat, np.linalg.inv(XX), S).logpdf(B))

    B, S = np.array([[0.2, 0.1], [0.0, 0.3], [0.1, 0.0]]), np.array([[1.1, 0.2], [0.2, 0.9]])
    E = y - x @ B
    loglik = (-0.5 * 40 * (2 * np.log(2 * np.pi) + np.linalg.slogdet(S)[1])
              - 0.5 * np.trace(np.linalg.solve(S, E.T @ E)))
    k = 3
    log_prior = niw_logpdf(B, S, prior.XX, prior.XY, prior.YY, prior.rows - k)
    log_post = niw_logpdf(B, S, prior.XX + x.T @ x, prior.XY + x.T @ y, prior.YY + y.T @ y,
                          prior.rows + 40 - k)
    ml = prior.log_ml(x.T @ x, x.T @ y, y.T @ y, 40)
    assert ml == pytest.approx(loglik + log_prior - log_post, abs=1e-8)


def test_split_posterior_matches_direct_computation():
    df, _ = _data(T=120)
    m = ThresholdVAR(df, threshold_var="f", lags=1, max_delay=3, seed=0)
    y, x, Fstar, prior, tar_mean = m._setup()
    sp = tv._split_posterior(y, x, Fstar, prior, m.k, tar_mean, 10.0)
    assert sp["probability"].sum() == pytest.approx(1.0)
    row = sp.iloc[len(sp) // 3]
    low = Fstar[int(row["delay"]) - 1] <= row["lower"]
    assert low.sum() == row["n_regime1"]
    assert not (Fstar[int(row["delay"]) - 1][~low] < row["upper"]).any()
    direct = sum(prior.log_ml(xi.T @ xi, xi.T @ yi, yi.T @ yi, len(yi))
                 for yi, xi in ((y[low], x[low]), (y[~low], x[~low])))
    assert row["log_ml"] == pytest.approx(direct, abs=1e-8)
    assert sp["n_regime1"].min() >= m.k and (m.T - sp["n_regime1"]).min() >= m.k


def test_truncated_threshold_draw_stays_in_interval():
    rng = np.random.default_rng(3)
    for lo, hi in ((-0.2, 0.1), (5.0, 5.3), (-9.0, -8.5)):     # centre and both far tails
        z = np.array([tv._draw_threshold(lo, hi, 0.0, 1.0, rng) for _ in range(2000)])
        assert z.min() >= lo and z.max() < hi


def test_dummies_match_create_dummies_layout():
    yd, xd = tv._create_dummies(0.2, 1.0, np.array([0.9, 1.0]), 0.5, 2,
                                np.array([1.0, 2.0]), np.array([0.5, 2.0]), 2)
    assert yd.shape == (2 * 2 + 2 + 1 + 2, 2) and xd.shape == (9, 5)
    np.testing.assert_allclose(yd[:2], np.diag([0.5 * 0.9, 2.0]) / 0.2)
    np.testing.assert_allclose(xd[2:4, 2:4], 2 * np.diag([0.5, 2.0]) / 0.2)   # second lag
    np.testing.assert_allclose(yd[4:6], np.diag([0.5, 2.0]))                  # covariance block
    assert xd[6, -1] == 0.5                                                   # constant
    np.testing.assert_allclose(xd[7:, :4], np.tile(np.diag([0.9, 2.0]), 2))   # sum of coefficients


def test_sampler_recovers_threshold_delay_and_covariances(fitted):
    m, low = fitted
    summ = m.threshold_summary()
    assert abs(summ.loc["threshold (Z*)", "p50"]) < 0.1
    assert np.mean(m.draws["delay"] == 2) > 0.95
    accuracy = np.mean((m.regime_probability().values <= 0.5) == low[m.start:])
    assert accuracy > 0.95
    np.testing.assert_allclose(m.draws["S1"].mean(0), S1, atol=0.2)
    np.testing.assert_allclose(m.draws["S2"].mean(0), S2, atol=0.6)
    assert m.stable_share > 0.9 and m.failed_stability_draws == 0
    exact = m.split_posterior.groupby("delay")["probability"].sum()
    assert np.mean(m.draws["delay"] == 2) == pytest.approx(exact[2], abs=0.02)


def test_marginal_likelihood_favours_threshold_when_present(fitted):
    m, _ = fitted
    ml = m.log_marginal_likelihood()
    assert ml["log Bayes factor"] > 10
    linear, _ = _data(T=400, tar=np.inf)               # one regime only
    lin = ThresholdVAR(linear, threshold_var="f", lags=1, max_delay=4, post_draws=50, seed=1)
    lin.sample_posterior(show_progress=False)
    assert lin.log_marginal_likelihood()["log Bayes factor"] < 0


def test_stored_regimes_match_stored_threshold_and_delay(fitted):
    m, _ = fitted
    f = m.data[:, m.tarvar]
    for i in (0, 500, m.n_draws - 1):
        d = m.draws["delay"][i]
        lagged = f[m.start - d:m.data.shape[0] - d]
        np.testing.assert_array_equal(m.draws["e1"][i], lagged <= m.draws["tar"][i])


def test_girf_reduces_to_linear_irf_in_one_regime(fitted):
    m, _ = fitted
    draw = m._draw_dict(0)
    draw["tar"] = np.inf                      # every path stays in regime 1
    draw["e1"] = np.ones_like(draw["e1"])
    out = tv._girf_draw(draw, reps=3, scale=1.0, hor=6, L=1, N=2, tarvar=1, seed=0)
    A, C = draw["B1"][:2].T, np.linalg.cholesky(draw["S1"])
    expected = np.stack([np.linalg.matrix_power(A, h) @ C for h in range(6)])
    np.testing.assert_allclose(out["regime1"]["level"], expected, atol=1e-10)
    assert out["regime2"] is None
    unit = tv._girf_draw(draw, reps=3, scale=1.0, hor=2, L=1, N=2, tarvar=1, seed=0, irf_1std=0)
    np.testing.assert_allclose(np.diag(unit["regime1"]["level"][0]), 1.0)


def test_irfs_layout_and_validation(fitted):
    m, _ = fitted
    irf = m.compute_irfs(n_draws=20, reps=5, n_jobs=1, show_progress=False)
    assert irf["regime1"]["level"].shape == (20, 8, 2, 2)
    assert irf["regime2"]["level"].shape == (20, 8, 2, 2)
    with pytest.raises(ValueError):
        m.compute_irfs(shock="uncertainty")


def test_reproducible_with_seed():
    df, _ = _data(T=150)
    a = ThresholdVAR(df, lags=1, max_delay=2, post_draws=200, seed=3)
    b = ThresholdVAR(df, lags=1, max_delay=2, post_draws=200, seed=3)
    da, db = a.sample_posterior(show_progress=False), b.sample_posterior(show_progress=False)
    np.testing.assert_array_equal(da["tar"], db["tar"])
    np.testing.assert_array_equal(da["B2"], db["B2"])


def test_validation_and_sample_start():
    df, _ = _data(T=150)
    with pytest.raises(ValueError):
        ThresholdVAR(df, prior_params={"lambda1": 0.2})
    with pytest.raises(ValueError):
        ThresholdVAR(df, threshold_var="zz")
    m = ThresholdVAR(df, lags=2, max_delay=6, training=0)
    assert m.start == 6 and m.T == 144                 # every delay on the same sample
    assert ThresholdVAR(df, lags=2, max_delay=6, training=10).start == 12


def test_threshold_var_sv_shares_sample_and_comparison_tools(fitted):
    """The SV model no longer fails when max_delay > lags + training, and both
    classes feed the comparison table and plots."""
    m, _ = fitted
    sv = ThresholdVARSV(m.y_df, threshold_var="f", lags=1, vol_lags=1, max_delay=4,
                        training=0, post_draws=40, burnin=0.5, hor=8, seed=1,
                        prior_params={"init_reps": 5, "init_burn": 5})
    assert sv.start == m.start == 4
    sv.sample_posterior(show_progress=False)
    assert sv.regime_probability().index.equals(m.regime_probability().index)
    vol = sv.residual_volatility()
    assert vol.shape == (m.T, 6) and (vol.xs("lower", axis=1, level=1)
                                      <= vol.xs("upper", axis=1, level=1)).all().all()
    sv.irf_1std = 0
    sv.compute_irfs(shock="level", n_draws=5, reps=2, n_jobs=1, show_progress=False)
    m.compute_irfs(n_draws=5, reps=2, n_jobs=1, show_progress=False)

    table = compare_threshold_models({"TVAR": m, "TVAR-SV": sv})
    assert list(table["specification"]) == ["TVAR", "TVAR-SV"]
    assert set(table["T"]) == {m.T}

    models = {"TVAR": m, "TVAR-SV": sv}
    for fig, _ in (plot_regime_comparison(models), plot_irf_comparison(models, shock="a"),
                   plot_volatility_comparison(models), plot_regimes(m), plot_irfs(m)):
        assert fig.axes
    with pytest.raises(RuntimeError):
        plot_irf_comparison(models, shock="uncertainty")
    matplotlib.pyplot.close("all")


def _one_draw(model, i=0, **overrides):
    """Keep posterior draw ``i`` only, with some entries replaced."""
    d = {k: (v[i:i + 1] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == model.n_draws
             else v) for k, v in model.draws.items()}
    for key, value in overrides.items():
        d[key] = np.array([value])
    model.draws = d
    model.n_draws = 1
    return model


def _future(model, values):
    idx = pd.date_range(model.dates[-1], periods=len(values) + 1, freq="MS")[1:]
    return pd.DataFrame(values, index=idx, columns=model.names)


def test_tvar_predictive_density_matches_closed_form_in_one_regime():
    df, _ = _data(T=200)
    m = ThresholdVAR(df, threshold_var="f", lags=1, max_delay=2, post_draws=40, seed=2)
    m.sample_posterior(show_progress=False)
    m = _one_draw(m, tar=np.inf)                        # a linear VAR(1)
    B, S = m.draws["B1"][0], m.draws["S1"][0]
    A, c, yT = B[:2].T, B[2], m.draws["y"][-1]
    actual = np.array([[0.4, -0.2], [1.0, 0.3]])
    out = m.log_predictive_score(_future(m, actual), horizons=(1, 2), n_draws=None, reps=40000)
    m1, V1 = c + A @ yT, S
    m2, V2 = c + A @ m1, S + A @ S @ A.T
    for h, mean, V in ((1, m1, V1), (2, m2, V2)):
        exact = [multivariate_normal(mean, V).logpdf(actual[h - 1])] + [
            multivariate_normal(mean[i], V[i, i]).logpdf(actual[h - 1, i]) for i in range(2)]
        np.testing.assert_allclose(out.loc[h].to_numpy(), exact, atol=1e-10 if h == 1 else 5e-3)


def test_tvarsv_predictive_density_without_volatility_noise(fitted):
    m, _ = fitted
    sv = ThresholdVARSV(m.y_df, threshold_var="f", lags=1, vol_lags=1, max_delay=4, training=0,
                        post_draws=30, seed=3, prior_params={"init_reps": 5, "init_burn": 5})
    sv.sample_posterior(show_progress=False)
    sv = _one_draw(sv, tar=np.inf, Q=1e-16)
    d = {k: (v[0] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == 1 else v)
         for k, v in sv.draws.items()}
    lam_T = d["lam"][-1]
    h_next = d["mu"] + d["F"] * np.log(lam_T)
    x = np.r_[d["y"][-1], h_next, np.log(lam_T), 1.0]
    mean = x @ d["B1"]
    V = np.exp(h_next) * d["iamat1"] @ np.diag(d["Sbig"]) @ d["iamat1"].T
    actual = np.array([[0.3, -0.1]])
    out = sv.log_predictive_score(_future(sv, actual), n_draws=None, reps=1)
    assert out.loc[1, "joint"] == pytest.approx(multivariate_normal(mean, V).logpdf(actual[0]),
                                                abs=1e-6)          # sqrt(Q) = 1e-8 of noise


def test_recursive_scores_and_comparison():
    df, _ = _data(T=160)

    def make(prior):
        return lambda d: ThresholdVAR(d, threshold_var="f", lags=1, max_delay=2,
                                      post_draws=60, seed=4, prior_params=prior)
    origins = df.index[[120, 130, 140, 159]]
    scores = {name: recursive_log_scores(make(prior), df, origins, horizons=(1, 3), n_draws=20,
                                         reps=5, n_jobs=1, show_progress=False)
              for name, prior in (("tight", {"lamdaP": 0.05}), ("loose", {"lamdaP": 1.0}))}
    s = scores["tight"]
    assert s.index.names == ["origin", "horizon"] and list(s.columns) == ["joint", "a", "f"]
    assert s.attrs["spacing"] == 10
    assert s.loc[(origins[-1], 1)].isna().all()          # target beyond the data
    assert np.isfinite(s.loc[(origins[0], 3)]).all()
    table = compare_log_scores(scores, benchmark="tight")
    assert table.index.names == ["model", "horizon", "variable"]
    row = table.loc[("loose", 1, "joint")]
    assert row["difference"] == pytest.approx(row["model"] - row["benchmark"])
    assert row["n"] == 3
    fig, _ = plot_log_scores(scores, benchmark="tight", horizons=(1, 3), variables=("joint", "a"))
    assert len(fig.axes) == 4
    matplotlib.pyplot.close("all")
    m = make({})(df)
    m.sample_posterior(show_progress=False)
    with pytest.raises(ValueError):
        m.log_predictive_score(df.iloc[-5:])               # overlaps the sample
