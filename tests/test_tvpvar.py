"""Tests for TimeVaryingVAR (Chan & Eisenstat, 2018)."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln

from MacroPy import TimeVaryingVAR, compare_tvpvar
from MacroPy import tvpvar as tv


def _data(T=70, n=3, seed=0):
    """A stable VAR(1) with a volatility break, quarterly index."""
    rng = np.random.default_rng(seed)
    A = 0.5 * np.eye(n) + 0.1 * np.tri(n, k=-1)
    y = np.zeros((T, n))
    for t in range(1, T):
        scale = 2.0 if t > T // 2 else 1.0
        y[t] = 0.3 + y[t - 1] @ A.T + scale * rng.standard_normal(n)
    idx = pd.date_range("2000-03-01", periods=T, freq="QS-MAR")
    return pd.DataFrame(y, index=idx, columns=["a", "b", "c"][:n])


def _kalman_loglik(Y, X, var, Q, theta0):
    T, n, k = X.shape
    a, P, ll = theta0.copy(), np.zeros((k, k)), 0.0
    for t in range(T):
        P = P + np.diag(Q)
        F = X[t] @ P @ X[t].T + np.diag(var[t])
        v = Y[t] - X[t] @ a
        ll += -0.5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(F)[1] + v @ np.linalg.solve(F, v))
        K = P @ X[t].T @ np.linalg.inv(F)
        a, P = a + K @ v, P - K @ X[t] @ P
    return ll


@pytest.fixture(scope="module")
def fitted():
    out = {}
    for model in ("C", "C-SV", "TVP-SV", "TVP-R3-SV"):
        m = TimeVaryingVAR(_data(), model=model, lags=1, post_draws=400, burnin=0.5,
                           seed=3, progress=False)
        m.sample_posterior()
        out[model] = m
    return out


def test_integrated_loglik_matches_kalman_filter():
    m = TimeVaryingVAR(_data(), model="TVP-SV", lags=2, progress=False)
    rng = np.random.default_rng(1)
    T, n, k = m.X.shape
    Q = rng.uniform(1e-4, 1e-3, k)
    theta0 = rng.normal(0.0, 0.3, k)
    h = rng.normal(0.0, 0.3, (T, n))
    band = tv._ll_states(m.yy, m.X, np.exp(-h), h.sum(), Q, theta0, tv._Band(T, k))
    assert band == pytest.approx(_kalman_loglik(m.yy, m.X, np.exp(h), Q, theta0), abs=1e-8)


def test_block_recursion_matches_dense_inverse():
    m = TimeVaryingVAR(_data(T=30), model="TVP-SV", lags=1, progress=False)
    T, n, k = m.X.shape
    rng = np.random.default_rng(2)
    w = np.exp(rng.normal(0.0, 0.3, (T, n)))
    Q = rng.uniform(1e-4, 1e-2, k)
    ab = tv._state_precision(m.X, w, Q, tv._Band(T, k))
    K = np.zeros((T * k, T * k))
    for r in range(k + 1):
        K[np.arange(T * k - (k - r)), np.arange(k - r, T * k)] = ab[r, k - r:]
    K = K + np.triu(K, 1).T
    Xd = np.zeros((T * n, T * k))
    for t in range(T):
        Xd[t * n:(t + 1) * n, t * k:(t + 1) * k] = m.X[t]
    G = np.matmul(np.swapaxes(m.X * w[:, :, None], 1, 2), m.X)
    diag, M = tv._inverse_blocks(G, Q, m.X)
    Kinv = np.linalg.inv(K)
    for t in range(T):
        np.testing.assert_allclose(diag[t], Kinv[t * k:(t + 1) * k, t * k:(t + 1) * k], atol=1e-10)
    np.testing.assert_allclose(M, Xd @ Kinv @ Xd.T, atol=1e-10)


def test_constant_model_marginal_likelihood_matches_quadrature():
    m = TimeVaryingVAR(_data(T=60, n=2), model="C", lags=1, post_draws=3000, burnin=0.2,
                       seed=2, progress=False)
    m.sample_posterior()
    ml = m.log_marginal_likelihood(n_draws=2000)
    pp = m.prior_params
    nu, S, V = pp["sigma_df"], (pp["sigma_df"] - 1) * pp["sigma_var"], pp["coef_var"]
    exact = 0.0
    g = np.linspace(np.log(1e-3), np.log(1e3), 40001)
    for i in range(m.n_endo):
        Zi = m.Z[:, i, :]
        Zi = Zi[:, np.abs(Zi).sum(axis=0) > 0]
        lam, U = np.linalg.eigh(V * Zi @ Zi.T)
        q = U.T @ m.yy[:, i]
        s2 = np.exp(g)[:, None]
        f = (-0.5 * m.T * np.log(2 * np.pi) - 0.5 * np.log(s2 + lam).sum(1)
             - 0.5 * (q * q / (s2 + lam)).sum(1)
             + nu * np.log(S) - gammaln(nu) - (nu + 1) * g - S / np.exp(g) + g)
        exact += f.max() + np.log(np.trapezoid(np.exp(f - f.max()), g))
    assert ml["logml"] == pytest.approx(exact, abs=max(0.05, 4 * ml["nse"]))


def test_logvol_integral_reduces_to_gaussian_without_volatility_noise():
    rng = np.random.default_rng(4)
    e = rng.standard_normal((50, 2)) * [1.0, 2.0]
    h0 = np.log([1.0, 4.0])
    ll = tv._ll_csv(e * e, np.array([1e-9, 1e-9]), h0, rng, R=50, adaptive=False)
    exact = float(np.sum(-0.5 * (np.log(2 * np.pi) + h0) - 0.5 * e * e * np.exp(-h0)))
    assert ll == pytest.approx(exact, abs=1e-3)


def test_gamma_fit_recovers_parameters():
    x = np.random.default_rng(0).gamma(4.0, 1.0 / 2.5, size=200000)
    shape, rate = tv._fit_gamma(x)
    assert shape == pytest.approx(4.0, rel=0.02)
    assert rate == pytest.approx(2.5, rel=0.02)


@pytest.mark.parametrize("model", ["C", "C-SV", "TVP-SV", "TVP-R3-SV"])
def test_historical_decomposition_identity(fitted, model):
    m = fitted[model]
    hd = m.compute_hd()
    np.testing.assert_allclose(hd["baseline"] + hd["shock"].sum(axis=1), m.yy, atol=1e-10)


def test_irf_impact_is_cholesky_of_reduced_form(fitted):
    m = fitted["TVP-SV"]
    t = 20
    A, _, logvar = m._coef_at(t)
    irf = m._irf_at(t, 3)
    Ainv = np.linalg.inv(A[0])
    omega = Ainv @ np.diag(np.exp(logvar[0])) @ Ainv.T
    np.testing.assert_allclose(irf[0, 0], np.linalg.cholesky(omega), atol=1e-10)


def test_irfs_fevd_and_dates(fitted):
    m = fitted["TVP-SV"]
    irf = m.compute_irfs(dates=["2005Q4", m.yy_dates[-1]], cred_interval=[0.68, 0.9], hor=6)
    assert irf["median"].shape == (m.T, 6, 3, 3)
    assert set(irf["draws"]) == {"2005Q4", str(m.yy_dates[-1].to_period("Q"))}
    lo, hi = irf["bands"][0.9]
    assert np.all(lo <= irf["median"] + 1e-12) and np.all(irf["median"] <= hi + 1e-12)
    fe = m.compute_fevd(hor=6)["fevd"]
    np.testing.assert_allclose(fe.sum(axis=2), 100.0)
    with pytest.raises(ValueError):
        m.compute_irfs(dates=["1990Q1"])


def test_constant_model_irfs_do_not_vary_over_time(fitted):
    irf = fitted["C"].compute_irfs(hor=5)
    np.testing.assert_allclose(irf["median"], irf["median"][:1].repeat(fitted["C"].T, 0))


def test_forecast_shapes_and_reproducibility(fitted):
    m = fitted["TVP-R3-SV"]
    a = m.forecast(fhor=4, reps=2)["forecast_draws"]
    b = m.forecast(fhor=4, reps=2)["forecast_draws"]
    assert a.shape == (2 * m.draws["save_idx"].shape[0], 4, 3)
    np.testing.assert_array_equal(a, b)
    frame = m.forecast_frame(fhor=4)
    assert list(frame.columns[:3]) == ["date", "mean", "median"]
    assert frame.loc[("a", 1), "date"] == m.yy_dates[-1] + pd.DateOffset(months=3)


def test_storage_layers_and_thinning():
    m = TimeVaryingVAR(_data(), model="TVP-SV", lags=1, post_draws=300, burnin=0.5,
                       progress=False)
    d = m.sample_posterior(thin=10)
    assert d["state_var"].shape == (150, m.ks)
    assert d["states"].shape == (15, m.T, m.ks)
    assert d["h"].shape == (15, m.T, 3)
    assert m.storage_mb(thin=10) > 0


def test_validation_and_comparison(fitted):
    with pytest.raises(ValueError):
        TimeVaryingVAR(_data(), model="TVP-R4")
    with pytest.raises(ValueError):
        TimeVaryingVAR(_data(), prior_params={"lambda1": 0.2})
    assert TimeVaryingVAR(_data(), model="var-sv").model == "C-SV"
    other = TimeVaryingVAR(_data(), model="C", lags=1, presample=3, progress=False)
    with pytest.raises(ValueError):
        compare_tvpvar({"a": fitted["C"], "b": other})


def test_grid_layout_is_balanced():
    from MacroPy._plotting import _grid_cols
    shapes = {n: (-(-n // _grid_cols(n)), _grid_cols(n)) for n in (1, 2, 3, 4, 5, 6, 9, 12)}
    assert shapes == {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3),
                      9: (3, 3), 12: (3, 4)}


def test_tvp_plots_use_grid_and_long_fevd_horizon():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = pd.concat([_data(n=3), _data(n=3, seed=5).add_suffix("2").iloc[:, :1]], axis=1)
    m = TimeVaryingVAR(data, model="C-SV", lags=1, post_draws=200, burnin=0.5, seed=3,
                       progress=False, hor=24)
    m.sample_posterior()
    fig, axes = m.plot_volatility()
    assert len(axes) == 4 and axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 2)
    m.compute_irfs(dates=[m.yy_dates[-1]])
    fig, axes = m.plot_irfs(shock=0)
    assert axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 2)
    m.compute_fevd()
    fig, axes = m.plot_fevd()
    assert "20 steps" in fig._suptitle.get_text()
    fig, axes = m.plot_fevd(ncol=4)
    assert axes[0].get_subplotspec().get_gridspec().get_geometry() == (1, 4)
    plt.close("all")


def _correlated(T=80, seed=0):
    """A VAR(1) like ``_data`` with contemporaneously correlated shocks, so that the
    impact variance decomposition is not trivially 0 / 100."""
    rng = np.random.default_rng(seed)
    A = 0.5 * np.eye(3) + 0.1 * np.tri(3, k=-1)
    C = np.array([[1.0, 0.0, 0.0], [0.8, 1.0, 0.0], [0.5, -0.6, 1.0]])
    y = np.zeros((T, 3))
    for t in range(1, T):
        scale = 2.0 if t > T // 2 else 1.0
        y[t] = 0.3 + y[t - 1] @ A.T + scale * C @ rng.standard_normal(3)
    idx = pd.date_range("2000-03-01", periods=T, freq="QS-MAR")
    return pd.DataFrame(y, index=idx, columns=["a", "b", "c"])


def test_fevd_does_not_depend_on_irf_1std():
    """The FEVD always uses one-standard-deviation shocks; irf_1std only scales the IRFs."""
    fevd, impact = {}, {}
    for flag in (0, 1):
        m = TimeVaryingVAR(_correlated(), model="TVP-SV", lags=1, post_draws=200, burnin=0.5,
                           seed=3, irf_1std=flag, progress=False)
        m.sample_posterior()
        fevd[flag] = m.compute_fevd(hor=4)["fevd"]
        impact[flag] = m.compute_irfs(hor=1)["median"][:, 0]
    np.testing.assert_allclose(fevd[0], fevd[1])
    np.testing.assert_allclose(fevd[0].sum(axis=2), 100.0)       # (T, hor, shock, variable)
    assert np.abs(fevd[0][:, 0, 0, 1]).max() > 5                 # a non-trivial impact split
    assert not np.allclose(impact[0], impact[1])                 # the IRFs do depend on it
    np.testing.assert_allclose(np.diagonal(impact[0], axis1=1, axis2=2), 1.0)
    one_sd = m._irf_at(10, 2, one_sd=True)
    assert m.irf_1std == 1 and np.array_equal(one_sd, m._irf_at(10, 2))
    assert not np.allclose(one_sd, m._irf_at(10, 2, one_sd=False))


def test_fevd_is_scale_invariant_but_unit_irfs_scale():
    """Rescaling a variable leaves the impact FEVD unchanged and scales the unit IRFs.

    C-SV with flat priors on the coefficients and initial log-volatilities is
    scale equivariant, so the posterior itself does not move. TVP-SV is not:
    its state-innovation priors are in the units of the data (0.01), and
    multiplying a variable by 100 changes the impact FEVD by tens of points
    through the prior, not through the decomposition.
    """
    df = _correlated()
    scaled = df.copy()
    scaled["b"] *= 100.0
    fits = []
    for data in (df, scaled):
        m = TimeVaryingVAR(data, model="C-SV", lags=1, post_draws=300, burnin=0.5, seed=5,
                           irf_1std=0, progress=False,
                           prior_params={"coef_var": 1e6, "h0_var": 1e4})
        m.sample_posterior()
        fits.append(m)
    f0, f1 = (m.compute_fevd(hor=3)["fevd"][:, 0] for m in fits)   # impact, (T, shock, variable)
    assert f0[:, 0, 1].mean() > 20                                 # shock a explains b on impact
    np.testing.assert_allclose(f0, f1, atol=1.0)                   # Monte Carlo error only
    i0, i1 = (m.compute_irfs(hor=1)["median"][:, 0] for m in fits)  # unit shocks
    assert np.median(i1[:, 1, 0] / i0[:, 1, 0]) == pytest.approx(100.0, rel=0.05)   # a -> b
    assert np.median(i1[:, 2, 1] / i0[:, 2, 1]) == pytest.approx(0.01, rel=0.05)    # b -> c

    def unit_shares(m, t):                                         # the decomposition before the fix
        th = m._irf_at(t, 1, one_sd=False)[:, 0]
        return (100.0 * th ** 2 / (th ** 2).sum(axis=2, keepdims=True)).mean(axis=0)
    assert np.abs(unit_shares(fits[0], 40) - unit_shares(fits[1], 40)).max() > 10


@pytest.mark.parametrize("model", ["C-SV", "TVP-SV", "TVP-R2-SV"])
def test_block_exogeneity_shuts_domestic_feedback(model):
    """With ``b_exo[0, 1:] = 1`` and ``a`` ordered first, shocks to ``b`` and ``c``
    never move ``a``: the excluded lags are exact zeros in every draw and date."""
    mask = np.zeros((3, 3), dtype=int)
    mask[0, 1:] = 1
    free = TimeVaryingVAR(_data(), model=model, lags=2, progress=False)
    m = TimeVaryingVAR(_data(), model=model, lags=2, post_draws=200, burnin=0.5, seed=4,
                       progress=False, b_exo=mask)
    assert m.ks + m.kc == free.ks + free.kc - 4                # 2 variables x 2 lags
    m.sample_posterior()
    lags = m.coefficient_frame("lags")
    for v in ("b", "c"):
        for lag in (1, 2):
            assert np.all(lags[(f"a<-{v}(-{lag})", "median")] == 0.0)
    irf = m.compute_irfs(hor=8, dates=[m.yy_dates[-1]])
    np.testing.assert_array_equal(irf["median"][:, :, 0, 1:], 0.0)
    assert np.all(list(irf["draws"].values())[0][:, :, 0, 1:] == 0.0)
    assert np.abs(irf["median"][:, 1:, 1:, 0]).max() > 0       # a still moves b and c
    hd = m.compute_hd()
    np.testing.assert_allclose(hd["shock"][:, 1:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(hd["baseline"] + hd["shock"].sum(axis=1), m.yy, atol=1e-10)
    fc = m.forecast(fhor=3)["forecast_draws"]
    assert np.isfinite(fc).all()


def test_block_exogeneity_validation_and_likelihood():
    with pytest.raises(ValueError):
        TimeVaryingVAR(_data(), b_exo=np.ones((2, 2)))
    with pytest.raises(ValueError):
        TimeVaryingVAR(_data(), b_exo=np.full((3, 3), 2))
    mask = np.zeros((3, 3), dtype=int)
    mask[0, 1:] = 1
    m = TimeVaryingVAR(_data(), model="C", lags=1, post_draws=300, burnin=0.5, seed=2,
                       progress=False, b_exo=mask)
    m.sample_posterior()
    # equation a keeps only its intercept and its own lag
    assert m.Z.shape[2] == 3 * 4 + 3 - 2
    assert np.count_nonzero(np.any(m.Z[:, 0, :] != 0, axis=0)) == 2
    ml = m.log_marginal_likelihood(n_draws=200, n_batches=10)
    assert np.isfinite(ml["logml"])


@pytest.mark.parametrize("sig2h,h0", [(0.5, 3.0), (1.0, 5.0), (0.5, 0.0)])
def test_logvol_mode_converges_with_wide_residual_range(sig2h, h0):
    """A turbulent start and then a rate that barely moves (squared residuals
    from 1e-13 to 500): full Newton steps cycle; the damped search converges to
    the stationary point."""
    T = 120
    rng = np.random.default_rng(0)
    z = rng.standard_normal((T, 1)) ** 2
    z[:20] *= 100.0
    z[20:] *= 1e-3 * rng.random((T - 20, 1)) ** 4
    s2, m0 = np.array([sig2h]), np.array([h0])
    h, ok = tv._newton_logvol(z, s2, m0, np.full((T, 1), h0), 1e-8, 500)
    assert ok
    grad = -tv._rw_grad(h, m0, s2) - 0.5 * (1.0 - np.exp(-h) * z)
    assert np.abs(grad).max() < 1e-8
    # on well-behaved data the full step is always accepted (the authors' iterates)
    clean = rng.standard_normal((T, 1)) ** 2
    hc, okc = tv._newton_logvol(clean, s2, m0, np.full((T, 1), h0), 0.01, 100)
    assert okc and np.all(np.isfinite(hc))
