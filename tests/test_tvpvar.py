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
