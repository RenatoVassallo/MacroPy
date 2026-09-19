"""Regression tests for the ClassicVAR bootstrap, prior-parameter hygiene and
import side effects (feedback round from the Macroeconometria book)."""
import warnings

import numpy as np
import pandas as pd
import pytest

import MacroPy
from MacroPy import BayesianVAR, ClassicVAR
from MacroPy.data_handling import coefficients_by_lag
from MacroPy.priors import MinnesotaPrior, normalize_prior_params

PHI = np.array([[0.6, 0.1], [0.0, 0.5]])
S_TRUE = np.array([[1.0, 0.0], [0.5, 0.4]])


def simulated_var(T=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = PHI @ y[t - 1] + S_TRUE @ rng.standard_normal(2)
    index = pd.date_range("1900-01-01", periods=T, freq="MS")
    return pd.DataFrame(y, columns=["a", "b"], index=index)


def test_bootstrap_recovers_true_impact():
    # The old design (refitted values re-lagged) inflated the impact by ~28 %.
    draws = ClassicVAR(simulated_var(), lags=1, bstrap=True, nreps=100, hor=2,
                       seed=1, progress=False).compute_irfs()
    assert draws.shape == (100, 2, 2, 2)
    assert np.allclose(draws[:, 0].mean(axis=0), S_TRUE, atol=0.06)


def test_montecarlo_impact_carries_sigma_uncertainty():
    draws = ClassicVAR(simulated_var(), lags=1, bstrap=False, nreps=200, hor=2,
                       seed=1, progress=False).compute_irfs()
    assert draws[:, 0, 0, 0].std() > 0          # no longer a single point
    assert np.allclose(draws[:, 0].mean(axis=0), S_TRUE, atol=0.06)


def test_classic_var_seed_is_reproducible():
    df = simulated_var(T=300)
    kw = dict(lags=1, bstrap=True, nreps=20, hor=3, progress=False)
    a = ClassicVAR(df, seed=7, **kw).compute_irfs()
    b = ClassicVAR(df, seed=7, **kw).compute_irfs()
    c = ClassicVAR(df, seed=8, **kw).compute_irfs()
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_prior_params_defaults_and_deprecated_spelling():
    first, second = normalize_prior_params(), normalize_prior_params()
    first["lambda1"] = 99.0
    assert second["lambda1"] == 0.2                # no shared mutable default
    assert normalize_prior_params({"lambda1": 0.3})["lambda1"] == 0.3
    with pytest.warns(DeprecationWarning, match="lamda1"):
        assert normalize_prior_params({"lamda1": 0.3})["lambda1"] == 0.3
    with pytest.raises(ValueError, match="Unknown prior_params key"):
        normalize_prior_params({"lambda_1": 0.3})


def test_minnesota_old_spelling_matches_new():
    df = simulated_var(T=200)
    model = BayesianVAR(df, lags=2, prior_params={"lambda1": 0.3, "lambda2": 0.4},
                        post_draws=10, progress=False)
    with pytest.warns(DeprecationWarning):
        old = MinnesotaPrior(model.yy, model.XX, 2, model.ncoeff_eq,
                             {"lamda1": 0.3, "lamda2": 0.4})
    new = MinnesotaPrior(model.yy, model.XX, 2, model.ncoeff_eq,
                         {"lambda1": 0.3, "lambda2": 0.4})
    assert np.allclose(old["H"], new["H"]) and np.allclose(old["b0"], new["b0"])
    assert model.prior_params["lambda1"] == 0.3


def test_coefficients_by_lag_recovers_blocks():
    n, p = 2, 2
    phi1, phi2, c = PHI, 0.1 * np.eye(2), np.array([0.5, -0.2])
    B = np.vstack([phi1.T, phi2.T, c])             # MacroPy layout: lags, then constant
    phis, const = coefficients_by_lag(B, n, p)
    assert np.allclose(phis[0], phi1) and np.allclose(phis[1], phi2)
    assert np.allclose(const, c)


def test_import_does_not_silence_warnings():
    global_ignore = [f for f in warnings.filters
                     if f[0] == "ignore" and f[1] is None and f[2] is Warning and f[3] is None]
    assert not global_ignore
    assert MacroPy.__version__
