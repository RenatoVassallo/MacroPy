"""Historical decomposition: the accounting identity holds for both VAR classes."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from MacroPy import BayesianVAR, ClassicVAR

A = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.2, 0.5]])


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    y = np.zeros((180, 3))
    for t in range(1, 180):
        y[t] = A @ y[t - 1] + rng.standard_normal(3)
    return pd.DataFrame(y, columns=["a", "b", "c"],
                        index=pd.date_range("1990-03-01", periods=180, freq="QS"))


def test_classic_var_hd_reproduces_the_sample(data):
    model = ClassicVAR(data, lags=2, progress=False)
    hd = model.compute_hd(plot_hd=True)
    T, n = model.yy.shape
    assert hd["shock"].shape == (T, n, n) and hd["baseline"].shape == (T, n)
    assert np.allclose(hd["baseline"] + hd["shock"].sum(axis=1), model.yy)
    assert np.allclose(hd["endo"], model.yy)
    assert len(hd["dates"]) == T and hd["dates"][-1] == data.index[-1]
    assert model.figures["hd"][0].axes                      # the figure was built


def test_classic_var_hd_with_time_trend(data):
    model = ClassicVAR(data, lags=2, timetrend=True, progress=False)
    hd = model.compute_hd()
    assert np.allclose(hd["endo"], model.yy)


def test_bayesian_var_hd_reproduces_the_sample(data):
    model = BayesianVAR(data, lags=2, post_draws=200, seed=4, progress=False)
    model.sample_posterior()
    hd = model.compute_hd(store_draws=True)
    T, n = model.yy.shape
    assert hd["shock"].shape == (T, n, n)
    assert np.allclose(hd["endo"], model.yy)                # linear within each draw
    assert model.hd_draws.shape[1:] == (T, n, n)
    assert np.allclose(model.hd_draws.mean(axis=0), hd["shock"])


def test_bayesian_var_hd_with_exogenous_block(data):
    exog = pd.DataFrame({"d": (np.arange(len(data)) > 120).astype(float)},
                        index=data.index)
    model = BayesianVAR(data, lags=2, exog=exog, post_draws=100, seed=4, progress=False)
    model.sample_posterior()
    hd = model.compute_hd()
    assert np.allclose(hd["endo"], model.yy)                # exog sits in the baseline


def test_bayesian_var_hd_needs_the_posterior(data):
    model = BayesianVAR(data, lags=2, post_draws=50, seed=4, progress=False)
    with pytest.raises(RuntimeError, match="sample_posterior"):
        model.compute_hd()
