"""Figures: one matplotlib backend, returned objects, no global style changes,
nothing drawn by default."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import MacroPy
from MacroPy import BayesianVAR, ClassicVAR, generate_series_plot, plot_fan_chart


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    y = np.zeros((160, 2))
    for t in range(1, 160):
        y[t] = np.array([[0.5, 0.1], [0.2, 0.4]]) @ y[t - 1] + rng.standard_normal(2)
    return pd.DataFrame(y, columns=["a", "b"],
                        index=pd.date_range("1985-03-01", periods=160, freq="QS"))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def snapshot():
    keys = ("font.family", "font.size", "axes.grid", "axes.spines.top", "axes.titleweight")
    return {k: plt.rcParams[k] for k in keys}


def test_plots_return_figures_and_leave_rcparams(data):
    before = snapshot()
    fig, axes = generate_series_plot(data, title="series")
    assert isinstance(fig, Figure) and len(axes) == 2
    model = ClassicVAR(data, lags=1, nreps=30, hor=6, seed=1, progress=False)
    model.compute_irfs(plot_irfs=True)
    model.compute_fevd(plot_fevd=True)
    model.compute_hd(plot_hd=True)
    assert len(model.figures["irfs"]) == 2
    assert all(isinstance(f, Figure) for f, _ in model.figures["irfs"])
    assert isinstance(model.figures["fevd"][0], Figure)
    assert isinstance(model.figures["hd"][0], Figure)
    draws = np.random.default_rng(0).normal(size=(200, 4))
    fig, ax = plot_fan_chart(data["a"], draws, pd.date_range("2025-03-01", periods=4, freq="QS"))
    assert isinstance(fig, Figure)
    assert snapshot() == before                   # no global side effects


def test_nothing_is_drawn_by_default(data):
    model = ClassicVAR(data, lags=1, nreps=20, hor=4, seed=1, progress=False)
    model.compute_irfs()
    model.compute_fevd()
    model.compute_hd()
    bvar = BayesianVAR(data, lags=1, post_draws=60, seed=1, progress=False)
    bvar.sample_posterior()
    bvar.forecast(fhor=4)
    assert plt.get_fignums() == []


def test_bayesian_plots_are_stored(data):
    bvar = BayesianVAR(data, lags=1, post_draws=60, seed=1, progress=False)
    bvar.sample_posterior()
    bvar.forecast(fhor=4, plot_forecast=True)
    bvar.compute_irfs(plot_irfs=True)
    bvar.compute_fevd(plot_fevd=True)
    assert isinstance(bvar.figures["forecast"][0], Figure)
    assert len(bvar.figures["irfs"]) == 2
    assert isinstance(bvar.figures["fevd"][0], Figure)


def test_use_own_style_switch(data):
    fig, axes = generate_series_plot(data)
    assert not axes[0].spines["top"].get_visible()          # MacroPy style
    MacroPy.use_own_style = False
    try:
        with plt.rc_context({"axes.spines.top": True}):
            fig, axes = generate_series_plot(data)
        assert axes[0].spines["top"].get_visible()          # caller's style kept
    finally:
        MacroPy.use_own_style = True


def test_public_api_is_explicit():
    assert all(hasattr(MacroPy, name) for name in MacroPy.__all__)
    assert {"BayesianVAR", "ClassicVAR", "PALETTE", "macropy_style", "set_bse_style",
            "coefficients_by_lag", "plot_fan_chart"} <= set(MacroPy.__all__)
    assert not hasattr(MacroPy, "ggplot")                   # no leaked star imports
    assert isinstance(MacroPy.PALETTE["stack"], list)


def test_style_block_keeps_backend_state():
    # IPython turns interactive mode on when the inline backend is activated;
    # that must survive the style block or later figures are never shown.
    before = plt.rcParams["interactive"]
    try:
        with MacroPy.macropy_style():
            plt.rcParams["interactive"] = True
        assert plt.rcParams["interactive"] is True
    finally:
        plt.rcParams["interactive"] = before
