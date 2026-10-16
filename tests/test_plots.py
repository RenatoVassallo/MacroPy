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
    assert {"BayesianVAR", "ClassicVAR", "PALETTE", "macropy_style", "set_style",
            "reset_style", "style_context", "coefficients_by_lag",
            "plot_fan_chart"} <= set(MacroPy.__all__)
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


def test_smooth_local_projections_rejects_unknown_projection_type():
    from MacroPy import SmoothLocalProjections
    rng = np.random.default_rng(1)
    shock = pd.Series(rng.standard_normal(80))
    response = pd.Series(rng.standard_normal(80))
    model = SmoothLocalProjections(response=response, shock=shock,
                                   controls=response.to_frame("y"), lags=1,
                                   horizons=list(range(1, 5)))
    with pytest.raises(ValueError):
        model.estimate("smoooth")


def test_smooth_local_projections_has_its_methods():
    # cross_validate_lambda and plot_irf_with_confidence use self.result["X"/"Y"/"P"]
    # and self.h_max: they belong to SmoothLocalProjections, not to the panel class.
    from MacroPy import SmoothLocalProjections
    rng = np.random.default_rng(5)
    shock = pd.Series(rng.standard_normal(200))
    response = pd.Series(np.r_[0.0, 0.5 * shock.to_numpy()[:-1]]
                         + 0.3 * rng.standard_normal(200))
    model = SmoothLocalProjections(response=response, shock=shock,
                                   controls=response.to_frame("y"), lags=2,
                                   horizons=list(range(1, 9)))
    model.estimate(projection_type="smooth", penalty_order=2, penalty_lambda=0.01)
    best = model.cross_validate_lambda([10.0, 50.0, 100.0], plot=True)
    assert best in (10.0, 50.0, 100.0)
    assert isinstance(model.cv_figure[0], Figure)
    model.estimate(projection_type="smooth", penalty_lambda=best)
    model.compute_confidence_intervals(alpha=0.1)
    fig, ax = model.plot_irf_with_confidence()
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# The MacroPy style: opt-in, local by default, one source of truth
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_style():
    yield
    MacroPy.reset_style()


def test_set_style_applies_globally_and_resets(clean_style, data):
    before = {k: plt.rcParams[k] for k in ("font.size", "font.family", "axes.prop_cycle")}
    MacroPy.set_style("slides", font="sans")
    assert plt.rcParams["font.size"] == pytest.approx(11 * 1.35, abs=0.05)
    assert plt.rcParams["font.family"] == ["sans-serif"]
    assert plt.rcParams["mathtext.fontset"] == "dejavusans"
    fig, ax = plt.subplots()
    line, = ax.plot([0, 1], [0, 1])
    assert line.get_color() == MacroPy.PALETTE["cycle"][0]       # your figures, MacroPy colours
    fig, axes = generate_series_plot(data)                        # MacroPy's figures follow
    assert axes[0].title.get_fontsize() == pytest.approx(12 * 1.35, abs=0.05)
    MacroPy.reset_style()
    assert {k: plt.rcParams[k] for k in before} == before


def test_style_context_is_local():
    before = snapshot()
    with MacroPy.style_context("paper"):
        assert plt.rcParams["savefig.dpi"] == 300
        assert plt.rcParams["font.size"] == pytest.approx(11 * 0.85, abs=0.05)
    assert snapshot() == before
    with pytest.raises(ValueError):
        MacroPy.macropy_rcparams("poster")


def test_bundled_style_sheet_matches_the_code():
    from importlib.resources import files
    from MacroPy.style import _mplstyle_text
    assert files("MacroPy").joinpath("macropy.mplstyle").read_text() == _mplstyle_text()
    with plt.rc_context():
        plt.style.use("MacroPy.macropy")
        params = MacroPy.macropy_rcparams("notebook", "serif")
        for key in ("axes.titlecolor", "grid.color", "font.size", "savefig.bbox",
                    "axes.spines.top", "date.converter"):
            expected = params[key]
            if isinstance(expected, str) and expected.startswith("#"):
                expected = expected.lower()
            got = plt.rcParams[key]
            assert (got.lower() if isinstance(got, str) else got) == expected, key
        assert plt.rcParams["axes.prop_cycle"].by_key()["color"] == MacroPy.PALETTE["cycle"]


def test_deprecated_style_names_still_work(clean_style):
    with pytest.warns(DeprecationWarning):
        MacroPy.set_bse_style()
    assert plt.rcParams["axes.spines.top"] is False
    with pytest.warns(DeprecationWarning):
        assert MacroPy.BSE_ORANGE == MacroPy.PALETTE["accent"]
    from MacroPy import plots_kalman
    with pytest.warns(DeprecationWarning):
        assert plots_kalman.BSE_NAVY == MacroPy.PALETTE["navy"]
    with pytest.raises(AttributeError):
        MacroPy.NOT_A_NAME


def test_palette_edits_reach_every_module(data):
    from MacroPy import plot_with_bands
    original = MacroPy.PALETTE["accent"]
    MacroPy.PALETTE["accent"] = "#123456"
    try:
        x = np.arange(5.0)
        fig, ax = plot_with_bands(x, x, np.ones(5))
        assert ax.lines[0].get_color() == "#123456"
    finally:
        MacroPy.PALETTE["accent"] = original
    assert MacroPy.color_palette(10)[8] == MacroPy.PALETTE["cycle"][0]
    assert MacroPy.colormap("diverging")(0.5)[:3] == pytest.approx(
        matplotlib.colors.to_rgb(MacroPy.PALETTE["light"]), abs=0.01)


def test_figures_unpack_but_are_not_echoed_in_notebooks(data):
    result = generate_series_plot(data)
    fig, axes = result                                   # ordinary tuple
    assert isinstance(fig, Figure) and isinstance(result, tuple)
    assert hasattr(result, "_ipython_display_")         # no Out[] echo of the tuple
    model = ClassicVAR(data, lags=1, nreps=20, hor=4, seed=1, progress=False)
    model.compute_irfs(plot_irfs=True)
    assert isinstance(model.figures["irfs"], list)
    assert hasattr(model.figures["irfs"], "_ipython_display_")
