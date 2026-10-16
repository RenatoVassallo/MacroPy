"""
Figures for the VAR classes (ClassicVAR, BayesianVAR, BayesianPanelVAR).

Every function draws with matplotlib inside ``macropy_style()`` (no global
rcParams changes) and returns ``(fig, axes)``, or a list of them; nothing is
displayed here. In a notebook the inline backend shows the figures at the end
of the cell; in a script call ``plt.show()``.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from ._plotting import (  # noqa: F401  (re-exported for older private imports)
    _bottom_legend, _date_axis, _facet_grid_shape, _panels, _stack_colors,
)
from .style import PALETTE, styled

__all__ = [
    "generate_series_plot",
    "generate_coeff_plot",
    "generate_exog_coeff_plot",
    "generate_panel_coeff_plots",
    "generate_irf_plots",
    "generate_fevd_plot",
    "generate_hd_plot",
    "generate_forecast_plots",
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _levels(cred_interval) -> list:
    levels = [cred_interval] if isinstance(cred_interval, (float, int)) else list(cred_interval)
    return sorted(levels, reverse=True)                     # widest first


def _band_alpha(level: float) -> float:
    return 0.2 if level >= 0.9 else 0.4


def _band_label(level: float) -> str:
    return f"{100 * level:g} %"


def _density(ax, x, color) -> None:
    """Kernel density with the posterior median marked."""
    x = np.asarray(x, dtype=float)
    median = float(np.median(x))
    if np.ptp(x) > 0:
        try:
            grid = np.linspace(x.min(), x.max(), 200)
            dens = gaussian_kde(x)(grid)
            ax.fill_between(grid, dens, color=color, alpha=0.4, lw=0)
            ax.plot(grid, dens, color=color, lw=1.0)
        except np.linalg.LinAlgError:
            ax.hist(x, bins=30, density=True, color=color, alpha=0.4)
    ax.axvline(median, color=PALETTE["conditional"], ls="--", lw=1.0, zorder=3)
    ax.text(median, 0, f" {median:.2f}", color=PALETTE["conditional"],
            va="bottom", ha="left", fontsize=9)
    ax.set_yticks([])


# ----------------------------------------------------------------------------
# Data and posterior coefficients
# ----------------------------------------------------------------------------

@styled
def generate_series_plot(yy, yy_dates=None, yy_names=None, series_titles=None, title=None,
                         color_scheme=1, n_breaks: int = 10, zero_line: bool = False):
    """
    One panel per series, two per row.

    Parameters
    ----------
    yy : DataFrame or array (T x n)
    yy_dates : dates, required when ``yy`` is an array
    yy_names : column names when ``yy`` is an array
    series_titles : panel titles (default: column names)
    title : figure title
    color_scheme : 1-4, line colour
    n_breaks : approximate number of date ticks
    zero_line : draw a horizontal line at zero

    Returns
    -------
    (fig, axes)
    """
    colors = {1: PALETTE["navy"], 2: PALETTE["conditional"], 3: "#55A868", 4: "#8172B3"}
    color = colors.get(color_scheme, PALETTE["navy"])
    if isinstance(yy, pd.DataFrame):
        df = yy.copy()
        df.index = pd.to_datetime(yy_dates if yy_dates is not None else df.index)
    else:
        df = pd.DataFrame(np.asarray(yy), index=pd.to_datetime(yy_dates), columns=yy_names)
    labels = list(series_titles) if series_titles else [str(c) for c in df.columns]
    fig, axes = _panels(df.shape[1], 2)
    for ax, column, label in zip(axes, df.columns, labels):
        ax.plot(df.index, df[column], color=color, lw=1.4)
        if zero_line:
            ax.axhline(0, color=PALETTE["median"], lw=0.8, ls="--")
        ax.set_title(label)
        _date_axis(ax, n_breaks)
    if title:
        fig.suptitle(title, fontweight="bold")
    return fig, axes


@styled
def generate_coeff_plot(self):
    """
    Posterior densities of the constants and of the lag coefficients.

    Returns
    -------
    (const, lags) : ``const`` is ``(fig, axes)`` or None without a constant;
        ``lags`` is a list with one ``(fig, axes)`` per lag, panel ``[i, j]``
        showing the coefficient of variable ``j`` in equation ``i``.
    """
    if not hasattr(self, "beta_draws") or len(self.beta_draws) == 0:
        raise ValueError("Posterior draws for beta coefficients are not available.")
    beta = np.asarray(self.beta_draws)
    n, p, k = self.n_endo, self.lags, self.ncoeff_eq
    coef = beta.reshape(beta.shape[0], n, k)                # [draw, equation, regressor]
    names = [str(v) for v in self.names]

    const = None
    if self.constant:
        fig, axes = _panels(n, min(4, n), panel=(4.0, 3.0))
        for i, ax in enumerate(axes):
            _density(ax, coef[:, i, n * p], PALETTE["accent"])
            ax.set_title(f"constant, eq. {names[i]}", fontsize=10)
        fig.suptitle("Posterior distributions of the constants", fontweight="bold")
        const = (fig, axes)

    lags = []
    for lag in range(1, p + 1):
        fig, axes = _panels(n * n, n, panel=(4.0, 2.5))
        for i in range(n):
            for j in range(n):
                ax = axes[i * n + j]
                _density(ax, coef[:, i, (lag - 1) * n + j], PALETTE["density"])
                ax.set_title(f"{names[j]} -> eq. {names[i]}", fontsize=9)
        fig.suptitle(f"Posterior distributions of VAR coefficients (lag {lag})",
                     fontweight="bold")
        lags.append((fig, axes))
    return const, lags


@styled
def generate_exog_coeff_plot(self, bins: int = 30, ncol: int = None,
                             series_titles: list = None, palette: list = None):
    """
    Posterior histograms of the user-supplied exogenous coefficients: one panel
    per equation, one colour per regressor (e.g. one per pandemic dummy).

    Returns
    -------
    (fig, axes)
    """
    if not hasattr(self, "beta_draws") or len(self.beta_draws) == 0:
        raise ValueError("No posterior draws available. Run sample_posterior() first.")
    if getattr(self, "n_exog_user", 0) == 0:
        raise ValueError("This model was estimated without user-supplied exogenous variables.")
    n, k, m = self.n_endo, self.ncoeff_eq, self.n_exog_user
    if series_titles is not None and len(series_titles) != n:
        raise ValueError(f"`series_titles` must have length {n}; got {len(series_titles)}.")
    labels = [str(t) for t in (series_titles if series_titles is not None else self.names)]
    exog_names = [str(e) for e in self.exog_names]
    start = n * self.lags + int(self.constant) + int(self.timetrend)
    coef = np.asarray(self.beta_draws).reshape(-1, n, k)
    colors = list(palette) if palette is not None else list(PALETTE["pastel"])

    _, ncol_eff = _facet_grid_shape(n, ncol)
    fig, axes = _panels(n, ncol_eff, panel=(4.5, 2.8))
    for i, ax in enumerate(axes):
        for e in range(m):
            ax.hist(coef[:, i, start + e], bins=bins, alpha=0.55,
                    color=colors[e % len(colors)], edgecolor=PALETTE["median"], linewidth=0.2,
                    label=exog_names[e])
        ax.set_title(labels[i])
    _bottom_legend(fig, axes[0], m)
    fig.suptitle("Posterior distributions of exogenous coefficients", fontweight="bold")
    return fig, axes


@styled
def generate_panel_coeff_plots(self):
    """
    Posterior densities of the pooled mean lag coefficients and of the pooling
    parameter of a BayesianPanelVAR.

    Returns
    -------
    (pooling, lags) : ``(fig, ax)`` for the pooling parameter and a list with
        one ``(fig, axes)`` per lag.
    """
    if not hasattr(self, "bbar_draws") or len(self.bbar_draws) == 0:
        raise ValueError("Posterior draws for pooled mean coefficients are not available.")
    bbar = np.asarray(self.bbar_draws)
    n, p = self.n_endo, self.lags
    if bbar.shape[1] != n * n * p:
        raise ValueError("Mismatch between `bbar_draws` and pooled mean coefficient labels.")
    names = [str(v) for v in self.names]

    lags = []
    for lag in range(1, p + 1):
        fig, axes = _panels(n * n, n, panel=(4.0, 2.6))
        for i in range(n):
            for j in range(n):
                ax = axes[i * n + j]
                _density(ax, bbar[:, i * n * p + (lag - 1) * n + j], PALETTE["density"])
                ax.set_title(f"{names[j]} -> eq. {names[i]}", fontsize=9)
        fig.suptitle(f"Pooled mean coefficients (lag {lag})", fontweight="bold")
        lags.append((fig, axes))

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    _density(ax, np.asarray(self.lambda_draws), PALETTE["accent"])
    ax.set_title("Posterior distribution of the pooling parameter")
    return (fig, ax), lags


# ----------------------------------------------------------------------------
# Structural analysis and forecasts
# ----------------------------------------------------------------------------

@styled
def generate_irf_plots(self, cred_interval=(0.68, 0.95)):
    """
    One figure per shock with the responses of every variable: posterior
    (or bootstrap) median and bands.

    ``self.ir_draws`` has shape ``(draws, horizon, variable, shock)``.

    Returns
    -------
    list of (fig, axes), one per shock
    """
    levels = _levels(cred_interval)
    irf = np.asarray(self.ir_draws)
    H, n = irf.shape[1], self.n_endo
    names = [str(v) for v in self.names]
    shock_label = "unit" if self.irf_1std == 0 else "1 s.d."
    plot_label = getattr(self, "plot_label", None)
    horizons = np.arange(H)

    figures = []
    for k in range(n):
        fig, axes = _panels(n, 2)
        for i, ax in enumerate(axes):
            draws = irf[:, :, i, k]
            for level in levels:
                lower = np.percentile(draws, 50 - 50 * level, axis=0)
                upper = np.percentile(draws, 50 + 50 * level, axis=0)
                ax.fill_between(horizons, lower, upper, color=PALETTE["band"],
                                alpha=_band_alpha(level), lw=0, label=_band_label(level))
            ax.plot(horizons, np.percentile(draws, 50, axis=0), color=PALETTE["median"],
                    lw=1.6, label="median")
            ax.axhline(0, color=PALETTE["gray"], lw=0.8, ls="--")
            ax.set_title(names[i])
            ax.set_xlabel("Horizon")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].set_ylabel("Response")
        axes[0].legend(fontsize=8)
        title = f"Impulse responses to a {shock_label} shock in {names[k]}"
        if plot_label is not None:
            title += f" ({plot_label})"
        fig.suptitle(title, fontweight="bold")
        figures.append((fig, axes))
    return figures


@styled
def generate_fevd_plot(self, series_titles=None, shock_titles=None, title=None,
                       color_palette=None):
    """
    Forecast error variance decomposition as stacked areas, one panel per
    variable. ``self.fevd`` has shape ``(steps, shocks, variables)`` in percent.

    Returns
    -------
    (fig, axes)
    """
    fevd = np.asarray(self.fevd)
    steps, n_shocks, n_vars = fevd.shape
    var_labels = list(series_titles) if series_titles else [str(v) for v in self.names]
    shock_labels = list(shock_titles) if shock_titles else [f"Shock {s + 1}" for s in range(n_shocks)]
    colors = _stack_colors(color_palette, n_shocks)
    horizon = np.arange(1, steps + 1)

    fig, axes = _panels(n_vars, 2)
    for v, ax in enumerate(axes):
        ax.stackplot(horizon, fevd[:, :, v].T, colors=colors, labels=shock_labels,
                     alpha=0.9, lw=0)
        ax.set_xlim(1, steps)
        ax.set_ylim(0, 100)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(var_labels[v])
        ax.set_xlabel("Horizon (steps)")
    axes[0].set_ylabel("Contribution to FEVD (%)")
    _bottom_legend(fig, axes[0], n_shocks)
    fig.suptitle(title or "Forecast error variance decomposition", fontweight="bold")
    return fig, axes


@styled
def generate_hd_plot(self, series_titles=None, shock_titles=None, title=None,
                     color_palette=None):
    """
    Historical decomposition: stacked bars by shock (positive and negative
    contributions stacked separately) and the total as a line.
    ``self.HD['shock']`` has shape ``(T, shocks, variables)``.

    Returns
    -------
    (fig, axes)
    """
    hd = np.asarray(self.HD["shock"])
    T, n_shocks, n_vars = hd.shape
    dates = self.HD.get("dates")
    dates = pd.DatetimeIndex(pd.to_datetime(self.dates if dates is None else dates))[-T:]
    var_labels = list(series_titles) if series_titles else [str(v) for v in self.names]
    shock_labels = list(shock_titles) if shock_titles else [f"Shock {s + 1}" for s in range(n_shocks)]
    colors = _stack_colors(color_palette, n_shocks)
    step = np.median(np.diff(dates.values).astype("timedelta64[D]").astype(float)) if T > 1 else 30.0
    width = 0.8 * step

    fig, axes = _panels(n_vars, 2)
    for v, ax in enumerate(axes):
        positive, negative = np.zeros(T), np.zeros(T)
        for s in range(n_shocks):
            value = hd[:, s, v]
            up, down = np.clip(value, 0, None), np.clip(value, None, 0)
            ax.bar(dates, up, width=width, bottom=positive, color=colors[s],
                   label=shock_labels[s], lw=0)
            ax.bar(dates, down, width=width, bottom=negative, color=colors[s], lw=0)
            positive += up
            negative += down
        ax.plot(dates, hd[:, :, v].sum(axis=1), color=PALETTE["median"], lw=1.2, label="Total")
        ax.axhline(0, color=PALETTE["gray"], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(var_labels[v])
        _date_axis(ax)
    axes[0].set_ylabel("Shock contribution")
    _bottom_legend(fig, axes[0], n_shocks + 1)
    fig.suptitle(title or "Historical decomposition", fontweight="bold")
    return fig, axes


@styled
def generate_forecast_plots(self, forecasts: np.ndarray, cred_interval=(0.68, 0.95),
                            last_k: int = None, n_breaks: int = 10, zero_line: bool = False,
                            forecast_type: str = "Unconditional"):
    """
    Observed data and the forecast distribution, one panel per variable.

    Parameters
    ----------
    forecasts : array (n_draws, steps, n_endo)
    cred_interval : level or list of levels for the bands, e.g. [0.68, 0.95]
    last_k : show only the last ``last_k`` observations (default: all)
    n_breaks : approximate number of date ticks
    zero_line : draw a horizontal line at zero
    forecast_type : "Unconditional" or "Conditional" (sets title and colour)

    Returns
    -------
    (fig, axes)
    """
    forecasts = np.asarray(forecasts)
    _, steps, n_endo = forecasts.shape
    levels = _levels(cred_interval)
    hist_dates = pd.DatetimeIndex(self.yy_dates)
    yy = np.asarray(self.yy)
    if last_k is not None:
        yy, hist_dates = yy[-last_k:], hist_dates[-last_k:]
    freq = pd.infer_freq(hist_dates[:5]) or "QS"
    future = pd.date_range(start=hist_dates[-1], periods=steps + 1, freq=freq)[1:]
    edge = hist_dates[-1:].append(future)                   # connect to the last observation
    color = PALETTE["band"] if forecast_type.startswith("Unconditional") else PALETTE["conditional"]
    names = [str(v) for v in self.names]

    fig, axes = _panels(n_endo, 2)
    for i, ax in enumerate(axes):
        last = yy[-1, i]
        draws = forecasts[:, :, i]
        for level in levels:
            lower = np.r_[last, np.percentile(draws, 50 - 50 * level, axis=0)]
            upper = np.r_[last, np.percentile(draws, 50 + 50 * level, axis=0)]
            ax.fill_between(edge, lower, upper, color=color, alpha=_band_alpha(level), lw=0,
                            label=_band_label(level))
        ax.plot(hist_dates, yy[:, i], color=PALETTE["median"], lw=1.2, label="observed")
        ax.plot(edge, np.r_[last, np.percentile(draws, 50, axis=0)], color=color, lw=1.6,
                label="median")
        if zero_line:
            ax.axhline(0, color=PALETTE["gray"], lw=0.8, ls="--")
        ax.set_title(names[i])
        _date_axis(ax, n_breaks)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{forecast_type} forecast", fontweight="bold")
    return fig, axes
