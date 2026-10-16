"""
Figures for :class:`MacroPy.TimeVaryingVAR` (Chan & Eisenstat, 2018), in the
MacroPy style (:mod:`MacroPy.style`).

* :func:`plot_tvp_volatility` - structural shock volatilities over time;
* :func:`plot_tvp_coefficients` - structural coefficients over time;
* :func:`plot_tvp_irfs`       - IRFs at selected dates, median and bands;
* :func:`plot_irf_surface`    - 3D surface of the median IRF (date x horizon);
* :func:`plot_irf_over_time`  - the response at fixed horizons, date by date;
* :func:`plot_fevd_over_time` - FEVD shares at one horizon over time;
* :func:`plot_tvp_fan_chart`  - BCRP-style fan charts of the predictive density.

The historical decomposition uses :func:`MacroPy.generate_hd_plot`. Every
function runs inside ``macropy_style()``, returns ``(fig, axes)`` and accepts a
``savepath``; nothing is displayed or computed here (call the model's
``compute_*`` or ``forecast`` method first).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from matplotlib.colors import TwoSlopeNorm

from ._plotting import _bottom_legend, _date_axis, _panels, _save, _stack_colors
from .plots_uc import plot_fan_chart
from .style import PALETTE, colormap, styled

__all__ = [
    "plot_tvp_volatility", "plot_tvp_coefficients", "plot_tvp_irfs", "plot_irf_surface", "plot_irf_over_time",
    "plot_fevd_over_time", "plot_tvp_fan_chart",
]


def _index(model, which: Union[int, str]) -> int:
    if isinstance(which, str):
        return model.names.index(which)
    return int(which) % model.n_endo


def _labels(model, titles):
    return list(titles) if titles else list(model.names)


def _require(model, attr: str, method: str):
    if getattr(model, attr, None) is None:
        raise RuntimeError(f"Call {method} before plotting.")


@styled
def plot_tvp_volatility(model, cred: float = 0.68, series_titles=None,
                        title: Optional[str] = None, savepath: Optional[str] = None):
    """Posterior median and band of the structural shock standard deviations
    ``exp(h_t / 2)``, one panel per equation (flat for constant variances)."""
    frame = model.volatility_frame(cred=cred)
    labels = _labels(model, series_titles)
    fig, axes = _panels(model.n_endo, min(model.n_endo, 3), panel=(4.6, 2.9))
    for ax, name, label in zip(axes, model.names, labels):
        f = frame[name]
        ax.fill_between(f.index, f["lower"], f["upper"], color=PALETTE["band"], alpha=0.25,
                        lw=0, label=f"{100 * cred:g} %")
        ax.plot(f.index, f["median"], color=PALETTE["median"], lw=1.4, label="median")
        ax.set_title(label)
        _date_axis(ax)
    axes[0].legend(fontsize=8)
    fig.suptitle(title or f"Structural shock volatility ({model.model})", fontweight="bold")
    _save(fig, savepath)
    return fig, axes


@styled
def plot_tvp_coefficients(model, block: str = "impact", cred: float = 0.68,
                          title: Optional[str] = None, ncol: int = 3,
                          savepath: Optional[str] = None):
    """Posterior median and band of the structural coefficients over time:
    ``block`` is ``"intercept"``, ``"impact"`` (free elements of ``A_t``) or
    ``"lags"``. Blocks that are constant in the chosen model appear flat."""
    frame = model.coefficient_frame(block=block, cred=cred)
    labels = list(dict.fromkeys(frame.columns.get_level_values(0)))
    fig, axes = _panels(len(labels), min(ncol, len(labels)), panel=(4.2, 2.6))
    for ax, label in zip(axes, labels):
        f = frame[label]
        ax.fill_between(f.index, f["lower"], f["upper"], color=PALETTE["band"], alpha=0.25, lw=0)
        ax.plot(f.index, f["median"], color=PALETTE["median"], lw=1.3)
        ax.set_title(label, fontsize=10)
        _date_axis(ax)
    names = {"intercept": "intercepts", "impact": "contemporaneous coefficients (A_t)",
             "lags": "lag coefficients"}
    fig.suptitle(title or f"Structural {names[block]}, median and {100 * cred:g} % band "
                          f"({model.model})", fontweight="bold")
    _save(fig, savepath)
    return fig, axes


@styled
def plot_tvp_irfs(model, dates: Optional[Sequence] = None, shock: Union[int, str] = -1,
                  cred: Optional[float] = None, series_titles=None, shock_titles=None,
                  title: Optional[str] = None, savepath: Optional[str] = None):
    """
    Responses of every variable to one structural shock at selected dates:
    posterior median (line) and credible band (shaded), one colour per date.

    ``dates`` are any dates of the estimation sample (default: those whose
    draws ``compute_irfs`` stored); ``shock`` an index or a variable name.
    """
    _require(model, "irf", "compute_irfs()")
    irf = model.irf
    cred = irf["levels"][0] if cred is None else cred
    if cred not in irf["bands"]:
        raise ValueError(f"Band {cred} not computed; available: {list(irf['bands'])}.")
    dates = list(irf["draws"]) if dates is None else list(dates)
    positions = [model._date_position(d) for d in dates]
    j = _index(model, shock)
    labels = _labels(model, series_titles)
    shock_label = (shock_titles[j] if shock_titles else model.names[j])
    colors = _stack_colors(None, len(positions))
    lower, upper = irf["bands"][cred]
    hor = irf["median"].shape[1]
    steps = np.arange(hor)
    fig, axes = _panels(model.n_endo, min(model.n_endo, 3), panel=(4.6, 3.0))
    for i, ax in enumerate(axes):
        for pos, color in zip(positions, colors):
            ax.fill_between(steps, lower[pos, :, i, j], upper[pos, :, i, j], color=color,
                            alpha=0.18, lw=0)
            ax.plot(steps, irf["median"][pos, :, i, j], color=color, lw=1.7,
                    label=model._date_label(pos))
        ax.axhline(0, color=PALETTE["gray"], lw=0.8, ls="--")
        ax.set_title(labels[i])
        ax.set_xlabel("Horizon")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(fontsize=8)
    size = "unit" if model.irf_1std == 0 else "1 s.d."
    fig.suptitle(title or f"Responses to a {size} shock in {shock_label} "
                          f"(median and {100 * cred:g} % band)", fontweight="bold")
    _save(fig, savepath)
    return fig, axes


def _decimal_years(index: pd.DatetimeIndex) -> np.ndarray:
    return index.year + (index.dayofyear - 1) / 365.25


@styled
def plot_irf_surface(model, response: Union[int, str] = 0, shock: Union[int, str] = -1,
                     cmap=None, elev: float = 28.0, azim: float = -58.0,
                     title: Optional[str] = None, savepath: Optional[str] = None):
    """3D surface of the posterior-median response of ``response`` to ``shock``
    over dates (x) and horizons (y). The default colour map is MacroPy's
    diverging one centred at zero (navy negative, accent positive)."""
    _require(model, "irf", "compute_irfs()")
    i, j = _index(model, response), _index(model, shock)
    med = model.irf["median"][:, :, i, j]                      # (T, hor)
    x = _decimal_years(pd.DatetimeIndex(model.irf["dates"]))
    hor = med.shape[1]
    Xg, Yg = np.meshgrid(x, np.arange(hor), indexing="ij")
    fig = plt.figure(figsize=(8.5, 6.0), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    norm = None
    if cmap is None:
        cmap = colormap("diverging")
        bound = float(np.max(np.abs(med))) or 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
    ax.plot_surface(Xg, Yg, med, cmap=cmap, norm=norm, linewidth=0, antialiased=True,
                    alpha=0.95)
    ax.set_xlabel("Date")
    ax.set_ylabel("Horizon")
    ax.set_zlabel("Response")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    fig.suptitle(title or f"Response of {model.names[i]} to a shock in {model.names[j]}",
                 fontweight="bold")
    _save(fig, savepath)
    return fig, ax


@styled
def plot_irf_over_time(model, response: Union[int, str] = 0, shock: Union[int, str] = -1,
                       horizons: Sequence[int] = (0, 4, 8), cred: Optional[float] = None,
                       title: Optional[str] = None, savepath: Optional[str] = None):
    """The response at fixed horizons, date by date (median and band): a
    readable cut of the 3D surface."""
    _require(model, "irf", "compute_irfs()")
    irf = model.irf
    cred = irf["levels"][0] if cred is None else cred
    i, j = _index(model, response), _index(model, shock)
    lower, upper = irf["bands"][cred]
    dates = pd.DatetimeIndex(irf["dates"])
    colors = _stack_colors(None, len(horizons))
    fig, ax = plt.subplots(figsize=(9.5, 3.8), constrained_layout=True)
    for h, color in zip(horizons, colors):
        ax.fill_between(dates, lower[:, h, i, j], upper[:, h, i, j], color=color,
                        alpha=0.16, lw=0)
        ax.plot(dates, irf["median"][:, h, i, j], color=color, lw=1.6, label=f"h = {h}")
    ax.axhline(0, color=PALETTE["gray"], lw=0.8, ls="--")
    _date_axis(ax)
    ax.legend(fontsize=8, ncol=len(horizons))
    ax.set_title(title or f"Response of {model.names[i]} to a shock in {model.names[j]}, "
                          f"by date ({100 * cred:g} % band)")
    _save(fig, savepath)
    return fig, ax


@styled
def plot_fevd_over_time(model, horizon: int = 8, series_titles=None, shock_titles=None,
                        title: Optional[str] = None, color_palette=None,
                        savepath: Optional[str] = None):
    """Share of each shock in the ``horizon``-step forecast error variance of
    each variable, date by date (stacked areas, posterior mean)."""
    _require(model, "fevd", "compute_fevd()")
    fevd = np.asarray(model.fevd)                              # (T, hor, shock, var)
    if not 1 <= horizon <= fevd.shape[1]:
        raise ValueError(f"horizon must be between 1 and {fevd.shape[1]}.")
    n = model.n_endo
    labels = _labels(model, series_titles)
    shocks = list(shock_titles) if shock_titles else [f"Shock {s}" for s in model.names]
    colors = _stack_colors(color_palette, n)
    dates = pd.DatetimeIndex(model.yy_dates)
    fig, axes = _panels(n, min(n, 3), panel=(4.6, 3.0))
    for v, ax in enumerate(axes):
        ax.stackplot(dates, fevd[:, horizon - 1, :, v].T, colors=colors, labels=shocks,
                     alpha=0.9, lw=0)
        ax.set_ylim(0, 100)
        ax.set_xlim(dates[0], dates[-1])
        ax.set_axisbelow(True)
        ax.set_title(labels[v])
        _date_axis(ax)
    axes[0].set_ylabel("Share of FEV (%)")
    _bottom_legend(fig, axes[0], n)
    fig.suptitle(title or f"Forecast error variance decomposition at {horizon} steps",
                 fontweight="bold")
    _save(fig, savepath)
    return fig, axes


@styled
def plot_tvp_fan_chart(model, variables: Optional[Sequence] = None,
                       bands=(0.90, 0.70, 0.50, 0.30, 0.10), history_from=None,
                       series_titles=None, title: Optional[str] = None,
                       hline: Optional[float] = None, ncol: Optional[int] = None,
                       savepath: Optional[str] = None, **kwargs):
    """
    Fan charts of the predictive density stored by ``forecast()``, one panel
    per variable, with :func:`MacroPy.plot_fan_chart` (nested central bands in
    one hue, continuous with the observed history). Extra keyword arguments
    go to :func:`MacroPy.plot_fan_chart` (``method``, ``smooth``, ``color``...).
    """
    _require(model, "forecasts", "forecast()")
    draws = np.asarray(model.forecasts)
    idx = [_index(model, v) for v in (variables if variables is not None else model.names)]
    labels = _labels(model, series_titles)
    future = model._forecast_index(draws.shape[1])
    if future is None:
        raise ValueError("Forecast dates need a regular (monthly, quarterly or annual) index.")
    fig, axes = _panels(len(idx), ncol or min(len(idx), 2), panel=(5.6, 3.4))
    for ax, i in zip(axes, idx):
        history = pd.Series(model.yy[:, i], index=model.yy_dates)
        plot_fan_chart(history, draws[:, :, i], future, bands=bands, history_from=history_from,
                       ax=ax, title=labels[i], hline=hline,
                       history_label="observed", center_label="median", **kwargs)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        if ax is not axes[0] and ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.suptitle(title or f"Predictive density ({model.model})", fontweight="bold")
    _save(fig, savepath)
    return fig, axes
