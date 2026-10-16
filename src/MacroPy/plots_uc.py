"""
Plotting helpers for the UC trend-cycle class (:mod:`MacroPy.uc`).

* :func:`plot_uc_decomposition` - observed series with the estimated trend
  (left) and the cycle / output gap (right), with credible or confidence
  bands; optional third panel for the trend growth rate (drift).
* :func:`plot_covid_scales`     - posterior COVID variance-scale paths.
* :func:`plot_gap_comparison`   - overlay the cycles implied by several
  fitted models (the "ladder" exhibit).

Every figure uses the MacroPy style (:mod:`MacroPy.style`) and every function
accepts a ``savepath`` to write a tight-bbox vector file for LaTeX.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from ._plotting import _save
from .style import PALETTE, styled

_COMP_LABELS = {"cycle": "Cycle shock", "trend_level": "Trend-level shock",
                "trend_growth": "Trend-growth shock"}


def _shade_spans(ax, shade):
    """Shade (start, end) date spans (e.g. recession dating) on an axis."""
    for start, end in shade or []:
        ax.axvspan(start, end, color=PALETTE["gray"], alpha=0.14, lw=0)


@styled
def plot_uc_decomposition(model, cred: float = 0.68, show_drift: bool = False,
                          y_label: Optional[str] = None, title: str = "",
                          shade=None, labels: Optional[dict] = None,
                          savepath: Optional[str] = None,
                          outer_cred: Optional[float] = None):
    """
    Observed series + trend (left) and cycle (right), with central ``cred``
    bands (posterior bands after ``sample_posterior()``; smoothed-state
    confidence bands after ``fit_mle()``). ``show_drift=True`` adds a third
    panel with the trend growth rate g_t. ``shade`` takes (start, end) date
    pairs (e.g. recession dating) shaded in every panel; ``labels`` overrides
    any of the default text keys
    ``observed/trend/band/outer_band/level_title/cycle_title/drift_title``.

    ``outer_cred`` adds a second, lighter band (e.g. ``cred=0.68`` inside
    ``outer_cred=0.90``) so the reader sees how fast the interval widens with
    the coverage level; the band level is always written in the legend.
    """
    lab = {"observed": "Observed", "trend": "Trend",
           "band": f"{cred:.0%} band", "level_title": "Level and trend",
           "cycle_title": "Cycle", "drift_title": "Trend growth (drift)"}
    if outer_cred:
        lab["outer_band"] = f"{outer_cred:.0%} band"
    lab.update(labels or {})
    frame = model.decomposition_frame(cred=cred)
    outer = model.decomposition_frame(cred=outer_cred) if outer_cred else None
    ncols = 3 if show_drift else 2
    fig, axes = plt.subplots(1, ncols, figsize=(11 if ncols == 2 else 14, 3.6),
                             constrained_layout=True)
    time = frame.index

    axes[0].plot(time, model.y, color=PALETTE["median"], lw=1.0, label=lab["observed"])
    axes[0].plot(time, frame["trend"], color=PALETTE["accent"], lw=1.8,
                 label=lab["trend"])
    if outer is not None:
        axes[0].fill_between(time, outer["trend_lower"], outer["trend_upper"],
                             color=PALETTE["accent"], alpha=0.10,
                             label=lab["outer_band"])
    axes[0].fill_between(time, frame["trend_lower"], frame["trend_upper"],
                         color=PALETTE["accent"], alpha=0.18, label=lab["band"])
    axes[0].set_title(lab["level_title"])
    if y_label:
        axes[0].set_ylabel(y_label)
    axes[0].legend(loc="best")

    axes[1].axhline(0, color=PALETTE["gray"], lw=0.7)
    axes[1].plot(time, frame["cycle"], color=PALETTE["navy"], lw=1.5)
    if outer is not None:
        axes[1].fill_between(time, outer["cycle_lower"], outer["cycle_upper"],
                             color=PALETTE["navy"], alpha=0.09)
    axes[1].fill_between(time, frame["cycle_lower"], frame["cycle_upper"],
                         color=PALETTE["navy"], alpha=0.15)
    axes[1].set_title(lab["cycle_title"])

    if show_drift:
        axes[2].plot(time, frame["drift"], color=PALETTE["ink"], lw=1.5)
        if outer is not None:
            axes[2].fill_between(time, outer["drift_lower"],
                                 outer["drift_upper"], color=PALETTE["ink"],
                                 alpha=0.09)
        axes[2].fill_between(time, frame["drift_lower"], frame["drift_upper"],
                             color=PALETTE["ink"], alpha=0.15)
        axes[2].set_title(lab["drift_title"])

    for ax in axes:
        _shade_spans(ax, shade)
    if title:
        fig.suptitle(title, color=PALETTE["ink"], weight="bold")
    _save(fig, savepath)
    return fig, axes


@styled
def plot_covid_scales(model, cred: float = 0.68, log_scale: bool = True,
                      window: int = 16, title: str = "COVID variance scales",
                      savepath: Optional[str] = None):
    """
    Posterior median and central ``cred`` band of the variance-scale paths
    ``lambda_{i,t}`` for each COVID-flagged component, zoomed to the window
    around the pandemic onset (``window`` quarters after t0).
    """
    model._require_draws()
    if not model.covid_components:
        raise ValueError("The model has no COVID components (covid_t0 is None).")
    comps = list(model.covid_components)
    fig, axes = plt.subplots(1, len(comps), figsize=(4.2 * len(comps), 3.4),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    lo_q, hi_q = 100 * (1 - cred) / 2, 100 * (1 + cred) / 2
    t0 = model.covid_t0
    sl = slice(max(0, t0 - 4), min(model.T, t0 + window + 1))
    time = model.dates[sl]

    for ax, comp in zip(axes, comps):
        lam = model.draws[f"lambda_{comp}"][:, sl]
        med = np.median(lam, axis=0)
        ax.plot(time, med, color=PALETTE["accent"], lw=1.8)
        ax.fill_between(time, np.percentile(lam, lo_q, axis=0),
                        np.percentile(lam, hi_q, axis=0),
                        color=PALETTE["accent"], alpha=0.2)
        ax.axhline(1.0, color=PALETTE["gray"], lw=0.8, ls=":")
        ax.set_title(_COMP_LABELS.get(comp, comp))
        if log_scale:
            ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=45)
    if title:
        fig.suptitle(title, color=PALETTE["ink"], weight="bold")
    _save(fig, savepath)
    return fig, axes


@styled
def plot_gap_comparison(models: Dict[str, "object"], cred: Optional[float] = None,
                        title: str = "Output gap across models",
                        shade=None, extra_series: Optional[dict] = None,
                        savepath: Optional[str] = None,
                        outer_cred: Optional[float] = None):
    """
    Overlay the estimated cycles of several fitted UC models (each estimated
    by MLE or by Gibbs). When ``cred`` is given, the LAST model's band is
    shaded as a reference for estimation uncertainty (default 68%), and
    ``outer_cred`` adds a lighter outer band for that same model. ``shade``
    takes (start, end) date pairs (e.g. NBER recessions); ``extra_series``
    maps labels to pd.Series plotted dashed (e.g. the CBO output gap).
    """
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    colors = PALETTE["cycle"]
    ax.axhline(0, color=PALETTE["gray"], lw=0.7)
    _shade_spans(ax, shade)
    items = list(models.items())
    for i, (name, m) in enumerate(items):
        frame = m.decomposition_frame(cred=cred or 0.68)
        ax.plot(frame.index, frame["cycle"], lw=1.5,
                color=colors[i % len(colors)], label=name)
        if cred is not None and i == len(items) - 1:
            if outer_cred:
                outer = m.decomposition_frame(cred=outer_cred)
                ax.fill_between(frame.index, outer["cycle_lower"],
                                outer["cycle_upper"],
                                color=colors[i % len(colors)], alpha=0.08,
                                label=f"{name}: {outer_cred:.0%}")
            ax.fill_between(frame.index, frame["cycle_lower"],
                            frame["cycle_upper"],
                            color=colors[i % len(colors)], alpha=0.15,
                            label=f"{name}: {cred:.0%}")
    for j, (name, series) in enumerate((extra_series or {}).items()):
        ax.plot(series.index, series.to_numpy(), lw=1.4, ls="--",
                color=[PALETTE["median"], PALETTE["conditional"]][j % 2], label=name)
    ax.legend(loc="best")
    ax.set_title(title)
    _save(fig, savepath)
    return fig, ax


def _smooth_ma(x, w: int):
    """Centred moving average with shrinking edge windows (odd w)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    half = int(w) // 2
    out = np.empty(n)
    for i in range(n):
        out[i] = x[max(0, i - half): min(n, i + half) + 1].mean()
    return out


@styled
def plot_fan_chart(history, draws, dates,
                   bands=(0.90, 0.70, 0.50, 0.30, 0.10),
                   method: str = "quantile", smooth: Optional[int] = None,
                   color: Optional[str] = None, center: str = "median",
                   center_label: str = "Central projection",
                   history_label: Optional[str] = None,
                   annotations: Optional[dict] = None,
                   hline: Optional[float] = None, title: str = "",
                   y_label: str = "", history_from=None,
                   ax=None, savepath: Optional[str] = None):
    """
    Central-bank style fan chart (Bank of England / BCRP Inflation Report):
    nested central prediction bands in a single graded hue, continuous with
    the observed history.

    Parameters
    ----------
    history : pd.Series
        Observed values plotted as a line before the fan.
    draws : ndarray (n_draws, H)
        Predictive draws of the plotted variable at ``dates`` (e.g. from
        ``UCTrendCycle.conditional_forecast``, transformed as needed).
    dates : sequence of length H
        Forecast dates.
    bands : tuple of floats
        Nested central coverage levels, e.g. 0.90 = the 5th-95th percentile
        band. Drawn widest first with increasing opacity toward the center.
    method : {"quantile", "normal"}
        Band edges from the empirical quantiles of ``draws`` (default), or
        from a per-horizon normal approximation ``mean +/- z * sd``. The
        normal option removes Monte Carlo jaggedness from the edges while
        keeping the (possibly non-flat) central path; use it when the
        per-horizon predictive is approximately symmetric.
    smooth : odd int, optional
        Light cosmetic smoothing of the band HALF-WIDTHS across horizons
        (centred moving average of that window). The central path is never
        smoothed, so any genuine base-effect zigzag in the projection stays;
        only the band edges are regularised, as in published fan charts.
    center : {"median", "mean", None}
        Central line drawn over the fan.
    annotations : dict {date-like: value}, optional
        Reference markers (e.g. survey targets for annual growth).
    hline : float, optional
        Horizontal reference line (e.g. 0).
    history_from : date-like, optional
        Trim the plotted history start (the fan is unaffected).
    """
    import pandas as pd
    color = PALETTE["conditional"] if color is None else color
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    else:
        fig = ax.figure
    hist = history if history_from is None else history[history.index >= pd.Timestamp(history_from)]
    if hline is not None:
        ax.axhline(hline, color=PALETTE["gray"], lw=0.7)

    # prepend the last observed point so the fan is continuous with history
    dates = pd.DatetimeIndex(dates)
    x = dates.insert(0, hist.index[-1])
    last = float(hist.iloc[-1])
    if center == "mean" or (center is None and method == "normal"):
        path = draws.mean(axis=0)
    elif center == "median":
        path = np.median(draws, axis=0)
    else:
        path = None
    ref = path if path is not None else np.median(draws, axis=0)

    levels = sorted(float(b) for b in bands)      # narrow..wide
    n_levels = len(levels)
    if method == "normal":
        from scipy.stats import norm
        mu, sd = draws.mean(axis=0), draws.std(axis=0, ddof=1)
    for rank, cov in enumerate(sorted(levels, reverse=True)):
        if method == "normal":
            z = norm.ppf((1 + cov) / 2)
            lo, hi = mu - z * sd, mu + z * sd
        else:
            lo = np.quantile(draws, (1 - cov) / 2, axis=0)
            hi = np.quantile(draws, (1 + cov) / 2, axis=0)
        if smooth:
            lo = ref - _smooth_ma(ref - lo, smooth)
            hi = ref + _smooth_ma(hi - ref, smooth)
        alpha = 0.16 + 0.60 * (rank + 1) / n_levels
        ax.fill_between(x, np.insert(lo, 0, last), np.insert(hi, 0, last),
                        color=color, alpha=alpha, lw=0,
                        label=f"{cov:.0%}")
    if path is not None:
        ax.plot(x, np.insert(path, 0, last), color="white", lw=2.4)
        ax.plot(x, np.insert(path, 0, last), color=color, lw=1.4,
                label=center_label)
    ax.plot(hist.index, hist.to_numpy(), color=PALETTE["median"], lw=1.3,
            label=history_label)
    ax.axvline(hist.index[-1], color=PALETTE["gray"], lw=0.8, ls=":")
    for key, val in (annotations or {}).items():
        ax.plot([pd.Timestamp(key)], [val], marker="D", ms=6, color=PALETTE["median"],
                mfc="white", zorder=5)
    handles, labels_ = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels_) if l and l != "_nolegend_"]
    ax.legend(*zip(*keep), loc="best", ncol=2, fontsize=9)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    _save(fig, savepath)
    return fig, ax


__all__ = ["plot_uc_decomposition", "plot_covid_scales", "plot_gap_comparison",
           "plot_fan_chart"]
