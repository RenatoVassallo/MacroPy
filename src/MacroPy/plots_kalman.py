"""
Plotting helpers for trend-cycle decomposition (Kalman class).

All figures use the BSE palette so they drop into the beamer deck without
re-styling. Functions accept a `savepath` argument that, when given,
writes a tight-bbox PDF (vector) suitable for `\includegraphics` in LaTeX.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# BSE palette (matches beamerthemebse.sty)
BSE_TEAL = "#233746"
BSE_TEAL_SOFT = "#2D4253"
BSE_ORANGE = "#E78033"
BSE_NAVY = "#1F3D5C"
BSE_GRAY = "#4A4A4A"
BSE_LIGHT = "#F4F4F4"


def set_bse_style() -> None:
    """Apply matplotlib rcParams matching the BSE beamer theme."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": BSE_TEAL,
        "axes.labelcolor": BSE_GRAY,
        "axes.edgecolor": BSE_GRAY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "xtick.color": BSE_GRAY,
        "ytick.color": BSE_GRAY,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def _save(fig, savepath: Optional[str]) -> None:
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")


def plot_series_with_trend(time, y, trend, *, gap=None, y_label: str = "Log GDP × 100",
                           title: str = "", savepath: Optional[str] = None,
                           trend_band: Optional[tuple] = None):
    """Two-panel layout: observed + trend (left), gap or cycle (right)."""
    set_bse_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    axes[0].plot(time, y, color="black", lw=1.0, label="Observado")
    axes[0].plot(time, trend, color=BSE_ORANGE, lw=1.8, label="Tendencia")
    if trend_band is not None:
        lo, hi = trend_band
        axes[0].fill_between(time, lo, hi, color=BSE_ORANGE, alpha=0.18,
                              label="IC 95%")
    axes[0].set_title("Nivel y tendencia")
    axes[0].set_ylabel(y_label)
    axes[0].legend(loc="best")

    if gap is not None:
        axes[1].axhline(0, color=BSE_GRAY, lw=0.7)
        axes[1].plot(time, gap, color=BSE_NAVY, lw=1.5)
        axes[1].set_title("Brecha (ciclo)")
        axes[1].set_ylabel("Desvío %")
    if title:
        fig.suptitle(title, color=BSE_TEAL, weight="bold")
    _save(fig, savepath)
    return fig, axes


def plot_decomposition(time, *, trend, cycle, seasonal=None, irregular=None,
                       y=None, title: str = "",
                       savepath: Optional[str] = None):
    """Two-by-two grid showing each component of the decomposition."""
    set_bse_style()
    n_panels = 2 + int(seasonal is not None) + int(irregular is not None)
    nrows = 2 if n_panels >= 3 else 1
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(11, 3.6 * nrows),
                           constrained_layout=True)
    ax = np.atleast_2d(ax).ravel()

    if y is not None:
        ax[0].plot(time, y, color="black", lw=1.0, label="Observado")
    ax[0].plot(time, trend, color=BSE_ORANGE, lw=1.8, label="Tendencia")
    ax[0].set_title("Tendencia")
    ax[0].legend(loc="best")

    ax[1].axhline(0, color=BSE_GRAY, lw=0.7)
    ax[1].plot(time, cycle, color=BSE_NAVY, lw=1.5)
    ax[1].set_title("Ciclo")

    idx = 2
    if seasonal is not None:
        ax[idx].axhline(0, color=BSE_GRAY, lw=0.7)
        ax[idx].plot(time, seasonal, color="#7E57C2", lw=1.3)
        ax[idx].set_title("Componente estacional")
        idx += 1
    if irregular is not None:
        ax[idx].axhline(0, color=BSE_GRAY, lw=0.7)
        ax[idx].plot(time, irregular, color="#4CAF50", lw=1.0)
        ax[idx].set_title("Irregular")
        idx += 1
    for k in range(idx, len(ax)):
        ax[k].axis("off")
    if title:
        fig.suptitle(title, color=BSE_TEAL, weight="bold")
    _save(fig, savepath)
    return fig, ax


def plot_with_bands(time, mean, std, *, level: float = 1.96, label: str = "",
                    color: str = BSE_ORANGE, ax=None,
                    savepath: Optional[str] = None):
    """Line with symmetric confidence/credibility band."""
    if ax is None:
        set_bse_style()
        fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    else:
        fig = ax.figure
    ax.plot(time, mean, color=color, lw=1.8, label=label)
    ax.fill_between(time, mean - level * std, mean + level * std,
                    color=color, alpha=0.18)
    if label:
        ax.legend(loc="best")
    _save(fig, savepath)
    return fig, ax


def plot_kalman_step(t_idx: int, y, a_pred, P_pred, a_filt, P_filt,
                     state_idx: int = 0, window: int = 30,
                     savepath: Optional[str] = None):
    """
    Snapshot of one Kalman update at time `t_idx`: shows the prior (predicted)
    state density, the new observation, and the posterior (filtered) density.
    Useful for animating the recursion in a slide.
    """
    set_bse_style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
    t0 = max(0, t_idx - window)
    rng = np.arange(t0, t_idx + 1)
    ax[0].plot(rng, y[t0:t_idx + 1], "ko-", ms=4, lw=0.8, label="y observado")
    ax[0].plot(rng, a_filt[state_idx, t0:t_idx + 1], color=BSE_ORANGE, lw=1.5,
               label="Estado filtrado")
    ax[0].axvline(t_idx, color=BSE_GRAY, ls=":", lw=0.8)
    ax[0].set_title(f"Filtro en t = {t_idx}")
    ax[0].legend(loc="best")

    xs = np.linspace(a_pred[state_idx, t_idx] - 4 * np.sqrt(P_pred[state_idx, state_idx, t_idx]),
                     a_pred[state_idx, t_idx] + 4 * np.sqrt(P_pred[state_idx, state_idx, t_idx]),
                     200)
    prior = _gauss_pdf(xs, a_pred[state_idx, t_idx], np.sqrt(P_pred[state_idx, state_idx, t_idx]))
    post = _gauss_pdf(xs, a_filt[state_idx, t_idx], np.sqrt(P_filt[state_idx, state_idx, t_idx]))
    ax[1].plot(xs, prior, color=BSE_NAVY, lw=1.5, label="Predicho (a priori)")
    ax[1].plot(xs, post, color=BSE_ORANGE, lw=1.8, label="Actualizado (a posteriori)")
    ax[1].axvline(y[t_idx], color="black", ls="--", lw=0.8, label="y_t")
    ax[1].set_title("Densidad del estado")
    ax[1].set_xlabel("Valor del estado")
    ax[1].legend(loc="best")
    _save(fig, savepath)
    return fig, ax


def _gauss_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def plot_posterior_traces(draws: dict, savepath: Optional[str] = None):
    """Trace and posterior histogram for each scalar parameter in `draws`."""
    set_bse_style()
    keys = list(draws.keys())
    n = len(keys)
    fig, ax = plt.subplots(n, 2, figsize=(10, 1.8 * n), constrained_layout=True)
    if n == 1:
        ax = np.atleast_2d(ax)
    for i, k in enumerate(keys):
        ax[i, 0].plot(draws[k], color=BSE_NAVY, lw=0.6)
        ax[i, 0].set_title(f"Traza: {k}")
        ax[i, 1].hist(draws[k], bins=40, color=BSE_ORANGE, alpha=0.8)
        ax[i, 1].set_title(f"Posterior: {k}")
    _save(fig, savepath)
    return fig, ax


__all__ = [
    "set_bse_style",
    "plot_series_with_trend",
    "plot_decomposition",
    "plot_with_bands",
    "plot_kalman_step",
    "plot_posterior_traces",
    "BSE_TEAL", "BSE_ORANGE", "BSE_NAVY", "BSE_GRAY",
]
