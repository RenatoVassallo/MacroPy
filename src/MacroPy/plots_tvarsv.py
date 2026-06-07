"""
Plotting helpers for the Threshold VAR-SV class (Alessandri & Mumtaz, 2019).

The figures are written to be **general** (any threshold variable, any pair of
regimes, any dataset), with sensible defaults and fully customizable labels:

* :func:`plot_regimes`    - the threshold variable with one regime shaded
  (paper Fig. 1);
* :func:`plot_volatility` - the common stochastic-volatility factor (Fig. 2);
* :func:`plot_irfs`       - generalized impulse responses, regime 1 vs. regime 2,
  for the uncertainty shock *or* any recursive (Cholesky) structural shock.

Styling follows :mod:`MacroPy.plots_kalman` (BSE palette, matplotlib). Every
function accepts a ``savepath`` to write a tight-bbox vector file for LaTeX.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .plots_kalman import (
    set_bse_style, _save, BSE_TEAL, BSE_ORANGE, BSE_NAVY, BSE_GRAY,
)


def plot_regimes(model, shade_regime: int = 2, prob_threshold: float = 0.5,
                 show_prob: bool = True, regime_label: Optional[str] = None,
                 prob_label: Optional[str] = None, var_label: Optional[str] = None,
                 title: str = "Estimated regimes",
                 savepath: Optional[str] = None):
    """Plot the threshold variable with one regime's periods shaded.

    Works for any threshold variable / dataset. Gray bands mark the dates where
    the posterior probability of the selected regime exceeds ``prob_threshold``;
    the dashed line is the posterior-median threshold ``Z*``.

    Parameters
    ----------
    shade_regime : {1, 2}, default 2
        Which regime to shade. ``2`` is the high regime (threshold variable above
        ``Z*``); ``1`` is the low regime (at or below ``Z*``). Use ``1`` to
        highlight, e.g., low-confidence / pessimism episodes.
    prob_threshold : float, default 0.5
        Posterior-probability cut-off used to shade the selected regime.
    show_prob : bool, default True
        Overlay the posterior probability of the selected regime.
    regime_label, prob_label, var_label : str, optional
        Custom labels (defaults are derived from ``shade_regime`` and the column
        name).
    title : str
        Figure title.
    """
    model._require_draws()
    set_bse_style()
    p2 = model.regime_probability()              # P(regime 2) = P(above threshold)
    prob = p2 if shade_regime == 2 else (1.0 - p2)
    if regime_label is None:
        regime_label = ("Regime 2 (above threshold)" if shade_regime == 2
                        else "Regime 1 (at/below threshold)")
    if prob_label is None:
        prob_label = f"P(regime {shade_regime})"

    name = var_label or model.threshold_name
    series = model.y_df[model.threshold_name].reindex(prob.index)
    zstar = float(np.median(model.draws["tar"]))
    shaded = prob.values > prob_threshold

    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    ylo, yhi = float(series.min()) - 0.3, float(series.max()) + 0.3
    ax.fill_between(prob.index, ylo, yhi, where=shaded, color=BSE_GRAY, alpha=0.22,
                    step="mid", label=regime_label)
    ax.plot(prob.index, series.values, color=BSE_TEAL, lw=1.2, label=name)
    ax.axhline(zstar, color=BSE_ORANGE, lw=1.6, ls="--",
               label=f"Threshold $Z^*$ = {zstar:.2f}")
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel(name)
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=3)

    if show_prob:
        ax2 = ax.twinx()
        ax2.fill_between(prob.index, 0, prob.values, color=BSE_NAVY, alpha=0.12)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel(prob_label)
        ax2.grid(False)
    _save(fig, savepath)
    return fig, ax


def plot_volatility(model, cred: float = 0.68, log_scale: bool = False,
                    ylabel: str = r"$\lambda_t$",
                    title: str = r"Common stochastic volatility  $\lambda_t$",
                    savepath: Optional[str] = None):
    """Plot the posterior common volatility factor ``lambda_t`` (paper Fig. 2).

    This figure is already dataset-agnostic: it shows the single scalar volatility
    factor with a credible band. Use ``log_scale=True`` when the factor spans
    several orders of magnitude.
    """
    model._require_draws()
    set_bse_style()
    vp = model.volatility_path(cred=cred)
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    ax.fill_between(vp.index, vp["lower"], vp["upper"], color=BSE_ORANGE,
                    alpha=0.20, label=f"{int(cred * 100)}% band")
    ax.plot(vp.index, vp["median"], color=BSE_ORANGE, lw=1.7, label="median")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(loc="upper left")
    _save(fig, savepath)
    return fig, ax


def plot_irfs(model, shock: Union[str, int] = "uncertainty",
              response_vars: Optional[List[Union[str, int]]] = None,
              cred: float = 0.68,
              regime_labels: Tuple[str, str] = ("Regime 1", "Regime 2"),
              series_titles: Optional[List[str]] = None, ncol: int = 2,
              title: Optional[str] = None, savepath: Optional[str] = None):
    """Plot generalized impulse responses, regime 1 vs. regime 2 (paper Fig. 3).

    Parameters
    ----------
    shock : str or int, default "uncertainty"
        ``"uncertainty"`` for the volatility shock, or a variable name / index for
        a recursive (Cholesky) structural shock to that variable. The latter
        requires ``compute_irfs(shock="level")`` to have been run.
    response_vars : list of str or int, optional
        Variables whose responses are plotted (default: all endogenous variables).
    cred : float, default 0.68
        Width of the posterior credible band.
    regime_labels : (str, str)
        Legend labels for (regime 1, regime 2).
    series_titles : list of str, optional
        Panel titles (default: the response-variable names).
    """
    if not hasattr(model, "irf"):
        raise RuntimeError("Call compute_irfs() before plot_irfs().")
    set_bse_style()
    names = list(model.names)

    # response-variable selection
    if response_vars is None:
        resp_idx = list(range(model.N))
    else:
        resp_idx = [names.index(v) if isinstance(v, str) else int(v)
                    for v in response_vars]
    resp_names = series_titles or [names[i] for i in resp_idx]

    # shock selection: uncertainty vs. recursive (level) shock to a variable
    if isinstance(shock, str) and shock == "uncertainty":
        key, shock_title = "uncertainty", "uncertainty"

        def fetch(reg, v):
            return model.irf[reg]["uncertainty"][:, :, v]
    else:
        s = names.index(shock) if isinstance(shock, str) else int(shock)
        shock_title = names[s]
        for reg in ("regime1", "regime2"):
            if model.irf.get(reg) is not None and "level" not in model.irf[reg]:
                raise RuntimeError(
                    "Level IRFs not available: run compute_irfs(shock='level').")

        def fetch(reg, v):
            return model.irf[reg]["level"][:, :, v, s]

    # horizon from the first available regime
    ref = model.irf.get("regime1") or model.irf.get("regime2")
    hor = (ref["uncertainty"] if shock == "uncertainty" else ref["level"]).shape[1]
    h = np.arange(hor)
    lo = (1 - cred) / 2

    nplot = len(resp_idx)
    nrow = int(np.ceil(nplot / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.0 * nrow),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()

    style = [("regime1", BSE_NAVY, regime_labels[0]),
             ("regime2", BSE_ORANGE, regime_labels[1])]
    for p, v in enumerate(resp_idx):
        ax = axes[p]
        ax.axhline(0, color=BSE_GRAY, lw=0.7)
        for reg, color, lab in style:
            if model.irf.get(reg) is None:
                continue
            a = fetch(reg, v)
            med = np.median(a, axis=0)
            band = np.percentile(a, [100 * lo, 100 * (1 - lo)], axis=0)
            ax.plot(h, med, color=color, lw=1.8, label=lab)
            ax.fill_between(h, band[0], band[1], color=color, alpha=0.16)
        ax.set_title(resp_names[p])
        ax.set_xlabel("Horizon")
        if p == 0:
            ax.legend(loc="best")
    for k in range(nplot, len(axes)):
        axes[k].axis("off")
    fig.suptitle(title or f"Response to a {shock_title} shock",
                 color=BSE_TEAL, weight="bold")
    _save(fig, savepath)
    return fig, axes


__all__ = ["plot_regimes", "plot_volatility", "plot_irfs"]
