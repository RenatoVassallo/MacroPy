"""
Plotting helpers for the threshold VARs: :class:`MacroPy.ThresholdVARSV`
(Alessandri & Mumtaz, 2019) and :class:`MacroPy.ThresholdVAR` (Alessandri &
Mumtaz, 2017).

The figures are written to be **general** (any threshold variable, any pair of
regimes, any dataset), with sensible defaults and fully customizable labels:

* :func:`plot_regimes`    - the threshold variable with one regime shaded
  (paper Fig. 1);
* :func:`plot_volatility` - the common stochastic-volatility factor (Fig. 2,
  ``ThresholdVARSV`` only);
* :func:`plot_irfs`       - generalized impulse responses, regime 1 vs. regime 2,
  for the uncertainty shock *or* any recursive (Cholesky) structural shock.

Side-by-side comparison of several estimated models (``{label: model}``):

* :func:`plot_regime_comparison`     - thresholds and regime probabilities;
* :func:`plot_irf_comparison`        - GIRFs to the same shock, by regime;
* :func:`plot_volatility_comparison` - reduced-form shock volatility over time;
* :func:`plot_log_scores`            - cumulative log-score differences from
  :func:`MacroPy.recursive_log_scores`.

Every figure uses the MacroPy style (:mod:`MacroPy.style`) and every function
accepts a ``savepath`` to write a tight-bbox vector file for LaTeX.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from ._plotting import _bottom_legend, _date_axis, _save
from .style import PALETTE, color_palette, styled


@styled
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
    ax.fill_between(prob.index, ylo, yhi, where=shaded, color=PALETTE["gray"], alpha=0.22,
                    step="mid", label=regime_label)
    ax.plot(prob.index, series.values, color=PALETTE["ink"], lw=1.2, label=name)
    ax.axhline(zstar, color=PALETTE["accent"], lw=1.6, ls="--",
               label=f"Threshold $Z^*$ = {zstar:.2f}")
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel(name)
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=3)

    if show_prob:
        ax2 = ax.twinx()
        ax2.fill_between(prob.index, 0, prob.values, color=PALETTE["navy"], alpha=0.12)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel(prob_label)
        ax2.grid(False)
    _save(fig, savepath)
    return fig, ax


@styled
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
    vp = model.volatility_path(cred=cred)
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    ax.fill_between(vp.index, vp["lower"], vp["upper"], color=PALETTE["accent"],
                    alpha=0.20, label=f"{int(cred * 100)}% band")
    ax.plot(vp.index, vp["median"], color=PALETTE["accent"], lw=1.7, label="median")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(loc="upper left")
    _save(fig, savepath)
    return fig, ax


def _resolve_shock(model, shock):
    """``None`` means the uncertainty shock when it was computed, otherwise the
    threshold variable's own shock. Returns ``("uncertainty", None)`` or
    ``(name, index)``."""
    names = list(model.names)
    if shock is None:
        ref = model.irf.get("regime1") or model.irf.get("regime2") or {}
        shock = "uncertainty" if "uncertainty" in ref else model.threshold_name
    if isinstance(shock, str) and shock == "uncertainty":
        return "uncertainty", None
    s = names.index(shock) if isinstance(shock, str) else int(shock)
    return names[s], s


def _irf_draws(model, reg: str, shock_idx, v: int):
    """``(draws, horizon)`` responses of variable ``v`` in regime ``reg``, or None."""
    block = model.irf.get(reg)
    if not block:
        return None
    if shock_idx is None:
        if "uncertainty" not in block:
            raise RuntimeError(f"{type(model).__name__} has no uncertainty-shock IRFs; run "
                               "compute_irfs(shock='uncertainty') on a ThresholdVARSV.")
        return block["uncertainty"][:, :, v]
    if "level" not in block:
        raise RuntimeError("Level IRFs not available: run compute_irfs(shock='level').")
    return block["level"][:, :, v, shock_idx]


@styled
def plot_irfs(model, shock: Union[str, int, None] = None,
              response_vars: Optional[List[Union[str, int]]] = None,
              cred: float = 0.68,
              regime_labels: Tuple[str, str] = ("Regime 1", "Regime 2"),
              series_titles: Optional[List[str]] = None, ncol: int = 2,
              title: Optional[str] = None, savepath: Optional[str] = None):
    """Plot generalized impulse responses, regime 1 vs. regime 2 (paper Fig. 3).

    Parameters
    ----------
    shock : str or int, optional
        ``"uncertainty"`` for the volatility shock (``ThresholdVARSV`` only), or
        a variable name / index for a recursive (Cholesky) structural shock to
        that variable, which requires ``compute_irfs(shock="level")``. Default:
        the uncertainty shock when it was computed, otherwise the threshold
        variable's shock.
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
    names = list(model.names)

    # response-variable selection
    if response_vars is None:
        resp_idx = list(range(model.N))
    else:
        resp_idx = [names.index(v) if isinstance(v, str) else int(v)
                    for v in response_vars]
    resp_names = series_titles or [names[i] for i in resp_idx]

    shock_title, s = _resolve_shock(model, shock)

    def fetch(reg, v):
        return _irf_draws(model, reg, s, v)

    hor = next(a.shape[1] for reg in ("regime1", "regime2")
               if (a := fetch(reg, resp_idx[0])) is not None)
    h = np.arange(hor)
    lo = (1 - cred) / 2

    nplot = len(resp_idx)
    nrow = int(np.ceil(nplot / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.0 * nrow),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()

    style = [("regime1", PALETTE["navy"], regime_labels[0]),
             ("regime2", PALETTE["accent"], regime_labels[1])]
    for p, v in enumerate(resp_idx):
        ax = axes[p]
        ax.axhline(0, color=PALETTE["gray"], lw=0.7)
        for reg, color, lab in style:
            a = fetch(reg, v)
            if a is None:
                continue
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
                 color=PALETTE["ink"], weight="bold")
    _save(fig, savepath)
    return fig, axes


def _model_colors(models: dict) -> dict:
    return dict(zip(models, color_palette(len(models))))


@styled
def plot_regime_comparison(models: Dict[str, object], regime: int = 2,
                           var_label: Optional[str] = None,
                           title: str = "Regimes across specifications",
                           savepath: Optional[str] = None):
    """Thresholds and regime probabilities of several threshold models.

    Top panel: the threshold variable of the first model with each model's
    posterior-median threshold ``Z*``. Bottom panel: each model's posterior
    probability of ``regime`` over its estimation sample.

    Parameters
    ----------
    models : dict
        ``{label: model}`` of sampled :class:`ThresholdVAR` / :class:`ThresholdVARSV`.
    regime : {1, 2}, default 2
        Regime whose probability is drawn (2: threshold variable above ``Z*``).
    """
    colors = _model_colors(models)
    first = next(iter(models.values()))
    name = var_label or first.threshold_name
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                                  height_ratios=[1.3, 1], constrained_layout=True)
    series = first.y_df[first.threshold_name]
    series = series[series.index >= min(m.regime_probability().index[0]
                                        for m in models.values())]
    ax.plot(series.index, series.values, color=PALETTE["ink"], lw=1.1, label=name)
    for label, m in models.items():
        m._require_draws()
        zstar = float(np.median(m.draws["tar"]))
        ax.axhline(zstar, color=colors[label], lw=1.5, ls="--",
                   label=f"{label}: $Z^*$ = {zstar:.2f}")
        p2 = m.regime_probability()
        prob = p2 if regime == 2 else 1.0 - p2
        axp.plot(prob.index, prob.values, color=colors[label], lw=1.4, label=label)
    ax.set_ylabel(name)
    ax.legend(loc="upper left", ncol=len(models) + 1)
    axp.set_ylim(-0.02, 1.02)
    axp.set_ylabel(f"P(regime {regime})")
    _date_axis(axp)
    fig.suptitle(title)
    _save(fig, savepath)
    return fig, (ax, axp)


@styled
def plot_irf_comparison(models: Dict[str, object], shock: Union[str, int, None] = None,
                        regimes: Sequence[int] = (1, 2),
                        response_vars: Optional[List[Union[str, int]]] = None,
                        cred: float = 0.68,
                        regime_labels: Optional[Dict[int, str]] = None,
                        series_titles: Optional[List[str]] = None,
                        title: Optional[str] = None, savepath: Optional[str] = None):
    """GIRFs of several threshold models to the same shock, by regime.

    Rows are regimes, columns are responses; each panel overlays the models'
    posterior medians and central ``cred`` bands. For a like-for-like
    comparison use unit shocks (``irf_1std=0`` in every model): one-standard-
    deviation shocks differ across specifications.

    Parameters
    ----------
    models : dict
        ``{label: model}``, each with ``compute_irfs`` already run.
    shock : str or int, optional
        Variable name or index (Cholesky shock), or ``"uncertainty"`` (only for
        ``ThresholdVARSV`` models). Default: the first model's threshold variable.
    regimes : sequence of {1, 2}, default (1, 2)
    regime_labels : dict, optional
        Row labels, e.g. ``{1: "Normal times", 2: "Crisis"}``.
    """
    first = next(iter(models.values()))
    for label, m in models.items():
        if not hasattr(m, "irf"):
            raise RuntimeError(f"Call compute_irfs() on model {label!r} first.")
    shock_title, s = _resolve_shock(first, first.threshold_name if shock is None else shock)
    names = list(first.names)
    resp_idx = (list(range(first.N)) if response_vars is None else
                [names.index(v) if isinstance(v, str) else int(v) for v in response_vars])
    titles = series_titles or [names[i] for i in resp_idx]
    regime_labels = {**{1: "Regime 1", 2: "Regime 2"}, **(regime_labels or {})}
    colors = _model_colors(models)
    lo = 50 * (1 - cred)

    nrow, ncol = len(regimes), len(resp_idx)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), squeeze=False,
                             sharex=True, constrained_layout=True)
    for r, regime in enumerate(regimes):
        for c, v in enumerate(resp_idx):
            ax = axes[r, c]
            ax.axhline(0, color=PALETTE["gray"], lw=0.7)
            for label, m in models.items():
                a = _irf_draws(m, f"regime{regime}", s, v)
                if a is None:
                    continue
                h = np.arange(a.shape[1])
                ax.plot(h, np.median(a, axis=0), color=colors[label], lw=1.7, label=label)
                band = np.percentile(a, [lo, 100 - lo], axis=0)
                ax.fill_between(h, band[0], band[1], color=colors[label], alpha=0.15, lw=0)
            if r == 0:
                ax.set_title(titles[c])
            if c == 0:
                ax.set_ylabel(regime_labels[regime])
            if r == nrow - 1:
                ax.set_xlabel("Horizon")
    _bottom_legend(fig, axes[0, 0], len(models))
    fig.suptitle(title or f"Response to a {shock_title} shock")
    _save(fig, savepath)
    return fig, axes


@styled
def plot_volatility_comparison(models: Dict[str, object],
                               variables: Optional[List[Union[str, int]]] = None,
                               cred: float = 0.68, ncol: int = 2,
                               series_titles: Optional[List[str]] = None,
                               title: str = "Reduced-form shock volatility",
                               savepath: Optional[str] = None):
    """Posterior standard deviation of each reduced-form shock over time.

    Uses each model's ``residual_volatility``: in a :class:`ThresholdVAR` it
    switches between two levels with the regime; in a :class:`ThresholdVARSV`
    it also moves with the common volatility factor.
    """
    first = next(iter(models.values()))
    names = list(first.names)
    idx = (list(range(first.N)) if variables is None else
           [names.index(v) if isinstance(v, str) else int(v) for v in variables])
    titles = series_titles or [names[i] for i in idx]
    frames = {label: m.residual_volatility(cred=cred) for label, m in models.items()}
    colors = _model_colors(models)

    n = len(idx)
    ncol = max(1, min(ncol, n))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 3.0 * nrow), squeeze=False,
                             sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for p, v in enumerate(idx):
        ax = axes[p]
        for label, fr in frames.items():
            f = fr[names[v]]
            ax.fill_between(f.index, f["lower"], f["upper"], color=colors[label], alpha=0.15,
                            lw=0, step="mid")
            ax.plot(f.index, f["median"], color=colors[label], lw=1.3, label=label,
                    drawstyle="steps-mid")
        ax.set_title(titles[p])
        _date_axis(ax)
    for ax in axes[n:]:
        ax.remove()
    _bottom_legend(fig, axes[0], len(models))
    fig.suptitle(title)
    _save(fig, savepath)
    return fig, axes[:n]


@styled
def plot_log_scores(scores: Dict[str, object], benchmark: str, horizons: Sequence[int] = (1,),
                    variables: Sequence[str] = ("joint",),
                    series_titles: Optional[List[str]] = None, title: Optional[str] = None,
                    savepath: Optional[str] = None):
    """Cumulative log-score differences against a benchmark, by forecast origin.

    One panel per (horizon, variable); one line per model. The line rises when
    the model's predictive density beats the benchmark's at the realised value.

    Parameters
    ----------
    scores : dict
        ``{label: frame}`` from :func:`MacroPy.recursive_log_scores`.
    benchmark : str
        Label of the reference model (drawn as the zero line).
    horizons, variables : sequences
        Panels: every horizon for every variable (``"joint"`` or a column).
    """
    base = scores[benchmark]
    others = {k: v for k, v in scores.items() if k != benchmark}
    colors = dict(zip(others, color_palette(len(scores))[1:]))
    panels = [(h, v) for h in horizons for v in variables]
    titles = series_titles or [f"{v}, h = {h}" for h, v in panels]
    ncol = min(len(panels), 3)
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.2 * nrow), squeeze=False,
                             sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for p, (h, var) in enumerate(panels):
        ax = axes[p]
        ax.axhline(0, color=PALETTE["gray"], lw=0.8)
        b = base.xs(h, level="horizon")[var]
        for label, frame in others.items():
            diff = (frame.xs(h, level="horizon")[var] - b).dropna()
            ax.plot(diff.index, diff.cumsum().values, color=colors[label], lw=1.6,
                    label=f"{label} minus {benchmark}")
        ax.set_title(titles[p])
        _date_axis(ax)
    for ax in axes[len(panels):]:
        ax.remove()
    _bottom_legend(fig, axes[0], len(others))
    fig.suptitle(title or "Cumulative log-score difference")
    _save(fig, savepath)
    return fig, axes[:len(panels)]


__all__ = ["plot_regimes", "plot_volatility", "plot_irfs", "plot_regime_comparison",
           "plot_irf_comparison", "plot_volatility_comparison", "plot_log_scores"]
