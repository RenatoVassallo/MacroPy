"""
Layout helpers shared by every ``plots*`` module (private).

Appearance (colours, type, rcParams) lives in :mod:`MacroPy.style`; this
module only arranges and exports figures.
"""
from __future__ import annotations

import math
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from .style import PALETTE

__all__: list = []          # private helpers only


def _facet_grid_shape(n: int, ncol: int = None) -> tuple:
    """Balanced ``(nrow, ncol)`` for ``n`` panels (1x1, 1x2, 1x3, 2x2, 2x3, 3x3...)."""
    if ncol is not None:
        ncol = max(1, int(ncol))
        return (int(math.ceil(n / ncol)), ncol)
    if n <= 3:
        return (1, max(n, 1))
    if n == 4:
        return (2, 2)
    if n <= 6:
        return (2, 3)
    if n <= 9:
        return (3, 3)
    ncol = int(math.ceil(math.sqrt(n)))
    return (int(math.ceil(n / ncol)), ncol)


def _panels(n: int, ncol: int, panel=(5.0, 3.0)):
    """Figure with ``n`` panels on a grid of ``ncol`` columns; unused axes are removed."""
    ncol = max(1, min(ncol, n))
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel[0] * ncol, panel[1] * nrow),
                             squeeze=False, constrained_layout=True)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.remove()
    return fig, list(axes[:n])


def _date_axis(ax, n_breaks: int = 10) -> None:
    """Concise date ticks (explicit, so they survive ``use_own_style = False``)."""
    locator = mdates.AutoDateLocator(minticks=3, maxticks=max(3, n_breaks))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _bottom_legend(fig, ax, ncol: int) -> None:
    """One legend for the whole figure, below the panels."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=max(1, min(ncol, 6)))


def _stack_colors(color_palette, n: int) -> list:
    """``n`` colours for stacked areas and bars (``PALETTE["stack"]`` by default)."""
    colors = list(color_palette) if color_palette else list(PALETTE["stack"])
    return [colors[i % len(colors)] for i in range(n)]


def _save(fig, savepath: Optional[str]) -> None:
    """Write ``fig`` to ``savepath`` (tight bounding box; PDF or SVG for LaTeX)."""
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
