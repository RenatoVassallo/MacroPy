"""
Figure style for MacroPy: one matplotlib backend and no global side effects.

- ``PALETTE`` is public: edit it in place to restyle every MacroPy figure.
- Plotting functions run inside :func:`macropy_style`, a ``plt.rc_context``,
  so calling them never changes ``plt.rcParams``.
- ``MacroPy.use_own_style = False`` makes the plotting functions keep the
  caller's style untouched (for documents with their own style sheet).
- :func:`set_bse_style` applies the style globally and is never called by the
  plotting functions: it is an explicit opt-in.
"""
from __future__ import annotations

import functools
import sys
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ["PALETTE", "macropy_rcparams", "macropy_style", "set_bse_style", "styled"]

PALETTE = {
    "ink": "#233746",          # titles
    "ink_soft": "#2D4253",
    "accent": "#E78033",
    "navy": "#1F3D5C",
    "gray": "#4A4A4A",
    "light": "#F4F4F4",
    "grid": "#DDDDDD",
    "median": "#1A1A1A",       # medians and observed data
    "band": "#1F3D5C",         # IRF and unconditional-forecast bands
    "conditional": "#B0413E",  # conditional forecasts
    "density": "#1F3D5C",      # posterior densities
    # Okabe-Ito: distinguishable under the common colour-vision deficiencies.
    # Used for stacked areas and bars, where colour is the only cue.
    "stack": ["#0072B2", "#E69F00", "#009E73", "#D55E00",
              "#56B4E9", "#CC79A7", "#F0E442", "#000000"],
    "pastel": ["#a6cee3", "#cab2d6", "#b2df8a", "#fdbf6f", "#fb9a99", "#ccebc5",
               "#fde0a3", "#e0c3a3", "#c0c0e8", "#a8d8e0", "#f0b0a8", "#bcbddc"],
}


def macropy_rcparams() -> dict:
    """MacroPy's rcParams, built from the current ``PALETTE``."""
    p = PALETTE
    return {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": p["ink"],
        "axes.labelcolor": p["gray"],
        "axes.edgecolor": p["gray"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": p["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": p["gray"],
        "ytick.color": p["gray"],
        "legend.frameon": False,
        "legend.fontsize": 10,
    }


def _own_style_enabled() -> bool:
    package = sys.modules.get("MacroPy")
    return bool(getattr(package, "use_own_style", True))


@contextmanager
def macropy_style():
    """
    Apply MacroPy's rcParams inside a ``with`` block and restore them after.

    Only the keys MacroPy sets are restored. ``plt.rc_context`` would restore
    every rcParam, including ``interactive``: if the first figure of a session
    is created inside the block, IPython activates the inline backend there,
    and restoring ``interactive=False`` afterwards stops later figures from
    being shown at the end of their cells.
    """
    if not _own_style_enabled():
        yield
        return
    params = macropy_rcparams()
    saved = {key: mpl.rcParams[key] for key in params}
    mpl.rcParams.update(params)
    try:
        yield
    finally:
        mpl.rcParams.update(saved)


def set_bse_style() -> None:
    """Apply MacroPy's rcParams globally (explicit opt-in; nothing calls it)."""
    plt.rcParams.update(macropy_rcparams())


def styled(func):
    """Decorator: run a plotting function inside :func:`macropy_style`."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with macropy_style():
            return func(*args, **kwargs)
    return wrapper
