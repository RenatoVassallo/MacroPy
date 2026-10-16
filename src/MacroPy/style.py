"""
The MacroPy look: one style for every figure, an explicit opt-in for yours.

MacroPy's plotting functions always draw in the MacroPy style, inside a
temporary context, so calling them never changes your matplotlib settings. To
give your own figures the same look, opt in once::

    import MacroPy as mp

    mp.set_style()                        # notebook defaults
    mp.set_style("slides", font="sans")   # larger type for beamer slides
    mp.set_style("paper")                 # smaller type, 300 dpi, for LaTeX
    mp.reset_style()                      # back to your previous settings

    with mp.style_context("paper"):       # or only for a block
        fig, ax = plt.subplots()

or, without importing anything from MacroPy, with the bundled style sheet
``plt.style.use("MacroPy.macropy")`` (the notebook context, serif type).

``set_style`` also sets the context and font of MacroPy's own figures, so one
call restyles everything. ``MacroPy.use_own_style = False`` makes MacroPy's
figures keep the caller's style instead (documents with their own sheet).

The look is defined in one place:

* :data:`PALETTE` holds the colours by role (``navy`` and ``accent`` are the
  identity pair; ``median``, ``band``, ``conditional``...) and the categorical
  ``cycle``, the default line colours under the style. Edit it in place to
  restyle every MacroPy figure.
* :func:`macropy_rcparams` turns it into matplotlib settings, scaled by
  context (:data:`CONTEXTS`).
"""
from __future__ import annotations

import functools
import sys
import warnings
from contextlib import contextmanager
from typing import List, Optional

import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "PALETTE", "CONTEXTS", "set_style", "reset_style", "style_context", "styled",
    "macropy_rcparams", "macropy_style", "color_palette", "colormap", "set_bse_style",
]

# Categorical colours: the identity pair first, then Okabe-Ito hues chosen to
# stay distinguishable under the common colour-vision deficiencies.
_CYCLE = ["#1F3D5C", "#E78033", "#009E73", "#B0413E",
          "#56B4E9", "#CC79A7", "#F0E442", "#7F7F7F"]

PALETTE = {
    # neutrals
    "ink": "#233746",          # titles
    "ink_soft": "#2D4253",
    "gray": "#4A4A4A",         # axes, labels, reference lines
    "light": "#F4F4F4",
    "grid": "#DDDDDD",
    # identity pair
    "navy": "#1F3D5C",
    "accent": "#E78033",
    # roles
    "median": "#1A1A1A",       # medians and observed data
    "band": "#1F3D5C",         # IRF and unconditional-forecast bands
    "conditional": "#B0413E",  # conditional forecasts and scenarios
    "density": "#1F3D5C",      # posterior densities
    # categorical
    "cycle": list(_CYCLE),     # line colours under the style
    "stack": list(_CYCLE),     # stacked areas and bars (FEVD, HD)
    "pastel": ["#a6cee3", "#cab2d6", "#b2df8a", "#fdbf6f", "#fb9a99", "#ccebc5",
               "#fde0a3", "#e0c3a3", "#c0c0e8", "#a8d8e0", "#f0b0a8", "#bcbddc"],
}

#: Type scale, default figure size and export resolution per context.
CONTEXTS = {
    "paper": {"scale": 0.85, "figsize": (6.4, 3.6), "savefig_dpi": 300},
    "notebook": {"scale": 1.0, "figsize": (8.0, 4.5), "savefig_dpi": 200},
    "slides": {"scale": 1.35, "figsize": (9.0, 5.0), "savefig_dpi": 200},
}

_DEFAULTS = {"context": "notebook", "font": "serif"}
_active = dict(_DEFAULTS)          # what MacroPy's own figures use
_saved: Optional[dict] = None      # the caller's settings before set_style()


def _font(font: str):
    """``(font.family, mathtext.fontset)``: math type matches the text type."""
    if font == "serif":
        return ["serif"], "dejavuserif"
    if font == "sans":
        return ["sans-serif"], "dejavusans"
    return [font, "serif"], "dejavuserif"          # a named family, e.g. "Fira Sans"


def macropy_rcparams(context: Optional[str] = None, font: Optional[str] = None) -> dict:
    """
    MacroPy's matplotlib settings, built from the current :data:`PALETTE`.

    Parameters
    ----------
    context : {"paper", "notebook", "slides"}, optional
        Type scale, default figure size and export dpi (default: the active
        context, ``"notebook"`` unless :func:`set_style` changed it).
    font : {"serif", "sans"} or a font family name, optional
        Text type; formulas use the matching DejaVu math set. Default: the
        active font, ``"serif"``.
    """
    context = _active["context"] if context is None else context
    font = _active["font"] if font is None else font
    if context not in CONTEXTS:
        raise ValueError(f"context must be one of {list(CONTEXTS)}; got {context!r}.")
    p, c = PALETTE, CONTEXTS[context]
    s = c["scale"]
    family, math = _font(font)
    return {
        "figure.dpi": 110,
        "figure.figsize": c["figsize"],
        "figure.titlesize": round(13 * s, 1),
        "figure.titleweight": "bold",
        "savefig.dpi": c["savefig_dpi"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": family,
        "mathtext.fontset": math,
        "font.size": round(11 * s, 1),
        "axes.titlesize": round(12 * s, 1),
        "axes.labelsize": round(11 * s, 1),
        "xtick.labelsize": round(10 * s, 1),
        "ytick.labelsize": round(10 * s, 1),
        "legend.fontsize": round(10 * s, 1),
        "axes.titleweight": "bold",
        "axes.titlecolor": p["ink"],
        "axes.labelcolor": p["gray"],
        "axes.edgecolor": p["gray"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=list(p["cycle"])),
        "grid.color": p["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": p["gray"],
        "ytick.color": p["gray"],
        "lines.linewidth": round(1.5 * s, 2),
        "legend.frameon": False,
        "date.converter": "concise",
    }


def set_style(context: str = "notebook", font: str = "serif") -> None:
    """
    Apply the MacroPy style globally: to your own figures and to MacroPy's.

    The first call remembers your previous settings, which
    :func:`reset_style` restores; later calls only change the context or the
    font.

    Parameters
    ----------
    context : {"paper", "notebook", "slides"}, default "notebook"
        ``"paper"``: smaller type and 300 dpi exports for LaTeX documents;
        ``"slides"``: larger type for presentations.
    font : {"serif", "sans"} or a font family name, default "serif"
    """
    global _saved
    params = macropy_rcparams(context, font)
    if _saved is None:
        _saved = {key: mpl.rcParams[key] for key in params}
    mpl.rcParams.update(params)
    _active.update(context=context, font=font)


def reset_style() -> None:
    """Undo :func:`set_style`: restore your settings and MacroPy's defaults."""
    global _saved
    if _saved is not None:
        mpl.rcParams.update(_saved)
        _saved = None
    _active.update(_DEFAULTS)


@contextmanager
def style_context(context: Optional[str] = None, font: Optional[str] = None):
    """
    The MacroPy style inside a ``with`` block, restored on exit.

    Only the keys MacroPy sets are restored. ``plt.rc_context`` would restore
    every rcParam, including ``interactive``: if the first figure of a session
    is created inside the block, IPython activates the inline backend there,
    and restoring ``interactive=False`` afterwards stops later figures from
    being shown at the end of their cells.
    """
    params = macropy_rcparams(context, font)
    saved = {key: mpl.rcParams[key] for key in params}
    mpl.rcParams.update(params)
    try:
        yield
    finally:
        mpl.rcParams.update(saved)


def _own_style_enabled() -> bool:
    package = sys.modules.get("MacroPy")
    return bool(getattr(package, "use_own_style", True))


@contextmanager
def macropy_style():
    """The style of MacroPy's own figures: :func:`style_context` with the
    active context and font, or nothing when ``MacroPy.use_own_style`` is
    False."""
    if not _own_style_enabled():
        yield
        return
    with style_context():
        yield


class _FigureTuple(tuple):
    """What plotting functions return: an ordinary tuple (``fig, axes = ...``
    unpacks as usual) that a notebook does not echo when it ends a cell, since
    the figure itself is already shown."""
    __slots__ = ()

    def _ipython_display_(self):
        pass


class _FigureList(list):
    """List counterpart of :class:`_FigureTuple` (one ``(fig, axes)`` per item)."""

    def _ipython_display_(self):
        pass


def _quiet(result):
    if type(result) is tuple:
        return _FigureTuple(result)
    if type(result) is list:
        return _FigureList(result)
    return result


def styled(func):
    """Decorator for plotting functions: run them inside :func:`macropy_style`
    and return their figures without a notebook echo of ``(fig, axes)``."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with macropy_style():
            return _quiet(func(*args, **kwargs))
    return wrapper


def color_palette(n: Optional[int] = None) -> List[str]:
    """The categorical colours, repeated to length ``n`` (all of them by default)."""
    colors = list(PALETTE["cycle"])
    if n is None:
        return colors
    return [colors[i % len(colors)] for i in range(int(n))]


def colormap(kind: str = "diverging") -> LinearSegmentedColormap:
    """
    MacroPy colour maps (not registered globally).

    ``"diverging"``: navy, light, accent, for signed quantities centred at
    zero (impulse-response surfaces, changes); ``"sequential"``: light to
    navy, for magnitudes.
    """
    p = PALETTE
    if kind == "diverging":
        return LinearSegmentedColormap.from_list("macropy_diverging",
                                                 [p["navy"], p["light"], p["accent"]])
    if kind == "sequential":
        return LinearSegmentedColormap.from_list("macropy_sequential",
                                                 [p["light"], p["navy"]])
    raise ValueError("kind must be 'diverging' or 'sequential'.")


def _mplstyle_text() -> str:
    """Contents of the bundled ``macropy.mplstyle`` (notebook context, serif)."""
    def fmt(value):
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, tuple):
            return ", ".join(str(v) for v in value)
        if isinstance(value, list):
            return ", ".join(value)
        if isinstance(value, str) and value.startswith("#"):
            return value[1:]
        return str(value)

    lines = ["# MacroPy figure style: notebook context, serif type.",
             "# Generated by MacroPy.style._mplstyle_text(); do not edit by hand.",
             '# Use with plt.style.use("MacroPy.macropy").', ""]
    for key, value in macropy_rcparams("notebook", "serif").items():
        if key == "axes.prop_cycle":
            colors = ", ".join(f"'{c[1:]}'" for c in PALETTE["cycle"])
            lines.append(f"{key}: cycler('color', [{colors}])")
        else:
            lines.append(f"{key}: {fmt(value)}")
    return "\n".join(lines) + "\n"


def set_bse_style() -> None:
    """Deprecated alias of :func:`set_style`."""
    warnings.warn("set_bse_style() is deprecated; use MacroPy.set_style().",
                  DeprecationWarning, stacklevel=2)
    set_style()
