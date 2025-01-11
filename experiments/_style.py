"""Shared matplotlib styling for the experiment figures.

One place for the palette and chart chrome so every figure in ``results/``
reads as part of the same system. The categorical palette is CVD-validated
(worst adjacent ΔE 47.2, protan) against the light surface; slots are always
assigned in fixed order, never cycled.
"""

from __future__ import annotations

import matplotlib as mpl

# Categorical slots (fixed order — identity encoding).
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"
RED = "#e34948"

# Sequential blue ramp (ordinal use: no lighter than step 250 on light surface).
BLUE_250 = "#86b6ef"
BLUE_450 = "#2a78d6"
BLUE_650 = "#104281"

# Chrome & ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply() -> None:
    """Install the shared rcParams. Call once, before creating any figure."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10.5,
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlepad": 12,
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
    })


def style_axes(ax, y_only: bool = True) -> None:
    """Recessive grid: horizontal hairlines only (default), baseline kept."""
    if y_only:
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
