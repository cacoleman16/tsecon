"""Prototype of the tsecon visual identity (Module 13).

Design tokens + a matplotlib theme implementing the style contract in
docs/roadmap/13-visualization.md: data-ink first, colorblind-safe validated
palette, banded uncertainty, recessive chrome, publication sizing.

This is a prototype: the production version lives in the future tsecon.plots
package with themes (paper/presentation/dark/draft) and export presets.
"""
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- tokens
# Categorical palette — validated (CVD worst adjacent dE 24.2, >= 12 target).
SERIES = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
CATEGORICAL = list(SERIES.values())

# Sequential blue ramp (light -> dark), for nested coverage bands.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text
MUTED = "#898781"        # axis labels, captions
GRID = "#e1e0d9"         # hairline gridlines
BASELINE = "#c3c2b7"     # axis spines
SURFACE = "#fcfcfb"      # chart surface
SHADE = "#eceae4"        # recession/regime shading
REF = "#a5a39c"          # semantic reference lines (zero lines, critical values)

# Physical figure widths (inches) — export presets.
WIDTH_SINGLE = 3.25      # journal single column
WIDTH_DOUBLE = 6.75      # journal double column
WIDTH_SLIDE = 10.0

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.0,
    "axes.titlesize": 11.0,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.titlepad": 10.0,
    "axes.labelsize": 9.0,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": (0, (1, 3)),
    "axes.axisbelow": True,
    "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    "xtick.color": BASELINE,
    "ytick.color": BASELINE,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.major.size": 3.0,
    "ytick.major.size": 0.0,
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "legend.labelcolor": INK_2,
}


@contextmanager
def theme():
    """Scoped 'paper' theme — never mutates the user's global rcParams."""
    with mpl.rc_context(RC):
        yield


# ---------------------------------------------------------------- helpers
def new_fig(ncols=1, nrows=1, width=WIDTH_DOUBLE, aspect=0.42, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, width * aspect), **kw)
    return fig, axes


def zero_line(ax):
    """Semantic zero reference — muted, under the data, never data-weight."""
    ax.axhline(0.0, color=REF, lw=0.9, zorder=1.5)


def shade_period(ax, start, end, label=None):
    """Recession/regime shading with an optional top label."""
    ax.axvspan(start, end, color=SHADE, zorder=0, lw=0)
    if label:
        ax.annotate(
            label, xy=((start + end) / 2, 1.0), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=7.5, color=MUTED,
        )


def nested_bands(ax, x, center, half_widths, coverages, color_steps=None, alpha=1.0):
    """Nested coverage bands: widest coverage drawn first in the lightest step."""
    order = np.argsort(coverages)[::-1]
    steps = color_steps or SEQ_BLUE
    for rank, i in enumerate(order):
        ax.fill_between(
            x, center - half_widths[i], center + half_widths[i],
            color=steps[min(rank, len(steps) - 1)], lw=0, alpha=alpha, zorder=2,
        )


def band_label(ax, x, y, text):
    ax.annotate(text, xy=(x, y), fontsize=7.5, color=INK_2, ha="left", va="center")


def stamp(fig, text):
    """Metadata stamp (identification method, band type) — self-documenting figures."""
    fig.text(0.005, -0.02, text, fontsize=7.0, color=MUTED, ha="left", va="top")


def despine_x_only(ax):
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
