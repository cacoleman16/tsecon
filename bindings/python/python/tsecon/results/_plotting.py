"""Lazy matplotlib access and the shipped house style.

matplotlib is an **optional** dependency (the ``plots`` extra). Nothing here
imports it at module load; every plotting method calls :func:`pyplot` first, so
installing tsecon without matplotlib costs nothing and the failure — if you do
call a plot method — is a message that tells you exactly what to install.

The palette mirrors the documentation figures' house style. That style lives in
``prototypes/viz/tsecon_style.py``, which is a repo-only development tool and is
not shipped in the wheel, so the handful of constants a shipped plot needs are
restated here rather than imported.
"""

from __future__ import annotations

__all__ = ["pyplot", "SERIES", "INK", "INK_2", "MUTED", "GRID", "REF", "BAND", "apply_style"]

# House palette (kept in sync with prototypes/viz/tsecon_style.py).
SERIES = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
REF = "#a5a39c"
BAND = "#9ec5f4"


def pyplot():
    """Return ``matplotlib.pyplot``, or raise a message that says what to do."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "plotting requires matplotlib, which is an optional dependency of "
            "tsecon.\n"
            "    pip install 'tsecon[plots]'      # or: pip install matplotlib\n"
            "Every plot method has a data-returning twin, so you can also render "
            "the numbers yourself with any library — e.g. use the arrays in this "
            "results object directly."
        ) from exc
    return plt


def apply_style(ax, *, zero_line: bool = False) -> None:
    """Nudge an Axes toward the house look: quiet spines, hairline grid."""
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REF)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=7.5, colors=INK_2)
    if zero_line:
        ax.axhline(0.0, color=REF, lw=0.9, zorder=1.5)
