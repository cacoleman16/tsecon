"""Results facade for the DSGE-lite (linear rational-expectations) solver.

:func:`tsecon.dsge_solve` solves ``A E_t[y_{t+1}] = B y_t + C z_{t+1}`` by
Blanchard-Kahn (1980) and returns the decision rule ``g``, the law of motion
``p``/``q``, the ``eigenvalue_moduli`` and a ``verdict`` string.

:class:`DSGEResults` keeps every one of those keys — it *is* the dict the
compiled function returned — and adds the reading that the raw numbers do not
give you: a rendered Blanchard-Kahn verdict with the eigenvalues split at the
unit circle, a determinacy predicate, and the saddle-path impulse response.
That last one is genuinely new: the compiled binding exposes the matrices but
never iterates them, so tracing a shock through the model was left to the
caller until now.
"""

from __future__ import annotations

import textwrap
from typing import Sequence

import numpy as np

from ._base import Results, fmt_row, kv_line, rule
from ._plotting import SERIES, apply_style, pyplot

__all__ = ["DSGEResults"]

_WIDTH = 68
_MAX_COLS = 6  # columns of a matrix printed before we elide the rest
_CYCLE = [SERIES["blue"], SERIES["red"], SERIES["aqua"], SERIES["violet"], SERIES["yellow"], SERIES["green"]]


class DSGEResults(Results):
    """Blanchard-Kahn solution of a linear rational-expectations model.

    A :class:`dict` with the keys :func:`tsecon.dsge_solve` has always
    returned — ``g``, ``p``, ``q``, ``eigenvalue_moduli``, ``verdict`` — plus
    :meth:`summary`, :meth:`is_determinate`, :meth:`impulse_response` and
    :meth:`plot_impulse_response`.
    """

    _kind = "DSGEResults"

    # ---------------------------------------------------------------- build
    @classmethod
    def solve(cls, a, b, c, n_predetermined: int) -> "DSGEResults":
        """Solve the model and wrap the result.

        Same arguments and same returned keys as :func:`tsecon.dsge_solve`.
        """
        from .. import dsge_solve  # local: tsecon imports this subpackage

        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        return cls(dsge_solve(a, b, c, int(n_predetermined)))

    # ----------------------------------------------------------- accessors
    def policy(self) -> np.ndarray:
        """``G`` — the decision rule, ``jump_t = G @ predetermined_t``."""
        return np.atleast_2d(np.asarray(self["g"], dtype=np.float64))

    def transition(self) -> np.ndarray:
        """``P`` — the state transition, ``k_{t+1} = P @ k_t + Q @ z_{t+1}``."""
        return np.atleast_2d(np.asarray(self["p"], dtype=np.float64))

    def impact(self) -> np.ndarray:
        """``Q`` — the shock loading in the law of motion."""
        return np.atleast_2d(np.asarray(self["q"], dtype=np.float64))

    @property
    def n_predetermined(self) -> int:
        return int(self.transition().shape[0])

    @property
    def n_jump(self) -> int:
        return int(self.policy().shape[0])

    @property
    def n_shocks(self) -> int:
        return int(self.impact().shape[1])

    def moduli(self) -> np.ndarray:
        """Eigenvalue moduli in solver order, as a float array."""
        return np.asarray(self["eigenvalue_moduli"], dtype=np.float64).ravel()

    def stable_moduli(self, tol: float = 1.0) -> np.ndarray:
        """The moduli inside the unit circle (``<= tol``), ascending."""
        m = self.moduli()
        return np.sort(m[m <= tol])

    def unstable_moduli(self, tol: float = 1.0) -> np.ndarray:
        """The moduli outside the unit circle (``> tol``), ascending."""
        m = self.moduli()
        return np.sort(m[m > tol])

    def is_determinate(self) -> bool:
        """``True`` iff Blanchard-Kahn returned the unique-stable-solution case.

        The count of unstable eigenvalues then equals the count of jump
        variables; fewer means indeterminacy, more means no stable solution.
        """
        return str(self["verdict"]).startswith("unique stable solution")

    # --------------------------------------------------------------- IRF
    def impulse_response(self, horizon: int = 24, shock=None) -> dict:
        """Trace the saddle path of a one-off innovation.

        The state is put on impact at ``k_0 = Q @ shock``, then iterated with
        ``k_{t+1} = P @ k_t``; the jumps follow the decision rule at every
        date, ``x_t = G @ k_t``. ``shock`` defaults to a unit impulse to the
        first innovation.

        Returns a plain dict with ``horizon`` (int), ``shock`` (n_shocks,),
        ``predetermined`` (horizon x n_predetermined) and ``jump``
        (horizon x n_jump). Period 0 is the impact period.
        """
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")

        p, q, g = self.transition(), self.impact(), self.policy()
        n_shocks = q.shape[1]
        if n_shocks == 0:
            raise ValueError(
                "this solution has no innovations (Q has zero columns), so "
                "there is no impulse response to trace"
            )

        if shock is None:
            shock = np.zeros(n_shocks, dtype=np.float64)
            shock[0] = 1.0
        else:
            shock = np.asarray(shock, dtype=np.float64).ravel()
            if shock.shape[0] != n_shocks:
                raise ValueError(
                    f"shock has {shock.shape[0]} element(s) but the model has "
                    f"{n_shocks} innovation(s)"
                )

        n_pre = p.shape[0]
        pre = np.zeros((horizon, n_pre), dtype=np.float64)
        k = q @ shock
        for t in range(horizon):
            pre[t] = k
            k = p @ k
        jump = pre @ g.T

        return {
            "horizon": horizon,
            "shock": shock,
            "predetermined": pre,
            "jump": jump,
        }

    # ----------------------------------------------------------- rendering
    def summary(self) -> str:
        verdict = str(self["verdict"])
        head = verdict.split(" (")[0]
        lines = [
            rule(_WIDTH),
            f"Linear RE model (Blanchard-Kahn): {head}",
            rule(_WIDTH),
            kv_line(
                [
                    ("predetermined", self.n_predetermined),
                    ("jump", self.n_jump),
                    ("shocks", self.n_shocks),
                    ("determinate", "yes" if self.is_determinate() else "no"),
                ]
            ),
        ]
        lines += textwrap.wrap(
            f"verdict: {verdict}", width=_WIDTH, subsequent_indent="         "
        )
        lines.append(rule(_WIDTH, "-"))

        stable, unstable = self.stable_moduli(), self.unstable_moduli()
        lines.append(
            kv_line(
                [
                    ("eigenvalue moduli:  stable (<1)", stable.size),
                    ("unstable (>1)", unstable.size),
                ]
            )
        )
        lines += _value_lines("stable", stable)
        lines += _value_lines("unstable", unstable)
        lines.append(rule(_WIDTH, "-"))

        lines += _matrix_lines("G", "policy: jump = G . predetermined", self.policy())
        lines += _matrix_lines("P", "transition: k(t+1) = P . k(t) + Q . z", self.transition())
        lines += _matrix_lines("Q", "impact: shock loading on the state", self.impact())
        lines.append(rule(_WIDTH))
        return "\n".join(lines)

    # -------------------------------------------------------------- plots
    def plot_impulse_response(
        self,
        horizon: int = 24,
        shock=None,
        *,
        names: Sequence[str] | None = None,
        ax=None,
        path: str | None = None,
        title: str | None = None,
    ):
        """Plot the saddle path from :meth:`impulse_response`.

        Predetermined states are drawn solid, jumps dashed, each labelled at
        the end of its own line so nothing sits on top of the data. Returns
        the :class:`matplotlib.figure.Figure`; saves it if ``path`` is given.
        """
        plt = pyplot()
        irf = self.impulse_response(horizon=horizon, shock=shock)
        pre, jump = irf["predetermined"], irf["jump"]
        h = irf["horizon"]

        labels = list(names) if names is not None else (
            [f"k{i + 1}" for i in range(pre.shape[1])]
            + [f"x{i + 1}" for i in range(jump.shape[1])]
        )
        n_series = pre.shape[1] + jump.shape[1]
        if len(labels) != n_series:
            raise ValueError(
                f"names has {len(labels)} entries but there are {n_series} series"
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
        else:
            fig = ax.figure

        apply_style(ax, zero_line=True)
        periods = np.arange(h)
        series = [(pre[:, i], "-") for i in range(pre.shape[1])]
        series += [(jump[:, i], "--") for i in range(jump.shape[1])]

        for i, (values, style) in enumerate(series):
            colour = _CYCLE[i % len(_CYCLE)]
            ax.plot(periods, values, style, color=colour, lw=1.6, zorder=3)

        # Label each line at its own end rather than in a legend, so no text
        # ever lands on the data; nudge apart any labels that would collide.
        stacked = np.column_stack([values for values, _ in series])
        label_y = _stagger([values[-1] for values, _ in series], stacked)
        for i, y in enumerate(label_y):
            ax.annotate(
                labels[i],
                xy=(periods[-1], y),
                xytext=(5, 0),
                textcoords="offset points",
                color=_CYCLE[i % len(_CYCLE)],
                fontsize=8,
                va="center",
                ha="left",
                annotation_clip=False,
            )

        lo = float(min(stacked.min(), min(label_y)))
        hi = float(max(stacked.max(), max(label_y)))
        margin = 0.08 * max(hi - lo, 1e-12)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlim(0, h - 1 + max(0.6, 0.03 * h))
        ax.set_xlabel("periods after the shock", fontsize=8)
        ax.set_ylabel("deviation from steady state", fontsize=8)
        ax.set_title(
            title or "DSGE impulse response (Blanchard-Kahn saddle path)",
            fontsize=9.5,
            loc="left",
        )
        if path is not None:
            fig.savefig(path, dpi=150)
        return fig


# --------------------------------------------------------------------------- #
# private formatting helpers
# --------------------------------------------------------------------------- #
def _value_lines(label: str, values: np.ndarray, per_line: int = 5) -> list[str]:
    """``  label   v v v`` — wrapped so long spectra stay inside the rule."""
    if values.size == 0:
        return [f"  {label:<10}(none)"]
    out = []
    for start in range(0, values.size, per_line):
        chunk = values[start : start + per_line]
        cells = [f"{v:.5f}" for v in chunk]
        head = label if start == 0 else ""
        out.append("  " + fmt_row([head] + cells, [10] + [10] * len(cells), ["l"] + ["r"] * len(cells)))
    return out


def _stagger(ends: Sequence[float], data: np.ndarray, min_gap: float = 0.055) -> list[float]:
    """Push end-of-line label positions apart so they never overlap."""
    ends = [float(v) for v in ends]
    span = float(data.max() - data.min())
    gap = min_gap * (span if span > 0 else max(abs(max(ends, key=abs)), 1.0))
    placed = list(ends)
    prev = -np.inf
    for i in sorted(range(len(ends)), key=lambda j: ends[j]):
        placed[i] = max(placed[i], prev + gap)
        prev = placed[i]
    return placed


def _matrix_lines(name: str, caption: str, mat: np.ndarray) -> list[str]:
    """A small labelled matrix: a caption with its shape, then the rows."""
    n_rows, n_cols = mat.shape
    shown = min(n_cols, _MAX_COLS)
    header = f"{name}  {caption}"
    shape = f"[{n_rows}x{n_cols}]"
    pad = max(1, _WIDTH - len(header) - len(shape))
    lines = [header + " " * pad + shape]
    for r in range(n_rows):
        cells = [f"{mat[r, c]:+.5f}" for c in range(shown)]
        if n_cols > shown:
            cells.append("...")
        lines.append("  " + fmt_row(cells, [11] * len(cells), ["r"] * len(cells)))
    return lines
