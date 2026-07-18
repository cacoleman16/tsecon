"""Results facade for local projections (:func:`tsecon.lp`).

:class:`LPResults` is a ``dict`` subclass. Everything :func:`tsecon.lp` has
always returned — ``horizons``, ``irf``, ``se`` — is present and unchanged;
this object only *adds* a rendered :meth:`summary`, confidence intervals, a
peak-response accessor, and a plot.

The default inference is lag-augmented local projection (Montiel Olea &
Plagborg-Møller, 2021), which is uniformly valid whether the underlying
process is stationary or has a near-unit root; that is the library's
documented default and the summary says so.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

from ._base import Results, fmt_row, rule
from ._plotting import BAND, INK_2, SERIES, apply_style, pyplot

__all__ = ["LPResults"]

_WIDTH = 68


def _z(level: float) -> float:
    """Two-sided normal critical value for a confidence ``level``."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    return NormalDist().inv_cdf(0.5 + level / 2.0)


class LPResults(Results):
    """Local-projection impulse responses with horizon-wise inference.

    A ``dict`` with keys ``horizons``, ``irf`` and ``se`` (all length ``H+1``),
    plus :meth:`summary`, :meth:`conf_int`, :meth:`peak` and :meth:`plot_irf`.
    """

    _kind = "LPResults"

    # Estimation metadata, kept off the dict so the key contract is untouched.
    # Class-level defaults mean the summary still renders on an instance built
    # by hand or restored from a pickle written by an older version.
    _nobs: int | None = None
    _se_kind: str = "lag_augmented"
    _cumulative: bool = False
    _n_lag_controls: int | None = None

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        y: Any,
        shock: Any,
        horizons: int = 12,
        **kw: Any,
    ) -> "LPResults":
        """Estimate local projections and wrap the result.

        Extra keyword arguments are forwarded verbatim to :func:`tsecon.lp`:
        ``n_lag_controls``, ``se`` (``"lag_augmented"`` or ``"hac"``),
        ``maxlags``, ``cumulative``.
        """
        from tsecon._core import lp as _lp  # lazy: avoids an import cycle

        raw = _lp(y, shock, horizons=horizons, **kw)
        out = cls(raw)
        out._nobs = int(np.asarray(y).shape[0])
        out._se_kind = str(kw.get("se", "lag_augmented"))
        out._cumulative = bool(kw.get("cumulative", False))
        ncontrols = kw.get("n_lag_controls")
        out._n_lag_controls = None if ncontrols is None else int(ncontrols)
        return out

    # ------------------------------------------------------------ accessors
    @property
    def irf(self) -> np.ndarray:
        """The impulse response, one entry per horizon."""
        return np.asarray(self["irf"], dtype=float)

    @property
    def se(self) -> np.ndarray:
        """Horizon-wise standard errors."""
        return np.asarray(self["se"], dtype=float)

    @property
    def horizons(self) -> np.ndarray:
        """Horizons ``0 .. H`` as plain ints."""
        return np.asarray(self["horizons"]).astype(int)

    def conf_int(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` pointwise bands, ``irf ± z * se``.

        ``z`` is the two-sided normal critical value for ``level``; these are
        pointwise, not simultaneous, bands.
        """
        z = _z(level)
        irf, se = self.irf, self.se
        return irf - z * se, irf + z * se

    def peak(self) -> tuple[int, float]:
        """``(horizon, value)`` of the largest response in absolute value.

        Ties go to the earliest horizon.
        """
        irf = self.irf
        i = int(np.argmax(np.abs(irf)))
        return int(self.horizons[i]), float(irf[i])

    # -------------------------------------------------------------- summary
    def summary(self, level: float = 0.95) -> str:
        irf, se, hs = self.irf, self.se, self.horizons
        lo, hi = self.conf_int(level)
        peak_h, peak_v = self.peak()

        se_label = {
            "lag_augmented": "lag-augmented, HAC standard errors",
            "hac": "HAC (Newey-West) standard errors",
        }.get(self._se_kind, f"{self._se_kind} standard errors")

        kind = "cumulative " if self._cumulative else ""
        pct = f"{level:.0%}"

        lines = [
            rule(_WIDTH),
            f"Local projection {kind}IRF — {se_label}",
            rule(_WIDTH),
        ]

        stats: list[str] = [f"horizons  0-{int(hs[-1])}"]
        if self._nobs is not None:
            stats.append(f"obs  {self._nobs}")
        if self._n_lag_controls is not None:
            stats.append(f"lag controls  {self._n_lag_controls}")
        stats.append(f"peak  h={peak_h} ({peak_v:+.5f})")
        lines.append("    ".join(stats))
        lines.append(
            "Inference: lag-augmented LP (Montiel Olea & Plagborg-Moller 2021)"
            if self._se_kind == "lag_augmented"
            else f"Inference: {self._se_kind} standard errors"
        )
        lines.append(rule(_WIDTH, "-"))

        widths = [3, 13, 12, 24]
        aligns = ["r", "r", "r", "r"]
        lines.append(
            fmt_row(["h", "IRF", "std err", f"[{pct} conf. int.]"], widths, aligns)
        )
        lines.append(rule(_WIDTH, "-"))
        for i in range(len(irf)):
            ci = f"[{lo[i]:+.5f}, {hi[i]:+.5f}]"
            lines.append(
                fmt_row(
                    [int(hs[i]), f"{irf[i]:+.5f}", f"{se[i]:.5f}", ci],
                    widths,
                    aligns,
                )
            )
        lines.append(rule(_WIDTH))
        return "\n".join(lines)

    # ----------------------------------------------------------------- plot
    def plot_irf(
        self,
        level: float = 0.95,
        ax: Any = None,
        path: str | None = None,
    ):
        """Plot the IRF with a shaded confidence band and a zero line.

        Returns the :class:`matplotlib.figure.Figure`; saves it to ``path``
        first if one is given. Never calls ``plt.show()``.
        """
        plt = pyplot()

        h = self.horizons
        irf = self.irf
        lo, hi = self.conf_int(level)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.2, 3.4))
        else:
            fig = ax.figure

        apply_style(ax, zero_line=True)
        ax.fill_between(h, lo, hi, color=BAND, alpha=0.45, lw=0, zorder=2)
        ax.plot(h, irf, color=SERIES["blue"], lw=1.8, zorder=3)
        ax.plot(h, irf, ".", color=SERIES["blue"], ms=4.5, zorder=4)

        ax.set_xlim(float(h[0]), float(h[-1]))
        ax.set_xlabel("horizon", fontsize=8)
        ax.set_ylabel("response", fontsize=8)
        kind = "Cumulative local projection" if self._cumulative else "Local projection"
        ax.set_title(
            f"{kind} IRF with {level:.0%} band", loc="left", fontsize=9, color=INK_2
        )
        fig.tight_layout()

        if path is not None:
            fig.savefig(path, dpi=200)
        return fig
