"""Results objects for the VAR family: :class:`VARResults` and :class:`IRFArray`.

Both wrappers are *additive*. ``VARResults`` is a ``dict`` — every key
``tsecon.var_fit`` has ever returned (``params``, ``sigma_u``, ``llf``, ``aic``,
``bic``, ``hqic``, ``max_root``, ``min_root``, ``is_stable``) is still there,
still under the same name, still holding the same value. ``IRFArray`` is a
``list`` for the same reason: ``irf[h][i][j]``, ``len(irf)`` and
``np.array(irf)`` must keep behaving exactly as they did when ``var_irf``
returned a bare nested list.

A note on stability, because it is easy to get backwards: ``max_root`` is the
modulus of the reciprocal characteristic root *farthest* from the unit circle
and is not a verdict about anything. The system is stable iff **every**
reciprocal root lies outside the unit circle, i.e. iff ``min_root > 1`` — which
is precisely what the compiled ``is_stable`` flag reports. The summary quotes
``is_stable``.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import numpy as np

from ._base import Results, fmt_row, kv_line, rule
from ._plotting import INK, INK_2, SERIES, apply_style, pyplot

__all__ = ["VARResults", "IRFArray", "CoefficientFrame", "var_fit", "var_irf"]

# Equation columns per printed block: 12 + 4 * (12 + 2) = 68, the house width.
_EQ_PER_BLOCK = 4
_WIDTH = 68


def _compiled():
    """The compiled estimator module, imported lazily to dodge import cycles."""
    from .. import _core

    return _core


def _default_names(data: Any, k: int, names: Sequence[str] | None) -> list[str]:
    """Column names: explicit, else a DataFrame's columns, else ``y1..yk``."""
    if names is not None:
        names = [str(n) for n in names]
        if len(names) != k:
            raise ValueError(f"names has {len(names)} entries but the system has {k} variables")
        return names
    columns = getattr(data, "columns", None)
    if columns is not None:
        try:
            candidate = [str(c) for c in columns]
        except TypeError:  # pragma: no cover - exotic column containers
            candidate = []
        if len(candidate) == k:
            return candidate
    return [f"y{i + 1}" for i in range(k)]


def _det_labels(n_det: int, trend: str) -> list[str]:
    """Row labels for the deterministic block, ordered as the design matrix."""
    canonical = {"n": [], "c": ["const"], "ct": ["const", "trend"],
                 "ctt": ["const", "trend", "trend^2"]}.get(trend.lower())
    if canonical is not None and len(canonical) == n_det:
        return list(canonical)
    # Unknown/extended trend strings: stay generic rather than mislabel.
    fallback = ["const", "trend", "trend^2"]
    return [fallback[i] if i < len(fallback) else f"det{i + 1}" for i in range(n_det)]


class CoefficientFrame(NamedTuple):
    """The coefficient matrix with its labels, ready for any table library.

    ``values`` is ``(len(rows), len(columns))`` — regressors down, equations
    across — matching the layout of the raw ``params`` key exactly.
    """

    rows: list[str]
    columns: list[str]
    values: np.ndarray

    def to_pandas(self):
        """A ``pandas.DataFrame`` view. pandas is not a tsecon dependency."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "coefficient_frame().to_pandas() requires pandas, which tsecon "
                "does not depend on.\n"
                "    pip install pandas\n"
                "The .rows / .columns / .values fields carry the same numbers if "
                "you would rather not install it."
            ) from exc
        return pd.DataFrame(self.values, index=self.rows, columns=self.columns)


class VARResults(Results):
    """A fitted VAR — the ``var_fit`` dict, plus a summary and IRFs."""

    _kind = "VARResults"

    @classmethod
    def fit(
        cls,
        data,
        lags: int = 2,
        trend: str = "c",
        names: Sequence[str] | None = None,
    ) -> "VARResults":
        """Fit a VAR(p) by OLS and wrap the result.

        Parameters mirror :func:`tsecon.var_fit`; ``names`` labels the equations
        (default ``y1..yk``, or a DataFrame's columns when one is passed).
        """
        array = np.asarray(data, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"data must be 2-D (obs x variables), got shape {array.shape}")
        raw = _compiled().var_fit(array, lags, trend)

        obj = cls(raw)
        obj.names = _default_names(data, array.shape[1], names)
        obj.lags = int(lags)
        obj.trend = str(trend)
        obj.neqs = int(array.shape[1])
        obj.nobs = int(array.shape[0] - lags)
        # Deliberately an attribute, not a key: to_dict()/json.dumps stay clean.
        obj._data = array
        return obj

    # ----------------------------------------------------------------- views
    @property
    def stable(self) -> bool:
        """The stability verdict, straight from ``is_stable``."""
        return bool(self["is_stable"])

    def regressor_labels(self) -> list[str]:
        """Row labels of ``params``: deterministic terms, then ``L<p>.<name>``."""
        params = np.asarray(self["params"], dtype=float)
        n_det = params.shape[0] - self.lags * self.neqs
        labels = _det_labels(n_det, self.trend)
        for lag in range(1, self.lags + 1):
            labels.extend(f"L{lag}.{name}" for name in self.names)
        return labels

    def coefficient_frame(self) -> CoefficientFrame:
        """Row labels, equation names and the coefficient array, unmodified."""
        return CoefficientFrame(
            rows=self.regressor_labels(),
            columns=list(self.names),
            values=np.asarray(self["params"], dtype=float),
        )

    def irf(self, horizon: int = 12, orth: bool = True, cumulative: bool = False) -> "IRFArray":
        """Impulse responses from the same data this VAR was fitted on."""
        if getattr(self, "_data", None) is None:
            raise ValueError(
                "this VARResults was not built by VARResults.fit(), so the original "
                "data is unavailable; call tsecon.results.var_irf(data, ...) instead"
            )
        return var_irf(
            self._data,
            lags=self.lags,
            horizon=horizon,
            orth=orth,
            trend=self.trend,
            cumulative=cumulative,
            names=self.names,
        )

    # --------------------------------------------------------------- summary
    def summary(self) -> str:
        verdict = "stable" if self.stable else "UNSTABLE"
        lines = [
            rule(_WIDTH),
            f"VAR({self.lags}) — {self.neqs} equations, trend={self.trend!r} — {verdict}",
            rule(_WIDTH),
            kv_line(
                [
                    ("llf", f"{self['llf']:.3f}"),
                    ("aic", f"{self['aic']:.4f}"),
                    ("bic", f"{self['bic']:.4f}"),
                    ("hqic", f"{self['hqic']:.4f}"),
                ]
            ),
            kv_line(
                [
                    ("reciprocal roots — min", f"{self['min_root']:.4f}"),
                    ("max", f"{self['max_root']:.4f}"),
                    ("", "(stable iff min > 1)"),
                ]
            ),
            rule(_WIDTH, "-"),
            "coefficients — rows = regressors, cols = equations",
        ]

        frame = self.coefficient_frame()
        label_width = max(12, max((len(r) for r in frame.rows), default=12))
        for start in range(0, len(frame.columns), _EQ_PER_BLOCK):
            block = list(range(start, min(start + _EQ_PER_BLOCK, len(frame.columns))))
            widths = [label_width] + [12] * len(block)
            aligns = ["l"] + ["r"] * len(block)
            lines.append(
                fmt_row(["regressor"] + [frame.columns[j] for j in block], widths, aligns)
            )
            lines.append(rule(_WIDTH, "-"))
            for i, row_label in enumerate(frame.rows):
                cells: list[object] = [row_label]
                cells.extend(f"{frame.values[i, j]:+.5f}" for j in block)
                lines.append(fmt_row(cells, widths, aligns))
            if block[-1] + 1 < len(frame.columns):
                lines.append("")
        lines.append(rule(_WIDTH))
        return "\n".join(lines)


class IRFArray(list):
    """Impulse responses as ``irf[h][response][shock]`` — still a plain list.

    Subclassing ``list`` (not ``dict``) is the whole point: every access pattern
    that worked against the old nested-list return value still works, including
    ``np.array(irf)``, slicing and unpacking. The class only adds ``.response``,
    ``.names`` and ``.plot``.
    """

    def __init__(self, irfs, names: Sequence[str] | None = None, *, orth: bool = True,
                 cumulative: bool = False):
        super().__init__(irfs)
        k = len(self[0]) if len(self) else 0
        self.names = _default_names(None, k, names)
        self.orth = bool(orth)
        self.cumulative = bool(cumulative)

    # ---------------------------------------------------------------- access
    @property
    def neqs(self) -> int:
        return len(self.names)

    @property
    def horizon(self) -> int:
        """Largest horizon; the array holds ``horizon + 1`` matrices."""
        return max(len(self) - 1, 0)

    def _index(self, key) -> int:
        if isinstance(key, str):
            try:
                return self.names.index(key)
            except ValueError:
                raise KeyError(f"{key!r} is not one of {self.names}") from None
        return int(key)

    def to_array(self) -> np.ndarray:
        """The responses as a ``(horizon + 1, k, k)`` array."""
        return np.asarray(self, dtype=float)

    def response(self, i, j) -> np.ndarray:
        """Path of variable ``i``'s response to a shock in ``j``, over horizons.

        Both arguments accept an integer position or a variable name.
        """
        row, col = self._index(i), self._index(j)
        return np.array([self[h][row][col] for h in range(len(self))], dtype=float)

    def summary(self) -> str:
        kind = "orthogonalised" if self.orth else "reduced-form"
        if self.cumulative:
            kind = "cumulative " + kind
        return (
            f"IRFArray({kind}, horizons 0..{self.horizon}, "
            f"{self.neqs} variables: {', '.join(self.names)})"
        )

    def __repr__(self) -> str:
        # The bare nested list repr is thousands of unreadable floats; the shape
        # is what a human wants. list(irf) still shows the raw numbers.
        return self.summary()

    # ------------------------------------------------------------------ plot
    def plot(self, *, path=None, figsize=None, color=None):
        """A k x k grid of impulse responses: shocks across, responses down.

        Returns the :class:`matplotlib.figure.Figure`; saves it first when
        ``path`` is given. A zero reference line is always drawn.
        """
        plt = pyplot()
        k = self.neqs
        if k == 0:
            raise ValueError("no impulse responses to plot")
        arr = self.to_array()
        horizons = np.arange(arr.shape[0])
        palette = list(SERIES.values())

        figsize = figsize or (1.9 * k + 1.1, 1.55 * k + 0.9)
        fig, axes = plt.subplots(
            k, k, figsize=figsize, sharex=True, squeeze=False, constrained_layout=True
        )
        for i in range(k):          # response variable — rows
            for j in range(k):      # shock variable — columns
                ax = axes[i][j]
                apply_style(ax, zero_line=True)
                ax.plot(
                    horizons,
                    arr[:, i, j],
                    color=color or palette[j % len(palette)],
                    lw=1.6,
                    solid_capstyle="round",
                )
                ax.set_title(
                    f"{self.names[i]} ← shock {self.names[j]}",
                    fontsize=8,
                    color=INK,
                    pad=4,
                )
                ax.margins(x=0)
                ax.set_xlim(horizons[0], horizons[-1])
                if i == k - 1:
                    ax.set_xlabel("horizon", fontsize=8, color=INK_2)

        kind = "cumulative " if self.cumulative else ""
        kind += "orthogonalised" if self.orth else "reduced-form"
        fig.suptitle(f"{kind} impulse responses", fontsize=10, color=INK_2)
        if path is not None:
            fig.savefig(path, dpi=150)
        return fig


# --------------------------------------------------------------------------- #
# module-level helpers
# --------------------------------------------------------------------------- #
def var_fit(data, lags: int = 2, trend: str = "c", names: Sequence[str] | None = None):
    """Fit a VAR(p) and return a :class:`VARResults` (a dict, plus methods)."""
    return VARResults.fit(data, lags=lags, trend=trend, names=names)


def var_irf(
    data,
    lags: int = 2,
    horizon: int = 12,
    orth: bool = True,
    trend: str = "c",
    cumulative: bool = False,
    names: Sequence[str] | None = None,
) -> IRFArray:
    """Impulse responses as an :class:`IRFArray` (a list, plus methods)."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"data must be 2-D (obs x variables), got shape {array.shape}")
    raw = _compiled().var_irf(
        array, lags=lags, horizon=horizon, orth=orth, trend=trend, cumulative=cumulative
    )
    return IRFArray(
        raw,
        _default_names(data, array.shape[1], names),
        orth=orth,
        cumulative=cumulative,
    )
