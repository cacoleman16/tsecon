"""Results facade for ARIMA fits.

:class:`ARIMAResults` wraps the dict returned by :func:`tsecon.arima_fit`
without changing it: every documented key (``params``, ``param_names``,
``loglik``, ``aic``, ``bic``, ``forecast_mean``, ``forecast_se``,
``residuals``, and the optional ``forecast_lower``/``forecast_upper``/
``conf_alpha``) is still present and still equal to what the compiled
function produced. The subclass only *adds* a rendered
:meth:`~ARIMAResults.summary`, a tidy :meth:`~ARIMAResults.forecast_frame`,
and the fan chart in :meth:`~ARIMAResults.plot_forecast`.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Sequence

import numpy as np

from ._base import Results, kv_line, param_table, rule
from ._plotting import BAND, INK, INK_2, MUTED, REF, SERIES, apply_style, pyplot

__all__ = ["ARIMAResults"]

_WIDTH = 68

#: Nested bands drawn by the fan chart, innermost first.
_FAN_LEVELS = (0.50, 0.80, 0.95)


def _z(level: float) -> float:
    """Two-sided normal critical value for a coverage ``level`` in (0, 1)."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    return NormalDist().inv_cdf(0.5 + level / 2.0)


class ARIMAResults(Results):
    """The dict returned by :func:`tsecon.arima_fit`, plus rendering.

    >>> res = ARIMAResults.fit(y, 1, 0, 1, forecast_steps=12)
    >>> res["aic"]            # the dict contract is untouched
    >>> print(res.summary())  # ... and now it prints itself
    """

    _kind = "ARIMAResults"

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(self, *args, order=(1, 0, 0), y=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._order = tuple(int(v) for v in order)
        # The input series lives on the instance, not in the dict: the fan
        # chart wants the history, but `to_dict()` must stay exactly the set
        # of keys the compiled function documents.
        self._y = None if y is None else np.asarray(y, dtype=float)

    @classmethod
    def fit(cls, y, p: int = 1, d: int = 0, q: int = 0, **kw) -> "ARIMAResults":
        """Fit an ARIMA(p, d, q) and wrap the result.

        ``**kw`` is forwarded verbatim to :func:`tsecon.arima_fit` — notably
        ``constant``, ``forecast_steps`` and ``conf_alpha``.
        """
        from .. import arima_fit  # local: avoids a package-level import cycle

        raw = arima_fit(y, p, d, q, **kw)
        return cls(raw, order=(p, d, q), y=y)

    # ------------------------------------------------------------------ #
    # accessors
    # ------------------------------------------------------------------ #
    @property
    def order(self) -> tuple:
        """The ``(p, d, q)`` order this fit was estimated at."""
        return self._order

    @property
    def y(self):
        """The input series, or ``None`` if the object was built without it."""
        return self._y

    @property
    def nobs(self) -> int:
        """Number of observations used by the likelihood."""
        if self._y is not None:
            return int(np.asarray(self._y).shape[0])
        return int(np.asarray(self["residuals"]).shape[0]) + self._order[1]

    @property
    def has_forecast(self) -> bool:
        """True when the fit was asked for (and returned) a forecast."""
        mean = self.get("forecast_mean")
        return mean is not None and int(np.asarray(mean).shape[0]) > 0

    @property
    def forecast_steps(self) -> int:
        """Length of the stored forecast (0 when there is none)."""
        if not self.has_forecast:
            return 0
        return int(np.asarray(self["forecast_mean"]).shape[0])

    def params_dict(self) -> dict:
        """``{name: estimate}`` for the fitted parameters, in order."""
        return {
            str(n): float(v)
            for n, v in zip(self["param_names"], np.asarray(self["params"], float))
        }

    def _require_forecast(self, what: str) -> None:
        if not self.has_forecast:
            p, d, q = self._order
            raise ValueError(
                f"{what} needs a forecast, but this ARIMA({p},{d},{q}) fit has "
                "none: `forecast_steps` was 0 or absent.\n"
                f"    res = ARIMAResults.fit(y, {p}, {d}, {q}, forecast_steps=12)"
            )

    # ------------------------------------------------------------------ #
    # forecast table
    # ------------------------------------------------------------------ #
    def forecast_frame(self, level: float = 0.95):
        """Forecast path as a numpy structured array.

        Fields: ``step`` (1-based), ``mean``, ``se``, ``lower``, ``upper``.
        Column access works directly (``frame["upper"]``), and the array feeds
        ``pandas.DataFrame(frame)`` or ``polars.from_numpy(frame)`` unchanged,
        so no dataframe library is required to hold the result.

        When ``level`` matches the ``conf_alpha`` the fit was run with, the
        bands the compiled routine returned are reused verbatim rather than
        recomputed.
        """
        self._require_forecast("forecast_frame()")
        mean = np.asarray(self["forecast_mean"], dtype=float)
        se = np.asarray(self["forecast_se"], dtype=float)

        alpha = self.get("conf_alpha")
        stored = "forecast_lower" in self and "forecast_upper" in self
        if stored and alpha is not None and abs((1.0 - float(alpha)) - level) < 1e-12:
            lower = np.asarray(self["forecast_lower"], dtype=float)
            upper = np.asarray(self["forecast_upper"], dtype=float)
        else:
            half = _z(level) * se
            lower, upper = mean - half, mean + half

        frame = np.empty(
            mean.shape[0],
            dtype=[
                ("step", "i8"),
                ("mean", "f8"),
                ("se", "f8"),
                ("lower", "f8"),
                ("upper", "f8"),
            ],
        )
        frame["step"] = np.arange(1, mean.shape[0] + 1)
        frame["mean"] = mean
        frame["se"] = se
        frame["lower"] = lower
        frame["upper"] = upper
        return frame

    # ------------------------------------------------------------------ #
    # summary
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        p, d, q = self._order
        names: Sequence[str] = [str(n) for n in self["param_names"]]
        values = np.asarray(self["params"], dtype=float)
        resid = np.asarray(self["residuals"], dtype=float)

        lines = [
            rule(_WIDTH),
            f"ARIMA({p},{d},{q})".ljust(28)
            + f"log-likelihood {float(self['loglik']):>13.4f}",
            rule(_WIDTH),
            kv_line(
                [
                    ("No. Observations", f"{self.nobs:>6d}"),
                    ("AIC", f"{float(self['aic']):>11.3f}"),
                    ("BIC", f"{float(self['bic']):>11.3f}"),
                ]
            ),
            kv_line(
                [
                    ("Residual s.d.", f"{resid.std(ddof=0):>9.4f}"),
                    ("Forecast steps", f"{self.forecast_steps:>4d}"),
                ]
            ),
            rule(_WIDTH, "-"),
        ]
        lines += param_table(names, values, width=_WIDTH)
        lines.append(rule(_WIDTH))
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # fan chart
    # ------------------------------------------------------------------ #
    def plot_forecast(
        self, y=None, level: float = 0.95, ax=None, path=None, max_history="auto"
    ):
        """Fan chart: history, forecast mean, and nested prediction bands.

        Bands are drawn at 50/80/``level``×100 percent (any of the defaults at
        or above ``level`` are dropped), innermost darkest. ``y`` overrides the
        stored history; pass ``ax`` to draw into an existing Axes and ``path``
        to save the figure.

        ``max_history`` caps how many trailing observations are shown, because
        a fan twelve steps wide is invisible next to a thousand-point history.
        The default keeps roughly six forecast-widths of context (at least 40
        points); pass ``None`` to plot the whole series, or an integer to set
        the window yourself. The x-axis stays on the original period index
        either way.
        """
        self._require_forecast("plot_forecast()")
        plt = pyplot()

        hist = self._y if y is None else np.asarray(y, dtype=float)
        mean = np.asarray(self["forecast_mean"], dtype=float)
        se = np.asarray(self["forecast_se"], dtype=float)
        h = mean.shape[0]

        levels = sorted({lv for lv in _FAN_LEVELS if lv < level} | {float(level)})

        if ax is None:
            fig, ax = plt.subplots(figsize=(7.6, 3.9), constrained_layout=True)
        else:
            fig = ax.figure

        # Anchor the fan on the last observation so the bands emerge from the
        # history rather than floating away from it.
        if hist is not None and hist.shape[0]:
            n = int(hist.shape[0])
            if max_history == "auto":
                window = min(n, max(40, 6 * h))
            elif max_history is None:
                window = n
            else:
                window = min(n, max(2, int(max_history)))
            start = n - window
            hx = np.arange(start, n, dtype=float)
            ax.plot(hx, hist[start:], color=INK, lw=1.2, label="observed", zorder=3)
            fx = np.arange(n - 1, n + h, dtype=float)
            fmean = np.concatenate([hist[-1:], mean])
            fse = np.concatenate([[0.0], se])
            x_left = float(start)
        else:
            n = 0
            fx = np.arange(1, h + 1, dtype=float)
            fmean, fse = mean, se
            x_left = 1.0

        # Widest band first so the narrow ones stack visibly on top.
        for i, lv in enumerate(reversed(levels)):
            half = _z(lv) * fse
            ax.fill_between(
                fx,
                fmean - half,
                fmean + half,
                color=BAND,
                alpha=0.22 + 0.20 * i,
                lw=0,
                zorder=2,
                label=f"{lv * 100:g}%",
            )

        ax.plot(fx, fmean, color=SERIES["blue"], lw=1.6, label="forecast", zorder=4)
        if n:
            ax.axvline(n - 1, color=REF, lw=0.8, ls=(0, (3, 3)), zorder=1.2)

        apply_style(ax)
        ax.set_xlim(x_left, fx[-1])
        ax.margins(x=0)
        ax.set_xlabel("period", fontsize=8, color=INK_2)

        p, d, q = self._order
        ax.set_title(
            f"ARIMA({p},{d},{q}) forecast · {h} steps",
            loc="left",
            fontsize=10,
            color=INK,
            pad=13,
        )
        # Legend in the title band, never over the data.
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.005),
            ncol=len(levels) + (2 if n else 1),
            frameon=False,
            fontsize=7.5,
            handlelength=1.4,
            handletextpad=0.5,
            columnspacing=1.1,
            borderaxespad=0.0,
            labelcolor=MUTED,
        )

        if path is not None:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig
