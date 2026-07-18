"""Results objects for predictive regressions with a persistent predictor.

The single most valuable summary in the library, because it puts **three views
of the same regression side by side**:

* **OLS** — what everyone runs, and what is biased when the predictor is
  persistent and its innovation correlates with the return innovation.
* **Stambaugh (1999)** — the same slope with the analytic small-sample bias
  removed, so you can see *how much* bias there was.
* **IVX** (Kostakis-Magdalinos-Stamatogiannis 2015) — a slope whose Wald test
  is asymptotically chi-square *uniformly over the persistence of x*. This is
  the p-value to report.

Everything here is additive: :class:`PredictiveRegressionResults` is the plain
nested dict the compiled ``predictive_regression`` has always returned, with
rendering and a few accessors bolted on.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from ._base import Results, rule, fmt_row, kv_line, param_table
from ._plotting import pyplot, apply_style, SERIES, REF

__all__ = ["PredictiveRegressionResults", "IVXTestResults"]

_W = 68
_NORM = NormalDist()

#: Above this OLS-estimated AR(1) coefficient the predictor is called "highly
#: persistent" and the naive OLS t-statistic is flagged as unreliable.
PERSISTENCE_WARN = 0.95


def _naive_p(t: float) -> float:
    """Two-sided normal p-value for a t-statistic (deliberately *naive*)."""
    if not math.isfinite(t):
        return float("nan")
    return 2.0 * (1.0 - _NORM.cdf(abs(t)))


def _fnum(x: float, spec: str = "+.5f") -> str:
    """Format a float, degrading gracefully on nan/inf."""
    try:
        if not math.isfinite(float(x)):
            return "n/a"
        return format(float(x), spec)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return "n/a"


class PredictiveRegressionResults(Results):
    """``r(t+1) = a + b*x(t) + u(t+1)`` estimated three ways.

    A ``dict`` with the original nested keys ``ols``, ``stambaugh``, ``ivx``
    and ``nobs`` untouched, plus :meth:`summary`, :meth:`significant`,
    :meth:`conf_int` and :meth:`plot_estimates`.
    """

    _kind = "PredictiveRegressionResults"

    #: Label used for the predictor in the summary (see :meth:`fit`).
    _name = "x"

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def fit(cls, r, x, *, name: str = "x", **kw) -> "PredictiveRegressionResults":
        """Run ``tsecon.predictive_regression(r, x, **kw)`` and wrap the result.

        ``name`` only labels the predictor in the summary; it changes no key.
        Remaining keyword arguments (``cz``, ``alpha``) pass straight through.
        """
        from .._core import predictive_regression

        raw = predictive_regression(r, x, **kw)
        out = cls(raw)
        out._name = str(name)
        return out

    # ------------------------------------------------------------------ #
    # convenience views (the dict keys stay authoritative)
    # ------------------------------------------------------------------ #
    @property
    def ols(self) -> dict:
        """The ``ols`` sub-dict: ``alpha``, ``beta``, ``se``, ``tstat``."""
        return self["ols"]

    @property
    def stambaugh(self) -> dict:
        """The ``stambaugh`` sub-dict (bias-corrected slope and its inputs)."""
        return self["stambaugh"]

    @property
    def ivx(self) -> dict:
        """The ``ivx`` sub-dict: ``beta_ivx``, ``wald``, ``pvalue``, ``rz``."""
        return self["ivx"]

    def betas(self) -> dict:
        """The three slope estimates, keyed by estimator."""
        return {
            "ols": float(self["ols"]["beta"]),
            "stambaugh": float(self["stambaugh"]["beta_corrected"]),
            "ivx": float(self["ivx"]["beta_ivx"]),
        }

    def ivx_se(self) -> float:
        """Standard error implied by the IVX Wald statistic.

        The compiled function reports ``wald = (beta_ivx / se_ivx)**2`` for the
        single-regressor case, so ``se_ivx = |beta_ivx| / sqrt(wald)``. Returned
        as ``nan`` when the Wald statistic or slope is degenerate.
        """
        beta = float(self["ivx"]["beta_ivx"])
        wald = float(self["ivx"]["wald"])
        if wald > 0.0 and beta != 0.0 and math.isfinite(wald) and math.isfinite(beta):
            return abs(beta) / math.sqrt(wald)
        return float("nan")

    def conf_int(self, level: float = 0.95) -> tuple[float, float]:
        """Normal confidence interval for the **IVX** slope at ``level``."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level!r}")
        beta = float(self["ivx"]["beta_ivx"])
        se = self.ivx_se()
        z = _NORM.inv_cdf(0.5 + level / 2.0)
        return (beta - z * se, beta + z * se)

    def rho(self) -> float:
        """OLS estimate of the predictor's AR(1) coefficient."""
        return float(self["stambaugh"]["rho_ols"])

    def is_persistent(self, threshold: float = PERSISTENCE_WARN) -> bool:
        """Whether the predictor's estimated ``rho`` exceeds ``threshold``."""
        r = self.rho()
        return math.isfinite(r) and r > threshold

    # ------------------------------------------------------------------ #
    # the headline decision
    # ------------------------------------------------------------------ #
    def significant(self, level: float = 0.05) -> bool:
        """Is the predictor significant at ``level``, **by the IVX Wald test**?

        Deliberately *not* the OLS t-statistic. When ``x`` is persistent and
        its innovation correlates with the return innovation, OLS is biased in
        finite samples and the naive t-test over-rejects — sometimes badly, so
        a nominal 5% test can reject far more than 5% of the time under the
        null. The IVX Wald statistic is asymptotically chi-square uniformly
        over the persistence of ``x``, so its p-value is the one that means
        what it says. That is what this method reads.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level!r}")
        p = float(self["ivx"]["pvalue"])
        return math.isfinite(p) and p < level

    # ------------------------------------------------------------------ #
    # summary
    # ------------------------------------------------------------------ #
    def summary(self, level: float = 0.05) -> str:
        ols, stam, ivx = self["ols"], self["stambaugh"], self["ivx"]
        name = self._name

        b_ols = float(ols["beta"])
        se_ols = float(ols["se"])
        t_ols = float(ols["tstat"])
        p_ols = _naive_p(t_ols)

        b_stam = float(stam["beta_corrected"])
        se_stam = float(stam["se"])
        bias = float(stam["bias_term"])
        t_stam = b_stam / se_stam if se_stam else float("nan")
        p_stam = _naive_p(t_stam)

        b_ivx = float(ivx["beta_ivx"])
        se_ivx = self.ivx_se()
        wald = float(ivx["wald"])
        p_ivx = float(ivx["pvalue"])

        left = f"Predictive regression   r(t+1) = a + b*{name}(t)"
        right = f"IVX p = {p_ivx:.4f}"
        pad = max(2, _W - len(left) - len(right))
        title = f"{left}{' ' * pad}{right}"

        lines = [rule(_W), title, rule(_W)]
        lines.append(
            kv_line(
                [
                    ("nobs", self.get("nobs", "n/a")),
                    (f"rho({name})", _fnum(stam["rho_ols"], ".4f")),
                    ("IVX rz", _fnum(ivx["rz"], ".4f")),
                ]
            )
        )
        lines.append(
            kv_line(
                [
                    ("intercept", _fnum(ols["alpha"])),
                    ("Stambaugh bias removed", _fnum(bias)),
                ]
            )
        )
        lines.append(rule(_W, "-"))

        widths = [11, 11, 10, 13, 9]
        aligns = ["l", "r", "r", "r", "r"]
        lines.append(
            fmt_row(["estimator", "beta", "std err", "test stat", "p-value"], widths, aligns)
        )
        lines.append(rule(_W, "-"))
        lines.append(
            fmt_row(
                ["OLS", _fnum(b_ols), _fnum(se_ols, ".5f"), f"t {t_ols:+.3f}", f"{p_ols:.4f}"],
                widths,
                aligns,
            )
        )
        lines.append(
            fmt_row(
                [
                    "Stambaugh",
                    _fnum(b_stam),
                    _fnum(se_stam, ".5f"),
                    f"t {t_stam:+.3f}",
                    f"{p_stam:.4f}",
                ],
                widths,
                aligns,
            )
        )
        lines.append(
            fmt_row(
                ["IVX", _fnum(b_ivx), _fnum(se_ivx, ".5f"), f"W {wald:.4f}", f"{p_ivx:.4f}"],
                widths,
                aligns,
            )
        )
        lines.append(rule(_W, "-"))

        # --- interpretation ------------------------------------------------
        lines.append(f"Report the IVX Wald p-value ({p_ivx:.4f}). It is valid whatever the")
        lines.append(f"persistence of {name}; the OLS t over-rejects when {name} is persistent,")
        lines.append("so the OLS and Stambaugh p-values above are naive normal ones.")

        if self.is_persistent():
            lines.append(
                f"WARNING: rho({name}) = {self.rho():.4f} > {PERSISTENCE_WARN:g}, so {name} "
                f"is highly persistent"
            )
            lines.append(
                "         and the OLS t-statistic is unreliable here."
            )

        sig = self.significant(level)
        pct = f"{level:.0%}"
        if sig:
            lines.append(f"At the {pct} level IVX rejects b = 0: {name} predicts r(t+1).")
        else:
            lines.append(f"At the {pct} level IVX does not reject b = 0: no evidence")
            lines.append(f"that {name} predicts r(t+1).")
        if (not sig) and math.isfinite(p_ols) and p_ols < level:
            lines.append(
                "Note: OLS alone would have called this significant. That gap is"
            )
            lines.append("      exactly the over-rejection IVX is designed to remove.")

        lines.append(rule(_W))
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # plot
    # ------------------------------------------------------------------ #
    def plot_estimates(self, ax=None, *, level: float = 0.95, path: str | None = None):
        """Forest plot of the three slope estimates with confidence intervals.

        One dot per estimator with a horizontal interval, and a zero reference
        line — so a slope that OLS calls significant and IVX does not shows up
        as an interval that crosses zero. Returns the ``Figure``.
        """
        plt = pyplot()
        z = _NORM.inv_cdf(0.5 + level / 2.0)

        se_ols = float(self["ols"]["se"])
        se_stam = float(self["stambaugh"]["se"])
        se_ivx = self.ivx_se()
        rows = [
            ("IVX", float(self["ivx"]["beta_ivx"]), se_ivx, SERIES["blue"]),
            ("Stambaugh", float(self["stambaugh"]["beta_corrected"]), se_stam, SERIES["aqua"]),
            ("OLS", float(self["ols"]["beta"]), se_ols, SERIES["red"]),
        ]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.2, 2.6))
        else:
            fig = ax.figure

        ax.axvline(0.0, color=REF, lw=0.9, zorder=1.5)
        for i, (_label, beta, se, colour) in enumerate(rows):
            if math.isfinite(se):
                ax.plot(
                    [beta - z * se, beta + z * se],
                    [i, i],
                    color=colour,
                    lw=2.4,
                    solid_capstyle="round",
                    alpha=0.55,
                    zorder=2,
                )
            ax.plot([beta], [i], "o", color=colour, ms=6.5, zorder=3)

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.margins(x=0.06)
        ax.set_xlabel(f"slope on {self._name}(t)", fontsize=8)
        ax.set_title(
            f"Predictive slope, {level:.0%} intervals  "
            f"(IVX p = {float(self['ivx']['pvalue']):.4f})",
            fontsize=9,
            loc="left",
        )
        apply_style(ax)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=150)
        return fig


class IVXTestResults(Results):
    """Joint IVX predictability test across several persistent predictors.

    Wraps ``tsecon.ivx_test``: keys ``beta_ivx``, ``wald``, ``pvalue``, ``rz``,
    ``nobs`` and ``nregressors`` are preserved exactly.
    """

    _kind = "IVXTestResults"

    #: Predictor labels used in the summary.
    _names: tuple[str, ...] = ()

    @classmethod
    def fit(cls, r, xs, *, names=None, **kw) -> "IVXTestResults":
        """Run ``tsecon.ivx_test(r, xs, **kw)`` and wrap the result."""
        from .._core import ivx_test

        raw = ivx_test(r, xs, **kw)
        out = cls(raw)
        if names is not None:
            out._names = tuple(str(n) for n in names)
        return out

    def names(self) -> list[str]:
        """Predictor labels, defaulting to ``x1 .. xk``."""
        k = int(self["nregressors"])
        if len(self._names) == k:
            return list(self._names)
        return [f"x{i + 1}" for i in range(k)]

    def significant(self, level: float = 0.05) -> bool:
        """Does the *joint* IVX Wald test reject ``H0: beta = 0`` at ``level``?

        As in the single-predictor case this reads the IVX p-value, whose
        chi-square limit holds uniformly over the predictors' persistence.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level!r}")
        p = float(self["pvalue"])
        return math.isfinite(p) and p < level

    def summary(self, level: float = 0.05) -> str:
        k = int(self["nregressors"])
        wald = float(self["wald"])
        p = float(self["pvalue"])

        left = f"Joint IVX test   H0: b = 0 for all {k} predictors"
        right = f"IVX p = {p:.4f}"
        pad = max(2, _W - len(left) - len(right))
        lines = [rule(_W), f"{left}{' ' * pad}{right}", rule(_W)]
        lines.append(
            kv_line(
                [
                    ("nobs", self.get("nobs", "n/a")),
                    ("predictors", k),
                    (f"Wald chi2({k})", _fnum(wald, ".4f")),
                    ("IVX rz", _fnum(self["rz"], ".4f")),
                ]
            )
        )
        lines.append(rule(_W, "-"))
        betas = [float(b) for b in self["beta_ivx"]]
        lines.extend(param_table(self.names(), betas, value_label="beta_ivx", width=_W))
        lines.append(rule(_W, "-"))
        pct = f"{level:.0%}"
        if self.significant(level):
            lines.append(f"At the {pct} level the predictors jointly predict r(t+1).")
        else:
            lines.append(f"At the {pct} level there is no joint evidence of predictability.")
        lines.append("Individual slopes above are point estimates; the p-value is joint.")
        lines.append(rule(_W))
        return "\n".join(lines)
