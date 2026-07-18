"""Results facade for the GARCH-family volatility models.

Wraps :func:`tsecon.garch_fit` (GARCH / GJR / EGARCH by QMLE, with
Bollerslev-Wooldridge robust standard errors).

:class:`GARCHResults` **is** the dict ``garch_fit`` has always returned —
``res["params"]``, ``json.dumps``, ``pickle`` and ``**res`` unpacking all keep
working unchanged. It only *adds* rendering and a couple of accessors.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from ._base import Results, kv_line, param_table, rule
from ._plotting import BAND, MUTED, REF, SERIES, apply_style, pyplot

__all__ = ["GARCHResults"]

_WIDTH = 68

#: ``alpha``/``beta``/``gamma``, with or without a lag index: ``beta[1]``.
_TERM_RE = re.compile(r"^(alpha|beta|gamma)(?:\[(\d+)\])?$")

#: Above this, the variance process is effectively integrated.
_IGARCH_THRESHOLD = 0.99


def _term_index(names: Sequence[str]) -> dict[str, list[int]]:
    """Map ``alpha``/``beta``/``gamma`` to the positions holding them.

    Names drive this, never positions: the parameter vector's layout differs
    between ``vol``/``mean``/``dist`` settings (a ``mu`` in front for a constant
    mean, a ``nu`` at the back for Student-t errors), so indexing positionally
    would silently report the wrong number.
    """
    found: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        m = _TERM_RE.match(str(name).strip())
        if m is not None:
            found.setdefault(m.group(1), []).append(i)
    return found


class GARCHResults(Results):
    """Results of a GARCH-family QMLE fit.

    A ``dict`` with every key :func:`tsecon.garch_fit` returns —
    ``params``, ``param_names``, ``loglik``, ``aic``, ``bic``, ``se_mle``,
    ``se_robust``, ``conditional_volatility``, ``std_residuals`` (and
    ``variance_forecast`` when a horizon was requested) — plus
    :meth:`summary`, :meth:`persistence` and the plot methods.
    """

    _kind = "GARCHResults"

    #: Volatility spec this was fitted under, when known ("garch"/"gjr"/
    #: "egarch"). ``None`` when the object was built straight from a raw dict:
    #: GJR and EGARCH share a parameter naming scheme, so the family cannot be
    #: recovered from the names alone, and persistence is then left unreported
    #: rather than reported under the wrong formula.
    _vol: str | None = None

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def fit(cls, y, **kw: Any) -> "GARCHResults":
        """Fit with :func:`tsecon.garch_fit` and wrap the returned dict.

        All keyword arguments are forwarded verbatim (``vol``, ``mean``,
        ``dist``, ``p``, ``o``, ``q``, ``forecast_horizon``).
        """
        from .. import garch_fit  # local: avoids a package-level import cycle

        raw = garch_fit(y, **kw)
        out = cls(raw)
        out._vol = str(kw.get("vol", "garch")).lower()
        return out

    # ------------------------------------------------------------------ #
    # labels
    # ------------------------------------------------------------------ #
    def _names(self) -> list[str]:
        return [str(n) for n in self["param_names"]]

    def _mean_label(self) -> str:
        return "constant mean" if "mu" in self._names() else "zero mean"

    def _dist_label(self) -> str:
        return "Student-t errors" if "nu" in self._names() else "Normal errors"

    def model_name(self) -> str:
        """e.g. ``"GARCH(1,1)"`` / ``"GJR-GARCH(1,1,1)"`` / ``"EGARCH(1,1,1)"``."""
        terms = _term_index(self._names())
        p = len(terms.get("alpha", ()))
        o = len(terms.get("gamma", ()))
        q = len(terms.get("beta", ()))
        if self._vol == "egarch":
            return f"EGARCH({p},{o},{q})"
        if self._vol == "gjr":
            return f"GJR-GARCH({p},{o},{q})"
        if o:
            # Asymmetric, but which asymmetric model is not recoverable.
            return f"GARCH-family({p},{o},{q})"
        return f"GARCH({p},{q})"

    # ------------------------------------------------------------------ #
    # accessors
    # ------------------------------------------------------------------ #
    def params_named(self) -> dict[str, float]:
        """``{name: estimate}``, in the order the estimator returned them."""
        return {n: float(v) for n, v in zip(self._names(), self["params"])}

    def tstats(self) -> np.ndarray:
        """Robust t-statistics: ``params / se_robust``."""
        se = np.asarray(self["se_robust"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(self["params"], dtype=float) / se

    def volatility(self) -> np.ndarray:
        """Conditional volatility as a **standard deviation**, one per obs.

        ``garch_fit`` already returns a standard deviation, not a variance —
        it is ``sqrt(sigma2_t)`` at source, and its level matches the sample
        standard deviation of the input series rather than its square. Nothing
        is re-scaled here.
        """
        return np.asarray(self["conditional_volatility"], dtype=float)

    def persistence(self) -> float | None:
        """Volatility persistence, or ``None`` when it is not identifiable.

        ``alpha + beta`` for a GARCH(p,q); ``alpha + gamma/2 + beta`` for a GJR
        (the leverage term is active half the time under a symmetric error
        distribution); ``beta`` alone for an EGARCH, where persistence governs
        the *log* variance. ``None`` if the names carry no alpha/beta pair, or
        if the model is asymmetric but the family was not recorded — a wrong
        persistence is worse than no persistence.
        """
        terms = _term_index(self._names())
        params = np.asarray(self["params"], dtype=float)

        if self._vol == "egarch":
            if not terms.get("beta"):
                return None
            return float(sum(params[i] for i in terms["beta"]))

        if not terms.get("alpha") or not terms.get("beta"):
            return None

        total = float(sum(params[i] for i in terms["alpha"] + terms["beta"]))
        if terms.get("gamma"):
            if self._vol != "gjr":
                return None  # GJR vs EGARCH is ambiguous — refuse to guess
            total += 0.5 * float(sum(params[i] for i in terms["gamma"]))
        return total

    def _persistence_line(self) -> str | None:
        value = self.persistence()
        if value is None:
            return None
        if self._vol == "egarch":
            label = "beta (log-variance)"
        elif _term_index(self._names()).get("gamma"):
            label = "alpha + gamma/2 + beta"
        else:
            label = "alpha + beta"
        flag = "   [near-IGARCH]" if value > _IGARCH_THRESHOLD else ""
        return f"Persistence  {label} = {value:.5f}{flag}"

    # ------------------------------------------------------------------ #
    # summary
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        names = self._names()
        n_obs = len(self.volatility())
        lines = [
            rule(_WIDTH),
            f"{self.model_name()}, {self._mean_label()}, {self._dist_label()}  (QMLE)",
            "Standard errors: Bollerslev-Wooldridge robust",
            rule(_WIDTH),
            kv_line(
                [
                    ("No. obs", f"{n_obs}"),
                    ("Log-lik", f"{float(self['loglik']):.3f}"),
                    ("AIC", f"{float(self['aic']):.3f}"),
                    ("BIC", f"{float(self['bic']):.3f}"),
                ]
            ),
            rule(_WIDTH, "-"),
        ]
        lines += param_table(
            names,
            np.asarray(self["params"], dtype=float),
            se=np.asarray(self["se_robust"], dtype=float),
            tstats=self.tstats(),
            se_label="robust SE",
            width=_WIDTH,
        )
        persistence = self._persistence_line()
        if persistence is not None:
            lines += [rule(_WIDTH, "-"), persistence]
        lines.append(rule(_WIDTH))
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # plots
    # ------------------------------------------------------------------ #
    def plot_volatility(self, ax=None, path=None):
        """Plot the conditional volatility path. Returns the Figure.

        The plotted quantity is a **standard deviation**: ``garch_fit`` returns
        ``conditional_volatility`` already square-rooted (confirmed against the
        input series' scale — its mean sits near the sample standard deviation,
        not its square), so no ``sqrt`` is applied here.
        """
        plt = pyplot()
        sigma = self.volatility()

        if ax is None:
            fig, ax = plt.subplots(figsize=(7.4, 3.1))
        else:
            fig = ax.figure

        t = np.arange(len(sigma))
        ax.fill_between(t, 0.0, sigma, color=BAND, alpha=0.35, lw=0)
        ax.plot(t, sigma, color=SERIES["blue"], lw=1.0)
        ax.axhline(float(sigma.mean()), color=REF, lw=0.9, ls="--", zorder=1.5)
        apply_style(ax)
        ax.set_xlim(0, max(len(sigma) - 1, 1))
        ax.set_ylim(0.0, float(sigma.max()) * 1.08)
        ax.set_xlabel("observation", fontsize=8, color=MUTED)
        ax.set_ylabel("conditional sigma", fontsize=8, color=MUTED)
        ax.set_title(
            f"{self.model_name()} conditional volatility (std. dev.)",
            fontsize=9,
            loc="left",
        )
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=150)
        return fig

    def plot_diagnostics(self, path=None, lags: int = 20):
        """Standardized-residual histogram and ACF of squared std residuals.

        Two panels. Under a correctly specified volatility model the squared
        standardized residuals should show no remaining autocorrelation, so
        bars inside the +/-1.96/sqrt(n) band are the thing to look for.
        Returns the Figure.
        """
        plt = pyplot()
        z = np.asarray(self["std_residuals"], dtype=float)
        n = len(z)

        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
        hist_ax, acf_ax = axes

        hist_ax.hist(z, bins=40, color=SERIES["blue"], alpha=0.75, lw=0, density=True)
        grid = np.linspace(z.min(), z.max(), 200)
        normal = np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi)
        hist_ax.plot(grid, normal, color=SERIES["red"], lw=1.1)
        apply_style(hist_ax)
        hist_ax.set_title("standardized residuals", fontsize=9, loc="left")
        hist_ax.set_xlabel("z", fontsize=8, color=MUTED)

        acf = self._acf(z**2, lags)
        bound = 1.96 / np.sqrt(n)
        acf_ax.bar(np.arange(1, lags + 1), acf, width=0.55, color=SERIES["violet"], lw=0)
        acf_ax.axhline(bound, color=REF, lw=0.8, ls="--")
        acf_ax.axhline(-bound, color=REF, lw=0.8, ls="--")
        apply_style(acf_ax, zero_line=True)
        acf_ax.set_xlim(0.3, lags + 0.7)
        acf_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        acf_ax.set_title("ACF of squared std residuals", fontsize=9, loc="left")
        acf_ax.set_xlabel("lag", fontsize=8, color=MUTED)

        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=150)
        return fig

    @staticmethod
    def _acf(x: np.ndarray, lags: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x - x.mean()
        denom = float(x @ x)
        if denom == 0.0:
            return np.zeros(lags)
        return np.array([float(x[k:] @ x[:-k]) / denom for k in range(1, lags + 1)])
