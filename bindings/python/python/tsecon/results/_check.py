"""Results facade for the :func:`tsecon.check_series` diagnostic battery.

:class:`CheckSeriesResults` **is** the dict ``check_series`` has always
returned — ``res["recommendations"]``, ``json.dumps``, ``pickle`` and ``**res``
unpacking all keep working unchanged. It only *adds* a sectioned summary and a
2x2 diagnostic figure on top.

The rendering rule is strict: the summary quotes numbers that live in the
dict and nothing else. The battery's design stance — evidence in families,
multiple-testing arithmetic shown, never silently corrected — is preserved
verbatim; the summary is a *view* of the report, not a second opinion.
"""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np

from ._base import Results, fmt_row, kv_line, rule
from ._plotting import MUTED, REF, SERIES, apply_style, pyplot

__all__ = ["CheckSeriesResults", "check_series"]

_WIDTH = 68


def _compiled():
    """The compiled primitive module, imported lazily to dodge import cycles."""
    from .. import _core

    return _core


def _wrap(text: Any, indent: str = "  ") -> list[str]:
    """House-width wrapped lines, every line carrying ``indent``."""
    return textwrap.wrap(
        str(text), width=_WIDTH, initial_indent=indent, subsequent_indent=indent
    )


def _p(value: Any) -> str:
    return f"{float(value):.3g}"


def _lags_text(lags: list) -> str:
    return ", ".join(str(v) for v in lags) if lags else "none"


class CheckSeriesResults(Results):
    """The ``check_series`` report — the same dict, plus rendering.

    A ``dict`` with every key :func:`tsecon.check_series` returns for its
    ``kind`` (``"univariate"``: descriptives, outliers, stationarity,
    analysis_scale, serial_correlation, arch_effects, normality, breaks,
    long_memory, seasonality, tests_run, multiple_testing, recommendations;
    ``"multivariate"``: per_series, integration_summary, cointegration,
    var_lag_selection, stability, tests_run, multiple_testing,
    recommendations) — plus :meth:`summary` and :meth:`plot_diagnostics`.
    """

    _kind = "CheckSeriesResults"

    #: The coerced input series/panel, kept for plotting. Deliberately an
    #: attribute, not a key: ``to_dict()``/``json.dumps`` stay exactly the
    #: report ``check_series`` documents. ``None`` when the object was built
    #: straight from a raw dict.
    _data: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def run(cls, data, **kw: Any) -> "CheckSeriesResults":
        """Run the battery with :func:`tsecon.check_series` and wrap the dict.

        All keyword arguments are forwarded verbatim (``seasonal_period``,
        ``lags``, ``alpha``, ``max_breaks``, ``trim``).
        """
        from .. import check_series  # local: avoids a package-level import cycle

        raw = check_series(data, **kw)
        out = cls(raw)
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 1:  # same squeeze the battery does
            arr = arr[:, 0]
        out._data = arr
        return out

    # ------------------------------------------------------------------ #
    # summary
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        lines = self._header_lines()
        if self["kind"] == "univariate":
            lines += self._univariate_family_lines()
        else:
            lines += self._multivariate_family_lines()
        lines += self._recommendation_lines()
        lines += self._footer_lines()
        return "\n".join(lines)

    # ------------------------------------------------------- header/footer
    def _header_lines(self) -> list[str]:
        alpha = f"{float(self['alpha']):g}"
        if self["kind"] == "univariate":
            head = f"check_series — univariate, n={self['n']}, alpha={alpha}"
        else:
            head = (
                f"check_series — multivariate, n={self['n']}, k={self['k']}, "
                f"alpha={alpha}"
            )
        lines = [rule(_WIDTH), head, rule(_WIDTH)]

        if self["kind"] == "univariate":
            d, outliers = self["descriptives"], self["outliers"]
            lines.append(
                kv_line(
                    [
                        ("mean", f"{d['mean']:+.4f}"),
                        ("sd", f"{d['sd']:.4f}"),
                        ("skew", f"{d['skew']:+.3f}"),
                        ("ex.kurt", f"{d['excess_kurtosis']:+.3f}"),
                    ]
                )
            )
            lines.append(
                kv_line(
                    [
                        ("min", f"{d['min']:+.4f}"),
                        ("max", f"{d['max']:+.4f}"),
                        ("outliers", f"{outliers['count']}"),
                    ]
                )
            )
            lines += _wrap(f"outlier screen: {outliers['method']}", "  ")
        else:
            summary = self["integration_summary"]
            lines.append(
                kv_line(
                    [
                        ("stationary", f"{summary['n_stationary']}"),
                        (
                            "difference-recommended",
                            f"{summary['n_difference_recommended']}",
                        ),
                    ]
                )
            )
            lines += _wrap(summary["text"], "  ")
        return lines

    def _footer_lines(self) -> list[str]:
        mt = self["multiple_testing"]
        footer = (
            f"{mt['n_tests']} hypothesis tests at alpha={float(mt['alpha']):g} "
            f"- expect ~{mt['expected_false_rejections']:.2f} false alarms "
            f"from the {mt['n_true_null']} true-null tests on a clean series."
        )
        return [rule(_WIDTH), footer, rule(_WIDTH)]

    # --------------------------------------------------- univariate blocks
    def _univariate_family_lines(self) -> list[str]:
        lines: list[str] = []

        # stationarity + the scale decision it implies
        st, scale = self["stationarity"], self["analysis_scale"]
        lines += [rule(_WIDTH, "-"), "Stationarity (ADF + KPSS) — on level"]
        lines.append(
            f"  quadrant {st['quadrant']} -> recommendation "
            f"{st['recommendation']}"
        )
        lines.append(
            f"  adf    stat {st['adf_statistic']:+.4f}   "
            f"p {_p(st['adf_p_value'])}"
        )
        lines.append(
            f"  kpss   stat {st['kpss_statistic']:+.4f}   "
            f"p {_p(st['kpss_p_value'])}"
        )
        lines.append(f"  analysis scale: {scale['scale']}")
        lines += _wrap(scale["rationale"], "    ")

        # serial correlation
        sc = self["serial_correlation"]
        lb = sc["ljung_box"]
        orders = sc["suggested_arma_orders"]
        lines += [
            rule(_WIDTH, "-"),
            f"Serial correlation — on {sc['computed_on']}",
            f"  ljung_box (lags 1..{lb['lags'][-1]})   "
            f"stat {float(lb['lb_stat'][-1]):.4f}   "
            f"p {_p(lb['lb_pvalue'][-1])}",
            f"  significant ACF lags: {_lags_text(sc['significant_acf_lags'])}",
            f"  significant PACF lags: "
            f"{_lags_text(sc['significant_pacf_lags'])}",
            f"  suggested starting orders   p {orders['p']}   q {orders['q']}",
        ]

        # ARCH
        arch = self["arch_effects"]
        lines += [
            rule(_WIDTH, "-"),
            f"ARCH effects — on {arch['computed_on']}",
            f"  arch_lm   stat {arch['statistic']:.4f}   "
            f"p {_p(arch['p_value'])}   "
            f"{'rejected' if arch['rejected'] else 'not rejected'}",
        ]

        # normality
        norm = self["normality"]
        lines += [
            rule(_WIDTH, "-"),
            f"Normality (Jarque-Bera) — on {norm['computed_on']}",
            f"  jarque_bera   stat {norm['statistic']:.4f}   "
            f"p {_p(norm['p_value'])}   "
            f"{'rejected' if norm['rejected'] else 'not rejected'}",
            f"  skew {norm['skewness']:+.3f}   "
            f"ex.kurt {norm['excess_kurtosis']:+.3f}",
        ]

        # breaks
        br = self["breaks"]
        lines += [rule(_WIDTH, "-"), f"Structural breaks — on {br['computed_on']}"]
        if br["sup_f"] is not None:
            sf = br["sup_f"]
            lines.append(
                f"  sup_f_test   stat {sf['stat']:.4f}   "
                f"p {_p(sf['p_value'])}   break_date {sf['break_date']}"
            )
        if br["bai_perron"] is not None:
            bp = br["bai_perron"]
            lines.append(
                f"  bai_perron   n_breaks {bp['n_breaks']}   "
                f"dates {bp['break_dates']}"
            )
            lines.append(
                f"    95% CI   lower {bp['ci_95']['lower']}   "
                f"upper {bp['ci_95']['upper']}"
            )
        if br["skipped_reason"] is not None:
            lines += _wrap(br["skipped_reason"], "  ")

        # long memory — always estimated on the level
        lm = self["long_memory"]
        gph = lm["gph"]
        lines += [
            rule(_WIDTH, "-"),
            f"Long memory (GPH) — on {lm['computed_on']}",
            f"  gph (level)   d {gph['d']:+.4f}   se {gph['se']:.4f}   "
            f"m {gph['m']}",
        ]
        if lm["gph_on_differences"] is not None:
            gd = lm["gph_on_differences"]
            lines.append(
                f"  gph (differences)   d {gd['d']:+.4f}   "
                f"se {gd['se']:.4f}   m {gd['m']}"
            )
        lines += _wrap(lm["joint_interpretation"], "  ")

        # seasonality
        se = self["seasonality"]
        lines += [rule(_WIDTH, "-"), f"Seasonality — on {se['computed_on']}"]
        if se["seasonal_period"] is not None:
            lines.append(f"  seasonal_period {se['seasonal_period']}")
            for entry in se["acf_at_seasonal_lags"] or []:
                lines.append(f"  lag {entry['lag']}   acf {entry['acf']:+.4f}")
            ordinate = se["periodogram_ordinate"]
            if ordinate is not None:
                period = ordinate["period"]
                lines.append(
                    f"  periodogram ordinate   "
                    f"frequency {ordinate['frequency']:.4f}   "
                    f"period {'None' if period is None else f'{period:.2f}'}   "
                    f"psd {ordinate['psd']:.4g}"
                )
        elif se["detected_period"] is not None:
            lines.append(f"  detected_period {se['detected_period']:.2f}")
        else:
            lines.append("  detected_period None")
        if se["note"]:
            lines += _wrap(se["note"], "  ")
        return lines

    # ------------------------------------------------- multivariate blocks
    def _multivariate_family_lines(self) -> list[str]:
        lines: list[str] = []
        pvalues = {t["name"]: t["pvalue"] for t in self["tests_run"]}

        # per-series verdicts, with the ADF/KPSS p-values from tests_run
        lines += [
            rule(_WIDTH, "-"),
            "Per-series integration (ADF + KPSS) — on level",
        ]
        widths = [7, 13, 15, 9, 9]
        aligns = ["l", "l", "l", "r", "r"]
        lines.append(
            fmt_row(["series", "verdict", "recommendation", "adf p", "kpss p"],
                    widths, aligns)
        )
        lines.append(rule(_WIDTH, "-"))
        for entry in self["per_series"]:
            j = entry["index"]
            lines.append(
                fmt_row(
                    [
                        f"y{j + 1}",
                        entry["verdict"],
                        entry["recommendation"],
                        _p(pvalues[f"adf (series {j})"]),
                        _p(pvalues[f"kpss (series {j})"]),
                    ],
                    widths,
                    aligns,
                )
            )

        # cointegration
        co = self["cointegration"]
        lines += [rule(_WIDTH, "-"), "Cointegration (Johansen)"]
        if "skipped_reason" in co:
            lines += _wrap(co["skipped_reason"], "  ")
        else:
            lines.append(f"  {co['method']}")
            lines.append(
                kv_line(
                    [
                        ("  rank (trace 5%)", f"{co['rank']}"),
                        ("rank (max-eig 5%)", f"{co['rank_max_eig_5pct']}"),
                    ]
                )
            )
            lines += _wrap(co["interpretation"], "  ")

        # VAR lag selection
        sel = self["var_lag_selection"]
        lines += [rule(_WIDTH, "-")]
        if "skipped_reason" in sel:
            lines.append("VAR lag selection")
            lines += _wrap(sel["skipped_reason"], "  ")
        else:
            lines.append(f"VAR lag selection — on {sel['scale']}")
            lines += _wrap(sel["scale_note"], "  ")
            table_widths = [5, 12, 12, 12]
            table_aligns = ["l", "r", "r", "r"]
            lines.append(fmt_row(["lag", "aic", "bic", "hqic"],
                                 table_widths, table_aligns))
            for i, lag in enumerate(sel["lags_tried"]):
                lines.append(
                    fmt_row(
                        [
                            f"{lag}",
                            f"{sel['aic'][i]:.4f}",
                            f"{sel['bic'][i]:.4f}",
                            f"{sel['hqic'][i]:.4f}",
                        ],
                        table_widths,
                        table_aligns,
                    )
                )
            lines.append(f"  selected by BIC: {sel['selected_by_bic']}")

        # stability of the selected VAR
        stab = self["stability"]
        lines += [rule(_WIDTH, "-")]
        if "skipped_reason" in stab:
            lines.append("Stability")
            lines += _wrap(stab["skipped_reason"], "  ")
        else:
            verdict = "stable" if stab["is_stable"] else "UNSTABLE"
            lines.append(f"Stability — on {stab['scale']}")
            lines.append(
                f"  VAR({stab['lags']}) is {verdict}   "
                f"min_root {stab['min_root']:.4f}"
            )
            lines += _wrap(stab["note"], "  ")
        return lines

    # ------------------------------------------------------ recommendations
    def _recommendation_lines(self) -> list[str]:
        lines = [rule(_WIDTH, "-"), "Recommendations"]
        for i, rec in enumerate(self["recommendations"], start=1):
            lines.append(rule(_WIDTH, "-"))
            lines.append(f"{i:>2}. {rec['topic']}")
            lines += _wrap(rec["finding"], "    ")
            lines += textwrap.wrap(
                rec["suggestion"],
                width=_WIDTH,
                initial_indent="    -> ",
                subsequent_indent="       ",
            )
            lines.append(f"    functions: {', '.join(rec['functions'])}")
            lines += _wrap(f"caveat: {rec['caveat']}", "    ")
        return lines

    # ------------------------------------------------------------------ #
    # plot
    # ------------------------------------------------------------------ #
    def plot_diagnostics(self, path=None):
        """A 2x2 diagnostic figure. Returns the Figure.

        Panels: the input series, the ACF and PACF against the white-noise
        band, and a histogram of the analysis object with a normal overlay.
        For a multivariate report the panels overlay every column, and the
        histogram pools the standardized columns against a standard normal.
        """
        plt = pyplot()
        if getattr(self, "_data", None) is None:
            raise ValueError(
                "this CheckSeriesResults was not built by "
                "CheckSeriesResults.run(), so the original data is "
                "unavailable; call tsecon.results.check_series(data, ...) "
                "instead"
            )
        fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.8))
        if self["kind"] == "univariate":
            self._plot_univariate(plt, axes)
        else:
            self._plot_multivariate(plt, axes)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=150)
        return fig

    def _plot_correlogram(self, ax, values, band, title, color):
        lags = np.arange(1, len(values) + 1)
        ax.bar(lags, np.asarray(values, dtype=float), width=0.55, color=color, lw=0)
        ax.axhline(band, color=REF, lw=0.8, ls="--")
        ax.axhline(-band, color=REF, lw=0.8, ls="--")
        apply_style(ax, zero_line=True)
        ax.set_xlim(0.3, len(values) + 0.7)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("lag", fontsize=8, color=MUTED)

    @staticmethod
    def _plot_histogram(ax, values, mean, sd, title):
        ax.hist(values, bins=40, color=SERIES["blue"], alpha=0.75, lw=0,
                density=True)
        if sd > 0:
            grid = np.linspace(float(np.min(values)), float(np.max(values)), 200)
            density = np.exp(-0.5 * ((grid - mean) / sd) ** 2) / (
                sd * np.sqrt(2.0 * np.pi)
            )
            ax.plot(grid, density, color=SERIES["red"], lw=1.1)
        apply_style(ax)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("value", fontsize=8, color=MUTED)

    def _plot_univariate(self, plt, axes):
        (ax_series, ax_acf), (ax_pacf, ax_hist) = axes
        y = self._data
        scale = self["analysis_scale"]["scale"]
        if scale == "first_difference":
            z = np.diff(y)
        elif scale == "detrended_level":
            # the same OLS detrend the battery applied (Conflict quadrant)
            x = np.column_stack([np.ones(y.size), np.arange(y.size, dtype=float)])
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            z = y - x @ beta
        else:
            z = y

        ax_series.plot(np.arange(y.size), y, color=SERIES["blue"], lw=1.0)
        apply_style(ax_series)
        ax_series.set_title("series — level", fontsize=9, loc="left")
        ax_series.set_xlabel("observation", fontsize=8, color=MUTED)

        sc = self["serial_correlation"]
        band = float(sc["conf_band"])
        self._plot_correlogram(
            ax_acf, sc["acf"], band, f"ACF — {sc['computed_on']}", SERIES["violet"]
        )
        self._plot_correlogram(
            ax_pacf, sc["pacf"], band, f"PACF — {sc['computed_on']}",
            SERIES["aqua"],
        )
        self._plot_histogram(
            ax_hist, z, float(z.mean()), float(z.std(ddof=1)),
            f"histogram — {scale}",
        )

    def _plot_multivariate(self, plt, axes):
        (ax_series, ax_acf), (ax_pacf, ax_hist) = axes
        core = _compiled()
        data = self._data
        k = data.shape[1]
        palette = list(SERIES.values())
        scale = self["var_lag_selection"].get("scale", "level")
        z = np.diff(data, axis=0) if scale == "first_difference" else data
        m = z.shape[0]

        t = np.arange(data.shape[0])
        for j in range(k):
            ax_series.plot(t, data[:, j], color=palette[j % len(palette)],
                           lw=1.0, label=f"y{j + 1}")
        apply_style(ax_series)
        ax_series.legend(fontsize=7, frameon=False, ncol=min(k, 4))
        ax_series.set_title("series — level", fontsize=9, loc="left")
        ax_series.set_xlabel("observation", fontsize=8, color=MUTED)

        band = 1.96 / np.sqrt(m)
        n_acf = max(1, min(20, m - 1))
        n_pacf = max(1, min(20, m // 2))
        for j in range(k):
            column = np.ascontiguousarray(z[:, j])
            acf = np.asarray(core.acf(column, nlags=n_acf)["acf"])[1:]
            pacf = np.asarray(core.pacf(column, nlags=n_pacf))[1:]
            color = palette[j % len(palette)]
            ax_acf.plot(np.arange(1, acf.size + 1), acf, color=color,
                        lw=1.0, marker="o", ms=2.4)
            ax_pacf.plot(np.arange(1, pacf.size + 1), pacf, color=color,
                         lw=1.0, marker="o", ms=2.4)
        for ax, name in ((ax_acf, "ACF"), (ax_pacf, "PACF")):
            ax.axhline(band, color=REF, lw=0.8, ls="--")
            ax.axhline(-band, color=REF, lw=0.8, ls="--")
            apply_style(ax, zero_line=True)
            ax.set_title(f"{name} per series — {scale}", fontsize=9, loc="left")
            ax.set_xlabel("lag", fontsize=8, color=MUTED)

        standardized = []
        for j in range(k):
            column = z[:, j]
            sd = float(column.std(ddof=1))
            if sd > 0:
                standardized.append((column - column.mean()) / sd)
        pooled = np.concatenate(standardized) if standardized else z.ravel()
        self._plot_histogram(
            ax_hist, pooled, 0.0, 1.0,
            f"pooled standardized histogram — {scale}",
        )


# --------------------------------------------------------------------------- #
# module-level helper
# --------------------------------------------------------------------------- #
def check_series(data, **kwargs) -> CheckSeriesResults:
    """Run the diagnostic battery and return a :class:`CheckSeriesResults`
    (a dict, plus methods). Keyword arguments mirror
    :func:`tsecon.check_series`."""
    return CheckSeriesResults.run(data, **kwargs)
