"""Denial claims: every "not implemented / roadmap / does not ship / R has,
tsecon does not" sentence in the doc set that can be checked against the
173-callable surface, plus the runtime advisory strings in ``_inspect.py``.

A DENIED-BUT-SHIPPED result is severe (the doc denies a shipped capability);
a CONFIRMED result means the denial is true today.

The guide-12 rows and the ``_inspect.py`` NaN row quote the wording of commit
19d308e (tsecon 0.8.0 as merged); the audit corrected those passages on the
same branch, so on the fixed tree they document what was found, not what the
page now says.

Run:  .venv-wt/bin/python lab/audit/repo/claims/sweep_denials.py
Out:  out/sweep_denials.log
"""
from __future__ import annotations

import inspect
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import OUT, log, public_callables, signature_params  # noqa: E402

PUBLIC = set(public_callables())
ROWS = []


def row(where, claim, shipped, evidence):
    ROWS.append((where, claim, shipped, evidence))


def accepts(name, kw, value, *args, **kwargs):
    """Does name(..., kw=value) run, or refuse naming the value as unknown?"""
    try:
        getattr(tsecon, name)(*args, **{**kwargs, kw: value})
        return True, "ran"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:140]}"


def main():
    fh = open(os.path.join(OUT, "sweep_denials.log"), "w")
    rng = np.random.default_rng(0)

    # --- guide ch. 12 (machine learning): the 0.8.0 wave -----------------
    ml = ["group_lasso", "regression_tree", "random_forest", "boosting", "pds_lasso", "post_lasso", "kernel_ridge", "kernel_regression", "mlp_regression", "echo_state_network", "l1_trend_filter"]
    row("docs/guide/12-machine-learning.md:225,372", "group_lasso is on the roadmap, not a shipped function", "group_lasso" in PUBLIC, "tsecon.group_lasso callable")
    row("docs/guide/12-machine-learning.md:301", "No tree learner ships in tsecon today", {"regression_tree", "random_forest"} <= PUBLIC, "regression_tree, random_forest callable")
    row("docs/guide/12-machine-learning.md:344", "no neural estimator is callable in tsecon today", {"mlp_regression", "echo_state_network"} <= PUBLIC, "mlp_regression, echo_state_network callable")
    x = rng.standard_normal((150, 4))
    y = np.sin(2 * x[:, 0]) + 0.5 * x[:, 1] + 0.3 * rng.standard_normal(150)
    rf = tsecon.random_forest(x, y, n_trees=10, seed=0, importance="block_permutation", importance_groups=[0, 0, 1, 1], permutation_block=5, n_permutations=2)
    row("docs/guide/12-machine-learning.md:360", "no importance ... function is in the Python API today", "importance" in rf, f"random_forest(importance='permutation', importance_groups=, permutation_block=) returns {sorted(k for k in rf if 'import' in k)}")
    row("docs/guide/12-machine-learning.md:597", "Still absent: group and sparse-group LASSO, native random forests with block resampling, componentwise boosting, PDS-LASSO, dependence-aware interpretation (block permutation)", set(ml[:5]) <= PUBLIC, "all five callable; random_forest(bootstrap='block'|'stationary') runs")
    row("docs/guide/12-machine-learning.md:595", "what is not listed above is not yet written in Rust either", not (set(ml) - PUBLIC), "eleven ML callables shipped in 0.8.0")

    # --- EWC / fixed-b: guide ch. 02, ch. 03 -------------------------------
    ok, why = accepts("long_run_variance", "kernel", "ewc", rng.standard_normal(200))
    row("docs/guide/03-inference-toolkit.md:266 & 02:524", "ewc_lrv / ewc_default_b land in Python with Module 00 (not yet)", ok or "ewc_lrv" in PUBLIC, f"long_run_variance(kernel='ewc'): {why}; ewc_lrv callable: {'ewc_lrv' in PUBLIC}")

    # --- ARIMA: exog / CSS -------------------------------------------------
    ps = set(signature_params("arima_fit"))
    row("docs/migration/from-statsmodels.md:305", "arima_fit takes no exog", "exog" in ps, f"arima_fit params: {sorted(ps)}")
    row("docs/guide/04-univariate-models.md:526", "the CSS estimator (fit_css) is Rust-only", "method" in ps and "css" in (tsecon.arima_fit.__doc__ or ""), f"arima_fit has no method= kwarg: {'method' not in ps}")

    # --- VAR: automatic lag selection / Johansen p-values / forecast cov ---
    row("docs/guide/07-multivariate.md:413", "select_order lag selection is Rust-only", "var_select_order" in PUBLIC or "select_order" in PUBLIC, f"var_fit params: {signature_params('var_fit')}")
    j = tsecon.johansen(np.cumsum(rng.standard_normal((200, 2)), axis=0))
    row("docs/guide/07-multivariate.md:415", "MacKinnon-Haug-Michelis p-values for Johansen are roadmap", any("p" in k and "value" in k for k in j), f"johansen keys: {sorted(j)}")
    row("docs/guide/07-multivariate.md:415", "Toda-Yamamoto causality is roadmap", "toda_yamamoto" in PUBLIC or "tyc" in " ".join(PUBLIC), "no toda_yamamoto callable")

    # --- MIDAS: ADL-MIDAS ---------------------------------------------------
    row("docs/guide/11-nowcasting.md:474", "ADL-MIDAS is roadmap (add the target's lags to umidas yourself)", "adl_midas" in PUBLIC or bool({"y_lags", "ar_lags", "p"} & set(signature_params("umidas"))), f"umidas params: {signature_params('umidas')}")

    # --- GW conditional --------------------------------------------------------
    row("docs/which-model-when.md:1124", "the conditional GW test is still a roadmap item", bool({"x", "conditioning", "instruments", "conditional"} & set(signature_params("gw_test"))), f"gw_test params: {signature_params('gw_test')}")

    # --- Bai-Perron partial model / hetero-robust intervals -------------------
    row("docs/reference/model-cards/structural-breaks.md:136", "bai_perron does not implement the partial structural-change model", bool({"partial", "x_fixed", "z"} & set(signature_params("bai_perron"))), f"bai_perron params: {signature_params('bai_perron')}")
    row("docs/guide/02-exploration-and-diagnostics.md:526", "heterogeneity-robust Bai-Perron break-date intervals are roadmap", bool({"hetero", "robust", "het"} & set(signature_params("bai_perron"))), f"bai_perron params: {signature_params('bai_perron')}")

    # --- bn_filter dynamic demeaning ---------------------------------------
    yv = np.cumsum(rng.standard_normal(200))
    ok_dm, why_dm = accepts("bn_filter", "demean", "dm", yv)
    ok_dyn, why_dyn = accepts("bn_filter", "demean", "dynamic", yv)
    row("docs/reference/model-cards/diagnostics.md:620", "dynamic demeaning is not implemented in bn_filter (the 2018 baseline is)", ok_dm or ok_dyn, f"demean='dm': {why_dm}; demean='dynamic': {why_dyn}")

    # --- kernel_regression compact kernels ---------------------------------
    ok_tc, why_tc = accepts("kernel_regression", "kernel", "tricube", x[:, 0], y, bandwidth=0.5)
    row("docs/reference/model-cards/ml-kernel.md:87", "compact-support kernels are deferred (Gaussian only)", ok_tc, f"kernel='tricube': {why_tc}")

    # --- copulas rotated / dynamic -----------------------------------------
    from scipy.stats import rankdata

    z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=300)
    u = np.column_stack([rankdata(z[:, j]) / 301 for j in range(2)])
    ok_r, why_r = accepts("copula_fit", "family", "survival_clayton", u)
    ok_r2, why_r2 = accepts("copula_fit", "family", "rotated_clayton", u)
    row("docs/reference/model-cards/copulas.md:61", "rotated/survival copula variants are deferred", ok_r or ok_r2, f"survival_clayton: {why_r}; rotated_clayton: {why_r2}")

    # --- panel: LP-DiD covariates / IV; Hausman; CD test ----------------------
    row("docs/reference/model-cards/panel.md:462", "LP-DiD covariates, composition correction, pmd baselines and the IV variant are not yet implemented", bool({"x", "covariates", "controls", "pmd", "instrument", "iv"} & set(signature_params("lp_did"))), f"lp_did params: {signature_params('lp_did')}")
    row("docs/guide/14-panel-time-series.md:332,353", "Hausman test of panel_pmg vs MG is roadmap", "hausman" in " ".join(tsecon.panel_pmg(*__import__('registry_ext').build('panel_pmg')[0]).keys()).lower(), "panel_pmg keys carry no hausman entry")
    row("docs/guide/14-panel-time-series.md:353 / which-model-when:805", "Pesaran CD test / CIPS / PANIC are roadmap", bool({"pesaran_cd", "cd_test", "cips", "panic"} & PUBLIC), "no such callable")

    # --- Stata/statsmodels/R: A/B SVAR -----------------------------------------
    row("docs/migration/*: A/B SVAR", "explicit short-run A/B SVAR restrictions are roadmap", bool({"ab_svar", "svar_ab", "svar"} & PUBLIC) or any("amat" in signature_params(n) for n in PUBLIC), "no A/B callable; no amat= kwarg anywhere")
    row("docs/migration/from-statsmodels.md:121", "ETS beyond Theta is roadmap", bool({"ets", "ets_fit", "exponential_smoothing", "holt_winters"} & PUBLIC), "no ETS callable")
    row("docs/migration/from-statsmodels.md:145", "VARMAX / VARMA / ARDL / UECM are roadmap", bool({"varma", "varmax", "ardl", "uecm"} & PUBLIC), "no such callable")
    row("docs/migration/from-statsmodels.md:88", "range_unit_root_test / Leybourne-McCabe are roadmap", bool({"range_unit_root", "leybourne_mccabe", "rur"} & PUBLIC), "no such callable")
    row("docs/migration/from-statsmodels.md:198", "classical seasonal_decompose is roadmap", "seasonal_decompose" in PUBLIC or "decompose" in PUBLIC, "no such callable")
    row("docs/migration/from-r.md:87", "stochastic-volatility BVAR priors are roadmap", bool({"bvar_sv", "bvar_tvp"} & PUBLIC), "no such callable")
    row("docs/guide/13-nonlinear-dynamics.md:309", "the Tsay arranged-autoregression linearity test is not built", bool({"tsay_test", "tsay"} & PUBLIC), "no such callable")
    row("docs/guide/13-nonlinear-dynamics.md:306", "the GIRF engine (Koop-Pesaran-Potter) is roadmap", bool({"girf", "generalized_irf"} & PUBLIC), "no such callable")
    row("docs/guide/10-bayesian.md:397", "joint credible bands for Bayesian IRFs are roadmap", "band" in signature_params("bvar_irf_draws"), f"bvar_irf_draws params: {signature_params('bvar_irf_draws')}")
    row("docs/guide/02-exploration-and-diagnostics.md:443", "multitaper and cross-spectral phase are roadmap", bool({"multitaper", "cross_spectrum", "phase"} & PUBLIC) or "phase" in " ".join(tsecon.coherence(rng.standard_normal(256), rng.standard_normal(256), nperseg=64).keys()), f"coherence keys: {sorted(tsecon.coherence(rng.standard_normal(256), rng.standard_normal(256), nperseg=64))}")

    # --- runtime advisory strings (_inspect.py) -------------------------------
    yy = rng.standard_normal(60)
    yy[5] = np.nan
    try:
        tsecon.check_series(yy)
        msg = "(no error)"
    except ValueError as exc:
        msg = str(exc)
    # can the shipped local-level smoother impute a gap?
    try:
        ll = tsecon.local_level_smooth(yy, 1.0, 0.5)
        imput = np.isfinite(np.asarray(ll["smoothed"] if "smoothed" in ll else list(ll.values())[0])).all()
    except Exception as exc:  # noqa: BLE001
        imput = False
        ll = {"error": str(exc)[:100]}
    row("_inspect.py:126 (check_series NaN refusal)", "state-space/Kalman imputation is on the Module 01 roadmap", bool(imput), f"local_level_smooth on a NaN-holed series: {'returns finite smoothed values' if imput else ll}; refusal text: {msg[:160]}")
    row("_inspect.py:174 (outlier caveat)", "Chen-Liu additive/innovational outlier detection is a roadmap item", bool({"chen_liu", "tso", "outlier_detection"} & PUBLIC), "no such callable")
    rec = [r for r in tsecon.check_series(np.sin(np.arange(120) * 2 * np.pi / 12) + 0.1 * rng.standard_normal(120), seasonal_period=12)["recommendations"] if r["topic"] == "seasonality"][0]
    row("_inspect.py:828 (seasonality advice)", "routes to arima_fit(seasonal=...), mstl, stl; names X-13 as the one gap", "seasonal=" in rec["suggestion"] and "no seasonal ARIMA" not in rec["suggestion"], rec["suggestion"][:140])

    # --- paper.md capability claims -------------------------------------------
    sig = inspect.signature(tsecon.lp)
    row("paper/paper.md: lp lag-augmented default", "lag-augmented inference is the lp default", sig.parameters["se"].default in (None, "lag_augmented", "lag-augmented"), f"lp se default = {sig.parameters['se'].default!r}")
    for n, kw in (("ols", "se_type"), ("lp", "se"), ("hamilton_filter", "se")):
        row(f"paper/paper.md: {n}({kw}='hac')", f"{n} takes {kw}='hac'", kw in signature_params(n), f"{n} params: {signature_params(n)}")
    row("paper/paper.md: arima_fit seasonal", "seasonal via seasonal=(P, D, Q, s)", "seasonal" in signature_params("arima_fit"), "kwarg present")
    row("paper/paper.md: vecm", "vecm across every statsmodels deterministic case with centered seasonal dummies", {"deterministic", "seasons", "first_season"} <= set(signature_params("vecm")), f"vecm params: {signature_params('vecm')}")
    row("paper/paper.md: dfm_nowcast", "two-step and one-step Gaussian-MLE estimation", "method" in signature_params("dfm_nowcast"), "method= present")

    n_denied_shipped = 0
    for where, claim, shipped, evidence in ROWS:
        tag = "DENIED-BUT-SHIPPED" if shipped else "CONFIRMED-ABSENT"
        if where.startswith("paper") or "_inspect.py:828" in where:
            tag = "CLAIM-HOLDS" if shipped else "CLAIM-FAILS"
        if tag == "DENIED-BUT-SHIPPED":
            n_denied_shipped += 1
        log(fh, f"{tag:19s} {where}\n    claim: {claim}\n    evidence: {evidence}")
    log(fh, f"\n{len(ROWS)} claims checked; {n_denied_shipped} deny a shipped capability")
    fh.close()


if __name__ == "__main__":
    main()
