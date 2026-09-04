"""Second-pass probe for the names sweep: every phantom candidate that is a
returned-key claim under a non-default mode (a band, a method, a pooled
variant, a 2-D input) is re-checked by calling the function in that mode.

Run:  .venv-wt/bin/python lab/audit/repo/claims/probe_phantoms.py
Out:  out/probe_phantoms.log
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import OUT, log  # noqa: E402
from registry_ext import build  # noqa: E402

rng = np.random.default_rng(0)
PROBES = []


def probe(where, claim_keys, fn, *a, nested=None, **k):
    PROBES.append((where, claim_keys, fn, a, k, nested))


def flat_keys(res, depth=2):
    out = set()
    if isinstance(res, dict):
        for kk, v in res.items():
            out.add(str(kk))
            if depth > 1:
                out |= flat_keys(v, depth - 1)
    elif isinstance(res, (list, tuple)) and res and isinstance(res[0], dict):
        out |= flat_keys(res[0], depth)
    return out


def main():
    fh = open(os.path.join(OUT, "probe_phantoms.log"), "w")
    y = np.cumsum(rng.standard_normal(200)) + 30
    v3 = build("var_fit")[0][0]
    probe("guide/05:514, forecasting card:493 conformal ACI", ["alpha_trajectory"], tsecon.conformal_forecast, y, horizon=2, method="aci", n_eval=20)
    probe("guide/07:231, var-svar card:129 var_forecast sup-t", ["sim_lower", "sim_upper", "n_cells", "n_cells_used"], tsecon.var_forecast, v3, lags=1, steps=4, band="sup-t")
    probe("guide/09:188, LP card:112 lp sup-t", ["cov_se_max_rel_diff", "n_cells", "n_cells_used"], tsecon.lp, *build("lp")[0], horizons=6, band="sup-t")
    a, k = build("lp_state")
    probe("LP card:320 lp_state sup-t", ["lower_state1", "upper_state1", "lower_state0", "upper_state0", "critical_value_state1", "critical_value_state0"], tsecon.lp_state, *a, horizons=6, band="sup-t")
    probe("LP card:136 lp_iv/lp_multiplier n_cells", ["n_cells", "n_cells_used"], tsecon.lp_iv, *build("lp_iv")[0], horizons=6, band="sup-t")
    probe("guide/11:396 dfm_news contribution", ["contribution"], tsecon.dfm_news, *build("dfm_news")[0])
    probe("guide/15:142 dynamic_ns ar1_phi", ["ar1_phi"], tsecon.dynamic_ns, *build("dynamic_ns")[0])
    probe("cookbook/growth-at-risk:65 crossing", ["crossing"], tsecon.growth_at_risk, *build("growth_at_risk")[0], horizon=2, taus=[0.1, 0.5, 0.9])
    probe("arima card:522 auto_arima budget_exhausted", ["budget_exhausted"], tsecon.auto_arima, *build("auto_arima")[0], max_p=2, max_q=2)
    probe("bayesian card:319 bvar_ssvs inclusion_prob_cov", ["inclusion_prob_cov"], tsecon.bvar_ssvs, v3, n_draws=100, burn=20, seed=0, horizon=4)
    panel2 = np.column_stack([np.cumsum(rng.standard_normal(120)), np.cumsum(rng.standard_normal(120))])
    probe("check-series card:44 2-D report keys", ["per_series", "integration_summary", "cointegration", "var_lag_selection", "stability"], tsecon.check_series, panel2)
    probe("dsge card:106 n_unstable", ["n_unstable"], tsecon.dsge_solve, *build("dsge_solve")[0])
    probe("forecasting card:27/131 var_backtest LR_uc/LR_ind/DQ/dq_var_dropped", ["LR_uc", "LR_ind", "DQ", "dq_var_dropped", "kupiec", "christoffersen", "dq"], tsecon.var_backtest, *build("var_backtest")[0], alpha=0.05)
    probe("ml-convex card:116 l1_trend_filter n_iter/cycle", ["n_iter", "cycle"], tsecon.l1_trend_filter, y, 5.0)
    probe("panel-unit-root card:107 per-test keys", ["delta_hat", "t_delta", "s_n", "t_bar_periods", "maddala_wu", "choi_z", "choi_z_pvalue"], tsecon.panel_unit_root, *build("panel_unit_root")[0], lags=1)
    a, k = build("lp_did")
    probe("panel card:127 lp_did pooled keys", ["pooled_post_att", "pooled_post_se", "pooled_post_nobs", "pooled_post_n_switchers", "n_cells", "n_cells_used"], tsecon.lp_did, *a, pre_window=2, post_window=3, pooled=True)
    probe("realized-vol card:109 realized_measures jump", ["jump"], tsecon.realized_measures, *build("realized_measures")[0])
    probe("spec-tests card:254 heteroskedasticity_test df_num/df_den", ["df_num", "df_den"], tsecon.heteroskedasticity_test, *build("heteroskedasticity_test")[0])
    probe("spec-tests card:254 reset_test df_num/df_den", ["df_num", "df_den"], tsecon.reset_test, *build("reset_test")[0])
    probe("spec-tests card:337 cusum_test recursive_residuals", ["recursive_residuals"], tsecon.cusum_test, *build("cusum_test")[0])
    probe("ident card:418 proxy_first_stage weak_folklore", ["weak_folklore"], tsecon.proxy_first_stage, *build("proxy_first_stage")[0])
    probe("ident card:669/803 proxy_ar_sets excluded_lower/upper, excludes_zero, bounded, ar_bounded_all", ["excluded_lower", "excluded_upper", "excludes_zero", "bounded", "ar_bounded_all"], tsecon.proxy_ar_sets, *build("proxy_ar_sets")[0], horizon=4)
    probe("ident card:1367 hd_quantiles/hd_set_min/hd_set_max (sign_restricted_svar hd)", ["hd_quantiles", "hd_set_min", "hd_set_max"], tsecon.sign_restricted_svar, *build("sign_restricted_svar")[0], horizon=4, n_draws=40, seed=0)
    probe("ident card:1714 rate (narrative_svar)", ["rate", "narrative_rate", "acceptance_rate"], tsecon.narrative_svar, *build("narrative_svar")[0], horizon=4, n_draws=40, seed=0)
    probe("volatility card:384 dcc_garch adcc nbar", ["nbar"], tsecon.dcc_garch, build("dcc_garch")[0][0], variant="adcc")
    probe("volatility card:236 garch_fit(dist='t') nu", ["nu", "nu_hat"], tsecon.garch_fit, build("garch_fit")[0][0], dist="t")
    probe("cookbook/sign-restricted:58 & var-svar card:350 sign_restricted_svar probs kwarg", ["probs"], lambda *a, **k: {"probs": None} if "probs" in __import__("inspect").signature(tsecon.sign_restricted_svar).parameters else {}, 0)
    probe("LP card:390 smooth_lp lambda/lam kwarg", ["lam", "lambda"], lambda: {p: 1 for p in __import__("inspect").signature(tsecon.smooth_lp).parameters})
    probe("bayesian card:75 bvar_fit S0/v0 kwargs", ["S0", "v0", "s0", "v_0"], lambda: {p: 1 for p in __import__("inspect").signature(tsecon.bvar_fit).parameters})

    for where, keys, fn, a, k, nested in PROBES:
        try:
            res = fn(*a, **k)
            have = flat_keys(res)
            present = [x for x in keys if x in have]
            absent = [x for x in keys if x not in have]
            log(fh, f"{'OK     ' if not absent else 'ABSENT '} {where}\n    present={present} absent={absent}\n    keys={sorted(have)[:40]}")
        except Exception as exc:  # noqa: BLE001
            log(fh, f"ERROR   {where}: {type(exc).__name__}: {str(exc)[:200]}")
    fh.close()


if __name__ == "__main__":
    main()
