"""Parameter-name and return-key clusters from surface.json (items 2, 3).

Every parameter name and every returned key (top-level and one level of
nesting) is inventoried with the functions that use it, then assigned to a
concept cluster by the CONCEPTS / KEY_CONCEPTS tables below. Unassigned
names are printed so the tables can be completed by hand.

Run:  .venv-wt/bin/python lab/audit/repo/api/probe_clusters.py
Out:  lab/audit/repo/api/out/clusters.json, clusters.md (markdown tables)
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SURFACE = json.load(open(os.path.join(HERE, "surface.json")))

# concept -> list of (regex, note)
CONCEPTS = {
    "randomness": [r"^(seed|rng|random_state|band_seed|rf_seed)$"],
    "replication count": [r"^(n_boot|n_bootstrap|reps|n_rep|n_reps|nboot|n_draws|draws|n_sim|n_permutations|n_seeds|n_trees|n_grid\w*|n_lambdas|n_gamma|n_c|n_eval|n_iter_boot|burn|thin|n_paths|n_members|band_n_sim|n_weight_draws|n_chains|max_tries)$"],
    "confidence level": [r"^(alpha|level|coverage|conf_level|conf_alpha|ci|ci_level|band_alpha|significance)$"],
    "penalty strength": [r"^(alpha|lam|lamb|lambda_|lambda_1|lambda_2|l1_ratio|ridge_alpha|penalty|weight_decay|mu|lambda0|lambda1|lambda3|lambda1_init|lambda1_lo|lambda1_hi)$"],
    "lag / order": [r"^(lags|p|q|order|maxlags|maxlag|nlags|ar_order|ma_order|lag|n_lags|max_lag|max_lags|max_p|max_q|max_P|max_Q|max_d|max_D|max_order|hac_lags|hac_maxlags|lrv_lags|bandwidth_lags|var_lags|lag_order|n_lag_controls|controls_lags|d|D|delays|delay|k_ar_diff|factor_order|ar|ma)$"],
    "horizon": [r"^(horizon|horizons|h|steps|n_ahead|forecast_steps|forecast_horizon|h1|h_max|max_horizon|n_steps|post_window|pre_window)$"],
    "trend / deterministic": [r"^(trend|deterministic|constant|det|regression|include_constant|intercept|const|seasons|first_season|season|drift|include_trend|include_intercept|fit_intercept)$"],
    "standard-error type": [r"^(se_type|cov_type|robust|hac|se|kernel|hac_kernel|bandwidth|hac_bandwidth|use_correction|hc|vce|cov|ses|robust_se|bw)$"],
    "tolerance / iterations": [r"^(tol|max_iter|maxiter|max_iters|max_epochs|x_tol|f_tol|ftol|xtol|gtol|n_iter|iterations|max_newton|patience|rtol|atol|tolerance|max_steps|inner_iter|outer_iter)$"],
    "method selector (string)": [r"^(method|kind|mode|model|test|test_type|variant|dist|scheme|ic|family|stop|forecaster|base|solver|activation|identification|bootstrap|importance|univariate_dist|vol|mean|window|loss|policy|band|band_scope|autolag|hyperprior|weight|split|calib|regression|criterion|se_type|sign_normalization|penalty|bandwidth_method|max_features|group_weights|kernel|trend|deterministic|cumulative)$"],
    "data (first positional)": [r"^(y|x|data|returns|resid|u|r|xs|ys|curves|chains|panel|outcome|regressors|treatment|proxy|yields|hf_lags|rv|e1|e2|loss1|loss2|actual|forecast|insample|z|d|scores|entities|instrument|shock|impulse|target|maturities|high|low|residuals|returns_or_hits|state_indicator|conditions|obj|forecasts|e_small|e_large|yhat_small|yhat_large|old_vintage|new_vintage)$"],
    "prediction input": [r"^(x_test|x_new|X_test|newdata|exog_future|x_future|test|x_eval)$"],
    "train/test split": [r"^(train|n_train|train_size|test_size|window|n_eval|validation_fraction|min_train|scheme|expanding)$"],
}

KEY_CONCEPTS = {
    "standard errors": [r"^(se|std_err|stderr|bse|bse_\w+|se_\w+|\w+_se|ses|standard_errors?|coefs_se|orth_irfs_se|se_valid)$"],
    "p-values": [r"^(p_value|pvalue|p|pval|pvals|p_values|pvalues|\w+_pvalue|\w+_p_value|p_\w+|\w+_p)$"],
    "coefficients": [r"^(coef|coefs|params|params_\w+|param_names|beta|betas|coefficients|coefficient|b|theta|params_named|coef_\w+|beta_\w+|\w+_coef|\w+_coefs|estimate|estimates|weights|readout|dual_coef|phi|a|ar|ma|ar_params|ma_params|gamma|alpha|omega|delta|B|coint_coefs|posterior_mean_coefs|coef_mean|loadings|factor_loadings)$"],
    "fitted values": [r"^(fitted|fitted_values|yhat|y_hat|fit|fittedvalues|in_sample|prediction|predicted|forecast|forecasts|point|mean|nowcast|trend|cycle|smoothed\w*|filtered\w*)$"],
    "residuals": [r"^(resid|residuals|residual|u|eps|errors|innovations|e|resids|std_resid|standardized_residuals)$"],
    "log-likelihood": [r"^(loglik|llf|log_likelihood|loglikelihood|ll|logl|loglike|log_lik)$"],
    "information criteria": [r"^(aic|bic|hqic|aicc|ic|criterion|aic_path|bic_path|hq|ics|information_criteria|lag_selection)$"],
    "convergence": [r"^(converged|success|n_iter|iterations|n_iterations|niter|status|n_evaluations|n_criterion_evaluations|n_func_evals|nit|message|best_epoch|boundary|boundary_note|n_evals|fevals|at_bound|bandwidth_at_boundary|budget_exhausted|cov_ok|n_accepted)$"],
    "confidence intervals": [r"^(conf_int|ci|lower|upper|bands|band|ci_lower|ci_upper|ci_lower_\d+|ci_upper_\d+|lo|hi|interval|intervals|quantiles|q_lower|q_upper|conf_int_\w+|\w+_lower|\w+_upper|\w+_ci|\w+_bands|set_min|set_max|lower_quantiles|upper_quantiles|robust_ci_lower|robust_ci_upper|lower_efron|upper_efron|conf_alpha|alpha|ci_scale)$"],
    "R-squared": [r"^(rsquared|r_squared|r2|rsq|\w+_rsquared|adj_rsquared)$"],
    "sample-size echo": [r"^(nobs|n_obs|n|nobs_per_h|n_units|n_train|n_validation|n_used|n_origins|n_breaks|n_regressors|n_vars|n_endog|n_factors|n_proxy|n_calib|n_eval|n_washout|n_maxima|n_models|n_knots|neqs|n_stacked|adf_nobs|n_controls_selected|n_parameters)$"],
    "lag echo": [r"^(lags|used_lag|maxlags|hac_lags|hac_lags_resolved|order|delay|k_ar_diff|lag|nlags|max_d|seasonal_order|factor_order)$"],
    "variance / covariance": [r"^(sigma|sigma2|sigma_u|sigma2_\w+|variance|variances|cov|covariance|param_cov|sigma_\w+|omega_bar|s_bar|covs|qbar|correlation|correlation_\w+|covariance_forecast|variance_forecast|scale|factor_cov)$"],
    "test statistic": [r"^(stat|statistic|t_stat|tstat|t|tvalue|tvalues|t_values|z|zstat|f|f_stat|fstat|wald|lm|lr|chi2|q|q_stat|\w+_stat|\w+_statistic|statistics|test_stat)$"],
    "critical values": [r"^(crit|critical_values|crit_values|cv|critvals|critical|cvs|\w+_crit)$"],
}


def inventory():
    params = defaultdict(list)
    keys = defaultdict(list)
    nested = defaultdict(list)
    for name, rec in sorted(SURFACE.items()):
        for p in rec.get("params") or []:
            params[p["name"]].append(name)
        for k in (rec.get("keys") or {}):
            keys[k].append(name)
        for k, sub in (rec.get("nested") or {}).items():
            for kk in sub:
                nested[kk].append(f"{name}.{k}")
    return params, keys, nested


def assign(inv, concepts):
    table = {c: defaultdict(list) for c in concepts}
    unassigned = {}
    for name, fns in inv.items():
        hit = False
        for c, pats in concepts.items():
            if any(re.match(p, name) for p in pats):
                table[c][name] = fns
                hit = True
        if not hit:
            unassigned[name] = fns
    return table, unassigned


def md_table(title, table, per_fn_default=None):
    lines = [f"### {title}", ""]
    lines.append("| concept | spelling | n functions | functions |")
    lines.append("|---|---|---|---|")
    for c, names in table.items():
        rows = sorted(names.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        for spelling, fns in rows:
            shown = ", ".join(f"`{f}`" for f in fns[:12]) + (f" … (+{len(fns) - 12})" if len(fns) > 12 else "")
            lines.append(f"| {c} | `{spelling}` | {len(fns)} | {shown} |")
    lines.append("")
    return "\n".join(lines)


def main():
    params, keys, nested = inventory()
    ptab, punk = assign(params, CONCEPTS)
    ktab, kunk = assign(keys, KEY_CONCEPTS)
    ntab, nunk = assign(nested, KEY_CONCEPTS)
    # defaults per parameter spelling (for the randomness / level clusters)
    defaults = defaultdict(lambda: defaultdict(list))
    for name, rec in sorted(SURFACE.items()):
        for p in rec.get("params") or []:
            defaults[p["name"]][repr(p.get("default")) if p.get("has_default") else "<required>"].append(name)
    out = {
        "param_clusters": {c: dict(v) for c, v in ptab.items()},
        "param_unassigned": punk,
        "key_clusters": {c: dict(v) for c, v in ktab.items()},
        "key_unassigned": kunk,
        "nested_key_clusters": {c: dict(v) for c, v in ntab.items()},
        "nested_key_unassigned": nunk,
        "param_defaults": {k: dict(v) for k, v in defaults.items()},
        "param_inventory": dict(params),
        "key_inventory": dict(keys),
    }
    json.dump(out, open(os.path.join(OUT, "clusters.json"), "w"), indent=1, sort_keys=True)
    md = [md_table("Parameter-name clusters", ptab), md_table("Return-key clusters (top level)", ktab), md_table("Return-key clusters (nested, `function.key`)", ntab)]
    open(os.path.join(OUT, "clusters.md"), "w").write("\n".join(md))
    print(f"params: {len(params)} distinct names over {sum(len(v) for v in params.values())} slots; unassigned {len(punk)}")
    print("unassigned params:", sorted(punk, key=lambda k: -len(punk[k]))[:200])
    print(f"keys: {len(keys)} distinct top-level keys; unassigned {len(kunk)}")
    print("unassigned keys:", sorted(kunk, key=lambda k: -len(kunk[k]))[:300])
    print(f"nested keys: {len(nested)} distinct; unassigned {len(nunk)}")


if __name__ == "__main__":
    main()
