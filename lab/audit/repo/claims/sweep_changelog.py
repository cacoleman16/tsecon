"""CHANGELOG spot check: every 0.8.0 / 0.7.0 / 0.6.0 bullet that names a
function or a behaviour, verified against the installed wheel — the function
exists, the kwarg exists, the refusal fires, the returned key exists.

Each check is a (label, thunk) pair; a thunk returns None on PASS or a string
saying what was observed. Refute before recording: a FAIL here is a candidate,
not a finding, until read charitably against the bullet.

Run:  .venv-wt/bin/python lab/audit/repo/claims/sweep_changelog.py
Out:  out/sweep_changelog.log
"""
from __future__ import annotations

import inspect
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import OUT, log, signature_params  # noqa: E402
from registry_ext import build  # noqa: E402

CHECKS = []


def check(label):
    def deco(fn):
        CHECKS.append((label, fn))
        return fn

    return deco


def exists(*names):
    missing = [n for n in names if not callable(getattr(tsecon, n, None))]
    return f"missing callables {missing}" if missing else None


def has_kw(name, *kws):
    ps = set(signature_params(name))
    missing = [k for k in kws if k not in ps]
    return f"{name} lacks kwargs {missing} (has {sorted(ps)})" if missing else None


def has_keys(res, *keys):
    missing = [k for k in keys if k not in res]
    return f"missing keys {missing} (got {sorted(res)})" if missing else None


def raises(match, fn, *a, **k):
    try:
        fn(*a, **k)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if match is None or re.search(match, msg):
            return None
        return f"raised {type(exc).__name__} but message lacks /{match}/: {msg[:200]}"
    return f"did not raise (expected /{match}/)"


def call(name, **override):
    a, k = build(name, T=200, seed=0)
    k.update(override)
    return getattr(tsecon, name)(*a, **k)


rng = np.random.default_rng(0)


def _xy(T=120, p=6, s=0):
    r = np.random.default_rng(s)
    x = r.standard_normal((T, p))
    y = np.sin(2 * x[:, 0]) + (0.5 * x[:, 1] if p > 1 else 0.0) + 0.3 * r.standard_normal(T)
    return x, y


# ------------------------------------------------------------------ 0.8.0
@check("0.8.0 the eleven ML callables exist")
def _():
    return exists("kernel_ridge", "kernel_regression", "group_lasso", "post_lasso", "pds_lasso", "regression_tree", "random_forest", "l1_trend_filter", "boosting", "mlp_regression", "echo_state_network")


@check("0.8.0 kernel_ridge kwargs + exact keys")
def _():
    e = has_kw("kernel_ridge", "alpha", "kernel", "gamma", "degree", "coef0", "x_test", "rff_features", "seed")
    if e:
        return e
    x, y = _xy(80, 2)
    r = tsecon.kernel_ridge(x, y, alpha=1.0, kernel="rbf", x_test=x[:3])
    return has_keys(r, "dual_coef", "fitted", "predicted", "kernel", "gamma", "n_rff_features")


@check("0.8.0 kernel_ridge RFF mode returns coef, laplacian/polynomial/linear kernels accepted")
def _():
    x, y = _xy(80, 2)
    r = tsecon.kernel_ridge(x, y, alpha=1.0, kernel="rbf", rff_features=50, seed=1)
    e = has_keys(r, "coef")
    if e:
        return e
    for kern in ("laplacian", "polynomial", "linear"):
        tsecon.kernel_ridge(x, y, alpha=1.0, kernel=kern)
    return None


@check("0.8.0 kernel_ridge refusals: gamma with linear, rff with non-rbf, seed in exact mode, degree with non-polynomial")
def _():
    x, y = _xy(60, 2)
    out = []
    out.append(raises("gamma", tsecon.kernel_ridge, x, y, alpha=1.0, kernel="linear", gamma=0.5))
    out.append(raises("rff", tsecon.kernel_ridge, x, y, alpha=1.0, kernel="linear", rff_features=10))
    out.append(raises("seed", tsecon.kernel_ridge, x, y, alpha=1.0, kernel="rbf", seed=3))
    out.append(raises("degree", tsecon.kernel_ridge, x, y, alpha=1.0, kernel="rbf", degree=2.0))
    out = [o for o in out if o]
    return "; ".join(out) if out else None


@check("0.8.0 kernel_ridge alpha=0 with duplicate rows refuses naming alpha")
def _():
    x, y = _xy(40, 2)
    x[1] = x[0]
    return raises("alpha", tsecon.kernel_ridge, x, y, alpha=0.0, kernel="rbf")


@check("0.8.0 kernel_regression kinds, bandwidth methods, keys, 1-D x")
def _():
    e = has_kw("kernel_regression", "bandwidth", "kind", "kernel", "bandwidth_method", "block", "x_test")
    if e:
        return e
    x, y = _xy(80, 1)
    r = tsecon.kernel_regression(x[:, 0], y, bandwidth=0.5, x_test=x[:3, 0])
    e = has_keys(r, "fitted", "predicted", "bandwidth", "bandwidth_method", "block", "cv_criterion", "effective_df", "kind", "kernel", "bandwidth_at_boundary", "n_criterion_evaluations")
    if e:
        return e
    if r["kind"] != "local_linear":
        return f"default kind is {r['kind']!r}, changelog says local_linear"
    tsecon.kernel_regression(x[:, 0], y, kind="nadaraya_watson", bandwidth=0.5)
    r2 = tsecon.kernel_regression(x[:, 0], y, bandwidth_method="loo_cv")
    r3 = tsecon.kernel_regression(x[:, 0], y, bandwidth_method="block_cv")
    if r3["block"] != int(np.ceil(80 ** (1 / 3))):
        return f"default block {r3['block']} != ceil(n^(1/3)) = {int(np.ceil(80 ** (1/3)))}"
    return None


@check("0.8.0 kernel_regression refusals: bandwidth with CV, block with fixed, block=0")
def _():
    x, y = _xy(60, 1)
    out = [
        raises("bandwidth", tsecon.kernel_regression, x[:, 0], y, bandwidth=0.5, bandwidth_method="loo_cv"),
        raises("block", tsecon.kernel_regression, x[:, 0], y, bandwidth=0.5, block=3),
        raises("block", tsecon.kernel_regression, x[:, 0], y, bandwidth_method="block_cv", block=0),
    ]
    out = [o for o in out if o]
    return "; ".join(out) if out else None


@check("0.8.0 group_lasso keys, l1_ratio=1 == lasso, group_weights options, refusals")
def _():
    x, y = _xy(120, 6)
    g = [0, 0, 1, 1, 2, 2]
    r = tsecon.group_lasso(x, y, g, 0.05)
    e = has_keys(r, "coef", "n_iter", "converged", "active_groups", "active_set", "objective", "kkt_violation", "max_rel_change", "alpha_max")
    if e:
        return e
    a = tsecon.group_lasso(x, y, g, 0.05, l1_ratio=1.0)["coef"]
    b = tsecon.lasso(x, y, 0.05)["coef"]
    if np.max(np.abs(np.asarray(a) - np.asarray(b))) > 1e-6:
        return f"l1_ratio=1 differs from lasso by {np.max(np.abs(np.asarray(a) - np.asarray(b))):.2e}"
    tsecon.group_lasso(x, y, g, 0.05, group_weights="none")
    tsecon.group_lasso(x, y, g, 0.05, group_weights=[1.0, 2.0, 1.5])
    out = [raises("l1_ratio", tsecon.group_lasso, x, y, g, 0.05, l1_ratio=1.5), raises("groups", tsecon.group_lasso, x, y, [0, 1], 0.05)]
    out = [o for o in out if o]
    return "; ".join(out) if out else None


@check("0.8.0 post_lasso keys and no se key")
def _():
    x, y = _xy(120, 6)
    r = tsecon.post_lasso(x, y, 0.05)
    e = has_keys(r, "support", "coef_lasso", "coef_ols", "n_selected", "rss")
    if e:
        return e
    return "returns a se-like key" if any(k in r for k in ("se", "bse", "std_err")) else None


@check("0.8.0 pds_lasso keys, alpha='bic' default, hac_lags=0, negative hac_lags refused")
def _():
    y, d, x = build("pds_lasso", T=200, seed=0)[0]
    r = tsecon.pds_lasso(y, d, x)
    e = has_keys(r, "coef", "se", "t_stat", "p_value", "conf_int", "support_y", "support_d", "union_support", "n_controls_selected", "alpha_y", "alpha_d", "hac_lags_resolved")
    if e:
        return e
    if r["hac_lags_resolved"] != int(np.floor(4 * (200 / 100) ** (2 / 9))):
        return f"hac_lags_resolved={r['hac_lags_resolved']} vs Newey-West rule {int(np.floor(4 * (200/100)**(2/9)))}"
    tsecon.pds_lasso(y, d, x, hac_lags=0)
    return raises("hac_lags", tsecon.pds_lasso, y, d, x, hac_lags=-1)


@check("0.8.0 regression_tree keys (n_nodes, n_leaves, depth, importance) and predictions")
def _():
    x, y = _xy(150, 4)
    r = tsecon.regression_tree(x, y, max_depth=3, x_test=x[:3])
    return has_keys(r, "n_nodes", "n_leaves", "depth", "fitted", "predicted", "feature_importance", "splits")


@check("0.8.0 random_forest: bootstrap schemes, block_length contract, max_features, oob, quantiles, importance")
def _():
    x, y = _xy(150, 4)
    r = tsecon.random_forest(x, y, n_trees=10, seed=1, x_test=x[:3], quantiles=[0.1, 0.9])
    e = has_keys(r, "fitted", "predicted", "oob_prediction", "oob_mse", "importance", "quantile_predictions", "max_features_resolved")
    if e:
        return e
    out = []
    out.append(raises("block_length", tsecon.random_forest, x, y, n_trees=5, bootstrap="block"))
    out.append(raises("block_length", tsecon.random_forest, x, y, n_trees=5, bootstrap="iid", block_length=5))
    for bs in ("block", "stationary"):
        tsecon.random_forest(x, y, n_trees=5, bootstrap=bs, block_length=8)
    for mf in ("sqrt", "third", "all", 2):
        tsecon.random_forest(x, y, n_trees=3, max_features=mf)
    tsecon.random_forest(x, y, n_trees=5, importance="block_permutation", importance_groups=[0, 0, 1, 1], permutation_block=5, n_permutations=2)
    tsecon.random_forest(x, y, n_trees=5, importance="impurity")
    out = [o for o in out if o]
    return "; ".join(out) if out else None


@check("0.8.0 random_forest(bootstrap='none', max_features='all', n_trees=1) == regression_tree bit-for-bit")
def _():
    x, y = _xy(150, 4)
    a = np.asarray(tsecon.regression_tree(x, y, max_depth=4, min_samples_leaf=5, x_test=x[:5])["predicted"])
    b = np.asarray(tsecon.random_forest(x, y, bootstrap="none", max_features="all", n_trees=1, max_depth=4, min_samples_leaf=5, x_test=x[:5])["predicted"])
    return None if a.tobytes() == b.tobytes() else f"differ by {np.max(np.abs(a-b)):.2e}"


@check("0.8.0 l1_trend_filter: keys, orders, l2 == hp_filter, tol/max_iter refused under l2")
def _():
    y = np.cumsum(np.random.default_rng(0).standard_normal(300))
    r = tsecon.l1_trend_filter(y, 5.0)
    e = has_keys(r, "trend", "knots", "lam_max", "duality_gap", "converged", "objective")
    if e:
        return e
    tsecon.l1_trend_filter(y, 5.0, order=1)
    l2 = np.asarray(tsecon.l1_trend_filter(y, 1600.0, order=2, penalty="l2")["trend"])
    hp = np.asarray(tsecon.hp_filter(y, lamb=1600.0)["trend"]) if "lamb" in signature_params("hp_filter") else np.asarray(tsecon.hp_filter(y, 1600.0)["trend"])
    if np.max(np.abs(l2 - hp)) > 1e-8:
        return f"l2 trend vs hp_filter differ by {np.max(np.abs(l2-hp)):.2e}"
    out = [raises("tol", tsecon.l1_trend_filter, y, 5.0, penalty="l2", tol=1e-6), raises("max_iter", tsecon.l1_trend_filter, y, 5.0, penalty="l2", max_iter=10)]
    out = [o for o in out if o]
    return "; ".join(out) if out else None


@check("0.8.0 boosting: keys, stop options, learning_rate refusal")
def _():
    x, y = _xy(120, 6)
    r = tsecon.boosting(x, y, n_steps=50, x_test=x[:3])
    e = has_keys(r, "coef_path", "selected", "rss_path", "df_path", "aic_path", "best_step", "fitted", "predicted")
    if e:
        return e
    tsecon.boosting(x, y, n_steps=20, stop="none")
    out = [raises("learning_rate", tsecon.boosting, x, y, learning_rate=1.5), raises("stop", tsecon.boosting, x, y, stop="bogus")]
    out = [o for o in out if o]
    return "; ".join(out) if out else None


@check("0.8.0 mlp_regression: keys, adam/lbfgs, lbfgs refuses learning_rate/batch_size/patience")
def _():
    x, y = _xy(150, 2)
    r = tsecon.mlp_regression(x, y, hidden=8, max_epochs=20, n_seeds=2, seed=0, x_test=x[:3])
    e = has_keys(r, "fitted", "predicted", "member_predictions", "train_loss_path", "validation_loss_path", "best_epoch", "converged", "n_parameters", "weights")
    if e:
        return e
    tsecon.mlp_regression(x, y, hidden=8, max_epochs=20, n_seeds=1, solver="lbfgs")
    out = [raises("learning_rate", tsecon.mlp_regression, x, y, hidden=8, max_epochs=5, n_seeds=1, solver="lbfgs", learning_rate=0.01),
           raises("batch_size", tsecon.mlp_regression, x, y, hidden=8, max_epochs=5, n_seeds=1, solver="lbfgs", batch_size=16),
           raises("patience", tsecon.mlp_regression, x, y, hidden=8, max_epochs=5, n_seeds=1, solver="lbfgs", patience=3),
           raises("activation", tsecon.mlp_regression, x, y, hidden=8, max_epochs=5, n_seeds=1, activation="swish")]
    out = [o for o in out if o]
    return "; ".join(out) if out else None


@check("0.8.0 echo_state_network keys and washout >= n refusal")
def _():
    x, y = _xy(150, 2)
    r = tsecon.echo_state_network(x, y, reservoir_size=30, washout=10, seed=0, x_test=x[:3])
    e = has_keys(r, "fitted", "predicted", "readout", "spectral_radius_achieved", "reservoir_size", "n_washout", "n_train")
    if e:
        return e
    return raises("washout", tsecon.echo_state_network, x, y, reservoir_size=30, washout=150, seed=0)


@check("0.8.0 (round 11) runtime docstrings name every returned key: the five ident functions + gpd/gev + 22 more")
def _():
    names = ["robust_svar_bounds", "fry_pagan_svar", "hetero_svar", "historical_decomposition", "narrative_svar", "gpd_fit", "gev_fit",
             "adaptive_lasso", "check_series", "connectedness", "factor_model", "flp", "flp_scenario", "functional_pca", "fvar_scenario",
             "iv_gmm", "ivx_test", "jarque_bera", "johansen", "nongaussian_svar", "panel_lp", "panel_pmg", "panel_unit_root", "proxy_first_stage",
             "quantile_regression", "setar", "smooth_lp", "sup_f_test", "var_backtest", "local_level_smooth", "engle_granger", "gas_volatility", "zero_sign_svar"]
    bad = {}
    for n in names:
        try:
            res = call(n)
        except Exception as exc:  # noqa: BLE001
            bad[n] = f"call failed: {type(exc).__name__}"
            continue
        if not isinstance(res, dict):
            continue
        doc = getattr(tsecon, n).__doc__ or ""
        missing = [k for k in res if not re.search(rf"\b{re.escape(str(k))}\b", doc)]
        if missing:
            bad[n] = missing
    return f"keys unnamed in __doc__: {bad}" if bad else None


@check("0.8.0 (round 11) max_rel_change documented on lasso/elastic_net/adaptive_lasso and returned")
def _():
    bad = []
    for n in ("lasso", "elastic_net", "adaptive_lasso"):
        res = call(n)
        if "max_rel_change" not in res:
            bad.append(f"{n}: key missing")
        if "max_rel_change" not in (getattr(tsecon, n).__doc__ or ""):
            bad.append(f"{n}: not in __doc__")
    return "; ".join(bad) if bad else None


@check("0.8.0 (round 11) cg_regression keys se_intercept/se_slope/t_slope/p_slope named in __doc__, and no bare se/t/p")
def _():
    res = call("cg_regression")
    e = has_keys(res, "se_intercept", "se_slope", "t_slope", "p_slope")
    if e:
        return e
    doc = tsecon.cg_regression.__doc__ or ""
    missing = [k for k in ("se_intercept", "se_slope", "t_slope", "p_slope") if k not in doc]
    return f"not in __doc__: {missing}" if missing else None


@check("0.8.0 (round 11) seed=None documented as seed 0 on conformal_forecast/conformal_backtest/proxy_ar_sets and measured None == 0 != 1")
def _():
    bad = []
    for n, kw in (("conformal_forecast", "seed"), ("conformal_backtest", "seed"), ("proxy_ar_sets", "rf_seed")):
        doc = getattr(tsecon, n).__doc__ or ""
        if not re.search(r"not fresh entropy", doc, re.I):
            bad.append(f"{n}: __doc__ does not state {kw}=None -> 0")
    y = np.cumsum(np.random.default_rng(1).standard_normal(160)) + 20
    a = tsecon.conformal_forecast(y, horizon=2, method="enbpi", base="ar", seed=None, n_boot=10)
    b = tsecon.conformal_forecast(y, horizon=2, method="enbpi", base="ar", seed=0, n_boot=10)
    c = tsecon.conformal_forecast(y, horizon=2, method="enbpi", base="ar", seed=1, n_boot=10)
    if not np.array_equal(np.asarray(a["lower"]), np.asarray(b["lower"])):
        bad.append("conformal_forecast enbpi seed=None != seed=0")
    if np.array_equal(np.asarray(a["lower"]), np.asarray(c["lower"])):
        bad.append("conformal_forecast enbpi seed=None == seed=1")
    return "; ".join(bad) if bad else None


@check("0.8.0 (round 11) EGARCH forecast_horizon: 1 works, 2 refuses with a clean message (no TODO), on garch_fit/ccc_garch/dcc_garch")
def _():
    r = np.random.default_rng(3).standard_normal(400)
    R2 = np.random.default_rng(4).standard_normal((400, 2))
    bad = []
    tsecon.garch_fit(r, vol="egarch", forecast_horizon=1)
    for name, data in (("garch_fit", r), ("ccc_garch", R2), ("dcc_garch", R2)):
        try:
            getattr(tsecon, name)(data, vol="egarch", forecast_horizon=2)
            bad.append(f"{name}: horizon 2 did not raise")
        except Exception as exc:  # noqa: BLE001
            m = str(exc)
            if "TODO" in m or "phase0" in m:
                bad.append(f"{name}: message carries an internal marker: {m[:120]}")
            if "forecast_horizon" not in m:
                bad.append(f"{name}: message does not name forecast_horizon: {m[:120]}")
    return "; ".join(bad) if bad else None


@check("0.8.0 (round 11) cv_splits docstring says train defaults to 0 and walk-forward refuses it")
def _():
    doc = tsecon.cv_splits.__doc__ or ""
    if not re.search(r"train.{0,120}(0|zero)", doc):
        return "docstring does not state the train=0 default/refusal"
    return raises("train", tsecon.cv_splits, 100)


@check("0.8.0 group_lasso integer groups pass through coercion (ndarray of ints accepted)")
def _():
    x, y = _xy(120, 6)
    tsecon.group_lasso(x, y, np.array([0, 0, 1, 1, 2, 2]), 0.05)
    return None


# ------------------------------------------------------------------ 0.7.0
@check("0.7.0 star_test: delay past the sample / empty series raise ValueError 'insufficient data' (no PanicException)")
def _():
    y = build("star_test")[0][0]
    T = len(y)
    bad = []
    for delays in ([T + 1], [1, T + 50], [T - 1], [T]):
        try:
            tsecon.star_test(y, 1, delays=delays)
            bad.append(f"delays={delays} did not raise")
        except ValueError as exc:
            if "insufficient data" not in str(exc):
                bad.append(f"delays={delays}: {str(exc)[:100]}")
        except BaseException as exc:  # noqa: BLE001
            bad.append(f"delays={delays}: {type(exc).__name__}")
    try:
        tsecon.star_test(np.array([]), 1)
        bad.append("empty did not raise")
    except ValueError:
        pass
    except BaseException as exc:  # noqa: BLE001
        bad.append(f"empty: {type(exc).__name__}")
    return "; ".join(bad) if bad else None


@check("0.7.0 threshold_vecm(k>2, beta=None) recommends the Python route vecm(..., coint_rank=1, deterministic='co')['beta'] — and that route exists")
def _():
    d = np.random.default_rng(0).standard_normal((150, 3)).cumsum(axis=0)
    try:
        tsecon.threshold_vecm(d, n_grid_gamma=10, n_grid_beta=5)
        return "did not raise for k=3 without beta"
    except Exception as exc:  # noqa: BLE001
        m = str(exc)
    kws = re.findall(r"vecm\(([^)]*)\)", m)
    e = has_kw("vecm", "coint_rank", "deterministic")
    if e:
        return f"message says {kws}; {e}"
    if "coint_rank" not in m:
        return f"message does not name coint_rank: {m[:200]}"
    return None


@check("0.7.0 proxy_ar_sets(lags=0, reduced_form_uncertainty=True) refuses naming lags and the alternative")
def _():
    a, k = build("proxy_ar_sets")
    return raises("lags.*reduced_form_uncertainty|reduced_form_uncertainty.*lags", tsecon.proxy_ar_sets, *a, lags=0, reduced_form_uncertainty=True, horizon=4)


@check("0.7.0 hansen_seo_test(n_grid=0) names n_grid; threshold_vecm keeps n_grid_gamma")
def _():
    a, k = build("hansen_seo_test")
    e = raises("n_grid", tsecon.hansen_seo_test, *a, n_grid=0, n_boot=9, seed=0)
    if e:
        return e
    a2, k2 = build("threshold_vecm")
    return raises("n_grid_gamma", tsecon.threshold_vecm, *a2, n_grid_gamma=0, n_grid_beta=5)


@check("0.7.0 bn_filter on a linear ramp says the first differences are constant")
def _():
    return raises("differen", tsecon.bn_filter, np.arange(100.0))


@check("0.7.0 ou_fit / spread_zscore NaN refusal names the index; spread_zscore(kappa=inf) refused for finiteness")
def _():
    x = build("ou_fit")[0][0].copy()
    x[7] = np.nan
    bad = [raises(r"\b7\b", tsecon.ou_fit, x), raises("finite|inf", tsecon.spread_zscore, build("spread_zscore")[0][0], kappa=np.inf, mu=0.0, sigma=1.0)]
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.7.0 vecm: all nine deterministic cases fit, det_coef_coint under ci/li, seasons=, seasons=1 refused, invalid strings refused naming the cases")
def _():
    d = build("vecm")[0][0]
    bad = []
    for case in ("n", "co", "ci", "li", "lo", "colo", "coli", "cilo", "cili"):
        r = tsecon.vecm(d, deterministic=case)
        if case in ("ci", "li", "coli", "cili", "cilo") and "det_coef_coint" not in r:
            bad.append(f"{case}: no det_coef_coint")
    tsecon.vecm(d, seasons=4)
    tsecon.vecm(d, seasons=4, first_season=2)
    bad.append(raises("seasons", tsecon.vecm, d, seasons=1))
    bad.append(raises("nine|n\\b.*co\\b.*ci\\b", tsecon.vecm, d, deterministic="coci"))
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.7.0 star family: gamma_standardized, converged, se_valid, gamma_at_boundary; star_eval; star_test LM3 chi2+F and H03/H02/H01")
def _():
    r = call("star")
    e = has_keys(r, "gamma", "gamma_standardized", "converged", "se_valid", "gamma_at_boundary")
    if e:
        return e
    call("star_eval")
    t = call("star_test")
    txt = " ".join(map(str, t.keys()))
    want = ["lm3", "h3_", "h2_", "h1_"]
    missing = [w for w in want if w not in txt.lower()]
    return f"star_test keys lack {missing}: {sorted(t)}" if missing else None


@check("0.7.0 threshold_var(delays=[...]) and trim default 0.10; threshold_var_test exists")
def _():
    e = has_kw("threshold_var", "delays", "trim")
    if e:
        return e
    sig = inspect.signature(tsecon.threshold_var)
    if sig.parameters["trim"].default not in (0.1, 0.10):
        return f"trim default is {sig.parameters['trim'].default}"
    a, k = build("threshold_var")
    tsecon.threshold_var(*a, delays=[1, 2])
    return exists("threshold_var_test")


@check("0.7.0 inert-kwarg refusals: ccc/dcc/dcc_test o>0 under vol='garch'")
def _():
    R2 = build("ccc_garch")[0][0]
    bad = [raises("o\\b|asymmetry|gjr", getattr(tsecon, n), R2, o=1) for n in ("ccc_garch", "dcc_garch", "dcc_test")]
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.7.0 inert-kwarg refusals: conformal order/lags/gamma/n_boot/calib/n_eval/batch")
def _():
    y = np.cumsum(np.random.default_rng(1).standard_normal(200)) + 20
    cf, cb = tsecon.conformal_forecast, tsecon.conformal_backtest
    bad = [
        raises("order", cf, y, horizon=2, base="ar", order=(1, 0, 0)),
        raises("lags", cf, y, horizon=2, base="theta", lags=2),
        raises("gamma", cf, y, horizon=2, method="split", gamma=0.05),
        raises("n_boot", cf, y, horizon=2, method="split", n_boot=10),
        raises("calib", cf, y, horizon=2, method="enbpi", calib=20),
        raises("n_eval", cf, y, horizon=2, method="split", n_eval=10),
        raises("batch", cb, y, horizon=2, method="split", n_eval=10, batch=2),
    ]
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.7.0 inert-kwarg refusals: hamilton_filter maxlags/use_correction on non-HAC; bn_filter d0 with delta; backtest period with naive; spread_zscore dt with frozen triple; threshold_vecm n_grid_beta with beta; vecm first_season with seasons=0")
def _():
    y = np.cumsum(np.random.default_rng(1).standard_normal(200))
    bad = [
        raises("maxlags", tsecon.hamilton_filter, y, se="nonrobust", maxlags=4),
        raises("use_correction", tsecon.hamilton_filter, y, method="random_walk", use_correction=False),
        raises("d0|dt", tsecon.bn_filter, y, delta=0.2, d0=0.01),
        raises("period", tsecon.backtest, y + 50, train=100, horizon=2, forecaster="naive", period=4),
        raises("dt", tsecon.spread_zscore, y, kappa=0.5, mu=0.0, sigma=1.0, dt=2.0),
        raises("n_grid_beta|beta_span", tsecon.threshold_vecm, build("threshold_vecm")[0][0], beta=[1.0, -1.5], n_grid_beta=5, n_grid_gamma=10),
        raises("first_season|seasons", tsecon.vecm, build("vecm")[0][0], seasons=0, first_season=1),
    ]
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.7.0 an infinite proxy value raises across the proxy family; an all-NaN proxy raises a teaching error")
def _():
    (d, proxy), k = build("proxy_svar")
    p2 = proxy.copy()
    p2[10] = np.inf
    bad = []
    for n in ("proxy_svar", "proxy_first_stage", "proxy_svar_bands", "proxy_ar_sets"):
        kw = {"horizon": 4} if n != "proxy_first_stage" else {}
        if n == "proxy_svar_bands":
            kw.update(n_boot=9, seed=0)
        bad.append(raises("inf", getattr(tsecon, n), d, p2, **kw))
    bad.append(raises("NaN|nan|missing", tsecon.proxy_svar, d, np.full_like(proxy, np.nan), horizon=4))
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.7.0 MASE zero-scale error names insample_period / train")
def _():
    y = np.r_[np.ones(60), np.cumsum(np.random.default_rng(0).standard_normal(60)) + 1]
    return raises("insample_period|train", tsecon.backtest, y, train=50, horizon=1)


@check("0.7.0 ou_fit returns level; markov_switching_ar returns iterations (and 0.6.0 ar)")
def _():
    bad = [has_keys(call("ou_fit"), "level"), has_keys(call("markov_switching_ar"), "iterations", "converged", "ar")]
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


# ------------------------------------------------------------------ 0.6.0
@check("0.6.0 cv_splits purged_kfold: right gap = purge + embargo (21,10 -> 31); embargo refused on expanding")
def _():
    sp = tsecon.cv_splits(300, scheme="purged_kfold", k=5, purge=21, embargo=10)
    f = sp[1]
    test_hi = max(f["test"])
    right = [i for i in f["train"] if i > test_hi]
    gap = (min(right) - test_hi - 1) if right else None  # indices excluded between test end and the next train row
    if gap != 31:
        return f"right gap {gap}, expected 31"
    return raises("embargo", tsecon.cv_splits, 300, scheme="expanding", train=100, embargo=5)


@check("0.6.0 backtest accepts a Python callable; conformal base= callable; EnbPI refuses a callable base; defaults None")
def _():
    y = np.cumsum(np.random.default_rng(1).standard_normal(160)) + 30
    naive = lambda train, h: np.repeat(train[-1], h)  # noqa: E731
    a = tsecon.backtest(y, train=100, horizon=2, forecaster=naive)
    b = tsecon.backtest(y, train=100, horizon=2, forecaster="naive")
    bad = []
    if not np.array_equal(np.asarray(a["forecasts"]), np.asarray(b["forecasts"])):
        bad.append("callable naive != 'naive'")
    r = tsecon.conformal_forecast(y, horizon=2, base=naive)
    if not str(r["base"]).startswith("<callable"):
        bad.append(f"conformal base key = {r['base']!r}")
    bad.append(raises("callable|enbpi", tsecon.conformal_forecast, y, horizon=2, method="enbpi", base=naive, n_boot=5))
    sig = inspect.signature(tsecon.backtest)
    if sig.parameters["forecaster"].default is not None:
        bad.append(f"backtest forecaster default {sig.parameters['forecaster'].default!r}")
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.6.0 ou_fit(x, dt, level) keys; spread_zscore frozen triple; phi>=1 returned not raised")
def _():
    e = has_kw("ou_fit", "dt", "level")
    if e:
        return e
    r = call("ou_fit")
    e = has_keys(r, "kappa", "mu", "sigma", "half_life", "half_life_ci", "mean_reverting", "stationary_sd")
    if e:
        return e
    ex = np.zeros(300)
    for t in range(1, 300):
        ex[t] = 1.05 * ex[t - 1] + np.random.default_rng(t).standard_normal()
    r2 = tsecon.ou_fit(ex)
    if r2["mean_reverting"] or np.isfinite(r2["half_life"]) or r2["half_life_ci"] is not None:
        return f"explosive series: mean_reverting={r2['mean_reverting']} half_life={r2['half_life']} ci={r2['half_life_ci']}"
    tsecon.spread_zscore(ex, kappa=0.5, mu=0.0, sigma=1.0)
    return None


@check("0.6.0 binding-gap keys: dfm_nowcast loadings/factor_ar/factor_cov/idiosyncratic/center/scale; bvar_fit omega_bar/s_bar/v_bar; var_fit resid/fitted/nobs/df_resid; dcc_garch univariate/std_residuals (+nbar under adcc)")
def _():
    bad = [
        has_keys(call("dfm_nowcast"), "loadings", "factor_ar", "factor_cov", "idiosyncratic", "center", "scale"),
        has_keys(call("dfm_nowcast", method="mle"), "loadings", "factor_ar", "factor_cov", "idiosyncratic", "center", "scale", "converged", "iterations"),
        has_keys(call("bvar_fit"), "omega_bar", "s_bar", "v_bar"),
        has_keys(call("var_fit"), "resid", "fitted", "nobs", "df_resid"),
        has_keys(call("dcc_garch"), "univariate", "std_residuals", "covariance", "sigma2"),
        has_keys(call("dcc_garch", variant="adcc"), "nbar"),
        has_keys(call("ccc_garch"), "covariance_forecast", "variance_forecast", "covariance", "sigma2"),
    ]
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.6.0 var_fit fitted + resid reproduces data[lags:]")
def _():
    d = build("var_fit")[0][0]
    r = tsecon.var_fit(d, lags=2)
    return None if np.allclose(np.asarray(r["fitted"]) + np.asarray(r["resid"]), d[2:]) else "fitted + resid != data[lags:]"


@check("0.6.0 ccc/dcc/dcc_test take vol/mean/p/o/q and univariate_dist; garch_fit params_named")
def _():
    bad = [has_kw(n, "vol", "mean", "p", "o", "q", "univariate_dist") for n in ("ccc_garch", "dcc_garch", "dcc_test")]
    bad.append(raises("univariate_dist", tsecon.ccc_garch, build("ccc_garch")[0][0], univariate_dist="bogus"))
    g = call("garch_fit")
    bad.append(has_keys(g, "params_named"))
    if "params_named" in g and dict(zip(g["param_names"], g["params"])) != g["params_named"]:
        bad.append("params_named != dict(zip(param_names, params))")
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.6.0 hamilton_filter method='random_walk', se='hac'/'nonrobust' with bse/tvalues; bn_decomposition, bn_filter exist")
def _():
    y = np.cumsum(np.random.default_rng(1).standard_normal(200))
    bad = [has_kw("hamilton_filter", "method", "se", "maxlags", "use_correction")]
    tsecon.hamilton_filter(y, method="random_walk")
    r = tsecon.hamilton_filter(y, se="hac")
    bad.append(has_keys(r, "bse", "tvalues"))
    tsecon.hamilton_filter(y, se="nonrobust")
    bad.append(exists("bn_decomposition", "bn_filter"))
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.6.0 vecm deterministic='co' returns det_coef (k x 0 under 'n'); johansen returns evec")
def _():
    d = build("vecm")[0][0]
    r = tsecon.vecm(d, deterministic="co")
    bad = [has_keys(r, "det_coef"), has_keys(tsecon.johansen(d), "evec")]
    n = tsecon.vecm(d)
    if np.asarray(n["det_coef"]).shape[1] != 0:
        bad.append(f"'n' det_coef shape {np.asarray(n['det_coef']).shape}")
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.6.0 panel_pmg exposes tol= and max_iter=")
def _():
    return has_kw("panel_pmg", "tol", "max_iter")


@check("0.6.0 convergence flags: arima_fit/auto_arima converged+boundary+se_valid+boundary_note; quantile_lp/growth_at_risk converged; dfm_nowcast two-step carries no converged")
def _():
    bad = [
        has_keys(call("arima_fit"), "converged", "boundary", "se_valid", "boundary_note"),
        has_keys(call("auto_arima"), "converged", "boundary", "se_valid", "boundary_note"),
        has_keys(call("quantile_lp"), "converged"),
        has_keys(call("growth_at_risk"), "converged"),
    ]
    if "converged" in call("dfm_nowcast"):
        bad.append("two-step dfm_nowcast carries converged")
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.6.0 var_fevd is horizon-first: k=3, horizon=6 -> shape (6, 3, 3)")
def _():
    d = build("var_fevd")[0][0]
    s = np.asarray(tsecon.var_fevd(d, lags=1, horizon=6)).shape
    return None if s == (6, 3, 3) else f"shape {s}"


@check("0.6.0 periodogram/welch/coherence default detrend='constant'")
def _():
    bad = []
    for n in ("periodogram", "welch", "coherence"):
        dflt = inspect.signature(getattr(tsecon, n)).parameters["detrend"].default
        if dflt != "constant":
            bad.append(f"{n}: detrend default {dflt!r}")
    return "; ".join(bad) if bad else None


@check("0.6.0 garch_fit(o=1) under vol='garch' raises naming gjr; panel_fe(bandwidth=8) under cluster raises; o default None")
def _():
    bad = [raises("gjr", tsecon.garch_fit, build("garch_fit")[0][0], o=1)]
    a, k = build("panel_fe")
    bad.append(raises("bandwidth|driscoll", tsecon.panel_fe, *a, bandwidth=8.0))
    if inspect.signature(tsecon.garch_fit).parameters["o"].default is not None:
        bad.append("garch_fit o default is not None")
    bad = [b for b in bad if b]
    return "; ".join(bad) if bad else None


@check("0.6.0 proxy_ar_sets(rf_method='second_order_bc') runs")
def _():
    a, k = build("proxy_ar_sets")
    tsecon.proxy_ar_sets(*a, horizon=4, rf_method="second_order_bc", rf_draws=32)
    return None


@check("0.6.0 johansen det_order=0 <-> vecm(deterministic='co'): beta spans evec direction (cosine ~ 1)")
def _():
    d = build("vecm")[0][0]
    j = tsecon.johansen(d)
    v = tsecon.vecm(d, deterministic="co")
    b = np.asarray(v["beta"])[:, 0]
    e = np.asarray(j["evec"])[:, 0]
    cos = abs(b @ e) / (np.linalg.norm(b) * np.linalg.norm(e))
    return None if abs(cos - 1) < 1e-8 else f"cosine {cos}"


def main():
    fh = open(os.path.join(OUT, "sweep_changelog.log"), "w")
    n_pass = n_fail = 0
    for label, fn in CHECKS:
        try:
            r = fn()
        except Exception as exc:  # noqa: BLE001
            r = f"CHECK RAISED {type(exc).__name__}: {str(exc)[:300]}"
        if r is None:
            n_pass += 1
            log(fh, f"PASS  {label}")
        else:
            n_fail += 1
            log(fh, f"FAIL  {label}\n      -> {r}")
    log(fh, f"\n{n_pass} pass, {n_fail} fail of {len(CHECKS)} checks")
    fh.close()


if __name__ == "__main__":
    main()
