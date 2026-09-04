"""Re-verification probes for the consolidated open-items ledger.

Each probe checks one claim the ledger makes about the *installed* wheel —
that a refusal now fires, a kwarg or key now exists, a default was flipped,
or (for the still-open rows) that a gap is still there. Every probe is
wrapped so one failure cannot hide the others; the log prints one line per
probe with a verdict:

  CLOSED   the item the ledger marks fixed/superseded is provably so
  OPEN     the item the ledger marks open is still open on this wheel
  INFO     recorded observation, no verdict
  ERROR    the probe itself failed (stated, never silently dropped)

Run from the repository root with the audit venv:

    .venv-wt/bin/python lab/audit/repo/ledger/probes.py > lab/audit/repo/ledger/probes.log

Read-only: no file under the repository is modified.
"""
from __future__ import annotations

import inspect
import sys
import traceback

import numpy as np

import tsecon

RESULTS: list[tuple[str, str, str]] = []
ATTEMPTED = 0


def probe(pid: str, claim: str):
    def deco(fn):
        global ATTEMPTED
        ATTEMPTED += 1
        try:
            verdict, detail = fn()
        except Exception as exc:  # noqa: BLE001
            verdict, detail = "ERROR", f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=sys.stderr)
        RESULTS.append((pid, verdict, f"{claim} -- {detail}"))
        print(f"[{pid}] {verdict:6s} {claim} -- {detail}", flush=True)
        return fn

    return deco


def raises(fn, *exc_types):
    """Return (True, message) if fn() raises one of exc_types, else (False, repr)."""
    exc_types = exc_types or (Exception,)
    try:
        out = fn()
    except exc_types as exc:
        return True, f"{type(exc).__name__}: {str(exc)[:160]}"
    except BaseException as exc:  # noqa: BLE001  (PanicException is not Exception)
        return False, f"NON-Exception raised: {type(exc).__name__}: {str(exc)[:120]}"
    return False, f"returned {type(out).__name__}"


def sig(fn):
    return inspect.signature(fn)


def default_of(fn, name):
    return sig(fn).parameters[name].default


rng = np.random.default_rng(20260904)
T = 240
y1 = rng.standard_normal(T).cumsum() * 0.1 + rng.standard_normal(T)
ar1 = np.zeros(T)
for t in range(1, T):
    ar1[t] = 0.6 * ar1[t - 1] + rng.standard_normal()
X2 = rng.standard_normal((T, 2))
Y3 = np.zeros((T, 3))
for t in range(1, T):
    Y3[t] = 0.5 * Y3[t - 1] + rng.standard_normal(3)


@probe("P01", "wheel identity")
def _p01():
    n = len([n for n in dir(tsecon) if not n.startswith("_") and callable(getattr(tsecon, n))])
    return "INFO", f"tsecon {tsecon.__version__}, {n} public callables"


@probe("P02", "spec-15 'not yet implemented' banner: proxy_svar_bands/proxy_ar_sets exist")
def _p02():
    ok = callable(getattr(tsecon, "proxy_svar_bands", None)) and callable(getattr(tsecon, "proxy_ar_sets", None))
    return ("CLOSED" if ok else "OPEN"), f"proxy_svar_bands={hasattr(tsecon,'proxy_svar_bands')} proxy_ar_sets={hasattr(tsecon,'proxy_ar_sets')}"


@probe("P03", "rounds 3-4 #4: ivx_test joint default flipped to bonferroni; scalar keys present")
def _p03():
    d = default_of(tsecon.ivx_test, "joint")
    r = rng.standard_normal(T)
    x = np.column_stack([ar1, np.roll(ar1, 3)])
    out = tsecon.ivx_test(r, x)
    keys = sorted(out.keys())
    ok = d == "bonferroni" and "wald_scalar" in out
    return ("CLOSED" if ok else "OPEN"), f"default joint={d!r}; keys={keys}"


@probe("P04", "round 6 #8 / note 21: proxy_ar_sets rf_method menu incl. second_order_bc; default still delta")
def _p04():
    d = default_of(tsecon.proxy_ar_sets, "rf_method")
    doc = tsecon.proxy_ar_sets.__doc__ or ""
    has = all(s in doc for s in ("second_order", "second_order_bc"))
    return ("CLOSED" if has else "OPEN"), f"default rf_method={d!r}; docstring names second_order/second_order_bc={has}"


@probe("P05", "rounds 3-4 #3a: growth_at_risk returns bse_powell")
def _p05():
    out = tsecon.growth_at_risk(ar1, X2[:, :1], horizon=4)
    return ("CLOSED" if "bse_powell" in out else "OPEN"), f"keys={sorted(out.keys())}"


@probe("P06", "rounds 3-4 #3b + field item 2: markov_switching_ar returns (n,k) smoothed_prob, filtered_prob, ar")
def _p06():
    out = tsecon.markov_switching_ar(ar1, k_regimes=2, order=1)
    sp = np.asarray(out["smoothed_prob"])
    ok = sp.ndim == 2 and "filtered_prob" in out and "ar" in out
    return ("CLOSED" if ok else "OPEN"), f"smoothed_prob shape={sp.shape}; has filtered_prob={'filtered_prob' in out}; has ar={'ar' in out}"


@probe("P07", "rounds 3-4 #3c: lp(se='hac', band='sup-t') returns cov_se_max_rel_diff")
def _p07():
    out = tsecon.lp(ar1, X2[:, 0], horizons=6, se="hac", band="sup-t")
    v = out.get("cov_se_max_rel_diff", "MISSING")
    return ("CLOSED" if v != "MISSING" else "OPEN"), f"cov_se_max_rel_diff={v}"


@probe("P08", "round 2 #3 / rounds 3-4 #1: panel_fe refuses an entity-constant regressor (k=2)")
def _p08():
    N, Tn = 20, 30
    x_live = rng.standard_normal((N, Tn))
    x_dead = np.repeat(rng.uniform(0, 1, N)[:, None], Tn, axis=1)  # a share in [0,1], constant within entity
    yy = 0.5 * x_live + rng.standard_normal((N, Tn))
    ok, msg = raises(lambda: tsecon.panel_fe(yy, np.stack([x_live, x_dead])), ValueError)
    return ("CLOSED" if ok else "OPEN"), msg


@probe("P09", "round 2 #1: lp(cumulative='both') defaults to HAC; lag_augmented pairing refused")
def _p09():
    out = tsecon.lp(ar1, X2[:, 0], horizons=6, cumulative="both")
    se_used = out.get("se_method", "?")
    ok1 = str(se_used).lower().startswith("hac")
    ok2, msg = raises(lambda: tsecon.lp(ar1, X2[:, 0], horizons=6, cumulative="both", se="lag_augmented"), ValueError)
    return ("CLOSED" if (ok1 and ok2) else "OPEN"), f"default se_type={se_used!r}; explicit lag_augmented refused={ok2} ({msg[:80]})"


@probe("P10", "round 2 (cv_splits inert purge/embargo) + field item 11: purged_kfold additive; expanding refuses purge")
def _p10():
    a = tsecon.cv_splits(300, scheme="purged_kfold", k=5, purge=21, embargo=10)
    b = tsecon.cv_splits(300, scheme="purged_kfold", k=5, purge=21, embargo=0)
    # measure the right-hand gap between the test block and the next training index
    def right_gap(s):
        tr = np.asarray(s[0]["train"]); te = np.asarray(s[0]["test"])
        after = tr[tr > te.max()]
        return int(after.min() - te.max()) if after.size else None
    ga, gb = right_gap(a), right_gap(b)
    ok_raise, msg = raises(lambda: tsecon.cv_splits(300, scheme="expanding", train=100, embargo=5), ValueError)
    e0 = max(tsecon.cv_splits(300, scheme="expanding", train=100)[0]["train"])
    e5 = max(tsecon.cv_splits(300, scheme="expanding", train=100, purge=5)[0]["train"])
    ok = (ga is not None and gb is not None and ga == gb + 10) and ok_raise and (e0 - e5 == 5)
    return ("CLOSED" if ok else "OPEN"), f"purged_kfold right gap (purge=21,embargo=10)={ga} vs (21,0)={gb} (additive: +10); expanding: purge=5 shortens the first train window by {e0 - e5} (acts), embargo refused={ok_raise} (act-or-raise)"


@probe("P11", "round 6 #3: seasonal_strength refuses a constant series")
def _p11():
    ok, msg = raises(lambda: tsecon.seasonal_strength(np.full(48, 3.7), period=12), ValueError)
    return ("CLOSED" if ok else "OPEN"), msg


@probe("P12", "round 6 #1: bvar_hierarchical default hyperprior is 'glp'")
def _p12():
    d = default_of(tsecon.bvar_hierarchical, "hyperprior")
    return ("CLOSED" if d == "glp" else "OPEN"), f"default hyperprior={d!r}"


@probe("P13", "round 7 F1 + field 16 + round 9: garch_fit boundary flags, converged, params_named; explicit o under vol='garch' refused")
def _p13():
    r = rng.standard_normal(500) * 0.01
    out = tsecon.garch_fit(r)
    need = ["se_valid", "boundary", "boundary_note", "converged", "params_named"]
    missing = [k for k in need if k not in out]
    ok_o, msg = raises(lambda: tsecon.garch_fit(r, o=1), ValueError)
    return ("CLOSED" if not missing and ok_o else "OPEN"), f"missing={missing}; o=1 under garch refused={ok_o}"


@probe("P14", "round 9: arima_fit reports converged + boundary/se_valid/boundary_note")
def _p14():
    out = tsecon.arima_fit(ar1, p=1, d=0, q=1)
    need = ["converged", "boundary", "se_valid", "boundary_note"]
    missing = [k for k in need if k not in out]
    return ("CLOSED" if not missing else "OPEN"), f"missing={missing}"


@probe("P15", "coverage page rec. #5: quantile_regression per-tau converged flag (still a single bool?)")
def _p15():
    out = tsecon.quantile_regression(ar1, X2, taus=[0.05, 0.5, 0.95])
    c = out["converged"]
    is_scalar = np.ndim(c) == 0
    return ("OPEN" if is_scalar else "CLOSED"), f"converged={c!r} (ndim {np.ndim(c)})"


@probe("P16", "lab/REPORT failure mode 1: dm_test has no long-run-variance kernel option")
def _p16():
    params = list(sig(tsecon.dm_test).parameters)
    has = any(p in params for p in ("kernel", "variance", "lrv", "bartlett", "maxlags"))
    return ("CLOSED" if has else "OPEN"), f"signature={params}"


@probe("P17", "rounds 3-4 incidental: iv_gmm positional order (x, z, y) is documented up front (keyword-only NOT adopted)")
def _p17():
    params = list(sig(tsecon.iv_gmm).parameters)[:3]
    doc = (tsecon.iv_gmm.__doc__ or "")[:400]
    leads = "(x, z, y)" in doc or "x, z, y" in doc
    return ("CLOSED" if leads else "OPEN"), f"first three params={params}; doc leads with order={leads}"


@probe("P18", "round 8 residue: theta_forecast still returns a bare array (alpha/b0 unexposed)")
def _p18():
    out = tsecon.theta_forecast(np.abs(ar1) + 5, steps=4)
    bare = isinstance(out, np.ndarray)
    return ("OPEN" if bare else "CLOSED"), f"type={type(out).__name__}"


@probe("P19", "round 2 residue: recession_probit(link='banana', dynamic=True) silently accepted?")
def _p19():
    yb = (rng.uniform(size=T) < 0.25).astype(float)
    ok, msg = raises(lambda: tsecon.recession_probit(yb, X2, link="banana", dynamic=True), ValueError)
    return ("CLOSED" if ok else "OPEN"), msg


@probe("P20", "round 11 OPEN 1: inspect.signature renders defaults as Ellipsis")
def _p20():
    checks = {
        "adf.autolag": default_of(tsecon.adf, "autolag"),
        "zivot_andrews.autolag": default_of(tsecon.zivot_andrews, "autolag"),
        "engle_granger.autolag": default_of(tsecon.engle_granger, "autolag"),
        "box_cox_lambda.bounds": default_of(tsecon.box_cox_lambda, "bounds"),
        "historical_decomposition.restrictions": default_of(tsecon.historical_decomposition, "restrictions"),
        "narrative_svar.sign_restrictions": default_of(tsecon.narrative_svar, "sign_restrictions"),
        "predictive_regression.cz": default_of(tsecon.predictive_regression, "cz"),
        "ivx_test.cz": default_of(tsecon.ivx_test, "cz"),
    }
    ell = [k for k, v in checks.items() if v is Ellipsis]
    return ("OPEN" if ell else "CLOSED"), f"Ellipsis defaults: {ell}"


@probe("P21", "round 11 OPEN 2: check_stationarity docstring names adf_statistic/kpss_p_value/alpha")
def _p21():
    doc = tsecon.check_stationarity.__doc__ or ""
    out = tsecon.check_stationarity(ar1)
    unnamed = [k for k in out.keys() if k not in doc]
    return ("OPEN" if unnamed else "CLOSED"), f"returned keys not in __doc__: {unnamed}"


@probe("P22", "round 10 OPEN: conformal_forecast(method='enbpi') with base omitted refuses")
def _p22():
    ok, msg = raises(lambda: tsecon.conformal_forecast(ar1, method="enbpi", horizon=1), ValueError)
    return ("OPEN" if ok else "CLOSED"), f"refuses with base omitted={ok}: {msg[:110]}"


@probe("P23", "round 6 #9: nsdiffs carries no minimum-cycles advisory key (n=24, period 12 noise)")
def _p23():
    out = tsecon.nsdiffs(rng.standard_normal(24), period=12)
    keys = sorted(out.keys()) if isinstance(out, dict) else [type(out).__name__]
    adv = [k for k in keys if any(s in k.lower() for s in ("cycle", "advis", "warn", "short"))]
    return ("CLOSED" if adv else "OPEN"), f"keys={keys}; advisory-like keys={adv}"


@probe("P24", "brief 'still open': lp_iv / lp_multiplier / lp_state / panel_lp refuse band='sup-t'")
def _p24():
    z = X2[:, 0] + 0.3 * rng.standard_normal(T)
    res = {}
    res["lp_iv"] = raises(lambda: tsecon.lp_iv(ar1, X2[:, 0], z, horizons=4, band="sup-t"), ValueError)[0]
    res["lp_state"] = raises(lambda: tsecon.lp_state(ar1, X2[:, 0], (ar1 > 0).astype(float), horizons=4, band="sup-t"), ValueError)[0]
    N, Tn = 10, 40
    sh = rng.standard_normal(Tn); yy = 0.5 * sh[None, :] + rng.standard_normal((N, Tn))
    res["panel_lp"] = raises(lambda: tsecon.panel_lp(yy, sh, horizon=3, band="sup-t"), ValueError)[0]
    try:
        res["lp_multiplier"] = raises(lambda: tsecon.lp_multiplier(ar1, X2[:, 0], X2[:, 1], horizons=4, band="sup-t"), ValueError)[0]
    except TypeError as exc:
        res["lp_multiplier"] = f"probe-signature: {exc}"
    all_refuse = all(v is True for v in res.values())
    return ("OPEN" if all_refuse else "INFO"), f"sup-t refused: {res}"


@probe("P25", "rounds 3-4 #2: flp docstring discloses generated-regressor SEs (correction not built)")
def _p25():
    doc = (tsecon.flp.__doc__ or "").lower()
    has = "generated" in doc
    params = list(sig(tsecon.flp).parameters)
    corr = any(p in params for p in ("se_correction", "bootstrap", "n_boot"))
    return ("OPEN" if (has and not corr) else "INFO"), f"docstring discloses={has}; correction kwarg present={corr}; params={params}"


@probe("P26", "ROADMAP §0: smooth_fixed still unbound")
def _p26():
    has = hasattr(tsecon, "smooth_fixed")
    return ("CLOSED" if has else "OPEN"), f"hasattr(tsecon,'smooth_fixed')={has}"


@probe("P27", "round 9: var_fevd is horizon-first")
def _p27():
    r = tsecon.var_fevd(Y3, lags=1, horizon=7)
    fv = np.asarray(r["fevd"]) if isinstance(r, dict) else np.asarray(r)
    return ("CLOSED" if fv.shape[0] in (7, 8) and fv.shape[1] == 3 else "OPEN"), f"shape={fv.shape}"


@probe("P28", "round 9: dfm_nowcast loadings / bvar_fit omega_bar / var_fit resid are returned")
def _p28():
    panel = Y3 + rng.standard_normal(Y3.shape) * 0.1
    dfm = tsecon.dfm_nowcast(panel, n_factors=1)
    bv = tsecon.bvar_fit(Y3, lags=1)
    vf = tsecon.var_fit(Y3, lags=1)
    got = {"dfm.loadings": "loadings" in dfm, "bvar.omega_bar": "omega_bar" in bv, "var.resid": "resid" in vf}
    return ("CLOSED" if all(got.values()) else "OPEN"), f"{got}"


@probe("P29", "0.3.0 'not in this release' / SI card: long_run/max_share/hetero/nongaussian SVAR still point-only")
def _p29():
    fns = ["long_run_svar", "max_share_svar", "hetero_svar", "nongaussian_svar"]
    got = {}
    for f in fns:
        params = list(sig(getattr(tsecon, f)).parameters)
        got[f] = [p for p in params if any(s in p for s in ("band", "boot", "n_draws", "alpha"))]
    point_only = all(not v for v in got.values())
    return ("OPEN" if point_only else "INFO"), f"band-like kwargs: {got}"


@probe("P30", "quantile card: quantile_lp still has no HAC/overlap-aware SE option")
def _p30():
    params = list(sig(tsecon.quantile_lp).parameters)
    has = any(p in params for p in ("se", "se_type", "hac_lags", "maxlags"))
    return ("CLOSED" if has else "OPEN"), f"params={params}"


@probe("P31", "0.2.0/0.3.0 'not in this release': no Anderson-Rubin set for iv_gmm/lp_iv; no AP/CD/KP statistics")
def _p31():
    names = [n for n in dir(tsecon) if not n.startswith("_")]
    hits = [n for n in names if any(s in n.lower() for s in ("anderson", "ar_set", "cragg", "kleibergen", "angrist"))]
    ivk = list(sig(tsecon.iv_gmm).parameters)
    ark = [k for k in tsecon.iv_gmm(X2, np.column_stack([X2, rng.standard_normal((T, 1))]), ar1).keys() if "anderson" in k.lower() or "ar_" in k.lower() or "cragg" in k.lower() or "kleibergen" in k.lower()]
    return ("OPEN" if not hits and not ark else "INFO"), f"callable hits={hits}; iv_gmm AR/CD/KP keys={ark}; iv_gmm params={ivk}"


@probe("P32", "var-svar card: zero_sign_svar with a horizon>=1 zero and weighted=True (exact ARW weight unimplemented)")
def _p32():
    try:
        s = sig(tsecon.zero_sign_svar)
        names = list(s.parameters)
    except Exception as exc:  # noqa: BLE001
        return "INFO", f"signature failed: {exc}"
    return "INFO", f"zero_sign_svar params={names}; crate zero.rs documents 'no ARW weight exists' for non-impact zeros (swap point)"


@probe("P33", "round 1: iv_gmm(weight='hac', bandwidth=0.0) refuses; hac_bandwidth returned")
def _p33():
    z = np.column_stack([X2, rng.standard_normal((T, 1))])
    ok, msg = raises(lambda: tsecon.iv_gmm(X2, z, ar1, weight="hac", bandwidth=0.0), ValueError)
    out = tsecon.iv_gmm(X2, z, ar1, weight="hac")
    return ("CLOSED" if ok and "hac_bandwidth" in out else "OPEN"), f"bandwidth=0 refused={ok}; hac_bandwidth={out.get('hac_bandwidth')}"


@probe("P34", "round 1: ols se_type hc2/hc3 exist")
def _p34():
    o2 = tsecon.ols(ar1, X2, se_type="hc2"); o3 = tsecon.ols(ar1, X2, se_type="hc3")
    return "CLOSED", f"hc2 bse[0]={float(np.asarray(o2['bse'])[0]):.4g} hc3 bse[0]={float(np.asarray(o3['bse'])[0]):.4g}"


@probe("P35", "round 1: var_irf_bands(bias_correct=True) on the asymptotic default refuses")
def _p35():
    ok, msg = raises(lambda: tsecon.var_irf_bands(Y3, lags=1, horizon=4, bias_correct=True), ValueError)
    return ("CLOSED" if ok else "OPEN"), msg


@probe("P36", "round 11 L1: lasso returns max_rel_change and names it in __doc__")
def _p36():
    Xc = X2 - X2.mean(0); yc = ar1 - ar1.mean()
    out = tsecon.lasso(Xc, yc, alpha=0.1)
    ok = "max_rel_change" in out and "max_rel_change" in (tsecon.lasso.__doc__ or "")
    return ("CLOSED" if ok else "OPEN"), f"key={'max_rel_change' in out}; doc={'max_rel_change' in (tsecon.lasso.__doc__ or '')}"


@probe("P37", "round 11 M1/M2/M6: every returned key of gpd_fit / robust_svar_bounds / cg_regression is named in __doc__")
def _p37():
    res = {}
    g = tsecon.gpd_fit(np.abs(rng.standard_t(4, 2000)), quantile=0.9)
    res["gpd_fit"] = [k for k in g if k not in (tsecon.gpd_fit.__doc__ or "")]
    c = tsecon.cg_regression(ar1[1:], np.diff(ar1))
    res["cg_regression"] = [k for k in c if k not in (tsecon.cg_regression.__doc__ or "")]
    doc = tsecon.robust_svar_bounds.__doc__ or ""
    res["robust_svar_bounds_doc_len"] = len(doc)
    ok = not res["gpd_fit"] and not res["cg_regression"] and len(doc) > 500
    return ("CLOSED" if ok else "OPEN"), f"{res}"


@probe("P38", "rounds 3-4 #5: long_memory_d.__doc__ names se_asymptotic; predictive_regression.__doc__ names rho_ols")
def _p38():
    a = "se_asymptotic" in (tsecon.long_memory_d.__doc__ or "")
    b = "rho_ols" in (tsecon.predictive_regression.__doc__ or "")
    return ("CLOSED" if a and b else "OPEN"), f"long_memory_d se_asymptotic={a}; predictive_regression rho_ols={b}"


@probe("P39", "round 2 #8: engle_granger on a (1,k) sample raises ValueError, not PanicException")
def _p39():
    ok, msg = raises(lambda: tsecon.engle_granger(Y3[:1]), Exception)
    return ("CLOSED" if ok else "OPEN"), msg


@probe("P40", "round 10 severe: star_test(delay=T+1) raises a catchable ValueError")
def _p40():
    ok, msg = raises(lambda: tsecon.star_test(ar1, 2, delay=T + 1), Exception)
    return ("CLOSED" if ok else "OPEN"), msg


@probe("P41", "round 11 M3: EGARCH multi-step forecast refused cleanly (simulation route still unshipped)")
def _p41():
    r = rng.standard_normal(600) * 0.01
    ok, msg = raises(lambda: tsecon.garch_fit(r, vol="egarch", forecast_horizon=2), ValueError)
    clean = ok and "TODO" not in msg
    return ("OPEN" if clean else "INFO"), f"horizon=2 refused={ok}, message clean={clean}: {msg[:100]}"


@probe("P42", "field item 12 / 0.7.0: vecm(deterministic='ci') and seasons supported")
def _p42():
    out = tsecon.vecm(Y3, k_ar_diff=1, coint_rank=1, deterministic="ci")
    return "CLOSED", f"keys={sorted(out.keys())[:8]}..."


@probe("P43", "round 11 M4: conformal_forecast seed=None documented as seed 0")
def _p43():
    doc = tsecon.conformal_forecast.__doc__ or ""
    has = "None" in doc and "0" in doc and "seed" in doc
    return ("CLOSED" if has else "OPEN"), f"docstring mentions seed None->0={has}"


@probe("P44", "round 10 sweep B: hamilton_filter(maxlags=…, se='nonrobust') refuses")
def _p44():
    ok, msg = raises(lambda: tsecon.hamilton_filter(ar1, h=8, p=4, se="nonrobust", maxlags=3), ValueError)
    return ("CLOSED" if ok else "OPEN"), msg


@probe("P45", "validation-matrix named follow-up: mapie TimeSeriesRegressor (enbpi/aci) is importable in the audit venv")
def _p45():
    import mapie  # noqa: F401
    from mapie.regression import TimeSeriesRegressor  # noqa: F401
    import inspect as _i
    s = str(_i.signature(TimeSeriesRegressor))
    return "INFO", f"mapie {mapie.__version__}; TimeSeriesRegressor{s[:160]} -- reference exists, cross-check still not run"


@probe("P46", "panel card: lp_did has no covariates / composition / pmd / IV kwargs")
def _p46():
    params = list(sig(tsecon.lp_did).parameters)
    has = [p for p in params if any(s in p for s in ("covar", "control", "pmd", "iv", "instrument", "composition"))]
    return ("OPEN" if not has else "INFO"), f"params={params}"


@probe("P47", "copulas card: d>2 refused (deferred)")
def _p47():
    u = tsecon.pseudo_obs(rng.standard_normal((300, 3)))
    ok, msg = raises(lambda: tsecon.copula_fit(u, family="gaussian"), ValueError)
    return ("OPEN" if ok else "INFO"), f"3-column u refused={ok}: {msg[:100]}"


@probe("P48", "coint-regime card: no regime-dependent GIRF for threshold_var")
def _p48():
    names = [n for n in dir(tsecon) if "girf" in n.lower() or "generalized_irf" in n.lower()]
    return ("OPEN" if not names else "CLOSED"), f"girf-like callables={names}"


@probe("P49", "spec-15 uncertain #2/#4: proxy_svar_bands ships Hall+Efron; block_length rule; k>1 proxies")
def _p49():
    m = Y3[1:, 0] + rng.standard_normal(T - 1)
    out = tsecon.proxy_svar_bands(Y3, m, lags=1, horizon=4, n_boot=50, seed=1)
    keys = sorted(out.keys())
    m2 = np.column_stack([m, rng.standard_normal(T - 1)])
    ok2, msg2 = raises(lambda: tsecon.proxy_svar_bands(Y3, m2, lags=1, horizon=4, n_boot=50, seed=1), Exception)
    return "INFO", f"keys={keys}; block_length={out.get('block_length')}; two-column proxy refused={ok2} ({msg2[:80]})"


@probe("P50", "coverage page: nongaussian_svar and garch variance_forecast ship no interval (key-set tripwire)")
def _p50():
    ng = tsecon.nongaussian_svar(Y3, lags=1, horizon=4)
    ik = [k for k in ng if any(s in k for s in ("lower", "upper", "se", "band", "ci"))]
    return "INFO", f"nongaussian_svar interval-like keys={ik} (ships no interval, by design; the coverage page tripwires it)"


@probe("P51", "0.3.0 'not in this release': SARIMA seasonal orders shipped")
def _p51():
    has = "seasonal" in sig(tsecon.arima_fit).parameters
    return ("CLOSED" if has else "OPEN"), f"arima_fit has seasonal kwarg={has}"


@probe("P52", "round 10 OPEN 2: bn_decomposition documents that p/q are ignored on the fixed path")
def _p52():
    doc = tsecon.bn_decomposition.__doc__ or ""
    has = "ignored" in doc and "`p`/`q`" in doc or ("p`/`q` are ignored" in doc)
    return ("CLOSED" if has else "OPEN"), f"docstring says p/q ignored on fixed path={has}"


@probe("P53", "round 8 R2/R3: recession_probit stub says link is probit or logit with no dynamic caveat")
def _p53():
    doc = tsecon.recession_probit.__doc__ or ""
    return "INFO", f"runtime __doc__ mentions 'probit only'={'probit only' in doc}; 'ignored'={'ignored' in doc}"


@probe("P54", "lab/REPORT graduation: l1_trend_filter (salvage b) shipped; no Fourier-terms builder (salvage a); no dynamic_quantile (AL-GAS)")
def _p54():
    got = {
        "l1_trend_filter": hasattr(tsecon, "l1_trend_filter"),
        "fourier_terms": any("fourier" in n.lower() for n in dir(tsecon)),
        "dynamic_quantile": any(n.lower() in ("dynamic_quantile", "al_gas", "caviar") for n in dir(tsecon)),
        "dcs_local_level": hasattr(tsecon, "dcs_local_level"),
        "var_backtest": hasattr(tsecon, "var_backtest"),
    }
    return "INFO", f"{got}"


@probe("P55", "round 11 OPEN 5-8 (performance, defaults @T=3200): arima_fit / historical_decomposition / mcmc_diagnostics timings")
def _p55():
    import time
    yl = rng.standard_normal(3200).cumsum() * 0.05 + rng.standard_normal(3200)
    t0 = time.perf_counter(); tsecon.arima_fit(yl); t1 = time.perf_counter()
    Yl = np.zeros((3200, 3))
    for t in range(1, 3200):
        Yl[t] = 0.5 * Yl[t - 1] + rng.standard_normal(3)
    t2 = time.perf_counter(); tsecon.historical_decomposition(Yl, lags=1); t3 = time.perf_counter()
    draws = rng.standard_normal((2, 3200))
    t4 = time.perf_counter(); tsecon.mcmc_diagnostics(draws); t5 = time.perf_counter()
    return "OPEN", f"arima_fit {t1-t0:.2f}s; historical_decomposition {t3-t2:.3f}s; mcmc_diagnostics {t5-t4:.3f}s (one machine; the round-11 numbers were 6.10 / 0.43 / 0.09)"


@probe("P56", "brief lens-7 target 'adl_midas': no such callable; the MIDAS surface is midas_weights/umidas/weighted_midas")
def _p56():
    names = [n for n in dir(tsecon) if "midas" in n.lower()]
    return ("INFO"), f"midas callables={names}"


@probe("P57", "round 2 #6 / 0.3.0: panel_lp has bias_correction with 'spj'")
def _p57():
    params = list(sig(tsecon.panel_lp).parameters)
    doc = tsecon.panel_lp.__doc__ or ""
    return ("CLOSED" if "bias_correction" in params and "spj" in doc else "OPEN"), f"bias_correction kwarg={'bias_correction' in params}; spj documented={'spj' in doc}"


@probe("P58", "round 9: panel_fe/panel_lp refuse bandwidth under cluster SEs; spectral detrend default 'constant'")
def _p58():
    N, Tn = 10, 40
    sh = rng.standard_normal((N, Tn)); yy = 0.5 * sh + rng.standard_normal((N, Tn))
    ok, msg = raises(lambda: tsecon.panel_fe(yy, sh[None], bandwidth=3.0), ValueError)
    d = default_of(tsecon.welch, "detrend")
    return ("CLOSED" if ok and d == "constant" else "OPEN"), f"bandwidth under cluster refused={ok}; welch detrend default={d!r}"


print(f"\nprobes attempted: {ATTEMPTED}, reached: {len(RESULTS)}")
from collections import Counter
print("verdicts:", dict(Counter(v for _, v, _ in RESULTS)))
