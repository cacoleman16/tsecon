"""Sweep E — the result-object contract for the 0.10.0 surface.

The thirteen new callables plus the four `mask=` cells:

(i)   tsecon.summarize(result).summary() renders;
(ii)  json.dumps(default=...) and pickle round-trip the result bit-for-bit;
(iii) every float is finite, or its NaN/inf is mentioned in __doc__;
(iv)  returned keys vs the keys named by __doc__, the stub and the model card,
      in BOTH directions (a returned key named nowhere; a backticked name in
      a Returns/Keys sentence that is never returned);
(v)   array shapes vs the shapes the docstrings state (explicit table below);
(vi)  the container convention (ndarray vs nested list vs Python list) of every
      returned key, censused over the WHOLE library so the thirteen can be
      compared with the 179 that came before.

Run:  .venv/bin/python lab/audit/round14/sweep_e_contract.py
Out:  lab/audit/round14/out/sweep_e.txt, sweep_e.json, sweep_e_conventions.txt
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import (  # noqa: E402
    OUT, PYI, bits_equal, card_section, code_tokens, doc_tokens, json_roundtrip, log,
    nonfinite_paths, pickle_roundtrip, shapes, top_keys,
)
from registry import MASKED, NAMES, NEW14, build  # noqa: E402

SCOPE = NEW14 + MASKED


def stub_docs():
    tree = ast.parse(open(PYI, encoding="utf-8").read())
    return {n.name: ast.get_docstring(n) or "" for n in tree.body if isinstance(n, ast.FunctionDef)}


def key_sentence_names(doc):
    """Backticked names inside 'Returns ...'/'Returned keys:'/'Keys:'/'→'."""
    flat = re.sub(r"\s+", " ", doc or "")
    names = set()
    for m in re.finditer(r"(?:Returned keys:|Keys:|Returns|What is returned|→)(.*?)(?:\. [A-Z]|$)", flat):
        names |= set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", m.group(1)))
    return names


def _n(v):
    return tuple(np.asarray(v, dtype=float).shape)


def expected_shapes(name, res, args, kwargs):  # noqa: C901
    """The shapes the docstrings state, as {key: expected}."""
    exp = {}
    if name == "unobserved_components":
        nobs, ks, kp = res["nobs"], res["k_states"], res["k_params"]
        for k in ("params", "se", "at_boundary", "param_names"):
            exp[k] = (kp,)
        exp["state_names"] = (ks,)
        for k in ("filtered_state", "filtered_state_var", "smoothed_state", "smoothed_state_var"):
            exp[k] = (nobs, ks)
        for k in ("fitted", "resid", "std_resid"):
            exp[k] = (nobs,)
        for k in ("level", "level_var", "filtered_level", "filtered_level_var",
                  "slope", "slope_var", "filtered_slope", "filtered_slope_var",
                  "seasonal", "seasonal_var", "filtered_seasonal", "filtered_seasonal_var"):
            if res.get(k) is not None:
                exp[k] = (nobs,)
        h = kwargs.get("forecast_steps", 0)
        exp["forecast"] = exp["forecast_var"] = (h,)
    elif name == "tvp_regression":
        nobs, k, kp = res["nobs"], res["k"], res["k_params"]
        for key in ("params", "se", "at_boundary", "param_names"):
            exp[key] = (kp,)
        for key in ("sigma2_beta", "pile_up", "coef_names"):
            exp[key] = (k,)
        for key in ("beta_filtered", "beta_filtered_var", "beta_smoothed", "beta_smoothed_var"):
            exp[key] = (nobs, k)
        for key in ("fitted", "resid", "std_resid"):
            exp[key] = (nobs,)
    elif name in ("ets_fit", "auto_ets"):
        nobs, h = res["nobs"], res["horizon"]
        m = res["seasonal_periods"] or 0
        for key in ("fitted", "resid", "level_path"):
            exp[key] = (nobs,)
        if res["trend"]:
            exp["trend_path"] = (nobs,)
        if res["seasonal"]:
            exp["seasonal_path"] = (nobs,)
            exp["final_seasonal"] = exp["initial_seasonal"] = (m,)
        for key in ("forecast", "forecast_variance", "forecast_lower", "forecast_upper"):
            exp[key] = (h,)
        nstate = 1 + (1 if res["trend"] else 0) + m
        exp["final_states"] = exp["initial_states"] = exp["initial_state_names"] = (nstate,)
        exp["params"] = exp["param_names"] = (len(res["params"]),)
        if name == "auto_ets":
            exp["candidates"] = (res["n_candidates"],)
    elif name == "var_conditional_forecast":
        steps, k = res["steps"], np.asarray(args[0]).shape[1]
        for key in ("point", "unconditional", "se", "unconditional_se", "lower", "upper",
                    "shocks", "orth_shocks", "constrained"):
            exp[key] = (steps, k)
        exp["cov"] = (steps, k, k)
    elif name == "var_diagnostics":
        k, p = res["k"], res["lags"]
        exp["skewness_components"] = exp["kurtosis_components"] = (k,)
        exp["roots"] = exp["eigenvalue_moduli"] = (k * p,)
    elif name == "var_select_order":
        n = len(res["candidates"])
        for key in ("candidates", "aic_values", "bic_values", "hqic_values", "fpe_values"):
            exp[key] = (n,)
    elif name in ("spa_test", "stepm_test"):
        m, reps = res["m"], res["reps"]
        for key in ("mean_loss_diff", "loss_diff_var", "recentered"):
            exp[key] = (m,)
        for key in ("crit_levels", "crit_lower", "crit_consistent", "crit_upper"):
            exp[key] = (3,)
        for key in ("boot_lower", "boot_consistent", "boot_upper"):
            exp[key] = (reps,)
        if name == "stepm_test":
            exp["step_crit_values"] = (res["n_steps"],)
            # `steps` is a ragged list of lists (one index list per step)
            exp["steps"] = (res["n_steps"],)
            exp["superior_models"] = (res["n_superior"],)
    elif name == "model_confidence_set":
        m = res["m"]
        for key in ("mcs_p_values", "mean_losses", "step_p_values", "elimination_order"):
            exp[key] = (m,)
        exp["statistics"] = (res["n_steps"],)
    elif name in ("fmols", "ccr"):
        K = res["n_x"] + res["n_det"]
        kx = res["n_x"]
        for key in ("params", "se", "tvalues", "pvalues", "param_names", "ols_params", "ols_se"):
            exp[key] = (K,)
        exp["cov"] = (K, K)
        for key in ("omega", "lambda", "sigma"):
            exp[key] = (1 + kx, 1 + kx)
        exp["resid"] = (res["nobs"],)
    elif name == "dols":
        K, P = res["n_x"] + res["n_det"], res["n_params"]
        for key in ("params", "se", "tvalues", "pvalues", "param_names"):
            exp[key] = (K,)
        exp["cov"] = (K, K)
        for key in ("full_params", "full_se", "full_param_names"):
            exp[key] = (P,)
        exp["full_cov"] = (P, P)
        exp["resid"] = (res["nobs"],)
        for key in ("ols_params", "ols_se"):
            exp[key] = (K,)
    elif name == "panel_fe@mask":
        K = len(res["params"])
        for key in ("params", "bse", "tvalues"):
            exp[key] = (K,)
    elif name == "panel_lp@mask":
        h = kwargs["horizon"] + 1
        for key in ("irf", "se", "nobs"):
            exp[key] = (h,)
    elif name == "lp_did@mask":
        n = kwargs["pre_window"] + kwargs["post_window"] + 1
        for key in ("coef", "se", "horizons", "nobs", "n_switchers"):
            exp[key] = (n,)
    elif name == "panel_distributed_lag@mask":
        K = len(res["params"])
        k, L, P = np.asarray(args[1]).shape[0], kwargs["lags"], kwargs["powers"]
        for key in ("params", "bse", "tvalues"):
            exp[key] = (K,)
        exp["cov"] = (K, K)
        exp["lag_effects"] = exp["lag_se"] = (k, P, L + 1)
        for key in ("cumulative_effect", "cumulative_se", "cumulative_ci_low", "cumulative_ci_high"):
            exp[key] = (k, P)
        npts = len(kwargs["eval_points"])
        for key in ("eval_points", "marginal_effect", "marginal_se"):
            exp[key] = (k, npts)
        exp["turning_point"] = exp["turning_point_se"] = (k,)
    return exp


def container(v):
    """(kind, ndim, dtype-kind) for one returned value."""
    if isinstance(v, np.ndarray):
        return ("ndarray", v.ndim, v.dtype.kind)
    if isinstance(v, list):
        a = v
        nd = 0
        while isinstance(a, list) and a:
            nd += 1
            a = a[0]
        if isinstance(a, list):
            nd += 1
        kind = {float: "f", int: "i", bool: "b", str: "U", dict: "O"}.get(type(a), "O")
        return ("list", nd, kind)
    return (type(v).__name__, 0, "")


def convention_census(fh):
    """Every returned key of every registry callable, bucketed by container."""
    from collections import Counter, defaultdict

    buckets = defaultdict(list)
    reached = 0
    for name in NAMES:
        try:
            args, kwargs = build(name, T=120, seed=0)
            res = getattr(tsecon, name.split("@")[0])(*args, **kwargs)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(res, dict):
            continue
        reached += 1
        for k, v in res.items():
            buckets[container(v)].append(f"{name}.{k}")
    log(fh, f"container census: {reached} callables reached, "
            f"{sum(len(v) for v in buckets.values())} keys")
    for key in sorted(buckets, key=lambda c: -len(buckets[c])):
        log(fh, f"  {key}: {len(buckets[key])}  e.g. {buckets[key][:3]}")
    # the interesting cell: 1-D float payloads — ndarray or list?
    nd1 = Counter()
    for (kind, nd, dk), names in buckets.items():
        if nd == 1 and dk == "f":
            nd1[kind] += len(names)
    log(fh, f"1-D float payloads: {dict(nd1)}")
    return {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in buckets.items()}


def main():
    fh = open(os.path.join(OUT, "sweep_e.txt"), "w")
    stubs = stub_docs()
    report = {}
    n_cand = 0
    for name in SCOPE:
        base = name.split("@")[0]
        fn = getattr(tsecon, base)
        args, kwargs = build(name, T=200, seed=0)
        res = fn(*args, **kwargs)
        rec = {"keys": sorted(res)}
        try:
            text = tsecon.summarize(res, title=base).summary()
            rec["summarize_lines"] = text.count("\n") + 1
        except Exception as exc:  # noqa: BLE001
            rec["summarize_err"] = f"{type(exc).__name__}: {exc}"
            log(fh, f"[{name}] SUMMARIZE FAILED {rec['summarize_err']}")
            n_cand += 1
        ok, why, back = json_roundtrip(res)
        rec["json"] = ok and bits_equal(res, back)[0]
        if not rec["json"]:
            log(fh, f"[{name}] JSON round-trip: {why or bits_equal(res, back)[1]}")
            n_cand += 1
        ok, why, pb = pickle_roundtrip(res)
        rec["pickle"] = ok and bits_equal(res, pb)[0]
        if not rec["pickle"]:
            log(fh, f"[{name}] PICKLE round-trip: {why or bits_equal(res, pb)[1]}")
            n_cand += 1
        nf = nonfinite_paths(res)
        doc = fn.__doc__ or ""
        mentions = bool(re.search(r"\bNaN\b|\bnan\b|\binf\b|non-finite", doc))
        rec["nonfinite"] = nf
        if nf:
            log(fh, f"[{name}] non-finite ({'documented' if mentions else 'UNDOCUMENTED'}): {nf}")
            if not mentions:
                n_cand += 1
        keys = top_keys(res)
        card = card_section(base)
        surfaces = {"__doc__": doc, "stub": stubs.get(base, ""), "card": card}
        rec["keys_unnamed"] = {}
        for sname, text in surfaces.items():
            toks = code_tokens(text) if sname == "card" else doc_tokens(text)
            missing = sorted(k for k in keys if k not in toks)
            rec["keys_unnamed"][sname] = missing
            if missing:
                log(fh, f"[{name}] returned keys NOT named in {sname}: {missing}")
                n_cand += 1
        phantom = {}
        for sname, text in surfaces.items():
            cand = key_sentence_names(text)
            ph = sorted(c for c in cand if c not in keys)
            phantom[sname] = ph
            if ph:
                log(fh, f"[{name}] phantom candidates in {sname}: {ph}")
        rec["phantom_candidates"] = phantom
        got = {p.replace("$.", ""): s for p, s in shapes(res).items()}
        # ragged payloads (a list of per-step index lists) are checked by length
        for ragged in ("steps", "candidates", "trace"):
            if isinstance(res.get(ragged), list):
                got[ragged] = (len(res[ragged]),)
        exp = expected_shapes(name, res, args, kwargs)
        bad = {}
        for key, want in exp.items():
            have = got.get(key)
            if have is None:
                v = res.get(key)
                if isinstance(v, (list, tuple)) and v and isinstance(v[0], (str, dict, list)):
                    have = (len(v),)
                elif isinstance(v, (list, np.ndarray)):
                    have = _n(v)
                else:
                    have = ()
            if tuple(have) != tuple(want):
                bad[key] = (list(have), list(want))
        rec["shape_mismatch"] = bad
        if bad:
            log(fh, f"[{name}] SHAPE MISMATCH (got vs documented): {bad}")
            n_cand += 1
        rec["shapes_checked"] = len(exp)
        rec["containers"] = {k: list(container(v)) for k, v in res.items()}
        log(fh, f"[{name}] {len(keys)} keys; summarize={rec.get('summarize_lines', 'FAIL')} lines; "
                f"json={rec['json']} pickle={rec['pickle']} nonfinite={len(nf)} "
                f"shapes_checked={len(exp)}")
        report[name] = rec
    log(fh, f"\nREACHED {len(report)}/{len(SCOPE)}; candidates raised: {n_cand}")
    cfh = open(os.path.join(OUT, "sweep_e_conventions.txt"), "w")
    census = convention_census(cfh)
    cfh.close()
    json.dump({"per_call": report, "census": {k: len(v) for k, v in census.items()}},
              open(os.path.join(OUT, "sweep_e.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
