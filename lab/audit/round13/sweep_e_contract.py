"""Sweep E — the result-object contract for the six 0.9.0 callables.

(i)   tsecon.summarize(result).summary() renders;
(ii)  json.dumps(default=...) and pickle round-trip the result bit-for-bit;
(iii) every float is finite, or its NaN/inf is mentioned in __doc__;
(iv)  returned keys vs the keys named by __doc__, the stub and the model card,
      in BOTH directions (a returned key named nowhere; a backticked name in
      a Returns/Keys sentence that is never returned);
(v)   array shapes vs the shapes the docstrings state (explicit table below);
(vi)  the tsecon.results facade: none of the six has a typed results class,
      so only the generic summarize path applies (recorded).

Run:  .venv/bin/python lab/audit/round13/sweep_e_contract.py
Out:  lab/audit/round13/out/sweep_e.txt, sweep_e.json
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
from registry import NEW, build  # noqa: E402

def stub_docs():
    tree = ast.parse(open(PYI, encoding="utf-8").read())
    return {n.name: ast.get_docstring(n) or "" for n in tree.body if isinstance(n, ast.FunctionDef)}


def key_sentence_names(doc):
    """Backticked names inside 'Returns ...'/'Returned keys:'/'→' sentences."""
    flat = re.sub(r"\s+", " ", doc or "")
    names = set()
    for m in re.finditer(r"(?:Returned keys:|Returns|What is returned|→)(.*?)(?:\. [A-Z]|$)", flat):
        names |= set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", m.group(1)))
    return names


def expected_shapes(name, res, args, kwargs):
    """The shapes the docstrings state, as (path, expected) pairs."""
    exp = {}
    if name == "setar_threshold_ci":
        n = len(res["thresholds"])
        for k in ("ssr_path", "lr_stat", "in_set"):
            exp[k] = (n,)
        kk = res["k"]
        exp["slope_ci_low"] = (2, kk)
        exp["slope_ci_high"] = (2, kk)
    elif name in ("var_girf", "threshold_var_girf"):
        h, k = res["horizon"], np.asarray(args[0]).shape[1]
        for key in ("girf", "lower", "upper", "mc_se", "draw_sd", "draw_lower", "draw_upper"):
            exp[key] = (h + 1, k)
        exp["per_history"] = (res["n_histories"], h + 1, k)
        if name == "var_girf":
            exp["shock_vector"] = (k,)
        else:
            exp["shock_vector"] = (2, k)
            exp["shock_size_used"] = (2,)
            exp["girf_low_regime"] = (h + 1, k)
            exp["girf_high_regime"] = (h + 1, k)
            exp["history_regimes"] = (res["n_histories"],)
            exp["history_times"] = (res["n_histories"],)
    elif name == "jsz_fit":
        T, M = np.asarray(args[0]).shape
        N = res["n_factors"]
        for key in ("lambda_q", "mu_p", "mu_p_se", "k0_q_p", "lambda0"):
            exp[key] = (N,)
        for key in ("sigma", "phi_p", "phi_p_se", "sigma_ols", "k1_q_p", "lambda1"):
            exp[key] = (N, N)
        for key in ("a_p", "a_x", "rmse"):
            exp[key] = (M,)
        for key in ("b_p", "b_x"):
            exp[key] = (M, N)
        for key in ("fitted", "risk_neutral", "term_premium"):
            exp[key] = (T, M)
        exp["factors"] = (T, N)
        exp["w"] = (N, M)
    elif name == "jsz_loadings":
        M, N = len(args[3]), len(args[0])
        exp.update({"a_x": (M,), "b_x": (M, N), "k0_q": (N,), "k1_q": (N, N), "maturities": (M,)})
    elif name == "panel_distributed_lag":
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


def main():
    fh = open(os.path.join(OUT, "sweep_e.txt"), "w")
    stubs = stub_docs()
    report = {}
    n_cand = 0
    for name in NEW:
        fn = getattr(tsecon, name)
        args, kwargs = build(name, T=200, seed=0)
        res = fn(*args, **kwargs)
        rec = {"keys": sorted(res)}
        # (i) summarize
        try:
            text = tsecon.summarize(res, title=name).summary()
            rec["summarize_lines"] = text.count("\n") + 1
        except Exception as exc:  # noqa: BLE001
            rec["summarize_err"] = f"{type(exc).__name__}: {exc}"
            log(fh, f"[{name}] SUMMARIZE FAILED {rec['summarize_err']}")
            n_cand += 1
        # (ii) json / pickle
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
        # (iii) non-finite
        nf = nonfinite_paths(res)
        doc = fn.__doc__ or ""
        mentions = bool(re.search(r"\bNaN\b|\bnan\b|\binf\b|non-finite", doc))
        rec["nonfinite"] = nf
        if nf:
            log(fh, f"[{name}] non-finite ({'documented' if mentions else 'UNDOCUMENTED'}): {nf}")
            if not mentions:
                n_cand += 1
        # (iv) keys, three surfaces, both directions
        keys = top_keys(res)
        card = card_section(name)
        surfaces = {"__doc__": doc, "stub": stubs.get(name, ""), "card": card}
        rec["keys_unnamed"] = {}
        for sname, text in surfaces.items():
            toks = code_tokens(text) if sname == 'card' else doc_tokens(text)
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
                log(fh, f"[{name}] phantom candidates in {sname} (named in a returns sentence, not returned): {ph}")
        rec["phantom_candidates"] = phantom
        # (v) shapes
        got = shapes(res)
        got = {p.replace("$.", ""): s for p, s in got.items()}
        exp = expected_shapes(name, res, args, kwargs)
        bad = {}
        for key, want in exp.items():
            have = got.get(key)
            if have is None:
                v = res.get(key)
                have = () if not isinstance(v, (list, np.ndarray)) else tuple(np.asarray(v, dtype=float).shape)
            if tuple(have) != tuple(want):
                bad[key] = (list(have), list(want))
        rec["shape_mismatch"] = bad
        if bad:
            log(fh, f"[{name}] SHAPE MISMATCH (got vs documented): {bad}")
            n_cand += 1
        rec["shapes_checked"] = len(exp)
        rec["shapes"] = {k: list(v) for k, v in got.items()}
        log(fh, f"[{name}] {len(keys)} keys; summarize={rec.get('summarize_lines', 'FAIL')} lines; "
                f"json={rec['json']} pickle={rec['pickle']} nonfinite={len(nf)} shapes_checked={len(exp)}")
        report[name] = rec
    log(fh, f"\nREACHED {len(report)}/{len(NEW)}; candidates raised: {n_cand}")
    json.dump(report, open(os.path.join(OUT, "sweep_e.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
