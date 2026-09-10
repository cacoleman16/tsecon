"""Sweep C — claims vs reality for the six 0.9.0 callables.

Every number quoted in the six model-card sections, the validation-matrix
rows, the guide paragraphs (13, 14, 15, 16) and docs/reference/speed.md must
be reproduced by the committed test or script it cites. This driver re-runs
them and diffs:

  (a) the binding tests that pin printed numbers (guide 13 / 14 blocks, the
      var-svar card example, the GSW illustration, Hansen's Table 1, the
      worst linearmodels error, the showcase timing) — pytest, -s;
  (b) the term-structure card's runnable JSZ example against its "Expected
      output" block, and the cointegration card's setar_threshold_ci example
      (no expected block: must run);
  (c) benchmarks/bench.py --json into the scratchpad, the parity rows diffed
      against the committed benchmarks/results/latest.json (machine-
      independent), the "faster on N/25" and per-call-overhead sentences of
      speed.md re-derived from the committed JSON, render_dashboard --check;
  (d) docs/examples/lp_vs_var_head_to_head.py (~2 min) against the verbatim
      block of guide chapter 16;
  (e) the Rust property-test output (out/rust_props.txt, run separately with
      --nocapture) against the Monte-Carlo numbers the cards and matrix rows
      quote;
  (f) the DJO script is marked network-dependent (not run).

Run:  .venv/bin/python lab/audit/round13/sweep_c_claims.py [--skip-lp] [--skip-bench]
Out:  lab/audit/round13/out/sweep_c.txt (+ sweep_c_*.txt raw outputs)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CARDS, OUT, REPO, log  # noqa: E402

PY = sys.executable
SCRATCH = os.environ.get("R13_SCRATCH", os.path.join(OUT, "scratch"))
os.makedirs(SCRATCH, exist_ok=True)


def run(cmd, timeout=1800, cwd=REPO, env=None):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
    return p.returncode, p.stdout + p.stderr, time.time() - t0


def code_blocks(text, lang="python"):
    return re.findall(rf"```{lang}\n(.*?)```", text, re.S)


def main(argv):
    fh = open(os.path.join(OUT, "sweep_c.txt"), "w")
    # (a) pinned tests
    tests = [
        "bindings/python/tests/test_girf.py::test_guide_chapter_13_example_numbers",
        "bindings/python/tests/test_girf.py::test_var_svar_card_example_prints_what_it_says",
        "bindings/python/tests/test_girf.py::test_showcase_configuration_timing_is_reasonable",
        "bindings/python/tests/test_panel_dl.py::test_guide_chapter_example_numbers",
        "bindings/python/tests/test_panel_dl.py::test_worst_relative_error_against_linearmodels_is_reported",
        "bindings/python/tests/test_jsz.py::test_jsz_fit_gsw_reproduces_the_fixture_and_the_illustration",
        "bindings/python/tests/test_jsz.py::test_jsz_afns_case_spans_nelson_siegel_and_converges_to_afns_adjustment",
        "bindings/python/tests/test_jsz.py::test_jsz_fit_sim_mle_agrees_with_scipy_and_recovers_the_truth",
        "bindings/python/tests/test_setar_ci.py::test_hansen_2000_table_1_through_lr_crit",
        "bindings/python/tests/test_setar_ci.py::test_fixture_exercises_a_disjoint_set_and_the_correction",
    ]
    rc, out, dt = run([PY, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", *tests])
    open(os.path.join(OUT, "sweep_c_pytest.txt"), "w").write(out)
    log(fh, f"(a) pinned binding tests: rc={rc} in {dt:.0f}s — {out.strip().splitlines()[-1]}")
    for line in out.splitlines():
        if "showcase" in line or "worst" in line.lower():
            log(fh, "    " + line.strip())
    # (b) card examples
    ts = open(os.path.join(CARDS, "term-structure.md"), encoding="utf-8").read()
    jsz_block = [b for b in code_blocks(ts) if "jsz_fit(" in b][-1]
    expected = re.search(r"Expected output:\n\n```\n(.*?)```", ts[ts.index("### JSZ canonical"):], re.S).group(1)
    rc, out, dt = run([PY, "-c", jsz_block])
    got = out.strip()
    same = got.splitlines() == expected.strip().splitlines()
    log(fh, f"(b) term-structure card JSZ example: rc={rc} in {dt:.1f}s; printed == expected block: {same}")
    if not same:
        log(fh, "    expected:\n" + expected + "    got:\n" + got)
    cr = open(os.path.join(CARDS, "cointegration-regime.md"), encoding="utf-8").read()
    ci_block = [b for b in code_blocks(cr) if "setar_threshold_ci(" in b][0]
    rc, out, dt = run([PY, "-c", ci_block])
    log(fh, f"(b) cointegration card setar_threshold_ci example: rc={rc} in {dt:.1f}s; output:\n" + "\n".join("      " + l for l in out.strip().splitlines()))
    vs = open(os.path.join(CARDS, "var-svar.md"), encoding="utf-8").read()
    vg_block = [b for b in code_blocks(vs) if "var_girf(" in b][0]
    rc, out, dt = run([PY, "-c", vg_block])
    log(fh, f"(b) var-svar card var_girf example: rc={rc}; printed {out.strip().splitlines()} (card comments: True / (9, 3) 398)")
    # (c) bench + dashboard
    if "--skip-bench" not in argv:
        jpath = os.path.join(SCRATCH, "latest.json")
        rc, out, dt = run([PY, "benchmarks/bench.py", "--json", jpath], timeout=3600)
        open(os.path.join(OUT, "sweep_c_bench.txt"), "w").write(out)
        log(fh, f"(c) bench.py --json: rc={rc} in {dt:.0f}s — {[l for l in out.splitlines() if 'RESULT' in l]}")
        new = json.load(open(jpath)); old = json.load(open(os.path.join(REPO, "benchmarks", "results", "latest.json")))
        def parity(d):
            rows = {}
            for op in d.get("operations", d.get("parity", [])):
                pass
            return rows
        # compare parity rows generically: walk both JSONs for max_abs_diff/tol/pass triples
        def triples(d, path="$"):
            if isinstance(d, dict):
                if "max_abs_diff" in d and "tol" in d:
                    yield path, d
                for k, v in d.items():
                    yield from triples(v, f"{path}.{k}")
            elif isinstance(d, list):
                for i, v in enumerate(d):
                    yield from triples(v, f"{path}[{i}]")
        o = {p: t for p, t in triples(old)}; n = {p: t for p, t in triples(new)}
        n_rows = len(o); n_same = 0; diffs = []
        for p, t in o.items():
            t2 = n.get(p)
            if t2 is None:
                diffs.append((p, "missing in re-run")); continue
            if t.get("passed", t.get("ok")) == t2.get("passed", t2.get("ok")) and t["tol"] == t2["tol"]:
                n_same += 1
            ratio = (t2["max_abs_diff"] or 0) / (t["max_abs_diff"] or 1e-300) if t["max_abs_diff"] else None
            if t["max_abs_diff"] != t2["max_abs_diff"]:
                diffs.append((p, f"max_abs_diff {t['max_abs_diff']:.3g} -> {t2['max_abs_diff']:.3g}"))
        log(fh, f"(c) parity rows: {n_rows} committed, {n_same} with identical pass/tol in the re-run; max_abs_diff changed on {len([d for d in diffs if 'max_abs_diff' in d[1]])} rows")
        for d in diffs[:12]:
            log(fh, f"    {d[0]}: {d[1]}")
        # speed.md sentences vs committed json
        speed = open(os.path.join(REPO, "docs", "reference", "speed.md"), encoding="utf-8").read()
        def faster_count(d):
            k = 0; tot = 0
            for row in d.get("timings", []):
                tot += 1; k += 1 if row.get("ratio", 0) > 1 else 0
            return k, tot
        said_a = re.findall(r"faster on \*\*(\d+ of \d+)\*\*", speed)
        said_b = re.findall(r"Faster than the reference \| (\d+/\d+)", speed)
        log(fh, f"(c) committed json: faster on {faster_count(old)}; re-run: {faster_count(new)}; speed.md says: {said_a} / {said_b}")
        ov = re.findall(r"took ([\d.]+) ms against ([\d.]+) ms.*?~([\d.]+) ms", speed)
        log(fh, f"(c) speed.md per-call overhead sentence: {ov}; json overhead: {old.get('overhead') or old.get('call_overhead')}; re-run: {new.get('overhead') or new.get('call_overhead')}")
        rc, out, dt = run([PY, "benchmarks/render_dashboard.py", "--check"])
        log(fh, f"(c) render_dashboard.py --check: rc={rc} {out.strip()[-160:]}")
    # (d) LP vs VAR
    if "--skip-lp" not in argv:
        rc, out, dt = run([PY, "docs/examples/lp_vs_var_head_to_head.py"], timeout=3600)
        open(os.path.join(OUT, "sweep_c_lpvar.txt"), "w").write(out)
        ch = open(os.path.join(REPO, "docs", "guide", "16-lp-vs-var-head-to-head.md"), encoding="utf-8").read()
        verb = ch[ch.index("## Provenance and the verbatim output"):]
        block = code_blocks(verb, "text") or code_blocks(verb, "")
        block = block[0] if block else ""
        def numbers(s):
            return re.findall(r"[-+]?\d+\.\d+(?:e[-+]?\d+)?", s)
        got_lines = [l.rstrip() for l in out.splitlines()]
        exp_lines = [l.rstrip() for l in block.splitlines()]
        missing = [l for l in exp_lines if l.strip() and l not in got_lines and not re.search(r"timestamp|wall-clock|per replication|tsecon\s*:|build|elapsed|seconds|python|platform|cpu|numpy", l, re.I)]
        log(fh, f"(d) lp_vs_var_head_to_head.py: rc={rc} in {dt:.0f}s; verbatim block {len(exp_lines)} lines, {len(missing)} content lines not reproduced verbatim")
        for l in missing[:15]:
            log(fh, "    MISSING: " + l)
        ver = re.findall(r"tsecon\s*:\s*([\d.]+)", out)
        said = re.findall(r"tsecon ([\d.]+) on a", ch)
        log(fh, f"    script banner version {ver}; chapter says: {said}")
    # (e) rust property outputs vs quoted numbers
    rp = os.path.join(OUT, "rust_props.txt")
    if os.path.exists(rp):
        txt = open(rp).read()
        results = re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed", txt)
        log(fh, f"(e) rust_props.txt present: {len(txt.splitlines())} lines; test results: {results}")
        quoted = {
            "panel cluster N=50 0.928": r"cluster N=50 .*coverage 0\.928", "panel cluster N=200 0.938": r"cluster N=200.*coverage 0\.938",
            "panel dk T=50 0.886": r"dk N=25 T=50 .*coverage 0\.886", "panel dk T=200 0.924": r"dk N=25 T=200.*coverage 0\.924",
            "panel mean se 0.0642/0.0327/0.0611/0.0335": r"mean se 0\.0642[\s\S]*mean se 0\.0327[\s\S]*mean se 0\.0611[\s\S]*mean se 0\.0335",
            "setar_ci 0.886/0.930 (n=100, d=0.5)": r"0\.886[\s\S]{0,40}0\.930", "setar_ci 0.946/0.976": r"0\.946[\s\S]{0,40}0\.976",
            "setar_ci setar2 0.966/0.984": r"0\.966[\s\S]{0,40}0\.984", "setar_ci eta2 468/500 mean 0.617": r"468[\s\S]{0,120}0\.617",
            "setar_ci corrected 0.915/0.938": r"0\.915[\s\S]{0,40}0\.938",
            "tvar sign asymmetry 0.149 t=37": r"0\.149[\s\S]{0,60}37\.", "tvar size 0.040 t=11.5": r"0\.040[\s\S]{0,60}11\.5",
            "tvar regime 0.111 t=60": r"0\.111[\s\S]{0,60}60\.", "tvar ratio 4.02": r"4\.02", "tvar antithetic 1.000": r"1\.000",
            "var girf ratio 4.58": r"4\.58", "var girf antithetic 0.953": r"0\.953",
            "jsz gaps 9.80e-6 / ratios 0.2504 0.2501": r"9\.8\d?e-0?6[\s\S]{0,200}0\.250",
        }
        for label, pat in quoted.items():
            log(fh, f"    {'OK ' if re.search(pat, txt) else 'NOT FOUND'}: {label}")
    else:
        log(fh, "(e) rust_props.txt not present — run the cargo property tests first")
    # (f) DJO
    djo = open(os.path.join(REPO, "docs", "examples", "panel_distributed_lag_djo.py"), encoding="utf-8").read()
    log(fh, f"(f) DJO script: urllib download present={'urllib' in djo}; header states the network fetch and the temp-dir cache: "
            f"{'unreachable' in djo and 'temporary directory' in djo}; card says fetched from a mirror: "
            f"{'course mirror' in open(os.path.join(CARDS, 'panel.md'), encoding='utf-8').read()}")


if __name__ == "__main__":
    main(sys.argv[1:])
