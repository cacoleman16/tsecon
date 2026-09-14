"""Sweep S — the malformed-input seal, for every parameter of the six.

From each callable's canonical call, ONE argument at a time is replaced by a
hostile value: arrays -> all-NaN, one NaN, one inf, empty, one row, zero
columns, wrong rank (down and up), one row/column/entity short (so paired
arguments mismatch), a nested list, an int array, a bool array, transposed,
a duplicated row, a string, None, a scalar; ints -> 0, 1, 2, -1, 2^31, 2^47,
2^63, 1.5, True, "3", None, [1]; floats -> nan, ±inf, -1, 0, 1e300, 1e-300,
an int, True, "abc", None, [0.5]; bools -> 2, 0, "yes", None, 1.5; strings ->
"abc", "", 3, None; integer lists -> [0], [-1], [2^31], [2^47], [2^63], [],
[1.5], [True], "abc", int array, float array, nested, None, 5, one short, one
long; the (lower, upper) tuple -> reversed, (0, 1), (-1, 2), (nan, .5), one
element, three elements, a list, "abc", 3, None; plus the Optional slots the
canonical call leaves at None (w, histories, bandwidth, ...), driven from a
valid base value. Every cell runs in a child under a 4 GB virtual-memory cap
and a deadline. Outcomes: ok / refusal (ValueError, TypeError, OverflowError,
with "names the parameter" recorded) / exc / PANIC / CRASH / ALLOC-ABORT /
HANG.

Run:  .venv/bin/python lab/audit/round14/sweep_s_malformed.py [--only name]
Out:  lab/audit/round14/out/sweep_s.txt (summary), sweep_s_cells.json
"""
from __future__ import annotations

import inspect
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import OUT, log  # noqa: E402
from parent import run_cells  # noqa: E402
from registry import NEW, build  # noqa: E402

ARRAY_VARIANTS = [
    "nan_all", "nan_one", "inf_one", "empty", "one_row", "zero_cols", "rank_down", "rank_up",
    "short_rows", "short_cols", "short_mid", "nested_list", "int_array", "bool_array", "transposed", "dup_row",
]
ARRAY_VALUES = [("str", "abc"), ("none", None), ("scalar", 1.0)]
INT_VALUES = [("0", 0), ("1", 1), ("2", 2), ("neg1", -1), ("2^31", 2**31), ("2^47", 2**47), ("2^63", 2**63),
              ("1.5", 1.5), ("True", True), ("str", "3"), ("none", None), ("list", [1])]
FLOAT_VALUES = [("nan", "f:nan"), ("inf", "f:inf"), ("-inf", "f:-inf"), ("neg1", -1.0), ("zero", 0.0),
                ("1e300", 1e300), ("1e-300", 1e-300), ("int", 3), ("True", True), ("str", "abc"),
                ("none", None), ("list", [0.5])]
BOOL_VALUES = [("2", 2), ("0", 0), ("str", "yes"), ("none", None), ("1.5", 1.5)]
STR_VALUES = [("abc", "abc"), ("empty", ""), ("int", 3), ("none", None)]
ILIST_VALUES = [("[0]", [0]), ("[-1]", [-1]), ("[2^31]", [2**31]), ("[2^47]", [2**47]), ("[2^63]", [2**63]),
                ("[]", []), ("[1.5]", [1.5]), ("[True]", [True]), ("str", "abc"),
                ("int_array", {"ndarray": [1, 2], "dtype": "int64"}), ("float_array", {"ndarray": [1.0, 2.0]}),
                ("nested", [[1]]), ("none", None), ("5", 5)]
TUPLE_VALUES = [("reversed", {"tuple": [0.84, 0.16]}), ("(0,1)", {"tuple": [0.0, 1.0]}), ("(-1,2)", {"tuple": [-1.0, 2.0]}),
                ("(nan,.5)", {"tuple": ["f:nan", 0.5]}), ("one", {"tuple": [0.5]}), ("three", {"tuple": [0.1, 0.5, 0.9]}),
                ("list", [0.16, 0.84]), ("str", "abc"), ("3", 3), ("none", None)]

# Optional slots the canonical call leaves unset: (function, kw) -> base value
# and the family of variants to drive from it.
BASES = {
    ("var_girf", "histories"): ("int", 10),
    ("threshold_var_girf", "histories"): ("int", 10),
    ("threshold_var_girf", "delays"): ("ilist", [1, 2]),
    ("setar_threshold_ci", "delays"): ("ilist", [1, 2]),
    ("setar_threshold_ci", "het_robust"): ("bool", True),
    ("jsz_fit", "w"): ("array", None),  # built below (orthonormal 3 x 12)
    ("panel_distributed_lag", "bandwidth"): ("float_dk", 4.0),
    ("panel_distributed_lag", "entity_trends"): ("bool", True),
}
EXTRA_INT_LISTS = {"setar_threshold_ci": ["delays"], "threshold_var_girf": ["delays"],
                   "jsz_fit": ["maturities"], "jsz_loadings": ["maturities"]}


def w_base():
    raw = np.array([np.ones(12), np.arange(1, 13) / 12.0, np.arange(1, 13) / 4.0 * np.exp(-np.arange(1, 13) / 4.0)])
    q, _ = np.linalg.qr(raw.T)
    return {"ndarray": q.T.tolist()}


def enumerate_cells(name):
    fn = getattr(tsecon._core, name)
    params = list(inspect.signature(fn).parameters.values())
    args, kwargs = build(name)
    cells = []

    def add(slot, pname, label, variant, deadline=30.0):
        cells.append({"id": f"{pname}:{label}", "slot": slot, "variant": variant, "param": pname, "deadline": deadline})

    def family(slot, pname, value):
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            for v in ARRAY_VARIANTS:
                if v == "zero_cols" and value.ndim < 2:
                    continue
                if v in ("short_cols",) and value.ndim < 2:
                    continue
                if v == "short_mid" and value.ndim < 3:
                    continue
                if v == "transposed" and value.ndim < 2:
                    continue
                add(slot, pname, v, v)
            for lbl, v in ARRAY_VALUES:
                add(slot, pname, lbl, ["value", v])
        elif isinstance(value, bool):
            for lbl, v in BOOL_VALUES:
                add(slot, pname, lbl, ["value", v])
        elif isinstance(value, (int, np.integer)):
            for lbl, v in INT_VALUES:
                add(slot, pname, lbl, ["value", v], 15.0 if lbl in ("2^31", "2^47", "2^63") else 30.0)
        elif isinstance(value, float):
            for lbl, v in FLOAT_VALUES:
                add(slot, pname, lbl, ["value", v])
        elif isinstance(value, str):
            for lbl, v in STR_VALUES:
                add(slot, pname, lbl, ["value", v])
        elif isinstance(value, (list, tuple)) and value and all(isinstance(e, (int, np.integer)) for e in value):
            for lbl, v in ILIST_VALUES:
                add(slot, pname, lbl, ["value", v], 15.0 if "2^" in lbl else 30.0)
            add(slot, pname, "one_short", ["value", list(value)[:-1]])
            add(slot, pname, "one_long", ["value", list(value) + [int(max(value)) + 1]])
        elif isinstance(value, list) and value and all(isinstance(e, float) for e in value):  # eval_points
            for v in ("nan_all", "empty", "rank_up"):
                add(slot, pname, v, v)
            for lbl, v in [("str", "abc"), ("int_list", [10, 20]), ("scalar", 20.0), ("2d", [[10.0, 20.0]])]:
                add(slot, pname, lbl, ["value", v])

    for i, p in enumerate(params):
        if i < len(args):
            family(["pos", i], p.name, args[i])
        elif p.name in kwargs:
            family(["kw", p.name], p.name, kwargs[p.name])
        else:
            d = p.default
            base = BASES.get((name, p.name))
            if base is not None:
                kind, v = base
                if kind == "int":
                    for lbl, val in INT_VALUES:
                        add(["kw", p.name], p.name, lbl, ["value", val], 15.0 if "2^" in lbl else 30.0)
                elif kind == "ilist":
                    for lbl, val in ILIST_VALUES:
                        add(["kw", p.name], p.name, lbl, ["value", val], 15.0 if "2^" in lbl else 30.0)
                elif kind == "bool":
                    for lbl, val in BOOL_VALUES:
                        add(["kw", p.name], p.name, lbl, ["value", val])
                elif kind == "array":
                    w = w_base()
                    W = np.asarray(w["ndarray"])
                    variants = {
                        "nan_all": np.full_like(W, np.nan), "rows_short": W[:2], "cols_short": W[:, :-1],
                        "transposed": W.T, "rank_deficient": np.vstack([W[:2], W[0]]), "rank_down": W[0],
                        "zeros": np.zeros_like(W), "nested_list": None, "str": None, "int_array": None,
                    }
                    for lbl, val in variants.items():
                        if lbl == "nested_list":
                            add(["kw", p.name], p.name, lbl, ["value", W.tolist()])
                        elif lbl == "str":
                            add(["kw", p.name], p.name, lbl, ["value", "abc"])
                        elif lbl == "int_array":
                            add(["kw", p.name], p.name, lbl, ["value", {"ndarray": np.rint(W * 3).astype(int).tolist(), "dtype": "int64"}])
                        else:
                            add(["kw", p.name], p.name, lbl, ["value", {"ndarray": np.asarray(val).tolist()}])
                    add(["kw", p.name], p.name, "valid", ["value", w])
                elif kind == "float_dk":
                    for lbl, val in FLOAT_VALUES:
                        cells.append({"id": f"{p.name}:{lbl}@dk", "slot": ["kw", p.name], "variant": ["value", val],
                                      "param": p.name, "deadline": 30.0, "also": {"se_type": "driscoll_kraay"}})
            elif isinstance(d, bool):
                for lbl, val in BOOL_VALUES:
                    add(["kw", p.name], p.name, lbl, ["value", val])
            elif isinstance(d, (int, np.integer)):
                for lbl, val in INT_VALUES:
                    add(["kw", p.name], p.name, lbl, ["value", val], 15.0 if "2^" in lbl else 30.0)
            elif isinstance(d, float):
                for lbl, val in FLOAT_VALUES:
                    add(["kw", p.name], p.name, lbl, ["value", val])
            elif isinstance(d, str):
                for lbl, val in STR_VALUES:
                    add(["kw", p.name], p.name, lbl, ["value", val])
            elif isinstance(d, tuple):
                for lbl, val in TUPLE_VALUES:
                    add(["kw", p.name], p.name, lbl, ["value", val])
            elif d is None and p.name in ("slope_level", "slope_region_level", "null_threshold"):
                for lbl, val in FLOAT_VALUES:
                    extra = {"slope_level": 0.95} if p.name == "slope_region_level" else {}
                    cells.append({"id": f"{p.name}:{lbl}", "slot": ["kw", p.name], "variant": ["value", val],
                                  "param": p.name, "deadline": 30.0, "also": extra})
    return cells


def names_param(msg, pname):
    return bool(re.search(rf"(?<![\w.]){re.escape(pname)}(?![\w])", msg or ""))


def main():
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(NEW)
    fh = open(os.path.join(OUT, "sweep_s.txt"), "w")
    plan = {n: enumerate_cells(n) for n in only}
    total = sum(len(v) for v in plan.values())
    log(fh, f"{len(plan)} callables, {total} cells, 4 GB cap per child")

    def job(n):
        muts = []
        for c in plan[n]:
            m = {"id": c["id"], "slot": c["slot"], "variant": c["variant"], "deadline": c["deadline"]}
            if c.get("also"):
                m["also"] = c["also"]
            muts.append(m)
        return n, run_cells(n, muts, rlimit_gb=4.0)

    all_recs = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        for n, recs in ex.map(job, list(plan)):
            all_recs[n] = recs
    tally = {}
    offenders, unnamed, exc_cells, ok_cells = [], [], [], []
    for n, recs in all_recs.items():
        pmap = {c["id"]: c["param"] for c in plan[n]}
        for r in recs:
            o = r["outcome"]
            tally[o] = tally.get(o, 0) + 1
            if o in ("PANIC", "CRASH", "ALLOC-ABORT", "CRASH-CAPACITY-OVERFLOW", "HANG", "memerr"):
                offenders.append((n, r["id"], o, r.get("msg") or r.get("detail") or r.get("stderr_tail", "")))
            elif o == "exc":
                exc_cells.append((n, r["id"], r.get("exc"), r.get("msg")))
            elif o == "refusal":
                if not names_param(r.get("msg", ""), pmap[r["id"]]):
                    unnamed.append((n, r["id"], r.get("exc"), (r.get("msg") or "")[:110]))
            elif o == "ok":
                ok_cells.append((n, r["id"], r.get("nonfinite", 0), r.get("seconds")))
    log(fh, f"\ncells: {sum(tally.values())}; outcomes: {tally}")
    log(fh, f"\n== offenders (panic/crash/abort/hang/memerr): {len(offenders)}")
    for o in offenders:
        log(fh, f"  {o[0]} {o[1]}: {o[2]} {str(o[3])[:160]}")
    log(fh, f"\n== other exceptions: {len(exc_cells)}")
    for e in exc_cells:
        log(fh, f"  {e[0]} {e[1]}: {e[2]}: {str(e[3])[:140]}")
    log(fh, f"\n== refusals whose message does not name the parameter: {len(unnamed)}")
    for u in unnamed:
        log(fh, f"  {u[0]} {u[1]}: {u[2]}: {u[3]}")
    log(fh, f"\n== normal returns on a corrupted argument: {len(ok_cells)} (nonfinite count, seconds)")
    for c in ok_cells:
        log(fh, f"  {c[0]} {c[1]}: nonfinite={c[2]} t={c[3]}")
    json.dump(all_recs, open(os.path.join(OUT, "sweep_s_cells.json"), "w"), indent=0, default=str)


if __name__ == "__main__":
    main()
