"""Sweep S — the malformed-input seal for the 0.10.0 surface.

From each callable's canonical call, ONE argument at a time is replaced by a
hostile value: arrays -> all-NaN, one NaN, one inf, empty, one row, zero
columns, wrong rank (down and up), one row/column/entity short (so paired
arguments mismatch), a nested list, an int array, a bool array, transposed,
a duplicated row, a string, None, a scalar; ints -> 0, 1, 2, -1, 2^31, 2^47,
2^63, 1.5, True, "3", None, [1]; floats -> nan, +/-inf, -1, 0, 1e300, 1e-300,
an int, True, "abc", None, [0.5]; bools -> 2, 0, "yes", None, 1.5; strings ->
"abc", "", 3, None; integer lists -> [0], [-1], [2^31], [2^47], [2^63], [],
[1.5], [True], "abc", int array, float array, nested, None, 5, one short, one
long; float lists -> NaN, inf, negative, zero, 1e300, empty, a string, a
scalar, nested, None, an int list, one short, one long; the `mask=` arrays ->
all zero, half zero, 2, -1, NaN, inf, wrong shape, transposed, a string, a
scalar; `var_conditional_forecast`'s `conditions` -> all-free, empty, ragged,
NaN-only, a string, a scalar, too many columns; plus every Optional slot the
canonical call leaves at None, driven from a valid base value with the extra
keyword that makes it live (`also`).

Every cell runs in a child under a 4 GB virtual-memory cap and a deadline.
Outcomes: ok / refusal (ValueError, TypeError, OverflowError, with "names the
parameter" recorded) / exc / PANIC / CRASH / ALLOC-ABORT / HANG.

Run:  .venv/bin/python lab/audit/round14/sweep_s14_malformed.py [--only name]
Out:  lab/audit/round14/out/sweep_s14.txt (summary), sweep_s14_cells.json
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
from registry import MASKED, NEW14, build  # noqa: E402

SCOPE = NEW14 + MASKED

ARRAY_VARIANTS = [
    "nan_all", "nan_one", "inf_one", "empty", "one_row", "zero_cols", "rank_down", "rank_up",
    "short_rows", "short_cols", "short_mid", "nested_list", "int_array", "bool_array",
    "transposed", "dup_row",
]
ARRAY_VALUES = [("str", "abc"), ("none", None), ("scalar", 1.0)]
INT_VALUES = [("0", 0), ("1", 1), ("2", 2), ("neg1", -1), ("2^31", 2**31), ("2^47", 2**47),
              ("2^63", 2**63), ("1.5", 1.5), ("True", True), ("str", "3"), ("none", None), ("list", [1])]
FLOAT_VALUES = [("nan", "f:nan"), ("inf", "f:inf"), ("-inf", "f:-inf"), ("neg1", -1.0), ("zero", 0.0),
                ("1e300", 1e300), ("1e-300", 1e-300), ("int", 3), ("True", True), ("str", "abc"),
                ("none", None), ("list", [0.5])]
BOOL_VALUES = [("2", 2), ("0", 0), ("str", "yes"), ("none", None), ("1.5", 1.5)]
STR_VALUES = [("abc", "abc"), ("empty", ""), ("int", 3), ("none", None)]
ILIST_VALUES = [("[0]", [0]), ("[-1]", [-1]), ("[2^31]", [2**31]), ("[2^47]", [2**47]),
                ("[2^63]", [2**63]), ("[]", []), ("[1.5]", [1.5]), ("[True]", [True]), ("str", "abc"),
                ("int_array", {"ndarray": [1, 2], "dtype": "int64"}), ("float_array", {"ndarray": [1.0, 2.0]}),
                ("nested", [[1]]), ("none", None), ("5", 5)]
FLIST_VALUES = [("nan", ["f:nan"]), ("inf", ["f:inf"]), ("neg", [-1.0]), ("zero", [0.0]),
                ("1e300", [1e300]), ("empty", []), ("str", "abc"), ("scalar", 0.5),
                ("nested", [[0.5]]), ("none", None), ("int_list", [1]), ("2^63", [float(2**63)])]
BLIST_VALUES = [("[True]", [True]), ("[False]", [False]), ("[]", []), ("[2]", [2]),
                ("str", "abc"), ("scalar", True), ("none", None), ("one_long", [True, True])]

# Optional slots the canonical call leaves at None: (function, kw) -> family.
BASES = {
    ("unobserved_components", "exog"): "array2_T",
    ("unobserved_components", "forecast_exog"): "array2_h",
    ("unobserved_components", "fixed_params"): "flist",
    ("unobserved_components", "cycle_period_bounds"): "flist",
    ("unobserved_components", "freq_seasonal"): "flist",
    ("unobserved_components", "freq_seasonal_harmonics"): "ilist",
    ("unobserved_components", "stochastic_seasonal"): "bool",
    ("unobserved_components", "stochastic_freq_seasonal"): "blist",
    ("unobserved_components", "damped_cycle"): "bool",
    ("unobserved_components", "stochastic_cycle"): "bool",
    ("tvp_regression", "fixed_params"): "flist",
    ("ets_fit", "level"): "float",
    ("ets_fit", "n_sim"): "int",
    ("ets_fit", "seed"): "int",
    ("ets_fit", "optimizer"): "str",
    ("ets_fit", "max_iter"): "int",
    ("ets_fit", "smoothing_params"): "flist",
    ("ets_fit", "initial_states"): "flist",
    ("auto_ets", "level"): "float",
    ("auto_ets", "n_sim"): "int",
    ("auto_ets", "seed"): "int",
    ("auto_ets", "optimizer"): "str",
    ("auto_ets", "damped"): "bool",
    ("var_conditional_forecast", "steps"): "int",
    ("spa_test", "block_size"): "int",
    ("model_confidence_set", "block_size"): "int",
    ("stepm_test", "block_size"): "int",
    ("fmols", "bandwidth"): "float",
    ("fmols", "bandwidth_rule"): "str",
    ("fmols", "diff"): "bool",
    ("fmols", "x_trend"): "str",
    ("ccr", "bandwidth"): "float",
    ("ccr", "bandwidth_rule"): "str",
    ("ccr", "diff"): "bool",
    ("ccr", "x_trend"): "str",
    ("dols", "bandwidth"): "float",
    ("dols", "ic"): "str",
    ("dols", "common"): "bool",
    ("dols", "max_lag"): "int",
    ("dols", "max_lead"): "int",
    ("panel_fe@mask", "bandwidth"): "float",
    ("panel_lp@mask", "band"): "str",
    ("panel_lp@mask", "band_alpha"): "float",
}
# extra keywords that put the call in the mode where the mutated slot is live
ALSO = {
    ("unobserved_components", "forecast_exog"): {"exog": {"ndarray": np.ones((200, 2)).tolist()},
                                                 "forecast_steps": 3},
    ("unobserved_components", "cycle_period_bounds"): {"cycle": True},
    ("unobserved_components", "damped_cycle"): {"cycle": True},
    ("unobserved_components", "stochastic_cycle"): {"cycle": True},
    ("unobserved_components", "stochastic_freq_seasonal"): {"freq_seasonal": [12.0]},
    ("unobserved_components", "freq_seasonal_harmonics"): {"freq_seasonal": [12.0]},
    ("ets_fit", "smoothing_params"): {"initialization": "heuristic"},
    ("ets_fit", "initial_states"): {"initialization": "known"},
    ("ets_fit", "level"): {},
    ("ets_fit", "n_sim"): {"error": "mul", "seasonal": "mul"},
    ("ets_fit", "seed"): {"error": "mul", "seasonal": "mul"},
    ("auto_ets", "n_sim"): {},
    ("auto_ets", "seed"): {},
    ("fmols", "diff"): {"trend": "ct"},
    ("ccr", "diff"): {"trend": "ct"},
    ("fmols", "bandwidth_rule"): {},
    ("ccr", "bandwidth_rule"): {},
    ("dols", "ic"): {"lags": None, "leads": None},
    ("dols", "common"): {"lags": None, "leads": None},
    ("dols", "max_lag"): {"lags": None},
    ("dols", "max_lead"): {"leads": None},
    ("panel_fe@mask", "bandwidth"): {"se_type": "driscoll_kraay"},
}
# the observation mask: its own hostile family (shape, flag values, dtype)
MASK_VARIANTS = ["nan_all", "nan_one", "inf_one", "empty", "one_row", "zero_cols",
                 "rank_down", "rank_up", "short_rows", "short_cols", "transposed", "int_array",
                 "bool_array"]


def mask_values(n, t):
    ones = np.ones((n, t))
    half = ones.copy()
    half[:, : t // 2] = 0.0
    row_gone = ones.copy()
    row_gone[0] = 0.0
    return [
        ("all_zero", {"ndarray": np.zeros((n, t)).tolist()}),
        ("half_zero", {"ndarray": half.tolist()}),
        ("one_entity_gone", {"ndarray": row_gone.tolist()}),
        ("twos", {"ndarray": (ones * 2.0).tolist()}),
        ("negative", {"ndarray": (-ones).tolist()}),
        ("fractional", {"ndarray": (ones * 0.5).tolist()}),
        ("str", "abc"),
        ("scalar", 1.0),
        ("none", None),
    ]


# `conditions` is a nested list, not an array: its own family
def condition_values(steps, k):
    free = [[None] * k for _ in range(steps)]
    pinned = [[None] * k for _ in range(steps)]
    pinned[0][1] = 0.5
    return [
        ("all_free", free),
        ("empty", []),
        ("empty_rows", [[] for _ in range(steps)]),
        ("ragged", [[0.5, None], [None, None, None]]),
        ("too_wide", [[0.5] * (k + 1)]),
        ("nan_only", [["f:nan"] * k]),
        ("inf", [["f:inf"] + [None] * (k - 1)]),
        ("str_cell", [["x"] + [None] * (k - 1)]),
        ("str", "abc"),
        ("scalar", 0.5),
        ("none", None),
        ("flat", [0.5, None, None]),
        ("1e300", [[1e300] + [None] * (k - 1)]),
    ]


def enumerate_cells(name):  # noqa: C901
    base = name.split("@")[0]
    fn = getattr(tsecon._core, base)
    params = list(inspect.signature(fn).parameters.values())
    args, kwargs = build(name)
    cells = []

    def add(slot, pname, label, variant, deadline=45.0, also=None):
        c = {"id": f"{pname}:{label}", "slot": slot, "variant": variant, "param": pname,
             "deadline": deadline}
        extra = dict(ALSO.get((name, pname), {}))
        if also:
            extra.update(also)
        if extra:
            c["also"] = extra
        cells.append(c)

    def int_family(slot, pname):
        for lbl, v in INT_VALUES:
            add(slot, pname, lbl, ["value", v], 20.0 if "2^" in lbl else 45.0)

    def family(slot, pname, value):
        if pname == "mask":
            n, t = np.asarray(value).shape
            for v in MASK_VARIANTS:
                add(slot, pname, v, v)
            for lbl, v in mask_values(n, t):
                add(slot, pname, lbl, ["value", v])
            return
        if pname == "conditions":
            k = np.asarray(args[0]).shape[1]
            for lbl, v in condition_values(len(value), k):
                add(slot, pname, lbl, ["value", v])
            return
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            for v in ARRAY_VARIANTS:
                if v in ("zero_cols", "short_cols", "transposed") and value.ndim < 2:
                    continue
                if v == "short_mid" and value.ndim < 3:
                    continue
                add(slot, pname, v, v)
            for lbl, v in ARRAY_VALUES:
                add(slot, pname, lbl, ["value", v])
        elif isinstance(value, bool):
            for lbl, v in BOOL_VALUES:
                add(slot, pname, lbl, ["value", v])
        elif isinstance(value, (int, np.integer)):
            int_family(slot, pname)
        elif isinstance(value, float):
            for lbl, v in FLOAT_VALUES:
                add(slot, pname, lbl, ["value", v])
        elif isinstance(value, str):
            for lbl, v in STR_VALUES:
                add(slot, pname, lbl, ["value", v])
        elif isinstance(value, (list, tuple)) and value and all(
                isinstance(e, (int, np.integer)) and not isinstance(e, bool) for e in value):
            for lbl, v in ILIST_VALUES:
                add(slot, pname, lbl, ["value", v], 20.0 if "2^" in lbl else 45.0)
            add(slot, pname, "one_short", ["value", list(value)[:-1]])
            add(slot, pname, "one_long", ["value", list(value) + [int(max(value)) + 1]])
        elif isinstance(value, list) and value and all(isinstance(e, float) for e in value):
            for lbl, v in FLIST_VALUES:
                add(slot, pname, lbl, ["value", v])
            add(slot, pname, "one_short", ["value", list(value)[:-1]])
            add(slot, pname, "one_long", ["value", list(value) + [list(value)[0]]])

    def optional_family(pname, kind):
        slot = ["kw", pname]
        if kind == "int":
            int_family(slot, pname)
        elif kind == "float":
            for lbl, v in FLOAT_VALUES:
                add(slot, pname, lbl, ["value", v])
        elif kind == "bool":
            for lbl, v in BOOL_VALUES:
                add(slot, pname, lbl, ["value", v])
            add(slot, pname, "True", ["value", True])
        elif kind == "str":
            for lbl, v in STR_VALUES:
                add(slot, pname, lbl, ["value", v])
        elif kind == "ilist":
            for lbl, v in ILIST_VALUES:
                add(slot, pname, lbl, ["value", v], 20.0 if "2^" in lbl else 45.0)
        elif kind == "flist":
            for lbl, v in FLIST_VALUES:
                add(slot, pname, lbl, ["value", v])
        elif kind == "blist":
            for lbl, v in BLIST_VALUES:
                add(slot, pname, lbl, ["value", v])
        elif kind in ("array2_T", "array2_h"):
            rows = 200 if kind == "array2_T" else 3
            good = np.ones((rows, 2))
            for lbl, v in [
                ("valid", good), ("nan_all", np.full_like(good, np.nan)),
                ("inf_one", np.where(np.arange(good.size).reshape(good.shape) == 0, np.inf, good)),
                ("empty", np.empty((0, 2))), ("short_rows", good[:-1]), ("zero_cols", np.empty((rows, 0))),
                ("rank_down", good[:, 0]), ("rank_up", good[None]), ("transposed", good.T),
            ]:
                add(slot, pname, lbl, ["value", {"ndarray": np.asarray(v).tolist()}])
            for lbl, v in [("nested_list", good.tolist()), ("str", "abc"), ("scalar", 1.0), ("none", None)]:
                add(slot, pname, lbl, ["value", v])

    for i, p in enumerate(params):
        if i < len(args):
            family(["pos", i], p.name, args[i])
        elif p.name in kwargs:
            family(["kw", p.name], p.name, kwargs[p.name])
        else:
            kind = BASES.get((name, p.name))
            if kind is not None:
                optional_family(p.name, kind)
            elif isinstance(p.default, bool):
                for lbl, v in BOOL_VALUES:
                    add(["kw", p.name], p.name, lbl, ["value", v])
            elif isinstance(p.default, (int, np.integer)):
                int_family(["kw", p.name], p.name)
            elif isinstance(p.default, float):
                for lbl, v in FLOAT_VALUES:
                    add(["kw", p.name], p.name, lbl, ["value", v])
            elif isinstance(p.default, str):
                for lbl, v in STR_VALUES:
                    add(["kw", p.name], p.name, lbl, ["value", v])
    return cells


def names_param(msg, pname):
    return bool(re.search(rf"(?<![\w.]){re.escape(pname)}(?![\w])", msg or ""))


def main():
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(SCOPE)
    fh = open(os.path.join(OUT, "sweep_s14.txt"), "w")
    plan = {n: enumerate_cells(n) for n in only}
    total = sum(len(v) for v in plan.values())
    log(fh, f"{len(plan)} callables, {total} cells, 4 GB cap per child")
    for n in plan:
        log(fh, f"  {n}: {len(plan[n])} cells")

    def job(n):
        muts = []
        for c in plan[n]:
            m = {"id": c["id"], "slot": c["slot"], "variant": c["variant"], "deadline": c["deadline"]}
            if c.get("also"):
                m["also"] = c["also"]
            muts.append(m)
        return n, run_cells(n, muts, rlimit_gb=4.0, deadline=45.0)

    all_recs = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        for n, recs in ex.map(job, list(plan)):
            all_recs[n] = recs
    tally = {}
    offenders, unnamed, exc_cells, ok_cells, harness = [], [], [], [], []
    for n, recs in all_recs.items():
        pmap = {c["id"]: c["param"] for c in plan[n]}
        for r in recs:
            o = r["outcome"]
            tally[o] = tally.get(o, 0) + 1
            if o in ("PANIC", "CRASH", "ALLOC-ABORT", "CRASH-CAPACITY-OVERFLOW", "HANG", "memerr"):
                offenders.append((n, r["id"], o, r.get("msg") or r.get("detail") or r.get("stderr_tail", "")))
            elif o == "harness-error":
                harness.append((n, r["id"], r.get("msg")))
            elif o == "exc":
                exc_cells.append((n, r["id"], r.get("exc"), r.get("msg")))
            elif o == "refusal":
                if not names_param(r.get("msg", ""), pmap[r["id"]]):
                    unnamed.append((n, r["id"], r.get("exc"), (r.get("msg") or "")[:200]))
            elif o == "ok":
                ok_cells.append((n, r["id"], r.get("nonfinite", 0), r.get("seconds")))
    log(fh, f"\ncells: {sum(tally.values())}; outcomes: {tally}")
    log(fh, f"\n== offenders (panic/crash/abort/hang/memerr): {len(offenders)}")
    for o in offenders:
        log(fh, f"  {o[0]} {o[1]}: {o[2]} {str(o[3])[:200]}")
    log(fh, f"\n== harness errors: {len(harness)}")
    for h in harness:
        log(fh, f"  {h[0]} {h[1]}: {h[2]}")
    log(fh, f"\n== other exceptions: {len(exc_cells)}")
    for e in exc_cells:
        log(fh, f"  {e[0]} {e[1]}: {e[2]}: {str(e[3])[:160]}")
    log(fh, f"\n== refusals whose message does not name the parameter: {len(unnamed)}")
    for u in unnamed:
        log(fh, f"  {u[0]} {u[1]}: {u[2]}: {u[3]}")
    log(fh, f"\n== normal returns on a corrupted argument: {len(ok_cells)} (nonfinite count, seconds)")
    for c in ok_cells:
        log(fh, f"  {c[0]} {c[1]}: nonfinite={c[2]} t={c[3]}")
    json.dump(all_recs, open(os.path.join(OUT, "sweep_s14_cells.json"), "w"), indent=0, default=str)


if __name__ == "__main__":
    main()
