#!/usr/bin/env python
"""Adversarial-input matrix over every public tsecon callable (repo audit, security).

For each of the 173 callables, start from its canonical valid call
(``registry_ml.build``) and corrupt ONE argument at a time:

* every integer argument (positional, keyword, or a signature default not in
  the canonical call): 0, 1, 2, -1, 2**31, 2**63, 2**64;
* every float argument: nan, inf, -1.0, 0.0, and the string "abc";
* every float array argument: all-NaN, one NaN, one inf, empty, a single
  row (1 or 1x1), zero columns (2-D only), and the string "abc";
* every list-of-arrays (ragged panel): empty list, one empty unit, all-NaN;
* every integer list (periods, groups, delays, maturities): [0], [-1], [2**31], [];
* the whole call rebuilt at T = 10**5 (the allocation probe: an O(T^2)
  working set at 10**5 is 80 GB, which the memory cap turns into an instant
  abort; T = 10**6 was dropped after a first pass — it only added wall time).

Every cell runs in a CHILD process with a hard virtual-memory cap
(``--rlimit-gb``, default 6) and a per-cell deadline, so the four outcomes
that matter are attributed to the exact cell that produced them:

* ``PANIC``       — pyo3 ``PanicException`` reached Python (uncatchable by
                    ``except Exception``);
* ``CRASH``       — the child died (signal or abort); an abort whose stderr
                    says ``memory allocation of N bytes failed`` is reported
                    as ``ALLOC-ABORT`` with N — an unbounded allocation attempt;
* ``HANG``        — the cell exceeded its deadline (15 s for the huge-integer
                    cells, 45 s otherwise);
* ``refusal``     — ValueError / TypeError / OverflowError (the teaching path).

Output: ``out/sweep_adversarial.jsonl`` (one record per cell) and
``out/sweep_adversarial.md`` (the summary table plus every non-refusal cell).

    .venv/bin/python sweep_adversarial.py                # full run (~1-2 h, 4 workers)
    .venv/bin/python sweep_adversarial.py --only adf,acf # a subset
    .venv/bin/python sweep_adversarial.py --skip-big     # no T=1e5 rebuilds
    .venv/bin/python sweep_adversarial.py --resume       # skip callables already in the JSONL
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import queue
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out")

INT_VALUES = [("0", 0), ("1", 1), ("2", 2), ("neg1", -1), ("2^31", 2**31), ("2^63", 2**63), ("2^64", 2**64)]
HUGE_INT = {"2^31", "2^63"}
FLOAT_VALUES = [("nan", float("nan")), ("inf", float("inf")), ("neg1", -1.0), ("zero", 0.0), ("str", "abc")]
ARRAY_VARIANTS = ["nan_all", "nan_one", "inf_one", "empty", "one", "zero_cols", "str"]
PANEL_VARIANTS = ["empty", "one_empty", "nan"]
ILIST_VARIANTS = [("zero", [0]), ("neg1", [-1]), ("2^31", [2**31]), ("empty", [])]
BIG_T = [("T=1e5", 100_000)]

DEADLINE_DEFAULT = 45.0
DEADLINE_HUGE = 15.0
DEADLINE_BIG = 45.0


class Skip(Exception):
    """The mutation does not apply to this slot."""


# --------------------------------------------------------------------------- #
# mutation enumeration (shared by parent and child, so ids agree)
# --------------------------------------------------------------------------- #
def _array_variant(a, variant):
    if variant == "str":
        return "abc"
    if variant == "nan_all":
        return np.full_like(a, np.nan)
    if variant == "nan_one":
        b = a.copy()
        b.flat[b.size // 2] = np.nan
        return b
    if variant == "inf_one":
        b = a.copy()
        b.flat[b.size // 2] = np.inf
        return b
    if variant == "empty":
        return np.empty((0,) + a.shape[1:])
    if variant == "one":
        return a[:1].copy() if a.ndim == 1 else a[:1, :1].copy()
    if variant == "zero_cols":
        if a.ndim != 2:
            raise Skip
        return np.empty((a.shape[0], 0))
    raise Skip


def _panel_variant(lst, variant):
    if variant == "empty":
        return []
    if variant == "one_empty":
        return [np.empty((0,) + lst[0].shape[1:])] + [x for x in lst[1:]]
    if variant == "nan":
        return [np.full_like(x, np.nan) for x in lst]
    raise Skip


def _is_float_array(v):
    return isinstance(v, np.ndarray) and v.dtype.kind == "f" and v.ndim >= 1


def _is_panel(v):
    return isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(e, np.ndarray) for e in v)


def _is_int_list(v):
    return (
        isinstance(v, (list, tuple))
        and len(v) > 0
        and all(isinstance(e, (int, np.integer)) and not isinstance(e, bool) for e in v)
    )


def _is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _slot_mutations(prefix, value):
    """Yield (id, kind, replacement-thunk) for one argument slot."""
    if _is_float_array(value):
        for variant in ARRAY_VARIANTS:
            if variant == "zero_cols" and value.ndim != 2:
                continue
            yield f"{prefix}:{variant}", "array", (lambda v=value, va=variant: _array_variant(v, va))
    elif _is_panel(value):
        for variant in PANEL_VARIANTS:
            yield f"{prefix}:panel_{variant}", "panel", (lambda v=value, va=variant: _panel_variant(v, va))
    elif _is_int_list(value):
        for name, rep in ILIST_VARIANTS:
            yield f"{prefix}:ilist_{name}", "ilist", (lambda r=rep: list(r))
    elif _is_int(value):
        for name, rep in INT_VALUES:
            yield f"{prefix}:int={name}", "int", (lambda r=rep: r)
    elif isinstance(value, float):
        for name, rep in FLOAT_VALUES:
            yield f"{prefix}:float={name}", "float", (lambda r=rep: r)


def enumerate_mutations(fn_name, fn):
    """Return an ordered list of (id, kind, build) where build() -> (args, kwargs)."""
    from registry_ml import build

    args, kwargs = build(fn_name)
    muts = []

    def positional(i, thunk):
        def make():
            a = list(args)
            a[i] = thunk()
            return a, dict(kwargs)

        return make

    def keyword(k, thunk):
        def make():
            kw = dict(kwargs)
            kw[k] = thunk()
            return list(args), kw

        return make

    for i, a in enumerate(args):
        for mid, kind, thunk in _slot_mutations(f"arg{i}", a):
            muts.append((mid, kind, positional(i, thunk)))
    for k, v in kwargs.items():
        for mid, kind, thunk in _slot_mutations(f"kw:{k}", v):
            muts.append((mid, kind, keyword(k, thunk)))

    # Signature defaults the canonical call leaves untouched.
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
    except (TypeError, ValueError):
        params = []
    bound = set(kwargs)
    for p, _ in zip(params, args):
        bound.add(p.name)
    for p in params:
        if p.name in bound or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        d = p.default
        if d is inspect.Parameter.empty:
            continue
        if _is_int(d):
            for name, rep in INT_VALUES:
                muts.append((f"kw:{p.name}:int={name}", "int", keyword(p.name, lambda r=rep: r)))
        elif isinstance(d, float):
            for name, rep in FLOAT_VALUES:
                muts.append((f"kw:{p.name}:float={name}", "float", keyword(p.name, lambda r=rep: r)))
        elif d is None and p.name in _NONE_INT_HINTS:
            # Optional integer sizes whose None default hides them from the
            # type-based rule: probe them as ints.
            for name, rep in INT_VALUES:
                muts.append((f"kw:{p.name}:int={name}", "int", keyword(p.name, lambda r=rep: r)))

    for name, T in BIG_T:
        muts.append((f"big:{name}", "big", (lambda T=T: build(fn_name, T=T))))
    return muts


# Optional-int parameters (default None) worth probing as sizes.
_NONE_INT_HINTS = {
    "n_boot", "n_draws", "n_eval", "calib", "lags", "max_depth", "block", "block_length",
    "batch", "batch_size", "patience", "max_iter", "rff_features", "n_permutations",
    "permutation_block", "hac_lags", "seed", "max_lags", "nlags", "burn", "kmax", "n_grid",
    "horizon", "steps", "window", "bandwidth", "period", "nperseg", "train",
}


def deadline_for(mid, kind):
    if kind == "big":
        return DEADLINE_BIG
    if kind in ("int", "ilist") and any(mid.endswith(h) for h in HUGE_INT):
        return DEADLINE_HUGE
    return DEADLINE_DEFAULT


# --------------------------------------------------------------------------- #
# child
# --------------------------------------------------------------------------- #
def _maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def child(fn_name, ids, rlimit_gb):
    cap = int(rlimit_gb * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    import tsecon

    fn = getattr(tsecon, fn_name)
    muts = {m[0]: m for m in enumerate_mutations(fn_name, fn)}
    for mid in ids:
        print(f"START {mid}", flush=True)
        rss0 = _maxrss_mb()
        t0 = time.time()
        rec = {}
        try:
            args, kwargs = muts[mid][2]()
        except Skip:
            print(f"DONE {mid} " + json.dumps({"outcome": "skip"}), flush=True)
            continue
        except Exception as exc:  # noqa: BLE001 — the harness, not the library
            print(f"DONE {mid} " + json.dumps({"outcome": "harness-error", "exc": type(exc).__name__, "msg": str(exc)[:200]}), flush=True)
            continue
        built = time.time() - t0
        print(f"BUILT {mid} {built:.3f}", flush=True)
        t1 = time.time()
        try:
            fn(*args, **kwargs)
            rec = {"outcome": "ok"}
        except MemoryError as exc:
            rec = {"outcome": "memerr", "exc": "MemoryError", "msg": str(exc)[:300]}
        except (ValueError, TypeError, OverflowError) as exc:
            rec = {"outcome": "refusal", "exc": type(exc).__name__, "msg": str(exc)[:300]}
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            rec = {"outcome": "exc", "exc": type(exc).__name__, "msg": str(exc)[:300]}
        except BaseException as exc:  # noqa: BLE001 — PanicException lands here
            rec = {"outcome": "PANIC", "exc": type(exc).__name__, "msg": str(exc)[:300]}
        rec["seconds"] = round(time.time() - t1, 3)
        rec["build_seconds"] = round(built, 3)
        rec["rss_delta_mb"] = round(_maxrss_mb() - rss0, 1)
        print(f"DONE {mid} " + json.dumps(rec), flush=True)
    print("END", flush=True)


# --------------------------------------------------------------------------- #
# parent
# --------------------------------------------------------------------------- #
def _reader(proc, q):
    for line in proc.stdout:
        q.put(line.rstrip("\n"))
    q.put(None)


def run_function(fn_name, muts, rlimit_gb, python):
    ids = [m[0] for m in muts]
    kinds = {m[0]: m[1] for m in muts}
    pending = list(ids)
    records = []
    env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    spawn_count = 0
    while pending:
        spawn_count += 1
        err_path = os.path.join(OUT, "tmp", f"{fn_name}.{spawn_count}.stderr")
        os.makedirs(os.path.dirname(err_path), exist_ok=True)
        with open(err_path, "w") as err_fh:
            proc = subprocess.Popen(
                [python, os.path.abspath(__file__), "--child", fn_name, "--rlimit-gb", str(rlimit_gb), "--ids", json.dumps(pending)],
                stdout=subprocess.PIPE,
                stderr=err_fh,
                text=True,
                env=env,
                cwd=HERE,
            )
            q: queue.Queue = queue.Queue()
            threading.Thread(target=_reader, args=(proc, q), daemon=True).start()
            current = None
            t_start = None
            dead = False
            finished_this_spawn = False
            while True:
                timeout = deadline_for(current, kinds[current]) if current else 120.0
                try:
                    line = q.get(timeout=timeout)
                except queue.Empty:
                    proc.kill()
                    proc.wait()
                    records.append({"fn": fn_name, "id": current or "<import>", "outcome": "HANG", "deadline": timeout})
                    dead = True
                    break
                if line is None:
                    # EOF: the child ended. Was it clean?
                    proc.wait()
                    if not finished_this_spawn:
                        tail = open(err_path).read()[-1500:]
                        rc = proc.returncode
                        outcome = "CRASH"
                        detail = f"rc={rc}"
                        if "memory allocation of" in tail:
                            outcome = "ALLOC-ABORT"
                            nums = [w for w in tail.replace("\n", " ").split() if w.isdigit()]
                            detail = f"rc={rc} bytes={nums[-1] if nums else '?'}"
                        elif "capacity overflow" in tail:
                            outcome = "CRASH-CAPACITY-OVERFLOW"
                        records.append({"fn": fn_name, "id": current or "<import>", "outcome": outcome, "detail": detail, "stderr_tail": tail[-600:]})
                        dead = True
                    break
                if line.startswith("START "):
                    current = line[6:]
                    t_start = time.time()
                elif line.startswith("BUILT "):
                    pass
                elif line.startswith("DONE "):
                    _, mid, payload = line.split(" ", 2)
                    rec = json.loads(payload)
                    rec.update(fn=fn_name, id=mid)
                    records.append(rec)
                    pending.remove(mid)
                    current = None
                elif line == "END":
                    finished_this_spawn = True
            if dead and current in pending:
                pending.remove(current)
            elif dead and current is None and pending:
                # died before the first START (import failure): give up on the function
                records.append({"fn": fn_name, "id": "<import>", "outcome": "CRASH", "detail": "died before first cell"})
                pending = []
        try:
            os.remove(err_path)
        except OSError:
            pass
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", metavar="FN")
    ap.add_argument("--ids")
    ap.add_argument("--rlimit-gb", type=float, default=6.0)
    ap.add_argument("--only")
    ap.add_argument("--skip-big", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="append; skip callables already recorded")
    a = ap.parse_args()

    if a.child:
        child(a.child, json.loads(a.ids), a.rlimit_gb)
        return

    os.makedirs(OUT, exist_ok=True)
    import tsecon
    from registry_ml import NAMES

    names = a.only.split(",") if a.only else NAMES
    jsonl = os.path.join(OUT, "sweep_adversarial.jsonl")
    done_names = set()
    if a.resume and os.path.exists(jsonl):
        with open(jsonl) as fh:
            for line in fh:
                if line.strip():
                    done_names.add(json.loads(line)["fn"])
        names = [n for n in names if n not in done_names]
        print(f"resume: {len(done_names)} callables already recorded, {len(names)} to run", flush=True)
    elif os.path.exists(jsonl):
        os.remove(jsonl)
    plan = {}
    for n in names:
        muts = enumerate_mutations(n, getattr(tsecon, n))
        if a.skip_big:
            muts = [m for m in muts if m[1] != "big"]
        plan[n] = muts
    total = sum(len(v) for v in plan.values())
    print(f"{len(plan)} callables, {total} cells, {a.workers} workers, rlimit {a.rlimit_gb} GB", flush=True)

    all_records = []
    lock = threading.Lock()
    t0 = time.time()
    done = [0]

    def job(n):
        recs = run_function(n, plan[n], a.rlimit_gb, sys.executable)
        with lock:
            all_records.extend(recs)
            with open(jsonl, "a") as fh:  # persist per callable: a killed run keeps its records
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            done[0] += 1
            bad = [r for r in recs if r["outcome"] not in ("ok", "refusal", "skip", "harness-error")]
            print(f"[{done[0]}/{len(plan)}] {n}: {len(recs)} cells, {len(bad)} non-refusal " + (", ".join(f"{r['id']}={r['outcome']}" for r in bad[:6]) if bad else "") + f"  ({time.time() - t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(job, plan))

    # Summarise everything on disk (this run plus any resumed-over runs).
    all_records = []
    with open(jsonl) as fh:
        for line in fh:
            if line.strip():
                all_records.append(json.loads(line))
    plan_size = len({r["fn"] for r in all_records})

    counts = {}
    for r in all_records:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    with open(os.path.join(OUT, "sweep_adversarial.md"), "w") as fh:
        fh.write("# Adversarial-input matrix — summary\n\n")
        fh.write(f"{plan_size} callables, {len(all_records)} cells, {time.time() - t0:.0f} s wall (this run).\n\n")
        fh.write("| outcome | cells |\n|---|---:|\n")
        for k in sorted(counts, key=lambda k: -counts[k]):
            fh.write(f"| {k} | {counts[k]} |\n")
        fh.write("\n## Every cell that was not a refusal, a success, or a skip\n\n")
        fh.write("| callable | cell | outcome | detail |\n|---|---|---|---|\n")
        for r in sorted(all_records, key=lambda r: (r["fn"], r["id"])):
            if r["outcome"] in ("ok", "refusal", "skip"):
                continue
            detail = r.get("detail") or r.get("msg") or (f"deadline {r['deadline']}s" if "deadline" in r else "")
            fh.write(f"| `{r['fn']}` | `{r['id']}` | {r['outcome']} | {str(detail)[:160].replace('|', '/')} |\n")
        fh.write("\n## Largest RSS deltas (top 25 cells)\n\n| callable | cell | outcome | RSS delta (MB) | seconds |\n|---|---|---|---:|---:|\n")
        for r in sorted((r for r in all_records if "rss_delta_mb" in r), key=lambda r: -r["rss_delta_mb"])[:25]:
            fh.write(f"| `{r['fn']}` | `{r['id']}` | {r['outcome']} | {r['rss_delta_mb']} | {r['seconds']} |\n")
        fh.write("\n## Slowest cells (top 25, completed)\n\n| callable | cell | outcome | seconds |\n|---|---|---|---:|\n")
        for r in sorted((r for r in all_records if "seconds" in r), key=lambda r: -r["seconds"])[:25]:
            fh.write(f"| `{r['fn']}` | `{r['id']}` | {r['outcome']} | {r['seconds']} |\n")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
