"""One-cell-per-line child harness shared by sweeps G (caps) and S (malformed
input): every cell runs in a child process under a virtual-memory cap and a
wall-clock deadline, so a panic, an abort, or a hang is attributed to the
exact cell that produced it.

Protocol: the parent starts `python cell_runner.py <name> <rlimit_gb>` and
writes one JSON mutation per line to its stdin; the child answers
`DONE <json>` per cell. A mutation is
    {"slot": ["pos", i] | ["kw", k], "variant": "<token>" | ["value", v]}
where a token names a corruption of the canonical value in that slot (arrays:
nan_all, nan_one, inf_one, empty, one_row, zero_cols, rank_down, rank_up,
short_rows, short_cols, short_mid, nested_list, int_array, bool_array,
transposed, dup_row) and ["value", v] substitutes a JSON-encodable literal
(floats "nan"/"inf"/"-inf" spelled as strings with the "f:" prefix; tuples as
{"tuple": [...]}).
"""
from __future__ import annotations

import json
import math
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def decode(v):
    if isinstance(v, str) and v.startswith("f:"):
        return float(v[2:])
    if isinstance(v, dict) and "tuple" in v:
        return tuple(decode(e) for e in v["tuple"])
    if isinstance(v, dict) and "ndarray" in v:
        return np.asarray(v["ndarray"], dtype=v.get("dtype", "float64"))
    if isinstance(v, list):
        return [decode(e) for e in v]
    return v


def array_variant(a, variant):
    a = np.asarray(a)
    if variant == "nan_all":
        return np.full_like(a, np.nan)
    if variant == "nan_one":
        b = a.copy(); b.flat[b.size // 2] = np.nan; return b
    if variant == "inf_one":
        b = a.copy(); b.flat[b.size // 2] = np.inf; return b
    if variant == "empty":
        return np.empty((0,) + a.shape[1:])
    if variant == "one_row":
        return a[:1].copy()
    if variant == "zero_cols":
        return np.empty(a.shape[:-1] + (0,))
    if variant == "rank_down":
        return a[..., 0].copy() if a.ndim >= 2 else float(a[0])
    if variant == "rank_up":
        return a[None].copy()
    if variant == "short_rows":
        return a[:-1].copy()
    if variant == "short_cols":
        return a[..., :-1].copy()
    if variant == "short_mid":
        return a[:, :-1].copy()
    if variant == "nested_list":
        return a.tolist()
    if variant == "int_array":
        return np.rint(a).astype(np.int64)
    if variant == "bool_array":
        return a > 0
    if variant == "transposed":
        return np.ascontiguousarray(a.T)
    if variant == "dup_row":
        b = a.copy(); b[-1] = b[0]; return b
    raise KeyError(variant)


def apply(args, kwargs, mut):
    args, kwargs = list(args), dict(kwargs)
    kind, key = mut["slot"]
    var = mut["variant"]
    if isinstance(var, list) and var[0] == "value":
        new = decode(var[1])
    else:
        base = args[key] if kind == "pos" else kwargs[key]
        new = array_variant(base, var)
    if kind == "pos":
        args[key] = new
    else:
        kwargs[key] = new
    return args, kwargs


def main():
    name, rlimit_gb = sys.argv[1], float(sys.argv[2])
    cap = int(rlimit_gb * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    import tsecon
    from registry import build

    fn = getattr(tsecon, name)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        mut = json.loads(line)
        rec = {"id": mut["id"]}
        try:
            args, kwargs = build(name, T=mut.get("T", 200), seed=0)
            args, kwargs = apply(args, kwargs, mut)
        except Exception as exc:  # noqa: BLE001
            rec.update(outcome="harness-error", msg=f"{type(exc).__name__}: {exc}"[:200])
            print("DONE " + json.dumps(rec), flush=True)
            continue
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        t0 = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
            rec["outcome"] = "ok"
            if isinstance(out, dict):
                nf = 0
                for v in out.values():
                    try:
                        arr = np.asarray(v, dtype=float)
                        nf += int((~np.isfinite(arr)).sum()) if arr.size else 0
                    except (TypeError, ValueError):
                        pass
                rec["nonfinite"] = nf
        except MemoryError as exc:
            rec.update(outcome="memerr", exc="MemoryError", msg=str(exc)[:200])
        except (ValueError, TypeError, OverflowError) as exc:
            rec.update(outcome="refusal", exc=type(exc).__name__, msg=str(exc)[:300])
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            rec.update(outcome="exc", exc=type(exc).__name__, msg=str(exc)[:300])
        except BaseException as exc:  # noqa: BLE001 — PanicException lands here
            rec.update(outcome="PANIC", exc=type(exc).__name__, msg=str(exc)[:300])
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        rec["rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 - rss0, 1)
        print("DONE " + json.dumps(rec), flush=True)
    print("END", flush=True)


if __name__ == "__main__":
    main()
