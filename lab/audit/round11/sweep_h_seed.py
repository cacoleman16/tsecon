"""Sweep H — the seed contract.

For every function with a `seed`/`band_seed`/`rf_seed` parameter, on a call
configuration where that seed is LIVE:
  (1) same seed twice, in-process         -> bit-identical;
  (2) same seed in a fresh subprocess      -> bit-identical;
  (3) a different seed                     -> differs;
  (4) seed=None                            -> accepted? and if so documented?
Plus, for EVERY callable, two in-process calls must be bit-identical
(determinism), and for the parallel bootstrap tests a RAYON_NUM_THREADS=1
subprocess must match the default-thread run ("bit-identical at any thread
count" is a documented promise).

Run:  .venv-wt/bin/python lab/audit/round11/sweep_h_seed.py
Out:  lab/audit/round11/out/sweep_h.log, sweep_h.json
"""
from __future__ import annotations

import inspect
import json
import os
import pickle
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import HERE, bits_equal, log  # noqa: E402
from registry import NAMES, build  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# Where the default configuration leaves a seed inert (documented), the live
# configuration used for the seed contract.
LIVE = {
    "var_irf_bands": {"seed": {"method": "bootstrap", "n_boot": 30},
                      "band_seed": {"band": "sup-t", "band_n_sim": 500}},
    "var_forecast": {"band_seed": {"band": "sup-t", "band_n_sim": 500}},
    "lp": {"band_seed": {"band": "sup-t", "band_n_sim": 500}},
    "smooth_lp": {"band_seed": {"band": "sup-t", "band_n_sim": 500}},
    "proxy_ar_sets": {"rf_seed": {"rf_method": "second_order", "rf_draws": 40}},
    "conformal_forecast": {"seed": {"method": "enbpi", "base": "ar", "n_boot": 10}},
    "conformal_backtest": {"seed": {"method": "enbpi", "base": "ar", "n_boot": 10, "n_eval": 10, "horizon": 1, "batch": 1}},
    "historical_decomposition": {"seed": {"identification": "sign", "restrictions": [(0, 0, 0, "+")], "n_draws": 40, "n_weight_draws": 20}},
}
THREAD_PROMISE = {"hansen_seo_test", "setar_test", "threshold_var_test"}

CHILD = r"""
import sys, pickle, os
sys.path.insert(0, %r)
import numpy as np
import tsecon
from registry import build
name, seedkey, seed, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
import json
extra = json.loads(extra)
for key_ in ("restrictions", "sign_restrictions"):
    if key_ in extra:
        extra[key_] = [tuple(r) for r in extra[key_]]
args, kwargs = build(name, T=200, seed=0)
kwargs.update(extra)
if name == "philox_uniforms":
    kwargs["n"] = args[1]
    args = []
if seedkey != "-":
    kwargs[seedkey] = int(seed)
res = getattr(tsecon, name)(*args, **kwargs)
sys.stdout.buffer.write(pickle.dumps(res))
"""


def in_subprocess(name, seedkey, seed, extra, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run([sys.executable, "-c", CHILD % HERE, name, seedkey, str(seed), json.dumps(extra)],
                       capture_output=True, cwd=HERE, env=e, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-400:])
    return pickle.loads(p.stdout)


def seed_params(name):
    fn = getattr(tsecon._core, name, None) or getattr(tsecon, name)
    try:
        return [p for p in inspect.signature(fn).parameters if p in ("seed", "band_seed", "rf_seed")]
    except (TypeError, ValueError):
        return []


def main():
    fh = open(os.path.join(OUT, "sweep_h.log"), "a" if "--only" in sys.argv else "w")
    report = {}
    # ---- universal determinism: every callable, two in-process calls
    nondet = []
    for name in ([] if "--only" in sys.argv else NAMES):
        fn = getattr(tsecon, name)
        try:
            a1, k1 = build(name, T=200, seed=0)
            a2, k2 = build(name, T=200, seed=0)
            r1, r2 = fn(*a1, **k1), fn(*a2, **k2)
        except Exception as exc:  # noqa: BLE001
            log(fh, f"[{name}] determinism: call failed {type(exc).__name__}: {str(exc)[:120]}")
            continue
        ok, why = bits_equal(r1, r2)
        if not ok:
            nondet.append(name)
            log(fh, f"[{name}] NOT DETERMINISTIC across two in-process calls: {why}")
    log(fh, f"determinism: {len(NAMES) - len(nondet)}/{len(NAMES)} bit-identical twice in-process; non-deterministic: {nondet}")
    report["nondeterministic"] = nondet
    # ---- seed contract
    seeded = [n for n in NAMES if seed_params(n)]
    if "--only" in sys.argv:
        seeded = sys.argv[sys.argv.index("--only") + 1].split(",")
    log(fh, f"seeded functions ({len(seeded)}): {seeded}")
    for name in seeded:
        fn = getattr(tsecon, name)
        for key in seed_params(name):
            extra = LIVE.get(name, {}).get(key, {})
            rec = {"extra": extra}
            args, kwargs = build(name, T=200, seed=0)
            kwargs.update(extra)
            if name == "philox_uniforms":
                kwargs["n"] = args[1]
                args = []
            try:
                r1 = fn(*args, **{**kwargs, key: 11})
                r2 = fn(*args, **{**kwargs, key: 11})
                r3 = fn(*args, **{**kwargs, key: 12})
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"
                log(fh, f"[{name}.{key}] CALL FAILED {rec['error'][:200]}")
                report[f"{name}.{key}"] = rec
                continue
            ok, why = bits_equal(r1, r2)
            rec["same_seed_identical"] = ok
            if not ok:
                log(fh, f"[{name}.{key}] SAME SEED DIFFERS in-process: {why}")
            ok3, why3 = bits_equal(r1, r3)
            rec["different_seed_differs"] = not ok3
            if ok3:
                log(fh, f"[{name}.{key}] DIFFERENT SEED IDENTICAL (seed inert on this configuration): {extra}")
            try:
                rs = in_subprocess(name, key, 11, extra)
                ok4, why4 = bits_equal(r1, rs)
                rec["subprocess_identical"] = ok4
                if not ok4:
                    log(fh, f"[{name}.{key}] SUBPROCESS DIFFERS: {why4}")
            except Exception as exc:  # noqa: BLE001
                rec["subprocess_error"] = str(exc)[:300]
                log(fh, f"[{name}.{key}] subprocess failed: {str(exc)[:200]}")
            # seed=None
            try:
                rn = fn(*args, **{**kwargs, key: None})
                rec["none_accepted"] = True
                doc = fn.__doc__ or ""
                rec["none_documented"] = bool(re.search(rf"{key}\s*=\s*None|`{key}`[^.]*None|None[^.]*`{key}`", re.sub(r"\s+", " ", doc)))
                # is None deterministic? (two calls)
                rn2 = fn(*args, **{**kwargs, key: None})
                okn, _ = bits_equal(rn, rn2)
                rec["none_deterministic"] = okn
                log(fh, f"[{name}.{key}] seed=None ACCEPTED documented={rec['none_documented']} deterministic={okn}")
            except Exception as exc:  # noqa: BLE001
                rec["none_accepted"] = False
                rec["none_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            report[f"{name}.{key}"] = rec
    # ---- thread-count promise
    for name in ([] if "--only" in sys.argv else sorted(THREAD_PROMISE)):
        try:
            a = in_subprocess(name, "seed", 5, {}, env={"RAYON_NUM_THREADS": "1"})
            b = in_subprocess(name, "seed", 5, {}, env={"RAYON_NUM_THREADS": "4"})
            ok, why = bits_equal(a, b)
            report[f"{name}.threads"] = ok
            log(fh, f"[{name}] threads 1 vs 4: {'bit-identical' if ok else 'DIFFER ' + why}")
        except Exception as exc:  # noqa: BLE001
            log(fh, f"[{name}] thread probe failed: {str(exc)[:200]}")
    json.dump(report, open(os.path.join(OUT, "sweep_h.json"), "w"), indent=1, default=str)
    fh.close()


if __name__ == "__main__":
    main()
