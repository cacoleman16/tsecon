"""Sweep H — the seed contract for the six 0.9.0 callables.

For every seed-taking parameter (var_girf.seed, threshold_var_girf.seed,
jsz_fit.seed), on a configuration where the seed is documented LIVE:
  (1) same seed twice in-process           -> bit-identical;
  (2) same seed in a fresh subprocess       -> bit-identical;
  (3) a different seed                      -> differs (the echoed `seed` key
                                               excluded from the comparison);
  (4) seed=None                             -> accepted? documented? and, if
                                               accepted, equal to which seed?
Plus, for all six, two in-process calls bit-identical (determinism), and for
the two GIRF callables (whose docstrings promise it) bit-identity at
RAYON_NUM_THREADS=1 vs 4 in fresh subprocesses; jsz_fit is measured at 1 vs 4
threads too, for the record (no promise).

Run:  .venv/bin/python lab/audit/round13/sweep_h_seed.py
Out:  lab/audit/round13/out/sweep_h.txt, sweep_h.json
"""
from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import HERE, OUT, bits_equal, log  # noqa: E402
from registry import NEW, build  # noqa: E402

SEEDED = {"var_girf": "seed", "threshold_var_girf": "seed", "jsz_fit": "seed"}
# documented: var_girf's result "does not depend on n_draws or seed"
DOCUMENTED_INERT = {"var_girf"}
THREAD_PROMISE = {"var_girf", "threshold_var_girf"}

CHILD = r"""
import sys, pickle, json
sys.path.insert(0, %r)
import tsecon
from registry import build
name, key, seed, extra = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
args, kwargs = build(name, T=200, seed=0)
kwargs.update(extra)
if key != "-":
    kwargs[key] = None if seed == "None" else int(seed)
sys.stdout.buffer.write(pickle.dumps(getattr(tsecon, name)(*args, **kwargs)))
"""


def in_subprocess(name, key, seed, extra=None, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run([sys.executable, "-c", CHILD % HERE, name, key, str(seed), json.dumps(extra or {})],
                       capture_output=True, cwd=HERE, env=e, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-400:])
    return pickle.loads(p.stdout)


def strip_echo(res, key):
    return {k: v for k, v in res.items() if k != key}


def main():
    fh = open(os.path.join(OUT, "sweep_h.txt"), "w")
    report = {}
    nondet = []
    for name in NEW:
        fn = getattr(tsecon, name)
        a1, k1 = build(name); a2, k2 = build(name)
        ok, why = bits_equal(fn(*a1, **k1), fn(*a2, **k2))
        if not ok:
            nondet.append(name); log(fh, f"[{name}] NOT DETERMINISTIC twice in-process: {why}")
    log(fh, f"determinism: {len(NEW) - len(nondet)}/{len(NEW)} bit-identical twice in-process; non-deterministic: {nondet}")
    report["nondeterministic"] = nondet
    for name, key in SEEDED.items():
        fn = getattr(tsecon, name)
        args, kwargs = build(name)
        rec = {"config": {k: v for k, v in kwargs.items() if k != key}}
        r1 = fn(*args, **{**kwargs, key: 11}); r2 = fn(*args, **{**kwargs, key: 11}); r3 = fn(*args, **{**kwargs, key: 12})
        ok, why = bits_equal(r1, r2); rec["same_seed_identical"] = ok
        if not ok:
            log(fh, f"[{name}.{key}] SAME SEED DIFFERS in-process: {why}")
        ok3, why3 = bits_equal(strip_echo(r1, key), strip_echo(r3, key))
        rec["different_seed_differs"] = not ok3
        if ok3:
            tag = "documented (linear reduction)" if name in DOCUMENTED_INERT else "SEED INERT on this configuration — candidate"
            log(fh, f"[{name}.{key}] different seed IDENTICAL: {tag}")
        else:
            log(fh, f"[{name}.{key}] different seed differs: {why3}")
        try:
            rs = in_subprocess(name, key, 11)
            ok4, why4 = bits_equal(r1, rs); rec["subprocess_identical"] = ok4
            if not ok4:
                log(fh, f"[{name}.{key}] SUBPROCESS DIFFERS: {why4}")
        except Exception as exc:  # noqa: BLE001
            rec["subprocess_error"] = str(exc)[:300]; log(fh, f"[{name}.{key}] subprocess failed: {str(exc)[:200]}")
        try:
            rn = fn(*args, **{**kwargs, key: None})
            rec["none_accepted"] = True
            doc = re.sub(r"\s+", " ", fn.__doc__ or "")
            rec["none_documented"] = bool(re.search(rf"`{key}`[^.]*None|None[^.]*`{key}`|{key}\s*=\s*None|\({key}\)", doc))
            eq0, _ = bits_equal(strip_echo(rn, key), strip_echo(fn(*args, **{**kwargs, key: 0}), key))
            rn2 = fn(*args, **{**kwargs, key: None})
            det, _ = bits_equal(rn, rn2)
            rec["none_equals_seed0"] = eq0; rec["none_deterministic"] = det
            rec["none_echo"] = rn.get(key, "<no echo>")
            log(fh, f"[{name}.{key}] seed=None ACCEPTED: documented={rec['none_documented']} deterministic={det} equals seed=0: {eq0} echoed as {rec['none_echo']!r}")
        except Exception as exc:  # noqa: BLE001
            rec["none_accepted"] = False; rec["none_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            log(fh, f"[{name}.{key}] seed=None refused: {rec['none_error']}")
        report[f"{name}.{key}"] = rec
    # history subsample keyed by seed (documented in the hand-off; is it on a surface?)
    fn = tsecon.threshold_var_girf
    args, kwargs = build("threshold_var_girf")
    a = fn(*args, **{**kwargs, "histories": 40, "seed": 1}); b = fn(*args, **{**kwargs, "histories": 40, "seed": 2})
    log(fh, f"[threshold_var_girf] histories=40: seed 1 vs 2 history_times identical={list(a['history_times']) == list(b['history_times'])} "
            f"(n_histories {a['n_histories']}); histories=10**6 -> n_histories={fn(*args, **{**kwargs, 'histories': 10**6})['n_histories']} (clamp)")
    fn = tsecon.var_girf
    args, kwargs = build("var_girf")
    log(fh, f"[var_girf] histories=10**6 -> n_histories={fn(*args, **{**kwargs, 'histories': 10**6})['n_histories']} (clamp; T-p={200 - args[1]})")
    for name in sorted(THREAD_PROMISE | {"jsz_fit"}):
        key = SEEDED[name]
        try:
            a = in_subprocess(name, key, 5, env={"RAYON_NUM_THREADS": "1"})
            b = in_subprocess(name, key, 5, env={"RAYON_NUM_THREADS": "4"})
            ok, why = bits_equal(a, b)
            report[f"{name}.threads"] = ok
            log(fh, f"[{name}] RAYON_NUM_THREADS 1 vs 4: {'bit-identical' if ok else 'DIFFER ' + why} ({'promised' if name in THREAD_PROMISE else 'no promise'})")
        except Exception as exc:  # noqa: BLE001
            log(fh, f"[{name}] thread probe failed: {str(exc)[:200]}")
    json.dump(report, open(os.path.join(OUT, "sweep_h.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
