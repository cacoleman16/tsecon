"""Sweep H — the seed contract for the 0.10.0 surface.

The five new callables that take a `seed` (`ets_fit`, `auto_ets`, `spa_test`,
`model_confidence_set`, `stepm_test`), each on a configuration where the seed
is documented LIVE (ETS needs a non-class-1 model and a forecast horizon, or
`seed` is refused as inert):
  (1) same seed twice in-process        -> bit-identical;
  (2) same seed in a fresh subprocess    -> bit-identical;
  (3) a different seed                   -> differs (the echoed `seed` key
                                            excluded from the comparison);
  (4) seed=None                          -> accepted? documented? equal to
                                            which seed? deterministic?
Plus, for all thirteen, two in-process calls bit-identical (determinism) and,
for the callables whose docstrings promise it (`spa_test`,
`model_confidence_set`, `stepm_test`: "one Philox substream per replication,
bit-identical at any thread count"), bit-identity at RAYON_NUM_THREADS 1 vs 4
in fresh subprocesses; the ETS simulators are measured at 1 vs 4 too, for the
record.

Run:  .venv/bin/python lab/audit/round14/sweep_h_seed.py
Out:  lab/audit/round14/out/sweep_h.txt, sweep_h.json
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
from registry import MASKED, NEW14, build  # noqa: E402

SEEDED = {
    "ets_fit": "seed",
    # the search must land on a non-class-1 model or nothing is simulated and
    # the seed is inert by design: `auto_ets@mul` is that cell
    "auto_ets@mul": "seed",
    "spa_test": "seed",
    "model_confidence_set": "seed",
    "stepm_test": "seed",
}
# the configuration on which the seed is documented to act
LIVE = {
    # class-1 models get exact closed-form intervals and REFUSE a seed; a
    # multiplicative error makes the intervals simulated, so the seed acts
    "ets_fit": {"error": "mul", "seasonal": "mul", "horizon": 4, "n_sim": 200},
    "auto_ets@mul": {},
}
THREAD_PROMISE = {"spa_test", "model_confidence_set", "stepm_test"}
SEED_BASE = {n.split("@")[0] for n in SEEDED}

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
sys.stdout.buffer.write(pickle.dumps(getattr(tsecon, name.split("@")[0])(*args, **kwargs)))
"""


def in_subprocess(name, key, seed, extra=None, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run([sys.executable, "-c", CHILD % HERE, name, key, str(seed), json.dumps(extra or {})],
                       capture_output=True, cwd=HERE, env=e, timeout=900)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-400:])
    return pickle.loads(p.stdout)


def strip_echo(res, key):
    return {k: v for k, v in res.items() if k != key}


def main():
    fh = open(os.path.join(OUT, "sweep_h.txt"), "w")
    report = {}
    scope = NEW14 + MASKED
    nondet = []
    for name in scope:
        fn = getattr(tsecon, name.split("@")[0])
        a1, k1 = build(name)
        a2, k2 = build(name)
        ok, why = bits_equal(fn(*a1, **k1), fn(*a2, **k2))
        if not ok:
            nondet.append(name)
            log(fh, f"[{name}] NOT DETERMINISTIC twice in-process: {why}")
    log(fh, f"determinism: {len(scope) - len(nondet)}/{len(scope)} bit-identical twice in-process; "
            f"non-deterministic: {nondet}")
    report["nondeterministic"] = nondet
    for name, key in SEEDED.items():
        fn = getattr(tsecon, name.split("@")[0])
        args, kwargs = build(name)
        kwargs.update(LIVE.get(name, {}))
        extra = LIVE.get(name, {})
        rec = {"config": {k: v for k, v in kwargs.items() if k != key}}
        r1 = fn(*args, **{**kwargs, key: 11})
        r2 = fn(*args, **{**kwargs, key: 11})
        r3 = fn(*args, **{**kwargs, key: 12})
        ok, why = bits_equal(r1, r2)
        rec["same_seed_identical"] = ok
        if not ok:
            log(fh, f"[{name}.{key}] SAME SEED DIFFERS in-process: {why}")
        ok3, why3 = bits_equal(strip_echo(r1, key), strip_echo(r3, key))
        rec["different_seed_differs"] = not ok3
        if ok3:
            log(fh, f"[{name}.{key}] different seed IDENTICAL — SEED INERT on this configuration (candidate)")
        else:
            log(fh, f"[{name}.{key}] different seed differs: {why3}")
        try:
            rs = in_subprocess(name, key, 11, extra=extra)
            ok4, why4 = bits_equal(r1, rs)
            rec["subprocess_identical"] = ok4
            if not ok4:
                log(fh, f"[{name}.{key}] SUBPROCESS DIFFERS: {why4}")
        except Exception as exc:  # noqa: BLE001
            rec["subprocess_error"] = str(exc)[:300]
            log(fh, f"[{name}.{key}] subprocess failed: {str(exc)[:200]}")
        try:
            rn = fn(*args, **{**kwargs, key: None})
            rec["none_accepted"] = True
            doc = re.sub(r"\s+", " ", fn.__doc__ or "")
            rec["none_documented"] = bool(
                re.search(rf"`{key}` \(None: 0|`{key}`[^.]*None|None[^.]*`{key}`|{key}\s*=\s*None", doc))
            eq0, _ = bits_equal(strip_echo(rn, key), strip_echo(fn(*args, **{**kwargs, key: 0}), key))
            det, _ = bits_equal(rn, fn(*args, **{**kwargs, key: None}))
            rec["none_equals_seed0"] = eq0
            rec["none_deterministic"] = det
            rec["none_echo"] = rn.get(key, "<no echo>")
            log(fh, f"[{name}.{key}] seed=None ACCEPTED: documented={rec['none_documented']} "
                    f"deterministic={det} equals seed=0: {eq0} echoed as {rec['none_echo']!r}")
        except Exception as exc:  # noqa: BLE001
            rec["none_accepted"] = False
            rec["none_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            log(fh, f"[{name}.{key}] seed=None refused: {rec['none_error']}")
        report[f"{name}.{key}"] = rec
    for name in sorted(SEEDED):
        key = SEEDED[name]
        extra = LIVE.get(name, {})
        try:
            a = in_subprocess(name, key, 5, extra=extra, env={"RAYON_NUM_THREADS": "1"})
            b = in_subprocess(name, key, 5, extra=extra, env={"RAYON_NUM_THREADS": "4"})
            ok, why = bits_equal(a, b)
            report[f"{name}.threads"] = ok
            log(fh, f"[{name}] RAYON_NUM_THREADS 1 vs 4: "
                    f"{'bit-identical' if ok else 'DIFFER ' + why} "
                    f"({'promised' if name in THREAD_PROMISE else 'no promise'})")
        except Exception as exc:  # noqa: BLE001
            log(fh, f"[{name}] thread probe failed: {str(exc)[:200]}")
    # the unseeded callables: two threads must not move them either
    for name in [n for n in scope if n.split("@")[0] not in SEED_BASE]:
        try:
            a = in_subprocess(name, "-", 0, env={"RAYON_NUM_THREADS": "1"})
            b = in_subprocess(name, "-", 0, env={"RAYON_NUM_THREADS": "4"})
            ok, why = bits_equal(a, b)
            report[f"{name}.threads"] = ok
            if not ok:
                log(fh, f"[{name}] RAYON_NUM_THREADS 1 vs 4 DIFFER: {why}")
        except Exception as exc:  # noqa: BLE001
            log(fh, f"[{name}] thread probe failed: {str(exc)[:200]}")
    unseeded = [n for n in scope if n.split("@")[0] not in SEED_BASE]
    log(fh, f"thread-invariance (unseeded, 1 vs 4): "
            f"{sum(1 for n in unseeded if report.get(f'{n}.threads'))}/{len(unseeded)} bit-identical")
    json.dump(report, open(os.path.join(OUT, "sweep_h.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
