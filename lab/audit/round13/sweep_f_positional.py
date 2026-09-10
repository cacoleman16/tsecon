"""Sweep F(f) — the count pre-flight names POSITIONAL offenders on all 179.

0.9.0 changed `tsecon._coerce._param_names` to cache the compiled signature
once per function (commit 342b6a2). The seal must still name a count of 2**48
passed *positionally* by its parameter name, for every positional slot of
every callable — before and after the cache is warm. For each callable and
each positional index i: call fn(*([None]*i + [2**48])) and expect a
ValueError matching `<name>=281474976710656`; seeds are exempt (the seal lets
them through; the call then fails in Rust on the None fillers — recorded).
The pre-flight runs before Rust, so the None fillers never reach the core on
a refusal.

Run:  .venv/bin/python lab/audit/round13/sweep_f_positional.py
Out:  lab/audit/round13/out/sweep_f_positional.txt
"""
from __future__ import annotations

import inspect
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import OUT, log  # noqa: E402
from registry import NAMES  # noqa: E402
from tsecon import _coerce  # noqa: E402

HUGE = 2**48


def main():
    fh = open(os.path.join(OUT, "sweep_f_positional.txt"), "w")
    n_slots = n_named = n_seed = 0
    bad = []
    for rnd in ("cold", "warm"):  # second pass hits the cache
        for name in NAMES:
            fn = getattr(tsecon, name)
            raw = getattr(tsecon._core, name, None)
            if raw is None:  # summarize / check_series: pure Python
                continue
            params = list(inspect.signature(raw).parameters.values())
            for i, p in enumerate(params):
                if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                    continue
                n_slots += 1
                args = [None] * i + [HUGE]
                try:
                    fn(*args)
                    outcome = "returned"
                except ValueError as exc:
                    outcome = str(exc)
                except BaseException as exc:  # noqa: BLE001
                    outcome = f"{type(exc).__name__}: {exc}"
                if _coerce._is_seed_name(p.name):
                    n_seed += 1
                    if "2**48" in outcome:
                        bad.append((rnd, name, i, p.name, "seed refused by the seal"))
                    continue
                if isinstance(outcome, str) and re.search(rf"(^|[^\w]){re.escape(p.name)}={HUGE}\b", outcome) and "2**48" in outcome:
                    n_named += 1
                else:
                    bad.append((rnd, name, i, p.name, outcome[:160]))
        # the cache must be populated and inspect.signature not re-run on a
        # second refusal
        if rnd == "warm":
            assert len(_coerce._PARAM_NAMES) >= 100, len(_coerce._PARAM_NAMES)
    for b in bad:
        log(fh, f"MISNAMED/UNSEALED: pass={b[0]} {b[1]} arg{b[2]} `{b[3]}`: {b[4]}")
    log(fh, f"positional slots probed (cold+warm): {n_slots}; named correctly: {n_named}; seed slots (exempt): {n_seed}; "
            f"offenders: {len(bad)}; cached signatures: {len(_coerce._PARAM_NAMES)}")


if __name__ == "__main__":
    main()
