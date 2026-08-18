"""Run the full lab simulation study (exp01-exp05) in sequence.

Each experiment is an independent, seeded, standalone script; this driver
just runs them in order and reports wall time.  Tables land on stdout and
in results/expNN.md (+ .json payloads); lab/REPORT.md embeds the same
tables.

  cd lab/experiments && /path/to/.venv/bin/python run_all.py

Total runtime: roughly 10-13 minutes on the dev container.
"""

from __future__ import annotations

import subprocess
import sys
import time

SCRIPTS = [
    "exp01_point_horse_race.py",
    "exp02_interval_calibration.py",
    "exp03_robust_filtering.py",
    "exp04_tail_quantiles.py",
    "exp05_lad_arima.py",
]


def main():
    t0 = time.time()
    for s in SCRIPTS:
        print(f"\n{'='*72}\n>> {s}\n{'='*72}", flush=True)
        r = subprocess.run([sys.executable, s])
        if r.returncode != 0:
            print(f"FAILED: {s} (exit {r.returncode})", file=sys.stderr)
            sys.exit(r.returncode)
    print(f"\nAll experiments done in {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
