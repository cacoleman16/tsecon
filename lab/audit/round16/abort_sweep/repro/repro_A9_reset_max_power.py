"""A9_reset_max_power

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_A9_reset_max_power.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('max_power', 'tsecon.reset_test(y=YD, x=XD, max_power=2**31)'),  # reset_test:max_power=2**31 -> abort 
]

if __name__ == "__main__":
    run(CASES, timeout=60)
