"""Shared runner for the abort-sweep repro scripts.

Each call runs in its own subprocess under `ulimit -v 4000000; timeout 60`
with the sweep's fixtures (../prelude.py: small seeded series, T = 200).
For every call it prints the EXPECTED outcome (a ValueError that names the
parameter) and the ACTUAL one: return code (-6 = SIGABRT from the Rust
allocator, 124 = timeout), the exception, or the first line of stderr.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PRELUDE = os.path.join(os.path.dirname(HERE), "prelude.py")
PY = sys.executable

CHILD = r'''
exec(open(PRELUDE).read())
import resource, time
t0 = time.time()
try:
    r = CALL
    print("RESULT: returned", type(r).__name__, "in %.2fs" % (time.time() - t0), "maxrss %.0f MB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
except BaseException as e:
    print("RESULT: raised", type(e).__name__ + ":", str(e)[:400].replace("\n", " "))
'''


def run(cases, timeout=60):
    """cases: list of (param_name, call_code)."""
    bad = 0
    for param, call in cases:
        code = CHILD.replace("PRELUDE", repr(PRELUDE)).replace("CALL", call)
        p = subprocess.run(["bash", "-c", f'ulimit -v 4000000; timeout {timeout} "{PY}" -c "$0"', code],
                           capture_output=True, text=True)
        out = [l for l in p.stdout.splitlines() if l.startswith("RESULT:")]
        if p.returncode == 124:
            actual = f"TIMEOUT after {timeout}s (rc=124)"
        elif out:
            actual = out[-1][len("RESULT: "):]
        else:
            err = p.stderr.strip().splitlines()
            actual = f"process died rc={p.returncode} ({'SIGABRT' if p.returncode in (-6, 134) else 'signal/exit'}): {err[0][:160] if err else ''}"
        ok = actual.startswith("raised") and param in actual
        bad += not ok
        print(f"CALL     {call}")
        print(f"EXPECTED a ValueError/TypeError naming `{param}` (house rule: absurd sizes refused by name before allocating)")
        print(f"ACTUAL   {actual}")
        print(f"VERDICT  {'ok' if ok else 'DEFECT'}\n", flush=True)
    print(f"{bad} of {len(cases)} calls violate the house rule")
