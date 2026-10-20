"""mlp_regression(hidden=[<huge>]) as a LIST escapes the house 2**48
pre-flight (the coercion layer turns the list into an int64 ndarray before
_huge_int_offenders scans for lists/tuples) and reaches Rust:
  * hidden=[2**62] -> pyo3_runtime.PanicException (a BaseException that
    `except Exception` does not catch) "range end index 9223372036854775808 out
    of range for slice of length 1" at crates/tsecon-ml/src/mlp.rs:999 on a
    (50, 2) design; on a (200, 3) design the same call is a capacity-overflow
    panic rebuilt into a ValueError that names no argument;
  * hidden=[2**31] / [2**40] -> SIGABRT in the allocator.
The TUPLE hidden=(2**62,) and the int hidden=2**62 are refused naming hidden.
Each call runs in a subprocess under ulimit -v 4000000; timeout 60.
"""
import subprocess, sys
CHILD = r'''
import numpy as np, tsecon
x = np.random.default_rng(0).standard_normal((N, K)); y = x[:, 0]
try:
    tsecon.mlp_regression(x, y, hidden=H, max_epochs=2, n_seeds=1)
    print("RESULT returned")
except Exception as e:
    print("RESULT Exception", type(e).__name__, str(e)[:150])
except BaseException as e:
    print("RESULT BaseException-not-Exception", type(e).__name__, str(e)[:150])
'''
for n, k, h in [(50, 2, "[2**62]"), (200, 3, "[2**62]"), (50, 2, "(2**62,)"), (50, 2, "2**62"), (50, 2, "[2**31]"), (50, 2, "[2**40]")]:
    code = CHILD.replace("N", str(n)).replace("K", str(k)).replace("H", h)
    p = subprocess.run(["bash", "-c", f'ulimit -v 4000000; timeout 60 "{sys.executable}" -c "$0"', code], capture_output=True, text=True)
    out = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
    err = [l for l in p.stderr.splitlines() if l.strip()]
    actual = out[-1] if out else f"process died rc={p.returncode}: {err[0][:120] if err else ''}"
    print(f"x shape ({n},{k}) hidden={h}: EXPECTED ValueError naming `hidden`; ACTUAL {actual}")
