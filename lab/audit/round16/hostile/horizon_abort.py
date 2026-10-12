"""long_run_svar and structural_fevd have no horizon budget: horizon=2**31
(below the coercion layer's 2**48 guard) reaches Rust's
Vec::with_capacity(horizon + 1) and ABORTS the interpreter ('memory allocation
of 120259084344 bytes failed', SIGABRT), and horizon=10**7 runs for minutes
and many GB. The sibling long_run_svar_bands refuses the same horizons in
microseconds, naming horizon; historical_decomposition caps its horizon.
Each call runs in a child process so this script survives the abort."""
import subprocess, sys
PY = sys.executable
pre = ("import numpy as np, tsecon\n"
       "rng=np.random.default_rng(1); X=np.zeros((200,2))\n"
       "for t in range(1,200): X[t]=0.5*X[t-1]+rng.standard_normal(2)\n")
calls = [
    "tsecon.long_run_svar_bands(X, n_boot=50, horizon=2**31)",
    "tsecon.long_run_svar_bands(X, method='asymptotic', horizon=2**31)",
    "tsecon.long_run_svar(X, horizon=2**31)",
    "tsecon.long_run_svar(X, horizon=2**31, impact_signs=[(0, 1)])",
    "tsecon.structural_fevd(X, horizon=2**31)",
    "tsecon.structural_fevd(X, horizon=2**31, impact=tsecon.long_run_svar(X)['impact'])",
    "tsecon.long_run_svar(X, horizon=10**7)",
]
for c in calls:
    code = pre + f"try:\n    {c}\n    print('RETURNED')\nexcept Exception as e:\n    print('REFUSED', type(e).__name__, str(e)[:110])\n"
    p = subprocess.run(["bash", "-c", f"ulimit -v 4000000; timeout 60 {PY} -c \"{code}\""], capture_output=True, text=True)
    first_err = next((l for l in p.stderr.splitlines() if "memory allocation" in l or "panicked" in l), "")
    status = {0: "", 124: "TIMEOUT(60s)", 134: "ABORT(SIGABRT)", -6: "ABORT(SIGABRT)"}.get(p.returncode, f"rc={p.returncode}")
    print(f"{c}\n   -> {status} {p.stdout.strip()[:160]} {first_err[:110]}")
