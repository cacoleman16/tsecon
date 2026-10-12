"""var_irf_bands' bootstrap arm -- the engine this batch shares with
long_run_svar_bands -- has no working-set budget: n_boot=10**7 (or a large
horizon) is not refused; under a 4 GB address-space limit a worker thread
panics on a failed faer allocation and the process then hangs or aborts.
long_run_svar_bands refuses the same n_boot / horizon in microseconds naming
them. (The known-defect list covers only var_irf_bands' ASYMPTOTIC horizon.)"""
import subprocess, sys
PY = sys.executable
pre = ("import numpy as np, tsecon\n"
       "rng=np.random.default_rng(1); X=np.zeros((200,2))\n"
       "for t in range(1,200): X[t]=0.5*X[t-1]+rng.standard_normal(2)\n")
calls = ["tsecon.long_run_svar_bands(X, n_boot=10**7)",
         "tsecon.var_irf_bands(X, method='bootstrap', n_boot=10**7)",
         "tsecon.long_run_svar_bands(X, n_boot=50, horizon=2**31)",
         "tsecon.var_irf_bands(X, method='bootstrap', n_boot=50, horizon=2**31)"]
for c in calls:
    code = pre + f"try:\n    {c}\n    print('RETURNED')\nexcept Exception as e:\n    print('REFUSED', type(e).__name__, str(e)[:100])\n"
    p = subprocess.run(["bash", "-c", f"ulimit -v 4000000; timeout 90 {PY} -c \"{code}\""], capture_output=True, text=True)
    err = next((l for l in p.stderr.splitlines() if "memory allocation" in l or "AllocError" in l or "panicked" in l), "")
    status = {0: "", 124: "TIMEOUT(90s)", -6: "ABORT(SIGABRT)", 134: "ABORT(SIGABRT)"}.get(p.returncode, f"rc={p.returncode}")
    print(f"{c}\n   -> {status} {p.stdout.strip()[:150]} {err[:120]}", flush=True)
