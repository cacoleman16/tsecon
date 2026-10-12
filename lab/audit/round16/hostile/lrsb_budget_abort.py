"""long_run_svar_bands' bootstrap working-set budget (documented: 'counted as
real storage (padded matrices and the Python lists included), budgeted at
2^28 values (2 GiB) before anything is allocated') under-counts the real
footprint. On a 200 x 2 VAR(2) at horizon=0 it admits n_boot up to ~4.19e6
(64 counted values per draw); n_boot=4_000_000 (2.05 GB counted) is admitted,
runs ~3.5 minutes with the GIL held (SIGINT ignored), and then ABORTS the
interpreter on a failed allocation under a 4 GB address-space limit.
Peak RSS per draw is measured below at n_boot=250_000 and 1_000_000 to show
the ratio to the counted 64 values (512 bytes) per draw.
Usage: python lrsb_budget_abort.py [--abort]   (--abort runs the 4e6 case, ~4 min)"""
import subprocess, sys
PY = sys.executable
pre = ("import numpy as np, tsecon, time, resource\n"
       "rng=np.random.default_rng(1); X=np.zeros((200,2))\n"
       "for t in range(1,200): X[t]=0.5*X[t-1]+rng.standard_normal(2)\n")
sizes = [250_000, 1_000_000] + ([4_000_000] if "--abort" in sys.argv else [])
for nb in sizes:
    code = pre + (f"t=time.time(); r=tsecon.long_run_svar_bands(X, n_boot={nb}, horizon=0)\n"
                  f"rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024\n"
                  f"print('returned in %.1f s; peak RSS %.2f GB = %.0f bytes/draw (budget counts 64 values = 512 bytes/draw)' % (time.time()-t, rss/1e9, rss/{nb}), flush=True)\n")
    p = subprocess.run(["bash", "-c", f"ulimit -v 4000000; timeout 400 {PY} -c \"{code}\""], capture_output=True, text=True)
    err = [l for l in p.stderr.splitlines() if "memory allocation" in l or "panicked" in l or "AllocError" in l]
    print(f"n_boot={nb}: rc={p.returncode} {p.stdout.strip()} {err[:2]}", flush=True)
