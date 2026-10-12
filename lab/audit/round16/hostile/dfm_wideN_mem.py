"""dfm_smooth at N = 8192 (the largest N its 2^26-cell guard admits) on a
1 x 8192 panel (64 KiB of data): the docstring prices the N x N observation
covariance at 2^26 cells (512 MiB) and the work at ~7.5e7 units (0.4% of the
2^34 cap, 'about 18 s at most' at the cap). Run under a 2.5 GB address-space
cap, the call aborts the process instead of refusing; under 4 GB it needs
~3.1 GB RSS and ~20 s (265 ns per budgeted unit vs the documented 0.17-1.03)."""
import subprocess, sys
PY = sys.executable
code = r'''
import time, resource, numpy as np, tsecon
N = 8192
t = time.time()
r = tsecon.dfm_smooth(np.zeros((1, N)), np.ones((N, 1)), np.array([[0.5]]), np.ones(N))
print("returned in %.1f s, maxrss %.0f MiB" % (time.time() - t, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), flush=True)
'''
for cap_kb in (2500000, 4000000):
    p = subprocess.run(["bash", "-c", f"ulimit -v {cap_kb}; timeout 200 {PY} -c '{code}'"], capture_output=True, text=True)
    tail = (p.stdout + p.stderr).strip().splitlines()
    print(f"ulimit -v {cap_kb} KB: rc={p.returncode}  {tail[0] if tail else ''}  {tail[-1] if len(tail) > 1 else ''}")
