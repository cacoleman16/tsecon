"""multivariate_normality has no size/work cap: the cost grows like T*k^2 + k^3
(three full validations plus three statistics), so a 12 MB residual matrix
(T=1500, k=1000) runs for minutes with the GIL held, uninterruptibly.
Expected: a named refusal of resid before any work (as dfm_smooth / adl_midas
budget their work), or a documented cost. Actual: timings below."""
import sys, time
import numpy as np
import tsecon
rng = np.random.default_rng(0)
for T, k in [(300, 100), (600, 200), (1200, 400)] + ([(1500, 1000)] if "--big" in sys.argv else []):
    x = rng.standard_normal((T, k))
    t = time.time()
    r = tsecon.multivariate_normality(x)
    dt = time.time() - t
    print(f"T={T:5d} k={k:5d} input={x.nbytes/2**20:6.1f} MiB  time={dt:7.2f} s  doornik_hansen={r['doornik_hansen']:.3f}", flush=True)
