"""durbin_watson_test: the p-value's cost is bounded only through T (computed for T <= 4000,
"measured 14.7 s at the cap" in the specification-tests card, on a k = 3 design), but it also
grows with k: dw_null_eigenvalues forms B = A + C Q' - Q W' with two dense n x n x k products
(Par::Seq) before the n x n eigendecomposition. No guard on k.

Measured here (shared 4-core machine, load ~5): T=4000 k=2 7.6 s; k=1000 18.4 s; k=2000 66.7 s;
k=3001 330.5 s. T=2000: k=2 1.1 s, k=1500 16.4 s.

Usage: python repro_dw_wide.py [T k] ...   (default: 4000 3 then 4000 2000)
"""
import sys, time
import numpy as np, tsecon
pairs = [(int(a), int(b)) for a, b in zip(sys.argv[1::2], sys.argv[2::2])] or [(4000, 3), (4000, 2000)]
rng = np.random.default_rng(0)
for T, k in pairs:
    y = rng.standard_normal(T)
    x = np.column_stack([np.ones(T), rng.standard_normal((T, k - 1))]) if k > 1 else np.ones((T, 1))
    t0 = time.time()
    r = tsecon.durbin_watson_test(y, x)
    print(f"T={T} k={k}: {time.time()-t0:.1f} s  pvalue={r['pvalue']}", flush=True)
