"""REPRO (finding ARDL-size): ardl_bounds_test docstring, CHANGELOG and cointegration-regime card quote
'null size 0.048 (upper bound, I(1) regressor) and 0.047 (lower bound, I(0) regressor)' from 1000 draws
each (crates/tsecon-coint/tests/ardl_properties.rs). Same design (T=200, one regressor, case 3,
ARDL(1,1), y an independent random walk), fresh seed. Runtime ~5 s."""
import math, numpy as np, tsecon
R, T = 40000, 200; rng = np.random.default_rng(13579)
for i1, key, claim in ((True, "p_value_upper", 0.048), (False, "p_value_lower", 0.047)):
    rej = 0
    for _ in range(R):
        y = np.cumsum(rng.standard_normal(T)); e = rng.standard_normal(T)
        rej += tsecon.ardl_bounds_test(y, (np.cumsum(e) if i1 else e)[:, None], lags=1, order=1, case=3)[key] < 0.05
    p = rej / R; se = math.sqrt(p * (1 - p) / R)
    print(f"x {'I(1)' if i1 else 'I(0)'}, {key} < .05: claimed {claim}  measured {p:.4f} (se {se:.4f})  z = {(claim - p)/se:+.1f}")
