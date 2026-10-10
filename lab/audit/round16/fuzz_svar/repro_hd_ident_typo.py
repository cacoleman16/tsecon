"""historical_decomposition: a mistyped `identification` combined with
`impact=` is refused with a message that treats the typo as a real mode
("identification=\"Impact\" identifies its own shocks and would ignore it")
and blames `impact`, instead of reporting the unknown identification value
(which is what the same typo gets without `impact`)."""
import numpy as np
import tsecon

Y = np.random.default_rng(0).standard_normal((150, 3))
A0 = np.eye(3)
for kw in [dict(identification="Impact"), dict(identification="Impact", impact=A0),
           dict(identification="impcat", impact=A0), dict(identification="long_run", impact=A0)]:
    try:
        tsecon.historical_decomposition(Y, lags=2, **kw)
        print(kw.get("identification"), "impact" in kw, "-> OK")
    except Exception as e:
        print(f"identification={kw['identification']!r}, impact given={'impact' in kw}:\n    {type(e).__name__}: {e}")
