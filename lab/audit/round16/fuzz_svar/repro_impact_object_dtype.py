"""impact= coercion (structural_fevd / historical_decomposition): the same
object-dtype matrix of plain floats is refused as an ndarray but accepted as a
pandas DataFrame, so the "object matrix is refused" rule depends on the
container, not the content."""
import numpy as np
import pandas as pd
import tsecon

Y = np.random.default_rng(0).standard_normal((120, 2))
A0 = [[1.0, 0.2], [0.3, 0.9]]
for name, v in [("ndarray float64", np.array(A0)),
                ("ndarray object", np.array(A0, dtype=object)),
                ("DataFrame object", pd.DataFrame(A0).astype(object)),
                ("nested list", A0)]:
    for f, kw in [(tsecon.structural_fevd, dict(lags=1)), (tsecon.historical_decomposition, dict(lags=1, identification="impact"))]:
        try:
            r = f(Y, impact=v, **kw)
            print(f"{f.__name__:24s} {name:18s} accepted, impact={np.asarray(r['impact']).tolist()}")
        except Exception as e:
            print(f"{f.__name__:24s} {name:18s} refused: {str(e)[:90]}")
