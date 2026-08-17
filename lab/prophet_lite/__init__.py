"""prophet_lite — a from-scratch decomposable structural forecaster.

Part of tsecon's PRIVATE research lab (not shipped, not public API).

y(t) = piecewise-linear trend + Fourier seasonality + beta' x(t) + eps,
MAP-estimated with a Laplace prior on the changepoint rate adjustments
(exact L1-penalized least squares), Prophet-style simulation intervals.

Provenance: implemented from Taylor SJ & Letham B (2018), "Forecasting at
Scale", The American Statistician 72(1) 37-45 (preprint PeerJ 2017).  The
reference implementation (facebook/prophet) is MIT; no code copied.  See
README.md for what is deliberately omitted.

Usage
-----
>>> from prophet_lite import fit
>>> res = fit(y, dates)                       # or fit(y, [(12, 3)]) etc.
>>> fc = res.forecast(24, level=[0.8, 0.95])
>>> comp = res.components()
>>> draws = res.predictive_draws(24, n_draws=1000, seed=0)
"""

from .api import (
    fit,
    ProphetLiteResult,
    forecast_from_result,
    components_from_result,
    predictive_draws_from_result,
)

__all__ = [
    "fit",
    "ProphetLiteResult",
    "forecast_from_result",
    "components_from_result",
    "predictive_draws_from_result",
]

__version__ = "0.1.0"
