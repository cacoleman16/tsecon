"""Rich results objects for tsecon estimators.

Every class here is a **`dict` subclass**. The estimators have always returned
plain dicts of documented keys and that stays true — ``res["params"]``,
``isinstance(res, dict)``, ``json``, ``pickle`` and ``**res`` all keep working.
A results object only *adds* rendering and accessors on top:

    >>> from tsecon.results import VARResults
    >>> fit = VARResults.fit(data, lags=2, names=["gdp", "infl", "rate"])
    >>> fit["aic"]                 # still a dict, exactly as before
    >>> print(fit.summary())       # ...that can also render itself
    >>> fit.irf(horizon=12).plot() # ...and plot itself

Because the dict contract is preserved as a subset of the object, adopting these
is additive rather than a breaking change.

Plotting requires matplotlib, an optional dependency::

    pip install 'tsecon[plots]'

Each plot method raises a message naming that extra if it is missing.
"""

from ._base import Results
from ._arima import ARIMAResults
from ._dsge import DSGEResults
from ._garch import GARCHResults
from ._lp import LPResults
from ._predreg import IVXTestResults, PredictiveRegressionResults
from ._var import CoefficientFrame, IRFArray, VARResults, var_fit, var_irf

__all__ = [
    "Results",
    "ARIMAResults",
    "CoefficientFrame",
    "DSGEResults",
    "GARCHResults",
    "IRFArray",
    "IVXTestResults",
    "LPResults",
    "PredictiveRegressionResults",
    "VARResults",
    "var_fit",
    "var_irf",
]
