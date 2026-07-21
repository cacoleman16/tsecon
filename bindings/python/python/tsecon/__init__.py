"""tsecon — high-performance time series econometrics.

A Rust core with a Python-first API. The estimators themselves are compiled
(``tsecon._core``, built with PyO3) and re-exported here, so ``tsecon.var_fit``
and friends are the compiled functions with no Python-level indirection.

This package layer exists so that pure-Python pieces can sit alongside the Rust
core — currently :mod:`tsecon.results` (the opt-in rendering layer) and
:func:`tsecon.check_series` (the one-call diagnostic battery composed from the
compiled tests). The library ships no data loaders and makes no network calls;
the only runtime dependency is NumPy. The public surface and its type stubs
live in ``__init__.pyi``.
"""

from . import _core as _core
from ._core import *  # noqa: F401,F403  — the compiled estimator surface
from ._core import __version__ as __version__

from . import results as results

# check_series is pure Python (a composition over the compiled tests). It is
# imported AFTER the star import above, and the name does not exist in _core,
# so nothing compiled can be shadowed in either direction.
from ._inspect import check_series as check_series

# NOTE: `results` is exposed as a NAMESPACE only — deliberately never
# star-imported. It defines its own `var_fit`/`var_irf` helpers that return rich
# objects, and star-importing them here would silently shadow the compiled
# `tsecon.var_fit`/`tsecon.var_irf`. Opt in explicitly instead:
#
#     from tsecon.results import VARResults
#     fit = VARResults.fit(data, lags=2)      # a dict that also renders
#
# Every results class is a dict subclass, so adopting them is additive: the
# plain-dict contract of the compiled functions is unchanged.
#
# The library never fetches data over the network: it ships no data loaders, so
# `import tsecon` makes no external requests and the only runtime dependency is
# numpy. Bring your own arrays (e.g. via pandas / pandas-datareader).

# The compiled module defines no __all__, so `from ._core import *` above pulls
# in every public name. Rebuild __all__ explicitly for `from tsecon import *`
# and for tooling that introspects it.
__all__ = [_n for _n in dir(_core) if not _n.startswith("_")] + [
    "check_series",
    "results",
]
