"""tsecon — high-performance time series econometrics.

A Rust core with a Python-first API. The estimators themselves are compiled
(``tsecon._core``, built with PyO3) and re-exported here, so ``tsecon.var_fit``
and friends are the compiled functions with no Python-level indirection.

This package layer exists so that pure-Python pieces can sit alongside the Rust
core — currently :mod:`tsecon.datasets` (bundled/downloadable reference data).
The public surface and its type stubs live in ``__init__.pyi``.
"""

from . import _core as _core
from ._core import *  # noqa: F401,F403  — the compiled estimator surface
from ._core import __version__ as __version__

from . import datasets as datasets

# The compiled module defines no __all__, so `from ._core import *` above pulls
# in every public name. Rebuild __all__ explicitly for `from tsecon import *`
# and for tooling that introspects it.
__all__ = [_n for _n in dir(_core) if not _n.startswith("_")] + ["datasets"]
