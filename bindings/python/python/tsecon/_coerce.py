"""Input coercion: accept pandas / array-likes and off-dtype arrays everywhere.

The compiled estimators require C-contiguous ``float64`` NumPy arrays and
reject anything else at the Rust boundary with a low-level ``TypeError``. This
module wraps every compiled function so that the common ease-of-use inputs —
a pandas ``DataFrame``/``Series`` (or any object exposing ``.to_numpy``, e.g.
polars), a ``float32``/``float16`` array, or a non-contiguous slice — are
converted to ``float64`` C-contiguous arrays before the call.

The rule is deliberately **type-based and conservative**, because a blanket
"convert everything to float64" would corrupt arguments that are *not* data:

* integer / bool NumPy arrays are left **untouched** — they are the only place
  the boundary legitimately wants a non-float array (``hetero_svar``'s integer
  ``regime_labels``, the ``var_granger`` cause/effect index arrays,
  ``favar.slow_indices``). Type alone cannot tell an integer *label* array from
  an integer *data* array, so neither is coerced;
* restriction specs (lists of tuples/dicts), ragged panels (Python lists of
  per-unit arrays), callables, scalars, strings, and ``None`` are all
  non-ndarray, non-pandas objects and pass straight through.

Consequently the *only* copy happens on floating-point arrays and pandas
inputs, which are always genuine data. An integer **data** array still raises
at the boundary (the safe, documented gap); pass ``x.astype(float)`` or a
DataFrame instead. ``check_series`` and the ``tsecon.results`` facade do their
own coercion on the raw ``_core`` primitives, so they bypass this layer.
"""
from __future__ import annotations

import functools
import types

import numpy as np


def _is_pandas(x: object) -> bool:
    # Duck-typed so pandas/polars are not import-time dependencies. A NumPy
    # ndarray has no ``.to_numpy`` in the relevant sense, so it never matches
    # here (it is handled by the ndarray branch of ``_coerce``).
    return hasattr(x, "to_numpy") and not isinstance(x, np.ndarray)


def _coerce(x: object) -> object:
    """Return ``x`` as a C-contiguous float64 array if it is data, else ``x``."""
    if isinstance(x, np.ndarray):
        # Only floating-kind arrays are touched: float16/float32/longdouble get
        # upcast, and a non-contiguous float64 view (a slice or Fortran layout)
        # is made contiguous. Integer/bool/complex/object arrays pass through.
        if x.dtype.kind == "f" and (x.dtype != np.float64 or not x.flags["C_CONTIGUOUS"]):
            return np.ascontiguousarray(x, dtype=np.float64)
        return x
    if _is_pandas(x):
        return np.ascontiguousarray(x.to_numpy(), dtype=np.float64)
    return x


def _wrap(fn):
    """Wrap a compiled function so its array-like arguments are coerced."""

    @functools.wraps(fn)  # copies __name__/__qualname__/__doc__/__module__, sets __wrapped__
    def wrapper(*args, **kwargs):
        return fn(
            *(_coerce(a) for a in args),
            **{k: _coerce(v) for k, v in kwargs.items()},
        )

    return wrapper


def install(namespace: dict, compiled: types.ModuleType) -> set[str]:
    """Rebind wrapped copies of every public compiled function into ``namespace``.

    ``namespace`` is the package globals (``tsecon``'s ``globals()``), which
    already hold the star-imported raw builtins; each is replaced by a wrapper
    that coerces its inputs. ``compiled._core.<fn>`` is left as the raw builtin.
    Returns the set of names that were wrapped.
    """
    wrapped: set[str] = set()
    for name in dir(compiled):
        if name.startswith("_"):
            continue
        obj = getattr(compiled, name)
        if isinstance(obj, types.BuiltinFunctionType):
            namespace[name] = _wrap(obj)
            wrapped.add(name)
    return wrapped
