"""Input coercion: accept pandas / array-likes and off-dtype arrays everywhere.

The compiled estimators require C-contiguous ``float64`` NumPy arrays and
reject anything else at the Rust boundary with a low-level ``TypeError``. This
module wraps every compiled function so that the common ease-of-use inputs —
a pandas ``DataFrame``/``Series`` (or any object exposing ``.to_numpy``, e.g.
polars), a ``float32``/``float16`` array, a non-contiguous slice, or an
**integer / boolean** array (data read with ``dtype=int``, ``np.arange``, a
``0/1`` recession dummy, a ``y > 0`` mask) — are converted to ``float64``
C-contiguous arrays before the call.

Integer arrays are the delicate case, because a handful of parameters
legitimately want integers: they are *labels* and *column indices*, not data,
and converting them to ``float64`` breaks the call. Type alone cannot tell an
integer label array from an integer data array — but the **parameter** can, and
the wrapper knows which function it wraps. So coercion is *parameter-aware*:

* ``_EXEMPT`` lists, per compiled function, the parameters that are not data
  (audited against the Rust signatures: the only integer-sequence arguments in
  the whole API). Arguments landing there — positionally or by keyword — are
  passed through **completely untouched**;
* every other argument is data, so integer/bool/float arrays and pandas objects
  are all converted to C-contiguous ``float64``;
* restriction specs (lists of tuples/dicts), ragged panels (Python lists of
  per-unit arrays), callables, scalars, strings, and ``None`` are non-ndarray,
  non-pandas objects and still pass straight through.

Mapping a positional argument to its parameter name relies on
``inspect.signature`` of the compiled builtin (PyO3 emits a
``__text_signature__``). If that introspection fails or disagrees with
``_EXEMPT`` — a renamed parameter, a ``*args`` signature — the wrapper falls
back to the older, conservative rule for that function (integer arrays left
untouched) rather than guessing and corrupting a label array.

``check_series`` and the ``tsecon.results`` facade do their own coercion on the
raw ``_core`` primitives, so they bypass this layer.
"""
from __future__ import annotations

import functools
import inspect
import types

import numpy as np

# Parameters that are NOT data: integer labels and column indices. Audited
# against every integer-sequence argument in bindings/python/src/lib.rs
# (``Vec<usize>`` / ``Vec<i64>`` / ``Option<Vec<usize>>``); passing float64
# there raises "cannot be interpreted as an integer" at the boundary.
# ``tests/test_coerce.py::test_exempt_set_covers_every_integer_rust_parameter``
# re-derives this from the Rust source so a new integer parameter cannot be
# added without updating the set.
_EXEMPT: dict[str, frozenset[str]] = {
    "hetero_svar": frozenset({"regime_labels"}),
    "var_granger": frozenset({"caused", "causing"}),
    "favar": frozenset({"slow_indices"}),
    # (P, D, Q, s) integer orders, not data: coercing to float64 would make
    # the compiled parser reject a plain [0, 1, 1, 12].
    "arima_fit": frozenset({"seasonal"}),
    # Candidate threshold delays d in y_{t-d}: integer lags, not data.
    "setar": frozenset({"delays"}),
}

_POSITIONAL = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _is_pandas(x: object) -> bool:
    # Duck-typed so pandas/polars are not import-time dependencies. A NumPy
    # ndarray has no ``.to_numpy`` in the relevant sense, so it never matches
    # here (it is handled by the ndarray branch of ``_coerce``).
    return hasattr(x, "to_numpy") and not isinstance(x, np.ndarray)


def _coerce(x: object) -> object:
    """Conservative rule: coerce floating-point arrays and pandas objects only.

    Integer/bool arrays are left untouched. Used for functions whose parameter
    names could not be resolved, where converting an integer array might
    corrupt a label argument.
    """
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


def _coerce_data(x: object) -> object:
    """Coerce a known-data argument to a C-contiguous ``float64`` array.

    Same as :func:`_coerce` but integer (``i``/``u``) and boolean (``b``)
    arrays of rank >= 1 are converted too, since this argument cannot be a
    label array. Complex/object/datetime arrays are still left alone:
    converting them would silently drop information (or raise something less
    clear than the boundary error they already produce). A **0-d** integer
    array is left alone as well — no estimator takes rank-0 data, so it is a
    scalar argument in array clothing (``lags=np.array(2)``, which the
    boundary accepts via ``__index__``).
    """
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "iub" and x.ndim > 0:
            return np.ascontiguousarray(x, dtype=np.float64)
        if x.dtype.kind == "f" and (x.dtype != np.float64 or not x.flags["C_CONTIGUOUS"]):
            return np.ascontiguousarray(x, dtype=np.float64)
        return x
    if _is_pandas(x):
        return np.ascontiguousarray(x.to_numpy(), dtype=np.float64)
    if isinstance(x, list):
        # Only *lists* are considered data. A tuple is how this API spells a
        # fixed-arity option (`weighted_midas(weight_start=(2.0, 3.0))`,
        # `box_cox_lambda(bounds=(-2, 2))`), and converting one to an array
        # breaks the binding.
        return _coerce_sequence(x)
    return x


def _all(seq, pred) -> bool:
    return len(seq) > 0 and all(pred(e) for e in seq)


def _is_number(e: object) -> bool:
    return isinstance(e, (int, float, np.integer, np.floating)) and not isinstance(e, bool)


def _is_arraylike(e: object) -> bool:
    return isinstance(e, np.ndarray) or _is_pandas(e)


def _coerce_sequence(x):
    """Convert a data *sequence* to float64, leaving specs and indices alone.

    Three shapes of Python sequence reach the boundary and only one is data:

    * ``[1.0, 2.0, ...]`` (or a nested numeric list) — a series/matrix written
      literally. Converted, since a plain list is the most natural thing for a
      newcomer to pass and the boundary otherwise rejects it outright.
    * ``[arr0, arr1, ...]`` — a **ragged panel** of per-unit arrays. The list
      container is preserved (the boundary parses it as a sequence of arrays),
      but each element is coerced, so per-unit pandas Series or ``float32``
      arrays now work.
    * ``[(0, 1, 1, "+"), ...]`` or ``[{...}]`` — a restriction **spec**. Left
      untouched: the elements are tuples/dicts/strings, never numbers.

    Anything that does not match one of these patterns is returned unchanged
    rather than guessed at.
    """
    if not x:
        return x
    if _all(x, _is_number):
        return np.ascontiguousarray(x, dtype=np.float64)
    if _all(x, _is_arraylike):
        # Ragged panel: keep the container, coerce the elements.
        return [_coerce_data(e) for e in x]
    # A NESTED sequence is deliberately left alone. `[(0, 1), (0, 2)]` is a
    # restriction spec and `[[1.0, 2.0], [3.0, 4.0]]` is a matrix, and nothing
    # in the value distinguishes them — a spec is all-numeric and rectangular
    # too. Converting a spec by mistake would corrupt it silently, whereas
    # declining to convert a nested literal only costs the user a clear error
    # telling them to wrap it in `np.array(...)`. The asymmetry decides it.
    return x


def _exempt_positions(fn, exempt: frozenset[str]) -> frozenset[int] | None:
    """Positional indices of ``exempt`` in ``fn``'s signature, or ``None``.

    ``None`` means "could not be resolved with certainty" and puts the wrapper
    into the conservative fallback. That happens when the compiled function
    exposes no signature, when a name in ``exempt`` is absent from it (the
    parameter was renamed and the exempt set is stale), or when a ``*args``
    parameter makes the index-to-name mapping ambiguous.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (ValueError, TypeError):
        return None
    if not exempt <= {p.name for p in params}:
        return None
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return None
    # Positional parameters always precede keyword-only ones, so enumerating
    # the parameter list gives the caller-visible positional index directly.
    return frozenset(
        i for i, p in enumerate(params) if p.kind in _POSITIONAL and p.name in exempt
    )


# PyO3 reports a rank mismatch by failing the downcast to PyReadonlyArray{1,2},
# which renders as "'ndarray' object is not an instance of 'ndarray'" — true but
# useless, and one of the most common first-run errors (passing df["col"] where a
# 2-D panel is wanted, or a column matrix where a series is wanted).
_RANK_HINT = "is not an instance of"


def _rank_error(fn_name: str, args, kwargs, original: TypeError) -> TypeError:
    """Rebuild an array-argument TypeError into one that says what to do."""
    described = []
    for label, v in [
        *((f"arg{i}", a) for i, a in enumerate(args)),
        *((k, v) for k, v in kwargs.items()),
    ]:
        if isinstance(v, np.ndarray):
            described.append(f"{label}: array{v.shape}")
        elif isinstance(v, (list, tuple)):
            described.append(f"{label}: {type(v).__name__} of {len(v)}")
    shown = "; ".join(described) if described else "no array arguments"
    return TypeError(
        f"{fn_name}: an array argument is the wrong shape or type (got {shown}). "
        f"Estimators that model a system want a 2-D array shaped "
        f"(observations, series); estimators that model one series want a 1-D "
        f"array. If you sliced one column out of a DataFrame, `df['x']` is 1-D "
        f"— use `df[['x']]` (or `x.reshape(-1, 1)`) where 2-D is wanted, and "
        f"`arr.ravel()` to go the other way. A nested Python list is not "
        f"converted automatically (it is indistinguishable from a restriction "
        f"spec) — wrap it with `np.array(...)`. Original error: {original}"
    )


def _call(fn, args, kwargs):
    """Invoke the compiled function, upgrading rank errors to teaching errors."""
    try:
        return fn(*args, **kwargs)
    except TypeError as exc:  # noqa: PERF203 - only on the error path
        if _RANK_HINT in str(exc):
            raise _rank_error(fn.__name__, args, kwargs, exc) from exc
        raise


def _wrap(fn):
    """Wrap a compiled function so its array-like arguments are coerced."""
    exempt = _EXEMPT.get(fn.__name__, frozenset())

    if not exempt:
        # No label parameters: every array-like argument is data.
        @functools.wraps(fn)  # copies __name__/__qualname__/__doc__/__module__, sets __wrapped__
        def wrapper(*args, **kwargs):
            return _call(
                fn,
                [_coerce_data(a) for a in args],
                {k: _coerce_data(v) for k, v in kwargs.items()},
            )

        return wrapper

    positions = _exempt_positions(fn, exempt)

    if positions is None:
        # Cannot map positions to names: never guess, keep integer arrays as-is.
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return _call(
                fn,
                [_coerce(a) for a in args],
                {k: _coerce(v) for k, v in kwargs.items()},
            )

        return wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return _call(
            fn,
            [a if i in positions else _coerce_data(a) for i, a in enumerate(args)],
            {k: (v if k in exempt else _coerce_data(v)) for k, v in kwargs.items()},
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
