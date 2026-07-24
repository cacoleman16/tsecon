"""A generic, honest renderer for any estimator's output.

The six bespoke families (VAR, LP, GARCH, ARIMA, predictive-regression, DSGE)
have hand-written ``.summary()`` output. :class:`Result` fills the gap for the
other ~116 functions: it renders *any* returned ``dict`` as an aligned,
structural report — scalars in a key/value block, small vectors inline, small
matrices as shaped tables, large arrays honestly summarized by shape and range,
nested dicts one level deep. It never invents or hides a key, and its header
says "generic Result" so it is never mistaken for a bespoke model summary.

:func:`summarize` is the single front door: pass it any tsecon output. A
bespoke results object is returned unchanged (you keep the good summary); a
plain dict becomes a :class:`Result`. It deliberately does **not** guess a
bespoke class from a raw dict — the raw dict has already discarded the labels,
lags/trend, and original data those summaries need, and shape-guessing collides
(a three-vector dict could be ``ols`` or ``var_fit``). Silent mislabeling is
worse than a clean generic render, so the primary path to a full bespoke object
stays its constructor (``tsecon.results.var_fit(data, ...)``), which has the data.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._base import Results, fmt_row, rule

__all__ = ["Result", "summarize"]

_WIDTH = 68


def _fmt_scalar(x: object) -> str:
    if isinstance(x, bool):
        return "True" if x else "False"
    if x is None:
        return "None"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if v == 0.0:
            return "0"
        return f"{v:.6f}" if 1e-4 <= abs(v) < 1e6 else f"{v:.4e}"
    return str(x)


def _as_float_array(x: object):
    """Return x as a float ndarray, or None if it is not numeric/array-like."""
    if isinstance(x, str) or isinstance(x, Mapping):
        return None
    try:
        arr = np.asarray(x, dtype=float)
    except (ValueError, TypeError):
        return None
    return arr if arr.ndim >= 1 and arr.size > 0 else None


def _is_strlist(x: object) -> bool:
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(e, str) for e in x)


def _vec_line(key: str, arr, labels=None) -> list[str]:
    if arr.size <= 8:
        vals = ", ".join(_fmt_scalar(v) for v in arr.tolist())
        if labels is not None and len(labels) == arr.size:
            body = ", ".join(f"{n}={_fmt_scalar(v)}" for n, v in zip(labels, arr.tolist()))
            return [f"{key}  {arr.shape}  [{body}]"]
        return [f"{key}  {arr.shape}  [{vals}]"]
    head = ", ".join(_fmt_scalar(v) for v in arr.ravel()[:3].tolist())
    return [
        f"{key}  {arr.dtype} {arr.shape}   min {arr.min():.4g}  "
        f"max {arr.max():.4g}  mean {arr.mean():.4g}",
        f"    [first 3: {head}, ...]",
    ]


def _mat_lines(key: str, arr) -> list[str]:
    nr, nc = arr.shape
    if nr <= 10 and nc <= 8:
        cols = [""] + [f"c{j}" for j in range(nc)]
        widths = [6] + [11] * nc
        lines = [f"{key}  {arr.shape}", fmt_row(cols, widths)]
        for i in range(nr):
            cells = [f"r{i}"] + [f"{arr[i, j]:+.5f}" for j in range(nc)]
            lines.append(fmt_row(cells, widths))
        return lines
    return [
        f"{key}  {arr.dtype} {arr.shape}   min {arr.min():.4g}  "
        f"max {arr.max():.4g}  mean {arr.mean():.4g}"
    ]


class Result(Results):
    """A dict subclass that renders any estimator output via ``.summary()``.

    Adds only rendering: ``title`` is an instance attribute, so the key set,
    ``dict(res)``, ``json.dumps``, ``pickle`` and ``**res`` are exactly what the
    compiled function returned.
    """

    _kind = "Result"

    def __init__(self, mapping=(), /, *, title=None, **kw):
        super().__init__(mapping, **kw)
        self._title = None if title is None else str(title)

    def summary(self) -> str:
        title = self._title or "result"
        scalars, strlists, vecs, mats, dicts, others = [], [], [], [], [], []
        # a strlist sibling (e.g. param_names) can label an equal-length vector
        name_pool = [v for v in self.values() if _is_strlist(v)]
        for key, val in self.items():
            if isinstance(val, (bool, int, float, np.integer, np.floating, str)) or val is None:
                scalars.append((key, val))
            elif _is_strlist(val):
                strlists.append((key, val))
            elif isinstance(val, Mapping):
                dicts.append((key, val))
            else:
                arr = _as_float_array(val)
                if arr is None:
                    others.append((key, val))
                elif arr.ndim == 1:
                    vecs.append((key, arr))
                elif arr.ndim == 2:
                    mats.append((key, arr))
                else:
                    others.append((key, arr))

        out = [rule(_WIDTH), f"{title} — generic Result  ({len(self)} fields)", rule(_WIDTH)]
        if scalars:
            kw = max(len(k) for k, _ in scalars)
            vw = max(len(_fmt_scalar(v)) for _, v in scalars)
            out += [f"{k.ljust(kw)}   {_fmt_scalar(v).rjust(vw)}" for k, v in scalars]
        for key, val in strlists:
            out += [rule(_WIDTH, "-"), f"{key}  [{', '.join(val)}]"]
        for key, arr in vecs:
            labels = next((n for n in name_pool if len(n) == arr.size), None)
            out += [rule(_WIDTH, "-"), *_vec_line(key, arr, labels)]
        for key, arr in mats:
            out += [rule(_WIDTH, "-"), *_mat_lines(key, arr)]
        for key, val in dicts:
            out += [rule(_WIDTH, "-"), f"{key}  (dict, {len(val)} entries)"]
            for k, v in val.items():
                out.append(f"    {str(k).ljust(8)} {_fmt_scalar(v)}")
        for key, val in others:
            shp = getattr(val, "shape", None)
            out += [rule(_WIDTH, "-"), f"{key}  {type(val).__name__}" + (f" {shp}" if shp else "")]
        out.append(rule(_WIDTH))
        return "\n".join(out)


def _as_mapping(obj: object) -> dict:
    if isinstance(obj, Mapping):
        return dict(obj)
    return {"value": obj}


def summarize(obj: object, *, title: str | None = None, wrap: str = "auto") -> Results:
    """Return a renderable results object for any tsecon output.

    ``print(tsecon.summarize(tsecon.adf(y)))`` works for every function. A
    bespoke results object (from ``tsecon.results.*``) is returned unchanged;
    a plain ``dict`` becomes a generic :class:`Result`. ``wrap="generic"``
    forces the structural dump even on a bespoke object.
    """
    if wrap == "generic":
        return Result(_as_mapping(obj), title=title)
    if isinstance(obj, Results):
        return obj  # bespoke or already-generic: keep the best summary (idempotent)
    return Result(_as_mapping(obj), title=title)
