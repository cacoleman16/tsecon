"""The Results base class and text-formatting helpers.

The design commitment: **a Results object IS a `dict`.** Every estimator has
always returned a plain dict of documented keys, and that stays true —
`res["params"]` keeps working, as do `json.dumps`, `pickle`, `**res` unpacking,
and `isinstance(res, dict)`. A Results subclass only *adds* rendering
(`.summary()`) and family-specific accessors on top.

That is what makes the richer API additive rather than a breaking change: the
dict contract is preserved forever as a subset of the object.
"""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = ["Results", "rule", "fmt_row", "kv_line", "param_table"]


class Results(dict):
    """A results dict that can also render itself.

    Subclasses override :meth:`summary`. Everything else — indexing, iteration,
    serialisation — is inherited from :class:`dict` unchanged.
    """

    #: Short human label used in the default summary header.
    _kind = "Results"

    def to_dict(self) -> dict:
        """A plain :class:`dict` copy, for code that wants no subclass."""
        return dict(self)

    def summary(self) -> str:
        """A formatted, human-readable summary. Overridden per family."""
        keys = ", ".join(sorted(self))
        return f"{self._kind}({keys})"

    def __repr__(self) -> str:
        # Notebooks and the REPL echo repr(); showing the summary is the whole
        # point of the object. Fall back to the dict repr if a summary raises,
        # so a formatting bug can never make results unprintable.
        try:
            return self.summary()
        except Exception:  # pragma: no cover - defensive
            return dict.__repr__(self)


# --------------------------------------------------------------------------- #
# text helpers — shared so every family's summary looks like the others
# --------------------------------------------------------------------------- #
def rule(width: int = 68, char: str = "=") -> str:
    return char * width


def fmt_row(
    cells: Sequence[object],
    widths: Sequence[int],
    aligns: Sequence[str] | None = None,
    sep: str = "  ",
) -> str:
    """One aligned row. ``aligns`` entries are ``"l"`` or ``"r"``."""
    aligns = aligns or ["l"] * len(cells)
    out = []
    for cell, width, align in zip(cells, widths, aligns):
        text = f"{cell}"
        out.append(text.rjust(width) if align == "r" else text.ljust(width))
    return sep.join(out).rstrip()


def kv_line(pairs: Iterable[tuple[str, object]], sep: str = "    ") -> str:
    """``key value`` pairs on one line, e.g. fit statistics."""
    return sep.join(f"{k} {v}" for k, v in pairs)


def param_table(
    names: Sequence[str],
    values: Sequence[float],
    se: Sequence[float] | None = None,
    tstats: Sequence[float] | None = None,
    *,
    value_label: str = "coef",
    se_label: str = "std err",
    width: int = 56,
) -> list[str]:
    """A coefficient block: name, estimate, and optionally SE and t-statistic.

    Returned as a list of lines so callers can splice it into a larger summary.
    """
    cols = ["param", value_label]
    widths = [14, 13]
    aligns = ["l", "r"]
    if se is not None:
        cols.append(se_label)
        widths.append(12)
        aligns.append("r")
    if tstats is not None:
        cols.append("t")
        widths.append(9)
        aligns.append("r")

    lines = [fmt_row(cols, widths, aligns), "-" * width]
    for i, name in enumerate(names):
        cells: list[object] = [name, f"{values[i]:+.5f}"]
        if se is not None:
            cells.append(f"{se[i]:.5f}")
        if tstats is not None:
            cells.append(f"{tstats[i]:+.2f}")
        lines.append(fmt_row(cells, widths, aligns))
    return lines
