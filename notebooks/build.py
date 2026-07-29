"""Build .ipynb notebooks from the plain-Python sources in this directory.

Each ``*_src.py`` file is a normal, runnable Python script divided into cells by
``# %% [markdown]`` and ``# %%`` markers (the Jupytext "percent" convention).
Keeping the source as real Python means the notebooks are diffable, greppable,
and — crucially — *executable in CI* without a notebook runner: ``python
notebooks/tour_src.py`` runs the same code the notebook contains.

    python notebooks/build.py          # regenerate every .ipynb

The generated notebooks carry no outputs. They are meant to be run by the
reader (locally or in Colab), and empty outputs keep the repository small and
the diffs readable.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

# Prepended to every notebook: installs tsecon when running on Colab, and is a
# no-op on a machine that already has it.
INSTALL_CELL = """\
# Colab/one-click setup: installs tsecon if it is missing, no-op otherwise.
try:
    import tsecon
except ImportError:
    %pip install -q tsecon matplotlib pandas
    import tsecon
print("tsecon", tsecon.__version__)\
"""


def parse_cells(text: str) -> list[tuple[str, str]]:
    """Split percent-format source into (kind, body) cells."""
    cells: list[tuple[str, str]] = []
    kind, buf = "code", []
    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            if buf:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = "markdown", []
        elif line.startswith("# %%"):
            if buf:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = "code", []
        else:
            buf.append(line)
    if buf:
        cells.append((kind, "\n".join(buf).strip("\n")))
    # markdown cells are written as comments in the .py; strip the leading "# "
    out = []
    for kind, body in cells:
        if kind == "markdown":
            body = "\n".join(
                ln[2:] if ln.startswith("# ") else ("" if ln.strip() == "#" else ln)
                for ln in body.splitlines()
            )
        if body.strip():
            out.append((kind, body))
    return out


def to_notebook(cells: list[tuple[str, str]]) -> dict:
    nb_cells = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": INSTALL_CELL.splitlines(keepends=True),
        }
    ]
    for kind, body in cells:
        src = body.splitlines(keepends=True)
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
        else:
            nb_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": src,
                }
            )
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    built = 0
    for src in sorted(HERE.glob("*_src.py")):
        cells = parse_cells(src.read_text(encoding="utf-8"))
        nb = to_notebook(cells)
        out = HERE / (src.name.replace("_src.py", ".ipynb"))
        out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out.name}  ({len(cells)} cells)")
        built += 1
    if not built:
        print("no *_src.py files found")


if __name__ == "__main__":
    main()
