"""Render the compact master table from surface.json as markdown (item 1).

Run:  .venv-wt/bin/python lab/audit/repo/api/render_surface_table.py > out/surface_table.md
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "surface.json")))


def sig(rec):
    parts = []
    for p in rec.get("params") or []:
        if p.get("has_default"):
            d = p.get("default")
            d = "…" if d == "Ellipsis" else (f'"{d}"' if isinstance(d, str) else repr(d))
            parts.append(f"{p['name']}={d}")
        else:
            parts.append(p["name"])
    return ", ".join(parts)


def keys(rec):
    ks = rec.get("keys")
    if ks is None:
        return f"→ {rec.get('return_kind')}"
    abbrev = {"float": "f", "int": "i", "bool": "b", "str": "s", "none": "∅", "dict": "{}", "1-D[f]": "1D", "2-D[f]": "2D", "3-D[f]": "3D",
              "1-D[i]": "1Di", "1-D[u]": "1Du", "1-D[b]": "1Db", "list[list]": "LL", "list[num]": "L", "list[str]": "Ls", "list[dict]": "L{}",
              "list[array]": "La", "list[bool]": "Lb", "list[]": "L0", "list[mixed]": "Lm"}
    return " ".join(f"`{k}`:{abbrev.get(v, v)}" for k, v in ks.items())


rows = ["| function | family | signature (compiled) | returned keys : kind | docstring first line |", "|---|---|---|---|---|"]
for name, rec in sorted(S.items()):
    fam = rec.get("family") or ""
    first = (rec.get("doc_first_line") or "").replace("|", "\\|")
    ks = keys(rec).replace("|", "\\|")
    rows.append(f"| `{name}` | {fam} | `{sig(rec)}` | {ks} | {first[:90]} |")
print("\n".join(rows))
