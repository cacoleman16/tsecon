"""Fixture provenance drift (repo audit, tests sweep, item 5).

For every fixtures/*.json this reads the provenance block (`_meta`, `meta`,
`_doc`, `_source`, `_note` — the tree uses all five spellings) and compares any
recorded reference-library version against the version installed in the
running interpreter. It also checks that every fixtures/generate_*.py parses,
imports nothing that is missing from this venv (statically, by scanning its
import statements — the generators are NOT executed), and says in its
docstring how to run it (an R-dependent generator must say so).

Run:  .venv-wt/bin/python lab/audit/repo/tests/fixture_meta_drift.py
Out:  lab/audit/repo/tests/out/fixture_drift.json
"""
from __future__ import annotations

import ast
import importlib
import importlib.metadata as md
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
FIX = os.path.join(REPO, "fixtures")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# fixture key -> distribution name
LIBS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "arch": "arch",
    "linearmodels": "linearmodels",
    "sklearn": "scikit-learn",
    "scikit_learn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "skglm": "skglm",
    "mapie": "MAPIE",
    "arviz": "arviz",
    "cvxpy": "cvxpy",
    "clarabel": "clarabel",
    "pandas": "pandas",
}
PROV_KEYS = ("_meta", "meta", "_doc", "_source", "_note")


def installed(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _find_versions(obj, path="", out=None):
    """Recursively find {libname: 'x.y.z'} pairs anywhere in the provenance block."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in LIBS and isinstance(v, str) and re.match(r"^\d+\.\d+", v):
                out[LIBS[kl]] = (v, f"{path}.{k}".lstrip("."))
            elif kl in LIBS and isinstance(v, dict) and isinstance(v.get("version"), str):
                out[LIBS[kl]] = (v["version"], f"{path}.{k}.version".lstrip("."))
            elif kl in {"versions", "reference_versions", "library_versions"} and isinstance(v, dict):
                _find_versions(v, f"{path}.{k}", out)
            elif isinstance(v, (dict, list)):
                _find_versions(v, f"{path}.{k}", out)
    elif isinstance(obj, str):
        # free-text "statsmodels 0.14.4" / "statsmodels==0.14.4" / "statsmodels==0.14.4"
        for m in re.finditer(r"\b(numpy|scipy|statsmodels|arch|linearmodels|scikit-learn|sklearn|skglm|mapie|arviz|cvxpy|clarabel)\s*(?:==|=|v|\s)\s*(\d+\.\d+(?:\.\d+)?)", obj, re.I):
            lib = LIBS.get(m.group(1).lower())
            if lib and lib not in out:
                out[lib] = (m.group(2), f"{path} (text)")
    return out


def fixtures():
    rows = []
    for f in sorted(os.listdir(FIX)):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(FIX, f)))
        prov = None
        prov_key = None
        if isinstance(d, dict):
            for k in PROV_KEYS:
                if k in d:
                    prov, prov_key = d[k], k
                    break
        versions = _find_versions(prov) if prov is not None else {}
        if not versions and isinstance(d, dict):
            # some fixtures put versions at top level or in a 'generator' block
            versions = _find_versions({k: v for k, v in d.items() if k in {"generator", "references", "reference", "python", "versions"}})
        drift = {}
        for lib, (ver, where) in versions.items():
            cur = installed(lib)
            drift[lib] = {"recorded": ver, "installed": cur, "where": where, "same": (cur == ver)}
        rows.append({"file": f, "prov_key": prov_key, "versions": drift})
    return rows


def generators():
    rows = []
    for f in sorted(os.listdir(FIX)):
        if not (f.startswith("generate_") and f.endswith(".py")):
            continue
        path = os.path.join(FIX, f)
        src = open(path, encoding="utf-8").read()
        row = {"file": f, "parses": True, "missing_imports": [], "run_doc": False, "mentions_R": False, "imports_tsecon": False}
        try:
            tree = ast.parse(src, filename=path)
        except SyntaxError as e:
            row["parses"] = False
            row["error"] = str(e)
            rows.append(row)
            continue
        doc = ast.get_docstring(tree) or ""
        head = "\n".join(src.splitlines()[:60])
        row["run_doc"] = bool(re.search(r"(python\S*\s+\S*generate_|\.venv/bin/python|Run:|Usage:|usage:|python -m|Regenerate)", doc + "\n" + head))
        row["mentions_R"] = bool(re.search(r"\bRscript\b|\bfixest\b|\bR\b.{0,40}(package|PATH|install)|requires R\b|BNFILTER_R_DIR", src))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    mods.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
        row["imports_tsecon"] = "tsecon" in mods
        for m in sorted(mods):
            if m in {"tsecon"}:
                continue
            try:
                importlib.import_module(m)
            except Exception as e:  # noqa: BLE001
                row["missing_imports"].append(f"{m} ({type(e).__name__})")
        rows.append(row)
    return rows


def main():
    fx = fixtures()
    gen = generators()
    json.dump({"fixtures": fx, "generators": gen}, open(os.path.join(OUT, "fixture_drift.json"), "w"), indent=1)

    print("== installed reference versions ==")
    for dist in sorted(set(LIBS.values())):
        print(f"  {dist:14s} {installed(dist)}")
    print()
    print(f"== fixtures: {len(fx)} json files ==")
    no_prov = [r["file"] for r in fx if r["prov_key"] is None]
    no_ver = [r["file"] for r in fx if r["prov_key"] is not None and not r["versions"]]
    print(f"  no provenance block at all: {len(no_prov)}  {no_prov}")
    print(f"  provenance block but no library version recorded: {len(no_ver)}  {no_ver}")
    print()
    print("== per-fixture recorded vs installed (only libraries recorded) ==")
    drift_count = Counter()
    for r in fx:
        if not r["versions"]:
            continue
        cells = []
        for lib, v in sorted(r["versions"].items()):
            mark = "=" if v["same"] else "!="
            cells.append(f"{lib} {v['recorded']} {mark} {v['installed']}")
            if not v["same"]:
                drift_count[(lib, v["recorded"], v["installed"])] += 1
        print(f"  {r['file']:40s} [{r['prov_key']}]  " + "; ".join(cells))
    print()
    print("== drift summary (lib, recorded, installed) -> n fixtures ==")
    for (lib, rec, cur), n in sorted(drift_count.items()):
        print(f"  {lib:14s} {rec!s:10s} -> {cur!s:10s}  {n}")
    print()
    print(f"== generators: {len(gen)} ==")
    for g in gen:
        flags = []
        if not g["parses"]:
            flags.append("DOES NOT PARSE: " + g.get("error", ""))
        if g["missing_imports"]:
            flags.append("missing imports: " + ", ".join(g["missing_imports"]))
        if not g["run_doc"]:
            flags.append("no run instructions in docstring/header")
        if g["mentions_R"]:
            flags.append("R-dependent")
        if g["imports_tsecon"]:
            flags.append("imports tsecon")
        print(f"  {g['file']:52s} {'; '.join(flags) if flags else 'ok'}")


if __name__ == "__main__":
    sys.exit(main())
