"""Tally the reference-library versions recorded in fixtures/*.json `_meta`.

usage: python scripts/fixture_provenance.py [fixtures_dir]   (default: fixtures/)

The validation matrix's Provenance table is this script's output.
"""
import collections
import json
import pathlib
import re
import sys

REFS = {
    "statsmodels": r"statsmodels", "NumPy": r"numpy", "SciPy": r"scipy", "arch": r"arch",
    "linearmodels": r"linearmodels", "scikit-learn": r"scikit[-_]learn|sklearn",
    "ArviZ": r"arviz", "pandas": r"pandas", "statsforecast": r"statsforecast",
    "mapie": r"mapie", "Python": r"python", "R": r"^r$|r_version",
}
VER = re.compile(r"\d+\.\d+(?:\.\d+)?")


def walk(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v, path)
    else:
        yield path, obj


root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent / "fixtures"
files = sorted(root.glob("*.json"))
tally = {name: collections.Counter() for name in REFS}
with_meta = 0
for f in files:
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    meta = d.get("_meta") if isinstance(d, dict) else None
    if meta is None:
        continue
    with_meta += 1
    seen = {}
    for path, val in walk(meta):
        key = path[-1].lower() if path else ""
        for name, pat in REFS.items():
            if re.search(pat, key) and isinstance(val, str):
                m = VER.search(val)
                if m and name not in seen:
                    seen[name] = m.group(0)
    for name, v in seen.items():
        tally[name][v] += 1
print(f"{len(files)} JSON files, {with_meta} with a _meta block")
for name, c in tally.items():
    if c:
        print(f"{name:13s} total {sum(c.values()):3d}: " + ", ".join(f"{v} ({n})" for v, n in c.most_common()))
