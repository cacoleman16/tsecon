#!/usr/bin/env python3
"""Regenerate the crate table in THIRD-PARTY-LICENSES.md from `cargo metadata`.

The table lists every third-party package in the locked dependency graph
(every package `cargo metadata --locked` reports that is not a workspace
member), sorted by name, with its version and its declared SPDX licence
expression. It is the per-crate inventory that sits beside the generated
licence texts in bindings/python/THIRD-PARTY-NOTICES.md, so it has to move
with Cargo.lock just as those do; scripts/gen_third_party_notices.sh runs this
script in both of its modes.

    scripts/gen_license_inventory.py          # rewrite the table in place
    scripts/gen_license_inventory.py --check  # exit 1 if the committed table is stale

Standard library only: CI runs it with the runner's system Python.
"""

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PATH = ROOT / "THIRD-PARTY-LICENSES.md"
# The generated block: the count sentence, a blank line, then the table up to
# the first line that is not a table row.
BLOCK = re.compile(
    r"^Generated from `cargo metadata` — \d+ third-party packages\.\n\n"
    r"\| Crate \| Version \| License \|\n\|---\|---\|---\|\n(?:\|.*\|\n)*",
    re.M,
)


def render() -> str:
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--locked"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    members = set(meta["workspace_members"])
    rows = sorted(
        (p["name"], p["version"], p.get("license") or "UNKNOWN")
        for p in meta["packages"]
        if p["id"] not in members
    )
    lines = [
        f"Generated from `cargo metadata` — {len(rows)} third-party packages.",
        "",
        "| Crate | Version | License |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {version} | {lic} |" for name, version, lic in rows]
    return "\n".join(lines) + "\n"


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    if len(BLOCK.findall(text)) != 1:
        print(f"{PATH.name}: could not find exactly one generated crate table", file=sys.stderr)
        return 2
    new = BLOCK.sub(lambda _: render(), text)
    if sys.argv[1:] == ["--check"]:
        if new != text:
            print(
                f"{PATH.name} is stale: run scripts/gen_third_party_notices.sh and commit the result",
                file=sys.stderr,
            )
            return 1
        print(f"{PATH.name} is current")
        return 0
    if sys.argv[1:]:
        print(__doc__, file=sys.stderr)
        return 2
    PATH.write_text(new, encoding="utf-8")
    print(f"wrote {PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
