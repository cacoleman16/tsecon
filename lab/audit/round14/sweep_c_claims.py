"""Sweep C — claims vs reality for the 0.10.0 surface.

Every number the wave's model cards, validation-matrix rows, guide paragraphs,
CHANGELOG entry, ROADMAP section 0 and the Blanchard-Quah replication page
assert about the thirteen new callables (and the `mask=` parameter) must be
reproduced by a committed artifact: a printed line of the Rust property /
golden binaries the cards cite (run with `--nocapture` into
`out/rust_props.txt`), a numeric literal asserted in a committed test, or a
literal in a committed fixture generator or example script.

The finder is mechanical: every numeric literal in the doc spans below that
carries information (a decimal point, an exponent, or three or more
significant digits — years, section numbers and one/two-digit counts are
skipped) is looked up in the corpus. A number no committed artifact carries is
a CANDIDATE, adjudicated by hand in the report.

Run:  .venv/bin/python lab/audit/round14/sweep_c_claims.py
Out:  lab/audit/round14/out/sweep_c.txt, sweep_c.json
      (out/rust_props.txt and out/sweep_c_pytest.txt are produced separately)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CARDS, GUIDE, OUT, REPO, card_section, log  # noqa: E402
from registry import NEW14  # noqa: E402

NUM = re.compile(r"(?<![\w.])-?\d+(?:[_,]\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")


def informative(tok: str) -> bool:
    """Numbers that carry a measurement: a decimal point, an exponent, or at
    least three significant digits. Years and small counts are skipped."""
    t = tok.lstrip("-")
    if re.fullmatch(r"(19|20)\d\d", t):
        return False
    if "e" in t.lower() or "." in t:
        digits = re.sub(r"[^0-9]", "", t.split("e")[0].split("E")[0]).lstrip("0")
        return len(digits) >= 2
    return len(t.replace("_", "").replace(",", "")) >= 3


def spans():
    """(label, text) of every doc surface the wave wrote about the new API."""
    out = []
    for name in NEW14:
        out.append((f"card:{name}", card_section(name)))
    panel = open(os.path.join(CARDS, "panel.md"), encoding="utf-8").read()
    m = re.search(r"^## Unbalanced panels.*?(?=^## Nickell bias)", panel, re.M | re.S)
    out.append(("card:panel-mask", m.group(0)))
    matrix = open(os.path.join(REPO, "docs", "reference", "validation-matrix.md"), encoding="utf-8").read()
    keys = ("var_conditional_forecast", "var_diagnostics", "fmols", "observation mask (`mask=`)",
            "spa_test", "ets_fit", "auto_ets", "unobserved_components", "tvp_regression")
    for line in matrix.splitlines():
        if line.startswith("|") and any(k in line for k in keys):
            out.append(("matrix", line))
    # the guide paragraphs the wave added or rewrote
    rc = subprocess.run(["git", "diff", "-U0", "dde820d..HEAD", "--", "docs/guide", "docs/quickstart.md",
                         "docs/index.md", "docs/examples/replication-blanchard-quah.md",
                         "docs/reference/testing.md"],
                        capture_output=True, text=True, cwd=REPO)
    added = "\n".join(l[1:] for l in rc.stdout.splitlines() if l.startswith("+") and not l.startswith("+++"))
    out.append(("guide+bq(added lines)", added))
    ch = open(os.path.join(REPO, "CHANGELOG.md"), encoding="utf-8").read()
    m = re.search(r"^## \[?0\.10\.0.*?(?=^## \[?0\.9)", ch, re.M | re.S)
    if m:
        out.append(("CHANGELOG 0.10.0", m.group(0)))
    rm = open(os.path.join(REPO, "ROADMAP.md"), encoding="utf-8").read()
    m = re.search(r"^## 0\..*?(?=^## 1\.|^## [A-Z])", rm, re.M | re.S)
    out.append(("ROADMAP section 0", m.group(0) if m else rm[:20000]))
    return out


def corpus():
    """Everything a number may be reproduced by."""
    parts = []
    for path in (os.path.join(OUT, "rust_props.txt"), os.path.join(OUT, "sweep_c_pytest.txt")):
        if os.path.exists(path):
            parts.append(open(path, encoding="utf-8", errors="replace").read())
    for root, _dirs, files in os.walk(os.path.join(REPO, "crates")):
        if os.sep + "tests" not in root and os.sep + "src" not in root:
            continue
        for f in files:
            if f.endswith(".rs"):
                parts.append(open(os.path.join(root, f), encoding="utf-8", errors="replace").read())
    for d in (os.path.join(REPO, "bindings", "python", "tests"),
              os.path.join(REPO, "docs", "examples"),
              os.path.join(REPO, "fixtures"),
              os.path.join(REPO, "scripts"),
              os.path.join(REPO, "benchmarks")):
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.endswith((".py", ".json")):
                    parts.append(open(os.path.join(root, f), encoding="utf-8", errors="replace").read())
    return "\n".join(parts)


def sigfigs(tok: str) -> int:
    body = tok.lstrip("-").split("e")[0].split("E")[0].replace(".", "").lstrip("0")
    return max(len(body.rstrip("0")) or 1, 1)


def rounds_to(hay: str, tok: str):
    """A committed number that ROUNDS to `tok` (the cards quote 1469.18 for a
    fixture's 1469.176115754801). Returns the first such number, or None."""
    try:
        want = float(tok.replace(",", "").replace("_", ""))
    except ValueError:
        return None
    sig = sigfigs(tok)
    if sig < 2:
        return None
    # search on the leading significant digits, then parse the full number
    lead = f"{want:.{sig - 1}e}"
    mant, exp = lead.split("e")
    mant = mant.rstrip("0").rstrip(".")
    digits = mant.lstrip("-").replace(".", "")
    if len(digits) < 2:
        return None
    pat = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
    for m in pat.finditer(hay):
        try:
            v = float(m.group(0))
        except ValueError:
            continue
        if v == 0.0:
            continue
        if float(f"{v:.{sig}g}") == float(f"{want:.{sig}g}") and v != want:
            return m.group(0)
    return None


# the extracted card blocks are written here to be executed; they are
# regenerated on every run and are not committed
SCRATCH = os.path.join(OUT, "scratch")


def run_card_examples(fh):
    """Every ```python block of the new card sections is executed and the
    lines its trailing `# ` comments claim as output are diffed against what
    it actually prints (round 13's "the block itself is the artifact" rule)."""
    os.makedirs(SCRATCH, exist_ok=True)
    n_blocks = n_lines = n_bad = 0
    for label, text in spans():
        if not label.startswith("card:"):
            continue
        for i, blk in enumerate(re.findall(r"```python\n(.*?)```", text, re.S)):
            if "tsecon." not in blk:
                continue
            path = os.path.join(SCRATCH, f"{label.replace(':', '_')}_{i}.py")
            open(path, "w", encoding="utf-8").write(blk)
            p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=900)
            n_blocks += 1
            if p.returncode != 0:
                n_bad += 1
                log(fh, f"[{label} block {i}] FAILED TO RUN: {p.stderr.strip().splitlines()[-1][:160]}")
                continue
            got = p.stdout.splitlines()
            # a trailing comment is an OUTPUT CLAIM when it carries a digit and
            # is not a numbered step ("1. ...") or a prose aside; a claim is
            # met when a printed line equals it or is its prefix (the cards
            # annotate a printed line with " -> what it means")
            # an output claim is a `# ` comment in the run that FOLLOWS a
            # print(...) statement (the cards' "expected output" convention);
            # a comment before the code is prose, not a claim
            expected, after_print = [], False
            for line in blk.splitlines():
                st = line.strip()
                if st.startswith("# "):
                    if after_print and re.search(r"\d", st) and not re.match(r"#\s*\d+\.\s", st):
                        expected.append(st[2:])
                    continue
                if st:
                    after_print = st.startswith("print(") or st.endswith(")") and "print(" in st
            def met(e):
                return any(g == e or (g and e.startswith(g)) for g in got)
            claimed = [e for e in expected if met(e)]
            missing = [e for e in expected if not met(e)]
            n_lines += len(claimed)
            if missing:
                n_bad += len(missing)
                for e in missing:
                    log(fh, f"[{label} block {i}] EXPECTED LINE NOT PRINTED: {e!r}")
    log(fh, f"\ncard examples: {n_blocks} runnable blocks executed, "
            f"{n_lines} claimed output lines reproduced verbatim, {n_bad} not")


def main():
    fh = open(os.path.join(OUT, "sweep_c.txt"), "w")
    hay = corpus()
    log(fh, f"corpus: {len(hay):,} characters "
            f"(rust_props + pytest -s + every crate src/tests + python tests + examples + "
            f"fixtures + scripts + benchmarks)")
    report = {}
    total = found = 0
    for label, text in spans():
        toks = [t for t in NUM.findall(text) if informative(t)]
        uniq = sorted(set(toks), key=toks.index)
        miss = []
        for t in uniq:
            forms = {t, t.replace(",", ""), t.replace("_", ""), t.lstrip("-")}
            if t.startswith("0."):
                forms.add(t[1:])
            if re.fullmatch(r"-?\d+\.0", t):
                forms.add(t[:-2])
            # 1e-8 <-> 1e-08 <-> 1E-8
            for f in list(forms):
                mm = re.fullmatch(r"(-?\d+(?:\.\d+)?)[eE]([-+]?)(\d+)", f)
                if mm:
                    sign = mm.group(2) or ""
                    forms.add(f"{mm.group(1)}e{sign}{int(mm.group(3))}")
                    forms.add(f"{mm.group(1)}e{sign}{int(mm.group(3)):02d}")
                    forms.add(f"{mm.group(1)}E{sign}{int(mm.group(3))}")
            if not any(f in hay for f in forms):
                miss.append(t)
        total += len(uniq)
        found += len(uniq) - len(miss)
        rounded = {}
        for t in list(miss):
            hit = rounds_to(hay, t)
            if hit is not None:
                rounded[t] = hit
                miss.remove(t)
        found += len(rounded)
        report[label] = {"numbers": len(uniq), "missing": miss, "rounded": rounded}
        if rounded:
            log(fh, f"[{label}] reproduced by rounding a committed number: "
                    + ", ".join(f"{k} <- {v}" for k, v in rounded.items()))
        if miss:
            log(fh, f"[{label}] {len(uniq)} informative numbers, {len(miss)} NOT found in the corpus: {miss}")
        else:
            log(fh, f"[{label}] {len(uniq)} informative numbers, all reproduced")
    log(fh, f"\nTOTAL: {total} informative numbers across {len(report)} spans; "
            f"{found} found in a committed artifact, {total - found} candidates")
    run_card_examples(fh)
    json.dump(report, open(os.path.join(OUT, "sweep_c.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
