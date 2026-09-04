"""Executable-claims sweep: every fenced ``python`` block in the cookbook, the
quickstart, the index, the README, the guide chapters and the model cards is
run on the installed wheel (sequentially per page, one namespace) and, where a
pasted-output block follows it, the printed output is compared to the pasted
one — numbers at the printed precision, text exactly.

Statuses
  MATCH      pasted output reproduced at printed precision
  MISMATCH   at least one token differs (the diff is in the JSON)
  NO-OUTPUT  block ran, nothing pasted to compare (inline ``# -> x`` checks may apply)
  ERROR      block raised (the error is recorded; preview-labelled blocks are expected to)
  TIMEOUT    block exceeded the per-block budget
  SKIP       needs a data file / optional extra that is absent (reason recorded)

Run:  .venv-wt/bin/python lab/audit/repo/claims/sweep_exec.py [page ...]
Out:  out/sweep_exec.json, out/sweep_exec.log
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, OUT, REPO, log, split_blocks  # noqa: E402

PAGES = (
    ["README.md", "docs/index.md", "docs/quickstart.md"]
    + sorted(os.path.relpath(p, REPO) for p in __import__("glob").glob(os.path.join(REPO, "docs/cookbook/*.md")))
    + sorted(os.path.relpath(p, REPO) for p in __import__("glob").glob(os.path.join(REPO, "docs/guide/[0-9]*.md")))
    + sorted(os.path.relpath(p, REPO) for p in __import__("glob").glob(os.path.join(REPO, "docs/reference/model-cards/*.md")))
    + ["docs/reference/results.md"]
)
OUTPUT_LANGS = {"", "text", "txt", "console", "output", "plaintext"}
NUM = re.compile(r"[-+−]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+−]?(?:nan|inf|NaN|Inf)")
PREVIEW = re.compile(r"preview", re.I)
PYTHON = os.environ.get("CLAIMS_PYTHON", os.path.join(REPO, ".venv-wt", "bin", "python"))


def tokens(s):
    return s.replace("−", "-").split()


def num_equal(exp, act):
    """Compare at the precision the expected value was printed with."""
    e, a = exp.replace("−", "-"), act.replace("−", "-")
    try:
        ev, av = float(e), float(a)
    except ValueError:
        return e == a
    if e.lower() in ("nan", "-nan") or a.lower() in ("nan", "-nan"):
        return e.lower().strip("-+") == a.lower().strip("-+")
    if "e" in e.lower():
        mant = e.lower().split("e")[0]
        digits = len(mant.split(".")[1]) if "." in mant else 0
        return abs(ev - av) <= 0.51 * abs(ev) * 10 ** (-digits) + 1e-300 if ev != 0 else abs(av) < 1e-12
    digits = len(e.split(".")[1]) if "." in e else 0
    return abs(ev - av) <= 0.51 * 10 ** (-digits)


def compare(expected, actual):
    et, at = tokens(expected), tokens(actual)
    diffs = []
    if len(et) != len(at):
        diffs.append(f"token count {len(et)} pasted vs {len(at)} actual")
    for i, (e, a) in enumerate(zip(et, at)):
        if e == a:
            continue
        if NUM.fullmatch(e) and NUM.fullmatch(a):
            if not num_equal(e, a):
                diffs.append(f"#{i}: {e} vs {a}")
        else:
            # numbers embedded in text tokens (e.g. "p=0.031," or "[0.395")
            en, an = NUM.findall(e), NUM.findall(a)
            if en and an and len(en) == len(an) and NUM.sub("#", e) == NUM.sub("#", a):
                for x, y in zip(en, an):
                    if not num_equal(x, y):
                        diffs.append(f"#{i}: {e} vs {a}")
                        break
            else:
                diffs.append(f"#{i}: {e!r} vs {a!r}")
    return diffs


def page_blocks(rel):
    text = open(os.path.join(REPO, rel), encoding="utf-8").read()
    lines = text.split("\n")
    seq = list(split_blocks(text))
    blocks = []
    for n, (kind, lang, start, body) in enumerate(seq):
        if kind != "code" or lang not in ("python", "py"):
            continue
        # a pasted-output block: the next block, if only whitespace prose sits between
        expected = None
        if n + 2 < len(seq) and seq[n + 1][0] == "prose" and seq[n + 1][3].strip() == "" and seq[n + 2][0] == "code" and seq[n + 2][1] in OUTPUT_LANGS:
            expected = seq[n + 2][3]
        elif n + 1 < len(seq) and seq[n + 1][0] == "code" and seq[n + 1][1] in OUTPUT_LANGS:
            expected = seq[n + 1][3]
        above = "\n".join(lines[max(0, start - 4) : start - 1])
        preview = bool(PREVIEW.search(above))
        blocks.append({"idx": len(blocks), "line": start, "code": body, "expected": expected, "preview": preview})
    return blocks


def run_page(rel):
    blocks = page_blocks(rel)
    if not blocks:
        return rel, []
    payload = [{"idx": b["idx"], "line": b["line"], "code": b["code"]} for b in blocks]
    try:
        proc = subprocess.run(
            [PYTHON, os.path.join(HERE, "exec_runner.py")],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=int(os.environ.get("CLAIMS_PAGE_SECONDS", "2400")),
            env={**os.environ, "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"},
        )
        results = json.loads(proc.stdout) if proc.stdout.strip() else []
        runner_err = proc.stderr[-800:] if not results else ""
    except subprocess.TimeoutExpired:
        results, runner_err = [], "page-level timeout"
    by_idx = {r["idx"]: r for r in results}
    out = []
    for b in blocks:
        r = by_idx.get(b["idx"])
        rec = {"file": rel, "line": b["line"], "preview": b["preview"], "has_output": b["expected"] is not None}
        if r is None:
            rec.update(status="TIMEOUT" if not runner_err else "ERROR", detail=runner_err or "no result (page timed out earlier)")
        elif r["error"]:
            err = r["error"]
            if err.startswith("Timeout"):
                rec.update(status="TIMEOUT", detail=err)
            elif "FileNotFoundError" in err or ("No such file" in err):
                rec.update(status="SKIP", detail=f"needs a data file: {err}")
            elif "ModuleNotFoundError" in err or "ImportError" in err:
                rec.update(status="SKIP", detail=f"needs an optional extra: {err}")
            else:
                rec.update(status="ERROR", detail=err)
        elif b["expected"] is not None:
            diffs = compare(b["expected"], r["stdout"])
            rec.update(status="MATCH" if not diffs else "MISMATCH", detail="; ".join(diffs[:8]), actual=r["stdout"], expected=b["expected"])
        else:
            rec.update(status="NO-OUTPUT", detail="", actual=r["stdout"])
        if r is not None:
            rec["seconds"] = r["seconds"]
            bad = [c for c in r.get("inline", []) if compare(c["expected"], c["actual"])]
            rec["inline_checks"] = len(r.get("inline", []))
            rec["inline_mismatch"] = bad
        out.append(rec)
    return rel, out


def main():
    pages = sys.argv[1:] or PAGES
    fh = open(os.path.join(OUT, "sweep_exec.log"), "w")
    all_out = []
    workers = int(os.environ.get("CLAIMS_WORKERS", "4"))
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for rel, recs in ex.map(run_page, pages):
            for r in recs:
                tag = r["status"]
                extra = f" ({r['detail'][:160]})" if r.get("detail") else ""
                inl = f" inline={r.get('inline_checks', 0)}" + (f" INLINE-MISMATCH={r['inline_mismatch']}" if r.get("inline_mismatch") else "") if r.get("inline_checks") else ""
                log(fh, f"{rel}:{r['line']} {tag}{' [preview]' if r['preview'] else ''}{extra}{inl}")
            all_out.extend(recs)
    json.dump(all_out, open(os.path.join(OUT, "sweep_exec.json"), "w"), indent=1)
    from collections import Counter

    c = Counter(r["status"] for r in all_out)
    log(fh, "TOTAL", len(all_out), dict(sorted(c.items())))
    log(fh, "with pasted output:", sum(1 for r in all_out if r["has_output"]), "matched:", c["MATCH"], "mismatched:", c["MISMATCH"])
    fh.close()


if __name__ == "__main__":
    main()
