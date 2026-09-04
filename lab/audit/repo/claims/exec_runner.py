"""Subprocess runner for sweep_exec.py: execute one page's python blocks in a
single namespace, capturing per-block stdout, with a per-block alarm.

Reads a JSON list of {"idx", "line", "code"} on stdin; writes a JSON list of
{"idx", "stdout", "error", "seconds", "inline"} on stdout. ``inline`` holds
the results of ``expr   # -> literal`` / ``print(...)   # literal`` comment
checks found on single-line statements.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import signal
import sys
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")

INLINE = re.compile(r"^(?P<code>[^#\n]+?)\s+#\s*(?:->|=>)?\s*(?P<lit>\S.*?)\s*$")
PER_BLOCK_SECONDS = int(os.environ.get("CLAIMS_BLOCK_SECONDS", "300"))


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout(f"block exceeded {PER_BLOCK_SECONDS}s")


def run_inline_checks(code, ns):
    """For single-line statements carrying a trailing ``# -> literal`` or a
    ``print(...)  # literal`` comment, evaluate and compare."""
    out = []
    for lineno, line in enumerate(code.split("\n"), 1):
        m = INLINE.match(line)
        if not m:
            continue
        src, lit = m.group("code").strip(), m.group("lit").strip()
        if not re.search(r"\d|True|False|None|\"|'", lit):
            continue  # a prose comment, not a value claim
        try:
            tree = ast.parse(src, mode="exec")
        except SyntaxError:
            continue
        if len(tree.body) != 1:
            continue
        node = tree.body[0]
        try:
            if isinstance(node, ast.Expr):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    val = eval(compile(ast.Expression(node.value), "<inline>", "eval"), ns)
                printed = buf.getvalue().strip()
                shown = printed if printed else (repr(val) if val is not None else "")
            else:
                continue
        except Exception as exc:  # noqa: BLE001
            shown = f"<error {type(exc).__name__}: {exc}>"
        out.append({"line": lineno, "code": src, "expected": lit, "actual": shown})
    return out


def main():
    blocks = json.load(sys.stdin)
    ns = {"__name__": "__main__"}
    results = []
    signal.signal(signal.SIGALRM, _alarm)
    for b in blocks:
        buf = io.StringIO()
        err = None
        t0 = time.time()
        signal.alarm(PER_BLOCK_SECONDS)
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(b["code"], f"<block@{b['line']}>", "exec"), ns)
        except Timeout as exc:
            err = f"Timeout: {exc}"
        except BaseException as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                err = f"{type(exc).__name__}"
        finally:
            signal.alarm(0)
        inline = []
        if err is None:
            try:
                signal.alarm(PER_BLOCK_SECONDS)
                inline = run_inline_checks(b["code"], ns)
            except BaseException:  # noqa: BLE001
                inline = []
            finally:
                signal.alarm(0)
        results.append({"idx": b["idx"], "stdout": buf.getvalue(), "error": err, "seconds": round(time.time() - t0, 2), "inline": inline})
    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
