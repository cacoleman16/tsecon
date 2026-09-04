"""Rust tests with no assertion (repo audit, tests sweep, item 3, Rust half).

Walks every `#[test]` function under crates/*/tests and crates/*/src and
reports the ones whose body contains none of: an `assert*!` macro, a `?`,
`.unwrap_err()`, `.expect_err(`, `panic!`, `unreachable!`, a
`#[should_panic]` attribute, or a call to a same-file helper `fn` that itself
asserts. `.unwrap()` / `.expect(` are reported separately: a test that only
unwraps is a valid "does not error" smoke test, but it cannot fail on a wrong
number.

Run:  python3 lab/audit/repo/tests/rust_no_assert.py
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CRATES = os.path.join(REPO, "crates")

# `assert_\w+(` covers the tests/common/mod.rs helpers (assert_rel_close,
# assert_mat_close, assert_abs_close, ...) every crate shares.
ASSERT_RE = re.compile(r"\b(assert(?:_eq|_ne)?!|debug_assert!|assert_abs_diff_eq!|assert_relative_eq!|assert_ulps_eq!|approx::assert|panic!|unreachable!|\.unwrap_err\(|\.expect_err\(|assert_\w+\s*\()")
UNWRAP_RE = re.compile(r"\.unwrap\(\)|\.expect\(")
QMARK_RE = re.compile(r"\?\s*[;.)]|\?\s*$", re.M)


def fn_bodies(src: str):
    """Yield (name, attrs, body) for every fn with a brace-matched body."""
    i = 0
    n = len(src)
    while True:
        m = re.search(r"(?:^|\n)((?:\s*#\[[^\]]*\]\s*\n)*)\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)", src[i:])
        if not m:
            return
        attrs = m.group(1)
        name = m.group(2)
        start = i + m.end()
        # find the opening brace of the body
        brace = src.find("{", start)
        if brace < 0:
            return
        depth = 0
        j = brace
        while j < n:
            c = src[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = src[brace : j + 1]
        yield name, attrs, body
        i = j + 1


def main():
    rows_none = []
    rows_unwrap_only = []
    total = 0
    for root, _d, files in os.walk(CRATES):
        for fn in files:
            if not fn.endswith(".rs"):
                continue
            path = os.path.join(root, fn)
            src = open(path, encoding="utf-8").read()
            fns = list(fn_bodies(src))
            helpers_assert = {name for name, attrs, body in fns if "#[test]" not in attrs and ASSERT_RE.search(body)}
            for name, attrs, body in fns:
                if "#[test]" not in attrs:
                    continue
                total += 1
                if "should_panic" in attrs:
                    continue
                if ASSERT_RE.search(body) or QMARK_RE.search(body):
                    continue
                if any(re.search(rf"\b{h}\s*\(", body) for h in helpers_assert):
                    continue
                rel = os.path.relpath(path, REPO)
                line = src[: src.find(body)].count("\n") + 1
                if UNWRAP_RE.search(body):
                    rows_unwrap_only.append((rel, line, name))
                else:
                    rows_none.append((rel, line, name))
    print(f"#[test] fns scanned: {total}")
    print(f"== no assert / ? / unwrap / expect at all: {len(rows_none)} ==")
    for r in rows_none:
        print("  %s:%d  %s" % r)
    print(f"== only .unwrap()/.expect() (does-not-error smoke tests): {len(rows_unwrap_only)} ==")
    for r in rows_unwrap_only:
        print("  %s:%d  %s" % r)


if __name__ == "__main__":
    sys.exit(main())
