#!/usr/bin/env bash
# Rust line/region coverage for the tsecon workspace.
#
# Usage:
#   scripts/coverage.sh              # per-crate summary to stdout
#   scripts/coverage.sh --html       # also write target/llvm-cov/html/index.html
#   scripts/coverage.sh --lcov       # also write target/coverage/lcov.info
#   scripts/coverage.sh -- -p tsecon-var   # extra args passed to cargo llvm-cov
#
# Prerequisites (one time):
#   rustup component add llvm-tools-preview
#   cargo install cargo-llvm-cov
#
# NOTE (macOS): the `tsecon-python` crate's test binary links libpython and
# aborts under the coverage harness (SIGABRT), which poisons the whole run.
# It is excluded here for the same reason it is excluded from `cargo test`.
# The Python bindings are covered separately by `pytest bindings/python/tests`.
#
# A full clean run takes roughly 6-10 minutes on an M-series Mac (the
# instrumented build is the bulk of it); re-runs that hit the build cache are
# much faster.
#
# There is deliberately NO coverage threshold here. Coverage is a finder, not a
# target -- a percentage gate creates pressure to write make-work tests.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMMON=(--workspace --exclude tsecon-python)

EXTRA_HTML=0
EXTRA_LCOV=0
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --html) EXTRA_HTML=1; shift ;;
    --lcov) EXTRA_LCOV=1; shift ;;
    --) shift; PASSTHROUGH+=("$@"); break ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

echo "==> cleaning stale coverage profraw data"
cargo llvm-cov clean --workspace

echo "==> running instrumented workspace tests"
cargo llvm-cov "${COMMON[@]}" ${PASSTHROUGH+"${PASSTHROUGH[@]}"} --summary-only

if [[ "$EXTRA_HTML" == "1" ]]; then
  echo "==> writing HTML report"
  cargo llvm-cov report "${COMMON[@]}" --html
  echo "    target/llvm-cov/html/index.html"
fi

if [[ "$EXTRA_LCOV" == "1" ]]; then
  echo "==> writing lcov report"
  mkdir -p target/coverage
  cargo llvm-cov report "${COMMON[@]}" --lcov --output-path target/coverage/lcov.info
  echo "    target/coverage/lcov.info"
fi
