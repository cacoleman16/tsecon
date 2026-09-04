#!/usr/bin/env bash
# Full Python suite, logged, with per-test outcomes and durations captured
# (repo audit, tests sweep, items 1 and 2).
#
#   lab/audit/repo/tests/run_suite.sh <label> [extra pytest args...]
#
# Writes lab/audit/repo/tests/out/run_<label>.{log,xml}. The log is the -rA
# summary (every outcome line plus captured stdout of passing tests, so
# printed numbers can be diffed between runs); the XML is junit with a per-test
# `time` attribute, which is what durations_report.py and diff_runs.py read.
#
# Run from the repo root of the worktree with the extension already built
# into .venv-wt (maturin develop --release). Never pipe this through
# head/tail; the exit code is pytest's.
set -u
label="${1:?label}"; shift
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../../.." && pwd)"
out="$here/out"
mkdir -p "$out"
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
start=$(date +%s.%N)
.venv-wt/bin/python -m pytest bindings/python/tests -q -p no:cacheprovider \
  --durations=40 -rA --junitxml="$out/run_${label}.xml" "$@" > "$out/run_${label}.log" 2>&1
rc=$?
end=$(date +%s.%N)
printf 'wall_seconds=%.1f exit=%d\n' "$(echo "$end - $start" | bc)" "$rc" | tee -a "$out/run_${label}.log"
exit $rc
