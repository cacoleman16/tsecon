#!/bin/bash
# Round-8 build helper: builds the worktree's bindings into .venv-wt with an
# isolated target dir, capturing the exit status explicitly (no cmd|tail).
set -u
cd "$(dirname "$0")/../.." || exit 9
export CARGO_TARGET_DIR="$PWD/target-wt"
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_TEST_DEBUG=0
export CARGO_PROFILE_DEV_DEBUG=0
export VIRTUAL_ENV="$PWD/.venv-wt"
export PATH="$PWD/.venv-wt/bin:$PATH"
/home/user/tsecon/.venv/bin/maturin develop --release -m bindings/python/Cargo.toml \
    > scratch/round8/maturin.log 2>&1
status=$?
echo "MATURIN_EXIT=$status"
tail -5 scratch/round8/maturin.log
exit $status
