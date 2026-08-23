#!/bin/bash
set -u
cd "$(dirname "$0")/../.." || exit 9
export CARGO_TARGET_DIR="$PWD/target-wt"
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_TEST_DEBUG=0
export CARGO_PROFILE_DEV_DEBUG=0
cargo test -p tsecon-copula > scratch/round8/cargo_copula.log 2>&1
s=$?
echo "CARGO_EXIT=$s"
grep -E "test result|running" scratch/round8/cargo_copula.log
exit $s
