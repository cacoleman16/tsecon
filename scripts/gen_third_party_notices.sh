#!/usr/bin/env bash
# Regenerate bindings/python/THIRD-PARTY-NOTICES.md — the licence texts and
# copyright notices of every crate statically linked into tsecon._core —
# with cargo-about, using about.toml (the accepted-licence gate) and
# about.hbs (the layout) at the repository root. The file ships in every
# wheel and sdist through pyproject.toml's `license-files`; CI regenerates
# it and fails if the committed copy is stale. Both modes also run
# scripts/gen_license_inventory.py, which keeps the per-crate version table in
# THIRD-PARTY-LICENSES.md in step with the same Cargo.lock.
#
#   scripts/gen_third_party_notices.sh          # regenerate in place
#   scripts/gen_third_party_notices.sh --check  # exit 1 if either committed copy is stale
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION=0.9.2
if ! cargo about --version 2>/dev/null | grep -q "$VERSION"; then
  cargo install cargo-about --locked --version "$VERSION" --features cli
fi
cd "$ROOT/bindings/python"
OUT=THIRD-PARTY-NOTICES.md
if [[ "${1:-}" == "--check" ]]; then
  TMP="$(mktemp)"
  trap 'rm -f "$TMP"' EXIT
  cargo about generate --config ../../about.toml ../../about.hbs -o "$TMP"
  stale=0
  if ! diff -q "$TMP" "$OUT" >/dev/null; then
    echo "bindings/python/$OUT is stale: run scripts/gen_third_party_notices.sh and commit the result" >&2
    diff "$TMP" "$OUT" | head -40 >&2 || true
    stale=1
  else
    echo "bindings/python/$OUT is current"
  fi
  python3 "$ROOT/scripts/gen_license_inventory.py" --check || stale=1
  exit "$stale"
else
  cargo about generate --config ../../about.toml ../../about.hbs -o "$OUT"
  echo "wrote bindings/python/$OUT"
  python3 "$ROOT/scripts/gen_license_inventory.py"
fi
