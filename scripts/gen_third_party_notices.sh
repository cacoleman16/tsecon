#!/usr/bin/env bash
# Regenerate bindings/python/THIRD-PARTY-NOTICES.md — the licence texts and
# copyright notices of every crate statically linked into tsecon._core —
# with cargo-about, using about.toml (the accepted-licence gate) and
# about.hbs (the layout) at the repository root. The file ships in every
# wheel and sdist through pyproject.toml's `license-files`; CI regenerates
# it and fails if the committed copy is stale.
#
#   scripts/gen_third_party_notices.sh          # regenerate in place
#   scripts/gen_third_party_notices.sh --check  # exit 1 if the committed copy is stale
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
  if ! diff -q "$TMP" "$OUT" >/dev/null; then
    echo "$OUT is stale: run scripts/gen_third_party_notices.sh and commit the result" >&2
    diff "$TMP" "$OUT" | head -40 >&2 || true
    exit 1
  fi
  echo "$OUT is current"
else
  cargo about generate --config ../../about.toml ../../about.hbs -o "$OUT"
  echo "wrote bindings/python/$OUT"
fi
