#!/usr/bin/env bash
# Credential-pattern scan over the ENTIRE git history (every ref, every
# commit, full patch text) plus unreachable/dangling blobs — repo audit,
# security sweep. Prints COUNTS and locations only; never prints a match.
#
#   bash lab/audit/repo/security/scan_history.sh            # from the repo root
#
# Exit code 1 if any pattern matched anywhere.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "commits (all refs): $(git log --all --oneline | wc -l)"
git log --all -p --no-color > "$TMP/history.patch"
echo "patch text: $(wc -c < "$TMP/history.patch") bytes"

# Provider-shaped tokens and generic credential assignments. Each pattern is
# specific enough that a hit is worth a human look; the generic ones are
# deliberately loose and get a context line (path + first 80 chars) so a
# false positive can be dismissed without printing any value.
PATTERNS=(
  'AKIA[0-9A-Z]{16}'                                        # AWS access key id
  '(?i)aws_secret_access_key\s*[=:]'                        # AWS secret assignment
  'ghp_[A-Za-z0-9]{36}'                                     # GitHub classic PAT
  'github_pat_[A-Za-z0-9_]{22,}'                            # GitHub fine-grained PAT
  'gh[osu]_[A-Za-z0-9]{36}'                                 # GitHub OAuth/server/user tokens
  'pypi-AgEIcHlwaS5vcmc'                                    # PyPI API token prefix (base64 of "pypi.org")
  'pypi-[A-Za-z0-9_-]{60,}'                                 # PyPI token, generic length
  '-----BEGIN (RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY'    # private key blocks
  'xox[baprs]-[A-Za-z0-9-]{10,}'                            # Slack tokens
  'hooks\.slack\.com/services/'                             # Slack webhooks
  'sk-ant-[A-Za-z0-9_-]{20,}'                               # Anthropic API key
  'sk-[A-Za-z0-9]{32,}'                                     # OpenAI-style key
  'AIza[0-9A-Za-z_-]{35}'                                   # Google API key
  'https?://[^/\s:@]+:[^/\s:@]+@[^\s/]+'                    # URL with embedded credentials
  '(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd)\s*[=:]\s*["'"'"'][^"'"'"'\s]{8,}'
  '(?i)\btoken\s*[=:]\s*["'"'"'][A-Za-z0-9_\-]{16,}'
  '(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}'
  '(?i)(HF_TOKEN|OPENAI_API_KEY|ANTHROPIC_API_KEY|AWS_SECRET_ACCESS_KEY|PYPI_TOKEN|TWINE_PASSWORD)\s*='
)

total=0
for pat in "${PATTERNS[@]}"; do
  n=$(grep -cP -- "$pat" "$TMP/history.patch" || true)
  printf 'history  hits=%-4s pattern=%s\n' "$n" "$pat"
  total=$((total + n))
done

# Sensitive file paths ever committed (any ref).
echo "--- sensitive paths ever committed ---"
git log --all --name-only --format= | sort -u \
  | grep -iE '(^|/)(\.env[^/]*|\.pypirc|\.netrc|\.npmrc|\.git-credentials|id_rsa[^/]*|id_ed25519[^/]*|[^/]*\.(pem|key|p12|pfx)|settings\.local\.json|credentials[^/]*|secrets?[^/]*)$' \
  || echo "(none)"

# Unreachable blobs (dangling objects survive a history rewrite that dropped a file).
echo "--- unreachable blobs ---"
git fsck --unreachable --no-reflogs 2>/dev/null | awk '$2=="blob"{print $3}' > "$TMP/blobs.txt"
echo "unreachable blobs: $(wc -l < "$TMP/blobs.txt")"
bhits=0
while read -r b; do
  if git cat-file -p "$b" 2>/dev/null | grep -qP 'AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}|pypi-AgEI|-----BEGIN [A-Z ]*PRIVATE KEY|xox[baprs]-|sk-ant-[A-Za-z0-9_-]{20,}|https?://[^/\s:@]+:[^/\s:@]+@'; then
    bhits=$((bhits + 1)); echo "HIT in unreachable blob $b"
  fi
done < "$TMP/blobs.txt"
echo "unreachable-blob hits: $bhits"
total=$((total + bhits))

echo "--- stashes: $(git stash list | wc -l) ---"
echo "TOTAL HITS: $total"
[ "$total" -eq 0 ]
