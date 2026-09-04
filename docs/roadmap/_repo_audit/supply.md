# Repository audit — supply chain, dependencies, CI/workflow security

*Audit round 12, sweep 6 of 7. Branch `audit12/supply`, from `origin/main` at
19d308e (tsecon 0.8.0). Date: 2026-09-04.*

## Scope and method

The first eleven audit rounds tested what the estimators compute. This sweep
tests the repository as an artifact: what gets linked into the wheel, what
runs in CI with which credentials, what a release actually ships, and whether
the claims the repository makes about those things are true.

Everything below was run in a fresh worktree with `CARGO_TARGET_DIR=target-wt`
and `CARGO_PROFILE_DEV_DEBUG=0`; no test suite was run (that is another
sweep's job). Commands and outputs are quoted at each finding; full logs are
in the session scratchpad and are not committed.

| Area | What was done |
|---|---|
| Rust dependencies | `cargo install cargo-audit --locked` (v0.22.x) and `cargo audit -f Cargo.lock`; `cargo tree --workspace --duplicates`; `cargo tree --workspace -e no-dev --prefix none \| sort -u`; `cargo tree -i <crate> --target all` for every crate that looked odd; `cargo metadata --format-version 1` for licences and `rust_version`; the crates.io API (`/api/v1/crates/<name>`, 94 distinct names, 1 request/s) for newest version, yank status and last-release date of every third-party package; a read of the `numpy` 0.29.0 crate source (`src/npyffi/{array,numpyconfig}.rs`) for its runtime NumPy checks. |
| Python packaging | `bindings/python/pyproject.toml`, `bindings/python/Cargo.toml`, the pure-Python layer's NumPy usage (`grep -ohE '\bnp\.[A-Za-z_.]+'`), the published 0.6.0 wheel and sdist downloaded from PyPI and listed, the PyPI provenance endpoint, and `pip-audit 2.10.1 -r` over the exact unpinned set the workflows install (resolved fresh, as CI would). |
| GitHub Actions | Every `uses:` in the three workflows resolved with `git ls-remote` to the commit it points at today; every `permissions:` block; a grep for `pull_request_target`, `secrets.`, and `${{ github.event.* }}` inside `run:`; the concurrency groups; the cache surfaces; PR #18 via the GitHub API (`get`, `get_files`, `get_check_runs`). The proxy refused `/environments`, `/actions/permissions*`, `/branches/main/protection` and `/rulesets`, so environment protection and the fork-approval setting are recorded under **Open**. |
| Licence inventory | `THIRD-PARTY-LICENSES.md` reconciled row-by-row against `cargo metadata` (name, version, licence string); every licence expression tokenised and checked against the permissive set; the wheel's `.dist-info/licenses/` compared with what the inventory says ships. |
| Dependabot | None existed. `.github/dependabot.yml` written and committed (cargo, pip for `bindings/python` and `docs`, github-actions; weekly; minor+patch grouped). This is the only change applied; workflow edits are proposed as diffs below. |

## Totals

| | |
|---|---|
| Workspace | 43 library crates + `bindings/python` (`tsecon-python`, cdylib `_core`) |
| `Cargo.lock` | v4, 141 packages: 44 workspace, **97 third-party** (all targets, incl. dev/build) |
| Linked or built for the wheel on the host (`-e no-dev`) | **71 crates** (7 proc-macro; build-script-only: autocfg, version_check, defer, pyo3-build-config, nano-gemm-codegen, interpol) |
| Direct external dependencies | 5 runtime — `faer 0.24` (→0.24.4), `rayon 1` (→1.12.0), `rustfft 6` (→6.4.1), `pyo3 0.29.0` (→0.29.0), `numpy 0.29.0` (→0.29.0) — plus dev-only `serde_json 1` (→1.0.150). All caret ranges, all resolved by the committed lockfile. |
| Duplicate versions | 3 names: `equator` 0.2.2/0.6.0, `equator-macro` 0.2.1/0.6.0, `syn` 1.0.109/2.0.119 — all inside faer's own subtree (nano-gemm → equator 0.2; interpol → syn 1), none introduced by tsecon |
| Yanked pinned versions | 0 |
| Crates whose latest release is > 2 years old | 14 (all small, "finished" crates: rawpointer, same-file, interpol, strength_reduce, reborrow, byteorder, defer, transpose, walkdir, heck, num-traits, paste, primal-check, version_check) |
| Direct deps behind newest | `pyo3` 0.29.0 vs 0.29.2 (patch); the other four are at newest |
| `cargo audit` | cargo-audit 0.22.2, advisory DB of 1,239 entries, 141 lockfile crates scanned: **0 vulnerabilities**, 1 warning (`paste 1.0.15` unmaintained, RUSTSEC-2024-0436 — build-time proc-macro via faer → gemm) |
| `pip-audit` over CI's install set | 60 packages resolved (numpy 2.4.6, scipy 1.17.1, pytest 9.1.1, statsmodels 0.15.0, arch 8.0.0, scikit-learn 1.9.0, mypy 2.3.1, matplotlib 3.11.1, pandas 3.0.5, mkdocs-material 9.7.7, maturin 1.15.0, polars 1.44.1, …): **0 known vulnerabilities** |
| Licence inventory | 97/97 rows match `cargo metadata` exactly; 0 missing, 0 extra, 0 licence-string mismatches; 0 copyleft tokens |
| Actions | 10 distinct actions, 37 `uses:` sites, **0 pinned to a commit SHA** (8 mutable tags, 2 branches) |
| Workflow permissions | all three workflows `contents: read` at top level; `id-token: write` in exactly the two jobs that need it (`release.publish`, `docs.deploy`) |
| Findings | **1 severe, 5 moderate, 5 low** written up; 9 further one-liners |

## Findings

### S1 — [severe] The wheel ships none of the third-party licence notices it says it ships; 29 statically linked crates are MIT-only

**Evidence.**

- `THIRD-PARTY-LICENSES.md:6-8`: "the full verbatim copyright notices for each crate are reproduced in released wheels (generated with `cargo about` at release time, per the packaging plan…)".
- `grep -rn 'cargo[- ]about' --include=*.yml --include=*.py --include=*.toml .` → only that sentence and one roadmap row. No workflow, script or config runs `cargo about`.
- `bindings/python/pyproject.toml:14`: `license-files = ["LICENSE-MIT", "LICENSE-APACHE"]` — tsecon's own licences only.
- The published wheel (`tsecon-0.6.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`, PyPI, 2026-09-03):

  ```
  tsecon-0.6.0.dist-info/licenses/LICENSE-APACHE
  tsecon-0.6.0.dist-info/licenses/LICENSE-MIT
  tsecon-0.6.0.dist-info/sboms/tsecon-python.cyclonedx.json   (119 components, licence *ids* only)
  ```

  No `THIRD-PARTY-*`, no per-crate notice. The `.so` is 16.3 MB and links every crate in the no-dev graph.
- Of the 71 crates linked or built on the host, **29 are licensed MIT only** (no Apache alternative): dyn-stack, dyn-stack-macros, equator ×2, equator-macro ×2, faer, faer-traits, gemm, gemm-c32, gemm-c64, gemm-common, gemm-f32, gemm-f64, interpol, libm, nano-gemm ×7, private-gemm-x86, pulp, pulp-wasm-simd-flag, qd, raw-cpuid, reborrow. MIT's one condition is that "the above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software"; Apache-2.0 §4(c)/(d) carries the equivalent retain-notices condition for the other 42.

**Refutation attempted.** Does the SBOM satisfy the condition? No — CycloneDX carries SPDX identifiers, not the copyright lines or the licence text. Does the sdist? It ships the crate *sources* only as tsecon's own workspace (571 `crates/` entries); third-party sources are fetched at build time, so the sdist is clean but the binary wheel is not. Is the .so "substantial portions"? faer/gemm/pulp are the entire linear-algebra core; yes.

**Impact.** Every published wheel (0.1.0–0.6.0, and 0.8.0 when tagged) distributes 29 MIT-only crates without their notices, and the repository's own inventory asserts the opposite. Practical exposure is low (nobody enforces this against a research library), certainty is total, and the fix is five lines.

**Fix (proposed, not applied — touches pyproject.toml and the release path).**

1. Generate the notices once per build with `cargo-about` and ship the result:

   ```toml
   # bindings/python/pyproject.toml
   license-files = ["LICENSE-MIT", "LICENSE-APACHE", "THIRD-PARTY-NOTICES.txt"]
   ```

   with a step before every `maturin build` (ci.yml and release.yml):

   ```yaml
   - name: Generate third-party notices
     run: |
       cargo install cargo-about --locked --version 0.7.1
       cargo about generate --manifest-path bindings/python/Cargo.toml \
         --format json -o /dev/null   # fails on an unknown licence
       cargo about generate --manifest-path bindings/python/Cargo.toml \
         about.hbs -o bindings/python/THIRD-PARTY-NOTICES.txt
   ```

   (`about.toml` at the root with `accepted = ["MIT","Apache-2.0","BSD-2-Clause","BSD-3-Clause","Zlib","Unicode-3.0","Unlicense","Apache-2.0 WITH LLVM-exception"]` doubles as the copyleft gate the README claims.)
2. Until that lands, correct `THIRD-PARTY-LICENSES.md:6-8` so it stops claiming the notices ship, and add the file itself to `license-files` (it names every crate, version and licence, which is most of the attribution even without the verbatim texts).

### M1 — [moderate] In `release.yml` the wheel is smoke-tested with unpinned PyPI packages *before* it is uploaded, so anything pip installs there can rewrite the artifact that gets published and attested

**Evidence.** `release.yml:41-62`, step order in the `wheels` job:

```
41  - name: Build abi3 wheel            (maturin → bindings/python/dist/*.whl)
50  - name: Smoke-test the built wheel  (pip install --upgrade pip pytest numpy scipy; pytest)
59  - uses: actions/upload-artifact@v4  (path: bindings/python/dist/*.whl)
```

`python -m pip install --upgrade pip pytest numpy scipy` (line 54) resolves the newest release of each on the day, with no pin and no hash. Every package installed there, and everything pytest imports, runs with write access to `bindings/python/dist/` for roughly a minute before line 59 reads it. The `publish` job then uploads whatever it downloads with `id-token: write` and PyPI signs a PEP 740 attestation for it (verified present for 0.6.0 via `pypi.org/integrity/…/provenance`).

**Refutation attempted.** Could the tamper be caught? Nothing hashes the wheel between build and upload; the attestation is generated *after* the tamper window and would certify the tampered bytes. Is the path realistic? It requires a malicious release of pytest/numpy/scipy or one of their ~25 transitive dependencies to be live on PyPI at the moment of a tag push — rare, but it is exactly the class of incident (ultralytics 2024, the 2025 npm worm) that motivates ordering artifacts before untrusted installs. Today: `pip-audit` reports 0 vulnerabilities in that set.

**Impact.** A compromised test dependency could ship a poisoned tsecon wheel under a valid attestation. Not exploitable today; structurally wrong.

**Fix (proposed diff below).** Move `upload-artifact` to immediately after the build, so the smoke test operates on a copy that has already left the runner. Additionally pin the test set (M4).

### M2 — [moderate] No action is pinned to a commit; two are pinned to *branches*; the publish job (`id-token: write`) consumes two of them

**Evidence.** `git ls-remote` on 2026-09-04:

| `uses:` (sites) | Publisher | Ref kind | Resolves today to |
|---|---|---|---|
| `actions/checkout@v4` (8) | GitHub | moving tag | `11d5960a…` (v4.4.0) |
| `actions/setup-python@v5` (7) | GitHub | moving tag | `a26af69b…` (v5.6.0) |
| `dtolnay/rust-toolchain@1.97.1` (5) | individual | **branch** `refs/heads/1.97.1` | `4716b85f…` |
| `Swatinem/rust-cache@v2` (5) | individual | moving tag | `6323deb1…` (v2.9.2) |
| `PyO3/maturin-action@v1` (6) | PyO3 org | moving tag | `e83996d1…` (v1.51.0) |
| `actions/upload-artifact@v4` (2) | GitHub | moving tag | `ea165f8d…` (v4.6.2) |
| `actions/download-artifact@v4` (1, publish job) | GitHub | moving tag | `d3f86a10…` (v4.3.0) |
| `pypa/gh-action-pypi-publish@release/v1` (1, publish job) | PyPA | **branch** | `dc37677b…` (v1.14.2) |
| `actions/upload-pages-artifact@v3` (1) | GitHub | moving tag | `56afc609…` (v3.0.1) |
| `actions/deploy-pages@v4` (1, deploy job) | GitHub | moving tag | `d6db9016…` (v4.0.5) |

**Refutation attempted.** Is a moving tag on `actions/*` a real risk? GitHub's own actions are the lowest-risk publisher, but the mechanism is identical to `tj-actions/changed-files` (March 2025: every `v*` tag retargeted to a credential-dumping commit, 23,000 repositories). The two branch pins are worse than tags: a branch is *designed* to move. The publish job has only `id-token: write`, so the blast radius is "mint an OIDC token good for uploading to PyPI as tsecon" — which is the whole game for this repository.

**Impact.** Compromise of any one of five publisher accounts becomes arbitrary code in CI; for two of them it becomes a signed PyPI release. Not exploitable today.

**Fix (proposed diff below; Dependabot applied).** Pin every `uses:` to the full SHA it resolves to today with a trailing `# vX.Y.Z` comment. `dtolnay/rust-toolchain` reads the toolchain from its *ref name*, so the SHA pin needs `with: toolchain: 1.97.1` added — the diff does this. The committed `.github/dependabot.yml` keeps the SHAs current (Dependabot updates the SHA and the comment together).

### M3 — [moderate] The release build's inputs float: maturin `latest`, the manylinux container's toolchain defaults to `stable`, and neither `cargo` nor `maturin` is run `--locked`

**Evidence.**

- `release.yml:42-49`: `PyO3/maturin-action@v1` with `command: build`, `args: --release --out dist`, no `maturin-version`, no `rust-toolchain`. `ci.yml:54-58, 86-90, 117-121, 199-203`: same. `ci.yml:32-33`: `cargo clippy` / `cargo test` without `--locked`.
- `PyO3/maturin-action` README (fetched at `v1`): `maturin-version` — "Must match a tagged release. Default: `latest`"; `rust-toolchain` — "Defaults to `stable` for Docker build. To use the latest available version for the host build, the user must specify this in the CI config".
- The published 0.6.0 wheel's `WHEEL` file: `Generator: maturin (1.15.0)` — i.e. whatever was latest on 2026-09-03.
- `rust-toolchain.toml` pins 1.97.1 and its comment says CI and local builds use "the SAME compiler"; `release.yml` has no toolchain step at all. Inside the manylinux container the build only lands on 1.97.1 if rustup walks up from `bindings/python` to the root toolchain file and auto-installs it — plausible, undocumented, and unverified by anything in the repository.

**Refutation attempted.** Without `--locked`, does cargo actually drift? Only if `Cargo.lock` is stale relative to a `Cargo.toml`, in which case cargo re-resolves silently to the newest compatible versions — precisely the case a lockfile is supposed to make loud. Is `maturin latest` dangerous? maturin is the tool that *writes the wheel*; a bad release (or a compromised one) changes every artifact. The 1.14→1.15 jump between two tsecon releases is already visible in the WHEEL metadata.

**Impact.** Two consecutive tags can be built with different compilers and different maturin versions from the same lockfile, and a stale lockfile would be silently re-resolved on the release runner. Reproducibility claim ("builds from a clean clone, CI-verified") is weaker than stated.

**Fix (proposed diff below).** `maturin-version: v1.15.0`, `rust-toolchain: 1.97.1`, `args: --release --locked --out dist` on every maturin step; `--locked` on `cargo clippy`/`cargo test`. Dependabot (github-actions ecosystem) will not track `maturin-version`; bump it with the `maturin>=1.14,<2.0` build-system floor.

### M4 — [moderate] Every `pip install` in CI, release and docs is unpinned and hash-less; the Evidence gate compares against whatever statsmodels/arch/scikit-learn resolve on the day

**Evidence.** `ci.yml:62` (`pip pytest numpy scipy`), `:92` (`pip mypy numpy`), `:127` (`pip numpy scipy statsmodels arch scikit-learn`), `:146` (`matplotlib pandas`), `:212`; `release.yml:54`; `docs.yml:38` via `docs/requirements.txt` (`mkdocs-material>=9.5`, deployed with `pages: write` + `id-token: write`). `pip-audit -r` over that set today: 60 packages, 0 vulnerabilities, resolved to numpy 2.4.6 / scipy 1.17.1 / statsmodels 0.15.0 / arch 8.0.0 / scikit-learn 1.9.0 / pandas 3.0.5.

**Refutation attempted.** The parity gate *should* track the newest references, one could argue — but then a reference library's behaviour change (statsmodels 0.15 changed several defaults) fails tsecon's build with no tsecon change, and the failure is indistinguishable from a regression. A pinned set with Dependabot bumps turns "statsmodels changed" into a reviewable PR.

**Impact.** Non-reproducible CI (a red build on an untouched branch), and a wider supply-chain surface than necessary in the two jobs that hold `id-token: write` (docs deploy) or produce the release artifact (M1).

**Fix (proposed).** A `bindings/python/requirements-ci.txt` with `==` pins and environment markers (the abi3 3.9 leg needs `numpy==2.0.2; python_version < "3.10"` and `scipy==1.13.1; python_version < "3.10"`), installed with `python -m pip install -r`; Dependabot's pip entry for `/bindings/python` (applied) will then track it. Hash pinning (`pip-compile --generate-hashes`) is the follow-up once the file exists. `docs/requirements.txt` → `mkdocs-material==9.7.7` (Dependabot `/docs` entry applied).

### M5 — [moderate] One unmaintained crate in the build (`paste`, RUSTSEC-2024-0436) and a single-maintainer concentration under the entire linear-algebra path

**Evidence.**

- `cargo audit -f Cargo.lock` (cargo-audit 0.22.2, 1,239 advisories loaded, 141 crates scanned): `vulnerabilities: 0`; one warning — `Crate: paste / Version: 1.0.15 / Warning: unmaintained / ID: RUSTSEC-2024-0436 / Date: 2024-10-07`. Exit code 0 (unmaintained is a warning, not a failure, by default).
- `cargo tree -i paste --target all`: `paste 1.0.15 (proc-macro)` ← `gemm`, `gemm-c32`, `gemm-c64`, `gemm-common`, … ← `faer 0.24.4` ← `tsecon-linalg`. Build-time only (proc-macro), no runtime code.
- crates.io repository fields: 25 of the 71 host crates — faer, faer-traits, gemm ×7, nano-gemm ×7, private-gemm-x86, pulp, pulp-wasm-simd-flag, dyn-stack, dyn-stack-macros, equator ×2, equator-macro ×2, reborrow, qd — resolve to one maintainer's repositories (`sarah-ek` / `sarah-quinones` on GitHub, `faer` and `dyn-stack` now on Codeberg). Every matrix operation in tsecon goes through them.
- Also via that subtree, Apple targets only: `sysctl 0.6.0` (newest 0.7.1) → `thiserror 1.0.69` (2.x current) and `enum-as-inner 0.6.1` (0.7 current). Not reachable on Linux/Windows; not vulnerable.

**Refutation attempted.** `paste` is a proc-macro: it runs on the build machine, never in the wheel; its "unmaintained" status means no fixes, not a known defect. The concentration is a bus-factor and account-takeover risk, not a code defect — and the alternative (a system BLAS) was rejected deliberately in Module 00 for good reasons.

**Impact.** No exploit path. A takeover of one crates.io account would flow straight into the wheel on the next Dependabot merge — which is the argument for the audit gate below, and for reviewing (not auto-merging) the grouped cargo PRs.

**Fix (proposed).** A scheduled + on-PR `audit.yml` running `cargo audit --deny warnings` (with `--ignore RUSTSEC-2024-0436` until faer drops `paste`) and `pip-audit`; see *Proposed workflow diffs*. Keep Dependabot's cargo group at review-required.

### L1 — [low] Fork pull requests: the first-time-contributor approval prompt is the only gate, and the compensating controls behind it are correct. What PR #18's empty check list means

**Evidence.** PR #18 (`freddiejoane854-cyber:docs/var-lag-selection-cookbook` → `main`), opened 2026-09-01, cross-repository, `mergeable_state: dirty`, base `6bd023c` (0.6.0), 4 commits, 4 files (`CHANGELOG.md`, `docs/cookbook/README.md`, `docs/cookbook/var-lag-selection.md` (new), `mkdocs.yml`), `get_check_runs` → `total_count: 0`. Both `ci.yml` and `docs.yml` trigger on `pull_request` (docs.yml path-filtered to `docs/**`, `mkdocs.yml`, the stub) — so a run *should* exist. Zero runs from a first-time contributor's fork is the signature of GitHub's default **"Require approval for first-time contributors"** setting: the workflows are queued behind a maintainer's "Approve and run" click and nothing from the fork has executed. (The setting itself could not be read through the proxy — see Open.)

What approval would run: `docs.validate` (`python docs/gen_api_reference.py`, `pip install -r docs/requirements.txt`, `mkdocs build --strict`) and all of `ci.yml`, including the Evidence job, which executes `docs/examples/*.py`, `benchmarks/bench.py` and `notebooks/*_src.py` from the merged tree — i.e. any Python the PR touches. #18 touches none.

Controls that hold regardless of approval: no `pull_request_target` anywhere; `GITHUB_TOKEN` is read-only (`contents: read`, and GitHub forces read-only for fork PRs anyway); no `secrets.*` referenced in any workflow; `Swatinem/rust-cache` writes from a PR are scoped to that PR's merge ref and cannot poison `main`'s cache; `release.yml` does not use the cache at all, so no PR can influence a release artifact through it.

**Assessment.** Safe as configured. Two notes for the maintainer: (1) the PR is docs-only and benign on read, but it conflicts with `main` (CHANGELOG), targets 0.6.0, and claims verification "against tsecon 0.4.0" — ask for a rebase before approving a run; (2) for a repository whose CI *executes repository Python* on every PR, the stricter setting **"Require approval for all outside collaborators"** costs nothing and closes the gap where a contributor's *second* PR runs unreviewed.

### L2 — [low] Two compatibility floors are declared and never exercised: `numpy>=1.22` and `rust-version = "1.85"`

**Evidence.**

- `pyproject.toml:42` `numpy>=1.22`; the binding compiles against `numpy 0.29.0` whose runtime checks (`numpy-0.29.0/src/npyffi/array.rs:94-116`, `numpyconfig.rs:6-9`) reject a NumPy whose ABI is *newer* than 2.0 or whose C-API feature version is *older* than `0xc` (NumPy 1.15). NumPy 1.22 reports `0xf`: **a 1.22 wheel would import.** The pure-Python layer uses 39 distinct `np.*` names, all present since NumPy ≤1.17 (`asarray`, `ascontiguousarray`, `flatnonzero`, `column_stack`, `linalg.lstsq`, `errstate`, …); no `copy=` keyword, no `np.exceptions`, no 2.0-only names. So the floor is *probably* honest — but every CI job installs the newest NumPy (2.4.6 on 3.12/3.13; 2.0.2 on the 3.9 abi3 leg), so no 1.x NumPy has ever run the suite. A release wheel was built locally from this tree to import it under a NumPy 1.x, but the box was shared with six sibling audit builds and the build had not finished inside the budget; see Open.
- `Cargo.toml:18` `rust-version = "1.85"`; CI builds only on 1.97.1 (`ci.yml:27,48,83,114,191`, `rust-toolchain.toml:8`). `cargo metadata` `rust_version` over all 97 packages: none above 1.85, and `generativity 1.2.1` is *exactly* 1.85 — one Dependabot minor bump away from a silently false MSRV.

**Impact.** A user on NumPy 1.2x or a packager on Rust 1.85 is the first to find out. Hygiene.

**Fix (proposed).** Add a `numpy==1.22.4` (Python 3.9) row to the abi3 matrix, and a `cargo check --workspace --locked` job on `dtolnay/rust-toolchain` `1.85` (check only; no clippy, which is not MSRV-stable).

### L3 — [low] Release hygiene: two of four wheels are published untested, no concurrency guard on release or Pages deploy, and `workflow_dispatch` on a tag ref would publish

**Evidence.** `release.yml:51`: `if: matrix.platform.target == 'x86_64' || matrix.platform.target == 'x64'` — the macOS `aarch64` wheel is built *natively* on `macos-latest` (Apple Silicon) and could be imported there, but is skipped along with the genuinely cross-built Linux aarch64 wheel; both go to PyPI having never been imported. `release.yml` and `docs.yml` have no `concurrency:` (two pushes to `main` a minute apart can deploy Pages out of order; `actions/deploy-pages` documents `group: pages, cancel-in-progress: false`). `release.yml:14,85`: `workflow_dispatch` is allowed and `publish` gates only on `startsWith(github.ref, 'refs/tags/')` — a dispatch *from* a tag satisfies it; only the `pypi` environment's protection rules (unverifiable, see Open) stand between a write-access user and a re-publish attempt.

**Fix (proposed diff below).** Smoke-test everything but Linux aarch64; add the two concurrency groups; confirm required reviewers on the `pypi` environment.

### L4 — [low] Repository hygiene: no `SECURITY.md`, no `CODEOWNERS`, `persist-credentials` left on, and the licence-inventory header understates its own table

**Evidence.** `ls -a . .github` → no `SECURITY.md`, no `CODEOWNERS`, no issue/PR templates (GitHub's "Report a vulnerability" flow is therefore off). `actions/checkout@v4` default `persist-credentials: true` leaves the token in `.git/config` for every later `run:` step (read-only here, so harmless today; it is the setting that turns a `contents: write` grant into a push). `THIRD-PARTY-LICENSES.md:3-5` lists "MIT / Apache-2.0 / BSD / Zlib / Unicode-3.0" but the table also contains `Unlicense OR MIT` (byteorder, memchr, same-file, walkdir, winapi-util) and `Apache-2.0 WITH LLVM-exception` (target-lexicon) — all permissive, all fine, just not what the header says. `pyproject.toml:31-35` classifiers stop at 3.13 (3.14 has been GA since 2025-10; the abi3 wheel runs on it).

**Fix (proposed).** `SECURITY.md` (private-report channel + supported versions), `persist-credentials: false` (in the diffs), header wording, add the 3.14 classifier.

### L5 — [low] `serde_json` is declared 43 times with an identical spec instead of once in `[workspace.dependencies]`

**Evidence.** `grep -c serde_json crates/*/Cargo.toml` → 43 crates, 41 with `{ version = "1", features = ["float_roundtrip"] }`, 2 (`tsecon-linalg`, `tsecon-stats`) with plain `"1"`. Dev-only, so nothing ships; but a future feature or version change has to be made in 43 places and the two odd ones show it already drifted. Same pattern for `rayon = "1"` (4 sites). Hygiene.

### The rest, one line each

- `interpol 0.2.1` (2020, last release; pulls `syn 1.0.109`) is a build-dependency of `private-gemm-x86`: dormant build-time proc-macro on x86_64 only; nothing to do but watch faer.
- ~15 crates have semver-compatible updates pending (`pyo3` 0.29.0→0.29.2, `zerocopy`, `bytemuck`, `libc`, `either`, `portable-atomic`, `proc-macro2`, `quote`, `serde*`, …); the first Dependabot cargo PR will collect them.
- `notebooks/04_gertler_karadi_src.py:76,138` downloads `Ramey_HOM_monetary.zip` over HTTPS at CI time with no checksum; it is parsed as XLSX XML (no code path), so data-integrity only.
- No `curl | sh`, no `pip install git+…`, no `${{ github.event.* }}` inside any `run:`, no `secrets.*`: all clean.
- `docs.yml:38` uses bare `pip` rather than `python -m pip`; harmless on the `setup-python` runner, fixed in the diff for consistency.
- `faer` is consumed only through `tsecon-linalg` (the single `faer =` line in the workspace, `crates/tsecon-linalg/Cargo.toml:9`) and the lockfile holds one `faer` and one `faer-traits`: the single-version policy stated in `crates/tsecon-ml/Cargo.toml:10-13` **holds workspace-wide**.
- PyPI's latest is 0.6.0 (2026-09-03) while `main` is 0.8.0 with no `v0.7.0`/`v0.8.0` tag pushed; the next tag push exercises `release.yml` as audited here, so M1/M3 apply to it.
- `rust-toolchain.toml` (1.97.1) and the five `dtolnay/rust-toolchain@1.97.1` sites agree today; the version is written in six places and drifts by hand.
- Python 3.9 reached end-of-life 2025-10 and is still the floor (`requires-python = ">=3.9"`, abi3-py39); a deliberate choice, noted only so the next audit does not re-discover it.

## Clean bills

- **`cargo audit`**: 0 vulnerabilities in 141 lockfile crates against the 1,239-advisory RustSec database (cargo-audit 0.22.2, DB fetched 2026-09-04). The single warning (`paste`, unmaintained) is written up as M5.
- **`pip-audit`**: 0 known vulnerabilities across the 60 packages CI's unpinned installs resolve to today.
- **Licence inventory**: `THIRD-PARTY-LICENSES.md` matches `cargo metadata` exactly (97/97, names, versions and licence strings); every licence token is in {MIT, Apache-2.0, BSD-2-Clause, Zlib, Unicode-3.0, Unlicense, Apache-2.0 WITH LLVM-exception}; **zero copyleft** — the README's "100%-permissive dependency tree" claim holds for the tree (S1 is about *shipping the notices*, not the licences themselves).
- **Workflow permissions and triggers**: `contents: read` at the top of all three workflows; `id-token: write` only in `release.publish` (with `environment: pypi`, gated on a tag ref) and `docs.deploy` (with `environment: github-pages`, gated on push to `main`); no `pull_request_target`; no secrets; no expression injection; PEP 740 attestations verified live on PyPI for 0.6.0.
- **Cache poisoning**: `Swatinem/rust-cache` appears only in `ci.yml`; PR writes are PR-scoped; `release.yml` uses no cache, so no cache content can reach a published artifact.
- **Toolchain consistency**: `rust-toolchain.toml` = CI = 1.97.1; all 97 packages' declared MSRVs ≤ the workspace's 1.85.
- **Lockfile**: v4, committed, shipped inside the sdist (`tsecon-0.6.0/Cargo.lock` present); no yanked versions; all five direct dependencies at (or within one patch of) newest; caret pins everywhere, no `=` pins and no git dependencies.
- **faer single-version policy**: holds (see one-liners).
- **The sdist**: carries both licence files at root and under `bindings/python`, `pyproject.toml`, `Cargo.lock`, and 571 workspace crate files; nothing third-party vendored.

## Proposed workflow diffs

Generated with `diff -u` against the files at 19d308e; the edited copies parse as YAML. Apply after review — pins are the SHAs resolved on 2026-09-04 (table in M2), and Dependabot will move them from there.

### `.github/workflows/ci.yml`

```diff
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -23,14 +23,17 @@
     name: Rust workspace (test + clippy + fmt)
     runs-on: ubuntu-latest
     steps:
-      - uses: actions/checkout@v4
-      - uses: dtolnay/rust-toolchain@1.97.1
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0
         with:
+          persist-credentials: false
+      - uses: dtolnay/rust-toolchain@4716b85f2fac3e324e64fa2810f6b5c3905760a5 # 1.97.1
+        with:
+          toolchain: 1.97.1
           components: clippy, rustfmt
-      - uses: Swatinem/rust-cache@v2
+      - uses: Swatinem/rust-cache@6323deb102c322ba6fcbdcafc7e3dddab59af2b6 # v2.9.2
       - run: cargo fmt --all --check
-      - run: cargo clippy --workspace --all-targets -- -D warnings
-      - run: cargo test --workspace
+      - run: cargo clippy --workspace --all-targets --locked -- -D warnings
+      - run: cargo test --workspace --locked
 
   python:
     name: Python wheel (${{ matrix.os }})
@@ -41,21 +44,27 @@
         os: [ubuntu-latest, macos-latest, windows-latest]
     runs-on: ${{ matrix.os }}
     steps:
-      - uses: actions/checkout@v4
-      - uses: actions/setup-python@v5
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0
+        with:
+          persist-credentials: false
+      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
         with:
           python-version: "3.12"
-      - uses: dtolnay/rust-toolchain@1.97.1
-      - uses: Swatinem/rust-cache@v2
+      - uses: dtolnay/rust-toolchain@4716b85f2fac3e324e64fa2810f6b5c3905760a5 # 1.97.1
+        with:
+          toolchain: 1.97.1
+      - uses: Swatinem/rust-cache@6323deb102c322ba6fcbdcafc7e3dddab59af2b6 # v2.9.2
       # Build the abi3 wheel, install it, and run the test suite against the
       # installed artifact — not a `maturin develop` editable install — so
       # packaging bugs (missing stub, wrong module name) surface here.
       - name: Build wheel
-        uses: PyO3/maturin-action@v1
+        uses: PyO3/maturin-action@e83996d129638aa358a18fbd1dfb82f0b0fb5d3b # v1.51.0
         with:
           working-directory: bindings/python
+          maturin-version: v1.15.0
+          rust-toolchain: 1.97.1
           command: build
-          args: --release --out dist
+          args: --release --locked --out dist
       - name: Install wheel and run tests
         shell: bash
         run: |
@@ -76,18 +85,24 @@
     needs: rust
     runs-on: ubuntu-latest
     steps:
-      - uses: actions/checkout@v4
-      - uses: actions/setup-python@v5
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0
+        with:
+          persist-credentials: false
+      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
         with:
           python-version: "3.12"
-      - uses: dtolnay/rust-toolchain@1.97.1
-      - uses: Swatinem/rust-cache@v2
+      - uses: dtolnay/rust-toolchain@4716b85f2fac3e324e64fa2810f6b5c3905760a5 # 1.97.1
+        with:
+          toolchain: 1.97.1
+      - uses: Swatinem/rust-cache@6323deb102c322ba6fcbdcafc7e3dddab59af2b6 # v2.9.2
       - name: Build + install wheel
-        uses: PyO3/maturin-action@v1
+        uses: PyO3/maturin-action@e83996d129638aa358a18fbd1dfb82f0b0fb5d3b # v1.51.0
         with:
           working-directory: bindings/python
+          maturin-version: v1.15.0
+          rust-toolchain: 1.97.1
           command: build
-          args: --release --out dist
+          args: --release --locked --out dist
       - run: |
           python -m pip install --upgrade pip mypy numpy
           python -m pip install --no-index --find-links bindings/python/dist tsecon
@@ -107,18 +122,24 @@
     needs: rust
     runs-on: ubuntu-latest
     steps:
-      - uses: actions/checkout@v4
-      - uses: actions/setup-python@v5
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0
+        with:
+          persist-credentials: false
+      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
         with:
           python-version: "3.12"
-      - uses: dtolnay/rust-toolchain@1.97.1
-      - uses: Swatinem/rust-cache@v2
+      - uses: dtolnay/rust-toolchain@4716b85f2fac3e324e64fa2810f6b5c3905760a5 # 1.97.1
+        with:
+          toolchain: 1.97.1
+      - uses: Swatinem/rust-cache@6323deb102c322ba6fcbdcafc7e3dddab59af2b6 # v2.9.2
       - name: Build + install wheel (release — timings are meaningless otherwise)
-        uses: PyO3/maturin-action@v1
+        uses: PyO3/maturin-action@e83996d129638aa358a18fbd1dfb82f0b0fb5d3b # v1.51.0
         with:
           working-directory: bindings/python
+          maturin-version: v1.15.0
+          rust-toolchain: 1.97.1
           command: build
-          args: --release --out dist
+          args: --release --locked --out dist
       - name: Install
         run: |
           # Every reference the parity matrix compares against must be present,
@@ -187,21 +208,27 @@
         # and newest supported interpreters, which is exactly what abi3 claims.
         python-version: ["3.9", "3.13"]
     steps:
-      - uses: actions/checkout@v4
-      - uses: dtolnay/rust-toolchain@1.97.1
-      - uses: Swatinem/rust-cache@v2
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0
+        with:
+          persist-credentials: false
+      - uses: dtolnay/rust-toolchain@4716b85f2fac3e324e64fa2810f6b5c3905760a5 # 1.97.1
+        with:
+          toolchain: 1.97.1
+      - uses: Swatinem/rust-cache@6323deb102c322ba6fcbdcafc7e3dddab59af2b6 # v2.9.2
       # Build with a single fixed interpreter; abi3 means the artifact is not
       # tied to it. Testing under a DIFFERENT interpreter is the whole point.
-      - uses: actions/setup-python@v5
+      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
         with:
           python-version: "3.12"
       - name: Build the abi3 wheel (on 3.12)
-        uses: PyO3/maturin-action@v1
+        uses: PyO3/maturin-action@e83996d129638aa358a18fbd1dfb82f0b0fb5d3b # v1.51.0
         with:
           working-directory: bindings/python
+          maturin-version: v1.15.0
+          rust-toolchain: 1.97.1
           command: build
-          args: --release --out dist
-      - uses: actions/setup-python@v5
+          args: --release --locked --out dist
+      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
         with:
           python-version: ${{ matrix.python-version }}
       - name: Install that same wheel on ${{ matrix.python-version }} and test
```

### `.github/workflows/release.yml`

```diff
--- a/.github/workflows/release.yml
+++ b/.github/workflows/release.yml
@@ -16,6 +16,12 @@
 permissions:
   contents: read
 
+# One release at a time per ref; never cancel a publish that is already
+# uploading (a half-published version set cannot be re-uploaded to PyPI).
+concurrency:
+  group: release-${{ github.ref }}
+  cancel-in-progress: false
+
 jobs:
   wheels:
     name: Wheels (${{ matrix.platform.os }} / ${{ matrix.platform.target }})
@@ -34,21 +40,39 @@
           - { os: macos-latest,   target: aarch64, manylinux: "" }
           - { os: windows-latest, target: x64,     manylinux: "" }
     steps:
-      - uses: actions/checkout@v4
-      - uses: actions/setup-python@v5
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0
+        with:
+          persist-credentials: false
+      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
         with:
           python-version: "3.12"
       - name: Build abi3 wheel
-        uses: PyO3/maturin-action@v1
+        uses: PyO3/maturin-action@e83996d129638aa358a18fbd1dfb82f0b0fb5d3b # v1.51.0
         with:
           working-directory: bindings/python
+          maturin-version: v1.15.0
+          # The manylinux container defaults to `stable`; pin it to the same
+          # compiler rust-toolchain.toml and ci.yml use.
+          rust-toolchain: 1.97.1
           target: ${{ matrix.platform.target }}
           manylinux: ${{ matrix.platform.manylinux }}
           command: build
           # abi3-py39 (set in Cargo.toml) => one wheel covers all Python >= 3.9.
-          args: --release --out dist
-      - name: Smoke-test the built wheel (native targets only)
-        if: matrix.platform.target == 'x86_64' || matrix.platform.target == 'x64'
+          # --locked: the wheel is built from the committed Cargo.lock or not at all.
+          args: --release --locked --out dist
+      # Upload BEFORE the smoke test. The test step pip-installs unpinned
+      # packages from PyPI; nothing installed there can touch the artifact
+      # that gets published, because it has already left the runner.
+      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2
+        with:
+          name: wheels-${{ matrix.platform.os }}-${{ matrix.platform.target }}
+          path: bindings/python/dist/*.whl
+      # Every wheel that can run on its build host is smoke-tested: Linux
+      # x86_64, Windows x64, AND macOS arm64 (macos-latest is Apple Silicon,
+      # so the aarch64 wheel is native there). Only the Linux aarch64 wheel
+      # is cross-built and cannot be imported on the runner.
+      - name: Smoke-test the built wheel (every natively runnable target)
+        if: ${{ !(matrix.platform.os == 'ubuntu-latest' && matrix.platform.target == 'aarch64') }}
         shell: bash
         run: |
           python -m pip install --upgrade pip pytest numpy scipy
@@ -56,23 +80,22 @@
           python -c "import tsecon; assert 'site-packages' in tsecon.__file__, tsecon.__file__"
           # Run in place so the tests' repo-relative fixture paths resolve.
           python -m pytest bindings/python/tests -q
-      - uses: actions/upload-artifact@v4
-        with:
-          name: wheels-${{ matrix.platform.os }}-${{ matrix.platform.target }}
-          path: bindings/python/dist/*.whl
 
   sdist:
     name: Source distribution
     runs-on: ubuntu-latest
     steps:
-      - uses: actions/checkout@v4
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0
+        with:
+          persist-credentials: false
       - name: Build sdist
-        uses: PyO3/maturin-action@v1
+        uses: PyO3/maturin-action@e83996d129638aa358a18fbd1dfb82f0b0fb5d3b # v1.51.0
         with:
           working-directory: bindings/python
+          maturin-version: v1.15.0
           command: sdist
           args: --out dist
-      - uses: actions/upload-artifact@v4
+      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2
         with:
           name: sdist
           path: bindings/python/dist/*.tar.gz
@@ -91,12 +114,12 @@
       # there is no API token to leak. Also enables PEP 740 build attestations.
       id-token: write
     steps:
-      - uses: actions/download-artifact@v4
+      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093 # v4.3.0
         with:
           path: dist
           merge-multiple: true
       - name: Publish
-        uses: pypa/gh-action-pypi-publish@release/v1
+        uses: pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33 # v1.14.2
         with:
           packages-dir: dist
           # For the Test PyPI dry-run, uncomment:
```

### `.github/workflows/docs.yml`

```diff
--- a/.github/workflows/docs.yml
+++ b/.github/workflows/docs.yml
@@ -24,23 +24,31 @@
 permissions:
   contents: read
 
+# Pages deploys must land in push order; a cancelled deploy can leave the
+# site half-updated, so never cancel-in-progress here.
+concurrency:
+  group: pages
+  cancel-in-progress: false
+
 jobs:
   validate:
     runs-on: ubuntu-latest
     steps:
-      - uses: actions/checkout@v4
-      - uses: actions/setup-python@v5
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0
+        with:
+          persist-credentials: false
+      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
         with:
           python-version: "3.12"
       - name: Regenerate the API reference from the type stub
         run: python docs/gen_api_reference.py
       - name: Install docs dependencies
-        run: pip install -r docs/requirements.txt
+        run: python -m pip install -r docs/requirements.txt
       - name: Build the site (strict — fails on a broken link or nav entry)
         run: mkdocs build --strict
       - name: Upload the built site for the deploy job
         if: github.event_name == 'push' && github.ref == 'refs/heads/main'
-        uses: actions/upload-pages-artifact@v3
+        uses: actions/upload-pages-artifact@56afc609e74202658d3ffba0e8f6dda462b719fa # v3.0.1
         with:
           path: site
 
@@ -58,4 +66,4 @@
     steps:
       - name: Deploy to GitHub Pages
         id: deployment
-        uses: actions/deploy-pages@v4
+        uses: actions/deploy-pages@d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e # v4.0.5
```

### New: `.github/workflows/audit.yml` (M5, M4)

```yaml
name: Audit

# Known-advisory gate for both dependency surfaces. Runs on every PR that
# touches a manifest, weekly against main (new advisories land without a
# commit), and on demand. Fails on any vulnerability; unmaintained-crate
# warnings fail too, except those listed as accepted below.

on:
  pull_request:
    paths: ["Cargo.lock", "Cargo.toml", "crates/*/Cargo.toml", "bindings/python/**", "docs/requirements.txt", ".github/workflows/audit.yml"]
  push:
    branches: [main]
  schedule:
    - cron: "17 6 * * 1"
  workflow_dispatch:

permissions:
  contents: read

jobs:
  cargo-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0
        with:
          persist-credentials: false
      - uses: dtolnay/rust-toolchain@4716b85f2fac3e324e64fa2810f6b5c3905760a5 # 1.97.1
        with:
          toolchain: 1.97.1
      - run: cargo install cargo-audit --locked
      # RUSTSEC-2024-0436: `paste` is unmaintained; build-time proc-macro
      # reached only through faer -> gemm. Drop the ignore when faer drops it.
      - run: cargo audit --deny warnings --ignore RUSTSEC-2024-0436

  pip-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0
        with:
          persist-credentials: false
      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
        with:
          python-version: "3.12"
      - run: python -m pip install pip-audit
      # Audits the exact set CI installs (bindings/python/requirements-ci.txt
      # once M4 lands; until then the unpinned names, resolved fresh).
      - run: python -m pip-audit -r docs/requirements.txt -r bindings/python/requirements-ci.txt
```

### Applied: `.github/dependabot.yml`

Committed on this branch. Three ecosystems (`cargo` at `/`, `pip` at `/bindings/python` and `/docs`, `github-actions` at `/`), weekly Monday 06:00 UTC, minor+patch grouped into one PR per ecosystem, majors separate, five open PRs per ecosystem, `deps(cargo)`/`deps(pip)`/`deps(actions)` commit prefixes. Note for the pip surface: `pyproject.toml` carries only lower bounds (`numpy>=1.22`), so Dependabot will be quiet there until `requirements-ci.txt` (M4) gives it pins to move.

## Open

- **Environment protection on `pypi` and `github-pages`, the fork-approval setting, branch protection/rulesets**: the session proxy refuses `/repos/…/environments`, `/actions/permissions*`, `/branches/main/protection` and `/rulesets`. Whether `pypi` requires a reviewer (which is what makes the `workflow_dispatch`-on-a-tag path in L3 safe) must be confirmed in Settings → Environments by a maintainer.
- **Empirical NumPy 1.x run**: not completed. The source-level answer (compiled floor = NumPy C-API 0xc / 1.15; Python layer uses pre-1.17 names only) says a 1.22 wheel imports; a `maturin build --release --locked` from this tree was started to confirm it under NumPy 1.26.4 on Python 3.11 (1.22 itself has no 3.11 wheels, so 1.22 must be checked on the 3.9 leg in CI), and had not completed when this file was written.
- **`cargo deny`** (licence allow-list + source restriction + advisories in one gate) was not installed or run; the `about.toml`/`audit.yml` proposals cover the same ground in two tools, and `cargo deny` would be the single-tool replacement.
- **Hash-pinned requirements** (`pip-compile --generate-hashes`) were not produced; the `==` file is the prerequisite.
- **`actions/checkout` v5 / `setup-python` v6** exist; the diffs pin the majors the repository already uses (v4/v5) rather than bundling a major bump into a security change. Dependabot will offer the majors separately.
- Not re-verified: the private-gemm-x86 / interpol build script contents (a `build.rs` in the linear-algebra path is the one place a proc-macro-level compromise would live; reading 25 crates' build scripts was outside the two-hour budget).
