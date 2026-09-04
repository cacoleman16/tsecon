# Repository audit, September 2026 (audit round 12) — findings

> **Working document.** The first audit of the **repository as a whole**
> rather than of the estimator surface: supply chain and CI, hygiene and
> dead weight, every claim the documentation makes, the health of the
> test suite itself, API consistency across all 173 callables, the
> security posture of the Python boundary, and a reconciliation of every
> finding the eleven previous rounds ever recorded. Run under
> [the brief](16-adversarial-audit-brief.md) at `19d308e` (main after the
> 0.8.0 squash merge). Excluded from the published site.

Seven sweeps ran in parallel, each in its own worktree from the same
commit, each writing one findings page and committing its probe scripts
under `lab/audit/repo/<sweep>/`. The sweep pages are the record; this
page is the roll-up, the integrator's decisions, and the open list.

| Sweep | Page | Confirmed findings (severe / moderate / low) | Fixed on this branch |
|---|---|---|---|
| Supply chain, dependencies, CI security | [supply](27-repo-audit-2026-09/supply.md) | 11 (1 / 5 / 5) + 9 one-liners | Dependabot config; the three workflow diffs and the new audit workflow applied by the integrator |
| Hygiene and dead weight | [hygiene](27-repo-audit-2026-09/hygiene.md) | 22 (2 / 8 / 12) | `.gitignore` gaps; six force-added logs untracked |
| Claims versus reality | [claims](27-repo-audit-2026-09/claims.md) | 25 (2 / 9 / 14) | all 25 |
| Test-suite health | [tests](27-repo-audit-2026-09/tests.md) | 8 (0 / 3 / 5) | none needed (no stale skip, no fixture typo) |
| API consistency | [api](27-repo-audit-2026-09/api.md) | 15 (0 / 5 / 10) | 5 (docstring and stub surfaces, with tripwire tests) |
| Security review | [security](27-repo-audit-2026-09/security.md) | 14 (3 / 5 / 6) | 2 of the 3 severe: the huge-count refusal and the allocation-panic rebuild, with 17 pinned tests |
| Open-findings ledger | [ledger](27-repo-audit-2026-09/ledger.md) | 275 items reconciled: 98 fixed, **104 open**, 25 superseded, 45 n/a, 3 unverified | 7 silently-closed items annotated in their source pages |

## What the audit established

**Clean bills, measured.** `cargo audit` over the 141-crate lockfile: 0
vulnerabilities (one unmaintained build-time proc-macro, `paste`, reached
only through faer). `pip-audit` over the 60 packages CI's installs
resolve to today: 0. The licence inventory reconciles 97/97 against
`cargo metadata` with no copyleft. Workflow permissions are minimal and
`id-token: write` is confined to the two jobs that need it. The Python
suite is reproducible to the printed digit across three runs (one under
`PYTHONHASHSEED=1`); the two Monte-Carlo-heaviest Rust crates rerun
byte-identical. 173/173 callables are reached by the tests, 157 of them
against a reference value. 0 panics escape the binding across 865
malformed calls. Every 0.6.0–0.8.0 CHANGELOG bullet (54 checks) holds
on the wheel, all 5,454 callable mentions across the docs resolve, and
the mkdocs nav is 88/88 with zero broken internal links.

**What was wrong.** Ranked by what it would cost a reader or a user:

1. **The wheel ships no third-party licence notices** although
   `THIRD-PARTY-LICENSES.md` says it does; 29 of the 71 linked crates are
   MIT-only, whose one condition is that the notice travels with the
   binary. Verified on the published 0.6.0 wheel and on a 0.8.0 wheel
   built from this tree (supply S1). Fixed on this branch: `cargo about` now generates
   `bindings/python/THIRD-PARTY-NOTICES.md` (86 crates across every wheel
   target, each licence text with its copyright lines), `license-files`
   ships it, a wheel built from this tree carries it in
   `dist-info/licenses/`, and CI fails when the committed copy is stale.
2. **Guide chapter 12 still denied the entire 0.8.0 machine-learning
   wave** in six places ("`group_lasso` not shipped", "no tree learner",
   "no neural estimator") a day after it shipped; the HAR-RV example in
   chapter 6 carried pre-0.5.0 numbers and a significance conclusion that
   is false on 0.8.0 (claims 1–2). Both rewritten from live runs.
3. **The release pipeline smoke-tested the wheel with unpinned PyPI
   installs before uploading it**, so a compromised test dependency could
   have rewritten the artifact that PyPI then attested; no action was
   pinned to a commit, two were pinned to branches, and the manylinux
   build floated both its compiler and its maturin version (supply M1–M3).
   All three fixed on this branch, see below.
4. **A quarter of the API named none of its returned keys** on either
   documentation surface (33 functions), and 73 functions never named 204
   of their option parameters in `help()` (api F4–F5). Fixed by generated
   lines on both surfaces, with a parametrized tripwire so it cannot
   recur; `ar_loglik`'s silent NaN-as-missing behaviour is now documented
   and tested (F3).
5. **`test_garch_gjr_asymmetry_detected` cannot fail on its claim** — it
   loads the arch reference and asserts only a parameter name and a
   finite log-likelihood (tests 1); 14 Python goldens are looser than the
   matrix documents while the achieved error has orders of magnitude of
   headroom (tests 2). Both left for the owner: assertions are outside an
   audit's fix remit.
6. **Dead weight**: 21 round-8 debugging scripts under `scratch/`, a
   stray `irf_table.tex` at the root, a 247 KB demo site stamped 0.0.1,
   nine `pub fn` with zero references, 52 `TODO(phase0)` markers tracked
   by no page (hygiene H1–H7). Proposed removals are tabulated; none
   applied beyond the ignore-rule and log fixes.
7. **Integer counts of 2^63 escaped the binding as an uncatchable
   `PanicException`** in 113 cells across 67 callables, and product
   overflows below that line (`bvar_fit(lags=2**31)`,
   `echo_state_network(reservoir_size=2**31)`) did the same (security
   S1–S2). Sealed in `_coerce`: counts at or beyond 2^48 are refused before
   the call reaches Rust (seeds exempt), and an allocation-sizing panic is
   rebuilt into a `ValueError` naming the argument and the size, panic
   chained as `__cause__`; the post-fix matrix shows 0 panics over 4,596
   cells. What remains is policy: at 2^31 an allocation the machine
   cannot honour still aborts the process where NumPy would raise
   `MemoryError` (S3, 63 cells), the GIL is held for the whole of every
   compiled call (M1), and `mstl(iterate=…)`/`reset_test(max_power=…)`
   have no cap (M4). The history scan is clean: 248 commits, every ref,
   18 credential patterns, 0 hits; `import tsecon` opens no socket and
   reads no environment variable.
8. **104 recorded findings remain open across eleven rounds**, and seven
   items recorded as open had in fact been fixed without the record being
   updated (ledger §4). The top ten by value are listed in the ledger's
   §3; the first three are the never-run `mapie` cross-check for
   EnbPI/ACI, the missing cross-horizon covariance behind every sup-t
   refusal on `lp_iv`/`lp_multiplier`/`lp_state`/`panel_lp`, and the
   0.789 marginal coverage of the asymptotic IRF band at h = 12.

## Applied on this branch (integrator)

- **Workflows hardened** per the supply sweep's diffs, each SHA re-resolved
  against upstream by the integrator before applying: every `uses:` in
  `ci.yml`, `release.yml` and `docs.yml` pinned to a full commit with a
  version comment; `persist-credentials: false` on every checkout;
  `dtolnay/rust-toolchain` given an explicit `toolchain: 1.97.1` (required
  once the ref is a SHA); `maturin-version: v1.15.0` and
  `rust-toolchain: 1.97.1` on every maturin step; `--locked` on every
  cargo and maturin invocation; in `release.yml` the artifact upload now
  precedes the smoke test, the smoke test also runs the natively runnable
  macOS arm64 wheel, and a `concurrency` group forbids overlapping
  releases; `docs.yml` gets a Pages concurrency group.
- **New `audit.yml`**: `cargo audit --deny warnings` (with the one
  accepted advisory named and explained) and `pip-audit --strict` over
  the docs and CI install sets, on manifest PRs, weekly, and on demand.
- **`.github/dependabot.yml`** (supply): cargo, pip (bindings and docs)
  and github-actions ecosystems, weekly, minor+patch grouped.
- **Third-party notices** (supply S1): `about.toml` (the accepted-licence
  gate), `about.hbs`, `scripts/gen_third_party_notices.sh`, the generated
  `THIRD-PARTY-NOTICES.md` in `license-files`, a CI check that it is
  current, and `publish = false` on every workspace crate so cargo-about
  keeps tsecon's own crates out of the notices. `THIRD-PARTY-LICENSES.md`
  and ROADMAP.md now describe what ships and that wheels 0.1.0–0.6.0
  predate the file.
- **Documentation**: the 25 claims fixes; the API docstring and stub
  generation; `lab/README.md` now maps `lab/audit/`; the seven
  silently-closed ledger items annotated in place.
- **Probe artefacts**: the sweeps' scripts and text summaries are
  committed under `lab/audit/repo/`. Raw machine dumps above roughly
  150 KB (a 3 MB names-sweep JSON, four junit XML files, the assertion
  scan, two API cluster dumps) are not — the scripts regenerate them and
  `.gitignore` now names them — and the ledger's probe run is committed as
  `probes_run.txt` because `*.log` has been ignored repo-wide since 0.3.0
  (the same rule under which the hygiene sweep untracked six round-11
  logs).
- The hygiene sweep's untracking of the six round-11 `.log` files stands:
  nothing in the round-11 findings page cites them, and each duplicates a
  tracked JSON.

## Not applied — the owner's calls

From the sweep pages, in the order they should be decided:

- **Hygiene removals** (hygiene "Proposed removals"): `scratch/round8/`
  (move under `lab/audit/round8/` or delete), `irf_table.tex`,
  `docs/demo/`, the round-11 `out/` JSON (keep-and-cite or ignore
  `lab/audit/*/out/`), the unreferenced pmdarima cross-check script, the
  two `viz-preview` PNGs, re-housing `prototypes/viz/` (a live dependency
  of five gallery generators — do not delete), the erf/erfc copy in
  `tsecon-ml/src/pds.rs`, and the nine dead `pub fn`.
- **Test assertions** (tests 1–2): make the GJR test assert the arch
  reference it already loads; tighten the 14 loose Python goldens to the
  matrix tolerance (`tolerance_headroom.py` shows every one has room);
  decide whether the 241-second `auto_arima` Monte-Carlo test keeps 12
  replications.
- **API renames** (api "Rename proposal"): `alpha` as miscoverage
  everywhere with the IVX exponent renamed, `max_lags`/`hac_lags`,
  `horizon`, the `se`/`p_value` key spellings, and the list→array
  conversion of 197 matrix keys at the boundary — alias-then-deprecate,
  removal at 1.0; blast radius per spelling is tabulated.
- **Pinned CI requirements** (supply M4): a `requirements-ci.txt` with
  `==` pins and markers so the Evidence parity gate stops tracking
  whatever statsmodels/arch/scikit-learn resolve on the day; Dependabot's
  pip entry is already in place to move them.
- **Repository settings the proxy could not read** (supply "Open"): the
  `pypi` environment's reviewer protection and the fork-workflow approval
  setting must be confirmed in Settings; the stricter "all outside
  collaborators" setting is recommended because the Evidence job executes
  repository Python on pull requests.
- **Fixture provenance** (tests 3, hygiene H8): 16 of 96 fixtures record
  no reference-library version, four of them third-party goldens.
- **Process aborts on impossible allocations** (security S3): a memory
  budget with an override, or `try_reserve` at the Rust allocation sites
  so refusal becomes an exception; the GIL release (M1), the GMM callback
  re-invocation after its first exception (M2), `KeyboardInterrupt`
  wrapped into `RuntimeError` inside callback forecasters (M3), the
  uncapped `mstl`/`reset_test` loops (M4), the absolute build path in the
  wheel's SBOM (L1), `#![forbid(unsafe_code)]` (L3), and a checksum on
  notebook 04's download (L5) — each with a line-level proposal on the
  security page.

## The open ledger, by theme

The ledger page groups the 104 open items into eight ranked themes; the
short form:

1. Cross-horizon inference the library refuses rather than fakes (sup-t on
   the four LP variants, Anderson-Rubin sets for `iv_gmm`/`lp_iv`, the
   `flp` generated-regressor correction, the `quantile_lp` HAC sandwich).
2. Coverage gaps measured and inherited (the h = 12 IRF band at 0.789,
   `var_forecast` at 0.934, two nominal levels only in the registry).
3. Reference runs never made (the `mapie` EnbPI/ACI cross-check, the tsDyn
   TVECM/TVAR/STAR runs, `panelLP.R`).
4. Defaults measured but not flipped (`proxy_ar_sets` still `"delta"`).
5. Round-2 residue (`recession_probit` accepting `link="banana"`, eight
   `Ellipsis` defaults, `check_stationarity` keys unnamed,
   `quantile_regression.converged` a single bool, `dm_test` without a
   kernel option, `smooth_fixed` unbound).
6. Module 10 Tier 3/4 (14 items) and the research-scan list (16).
7. `arima_fit` reduced-Hessian SEs at a boundary; the JL block-length rule
   reconstructed rather than checked against the paper.
8. Documentation with no reproducible rule (`testing.md`'s "290 tests
   across 39 crates load fixtures").

## Not done

Each sweep page ends with its own "Open" section; the items that limit
what this round can claim:

- No Monte-Carlo re-measurement anywhere: every coverage number in an
  open ledger row is the source's own.
- `cargo deny` was not run; hash-pinned requirements were not generated;
  the build scripts of the 25 faer-family crates were not read.
- The 43 `pub(crate)` candidates and 176 crate-private types were counted
  textually, not compile-verified one by one.
- The notebooks' pasted outputs and the showcase figure re-renders were
  not re-executed (claims); the three lab experiments were not re-run
  (hygiene).
- NumPy 1.22 itself was not exercised (1.26.4 was: 1388 passed, 67
  skipped, 0 failed on a `--locked` 0.8.0 wheel); the `rust-version =
  "1.85"` floor is declared but no CI job checks it.
- The published 0.8.0 wheel was not inspected (PyPI unreachable from the
  build container); the T = 10^5 whole-call cells were not re-run after
  the seal; no fuzzing beyond the fixed mutation set.
- All timings were taken on a four-core host shared by seven sweeps;
  orderings are evidence, absolute seconds are not.
