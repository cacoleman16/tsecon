# Module 14 — Packaging and Distribution (PyPI, conda-forge, and beyond)

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**The library is only as useful as `pip install <name>` is reliable. This module is the release-engineering plan: cross-platform wheels built and published automatically from a git tag, a source distribution that builds anywhere a Rust toolchain exists, type stubs so the compiled extension is a first-class typed Python citizen, a conda-forge feedstock, and the versioning, provenance, and reproducibility discipline that a library used in published research must have.** A native-extension library that is hard to install is a library nobody adopts; the packaging is not an afterthought to the econometrics, it is the delivery mechanism for all of it.

## Purpose and scope

tsecon is a Rust core wrapped with PyO3 and built by maturin. That choice was made partly *for* distribution: the pure-Rust numerical stack (faer, no system BLAS) means a wheel is a single self-contained artifact with no shared-library dependencies to hunt down at install time — the class of failure that makes native Python packages painful. This module turns that architectural advantage into an actual, automated, trustworthy release pipeline.

The deliverable is concrete: a user on Linux, macOS (Intel and Apple Silicon), or Windows runs `pip install <name>` and gets a working, correctly-typed library in seconds, with no compiler and no configuration; a user on an unusual platform gets a source distribution that builds cleanly given a Rust toolchain; a conda user gets it from conda-forge; and every published artifact is traceable to the exact commit and built by CI, never by a human laptop.

Scope boundaries: this module owns the *release* mechanics — wheels, sdist, CI, metadata, stubs, publishing, versioning. It does not own the *build* mechanics of the Rust workspace (that is [Module 00](00-architecture.md), which fixes maturin, abi3, the crate layout, and the thin binding crate) nor the API surface being packaged (each domain module). It is the last mile.

## Where the friction is today

- **The current state is a development install only**: `maturin develop` into a local venv. There is no wheel-building CI, no published package, no way for anyone but a contributor with the repo checked out to use the library.
- **`pyproject.toml` is minimal**: name, version, one dependency, the maturin backend. It is missing the metadata PyPI needs to present the package well — long description, classifiers, license expression, project URLs, keywords, Python-version floor, author.
- **No type stubs**: a compiled extension module (`.so`/`.pyd`) exposes nothing to static analyzers or IDEs by default. Without a `.pyi` stub and a `py.typed` marker, every user loses autocomplete and type checking — a large, silent usability tax on a numeric library with many keyword arguments.
- **The name is resolved**: the library ships as `tsecon`. PyPI availability was verified (unregistered) before committing, because the first upload claims the name permanently — see [Module 11](11-docs-ux-adoption.md).
- **No provenance story**: research users need to cite an exact version and, ideally, verify the artifact they installed matches the source. Trusted publishing and build attestations exist to provide this and are not yet configured.

## Inventory

### Tier 1 — Core (blocks the first public release)

| Capability | What it is / why it blocks release | Difficulty | Implementation notes |
|---|---|---|---|
| Complete `pyproject.toml` metadata | The package's PyPI storefront: `description`, `readme` (the repo README rendered on the project page), `license` SPDX expression, `authors`, `keywords`, `classifiers` (Development Status, intended audience = Science/Research, topic = Scientific/Engineering, the supported `Programming Language :: Python` versions and `Rust`), `requires-python`, and `[project.urls]` (Homepage, Documentation, Repository, Issues, Changelog). | Low | maturin reads standard PEP 621 `[project]` metadata; only the build-backend section is maturin-specific. Pin `requires-python = ">=3.9"` to match the abi3-py39 wheel. |
| Cross-platform wheel matrix via `cibuildwheel` / maturin-action | Build wheels for: Linux `x86_64` and `aarch64` (manylinux2014 + musllinux), macOS `x86_64` and `arm64` (or a universal2 wheel), Windows `x86_64`. One abi3 wheel per platform covers all Python ≥ 3.9, so the matrix is platform-only, not platform×Python. | Medium | Use `PyO3/maturin-action` in GitHub Actions (wraps cibuildwheel for Rust). aarch64 Linux via QEMU or native ARM runners. macOS arm64 needs an Apple-Silicon runner or cross-compile with the right target. The abi3 feature (already set) is what collapses the Python dimension. |
| Source distribution (sdist) | A `pip install` fallback for any platform without a prebuilt wheel: ships the Rust source and `Cargo.lock`, builds on the user's machine given a Rust toolchain. | Low | `maturin sdist`; ensure the workspace crates are vendored or path-resolvable inside the sdist (maturin handles workspace path-deps by including them). Test that the sdist builds in a clean container with only rustup + pip. |
| Type stubs (`.pyi`) + `py.typed` | A stub file describing every function signature, keyword, and return type, plus the `py.typed` marker so checkers use it. Without this the compiled module is opaque to mypy/pyright/IDEs. | Medium | Hand-write `tsecon.pyi` initially (the surface is small and stable); later generate from the PyO3 signatures or maintain by hand with a CI check that stub and runtime signatures agree. Ship it inside the wheel next to the extension. This is the single biggest day-one usability win. |
| GitHub Actions release pipeline | On a version tag (`v*`): build the full wheel matrix + sdist, run the Python test suite against each built wheel (not against `maturin develop`), and on success publish to PyPI. On every PR/push: build a representative subset and run tests. | Medium | Two workflows: `ci.yml` (test on push/PR across the OS matrix) and `release.yml` (tag-triggered build-all + publish). Gate publish on all tests green. Artifacts uploaded to the workflow run for inspection before/independent of publish. |
| PyPI Trusted Publishing (OIDC) | Publish via GitHub's OIDC identity instead of a long-lived API token — no secret to leak, and the upload is cryptographically tied to the workflow. | Low | Configure a trusted publisher on PyPI for the repo + `release.yml` environment; use `pypa/gh-action-pypi-publish`. Also enables PEP 740 build attestations automatically. |
| Name resolution + first-publish checklist | Verify the chosen real name is free on PyPI (and conda-forge, and as an import name) before the first upload, since the first upload is irreversible. Reserve it with a `0.0.x` placeholder release if needed. | Low | Coordinate with [Module 11](11-docs-ux-adoption.md) naming. Register on Test PyPI first, do a full dry-run release there, then production. Set the `name` in `pyproject.toml` and the PyO3 `module-name` consistently. |
| Version single-sourcing | One authoritative version, read by both the Rust crate and the Python package, surfaced as `tsecon.__version__`. | Low | Drive from the workspace `Cargo.toml` version (already `env!("CARGO_PKG_VERSION")` at runtime); have maturin read the same for the wheel metadata so the crate, the wheel, and `__version__` can never disagree. |

### Tier 2 — Standard (expected of a mature package)

| Capability | What it is / why it matters | Difficulty | Implementation notes |
|---|---|---|---|
| conda-forge feedstock | Many scientific/econometrics users live in conda; a feedstock makes `conda install -c conda-forge <name>` work and pulls the package into the scientific-Python distribution ecosystem. | Medium | Submit a `staged-recipes` PR after the first PyPI release; the recipe builds from the sdist with a Rust build requirement. conda-forge's bots then auto-maintain version bumps. The architecture doc already commits to "conda-forge from day one." |
| Reproducibility & provenance manifest | Every published wheel records the exact commit, Rust/maturin versions, and build environment; combined with the RNG reproducibility contract, this lets a paper's results be reproduced from a pinned version. | Medium | PEP 740 attestations (free with trusted publishing) cover provenance; add a `tsecon.show_versions()` (numpy/sklearn style) dumping the library version, Rust build info, and key dependency versions for bug reports and replication files. |
| Optional-dependency extras | `pip install <name>[plots]` (matplotlib), `[polars]`, `[all]`; the core wheel stays dependency-minimal (numpy only) per the architecture pillar. | Low | Declare in `[project.optional-dependencies]`. The plotting layer already lazy-imports matplotlib and raises a teaching ImportError naming the extra when absent. |
| Wheel-level smoke tests in CI | After building each wheel, install it into a *clean* environment (no repo checkout, no Rust) and run a representative subset of the Python suite against the installed package — catches packaging bugs (missing stub, wrong module name, forgotten data file) that a `maturin develop` test never sees. | Low | `cibuildwheel`'s `test-command`/`test-requires` run the suite against each built wheel in isolation. This is where "it works on my machine" packaging failures get caught. |
| Changelog + semantic-version policy | A `CHANGELOG.md` (Keep-a-Changelog style) and a documented semver policy: pre-1.0 minor = breaking allowed, patch = fixes; post-1.0 strict semver, with the API-stability policy from [Module 11](11-docs-ux-adoption.md). | Low | Automate release notes from Conventional-Commits-style messages if desired; at minimum keep the changelog by hand and link it from the PyPI project URLs. |
| Documentation site build + hosting | The guide, gallery, and API reference rendered as a versioned site (the docs currently render on GitHub as Markdown). | Medium | MkDocs Material or Sphinx; build in CI and deploy to GitHub Pages / Read the Docs per release. Ties to the [Module 11](11-docs-ux-adoption.md) docs-as-product plan; the executable gallery doubles as the site's examples. |

### Tier 3 — Advanced

| Capability | What it is | Difficulty | Notes |
|---|---|---|---|
| Nightly / pre-release channel | `pip install --pre` builds from `main` for early adopters and downstream CI. | Low | A scheduled workflow publishing `X.Y.Z.devN` wheels to Test PyPI or a separate index. |
| SBOM + supply-chain hardening | A software bill of materials (the Rust dependency tree) attached to releases; `cargo audit` and `cargo deny` gates in CI. | Medium | Rust's dependency tree is small and pure; `cargo-auditable` embeds the dep list in the binary, and `cargo audit` fails CI on known advisories. |
| `pyodide` / WebAssembly wheel | A WASM build so the library runs in the browser (JupyterLite, the docs' live examples) — enabled by the pure-Rust, BLAS-free core. | Research-grade | Emscripten target for the Rust core; the no-system-BLAS choice is what makes this feasible at all. A future differentiator: an econometrics library that runs in a browser notebook. |
| ABI/stability regression tests | CI check that the public Python API surface (function names, signatures, stub) has not changed incompatibly between versions. | Medium | Snapshot the stub + a signature dump; diff on PR; require a version-bump label for intentional breaks. |

## Implementation warnings

- **The first PyPI upload is irreversible** — the name and that version number are claimed forever. Do the entire release dry-run on **Test PyPI** first, and confirm the real name is free everywhere before the production upload.
- **abi3 is what keeps the matrix sane**: one wheel per platform serves all Python ≥ 3.9. If the abi3 feature is ever dropped, the matrix explodes to platform×Python and builds get much slower — treat abi3 as a load-bearing invariant, not a convenience.
- **Test the built wheel, not the dev install.** A `maturin develop` test passing proves the code works; it does *not* prove the *wheel* works. Missing type stubs, a wrong `module-name`, an un-included data file, or a platform-specific symbol only surface when a freshly-built wheel is installed into a clean environment. Wheel-level CI smoke tests are non-negotiable.
- **manylinux/musllinux compliance**: Linux wheels must not link anything outside the manylinux baseline. The pure-Rust core makes this automatic — but a future dependency that pulls in a system library would silently break it. `auditwheel` in CI is the guard.
- **macOS arm64 needs a real target**: cross-compiling or building on an Apple-Silicon runner; a universal2 wheel doubles the binary size. Decide per-arch wheels vs universal2 deliberately.
- **Version drift**: if the crate version, the wheel metadata, and `__version__` are not single-sourced, they *will* diverge and confuse users' bug reports. Read one source.
- **Trusted publishing beats tokens**: a leaked PyPI API token is a supply-chain incident. OIDC trusted publishing has no token to leak and should be the only publish path from day one.

## Dependencies and shared infrastructure

- **Consumes**: the maturin/abi3/thin-binding-crate build setup and the workspace layout ([Module 00](00-architecture.md)); the chosen library name and the README/docs content ([Module 11](11-docs-ux-adoption.md)); the RNG reproducibility contract ([Module 00](00-architecture.md)) that makes version-pinned replication meaningful.
- **Exposes**: the installable package itself — the delivery surface every other module's work reaches users through; `tsecon.__version__` and `show_versions()` for provenance; the CI pipeline other contributors rely on.

## Validation gallery

- **Clean-environment install test** — the built wheel installs and imports in a container with only Python (no Rust, no repo), and a representative test subset passes against it, on every platform in the matrix.
- **sdist build-from-source test** — the source distribution builds and passes tests in a clean container given only rustup + pip.
- **Test PyPI end-to-end** — a full tagged release published to Test PyPI, then `pip install -i test.pypi.org` in a clean environment reproduces the working library, before any production upload.
- **Type-checker acceptance** — `mypy`/`pyright` against a small user script using the stubs produces no errors and full completion, proving the stubs match the runtime surface.
- **conda-forge CI green** — the feedstock's own CI builds the recipe on all platforms.
