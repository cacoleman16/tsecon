# Contributing to tsecon

Thanks for your interest in contributing. `tsecon` (a working codename — the
public name is resolved before the first release) is a Rust-core, Python-first
time-series econometrics library, and it is built on one non-negotiable rule:
**nothing lands without a golden target it has to hit.** This guide explains how
to build it, how to run the tests, the validation discipline every change must
respect, the CI gates you need to keep green, and how to add a new estimator end
to end.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md) and to
the [governance model](GOVERNANCE.md). Public-API changes go through a short RFC
step described in that governance doc — read it before proposing one.

## Table of contents

- [Prerequisites](#prerequisites)
- [Building the library](#building-the-library)
- [Running the tests](#running-the-tests)
- [Validation-first: the golden-fixture discipline](#validation-first-the-golden-fixture-discipline)
- [The CI gates](#the-ci-gates)
- [Commit and pull-request conventions](#commit-and-pull-request-conventions)
- [How to add a new estimator](#how-to-add-a-new-estimator)
- [Licensing of contributions](#licensing-of-contributions)

## Prerequisites

- **Rust** — the toolchain is *pinned* to **1.97.1** in
  [`rust-toolchain.toml`](rust-toolchain.toml), so a `rustup`-managed checkout
  automatically uses the same compiler and clippy that CI does. (The workspace's
  declared minimum supported Rust version in [`Cargo.toml`](Cargo.toml) is
  `1.85`; CI builds and lints on the pinned 1.97.1.) `rustfmt` and `clippy` are
  listed as required components, so `rustup` installs them for you.
- **Python ≥ 3.9** — the wheel targets `abi3-py39`, so one build serves every
  Python from 3.9 up (`requires-python = ">=3.9"` in
  [`bindings/python/pyproject.toml`](bindings/python/pyproject.toml)).
- **maturin ≥ 1.14, < 2.0** — the PyO3 build backend that turns the Rust crate
  into an importable Python extension.

A working development environment is a Python virtual environment with `maturin`
installed and the wheel built into it. The reference dev venv in this repo lives
at `.venv/`.

## Building the library

Create/activate a virtual environment, install `maturin`, then build the
extension **into** that environment:

```sh
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install maturin
maturin develop -m bindings/python/Cargo.toml   # compiles the Rust core + installs into the active venv
```

`maturin develop` compiles the whole Rust workspace and installs the resulting
extension into the active interpreter.

The package uses maturin's **mixed Rust/Python layout**. The compiled extension
is installed as the private `tsecon._core` (`module-name = "tsecon._core"`,
`python-source = "python"` in `bindings/python/pyproject.toml`), and the public
`tsecon` package lives at `bindings/python/python/tsecon/`, whose `__init__.py`
re-exports the whole compiled surface. Estimators are therefore still the
compiled functions with no Python indirection — the layer exists so that
pure-Python submodules can sit beside the Rust core (today
[`tsecon.results`](bindings/python/python/tsecon/results/__init__.py), the
opt-in rendering layer).

Layout, and where to add things:

```
bindings/python/
  src/lib.rs                     # the PyO3 bindings (one shared file)
  python/tsecon/__init__.py      # re-exports tsecon._core; add submodules here
  python/tsecon/__init__.pyi     # the hand-written type stub (public surface)
  python/tsecon/py.typed         # PEP 561 marker
  python/tsecon/results/         # pure-Python: the Results-object facade
```

The stub and `py.typed` ship inside the package so autocomplete and `mypy` see
fully typed signatures. Because a source package now exists in the tree, the
wheel test in CI asserts the *installed* module is under test, not the source.

Verify the build:

```sh
python -c "import tsecon; print(tsecon.__version__)"
```

For a release-optimized local build add `--release` to the `maturin develop`
call. CI builds and tests the *installed wheel* (not an editable
`maturin develop` install) precisely so packaging bugs — a missing stub, a wrong
module name, un-included data — surface in CI rather than for a user.

## Running the tests

### Rust (the core)

```sh
cargo test --workspace
```

This is exactly what CI runs. It exercises the golden-fixture tests and the
property/Monte-Carlo tests across all 37 crates. If you want a pure-Rust run that
skips building the PyO3 extension crate (which links against Python), exclude it:

```sh
cargo test --workspace --exclude tsecon-python
```

(`tsecon-python`, in `bindings/python/`, is the `cdylib` binding crate; it holds
no Rust tests of its own — its behavior is covered by the Python suite.)

### Python (the bindings)

```sh
.venv/bin/python -m pytest bindings/python/tests
```

Run the Python tests **in place** (from the repo root, pointing pytest at
`bindings/python/tests`). The tests locate their golden fixtures with a
repo-relative path (`Path(__file__).parents[3] / "fixtures"`), so moving them or
running from a copied-out directory breaks fixture resolution. These tests assert
the Python bindings reproduce the same golden values the Rust tests hit, and they
include the stub-sync and API-reference guards described below.

## Validation-first: the golden-fixture discipline

This is the heart of the project. Every estimator is gated against a **golden
fixture**: a JSON file of reference values under [`fixtures/`](fixtures/) that the
Rust (and Python) tests must reproduce to a tight tolerance. The discipline has a
few hard rules — please internalize them before adding code:

1. **Reference libraries are used offline, at fixture-generation time only —
   never at runtime.** `tsecon` depends on nothing heavier than NumPy at runtime.
   statsmodels, `arch`, `linearmodels`, scikit-learn, ArviZ, and SciPy appear
   only in the `fixtures/generate_*.py` scripts, which you run once to *produce*
   the JSON goldens. The shipped library never imports them.

2. **Fixture generators must not import `tsecon`.** A golden reference that called
   the code it is supposed to validate would be circular and worthless. Generators
   compute their reference values from an *independent* library or from a
   documented closed-form formula transcribed in the generator's docstring — never
   from `tsecon` itself. (You can confirm the current tree honors this:
   `grep -l "import tsecon" fixtures/*.py` returns nothing.)

3. **Fixtures store only derived numeric values — never a redistributed dataset.**
   Each generator produces its numbers one of two ways: from *seeded* NumPy
   `default_rng` draws through a known data-generating process, or as
   *transformations* of the two public-domain reference series bundled with
   statsmodels (the Nile river-flow series and the US macrodata series). Only the
   statistics and transforms are stored, not the raw data. See
   [`fixtures/README.md`](fixtures/README.md).

4. **Every fixture pins the reference versions used.** Generators write a `_meta`
   block (NumPy / SciPy / statsmodels / Python versions) into each JSON so the
   values are reproducible. Regenerate with, e.g.:

   ```sh
   .venv/bin/python fixtures/generate_fixtures.py
   ```

In short: a new method needs a **named validation target** — a reference value, a
documented formula, or a Monte-Carlo size/power check — *before* it lands. A
feature without one is not done.

## The CI gates

Three GitHub Actions workflows must pass. Reproduce them locally before you open
a PR so nothing surprises you.

### `ci.yml` — the core

| Gate | Command CI runs | What it protects |
|---|---|---|
| Formatting | `cargo fmt --all --check` | Uniform Rust formatting |
| Lints | `cargo clippy --workspace --all-targets -- -D warnings` | **Zero** clippy warnings — warnings fail the build |
| Rust tests | `cargo test --workspace` | Golden + property tests, all crates |
| Wheel + Python tests | build the abi3 wheel, `pip install` it, then `pytest bindings/python/tests` on Linux/macOS/Windows | The *installed* artifact works and reproduces the goldens cross-platform |
| Type stubs | `mypy --strict` over a script that exercises typed signatures | The hand-written stubs actually type-check |

Because clippy runs with `-D warnings` on the pinned toolchain, run
`cargo clippy --workspace --all-targets` locally on 1.97.1 and fix every lint —
a warning your local floating `stable` doesn't emit will still fail CI. That is
exactly why the toolchain is pinned.

Two guards in the Python suite
([`bindings/python/tests/test_stub_sync.py`](bindings/python/tests/test_stub_sync.py))
are worth calling out because they catch the most common "forgot a step" mistakes:

- **Stub sync** — `tsecon.pyi` must describe *exactly* the runtime function
  surface. Add or remove a binding without updating the stub and
  `test_stub_matches_runtime` fails. The `py.typed` marker must also be present.
- **API reference not stale** — `docs/reference/api.md` is generated from the stub
  by [`docs/gen_api_reference.py`](docs/gen_api_reference.py).
  `test_api_reference_not_stale` regenerates it and asserts the committed file is
  byte-identical, so a forgotten regeneration fails CI instead of shipping a stale
  reference. After changing the stub, run:

  ```sh
  .venv/bin/python docs/gen_api_reference.py
  ```

  and commit the result.

### `docs.yml` — the documentation site

The docs site (mkdocs-material) is validated with:

```sh
pip install -r docs/requirements.txt
python docs/gen_api_reference.py     # regenerate the API reference from the stub
mkdocs build --strict                # fails on ANY broken link, missing nav entry, or stale reference
```

`mkdocs build --strict` must stay clean. Preview locally with `mkdocs serve`.

### `release.yml`

Tag-triggered; builds the cross-platform wheel matrix + sdist and publishes to
PyPI via trusted publishing. You do not run this — it fires on a `v*` tag — but be
aware that a green `main` is what a release is cut from.

## Commit and pull-request conventions

- **Branch and PR against `main`.** Open a pull request; all three workflows above
  must be green before it can merge.
- **Keep changes focused.** One logical change per PR makes review (and later
  `git bisect`) tractable.
- **Update the changelog.** Add a bullet under the `[Unreleased]` section of
  [`CHANGELOG.md`](CHANGELOG.md), which follows
  [Keep a Changelog](https://keepachangelog.com/). Versioning follows the pre-1.0
  policy documented there (minor = breaking allowed, patch = fixes) until 1.0.
- **Write clear commit messages.** [Conventional Commits](https://www.conventionalcommits.org/)
  style (`feat:`, `fix:`, `docs:`, …) is encouraged — it makes release notes easy
  to assemble — but is not currently enforced by a hook.
- **Docs travel with code.** A feature without its model card is unfinished (see
  the estimator checklist below). Model cards follow the house anatomy exemplified
  by
  [`docs/reference/model-cards/predictive-regressions.md`](docs/reference/model-cards/predictive-regressions.md).
- **API changes need an RFC first.** Adding, renaming, removing, or changing the
  signature or defaults of any public function goes through the short RFC step in
  [GOVERNANCE.md](GOVERNANCE.md).

## How to add a new estimator

The end-to-end path for a new method. Using the predictive-regressions family as
a concrete example of the shape (crate `crates/tsecon-predreg`, functions
`predictive_regression` / `ivx_test`, card
`docs/reference/model-cards/predictive-regressions.md`):

1. **Write the Rust crate.** Create `crates/tsecon-<name>/` with `Cargo.toml`, a
   `src/` implementing the estimator, and a `tests/` directory. The established
   pattern is a `tests/golden.rs` (reproduces the JSON fixture to tolerance) plus a
   `tests/properties.rs` (seeded Monte-Carlo / invariance checks). Add the crate to
   the `members` list in the workspace [`Cargo.toml`](Cargo.toml).

2. **Generate the golden fixture.** Add `fixtures/generate_<name>_fixtures.py`
   that writes `fixtures/<name>.json` from an independent reference or a
   documented formula (see the discipline above — **do not import `tsecon`**, and
   pin the reference versions in `_meta`). Run it and commit the JSON.

3. **Add the Python binding.** In
   [`bindings/python/src/lib.rs`](bindings/python/src/lib.rs), write a
   `#[pyfunction]` that delegates to your crate (the binding crate should already
   depend on it — add a `path` dependency in
   [`bindings/python/Cargo.toml`](bindings/python/Cargo.toml)), and register it in
   the `#[pymodule] fn tsecon(...)` block with
   `m.add_function(wrap_pyfunction!(<name>, m)?)?;`. Return plain NumPy arrays and
   dicts — no framework objects.

4. **Update the type stub.** Add the signature and a docstring to
   [`bindings/python/python/tsecon/__init__.pyi`](bindings/python/python/tsecon/__init__.pyi) under the appropriate
   `# ---- section ----` heading (the section drives the API-reference grouping).
   The stub-sync guard requires the stub to match the runtime surface exactly.

5. **Regenerate the API reference.** Run
   `.venv/bin/python docs/gen_api_reference.py` and commit the updated
   `docs/reference/api.md` (the not-stale guard checks this).

6. **Write the model card.** Add `docs/reference/model-cards/<family>.md` following
   the house anatomy (**what it estimates · assumptions · when to use (and when
   not) · key arguments and defaults (and why) · how to read the output · failure
   modes · validated against · references · a runnable example**). Add a row to the
   model-card table in `docs/reference/README.md`, and a nav entry in
   [`mkdocs.yml`](mkdocs.yml). Every number in the card must be real output you ran.

7. **Add a Python binding test.** Add `bindings/python/tests/test_<name>.py` that
   calls the new function and checks it against the same fixture / self-consistency
   properties.

8. **Run the full local gate** — `cargo fmt --all --check`,
   `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test
   --workspace`, `pytest bindings/python/tests`, and `mkdocs build --strict` —
   before opening the PR.

## Licensing of contributions

`tsecon` is dual-licensed under [MIT](LICENSE-MIT) **or** [Apache-2.0](LICENSE-APACHE),
at the user's option. Unless you state otherwise, any contribution you
intentionally submit for inclusion in the work, as defined in the Apache-2.0
license, shall be dual-licensed as above, without any additional terms or
conditions.
