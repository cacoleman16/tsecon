# tsecon

**A high-performance time series econometrics library — a Rust core with a
Python-first API — built to be the centralized home for macro and financial
time series work.**

[![CI](https://github.com/cacoleman16/tsecon/actions/workflows/ci.yml/badge.svg)](https://github.com/cacoleman16/tsecon/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)

> Pre-1.0 and moving fast. The name is settled (`tsecon`), but the API may
> still change before the first release — see [ROADMAP.md](ROADMAP.md).

Most of what economists actually do — structural identification, honest
inference, Bayesian VARs, local projections, nowcasting, volatility, panels —
is scattered across slow, unmaintained, non-interoperable packages. `tsecon`
brings it together in one library, fast enough for simulation work, with
**every estimator validated against a golden reference** (statsmodels, `arch`,
`linearmodels`, scikit-learn, ArviZ, SciPy) so the numbers are trustworthy, not
just present.

## Status

Phases 0–1 complete; Phases 2–4 substantially landed. **41 Rust crates,
1001 Rust + 491 Python tests — all green and golden-fixture-gated.** The whole
library builds and tests from a clean checkout on every push (CI matrix on
Linux/macOS/Windows), and a strict-built docs site keeps the documentation
honest. See [ROADMAP.md §0](ROADMAP.md#0-current-build-status) for the live
snapshot of what's built and what's next.

## Install

Build the wheel from source with [maturin](https://www.maturin.rs/) (a Rust
toolchain and Python ≥ 3.9 are required):

```sh
pip install maturin
maturin develop -m bindings/python/Cargo.toml   # builds + installs into the active venv
```

A published wheel (no Rust toolchain needed) lands on PyPI with the first
tagged release.

## Quickstart

```python
import numpy as np
import tsecon

rng = np.random.default_rng(0)

# Is my series stationary? (ADF + KPSS, with a verdict)
y = np.cumsum(rng.standard_normal(200))          # a random walk
tsecon.check_stationarity(y)["quadrant"]          # -> "UnitRoot"
tsecon.check_stationarity(np.diff(y))["quadrant"] # -> "Stationary"

# A small macro panel: fit a VAR and read an impulse response.
data = np.cumsum(rng.standard_normal((200, 3)), axis=0)
fit = tsecon.var_fit(data, lags=2)
irf = tsecon.var_irf(data, lags=2, horizon=12)   # orthogonalized IRF
```

The **[Quickstart](docs/quickstart.md)** and the symptom-driven
**[Which model when?](docs/which-model-when.md)** guide are the fastest way in.

## Documentation

- **[The Guide](docs/guide/README.md)** — a free 15-chapter course in time
  series econometrics, beginner to research-grade, mirroring the library.
- **[Which model when?](docs/which-model-when.md)** — start from your problem,
  get routed to the right function.
- **[Model cards & API reference](docs/reference/README.md)** — the
  assumptions, defaults, failure modes, and validation target of every
  estimator, plus the full 121-function reference.
- **[Migration guides](docs/migration/from-statsmodels.md)** — from
  statsmodels, R, and Stata, with a Rosetta glossary.
- **[Gallery](docs/examples/README.md)** — worked figures in a professional
  house style.

The docs render as a site with `pip install -r docs/requirements.txt &&
mkdocs serve`.

## What's inside

121 functions callable from Python today: diagnostics, unit-root and
specification tests (White/Breusch-Pagan, RESET, Chow, CUSUM);
ARIMA, GARCH, and GAS score-driven volatility; VAR/SVAR with sign-restricted
identification, FAVAR, and Diebold-Yilmaz connectedness; local projections
(state-dependent and LP-IV); Bayesian VARs; GMM/IV-GMM and IVX predictive
regressions; the heterogeneous-panel trio (mean-group, CCE-MG, PMG); DFM
nowcasting (two-step and one-step MLE) with a ragged edge and a news
decomposition; MIDAS; realized volatility; the Nelson-Siegel term structure;
forecast backtesting; and leakage-safe machine learning.

## Architecture

- **Rust core, Python API.** 37 workspace crates behind PyO3/`abi3` bindings;
  a single self-contained wheel with no heavy runtime dependencies.
- **Validation-gated.** Nothing lands without a golden target — a reference
  value, a documented formula, or a Monte-Carlo size/power check. Reference
  libraries are used *only* to generate fixtures offline, never at runtime.
- **Single owner.** Shared capabilities (RNG, HAC, state space, distributions,
  bootstrap) are implemented once and consumed everywhere (ROADMAP §5).
- **Reproducible.** All randomness flows through NumPy-bit-compatible Philox
  substreams; results are identical at any thread count.

## Development

```sh
cargo test --workspace --exclude tsecon-python      # Rust tests (golden + property)
cargo clippy --workspace --all-targets -- -D warnings
maturin develop -m bindings/python/Cargo.toml && pytest bindings/python/tests
python fixtures/generate_fixtures.py                # regenerate goldens (pinned versions in each JSON)
```

(On macOS the `tsecon-python` crate's test binary can't find `libpython`, which
aborts the run and truncates the count — hence `--exclude`. Linux CI runs
everything. The bindings are covered by the pytest suite.)

## Correctness and performance

Two independent kinds of evidence, both reproducible:

- **[Validation matrix](docs/reference/validation-matrix.md)** — what each family
  is checked against (statsmodels, SciPy, `arch`, `linearmodels`, scikit-learn,
  ArviZ, or a documented closed form), with fixture, test, and tolerance.
- **[Monte Carlo suite](docs/examples/monte-carlo.md)** — the statistical
  properties a fixture match can't prove: IVX holds its 5% size at an exact unit
  root where the OLS t-test rejects 28% of the time; HAC restores CI coverage;
  the AR(1) estimator is consistent with the textbook finite-sample bias.
- **[Benchmarks](benchmarks/)** — a parity-first harness: estimates must match a
  reference *before* anything is timed. On a release build, ADF is ~13× and
  VAR(2) ~24× faster than statsmodels — and GARCH QMLE is ~4× *slower* than
  `arch`, which we publish too.

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) for the
build, the validation-first discipline (no estimator merges without a named
golden target), and the CI gates. Participation is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md); project decision-making is described in
[GOVERNANCE.md](GOVERNANCE.md).

## Citation

If you use tsecon in research, please cite it via [CITATION.cff](CITATION.cff)
(GitHub's "Cite this repository" button renders it). A software paper draft
lives in [`paper/`](paper/).

## License

Dual-licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
