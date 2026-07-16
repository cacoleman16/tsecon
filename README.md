# tsecon (working codename)

A high-performance time series econometrics library: Rust core, Python-first API.
The name `tsecon` is a placeholder — the real name is decided before first public
release (see [ROADMAP.md §9](ROADMAP.md)).

**Status: Phase 0 (foundations) under construction.** See [ROADMAP.md](ROADMAP.md)
for the master plan and [docs/roadmap/](docs/roadmap/) for the 13 module
specifications.

## Layout

| Path | Contents |
|---|---|
| `crates/tsecon-rng` | Philox counter-based RNG, NumPy bit-compatible; SeedSequence; parallel substreams |
| `crates/tsecon-stats` | Special functions and the innovation-distribution zoo (normal, t, GED, skew-t) |
| `crates/tsecon-linalg` | Structured solvers: Levinson-Durbin, Toeplitz, discrete Lyapunov, companion-form utilities |
| `crates/tsecon-bootstrap` | Resampling engine: moving-block / stationary / wild bootstrap on RNG substreams |
| `crates/tsecon-diag` | Diagnostics: ACF/PACF, Ljung-Box, Jarque-Bera, ARCH-LM |
| `crates/tsecon-ssm` | Linear-Gaussian state-space engine: Kalman filter/smoother, exact diffuse initialization |
| `fixtures/` | Golden values generated from NumPy/SciPy/statsmodels (`generate_fixtures.py`); Rust tests must match them |
| `docs/roadmap/` | Module specifications |

## Development

```sh
cargo test                          # run all Rust tests (golden-value + property)
python3 fixtures/generate_fixtures.py   # regenerate golden fixtures (pinned versions recorded in each JSON)
```

## Design rules (short form)

- **Validation-gated**: nothing merges without a golden target (fixture, reference implementation, or published table).
- **Reproducible parallelism**: all randomness flows through Philox substreams; results are bit-identical at any thread count.
- **Single owner**: shared capabilities (bootstrap, HAC, SSM, distributions...) are implemented once, consumed everywhere (ROADMAP §5).
