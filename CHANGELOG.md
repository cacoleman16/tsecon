# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows the
pre-1.0 policy in [ROADMAP.md](ROADMAP.md) (minor = breaking allowed, patch =
fixes) until 1.0, then strict [SemVer](https://semver.org/).

## [Unreleased]

Pre-alpha, under active development. Not yet published to PyPI; the library
name is a working codename that will change before the first public release.

### Added — foundations and first model classes
- **Foundations**: Philox RNG (bit-identical to NumPy), special functions and
  the distribution zoo, structured linear algebra (Levinson-Durbin, Toeplitz,
  discrete Lyapunov), the resampling/bootstrap engine, the exact-diffuse
  linear-Gaussian state-space (Kalman) engine, the numerical optimizer suite
  with the Monahan stationarity transform, and the HAC/robust-inference module.
- **Diagnostics**: ACF/PACF, Ljung-Box, Jarque-Bera, ARCH-LM; the full
  unit-root workflow (ADF with MacKinnon p-values, KPSS, `check_stationarity`).
- **Models**: exact-MLE ARIMA; GARCH/GJR/EGARCH with normal and Student-t
  QMLE and Bollerslev-Wooldridge robust standard errors; reduced-form VAR with
  IRF/FEVD/Granger causality/forecasting; trend-cycle filters (HP, one-sided
  HP, Baxter-King, Christiano-Fitzgerald, Hamilton); forecast evaluation
  (Diebold-Mariano with HLN, Theta, accuracy measures); a Minnesota-NIW
  Bayesian VAR with closed-form posterior, posterior impulse-response draws,
  and ArviZ-exact convergence diagnostics; and a first panel slice
  (fixed effects with clustered/Driscoll-Kraay SEs, panel LP, mean-group VAR).
- **Configurable inference**: a uniform `se_type=` on regression estimators;
  configurable interval coverage (`alpha`/`conf_alpha`); cumulative IRF views
  that cumulate draws (Bayesian) or the running sum (frequentist).
- **Docs**: a 13-chapter teaching guide, a worked figure gallery, the
  module-by-module roadmap, and an interactive demo.
- **Packaging**: complete `pyproject.toml` metadata, type stubs with
  `py.typed`, and GitHub Actions CI + a tag-triggered release pipeline with
  PyPI trusted publishing (see [Module 14](docs/roadmap/14-packaging-distribution.md)).

Every estimator is validated against a reference implementation (statsmodels,
SciPy, NumPy, `arch`, `linearmodels`) in the test suite.
