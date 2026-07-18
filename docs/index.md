# tsecon

**A high-performance time series econometrics library — a Rust core with a
Python-first API — built to be the centralized home for macro and financial
time series work.**

Most of what economists actually do — structural identification, honest
inference, Bayesian VARs, local projections, nowcasting, volatility, panels —
is scattered across slow, unmaintained, non-interoperable packages. `tsecon`
brings it together: one library, fast simulation, and **every estimator
validated against a golden reference** (statsmodels, `arch`, `linearmodels`,
scikit-learn, ArviZ, SciPy) so the numbers are trustworthy, not just present.

!!! note "Working name"
    `tsecon` is a working codename; the public name is chosen before the first
    release. See the [roadmap](https://github.com/cacoleman16/tsecon/blob/main/ROADMAP.md).

## Start here

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quickstart](quickstart.md)** — install and read
  your first impulse response in about a minute.

- :material-map-marker-path: **[Which model when?](which-model-when.md)** —
  start from your problem ("my regressor is persistent", "I have quarterly GDP
  and monthly indicators") and get routed to the right function.

- :material-book-open-variant: **[The Guide](guide/README.md)** — a full course
  in time series econometrics, from your first autocorrelation plot to
  research-grade structural identification, every concept paired with code that
  runs.

- :material-card-text: **[Model cards & API](reference/README.md)** — the
  assumptions, defaults, failure modes, and validation target of every
  estimator, plus the complete function reference.

- :material-swap-horizontal: **[Migrating?](migration/from-statsmodels.md)** —
  side-by-side maps from statsmodels, R, and Stata, and a cross-package Rosetta
  glossary.

- :material-chart-line: **[Gallery](examples/README.md)** — worked figures in a
  professional house style: IRFs with honest bands, identified sets, forecast
  fans, score-driven volatility.

</div>

## What's inside

93 functions callable from Python today, spanning diagnostics, unit-root and
specification tests; ARIMA, GARCH, and GAS score-driven volatility; VAR / SVAR with
sign-restricted identification, FAVAR, and Diebold-Yilmaz connectedness; local
projections (including state-dependent and LP-IV); Bayesian VARs; GMM / IV-GMM
and IVX predictive regressions; the heterogeneous-panel trio (mean-group,
CCE-MG, PMG); DFM nowcasting with a ragged edge and a news decomposition;
MIDAS; realized volatility; the Nelson-Siegel term structure; forecast
backtesting; and leakage-safe machine learning.

Everything installs as a single wheel with no heavy runtime dependencies, and
the whole library builds and validates from a clean checkout on every push.
