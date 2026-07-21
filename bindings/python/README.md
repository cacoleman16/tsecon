# tsecon

**High-performance time series econometrics: a Rust core with a Python-first API.**

> Pre-1.0 software under active development. The name is settled — `tsecon` is
> what you install and import — but the API may still change before the first
> stable release.

`tsecon` brings the scattered toolkit of modern time series econometrics —
diagnostics and specification tests, ARIMA, GARCH, VARs and structural
identification, local projections, Bayesian VARs, nowcasting, predictive
regressions, panels, the term structure — into one fast, validated, Python-first
library. The compute core is written from scratch in Rust (no BLAS, no heavy
dependencies) so bootstrap inference and Monte Carlo work that is painfully slow
elsewhere runs in seconds, with results bit-reproducible at any thread count.

Every estimator is **validation-gated**: its numbers are checked against a
reference implementation (statsmodels, SciPy, NumPy, `arch`, `linearmodels`,
scikit-learn, ArviZ) or a documented closed form before it ships, and a
cross-library parity gate re-verifies the agreement in CI on every push. The
library ships **no data loaders and makes no network calls** — the only runtime
dependency is NumPy.

## Install

```sh
pip install tsecon            # once published; a single self-contained wheel
```

Until the first PyPI release, build from source with
[maturin](https://www.maturin.rs/) (needs a Rust toolchain and Python >= 3.9):

```sh
pip install maturin
maturin develop --release -m bindings/python/Cargo.toml   # builds + installs into the active venv
```

## A taste

```python
import numpy as np
import tsecon

rng = np.random.default_rng(0)
y = np.cumsum(rng.standard_normal(300))          # a random walk

tsecon.check_stationarity(y)["recommendation"]   # -> "Difference"

# Fit a VAR, read off impulse responses, decompose the variance
data = rng.standard_normal((200, 3))
irf  = tsecon.var_irf(data, lags=2, horizon=16)          # [h][response][shock]
fevd = tsecon.var_fevd(data, lags=2, horizon=16)

# Robust SEs, exact-MLE ARIMA with a forecast fan, GARCH with robust SEs,
# instrumented local projections, a fiscal multiplier, a Bayesian VAR...
tsecon.ols(y, X, se_type="hac")
tsecon.arima_fit(y, p=1, d=1, q=1, forecast_steps=12, conf_alpha=0.1)
tsecon.garch_fit(returns, vol="gjr", dist="t")
tsecon.lp_multiplier(y_out, spending, instrument, horizons=20)
tsecon.bvar_irf_draws(data, lags=2, horizon=12, n_draws=800)
```

Every function takes plain NumPy arrays and returns arrays or dicts of
documented keys. An opt-in `tsecon.results` layer wraps the same calls in
objects that render themselves (`.summary()`, `.plot_irf()`) without changing
the dict contract.

## What's here today

94 functions across the full applied workflow: the diagnostic battery and
unit-root workflow (ADF, KPSS, `check_stationarity`); specification and
stability tests (White, Breusch-Pagan, RESET, Chow, CUSUM); robust and HAC
standard errors; the bootstrap family; an exact-diffuse Kalman filter; ARIMA,
GARCH/GJR/EGARCH, and GAS score-driven volatility; VAR/SVAR with sign-restricted
identification, FAVAR, and Diebold-Yilmaz connectedness; local projections
(lag-augmented, LP-IV, state-dependent, and the Ramey-Zubairy integral
multiplier); a Minnesota-NIW Bayesian VAR; GMM and IV-GMM; predictive
regressions with IVX inference; heterogeneous panels (mean group, CCE-MG, PMG);
cointegration and Markov switching; MIDAS and DFM nowcasting with a news
decomposition; multivariate GARCH; realized volatility; spectral analysis;
long memory; recession-probability models; survey-expectations tools; the
Nelson-Siegel/Svensson term structure with the arbitrage-free (AFNS)
adjustment; and a linear rational-expectations (DSGE-lite) solver.

The library ships with complete type stubs (`py.typed`), so autocomplete and
type checking work out of the box.

## Learn more

The full documentation — a 15-chapter guide to time series econometrics, model
cards for every estimator family, a worked figure gallery, two replications of
published results, a Monte Carlo validation suite, and an honest benchmark
harness — lives in the [project repository](https://github.com/cacoleman16/tsecon).

## License

MIT OR Apache-2.0.
