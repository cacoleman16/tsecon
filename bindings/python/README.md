# tsecon

**High-performance time series econometrics: a Rust core with a Python-first API.**

> `tsecon` is a working codename. The name will change before the first stable
> release. This is pre-alpha software under active development.

`tsecon` brings the scattered toolkit of modern time series econometrics —
diagnostics, ARIMA, GARCH, VARs, structural identification, local projections,
Bayesian VARs, forecasting evaluation — into one fast, validated, Python-first
library. The compute core is written from scratch in Rust (no BLAS, no heavy
dependencies) so bootstrap inference and Monte Carlo work that is painfully
slow elsewhere runs in seconds, with results bit-reproducible at any thread
count.

Every estimator is **validation-gated**: its numbers are checked against a
reference implementation (statsmodels, SciPy, NumPy, `arch`, `linearmodels`)
down to tight numerical tolerances before it ships.

## Install

Not yet on PyPI — the public package name is being finalized, and the wheel
is published under it at first release. For now, build from source with
[maturin](https://www.maturin.rs/) (needs a Rust toolchain and Python ≥ 3.9):

```sh
pip install maturin
maturin develop -m bindings/python/Cargo.toml   # builds + installs into the active venv
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

# Robust standard errors, exact-MLE ARIMA with a forecast fan,
# GARCH with Bollerslev-Wooldridge robust SEs, a Bayesian VAR posterior...
tsecon.ols(y, X, se_type="hac")
tsecon.arima_fit(y, p=1, d=1, q=1, forecast_steps=12, conf_alpha=0.1)
tsecon.garch_fit(returns, vol="gjr", dist="t")
tsecon.bvar_irf_draws(data, lags=2, horizon=12, n_draws=800)
```

## What's here today

Diagnostics (ACF/PACF, Ljung-Box, Jarque-Bera, ARCH-LM), the full unit-root
workflow (ADF, KPSS, `check_stationarity`), robust standard errors
(HAC/Newey-West), the bootstrap family, an exact-diffuse Kalman filter,
ARIMA, GARCH/GJR/EGARCH, VAR (IRF/FEVD/Granger/forecast), trend-cycle filters,
forecast evaluation (Diebold-Mariano, Theta, accuracy measures), and a
Minnesota-NIW Bayesian VAR with posterior impulse responses.

The library ships with type stubs (`py.typed`), so autocomplete and type
checking work out of the box.

## Learn more

The full documentation — a 13-chapter guide to time series econometrics, a
worked gallery, the module-by-module roadmap, and an interactive demo — lives
in the [project repository](https://github.com/cacoleman16/tsecon).

## License

MIT OR Apache-2.0.
