# tsecon Quickstart — 60 seconds to your first impulse response

This is the on-ramp. In one page you will install **tsecon**, fit a vector
autoregression to a bundled dataset, and read an impulse response off it — the
single most common thing an empirical macroeconomist asks a time-series
library to do. Everything below runs today against the shipped API; the code
blocks are the same ones the test suite exercises.

> **A note on the name.** `tsecon` is a working codename. The final name (and
> its PyPI availability) is settled before the first public release — see
> [ROADMAP.md §9](../ROADMAP.md). Until then, `tsecon` is what you install and
> what you import.

---

## Install

tsecon is a compiled Rust extension with a thin Python API, distributed as a
wheel. Build it from the repository with [maturin](https://www.maturin.rs/) and
install the result:

```sh
maturin build --release                       # writes target/wheels/tsecon-0.0.1-*.whl
pip install target/wheels/tsecon-0.0.1-*.whl  # the codename package: `tsecon`
```

The core wheel depends only on NumPy. Plotting is opt-in (`pip install
'tsecon[plots]'` pulls in matplotlib). Confirm the install and see how much is
on the shelf:

```python
import tsecon
print(tsecon.__version__)                                       # 0.0.1
print(sum(callable(getattr(tsecon, n)) for n in dir(tsecon)     # 93
          if not n.startswith("_")))
```

---

## Hello, impulse response

The repository ships golden fixtures in [`fixtures/`](../fixtures/) — the same
data the library is validated against. One of them, `var.json`, holds 100
quarters of (100×dlog) GDP, consumption, and investment growth. Run this from
the repository root:

```python
import json, numpy as np, tsecon

y = np.array(json.load(open("fixtures/var.json"))["data_100dlog_gdp_cons_inv"])
fit = tsecon.var_fit(y, lags=2)                                  # VAR(2) by OLS
irf = np.array(tsecon.var_irf(y, lags=2, horizon=10, orth=True))  # [horizon][response][shock]

np.set_printoptions(precision=3, suppress=True)
print(irf[:, 1, 0])   # consumption's response to a one-SD GDP shock, h = 0..10
```

```
[0.395 0.107 0.106 0.056 0.035 0.022 0.013 0.008 0.005 0.003 0.002]
```

That is the whole idea: a one-standard-deviation surprise to GDP lifts
consumption by 0.40 on impact, and the effect decays smoothly toward zero over
the following quarters — a stable, sensible dynamic response. `orth=True`
orthogonalizes the shocks through the Cholesky factor of the residual
covariance, using the column ordering of `y` (GDP → consumption → investment);
`irf[h][i][j]` is the response of variable `i` to a shock in variable `j` at
horizon `h`.

The `fit` object carries the rest of the story: `fit["params"]`, `fit["aic"]`
/ `fit["bic"]` / `fit["hqic"]`, the residual covariance `fit["sigma_u"]`, and
`fit["is_stable"]` — the stability verdict. (The roots are the *reciprocal*
characteristic roots, so a stable VAR keeps them all outside the unit circle;
`fit["min_root"] > 1` is the equivalent numeric check, while `fit["max_root"]`
is the root farthest from the circle and is not a verdict on its own.) From
here, `tsecon.var_fevd`, `tsecon.var_forecast`, and `tsecon.var_granger` take
the same `(data, lags)` arguments.

---

## The API at a glance

The 93 functions, grouped by the task they serve. Every one is a plain
function that takes arrays and returns a NumPy array or a dict of documented
keys — no fit/predict objects to learn. Authoritative signatures, defaults,
and docstrings live in
[`bindings/python/python/tsecon/__init__.pyi`](../bindings/python/python/tsecon/__init__.pyi).

### Diagnostics and data prep

| Function | What it does |
|---|---|
| `acf` | Autocorrelation function with Bartlett standard errors |
| `pacf` | Partial autocorrelations (Yule-Walker or OLS) |
| `ljung_box` | Portmanteau white-noise test |
| `jarque_bera` | Normality test from skewness and kurtosis |
| `arch_lm` | Engle's test for conditional heteroskedasticity |
| `adf` | Augmented Dickey-Fuller unit-root test (MacKinnon p-values) |
| `kpss` | KPSS stationarity test — the ADF complement |
| `check_stationarity` | The ADF + KPSS confirmatory workflow, with a recommendation |
| `ols` | Linear regression with nonrobust / HC / HAC standard errors |
| `long_run_variance` | Kernel long-run variance of a series |
| `periodogram` | Raw spectral density (matches SciPy) |
| `welch` | Welch averaged-periodogram spectral density |
| `coherence` | Magnitude-squared coherence between two series |
| `bootstrap_indices` | iid / moving-block / circular / stationary resampling indices |
| `optimal_block_length` | Politis-White automatic block length |
| `philox_uniforms` | Reproducible uniform draws, bit-identical to NumPy |

### Univariate models and filters

| Function | What it does |
|---|---|
| `arima_fit` | Exact-MLE ARIMA(p,d,q) with optional forecast bands |
| `ar_loglik` | Exact Gaussian AR(p) log-likelihood at fixed parameters |
| `local_level_smooth` | Local-level Kalman filter + smoother (handles missing data) |
| `hp_filter` | Hodrick-Prescott trend/cycle decomposition |
| `bk_filter` | Baxter-King band-pass filter |
| `cf_filter` | Christiano-Fitzgerald band-pass filter |
| `hamilton_filter` | Hamilton's regression-based HP alternative |
| `markov_switching_ar` | Regime-switching AR fitted by EM (Hamilton 1989) |

### Volatility

| Function | What it does |
|---|---|
| `garch_fit` | GARCH / GJR / EGARCH by QMLE with robust standard errors |
| `gas_volatility` | GAS(1,1) score-driven volatility |
| `ccc_garch` | Constant-conditional-correlation multivariate GARCH |
| `dcc_garch` | Dynamic-conditional-correlation multivariate GARCH |
| `realized_measures` | Realized variance, bipower variation, and jump component |
| `har_rv` | Corsi HAR-RV regression with HAC standard errors |
| `realized_quarticity` | Realized quarticity (the sampling variance of RV) |
| `tripower_quarticity` | Jump-robust integrated quarticity |
| `bns_jump_test` | Barndorff-Nielsen-Shephard ratio jump test |
| `realized_range` | Parkinson / Garman-Klass range variance from OHLC bars |

### Multivariate and structural

| Function | What it does |
|---|---|
| `var_fit` | Fit a VAR(p) by OLS: params, covariance, ICs, stability |
| `var_irf` | Orthogonalized or raw impulse responses |
| `var_fevd` | Forecast-error variance decomposition |
| `var_forecast` | Iterated VAR forecasts with intervals |
| `var_granger` | Granger-causality F test |
| `sign_restricted_svar` | Sign-restricted Bayesian SVAR identified-set bands |
| `favar` | Factor-augmented VAR policy-shock IRFs (Bernanke-Boivin-Eliasz) |
| `johansen` | Johansen cointegration rank test |
| `vecm` | VECM maximum-likelihood estimation |
| `connectedness` | Diebold-Yilmaz spillover / connectedness measures |
| `factor_model` | PCA factor model with Bai-Ng factor selection |

### Local projections

| Function | What it does |
|---|---|
| `lp` | Local-projection impulse responses (lag-augmented or HAC SEs) |
| `lp_iv` | Instrumented local projections with a first-stage F diagnostic |
| `lp_state` | State-dependent (interacted) local projections (Ramey-Zubairy) |

### Forecasting and evaluation

| Function | What it does |
|---|---|
| `theta_forecast` | The Theta method (Assimakopoulos-Nikolopoulos) |
| `accuracy` | Forecast accuracy measures (RMSE, MAE, MAPE, MASE, RMSSE…) |
| `backtest` | Rolling / expanding pseudo-out-of-sample backtest |
| `dm_test` | Diebold-Mariano equal-accuracy test (HLN-corrected) |
| `cw_test` | Clark-West test for nested models |
| `gw_test` | Giacomini-White test of equal predictive ability |

### Bayesian

| Function | What it does |
|---|---|
| `bvar_fit` | Minnesota-NIW conjugate BVAR posterior + log marginal likelihood |
| `bvar_irf_draws` | Posterior Cholesky-IRF draws for credible bands |
| `mcmc_diagnostics` | Split R-hat and bulk/tail effective sample size |

### Panel time series

| Function | What it does |
|---|---|
| `panel_fe` | Fixed-effects panel OLS (cluster or Driscoll-Kraay SEs) |
| `panel_lp` | Panel local projection of a common shock |
| `panel_mean_group` | Mean-group / CCE-MG heterogeneous-panel estimator (Pesaran) |
| `panel_pmg` | Pooled Mean Group ARDL estimator (Pesaran-Shin-Smith) |
| `mean_group_var` | Pesaran-Smith mean-group panel VAR |

### Nowcasting and mixed frequency

| Function | What it does |
|---|---|
| `dfm_nowcast` | Dynamic-factor-model nowcast over a ragged data edge |
| `dfm_news` | News / update decomposition of a nowcast revision |
| `midas_weights` | MIDAS weighting kernels (exp-Almon or beta) |
| `umidas` | Unrestricted mixed-frequency (U-MIDAS) regression |
| `weighted_midas` | Weighted MIDAS estimated by nonlinear least squares |

### Regression, machine learning, and GMM

| Function | What it does |
|---|---|
| `ridge` | Ridge regression, closed form (scikit-learn objective) |
| `lasso` | Lasso via coordinate descent |
| `elastic_net` | Elastic net via coordinate descent |
| `adaptive_lasso` | Adaptive LASSO with oracle-property weights (Zou) |
| `lasso_path` | Elastic-net regularization path with AIC/BIC selection |
| `cv_splits` | Leakage-safe CV splits (expanding / rolling / purged k-fold) |
| `iv_gmm` | Linear IV-GMM with robust or HAC weighting and a Hansen J test |
| `gmm_nonlinear` | Nonlinear GMM over a Python moment function |

### Term structure

| Function | What it does |
|---|---|
| `nelson_siegel` | Nelson-Siegel yield-curve fit (Diebold-Li) |
| `svensson` | Svensson four-factor yield-curve fit |
| `dynamic_ns` | Dynamic Nelson-Siegel factors + one-step forecast |

---

## Results objects — the same dict, with a summary

Plain dicts are the contract, and they stay the contract. But when you are
reading output rather than piping it somewhere, you want a table. `tsecon.results`
is an **opt-in** layer of `dict` subclasses that carry the identical data and can
also render themselves:

```python
import json, numpy as np, tsecon
from tsecon.results import VARResults

y = np.array(json.load(open("fixtures/var.json"))["data_100dlog_gdp_cons_inv"])
fit = VARResults.fit(y, lags=2, names=["gdp", "cons", "inv"])
print(fit.summary())
```

```
====================================================================
VAR(2) — 3 equations, trend='c' — stable
====================================================================
llf -800.531    aic -0.2983    bic 0.0480    hqic -0.1582
reciprocal roots — min 1.6275    max 4.2538     (stable iff min > 1)
--------------------------------------------------------------------
coefficients — rows = regressors, cols = equations
regressor              gdp          cons           inv
--------------------------------------------------------------------
const             +0.15270      +0.54596      -2.39025
L1.gdp            -0.27943      -0.10047      -1.97097
L1.cons           +0.67502      +0.26864      +4.41416
L1.inv            +0.03322      +0.02574      +0.22548
L2.gdp            +0.00822      -0.12317      +0.38079
L2.cons           +0.29046      +0.23250      +0.80028
L2.inv            -0.00732      +0.02350      -0.12408
====================================================================
```

The point worth internalising: **it is still a dict.** Adopting this layer breaks
nothing, because it only *adds* methods to the object you already had.

```python
print(fit["aic"])                              # -0.2983183237427347
print(isinstance(fit, dict))                   # True
print(set(fit) == set(tsecon.var_fit(y, 2)))   # True — identical keys
```

`tsecon.var_fit` is untouched: it is still the compiled builtin returning a plain
dict, and `tsecon.results` is a namespace you reach into deliberately. `fit.irf(
horizon=10)` returns an `IRFArray` (a `list` subclass) whose `.response(1, 0)`
reproduces the raw `var_irf` numbers from the top of this page exactly. Plot
methods lazy-import matplotlib — install it with `pip install 'tsecon[plots]'`,
and until you call one, nothing imports it.

Every wrapper — `VARResults`, `LPResults`, `GARCHResults`, `ARIMAResults`,
`DSGEResults`, and the rest — is catalogued in
[reference/results.md](reference/results.md).

---

## Datasets — real data, no API key

Worked examples need real series, and vendoring macro data into a repository ages
badly. `tsecon.datasets` instead downloads **on first use** and caches: nothing is
fetched at import time, and the second call reads from disk.

```python
from tsecon import datasets as ds
import numpy as np

gs10 = ds.fred_series("GS10")                     # 10-year Treasury, monthly
print(gs10["nobs"], gs10["dates"][0], gs10["values"][-1])
# 879 1953-04-01 4.47
```

**No API key is required** — this uses FRED's public keyless CSV endpoint. (If
`FRED_API_KEY` happens to be set it is passed along, which is harmless.) The
printed numbers above are a real run on 2026-07-18; `nobs` and the last value grow
as FRED publishes.

`fred_md()` pulls the McCracken-Ng FRED-MD panel — the standard monthly macro
dataset for factor models and nowcasting — together with its **transform codes**,
the per-series integer recipe (1 level, 2 first difference, 5 first difference of
logs, …) for making each column stationary:

```python
md = ds.fred_md()
print(np.asarray(md["data"]).shape, md["dates"][0], md["dates"][-1])
# (801, 126) 1959-01-01 2025-09-01
print(md["names"][:3], md["transform_codes"][:3])
# ['RPI', 'W875RX1', 'DPCERA3M086SBEA'] [5 5 5]

stationary = ds.apply_fred_md_transforms(md["data"], md["transform_codes"])
```

Each loader records the source URL and a SHA-256 of the bytes it parsed, so a
dataset can be pinned and audited. `ds.cache_dir()` reports where things land
(`~/.cache/tsecon` by default; override with `TSECON_DATA_DIR`), `refresh=True`
re-downloads, and `local_path=...` parses a file you already have — which is also
how you work fully offline. See
[reference/datasets.md](reference/datasets.md).

---

## Where to go next

- **Not sure which model your problem calls for?** Start at the
  [which-model-when guide](which-model-when.md) — symptom-driven entry points
  ("my series is persistent and I need an impulse response"; "I have quarterly
  GDP and monthly indicators") that route you to the right estimator.
- **Want to learn the ideas, not just the calls?** The
  [tsecon Guide to Time Series Econometrics](guide/README.md) is a free,
  full-length course — from your first autocorrelation plot to research-grade
  structural identification — with runnable code in every chapter.
- **Want to see each method worked end to end?** The
  [gallery](examples/README.md) shows every function with a use case, code on
  real data, and the figure it produces.
</content>
</invoke>
