# tsecon Quickstart — 60 seconds to your first impulse response

This is the on-ramp. In one page you will install **tsecon**, fit a vector
autoregression to a bundled dataset, and read an impulse response off it — the
single most common thing an empirical macroeconomist asks a time-series
library to do. Everything below runs today against the shipped API; the code
blocks are the same ones the test suite exercises.

> **Pre-1.0.** The name is settled — `tsecon` is what you install and what you
> import — but the API may still change before 1.0. See
> [ROADMAP.md](../ROADMAP.md).

---

## Install

```sh
pip install tsecon
```

tsecon is a compiled Rust extension with a thin Python API, shipped as a single
self-contained wheel whose only runtime dependency is NumPy — no Rust toolchain,
no system BLAS. Prebuilt wheels cover Linux (`x86_64`, `aarch64`), macOS on
Apple Silicon (`arm64`), and Windows (`x64`), for every Python ≥ 3.9. Plotting
is opt-in (`pip install 'tsecon[plots]'` pulls in matplotlib). Confirm the
install and see how much is on the shelf:

```python
import tsecon
print(tsecon.__version__)                                       # 0.8.0
print(sum(callable(getattr(tsecon, n)) for n in dir(tsecon)     # 162
          if not n.startswith("_")))
```

There is no prebuilt wheel for Intel macOS (`x86_64`); on that platform pip
builds from the source distribution, which needs a
[Rust toolchain](https://rustup.rs/). That same from-source path is how
contributors build from a checkout with [maturin](https://www.maturin.rs/):

```sh
maturin build --release -m bindings/python/Cargo.toml   # writes target/wheels/tsecon-0.8.0-*.whl
pip install target/wheels/tsecon-0.8.0-*.whl            # installs the `tsecon` package
```

---

## Hello, impulse response

The repository ships golden fixtures in [`fixtures/`](../fixtures/) — the same
data the library is validated against. One of them, `var.json`, holds 202
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
here, `tsecon.var_fevd` and `tsecon.var_forecast` take the same `(data, lags)`
arguments; `tsecon.var_granger` needs the hypothesis too — `var_granger(y,
caused, causing, lags=2)`, each of `caused`/`causing` a sequence of column
indices.

---

## The API at a glance

The 162 functions, grouped by the task they serve. Every one is a plain
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
| `phillips_perron` | Phillips-Perron semiparametric unit-root test (ADF alternative) |
| `phillips_ouliaris` | Phillips-Ouliaris residual cointegration test |
| `engle_granger` | Engle-Granger two-step cointegration test (matches statsmodels `coint`) |
| `dfgls` | DF-GLS: the GLS-detrended Elliott-Rothenberg-Stock unit-root test |
| `ng_perron` | Ng-Perron M unit-root tests (MZa, MZt, MSB, MPT) with MAIC lag choice |
| `zivot_andrews` | Unit-root test against one endogenous structural break |
| `ndiffs` | How many differences a series needs, with the evidence at every order |
| `nsdiffs` | How many *seasonal* differences a series needs (Hyndman-Khandakar) |
| `box_cox_lambda` | Variance-stabilizing Box-Cox lambda (MLE or Guerrero) |
| `check_stationarity` | The ADF + KPSS confirmatory workflow, with a recommendation |
| `check_series` | One-call diagnostic battery: runs the test families, suggests models with evidence |
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
| `auto_arima` | Automatic ARIMA order selection (Hyndman-Khandakar stepwise) |
| `ar_loglik` | Exact Gaussian AR(p) log-likelihood at fixed parameters |
| `local_level_smooth` | Local-level Kalman filter + smoother (handles missing data) |
| `dcs_local_level` | Score-driven robust local level (Harvey-Luati) |
| `hp_filter` | Hodrick-Prescott trend/cycle decomposition |
| `bk_filter` | Baxter-King band-pass filter |
| `cf_filter` | Christiano-Fitzgerald band-pass filter |
| `hamilton_filter` | Hamilton's regression-based HP alternative |
| `bn_filter` | Kamber-Morley-Wong BN filter: output gap at a pinned signal-to-noise ratio |
| `bn_decomposition` | Classic Beveridge-Nelson trend/cycle from an ARIMA(p,1,q) |
| `stl` | STL seasonal-trend decomposition by loess (Cleveland et al. 1990) |
| `mstl` | MSTL — STL iterated over several seasonal periods |
| `seasonal_strength` | Wang-Smith-Hyndman seasonal and trend strength from an STL fit |
| `markov_switching_ar` | Regime-switching AR fitted by EM (Hamilton 1989) |

### Volatility

| Function | What it does |
|---|---|
| `garch_fit` | GARCH / GJR / EGARCH by QMLE with robust standard errors |
| `gas_volatility` | GAS(1,1) score-driven volatility |
| `ccc_garch` | Constant-conditional-correlation multivariate GARCH |
| `dcc_garch` | Dynamic-conditional-correlation multivariate GARCH |
| `dcc_test` | Engle-Sheppard test of constant conditional correlation (CCC vs DCC) |
| `realized_measures` | Realized variance, bipower variation, and jump component |
| `har_rv` | Corsi HAR-RV regression with HAC standard errors |
| `realized_quarticity` | Realized quarticity (the sampling variance of RV) |
| `tripower_quarticity` | Jump-robust integrated quarticity |
| `bns_jump_test` | Barndorff-Nielsen-Shephard ratio jump test |
| `realized_range` | Parkinson / Garman-Klass range variance from OHLC bars |

### Tail risk, dependence, and spreads

| Function | What it does |
|---|---|
| `gpd_fit` | Peaks-over-threshold GPD tail fit with McNeil-Frey VaR / ES |
| `gev_fit` | GEV block-maxima fit with return levels |
| `var_backtest` | VaR backtests: Kupiec, Christoffersen, and the Engle-Manganelli DQ test |
| `pseudo_obs` | Pseudo-observations: the average-rank probability-scale transform |
| `copula_fit` | Fit a bivariate copula to probability-scale pseudo-observations |
| `copula_select` | Rank copula families by AIC/BIC, with a teaching verdict |
| `ou_fit` | Ornstein-Uhlenbeck mean-reversion fit for a spread (exact-discretization MLE) |
| `spread_zscore` | Z-score of a spread against its stationary OU law |

### Multivariate and structural

| Function | What it does |
|---|---|
| `var_fit` | Fit a VAR(p) by OLS: params, covariance, ICs, stability |
| `var_irf` | Orthogonalized or raw impulse responses |
| `var_irf_bands` | Frequentist IRF confidence bands (delta-method or bootstrap) |
| `var_fevd` | Forecast-error variance decomposition |
| `var_forecast` | Iterated VAR forecasts with intervals |
| `var_granger` | Granger-causality F test |
| `sign_restricted_svar` | Sign-restricted Bayesian SVAR identified-set bands |
| `zero_sign_svar` | Zero + sign restricted Bayesian SVAR (RWZ 2010 / ARW 2018) |
| `long_run_svar` | Blanchard-Quah long-run SVAR (supply/demand decomposition) |
| `max_share_svar` | Max-share / maximum-FEV shock (main business cycle, news) |
| `proxy_svar` | Proxy SVAR / external-instrument identification (SVAR-IV) |
| `proxy_svar_bands` | Proxy-SVAR IRF bands (Jentsch-Lunsford moving-block bootstrap) |
| `proxy_ar_sets` | Weak-instrument-robust (Anderson-Rubin) confidence *sets* for a proxy-SVAR IRF |
| `proxy_first_stage` | Proxy strength: the Montiel Olea-Pflueger effective F |
| `hetero_svar` | SVAR identification through heteroskedasticity (Rigobon) |
| `nongaussian_svar` | Non-Gaussian / independent-component SVAR identification (ICA; fails if Gaussian) |
| `structural_fevd` | FEVD for an arbitrary structural impact matrix A0 (any scheme) |
| `historical_decomposition` | Per-(time, variable, shock) historical decomposition (exact adding-up) |
| `fry_pagan_svar` | Fry-Pagan median-target: the coherent draw closest to the median band |
| `robust_svar_bounds` | Giacomini-Kitagawa prior-robust identified-set bounds |
| `narrative_svar` | Narrative sign-restricted SVAR (Antolín-Díaz-Rubio-Ramírez) |
| `favar` | Factor-augmented VAR policy-shock IRFs (Bernanke-Boivin-Eliasz) |
| `johansen` | Johansen cointegration rank test |
| `vecm` | VECM maximum-likelihood estimation |
| `connectedness` | Diebold-Yilmaz spillover / connectedness measures |
| `factor_model` | PCA factor model with Bai-Ng factor selection |

### Threshold and smooth-transition models

| Function | What it does |
|---|---|
| `setar` | Two-regime SETAR(p) by concentrated least squares (Tong-Lim; Hansen 1997) |
| `setar_test` | Hansen sup-F linearity test against a SETAR, bootstrap p-value |
| `threshold_var` | Two-regime threshold VAR — the multivariate SETAR |
| `threshold_var_test` | Sup-Wald linearity test of a linear VAR against the threshold VAR |
| `threshold_vecm` | Hansen-Seo two-regime threshold VECM (threshold cointegration) |
| `hansen_seo_test` | Hansen-Seo sup-LM test: linear versus threshold cointegration |
| `star` | Smooth-transition AR — LSTAR / ESTAR by concentrated NLS (Teräsvirta) |
| `star_eval` | Score a STAR fit at fixed `(gamma, c)` — for a published parameterization |
| `star_test` | Teräsvirta LM3 linearity test plus the H03/H02/H01 modeling cycle |

### Local projections

| Function | What it does |
|---|---|
| `lp` | Local-projection impulse responses (lag-augmented or HAC SEs) |
| `lp_iv` | Instrumented local projections with a first-stage F diagnostic |
| `lp_multiplier` | Integral multiplier (Ramey-Zubairy): cumulated outcome on cumulated impulse, instrumented |
| `lp_state` | State-dependent (interacted) local projections (Ramey-Zubairy) |
| `smooth_lp` | Smooth local projections: B-spline-penalized IRFs (Barnichon-Brownlees) |
| `lp_did` | LP-DiD event-study difference-in-differences (Dube-Girardi-Jordà-Taylor) |

### Functional shocks (FVAR / FLP)

| Function | What it does |
|---|---|
| `functional_pca` | Functional PCA of curve-valued shocks into interpretable scores |
| `flp` | Functional local projections: the response to each shock-curve score |
| `flp_scenario` | Response to a user-drawn shock *curve* (scenario analysis via FLP) |
| `fvar_scenario` | The same scenario answer through a VAR in the curve scores (FVAR) |

### Forecasting and evaluation

| Function | What it does |
|---|---|
| `theta_forecast` | The Theta method (Assimakopoulos-Nikolopoulos) |
| `accuracy` | Forecast accuracy measures (RMSE, MAE, MAPE, MASE, RMSSE…) |
| `backtest` | Rolling / expanding pseudo-out-of-sample backtest |
| `conformal_forecast` | Distribution-free conformal forecast intervals (split / EnbPI / ACI) |
| `conformal_backtest` | Online out-of-sample coverage evaluation of those intervals |
| `dm_test` | Diebold-Mariano equal-accuracy test (HLN-corrected) |
| `cw_test` | Clark-West test for nested models |
| `gw_test` | Giacomini-White test of equal predictive ability |

### Bayesian

| Function | What it does |
|---|---|
| `bvar_fit` | Minnesota-NIW conjugate BVAR posterior + log marginal likelihood |
| `bvar_hierarchical` | Empirical-Bayes BVAR: tightness chosen by marginal likelihood (GLP) |
| `bvar_ssvs` | Spike-and-slab SSVS-BVAR: posterior coefficient inclusion probabilities |
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
| `panel_unit_root` | First-generation panel unit-root tests (LLC / IPS / Fisher) |

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
| `afns_adjustment` | Arbitrage-free (AFNS) yield adjustment (Christensen-Diebold-Rudebusch) |
| `acm_term_premium` | Adrian-Crump-Moench regression-based term premium |

### Specification & stability tests

| Function | What it does |
|---|---|
| `heteroskedasticity_test` | White or Koenker-Breusch-Pagan heteroskedasticity test |
| `reset_test` | Ramsey RESET functional-form test |
| `chow_test` | Chow break test at a known split date |
| `cusum_test` | Brown-Durbin-Evans CUSUM parameter-stability test |
| `sup_f_test` | Andrews sup-F test for a break at an unknown date (Hansen p-values) |
| `bai_perron` | Multiple structural breaks: dates, confidence intervals, IC selection |

### Predictive regressions & recession probability

| Function | What it does |
|---|---|
| `predictive_regression` | OLS + Stambaugh correction + IVX inference in one call |
| `ivx_test` | Joint IVX predictability test for several persistent predictors |
| `recession_probit` | Static or Kauppi-Saikkonen dynamic recession probit/logit |

### Quantile regression & growth-at-risk

| Function | What it does |
|---|---|
| `quantile_regression` | Koenker-Bassett quantile regression with sandwich SEs |
| `quantile_lp` | Quantile local projections: the IRF at chosen quantiles |
| `growth_at_risk` | Conditional quantiles of future growth (Adrian-Boyarchenko-Giannone) |

### Survey expectations & long memory

| Function | What it does |
|---|---|
| `cg_regression` | Coibion-Gorodnichenko information-rigidity regression (HAC SEs) |
| `forecast_efficiency` | Mincer-Zarnowitz unbiasedness/efficiency test |
| `forecast_disagreement` | Cross-forecaster dispersion, quartiles, IQR per period |
| `frac_diff` | Fractional differencing `(1 − L)^d x` |
| `frac_integrate` | Fractional integration (the inverse of `frac_diff`) |
| `long_memory_d` | Estimate `d` (GPH log-periodogram or Robinson local Whittle) |

### Structural models

| Function | What it does |
|---|---|
| `dsge_solve` | Blanchard-Kahn solution of a linear rational-expectations model |

### Presentation

| Function | What it does |
|---|---|
| `summarize` | Wrap any tsecon output in a renderable results object (see below) |

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
print(fit["aic"])                              # -0.29831832374273115
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

## Bring your own arrays — no data loaders, no network

tsecon deliberately ships **no data-fetching loaders**. `import tsecon` makes no
network request, and the only runtime dependency is NumPy. Every function takes
plain arrays, so bring data in with whatever you already use — `pandas`,
`pandas.read_csv`, `pandas-datareader` for FRED, or a CSV — and hand tsecon the
columns:

```python
import pandas as pd, tsecon

df = pd.read_csv("my_macro_panel.csv", parse_dates=["date"]).set_index("date")
fit = tsecon.var_fit(df[["gdp", "cpi", "ffr"]], lags=2)   # a DataFrame goes straight in
```

Keeping data acquisition out of the library is a deliberate boundary: a loader
that hardcodes external URLs becomes a maintenance liability the moment a
provider reorganizes its site (FRED, for one, has already moved the canonical
FRED-MD file). Fetching is a solved problem with well-maintained specialist
tools; tsecon does the econometrics. The [replication gallery](examples/README.md)
shows real-data workflows end to end, running on small public datasets committed
to the repository.

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
