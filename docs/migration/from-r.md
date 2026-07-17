# Migrating from R

> Part of [The tsecon Guide to Time Series Econometrics](../guide/README.md). An
> adoption guide for R users: it maps the packages you already load — `vars`,
> `svars`, `lpirfs`, `BVAR`, `urca`, `rugarch`, `midasr`, `plm`, and friends — to
> tsecon functions, and says plainly where tsecon has no equivalent yet. Every
> Python block runs against the current library.

R's time-series econometrics is spread across dozens of specialized packages,
each with its own conventions, and much of it is unmaintained. tsecon's pitch to
an R user is consolidation: one installed library, one calling grammar, one HAC
implementation shared across every estimator, and a compiled core that turns the
overnight Monte Carlo into a coffee-break one. This page is the translation
layer.

## What changes when you cross over

Four habits to reset:

1. **No formulas, no `data.frame`, no model objects.** R leans on the formula
   interface (`y ~ x1 + x2`) and returns rich S3/S4 objects you inspect with
   `summary()`, `coef()`, `irf()`. tsecon takes NumPy arrays and returns plain
   `dict`s and arrays. You index `fit["params"]`, there is no `summary()` method,
   and column *order* in a matrix carries the meaning R attaches to variable
   names.

2. **Matrices are `T x k` (time down the rows).** A VAR, a cointegration test, a
   connectedness table — all take a `T x k` array with observations in rows and
   variables in columns, matching how `vars` and `urca` expect their input. Panel
   estimators instead take a *list* of per-unit arrays (see below), which maps
   cleanly onto R's split-by-`id` idiom.

3. **`se_type=` replaces the vcov zoo.** Instead of `sandwich::vcovHC`,
   `NeweyWest`, `vcovSCC`, and per-package robust options, every regression
   estimator takes a uniform `se_type=` argument (`"nonrobust"`, `"hc0"`–`"hc3"`,
   `"hac"`; panels add `"cluster"` and `"driscoll_kraay"`). One implementation,
   stamped into the result.

4. **Reproducible RNG lives in the arguments.** Where R uses a global
   `set.seed()`, tsecon's stochastic routines take an explicit `seed=` and run a
   parallel Philox stream, so results are bit-reproducible at any thread count.

## The mapping tables

"Roadmap" marks a capability tsecon does not ship today. Everything else is
callable now.

### VARs and structural analysis — `vars`, `svars`

| R | tsecon | Notes |
|---|---|---|
| `vars::VAR(data, p, type="const")` | `var_fit(data, lags=p, trend="c")` | `data` is `T x k`. `type="none"/"trend"` → `trend="n"/"t"`. |
| `vars::irf(v, n.ahead=h, ortho=TRUE)` | `var_irf(data, lags, horizon=h, orth=True)` | Nested list `[h][response][shock]`. |
| `vars::irf(..., cumulative=TRUE)` | `var_irf(..., cumulative=True)` | Running sums. |
| `vars::fevd(v, n.ahead=h)` | `var_fevd(data, lags, horizon=h)` | `[variable][horizon][shock]`; sums to 1 across shocks. |
| `vars::causality(v, cause=)` | `var_granger(data, caused, causing, lags)` | F-test. Pass integer column indices. |
| `predict(v, n.ahead=h)` | `var_forecast(data, lags, steps=h, alpha=0.05)` | `{"point", "lower", "upper"}`. |
| `svars::id.chol(v)` | `var_irf(..., orth=True)` | Recursive/Cholesky = column order of `data`. |
| `VARsignR`, `svars` sign restrictions | `sign_restricted_svar(data, restrictions, ...)` | Sign-restricted Bayesian SVAR + identified-set bands. |
| `svars::id.dc`, `id.ngml` (independence/heteroskedasticity ID) | — | Statistical identification: **roadmap**. |
| `svars` long-run / Blanchard-Quah | — | Long-run restrictions: **roadmap**. |

### Local projections — `lpirfs`

`lpirfs` is the standard R local-projection package; tsecon covers its main
entry points.

| R | tsecon | Notes |
|---|---|---|
| `lpirfs::lp_lin(..., shock_type)` | `lp(y, shock, horizons, se="lag_augmented")` | Lag-augmented inference by default (Montiel Olea–Plagborg-Møller 2021). |
| `lpirfs::lp_lin_iv(...)` | `lp_iv(y, impulse, instrument, horizons)` | Reports a first-stage F. |
| `lpirfs::lp_nl(...)` (state-dependent) | `lp_state(y, shock, state_indicator, ...)` | Ramey-Zubairy (2018) per-regime IRFs. |
| `lpirfs::lp_lin_panel(...)` | `panel_lp(outcome, shock, ...)` | Panel LP with fixed effects; Driscoll-Kraay SEs. |

### Bayesian VARs — `BVAR`, `bvartools`

| R | tsecon | Notes |
|---|---|---|
| `BVAR::bvar(data, lags, priors=)` | `bvar_fit(data, lags, lambda0, lambda1, lambda3, delta)` | Minnesota-NIW conjugate posterior + log marginal likelihood. |
| `irf(bv, horizon=h)` posterior draws | `bvar_irf_draws(data, lags, horizon, n_draws, seed)` | 4-D list `[draw][h][variable][shock]`; take quantiles for bands. |
| `coda`/`rstan` R-hat, ESS | `mcmc_diagnostics(chains)` | Rank-normalized split R-hat and bulk/tail ESS (ArviZ-exact). |
| Hierarchical / SSVS / hyperparameter priors | — | Only the conjugate Minnesota-NIW prior ships; other priors: **roadmap**. |

### Cointegration and unit roots — `urca`, `tseries`

| R | tsecon | Notes |
|---|---|---|
| `urca::ur.df(y, type="drift")` | `adf(y, regression="c")` | Or `tseries::adf.test`. Dict return with MacKinnon p-value. |
| `urca::ur.kpss(y)` | `kpss(y, regression="c")` | Null is stationarity. |
| `urca::ca.jo(data, type="trace", K=)` | `johansen(data, k_ar_diff=K-1)` | Trace + max-eig stats and selected ranks. |
| `urca::cajorls`, `vars::vec2var` | `vecm(data, k_ar_diff, coint_rank)` | ML VECM: `alpha`, `beta`, `gamma`, `sigma_u`, `llf`. |
| `tseries::Box.test(y, type="Ljung-Box")` | `ljung_box(y, nlags)` | Box-Pierce also returned. |
| `tseries::jarque.bera.test(y)` | `jarque_bera(y)` | |
| `FinTS::ArchTest(y)` | `arch_lm(y, nlags)` | Engle's ARCH-LM. |
| `urca::ca.po` (Phillips-Ouliaris), `ur.pp` | — | Phillips-Perron / Phillips-Ouliaris: **roadmap**. |
| `urca::ca.jo` (Engle-Granger via `po.test`) | — | Engle-Granger two-step: **roadmap**; use `johansen`. |

### Univariate models and volatility — `forecast`, `rugarch`, `rmgarch`, `MSwM`

| R | tsecon | Notes |
|---|---|---|
| `forecast::Arima(y, order=c(p,d,q))` | `arima_fit(y, p, d, q, constant=True)` | Exact-MLE. |
| `forecast::auto.arima(y)` | — | Automatic order selection: **roadmap** (compare AIC/BIC from `arima_fit` by hand). |
| `forecast::thetaf(y, h)` | `theta_forecast(y, steps=h, period=)` | The Theta method. |
| `forecast::accuracy(f, y)` | `accuracy(actual, forecast, insample=, period=)` | ME/RMSE/MAE/MAPE/sMAPE/MASE/RMSSE. |
| `forecast::dm.test(e1, e2)` | `dm_test(e1, e2, h=1, loss="squared")` | HLN small-sample correction. |
| `rugarch::ugarchfit(spec, y)` | `garch_fit(y, vol="garch"/"egarch", p, o, q, dist=)` | GJR via `o=`. `se_robust` = Bollerslev-Wooldridge. |
| `rmgarch::dccfit(...)` | `dcc_garch(returns)` | Engle (2002) DCC; `returns` is `T x k`. |
| `ccgarch`, constant-correlation | `ccc_garch(returns)` | Bollerslev (1990) CCC. |
| `GAS::UniGASFit(...)` | `gas_volatility(y, density="gaussian"/"student_t")` | Creal-Koopman-Lucas score-driven volatility. |
| `MSwM::msmFit`, `MSGARCH` | `markov_switching_ar(y, k_regimes, order, switching_variance=)` | Hamilton EM; regimes, transition, durations. |

### Mixed frequency and nowcasting — `midasr`, `nowcasting`

| R | tsecon | Notes |
|---|---|---|
| `midasr::midas_r(y ~ ..., nbeta/nealmon)` | `weighted_midas(y, hf_lags, scheme="beta"/"exp_almon")` | NLS-estimated restricted weights; `hf_lags` is `nobs x K`. |
| `midasr::midas_u` / U-MIDAS | `umidas(y, hf_lags, se_type="hac")` | Unrestricted mixed-frequency regression. |
| `midasr::nbeta`, `nealmon` weight builders | `midas_weights(scheme, theta1, theta2, k)` | The weight vector alone. |
| `nowcasting::nowcast(...)` (DFM) | `dfm_nowcast(data, n_factors, factor_order)` | Two-step DGR (2011); handles the ragged edge (NaNs). |
| `nowcasting` news decomposition | `dfm_news(old_vintage, new_vintage, target_series, ...)` | Banbura-Modugno (2014) per-datapoint news contributions. |

### Panel time series — `plm`, `xtmg`-style estimators

R's panel workflow (`plm`) uses a long `pdata.frame`. tsecon instead takes a
**list of per-unit arrays**: `ys` is a list of response vectors, `xs` a list of
`T_i x k` regressor matrices — which is what you get from `split(df, df$id)`. The
alternative `panel_fe`/`panel_lp` layout is a dense `N x T` outcome with a
`k x N x T` regressor tensor.

| R | tsecon | Notes |
|---|---|---|
| `plm::pmg(..., model="mg")` | `panel_mean_group(ys, xs, method="mg")` | Pesaran-Smith (1995) mean group: per-unit OLS, then average. |
| `plm::pmg(..., model="pmg")` | `panel_pmg(ys, xs)` | Pooled Mean Group ARDL(1,1) (Pesaran-Shin-Smith 1999); pools the long-run coef by ML. |
| `xtmg`/CCEMG (Pesaran 2006) | `panel_mean_group(ys, xs, method="cce")` | Common-correlated-effects mean group. |
| `plm::plm(..., model="within")` | `panel_fe(outcome, regressors, se_type=)` | Fixed effects; `outcome` is `N x T`, `regressors` is `k x N x T`. |
| `plm` + `vcovSCC` (Driscoll-Kraay) | `panel_fe(..., se_type="driscoll_kraay")` | Same SE, one argument. |
| Panel VAR (`panelvar`) | `mean_group_var(entities, lags, horizon)` | Pesaran-Smith mean-group panel VAR over per-entity `T_i x k` matrices. |
| Panel unit-root (IPS, LLC via `plm::purtest`) | — | **Roadmap.** |

### Realized volatility, connectedness, term structure

| R | tsecon | Notes |
|---|---|---|
| `highfrequency::rCov`, `rBPCov` | `realized_measures(returns)` | RV, bipower variation, jump component (BNS 2004). |
| `HARModel::HARestimate` | `har_rv(rv, variant="level"/"log"/"sqrt")` | HAR-RV (Corsi 2009) with HAC SEs. |
| `highfrequency::medRQ`, `rQuar` | `realized_quarticity`, `tripower_quarticity` | Integrated-quarticity estimators. |
| `highfrequency::BNSjumptest` | `bns_jump_test(returns)` | BNS ratio jump test. |
| `TTR`/`highfrequency` range vol | `realized_range(high, low, method="parkinson"/"garman_klass")` | From OHLC bars. |
| `frequencyConnectedness`, `ConnectednessApproach` | `connectedness(data, lags, horizon)` | Diebold-Yilmaz spillover table (GFEVD). |
| `YieldCurve::Nelson.Siegel` | `nelson_siegel(maturities, yields, optimal_lambda=)` | Level/slope/curvature + fit. |
| `YieldCurve::Svensson` | `svensson(maturities, yields, lambda1, lambda2)` | Four-factor, nests Nelson-Siegel. |
| `YieldCurve` dynamic NS (Diebold-Li) | `dynamic_ns(panel, maturities, decay)` | Factor series + one-step forecast. |

### GMM and penalized regression — `gmm`, `glmnet`

| R | tsecon | Notes |
|---|---|---|
| `gmm::gmm(g, x, ...)` linear IV | `iv_gmm(x, z, y, method="2step"/"iterated", weight="robust"/"hac")` | Hansen (1982); over-identified fits report the Hansen J. |
| `gmm::gmm(...)` nonlinear moments | `gmm_nonlinear(moments_fn, initial, weight=)` | Custom moment function as a Python callback. |
| `AER::ivreg(y ~ x | z)` (2SLS) | `iv_gmm(x, z, y, method="2sls")` | The 2SLS special case. |
| `glmnet(x, y, alpha=1)` (lasso) | `lasso(x, y, alpha)` / `lasso_path(x, y)` | `lasso_path` returns the full path with AIC/BIC selection. |
| `glmnet(x, y, alpha=a)` (elastic net) | `elastic_net(x, y, alpha, l1_ratio)` | scikit-learn objective. |
| `glmnet(x, y, alpha=0)` (ridge) | `ridge(x, y, alpha)` | Closed form. |
| adaptive lasso (`glmnet` + weights) | `adaptive_lasso(x, y, alpha, gamma=)` | Zou (2006) oracle-property weights. |

### Filters and spectra — `mFilter`, `neverhpfilter`

| R | tsecon | Notes |
|---|---|---|
| `mFilter::hpfilter(y, freq)` | `hp_filter(y, lamb, one_sided=)` | |
| `mFilter::bkfilter(y)` | `bk_filter(y, low, high, k)` | |
| `mFilter::cffilter(y)` | `cf_filter(y, low, high, drift)` | |
| `neverhpfilter::yth_filter` | `hamilton_filter(y, h=8, p=4)` | Hamilton (2018) regression filter. |
| `spectrum(y)`, `stats::spec.pgram` | `periodogram(x)`, `welch(x)`, `coherence(x, y)` | Match SciPy's spectral estimators. |

## Worked translations

Five you can run. The R call is shown as a comment.

### `urca::ca.jo` → `johansen` + `vecm`

```python
import numpy as np, tsecon
rng = np.random.default_rng(7)
trend = np.cumsum(rng.standard_normal(200))           # one shared stochastic trend
data = np.column_stack([trend + rng.standard_normal(200) for _ in range(3)])

jo = tsecon.johansen(data, k_ar_diff=1)               # urca::ca.jo(data, type="trace")
print(np.round(jo["trace_stat"], 2), jo["rank_trace_5pct"])

vm = tsecon.vecm(data, k_ar_diff=1, coint_rank=1)     # cajorls / vec2var
print(np.round(np.asarray(vm["beta"]).ravel(), 3))    # the cointegrating vector
```

### `BVAR::bvar` → `bvar_fit` + `bvar_irf_draws`

```python
import numpy as np, tsecon
rng = np.random.default_rng(5)
data = rng.standard_normal((200, 3))

bv = tsecon.bvar_fit(data, lags=2, lambda1=0.2)       # BVAR::bvar(data, lags=2)
draws = tsecon.bvar_irf_draws(data, lags=2, horizon=12,
                              n_draws=500, seed=1)      # [draw, h, var, shock]
med = np.median(np.asarray(draws), axis=0)             # posterior-median IRF surface
print(round(bv["log_marginal_likelihood"], 2), med.shape)
```

### `rmgarch::dccfit` → `dcc_garch`

```python
import numpy as np, tsecon
def garch_sim(seed, T=1000):                           # a clustered return series
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T); h = np.empty(T); r = np.empty(T)
    h[0] = 0.5; r[0] = np.sqrt(h[0]) * e[0]
    for t in range(1, T):
        h[t] = 0.05 + 0.08 * r[t-1]**2 + 0.90 * h[t-1]
        r[t] = np.sqrt(h[t]) * e[t]
    return r

ret = np.column_stack([garch_sim(s) for s in (6, 7, 8)])
dcc = tsecon.dcc_garch(ret)                            # rmgarch::dccfit(spec, ret)
print(round(dcc["a"], 4), round(dcc["b"], 4))
```

### `midasr::midas_r` → `weighted_midas`

```python
import numpy as np, tsecon
rng = np.random.default_rng(8)
hf = rng.standard_normal((120, 3))                     # three high-frequency lags
y = hf @ np.array([0.6, 0.3, 0.1]) + rng.standard_normal(120)

wm = tsecon.weighted_midas(y, hf, scheme="beta")       # midasr::midas_r(y ~ ..., nbeta)
print(round(wm["slope"], 3), round(wm["rsquared"], 3))
```

### `plm::pmg` → `panel_pmg` (and mean group)

```python
import numpy as np, tsecon
rng = np.random.default_rng(9)
ys = [rng.standard_normal(40) for _ in range(15)]      # split(df, df$id) -> per-unit
xs = [rng.standard_normal((40, 2)) for _ in range(15)]

pmg = tsecon.panel_pmg(ys, xs)                         # plm::pmg(..., model="pmg")
mg = tsecon.panel_mean_group(ys, xs, method="mg")      # plm::pmg(..., model="mg")
print(np.round(pmg["theta"], 3), np.round(mg["coef"], 3))
```

## What R has that tsecon does not (yet)

R's long tail is deep; several widely used capabilities are **roadmap**:

- **`auto.arima`-style automatic order selection** (compare `arima_fit` ICs by hand).
- **Phillips-Perron / Phillips-Ouliaris** unit-root and **Engle-Granger** cointegration tests.
- **Statistical SVAR identification** (`svars::id.dc`/`id.ngml`) and **long-run / Blanchard-Quah** restrictions.
- **Non-conjugate BVAR priors** (SSVS, hierarchical, stochastic volatility).
- **Panel unit-root tests** (IPS, Levin-Lin-Chu) and the fuller `panelvar` toolkit.
- **STL / seasonal decomposition** and **TBATS/Prophet-style** seasonal forecasters.
- **Threshold models** (SETAR/STAR via `tsDyn`); tsecon offers `markov_switching_ar` and `lp_state` for nonlinearity.

Where tsecon repays the switch is speed and coherence: one library instead of a
dozen, a single shared HAC/bootstrap core, reproducible parallel RNG, and the
modern macro-structural methods (`lp`, `sign_restricted_svar`, `dfm_nowcast`,
`favar`, `connectedness`) delivered under one calling grammar.

See also the [statsmodels](from-statsmodels.md) and [Stata](from-stata.md)
guides, and the cross-package [Rosetta glossary](rosetta.md).
