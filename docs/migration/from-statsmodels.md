# Migrating from statsmodels

> Part of [The tsecon Guide to Time Series Econometrics](../guide/README.md). This
> is an adoption guide, not a tutorial: it maps calls you already know in
> `statsmodels.tsa` to their tsecon equivalents and is honest about what tsecon
> does not yet cover. Every Python block runs against the current library.

If you write empirical time series in Python today, you almost certainly write
`statsmodels`. tsecon is not a drop-in replacement — it deliberately makes
different choices — but most of the `statsmodels.tsa` surface has a direct
counterpart here, and the parts that do not (local projections, sign-restricted
SVARs, BVARs, nowcasting) are exactly the reasons to reach for tsecon. This page
gets you across.

## Five differences to internalize first

Before the table, five conventions that will trip you up if you carry
`statsmodels` habits over unexamined:

1. **Functions return dicts and arrays, not results objects.** `statsmodels`
   gives you a fitted `Results` object you interrogate with attributes and
   methods (`res.params`, `res.summary()`, `res.irf(10)`). tsecon gives you a
   plain `dict` (or a NumPy array, or a nested list). You read `fit["params"]`,
   not `fit.params`. There is no `.summary()` yet — you format output yourself.
   This keeps the boundary between the Rust core and Python thin and the return
   values trivially serializable.

2. **You pass NumPy arrays, not DataFrames.** tsecon has no pandas dependency and
   no notion of a date index. A VAR takes a `T x k` float array; a panel takes
   raw arrays in a documented shape. Column *order* is therefore load-bearing —
   it is your Cholesky ordering, your variable labels, everything. Keep your own
   column-name list alongside the array.

3. **No automatic intercept.** `tsecon.ols(y, x, ...)` regresses `y` on the
   columns of `x` *exactly as given*. Like `statsmodels`' `OLS` (which also
   requires `add_constant`), it will not invent a constant for you — but there is
   no `add_constant` helper, so prepend a column of ones yourself. The VAR/BVAR
   and filter routines handle their own trend terms via a `trend=` argument.

4. **One `se_type=` grammar for robust inference.** Where `statsmodels` spreads
   robust covariance across `cov_type=`/`cov_kwds=` and per-model flags, tsecon
   exposes a uniform `se_type=` on every regression estimator:
   `"nonrobust"`, `"hc0"`–`"hc3"`, and `"hac"` (Newey-West), with `maxlags=` for
   the HAC bandwidth. Panels add `"cluster"` and `"driscoll_kraay"`. Local
   projections use `se=` with `"lag_augmented"` (the default, per Montiel
   Olea–Plagborg-Møller 2021) or `"hac"`. The method chosen is stamped back into
   the result (`res["se_type"]`).

5. **Impulse responses are nested lists, indexed `[horizon][response][shock]`.**
   `tsecon.var_irf(...)[h][i][j]` is the response of variable `i` at horizon `h`
   to a shock in variable `j`. Horizon runs `0..=horizon`, so the outer length is
   `horizon + 1`. The variance decomposition `var_fevd` uses a *different* axis
   order — `[variable][horizon][shock]` — and its innermost slice sums to one
   across shocks. Convert to a NumPy array and index deliberately.

## The mapping table

"Roadmap" marks a capability tsecon does not ship today; see
[the roadmap](../../ROADMAP.md) and the [module specs](../roadmap/). Everything
else is a shipped function you can call now.

### Diagnostics and unit roots

| statsmodels | tsecon | Notes |
|---|---|---|
| `adfuller(y)` | `adf(y, regression="c", autolag=..., maxlag=None)` | Returns a dict (`statistic`, `p_value`, `used_lag`, `nobs`, `crit`), not a tuple. MacKinnon p-values. |
| `kpss(y)` | `kpss(y, regression="c", nlags=None)` | Null is stationarity. Dict return. |
| — | `check_stationarity(y, alpha=0.05)` | The ADF+KPSS confirmatory quadrant with a `recommendation`. No `statsmodels` analogue. |
| *(run each test by hand)* | `check_series(data, seasonal_period=None, lags=None, alpha=0.05)` | One-call diagnostic battery: descriptives, outliers, the ADF+KPSS quadrant, Ljung-Box/ACF/PACF, ARCH-LM, Jarque-Bera, a sup-F/Bai-Perron mean-shift scan, GPH long memory, and seasonality — ending in an ordered `recommendations` list routing to concrete tsecon calls. 2D `(n, k)` input adds per-series integration, Johansen, and VAR lag selection. A tsecon convenience, not a `statsmodels` port. |
| `acf(y, nlags=n)` | `acf(y, nlags=20, adjusted=False)` | Returns `{"acf", "bartlett_se"}`. |
| `pacf(y, nlags=n, method="yw")` | `pacf(y, nlags=20, method="yw")` | Returns a bare array. `method` is `"yw"` or `"ols"`. |
| `acorr_ljungbox(y, lags=n)` | `ljung_box(y, nlags=10)` | Returns Ljung-Box *and* Box-Pierce for lags `1..=nlags`. |
| `jarque_bera(x)` | `jarque_bera(x)` | Dict with `statistic`, `p_value`, `skewness`, `kurtosis`, `n`. |
| `het_arch(resid)` | `arch_lm(resid, nlags=...)` | Engle's ARCH-LM test. |
| `q_stat(...)` | use `ljung_box` | Box-Pierce is the `bp_stat`/`bp_pvalue` keys. |
| `adfuller` (PP variant), `PhillipsPerron` | — | Phillips-Perron: **roadmap**. |
| `range_unit_root_test`, `zivot_andrews` | — | **Roadmap.** |

### Regression with dependent-data standard errors

| statsmodels | tsecon | Notes |
|---|---|---|
| `OLS(y, add_constant(X)).fit()` | `ols(y, Xc, se_type="nonrobust")` | Prepend your own constant column `Xc`. |
| `... .fit(cov_type="HAC", cov_kwds={"maxlags": L})` | `ols(y, Xc, se_type="hac", maxlags=L)` | Newey-West. `use_correction=` toggles the small-sample factor. |
| `... .fit(cov_type="HC0".."HC3")` | `ols(y, Xc, se_type="hc0".."hc3")` | Same White family. |
| `cov_hac`, `cov_nw_panel` helpers | `long_run_variance(x, kernel=, bandwidth=)` | The kernel LRV as a standalone number. |

### Specification and structural-break tests

`statsmodels` scatters these across `stats.diagnostic`; tsecon gathers them behind
a uniform `(y, x)` signature — pass the regression's design matrix (with its
constant column), not pre-computed residuals.

| statsmodels | tsecon | Notes |
|---|---|---|
| `het_white(resid, exog)` | `heteroskedasticity_test(y, x, test="white")` | Pass the regression `(y, x-with-constant)`, not residuals. Dict `statistic`/`pvalue` (LM) plus `fstat`/`f_pvalue`. |
| `het_breuschpagan(resid, exog)` | `heteroskedasticity_test(y, x, test="breusch_pagan")` | Same signature; Breusch-Pagan variant. |
| `linear_reset(res, power=[2,3])` | `reset_test(y, x, max_power=3)` | Ramsey RESET functional-form F-test; fitted powers `2..=max_power` of ŷ. |
| `breaks_cusumolsresid(resid)` | `cusum_test(y, x)` | Brown-Durbin-Evans recursive-residual CUSUM; returns the `path` and 5% `bound_lower`/`bound_upper`. (`statsmodels`' test is the OLS-residual variant.) |
| *(no direct Chow test)* | `chow_test(y, x, split=k)` | Structural-break F-test at a *known* 0-indexed split `k`. |
| *(only `breaks_cusumolsresid`)* | `bai_perron(y, x, max_breaks=m, trim=0.15)` | Bai-Perron multiple breaks (global DP partitions + sequential supF selection). **No `statsmodels` analogue.** |
| — | `sup_f_test(y, x, trim=0.15)` | Andrews sup-F (Quandt) *unknown*-break test, Hansen (1997) p-value. **No `statsmodels` analogue.** |

### Univariate models and volatility

| statsmodels / arch | tsecon | Notes |
|---|---|---|
| `ARIMA(y, order=(p,d,q)).fit()` | `arima_fit(y, p, d, q, constant=True, forecast_steps=0)` | Exact-MLE. Forecast bands via `conf_alpha=`. |
| `SARIMAX(... seasonal_order=...)` | — | Seasonal terms: **roadmap**. `arima_fit` is non-seasonal ARIMA. |
| `ExponentialSmoothing`, `ETSModel` | `theta_forecast(y, steps, period)` | Only the Theta method ships; general ETS is **roadmap**. |
| `arch_model(r, vol="Garch", p, q).fit()` | `garch_fit(y, vol="garch", p=1, q=1, ...)` | Also `vol="egarch"` and GJR via `o=`. Robust SEs in `se_robust`. |
| `arch_model(..., dist="StudentsT")` | `garch_fit(..., dist="studentst")` | Distribution string. |
| *(score-driven / DCS)* | `gas_volatility(y, density="gaussian")` | GAS(1,1), Gaussian or `"student_t"`. No `statsmodels` analogue. |
| `MarkovAutoregression(y, k_regimes, order)` | `markov_switching_ar(y, k_regimes=2, order=1, switching_variance=True)` | Hamilton (1989) EM. Returns regimes, transition, durations. |
| `MarkovRegression` | `markov_switching_ar(..., order=0)` | Set `order=0` for a switching-mean model. |
| `UnobservedComponents`, `MLEModel` | `local_level_smooth(y, sigma2_eps, sigma2_eta)` | Only the local-level (exact-diffuse) filter ships; general custom state space is **roadmap**. |

### Systems: VAR, cointegration, factors

| statsmodels | tsecon | Notes |
|---|---|---|
| `VAR(df).fit(p)` | `var_fit(data, lags=p, trend="c")` | `data` is `T x k`. Dict: params, `sigma_u`, ICs, `max_root`. |
| `res.irf(h).orth_irfs` | `var_irf(data, lags, horizon, orth=True)` | Nested list `[h][resp][shock]`. `orth=False` for non-orthogonalized. |
| `res.irf(h).cum_effects` | `var_irf(..., cumulative=True)` | Running sums. |
| `res.fevd(h).decomp` | `var_fevd(data, lags, horizon)` | `[variable][horizon][shock]`; note the axis order differs from IRFs. |
| `res.forecast_interval(y, h)` | `var_forecast(data, lags, steps, alpha=0.05)` | Dict `{"point", "lower", "upper"}`. |
| `res.test_causality(caused, causing)` | `var_granger(data, caused, causing, lags)` | F-test; matches `statsmodels`' `test_causality`. Index lists, not names. |
| `grangercausalitytests(...)` | `var_granger(...)` | Same test, VAR-based. |
| `coint_johansen(data, det, k_ar_diff)` | `johansen(data, k_ar_diff=1)` | Trace and max-eig stats + selected ranks at 5%. |
| `VECM(data, k_ar_diff, coint_rank).fit()` | `vecm(data, k_ar_diff=1, coint_rank=1)` | ML estimation; `alpha`, `beta`, `gamma`, `sigma_u`, `llf`. |
| `coint(y0, y1)` (Engle-Granger) | — | Engle-Granger two-step: **roadmap**. Use `johansen`. |
| `DynamicFactor`, `DynamicFactorMQ` | `dfm_nowcast(data, n_factors, factor_order)` | Two-step DGR (2011) nowcaster with a ragged edge; `factor_model` for static PCA factors + Bai-Ng selection. Full MLE mixed-frequency DFM: **roadmap**. |
| `VARMAX` | — | **Roadmap.** |

Structural identification beyond Cholesky is where tsecon pulls ahead of
`statsmodels`, which offers only the recursive ordering:

| statsmodels | tsecon | Notes |
|---|---|---|
| *(recursive only)* | `var_irf(..., orth=True)` | Cholesky IRFs = the column order of `data`. |
| — | `sign_restricted_svar(data, restrictions, ...)` | Sign-restricted Bayesian SVAR with identified-set bands. **No `statsmodels` analogue.** |
| — | `bvar_fit`, `bvar_irf_draws` | Minnesota-NIW BVAR + posterior IRF draws for credible bands. |
| — | `favar(panel, policy, ...)` | Two-step FAVAR (Bernanke-Boivin-Eliasz 2005). |
| — | `connectedness(data, ...)` | Diebold-Yilmaz spillover tables from a VAR's GFEVD. |
| SVAR short/long-run (A/B, Blanchard-Quah) | — | Explicit A/B and long-run restrictions: **roadmap**. |

### Local projections — tsecon's headline addition

`statsmodels` has no local-projection module at all; this is one of the main
reasons to adopt tsecon.

| statsmodels | tsecon | Notes |
|---|---|---|
| — | `lp(y, shock, horizons=12, se="lag_augmented")` | Jordà (2005) LP IRFs, lag-augmented inference by default. |
| — | `lp_iv(y, impulse, instrument, horizons=8)` | LP-IV with a first-stage F diagnostic. |
| — | `lp_state(y, shock, state_indicator, ...)` | State-dependent (Ramey-Zubairy 2018) per-regime IRFs. |
| — | `panel_lp(outcome, shock, ...)` | Panel local projection with fixed effects. |

### Quantile regression and Growth-at-Risk

`statsmodels` ships `QuantReg`; tsecon matches it and adds the projection and
tail-risk extensions built on the same check-loss estimator.

| statsmodels | tsecon | Notes |
|---|---|---|
| `QuantReg(y, X).fit(q=tau)` | `quantile_regression(y, x, taus=[tau])` | IRLS check-loss with Powell kernel-sandwich SEs; matches `QuantReg` defaults. Include the constant column in `x`; pass several `taus` in one call. |
| — | `quantile_lp(y, shock, taus=..., horizons=8)` | Quantile local projections, `irf[tau][h]`. **No `statsmodels` analogue.** |
| — | `growth_at_risk(y, conditions, horizon=4, taus=...)` | Adrian-Boyarchenko-Giannone (2019) conditional-quantile Growth-at-Risk; `current` is the latest tail read. **No `statsmodels` analogue.** |

### Filters, spectral, forecast evaluation

| statsmodels / scipy | tsecon | Notes |
|---|---|---|
| `hpfilter(y, lamb)` | `hp_filter(y, lamb=1600, one_sided=False)` | Returns `{"trend", "cycle", ...}`. |
| `bkfilter(y, low, high, K)` | `bk_filter(y, low=6, high=32, k=12)` | Loses `k` obs each end; `first_index` tells you where the cycle starts. |
| `cffilter(y, low, high, drift)` | `cf_filter(y, low=6, high=32, drift=True)` | Christiano-Fitzgerald. |
| *(none)* | `hamilton_filter(y, h=8, p=4)` | Hamilton (2018) regression filter — the modern HP alternative. |
| `scipy.signal.periodogram` | `periodogram(x, fs, window, detrend)` | Matches SciPy. |
| `scipy.signal.welch` | `welch(x, nperseg, ...)` | Matches SciPy. |
| `scipy.signal.coherence` | `coherence(x, y, nperseg, ...)` | Magnitude-squared coherence. |
| `seasonal_decompose`, `STL` | — | **Roadmap.** |

For forecast comparison, tsecon ships a fuller battery than `statsmodels.tsa`:
`dm_test` (Diebold-Mariano with the Harvey-Leybourne-Newbold correction),
`cw_test` (Clark-West, nested models), `gw_test` (Giacomini-White), `accuracy`
(ME/RMSE/MAE/MAPE/sMAPE/MASE/RMSSE), and a rolling/expanding `backtest` engine.

## Worked translations

The tables above are the reference; here are five you can run. Each shows the
`statsmodels` call as a comment.

### Unit-root screen and a portmanteau test

```python
import numpy as np, tsecon
rng = np.random.default_rng(0)
y = np.cumsum(rng.standard_normal(300))               # a random walk

r = tsecon.adf(y, regression="c")                     # adfuller(y)
print(round(r["statistic"], 3), round(r["p_value"], 3), r["used_lag"])

lb = tsecon.ljung_box(y, nlags=10)                    # acorr_ljungbox(y, 10)
print(round(lb["lb_stat"][-1], 2), round(lb["lb_pvalue"][-1], 4))
```

### OLS with Newey-West standard errors

Note the explicit constant column — there is no `add_constant`.

```python
import numpy as np, tsecon
rng = np.random.default_rng(2)
X = rng.standard_normal((200, 2))
y = 3.0 + X @ np.array([1.0, -0.5]) + rng.standard_normal(200)

Xc = np.column_stack([np.ones(len(y)), X])            # add the intercept yourself
# statsmodels: OLS(y, add_constant(X)).fit(cov_type="HAC",
#                                          cov_kwds={"maxlags": 4})
res = tsecon.ols(y, Xc, se_type="hac", maxlags=4)
print(np.round(res["params"], 3), res["se_type"])
```

### A VAR, its orthogonalized IRFs, and its FEVD

```python
import numpy as np, tsecon
rng = np.random.default_rng(1)
data = rng.standard_normal((200, 3))                  # T x k; columns are variables

fit = tsecon.var_fit(data, lags=2, trend="c")         # VAR(df).fit(2)
print(round(fit["aic"], 3), round(fit["max_root"], 3))

irf = tsecon.var_irf(data, lags=2, horizon=10, orth=True)   # .irf(10).orth_irfs
resp = irf[5][0][1]          # horizon 5, response of var 0 to a shock in var 1
fevd = tsecon.var_fevd(data, lags=2, horizon=10)            # .fevd(10)
share = fevd[0][9][1]        # var 0, horizon 10, share explained by shock 1
print(round(resp, 4), round(share, 4))
```

### GARCH(1,1) with robust standard errors

The `arch` package is the usual companion to `statsmodels` for volatility;
`garch_fit` covers the common cases.

```python
import numpy as np, tsecon
rng = np.random.default_rng(3)
# a GARCH(1,1) return series so the QMLE has clustering to fit
e = rng.standard_normal(1500); h = np.empty(1500); r = np.empty(1500)
h[0] = 0.5; r[0] = np.sqrt(h[0]) * e[0]
for t in range(1, 1500):
    h[t] = 0.05 + 0.08 * r[t-1]**2 + 0.90 * h[t-1]
    r[t] = np.sqrt(h[t]) * e[t]

g = tsecon.garch_fit(r, vol="garch", p=1, q=1, dist="normal")
# arch: arch_model(r, vol="Garch", p=1, q=1, mean="Zero").fit()
print(dict(zip(g["param_names"], np.round(g["params"], 4))))
print(np.round(g["se_robust"], 4))                    # Bollerslev-Wooldridge SEs
```

### A local projection — the thing you came here for

```python
import numpy as np, tsecon
rng = np.random.default_rng(4)
n = 300
shock = rng.standard_normal(n)
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.5 * y[t-1] + 0.8 * shock[t] + rng.standard_normal()

out = tsecon.lp(y, shock, horizons=12, se="lag_augmented")   # no statsmodels analogue
print(np.round(out["irf"][:3], 3), np.round(out["se"][:3], 3))
```

## What `statsmodels` has that tsecon does not (yet)

Be direct about the gaps so you can plan around them. As of this writing the
following common `statsmodels.tsa` capabilities are **roadmap**, not shipped:

- **Seasonal ARIMA / SARIMAX** seasonal orders (non-seasonal `arima_fit` ships).
- **ExponentialSmoothing / ETS** beyond the Theta method.
- **Phillips-Perron, Zivot-Andrews, Elliott-Rothenberg-Stock** unit-root tests.
- **STL / seasonal_decompose** classical decomposition.
- **VARMAX / VARMA**, and the **Engle-Granger** two-step cointegration test.
- **Explicit SVAR restrictions** (short-run A/B, long-run / Blanchard-Quah);
  tsecon ships Cholesky (`var_irf(orth=True)`) and sign restrictions
  (`sign_restricted_svar`) instead.
- **Custom state-space models** (`MLEModel`, `UnobservedComponents`); only the
  local-level filter and the internal DFM state space ship.
- **Frequentist VAR IRF confidence bands.** `var_irf` returns point responses;
  for bands use the Bayesian `bvar_irf_draws` or `sign_restricted_svar`.

Where tsecon *leads* `statsmodels.tsa` — and why the switch is often worth it —
is the modern macro-structural toolkit: local projections (`lp`/`lp_iv`/
`lp_state`/`panel_lp`), Bayesian VARs (`bvar_fit`/`bvar_irf_draws`),
sign-restricted SVARs (`sign_restricted_svar`), FAVAR (`favar`), nowcasting
(`dfm_nowcast`/`dfm_news`), MIDAS mixed frequency (`umidas`/`weighted_midas`),
IV-GMM (`iv_gmm`), heterogeneous panels (`panel_mean_group`/`panel_pmg`),
conditional-quantile Growth-at-Risk (`growth_at_risk`/`quantile_lp`),
Bai-Perron multiple-break estimation (`bai_perron`/`sup_f_test`), and a Rust core
fast enough to make 5,000-draw bootstraps a default rather than a luxury.

See also the [R](from-r.md) and [Stata](from-stata.md) guides, and the
cross-package [Rosetta glossary](rosetta.md).
