# Migrating from Stata

> Part of [The tsecon Guide to Time Series Econometrics](../guide/README.md). An
> adoption guide for Stata users: it maps the `ts` and `xt` commands you know —
> `var`, `irf`, `svar`, `dfactor`, `arch`, `mgarch`, `ivregress gmm`, `xtpmg`,
> `vec` — to tsecon functions, and is candid about the gaps. Every Python block
> runs against the current library.

Stata's time-series suite is polished, consistent, and command-driven: you
`tsset` your data, run a command, and read a formatted table. tsecon trades that
turnkey feel for a programmable one — you assemble arrays, call a function, and
get back a `dict` you can compute on. The reasons to make the trade are the
methods Stata does not have (local projections, sign-restricted SVARs, Bayesian
VARs beyond `bayes:`, nowcasting, threshold and smooth-transition models) and a
compiled core built for simulation. This page is the phrasebook.

## What changes when you cross over

Four adjustments:

1. **No `tsset`, no variable names — arrays and column order.** Stata carries a
   time index and named variables through every command. tsecon takes a plain
   `T x k` NumPy array; the *order of the columns* is what Stata's variable list
   encodes, and it is also your Cholesky ordering. Keep a Python list of column
   names alongside the array.

2. **Results are dictionaries, not `e()` returns and tables.** Instead of reading
   a printed table and pulling scalars from `e(b)`/`e(V)`, you get a `dict`:
   `res["params"]`, `res["bse"]`, `res["se_type"]`. There is no `estimates
   store`/`esttab`; you format output yourself.

3. **Robust SEs are one argument, not a `vce()` option per command.** Every
   regression estimator takes `se_type=`: `"nonrobust"`, `"hc0"`–`"hc3"`,
   `"hac"` (the `newey` equivalent), and for panels `"cluster"` and
   `"driscoll_kraay"` (the `xtscc` equivalent). The choice is stamped into the
   result.

4. **Panels are lists of per-unit arrays.** Stata's long `xt` layout (one row per
   `panelvar × timevar`) becomes either a *list* of per-unit arrays (`ys`, `xs`)
   for the mean-group/PMG estimators, or a dense `N x T` outcome with a
   `k x N x T` regressor tensor for `panel_fe`/`panel_lp`.

## The mapping tables

"Roadmap" marks a capability tsecon does not ship today. Everything else is
callable now.

### VAR, SVAR, IRF, FEVD

| Stata | tsecon | Notes |
|---|---|---|
| `var y1 y2 y3, lags(1/p)` | `var_fit(data, lags=p, trend="c")` | `data` is `T x k`; `noconstant` → `trend="n"`, `trend` option → `trend="t"`. |
| `irf create ..., step(h)` then `irf graph oirf` | `var_irf(data, lags, horizon=h, orth=True)` | Orthogonalized IRFs, nested list `[h][response][shock]`. |
| `irf graph irf` (non-orthogonal) | `var_irf(..., orth=False)` | |
| `irf table fevd` | `var_fevd(data, lags, horizon=h)` | `[horizon][variable][shock]`, horizon-first like the IRFs; sums to 1 across shocks. |
| `irf ..., cumulative` | `var_irf(..., cumulative=True)` | |
| `vargranger` | `var_granger(data, caused, causing, lags)` | F-test; pass integer column indices. |
| `fcast compute, step(h)` | `var_forecast(data, lags, steps=h, alpha=0.05)` | `{"point", "lower", "upper"}`. |
| `varstable` | read `var_fit(...)["is_stable"]` | Stable iff `is_stable` (equivalently `min_root > 1`; these are *reciprocal* roots, so `max_root` is not a verdict). |
| `varsoc` (lag-order selection) | compare `var_fit(...)["aic"/"bic"/"hqic"]` | No single command; loop over `lags`. |
| `svar ..., aeq() beq()` (short-run A/B) | — | Explicit A/B restrictions: **roadmap**. Use Cholesky (`var_irf`), sign restrictions, or `zero_sign_svar` for zero-*and*-sign set identification. |
| `svar ..., lreq()` (long-run) | `long_run_svar(data, lags, horizon, restrictions=None)` | Blanchard-Quah long-run zeros in closed form: `impact`, `long_run`, `irf`, `cumulative_irf`, `fevd`, `long_run_multiplier`. Point estimates, no RNG. |
| *(sign restrictions — not in core Stata)* | `sign_restricted_svar(data, restrictions, ...)` | Sign-restricted Bayesian SVAR + identified-set bands. |
| *(statistical identification — not in Stata)* | `hetero_svar(data, regime_labels, ...)`, `nongaussian_svar(data, lags, ...)` | Identification through heteroskedasticity (Rigobon 2003, two known variance regimes) and through non-Gaussianity (FastICA on the residuals alone). Also `proxy_svar`, `narrative_svar`, `max_share_svar`, `fry_pagan_svar`. |
| `bayes: var ...` | `bvar_fit`, `bvar_irf_draws` | Minnesota-NIW BVAR + posterior IRF draws. |

### Cointegration and unit roots

| Stata | tsecon | Notes |
|---|---|---|
| `dfuller y, lags(k)` | `adf(y, regression="c", maxlag=k)` | Dict with MacKinnon p-value. `trend` → `regression="ct"`. |
| `kpss y` | `kpss(y, regression="c")` | Null is stationarity. |
| `pperron y` | `phillips_perron(y, regression="c", test_type="tau")` | Semiparametric unit-root test; `test_type="rho"` is Z-alpha. Matches `arch.unitroot.PhillipsPerron` to 1e-10. |
| `dfgls y` | `dfgls(y, regression="c", method="aic")` | Elliott-Rothenberg-Stock GLS-detrended ADF. `ng_perron(y, trend=)` adds the Ng-Perron M tests on the same engine (statistic-only — no p-value surface exists, so none is invented). |
| `wntestq y` | `ljung_box(y, nlags)` | Ljung-Box (and Box-Pierce). |
| `estat archlm` | `arch_lm(resid, nlags)` | Engle's ARCH-LM. |
| `vecrank y1 y2 y3` | `johansen(data, k_ar_diff)` | Trace + max-eig ranks. |
| `vec y1 y2 y3, rank(r)` | `vecm(data, k_ar_diff, coint_rank=r, deterministic="co")` | ML VECM: `alpha`, `beta`, `gamma`, `det_coef`, `sigma_u`, `llf`. Stata's default `trend(constant)` is the unrestricted constant → `deterministic="co"`; tsecon's default `"n"` is `trend(none)`. |
| `egranger` (user-written) | `engle_granger(data, trend="c", autolag="aic")` | Engle-Granger two-step: column 0 of `data` is the regressand, columns 1.. its regressors, deterministics come from `trend` (do not add a constant column). Statistic and p-value match `statsmodels.tsa.stattools.coint` at 1e-10 / 1e-9; the step-1 coefficients and residuals come back too. The Phillips-Ouliaris residual test is `phillips_ouliaris(y, x, trend="c", test_type="Zt")`. |

### Univariate models and volatility

| Stata | tsecon | Notes |
|---|---|---|
| `arima y, arima(p,d,q)` | `arima_fit(y, p, d, q, constant=True)` | Exact-MLE. |
| `arima y, arima(p,d,q) sarima(P,D,Q,s)` | `arima_fit(y, p, d, q, seasonal=(P, D, Q, s))` | Multiplicative seasonal ARIMA on the same exact-MLE engine; the airline model is `arima_fit(np.log(air), p=0, d=1, q=1, seasonal=(0, 1, 1, 12), constant=False)`. Seasonal parameters are named statsmodels-style (`ar.S.L12`, `ma.S.L12`). |
| *(automatic order selection — no native command)* | `auto_arima(y, seasonal_period=, ic="aicc", stepwise=True)` | The Hyndman-Khandakar (2008) `auto.arima` algorithm on the `arima_fit` engine: `nsdiffs` for `D`, `ndiffs` for `d`, then a stepwise (or, with `stepwise=False`, exhaustive) order search at those fixed differencing orders. |
| `arch y, arch(1) garch(1)` | `garch_fit(y, vol="garch", p=1, q=1)` | Robust SEs in `se_robust` (Bollerslev-Wooldridge). |
| `arch y, arch(1) garch(1) tarch(1)` (GJR) | `garch_fit(y, vol="gjr", o=1)` | Asymmetry via `o=`, which requires `vol="gjr"`/`"egarch"` — `o > 0` with `vol="garch"` raises (0.6.0). |
| `arch ..., earch(1) egarch(1)` | `garch_fit(y, vol="egarch")` | |
| `arch ..., distribution(t)` | `garch_fit(y, dist="studentst")` | |
| `mgarch dcc (y1 y2 y3), arch(1) garch(1)` | `dcc_garch(returns)` | Engle (2002) DCC; `returns` is `T x k`. |
| `mgarch ccc (...)` | `ccc_garch(returns)` | Bollerslev (1990) CCC. |
| `mswitch ar y, states(k)` | `markov_switching_ar(y, k_regimes=k, order=1, switching_variance=)` | Hamilton EM; regimes, transition, durations. |
| `mswitch dr y` (dynamic regression) | `markov_switching_ar(..., order=0)` | Switching-mean model. |
| *(score-driven — not in Stata)* | `gas_volatility(y, density=)` | GAS(1,1), Gaussian or Student-t. |

### Threshold and smooth-transition models

Stata's nonlinear time-series menu stops at `mswitch` and the `threshold`
regression command. The self-exciting and multivariate threshold family — the R
`tsDyn` territory — is where tsecon goes furthest past it; the VAR, VECM and
STAR blocks are new in 0.7.0.

| Stata | tsecon | Notes |
|---|---|---|
| `threshold y x, ...` (threshold *regression*) | `setar(y, p, delay=, delays=)`, `setar_test(y, p, n_boot=, seed=)` | Stata's `threshold` splits a regression on an exogenous threshold variable; `setar` is the *self-exciting* threshold AR (Tong-Lim 1980), split on `y_{t-delay}` and fitted by concentrated LS. `setar_test` is Hansen's (1996) sup-F linearity test — bootstrap p-value only, because the threshold is an unidentified nuisance parameter under the null and a chi-squared p-value would be wrong. |
| *(smooth transition — no native command)* | `star(y, p, model="lstar"/"estar")`, `star_eval(...)`, `star_test(y, p)` | LSTAR/ESTAR by concentrated NLS (Terasvirta 1994); `star_test` is the LM3 linearity test plus the H03/H02/H01 sequence that picks between them. `gamma` is the raw transition slope, `gamma_standardized` the scale-free one; `converged`/`gamma_at_boundary`/`se_valid` report honestly when the surface is flat. |
| *(threshold VAR — no native command)* | `threshold_var(data, p, threshold_index=, delay=)`, `threshold_var_test(...)` | Two-regime threshold VAR by concentrated LS on `log_det_sigma`, with a robust score-form sup-Wald test bootstrapped à la Hansen (1996). |
| *(threshold cointegration — no native command)* | `threshold_vecm(data, k_ar_diff, beta=None)`, `hansen_seo_test(...)` | Hansen-Seo (2002): the error-correction term drives the regime split, estimation is the concentrated Gaussian MLE over a `(beta, gamma)` grid, and `hansen_seo_test` is their sup-LM test of linear against threshold cointegration. `beta=None` estimates the cointegrating vector, bivariate only. |

**How this block is graded.** R's `tsDyn` — the only package implementing these
estimators — could not be installed in the fixture container (CRAN unreachable
through its egress proxy), so **no third-party reference run exists for any row
above**. They are carried by closed forms transcribed from the published papers
and pinned at 1e-10 against an independent NumPy implementation (1e-8 for the
estimated-`beta` threshold-VECM cases, where an eigensolver is in the path), plus
seeded Monte-Carlo size, power and recovery: threshold-VECM null size 0.100 at
`T=150` falling to 0.065 at `T=400`, threshold-VAR 0.100 → 0.085 over the same
range, STAR LM3-F size 0.060/0.028 at `T=200/500`. The `tsDyn` reference run is
named follow-up work. Do not read these rows as "validated against R" — the
per-function grades are in the [validation matrix](../reference/validation-matrix.md).

### Quantile regression and Growth-at-Risk

| Stata | tsecon | Notes |
|---|---|---|
| `qreg y x` | `quantile_regression(y, X)` | Median regression by default; `X` is `T x k` with a constant. Powell kernel-sandwich SEs (`bse`, `tvalues`). |
| `sqreg y x, quantiles(10 50 90)` | `quantile_regression(y, X, taus=[0.1, 0.5, 0.9])` | Simultaneous multi-tau fit; `params`/`bse` are per-tau. |
| *(quantile LP — user-written)* | `quantile_lp(y, shock, taus=, horizons=h)` | Quantile local projections, `irf[tau][h]` with sandwich `se`. |
| *(Growth-at-Risk — no native command)* | `growth_at_risk(y, conditions, horizon=h)` | Adrian-Boyarchenko-Giannone conditional-quantile GaR; `current` is the latest risk read across `taus`. |

### Specification tests and structural breaks

| Stata | tsecon | Notes |
|---|---|---|
| `estat hettest` (Breusch-Pagan) | `heteroskedasticity_test(y, X, test="breusch_pagan")` | `X` is `T x k` with a constant; `test="white"` for `estat imtest, white`. |
| `estat ovtest` (Ramsey RESET) | `reset_test(y, X, max_power=3)` | Functional-form F-test on fitted powers. |
| `estat sbknown, break(t)` | `chow_test(y, X, split=t)` | Chow break at a known 0-indexed split; `fstat`, `pvalue`, per-regime SSR. |
| `estat sbsingle` (unknown break) | `sup_f_test(y, X, trim=0.15)` | Andrews-Quandt sup-F, Hansen (1997) p-value, estimated `break_date`. |
| *(multiple breaks — no native command)* | `bai_perron(y, X, max_breaks=m)` | Bai-Perron global partition + sequential supF(l+1\|l); `break_dates` with Bai (1997) CIs. |
| `estat sbcusum` / `cusum` | `cusum_test(y, X)` | Brown-Durbin-Evans CUSUM `path` with 5% bounds. |
| *(one-call diagnostics — run each `estat` by hand)* | `check_series(y)` | Battery: stationarity, serial correlation, ARCH, normality, a break scan, long memory, seasonality, plus an ordered `recommendations` list routing to concrete calls. |

### Panels — `xt` commands

Stata's long `xt` layout maps to lists of per-unit arrays or a dense `N x T`
outcome, as noted above.

| Stata | tsecon | Notes |
|---|---|---|
| `xtreg y x, fe` | `panel_fe(outcome, regressors, se_type="nonrobust")` | `outcome` is `N x T`, `regressors` is `k x N x T`. |
| `xtreg y x, fe vce(cluster id)` | `panel_fe(..., se_type="cluster")` | Clustered by entity. |
| `xtscc y x, fe` (Driscoll-Kraay) | `panel_fe(..., se_type="driscoll_kraay")` | Same SE, one argument. |
| `xtpmg d.y ..., pmg` | `panel_pmg(ys, xs)` | Pooled Mean Group ARDL(1,1) (Pesaran-Shin-Smith 1999). |
| `xtpmg d.y ..., mg` | `panel_mean_group(ys, xs, method="mg")` | Mean group (Pesaran-Smith 1995). |
| `xtmg y x, cce` (Eberhardt CCEMG) | `panel_mean_group(ys, xs, method="cce")` | Common-correlated-effects mean group. |
| Panel VAR (`pvar`, user-written) | `mean_group_var(entities, lags, horizon)` | Mean-group panel VAR over per-entity `T_i x k` matrices. |
| Panel LP (user-written `lp`) | `panel_lp(outcome, shock, ...)` | Panel local projection with fixed effects. |
| `xtunitroot ips/llc/fisher` | `panel_unit_root(data, test="ips"/"llc"/"fisher")` | Levin-Lin-Chu, Im-Pesaran-Shin and the Fisher-type (Maddala-Wu / Choi) combinations. `data` is a balanced `N x T` array (a row per unit) or a list of per-unit series — unbalanced is fine for `"ips"`/`"fisher"`. Conventions follow `plm::purtest`. |
| *(short-`T` LP bias correction — no native command)* | `panel_lp(..., bias_correction="spj")` | Split-panel jackknife for the panel-LP bias; Monte-Carlo measured at a 15x bias cut and coverage 0.74 → 0.82 at `T=20`, which is an improvement and still short of nominal — documented, not smoothed over. |

### GMM and IV

| Stata | tsecon | Notes |
|---|---|---|
| `ivregress gmm y (x1 = z1) x2, wmatrix(robust)` | `iv_gmm(x, z, y, method="2step", weight="robust")` | `x` = all regressors, `z` = instruments *including* the exogenous columns. |
| `ivregress gmm ..., igmm` | `iv_gmm(..., method="iterated")` | Iterated GMM. |
| `ivregress 2sls ...` | `iv_gmm(x, z, y, method="2sls")` | The 2SLS special case. |
| `estat overid` (Hansen J) | `iv_gmm(...)["j_stat"/"j_dof"/"j_pval"]` | Present only when over-identified. |
| `gmm (...)` (nonlinear moment eqs) | `gmm_nonlinear(moments_fn, initial, weight=)` | Custom moment function as a Python callback. |
| `ivregress ..., wmatrix(hac ...)` | `iv_gmm(..., weight="hac", bandwidth=)` | HAC weighting matrix. |

### Filters, spectra, forecast evaluation, mixed frequency

| Stata | tsecon | Notes |
|---|---|---|
| `tsfilter hp cyc = y, smooth(1600)` | `hp_filter(y, lamb=1600)` | Returns `{"trend", "cycle", ...}`. |
| `tsfilter bk cyc = y` | `bk_filter(y, low=6, high=32, k=12)` | |
| `tsfilter cf cyc = y` | `cf_filter(y, low=6, high=32, drift=True)` | |
| *(Hamilton filter — not in Stata)* | `hamilton_filter(y, h=8, p=4)` | Hamilton (2018) regression filter. |
| *(STL — no native command)* | `stl(y, period, ...)`, `mstl(y, periods)` | Cleveland et al. (1990) seasonal-trend decomposition and the multi-seasonal MSTL iteration, both pinned elementwise to statsmodels at 1e-8; `seasonal_strength`, `ndiffs` and `nsdiffs` sit on top. |
| `psdensity`, `pergram` | `periodogram(x)`, `welch(x)`, `coherence(x, y)` | Match SciPy's spectral estimators. |
| `newey y x, lag(L)` | `ols(y, Xc, se_type="hac", maxlags=L)` | Prepend your own constant column `Xc`. |
| *(DM test — user-written `dmariano`)* | `dm_test(e1, e2, h, loss)` | With HLN correction; also `cw_test`, `gw_test`. |
| `dfactor` (state-space DFM) | `dfm_nowcast(data, n_factors, factor_order, method="two_step"/"mle")` | `"two_step"` (default) is the Doz-Giannone-Reichlin nowcaster with a ragged edge and no iterative optimizer. `"mle"` fits the same state space by exact Gaussian likelihood — the `dfactor` estimand — and reports `converged`/`iterations` so a budget-limited fit says so. `factor_model` gives static PCA factors with the Bai-Ng criteria. |
| *(MIDAS — user-written `midasreg`)* | `weighted_midas(y, hf, scheme=)`, `umidas(y, hf)` | Restricted and unrestricted mixed-frequency regressions. |

### Realized volatility and term structure

| Stata | tsecon | Notes |
|---|---|---|
| *(realized measures — user-written)* | `realized_measures(returns)` | RV, bipower variation, jump component. |
| *(HAR — user-written)* | `har_rv(rv, variant=)` | HAR-RV (Corsi 2009), HAC SEs. |
| *(Diebold-Yilmaz — user-written)* | `connectedness(data, lags, horizon)` | Spillover table from a VAR's GFEVD. |
| *(Nelson-Siegel — user-written)* | `nelson_siegel(maturities, yields)`, `svensson(...)`, `dynamic_ns(...)` | Yield-curve fitting and dynamic factors. |

## Worked translations

Six you can run. The Stata command is shown as a comment.

### `arch` → `garch_fit`

```python
import numpy as np, tsecon
# a GARCH(1,1) return series so the QMLE has clustering to fit
rng = np.random.default_rng(10)
e = rng.standard_normal(1500); h = np.empty(1500); r = np.empty(1500)
h[0] = 0.5; r[0] = np.sqrt(h[0]) * e[0]
for t in range(1, 1500):
    h[t] = 0.05 + 0.08 * r[t-1]**2 + 0.90 * h[t-1]
    r[t] = np.sqrt(h[t]) * e[t]

g = tsecon.garch_fit(r, vol="garch", p=1, q=1)         # arch r, arch(1) garch(1)
print(dict(zip(g["param_names"], np.round(g["params"], 4))))
```

### `ivregress gmm` → `iv_gmm`

The instrument matrix `z` must include the exogenous regressor columns, exactly
as Stata's included exogenous variables enter both sides.

```python
import numpy as np, tsecon
rng = np.random.default_rng(11)
n = 400
z = rng.standard_normal((n, 3))                        # instruments incl. exog columns
x = np.column_stack([z[:, 0] + rng.standard_normal(n), z[:, 1], z[:, 2]])
y = x @ np.array([1.0, 0.5, -0.3]) + rng.standard_normal(n)

# ivregress gmm y (x1 = z1) x2 x3, wmatrix(robust)
res = tsecon.iv_gmm(x, z, y, method="2step", weight="robust")
print(np.round(res["params"], 3), np.round(res["bse"], 3))
```

### `xtpmg ..., pmg` → `panel_pmg`

```python
import numpy as np, tsecon
rng = np.random.default_rng(12)
ys = [rng.standard_normal(50) for _ in range(20)]      # one array per panel unit
xs = [rng.standard_normal((50, 2)) for _ in range(20)]

pmg = tsecon.panel_pmg(ys, xs)                         # xtpmg d.y ..., pmg
print(np.round(pmg["theta"], 3), round(pmg["phi_bar"], 3))   # long-run coefs, EC speed
```

### `var` + `irf` → `var_fit` + `var_irf`

```python
import numpy as np, tsecon
rng = np.random.default_rng(13)
data = rng.standard_normal((200, 3))

fit = tsecon.var_fit(data, lags=2)                     # var y1 y2 y3, lags(1/2)
irf = tsecon.var_irf(data, lags=2, horizon=8, orth=True)   # irf create; irf graph oirf
print(round(fit["hqic"], 3), np.asarray(irf).shape)    # (horizon+1, k, k)
```

### `dfactor` / nowcast → `dfm_nowcast`

```python
import numpy as np, tsecon
rng = np.random.default_rng(14)
data = rng.standard_normal((100, 8))
data[-1, 4:] = np.nan                                  # a ragged edge (late releases)

nc = tsecon.dfm_nowcast(data, n_factors=2, factor_order=1)
print(np.round(nc["nowcast"][4:], 4))                  # the filled-in missing entries
```

### Growth-at-Risk → `growth_at_risk` (no Stata equivalent)

Stata has `qreg`/`sqreg` but no conditional-quantile Growth-at-Risk in one call;
`growth_at_risk` fits the h-ahead quantiles on `[const, conditions, y_t]` at every
date and hands back the latest read.

```python
import numpy as np, tsecon
rng = np.random.default_rng(20)
n = 200
fci = rng.standard_normal(n)                           # a financial-conditions index
g = 0.5 - 0.8 * fci + rng.standard_normal(n)           # growth, downside-sensitive to FCI

gar = tsecon.growth_at_risk(g, fci[:, None], horizon=4, taus=[0.05, 0.5, 0.95])
print(np.round(gar["current"], 3))                     # 5% / 50% / 95% four-quarter-ahead
```

## What Stata has that tsecon does not (yet)

Being direct about the gaps, the following are **roadmap**, not shipped:

- **Explicit short-run SVAR restrictions** — `svar`'s `aeq()`/`beq()` A/B
  system. Everything else on that menu ships: Cholesky (`var_irf(orth=True)`),
  long-run (`long_run_svar`), sign (`sign_restricted_svar`), zero-and-sign
  (`zero_sign_svar`), narrative, proxy and max-share.
- **`ucm` and general `sspace`** — user-specified unobserved-components and
  state-space models. `local_level_smooth` and `dcs_local_level` are the shipped
  state-space pieces; `dfm_nowcast` covers the factor case.
- **Formatted `esttab`/`irf table` output** — you format results from the
  returned dicts yourself. (The individual `estat` diagnostics *do* ship —
  `hettest`, `ovtest`, `sbknown`, `sbsingle`, `sbcusum` map to functions; see
  the specification-tests table.)

Five items that used to sit on this list have shipped and now appear in the
tables above: `pperron` → `phillips_perron`, `xtunitroot` → `panel_unit_root`,
seasonal `arima` → `arima_fit(seasonal=(P, D, Q, s))`, `egranger` →
`engle_granger`, and the full-MLE `dfactor` estimand →
`dfm_nowcast(..., method="mle")`.

Where tsecon pays you back is the frontier Stata reaches only through
user-written `.ado` files or not at all: local projections
(`lp`/`lp_iv`/`lp_state`/`panel_lp`, plus smooth and quantile variants),
sign-restricted and Bayesian VARs, FAVAR, Diebold-Yilmaz connectedness,
realized-volatility measures, DFM nowcasting, conditional-quantile
Growth-at-Risk, and the threshold/smooth-transition family (`setar`, `star`,
`threshold_var`, `threshold_vecm` and their linearity tests) — all under one
calling grammar, on a core fast enough to bootstrap by default.

See also the [statsmodels](from-statsmodels.md) and [R](from-r.md) guides, and
the cross-package [Rosetta glossary](rosetta.md).
