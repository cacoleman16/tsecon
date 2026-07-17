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
VARs beyond `bayes:`, nowcasting) and a compiled core built for simulation. This
page is the phrasebook.

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
| `irf table fevd` | `var_fevd(data, lags, horizon=h)` | `[variable][horizon][shock]`; sums to 1 across shocks. |
| `irf ..., cumulative` | `var_irf(..., cumulative=True)` | |
| `vargranger` | `var_granger(data, caused, causing, lags)` | F-test; pass integer column indices. |
| `fcast compute, step(h)` | `var_forecast(data, lags, steps=h, alpha=0.05)` | `{"point", "lower", "upper"}`. |
| `varstable` | read `var_fit(...)["max_root"]` | Stable iff `max_root < 1`. |
| `varsoc` (lag-order selection) | compare `var_fit(...)["aic"/"bic"/"hqic"]` | No single command; loop over `lags`. |
| `svar ..., aeq() beq()` (short-run A/B) | — | Explicit A/B restrictions: **roadmap**. Use Cholesky (`var_irf`) or sign restrictions. |
| `svar ..., lreq()` (long-run) | — | Long-run / Blanchard-Quah: **roadmap**. |
| *(sign restrictions — not in core Stata)* | `sign_restricted_svar(data, restrictions, ...)` | Sign-restricted Bayesian SVAR + identified-set bands. |
| `bayes: var ...` | `bvar_fit`, `bvar_irf_draws` | Minnesota-NIW BVAR + posterior IRF draws. |

### Cointegration and unit roots

| Stata | tsecon | Notes |
|---|---|---|
| `dfuller y, lags(k)` | `adf(y, regression="c", maxlag=k)` | Dict with MacKinnon p-value. `trend` → `regression="ct"`. |
| `kpss y` | `kpss(y, regression="c")` | Null is stationarity. |
| `pperron y` | — | Phillips-Perron: **roadmap**. |
| `wntestq y` | `ljung_box(y, nlags)` | Ljung-Box (and Box-Pierce). |
| `estat archlm` | `arch_lm(resid, nlags)` | Engle's ARCH-LM. |
| `vecrank y1 y2 y3` | `johansen(data, k_ar_diff)` | Trace + max-eig ranks. |
| `vec y1 y2 y3, rank(r)` | `vecm(data, k_ar_diff, coint_rank=r)` | ML VECM: `alpha`, `beta`, `gamma`, `sigma_u`, `llf`. |
| `egranger` (user-written) | — | Engle-Granger two-step: **roadmap**; use `johansen`. |

### Univariate models and volatility

| Stata | tsecon | Notes |
|---|---|---|
| `arima y, arima(p,d,q)` | `arima_fit(y, p, d, q, constant=True)` | Exact-MLE. Seasonal `arima(...)(P,D,Q)`: **roadmap**. |
| `arch y, arch(1) garch(1)` | `garch_fit(y, vol="garch", p=1, q=1)` | Robust SEs in `se_robust` (Bollerslev-Wooldridge). |
| `arch y, arch(1) garch(1) tarch(1)` (GJR) | `garch_fit(y, vol="garch", o=1)` | Asymmetry via `o=`. |
| `arch ..., earch(1) egarch(1)` | `garch_fit(y, vol="egarch")` | |
| `arch ..., distribution(t)` | `garch_fit(y, dist="studentst")` | |
| `mgarch dcc (y1 y2 y3), arch(1) garch(1)` | `dcc_garch(returns)` | Engle (2002) DCC; `returns` is `T x k`. |
| `mgarch ccc (...)` | `ccc_garch(returns)` | Bollerslev (1990) CCC. |
| `mswitch ar y, states(k)` | `markov_switching_ar(y, k_regimes=k, order=1, switching_variance=)` | Hamilton EM; regimes, transition, durations. |
| `mswitch dr y` (dynamic regression) | `markov_switching_ar(..., order=0)` | Switching-mean model. |
| *(score-driven — not in Stata)* | `gas_volatility(y, density=)` | GAS(1,1), Gaussian or Student-t. |

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
| `xtunitroot ips/llc` | — | Panel unit-root tests: **roadmap**. |

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
| `psdensity`, `pergram` | `periodogram(x)`, `welch(x)`, `coherence(x, y)` | Match SciPy's spectral estimators. |
| `newey y x, lag(L)` | `ols(y, Xc, se_type="hac", maxlags=L)` | Prepend your own constant column `Xc`. |
| *(DM test — user-written `dmariano`)* | `dm_test(e1, e2, h, loss)` | With HLN correction; also `cw_test`, `gw_test`. |
| `dfactor` (state-space DFM) | `dfm_nowcast(data, n_factors, factor_order)` | Two-step nowcaster with a ragged edge; `factor_model` for static PCA factors. Full MLE `dfactor`: **roadmap**. |
| *(MIDAS — user-written `midasreg`)* | `weighted_midas(y, hf, scheme=)`, `umidas(y, hf)` | Restricted and unrestricted mixed-frequency regressions. |

### Realized volatility and term structure

| Stata | tsecon | Notes |
|---|---|---|
| *(realized measures — user-written)* | `realized_measures(returns)` | RV, bipower variation, jump component. |
| *(HAR — user-written)* | `har_rv(rv, variant=)` | HAR-RV (Corsi 2009), HAC SEs. |
| *(Diebold-Yilmaz — user-written)* | `connectedness(data, lags, horizon)` | Spillover table from a VAR's GFEVD. |
| *(Nelson-Siegel — user-written)* | `nelson_siegel(maturities, yields)`, `svensson(...)`, `dynamic_ns(...)` | Yield-curve fitting and dynamic factors. |

## Worked translations

Five you can run. The Stata command is shown as a comment.

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

## What Stata has that tsecon does not (yet)

Being direct about the gaps, the following are **roadmap**, not shipped:

- **Explicit SVAR restrictions** — `svar`'s short-run (A/B) and long-run
  (`lreq`) identification. tsecon offers Cholesky (`var_irf(orth=True)`) and
  sign restrictions (`sign_restricted_svar`).
- **Phillips-Perron** (`pperron`) and **panel unit-root tests** (`xtunitroot`).
- **Seasonal ARIMA** (`arima ...(P,D,Q)`) and general **`ucm`** unobserved-components models.
- **Engle-Granger** (`egranger`) two-step cointegration; use `johansen`.
- **`estat` post-estimation batteries** and formatted `esttab`/`irf table`
  output — you format results from the returned dicts yourself.
- **Full MLE `dfactor`**; the shipped nowcaster is the two-step DGR estimator.

Where tsecon pays you back is the frontier Stata reaches only through
user-written `.ado` files or not at all: local projections
(`lp`/`lp_iv`/`lp_state`/`panel_lp`), sign-restricted and Bayesian VARs, FAVAR,
Diebold-Yilmaz connectedness, realized-volatility measures, and DFM nowcasting —
all under one calling grammar, on a core fast enough to bootstrap by default.

See also the [statsmodels](from-statsmodels.md) and [R](from-r.md) guides, and
the cross-package [Rosetta glossary](rosetta.md).
