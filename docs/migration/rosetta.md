# The Rosetta glossary

> Part of [The tsecon Guide to Time Series Econometrics](../guide/README.md). One
> table, four dialects. For each core time-series concept it gives the tsecon
> function and the closest call in `statsmodels`, R, and Stata — so you can read a
> method named in any of them and find its home here.

This is a lookup table, not a tutorial. If you know a method by its name in one
package, scan the row and read across. The tsecon column lists **only functions
that ship today**; where a concept is on the library's
[roadmap](../../ROADMAP.md) it is marked *(roadmap)*. A dash (—) in another
package's column means that package has no first-class equivalent (some are
available only through user-written add-ons, noted where it matters). For the
full narrative translations, see the [statsmodels](from-statsmodels.md),
[R](from-r.md), and [Stata](from-stata.md) guides.

Every tsecon name below is a real function. The canonical idiom — arrays in,
a dict out — looks like this:

```python
import numpy as np, tsecon
rng = np.random.default_rng(20)
y = np.cumsum(rng.standard_normal(300))               # a random walk
rep = tsecon.check_stationarity(y)
print(rep["quadrant"], "->", rep["recommendation"])   # UnitRoot -> Difference
```

## Concept → package call

| Concept | tsecon | statsmodels | R | Stata |
|---|---|---|---|---|
| Autocorrelation function | `acf` | `acf` | `acf` / `forecast::Acf` | `ac`, `corrgram` |
| Partial autocorrelation | `pacf` | `pacf` | `pacf` | `pac` |
| White-noise / portmanteau test | `ljung_box` | `acorr_ljungbox` | `Box.test` | `wntestq` |
| Normality test | `jarque_bera` | `jarque_bera` | `tseries::jarque.bera.test` | `sktest` (approx.) |
| ARCH / conditional-heteroskedasticity test | `arch_lm` | `het_arch` | `FinTS::ArchTest` | `estat archlm` |
| Unit-root test (ADF) | `adf` | `adfuller` | `urca::ur.df`, `tseries::adf.test` | `dfuller` |
| Stationarity test (KPSS) | `kpss` | `kpss` | `urca::ur.kpss` | `kpss` |
| Confirmatory stationarity workflow | `check_stationarity` | — | — | — |
| Phillips-Perron test | *(roadmap)* | `PhillipsPerron` | `urca::ur.pp` | `pperron` |
| HAC / Newey-West standard errors | `ols(se_type="hac")` | `cov_type="HAC"` | `sandwich::NeweyWest` | `newey` |
| ARIMA | `arima_fit` | `ARIMA` | `forecast::Arima` | `arima` |
| Seasonal ARIMA (SARIMA) | *(roadmap)* | `SARIMAX` | `forecast::auto.arima` | `arima ...(P,D,Q)` |
| Exponential smoothing / Theta | `theta_forecast` | `ETSModel` | `forecast::ets`, `thetaf` | `tssmooth` |
| GARCH family | `garch_fit` | `arch.arch_model` | `rugarch::ugarchfit` | `arch` |
| Multivariate GARCH (CCC / DCC) | `ccc_garch`, `dcc_garch` | — | `rmgarch::dccfit` | `mgarch ccc/dcc` |
| Score-driven volatility (GAS/DCS) | `gas_volatility` | — | `GAS::UniGASFit` | — |
| VAR | `var_fit` | `VAR` | `vars::VAR` | `var` |
| Impulse response (IRF) | `var_irf` | `.irf()` | `vars::irf` | `irf create`, `irf graph` |
| Forecast-error variance decomposition | `var_fevd` | `.fevd()` | `vars::fevd` | `irf table fevd` |
| Granger causality | `var_granger` | `test_causality` | `vars::causality` | `vargranger` |
| Cointegration rank (Johansen) | `johansen` | `coint_johansen` | `urca::ca.jo` | `vecrank` |
| Vector error-correction model | `vecm` | `VECM` | `urca::cajorls`, `vars::vec2var` | `vec` |
| Bayesian VAR (Minnesota) | `bvar_fit`, `bvar_irf_draws` | — | `BVAR::bvar` | `bayes: var` |
| Sign-restricted SVAR | `sign_restricted_svar` | — | `VARsignR`, `svars` | — |
| SVAR short-/long-run restrictions | *(roadmap)* | `SVAR` | `svars::id.*` | `svar` |
| FAVAR | `favar` | — | — | — |
| Connectedness (Diebold-Yilmaz) | `connectedness` | — | `frequencyConnectedness` | — |
| Local projection | `lp` | — | `lpirfs::lp_lin` | — (user `lp`) |
| Local projection with IV (multiplier) | `lp_iv` | — | `lpirfs::lp_lin_iv` | — |
| State-dependent local projection | `lp_state` | — | `lpirfs::lp_nl` | — |
| Markov-switching model | `markov_switching_ar` | `MarkovAutoregression` | `MSwM::msmFit` | `mswitch` |
| HP filter | `hp_filter` | `hpfilter` | `mFilter::hpfilter` | `tsfilter hp` |
| Baxter-King / Christiano-Fitzgerald filter | `bk_filter`, `cf_filter` | `bkfilter`, `cffilter` | `mFilter::bkfilter`/`cffilter` | `tsfilter bk`/`cf` |
| Hamilton regression filter | `hamilton_filter` | — | `neverhpfilter::yth_filter` | — |
| Spectral density | `periodogram`, `welch`, `coherence` | `scipy.signal.*` | `spectrum`, `spec.pgram` | `psdensity`, `pergram` |
| Diebold-Mariano test | `dm_test` | — | `forecast::dm.test` | `dmariano` (user) |
| Clark-West / Giacomini-White test | `cw_test`, `gw_test` | — | `sandwich`+custom | — |
| Forecast accuracy measures | `accuracy` | — | `forecast::accuracy` | — |
| Rolling/expanding backtest | `backtest` | — | `forecast::tsCV` | `rolling:` |
| Realized variance / bipower | `realized_measures` | — | `highfrequency::rCov`, `rBPCov` | — |
| HAR-RV | `har_rv` | — | `HARModel::HARestimate` | — |
| Panel fixed effects | `panel_fe` | — | `plm(model="within")` | `xtreg, fe` |
| Driscoll-Kraay standard errors | `panel_fe(se_type="driscoll_kraay")` | — | `plm` + `vcovSCC` | `xtscc` |
| Mean-group / CCE-MG estimator | `panel_mean_group` | — | `plm::pmg(model="mg")`, `xtmg` | `xtpmg mg`, `xtmg cce` |
| Pooled mean group (PMG) | `panel_pmg` | — | `plm::pmg(model="pmg")` | `xtpmg pmg` |
| Mean-group panel VAR | `mean_group_var` | — | `panelvar` (approx.) | `pvar` (user) |
| Panel local projection | `panel_lp` | — | `lpirfs::lp_lin_panel` | — |
| Nowcast (dynamic factor model) | `dfm_nowcast` | `DynamicFactorMQ` | `nowcasting::nowcast` | `dfactor` (approx.) |
| News / update decomposition | `dfm_news` | — | `nowcasting` | — |
| Static factor model (PCA + Bai-Ng) | `factor_model` | — | — | — |
| MIDAS mixed-frequency regression | `weighted_midas`, `umidas` | — | `midasr::midas_r` | `midasreg` (user) |
| Linear IV-GMM (+ Hansen J) | `iv_gmm` | — (`linearmodels`) | `gmm::gmm`, `AER::ivreg` | `ivregress gmm` |
| Nonlinear GMM (custom moments) | `gmm_nonlinear` | — | `gmm::gmm` | `gmm` |
| Ridge / lasso / elastic net | `ridge`, `lasso`, `elastic_net` | — | `glmnet` | `lasso`, `elasticnet` |
| Adaptive lasso / penalized path | `adaptive_lasso`, `lasso_path` | — | `glmnet` (+ weights) | `lasso ..., selection()` |
| Leakage-safe time-series CV | `cv_splits` | — | `rsample::rolling_origin` | — |
| Yield curve (Nelson-Siegel / Svensson) | `nelson_siegel`, `svensson` | — | `YieldCurve::Nelson.Siegel` | — |
| Dynamic Nelson-Siegel (Diebold-Li) | `dynamic_ns` | — | `YieldCurve` | — |
| Kalman filter / smoother (local level) | `local_level_smooth` | `UnobservedComponents` | `dlm`, `KFAS` | `sspace` |
| Bootstrap resampling (block/stationary) | `bootstrap_indices`, `optimal_block_length` | — | `boot`, `np::b.star` | `bootstrap:` |

## How to read the roadmap gaps

A handful of concepts above are marked *(roadmap)* in the tsecon column —
Phillips-Perron, seasonal ARIMA, and explicit short-/long-run SVAR restrictions
are the notable ones. tsecon covers the *identification* frontier that the other
packages mostly lack (sign restrictions, BVARs, local projections, FAVAR,
nowcasting), and is still filling in some classical corners. The per-package
guides spell out each gap and the nearest shipped substitute; nothing in the
tsecon column is aspirational unless it carries the *(roadmap)* tag.
