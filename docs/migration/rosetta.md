# The Rosetta glossary

> Part of [The tsecon Guide to Time Series Econometrics](../guide/README.md). One
> table, four dialects. For each core time-series concept it gives the tsecon
> function and the closest call in `statsmodels`, R, and Stata — so you can read a
> method named in any of them and find its home here.

This is a lookup table, not a tutorial. If you know a method by its name in one
package, scan the row and read across. The tsecon column lists **only functions
that ship today**; where a concept is on the library's
[roadmap](../../ROADMAP.md) it is marked *(roadmap)* — exactly one row is, and
it is named at the bottom of this page. A dash (—) in another package's column
means that package has no first-class equivalent; where a neighbouring Python
package (`arch`, `linearmodels`, scikit-learn) or a `statsmodels` sandbox module
covers it, that is named in parentheses after the dash, and user-written add-ons
are marked "(user)". The statsmodels column is checked against **0.15.0**, which
added `LocalProjections`, `hamilton_filter` and `diebold_mariano_test`. For the
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
| One-call diagnostic battery + model routing | `check_series` | — | — | — |
| Phillips-Perron test | `phillips_perron` | `arch.unitroot.PhillipsPerron` | `urca::ur.pp` | `pperron` |
| GLS-detrended unit-root tests (DF-GLS, Ng-Perron M) | `dfgls`, `ng_perron` | `arch.unitroot.DFGLS` (no M tests) | `urca::ur.ers` (no M tests) | `dfgls` |
| Single-break unit-root test (Zivot-Andrews) | `zivot_andrews` | `zivot_andrews` | `urca::ur.za` | `zandrews` (user) |
| Panel unit-root tests (IPS / LLC / Fisher) | `panel_unit_root` | — | `plm::purtest` | `xtunitroot` |
| Heteroskedasticity test (White / Breusch-Pagan) | `heteroskedasticity_test` | `het_white`, `het_breuschpagan` | `lmtest::bptest` | `estat hettest`, `estat imtest, white` |
| Ramsey RESET functional-form test | `reset_test` | `linear_reset` | `lmtest::resettest` | `estat ovtest` |
| Chow known-break test | `chow_test` | — | `strucchange::sctest(type="Chow")` | `estat sbknown` |
| Unknown single-break sup-F (Quandt-Andrews) | `sup_f_test` | `breaks_cusumolsresid` (approx.) | `strucchange::Fstats`, `sctest(type="supF")` | `estat sbsingle` |
| Multiple structural breaks (Bai-Perron) | `bai_perron` | — | `strucchange::breakpoints` | — |
| CUSUM parameter-stability test | `cusum_test` | `breaks_cusumolsresid` | `strucchange::efp` (OLS-CUSUM) | `cusum` |
| HAC / Newey-West standard errors | `ols(se_type="hac")` | `cov_type="HAC"` | `sandwich::NeweyWest` | `newey` |
| ARIMA | `arima_fit` | `ARIMA` | `forecast::Arima` | `arima` |
| Seasonal ARIMA (SARIMA) | `arima_fit(seasonal=(P,D,Q,s))` | `SARIMAX` | `forecast::Arima(seasonal=)` | `arima ..., sarima(P,D,Q,s)` |
| Automatic ARIMA order selection | `auto_arima` | `arma_order_select_ic` (approx.) | `forecast::auto.arima` | — |
| Differencing advisors (`d`, `D`) | `ndiffs`, `nsdiffs` | — | `forecast::ndiffs`, `nsdiffs` | — |
| STL / multiple-seasonal decomposition | `stl`, `mstl`, `seasonal_strength` | `STL`, `MSTL` | `stats::stl`, `forecast::mstl` | — |
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
| Residual cointegration test (Engle-Granger / Phillips-Ouliaris) | `engle_granger`, `phillips_ouliaris` | `coint` | `tseries::po.test`, `urca::ca.po` | `egranger` (user) |
| Bayesian VAR (Minnesota) | `bvar_fit`, `bvar_irf_draws` | — | `BVAR::bvar` | `bayes: var` |
| Sign-restricted SVAR | `sign_restricted_svar` | — | `VARsignR`, `svars` | — |
| SVAR long-run (Blanchard-Quah) restrictions | `long_run_svar` | — (`SVAR` is A/B only) | `vars::BQ` | `svar ..., lreq()` |
| SVAR short-run A/B restrictions | *(roadmap)* | `SVAR(svar_type="AB")` | `vars::SVAR(Amat, Bmat)` | `svar ..., aeq() beq()` |
| Statistical SVAR identification (heteroskedasticity / non-Gaussianity) | `hetero_svar`, `nongaussian_svar` | — | `svars::id.cv`, `id.ngml` | — |
| Zero-and-sign / narrative / proxy / max-share SVAR | `zero_sign_svar`, `narrative_svar`, `proxy_svar`, `max_share_svar` | — | `VARsignR`, `svars` (partial) | — |
| FAVAR | `favar` | — | — | — |
| Connectedness (Diebold-Yilmaz) | `connectedness` | — | `frequencyConnectedness` | — |
| Local projection | `lp` | `LocalProjections` (0.15.0) | `lpirfs::lp_lin` | — (user `lp`) |
| Local projection with external IV (LP-IV) | `lp_iv` | — | `lpirfs::lp_lin_iv` | — |
| Integral multiplier (Ramey-Zubairy) | `lp_multiplier` | — | `lpirfs::lp_lin_iv` (approx.) | — |
| State-dependent local projection | `lp_state` | — | `lpirfs::lp_nl` | — |
| Smooth local projection (penalized B-spline) | `smooth_lp` | — | — | — |
| Quantile regression | `quantile_regression` | `QuantReg` | `quantreg::rq` | `qreg`, `sqreg`, `bsqreg` |
| Quantile local projection | `quantile_lp` | — | — | — |
| Growth-at-Risk (conditional quantiles) | `growth_at_risk` | — | — | — |
| Functional PCA of a curve panel | `functional_pca` | — | — | — |
| Functional shocks (FLP / FVAR) | `flp`, `flp_scenario`, `fvar_scenario` | — | — | — |
| Markov-switching model | `markov_switching_ar` | `MarkovAutoregression` | `MSwM::msmFit` | `mswitch` |
| Threshold autoregression (SETAR) + linearity test | `setar`, `setar_test` | — | `tsDyn::setar`, `setarTest` | `threshold` (regression only) |
| Smooth-transition AR (LSTAR / ESTAR) + Terasvirta cycle | `star`, `star_eval`, `star_test` | — | `tsDyn::lstar` | — |
| Threshold VAR + linearity test | `threshold_var`, `threshold_var_test` | — | `tsDyn::TVAR`, `TVAR.LRtest` | — |
| Threshold cointegration (Hansen-Seo) + sup-LM test | `threshold_vecm`, `hansen_seo_test` | — | `tsDyn::TVECM`, `TVECM.HStest` | — |
| HP filter | `hp_filter` | `hpfilter` | `mFilter::hpfilter` | `tsfilter hp` |
| Baxter-King / Christiano-Fitzgerald filter | `bk_filter`, `cf_filter` | `bkfilter`, `cffilter` | `mFilter::bkfilter`/`cffilter` | `tsfilter bk`/`cf` |
| Hamilton regression filter | `hamilton_filter` | `hamilton_filter` (0.15.0) | `neverhpfilter::yth_filter` | — |
| Spectral density | `periodogram`, `welch`, `coherence` | `scipy.signal.*` | `spectrum`, `spec.pgram` | `psdensity`, `pergram` |
| Diebold-Mariano test | `dm_test` | `diebold_mariano_test` (0.15.0) | `forecast::dm.test` | `dmariano` (user) |
| Clark-West / Giacomini-White test | `cw_test`, `gw_test` | — | `sandwich`+custom | — |
| Forecast accuracy measures | `accuracy` | — (`tools.eval_measures`: RMSE/MAE, no MASE/sMAPE) | `forecast::accuracy` | — |
| Rolling/expanding backtest | `backtest` | — | `forecast::tsCV` | `rolling:` |
| Realized variance / bipower | `realized_measures` | — | `highfrequency::rCov`, `rBPCov` | — |
| HAR-RV | `har_rv` | — | `HARModel::HARestimate` | — |
| Panel fixed effects | `panel_fe` | — (`linearmodels.PanelOLS`) | `plm(model="within")` | `xtreg, fe` |
| Driscoll-Kraay standard errors | `panel_fe(se_type="driscoll_kraay")` | — (`PanelOLS(cov_type="driscoll-kraay")`) | `plm` + `vcovSCC` | `xtscc` |
| Mean-group / CCE-MG estimator | `panel_mean_group` | — | `plm::pmg(model="mg")`, `xtmg` | `xtpmg mg`, `xtmg cce` |
| Pooled mean group (PMG) | `panel_pmg` | — | `plm::pmg(model="pmg")` | `xtpmg pmg` |
| Mean-group panel VAR | `mean_group_var` | — | `panelvar` (approx.) | `pvar` (user) |
| Panel local projection | `panel_lp` | — | `lpirfs::lp_lin_panel` | — |
| Nowcast (dynamic factor model) | `dfm_nowcast` | `DynamicFactorMQ` | `nowcasting::nowcast` | `dfactor` (approx.) |
| News / update decomposition | `dfm_news` | — | `nowcasting` | — |
| Static factor model (PCA + Bai-Ng) | `factor_model` | — | — | — |
| MIDAS mixed-frequency regression | `weighted_midas`, `umidas` | — | `midasr::midas_r` | `midasreg` (user) |
| Linear IV-GMM (+ Hansen J) | `iv_gmm` | — (`linearmodels`) | `gmm::gmm`, `AER::ivreg` | `ivregress gmm` |
| Nonlinear GMM (custom moments) | `gmm_nonlinear` | — (`sandbox.regression.gmm.GMM`) | `gmm::gmm` | `gmm` |
| Ridge / lasso / elastic net | `ridge`, `lasso`, `elastic_net` | `OLS.fit_regularized` | `glmnet` | `lasso`, `elasticnet` |
| Adaptive lasso / penalized path | `adaptive_lasso`, `lasso_path` | — (`sklearn.linear_model.lasso_path`) | `glmnet` (+ weights) | `lasso ..., selection()` |
| Leakage-safe time-series CV | `cv_splits` | — | `rsample::rolling_origin` | — |
| Yield curve (Nelson-Siegel / Svensson) | `nelson_siegel`, `svensson` | — | `YieldCurve::Nelson.Siegel` | — |
| Dynamic Nelson-Siegel (Diebold-Li) | `dynamic_ns` | — | `YieldCurve` | — |
| Arbitrage-free Nelson-Siegel (AFNS) | `afns_adjustment` | — | — | — |
| Kalman filter / smoother (local level) | `local_level_smooth` | `UnobservedComponents` | `dlm`, `KFAS` | `sspace` |
| Linear RE / DSGE solution (Blanchard-Kahn) | `dsge_solve` | — | `gEcon` (approx.) | `dsge` |
| Bootstrap resampling (block/stationary) | `bootstrap_indices`, `optimal_block_length` | — (`arch.bootstrap.StationaryBootstrap`, `optimal_block_length`) | `boot`, `np::b.star` | `bootstrap:` |

## How to read the roadmap gaps

Exactly one row above carries *(roadmap)*: explicit short-run **A/B SVAR
restrictions**, the one identification scheme on this page tsecon does not
implement. Everything else in the tsecon column is a function you can call
today, including the rows that used to be tagged — Phillips-Perron is
`phillips_perron`, seasonal ARIMA is `arima_fit(seasonal=(P, D, Q, s))`, and
long-run Blanchard-Quah restrictions are `long_run_svar`.

Two cautions on how to read the rest of the table. First, a dash is a claim
about *that* package, and packages move: `statsmodels` 0.15.0 added
`LocalProjections`, `hamilton_filter` and `diebold_mariano_test`, so those three
cells are filled where they used to be empty. Second, a filled cell says the two
calls target the same estimand — not that tsecon was validated against that
call. The threshold and smooth-transition rows are the sharpest example: the R
package they name, `tsDyn`, could not be installed in the build container, so
those four rows have **no reference run behind them** and rest on transcribed
closed forms pinned to an independent NumPy implementation plus seeded
Monte-Carlo evidence. The [validation matrix](../reference/validation-matrix.md)
grades every function this way, and the per-package guides spell out each gap
and the nearest shipped substitute.
