# Validation matrix

Correctness is the contract. Nothing lands in `tsecon` without a **golden
fixture** it has to reproduce to a stated tolerance — a JSON file of reference
values (in [`fixtures/`](../../fixtures/README.md)) that the Rust crate tests,
and in most cases the Python binding tests, must hit on every run. This page is
the map: for each method family it names the *reference* the golden is measured
against, the fixture file, the test that enforces the match, and the tolerance
it is held to.

The reference is honest about its own strength. There are three kinds, and the
table says which one each row is:

- **Independent package.** The golden values come from a mature, independently
  written library — statsmodels, SciPy, `arch`, `linearmodels`, scikit-learn,
  or ArviZ — computing the *same* estimand through a completely separate code
  path. Reproducing it is a genuine cross-implementation check.
- **Documented-formula golden.** No package computes the quantity, so the
  generator transcribes the *published closed form* directly into NumPy (the
  formula is written out in the generator's docstring) and pins the crate to it.
  This proves the Rust reproduces the documented algebra; it does **not** by
  itself prove the algebra is the statistically right choice — that claim is
  carried by the crate's seeded Monte-Carlo *property* tests, noted where it
  applies. A few of these are *cross-implementation* goldens (an independent
  NumPy re-implementation of the same estimator, not an independent authority);
  the table flags those explicitly.
- **Property / simulation-recovery.** Where no reference of either kind exists
  (multivariate DCC dynamics, dynamic probit), the object is validated by
  invariants it must satisfy — positive-definiteness, correlation targeting,
  parameter recovery on simulated data within Monte-Carlo bands — rather than a
  golden. The table says so plainly rather than overclaiming a package match.

Tolerances below are the **asserted** bounds in the test source. Many are far
tighter than the crate spec requires and the achieved agreement is tighter
still (frequently machine precision); where the two differ the table quotes the
asserted bound.

---

## Estimator families

| Family | Validated against | Fixture | Test | Tolerance |
|---|---|---|---|---|
| Diagnostics — `acf`, `pacf`, `ljung_box`, `jarque_bera`, `arch_lm` | statsmodels (`acf`, `pacf`, `acorr_ljungbox`, `het_arch`) + `scipy.stats` — independent package | [`diagnostics.json`](../../fixtures/diagnostics.json) | [`tsecon-diag/…/golden.rs`](../../crates/tsecon-diag/tests/golden.rs) | 1e-12 rel (spec floor 1e-8) |
| Unit root & stationarity — `adf`, `kpss`, `check_stationarity` | statsmodels `adfuller`, `kpss` + MacKinnon p-value grid — independent package | [`unitroot.json`](../../fixtures/unitroot.json) | [`tsecon-diag/…/unitroot_golden.rs`](../../crates/tsecon-diag/tests/unitroot_golden.rs) | ADF stat 1e-8 / p 1e-7; KPSS stat 1e-8 / p 1e-6 |
| Volatility (GARCH) — `garch_fit` | `arch` (Sheppard) `arch_model` — independent package | [`garch.json`](../../fixtures/garch.json) | [`tsecon-garch/…/golden.rs`](../../crates/tsecon-garch/tests/golden.rs) | fixed-param loglike 1e-8; QMLE params 1e-3, loglike ≥ `arch` − 1e-6 |
| Score-driven volatility (GAS/DCS) — `gas_volatility` | [documented closed form](../../fixtures/generate_tsecon-gas_fixtures.py) (Creal-Koopman-Lucas 2013), NumPy | [`tsecon-gas.json`](../../fixtures/tsecon-gas.json) | [`tsecon-gas/…/golden.rs`](../../crates/tsecon-gas/tests/golden.rs) | 1e-10 |
| Multivariate GARCH — `ccc_garch`, `dcc_garch` | **no external DCC reference**: univariate stage `arch`-pinned (via tsecon-garch); DCC dynamics property-validated (every `R_t`/`H_t` PD, correlation targeting) + single-realization recovery | [`mgarch.json`](../../fixtures/mgarch.json) (simulated DCC-GARCH(1,1), true params attached) | [`tsecon-mgarch/…/mgarch.rs`](../../crates/tsecon-mgarch/tests/mgarch.rs) | property bounds (PD 1e-8…1e-14); loose MC recovery |
| VAR / SVAR — `var_fit`, `var_irf`, `var_fevd`, `var_granger`, `var_forecast` | statsmodels `VAR` — independent package | [`var.json`](../../fixtures/var.json) | [`tsecon-var/…/golden.rs`](../../crates/tsecon-var/tests/golden.rs) | 1e-8 (params, `sigma_u`, IRF, FEVD, forecast); Granger p 1e-6 |
| FAVAR / factor extraction — `favar` | [NumPy SVD / PCA](../../fixtures/generate_depth_fixtures.py) — documented | [`favar.json`](../../fixtures/favar.json) | [`tsecon-favar/…/golden.rs`](../../crates/tsecon-favar/tests/golden.rs) | 1e-6 (eigenvalues, \|PC\|, \|loadings\|) |
| Connectedness (Diebold-Yilmaz) — `connectedness` | [self-authored GFEVD](../../fixtures/generate_depth_fixtures.py) (Diebold-Yilmaz 2012), documented from a VAR | [`connect.json`](../../fixtures/connect.json) | [`tsecon-connect/…/golden.rs`](../../crates/tsecon-connect/tests/golden.rs) | GFEVD matrix 1e-8; to/from/total 1e-6 |
| Local projections — `lp`, `lp_iv`, `lp_state` | statsmodels OLS + Newey-West HAC and linearmodels `IV2SLS` (kernel-HAC) — independent package | [`lp.json`](../../fixtures/lp.json) | [`tsecon-lp/…/golden.rs`](../../crates/tsecon-lp/tests/golden.rs) | OLS β 1e-10 / HAC se 1e-8; IV β 1e-8 / se 1e-6 |
| Bayesian BVAR (conjugate NIW) — `bvar_fit`, `bvar_irf_draws` | [documented closed-form conjugate NIW posterior](../../fixtures/generate_bayes_fixtures.py) (NumPy / SciPy `multigammaln`) | [`bvar_niw.json`](../../fixtures/bvar_niw.json) | [`tsecon-bayes/…/golden.rs`](../../crates/tsecon-bayes/tests/golden.rs) | 1e-9 (posterior moments, log-marginal-likelihood) |
| MCMC diagnostics — `mcmc_diagnostics` | ArviZ (`rhat`, `ess_bulk`, `ess_tail`) — independent package | [`convergence.json`](../../fixtures/convergence.json) | [`tsecon-bayes/…/golden.rs`](../../crates/tsecon-bayes/tests/golden.rs) | 1e-9 |
| GMM / IV-GMM — `iv_gmm`, `gmm_nonlinear` | linearmodels `IVGMM` (2-step efficient, robust) — independent package | [`gmm.json`](../../fixtures/gmm.json) | [`tsecon-gmm/…/golden.rs`](../../crates/tsecon-gmm/tests/golden.rs) | params 1e-9; bse 1e-6; Hansen J & p 1e-6 |
| Cointegration — `johansen`, `vecm` | statsmodels `coint_johansen`, `VECM` — independent package | [`coint.json`](../../fixtures/coint.json) | [`tsecon-coint/…/golden.rs`](../../crates/tsecon-coint/tests/golden.rs) | eigenvalues 1e-8; trace/max-eig LR 1e-6; VECM α/β/Γ/llf 1e-6 |
| Markov-switching AR — `markov_switching_ar` | statsmodels `MarkovAutoregression` — independent package | [`regime.json`](../../fixtures/regime.json) | [`tsecon-regime/…/golden.rs`](../../crates/tsecon-regime/tests/golden.rs) | 1e-6 (fixed-param loglike, filtered / smoothed probs) |
| Forecasting metrics & tests — `backtest`, `dm_test`, `cw_test`, `gw_test`, `theta_forecast`, `accuracy` | [documented hand-computed metrics](../../fixtures/generate_phase2_fixtures.py) + statsmodels `ThetaModel` + self-authored CW/GW | [`forecast.json`](../../fixtures/forecast.json), [`forecast_eval2.json`](../../fixtures/forecast_eval2.json) | [`tsecon-forecast/…/golden.rs`](../../crates/tsecon-forecast/tests/golden.rs) | metrics 1e-14; theta 1e-6; DM / GW 1e-10 |
| Machine learning — `ridge`, `lasso`, `elastic_net`, `adaptive_lasso`, `lasso_path` | scikit-learn `Ridge`, `Lasso`, `ElasticNet` — independent package | [`ml.json`](../../fixtures/ml.json) | [`tsecon-ml/…/golden.rs`](../../crates/tsecon-ml/tests/golden.rs) | 1e-6 (achieved ~1e-9) |
| Panel FE / panel LP — `panel_fe`, `panel_lp` | linearmodels `PanelOLS` (clustered, Driscoll-Kraay, nonrobust) — independent package | [`panel.json`](../../fixtures/panel.json) | [`tsecon-panel/…/golden.rs`](../../crates/tsecon-panel/tests/golden.rs) | 1e-6 (slopes, se, R²) |
| Heterogeneous panel MG / CCE-MG — `mean_group_var`, `panel_mean_group` | statsmodels OLS per-unit (independent) + documented MG / CCE averaging (Pesaran-Smith 1995 / Pesaran 2006) | [`tsecon-panelts.json`](../../fixtures/tsecon-panelts.json) | [`tsecon-panelts/…/golden.rs`](../../crates/tsecon-panelts/tests/golden.rs) | 1e-10 (coef, se, tstat, per-unit slopes) |
| Pooled mean group (PMG) — `panel_pmg` | [documented-formula **cross-implementation**](../../fixtures/generate_pmg_fixtures.py): independent NumPy re-impl of PSS 1999 — same estimator, different numerical path, **not** an independent authority | [`pmg.json`](../../fixtures/pmg.json) | [`tsecon-panelts/…/pmg_golden.rs`](../../crates/tsecon-panelts/tests/pmg_golden.rs) | θ, φ̄, se 1e-8; loglik 1e-6 |
| Nowcasting DFM (two-step Kalman) — `dfm_nowcast` | statsmodels `DynamicFactor` (Kalman step at fixed params) — independent package; the DGR two-step *estimates* are property-only | [`tsecon-nowcast.json`](../../fixtures/tsecon-nowcast.json) | [`tsecon-nowcast/…/golden.rs`](../../crates/tsecon-nowcast/tests/golden.rs) | 1e-8 (llf, smoothed states) |
| Nowcasting DFM one-step MLE — `dfm_nowcast` (MLE path) | statsmodels `DynamicFactor` fitted (exact-likelihood optimum) — independent package | [`nowcast_mle.json`](../../fixtures/nowcast_mle.json) | [`tsecon-nowcast/…/mle.rs`](../../crates/tsecon-nowcast/tests/mle.rs) | smooth-at-fitted 1e-6; optimiser gap honest ≤ 1e-2 rel |
| Nowcast news decomposition — `dfm_news` | [independent NumPy Kalman + RTS smoother](../../fixtures/generate_nowcast_news_fixtures.py) (Banbura-Modugno 2014) — a different implementation | [`nowcast_news.json`](../../fixtures/nowcast_news.json) | [`tsecon-nowcast/…/news.rs`](../../crates/tsecon-nowcast/tests/news.rs) | weights 1e-6; forecasts / news 1e-7; actuals 1e-9 |
| MIDAS — `midas_weights`, `umidas`, `weighted_midas` | statsmodels OLS (U-MIDAS) + [documented weight formulas](../../fixtures/generate_phase34_fixtures.py) (exp-Almon, Beta) | [`midas.json`](../../fixtures/midas.json) | [`tsecon-midas/…/golden.rs`](../../crates/tsecon-midas/tests/golden.rs) | weights 1e-10; U-MIDAS params / bse / R² 1e-8 |
| Term structure (NS / dynamic NS) — `nelson_siegel`, `dynamic_ns` | statsmodels OLS on Nelson-Siegel loadings at Diebold-Li (2006) fixed λ; Svensson validated by nesting property | [`termstructure.json`](../../fixtures/termstructure.json) | [`tsecon-termstructure/…/golden.rs`](../../crates/tsecon-termstructure/tests/golden.rs) | loadings 1e-10; factors / R² 1e-8 |
| Arbitrage-free Nelson-Siegel — `afns_adjustment` | [documented closed-form yield-adjustment term](../../fixtures/generate_afns_fixtures.py) (Christensen-Diebold-Rudebusch 2011), NumPy | [`afns.json`](../../fixtures/afns.json) | [`tsecon-termstructure/…/afns.rs`](../../crates/tsecon-termstructure/tests/afns.rs) | 1e-10 |
| Realized volatility — `realized_measures`, `har_rv`, `realized_quarticity`, `tripower_quarticity`, `bns_jump_test`, `realized_range` | statsmodels OLS (HAR-RV, Corsi 2009) + [documented measures](../../fixtures/generate_depth_fixtures.py) (RV / BV / quarticity, Barndorff-Nielsen-Shephard) | [`realized.json`](../../fixtures/realized.json) | [`tsecon-realized/…/golden.rs`](../../crates/tsecon-realized/tests/golden.rs) | RV / BV 1e-12; HAR params / bse / R² 1e-8 |
| Predictive regressions & IVX — `predictive_regression`, `ivx_test` | [documented closed form](../../fixtures/generate_predreg_fixtures.py) (Stambaugh 1999 / Kostakis-Magdalinos-Stamatogiannis 2015), NumPy; size / power are property tests | [`predreg.json`](../../fixtures/predreg.json) | [`tsecon-predreg/…/golden.rs`](../../crates/tsecon-predreg/tests/golden.rs) | slopes / Wald 1e-9; p-value 1e-8 |
| Recession probability — `recession_probit` | statsmodels `Probit` / `Logit` (static); the dynamic Kauppi-Saikkonen model has no reference → property-only | [`tsecon-recession.json`](../../fixtures/tsecon-recession.json) | [`tsecon-recession/…/golden.rs`](../../crates/tsecon-recession/tests/golden.rs) | 1e-6 |
| Survey expectations — `cg_regression`, `forecast_efficiency`, `forecast_disagreement` | statsmodels OLS + Newey-West HAC + NumPy (`std`, percentiles) + [documented closed forms](../../fixtures/generate_survey_fixtures.py) (implied rigidity, IQR) | [`tsecon-survey.json`](../../fixtures/tsecon-survey.json) | [`tsecon-survey/…/golden.rs`](../../crates/tsecon-survey/tests/golden.rs) | 1e-8 |
| Long memory — `frac_diff`, `frac_integrate`, `long_memory_d` | [documented closed form](../../fixtures/generate_longmemory_fixtures.py) (binomial (1−L)ᵈ; GPH 1983; Robinson 1995 local Whittle), NumPy; recovery is a property test | [`longmemory.json`](../../fixtures/longmemory.json) | [`tsecon-longmemory/…/golden.rs`](../../crates/tsecon-longmemory/tests/golden.rs) | frac diff / int 1e-12; GPH d 1e-8, se 1e-12; Whittle d 1e-6 |
| Specification & diagnostic tests — `heteroskedasticity_test`, `reset_test`, `chow_test`, `cusum_test` | statsmodels `het_white`, `het_breuschpagan` (Koenker), `linear_reset` + [documented Chow / CUSUM](../../fixtures/generate_tsecon-spectest_fixtures.py) | [`tsecon-spectest.json`](../../fixtures/tsecon-spectest.json) | [`tsecon-spectest/…/golden.rs`](../../crates/tsecon-spectest/tests/golden.rs) | 1e-8 |
| DSGE (linear RE solver) — `dsge_solve` | [documented closed-form Blanchard-Kahn solution](../../fixtures/generate_tsecon-dsge_fixtures.py) (NumPy; eigenvalues independently cross-checked via `numpy.linalg.eigvals`) | [`tsecon-dsge.json`](../../fixtures/tsecon-dsge.json) | [`tsecon-dsge/…/golden.rs`](../../crates/tsecon-dsge/tests/golden.rs) | 1e-8 |

## Foundational numerics

The primitives every estimator above leans on are held to the same standard.

| Family | Validated against | Fixture | Test | Tolerance |
|---|---|---|---|---|
| ARIMA / SARIMAX | statsmodels `SARIMAX` (fixed-param loglike, forecast); MLE optimum independently cross-verified — independent package | [`arima.json`](../../fixtures/arima.json) | [`tsecon-arima/…/golden.rs`](../../crates/tsecon-arima/tests/golden.rs) | loglike 1e-8; forecast 1e-6; optimum params 1e-4 |
| State-space / Kalman filter & smoother | statsmodels statespace / `SARIMAX` with exact-diffuse initialization — independent package | [`ssm.json`](../../fixtures/ssm.json) | [`tsecon-ssm/…/golden.rs`](../../crates/tsecon-ssm/tests/golden.rs) | 1e-6 (achieved ≤ 1e-11) |
| Filters — HP / Baxter-King / Hamilton | statsmodels `hpfilter`, `bkfilter` + [documented Hamilton (2018) regression filter](../../fixtures/generate_fixtures.py) | [`filters.json`](../../fixtures/filters.json) | [`tsecon-filters/…/golden.rs`](../../crates/tsecon-filters/tests/golden.rs) | 1e-8 |
| HAC / long-run variance — Newey-West, EWC | statsmodels OLS with HAC covariance — independent package | [`hac.json`](../../fixtures/hac.json) | [`tsecon-hac/…/golden.rs`](../../crates/tsecon-hac/tests/golden.rs) | 1e-10 |
| Spectral analysis — periodogram / Welch / coherence | `scipy.signal` (`periodogram`, `welch`, `coherence`) — independent package | [`spectral.json`](../../fixtures/spectral.json) | [`tsecon-spectral/…/golden.rs`](../../crates/tsecon-spectral/tests/golden.rs) | 1e-8 |
| Distributions & special functions | `scipy.stats` (normal, Student-t, GED, …) — independent package | [`distributions.json`](../../fixtures/distributions.json) | [`tsecon-stats/…/golden.rs`](../../crates/tsecon-stats/tests/golden.rs) | pdf / logpdf / cdf 1e-12; ppf 1e-9 |
| Linear algebra — Toeplitz solve / discrete Lyapunov / Levinson-Durbin | `scipy.linalg` (`solve_toeplitz`, `solve_discrete_lyapunov`) + statsmodels `levinson_durbin` — independent package | [`linalg.json`](../../fixtures/linalg.json) | [`tsecon-linalg/…/golden.rs`](../../crates/tsecon-linalg/tests/golden.rs) | 1e-10 (Levinson-Durbin 1e-12) |
| RNG — Philox counter-based generator | NumPy `Philox` bit-stream — independent package | [`philox.json`](../../fixtures/philox.json) | [`tsecon-rng/…/golden.rs`](../../crates/tsecon-rng/tests/golden.rs) | bit-exact |

---

## Provenance

Each fixture records, in its `_meta` block, the exact reference-library versions
used to produce it, so the values are reproducible. The pinned versions across
the suite are:

| Reference | Version |
|---|---|
| statsmodels | 0.14.6 |
| SciPy | 1.17.1 |
| NumPy | 2.5.1 |
| arch | 8.0.0 |
| linearmodels | 7.0 |
| scikit-learn | 1.9.0 |
| ArviZ | 1.2.0 |
| Python | 3.12.7 |

The goldens gate the **Rust** crate tests directly. They are additionally
exercised through the **Python** API — the binding suite in
[`bindings/python/tests/`](../../bindings/python/tests) reloads the same JSON
fixtures and checks the shipped module reproduces them — so the guarantee holds
end-to-end, not just in the core. The fixtures themselves store only derived
numeric values and transformations of two public-domain reference series (the
Nile river-flow series and US macrodata); no licensed dataset is redistributed.
See the [fixtures README](../../fixtures/README.md) for how each file is
generated and regenerated.
