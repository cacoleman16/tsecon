# Module 01 — Diagnostics, Data Exploration, Filters, and Seasonal Adjustment

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**This module is the base layer of the library: the correlation and spectral diagnostics, hypothesis-test batteries (whiteness, normality, nonlinearity, unit roots, seasonal unit roots, cointegration, structural breaks), trend-cycle filters, outlier and missing-data machinery, and seasonal adjustment tooling that every downstream model consumes and every user touches first. Its mandate is twofold: implement each statistic to reference-implementation accuracy with correct nonstandard critical values, and — uniquely among existing libraries — codify the workflows (ADF+KPSS confirmatory logic, filter choice after Hamilton 2018, one-call `check_series()` batteries) that currently live only in textbooks and blog posts.**

## Purpose and scope

This module covers everything a practitioner does to a series before and after fitting a model: exploratory second-moment analysis (ACF/PACF/CCF, periodograms, wavelets), the full inferential test battery (portmanteau and LM residual tests, nonlinearity screens, the unit-root and stationarity families with break-robust variants, seasonal unit roots, cointegration tests, structural-break estimation and inference), trend-cycle extraction (HP, Hamilton, band-pass, Beveridge-Nelson), long-memory estimation, outlier detection, missing-value imputation, variance-stabilizing transformations, and seasonal adjustment from classical decomposition through STL/MSTL to a production-grade X-13ARIMA-SEATS wrapper. It also owns the workflow layer: differencing advisors, the unit-root decision tree, and the flagship `check_series()` diagnostic report.

Its users span the library's whole audience: applied macroeconomists screening data before VAR or local-projection work, central-bank and statistical-office staff running seasonal adjustment and temporal-benchmarking pipelines, financial econometricians testing for ARCH effects, long memory, and explosive bubbles, and industry forecasters triaging millions of series with feature extraction and anomaly screens. Because nearly every test here has a nonstandard null distribution, the module is the largest customer of the foundations critical-value engine, HAC/long-run-variance library, and bootstrap suite — its requirements effectively specify those components.

Relative to the rest of the roadmap, this module is upstream of everything: univariate and multivariate modeling consume its residual diagnostics and differencing advisors; forecasting consumes its decompositions and seasonality tests; the nowcasting and identification modules consume its break and stability tests. It deliberately excludes model estimation itself, forecast evaluation (owned by forecasting-evaluation), causality testing (owned by multivariate, re-exported here), and the shared statistical infrastructure now owned by foundations. Panel unit-root tests are housed here provisionally and will seed a future panel-time-series extension module.

## Where existing tools fall short

- **Seasonal unit roots are a dead zone in Python**: statsmodels has no HEGY, Canova-Hansen, or OCSB; users must go to R's `uroot` or Gretl, and even uroot's monthly response surfaces are little-known.
- **No credible Python implementation of Bai-Perron with full inference** (sequential sup-F, break-date confidence intervals): `ruptures` does detection without econometric asymptotics; R `strucchange` is slow, single-threaded, and awkward for HAC-robust variants.
- **X-13ARIMA-SEATS access is primitive everywhere**: every ecosystem shells out to the Census `x13as` binary (statsmodels `x13`, R `seasonal`) or needs a JVM (JDemetra+); nobody exposes programmatic, typed access to M/Q statistics, sliding spans, and revision diagnostics — let alone applies those diagnostics to STL/MSTL output.
- **Critical values are sparse hard-coded tables with silent interpolation**: response surfaces beyond ADF (Hansen 1997 sup-Wald, Díaz-Emparanza HEGY, MacKinnon-Haug-Michelis Johansen) each exist in at most one tool, and none offer on-demand simulation of exact critical values.
- **ARDL bounds testing with finite-sample and surface-response p-values (Kripfganz-Schneider) is Stata-only**; statsmodels' `bounds_test` covers only asymptotic cases and leaves degenerate-case checks to the user.
- **Long-memory tooling in Python is essentially absent**: no local Whittle, no exact local Whittle, no Qu spurious-long-memory test; R's `longmemo`/`fracdiff` are dated and `LongMemoryTS` is niche.
- **Break-robust unit root tests beyond Zivot-Andrews** (Lee-Strazicich, Lumsdaine-Papell, Carrion-i-Silvestre et al., Perron-Yabu) live only in GAUSS programs, EViews add-ins, or unmaintained Stata ados — a reproducibility problem this library can own.
- **Bootstrap inference is not built in where asymptotics are known to be poor** (Hansen threshold test, wild-bootstrap variance ratios, GSADF, small-sample BDS); the R packages that do it are slow, which discourages proper replication counts.
- **No library ships an integrated "check my series" battery or codified decision workflows** (ADF+KPSS confirmatory logic, filter-choice guidance post-Hamilton 2018); tests exist, workflows don't.
- **HAC/LRV code is duplicated inconsistently** even within statsmodels (different bandwidth defaults in KPSS vs regression SEs), and no mainstream tool has adopted Lazarus-Lewis-Stock-Watson fixed-b/EWC recommendations as usable defaults.
- **Multiple/complex seasonality is fragmented**: MSTL only recently in statsmodels, STR in exactly one R package, seasonality tests (QS/Friedman/WO) only in R `seastests`; nothing unifies detection → decomposition → quality diagnostics.
- **Monte Carlo performance**: neither statsmodels nor base R packages are designed for 10^5-replication simulation of these tests; interpreter overhead dominates — exactly the niche a Rust core with zero-copy Python bindings fills.
- **Missing-value policies are inconsistent within every ecosystem** (some functions error, some silently drop, some interpolate), changing test distributions without warning; a uniform policy object is an easy differentiator.
- **Serious outlier detection is R-only**: Chen-Liu exists only in R `tsoutliers` (with results sensitive to undocumented iteration details) and inside X-13; Python has nothing credible for AO/LS/TC classification.

## Inventory

Source priorities map directly to tiers: core → Tier 1, standard → Tier 2, advanced → Tier 3, frontier → Tier 4. Items reassigned by the master-plan ownership map (causality tests; HAC/LRV, bootstrap, critical-value, calendar, and temporal-disaggregation infrastructure) appear under Dependencies, not here.

### Tier 1 — Core (v1-blocking)

#### Correlation diagnostics

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| ACF with white-noise and Bartlett confidence bands | Sample autocorrelation with both iid bands (±1.96/√T) and Bartlett MA(q) bands; the first plot every user makes, drives ARMA identification. | Low | Biased denominator T (guarantees PSD, matches R/statsmodels); Bartlett band var(r_k)≈(1+2Σ r_j²)/T holds under an MA(k−1) null, not white noise — label both band types. FFT for long series; explicit missing-value policy. Bartlett 1946; Brockwell-Davis 1991. Validate vs R `acf()` and statsmodels `acf(bartlett_confint=…)`. |
| PACF (Durbin-Levinson, Yule-Walker, OLS, Burg variants) | Partial autocorrelation for AR order identification; estimators differ in small samples. | Low | Durbin-Levinson default; "yw adjusted" (unbiased denominator) can yield pacf values exceeding 1 — document the choice (statsmodels changed its default over this). Box-Jenkins 1976. Validate vs R `pacf()` and statsmodels `pacf(method=…)`. |
| CCF with prewhitening and Haugh independence test | Cross-correlation of two series; prewhitening (fit AR to x, filter both) avoids spurious cross-correlation; Haugh (1976) portmanteau tests independence. | Medium | Naive CCF between autocorrelated series misleads users constantly — make prewhitening the documented default workflow. Box-Jenkins 1976 ch. 11; Haugh 1976. Validate vs R `ccf()` + `TSA::prewhiten()`. |

#### Whiteness tests

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Ljung-Box / Box-Pierce portmanteau | The default residual whiteness test at multiple horizons. | Low | df must be h−p−q on ARMA residuals — expose fitted-model awareness in the API, keep raw and residual modes distinct. Small-sample size issues at large h/T; invalid on GARCH-standardized residuals (route to Li-Mak). Ljung-Box 1978. Validate vs R `Box.test()` and statsmodels `acorr_ljungbox`. |
| Breusch-Godfrey LM serial correlation test | Regression-based serial correlation test; valid with lagged dependent variables, unlike Durbin-Watson. | Low | Auxiliary regression of residuals on regressors + lagged residuals; T·R² ~ χ²(p) and F variants. Breusch 1978; Godfrey 1978. Validate vs `lmtest::bgtest`. |

#### Residual diagnostics

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Jarque-Bera normality test (+ Urzúa correction, MC p-values) | Skewness/kurtosis normality test; the asymptotic χ²(2) approximation is famously poor in small samples. | Low | Offer simulated small-sample p-values (fast in Rust) and the Urzúa 1996 ALM correction. Jarque-Bera 1980. Validate vs `tseries::jarque.bera.test` and the R moments package. |
| ARCH-LM test (Engle) | LM test for conditional heteroskedasticity; the gateway diagnostic before GARCH modeling. | Low | T·R² from regressing squared residuals on own lags. Document invalidity on standardized GARCH residuals (use Li-Mak). Engle 1982. Validate vs `FinTS::ArchTest` and statsmodels `het_arch`. |

#### Nonlinearity and randomness screens

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| BDS independence test | Correlation-integral test of iid; the workhorse screen for neglected nonlinearity on residuals. | High | O(T²) pair counting — bit-parallel Grassberger-Procaccia with SIMD; epsilon grid 0.5–2×sd, embedding dims 2–8. Asymptotic p-values unreliable for T<500: ship bootstrap p-values. Nuisance-parameter caveat on fitted residuals (fine for ARMA, not GARCH). Brock-Dechert-Scheinkman-LeBaron 1996. Validate vs `tseries::bds.test` and the tables in Brock et al 1991. |
| Runs test (Wald-Wolfowitz) and turning-point tests | Nonparametric randomness screens on signs/median crossings. | Low | Exact small-sample distribution plus normal approximation. Wald-Wolfowitz 1940. Validate vs the R randtests package. |

#### Unit root and stationarity tests

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| ADF test with automatic lag selection (AIC/BIC/t-sig/MAIC) | The unit root test; lag selection and deterministic-case handling are the whole game. | Medium | MAIC of Ng-Perron 2001 on GLS-detrended data per Perron-Qu 2007 (OLS-detrended MAIC improves power); p-values from MacKinnon 1996/2010 response surfaces, never DF tables; cases none/constant/constant+trend with a workflow helper. Validate vs `urca::ur.df`, arch `ADF`, and MacKinnon's published surface coefficients. |
| Phillips-Perron test (Z-alpha, Z-tau) | Semiparametric unit root test using HAC correction instead of lag augmentation. | Medium | Severe size distortion with negative MA errors — docs must steer users to DF-GLS/Ng-Perron in that case. LRV kernel/bandwidth exposed. Phillips-Perron 1988. Validate vs `urca::ur.pp` and arch `PhillipsPerron`. |
| KPSS stationarity test | Stationarity-null complement to ADF; basis of confirmatory joint workflows. | Medium | Extremely bandwidth-sensitive: expose Newey-West/Andrews/fixed choices and print the bandwidth used. P-values exist only as a sparse table (0.01–0.10): interpolate but warn outside range (statsmodels gets this right). Kwiatkowski et al 1992. Validate vs `urca::ur.kpss`, statsmodels `kpss`. |
| DF-GLS (Elliott-Rothenberg-Stock) | GLS-detrended ADF with near-optimal local power; the recommended default over plain ADF. | Medium | Quasi-differencing at c̄=−7 (constant) / −13.5 (trend); ERS critical values with MacKinnon-style finite-sample surfaces. Elliott-Rothenberg-Stock 1996. Validate vs `urca::ur.ers`, arch `DFGLS`, Stata `dfgls`. |

#### Seasonal unit roots

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| HEGY seasonal unit root test (quarterly/monthly/general s) | Tests unit roots at zero and each seasonal frequency separately; decides between seasonal differencing and deterministic seasonality. | High | Quarterly original plus Beaulieu-Miron/Franses monthly extension and general-s formulation; t and F statistics per frequency; deterministic cases change all critical values — use Díaz-Emparanza 2014 response surfaces or built-in simulation. No maintained Python implementation exists — headline gap. Hylleberg-Engle-Granger-Yoo 1990. Validate vs `uroot::hegy.test` and Gretl. |

#### Cointegration tests

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Engle-Granger two-step cointegration test | Residual-based single-equation test; the pedagogical and practical baseline. | Medium | Critical values depend on the number of I(1) regressors and deterministics — MacKinnon 2010 response surfaces. Document normalization dependence (which variable is dependent). Engle-Granger 1987. Validate vs statsmodels `coint` and urca. |
| Phillips-Ouliaris tests (Zt, Za, Pu, Pz) | Residual and variance-ratio cointegration tests robust to endogeneity; Pz is invariant to normalization. | Medium | HAC LRV inside; critical values from PO tables/surfaces. Phillips-Ouliaris 1990. Validate vs `urca::ca.po` and `tseries::po.test`. |
| Johansen trace and max-eigenvalue tests | System ML cointegration rank determination; the standard multivariate procedure. | High | Reduced-rank regression via Cholesky whitening + SVD — never explicit inversion. Five deterministic cases each with own p-value surfaces (MacKinnon-Haug-Michelis 1999); offer the Johansen 2002 Bartlett small-sample correction. Johansen 1991, 1995. Validate vs `urca::ca.jo`, Stata `vecrank`, and the Johansen-Juselius 1990 Danish money demand results. |
| ARDL bounds test (Pesaran-Shin-Smith) with surface p-values | Level-relationship test valid for mixed I(0)/I(1) regressors; enormously popular in applied work. | High | F and t bounds; asymptotic PSS 2001 CVs, Narayan 2005 finite-sample CVs, and Kripfganz-Schneider 2020 response-surface p-values (currently Stata-only — implementing these in open source is a coup). Check degenerate cases #1/#2 explicitly and warn; cases I–V deterministics. Pesaran-Shin-Smith 2001. Validate vs Stata `ardl`; statsmodels `bounds_test` is incomplete on p-values. |

#### Structural breaks

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Chow break and predictive-failure tests | Known-date break test and out-of-sample predictive failure variant. | Low | F-form with heteroskedasticity-robust option; predictive Chow for short second regimes. Chow 1960. Validate vs `strucchange::sctest(type='Chow')`. |
| Quandt-Andrews sup/exp/ave-Wald unknown-break tests | Unknown single-break tests with nonstandard asymptotics; the standard first break screen. | Medium | 15% trimming default; p-values via the Hansen 1997 response-surface approximation; exp/ave variants of Andrews-Ploberger 1994. Andrews 1993. Validate vs strucchange and EViews. |
| CUSUM/CUSUMSQ (recursive) and OLS-CUSUM/MOSUM fluctuation tests | Graphical parameter-stability monitoring via recursive or OLS residual partial sums. | Medium | Recursive residuals via QR updating; the 5% boundary functions differ between recursive and OLS variants (Brown-Durbin-Evans 1975; Ploberger-Krämer 1992); CUSUMSQ boundaries from Edgerton-Wells tables. Validate vs `strucchange::efp` plots. |
| Bai-Perron multiple structural breaks (estimation + inference) | Global SSR minimization over m breaks via dynamic programming, sequential sup-F(l+1 given l) tests, IC-based break-number selection, and break-date confidence intervals. Flagship item. | High | Precompute the triangular SSR array with recursive updating (O(T²) time/memory — tile for cache); trimming h and max breaks; heterogeneous error/regressor distributions across regimes affect CI construction (Bai 1997); HAC-robust versions change the limiting distributions. Bai-Perron 1998, 2003. Validate vs `strucchange::breakpoints` and the BP 2003 US ex-post real interest rate example — exact break dates and CIs. Rust speed is a major selling point. |

#### Seasonality detection

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| QS seasonality test | Ljung-Box-type statistic at seasonal lags; X-13's headline residual-seasonality diagnostic. | Low | Applied to differenced and/or seasonally adjusted series; only positive autocorrelations count. Validate vs `seastests::qs` and X-13's QS output — replicate exactly, sign conventions matter. |
| Friedman and Kruskal-Wallis rank tests for stable seasonality | Nonparametric tests of equality of period means; robust seasonality screens. | Low | As implemented in seastests; also X-11's stable-seasonality F (D8 table) and moving-seasonality F. Validate vs seastests and X-13 F-tables. |

#### Seasonal adjustment

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Classical moving-average decomposition | Textbook additive/multiplicative decomposition; baseline and pedagogy. | Low | Centered MA trend, period-averaged seasonals. Validate vs R `decompose()` and statsmodels `seasonal_decompose`. |
| STL decomposition (exact Cleveland implementation) | Loess-based robust seasonal-trend decomposition; the workhorse for exploratory SA outside official statistics. | High | Port the netlib Fortran semantics exactly: inner/outer loops, bisquare robustness weights, loess degree 0/1/2 with "jump" speedups, s.window='periodic' special case. Cleveland et al 1990. Validate bit-level vs R `stl()` (statsmodels STL already matches — meet that bar). Bootstrap wrapper for uncertainty. |
| X-13ARIMA-SEATS bindings (full spec + diagnostics parsing) | Production seasonal adjustment via the Census binary: spec-file generation, run management, complete parsing of output tables into typed objects. | Medium | Ship this first; pure X-11/SEATS reimplementation is a multi-year research-grade goal (bit-for-bit vs the Census test suite; extreme-value replacement iterations and Musgrave asymmetric Henderson end-weights are the hard parts). Parse M1–M11/Q, sliding spans, revision history, QS, spectral diagnostics — R `seasonal` is the ergonomics bar to beat. Binary packaging is handled at the architecture level. Validate vs R seasonal outputs on the Census test series. |

#### Calendar effects

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Trading-day, working-day, leap-year, length-of-month regressors | Calendar regressor construction for regARIMA/regression pre-adjustment. | Medium | Contrast coding (6 TD contrasts vs 1 WD); centering against long-run day-type means to avoid confounding the seasonal; country calendar support via the foundations calendar engine. Bell-Hillmer 1983. Validate vs X-13 td/lpyear regressor values exactly. |

#### Filters and detrending

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| HP filter (sparse) with Ravn-Uhlig lambda rules + one-sided variant | The Hodrick-Prescott filter with frequency-correct lambda and a real-time one-sided version via the Kalman filter. | Medium | Pentadiagonal sparse Cholesky O(T) — never dense inversion; lambda 1600 quarterly / 6.25 annual / 129600 monthly (Ravn-Uhlig 2002 fourth-power rule); one-sided via local-linear-trend Kalman (Stock-Watson 1999). Docs must present the Hamilton 2018 critique and route users to alternatives — the "which filter when" page is a documentation flagship. Validate vs `mFilter::hpfilter` and the Meyer-Winkler one-sided implementation. |
| Hamilton regression filter | Hamilton's (2018) proposed HP replacement: h-step-ahead OLS projection residual as the cycle (h=8, p=4 quarterly). | Low | Simple OLS but handle sample loss and frequency-dependent (h,p) defaults. Hamilton 2018. Validate vs the neverhpfilter R package replication of Hamilton's employment example. |
| Baxter-King band-pass filter | Symmetric truncated ideal band-pass isolating business-cycle frequencies (6–32 quarters); loses K observations at each end. | Low | Weights normalized to sum to zero (removes unit root). Baxter-King 1999. Validate vs `mFilter::bkfilter` and statsmodels `bkfilter`. |
| Christiano-Fitzgerald asymmetric band-pass filter | Full-sample asymmetric band-pass optimal under a random-walk assumption; no lost endpoints, but time-varying weights mean revisions. | Medium | Drift removal and RW-assumption options; implement stationary variants too. Christiano-Fitzgerald 2003. Validate vs `mFilter::cffilter` and statsmodels `cffilter`. |

#### Spectral analysis

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Periodogram and smoothed periodogram (Daniell) with tapering | Base spectral estimator with detrending, split-cosine tapering, Daniell smoothing. | Medium | Normalization conventions differ across R/Matlab/scipy (2π placement, one- vs two-sided) — pick one, document loudly, provide converters; the spectrum must integrate to the variance in the chosen convention. Chi-squared CIs from equivalent df. Validate vs R `spec.pgram` with matched options. |
| Welch averaged periodogram | Segment-averaged PSD, familiar to engineers; useful for long high-frequency series. | Low | Overlap/window options. Validate vs `scipy.signal.welch`. |

#### Long memory

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| GPH log-periodogram estimator of d (+ bias-reduced variants) | Baseline semiparametric long-memory estimation from low-frequency periodogram ordinates. | Medium | Bandwidth m=T^0.5 default is arbitrary — ship a bandwidth-sensitivity plot; Andrews-Guggenberger 2003 bias-reduced version. Geweke-Porter-Hudak 1983. Validate vs R longmemo / `fracdiff::fdGPH`. |

#### Outlier detection

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Chen-Liu ARIMA outlier detection (AO/LS/TC/IO) with joint re-estimation | The canonical econometric outlier procedure: detect, classify, and adjust outliers jointly with ARIMA estimation; also feeds regARIMA. | High | Iterative outer/inner loops (detect → joint estimate → re-detect); results depend on critical value C (≈3–4 scaled by T), iteration order, and the ARIMA engine — divergences from R tsoutliers must be documented test-by-test. Chen-Liu 1993. Validate vs `tsoutliers::tso` and X-13's outlier spec on identical series. |
| Rolling-window robust outlier screens (Hampel, rolling MAD, tsclean-style) | Fast nonparametric anomaly flags for data-quality screening. | Low | Hyndman's tsclean = STL remainder outside ±3·IQR with seasonal-aware replacement. Validate vs `forecast::tsoutliers`/`tsclean`. |

#### Missing data

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Kalman interpolation via auto-fitted structural/ARIMA models | State-space smoothing imputation — the best general-purpose method and the benchmark in imputeTS studies. | Medium | Exact Kalman handling of NA (skip the update step) via the foundations state-space engine; provide imputation uncertainty. Validate vs `imputeTS::na_kalman` and the Moritz-Bartz-Beielstein 2017 benchmark. |
| Interpolation suite (linear, spline, Stineman, seasonal-aware na.interp, LOCF with warnings) | Fast gap-filling utilities with seasonality handling. | Low | na.interp = linear on the STL-adjusted series, then re-add the seasonal; LOCF must warn about distortion of dynamics. Validate vs `forecast::na.interp`, imputeTS. |

#### Transformations

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Box-Cox transformation (Guerrero and MLE lambda, profile CI, bias-adjusted inversion) | Variance stabilization with automatic lambda selection and correct back-transformation. | Medium | Guerrero 1993 (forecast-package default) and profile-likelihood MLE with CI; handle zeros (log1p/shift with warning); bias-adjusted mean back-transform. Validate vs `forecast::BoxCox.lambda` and R `boxcox()`. |
| Differencing advisors: ndiffs / nsdiffs | Automatic recommendation of d and D from unit-root/seasonal-strength test sequences — the entry point of every auto-ARIMA workflow. | Low | Replicate forecast defaults (KPSS for d; Wang-Smith-Hyndman 2006 seasonal strength or CH/OCSB for D) but expose the underlying test evidence, not just the integer. Validate vs `forecast::ndiffs`/`nsdiffs` across the option grid. |

#### Shared statistical plumbing (module-owned)

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Unified information criteria infrastructure (AIC/AICc/BIC/HQIC) | One consistent IC implementation shared by all models, with explicit likelihood-constant and effective-sample conventions. | Medium | The classic cross-package bug: comparing ICs across models fit on different effective samples (lag conditioning) or with/without likelihood constants. Enforce common-sample comparison in APIs; document conventions vs R/statsmodels (they differ). Infrastructure, not a feature — design early. |
| Rolling/recursive statistics engine | O(1)-update rolling mean/var/cov/corr/beta/quantiles and recursive residuals — shared kernel for CUSUM, monitoring, GSADF, and exploration plots. | Low | Numerically stable updates (Welford; avoid catastrophic cancellation via compensated or chunked recomputation); recursive OLS via QR updating. Validate vs brute-force recomputation at 1e-12. |

#### Workflow automation

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| check_series(): one-call diagnostic battery with recommendations report | The "check my series" flagship: frequency/calendar detection, missingness, outliers, seasonality tests, ADF+KPSS decision matrix with break-aware branch, ARCH, normality, long memory, break scan — a typed object plus HTML report with model recommendations. | Medium | Composable pipeline over the primitives above; the report must state test assumptions and multiple-testing caveats (never auto-Bonferroni silently — show families). No existing library has this. Validate components individually; snapshot-test the report. |
| Unit-root/stationarity decision workflow | Codified decision tree: joint ADF/KPSS interpretation, trend-vs-difference stationarity calls, automatic escalation to break-robust tests when fluctuation tests fire. | Medium | Output an evidence table, never a bare verdict; based on Elder-Kennedy 2001 pedagogy plus Harvey-Leybourne-Taylor union-of-rejections results. Differentiator: everyone ships tests, nobody ships the workflow. |

### Tier 2 — Standard (expected of a serious library)

#### Correlation diagnostics

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Extended ACF (EACF, Tsay-Tiao) | Iterated-regression autocorrelation table whose "triangle of zeros" identifies ARMA(p,q) orders jointly. | Medium | Iterated AR fits then ACF of transformed residuals; render the O/X table. Tsay-Tiao 1984. Validate vs `TSA::eacf` on Cryer-Chan textbook examples. |

#### Whiteness tests

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Durbin-Watson and Durbin's h | Classical first-order serial correlation test; mostly pedagogical but expected. | Low | Exact p-values via the Imhof/Pan weighted chi-square algorithm, not just dL/dU bounds tables. Invalid with lagged dependent variables (offer Durbin h). Durbin-Watson 1950. Validate vs `lmtest::dwtest` exact p-values. |
| Multivariate portmanteau (Hosking, Li-McLeod) | Whiteness of VAR residual vectors; the standard VAR adequacy check. | Medium | df = K²(h−p); small-sample corrected version. Hosking 1980; Li-McLeod 1981. Validate vs `vars::serial.test` and JMulTi outputs from Lütkepohl 2005 examples. |

#### Residual diagnostics

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Doornik-Hansen multivariate normality | Omnibus multivariate normality test for VAR residuals. | Medium | Transformed skewness/kurtosis after orthogonalization; the orthogonalization choice (Cholesky vs symmetric) changes results — expose both, like JMulTi. Doornik-Hansen 2008. Validate vs `vars::normality.test` and JMulTi. |
| Engle-Ng sign bias tests | Tests for asymmetric volatility response (sign, negative size, positive size, joint) guiding GJR/EGARCH choice. | Low | Auxiliary regressions on standardized residuals. Engle-Ng 1993. Validate vs rugarch `signbias()`. |
| Ramsey RESET functional form test | Omitted nonlinearity in the regression mean; cheap general misspecification screen. | Low | Powers of fitted values in an auxiliary regression, F test. Ramsey 1969. Validate vs `lmtest::resettest`. |

#### Nonlinearity tests

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| McLeod-Li test | Ljung-Box on squared residuals; detects ARCH-type nonlinearity. | Low | McLeod-Li 1983. Validate vs `TSA::McLeod.Li.test`; share the Ljung-Box code path. |
| Keenan and Tsay F-tests for nonlinearity | Volterra-expansion tests against quadratic nonlinearity in the mean. | Low | Auxiliary regressions with fitted-value squares (Keenan) / cross-products (Tsay). Keenan 1985; Tsay 1986. Validate vs `TSA::Keenan.test`, `Tsay.test`. |
| Teräsvirta STAR linearity sequence | LM tests from a third-order Taylor expansion; the sequential procedure also selects LSTAR vs ESTAR. Core of the classical nonlinear workflow. | Medium | Auxiliary regressions with transition-variable power interactions; F versions preferred in small samples; also the Luukkonen-Saikkonen-Teräsvirta 1988 variant. Teräsvirta 1994. Validate vs tsDyn and Teräsvirta's published sunspot/lynx applications. |
| Hansen threshold linearity test (bootstrap sup-LM) | Linear AR vs TAR/SETAR when the threshold is unidentified under the null; bootstrap p-values required. | High | Grid over threshold candidates, sup-F, heteroskedasticity-robust fixed-regressor bootstrap — where the speed pillar pays off. Hansen 1996; Hansen 1999 sunspot/GNP applications to replicate. Validate vs Hansen's Matlab/GAUSS outputs. |

#### Unit root and stationarity tests

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| ERS point-optimal test (P_T) | Feasible point-optimal unit root test; completes the ERS family. | Medium | Requires an AR-based spectral density estimate at frequency zero. ERS 1996. Validate vs `urca::ur.ers(type='P-test')`. |
| Ng-Perron M-tests (MZa, MZt, MSB, MPT) | Modified tests with good size under negative MA errors, combined with MAIC; the serious practitioner's toolkit. | Medium | AR spectral density estimator at frequency zero (not kernel) is essential for the size properties. Ng-Perron 2001. Validate vs Stata `ngperron`, EViews, and the paper's tables. |
| Zivot-Andrews endogenous one-break unit root test | Unit root allowing one endogenous break in intercept/trend/both under the alternative. | Medium | Grid over break dates with trimming, min-t. Break only under the alternative — a known criticism; cross-link Lee-Strazicich. Zivot-Andrews 1992. Validate vs statsmodels `zivot_andrews` and `urca::ur.za`. |
| Lee-Strazicich minimum LM unit root test (1 and 2 breaks) | Break-in-null-and-alternative unit root test; avoids Zivot-Andrews-type spurious rejections. | High | LM detrending, grid search over one/two break dates, crash vs trend-shift models; critical values depend on break location λ — interpolate their tables or simulate. Lee-Strazicich 2003 (2 breaks), 2013 (1 break). Validate vs the authors' GAUSS code / Stata addon; almost nothing reliable exists in Python — flagship gap. |

#### Seasonal unit roots

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Canova-Hansen test for seasonal stability | Null of deterministic (stable) seasonality against unit-root seasonality; the KPSS-style complement to HEGY. | Medium | LM-type with HAC long-run variance; per-frequency and joint statistics; basis of `forecast::nsdiffs` default. Canova-Hansen 1995. Validate vs `uroot::ch.test` and `forecast::nsdiffs`. |
| OCSB test | Regression-based test for seasonal differencing used by auto.arima's nsdiffs. | Medium | Critical values are simulation/surface based (the forecast package regenerated them). Osborn-Chui-Smith-Birchenhall 1988. Validate vs `forecast::ocsb.test`. |

#### Cointegration tests

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Gregory-Hansen cointegration test with regime shift | Residual-based cointegration allowing one break in intercept/trend/slope under the alternative. | Medium | Inf-ADF/Zt/Za over a trimmed break grid. Gregory-Hansen 1996. Validate vs their published critical values and the Stata ghansen addon. |

#### Structural breaks and trend tests

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Nyblom-Hansen parameter stability L-test | LM test of constancy against martingale parameter variation; per-parameter and joint. | Medium | Cramér-von Mises-type limit; critical values depend on the number of parameters. Nyblom 1989; Hansen 1992. Validate vs strucchange and rugarch's Nyblom output. |
| Mann-Kendall trend test + Sen's slope (autocorrelation-corrected) | Nonparametric monotonic trend detection, ubiquitous in climate/hydrology and ESG-adjacent economics. | Low | Implement the Hamed-Rao variance correction and Yue-Wang prewhitening — naive MK is badly oversized on autocorrelated data. Validate vs R modifiedmk / pymannkendall. |

#### Seasonality detection

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Automatic period detection (findfrequency / autoperiod / spectral peaks) | Infers possibly multiple seasonal periods from data — essential for daily/high-frequency series with unknown periodicity. | Medium | AR-spectrum peak detection (Hyndman's findfrequency) plus autoperiod-style hill-climbing validation for multiple/non-integer periods. Validate vs `forecast::findfrequency` on M4/daily datasets. |

#### Seasonal adjustment

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| MSTL for multiple seasonalities | Iterated STL extracting several seasonal components (daily+weekly+yearly). | Medium | Ordering of periods and lambda pre-transformation matter. Bandara-Hyndman-Bergmeir 2021. Validate vs `forecast::mstl` and statsmodels MSTL. |
| regARIMA pre-adjustment (auto outliers, trading day, Easter, auto model) | TRAMO-style automatic pre-treatment before decomposition: outlier regressors, calendar regressors, automatic ARIMA selection, forecast extension. | High | Automatic model identification per TRAMO (Gómez-Maravall 1996); outlier detection shares the Chen-Liu engine. Validate vs X-13 regARIMA output tables and JDemetra+. |
| Seasonal adjustment quality diagnostics (M/Q statistics, sliding spans, revisions) | Programmatic quality assessment of any SA output: M1–M11/Q composite, sliding-spans stability, revision-history triangles, residual seasonality checks. | Medium | Make these first-class objects usable on ANY decomposition (STL/MSTL/X-13/SEATS) — nobody offers that unification. Lothian-Morry 1978 (M stats); Findley et al 1990 (sliding spans). Validate vs X-13's own tables. |

#### Calendar effects

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Easter and moving-holiday regressors (genhol equivalent) | Easter[w], Chinese New Year, Ramadan, Diwali regressors built from holiday windows with centering — required for non-US/European official statistics. | Medium | Computus algorithm for Easter (Gregorian and Julian) from the foundations calendar engine; genhol-style before/during/after windows, mean-centering by calendar month. Validate vs the Census genhol utility and JDemetra+ national calendars. |
| Fourier/harmonic seasonal terms and dummy builders | Deterministic seasonality regressor construction (harmonics with selectable K, seasonal dummies) for regression and ARIMAX use. | Low | K selection via AICc as in Hyndman's `fourier()`; non-integer periods supported (365.25). Trivial but heavily used. |

#### Filters and trend-cycle decomposition

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Butterworth filter (Gómez) | Rational square-wave filters for flexible low-pass/band-pass trend extraction; used inside the TRAMO-SEATS ecosystem. | Medium | Implement in state-space form for stability rather than direct polynomial filtering. Gómez 2001. Validate vs `mFilter::bwfilter`. |
| Henderson moving averages with Musgrave asymmetric ends | The trend filters at the heart of X-11; also useful standalone for smooth trends. | Medium | Exact Henderson weight formulas for any odd length; Musgrave end-weights depend on the I/C ratio — replicate Census values precisely (prerequisite for any X-11 reimplementation). Musgrave 1964; Doherty 2001. |
| Beveridge-Nelson decomposition (ARIMA and multivariate/VAR forms) | Permanent-transitory decomposition defined by long-horizon forecasts; the model-based benchmark for output-gap questions. | Medium | Implement via the state-space companion form (Morley 2002) — analytic, no truncation of the long-run forecast sum. Beveridge-Nelson 1981. Validate vs published US GDP decompositions (Morley-Nelson-Zivot 2003). |

#### Spectral analysis

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Multitaper spectral estimation (Thomson DPSS, adaptive weights, jackknife CIs) | Best-in-class bias/variance tradeoff for spectra; includes the Thomson harmonic F-test for embedded periodicities. | High | Compute DPSS via the symmetric tridiagonal formulation (Slepian) — never naive Toeplitz eigendecomposition; adaptive eigenvalue weighting; jackknife confidence bands. Thomson 1982; Percival-Walden 1993. Validate vs the R multitaper package. |
| Lag-window (Blackman-Tukey) and AR/maximum-entropy spectra | Classical alternatives: kernel-on-ACF and AR-model-implied spectral densities. | Low | AR spectrum via Burg/YW with order by AIC (matches R `spec.ar`). Validate vs R `spec.ar`. |
| Cross-spectrum: coherence, phase, gain with CIs | Frequency-domain comovement — lead/lag by frequency band; central for business-cycle comovement work. | Medium | Unsmoothed coherence is identically 1 — enforce smoothing; bias corrections and CIs via equivalent df; phase CI blows up when coherence is low. Priestley 1981. Validate vs R `spec.pgram` bivariate and Matlab `mscohere`. |
| Bartlett cumulative periodogram white-noise test | Kolmogorov-Smirnov-type test on the cumulated periodogram; elegant whiteness check complementing Ljung-Box. | Low | Bartlett 1955; Durbin 1969. Validate vs R `cpgram()`. |

#### Long memory and persistence

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Local Whittle and Exact Local Whittle estimators | Efficient semiparametric d estimation; ELW extends validity to nonstationary d≥0.5, with unknown-mean/trend variants. | High | Optimize the Whittle objective (well-behaved, 1-D); ELW requires fractional differencing inside the objective — use FFT fracdiff; Shimotsu 2010 two-step for unknown mean/trend. Robinson 1995; Shimotsu-Phillips 2005; Shimotsu 2010. Validate vs Shimotsu's Matlab code and R LongMemoryTS. Python has essentially nothing here. |
| R/S and Lo's modified R/S statistics | Classical rescaled-range long-memory tests; Lo's version robust to short memory. | Low | Lo 1991; the correction's bandwidth choice matters. Validate vs R pracma/longmemo Hurst implementations and Lo's published stock-return results. |
| Fast fractional differencing and integration utilities | FFT-based (1−L)^d transforms; building block for ARFIMA, ELW, and fractional cointegration elsewhere in the library. | Medium | Jensen-Nielsen 2014 fast fractional difference — O(T log T), exact. Validate vs the naive O(T²) recursion. |
| Variance-ratio tests (Lo-MacKinlay, Chow-Denning, automatic, wild bootstrap) | Random-walk tests at multiple horizons; standard market-efficiency and persistence diagnostics. | Medium | Overlapping-observation estimator with heteroskedasticity-robust SEs; Chow-Denning 1993 multiple-horizon control; Choi 1999 automatic horizon; Kim 2006 wild bootstrap for small samples. Validate vs R vrtest and arch's `VarianceRatio`. |

#### Transformations

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Macro data utilities: growth rates, annualization, rebasing, deflation, chain-linking | Quality-of-life transforms every central-bank user writes by hand (log-diff vs percent, QoQ annualized, index rebasing, real-izing nominal series, chain-linked aggregation warnings). | Low | Trivial math, high API value; get compounding conventions exactly right and unit-test against FRED transformations. |

### Tier 3 — Advanced (differentiators)

#### Correlation and whiteness diagnostics

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Robust autocorrelation (rank/Gnanadesikan-Kettenring based) | Outlier-resistant ACF for exploratory work on contaminated data. | Medium | Ma-Genton 2000 robust ACF via the Qn scale estimator; also Spearman autocorrelation. Validate vs R robts/robustbase components. Document as exploratory, not for inference. |
| Weighted portmanteau tests (Fisher-Gallagher) | Weighted Ljung-Box/McLeod-Li variants with better size and power than classic Q. | Medium | Weighted sum of squared ACF; the null is a linear combination of χ²(1) — use a gamma/Satterthwaite approximation. Fisher-Gallagher 2012. Validate vs R WeightedPortTest. |
| Automatic portmanteau (Escanciano-Lobato) | Data-driven lag choice, robust to conditional heteroskedasticity — removes the arbitrary "lags=" choice. | Medium | AIC/BIC-type penalized lag selection; heteroskedasticity-robust correlations in the statistic. Escanciano-Lobato 2009. Validate vs R vrtest/Auto.Q. A great default inside the automatic diagnostic report. |
| Li-Mak test for standardized GARCH residuals | Portmanteau on squared standardized residuals valid after GARCH fitting, where Ljung-Box is not. | Medium | Accounts for the estimation effect in the asymptotic distribution. Li-Mak 1994. Validate vs `WeightedPortTest::Weighted.LM.test` and published GARCH diagnostics in Tsay's textbook. |

#### Nonlinearity and local stationarity

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| White / Lee-White-Granger neural network test | Neglected-nonlinearity test using random hidden-unit activations. | Medium | Random logistic activations, principal components of them in an auxiliary regression; results depend on the RNG — fix seeds and document. Lee-White-Granger 1993. Validate vs `tseries::white.test`. |
| Time-reversibility test (Ramsey-Rothman TR) | Tests symmetry of the process in time — evidence against linear-Gaussian generation. | Medium | Bicovariance-based statistic. Ramsey-Rothman 1996. Niche but trivial once the bootstrap infrastructure exists. |
| Hinich bispectrum Gaussianity/linearity tests | Frequency-domain tests: flat bicoherence implies linearity; zero implies Gaussianity. | High | Bispectrum estimation smoothing choices dominate the results. Hinich 1982. Validate against Matlab HOSA toolbox conventions. |
| Tests for second-order/local stationarity (Priestley-Subba Rao, wavelet-based) | Tests whether the spectral structure itself changes over time — flags series where global stationarity assumptions fail. | High | PSR 1969 evolutionary spectrum ANOVA; Nason 2013 wavelet test (validate vs R locits). A good citizen of the automatic report. |

#### Unit root tests

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Perron (1997) / Vogelsang-Perron AO-IO break tests | Classical endogenous-break unit root tests (additive vs innovational outlier forms). | Medium | Perron 1997. Validate vs EViews breakur and the published Nelson-Plosser reanalysis — replicating Perron's Nelson-Plosser tables is a great accuracy showcase. |
| Lumsdaine-Papell two-break ADF | Two endogenous breaks under the alternative; common in applied energy/macro literatures. | Medium | O(T²) grid — parallelize. Lumsdaine-Papell 1997. Validate vs GAUSS/Stata addons. |
| Hansen covariate-augmented ADF (CADF) | Uses correlated stationary covariates to gain large power over univariate ADF. | Medium | The nuisance parameter rho² indexes the limit distribution — interpolate critical-value surfaces. Hansen 1995. Validate vs the R CADFtest package. |
| Leybourne-McCabe stationarity test | Parametric alternative to KPSS with better size in autocorrelated data. | Medium | Requires ARIMA(p,0,1) auxiliary estimation. Leybourne-McCabe 1994. Validate vs EViews' implementation. |
| Panel unit root and stationarity tests (LLC, IPS, Fisher-type, Hadri, CIPS) | First-generation and cross-sectionally dependent (Pesaran CIPS) panel unit root tests used by central-bank researchers. | High | Cross-sectional dependence handling is the key differentiator (Pesaran 2007 CADF/CIPS). Levin-Lin-Chu 2002; Im-Pesaran-Shin 2003; Pesaran 2007. Validate vs `plm::purtest` and Stata pescadf. Housed here for API adjacency to the unit-root family; will seed the future panel-time-series extension module. |

#### Bubble detection

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| SADF/GSADF explosive-root (bubble) tests with date-stamping | Right-tailed recursive ADF tests detecting and dating explosive episodes; heavily used by central banks. | High | O(T²) regressions — rank-one updatable OLS plus parallelism; wild bootstrap of Phillips-Shi 2020 for multiplicity/heteroskedasticity; BSADF date-stamping sequences. Phillips-Wu-Yu 2011; Phillips-Shi-Yu 2015. Validate vs R exuber and psymonitor, replicating the S&P 500 episodes in PSY 2015. |

#### Cointegration tests

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Hatemi-J two-break cointegration test | Extension of Gregory-Hansen to two regime shifts. | Medium | Hatemi-J 2008. Validate vs the author's GAUSS code / Stata addon. |
| Saikkonen-Lütkepohl cointegration tests | Rank tests with prior adjustment for deterministics; better small-sample behavior than Johansen in some cases; standard in the German school. | Medium | GLS detrending, then a Johansen-type test. Saikkonen-Lütkepohl 2000. Validate vs JMulTi. |
| Threshold cointegration tests (Enders-Siklos TAR/M-TAR) | Asymmetric adjustment toward long-run equilibrium; common in price-transmission literatures. | Medium | Engle-Granger residuals with TAR/M-TAR adjustment; the F for the threshold effect is nonstandard — simulate. Enders-Siklos 2001. Validate vs R apt/tsDyn pieces. |
| Shin test (cointegration as null) | KPSS-analogue with a null of cointegration; enables confirmatory analysis with EG/PO. | Medium | Requires DOLS/leads-lags residuals. Shin 1994. Validate vs cointReg-related implementations. |
| Bayer-Hanck combined cointegration test | Fisher combination of EG, Johansen, Boswijk, Banerjee tests into one decision. | Low | Fisher χ² combination with simulated CVs. Bayer-Hanck 2013. Cheap once the components exist; popular in applied energy economics. |

#### Structural breaks and trend inference

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Elliott-Müller qLL stability test | Efficient test against diverse instability (many small breaks, random-walk parameters); more powerful and easier than sup-tests in that direction. | Medium | qLL statistic from quasi-differenced residual regressions; critical values from their table. Elliott-Müller 2006. Almost no open-source implementation — differentiator. |
| Andrews end-of-sample instability test | Valid when the potential post-break sample is very short (subsampling-based). | Medium | P-values by subsampling permutation. Andrews 2003. Useful for real-time "did COVID break my model" questions. |
| ICSS variance-break detection (Inclan-Tiao + Sansó corrections) | Iterated cumulative sum of squares for multiple variance breaks; standard in the volatility literature. | Medium | The original test is badly oversized under kurtosis/persistence — make the Sansó-Aragó-Carrion 2004 kappa-1/kappa-2 corrections the default. Validate vs the R ICSS package and published crash-dating papers. |
| Modern changepoint algorithms (PELT, WBS, MOSUM, BOCPD) | Fast multiple-changepoint detection in mean/variance for long series where Bai-Perron asymptotics or O(T²) cost don't fit; bridges to the ML changepoint literature. | High | PELT O(T) with penalty (Killick et al 2012); Wild Binary Segmentation (Fryzlewicz 2014); MOSUM with honest asymptotic inference (Eichinger-Kirch 2018); Bayesian online CPD (Adams-MacKay 2007) for monitoring. Validate vs R changepoint/mosum and Python ruptures. Position clearly relative to Bai-Perron in docs (detection vs inference). |
| Robust trend inference (Vogelsang t-PS; Bunzel-Vogelsang; HLT) | Inference on a deterministic linear trend valid under I(0) or I(1) errors. | Medium | Vogelsang 1998; Bunzel-Vogelsang 2005; Harvey-Leybourne-Taylor 2007. Validate vs published examples on temperature and GDP trends. |

#### Seasonal adjustment and aggregation

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| SEATS/canonical ARIMA-model-based decomposition | Decomposes a fitted ARIMA into trend/seasonal/irregular via partial fractions of the pseudo-spectrum; the model-based alternative to X-11 filters. | Research-grade | Admissibility can fail (negative component spectra) — SEATS then approximates the model, and replicating those approximations is the hard part. Burman 1980; Hillmer-Tiao 1982. Validate vs SEATS output through X-13 and JDemetra+. |
| Direct vs indirect adjustment utilities for aggregates | Compare adjusting an aggregate directly vs summing adjusted components; consistency diagnostics. | Low | Pure bookkeeping plus diagnostics on the discrepancy series. Standard central-bank workflow (Eurostat guidelines). |

#### Spectral and time-frequency analysis

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Lomb-Scargle periodogram for irregular sampling | Spectral estimation for unevenly spaced observations; also useful under missing data. | Medium | Fast O(N log N) algorithm (Press-Rybicki 1989). Validate vs astropy LombScargle. |
| Wavelet analysis: DWT/MODWT, wavelet variance, CWT with coherence | Time-frequency decomposition; wavelet coherence is popular in applied macro/finance comovement papers. | High | MODWT with reflection boundary and cone-of-influence tracking; significance via AR(1) surrogate simulation (Torrence-Compo 1998); wavelet coherence per Grinsted et al 2004. Validate vs R waveslim/WaveletComp and the Torrence-Compo reference implementation. |

#### Trend-cycle decomposition

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| L1 trend filtering | Piecewise-linear trend extraction via an L1 penalty on second differences; kink locations double as break candidates. | Medium | Specialized ADMM or primal-dual interior point on banded systems. Kim-Koh-Boyd 2009. Validate vs cvxpy reference solutions. |

#### Outliers, missing data, and features

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Seasonal Hybrid ESD (S-H-ESD) anomaly detection | Twitter's STL + generalized ESD procedure for anomalies in high-frequency seasonal data; industry-forecaster staple. | Medium | Generalized extreme studentized deviate on the STL remainder with a median-based trend. Hochenbaum-Vallis-Kejariwal 2017. Validate vs Twitter's AnomalyDetection R package. |
| Missing-aware ACF/spectrum and gap diagnostics | Correct second-moment estimation under missingness (pairwise-complete ACF with bias notes, Lomb-Scargle spectrum) plus missingness-pattern reporting. | Medium | Pairwise ACF is not PSD — document; state-space exact likelihood is the gold standard. Feed gap statistics into the data-quality report. |
| Time-series feature extraction (tsfeatures/catch22 canonical sets) | Canonical feature vectors (trend/seasonality strength, spectral entropy, stability, lumpiness, catch22) for exploration, clustering, and forecast-model selection. | Medium | STL-based strength measures (Wang-Smith-Hyndman 2006); catch22 (Lubba et al 2019) has a C reference implementation to validate against; R tsfeatures for the Hyndman set. Fast Rust execution across millions of series — industry-forecaster catnip. |

#### Business-cycle dating

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Business-cycle dating: Bry-Boschan / BBQ (Harding-Pagan) | Turning-point detection and cycle-phase statistics (duration, amplitude, concordance) for monthly/quarterly indicators. | Medium | Censoring rules (minimum phase/cycle lengths) drive everything — expose them; concordance index with HAC inference. Bry-Boschan 1971; Harding-Pagan 2002. Validate vs R BCDating and NBER reference dates. |

### Tier 4 — Frontier (research-grade, each gated on a named validation target)

#### Unit root strategies

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Carrion-i-Silvestre-Kim-Perron GLS unit root with multiple breaks | State of the art: GLS-detrended tests allowing multiple breaks under both null and alternative. | Research-grade | Quasi-GLS with break dates estimated by SSR minimization (dynamic programming shared with Bai-Perron); bootstrap or simulated critical values as a function of break fractions. Carrion-i-Silvestre, Kim, Perron 2009. Gate: reproduce the authors' GAUSS code outputs. |
| Union-of-rejections unit root strategies | Combines OLS- and GLS-detrended tests (and trend/no-trend) for robustness to uncertain initial condition and trend. | Medium | Scaled union critical values. Harvey-Leybourne-Taylor 2009, 2012. Almost no library implements; a natural component of the decision workflow. Gate: reproduce the HLT published simulation tables. |

#### Structural breaks and persistence

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Structural change monitoring (Chu-Stinchcombe-White) | Sequential monitoring with controlled size for detecting breaks in real time as new data arrive. | Medium | Fluctuation monitoring boundaries; pairs naturally with nowcasting workflows. Chu-Stinchcombe-White 1996. Gate: match `strucchange::mefp`. |
| Perron-Yabu trend-break test robust to I(0)/I(1) | Tests for a break in trend without knowing the order of integration — solves a nasty pretest circularity. | Medium | Feasible quasi-GLS with a superefficient rho estimate. Perron-Yabu 2009. Essentially unavailable outside GAUSS — differentiator. Gate: reproduce the authors' GAUSS outputs. |
| Change-in-persistence tests (I(0)↔I(1) switching) | Detects shifts between stationary and unit-root regimes — inflation persistence and bond-yield applications. | Medium | Ratio-based tests of Kim 2000/Busetti-Taylor 2004; Harvey-Leybourne-Taylor 2006 robust versions; Kejriwal-Perron-Zhou 2013 Wald approach. Essentially unavailable in mainstream Python/R. Gate: match published simulation tables. |

#### Seasonality

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Webel-Ollech combined seasonality test (WO) | Decision-combined QS + Kruskal-Wallis test tuned for automatic pipelines; current Bundesbank recommendation. | Low | Webel-Ollech 2019+ (Bundesbank discussion paper; seastests R package by Ollech). Trivial once components exist; nobody outside R has it. Gate: match `seastests::isSeasonal`. |
| STR: Seasonal-Trend decomposition by Regression | Regression/regularization-based decomposition with confidence intervals, exogenous regressors, and complex/multiple seasonality — the modern flexible alternative to STL. | High | Large sparse ridge problem; cross-validated smoothing parameters. Dokumentov-Hyndman 2022. Almost no adoption outside R — differentiator. Gate: match the R stR package. |

#### Filters and trend-cycle decomposition

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Boosted HP filter (bHP) | Iterated HP filtering with data-driven stopping (BIC/ADF) that fixes HP's under-smoothing of stochastic trends; Phillips's answer to Hamilton. | Medium | Iterate HP on the residual, stop by ADF test or BIC. Phillips-Shi 2021. Nobody mainstream ships this. Gate: match the authors' bHP R/Matlab code. |
| BN filter with dynamic demeaning (Kamber-Morley-Wong) | Regularized BN decomposition imposing a low signal-to-noise ratio — intuitive, large-amplitude output gaps in real time; adopted by several central banks. | Medium | AR(p) with restricted signal-to-noise δ, automatic δ selection, dynamic demeaning for slowly moving drift; 2025 refinements add structural-break handling. Kamber-Morley-Wong 2018 (+2025 update). Gate: match the authors' bnfiltering R/Matlab code. |
| Müller-Watson low-frequency projections | Inference on long-run properties (means, trends, covariability) via a small number of cosine-transform projections; a principled alternative to filtering for long-run questions. | High | Project onto the first q cosine basis functions; Bayes/frequentist inference on the low-frequency components. Müller-Watson 2018 and the 2008–2020 series. Nothing mainstream implements this. Gate: reproduce the authors' replication files. |

#### Spectral and long-memory frontier

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Quantilogram, cross-quantilogram, and quantile spectral densities | Dependence in quantiles across lags/frequencies — captures tail-dependence dynamics invisible to the ACF. | High | Stationary bootstrap for inference. Linton-Whang 2007; Han-Linton-Oka-Whang 2016; Baruník-Kley 2019. Essentially absent from Python. Gate: match the R quantilogram/quantspec packages. |
| Spurious long memory vs level shifts test (Qu) | Distinguishes true long memory from regime changes/level shifts that mimic it — a first-order applied concern in volatility series. | High | Local Whittle likelihood derivative-based sup statistic. Qu 2011; also Lu-Perron perspectives. Rarely available anywhere. Gate: match the R LongMemoryTS implementation. |

## Frontier watchlist

Source frontier themes not already carried as Tier 4 rows:

- **Real-time bubble monitoring**: GSADF/BSADF (Tier 3) extended into a live monitoring workflow with Phillips-Shi 2020 wild-bootstrap multiplicity control and CUSUM-style boundaries.
- **Seeded binary segmentation** (Kovács et al 2023) alongside the Tier 3 MOSUM/WBS changepoint suite.
- **Time-varying/recursive-evolving Granger causality** with bootstrap date-stamping (Shi-Phillips-Hurn 2018–2020) — owned by the multivariate module; tracked here because the recursive Wald engine reuses this module's rolling/recursive statistics kernel.
- **Library-wide HAR defaults per Lazarus-Lewis-Stock-Watson (2018)**: EWC/fixed-b critical values instead of Newey-West 1987 habits — owned by foundations; this module is the advocacy point (trend and mean inference) and largest consumer.
- **Exact-critical-value-on-demand**: parallel Rust simulation of null distributions for the user's exact T, deterministics, and break fractions — a foundations capability play; this module's tests define the demand.
- **The "which filter when" documentation flagship**: operationalizing the HP-vs-Hamilton-vs-BN debate with frequency-aware defaults, cross-linked from every filter.
- **catch22/tsfeatures at millions-of-series scale** (Lubba et al 2019) — the Tier 3 feature-extraction item pushed to industrial throughput.

## Implementation warnings

The "easy to get statistically or numerically wrong" list. Every item below is a known failure mode in existing tools.

1. **ACF conventions**: use the biased (divide-by-T) autocovariance so the sequence is PSD; Bartlett bands are for an MA(q) null while ±1.96/√T is for white noise — plotting the wrong one inverts users' conclusions. PACF "yule-walker adjusted" can exceed 1 in small samples.
2. **Ljung-Box degrees of freedom** must drop to h−p−q on ARMA residuals, and the test is invalid on GARCH-standardized residuals (route to Li-Mak); an API that does not know whether its input is raw data or residuals will be silently wrong.
3. **Unit-root tests live or die on deterministic-case handling and lag/bandwidth selection**: MAIC should use GLS/OLS detrending per Perron-Qu 2007; PP has severe size distortion under negative MA errors; KPSS conclusions flip with bandwidth — always print the case, lags, and bandwidth used, and use MacKinnon response surfaces rather than Dickey-Fuller tables.
4. **Johansen**: solve the generalized eigenproblem by Cholesky whitening + SVD (never invert moment matrices); p-values depend on which of the five deterministic cases is used, and asymptotic tables are oversized in small samples — offer the Johansen 2002 Bartlett correction.
5. **HEGY/seasonal tests**: critical values vary with the seasonal period AND deterministics (constant/dummies/trend); hard-coding quarterly tables and applying them to monthly data is a common published error — use Díaz-Emparanza surfaces or simulate.
6. **Bai-Perron**: precompute the triangular SSR array with recursive updating and mind O(T²) memory; trimming/minimum-segment choices change the sup-F distributions; heteroskedasticity/autocorrelation-robust variants alter both the test limits and the break-date CI construction — do not mix and match formulas.
7. **Filters**: never invert the dense HP matrix (sparse pentadiagonal Cholesky is O(T)); convert lambda across frequencies with the Ravn-Uhlig fourth-power rule; document end-point revision behavior (two-sided HP, CF) because real-time users get burned; BK loses K points at each end and users must be told.
8. **X-11/STL reimplementation traps**: Musgrave asymmetric Henderson end-weights depend on the I/C ratio, extreme-value replacement is iterative, and STL's loess uses degree/jump shortcuts — "close enough" reimplementations produce visibly different seasonal factors; validate bit-for-bit against Census/netlib references before claiming compatibility.
9. **Spectral estimation**: detrend and taper before the FFT; normalization conventions (2π placement, one- vs two-sided) differ across R/Matlab/scipy — pick one, document it, and test that the spectrum integrates to the variance; unsmoothed coherence is identically 1; compute DPSS tapers via the tridiagonal formulation for stability.
10. **Long memory**: GPH/local-Whittle results are dominated by the bandwidth m — ship sensitivity plots, not point estimates; standard LW is invalid for d≥0.5 (use exact LW) and biased under unmodeled means/trends (Shimotsu 2010 variant); use FFT-based fractional differencing to avoid O(T²).
11. **BDS**: use bit-parallel correlation-integral counting or it is unusably slow; asymptotic p-values are unreliable below T≈500 (bootstrap them); applying BDS to GARCH-standardized residuals changes the null distribution — document nuisance-parameter caveats.
12. **ARDL bounds**: the F-bound alone is not enough — degenerate cases (insignificant lagged level of y or of x) must be explicitly checked, and case I–V deterministic handling changes both bounds; use finite-sample/surface p-values for T<80.
13. **Toda-Yamamoto** (implemented in multivariate, surfaced in this module's workflows): augment with d_max lags but restrict the Wald test to the original p lags only — including the augmentation lags in the restriction (the most common bug) destroys the chi-squared limit.
14. **HAC/LRV**: implement once, use everywhere — inconsistent kernels/bandwidths across tests inside one library produce contradictory test outcomes on the same data; truncated kernels are not PSD; cap prewhitening AR roots (Andrews-Monahan 0.97 rule) or the LRV explodes near unit roots. (Enforced by consuming the single foundations implementation.)
15. **Bootstrap engineering**: use counter-based RNGs (e.g., Philox) with per-replication substreams so results are reproducible under any thread count; automatic block length needs the Patton-Politis-White 2009 correction; wild bootstrap weight choice (Rademacher vs Mammen) matters at small T. (Enforced via the foundations bootstrap engine.)
16. **Critical-value tables**: never silently clamp p-values at table edges (statsmodels' KPSS warning is the correct behavior); store validity ranges with every response surface and fall back to simulation, not extrapolation.
17. **Numerical hygiene for the accuracy pillar**: compensated summation for long-series moments, stable rolling-variance updates (Welford), QR-based recursive residuals, and a cross-platform determinism policy (SIMD/FMA reduction order) — otherwise the validation suite will flake across OS/architectures.
18. **Missing values change null distributions**: a uniform, explicit NA policy (error/drop/impute) must be enforced across every test; silent pairwise deletion in ACF-based statistics is a subtle correctness bug.
19. **Box-Cox**: back-transforming forecasts/means without the bias adjustment systematically understates levels; lambda has real sampling uncertainty — report a profile-likelihood CI, and handle zeros/negatives explicitly rather than by silent shifting.
20. **Calendar correctness underpins everything seasonal**: Easter computus, ISO week-53 years, leap-year regressors, and holiday windows must be exact and unit-tested per country — a one-day holiday misalignment shows up as residual seasonality that users will blame on the adjustment engine.
21. **Chen-Liu outlier detection** results depend on the critical-value default, iteration order, and the underlying ARIMA estimator; replicate R tsoutliers on a fixed corpus and document every intentional divergence, or users will file "wrong answer" bugs.
22. **Multiple testing in diagnostic batteries**: running 30 tests on one series guarantees false alarms — the report must group tests into families and present adjusted and unadjusted evidence, but never silently Bonferroni individual published statistics.

## Dependencies and shared infrastructure

### Consumed from foundations

- **Critical-value engine (response surfaces + on-demand simulation)** — CONSUMED. This module is its largest customer and effectively specifies it: MacKinnon 1996/2010 surfaces (ADF, Engle-Granger), MacKinnon-Haug-Michelis 1999 (Johansen), Hansen 1997 (sup-Wald), Díaz-Emparanza 2014 (HEGY), PSS 2001/Narayan 2005/Kripfganz-Schneider 2020 (ARDL bounds), plus break-fraction-indexed values (Lee-Strazicich, Gregory-Hansen, CKP). Requirements: validity ranges stored with every surface, warn-never-clamp at edges, simulation fallback keyed to the user's exact T/deterministics/break fractions, cached and versioned.
- **HAC/long-run variance + fixed-b/EWC inference** — CONSUMED. One audited implementation (Bartlett/Parzen/QS kernels; Andrews 1991 and Newey-West 1994 bandwidths; Andrews-Monahan 1992 prewhitening with the 0.97 root cap) feeding KPSS, PP, Phillips-Ouliaris, Canova-Hansen, robust trend tests, and Mann-Kendall corrections. Requirements: per-call bandwidth/kernel reporting hooks so every test prints what it used, and Lazarus-Lewis-Stock-Watson 2018 EWC/fixed-b defaults surfaced in this module's trend and mean inference.
- **Bootstrap/resampling engine** — CONSUMED. Wild/block/stationary/sieve bootstraps with Politis-White 2004 block length and the Patton-Politis-White 2009 correction, on Philox parallel substreams; consumed by the Hansen threshold test, GSADF/BSADF, variance ratios, small-sample BDS, HLT union strategies, quantilograms, and STL uncertainty wrappers.
- **Time-index/calendar/frequency/holiday engine** — CONSUMED. Exact Easter computus (Gregorian and Julian), ISO week-53 handling, country holiday databases, frequency detection, mixed-frequency alignment. This module's trading-day/holiday regressors, X-13 specs, and period detection are thin layers over it; a one-day error there becomes residual seasonality here.
- **Linear-Gaussian state-space engine** — CONSUMED. Exact NA handling for Kalman imputation, the one-sided HP filter, Butterworth in state-space form, and the Beveridge-Nelson companion-form computation.
- **Temporal disaggregation and benchmarking (Chow-Lin, Denton, Fernandez, Litterman)** — CONSUMED and re-exported next to the seasonal adjustment workflows (statistical-office users expect them side by side). Requirement: typed results with diagnostics on the discrepancy series, validated vs R tempdisagg.
- **Deterministic-terms toolkit, numerical optimizers, innovation-distribution zoo** — CONSUMED for test regressions, auxiliary ARIMA fits (Leybourne-McCabe, Chen-Liu, regARIMA), and simulated p-values.
- **Exogenous-regressor (covariate) contract** — CONSUMED. Covariate-augmented tests (Hansen CADF), regARIMA pre-adjustment regressors (trading day, Easter, outlier dummies), and user-supplied intervention variables enter through the shared aligned interface with loud alignment diagnostics.
- **Philox reproducible parallel RNG, unified forecast object, golden-value validation harness** — CONSUMED, per the library-wide contract.

### Consumed from other modules

- **Granger-causality tooling (multivariate)** — pairwise/block Wald Granger tests, Toda-Yamamoto, Breitung-Candelon frequency-domain causality, Geweke feedback measures, time-varying (Shi-Phillips-Hurn) and nonparametric (Diks-Panchenko) variants are owned by the multivariate module. This module re-exports them in its diagnostics namespace for discoverability and wires them into `check_series()` companion workflows; the recursive-Wald and bootstrap machinery they need comes from this module's rolling-statistics kernel and the foundations bootstrap engine.
- **Univariate ARIMA engine** — Chen-Liu outlier detection and regARIMA pre-adjustment estimate ARIMA models in their inner loops.

### Exposed to other modules

- **Residual diagnostic batteries** (portmanteau, normality, ARCH-LM, stability) as typed, fitted-model-aware objects — consumed by univariate, multivariate, and volatility modules for their `.diagnose()` methods.
- **STL/MSTL** (module-owned single implementation) — consumed by forecasting (STL-based methods), outlier screens, and feature extraction.
- **X-13ARIMA-SEATS wrapper** and the unified SA quality diagnostics — consumed by nowcasting and official-statistics workflows.
- **Differencing advisors and seasonality tests** — consumed by auto-ARIMA/model-selection logic in the forecasting module.
- **Fast fractional differencing** — consumed by ARFIMA estimation and fractional cointegration elsewhere.
- **Rolling/recursive statistics engine and unified IC infrastructure** — shared kernels for monitoring, GSADF-style recursions, and cross-model IC comparison.
- **Feature extraction (tsfeatures/catch22)** — consumed by the ML module for model triage.
- **check_series()** — the library's front door; downstream modules register recommendation rules with it.

## Validation gallery

Golden targets this module must reproduce before claiming correctness:

- **Bai-Perron 2003 US ex-post real interest rate application** — exact break dates, sup-F sequence, and break-date confidence intervals vs `strucchange::breakpoints` and the published JAE tables.
- **Johansen-Juselius 1990 Danish money demand** — trace and max-eigenvalue statistics and rank decision vs `urca::ca.jo` and Stata `vecrank`, all five deterministic cases.
- **Perron's Nelson-Plosser reanalysis (Perron 1997)** — break-date and test-statistic tables reproduced series by series.
- **Phillips-Shi-Yu 2015 S&P 500 bubble episodes** — GSADF statistics and BSADF date-stamped episodes vs R exuber/psymonitor.
- **MacKinnon 2010 response surfaces** — published surface coefficients and implied p-values for ADF/Engle-Granger matched to reference precision.
- **R stl() bit-level parity** — STL components on canonical series (co2, nottem) matching the netlib/R implementation exactly.
- **Census X-13 test series** — spec generation and parsed M/Q, sliding-spans, and QS diagnostics matching R `seasonal` on the Census test suite.
- **Hansen 1999 sunspot/GNP threshold applications** — bootstrap sup-LM p-values vs Hansen's Matlab/GAUSS outputs.
- **Díaz-Emparanza 2014 HEGY surfaces** — per-frequency critical values vs `uroot::hegy.test` and Gretl, quarterly and monthly.
- **Kripfganz-Schneider 2020 ARDL bounds p-values** — surface-response p-values matching Stata `ardl` (first open-source reproduction).
- **Moritz-Bartz-Beielstein 2017 imputeTS benchmark** — Kalman imputation accuracy vs `imputeTS::na_kalman` on the published corpus.
- **Hamilton 2018 employment example** — regression-filter cycle vs the neverhpfilter replication.
- **Shimotsu Matlab ELW code** — exact local Whittle estimates on his published examples.
- **Kamber-Morley-Wong bnfiltering code** — real-time output-gap series matching the authors' R/Matlab implementation (Tier 4 gate).
- **Lo 1991 stock-return R/S results** — modified R/S statistics on the published CRSP examples.
- **catch22 C reference implementation** — feature values to machine precision (Lubba et al 2019).
