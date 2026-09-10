# Model card — The term structure of interest rates

**Family:** `nelson_siegel`, `svensson`, `dynamic_ns`, `acm_term_premium`, `jsz_fit`, `jsz_loadings`

Fitting, forecasting, and decomposing the yield curve. A cross-section of
yields at many maturities is summarized by a handful of interpretable factors —
level, slope, and curvature — through the Nelson-Siegel functional form;
Svensson adds a second curvature hump for richer long-end shapes; the dynamic
Nelson-Siegel (Diebold-Li) turns the static fit into a small forecasting model
by letting the factors evolve over time; and `acm_term_premium` is the
regression-based affine model (Adrian-Crump-Moench) that splits every fitted
yield into expected short rates and a **term premium** — the practitioner
standard the NY Fed's published ACM series is built on.

| Function | Role |
|----------|------|
| `nelson_siegel` | Three-factor (level/slope/curvature) curve fit |
| `svensson` | Four-factor extension with a second hump |
| `dynamic_ns` | Time series of NS factors + one-step curve forecast |
| `acm_term_premium` | Regression-based affine model: fitted vs risk-neutral yields, term premium |
| `jsz_fit` | Maximum-likelihood canonical affine model (Joslin-Singleton-Zhu): Q-eigenvalues, concentrated P-VAR, the same decomposition |
| `jsz_loadings` | The JSZ canonical bond-loading recursions at given parameters |

**AFNS, ACM or JSZ?** The [arbitrage-free Nelson-Siegel](afns.md)
(`afns_adjustment`) *restricts* the loadings to the Nelson-Siegel shapes and
adds a closed-form convexity term — reach for it when you want one curve
fitted or interpolated consistently with no-arbitrage. ACM leaves the
loadings free (estimated principal components) and prices the *time series*
of bond returns by regression — reach for it when the object of interest is
the **term premium** and you want it without an optimizer. JSZ (`jsz_fit`)
is the **maximum-likelihood** Gaussian affine model in its canonical form:
the risk-neutral dynamics carry only `N` ordered eigenvalues and one drift,
the loadings are *implied* by no-arbitrage (not estimated as principal
components), and the physical VAR is concentrated out by OLS. Reach for it
when you want the likelihood-based model the literature's replication code
passes around, the AFNS restriction tested rather than imposed, or the same
fitted / risk-neutral / term-premium decomposition as ACM from an estimator
that prices the cross-section exactly.

## What it estimates

- **`nelson_siegel(maturities, yields)`** — fits y(τ) = β₀ + β₁·f₁(τ) +
  β₂·f₂(τ) with the Nelson-Siegel loadings governed by a decay λ, giving the
  **level** (β₀), **slope** (β₁), and **curvature** (β₂) factors. With λ fixed
  the fit is linear (OLS on the loadings); `optimal_lambda=True` estimates λ by
  nonlinear least squares.
- **`svensson(maturities, yields)`** — the four-factor Svensson (1994) form,
  which nests Nelson-Siegel and adds a second curvature term with its own decay,
  letting the curve take a second hump at longer maturities. Here the two decays
  `lambda1`, `lambda2` are supplied.
- **`dynamic_ns(panel, maturities)`** — the Diebold-Li (2006) dynamic
  Nelson-Siegel: fit the three NS factors at *each* date in a T×n_maturities
  panel, treat the resulting factor series as the state, fit an AR(1) to each,
  and produce a one-step-ahead forecast of the factors and hence of the whole
  curve.
- **`acm_term_premium(yields, maturities)`** — the Adrian-Crump-Moench (2013)
  three-step estimator of a Gaussian affine term-structure model, entirely by
  OLS: (1) principal-component factors from the yield panel and a factor
  VAR(1) `X_{t+1} = μ + Φ X_t + v_{t+1}`; (2) one-period holding excess
  returns `rx_{t+1}(n) = p_{t+1}(n−1) − p_t(n) − r_t` regressed on a constant,
  the lagged factors, and the contemporaneous innovations, `rx = a + c'X +
  β'v + e`; (3) the convexity-adjusted prices of risk `λ₀ = (β'β)⁻¹β'(a +
  ½(B* vec(Σ) + σ²1))`, `λ₁ = (β'β)⁻¹β'c`. Affine recursions `A_n, B_n`
  (seeded at `A₁ = −δ₀, B₁ = −δ₁` from the short-rate regression) then price
  the whole curve twice — with the estimated λ's (**fitted** yields) and with
  λ = 0 (**risk-neutral** yields, the expected-short-rate component) — and
  the **term premium** is their difference at every date and maturity.

- **`jsz_fit(yields, maturities)`** — the Joslin-Singleton-Zhu (2011)
  canonical Gaussian dynamic term-structure model by maximum likelihood.
  A latent state `X_t` (`N` factors) follows, under the risk-neutral
  measure, `X_{t+1} = K0^Q + K1^Q X_t + Σ_X ε` with the **canonical**
  normalization `K0^Q = (k_∞^Q, 0, …, 0)'`, `K1^Q = J(λ^Q)` (the real Jordan
  form of the ordered eigenvalues `λ_1 ≥ … ≥ λ_N`; equal neighbours form a
  Jordan block) and short rate `r_t = ι'X_t`; bond prices follow the
  Riccati recursions `A_{n+1} = A_n + K0^Q'B_n + ½B_n'Σ_X B_n`,
  `B_{n+1} = K1^Q'B_n − ι`, yields `y_t^{(n)} = −(A_n + B_n'X_t)/n`. The
  state is rotated onto `N` observed portfolios `P_t = W y_t` (the first `N`
  principal-component loadings by default, or any full-row-rank `w`) that
  are priced **without error**; the remaining `M − N` yield directions carry
  iid error `σ_e`. JSZ's insight is that the likelihood then *factors*:
  `f(y_t | y_{t−1}) = f^P(P_t | P_{t−1}) × f^Q(y_t | P_t)`, and the P-measure
  VAR(1) `(μ_P, Φ_P)` appears only in the first factor, whose maximizer is
  OLS for **any** `Σ_P` — so it is concentrated out exactly (bit-for-bit
  statsmodels `VAR(1)`). `k_∞^Q` and `σ_e` are profiled analytically, and the
  numerical search runs only over `λ^Q` and the Cholesky factor of `Σ_P`,
  from JSZ's recommended start (the eigenvalues of the OLS feedback matrix)
  plus seeded perturbations. The same recursion run with `(μ_P, Φ_P)` in
  place of the Q dynamics gives the **risk-neutral** yields, and
  `term_premium = fitted − risk_neutral` — exactly `acm_term_premium`'s
  convention, so the two premia are directly comparable.
- **`jsz_loadings(lambda_q, k_inf_q, sigma_x, maturities)`** — the
  recursions above at given parameters in the literal canonical form (a
  Jordan block wherever two consecutive `λ^Q` are exactly equal), returning
  the per-maturity yield coefficients `a_x`, `b_x`. The AFNS loadings are the
  special case `λ^Q = (1, e^{−λ}, e^{−λ})`.

## Assumptions

- **The curve is smooth and low-dimensional.** Nelson-Siegel imposes exactly
  one hump; three factors explain the cross-section. Curves with multiple humps
  or sharp kinks (segmented markets, distressed short ends) are misfit — that is
  when you move to Svensson or a spline.
- **`nelson_siegel` at fixed λ is linear**; `optimal_lambda=True` makes it a 1-D
  nonlinear search over λ, which is well-behaved but can settle on a local
  optimum for unusual curves. `dynamic_ns` uses a **fixed** decay (default
  0.0609, the Diebold-Li monthly value) so the per-date fits stay linear and
  comparable across time.
- **Svensson can be weakly identified** when the two decays are close: the two
  curvature terms become collinear and the factor split is unstable. Choose
  `lambda1`, `lambda2` well apart.
- **`dynamic_ns` forecasts assume the factors follow independent AR(1)s** — a
  deliberately simple, robust dynamic. It is a reduced-form forecast, not an
  arbitrage-free affine model; it says nothing about risk premia.
- Maturities and yields must be aligned and in consistent units (the examples
  use years and percent). At least as many maturities as factors are needed to
  identify the fit.
- **`acm_term_premium` has its own unit contract**: yields are *annualized,
  continuously-compounded zero-coupon log yields in decimal* (0.05, not 5.0)
  and maturities are *integer periods* (months for monthly data) containing 1,
  with `n − 1` present for every excess-return maturity `n`. It assumes the
  factor VAR(1) is stationary, prices of risk are affine in the factors, and
  return pricing errors are homoskedastic (the pooled σ² of the paper).
- **`jsz_fit` shares the unit contract** (annualized decimal yields, integer
  periods) but does not need the one-period maturity. It assumes Gaussian
  dynamics with real, ordered Q-eigenvalues (the JSZ canonical form; complex
  Q-eigenvalues are not supported — the P-feedback matrix may have them),
  that the `N` portfolios are priced *exactly* (the choice of the portfolio
  *space* is an assumption; the basis within it is a normalization the fit
  is invariant to), and iid homoskedastic pricing errors on the other
  `M − N` directions. Nothing constrains `λ_1 < 1`: the recursions are
  evaluated by recurrence, so a unit or slightly explosive Q level factor
  is estimated and reported rather than clipped.

## When to use

- **`nelson_siegel`** — the default curve summary: three numbers that
  economists read directly (level ≈ long rate, slope ≈ short minus long,
  curvature ≈ medium-term hump), and a clean way to interpolate/smooth a noisy
  quoted curve.
- **`svensson`** — central-bank-style fitting (the ECB and others publish
  Svensson parameters) when the long end needs a second hump the three-factor
  form cannot capture.
- **`dynamic_ns`** — when you want to *forecast* the curve, decompose its
  historical movements into level/slope/curvature dynamics, or build a
  factor-based trading or risk signal.
- **`acm_term_premium`** — when the question is "how much of the 10-year
  yield is expected policy rates, and how much is risk compensation?": term
  premium estimation for policy analysis, bond-return predictability work,
  and any exercise that needs a risk-neutral (expectations) yield curve. Not
  the tool for fitting a single day's curve (use `nelson_siegel`/`svensson`)
  or for a no-arbitrage *cross-sectional* fit (use
  [`afns_adjustment`](afns.md)).
- **`jsz_fit`** — when you want the *maximum-likelihood* affine model:
  loadings implied by no-arbitrage rather than estimated as principal
  components, the Q-eigenvalues themselves (the persistence of the level
  factor under Q is the number JSZ, Bauer-Rudebusch-Wu and the shadow-rate
  literature argue about), a likelihood value for model comparison (`llf`),
  or a term premium from an estimator that prices the cross-section exactly.
  Use `jsz_loadings` to evaluate the canonical recursions at parameters of
  your own — e.g. the AFNS eigenvalue pattern — without fitting anything.
  Not the tool when the P-dynamics are the point (they are the OLS VAR of
  the portfolios, exactly as in ACM) or when you want standard errors on
  the Q parameters (see the failure modes).

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `nelson_siegel` | `decay` | `0.0609` | fixed λ when `optimal_lambda=False` |
| | `optimal_lambda` | `False` | `True` estimates λ by NLS |
| `svensson` | `lambda1`, `lambda2` | — (required) | the two decay parameters; keep them well separated |
| `dynamic_ns` | `decay` | `0.0609` | fixed λ used for every per-date fit |
| `acm_term_premium` | `n_factors` | `5` | ACM's baseline: five principal components of the yield panel |
| | `periods_per_year` | `12.0` | monthly maturities; use 4 for quarterly. Converts annualized yields to the per-period log yields the recursions price |
| `jsz_fit` | `n_factors` | `3` | JSZ's baseline; the number of exactly-priced portfolios, `1 ≤ N < M` |
| | `periods_per_year` | `12.0` | as for ACM; `lambda_q`/`k_inf_q` are reported per period |
| | `w` | `None` | portfolio weights (`N × M`); `None` = first `N` PCA loadings. Only the row space matters |
| | `n_starts` | `5` | start 0 is JSZ's recommendation (OLS eigenvalues + OLS covariance); starts 1.. redraw the eigenvalue pattern. On the 1990-2007 GSW panel the surface has three basins and the JSZ start alone lands in the worst (llf 8747 vs 8934) |
| | `seed` | `None` (→ 0) | seeds the perturbed starts (`tsecon_rng`); `None` means seed 0, not fresh entropy (the returned `seed` key is the value used); raises if passed with `n_starts=1`, where it would be inert |
| `jsz_loadings` | `periods_per_year` | `1.0` | only rescales the intercept `a_x`; parameters stay per period |

## How to read the output

- **`nelson_siegel`** → `{"level", "slope", "curvature", "factors", "lambda",
  "residuals", "rsquared"}`. `factors` is `[level, slope, curvature]`; `lambda`
  is the decay actually used (the NLS estimate when `optimal_lambda=True`).
  `rsquared` near 1 means the three-factor form captured the curve.
- **`svensson`** → `{"factors", "lambda1", "lambda2", "residuals",
  "rsquared"}`; `factors` has the four β's.
- **`dynamic_ns`** → `{"maturities", "lambda", "factors", "rsquared", "level",
  "slope", "curvature", "forecast"}`. `factors` is T×3 (and `level`/`slope`/
  `curvature` are its columns as separate series); `rsquared` is the per-date
  fit. `forecast` is a dict with the one-step-ahead `factors`, the implied
  `yields` at each maturity, and the fitted `ar1_intercept`/`ar1_phi` of the
  factor AR(1)s.
- **`acm_term_premium`** → the decomposition `fitted`, `risk_neutral`,
  `term_premium` (each T×M, annualized decimal, with `fitted = risk_neutral +
  term_premium` exactly); the model pieces `factors`, `factor_loadings`,
  `mu`/`phi`/`sigma` (the VAR), `a`/`beta`/`c`/`sigma2` (the excess-return
  regressions at `rx_maturities`), `lambda0`/`lambda1` (prices of risk),
  `delta0`/`delta1` (the short rate), and the recursion coefficients
  `A`/`B`/`A_rn`/`B_rn`; plus diagnostics `var_rsquared`, `rx_rsquared`
  (high — the contemporaneous innovations absorb most return variation),
  `short_rate_rsquared`, and per-maturity `yield_rsquared` (should be ≈1 for
  a smooth curve panel); plus the echoed inputs `maturities`, `n_factors`,
  `periods_per_year`. A positive `term_premium` says investors are paid to
  hold duration; a negative one (post-2015 US data, per the published ACM
  series) says they pay for it.
- **`jsz_fit`** → the Q parameters `lambda_q` (ordered, per period; `λ_1`
  near 1 is the persistent level factor — on monthly GSW 1990-2007 it is
  0.9965, a 16-year half-life) and `k_inf_q`; `sigma` (the MLE innovation
  covariance of the portfolio VAR) and `sigma_e` (the pricing-error standard
  deviation of the non-portfolio directions; 2.7bp on that panel); the OLS
  P-VAR `mu_p`, `phi_p` with statsmodels-convention `mu_p_se`, `phi_p_se`
  and `sigma_ols` (`sigma_u_mle`); the Q-VAR in the portfolio rotation
  `k0_q_p`, `k1_q_p` and the ACM-unit prices of risk `lambda0 = mu_p −
  k0_q_p`, `lambda1 = phi_p − k1_q_p`; loadings `a_p`, `b_p`
  (`fitted = a_p + b_p P`, with `w b_p = I`, `w a_p = 0`) and, for the
  literal canonical latent state, `a_x`, `b_x`; the decomposition `fitted`,
  `risk_neutral`, `term_premium` (`T × M`, exact) and `rmse` per maturity;
  `factors` (`P_t`) and `w`; `llf` (basis-invariant; equals the JSZ
  replication code's `llkP + llkQ` for an orthonormal `w`), `converged`,
  `n_iter`; the echoed `maturities`, `n_factors`, `periods_per_year`,
  `n_starts`, `seed`. **No standard errors for the Q parameters** — see
  below.
- **`jsz_loadings`** → `a_x`, `b_x` (per maturity), `k0_q`, `k1_q` (the
  literal `J(λ^Q)`, showing the Jordan blocks), `maturities`.

## Failure modes

- **Forcing three factors on a multi-hump curve.** A poor `nelson_siegel`
  `rsquared` (well below ~0.99 for a normal government curve) signals the form
  is too rigid; switch to `svensson`.
- **Svensson decay collinearity.** `lambda1 ≈ lambda2` makes the two curvature
  factors nearly identical and the estimated β's wild even at high R²; separate
  the decays.
- **Over-reading `optimal_lambda`.** The NLS λ can jump between local optima
  across dates, making the factor series jittery — for time series work prefer
  the fixed-λ `dynamic_ns`, which is designed for exactly that comparability.
- **Extrapolating beyond the fitted maturities.** Nelson-Siegel behaves
  smoothly but the long-end asymptote is driven entirely by the level factor;
  do not trust yields far outside the quoted maturity range.
- **AR(1) forecast on a trending factor.** If the level factor is very
  persistent (near unit root), the AR(1) one-step forecast is fine but
  multi-step extrapolation (not provided here) would be unreliable.
- **Feeding `acm_term_premium` percent instead of decimal.** The Jensen
  convexity terms are quadratic while everything else is linear, so percent
  input misprices them by a factor of 100 — it does *not* just rescale the
  answer. Divide by 100 first.
- **Reading the ACM premium's level as sample-free truth.** The prices of
  risk are estimated mean excess returns; re-estimating on a subsample moves
  the premium's *level* substantially while its *shape* barely moves (on
  1983-2014 alone the 10-year premium sits ~1.1pp above the full-sample
  estimate at the same 0.97+ correlation). Compare premia only across models
  estimated on the same sample — and expect published vintages to differ.
- **Too few excess-return maturities.** `λ₀`/`λ₁` come from a cross-sectional
  regression on β (N×K), so you need strictly more return maturities than
  factors — with `n_factors=5`, at least six `(n−1, n)` pairs in the grid.
- **Reading `jsz_fit`'s prices of risk as precise.** The likelihood is *flat*
  in the market prices of risk: `lambda_q` and `k_inf_q` are pinned by the
  cross-section (hundreds of pricing equations per date — recovered to
  ~1e-4 in simulation), but `mu_p`/`phi_p` come from a `T`-observation VAR
  of very persistent factors, and `lambda0`/`lambda1` (and the *level* of the
  term premium) inherit that imprecision. That is why `mu_p_se`/`phi_p_se`
  are reported. No standard errors are reported for the Q parameters: a
  numerical Hessian of the profile likelihood at a near-unit-root optimum is
  not an honest asymptotic covariance, and a precise-looking number that is
  not would be worse than none.
- **A single start on real data.** The JSZ start (OLS eigenvalues) sits at a
  near-tie of two eigenvalues whenever the P-feedback matrix has a complex
  pair — on GSW 1990-2007 that start alone ends in a local optimum 186
  log-likelihood points below the best of three basins. Keep `n_starts ≥ 5`
  (the default); compare `llf` across seeds if in doubt.
- **The near-unit-root level factor.** `λ_1` sits at 0.9965 (monthly) on
  1990-2007 GSW; the recursions handle `λ_1 = 1` and above exactly, and the
  profile of `k_inf_q` stays identified there (it becomes the drift of a
  unit-root level), but the *long-run Q mean* `k_inf/(1−λ_1)` is not a number
  to quote. An estimate of `λ_1` above 1 means explosive risk-neutral
  dynamics — a statement about the sample, reported rather than hidden.
- **Different portfolio spaces are different models.** Two bases of the same
  space (`w` and `G w`) give identical results; portfolios spanning a
  different space (say three specific yields instead of three PCs) change
  which yields are priced exactly and give a different — usually very
  close — fit.

## Validated against

`nelson_siegel` and `svensson` are validated as OLS-at-fixed-λ (and the NLS λ
search) against a documented reference, and `dynamic_ns` reproduces the
Diebold-Li (2006) per-date fits and AR(1) factor dynamics. Golden values are
pinned in [`fixtures/termstructure.json`](../../../fixtures/termstructure.json).

`acm_term_premium` is validated three ways
([`fixtures/acm.json`](../../../fixtures/acm.json), produced by
[`fixtures/generate_acm_fixtures.py`](../../../fixtures/generate_acm_fixtures.py),
which builds the entire pipeline independently in NumPy and never calls
tsecon):

- **Documented-formula golden** — every pipeline quantity (factors, VAR,
  `a`/`β`/`c`, `λ₀`/`λ₁`, recursions, fitted/risk-neutral/term-premium paths)
  reproduces the NumPy transcription to 1e-8, on both a simulated affine DGP
  and the real 1961-2014 monthly GSW zero-coupon panel
  ([`fixtures/gsw_nss_params.csv`](../../../fixtures/gsw_nss_params.csv),
  Federal Reserve Board data, vendored with attribution).
- **Recovery on a known-truth DGP** — with known prices of risk, the
  estimated 5-year premium tracks the true premium at correlation 0.98 (mean
  over 30 Monte-Carlo draws; minimum 0.93) with mean absolute error 22bp
  against a ~367bp premium.
- **The NY Fed's published ACM series** — on the same 1961-2014 GSW panel,
  the estimated 10-year premium matches the published `ACMTP10` (2021
  vintage, quarterly, 212 overlapping quarters;
  [`fixtures/acm_published_10y.csv`](../../../fixtures/acm_published_10y.csv))
  with correlation **0.985**, mean gap **−0.10pp**, RMSE **0.31pp**, and the
  fitted 10-year yield matches `ACMY10` at correlation 0.99999 (RMSE 1.3bp) —
  despite our raw-GSW short rate (the Fed splices the federal funds rate
  before 1982). A level/shape validation with vintage caveats, not a
  bit-exact golden.

`jsz_fit` / `jsz_loadings` are validated in four blocks
([`fixtures/jsz.json`](../../../fixtures/jsz.json), produced by
[`fixtures/generate_jsz_fixtures.py`](../../../fixtures/generate_jsz_fixtures.py),
which never calls tsecon; Rust tests
[`jsz_golden.rs`](../../../crates/tsecon-termstructure/tests/jsz_golden.rs)
and [`jsz_properties.rs`](../../../crates/tsecon-termstructure/tests/jsz_properties.rs)):

- **Documented-formula golden of the recursions** — `jsz_loadings`
  reproduces a NumPy transcription of the Riccati recursions at 1e-12 on
  four stated parameter sets (distinct eigenvalues, the AFNS Jordan block,
  `N = 2`, `N = 4` with an interior tie), and the Jordan-block column against
  its closed form `(1/n) Σ_{j<n} (j ρ^{j−1} + ρ^j)` at 1e-12.
- **The AFNS special case, pinned to the crate's own closed form** — at
  `λ^Q = (1, e^{−λΔ}, e^{−λΔ})` the JSZ yield loadings span the Nelson-Siegel
  loadings *exactly* (a stored 3×3 rotation, residual ≤ 1.6e-14 at every
  period length), and the discrete convexity intercept converges to the
  independent Christensen-Diebold-Rudebusch closed form of
  `afns_adjustment` at **first order in the period length**: max gap
  9.8e-6 at Δ = 1/12, 2.5e-6 at 1/48, 6.1e-7 at 1/192 (ratios 0.2504,
  0.2501). Discrete-time JSZ and continuous-time AFNS differ by a Riemann
  sum, so an exact pin would be wrong; the measured rate is the honest one.
- **Independent-package golden of the concentrated step** — `mu_p`,
  `phi_p`, their standard errors and `sigma_ols` reproduce statsmodels
  `VAR(1)` at 1e-9 on the simulated panel and on GSW.
- **Documented-formula golden of the likelihood** — `jsz_loglik` (Rust)
  reproduces the transcribed `llk_P + llk_Q + Jacobian` at the true
  parameters, in a non-orthonormal basis of the same portfolio space (the
  invariance formula asserted in the generator), and at a second stated
  point, at 1e-7 absolute on values of order 3e4.
- **Simulation recovery and a cross-optimizer target** — on a simulated
  canonical model (`T = 500`, 10 maturities, 3 factors, portfolios priced
  exactly, `σ_e = 0.2bp` elsewhere) the MLE recovers `λ^Q` to **8.7e-5** (max
  abs error), `k_∞^Q` to **1.2%**, `σ_e` to **0.3%**, and `Σ_P` to 13.6%
  (max-entry relative — the same 13.6% the OLS covariance is off by:
  sampling error at `T = 500`, not estimation). The Rust MLE and a SciPy
  multi-start MLE on the same likelihood agree to 1e-5 in `λ^Q`, 1e-3 in
  `k_∞^Q`, 1e-4 in `σ_e`; the Rust property suite adds an independent Rust-
  simulated DGP (`λ^Q` within 1.8e-4 at `σ_e = 0.1bp`), invariance to the
  basis of the portfolio space (llf within 1.1e-11, `λ^Q` 1.6e-10, fitted
  yields < 1e-8; `Σ_P`, the flattest direction, 2.1e-7), exact pricing of
  the portfolios, determinism, and local optimality of the maximized
  likelihood in every parameter direction through `jsz_loglik`.
- **Real data** — the GSW zero-coupon panel 1990-01..2007-12 (JSZ's own
  window; 216 months, maturities 6m-10y, three PCA portfolios): the PCA
  weights pinned to NumPy (1e-10), the VAR to statsmodels (1e-9), the MLE
  to SciPy's multi-start optimum (`λ^Q` 1e-5), and the illustration numbers
  reproduced: `λ^Q = (0.9965, 0.9624, 0.9092)`, `k_∞^Q = 3.27e-5`/month,
  `σ_e = 2.69bp`, RMSE 1.35-2.86bp per maturity, mean 10-year term premium
  **2.30pp** (0.18-4.84pp; 4.10pp in Jan 1990, 1.11pp in Dec 2007). The
  surface is multimodal there — three basins at llf 8933.8 / 8917.2 /
  8747.4, the JSZ start alone in the last — which the seeded multi-start
  is for. There is no published JSZ estimate on exactly this panel to pin
  to, so the real-data leg is an *illustration with a cross-optimizer
  check*, not a literature golden.

## References

- Nelson, C. & Siegel, A. (1987). "Parsimonious Modeling of Yield Curves."
  *J. Business* 60.
- Svensson, L. (1994). "Estimating and Interpreting Forward Interest Rates:
  Sweden 1992-1994." NBER WP 4871.
- Diebold, F. & Li, C. (2006). "Forecasting the term structure of government
  bond yields." *J. Econometrics* 130.
- Diebold, F., Rudebusch, G. & Aruoba, B. (2006). "The macroeconomy and the
  yield curve: a dynamic latent factor approach." *J. Econometrics* 131.
- Adrian, T., Crump, R. K. & Moench, E. (2013). "Pricing the Term Structure
  with Linear Regressions." *J. Financial Economics* 110(1). (FRBNY Staff
  Report 340; the published series lives at the NY Fed's "Treasury Term
  Premia" data page.)
- Gürkaynak, R., Sack, B. & Wright, J. (2007). "The U.S. Treasury Yield
  Curve: 1961 to the Present." *J. Monetary Economics* 54(8).
- Joslin, S., Singleton, K. J. & Zhu, H. (2011). "A New Perspective on
  Gaussian Dynamic Term Structure Models." *Review of Financial Studies*
  24(3), 926-970.
- Dai, Q. & Singleton, K. J. (2000). "Specification Analysis of Affine Term
  Structure Models." *J. Finance* 55(5).

See the guide: [The Term Structure of Interest Rates](../../guide/15-term-structure.md).

## Runnable example

```python
import numpy as np
import tsecon

# maturities in years, yields in percent
mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
ylds = 4.0 - 1.5 * np.exp(-0.5 * mats) + 0.8 * (1 - np.exp(-0.5 * mats)) / (0.5 * mats)

# 1. Nelson-Siegel: three interpretable factors (level, slope, curvature).
ns = tsecon.nelson_siegel(mats, ylds, optimal_lambda=True)
print("NS level/slope/curvature:",
      round(ns["level"], 3), round(ns["slope"], 3), round(ns["curvature"], 3),
      " lambda:", round(ns["lambda"], 4), " R^2:", round(ns["rsquared"], 4))

# 2. Svensson: adds a second hump for richer long-end shapes (lambdas fixed).
sv = tsecon.svensson(mats, ylds, lambda1=0.6, lambda2=0.1)
print("Svensson 4 factors:", np.round(sv["factors"], 3), " R^2:", round(sv["rsquared"], 4))

# 3. Dynamic Nelson-Siegel over a T x n_maturities panel of curves.
T = 80
L = 4 + 0.3 * np.cumsum(np.random.default_rng(11).standard_normal(T)) * 0.1
panel = np.empty((T, len(mats)))
for t in range(T):
    panel[t] = (L[t] - 1.5 * np.exp(-0.5 * mats)
                + 0.8 * (1 - np.exp(-0.5 * mats)) / (0.5 * mats)
                + 0.02 * np.random.default_rng(100 + t).standard_normal(len(mats)))
dns = tsecon.dynamic_ns(panel, mats)
print("DNS factor series shape:", np.asarray(dns["factors"]).shape,
      " next-period yield forecast:", np.round(dns["forecast"]["yields"][:3], 3), "...")
```

Expected output:

```
NS level/slope/curvature: 4.0 -0.7 1.5  lambda: 0.5  R^2: 1.0
Svensson 4 factors: [ 3.819 -0.549  1.452  0.685]  R^2: 0.9981
DNS factor series shape: (80, 3)  next-period yield forecast: [3.616 3.645 3.701] ...
```

### ACM term premium

```python
import numpy as np
import tsecon

# Monthly zero-coupon panel, maturities 1..60 months, yields in DECIMAL.
# (Here: a persistent two-factor curve simulation; use your own panel.)
rng = np.random.default_rng(7)
T, mats = 300, np.arange(1, 61)
level, slope = 0.04, 0.01
rows = []
for t in range(T):
    level += 0.001 * rng.standard_normal() - 0.02 * (level - 0.04)
    slope += 0.0012 * rng.standard_normal() - 0.10 * (slope - 0.01)
    curve = level + slope * (1 - np.exp(-mats / 24.0)) - 0.005 * np.exp(-mats / 24.0)
    rows.append(curve + 2e-5 * rng.standard_normal(len(mats)))

acm = tsecon.acm_term_premium(np.array(rows), list(mats), n_factors=3)

tp = np.asarray(acm["term_premium"])       # T x M, annualized decimal
fit = np.asarray(acm["fitted"])
rn = np.asarray(acm["risk_neutral"])
j5y = list(acm["maturities"]).index(60)
print("5y fitted / risk-neutral / premium (last date, %):",
      round(fit[-1, j5y] * 100, 2), "/", round(rn[-1, j5y] * 100, 2),
      "/", round(tp[-1, j5y] * 100, 2))
print("lambda0:", np.round(acm["lambda0"], 3))
print("mean 5y premium (%):", round(tp[:, j5y].mean() * 100, 2),
      " yield R^2 at 5y:", round(acm["yield_rsquared"][j5y], 4))
print("decomposition exact:", np.allclose(fit, rn + tp))
```

Expected output:

```
5y fitted / risk-neutral / premium (last date, %): 5.25 / 3.89 / 1.36
lambda0: [-0.151  0.423  0.337]
mean 5y premium (%): 1.29  yield R^2 at 5y: 1.0
decomposition exact: True
```

### JSZ canonical affine term structure

```python
import numpy as np
import tsecon

# A simulated monthly panel from a known JSZ canonical model: three
# orthonormal portfolios follow a VAR(1), yields are their exact affine
# prices plus a 1bp error orthogonal to the portfolios. (Use your own panel.)
mats = [1, 3, 6, 12, 24, 36, 60, 84, 120]
zero = np.zeros((3, 3))
lam_true, kinf_true = np.array([0.995, 0.96, 0.85]), 2e-5
b_x = np.asarray(tsecon.jsz_loadings(lam_true, 0.0, zero, mats, periods_per_year=12.0)["b_x"])
raw = np.array([np.ones(9), np.array(mats) / 120.0, np.array(mats) / 24.0 * np.exp(-np.array(mats) / 24.0)])
w, _ = np.linalg.qr(raw.T); w = w.T                      # orthonormal rows
d_inv = np.linalg.inv(w @ b_x)
chol = np.array([[0.0022, 0, 0], [0.0006, 0.0012, 0], [-0.0002, 0.0003, 0.0008]])
sigma_x = d_inv @ (chol @ chol.T / 144.0) @ d_inv.T
a_x = np.asarray(tsecon.jsz_loadings(lam_true, kinf_true, sigma_x, mats, periods_per_year=12.0)["a_x"])
b_p, a_p = b_x @ d_inv, a_x - b_x @ d_inv @ (w @ a_x)
mu, phi = np.array([0.0025, 0.0004, -0.0001]), np.array([[0.98, 0.01, -0.01], [0.005, 0.93, 0.02], [0, -0.01, 0.85]])
rng = np.random.default_rng(3)
p = np.linalg.solve(np.eye(3) - phi, mu)
rows = []
for t in range(400):
    p = mu + phi @ p + chol @ rng.standard_normal(3)
    e = 1e-4 * rng.standard_normal(9)
    rows.append(a_p + b_p @ p + e - w.T @ (w @ e))
y = np.array(rows)

fit = tsecon.jsz_fit(y, mats, n_factors=3, periods_per_year=12.0)
print("lambda_q:", np.round(fit["lambda_q"], 4), " true:", lam_true)
print("k_inf_q: %.2e  (true %.2e)   sigma_e: %.1f bp   converged: %s" %
      (fit["k_inf_q"], kinf_true, fit["sigma_e"] * 1e4, fit["converged"]))
tp = np.asarray(fit["term_premium"])
print("mean 10y term premium (pp): %.2f   llf: %.1f" % (tp[:, -1].mean() * 100, fit["llf"]))
print("phi_p eigenvalues:", np.round(np.sort(np.linalg.eigvals(fit["phi_p"]).real)[::-1], 3),
      " vs Q:", np.round(fit["lambda_q"], 3))
```

Expected output:

```
lambda_q: [0.995  0.96   0.8496]  true: [0.995 0.96  0.85 ]
k_inf_q: 1.99e-05  (true 2.00e-05)   sigma_e: 1.0 bp   converged: True
mean 10y term premium (pp): 1.64   llf: 24978.5
phi_p eigenvalues: [0.979 0.903 0.903]  vs Q: [0.995 0.96  0.85 ]
```
