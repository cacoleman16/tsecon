# Model card — The term structure of interest rates

**Family:** `nelson_siegel`, `svensson`, `dynamic_ns`, `acm_term_premium`

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

**AFNS or ACM?** The [arbitrage-free Nelson-Siegel](afns.md)
(`afns_adjustment`) *restricts* the loadings to the Nelson-Siegel shapes and
adds a closed-form convexity term — reach for it when you want one curve
fitted or interpolated consistently with no-arbitrage. ACM leaves the
loadings free (estimated principal components) and prices the *time series*
of bond returns — reach for it when the object of interest is the **term
premium**, the decomposition of a long yield into expected short rates and
risk compensation.

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

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `nelson_siegel` | `decay` | `0.0609` | fixed λ when `optimal_lambda=False` |
| | `optimal_lambda` | `False` | `True` estimates λ by NLS |
| `svensson` | `lambda1`, `lambda2` | — (required) | the two decay parameters; keep them well separated |
| `dynamic_ns` | `decay` | `0.0609` | fixed λ used for every per-date fit |
| `acm_term_premium` | `n_factors` | `5` | ACM's baseline: five principal components of the yield panel |
| | `periods_per_year` | `12.0` | monthly maturities; use 4 for quarterly. Converts annualized yields to the per-period log yields the recursions price |

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
