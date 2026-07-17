# Model card — The term structure of interest rates

**Family:** `nelson_siegel`, `svensson`, `dynamic_ns`

Fitting and forecasting the yield curve. A cross-section of yields at many
maturities is summarized by a handful of interpretable factors — level, slope,
and curvature — through the Nelson-Siegel functional form; Svensson adds a
second curvature hump for richer long-end shapes; and the dynamic Nelson-Siegel
(Diebold-Li) turns the static fit into a small forecasting model by letting the
factors evolve over time. These are the standard, transparent tools before one
reaches for full affine (JSZ/ACM) models.

| Function | Role |
|----------|------|
| `nelson_siegel` | Three-factor (level/slope/curvature) curve fit |
| `svensson` | Four-factor extension with a second hump |
| `dynamic_ns` | Time series of NS factors + one-step curve forecast |

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

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `nelson_siegel` | `decay` | `0.0609` | fixed λ when `optimal_lambda=False` |
| | `optimal_lambda` | `False` | `True` estimates λ by NLS |
| `svensson` | `lambda1`, `lambda2` | — (required) | the two decay parameters; keep them well separated |
| `dynamic_ns` | `decay` | `0.0609` | fixed λ used for every per-date fit |

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

## Validated against

`nelson_siegel` and `svensson` are validated as OLS-at-fixed-λ (and the NLS λ
search) against a documented reference, and `dynamic_ns` reproduces the
Diebold-Li (2006) per-date fits and AR(1) factor dynamics. Golden values are
pinned in [`fixtures/termstructure.json`](../../../fixtures/termstructure.json).

## References

- Nelson, C. & Siegel, A. (1987). "Parsimonious Modeling of Yield Curves."
  *J. Business* 60.
- Svensson, L. (1994). "Estimating and Interpreting Forward Interest Rates:
  Sweden 1992-1994." NBER WP 4871.
- Diebold, F. & Li, C. (2006). "Forecasting the term structure of government
  bond yields." *J. Econometrics* 130.
- Diebold, F., Rudebusch, G. & Aruoba, B. (2006). "The macroeconomy and the
  yield curve: a dynamic latent factor approach." *J. Econometrics* 131.

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
