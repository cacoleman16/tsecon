# Model card — Arbitrage-free Nelson-Siegel (AFNS)

`afns_adjustment`

The plain Nelson-Siegel curve on the [term structure card](term-structure.md) is a
*reduced-form* factor model: three interpretable loadings — level, slope, and
curvature — fit the cross-section of yields beautifully, but nothing in the fit
forces those loadings to be consistent with the no-arbitrage restrictions of a
dynamic term-structure model. Left alone, a Nelson-Siegel curve can quote a set
of yields across maturities that a trader could, in principle, arbitrage: the
long end does not bend down by the amount that the volatility of the factors, and
Jensen's inequality, require. That gap is small at the short end and grows with
maturity, which is exactly where it matters for pricing.

Christensen, Diebold & Rudebusch (2011) show how to close the gap without
touching the part of the model you like. Keep all three Nelson-Siegel factor
loadings **unchanged**, and add a single deterministic, maturity-dependent
**yield-adjustment term** `−A(τ)/τ`. That one extra term — a function of the
factor volatilities and the decay, not of the fitted factors — is what makes the
curve arbitrage-free. `afns_adjustment` computes precisely this term on a
maturity grid; you add it to any Nelson-Siegel fit to obtain its arbitrage-free
(AFNS) companion. The adjustment is non-positive and deepens with maturity, so it
pulls long yields *down* relative to the reduced-form curve — the convexity
effect that reduced-form Nelson-Siegel omits.

The AFNS curve, for the independent-factor case (a diagonal factor-volatility
matrix `Σ = diag(σ₁₁, σ₂₂, σ₃₃)`), is

```text
y(tau) = L
       + S * (1 - e^{-lam tau}) / (lam tau)
       + C * [ (1 - e^{-lam tau}) / (lam tau) - e^{-lam tau} ]
       - A(tau) / tau
```

The first three terms are the ordinary Nelson-Siegel level/slope/curvature
loadings — **identical** to `nelson_siegel` on the [term structure
card](term-structure.md). AFNS adds only the last term, whose closed form (CDR
2011, independent-factor case) is

```text
A(tau)/tau =
    sigma_11^2 * ( tau^2 / 6 )
  + sigma_22^2 * [ 1/(2 lam^2)
                   - (1 - e^{-lam tau}) / (lam^3 tau)
                   + (1 - e^{-2 lam tau}) / (4 lam^3 tau) ]
  + sigma_33^2 * [ 1/(2 lam^2)
                   + e^{-lam tau} / lam^2
                   - (tau e^{-2 lam tau}) / (4 lam)
                   - 3 e^{-2 lam tau} / (4 lam^2)
                   - 2 (1 - e^{-lam tau}) / (lam^3 tau)
                   + 5 (1 - e^{-2 lam tau}) / (8 lam^3 tau) ]
```

`A(τ)/τ` is non-negative; through the `σ₁₁²·τ²/6` term it grows without bound in
maturity, so the *signed* adjustment `−A(τ)/τ` is negative and its magnitude
grows with maturity. As `Σ → 0` the adjustment vanishes and AFNS nests plain
Nelson-Siegel exactly.

---

## `afns_adjustment` — the arbitrage-free yield-adjustment term

**What it estimates.** Nothing statistical — it *evaluates* the
Christensen-Diebold-Rudebusch (2011) closed-form yield-adjustment term. Given a
grid of maturities `τ`, the three diagonal factor volatilities
`σ = [σ₁₁, σ₂₂, σ₃₃]` (level, slope, curvature factor vols), and the decay `λ`,
it returns the **signed** adjustment `−A(τ)/τ`, one value per maturity, that you
add to a reduced-form Nelson-Siegel curve to make it arbitrage-free:
`y_AFNS(τ) = y_NS(τ) + afns_adjustment(τ)`. It does not fit factors or estimate
`σ`; the volatilities are inputs you bring from a dynamic AFNS estimation, a
calibration, or a scenario.

**Assumptions.** The **independent-factor** AFNS: the factor-volatility matrix is
diagonal, `Σ = diag(σ₁₁, σ₂₂, σ₃₃)`, so the level, slope, and curvature factors
have uncorrelated innovations and the adjustment has the simple closed form
above. (The general correlated-factor AFNS has a longer expression with
cross-terms; this function does not cover it.) The three Nelson-Siegel loadings
are taken as given and unchanged — AFNS is a restriction *on top of* the
reduced-form loadings, not a re-parameterization of them. The decay `λ` is the
same `λ` that governs the loadings; maturities, `λ`, and the volatilities must be
in consistent time units (the examples use years).

**When to use (and when not).** Use it whenever you have a Nelson-Siegel fit and
want its arbitrage-free counterpart — pricing long-dated instruments, comparing a
reduced-form curve against a no-arbitrage one, or quantifying how much the
convexity correction moves the long end for a given level of factor volatility.
It is the arbitrage-free companion to the reduced-form curves on the [term
structure card](term-structure.md); reach for it exactly when the *no-arbitrage*
property matters and a plain Nelson-Siegel fit is not enough. Do **not** use it
as a fitting routine — it computes a deterministic term from volatilities you
supply, so if you do not have credible factor volatilities the adjustment is only
as meaningful as your `σ`. Do not use it when the factors are strongly correlated
(the diagonal-`Σ` closed form is then only an approximation). And do not read the
adjustment as a forecast or a risk premium — it is the deterministic convexity
term, nothing more.

**Key arguments and defaults (and why).** `maturities` is a 1-D array of
maturities `τ` in years (e.g. `[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]`); the
output has one value per entry. `sigma` is the three-element diagonal
`[σ₁₁, σ₂₂, σ₃₃]` — it must have **exactly three** elements (level, slope,
curvature factor vols) or the binding raises `ValueError`
(`"sigma must have 3 elements …"`), and each must be finite and non-negative
(`0` is allowed and switches off that factor's contribution). `decay = 0.0609` is
the Nelson-Siegel decay `λ` — note the keyword is spelled **`decay`, not
`lambda`**, because `lambda` is a reserved word in Python. The default `0.0609`
is the Diebold-Li (2006) monthly convention: with maturities measured in
**months** it places the peak of the curvature loading near the 30-month point
(verified below at ~29.5 months), the canonical choice for U.S. Treasury curves.
If your maturities are in **years**, pass the matching decay (the examples use
`decay = 0.5`); `decay` must be positive and finite.

**How to read the output.** A NumPy array the same length as `maturities`, giving
the signed adjustment `−A(τ)/τ` at each maturity. Every entry is `≤ 0`, and the
array is **monotonically decreasing** (more negative) with maturity — the
arbitrage-free concavity effect that pulls long yields down relative to
reduced-form Nelson-Siegel. You **add** it to a Nelson-Siegel curve:
`y_AFNS = y_NS + adjustment`. Near the short end the adjustment is negligible
(basis points of a basis point); it widens into a visible gap at 10, 20, 30
years, driven by the `σ₁₁²·τ²/6` level-vol term. As `σ → 0` every entry goes to
zero and `y_AFNS` collapses back onto `y_NS` — the plain Nelson-Siegel nesting.

**Failure modes.** Passing a `sigma` of the wrong length is the common one — it
must be exactly three (level, slope, curvature), not one per maturity; a
mismatch raises `ValueError` rather than silently broadcasting. Supplying `λ`
in the wrong time units (a monthly `0.0609` against maturities in years, or vice
versa) mis-scales the slope- and curvature-vol terms — keep `decay`, maturities,
and volatilities in one consistent unit. Treating the adjustment as a *fitted*
object: it is a deterministic function of your inputs, so garbage-in volatilities
give a garbage adjustment with no diagnostic to warn you. Assuming the diagonal
form when factors are strongly correlated understates or mis-shapes the true
adjustment. And reading a small short-end value as "the correction is
negligible" — it is negligible *there*; the whole point is the long end.

**Validated against.** A documented-formula golden,
[`fixtures/afns.json`](../../../fixtures/afns.json), produced by
[`fixtures/generate_afns_fixtures.py`](../../../fixtures/generate_afns_fixtures.py).
The generator is deliberately **non-circular**: it never calls tsecon — it
transcribes the CDR (2011) independent-factor closed form directly into NumPy and
evaluates it on a grid of `(maturities, λ, σ)` cases, recording both the positive
term `c_over_tau` (`A(τ)/τ`) and the signed `adjustment` (`−A(τ)/τ`). The Python
binding is golden-tested in
[`bindings/python/tests/test_spectest_afns_dsge.py`](../../../bindings/python/tests/test_spectest_afns_dsge.py):
`afns_adjustment` reproduces every fixture case's `adjustment` to `~1e-9`, the
output is checked non-positive (`≤ 0`) and monotonically non-increasing, and the
three-element `sigma` validation error is asserted. This is a transcribed-formula
check, not a statsmodels cross-comparison (there is no statsmodels AFNS routine).

**References.**

- Christensen, J. H. E., Diebold, F. X., & Rudebusch, G. D. (2011). "The affine
  arbitrage-free class of Nelson-Siegel term structure models."
  *Journal of Econometrics* 164(1):4-20. *(The AFNS class and the
  independent-factor closed-form yield-adjustment term computed here.)*
- Nelson, C. R., & Siegel, A. F. (1987). "Parsimonious Modeling of Yield
  Curves." *Journal of Business* 60(4):473-489. *(The reduced-form loadings AFNS
  leaves unchanged.)*
- Diebold, F. X., & Li, C. (2006). "Forecasting the term structure of government
  bond yields." *Journal of Econometrics* 130(2):337-364. *(The dynamic
  Nelson-Siegel and the `λ = 0.0609` monthly-decay convention.)*

See also the reduced-form curves on the [term structure card](term-structure.md).

```python
import numpy as np, tsecon

# A standard maturity grid (years) and plausible independent-factor vols
# [level, slope, curvature], with a years-consistent decay lambda = 0.5.
mats  = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
sigma = np.array([0.010, 0.008, 0.012])
lam   = 0.5

adj = tsecon.afns_adjustment(mats, sigma, decay=lam)   # signed -A(tau)/tau
print("adjustment -A(tau)/tau :", np.round(adj, 5))
print("non-positive           :", bool(np.all(adj <= 0.0)))
print("deepens with maturity  :", bool(np.all(np.diff(adj) <= 0.0)))

# Make a plain Nelson-Siegel curve arbitrage-free: y_AFNS = y_NS + adjustment.
L, S, C = 4.0, -1.5, 0.8                                # level, slope, curvature
b_slope = (1 - np.exp(-lam*mats)) / (lam*mats)
b_curv  = b_slope - np.exp(-lam*mats)
y_ns    = L + S*b_slope + C*b_curv
y_afns  = y_ns + adj
print("\n  tau   y_NS   y_AFNS   gap(bp)")
for t, a, b in zip(mats, y_ns, y_afns):
    print(f"{t:5.2f}  {a:5.3f}  {b:5.3f}   {(b-a)*100:6.2f}")
# adjustment -A(tau)/tau : [-0.000e+00 -1.000e-05 -2.000e-05 -9.000e-05 -2.000e-04 -5.300e-04
#  -9.800e-04 -1.890e-03 -6.980e-03 -1.535e-02]
# non-positive           : True
# deepens with maturity  : True
#
#   tau   y_NS   y_AFNS   gap(bp)
#  0.25  2.636  2.636    -0.00
#  0.50  2.758  2.758    -0.00
#  1.00  2.964  2.964    -0.00
#  2.00  3.263  3.263    -0.01
#  3.00  3.459  3.459    -0.02
#  5.00  3.677  3.677    -0.05
#  7.00  3.782  3.781    -0.10
# 10.00  3.856  3.854    -0.19
# 20.00  3.930  3.923    -0.70
# 30.00  3.953  3.938    -1.54
```

The adjustment is invisible inside the first year and grows into ~1.5 basis
points by 30 years — small in this modest-volatility calibration, but the sign
and the widening are exactly the arbitrage-free convexity pull, and both scale
with `σ₁₁²`.

Two guardrails — the three-element `sigma` requirement and the `σ → 0` nesting:

```python
import numpy as np, tsecon

# sigma must have EXACTLY 3 elements (the factor-volatility diagonal),
# not one per maturity.
try:
    tsecon.afns_adjustment(np.array([1.0, 5.0, 10.0]), np.array([0.01, 0.01]), decay=0.5)
except ValueError as e:
    print("ValueError:", e)

# Sigma -> 0 nests plain Nelson-Siegel: the adjustment is identically zero.
zero = tsecon.afns_adjustment(np.array([1.0, 5.0, 10.0, 30.0]),
                              np.array([0.0, 0.0, 0.0]), decay=0.5)
print("sigma = 0 -> adjustment all zero:", bool(np.all(zero == 0.0)))
# ValueError: sigma must have 3 elements (the factor-volatility diagonal); got 2
# sigma = 0 -> adjustment all zero: True
```

The default `decay = 0.0609` is the Diebold-Li monthly convention — with
maturities in **months** it peaks the curvature loading near 30 months:

```python
import numpy as np
lam = 0.0609
tau = np.arange(1, 121, 0.25)                          # maturities in MONTHS
curv = (1 - np.exp(-lam*tau))/(lam*tau) - np.exp(-lam*tau)
print("curvature loading peaks at tau =", round(tau[np.argmax(curv)], 1), "months")
# curvature loading peaks at tau = 29.5 months
```
