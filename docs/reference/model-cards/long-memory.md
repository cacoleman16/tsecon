# Model card — Long memory / fractional integration

`frac_diff` · `frac_integrate` · `long_memory_d`

Some series are neither `I(0)` (short memory, summable autocovariances) nor
`I(1)` (a unit root you difference away once) but somewhere *between*: their
autocorrelations decay hyperbolically, too slowly to sum yet too fast for a unit
root. Fractional integration captures this with a single continuous parameter
`d`. The memory parameter reads as a spectrum:

- **`0 < d < 0.5`** — stationary long memory: shocks persist, but the variance
  is finite and the series is mean-reverting.
- **`d >= 0.5`** — nonstationary: the variance grows without bound (though for
  `0.5 <= d < 1` the series is still mean-reverting, unlike a unit root).
- **`d = 1`** — a unit root, the ordinary `I(1)` random walk.
- **`d < 0`** — antipersistence (over-differencing): negative autocorrelation
  that also decays slowly.

This family gives you the operator `(1 - L)^d` (and its inverse) plus two
semiparametric estimators of `d` itself.

---

## `frac_diff` / `frac_integrate` — the fractional-difference operator

**What it estimates.** Nothing statistical — these *apply* a filter. `frac_diff`
computes `(1 - L)^d x`, the fractional-difference operator expanded as a binomial
series with the exact recursion `π₀ = 1`, `π_k = π_{k-1}·(k-1-d)/k`, applied as a
start-of-sample-truncated convolution `y_t = Σ_{k=0}^{t} π_k x_{t-k}`.
`frac_integrate` applies `(1 - L)^{-d}` (the same convolution with weights
`π_k(-d)`) and is the exact inverse: because the filter is lower-triangular
Toeplitz with unit diagonal, `frac_integrate(frac_diff(x, d), d) == x` to
round-off. Fractionally integrating white noise by `d` is precisely how you
simulate an ARFIMA(0, d, 0) long-memory series.

**Assumptions.** None on the input beyond finiteness — this is deterministic
algebra, not estimation. The start-of-sample truncation means the first few `y_t`
use fewer weights than the tail; the transient is largest for `|d|` near `0.5`,
where the weights decay slowly.

**When to use (and when not).** Use `frac_diff` to *whiten* a long-memory series
once you have an estimate of `d` (see `long_memory_d`), turning an ARFIMA into an
approximately short-memory series you can then fit with ARMA tools. Use
`frac_integrate` to *simulate* long memory from an innovation sequence. Do not
reach for it when integer differencing suffices — if `d ≈ 1`, plain first
differencing is cheaper and better conditioned.

**Key arguments and defaults (and why).** Both take just `x` and the order `d`
(a plain float, positive or negative). There is no truncation-lag argument: the
expansion is carried to the full sample length, so no user tuning is needed and
the inverse is exact.

**How to read the output.** A plain float array the same length as `x`. Sign
convention: `frac_diff` with `d > 0` *removes* memory; `frac_integrate` with
`d > 0` *adds* it. If you differenced by too large a `d`, the result shows
antipersistence (negative lag-1 autocorrelation) — the tell-tale of
over-differencing.

**Failure modes.** Passing the wrong sign (differencing when you meant to
integrate, or vice versa); expecting the leading observations to be as clean as
the tail (the truncation transient); using it as an estimator (it does not
estimate `d` — you supply `d`).

**Validated against.** A documented-formula golden: NumPy runs the identical
binomial recursion and convolution on fixed inputs and matches the crate to
~1e-12, with the round-trip `frac_integrate ∘ frac_diff` recovering the input to
round-off (`fixtures/longmemory.json`).

**References.** Granger & Joyeux (1980); Hosking (1981).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
eps = rng.standard_normal(2000)

x = tsecon.frac_integrate(eps, 0.4)          # add memory: ARFIMA(0, 0.4, 0)
back = tsecon.frac_diff(x, 0.4)              # remove it again — exact inverse
print("round-trip max error:", np.max(np.abs(back - eps)))   # ~1e-14
print("lag-1 autocorr, memory series:", round(np.corrcoef(x[1:], x[:-1])[0, 1], 3))
print("lag-1 autocorr, whitened     :", round(np.corrcoef(back[1:], back[:-1])[0, 1], 3))
```

---

## `long_memory_d` — estimate the memory parameter `d`

**What it estimates.** The single memory parameter `d` from the low-frequency
behavior of the periodogram, without committing to a full parametric ARFIMA
model. Two semiparametric estimators share one interface:

- **`method="gph"`** — the Geweke & Porter-Hudak (1983) log-periodogram
  regression. Over the lowest `m` Fourier frequencies `λ_j = 2πj/n`, regress
  `log I(λ_j)` on `R_j = -2·log(2·sin(λ_j/2))` by OLS; the slope estimates `d`.
- **`method="local_whittle"`** — the Robinson (1995) Gaussian semiparametric
  estimator, minimizing the concentrated Whittle objective
  `R(d) = log((1/m)·Σ λ_j^{2d} I_j) - (2d/m)·Σ log λ_j` over `d ∈ (-1/2, 1)`.
  It is more efficient than GPH and is the modern default choice.

**Assumptions.** The spectral density behaves like `λ^{-2d}` near the origin — a
local, low-frequency assumption only; the short-run dynamics away from zero are
left unmodeled. Both estimators are semiparametric: consistency does not require
you to specify the full ARMA structure, only that `m` grows slower than `n` so
the low-frequency window stays local.

**When to use (and when not).** Use it to *measure* persistence — is a
volatility, inflation, or realized-variance series genuinely long-memory, or a
short-memory process masquerading as one? Feed the estimate into `frac_diff` to
whiten before ARMA modeling. Prefer `local_whittle` for efficiency; keep `gph`
when you want the transparency and easy diagnostics of a regression. Do not read
`d` near `0.5` as a sharp verdict on stationarity — the confidence band usually
straddles the boundary, and neither estimator is designed to *test* `d = 1`
(use a unit-root test for that).

**Key arguments and defaults (and why).** `method="gph"` is the default;
`method="local_whittle"` switches estimators. `m` is the number of low
frequencies used and is the one real tuning knob — it trades bias against
variance: too large and short-run dynamics contaminate `d` (bias), too small and
the estimate is noisy (variance). Leaving `m=None` applies the textbook
`m = floor(sqrt(n))` rule. A common sensitivity check is to re-estimate across a
grid of `m` and confirm `d` is stable.

**How to read the output.** A dict with `d` (the estimate), `se` (its asymptotic
standard error), and `m` (the bandwidth actually used — useful when you left it
at the default). The GPH `se` is the documented `π / sqrt(24m)`; the local-Whittle
`se` is `1 / (2·sqrt(m))` — notice both shrink only as `sqrt(m)`, so honest bands
are wide unless `m` is large. Locate `d` on the spectrum in the intro:
`0 < d < 0.5` is stationary long memory, `d ≥ 0.5` nonstationary, `d ≈ 1` a unit
root, `d < 0` over-differenced.

**Failure modes.** Choosing `m` too large so ARMA short-run structure leaks into
`d` (the dominant bias); over-interpreting a single `d` without an `m`-sensitivity
sweep; treating an estimate near `0.5` as a stationarity verdict when the band is
wide; confusing genuine long memory with a structural break or slowly-varying
mean, both of which mimic a low-frequency spectral peak.

**Validated against.** Documented-formula goldens — NumPy builds the periodogram
(FFT), the GPH regressor and OLS, and a grid evaluation of the Whittle objective
`R(d)` with its minimizer, never calling the crate: GPH matched to ~1e-8 and
local Whittle to ~1e-6 (there is no mainstream Python GPH/local-Whittle package
to reference). Separately, seeded Monte-Carlo property tests establish the
statistical claim the algebra supports: on simulated ARFIMA(0, d, 0) series with
known `d ∈ {0.2, 0.4}`, both estimators recover `d` within Monte-Carlo bands
(`fixtures/longmemory.json`).

**References.** Geweke & Porter-Hudak (1983, GPH); Robinson (1995, local
Whittle); Granger & Joyeux (1980) and Hosking (1981) for the ARFIMA model.

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n, d_true = 8000, 0.35
eps = rng.standard_normal(n)
x = tsecon.frac_integrate(eps, d_true)       # ARFIMA(0, d, 0): known memory d

gph = tsecon.long_memory_d(x, m=400, method="gph")
lw  = tsecon.long_memory_d(x, m=400, method="local_whittle")
print(f"GPH: d={gph['d']:.3f}  se={gph['se']:.3f}  (m={gph['m']})")   # d ~ 0.383
print(f"LW : d={lw['d']:.3f}  se={lw['se']:.3f}  (m={lw['m']})")      # d ~ 0.378

white = tsecon.frac_diff(x, lw["d"])         # whiten using the estimate
print("whitened series std:", round(white.std(), 3))                 # ~1.0
```
