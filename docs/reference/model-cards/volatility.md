# Model card — Volatility

`garch_fit` · `gas_volatility` · `ccc_garch` · `dcc_garch`

Conditional-variance models: they leave the mean alone and model how the
*spread* of a return series evolves. Reach for this family when the level is
roughly unpredictable but the turbulence is not — the hallmark of financial
returns, where large moves cluster.

---

## `garch_fit` — GARCH / GJR / EGARCH

**What it estimates.** A univariate conditional-variance process for one return
series: today's variance as a function of yesterday's squared shock and
yesterday's variance (GARCH), optionally with a leverage term that lets bad
news raise variance more than good news (GJR, EGARCH). Fit by Gaussian or
Student-*t* quasi-maximum likelihood.

**Assumptions.** A correctly specified mean (constant/zero/AR), i.i.d.
standardized innovations from the chosen density, and stationary variance
(`alpha + beta < 1` for GARCH). QMLE is consistent for the variance parameters
even if the innovation density is wrong — that is what the robust SEs protect.

**When to use (and when not).** Use it whenever volatility clusters and you
need a variance forecast or filtered conditional volatility — VaR/ES inputs,
option-style risk. Prefer `vol="gjr"` or `"egarch"` for equity indices, where
leverage is real. Do **not** use it as a mean model, on a series with no ARCH
effect (check `arch_lm` first), or on daily data when you have intraday data —
realized measures (`har_rv`) dominate there.

**Key arguments and defaults (and why).** `vol="garch"` is the workhorse.
The default is `mean="zero"` — it assumes you feed *pre-demeaned* returns;
pass `mean="constant"` to have the fit estimate `mu` for you (as the example
below does). This is a real porting gotcha: the `arch` package defaults to a
*constant* mean, so `arch_model(r).fit()` and `tsecon.garch_fit(r)` are not the
same model unless `r` is already demeaned or you say `mean="constant"`.
`dist="normal"` gives clean QMLE, switch to `dist="t"` when standardized
residuals stay fat-tailed. `p=1, q=1` is the near-universal order; `o=1` turns
on the asymmetry term for GJR/EGARCH. `forecast_horizon` returns the multi-step
variance path.

**How to read the output.** `params` are named by `param_names`
(`omega, alpha[1], beta[1]`, with `mu` prepended under `mean="constant"` and
`nu` appended for *t*). Trust **`se_robust`**
(Bollerslev-Wooldridge) over `se_mle` unless you believe the density.
`conditional_volatility` is the filtered σ_t, `std_residuals` should look
i.i.d. (re-run `arch_lm` on them), and `variance_forecast` is the horizon path.
`alpha[1] + beta[1]` near 1 means shocks persist for a long time.

**Failure modes.** Near-integrated variance (`alpha + beta ≈ 1`) flattens the
likelihood and destabilizes SEs; a mis-specified mean leaks into the variance;
on genuinely Gaussian data the *t* degrees of freedom `nu` drift very large
(the *t* nesting the normal). Optimizer failures usually mean the series has no
ARCH structure to fit.

**Validated against.** Kevin Sheppard's [`arch`](https://arch.readthedocs.io)
package — GARCH/GJR/EGARCH QMLE point estimates, log-likelihood, and robust
SEs (`fixtures/garch.json`).

**References.** Bollerslev (1986); Nelson (1991, EGARCH); Glosten, Jagannathan
& Runkle (1993, GJR); Bollerslev & Wooldridge (1992, robust SEs).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n, nu = 2000, 7.0
eps = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)   # unit-variance t shocks
r = np.zeros(n); sig2 = np.zeros(n)
sig2[0] = 0.05 / (1 - 0.08 - 0.90)                     # unconditional variance
for t in range(1, n):
    sig2[t] = 0.05 + 0.08 * r[t - 1] ** 2 + 0.90 * sig2[t - 1]
    r[t] = np.sqrt(sig2[t]) * eps[t]

fit = tsecon.garch_fit(r, vol="garch", mean="constant", dist="t",
                       p=1, q=1, forecast_horizon=5)
print(dict(zip(fit["param_names"], np.round(fit["params"], 4))))
# {'mu': -0.0004, 'omega': 0.0267, 'alpha[1]': 0.0615, 'beta[1]': 0.9239, 'nu': 8.37}
print("robust SEs:", np.round(fit["se_robust"], 4))
print("5-step variance path:", np.round(fit["variance_forecast"], 4))
```

---

## `gas_volatility` — score-driven (GAS/DCS) volatility

**What it estimates.** A GAS(1,1) score-driven variance: the variance is
updated each period by the *score* of the observation density, which makes the
Student-*t* version automatically down-weight outliers. Gaussian GAS(1,1) is
algebraically GARCH(1,1) rewritten.

**Assumptions / when to use.** Same stationarity/mean assumptions as GARCH.
Use `density="student_t"` precisely when standardized residuals stay fat-tailed
after a GARCH fit and you want extremes treated as outliers rather than allowed
to dominate the variance. Do **not** expect `density="gaussian"` to beat GARCH —
it *is* GARCH.

**Key arguments and defaults.** `density="gaussian"` (change to `"student_t"`
for the payoff); `horizon=0` (set >0 for a variance forecast).

**How to read the output.** `omega, a, b` are the intercept, score-loading, and
persistence; `nu` the *t* degrees of freedom; `variance` the filtered path;
`next_variance` and `forecast` the projection. **Read `params` and `loglik`,
not `converged` alone** — a persistence `b` near 1 flattens the surface and the
flag can read `False` at a good optimum; on Gaussian data `nu` drifts huge.

**Failure modes.** Symmetric (no leverage) — pair with GJR/EGARCH for equities.
`converged=False` is often benign near `b≈1`; huge `nu` signals Gaussian data.

**Validated against.** Hand-derived analytic score/density references (no
external Python GAS library in the venv); the Gaussian recursion is
cross-checked to reproduce GARCH(1,1) and simulated parameters are recovered
(`fixtures/tsecon-gas.json`).

**References.** Creal, Koopman & Lucas (2013); Harvey (2013).

---

## `ccc_garch` / `dcc_garch` — multivariate GARCH

**What they estimate.** The conditional covariance of a *panel* of returns
(`returns` is T×k). CCC fits per-series GARCH and holds the correlation matrix
**constant**; DCC lets that correlation matrix **evolve** with two extra scalars
`a, b` (mean-reverting to the unconditional `qbar`).

**Assumptions / when to use.** Each series is GARCH-like; CCC assumes the
cross-correlations do not move (often violated in crises), DCC relaxes exactly
that. Use CCC for a fast, parsimonious baseline; use DCC when correlations
plausibly rise together in stress (portfolio risk, contagion). Not for very
large k without regularization.

**Key arguments.** Both take only `returns` (T×k) in the shipped surface;
defaults handle the two-step estimation internally.

**How to read the output.** CCC returns the constant `correlation` matrix and
`loglik`. DCC returns `a, b` (dynamics), `qbar` (targeted long-run covariance),
`correlation_last` (the most recent conditional correlation), `loglik`, and
`converged`. `a + b` near 1 means correlations move slowly and persistently.

**Failure modes.** A stage-one univariate GARCH fit can fail on a series with
no ARCH effect (the error names the offending series); DCC on near-constant
correlations collapses toward the CCC special case.

**Validated against.** No external Python/R DCC reference in the venv; validated
by the CCC special case, recovery of simulated DCC parameters, and
positive-definiteness / variance-targeting properties (`fixtures/mgarch.json`).

**References.** Bollerslev (1990, CCC); Engle (2002, DCC).

The DGP below is chosen to make the CCC/DCC contrast visible: the true
correlation *moves* — a calm regime (ρ = 0.2) followed by a crisis regime
(ρ = 0.8). On constant-correlation data DCC would (correctly) collapse to the
CCC special case with `b ≈ 0`; here the dynamics have something to track.

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n = 2000
rho = np.where(np.arange(n) < n // 2, 0.2, 0.8)   # calm rho=0.2, then crisis rho=0.8
R = np.zeros((n, 2)); s2 = np.full(2, 0.5)
for t in range(n):
    z1 = rng.standard_normal()                     # correlated unit shocks at rho[t]
    z2 = rho[t] * z1 + np.sqrt(1.0 - rho[t] ** 2) * rng.standard_normal()
    R[t] = np.sqrt(s2) * np.array([z1, z2])
    s2 = 0.05 + 0.08 * R[t] ** 2 + 0.90 * s2       # per-series GARCH(1,1) recursion

ccc = tsecon.ccc_garch(R)                           # returns is T x k
print("CCC correlation:", round(ccc["correlation"][0][1], 3))
# CCC correlation: 0.479

dcc = tsecon.dcc_garch(R)
print("a, b:", round(dcc["a"], 3), round(dcc["b"], 3), " converged:", dcc["converged"])
print("last conditional correlation:", round(dcc["correlation_last"][0][1], 3))
# a, b: 0.026 0.974  converged: True
# last conditional correlation: 0.802
```

The two fits tell the story. CCC reports 0.479 — a blend of the two regimes
that is true of *neither*. DCC estimates persistent dynamics (`a + b ≈ 0.999`,
the near-unit persistence a one-time break masquerades as) and its most recent
conditional correlation, 0.802, has tracked its way to the crisis regime's
true ρ = 0.8.
