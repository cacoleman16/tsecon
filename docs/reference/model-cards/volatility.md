# Model card — Volatility

`garch_fit` · `gas_volatility` · `dcs_local_level` · `ccc_garch` · `dcc_garch`

Conditional-variance models: they leave the mean alone and model how the
*spread* of a return series evolves. Reach for this family when the level is
roughly unpredictable but the turbulence is not — the hallmark of financial
returns, where large moves cluster.

One member is the exception that proves the family rule: `dcs_local_level`
applies the same score-driven (GAS/DCS) machinery as `gas_volatility` to a
time-varying **level** instead of a variance — it lives here because the
score-driven house is one family, and its payoff (outlier-robust trend
filtering) is the level-side twin of GAS-t's outlier-robust variance.

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

## `dcs_local_level` — score-driven robust local level (DCS-t)

**What it estimates.** A time-varying *level* `mu_{t+1} = mu_t + kappa·u_t`,
driven by the conditional score `u_t` of the chosen observation density —
the DCS local level (Harvey 2013; Harvey & Luati 2014, the DCS-t case). With
`density="t"` the driver `u_t = (nu+1)e_t/(nu + e_t²/scale²)` is bounded and
*redescending*: a genuine level shift moves the filter, an 8-sigma outlier
moves it almost not at all. `density="laplace"` gives the sign filter (the
level tracks a local median); `density="gaussian"` gives `u_t = e_t` — which
is *exactly* the steady-state Kalman local level, the nested control. MLE of
`(kappa, scale[, nu])` on the exact conditional likelihood given a robust
initial level (median of the first ten observations).

**Assumptions.** A local-level signal (slow-moving mean, no slope/seasonal
component), i.i.d. errors from the chosen density with constant scale, and
`nu > 2` for the *t*. There is no smoother — `level[t]` is the one-step
*prediction* of `y[t]` given data through `t-1` (the DCS literature
filters), and the h-step forecast is flat at `next_level`.

**When to use (and when not).** Use it to track a trend/level through data
where additive outliers are plausible — the lab study that graduated it
measured **−22%/−31% level RMSE vs the Kalman local-level pipeline at 5%/10%
contamination with zero measurable cost on clean data**, because the
contaminated Gaussian MLE absorbs outliers by collapsing its gain (going
blind to real level movement) while the bounded *t* score discounts them
point by point. On clean Gaussian data it matches the Kalman filter — so the
robust default is cheap. Do **not** use it when you need smoothed (two-sided)
estimates, slope/seasonal components, or time-varying volatility
(`sigma` here is constant; pair with `gas_volatility` thinking, not inside
it) — `local_level_smooth` covers the Gaussian smoothing case.

**Key arguments and defaults (and why).** `density="t"` is the default —
robustness is the point of the estimator, and on clean data it costs
nothing. Switch to `"gaussian"` only as the nested control (it *is* the
steady-state Kalman filter), or `"laplace"` for a median-tracking filter.

**How to read the output.** `kappa` is the constant gain (for
`"gaussian"` it is the steady-state Kalman gain: `kappa = p/(1+p)`,
`p = (q + √(q²+4q))/2`, `q = sigma2_eta/sigma2_eps`, inverse
`q = kappa²/(1−kappa)`); `scale` is the density's scale parameter (for
`"gaussian"` the one-step prediction-error sd, `sigma_eps/√(1−kappa)`; for
`"t"` the *t* scale, not the sd); `nu` the estimated dof. `*_se` are
observed-information SEs — NaN means the Hessian was singular or the
optimum sits at a boundary, reported honestly rather than clipped. `level`
is the one-step-predicted path, `resid = y − level`, `next_level` the
out-of-sample prediction. **Read `converged` as the optimizer's
certificate, not a fit grade**: on (near-)Gaussian data the *t* fit's `nu`
runs to the boundary and the flag reads `False` while `kappa`, `scale`, and
the level path are fine.

**Failure modes.** Under heavy contamination `nu` pins near its lower bound
2 — the fat tail is doing outlier duty, so do **not** report `nu_hat` as the
clean noise's tail index (and expect NaN SEs there: the boundary has no
interior curvature). The Laplace likelihood is piecewise in `kappa` (every
sign flip is a kink): a denser multistart is applied, but `converged`
certifies the best basin found, not global optimality over the kinks, and
single-sample `kappa` for the sign filter is noisy. A constant series is
refused outright (the likelihood is unbounded as `scale → 0`).

**Validated against.** statsmodels `UnobservedComponents(y, 'llevel')` for
the Gaussian limit, pinned *through the steady-state mapping* above:
statsmodels' UC-MLE variances are mapped to `(kappa, scale)` and statsmodels
itself re-run at those values with known steady-state initialization — its
constant Kalman gain equals `kappa` and the crate reproduces its level path
(1e-6) and full log-likelihood (1e-8) on two seeded series and the Nile,
plus the fitted params against a scipy MLE of the identical criterion at
1e-4 across two optimizers (`fixtures/tsecon-dcs.json`). The t/Laplace
filters have **no runnable third-party reference** (DCS reference code is
R/Matlab) and are Monte-Carlo graded: 200-rep seeded recovery on simulated
DCS-t data (kappa bias −0.003, RMSE 0.033; scale bias +0.001, RMSE 0.058;
median `nu_hat` 5.17 at true 5; 200/200 converged), and a 20-rep replication
of the lab's contamination study — mean one-step-level RMSE ratios vs the
fitted Gaussian control of **1.00/0.77/0.69** for DCS-t at 0/5/10% additive
8-sigma outliers (Laplace 1.10/0.81/0.74), with the Gaussian gain collapsing
0.086 → 0.034 while DCS-t raises its own to 0.122
(`crates/tsecon-gas/tests/dcs_properties.rs`).

**References.** Harvey & Luati (2014), "Filtering with Heavy Tails", *JASA*
109(507); Harvey (2013); Creal, Koopman & Lucas (2013); Durbin & Koopman
(2012, steady-state Kalman filter).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
T = 500
mu = np.cumsum(rng.normal(0.0, 0.1, T))          # slow-moving true level
y = mu + rng.normal(0.0, 1.0, T)                 # noisy observations
out = rng.choice(T, 50, replace=False)           # 10% additive outliers at 8 sigma
y[out] += rng.choice([-1.0, 1.0], 50) * 8.0

g = tsecon.dcs_local_level(y, density="gaussian")
t = tsecon.dcs_local_level(y, density="t")
rmse = lambda r: float(np.sqrt(np.mean((np.asarray(r["level"]) - mu) ** 2)))
print(f"gaussian: kappa={g['kappa']:.3f}  level RMSE={rmse(g):.3f}")
print(f"t:        kappa={t['kappa']:.3f} (se {t['kappa_se']:.3f})  "
      f"nu={t['nu']:.2f}  level RMSE={rmse(t):.3f}")
# gaussian: kappa=0.023  level RMSE=0.440
# t:        kappa=0.118 (se nan)  nu=2.00  level RMSE=0.311
```

The two lines are the whole story. The contaminated Gaussian MLE has
collapsed its gain to 0.023 — nearly blind to the real level — where the
*t* filter keeps `kappa = 0.118` and cuts the level RMSE by 29%. And the
fit is honest about its edges: `nu` has pinned at its lower bound (the fat
tail is absorbing the outliers — not a tail estimate), so the
observed-information SE correctly comes back NaN rather than a number
computed from boundary curvature.

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
