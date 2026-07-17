# Model card — Predictive regressions & IVX

`predictive_regression` · `ivx_test`

Return predictability regressions have an awkward econometric problem baked in.
The forecasting variable — a dividend yield, a term spread, a valuation ratio —
is typically *persistent*, its autoregressive root sitting near one, and its
innovation is correlated with the return it is meant to predict. That is the
**Stambaugh setting**, and in it ordinary least squares misbehaves twice over:
the slope is biased in finite samples, and the naive t-test over-rejects a true
"no predictability" null, the more so as the predictor approaches a unit root.
This family gives three answers on the same regression — plain OLS, a
closed-form bias correction, and an instrument (IVX) whose test keeps its size
whether the predictor is stationary, near-integrated, or an exact unit root.

The one-step regression throughout is

```text
r_{t+1} = alpha + beta * x_t + u_{t+1},        x_t = rho * x_{t-1} + e_t
```

with `x` persistent (`rho` near 1) and `corr(u, e) != 0` (endogeneity).

---

## `predictive_regression` — three views of one regression

**What it estimates.** The predictive slope `beta` of a one-step-ahead return
`r_{t+1}` on a single persistent predictor `x_t`, returned three ways in one
call: `ols` (the uncorrected least-squares slope, its standard error and
t-statistic), `stambaugh` (the Stambaugh 1999 finite-sample bias-corrected
slope), and `ivx` (the Kostakis-Magdalinos-Stamatogiannis 2015 estimator with a
Wald test of `H0: beta = 0` that is asymptotically chi-square **uniformly over
the persistence of `x`**). The three share the same regression; they differ in
how they handle the persistence-plus-endogeneity problem.

**Assumptions.** A single predictor observed one period before the return; an
AR(1) predictor whose root may be anywhere up to and including unity; and the
Stambaugh endogeneity structure (`e_t` correlated with `u_{t+1}`) — which is
precisely the case that breaks OLS and motivates the other two views. The
Stambaugh correction additionally leans on the AR(1) being a good model of the
predictor and on the Kendall approximation to the least-squares root bias,
`E[rho_hat - rho] ≈ -(1 + 3·rho)/n`. IVX assumes the predictor is (at worst)
local-to-unity, so its self-generated "mildly integrated" instrument is valid.

**When to use (and when not).** Use it whenever you regress a return (or any
one-step target) on a slow-moving, persistent predictor and want inference you
can trust near the unit root — the classic return-predictability question. Read
the `ivx` Wald test as your headline significance verdict; use `stambaugh` to
report a debiased point estimate; keep `ols` only as the (misleading) benchmark
that shows what the correction bought you. Do **not** trust the `ols` t-statistic
for a persistent predictor — that is the whole point. Do not use this for a
stationary, weakly-dependent regressor with no endogeneity (plain `ols`
suffices), nor for multi-step overlapping returns without accounting for the
induced serial correlation, nor as a joint test of several predictors — reach
for `ivx_test` there.

**Key arguments and defaults (and why).** `cz = -1.0` and `alpha = 0.95` tune
the IVX instrument, whose persistence is `Rz = 1 + cz / n^alpha`. `cz` must be a
finite **negative** constant so `Rz` sits just inside the unit circle; `alpha`
must lie in the open interval `(0, 1)` so the instrument is *mildly* integrated —
more persistent than any stationary process but strictly less than a unit root,
which is what makes the Wald limit hold uniformly. The KMS defaults
(`cz = -1`, `alpha = 0.95`) are the values from the source paper and the ones
you should keep unless you are deliberately studying instrument sensitivity;
larger `alpha` pushes `Rz` closer to one (more persistent instrument), a smaller
magnitude of `cz` does likewise.

**How to read the output.** A nested dict. `fit["ols"]` has `alpha`, `beta`,
`se`, `tstat`. `fit["stambaugh"]` has `beta_ols`, `beta_corrected` (the debiased
slope), `bias_term` (what was subtracted, `= (sigma_ue/sigma_ee)·kendall_bias`),
`rho_ols` (the estimated predictor root), and `se` (to first order the OLS
standard error — the correction is a data-dependent location shift). `fit["ivx"]`
has `beta_ivx`, **`wald`** (the chi-square(1) statistic for `H0: beta = 0`),
`pvalue`, and `rz` (the realized instrument persistence). `fit["nobs"]` is the
aligned sample size `N = n - 1`. The comparison to internalize: when `rho` is
near 1 the `ols` `tstat` is inflated, `beta_corrected` pulls the point estimate
back toward the truth, and the `ivx` `wald`/`pvalue` is the one you report.

**Failure modes.** Reading the `ols` t-statistic as if it were valid near the
unit root — the error IVX exists to prevent. A constant or non-varying predictor
makes the AR(1) fit or the IVX cross-moment singular (raised as an error, not
silently). A predictor that is genuinely stationary and exogenous gains nothing
from the correction — the three views collapse together, which is informative,
not a bug. Multi-step / overlapping returns violate the one-step DGP and need
separate handling. Finally, IVX controls *size*, not power: a large p-value near
a unit root is "no evidence", not "proven no predictability".

**Validated against.** A documented-formula NumPy golden
(`fixtures/predreg.json`, generated by `fixtures/generate_predreg_fixtures.py`),
which writes every published quantity — the OLS slope, the Stambaugh correction
of Stambaugh (1999, eqs. 4-6), and the KMS (2015) instrument, slope, and Wald
statistic — directly as its closed-form formula in NumPy and pins the crate to
it to ~1e-9. More importantly, the statistical claim is established by seeded
Monte-Carlo property tests: the **IVX-Wald test holds nominal 5% size across
`rho ∈ {0.9, 0.95, 0.99, 1.0}` including the exact unit root, where the naive
OLS t-test over-rejects two-to-five-fold**; it has power against a true slope;
and the Stambaugh correction measurably reduces the finite-sample bias of the
OLS slope.

**References.** Stambaugh (1999, *Journal of Financial Economics* 54:375-421);
Kostakis, Magdalinos & Stamatogiannis (2015, *Review of Financial Studies*
28:1506-1553); Phillips & Magdalinos (2009, IVX / mildly-integrated
asymptotics); Kendall (1954, bias of the least-squares AR root).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n, rho, corr_ue, beta = 300, 0.99, -0.9, 0.0   # persistent predictor, TRUE slope beta = 0
e = rng.standard_normal(n)
x = np.zeros(n)
for t in range(1, n):
    x[t] = rho * x[t - 1] + e[t]               # near-unit-root AR(1) predictor
u = corr_ue * e + np.sqrt(1 - corr_ue**2) * rng.standard_normal(n)  # error correlated with e
r = beta * x + u                               # regress r_{t+1} on x_t

fit = tsecon.predictive_regression(r, x)       # defaults cz=-1.0, alpha=0.95
ols, stb, ivx = fit["ols"], fit["stambaugh"], fit["ivx"]
print(f"OLS       : beta={ols['beta']:+.4f}  t={ols['tstat']:+.2f}")
print(f"Stambaugh : beta_corrected={stb['beta_corrected']:+.4f}  "
      f"(bias removed {stb['bias_term']:+.4f}, rho_hat={stb['rho_ols']:.3f})")
print(f"IVX       : beta_ivx={ivx['beta_ivx']:+.4f}  "
      f"Wald={ivx['wald']:.2f}  p={ivx['pvalue']:.3f}  (Rz={ivx['rz']:.4f})")
print(f"aligned obs N = {fit['nobs']}")
# OLS       : beta=+0.0123  t=+1.06
# Stambaugh : beta_corrected=+0.0007  (bias removed +0.0116, rho_hat=0.987)
# IVX       : beta_ivx=+0.0108  Wald=0.85  p=0.358  (Rz=0.9956)
# aligned obs N = 299
```

**The property, made visible.** One draw shows the machinery; the size of the
test is a claim about repeated sampling. This short simulation is the headline:
at an exact unit root with a true null, the IVX-Wald rejection rate stays near
its nominal 5%, while the naive OLS t-test rejects several times too often.

```python
import numpy as np, tsecon

rng = np.random.default_rng(1)
reps, n, corr_ue = 1000, 250, -0.9         # unit root, TRUE null beta = 0, strong endogeneity
chi2_95, z_95 = 3.841, 1.96
ivx_rej = ols_rej = 0
for _ in range(reps):
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] + e[t]             # rho = 1 exactly
    u = corr_ue * e + np.sqrt(1 - corr_ue**2) * rng.standard_normal(n)
    r = u                                   # beta = 0
    fit = tsecon.predictive_regression(r, x)
    ivx_rej += fit["ivx"]["wald"] > chi2_95
    ols_rej += abs(fit["ols"]["tstat"]) > z_95
print(f"IVX-Wald rejection rate : {ivx_rej/reps:.3f}   (nominal 0.05)")
print(f"naive OLS-t rejection   : {ols_rej/reps:.3f}   (over-rejects)")
# IVX-Wald rejection rate : 0.059   (nominal 0.05)
# naive OLS-t rejection   : 0.269   (over-rejects)
```

---

## `ivx_test` — joint IVX predictability test for several predictors

**What it estimates.** The multivariate extension: IVX slopes for a *panel* of
persistent predictors at once (`xs` is `T × k`) and a single **joint** Wald test
of `H0: beta = 0` (no predictor forecasts), asymptotically chi-square(`k`) and
uniform over the predictors' persistence. Each predictor is instrumented with
its own IVX process built from the shared `Rz`; the joint statistic is the
quadratic form `c' M⁻¹ c` in the instrumented cross-moments.

**When to use (and when not).** Use it to ask "do *any* of these persistent
variables predict the return, controlling for the others?" without the
size distortion a naive joint F-test would carry near unit roots — a competing-
predictors horse race. It is not a model-selection tool: rejection says at least
one slope is non-zero, not which one; read the per-predictor `beta_ivx` for
direction and rough magnitude, but there is no separate per-coefficient p-value
in the returned surface. Collinear or degenerate predictors make the
cross-moment matrix singular (raised as an error). For a single predictor use
`predictive_regression`, which additionally gives you the OLS and Stambaugh
views on the same call.

**Key arguments and defaults.** Same instrument tuning as above — `cz = -1.0`,
`alpha = 0.95` — with the identical roles and constraints (`cz < 0`,
`alpha ∈ (0,1)`); the scalar `Rz` is shared across the predictor columns (the
KMS matrix instrument specializes to `Rz·I`).

**How to read the output.** `beta_ivx` is the length-`k` slope vector (column
order of `xs`); `wald` is the joint statistic on `nregressors` degrees of
freedom with `pvalue`; `rz` is the shared instrument persistence; `nobs` is the
aligned `N = n - 1`. A small `pvalue` rejects joint no-predictability.

**Validated against.** The same `fixtures/predreg.json` documented-formula
golden (its `multi` block pins the joint slope vector, the residual variance,
and the chi-square(`k`) Wald statistic to ~1e-9), plus a crate property test
confirming the multivariate path specializes exactly to the scalar `ivx` when
`k = 1`.

**References.** Kostakis, Magdalinos & Stamatogiannis (2015, *Review of
Financial Studies* 28:1506-1553); Phillips & Magdalinos (2009).

```python
import numpy as np, tsecon

rng = np.random.default_rng(2)
n, rho = 400, 0.98
# Two persistent predictors; only the first truly forecasts (slopes 0.06, 0.0).
X = np.zeros((n, 2))
E = rng.standard_normal((n, 2))
for t in range(1, n):
    X[t] = rho * X[t - 1] + E[t]
u = -0.8 * E[:, 0] + rng.standard_normal(n)          # endogeneity through predictor 1
r = 0.06 * X[:, 0] + u                                # x2 carries no predictive content

joint = tsecon.ivx_test(r, X)                        # xs is T x k = 400 x 2
print(f"beta_ivx    : {np.round(joint['beta_ivx'], 4)}")
print(f"joint Wald  : {joint['wald']:.2f}  on {joint['nregressors']} df,  p = {joint['pvalue']:.4f}")
print(f"aligned obs : {joint['nobs']}   (Rz = {joint['rz']:.4f})")
# beta_ivx    : [0.0727 0.0179]
# joint Wald  : 18.00  on 2 df,  p = 0.0001
# aligned obs : 399   (Rz = 0.9966)
```
