# Model card — GMM

`iv_gmm` · `gmm_nonlinear`

The generalized method of moments estimates parameters by forcing a set of
sample moment conditions — things the model says should be zero in expectation —
as close to zero as a weighting matrix allows. When there are more moments than
parameters, the *remaining* slack is itself a specification test. This family
covers the linear instrumental-variables case in closed form and arbitrary
nonlinear moment systems through a Python callback.

---

## `iv_gmm` — linear IV-GMM

**What it estimates.** The coefficients of a linear model `y = X·beta + u` where
some columns of `X` are endogenous, identified by instruments `Z` (Hansen 1982).
With more instruments than regressors the system is over-identified and the
efficient GMM estimator uses a robust or HAC weighting matrix, plus the Hansen
J-test of the over-identifying restrictions.

**Assumptions.** Instrument relevance (`Z` correlated with the endogenous
regressors) and exogeneity (`E[Z·u] = 0`). **`Z` must include the exogenous
regressor columns** (intercept, exogenous controls) alongside the excluded
instruments — those regressors instrument themselves. HAC weighting assumes the
moment process is stationary with summable autocovariances.

**When to use (and when not).** Use for endogenous regressors with valid
instruments, over-identified systems where you want efficiency and a
specification test, and time-series moments needing HAC weighting. Do not use
with weak instruments (the estimator is biased toward OLS and the J-test
misleads), and prefer plain `ols` when nothing is endogenous.

**Key arguments and defaults (and why).** `method="2sls"` (one-step, robust to
weak-ID concerns) vs `"2step"` (efficient GMM, the usual default choice) vs
`"iterated"` (iterate the weighting matrix to convergence). `weight="robust"`
(heteroskedasticity-robust) or `"hac"` (adds autocorrelation robustness, with
`bandwidth`). `tol`/`max_iter` govern the iterated variant.

**How to read the output.** `params` (in the column order of `X`), `bse`
(robust/HAC standard errors), `residuals`, `nmoments`/`nparams`, `steps`. When
over-identified (`nmoments > nparams`): the Hansen **`j_stat`** with `j_dof`
degrees of freedom and `j_pval` — a small `j_pval` rejects the moment
conditions (some instrument is invalid or the model is misspecified). A large
`j_pval` is reassuring, not proof of validity.

**Failure modes.** Weak instruments (dominant failure — check the first stage);
forgetting to put the exogenous columns into `Z`; reading a passing J-test as
proof of exogeneity rather than absence of contradiction.

**Validated against.** `linearmodels` `IVGMM` — 2-step robust weighting, robust
covariance, and the Hansen J statistic (`fixtures/gmm.json`).

**References.** Hansen (1982); Hansen, Heaton & Yaron (1996, iterated/CUE);
Newey & West (1987, HAC).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n = 500
z = rng.standard_normal((n, 2))              # two excluded instruments
u = rng.standard_normal(n)
x = z @ np.array([0.8, 0.5]) + 0.7 * u + 0.5 * rng.standard_normal(n)   # endogenous
w = rng.standard_normal(n)                    # an exogenous regressor
y = 1.0 + 0.3 * w + 1.5 * x + u               # true (const, w, x) = (1.0, 0.3, 1.5)

X = np.column_stack([np.ones(n), w, x])                  # regressors
Z = np.column_stack([np.ones(n), w, z])                  # instruments incl. exogenous cols
fit = tsecon.iv_gmm(X, Z, y, method="2step", weight="robust")
print("params [const, w, x]:", np.round(fit["params"], 3))   # ~[1.05, 0.29, 1.52]
print("robust SEs          :", np.round(fit["bse"], 3))
print(f"Hansen J = {fit['j_stat']:.3f} (dof {fit['j_dof']}), p = {fit['j_pval']:.3f}")
```

---

## `gmm_nonlinear` — nonlinear GMM via a moment callback

**What it estimates.** GMM for an arbitrary moment system you write in Python:
you supply a function mapping a parameter vector to an `n`-by-`m` matrix of
per-observation moment contributions, and a derivative-free Nelder-Mead search
minimizes the GMM objective `ḡ' W ḡ`. Handles exactly-identified and
over-identified systems.

**Assumptions.** The moment conditions hold at the truth (`E[g(θ₀)] = 0`) and
identify the parameters; the objective is smooth enough for Nelder-Mead to make
progress from `initial`.

**When to use (and when not).** Use for custom estimators — Euler-equation
moments, method-of-moments, simulated moments — where no closed form exists. For
a *linear* IV model use `iv_gmm` (faster and analytic). Nelder-Mead is
derivative-free and robust but slow in high dimensions; keep the parameter count
modest.

**Key arguments and defaults (and why).** `moments_fn` returns an `n×m` array
(rows = observations, columns = moments); `initial` is the starting parameter
vector (its length sets `nparams`); `weight` is the flattened `m×m` weighting
matrix (row-major) or `None` for the identity. Start with the identity, then
optionally re-weight by the inverse moment covariance for efficiency.

**How to read the output.** `params`, `objective` (the minimized `ḡ' W ḡ` —
near zero when exactly identified), `gbar` (the average moments at the optimum —
should be ~0 when exactly identified), `converged`, and `iterations`/`fevals`/
`nmoments`/`nparams`. A non-zero `objective` in an *over*-identified system is
the analogue of the J-statistic slack.

**Failure modes.** A moment function that returns the wrong shape, poor starting
values (Nelder-Mead is local — try several `initial` points), and flat or
discontinuous objectives that stall the simplex.

**Validated against.** No external golden; the crate property test recovers the
closed-form mean/variance method-of-moments solution to ~1e-4, and the identity
weight reproduces the default (Hansen 1982).

**References.** Hansen (1982); McFadden (1989, simulated moments).

```python
import numpy as np, tsecon
rng = np.random.default_rng(0)
y = 2.0 + 1.5 * rng.standard_normal(400)

def moments(theta):                          # E[y-mu]=0, E[(y-mu)^2 - s2]=0
    resid = y - theta[0]
    return np.column_stack([resid, resid ** 2 - theta[1]])

g = tsecon.gmm_nonlinear(moments, initial=[0.0, 1.0])
print("(mean, var):", np.round(g["params"], 3), " converged:", g["converged"])
print("avg moments at optimum:", np.round(g["gbar"], 5))
```
