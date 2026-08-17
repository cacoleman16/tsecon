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

!!! danger "Correctness notice: HAC weighting was a silent no-op before 0.2.0"

    `bandwidth` used to default to `0.0`, and a Bartlett kernel truncated at
    zero lags **is** the White estimator. So `iv_gmm(..., weight="hac")` with no
    explicit `bandwidth` returned standard errors bit-identical to
    `weight="robust"` — verified at max `|Δ se| = 0.000e+00` over 3000
    replications — while the caller believed they had asked for
    serial-correlation robustness. **Anyone who called `weight="hac"` without
    naming a bandwidth before this release got heteroskedasticity-robust
    standard errors and nothing more.** If you have published results that
    relied on that call, re-run them.

    What changed: `bandwidth=None` is now the default and selects the
    Newey-West rule of thumb `floor(4 (n/100)^(2/9))`; an explicit
    `bandwidth=0.0` raises a `ValueError` instead of silently degenerating; and
    the truncation actually used is always reported back as `hac_bandwidth`, so
    "which bandwidth did I get?" is never a guess.

    **This does not restore coverage.** The
    [interval-coverage audit](../../examples/interval-coverage.md) measured
    this estimator at **0.868 ± 0.006 against a nominal 0.95** under AR(1)
    moments with `phi = 0.8`, `T = 250`, and an explicit `bandwidth=10` — and
    the automatic rule picks *4* lags at that `T`, **fewer** than the setting
    that under-covered. The new default is a sensible default, not a remedy.
    Treat a nominal-95% GMM interval under persistent moments as narrower than
    its label.

**Key arguments and defaults (and why).** The positional order is
**`(x, z, y)`** — regressors, instruments, outcome. `x` and `z` are both 2-D
float matrices, so swapping them coerces cleanly and returns plausible-looking
garbage rather than raising; prefer keywords, `iv_gmm(x=X, z=Z, y=y)`.
`method="2sls"` (one-step, robust to
weak-ID concerns) vs `"2step"` (efficient GMM, the usual default choice) vs
`"iterated"` (iterate the weighting matrix to convergence). `weight="robust"`
(heteroskedasticity-robust) or `"hac"` (adds autocorrelation robustness).
`bandwidth=None` is the HAC lag truncation: `None` resolves to the Newey-West
rule of thumb, a positive float sets it explicitly, and `0.0` is refused (see
the notice above). `method="2sls"` combined with any `weight` other than
`"robust"` now **raises** — 2SLS fixes its weighting matrix at `(Z'Z/n)^-1` by
construction and never reads `weight`, so accepting the argument there was the
same silent no-op in a different place. Use `"2step"` or `"iterated"` to
actually get HAC weighting. `tol`/`max_iter` govern the iterated variant.

**How to read the output.** `params` (in the column order of `X`), `bse`
(robust/HAC standard errors), `residuals`, `nobs`/`nmoments`/`nparams`,
`steps`, and `hac_bandwidth` — the lag truncation actually used, `None` under
`weight="robust"` and the resolved rule-of-thumb value under `weight="hac"`
with `bandwidth=None`. When over-identified (`nmoments > nparams`): the Hansen
**`j_stat`** with `j_dof` degrees of freedom and `j_pval` — a small `j_pval`
rejects the moment conditions (some instrument is invalid or the model is
misspecified). A large `j_pval` is reassuring, not proof of validity.

**`first_stage` — the weak-instrument diagnostic, and what it is not.** A
*list of dicts*, each with keys `regressor` (the column index within the `X`
you passed), `fstat`, `dof_num`, `dof_den`, and `pval`: a robust first-stage F
for one instrumented regressor on the excluded instruments (the Wald statistic
on the `q` excluded-instrument coefficients, HC1, divided by `q` and referred
to `F(q, n - L)` — the Stata `estat firststage` convention).

* **Index by `regressor`, never by position.** Entries are *omitted* — not
  zeroed, not `NaN` — wherever the statistic is undefined or not computable: an
  exogenous regressor (its column also appears in `Z`, so it instruments
  itself), no excluded instruments at all, no residual first-stage degrees of
  freedom, a regressor the instruments reproduce numerically exactly, a
  rank-deficient `Z`, or a non-finite statistic. The list can therefore be
  shorter than the regressor count, and can be empty. **A missing entry is not
  a failed fit** — the diagnostic is infallible by construction and never takes
  down the estimate; a missing row is visible to you, whereas a fabricated
  number would not be.
* The exogenous/endogenous split is *inferred* by matching `X` columns against
  `Z` columns to twelve relative digits, because the `(X, Z)` interface does
  not take the split as an argument. Pass the **identical** array values for a
  shared exogenous column in both, not a recomputed copy — an `f32` round trip
  is enough to reclassify it as an excluded instrument and change every
  reported degree of freedom.
* **`F > 10` is not a safety threshold.** The
  [interval-coverage audit](../../examples/interval-coverage.md) measured
  **0.915 coverage at a median first-stage F of 10.5**, against a nominal 0.95.
  The Staiger-Stock rule of thumb bounds bias, not interval accuracy.
* **With two or more endogenous regressors this is not a weak-identification
  test at all.** Every per-regressor F can clear 10 while the system is
  under-identified, because the instruments may predict only a single common
  combination of the endogenous regressors. The right objects there are
  **Angrist-Pischke** F statistics (per regressor, partialling out the other
  endogenous regressors) and **Cragg-Donald** / **Kleibergen-Paap** against the
  **Stock-Yogo** critical values (joint). **None of those are implemented**, and
  neither are Anderson-Rubin or other weak-IV-robust confidence sets. Until they
  are, read a multi-endogenous `first_stage` entry as a per-regressor *fit*
  summary and not as evidence that the system is identified.

**Failure modes.** Weak instruments (dominant failure — `first_stage` is a
first look, not a verdict; see above); forgetting to put the exogenous columns
into `Z`; passing a *recomputed* rather than identical exogenous column in `X`
and `Z`; reading a passing J-test as proof of exogeneity rather than absence of
contradiction; and reading a nominal-95% HAC interval as 95% under persistent
moments, where it is closer to 87%.

**Validated against.** `linearmodels` 7.0 `IVGMM` — 2-step robust weighting,
robust covariance, and the Hansen J statistic (`fixtures/gmm.json`). The
first-stage F and the HAC weighting path are pinned separately
(`fixtures/gmm_first_stage.json`) against
`IV2SLS(...).fit(cov_type="robust").first_stage.diagnostics` on one-endogenous
and two-endogenous designs; `linearmodels` reports an *undivided* HC0 Wald
referred to `chi2(q)` while this crate reports the HC1 F referred to
`F(q, n - L)`, and the golden pins the exact identity
`f.stat = fstat · q · n/(n - L)` connecting them (the HC1 form is the more
conservative of the two).

**References.** Hansen (1982); Hansen, Heaton & Yaron (1996, iterated/CUE);
Newey & West (1987, HAC); Staiger & Stock (1997, weak instruments); Stock &
Yogo (2005); Angrist & Pischke (2009, §4.6.4); Kleibergen & Paap (2006).

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

# first_stage covers only the INSTRUMENTED regressors: const and w are in Z,
# so they instrument themselves and get no entry. Index by "regressor".
print(f"first_stage: {len(fit['first_stage'])} entry for {X.shape[1]} regressors")
for e in fit["first_stage"]:
    print(f"  regressor {e['regressor']}: F = {e['fstat']:.1f} "
          f"({e['dof_num']}, {e['dof_den']}), p = {e['pval']:.2e}")
print("hac_bandwidth:", fit["hac_bandwidth"], "(None under weight='robust')")
# params [const, w, x]: [1.047 0.292 1.515]
# robust SEs          : [0.045 0.046 0.046]
# Hansen J = 1.418 (dof 1), p = 0.234
# first_stage: 1 entry for 3 regressors
#   regressor 2: F = 304.7 (2, 496), p = 4.94e-87
# hac_bandwidth: None (None under weight='robust')
```

**The HAC bandwidth, and the no-op it used to be.** With serially correlated
moments the choice is not cosmetic. Below, `weight="robust"` reports 0.046 for
the endogenous coefficient and `weight="hac"` reports 0.077 — a 67% wider
standard error. Before 0.2.0, `weight="hac"` on this design returned **0.046**:
the robust number, because the default `bandwidth=0.0` truncated the Bartlett
kernel at zero lags.

```python
import numpy as np, tsecon

def ar1(rng, n, phi):
    v = np.empty(n); v[0] = rng.standard_normal() / np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        v[t] = phi * v[t - 1] + rng.standard_normal()
    return v

rng = np.random.default_rng(7)
n, phi = 250, 0.8
u  = ar1(rng, n, phi)                                        # serially correlated errors
zx = np.column_stack([ar1(rng, n, phi), ar1(rng, n, phi)])   # persistent instruments
x  = zx @ np.array([0.9, 0.6]) + 0.7 * u + 0.5 * rng.standard_normal(n)
y  = 1.0 + 1.5 * x + u
X  = np.column_stack([np.ones(n), x])          # [const (exogenous), x (endogenous)]
Z  = np.column_stack([np.ones(n), zx])

for label, kw in [("robust    ", dict(weight="robust")),
                  ("hac (auto)", dict(weight="hac")),
                  ("hac bw=10 ", dict(weight="hac", bandwidth=10.0))]:
    f = tsecon.iv_gmm(X, Z, y, method="2step", **kw)
    print(f"{label}  bse[x] = {f['bse'][1]:.5f}   hac_bandwidth = {f['hac_bandwidth']}")

# Both former silent no-ops are now refusals, with the reason attached.
for bad in [dict(weight="hac", bandwidth=0.0), dict(method="2sls", weight="hac")]:
    try:
        tsecon.iv_gmm(X, Z, y, **{"method": "2step", **bad})
    except ValueError as err:
        print("refused:", str(err).split(":")[0])
# robust      bse[x] = 0.04642   hac_bandwidth = None
# hac (auto)  bse[x] = 0.07732   hac_bandwidth = 4.0
# hac bw=10   bse[x] = 0.08863   hac_bandwidth = 10.0
# refused: bandwidth=0.0 with weight="hac" is a no-op
# refused: method="2sls" ignores weight="hac"
```

The automatic rule picked 4 lags at `n = 250`; an explicit 10 widens further
still. Neither is calibrated for coverage — that is the 0.868 measured above.

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
(rows = observations, columns = moments) — **2-D even for a single moment
condition**: with `m = 1`, return `g.reshape(-1, 1)`, not the 1-D vector. A
badly-shaped return raises a `TypeError` that names `moments_fn` and this
contract. `initial` is the starting parameter
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
# (mean, var): [1.945 2.228]  converged: True
# avg moments at optimum: [-0. -0.]
```
