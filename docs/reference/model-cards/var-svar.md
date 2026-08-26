# Model card — VAR and structural VAR

`var_fit` · `var_irf` · `var_irf_bands` · `var_fevd` · `var_granger` ·
`var_forecast` · `sign_restricted_svar` · `zero_sign_svar` · `favar` ·
`connectedness`

The vector autoregression treats a handful of series as one system: every
variable is regressed on the recent past of every variable. From that one
reduced form come forecasts, Granger-causality tests, impulse responses,
variance decompositions, structural shocks, and connectedness measures.

---

## Reduced-form VAR — `var_fit`, `var_irf`, `var_fevd`, `var_granger`, `var_forecast`

**What it estimates.** `var_fit` fits a VAR(p) by equation-by-equation OLS
(coefficient matrix `params`, residual covariance `sigma_u`, information
criteria). The rest read off that fit: `var_irf` traces impulse responses (raw
or Cholesky-orthogonalized), `var_fevd` the forecast-error variance
decomposition, `var_granger` a block Granger-causality F test, `var_forecast`
iterated point forecasts with intervals.

**Assumptions.** Covariance-stationary inputs, correctly chosen lag length,
and — for `orth=True` IRFs and FEVD — that the Cholesky ordering encodes a
defensible contemporaneous recursion (the *first* variable reacts to nothing
within the period). Interval/Granger inference assumes stability.

**When to use (and when not).** Use for multivariate forecasting, testing
predictive precedence, and descriptive dynamics. For a *causal* impulse
response you must identify the system — a bare Cholesky ordering is a strong,
often indefensible assumption; use `sign_restricted_svar` or the local
projection family instead. Do not fit to unit-root levels without thinking:
OLS stays consistent but stability checks, IC comparisons, and Granger
distributions become fragile — difference, or use `johansen`/`vecm` if the
series trend together.

**Key arguments and defaults (and why).** `lags` — set it deliberately
(compare `aic`/`bic`/`hqic` on a *common* sample). `trend="c"` includes an
intercept. `orth=True` orthogonalizes IRFs via Cholesky; `cumulative=True`
reports running sums (level responses). `horizon` and `steps` control length;
`alpha=0.05` sets forecast-interval coverage.

**How to read the output.** `var_fit`: `sigma_u`, `aic/bic/hqic`, and the
stability block — **`is_stable`** (the verdict; read this one), `min_root`, and
`max_root`. These roots are the *inverse* characteristic roots (statsmodels
`VARResults.roots` convention), so stability requires the **smallest** inverse
root to exceed 1 — equivalently all companion eigenvalues inside the unit
circle. `max_root` is the root *farthest* from the unit circle and remains above
1 even for an explosive system, so it is not a stability verdict on its own.
`var_fit` also returns the residual surface (0.6 — previously computed in Rust
but never bound): `resid` (`(T, k)` — the OLS residuals `U = Y − ZB` over the
effective sample, row `t` belonging to observation `lags + t`; statsmodels
`results.resid`), `fitted` (`(T, k)` — the one-step fitted values, *defined* as
`data[lags:] − resid`, i.e. the OLS projection `Z @ B`, so
`fitted + resid` reproduces `data[lags:]` exactly; statsmodels
`results.fittedvalues`), `nobs` (`T = len(data) − lags`, the row count of
`resid`/`fitted`), and `df_resid` (`T − m` with `m = n_trend + k·lags`
regressors per equation — `sigma_u`'s divisor). Run your residual diagnostics
on these directly: `tsecon.ljung_box(np.asarray(fit["resid"])[:, i], 10)`.
`var_irf` returns `[h][response][shock]` (horizon 0..H). `var_fevd`
returns `[h][variable][shock]`, each variable's shares summing to 1.
`var_granger`: `statistic`, `p_value`, `df_num/df_den`. `var_forecast`:
`point`, `lower`, `upper` (each steps×k) — **marginal** intervals, one cell at a
time, which is not what a fan chart is read as; see
[simultaneous forecast bands](#simultaneous-forecast-bands-var_forecast).

**Failure modes.** A pointwise IRF median is not itself a model at long
horizons; Cholesky ordering silently drives every "structural" reading;
comparing ICs across different effective samples flips rankings; near-unit
roots make long-horizon responses and bands unreliable.

**Validated against.** statsmodels `VAR` — coefficients, `sigma_u`, IRF/FEVD,
`test_causality`, and forecasts; Lütkepohl (2005) textbook conventions
(`fixtures/var.json`).

**References.** Sims (1980); Lütkepohl, *New Introduction to Multiple Time
Series Analysis* (2005).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
k, n = 3, 400
A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.5]])
Y = np.zeros((n, k))
for t in range(1, n):
    Y[t] = A @ Y[t - 1] + 0.3 * rng.standard_normal(k)

fit = tsecon.var_fit(Y, lags=2, trend="c")
print("AIC:", round(fit["aic"], 3))
irf = np.asarray(tsecon.var_irf(Y, lags=2, horizon=10, orth=True))   # [h][resp][shock]
print("IRF shape:", irf.shape, " var0<-shock2 @ h=4:", round(irf[4, 0, 2], 4))
gc = tsecon.var_granger(Y, caused=[0], causing=[2], lags=2)
print("var2 Granger-causes var0? p =", round(gc["p_value"], 4))
```

### Simultaneous forecast bands — `var_forecast`

**What it estimates.** The same iterated point forecasts and the same per-cell
standard errors $\sqrt{\mathrm{diag}\,\mathrm{MSE}(h)}$, with a multiplier chosen
so that a **whole declared family** of `(horizon, series)` cells is inside the
band at once, rather than each cell separately at level `alpha`.

**Why it exists.** A multi-panel fan chart is read as one statement, and the
default marginal bands do not support one. The interval-coverage audit measured
nominal 95% marginal bands containing every horizon of every series
simultaneously in **40.9%** of samples at T=100 — and **48.1%** at T=800, so the
gap is multiplicity, not sample size.

**Assumptions.** `band="sup-t"` needs the **cross-horizon and cross-series**
blocks of the forecast MSE, which the marginal interval never forms; the
critical value is the `1-alpha` quantile of $\max_\text{cell}|t|$ under
$N(0,\Sigma)$, simulated. Everything the marginal interval assumes still
applies — in particular this remains a **plug-in** band that conditions on the
estimated coefficients.

**Key arguments and defaults (and why).** `band="pointwise"` (the **default**,
which adds nothing to the existing return), `"sup-t"`, `"sidak"`,
`"bonferroni"`. `band_scope="all"` is the **default** — one family covering every
horizon of every series, $K = \text{steps} \times k$, because that is how a
forecast figure is normally read. `band_scope="horizon"` gives one family per
series ($K = \text{steps}$), the right scope for a single-series fan chart.
`band_seed` and `band_n_sim` drive the sup-t Gaussian simulation and make that
band a pure function of `band_seed`; the closed forms use neither.

**How to read the output.** **`lower`/`upper` stay marginal whatever you pass** —
nothing that already reads them changes meaning. Asking for a simultaneous band
*adds* keys: `sim_lower`/`sim_upper`, the per-cell `se` the band is built from,
`critical_value` (one per series), `pointwise_critical_value`, `band_scope`,
`n_cells` (the `K`) and `n_cells_used`. Seeing the same `point` and `se` behind
both pairs of edges is the check that a simultaneous band is a **multiplier
change and nothing else**. Always quote the scope: the same 95% over $K=24$ is a
different band from the same 95% over $K=12$.

**Failure modes.** The multiplier cannot fix a band that is off *marginally*. On
the audit's design `var_forecast`'s pooled per-cell marginal coverage is
**93.4%** rather than 95%, because the band ignores coefficient sampling error —
see the plug-in decomposition in
[the audit](../../examples/interval-coverage.md#predictive-intervals) — and the
simultaneous band inherits that shortfall exactly.

**Validation target.** Nominal 95%, T=100, 12 horizons × 2 series (K=24), 2000
replications, both arms read off the **same** call: joint coverage
**42.0% ± 1.1 → 90.5% ± 0.7** ([`forecast_intervals.py`](../../examples/coverage/forecast_intervals.py),
which asserts it). A 48-point repair that still misses nominal by 4.5 points,
all of it inherited from the marginal band.

### Confidence bands on the IRF — `var_irf_bands`

`var_irf` returns the point path only. **`var_irf_bands`** is its banded
companion: same estimand, same `[h][i][j]` layout, but a `dict` with
`point`/`se`/`lower`/`upper` plus the echoed `method`/`alpha`/`n_boot`/`band`
(`n_boot` is `None` on the asymptotic branch; `band` echoes the band family and
is `"pointwise"` unless you ask for a simultaneous one). Two methods, one flag
apart:

- **`method="asymptotic"`** (default) — the Lütkepohl (1990) **delta-method**
  standard errors: the analytic derivative of the MA / orthogonalized responses
  propagated through the estimated coefficient covariance, with symmetric Wald
  bands `point ± z_{1-alpha/2}·se`. These are statsmodels' `irf.stderr`. Closed
  form, no simulation.
- **`method="bootstrap"`** — a residual (Efron/Kilian) recursive-design
  bootstrap: resample the fitted residuals, rebuild the sample through the
  estimated VAR, refit, and read **percentile** bands off the `n_boot` IRF
  draws (`se` is the draw SD). `bias_correct=True` adds the **Kilian (1998)**
  bias correction that the frontier made the frequentist default for persistent
  data. Reproducible through `seed`.

**The orthogonalization caveat.** `orth=True` bands are *not* the reduced-form
bands rescaled. The Cholesky factor $P$ in $\Theta_h = \Phi_h P$ is itself a
function of the estimated $\Sigma_u$, so the delta-method SE of an
orthogonalized response carries an extra term for
$\partial\,\mathrm{vech}(P)/\partial\,\mathrm{vech}(\Sigma_u)$ (and the
bootstrap re-factors $\Sigma_u$ on every draw). `cumulative=True` puts the
bands on the cumulated IRF — delta method via statsmodels `cum_effect_stderr`,
bootstrap by cumulating each draw first.

**The honest caveat.** By default these are **pointwise** bands: each covers one
$(h, i, j)$ cell at level `alpha`. They are *not* joint over the horizon, so a
reader who traces the whole shaded path is over-reading the coverage — the
interval-coverage audit measured a nominal 90% pointwise band containing the
whole $h = 0..12$ path in **72.2%** of samples at T=500. `band="sup-t"` below is
the fix for exactly that, and only that.

**Validated against.** statsmodels `VARResults.irf().stderr()` and
`cum_effect_stderr()` (reduced-form and orthogonalized) to machine precision;
the bootstrap by seed reproducibility and Monte-Carlo coverage. See the
[validation matrix](../validation-matrix.md).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
k, n = 3, 400
A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.5]])
Y = np.zeros((n, k))
for t in range(1, n):
    Y[t] = A @ Y[t - 1] + 0.3 * rng.standard_normal(k)

# 90% asymptotic (Lütkepohl delta-method) bands on the orthogonalized IRF
band = tsecon.var_irf_bands(Y, lags=2, horizon=8, orth=True,
                            method="asymptotic", alpha=0.1)
pt = np.asarray(band["point"]); se = np.asarray(band["se"])
lo = np.asarray(band["lower"]); hi = np.asarray(band["upper"])
print("keys:", sorted(band), " n_boot:", band["n_boot"])

# variable 0's response to its OWN shock, h = 0..8, with the 90% band
print(" h   point      se     [ lower ,  upper ]")
for h in range(9):
    print(f" {h}  {pt[h,0,0]:+.4f}  {se[h,0,0]:.4f}  [{lo[h,0,0]:+.4f}, {hi[h,0,0]:+.4f}]")

# bootstrap cross-check at h=1 (residual bootstrap, percentile band)
boot = tsecon.var_irf_bands(Y, lags=2, horizon=8, orth=True,
                            method="bootstrap", alpha=0.1, n_boot=2000, seed=0)
blo = np.asarray(boot["lower"]); bhi = np.asarray(boot["upper"])
print("bootstrap h=1 band",
      f"[{blo[1,0,0]:+.4f}, {bhi[1,0,0]:+.4f}]  vs asymptotic",
      f"[{lo[1,0,0]:+.4f}, {hi[1,0,0]:+.4f}]")
```

```
keys: ['alpha', 'band', 'lower', 'method', 'n_boot', 'point', 'se', 'upper']  n_boot: None
 h   point      se     [ lower ,  upper ]
 0  +0.2963  0.0105  [+0.2790, +0.3136]
 1  +0.1584  0.0160  [+0.1321, +0.1847]
 2  +0.0742  0.0155  [+0.0487, +0.0998]
 3  +0.0351  0.0142  [+0.0117, +0.0585]
 4  +0.0174  0.0104  [+0.0003, +0.0345]
 5  +0.0089  0.0070  [-0.0026, +0.0204]
 6  +0.0046  0.0046  [-0.0029, +0.0121]
 7  +0.0024  0.0029  [-0.0024, +0.0072]
 8  +0.0013  0.0018  [-0.0017, +0.0042]
bootstrap h=1 band [+0.1269, +0.1816]  vs asymptotic [+0.1321, +0.1847]
```

The impact response is a clean 0.30 with a band well clear of zero; by $h=5$
the band straddles zero — the response is no longer distinguishable from noise.
The bootstrap band at $h=1$ lands within a whisker of the delta-method band, the
reassurance you want when the asymptotics are the thing being trusted.

**References (bands).** Lütkepohl (1990, asymptotic IRF SEs); Kilian (1998,
bias-corrected bootstrap); Montiel Olea and Plagborg-Møller, *Simultaneous
confidence bands: theory, implementation, and an application to SVARs* — the
sup-t construction `band="sup-t"` implements. Sims & Zha (1999) likelihood-shape
bands are a different object and are **not** implemented.

#### Simultaneous bands — `band` and `band_scope`

**What it changes.** Only the multiplier. `point` and `se` are bit-identical to
the pointwise call, and `lower`/`upper` stay **pointwise whatever you pass**; the
simultaneous band arrives as *extra* keys `sim_lower`/`sim_upper`, equal to
`point ± c·se` with a constant `c` chosen so that **every cell of a declared
family** is covered at once — the sup-t construction of Montiel Olea and
Plagborg-Møller.

**Assumptions.** `band="sup-t"` needs the dependence *across* cells. On
`method="asymptotic"` that is the delta-method covariance including the
cross-horizon blocks, which the pointwise path never forms; the `1-alpha`
quantile of $\max_\text{cell} |t|$ under $N(0,\Sigma)$ is then simulated. On
`method="bootstrap"` the replications *are* draws of the estimand, so the
statistic is read off them directly with no extra simulation — centred at the
point estimate, not the bootstrap mean. `sidak` and `bonferroni` assume nothing
beyond `K` and are correspondingly loose.

**Key arguments and defaults (and why).**

- `band="pointwise"` is the **default** and adds nothing to the returned dict,
  so an existing call is untouched. The other three are `"sup-t"` (tightest —
  prefer it wherever it is available), `"sidak"`, `"bonferroni"`.
- `band_scope="horizon"` is the **default**: one family per response-shock pair,
  $K = h_{\max}+1$. That is the narrowest defensible family and the one a reader
  of a single IRF panel is implicitly asking about. `"shock"` gives one family
  per shock ($K = k(h_{\max}+1)$) — the right scope when a conclusion is drawn
  from a whole column of panels. `"all"` covers the entire grid
  ($K = k^2(h_{\max}+1)$) and is much the most conservative at large $k$.
- `band_seed` and `band_n_sim` drive the sup-t Gaussian simulation on the
  **asymptotic branch only**, where the band is a pure function of `band_seed`.
  Do not cut `band_n_sim` far down — this is a quantile in the tail of a
  maximum. On the **bootstrap branch** sup-t reads its quantile straight off the
  replications, so the existing `seed` reproduces it and neither band argument is
  used; give it `n_boot` ≥ 999 for the same tail-quantile reason. Šidák and
  Bonferroni are closed forms in `K` and need neither.

**How to read the output.** The extra keys are `sim_lower`/`sim_upper`,
`critical_value` (a k×k grid — one multiplier per response-shock cell, so you can
see exactly which cell got which), `pointwise_critical_value`, `band_scope`,
`n_cells` (the `K`) and `n_cells_used`. Report the **method, the scope, and `K`**
next to any simultaneous band; the same `alpha` over a different family is a
different band, and a band whose scope is ambiguous is worse than no band. The
ratio of `critical_value` to `pointwise_critical_value` is exactly what
simultaneity cost on this path, and `n_cells_used` is how many cells actually
entered the maximum. Cells pinned by construction have `se = 0` (the above-diagonal
Cholesky impact responses, and the whole `orth=False` impact matrix); they are
excluded from the maximum and from the Šidák/Bonferroni cell count, and keep
their zero-width band.

**Failure modes.** (i) **It fixes multiplicity and nothing else.** It reuses the
pointwise standard errors, so it inherits the delta-method decay documented
above: on the audit's design the *marginal* coverage of this band is already
91.0% at h=0 and 85.3% at h=12, and no multiplier repairs a standard error.
(ii) On the bootstrap branch the simultaneous band is **symmetric**
`point ± c·se` while the reported `lower`/`upper` are **asymmetric** Efron
percentiles — two different shapes of interval, so `sim_lower` is **not**
guaranteed to sit below `lower` cell by cell. What it is guaranteed to contain is
the like-for-like symmetric band `point ± pointwise_critical_value·se`, in which
only the multiplier differs. (iii) Šidák is exact under independence across cells, a
condition no impulse response meets; both closed forms pay for a worst case a
smooth response path does not present. At $K = 13$, $\alpha = 0.10$ the closed
forms are fixed at Šidák 2.6490 and Bonferroni 2.6653 against a pointwise
1.6449, while sup-t is a property of the path: it averages 2.0742 on the audit's
`BASE` VAR(1) and runs up to about 2.65 on more persistent ones. Read the
`critical_value` your own fit returns rather than assuming the saving.

**Validation target.** Measured on the audit's own DGP at nominal 90%, T=500,
$h = 0..12$ (K=13), 1000 replications, pointwise and sup-t read off the **same**
call: joint coverage **71.7% ± 1.4 → 85.2% ± 1.1**
([`irf_bands.py`](../../examples/coverage/irf_bands.py), which asserts it; the
crate's own tests see 70.4% → 84.8% at 3000 replications). That repairs most of
the multiplicity gap and **does not reach nominal**, for the reason in failure
mode (i). See
[pointwise is not joint](../../examples/interval-coverage.md#the-remedy-and-the-two-places-it-stops).

---

## `sign_restricted_svar` — set identification by sign restrictions

**What it estimates.** A *set* of structural VARs consistent with the data and
a handful of sign restrictions on impulse responses (e.g. "a contractionary
policy shock raises the rate and lowers prices for two quarters"). Draws random
Haar rotations, keeps those whose IRFs satisfy the signs, and summarizes the
survivors — the width of the resulting band **is** the finding.

**Assumptions.** The reduced form is correct; the signs are economically
defensible; and — the caveat a decade of applied work learned — the uniform
(Haar) prior on rotations is *not* uninformative about the responses you care
about (Baumeister-Hamilton 2015), so part of any band is prior, not evidence.

**When to use (and when not).** Use when you have credible sign information but
not enough for a recursive/long-run point identification. Do not stack on
restrictions to narrow the band without watching the acceptance rate; do not
read the pointwise median as "the" IRF (it mixes rotations across horizons).

**Key arguments and defaults.** `restrictions` are `(variable, shock, horizon,
sign)` tuples with sign in `{"+","-"}`. `lags`, `horizon`, `n_draws` (more for
smoother bands), `seed` (reproducible), `max_tries` caps rotation attempts.

**How to read the output.** `quantiles` are per-`(horizon, variable, shock)` at
`probs=[0.05,0.16,0.50,0.84,0.95]`; `set_min`/`set_max` give the identified-set
envelope; `diagnostics["acceptance_rate"]` is itself an identification
diagnostic — a rate near `1e-5` means your "posterior" is a handful of draws
and the restrictions may be near-inconsistent.

**Failure modes.** Acceptance decays roughly exponentially in the number of
restrictions; leaving a response *unrestricted* is the point (its band is the
answer, not an assumption).

**Validated against.** The **Uhlig (2005) replication on the paper's own
data** ([replication page](../../examples/replication-uhlig-monetary.md)):
his monthly VAR(12), restriction set and K = 5 window reproduce both
published findings — no price puzzle (the deflator's 84% quantile is
negative at every horizon through month 60) and the ambiguous output
response (the 68% band on real GDP straddles zero at months 6–60, within
the ±0.2% magnitude the paper's text states) — pinned offline by
[`test_replication_uhlig.py`](../../../bindings/python/tests/test_replication_uhlig.py)
against the committed [`fixtures/uhlig2005.csv`](../../../fixtures/uhlig2005.csv).
Beneath the replication, property validation: Haar-rotation moments,
sign-satisfaction of accepted draws, seed bit-reproducibility, and the same
punchline on synthetic data in the
[guide](../../guide/08-causal-identification.md).

**References.** Uhlig (2005); Rubio-Ramírez, Waggoner & Zha (2010); Arias,
Rubio-Ramírez & Waggoner (2018, corrected zero+sign).

---

## `zero_sign_svar` — zero **and** sign restrictions together

**What it estimates.** A *set* of structural VARs consistent with the data, a
handful of **exact zero** restrictions on impulse responses, and a handful of
**sign** restrictions — the Rubio-Ramírez-Waggoner-Zha (2010) exact-zero column
recursion combined with sign rejection, importance-weighted by Arias-Rubio-
Ramírez-Waggoner (2018). A strict superset of `sign_restricted_svar`: the zeros
carve exact structure into the rotation (a shock has *no* effect on some
variable at some horizon — a timing zero, a neutrality, a recursive block),
while the signs prune the rest. `sign_restrictions` are `(variable, shock,
horizon, sign)` tuples (may be empty); `zero_restrictions` are `(variable,
shock, horizon)` tuples imposing $\Theta_h[\text{variable},\text{shock}] = 0$
exactly (horizon 0 = impact); at least one list must be non-empty.

**The recursive special case.** With strict-upper-triangle **impact** zeros
($\Theta_0[i,j]=0$ for $i<j$) and no sign restrictions, the RWZ column recursion
is one-dimensional at every step: the rotation is pinned to $Q = I$, the ARW
weight is exactly 1, and each draw's structural IRF collapses to that draw's
Cholesky IRF. The scheme then reproduces `var_irf(orth=True)` — this is the
degenerate, point-identified corner of the set-identified family, and it is how
the crate golden pins the whole machinery.

**Assumptions.** Everything `sign_restricted_svar` assumes (correct reduced
form; economically defensible signs; the Haar/Minnesota prior is *not*
uninformative about the responses — Baumeister-Hamilton 2015), plus that the
imposed zeros are economically true. The zeros are enforced by construction to
machine precision; the signs by accept-reject.

**When to use (and when not).** Use when your identification mixes hard zeros
with soft signs — the modern applied pattern (e.g. a monetary shock with a
zero-impact-on-output timing restriction *and* a sign on the rate and prices).
Use it also as the honest way to impose *any* zeros alongside signs: naively
zeroing then sign-checking samples from the wrong distribution — this is the
corrected sampler. Do not stack restrictions without watching the acceptance
rate; do not read the pointwise median as "the" IRF.

**Key arguments and defaults.** `sign_restrictions` / `zero_restrictions` (at
least one non-empty); `lags`, `horizon`, `n_draws`, `max_tries` (rotation-attempt
cap), `seed`; `lambda1=0.2` (the Minnesota tightness of the reduced-form
posterior the sampler draws from); `weighted=True` (apply the ARW importance
weights to the pointwise quantiles).

**How to read the output.** `set_min` / `set_max` per `(horizon, variable,
shock)` — the **weight-invariant identified-set envelope**, and the
prior-robust object to read. `quantiles` at `probs=[0.05,0.16,0.50,0.84,0.95]`
(ARW-weighted when `weighted=True`) are the descriptive pointwise bands *inside*
that envelope. `weights` (per accepted draw, normalized) and `ess` (their
effective sample size); `diagnostics["acceptance_rate"]` is itself an
identification diagnostic. A response left *unrestricted* — its envelope is the
finding.

**The ARW importance-weight caveat — read this.** The ARW weight is **exactly 1**
for **impact-only** zero patterns (the restriction functions are linear in $Q$,
so the volume element is $Q$-independent — the recursive golden and every
impact-only applied SVAR are unweighted, and `ess` equals the accepted count).
For zeros at horizon $\ge 1$ (or on a long-run matrix) the ARW volume element is
genuinely non-constant, and **this build does not yet apply the exact ARW
volume-element correction** — it returns the conditionally-uniform (unit) weight,
i.e. the honest RWZ-2010 draw. In that case the **weight-invariant `set_min` /
`set_max` envelope is the deliverable to trust**, not the pointwise weighted
bands; the exact ARW weight for non-impact zeros is a roadmap swap-point.

**Failure modes.** Acceptance decays roughly exponentially in the number of sign
restrictions; over-reading the pointwise median (it mixes rotations across
horizons); and, for non-impact zeros, reading the weighted quantiles as if the
ARW correction were applied — read the envelope instead.

**Validated against.** A **documented-formula cross-implementation golden**: the
generator ([`generate_zero_sign_svar_fixtures.py`](../../../fixtures/generate_zero_sign_svar_fixtures.py),
never imports tsecon) transcribes $\Theta_h = \Psi_h\,\mathrm{chol}_{\text{lower}}(\Sigma)$
from the pure companion-power MA recursion. The **primary** golden is the
recursive/Cholesky recovery — strict-upper-triangle impact zeros, no signs,
positive-diagonal normalization — which the RWZ recursion reproduces
deterministically (weight 1) to `1e-10`, validating `cholesky_irf` and the
null-space recursion at once; an end-to-end binding check confirms the posterior
median recovers the `var_irf(orth=True)` structure through the Minnesota-NIW
posterior (approximately, up to posterior scatter — the machine-precision
identity is per-draw at a fixed reduced form). Sign
behavior, feasibility, and reproducibility are property-tested alongside.
Fixture: [`zero_sign_svar.json`](../../../fixtures/zero_sign_svar.json); test:
[`zero_sign.rs`](../../../crates/tsecon-ident/tests/zero_sign.rs). See the
[validation matrix](../validation-matrix.md).

**References.** Rubio-Ramírez, Waggoner & Zha (2010); Arias, Rubio-Ramírez &
Waggoner (2018, corrected zero+sign); Baumeister & Hamilton (2015).

```python
import numpy as np, tsecon

# monetary system (output, prices, ffr); shock 0 = the monetary shock.
rng = np.random.default_rng(11)
T = 500
eps = rng.standard_normal((T, 3))
B0 = np.array([[0.8, -0.3, -0.4], [0.5, 0.6, -0.5], [0.1, 0.4, 0.9]])
A1 = np.array([[0.5, 0.0, -0.1], [0.1, 0.4, 0.0], [0.0, 0.1, 0.6]])
y = np.zeros((T, 3))
for t in range(1, T):
    y[t] = A1 @ y[t - 1] + B0 @ eps[t]

zeros = [(0, 0, 0)]                                       # output: zero IMPACT to shock 0
signs = [(2, 0, 0, "+"), (1, 0, 0, "-"), (1, 0, 1, "-")]  # ffr up; prices down for two quarters
zs = tsecon.zero_sign_svar(y, sign_restrictions=signs, zero_restrictions=zeros,
                           lags=1, horizon=12, n_draws=500, max_tries=2000, seed=0)

d = zs["diagnostics"]
smin = np.asarray(zs["set_min"]); smax = np.asarray(zs["set_max"])
print("acceptance_rate:", round(d["acceptance_rate"], 3), " accepted:", d["accepted"])
print("ARW ess:", round(zs["ess"], 1), "of", d["accepted"],
      "(impact-only zero -> weight exactly 1)")
print("output IMPACT response (imposed zero):",
      f"[{smin[0,0,0]:+.1e}, {smax[0,0,0]:+.1e}]")
print("output identified set h=0..4  set_min:", np.round(smin[:5, 0, 0], 3))
print("                              set_max:", np.round(smax[:5, 0, 0], 3))
```

```
acceptance_rate: 0.41  accepted: 500
ARW ess: 500.0 of 500 (impact-only zero -> weight exactly 1)
output IMPACT response (imposed zero): [-2.2e-15, +1.9e-15]
output identified set h=0..4  set_min: [-0.    -0.127 -0.129 -0.114 -0.088]
                              set_max: [0.    0.135 0.13  0.098 0.069]
```

The imposed impact zero holds to machine precision (output's contemporaneous
response to the monetary shock is $\pm 2\times10^{-15}$), the impact-only zero
leaves the ARW weight at exactly 1 (`ess` = the full 500 accepted draws), and
the *free* output response at every later horizon straddles zero — the sign and
zero restrictions together simply do not pin its direction, and that envelope is
the finding.

---

## `favar` — factor-augmented VAR

**What it estimates.** A two-step FAVAR (Bernanke-Boivin-Eliasz 2005): extract
`n_factors` principal components from a large informational panel, then fit a
VAR on `[factors, policy]` with the policy variable ordered last, so a Cholesky
shock to the last equation is the recursive policy shock — mapped back onto
every series in the panel via the factor loadings.

**Key arguments.** `panel` (T×N), `policy` (T,), `n_factors`, `lags`, `trend`,
`slow_indices` (variables that do not respond within the period), `horizon`,
`orth=True`.

**How to read the output.** `factors` (T×r), the VAR `params`/`sigma_u`,
`policy_index` (last equation), `irf_panel` (N×(H+1), one row per series) and
`irf_policy` (the rate's own response). Panel IRFs start at exactly zero on
impact under the recursive ordering, then build with the sign of each series'
loading.

**Validated against.** The factor step against NumPy's PCA/SVD
(`fixtures/favar.json`); the recursive IRF is built on the validated VAR core.

**References.** Bernanke, Boivin & Eliasz (2005); Stock & Watson (2002, factors).

---

## `connectedness` — Diebold-Yilmaz spillovers

**What it estimates.** A directional connectedness table from a VAR's
*generalized* forecast-error variance decomposition (order-invariant, Pesaran-
Shin): who transmits shocks to whom, in percent.

**Key arguments.** `data` (T×k), `lags`, `horizon`, `trend`.

**How to read the output.** `total` (system-wide spillover index), `to_others`
/ `from_others` / `net` (per variable), the `gfevd` matrix, and `pairwise_net`.
Positive `net` marks a net transmitter; negative a net receiver.

**Validated against.** Diebold-Yilmaz (2012) connectedness on a VAR(2) of macro
data, GFEVD row-normalized (`fixtures/connect.json`).

**References.** Diebold & Yilmaz (2012, 2014); Pesaran & Shin (1998, GFEVD).

```python
import numpy as np, tsecon
rng = np.random.default_rng(0)
k, n = 3, 400
A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.5]])
Y = np.zeros((n, k))
for t in range(1, n):
    Y[t] = A @ Y[t - 1] + 0.3 * rng.standard_normal(k)
c = tsecon.connectedness(Y, lags=2, horizon=10)
print("total connectedness:", round(c["total"], 1), "%  net:", np.round(c["net"], 2))
```
