# Model card — Local projections

`lp` · `lp_iv` · `lp_multiplier` · `lp_state` · `smooth_lp`

The modern impulse-response workhorse. Instead of inverting a fitted VAR, a
local projection runs one regression *per horizon* — regress the outcome `h`
periods ahead on today's shock (plus controls) — and reads the sequence of
slope coefficients as the impulse response. Robust to misspecification of the
long-run dynamics, and honest about uncertainty at each horizon separately.

---

## `lp` — local projection IRFs

**What it estimates.** For each horizon `h = 0..H`, the coefficient on the shock
in a regression of `y_{t+h}` on `shock_t` and lagged controls. The collected
coefficients are the impulse response. `cumulative` selects which side(s)
accumulate over the horizon: `False`/`"none"` (level response), `True`/
`"outcome"` (the cumulated outcome `sum_j y_{t+j}` on the *contemporaneous*
shock — a cumulative impulse response), or `"both"` (cumulated outcome on
cumulated shock). See [the multiplier trap](#lp_multiplier-integral-multipliers)
before reaching for a multiplier.

**Assumptions.** The shock is exogenous conditional on the controls (already an
identified shock — a monetary surprise, a narrative series, a Cholesky
innovation). Serial correlation in the horizon-`h` residuals is expected and
must be handled by the standard errors, not assumed away.

**When to use (and when not).** Use when you want horizon-robust responses,
state dependence, or a shock series you trust more than a full VAR
identification; LP responses need no stability or invertibility assumption. Use
a VAR instead when you need a tight, model-consistent long-horizon response
from short samples — LP standard errors widen with the horizon and can be noisy
far out.

**Key arguments and defaults (and why).** `horizons` (H). `n_lag_controls` sets
how many own-lags enter as controls. `se=None` (the default) resolves to
`"lag_augmented"` — the **recommendation** (Montiel Olea & Plagborg-Møller
2021): it augments the regression with the impulse's own lags so the response is
inference-robust even under persistence, without hand-tuning a bandwidth —
**except under `cumulative="both"`, where the default resolves to `"hac"`**
(Newey-West with `maxlags`, defaulting to `h + n_lag_controls`). The exception
is not a taste: lag augmentation works by making the horizon-`h` score serially
uncorrelated, and it does that by projecting out *past* shocks. Under `"both"`
the regressor is `Σ_{j=0..h} shock_{t+j}`, so base times up to `h` apart share
**future** shocks that no past-lag augmentation can reach; the score is then
serially correlated and HC1 standard errors omit the overlap entirely. The
audit measured the damage: a nominal 95% interval covered **0.507** at `h=12`,
with a reported `se` flat across horizons while the true sampling sd grew
2.7×, and quadrupling `T` did not repair it — the shortfall matches the omitted
autocovariance terms in closed form. `se="hac"` restores 0.90+ (its default
bandwidth `h + n_lag_controls ≥ h` covers the induced MA(`h`) overlap at every
horizon), so that is what the default now selects for this mode, and an
explicit `se="lag_augmented"` with `cumulative="both"` **raises** rather than
answering wrongly. `cumulative` as above — note `True` is a cumulative *impulse
response*, not a multiplier. For the other cumulation modes the lag-augmented
default is measured sound (0.92–0.97 on the same draws): only the cumulated
*impulse* imports future shocks into the regressor.

**How to read the output.** `horizons`, `irf` (the response path), `se` (one
standard error per horizon — build bands as `irf ± z·se`), and `se_method` —
the inference route actually used (`"lag_augmented"` or `"hac"`); report it,
since the default depends on `cumulative`. Plot `irf` against
`horizons`; the per-horizon `se` widening is a feature, not a defect. `irf ±
z·se` is a **pointwise** band and makes no promise about the path as a whole; for
that, see [the band selector](#simultaneous-bands-over-the-horizons-lp).

**Failure modes.** Feeding a *non*-identified shock (a raw endogenous variable)
returns a correlation, not a causal response — use `lp_iv`. Too few
`n_lag_controls` leaves the shock endogenous; very long horizons on short
samples give wide, unstable bands.

**Validated against.** statsmodels OLS with HAC (Newey-West) standard errors,
horizon by horizon (`fixtures/lp.json`).

**References.** Jordà (2005); Montiel Olea & Plagborg-Møller (2021,
lag-augmented inference); Plagborg-Møller & Wolf (2021, LP = VAR).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n = 400
shock = rng.standard_normal(n)
y = np.zeros(n)                              # y_t = sum_h 0.9^h * shock_{t-h} + noise
for t in range(n):
    y[t] = sum(0.9 ** h * shock[t - h] for h in range(min(t, 20) + 1))
y += 0.3 * rng.standard_normal(n)

out = tsecon.lp(y, shock, horizons=12, n_lag_controls=4)   # lag-augmented SEs (default)
print("IRF (h=0..3):", np.round(out["irf"][:4], 3))        # ~[1.0, 0.9, 0.81, 0.73]
print("SEs (h=0..3):", np.round(out["se"][:4], 3))
```

### Simultaneous bands over the horizons — `lp`

**What it estimates.** The same `irf` and the same per-horizon `se`, with one
multiplier `c` chosen so that the **whole path** `irf_h ± c·se_h`, `h = 0..H`,
is inside the band at once. The default band is pointwise and promises nothing
about the path: the interval-coverage audit measured a nominal 90% pointwise LP
band containing all 13 horizons in **36.5%** of samples at T=240 and **42.7%** at
T=720. Tripling the sample bought six points; this is multiplicity, not a
small-sample caveat.

**Assumptions.** `band="sup-t"` needs the `K x K` cross-horizon covariance
of the response path. LP fits every horizon in its own regression, so only the
diagonal of that matrix ever existed; it is assembled from the
Frisch-Waugh-Lovell influence representation and is positive semi-definite by
construction. On the default `se="lag_augmented"` path its `sqrt(diag)`
reproduces the reported `se` to floating-point noise — that agreement is the
check that the covariance and the standard errors are the *same* estimator. On
`se="hac"` one **common** Bartlett bandwidth serves the whole matrix (the
per-horizon default `maxlags` grows with `h`), so `sqrt(diag)` can
differ from the reported `se`; the multiplier uses only the correlation matrix,
and the largest relative gap is reported as `cov_se_max_rel_diff` so you can
see how far apart they were (the audit reconstructed gaps up to ~7.5% on a
routine design — this is not a diagnostic that is always ≈0).

**Key arguments and defaults (and why).** `band=None` is the **default** and
returns exactly what `lp` always returned — the point path and its standard
errors, no band at all. `band="pointwise"` adds the ordinary per-horizon band;
`"sup-t"` is the one to prefer when you want a joint statement; `"sidak"` and
`"bonferroni"` are closed forms that need nothing but `K`. The band's level is
its own argument, `band_alpha` — a band is not the same object as an `se`.
`band_n_sim` and `band_seed` drive the sup-t simulation and make the band a
**pure function** of `band_seed`; do not cut `band_n_sim` far down, since this is
a quantile in the tail of a maximum.

**How to read the output.** Asking for a band adds `lower`/`upper`,
`critical_value`, `pointwise_critical_value`, `band_scope`, `n_cells` (the `K`),
`n_cells_used` and `cov_se_max_rel_diff` (the sup-t diagnostic above —
~machine epsilon on the lag-augmented path, materially non-zero on the HAC
path, `None` on the routes that build no covariance). The family is fixed and
simple here — the horizons of this
one response, `K = horizons + 1`, and `band_scope` echoes `"horizon"` — but
**report it anyway**, together with the method. The ratio of `critical_value` to
`pointwise_critical_value` is exactly what simultaneity cost on this path. At
`K = 13`, `alpha = 0.10` the closed forms are fixed — Šidák 2.6490, Bonferroni
2.6653 against a pointwise 1.6449 — while sup-t depends on the path: the audit
measures it averaging 2.0742 on a moderately persistent VAR IRF and running up
to about 2.65 on more persistent ones. Read the value your own fit returns.

**Failure modes.** A simultaneous band fixes multiplicity and inherits
everything else — if a horizon's standard error is too small, widening the
multiplier does not repair it. Šidák and Bonferroni are conservative on an IRF
path, whose adjacent horizons are strongly positively correlated; use them only
where sup-t is unavailable.

**Validation target.** Nominal 90%, 13 horizons, 400 replications (MC standard
error ≈ 1.5pp), pointwise and sup-t scored on the **same** replications:
**36.5% → 81.8%** at T=240 and **42.7% → 89.5%** at T=720. These are the crate's
own coverage tests: `lp` has **no arm in the Python coverage harness** yet, so
they do not appear in the audit's tables. LP is the clean case
in the library: at T=720, where the per-horizon rates sit on nominal, the
simultaneous band lands on nominal too. Where the marginals are off (T=240), so
is the joint rate — see
[pointwise is not joint](../../examples/interval-coverage.md#the-remedy-and-the-two-places-it-stops).

**References (bands).** Montiel Olea and Plagborg-Møller, *Simultaneous
confidence bands: theory, implementation, and an application to SVARs*.

---

## `lp_iv` — instrumented local projections (LP-IV)

**What it estimates.** The same horizon-by-horizon response, but the impulse
variable is *endogenous* and instrumented — the coefficient is identified by an
external instrument (a proxy: high-frequency surprise, narrative shock). For a
**multiplier** (e.g. the fiscal multiplier) use `lp_multiplier` below, not
`lp_iv(..., cumulative=True)`.

**Assumptions.** Instrument relevance (a strong first stage) and exogeneity
(the instrument affects `y` only through the impulse). Weak instruments bias
the response and understate uncertainty.

**Key arguments.** `impulse` (endogenous), `instrument`, `horizons`,
`n_lag_controls`, `cumulative` (`False`/`"none"`, `True`/`"outcome"`,
`"both"`). The instrument stays contemporaneous under every cumulation mode.

**How to read the output.** `horizons`, `irf`, `se`, and **`first_stage_f`** —
the first-stage F at each horizon. Treat `first_stage_f` below ~10 as a
weak-instrument warning: the point estimate and band at that horizon are not to
be trusted.

**Bands over the horizons — closed forms only.** `band` defaults to `None` (no
band). `"pointwise"`, `"sidak"` and `"bonferroni"` add `lower`/`upper` over the
horizons of this response at level `band_alpha`, with `critical_value`,
`pointwise_critical_value`, `n_cells` and `n_cells_used`; **`band="sup-t"` is
refused, with an error naming the reason.** LP-IV has no cross-horizon covariance in this library — the
kernel covariance is formed one horizon at a time and no joint object exists —
so there is nothing for a sup-t simulation to draw from. None of these bands may
be described as sup-t. Šidák and Bonferroni are valid under arbitrary dependence
across horizons and are simply *wider* than a sup-t band would be; on a
persistent response path that gap is real. And the multiplicity question sits on
top of the marginal one, which is already off here: the audit measured **0.930 ±
0.005** at the *best* horizon against a nominal 0.95.

**Failure modes.** Weak instruments (low `first_stage_f`) are the dominant
failure; a proxy correlated with other shocks violates exogeneity silently.

**Validated against.** `linearmodels` IV2SLS with a Bartlett-kernel HAC
covariance, horizon by horizon (`fixtures/lp.json`).

**References.** Stock & Watson (2018); Ramey & Zubairy (2018); Jordà, Schularick
& Taylor (2015).

---

## `lp_multiplier` — integral multipliers

**What it estimates.** The Ramey-Zubairy (2018) **integral multiplier** by
one-step LP-IV. At each horizon `h`:

```text
sum_{j=0..h} y_{t+j} = m_h * sum_{j=0..h} x_{t+j} + c
                     + sum_{l=1..p} (phi_l y_{t-l} + psi_l x_{t-l}) + u_{t,h}
```

with the cumulated impulse instrumented by the **contemporaneous** instrument.
Both sides accumulate over the same window, so `m_h` is extra cumulated outcome
per extra cumulated impulse — a multiplier, in the units of the two series.

**Why this is its own function.** A cumulative response and a multiplier differ
only in whether the *denominator* accumulates too, and nothing about the call
site tells you which one you got. `lp_iv(..., cumulative=True)` accumulates only
the outcome: its coefficient is cumulated output per unit of *contemporaneous*
spending, so it inherits the growth of the spending path instead of measuring
anything per-dollar. On the Ramey-Zubairy data that quantity runs from 7.4 at
h = 4 to 48.7 at h = 20, with a first-stage F of 1.68 — while the actual
multiplier sits flat around 0.7 with F above 10 throughout. Giving the correct
estimator its own name makes the correct thing the easy thing to write.

**Assumptions.** Instrument relevance and exogeneity, as for `lp_iv`. Additional
to `lp_iv`, the design controls for `n_lag_controls` lags of the **impulse** as
well as the outcome: the denominator is now an endogenous quantity whose own
dynamics have to be soaked up for the ratio to be interpretable.

**Key arguments.** `y` (outcome), `impulse` (the endogenous quantity being
accumulated, e.g. government spending), `instrument`, `horizons`,
`n_lag_controls`, `maxlags` (overrides the default HAC bandwidth `h + p`).

**Standard errors — what `se` is.** The multiplier is estimated as a **single
2SLS coefficient**, not as a ratio of two separately estimated responses, so
`se` is the kernel-HAC standard error of the parameter actually being reported.
It is not a delta-method approximation to a ratio, and it is not one leg's
standard error relabelled. The two reduced-form legs are returned as
`cumulative_outcome` and `cumulative_impulse` for transparency and carry **no**
standard errors; by the just-identified IV algebra their ratio equals
`multiplier` to numerical precision.

**How to read the output.** `horizons`, `multiplier`, `se`, `first_stage_f`,
`cumulative_outcome`, `cumulative_impulse`, `nobs_per_h`. Treat `first_stage_f`
below ~10 as a weak-instrument warning at that horizon.

**Bands over the horizons — closed forms only.** As for `lp_iv`: `band` defaults
to `None`, `"pointwise"`/`"sidak"`/`"bonferroni"` put `lower`/`upper` around
`multiplier` at level `band_alpha`, and **`band="sup-t"` is refused with an error
naming the reason** — no cross-horizon covariance of the multiplier path exists.
Do not call the resulting band sup-t. Note also that the horizons of an integral
multiplier are *nested* windows and therefore very strongly dependent, which is
exactly the case where a closed-form multiplier is at its most conservative.

**Failure modes.** A weak instrument in the *cumulated* first stage; an impulse
that is not measured in the same units as the outcome (the coefficient is then
an elasticity-like object, not a multiplier — this is why Ramey-Zubairy divide
by potential output rather than logging).

**Validated against.** The published Ramey & Zubairy (2018) headline: 0.64-0.74
across h = 4..20 on the authors' own data, inside their reported 0.6-0.8 range
— see the [replication](../../examples/replication-ramey-zubairy.md).

**References.** Ramey & Zubairy (2018); Gordon & Krenn (2010, the potential-output
normalisation); Stock & Watson (2018).

```python
r = tsecon.lp_multiplier(y, g, news, horizons=20, n_lag_controls=4)
r["multiplier"][8]      # dollars of output per dollar of spending through h=8
r["se"][8]              # standard error OF the multiplier
r["first_stage_f"][8]   # weak-instrument diagnostic
```

---

## `lp_state` — state-dependent local projections

**What it estimates.** Ramey-Zubairy (2018) interacted local projections: the
shock is interacted with a state indicator so the impulse response is allowed to
differ across regimes (e.g. recession vs expansion, slack vs tight). One
regression per horizon delivers a separate IRF and SE for each state.

**Assumptions.** Same exogeneity requirement as `lp`, plus a state indicator
that is predetermined (does not itself respond to the shock within the period).

**Key arguments.** `state_indicator` (per-period 0/1, or a continuous transition
weight), `horizons`, `n_lag_controls`, `se` (`None` resolves to lag-augmented,
except under `cumulative="both"` where it resolves to HAC, exactly as in `lp`
and for the same reason — the cumulated impulse shares future shocks across
nearby base times, and the audit measured 0.640 coverage at a nominal 95% for
the lag-augmented pairing here; `se="lag_augmented"` with `"both"` raises),
`cumulative` (`False`/`"none"`, `True`/`"outcome"`, `"both"`).

**How to read the output.** `horizons` and, per regime, `irf_state1`/`se_state1`
and `irf_state0`/`se_state0`, plus `se_method` (the inference route actually
used, shared by both regimes). Compare the two paths — a gap that exceeds the
combined bands is the state-dependence finding.

**Bands over the horizons — closed forms only, and per regime.** `band` defaults
to `None`; `"pointwise"`, `"sidak"` and `"bonferroni"` add **one band per
regime** — `lower_state1`/`upper_state1` and `lower_state0`/`upper_state0`, with
`critical_value_state1`/`critical_value_state0` and the matching
`n_cells_used_*` — each over that regime's own horizons at level `band_alpha`.
**Nothing here is simultaneous *across* regimes**: if your claim is that the two
paths differ, that is a larger family than either band answers for.
**`band="sup-t"` is refused with an error naming the reason**, because the
interacted design produces no joint cross-horizon covariance here.

**Failure modes.** Thin regimes (few periods in one state) give noisy,
unreliable per-state estimates; a state that reacts to the shock contaminates
the split.

**Validated against.** Built on the validated `lp` OLS-HAC machinery (the
interacted design of Ramey-Zubairy 2018); shares the `fixtures/lp.json` golden.

**References.** Ramey & Zubairy (2018); Tenreyro & Thwaites (2016).

The DGP below builds in real state dependence — the shock's effect is 1.5 in
regime 1 and 0.5 in regime 0, decided by the regime the shock *landed* in — so
the two estimated IRFs have something genuine to disagree about:

```python
import numpy as np, tsecon
rng = np.random.default_rng(0)
n = 600
shock = rng.standard_normal(n)
state = ((np.arange(n) // 40) % 2).astype(float)   # 40-period spells: 0,1,0,1,...
y = np.zeros(n)
for t in range(n):
    # The multiplier depends on the regime the shock LANDED in:
    # 1.5 in state 1 (slack), 0.5 in state 0 (tight), both decaying at 0.9.
    y[t] = sum(0.9 ** h * (1.5 if state[t - h] == 1.0 else 0.5) * shock[t - h]
               for h in range(min(t, 20) + 1))
y += 0.3 * rng.standard_normal(n)

out = tsecon.lp_state(y, shock, state, horizons=8, n_lag_controls=2)
print("state 1 IRF (h=0..3):", np.round(out["irf_state1"][:4], 3))
print("state 0 IRF (h=0..3):", np.round(out["irf_state0"][:4], 3))
print("state 1 SE  (h=0..3):", np.round(out["se_state1"][:4], 3))
print("state 0 SE  (h=0..3):", np.round(out["se_state0"][:4], 3))
# state 1 IRF (h=0..3): [1.489 1.505 1.35  1.154]
# state 0 IRF (h=0..3): [0.545 0.396 0.338 0.336]
# state 1 SE  (h=0..3): [0.026 0.08  0.122 0.143]
# state 0 SE  (h=0..3): [0.024 0.042 0.049 0.059]
```

The impact responses recover the true regime multipliers (1.489 vs 0.545
against true 1.5 vs 0.5), each path decays at roughly the true 0.9 rate, and
the gap between them dwarfs the combined standard errors at every horizon —
the state-dependence finding, read exactly as described above.

---

## `smooth_lp` — smooth local projections (Barnichon-Brownlees)

**What it estimates.** The same per-horizon LP regressions as `lp`, but with
the IRF path restricted to a B-spline in the *horizon*,
`beta_h = sum_k theta_k B_k(h)`, and estimated **jointly** across horizons as
one penalized least-squares problem:

```text
theta_hat = (X'X + lambda * P)^{-1} X'y,      P = blkdiag(D_r' D_r, 0)
```

where `D_r` is the r-th difference matrix on the basis coefficients (the
Eilers-Marx P-spline penalty) and the zero block leaves the per-horizon
intercepts and lag controls unpenalized.

**The bias-variance logic.** Raw LP estimates each `beta_h` from its own
regression, so the IRF inherits one regression's noise per point — jagged
paths in short or noisy samples, with the jaggedness carrying no information
(true macro IRFs are smooth). The penalty trades a little bias (shrinking
wiggles) for a lot of variance (pooling information across neighboring
horizons). `lambda` indexes the whole path between two interpretable poles:
`lambda = 0` is exactly raw LP, and `lambda -> inf` with the default
`penalty_order = 2` shrinks the IRF toward a straight line in `h`
(`penalty_order = 1` toward a constant). Cross-validation picks the point on
that path that predicts best.

**The consistency anchor.** With the default interpolating basis
(`n_basis = horizons + 1`), `lam = 0.0` reproduces the per-horizon
`lp(se="hac")` **point estimates exactly** — test-pinned, and shown live in
the example below. Nothing exotic happens at the boundary: smooth LP *is* LP,
plus a penalty you control. (The standard errors at `lam = 0` are close but
not bit-identical to `lp`'s: smooth LP computes one joint HAC covariance over
the stacked problem, aggregating scores that share a base period, rather than
a separate HAC fit per horizon.)

**Assumptions.** Everything `lp` assumes (identified shock, lag controls),
plus one more: the true IRF is *smooth in the horizon*. That is what the
penalty encodes; a genuinely discontinuous response (an announcement effect
that dies in exactly one period) will be over-smoothed.

**When to use (and when not).** Use it when the raw LP path is visibly jagged
— short samples, noisy outcomes, many horizons — and you would otherwise be
tempted to eyeball-smooth the plot; the CV choice does that honestly. Skip it
when samples are long and raw LP is already smooth (the penalty then has
nothing to buy), or when the sharp-kink shape of the response is itself the
finding.

**Key arguments and defaults (and why).** `lam`: a float fixes the smoothing
parameter (`0.0` = raw LP); `"cv"` or `None` (the default) selects it by
leave-h-block-out cross-validation — blocks of adjacent base periods are held
out to respect the serial dependence of the stacked residuals — over
`lambda_grid`. The default grid is **scale-relative**: a 17-point log ladder
spanning eight decades, anchored to the mean diagonal of the spline block of
the stacked `X'X`. The anchor matters because `λ` competes with `X'X`, which
carries the *squared units* of your data — an absolute default grid (the
pre-0.3 behaviour, a fixed 1e-2..1e6 ladder) tied the amount of smoothing to
the units of the series: rescaling the shock by 100 walked the CV optimum off
the grid, pinned `lambda_used` at the endpoint, and changed the
unit-normalized IRF materially. With the anchored grid the *selection* is
exactly invariant to rescaling `y` and/or the shock — `lambda_used` tracks the
units, the unit response does not move. An **explicit** `lambda_grid` is
absolute, in the units of your data, and is used verbatim; `cv_grid` always
reports the grid actually searched. `degree = 3`
(cubic splines), `n_basis = horizons + 1` (the interpolating size that makes
the `lam = 0` anchor exact), `penalty_order = 2` (shrink toward a line),
`n_folds = 5`, `hac_maxlags = horizons + n_lag_controls` by default.

**How to read the output.** `irf`/`se` are the smoothed path and its
delta-method-through-the-basis standard errors; `irf_raw`/`se_raw` are the
unsmoothed per-horizon HAC LP on the same sample — **always plot both**: the
vertical gaps show you exactly what the penalty did. `lambda_used` is the
selected (or fixed) value; `cv_grid`/`cv_scores` expose the whole CV objective
(a `lambda_used` at the top of the grid means "as smooth as allowed" — extend
`lambda_grid` if that worries you). `theta` is the basis coefficient vector.
Two honest caveats on `se`, stated rather than hidden: it conditions on
`lambda` (treated as fixed even when cross-validated) and does not account for
shrinkage bias — bands are around the estimator's own smoothed target.

**Bands over the horizons — `band=...`.** Smooth LP is the one estimator in this
family that already held the joint object. The IRF is `irf_h = B_h' theta` for a
single jointly estimated coefficient vector, so the cross-horizon covariance is
`B V B'`, it is already computed, and the shipped `se` is exactly its
`sqrt(diag)` — bit for bit (the returned `cov_se_max_rel_diff` says so:
~machine epsilon here, unlike `lp`'s HAC path). `band="sup-t"` therefore costs no extra estimation
and makes no approximation that the point estimate has not already made (it
simulates `band_n_sim` draws from `band_seed`, so the band is a pure function of
that seed); `"pointwise"`, `"sidak"` and `"bonferroni"` are the other three, and
`band=None` — no band — remains the default. What a simultaneous band **cannot**
fix is the caveat above: it is a joint statement about the *penalized* path, not
about the truth. The audit
measured the pointwise `lam="cv"` band covering **0.640 ± 0.018** at impact
against a nominal 0.95 with |bias|/sd = 1.22 — no multiplier repairs a band
centred in the wrong place.

**Failure modes.** Over-smoothing a genuinely kinked IRF (compare against
`irf_raw`; if the raw path departs from the band systematically rather than
noisily, lower `lam`). Reading the bands as covering the *unsmoothed* truth —
they condition on the shrinkage, per the caveat above. And treating the CV
choice as sacred: it minimizes out-of-sample prediction error of the stacked
regression, which is a fine but not unique criterion for "the right amount of
smoothing".

**Validated against.** A scipy/NumPy golden
([`fixtures/smoothlp.json`](../../../fixtures/smoothlp.json), generated by
[`fixtures/generate_smoothlp_fixtures.py`](../../../fixtures/generate_smoothlp_fixtures.py)):
the B-spline basis against `scipy.interpolate.BSpline.design_matrix` on the
same knots (1e-10); the penalized `theta`/IRF/sandwich-SE paths against
plain-NumPy normal equations at several `lambda` (~1e-8); the `lambda = 0` IRF
against statsmodels per-horizon OLS (1e-8); and the leave-h-block-out CV
scores and chosen `lambda` against the same rule in NumPy. Property tests pin
the `lambda -> 0` / `lambda -> inf` limits and the MSE gain over raw LP under
a smooth true IRF.

**References.** Barnichon & Brownlees (2019, *Review of Economics and
Statistics* 101:522-530); Eilers & Marx (1996, P-splines); Jordà (2005).

A short, noisy sample where the true IRF is a clean `0.85^h` decay — the
setting the estimator was built for:

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n = 250                                # short sample ...
shock = rng.standard_normal(n)
y = np.zeros(n)
for t in range(n):
    y[t] = sum(0.85 ** h * shock[t - h] for h in range(min(t, 24) + 1))
y += 1.5 * rng.standard_normal(n)      # ... and noisy: raw LP will be jagged

# The consistency anchor: lam=0 IS the per-horizon HAC LP.
s0 = tsecon.smooth_lp(y, shock, horizons=16, n_lag_controls=4, lam=0.0)
base = tsecon.lp(y, shock, horizons=16, n_lag_controls=4, se="hac")
print("max |smooth_lp(lam=0).irf - lp(se='hac').irf| =",
      float(np.max(np.abs(np.asarray(s0["irf"]) - np.asarray(base["irf"])))))

# Cross-validated smoothing.
s = tsecon.smooth_lp(y, shock, horizons=16, n_lag_controls=4, lam="cv")
print(f"lambda_used = {s['lambda_used']:.3g}")
irf, raw = np.asarray(s["irf"]), np.asarray(s["irf_raw"])
true = 0.85 ** np.arange(17)
print("h        :", "  ".join(f"{h:5d}" for h in range(0, 9, 2)))
print("raw LP   :", "  ".join(f"{raw[h]:5.2f}" for h in range(0, 9, 2)))
print("smoothed :", "  ".join(f"{irf[h]:5.2f}" for h in range(0, 9, 2)))
print("true     :", "  ".join(f"{true[h]:5.2f}" for h in range(0, 9, 2)))
print(f"RMSE vs truth: raw {np.sqrt(np.mean((raw - true) ** 2)):.4f}"
      f"  smoothed {np.sqrt(np.mean((irf - true) ** 2)):.4f}")
# max |smooth_lp(lam=0).irf - lp(se='hac').irf| = 1.503241975342462e-13
# lambda_used = 3.7e+05
# h        :     0      2      4      6      8
# raw LP   :  0.95   0.79   0.67   0.49   0.48
# smoothed :  0.90   0.75   0.61   0.46   0.31
# true     :  1.00   0.72   0.52   0.38   0.27
# RMSE vs truth: raw 0.1822  smoothed 0.1436
```

The anchor holds to 1.5e-13, CV lands one notch below the top of its grid (a
noisy short sample wants heavy smoothing; `lambda_used` near the top is the
"as smooth as allowed" signal described above, and extending `lambda_grid` is
the check), and the smoothed path cuts the
RMSE against the true IRF by about a fifth on this draw — the bias-variance
trade doing exactly what it promises: giving up nothing at the horizons where
raw LP was right, and pulling in the ones where noise had it wandering.
