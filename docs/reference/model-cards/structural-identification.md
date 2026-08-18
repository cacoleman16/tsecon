# Model card — Structural identification (advanced)

`long_run_svar` · `max_share_svar` · `proxy_svar` · `proxy_svar_bands` ·
`proxy_ar_sets` · `hetero_svar` · `nongaussian_svar` · `structural_fevd` ·
`historical_decomposition` · `narrative_svar` · `fry_pagan_svar` ·
`robust_svar_bounds`

A structural VAR is a reduced-form VAR plus one identifying assumption that
rotates the estimated residuals into economically meaningful shocks. The
[VAR/SVAR card](var-svar.md) covers the recursive (Cholesky) and
sign-restricted schemes; this card covers two families that build on them.

**Point-identification schemes** ([below](#long_run_svar-blanchard-quah-long-run-restrictions))
spend a *different* kind of outside information — a long-run neutrality, a
variance-share objective, an external instrument, a documented variance regime,
or the non-Gaussianity of the shocks themselves (a distributional assumption
rather than an economic restriction). Each returns a **point** identification:
the estimand is one impact matrix or one structural column. All five take a plain
data matrix, estimate the reduced form internally, and are deterministic — no
RNG, no rejection sampling.

**The external-instrument scheme is the one with honest uncertainty attached.**
`proxy_svar_bands` supplies Jentsch-Lunsford moving-block bootstrap bands, and
`proxy_ar_sets` supplies weak-instrument-robust Anderson-Rubin confidence sets;
the other four schemes are still point-only, and their bands remain an open item.

**Post-identification and prior-robust tools**
([below](#post-identification-and-prior-robust-tools)) do not identify a new
scheme; they *take* an identification (any impact matrix `A0`, or a
sign-restricted set) and answer the questions that come after: how a shock splits
a variable's forecast-error variance (`structural_fevd`); how it drove each
historical observation (`historical_decomposition`); which single coherent draw
sits at the middle of a sign-restricted set (`fry_pagan_svar`); how the
identified set widens once the Haar-prior artifact is removed
(`robust_svar_bounds`); and how episode knowledge from the historical record
shrinks it (`narrative_svar`). These are the answers to the two honesty
critiques the [sign-restriction section](../../guide/08-causal-identification.md#sign-restrictions-honest-bands-not-points)
raises — pointwise medians mix models, and the rotation prior never washes out.

Which one you reach for is a question about *what you can defend*, laid out in
[chapter 8](../../guide/08-causal-identification.md) and the
[decision guide](../../which-model-when.md#2-i-want-an-impulse-response). The
one-line map: **long-run** when theory speaks about permanent vs. transitory
effects; **max-share** when you want the single shock that drives a target's
business-cycle variance; **proxy** when you have a measured instrument for one
shock; **heteroskedasticity** when you have documented variance regimes;
**non-Gaussianity** when you distrust every economic restriction but the shocks
are plausibly non-Gaussian; then the post-identification tools once a scheme is
chosen.

**Within the proxy family the choice is about instrument strength, and it is not
a matter of taste.** Reach for
[`proxy_svar_bands`](#proxy_svar_bands-moving-block-bootstrap-bands-for-the-proxy-svar)
when the instrument is **strong** — a healthy first-stage F, and `n_failed` back
at zero. Reach for
[`proxy_ar_sets`](#proxy_ar_sets-weak-instrument-robust-anderson-rubin-sets) when
the instrument is **weak**, when the first stage is marginal, or whenever
`proxy_svar_bands` returns a **nonzero `n_failed`**: the bootstrap is telling you
its own denominator went near zero, and a Wald-type band is then the wrong
object. Running both is cheap, and disagreement between them is itself the
finding.

---

## `long_run_svar` — Blanchard-Quah long-run restrictions

**What it estimates.** Structural IRFs under the Blanchard-Quah (1989)
frequency-zero restriction: some shocks are constrained to have **zero
cumulative (long-run) effect** on some variables. The classic bivariate case —
output growth and unemployment — imposes that the "demand" shock has no
permanent effect on the level of output, leaving the "supply" shock as the only
source of the stochastic trend. Closed-form, the exact analog of R
`vars::BQ` (Pfaff 2008).

**Assumptions.** A correct reduced form; the long-run neutrality is economically
true; and — the caveat the scheme is famous for — the VAR's largest roots are
not too close to one. The long-run multiplier is $C(1) = (I - A_1 - \cdots -
A_p)^{-1}$, which blows up as persistence approaches a unit root, so small
coefficient errors become large long-run-matrix errors (Faust-Leeper 1997).
Check the VAR's stability before trusting a long-run scheme; prefer a VECM when
cointegration is plausible.

**When to use (and when not).** Use when theory is silent about within-period
timing but loud about the long run (supply/demand decompositions, permanent
vs. transitory income). Do not use on highly persistent levels without checking
the roots; do not read the *impact* matrix as the finding — the restriction
lives at the infinite horizon, so the cumulative IRF is the object to read.

**Key arguments and defaults (and why).** `lags`, `horizon`, `trend="c"`.
`restrictions=None` gives the classic recursive BQ (long-run matrix lower
triangular); pass a list of `(variable, shock)` long-run-zero pairs for a custom
pattern. `normalize="long_run"` (default) makes the long-run diagonal positive;
`"impact"` makes the impact diagonal positive instead — a sign convention, not a
different model.

**How to read the output.** `impact` (B), `long_run` (LR = C(1)·B, the cumulative
structural impact — **check its imposed zeros**), `long_run_multiplier` (C(1)),
`irf` `[h][i][j]`, `cumulative_irf` (the level response for differenced
variables — the one to plot), and `fevd`. The demand shock's *cumulative* effect
on output should visibly decay to zero: that is the restriction, echoed back as a
built-in sanity check.

**Failure modes.** Near-unit roots make the long-run matrix unreliable (fragile,
silent); reading the impact IRF instead of the cumulated one for differenced
data; forgetting that "supply"/"demand" are labels you attach, not properties the
math knows.

**Validated against.** An independent NumPy transcription of the documented
closed form (faer LU inverse + lower Cholesky vs. NumPy) — a cross-implementation
golden ([`long_run_svar.json`](../../../fixtures/long_run_svar.json),
[`long_run.rs`](../../../crates/tsecon-ident/tests/long_run.rs)). See the
[validation matrix](../validation-matrix.md).

**References.** Blanchard & Quah (1989); Faust & Leeper (1997); Pfaff (2008,
`vars`).

```python
import numpy as np, tsecon

# Blanchard-Quah bivariate: output growth (dy) and unemployment (u).
rng = np.random.default_rng(0)
T = 400
es = rng.standard_normal(T)   # supply (permanent)
ed = rng.standard_normal(T)   # demand (transitory)
dy = np.zeros(T); u = np.zeros(T)
for t in range(2, T):
    dy[t] = 0.2 * dy[t - 1] + es[t] + 0.5 * ed[t] - 0.5 * ed[t - 1]
    u[t] = 0.6 * u[t - 1] - 0.3 * es[t] + 0.7 * ed[t]
data = np.column_stack([dy, u])

bq = tsecon.long_run_svar(data, lags=4, horizon=20, trend="c")
lr = np.asarray(bq["long_run"])
print("long-run matrix LR (lower-triangular by construction):\n", np.round(lr, 4))

cum = np.asarray(bq["cumulative_irf"])   # [h][response][shock]
print("cumulative output response to the demand shock, h = 0, 4, 20:",
      np.round(cum[[0, 4, 20], 0, 1], 6))
```

```
long-run matrix LR (lower-triangular by construction):
 [[ 1.2008  0.    ]
 [-0.3017  1.6799]]
cumulative output response to the demand shock, h = 0, 4, 20: [ 3.75038e-01 -1.11681e-01 -1.10000e-05]
```

The upper-right entry of `long_run` is exactly zero — the imposed neutrality —
and output's *cumulative* response to the demand shock (0.375 on impact) decays
to $-1.1\times10^{-5}$ by horizon 20: the level of output returns to baseline, as
the restriction requires.

---

## `max_share_svar` — maximum forecast-error-variance-share shock

**What it estimates.** The single unit-variance structural shock whose share of a
**target** variable's forecast-error variance, accumulated over a horizon window
`[h0, h1]`, is maximal — Uhlig's (2004) penalty-free eigenvalue variant, the
Francis-Owyang-Roush-DiCecio (2014) main-business-cycle shock, and (with a zero
impact) the Barsky-Sims (2011) news shock. Closed-form: the identified impact
direction is the leading eigenvector of a small symmetric PSD matrix built from
the orthogonalized MA coefficients. No rotation sampling.

**Assumptions.** A correct reduced form and a target/window that encode a real
economic question ("the shock that drives medium-run output"). The identified
shock is defined *purely* by the variance objective — it carries no economic
label until you check its IRF signs or its correlation with an external series.

**When to use (and when not).** Use to extract a single dominant driver of a
target's low- or business-cycle-frequency variance without committing to signs
or an ordering — technology/news shocks, "the" financial shock. Do not use it as
a general SVAR (it identifies one shock, not the whole B); do not over-interpret
the label; watch that the leading eigenvalue is well separated from the rest
(otherwise the max-share direction is only weakly pinned down).

**Key arguments and defaults (and why).** `target=0` (the variable whose FEV is
maximized), `h0`/`h1` (the accumulation window — e.g. `6..32` quarters for the
business cycle), `horizon`, `lags`, `trend`. `weighting="window"` (Uhlig/Francis;
maximizes the *incremental* windowed FEV — `share_window` is then an exact
accumulated-FEV fraction) or `"cumulative"` (Barsky-Sims window-mean cumulative
share). `exclude_impact=True` forces zero impact on the target (the Barsky-Sims
news shock). `sign` pins the identified sign (`"cumsum"`/`"impact"`/`"none"`).

**How to read the output.** `share_window` (the maximand — the accumulated-FEV
fraction the identified shock achieves over the window), `impact` `[k]` (its
impact vector), `irf` `[h][k]` (the response of every variable to it),
`fev_share` `[h]` (its share of the target's *total* FEV at each horizon — lower
than `share_window`, because the objective targets the window's incremental
variance, not the total at any one horizon), `q` (the rotation weights), and
`eigenvalues` (ascending; the identified shock is the top eigenvector, and the
gap to the next eigenvalue is the identification margin).

**Failure modes.** A poorly separated leading eigenvalue (the max-share direction
is nearly a tie); reading the FEV-maximizing shock as "the technology shock"
without corroboration; choosing a window that does not match the frequency band
you mean.

**Validated against.** An independent NumPy reference — `numpy.linalg.lstsq` for
the reduced form, `numpy.linalg.cholesky` for the orthogonalization, and a NumPy
eigensolver for the leading eigenvector
([`max_share_svar.json`](../../../fixtures/max_share_svar.json),
[`max_share.rs`](../../../crates/tsecon-ident/tests/max_share.rs)).

**References.** Uhlig (2004); Barsky & Sims (2011); Francis, Owyang, Roush &
DiCecio (2014).

```python
import numpy as np, tsecon

rng = np.random.default_rng(3)
T = 500
eps = rng.standard_normal((T, 3))
B0 = np.array([[0.9, 0.6, 0.5],
               [0.4, 0.9, 0.30],
               [0.3, 0.25, 0.8]])
A1 = np.array([[0.4, 0.05, 0.0],
               [0.1, 0.4, 0.05],
               [0.0, 0.1, 0.45]])
y = np.zeros((T, 3))
for t in range(1, T):
    y[t] = A1 @ y[t - 1] + B0 @ eps[t]

ms = tsecon.max_share_svar(y, lags=2, target=0, h0=6, h1=32, horizon=40,
                           weighting="window", sign="cumsum")
print("share_window (accumulated FEV of variable 0 over [6,32]):", round(ms["share_window"], 4))
print("impact vector:", np.round(np.asarray(ms["impact"]), 4))
print("target response h = 0, 4, 8:", np.round(np.asarray(ms["irf"])[[0, 4, 8], 0], 4))

# Barsky-Sims news shock: zero impact on the target, cumulative weighting
news = tsecon.max_share_svar(y, lags=2, target=0, h0=0, h1=40, horizon=40,
                             exclude_impact=True, weighting="cumulative")
print("news-shock impact on target (forced to zero):",
      round(float(np.asarray(news["impact"])[0]), 6))
```

```
share_window (accumulated FEV of variable 0 over [6,32]): 0.9499
impact vector: [0.7025 0.9239 0.3703]
target response h = 0, 4, 8: [0.7025 0.0357 0.0028]
news-shock impact on target (forced to zero): 0.0
```

The identified shock explains 95% of variable 0's forecast-error variance
accumulated across the `[6, 32]` window — it *is* the business-cycle driver of
that variable in this synthetic system. Flipping `exclude_impact=True` re-poses
the problem as a news shock and drives the impact response to an exact zero.

---

## `proxy_svar` — external-instrument identification (SVAR-IV)

**What it estimates.** One structural shock's impact column from a single
external instrument (proxy) — the modern applied default for monetary and tax
questions (Stock-Watson 2018; Mertens-Ravn 2013; Gertler-Karadi 2015). The
covariance of the instrument with the reduced-form residuals pins the target
shock's impact column *up to scale*; a unit-effect normalization fixes the scale
and sign. Nothing is assumed about the other columns of B — all you need if one
shock is the question.

**Assumptions.** The instrument is **relevant** ($\mathbb{E}[z\varepsilon_1]\ne0$)
and **exogenous** ($\mathbb{E}[z\varepsilon_j]=0$ for $j\ne1$). Relevance is
testable (the first-stage F); exogeneity is the identifying assumption you must
defend. A weak proxy makes the normalized IRFs heavy-tailed and conventional
bands junk — check `first_stage_f` first.

**When to use (and when not).** Use with a measured surprise or narrative series
(high-frequency futures surprises, Romer-Romer shocks) — especially when the
system contains fast-moving financial variables that admit no defensible Cholesky
ordering. Do not report a point IRF as if it had a band — `proxy_svar` itself
returns none; reach for [`proxy_svar_bands`](#proxy_svar_bands-moving-block-bootstrap-bands-for-the-proxy-svar)
(strong instrument) or [`proxy_ar_sets`](#proxy_ar_sets-weak-instrument-robust-anderson-rubin-sets)
(weak instrument). Do not proceed on a first-stage F below ~10 with a Wald band.

**Key arguments and defaults (and why).** `proxy` aligns to `data` rows (length
`n_obs` — the first `lags` presample rows are dropped — or the residual length
`T`); **NaN entries outside the instrument's availability window are dropped**
from the moments and the first stage, so a short/gappy proxy is handled
correctly. `norm_var=0` and `unit=1.0` set the normalization (a positive shock
raises `norm_var` by `unit` on impact). `lags`, `horizon`, `trend`,
`robust_f=True`.

**How to read the output.** `impact`/`relative_impact` (the identified column,
normalized), `irf` `[h][n]`, `first_stage_f` (**weak below 10**), `reliability`
= Corr(m, u_norm)² (how much of the normalized residual the proxy explains),
`cov_um` (the raw residual-instrument covariances), `n_proxy` (effective
non-missing obs), and the estimated structural `shock` (length T).

**Failure modes.** A weak instrument reported with a Wald band (the cardinal
sin — `proxy_ar_sets` exists for exactly this case); dividing by a
near-zero impact coefficient in the normalization (fragility); silently
truncating a short proxy to the overlap and misaligning it with the residuals —
which the NaN-drop path is designed to prevent.

**Validated against.** An independent reference — statsmodels VAR for the reduced
form and its MA representation, plus plain-NumPy method-of-moments for the
identification ([`proxy_svar.json`](../../../fixtures/proxy_svar.json),
[`proxy.rs`](../../../crates/tsecon-ident/tests/proxy.rs)).

**References.** Mertens & Ravn (2013); Gertler & Karadi (2015); Stock & Watson
(2018); Montiel Olea, Stock & Watson (2021, weak-IV-robust bands).

```python
import numpy as np, tsecon

rng = np.random.default_rng(5)
T = 500
eps = rng.standard_normal((T, 3))     # structural: [output, prices, policy]
mono = eps[:, 2]                      # the policy shock is column 2
B0 = np.array([[0.8, -0.2, -0.5],     # variables: output, prices, ffr
               [0.3, 0.7, -0.4],
               [0.1, 0.2, 0.9]])
A1 = np.array([[0.5, 0.0, -0.1],
               [0.1, 0.4, 0.0],
               [0.0, 0.1, 0.6]])
y = np.zeros((T, 3))
for t in range(1, T):
    y[t] = A1 @ y[t - 1] + B0 @ eps[t]

proxy = mono + 0.7 * rng.standard_normal(T)   # noisy measure of the policy shock
proxy[:120] = np.nan                          # unavailable early in the sample

pr = tsecon.proxy_svar(y, proxy, lags=2, horizon=16, norm_var=2, unit=1.0)
print("first-stage F (weak below 10):", round(pr["first_stage_f"], 2))
print("reliability Corr(m,u)^2:", round(pr["reliability"], 4), " effective obs:", pr["n_proxy"])
irf = np.asarray(pr["irf"])
print("ffr response  h = 0, 1, 4, 8:", np.round(irf[[0, 1, 4, 8], 2], 4))
print("output response h = 0, 1, 4, 8:", np.round(irf[[0, 1, 4, 8], 0], 4))
```

```
first-stage F (weak below 10): 475.45
reliability Corr(m,u)^2: 0.5797  effective obs: 380
ffr response  h = 0, 1, 4, 8: [1.     0.5947 0.1548 0.0265]
output response h = 0, 1, 4, 8: [-0.6957 -0.3841 -0.0914 -0.0147]
```

The proxy is strong (F ≈ 475) and available on 380 of 500 observations; the
unit-effect normalization sets the impact on the policy rate to exactly 1, and
output falls on impact — the contractionary-policy pattern, identified from one
column with no assumption on the rest of the system.

---

## `proxy_svar_bands` — moving-block bootstrap bands for the proxy SVAR

**What it estimates.** Confidence bands for the `proxy_svar` impulse response —
the Jentsch & Lunsford (2019) **moving-block bootstrap**. The joint pair
$(u_t, m_t)$ — the reduced-form residual vector and the instrument, aligned date
by date — is resampled in overlapping blocks under **one** set of block starts,
so the residual and its instrument travel together and the identifying moment
$\sum_t m_t u_t'$ inherits real sampling variability. Inside every draw the VAR
is reconstructed recursively, **re-estimated**, re-identified, and the
unit-effect normalization is **re-imposed**. Nothing is held fixed at its sample
value; the estimator is run end to end $B$ times.

**Why the moving block and not the wild bootstrap.** Mertens-Ravn (2013) and
Gertler-Karadi (2015) draw a common Rademacher weight $e_t\in\{-1,+1\}$ and apply
it to *both* the residuals and the proxy: $u^*_t = e_t\hat u_t$, $m^*_t = e_t
m_t$. Then

$$m^*_t u^{*\prime}_t = e_t^2\, m_t \hat u_t' = m_t \hat u_t'$$

because $e_t^2 = 1$ identically. **The identifying moment is bit-identical in
every draw** — verified 200/200 with a maximum deviation of exactly
`0.000e+00` — so the wild bootstrap carries *no* variability at all in the step
that does the identifying. That is not a small distortion: the crate's
Monte-Carlo test measures the wild arm covering **0.113** at impact for a nominal
0.90, with a mean interval width of 0.018 against the moving block's 0.173,
against a moving-block impact coverage of 0.860. In the worked example below the
wild impact band is 0.0240 wide where the moving block's is 0.2128 — 11% of the
honest width. `bands="wild"` is offered because reproducing those published bands
is a legitimate thing to want, and it sets `asymptotically_valid=False` with a
`validity_note` saying so. Do not quote it as inference.

**Assumptions.** Everything `proxy_svar` assumes, plus a **strong** instrument:
these are strong-instrument asymptotics and the band is a Wald-type object. A
correct reduced form, a block length long enough to carry the serial dependence,
and enough effective proxy observations per block. When the instrument is weak
the band is not merely wide — it is *wrong*, and `proxy_ar_sets` is the object to
report instead.

**The `h=0` cell of `norm_var` is degenerate by construction.** The unit-effect
normalization pins variable `norm_var`'s impact response to exactly `unit` in
every draw, so its band is `[unit, unit]` — verified `[1.000000, 1.000000]`. That
is the free proof that the normalization is re-imposed *inside* the loop; a
non-degenerate value there would mean it had been hoisted out, which is the
classic way to get bands that look plausible and are not.

**The bands are POINTWISE, not joint.** A nominal $1-\alpha$ band covers each
$(h,\ \text{variable})$ cell at that rate. It does **not** cover the whole
impulse-response path simultaneously, and reading "the path lies inside the band
with 90% probability" off a pointwise band overstates what was computed. No
simultaneous band exists anywhere in this library.

**Key arguments and defaults (and why).** `alpha=0.10` (a 90% band, the
proxy-SVAR convention), `n_boot=2000`, `seed=0` (bit-reproducible).
`bands="moving_block"` (alias `"mbb"`) is the default and the only valid arm;
`"wild"` is the reproduction arm above. `block_length=None` picks a default from
the effective sample; pass an integer to override, and check that the answer is
not sensitive to it. `lags`, `horizon`, `norm_var`, `unit`, `trend`, `robust_f`
are `proxy_svar`'s.

**How to read the output.** `point` `[h][n]` (the `proxy_svar` IRF),
`lower`/`upper` (the **Hall / basic** band — the recommended one), and
`lower_efron`/`upper_efron` (the percentile band Mertens-Ravn and Gertler-Karadi
report). The two differ materially when the bootstrap distribution is skewed:
they are reflections of each other about the point estimate, so a right-skewed
draw distribution moves them in opposite directions. `se` is the bootstrap
standard deviation. `n_boot`/`n_used`/`n_failed` and `block_length`, `alpha`,
`method`, `asymptotically_valid`, `validity_note` describe what was run. The
per-draw diagnostic series — `gamma_norm_draws`, `first_stage_f_draws`,
`reliability_draws`, `rho_draws` — let you see the identification strength move
across draws, with `point_gamma_norm`, `point_first_stage_f`,
`point_reliability`, and `n_proxy` as their sample counterparts.

**Failed draws are counted, never dropped.** `failures` is a dict of **six**
counters — `too_few_proxy_obs`, `zero_proxy_variance`, `near_zero_gamma_norm`,
`refit_failed`, `identification_failed`, `non_finite` — and `n_failed` is their
total, with a `failure_warning` when it is nonzero. Silently discarding failed
draws would be the worst available choice: the failures are exactly the
near-zero-denominator tail, so dropping them trims the heavy side of the
distribution and *shrinks* the interval precisely when the instrument is weakest.
A nonzero `n_failed` is a signal that a Wald-type band is the wrong object —
switch to `proxy_ar_sets`.

**Failure modes.** Quoting the wild arm as inference (it is labelled invalid for
a reason); reading a pointwise band as a path statement; a block length too short
for the residual dependence; treating a weak instrument's wide-but-bounded band
as if width alone made it honest. The moving-block arm's own shortfall at longer
horizons (measured 0.78-0.81 for a nominal 0.90) is **inherited from the
reduced-form VAR bootstrap**, not introduced by the proxy layer — the Cholesky
reference lands within 0.07 at every horizon on the same replications, and this
build offers no Kilian bias correction on the proxy path. That is the honest
cost, documented rather than tuned away.

**Validated against.** A **documented-formula golden**, not an
independent-package match — no external package implements JL moving-block
proxy-SVAR bands, so there is no third-party number to copy. The generator
transcribes the documented algorithm into plain NumPy (with statsmodels' `VAR`
cross-checking the reduced form) and never imports tsecon; the block starts are
*pinned in the fixture* so the RNG becomes a shared input and everything
downstream — position-wise centering, reconstruction, re-estimation, per-draw
re-identification and re-normalization, both interval types — is compared cell
for cell ([`proxy_svar_bands.json`](../../../fixtures/proxy_svar_bands.json),
[`proxy_bands_golden.rs`](../../../crates/tsecon-var/tests/proxy_bands_golden.rs);
asserted 1e-10, largest observed deviation 6.7e-16). That pins the *arithmetic*,
not the *theory*. The theory is carried by
[`proxy_bands_props.rs`](../../../crates/tsecon-var/tests/proxy_bands_props.rs):
seed reproducibility, the degenerate impact cell, joint-versus-independent
blocking, the frozen-moment proof for the wild arm, failure accounting, and
seeded Monte-Carlo coverage against a known-truth DGP. See the
[validation matrix](../validation-matrix.md).

**References.** Jentsch & Lunsford (2019); Mertens & Ravn (2013); Gertler &
Karadi (2015); Stock & Watson (2018); Hall (1992, the basic/Hall interval).
Citation details beyond author and year are not asserted here.

```python
import numpy as np, tsecon

# the same system as the proxy_svar example above
bd = tsecon.proxy_svar_bands(y, proxy, lags=2, horizon=16, norm_var=2, unit=1.0,
                             alpha=0.10, n_boot=2000, seed=0)
print("method:", bd["method"], " block_length:", bd["block_length"],
      " asymptotically_valid:", bd["asymptotically_valid"])
print("n_used:", bd["n_used"], " n_failed:", bd["n_failed"], " failures:", bd["failures"])

pt = np.asarray(bd["point"]); lo = np.asarray(bd["lower"]); hi = np.asarray(bd["upper"])
print("h=0 ffr cell (degenerate at unit by construction): [%.6f, %.6f]" % (lo[0, 2], hi[0, 2]))
for h in [0, 1, 4, 8]:
    print(f"h={h} output<-policy {pt[h,0]:+.4f}  90% Hall [{lo[h,0]:+.4f}, {hi[h,0]:+.4f}]")

# Hall and Efron disagree when the bootstrap distribution is skewed
loe = np.asarray(bd["lower_efron"]); hie = np.asarray(bd["upper_efron"])
print("h=1 output  Hall [%+.4f, %+.4f]   Efron [%+.4f, %+.4f]"
      % (lo[1, 0], hi[1, 0], loe[1, 0], hie[1, 0]))

# the wild arm: reproduces the published bands, and says it is not inference
wild = tsecon.proxy_svar_bands(y, proxy, lags=2, horizon=16, norm_var=2,
                               alpha=0.10, n_boot=2000, seed=0, bands="wild")
wl = np.asarray(wild["lower"]); wh = np.asarray(wild["upper"])
print("\nwild asymptotically_valid:", wild["asymptotically_valid"])
print("impact width  moving_block %.4f   wild %.4f" % (hi[0, 0] - lo[0, 0], wh[0, 0] - wl[0, 0]))
```

```
method: moving_block  block_length: 24  asymptotically_valid: True
n_used: 2000  n_failed: 0  failures: {'too_few_proxy_obs': 0, 'zero_proxy_variance': 0, 'near_zero_gamma_norm': 0, 'refit_failed': 0, 'identification_failed': 0, 'non_finite': 0}
h=0 ffr cell (degenerate at unit by construction): [1.000000, 1.000000]
h=0 output<-policy -0.6957  90% Hall [-0.8013, -0.5885]
h=1 output<-policy -0.3841  90% Hall [-0.4862, -0.2910]
h=4 output<-policy -0.0914  90% Hall [-0.1474, -0.0350]
h=8 output<-policy -0.0147  90% Hall [-0.0262, +0.0008]
h=1 output  Hall [-0.4862, -0.2910]   Efron [-0.4771, -0.2819]

wild asymptotically_valid: False
impact width  moving_block 0.2128   wild 0.0240
```

The instrument is strong, every one of the 2000 draws survives, and all six
failure counters are zero. Output's fall is significant on impact and through
$h=4$, and by $h=8$ the band crosses zero — the response has died out. The
funds-rate cell at $h=0$ comes back as exactly $[1, 1]$: that is the
normalization being re-imposed in every draw, echoed back. And the same data
through the wild arm produce an impact band **one ninth** as wide, which is the
Jentsch-Lunsford result in one number: the interval that looks nine times sharper
is the one whose identifying moment never moved.

---

## `proxy_ar_sets` — weak-instrument-robust Anderson-Rubin sets

**What it estimates.** Weak-instrument-robust confidence **sets** for the
proxy-SVAR impulse response, obtained by inverting the Anderson-Rubin statistic
in **closed form** — no grid search. Under weak identification no *bounded*
confidence set can be honest (Dufour 1997): a procedure that always returns a
bounded interval must under-cover somewhere. Inverting AR instead of building a
Wald interval buys correct coverage at the price of a set that is sometimes not
an interval, and that shape *is* the answer.

**The four shapes.** Each cell reports a `kind`:

| `kind` | The set | Read it as |
|---|---|---|
| `"interval"` | `[lower, upper]`, bounded | the ordinary case — the data pin the response down |
| `"exterior"` | the **complement** of `(excluded_lower, excluded_upper)` — two rays; `lower`/`upper` are $-\infty$/$+\infty$ | the data reject a middle region and nothing else |
| `"whole"` | the entire real line | the data say nothing about this cell |
| `"empty"` | no value survives | the moment condition is rejected everywhere — the model, not the response, is in trouble |

Two degenerate shapes round it out: `"point"` (a single value — what the
normalizing variable's impact cell returns under a strong instrument) and the
one-sided rays `"ray_below"` / `"ray_above"`. Always branch on `kind`; `lower`
and `upper` alone do not tell you which object you have.

**Do not present an exterior set as an interval.** `lower` and `upper` are
$-\infty$ and $+\infty$ for an `"exterior"` cell precisely so that reading them as
endpoints cannot silently produce a plausible-looking number; the rejected region
is in `excluded_lower`/`excluded_upper`, and the set is everything *outside* it.
Plotting `[excluded_lower, excluded_upper]` as a band inverts the finding.

**`excludes_zero` does not establish a sign on an unbounded set.** On a bounded
interval, excluding zero does pin the sign. On an `"exterior"` set it does not:
the two rays can contain large negative *and* large positive values while zero
sits inside the rejected middle. The example below has exactly that — the data
reject $(-0.2778, +4.5066)$ for output's impact response, so `excludes_zero` is
`True`, and $-3$ and $+9$ are both members. "Not zero" and "negative" are
different claims.

**Reduced-form uncertainty is PROPAGATED by default, and it is not optional in
practice.** The AR statistic is built on the identification moment, which treats
the VAR coefficients as known. Every real caller estimates them. Measured at
nominal 0.95 on an estimated VAR, $T=300$, VAR(2), excluding the degenerate
`(norm_var, 0)` cell:

| $h$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| **omitted** | .952 | .529 | .458 | .315 | .247 | .195 | .163 | .135 | **.119** |
| **propagated** | .952 | .953 | .954 | .947 | .941 | .936 | .930 | .922 | **.913** |

That is not a drift, it is a collapse: a nominally 95% set covering 11.9% by
$h=8$. **The propagated row keeps declining past the table's edge** (audit
round 6): on this same DGP the mean coverage at the function's default
`horizon=12` is **0.876–0.894** (worst single cell ≈ 0.85), and on a routine
VAR(1) at $T=250$ it reaches **0.80–0.85**. The misses are one-sided — the
truth sits *above* the set, because the propagated variance shrinks together
with $\hat\Psi_h$ at long horizons — and they fade in $T$ (0.907 by
$T=1000$). Read long-horizon cells as approaching their nominal level from
below; prefer shorter horizons or larger samples when the exact level
matters. At $h=0$ the two agree exactly, because $\Psi_0 = I$ carries no estimated
coefficients. The correction is **conservative under weak instruments** — the
measured weak arm goes from .9413 omitted to **.9908** propagated, because the
extra variance turns exterior sets into the whole line — and erring wide is the
right direction under weak identification. The price is width: the paired median
set-width ratio at $h=8$ is **13.5x**. When
`reduced_form_uncertainty=False` the returned `level` is `None`, deliberately: a
set conditional on the reduced form has no honest $1-\alpha$ label to print.

**Assumptions.** Instrument exogeneity — the identifying assumption AR does *not*
relax, and the one you still have to defend. Relevance is exactly what AR is
robust to, so no first-stage threshold applies. A correct reduced-form
specification. The sets are pointwise across $(h, \text{variable})$ cells, like
the bands: not a joint region over the path.

**Key arguments and defaults (and why).** `alpha=0.05` (95% sets — the AR
convention, unlike the bands' 0.10). `variance="hc0"` is the heteroskedasticity-
robust moment variance; pass `variance="hac"` for a HAC estimate when the proxy
is serially correlated (`hac_lags` then sets its lag count, defaulting to the
Newey-West rule — it applies only on the HAC route, and passing it with
`"hc0"` raises rather than being silently ignored).
`reduced_form_uncertainty=True` — leave it on. `lags`,
`horizon`, `norm_var`, `unit`, `trend` are `proxy_svar`'s.

**How to read the output.** `cells[h][variable]` is the dict described above:
`kind`, `lower`, `upper`, `excluded_lower`, `excluded_upper`, `bounded`,
`excludes_zero`, `point`. Alongside: `level` (the honest $1-\alpha$, or `None`
when propagation is off), `critical_value` (the $c$ that was inverted),
`ar_bound_stat` (the robust relevance statistic $T_O\,\gamma_k^2/\Omega_{kk}$ on
the effective proxy sample $T_O$),
`ar_bounded_all`, `impact`, `n_proxy`, and `reduced_form_uncertainty`.

Boundedness is **all-or-nothing across the whole grid** — it depends only on the
denominator — and the rule is exactly `ar_bound_stat > critical_value`. Note what
that does *not* certify: at 95% the threshold is about 3.84, so a first-stage F
of 4.5 can produce a page of tidy bounded intervals and still be a weak
instrument. `ar_bounded_all=True` means the sets are intervals, not that the
instrument is strong.

**Failure modes.** Reading an exterior set as an interval; inferring a sign from
`excludes_zero` on an unbounded set; turning off `reduced_form_uncertainty` for a
narrower picture and then quoting 95%; reporting an `"empty"` cell as a very tight
result rather than as a specification rejection; forgetting that AR is robust to
weak *relevance* and not at all to a violated *exclusion* restriction; quoting
the nominal level for long-horizon cells without the caveat above (at the
default `horizon=12` the measured coverage is 3–10pp below nominal, one-sided).

**Validated against.** A **co-derived NumPy transcription**, not a third-party
reference: `fixtures/generate_proxy_ar_fixtures.py` takes its reduced form from
statsmodels but writes the AR algebra as a plain-NumPy transcription of the same
specification, by the same author — so agreement is a cross-implementation check
of the arithmetic, not a match against an independent authority. The
**load-bearing** validation is instead a **brute-force grid inversion**: the
closed-form quadratic is proved against a scan that re-tests
$\mathrm{AR}(\lambda)\le c$ directly at thousands of candidate values per cell,
for every shape the set can take, and that needs no external reference at all.
Third, the reduced-form correction `psi_reduced_form_cov` is checked against a
**numerical Jacobian** built by perturbing VAR coefficients one at a time — no
Kronecker product, no companion matrix, no shared code with the analytic route —
and is required to widen every set while leaving the weak-instrument algebra
bit-identical. Plus exact properties (`unit`-equivariance, nesting in the level,
NaN-prefix invariance, the point estimate always lying in its own set,
boundedness all-or-nothing, sets genuinely asymmetric about the point estimate)
and the seeded Monte-Carlo coverage that produced the table above
([`proxy_ar.json`](../../../fixtures/proxy_ar.json),
[`proxy_ar.rs`](../../../crates/tsecon-ident/tests/proxy_ar.rs),
[`proxy_ar_coverage.rs`](../../../crates/tsecon-ident/tests/proxy_ar_coverage.rs);
golden rtol 1e-9 / atol 1e-11). See the
[validation matrix](../validation-matrix.md).

**References.** Anderson & Rubin (1949); Dufour (1997); Staiger & Stock (1997);
Montiel Olea, Stock & Watson (2021). Citation details beyond author and year are
not asserted here.

```python
import numpy as np, tsecon

# same system; `proxy` is the strong instrument from the proxy_svar example
weak = 0.06 * mono + np.random.default_rng(6).standard_normal(T)   # nearly irrelevant

def show(tag, pz, cells_to_print):
    f = tsecon.proxy_svar(y, pz, lags=2, horizon=12, norm_var=2)["first_stage_f"]
    st = tsecon.proxy_ar_sets(y, pz, lags=2, horizon=12, norm_var=2, unit=1.0, alpha=0.05)
    print(f"--- {tag}: first-stage F {f:.2f}, level {st['level']}, "
          f"every cell bounded: {st['ar_bounded_all']}")
    for h, i, lab in cells_to_print:
        c = st["cells"][h][i]
        if c["kind"] == "exterior":
            print(f"  h={h} {lab:6s} {c['kind']:8s} the data REJECT "
                  f"({c['excluded_lower']:+.4f}, {c['excluded_upper']:+.4f}); the set is the "
                  f"two rays outside it")
            print(f"           excludes_zero={c['excludes_zero']}  bounded={c['bounded']}  "
                  f"point {c['point']:+.4f}")
        else:
            print(f"  h={h} {lab:6s} {c['kind']:8s} [{c['lower']:+.4f}, {c['upper']:+.4f}]"
                  f"  excludes_zero={c['excludes_zero']}  point {c['point']:+.4f}")
    return st

st = show("strong proxy", proxy,
          [(0, 0, "output"), (1, 0, "output"), (8, 0, "output"), (0, 2, "ffr")])
show("weak proxy", weak, [(0, 0, "output"), (2, 0, "output"), (8, 0, "output")])

# the Wald band on the SAME weak proxy is bounded and tidy
bd = tsecon.proxy_svar_bands(y, weak, lags=2, horizon=12, norm_var=2, alpha=0.05,
                             n_boot=2000, seed=0)
print("\nWald band on the weak proxy, h=0 output: [%+.4f, %+.4f]   n_failed=%d"
      % (np.asarray(bd["lower"])[0, 0], np.asarray(bd["upper"])[0, 0], bd["n_failed"]))

# what omitting the reduced-form uncertainty buys: a shorter set with no honest level
off = tsecon.proxy_ar_sets(y, proxy, lags=2, horizon=12, norm_var=2, alpha=0.05,
                           reduced_form_uncertainty=False)
on8, off8 = st["cells"][8][0], off["cells"][8][0]
print("h=8 output  propagated [%+.4f, %+.4f]  width %.4f  level %s"
      % (on8["lower"], on8["upper"], on8["upper"] - on8["lower"], st["level"]))
print("h=8 output  omitted    [%+.4f, %+.4f]  width %.4f  level %s"
      % (off8["lower"], off8["upper"], off8["upper"] - off8["lower"], off["level"]))
```

```
--- strong proxy: first-stage F 475.45, level 0.95, every cell bounded: True
  h=0 output interval [-0.8293, -0.5680]  excludes_zero=True  point -0.6957
  h=1 output interval [-0.5090, -0.2612]  excludes_zero=True  point -0.3841
  h=8 output interval [-0.0354, +0.0061]  excludes_zero=False  point -0.0147
  h=0 ffr    point    [+1.0000, +1.0000]  excludes_zero=True  point +1.0000
--- weak proxy: first-stage F 2.88, level 0.95, every cell bounded: False
  h=0 output exterior the data REJECT (-0.2778, +4.5066); the set is the two rays outside it
           excludes_zero=True  bounded=False  point -1.4524
  h=2 output exterior the data REJECT (-0.1060, +0.8205); the set is the two rays outside it
           excludes_zero=True  bounded=False  point -0.4051
  h=8 output whole    [-inf, +inf]  excludes_zero=False  point -0.0149

Wald band on the weak proxy, h=0 output: [-2.7009, +2.1263]   n_failed=0
h=8 output  propagated [-0.0354, +0.0061]  width 0.0415  level 0.95
h=8 output  omitted    [-0.0151, -0.0143]  width 0.0008  level None
```

With a strong instrument the sets are ordinary bounded intervals, and they land
close to the moving-block band on the same data at the same level: the 95% band's
impact cell is $[-0.8205, -0.5639]$ against this set's $[-0.8293, -0.5680]$,
about 0.01 apart at each endpoint. (The band printed in the previous section is a
*90%* band — do not read the two side by side without matching `alpha`.) The
funds rate's impact cell is the degenerate `"point"` at exactly 1 — the
normalization again. Weaken the
instrument to $F\approx2.9$ and the shapes change rather than the widths: impact
becomes an **exterior** set that rejects $(-0.28, +4.51)$ and nothing else, and by
$h=8$ the data have nothing to say at all (`"whole"`). Note what the exterior
cell does *not* license — `excludes_zero` is `True`, yet $-3$ and $+9$ are both
in the set, so there is no sign to report. Meanwhile a Wald band on the identical
data comes back as a tidy $[-2.70, +2.13]$ with zero failed draws: bounded,
plottable, and not entitled to its label. Finally, turning propagation off at
$h=8$ shrinks the set from 0.0415 wide to 0.0008 — a fifty-fold narrowing that
buys nothing, which is why `level` comes back `None` rather than `0.95`.

---

## `hetero_svar` — identification through heteroskedasticity

**What it estimates.** The constant SVAR impact matrix B from **two known
variance regimes** (Rigobon 2003; Lanne-Lütkepohl 2008). The two within-regime
residual covariances satisfy $\Sigma_1 = B\Lambda_1 B'$ and
$\Sigma_2 = B\Lambda_2 B'$ with $\Lambda_r$ diagonal; a generalized
eigendecomposition recovers B (up to column sign and order) — point-identified
**iff** the structural-shock variance ratios are pairwise distinct. No zeros, no
signs, no instruments: identification bought purely from second moments shifting.

**Assumptions.** The regime dates are known and correct; B is genuinely constant
across regimes; and the relative variances genuinely differ. The recovered shocks
are *statistically* identified and carry **no economic labels** — shock 2 is "the
one whose variance rose most," not "the monetary shock," until you attach meaning
via sign patterns or an external correlation.

**When to use (and when not).** Use with documented variance shifts — crisis vs.
calm windows, FOMC-announcement vs. control days (the Rigobon-Sack event-study
variant). Do not use when the relative variances barely differ (identification is
near-singular, with tight-looking bogus errors — read `min_ratio_gap`); do not
plot an unlabeled statistical shock as if it were a named structural shock.

**Key arguments and defaults (and why).** `regime_labels` — length T with exactly
**two** distinct integer values, aligned to observations (the first `lags` are
dropped to match residuals). `base_regime` is the label normalized to $\Lambda=I$
(default: the smaller label). `lags`, `horizon`, `trend`. `sign_normalization`:
`"max"` (largest-magnitude entry per B column made positive; default) or
`"diag"` (diagonal of B made non-negative).

**How to read the output.** `B` (the impact matrix; columns ordered by ascending
variance ratio), `variance_ratios` (the generalized eigenvalues — regime 2's
shock variances relative to regime 1's), `structural_irf` `[h][i][j]`,
`min_ratio_gap` and `ratio_dist_from_unity` (the **identification margins** —
larger is better), `identified` (a bool heuristic), `covariance_equality` (a
Bartlett-corrected Box's M test that the two regimes' covariances actually
differ — its `pvalue` should be small), the two `sigma_regime*`, `regime_sizes`,
and `sign_convention`. No standard errors in this closed-form build.

**Failure modes.** Similar variance ratios across two shocks → their columns of B
are near-unidentified (garbage estimates, bogus tight errors); mislabeling a
statistical shock; regimes that do not actually differ in covariance (the Box's M
test guards this).

**Validated against.** An independent NumPy/SciPy reference for the exact
estimator — pooled OLS reduced form and the generalized eigenproblem in NumPy,
recovering a known B from a simulated two-regime DGP
([`hetero_svar.json`](../../../fixtures/hetero_svar.json),
[`hetero.rs`](../../../crates/tsecon-ident/tests/hetero.rs)).

**References.** Rigobon (2003); Rigobon & Sack (2004); Lanne & Lütkepohl (2008).

```python
import numpy as np, tsecon

rng = np.random.default_rng(9)
T = 1000
B = np.array([[1.0, 0.5],
              [0.4, 1.0]])            # true impact matrix (constant across regimes)
labels = np.zeros(T, dtype=int)
labels[T // 2:] = 1                   # regime 0 first half, regime 1 second half
y = np.zeros((T, 2))
for t in range(T):
    scale = np.array([1.0, 1.0]) if labels[t] == 0 else np.array([2.0, 1.0])
    y[t] = B @ (rng.standard_normal(2) * scale)   # shock 0's variance quadruples in regime 1

het = tsecon.hetero_svar(y, labels, lags=1, horizon=8)
print("identified:", het["identified"], " min variance-ratio gap:", round(het["min_ratio_gap"], 3))
print("variance ratios (regime 1 / regime 0):", np.round(np.asarray(het["variance_ratios"]), 3))
print("recovered B (columns ordered by variance ratio):\n", np.round(np.asarray(het["B"]), 4))
ce = het["covariance_equality"]
print("regimes differ? Box's M p-value:", round(ce["pvalue"], 4))
```

```
identified: True  min variance-ratio gap: 3.09
variance ratios (regime 1 / regime 0): [0.962 4.053]
recovered B (columns ordered by variance ratio):
 [[0.4341 0.9798]
 [1.0013 0.4017]]
regimes differ? Box's M p-value: 0.0
```

The variance ratios (≈1 and ≈4) recover the design — shock 0's variance
quadruples in regime 1 while shock 1's is unchanged — and are well separated
(`min_ratio_gap` ≈ 3.09), so B is identified. The recovered columns match the
true `B = [[1, 0.5], [0.4, 1]]` up to the variance-ratio ordering and scale: the
low-ratio column ≈ true shock 1 `[0.5, 1]`, the high-ratio column ≈ true shock 0
`[1, 0.4]`. Box's M rejects covariance equality (p ≈ 0), confirming the two
regimes genuinely differ — the precondition for the whole scheme.

---

## `nongaussian_svar` — independent-component (non-Gaussian) identification

**What it estimates.** The full structural impact matrix B in
$u_t = B\varepsilon_t$ from the reduced-form residuals **alone** — no sign, zero,
long-run, proxy, or variance-regime restriction — by exploiting the statistical
**independence and non-Gaussianity** of the structural shocks (Lanne, Meitz &
Saikkonen 2017; Gouriéroux, Monfort & Renne 2017). It whitens the residuals by
$\Sigma_u^{-1/2}$, rotates them to be **maximally non-Gaussian** with a
deterministic symmetric FastICA fixed point (Hyvärinen's log-cosh contrast,
identity initialization — bit-reproducible, no RNG), and sets $B = \Sigma_u^{1/2}
Q$ for the recovered orthogonal rotation $Q$. By the ICA theorem (Comon 1994) B
is point-identified up to column sign and order **iff at most one** structural
shock is Gaussian.

**Assumptions.** The structural shocks are **mutually independent** — strictly
stronger than the orthogonality every SVAR assumes — and **at most one is
Gaussian**. Independence is itself an economic claim, and the honest open problem
of the whole family: two shocks driven by a common volatility factor are
dependent and violate it silently (Montiel Olea, Plagborg-Møller & Qian 2022;
Drautzburg & Wright 2023 relax independence into bounds). Plus a correct reduced
form and enough non-Gaussianity to estimate — heavier tails or stronger skew give
sharper identification.

**When to use (and when not).** Use when you distrust every economic restriction
on hand — no defensible recursive ordering, no credible instrument, no documented
variance regime — but the shocks are plausibly non-Gaussian (fat-tailed financial
innovations, skewed macro shocks). It is the data-driven fallback: identification
is bought from the *shape of the shock distribution*, not a story you must defend.
Do **not** use it when the shocks are near-Gaussian (it fails — see below), when
independence is implausible (a common-volatility system), or as a labeled scheme
without corroboration: the recovered columns are *statistically* identified shocks
with no economic names until you check their IRF signs or an external correlation.

**It FAILS under Gaussianity — and says so.** Gaussian shocks have zero excess
kurtosis, and *every* orthogonal rotation of a whitened Gaussian vector is again
i.i.d. Gaussian — there is no "most non-Gaussian" direction to find, so B is **not
identified**. This is the theorem's boundary, not a numerical nuisance: the method
has nothing to exploit. The `shock_kurtosis` diagnostic is the tell — a value near
zero flags a column whose shock is near-Gaussian and therefore weakly (or not)
identified. The example below shows it directly: swap in Gaussian shocks and the
kurtoses collapse to ≈0 while the recovered B drifts far from the truth.

**Column sign and order are conventions.** ICA recovers the shocks only up to
*which column is which* and *each column's sign* — the math cannot know that
"column 0 is the demand shock" or that a positive shock raises output.
`order_by="kurtosis"` (default) orders columns by descending |excess kurtosis|
(most non-Gaussian first); `"colnorm"` orders by impact-column norm. Each column is
then signed so its largest-magnitude entry is positive. Both are **labels you
impose**, exactly as in `hetero_svar` — reorder or re-sign to match your economic
reading and it is the same model.

**Key arguments and defaults (and why).** `lags`, `horizon`, `trend="c"`.
`contrast="logcosh"` is Hyvärinen's general-purpose robust nonlinearity (the
FastICA default). `max_iter=200`, `tol=1e-8` govern the symmetric fixed-point
iteration — from the identity initialization it is deterministic and typically
converges in a handful of steps (`n_iter` reports how many, `converged` whether
`tol` was met). `order_by="kurtosis"` / `"colnorm"` chooses the column-ordering
convention.

**How to read the output.** `impact` (B — its columns are the
one-standard-deviation structural shocks, $BB' = \Sigma_u$ **exactly**),
`rotation` (the orthogonal $Q$ acting on the whitened residuals), `irf` `[h][i][j]`
(the structural IRF, `irf[0]` $=$ `impact`), `shock_kurtosis` `[j]` (each
identified shock's excess kurtosis, in the reported order — **the
identification-strength diagnostic; near zero ⇒ weak or unidentified**), `order`
(the permutation applied), and `converged`/`n_iter`. No standard errors in this
build — an honest bootstrap band is a v2 item.

**Failure modes.** Near-Gaussian shocks (identification silently vanishes — read
`shock_kurtosis`); genuinely dependent shocks violating the independence
assumption (the ICA estimand is then not the structural B); reading an unlabeled
statistical shock as a named one; too few observations to pin down the higher
moments the contrast leans on (the weakest-kurtosis column degrades first).

**Validated against.** An independent NumPy FastICA pipeline
(`numpy.linalg.lstsq` OLS, `numpy.linalg.eigh` for the whitening inverse-square-root
and the decorrelation, `numpy.tanh` for the log-cosh contrast) that never imports
tsecon — a genuine cross-implementation golden bit-matching B, $Q$, the per-shock
excess kurtosis, the structural IRF, the ordering, and the convergence
flag/iteration count (tol 1e-10; achieved ~1e-15). That NumPy reference is itself
cross-checked against `sklearn.decomposition.FastICA` at generation (~4e-16), so
it is a faithful FastICA, not a bespoke re-derivation. Two statistical **property**
checks carry the estimand: the recovered B equals the *true* DGP B up to
sign+permutation on simulated non-Gaussian data (MC tol 5e-2), and the ICA
rotation provably lowers fourth-order cross-dependence relative to the raw whitened
residuals; plus $BB' = \Sigma_u$, $Q$ orthogonal, and bit-identical
reproducibility ([`nongaussian_svar.json`](../../../fixtures/nongaussian_svar.json),
[`nongaussian.rs`](../../../crates/tsecon-ident/tests/nongaussian.rs)). The novel
ICA core is pinned *exactly*; the *statistical-identification* claim rests on the
recovery property — honestly weaker than a closed-form golden. See the
[validation matrix](../validation-matrix.md).

**References.** Comon (1994); Hyvärinen & Oja (2000, FastICA); Lanne, Meitz &
Saikkonen (2017, *Journal of Econometrics*); Gouriéroux, Monfort & Renne (2017,
*Journal of Econometrics*); Montiel Olea, Plagborg-Møller & Qian (2022);
Drautzburg & Wright (2023).

```python
import numpy as np, tsecon
import itertools

def best_align(B_hat, B_true):
    # align recovered columns to the true B up to sign + permutation (n = 3)
    best, aligned = np.inf, None
    for perm in itertools.permutations(range(B_true.shape[1])):
        for signs in itertools.product([1, -1], repeat=B_true.shape[1]):
            cand = B_hat[:, perm] * np.array(signs)
            d = np.max(np.abs(cand - B_true))
            if d < best:
                best, aligned = d, cand
    return best, aligned

rng = np.random.default_rng(0)
T = 2000
# independent, standardized Student-t(5) structural shocks (excess kurtosis = 6)
eps = rng.standard_t(5, size=(T, 3)) / np.sqrt(5 / 3)
B_true = np.array([[1.0,  0.5, -0.3],       # true impact matrix, u = B eps
                   [0.4,  1.0,  0.2],
                   [-0.2, 0.3,  1.0]])
A1 = np.array([[0.5, 0.0, -0.1],
               [0.1, 0.4,  0.0],
               [0.0, 0.1,  0.5]])
y = np.zeros((T, 3))
u = eps @ B_true.T
for t in range(1, T):
    y[t] = A1 @ y[t - 1] + u[t]

ng = tsecon.nongaussian_svar(y, lags=1, horizon=8)
print("converged:", ng["converged"], " n_iter:", ng["n_iter"],
      " identified order:", np.asarray(ng["order"]))
print("shock excess kurtosis (identified order):",
      np.round(np.asarray(ng["shock_kurtosis"]), 3))
err, B_aligned = best_align(np.asarray(ng["impact"]), B_true)
print("recovered B, aligned to true B up to sign+permutation:\n", np.round(B_aligned, 4))
print("max|recovered B - true B|:", round(err, 4))

# FAILS under Gaussianity: same B, Gaussian shocks -> kurtosis ~ 0, rotation arbitrary
rng2 = np.random.default_rng(1)
yG = np.zeros((T, 3))
uG = rng2.standard_normal((T, 3)) @ B_true.T
for t in range(1, T):
    yG[t] = A1 @ yG[t - 1] + uG[t]
ngG = tsecon.nongaussian_svar(yG, lags=1, horizon=8)
errG, _ = best_align(np.asarray(ngG["impact"]), B_true)
print("\nGaussian shocks -- identification FAILS")
print("shock excess kurtosis (all near zero):",
      np.round(np.asarray(ngG["shock_kurtosis"]), 3))
print("max|recovered B - true B|:", round(errG, 4))
```

```
converged: True  n_iter: 4  identified order: [2 0 1]
shock excess kurtosis (identified order): [6.604 3.806 3.387]
recovered B, aligned to true B up to sign+permutation:
 [[ 0.9998  0.5073 -0.2329]
 [ 0.3814  0.9931  0.2283]
 [-0.2418  0.2898  0.9486]]
max|recovered B - true B|: 0.0671

Gaussian shocks -- identification FAILS
shock excess kurtosis (all near zero): [ 0.209 -0.207  0.084]
max|recovered B - true B|: 0.5695
```

With independent, heavy-tailed shocks the FastICA fixed point converges in four
steps and recovers the true impact matrix to within `0.067` — no ordering, no
sign, no instrument, no variance regime spent, only the non-Gaussianity of the
shocks. The three `shock_kurtosis` values (6.6, 3.8, 3.4) are all comfortably
positive: the leverage is real, and the columns are ordered most-non-Gaussian
first. Feed the *same* system Gaussian shocks and the story collapses exactly as
the theorem promises — the excess kurtoses fall to ≈0, there is no most-non-Gaussian
direction left to find, and the recovered B wanders `0.57` from the truth. The
`shock_kurtosis` diagnostic is what turns that failure from silent to loud: when
it is near zero, the identification is not there to be had.

---

## Post-identification and prior-robust tools

The four schemes above (and the recursive / sign / zero-sign schemes in the
[VAR/SVAR card](var-svar.md)) each hand you an identification. The five tools
below answer what comes next. Three take a single structural impact matrix `A0`
(columns = one-standard-deviation shocks, $A_0 A_0' = \Sigma_u$) from *any*
scheme; two operate on the sign-restricted set directly.

The shared object is the structural moving-average representation
$\Theta_h = \Psi_h A_0$, where $\Psi_h$ are the reduced-form MA weights
($\Psi_0 = I$, $\Psi_h = \sum_{i=1}^{\min(h,p)} \Psi_{h-i} A_i$) and the columns
of $\Theta_h$ are the horizon-$h$ impulse responses. Because $A_0 = P Q$ for a
lower-Cholesky $P$ and *any* orthogonal $Q$, every one of these tools reads the
same $(\Psi_h, P)$ off the reduced form and differs only in what it does with the
rotation $Q$ — a fixed one, a sampled set, or the whole admissible set.

The examples below share one 3-variable macro system — output, prices, policy
rate — with a genuine simultaneity between the three shocks:

```python
import numpy as np, tsecon

rng = np.random.default_rng(7)
T = 300
eps = rng.standard_normal((T, 3))          # structural: [demand, cost, policy]
B0 = np.array([[0.8,  0.4, -0.3],          # variables: output, prices, ffr
               [0.2,  0.9, -0.2],
               [0.3, -0.1,  0.7]])
A1 = np.array([[0.5,  0.0, -0.1],
               [0.1,  0.4,  0.0],
               [0.0,  0.1,  0.6]])
data = np.zeros((T, 3))
for t in range(1, T):
    data[t] = A1 @ data[t - 1] + B0 @ eps[t]
```

---

## `structural_fevd` — variance decomposition for an arbitrary impact matrix

**What it estimates.** The forecast-error variance decomposition
`fevd[h][i][j]` — the share of variable $i$'s $(h{+}1)$-step forecast-error
variance attributable to structural shock $j$ — for a **general** structural
impact matrix $A_0$. `var_fevd` computes this only for the recursive-Cholesky
$A_0 = P$; `structural_fevd` fills the gap, accepting the $A_0$ from a sign-,
zero-, proxy-, max-share-, long-run-, or heteroskedasticity-identified model.
The share is $\omega_{ij}(h) = \big[\sum_{s\le h}\Theta_s[i,j]^2\big] /
\big[\sum_m\sum_{s\le h}\Theta_s[i,m]^2\big]$ with $\Theta_s = \Psi_s A_0$.

**Assumptions.** A correct reduced form and an $A_0$ that satisfies
$A_0 A_0' = \Sigma_u$. That is the *only* requirement — the shares inherit
whatever identification produced $A_0$, and carry no more economic content than
it does.

**The invariant that makes it honest.** The denominator — variable $i$'s total
$(h{+}1)$-step forecast MSE — is **rotation-invariant**: $A_0 A_0' = PQQ'P' =
\Sigma_u$ regardless of $Q$, so the total variance being split does not depend on
the identification. Only the split across shocks $j$ changes. Consequently each
row sums to exactly 1, and column sign-flips of $A_0$ leave the shares unchanged
(they enter squared). With $A_0 = P$ the result equals `var_fevd` and
statsmodels' `VARResults.fevd` exactly.

**When to use (and when not).** Use to report "shock $j$ explains X% of variable
$i$'s variance at horizon $h$" *after* you have identified $A_0$ — the standard
companion table to an IRF plot. Do not read a Cholesky FEVD when your shock is
sign- or proxy-identified: feed the actual $A_0$. Do not over-interpret shares
from a set-identified scheme without checking they are stable across the
admissible rotations (that is what `robust_svar_bounds` is for on the IRFs).

**Key arguments and defaults (and why).** `lags`, `horizon` (the FEVD is
reported for steps $0..\,$`horizon`), `trend="c"`. `impact=None` uses the lower
Cholesky of $\Sigma_u$ (so the result reproduces `var_fevd`); pass an
$(n\times n)$ `impact` for any other scheme. `sigma="dfadj"` (default) or
`"mle"` sets the default Cholesky's df scaling — the **shares are invariant to
it** (numerator and denominator scale together); it only rescales the reported
`impact`.

**How to read the output.** `fevd` `[horizon+1][variable][shock]` (each
`fevd[h][i]` sums to 1), and `impact` `[n][n]` (the $A_0$ used — the Cholesky
factor when `impact=None`).

**Failure modes.** Passing an $A_0$ that does not satisfy $A_0 A_0' = \Sigma_u$
(the row sums stay 1 by construction, but the shares are then meaningless);
reading a recursive FEVD for a non-recursive shock; confusing the `[h][i][j]`
layout (variable then shock) with `var_fevd`'s `[i][h][j]`.

**Validated against.** statsmodels `VARResults.fevd` and the independent
`tsecon-var` `var_fevd`, an exact cross-implementation golden for the Cholesky
case (tol 1e-10); the general-$A_0$ shares are pinned by the exact algebraic
invariants — row sums = 1 and denominator rotation-invariance under a random
orthogonal $Q$ (tol 1e-12)
([`structural_fevd.json`](../../../fixtures/structural_fevd.json),
[`structural_fevd.rs`](../../../crates/tsecon-ident/tests/structural_fevd.rs), 7
tests). See the [validation matrix](../validation-matrix.md).

**References.** Lütkepohl (2005, §2.3.3); Kilian & Lütkepohl (2017, ch. 4).

```python
sf = tsecon.structural_fevd(data, lags=2, horizon=12)
fevd = np.asarray(sf["fevd"])          # [h][variable][shock]
print("row sums at h=12 (each variable's shares):", np.round(fevd[12].sum(axis=1), 12))
print("ffr (variable 2) FEVD at h = 0, 4, 12:\n", np.round(fevd[[0, 4, 12], 2, :], 4))

# impact=None reproduces var_fevd exactly (aligning the two array layouts)
vf = np.asarray(tsecon.var_fevd(data, lags=2, horizon=12))     # [variable][step][shock]
print("matches var_fevd:", np.allclose(np.transpose(fevd, (1, 0, 2))[:, :12, :], vf))

# feed a rotated A0 = P @ Q: the total MSE is invariant, only the split moves
Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
sf2 = tsecon.structural_fevd(data, lags=2, horizon=12, impact=np.asarray(sf["impact"]) @ Q)
row = np.asarray(sf2["fevd"])[12, 2, :]
print("rotated-A0 ffr FEVD at h=12:", np.round(row, 4), " sum:", round(row.sum(), 12))
```

```
row sums at h=12 (each variable's shares): [1. 1. 1.]
ffr (variable 2) FEVD at h = 0, 4, 12:
 [[4.000e-04 1.429e-01 8.567e-01]
 [1.190e-02 9.210e-02 8.960e-01]
 [1.180e-02 9.030e-02 8.979e-01]]
matches var_fevd: True
rotated-A0 ffr FEVD at h=12: [0.0377 0.299  0.6633]  sum: 1.0
```

Under the Cholesky ordering the funds rate's own shock explains 86% of its
one-step forecast error and 90% by horizon 12. Rotate the impact matrix and the
split changes completely (4% / 30% / 66%) — yet the row still sums to exactly 1,
because the *total* variance being decomposed is the reduced-form object the
rotation cannot touch. That is the whole point: the FEVD is only as identified as
the $A_0$ you feed it.

---

## `historical_decomposition` — who drove each observation

**What it estimates.** The exact split of each realized observation into a
deterministic/initial-condition **baseline** plus the cumulated contribution of
each structural shock: `hd[t][i][j]` is shock $j$'s contribution to variable $i$
at effective date $t$, with $\mathrm{hd}[t,i,j] = \sum_{s=0}^{t} \Theta_s[i,j]\,
\varepsilon_{t-s,j}$. It answers "how much did shock $j$ contribute to variable
$i$ during episode X" — the Kilian & Lütkepohl (2017, ch. 4) historical
decomposition, and the hard prerequisite for `narrative_svar`.

**The adding-up identity.** For *any* invertible $A_0$,
$$y_{t,i} = \mathrm{baseline}[t,i] + \sum_{j} \mathrm{hd}[t,i,j]$$
holds **exactly** — not asymptotically — because $y - \mathrm{baseline}$ is the
finite truncated MA sum from the initial condition, and the presample shocks are
fully absorbed into the baseline. The example below verifies it to $\sim10^{-15}$.

**Assumptions.** A correct reduced form and an $A_0$. In the default
`identification="cholesky"` mode the decomposition is *exactly identified* given
the reduced form — the only modeling choice is the ordering. In
`identification="sign"` mode the contributions become a set, summarized over the
sign- (and optionally narrative-) restricted rotations.

**When to use (and when not).** Use to attribute a specific historical episode —
"the 1979-82 funds-rate run-up was N% monetary shock" — or to plot the shock
contributions to a variable over time. Do not read the cholesky-mode
contributions as sign-identified shocks: in that mode the shocks are the
recursive ones (variable $i$'s own orthogonalized innovation is shock $i$). For a
set-identified scheme pass `identification="sign"` with `restrictions`.

**Key arguments and defaults (and why).** `restrictions` — traditional
`(variable, shock, horizon, sign)` tuples, needed only for
`identification="sign"`. `lags`, `horizon=None` (the MA is truncated at the exact
$T_{\mathrm{eff}}-1$ by default). `identification="cholesky"` (point, $Q=I$) or
`"sign"` (set). `n_draws`, `max_tries`, `seed`, `lambda1` control the sampler in
sign mode; `narrative_restrictions` and `n_weight_draws` add episode restrictions
(see `narrative_svar`).

**How to read the output.** `times` (0-based effective-sample indices, $=$
`data_row - lags`), `baseline` `[T_eff][n]`. In cholesky mode: `hd`
`[T_eff][variable][shock]` and the structural `shocks` `[T_eff][n]`. In sign
mode: `probs`, `hd_quantiles` `[T_eff][n][n][len(probs)]` (weighted type-7), the
weight-free `hd_set_min`/`hd_set_max` envelope, per-draw `weights`, and
`diagnostics`.

**Failure modes.** Reading cholesky-mode "shock 2" as an economically named
shock (it is the third variable's recursive innovation); a singular $A_0$ (the
structural shocks $\varepsilon = A_0^{-1}u$ are then undefined — reported as an
error); off-by-`lags` alignment between `times` and the original data rows.

**Validated against.** A self-contained NumPy closed-form reference that fits a
fixed VAR(2) by OLS, Cholesky-identifies, and computes $\varepsilon$, $\Theta_s$,
`hd`, and `baseline` — matched cell-by-cell (rtol 1e-8, atol 1e-10), with the
adding-up residual $\max|y - \mathrm{baseline} - \sum_j \mathrm{hd}| < 10^{-9}$
([`historical_decomposition_chol.json`](../../../fixtures/historical_decomposition_chol.json),
[`historical_decomposition.rs`](../../../crates/tsecon-ident/tests/historical_decomposition.rs)
plus the `shocks.rs`/`histdecomp.rs` unit tests).

**References.** Kilian & Lütkepohl (2017, ch. 4); Antolín-Díaz & Rubio-Ramírez
(2018, for the sign-mode set version).

```python
hd = tsecon.historical_decomposition(data, lags=2, identification="cholesky")
contrib = np.asarray(hd["hd"])         # [t][variable][shock]
base = np.asarray(hd["baseline"])      # [t][variable]
y_eff = data[2:]                       # the effective sample (lags dropped)

print("adding-up  max|y - baseline - sum_j hd|:",
      np.max(np.abs(y_eff - (base + contrib.sum(axis=2)))))

t = 150
print(f"at t={t}: ffr actual {y_eff[t, 2]:+.4f}  baseline {base[t, 2]:+.4f}")
print("  ffr contributions from shocks [0, 1, 2]:", np.round(contrib[t, 2, :], 4))
```

```
adding-up  max|y - baseline - sum_j hd|: 2.6645352591003757e-15
at t=150: ffr actual +0.1345  baseline -0.2017
  ffr contributions from shocks [0, 1, 2]: [-0.0845  0.3348  0.0859]
```

The identity holds to machine precision, and the funds rate's deviation from its
baseline at $t=150$ is decomposed into the three recursive shocks — here the
second shock (the price equation's innovation) is doing most of the work. Swap in
`identification="sign"` with the restrictions below and each `hd[t][i][j]` becomes
a band over the admissible monetary-shock rotations instead of a point.

---

## `fry_pagan_svar` — the coherent draw the median band is not

**What it estimates.** The single accepted, sign-normalized structural draw whose
IRFs are jointly closest to the pointwise median — the Fry-Pagan (2011)
median-target rotation. Sign restrictions identify a *set* of models; the
pointwise median band stitches together responses from mutually inconsistent
draws (the horizon-3 median and the horizon-8 median generally come from
different rotations), so it is **not the IRF of any admissible model**.
`fry_pagan_svar` returns one that is.

**The criterion.** Over a set of target cells $\mathcal{C}$ (by default all
response cells of the sign-restricted shocks, every variable and horizon), the
median-target statistic is $\mathrm{MT}(d) = \sum_{(i,j,h)\in\mathcal{C}}
z^{(d)}_{i,j,h}{}^2$ where $z^{(d)} = (\Theta^{(d)} - \mathrm{median})/\mathrm{sd}$
is each draw's standardized deviation from the pointwise median. The selected
draw is $d^\star = \arg\min_d \mathrm{MT}(d)$ — the interior point of the
identified set that is *internally coherent* and central.

**Assumptions.** Everything `sign_restricted_svar` assumes, plus the honest
caveat that **the selected draw is a descriptive summary, not a point estimate**:
it is one interior point of a set, and *which* point depends on the informative
Haar prior over rotations. It answers "give me one coherent model near the middle
of the band," not "give me the identified impulse response."

**When to use (and when not).** Use to report a single set of numbers — an IRF
table, an $A_0$ to feed `structural_fevd` or `historical_decomposition` — that
comes from one real model rather than a mix. Do not present it as *the* estimate,
and do not drop the band: the median-target IRF is a companion to the identified
set, not a replacement. When the prior matters, pair it with
`robust_svar_bounds`.

**Key arguments and defaults (and why).** `restrictions` (required) — the
`(variable, shock, horizon, sign)` tuples. `lags`, `horizon`, `n_draws=500`,
`max_tries=400`, `seed=0`, `lambda1=0.2` — same sampler as
`sign_restricted_svar`. `target="restricted"` scores only the response cells of
the sign-restricted shocks (default); `"all"` scores every cell.

**How to read the output.** `median_target_irf` `[horizon+1][n][n]` (the coherent
Fry-Pagan IRF — its `[0]` slice is a valid $A_0$), `median_irf` (the incoherent
pointwise median, for side-by-side), `mt_index` (0-based into the accepted set),
`mt_statistic`, `n_accepted`, and `diagnostics`
(`posterior_draws_used`/`rotations_tried`/`accepted`/`acceptance_rate`).
Reproducible bit-for-bit at a fixed `seed`.

**Failure modes.** Reporting the median-target IRF without the band (it hides the
set-identification width, which *is* the finding); reading it as prior-free (the
Haar prior selects which interior point); too few accepted draws to estimate a
stable pointwise median (watch `n_accepted`).

**Validated against.** A stored fixture of $D$ candidate structural IRFs (seeded
NumPy Haar rotations of a fixed Cholesky IRF, sign-filtered) with an independent
NumPy computation of the median, dispersion, $\mathrm{MT}(d)$, and $\arg\min$;
the Rust selection must return the same `mt_index` and `mt_statistic` (tol
1e-10), plus end-to-end seed reproducibility
([`fry_pagan_svar.json`](../../../fixtures/fry_pagan_svar.json),
[`fry_pagan.rs`](../../../crates/tsecon-ident/tests/fry_pagan.rs)). The *selection
rule* is validated exactly; the *estimand* inherits the set-identification
caveat.

**References.** Fry & Pagan (2011, *Journal of Economic Literature*).

```python
# policy shock (2): raises the funds rate, lowers output and prices on impact
restr = [(2, 2, 0, "+"), (0, 2, 0, "-"), (1, 2, 0, "-")]
fp = tsecon.fry_pagan_svar(data, restr, lags=2, horizon=12, n_draws=500, seed=0)

print("n_accepted:", fp["n_accepted"], " mt_index:", fp["mt_index"],
      " mt_statistic:", round(fp["mt_statistic"], 4))
mt = np.asarray(fp["median_target_irf"]); med = np.asarray(fp["median_irf"])
print("coherent  output<-policy  h = 0, 2, 4, 8:", np.round(mt[[0, 2, 4, 8], 0, 2], 4))
print("pointwise output<-policy  h = 0, 2, 4, 8:", np.round(med[[0, 2, 4, 8], 0, 2], 4))
```

```
n_accepted: 500  mt_index: 348  mt_statistic: 1.9921
coherent  output<-policy  h = 0, 2, 4, 8: [-0.3532 -0.0762 -0.0159 -0.0012]
pointwise output<-policy  h = 0, 2, 4, 8: [-0.4892 -0.0758 -0.0127 -0.0007]
```

Draw 348 of the 500 accepted is the single most central *coherent* model. Its
output-on-impact response ($-0.35$) differs from the pointwise median ($-0.49$)
precisely because the pointwise median is not a model — no single admissible
rotation produces the $-0.49$ impact together with the median responses at every
other horizon. Read the two together: the band for the set, the median-target for
one model that lives inside it.

---

## `robust_svar_bounds` — the identified set without the Haar artifact

**What it estimates.** The Giacomini-Kitagawa (2021) prior-robust identified-set
bounds. For each restricted shock and each response cell $(h, i, j)$, and *each*
reduced-form posterior draw, it computes the **exact min and max** of the
structural IRF over the entire admissible rotation set — not a sampled interval,
the whole set. It then summarizes those per-draw edges across the posterior. This
removes the informative-Haar-prior artifact that the pointwise
`sign_restricted_svar` bands carry: because the data cannot distinguish points
*within* the identified set, any single prior on rotations (the Haar default
included) injects information the data never provided, and that never washes out
(Baumeister-Hamilton 2015).

**The closed form.** For a shock restricted alone, each restriction is a linear
inequality $a_k' q_j \ge 0$ on that shock's rotation column, and the IRF
$\eta = g' q_j$ is optimized over $\{\|q\|=1,\ a_k'q\ge0\}$ — a quadratically
constrained linear program whose optimum is a KKT point found by active-set
enumeration (Gafarov-Meier-Montiel-Olea 2018). This is **exact for a single
restricted shock**. With several jointly-restricted shocks the admissible columns
must be mutually orthogonal, the per-column problem no longer decouples, and each
reported bound is that shock's *marginal* identified set — a **conservative outer
approximation** of the joint set, flagged honestly rather than oversold.

**Assumptions.** A correct reduced form and sign restrictions that are feasible
for at least some draws. The Minnesota-NIW posterior on the reduced form supplies
the draws; the *rotation* prior is exactly what this method refuses to commit to.

**When to use (and when not).** Use for any set-identified result headed for
publication: report the robust bounds alongside the sign-restricted band so a
reader can see how much of the band's apparent sharpness was prior rather than
data (if the robust region is much wider, the gap *is* the Haar artifact). Do not
use it as a point estimate; do not read the multi-shock bounds as certified joint
bounds — each is a per-shock *marginal* set that is a conservative **outer**
approximation of the true joint region (consistent with the "conservative outer
approximation" note above), never an inner one.

**Key arguments and defaults (and why).** `restrictions` (required). `lags`,
`horizon`, `n_draws=500`, `seed=0`, `lambda1=0.2`. `alpha=0.10` sets the robust
credible level (0.10 → a 90% robust credible region).

**How to read the output.** Per `[horizon+1][variable][shock]`:
`set_lower_mean`/`set_upper_mean` (posterior-mean identified-set edges,
$\hat{E}[l]$/$\hat{E}[u]$), `robust_ci_lower`/`robust_ci_upper` (the level-`alpha`
robust credible region — the $\alpha/2$ quantile of the lower edges and the
$1-\alpha/2$ quantile of the upper edges), and `lower_quantiles`/`upper_quantiles`
at `probs`. Unrestricted shocks are `NaN`; `restricted_shocks` lists the valid
$j$; `diagnostics` reports `empty_set_rate` (the share of draws whose restrictions
were mutually infeasible — a first-order GK diagnostic).

**Failure modes.** Treating the multi-shock bounds as exact joint bounds
(they are marginal); a high `empty_set_rate` signalling near-inconsistent
restrictions; reading the robust region as *narrower* than the sign band and
concluding the data are sharp — it is the opposite (the robust region is the
honest, wider object).

**Validated against.** An independent NumPy implementation of the
Gafarov-Meier-Montiel-Olea (2018) active-set closed form for a fixed
$(B, \Sigma)$ and single-shock restrictions (tol 1e-8), plus a brute-force
random-sphere search ($\ge10^6$ feasible unit vectors) that must bracket the
analytic optimum from the inside, and a NumPy aggregation golden for the
set-mean and robust-region quantiles
([`robust_svar_bounds.json`](../../../fixtures/robust_svar_bounds.json),
[`robust_bounds.rs`](../../../crates/tsecon-ident/src/robust_bounds.rs), 7 tests).
Strong for a single restricted shock; moderate (inside-bracket only) for the
multi-shock path.

**References.** Giacomini & Kitagawa (2021, *Econometrica*); Gafarov, Meier &
Montiel Olea (2018, *Journal of Econometrics*); Baumeister & Hamilton (2015).

```python
rb = tsecon.robust_svar_bounds(data, restr, lags=2, horizon=12, n_draws=500,
                               seed=0, alpha=0.10)
print("restricted_shocks:", rb["restricted_shocks"], " empty_set_rate:",
      rb["diagnostics"]["empty_set_rate"])
lo = np.asarray(rb["set_lower_mean"]); hi = np.asarray(rb["set_upper_mean"])
cil = np.asarray(rb["robust_ci_lower"]); cih = np.asarray(rb["robust_ci_upper"])
for h in [0, 2, 4]:
    print(f"h={h} output<-policy  set-mean [{lo[h,0,2]:+.4f}, {hi[h,0,2]:+.4f}]"
          f"  90% robust CI [{cil[h,0,2]:+.4f}, {cih[h,0,2]:+.4f}]")
print("unrestricted shock 0 is NaN:", bool(np.isnan(lo[0, 0, 0])))
```

```
restricted_shocks: [2]  empty_set_rate: 0.0
h=0 output<-policy  set-mean [-0.9062, +0.0000]  90% robust CI [-0.9716, +0.0000]
h=2 output<-policy  set-mean [-0.1559, +0.0341]  90% robust CI [-0.2329, +0.0978]
h=4 output<-policy  set-mean [-0.0376, +0.0203]  90% robust CI [-0.0750, +0.0590]
```

The impact bound's *upper* edge is exactly zero — the sign restriction
$(0,2,0,\text{"-"})$ forces output's on-impact response to the policy shock to be
$\le 0$, and the exact identified-set optimizer honors it to the last digit. Only
shock 2 is restricted, so shocks 0 and 1 return `NaN`. Away from impact the set
straddles zero (e.g. $[-0.16, +0.03]$ at $h=2$): the sign restrictions pin the
*sign on impact* but not the persistence, and the robust bounds say so without
borrowing sharpness from the rotation prior.

---

## `narrative_svar` — episode knowledge from the historical record

**What it estimates.** The Antolín-Díaz & Rubio-Ramírez (2018) narrative
sign-restricted SVAR: `sign_restricted_svar` augmented with restrictions on named
historical episodes — the sign of a structural shock in a specific quarter, or a
"most/least important contributor" statement about a shock's role in a variable's
historical decomposition over an episode. It is a strict superset of
`sign_restricted_svar` (with no narrative restrictions it reproduces it
bit-for-bit).

**How the episodes enter.** Shock-sign restrictions constrain the per-shock
orientation jointly with the traditional signs. Contribution restrictions are
checked on the historical decomposition (orientation-free, since both
$\Theta$ and $\varepsilon$ flip together). The AD&RR estimator keeps the
reduced-form marginal at the traditional posterior and imposes the narrative event
$N$ by **importance-reweighting**: each accepted draw $m$ carries weight
$w^{(m)} = 1/\hat{P}(N\mid S, \phi^{(m)})$, where $\hat{P}$ is a Monte-Carlo
estimate over `n_weight_draws` sign-passing rotations. A draw whose
narrative-admissible slice of the identified set is small is up-weighted, so all
bands and quantiles become **weighted**.

**Assumptions.** Everything `sign_restricted_svar` assumes, plus that your
episode statements are *true of the data-generating process* — a claim you defend
by reading the same historical record the restriction encodes. The honest caveat:
$1/\hat{P}$ is a biased (Jensen) estimator of $1/P(N\mid S)$, so use
`n_weight_draws` $\ge 100$ and **watch the effective sample size** — heavy-tailed
weights are the method's characteristic failure.

**When to use (and when not).** Use when you have credible episode knowledge — "the
monetary shock was contractionary in October 1979, and it was the dominant driver
of that quarter's funds-rate move" — and want to shrink a wide sign-identified set.
Do not use it to rescue restrictions the data reject (a low
`narrative_acceptance_rate` with a collapsing `ess` means the narrative is fighting
the traditional posterior); do not ignore the weights when reading the bands.

**Key arguments and defaults (and why).** `sign_restrictions` (the traditional
tuples; may be empty if narrative restrictions are given), `narrative_restrictions`
(a list of dicts, schema below), `lags`, `horizon`, `n_draws`, `max_tries`,
`seed`, `lambda1`, and `n_weight_draws=200` (the $K_w$ for $\hat{P}$). The dict
schemas use 0-based **effective-sample** indices ($=$ `data_row - lags`):

```python
{"type": "shock_sign",   "shock": int, "period": int, "sign": "+"|"-"}
{"type": "contribution", "variable": int, "shock": int, "start": int, "end": int,
                         "rule": "most"|"least", "strong": bool}
{"type": "contribution_sign", "variable": int, "shock": int,
                         "start": int, "end": int, "sign": "+"|"-"}
```

**How to read the output.** Same shape as `sign_restricted_svar` —
`quantiles` `[horizon+1][n][n][len(probs)]` (weighted type-7 at `probs =
[0.05, 0.16, 0.50, 0.84, 0.95]`), the weight-free `set_min`/`set_max` envelope —
plus `weights` (per accepted draw, mean 1) and an extended `diagnostics`:
`narrative_accepted`, `narrative_acceptance_rate`, `ess` (effective sample size),
`mean_weight`, and `min_ptilde` (the smallest $\hat{P}$ — a small value flags a
draw carrying a large weight).

**Failure modes.** A collapsing `ess` (a few draws carrying all the weight —
the bands are then unreliable); reading a redundant narrative (one already implied
by the traditional signs) as informative (its weights are ~uniform and the bands
barely move); off-by-`lags` episode indices.

**Validated against.** Reweighting-invariance (no narrative ⇒ every weight 1 and
quantiles equal `sign_restricted_svar` bit-for-bit; a redundant narrative ⇒
$\hat{P}=1$, uniform weights, bands unchanged to 1e-12) and a deterministic
weight-formula unit test against a brute-force high-$K$ Monte-Carlo $P(N\mid S)$;
the underlying HD core carries the strong closed-form golden above
([`narrative.rs`](../../../crates/tsecon-ident/src/narrative.rs) unit tests).
Set-identified and statistical — honestly weaker than the HD golden, validated by
property rather than a golden posterior.

**References.** Antolín-Díaz & Rubio-Ramírez (2018, *American Economic Review*);
the `bsvarSIGNs` R package implements the same estimator.

```python
# by construction, the largest policy innovation lands in this quarter
peak = int(np.argmax(eps[2:, 2]))      # effective-sample index = 136

# episode: the policy shock (2) was the MOST important driver of the ffr (2) over [peak-2, peak+2]
narr = [{"type": "contribution", "variable": 2, "shock": 2,
         "start": peak - 2, "end": peak + 2, "rule": "most", "strong": False}]
nv = tsecon.narrative_svar(data, restr, narr, lags=2, horizon=12,
                           n_draws=500, seed=0, n_weight_draws=200)
d = nv["diagnostics"]
print("accepted:", d["accepted"], " narrative_acceptance_rate:",
      round(d["narrative_acceptance_rate"], 3), " ess:", round(d["ess"], 1),
      " min_ptilde:", round(d["min_ptilde"], 3))

base = tsecon.sign_restricted_svar(data, restr, lags=2, horizon=12, n_draws=500, seed=0)
qb = np.asarray(base["quantiles"]); qn = np.asarray(nv["quantiles"])
for h in [0, 2, 4]:                    # output<-policy: median and 5-95 width
    mb, wb = qb[h,0,2,2], qb[h,0,2,4]-qb[h,0,2,0]
    mn, wn = qn[h,0,2,2], qn[h,0,2,4]-qn[h,0,2,0]
    print(f"h={h}: plain median {mb:+.4f} (width {wb:.4f}) | narrative {mn:+.4f} (width {wn:.4f})")

# with no narrative restrictions it IS sign_restricted_svar
none = tsecon.narrative_svar(data, restr, None, lags=2, horizon=12, n_draws=500, seed=0)
print("narrative=None reproduces sign_restricted_svar:",
      np.array_equal(np.asarray(none["quantiles"]), qb))
```

```
accepted: 163  narrative_acceptance_rate: 0.326  ess: 143.8  min_ptilde: 0.124
h=0: plain median -0.4892 (width 0.8097) | narrative -0.2131 (width 0.6648)
h=2: plain median -0.0758 (width 0.2108) | narrative -0.0423 (width 0.1599)
h=4: plain median -0.0127 (width 0.0814) | narrative -0.0089 (width 0.0742)
narrative=None reproduces sign_restricted_svar: True
```

The narrative binds: only a third of the sign-passing rotations (`rate` 0.326)
also make the policy shock the dominant driver of the funds rate in that episode,
and the smallest $\hat{P}$ (0.124) marks a draw whose slice is narrow enough to
earn an eightfold weight. The reweighting both shifts the output-on-impact median
(from $-0.49$ toward $-0.21$) and narrows the band (0.81 → 0.66) — episode
knowledge, imposed as an importance weight, is doing real work. And with no
narrative restriction the function is exactly `sign_restricted_svar`, so it is a
safe drop-in default. A shock-sign restriction that merely *agrees* with the
impact signs is nearly redundant instead — $\hat{P}\approx0.98$, weights ~uniform,
bands unchanged — which is the reweighting-invariance check the tests pin down.
