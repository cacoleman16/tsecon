# Model card — Structural identification (advanced)

`long_run_svar` · `max_share_svar` · `proxy_svar` · `hetero_svar` ·
`nongaussian_svar` · `structural_fevd` · `historical_decomposition` ·
`narrative_svar` · `fry_pagan_svar` · `robust_svar_bounds`

A structural VAR is a reduced-form VAR plus one identifying assumption that
rotates the estimated residuals into economically meaningful shocks. The
[VAR/SVAR card](var-svar.md) covers the recursive (Cholesky) and
sign-restricted schemes; this card covers two families that build on them.

**Point-identification schemes** ([below](#long_run_svar-blanchard-quah-long-run-restrictions))
spend a *different* kind of outside information — a long-run neutrality, a
variance-share objective, an external instrument, a documented variance regime,
or the non-Gaussianity of the shocks themselves (a distributional assumption
rather than an economic restriction). Each returns a **point** identification (no
bands in this build): the estimand is one impact matrix or one structural column,
and the honest uncertainty is a v2 bootstrap item flagged per method below. All
five take a plain data matrix, estimate the reduced form internally, and are
deterministic — no RNG, no rejection sampling.

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
ordering. Do not report a point IRF as if it had a band (this build is
point-only; valid bands need the Jentsch-Lunsford 2019 moving-block bootstrap,
a documented v2 item); do not proceed on a first-stage F below ~10.

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

**Failure modes.** A weak instrument reported with delta-method bands (the
cardinal sin — there are no bands here for exactly this reason); dividing by a
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
