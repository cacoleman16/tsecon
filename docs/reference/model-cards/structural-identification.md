# Model card — Structural identification (advanced)

`long_run_svar` · `max_share_svar` · `proxy_svar` · `hetero_svar`

A structural VAR is a reduced-form VAR plus one identifying assumption that
rotates the estimated residuals into economically meaningful shocks. The
[VAR/SVAR card](var-svar.md) covers the recursive (Cholesky) and
sign-restricted schemes; this card covers four schemes that spend a *different*
kind of outside information — a long-run neutrality, a variance-share objective,
an external instrument, or a documented variance regime. Each returns a **point**
identification (no bands in this build): the estimand is one impact matrix or one
structural column, and the honest uncertainty is a v2 bootstrap item flagged per
method below. All four take a plain data matrix, estimate the reduced form
internally, and are deterministic — no RNG, no rejection sampling.

Which one you reach for is a question about *what you can defend*, laid out in
[chapter 8](../../guide/08-causal-identification.md) and the
[decision guide](../../which-model-when.md#2-i-want-an-impulse-response). The
one-line map: **long-run** when theory speaks about permanent vs. transitory
effects; **max-share** when you want the single shock that drives a target's
business-cycle variance; **proxy** when you have a measured instrument for one
shock; **heteroskedasticity** when you have documented variance regimes.

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
