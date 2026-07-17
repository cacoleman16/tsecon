# Model card — Specification & diagnostic tests

`heteroskedasticity_test` · `reset_test` · `chow_test` · `cusum_test`

An ordinary-least-squares fit hands you coefficients and standard errors, but
every one of those numbers is trustworthy only under a set of *maintained*
assumptions the regression itself never checks: that the error variance is
constant, that the conditional mean is genuinely linear in the regressors, and
that the coefficients hold still across the whole sample. This family does not
estimate anything new — it interrogates those three assumptions and tells you
whether the inference you just ran deserves to be believed. Two of the tests ask
about the error variance (White and Koenker-Breusch-Pagan), one asks about
functional form (Ramsey's RESET), and two ask about parameter stability — Chow
at a break date you already know, and CUSUM scanning the whole sample when you
do not.

Every test here is built on a single OLS fit of `y` on `x`, and they all share
one hard input contract: **`x` is a 2-D design of shape `(n, k)` whose columns
are the regressors, and the intercept must be an explicit column of ones.** The
crate does not silently add a constant — it raises `MissingConstant` if none is
present (the White and Breusch-Pagan auxiliary regressions form a *centered*
`R^2`, which is only the LM statistic when the auxiliary design carries an
intercept) and `SingularDesign` if the columns are collinear. So every example
below builds `x = np.column_stack([np.ones(n), x1, x2, ...])`. Read a rejection
as a warning about the maintained model, not as a repair: a heteroskedasticity
flag points you to robust standard errors, a RESET flag to a richer functional
form, a break flag to a split-sample or time-varying model. Four of the five
statistics come with a p-value you compare to a conventional level; CUSUM is the
exception — it returns a path and a pair of boundary lines, and you read
instability off whether the path crosses them.

```text
y_t = x_t' beta + u_t ,      x_t = [ 1 , x_{t,1} , ... , x_{t,k-1} ]
                                     ^ the constant column is required, never implicit
```

---

## `heteroskedasticity_test` — White and Koenker-Breusch-Pagan

**What it estimates.** Not a parameter — a verdict on whether the OLS error
variance is constant. Two auxiliary-regression Lagrange-multiplier tests,
selected by the `test` string. **White (1980)** regresses the squared OLS
residuals on the design's columns, their squares, and all pairwise
cross-products; the statistic is `n * R^2 ~ chi2(m - 1)` with `m = k(k+1)/2`
auxiliary regressors. **Breusch-Pagan (1979)** in **Koenker's (1981)**
studentized form regresses the squared residuals on the design alone;
`LM = n * R^2 ~ chi2(k - 1)`. Both return the LM statistic together with the
equivalent F-form of the same auxiliary regression.

**Assumptions.** The residuals come from a correctly-specified conditional mean:
these are variance tests, and a functional-form error will masquerade as
heteroskedasticity, so the mean must be right first. Under the null of
homoskedasticity the LM statistic is chi-square in large samples. White is an
*omnibus* test — it looks for variance depending on the regressors' levels,
squares, and interactions, so it has power against general (including nonlinear)
heteroskedasticity but spreads that power thin and spends degrees of freedom
quadratically in `k`. Koenker-Breusch-Pagan tests only for variance that is
*linear* in the design columns: more powerful when the heteroskedasticity really
is linear in the regressors, but blind to variance that depends on them only
nonlinearly.

**When to use (and when not).** Use White as your default omnibus screen and
Koenker-Breusch-Pagan when you suspect a specific regressor drives the variance
and want the focused, higher-power linear test. Do **not** read a rejection as a
reason to abandon the model — it is a reason to switch to heteroskedasticity-
robust (White / HAC) standard errors for the coefficients you care about. Do not
run either test on residuals from a misspecified mean; run `reset_test` first, or
a form failure will show up here as spurious heteroskedasticity. On a wide design
White's degrees of freedom explode (`m = k(k+1)/2`), thinning power and possibly
exhausting residual degrees of freedom — prefer Breusch-Pagan there.

**Key arguments and defaults (and why).** `test="white"` is the default omnibus
choice; pass `"breusch_pagan"` (alias `"bp"`) for the studentized linear test.
Any other string raises `ValueError` (`unknown test ...; expected "white" or
"breusch_pagan"`) rather than silently guessing. `x` is the `(n, k)` design and
**must include an explicit constant column of ones** — the centered `R^2` is only
the LM statistic when the auxiliary design has an intercept, so a design without
one raises `MissingConstant`; a collinear design raises `SingularDesign`.

**How to read the output.** A dict. `statistic` is the `n * R^2` LM value; `df`
is its chi-square degrees of freedom (`m - 1` for White, `k - 1` for
Breusch-Pagan — the two differ by construction, since White adds the squares and
cross-products); `pvalue` is the upper-tail chi-square probability. `fstat` and
`f_pvalue` are the F-form of the same auxiliary regression (all auxiliary slopes
jointly zero), matching statsmodels' `fvalue`. A small p-value rejects
homoskedasticity; report either the LM or the F form — they test the same null.

**Failure modes.** Reading a rejection as model death rather than a cue for
robust standard errors. Running the test on a misspecified mean, so a
functional-form error is misdiagnosed as heteroskedasticity. Forgetting the
constant column (`MissingConstant`) or supplying collinear regressors
(`SingularDesign`). On a very wide design White's degrees of freedom blow up —
switch to Breusch-Pagan. A degenerate response (all squared residuals identical,
zero total sum of squares) makes the centered `R^2` undefined and is raised, not
returned.

**Validated against.** The golden fixture `fixtures/tsecon-spectest.json`
(generator `fixtures/generate_tsecon-spectest_fixtures.py`) is an independent
reference: the White statistic and p-value are pinned to statsmodels'
`het_white`, and Breusch-Pagan to `het_breuschpagan(robust=True)` — the Koenker
studentized form. The reference match is verified to `~1e-8` in the crate's
`tests/golden.rs`; the Python bindings are golden-tested in
`bindings/python/tests/test_spectest_afns_dsge.py`.

**References.** White (1980, *Econometrica* 48:817-838); Breusch & Pagan (1979,
*Econometrica* 47:1287-1294); Koenker (1981, *Journal of Econometrics*
17:107-112, the studentized version).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n = 200
x1 = rng.uniform(1.0, 4.0, size=n)                 # a positive regressor
x2 = rng.normal(size=n)
X = np.column_stack([np.ones(n), x1, x2])          # explicit constant column

# Homoskedastic: constant-variance errors.
y_homo = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(size=n)
# Heteroskedastic: the error standard deviation grows with x1.
y_het = 1.0 + 0.5 * x1 - 0.3 * x2 + x1 * rng.normal(size=n)

for label, y in [("homoskedastic", y_homo), ("heteroskedastic", y_het)]:
    w = tsecon.heteroskedasticity_test(y, X, test="white")
    bp = tsecon.heteroskedasticity_test(y, X, test="breusch_pagan")
    print(f"{label:>15}: White LM={w['statistic']:6.2f} (df={w['df']}) p={w['pvalue']:.4f}  |  "
          f"BP LM={bp['statistic']:5.2f} (df={bp['df']}) p={bp['pvalue']:.4f}")
#   homoskedastic: White LM=  4.76 (df=5) p=0.4459  |  BP LM= 0.59 (df=2) p=0.7458
# heteroskedastic: White LM= 29.32 (df=5) p=0.0000  |  BP LM=28.73 (df=2) p=0.0000
```

The homoskedastic design leaves both tests well short of significance; the
heteroskedastic one, whose error scale rises with `x1`, is flagged decisively by
White *and* by Breusch-Pagan — because the variance here is (in part) linear in
the regressor, the focused linear test has the same verdict as the omnibus one.

---

## `reset_test` — Ramsey functional-form test

**What it estimates.** The Ramsey (1969) RESET test of whether the linear
conditional mean is correctly specified. It refits `y` on the augmented design
`[X, yhat^2, ..., yhat^max_power]` — the original regressors plus low-order
powers of the fitted values — and F-tests the joint significance of the added
powers. The fitted values serve as a parsimonious index standing in for omitted
nonlinearity, neglected interactions, or a wrong link; a significant added power
says some such term belongs in the model.

**Assumptions.** The null is that the linear specification is correct; the
alternative is that a low-order polynomial in the fitted values improves the fit.
A rejection is a general symptom of misspecification, not a diagnosis of its
form: RESET tells you *that* the mean is wrong, never *how*. It inherits the OLS
assumptions of the base regression, and it needs enough residual degrees of
freedom to add the powers (`n > k + (max_power - 1)`).

**When to use (and when not).** Use it as a functional-form screen before you
trust the linear model's coefficients, and before running a heteroskedasticity
test that a form failure would contaminate. Do not read a non-rejection as proof
the model is right — it is only "no evidence of misspecification of this
particular kind", and RESET has little power against misspecification orthogonal
to the fitted-value index. It is a detector, not a diagnosis, and no substitute
for plotting residuals against candidate regressors.

**Key arguments and defaults (and why).** `max_power=3` adds `yhat^2` and
`yhat^3` — two terms, the standard RESET(2,3) that catches quadratic and cubic
curvature without overfitting the augmentation. It must be at least 2 (the test
adds `yhat^2 .. yhat^max_power`, so `max_power < 2` adds nothing and raises
`InvalidPower`). `x` is the `(n, k)` design with an explicit constant column.

**How to read the output.** A dict. `fstat` is the F statistic for the joint
significance of the added powers; `df_num` is `max_power - 1` (the number of
added terms) and `df_den` is `n - k - df_num`; `pvalue` is the upper-tail F
probability. A small p-value rejects correct specification.

**Failure modes.** On a saturated or dummy-heavy design the fitted values can be
a linear combination of the columns, making `yhat^2` collinear with `X` and
raising `SingularDesign`. Reading a large p-value as "correctly specified" rather
than "no evidence of this kind of misspecification". Setting `max_power` below 2
(`InvalidPower`). As with every test here, a missing constant column raises
`MissingConstant`.

**Validated against.** The `fixtures/tsecon-spectest.json` golden pins the RESET
F statistic and p-value to statsmodels' `linear_reset(power=3, use_f=True)`,
matched to `~1e-8` in `tests/golden.rs` and golden-tested in Python in
`bindings/python/tests/test_spectest_afns_dsge.py`.

**References.** Ramsey (1969, *Journal of the Royal Statistical Society B*
31:350-371).

```python
import numpy as np, tsecon

rng = np.random.default_rng(1)
n = 200
x1 = rng.uniform(-2.0, 2.0, size=n)
X = np.column_stack([np.ones(n), x1])              # explicit constant column

# Correctly specified: y is truly linear in x1.
y_linear = 1.0 + 0.8 * x1 + 0.5 * rng.normal(size=n)
# Misspecified: the true model is quadratic, but we fit only the linear design.
y_quad = 1.0 + 0.8 * x1 + 0.6 * x1**2 + 0.5 * rng.normal(size=n)

for label, y in [("linear (correct)", y_linear), ("omitted x1^2", y_quad)]:
    r = tsecon.reset_test(y, X, max_power=3)
    print(f"{label:>16}: RESET F={r['fstat']:7.2f}  "
          f"df=({r['df_num']}, {r['df_den']})  p={r['pvalue']:.4f}")
# linear (correct): RESET F=   0.13  df=(2, 196)  p=0.8776
#     omitted x1^2: RESET F= 177.65  df=(2, 196)  p=0.0000
```

The correctly-specified fit gives RESET nothing to find (`p = 0.88`); omitting
the true `x1^2` term leaves curvature in the residuals that the fitted-value
powers pick up immediately (`F = 177.65`, `p < 0.0001`).

---

## `chow_test` — structural break at a known split

**What it estimates.** The Chow (1960) test for a structural break at a break
date you already know. It cuts the sample at the integer index `split`, fits the
regression separately on the two regimes, and F-tests the null that both regimes
share a single coefficient vector:

```text
F = [ (SSR_pooled - SSR_1 - SSR_2) / k ] / [ (SSR_1 + SSR_2) / (n - 2k) ]  ~  F(k, n - 2k)
```

The numerator is the fit the pooled model gives up by forcing one coefficient
vector on both regimes; the denominator is the pooled two-regime residual
variance.

**Assumptions.** The break date is *known and exogenous* — chosen from theory or
an event, not searched for in the data (searching for the break that maximizes
the statistic and then applying Chow's critical values badly over-rejects; that
needs a sup-Wald / Quandt-Andrews test instead). The errors are homoskedastic
*with the same variance in both regimes*: a pure variance shift can trigger the
test, so a rejection conflates coefficient and variance instability. Each
sub-sample must be estimable, `k < split < n - k`.

**When to use (and when not).** Use it when a date pins the candidate break — a
policy change, a regime switch, a crisis with a known onset — and you want to ask
whether the relationship shifted there. Do not use it when the break date is
unknown or data-mined (reach for a whole-sample scan like `cusum_test`, or a
sup-Wald test), and do not lean on it when the error variance plausibly differs
across regimes. It tests exactly one candidate date.

**Key arguments and defaults (and why).** `split` is a required 0-indexed integer
with no default — the first regime is rows `0..split`, the second `split..n`. It
must satisfy `k < split < n - k` so both sub-samples are estimable and `n - 2k`
denominator degrees of freedom remain; otherwise it raises `InvalidSplit`. `x` is
the `(n, k)` design with an explicit constant column.

**How to read the output.** A dict. `fstat` is the F statistic on `df_num = k`
and `df_den = n - 2k` degrees of freedom, with upper-tail `pvalue`; a large F /
small p rejects coefficient stability across the split. The three residual sums
of squares are returned for diagnosis — `ssr_pooled` (the one-regime fit),
`ssr1`, `ssr2` (the two sub-samples). When `ssr_pooled` is close to
`ssr1 + ssr2` the split buys almost nothing and the statistic is small; a large
gap between them is exactly what the F statistic scales.

**Failure modes.** Searching over candidate dates and then applying Chow's
distribution (severe size distortion — use a sup-test). Mistaking a variance
shift for a coefficient break. A split too near either edge (`InvalidSplit`).
Collinear regressors within a regime (`SingularDesign`), or a missing constant
column (`MissingConstant`).

**Validated against.** The `fixtures/tsecon-spectest.json` golden assembles the
Chow statistic from statsmodels OLS residual sums of squares with a
`scipy.stats.f` p-value, matched to `~1e-8` in `tests/golden.rs` and
golden-tested in Python in `bindings/python/tests/test_spectest_afns_dsge.py`.

**References.** Chow (1960, *Econometrica* 28:591-605).

```python
import numpy as np, tsecon

rng = np.random.default_rng(2)
n = 200
split = 100                                        # 0-indexed cut: regime 1 is rows 0..100
x1 = rng.normal(size=n)
X = np.column_stack([np.ones(n), x1])              # explicit constant column

# Stable: one coefficient vector governs the whole sample.
y_stable = 1.0 + 0.7 * x1 + 0.5 * rng.normal(size=n)

# Broken: intercept and slope both shift after the split.
beta_pre = np.array([1.0, 0.7])
beta_post = np.array([2.5, -0.4])
y_break = np.empty(n)
y_break[:split] = X[:split] @ beta_pre + 0.5 * rng.normal(size=split)
y_break[split:] = X[split:] @ beta_post + 0.5 * rng.normal(size=n - split)

for label, y in [("stable", y_stable), ("break at 100", y_break)]:
    r = tsecon.chow_test(y, X, split)
    print(f"{label:>13}: Chow F={r['fstat']:8.2f}  df=({r['df_num']}, {r['df_den']})  "
          f"p={r['pvalue']:.4f}  "
          f"[SSR pooled={r['ssr_pooled']:.1f}, SSR1={r['ssr1']:.1f}, SSR2={r['ssr2']:.1f}]")
#        stable: Chow F=    2.13  df=(2, 196)  p=0.1218  [SSR pooled=55.2, SSR1=30.4, SSR2=23.7]
#  break at 100: Chow F=  354.00  df=(2, 196)  p=0.0000  [SSR pooled=215.8, SSR1=26.1, SSR2=20.7]
```

For the stable series the pooled fit is barely worse than the two-regime fit
(`SSR_pooled` ≈ `SSR1 + SSR2`, `p = 0.12`). When both coefficients jump at row
100 the pooled model is forced to a much larger residual sum of squares
(`215.8` vs `26.1 + 20.7`), and the F statistic reflects the gap.

---

## `cusum_test` — recursive-residual CUSUM stability scan

**What it estimates.** The Brown-Durbin-Evans (1975) CUSUM test of parameter
stability across the whole sample. It computes *recursive residuals* — one-step-
ahead forecast errors from expanding-window OLS fits — standardizes them, and
accumulates them into a cumulative-sum path. Under the null of stable
coefficients the path fluctuates inside a pair of straight boundary lines whose
5% width uses the documented constant `a = 0.948`; a systematic drift in the
coefficients pushes the cumulative sum out through a boundary. The Python binding
returns the standardized `path`, the two boundary lines, and the residual scale.

**Assumptions.** Time-ordered data — the recursion runs over the sample in order,
so row order is meaningful. Homoskedastic errors. The test has power against
instability that shows up as a *drift* in the running mean of the recursive
residuals (a level or slope shift in the coefficients); it is weaker against a
break that leaves the cumulative sum near zero (a symmetric wobble) and against a
variance-only break — that is the province of the CUSUM-of-squares test, which
this function does not compute. It needs enough initial observations to seed the
first expanding-window fit.

**When to use (and when not).** Use it when you *do not* know the break date and
want a whole-sample stability scan — the natural complement to `chow_test`'s
known-split F test. Read it as a graph: plot `path` against `bound_upper` and
`bound_lower`; the point where the path leaves the band dates the instability
roughly. Do not use it to pinpoint a break date precisely (it detects, it does
not time), and do not expect it to catch a variance-only break.

**Key arguments and defaults (and why).** Just `y` and the `(n, k)` design `x`
with its explicit constant column — there are no tuning knobs on the Python
surface. The 5% boundary constant `a = 0.948` is Brown-Durbin-Evans' documented
value, baked into the returned bounds rather than exposed as an argument. Note
that the raw `recursive_residuals` array and the `a` constant exist in the
underlying Rust struct but are deliberately **not** exposed in Python: the four
returned keys are `path`, `bound_upper`, `bound_lower`, and `sigma`, and nothing
else.

**How to read the output.** A dict of three numpy arrays and one float. `path` is
the standardized cumulative sum of recursive residuals; `bound_upper` and
`bound_lower` are the diverging 5% boundary lines; `sigma` is the residual
standard-error scale. There is no scalar p-value — the test rejects stability at
5% precisely when the path crosses a bound, which you read visually or compute as
`np.any(path > bound_upper) or np.any(path < bound_lower)`.

**Failure modes.** Treating it as a break-dating tool with a p-value — it returns
a path, not a scalar test. Expecting it to flag a variance-only break (that is
CUSUM-of-squares, not this function). Referencing Python keys that do not exist:
only `path`, `bound_upper`, `bound_lower`, and `sigma` are returned — there is no
`recursive_residuals` or `a` on the Python surface. A design that is collinear or
rank-deficient in an early expanding window raises `SingularDesign`; a missing
constant column raises `MissingConstant`.

**Validated against.** The `fixtures/tsecon-spectest.json` golden stores the
CUSUM `path`, bounds, and `sigma` as a documented-formula numpy reference — with
recursive residuals computed by refitting each expanding window, a *different
code path* from the crate's incremental recursion — matched to `~1e-8` in
`tests/golden.rs` and golden-tested in Python in
`bindings/python/tests/test_spectest_afns_dsge.py`.

**References.** Brown, Durbin & Evans (1975, *Journal of the Royal Statistical
Society B* 37:149-192).

```python
import numpy as np, tsecon

rng = np.random.default_rng(3)
n = 200
t = np.arange(n)
x1 = rng.normal(size=n)
X = np.column_stack([np.ones(n), x1])              # explicit constant column

# Stable: one coefficient vector governs the whole time-ordered sample.
y_stable = 1.0 + 0.7 * x1 + 0.5 * rng.normal(size=n)

# Unstable: the intercept jumps upward from the midpoint onward.
y_drift = 1.0 + 0.7 * x1 + 0.5 * rng.normal(size=n) + 3.0 * (t > 100)

for label, y in [("stable", y_stable), ("mid-sample shift", y_drift)]:
    r = tsecon.cusum_test(y, X)
    path, lo, hi = r["path"], r["bound_lower"], r["bound_upper"]
    breached = bool(np.any(path > hi) or np.any(path < lo))
    print(f"{label:>17}: sigma={r['sigma']:.3f}  "
          f"max|path|={np.max(np.abs(path)):6.2f}  bound@end={hi[-1]:.2f}  "
          f"crosses 5% bound? {breached}")
#            stable: sigma=0.495  max|path|= 11.77  bound@end=40.02  crosses 5% bound? False
#  mid-sample shift: sigma=1.575  max|path|=128.11  bound@end=40.02  crosses 5% bound? True
```

The stable series keeps its cumulative sum comfortably inside the band
(`max|path| = 11.77` against a terminal bound near `40`). The mid-sample
intercept jump drives the recursive residuals in one direction, the cumulative
sum runs far past the boundary (`max|path| = 128.11`), and the boolean crossing
check reports instability — with no break date supplied.
