# Model card — Specification & diagnostic tests

`heteroskedasticity_test` · `reset_test` · `chow_test` · `cusum_test` ·
`breusch_godfrey_test` · `durbin_watson_test`

An ordinary-least-squares fit hands you coefficients and standard errors, but
every one of those numbers is trustworthy only under a set of *maintained*
assumptions the regression itself never checks: that the error variance is
constant, that the errors are serially uncorrelated, that the conditional mean
is genuinely linear in the regressors, and that the coefficients hold still
across the whole sample. This family does not estimate anything new — it
interrogates those four assumptions and tells you whether the inference you
just ran deserves to be believed. Two of the tests ask about the error variance
(White and Koenker-Breusch-Pagan), two about serial correlation in the errors
(Breusch-Godfrey up to any order, Durbin-Watson at first order with an exact
p-value), one about functional form (Ramsey's RESET), and two about parameter
stability — Chow at a break date you already know, and CUSUM scanning the
whole sample when you do not.

Every test here is built on a single OLS fit of `y` on `x`, and they all share
one hard input contract: **`x` is a 2-D design of shape `(n, k)` whose columns
are the regressors, and the intercept must be an explicit column of ones.** The
crate does not silently add a constant to *your* regression. The White and
Breusch-Pagan tests raise `MissingConstant` if none is present (their auxiliary
regressions form a *centered* `R^2`, which is only the LM statistic when the
auxiliary design carries an intercept); the serial-correlation tests accept a
regression through the origin and say what they do with it (Breusch-Godfrey
adds the intercept to its *auxiliary* regression only, the statsmodels
convention). A collinear design raises `SingularDesign` everywhere. So every
example below builds `x = np.column_stack([np.ones(n), x1, x2, ...])`. Read a
rejection as a warning about the maintained model, not as a repair: a
heteroskedasticity flag points you to robust standard errors, a
serial-correlation flag to HAC standard errors or a dynamic specification, a
RESET flag to a richer functional form, a break flag to a split-sample or
time-varying model. Every statistic but one comes with a p-value you compare
to a conventional level; CUSUM is the exception — it returns a path and a pair
of boundary lines, and you read instability off whether the path crosses them.

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

---

## Serial correlation — why two functions

Serial correlation in the errors is the common assumption failure of
time-series regressions, and a costly one: under strict exogeneity the
coefficients stay unbiased, but the conventional standard errors are wrong,
often badly. Two classical tests interrogate it,
and the library ships both as **separate callables**, `breusch_godfrey_test`
and `durbin_watson_test`, rather than one `serial_correlation_test(test=...)`
switch in the style of `heteroskedasticity_test`. The reason is the family's
own rule: `heteroskedasticity_test` merges White and Breusch-Pagan because
they take the same arguments and return the same record (an LM statistic, its
F-form, nothing else). Breusch-Godfrey and Durbin-Watson share neither. The
first takes an order `nlags` and returns an LM and an F test with their
degrees of freedom; the second has no order, takes the direction of the
alternative, returns one statistic with an exact p-value, and is subject to a
sample-size cap. One switch would make `nlags` inert under
`test="durbin_watson"` and `alternative` inert under `test="breusch_godfrey"`,
and the house rule is to refuse inert arguments, not ignore them. Two functions
read like their neighbours `reset_test`, `chow_test` and `cusum_test`: one
test, one name, one signature. Both take the family's `(y, x)` pair, with `x`
the `T x k` design including the constant, and both work on the OLS residuals
of `y` on `x` — unlike `ljung_box` and `arch_lm`, which take a residual series
you computed yourself and therefore cannot account for the regressors.

---

## `breusch_godfrey_test` — LM test for serial correlation up to order p

**What it estimates.** Not a parameter — a verdict on whether the errors of
`y = X b + u` are serially correlated at any order up to `p = nlags`. The
Breusch (1978) / Godfrey (1978) Lagrange-multiplier test regresses the OLS
residuals on the design and on `p` of their own lags,

```text
u_t = x_t' g + a_1 u_{t-1} + ... + a_p u_{t-p} + v_t ,     H0: a_1 = ... = a_p = 0
LM  = T * R^2_aux                                          ~ chi2(p)
F   = [(SSR_r - SSR_aux) / p] / [SSR_aux / (T - k_aux)]    ~ F(p, T - k_aux)
```

and returns both forms of the same test: the LM statistic with its chi-square
p-value and the F-form (the lag coefficients jointly zero in the auxiliary
regression) with its F p-value.

**Assumptions.** Under the null, the errors are serially uncorrelated and the
regression is correctly specified; the chi-square and F distributions are
large-sample approximations (the F-form is the usual small-sample choice). The
regressors need not be strictly exogenous: **a lagged dependent variable is
allowed**, because the auxiliary regression includes `x` itself. That is the
test's main advantage over Durbin-Watson. It is a test against AR(p) *and*
MA(p) errors alike (the two give the same LM test), so a rejection says the
errors are autocorrelated without saying which process generated them.
Both forms assume homoskedastic errors. Under strong conditional
heteroskedasticity their size can drift, and a heteroskedasticity-robust
variant of the test (not provided here) would be needed.

**When to use (and when not).** Use it as the default serial-correlation check
after any time-series regression, and always when the regressors include a
lagged `y` or anything else that is not strictly exogenous. Choose `nlags` to
match the data frequency (4 or 8 for quarterly, 12 for monthly). A rejection
is a cue to switch to HAC standard errors (`ols(..., se_type="hac")`) or to
model the dynamics (add lags, or fit ARMA errors), not to discard the
regression. Do not run it on residuals you computed elsewhere: it needs the
design to form the auxiliary regression. For a series with no regressors,
`ljung_box` is the natural test, and it is what `check_series` runs.

**Key arguments and defaults (and why).** `nlags=None` uses
`min(10, T // 5)`, the statsmodels default, so the numbers match
`acorr_breusch_godfrey` with no argument. A data-dependent default needs
care, and the measurements below show its cost: at `T = 50` it means
`p = 10`, which makes the test conservative (2.7% size at a nominal 5%) and
spends power on lags that are not there. Pass the order you mean. The lagged
residuals are **zero-padded**: `u_{t-j}` is taken as 0 for `t - j < 1`, so all
`T` observations stay in the auxiliary regression. This is the statsmodels
convention, and the result echoes `nobs = T`. The other textbook convention
drops the first `p` observations instead. It is asymptotically equivalent but
gives different numbers in a finite sample, so the two must not be compared.
If `x` spans no constant (no constant column and no implicit one, such as a
set of complementary dummies), a column of ones is added to the **auxiliary**
regression only, so the centered `R^2` is defined. The result says so
(`constant_added`), and then `k_aux = k + nlags + 1` instead of `k + nlags`.
`nlags` must be at least 1 and leave `T > k_aux`. A sample shorter than five
observations has a default order of 0, and the refusal says to pass `nlags`
or a longer `y`.

**How to read the output.** A dict. `statistic` is `LM = T * R^2_aux`, `df`
its chi-square degrees of freedom (`= nlags`), and `pvalue` the upper-tail
probability. `fstat` is the F-form on `f_df_num = nlags` and
`f_df_den = T - k_aux` degrees of freedom, with `f_pvalue`. A small p-value
rejects "no serial correlation up to order `nlags`". The two forms agree
closely beyond a few hundred observations. At `p = 1` and `p = 4` the F-form
rejects 0.1 to 0.6 points less often than the LM form in the table below, so
it is the more conservative of the two in small samples.
`nlags` echoes the order used, with the default resolved; `nobs` is `T`; and
`constant_added` is `True` when the auxiliary regression gained an intercept.

**Failure modes.** An order much longer than the true dependence costs power,
and the `min(10, T // 5)` default is too long at moderate `T`. A missing
dynamic term shows up here as serial correlation: it is a specification
signal, not only an inference one. Strong heteroskedasticity distorts both
forms. A design that is collinear (`x` rank deficient), an order that leaves
no residual degrees of freedom (`nlags` too long), or a `y` that `x` fits
exactly (zero residuals) are refused with the parameter named.

**Validated against.** statsmodels `acorr_breusch_godfrey` on an OLS fit:
`fixtures/serial_corr.json` (generator
`fixtures/generate_serial_corr_fixtures.py`) holds 34 cells (LM statistic,
p-value, F, F p-value, the degrees of freedom and `constant_added`) over 12
cases. The cases are seeded regressions with AR(1) errors (`rho` = 0.5, 0.8,
0.4, 0.3, 0.2, -0.6) and iid errors, constant-only designs and designs with
up to three regressors, a design without a constant, an implicit-constant
design, `T = 15` and `T = 500`, and three US macro regressions built from
statsmodels' macrodata by transformation (consumption growth on income growth,
inflation on unemployment, investment growth on GDP growth and the change in
the real rate). Every `nlags` is pinned, including the default. The match
holds at `1e-10` in `crates/tsecon-spectest/tests/serial_golden.rs` (measured
worst deviation `2.4e-13`) and again through Python in
`bindings/python/tests/test_serial_corr.py`, which also runs statsmodels live
on six fresh seeded regressions.

Size and power by seeded Monte Carlo
(`lab/serial_corr/size_power.py`,
output in `lab/serial_corr/out/size_power.txt`). The DGP is
`y = 1 + 0.5 x1 - 0.5 x2 + u`, with `x1` AR(1)(0.5) and `x2` iid, both fixed
across 2000 replications per cell. The table gives rejection rates at 5%; the
Monte Carlo standard error is 0.5 points.

| T | test | iid N(0,1) | iid t(5) | AR(1) ρ = 0.2 | AR(1) ρ = 0.4 |
|---|---|---|---|---|---|
| 50 | LM, p = 1 | 0.043 | 0.053 | 0.192 | 0.656 |
| 50 | F, p = 1 | 0.037 | 0.048 | 0.178 | 0.635 |
| 50 | LM, p = 4 | 0.048 | 0.045 | 0.096 | 0.418 |
| 50 | F, p = 4 | 0.044 | 0.040 | 0.088 | 0.407 |
| 50 | LM, default p = 10 | 0.027 | 0.021 | 0.030 | 0.202 |
| 100 | LM, p = 1 | 0.054 | 0.051 | 0.422 | 0.960 |
| 100 | F, p = 1 | 0.050 | 0.049 | 0.411 | 0.959 |
| 100 | LM, p = 4 | 0.056 | 0.054 | 0.245 | 0.859 |
| 100 | F, p = 4 | 0.053 | 0.050 | 0.237 | 0.854 |
| 100 | LM, default p = 10 | 0.038 | 0.029 | 0.132 | 0.683 |
| 200 | LM, p = 1 | 0.052 | 0.043 | 0.777 | 1.000 |
| 200 | F, p = 1 | 0.050 | 0.040 | 0.773 | 1.000 |
| 200 | LM, p = 4 | 0.054 | 0.048 | 0.569 | 0.997 |
| 200 | F, p = 4 | 0.053 | 0.048 | 0.568 | 0.997 |
| 200 | LM, default p = 10 | 0.051 | 0.049 | 0.368 | 0.985 |

At `p = 1` and `p = 4` both forms stay within 1.3 points of 5%, under normal
and t(5) errors alike (the most conservative cell is the F-form at `T = 50`,
0.037). The default `p = 10` is conservative below `T` of about 200 and gives up most of
its power against an AR(1) at `T = 50`. With a lagged dependent variable in
the regression (`y_t = 0.5 y_{t-1} + ...`), BG keeps its size: 0.051, 0.052
and 0.054 for the LM form at `p = 1` and `T` = 50, 100, 200, where
Durbin-Watson falls to about 0.02 (below).

**References.** Breusch (1978, *Australian Economic Papers* 17:334-355);
Godfrey (1978, *Econometrica* 46:1293-1301); Godfrey (1988, *Misspecification
Tests in Econometrics*, Cambridge University Press).

```python
import numpy as np, tsecon

rng = np.random.default_rng(4)
n = 200
x1 = rng.normal(size=n)
X = np.column_stack([np.ones(n), x1])              # explicit constant column

u_iid = rng.normal(size=n)                          # no serial correlation
u_ar = np.empty(n)                                  # AR(2) errors: 0.3, 0.3
u_ar[:2] = rng.normal(size=2)
for t in range(2, n):
    u_ar[t] = 0.3 * u_ar[t - 1] + 0.3 * u_ar[t - 2] + rng.normal()

for label, u in [("iid errors", u_iid), ("AR(2) errors", u_ar)]:
    y = 1.0 + 0.8 * x1 + u
    bg = tsecon.breusch_godfrey_test(y, X, nlags=4)
    print(f"{label:>12}: LM={bg['statistic']:6.2f} (df={bg['df']}) p={bg['pvalue']:.4f}  |  "
          f"F={bg['fstat']:5.2f} df=({bg['f_df_num']}, {bg['f_df_den']}) p={bg['f_pvalue']:.4f}")
#   iid errors: LM=  6.85 (df=4) p=0.1442  |  F= 1.72 df=(4, 194) p=0.1473
# AR(2) errors: LM= 55.86 (df=4) p=0.0000  |  F=18.80 df=(4, 194) p=0.0000
```

With white-noise errors neither form finds anything (`p = 0.14`). The AR(2)
errors are flagged decisively by both.
`f_df_den = 200 - 2 - 4 = 194`: the zero-padded auxiliary regression keeps all
200 observations.

---

## `durbin_watson_test` — first-order serial correlation, exact p-value

**What it estimates.** The Durbin-Watson (1950, 1951) statistic on the OLS
residuals `e`,

```text
d = sum_{t=2}^T (e_t - e_{t-1})^2 / sum_{t=1}^T e_t^2   ≈   2 (1 - r_1)
```

where `r_1` is the first-order residual autocorrelation. `d` is near 2 under
no autocorrelation, below 2 under positive and above 2 under negative
autocorrelation. Together with `d`, the function returns its **exact p-value**
under normal errors, not the inconclusive-region bounds tables. With
`M = I - X (X'X)^{-1} X'` and `A` the first-difference matrix `D'D`,
`d = u'MAMu / u'Mu`, so

```text
P(d <= c) = P( sum_i (nu_i - c) z_i^2 <= 0 ),    z_i iid N(0, 1),
```

where `nu_1..nu_{T-k}` are the eigenvalues of `M A M` on the complement of
`col(X)`. The library computes them by one dense symmetric
eigendecomposition and evaluates the probability by Imhof's (1961) numerical
inversion of the characteristic function, with adaptive Gauss-Kronrod
quadrature and a truncation point from Imhof's tail bound, to about `1e-13`
absolute.

**Assumptions.** The p-value is exact when the errors are Gaussian and iid
under the null **and the regressors are fixed** (strictly exogenous): the
distribution of `d` depends on `X`, which is why the classical tables could
only give bounds `d_L`, `d_U`. With non-normal errors it is an approximation;
the measured size is below. The test looks at first-order correlation only,
so a pure second-order (or seasonal) process can pass unnoticed. It is
**invalid with a lagged dependent variable** in `x`, because the regressors
are then not fixed, and `d` is biased toward 2.

**When to use (and when not).** Use it for a static regression with
exogenous regressors when you want the classical first-order test with an
exact verdict instead of an inconclusive region. In that setting it is the
more powerful first-order test: on the Monte Carlo below it matches or beats
BG at `p = 1` at every sample size and error correlation, by a wide margin at
`T = 50`. Do **not** use it with a lagged dependent variable, for
higher-order or seasonal dependence, or on a long sample: use
`breusch_godfrey_test` there. Dynamic models are
exactly where Durbin's `h` was once used, and BG has replaced it.

**Key arguments and defaults (and why).** `alternative="positive"` is the
classical one-sided test against `rho > 0`, `p = P(d <= d_obs)`, because
positive autocorrelation is the usual failure in economic time series.
`"negative"` gives `P(d >= d_obs)` and `"two_sided"` gives
`min(1, 2 min(P(d <= d_obs), P(d >= d_obs)))`. Any other string is refused,
naming `alternative`. `x` is the `(n, k)` design. A constant is not required,
because the exact distribution uses `M` whatever `X` is; the bounds tables did
need one, which is one more reason the tables are not used here. The sample
must satisfy `T >= k + 2` (with `T = k + 1`, `d` is a constant).

**COST and the sample cap.** The eigendecomposition is cubic in `T` and holds a
`T x T` matrix, so `T` is capped at 4000 (128 MB). A longer `y` is refused
with a pointer to `breusch_godfrey_test`, whose cost is linear in `T` and whose
chi-square approximation is accurate at that length. Measured on the release
wheel (median of three calls, a 4-core machine shared with another job,
`lab/serial_corr/dw_cost.py`):

| T | 100 | 250 | 500 | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|---|---|---|
| `durbin_watson_test` | 0.002 s | 0.008 s | 0.023 s | 0.136 s | 1.35 s | 6.04 s | 14.7 s |
| `breusch_godfrey_test` | 0.1 ms | 0.1 ms | 0.1 ms | 0.3 ms | 0.5 ms | 0.7 ms | 0.9 ms |

**How to read the output.** A dict. `statistic` is `d`. `pvalue` is the exact
probability for the chosen `alternative`, which is echoed. `nobs` is `T` and
`df_resid` is `T - k`, the number of `nu_i`. The p-value is computed to about
`1e-13` absolute, so a printed `1e-14` or `0.0` means "below `1e-13`". The
one-sided p-values are complementary, so a second call to read the other tail
is unnecessary: `P(d >= d_obs) = 1 - P(d <= d_obs)`.

**Failure modes.** Using it with a lagged dependent variable: under the null
the test then rejects only about 2% of the time at a nominal 5% (measured
below), and its p-value is no longer exact. Reading a non-rejection as "no
serial correlation" when the dependence is not first order. A long sample is
refused by the cap rather than taking minutes, and a collinear `x`, a `y` that
`x` fits exactly, or a sample with `T < k + 2` are refused with the parameter
named.

**Validated against.** The statistic is checked against statsmodels
`durbin_watson`, and the exact p-value against a documented-formula golden on
the same 12 cases as Breusch-Godfrey (`fixtures/serial_corr.json`). The
generator transcribes Imhof's formula itself, independently of the crate. Its
eigenvalues come from NumPy's complete QR and LAPACK `eigvalsh` on the
complement block, where the crate uses a shifted `faer` eigendecomposition,
and its integral from QUADPACK by two routes that agree to `1e-12`, where the
crate uses its own Gauss-Kronrod rule. All three alternatives match at
`1e-10`; the measured worst deviations are `7.9e-16` on the statistic and
`3.1e-15` on the p-values. No package in the fixture environment computes the
exact Durbin-Watson p-value, so the *formula* is graded separately:

* **The null CDF against simulation**
  (`lab/serial_corr/dw_exact_mc.py`).
  On three fixed designs (`T` = 20, 40, 80; constant-only to four
  regressors), the exact `P(d <= c)` was compared with 200 000 simulated null
  draws at the 1, 5, 10, 50, 90, 95 and 99% points. The largest deviation is
  1.6 Monte Carlo standard errors.
* **Closed forms and theorems** (unit tests in
  `crates/tsecon-spectest/src/durbin_watson.rs` and `imhof.rs`). With a
  constant-only design, the `nu_i` equal the closed-form eigenvalues
  `2 - 2 cos(pi j / T)` to `1e-12`. They obey the Durbin-Watson bounds
  theorem `lambda_{i+1} <= nu_i <= lambda_{i+k}`, and the exact CDF lies
  between the `d_L` and `d_U` distributions. The `T = 3` mean model matches
  its closed form `(2/pi) arctan(sqrt((d-1)/(3-d)))` to `1e-12`, and the
  quadrature reproduces the closed forms of two-term and chi-square-difference
  quadratic forms.
* **Uniformity under the null**
  (`crates/tsecon-spectest/tests/serial_properties.rs`). Over 3000 seeded
  replications the empirical CDF of the p-values lies within 4 Monte Carlo
  standard errors of the uniform at eight points.

Size and power (same DGP and output file as the Breusch-Godfrey table, test
`alternative="positive"`, rejection rates at 5%):

| T | iid N(0,1) | iid t(5) | AR(1) ρ = 0.2 | AR(1) ρ = 0.4 | lagged `y`, iid errors | lagged `y`, ρ = 0.4 |
|---|---|---|---|---|---|---|
| 50 | 0.041 | 0.043 | 0.368 | 0.806 | 0.021 | 0.478 |
| 100 | 0.059 | 0.051 | 0.610 | 0.987 | 0.022 | 0.739 |
| 200 | 0.046 | 0.048 | 0.881 | 1.000 | 0.017 | 0.958 |

Under normal errors the size is within two Monte Carlo standard errors of 5%,
as it should be for an exact test. Under non-normal errors it stays close: in
a separate run of 4000 replications per cell (`dw_exact_mc.py`, Monte Carlo
standard error 0.0034), t(5) errors gave 0.0488 at `T = 30` and 0.0493 at
`T = 100`, t(3) gave 0.0423 and 0.0490, and centered chi2(1) gave 0.0515 and
0.0585 — the last 2.5 standard errors above 5%. In the static regression its
power at `rho = 0.2` is about twice that of BG at `p = 1` when `T = 50`
(0.368 against 0.192). With a lagged dependent variable its size collapses to
about 2%. In that DGP it still detected
`rho = 0.4` about as often as BG did (0.478 against 0.387 at `T = 50`), but
its p-value no longer means what it says.

**References.** Durbin & Watson (1950, *Biometrika* 37:409-428; 1951,
*Biometrika* 38:159-178); Imhof (1961, *Biometrika* 48:419-426); Savin &
White (1977, *Econometrica* 45:1989-1996, the extended bounds tables this
replaces).

```python
import numpy as np, tsecon

rng = np.random.default_rng(5)
n = 60
x1 = rng.normal(size=n)
trend = np.arange(n) / n
X = np.column_stack([np.ones(n), x1, trend])       # explicit constant column

e = rng.normal(size=n)
u = np.empty(n)
u[0] = e[0] / np.sqrt(1 - 0.4**2)
for t in range(1, n):
    u[t] = 0.4 * u[t - 1] + e[t]                    # AR(1) errors, rho = 0.4

for label, err in [("iid errors", e), ("AR(1) errors", u)]:
    y = 1.0 + 0.5 * x1 - 1.0 * trend + err
    dw = tsecon.durbin_watson_test(y, X)            # H1: rho > 0
    two = tsecon.durbin_watson_test(y, X, alternative="two_sided")
    print(f"{label:>12}: d={dw['statistic']:.3f}  exact p(rho>0)={dw['pvalue']:.4f}  "
          f"two-sided p={two['pvalue']:.4f}  (T={dw['nobs']}, T-k={dw['df_resid']})")
#   iid errors: d=2.051  exact p(rho>0)=0.5237  two-sided p=0.9527  (T=60, T-k=57)
# AR(1) errors: d=1.300  exact p(rho>0)=0.0015  two-sided p=0.0030  (T=60, T-k=57)
```

With white-noise errors `d` sits at 2.05 and the exact p-value is 0.52. With
AR(1) errors `d = 1.30`. For `T = 60` and two regressors besides the constant
the 5% bounds are `d_L = 1.514` and `d_U = 1.652` (the 5% quantiles of the
`d_L` and `d_U` distributions, which is what the Savin-White tables
tabulate, computed from their definition in
`lab/serial_corr/dw_bounds.py`), so
the bounds test would also reject here. The exact p-value says by how much
(`p = 0.0015`), and it still gives a verdict when `d` falls between `d_L` and
`d_U`, where the tables cannot.
