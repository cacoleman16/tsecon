# Model card — Robust inference: HC, HAC and EWC

`ols` · `ewc_lrv` · `ewc_default_b` · `long_run_variance`

A regression coefficient is only as trustworthy as the standard error printed
next to it, and the textbook standard error assumes errors that are
homoskedastic and serially uncorrelated. Time-series data are neither. This
family is the library's one engine for the denominator — the `tsecon-hac`
crate, which every regression-based estimator draws on so that identical
settings never produce different p-values in different modules — exposed
directly: OLS with a ladder of covariance estimators, the kernel long-run
variance, and the equal-weighted cosine (EWC) estimator that Lazarus, Lewis,
Stock and Watson (2018) recommend for heteroskedasticity-and-autocorrelation
robust (HAR) inference.

The three things to keep apart:

- **HC (`hc0`…`hc3`)** repairs heteroskedasticity only. Under serial
  correlation it repairs nothing.
- **HAC (`hac`, Newey-West)** repairs both, asymptotically, and is compared to
  the normal. In samples of a few hundred with persistent errors it
  under-covers, because the estimated long-run variance is itself noisy and
  the normal critical value ignores that noise.
- **EWC (`ewc`)** prices that noise in: its long-run variance behaves like
  `Omega chi2_B / B`, so its t-statistic is Student t with `B` degrees of
  freedom, and the interval is built with the `t_B` critical value.

---

## `ols` — OLS with a standard-error ladder

**What it estimates.** The least-squares coefficients `params` of `y` on the
columns of `x` (used as given — add your own constant column), with standard
errors `bse` and t-statistics `tvalues` from one of seven covariance
estimators chosen by `se_type`:

| `se_type` | Covariance | Robust to | Reference distribution |
|---|---|---|---|
| `"nonrobust"` | `s^2 (X'X)^-1`, `s^2 = RSS/(n-k)` | nothing | Student t with `n-k` df (exact under Gaussian errors; normal asymptotically) |
| `"hc0"` | White (1980) sandwich | heteroskedasticity | normal |
| `"hc1"` | HC0 × `n/(n-k)` | heteroskedasticity | normal |
| `"hc2"` | `u_t^2 / (1-h_t)` weights | heteroskedasticity, leverage | normal |
| `"hc3"` | `u_t^2 / (1-h_t)^2` weights | heteroskedasticity, leverage | normal |
| `"hac"` (**default**) | Newey-West Bartlett kernel, `maxlags` | heteroskedasticity, autocorrelation | normal |
| `"ewc"` | `(X'X)^-1 [n Omega_hat(x u)] (X'X)^-1`, `Omega_hat` the EWC long-run covariance of the scores over `ewc_b` cosine frequencies | heteroskedasticity, autocorrelation | **Student t with `ewc_b` df** |

**Assumptions.** The regression is correctly specified and its coefficients
are stable — a robust standard error is a statement about sampling
variability *around a target*, and no denominator fixes a wrong target (test
the target first: `reset_test`, `heteroskedasticity_test`, `chow_test`,
`cusum_test`). For `hac`/`ewc`: the scores `x_t u_t` are stationary and
weakly dependent. EWC's `t_B` result is a fixed-smoothing asymptotic: it holds
with `B` held fixed as `n` grows, which is what makes it honest at the `B` you
actually use.

**When to use (and when not).** Independent observations, unequal variances:
`hc1` once `n` is in the hundreds, `hc3` when `n` is small or a few
observations dominate the design. Time-series regressions: `hac` for
statsmodels-compatible numbers, **`ewc` when the size of the test matters** —
persistent regressors and errors, samples of a few hundred. Do not use any of
them to rescue a misspecified mean; do not use HC under serial correlation.
EWC is not a near-unit-root method: as the regressor and error persistence
approach one, every HAR procedure, EWC included, loses size control (Müller
2014).

**Key arguments and defaults (and why).**

- `se_type="hac"` is the default **for statsmodels compatibility**, even though
  the library's HAR policy (ROADMAP §5, following LLSW 2018) names EWC as the
  recommended inference. Changing the default would silently move every
  existing `ols` p-value, so EWC is opt-in. In this release **only `ols`
  offers EWC**: `lp`, `umidas`, `har_rv` and the other regression families
  keep their own `se=`/`se_type=` vocabularies without it.
- `maxlags=None` (`hac` only) applies the Newey-West rule of thumb
  `floor(4 (n/100)^(2/9))`.
- `use_correction=True` (`hac` only) applies the finite-sample `n/(n-k)`
  factor. statsmodels' `cov_type="HAC"` defaults it **off**, so at each
  library's own defaults the HAC `bse` differ by exactly `sqrt(n/(n-k))`; pass
  `use_correction=False` to reproduce a default statsmodels call.
- `ewc_b=None` (`ewc` only) uses `ewc_default_b(n) = round(0.4 n^(2/3))`, the
  LLSW size-power recommendation (9 at `n = 100`, 22 at `n = 400`). Smaller
  `B` smooths more and is wider; `B` must lie in `[1, n - 1]`.
- **Inert combinations are refused, naming the keyword**: `ewc_b` with any
  `se_type` other than `"ewc"`; an explicit `maxlags` with `"ewc"` (EWC has no
  lag truncation); `use_correction=False` with `"ewc"` (EWC never takes the
  `n/(n-k)` factor — an explicit `True` cannot be told from the default and is
  a documented no-op). The older silent absorption is unchanged and documented:
  under `nonrobust`/`hc*` an explicit `maxlags`/`use_correction` is still
  ignored, as before 0.11. Changing that is a separate behaviour change. An
  unknown `se_type` is checked before any of these, so `se_type="ewx"` (or
  `"EWC"`) beside `ewc_b` is refused as an unknown `se_type`, not as an inert
  `ewc_b`.

**How to read the output.** A dict with `params`, `bse`, `tvalues` and the
echoed `se_type`. Under `"ewc"` two keys are added: `ewc_b` (the resolved `B`,
which is also the t degrees of freedom) and `pvalues` (two-sided, from
Student `t_{ewc_b}`). Build EWC intervals as
`params ± t_{ewc_b, 0.975} · bse` — at `B = 9` that critical value is 2.26,
not 1.96. The other routes return no p-values: refer their `tvalues` to the
normal (to Student t with `n-k` df for `nonrobust`, as statsmodels does).

**Failure modes.**

- **Kernel HAC under-covers in small samples**, and the remedy is not a bigger
  `maxlags` with normal critical values — wide smoothing with ±1.96 recreates
  the overconfidence. Measured below.
- **EWC buys size with width.** Its intervals are wider (about 26% at
  `T = 100` in the table below); with weakly persistent data the size gain is
  small and the width is the price.
- **`hc2`/`hc3` near leverage one**: `1 - h_t` is computed by cancellation, so
  agreement with statsmodels degrades smoothly once `1 - h_t` falls below about
  `1e-5`; an observation with leverage numerically equal to one is refused
  rather than returned as a near-infinite standard error.
- Strong persistence (roots near one) defeats every HAR estimator here.
- **EWC is SUPERLINEAR in `n`.** The cosines are evaluated directly —
  `n · B` of them, about `n · B · (k + 8)` operations — so at the default `B`
  the cost grows like `n^(5/3)`: for one series 0.03 s at `n = 10^4`, 1.3 s at
  `10^5` (1.4 s for `ols` with `k = 2`) and 57 s at `n = 972427`, the largest
  series whose default `B` fits a work budget of 2^35 (about a minute). Past
  it `ewc_lrv`/`ols` refuse before any work, naming `b`/`ewc_b` and the
  largest value the budget admits: a series of 10^6 at its default `B = 4000`
  is just past it (`b <= 3817` runs). EWC stays valid at a smaller `B`,
  paired with `t_B` (`lab/audit/wave11/ewc_varcov_measurements.py --ewc-cost`,
  output committed next to it).

**Measured coverage.** Nominal 95% intervals for a slope whose true value is
zero, `y = 1 + 0·x + u` with `x` and `u` independent AR(1)s (ρ = 0.7), 10000
seeded replications per row, `hac` at its defaults with normal critical
values, `ewc` at its defaults with `t_B` critical values — both through
`tsecon.ols` (`lab/audit/wave11/ewc_varcov_measurements.py`,
output committed next to it):

| T | `hac` (Newey-West, `maxlags`) | `ewc` (`B`) | mean width `hac` → `ewc` | MC se `hac` / `ewc` |
|---|---|---|---|---|
| 100 | **0.855** (4) | **0.915** (9) | 0.520 → 0.656 | 0.0035 / 0.0028 |
| 400 | **0.905** (5) | **0.940** (22) | 0.288 → 0.336 | 0.0029 / 0.0024 |

The MC standard error is the binomial `sqrt(p (1 − p) / 10000)` at each arm's
measured coverage `p`, not at the nominal 0.95.

EWC cuts the shortfall from 9.5 points to 3.5 at `T = 100` and from 4.5 to 1.0
at `T = 400`. Neither reaches nominal: with ρ = 0.7 in both the regressor and
the error the scores are strongly persistent, and EWC is a size improvement,
not a guarantee.

**Validated against.**

- `nonrobust`, `hac`: statsmodels `OLS(...).fit(cov_type="HAC",
  cov_kwds={"maxlags": L, "use_correction": ...})` — independent package —
  params, bse, t-values over `maxlags` {4, 8, 12} × `use_correction`
  ([`hac.json`](../../../fixtures/hac.json)), 1e-10.
- `hc0`…`hc3`: statsmodels `fit(cov_type="HC*")` — bse, t-values and full
  covariance matrices on two designs, plus the hat diagonal
  ([`hc_robust.json`](../../../fixtures/hc_robust.json)), 1e-10.
- `ewc`: **documented closed form, not an independent package** — no reference
  library implements EWC (the generator greps every installed source file of
  statsmodels 0.15.0, `arch` 8.0.0 and `linearmodels` 7.0 and finds none). The
  LLSW formula is evaluated through an independent code path, SciPy's
  FFT-based type-II DCT, on statsmodels' OLS residuals:
  params / bse / t-values / covariance at 1e-10 and `t_B` p-values against
  `scipy.stats.t.sf` at rtol 1e-9, on a `k = 2` and a `k = 3` design at the
  default `B` and at `B = 8` ([`ewc.json`](../../../fixtures/ewc.json),
  [`ewc_golden.rs`](../../../crates/tsecon-hac/tests/ewc_golden.rs),
  [`test_ewc_inference.py`](../../../bindings/python/tests/test_ewc_inference.py)).
  Exact identities as property tests: at `B = n - 1` the EWC covariance is
  HC0's times `n/(n-1)`; the `k = 1` matrix form is bit-identical to the
  scalar estimator.

**References.** White (1980); MacKinnon & White (1985); Long & Ervin (2000);
Newey & West (1987, 1994); Kiefer & Vogelsang (2005); Lazarus, Lewis, Stock &
Watson (2018, *JBES*, "HAR Inference: Recommendations for Practice"); Müller
(2014).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
T = 200
x, u = np.zeros(T), np.zeros(T)
for t in range(1, T):                          # persistent regressor AND persistent errors
    x[t] = 0.7 * x[t - 1] + rng.standard_normal()
    u[t] = 0.7 * u[t - 1] + rng.standard_normal()
X = np.column_stack([np.ones(T), x])           # add the constant yourself
y = 1.0 + 0.3 * x + u

hac = tsecon.ols(y, X)                         # default: Newey-West HAC, maxlags = floor(4 (T/100)^(2/9))
ewc = tsecon.ols(y, X, se_type="ewc")          # LLSW: B = round(0.4 T^(2/3)), Student t_B
print("slope       :", round(ewc["params"][1], 4))
print("HAC bse / t :", round(hac["bse"][1], 4), round(hac["tvalues"][1], 2))
print("EWC bse / t :", round(ewc["bse"][1], 4), round(ewc["tvalues"][1], 2), " B =", ewc["ewc_b"])
print("EWC p (t_B) :", round(ewc["pvalues"][1], 4))
# slope       : 0.2744
# HAC bse / t : 0.1119 2.45
# EWC bse / t : 0.1212 2.26  B = 14
# EWC p (t_B) : 0.0401
```

---

## `ewc_lrv`, `ewc_default_b` — the equal-weighted cosine long-run variance

**What it estimates.** The long-run variance `Omega = sum_j gamma_j` of a
stationary series — the variance that belongs under `sqrt(n)` times its mean —
by projecting the series on the first `B` type-II cosine basis vectors and
averaging the squared coefficients:

```text
Lambda_j  = sqrt(2/n) * sum_{t=1..n} cos(pi j (t - 1/2) / n) * x_t,   j = 1..B
Omega_hat = (1/B) * sum_{j=1..B} Lambda_j^2
```

`ewc_default_b(n)` returns the LLSW recommendation `round(0.4 n^(2/3))`,
clamped to `[1, n - 1]`. The rounding is exact for every `n`: it is decided in
integer arithmetic (`r` is the rounding iff `125 (2r - 1)^3 <= 64 n^2 <
125 (2r + 1)^3`), because a double-precision `n^(2/3)` rounds the wrong way
near a half-integer from `n = 1,755,027,219` on (`fixtures/ewc.json` pins
eleven such `n`, and ten just below a half-integer that it rounds correctly).

**Assumptions.** Stationarity and weak dependence. The `Lambda_j` at low
frequencies are then approximately independent `N(0, Omega)`, which is what
makes `Omega_hat ~ Omega chi2_B / B` and the t-statistic Student `t_B`.

**When to use (and when not).** For the standard error of a mean, a
long-run-variance-normalised test statistic, or any place you would otherwise
plug a kernel LRV into a normal critical value. Not for near-unit-root series
(see `ols` above), and not as a spectral-density estimate at a nonzero
frequency.

**Key arguments and defaults (and why).** `x` is demeaned internally — the
mean frequency `j = 0` is excluded, so the estimate does not depend on the
mean in exact arithmetic, and demeaning first only helps conditioning (the
same convention as `long_run_variance`). `b=None` uses `ewc_default_b(len(x))`;
an explicit `b` must lie in `[1, n - 1]`. At `b = n - 1` the cosine basis is
complete and the estimate collapses to the ordinary sample variance
`sum (x - mean)^2 / (n - 1)` — no long-run smoothing at all.

**How to read the output.** A float. Pair it with Student-t critical values
with `b` degrees of freedom: `(mean - mu0) / sqrt(Omega_hat / n)` is `t_b`
under the null. Comparing it with ±1.96 throws away the reason to use it.

**Failure modes.** Refused with a `ValueError` naming the argument: `b`
outside `[1, n - 1]` (named `b`, with the default it would have used), fewer
than two observations, NaN/inf in `x`, or values so large the computation
overflows double precision (all named `x`), `n < 2` in
`ewc_default_b` (named `n`); a negative or absurd (≥ 2^48) count is named by
the coercion layer (`b=-1`); a request past the work budget is refused naming
`b` (the cost is above, under `ols`). Small `B` is noisy by design — that is what the
`t_B` critical value pays for.

**Validated against.** The same documented-closed-form golden as `ols`'s EWC
route: `ewc_lrv` on the Nile series and a seeded AR(1), `b` ∈ {1, 4, 9, 16,
n − 1}, raw and demeaned, against `scipy.fft.dct(type=2, norm="ortho")` at
1e-12; `ewc_default_b` exactly, on the spec's `n`, the ten `n` below 10^6
whose `0.4 n^(2/3)` lies closest to a half-integer, the ten `n` in
10^6–3·10^10 closest above a half-integer (`n = 1,755,027,219` among them; a
float power rounds each one down) and the ten closest below one, and
`n = 100,292,764,931,117` — all within 2e-7 of a tie — against the integer
rule.

```python
import numpy as np, tsecon
from scipy import stats

rng = np.random.default_rng(1)
reps, n = 1000, 200
e = rng.standard_normal((reps, n))
y = np.empty((reps, n))
y[:, 0] = e[:, 0] / np.sqrt(1 - 0.5**2)        # stationary start
for t in range(1, n):                          # AR(1), phi = 0.5, mean 0: long-run variance 4
    y[:, t] = 0.5 * y[:, t - 1] + e[:, t]

b = tsecon.ewc_default_b(n)                    # round(0.4 * 200^(2/3)) = 14
crit = stats.t.ppf(0.975, b)                   # the t_14 critical value, not 1.96
lrv_ewc = np.array([tsecon.ewc_lrv(r, b=b) for r in y])     # demeaned internally
lrv_nw = np.array([tsecon.long_run_variance(r) for r in y])  # Bartlett, Newey-West maxlags
t_ewc = y.mean(axis=1) / np.sqrt(lrv_ewc / n)
t_nw = y.mean(axis=1) / np.sqrt(lrv_nw / n)
print("B =", b, " t crit =", round(crit, 3))
print("mean LRV  EWC:", round(lrv_ewc.mean(), 2), " Bartlett:", round(lrv_nw.mean(), 2), " (truth 4)")
print("size at 5%  EWC + t_B:", np.mean(np.abs(t_ewc) > crit), " Bartlett + normal:", np.mean(np.abs(t_nw) > 1.96))
# B = 14  t crit = 2.145
# mean LRV  EWC: 3.78  Bartlett: 2.82  (truth 4)
# size at 5%  EWC + t_B: 0.066  Bartlett + normal: 0.111
```

The Bartlett estimate is biased down (the kernel truncates the
autocovariances it cannot see) and the normal critical value then ignores its
noise; EWC is nearly unbiased, noisier, and the `t_B` critical value accounts
for exactly that noise — 1000 replications, one seed, so each printed size
carries a binomial Monte Carlo standard error of about 0.008 (0.010 for the
Bartlett rate).

---

## `long_run_variance` — the kernel long-run variance

**What it estimates.** `gamma_0 + 2 sum_j w_j gamma_j` with kernel weights
`w_j` — Bartlett (Newey-West), Parzen, quadratic spectral, or truncated — on
the demeaned series.

**Key arguments and defaults (and why).** `kernel="bartlett"`;
`bandwidth=None` applies the Newey-West rule of thumb
`floor(4 (n/100)^(2/9))`, the value `ols(se_type="hac")` uses.

**How to read the output and when not to.** A float, to be compared with
**normal** critical values — the classical HAC convention whose small-sample
under-coverage the EWC route exists to fix. The truncated kernel is not
positive semi-definite and can return a negative variance on short, strongly
dependent series.

**Validated against.** Bartlett LRVs on the demeaned Nile at bandwidths 5, 10,
20 ([`hac.json`](../../../fixtures/hac.json), documented formula), and the
statsmodels HAC sandwich that shares its kernel weights, 1e-10.
