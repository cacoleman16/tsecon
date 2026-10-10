# Model card — DF-GLS, Ng-Perron, Phillips-Perron, Phillips-Ouliaris, Zivot-Andrews, and variance-ratio tests

`dfgls` · `ng_perron` · `phillips_perron` · `phillips_ouliaris` · `zivot_andrews` · `variance_ratio`

Five unit-root-family tests beyond the core ADF/KPSS pair, and one test that
starts where they stop. `dfgls` attacks the
ADF's power problem: estimating a constant or trend by OLS costs the plain ADF
real power near the unit-root boundary, and GLS-detrending at the ERS local
alternative recovers most of it — it is the recommended default over plain ADF
whenever deterministics must be estimated. `ng_perron` builds on the same
GLS-detrending engine and attacks the *size* problem too: the M statistics
with MAIC lag selection are the standard remedy when a large negative MA root
(an over-differenced series) wrecks the size of every ADF-family test.
`phillips_perron` and
`phillips_ouliaris` are semiparametric: instead of adding lagged differences to
soak up serial correlation (the augmentation in ADF and Engle-Granger), they
estimate a simple regression by OLS and then **correct the test statistic** for
the residual's long-run variance with a nonparametric (Bartlett) kernel.
`phillips_perron` is the unit-root test — a drop-in alternative to
[`adf`](diagnostics.md); `phillips_ouliaris` is its cointegration analog — a
residual-based alternative to [`johansen`](cointegration-regime.md) for a single
cointegrating relationship. `zivot_andrews` guards the workflow against a
one-time structural break masquerading as a unit root. All five are companions
to the confirmatory stationarity workflow, not replacements for reading ADF and
KPSS together.

`variance_ratio` is **not** a unit-root or stationarity test, and it sits on
this card because it is so often mistaken for one. It takes the unit root as
given — `y` is the level of a price or index — and asks the next question: are
the *increments* serially uncorrelated (a random walk, a martingale), or do
they trend (momentum) or revert? A series with autocorrelated increments has a
unit root too; the variance-ratio test tells the two apart, and its default
statistic stays valid under the volatility clustering that makes Ljung-Box on
returns over-reject.

| Function | Null hypothesis | The analog it complements |
|----------|-----------------|---------------------------|
| `dfgls` | the series has a unit root | `adf` (GLS-detrended, near-optimal local power) |
| `ng_perron` | the series has a unit root | `dfgls` (the M statistics + MAIC, size-robust under a negative MA root) |
| `phillips_perron` | the series has a unit root | `adf` (semiparametric, no lag augmentation) |
| `phillips_ouliaris` | the regressors are **not** cointegrated with `y` | Engle-Granger; `johansen` (single-equation route) |
| `zivot_andrews` | a unit root with **no** break | `adf` when a one-time structural break may masquerade as a root |
| `variance_ratio` | the **increments** are serially uncorrelated (a random walk) | `ljung_box` on the differences (robust to ARCH; one joint test across horizons) |

---

## `dfgls` — GLS-detrended Dickey-Fuller (Elliott-Rothenberg-Stock)

**What it estimates.** The DF-GLS statistic of Elliott, Rothenberg & Stock
(1996): the deterministics are first estimated by GLS under the *local
alternative* $\rho = 1 + \bar c/T$ — quasi-differencing $y$ and the trend
columns at $\bar c = -7$ (`regression="c"`) or $\bar c = -13.5$ (`"ct"`), the
points where the local asymptotic power envelope is tangent at 50% power — and
the ADF regression is then run on the detrended series **with no deterministic
terms**. The statistic is the usual $t$-ratio on the lagged level; only the
detrending changed, but that change buys near-optimal local power where the
plain ADF wastes it re-estimating the mean or trend.

**Assumptions.** Same as ADF: the only nonstationarity under the null is a unit
root, and the augmentation lags must absorb the serial correlation. There is no
`"n"` case — with no deterministics to estimate, GLS detrending is a no-op and
plain `adf(regression="n")` already sits on the power envelope. The test
inherits ADF's size distortion under a large negative MA root (the Ng-Perron
MAIC lag rule is the standard remedy — reach for [`ng_perron`](#ng_perron-the-m-unit-root-tests-with-maic-ng-perron-2001)
in that case).

**When to use (and when not).** Use as your default unit-root test whenever a
constant or trend must be estimated — which is nearly always — and especially
when the series is suspected to be *near*-integrated (ρ close to 1), where the
ADF's power loss bites hardest. Do not use it to dodge the deterministics
decision: choosing `"c"` vs `"ct"` matters exactly as much as it does for ADF.
Do not read a non-rejection alone as proof of a unit root — pair it with `kpss`
for the confirmatory quadrant, as with every unit-root test.

**Key arguments and defaults (and why).** `regression`: `"c"` (constant;
default) or `"ct"` (constant + trend). `lags`: a fixed number of augmentation
lags; `None` (default) selects automatically. `method`: `"aic"` (default),
`"bic"`, or `"t-stat"`; following Perron & Qu (2007) — and arch — the selection
runs on the **OLS**-detrended series with no deterministics, which improves
finite-sample power over selecting on the GLS-detrended series. `max_lags`
caps the search; `None` uses Schwert's $\lceil 12 (T/100)^{1/4}\rceil$ capped
at $(T-1)/2 - 1$. When `lags` is given, `method`/`max_lags` are ignored (arch
behavior).

**How to read the output.** `statistic`, `p_value`, `used_lag`, `nobs`
($= T - 1 - \texttt{used\_lag}$), `crit` (1/5/10% critical values at `nobs`),
`trend`. **Small `p_value` ⇒ reject the unit root.** The critical values are
*not* the ADF ones — the `"ct"` case in particular has its own distribution —
so compare the statistic only against the `crit` values reported here. Quote
`used_lag`: a DF-GLS verdict without its lag length is not reproducible.

**Failure modes.** Reading it alone (pair with KPSS); leaving `"c"` when the
alternative is trend-stationary (the test then *diverges toward* non-rejection
on trending data — see the trend-stationary fixture case, p ≈ 0.99 under `"c"`
vs p ≈ 0 under `"ct"`); heavy negative MA errors (size distortion; AIC tends
to pick too few lags there). A `lags` or `max_lags` the series cannot carry
(`n >= 2·lags + 3`, and `n >= 3 + ntrend + lags` for a fixed lag) or past
the lag budget shared with `adf` (2^26 design entries, 2^35 multiply-adds;
see the diagnostics card) is refused before any lag column is built, naming
the keyword and the largest value that fits.

**Validated against.** `arch.unitroot.DFGLS` (arch 8.0.0, an independent
implementation) for the statistic, the AIC/BIC/t-stat selected lag, `nobs`,
p-value, and critical values on the Nile series, three seeded random walks, a
trend-stationary series, i.i.d. noise, and fixed-lag/max-lags cases — statistic
at 1e-10 relative, lags exact ([`dfgls.json`](../../../fixtures/dfgls.json),
[`dfgls_golden.rs`](../../../crates/tsecon-diag/tests/dfgls_golden.rs)).
**Honest grading of the p-value/CV layer:** they reproduce the response
surfaces shipped with arch (Sheppard's MacKinnon-methodology simulations,
transcribed with attribution) bit-for-bit; they are not an independently
published table. See the [validation matrix](../validation-matrix.md).

**References.** Elliott, Rothenberg & Stock (1996); Ng & Perron (2001); Perron
& Qu (2007); MacKinnon (1994, 2010, the response-surface methodology).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
walk = np.cumsum(rng.standard_normal(200))          # a random walk (unit root)

r = tsecon.dfgls(walk, regression="c")
print("DF-GLS(walk):", round(r["statistic"], 4), " p:", round(r["p_value"], 4),
      " lag:", r["used_lag"], " 5% cv:", round(r["crit"]["5%"], 3))

ar = np.empty(200); ar[0] = 0.0
for t in range(1, 200):
    ar[t] = 0.85 * ar[t - 1] + rng.standard_normal()  # near-integrated AR(1)
print("DF-GLS(AR 0.85):", round(tsecon.dfgls(ar)["statistic"], 4),
      " p:", round(tsecon.dfgls(ar)["p_value"], 4))
```

```
DF-GLS(walk): -1.1486  p: 0.2358  lag: 0  5% cv: -2.047
DF-GLS(AR 0.85): -4.7458  p: 0.0
```

The random walk cannot reject; the stationary-but-persistent AR(0.85) rejects
decisively — the near-integrated regime is exactly where DF-GLS's power edge
over plain ADF shows. These match `arch.unitroot.DFGLS` to machine precision.

---

## `ng_perron` — the M unit-root tests with MAIC (Ng & Perron 2001)

**What it estimates.** The four M statistics of Ng & Perron (2001) — MZa, MZt,
MSB, MPT — on the GLS-detrended series, with the lag length chosen by their
**MAIC** and the long-run variance from the **autoregressive spectral density
estimator at frequency zero**. The detrending is bit-for-bit the `dfgls`
engine (same quasi-differencing, same $\bar c = -7$ / $-13.5$). With
$\tilde y_t$ the detrended series, $\kappa = T^{-2}\sum_{t=1}^{T-1}\tilde y_t^2$
and $s^2_{AR} = \hat\sigma_e^2 / (1-\hat b(1))^2$ from the trendless ADF
autoregression:

$$MZ_\alpha = \frac{T^{-1}\tilde y_T^2 - s^2_{AR}}{2\kappa},\qquad
  MSB = \sqrt{\kappa / s^2_{AR}},\qquad MZ_t = MZ_\alpha \times MSB,$$

and MPT is the modified ERS point-optimal statistic. All four reject the
unit-root null when **small** (below the critical value); $MZ_t = MZ_\alpha
\times MSB$ is an exact identity, enforced as an internal invariant test.

**Assumptions.** As for DF-GLS. The M statistics were built precisely for the
case that breaks the rest of the family: a large negative MA root in the
differences (the classic outcome of over-differencing), where ADF/DF-GLS with
AIC/BIC lag selection reject a true unit root far too often. MAIC's
data-dependent penalty lengthens the lag there, and the M statistics keep size
close to nominal where the Phillips-Perron kernel correction fails completely.

**When to use (and when not).** Use whenever you suspect over-differencing or
an MA component in the shocks, or as the size-robust confirmation of a `dfgls`
rejection. Do not expect p-values: none exist (no published response surface;
this library declines to fabricate one), so read the verdict off the reported
critical values. And know the documented flip side (Perron & Qu 2007): on data
*far* from the null — an obviously stationary series — MAIC drives the lag to
its maximum and power collapses (the crate pins a case where lag-0 MZa is
−124.9 and the MAIC lag-16 MZa fails to reject). If the series is plainly
stationary, a unit-root test is the wrong tool; if you must, cap `max_lags` or
fix `lags`.

**Key arguments and defaults (and why).** `trend`: `"c"` (constant; default)
or `"ct"` (constant + trend) — the same choice, with the same stakes, as
`dfgls`. `lags`: `None` or `"maic"` (default) selects by MAIC **on the
GLS-detrended series** (the paper's own rule — not the Perron-Qu OLS-detrended
variant `dfgls` uses for AIC/BIC); an integer fixes the lag. `max_lags` caps
the MAIC search; `None` uses Schwert's $\lceil 12(T/100)^{1/4}\rceil$ capped at
$(T-1)/2-1$ (the library-wide default; the paper's simulations truncate rather
than round up — pass an explicit cap to reproduce a specific study).

**How to read the output.** `mza`, `mzt`, `msb`, `mpt`, `used_lag`, `nobs`
($= T - 1 - \texttt{used\_lag}$), `s2_ar`, `crit` (per-statistic 1/5/10%
values from Ng-Perron 2001, Table 1), `trend`. **Reject the unit root when a
statistic is below its critical value** — for all four, including MSB and MPT
(positive statistics that shrink under the alternative). Quote `used_lag`; an
M-test verdict without its lag is not reproducible. The statistics agree in
spirit but are four tests, not one: MZa/MZt near their DF-GLS analogs, MSB
the Sargan-Bhargava ratio, MPT the point-optimal form.

**Failure modes.** Reading a non-rejection alone as proof of a unit root (pair
with `kpss`, as always); the MAIC power reversal on far-from-null data (above);
treating the asymptotic critical values as finite-sample exact in very short
samples (the table is asymptotic-only — the measured size at $T=250$ runs
0.027–0.045 at the 5% values, slightly conservative, matching the paper's
Table 2); comparing MZt to ADF critical values (its distribution matches
DF-GLS, not ADF, and the other three have their own tables). Lag orders are
sized and refused as for `dfgls`, naming `lags` or `max_lags`.

**Validated against.** **No runnable independent implementation exists** —
statsmodels 0.14.6 and arch 8.0.0 do not ship the M tests (a canary test
asserts this stays true), and no CRAN package does (`urca::ur.ers` stops at
DF-GLS + the ERS P-test; `bootUR`/`CADFtest` borrow only MAIC) — so there is
no reference-run golden, and the claim is carried honestly by: the transcribed
Table 1 (cross-checked against the independent transcription in the GAUSS
`tspdlib`); seeded Monte-Carlo **size at the asymptotic critical values** —
lag-0 at $T=1000$, 2000 reps: 5% rejection measured 0.039–0.058 and 1%
measured 0.006–0.010 across all eight statistic/trend combinations; the full
MAIC pipeline at $T=250$: 0.027–0.045 at 5%; **power ordering** (AR(0.8)
rejects ~0.80 vs size ~0.03); the MAIC mechanism (mean lag 0.9 i.i.d. vs 7.8
under MA(−0.8)); the exact $MZ_t = MZ_\alpha MSB$ identity; a bitwise pin
that the detrended series is the `dfgls` engine's; and an independent NumPy
re-implementation of the whole pipeline re-pinned through the Python binding
at 1e-9. See the [validation matrix](../validation-matrix.md).

**References.** Ng & Perron (2001); Perron & Ng (1996, the M statistics);
Stock (1999); Elliott, Rothenberg & Stock (1996, the detrending); Perron & Qu
(2007, the power-reversal caveat).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
walk = np.cumsum(rng.standard_normal(200))          # a random walk (unit root)

r = tsecon.ng_perron(walk, trend="c")
print("MZa:", round(r["mza"], 4), " MZt:", round(r["mzt"], 4),
      " MSB:", round(r["msb"], 4), " MPT:", round(r["mpt"], 4),
      " lag:", r["used_lag"])
print("5% cvs:", r["crit"]["mza"]["5%"], r["crit"]["mzt"]["5%"],
      r["crit"]["msb"]["5%"], r["crit"]["mpt"]["5%"])

ar = np.empty(200); ar[0] = 0.0
for t in range(1, 200):
    ar[t] = 0.85 * ar[t - 1] + rng.standard_normal()  # near-integrated AR(1)
r2 = tsecon.ng_perron(ar)
print("AR(0.85) MZa:", round(r2["mza"], 4), " MZt:", round(r2["mzt"], 4),
      " lag:", r2["used_lag"])
```

```
MZa: -2.6666  MZt: -1.1431  MSB: 0.4287  MPT: 9.144  lag: 0
5% cvs: -8.1 -1.98 0.233 3.17
AR(0.85) MZa: -27.3873  MZt: -3.7002  lag: 3
```

The random walk sits well inside the acceptance region on all four statistics;
the near-integrated AR(0.85) rejects decisively (MZa −27.4 against a 1% value
of −13.8). Same detrended series as `dfgls`, but with the size insurance the
M statistics were designed to buy.

---

## `phillips_perron` — semiparametric unit-root test

**What it estimates.** The Phillips-Perron (1988) $Z_\tau$ (default) or $Z_\alpha$
statistic for a unit root. It runs the Dickey-Fuller *level* regression
$y_t = \mu + \delta t + \rho\, y_{t-1} + u_t$ by OLS with **no** lagged
differences, then corrects the raw $t$-statistic (or the $T(\hat\rho-1)$
statistic) for serial correlation using a Bartlett kernel estimate of the
residual long-run variance. Same nonstandard Dickey-Fuller null distribution as
ADF, so the MacKinnon (1996, 2010) response-surface p-values apply.

**Assumptions.** The only nonstationarity is a unit root (a deterministic trend
must be modeled through `regression="ct"`). The nonparametric correction handles
serial correlation and heteroskedasticity of *unknown* form — its strength — but
the test is known to have size distortions when the series has a large negative
MA root (a shared weakness with ADF), and low power near the unit-root boundary
(why you still pair it with KPSS).

**When to use (and when not).** Use as an ADF alternative when you would rather
not choose an augmentation lag length, or as a robustness cross-check on an ADF
verdict — agreement between the two is reassuring. Do not treat a failure to
reject as evidence *of* a unit root (low power); do not read it alone — run
[`check_stationarity`](diagnostics.md) or pair it with `kpss` for the
confirmatory quadrant. Prefer ADF when a strong negative MA component is
suspected.

**Key arguments and defaults (and why).** `regression`: `"n"` (no
deterministics), `"c"` (constant; default), `"ct"` (constant + trend) — the same
"match the deterministics to the stationary alternative" choice that dominates
ADF. `test_type`: `"tau"` (the $Z_\tau$ $t$-form; default) or `"rho"` (the
$Z_\alpha$ coefficient form). `lags`: the Bartlett bandwidth; `None` uses the
$\lceil 12\,(n/100)^{1/4}\rceil$ rule (arch's default).

**How to read the output.** `stat` (the requested statistic), `pvalue`
(MacKinnon), `crit` (the 1/5/10% critical values), `lags` (the bandwidth used),
`nobs`, plus both `ztau` and `zalpha` for convenience. **Small `pvalue` ⇒ reject
the unit root** (the series looks stationary). Quote the bandwidth: a PP result
without its `lags` is not reproducible.

**Failure modes.** Reading PP alone (same trap as ADF alone); size distortion
under a large negative MA root; mistaking a deterministic trend for a root by
leaving `regression="c"` when `"ct"` is called for.

**Validated against.** `arch.unitroot.PhillipsPerron` (Sheppard) for both $Z_\tau$
and $Z_\alpha$ — an independent package — to < 1e-10, with MacKinnon
response-surface p-values ([`phillips.json`](../../../fixtures/phillips.json),
[`phillips_golden.rs`](../../../crates/tsecon-diag/tests/phillips_golden.rs)).
See the [validation matrix](../validation-matrix.md).

**References.** Phillips & Perron (1988); MacKinnon (1996, 2010); Newey & West
(1987, the long-run-variance kernel).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
walk = np.cumsum(rng.standard_normal(300))          # a random walk (unit root)
stat = rng.standard_normal(300)                     # i.i.d. (stationary)

pp = tsecon.phillips_perron(walk, regression="c", test_type="tau")
print("PP(walk)  Z-tau:", round(pp["stat"], 4), " p:", round(pp["pvalue"], 4),
      " bandwidth:", pp["lags"])
print("  5% critical value:", round(pp["crit"]["5%"], 3))
print("PP(stationary) Z-tau:", round(tsecon.phillips_perron(stat)["stat"], 4),
      " p:", round(tsecon.phillips_perron(stat)["pvalue"], 4))
```

```
PP(walk)  Z-tau: -0.7675  p: 0.8285  bandwidth: 16
  5% critical value: -2.871
PP(stationary) Z-tau: -18.7697  p: 0.0
```

The random walk cannot reject the unit root (p ≈ 0.83, statistic well above the
−2.87 critical value); the i.i.d. series rejects overwhelmingly. These match
`arch.unitroot.PhillipsPerron` to machine precision.

---

## `phillips_ouliaris` — residual cointegration test

**What it estimates.** The Phillips-Ouliaris (1990) $Z_t$ (default) or $Z_\alpha$
residual test for cointegration. It regresses `y` on the stochastic regressors
`x` (plus the chosen deterministics) by OLS, then applies the Phillips-Perron
correction to a unit-root test **on the regression residual**. Under the null of
no cointegration that residual has a unit root; a cointegrating relationship
makes it stationary, so a large negative statistic rejects "no cointegration".

**Assumptions.** The variables in `[y, x]` are each I(1); a single cointegrating
vector is the alternative (this is a single-equation test — for the *number* of
cointegrating relations use [`johansen`](cointegration-regime.md)). The null
distribution depends on the number of regressors, so the critical values are
indexed by $N = 1 + \dim(x)$.

**When to use (and when not).** Use for a quick, single-equation cointegration
check when one series is a natural dependent variable (a spread, an arbitrage
relation) — the Engle-Granger workflow, with the semiparametric correction.
Do not add your own constant column to `x` (deterministics come from `trend`);
do not use it to count cointegrating vectors (that is Johansen's job); remember
the test is not invariant to which variable you place on the left.

**Key arguments and defaults (and why).** `x` is a 2-D `(T, m)` matrix of the `m`
stochastic regressors, used as-is. `trend`: `"n"`, `"c"` (default), `"ct"` — the
deterministics in the cointegrating regression. `test_type`: `"Zt"` (default) or
`"Za"`. `bandwidth`: the Bartlett bandwidth of the residual AR(1); `None` uses
the $\lfloor 4((T-1)/100)^{2/9}\rfloor$ rule.

**How to read the output.** `stat`, `pvalue`, `crit`, `lags` (bandwidth),
`nobs`, `n_vars` ($N = 1 + m$). **Small `pvalue` ⇒ reject no cointegration** (the
series move together in the long run). `Zt` p-values and critical values use the
MacKinnon N-surfaces (the statsmodels `coint` route); **`Za` is statistic-only**
(`pvalue`/`crit` are `None`) because the library deliberately declines to ship
arch's proprietary $Z_\alpha$ simulation surface.

**Failure modes.** Adding a redundant constant column to `x` (double-counts the
deterministic); reading it as a rank test; swapping the dependent variable and
getting a different verdict (a known non-invariance of single-equation tests).

**Validated against.** `arch.unitroot.cointegration.phillips_ouliaris` for the
statistics — an independent package — with `Zt` p-values/critical values from the
statsmodels MacKinnon cointegration N-surfaces
([`phillips.json`](../../../fixtures/phillips.json),
[`phillips_golden.rs`](../../../crates/tsecon-diag/tests/phillips_golden.rs)).

**References.** Phillips & Ouliaris (1990); Engle & Granger (1987); MacKinnon
(1996, 2010).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
T = 300
x = np.cumsum(rng.standard_normal(T))               # an I(1) regressor
y = 1.5 * x + rng.standard_normal(T)                # cointegrated with x
Xreg = x.reshape(-1, 1)                             # (T, 1) — no constant column

po = tsecon.phillips_ouliaris(y, Xreg, trend="c", test_type="Zt")
print("PO(cointegrated) Zt:", round(po["stat"], 4), " p:", round(po["pvalue"], 4),
      " N:", po["n_vars"])

y2 = np.cumsum(rng.standard_normal(T))              # an independent random walk
po2 = tsecon.phillips_ouliaris(y2, Xreg, trend="c", test_type="Zt")
print("PO(independent)  Zt:", round(po2["stat"], 4), " p:", round(po2["pvalue"], 4))
```

```
PO(cointegrated) Zt: -19.4407  p: 0.0  N: 2
PO(independent)  Zt: -2.9978  p: 0.1107
```

The genuinely cointegrated pair rejects "no cointegration" decisively; two
independent random walks do not (p ≈ 0.11) — the spurious-regression trap the
test exists to catch. Because it is single-equation, use `johansen` when you need
to know *how many* cointegrating relations a larger system supports.

---

## `zivot_andrews` — unit root against break-stationarity

**What it estimates.** The Zivot-Andrews (1992) minimum-t unit-root statistic
with **one endogenous structural break**. Perron (1989) showed that a stationary
series with a one-time level or trend shift fools the ADF test into "finding" a
unit root; Zivot-Andrews turns Perron's known-break-date test into an estimated
one: for every candidate break date inside a trimmed window it runs the ADF-style
regression with the chosen break dummies and takes the **minimum** t-statistic on
the lagged level — the break date least favorable to the unit-root null. Because
the date is estimated, the null distribution shifts far left of the ADF one and
has its own critical values (a simulated table, not MacKinnon surfaces).

**Assumptions.** At most one break, and only under the *alternative*: the null is
a no-break unit root. That asymmetry is the test's classic weakness — a unit root
*with* a genuine break (a broken drift) is outside both hypotheses and produces
spurious rejections (Lee & Strazicich 2003 fix this with a minimum-LM test that
allows the break under both; not yet shipped, flagged here for honesty). Lag
selection follows the statsmodels/Baum convention: a *single* up-front ADF
(`"ct"`, no dummies) autolag pass fixes the augmentation lag for all candidate
regressions — slightly more pessimistic than the paper's per-candidate
re-selection.

**When to use (and when not).** Use when ADF/PP fail to reject but the plot shows
a one-time event (a policy regime change, a dam, a reunification) — if
`zivot_andrews` rejects where `adf` did not, the "unit root" was likely a broken
deterministic. Do not use it to *date* breaks in a series you already believe is
stationary (that is [`bai_perron`](structural-breaks.md)'s job, with proper break
confidence intervals); do not read the estimated break date as inference — it is
a by-product of the min-t search. With two or more suspected breaks, no shipped
test applies (Lumsdaine-Papell / Lee-Strazicich territory).

**Key arguments and defaults (and why).** `regression`: which component breaks
under the alternative — `"c"` intercept shift (default, Perron's "crash" model),
`"t"` trend-slope shift, `"ct"` both; the regression itself always carries a
constant *and* a trend. `trim` (default 0.15, in `(0, 1/3]` — 0 itself is unreachable, since the candidate window must hold at least `lags + 1` observations): excludes the first
and last `int(n*trim)` observations from the break search — a break too near an
end is indistinguishable from the boundary. `autolag`/`max_lags`/`lags`: the
statsmodels lag conventions — `autolag="aic"` (default; also `"bic"`,
`"t-stat"`) with `max_lags` capping the search, or `autolag=None` with `lags`
fixed. Pass one of `lags`/`autolag`, not both — the binding refuses the
ambiguous spelling.

**How to read the output.** `stat`, `pvalue`, `crit` (1/5/10%), `break_index`,
`lags`, `nobs`, `trim`, `regression`. **Small `pvalue` ⇒ reject the no-break
unit root** in favor of break-stationarity. `break_index` is the **last
pre-break observation** — the estimated shift begins at `break_index + 1` (the
statsmodels `bpidx` convention). The p-value interpolates a simulated table
(100,000 replications): read at most two decimals into it, and treat values at
the clamps (1e-05, 0.999) as "off the table", not as exact probabilities.

**Failure modes.** Rejecting because of a break under the *null* (broken random
walk — the Lee-Strazicich critique); reading `break_index` as a dated,
confidence-bounded break (use `bai_perron`); a `trim` too small for the lag
order (the break dummy needs a pre-break regime, `lags <= int(n·trim) − 1` —
the error explains, naming `lags` when the lag is fixed and `max_lags` when the
search chose it); a `max_lags` past the base ADF's bound `n//2 − 3` or its lag
budget (refused naming `max_lags`); a lag whose break-date scan — a
`(n − 1 − lags) × (k + lags)` design refit at each of the `n − 2·int(n·trim)`
candidate breaks — exceeds the lag budget shared with `adf` (2^26 design
entries, 2^35 multiply-adds; lags up to the default cap are exempt from the
work budget), refused before the design is built, naming `lags` or the
`max_lags` that admitted it (see the diagnostics card); treating
the `"t"` model's dating convention as identical to `"ct"`'s (the reference
starts the `"t"` ramp one observation earlier; tsecon replicates it exactly and
documents it in the module).

**Validated against.** `statsmodels.tsa.stattools.zivot_andrews` (0.14.6) —
statistic to 1e-10, break index and selected lag exact, p-values/critical values
to the interpolation's resolution — with `arch.unitroot.ZivotAndrews` (8.0.0)
agreeing on every expressible case as a cross-check (the two share the same Baum
code lineage, so arch corroborates the transcription rather than independently
deriving it; graded accordingly in the fixture header)
([`zivot_andrews.json`](../../../fixtures/zivot_andrews.json),
[`zivot_andrews_golden.rs`](../../../crates/tsecon-diag/tests/zivot_andrews_golden.rs)).

**References.** Zivot & Andrews (1992); Perron (1989); Baum (2004, rev. 2015);
Schwert (1989); Lee & Strazicich (2003).

```python
import numpy as np, tsecon

rng = np.random.default_rng(7)
y = rng.standard_normal(200)                # stationary noise ...
y[100:] += 10.0                             # ... with a level shift at t = 100

za = tsecon.zivot_andrews(y, regression="c")
print("ZA(break-stationary):", round(za["stat"], 4), " p:", za["pvalue"],
      " break at:", za["break_index"] + 1)
print("ADF on the same series p:", round(tsecon.adf(y)["p_value"], 4))

rw = np.cumsum(rng.standard_normal(200))    # a genuine random walk
za2 = tsecon.zivot_andrews(rw, regression="c")
print("ZA(random walk):", round(za2["stat"], 4), " p:", round(za2["pvalue"], 4))
```

```
ZA(break-stationary): -16.3861  p: 1e-05  break at: 100
ADF on the same series p: 0.7496
ZA(random walk): -3.0233  p: 0.9051
```

The broken-but-stationary series: plain ADF is completely fooled (p ≈ 0.75, "unit
root"), while Zivot-Andrews rejects overwhelmingly *and* recovers the break date
exactly (shift begins at observation 100). The genuine random walk is correctly
not rejected. That contrast — same data, opposite verdicts — is exactly the
Perron (1989) point the test exists to make.

---

## `variance_ratio` — Lo-MacKinlay variance-ratio test (and the Chow-Denning joint test)

**What it estimates.** Whether the **increments** of a level series $y$ (a log
price, an index) are serially uncorrelated — the random-walk, or
martingale-difference, hypothesis of Lo & MacKinlay (1988). With $n_q = n - 1$
increments, the drift $\hat\mu = (y_{n-1} - y_0)/n_q$ and
$d_t = \Delta y_t - \hat\mu$, it compares the variance of $q$-period changes with
$q$ times the one-period variance:

$$\hat\sigma^2_1 = \frac{1}{n_q}\sum_t d_t^2,\qquad
\hat\sigma^2_q = \frac{1}{n_q\,q}\sum_{t=q}^{n-1}\left(y_t - y_{t-q} - q\hat\mu\right)^2,\qquad
\widehat{VR}(q) = \frac{\hat\sigma^2_q}{\hat\sigma^2_1}.$$

Under the null $VR(q) = 1$; in population $VR(q) = 1 + 2\sum_{k=1}^{q-1}(1 - k/q)\rho_k$,
so positive autocorrelation of the increments (momentum) pushes it above one and
negative autocorrelation (mean reversion) below. The statistic is
$z = \sqrt{n_q}\,(\widehat{VR} - 1)/\sqrt{V}$, asymptotically $N(0, 1)$, with
$V = 2(2q-1)(q-1)/(3q)$ under iid increments (Campbell, Lo & MacKinlay 1997, eq.
2.4.39) or the heteroskedasticity-consistent
$\hat\theta = \sum_{k=1}^{q-1} 4(1-k/q)^2 \hat\delta_k$,
$\hat\delta_k = n_q \sum_t d_t^2 d_{t-k}^2 / (\sum_t d_t^2)^2$ (eq. 2.4.43, Lo and
MacKinlay's $z^*$). With several horizons $q_1, \dots, q_m$ the Chow & Denning
(1993) joint test takes $CD = \max_i |z(q_i)|$ and refers it to the studentized
maximum modulus with infinite degrees of freedom — the Šidák bound: reject at
level $\alpha$ when $CD > \Phi^{-1}\!\left(1 - [1 - (1-\alpha)^{1/m}]/2\right)$, with
p-value $1 - (1 - 2\Phi(-CD))^m$.

**Assumptions.** $y$ is the **level** and has a unit root; the test is about its
increments, not about whether it has a root. The robust $z^*$ (the default)
needs only martingale-difference increments with finite fourth moments —
conditional heteroskedasticity such as GARCH is allowed — while the homoskedastic
$z$ needs iid increments. The Chow-Denning bound treats the $m$ statistics as
independent; they are strongly positively correlated (every horizon reuses the
same increments), so the joint test is **conservative**.

**When to use (and when not).** Use it for the market-efficiency question — is
this log price a random walk, or do its returns trend or revert? — and as the
ARCH-robust alternative to Ljung-Box on returns: Ljung-Box's null is *iid*
noise, so volatility clustering alone makes it reject uncorrelated returns,
which is exactly why [`check_series`](check-series.md) cites `variance_ratio`
when both Ljung-Box and ARCH-LM fire on the differences. Pass several horizons
and read the Chow-Denning verdict rather than the smallest of the per-horizon
p-values. Do **not** use it as a unit-root or stationarity test (a stationary
level has $VR(q)$ falling toward 0 as $q$ grows, and a trend-stationary one is
outside the null; run the ADF/KPSS quadrant first), and do not pass returns — see the failure modes.

**Key arguments and defaults (and why).** `lags` (2): one horizon $q \ge 2$, or
a list/tuple/integer array/Series of distinct horizons — then every per-horizon
key except the `lags` echo (a list of ints) is a NumPy array in the given order,
and the Chow-Denning keys span them all. Horizons are integers: a float is
refused alike as the scalar `4.0` and inside a sequence (`[4.0]`, a float array). `trend`:
`"c"` (default; a random walk with drift, increments centred at the sample
drift) or `"n"` (no drift). `overlap` (True): all $n - q$ overlapping
$q$-period sums — more efficient — versus non-overlapping blocks, which drop the
trailing $(n-1) \bmod q$ observations (`n_dropped`, as arch does).
`debiased` and `robust` are sentinels that resolve to True under
`overlap=True` (Lo-MacKinlay's unbiased variances; the robust $z^*$); the
non-overlapping estimator has neither, so passing either as True with
`overlap=False` **raises** rather than being silently ignored. The conventions
— levels in, the $n_q q$ divisor, the debiasing factor (applied under `"n"`
too) — are arch's, so a call ports one-to-one from `arch.unitroot.VarianceRatio`.

**How to read the output.** `vr`, `stat`, `pvalue` (two-sided), `stat_variance`
(the $V$ used), `nobs` (increments used, $n - 1 -$ `n_dropped`) and `n_dropped`
per horizon — Python scalars for a scalar `lags`, arrays for a sequence; `crit`
(arch's one-sided $N(0,1)$ quantiles at 1/5/10/90/95/99%); `cd_stat`, `cd_lag`
(the horizon attaining it), `cd_pvalue` and `cd_crit` (1/5/10%) for the joint
test — with one horizon, exactly the single test; and the echoes `trend`,
`se_type` (`"robust"`, `"homoskedastic"`, `"nonoverlapping"`) and `debiased`.
**Small `pvalue` ⇒ reject the random walk**; the sign of `stat` (or `vr` above or
below 1) says momentum or mean reversion.

**Failure modes.**

- **Returns passed as the level.** `y` is the level series. Passing returns
  tests the *differences of returns* — an MA(1) with a unit negative root — so
  `vr` lands near 0.5 at $q = 2$ and the test rejects loudly: in the example
  below, iid returns passed as `y` give `vr` 0.536 and `stat` −12.4. No reliable
  automatic detection exists; take `np.cumsum` of returns, or pass log prices.
- **The homoskedastic $z$ over-rejects under ARCH.** In the Monte Carlo below its
  per-horizon size under GARCH(1,1) increments is 0.085–0.113 at a nominal 5%;
  the robust $z^*$ holds 0.037–0.052. Keep the default.
- **The Chow-Denning bound is conservative.** Measured size 0.023–0.040 at a
  nominal 5% (below); a p-value just above 0.05 is weaker evidence for the null
  than it looks.
- **Short samples are refused, not answered.** Every horizon needs at least two
  $q$-period sums ($n \ge q + 2$ with overlap, $n \ge 2q + 1$ without). arch
  accepts one sum and returns a ratio of exactly 0 under trend `"c"` and a
  finite single-squared-sum ratio under `"n"` (0.943 overlapping and 1.886
  non-overlapping at $q = 2$ on the fixture's corner series) — neither carries information. Constant increments (a
  constant series, or an exact linear trend under `"c"`) are refused as 0/0.
- **P-values differ from arch's in the tail, deliberately.** tsecon computes
  $2\Phi(-|z|)$ through `erfc`; arch computes $2 - 2\Phi(|z|)$, which loses
  relative accuracy as $|z|$ grows (up to 2.56% relative where its p is still
  positive in the fixture grid) and underflows to exactly 0 beyond
  $|z| \approx 8.3$ (72 of the 250 fixture cases). Compare with arch at an
  absolute tolerance, not a relative one.
- **Horizons near $n$.** The robust variance costs $O(nq)$ per horizon —
  measured on 20,001 levels at 0.02 s for $q = 1{,}000$ and 0.18 s for
  $q = 10{,}000$, against under 1 ms with `robust=False`
  (`scripts/variance_ratio_cost.py`) — and $VR(q)$ at large
  $q/n$ is noisy and biased in finite samples even debiased. With several
  horizons (and `overlap=True`) the lag products are shared up to the longest
  horizon, so a whole profile costs about as much as that horizon alone: all
  999 horizons $q = 2, \dots, 1{,}000$ in one call take 0.05 s on the same
  series.

**Validated against.** `arch.unitroot.VarianceRatio` (arch 8.0.0, an
independent package) on 250 cases — five series (a Gaussian walk, a GARCH walk,
AR(0.3) and AR(0.6) momentum walks, log US real GDP) × $q \in \{2, 4, 5, 8, 16\}$ ×
both trends × five option sets — for `vr`, `stat` and the critical values;
the p-value against the accurate `2 * scipy.stats.norm.sf(|z|)` at 1e-9
relative with **no absolute floor** (the grid reaches $|z| = 34.6$ and
$p = 2.1\times10^{-262}$); `stat_variance` against the closed forms; 45
Chow-Denning cases against the documented closed form over arch's per-horizon
statistics ([`variance_ratio.json`](../../../fixtures/variance_ratio.json),
[`variance_ratio_golden.rs`](../../../crates/tsecon-diag/tests/variance_ratio_golden.rs),
[`variance_ratio_properties.rs`](../../../crates/tsecon-diag/tests/variance_ratio_properties.rs),
[`test_variance_ratio.py`](../../../bindings/python/tests/test_variance_ratio.py)).
The achieved deviations, from
`scripts/variance_ratio_golden_deviations.py`:

```
tsecon 0.10.0: 250 single-horizon cases vs arch 8.0.0, 45 Chow-Denning cases
  max relative deviation  vr 0.00e+00  stat 8.49e-16 (1e-12-guarded)  stat_variance 4.11e-15
  max relative deviation  pvalue vs 2*norm.sf 4.07e-13  cd_pvalue 3.02e-13
  largest |stat| 34.60, smallest p-value 2.135e-262; arch's own p is 0.0 in 72 single-horizon cases
```

`vr` is bit-identical to arch in every case (both sum sequentially). The
Chow-Denning leg has no independently *run* reference (R is unavailable to the
generator): CRAN vrtest 1.2's `LM_stat.R` and `Chow.Denning.R` were transcribed
from source and agree with arch's `debiased=False` statistics to 3.7e-13 and
with the closed-form critical values to 1.6e-14 (recorded in the fixture's
`_meta`). vrtest applies the same Šidák bound but reports no p-value and fixes
the overlapping, non-debiased, drift-centred statistics; tsecon uses the
caller's per-horizon options and inverts the bound for the p-value. The size
and power claims are carried by
`scripts/mc_variance_ratio_size.py`,
run on the shipped function:

```
tsecon 0.10.0; n = 500, q = (2, 4, 8, 16), 600 reps per DGP, seed 12345, nominal level 5%
iid N(0,1) increments (null)
  robust z*      per-horizon q=2/4/8/16: 0.042 / 0.043 / 0.042 / 0.063   Chow-Denning: 0.028
  homoskedastic  per-horizon q=2/4/8/16: 0.045 / 0.038 / 0.043 / 0.062   Chow-Denning: 0.023
GARCH(1,1) a=0.15 b=0.80 increments (null)
  robust z*      per-horizon q=2/4/8/16: 0.042 / 0.052 / 0.037 / 0.038   Chow-Denning: 0.040
  homoskedastic  per-horizon q=2/4/8/16: 0.113 / 0.113 / 0.093 / 0.085   Chow-Denning: 0.093
AR(1) phi=0.1 increments (alternative)
  robust z*      per-horizon q=2/4/8/16: 0.582 / 0.465 / 0.317 / 0.208   Chow-Denning: 0.438
  homoskedastic  per-horizon q=2/4/8/16: 0.583 / 0.462 / 0.322 / 0.207   Chow-Denning: 0.445
```

(The Monte Carlo standard error of a rate near 5% is about 0.009 at 600
replications.) See the [validation matrix](../validation-matrix.md).

**References.** Lo & MacKinlay (1988), "Stock market prices do not follow random
walks", *Review of Financial Studies* 1(1); Chow & Denning (1993), "A simple
multiple variance ratio test", *Journal of Econometrics* 58(3); Campbell, Lo &
MacKinlay (1997), *The Econometrics of Financial Markets*, §2.4; Hochberg &
Tamhane (1987), *Multiple Comparison Procedures* (the studentized maximum
modulus); Kim (2006) and Choi (1999) for the wild-bootstrap and
automatic-horizon refinements, not shipped; vrtest 1.2 (Kim, CRAN).

```python
import numpy as np, tsecon

rng = np.random.default_rng(2026)
n = 1000

# A log price whose increments are GARCH(1,1) martingale differences:
# a random walk with volatility clustering.
e, h = np.zeros(n), np.ones(n)
for t in range(1, n):
    h[t] = 0.05 + 0.15 * e[t - 1] ** 2 + 0.80 * h[t - 1]
    e[t] = np.sqrt(h[t]) * rng.standard_normal()
walk = np.cumsum(e)

rob = tsecon.variance_ratio(walk, lags=[2, 4, 8, 16])               # robust z* (default)
hom = tsecon.variance_ratio(walk, lags=[2, 4, 8, 16], robust=False)
print("GARCH walk  vr:", np.round(rob["vr"], 3))
print("  robust p:", np.round(rob["pvalue"], 3), " Chow-Denning p:", round(rob["cd_pvalue"], 3))
print("  homosk p:", np.round(hom["pvalue"], 3), " Chow-Denning p:", round(hom["cd_pvalue"], 3))

# Momentum: increments u_t = 0.2 u_{t-1} + noise.
u = np.zeros(n)
for t in range(1, n):
    u[t] = 0.2 * u[t - 1] + rng.standard_normal()
mom = tsecon.variance_ratio(np.cumsum(u), lags=[2, 4, 8, 16])
print("momentum    vr:", np.round(mom["vr"], 3), " cd_stat:", round(mom["cd_stat"], 2),
      "at q =", mom["cd_lag"], " Chow-Denning p:", f"{mom['cd_pvalue']:.1e}")
print("  5% critical value for 4 horizons:", round(mom["cd_crit"]["5%"], 3))

# The classic misuse: passing RETURNS where the levels belong.
returns = rng.standard_normal(n)
bad = tsecon.variance_ratio(returns, lags=2)
print("returns as y vr:", round(bad["vr"], 3), " stat:", round(bad["stat"], 1))
```

```
GARCH walk  vr: [1.041 1.091 1.065 1.094]
  robust p: [0.312 0.219 0.575 0.582]  Chow-Denning p: 0.628
  homosk p: [0.2   0.124 0.486 0.501]  Chow-Denning p: 0.412
momentum    vr: [1.19  1.307 1.404 1.329]  cd_stat: 5.86 at q = 2  Chow-Denning p: 1.9e-08
  5% critical value for 4 horizons: 2.491
returns as y vr: 0.536  stat: -12.4
```

The GARCH walk is a random walk, and neither statistic rejects it on this draw —
but the homoskedastic p-values are uniformly smaller, because under volatility
clustering its variance $V$ is too small; across draws that is the
8.5–11.3% per-horizon false-rejection rate in the table above. The momentum
walk is rejected jointly ($CD = 5.86$ at $q = 2$ against a 5% critical value of
2.49 for four horizons). Returns passed as the level produce a confident, meaningless
rejection.
