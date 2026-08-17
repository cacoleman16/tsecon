# Model card — DF-GLS, Phillips-Perron, Phillips-Ouliaris, and Zivot-Andrews tests

`dfgls` · `phillips_perron` · `phillips_ouliaris` · `zivot_andrews`

Four unit-root-family tests beyond the core ADF/KPSS pair. `dfgls` attacks the
ADF's power problem: estimating a constant or trend by OLS costs the plain ADF
real power near the unit-root boundary, and GLS-detrending at the ERS local
alternative recovers most of it — it is the recommended default over plain ADF
whenever deterministics must be estimated. `phillips_perron` and
`phillips_ouliaris` are semiparametric: instead of adding lagged differences to
soak up serial correlation (the augmentation in ADF and Engle-Granger), they
estimate a simple regression by OLS and then **correct the test statistic** for
the residual's long-run variance with a nonparametric (Bartlett) kernel.
`phillips_perron` is the unit-root test — a drop-in alternative to
[`adf`](diagnostics.md); `phillips_ouliaris` is its cointegration analog — a
residual-based alternative to [`johansen`](cointegration-regime.md) for a single
cointegrating relationship. `zivot_andrews` guards the workflow against a
one-time structural break masquerading as a unit root. All four are companions
to the confirmatory stationarity workflow, not replacements for reading ADF and
KPSS together.

| Function | Null hypothesis | The analog it complements |
|----------|-----------------|---------------------------|
| `dfgls` | the series has a unit root | `adf` (GLS-detrended, near-optimal local power) |
| `phillips_perron` | the series has a unit root | `adf` (semiparametric, no lag augmentation) |
| `phillips_ouliaris` | the regressors are **not** cointegrated with `y` | Engle-Granger; `johansen` (single-equation route) |
| `zivot_andrews` | a unit root with **no** break | `adf` when a one-time structural break may masquerade as a root |

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
MAIC lag rule is the standard remedy; until the Ng-Perron tests land, be
generous with `max_lags` in that case).

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
to pick too few lags there).

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
constant *and* a trend. `trim` (default 0.15, in `[0, 1/3]`): excludes the first
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
order (the break dummy needs a pre-break regime — the error explains); treating
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
