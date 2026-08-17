# Model card — DF-GLS, Phillips-Perron, and Phillips-Ouliaris tests

`dfgls` · `phillips_perron` · `phillips_ouliaris`

Three unit-root-family tests beyond the core ADF/KPSS pair. `dfgls` attacks the
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
cointegrating relationship. All three are companions to the confirmatory
stationarity workflow, not replacements for reading ADF and KPSS together.

| Function | Null hypothesis | The analog it complements |
|----------|-----------------|---------------------------|
| `dfgls` | the series has a unit root | `adf` (GLS-detrended, near-optimal local power) |
| `phillips_perron` | the series has a unit root | `adf` (semiparametric, no lag augmentation) |
| `phillips_ouliaris` | the regressors are **not** cointegrated with `y` | Engle-Granger; `johansen` (single-equation route) |

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
