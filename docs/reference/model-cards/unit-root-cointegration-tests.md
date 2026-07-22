# Model card — Phillips-Perron and Phillips-Ouliaris tests

`phillips_perron` · `phillips_ouliaris`

Two semiparametric tests built on the same idea: instead of adding lagged
differences to soak up serial correlation (the augmentation in ADF and
Engle-Granger), estimate a simple regression by OLS and then **correct the test
statistic** for the residual's long-run variance with a nonparametric (Bartlett)
kernel. `phillips_perron` is the unit-root test — a drop-in alternative to
[`adf`](diagnostics.md); `phillips_ouliaris` is its cointegration analog — a
residual-based alternative to [`johansen`](cointegration-regime.md) for a single
cointegrating relationship. Both are companions to the confirmatory stationarity
workflow, not replacements for reading ADF and KPSS together.

| Function | Null hypothesis | The analog it complements |
|----------|-----------------|---------------------------|
| `phillips_perron` | the series has a unit root | `adf` (semiparametric, no lag augmentation) |
| `phillips_ouliaris` | the regressors are **not** cointegrated with `y` | Engle-Granger; `johansen` (single-equation route) |

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
