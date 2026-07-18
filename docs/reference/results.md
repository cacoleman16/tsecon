# Results objects

Every tsecon estimator returns a plain `dict` of documented keys. That is a
deliberate design choice — dicts are transparent, serialisable, and impossible to
get locked out of.

`tsecon.results` adds an **opt-in** layer on top: objects that carry the same
data but can also *render* themselves — a `.summary()` an economist can read, and
`.plot_*()` methods for the standard figures.

The key property:

!!! note "A results object **is** a `dict`"
    Every class here subclasses `dict`. `res["params"]`, `isinstance(res, dict)`,
    `json`, `pickle`, and `**res` unpacking all keep working exactly as before —
    a results object only *adds* methods. Adopting them is additive, never a
    breaking change, and you can always drop back to plain data with
    `res.to_dict()`.

```python
import tsecon
from tsecon.results import VARResults

# the compiled function — unchanged, returns a plain dict
raw = tsecon.var_fit(data, lags=2)

# the same estimation, wrapped
fit = VARResults.fit(data, lags=2, names=["gdp", "infl", "rate"])
fit["aic"]              # still a dict
set(fit) == set(raw)    # True — identical keys
print(fit.summary())    # ...that can also render itself
fit.irf(horizon=12).plot()
```

The compiled functions are **not** shadowed: `tsecon.var_fit` remains the
compiled builtin returning a plain dict. `tsecon.results` is a namespace you
reach into deliberately.

## Plotting is optional

matplotlib is an optional dependency. Nothing imports it until you call a plot
method, and if it is missing the error tells you what to install:

```sh
pip install 'tsecon[plots]'
```

---

## What's available

| Class | Wraps | Highlights |
|---|---|---|
| [`VARResults`](#varresults) | `var_fit` | coefficient matrix, stability verdict, `.irf()` |
| `IRFArray` | `var_irf` | a `list` subclass — `.response(i, j)`, `.plot()` grid |
| `LPResults` | `lp` | per-horizon table, `.conf_int()`, `.peak()`, `.plot_irf()` |
| `GARCHResults` | `garch_fit` | robust SEs, `.persistence()`, `.plot_volatility()` |
| `ARIMAResults` | `arima_fit` | `.forecast_frame()`, fan-chart `.plot_forecast()` |
| [`PredictiveRegressionResults`](#predictiveregressionresults) | `predictive_regression` | three estimators side by side |
| `IVXTestResults` | `ivx_test` | joint IVX test |
| [`DSGEResults`](#dsgeresults) | `dsge_solve` | Blanchard-Kahn verdict, `.impulse_response()` |

---

## `VARResults`

```text
====================================================================
VAR(2) — 2 equations, trend='c' — stable
====================================================================
llf -339.646    aic -2.7376    bic -2.5917    hqic -2.6788
reciprocal roots — min 1.6438    max 11.9838     (stable iff min > 1)
--------------------------------------------------------------------
coefficients — rows = regressors, cols = equations
regressor              gdp          infl
--------------------------------------------------------------------
const             +0.00096      +0.04627
L1.gdp            +0.52077      +0.06563
L1.infl           +0.04191      +0.41989
L2.gdp            -0.00972      +0.06450
L2.infl           +0.03986      +0.00631
====================================================================
```

The stability line takes its verdict from `is_stable` and states the convention
explicitly — these are *reciprocal* characteristic roots, so stability means
`min > 1`. `max_root` alone is not a verdict (an explosive system can still have
`max_root > 1`); see the [VAR/SVAR card](model-cards/var-svar.md).

`IRFArray` subclasses `list`, so `irf[h][i][j]`, `len()`, slicing and
`np.array(irf)` behave exactly as the bare nested list always did — it just gains
`.response("gdp", "infl")`, a readable `repr`, and `.plot()` for the k×k
small-multiple grid.

## `PredictiveRegressionResults`

The most useful summary in the library: one regression, three estimators, and an
explicit statement of which p-value to trust.

```text
====================================================================
Predictive regression   r(t+1) = a + b*x(t)           IVX p = 0.4083
====================================================================
nobs 399    rho(x) 0.9713    IVX rz 0.9966
intercept +0.21606    Stambaugh bias removed +0.00902
--------------------------------------------------------------------
estimator           beta     std err      test stat    p-value
--------------------------------------------------------------------
OLS             +0.01867     0.01112       t +1.680     0.0930
Stambaugh       +0.00966     0.01112       t +0.869     0.3851
IVX             +0.01415     0.01711       W 0.6838     0.4083
--------------------------------------------------------------------
Report the IVX Wald p-value (0.4083). It is valid whatever the
persistence of x; the OLS t over-rejects when x is persistent,
so the OLS and Stambaugh p-values above are naive normal ones.
WARNING: rho(x) = 0.9713 > 0.95, so x is highly persistent
         and the OLS t-statistic is unreliable here.
At the 5% level IVX does not reject b = 0: no evidence
that x predicts r(t+1).
====================================================================
```

`.significant(level=0.05)` reads the **IVX** p-value, not the OLS t — the
[Monte Carlo suite](../examples/monte-carlo.md) shows why: at a unit root the OLS
t-test rejects a true null 28% of the time.

## `DSGEResults`

```text
====================================================================
Linear RE model (Blanchard-Kahn): unique stable solution
====================================================================
predetermined 1    jump 1    shocks 1    determinate yes
verdict: unique stable solution (1 unstable eigenvalue(s) = 1 jump
         variable(s))
--------------------------------------------------------------------
eigenvalue moduli:  stable (<1) 1    unstable (>1) 1
  stable         0.60000
  unstable       1.42857
--------------------------------------------------------------------
G  policy: jump = G . predetermined                            [1x1]
     +1.72414
P  transition: k(t+1) = P . k(t) + Q . z                       [1x1]
     +0.60000
Q  impact: shock loading on the state                          [1x1]
     +1.00000
====================================================================
```

`.impulse_response(horizon=24)` traces the saddle path by iterating
`k_{t+1} = P·k_t` from impact `Q·shock` and reading jumps off `x_t = G·k_t` — the
compiled binding returns matrices only, so this accessor genuinely adds
capability rather than just formatting.

---

## Design note: why a `dict` subclass

The alternative — a bespoke `Results` class with `.params` attributes — would
have made the plain-dict contract a legacy path the moment it shipped, forcing
either a breaking change or two parallel APIs forever.

Subclassing `dict` sidesteps that entirely: the dict contract is preserved
*as a subset* of the richer object, so the library can add rendering
incrementally, per family, without ever invalidating code that treats results as
data. It also keeps results trivially serialisable, which matters for Monte Carlo
work where results are written to disk by the thousand.

This layer earned its keep before it shipped: writing a correct stability line
for `VARResults.summary()` is what exposed that `var_fit` was reporting the wrong
characteristic root — a bug that had propagated into three documentation pages.
