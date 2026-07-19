# Replication — the yield curve predicts recessions (Estrella-Mishkin 1998)

Of all the leading indicators macroeconomists track, the slope of the Treasury
yield curve is the most famous: when the curve **inverts** — short-term rates
rise above long-term rates — a recession has tended to follow within a year or
so. Estrella & Mishkin (1998) put a number on it with a probit, and it has held
up across the cycles since.

This reproduces the core result with `tsecon.recession_probit` on **live FRED
data**, and recovers the canonical numbers.

```sh
.venv/bin/python docs/examples/replication_yield_curve_recession.py
```

The three series download on first use (keyless, cached) via
[`tsecon.datasets`](../reference/datasets.md): `GS10` (10-year Treasury),
`TB3MS` (3-month bill), and `USREC` (the NBER recession indicator). The term
spread is `GS10 − TB3MS`.

---

## The result

A probit of the recession indicator **twelve months ahead** on the current term
spread, monthly, 1953–2026 (867 observations after the 12-month lead):

```text
Probit:  P(recession in 12m) = Φ(b0 + b1 · term_spread)

  b0 (const)   = -0.6421
  b1 (spread)  = -0.5833   (z = -9.62)
  McFadden R²  =  0.187
```

The signature finding is the **sign and significance of `b1`**: strongly
negative, `z ≈ -10`. A flatter or inverted curve raises the recession
probability. Reading it as probabilities:

| term spread | P(recession within 12 months) |
|---|---|
| +3.0 pp (steep) | **0.8%** |
| +1.0 pp | 6% |
| 0.0 pp (flat) | 26% |
| −1.0 pp (inverted) | **48%** |

A steeply upward-sloping curve implies almost no chance of recession within the
year; a one-percentage-point inversion implies close to a coin flip. This is the
"inverted yield curve" signal that appears in the financial press every cycle —
here it is estimated from scratch in a dozen lines.

---

## How it is built

```python
from tsecon import datasets as ds
import tsecon, numpy as np

gs10 = ds.fred_series("GS10")     # keyless, downloaded once and cached
tb3  = ds.fred_series("TB3MS")
rec  = ds.fred_series("USREC")
# ... align on common monthly dates, spread = gs10 - tb3 ...

lead = 12
y = recession[lead:]                              # recession at t+12
X = np.column_stack([np.ones(len(y)), spread[:-lead]])   # explicit intercept
fit = tsecon.recession_probit(y, X, link="probit")
fit["params"]        # [b0, b1]
fit["pseudo_r2"]     # McFadden
fit["probabilities"] # fitted P(recession) for every month
```

Two modelling notes the [recession model card](../reference/model-cards/recession.md)
makes: the design must carry an **explicit intercept column**, and
`recession_probit` also fits the Kauppi-Saikkonen **dynamic** probit (an
autoregressive recession index) via `dynamic=True` — a natural extension when
recession states are persistent month to month.

---

## What this is, and is not

This reproduces the *shape* of the Estrella-Mishkin result — a term-spread probit
with a strongly significant negative slope and a pseudo-R² of the same order —
on current data. It is **not** their exact vintage, sample window, or spread
definition (they examine several horizons and financial variables). The point
being replicated is the economic one: the curve's slope is a genuine, strong,
out-of-model predictor of recessions, and `recession_probit` recovers it.

The result is guarded offline in CI: `test_replication_yield_curve.py` re-runs
the estimation against a committed snapshot of the FRED panel
([`fixtures/yield_curve_recession.csv`](../../fixtures/yield_curve_recession.csv))
and asserts the coefficient stays significantly negative, so this page cannot
quietly go stale.

**Data.** Federal Reserve Bank of St. Louis (FRED), series `GS10`, `TB3MS`,
`USREC` — public data, redistributed with attribution.

**Reference.** Estrella, A. & Mishkin, F. S. (1998), "Predicting U.S. Recessions:
Financial Variables as Leading Indicators," *Review of Economics and Statistics*
80(1):45-61.

**See also.** [recession-probability model card](../reference/model-cards/recession.md) ·
[datasets reference](../reference/datasets.md) ·
[Ramey-Zubairy replication](replication-ramey-zubairy.md).
