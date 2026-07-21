# Replication — the yield curve predicts recessions (Estrella-Mishkin 1998)

Of all the leading indicators macroeconomists track, the slope of the Treasury
yield curve is the most famous: when the curve **inverts** — short-term rates
rise above long-term rates — a recession has tended to follow within a year or
so. Estrella & Mishkin (1998) put a number on it with a probit, and it has held
up across the cycles since.

This reproduces the core result with `tsecon.recession_probit`, estimated from a
**committed FRED snapshot** (retrieved 2026-07-18) — it runs fully offline — and
recovers the canonical numbers.

```sh
.venv/bin/python docs/examples/replication_yield_curve_recession.py
```

The three FRED series — `GS10` (10-year Treasury), `TB3MS` (3-month bill), and
`USREC` (the NBER recession indicator) — are aligned into one monthly panel
committed at
[`fixtures/yield_curve_recession.csv`](../../fixtures/yield_curve_recession.csv),
so this runs fully offline (tsecon ships no data loaders). The term spread is
`GS10 − TB3MS`.

---

## The result

A probit of the recession indicator **twelve months ahead** on the current term
spread, monthly, 1953–2026 (867 observations after the 12-month lead). The
estimand is the **12-month-ahead probability** — the chance the economy is in
recession *in month t+12*, not at some point during the intervening year:

```text
Probit:  P(recession at t+12) = Φ(b0 + b1 · term_spread)

  b0 (const)   = -0.6421
  b1 (spread)  = -0.5833   (z = -9.62)
  McFadden R²  =  0.187
```

The signature finding is the **sign and significance of `b1`**: strongly
negative, `z ≈ -10`. One honest caveat on that z-statistic: these are i.i.d.
maximum-likelihood standard errors, and recession months are strongly serially
dependent, so `z ≈ -10` overstates the precision — the sign and economic
magnitude are the robust part. The shipped route to modelling that dependence
is the Kauppi-Saikkonen **dynamic** probit (`dynamic=True`, below). A flatter
or inverted curve raises the recession probability. Reading it as probabilities:

| term spread | P(recession in month t+12) |
|---|---|
| +3.0 pp (steep) | **0.8%** |
| +1.0 pp | 11% |
| 0.0 pp (flat) | 26% |
| −1.0 pp (inverted) | **48%** |

A steeply upward-sloping curve implies a near-zero probability of being in
recession a year later; a one-percentage-point inversion implies close to a
coin flip. This is the "inverted yield curve" signal that appears in the
financial press every cycle — here it is estimated from scratch in a dozen
lines.

---

## How it is built

```python
import csv, numpy as np, tsecon

# read the committed monthly panel: date, gs10, tb3ms, usrec
rows = [r for r in csv.reader(open("fixtures/yield_curve_recession.csv"))
        if r and not r[0].startswith("#")][1:]
spread    = np.array([float(r[1]) - float(r[2]) for r in rows])   # GS10 - TB3MS
recession = np.array([float(r[3]) for r in rows])

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
[Ramey-Zubairy replication](replication-ramey-zubairy.md).
