# Replication — Ramey & Zubairy (2018)

Every other page in this gallery runs on synthetic data, where the truth is
known because we built it. This one runs on **the authors' own data** and aims at
a **published number**.

Ramey & Zubairy (2018) estimate US government-spending multipliers from
historical data using Ramey's military-news series as the identifying shock.
Their headline finding is that the integral multiplier is roughly **0.6–0.8** —
crucially **below one**, meaning a dollar of government spending buys less than a
dollar of output.

```sh
.venv/bin/python docs/examples/replication_ramey_zubairy.py
```

The data downloads on first use from the authors' replication archive and is
cached; nothing is vendored into this repository.

```python
from tsecon import datasets as ds
rz = ds.ramey_zubairy()        # 564 quarters, 27 series, 1875Q1-2015Q4
rz["series"]["news"]           # Ramey's military-news shock
```

---

## The result

**Full historical sample, 1890Q1–2015Q4 (n = 504)**

| h (quarters) | cumulative ΔY | cumulative ΔG | **multiplier** |
|---|---|---|---|
| 2 | 0.270 | 0.293 | 0.921 |
| 4 | 0.510 | 0.764 | **0.667** |
| 8 | 1.382 | 2.097 | **0.659** |
| 12 | 2.492 | 3.572 | **0.698** |
| 16 | 3.316 | 4.863 | **0.682** |
| 20 | 3.763 | 5.331 | **0.706** |

Across horizons 4–20 the multiplier sits in **0.66 to 0.71** — inside RZ's
published 0.6–0.8 range, and comfortably below one. The central economic claim
replicates.

**Postwar subsample, 1947Q1–2015Q4 (n = 276)**

| h | cumulative ΔY | cumulative ΔG | multiplier |
|---|---|---|---|
| 4 | 0.221 | 0.204 | 1.085 |
| 8 | 0.404 | 0.692 | 0.583 |
| 12 | 0.644 | 1.235 | 0.522 |
| 16 | −0.072 | 1.926 | −0.037 |
| 20 | −0.033 | 2.297 | −0.014 |

The postwar ratios are unstable and go negative at long horizons. That is not a
bug in the estimator — it is the same point RZ themselves make: the identifying
variation in the military-news series is overwhelmingly pre-1950, which is
precisely why their headline estimates use the long historical sample. A small
denominator makes the ratio explode; the sign flips are noise, not economics.

---

## How the multiplier is constructed

**Normalisation (Gordon-Krenn).** Real quantities are divided by CBO potential
output rather than logged:

```python
g = (ngov / pgdp) / rgdp_potcbo     # real government spending / potential
y = rgdp / rgdp_potcbo              # real GDP / potential
newsy = news / lag(pgdp * rgdp_potcbo)
```

Dividing rather than logging is what makes the estimate a **multiplier** —
dollars of output per dollar of spending — instead of an elasticity.

**The integral multiplier** is the ratio of two cumulative local projections on
the same shock:

```python
cum_y = tsecon.lp(y, newsy, horizons=20, n_lag_controls=4, cumulative=True)["irf"]
cum_g = tsecon.lp(g, newsy, horizons=20, n_lag_controls=4, cumulative=True)["irf"]
multiplier = cum_y / cum_g
```

!!! warning "A trap worth naming"
    It is tempting to reach for `lp_iv(y, g, newsy, cumulative=True)` and read
    the coefficient as the multiplier. **It is not.** `cumulative=True` cumulates
    only the *outcome*, so that coefficient is output-per-unit-of-*contemporaneous*-
    spending, which grows without bound in the horizon — in this data it climbs
    past 48 by h=20. The multiplier needs *both* sides accumulated, which is what
    the ratio above does.

---

## What this is, and is not

This reproduces RZ's headline integral multiplier using their data, their
instrument, and their normalisation. It is **not** a line-by-line port of their
Stata code: their published tables involve sample splits, lag choices, and
standard-error conventions this script does not reproduce exactly, and this page
reports point estimates without RZ's inference.

The claim being checked is the one that carries the economics — that the
multiplier is well below one, and stable across horizons in the long sample —
not bitwise equality with a published table. Where the subsample results are
ugly, they are printed as they came out.

**Citation.** Ramey, V. A. & Zubairy, S. (2018), "Government Spending Multipliers
in Good Times and in Bad: Evidence from US Historical Data," *Journal of
Political Economy* 126(2):850-901. The replication archive is distributed by the
authors; please cite the paper if you use the data.

**See also.** [`lp` / `lp_iv` model card](../reference/model-cards/local-projections.md) ·
[datasets reference](../reference/datasets.md) ·
[frontier Monte Carlo](monte-carlo-frontier.md), which measures what LP-IV does
when the instrument is weak.
