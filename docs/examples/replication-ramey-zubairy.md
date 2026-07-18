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

| h (quarters) | cumulative ΔY | cumulative ΔG | **multiplier** | SE | first-stage F |
|---|---|---|---|---|---|
| 2 | 0.280 | 0.382 | 0.735 | 0.115 | 10.9 |
| 4 | 0.647 | 1.018 | **0.635** | 0.057 | 15.2 |
| 8 | 1.813 | 2.758 | **0.657** | 0.039 | 16.8 |
| 12 | 3.250 | 4.641 | **0.700** | 0.033 | 13.5 |
| 16 | 4.219 | 5.973 | **0.706** | 0.048 | 11.7 |
| 20 | 4.767 | 6.413 | **0.743** | 0.057 | 12.0 |

Across horizons 4–20 the multiplier sits in **0.64 to 0.74** — inside RZ's
published 0.6–0.8 range, and comfortably below one at every horizon (the
95% band at h = 8 is 0.58 to 0.73). The central economic claim replicates.

**Postwar subsample, 1947Q1–2015Q4 (n = 276)**

| h | cumulative ΔY | cumulative ΔG | multiplier | SE | first-stage F |
|---|---|---|---|---|---|
| 2 | 0.085 | 0.082 | 1.032 | 0.377 | 17.0 |
| 4 | 0.232 | 0.297 | 0.783 | 0.184 | 56.2 |
| 8 | 0.509 | 0.856 | 0.594 | 0.129 | 112.9 |
| 12 | 0.841 | 1.419 | 0.592 | 0.122 | 117.2 |
| 16 | 0.943 | 1.789 | 0.527 | 0.125 | 65.6 |
| 20 | 1.288 | 2.077 | 0.620 | 0.140 | 39.6 |

The postwar point estimates land lower, and their standard errors are roughly
three times the full-sample ones — at h = 2 the multiplier is not
distinguishable from anything between 0.3 and 1.8. That is the same point RZ
themselves make: the identifying variation in the military-news series is
overwhelmingly pre-1950, which is precisely why their headline estimates use the
long historical sample.

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

**The integral multiplier** is one call:

```python
r = tsecon.lp_multiplier(y, g, newsy, horizons=20, n_lag_controls=4)
r["multiplier"]      # extra dollar of output per extra dollar of spending
r["se"]              # standard error OF THE MULTIPLIER
r["first_stage_f"]   # weak-instrument diagnostic
```

`lp_multiplier` runs the one-step Ramey-Zubairy regression at each horizon:

```text
sum_{j=0..h} y_{t+j} = m_h * sum_{j=0..h} g_{t+j} + c
                     + sum_{l=1..p} (phi_l y_{t-l} + psi_l g_{t-l}) + u
```

with the cumulated spending instrumented by the contemporaneous news shock.
Because the multiplier is a single 2SLS coefficient rather than a ratio of two
separately estimated responses, `se` is the standard error of the number being
reported — not a delta-method approximation and not one leg's SE relabelled.
The two reduced-form legs are returned as `cumulative_outcome` and
`cumulative_impulse` for transparency; by the just-identified IV algebra their
ratio equals `multiplier` to numerical precision, which is exactly the identity
the tables above display.

!!! warning "A trap worth naming"
    It is tempting to reach for `lp_iv(y, g, newsy, cumulative=True)` and read
    the coefficient as the multiplier. **It is not.** `cumulative=True` (also
    spelled `cumulative="outcome"`) cumulates only the *outcome*, so the
    coefficient is output per unit of ***contemporaneous*** spending. Its
    numerator accumulates and its denominator does not, so it inherits the
    growth of the spending path rather than measuring anything per-dollar: on
    this data it runs 7.4 at h = 4 and **48.7** by h = 20, with a first-stage F
    of 1.68 into the bargain.

    That flag still exists and still means what it always meant — a cumulative
    *impulse response* is a perfectly good object. It is simply not a
    multiplier. A multiplier needs both sides accumulated, which is why the
    library now gives it its own entry point rather than a flag you have to
    know to set correctly. If you want the both-sides cumulation without the
    multiplier's control set you can also ask for it directly:
    `lp_iv(..., cumulative="both")`.

    *(Earlier versions of this page worked around the gap by running two
    cumulative `lp` calls and dividing. That ratio used a different control set
    in each leg — lags of `y` in the numerator, lags of `g` in the denominator —
    which is not any single well-defined IV estimator. The one-step version
    moves the h = 4..20 estimates from 0.66–0.71 to 0.64–0.74; the largest
    change is +0.037 at h = 20 and the economic conclusion is unchanged.)*

---

## What this is, and is not

This reproduces RZ's headline integral multiplier using their data, their
instrument, and their normalisation. It is **not** a line-by-line port of their
Stata code: their published tables involve sample splits, lag choices, and
standard-error conventions this script does not reproduce exactly. The standard
errors reported here are the library's kernel-HAC ones for the one-step
multiplier regression, not RZ's own convention.

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
