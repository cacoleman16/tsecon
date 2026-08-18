# Replication — Bai & Perron (2003)

Bai & Perron (2003, *J. Applied Econometrics* 18(1)) is the canonical paper for
dating multiple structural breaks, and `bai_perron` implements its machinery.
The natural validation target is therefore the paper's **own empirical
application**: a mean-shift model of the US ex-post real interest rate,
quarterly 1961Q1–1986Q3 (T = 103) — the Garcia-Perron (1996) series.

The published answer: the level of the real rate breaks at **1972:3** and
**1980:3** (the partition BP's BIC and LWZ criteria select), with the paper's
own HAC-robust sequential procedure adding a third, less sharply dated break at
**1966:4**. The two-break segment means are **1.36, −1.80, 5.64**: mildly
positive real rates through 1972, *negative* real rates through the 1970s, and
sharply positive rates after the Volcker disinflation. The 1980:3 break is
dated to within a few quarters; the earlier breaks are not.

```sh
.venv/bin/python docs/examples/replication_bai_perron_realint.py
```

The data ships as `RealInt` in R's strucchange package (byte-identical to
`real` in Perron's own mbreaks package) and is committed to the repository at
[`fixtures/realint_bai_perron.csv`](../../fixtures/realint_bai_perron.csv)
with attribution, so this runs fully offline — tsecon ships no data loaders.
The whole analysis is one call on an intercept-only design (mean shift = the
constant is the only switching coefficient):

```python
bp = tsecon.bai_perron(rate, np.ones((len(rate), 1)), max_breaks=5, trim=0.15)
```

---

## The result

**Break dates — exact match at every partition size**

| partition | published (BP 2003) | tsecon (dynamic program) |
|---|---|---|
| m = 1 | 1980:3 | 1980Q3 |
| m = 2 | 1972:3, 1980:3 | **1972Q3, 1980Q3** |
| m = 3 | 1966:4, 1972:3, 1980:3 | **1966Q4, 1972Q3, 1980Q3** |

The global SSR path (`ssr_path` = 1214.92, 645.00, 455.95, 445.18) matches R
strucchange's reported RSS for the same partitions to every printed digit.

**How many breaks — same procedure, one honest difference**

| criterion | published count | tsecon |
|---|---|---|
| sequential supF(l+1\|l), 5%, **HAC-robust F** (the paper's headline) | **3** | not implemented (classical F only) |
| sequential supF(l+1\|l), 5%, **classical F** | — | **2** |
| BIC, LWZ (BP's information criteria) | **2** | not implemented |

`bai_perron` runs exactly the paper's sequential procedure at the published 5%
critical values (8.58, 10.13, 11.14 for q = 1, trim 0.15 — reproduced
verbatim), but computes **classical** supF statistics, where the paper's
published statistics are HAC-robust. The two sequences on this data:

| | supF(1\|0) | supF(2\|1) | supF(3\|2) | supF(4\|3) |
|---|---|---|---|---|
| paper (HAC, prewhitened) | 57.91 | 33.93 | **14.73** | 0.03 |
| tsecon (classical) | 89.24 | 52.20 | **7.41** | 0.04 |
| 5% critical value | 8.58 | 10.13 | 11.14 | 11.83 |

Both sequences scream "at least two breaks". They part company at supF(3|2):
the HAC statistic clears the bar and BP's sequential procedure selects three
breaks, while the classical statistic does not, so **tsecon stops at two — the
same count BP's own BIC and LWZ select** (BP note the criteria are biased
downward under serial correlation, which is why they prefer three). tsecon's
classical sequence is not a private convention: Perron's mbreaks package with
`robust=0` produces 89.245, 52.204, 7.414 — matching tsecon to the third
decimal — and R strucchange, also classical, selects the same two-break model
by BIC. The third break tsecon would add is available and exact:
`break_dates_by_m[2]` is 1966Q4, 1972Q3, 1980Q3.

**Segment means — match to published rounding**

| regime | published mean (HAC se) | tsecon mean (classical OLS se) |
|---|---|---|
| 1961Q1–1972Q3 | 1.36 (0.16) | **1.3550** (0.188) |
| 1972Q4–1980Q3 | −1.80 (0.51) | **−1.7961** (0.452) |
| 1980Q4–1986Q3 | 5.64 (0.60) | **5.6429** (0.566) |

The point estimates are segment averages and agree with the published values
to their printed rounding (and with strucchange's coefficients to 1e-6). The
standard errors are *different constructions* — per-regime classical OLS here,
HAC in the paper — and are reported side by side, not compared. For the
three-break partition the segment means 1.8236, 0.8661, −1.7961, 5.6429
likewise match the published 1.82, 0.87, −1.80, 5.64.

**Break-date confidence intervals — different estimator, stated plainly**

| break | tsecon 90% | tsecon 95% | published 95% (BP 2003) |
|---|---|---|---|
| 1972Q3 | 1971Q2–1973Q4 | 1970Q4–1974Q2 | 1970:3–1972:4 *(3-break model, heterogeneity-robust HAC)* |
| 1980Q3 | 1980Q1–1981Q1 | 1980Q1–1981Q1 | *(no independently verified published value)* |

These columns are **not the same estimator**, on three counts, all documented
on the [structural-breaks model card](../reference/model-cards/structural-breaks.md):
tsecon ships the Bai (1997) *homogeneous* case with classical variance, the
paper's CIs are the heterogeneity-robust variant with HAC variance; the
paper's CIs condition on the three-break partition, tsecon reports CIs for its
selected two-break model; and this object is famously fragile — the JAE
replication study (Zeileis & Kleiber 2005) reports that some of BP's published
CIs could not be reproduced in strucchange under any settings, with the
discrepancy attributed to numerical problems in BP's original GAUSS code.
Every implementation of this CI disagrees at the level of a few quarters on
this series (strucchange's homogeneous 95% for 1972Q3: 1971Q2–1973Q4; mbreaks
with the paper's settings: 1970Q1–1973Q2).

What *does* replicate — and it is the paper's substantive point about the CIs —
is the asymmetry of precision: the 1980:3 Volcker break is pinned to a
5-quarter 95% window while the 1972:3 break needs 15 quarters, in every
implementation including this one.

---

## Cross-implementation validation

Because the published table mixes HAC statistics tsecon deliberately does not
compute, the replication was additionally cross-checked, number by number,
against the two maintained R implementations run on the identical committed
series:

- **strucchange** (`breakpoints(RealInt ~ 1, h = 15)`, classical, the
  implementation validated against BP in the JAE replication section): same
  partitions for m = 1..3, same RSS path to every printed digit, same
  coefficients to 1e-6, same BIC-selected count (2).
- **mbreaks** (Perron's own R port, `mdl('rate', eps1 = 0.15)`): with the
  paper's HAC settings it reproduces the published table (supF 57.91/33.93/
  14.73, sequential → 3, KT/BIC/LWZ → 2, means 1.355/−1.796/5.643); with
  `robust = 0` its classical supF sequence matches tsecon's to the third
  decimal.

One geometry detail for anyone re-running this: with `trim = 0.15` tsecon uses
minimum segment length `h = ceil(0.15 · 103) = 16` where the R packages use
`h = 15`. This changes nothing through m = 3 (identical partitions and SSRs);
at m = 4 and 5 — beyond anything selected — the admissible sets differ and so
do the optimal partitions.

## What this is, and is not

This reproduces Bai & Perron's break *dates* exactly at every partition size
the paper reports, their segment means to published rounding, their published
critical values verbatim, and their information-criteria break count via the
same sequential procedure the paper uses — with the variance-estimator caveat
above. It does **not** reproduce the paper's HAC-robust test statistics or its
heterogeneity-robust break-date CIs, because tsecon does not implement those
variants; where a published number is not comparable, the tables above say so
rather than stretching a match.

**Citation.** Bai, J. & Perron, P. (2003), "Computation and Analysis of
Multiple Structural Change Models," *Journal of Applied Econometrics*
18(1):1–22. Data: Garcia, R. & Perron, P. (1996), *REStat* 78:111–125, as
distributed in R strucchange (Zeileis et al.) and mbreaks (Nguyen, Perron &
Yamamoto). Validation study: Zeileis, A. & Kleiber, C. (2005), "Validating
Multiple Structural Change Models — A Case Study," *J. Applied Econometrics*
20:685–690.

**See also.** [`bai_perron` / `sup_f_test` model card](../reference/model-cards/structural-breaks.md) ·
[Ramey-Zubairy replication](replication-ramey-zubairy.md) ·
[yield-curve recession replication](replication-yield-curve-recession.md)
