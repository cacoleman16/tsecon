# Replication — Hansen (1999): SETAR on the sunspot numbers

The SETAR model was *invented for this series*. Tong & Lim (1980, JRSS-B)
introduced threshold autoregression with the annual Wolf sunspot numbers as
the headline application, and the series has been the canonical SETAR test bed
ever since. This page replicates the two-regime sunspot SETAR that
`tsecon.setar` can express **exactly**: Hansen (1999, "Testing for linearity",
*Journal of Economic Surveys* 13(5):551–576), who refits the series with a
**common AR order p = 11 in both regimes**, one threshold, one delay — and
reports a threshold of **7.4** (on the square-root scale), a delay of **2**,
and a bootstrap rejection of linearity with **p ≈ 0.03**.

```sh
.venv/bin/python docs/examples/replication_setar_sunspots.py
```

The data is the annual Wolf (Zürich) sunspot series, 1700–1988 — the sample of
Tong (1990, Appendix 3) that Hansen used — committed at
[`fixtures/sunspots_tong.csv`](../../fixtures/sunspots_tong.csv) (public
observational data, vendored from `statsmodels.datasets.sunspots` with
attribution), so this runs fully offline — tsecon ships no data loaders.

---

## The result

| quantity | Hansen (1999) | tsecon | verdict |
|---|---|---|---|
| sample | 1700–1988 annual means | same (289 years; 278 usable after 11 lags) | exact |
| transform | y = 2(√(1+N) − 1) | same | exact |
| AR order p, both regimes | 11 (with intercepts) | 11 | exact |
| trimming | ≥ 10% of observations per regime | 10% (identical fit at 15%: the split is interior) | match |
| delay d | **2** | **2** (searched over {1, 2}; also wins over 1..11) | exact |
| threshold γ̂ (transformed) | **7.4** | **7.4234** | match at published precision |
| threshold (raw counts) | ≈ 21 | 21.2 | match |
| linearity: bootstrap p | **≈ 0.03** | 0.022–0.032 across seeds (0.025 at the pinned seed) | match |

Two of these rows deserve their honesty spelled out:

**The threshold row.** A SETAR threshold is identified only up to the gap
between adjacent order statistics of the threshold variable — every γ in
[7.4234, 7.4446) (raw counts [21.2, 21.3)) produces the *identical* fitted
model, and implementations legitimately differ in which endpoint of that flat
interval they print. Hansen's published two-digit 7.4 lies in exactly this
gap, so "rounds to 7.4 and the gap is [21.2, 21.3)" is the strongest claim a
replication can honestly pin — and it is what the regression test pins,
alongside a 1e-10 pin on tsecon's own 7.4234 so the fit cannot drift while
the round-to-published check still passes.

**The p-value row.** tsecon's `setar_test` implements the homoskedastic
residual bootstrap for Hansen's F12 = n(S₀ − S₁)/S₁ (here **69.75** at d = 2,
with S₀ = 1134.85 from the AR(11) and S₁ = 907.24 from the SETAR; at the
rejected delay d = 1 the statistic drops to 40.09 and does not reject). The
paper reports p ≈ 0.03 for the sunspot series and also computes
heteroskedasticity-robust variants that moderate the evidence; the pinned
claim is therefore the **verdict** — rejection at 5% — with the seeded
p-value landing at 0.02–0.03, not bit-equality with any one of the paper's
bootstrap schemes.

---

## The exact spec, in tsecon

```python
import numpy as np, csv

rows = [r for r in csv.reader(open("fixtures/sunspots_tong.csv"))
        if r and not r[0].startswith("#")][1:]
counts = np.array([float(r[1]) for r in rows])       # raw Wolf numbers
y = 2.0 * (np.sqrt(1.0 + counts) - 1.0)              # Ghaddar-Tong (1981)

r = tsecon.setar(y, p=11, delays=[1, 2], trim=0.10)  # Hansen's search space
r["delay"], r["threshold"]                           # 2, 7.4234
t = tsecon.setar_test(y, p=11, delay=2, trim=0.10, n_boot=199, seed=7)
t["stat"], t["p_value"]                              # 69.75, 0.025
```

`delays=[1, 2]` reproduces Hansen's joint search over the delay (tsDyn's
replication of the same example searches `thDelay = 0:1`, which is d ∈ {1, 2}
in the literature's convention — tsDyn counts the delay from the first lag).
The choice is not load-bearing: an unrestricted search over d = 1..11 also
lands on d = 2, and the test suite checks that too.

The script also prints tsecon's per-regime coefficients, standard errors,
regime split (86 low / 192 high, 31%/69%) and per-regime variances
(6.12 / 2.52 — the low regime is genuinely noisier), for side-by-side reading
against the paper's Table 2. Those are **printed, not pinned**: the pinned
quantities are the ones whose published values this page states above.

An information-criteria footnote, because it is honest and instructive: under
the shared `n·ln(SSR/n) + penalty` convention, **AIC prefers the SETAR(2)**
(378.8 vs 415.0) while **BIC prefers the AR(11)** (458.6 vs 469.5) — the
threshold model carries 25 parameters and BIC charges for them. tsDyn's
executed replication of the same example orders the criteria the same way.
The reason the SETAR still wins the argument is the *test*, not the IC: the
F12 bootstrap rejects linearity, which is precisely why Hansen frames the
problem as testing rather than criterion-shopping.

---

## What this is, and is not

This reproduces the sunspot SETAR(2) quantities Hansen (1999) reports —
threshold, delay, and the linearity verdict — on the same data, transform,
order, and trimming. It is a **replication of the body-text estimates, not a
bitwise port of the paper's tables**: Table 2's per-regime least-squares
coefficients are printed for comparison but not pinned, and the paper's
several bootstrap/asymptotic p-value variants are represented here by the one
scheme tsecon implements.

The deeper honesty note is about *which* published sunspot SETAR is being
replicated. The most famous ones — Tong & Lim's (1980) SETAR(2; 3, 11) with
d = 8, and Ghaddar & Tong's (1981) SETAR(2; 4, 12) — use **regime-specific AR
orders**. `tsecon.setar` deliberately fits a **common order p** in both
regimes (the Hansen 1997/1999 design its estimator and test are built on), so
those models are expressible only as constrained versions of an order-11/12
fit — the constrained-vs-unconstrained relationship the literature itself
notes. Rather than fudge a comparison across that gap, this page targets the
published fit whose specification the shipped estimator expresses exactly and
says so. The transcription golden (`fixtures/setar.json`) already pins the
algorithm at 1e-10; what this page adds is the anchor to a published fit of
real data.

**Citation.** Hansen, B. E. (1999), "Testing for Linearity," *Journal of
Economic Surveys* 13(5):551–576. Tong, H. & Lim, K. S. (1980), "Threshold
Autoregression, Limit Cycles and Cyclical Data," *JRSS-B* 42(3):245–292.
Ghaddar, D. K. & Tong, H. (1981), "Data Transformation and Self-Exciting
Threshold Autoregression," *JRSS-C* 30(3). Tong, H. (1990),
*Non-linear Time Series: A Dynamical System Approach*, OUP. Sunspot data:
public-domain Wolf/Zürich annual means (NOAA NGDC; the series is maintained
today by SILSO, Royal Observatory of Belgium), vendored via statsmodels.

**See also.** [`setar` model card](../reference/model-cards/cointegration-regime.md) ·
[Ramey-Zubairy replication](replication-ramey-zubairy.md) ·
[yield-curve recession replication](replication-yield-curve-recession.md)
