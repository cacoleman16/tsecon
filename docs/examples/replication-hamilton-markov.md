# Replication — Hamilton (1989)

The founding paper of regime-switching econometrics, run on **the author's own
data** against **published numbers**. Hamilton (1989) models quarterly US real
GNP growth as an AR(4) whose *mean* jumps between two hidden states — expansion
and contraction — governed by a first-order Markov chain, and shows that the
model re-derives the NBER business-cycle chronology from GNP growth alone.

The published headline (Econometrica 57(2), Table I, p. 372): expansions grow
at about **+1.16%** per quarter, contractions shrink at about **−0.36%**, the
expansion state persists with probability **p = 0.9049** and the contraction
state with **q = 0.7550** — expected durations of roughly ten and four
quarters — and quarters where the smoothed contraction probability exceeds ½
line up with the NBER recession dates the model was never shown.

```sh
.venv/bin/python docs/examples/replication_hamilton_markov.py
```

The data is Hamilton's own series — `100 * diff(log(real GNP))`, seasonally
adjusted, 1951Q2–1984Q4, estimation sample 1952Q2–1984Q4 (n = 131) after four
lags of conditioning — committed at
[`fixtures/hamilton_gnp.csv`](../../fixtures/hamilton_gnp.csv) (US-government
statistical data, public domain; copied verbatim from the statsmodels
regime-switching test suite, which vendors it). Everything runs offline —
tsecon ships no data loaders.

This page is a **dual golden**: tsecon versus the published table, and tsecon
versus statsmodels `MarkovAutoregression(k_regimes=2, order=4,
switching_ar=False)` fitted to the identical committed series.

---

## The result

| parameter | Hamilton (1989) | tsecon | statsmodels |
|---|---|---|---|
| expansion mean μ₁ | **1.16** | 1.1654 | 1.1635 |
| contraction mean μ₀ | **−0.36** | −0.3435 | −0.3588 |
| P(stay in expansion) p | **0.9049** | 0.9022 | 0.9041 |
| P(stay in contraction) q | **0.7550** | 0.7629 | 0.7547 |
| σ² (common) | 0.5914 | 0.5940 | 0.5914 |
| E[expansion length] | ~10 qtrs | 10.2 | 10.4 |
| E[contraction length] | ~4 qtrs | 4.2 | 4.1 |
| log-likelihood | — | −181.269 | −181.263 |
| common AR φ₁…φ₄ | 0.014, −0.058, −0.247, −0.213 | 0.0147, −0.0532, −0.2459, −0.2120 | 0.0135, −0.0575, −0.2470, −0.2129 |

Published values: p and q are Table I's printed digits (verified against the
paper's Table I, p. 372); the means are the table's two-decimal headline;
σ² and the AR row are quoted at the E-views/statsmodels re-estimation
precision (Hamilton prints σ = 0.769, i.e. σ² ≈ 0.591). The statsmodels
column doubles as an independent check that the *data* is right: its MLE on
this fixture matches the E-views `SWITCHREG` benchmark stored in statsmodels'
own test suite to 1e-4 and its log-likelihood to −181.26339.

**The NBER dating.** Classifying each quarter by smoothed P(recession) > ½
matches the NBER indicator in **120 of 131 quarters (91.6%)**, and every one
of the seven NBER recessions in the sample carries a peak smoothed recession
probability of at least 0.94:

| NBER recession | peak P(recession), tsecon | statsmodels |
|---|---|---|
| 1953Q3–1954Q2 | 0.994 | 0.994 |
| 1957Q4–1958Q1 | 0.995 | 0.995 |
| 1960Q2–1961Q1 | 0.936 | 0.936 |
| 1970Q1–1970Q4 | 0.974 | 0.972 |
| 1974Q1–1975Q1 | 0.999 | 0.999 |
| 1980Q1–1980Q2 | 0.995 | 0.995 |
| 1981Q3–1982Q4 | 0.999 | 0.999 |

The eleven disagreeing quarters all sit at episode edges — the model calls
1957, 1969 and 1979 downturns one-to-three quarters before the NBER indicator
does — the same lead/lag pattern Hamilton discusses. tsecon and statsmodels
classify **every quarter identically**, and their smoothed recession
probability paths agree to a maximum absolute difference of 0.033
(correlation 0.9998).

---

## Matching the specification

Hamilton's model is a *mean*-switching AR — the autoregression applies to
deviations from the regime mean, so four lagged regimes enter the likelihood
alongside the current one:

```text
y_t − μ_{S_t} = Σ_{l=1..4} φ_l (y_{t−l} − μ_{S_{t−l}}) + ε_t,  ε_t ~ N(0, σ²)
```

with a common AR and a **single** variance. That is exactly what

```python
ms = tsecon.markov_switching_ar(y, k_regimes=2, order=4, switching_variance=False)
```

estimates: the crate implements the Hamilton mean convention over the expanded
state `(S_t, …, S_{t−4})` with a common AR block, and `switching_variance` is
the only switch to flip — its default `True` fits a two-variance
generalization that is *not* Hamilton's model. No re-parameterization is
needed on either side of the comparison: tsecon's `means` are regime means,
and statsmodels' switching `const` in `MarkovAutoregression` is also the
regime mean (not an intercept), so the columns above are directly comparable
with no `intercept/(1 − Σφ)` mapping. tsecon's `transition` is column-
stochastic — `transition[i][j] = P(S_t = i | S_{t−1} = j)` — so the staying
probabilities are its diagonal, same as statsmodels' `p[0->0]`. EM regime
indices are arbitrary: the script labels regimes by their means (contraction =
low mean), never by index.

---

## Honest deviations

**EM fixed point vs exact MLE.** tsecon fits by EM; statsmodels/E-views refine
to the exact MLE by quasi-Newton. The two differ here by more than a
convergence tolerance for a structural reason: the EM transition update is the
expected-count formula, which conditions on the model's stationary initial
state distribution instead of re-differentiating that distribution with
respect to the transition matrix. Its fixed point therefore sits an O(1/T)
distance from the exact MLE — on this sample, 0.006 log-likelihood points, and
at most 0.016 on any parameter (the contraction mean, the least-populated
corner of the sample). Tightening `tol` does not close the gap (the EM path is
converged; it is the *estimator* that differs slightly), and no economic
statement in the paper is sensitive to it.

**AR coefficients.** The binding returns the estimated common AR(4) under the
`ar` key (length-`order`, shared across regimes — it is the block Hamilton's
likelihood applies to deviations `y_t − μ_{S_t}`), so the φ row above is a
full three-way comparison. Measured: tsecon's φ sit within **0.0048** of the
published values and **0.0043** of statsmodels' exact MLE on identical data
(worst coefficient φ₂ in both cases) — the same EM-vs-MLE third-decimal gap
as every other parameter. Hamilton's printed φ are notoriously
optimizer-sensitive, so the CI tolerance stays at the module-wide 0.02
budget rather than the achieved 0.005. (Through 0.5.0 the binding did not
return the AR block and a CI guard pinned that gap; the guard has been
flipped into this comparison, as its own docstring instructed.)

**Published-digit precision.** Hamilton's own Table I was computed in 1989 by
numerical maximization on the same series; the modern E-views/statsmodels MLE
reproduces it to roughly two decimals (e.g. q: 0.7550 published vs 0.7547
re-estimated). The test tolerances below inherit both gaps — printed-digit
rounding and EM-vs-MLE — and are stated per parameter in the test file.

**What is being claimed.** This reproduces Hamilton's economics — two
discrete, persistent growth states with the published means and persistences,
and an NBER chronology recovered from GNP alone — plus a tight cross-package
agreement on identical data. It does not claim bitwise equality with a 1989
optimizer run.

**Citation.** Hamilton, J. D. (1989), "A New Approach to the Economic Analysis
of Nonstationary Time Series and the Business Cycle," *Econometrica*
57(2):357–384. US GNP data are US-government statistics (public domain),
vendored here from the statsmodels test suite.

**See also.** [`markov_switching_ar` model card](../reference/model-cards/cointegration-regime.md) ·
[Ramey-Zubairy replication](replication-ramey-zubairy.md) ·
[yield-curve recession replication](replication-yield-curve-recession.md).
