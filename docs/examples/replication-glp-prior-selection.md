# Replication — GLP prior selection (Giannone-Lenza-Primiceri 2015)

How tight should a Bayesian VAR's prior be? For twenty years the answer was
folklore — "overall tightness 0.2", inherited from Litterman and Sims-Zha.
Giannone, Lenza & Primiceri (2015, *REStat*) replaced the folklore with a
hierarchical model: treat the tightness as a parameter, give it a diffuse
Gamma hyperprior, and let the data choose it through the closed-form marginal
likelihood. `tsecon.bvar_hierarchical` implements exactly this machinery, and
this page runs it through GLP's published application design.

```sh
.venv/bin/python docs/examples/replication_glp_prior_selection.py
```

!!! warning "This is a design replication on nearby public data — not a run on GLP's dataset"
    GLP's application uses the **Stock-Watson (2008)** panel: real GDP, the
    **GDP deflator** and the **federal funds rate** for their small VAR
    (1959Q1–2008Q4, five lags). Their replication archive is distributed
    through their publishers and is not redistributed here. The committed
    fixture, [`fixtures/glp_smallvar.csv`](../../fixtures/glp_smallvar.csv),
    is the public-domain statsmodels `macrodata` panel (US-government
    statistical series, BEA/BLS/FRB) over the same sample, carrying the same
    *kind* of variables — but **CPI is not the GDP deflator**, the **3-month
    T-bill is not the federal funds rate**, and the vintage differs. Every
    *design* choice below is GLP's; no *number* below is GLP's, and none
    should be quoted as such.

---

## The design (verified against GLP's own materials)

The published details were checked against the paper draft and the authors'
web replication code (`setpriors.m`, `logMLVAR_formin.m`, `ExampleMinnesotaOnly.m`;
publicly mirrored in FRBNY-DSGE/BrookingsPC2020 on GitHub under
`src/ReplicationFilesVAR/GLPreplicationWeb/`), not quoted from memory:

| design choice | GLP (2015) | this page |
|---|---|---|
| small VAR | real GDP, GDP deflator, fed funds | realgdp, **cpi**, **tbilrate** |
| medium VAR | + consumption, investment, hours, wages | + realcons, realinv, **realgovt, realdpi** |
| transformation | annualized log-levels (4·log); rates in levels/100 | same |
| sample | 1959Q1–2008Q4 (200 quarters) | same (same span, different vintage) |
| lags | 5 | same |
| prior mean | own first lag = 1 (random walk, levels data) | same (`delta=1`) |
| lag decay | `1/l^2` in the prior variance | same (`lambda3=1`) |
| hyperprior on λ | Gamma, **mode 0.2, sd 0.4** (their `setpriors.m`) | same (`hyperprior="glp"`, the library default) |
| scale regressions σ²ⱼ | AR(1) residual variances (Figure 1 illustration) | **AR(4)** residual variances (tsecon's documented convention) |

GLP's medium VAR adds the Smets-Wouters real aggregates; macrodata has no
hours or wages series, so the medium analogue substitutes government spending
and disposable income — four real aggregates, keeping the small set nested,
which is what their cross-section claim is about.

---

## Published vs. obtained

**What GLP publish** (their Figure 1 plots the posterior of the overall
tightness λ for the small/medium/large VARs under the Gamma hyperprior; the
curve locations below were recovered from the figure's vector graphics in the
draft PDF, so read them as ±0.03):

* small VAR: a broad posterior peaking around **0.4–0.45**;
* medium VAR: a tighter posterior peaking around **0.17**;
* large VAR (22 variables): a spike around **0.09**;
* in their words: "the posterior mode (and variance) of λ decreases with the
  size of the model" — bigger system, tighter selected prior;
* the fixed Sims-Zha tightness (0.2, the value their hyperprior is centred
  on) is "too low" — i.e. too *tight* — for the small VAR, and the
  data-driven choice beats fixed and heuristic hyperparameters out of sample.

**What this design replication obtains** (macrodata stand-ins, tsecon's AR(4)
scale convention):

| quantity | published (GLP data) | obtained (nearby data) |
|---|---|---|
| selected λ₁, small (3-var) | mode ≈ 0.4–0.45 *(Figure 1)* | **0.215** |
| selected λ₁, medium (7-var) | mode ≈ 0.17 *(Figure 1)* | **0.145** |
| direction as the cross-section grows | tighter (small → medium → large) | tighter (0.215 → 0.145) ✓ |
| selected vs. near-flat prior (λ₁ = 5) | hierarchical wins (their MSFE tables) | +92 nats of log-ML (small) ✓ |
| selected vs. fixed 0.2 | small VAR wants *looser* than 0.2 | 0.215 > 0.2, log-ML gain +0.06 ✓ (direction) |

The λ₁ profile the script prints shows the small-VAR posterior kernel
concentrated on roughly **0.12–0.36** with its peak near 0.215 — the same
order of magnitude as the published behaviour, one documented convention away
from it (next section), and far from both the collapse-to-zero and the
flat-prior corner.

---

## Closing the gap on GLP's own data

The gap between 0.215 here and ≈0.45 in Figure 1 has two sources: the data
(CPI/T-bill/vintage vs. deflator/fed-funds) and one prior convention (AR(4)
vs. AR(1) scale regressions — documented in the
[Bayesian model card](../reference/model-cards/bayesian.md) precisely because
packages differ here and results are sensitive). During development both were
isolated on GLP's own web-replication dataset (`DataSW.mat`, the 7-variable
Stock-Watson panel from the public mirror above — verified against the same
design, but **not** committed to this repository, which is why these numbers
live on this page and not in the test suite):

| run (GLP's own data, small / medium VAR) | selected λ₁ |
|---|---|
| `tsecon.bvar_hierarchical` as shipped (AR(4) scales) | 0.260 / 0.142 |
| independent NumPy re-computation, AR(4) scales | ≈ 0.25 / — |
| same NumPy code, **only** the scales switched to GLP's AR(1) | **0.449 / 0.172** |
| published Figure 1 modes (vector-extracted) | ≈ 0.42–0.45 / ≈ 0.17 |

Switching the one documented convention reproduces the published figure's
small- and medium-VAR modes to the resolution the figure can be read at. The
residual difference between tsecon and the published curves is therefore the
σ²ⱼ convention plus the data — not the selection machinery, which is
golden-pinned to an independent implementation at 1e-9
([validation matrix](../reference/validation-matrix.md)).

---

## How it is run

```python
import csv, numpy as np, tsecon

rows = [r for r in csv.reader(open("fixtures/glp_smallvar.csv"))
        if r and not r[0].startswith("#")]
cols = {n: np.array([float(r[i]) for r in rows[1:]])
        for i, n in enumerate(rows[0])}

small = np.column_stack([4 * np.log(cols["realgdp"]),
                         4 * np.log(cols["cpi"]),
                         cols["tbilrate"] / 100.0])

fit = tsecon.bvar_hierarchical(small, lags=5, delta=1.0, hyperprior="glp")
fit["lambda1_opt"]              # 0.215 — the data-chosen tightness
fit["log_marginal_likelihood"]  # the evidence at the optimum
fit["grid_lambda1"], fit["grid_log_ml"]   # the profile behind the choice
```

`hyperprior="glp"` (the default) maximizes log-ML *plus* the log of GLP's
Gamma(mode 0.2, sd 0.4) hyperprior — MAP-II, exactly the object whose
posterior GLP plot in Figure 1. `hyperprior="none"` is pure ML-II; on this
data the two differ only in the third decimal (0.2155 vs 0.2150), because 200
quarters of likelihood dominate a diffuse hyperprior. The posterior refit at
the optimum is returned in the same call (`posterior_mean_coefs`,
`sigma_posterior_mean`) — a drop-in `bvar_fit` that tuned its own shrinkage.

!!! note "What the log-ML comparison does and does not replicate"
    GLP's published dominance results are out-of-sample MSFE and density-
    forecast tables from a 1975–2008 recursive forecasting exercise. The
    dominance shown here is in-sample *marginal likelihood* — the objective
    their method maximizes and interprets as a one-step density-forecast
    record — at the full-sample optimum. It is the machinery's own
    certificate (the optimum must beat every fixed λ₁), not a re-run of
    their forecasting horse race.

---

## What this is, and is not

This reproduces GLP's prior-selection *design* — their transformations, their
sample span, their lag length, their hyperprior — and checks the claims their
paper actually makes: the data-chosen tightness is a few tenths (not 1e-4,
not 10), it shrinks as the cross-section grows, a 3-variable VAR wants a
looser prior than the 0.2 folklore, and the selection dominates fixed
references in the evidence. All of those reproduce here.

It does **not** reproduce GLP's published numbers, because it does not use
GLP's data: the price and interest-rate series are different concepts, the
vintage differs, and tsecon's scale-regression convention is AR(4) where
their Figure-1 illustration uses AR(1). Where a published number is quoted
above it is clearly marked as GLP's; where a number was obtained here it is
never presented as theirs.

**Citation.** Giannone, D., Lenza, M. & Primiceri, G. E. (2015), "Prior
Selection for Vector Autoregressions," *The Review of Economics and
Statistics* 97(2):436–451. Data: statsmodels `macrodata` (public-domain US
government statistical series); GLP's own materials consulted via the public
mirror of their web replication files.

**See also.** [Bayesian VAR model card](../reference/model-cards/bayesian.md) ·
[Ramey-Zubairy replication](replication-ramey-zubairy.md) ·
[yield-curve recession replication](replication-yield-curve-recession.md)
