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

!!! warning "Two legs: a design replication on nearby data, and a point replication on GLP's own panel"
    GLP's application uses the **Stock-Watson (2008)** panel: real GDP, the
    **GDP deflator** and the **federal funds rate** for their small VAR
    (1959Q1–2008Q4, five lags). The first leg of this page runs their design
    on [`fixtures/glp_smallvar.csv`](../../fixtures/glp_smallvar.csv) — the
    public-domain statsmodels `macrodata` panel (US-government statistical
    series, BEA/BLS/FRB) over the same sample, carrying the same *kind* of
    variables — but **CPI is not the GDP deflator**, the **3-month T-bill is
    not the federal funds rate**, and the vintage differs. On that leg every
    *design* choice is GLP's and no *number* is GLP's.

    Since 0.4.0 there is a second leg: GLP's own panel, exactly as their web
    replication code consumes it, is committed as
    [`fixtures/glp_sw_panel.csv`](../../fixtures/glp_sw_panel.csv) (vendored
    from the public FRBNY-DSGE/BrookingsPC2020 GitHub mirror of their
    replication files; the mirror's redistribution notice is kept in the CSV
    header, and the underlying series are public-domain US-government
    statistics). On *that* data, with `scale_ar=1` — the option matching
    GLP's own residual-scale convention — the selected modes land on their
    published Figure-1 locations, and a CI test pins them
    ([`test_replication_glp_point.py`](../../bindings/python/tests/test_replication_glp_point.py)).

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
| scale regressions σ²ⱼ | AR(1) residual variances (their `setpriors.m`, `MNpsi=0`) | **AR(4)** by default; `scale_ar=1` gives GLP's AR(1) convention (new in 0.4.0) |

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
| selected λ₁, small (3-var) | mode ≈ 0.4–0.45 *(Figure 1)* | **0.215** (AR(4) default) / **0.269** (`scale_ar=1`) |
| selected λ₁, medium (7-var) | mode ≈ 0.17 *(Figure 1)* | **0.145** (AR(4) default) / **0.155** (`scale_ar=1`) |
| direction as the cross-section grows | tighter (small → medium → large) | tighter under both conventions ✓ |
| selected vs. near-flat prior (λ₁ = 5) | hierarchical wins (their MSFE tables) | +92 nats of log-ML (small) ✓ |
| selected vs. fixed 0.2 | small VAR wants *looser* than 0.2 | 0.215 > 0.2, log-ML gain +0.06 ✓ (direction) |

GLP's own `scale_ar=1` convention moves the nearby-data selection toward the
published modes (0.215 → 0.269 small, 0.145 → 0.155 medium) without reaching
them — the residual gap on this leg is the data, not the machinery, which is
exactly what the point replication below confirms.

The λ₁ profile the script prints shows the small-VAR posterior kernel
concentrated on roughly **0.12–0.36** with its peak near 0.215 — the same
order of magnitude as the published behaviour, one documented convention and
one dataset away from it (next section), and far from both the
collapse-to-zero and the flat-prior corner.

---

## Closing the gap on GLP's own data — the point replication

The gap between 0.215 here and ≈0.45 in Figure 1 has two sources: the data
(CPI/T-bill/vintage vs. deflator/fed-funds) and one prior convention (AR(4)
vs. AR(1) scale regressions — documented in the
[Bayesian model card](../reference/model-cards/bayesian.md) precisely because
packages differ here and results are sensitive). Since 0.4.0 both are
resolved *in the test suite*: the convention is a keyword (`scale_ar=1`,
matching GLP's own `setpriors.m`), and GLP's data is committed
([`fixtures/glp_sw_panel.csv`](../../fixtures/glp_sw_panel.csv) — their
`DataSW.mat`, the 7-variable Stock-Watson panel, vendored from the public
FRBNY-DSGE/BrookingsPC2020 mirror with its redistribution notice kept):

| run (GLP's own data, small / medium VAR) | selected λ₁ |
|---|---|
| `tsecon.bvar_hierarchical` as shipped (AR(4) scales) | 0.260 / 0.142 |
| same call, **only** `scale_ar=1` (GLP's AR(1) convention) | **0.420 / 0.172** |
| published Figure 1 modes (vector-extracted, ±0.03) | ≈ 0.42–0.45 / ≈ 0.17 |

Switching the one documented convention reproduces the published figure's
small- and medium-VAR modes to the resolution the figure can be read at, and
[`test_replication_glp_point.py`](../../bindings/python/tests/test_replication_glp_point.py)
pins both in CI: |λ_small − 0.449| ≤ 0.03 (obtained 0.420, at the lower edge
of the reading band) and |λ_medium − 0.172| ≤ 0.01 (obtained 0.1716). The
selection machinery itself is golden-pinned to an independent implementation
at 1e-9 under *both* conventions
([validation matrix](../reference/validation-matrix.md)).

One honesty note on the third decimal: the 0.3.0 development-time NumPy
re-computation of this exercise recorded 0.449 for the small VAR where the
shipped `scale_ar=1` path obtains 0.420. Both sit inside the figure-reading
band. On this heavily collinear 4·log-levels design the marginal likelihood
is flat near its peak (the log-ML within the band varies by well under one
nat), so the argmax's second decimal is soft — sensitive to the linear-algebra
path and library versions — while the log-ML *value* is sturdy. The CI pins
above state exactly the tolerances at which the published claim is tested.

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

# GLP-exact: their setpriors.m scales the prior with AR(1), not AR(4),
# residual variances. One keyword switches the convention (0.4.0+):
tsecon.bvar_hierarchical(small, lags=5, delta=1.0, hyperprior="glp",
                         scale_ar=1)["lambda1_opt"]   # 0.269 on this data
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

The macrodata leg reproduces GLP's prior-selection *design* — their
transformations, their sample span, their lag length, their hyperprior — and
checks the claims their paper actually makes: the data-chosen tightness is a
few tenths (not 1e-4, not 10), it shrinks as the cross-section grows, a
3-variable VAR wants a looser prior than the 0.2 folklore, and the selection
dominates fixed references in the evidence. All of those reproduce there,
and none of that leg's numbers are GLP's (different price and interest-rate
concepts, different vintage).

The `glp_sw_panel.csv` leg *is* a run on GLP's data with GLP's own
residual-scale convention (`scale_ar=1`), and it reproduces the published
Figure-1 small/medium modes at the stated reading tolerances — a point
replication of the figure's location claims. It is still not a re-run of
their full paper: the out-of-sample MSFE horse race, the large (22-variable)
VAR, and their sum-of-coefficients/single-unit-root dummy priors (which
tsecon's conjugate Minnesota-NIW does not implement) remain out of scope.

**Citation.** Giannone, D., Lenza, M. & Primiceri, G. E. (2015), "Prior
Selection for Vector Autoregressions," *The Review of Economics and
Statistics* 97(2):436–451. Data: statsmodels `macrodata` (public-domain US
government statistical series) for the design leg; GLP's own web replication
panel (Stock-Watson 2008 US-government series, via the public
FRBNY-DSGE/BrookingsPC2020 mirror, redistribution notice kept in the CSV
header) for the point leg.

**See also.** [Bayesian VAR model card](../reference/model-cards/bayesian.md) ·
[Ramey-Zubairy replication](replication-ramey-zubairy.md) ·
[yield-curve recession replication](replication-yield-curve-recession.md)
