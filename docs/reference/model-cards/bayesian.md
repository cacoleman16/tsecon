# Model card — Bayesian VAR

`bvar_fit` · `bvar_hierarchical` · `bvar_ssvs` · `bvar_irf_draws` ·
`mcmc_diagnostics`

A VAR has a lot of coefficients — `K(1 + pK)` of them — and short macro samples
cannot pin them all down. The Bayesian VAR fixes this with a prior that shrinks
the system toward a sensible default (each variable a random walk, distant lags
near zero), trading a little bias for a large variance reduction. The Minnesota
prior in conjugate Normal-Inverse-Wishart form makes the posterior available in
closed form — no sampler needed for the coefficients.

---

## `bvar_fit` — Minnesota-NIW posterior

**What it estimates.** The conjugate Normal-Inverse-Wishart posterior of a
VAR(p) under a Minnesota prior: posterior-mean coefficients, the posterior-mean
residual covariance, and the log marginal likelihood used to compare
hyperparameter settings.

**Assumptions.** Gaussian innovations; the Minnesota prior structure (own first
lag centered at `delta`, tighter shrinkage on other variables and higher lags);
covariance-stationary data is *not* required for estimation, but the random-walk
prior encodes a persistence belief you should mean to hold.

**When to use (and when not).** Use for medium-to-large systems on short
samples, density forecasts, and structural analysis where OLS VAR coefficients
are too noisy. Not needed for tiny systems on long samples (OLS `var_fit` is
fine), and the conjugate prior cannot express stochastic volatility or
time-varying parameters — those need a sampler beyond this card.

**Key arguments and defaults (and why).** `lags`; the shrinkage
hyperparameters `lambda0` (overall tightness — smaller = more shrinkage toward
the prior), `lambda1` (own-lag scale), `lambda3` (lag-decay rate), and `delta`
(prior mean of the own first lag; 1.0 = random-walk prior). Defaults follow the
standard Minnesota calibration; tune `lambda0`/`lambda1` by maximizing
`log_marginal_likelihood` (the Giannone-Lenza-Primiceri 2015 hierarchical
recommendation).

**How to read the output.** `posterior_mean_coefs` ((1+pK)×K),
`sigma_posterior_mean` (K×K), and `log_marginal_likelihood` — the model-
comparison score: fit at one hyperparameter setting is meaningful only *relative*
to another, so use it to choose shrinkage, not as an absolute number.

**Failure modes.** Over-shrinkage (`lambda0` too small) flattens dynamics toward
the random-walk prior; under-shrinkage buys nothing over OLS. Comparing marginal
likelihoods across different samples or variable transforms is meaningless.

**Validated against.** Self-authored closed-form NIW posterior updating checked
against the analytic conjugate formulas (`fixtures/bvar_niw.json`).

**References.** Doan, Litterman & Sims (1984); Kadiyala & Karlsson (1997);
Giannone, Lenza & Primiceri (2015).

---

## `bvar_hierarchical` — empirical-Bayes tightness selection (GLP)

**What it estimates.** The same conjugate Minnesota-NIW posterior as `bvar_fit`,
but with the overall tightness `lambda1` **chosen by the data** instead of set by
folklore. It maximizes the closed-form log marginal likelihood over `lambda1`
(the Giannone-Lenza-Primiceri 2015 empirical-Bayes / ML-II move), then refits the
conjugate posterior at the optimum — a drop-in richer `bvar_fit` that tunes its
own shrinkage. No new likelihood algebra: the marginal likelihood is the one the
NIW posterior already computes, maximized over the prior dial.

**Assumptions.** Everything `bvar_fit` assumes (Gaussian innovations, the
Minnesota prior structure), plus that the marginal likelihood is a defensible
criterion for the tightness — which requires keeping *every* `lambda`-dependent
term of the evidence (the "constants" people drop when comparing parameters
within one model are not constant across priors).

**When to use (and when not).** Use whenever you would otherwise pick `lambda1`
by hand or by RMSE grid search — it is the modern default for serious BVAR
forecasting, and it earns the most on short, persistent samples where the choice
of shrinkage matters. Not needed when you already have a defensible tightness or
when the sample is long enough that the likelihood is flat in `lambda1` (the
optimum then sits right next to the conventional 0.2, as it does on the fixture
below).

**Key arguments and defaults (and why).** `optimize="lambda1"` (default) tunes
only the overall tightness; `"lambda1+lambda3"` also tunes the lag-decay rate.
`hyperprior="none"` is pure ML-II (maximize the evidence); `"glp"` adds the GLP
Gamma hyperprior (mode 0.2, sd 0.4) and maximizes the log *posterior* instead
(MAP-II). `lambda1_lo`/`lambda1_hi` bracket the search; `n_grid` sets the
pre-scan resolution; `delta`/`lambda0`/`lambda3` are the fixed Minnesota dials
(as in `bvar_fit`).

**How to read the output.** `lambda1_opt` (and `lambda3_opt`) — the selected
tightness; `log_marginal_likelihood` / `log_posterior` at the optimum;
`lambda1_fixed_log_ml` — the evidence at the conventional `lambda1=0.2`, which the
optimum dominates (the whole point); `posterior_mean_coefs` and
`sigma_posterior_mean` — the refit posterior; `grid_lambda1` / `grid_log_ml` —
the pre-scan profile you can plot to see how peaked the evidence is; `converged`
and `n_evals`.

**Failure modes.** Dropping `lambda`-dependent constants from the marginal
likelihood silently corrupts the selection; reporting the ML-II tightness on a
sample so short the evidence is nearly flat (the "optimum" is then noise — check
the `grid_log_ml` profile); comparing the evidence across different samples or
variable transforms (meaningless, same as `bvar_fit`).

**Validated against.** An independent NumPy/SciPy re-implementation of the same
closed-form matrix-variate-t marginal likelihood (Kadiyala-Karlsson 1997 eq. 3.6),
maximized with `scipy.optimize` — a cross-implementation golden that never imports
tsecon ([`bvar_hierarchical.json`](../../../fixtures/bvar_hierarchical.json),
[`hierarchical.rs`](../../../crates/tsecon-bayes/tests/hierarchical.rs)). See the
[validation matrix](../validation-matrix.md).

**References.** Giannone, Lenza & Primiceri (2015, *REStat*); Kadiyala & Karlsson
(1997).

```python
import json, numpy as np, tsecon

y = np.array(json.load(open("fixtures/var.json"))["data_100dlog_gdp_cons_inv"])

h = tsecon.bvar_hierarchical(y, lags=2, optimize="lambda1")
print("selected lambda1:", round(h["lambda1_opt"], 4),
      " log-ML at optimum:", round(h["log_marginal_likelihood"], 4))
print("log-ML at the conventional lambda1 = 0.2:", round(h["lambda1_fixed_log_ml"], 4))
print("converged:", h["converged"], " evaluations:", h["n_evals"])

# On a short, persistent sample the data-chosen tightness moves off 0.2 and the
# marginal likelihood improves materially over the fixed default.
rng = np.random.default_rng(1)
k, n = 4, 60
A = 0.8 * np.eye(k)
Y = np.zeros((n, k))
for t in range(1, n):
    Y[t] = A @ Y[t - 1] + 0.3 * rng.standard_normal(k)
hs = tsecon.bvar_hierarchical(Y, lags=3, optimize="lambda1")
print("short sample (k=4, n=60, p=3): selected lambda1 =", round(hs["lambda1_opt"], 4),
      " log-ML gain over fixed 0.2 =",
      round(hs["log_marginal_likelihood"] - hs["lambda1_fixed_log_ml"], 3))
```

```
selected lambda1: 0.1942  log-ML at optimum: -861.5642
log-ML at the conventional lambda1 = 0.2: -861.5704
converged: True  evaluations: 81
short sample (k=4, n=60, p=3): selected lambda1 = 0.3058  log-ML gain over fixed 0.2 = 3.564
```

On the long fixture sample the evidence is nearly flat: the ML-II optimum (0.194)
sits a whisker from the conventional 0.2 and barely improves the marginal
likelihood. On the short, persistent 4-variable sample the story changes — the
data pull the tightness up to 0.31 and buy a 3.6-log-point improvement, exactly
the regime where letting the data set the dial matters.

---

## `bvar_ssvs` — spike-and-slab stochastic-search selection

**What it estimates.** The SSVS-BVAR of George, Sun & Ni (2008): a
**stochastic search variable selection** posterior over which VAR coefficients
(and, optionally, which off-diagonal error precisions) are non-zero. Every
coefficient gets a two-component **spike-and-slab** prior — a narrow "spike"
$N(0, (c_0\tau)^2)$ that pins it near zero, and a wide "slab"
$N(0, (c_1\tau)^2)$ that lets the data speak — governed by a latent 0/1
inclusion indicator with prior inclusion probability `prior_inclusion`. A
four-block Gibbs sampler visits (coefficients, indicators, error precision,
precision indicators), and the **fraction of draws in which each coefficient is
"in"** is its posterior inclusion probability. The prior scales $\tau$ are set
semi-automatically from the OLS standard errors, so the spike/slab dials are
scale-free across variables in different units. Where `bvar_fit` shrinks *every*
coefficient by a common Minnesota dial, SSVS lets the sampler decide, one
coefficient at a time, whether a lag or cross-effect belongs in the model at
all — the Bayesian analogue of the LASSO's selection, with a full posterior
instead of a point.

**Assumptions.** Gaussian innovations; the spike-and-slab prior structure (a
genuinely sparse coefficient matrix is the belief that earns SSVS its keep); the
intercept row is always included (pinned to inclusion 1, never searched). With
`ssvs_cov=True`, the same spike-and-slab machinery selects the off-diagonal
elements of the error **precision** (Cholesky) factor — i.e. which contemporaneous
links between equations are non-zero.

**When to use (and when not).** Use when you suspect the VAR is *sparse* — most
distant lags and cross-variable effects are truly zero — and you want the data,
not a single tightness dial, to say which ones survive, with inclusion
probabilities as an honest soft selector. Not needed when you only want
shrinkage (`bvar_fit` / `bvar_hierarchical` are closed-form and faster), when the
system is small on a long sample (OLS is fine), or when the coefficient matrix is
genuinely dense (SSVS then selects everything and buys nothing over shrinkage).
Because it is a sampler, it inherits every MCMC obligation — run multiple chains
and check convergence.

**Key arguments and defaults (and why).** `lags`; `n_draws` / `burn` / `thin`
(the Gibbs budget); `c0` (spike scale, small — e.g. `0.1`) and `c1` (slab scale,
large — e.g. `10.0`), whose *ratio* sets how sharply "in" and "out" are
distinguished; `prior_inclusion` (the prior probability a coefficient is in;
`0.5` is agnostic); `ssvs_cov` plus `kappa0` / `kappa1` / `prior_inclusion_cov`
(the same three dials for the error-precision selection); `gamma_a` / `gamma_b`
(the inverse-gamma prior on the precision diagonals); `horizon` (IRF length);
`n_chains` (≥ 2 to get `rhat` / `ess_bulk`); `seed` (reproducible via the Philox
stream).

**How to read the output.** `inclusion_prob` (k × n, same
regressor-by-equation layout as `bvar_fit["posterior_mean_coefs"]`, the intercept
row pinned to 1) — the headline: a coefficient with inclusion near 1 is firmly in
the model, near 0 firmly out, and the *interesting* ones are the ambiguous
middle. `coef_mean` (k × n) and `sigma_mean` (n × n) are the posterior means
(model-averaged over the visited sparsity patterns). `irf_draws`
`[draw][h][variable][shock]` are Cholesky-orthogonalized IRF draws for credible
bands. `inclusion_prob_cov` appears only with `ssvs_cov=True`. The `diagnostics`
dict carries `mean_model_size` (the average number of *selected slopes* — read it
against the true sparsity), `n_draws_kept`, `log_marginal_likelihood_median`, and
— with `n_chains ≥ 2` — `rhat` and `ess_bulk`.

**Failure modes.** Reading an inclusion probability as a frequentist p-value (it
is a posterior probability under *this* prior, and it moves with `c0`/`c1` and
`prior_inclusion` — report the dials). A too-similar spike and slab (`c0` and
`c1` close) makes "in" and "out" indistinguishable, so nothing is selected; too
extreme, and the indicator sticks and mixes terribly. **Low ESS on the inclusion
indicators is the SSVS-specific wrinkle** — the discrete indicators are highly
autocorrelated even when the continuous parameters mix well, so run long and
check `ess_bulk`. And the usual sampler traps: a single chain cannot diagnose
convergence, and SSVS on a dense system selects everything and wastes the effort.

**Validated against.** SSVS is a **sampler**, so there is no closed-form golden
to lock it to (unlike the golden-pinned `bvar_fit`). Its headline validation is
an honest **Monte-Carlo recovery** test on a stable *sparse* VAR(2): the fixture
stores the true lag matrices and the true-nonzero / true-zero coefficient masks;
the data are simulated from a `tsecon_rng::Stream`; and `bvar_ssvs` must drive
the posterior inclusion probabilities **near 1 on the true non-zeros and near 0
on the true zeros**. The remaining tests pin seed reproducibility, output shapes,
the covariance-selection path, the multi-chain diagnostics, and the input
guardrails; the closed-form conditional-moment anchors and the block-1 draw
kernel are checked in the crate's unit tests. Fixture:
[`ssvs.json`](../../../fixtures/ssvs.json); test:
[`ssvs.rs`](../../../crates/tsecon-bayes/tests/ssvs.rs). See the
[validation matrix](../validation-matrix.md).

**References.** George & McCulloch (1993); George, Sun & Ni (2008).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n, T, p = 3, 300, 2

# A deliberately SPARSE VAR(2): most coefficients are exactly zero.
A1 = np.array([[0.5, 0.0, 0.0],
               [0.3, 0.4, 0.0],
               [0.0, 0.0, 0.6]])
A2 = np.array([[0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.2]])
c = np.array([0.2, -0.1, 0.0])
Y = np.zeros((T + 50, n))
for t in range(2, T + 50):
    Y[t] = c + A1 @ Y[t - 1] + A2 @ Y[t - 2] + 0.5 * rng.standard_normal(n)
Y = Y[50:]

res = tsecon.bvar_ssvs(Y, lags=2, n_draws=4000, burn=1000, seed=1, n_chains=2)
inc = np.asarray(res["inclusion_prob"])   # (1+pK) x n; rows const,L1.y1..y3,L2.y1..y3
print("inclusion_prob (cols = equations):")
print(np.round(inc, 2))
print("mean model size:", round(res["diagnostics"]["mean_model_size"], 2),
      "selected slopes  (the truth has 5 non-zero slopes)")
print("R-hat:", round(res["diagnostics"]["rhat"], 4),
      " ESS bulk:", round(res["diagnostics"]["ess_bulk"], 0))
```

```
inclusion_prob (cols = equations):
[[1.   1.   1.  ]
 [1.   1.   0.11]
 [0.36 1.   0.08]
 [0.07 0.16 1.  ]
 [0.12 0.21 0.08]
 [0.09 0.09 0.09]
 [0.09 0.22 0.84]]
mean model size: 6.59 selected slopes  (the truth has 5 non-zero slopes)
R-hat: 1.0043  ESS bulk: 224.0
```

The five true non-zero coefficients (own lags `L1.y1→y1`, `L1.y2→y2`,
`L1.y3→y3`, `L2.y3→y3`, and the cross-effect `L1.y1→y2`) all carry inclusion
probabilities from 0.84 up to 1.00 — the weakest, the `0.2` second-lag
coefficient, still clears 0.8 — while every true zero sits at 0.36 or below. The
intercept row is pinned to 1 by design. SSVS recovered the sparsity pattern from
the data without a single hard threshold: the inclusion probabilities *are* the
soft variable selection, and the low-hundreds `ess_bulk` on the discrete
indicators (against 6,000 kept draws) is the SSVS mixing wrinkle to watch.

---

## `bvar_irf_draws` — posterior impulse-response draws

**What it estimates.** Draws from the posterior of the Cholesky-identified
impulse responses: sample `(coefs, Sigma)` from the NIW posterior, form the
recursive IRF for each draw. The spread across draws *is* the credible band —
correctly cumulated (draw-wise) when `cumulative=True`.

**Key arguments and defaults.** `horizon`, `n_draws` (more for smoother bands),
`seed` (reproducible via the Philox stream), the same shrinkage hyperparameters
as `bvar_fit`, `cumulative`.

**How to read the output.** A `[draw][h][variable][shock]` array. Summarize with
percentiles across the draw axis — e.g. the 16th/50th/84th percentiles give a
68% credible band. Because bands are built from whole draws, the cumulative
view cumulates uncertainty correctly (unlike gluing pointwise quantiles).

**Failure modes.** Too few draws leave ragged bands; the Cholesky ordering is a
structural assumption (see the [SVAR card](var-svar.md) for set-identified
alternatives).

**Validated against.** Same NIW posterior machinery as `bvar_fit`
(`fixtures/bvar_niw.json`); the recursive IRF shares the validated VAR core.

**References.** Sims & Zha (1998); Kilian & Lütkepohl (2017).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
k, n = 3, 300
A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.5]])
Y = np.zeros((n, k))
for t in range(1, n):
    Y[t] = A @ Y[t - 1] + 0.3 * rng.standard_normal(k)

post = tsecon.bvar_fit(Y, lags=2, lambda1=0.2)
print("log marginal likelihood:", round(post["log_marginal_likelihood"], 2))

draws = np.asarray(tsecon.bvar_irf_draws(Y, lags=2, horizon=8, n_draws=1000, seed=0))
lo, med, hi = np.percentile(draws[:, :, 0, 0], [16, 50, 84], axis=0)   # own response of var0
print("median IRF (h=0..2):", np.round(med[:3], 3))
print("68% band  (h=0..2):", np.round(lo[:3], 3), np.round(hi[:3], 3))
```

---

## `mcmc_diagnostics` — convergence checks

**What it estimates.** The two questions you must answer before trusting *any*
sampler's output: did the chains converge to the same distribution, and how many
*effective* independent draws do you have? Returns the rank-normalized split
R-hat and the bulk/tail effective sample sizes.

**When to use.** After running any MCMC sampler (here, on the draw dimension of
`bvar_irf_draws` reshaped into chains, or on external chains). This is a
diagnostic, not an estimator — run it every time.

**Key arguments.** `chains` — a `(n_chains, n_draws)` array for one scalar
quantity.

**How to read the output.** `rhat` should be **< 1.01** (values above flag
non-convergence — run longer or reparameterize). `ess_bulk` gauges precision of
the posterior center, `ess_tail` of the tails (credible-interval endpoints);
both should be comfortably in the hundreds-plus. Low tail ESS means your
interval endpoints are noisy even if the mean looks fine.

**Failure modes.** A single chain cannot diagnose convergence (R-hat needs ≥2);
high R-hat with high ESS still means non-convergence — R-hat governs.

**Validated against.** ArviZ — rank-normalized split-R-hat and bulk/tail ESS,
to matching precision (`fixtures/convergence.json`).

**References.** Gelman & Rubin (1992); Vehtari, Gelman, Simpson, Carpenter &
Bürkner (2021, rank-normalized R-hat and ESS).

```python
import numpy as np, tsecon
rng = np.random.default_rng(0)
chains = rng.standard_normal((4, 1000))          # 4 well-mixed chains
d = tsecon.mcmc_diagnostics(chains)
print("R-hat:", round(d["rhat"], 4), " ESS bulk/tail:",
      round(d["ess_bulk"], 0), round(d["ess_tail"], 0))
```
