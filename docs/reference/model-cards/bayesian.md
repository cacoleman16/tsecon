# Model card — Bayesian VAR

`bvar_fit` · `bvar_irf_draws` · `mcmc_diagnostics`

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
