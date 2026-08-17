# prophet_lite

A small, from-scratch implementation of the decomposable structural
forecaster of Taylor & Letham — the Prophet core — for tsecon's private
research lab. **Not part of the released tsecon surface** (not in the wheel,
not in the docs, not public API); it exists for a planned comparison study
against tsecon's own forecasters (`arima_fit`, `theta_forecast`, `backtest`,
...).

## What is implemented

`y(t) = trend(t) + seasonality(t) + beta' x(t) + eps`, `eps ~ N(0, sigma^2)`.

- **Piecewise-linear trend** with `n_changepoints = 25` candidates placed
  uniformly over the first 80% of the sample (both defaults as upstream),
  written in hinge form `g(t) = m + k t + sum_j delta_j (t - s_j)_+` on time
  scaled to [0, 1].
- **Laplace(0, tau) prior on the deltas** (`tau = 0.05` default =
  upstream's `changepoint_prior_scale`). MAP therefore reduces to
  L1-penalized least squares on the deltas with penalty weight
  `lam = sigma^2 / tau`; `sigma^2 = RSS/n` closes the loop
  (exact block minimization of the joint negative log posterior — the
  derivation is in `model.py`'s docstring).
- **Exact L1 solver**, not L-BFGS-B on a smoothed |.|: the unpenalized block
  (intercept, base slope, Fourier terms, extra regressors) is partialled out
  once via Frisch–Waugh–Lovell, then cyclic coordinate descent with
  soft-thresholding (Friedman–Hastie–Tibshirani 2010) solves the residual
  lasso to its global optimum. Justification: exact zeros in delta (the
  method's "active changepoint" semantics and the tau-path behaviour need
  them), no smoothing width to tune, and with ~25 penalized coordinates the
  exact solve is essentially free. A KKT certificate (`kkt_gap`) is stored
  on every fit; observed gaps are ~1e-10.
- **Fourier seasonality**: yearly (P=365.25 d, K=10) and weekly (P=7 d,
  K=3) auto-enabled for daily date indexes (yearly needs >= 2 years of
  span, weekly needs daily spacing and >= 2 weeks — upstream's enablement
  rules); any regular date spacing is accepted (e.g. weekly CO2 data gets
  yearly seasonality only); plain integer-indexed series take an explicit
  list of `(period, K)` pairs.
- **Optional extra regressors**, standardized as upstream, unpenalized.
- **Prophet's interval scheme** (`uncertainty.py`): future trend
  uncertainty from a changepoint bootstrap — new changepoints arrive as a
  Poisson process with the historical rate (`n_new ~ Poisson(S (T - 1))`),
  magnitudes `~ Laplace(0, mean|delta_hat| + 1e-8)` — plus Gaussian
  observation noise; intervals are empirical quantiles over simulated
  paths at the requested levels.
- **tsecon-style results**: `fit(...)` returns a `dict` subclass of
  documented keys; `.forecast(h, level=[0.8, 0.95])`, `.components()`, and
  `.predictive_draws(h, n_draws, seed)` also exist as module functions
  taking the bare dict, so JSON/pickle round-trips stay fully usable.
- **Deterministic seeding everywhere** — every stochastic call takes an
  explicit `seed`; no module-level RNG state (designed for the comparison
  study: `res.predictive_draws(h, n_draws, seed)` gives reproducible
  simulation ensembles).

## Deliberately omitted vs full Prophet (honest list)

- **MCMC / full-Bayes** (`mcmc_samples > 0` upstream): no parameter
  uncertainty in (k, m, delta, beta) enters the intervals — exactly like
  upstream's default MAP mode, whose intervals carry only the two
  ingredients implemented here.
- **Logistic growth with capacities** (and upstream's flat growth): trend
  is linear-only.
- **Holidays**: no holiday indicators, no country holiday table
  auto-loading, no holiday prior scales.
- **Multiplicative seasonality** and per-component/per-regressor prior
  scales (upstream's Normal(0, 10) seasonality prior is a near-flat ridge
  at MAP; we drop it and estimate the Fourier block unpenalized).
- **Conditional seasonalities, sub-daily seasonality, irregular/missing
  timestamps, dataframe plumbing, plotting, built-in cross-validation**
  (tsecon has `backtest` for that).

## Honest findings / known behaviour

- **tau direction**: tau is the prior *scale*, so `tau -> 0` means an
  infinite L1 penalty (0 active changepoints) and large tau means many
  (observed on the seeded path DGP: 0 / 6 / 23 active at
  tau = 1e-4 / 0.05 / 10). Any statement of the path in terms of a penalty
  *weight* runs in the opposite direction; the tests pin the prior-scale
  parametrization.
- On long, smooth, low-noise series the default tau is a *weak* effective
  penalty (lam = sigma_scaled^2/tau is tiny when sigma << y_scale): on the
  weekly CO2 series all 25 candidates go active with tiny deltas. Upstream
  behaves the same way; it is a property of the default prior, not a solver
  artifact.
- The sigma <-> lasso alternation converged in <= 6 iterations on every DGP
  tried; the joint problem is not jointly convex, so `converged`,
  `n_sigma_iter` and `kkt_gap` are reported rather than assumed.
- Interval calibration on the seeded test DGP: 0.797 empirical coverage at
  the 80% level and 0.950 at 95% (500 future paths x 40 horizons). When the
  future truth contains no new changepoints the bootstrap adds trend
  variance the truth lacks, so mild over-coverage at long horizons is the
  expected direction (a known Prophet property).
- The Poisson rate follows the reference implementation exactly
  (`S` per unit of scaled history) even though the S candidates occupy only
  the first 80% of history — the literal historical frequency would be
  `S/0.8`. Documented in `uncertainty.py`; kept for comparability.

## Provenance / licensing

Implemented from scratch from the published method: Taylor SJ, Letham B
(2018), "Forecasting at Scale", *The American Statistician* 72(1) 37–45
(preprint: PeerJ Preprints 5:e3190v2, 2017). The reference implementation
(facebook/prophet) is MIT-licensed; **no code was copied from it** — a
from-scratch implementation of a published method carries no IP problem.
The lasso solver follows Friedman, Hastie & Tibshirani (2010), *J. Stat.
Software* 33(1) — a published public algorithm. No network data is used
anywhere; tests use seeded synthetic DGPs and the demo uses statsmodels'
bundled CO2 dataset.

## Running it (from the shared venv)

```bash
# tests (8, all seeded)
/home/user/tsecon/.venv/bin/python -m pytest /home/user/tsecon/lab/prophet_lite/tests.py -q

# quick end-to-end demo on statsmodels' bundled CO2 data
cd /home/user/tsecon/lab && /home/user/tsecon/.venv/bin/python -c "
import statsmodels.api as sm
from prophet_lite import fit
co2 = sm.datasets.co2.load_pandas().data['co2'].interpolate()
res = fit(co2.to_numpy(), co2.index.to_numpy())
fc = res.forecast(104, level=[0.8, 0.95], n_draws=1000, seed=0)
print(res['n_active'], fc['mean'][-1], fc['lower']['0.8'][-1], fc['upper']['0.8'][-1])"
```

Usage:

```python
import sys; sys.path.insert(0, "/home/user/tsecon/lab")
from prophet_lite import fit

res = fit(y, dates)                  # dated series (auto seasonality)
res = fit(y, [(12, 3)])              # integer index, explicit (period, K)
res = fit(y, dates, X=X, tau=0.1)    # extra regressors, looser prior

fc    = res.forecast(24, level=[0.8, 0.95], n_draws=1000, seed=0)
comp  = res.components()             # trend / seasonal_* / regressors / residual
draws = res.predictive_draws(24, n_draws=1000, seed=0)   # (1000, 24)
```

Files: `model.py` (design matrices + exact MAP solver, with derivations),
`uncertainty.py` (changepoint bootstrap + noise, interval construction),
`api.py` (`fit`, `ProphetLiteResult`, dict-level functions), `tests.py`.
