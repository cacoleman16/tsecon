# lab/laplace — Laplace-family robust and quantile time series methods

Private research lab code. **Not** part of the tsecon public API, wheel, or
docs site. Exploratory implementations of score-driven quantile filtering and
Laplace/heavy-tail robust estimation, with seeded experiments.

## Contents

| File | Method |
|---|---|
| `al_gas.py` | Score-driven (GAS/DCS) time-varying quantile via the asymmetric-Laplace working likelihood: `q_{t+1} = omega + b q_t + a s_t`, `s_t = tau − 1{y_t ≤ q_t}` (the AL location score). Profiled AL scale ⇒ estimation ≡ pinball-loss minimisation. Multi-tau fitting with crossing check + CFG rearrangement. |
| `robust_filter.py` | DCS local level `mu_{t+1} = mu_t + kappa u_t` with Student-t (Harvey–Luati DCS-t, redescending score), Laplace (bounded sign driver ⇒ median tracking), and Gaussian observation densities. The Gaussian case with unrestricted `kappa` **is** the steady-state Kalman local level — the nested control. |
| `al_arima.py` | ARMA(p,q) with Laplace innovations by conditional (CSS) MLE. Profiled scale ⇒ ≡ LAD/median ARMA; targets the conditional median, robust to heavy tails. Gaussian-CSS twin in the same pipeline as the apples-to-apples benchmark. Stationarity/invertibility via the tanh-PACF (Monahan) reparameterisation. |
| `tests.py` | Seeded pytest suite (7 tests) covering tracking, robustness, nesting, and parameter recovery. |

## What each method is, exactly (references)

- **AL-GAS dynamic quantile.** GAS/DCS recursion (Creal–Koopman–Lucas 2013,
  JAE; Harvey 2013, CUP) applied to the location of an asymmetric-Laplace
  *working* density at level tau. The AL location score is the bounded
  indicator score `(tau − 1{y ≤ q})/sigma`; its Fisher information is constant,
  so all CKL scalings collapse into the loading `a`. The resulting update is
  the *adaptive* CAViaR of Engle & Manganelli (2004, JBES) plus mean
  reversion. AL ⇔ quantile loss: Koenker & Machado (1999, JASA), Yu & Moyeed
  (2001). Score-driven dynamic quantiles of this family are studied by
  Catania & Luati ("Semiparametric modeling of multiple quantiles", Journal of
  Econometrics 2022; drafts ~2019). Rearrangement: Chernozhukov,
  Fernández-Val & Galichon (2010, Econometrica).
- **DCS robust local level.** Harvey & Luati (2014), "Filtering with Heavy
  Tails", JASA 109(507): the DCS-t local level whose redescending score
  discounts additive outliers; Laplace = GED(1) gives the sign/median filter
  (Harvey 2013, ch. 3). Steady-state Kalman background: Durbin & Koopman
  (2012).
- **Laplace (LAD) ARMA.** CSS conditional likelihood (Box–Jenkins) with
  Laplace errors; profiling the scale makes it exactly LAD-ARMA: Davis &
  Dunsmuir (1997, Econometric Theory); Ling (2005, JRSS-B) for very heavy
  tails; Koenker & Bassett (1978) for the median-regression reading.

## Honest simplifications vs the literature

- `al_gas.py`: single-quantile estimation per tau (no *joint* multi-quantile
  system as in Catania–Luati — non-crossing is checked/rearranged, not imposed);
  the indicator score is logistic-smoothed by default (`bandwidth = 0.05·IQR`)
  for optimiser stability — set `bandwidth=0` for the pure indicator model;
  `q_1` initialised at the empirical quantile of the first `max(25, T//10)`
  points. No asymptotic standard errors.
- `robust_filter.py`: filter only (no DCS smoother), constant scale (no
  volatility recursion), level-only component. Gaussian nesting is at the
  *steady-state* filter, so it matches exact Kalman MLE up to transient terms
  (verified numerically to ~1e-3 below). The Laplace sign driver is
  tanh-smoothed by default for the optimiser (`smooth=0` for hard sign). The
  Laplace filter does **not** nest the Gaussian one — only DCS-t does
  (nu → ∞).
- `al_arima.py`: conditional (CSS) likelihood, not the exact unconditional
  one (no closed form under Laplace); `|e|` is smoothed by
  `sqrt(e² + δ²)`, δ = 1e-6·sd(y), inside the optimiser (reported
  loglik/scale use exact `|e|`); no LP-based exact LAD solver; no standard
  errors.
- Optimiser fragility, honestly: with hard indicators/signs (`bandwidth=0`,
  `smooth=0`) L-BFGS-B can stall on kinks — that is why smoothing is the
  default and small deterministic multi-start grids are built in. With the
  defaults, no convergence failures were observed in the seeded experiments.
- DCS-t on 7%-contaminated data drives `nu_hat` to ~1.6 — it uses the fat
  tail to explain the outliers. That is expected behaviour, not a bug, but
  it means `nu_hat` is *not* an estimate of the clean-noise tail index under
  contamination.

## Reproduce

```bash
cd /home/user/tsecon/lab/laplace
/home/user/tsecon/.venv/bin/python -m pytest tests.py -v -s   # full suite, ~55 s
/home/user/tsecon/.venv/bin/python al_gas.py                  # GARCH-quantile demo
/home/user/tsecon/.venv/bin/python robust_filter.py           # outlier demo
/home/user/tsecon/.venv/bin/python al_arima.py                # t(2.5) recovery demo
```

## Headline numbers (seeded; printed by `tests.py`)

- **AL-GAS tracking** (GARCH(1,1), T=3000, tau=0.05, true quantile path
  known): tracking RMSE **0.340** vs static empirical-quantile baseline
  **0.649** — ratio **0.523**; hit rate 0.054 at nominal 0.05. Multi-tau
  (0.05…0.95): raw crossing fraction **0.000**.
- **Robust local level** (7% additive outliers at 9 sd, T=800): level RMSE
  Gaussian **0.418**, DCS-t **0.311** (−26%), DCS-Laplace **0.324** (−22%);
  the Gaussian MLE collapses its gain to 0.016 (vs true-ish 0.10) to absorb
  outliers — exactly the failure the bounded scores avoid.
- **Nesting on clean data** (T=800): DCS-Gaussian `kappa = 0.0974` vs
  steady-state Kalman gain from statsmodels `UnobservedComponents` MLE
  `0.0980` (|diff| 6e-4); path RMSE vs exact Kalman predicted state 0.0013,
  and 0.0013 vs `tsecon.local_level_smooth` at the UC-MLE variances;
  DCS-t collapses to Gaussian (`nu_hat` at the 200 bound, path RMSE vs
  Gaussian path 0.002).
- **Laplace ARMA recovery** (ARMA(1,1), phi=0.6, theta=0.3, T=400, 30 reps;
  joint RMSE ratio Laplace/Gaussian): **0.688** under t(2.5) innovations,
  **0.751** under Laplace innovations, **1.304** under Gaussian innovations
  (20 reps) — consistent with the asymptotic LAD/LS ARE of pi/2. Gaussian-CSS
  twin checks out against statsmodels exact MLE (phi 0.5645 vs 0.5629,
  theta 0.4170 vs 0.4179 on a seeded series).

## Provenance

All methodology implemented from the published academic literature cited
above and in each module docstring. No proprietary code consulted. Research
code: no API-stability promises, not exported by `tsecon`.
