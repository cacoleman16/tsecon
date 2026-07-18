# Model card — VAR and structural VAR

`var_fit` · `var_irf` · `var_fevd` · `var_granger` · `var_forecast` ·
`sign_restricted_svar` · `favar` · `connectedness`

The vector autoregression treats a handful of series as one system: every
variable is regressed on the recent past of every variable. From that one
reduced form come forecasts, Granger-causality tests, impulse responses,
variance decompositions, structural shocks, and connectedness measures.

---

## Reduced-form VAR — `var_fit`, `var_irf`, `var_fevd`, `var_granger`, `var_forecast`

**What it estimates.** `var_fit` fits a VAR(p) by equation-by-equation OLS
(coefficient matrix `params`, residual covariance `sigma_u`, information
criteria). The rest read off that fit: `var_irf` traces impulse responses (raw
or Cholesky-orthogonalized), `var_fevd` the forecast-error variance
decomposition, `var_granger` a block Granger-causality F test, `var_forecast`
iterated point forecasts with intervals.

**Assumptions.** Covariance-stationary inputs, correctly chosen lag length,
and — for `orth=True` IRFs and FEVD — that the Cholesky ordering encodes a
defensible contemporaneous recursion (the *first* variable reacts to nothing
within the period). Interval/Granger inference assumes stability.

**When to use (and when not).** Use for multivariate forecasting, testing
predictive precedence, and descriptive dynamics. For a *causal* impulse
response you must identify the system — a bare Cholesky ordering is a strong,
often indefensible assumption; use `sign_restricted_svar` or the local
projection family instead. Do not fit to unit-root levels without thinking:
OLS stays consistent but stability checks, IC comparisons, and Granger
distributions become fragile — difference, or use `johansen`/`vecm` if the
series trend together.

**Key arguments and defaults (and why).** `lags` — set it deliberately
(compare `aic`/`bic`/`hqic` on a *common* sample). `trend="c"` includes an
intercept. `orth=True` orthogonalizes IRFs via Cholesky; `cumulative=True`
reports running sums (level responses). `horizon` and `steps` control length;
`alpha=0.05` sets forecast-interval coverage.

**How to read the output.** `var_fit`: `sigma_u`, `aic/bic/hqic`, and the
stability block — **`is_stable`** (the verdict; read this one), `min_root`, and
`max_root`. These roots are the *inverse* characteristic roots (statsmodels
`VARResults.roots` convention), so stability requires the **smallest** inverse
root to exceed 1 — equivalently all companion eigenvalues inside the unit
circle. `max_root` is the root *farthest* from the unit circle and remains above
1 even for an explosive system, so it is not a stability verdict on its own. `var_irf` returns `[h][response][shock]` (horizon 0..H). `var_fevd`
returns `[h][variable][shock]`, each variable's shares summing to 1.
`var_granger`: `statistic`, `p_value`, `df_num/df_den`. `var_forecast`:
`point`, `lower`, `upper` (each steps×k).

**Failure modes.** A pointwise IRF median is not itself a model at long
horizons; Cholesky ordering silently drives every "structural" reading;
comparing ICs across different effective samples flips rankings; near-unit
roots make long-horizon responses and bands unreliable.

**Validated against.** statsmodels `VAR` — coefficients, `sigma_u`, IRF/FEVD,
`test_causality`, and forecasts; Lütkepohl (2005) textbook conventions
(`fixtures/var.json`).

**References.** Sims (1980); Lütkepohl, *New Introduction to Multiple Time
Series Analysis* (2005).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
k, n = 3, 400
A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.5]])
Y = np.zeros((n, k))
for t in range(1, n):
    Y[t] = A @ Y[t - 1] + 0.3 * rng.standard_normal(k)

fit = tsecon.var_fit(Y, lags=2, trend="c")
print("AIC:", round(fit["aic"], 3))
irf = np.asarray(tsecon.var_irf(Y, lags=2, horizon=10, orth=True))   # [h][resp][shock]
print("IRF shape:", irf.shape, " var0<-shock2 @ h=4:", round(irf[4, 0, 2], 4))
gc = tsecon.var_granger(Y, caused=[0], causing=[2], lags=2)
print("var2 Granger-causes var0? p =", round(gc["p_value"], 4))
```

---

## `sign_restricted_svar` — set identification by sign restrictions

**What it estimates.** A *set* of structural VARs consistent with the data and
a handful of sign restrictions on impulse responses (e.g. "a contractionary
policy shock raises the rate and lowers prices for two quarters"). Draws random
Haar rotations, keeps those whose IRFs satisfy the signs, and summarizes the
survivors — the width of the resulting band **is** the finding.

**Assumptions.** The reduced form is correct; the signs are economically
defensible; and — the caveat a decade of applied work learned — the uniform
(Haar) prior on rotations is *not* uninformative about the responses you care
about (Baumeister-Hamilton 2015), so part of any band is prior, not evidence.

**When to use (and when not).** Use when you have credible sign information but
not enough for a recursive/long-run point identification. Do not stack on
restrictions to narrow the band without watching the acceptance rate; do not
read the pointwise median as "the" IRF (it mixes rotations across horizons).

**Key arguments and defaults.** `restrictions` are `(variable, shock, horizon,
sign)` tuples with sign in `{"+","-"}`. `lags`, `horizon`, `n_draws` (more for
smoother bands), `seed` (reproducible), `max_tries` caps rotation attempts.

**How to read the output.** `quantiles` are per-`(horizon, variable, shock)` at
`probs=[0.05,0.16,0.50,0.84,0.95]`; `set_min`/`set_max` give the identified-set
envelope; `diagnostics["acceptance_rate"]` is itself an identification
diagnostic — a rate near `1e-5` means your "posterior" is a handful of draws
and the restrictions may be near-inconsistent.

**Failure modes.** Acceptance decays roughly exponentially in the number of
restrictions; leaving a response *unrestricted* is the point (its band is the
answer, not an assumption).

**Validated against.** No external golden; validated internally by the
Haar-rotation properties, sign-satisfaction of accepted draws, and the Uhlig
(2005) punchline (an unrestricted output response straddling zero) reproduced
in the [guide](../../guide/08-causal-identification.md).

**References.** Uhlig (2005); Rubio-Ramírez, Waggoner & Zha (2010); Arias,
Rubio-Ramírez & Waggoner (2018, corrected zero+sign).

---

## `favar` — factor-augmented VAR

**What it estimates.** A two-step FAVAR (Bernanke-Boivin-Eliasz 2005): extract
`n_factors` principal components from a large informational panel, then fit a
VAR on `[factors, policy]` with the policy variable ordered last, so a Cholesky
shock to the last equation is the recursive policy shock — mapped back onto
every series in the panel via the factor loadings.

**Key arguments.** `panel` (T×N), `policy` (T,), `n_factors`, `lags`, `trend`,
`slow_indices` (variables that do not respond within the period), `horizon`,
`orth=True`.

**How to read the output.** `factors` (T×r), the VAR `params`/`sigma_u`,
`policy_index` (last equation), `irf_panel` (N×(H+1), one row per series) and
`irf_policy` (the rate's own response). Panel IRFs start at exactly zero on
impact under the recursive ordering, then build with the sign of each series'
loading.

**Validated against.** The factor step against NumPy's PCA/SVD
(`fixtures/favar.json`); the recursive IRF is built on the validated VAR core.

**References.** Bernanke, Boivin & Eliasz (2005); Stock & Watson (2002, factors).

---

## `connectedness` — Diebold-Yilmaz spillovers

**What it estimates.** A directional connectedness table from a VAR's
*generalized* forecast-error variance decomposition (order-invariant, Pesaran-
Shin): who transmits shocks to whom, in percent.

**Key arguments.** `data` (T×k), `lags`, `horizon`, `trend`.

**How to read the output.** `total` (system-wide spillover index), `to_others`
/ `from_others` / `net` (per variable), the `gfevd` matrix, and `pairwise_net`.
Positive `net` marks a net transmitter; negative a net receiver.

**Validated against.** Diebold-Yilmaz (2012) connectedness on a VAR(2) of macro
data, GFEVD row-normalized (`fixtures/connect.json`).

**References.** Diebold & Yilmaz (2012, 2014); Pesaran & Shin (1998, GFEVD).

```python
import numpy as np, tsecon
rng = np.random.default_rng(0)
k, n = 3, 400
A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.5]])
Y = np.zeros((n, k))
for t in range(1, n):
    Y[t] = A @ Y[t - 1] + 0.3 * rng.standard_normal(k)
c = tsecon.connectedness(Y, lags=2, horizon=10)
print("total connectedness:", round(c["total"], 1), "%  net:", np.round(c["net"], 2))
```
