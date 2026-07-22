# Model card — VAR and structural VAR

`var_fit` · `var_irf` · `var_irf_bands` · `var_fevd` · `var_granger` ·
`var_forecast` · `sign_restricted_svar` · `favar` · `connectedness`

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

### Confidence bands on the IRF — `var_irf_bands`

`var_irf` returns the point path only. **`var_irf_bands`** is its banded
companion: same estimand, same `[h][i][j]` layout, but a `dict` with
`point`/`se`/`lower`/`upper` plus the echoed `method`/`alpha`/`n_boot`
(`n_boot` is `None` on the asymptotic branch). Two methods, one flag apart:

- **`method="asymptotic"`** (default) — the Lütkepohl (1990) **delta-method**
  standard errors: the analytic derivative of the MA / orthogonalized responses
  propagated through the estimated coefficient covariance, with symmetric Wald
  bands `point ± z_{1-alpha/2}·se`. These are statsmodels' `irf.stderr`. Closed
  form, no simulation.
- **`method="bootstrap"`** — a residual (Efron/Kilian) recursive-design
  bootstrap: resample the fitted residuals, rebuild the sample through the
  estimated VAR, refit, and read **percentile** bands off the `n_boot` IRF
  draws (`se` is the draw SD). `bias_correct=True` adds the **Kilian (1998)**
  bias correction that the frontier made the frequentist default for persistent
  data. Reproducible through `seed`.

**The orthogonalization caveat.** `orth=True` bands are *not* the reduced-form
bands rescaled. The Cholesky factor $P$ in $\Theta_h = \Phi_h P$ is itself a
function of the estimated $\Sigma_u$, so the delta-method SE of an
orthogonalized response carries an extra term for
$\partial\,\mathrm{vech}(P)/\partial\,\mathrm{vech}(\Sigma_u)$ (and the
bootstrap re-factors $\Sigma_u$ on every draw). `cumulative=True` puts the
bands on the cumulated IRF — delta method via statsmodels `cum_effect_stderr`,
bootstrap by cumulating each draw first.

**The honest caveat.** These are **pointwise** bands: each covers one
$(h, i, j)$ cell at level `alpha`. They are *not* joint/simultaneous over the
horizon, so a reader who traces the whole shaded path is over-reading the
coverage. Sims-Zha (1999) likelihood-shape and Jordà (2009) /
Montiel Olea-Plagborg-Møller (2019) sup-t simultaneous bands remain on the
roadmap; use these for the honest per-horizon uncertainty, not for "does the
path lie in the band with 90% probability".

**Validated against.** statsmodels `VARResults.irf().stderr()` and
`cum_effect_stderr()` (reduced-form and orthogonalized) to machine precision;
the bootstrap by seed reproducibility and Monte-Carlo coverage. See the
[validation matrix](../validation-matrix.md).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
k, n = 3, 400
A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.5]])
Y = np.zeros((n, k))
for t in range(1, n):
    Y[t] = A @ Y[t - 1] + 0.3 * rng.standard_normal(k)

# 90% asymptotic (Lütkepohl delta-method) bands on the orthogonalized IRF
band = tsecon.var_irf_bands(Y, lags=2, horizon=8, orth=True,
                            method="asymptotic", alpha=0.1)
pt = np.asarray(band["point"]); se = np.asarray(band["se"])
lo = np.asarray(band["lower"]); hi = np.asarray(band["upper"])
print("keys:", sorted(band), " n_boot:", band["n_boot"])

# variable 0's response to its OWN shock, h = 0..8, with the 90% band
print(" h   point      se     [ lower ,  upper ]")
for h in range(9):
    print(f" {h}  {pt[h,0,0]:+.4f}  {se[h,0,0]:.4f}  [{lo[h,0,0]:+.4f}, {hi[h,0,0]:+.4f}]")

# bootstrap cross-check at h=1 (residual bootstrap, percentile band)
boot = tsecon.var_irf_bands(Y, lags=2, horizon=8, orth=True,
                            method="bootstrap", alpha=0.1, n_boot=2000, seed=0)
blo = np.asarray(boot["lower"]); bhi = np.asarray(boot["upper"])
print("bootstrap h=1 band",
      f"[{blo[1,0,0]:+.4f}, {bhi[1,0,0]:+.4f}]  vs asymptotic",
      f"[{lo[1,0,0]:+.4f}, {hi[1,0,0]:+.4f}]")
```

```
keys: ['alpha', 'lower', 'method', 'n_boot', 'point', 'se', 'upper']  n_boot: None
 h   point      se     [ lower ,  upper ]
 0  +0.2963  0.0105  [+0.2790, +0.3136]
 1  +0.1584  0.0160  [+0.1321, +0.1847]
 2  +0.0742  0.0155  [+0.0487, +0.0998]
 3  +0.0351  0.0142  [+0.0117, +0.0585]
 4  +0.0174  0.0104  [+0.0003, +0.0345]
 5  +0.0089  0.0070  [-0.0026, +0.0204]
 6  +0.0046  0.0046  [-0.0029, +0.0121]
 7  +0.0024  0.0029  [-0.0024, +0.0072]
 8  +0.0013  0.0018  [-0.0017, +0.0042]
bootstrap h=1 band [+0.1269, +0.1816]  vs asymptotic [+0.1321, +0.1847]
```

The impact response is a clean 0.30 with a band well clear of zero; by $h=5$
the band straddles zero — the response is no longer distinguishable from noise.
The bootstrap band at $h=1$ lands within a whisker of the delta-method band, the
reassurance you want when the asymptotics are the thing being trusted.

**References (bands).** Lütkepohl (1990, asymptotic IRF SEs); Kilian (1998,
bias-corrected bootstrap); Sims & Zha (1999) and Montiel Olea &
Plagborg-Møller (2019) for the simultaneous-band frontier.

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
