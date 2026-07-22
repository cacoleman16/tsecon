# Model card — Panel unit-root tests

**Family:** `panel_unit_root` (Levin-Lin-Chu, Im-Pesaran-Shin, Fisher/Maddala-Wu-Choi)

One entry point, three classic tests of the joint null **"every cross-section
unit has a unit root."** Stacking many short series into a panel buys the power
a single 60-quarter history cannot: where one ADF test on one country barely
distinguishes a persistent stationary series from a random walk, pooling or
averaging the evidence across `N` units sharpens the verdict enormously. All
three tests share the same front half — the already-validated per-unit augmented
Dickey-Fuller regression ([`adf`](diagnostics.md), matched to statsmodels at
`1e-8`) — and differ only in how they combine the per-unit results.

| `test` | Combines | Null / alternative | Panel shape |
|--------|----------|--------------------|-------------|
| `"llc"` | pooled ADF, common ρ | all units I(1) / **all** units share one stationary root | balanced (common T) |
| `"ips"` (default) | average of per-unit t-statistics | all units I(1) / **some** units stationary | unbalanced OK |
| `"fisher"` | combines per-unit p-values | all units I(1) / **some** units stationary | unbalanced OK |

## What it estimates

- **`panel_unit_root(data, test="llc")`** — Levin-Lin-Chu (2002): a **pooled**
  ADF with a single common autoregressive root δ across units. It removes
  per-unit means/trends, scales each unit by its own long-run-variance ratio,
  runs one pooled OLS for δ, and applies the tabulated `mu*`/`sigma*` bias
  adjustment (LLC Table 2) to give `t*_delta ~ N(0,1)` in the left tail. The
  homogeneous-alternative test: it asks whether *the* common root is below one.
- **`panel_unit_root(data, test="ips")`** — Im-Pesaran-Shin (2003): fit a
  **separate** ADF to each unit, average the t-statistics into `t_bar`, then
  standardize to `W_tbar ~ N(0,1)` with the tabulated mean/variance of the
  individual ADF t-statistic (IPS Table 3). The heterogeneous-alternative test:
  it allows each unit its own root and rejects when *enough* units are
  stationary.
- **`panel_unit_root(data, test="fisher")`** — the Fisher-type combination
  (Maddala-Wu 1999; Choi 2001): from the per-unit ADF p-values `p_i`, form
  Maddala-Wu's `P = -2 Σ ln p_i ~ χ²(2N)` (right tail) and Choi's inverse-normal
  `Z = N^{-1/2} Σ Φ⁻¹(p_i) ~ N(0,1)` (left tail). A deterministic,
  exact-arithmetic function of the per-unit p-values — it inherits the ADF's
  validated accuracy directly and is the crate's strongest-anchored panel
  statistic.

## Assumptions

- **Cross-sectional independence — the defining caveat of the whole family.**
  All three are *first-generation* tests: their asymptotic null distributions
  are derived assuming the units' errors are independent across `i`. When a
  common factor (a global business cycle, a common shock) correlates the units
  at each date — the rule, not the exception, in macro panels — the true
  variance of the pooled/averaged statistic is wrong, and the tests suffer size
  distortion (typically over-rejection). This is a genuine limitation, not a
  small-sample nuisance; the **second-generation** remedies (Pesaran's 2007
  CIPS, Bai-Ng PANIC, Moon-Perron) are on the roadmap. Before trusting a
  rejection, ask whether a common factor is plausible (Pesaran's CD test is the
  standard pre-check), and treat a first-generation "stationary" verdict on a
  strongly co-moving panel with suspicion.
- **LLC imposes a *common* root.** Its alternative is that **every** unit is
  stationary with the *same* δ. If the panel mixes stationary and unit-root
  units, or the stationary units have very different roots, LLC is
  misspecified — IPS or Fisher, whose alternative is heterogeneous, are the
  honest choice.
- **The deterministic specification matters.** `regression="c"` (constant) vs
  `"ct"` (constant + trend) changes the ADF null distribution and the tabulated
  moments; choose it to match the data's trend behavior, exactly as for a
  single-series ADF. `"n"` (no deterministic term) is invalid for IPS (the IPS
  moment tables are tabulated only for the `c`/`ct` cases).
- **Lag augmentation per unit.** Each unit's ADF needs enough lagged
  differences to whiten its residuals; too few and the size is wrong, too many
  and power bleeds away. The default per-unit auto-AIC handles heterogeneous
  serial correlation, but cap `max_lags` on short `T` so a unit cannot spend its
  degrees of freedom on spurious lags.

## When to use

- **`"ips"`** — the default, and the right first move for most macro panels:
  heterogeneous alternative, unbalanced panels allowed, and a transparent
  average of per-unit evidence. Reach for it when you believe *some* units may
  be stationary and others not.
- **`"fisher"`** — when you want the most robust p-value combination, unbalanced
  panels, and a statistic whose accuracy is inherited directly from the
  validated single-series ADF. Choi's Z is the better-sized variant for large
  `N`; Maddala-Wu's χ² is the classic.
- **`"llc"`** — when theory or design genuinely implies a *common* root across
  units (e.g. the same law of motion imposed on every unit), and the panel is
  balanced. Its pooling buys power, but only if the homogeneity is real.
- **Not a substitute for the ADF+KPSS quadrant per unit.** If you care about
  *which* units are nonstationary, run the single-series workflow
  ([`check_stationarity`](diagnostics.md)) unit by unit; the panel test answers
  the *joint* question only.

## Key arguments and defaults

| Argument | Default | Notes |
|----------|---------|-------|
| `data` | — | balanced `N × T` array (rows = units) **or** a list of 1-D per-unit series (unbalanced OK for `ips`/`fisher`; `llc` needs a common length) |
| `test` | `"ips"` | `"ips"`, `"llc"`, or `"fisher"` |
| `regression` | `"c"` | `"c"` constant, `"ct"` constant + trend, `"n"` none (`"n"` invalid for `ips`) |
| `lags` | `None` | `None` = per-unit auto-AIC; an `int` = a fixed common lag; or `"aic"`/`"bic"`/`"t-stat"` |
| `max_lags` | `None` | cap on the per-unit auto lag search |
| `lrv_kernel` / `lrv_bandwidth` | `"bartlett"` / `None` | the long-run-variance kernel LLC uses to scale each unit; ignored by IPS/Fisher |

## How to read the output

Every call returns `statistic`, `p_value`, the per-unit vectors
`per_unit_tstat` / `per_unit_pvalue` / `per_unit_lags` / `per_unit_nobs`,
`n_units`, and the echoed `regression`, plus test-specific extras:

- **`"ips"`** → `t_bar` (the raw average per-unit t-statistic, before
  standardization); `statistic` is the standardized `W_tbar ~ N(0,1)`.
- **`"llc"`** → `delta_hat` (the estimated common root adjustment), `t_delta`
  (the raw pooled t before the bias adjustment), `s_n` (the average long-run to
  short-run standard-deviation ratio), and `t_bar_periods`; `statistic` is the
  bias-adjusted `t*_delta ~ N(0,1)`.
- **`"fisher"`** → `maddala_wu` (the `χ²(2N)` statistic, equal to `statistic`),
  `choi_z` (the inverse-normal statistic), and `choi_z_pvalue`. The Maddala-Wu
  `p_value` is the right-tail χ²; Choi's is the left-tail normal.

A small `p_value` **rejects** the joint unit-root null in favor of (at least
some) stationarity. Inspect `per_unit_pvalue` to see which units drive the
verdict, and `per_unit_lags` / `per_unit_nobs` to check the augmentation and
effective sample were sane per unit.

## Failure modes

- **Reading a rejection as "the panel is stationary" under cross-sectional
  dependence.** A common factor inflates the test's size; the rejection may be
  the factor talking, not stationarity. This is the family's headline weakness —
  check for cross-sectional dependence first, and prefer the second-generation
  tests (roadmap) when it is present.
- **LLC on a heterogeneous panel.** Its common-root alternative is
  misspecified when units have different dynamics; the pooled δ is a
  hard-to-interpret average and the verdict can mislead. Use IPS/Fisher.
- **`regression="n"` with IPS.** Raises — the IPS moment tables exist only for
  `c`/`ct`. This is a correct refusal, not a bug.
- **Unbalanced input to LLC.** LLC needs a common `T`; hand it ragged per-unit
  series and it raises. Balance the panel or switch to IPS/Fisher.
- **Over-long auto lags on short `T`.** Uncapped AIC can pick a large lag on a
  short unit, distorting its ADF; set `max_lags`.

## Validated against

Golden fixtures reproduce R **`plm::purtest`**'s `Wtbar` (IPS), `levinlin`
(LLC), and `madwu` / `invnormal` (Fisher) statistics to floating-point
precision, and — for Fisher — an independent statsmodels/SciPy reference on the
p-value combination. The per-unit ADF is reused verbatim from `tsecon-diag`
(matched to statsmodels `adfuller` at `1e-8`); only the combination layer and
the transcribed IPS-2003 Table 3 and LLC-2002 Table 2 moment families are new.
Fixture: [`tsecon-panelroot.json`](../../../fixtures/tsecon-panelroot.json);
tests: [`golden.rs`](../../../crates/tsecon-panelroot/tests/golden.rs),
[`validation.rs`](../../../crates/tsecon-panelroot/tests/validation.rs),
[`properties.rs`](../../../crates/tsecon-panelroot/tests/properties.rs). See the
[validation matrix](../validation-matrix.md).

## References

- Levin, A., Lin, C.-F. & Chu, C.-S. J. (2002). "Unit root tests in panel data:
  asymptotic and finite-sample properties." *J. Econometrics* 108.
- Im, K. S., Pesaran, M. H. & Shin, Y. (2003). "Testing for unit roots in
  heterogeneous panels." *J. Econometrics* 115.
- Maddala, G. S. & Wu, S. (1999). "A comparative study of unit root tests with
  panel data and a new simple test." *Oxford Bull. Econ. Stat.* 61.
- Choi, I. (2001). "Unit root tests for panel data." *J. Int. Money Finance* 20.
- Pesaran, M. H. (2007). "A simple panel unit root test in the presence of
  cross-section dependence." *J. Applied Econometrics* 22. (The
  second-generation CIPS answer to the independence caveat — roadmap.)

See the guide: [Panel Time Series](../../guide/14-panel-time-series.md).

## Runnable example

```python
import numpy as np
import tsecon

rng = np.random.default_rng(31)
N, T = 12, 80

# ---- a STATIONARY panel: each unit an AR(1) with its own |rho| < 1 ----
stat = np.empty((N, T))
for i in range(N):
    rho = rng.uniform(0.5, 0.85)
    u = np.empty(T); u[0] = rng.standard_normal()
    for t in range(1, T):
        u[t] = rho * u[t - 1] + rng.standard_normal()
    stat[i] = u

# ---- a UNIT-ROOT panel: each unit its own independent random walk ----
rw = np.cumsum(rng.standard_normal((N, T)), axis=1)

print("STATIONARY panel (each row an AR(1), |rho|<1) -- should reject:")
for test in ["ips", "llc", "fisher"]:
    r = tsecon.panel_unit_root(stat, test=test, regression="c", max_lags=4)
    print(f"  {test:6s} stat={r['statistic']:+9.4f}  p={r['p_value']:.3g}")

print("UNIT-ROOT panel (each row a random walk) -- should NOT reject:")
for test in ["ips", "llc", "fisher"]:
    r = tsecon.panel_unit_root(rw, test=test, regression="c", max_lags=4)
    print(f"  {test:6s} stat={r['statistic']:+9.4f}  p={r['p_value']:.3g}")

# test-specific extras (each test carries its own named quantities)
ips = tsecon.panel_unit_root(stat, test="ips", regression="c", max_lags=4)
print("IPS   t_bar:", round(ips["t_bar"], 4), " n_units:", ips["n_units"])
fis = tsecon.panel_unit_root(stat, test="fisher", regression="c", max_lags=4)
print("Fisher Maddala-Wu:", round(fis["maddala_wu"], 3),
      " Choi Z:", round(fis["choi_z"], 4), " Choi p:", round(fis["choi_z_pvalue"], 3))
llc = tsecon.panel_unit_root(stat, test="llc", regression="c", max_lags=4)
print("LLC   delta_hat:", round(llc["delta_hat"], 4), " t_delta:", round(llc["t_delta"], 4))

# unbalanced list input for IPS/Fisher (LLC needs a common T)
unb = [rw[i, : T - int(rng.integers(0, 12))] for i in range(N)]
ru = tsecon.panel_unit_root(unb, test="ips", regression="c", max_lags=4)
print("unbalanced IPS p:", round(ru["p_value"], 3),
      " nobs range:", int(np.min(ru["per_unit_nobs"])), "-", int(np.max(ru["per_unit_nobs"])))
```

Expected output:

```
STATIONARY panel (each row an AR(1), |rho|<1) -- should reject:
  ips    stat=  -7.0591  p=8.38e-13
  llc    stat=  -5.9169  p=1.64e-09
  fisher stat=+105.9066  p=2.9e-12
UNIT-ROOT panel (each row a random walk) -- should NOT reject:
  ips    stat=  -0.0954  p=0.462
  llc    stat=  -0.2452  p=0.403
  fisher stat= +22.0614  p=0.576
IPS   t_bar: -3.2816  n_units: 12
Fisher Maddala-Wu: 105.907  Choi Z: -7.3878  Choi p: 0.0
LLC   delta_hat: -0.2375  t_delta: -10.8384
unbalanced IPS p: 0.288  nobs range: 68 - 79
```

All three tests strongly reject the joint unit-root null on the stationary panel
(p ≈ 1e-12) and comfortably fail to reject it on the independent random walks
(p ≈ 0.4-0.6) — the power the panel buys, and the size it keeps *when the units
are independent*. The last two calls show the unbalanced-list input path
(different `nobs` per unit) that IPS and Fisher accept but LLC does not.
