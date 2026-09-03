# Model card — Structured penalties and post-selection

**Family:** `group_lasso`, `post_lasso`, `pds_lasso`

The second layer of the penalized-regression stack: penalties that respect
the *structure* of a time-series design (all lags of one predictor as one
unit), and the two things people do after a LASSO has chosen — refit it, and
try to do inference on it. The first is safe if you know what you are
looking at; the second is where the literature's loudest warning lives, and
this card measures it rather than repeating it.

| Function | Role |
|----------|------|
| `group_lasso` | Group LASSO (Yuan-Lin 2006) and sparse-group LASSO (Simon et al. 2013): select whole groups, optionally thinning them inside, with a readable optimality certificate |
| `post_lasso` | Post-LASSO OLS refit (Belloni-Chernozhukov 2013): the shrinkage removed, **no standard errors by design** |
| `pds_lasso` | Post-double-selection (Belloni-Chernozhukov-Hansen 2014) for one treatment coefficient among high-dimensional controls, with Newey-West HAC inference |

## What it estimates

- **`group_lasso(x, y, groups, alpha, l1_ratio=0.0, group_weights="sqrt_size")`**
  minimizes

  ‖y − Xβ‖²/(2n) + α [ (1 − l1_ratio) Σ_g w_g ‖β_g‖₂ + l1_ratio ‖β‖₁ ].

  `groups` is one integer label per column — any integers, contiguous or
  not; columns sharing a label form a group. With `l1_ratio = 0` the group
  norm zeroes *whole* groups (a predictor's lag block enters or leaves
  together); with `0 < l1_ratio < 1` the sparse-group penalty also zeroes
  coordinates inside surviving groups; with `l1_ratio = 1` the group term
  vanishes and the call **is** `lasso(x, y, alpha)` — same 1/(2n) scaling,
  same `alpha` scale as scikit-learn. `group_weights="sqrt_size"` is the
  Yuan-Lin `w_g = √|g|` that puts unequal groups on the same footing;
  `"none"` is `w_g = 1`; an array gives one positive weight per distinct
  label in ascending label order.
- **`post_lasso(x, y, alpha, l1_ratio=1.0)`** fits `elastic_net`'s objective,
  takes the nonzero support, and refits OLS on those columns (the
  minimum-norm least-squares fit, so a collinear support does not fail). It
  returns the support, both coefficient vectors, and the refit RSS — and
  nothing that looks like a standard error.
- **`pds_lasso(y, d, x, alpha="bic", hac_lags=None)`** estimates τ in
  y = τ d + x'β + e with a scalar treatment `d` and p controls `x` (p > n is
  fine): LASSO of y on x (support S_y), LASSO of d on x (support S_d), OLS
  of y on [d, x_{S_y ∪ S_d}] with a Newey-West (Bartlett) sandwich, reading
  τ off `d`. The treatment is never penalized. `alpha="bic"` takes the
  per-equation BIC minimizer along `lasso_path`'s default grid; a float is
  applied to both equations.

## Assumptions

- **No intercept, no standardization — the most important line on this
  card, inherited from the family.** Every routine here fits the objective
  on the design *exactly as passed*. Center `y` (and `d`), standardize the
  columns of `x`. A group penalty is even more scale-sensitive than the
  plain LASSO: a group whose columns carry large units is barely penalized
  relative to one in small units.
- **Group weights are part of the estimator.** `w_g = √|g|` is the
  literature default and what the fixture was generated with; the
  sparse-group mixing and the weights interact (Simon et al. 2013, §3), so
  state both when reporting a fit.
- **Convergence is certified, not assumed.** The roadmap's warning is that
  the wrong per-block Lipschitz constant or the wrong prox order "converges
  smoothly to the wrong answer with no error raised". `group_lasso`
  therefore solves each block by proximal gradient with the exact
  `L_g = λ_max(X_g'X_g)/n`, applies Simon et al.'s exact group-zero test,
  and declares `converged` only when the subgradient Karush-Kuhn-Tucker
  residual it reports as `kkt_violation` is at or below
  `tol · max_j |x_j'y|/n`. The problem is convex, so a small residual is a
  proof of near-optimality whatever the solver did — read it.
- **Post-LASSO is a point estimate.** The selection event depends on the
  same sample; OLS standard errors on the selected columns are invalid
  (Leeb & Pötscher 2005), and `n − |S|` overstates the residual degrees of
  freedom after searching over p columns. `post_lasso` does not compute
  them, so nothing on this surface can be mistaken for inference.
- **PDS inference is asymptotic.** `p_value` and `conf_int` use the standard
  normal in both the HAC and the classical mode (statsmodels `use_t=False`),
  matching the Belloni-Chernozhukov-Hansen theory. The HAC covariance
  carries the `n/(n − k)` finite-sample factor (statsmodels
  `use_correction=True`; statsmodels' *own* default is `False`, so pass it
  explicitly when comparing). `hac_lags=None` resolves to the Newey-West
  rule `⌊4 (n/100)^{2/9}⌋`; `hac_lags=0` is the classical spherical-errors
  covariance, not a robust one.
- **PDS needs approximate sparsity in both equations.** The guarantee is
  that controls omitted by *both* LASSOs are small in *both* equations; a
  control that matters a lot for `y` and a lot for `d` and is dropped by
  both is a bias nothing here can see. There is no amelioration set
  (mandatory unpenalized controls) in this release: put such controls in
  by partialling them out of `y`, `d` and `x` first.

## When to use

- **`group_lasso`** when predictors come in blocks that should enter or
  leave together — all lags of one variable in an ARDL/VAR equation, a
  block of dummies, the columns of one factor. Use `l1_ratio > 0` when a
  block should survive as a whole but not every lag inside it is needed.
- **`post_lasso`** when the LASSO's *coefficients* are going to be read
  economically (a multiplier, an elasticity) rather than used for
  prediction: the refit removes the shrinkage. Do not put a standard error
  next to the number; see `pds_lasso`.
- **`pds_lasso`** when a single coefficient is the target and the controls
  are many — a policy variable, a shock, a treatment indicator — and you
  need an interval that survives having let the data choose the controls.
  Selecting on the treatment equation too is what makes it work; the
  "Failure modes" below measures what happens without it.

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `group_lasso` | `groups` | — (required) | one integer label per column; integer arrays pass through coercion untouched |
| | `alpha` | — (required) | penalty on scikit-learn's 1/(2n) scale; `alpha_max` is returned |
| | `l1_ratio` | `0.0` | 0 = group LASSO, 1 = `lasso`, between = sparse-group |
| | `group_weights` | `"sqrt_size"` | `"sqrt_size"`, `"none"`, or one positive weight per distinct label (ascending label order) |
| | `tol` / `max_iter` | `1e-8` / `10000` | `lasso`'s dimensionless coefficient-change rule, also bounding the KKT residual; sweeps of the block cycle |
| `post_lasso` | `alpha`, `l1_ratio` | — / `1.0` | exactly `elastic_net`'s |
| | `tol` / `max_iter` | `1e-8` / `100000` | first-stage coordinate descent |
| `pds_lasso` | `alpha` | `"bic"` | `"bic"` (per-equation BIC along `lasso_path`'s grid) or one float for both equations |
| | `hac_lags` | `None` | `None` → `⌊4 (n/100)^{2/9}⌋`; positive → that Bartlett truncation; `0` → classical standard errors |
| | `tol` / `max_iter` | `1e-8` / `100000` | selection-stage coordinate descent |

## How to read the output

- **`group_lasso`** → `{"coef", "n_iter", "converged", "active_groups",
  "active_set", "objective", "kkt_violation", "max_rel_change",
  "alpha_max"}`. `active_groups` lists the *labels* with a nonzero block;
  `active_set` the nonzero column indices. `converged=False` is not an
  exception: the last iterate comes back and `kkt_violation` says how far
  from optimal it is (on a converged return it is ≤ `tol · max_j|x_j'y|/n`;
  the fixture achieves ~2e-13). `alpha_max` is the top of the path — the
  fit is identically zero at and above it.
- **`post_lasso`** → `{"support", "coef_lasso", "coef_ols", "n_selected",
  "rss"}`. `coef_ols` is exactly zero off-support.
- **`pds_lasso`** → `{"coef", "se", "t_stat", "p_value", "conf_int",
  "support_y", "support_d", "union_support", "n_controls_selected",
  "alpha_y", "alpha_d", "hac_lags_resolved"}`. `coef` is τ. Look at the two
  supports: a control in `support_d` but not `support_y` is exactly the
  kind single selection would have dropped.

## Failure modes

- **Un-standardized groups.** The group norm inherits each column's units,
  so a block of large-valued columns is under-penalized relative to the
  rest. Standardize; the fixture does.
- **Trusting a smooth trajectory.** A block solver with a mis-set step
  looks converged — coefficient changes shrink geometrically — while
  sitting at the wrong point. That is why `converged` is gated on the KKT
  certificate and why `kkt_violation` is returned: if you tighten `tol`
  and the number does not fall, something is wrong with the *inputs* (a
  NaN would have been refused; a constant column is pinned at zero).
- **Standard errors after a single selection — measured.** On a seeded
  design with n = 400, p = 40 AR(1) controls (ρ = 0.5), AR(1) errors in
  both equations (ρ = 0.3), τ = 1, and four confounders that load strongly
  on `d` (γ = ±1) but weakly on `y` (β = ±0.15), the Belloni-Chernozhukov-
  Hansen *single-selection* comparator — LASSO of `y` on `x` with `d`
  unpenalized at its BIC pick, then OLS of `y` on `[d, x_S]` with the same
  Newey-West interval — covers τ **0.003** of the time at a nominal
  0.95 (300 replications, Monte-Carlo s.e. 0.013); the omitted
  confounders bias τ̂ by about a standard error. `pds_lasso` on the same
  draws covers **0.950**, indistinguishable from the infeasible oracle
  that knows the true support (**0.953**), selecting 10.6
  controls on average. The single-selection interval is not a
  conservative-but-usable approximation; it is wrong by a factor that does
  not shrink with n.
- **HAC's own small-sample shortfall — also measured, so it is not
  mistaken for a selection failure.** Raising the persistence to ρ = 0.5 in
  both error processes at n = 200 (same p, same loadings), the oracle's
  Newey-West interval at the rule-of-thumb bandwidth (4 lags) covers only
  **0.930**; `pds_lasso` covers **0.903** — tracking the oracle
  to within Monte-Carlo noise — while single selection covers
  **0.153**. The gap between 0.95 and the oracle is the Bartlett
  estimator's downward bias under this much score autocorrelation (the
  library's interval-coverage audit documents the same effect on `lp`), and
  it is inherited from the HAC engine rather than added by the selection
  step. Under persistent errors, treat the rule-of-thumb bandwidth as a
  floor and consider a longer `hac_lags`. Both 300-replication cells live
  in `pds_coverage_full_measurement` (ignored by default; reproduce with
  `cargo test -p tsecon-ml --release --test structured_properties --
  --ignored --nocapture`); the always-on test cell (n = 200, p = 16, 80
  replications, MC s.e. 0.024) asserts the same ordering on every run and
  measures PDS **0.938**, oracle **0.963**, single selection **0.075**.
- **`hac_lags=0` is classical, not heteroskedasticity-robust.** It is the
  spherical-errors covariance; there is no HC option on this surface.
- **Weak-signal groups near `alpha_max`.** Just below `alpha_max` a group
  enters with a tiny norm; its within-group pattern is not meaningful.
  Read `active_groups` along a grid of `alpha`, not at one value.

## Validated against

- **`group_lasso` — two grades, both stated.** *Optimality certificate*
  (primary): for every fixture case an **independent** evaluation of the
  subgradient KKT conditions (inactive groups
  ‖S(−∇_g, α l1)‖₂ ≤ α(1 − l1) w_g; active coordinates
  ∇_j + α(1 − l1) w_g β_j/‖β_g‖ + α l1 sign(β_j) = 0; zero coordinates of
  active groups |∇_j| ≤ α l1) written in the test, asserted ≤ 1e-8,
  achieved **2.3e-13** — rigorous for a convex problem. *Independent
  package*: **skglm 0.5** (`GroupLasso` for `l1_ratio = 0`, the
  `WeightedL1GroupL2` penalty with `GroupBCD` for the sparse-group cases,
  run to `tol = 1e-12`; its objective uses the same 1/(2n) scaling), ten
  cases over two designs (contiguous blocks with correlated columns;
  scattered non-contiguous labels of unequal sizes; three weight
  conventions; `l1_ratio ∈ {0, 0.2, 0.3, 0.5, 0.9, 1}`), asserted 1e-8,
  achieved **1.5e-12** — bounded by skglm's own convergence, whose KKT
  residual the fixture records (worst 5.7e-13). The `l1_ratio = 1` case is
  pinned to scikit-learn `Lasso` and reproduces the crate's `lasso` at
  1e-8 (achieved ~1e-13), as do singleton groups at `l1_ratio = 0` and
  custom-weighted singletons (a column-rescaled `lasso`). `alpha_max`
  matches a NumPy transcription at 1e-12 and zeroes the fit just above.
  The `group-lasso` (Moe) package was also installed and evaluated; its
  FISTA solver reaches only ~1e-4 and is not used as a reference.
- **`post_lasso`** — scikit-learn `LinearRegression(fit_intercept=False)` on
  the scikit-learn `Lasso`/`ElasticNet` support, three cases, support exact,
  refit asserted 1e-10, achieved **8.4e-15**; the first stage agrees with
  scikit-learn at 1.5e-12.
- **`pds_lasso`** — *exact leg*: statsmodels `OLS(...).fit(cov_type="HAC",
  cov_kwds={"maxlags": L, "use_correction": True}, use_t=False)` and
  `cov_type="nonrobust", use_t=False` on `[d, X_union]`, with the union
  forced to all 30 controls (L = 4, 8, 0) and with the BIC-selected union
  (L = 4, 0; `alpha_y`/`alpha_d` pinned at 1e-10 against scikit-learn
  `lasso_path` on the same grid): `coef`, `se`, `t_stat`, `conf_int`
  asserted 1e-8 relative, achieved **9.8e-15**; `p_value` 1e-12. *Coverage
  leg*: **Monte-Carlo grade** — R `hdm` and Stata `pdslasso` are not
  runnable in the reference environment, so the statistical claim is
  carried by the seeded experiment quoted above, in
  [`structured_properties.rs`](../../../crates/tsecon-ml/tests/structured_properties.rs).

Fixture: [`fixtures/structured.json`](../../../fixtures/structured.json)
(generator: `fixtures/generate_structured_fixtures.py`, which never imports
tsecon).

## References

- Yuan, M. & Lin, Y. (2006). "Model selection and estimation in regression
  with grouped variables." *JRSS-B* 68(1).
- Simon, N., Friedman, J., Hastie, T. & Tibshirani, R. (2013). "A
  sparse-group lasso." *JCGS* 22(2).
- Belloni, A. & Chernozhukov, V. (2013). "Least squares after model
  selection in high-dimensional sparse models." *Bernoulli* 19(2).
- Belloni, A., Chernozhukov, V. & Hansen, C. (2014). "Inference on
  treatment effects after selection among high-dimensional controls."
  *Review of Economic Studies* 81(2).
- Leeb, H. & Pötscher, B. (2005). "Model selection and inference: facts
  and fiction." *Econometric Theory* 21(1).
- Bertrand, Q., Klopfenstein, Q., Bannier, P.-A., Gidel, G. & Massias, M.
  (2022). "Beyond L1: faster and better sparse models with skglm."
  *NeurIPS* (the cross-package reference).

See also the family card:
[Penalized regression and leakage-safe validation](machine-learning.md).

## Runnable example

```python
import numpy as np
import tsecon

rng = np.random.default_rng(5)
n, p = 300, 12
X = rng.standard_normal((n, p))
# Three lag-blocks of four; only the first and third matter.
groups = np.repeat([0, 1, 2], 4)
beta = np.array([1.5, -0.9, 0.6, 0.0,  0, 0, 0, 0,  0.8, 0.0, -0.5, 0.3])
y = X @ beta + 1.2 * rng.standard_normal(n)

# No intercept, no standardization inside: center y, standardize X.
Xs = (X - X.mean(0)) / X.std(0)
yc = y - y.mean()

# 1. Group LASSO: whole blocks enter or leave.
gl = tsecon.group_lasso(Xs, yc, groups, alpha=0.15)
print("active groups:", gl["active_groups"], " converged:", gl["converged"],
      " KKT residual: %.1e" % gl["kkt_violation"])

# 2. Sparse-group: blocks survive, single lags inside them can still go.
sgl = tsecon.group_lasso(Xs, yc, groups, alpha=0.1, l1_ratio=0.5)
print("sparse-group nonzeros:", int(np.sum(sgl["coef"] != 0)), "/", p)

# 3. Post-LASSO: the refit removes shrinkage (no standard errors, on purpose).
pl = tsecon.post_lasso(Xs, yc, alpha=0.1)
print("selected:", pl["support"], " refit on x0: %.3f (lasso %.3f)"
      % (pl["coef_ols"][0], pl["coef_lasso"][0]))

# 4. Post-double-selection for one treatment among many controls.
d = X[:, :4] @ np.array([1.0, 1.0, -1.0, 1.0]) + rng.standard_normal(n)
y2 = 1.0 * d + X @ beta + rng.standard_normal(n)
pds = tsecon.pds_lasso(y2 - y2.mean(), d - d.mean(), Xs)
print("tau = %.3f  se = %.3f  95%% CI = (%.3f, %.3f)  controls: %d  lags: %d"
      % (pds["coef"], pds["se"], *pds["conf_int"], pds["n_controls_selected"],
         pds["hac_lags_resolved"]))
```

Expected output:

```
active groups: [0, 2]  converged: True  KKT residual: 8.9e-12
sparse-group nonzeros: 6 / 12
selected: [0, 1, 2, 8, 10, 11]  refit on x0: 1.567 (lasso 1.451)
tau = 0.959  se = 0.059  95% CI = (0.844, 1.074)  controls: 8  lags: 5
```
