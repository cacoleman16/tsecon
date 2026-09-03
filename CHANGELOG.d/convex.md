### Added

- **L1 trend filtering (`l1_trend_filter`)** — Kim, Koh & Boyd's (2009)
  piecewise-linear trend with data-chosen knots (roadmap Module 10, Tier
  2), in `tsecon-ml`. Minimizes `(1/2)‖y − x‖² + lam·‖D x‖₁` with `D` the
  `order`-th difference operator: `order=2` gives a piecewise-linear
  trend whose kinks are the `knots`, `order=1` a piecewise-constant one
  (the fused LASSO on the level). `penalty="l2"` swaps in the squared
  penalty `(lam/2)‖D x‖²`, which for `order=2` **is** the Hodrick-Prescott
  filter under HP's own `lam` — a cross-surface identity the suite asserts
  against `hp_filter` at 1e-10. Solver: Kim-Koh-Boyd's primal-dual
  interior-point method on the **banded dual** — every Newton step is an
  O(n) banded `LDLᵀ` factorization, no n×n matrix anywhere — followed by
  an exact active-set polish that zeros the inactive differences to
  rounding; the closed-form limits `lam = 0` (the data) and `lam ≥
  lam_max = ‖(DDᵀ)⁻¹Dy‖_∞` (the least-squares polynomial of degree
  `order − 1`) are returned directly, and `lam_max` is a returned key.
  Every fit carries a **certificate**: `duality_gap` is the primal
  objective at the returned trend minus a dual-feasible dual objective,
  so `objective − optimum ≤ duality_gap` by weak duality; `converged` is
  `duality_gap ≤ tol·objective`, and a starved budget or a `tol` below
  the certificate's floating-point floor (~1e-11 relative; a stall
  detector ends the loop in ~40 iterations instead of burning the
  budget) returns `converged=False` with the honest gap. Validation,
  graded per leg: the tests re-derive the KKT certificate for the
  crate's own trend from scratch on 14 fixture cases and assert a
  relative gap ≤ 1e-8 (achieved ≤ 3.3e-10; 1e-15 on order-1 cases);
  **cvxpy 1.9.2 + Clarabel 0.11.1** third-party trends converged at 1e-14 agree at
  1e-8 (achieved 1.4e-10); `lam_max` and the polynomial limit at 1e-10 /
  1e-8 (achieved 1.4e-10); the L2 form against the dense closed form at
  1e-10 (achieved 1.7e-12). `tol` / `max_iter` are inert under
  `penalty="l2"` (a closed-form solve) and follow the sentinel
  convention: explicitly passed there raises naming the kwarg; the
  default call is bit-identical; both are live under `"l1"`. Wall time
  (release wheel): 0.12 s at `n = 10000` (49 interior-point iterations),
  4 ms for the L2 form — the O(n) structure is asserted by test.
- **Componentwise L2 boosting (`boosting`)** — Bühlmann & Yu (2003) /
  Bühlmann (2006), the R mboost `glmboost` engine (roadmap Module 10,
  Tier 2 "boosted ARDL"), in `tsecon-ml`. Single-column least-squares
  base learners, greedy RSS selection (ties to the smallest index, no
  randomness anywhere — the `selected` sequence is a deterministic,
  seedless function of the inputs), `learning_rate` × the LS fit added
  per step from `F_0 = 0` (no intercept: pass a centered `y` and centered
  columns, as everywhere in the crate). The boosting operator `B_m =
  B_{m−1} + ν H_j (I − B_{m−1})` is tracked **exactly** in a rank-`m`
  factored form — no n×n matrix — and its trace feeds Bühlmann's (2006)
  corrected AIC `log(RSS_m/n) + (1 + df_m/n)/(1 − (df_m + 2)/n)`;
  `stop="aic"` reports the minimizing step, `stop="none"` the last, with
  `coef_path`, `selected`, `rss_path`, `df_path`, `aic_path`, `best_step`,
  `fitted`, and `predicted` (from `x_test`) returned either way.
  Validation, graded honestly as a **transcription**: an independent
  dense NumPy transcription of the published algorithm with the operator
  formed explicitly (so the trace is exact by construction) pins
  `coef_path`, `df_path`, and `aic_path` at 1e-12 (achieved 6.7e-16 /
  2.7e-15 / 1.6e-15) and `selected` / `best_step` exactly on five cases;
  R mboost is not runnable in the build environment, so this is not a
  third-party run and the card says so. Properties: RSS nonincreasing,
  the small-step limit reproduces OLS on the selected support at 3.3e-14,
  AIC stopping recovers a sparse truth's support (0.09 s at `n = 500,
  p = 50, 500 steps`, release wheel), and every teaching
  error (NaN naming the array, `insufficient data: {got} observations, at
  least 3 required`, `learning_rate` outside (0, 1], unknown `stop`
  listing the accepted values, `x_test` column mismatch) is pinned from
  Python.
