### Added — kernel methods (roadmap Module 10, Tier 2 "Nonparametric regression")

- **`kernel_ridge`** — kernel ridge regression with scikit-learn's exact
  conventions: the dual Cholesky solve `(K + alpha I) a = y` (no `1/n`, no
  intercept — the `Ridge` scale) for the rbf `exp(-gamma ||x-y||^2)`,
  laplacian `exp(-gamma ||x-y||_1)`, polynomial `(gamma <x,y> + coef0)^degree`
  and linear kernels, `gamma=None -> 1/n_features`, `x_test` predictions, and
  `rff_features=D` for the Rahimi-Recht (2007) random-Fourier-feature primal
  approximation of the rbf kernel (`z(x) = sqrt(2/D) cos(Wx + b)`, drawn from
  a Philox stream keyed by `seed`; `O(n D^2)` instead of `O(n^3)`). Returns
  `dual_coef` (exact) or `coef` (RFF), `fitted`, `predicted` (only with
  `x_test`), `kernel`, the resolved `gamma`, `n_rff_features`. **Validated
  against scikit-learn 1.9.0 `KernelRidge`** — `dual_coef_`, `predict(X)`,
  `predict(X_test)` for two parameterizations of each kernel, asserted at
  1e-8, achieved 2.2e-12 (independent package). The RFF mode is honestly
  *not* golden-pinned (it is a seeded Monte-Carlo approximation): property
  tests pin bit-identical output under the same seed, different output
  under different seeds, and RMSE against the exact fit falling
  0.307 → 0.049 → 0.016 at `D = 20, 200, 2000`. Where scikit-learn silently
  falls back to a least-squares solve when `K + alpha I` is singular
  (`alpha=0` with duplicate rows), tsecon refuses and names `alpha`.
- **`kernel_regression`** — Nadaraya-Watson (`kind="nadaraya_watson"`) and
  local-linear (`"local_linear"`, the default) nonparametric regression with a
  product Gaussian kernel for one to three regressors, matching statsmodels
  `KernelReg(reg_type="lc"|"ll", var_type="c"*k)` exactly (the local-linear
  fit goes through the pseudoinverse with NumPy's `1e-15` cutoff, as
  statsmodels' does). Bandwidths: `bandwidth_method="fixed"` (scalar or
  per-column), `"loo_cv"` (statsmodels' leave-one-out least-squares
  criterion), and **`"block_cv"` — the leave-block-out criterion of Chu &
  Marron (1991) / Hart & Vieu (1990)** that drops the `2*block + 1`
  observations around `i` when predicting `y_i` (default
  `block = ceil(n^(1/3))`), the dependence-aware selector the roadmap calls
  for: on an AR(1)-error design (`rho = 0.9`, `n = 200`) leave-one-out drives
  the bandwidth to the search's lower wall on every seed while block-CV
  selects a 10–400× wider one on 10/10 seeds. Selection is a deterministic
  21-point log grid on a common multiple of the Scott reference, golden-
  section refinement, and per-column coordinate polishing for `k >= 2` — it
  reaches a criterion no worse than statsmodels' Nelder-Mead `fmin` on all
  four fixture cases without chasing its path. Returns `fitted`,
  `predicted` (only with `x_test`), the resolved per-column `bandwidth`,
  `bandwidth_method`, `block`, `cv_criterion` (statsmodels' `cv_loo` value
  under `"fixed"`/`"loo_cv"`), `effective_df` (`tr(S)` of the linear
  smoother: `k+1` or `1` at huge bandwidths, `n` at tiny ones), `kind`,
  `kernel`, the honesty flag `bandwidth_at_boundary` (a selected bandwidth
  on a wall of the search range — proven to fire on pure noise), and
  `n_criterion_evaluations`. **Validated against statsmodels 0.15.0
  `KernelReg`** — `fit()` at the training rows and at `x_test` for twelve
  fixed-bandwidth cases (`k = 1, 2`, both estimators) asserted at 1e-8,
  achieved 6.7e-15; `cv_loo(bw, func)` asserted at 1e-10, achieved 3.0e-15
  (independent package). The leave-block-out criterion and `effective_df`
  have no package reference and are graded honestly as documented-formula
  transcriptions (NumPy in the generator; they reproduce `cv_loo` at `l = 0`
  and `fit()` at 1e-12), pinned at 1e-10, achieved 2.7e-15 / 3.6e-15.
  **Measured speed at `n = 2000`** (local linear, `k = 1`, fixed bandwidth,
  this environment): a local-linear fit at a fixed bandwidth takes 0.16 s against statsmodels' 0.59 s `fit()` (3.6×) and already includes the LOO criterion that statsmodels' `cv_loo` needs a further 0.97 s for (5.9×); the full leave-one-out bandwidth search reaches statsmodels' `bw="cv_ls"` optimum (`h = 0.1793`, identical criterion) in 6.5 s against 22.0 s (3.4×); at `k = 2` the search takes 48 s against 59 s (1.2×, same optimum). Both implementations are bound by the same four million kernel evaluations per criterion pass, so this is a constant-factor win, not an order of magnitude — stated as measured; the exact `kernel_ridge` solve at `n = 2000` takes 0.48 s against scikit-learn's 13.8 s in this environment (a LAPACK-backend effect, agreement 2e-14). The leave-block-out criterion has no statsmodels counterpart at any speed.
- Teaching errors throughout, following the round-10 sentinel convention:
  NaN/inf refused naming `x`, `y` or `x_test`; the house
  `insufficient data: {got} observations, at least {needed} required` with
  the exact minimum (`k + 2 + 2*block` for local linear); unknown
  `kernel`/`kind`/`bandwidth_method` strings list the accepted values; a
  non-positive bandwidth names the column and the fix; `bandwidth` passed
  with a CV method, `block` with `"fixed"`/`"loo_cv"`, `block=0`, `gamma`
  with the linear kernel, `rff_features` with a non-rbf kernel, a non-zero
  `seed` in exact mode, and non-default `degree`/`coef0` with a
  non-polynomial kernel all **raise** naming the argument and the mode that
  would use it — nothing is silently ignored. `x` and `x_test` accept a 1-D
  array for a single regressor. New crate surface in `tsecon-ml`:
  `kernel_ridge`, `kernel_matrix`, `kernel_regression`, `cv_criterion`,
  with `MlError::InsufficientData`, `MlError::InvalidValue` and
  `MlError::NotPositiveDefinite`. Model card:
  [`ml-kernel.md`](docs/reference/model-cards/ml-kernel.md); fixture
  `fixtures/kernel.json`.
