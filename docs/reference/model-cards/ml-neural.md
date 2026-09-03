# Model card — Neural regressors: MLP ensemble and echo state network

**Family:** `mlp_regression`, `echo_state_network`

The two native neural estimators of `tsecon` (roadmap Module 10, the Tier 2
"Neural" row and the contrib-tier echo state network). `mlp_regression` is
the "NN" entry of the macro-forecasting horse races (Medeiros et al. 2021;
Goulet Coulombe et al. 2022): a shallow feed-forward net with the guard
rails those papers use — a seed ensemble, early stopping on a *temporal*
validation split, standardization fit on the training rows only — and
scikit-learn `MLPRegressor`'s exact objective, so its numbers are drop-in
comparable. `echo_state_network` is reservoir computing (Jaeger 2001;
Lukoševičius 2012): a fixed random recurrent reservoir whose only trained
part is a ridge readout, so nonlinear dynamics cost one linear regression.
Both are pure Rust, single-threaded, and deterministic given `seed`; there
is no framework dependency, and deep-learning adapters (torch, N-BEATS,
foundation models) are out of core by scope ruling.

| Function | Role |
|----------|------|
| `mlp_regression` | One- or two-hidden-layer regressor, Adam or L-BFGS, temporal early stopping, seed ensemble |
| `echo_state_network` | Leaky-integrator tanh reservoir with a ridge readout, spectral-radius scaled |

## What it estimates

- **`mlp_regression(x, y, hidden=(16,), activation="tanh", alpha=1e-4, solver="adam", learning_rate=None, batch_size=None, max_epochs=500, validation_fraction=0.2, patience=None, n_seeds=5, seed=0, standardize=True, x_test=None)`**
  minimizes, per ensemble member, scikit-learn's objective

  L(W, b) = (1/2n) Σᵢ (yᵢ − f(xᵢ))² + (α/2n) Σₗ ‖Wₗ‖²_F,

  intercepts unpenalized, with hidden activation tanh / relu / logistic and
  an identity output. The prediction is the average over `n_seeds` members,
  each initialized (Glorot-uniform, sklearn's bounds) and, under
  mini-batches, shuffled from its own Philox substream of `seed`. Early
  stopping holds out the **last** ⌊validation_fraction · n⌋ rows and stops a
  member once its validation loss has failed to improve on its best by a
  relative 1e-4 for `patience` epochs, restoring the best epoch's weights.
- **`echo_state_network(x, y, reservoir_size=200, spectral_radius=0.9, leak_rate=1.0, input_scaling=1.0, sparsity=0.1, washout=50, ridge_alpha=1e-6, seed=0, x_test=None)`**
  runs the state recursion

  sₜ = (1 − a) sₜ₋₁ + a tanh(W sₜ₋₁ + W_in uₜ),  s₀ = 0,

  with `W` sparse-normal (each entry nonzero with probability `sparsity`)
  rescaled to the requested spectral radius and `W_in` uniform on
  ±`input_scaling`, discards the first `washout` states, and fits the readout
  b = argmin ‖y − Zb‖² + ridge_alpha ‖b‖² on Zₜ = [1, uₜ, sₜ] with the crate's
  `ridge` (scikit-learn `Ridge(fit_intercept=False)` convention: no 1/n
  factor, the constant column penalized like every other coefficient —
  Lukoševičius eq. 9). `x_test` rows are the *continuation* of `x`: the
  recursion carries on from the last training state, no washout re-applied.

## Assumptions

- **Both estimators are single-threaded and deterministic given `seed`.**
  The same call on the same build is bit-identical (tested). Across
  platforms and compilers the libm `tanh`/`exp` and the eigenvalue routine
  can differ in the last ulp, so the cross-platform promise is *statistical*
  reproducibility of the seed ensemble, not bitwise identity — the honest
  version of Module 10's "neural nondeterminism" warning. There is no BLAS
  threading here to make it worse.
- **The validation split is temporal, never random.** The last rows are held
  out because a random split on a time series lets the future into the
  training set. The scaler (`x_mean`, `x_scale`, `y_mean`, `y_scale`) is fit
  on the training rows only and replayed on the validation rows and on
  `x_test`; a property test perturbs the validation rows and checks the
  scaler is bit-identical.
- **`alpha` is on scikit-learn's MLP scale** (penalty divided by n, like the
  data-fit term) — not the `ridge` scale, and not glmnet's.
- **The optimizer trajectory is not a contract.** Adam uses sklearn's
  constants and L-BFGS reuses tsecon's quasi-Newton; neither reproduces
  sklearn's iterates, and the golden is deliberately designed not to need
  that (see *Validated against*).
- **`spectral_radius < 1` is the usual, not a guaranteed, route to the echo
  state property** (Yildiz, Jaeger & Kiebeling 2012); values above 1 are
  accepted, and the washout is what removes the initial-state transient.
- **`sparsity` is the connectivity** — the fraction of nonzero reservoir
  entries — so `0.1` means a 10% dense reservoir.

## When to use

- **`mlp_regression`** for a nonlinear benchmark in a forecasting horse race
  on a modest number of lagged predictors: the paper-standard shallow net
  with an honest out-of-sample protocol. Use `solver="lbfgs"` for small
  designs where full-batch quasi-Newton converges in seconds; `solver="adam"`
  with `batch_size` for larger ones or when you want an epoch-wise loss path
  to inspect.
- **`echo_state_network`** when the target has long, nonlinear memory in an
  input sequence (NARMA-style benchmarks, mixed-frequency indicators) and you
  want a nonlinear model that trains in the time of one ridge regression,
  then tune `input_scaling`, `spectral_radius`, `leak_rate` by walk-forward
  validation with `cv_splits`.
- **Not** for deep architectures (the MLP is limited to two hidden layers by
  design), images, or text — those belong to the framework adapters outside
  core.

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `mlp_regression` | `hidden` | `(16,)` | one or two widths (tuple or list; an int is one layer); more raises |
| | `activation` | `"tanh"` | `"tanh"`, `"relu"`, `"logistic"` |
| | `alpha` | `1e-4` | L2 on the weights, sklearn MLP scale |
| | `solver` | `"adam"` | `"adam"` or `"lbfgs"` |
| | `learning_rate` | `None` → `1e-3` | Adam step; passing it under lbfgs raises |
| | `batch_size` | `None` | full batch; an int → seeded shuffled mini-batches; under lbfgs raises |
| | `max_epochs` | `500` | epoch budget (Adam) / iteration cap (L-BFGS) |
| | `validation_fraction` | `0.2` | the LAST rows; `0` disables early stopping; ≤ 0.5 |
| | `patience` | `None` → `20` | epochs without a relative-1e-4 improvement; under lbfgs raises |
| | `n_seeds` / `seed` | `5` / `0` | ensemble size / root seed of the Philox substreams |
| | `standardize` | `True` | training-row scaler for `x` and `y` |
| `echo_state_network` | `reservoir_size` | `200` | units N |
| | `spectral_radius` | `0.9` | leading-eigenvalue modulus after rescaling |
| | `leak_rate` | `1.0` | a in (0, 1]; 1 is the plain ESN |
| | `input_scaling` | `1.0` | `W_in` uniform on ±input_scaling — the first knob to tune (0.3 for NARMA-10's u ∈ [0, 0.5]) |
| | `sparsity` | `0.1` | connectivity (fraction of nonzero entries) |
| | `washout` | `50` | leading states discarded; must leave ≥ 2 rows |
| | `ridge_alpha` | `1e-6` | readout penalty, `Ridge` scale |

## How to read the output

- **`mlp_regression`** → `fitted` (ensemble mean on every row of `x`, original
  y scale), `predicted` / `member_predictions` (ensemble mean and the
  `n_seeds × n_test` array on `x_test`; `None` without it), `train_loss_path` /
  `validation_loss_path` (lists of per-member per-epoch arrays; two entries
  — initial and final — under lbfgs; empty validation paths when
  `validation_fraction=0`), `best_epoch` and `converged` per member (`True` =
  early stopping fired / L-BFGS met its test; `False` = ran out of
  `max_epochs` — read a member with `False` as under-trained, not as a
  failure), `n_parameters`, `weights` (per member `{"coefs", "intercepts"}` in
  sklearn's `fan_in × fan_out` layout on the standardized scale; with the
  returned scaler they reproduce `fitted` exactly), `n_train`,
  `n_validation`, `x_mean`, `x_scale`, `y_mean`, `y_scale`, `solver`,
  `activation`.
- **`echo_state_network`** → `fitted` (readout on the rows that entered the
  fit, so length `n − washout`), `predicted` (on `x_test`, else `None`),
  `readout` (length `1 + p + N`: intercept, input weights, state weights),
  `spectral_radius_achieved` (recomputed on the scaled matrix — compare it to
  what you asked for), `reservoir_size`, `n_washout`, `n_train`.

## Failure modes

- **All members report `converged=False` with the defaults.** Full-batch Adam
  at `learning_rate=1e-3` moves each weight by at most 0.5 in 500 epochs; on
  a strongly nonlinear target it is still learning when the budget ends (the
  sin-AR(1) test below: 0.59–0.82 R² against 0.76–0.90 with mini-batches).
  Pass `batch_size=32, learning_rate=1e-2`, or `solver="lbfgs"`.
- **Passing `learning_rate` / `batch_size` / `patience` under
  `solver="lbfgs"` raises.** They cannot apply; the refusal names the
  argument. Leave them `None`.
- **`hidden=(64, 32, 16)` raises.** Two hidden layers is the design limit;
  the error says so.
- **Too few rows.** The insufficiency error counts the temporal split
  (`validation_fraction=0.2` needs 5 rows: 4 training + 1 validation) or the
  washout (`washout + 2` rows); `washout >= n` names the fix.
- **`input_scaling=1` on small-amplitude inputs** drives the tanh units
  toward saturation and the readout overfits (NARMA-10 at the defaults:
  NRMSE 0.39–0.56 across data seeds against 0.26–0.41 at `input_scaling=0.3`).
  Tune it first.
- **Interpreting `fitted` on the validation rows as in-sample fit.** They
  were held out from the weights *but* chose the stopping epoch; the honest
  out-of-sample number is `predicted` on `x_test`.
- **Expecting sklearn's exact weights.** Same objective, different
  optimizer paths and initializations; compare the *objective value* and
  out-of-sample error, not the weights.

## Validated against

**`mlp_regression` — independent package** (scikit-learn 1.9.0
`MLPRegressor`, `fixtures/neural.json`), designed so the golden never needs
an optimizer trajectory: the fixture stores sklearn's fitted `coefs_` /
`intercepts_` for four (architecture, activation) cases — tanh (16,), relu
(8, 4), logistic (10,), tanh (12, 6) — and the Rust test drives its own
forward pass, objective, and analytic gradient at those weights.

| Pin | Asserted | Achieved |
|---|---|---|
| forward pass = sklearn `predict` on 20 held-out rows | 1e-12 | 3.1e-15 |
| objective at the fitted weights = sklearn-convention loss (formula, `_backprop`, and `est.loss_` agree in the generator) | 1e-10 | 3.3e-16 |
| analytic gradient = sklearn's own `_backprop` at Glorot-scale random weights and at the fitted weights (relu included) | 1e-10 | 1.1e-15 |
| analytic gradient = central finite difference of our loss (tanh, logistic) | 1e-6 relative | 5.3e-8 |
| gradient inf-norm at sklearn's converged weights = the norm measured there (4.9e-5, 2.2e-3, 1.3e-4, 9.7e-5; fixture bar 1e-2 — scipy's L-BFGS-B stops on `ftol`, and the relu objective is only piecewise smooth) | 1e-8 | 5.2e-16 |

The estimator is **property / Monte-Carlo graded** (`neural_properties.rs`,
re-checked through Python in `test_neural.py` on the release wheel): on
y_t = sin(2 y_{t−1}) + 0.3 e_t (600 training / 100 test rows,
Campbell-Thompson out-of-sample R², six data seeds) the oracle map scores
0.78–0.90, mini-batch Adam and L-BFGS 0.76–0.90, the all-defaults call
0.59–0.82, a linear AR(1) 0.46–0.76; the ensemble beats the *mean* member MSE
in 10/10 replications (Jensen — a theorem) and the *median* member in 7/10
(Rust draws) and 10/10 (Python draws) on a documented overfitting DGP
(60 rows, σ = 0.7, (32, 16), L-BFGS, α = 1e-6, no validation split) — a
majority-of-replications claim, not per-replication dominance; early
stopping fires on an easy problem (best epochs ≈ 90–130 of 500) and cannot
at `max_epochs=1`; same seed bit-identical, different seeds differ; the
scaler is invariant to perturbing the validation rows. The default call on
n = 500, p = 5 takes 0.7–0.8 s on the release wheel (`solver="lbfgs"` ≈ 3 s).

**`echo_state_network` — transcription with third-party legs.** The state
recursion on an explicit 6-unit reservoir is pinned at 1e-12 (achieved
1.7e-16) against a NumPy transcription of Lukoševičius eqs. 2–3 that
`reservoirpy` 0.4.2's `Reservoir` — run with the same explicit `W`, `Win`,
`lr` — reproduced with max abs difference 0.0 at generation time
(`_meta.esn.reservoirpy`); the readout at 1e-10 (achieved 6.6e-13) against
the closed form (Z′Z + αI)⁻¹Z′y, itself cross-checked in the generator
against scikit-learn `Ridge` (gap 6.6e-13); the spectral radius against
`numpy.linalg.eigvals` on a 30 × 30 sparse matrix at 1e-6 (achieved
5.8e-15), and rescaling to a target lands within 5.8e-15 of it. Property
grade for the estimator: NARMA-10 out-of-sample NRMSE 0.32 (mean over four
data seeds, 0.26–0.41) with `input_scaling=0.3` and otherwise default
settings on 1000 training rows, 0.16–0.19 with `reservoir_size=400` on
2000 rows, 0.39–0.56 at the all-defaults call; `spectral_radius_achieved`
within 1e-6 of the target for 0.5, 0.9, 1.25; `x_test` states equal the
tail of one long run (1e-14); seed contract. The default call on n = 500,
p = 5 takes ≈ 0.1 s on the release wheel.

Fixture: [`fixtures/neural.json`](../../../fixtures/neural.json)
(generator `fixtures/generate_neural_fixtures.py`, which never imports
tsecon).

## References

- Jaeger, H. (2001). "The 'echo state' approach to analysing and training
  recurrent neural networks." GMD Report 148.
- Lukoševičius, M. (2012). "A practical guide to applying echo state
  networks." In *Neural Networks: Tricks of the Trade*, Springer.
- Yildiz, I. B., Jaeger, H. & Kiebeling, S. J. (2012). "Re-visiting the echo
  state property." *Neural Networks* 35.
- Kingma, D. P. & Ba, J. (2015). "Adam: A Method for Stochastic
  Optimization." ICLR.
- Medeiros, M. C., Vasconcelos, G. F. R., Veiga, Á. & Zilberman, E. (2021).
  "Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine
  Learning Methods." *JBES* 39.
- Goulet Coulombe, P., Leroux, M., Stevanovic, D. & Surprenant, S. (2022).
  "How is Machine Learning Useful for Macroeconomic Forecasting?" *Journal
  of Applied Econometrics* 37.
- Campbell, J. Y. & Thompson, S. B. (2008). "Predicting Excess Stock Returns
  Out of Sample." *RFS* 21 (the out-of-sample R² convention).

See also: [Penalized regression and leakage-safe validation](machine-learning.md)
for `ridge` (the ESN readout) and `cv_splits` (for tuning either estimator).

## Runnable example

```python
import numpy as np
import tsecon

rng = np.random.default_rng(3)
n = 700
# A nonlinear AR(1): y_t = sin(2 y_{t-1}) + 0.3 e_t, lagged into (x, y).
y = np.zeros(n + 101)
e = rng.standard_normal(n + 101)
for t in range(1, y.size):
    y[t] = np.sin(2.0 * y[t - 1]) + 0.3 * e[t]
y = y[100:]
x, target = y[:n].reshape(-1, 1), y[1:n + 1]
x_train, y_train, x_test, y_test = x[:600], target[:600], x[600:], target[600:]

# 1. The MLP: 5-seed ensemble, early stopping on the LAST 20% of the rows.
mlp = tsecon.mlp_regression(x_train, y_train, hidden=(16,), batch_size=32,
                            learning_rate=1e-2, max_epochs=200, x_test=x_test)
oos = 1 - np.sum((y_test - mlp["predicted"]) ** 2) / np.sum((y_test - y_train.mean()) ** 2)
print("MLP out-of-sample R^2:", round(float(oos), 3), " members converged:", mlp["converged"])
print("best epochs:", mlp["best_epoch"], " parameters:", mlp["n_parameters"])

# 2. The same net by L-BFGS (no learning rate, batches, or patience apply).
lb = tsecon.mlp_regression(x_train, y_train, solver="lbfgs", max_epochs=300, x_test=x_test)
oos_lb = 1 - np.sum((y_test - lb["predicted"]) ** 2) / np.sum((y_test - y_train.mean()) ** 2)
print("L-BFGS out-of-sample R^2:", round(float(oos_lb), 3))

# 3. An echo state network on NARMA-10 (the reservoir benchmark).
u = 0.5 * rng.uniform(size=1200)
z = np.zeros(1200)
for t in range(9, 1199):
    z[t + 1] = 0.3 * z[t] + 0.05 * z[t] * z[t - 9:t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
esn = tsecon.echo_state_network(u[:1000].reshape(-1, 1), z[:1000], input_scaling=0.3,
                                x_test=u[1000:].reshape(-1, 1))
nrmse = np.sqrt(np.mean((z[1000:] - esn["predicted"]) ** 2) / np.var(z[1000:]))
print("ESN NARMA-10 out-of-sample NRMSE:", round(float(nrmse), 3),
      " spectral radius:", round(esn["spectral_radius_achieved"], 6),
      " readout size:", esn["readout"].shape[0])
```

Expected output:

```
MLP out-of-sample R^2: 0.887  members converged: [True, True, True, True, True]
best epochs: [164, 123, 168, 123, 150]  parameters: 49
L-BFGS out-of-sample R^2: 0.888
ESN NARMA-10 out-of-sample NRMSE: 0.231  spectral radius: 0.9  readout size: 202
```
