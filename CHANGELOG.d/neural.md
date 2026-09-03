### Added

- **`mlp_regression` — the library's only native neural net: a one- or
  two-hidden-layer feed-forward regressor with scikit-learn `MLPRegressor`'s
  exact objective, Adam or L-BFGS, early stopping on a *temporal* validation
  split, a seed ensemble, and a leakage-safe scaler** (roadmap Module 10,
  Tier 2 "Neural"; the "NN" of Medeiros et al. 2021 / Goulet Coulombe et
  al. 2022). Objective `(1/(2n))‖y − f(x)‖² + (α/(2n))Σ_l‖W_l‖²_F`,
  intercepts unpenalized; `activation` in tanh / relu / logistic; `solver=
  "adam"` (sklearn's constants; full batch by default, seeded shuffled
  mini-batches with `batch_size`) or `"lbfgs"` (the crate-wide
  `tsecon_optim::lbfgs` with the analytic gradient). The LAST
  `floor(validation_fraction · n)` rows are the validation set — never a
  random split — with `patience` epochs and the best epoch's weights
  restored; `n_seeds` members from independent Philox substreams are
  averaged; the scaler is fit on the training rows only and replayed on the
  validation rows and `x_test` (a property test perturbs the validation rows
  and finds the scaler bit-identical). Returns `fitted`, `predicted`,
  `member_predictions`, per-member `train_loss_path` /
  `validation_loss_path` / `best_epoch` / `converged` (early-stopped vs
  ran out of epochs), `n_parameters`, `weights` in sklearn's layout, and the
  training-row scaler. No framework dependency: dense Rust loops,
  single-threaded, deterministic given `seed` — bit-identical on the same
  build, statistically reproducible across platforms (last-ulp libm
  differences). **Grade: independent package for the mechanics** —
  `fixtures/neural.json` stores sklearn 1.9.0's fitted weights for four
  (architecture, activation) cases and pins the forward pass to `predict`
  (1e-12, achieved 3.1e-15), the objective (1e-10, 3.3e-16), the analytic
  gradient to sklearn's own `_backprop` (1e-10, 1.1e-15) and to a central
  finite difference (1e-6 relative, 5.3e-8), and the gradient norm at
  sklearn's converged weights (1e-8, 5.2e-16) — designed so no optimizer
  trajectory has to match. **Property / Monte-Carlo grade for the
  estimator**: y_t = sin(2 y_{t−1}) + 0.3 e_t recovered out of sample
  (Campbell-Thompson R² 0.76–0.90 for mini-batch Adam and L-BFGS across six
  data seeds; 0.59–0.82 for the all-defaults call; 0.46–0.76 linear AR(1);
  0.78–0.90 the oracle map); the ensemble beats the mean member in 10/10
  replications (Jensen) and the median member in 7/10 (Rust draws) / 10/10
  (Python draws) on a documented overfitting DGP — a majority-of-replications
  claim, stated as such; early stopping fires on an easy problem and cannot
  at `max_epochs=1`. Default call on n = 500, p = 5: 0.7–0.8 s on the
  release wheel. Teaching errors name the array with NaN/inf, list the
  accepted activation/solver names, name the two-layer limit, count the
  validation split in `insufficient data: {got} observations, at least
  {needed} required`, and — sentinel convention — refuse `learning_rate`,
  `batch_size`, or `patience` passed explicitly under `solver="lbfgs"`
  (the default call is bit-identical to passing the Adam defaults, tested).
- **`echo_state_network` — reservoir computing (Jaeger 2001; Lukoševičius
  2012), contrib tier.** Sparse random reservoir rescaled to the requested
  spectral radius (leading-eigenvalue modulus from a dense eigenvalue
  decomposition, recomputed on the scaled matrix and returned as
  `spectral_radius_achieved` — a power iteration does not converge on the
  complex leading pair of a random reservoir), uniform input weights,
  leaky-integrator tanh states `s_t = (1 − a)s_{t−1} + a tanh(W s_{t−1} +
  W_in u_t)`, washout discard, and a ridge readout on `[1, u_t, s_t]`
  through the crate's `ridge` (scikit-learn `Ridge(fit_intercept=False)`
  objective; the constant column penalized, Lukoševičius eq. 9). `x_test`
  is the continuation of `x`. Returns `fitted`, `predicted`, `readout`,
  `spectral_radius_achieved`, `reservoir_size`, `n_washout`, `n_train`.
  **Grade, per leg**: the state path on an explicit 6-unit reservoir is a
  NumPy transcription that `reservoirpy` 0.4.2's `Reservoir` (same explicit
  `W`/`Win`/`lr`) reproduced with max abs difference 0.0 at generation time
  (recorded in `_meta.esn.reservoirpy`; Rust pinned at 1e-12, achieved
  1.7e-16); the readout is the closed form cross-checked against scikit-learn
  `Ridge` (gap 6.6e-13; Rust pinned at 1e-10, achieved 6.6e-13); the spectral
  radius against `numpy.linalg.eigvals` (1e-6, achieved 5.8e-15). Property
  grade for the estimator: NARMA-10 out-of-sample NRMSE 0.32 (mean over four
  data seeds) with `input_scaling=0.3` on 1000 training rows and 0.16–0.19
  with `reservoir_size=400` on 2000 rows — the all-defaults call averages
  0.43–0.46 because `input_scaling=1` over-drives tanh for NARMA's u ∈ [0,
  0.5], and the card says so; the achieved radius within 1e-6 of the target;
  seed contract. Default call on n = 500, p = 5: ≈ 0.1 s. `washout >= n`
  names the fix; fewer than two surviving rows reports the insufficiency
  count with the washout included.
- `tsecon-ml` gains the `tsecon-rng` and `tsecon-optim` dependencies (seeded
  draws and the shared L-BFGS) and four `MlError` variants
  (`InsufficientData`, `UnknownChoice`, `InvalidValue`, `Diverged`) the
  neural surfaces use for their teaching errors. Model card
  `docs/reference/model-cards/ml-neural.md`; validation-matrix rows in
  `docs/reference/_wave/neural-validation-rows.md`.
