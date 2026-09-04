//! Feed-forward neural regressor: one or two hidden layers, scikit-learn's
//! exact objective, Adam or L-BFGS, early stopping on a *temporal*
//! validation split, and a seed ensemble.
//!
//! This is the only neural network implemented natively in `tsecon` — the
//! "NN" entry of the macro-forecasting horse races (Medeiros, Vasconcelos,
//! Veiga & Zilberman 2021; Goulet Coulombe, Leroux, Stevanovic & Surprenant
//! 2022). Deep-learning adapters (torch models, N-BEATS, foundation models)
//! are out of core by scope ruling and this module adds no framework
//! dependency: dense loops over `Vec<f64>`, single-threaded, deterministic.
//!
//! # Objective (scikit-learn `MLPRegressor` convention)
//!
//! With `n` rows, hidden activation `phi` (tanh, relu, or logistic), an
//! identity output layer, and weights `W_l` / intercepts `b_l`,
//!
//! ```text
//! L(W, b) = (1/(2n)) sum_i (y_i - f(x_i))^2  +  (alpha/(2n)) sum_l ||W_l||_F^2 ,
//! ```
//!
//! intercepts unpenalized. This is exactly what
//! `sklearn.neural_network._multilayer_perceptron._backprop` computes
//! (`squared_loss = 0.5 * mean((y - yhat)^2)` plus
//! `0.5 * alpha * sum(coef.ravel() @ coef.ravel()) / n_samples`), and the
//! gradient is the same back-propagation: `delta_L = yhat - y`,
//! `dW_l = (A_{l-1}' delta_l + alpha W_l) / n`, `db_l = mean_rows(delta_l)`,
//! `delta_{l-1} = (delta_l W_l') * phi'(A_{l-1})`. Under mini-batches the
//! same formula is applied to each batch with `n` its row count, as
//! scikit-learn does. The golden fixture `fixtures/neural.json` pins the
//! forward pass, this objective, and this gradient at scikit-learn's own
//! fitted and at random weights (see [`mlp_forward`],
//! [`mlp_loss_gradient`]); the optimizer *trajectory* is deliberately not
//! pinned — no two Adam or L-BFGS implementations share one.
//!
//! # Training
//!
//! * **Adam** (Kingma & Ba 2015) with scikit-learn's constants
//!   (`beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-8`, bias-corrected
//!   step). `batch_size = None` takes one full-batch step per epoch;
//!   otherwise each epoch is a seeded Fisher-Yates shuffle of the training
//!   rows cut into `batch_size` chunks (the last chunk may be shorter).
//! * **L-BFGS** via [`tsecon_optim::lbfgs`] (strong-Wolfe line search) on
//!   the full training objective, with the analytic gradient. Epoch-wise
//!   arguments (`learning_rate`, `batch_size`, `patience`) have no meaning
//!   here and are *refused* when passed explicitly rather than silently
//!   ignored (the sentinel convention: pass `None`).
//! * **Early stopping on a temporal split.** The LAST
//!   `floor(validation_fraction * n)` rows are the validation set — never
//!   a random split, which on a time series leaks the future into the
//!   training set. The validation loss is the data-fit term alone,
//!   `0.5 * mean((y - yhat)^2)` on the validation rows (standardized
//!   units when `standardize = true`); training stops once it has failed
//!   to improve on its best value by a relative
//!   [`EARLY_STOPPING_TOL`] for `patience` consecutive epochs, and the
//!   weights of the best epoch are restored. `converged = true` means the
//!   rule fired; `false` means the member ran out of `max_epochs`.
//! * **Standardization fit on the training rows only.** Column means and
//!   scales of `x`, and the mean and scale of `y`, are computed on the
//!   first `n - n_validation` rows and replayed on the validation rows and
//!   on `x_test` — the validation rows never inform the scaler (a property
//!   test perturbs them and checks the scaler is bit-identical).
//! * **Seed ensemble.** `n_seeds` members are trained from independent
//!   Philox substreams of `seed` ([`tsecon_rng::Stream::substreams`]:
//!   member `m` always gets the same stream for a given `seed`), each with
//!   its own Glorot-uniform initialization (scikit-learn's `_init_coef`
//!   bounds: `sqrt(6 / (fan_in + fan_out))`, or `sqrt(2 / ...)` for the
//!   logistic activation) and, under mini-batches, its own shuffles. The
//!   reported `fitted` / `predicted` are the ensemble means; every member's
//!   predictions, loss paths, and weights are returned too.
//!
//! # Reproducibility — what is promised
//!
//! The Rust is single-threaded and every random draw is a pure function of
//! `seed`, so the same call on the same build returns bit-identical
//! results, every time (tested). Across platforms and compilers the
//! libm `tanh`/`exp` may differ in the last ulp, so the cross-platform
//! promise is *statistical* reproducibility of the seed ensemble, not
//! bitwise identity — the honest version of the "neural nondeterminism"
//! warning in roadmap Module 10 (there is no BLAS threading here to make
//! it worse).

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_optim::{lbfgs, LbfgsOptions, ObjectiveFn};
use tsecon_rng::Stream;

use crate::error::MlError;
use crate::standardize::Scaler;
use crate::util::check_xy;

/// Adam first-moment decay (scikit-learn `beta_1`).
const ADAM_BETA1: f64 = 0.9;
/// Adam second-moment decay (scikit-learn `beta_2`).
const ADAM_BETA2: f64 = 0.999;
/// Adam denominator guard (scikit-learn `epsilon`).
const ADAM_EPS: f64 = 1e-8;
/// Learning rate used when `learning_rate` is `None` under Adam.
pub const DEFAULT_LEARNING_RATE: f64 = 1e-3;
/// Patience used when `patience` is `None` under Adam.
pub const DEFAULT_PATIENCE: usize = 20;
/// Relative improvement in the validation loss that counts as progress:
/// an epoch improves on the best validation loss so far only if it beats
/// it by more than `EARLY_STOPPING_TOL * |best|`. Scale-free, so it means
/// the same thing whether or not `y` is standardized.
pub const EARLY_STOPPING_TOL: f64 = 1e-4;
/// L-BFGS gradient (inf-norm) tolerance on the `1/n`-scaled objective.
const LBFGS_GRAD_TOL: f64 = 1e-6;
/// L-BFGS relative function-decrease tolerance (scipy's `ftol` scale).
const LBFGS_F_TOL: f64 = 1e-9;
/// The deliberate depth limit of the native net.
const MAX_HIDDEN_LAYERS: usize = 2;
/// Fewest training rows the scaler and the fit are defined on.
const MIN_TRAIN_ROWS: usize = 2;

/// Hidden-layer activation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `tanh(z)`.
    Tanh,
    /// `max(z, 0)`.
    Relu,
    /// `1 / (1 + exp(-z))`.
    Logistic,
}

impl Activation {
    /// The accepted names, as rendered in the teaching error.
    pub const ACCEPTED: &'static str = "\"tanh\", \"relu\", \"logistic\"";

    /// Parses an activation name.
    ///
    /// # Errors
    ///
    /// [`MlError::UnknownChoice`] listing the accepted names.
    pub fn parse(name: &str) -> Result<Self, MlError> {
        match name {
            "tanh" => Ok(Self::Tanh),
            "relu" => Ok(Self::Relu),
            "logistic" => Ok(Self::Logistic),
            other => Err(MlError::UnknownChoice {
                what: "activation",
                got: other.to_string(),
                accepted: Self::ACCEPTED,
            }),
        }
    }

    /// The name this activation parses from.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Tanh => "tanh",
            Self::Relu => "relu",
            Self::Logistic => "logistic",
        }
    }

    #[inline]
    fn apply(self, z: f64) -> f64 {
        match self {
            Self::Tanh => z.tanh(),
            Self::Relu => {
                if z > 0.0 {
                    z
                } else {
                    0.0
                }
            }
            Self::Logistic => 1.0 / (1.0 + (-z).exp()),
        }
    }

    /// `phi'(z)` expressed through the activation *output* `a = phi(z)`,
    /// exactly as scikit-learn's in-place derivatives do it.
    #[inline]
    fn derivative_from_output(self, a: f64) -> f64 {
        match self {
            Self::Tanh => 1.0 - a * a,
            Self::Relu => {
                if a > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Logistic => a * (1.0 - a),
        }
    }

    /// Glorot-uniform factor (scikit-learn `_init_coef`).
    fn glorot_factor(self) -> f64 {
        match self {
            Self::Logistic => 2.0,
            Self::Tanh | Self::Relu => 6.0,
        }
    }
}

/// Weight optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Solver {
    /// Adam (Kingma & Ba 2015): full-batch or seeded mini-batches, epochs,
    /// early stopping.
    Adam,
    /// L-BFGS on the full training objective via [`tsecon_optim::lbfgs`].
    Lbfgs,
}

impl Solver {
    /// The accepted names, as rendered in the teaching error.
    pub const ACCEPTED: &'static str = "\"adam\", \"lbfgs\"";

    /// Parses a solver name.
    ///
    /// # Errors
    ///
    /// [`MlError::UnknownChoice`] listing the accepted names.
    pub fn parse(name: &str) -> Result<Self, MlError> {
        match name {
            "adam" => Ok(Self::Adam),
            "lbfgs" => Ok(Self::Lbfgs),
            other => Err(MlError::UnknownChoice {
                what: "solver",
                got: other.to_string(),
                accepted: Self::ACCEPTED,
            }),
        }
    }

    /// The name this solver parses from.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Adam => "adam",
            Self::Lbfgs => "lbfgs",
        }
    }
}

/// Weights of one network in scikit-learn's layout: `coefs[l]` is
/// `fan_in x fan_out` (rows index inputs, columns units) and
/// `intercepts[l]` has length `fan_out`. The last layer has one output
/// unit.
#[derive(Debug, Clone, PartialEq)]
pub struct MlpWeights {
    /// Per-layer coefficient matrices, `fan_in x fan_out`.
    pub coefs: Vec<Mat<f64>>,
    /// Per-layer intercept vectors, length `fan_out`.
    pub intercepts: Vec<Vec<f64>>,
}

impl MlpWeights {
    /// The layer sizes `[n_inputs, hidden..., 1]` implied by the shapes.
    ///
    /// # Errors
    ///
    /// [`MlError::DimensionMismatch`] / [`MlError::EmptyInput`] when the
    /// shapes do not chain, the intercept lengths disagree with the
    /// coefficient columns, or the output layer is not one unit wide.
    pub fn layer_units(&self) -> Result<Vec<usize>, MlError> {
        if self.coefs.is_empty() {
            return Err(MlError::EmptyInput { what: "coefs" });
        }
        if self.intercepts.len() != self.coefs.len() {
            return Err(MlError::DimensionMismatch {
                what: "one intercept vector per coefficient matrix",
                expected: self.coefs.len(),
                got: self.intercepts.len(),
            });
        }
        let mut units = Vec::with_capacity(self.coefs.len() + 1);
        units.push(self.coefs[0].nrows());
        for (l, c) in self.coefs.iter().enumerate() {
            if c.nrows() != units[l] {
                return Err(MlError::DimensionMismatch {
                    what: "coefficient rows must equal the previous layer's width",
                    expected: units[l],
                    got: c.nrows(),
                });
            }
            if c.ncols() == 0 || c.nrows() == 0 {
                return Err(MlError::EmptyInput { what: "coefs" });
            }
            if self.intercepts[l].len() != c.ncols() {
                return Err(MlError::DimensionMismatch {
                    what: "intercept length must equal the coefficient columns",
                    expected: c.ncols(),
                    got: self.intercepts[l].len(),
                });
            }
            units.push(c.ncols());
        }
        if units[units.len() - 1] != 1 {
            return Err(MlError::DimensionMismatch {
                what: "the output layer must have exactly one unit",
                expected: 1,
                got: units[units.len() - 1],
            });
        }
        Ok(units)
    }

    /// Total number of weights and intercepts.
    #[must_use]
    pub fn n_parameters(&self) -> usize {
        self.coefs
            .iter()
            .map(|c| c.nrows() * c.ncols())
            .sum::<usize>()
            + self.intercepts.iter().map(Vec::len).sum::<usize>()
    }

    /// Packs into one flat vector: every coefficient matrix row-major
    /// (input-major), then every intercept — scikit-learn's `_pack`.
    #[must_use]
    pub fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_parameters());
        for c in &self.coefs {
            for i in 0..c.nrows() {
                for j in 0..c.ncols() {
                    out.push(c[(i, j)]);
                }
            }
        }
        for b in &self.intercepts {
            out.extend_from_slice(b);
        }
        out
    }

    /// Unpacks a flat vector laid out as [`to_flat`](Self::to_flat) for the
    /// given `layer_units` (`[n_inputs, hidden..., 1]`).
    ///
    /// # Errors
    ///
    /// [`MlError::DimensionMismatch`] if `flat` has the wrong length;
    /// [`MlError::InvalidValue`] if `layer_units` has fewer than two
    /// entries.
    pub fn from_flat(layer_units: &[usize], flat: &[f64]) -> Result<Self, MlError> {
        let layout = Layout::new(layer_units)?;
        if flat.len() != layout.len {
            return Err(MlError::DimensionMismatch {
                what: "flat parameter vector length",
                expected: layout.len,
                got: flat.len(),
            });
        }
        let mut coefs = Vec::with_capacity(layout.n_layers());
        let mut intercepts = Vec::with_capacity(layout.n_layers());
        for l in 0..layout.n_layers() {
            let (fan_in, fan_out) = (layout.units[l], layout.units[l + 1]);
            let off = layout.coef_off[l];
            coefs.push(Mat::from_fn(fan_in, fan_out, |i, j| {
                flat[off + i * fan_out + j]
            }));
            let boff = layout.bias_off[l];
            intercepts.push(flat[boff..boff + fan_out].to_vec());
        }
        Ok(Self { coefs, intercepts })
    }
}

/// Offsets of each layer's block inside the flat parameter vector.
#[derive(Debug, Clone)]
struct Layout {
    units: Vec<usize>,
    coef_off: Vec<usize>,
    bias_off: Vec<usize>,
    len: usize,
}

impl Layout {
    fn new(units: &[usize]) -> Result<Self, MlError> {
        if units.len() < 2 {
            return Err(MlError::InvalidValue {
                what: "layer_units must list the input width, the hidden widths, and \
                       the single output unit"
                    .to_string(),
            });
        }
        if units.contains(&0) {
            return Err(MlError::InvalidValue {
                what: "every layer must have at least one unit".to_string(),
            });
        }
        let n_layers = units.len() - 1;
        let mut coef_off = Vec::with_capacity(n_layers);
        let mut off = 0usize;
        for l in 0..n_layers {
            coef_off.push(off);
            off += units[l] * units[l + 1];
        }
        let mut bias_off = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            bias_off.push(off);
            off += units[l + 1];
        }
        Ok(Self {
            units: units.to_vec(),
            coef_off,
            bias_off,
            len: off,
        })
    }

    fn n_layers(&self) -> usize {
        self.units.len() - 1
    }
}

/// The network as a minimization objective over the flat parameter
/// vector, with reusable activation / delta workspaces.
struct Net<'a> {
    layout: &'a Layout,
    act: Activation,
    alpha: f64,
    /// Row-major `n x p` inputs.
    x: &'a [f64],
    y: &'a [f64],
    n: usize,
    /// `acts[l]` holds layer `l + 1`'s activations, row-major `n x units[l+1]`.
    acts: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
    grad: Vec<f64>,
}

impl<'a> Net<'a> {
    fn new(layout: &'a Layout, act: Activation, alpha: f64, x: &'a [f64], y: &'a [f64]) -> Self {
        let n = y.len();
        let acts = (0..layout.n_layers())
            .map(|l| vec![0.0; n * layout.units[l + 1]])
            .collect();
        let deltas = (0..layout.n_layers())
            .map(|l| vec![0.0; n * layout.units[l + 1]])
            .collect();
        Self {
            layout,
            act,
            alpha,
            x,
            y,
            n,
            acts,
            deltas,
            grad: vec![0.0; layout.len],
        }
    }

    /// Forward pass; afterwards `self.acts[last]` holds the outputs.
    fn forward(&mut self, theta: &[f64]) {
        let n = self.n;
        let n_layers = self.layout.n_layers();
        for l in 0..n_layers {
            let (fan_in, fan_out) = (self.layout.units[l], self.layout.units[l + 1]);
            let w = &theta[self.layout.coef_off[l]..self.layout.coef_off[l] + fan_in * fan_out];
            let b = &theta[self.layout.bias_off[l]..self.layout.bias_off[l] + fan_out];
            let hidden = l + 1 < n_layers;
            let act = self.act;
            // Split borrows: the input of layer l is x or acts[l - 1].
            let (prev, rest) = self.acts.split_at_mut(l);
            let out = &mut rest[0];
            let inp: &[f64] = if l == 0 { self.x } else { &prev[l - 1] };
            for r in 0..n {
                let xr = &inp[r * fan_in..(r + 1) * fan_in];
                let orow = &mut out[r * fan_out..(r + 1) * fan_out];
                orow.copy_from_slice(b);
                for (i, &xi) in xr.iter().enumerate() {
                    let wrow = &w[i * fan_out..(i + 1) * fan_out];
                    for (o, &wij) in orow.iter_mut().zip(wrow) {
                        *o += xi * wij;
                    }
                }
                if hidden {
                    for o in orow.iter_mut() {
                        *o = act.apply(*o);
                    }
                }
            }
        }
    }

    fn outputs(&self) -> &[f64] {
        &self.acts[self.layout.n_layers() - 1]
    }

    /// `0.5 * mean((y - yhat)^2)` at the current activations.
    fn data_fit(&self) -> f64 {
        let out = self.outputs();
        let sse: f64 = out.iter().zip(self.y).map(|(o, t)| (o - t) * (o - t)).sum();
        0.5 * sse / self.n as f64
    }

    fn penalty(&self, theta: &[f64]) -> f64 {
        let mut ss = 0.0;
        for l in 0..self.layout.n_layers() {
            let (fan_in, fan_out) = (self.layout.units[l], self.layout.units[l + 1]);
            let w = &theta[self.layout.coef_off[l]..self.layout.coef_off[l] + fan_in * fan_out];
            ss += w.iter().map(|v| v * v).sum::<f64>();
        }
        0.5 * self.alpha * ss / self.n as f64
    }

    /// The objective at `theta`.
    fn loss(&mut self, theta: &[f64]) -> f64 {
        self.forward(theta);
        self.data_fit() + self.penalty(theta)
    }

    /// The objective and its gradient (left in `self.grad`).
    fn loss_grad(&mut self, theta: &[f64]) -> f64 {
        self.forward(theta);
        let loss = self.data_fit() + self.penalty(theta);
        let n = self.n;
        let nf = n as f64;
        let n_layers = self.layout.n_layers();
        let last = n_layers - 1;
        // delta_last = yhat - y
        {
            let out = &self.acts[last];
            let d = &mut self.deltas[last];
            for r in 0..n {
                d[r] = out[r] - self.y[r];
            }
        }
        for l in (0..n_layers).rev() {
            let (fan_in, fan_out) = (self.layout.units[l], self.layout.units[l + 1]);
            let coff = self.layout.coef_off[l];
            let boff = self.layout.bias_off[l];
            let w = &theta[coff..coff + fan_in * fan_out];
            let inp: &[f64] = if l == 0 { self.x } else { &self.acts[l - 1] };
            let delta = &self.deltas[l];
            // dW = (inp' delta + alpha W) / n ; db = mean_rows(delta)
            let gw = &mut self.grad[coff..coff + fan_in * fan_out];
            for (g, &wij) in gw.iter_mut().zip(w) {
                *g = self.alpha * wij;
            }
            for r in 0..n {
                let xr = &inp[r * fan_in..(r + 1) * fan_in];
                let dr = &delta[r * fan_out..(r + 1) * fan_out];
                for (i, &xi) in xr.iter().enumerate() {
                    let grow = &mut gw[i * fan_out..(i + 1) * fan_out];
                    for (g, &dj) in grow.iter_mut().zip(dr) {
                        *g += xi * dj;
                    }
                }
            }
            for g in gw.iter_mut() {
                *g /= nf;
            }
            let gb = &mut self.grad[boff..boff + fan_out];
            gb.iter_mut().for_each(|g| *g = 0.0);
            for r in 0..n {
                let dr = &delta[r * fan_out..(r + 1) * fan_out];
                for (g, &dj) in gb.iter_mut().zip(dr) {
                    *g += dj;
                }
            }
            for g in gb.iter_mut() {
                *g /= nf;
            }
            // delta_{l-1} = (delta_l W') * phi'(acts[l-1])
            if l > 0 {
                let act = self.act;
                let prev_act = &self.acts[l - 1];
                let (lower, upper) = self.deltas.split_at_mut(l);
                let dprev = &mut lower[l - 1];
                let delta = &upper[0];
                for r in 0..n {
                    let dr = &delta[r * fan_out..(r + 1) * fan_out];
                    let prow = &mut dprev[r * fan_in..(r + 1) * fan_in];
                    let arow = &prev_act[r * fan_in..(r + 1) * fan_in];
                    for i in 0..fan_in {
                        let wrow = &w[i * fan_out..(i + 1) * fan_out];
                        let s: f64 = wrow.iter().zip(dr).map(|(a, b)| a * b).sum();
                        prow[i] = s * act.derivative_from_output(arow[i]);
                    }
                }
            }
        }
        loss
    }
}

impl ObjectiveFn for Net<'_> {
    fn value(&mut self, theta: &[f64]) -> f64 {
        self.loss(theta)
    }

    fn gradient(&mut self, theta: &[f64]) -> Option<Vec<f64>> {
        self.loss_grad(theta);
        Some(self.grad.clone())
    }
}

/// Copies a faer matrix into a row-major buffer.
fn row_major(x: MatRef<'_, f64>) -> Vec<f64> {
    let (n, p) = (x.nrows(), x.ncols());
    let mut out = Vec::with_capacity(n * p);
    for i in 0..n {
        for j in 0..p {
            out.push(x[(i, j)]);
        }
    }
    out
}

fn check_matrix(x: MatRef<'_, f64>, what: &'static str, p: usize) -> Result<(), MlError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(MlError::EmptyInput { what });
    }
    if x.ncols() != p {
        return Err(MlError::DimensionMismatch {
            what: "column count must match x",
            expected: p,
            got: x.ncols(),
        });
    }
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            if !x[(i, j)].is_finite() {
                return Err(MlError::NonFinite { what });
            }
        }
    }
    Ok(())
}

/// The forward pass of a network with the given weights: the predicted
/// output for every row of `x` (`n x n_inputs`).
///
/// Pinned against scikit-learn `MLPRegressor.predict` at its fitted
/// weights (`fixtures/neural.json`, 1e-12).
///
/// # Errors
///
/// * shape errors from [`MlpWeights::layer_units`];
/// * [`MlError::DimensionMismatch`] if `x` has a different column count
///   than the first layer's fan-in;
/// * [`MlError::EmptyInput`] / [`MlError::NonFinite`] on a bad `x`.
pub fn mlp_forward(
    weights: &MlpWeights,
    activation: Activation,
    x: MatRef<'_, f64>,
) -> Result<Vec<f64>, MlError> {
    let units = weights.layer_units()?;
    check_matrix(x, "x", units[0])?;
    let layout = Layout::new(&units)?;
    let theta = weights.to_flat();
    let xr = row_major(x);
    let y = vec![0.0; x.nrows()];
    let mut net = Net::new(&layout, activation, 0.0, &xr, &y);
    net.forward(&theta);
    Ok(net.outputs().to_vec())
}

/// The scikit-learn objective `(1/(2n)) sum (y - f(x))^2 +
/// (alpha/(2n)) sum_l ||W_l||_F^2` at the given weights.
///
/// # Errors
///
/// As [`mlp_loss_gradient`].
pub fn mlp_loss(
    weights: &MlpWeights,
    activation: Activation,
    x: MatRef<'_, f64>,
    y: &[f64],
    alpha: f64,
) -> Result<f64, MlError> {
    let pre = prepare_eval(weights, x, y, alpha)?;
    let mut net = Net::new(&pre.layout, activation, alpha, &pre.x_rows, y);
    Ok(net.loss(&pre.theta))
}

/// The objective and its analytic gradient (back-propagation) at the
/// given weights, the gradient returned in the same layout as the weights.
///
/// Pinned against scikit-learn's own `_backprop` at its fitted and at
/// random weights (`fixtures/neural.json`, 1e-10), and against a central
/// finite difference of [`mlp_loss`] (1e-6 relative on the smooth
/// activations).
///
/// # Errors
///
/// * shape errors from [`MlpWeights::layer_units`];
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on a bad `x` or `y`;
/// * [`MlError::InvalidArgument`] if `alpha` is negative or non-finite.
pub fn mlp_loss_gradient(
    weights: &MlpWeights,
    activation: Activation,
    x: MatRef<'_, f64>,
    y: &[f64],
    alpha: f64,
) -> Result<(f64, MlpWeights), MlError> {
    let pre = prepare_eval(weights, x, y, alpha)?;
    let mut net = Net::new(&pre.layout, activation, alpha, &pre.x_rows, y);
    let loss = net.loss_grad(&pre.theta);
    let grad = MlpWeights::from_flat(&pre.units, &net.grad)?;
    Ok((loss, grad))
}

/// Validated inputs of an objective evaluation at explicit weights.
struct Prepared {
    units: Vec<usize>,
    layout: Layout,
    theta: Vec<f64>,
    x_rows: Vec<f64>,
}

fn prepare_eval(
    weights: &MlpWeights,
    x: MatRef<'_, f64>,
    y: &[f64],
    alpha: f64,
) -> Result<Prepared, MlError> {
    let units = weights.layer_units()?;
    check_xy(x, y)?;
    if x.ncols() != units[0] {
        return Err(MlError::DimensionMismatch {
            what: "column count of x must equal the first layer's fan-in",
            expected: units[0],
            got: x.ncols(),
        });
    }
    check_alpha(alpha)?;
    let layout = Layout::new(&units)?;
    Ok(Prepared {
        units,
        layout,
        theta: weights.to_flat(),
        x_rows: row_major(x),
    })
}

fn check_alpha(alpha: f64) -> Result<(), MlError> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(MlError::InvalidArgument {
            what: "alpha must be finite and non-negative",
        });
    }
    Ok(())
}

/// Configuration of [`mlp_regression`].
#[derive(Debug, Clone, PartialEq)]
pub struct MlpOptions {
    /// Hidden-layer widths: one or two entries, each at least 1.
    pub hidden: Vec<usize>,
    /// Hidden activation.
    pub activation: Activation,
    /// L2 penalty on the weights (scikit-learn's `alpha`; intercepts are
    /// not penalized). Must be finite and non-negative.
    pub alpha: f64,
    /// Optimizer.
    pub solver: Solver,
    /// Adam step size. `None` resolves to [`DEFAULT_LEARNING_RATE`] under
    /// Adam; `Some` under L-BFGS is refused (the solver has no learning
    /// rate).
    pub learning_rate: Option<f64>,
    /// Adam mini-batch size. `None` takes one full-batch step per epoch;
    /// `Some(b)` shuffles the training rows every epoch (seeded) and steps
    /// through `b`-row chunks. Refused under L-BFGS.
    pub batch_size: Option<usize>,
    /// Epoch budget under Adam; iteration budget under L-BFGS. At least 1.
    pub max_epochs: usize,
    /// Fraction of the sample held out as the *last* rows for early
    /// stopping, in `[0, 0.5]`; `0` disables early stopping.
    pub validation_fraction: f64,
    /// Epochs without a validation improvement before stopping. `None`
    /// resolves to [`DEFAULT_PATIENCE`] under Adam; `Some` under L-BFGS
    /// is refused (L-BFGS runs to its own convergence test).
    pub patience: Option<usize>,
    /// Ensemble size (independent seed substreams). At least 1.
    pub n_seeds: usize,
    /// Root seed of the Philox substreams.
    pub seed: u64,
    /// Standardize `x` columns and `y` with statistics of the training
    /// rows only, replayed on the validation rows and on `x_test`.
    pub standardize: bool,
}

impl Default for MlpOptions {
    fn default() -> Self {
        Self {
            hidden: vec![16],
            activation: Activation::Tanh,
            alpha: 1e-4,
            solver: Solver::Adam,
            learning_rate: None,
            batch_size: None,
            max_epochs: 500,
            validation_fraction: 0.2,
            patience: None,
            n_seeds: 5,
            seed: 0,
            standardize: true,
        }
    }
}

/// Result of [`mlp_regression`].
#[derive(Debug, Clone, PartialEq)]
pub struct MlpFit {
    /// Ensemble-mean prediction for every row of `x` (training and
    /// validation rows alike), on the original `y` scale.
    pub fitted: Vec<f64>,
    /// Ensemble-mean prediction for each row of `x_test`, when given.
    pub predicted: Option<Vec<f64>>,
    /// Per-member predictions on `x_test` (`n_seeds` vectors of `n_test`).
    pub member_predictions: Option<Vec<Vec<f64>>>,
    /// Per-member training objective per epoch (under L-BFGS: two entries,
    /// the objective at the initial and at the returned weights).
    pub train_loss_path: Vec<Vec<f64>>,
    /// Per-member validation loss (`0.5 * mean((y - yhat)^2)` on the
    /// validation rows, standardized units when `standardize`) per epoch;
    /// empty when `validation_fraction = 0`. Under L-BFGS: two entries, as
    /// above.
    pub validation_loss_path: Vec<Vec<f64>>,
    /// Per-member epoch whose weights were kept (1-based; under L-BFGS the
    /// iteration count; with no validation split, the epochs run).
    pub best_epoch: Vec<usize>,
    /// Per-member: `true` if early stopping fired (Adam) or the optimizer
    /// met its convergence test (L-BFGS); `false` if the member ran out of
    /// `max_epochs`.
    pub converged: Vec<bool>,
    /// Weights plus intercepts of one member.
    pub n_parameters: usize,
    /// Per-member fitted weights in scikit-learn layout, on the
    /// standardized scale when `standardize`.
    pub weights: Vec<MlpWeights>,
    /// Number of training rows (the first `n - n_validation`).
    pub n_train: usize,
    /// Number of validation rows (the last `floor(validation_fraction * n)`).
    pub n_validation: usize,
    /// Column means of `x` on the training rows (zeros when not
    /// standardizing).
    pub x_mean: Vec<f64>,
    /// Column scales of `x` on the training rows (ones when not
    /// standardizing; a constant column is recorded as 1).
    pub x_scale: Vec<f64>,
    /// Mean of `y` on the training rows (0 when not standardizing).
    pub y_mean: f64,
    /// Population standard deviation of `y` on the training rows (1 when
    /// not standardizing or when `y` is constant on them).
    pub y_scale: f64,
}

/// Validates `hidden` (one or two layers, every width at least 1).
fn check_hidden(hidden: &[usize]) -> Result<(), MlError> {
    if hidden.is_empty() {
        return Err(MlError::InvalidValue {
            what: "hidden lists no hidden layer (empty); pass one or two widths, e.g. \
                   hidden=(16,)"
                .to_string(),
        });
    }
    if hidden.len() > MAX_HIDDEN_LAYERS {
        return Err(MlError::InvalidValue {
            what: format!(
                "hidden lists {} layers but this native net is limited to {} hidden \
                 layers by design (the 'NN' of the macro horse races is shallow; \
                 deeper models belong to the torch adapters outside core); pass one or \
                 two widths, e.g. hidden=(32, 16)",
                hidden.len(),
                MAX_HIDDEN_LAYERS
            ),
        });
    }
    if let Some(i) = hidden.iter().position(|&h| h == 0) {
        return Err(MlError::InvalidValue {
            what: format!("hidden[{i}] is 0 units; every hidden layer needs at least one unit"),
        });
    }
    Ok(())
}

/// Smallest `n` for which the split `n - floor(vf n) >= MIN_TRAIN_ROWS`
/// (and `floor(vf n) >= 1` when `vf > 0`) is feasible.
fn required_n(vf: f64) -> usize {
    let mut m = 1usize;
    while m <= 10_000_000 {
        let nv = ((m as f64) * vf).floor() as usize;
        if m - nv >= MIN_TRAIN_ROWS && (vf == 0.0 || nv >= 1) {
            return m;
        }
        m += 1;
    }
    m
}

/// `(n_train, n_validation)` for `n` rows, or the insufficiency error.
fn split_counts(n: usize, vf: f64) -> Result<(usize, usize), MlError> {
    let n_val = ((n as f64) * vf).floor() as usize;
    let n_train = n.saturating_sub(n_val);
    if n_train < MIN_TRAIN_ROWS || (vf > 0.0 && n_val < 1) {
        return Err(MlError::InsufficientData {
            needed: required_n(vf),
            got: n,
            what: "mlp_regression",
        });
    }
    Ok((n_train, n_val))
}

/// Resolved per-solver training knobs.
struct Resolved {
    learning_rate: f64,
    batch_size: Option<usize>,
    patience: usize,
}

fn resolve_solver_args(opts: &MlpOptions, n_train: usize) -> Result<Resolved, MlError> {
    match opts.solver {
        Solver::Lbfgs => {
            if let Some(lr) = opts.learning_rate {
                return Err(MlError::InvalidValue {
                    what: format!(
                        "learning_rate={lr} has no effect under solver=\"lbfgs\": L-BFGS \
                         chooses its step by a strong-Wolfe line search, so the value \
                         would be silently discarded. Drop learning_rate (leave it None) \
                         or use solver=\"adam\""
                    ),
                });
            }
            if let Some(b) = opts.batch_size {
                return Err(MlError::InvalidValue {
                    what: format!(
                        "batch_size={b} has no effect under solver=\"lbfgs\": L-BFGS \
                         minimizes the full-batch objective, so the value would be \
                         silently discarded. Drop batch_size (leave it None) or use \
                         solver=\"adam\" for mini-batch training"
                    ),
                });
            }
            if let Some(p) = opts.patience {
                return Err(MlError::InvalidValue {
                    what: format!(
                        "patience={p} has no effect under solver=\"lbfgs\": there are no \
                         epochs to be patient over — L-BFGS stops on its own gradient / \
                         function-decrease tests, and max_epochs caps its iterations. \
                         Drop patience (leave it None) or use solver=\"adam\""
                    ),
                });
            }
            Ok(Resolved {
                learning_rate: DEFAULT_LEARNING_RATE,
                batch_size: None,
                patience: DEFAULT_PATIENCE,
            })
        }
        Solver::Adam => {
            let learning_rate = opts.learning_rate.unwrap_or(DEFAULT_LEARNING_RATE);
            if !learning_rate.is_finite() || learning_rate <= 0.0 {
                return Err(MlError::InvalidArgument {
                    what: "learning_rate must be finite and positive",
                });
            }
            if let Some(b) = opts.batch_size {
                if b == 0 {
                    return Err(MlError::InvalidArgument {
                        what: "batch_size must be at least 1 (or None for full batch)",
                    });
                }
                if b > n_train {
                    return Err(MlError::InvalidValue {
                        what: format!(
                            "batch_size={b} exceeds the {n_train} training rows (the rows \
                             left after the temporal validation split); pass \
                             batch_size=None for a full-batch step or a value at most \
                             {n_train}"
                        ),
                    });
                }
            }
            let patience = opts.patience.unwrap_or(DEFAULT_PATIENCE);
            if patience == 0 {
                return Err(MlError::InvalidArgument {
                    what: "patience must be at least 1",
                });
            }
            Ok(Resolved {
                learning_rate,
                batch_size: opts.batch_size,
                patience,
            })
        }
    }
}

/// Glorot-uniform initialization (scikit-learn `_init_coef` bounds) from
/// the member's stream: per layer, the coefficient matrix row-major, then
/// the intercepts.
fn init_weights(layout: &Layout, act: Activation, stream: &mut Stream) -> Vec<f64> {
    let mut theta = vec![0.0; layout.len];
    let factor = act.glorot_factor();
    for l in 0..layout.n_layers() {
        let (fan_in, fan_out) = (layout.units[l], layout.units[l + 1]);
        let bound = (factor / (fan_in + fan_out) as f64).sqrt();
        let coff = layout.coef_off[l];
        for v in &mut theta[coff..coff + fan_in * fan_out] {
            *v = -bound + 2.0 * bound * stream.uniform_f64();
        }
        let boff = layout.bias_off[l];
        for v in &mut theta[boff..boff + fan_out] {
            *v = -bound + 2.0 * bound * stream.uniform_f64();
        }
    }
    theta
}

/// Seeded Fisher-Yates shuffle.
fn shuffle(idx: &mut [usize], stream: &mut Stream) {
    for i in (1..idx.len()).rev() {
        let j = (stream.uniform_f64() * (i + 1) as f64).floor() as usize;
        let j = j.min(i);
        idx.swap(i, j);
    }
}

struct AdamState {
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
}

impl AdamState {
    fn new(len: usize) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
            t: 0,
        }
    }

    /// scikit-learn `AdamOptimizer.get_updates`.
    fn step(&mut self, theta: &mut [f64], grad: &[f64], lr: f64) {
        self.t += 1;
        let t = self.t as i32;
        let lr_t = lr * (1.0 - ADAM_BETA2.powi(t)).sqrt() / (1.0 - ADAM_BETA1.powi(t));
        for i in 0..theta.len() {
            self.m[i] = ADAM_BETA1 * self.m[i] + (1.0 - ADAM_BETA1) * grad[i];
            self.v[i] = ADAM_BETA2 * self.v[i] + (1.0 - ADAM_BETA2) * grad[i] * grad[i];
            theta[i] -= lr_t * self.m[i] / (self.v[i].sqrt() + ADAM_EPS);
        }
    }
}

struct MemberOutcome {
    theta: Vec<f64>,
    train_path: Vec<f64>,
    val_path: Vec<f64>,
    best_epoch: usize,
    converged: bool,
}

/// Validation loss (data-fit term only) at `theta`, or `None` without a
/// validation split.
fn validation_loss(val: &mut Option<Net<'_>>, theta: &[f64]) -> Option<f64> {
    val.as_mut().map(|net| {
        net.forward(theta);
        net.data_fit()
    })
}

#[allow(clippy::too_many_arguments)]
fn train_member(
    member: usize,
    layout: &Layout,
    act: Activation,
    opts: &MlpOptions,
    res: &Resolved,
    x_train: &[f64],
    y_train: &[f64],
    x_val: &[f64],
    y_val: &[f64],
    stream: &mut Stream,
) -> Result<MemberOutcome, MlError> {
    let p = layout.units[0];
    let n_train = y_train.len();
    let mut theta = init_weights(layout, act, stream);
    let mut full = Net::new(layout, act, opts.alpha, x_train, y_train);
    let mut val: Option<Net<'_>> = if y_val.is_empty() {
        None
    } else {
        Some(Net::new(layout, act, opts.alpha, x_val, y_val))
    };

    match opts.solver {
        Solver::Lbfgs => {
            let f0 = full.loss(&theta);
            let v0 = validation_loss(&mut val, &theta);
            let lopts = LbfgsOptions {
                grad_tol: LBFGS_GRAD_TOL,
                f_tol: LBFGS_F_TOL,
                max_iter: Some(opts.max_epochs),
                ..LbfgsOptions::default()
            };
            let result = lbfgs(&mut full, &theta, &lopts).map_err(|e| MlError::InvalidValue {
                what: format!("L-BFGS could not run: {e}"),
            })?;
            theta = result.x;
            let f1 = full.loss(&theta);
            if !f1.is_finite() || theta.iter().any(|v| !v.is_finite()) {
                return Err(MlError::Diverged {
                    member,
                    epoch: result.iterations,
                });
            }
            let v1 = validation_loss(&mut val, &theta);
            let val_path = match (v0, v1) {
                (Some(a), Some(b)) => vec![a, b],
                _ => Vec::new(),
            };
            Ok(MemberOutcome {
                theta,
                train_path: vec![f0, f1],
                val_path,
                best_epoch: result.iterations,
                converged: result.converged,
            })
        }
        Solver::Adam => {
            let mut adam = AdamState::new(layout.len);
            let mut train_path = Vec::with_capacity(opts.max_epochs);
            let mut val_path = Vec::with_capacity(opts.max_epochs);
            let mut best_val = f64::INFINITY;
            let mut best_theta = theta.clone();
            let mut best_epoch = 0usize;
            let mut no_improve = 0usize;
            let mut converged = false;
            let mut epochs_run = 0usize;
            // Mini-batch workspace (allocated once; the last chunk may be
            // shorter, so a fresh Net is built per chunk on the copied rows).
            let mut idx: Vec<usize> = (0..n_train).collect();
            let mut xb: Vec<f64> = Vec::new();
            let mut yb: Vec<f64> = Vec::new();
            for epoch in 1..=opts.max_epochs {
                epochs_run = epoch;
                match res.batch_size {
                    None => {
                        // The full-batch gradient pass evaluates the objective
                        // at the weights this epoch starts from, which are the
                        // weights the previous epoch ended with: record it as
                        // the previous epoch's training loss instead of paying
                        // a second forward pass per epoch.
                        let at_start = full.loss_grad(&theta);
                        if epoch > 1 {
                            train_path.push(at_start);
                        }
                        adam.step(&mut theta, &full.grad, res.learning_rate);
                    }
                    Some(b) => {
                        shuffle(&mut idx, stream);
                        for chunk in idx.chunks(b) {
                            xb.clear();
                            yb.clear();
                            for &r in chunk {
                                xb.extend_from_slice(&x_train[r * p..(r + 1) * p]);
                                yb.push(y_train[r]);
                            }
                            let mut batch = Net::new(layout, act, opts.alpha, &xb, &yb);
                            batch.loss_grad(&theta);
                            adam.step(&mut theta, &batch.grad, res.learning_rate);
                        }
                    }
                }
                // Mini-batch epochs (and the final full-batch epoch, below)
                // evaluate the training objective at the end of the epoch.
                if res.batch_size.is_some() {
                    train_path.push(full.loss(&theta));
                }
                if theta.iter().any(|v| !v.is_finite()) {
                    return Err(MlError::Diverged { member, epoch });
                }
                if let Some(v) = validation_loss(&mut val, &theta) {
                    val_path.push(v);
                    let improved = best_val.is_infinite()
                        || v < best_val - EARLY_STOPPING_TOL * best_val.abs();
                    if improved {
                        best_val = v;
                        best_theta.copy_from_slice(&theta);
                        best_epoch = epoch;
                        no_improve = 0;
                    } else {
                        no_improve += 1;
                        if no_improve >= res.patience {
                            converged = true;
                            break;
                        }
                    }
                }
            }
            if res.batch_size.is_none() {
                // Close the full-batch path with the objective at the final
                // weights (every epoch is then recorded at its end).
                train_path.push(full.loss(&theta));
            }
            if train_path.iter().any(|v| !v.is_finite()) {
                return Err(MlError::Diverged {
                    member,
                    epoch: epochs_run,
                });
            }
            if val.is_some() {
                theta = best_theta;
            } else {
                best_epoch = epochs_run;
            }
            Ok(MemberOutcome {
                theta,
                train_path,
                val_path,
                best_epoch,
                converged,
            })
        }
    }
}

/// Fits the seed-ensemble feed-forward regressor described in the
/// [module docs](self): `x` is `n x p`, `y` length `n`, `x_test` an
/// optional `n_test x p` matrix to predict.
///
/// The LAST `floor(validation_fraction * n)` rows are the validation set;
/// the scaler is fit on the remaining leading rows only. Returns the
/// ensemble-mean `fitted` values for every row of `x` and, when `x_test`
/// is given, the ensemble-mean `predicted` values plus every member's
/// predictions; see [`MlpFit`] for the diagnostics.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on bad `x`, `y`, or `x_test` (the message
///   names the array);
/// * [`MlError::InsufficientData`] when the temporal split leaves fewer
///   than two training rows or no validation row;
/// * [`MlError::InvalidValue`] for an empty or too-deep `hidden`, a zero
///   width, a `batch_size` beyond the training rows, or an epoch-wise
///   argument passed under `solver = Lbfgs`;
/// * [`MlError::InvalidArgument`] for a bad `alpha`, `learning_rate`,
///   `validation_fraction`, `max_epochs`, `patience`, or `n_seeds`;
/// * [`MlError::Diverged`] if a member's loss became non-finite.
pub fn mlp_regression(
    x: MatRef<'_, f64>,
    y: &[f64],
    x_test: Option<MatRef<'_, f64>>,
    opts: &MlpOptions,
) -> Result<MlpFit, MlError> {
    let (n, p) = check_xy(x, y)?;
    if let Some(xt) = x_test {
        check_matrix(xt, "x_test", p)?;
    }
    check_hidden(&opts.hidden)?;
    check_alpha(opts.alpha)?;
    if opts.max_epochs == 0 {
        return Err(MlError::InvalidArgument {
            what: "max_epochs must be at least 1",
        });
    }
    if !opts.validation_fraction.is_finite() || !(0.0..=0.5).contains(&opts.validation_fraction) {
        return Err(MlError::InvalidArgument {
            what: "validation_fraction must lie in [0, 0.5] (the LAST rows are held out)",
        });
    }
    if opts.n_seeds == 0 {
        return Err(MlError::InvalidArgument {
            what: "n_seeds must be at least 1",
        });
    }
    let (n_train, n_val) = split_counts(n, opts.validation_fraction)?;
    let res = resolve_solver_args(opts, n_train)?;

    // --- standardization, fit on the training rows only ---------------
    let (x_all, x_test_rows, x_mean, x_scale) = if opts.standardize {
        let x_tr = Mat::from_fn(n_train, p, |i, j| x[(i, j)]);
        let scaler = Scaler::fit(x_tr.as_ref())?;
        let xs = scaler.transform(x)?;
        let xt = match x_test {
            Some(xt) => Some(row_major(scaler.transform(xt)?.as_ref())),
            None => None,
        };
        (
            row_major(xs.as_ref()),
            xt,
            scaler.means().to_vec(),
            scaler.scales().to_vec(),
        )
    } else {
        (
            row_major(x),
            x_test.map(row_major),
            vec![0.0; p],
            vec![1.0; p],
        )
    };
    let (y_mean, y_scale) = if opts.standardize {
        let mean = y[..n_train].iter().sum::<f64>() / n_train as f64;
        let var = y[..n_train]
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / n_train as f64;
        let sd = var.sqrt();
        (mean, if sd > 0.0 { sd } else { 1.0 })
    } else {
        (0.0, 1.0)
    };
    let y_std: Vec<f64> = y.iter().map(|v| (v - y_mean) / y_scale).collect();
    let (x_train, x_val) = x_all.split_at(n_train * p);
    let (y_train, y_val) = y_std.split_at(n_train);

    let mut units = Vec::with_capacity(opts.hidden.len() + 2);
    units.push(p);
    units.extend_from_slice(&opts.hidden);
    units.push(1);
    let layout = Layout::new(&units)?;

    // --- the ensemble ---------------------------------------------------
    let mut streams =
        Stream::substreams(opts.seed, opts.n_seeds).map_err(|_| MlError::InvalidArgument {
            what: "n_seeds exceeds the substream spawn limit",
        })?;
    let mut weights = Vec::with_capacity(opts.n_seeds);
    let mut train_loss_path = Vec::with_capacity(opts.n_seeds);
    let mut validation_loss_path = Vec::with_capacity(opts.n_seeds);
    let mut best_epoch = Vec::with_capacity(opts.n_seeds);
    let mut converged = Vec::with_capacity(opts.n_seeds);
    let mut fitted = vec![0.0; n];
    let n_test = x_test.map_or(0, |xt| xt.nrows());
    let mut member_predictions: Vec<Vec<f64>> = Vec::with_capacity(opts.n_seeds);
    for (member, stream) in streams.iter_mut().enumerate() {
        let out = train_member(
            member,
            &layout,
            opts.activation,
            opts,
            &res,
            x_train,
            y_train,
            x_val,
            y_val,
            stream,
        )?;
        // Predictions on every row of x and on x_test, back on the y scale.
        {
            let zeros = vec![0.0; n];
            let mut net = Net::new(&layout, opts.activation, opts.alpha, &x_all, &zeros);
            net.forward(&out.theta);
            for (f, o) in fitted.iter_mut().zip(net.outputs()) {
                *f += (o * y_scale + y_mean) / opts.n_seeds as f64;
            }
        }
        if let Some(xt) = &x_test_rows {
            let zeros = vec![0.0; n_test];
            let mut net = Net::new(&layout, opts.activation, opts.alpha, xt, &zeros);
            net.forward(&out.theta);
            member_predictions.push(net.outputs().iter().map(|o| o * y_scale + y_mean).collect());
        }
        weights.push(MlpWeights::from_flat(&units, &out.theta)?);
        train_loss_path.push(out.train_path);
        validation_loss_path.push(out.val_path);
        best_epoch.push(out.best_epoch);
        converged.push(out.converged);
    }
    let (predicted, member_predictions) = if x_test.is_some() {
        let mut mean = vec![0.0; n_test];
        for mp in &member_predictions {
            for (m, v) in mean.iter_mut().zip(mp) {
                *m += v / opts.n_seeds as f64;
            }
        }
        (Some(mean), Some(member_predictions))
    } else {
        (None, None)
    };

    Ok(MlpFit {
        fitted,
        predicted,
        member_predictions,
        train_loss_path,
        validation_loss_path,
        best_epoch,
        converged,
        n_parameters: layout.len,
        weights,
        n_train,
        n_validation: n_val,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn flat_roundtrip_matches_sklearn_layout() {
        let w = MlpWeights {
            coefs: vec![
                Mat::from_fn(2, 3, |i, j| (i * 3 + j) as f64),
                Mat::from_fn(3, 1, |i, _| 10.0 + i as f64),
            ],
            intercepts: vec![vec![20.0, 21.0, 22.0], vec![30.0]],
        };
        let flat = w.to_flat();
        assert_eq!(
            flat,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0]
        );
        let back = MlpWeights::from_flat(&[2, 3, 1], &flat).unwrap();
        assert_eq!(back, w);
        assert_eq!(w.n_parameters(), 13);
    }

    #[test]
    fn required_n_is_the_smallest_feasible_sample() {
        for &vf in &[0.0, 0.1, 0.2, 0.25, 0.5] {
            let need = required_n(vf);
            assert!(split_counts(need, vf).is_ok(), "vf={vf} need={need}");
            if need > 1 {
                assert!(split_counts(need - 1, vf).is_err(), "vf={vf} need={need}");
            }
        }
    }

    #[test]
    fn hidden_limits_are_taught() {
        assert!(matches!(
            check_hidden(&[]),
            Err(MlError::InvalidValue { .. })
        ));
        assert!(matches!(
            check_hidden(&[4, 4, 4]),
            Err(MlError::InvalidValue { .. })
        ));
        assert!(matches!(
            check_hidden(&[4, 0]),
            Err(MlError::InvalidValue { .. })
        ));
        assert!(check_hidden(&[4, 2]).is_ok());
    }
}
