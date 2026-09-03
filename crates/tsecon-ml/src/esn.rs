//! Echo state networks (reservoir computing): a fixed random recurrent
//! reservoir, a leaky-integrator tanh state recursion, and a ridge-trained
//! linear readout — nonlinear dynamics at linear-regression cost,
//! deterministic given the seed (Jaeger 2001; Lukoševičius 2012).
//!
//! # Model
//!
//! With inputs `u_t` (`p`-vector), reservoir size `N`, input matrix
//! `W_in` (`N x p`), reservoir matrix `W` (`N x N`), and leak rate
//! `a in (0, 1]`,
//!
//! ```text
//! s_t = (1 - a) s_{t-1} + a tanh(W s_{t-1} + W_in u_t),   s_0 = 0,
//! ```
//!
//! (Lukoševičius 2012, eqs. 2-3, without a reservoir bias unit — the same
//! recursion `reservoirpy`'s `Reservoir` runs with `bias = 0`). The
//! readout is the ridge regression of `y_t` on `z_t = [1, u_t, s_t]` over
//! the rows `t >= washout`,
//!
//! ```text
//! b = argmin ||y - Z b||^2 + ridge_alpha ||b||^2 ,
//! ```
//!
//! computed by the crate's [`ridge`](crate::ridge) closed form — the
//! scikit-learn `Ridge(fit_intercept=False)` objective (no `1/n` factor).
//! Note that the constant column is penalized like every other readout
//! coefficient (Lukoševičius eq. 9); at the default `ridge_alpha = 1e-6`
//! that is immaterial, and it keeps the objective identical to the one
//! `fixtures/ml.json` already pins.
//!
//! # Reservoir construction
//!
//! `W` is drawn from the seed's Philox stream: each entry is nonzero with
//! probability `sparsity` (its value standard normal) — `sparsity` is the
//! *connectivity*, the fraction of nonzero entries, `0.1` meaning 10% —
//! and the matrix is then rescaled so its spectral radius equals
//! `spectral_radius`. The radius is the modulus of the leading eigenvalue
//! from a dense Hessenberg-QR eigenvalue decomposition
//! ([`tsecon_linalg::spectral_radius`]), **not** a power iteration: the
//! leading eigenvalues of a random reservoir are typically a complex pair
//! inside a dense cluster (the circular law), where power iteration does
//! not converge to 1e-6 in any reasonable budget. The achieved radius is
//! recomputed on the scaled matrix and reported as
//! `spectral_radius_achieved` (pinned against `numpy.linalg.eigvals` at
//! 1e-6 in `fixtures/neural.json`). `W_in` is `N x p` uniform on
//! `[-input_scaling, input_scaling]`.
//!
//! The echo state property (states forgetting their initial condition) is
//! *usually* obtained with `spectral_radius < 1`, but it is neither
//! necessary nor sufficient (Yildiz, Jaeger & Kiebeling 2012); the washout
//! is what removes the dependence on `s_0 = 0` in practice, and values
//! above 1 are accepted rather than refused.
//!
//! # Prediction
//!
//! `x_test` rows are treated as the *continuation* of `x`: the state
//! recursion carries on from the last training state and the fitted
//! readout is applied. No washout is re-applied. `fitted` covers only the
//! rows that entered the readout fit (`t >= washout`), so it has
//! `n - washout` entries.
//!
//! # Reproducibility
//!
//! Single-threaded, every draw a pure function of `seed`: bit-identical on
//! the same build (tested); cross-platform differences are confined to the
//! last ulp of libm `tanh` and the eigenvalue routine.
//!
//! # Validation
//!
//! `fixtures/neural.json` pins [`esn_states`] and [`esn_readout`] on an
//! explicit small reservoir against a NumPy transcription of the
//! equations above (states 1e-12, readout 1e-10; the readout also
//! cross-checked there against scikit-learn `Ridge`; the state path
//! against `reservoirpy` when the generator could import it — the
//! fixture's `_meta.esn` says which) and [`spectral_radius`] against
//! `numpy.linalg.eigvals` (1e-6). The public estimator is validated by
//! property: NARMA-10 out-of-sample NRMSE below a documented bar, the seed
//! contract, and the achieved radius.

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_rng::Stream;

use crate::error::MlError;
use crate::ridge::ridge;
use crate::util::check_xy;

/// Fewest rows that must survive the washout for a readout fit.
const MIN_READOUT_ROWS: usize = 2;

/// Configuration of [`echo_state_network`].
#[derive(Debug, Clone, PartialEq)]
pub struct EsnOptions {
    /// Number of reservoir units `N`. At least 1.
    pub reservoir_size: usize,
    /// Target spectral radius of `W`. Finite and positive.
    pub spectral_radius: f64,
    /// Leak rate `a` in `(0, 1]`; `1` is the plain (non-leaky) ESN.
    pub leak_rate: f64,
    /// Input weights are uniform on `[-input_scaling, input_scaling]`.
    /// Finite and positive.
    pub input_scaling: f64,
    /// Connectivity of `W`: the probability each entry is nonzero, in
    /// `(0, 1]`.
    pub sparsity: f64,
    /// Leading rows discarded before the readout fit (transient of
    /// `s_0 = 0`). Must leave at least two rows.
    pub washout: usize,
    /// Ridge penalty of the readout (scikit-learn `Ridge` scale). Finite
    /// and non-negative.
    pub ridge_alpha: f64,
    /// Seed of the reservoir and input-weight draws.
    pub seed: u64,
}

impl Default for EsnOptions {
    fn default() -> Self {
        Self {
            reservoir_size: 200,
            spectral_radius: 0.9,
            leak_rate: 1.0,
            input_scaling: 1.0,
            sparsity: 0.1,
            washout: 50,
            ridge_alpha: 1e-6,
            seed: 0,
        }
    }
}

/// Result of [`echo_state_network`].
#[derive(Debug, Clone, PartialEq)]
pub struct EsnFit {
    /// Readout predictions on the training rows that entered the fit
    /// (`t >= washout`; length `n - washout`).
    pub fitted: Vec<f64>,
    /// Readout predictions on `x_test`, run as the continuation of `x`.
    pub predicted: Option<Vec<f64>>,
    /// Readout coefficients on `[1, u_t, s_t]` (length `1 + p + N`).
    pub readout: Vec<f64>,
    /// Spectral radius of the scaled reservoir, recomputed after scaling.
    pub spectral_radius_achieved: f64,
    /// Reservoir size `N`.
    pub reservoir_size: usize,
    /// Rows discarded before the readout fit.
    pub n_washout: usize,
    /// Rows the readout was fit on (`n - washout`).
    pub n_train: usize,
}

fn check_square_finite(w: MatRef<'_, f64>, what: &'static str) -> Result<usize, MlError> {
    let n = w.nrows();
    if n == 0 || w.ncols() == 0 {
        return Err(MlError::EmptyInput { what });
    }
    if w.ncols() != n {
        return Err(MlError::DimensionMismatch {
            what: "the reservoir matrix must be square",
            expected: n,
            got: w.ncols(),
        });
    }
    for j in 0..n {
        for i in 0..n {
            if !w[(i, j)].is_finite() {
                return Err(MlError::NonFinite { what });
            }
        }
    }
    Ok(n)
}

/// Spectral radius (modulus of the leading eigenvalue) of a square matrix,
/// through a dense eigenvalue decomposition.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on a bad `w`;
/// * [`MlError::DecompositionFailed`] if the eigenvalue iteration fails.
pub fn spectral_radius(w: MatRef<'_, f64>) -> Result<f64, MlError> {
    check_square_finite(w, "w")?;
    tsecon_linalg::spectral_radius(w).map_err(|_| MlError::DecompositionFailed {
        what: "reservoir spectral radius (eigenvalue decomposition)",
    })
}

/// Rescales `w` so its spectral radius equals `target`, returning the
/// scaled matrix and the radius recomputed on it.
///
/// # Errors
///
/// * as [`spectral_radius`];
/// * [`MlError::InvalidArgument`] if `target` is not finite and positive;
/// * [`MlError::InvalidValue`] if `w` has spectral radius zero (a nilpotent
///   or all-zero reservoir cannot be scaled to a positive radius).
pub fn scale_to_spectral_radius(
    w: MatRef<'_, f64>,
    target: f64,
) -> Result<(Mat<f64>, f64), MlError> {
    if !target.is_finite() || target <= 0.0 {
        return Err(MlError::InvalidArgument {
            what: "spectral_radius must be finite and positive",
        });
    }
    let rho = spectral_radius(w)?;
    if rho <= 0.0 {
        return Err(MlError::InvalidValue {
            what: format!(
                "the drawn reservoir has spectral radius 0 (every eigenvalue vanishes — \
                 with reservoir_size={} the sparse draw has no cycle to carry a \
                 nonzero eigenvalue), so it cannot be scaled to spectral_radius={target}; \
                 raise sparsity or reservoir_size, or change seed",
                w.nrows()
            ),
        });
    }
    let factor = target / rho;
    let scaled = Mat::from_fn(w.nrows(), w.ncols(), |i, j| w[(i, j)] * factor);
    let achieved = spectral_radius(scaled.as_ref())?;
    Ok((scaled, achieved))
}

/// Runs the leaky-integrator state recursion of the [module docs](self)
/// from a zero initial state: returns the `n x N` matrix of states
/// `s_1, ..., s_n` for the inputs `x` (`n x p`), with `w_in` (`N x p`)
/// and `w` (`N x N`).
///
/// Pinned against a NumPy transcription (and `reservoirpy` when
/// available) in `fixtures/neural.json` at 1e-12.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::NonFinite`] on empty or
///   non-finite inputs;
/// * [`MlError::DimensionMismatch`] if `w` is not `N x N` or `w_in` is
///   not `N x p`;
/// * [`MlError::InvalidArgument`] if `leak_rate` is outside `(0, 1]`.
pub fn esn_states(
    w_in: MatRef<'_, f64>,
    w: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    leak_rate: f64,
) -> Result<Mat<f64>, MlError> {
    let n_units = check_square_finite(w, "w")?;
    let s0 = vec![0.0; n_units];
    esn_states_from(w_in, w, x, leak_rate, &s0)
}

/// As [`esn_states`], but starting from the state `s0` (used to continue
/// the recursion across `x_test`).
///
/// # Errors
///
/// As [`esn_states`], plus [`MlError::DimensionMismatch`] if `s0` is not
/// length `N`.
pub fn esn_states_from(
    w_in: MatRef<'_, f64>,
    w: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    leak_rate: f64,
    s0: &[f64],
) -> Result<Mat<f64>, MlError> {
    let n_units = check_square_finite(w, "w")?;
    let (n, p) = (x.nrows(), x.ncols());
    if n == 0 || p == 0 {
        return Err(MlError::EmptyInput { what: "x" });
    }
    if w_in.nrows() != n_units || w_in.ncols() != p {
        return Err(MlError::DimensionMismatch {
            what: "w_in must be reservoir_size x n_inputs (rows checked here)",
            expected: n_units,
            got: w_in.nrows(),
        });
    }
    if w_in.ncols() != p {
        return Err(MlError::DimensionMismatch {
            what: "w_in columns must equal the input width",
            expected: p,
            got: w_in.ncols(),
        });
    }
    if s0.len() != n_units {
        return Err(MlError::DimensionMismatch {
            what: "initial state length must equal reservoir_size",
            expected: n_units,
            got: s0.len(),
        });
    }
    for j in 0..p {
        for i in 0..n {
            if !x[(i, j)].is_finite() {
                return Err(MlError::NonFinite { what: "x" });
            }
        }
        for i in 0..n_units {
            if !w_in[(i, j)].is_finite() {
                return Err(MlError::NonFinite { what: "w_in" });
            }
        }
    }
    check_leak(leak_rate)?;
    // Row-major copies for the hot loop.
    let w_rows: Vec<Vec<f64>> = (0..n_units)
        .map(|i| (0..n_units).map(|j| w[(i, j)]).collect())
        .collect();
    let w_in_rows: Vec<Vec<f64>> = (0..n_units)
        .map(|i| (0..p).map(|j| w_in[(i, j)]).collect())
        .collect();
    let mut s = s0.to_vec();
    let mut next = vec![0.0; n_units];
    let mut u = vec![0.0; p];
    let mut out = Mat::zeros(n, n_units);
    for t in 0..n {
        for (j, uj) in u.iter_mut().enumerate() {
            *uj = x[(t, j)];
        }
        for i in 0..n_units {
            let rec: f64 = w_rows[i].iter().zip(&s).map(|(a, b)| a * b).sum();
            let inp: f64 = w_in_rows[i].iter().zip(&u).map(|(a, b)| a * b).sum();
            next[i] = (1.0 - leak_rate) * s[i] + leak_rate * (rec + inp).tanh();
        }
        s.copy_from_slice(&next);
        for i in 0..n_units {
            out[(t, i)] = s[i];
        }
    }
    Ok(out)
}

fn check_leak(leak_rate: f64) -> Result<(), MlError> {
    if !leak_rate.is_finite() || leak_rate <= 0.0 || leak_rate > 1.0 {
        return Err(MlError::InvalidArgument {
            what: "leak_rate must lie in (0, 1]",
        });
    }
    Ok(())
}

/// Builds the readout design `Z_t = [1, u_t, s_t]` for `t >= washout`.
fn readout_design(states: MatRef<'_, f64>, x: MatRef<'_, f64>, washout: usize) -> Mat<f64> {
    let (n, p, n_units) = (x.nrows(), x.ncols(), states.ncols());
    let rows = n - washout;
    Mat::from_fn(rows, 1 + p + n_units, |r, c| {
        let t = washout + r;
        if c == 0 {
            1.0
        } else if c <= p {
            x[(t, c - 1)]
        } else {
            states[(t, c - 1 - p)]
        }
    })
}

/// Fits the ridge readout of the [module docs](self) on the rows
/// `t >= washout` of `[1, x_t, states_t]` against `y`, returning the
/// coefficient vector (length `1 + p + N`).
///
/// Pinned in `fixtures/neural.json` against the closed form
/// `(Z'Z + alpha I)^{-1} Z'y` (1e-10), which the generator cross-checked
/// against scikit-learn `Ridge`.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on bad `states`, `x`, or `y`;
/// * [`MlError::InvalidValue`] if `washout >= n`;
/// * [`MlError::InsufficientData`] if fewer than two rows survive the
///   washout;
/// * [`MlError::InvalidArgument`] / [`MlError::DecompositionFailed`] from
///   [`ridge`](crate::ridge).
pub fn esn_readout(
    states: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    y: &[f64],
    washout: usize,
    ridge_alpha: f64,
) -> Result<Vec<f64>, MlError> {
    let (n, _p) = check_xy(x, y)?;
    if states.nrows() != n {
        return Err(MlError::DimensionMismatch {
            what: "states must have one row per row of x",
            expected: n,
            got: states.nrows(),
        });
    }
    if states.ncols() == 0 {
        return Err(MlError::EmptyInput { what: "states" });
    }
    for j in 0..states.ncols() {
        for i in 0..n {
            if !states[(i, j)].is_finite() {
                return Err(MlError::NonFinite { what: "states" });
            }
        }
    }
    check_washout(n, washout)?;
    let z = readout_design(states, x, washout);
    ridge(z.as_ref(), &y[washout..], ridge_alpha)
}

fn check_washout(n: usize, washout: usize) -> Result<(), MlError> {
    if washout >= n {
        return Err(MlError::InvalidValue {
            what: format!(
                "washout={washout} discards every row: the readout is fit on rows \
                 washout..n-1 and x has only n={n} rows; pass washout < n - \
                 {MIN_READOUT_ROWS} (e.g. washout={}) or supply more observations",
                n.saturating_sub(MIN_READOUT_ROWS + 1)
            ),
        });
    }
    if n - washout < MIN_READOUT_ROWS {
        return Err(MlError::InsufficientData {
            needed: washout + MIN_READOUT_ROWS,
            got: n,
        });
    }
    Ok(())
}

/// One standard normal draw via Box-Muller (two 64-bit draws, the sine
/// partner discarded so the per-draw stream cost is fixed).
#[inline]
fn standard_normal(stream: &mut Stream) -> f64 {
    let u1 = 1.0 - stream.uniform_f64(); // (0, 1] keeps ln finite
    let u2 = stream.uniform_f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

/// Fits the echo state network of the [module docs](self): `x` is
/// `n x p`, `y` length `n`, `x_test` an optional `n_test x p`
/// continuation of `x` to predict.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on bad `x`, `y`, or `x_test` (the message
///   names the array);
/// * [`MlError::InvalidValue`] if `washout >= n` (the message names the
///   fix) or the drawn reservoir has radius zero;
/// * [`MlError::InsufficientData`] if fewer than two rows survive the
///   washout;
/// * [`MlError::InvalidArgument`] for a bad `reservoir_size`,
///   `spectral_radius`, `leak_rate`, `input_scaling`, `sparsity`, or
///   `ridge_alpha`;
/// * [`MlError::DecompositionFailed`] from the eigenvalue or SVD step.
pub fn echo_state_network(
    x: MatRef<'_, f64>,
    y: &[f64],
    x_test: Option<MatRef<'_, f64>>,
    opts: &EsnOptions,
) -> Result<EsnFit, MlError> {
    let (n, p) = check_xy(x, y)?;
    if let Some(xt) = x_test {
        if xt.nrows() == 0 || xt.ncols() == 0 {
            return Err(MlError::EmptyInput { what: "x_test" });
        }
        if xt.ncols() != p {
            return Err(MlError::DimensionMismatch {
                what: "x_test column count must match x",
                expected: p,
                got: xt.ncols(),
            });
        }
        for j in 0..p {
            for i in 0..xt.nrows() {
                if !xt[(i, j)].is_finite() {
                    return Err(MlError::NonFinite { what: "x_test" });
                }
            }
        }
    }
    if opts.reservoir_size == 0 {
        return Err(MlError::InvalidArgument {
            what: "reservoir_size must be at least 1",
        });
    }
    if !opts.spectral_radius.is_finite() || opts.spectral_radius <= 0.0 {
        return Err(MlError::InvalidArgument {
            what: "spectral_radius must be finite and positive",
        });
    }
    check_leak(opts.leak_rate)?;
    if !opts.input_scaling.is_finite() || opts.input_scaling <= 0.0 {
        return Err(MlError::InvalidArgument {
            what: "input_scaling must be finite and positive",
        });
    }
    if !opts.sparsity.is_finite() || opts.sparsity <= 0.0 || opts.sparsity > 1.0 {
        return Err(MlError::InvalidArgument {
            what: "sparsity (the reservoir connectivity) must lie in (0, 1]",
        });
    }
    if !opts.ridge_alpha.is_finite() || opts.ridge_alpha < 0.0 {
        return Err(MlError::InvalidArgument {
            what: "ridge_alpha must be finite and non-negative",
        });
    }
    check_washout(n, opts.washout)?;

    // --- reservoir draws: W (sparse normal), then W_in (uniform) --------
    let n_units = opts.reservoir_size;
    let mut stream = Stream::new(opts.seed);
    let mut w_raw = Mat::zeros(n_units, n_units);
    for i in 0..n_units {
        for j in 0..n_units {
            if stream.uniform_f64() < opts.sparsity {
                w_raw[(i, j)] = standard_normal(&mut stream);
            }
        }
    }
    let (w, achieved) = scale_to_spectral_radius(w_raw.as_ref(), opts.spectral_radius)?;
    let mut w_in = Mat::zeros(n_units, p);
    for i in 0..n_units {
        for j in 0..p {
            w_in[(i, j)] = -opts.input_scaling + 2.0 * opts.input_scaling * stream.uniform_f64();
        }
    }

    // --- states, readout, predictions -----------------------------------
    let states = esn_states(w_in.as_ref(), w.as_ref(), x, opts.leak_rate)?;
    let readout = esn_readout(states.as_ref(), x, y, opts.washout, opts.ridge_alpha)?;
    let z = readout_design(states.as_ref(), x, opts.washout);
    let fitted: Vec<f64> = (0..z.nrows())
        .map(|r| (0..z.ncols()).map(|c| z[(r, c)] * readout[c]).sum())
        .collect();
    let predicted = match x_test {
        Some(xt) => {
            let last: Vec<f64> = (0..n_units).map(|i| states[(n - 1, i)]).collect();
            let st = esn_states_from(w_in.as_ref(), w.as_ref(), xt, opts.leak_rate, &last)?;
            let zt = readout_design(st.as_ref(), xt, 0);
            Some(
                (0..zt.nrows())
                    .map(|r| (0..zt.ncols()).map(|c| zt[(r, c)] * readout[c]).sum())
                    .collect(),
            )
        }
        None => None,
    };
    Ok(EsnFit {
        fitted,
        predicted,
        readout,
        spectral_radius_achieved: achieved,
        reservoir_size: n_units,
        n_washout: opts.washout,
        n_train: n - opts.washout,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn washout_rules_name_the_fix_or_the_count() {
        match check_washout(10, 10) {
            Err(MlError::InvalidValue { what }) => assert!(what.contains("washout=10")),
            other => panic!("expected InvalidValue, got {other:?}"),
        }
        match check_washout(10, 9) {
            Err(MlError::InsufficientData { needed, got }) => {
                assert_eq!((needed, got), (11, 10));
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
        assert!(check_washout(10, 8).is_ok());
    }

    #[test]
    fn scaling_hits_the_target_on_a_diagonal_matrix() {
        let w = Mat::from_fn(3, 3, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
        let (scaled, achieved) = scale_to_spectral_radius(w.as_ref(), 0.9).unwrap();
        assert!((achieved - 0.9).abs() < 1e-12);
        assert!((scaled[(2, 2)] - 0.9).abs() < 1e-12);
    }
}
