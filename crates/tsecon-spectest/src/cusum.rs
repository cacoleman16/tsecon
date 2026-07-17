//! CUSUM parameter-stability test (Brown, Durbin & Evans 1975) on recursive
//! residuals.
//!
//! For a design with `k` regressors and `n` observations, the recursive
//! residuals are, for `r = k, k+1, ..., n-1` (0-indexed; the `r`-th uses the
//! first `r` observations to predict observation `r`):
//!
//! ```text
//! b_r = OLS on the first r observations
//! f_r = 1 + x_r' (X[:r]' X[:r])^{-1} x_r
//! w_r = (y_r - x_r' b_r) / sqrt(f_r).
//! ```
//!
//! Under the null of stable coefficients the `w_r` are iid `N(0, sigma^2)`,
//! and the algebraic identity `sum_r w_r^2 = SSR_full` gives
//! `sigma = sqrt(SSR_full / (n - k))`. The standardized CUSUM path is the
//! running sum
//!
//! ```text
//! W_t = (1/sigma) * sum_{r=k}^{t} w_r,     t = k, ..., n-1   (length n - k).
//! ```
//!
//! The 5% significance boundary is the pair of straight lines through
//! `+/- a*sqrt(n-k)` at the first CUSUM point and `+/- 3a*sqrt(n-k)` at the
//! last, with the documented Brown-Durbin-Evans constant `a = 0.948` for the
//! 5% level:
//!
//! ```text
//! bound_upper[i] = a * ( sqrt(n-k) + 2*(i+1)/sqrt(n-k) ),   i = 0..n-k-1.
//! ```
//!
//! The recursion is evaluated with an incremental Gram matrix and a dense
//! Cholesky inverse from `faer` (re-exported through `tsecon-linalg`, the
//! workspace's single dense backend); `sigma` reuses [`tsecon_hac::ols`].

use tsecon_hac::ols;
use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, Side};

use crate::common::{ssr, validate};
use crate::error::SpecTestError;

/// The Brown-Durbin-Evans (1975) boundary constant for the 5% significance
/// level: the CUSUM crosses `+/- a*sqrt(n-k)` at the start and
/// `+/- 3a*sqrt(n-k)` at the end.
pub const CUSUM_A_5PCT: f64 = 0.948;

/// Outcome of the Brown-Durbin-Evans CUSUM parameter-stability test. All
/// vectors have length `n - k` and are aligned to `t = k, ..., n - 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct CusumTest {
    /// Recursive residuals `w_r`, `r = k, ..., n - 1`.
    pub recursive_residuals: Vec<f64>,
    /// Standardized CUSUM path `W_t`.
    pub path: Vec<f64>,
    /// Upper 5% significance boundary at each CUSUM point.
    pub bound_upper: Vec<f64>,
    /// Lower 5% significance boundary (`= -bound_upper`).
    pub bound_lower: Vec<f64>,
    /// Standard-deviation estimate `sigma = sqrt(SSR_full / (n - k))`.
    pub sigma: f64,
    /// The boundary constant used (`CUSUM_A_5PCT`).
    pub a: f64,
}

impl CusumTest {
    /// `true` if the CUSUM path breaches its 5% boundary at any point — the
    /// test's rejection of coefficient stability at the 5% level.
    pub fn rejects_5pct(&self) -> bool {
        self.path
            .iter()
            .zip(self.bound_upper.iter())
            .any(|(&w, &b)| w.abs() > b)
    }
}

/// Invert a symmetric positive-definite `k x k` matrix (row-major) via its
/// `faer` Cholesky factor; a rejected factorization means the recursive-window
/// design is rank deficient.
fn inv_spd(gram: &[f64], k: usize) -> Result<Vec<f64>, SpecTestError> {
    let m = Mat::from_fn(k, k, |i, j| gram[i * k + j]);
    let inv = m
        .llt(Side::Lower)
        .map_err(|_| SpecTestError::SingularDesign {
            what: "recursive-residual window",
        })?
        .inverse();
    let mut out = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            out[i * k + j] = inv[(i, j)];
        }
    }
    Ok(out)
}

/// Brown-Durbin-Evans CUSUM test on the recursive residuals of the regression
/// of `y` on `x_cols` (constant included explicitly).
///
/// # Errors
///
/// Returns [`SpecTestError::DegreesOfFreedom`] if `n <= k` (no recursive
/// residual exists); the usual empty/dimension/finite validation errors; and
/// [`SpecTestError::SingularDesign`] if the full design or any expanding
/// window `X[:r]` is collinear.
pub fn cusum_test(y: &[f64], x_cols: &[Vec<f64>]) -> Result<CusumTest, SpecTestError> {
    let (n, k) = validate(y, x_cols)?;
    if n <= k {
        return Err(SpecTestError::DegreesOfFreedom {
            what: "CUSUM regression",
            n,
            k,
        });
    }

    // sigma from the full-sample OLS (identically sqrt(sum w^2 / (n-k))).
    let full = ols(y, x_cols)?;
    let sigma = (ssr(&full) / (n - k) as f64).sqrt();

    // Row access into the column-major design.
    let row = |t: usize| -> Vec<f64> { x_cols.iter().map(|col| col[t]).collect() };

    // Incremental Gram G = X[:r]'X[:r] and rhs c = X[:r]'y[:r], seeded with the
    // first k observations.
    let mut gram = vec![0.0_f64; k * k];
    let mut rhs = vec![0.0_f64; k];
    let add_obs = |gram: &mut [f64], rhs: &mut [f64], xr: &[f64], yr: f64| {
        for i in 0..k {
            rhs[i] += xr[i] * yr;
            for j in 0..k {
                gram[i * k + j] += xr[i] * xr[j];
            }
        }
    };
    for (t, &yt) in y.iter().take(k).enumerate() {
        add_obs(&mut gram, &mut rhs, &row(t), yt);
    }

    let mut w = Vec::with_capacity(n - k);
    for (r, &yr) in y.iter().enumerate().skip(k) {
        let p = inv_spd(&gram, k)?; // (X[:r]'X[:r])^{-1}
        let xr = row(r);
        // b_r = P c ; pred = x_r' b_r ; f_r = 1 + x_r' P x_r.
        let mut pred = 0.0;
        let mut f = 1.0;
        for i in 0..k {
            let mut pc = 0.0; // (P c)_i
            let mut px = 0.0; // (P x_r)_i
            for j in 0..k {
                pc += p[i * k + j] * rhs[j];
                px += p[i * k + j] * xr[j];
            }
            pred += xr[i] * pc;
            f += xr[i] * px;
        }
        w.push((yr - pred) / f.sqrt());
        add_obs(&mut gram, &mut rhs, &xr, yr);
    }

    // Standardized CUSUM path and the 5% Brown-Durbin-Evans boundary.
    let nk = (n - k) as f64;
    let sq = nk.sqrt();
    let a = CUSUM_A_5PCT;
    let mut path = Vec::with_capacity(w.len());
    let mut bound_upper = Vec::with_capacity(w.len());
    let mut bound_lower = Vec::with_capacity(w.len());
    let mut running = 0.0;
    for (i, &wi) in w.iter().enumerate() {
        running += wi;
        path.push(running / sigma);
        let bound = a * (sq + 2.0 * (i as f64 + 1.0) / sq);
        bound_upper.push(bound);
        bound_lower.push(-bound);
    }

    Ok(CusumTest {
        recursive_residuals: w,
        path,
        bound_upper,
        bound_lower,
        sigma,
        a,
    })
}
