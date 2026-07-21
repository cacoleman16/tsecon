//! Segment sums of squared residuals via recursive normal-equation updates.
//!
//! The Bai–Perron dynamic program needs `SSR(i, j)` — the OLS sum of
//! squared residuals of `y[i..=j]` on `x[i..=j]` — for every admissible
//! segment (`j - i + 1 >= h`). Recomputing each regression from scratch
//! would cost `O(T^3)`; instead, for each start `i` the cross-moments
//! `X'X`, `X'y`, `y'y` are updated by one rank-one step per added
//! observation and the `q x q` normal equations are solved by Cholesky
//! (via [`tsecon_linalg::faer`]), giving `SSR = y'y - b'X'y` in
//! `O(T^2 q^3)` total. This is the same recursive-least-squares device as
//! Bai & Perron (2003, *J. Applied Econometrics*), Section 4.1.

use tsecon_linalg::faer::linalg::solvers::Solve;
use tsecon_linalg::faer::{Mat, Side};

use crate::error::BreaksError;

/// Precomputed `SSR(i, j)` for all admissible segments of one sample.
pub(crate) struct SegTable {
    /// Sample size `T`.
    pub t: usize,
    /// Minimal segment length `h`.
    pub h: usize,
    /// `rows[i][j - (i + h - 1)] = SSR(i, j)` for `j >= i + h - 1`.
    rows: Vec<Vec<f64>>,
}

impl SegTable {
    /// Build the table. `x` is a slice of `q` columns, each of length `T`.
    ///
    /// # Errors
    ///
    /// [`BreaksError::Singular`] when some admissible segment's `X'X` is
    /// not positive definite (collinear columns on that stretch).
    pub fn build(y: &[f64], x: &[Vec<f64>], h: usize) -> Result<Self, BreaksError> {
        let t = y.len();
        let q = x.len();
        let mut rows = Vec::with_capacity(t - h + 1);
        let mut xtx = vec![0.0_f64; q * q];
        let mut xty = vec![0.0_f64; q];
        for i in 0..=(t - h) {
            xtx.iter_mut().for_each(|v| *v = 0.0);
            xty.iter_mut().for_each(|v| *v = 0.0);
            let mut yty = 0.0_f64;
            let mut row = Vec::with_capacity(t - (i + h - 1));
            for j in i..t {
                let yj = y[j];
                yty += yj * yj;
                for a in 0..q {
                    let xa = x[a][j];
                    xty[a] += xa * yj;
                    for b in 0..=a {
                        xtx[a * q + b] += xa * x[b][j];
                    }
                }
                if j + 1 >= i + h {
                    // Scalar fast path for the ubiquitous mean-shift /
                    // single-regressor model; faer Cholesky otherwise.
                    let fitted = if q == 1 {
                        if xtx[0] <= 0.0 {
                            return Err(BreaksError::Singular { start: i, end: j });
                        }
                        xty[0] * xty[0] / xtx[0]
                    } else {
                        let m = Mat::from_fn(q, q, |r, c| {
                            if r >= c {
                                xtx[r * q + c]
                            } else {
                                xtx[c * q + r]
                            }
                        });
                        let rhs = Mat::from_fn(q, 1, |r, _| xty[r]);
                        let sol = m
                            .llt(Side::Lower)
                            .map_err(|_| BreaksError::Singular { start: i, end: j })?
                            .solve(&rhs);
                        (0..q).map(|a| sol[(a, 0)] * xty[a]).sum()
                    };
                    row.push((yty - fitted).max(0.0));
                }
            }
            rows.push(row);
        }
        Ok(SegTable { t, h, rows })
    }

    /// `SSR(i, j)` for the segment of observations `i..=j` (requires
    /// `j - i + 1 >= h`).
    #[inline]
    pub fn ssr(&self, i: usize, j: usize) -> f64 {
        self.rows[i][j - (i + self.h - 1)]
    }

    /// Best single split of the segment `s..=e`: the date `tau` minimizing
    /// `SSR(s, tau) + SSR(tau + 1, e)` over `tau in [s + h - 1, e - h]`
    /// (both halves at least `h` long), with the minimized total.
    /// Returns `None` when the segment is shorter than `2h`.
    pub fn best_split(&self, s: usize, e: usize) -> Option<(usize, f64)> {
        if e + 1 < s + 2 * self.h {
            return None;
        }
        let mut best = f64::INFINITY;
        let mut arg = s + self.h - 1;
        for tau in (s + self.h - 1)..=(e - self.h) {
            let v = self.ssr(s, tau) + self.ssr(tau + 1, e);
            if v < best {
                best = v;
                arg = tau;
            }
        }
        Some((arg, best))
    }
}
