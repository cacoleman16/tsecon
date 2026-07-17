//! Diebold-Yilmaz connectedness measures and the printable spillover
//! table.

use core::fmt;

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_var::VarResults;

use crate::error::ConnectError;
use crate::gfevd::generalized_fevd;

/// A Diebold-Yilmaz (2012, 2014) connectedness table built from a
/// row-normalized generalized FEVD.
///
/// Let `theta` be the `k x k` normalized generalized variance
/// decomposition ([`generalized_fevd`]); `theta[(i, j)]` is the share of
/// variable `i`'s `H`-step forecast-error variance attributable to shocks
/// in variable `j`, and each row sums to one. Writing the off-diagonal
/// (cross-variable) mass as the source of "spillovers", the measures are
/// (Diebold and Yilmaz 2012, *Int. J. Forecasting* 28, eqs. 5-9; 2014,
/// *J. Econometrics* 182, sec. 2):
///
/// ```text
/// total          C(H)   = 100 * (sum_{i != j} theta_ij) / k
/// from others    C_i<-. = 100 * (sum_{j != i} theta_ij) / k      (row i)
/// to   others    C_.<-i = 100 * (sum_{j != i} theta_ji) / k      (col i)
/// net            C_i    = C_.<-i - C_i<-.
/// pairwise net   C_ij   = 100 * (theta_ji - theta_ij) / k
/// ```
///
/// so the total index is both the average `from` and the average `to`
/// (they share the same off-diagonal mass), `sum_i net_i = 0`, and the
/// row sums of [`ConnectednessTable::pairwise_net`] reproduce
/// [`ConnectednessTable::net`].
#[derive(Debug, Clone)]
pub struct ConnectednessTable {
    /// Number of variables `k`.
    pub k: usize,
    /// Row-normalized generalized FEVD `theta` (`k x k`, rows sum to 1);
    /// `gfevd[(i, j)]` is the share of variable `i`'s variance from
    /// variable `j`.
    pub gfevd: Mat<f64>,
    /// Total connectedness index (spillover index), in percent, in
    /// `[0, 100]`.
    pub total: f64,
    /// Directional "to others" per variable, in percent: column
    /// off-diagonal sums scaled by `100 / k`. `to_others[i]` is what
    /// variable `i` transmits to the rest of the system.
    pub to_others: Vec<f64>,
    /// Directional "from others" per variable, in percent: row
    /// off-diagonal sums scaled by `100 / k`. `from_others[i]` is what
    /// variable `i` receives from the rest of the system.
    pub from_others: Vec<f64>,
    /// Net directional connectedness per variable, `to_others[i] -
    /// from_others[i]`; positive means a net transmitter. Sums to zero.
    pub net: Vec<f64>,
    /// Net pairwise directional connectedness (`k x k`, antisymmetric,
    /// zero diagonal): `pairwise_net[(i, j)] = 100 * (theta_ji -
    /// theta_ij) / k` is the net flow *from* `i` *to* `j`. Row `i` sums
    /// to `net[i]`.
    pub pairwise_net: Mat<f64>,
    /// Optional variable labels used by [`fmt::Display`]; when absent the
    /// table falls back to `x0, x1, ...`.
    pub labels: Option<Vec<String>>,
}

impl ConnectednessTable {
    /// Builds the connectedness measures from a row-normalized
    /// generalized FEVD `theta` (`k x k`, each row summing to one).
    ///
    /// The row-sum-to-one property is assumed, not re-imposed; pass the
    /// output of [`generalized_fevd`].
    ///
    /// # Errors
    ///
    /// [`ConnectError::Dimension`] if `theta` is not square, and
    /// [`ConnectError::InvalidArgument`] if it is empty.
    pub fn from_gfevd(theta: MatRef<'_, f64>) -> Result<Self, ConnectError> {
        let k = theta.nrows();
        if k == 0 {
            return Err(ConnectError::InvalidArgument {
                what: "gfevd must be non-empty",
            });
        }
        if theta.ncols() != k {
            return Err(ConnectError::Dimension {
                what: "gfevd must be square",
                expected: k,
                got: theta.ncols(),
            });
        }
        let scale = 100.0 / k as f64;

        let mut from_others = vec![0.0_f64; k];
        let mut to_others = vec![0.0_f64; k];
        for i in 0..k {
            for j in 0..k {
                if i == j {
                    continue;
                }
                from_others[i] += theta[(i, j)];
                to_others[j] += theta[(i, j)];
            }
        }
        for v in from_others.iter_mut().chain(to_others.iter_mut()) {
            *v *= scale;
        }

        let total: f64 = from_others.iter().sum::<f64>();
        let net: Vec<f64> = (0..k).map(|i| to_others[i] - from_others[i]).collect();

        let pairwise_net = Mat::from_fn(k, k, |i, j| {
            if i == j {
                0.0
            } else {
                scale * (theta[(j, i)] - theta[(i, j)])
            }
        });

        Ok(Self {
            k,
            gfevd: theta.to_owned(),
            total,
            to_others,
            from_others,
            net,
            pairwise_net,
            labels: None,
        })
    }

    /// Builds the table from a fitted reduced-form VAR: takes the
    /// MA(inf) weights `Psi_0, ..., Psi_horizon`
    /// ([`VarResults::ma_rep`]) and the df-adjusted residual covariance
    /// `sigma_u`, forms the generalized FEVD, and derives the measures.
    ///
    /// `horizon` is the forecast horizon `H`; the sum runs over
    /// `h = 0, ..., H` (`H + 1` MA matrices), matching the fixture
    /// convention.
    ///
    /// # Errors
    ///
    /// Propagates [`VarResults::ma_rep`] and [`generalized_fevd`]
    /// failures.
    pub fn from_var(res: &VarResults, horizon: usize) -> Result<Self, ConnectError> {
        let psi = res.ma_rep(horizon)?;
        let theta = generalized_fevd(&psi, res.sigma_u.as_ref())?;
        Self::from_gfevd(theta.as_ref())
    }

    /// Attaches display labels (one per variable); an incorrectly sized
    /// list is rejected.
    ///
    /// # Errors
    ///
    /// [`ConnectError::Dimension`] if `labels.len() != k`.
    pub fn with_labels(mut self, labels: Vec<String>) -> Result<Self, ConnectError> {
        if labels.len() != self.k {
            return Err(ConnectError::Dimension {
                what: "labels length must equal the number of variables",
                expected: self.k,
                got: labels.len(),
            });
        }
        self.labels = Some(labels);
        Ok(self)
    }

    fn label(&self, i: usize) -> String {
        match &self.labels {
            Some(l) => l[i].clone(),
            None => format!("x{i}"),
        }
    }
}

impl fmt::Display for ConnectednessTable {
    /// The standard Diebold-Yilmaz spillover table: the normalized
    /// generalized FEVD in percent, a right-margin "FROM others" column,
    /// a bottom "TO others" row, and the total connectedness index in the
    /// lower-right corner.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let k = self.k;
        // Column width from the widest label and a fixed numeric field.
        let labels: Vec<String> = (0..k).map(|i| self.label(i)).collect();
        let w = labels
            .iter()
            .map(String::len)
            .chain(std::iter::once("FROM".len()))
            .max()
            .unwrap_or(4)
            .max(7);

        // Header row.
        write!(f, "{:>w$}", "", w = w)?;
        for lbl in &labels {
            write!(f, " {lbl:>w$}")?;
        }
        writeln!(f, " {:>w$}", "FROM", w = w)?;

        // Body: gfevd in percent, then the row "from others".
        for (i, lbl) in labels.iter().enumerate() {
            write!(f, "{lbl:>w$}")?;
            for j in 0..k {
                write!(f, " {:>w$.2}", 100.0 * self.gfevd[(i, j)], w = w)?;
            }
            writeln!(f, " {:>w$.2}", self.from_others[i], w = w)?;
        }

        // Footer: "to others" per column, then the total in the corner.
        write!(f, "{:>w$}", "TO", w = w)?;
        for j in 0..k {
            write!(f, " {:>w$.2}", self.to_others[j], w = w)?;
        }
        writeln!(f, " {:>w$.2}", self.total, w = w)?;
        Ok(())
    }
}
