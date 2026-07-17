//! Rolling-window total connectedness.

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_var::VarSpec;

use crate::error::ConnectError;
use crate::table::ConnectednessTable;

/// Total connectedness index over a rolling window (Diebold and Yilmaz
/// 2012, sec. 3.2 "dynamic" / time-varying connectedness).
///
/// The VAR `spec` is re-estimated on each contiguous block of `window`
/// observations of `endog` (an `n x k` matrix, observations in rows,
/// oldest first); each fit yields a [`ConnectednessTable`] whose
/// [`ConnectednessTable::total`] is collected. The returned vector has
/// `n - window + 1` entries, one per window end, in chronological order;
/// entry `w` is estimated on rows `w ..= w + window - 1`.
///
/// # Errors
///
/// * [`ConnectError::InvalidArgument`] if `window == 0` or `window > n`;
/// * anything [`VarSpec::fit`] or [`ConnectednessTable::from_var`] can
///   return (e.g. a window too short for the specification, or a
///   window whose residual covariance is singular).
pub fn rolling_total_connectedness(
    endog: MatRef<'_, f64>,
    window: usize,
    spec: VarSpec,
    horizon: usize,
) -> Result<Vec<f64>, ConnectError> {
    let n = endog.nrows();
    let k = endog.ncols();
    if window == 0 {
        return Err(ConnectError::InvalidArgument {
            what: "rolling window must be positive",
        });
    }
    if window > n {
        return Err(ConnectError::InvalidArgument {
            what: "rolling window exceeds the number of observations",
        });
    }
    let n_windows = n - window + 1;
    let mut out = Vec::with_capacity(n_windows);
    for start in 0..n_windows {
        let block = Mat::from_fn(window, k, |i, j| endog[(start + i, j)]);
        let res = spec.fit(block.as_ref())?;
        let table = ConnectednessTable::from_var(&res, horizon)?;
        out.push(table.total);
    }
    Ok(out)
}
