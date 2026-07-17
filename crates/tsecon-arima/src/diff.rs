//! Simple differencing and the undifferencing anchors.

use crate::error::ArimaError;

/// The result of differencing a series `d` times.
#[derive(Debug, Clone)]
pub(crate) struct Differenced {
    /// The `d`-times-differenced series, length `n - d`.
    pub(crate) series: Vec<f64>,
    /// Undifferencing anchors: `anchors[k]` is the last in-sample value of
    /// the `k`-times-differenced series, `k = 0..d` exclusive (empty when
    /// `d = 0`). These are exactly the terminal conditions needed to
    /// cumulate forecasts of the `d`-th difference back to levels.
    pub(crate) anchors: Vec<f64>,
}

/// Differences `y` a total of `d` times (`x_t = y_t - y_{t-1}` applied
/// repeatedly — the statsmodels `simple_differencing=True` convention,
/// losing one observation per difference), recording the last value of
/// each intermediate difference order as an undifferencing anchor.
///
/// # Errors
///
/// * [`ArimaError::NonFinite`] if `y` contains NaN/infinity (NaN-coded
///   missing values are not supported on the simple-differencing path);
/// * [`ArimaError::InsufficientObservations`] if `y.len() <= d` (no
///   observations would remain).
pub(crate) fn difference(y: &[f64], d: usize) -> Result<Differenced, ArimaError> {
    if y.iter().any(|v| !v.is_finite()) {
        return Err(ArimaError::NonFinite { what: "y" });
    }
    if y.len() <= d {
        return Err(ArimaError::InsufficientObservations {
            needed: d + 1,
            got: y.len(),
        });
    }
    let mut series = y.to_vec();
    let mut anchors = Vec::with_capacity(d);
    for _ in 0..d {
        // `series` is non-empty by the length check above.
        anchors.push(series[series.len() - 1]);
        series = series.windows(2).map(|w| w[1] - w[0]).collect();
    }
    Ok(Differenced { series, anchors })
}
