//! Simple differencing (regular and seasonal) and the undifferencing
//! anchors.

use crate::error::ArimaError;

/// The result of differencing a series `D` times seasonally and then
/// `d` times regularly.
#[derive(Debug, Clone)]
pub(crate) struct Differenced {
    /// The fully differenced series, length `n - d - D*s`.
    pub(crate) series: Vec<f64>,
    /// Regular undifferencing anchors: `anchors[k]` is the last in-sample
    /// value of the `k`-times-regularly-differenced series (differencing
    /// the already seasonally-differenced data), `k = 0..d` exclusive
    /// (empty when `d = 0`). These are exactly the terminal conditions
    /// needed to cumulate forecasts of the `d`-th difference back to the
    /// seasonally-differenced scale.
    pub(crate) anchors: Vec<f64>,
    /// Seasonal undifferencing anchors: `seasonal_anchors[k]` holds the
    /// last `s` in-sample values of the `k`-times-seasonally-differenced
    /// series (`k = 0` is the raw series), in chronological order (the
    /// most recent value last), `k = 0..D` exclusive (empty when
    /// `D = 0`). Together with the regular anchors these are the terminal
    /// conditions needed to cumulate forecasts all the way back to
    /// levels.
    pub(crate) seasonal_anchors: Vec<Vec<f64>>,
}

/// Differences `y` seasonally `seasonal_d` times at period `s` and then
/// regularly `d` times (`x_t = y_t - y_{t-1}` applied repeatedly) — the
/// statsmodels `simple_differencing=True` convention with the
/// `statsmodels.tsa.statespace.tools.diff` operation order (seasonal
/// first), losing `s` observations per seasonal difference and one per
/// regular difference — recording the terminal values of each
/// intermediate stage as undifferencing anchors.
///
/// # Errors
///
/// * [`ArimaError::NonFinite`] if `y` contains NaN/infinity (NaN-coded
///   missing values are not supported on the simple-differencing path);
/// * [`ArimaError::InsufficientObservations`] if
///   `y.len() <= d + seasonal_d * s` (no observations would remain).
pub(crate) fn difference(
    y: &[f64],
    d: usize,
    seasonal_d: usize,
    s: usize,
) -> Result<Differenced, ArimaError> {
    if let Some(index) = y.iter().position(|v| !v.is_finite()) {
        return Err(ArimaError::NonFinite {
            what: "the series y",
            at: Some(index),
        });
    }
    let lost = d + seasonal_d * s;
    if y.len() <= lost {
        return Err(ArimaError::InsufficientObservations {
            needed: lost + 1,
            got: y.len(),
            nobs: y.len(),
            what: "differencing (each regular difference drops one observation and \
                   each seasonal difference drops a full period)",
        });
    }
    let mut series = y.to_vec();
    let mut seasonal_anchors = Vec::with_capacity(seasonal_d);
    for _ in 0..seasonal_d {
        // `series.len() > s` holds: the length check above guarantees at
        // least `s + 1` observations remain before each seasonal stage.
        seasonal_anchors.push(series[series.len() - s..].to_vec());
        series = series.windows(s + 1).map(|w| w[s] - w[0]).collect();
    }
    let mut anchors = Vec::with_capacity(d);
    for _ in 0..d {
        anchors.push(series[series.len() - 1]);
        series = series.windows(2).map(|w| w[1] - w[0]).collect();
    }
    Ok(Differenced {
        series,
        anchors,
        seasonal_anchors,
    })
}
