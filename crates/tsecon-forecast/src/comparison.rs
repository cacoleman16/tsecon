//! The [`ForecastComparison`] report: an accuracy table plus pairwise
//! Diebold-Mariano tests for a set of named forecasts of the same
//! actuals, with a teaching interpretation string in the house style of
//! `tsecon-diag`'s `DiagnosticReport`.

use core::fmt;

use crate::accuracy::{mae, mape, mase, mdae, me, mse, rmse, rmsse, smape};
use crate::cw::{cw_test, CwResult};
use crate::dm::{dm_test, DmLoss, DmResult};
use crate::error::ForecastError;

/// One row of the accuracy table: every measure for one named forecast.
///
/// `mape`/`smape` are `None` when undefined for these data (zero actuals
/// or zero denominators) — in a comparison table a per-measure hole
/// beats failing the whole report, and the standalone functions in
/// [`crate::accuracy`] still error loudly. `mase`/`rmsse` are `None`
/// when no training sample was supplied to scale by.
#[derive(Debug, Clone, PartialEq)]
pub struct AccuracyRow {
    /// The forecast's label.
    pub name: String,
    /// Mean error (bias).
    pub me: f64,
    /// Mean squared error.
    pub mse: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Mean absolute error.
    pub mae: f64,
    /// Median absolute error.
    pub mdae: f64,
    /// Mean absolute percentage error; `None` if any actual is zero.
    pub mape: Option<f64>,
    /// Symmetric MAPE (M4 definition); `None` on a zero denominator.
    pub smape: Option<f64>,
    /// Mean absolute scaled error; `None` without a training sample.
    pub mase: Option<f64>,
    /// Root mean squared scaled error; `None` without a training sample.
    pub rmsse: Option<f64>,
}

/// One pairwise Diebold-Mariano comparison (squared loss, HLN corrected).
#[derive(Debug, Clone, PartialEq)]
pub struct DmPair {
    /// Label of the first forecast (loss differential `g(e_a) - g(e_b)`).
    pub name_a: String,
    /// Label of the second forecast.
    pub name_b: String,
    /// The full DM test result.
    pub dm: DmResult,
}

/// One nested-model Clark-West comparison: a declared (small, large) pair.
#[derive(Debug, Clone, PartialEq)]
pub struct CwPair {
    /// Label of the restricted (small) model.
    pub name_small: String,
    /// Label of the unrestricted (large) model that nests the small one.
    pub name_large: String,
    /// The full Clark-West test result.
    pub cw: CwResult,
}

/// A forecast-comparison report: accuracy table + pairwise DM tests +
/// a plain-language interpretation.
///
/// Build with [`ForecastComparison::new`]; `Display` renders the table
/// and the interpretation. The interpretation implements the library's
/// "errors that teach" pillar: it names the best forecast, states each
/// DM decision, and points at the next methodological step.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastComparison {
    /// Per-forecast accuracy measures, in input order.
    pub measures: Vec<AccuracyRow>,
    /// Pairwise Diebold-Mariano tests (squared loss) for every unordered
    /// pair, in input order.
    pub dm_pairs: Vec<DmPair>,
    /// Clark-West tests for the declared nested `(small, large)` pairs; empty
    /// unless the comparison was built with [`ForecastComparison::new_nested`].
    pub cw_pairs: Vec<CwPair>,
    /// The forecast horizon used for the DM long-run variance.
    pub h: usize,
    /// The significance level the DM decisions were taken at.
    pub alpha: f64,
    /// The label of the forecast with the lowest RMSE.
    pub best_rmse: String,
    /// Plain-language interpretation of the comparison.
    pub interpretation: String,
}

impl ForecastComparison {
    /// Compare named forecasts of the same actuals.
    ///
    /// * `actual` — the realized values over the evaluation window.
    /// * `forecasts` — `(label, forecast)` pairs, each index-aligned with
    ///   `actual`.
    /// * `insample` — optional `(training sample, seasonal period)` for
    ///   the MASE/RMSSE scaling denominators (Hyndman & Koehler 2006);
    ///   pass `None` to omit the scaled columns.
    /// * `h` — the forecast horizon, used to truncate the DM long-run
    ///   variance at lag `h - 1` (use 1 for one-step forecasts).
    /// * `alpha` — significance level for the DM decisions.
    ///
    /// # Errors
    ///
    /// [`ForecastError::EmptyComparison`], [`ForecastError::DuplicateName`],
    /// [`ForecastError::InvalidAlpha`], the validation errors of the
    /// individual measures, and the DM errors — in particular
    /// [`ForecastError::DegenerateLossDifferential`] if two supplied
    /// forecasts are loss-identical (e.g. the same vector twice).
    pub fn new(
        actual: &[f64],
        forecasts: &[(&str, &[f64])],
        insample: Option<(&[f64], usize)>,
        h: usize,
        alpha: f64,
    ) -> Result<Self, ForecastError> {
        Self::build(actual, forecasts, insample, h, alpha, &[], 0)
    }

    /// Compare named forecasts and additionally run the Clark-West (2007)
    /// nested-model test for each declared `(small, large)` pair.
    ///
    /// Same as [`ForecastComparison::new`], plus:
    ///
    /// * `nested` — `(small_model_name, large_model_name)` label pairs, each
    ///   declaring that the large model **nests** the small one. The plain
    ///   Diebold-Mariano test is degenerate for nested models under recursive
    ///   schemes (Diebold 2015); Clark & West (2007) is the right tool, and
    ///   [`ForecastComparison::cw_pairs`] carries its one-sided results.
    /// * `lrv_lags` — the Bartlett long-run-variance lag truncation for the
    ///   Clark-West variance (use `h - 1` as the usual floor for `h`-step
    ///   forecasts).
    ///
    /// Both names in each pair must appear in `forecasts`.
    ///
    /// # Errors
    ///
    /// Everything [`ForecastComparison::new`] can return, plus
    /// [`ForecastError::DuplicateName`] used here to flag a nested pair whose
    /// name is not among the supplied forecasts, and the Clark-West errors.
    pub fn new_nested(
        actual: &[f64],
        forecasts: &[(&str, &[f64])],
        insample: Option<(&[f64], usize)>,
        h: usize,
        alpha: f64,
        nested: &[(&str, &str)],
        lrv_lags: usize,
    ) -> Result<Self, ForecastError> {
        Self::build(actual, forecasts, insample, h, alpha, nested, lrv_lags)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        actual: &[f64],
        forecasts: &[(&str, &[f64])],
        insample: Option<(&[f64], usize)>,
        h: usize,
        alpha: f64,
        nested: &[(&str, &str)],
        lrv_lags: usize,
    ) -> Result<Self, ForecastError> {
        if forecasts.is_empty() {
            return Err(ForecastError::EmptyComparison);
        }
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(ForecastError::InvalidAlpha { value: alpha });
        }
        for (i, (name, _)) in forecasts.iter().enumerate() {
            if forecasts[..i].iter().any(|(other, _)| other == name) {
                return Err(ForecastError::DuplicateName {
                    name: (*name).to_string(),
                });
            }
        }

        let mut measures = Vec::with_capacity(forecasts.len());
        for (name, f) in forecasts {
            let mape_v = match mape(actual, f) {
                Ok(v) => Some(v),
                Err(ForecastError::ZeroActualInMape { .. }) => None,
                Err(e) => return Err(e),
            };
            let smape_v = match smape(actual, f) {
                Ok(v) => Some(v),
                Err(ForecastError::ZeroDenominatorInSmape { .. }) => None,
                Err(e) => return Err(e),
            };
            let (mase_v, rmsse_v) = match insample {
                Some((tr, period)) => (
                    Some(mase(actual, f, tr, period)?),
                    Some(rmsse(actual, f, tr, period)?),
                ),
                None => (None, None),
            };
            measures.push(AccuracyRow {
                name: (*name).to_string(),
                me: me(actual, f)?,
                mse: mse(actual, f)?,
                rmse: rmse(actual, f)?,
                mae: mae(actual, f)?,
                mdae: mdae(actual, f)?,
                mape: mape_v,
                smape: smape_v,
                mase: mase_v,
                rmsse: rmsse_v,
            });
        }

        // Forecast errors e = actual - forecast for the DM tests.
        let errors: Vec<Vec<f64>> = forecasts
            .iter()
            .map(|(_, f)| actual.iter().zip(f.iter()).map(|(&y, &v)| y - v).collect())
            .collect();
        let mut dm_pairs = Vec::new();
        for i in 0..forecasts.len() {
            for j in (i + 1)..forecasts.len() {
                dm_pairs.push(DmPair {
                    name_a: forecasts[i].0.to_string(),
                    name_b: forecasts[j].0.to_string(),
                    dm: dm_test(&errors[i], &errors[j], h, DmLoss::Squared)?,
                });
            }
        }

        // Clark-West for declared nested (small, large) pairs. The DM test is
        // degenerate for nested models under recursive schemes (Diebold 2015),
        // so this is the scheme-appropriate substitute.
        let find = |name: &str| -> Result<usize, ForecastError> {
            forecasts
                .iter()
                .position(|(n, _)| *n == name)
                .ok_or_else(|| ForecastError::UnknownForecastName {
                    name: name.to_string(),
                })
        };
        let mut cw_pairs = Vec::with_capacity(nested.len());
        for (small, large) in nested {
            let si = find(small)?;
            let li = find(large)?;
            cw_pairs.push(CwPair {
                name_small: (*small).to_string(),
                name_large: (*large).to_string(),
                cw: cw_test(
                    &errors[si],
                    &errors[li],
                    forecasts[si].1,
                    forecasts[li].1,
                    lrv_lags,
                )?,
            });
        }

        // Best by RMSE (ties broken by input order).
        let best = measures
            .iter()
            .min_by(|a, b| a.rmse.total_cmp(&b.rmse))
            .map(|r| r.name.clone())
            .unwrap_or_default();

        let interpretation = build_interpretation(&measures, &dm_pairs, &cw_pairs, &best, alpha);

        Ok(ForecastComparison {
            measures,
            dm_pairs,
            cw_pairs,
            h,
            alpha,
            best_rmse: best,
            interpretation,
        })
    }
}

fn build_interpretation(
    measures: &[AccuracyRow],
    dm_pairs: &[DmPair],
    cw_pairs: &[CwPair],
    best: &str,
    alpha: f64,
) -> String {
    let mut s = String::new();
    let best_row = measures.iter().find(|r| r.name == best);
    if let Some(row) = best_row {
        s.push_str(&format!(
            "'{}' has the lowest RMSE ({:.4}).",
            row.name, row.rmse
        ));
        if let Some(mase_v) = row.mase {
            if mase_v < 1.0 {
                s.push_str(&format!(
                    " Its MASE of {mase_v:.4} < 1 means it beats the in-sample \
                     seasonal-naive benchmark on average."
                ));
            } else {
                s.push_str(&format!(
                    " Warning: its MASE of {mase_v:.4} >= 1 means even the best \
                     forecast here does not beat the in-sample seasonal-naive \
                     benchmark — consider the benchmark zoo before anything \
                     fancier."
                ));
            }
        }
    }
    if dm_pairs.is_empty() {
        s.push_str(
            " Only one forecast was supplied, so no pairwise Diebold-Mariano \
             tests were run; add a benchmark (naive, drift, Theta) to make \
             the accuracy numbers meaningful.",
        );
        return s;
    }
    for pair in dm_pairs {
        let better = if pair.dm.mean_loss_diff > 0.0 {
            &pair.name_b
        } else {
            &pair.name_a
        };
        if pair.dm.p_value < alpha {
            s.push_str(&format!(
                " DM '{}' vs '{}': reject equal predictive accuracy \
                 (HLN t = {:.3}, p = {:.4} < {alpha}); the squared-error \
                 loss difference favours '{}'.",
                pair.name_a, pair.name_b, pair.dm.hln_stat, pair.dm.p_value, better
            ));
        } else {
            s.push_str(&format!(
                " DM '{}' vs '{}': fail to reject equal predictive accuracy \
                 (HLN t = {:.3}, p = {:.4} >= {alpha}) — the accuracy gap is \
                 within sampling noise for this evaluation window.",
                pair.name_a, pair.name_b, pair.dm.hln_stat, pair.dm.p_value
            ));
        }
    }
    if measures.len() > 2 {
        s.push_str(
            " Note: pairwise DM tests control size per comparison only; \
             with several models use a multiple-comparison procedure \
             (model confidence set / SPA, planned for the evaluation \
             module) before declaring a winner.",
        );
    }
    if cw_pairs.is_empty() {
        s.push_str(
            " Remember the DM test compares forecasts, not models (Diebold \
             2015): for nested models under recursive schemes its distribution \
             is degenerate — use a Clark-West adjustment there.",
        );
        return s;
    }
    s.push_str(
        " The DM test is degenerate for nested models under recursive schemes \
         (Diebold 2015), so the declared nested pairs were tested with \
         Clark-West (2007) instead:",
    );
    for pair in cw_pairs {
        if pair.cw.p_value < alpha {
            s.push_str(&format!(
                " CW small='{}' vs large='{}': reject equal accuracy \
                 (z = {:.3}, one-sided p = {:.4} < {alpha}); the larger model \
                 adds genuine predictive content.",
                pair.name_small, pair.name_large, pair.cw.cw_stat, pair.cw.p_value
            ));
        } else {
            s.push_str(&format!(
                " CW small='{}' vs large='{}': fail to reject (z = {:.3}, \
                 one-sided p = {:.4} >= {alpha}) — the extra regressors in the \
                 larger model do not improve out-of-sample accuracy.",
                pair.name_small, pair.name_large, pair.cw.cw_stat, pair.cw.p_value
            ));
        }
    }
    s
}

impl fmt::Display for ForecastComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn opt(v: Option<f64>) -> String {
            v.map_or_else(|| "      --".to_string(), |x| format!("{x:8.4}"))
        }
        writeln!(
            f,
            "{:<16} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8}",
            "forecast", "ME", "RMSE", "MAE", "MdAE", "MAPE", "sMAPE", "MASE", "RMSSE"
        )?;
        for r in &self.measures {
            writeln!(
                f,
                "{:<16} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {} {} {} {}",
                r.name,
                r.me,
                r.rmse,
                r.mae,
                r.mdae,
                opt(r.mape),
                opt(r.smape),
                opt(r.mase),
                opt(r.rmsse)
            )?;
        }
        for p in &self.dm_pairs {
            writeln!(
                f,
                "DM({h}) {a} vs {b}: HLN t = {t:.4}, p = {pv:.4} [{decision}]",
                h = p.dm.h,
                a = p.name_a,
                b = p.name_b,
                t = p.dm.hln_stat,
                pv = p.dm.p_value,
                decision = if p.dm.p_value < self.alpha {
                    "reject equal accuracy"
                } else {
                    "fail to reject"
                }
            )?;
        }
        for p in &self.cw_pairs {
            writeln!(
                f,
                "CW small={s} vs large={l}: z = {z:.4}, one-sided p = {pv:.4} [{decision}]",
                s = p.name_small,
                l = p.name_large,
                z = p.cw.cw_stat,
                pv = p.cw.p_value,
                decision = if p.cw.p_value < self.alpha {
                    "large model adds content"
                } else {
                    "fail to reject"
                }
            )?;
        }
        write!(f, "{}", self.interpretation)
    }
}
