//! Optimal-lambda Nelson-Siegel: estimate the decay `lambda` by nonlinear
//! least squares, profiling out the linear factors.

use crate::error::TermStructureError;
use crate::fit::{fit_nelson_siegel, NsFit};
use tsecon_optim::{minimize, FnObjective, Method};

/// Fit a single yield curve by Nelson-Siegel with the decay `lambda`
/// **estimated** by nonlinear least squares, rather than fixed.
///
/// The curve is linear in the three factors given `lambda`, so for each trial
/// `lambda` the factors are *profiled out* by the linear OLS
/// [`fit_nelson_siegel`] and the objective is the resulting sum of squared
/// curve-fit residuals:
///
/// ```text
/// lambda_hat = argmin_lambda  min_beta  sum_i ( y_i - x_i(lambda)' beta )^2.
/// ```
///
/// The one-dimensional concentrated objective is minimized with the adaptive
/// Nelder-Mead simplex ([`tsecon_optim::minimize`]) started from `lambda0`.
/// Trial values with `lambda <= 0` (or that make the design singular) return
/// `+inf`, which the optimizer treats as infeasible, so the search stays on
/// the admissible `lambda > 0` half-line.
///
/// ## Fixed vs estimated lambda
///
/// Diebold & Li (2006) **fix** `lambda` (`0.0609` monthly) so that the factors
/// come from a single linear regression and are comparable across dates — the
/// convention their dynamic model is built on ([`fit_nelson_siegel`]). Much of
/// the curve-fitting literature (Nelson-Siegel 1987 themselves, and the
/// central-bank fitting practice around Svensson 1994) instead **estimates**
/// `lambda` per curve for the tightest cross-sectional fit, at the cost of a
/// nonlinear, occasionally multimodal, objective. This function is that second
/// convention.
///
/// # Errors
///
/// The validation errors of [`fit_nelson_siegel`] (evaluated at `lambda0` and
/// again at the optimum), plus
/// [`TermStructureError::InvalidLambda`] if `lambda0` is non-positive/
/// non-finite and [`TermStructureError::OptimizationFailed`] if the optimizer
/// returns a non-finite `lambda` or cannot fit a curve there.
///
/// # Example
///
/// ```
/// use tsecon_termstructure::fit_nelson_siegel_optimal_lambda;
/// let maturities = [3.0, 12.0, 36.0, 60.0, 84.0, 120.0];
/// let yields = [4.10, 3.98, 4.10, 4.25, 4.31, 4.43];
/// let fit = fit_nelson_siegel_optimal_lambda(&maturities, &yields, 0.0609).unwrap();
/// // The estimated lambda fits at least as well as any fixed one.
/// assert!(fit.rsquared > 0.9);
/// assert!(fit.lambda > 0.0);
/// ```
pub fn fit_nelson_siegel_optimal_lambda(
    maturities: &[f64],
    yields: &[f64],
    lambda0: f64,
) -> Result<NsFit, TermStructureError> {
    // Validate inputs (and reject a bad start) by fitting once at lambda0.
    // This surfaces empty/short/mismatched/non-finite errors up front.
    let _ = fit_nelson_siegel(maturities, yields, lambda0)?;

    // Concentrated SSR as a function of lambda; infeasible -> +inf.
    let mut objective = FnObjective::new(|x: &[f64]| {
        let lambda = x[0];
        match fit_nelson_siegel(maturities, yields, lambda) {
            Ok(fit) => fit.residuals.iter().map(|&r| r * r).sum(),
            Err(_) => f64::INFINITY,
        }
    });

    let result = minimize(&mut objective, &[lambda0], &Method::nelder_mead()).map_err(|_| {
        TermStructureError::OptimizationFailed {
            reason: "the Nelder-Mead search over lambda returned an error",
        }
    })?;

    let lambda_hat = result.x[0];
    if !lambda_hat.is_finite() || lambda_hat <= 0.0 {
        return Err(TermStructureError::OptimizationFailed {
            reason: "the search left the admissible lambda > 0 region",
        });
    }

    // Refit at the optimum to return the factors, residuals, and R^2 there.
    fit_nelson_siegel(maturities, yields, lambda_hat)
}
