//! The binary-choice link: the standard normal CDF (probit) or the logistic
//! CDF (logit), plus the per-observation likelihood, score, and information
//! quantities every static estimator needs.
//!
//! All normal-distribution values come from [`tsecon_stats::StdNormal`]; this
//! crate never re-implements the error function or a logistic CDF of its own.

use tsecon_stats::{ContinuousDist, StdNormal};

/// The smallest / largest fitted probability used in log terms, so that
/// `ln(p)` and `ln(1 - p)` stay finite even when the index runs deep into a
/// tail. `1e-300` keeps `ln` at about `-690`, far below any realistic index.
const P_CLAMP: f64 = 1e-300;

/// The link function mapping a linear index to a recession probability.
///
/// * [`Link::Probit`] uses the standard normal CDF `Phi`.
/// * [`Link::Logit`] uses the logistic CDF `Lambda(z) = 1 / (1 + e^{-z})`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    /// Probit: `P(y = 1) = Phi(index)`.
    Probit,
    /// Logit: `P(y = 1) = Lambda(index)`.
    Logit,
}

impl Link {
    /// The fitted probability `P(y = 1 | index) = F(index)`.
    #[inline]
    pub fn prob(self, index: f64) -> f64 {
        match self {
            Link::Probit => StdNormal.cdf(index),
            Link::Logit => logistic_cdf(index),
        }
    }

    /// Log-likelihood contribution of one observation,
    /// `y ln F(index) + (1 - y) ln(1 - F(index))`.
    ///
    /// For the probit the two tails use `Phi` and its survival function
    /// directly (rather than `1 - Phi`) so the answer keeps full precision in
    /// the right tail.
    #[inline]
    pub fn loglik_term(self, y: f64, index: f64) -> f64 {
        match self {
            Link::Probit => {
                if y > 0.5 {
                    StdNormal.cdf(index).max(P_CLAMP).ln()
                } else {
                    StdNormal.sf(index).max(P_CLAMP).ln()
                }
            }
            Link::Logit => {
                let p = logistic_cdf(index);
                if y > 0.5 {
                    p.max(P_CLAMP).ln()
                } else {
                    (1.0 - p).max(P_CLAMP).ln()
                }
            }
        }
    }

    /// The per-observation score factor `g` such that the log-likelihood
    /// gradient is `dLL/dbeta_j = sum_t g_t x_{t,j}`.
    ///
    /// * Logit: `g = y - p`.
    /// * Probit: `g = q phi(index) / Phi(q index)` with `q = 2y - 1`, the
    ///   inverse Mills-ratio form used by statsmodels; the `q` factoring keeps
    ///   the denominator away from zero on whichever tail the observation sits.
    #[inline]
    pub fn score_factor(self, y: f64, index: f64) -> f64 {
        match self {
            Link::Probit => {
                let q = 2.0 * y - 1.0;
                let phi = StdNormal.pdf(index); // symmetric: phi(q*index) = phi(index)
                let denom = StdNormal.cdf(q * index).max(P_CLAMP);
                q * phi / denom
            }
            Link::Logit => y - logistic_cdf(index),
        }
    }

    /// The per-observation information weight `w >= 0` in the negative Hessian
    /// `I = -H = sum_t w_t x_t x_t'` (observed information at any parameter).
    ///
    /// * Logit: `w = p (1 - p)`.
    /// * Probit: `w = L (L + index)` with `L = q phi / Phi(q index)`, the
    ///   analytic-Hessian weight of statsmodels' `Probit.hessian`.
    #[inline]
    pub fn info_weight(self, y: f64, index: f64) -> f64 {
        match self {
            Link::Probit => {
                let q = 2.0 * y - 1.0;
                let phi = StdNormal.pdf(index);
                let denom = StdNormal.cdf(q * index).max(P_CLAMP);
                let l = q * phi / denom;
                l * (l + index)
            }
            Link::Logit => {
                let p = logistic_cdf(index);
                p * (1.0 - p)
            }
        }
    }
}

/// Logistic CDF `Lambda(z) = 1 / (1 + e^{-z})`, evaluated in the numerically
/// stable branchwise form so neither tail overflows `exp`.
#[inline]
fn logistic_cdf(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logistic_matches_reference_points() {
        assert!((logistic_cdf(0.0) - 0.5).abs() < 1e-15);
        // Lambda(2) = 1/(1+e^-2)
        let want = 1.0 / (1.0 + (-2.0_f64).exp());
        assert!((logistic_cdf(2.0) - want).abs() < 1e-15);
        // Symmetry: Lambda(-z) = 1 - Lambda(z)
        assert!((logistic_cdf(-3.5) - (1.0 - logistic_cdf(3.5))).abs() < 1e-15);
    }

    #[test]
    fn logistic_tails_do_not_overflow() {
        assert_eq!(logistic_cdf(1000.0), 1.0);
        assert_eq!(logistic_cdf(-1000.0), 0.0);
        assert!(logistic_cdf(-1000.0).is_finite());
    }

    #[test]
    fn logit_score_and_weight_reduce_to_the_clean_forms() {
        let idx = 0.3;
        let p = logistic_cdf(idx);
        assert!((Link::Logit.score_factor(1.0, idx) - (1.0 - p)).abs() < 1e-15);
        assert!((Link::Logit.score_factor(0.0, idx) - (-p)).abs() < 1e-15);
        assert!((Link::Logit.info_weight(1.0, idx) - p * (1.0 - p)).abs() < 1e-15);
    }

    #[test]
    fn probit_score_sign_is_correct() {
        // A y=1 observation with positive index pushes the score positive.
        assert!(Link::Probit.score_factor(1.0, 0.5) > 0.0);
        // A y=0 observation with positive index pushes the score negative.
        assert!(Link::Probit.score_factor(0.0, 0.5) < 0.0);
    }
}
