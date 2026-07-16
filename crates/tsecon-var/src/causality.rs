//! Granger-causality F tests on the estimated VAR coefficients.

use tsecon_linalg::faer::linalg::solvers::Solve;
use tsecon_linalg::faer::{Mat, Side};
use tsecon_stats::special::beta_inc;

use crate::error::VarError;
use crate::results::VarResults;

/// Result of a Granger-causality F test, produced by
/// [`VarResults::test_causality`].
#[derive(Debug, Clone, PartialEq)]
pub struct CausalityTest {
    /// F statistic (Wald statistic divided by the number of
    /// restrictions).
    pub statistic: f64,
    /// Upper-tail p-value from the `F(df_num, df_den)` distribution.
    pub pvalue: f64,
    /// Numerator degrees of freedom: number of zero restrictions,
    /// `p * |causing| * |caused|`.
    pub df_num: usize,
    /// Denominator degrees of freedom, `k * (T - m)` (statsmodels
    /// convention: pooled residual degrees of freedom across the `k`
    /// equations).
    pub df_den: usize,
}

impl VarResults {
    /// F-type Wald test that the variables in `causing` do not
    /// Granger-cause the variables in `caused` (both are sets of
    /// zero-based column indices into the endogenous data).
    ///
    /// The null restricts to zero every coefficient of every lag of a
    /// causing variable in every caused equation — `N = p * |causing| *
    /// |caused|` restrictions. With `beta` the vectorized coefficients
    /// and `Cov(beta) = kron((Z'Z)^{-1}, sigma_u)` (Lütkepohl 2005, eq.
    /// 3.2.21), the Wald statistic is
    ///
    /// ```text
    /// W = (C beta)' [C Cov(beta) C']^{-1} (C beta)
    /// ```
    ///
    /// (Lütkepohl 2005, eq. 3.6.5) and the reported statistic is
    /// `F = W / N`, referred to the `F(N, k (T - m))` distribution —
    /// exactly statsmodels `test_causality(..., kind="f")`.
    ///
    /// # Errors
    ///
    /// * [`VarError::InvalidArgument`] if either index set is empty,
    ///   contains duplicates, or `lags == 0`;
    /// * [`VarError::Dimension`] if an index is out of range;
    /// * [`VarError::NotPositiveDefinite`] if the restriction covariance
    ///   is singular;
    /// * [`VarError::Stats`] if the F tail probability fails.
    pub fn test_causality(
        &self,
        caused: &[usize],
        causing: &[usize],
    ) -> Result<CausalityTest, VarError> {
        let k = self.neqs;
        let p = self.spec.lags;
        if p == 0 {
            return Err(VarError::InvalidArgument {
                what: "Granger causality needs at least one lag",
            });
        }
        for set in [caused, causing] {
            if set.is_empty() {
                return Err(VarError::InvalidArgument {
                    what: "caused/causing index sets must be non-empty",
                });
            }
            for (i, &v) in set.iter().enumerate() {
                if v >= k {
                    return Err(VarError::Dimension {
                        what: "variable index out of range",
                        expected: k,
                        got: v,
                    });
                }
                if set[..i].contains(&v) {
                    return Err(VarError::InvalidArgument {
                        what: "caused/causing index sets must not contain duplicates",
                    });
                }
            }
        }

        let n_trend = self.spec.trend.n_terms();
        // Restrictions enumerated as statsmodels does: lag-major, then
        // causing, then caused (the order is irrelevant to the
        // quadratic form).
        let mut regs: Vec<usize> = Vec::with_capacity(p * causing.len() * caused.len());
        let mut eqs: Vec<usize> = Vec::with_capacity(regs.capacity());
        let mut b: Vec<f64> = Vec::with_capacity(regs.capacity());
        for lag in 0..p {
            for &ing in causing {
                for &ed in caused {
                    let reg = n_trend + lag * k + ing;
                    regs.push(reg);
                    eqs.push(ed);
                    b.push(self.params[(reg, ed)]);
                }
            }
        }
        let nr = b.len();

        // C Cov(beta) C' picks entries of kron((Z'Z)^{-1}, sigma_u):
        // cov(beta_{reg_r, eq_r}, beta_{reg_s, eq_s})
        //   = (Z'Z)^{-1}[reg_r, reg_s] * sigma_u[eq_r, eq_s].
        let middle = Mat::from_fn(nr, nr, |r, s| {
            self.zz_inv[(regs[r], regs[s])] * self.sigma_u[(eqs[r], eqs[s])]
        });
        let rhs = Mat::from_fn(nr, 1, |i, _| b[i]);
        let x = middle
            .llt(Side::Lower)
            .map_err(|_| VarError::NotPositiveDefinite {
                what: "restriction covariance C Cov(beta) C'",
            })?
            .solve(&rhs);
        let wald: f64 = (0..nr).map(|i| b[i] * x[(i, 0)]).sum();

        let statistic = wald / nr as f64;
        let df_den = k * self.df_resid;
        let pvalue = f_sf(statistic, nr as f64, df_den as f64)?;
        Ok(CausalityTest {
            statistic,
            pvalue,
            df_num: nr,
            df_den,
        })
    }
}

/// Survival function of the F distribution with `d1`/`d2` numerator and
/// denominator degrees of freedom,
///
/// ```text
/// SF(x) = I_{d2 / (d2 + d1 x)}(d2 / 2, d1 / 2)
/// ```
///
/// via the regularized incomplete beta function (Abramowitz & Stegun
/// 1964, eq. 26.6.2 with the symmetry `I_x(a, b) = 1 - I_{1-x}(b, a)`).
/// Evaluating the upper tail directly through `beta_inc` avoids the
/// `1 - cdf` cancellation, so p-values of order `1e-8` keep full
/// relative accuracy.
pub(crate) fn f_sf(x: f64, d1: f64, d2: f64) -> Result<f64, VarError> {
    if !d1.is_finite() || !d2.is_finite() || d1 <= 0.0 || d2 <= 0.0 {
        return Err(VarError::InvalidArgument {
            what: "F degrees of freedom must be positive and finite",
        });
    }
    if x.is_nan() {
        return Err(VarError::NonFinite { what: "F statistic" });
    }
    if x <= 0.0 {
        return Ok(1.0);
    }
    if x == f64::INFINITY {
        return Ok(0.0);
    }
    Ok(beta_inc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * x))?)
}
