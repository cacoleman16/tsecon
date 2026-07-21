//! The Andrews (1993) sup-F test with Hansen (1997) approximate p-values.
//!
//! For the pure-structural-change regression with a single break at an
//! unknown date, the statistic is the maximum over candidate dates `d`
//! (last observation of the first regime, `d in [h-1, T-h-1]`,
//! `h = ceil(trim * T)`) of the Wald-form Chow statistic
//!
//! ```text
//! F(d) = (T - 2q) * (SSR_0 - SSR_1(d) - SSR_2(d)) / (SSR_1(d) + SSR_2(d)),
//! ```
//!
//! exactly the statistic Hansen's response surfaces are calibrated to
//! (and the one R's `strucchange::Fstats` computes). Because the date is
//! searched over, the null distribution is nonstandard; the approximate
//! asymptotic p-value comes from Hansen's published response surface at
//! effective trimming `tau = h / T` (see the `tables` module).

use crate::check::validate;
use crate::error::BreaksError;
use crate::tables::HANSEN_SUP;
use tsecon_linalg::faer::linalg::solvers::Solve;
use tsecon_linalg::faer::{Mat, Side};
use tsecon_stats::chi2_sf;

/// Result of the Andrews sup-F (Quandt) unknown-break test.
#[derive(Debug, Clone, PartialEq)]
pub struct SupFTest {
    /// The sup-F statistic (Wald form; maximum of `f_path`).
    pub stat: f64,
    /// Hansen (1997) approximate asymptotic p-value.
    pub p_value: f64,
    /// Date attaining the maximum: 0-indexed last observation of the
    /// first regime.
    pub break_date: usize,
    /// Candidate dates `h-1 ..= T-h-1`, aligned with `f_path`.
    pub dates: Vec<usize>,
    /// The full Chow-style F path over the candidate dates.
    pub f_path: Vec<f64>,
    /// Minimal segment length `h = ceil(trim * T)` actually used.
    pub h: usize,
}

/// SSR of `y[0..=d]` on `x[0..=d]` for every `d >= h - 1` (index `d`;
/// earlier entries NaN), by one recursive normal-equation sweep.
fn growing_ssrs(
    y: &[f64],
    x: &[Vec<f64>],
    h: usize,
    reversed: bool,
) -> Result<Vec<f64>, BreaksError> {
    let t = y.len();
    let q = x.len();
    let mut xtx = vec![0.0_f64; q * q];
    let mut xty = vec![0.0_f64; q];
    let mut yty = 0.0_f64;
    let mut out = vec![f64::NAN; t];
    for step in 0..t {
        let j = if reversed { t - 1 - step } else { step };
        let yj = y[j];
        yty += yj * yj;
        for a in 0..q {
            let xa = x[a][j];
            xty[a] += xa * yj;
            for b in 0..=a {
                xtx[a * q + b] += xa * x[b][j];
            }
        }
        if step + 1 >= h {
            let (lo, hi) = if reversed { (j, t - 1) } else { (0, j) };
            let fitted = if q == 1 {
                if xtx[0] <= 0.0 {
                    return Err(BreaksError::Singular { start: lo, end: hi });
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
                    .map_err(|_| BreaksError::Singular { start: lo, end: hi })?
                    .solve(&rhs);
                (0..q).map(|a| sol[(a, 0)] * xty[a]).sum()
            };
            out[j] = (yty - fitted).max(0.0);
        }
    }
    Ok(out)
}

/// Hansen (1997) approximate asymptotic p-value for the sup-F statistic.
///
/// `stat` is the Wald-form sup-F, `q` the number of switching regressors
/// (`1..=10`), and `tau` the effective symmetric trimming fraction
/// (`h / T`, in `(0, 0.5]`). Transcribes the published response-surface
/// evaluation exactly as distributed in `strucchange::pvalue.Fstats`:
/// each tau grid row gives `P(chi2_v > b0 + b1 * stat)` and rows are
/// linearly interpolated in `tau`; `tau = 0.5` degenerates to the plain
/// `chi2_q` tail.
///
/// # Errors
///
/// [`BreaksError::UnsupportedQ`] for `q` outside `1..=10`, and
/// [`BreaksError::InvalidArgument`] for a negative/non-finite statistic
/// or `tau` outside `(0, 0.5]`.
pub fn hansen_supf_pvalue(stat: f64, q: usize, tau: f64) -> Result<f64, BreaksError> {
    if !(1..=10).contains(&q) {
        return Err(BreaksError::UnsupportedQ { q });
    }
    if !stat.is_finite() || stat < 0.0 {
        return Err(BreaksError::InvalidArgument {
            what: "the sup-F statistic must be finite and non-negative",
        });
    }
    if !(tau > 0.0 && tau <= 0.5) {
        return Err(BreaksError::InvalidArgument {
            what: "tau (the effective trimming h/T) must be in (0, 0.5]",
        });
    }
    let rows = &HANSEN_SUP[(q - 1) * 25..q * 25];
    let pp = |i: usize| -> Result<f64, BreaksError> {
        let [b0, b1, df] = rows[i];
        Ok(chi2_sf((b0 + b1 * stat).max(0.0), df)?)
    };
    let p = if tau == 0.5 {
        chi2_sf(stat, q as f64)?
    } else if tau <= 0.01 {
        pp(24)?
    } else if tau >= 0.49 {
        ((0.5 - tau) * pp(0)? + (tau - 0.49) * chi2_sf(stat, q as f64)?) * 100.0
    } else {
        let taua = (0.51 - tau) * 50.0;
        let t1 = taua.floor() as usize; // in 1..=24
        (t1 as f64 + 1.0 - taua) * pp(t1 - 1)? + (taua - t1 as f64) * pp(t1)?
    };
    Ok(p.clamp(0.0, 1.0))
}

/// Andrews sup-F test for a single break at an unknown date.
///
/// `y` is the response, `x` the `q` regressor columns (include the
/// constant explicitly), `trim` the trimming fraction: candidate break
/// dates leave at least `h = ceil(trim * T)` observations in each regime.
///
/// # Errors
///
/// Input validation errors ([`BreaksError::EmptyInput`],
/// [`BreaksError::DimensionMismatch`], [`BreaksError::NonFinite`],
/// [`BreaksError::InvalidArgument`], [`BreaksError::TrimTooSmall`]),
/// [`BreaksError::TooShort`] when `T < 2h` leaves no candidate date,
/// [`BreaksError::UnsupportedQ`] for more than 10 regressors,
/// [`BreaksError::Singular`] for collinear columns on some candidate
/// segment, and [`BreaksError::DegenerateFit`] when a candidate split
/// fits exactly.
pub fn sup_f_test(y: &[f64], x: &[Vec<f64>], trim: f64) -> Result<SupFTest, BreaksError> {
    let (t, q, h) = validate(y, x, trim)?;
    if q > 10 {
        return Err(BreaksError::UnsupportedQ { q });
    }
    if t < 2 * h {
        return Err(BreaksError::TooShort {
            t,
            needed: 2 * h,
            what: "the sup-F break search (no admissible break date)",
        });
    }
    let prefix = growing_ssrs(y, x, h, false)?;
    let suffix = growing_ssrs(y, x, h, true)?;
    let ssr0 = prefix[t - 1];
    let dates: Vec<usize> = ((h - 1)..=(t - h - 1)).collect();
    let mut f_path = Vec::with_capacity(dates.len());
    let mut best = f64::NEG_INFINITY;
    let mut best_date = dates[0];
    for &d in &dates {
        let s = prefix[d] + suffix[d + 1];
        if s <= 0.0 {
            return Err(BreaksError::DegenerateFit {
                what: "sup-F denominator SSR_1(d) + SSR_2(d)",
            });
        }
        let f = (t - 2 * q) as f64 * (ssr0 - s).max(0.0) / s;
        if f > best {
            best = f;
            best_date = d;
        }
        f_path.push(f);
    }
    let p_value = hansen_supf_pvalue(best, q, h as f64 / t as f64)?;
    Ok(SupFTest {
        stat: best,
        p_value,
        break_date: best_date,
        dates,
        f_path,
        h,
    })
}
