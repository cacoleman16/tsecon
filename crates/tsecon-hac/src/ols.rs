//! OLS with nonrobust, heteroskedasticity-robust (HC0-HC3), and HAC
//! (kernel sandwich) standard errors, following statsmodels
//! `OLS(...).fit(cov_type=...)` conventions exactly and agreeing with it to
//! 1e-10 relative on the goldens. HC2/HC3 carry one documented exception:
//! see "when the leverage correction runs out of digits" below.
//!
//! The point estimates solve the least-squares problem by Householder QR of
//! the design itself, `X = Q R` (Golub & Van Loan 2013, algorithm 5.2.1),
//! and the sandwich bread is `(X'X)^{-1} = R^{-1} R^{-T}`; `X'X` is never
//! formed. The design matrix is passed as explicit columns — include the
//! constant column yourself, statsmodels-style.
//!
//! ## Why QR, and why the columns are rescaled first
//!
//! Forming `X'X` squares the condition number, and a regression of a
//! *level* on its own lags and a constant is exactly where that bites: the
//! lagged-level columns are nearly collinear with the constant whenever the
//! mean is large next to the spread. Until 0.11.0 this module solved the
//! normal equations by Cholesky (plus one refinement step, which works
//! against the already-rounded `X'X` and cannot recover what forming it
//! lost); on a log-GDP-level target `adl_midas` lost digits down to `7e-8`
//! relative in its coefficients, and on `1000 + 0.01 AR(2)` to `3e-5`,
//! where statsmodels' pinv solve stays within `2e-12` and `3e-10` of an
//! exact rational solve. QR works on `X` and loses digits only in
//! proportion to `cond(X)`: it is now as close to the exact solve as
//! statsmodels is (`fixtures/adl_midas.json`, `level_cases`).
//!
//! Before factoring, `y` and every column are divided by a power of two
//! near their largest magnitude (and the results multiplied back). Scaling
//! by a power of two is exact in floating point and Householder QR is
//! exactly equivariant under it, so on ordinary data this changes no bit of
//! any result; what it removes is overflow and underflow. Without it the
//! sums of squares of a series near `1e154` overflow (the HAC meat, of
//! order `y^4`, already at `1e77`), and finite inputs whose answer is
//! perfectly representable came back as NaN — or, once `X'X` overflowed, as
//! a false "collinear columns" refusal. With it every returned number is
//! finite whenever the answer itself is representable.
//!
//! A column is refused as collinear when its component orthogonal to the
//! columns before it, `|R_jj|`, is at most `sqrt(1e-13)` of its own norm —
//! the same verdict the Cholesky pivot test `R_jj^2 <= 1e-13 ||x_j||^2`
//! gave, now read off a factor that is accurate to `eps cond(X)`.
//!
//! Sandwich covariances are `(X'X)^{-1} S (X'X)^{-1}` with the meat `S`
//! built from scores `s_t = x_t u_t`:
//!
//! * nonrobust: `sigma2_hat (X'X)^{-1}`, `sigma2_hat = RSS/(n - k)`;
//! * HC0 (White 1980): `S = sum_t u_t^2 x_t x_t'`;
//! * HC1 (MacKinnon & White 1985): HC0 scaled by `n/(n - k)`;
//! * HC2 (Horn, Horn & Duncan 1975): `S = sum_t u_t^2/(1 - h_t) x_t x_t'`;
//! * HC3 (MacKinnon & White 1985): `S = sum_t u_t^2/(1 - h_t)^2 x_t x_t'`;
//! * HAC (Newey & West 1987): `S = Gamma_0 + sum_{j>=1} w_j (Gamma_j +
//!   Gamma_j')` with `Gamma_j = sum_{t>j} s_t s_{t-j}'` and kernel weights
//!   `w_j` from [`Kernel::weight`]; optionally scaled by `n/(n - k)`
//!   (statsmodels `use_correction=True`);
//! * EWC (Lazarus, Lewis, Stock & Watson 2018): `S = n * Omega_hat(s)`, the
//!   equal-weighted cosine long-run covariance of the scores
//!   ([`crate::ewc_lrv_matrix`]) with `B` degrees of freedom — no `n/(n - k)`
//!   factor, no prewhitening. Its t-statistics are referred to **Student t
//!   with `B` degrees of freedom**, not the normal; see [`SeType::Ewc`].
//!
//! ## Why the leverage-corrected members exist
//!
//! `h_t = x_t' (X'X)^{-1} x_t` is observation `t`'s leverage. OLS residuals
//! are shrunk toward zero exactly where leverage is high — under
//! homoskedasticity `E[u_t^2] = sigma^2 (1 - h_t)` — so the White meat
//! systematically understates the variance contribution of high-leverage
//! points, and HC0/HC1 standard errors come out too small in small samples
//! with unbalanced designs. HC2 divides that bias out (it is exactly
//! unbiased for the covariance under homoskedasticity); HC3 divides it out
//! twice, approximating the jackknife (Efron 1982) and deliberately
//! over-correcting, which is why Long & Ervin (2000) recommend it as the
//! default for `n < 250`.
//!
//! The library's interval-coverage audit measured this crate's HC1 at 0.682
//! coverage (nominal 0.95) on a `T = 25`, `x ~ chi2(1)`, `sd(e|x) = x`
//! design; a **NumPy HC2/HC3 reference on the identical draws** reached
//! 0.773 and 0.863 respectively. Those two are references, not measurements
//! of the code below — the audit predates these variants and has not been
//! re-run against them — but they size the prize: the `n/(n - k)` factor
//! buys ~0.015 at `k = 2`, the `1/(1 - h_t)` weights buy the rest. Neither
//! HC2 nor HC3 takes the `n/(n - k)` inflation on top (statsmodels does not
//! either).
//!
//! ## When the leverage correction runs out of digits
//!
//! `h_t = ||R^{-T} x_t||^2` is accumulated with an absolute error of a few
//! ulp, so forming `1 - h_t` is catastrophic cancellation once `h_t` is
//! close to 1: the complement's *relative* error is of order
//! `eps_mach / (1 - h_t)`, and HC2's weight inherits it once, HC3's twice.
//! This is a property of the subtraction, not of the conditioning — the
//! `near_singleton` fixture reaches `1 - h_t = 1e-11` with `cond(X'X)` of
//! only ~47, so no conditioning diagnostic would flag it.
//!
//! Consequence, measured in
//! `golden.rs::hc2_hc3_parity_degrades_as_the_leverage_complement_approaches_noise`:
//! HC2/HC3 hold the 1e-10 spec tolerance while `1 - h_t` stays above roughly
//! `1e-5`, and degrade smoothly below it — about 4e-10 at `1 - h_t = 1e-6`
//! and about 1e-5 at `1 - h_t = 1e-10`, with no error and no warning.
//! `LEVERAGE_FLOOR` stops the slide at `1 - h_t = 1e-12`. Everything else in
//! this module — including HC0/HC1 on the very same fits — stays at 1e-10.
//! Near-singleton controls are ordinary in dummy-heavy and fixed-effects
//! designs, so this band is reachable in practice: if a coefficient's HC3
//! standard error matters to three digits and its observation is nearly
//! interpolated, check `1 - h_t` before trusting the tail of the number.

use crate::error::HacError;
use crate::ewc::ewc_matrix_impl;
use crate::kernel::Kernel;
use crate::validate::{check_bandwidth, check_finite};

/// Standard-error flavor for [`OlsFit::inference`], mirroring statsmodels
/// `cov_type=`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeType {
    /// Classical spherical-errors covariance `sigma2_hat (X'X)^{-1}` with
    /// `sigma2_hat = RSS/(n - k)` (statsmodels `cov_type="nonrobust"`).
    NonRobust,
    /// White (1980) heteroskedasticity-robust covariance
    /// (statsmodels `cov_type="HC0"`).
    Hc0,
    /// HC0 with the `n/(n - k)` degrees-of-freedom inflation
    /// (MacKinnon & White 1985; statsmodels `cov_type="HC1"`).
    Hc1,
    /// Leverage-corrected robust covariance weighting `u_t^2` by
    /// `1/(1 - h_t)`, unbiased under homoskedasticity (Horn, Horn & Duncan
    /// 1975; statsmodels `cov_type="HC2"`). No `n/(n - k)` inflation.
    ///
    /// Matches statsmodels to the crate's 1e-10 spec tolerance *while
    /// `1 - h_t` is not near machine noise*; see the module docs' "when the
    /// leverage correction runs out of digits". Below `1 - h_t ~ 1e-5` the
    /// two implementations agree only to about `eps_mach / (1 - h_t)`.
    Hc2,
    /// Jackknife-approximating robust covariance weighting `u_t^2` by
    /// `1/(1 - h_t)^2` (MacKinnon & White 1985; statsmodels
    /// `cov_type="HC3"`). The small-sample default recommended by Long &
    /// Ervin (2000); no `n/(n - k)` inflation.
    ///
    /// Same parity caveat as [`SeType::Hc2`], and twice as sharp: the
    /// squared weight doubles the relative error of `1 - h_t`.
    Hc3,
    /// Kernel HAC sandwich covariance (statsmodels `cov_type="HAC"` with
    /// `cov_kwds={"maxlags": bandwidth, "use_correction": ...}` when the
    /// kernel is [`Kernel::Bartlett`]).
    Hac {
        /// Kernel weighting the score autocovariances.
        kernel: Kernel,
        /// Bandwidth in the [`Kernel::weight`] convention (for
        /// Bartlett/Parzen/truncated this is the lag-truncation `maxlags`).
        bandwidth: f64,
        /// Apply the small-sample `n/(n - k)` correction to the covariance
        /// (statsmodels `use_correction`). Note statsmodels is internally
        /// inconsistent about the default: `cov_type="HAC"` defaults it
        /// **off**, while the `cov_hac_simple` helper defaults it on.
        /// tsecon's Python surface deliberately defaults `true` (the
        /// statistically recommended finite-sample choice), so matching a
        /// default statsmodels `cov_type="HAC"` call requires `false` here.
        use_correction: bool,
    },
    /// Equal-weighted cosine (EWC) HAR covariance of Lazarus, Lewis, Stock &
    /// Watson (2018): the sandwich `(X'X)^{-1} [n Omega_hat(s)] (X'X)^{-1}`
    /// with `Omega_hat` the EWC long-run covariance of the scores
    /// `s_t = x_t u_t` over the first `b` type-II cosine frequencies
    /// ([`crate::ewc_lrv_matrix`]).
    ///
    /// **Inference contract.** `Omega_hat` is (asymptotically)
    /// `Omega * chi^2_B / B`, so a t-statistic built from these standard
    /// errors is Student t with `b` degrees of freedom under the null
    /// (fixed-b asymptotics) — refer `tvalues` to `t_B`, never to the
    /// normal. A Wald test of `q` restrictions scales as
    /// `F* = (B - q + 1)/(B q) W ~ F(q, B - q + 1)`.
    ///
    /// No `n/(n - k)` factor and no prewhitening: LLSW's size results are for
    /// the plain estimator. [`crate::ewc_default_b`] gives the recommended
    /// `b = round(0.4 n^(2/3))`. Requires `1 <= b <= n - 1`.
    Ewc {
        /// Degrees of freedom `B`: the number of cosine frequencies averaged,
        /// which is also the Student-t degrees of freedom of the t-statistic.
        b: usize,
    },
}

/// How much of the leverage correction a heteroskedasticity-robust meat
/// carries — the only thing HC0, HC2 and HC3 disagree about, which is why
/// they share one accumulation in [`OlsFit::hc_cov`]. (HC1 is HC0 with a
/// scalar applied afterwards, so it needs no variant of its own.)
#[derive(Debug, Clone, Copy)]
enum HcFlavor {
    Hc0,
    Hc2,
    Hc3,
}

impl HcFlavor {
    /// Exponent `p` in the weight `(1 - h_t)^{-p}` applied to `u_t^2`:
    /// 0 (HC0, no leverage correction), 1 (HC2), 2 (HC3).
    fn leverage_power(self) -> u32 {
        match self {
            HcFlavor::Hc0 => 0,
            HcFlavor::Hc2 => 1,
            HcFlavor::Hc3 => 2,
        }
    }

    /// statsmodels' `cov_type` spelling, for error messages.
    fn name(self) -> &'static str {
        match self {
            HcFlavor::Hc0 => "HC0",
            HcFlavor::Hc2 => "HC2",
            HcFlavor::Hc3 => "HC3",
        }
    }
}

/// Smallest `1 - h_t` HC2/HC3 will divide by.
///
/// This is an absolute floor on the *complement*, deliberately, and it is
/// not scaled by anything. `h_t = ||R^{-T} x_t||^2` is accumulated with an
/// absolute error of a few ulp of 1, so `1 - h_t` retains roughly
/// `eps_mach / (1 - h_t)` relative error — a cancellation law in the
/// complement itself, independent of how well conditioned `X'X` is. (The
/// `near_singleton` fixture drives `1 - h_t` to `1e-13` at `cond(X'X) ~ 47`;
/// scaling this constant by a conditioning proxy would not move on that
/// design at all.) So a floor on `1 - h_t` *is* a floor on precision: at
/// `1e-12` the HC3 weight retains about three correct digits, which is the
/// least this crate is willing to return without saying so. Below it the
/// complement is noise, the weight would exceed `1e24`, and one observation
/// would swamp every other.
///
/// The floor is a backstop, not a parity boundary: agreement with
/// statsmodels leaves the crate's 1e-10 spec tolerance around
/// `1 - h_t ~ 1e-5`, seven orders above this constant. See the module docs.
///
/// A design that trips the floor has an observation the fit interpolates
/// exactly (a singleton dummy, say), which no reweighting of a zero residual
/// can repair.
const LEVERAGE_FLOOR: f64 = 1e-12;

/// Standard errors, t-statistics, and the full parameter covariance for
/// one [`SeType`].
#[derive(Debug, Clone, PartialEq)]
pub struct OlsInference {
    /// Parameter covariance matrix, `k x k` row-major.
    pub cov: Vec<f64>,
    /// Standard errors `sqrt(diag(cov))`, one per parameter.
    pub bse: Vec<f64>,
    /// t-statistics `params / bse`, one per parameter.
    pub tvalues: Vec<f64>,
}

/// A fitted OLS regression; produced by [`ols`].
///
/// The private fields hold the problem in its *equilibrated* units (see
/// the module docs): every design column divided by its power-of-two scale
/// `col_scale[j]` and the response by `y_scale`. The covariance is built
/// there and multiplied back, which is exact.
#[derive(Debug, Clone, PartialEq)]
pub struct OlsFit {
    /// Coefficient estimates, in the order the design columns were passed.
    pub params: Vec<f64>,
    /// Residuals `u_t = y_t - x_t' b`, length `n`.
    pub residuals: Vec<f64>,
    /// Number of observations `n`.
    pub nobs: usize,
    /// Number of regressors `k` (including the constant if supplied).
    pub nparams: usize,
    /// `(X~'X~)^{-1} = R^{-1} R^{-T}` of the equilibrated design, `k x k`
    /// row-major — the nonrobust covariance's shape.
    xtx_inv: Vec<f64>,
    /// The triangular factor `R` of the equilibrated design (`k x k`,
    /// row-major, upper), from which the robust covariances build their
    /// influence vectors.
    r: Vec<f64>,
    /// The equilibrated design columns `x_j / col_scale[j]`, needed for the
    /// sandwich meats.
    x_cols: Vec<Vec<f64>>,
    /// The equilibrated residuals `u_t / y_scale`.
    resid_eq: Vec<f64>,
    /// Power-of-two scale of each design column.
    col_scale: Vec<f64>,
    /// Power-of-two scale of the response.
    y_scale: f64,
}

/// Fit `y = X b + u` by ordinary least squares.
///
/// `x_cols` are the columns of the design matrix; include the constant
/// column explicitly (statsmodels exog convention — the golden fixture
/// assembles `[const, x1, x2]`). Solved by Householder QR of the
/// equilibrated design (see the module docs), which matches statsmodels'
/// pinv-based solve to the accuracy the data allow, `eps cond(X)`.
///
/// # Errors
///
/// [`HacError::EmptyDesign`] with no columns,
/// [`HacError::DimensionMismatch`] if a column's length differs from
/// `y.len()`, [`HacError::DegreesOfFreedom`] unless `n > k`,
/// [`HacError::NonFinite`] on NaN/inf input, and
/// [`HacError::SingularDesign`] when a column is numerically a linear
/// combination of the columns before it (collinear columns).
pub fn ols(y: &[f64], x_cols: &[Vec<f64>]) -> Result<OlsFit, HacError> {
    let n = y.len();
    let k = x_cols.len();
    if k == 0 {
        return Err(HacError::EmptyDesign);
    }
    for (column, col) in x_cols.iter().enumerate() {
        if col.len() != n {
            return Err(HacError::DimensionMismatch {
                what: "OLS design column",
                column,
                expected: n,
                got: col.len(),
            });
        }
    }
    if n <= k {
        return Err(HacError::DegreesOfFreedom { n, k });
    }
    check_finite(y, "OLS response")?;
    for col in x_cols {
        check_finite(col, "OLS design column")?;
    }

    // Equilibrate by powers of two (exact; see the module docs).
    let y_scale = pow2_scale(y);
    let col_scale: Vec<f64> = x_cols.iter().map(|c| pow2_scale(c)).collect();
    let y_eq: Vec<f64> = y.iter().map(|v| v / y_scale).collect();
    let work: Vec<Vec<f64>> = x_cols
        .iter()
        .zip(&col_scale)
        .map(|(c, s)| c.iter().map(|v| v / s).collect())
        .collect();

    let (r, qty) =
        householder_qr(&work, &y_eq).ok_or(HacError::SingularDesign { what: "OLS design" })?;
    let params_eq = back_substitute(&r, k, &qty);
    let resid_eq: Vec<f64> = (0..n)
        .map(|t| {
            let mut fit = 0.0;
            for (b, col) in params_eq.iter().zip(work.iter()) {
                fit += b * col[t];
            }
            y_eq[t] - fit
        })
        .collect();

    // (X~'X~)^{-1} = R^{-1} R^{-T}.
    let r_inv = upper_triangular_inverse(&r, k);
    let mut xtx_inv = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            // R^{-1} is upper triangular: row i is zero before column i.
            let mut acc = 0.0;
            for l in i..k {
                acc += r_inv[i * k + l] * r_inv[j * k + l];
            }
            xtx_inv[i * k + j] = acc;
            xtx_inv[j * k + i] = acc;
        }
    }

    let params = params_eq
        .iter()
        .zip(&col_scale)
        .map(|(b, s)| rescale(*b, y_scale, *s))
        .collect();
    let residuals = resid_eq.iter().map(|u| u * y_scale).collect();

    Ok(OlsFit {
        params,
        residuals,
        nobs: n,
        nparams: k,
        xtx_inv,
        r,
        x_cols: work,
        resid_eq,
        col_scale,
        y_scale,
    })
}

/// The power of two at or just below `max_t |v_t|` (1 for an all-zero
/// slice), so `v / scale` has magnitudes in `[0, 2)` and the division is
/// exact. A subnormal maximum takes the smallest normal power, `2^-1022`.
fn pow2_scale(v: &[f64]) -> f64 {
    let m = v.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
    if m == 0.0 || !m.is_finite() {
        return 1.0;
    }
    let exponent_bits = m.to_bits() & (0x7ff_u64 << 52);
    if exponent_bits == 0 {
        f64::MIN_POSITIVE
    } else {
        f64::from_bits(exponent_bits)
    }
}

/// `v * num / den` for powers of two `num`, `den`: one exact multiplication
/// by their ratio, or two steps when the ratio itself would leave the
/// exponent range.
fn rescale(v: f64, num: f64, den: f64) -> f64 {
    let ratio = num / den;
    if ratio.is_finite() && ratio != 0.0 {
        v * ratio
    } else {
        (v * num) / den
    }
}

/// Rows per block of the row-streaming QR below.
const QR_BLOCK_ROWS: usize = 128;

/// Householder QR without pivoting (Golub & Van Loan 2013, algorithm
/// 5.2.1) of the `n x k` matrix held as the columns `cols`, with the
/// response `b` carried along as a `(k+1)`-th column. Returns `R` (`k x k`
/// upper triangular, row-major) and the first `k` entries of `Q'b`; the
/// inputs are not modified. `None` when a column's component orthogonal to
/// the columns before it is at most `sqrt(1e-13)` of its norm (collinear
/// columns; see the module docs).
///
/// The rows are streamed in blocks of [`QR_BLOCK_ROWS`]: each block is
/// stacked under the triangle so far, `[R; X_block]`, and reduced to a new
/// triangle by the same Householder reflections, which only touch row `j`
/// of `R` and the block (the rows of `R` below the diagonal are zero). It
/// is the same orthogonal reduction, so it is backward stable like the
/// textbook one-pass form, but its working set is one block instead of the
/// whole design: on a tall `n x k` design the one-pass form streams every
/// trailing column through memory once per reflection, which made
/// `breusch_godfrey_test`'s `100000 x 567` auxiliary regression at its work
/// cap take 39 s against 29 s for the normal equations it replaces; this
/// form takes 15 s (on a loaded 4-core machine).
fn householder_qr(cols: &[Vec<f64>], b: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let k = cols.len();
    let n = b.len();
    let w = k + 1;
    // The triangle so far, row-major k x (k+1); column k is (Q'b)[..k].
    let mut r = vec![0.0_f64; k * w];
    // One block, column-major: column c holds rows r0..r1 of column c.
    let mut block = vec![0.0_f64; w * QR_BLOCK_ROWS];
    let mut r0 = 0;
    while r0 < n {
        let r1 = (r0 + QR_BLOCK_ROWS).min(n);
        let m = r1 - r0;
        for (c, col) in cols.iter().map(|c| c.as_slice()).chain([b]).enumerate() {
            block[c * m..(c + 1) * m].copy_from_slice(&col[r0..r1]);
        }
        for j in 0..k {
            let x0 = r[j * w + j];
            let (head, rest) = block.split_at_mut((j + 1) * m);
            let tail = &head[j * m..(j + 1) * m];
            let tail_sq = dot(tail, tail);
            if tail_sq == 0.0 {
                // Nothing below the diagonal to annihilate in this block.
                continue;
            }
            let norm = (x0 * x0 + tail_sq).sqrt();
            let alpha = if x0 >= 0.0 { -norm } else { norm };
            let v0 = x0 - alpha;
            let vtv = v0 * v0 + tail_sq;
            r[j * w + j] = alpha;
            // Reflect the remaining columns (and b): c <- c - 2 v (v'c)/(v'v),
            // with v = [v0; tail] acting on [R_jc; block column c].
            for (cc, col) in rest[..(w - j - 1) * m].chunks_exact_mut(m).enumerate() {
                let c = j + 1 + cc;
                let fac = 2.0 * (v0 * r[j * w + c] + dot(tail, col)) / vtv;
                r[j * w + c] -= fac * v0;
                for (ci, vi) in col.iter_mut().zip(tail) {
                    *ci -= fac * vi;
                }
            }
        }
        r0 = r1;
    }
    // Relative pivot tolerance: guards against collinear columns.
    for (j, col) in cols.iter().enumerate() {
        let rjj = r[j * w + j];
        if rjj * rjj <= (1e-13 * dot(col, col)).max(f64::MIN_POSITIVE) {
            return None;
        }
    }
    let qtb = (0..k).map(|j| r[j * w + k]).collect();
    let r = (0..k * k).map(|ij| r[(ij / k) * w + ij % k]).collect();
    Some((r, qtb))
}

/// `a . b` with four interleaved partial sums: the reflections' dot
/// products are the QR's inner loop, and one running sum would leave it
/// bound by the latency of a dependent add chain. (Power-of-two scaling
/// still commutes with it exactly.)
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0_f64; 4];
    let mut ca = a.chunks_exact(4);
    let mut cb = b.chunks_exact(4);
    for (x, y) in (&mut ca).zip(&mut cb) {
        acc[0] += x[0] * y[0];
        acc[1] += x[1] * y[1];
        acc[2] += x[2] * y[2];
        acc[3] += x[3] * y[3];
    }
    let tail: f64 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(x, y)| x * y)
        .sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// Solve `R x = c` for upper-triangular `R` (`k x k`, row-major).
fn back_substitute(r: &[f64], k: usize, c: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let mut acc = c[i];
        for j in (i + 1)..k {
            acc -= r[i * k + j] * x[j];
        }
        x[i] = acc / r[i * k + i];
    }
    x
}

/// `R^{-1}` of an upper-triangular `R` (`k x k`, row-major), column by
/// column by back substitution; upper triangular itself.
fn upper_triangular_inverse(r: &[f64], k: usize) -> Vec<f64> {
    let mut inv = vec![0.0_f64; k * k];
    let mut unit = vec![0.0_f64; k];
    for j in 0..k {
        unit[j] = 1.0;
        let col = back_substitute(r, k, &unit);
        for i in 0..k {
            inv[i * k + j] = col[i];
        }
        unit[j] = 0.0;
    }
    inv
}

impl OlsFit {
    /// Standard errors, t-statistics, and parameter covariance under the
    /// requested [`SeType`] (see the module docs for the exact formulas
    /// and references).
    ///
    /// # Errors
    ///
    /// [`HacError::InvalidBandwidth`] for a negative/non-finite HAC
    /// bandwidth; [`HacError::FullLeverage`] if [`SeType::Hc2`]/
    /// [`SeType::Hc3`] meet an observation whose `1 - h_t` has fallen to the
    /// crate's leverage floor of `1e-12` (its weight would be noise);
    /// [`HacError::NumericalBreakdown`] if a covariance diagonal comes out
    /// negative, which can only happen with a non-positive-semi-definite
    /// kernel ([`Kernel::Truncated`]); [`HacError::InvalidDof`] if
    /// [`SeType::Ewc`] asks for `b = 0` or `b > n - 1`.
    pub fn inference(&self, se_type: SeType) -> Result<OlsInference, HacError> {
        let n = self.nobs;
        let k = self.nparams;
        let nf = n as f64;
        let dof_scale = nf / (n - k) as f64;

        // Built in the equilibrated units (module docs) and multiplied back
        // below; every step is equivariant under the power-of-two scales.
        let cov_eq: Vec<f64> = match se_type {
            SeType::NonRobust => {
                let rss: f64 = self.resid_eq.iter().map(|u| u * u).sum();
                let sigma2 = rss / (n - k) as f64;
                self.xtx_inv.iter().map(|v| sigma2 * v).collect()
            }
            SeType::Hc0 => self.hc_cov(HcFlavor::Hc0)?,
            SeType::Hc1 => {
                let mut cov = self.hc_cov(HcFlavor::Hc0)?;
                for v in &mut cov {
                    *v *= dof_scale;
                }
                cov
            }
            SeType::Hc2 => self.hc_cov(HcFlavor::Hc2)?,
            SeType::Hc3 => self.hc_cov(HcFlavor::Hc3)?,
            SeType::Hac {
                kernel,
                bandwidth,
                use_correction,
            } => {
                check_bandwidth(bandwidth)?;
                let mut cov = self.hac_cov(kernel, bandwidth);
                if use_correction {
                    for v in &mut cov {
                        *v *= dof_scale;
                    }
                }
                cov
            }
            SeType::Ewc { b } => self.ewc_cov(b)?,
        };

        let mut bse = Vec::with_capacity(k);
        for i in 0..k {
            let v = cov_eq[i * k + i];
            if v < 0.0 {
                return Err(HacError::NumericalBreakdown {
                    what: "sandwich covariance diagonal (non-PSD kernel?)",
                });
            }
            bse.push(rescale(v.sqrt(), self.y_scale, self.col_scale[i]));
        }
        let cov = (0..k * k)
            .map(|ij| {
                let (i, j) = (ij / k, ij % k);
                let v = rescale(cov_eq[ij], self.y_scale, self.col_scale[i]);
                rescale(v, self.y_scale, self.col_scale[j])
            })
            .collect();
        let tvalues = self
            .params
            .iter()
            .zip(bse.iter())
            .map(|(p, s)| p / s)
            .collect();
        Ok(OlsInference { cov, bse, tvalues })
    }

    /// Row `t` of `Q_1 = X~ R^{-1}`, `q_t = R^{-T} x~_t`, by forward
    /// substitution into `q` (length `k`): observation `t` in the
    /// orthonormal basis the QR found. Its squared norm is the leverage.
    fn q_row(&self, t: usize, q: &mut [f64]) {
        let k = self.nparams;
        let r = &self.r;
        for i in 0..k {
            let mut acc = self.x_cols[i][t];
            for l in 0..i {
                acc -= r[l * k + i] * q[l];
            }
            q[i] = acc / r[i * k + i];
        }
    }

    /// Influence vector `d_t = (X~'X~)^{-1} x~_t f = R^{-1} q_t f` of the
    /// row `q` from [`OlsFit::q_row`], by back substitution into `d`.
    ///
    /// Every robust covariance below is a weighted sum of their outer
    /// products, `sum_{t,s} W_ts d_t d_s'` with `f = u~_t` (times a
    /// leverage factor for HC2/HC3), which is the textbook sandwich
    /// `(X'X)^{-1} [sum W_ts s_t s_s'] (X'X)^{-1}` of the scores
    /// `s_t = x_t u_t` with the bread moved inside. Moving it inside is
    /// what keeps the digits: the outer `(X'X)^{-1} S (X'X)^{-1}` products
    /// cancel entries of order `cond(X)^2`, and on a level target (an index
    /// near 100 regressed on its own lags, `cond(X) ~ 1e5`) the HAC
    /// standard errors came out `4e-9` from the exact rational answer that
    /// way, and `6e-14` this way. statsmodels, which multiplies the
    /// sandwich out, sits at `8e-10` on the same design.
    fn d_row(&self, q: &[f64], f: f64, d: &mut [f64]) {
        let k = self.nparams;
        let r = &self.r;
        for i in (0..k).rev() {
            let mut acc = q[i] * f;
            for l in (i + 1)..k {
                acc -= r[i * k + l] * d[l];
            }
            d[i] = acc / r[i * k + i];
        }
    }

    /// The influence vectors `d_t` (`n x k`, row-major) with `f = u~_t`.
    fn influence(&self) -> Vec<f64> {
        let (n, k) = (self.nobs, self.nparams);
        let mut q = vec![0.0_f64; k];
        let mut d = vec![0.0_f64; n * k];
        for t in 0..n {
            self.q_row(t, &mut q);
            self.d_row(&q, self.resid_eq[t], &mut d[t * k..(t + 1) * k]);
        }
        d
    }

    /// Heteroskedasticity-robust covariance `sum_t d_t d_t'` of the
    /// influence vectors (see [`OlsFit::d_row`]) with `f = u~_t w_t^{1/2}`,
    /// `w_t` the leverage weight of the requested [`HcFlavor`] — the White
    /// sandwich of the meat `sum_t w_t u_t^2 x_t x_t'` (statsmodels'
    /// un-normalized convention). HC0, HC2 and HC3 differ only in `w_t`, so
    /// they share this accumulation, one observation at a time.
    ///
    /// # Errors
    ///
    /// [`HacError::FullLeverage`] when a leverage-corrected flavor meets
    /// `h_t` numerically equal to one; see [`OlsFit::leverage_weight`].
    fn hc_cov(&self, flavor: HcFlavor) -> Result<Vec<f64>, HacError> {
        let k = self.nparams;
        let mut q = vec![0.0_f64; k];
        let mut d = vec![0.0_f64; k];
        let mut s = vec![0.0_f64; k * k];
        for t in 0..self.nobs {
            self.q_row(t, &mut q);
            let w = Self::leverage_weight(flavor, t, &q)?;
            self.d_row(&q, self.resid_eq[t] * w.sqrt(), &mut d);
            for i in 0..k {
                for j in 0..=i {
                    s[i * k + j] += d[i] * d[j];
                }
            }
        }
        for i in 0..k {
            for j in 0..i {
                s[j * k + i] = s[i * k + j];
            }
        }
        Ok(s)
    }

    /// The weight `w_t` the HC family applies to `u_t^2`: `1` for HC0,
    /// `1/(1 - h_t)` for HC2, `1/(1 - h_t)^2` for HC3, with the leverage
    /// `h_t = x_t' (X'X)^{-1} x_t = ||q_t||^2` (the hat-matrix diagonal,
    /// statsmodels `get_influence().hat_matrix_diag`) read off the row `q`
    /// of [`OlsFit::q_row`] as a sum of squares.
    ///
    /// # Errors
    ///
    /// [`HacError::FullLeverage`] naming the first observation whose
    /// `1 - h_t` falls to [`LEVERAGE_FLOOR`] or below. In exact arithmetic
    /// `h_t = 1` means the fit passes through that point (`u_t = 0`), so the
    /// weighted term is `0/0`; numerically the complement is then pure
    /// cancellation noise and the HC3 weight explodes past `1e24`, swamping
    /// every other observation. Erroring out beats returning an inf/NaN
    /// standard error that looks like a number — statsmodels, which has no
    /// such guard, returns `inf` on the same design.
    fn leverage_weight(flavor: HcFlavor, t: usize, q: &[f64]) -> Result<f64, HacError> {
        let power = flavor.leverage_power();
        if power == 0 {
            return Ok(1.0);
        }
        let h: f64 = q.iter().map(|v| v * v).sum();
        let rest = 1.0 - h;
        if !rest.is_finite() || rest <= LEVERAGE_FLOOR {
            return Err(HacError::FullLeverage {
                what: flavor.name(),
                index: t,
                leverage: h,
            });
        }
        let inv = 1.0 / rest;
        Ok(if power == 1 { inv } else { inv * inv })
    }

    /// HAC covariance `sum_j w_j (Gamma_j + Gamma_j')` (`Gamma_0` once) of
    /// the influence vectors, `Gamma_j = sum_{t>j} d_t d_{t-j}'` — the
    /// sandwich of statsmodels' `sandwich_covariance.S_hac_simple` meat.
    fn hac_cov(&self, kernel: Kernel, bandwidth: f64) -> Vec<f64> {
        let n = self.nobs;
        let k = self.nparams;
        let d = self.influence();

        let mut s = vec![0.0_f64; k * k];
        for lag in 0..n {
            let w = kernel.weight(lag, bandwidth);
            if lag > 0 && w == 0.0 && kernel.truncates() {
                break;
            }
            // Gamma_lag[i][j] = sum_{t=lag}^{n-1} d_t[i] d_{t-lag}[j].
            for t in lag..n {
                let row_t = &d[t * k..(t + 1) * k];
                let row_l = &d[(t - lag) * k..(t - lag + 1) * k];
                for i in 0..k {
                    for j in 0..k {
                        let g = row_t[i] * row_l[j];
                        if lag == 0 {
                            s[i * k + j] += g;
                        } else {
                            // w * (Gamma + Gamma'): add to both (i,j) and (j,i).
                            s[i * k + j] += w * g;
                            s[j * k + i] += w * g;
                        }
                    }
                }
            }
        }
        s
    }

    /// EWC covariance `n * Omega_hat(d)`: the equal-weighted cosine long-run
    /// covariance over `b` frequencies of the influence vectors — the
    /// sandwich of the meat `n * Omega_hat(s)` of the scores, the scaling
    /// under which it estimates the same object as the un-normalized HAC
    /// meat (statsmodels' "sum" convention). `Omega_hat` is a fixed
    /// quadratic form in its argument, so moving the bread inside is exact
    /// in exact arithmetic.
    fn ewc_cov(&self, b: usize) -> Result<Vec<f64>, HacError> {
        let (n, k) = (self.nobs, self.nparams);
        let d = self.influence();
        let cols: Vec<Vec<f64>> = (0..k)
            .map(|i| (0..n).map(|t| d[t * k + i]).collect())
            .collect();
        let mut s = ewc_matrix_impl(&cols, b, "OLS EWC covariance")?;
        let nf = n as f64;
        for v in &mut s {
            *v *= nf;
        }
        Ok(s)
    }
}
