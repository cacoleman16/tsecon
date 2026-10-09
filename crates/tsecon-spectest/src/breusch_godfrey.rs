//! Breusch-Godfrey Lagrange-multiplier test for serial correlation of order
//! up to `p` in the errors of `y = X b + u` (Breusch 1978; Godfrey 1978).
//!
//! Fit the regression by OLS, then run the auxiliary regression of the
//! residuals on the design and `p` of their own lags,
//!
//! ```text
//! u_t = x_t' g + a_1 u_{t-1} + ... + a_p u_{t-p} + v_t,     t = 1..T,
//! ```
//!
//! and test `a_1 = ... = a_p = 0`. Two forms of the same test are returned:
//!
//! ```text
//! LM = T * R^2_aux                                        ~ chi2(p)
//! F  = [(SSR_r - SSR_aux) / p] / [SSR_aux / (T - k_aux)]  ~ F(p, T - k_aux)
//! ```
//!
//! with `R^2_aux` the centered coefficient of determination of the auxiliary
//! regression, `SSR_r` the residual sum of squares of the same regression
//! without the lag columns, and `k_aux` the number of auxiliary regressors.
//!
//! **Convention (pinned to statsmodels `acorr_breusch_godfrey`).** The lagged
//! residuals are padded with zeros: `u_{t-j}` is taken as `0` when
//! `t - j < 1`, so the auxiliary regression keeps all `T` observations.
//! (The other common convention drops the first `p` observations instead;
//! it is asymptotically equivalent but gives different numbers in a finite
//! sample, so the two must not be mixed.) The default order is
//! `p = min(10, T / 5)` (integer division). When the design does not span a
//! constant — no constant column and no implicit one such as a set of
//! complementary dummies — a column of ones is added to the auxiliary design
//! (`constant_added`), so `R^2_aux` is always centered on an intercept and
//! `k_aux = k + p + 1`; otherwise `k_aux = k + p`. In that case the LM
//! form also counts what the added constant explains (`R^2_aux` is measured
//! against the mean of `u`, and the constant explains its projection on
//! `M_X 1`), so it over-rejects; the F-form, whose restricted regression is
//! `u` on `[X, 1]`, tests the lags alone. The LM statistic is
//! valid with lagged dependent variables among the regressors, which is the
//! test's advantage over Durbin-Watson.
//!
//! **The implicit-constant rule.** The design spans a constant when the
//! regression of a column of ones on it (each column rescaled by a power of
//! two) leaves a residual `m = M_X 1` of relative size `|m| / |1|` at most
//! `1e-10`. The rule is invariant to rescaling a column. statsmodels instead
//! compares numpy's `matrix_rank` of `[1, X]` with that of `X`, a rule that
//! is not: for `X = [z, 1e9 + t]` it finds an implicit constant (and adds
//! none) where the rule here does not (`fixtures/serial_corr.json`,
//! `conditioning_cases`). The added constant enters the auxiliary design
//! as `m` itself, rescaled — the same column space as `[X, 1]`, but
//! orthogonal to `X` — so a design that comes close to spanning a constant
//! without passing the threshold (a column constant to within `1e-8`, say)
//! is tested, not refused as collinear.

use tsecon_hac::ols;
use tsecon_stats::chi2_sf;

use crate::common::{f_sf, has_constant, pow2_normalized, ssr, validate};
use crate::error::SpecTestError;

/// The statsmodels default order of the Breusch-Godfrey test,
/// `min(10, nobs / 5)` (integer division).
pub fn breusch_godfrey_default_nlags(nobs: usize) -> usize {
    (nobs / 5).min(10)
}

/// Largest auxiliary design, `T x k_aux` doubles, the test may build:
/// 2^26 entries, 512 MiB per copy (the OLS engine keeps a second one) —
/// the design budget of `ardl_bounds_test`. A larger one is refused before
/// any lag column is allocated, rather than left to abort the process when
/// an allocation fails.
pub const BG_MAX_AUX_CELLS: usize = 1 << 26;

/// Work budget of the auxiliary regression, in multiply-adds, priced as its
/// normal equations: forming them costs `T k_aux^2 / 2` and factoring and
/// inverting them about `6 k_aux^3` (see [`breusch_godfrey_work`]). Since
/// 0.11 the OLS engine solves by a blocked Householder QR instead, which
/// runs inside that price: measured at 0.26 to 1.47 ns per unit for `k_aux`
/// of 100 and more (`lab/serial_corr/bg_cost.py`), so `2^34` is at most
/// about half a minute (14.5 s at `T = 1e5` and 4.5 s at `T = 3000` at the
/// cap); a longer order is refused, naming `nlags`.
pub const BG_WORK_BUDGET: u128 = 1 << 34;

/// The work count [`BG_WORK_BUDGET`] is checked against for `n`
/// observations and `k_aux` auxiliary regressors:
/// `n k_aux^2 / 2 + 6 k_aux^3`. Saturating, so an absurd order cannot
/// overflow the check meant to refuse it.
pub fn breusch_godfrey_work(n: usize, k_aux: usize) -> u128 {
    let (n, k) = (n as u128, k_aux as u128);
    let k2 = k.saturating_mul(k);
    (n.saturating_mul(k2) / 2).saturating_add(k2.saturating_mul(k).saturating_mul(6))
}

/// Which budget an `n x k_aux` auxiliary design exceeds — the memory
/// budget ([`BG_MAX_AUX_CELLS`]), the work budget ([`BG_WORK_BUDGET`]) or
/// both — as the phrase its refusal quotes; `None` when it fits both.
pub fn breusch_godfrey_exceeded_budget(n: usize, k_aux: usize) -> Option<&'static str> {
    let memory = n.checked_mul(k_aux).is_none_or(|c| c > BG_MAX_AUX_CELLS);
    let work = breusch_godfrey_work(n, k_aux) > BG_WORK_BUDGET;
    match (memory, work) {
        (true, true) => Some(
            "both the memory budget of 2^26 = 67108864 entries (512 MiB) and the work \
             budget of 2^34 multiply-adds (about half a minute)",
        ),
        (true, false) => Some("the memory budget of 2^26 = 67108864 entries (512 MiB)"),
        (false, true) => Some("the work budget of 2^34 multiply-adds (about half a minute)"),
        (false, false) => None,
    }
}

/// `true` when an `n x k_aux` auxiliary design fits both budgets.
fn within_budget(n: usize, k_aux: usize) -> bool {
    breusch_godfrey_exceeded_budget(n, k_aux).is_none()
}

/// The largest order the test admits for `n` observations and `k_base`
/// auxiliary regressors besides the lags (the `k` columns of the design,
/// plus one when a constant is added): it keeps `n > k_base + nlags` and
/// the auxiliary design within both budgets. `0` when even one lag does not
/// fit.
pub fn breusch_godfrey_max_nlags(n: usize, k_base: usize) -> usize {
    // Both budgets grow with k_aux, so the admissible orders are 0..=max;
    // bisect on the largest k_aux below n that fits.
    let Some(mut hi) = n.checked_sub(1) else {
        return 0;
    };
    if hi <= k_base || !within_budget(n, k_base + 1) {
        return 0;
    }
    let mut lo = k_base + 1; // fits
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if within_budget(n, mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo - k_base
}

/// Outcome of the Breusch-Godfrey serial-correlation test.
#[derive(Debug, Clone, PartialEq)]
pub struct BreuschGodfreyTest {
    /// The LM statistic `T * R^2_aux`.
    pub statistic: f64,
    /// Chi-square degrees of freedom of [`Self::statistic`] (`= nlags`).
    pub df: usize,
    /// Upper-tail p-value `P(chi2(df) > statistic)`.
    pub pvalue: f64,
    /// The F-form: the lag coefficients jointly zero in the auxiliary
    /// regression (statsmodels' `fval`).
    pub fstat: f64,
    /// Numerator degrees of freedom of [`Self::fstat`] (`= nlags`).
    pub f_df_num: usize,
    /// Denominator degrees of freedom of [`Self::fstat`], `T - k_aux`.
    pub f_df_den: usize,
    /// Upper-tail p-value of [`Self::fstat`] under `F(f_df_num, f_df_den)`.
    pub f_pvalue: f64,
    /// The order `p` used (the caller's, or the default `min(10, T / 5)`).
    pub nlags: usize,
    /// Number of observations `T` (all of them: the lags are zero-padded).
    pub nobs: usize,
    /// `true` when the design did not span a constant and a column of ones
    /// was added to the auxiliary regression.
    pub constant_added: bool,
}

/// The part of the constant vector the design does not span, `m = M_X 1`,
/// or `None` when the columns span the constant: an explicit constant
/// column, or an implicit one (the regression of a column of ones on the
/// design leaves a residual of relative size below `1e-10` — e.g. a set of
/// complementary dummies).
fn constant_complement(x_cols: &[Vec<f64>], n: usize) -> Result<Option<Vec<f64>>, SpecTestError> {
    if has_constant(x_cols) {
        return Ok(None);
    }
    let fit = ols(&vec![1.0; n], x_cols)?;
    if ssr(&fit) <= 1e-20 * n as f64 {
        return Ok(None);
    }
    Ok(Some(fit.residuals))
}

/// Breusch-Godfrey LM test for serial correlation up to order `nlags` in the
/// errors of the OLS regression of `y` on `x_cols` (statsmodels exog
/// convention: include the constant column yourself; a design without one
/// gets a constant in the auxiliary regression, see the module docs).
/// `nlags = None` uses [`breusch_godfrey_default_nlags`].
///
/// # Errors
///
/// [`SpecTestError::DegreesOfFreedom`] if `T <= k` (the original
/// regression) or `T <= k_aux` (the auxiliary regression — the order is too
/// long for the sample); [`SpecTestError::AuxiliaryTooLarge`] if the
/// auxiliary design would exceed [`BG_MAX_AUX_CELLS`] entries or its
/// regression [`BG_WORK_BUDGET`] (checked before any lag column is built);
/// [`SpecTestError::InvalidLags`] if the order is zero
/// (including a default of zero for `T < 5`); [`SpecTestError::DegenerateResponse`]
/// if the residuals are (numerically) all zero — `y` is an exact linear
/// combination of the design — or, for a design without a constant, fully
/// explained by the added constant (`y` is an exact linear combination of
/// the design and a constant); the usual empty/dimension/finite validation
/// errors; and [`SpecTestError::SingularDesign`] if the design or the
/// auxiliary design is collinear.
pub fn breusch_godfrey_test(
    y: &[f64],
    x_cols: &[Vec<f64>],
    nlags: Option<usize>,
) -> Result<BreuschGodfreyTest, SpecTestError> {
    let (n, k) = validate(y, x_cols)?;
    if n <= k {
        return Err(SpecTestError::DegreesOfFreedom {
            what: "Breusch-Godfrey original regression",
            n,
            k,
        });
    }
    let p = nlags.unwrap_or_else(|| breusch_godfrey_default_nlags(n));
    if p == 0 {
        return Err(SpecTestError::InvalidLags {
            nlags: 0,
            n,
            defaulted: nlags.is_none(),
        });
    }
    // Both forms are invariant to rescaling y or a design column; the exact
    // power-of-two normalization keeps the sums of squares of extreme-scale
    // data from overflowing or underflowing.
    let y = pow2_normalized(y);
    let x_cols: Vec<Vec<f64>> = x_cols.iter().map(|c| pow2_normalized(c)).collect();
    let m1 = constant_complement(&x_cols, n)?;
    let spans = m1.is_none();
    let added = usize::from(!spans);
    // k_aux = k + p (+ 1); checked so an absurd order cannot overflow, and
    // compared with n and with the memory and work budgets BEFORE any lag
    // column is allocated.
    let k_aux = k
        .checked_add(p)
        .and_then(|v| v.checked_add(added))
        .unwrap_or(usize::MAX);
    if n <= k_aux {
        return Err(SpecTestError::DegreesOfFreedom {
            what: "Breusch-Godfrey auxiliary regression (design plus nlags lagged residuals)",
            n,
            k: k_aux,
        });
    }
    if !within_budget(n, k_aux) {
        return Err(SpecTestError::AuxiliaryTooLarge {
            n,
            k_aux,
            nlags: p,
            max_nlags: breusch_godfrey_max_nlags(n, k + added),
            defaulted: nlags.is_none(),
        });
    }

    let fit = ols(&y, &x_cols)?;
    let u = fit.residuals;
    let u_sq: f64 = u.iter().map(|v| v * v).sum();
    let y_sq: f64 = y.iter().map(|v| v * v).sum();
    if u_sq == 0.0 || u_sq <= 1e-26 * y_sq {
        return Err(SpecTestError::DegenerateResponse {
            what: "Breusch-Godfrey test (the OLS residuals are zero: y is an exact linear \
                   combination of the design columns)",
        });
    }
    let mean = u.iter().sum::<f64>() / n as f64;
    let tss: f64 = u.iter().map(|v| (v - mean) * (v - mean)).sum();

    // Restricted SSR: u on [X, (1)]. u is orthogonal to X, so with a constant
    // in the design it is sum u^2; with an added constant it is the squared
    // norm of r = u - (m'u / m'm) m, m = M_X 1. r is formed elementwise:
    // the closed form u'u - (m'u)^2 / m'm cancels catastrophically when the
    // added constant explains nearly all of u (a large omitted intercept).
    let ssr_r = match &m1 {
        None => u_sq,
        Some(m) => {
            let mu: f64 = m.iter().zip(&u).map(|(a, b)| a * b).sum();
            let mm: f64 = m.iter().map(|a| a * a).sum();
            let c = mu / mm;
            u.iter()
                .zip(m)
                .map(|(a, b)| (a - c * b) * (a - c * b))
                .sum()
        }
    };
    if tss <= 1e-24 * u_sq || ssr_r <= 1e-24 * u_sq {
        return Err(SpecTestError::DegenerateResponse {
            what: "Breusch-Godfrey test (x spans no constant and the constant added to the \
                   auxiliary regression explains the OLS residuals exactly: y is an exact \
                   linear combination of the design columns and a constant)",
        });
    }

    // Auxiliary design: [X, (1), u_{-1}, ..., u_{-p}], zero-padded lags.
    // The added constant enters as m = M_X 1 (rescaled by a power of two),
    // which spans the same columns as [X, 1] — m = 1 - X b — but is
    // orthogonal to X. A literal column of ones would make the normal
    // equations as ill-conditioned as x is close to spanning a constant:
    // with a column that is constant to within 1e-8 relative (m'm / n of
    // order 1e-16) the solver's pivot test refuses [X, 1] as collinear,
    // although the regression is well determined and the 1e-10 rule of
    // `constant_complement` has just said that x spans no constant.
    let mut aux: Vec<Vec<f64>> = x_cols;
    aux.reserve_exact(k_aux - k);
    if let Some(m) = &m1 {
        aux.push(pow2_normalized(m));
    }
    for j in 1..=p {
        let mut col = vec![0.0; n];
        col[j..].copy_from_slice(&u[..n - j]);
        aux.push(col);
    }
    let aux_fit = ols(&u, &aux).map_err(|e| match SpecTestError::from(e) {
        SpecTestError::SingularDesign { .. } => SpecTestError::SingularDesign {
            what: "Breusch-Godfrey auxiliary regression (design plus nlags lagged residuals)",
        },
        other => other,
    })?;
    let ssr_u = ssr(&aux_fit);

    let r2 = 1.0 - ssr_u / tss;
    let statistic = n as f64 * r2;
    let pvalue = chi2_sf(statistic, p as f64)?;

    let f_df_den = n - k_aux;
    let fstat = ((ssr_r - ssr_u) / p as f64) / (ssr_u / f_df_den as f64);
    let f_pvalue = f_sf(fstat, p as f64, f_df_den as f64)?;

    Ok(BreuschGodfreyTest {
        statistic,
        df: p,
        pvalue,
        fstat,
        f_df_num: p,
        f_df_den,
        f_pvalue,
        nlags: p,
        nobs: n,
        constant_added: !spans,
    })
}
