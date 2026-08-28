//! Johansen maximum-likelihood estimation of the vector error-correction
//! model at a fixed cointegration rank, and the mapping back to the level
//! VAR companion form.
//!
//! The model is
//!
//! ```text
//! Delta y_t = alpha [beta' eta'] [y_{t-1}; d^{ci/li}_{t-1}]
//!           + sum_{i=1}^{k_ar_diff} Gamma_i Delta y_{t-i} + C d_t + u_t,
//! ```
//!
//! with `beta` (`k x r`) the cointegrating vectors, `alpha` (`k x r`) the
//! error-correction loadings, `Gamma_i` the short-run dynamics, `eta`
//! (`n_coint x r`, returned as [`VecmResult::det_coef_coint`]) the
//! coefficients of any deterministic terms *restricted to the
//! cointegration relation*, and `C d_t` the unrestricted deterministic
//! terms of the short-run equations ([`VecmResult::det_coef`]). Which
//! terms appear where is chosen by [`VecmDeterministic`], named after the
//! statsmodels `VECM(deterministic = ...)` strings it reproduces:
//!
//! | statsmodels | variant | where the term lives |
//! |-------------|---------|----------------------|
//! | `"n"`  | [`VecmDeterministic::None`] | nothing |
//! | `"co"` | [`VecmDeterministic::Constant`] | constant in the short-run equations |
//! | `"ci"` | [`VecmDeterministic::RestrictedConstant`] | constant inside the cointegration relation |
//! | `"lo"` | [`VecmDeterministic::Trend`] | linear trend in the short-run equations |
//! | `"li"` | [`VecmDeterministic::RestrictedTrend`] | linear trend inside the cointegration relation |
//! | `"colo"` | [`VecmDeterministic::ConstantTrend`] | constant + trend, both unrestricted |
//! | `"coli"` | [`VecmDeterministic::ConstantRestrictedTrend`] | unrestricted constant, trend inside |
//! | `"cilo"` | [`VecmDeterministic::RestrictedConstantTrend`] | constant inside, unrestricted trend |
//! | `"cili"` | [`VecmDeterministic::RestrictedConstantRestrictedTrend`] | constant + trend, both inside |
//!
//! Centered seasonal dummies (statsmodels `seasons=` / `first_season=`)
//! can be added to the short-run equations through
//! [`fit_vecm_seasonal`]; they are always unrestricted.
//!
//! The reduced-rank maximum-likelihood estimator (Johansen 1988;
//! Lütkepohl 2005, section 7.2) partials the lagged differences and the
//! *unrestricted* deterministic terms out of `Delta y_t` and the
//! lagged-levels block, solves the canonical-correlation eigenproblem
//! [`crate::linalg::reduced_rank_eig`], takes the eigenvectors of the `r`
//! largest eigenvalues, and recovers `alpha`, `Gamma`, the deterministic
//! coefficients, and the residual covariance by least squares. In the
//! restricted cases (`ci`/`li`) the deterministic term is *appended to
//! the lagged-levels block*, so the eigenvectors — the cointegrating
//! vectors — gain one row per restricted term: the reduced-rank step
//! estimates the widened matrix `[beta; eta]` (`(k + n_coint) x r`),
//! normalized so its leading `r x r` block is the identity exactly as
//! statsmodels does, and the result splits it into [`VecmResult::beta`]
//! (the `k` variable rows) and [`VecmResult::det_coef_coint`] (the
//! deterministic rows: constant first, then trend — statsmodels
//! `VECMResults.beta` / `det_coef_coint`).
//!
//! The deterministic cases answer *different models*: on drifting data
//! the no-deterministic fit must absorb the drift and the mean of the
//! equilibrium error into `alpha beta' y_{t-1}`, which rotates `beta`
//! away from the constant-adjusted cointegrating space the Johansen rank
//! test (`det_order = 0`) works in. Fit with
//! [`VecmDeterministic::Constant`] when the rank came from
//! [`crate::johansen`].
//!
//! The golden fixtures `fixtures/coint.json` (`vecm_rank1` block,
//! `deterministic = "n"`) and `fixtures/vecm_deterministic.json` (every
//! deterministic case, on drifting and on trending data, plus two
//! seasonal fits) arbitrate `alpha`, `beta`, `det_coef_coint`, `gamma`,
//! `det_coef`, and the log-likelihood against statsmodels.

use tsecon_linalg::companion_from_var;
use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::CointError;
use crate::linalg::{
    check_finite, inv_general, inv_spd, ln_det_spd, partial_out, reduced_rank_eig,
};

/// The deterministic-term specification of a VECM fit.
///
/// Named after the statsmodels `VECM(..., deterministic = ...)` string it
/// reproduces (see [`Self::code`] / [`Self::from_code`]). "Restricted"
/// means *inside the cointegration relation* (Johansen's terminology):
/// the term is appended to the lagged-levels block of the reduced-rank
/// regression, its coefficient becomes extra rows of the cointegrating
/// matrix ([`VecmResult::det_coef_coint`]), and the short-run equations
/// get no separate copy of it. Unrestricted terms live in the short-run
/// equations and land in [`VecmResult::det_coef`]. statsmodels forbids
/// the same term on both sides (`"co"`+`"ci"`, `"lo"`+`"li"`), so the
/// nine variants here are exactly its nine accepted cases.
///
/// The Johansen (1995) five cases: I = `"n"`, II = `"ci"`, III = `"co"`,
/// IV = `"coli"`, V = `"colo"`. statsmodels `coint_johansen`'s
/// `det_order` corresponds as -1 ↔ `"n"`, 0 ↔ `"co"`, 1 ↔ `"colo"`;
/// [`crate::johansen`] implements `det_order = 0`, so a rank taken from
/// it matches [`VecmDeterministic::Constant`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VecmDeterministic {
    /// No deterministic terms — statsmodels `deterministic = "n"`. The
    /// historical (and current) default of [`fit_vecm`].
    #[default]
    None,
    /// An unrestricted constant outside the cointegration relation —
    /// statsmodels `deterministic = "co"`. The short-run equations gain
    /// an intercept (returned in [`VecmResult::det_coef`]), and the
    /// reduced-rank step partials the constant out alongside the lagged
    /// differences, which makes the estimated cointegrating space match
    /// the one [`crate::johansen`] (`det_order = 0`) tests.
    Constant,
    /// A constant *inside* the cointegration relation — statsmodels
    /// `"ci"` (Johansen's restricted constant, case II). The equilibrium
    /// error `beta' y + eta` has a freely estimated mean, but the data
    /// have no drift: the short-run equations get **no** separate
    /// intercept. The constant's coefficient per relation is the single
    /// row of [`VecmResult::det_coef_coint`].
    RestrictedConstant,
    /// An unrestricted linear trend in the short-run equations —
    /// statsmodels `"lo"` (trend column `t = p + 1, p + 2, ...` in
    /// statsmodels' indexing). Usually combined with a constant
    /// ([`VecmDeterministic::ConstantTrend`]).
    Trend,
    /// A linear trend *inside* the cointegration relation — statsmodels
    /// `"li"` (trend column `t = p, p + 1, ...` attached to
    /// `y_{t-1}`'s block): the equilibrium relation is trend-stationary
    /// rather than mean-zero. Its coefficient per relation is a row of
    /// [`VecmResult::det_coef_coint`].
    RestrictedTrend,
    /// Unrestricted constant + unrestricted trend — statsmodels
    /// `"colo"` (Johansen case V; the case `coint_johansen`'s
    /// `det_order = 1` convention corresponds to).
    ConstantTrend,
    /// Unrestricted constant + trend restricted to the cointegration
    /// relation — statsmodels `"coli"` (Johansen case IV: trending data
    /// whose equilibrium relation is trend-stationary).
    ConstantRestrictedTrend,
    /// Restricted constant + unrestricted trend — statsmodels `"cilo"`.
    RestrictedConstantTrend,
    /// Constant and trend both restricted to the cointegration relation
    /// — statsmodels `"cili"`: no drift in the differences, equilibrium
    /// error stationary around a constant plus a trend.
    RestrictedConstantRestrictedTrend,
}

impl VecmDeterministic {
    /// Parses the statsmodels `deterministic` string (`"n"`, `"co"`,
    /// `"ci"`, `"lo"`, `"li"`, `"colo"`, `"coli"`, `"cilo"`, `"cili"`).
    /// Returns `None` for anything else — including the statsmodels
    /// conflicts `"co"`+`"ci"` and `"lo"`+`"li"`, which are not
    /// representable here.
    pub fn from_code(code: &str) -> Option<Self> {
        Some(match code {
            "n" => Self::None,
            "co" => Self::Constant,
            "ci" => Self::RestrictedConstant,
            "lo" => Self::Trend,
            "li" => Self::RestrictedTrend,
            "colo" => Self::ConstantTrend,
            "coli" => Self::ConstantRestrictedTrend,
            "cilo" => Self::RestrictedConstantTrend,
            "cili" => Self::RestrictedConstantRestrictedTrend,
            _ => return None,
        })
    }

    /// The statsmodels `deterministic` string for this case.
    pub fn code(self) -> &'static str {
        match self {
            Self::None => "n",
            Self::Constant => "co",
            Self::RestrictedConstant => "ci",
            Self::Trend => "lo",
            Self::RestrictedTrend => "li",
            Self::ConstantTrend => "colo",
            Self::ConstantRestrictedTrend => "coli",
            Self::RestrictedConstantTrend => "cilo",
            Self::RestrictedConstantRestrictedTrend => "cili",
        }
    }

    /// `(const_inside, const_outside, trend_inside, trend_outside)`.
    fn flags(self) -> (bool, bool, bool, bool) {
        match self {
            Self::None => (false, false, false, false),
            Self::Constant => (false, true, false, false),
            Self::RestrictedConstant => (true, false, false, false),
            Self::Trend => (false, false, false, true),
            Self::RestrictedTrend => (false, false, true, false),
            Self::ConstantTrend => (false, true, false, true),
            Self::ConstantRestrictedTrend => (false, true, true, false),
            Self::RestrictedConstantTrend => (true, false, false, true),
            Self::RestrictedConstantRestrictedTrend => (true, false, true, false),
        }
    }
}

/// Result of a rank-`r` Johansen maximum-likelihood VECM fit.
///
/// Estimator conventions match statsmodels 0.14.6 `VECM(..., coint_rank =
/// r, deterministic = d, seasons = s).fit()` exactly, for every `d`
/// ([`VecmDeterministic`]): `alpha`/`beta`/`gamma`/`sigma_u`/`llf` are
/// `VECMResults`' attributes of the same names, and the deterministic
/// coefficients follow statsmodels' split — `det_coef_coint` for terms
/// inside the cointegration relation, `det_coef` for terms in the
/// short-run equations.
#[derive(Debug, Clone)]
pub struct VecmResult {
    /// Number of series `k`.
    pub neqs: usize,
    /// Effective sample size `T` (rows after `p = k_ar_diff + 1`
    /// presample rows).
    pub nobs: usize,
    /// Number of lagged differences `k_ar_diff = p - 1`.
    pub k_ar_diff: usize,
    /// Cointegration rank `r`.
    pub coint_rank: usize,
    /// The deterministic-term specification the model was fit under.
    pub deterministic: VecmDeterministic,
    /// Number of periods in a seasonal cycle (`0` = no seasonal
    /// dummies). When nonzero, `seasons - 1` centered seasonal-dummy
    /// columns sit in [`Self::det_coef`].
    pub seasons: usize,
    /// The season of the first observation (`0`-based), as statsmodels'
    /// `first_season`. Meaningful only when `seasons > 0`.
    pub first_season: usize,
    /// Error-correction loadings `alpha` (`k x r`).
    pub alpha: Mat<f64>,
    /// Cointegrating vectors `beta` (`k x r`) — the *variable* rows of
    /// the reduced-rank estimate, normalized so that the widened matrix
    /// `[beta; det_coef_coint]` has the identity as its leading `r x r`
    /// block (statsmodels' normalization; the leading block lies in
    /// `beta` because `r <= k`). The rows of any deterministic terms
    /// restricted to the cointegration relation are split into
    /// [`Self::det_coef_coint`], exactly as statsmodels'
    /// `VECMResults.beta` / `det_coef_coint`.
    pub beta: Mat<f64>,
    /// Coefficients of the deterministic terms *inside* the
    /// cointegration relation (statsmodels `det_coef_coint`):
    /// `n_coint x r`, one row per restricted term in this order —
    /// constant (`"ci"`) first, then linear trend (`"li"`). `0 x r`
    /// when no term is restricted. Column `j` extends cointegrating
    /// vector `j`: the equilibrium error is
    /// `beta[:, j]' y_t + det_coef_coint[:, j]' [1; t]`.
    pub det_coef_coint: Mat<f64>,
    /// Short-run coefficients `Gamma = [Gamma_1, ..., Gamma_{k_ar_diff}]`
    /// stacked horizontally (`k x k*k_ar_diff`); `gamma[(eq, i*k + var)]`
    /// is the effect of `Delta` variable `var` at lag `i + 1` on equation
    /// `eq`.
    pub gamma: Mat<f64>,
    /// Coefficients of the deterministic terms outside the cointegration
    /// relation (statsmodels `det_coef`): `k x n_det`, one column per
    /// unrestricted term in statsmodels' order — constant (`"co"`)
    /// first, then the `seasons - 1` centered seasonal dummies, then the
    /// linear trend (`"lo"`). `k x 0` when the short-run equations have
    /// no deterministic terms.
    pub det_coef: Mat<f64>,
    /// Maximum-likelihood residual covariance `U'U / T` (`k x k`).
    pub sigma_u: Mat<f64>,
    /// The Johansen eigenvalues from the canonical-correlation problem,
    /// decreasing. There are `k + n_coint` of them (the lagged-levels
    /// block widens by one row per deterministic term restricted to the
    /// cointegration relation); at most `k` are nonzero.
    pub eig: Vec<f64>,
    /// Gaussian log-likelihood at the maximum (Lütkepohl 2005, eq. 7.2.20).
    pub llf: f64,
}

impl VecmResult {
    /// The long-run impact matrix `Pi = alpha beta'` (`k x k`) — the
    /// variable part only; any restricted deterministic term contributes
    /// `alpha det_coef_coint'` to the intercept/trend of the level VAR,
    /// not to `Pi`.
    pub fn pi(&self) -> Mat<f64> {
        &self.alpha * self.beta.transpose()
    }

    /// The short-run matrix `Gamma_i` (`k x k`), for `i = 1 ..= k_ar_diff`.
    ///
    /// # Errors
    ///
    /// [`CointError::InvalidArgument`] if `i` is `0` or exceeds
    /// `k_ar_diff`.
    pub fn gamma_lag(&self, i: usize) -> Result<Mat<f64>, CointError> {
        if i == 0 || i > self.k_ar_diff {
            return Err(CointError::InvalidArgument {
                what: "gamma_lag index must satisfy 1 <= i <= k_ar_diff",
            });
        }
        let k = self.neqs;
        let base = (i - 1) * k;
        Ok(Mat::from_fn(k, k, |r, c| self.gamma[(r, base + c)]))
    }

    /// The coefficient matrices `[A_1, ..., A_p]` (`p = k_ar_diff + 1`) of
    /// the equivalent level VAR `y_t = sum_j A_j y_{t-j} + u_t`.
    ///
    /// The mapping is (Lütkepohl 2005, eq. 6.3.2, inverted)
    ///
    /// ```text
    /// A_1 = I + Pi + Gamma_1
    /// A_i = Gamma_i - Gamma_{i-1}          (2 <= i <= k_ar_diff)
    /// A_p = -Gamma_{k_ar_diff}
    /// ```
    ///
    /// with the obvious degeneracies when `k_ar_diff = 0` (`A_1 = I + Pi`).
    /// This is the utility the impulse-response layer consumes: feed the
    /// returned matrices to [`companion_from_var`] or to the VAR analysis
    /// crate. Only the autoregressive part is returned — the
    /// deterministic terms carry over to the level VAR without touching
    /// the `A_j`: an unrestricted term keeps its short-run coefficient
    /// ([`Self::det_coef`]), and a restricted term contributes
    /// `alpha * det_coef_coint'` applied to `[1; t-1]` (the constant
    /// and/or lagged trend inside the error-correction term).
    pub fn var_coefs(&self) -> Vec<Mat<f64>> {
        let k = self.neqs;
        let p = self.k_ar_diff + 1;
        let pi = self.pi();
        let ident = Mat::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 });
        // Gamma_i, with Gamma_0 and Gamma_{k_ar_diff+1} treated as zero.
        let gamma_block = |i: usize| -> Mat<f64> {
            if i == 0 || i > self.k_ar_diff {
                Mat::<f64>::zeros(k, k)
            } else {
                let base = (i - 1) * k;
                Mat::from_fn(k, k, |r, c| self.gamma[(r, base + c)])
            }
        };
        let mut coefs = Vec::with_capacity(p);
        for j in 1..=p {
            let a = if j == 1 {
                &(&ident + &pi) + &gamma_block(1)
            } else {
                &gamma_block(j) - &gamma_block(j - 1)
            };
            coefs.push(a);
        }
        coefs
    }

    /// The `kp x kp` companion matrix of the equivalent level VAR
    /// (Lütkepohl 2005, eq. 2.1.8), for downstream stability checks and
    /// impulse responses.
    ///
    /// # Errors
    ///
    /// [`CointError::Linalg`] if the companion assembly rejects the
    /// coefficient matrices (never on a well-formed fit).
    pub fn companion(&self) -> Result<Mat<f64>, CointError> {
        let coefs = self.var_coefs();
        let refs: Vec<MatRef<'_, f64>> = coefs.iter().map(Mat::as_ref).collect();
        Ok(companion_from_var(&refs)?)
    }
}

/// Estimates the VECM at cointegration rank `coint_rank` by Johansen
/// maximum likelihood, on `endog` (a `T x k` matrix, oldest row first)
/// with `k_ar_diff` lagged differences and **no deterministic terms**
/// (statsmodels `deterministic = "n"`).
///
/// This is [`fit_vecm_det`] with [`VecmDeterministic::None`], kept as the
/// historical default. Note the Johansen rank test ([`crate::johansen`])
/// assumes an unrestricted constant instead — to estimate the same model
/// the test ranks, call [`fit_vecm_det`] with
/// [`VecmDeterministic::Constant`].
///
/// # Errors
///
/// As [`fit_vecm_det`].
pub fn fit_vecm(
    endog: MatRef<'_, f64>,
    k_ar_diff: usize,
    coint_rank: usize,
) -> Result<VecmResult, CointError> {
    fit_vecm_det(endog, k_ar_diff, coint_rank, VecmDeterministic::None)
}

/// Estimates the VECM at cointegration rank `coint_rank` by Johansen
/// maximum likelihood, on `endog` (a `T x k` matrix, oldest row first)
/// with `k_ar_diff` lagged differences and the deterministic terms chosen
/// by `deterministic`.
///
/// This is [`fit_vecm_seasonal`] without seasonal dummies.
///
/// # Errors
///
/// As [`fit_vecm_seasonal`].
pub fn fit_vecm_det(
    endog: MatRef<'_, f64>,
    k_ar_diff: usize,
    coint_rank: usize,
    deterministic: VecmDeterministic,
) -> Result<VecmResult, CointError> {
    fit_vecm_seasonal(endog, k_ar_diff, coint_rank, deterministic, 0, 0)
}

/// Estimates the VECM at cointegration rank `coint_rank` by Johansen
/// maximum likelihood, on `endog` (a `T x k` matrix, oldest row first)
/// with `k_ar_diff` lagged differences, the deterministic terms chosen by
/// `deterministic`, and — when `seasons > 0` — `seasons - 1` centered
/// seasonal dummies in the short-run equations (statsmodels `seasons=` /
/// `first_season=`; `first_season` is the 0-based season of the first
/// row of `endog`). Centered dummies sum to zero over a full cycle, so
/// they shift the seasonal profile without moving the level — they are
/// orthogonal to a constant and combine with every deterministic case.
///
/// # Errors
///
/// * [`CointError::Dimension`] if `endog` has no columns;
/// * [`CointError::InvalidRank`] if `coint_rank` is outside `0 ..= k`;
/// * [`CointError::InvalidArgument`] if `seasons == 1` (a one-period
///   "cycle" has no seasonal dummies; pass `0` for none);
/// * [`CointError::NonFinite`] if `endog` contains a NaN or infinity;
/// * [`CointError::InsufficientObservations`] if the effective sample is
///   too small;
/// * [`CointError::NotPositiveDefinite`] / [`CointError::Singular`] /
///   [`CointError::Linalg`] on a degenerate design or a failed
///   factorization.
pub fn fit_vecm_seasonal(
    endog: MatRef<'_, f64>,
    k_ar_diff: usize,
    coint_rank: usize,
    deterministic: VecmDeterministic,
    seasons: usize,
    first_season: usize,
) -> Result<VecmResult, CointError> {
    let k = endog.ncols();
    if k == 0 {
        return Err(CointError::Dimension {
            what: "the data matrix has no columns; pass a 2-D array shaped \
                   (n_obs, n_series) with observations in rows, oldest first",
            expected: 1,
            got: 0,
        });
    }
    if coint_rank > k {
        return Err(CointError::InvalidRank {
            rank: coint_rank,
            neqs: k,
        });
    }
    if seasons == 1 {
        return Err(CointError::InvalidArgument {
            what: "seasons = 1 is a one-period \"cycle\" with no seasonal dummies \
                   (statsmodels builds seasons - 1 = 0 columns); pass seasons = 0 \
                   for no seasonal terms, or the true cycle length (4 = quarterly, \
                   12 = monthly)",
        });
    }
    check_finite(endog, "the data matrix")?;
    let n = endog.nrows();
    let p = k_ar_diff + 1;
    if n <= p {
        return Err(CointError::InsufficientObservations {
            needed: k * k_ar_diff + k + 1,
            got: 0,
            nobs: n,
            neqs: k,
            k_ar_diff,
        });
    }
    let t = n - p;
    let n_short = k * k_ar_diff;
    let (ci, co, li, lo) = deterministic.flags();
    let n_seas = seasons.saturating_sub(1);
    // Unrestricted deterministic columns of the short-run regressor
    // block, statsmodels' delta_x order: constant, seasonal dummies,
    // trend.
    let n_det = co as usize + n_seas + lo as usize;
    // Restricted terms appended to the lagged-levels block: constant,
    // then trend.
    let n_coint = ci as usize + li as usize;
    let n_reg = n_short + n_det;
    let kw = k + n_coint; // width of the (possibly widened) levels block
    if t <= n_reg + kw {
        return Err(CointError::InsufficientObservations {
            needed: n_reg + kw + 1,
            got: t,
            nobs: n,
            neqs: k,
            k_ar_diff,
        });
    }

    // Sample matrices (statsmodels _endog_matrices), in T x (.) layout.
    // Effective row i corresponds to level index p + i. The lagged-levels
    // block appends the restricted deterministic terms after the k
    // levels: a column of ones for "ci", then the trend t = p, p+1, ...
    // for "li" (statsmodels _linear_trend(T, p, coint=True)).
    let delta_y0 = Mat::from_fn(t, k, |i, j| endog[(p + i, j)] - endog[(p + i - 1, j)]);
    let y_lag1 = Mat::from_fn(t, kw, |i, j| {
        if j < k {
            endog[(p + i - 1, j)]
        } else if j == k && ci {
            1.0
        } else {
            (i + p) as f64 // the restricted trend
        }
    });
    // The short-run regressor block stacks the lagged differences first
    // and then the unrestricted deterministic terms — a column of ones
    // for "co", the seasons - 1 centered seasonal dummies, and the trend
    // t = p+1, p+2, ... for "lo" (statsmodels _linear_trend(T, p)) —
    // exactly as statsmodels stacks delta_x.
    let first_period = first_season + p; // statsmodels: first_season + diff_lags + 1
    let delta_x = Mat::from_fn(t, n_reg, |i, col| {
        if col < n_short {
            let lag = col / k + 1; // 1 ..= k_ar_diff
            let var = col % k;
            return endog[(p + i - lag, var)] - endog[(p + i - lag - 1, var)];
        }
        let mut d = col - n_short;
        if co {
            if d == 0 {
                return 1.0; // the unrestricted constant
            }
            d -= 1;
        }
        if d < n_seas {
            // Centered seasonal dummy d (statsmodels seasonal_dummies
            // with centered=True): 1 when effective row i falls in
            // season d, minus 1/seasons everywhere.
            let hit = (i + first_period) % seasons == d;
            return if hit { 1.0 } else { 0.0 } - 1.0 / seasons as f64;
        }
        (i + p + 1) as f64 // the unrestricted trend
    });

    // Auxiliary-regression residuals.
    let r0 = partial_out(delta_y0.as_ref(), delta_x.as_ref());
    let r1 = partial_out(y_lag1.as_ref(), delta_x.as_ref());

    let tf = t as f64;
    let s00 = Mat::from_fn(k, k, |i, j| dot_cols(r0.as_ref(), r0.as_ref(), i, j) / tf);
    let s01 = Mat::from_fn(k, kw, |i, j| dot_cols(r0.as_ref(), r1.as_ref(), i, j) / tf);
    let s11 = Mat::from_fn(kw, kw, |i, j| dot_cols(r1.as_ref(), r1.as_ref(), i, j) / tf);

    let (eig, evec) = reduced_rank_eig(s00.as_ref(), s01.as_ref(), s11.as_ref())?;

    let r = coint_rank;
    // The widened cointegrating matrix [beta; det_coef_coint]: the r
    // eigenvectors of the largest eigenvalues, normalized so that the
    // leading r x r block is the identity (statsmodels normalization;
    // the block lies in the variable rows because r <= k).
    let mut beta_full = Mat::from_fn(kw, r, |i, j| evec[(i, j)]);
    if r > 0 {
        let top = Mat::from_fn(r, r, |i, j| beta_full[(i, j)]);
        let top_inv = inv_general(
            top.as_ref(),
            "beta[:r, :r], the block used to normalize the cointegrating vectors; \
             coint_rank exceeds the number of independent cointegrating relations \
             in this sample — lower coint_rank (johansen() reports the rank it \
             selects at 5%)",
        )?;
        beta_full = &beta_full * &top_inv;
    }

    // alpha = S_01 beta (beta' S_11 beta)^{-1}, with the widened beta.
    let alpha = if r == 0 {
        Mat::<f64>::zeros(k, 0)
    } else {
        let bsb = beta_full.transpose() * &s11 * &beta_full;
        let bsb_inv = inv_general(
            bsb.as_ref(),
            "beta' S_11 beta, the cointegrating-space second moment; two series are \
             collinear, so the cointegrating space is not identified — drop the \
             redundant series",
        )?;
        &s01 * &beta_full * &bsb_inv
    };

    // Pi = alpha [beta; eta]'; Gamma (and the unrestricted deterministic
    // coefficients) from regressing the error-corrected differences on
    // the short-run block.
    let pi = &alpha * beta_full.transpose();
    // W = Delta y0 - Y_lag1 Pi'  (T x k), with the widened levels block.
    let w = &delta_y0 - &y_lag1 * pi.transpose();
    let coef = if n_reg == 0 {
        Mat::<f64>::zeros(k, 0)
    } else {
        let dxtdx = delta_x.transpose() * &delta_x;
        let dxtdx_inv = inv_spd(
            dxtdx.as_ref(),
            "Delta X' Delta X, the short-run regressor cross-product",
        )?;
        // coef = W' Delta X (Delta X' Delta X)^{-1}  (k x n_reg).
        &(w.transpose() * &delta_x) * &dxtdx_inv
    };
    // Split as statsmodels does: lagged-difference columns first, then
    // the deterministic terms.
    let gamma = Mat::from_fn(k, n_short, |i, j| coef[(i, j)]);
    let det_coef = Mat::from_fn(k, n_det, |i, j| coef[(i, n_short + j)]);
    // And split the widened cointegrating matrix into its variable and
    // deterministic rows (statsmodels np.vsplit(beta, [neqs])).
    let beta = Mat::from_fn(k, r, |i, j| beta_full[(i, j)]);
    let det_coef_coint = Mat::from_fn(n_coint, r, |i, j| beta_full[(k + i, j)]);

    // Full residuals and ML covariance.
    let resid = if n_reg == 0 {
        w.clone()
    } else {
        &w - &delta_x * coef.transpose()
    };
    let sigma_u = Mat::from_fn(k, k, |i, j| {
        dot_cols(resid.as_ref(), resid.as_ref(), i, j) / tf
    });

    // Concentrated log-likelihood (Lütkepohl 2005, eq. 7.2.20;
    // statsmodels VECMResults.llf):
    // llf = -kT/2 ln(2pi) - T/2 (ln|S_00| + sum_{i<r} ln(1 - lambda_i)) - kT/2.
    let ln_det_s00 = ln_det_spd(
        s00.as_ref(),
        "S_00, the second-moment matrix of the differenced residuals",
    )?;
    let mut sum_ln = 0.0;
    for &lam in eig.iter().take(r) {
        sum_ln += (1.0 - lam).ln();
    }
    let kf = k as f64;
    let llf = -kf * tf / 2.0 * core::f64::consts::TAU.ln()
        - tf / 2.0 * (ln_det_s00 + sum_ln)
        - kf * tf / 2.0;

    Ok(VecmResult {
        neqs: k,
        nobs: t,
        k_ar_diff,
        coint_rank: r,
        deterministic,
        seasons,
        first_season,
        alpha,
        beta,
        det_coef_coint,
        gamma,
        det_coef,
        sigma_u,
        eig,
        llf,
    })
}

/// Inner product of column `a` of `x` with column `b` of `y`.
fn dot_cols(x: MatRef<'_, f64>, y: MatRef<'_, f64>, a: usize, b: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..x.nrows() {
        s += x[(i, a)] * y[(i, b)];
    }
    s
}
