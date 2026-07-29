//! Preprocessing advisors: how many differences a series needs
//! ([`ndiffs`]) and which variance-stabilising Box-Cox power to use
//! ([`box_cox_lambda`]).
//!
//! Both answer a question a first-time user asks *before* fitting
//! anything, and both return the evidence behind the answer, not just the
//! answer: `ndiffs` reports the test statistic and p-value at every
//! differencing order it tried, and `box_cox_lambda` reports the objective
//! at the optimum (plus, for the MLE, the log-likelihood of the two
//! transforms people actually reach for — the log and no transform at
//! all).
//!
//! * [`ndiffs`] composes the shipped [`crate::kpss`] / [`crate::adf`] /
//!   [`crate::phillips_perron`] tests with the standard sequential rule
//!   (Hyndman & Khandakar 2008, JSS 27(3), sec. 3.2, as implemented by
//!   `forecast::ndiffs`): difference until the test stops calling for a
//!   difference, capped at `max_d`. No new statistics are introduced here.
//! * [`box_cox_lambda`] selects the Box-Cox power either by profile
//!   maximum likelihood (Box & Cox 1964 — the `scipy.stats.boxcox_normmax`
//!   objective, matched to `1e-15` relative) or by Guerrero's (1993)
//!   grouped coefficient-of-variation criterion (the `forecast` package
//!   default).
//!
//! The Box-Cox family needs strictly positive data; a non-positive
//! observation is an error that names the offending index rather than a
//! silent NaN.

use core::fmt;

use crate::error::DiagError;
use crate::phillips::{phillips_perron, PpTestType};
use crate::report::check_alpha;
use crate::unitroot::{adf, kpss, AdfLagSelection, AdfRegression, KpssLags, KpssRegression};
use crate::validate::check_series;

// ------------------------------------------------------------------ errors

/// Errors produced by the preprocessing advisors.
///
/// Everything the underlying diagnostics can raise arrives as
/// [`AdvisorError::Diag`] and prints exactly as it would from the test
/// itself; the remaining variants cover the advisors' own preconditions.
#[derive(Debug, Clone, PartialEq)]
pub enum AdvisorError {
    /// An error from the underlying diagnostic (the unit-root tests
    /// composed by [`ndiffs`], or the shared input validation).
    Diag(DiagError),
    /// The Box-Cox family is only defined for strictly positive data.
    NonPositive {
        /// Which advisor rejected the series.
        what: &'static str,
        /// Index of the first non-positive observation.
        index: usize,
        /// The offending value.
        value: f64,
        /// How many observations are non-positive in total.
        count: usize,
    },
    /// The `bounds` of the lambda search are not a valid interval.
    InvalidBounds {
        /// The supplied lower bound.
        lower: f64,
        /// The supplied upper bound.
        upper: f64,
    },
    /// The Guerrero grouping length is unusable for a series this long.
    InvalidPeriod {
        /// The supplied grouping length.
        period: usize,
        /// The number of observations supplied.
        n: usize,
    },
    /// A non-finite lambda was passed to an objective function.
    InvalidLambda {
        /// The offending value.
        value: f64,
    },
}

impl fmt::Display for AdvisorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdvisorError::Diag(e) => write!(f, "{e}"),
            AdvisorError::NonPositive {
                what,
                index,
                value,
                count,
            } => write!(
                f,
                "{what}: the Box-Cox family needs strictly positive data, but \
                 y[{index}] = {value} ({count} of the observations are <= 0); \
                 log(y) is undefined there, so no lambda exists. Fix this \
                 explicitly rather than silently: add a constant shift larger \
                 than the most negative value (and undo it after \
                 back-transforming), model log1p(y) if the series is a \
                 non-negative count, or use a transform defined on the whole \
                 line (the Yeo-Johnson family)"
            ),
            AdvisorError::InvalidBounds { lower, upper } => write!(
                f,
                "box_cox_lambda: bounds = ({lower}, {upper}) is not a valid \
                 search interval: requires finite bounds with lower < upper \
                 (the conventional choice is (-2, 2); R's forecast package \
                 searches (-1, 2))"
            ),
            AdvisorError::InvalidPeriod { period, n } => write!(
                f,
                "box_cox_lambda: the Guerrero criterion groups the series into \
                 blocks of {period} observations and compares their \
                 level-adjusted spreads, so it needs period >= 2 and at least \
                 two complete blocks (n >= 2 * period); got n = {n}. For \
                 seasonal data set period to the seasonal frequency (12 for \
                 monthly, 4 for quarterly); the non-seasonal default is 2"
            ),
            AdvisorError::InvalidLambda { value } => write!(
                f,
                "box-cox objective: lambda = {value} is not finite; the \
                 transform is only defined for finite powers"
            ),
        }
    }
}

impl std::error::Error for AdvisorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AdvisorError::Diag(e) => Some(e),
            _ => None,
        }
    }
}

impl From<DiagError> for AdvisorError {
    fn from(e: DiagError) -> Self {
        AdvisorError::Diag(e)
    }
}

// ------------------------------------------------------- numeric helpers

/// Neumaier (1974) compensated summation: the running compensation keeps
/// the golden comparisons against NumPy's pairwise sums at the 1e-15 level
/// instead of the 1e-13 a naive loop would give on long series.
fn ksum<I: IntoIterator<Item = f64>>(it: I) -> f64 {
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for x in it {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            comp += (sum - t) + x;
        } else {
            comp += (x - t) + sum;
        }
        sum = t;
    }
    sum + comp
}

/// Mean and *population* (`ddof = 0`) variance, two-pass.
fn mean_var_pop(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = ksum(x.iter().copied()) / n;
    let var = ksum(x.iter().map(|&v| (v - mean) * (v - mean))) / n;
    (mean, var)
}

/// Mean and *sample* (`ddof = 1`) standard deviation, two-pass — R's `sd`
/// and NumPy's `std(ddof=1)`, which is what the Guerrero criterion uses.
fn mean_sd_sample(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = ksum(x.iter().copied()) / n;
    let ss = ksum(x.iter().map(|&v| (v - mean) * (v - mean)));
    (mean, (ss / (n - 1.0)).sqrt())
}

/// Brent's (1973) derivative-free minimisation on a closed interval —
/// golden-section search with parabolic interpolation, the algorithm
/// behind `scipy.optimize.minimize_scalar(method="bounded")`.
///
/// The interior search is followed by an explicit check of both endpoints:
/// a constrained optimum can *sit* on a bound, which an interior search
/// approaches but never reaches, and reporting the bound exactly is what
/// lets [`BoxCoxLambda::at_bound`] be honest. A coarse pre-scan brackets
/// the best region first, so a criterion with a shallow secondary dip (the
/// Guerrero objective is not guaranteed unimodal) does not strand the
/// search in it.
fn argmin_bounded<F: FnMut(f64) -> f64>(mut f: F, lo: f64, hi: f64) -> (f64, f64) {
    const XATOL: f64 = 1e-10;
    const MAXITER: usize = 500;
    const PRESCAN: usize = 64;

    // Coarse pre-scan; keep the grid triple bracketing the best point.
    let mut best_i = 0usize;
    let mut best_f = f64::INFINITY;
    let grid: Vec<f64> = (0..=PRESCAN)
        .map(|i| lo + (hi - lo) * (i as f64) / (PRESCAN as f64))
        .collect();
    for (i, &g) in grid.iter().enumerate() {
        let v = f(g);
        if v < best_f {
            best_f = v;
            best_i = i;
        }
    }
    let (mut a, mut b) = (
        grid[best_i.saturating_sub(1)],
        grid[(best_i + 1).min(PRESCAN)],
    );

    let sqrt_eps = f64::EPSILON.sqrt();
    let golden = 0.5 * (3.0 - 5.0f64.sqrt());
    let mut fulc = a + golden * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let (mut rat, mut e) = (0.0f64, 0.0f64);
    let mut fx = f(xf);
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + XATOL / 3.0;
    let mut tol2 = 2.0 * tol1;

    let mut iter = 0usize;
    while (xf - xm).abs() > tol2 - 0.5 * (b - a) {
        let mut use_golden = true;
        if e.abs() > tol1 {
            // Fit a parabola through (fulc, nfc, xf) and accept its vertex
            // only if it falls inside the bracket and improves fast enough.
            let r0 = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r0;
            q = 2.0 * (q - r0);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let r_prev = e;
            e = rat;
            if p.abs() < (0.5 * q * r_prev).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                let x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    rat = if xm - xf >= 0.0 { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }
        if use_golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden * e;
        }
        let step = if rat >= 0.0 {
            rat.abs().max(tol1)
        } else {
            -rat.abs().max(tol1)
        };
        let x = xf + step;
        let fu = f(x);

        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + XATOL / 3.0;
        tol2 = 2.0 * tol1;
        iter += 1;
        if iter >= MAXITER {
            break;
        }
    }

    let mut best = (xf, fx);
    for edge in [lo, hi] {
        let fe = f(edge);
        if fe < best.1 {
            best = (edge, fe);
        }
    }
    best
}

// ------------------------------------------------------------------ ndiffs

/// Which unit-root/stationarity test drives the [`ndiffs`] sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdiffsTest {
    /// KPSS with a constant and the automatic (Hobijn-Franses-Ooms)
    /// bandwidth — the `forecast::ndiffs` default. Null: stationarity, so
    /// the rule differences while the test *rejects* (`p < alpha`).
    Kpss,
    /// ADF with a constant and AIC lag selection. Null: a unit root, so
    /// the rule differences while the test *fails to reject*
    /// (`p > alpha`).
    Adf,
    /// Phillips-Perron `Z-tau` with a constant and the default Bartlett
    /// bandwidth. Null: a unit root, same orientation as the ADF.
    Pp,
}

impl NdiffsTest {
    /// The short code used in reports (`"kpss"`, `"adf"`, `"pp"`).
    pub fn code(self) -> &'static str {
        match self {
            NdiffsTest::Kpss => "kpss",
            NdiffsTest::Adf => "adf",
            NdiffsTest::Pp => "pp",
        }
    }

    /// Whether the test's null hypothesis is stationarity (KPSS) rather
    /// than a unit root (ADF, PP) — the sign of the decision rule.
    fn null_is_stationarity(self) -> bool {
        matches!(self, NdiffsTest::Kpss)
    }
}

/// The evidence at one differencing order: what the test saw and what it
/// concluded, so the returned `d` can be audited rather than trusted.
#[derive(Debug, Clone, PartialEq)]
pub struct NdiffsStep {
    /// The differencing order this evidence refers to (0 = levels).
    pub d: usize,
    /// Number of observations at this order (`n - d`).
    pub n: usize,
    /// The test statistic.
    pub statistic: f64,
    /// The test's p-value. For KPSS this is table-bounded to
    /// `[0.01, 0.10]`, so a reported `0.01` means "at most 1%".
    pub p_value: f64,
    /// The lag order (ADF) or Bartlett bandwidth (KPSS, PP) the test used.
    pub lags: usize,
    /// Whether this evidence calls for another difference.
    pub needs_differencing: bool,
}

/// Why the [`ndiffs`] sequence stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdiffsStop {
    /// The test stopped calling for a difference — the intended exit.
    Stationary,
    /// The `max_d` cap was hit while the test still called for a
    /// difference. The answer is a floor, not a verdict.
    MaxD,
    /// Differencing left an exactly constant series, so no test is defined
    /// (and none is needed): the trend was deterministic, not stochastic.
    Constant,
}

/// Result of the differencing advisor: the recommended `d`, the evidence
/// at every order that was tried, and why the sequence stopped.
#[derive(Debug, Clone, PartialEq)]
pub struct NdiffsResult {
    /// The recommended number of first differences.
    pub d: usize,
    /// Which test produced the evidence.
    pub test: NdiffsTest,
    /// The significance level every decision was taken at.
    pub alpha: f64,
    /// The cap the search was given.
    pub max_d: usize,
    /// Why the sequence stopped.
    pub stop: NdiffsStop,
    /// One entry per differencing order actually tested, in order
    /// (`steps[k].d == k`). Empty only for an exactly constant input.
    pub steps: Vec<NdiffsStep>,
    /// Plain-language interpretation, including the over-differencing
    /// warning when it applies.
    pub interpretation: String,
}

impl fmt::Display for NdiffsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ndiffs({}) = {} [", self.test.code(), self.d)?;
        for (i, s) in self.steps.iter().enumerate() {
            if i > 0 {
                write!(f, "; ")?;
            }
            write!(
                f,
                "d = {}: stat = {:.4}, p = {:.4}",
                s.d, s.statistic, s.p_value
            )?;
        }
        write!(f, "] — {}", self.interpretation)
    }
}

/// Run one test on the current (differenced) series and package the
/// evidence, including the direction of the decision rule.
fn ndiffs_step(y: &[f64], test: NdiffsTest, alpha: f64, d: usize) -> Result<NdiffsStep, DiagError> {
    let (statistic, p_value, lags) = match test {
        NdiffsTest::Kpss => {
            let r = kpss(y, KpssRegression::Constant, KpssLags::Auto)?;
            (r.statistic, r.p_value, r.lags)
        }
        NdiffsTest::Adf => {
            let r = adf(y, AdfRegression::Constant, AdfLagSelection::Aic(None))?;
            (r.statistic, r.p_value, r.used_lag)
        }
        NdiffsTest::Pp => {
            let r = phillips_perron(y, AdfRegression::Constant, PpTestType::Tau, None)?;
            (r.stat, r.p_value, r.lags)
        }
    };
    // KPSS (H0 stationary): difference while the null is rejected.
    // ADF / PP (H0 unit root): difference unless the null is rejected.
    let needs_differencing = if test.null_is_stationarity() {
        p_value < alpha
    } else {
        p_value > alpha
    };
    Ok(NdiffsStep {
        d,
        n: y.len(),
        statistic,
        p_value,
        lags,
        needs_differencing,
    })
}

fn ndiffs_interpretation(
    d: usize,
    test: NdiffsTest,
    alpha: f64,
    max_d: usize,
    stop: NdiffsStop,
    steps: &[NdiffsStep],
) -> String {
    let pct = alpha * 100.0;
    let code = test.code();
    let last = steps.last();
    let evidence = match last {
        Some(s) => format!("stat = {:.4}, p = {:.4}", s.statistic, s.p_value),
        None => "no test was run".to_string(),
    };
    match stop {
        NdiffsStop::Constant if steps.is_empty() => "The series is exactly constant, so d = 0: \
             there is no variation to make stationary and no unit-root test \
             is defined. Check that the intended column was passed."
            .to_string(),
        NdiffsStop::Constant => format!(
            "d = {d}: differencing {d} time(s) leaves an exactly constant \
             series, so the trend was deterministic (a polynomial in time), \
             not stochastic. Differencing does remove it, but detrending — \
             regressing on a time trend, or fitting the trend explicitly — \
             keeps the interpretation and the standard errors honest. Do not \
             difference further."
        ),
        NdiffsStop::MaxD => format!(
            "d = {d}, the max_d cap — the {code} test still calls for another \
             difference at order {d} ({evidence}) at the {pct:.0}% level, so \
             this is a floor rather than a verdict. Before raising max_d, ask \
             whether the rejection really is a unit root: a structural break, \
             a deterministic trend, or long memory produces the same reading, \
             and check_stationarity's ADF + KPSS quadrant separates those \
             cases. Series genuinely needing d > 2 are rare in economics."
        ),
        NdiffsStop::Stationary if d == 0 => format!(
            "d = 0: at the {pct:.0}% level the {code} test does not call for \
             differencing in levels ({evidence}). Model the series as it is. \
             Differencing an already-stationary series is not a free \
             precaution — it injects a non-invertible MA(1) unit root, \
             inflates the variance of the residuals, and loses an \
             observation."
        ),
        NdiffsStop::Stationary if d == 1 => format!(
            "d = 1: at the {pct:.0}% level the {code} test calls for \
             differencing in levels but not after one difference \
             ({evidence}). Work with the first difference — for a price or \
             index series that is the (log) change, which is also the \
             quantity people interpret. In ARIMA terms this is the `d` of \
             ARIMA(p, 1, q); a regression of one I(1) level on another is \
             spurious unless you are explicitly testing cointegration."
        ),
        NdiffsStop::Stationary => format!(
            "d = {d}: at the {pct:.0}% level the {code} test only stops \
             calling for a difference after {d} of them ({evidence}). \
             Second differences are unusual outside strongly trending \
             nominal series — confirm with the ACF of the differenced series \
             (a large negative spike at lag 1 is the signature of \
             over-differencing) and consider whether a deterministic trend \
             or a level shift is doing the work instead. Cap reached: \
             max_d = {max_d}."
        ),
    }
}

/// How many first differences a series needs to look stationary — with the
/// test evidence at every order tried.
///
/// The sequential rule is the standard one (Hyndman & Khandakar 2008, JSS
/// 27(3), sec. 3.2, as implemented by `forecast::ndiffs`): test the series,
/// difference if the test calls for it, repeat, stop at `max_d`. Because
/// the KPSS null is stationarity and the ADF/PP null is a unit root, the
/// rule flips accordingly — KPSS differences while `p < alpha`, ADF and PP
/// difference while `p > alpha`. An exactly constant series (which
/// differencing a deterministic polynomial trend produces) stops the
/// sequence: no test is defined there.
///
/// This is composition, not new statistics: the per-order numbers are
/// exactly what [`crate::kpss`], [`crate::adf`] and
/// [`crate::phillips_perron`] return at their conventional defaults
/// (constant term; automatic bandwidth / AIC lag selection), so they are
/// pinned to statsmodels and `arch` by those tests' own goldens.
///
/// # Errors
///
/// * [`AdvisorError::Diag`] wrapping [`DiagError::InvalidAlpha`] unless
///   `0 < alpha < 1`.
/// * [`AdvisorError::Diag`] for anything the underlying test raises — most
///   often [`DiagError::SeriesTooShort`], since each difference costs an
///   observation.
pub fn ndiffs(
    y: &[f64],
    test: NdiffsTest,
    alpha: f64,
    max_d: usize,
) -> Result<NdiffsResult, AdvisorError> {
    check_alpha(alpha)?;
    check_series(y, 4, "ndiffs")?;

    let mut current: Vec<f64> = y.to_vec();
    let mut steps: Vec<NdiffsStep> = Vec::new();
    let mut d = 0usize;
    let stop = loop {
        if current.iter().all(|&v| v == current[0]) {
            break NdiffsStop::Constant;
        }
        let step = ndiffs_step(&current, test, alpha, d)?;
        let more = step.needs_differencing;
        steps.push(step);
        if !more {
            break NdiffsStop::Stationary;
        }
        if d >= max_d {
            break NdiffsStop::MaxD;
        }
        d += 1;
        current = current.windows(2).map(|w| w[1] - w[0]).collect();
    };

    Ok(NdiffsResult {
        d,
        test,
        alpha,
        max_d,
        stop,
        interpretation: ndiffs_interpretation(d, test, alpha, max_d, stop, &steps),
        steps,
    })
}

// ---------------------------------------------------------------- box-cox

/// How the Box-Cox power is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxCoxMethod {
    /// Profile maximum likelihood (Box & Cox 1964): maximise
    /// [`box_cox_llf`]. This is `scipy.stats.boxcox_normmax(method="mle")`
    /// and R's `BoxCox.lambda(method = "loglik")`.
    Mle,
    /// Guerrero's (1993) grouped coefficient-of-variation criterion, the
    /// `forecast::BoxCox.lambda` default: minimise [`guerrero_cv`].
    Guerrero {
        /// Length of the consecutive blocks the series is grouped into —
        /// the seasonal frequency for seasonal data (12 monthly, 4
        /// quarterly), 2 for non-seasonal data (the `forecast` default).
        period: usize,
    },
}

impl BoxCoxMethod {
    /// The short code used in reports (`"mle"`, `"guerrero"`).
    pub fn code(self) -> &'static str {
        match self {
            BoxCoxMethod::Mle => "mle",
            BoxCoxMethod::Guerrero { .. } => "guerrero",
        }
    }
}

/// Result of the Box-Cox lambda search: the power, the objective it
/// optimised, and enough context to argue with it.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxCoxLambda {
    /// The selected power.
    pub lambda: f64,
    /// The objective at `lambda`: the profile log-likelihood (maximised)
    /// for [`BoxCoxMethod::Mle`], the coefficient of variation of the
    /// grouped ratios (minimised) for [`BoxCoxMethod::Guerrero`].
    pub objective: f64,
    /// The method used.
    pub method: BoxCoxMethod,
    /// Lower end of the search interval.
    pub lower: f64,
    /// Upper end of the search interval.
    pub upper: f64,
    /// Whether the optimum sits on one of the bounds — the criterion was
    /// still improving there, so `lambda` is the constrained answer.
    pub at_bound: bool,
    /// Number of observations.
    pub n: usize,
    /// MLE only: the profile log-likelihood of the log transform
    /// (`lambda = 0`).
    pub loglik_at_zero: Option<f64>,
    /// MLE only: the profile log-likelihood of no transform at all
    /// (`lambda = 1`).
    pub loglik_at_one: Option<f64>,
    /// MLE only: the likelihood-ratio statistic `2 (l(lambda) - l(0))`
    /// against the log transform, asymptotically chi-squared with 1 df, so
    /// values below 3.84 mean the log is not rejected at 5%.
    pub lr_vs_zero: Option<f64>,
    /// MLE only: the same statistic against no transform (`lambda = 1`).
    pub lr_vs_one: Option<f64>,
    /// Plain-language interpretation with the practical rounding advice.
    pub interpretation: String,
}

impl fmt::Display for BoxCoxLambda {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "box_cox_lambda({}): lambda = {:.4} on [{}, {}], objective = \
             {:.6}{} — {}",
            self.method.code(),
            self.lambda,
            self.lower,
            self.upper,
            self.objective,
            if self.at_bound { " (at bound)" } else { "" },
            self.interpretation
        )
    }
}

/// Validate a series for the Box-Cox family: finite, long enough, and
/// strictly positive. Returns the length.
fn check_positive(y: &[f64], min_n: usize, what: &'static str) -> Result<usize, AdvisorError> {
    let n = check_series(y, min_n, what)?;
    let mut first: Option<(usize, f64)> = None;
    let mut count = 0usize;
    for (index, &value) in y.iter().enumerate() {
        // NaN cannot reach here (`check_series` rejects it first), but the
        // explicit test keeps the predicate total.
        if value <= 0.0 || value.is_nan() {
            count += 1;
            if first.is_none() {
                first = Some((index, value));
            }
        }
    }
    if let Some((index, value)) = first {
        return Err(AdvisorError::NonPositive {
            what,
            index,
            value,
            count,
        });
    }
    Ok(n)
}

/// The profile log-likelihood core, given the pre-computed logs.
///
/// ```text
/// l(lambda) = (lambda - 1) sum_i log x_i - (n/2) log var(w(lambda))
/// w_i(lambda) = x_i^lambda / lambda        (lambda != 0)
///             = log x_i                    (lambda  = 0)
/// ```
///
/// with `var` the population (`ddof = 0`) variance — the
/// `scipy.stats.boxcox_llf` convention, including its trick of dropping
/// the `-1/lambda` offset (which cancels in a variance but costs
/// precision).
///
/// The `lambda != 0` branch centres the exponent at its mean and uses
/// `expm1`, so `w_i - mean(w)` is formed without the catastrophic
/// cancellation that `exp` suffers as `lambda -> 0`; the identity
/// `var(c w) = c^2 var(w)` puts the scale back in log space. When the
/// centred exponent would overflow (a log-range wider than ~700, which
/// needs absurd data), it falls back to shifting by the maximum.
fn llf_core(logx: &[f64], sum_logx: f64, lambda: f64) -> f64 {
    let n = logx.len() as f64;
    let log_var = if lambda == 0.0 {
        let (_, var) = mean_var_pop(logx);
        var.ln()
    } else {
        let lg: Vec<f64> = logx.iter().map(|&v| lambda * v).collect();
        let lgbar = ksum(lg.iter().copied()) / n;
        let lgmax = lg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if lgmax - lgbar > 700.0 {
            let w: Vec<f64> = lg.iter().map(|&v| (v - lgmax).exp()).collect();
            let (_, var) = mean_var_pop(&w);
            var.ln() + 2.0 * lgmax - 2.0 * lambda.abs().ln()
        } else {
            let u: Vec<f64> = lg.iter().map(|&v| (v - lgbar).exp_m1()).collect();
            let (_, var) = mean_var_pop(&u);
            var.ln() + 2.0 * lgbar - 2.0 * lambda.abs().ln()
        }
    };
    (lambda - 1.0) * sum_logx - 0.5 * n * log_var
}

/// The Box-Cox profile log-likelihood at a given lambda (Box & Cox 1964).
///
/// Matches `scipy.stats.boxcox_llf(lmb, y)` to `1e-15` relative on the
/// golden grid, and is *more* accurate than SciPy's log-space
/// implementation for `|lambda|` below about `1e-6`, where SciPy's
/// `logsumexp` path loses digits and this one degrades smoothly into the
/// `lambda = 0` limit.
///
/// # Errors
///
/// * [`AdvisorError::NonPositive`] if any observation is `<= 0`, naming
///   the first offender.
/// * [`AdvisorError::Diag`] for non-finite input, fewer than two
///   observations, or a constant series (whose transformed variance is
///   zero, so the likelihood is unbounded).
/// * [`AdvisorError::InvalidLambda`] for a non-finite `lambda`.
pub fn box_cox_llf(y: &[f64], lambda: f64) -> Result<f64, AdvisorError> {
    if !lambda.is_finite() {
        return Err(AdvisorError::InvalidLambda { value: lambda });
    }
    check_positive(y, 2, "box_cox_llf")?;
    if y.iter().all(|&v| v == y[0]) {
        return Err(DiagError::ConstantSeries {
            what: "box_cox_llf",
        }
        .into());
    }
    let logx: Vec<f64> = y.iter().map(|&v| v.ln()).collect();
    let sum_logx = ksum(logx.iter().copied());
    Ok(llf_core(&logx, sum_logx, lambda))
}

/// Per-group means and sample standard deviations for the Guerrero
/// criterion: the last `floor(n / period) * period` observations split
/// into consecutive blocks of `period` (the leading partial block is
/// dropped, as in `forecast::BoxCox.lambda`).
fn guerrero_groups(y: &[f64], period: usize) -> Result<(Vec<f64>, Vec<f64>), AdvisorError> {
    let n = y.len();
    if period < 2 || n / period < 2 {
        return Err(AdvisorError::InvalidPeriod { period, n });
    }
    let ngroups = n / period;
    let start = n - ngroups * period;
    let mut mu = Vec::with_capacity(ngroups);
    let mut sd = Vec::with_capacity(ngroups);
    for k in 0..ngroups {
        let block = &y[start + k * period..start + (k + 1) * period];
        let (m, s) = mean_sd_sample(block);
        mu.push(m);
        sd.push(s);
    }
    Ok((mu, sd))
}

/// The Guerrero objective given the pre-computed group moments.
fn guerrero_core(mu: &[f64], sd: &[f64], lambda: f64) -> f64 {
    let ratio: Vec<f64> = mu
        .iter()
        .zip(sd)
        .map(|(&m, &s)| s / m.powf(1.0 - lambda))
        .collect();
    let (mean, sdev) = mean_sd_sample(&ratio);
    sdev / mean
}

/// Guerrero's (1993) grouped coefficient-of-variation criterion at a given
/// lambda — the objective `forecast::BoxCox.lambda` minimises.
///
/// Split the last `floor(n / period) * period` observations into
/// consecutive groups of `period`. With `m_k` the mean and `s_k` the
/// sample standard deviation (`ddof = 1`) of group `k`,
///
/// ```text
/// r_k        = s_k / m_k^(1 - lambda)
/// CV(lambda) = sd(r) / mean(r)            (sd again ddof = 1)
/// ```
///
/// The lambda that minimises `CV` makes the within-group spread as nearly
/// proportional to a fixed power of the level as the data allow — i.e. it
/// stabilises the variance. Unlike the MLE this is not a likelihood: it
/// says nothing about normality, and its value is not comparable across
/// series or across `period`.
///
/// # Errors
///
/// * [`AdvisorError::NonPositive`] if any observation is `<= 0`.
/// * [`AdvisorError::InvalidPeriod`] if `period < 2` or fewer than two
///   complete groups fit in the sample.
/// * [`AdvisorError::InvalidLambda`] for a non-finite `lambda`.
/// * [`AdvisorError::Diag`] wrapping [`DiagError::NumericalBreakdown`] if
///   every group has zero spread (the ratios are all zero, so the
///   coefficient of variation is 0/0).
pub fn guerrero_cv(y: &[f64], lambda: f64, period: usize) -> Result<f64, AdvisorError> {
    if !lambda.is_finite() {
        return Err(AdvisorError::InvalidLambda { value: lambda });
    }
    check_positive(y, 4, "guerrero_cv")?;
    let (mu, sd) = guerrero_groups(y, period)?;
    let value = guerrero_core(&mu, &sd, lambda);
    if !value.is_finite() {
        return Err(DiagError::NumericalBreakdown {
            what: "guerrero_cv",
        }
        .into());
    }
    Ok(value)
}

fn box_cox_interpretation(res: &BoxCoxLambda) -> String {
    let lam = res.lambda;
    let near = |target: f64| (lam - target).abs() < 0.15;
    let shape = if near(0.0) {
        "essentially the log transform"
    } else if near(0.5) {
        "essentially the square-root transform"
    } else if near(1.0) {
        "essentially no transform"
    } else if near(-1.0) {
        "essentially the reciprocal transform"
    } else if lam < 0.0 {
        "a stronger-than-log compression of the upper tail"
    } else {
        "a mild compression of the upper tail"
    };
    let bound_note = if res.at_bound {
        format!(
            " The optimum sits on the bound {:.4}: the criterion was still \
             improving there, so this lambda is constrained, not chosen. A \
             lambda that extreme usually means the *level* is mis-specified \
             (an untreated trend or level shift) rather than that the data \
             need such a violent power.",
            lam
        )
    } else {
        String::new()
    };
    match res.method {
        BoxCoxMethod::Mle => {
            let lr0 = res.lr_vs_zero.unwrap_or(f64::NAN);
            let lr1 = res.lr_vs_one.unwrap_or(f64::NAN);
            let verdict = |lr: f64, name: &str| {
                if lr < 3.841_458_820_694_124 {
                    format!(
                        "{name} is NOT rejected against it (LR = {lr:.2} < 3.84, \
                         chi2(1) at 5%)"
                    )
                } else {
                    format!("{name} IS rejected against it (LR = {lr:.2} > 3.84)")
                }
            };
            format!(
                "Profile-likelihood lambda = {lam:.4} ({shape}), log-likelihood \
                 {:.4} on {} observations. Compared with the two transforms \
                 people actually use: the log (lambda = 0) — {}; no transform \
                 (lambda = 1) — {}. Report a round lambda when the likelihood \
                 allows it (1 = none, 0.5 = square root, 0 = log, -1 = \
                 reciprocal): an interpretable transform beats a third decimal \
                 place, and forecasts made on the transformed scale must be \
                 back-transformed with a bias adjustment or they return the \
                 median, not the mean.{bound_note}",
                res.objective,
                res.n,
                verdict(lr0, "the log"),
                verdict(lr1, "no transform"),
            )
        }
        BoxCoxMethod::Guerrero { period } => format!(
            "Guerrero lambda = {lam:.4} ({shape}), grouping the series into \
             blocks of {period} observations; the coefficient of variation of \
             the level-adjusted group spreads is {:.6} at the optimum (lower \
             is flatter — the criterion is not a likelihood, so its value is \
             not comparable across series). Set `period` to the seasonal \
             frequency for seasonal data (12 monthly, 4 quarterly); the \
             default of 2 is the non-seasonal convention and will give a \
             different answer. Round to an interpretable lambda when you can \
             (1 = none, 0.5 = square root, 0 = log), and remember the \
             back-transform needs a bias adjustment to return means rather \
             than medians.{bound_note}",
            res.objective
        ),
    }
}

/// Select a variance-stabilising Box-Cox power, with the objective at the
/// optimum.
///
/// The Box-Cox (1964) family
///
/// ```text
/// y(lambda) = (y^lambda - 1) / lambda   (lambda != 0)
///           = log y                     (lambda  = 0)
/// ```
///
/// nests no transform (`lambda = 1`), the square root (`0.5`), the log
/// (`0`) and the reciprocal (`-1`). [`BoxCoxMethod::Mle`] maximises the
/// profile log-likelihood ([`box_cox_llf`]); [`BoxCoxMethod::Guerrero`]
/// minimises the grouped coefficient of variation ([`guerrero_cv`]).
///
/// `bounds` are *hard* bounds, and an optimum on a bound is reported as
/// such via [`BoxCoxLambda::at_bound`] — unlike
/// `scipy.stats.boxcox_normmax`, whose `brack` argument is only a starting
/// bracket for an unbounded search and can return a lambda far outside it.
///
/// # Errors
///
/// * [`AdvisorError::NonPositive`] if any observation is `<= 0`, naming
///   the first offender — the transform is undefined there.
/// * [`AdvisorError::InvalidBounds`] if `bounds` is not a finite interval
///   with `lower < upper`.
/// * [`AdvisorError::InvalidPeriod`] (Guerrero only) if `period < 2` or
///   fewer than two complete groups fit in the sample.
/// * [`AdvisorError::Diag`] for non-finite input, too few observations, or
///   a constant series.
pub fn box_cox_lambda(
    y: &[f64],
    method: BoxCoxMethod,
    bounds: (f64, f64),
) -> Result<BoxCoxLambda, AdvisorError> {
    let (lower, upper) = bounds;
    if !(lower.is_finite() && upper.is_finite() && lower < upper) {
        return Err(AdvisorError::InvalidBounds { lower, upper });
    }
    let n = check_positive(y, 4, "box_cox_lambda")?;
    if y.iter().all(|&v| v == y[0]) {
        return Err(DiagError::ConstantSeries {
            what: "box_cox_lambda",
        }
        .into());
    }

    let mut res = match method {
        BoxCoxMethod::Mle => {
            let logx: Vec<f64> = y.iter().map(|&v| v.ln()).collect();
            let sum_logx = ksum(logx.iter().copied());
            let (lambda, neg_llf) = argmin_bounded(|l| -llf_core(&logx, sum_logx, l), lower, upper);
            let loglik = -neg_llf;
            let l0 = llf_core(&logx, sum_logx, 0.0);
            let l1 = llf_core(&logx, sum_logx, 1.0);
            BoxCoxLambda {
                lambda,
                objective: loglik,
                method,
                lower,
                upper,
                at_bound: (lambda - lower).abs() < 1e-8 || (upper - lambda).abs() < 1e-8,
                n,
                loglik_at_zero: Some(l0),
                loglik_at_one: Some(l1),
                lr_vs_zero: Some(2.0 * (loglik - l0)),
                lr_vs_one: Some(2.0 * (loglik - l1)),
                interpretation: String::new(),
            }
        }
        BoxCoxMethod::Guerrero { period } => {
            let (mu, sd) = guerrero_groups(y, period)?;
            let (lambda, cv) = argmin_bounded(|l| guerrero_core(&mu, &sd, l), lower, upper);
            if !cv.is_finite() {
                return Err(DiagError::NumericalBreakdown {
                    what: "box_cox_lambda",
                }
                .into());
            }
            BoxCoxLambda {
                lambda,
                objective: cv,
                method,
                lower,
                upper,
                at_bound: (lambda - lower).abs() < 1e-8 || (upper - lambda).abs() < 1e-8,
                n,
                loglik_at_zero: None,
                loglik_at_one: None,
                lr_vs_zero: None,
                lr_vs_one: None,
                interpretation: String::new(),
            }
        }
    };
    res.interpretation = box_cox_interpretation(&res);
    Ok(res)
}
