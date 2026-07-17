//! Estimation: exact Gaussian MLE on the state-space form and
//! conditional-sum-of-squares (CSS) as the fast alternative, with
//! Hannan-Rissanen starting values.

use tsecon_linalg::faer::Mat;
use tsecon_linalg::{jittered_cholesky, levinson_durbin_from_series};
use tsecon_optim::{
    minimize, Method, ObjectiveFn, OptimError, OptimizeResult, StationaryAr, Transform,
    TransformedObjective,
};

use crate::diff::difference;
use crate::error::ArimaError;
use crate::results::{ArimaResults, EstimationMethod};
use crate::spec::ArimaSpec;
use crate::ssm::arma_ssm;

/// Composite reparameterization for the packed ARMA parameter vector
/// `[const?, ar_1..ar_p, ma_1..ma_q, sigma2?]`:
///
/// * constant — identity (unconstrained);
/// * AR block — the Monahan (1984) PACF stationarity transform
///   ([`StationaryAr`]): stationary by construction for every `z`;
/// * MA block — invertibility by duality (Monahan 1984): `1 + theta_1 L +
///   ... + theta_q L^q` is invertible iff the AR polynomial with
///   coefficients `-theta_j` is stationary, so `theta = -forward_ar(z)`
///   and `z = inverse_ar(-theta)`;
/// * `sigma2` — `exp` (the [`Positive`](tsecon_optim::Positive) map),
///   present only when `sigma2` is estimated jointly (exact MLE); CSS
///   concentrates `sigma2` out and omits it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArimaTransform {
    /// Whether the vector starts with a constant.
    pub(crate) constant: bool,
    /// AR order.
    pub(crate) p: usize,
    /// MA order.
    pub(crate) q: usize,
    /// Whether the vector ends with `sigma2`.
    pub(crate) sigma2: bool,
}

impl ArimaTransform {
    /// Total packed length.
    fn len(&self) -> usize {
        usize::from(self.constant) + self.p + self.q + usize::from(self.sigma2)
    }

    fn check(&self, z_len: usize, theta_len: usize) -> Result<(), OptimError> {
        if z_len != self.len() {
            return Err(OptimError::DimensionMismatch {
                what: "ARIMA working vector z",
                expected: self.len(),
                actual: z_len,
            });
        }
        if theta_len != z_len {
            return Err(OptimError::DimensionMismatch {
                what: "ARIMA parameter vector theta",
                expected: z_len,
                actual: theta_len,
            });
        }
        Ok(())
    }
}

impl Transform for ArimaTransform {
    fn forward(&self, z: &[f64], theta: &mut [f64]) -> Result<(), OptimError> {
        self.check(z.len(), theta.len())?;
        if z.iter().any(|v| !v.is_finite()) {
            return Err(OptimError::NonFinite { what: "z" });
        }
        let c = usize::from(self.constant);
        if self.constant {
            theta[0] = z[0];
        }
        StationaryAr.forward(&z[c..c + self.p], &mut theta[c..c + self.p])?;
        let (ms, me) = (c + self.p, c + self.p + self.q);
        StationaryAr.forward(&z[ms..me], &mut theta[ms..me])?;
        for t in &mut theta[ms..me] {
            *t = -*t;
        }
        if self.sigma2 {
            theta[me] = z[me].exp();
        }
        Ok(())
    }

    fn inverse(&self, theta: &[f64], z: &mut [f64]) -> Result<(), OptimError> {
        self.check(z.len(), theta.len())?;
        if theta.iter().any(|v| !v.is_finite()) {
            return Err(OptimError::NonFinite { what: "theta" });
        }
        let c = usize::from(self.constant);
        if self.constant {
            z[0] = theta[0];
        }
        StationaryAr.inverse(&theta[c..c + self.p], &mut z[c..c + self.p])?;
        let (ms, me) = (c + self.p, c + self.p + self.q);
        let neg_ma: Vec<f64> = theta[ms..me].iter().map(|t| -t).collect();
        StationaryAr.inverse(&neg_ma, &mut z[ms..me])?;
        if self.sigma2 {
            if theta[me] <= 0.0 {
                return Err(OptimError::Domain {
                    name: "sigma2",
                    value: theta[me],
                    requirement: "sigma2 > 0",
                });
            }
            z[me] = theta[me].ln();
        }
        Ok(())
    }

    fn log_jacobian(&self, z: &[f64]) -> Result<f64, OptimError> {
        if z.len() != self.len() {
            return Err(OptimError::DimensionMismatch {
                what: "ARIMA working vector z",
                expected: self.len(),
                actual: z.len(),
            });
        }
        if z.iter().any(|v| !v.is_finite()) {
            return Err(OptimError::NonFinite { what: "z" });
        }
        let c = usize::from(self.constant);
        // Constant: identity, log|1| = 0. MA block: the sign flip has
        // |det| = 1, so the Monahan Jacobian applies unchanged.
        let mut lj = StationaryAr.log_jacobian(&z[c..c + self.p])?;
        lj += StationaryAr.log_jacobian(&z[c + self.p..c + self.p + self.q])?;
        if self.sigma2 {
            lj += z[self.len() - 1];
        }
        Ok(lj)
    }
}

/// Exact negative log-likelihood of the ARMA sample via the Kalman
/// filter's prediction-error decomposition; non-finite / infeasible
/// parameter points evaluate to `+infinity` per the [`ObjectiveFn`]
/// contract.
struct ExactNegLoglik {
    spec: ArimaSpec,
    /// The (differenced) observations as an `n x 1` matrix.
    y: Mat<f64>,
}

impl ObjectiveFn for ExactNegLoglik {
    fn value(&mut self, theta: &[f64]) -> f64 {
        let blocks = match self.spec.unpack(theta) {
            Ok(b) => b,
            Err(_) => return f64::INFINITY,
        };
        let model = match arma_ssm(blocks.ar, blocks.ma, blocks.sigma2, blocks.constant) {
            Ok(m) => m,
            Err(_) => return f64::INFINITY,
        };
        match model.loglike(self.y.as_ref()) {
            Ok(ll) if ll.is_finite() => -ll,
            _ => f64::INFINITY,
        }
    }
}

/// Conditional sum of squares of the recursive residuals
///
/// ```text
/// eps_t = x_t - c - sum_i phi_i x_{t-i} - sum_j theta_j eps_{t-j},
/// t = p..n-1,   with eps_s = 0 for s < p,
/// ```
///
/// i.e. conditioning on the first `p` observations and zero pre-sample
/// innovations (Box & Jenkins 1976, chapter 7). Returns `(ssr, n_c)` with
/// `n_c = n - p` effective observations, or `None` when the recursion
/// overflows (numerically explosive coefficients).
fn css_ssr(x: &[f64], constant: f64, ar: &[f64], ma: &[f64]) -> Option<(f64, usize)> {
    let n = x.len();
    let p = ar.len();
    if n <= p {
        return None;
    }
    let mut eps = vec![0.0; n];
    let mut ssr = 0.0;
    for t in p..n {
        let mut e = x[t] - constant;
        for (i, phi) in ar.iter().enumerate() {
            e -= phi * x[t - 1 - i];
        }
        for (j, theta) in ma.iter().enumerate() {
            if t > j {
                e -= theta * eps[t - 1 - j];
            }
        }
        if !e.is_finite() {
            return None;
        }
        eps[t] = e;
        ssr += e * e;
    }
    if !ssr.is_finite() {
        return None;
    }
    Some((ssr, n - p))
}

/// CSS objective: the `sigma2`-concentrated conditional negative
/// log-likelihood
///
/// ```text
/// -l(c, phi, theta) = n_c/2 [ln 2*pi + 1 + ln(SSR / n_c)]
/// ```
///
/// (the Gaussian log-likelihood of the recursive residuals with
/// `sigma2 = SSR / n_c` profiled out).
struct CssNegLoglik<'a> {
    constant: bool,
    p: usize,
    q: usize,
    x: &'a [f64],
}

impl ObjectiveFn for CssNegLoglik<'_> {
    fn value(&mut self, theta: &[f64]) -> f64 {
        if theta.len() != usize::from(self.constant) + self.p + self.q {
            return f64::INFINITY;
        }
        let c = usize::from(self.constant);
        let constant = if self.constant { theta[0] } else { 0.0 };
        let ar = &theta[c..c + self.p];
        let ma = &theta[c + self.p..c + self.p + self.q];
        match css_ssr(self.x, constant, ar, ma) {
            Some((ssr, n_c)) if ssr > 0.0 => {
                let n_c = n_c as f64;
                0.5 * n_c * ((2.0 * std::f64::consts::PI).ln() + 1.0 + (ssr / n_c).ln())
            }
            _ => f64::INFINITY,
        }
    }
}

/// Sample mean.
fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

/// Population (divide-by-n) variance, floored away from zero so it is
/// always a usable `sigma2` starting value.
fn variance_floored(x: &[f64]) -> f64 {
    let mu = mean(x);
    let v = x.iter().map(|xi| (xi - mu) * (xi - mu)).sum::<f64>() / (x.len().max(1)) as f64;
    let floor = 1e-8 * (1.0 + mu * mu);
    if v.is_finite() && v > floor {
        v
    } else {
        floor
    }
}

/// Solves `L L' s = b` given the lower-triangular Cholesky factor `L`
/// (forward then backward substitution; `L` has strictly positive
/// diagonal by construction).
fn chol_solve(l: &Mat<f64>, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut s = b.to_vec();
    for i in 0..n {
        let mut v = s[i];
        for j in 0..i {
            v -= l[(i, j)] * s[j];
        }
        s[i] = v / l[(i, i)];
    }
    for i in (0..n).rev() {
        let mut v = s[i];
        for j in i + 1..n {
            v -= l[(j, i)] * s[j];
        }
        s[i] = v / l[(i, i)];
    }
    s
}

/// Hannan-Rissanen (1982) two-stage starting values for ARMA(p, q):
///
/// 1. fit a long autoregression of order `h = max(p + q, ceil(10 log10
///    n))` to the demeaned series (via Yule-Walker / Levinson-Durbin
///    rather than OLS — the difference is `O(1/n)` and immaterial for
///    starting values) and form its residuals `e_t`;
/// 2. regress `x_t` on `[1?, x_{t-1..t-p}, e_{t-1..t-q}]` by OLS (normal
///    equations through the jitter-ladder Cholesky); the coefficients are
///    the starting `[const?, ar, ma]` and the residual variance the
///    starting `sigma2`.
///
/// Without a constant the regression uses the demeaned series (better
/// conditioned; only a starting value). Returns the packed vector
/// `[const?, ar, ma, sigma2]`; errors bubble up to the caller, which
/// falls back to [`default_start`].
fn hannan_rissanen(
    x: &[f64],
    p: usize,
    q: usize,
    include_constant: bool,
) -> Result<Vec<f64>, ArimaError> {
    let n = x.len();
    let c = usize::from(include_constant);
    if p + q == 0 {
        let mut out = Vec::with_capacity(c + 1);
        if include_constant {
            out.push(mean(x));
        }
        out.push(if include_constant {
            variance_floored(x)
        } else {
            // Without a constant the model variance is the raw second
            // moment.
            let m2 = x.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64;
            if m2.is_finite() && m2 > 1e-12 {
                m2
            } else {
                1e-8
            }
        });
        return Ok(out);
    }

    let h = ((10.0 * (n as f64).log10()).ceil() as usize)
        .max(p + q)
        .max(1)
        .min(n / 2);
    let t0 = h + q;
    let k_reg = c + p + q;
    if n < t0 + k_reg + 2 || h < p {
        return Err(ArimaError::InsufficientObservations {
            needed: t0 + k_reg + 2,
            got: n,
        });
    }

    let mu = mean(x);
    let xc: Vec<f64> = x.iter().map(|v| v - mu).collect();
    let ld = levinson_durbin_from_series(&xc, h)?;
    let a = ld.ar_coefs_final();

    // Long-AR residuals, defined for t >= h.
    let mut e = vec![0.0; n];
    for t in h..n {
        let mut r = xc[t];
        for (i, ai) in a.iter().enumerate() {
            r -= ai * xc[t - 1 - i];
        }
        e[t] = r;
    }

    // Second-stage regression target/lags: raw series with an intercept
    // regressor, or the demeaned series without one.
    let target: &[f64] = if include_constant { x } else { &xc };
    let mut xtx = Mat::<f64>::zeros(k_reg, k_reg);
    let mut xty = vec![0.0; k_reg];
    let mut row = vec![0.0; k_reg];
    for t in t0..n {
        let mut idx = 0;
        if include_constant {
            row[idx] = 1.0;
            idx += 1;
        }
        for i in 1..=p {
            row[idx] = target[t - i];
            idx += 1;
        }
        for j in 1..=q {
            row[idx] = e[t - j];
            idx += 1;
        }
        for (ai, &ri) in row.iter().enumerate() {
            xty[ai] += ri * target[t];
            for (bi, &rj) in row.iter().enumerate() {
                xtx[(ai, bi)] += ri * rj;
            }
        }
    }
    let chol = jittered_cholesky(xtx.as_ref())?;
    let beta = chol_solve(&chol.factor, &xty);

    // Residual variance of the second-stage regression.
    let mut ssr = 0.0;
    for t in t0..n {
        let mut fit = 0.0;
        let mut idx = 0;
        if include_constant {
            fit += beta[idx];
            idx += 1;
        }
        for i in 1..=p {
            fit += beta[idx] * target[t - i];
            idx += 1;
        }
        for j in 1..=q {
            fit += beta[idx] * e[t - j];
            idx += 1;
        }
        let r = target[t] - fit;
        ssr += r * r;
    }
    let mut sigma2 = ssr / (n - t0) as f64;
    if !(sigma2.is_finite() && sigma2 > 0.0) {
        sigma2 = variance_floored(x);
    }

    let mut out = Vec::with_capacity(k_reg + 1);
    out.extend_from_slice(&beta);
    out.push(sigma2);
    if out.iter().any(|v| !v.is_finite()) {
        return Err(ArimaError::NonFinite {
            what: "Hannan-Rissanen starting values",
        });
    }
    Ok(out)
}

/// Bland always-valid fallback start: zero AR/MA coefficients (interior
/// of both the stationarity and invertibility regions), the sample mean
/// as the constant, and the floored sample variance as `sigma2`.
fn default_start(x: &[f64], p: usize, q: usize, include_constant: bool) -> Vec<f64> {
    let mut out = Vec::with_capacity(usize::from(include_constant) + p + q + 1);
    if include_constant {
        out.push(mean(x));
    }
    out.extend(std::iter::repeat_n(0.0, p + q));
    out.push(variance_floored(x));
    out
}

/// Starting values guaranteed to lie strictly inside the constrained
/// domain: Hannan-Rissanen, with the AR and MA blocks shrunk toward zero
/// (factor 0.9 per step) until stationary/invertible, falling back to
/// [`default_start`] if that fails.
fn starting_values(x: &[f64], p: usize, q: usize, include_constant: bool) -> Vec<f64> {
    let transform = ArimaTransform {
        constant: include_constant,
        p,
        q,
        sigma2: true,
    };
    let c = usize::from(include_constant);
    if let Ok(mut start) = hannan_rissanen(x, p, q, include_constant) {
        for _ in 0..80 {
            if transform.inverse_vec(&start).is_ok() {
                return start;
            }
            for v in &mut start[c..c + p + q] {
                *v *= 0.9;
            }
        }
    }
    default_start(x, p, q, include_constant)
}

/// Picks the better of two optional stage results (lower objective
/// wins; the later stage wins ties). When both stages agree to within
/// roundoff, a convergence flag from either counts — the typical pattern
/// is one stage stopping on `LineSearchFailed` at the same point the
/// other certified.
fn pick_best(
    first: Option<OptimizeResult>,
    second: Option<OptimizeResult>,
) -> Result<OptimizeResult, ArimaError> {
    match (first, second) {
        (Some(a), Some(b)) => {
            let tied = (a.f - b.f).abs() <= 1e-9 * (1.0 + a.f.abs().min(b.f.abs()));
            let converged_either = a.converged || b.converged;
            let mut best = if b.f <= a.f { b } else { a };
            if tied {
                best.converged = converged_either;
            }
            Ok(best)
        }
        (Some(a), None) => Ok(a),
        (None, Some(b)) => Ok(b),
        (None, None) => Err(ArimaError::EstimationFailed {
            what: "the objective was non-finite at every point visited by \
                   both optimization stages",
        }),
    }
}

/// The default two-stage optimization: L-BFGS with central-difference
/// numerical gradients from `z0`, then a Nelder-Mead polish from the
/// L-BFGS solution (or from `z0` directly when L-BFGS fails — the
/// derivative-free fallback). Returns the better of the two.
///
/// The strong-Wolfe line search only requires sufficient decrease, so on
/// a multimodal surface L-BFGS may greedily step across a likelihood
/// ridge into a deeper basin — acceptable (desirable, even) for the
/// default fit, which reports the best mode found.
fn optimize_two_stage<F: ObjectiveFn>(
    obj: &mut F,
    z0: &[f64],
) -> Result<OptimizeResult, ArimaError> {
    let first = minimize(obj, z0, &Method::lbfgs())
        .ok()
        .filter(|r| r.f.is_finite());
    let polish_from = first.as_ref().map_or(z0, |r| r.x.as_slice()).to_vec();
    let second = minimize(obj, &polish_from, &Method::nelder_mead())
        .ok()
        .filter(|r| r.f.is_finite());
    pick_best(first, second)
}

/// Basin-faithful variant for user-supplied starting values: Nelder-Mead
/// first (its reflect/contract steps stay at the scale of the initial
/// simplex, so it refines within the starting basin), then an L-BFGS
/// polish from the refined point (from a local optimum the line search
/// cannot accept a basin-crossing step, because no descent direction
/// exists there). Returns the better of the two.
fn optimize_local_first<F: ObjectiveFn>(
    obj: &mut F,
    z0: &[f64],
) -> Result<OptimizeResult, ArimaError> {
    let first = minimize(obj, z0, &Method::nelder_mead())
        .ok()
        .filter(|r| r.f.is_finite());
    let polish_from = first.as_ref().map_or(z0, |r| r.x.as_slice()).to_vec();
    let second = minimize(obj, &polish_from, &Method::lbfgs())
        .ok()
        .filter(|r| r.f.is_finite());
    pick_best(first, second)
}

impl ArimaSpec {
    /// Exact Gaussian log-likelihood of `y` at the packed parameters
    /// `[const?, ar_1..ar_p, ma_1..ma_q, sigma2]`, by the Kalman filter's
    /// prediction-error decomposition (Harvey 1989, section 3.4) on the
    /// simply-differenced data — `d` observations are lost to
    /// differencing, exactly as statsmodels `SARIMAX(...,
    /// simple_differencing=True).loglike(params)`.
    ///
    /// # Errors
    ///
    /// * [`ArimaError::Dimension`] / [`ArimaError::NonFinite`] /
    ///   [`ArimaError::InvalidArgument`] on a malformed parameter vector;
    /// * [`ArimaError::NonFinite`] / [`ArimaError::InsufficientObservations`]
    ///   on malformed data;
    /// * [`ArimaError::Ssm`] when filtering fails (e.g. non-stationary AR
    ///   coefficients, for which no stationary initialization exists).
    pub fn loglike(&self, y: &[f64], params: &[f64]) -> Result<f64, ArimaError> {
        let diffed = difference(y, self.d())?;
        let blocks = self.unpack(params)?;
        let model = arma_ssm(blocks.ar, blocks.ma, blocks.sigma2, blocks.constant)?;
        let n = diffed.series.len();
        let y_mat = Mat::from_fn(n, 1, |i, _| diffed.series[i]);
        Ok(model.loglike(y_mat.as_ref())?)
    }

    /// Evaluates the model at fixed, user-supplied parameters — no
    /// optimization — returning a full results object (exact
    /// log-likelihood, information criteria, forecasting, residuals at
    /// those parameters). The analog of statsmodels
    /// `SARIMAX(...).smooth(params)`; useful for likelihood-surface
    /// inspection and for forecasting at externally estimated
    /// parameters.
    ///
    /// # Errors
    ///
    /// As for [`ArimaSpec::loglike`].
    pub fn at_params(&self, y: &[f64], params: &[f64]) -> Result<ArimaResults, ArimaError> {
        let diffed = difference(y, self.d())?;
        let blocks = self.unpack(params)?;
        let model = arma_ssm(blocks.ar, blocks.ma, blocks.sigma2, blocks.constant)?;
        let n_eff = diffed.series.len();
        let y_mat = Mat::from_fn(n_eff, 1, |i, _| diffed.series[i]);
        let loglik = model.loglike(y_mat.as_ref())?;
        Ok(ArimaResults::from_fit(
            *self,
            EstimationMethod::Fixed,
            params.to_vec(),
            loglik,
            n_eff,
            true,
            diffed.series,
            diffed.anchors,
        ))
    }

    /// Fits the model by exact Gaussian maximum likelihood on the
    /// state-space form.
    ///
    /// The optimizer works in unconstrained space through
    /// [`StationaryAr`] (AR), its invertibility dual (MA), and `exp`
    /// (`sigma2`); starting values come from the Hannan-Rissanen (1982)
    /// two-stage regression with a safe fallback; the search is L-BFGS
    /// with central-difference gradients followed by a Nelder-Mead
    /// polish/fallback. Non-convergence is reported through
    /// [`ArimaResults::converged`](crate::ArimaResults), never a panic.
    ///
    /// ARMA likelihoods can be multimodal; this method reports the best
    /// mode its search reaches (which may *exceed* a reference
    /// implementation's reported optimum). Use
    /// [`ArimaSpec::fit_with_start`] to select a basin explicitly.
    ///
    /// # Errors
    ///
    /// * [`ArimaError::NonFinite`] if `y` contains NaN/infinity;
    /// * [`ArimaError::InsufficientObservations`] when fewer than
    ///   `k + 1` observations remain after differencing;
    /// * [`ArimaError::EstimationFailed`] if no optimization stage found
    ///   a finite likelihood.
    pub fn fit(&self, y: &[f64]) -> Result<ArimaResults, ArimaError> {
        let diffed = difference(y, self.d())?;
        let start = starting_values(
            &diffed.series,
            self.p(),
            self.q(),
            self.include_constant(),
        );
        self.fit_mle_core(diffed.series, diffed.anchors, &start, false)
    }

    /// Exact Gaussian MLE started from user-supplied packed parameters
    /// `[const?, ar.., ma.., sigma2]` instead of the Hannan-Rissanen
    /// values — the statsmodels `fit(start_params=...)` analog.
    ///
    /// The stage order here is *local-first* — Nelder-Mead refinement
    /// from the given start, then an L-BFGS polish — so that on a
    /// genuinely multimodal surface the search refines the starting
    /// basin rather than greedily line-searching across a likelihood
    /// ridge, making the choice of starting point meaningful. (When the
    /// start is merely a stalled point of another implementation's
    /// optimizer rather than a true local optimum — the Nile
    /// ARMA(1,1)+constant fixture is a documented example — both orders
    /// correctly continue to the genuine maximizer.)
    ///
    /// # Errors
    ///
    /// As for [`ArimaSpec::fit`], plus
    /// [`ArimaError::Optim`] wrapping
    /// [`OptimError::NotStationary`](tsecon_optim::OptimError) /
    /// [`OptimError::Domain`](tsecon_optim::OptimError) when the starting
    /// point is outside the stationary/invertible/positive-variance
    /// domain (including exactly on its boundary).
    pub fn fit_with_start(&self, y: &[f64], start_params: &[f64]) -> Result<ArimaResults, ArimaError> {
        let diffed = difference(y, self.d())?;
        // Validates layout, finiteness, and sigma2 > 0.
        self.unpack(start_params)?;
        self.fit_mle_core(diffed.series, diffed.anchors, start_params, true)
    }

    /// Shared exact-MLE optimization core on the already-differenced
    /// sample; `start` must be a valid interior point of the constrained
    /// domain (stationary AR, invertible MA, positive variance).
    /// `local_first` selects the basin-faithful stage order used by
    /// [`ArimaSpec::fit_with_start`].
    fn fit_mle_core(
        &self,
        x: Vec<f64>,
        anchors: Vec<f64>,
        start: &[f64],
        local_first: bool,
    ) -> Result<ArimaResults, ArimaError> {
        let n_eff = x.len();
        let k = self.k_params();
        if n_eff < k + 1 {
            return Err(ArimaError::InsufficientObservations {
                needed: k + 1,
                got: n_eff,
            });
        }

        let transform = ArimaTransform {
            constant: self.include_constant(),
            p: self.p(),
            q: self.q(),
            sigma2: true,
        };
        // Errors (NotStationary / Domain) surface for boundary or
        // non-invertible user-supplied starts; the internal
        // `starting_values` always produces an interior point.
        let z0 = transform.inverse_vec(start)?;

        let inner = ExactNegLoglik {
            spec: *self,
            y: Mat::from_fn(n_eff, 1, |i, _| x[i]),
        };
        let mut obj = TransformedObjective::new(inner, transform);
        let best = if local_first {
            optimize_local_first(&mut obj, &z0)?
        } else {
            optimize_two_stage(&mut obj, &z0)?
        };
        let params = obj.constrained(&best.x)?;
        let loglik = -best.f;

        Ok(ArimaResults::from_fit(
            *self,
            EstimationMethod::ExactMle,
            params,
            loglik,
            n_eff,
            best.converged,
            x,
            anchors,
        ))
    }

    /// Fits the model by conditional sum of squares (CSS): minimizes the
    /// sum of squared recursive residuals conditioning on the first `p`
    /// observations and zero pre-sample innovations (Box & Jenkins 1976,
    /// chapter 7), with `sigma2 = SSR / n_c` concentrated out
    /// (`n_c = n - d - p`).
    ///
    /// CSS is **not** exact MLE in small samples: it ignores the
    /// stationary distribution of the first `p` observations and the
    /// pre-sample innovations, so its estimates differ from
    /// [`ArimaSpec::fit`] by `O(1/n)` (they agree asymptotically). It is
    /// the fast alternative and a common starting point for exact MLE.
    /// The reported `loglik` (and hence AIC/BIC) is the *conditional*
    /// Gaussian log-likelihood of the `n_c` residuals — not comparable
    /// with the exact log-likelihood — and
    /// [`ArimaResults::nobs`](crate::ArimaResults) is `n_c`.
    ///
    /// # Errors
    ///
    /// As for [`ArimaSpec::fit`].
    pub fn fit_css(&self, y: &[f64]) -> Result<ArimaResults, ArimaError> {
        let diffed = difference(y, self.d())?;
        let x = diffed.series;
        let n_eff = x.len();
        let k = self.k_params();
        if n_eff < self.p() + k + 1 {
            return Err(ArimaError::InsufficientObservations {
                needed: self.p() + k + 1,
                got: n_eff,
            });
        }

        let c = usize::from(self.include_constant());
        let k_mean = c + self.p() + self.q();
        let (mean_params, converged) = if k_mean == 0 {
            // ARIMA(0, d, 0) without a constant: nothing to optimize.
            (Vec::new(), true)
        } else {
            let mut start = starting_values(&x, self.p(), self.q(), self.include_constant());
            start.truncate(k_mean); // drop the sigma2 slot
            let transform = ArimaTransform {
                constant: self.include_constant(),
                p: self.p(),
                q: self.q(),
                sigma2: false,
            };
            let z0 = transform.inverse_vec(&start)?;
            let inner = CssNegLoglik {
                constant: self.include_constant(),
                p: self.p(),
                q: self.q(),
                x: &x,
            };
            let mut obj = TransformedObjective::new(inner, transform);
            let best = optimize_two_stage(&mut obj, &z0)?;
            (obj.constrained(&best.x)?, best.converged)
        };

        let constant = if self.include_constant() {
            mean_params[0]
        } else {
            0.0
        };
        let ar = &mean_params[c..c + self.p()];
        let ma = &mean_params[c + self.p()..k_mean];
        let (ssr, n_c) =
            css_ssr(&x, constant, ar, ma).ok_or(ArimaError::EstimationFailed {
                what: "CSS residual recursion overflowed at the optimum",
            })?;
        let sigma2 = ssr / n_c as f64;
        if !(sigma2.is_finite() && sigma2 > 0.0) {
            return Err(ArimaError::EstimationFailed {
                what: "CSS residual variance is not strictly positive",
            });
        }
        let loglik =
            -0.5 * n_c as f64 * ((2.0 * std::f64::consts::PI).ln() + 1.0 + sigma2.ln());

        let mut params = mean_params;
        params.push(sigma2);
        Ok(ArimaResults::from_fit(
            *self,
            EstimationMethod::Css,
            params,
            loglik,
            n_c,
            converged,
            x,
            diffed.anchors,
        ))
    }
}
