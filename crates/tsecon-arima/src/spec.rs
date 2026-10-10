//! The ARIMA model specification: orders, constant, parameter layout.

use crate::error::ArimaError;
use crate::results::EstimationMethod;

/// Guard against accidental pathological orders (a mistyped order would
/// otherwise allocate enormous state matrices before any data check).
const MAX_ORDER: usize = 1000;

/// Work budget of one ARIMA estimation, in multiply-adds (`2^37`): the work
/// [`ArimaSpec::estimation_cost`] counts — what every fit is *guaranteed* to
/// do, whatever its optimizer does — may not pass it. Measured on the
/// release wheel (`lab/arima_budget/arima_cost.py`), see that script's
/// output for seconds per unit.
pub const ARIMA_WORK_BUDGET: u128 = 1 << 37;

/// Memory budget of one exact-likelihood evaluation, in `f64` values
/// (`2^28`, 2 GiB): the Kalman filter keeps three `m x m` covariances per
/// observation (predicted, predicted-diffuse and filtered), `3 (n_eff + 1)
/// m^2` values for a state of dimension `m`.
pub const ARIMA_BUDGET_VALUES: u128 = 1 << 28;

/// Specification of an ARIMA(p, d, q) model, optionally with a constant:
///
/// ```text
/// (1 - phi_1 L - ... - phi_p L^p) (1 - L)^d y_t
///     = c + (1 + theta_1 L + ... + theta_q L^q) eps_t,
/// eps_t ~ N(0, sigma2)
/// ```
///
/// following the Box-Jenkins orders (Box & Jenkins 1976) and the
/// statsmodels `SARIMAX` sign conventions: MA coefficients enter with a
/// *plus* sign, and the constant `c` is the regression intercept of the
/// (differenced) series — statsmodels `trend='c'` — *not* the process
/// mean (the mean of the differenced series is
/// `c / (1 - phi_1 - ... - phi_p)`).
///
/// Differencing (`d > 0`) uses **simple differencing**: the data are
/// differenced `d` times up front and the ARMA(p, q) model is fit to the
/// differences, losing `d` observations — the statsmodels
/// `simple_differencing=True` convention. Forecasts are re-cumulated to
/// levels with the correct cumulative variance (see
/// [`ArimaResults::forecast`](crate::ArimaResults::forecast)).
/// `// TODO(phase0)`: the levels state-space form
/// (`simple_differencing=False`), which keeps the `d + D*s` lost
/// observations via exact diffuse initialization.
///
/// Seasonal orders `(P, D, Q, s)` are added with
/// [`ArimaSpec::seasonal`], turning the model into the multiplicative
/// SARIMA(p, d, q)(P, D, Q)_s
///
/// ```text
/// phi(L) Phi(L^s) (1 - L)^d (1 - L^s)^D y_t = c + theta(L) Theta(L^s) eps_t
/// ```
///
/// with the same statsmodels sign conventions: seasonal MA coefficients
/// enter with a *plus* sign, and estimation multiplies the regular and
/// seasonal polynomials into a single dense ARMA(p + s*P, q + s*Q) that
/// runs through the same state-space engine. Seasonal differencing is
/// applied *before* regular differencing (the statsmodels
/// `tools.diff` order), losing `d + D*s` observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArimaSpec {
    p: usize,
    d: usize,
    q: usize,
    include_constant: bool,
    seasonal_p: usize,
    seasonal_d: usize,
    seasonal_q: usize,
    period: usize,
}

impl ArimaSpec {
    /// A non-seasonal ARIMA(p, d, q) specification without a constant.
    ///
    /// Add the constant with [`with_constant`](ArimaSpec::with_constant).
    /// ARIMA(0, d, 0) is valid (white noise after differencing).
    ///
    /// # Errors
    ///
    /// [`ArimaError::InvalidArgument`] when any order exceeds the sanity
    /// cap of 1000 (guards against mistyped orders allocating enormous
    /// state matrices).
    pub fn new(p: usize, d: usize, q: usize) -> Result<Self, ArimaError> {
        if p > MAX_ORDER || d > MAX_ORDER || q > MAX_ORDER {
            return Err(ArimaError::InvalidArgument {
                what: "each of the ARIMA orders p, d, q must be at most 1000; a larger \
                       one is almost always a typo and would allocate an enormous \
                       state-space matrix",
            });
        }
        Ok(Self {
            p,
            d,
            q,
            include_constant: false,
            seasonal_p: 0,
            seasonal_d: 0,
            seasonal_q: 0,
            period: 0,
        })
    }

    /// Adds multiplicative seasonal orders `(P, D, Q)` at period `s`
    /// (statsmodels `seasonal_order=(P, D, Q, s)`), returning the SARIMA
    /// specification. `(0, 0, 0)` at any period is the non-seasonal model
    /// (nothing to multiply in), matching statsmodels' default
    /// `seasonal_order=(0, 0, 0, 0)`.
    ///
    /// # Errors
    ///
    /// [`ArimaError::InvalidArgument`] when
    ///
    /// * any of `P`, `D`, `Q` is nonzero and `s < 2` — a period of 1
    ///   would put every "seasonal" lag on top of a regular lag, and a
    ///   period of 0 has no meaning;
    /// * an order or the period exceeds the sanity cap of 1000, or an
    ///   *expanded* order (`p + s*P`, `q + s*Q`, `d + s*D`) does — the
    ///   multiplied-out polynomials are what size the state matrices.
    pub fn seasonal(
        mut self,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        period: usize,
    ) -> Result<Self, ArimaError> {
        let any = seasonal_p > 0 || seasonal_d > 0 || seasonal_q > 0;
        if any && period < 2 {
            return Err(ArimaError::InvalidArgument {
                what: "a seasonal specification needs a period s >= 2: s = 1 would put \
                       every seasonal lag on top of a regular lag, and s = 0 has no \
                       meaning. For a non-seasonal model use seasonal orders (0, 0, 0)",
            });
        }
        if seasonal_p > MAX_ORDER
            || seasonal_d > MAX_ORDER
            || seasonal_q > MAX_ORDER
            || period > MAX_ORDER
        {
            return Err(ArimaError::InvalidArgument {
                what: "each of the seasonal orders P, D, Q and the period s must be at \
                       most 1000; a larger one is almost always a typo and would \
                       allocate an enormous state-space matrix",
            });
        }
        let expanded_cap = |regular: usize, seasonal: usize| -> bool {
            period
                .checked_mul(seasonal)
                .and_then(|sp| sp.checked_add(regular))
                .is_some_and(|full| full <= MAX_ORDER)
        };
        if any
            && (!expanded_cap(self.p, seasonal_p)
                || !expanded_cap(self.q, seasonal_q)
                || !expanded_cap(self.d, seasonal_d))
        {
            return Err(ArimaError::InvalidArgument {
                what: "the expanded SARIMA orders p + s*P, q + s*Q and d + s*D must each \
                       be at most 1000: the multiplied-out polynomials are what size the \
                       state-space matrices",
            });
        }
        if any {
            self.seasonal_p = seasonal_p;
            self.seasonal_d = seasonal_d;
            self.seasonal_q = seasonal_q;
            self.period = period;
        }
        Ok(self)
    }

    /// Toggles the constant term (statsmodels `trend='c'`; default off).
    #[must_use]
    pub fn with_constant(mut self, include_constant: bool) -> Self {
        self.include_constant = include_constant;
        self
    }

    /// Autoregressive order `p`.
    #[inline]
    pub fn p(&self) -> usize {
        self.p
    }

    /// Differencing order `d`.
    #[inline]
    pub fn d(&self) -> usize {
        self.d
    }

    /// Moving-average order `q`.
    #[inline]
    pub fn q(&self) -> usize {
        self.q
    }

    /// Whether the model includes a constant.
    #[inline]
    pub fn include_constant(&self) -> bool {
        self.include_constant
    }

    /// Seasonal autoregressive order `P` (0 when non-seasonal).
    #[inline]
    pub fn seasonal_p(&self) -> usize {
        self.seasonal_p
    }

    /// Seasonal differencing order `D` (0 when non-seasonal).
    #[inline]
    pub fn seasonal_d(&self) -> usize {
        self.seasonal_d
    }

    /// Seasonal moving-average order `Q` (0 when non-seasonal).
    #[inline]
    pub fn seasonal_q(&self) -> usize {
        self.seasonal_q
    }

    /// Seasonal period `s` (0 when non-seasonal).
    #[inline]
    pub fn period(&self) -> usize {
        self.period
    }

    /// Number of estimated parameters `k`: constant (if any) + `p` AR +
    /// `q` MA + `P` seasonal AR + `Q` seasonal MA + the innovation
    /// variance `sigma2` — statsmodels counts `sigma2` in `k` for
    /// AIC/BIC, and so does this crate.
    #[inline]
    pub fn k_params(&self) -> usize {
        usize::from(self.include_constant) + self.p + self.q + self.seasonal_p + self.seasonal_q + 1
    }

    /// The state dimension of the exact likelihood's Kalman filter,
    /// `m = max(p + s*P, q + s*Q + 1)` (the Harvey form of the multiplied-out
    /// ARMA(p + s*P, q + s*Q); see [`crate::arma_ssm`]).
    #[inline]
    pub fn state_dim(&self) -> usize {
        let ar = self.p + self.period * self.seasonal_p;
        let ma = self.q + self.period * self.seasonal_q;
        ar.max(ma + 1)
    }

    /// The cost of estimating this specification on `n_eff` usable
    /// observations (after differencing) by `method`: `(work, values)`, the
    /// multiply-adds every such fit is guaranteed to spend and the `f64`
    /// values one exact-likelihood evaluation holds. Counted in `u128`, so
    /// no order can overflow the check meant to refuse it.
    ///
    /// With `m` the [`state_dim`](ArimaSpec::state_dim) and `k` the
    /// [`k_params`](ArimaSpec::k_params):
    ///
    /// * one exact-likelihood evaluation costs `E = (4 n_eff + 200) m^3`
    ///   (each filter step forms `T P T'` for the covariance and its diffuse
    ///   companion, about `4 m^3`; the stationary initialization — Lyapunov
    ///   doubling plus the spectral-radius check — at most about `200 m^3`)
    ///   and holds `3 (n_eff + 1) m^2` values;
    /// * a fit evaluates its objective at least `2 k (k + 2) + 1` times
    ///   whatever the optimizer does: one central-difference gradient
    ///   (`2k + 1`) and the four-point observed-information Hessian behind
    ///   the standard errors (`2 k (k + 1)`);
    /// * exact MLE ([`EstimationMethod::ExactMle`]): `work = (2 k (k + 2) +
    ///   1) E`; CSS ([`EstimationMethod::Css`]): the recursion costs
    ///   `n_eff (p + s*P + q + s*Q + 1)` per evaluation and the residuals run
    ///   the exact filter once more, `work = (2 k (k + 2) + 1) n_eff (p + s*P
    ///   + q + s*Q + 1) + E`; a single evaluation at fixed parameters
    ///   ([`EstimationMethod::Fixed`]): `work = E`.
    pub fn estimation_cost(&self, n_eff: usize, method: EstimationMethod) -> (u128, u128) {
        cost_at(self.state_dim(), self, n_eff, method)
    }

    /// Refuses a specification whose [`estimation_cost`](Self::estimation_cost)
    /// passes [`ARIMA_WORK_BUDGET`] or [`ARIMA_BUDGET_VALUES`], naming the
    /// orders and the largest state dimension that fits.
    pub(crate) fn check_budget(
        &self,
        n_eff: usize,
        method: EstimationMethod,
    ) -> Result<(), ArimaError> {
        let fits = |m: usize| {
            let (work, values) = cost_at(m, self, n_eff, method);
            (work <= ARIMA_WORK_BUDGET, values <= ARIMA_BUDGET_VALUES)
        };
        let m = self.state_dim();
        let exceeded = match fits(m) {
            (true, true) => return Ok(()),
            (false, false) => {
                "both the work budget of 2^37 multiply-adds and the memory budget of 2^28 \
                 values (2 GiB)"
            }
            (false, true) => "the work budget of 2^37 multiply-adds",
            (true, false) => "the memory budget of 2^28 values (2 GiB) for the filter",
        };
        // Both counts grow with m: bisect on the largest that fits.
        let ok = |m: usize| fits(m) == (true, true);
        let max_state_dim = if ok(1) {
            let (mut lo, mut hi) = (1usize, m);
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                if ok(mid) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            lo
        } else {
            0
        };
        Err(ArimaError::TooLarge {
            state_dim: m,
            k_params: self.k_params(),
            n_eff,
            exceeded,
            max_state_dim,
        })
    }

    /// Parameter names in estimation order, statsmodels style:
    /// `["const"?, "ar.L1", ..., "ar.Lp", "ma.L1", ..., "ma.Lq",
    /// "ar.S.Ls", ..., "ar.S.L{P*s}", "ma.S.Ls", ..., "ma.S.L{Q*s}",
    /// "sigma2"]`.
    pub fn param_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.k_params());
        if self.include_constant {
            names.push("const".to_owned());
        }
        for i in 1..=self.p {
            names.push(format!("ar.L{i}"));
        }
        for j in 1..=self.q {
            names.push(format!("ma.L{j}"));
        }
        for i in 1..=self.seasonal_p {
            names.push(format!("ar.S.L{}", i * self.period));
        }
        for j in 1..=self.seasonal_q {
            names.push(format!("ma.S.L{}", j * self.period));
        }
        names.push("sigma2".to_owned());
        names
    }

    /// Splits a packed parameter vector `[const?, ar.., ma.., sar..,
    /// sma.., sigma2]` into its blocks, validating length, finiteness,
    /// and `sigma2 > 0`.
    pub(crate) fn unpack<'a>(&self, params: &'a [f64]) -> Result<ParamBlocks<'a>, ArimaError> {
        let k = self.k_params();
        if params.len() != k {
            return Err(ArimaError::Dimension {
                what: "the packed parameter vector must be [const?, ar.., ma.., \
                       seasonal ar.., seasonal ma.., sigma2]",
                expected: k,
                got: params.len(),
            });
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(ArimaError::NonFinite {
                what: "the packed parameter vector",
                at: None,
            });
        }
        let (constant, rest) = if self.include_constant {
            (params[0], &params[1..])
        } else {
            (0.0, params)
        };
        let ar = &rest[..self.p];
        let ma = &rest[self.p..self.p + self.q];
        let sar = &rest[self.p + self.q..self.p + self.q + self.seasonal_p];
        let sma_end = self.p + self.q + self.seasonal_p + self.seasonal_q;
        let sma = &rest[self.p + self.q + self.seasonal_p..sma_end];
        let sigma2 = rest[sma_end];
        if sigma2 <= 0.0 {
            return Err(ArimaError::InvalidArgument {
                what: "sigma2 (the innovation variance) must be strictly positive",
            });
        }
        Ok(ParamBlocks {
            constant,
            ar,
            ma,
            sar,
            sma,
            sigma2,
        })
    }
}

/// [`ArimaSpec::estimation_cost`] at state dimension `m` (the bisection in
/// [`ArimaSpec::check_budget`] varies `m` with the parameter count fixed).
fn cost_at(m: usize, spec: &ArimaSpec, n_eff: usize, method: EstimationMethod) -> (u128, u128) {
    let (m, n, k) = (m as u128, n_eff as u128, spec.k_params() as u128);
    let exact = (4 * n + 200) * m * m * m;
    let values = 3 * (n + 1) * m * m;
    let evals = 2 * k * (k + 2) + 1;
    let work = match method {
        EstimationMethod::ExactMle => evals * exact,
        EstimationMethod::Css => {
            let lags = (spec.p + spec.period * spec.seasonal_p) as u128
                + (spec.q + spec.period * spec.seasonal_q) as u128
                + 1;
            evals * n * lags + exact
        }
        EstimationMethod::Fixed => exact,
    };
    (work, values)
}

/// Multiplies a regular lag polynomial into a seasonal one, returning the
/// dense coefficient vector of the product in the same storage
/// convention as its inputs.
///
/// Both AR and MA polynomials are stored without their leading 1 and
/// without their sign convention: the AR polynomial `1 - sum_k c_k L^k`
/// and the MA polynomial `1 + sum_k c_k L^k` both store `[c_1, ..,
/// c_n]`. Under those conventions the product picks up its cross terms
/// with opposite signs —
///
/// ```text
/// (1 - sum phi_i L^i)(1 - sum Phi_j L^{js})
///     = 1 - sum phi_i L^i - sum Phi_j L^{js} + sum phi_i Phi_j L^{i+js}
/// (1 + sum theta_i L^i)(1 + sum Theta_j L^{js})
///     = 1 + sum theta_i L^i + sum Theta_j L^{js} + sum theta_i Theta_j L^{i+js}
/// ```
///
/// — so the stored cross coefficient is `-phi*Phi` for AR
/// (`cross_sign = -1`) and `+theta*Theta` for MA (`cross_sign = +1`).
/// The result has length `regular.len() + s * seasonal.len()` (or is a
/// copy of `regular` when there is no seasonal block).
pub(crate) fn multiply_lag_polys(
    regular: &[f64],
    seasonal: &[f64],
    s: usize,
    cross_sign: f64,
) -> Vec<f64> {
    if seasonal.is_empty() {
        return regular.to_vec();
    }
    let mut out = vec![0.0; regular.len() + s * seasonal.len()];
    out[..regular.len()].copy_from_slice(regular);
    for (j, &b) in seasonal.iter().enumerate() {
        let js = s * (j + 1);
        out[js - 1] += b;
        for (i, &a) in regular.iter().enumerate() {
            out[js + i] += cross_sign * a * b;
        }
    }
    out
}

/// Borrowed view of a packed parameter vector, split into blocks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ParamBlocks<'a> {
    /// Constant term (0 when the spec has no constant).
    pub(crate) constant: f64,
    /// AR coefficients `phi_1..phi_p`.
    pub(crate) ar: &'a [f64],
    /// MA coefficients `theta_1..theta_q`.
    pub(crate) ma: &'a [f64],
    /// Seasonal AR coefficients `Phi_1..Phi_P`.
    pub(crate) sar: &'a [f64],
    /// Seasonal MA coefficients `Theta_1..Theta_Q`.
    pub(crate) sma: &'a [f64],
    /// Innovation variance.
    pub(crate) sigma2: f64,
}

impl ParamBlocks<'_> {
    /// The multiplied-out dense AR and MA coefficient vectors
    /// `phi(L)Phi(L^s)` and `theta(L)Theta(L^s)`, of lengths `p + s*P`
    /// and `q + s*Q` — what the state-space form and the CSS recursion
    /// actually run on.
    pub(crate) fn expanded(&self, period: usize) -> (Vec<f64>, Vec<f64>) {
        (
            multiply_lag_polys(self.ar, self.sar, period, -1.0),
            multiply_lag_polys(self.ma, self.sma, period, 1.0),
        )
    }
}
