//! The ARIMA model specification: orders, constant, parameter layout.

use crate::error::ArimaError;

/// Guard against accidental pathological orders (a mistyped order would
/// otherwise allocate enormous state matrices before any data check).
const MAX_ORDER: usize = 1000;

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
/// (`simple_differencing=False`), which keeps the `d` lost observations
/// via exact diffuse initialization.
///
/// `// TODO(phase0)`: seasonal orders — the struct is constructed through
/// [`ArimaSpec::new`] precisely so `(P, D, Q, s)` fields can slot in
/// without breaking the API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArimaSpec {
    p: usize,
    d: usize,
    q: usize,
    include_constant: bool,
    // TODO(phase0): seasonal orders (seasonal_p, seasonal_d, seasonal_q,
    // period) slot in here; `new` gains a `seasonal(...)` builder method.
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
                what: "ARIMA orders p, d, q must each be at most 1000",
            });
        }
        Ok(Self {
            p,
            d,
            q,
            include_constant: false,
        })
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

    /// Number of estimated parameters `k`: constant (if any) + `p` AR +
    /// `q` MA + the innovation variance `sigma2` — statsmodels counts
    /// `sigma2` in `k` for AIC/BIC, and so does this crate.
    #[inline]
    pub fn k_params(&self) -> usize {
        usize::from(self.include_constant) + self.p + self.q + 1
    }

    /// Parameter names in estimation order, statsmodels style:
    /// `["const"?, "ar.L1", ..., "ar.Lp", "ma.L1", ..., "ma.Lq", "sigma2"]`.
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
        names.push("sigma2".to_owned());
        names
    }

    /// Splits a packed parameter vector `[const?, ar.., ma.., sigma2]`
    /// into its blocks, validating length, finiteness, and `sigma2 > 0`.
    pub(crate) fn unpack<'a>(&self, params: &'a [f64]) -> Result<ParamBlocks<'a>, ArimaError> {
        let k = self.k_params();
        if params.len() != k {
            return Err(ArimaError::Dimension {
                what: "params must be [const?, ar.., ma.., sigma2]",
                expected: k,
                got: params.len(),
            });
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(ArimaError::NonFinite { what: "params" });
        }
        let (constant, rest) = if self.include_constant {
            (params[0], &params[1..])
        } else {
            (0.0, params)
        };
        let ar = &rest[..self.p];
        let ma = &rest[self.p..self.p + self.q];
        let sigma2 = rest[self.p + self.q];
        if sigma2 <= 0.0 {
            return Err(ArimaError::InvalidArgument {
                what: "sigma2 must be strictly positive",
            });
        }
        Ok(ParamBlocks {
            constant,
            ar,
            ma,
            sigma2,
        })
    }
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
    /// Innovation variance.
    pub(crate) sigma2: f64,
}
