//! The ETS taxonomy: error / trend / seasonal component types and the
//! [`EtsSpec`] that names one of the 30 models.

use crate::error::EtsError;

/// The error (innovation) type: `y_t = mu_t + e_t` (additive) or
/// `y_t = mu_t (1 + e_t)` (multiplicative).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorType {
    /// Additive innovations.
    Additive,
    /// Multiplicative (relative) innovations; needs strictly positive data.
    Multiplicative,
}

/// A trend or seasonal component: absent, additive, or multiplicative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Component {
    /// The component is absent (`N`).
    None,
    /// Additive (`A`).
    Additive,
    /// Multiplicative (`M`); needs strictly positive data.
    Multiplicative,
}

impl Component {
    /// `true` unless the component is absent.
    pub fn is_present(self) -> bool {
        !matches!(self, Component::None)
    }

    /// The taxonomy letter (`N`, `A`, `M`).
    pub fn letter(self) -> char {
        match self {
            Component::None => 'N',
            Component::Additive => 'A',
            Component::Multiplicative => 'M',
        }
    }
}

/// One member of the ETS(Error, Trend, Seasonal) taxonomy — Hyndman,
/// Koehler, Snyder & Grose (2002); Hyndman et al. (2008, chapter 2).
///
/// `seasonal_periods` is the seasonal period `m` and must be `1` when
/// `seasonal` is [`Component::None`] (a non-seasonal model carries no
/// period), `>= 2` otherwise. Build with [`EtsSpec::new`], which enforces
/// those rules and refuses a damped trend without a trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EtsSpec {
    /// Error type.
    pub error: ErrorType,
    /// Trend component.
    pub trend: Component,
    /// Whether the trend is damped (`phi` estimated); only with a trend.
    pub damped: bool,
    /// Seasonal component.
    pub seasonal: Component,
    /// Seasonal period `m` (`1` for a non-seasonal model).
    pub seasonal_periods: usize,
}

impl EtsSpec {
    /// Builds and validates a specification. `seasonal_periods` must be
    /// `Some(m >= 2)` exactly when a seasonal component is requested, and
    /// `None` otherwise (passing a period to a non-seasonal model is
    /// refused rather than ignored: it would be inert).
    ///
    /// # Errors
    ///
    /// [`EtsError::InvalidSpec`] on any inconsistency.
    pub fn new(
        error: ErrorType,
        trend: Component,
        damped: bool,
        seasonal: Component,
        seasonal_periods: Option<usize>,
    ) -> Result<Self, EtsError> {
        if damped && !trend.is_present() {
            return Err(EtsError::InvalidSpec {
                what: "damped=True was given but trend=None: only a trend component \
                       can be damped (phi multiplies the trend), so the option is \
                       inert without one; pass trend=\"add\" or \"mul\", or damped=False",
            });
        }
        let m = match (seasonal.is_present(), seasonal_periods) {
            (false, None) => 1,
            (false, Some(_)) => {
                return Err(EtsError::InvalidSpec {
                    what: "seasonal_periods was given but seasonal=None: the period \
                           only sizes a seasonal component, so it is inert without one; \
                           pass seasonal=\"add\" or \"mul\", or drop seasonal_periods",
                })
            }
            (true, None) => {
                return Err(EtsError::InvalidSpec {
                    what: "seasonal=\"add\"/\"mul\" needs seasonal_periods (the period m \
                           of the seasonal pattern: 12 for monthly, 4 for quarterly); \
                           pass seasonal_periods=m",
                })
            }
            (true, Some(m)) if m >= 2 => m,
            (true, Some(_)) => {
                return Err(EtsError::InvalidSpec {
                    what: "seasonal_periods must be at least 2 (a seasonal pattern needs \
                           at least two periods); pass seasonal=None for a \
                           non-seasonal model",
                })
            }
        };
        let spec = EtsSpec {
            error,
            trend,
            damped,
            seasonal,
            seasonal_periods: m,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Validates a specification built by hand.
    ///
    /// # Errors
    ///
    /// [`EtsError::InvalidSpec`] when `damped` is set without a trend, or
    /// `seasonal_periods` disagrees with the seasonal component.
    pub fn validate(&self) -> Result<(), EtsError> {
        if self.damped && !self.trend.is_present() {
            return Err(EtsError::InvalidSpec {
                what: "damped=True was given but trend=None: only a trend component \
                       can be damped (phi multiplies the trend), so the option is \
                       inert without one; pass trend=\"add\" or \"mul\", or damped=False",
            });
        }
        if self.seasonal.is_present() && self.seasonal_periods < 2 {
            return Err(EtsError::InvalidSpec {
                what: "seasonal_periods must be at least 2 for a seasonal model",
            });
        }
        if !self.seasonal.is_present() && self.seasonal_periods != 1 {
            return Err(EtsError::InvalidSpec {
                what: "seasonal_periods must be 1 for a non-seasonal model (the period \
                       is inert without a seasonal component)",
            });
        }
        Ok(())
    }

    /// The taxonomy name, e.g. `ETS(M,Ad,M)`.
    pub fn name(&self) -> String {
        format!("ETS({})", self.short_name_inner(','))
    }

    /// The compact name, e.g. `MAdM`.
    pub fn short_name(&self) -> String {
        self.short_name_inner('\0')
    }

    fn short_name_inner(&self, sep: char) -> String {
        let e = match self.error {
            ErrorType::Additive => 'A',
            ErrorType::Multiplicative => 'M',
        };
        let mut s = String::new();
        s.push(e);
        if sep != '\0' {
            s.push(sep);
        }
        s.push(self.trend.letter());
        if self.damped {
            s.push('d');
        }
        if sep != '\0' {
            s.push(sep);
        }
        s.push(self.seasonal.letter());
        s
    }

    /// Whether a trend component is present.
    pub fn has_trend(&self) -> bool {
        self.trend.is_present()
    }

    /// Whether a seasonal component is present.
    pub fn has_seasonal(&self) -> bool {
        self.seasonal.is_present()
    }

    /// The seasonal period `m` (`1` when non-seasonal).
    pub fn m(&self) -> usize {
        self.seasonal_periods
    }

    /// Class 1 of Hyndman et al. (2008, chapter 6): additive error with
    /// additive-or-absent trend and seasonal — the linear models with
    /// closed-form forecast variances (their Table 6.1).
    pub fn is_class1(&self) -> bool {
        self.error == ErrorType::Additive
            && self.trend != Component::Multiplicative
            && self.seasonal != Component::Multiplicative
    }

    /// Whether any component is multiplicative (so the data must be
    /// strictly positive).
    pub fn needs_positive_data(&self) -> bool {
        self.error == ErrorType::Multiplicative
            || self.trend == Component::Multiplicative
            || self.seasonal == Component::Multiplicative
    }

    /// Number of smoothing parameters: `alpha`, plus `beta` with a trend,
    /// `gamma` with a seasonal, `phi` when damped.
    pub fn n_smoothing(&self) -> usize {
        1 + usize::from(self.has_trend())
            + usize::from(self.has_seasonal())
            + usize::from(self.damped)
    }

    /// Number of initial-state values: level, plus trend, plus `m`
    /// seasonal states.
    pub fn n_initial_states(&self) -> usize {
        1 + usize::from(self.has_trend()) + if self.has_seasonal() { self.m() } else { 0 }
    }

    /// Number of *free* initial states — the seasonal states are
    /// normalised (sum to zero / average one), so only `m - 1` of them are
    /// free parameters (R's `ets` counts them the same way).
    pub fn n_free_initial_states(&self) -> usize {
        1 + usize::from(self.has_trend()) + if self.has_seasonal() { self.m() - 1 } else { 0 }
    }

    /// Names of the smoothing parameters in the packed order
    /// (`alpha`, `beta`, `gamma`, `phi` — those present).
    pub fn smoothing_names(&self) -> Vec<&'static str> {
        let mut v = vec!["alpha"];
        if self.has_trend() {
            v.push("beta");
        }
        if self.has_seasonal() {
            v.push("gamma");
        }
        if self.damped {
            v.push("phi");
        }
        v
    }

    /// Names of the initial states in the packed order (`level`, `trend`,
    /// `seasonal[0..m)` — those present; `seasonal[j]` is the state in
    /// force for observation `j`).
    pub fn initial_state_names(&self) -> Vec<String> {
        let mut v = vec!["level".to_string()];
        if self.has_trend() {
            v.push("trend".to_string());
        }
        if self.has_seasonal() {
            for j in 0..self.m() {
                v.push(format!("seasonal[{j}]"));
            }
        }
        v
    }
}

/// The smoothing parameters of one ETS model.
#[derive(Debug, Clone, PartialEq)]
pub struct EtsParams {
    /// Level smoothing parameter `alpha`.
    pub alpha: f64,
    /// Trend smoothing parameter `beta` (Hyndman's `beta`, not
    /// `beta* = beta / alpha`); `None` without a trend.
    pub beta: Option<f64>,
    /// Seasonal smoothing parameter `gamma` (Hyndman's `gamma`, not
    /// `gamma* = gamma / (1 - alpha)`); `None` without a seasonal.
    pub gamma: Option<f64>,
    /// Damping parameter `phi`; `None` unless damped.
    pub phi: Option<f64>,
}

impl EtsParams {
    /// Unpacks `[alpha, beta?, gamma?, phi?]` (the components present, in
    /// that order) for `spec`, checking finiteness and the traditional
    /// domain `0 <= alpha <= 1`, `0 <= beta <= alpha`, `0 <= gamma <= 1 - alpha`,
    /// `0 < phi <= 1`. Evaluation at fixed parameters accepts the closed
    /// box (boundaries included) so the SES / naive limits can be scored.
    ///
    /// # Errors
    ///
    /// [`EtsError::DimensionMismatch`] on the wrong length,
    /// [`EtsError::InvalidParameter`] on a value outside its domain.
    pub fn from_slice(spec: &EtsSpec, v: &[f64]) -> Result<Self, EtsError> {
        let names = spec.smoothing_names();
        if v.len() != names.len() {
            return Err(EtsError::DimensionMismatch {
                what: "smoothing_params (the smoothing parameters [alpha, beta?, gamma?, phi?] this spec takes)",
                expected: names.len(),
                got: v.len(),
            });
        }
        for (name, &x) in names.iter().zip(v) {
            if !x.is_finite() {
                return Err(EtsError::InvalidParameter {
                    name: (*name).to_string(),
                    value: x,
                    requirement: "must be finite",
                });
            }
        }
        let mut it = v.iter().copied();
        let alpha = it.next().unwrap_or(0.0);
        let beta = if spec.has_trend() { it.next() } else { None };
        let gamma = if spec.has_seasonal() { it.next() } else { None };
        let phi = if spec.damped { it.next() } else { None };
        let p = EtsParams {
            alpha,
            beta,
            gamma,
            phi,
        };
        p.check_domain()?;
        Ok(p)
    }

    /// The packed vector `[alpha, beta?, gamma?, phi?]`.
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = vec![self.alpha];
        if let Some(b) = self.beta {
            v.push(b);
        }
        if let Some(g) = self.gamma {
            v.push(g);
        }
        if let Some(p) = self.phi {
            v.push(p);
        }
        v
    }

    /// Checks the traditional parameter box (boundaries included).
    ///
    /// # Errors
    ///
    /// [`EtsError::InvalidParameter`] naming the parameter.
    pub fn check_domain(&self) -> Result<(), EtsError> {
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(EtsError::InvalidParameter {
                name: "alpha".into(),
                value: self.alpha,
                requirement: "the level smoothing parameter must lie in [0, 1]",
            });
        }
        if let Some(b) = self.beta {
            if !(b >= 0.0 && b <= self.alpha) {
                return Err(EtsError::InvalidParameter {
                    name: "beta".into(),
                    value: b,
                    requirement: "the trend smoothing parameter must lie in [0, alpha] \
                                  (Hyndman's beta, not beta* = beta / alpha)",
                });
            }
        }
        if let Some(g) = self.gamma {
            if !(g >= 0.0 && g <= 1.0 - self.alpha) {
                return Err(EtsError::InvalidParameter {
                    name: "gamma".into(),
                    value: g,
                    requirement: "the seasonal smoothing parameter must lie in \
                                  [0, 1 - alpha] (Hyndman's gamma, not gamma* = gamma / \
                                  (1 - alpha))",
                });
            }
        }
        if let Some(p) = self.phi {
            if !(p > 0.0 && p <= 1.0) {
                return Err(EtsError::InvalidParameter {
                    name: "phi".into(),
                    value: p,
                    requirement: "the damping parameter must lie in (0, 1]",
                });
            }
        }
        Ok(())
    }

    /// The effective `(alpha, beta, gamma, phi)` with absent components as
    /// `0`, `0`, `1`.
    pub(crate) fn effective(&self) -> (f64, f64, f64, f64) {
        (
            self.alpha,
            self.beta.unwrap_or(0.0),
            self.gamma.unwrap_or(0.0),
            self.phi.unwrap_or(1.0),
        )
    }
}

/// The state vector of an ETS model at one point in time: the level, the
/// trend (when present) and the `m` seasonal states (when present).
///
/// **Seasonal ordering convention (time order):** `seasonal[j]` is the
/// seasonal state in force for the `j`-th observation *after* the anchor
/// — for the initial state, the state applied to observation `j`
/// (Hyndman's `s_{j-m}`); for the final state, the state applied to the
/// `j`-th forecast step. This is the order statsmodels' `initial_seasonal`
/// argument and R's `ets` initial states use.
#[derive(Debug, Clone, PartialEq)]
pub struct EtsStates {
    /// Level.
    pub level: f64,
    /// Trend (`None` without a trend component).
    pub trend: Option<f64>,
    /// The `m` seasonal states in time order (`None` without a seasonal).
    pub seasonal: Option<Vec<f64>>,
}

impl EtsStates {
    /// Unpacks `[level, trend?, seasonal[0..m)?]` for `spec`, checking
    /// finiteness (and positivity of the states a multiplicative component
    /// divides by).
    ///
    /// # Errors
    ///
    /// [`EtsError::DimensionMismatch`] or [`EtsError::InvalidParameter`].
    pub fn from_slice(spec: &EtsSpec, v: &[f64]) -> Result<Self, EtsError> {
        let n = spec.n_initial_states();
        if v.len() != n {
            return Err(EtsError::DimensionMismatch {
                what: "initial_states (the initial states [level, trend?, seasonal[0..m)?] this spec takes)",
                expected: n,
                got: v.len(),
            });
        }
        let names = spec.initial_state_names();
        for (name, &x) in names.iter().zip(v) {
            if !x.is_finite() {
                return Err(EtsError::InvalidParameter {
                    name: format!("initial_states: {name}"),
                    value: x,
                    requirement: "must be finite",
                });
            }
        }
        let level = v[0];
        let mut idx = 1;
        let trend = if spec.has_trend() {
            idx += 1;
            Some(v[1])
        } else {
            None
        };
        let seasonal = if spec.has_seasonal() {
            Some(v[idx..].to_vec())
        } else {
            None
        };
        let s = EtsStates {
            level,
            trend,
            seasonal,
        };
        s.check_domain(spec)?;
        Ok(s)
    }

    /// The packed vector `[level, trend?, seasonal[0..m)?]`.
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = vec![self.level];
        if let Some(b) = self.trend {
            v.push(b);
        }
        if let Some(s) = &self.seasonal {
            v.extend_from_slice(s);
        }
        v
    }

    /// Structural checks against `spec`: presence/length of the trend and
    /// seasonal parts, and positivity where a multiplicative component
    /// divides by the state.
    ///
    /// # Errors
    ///
    /// [`EtsError::DimensionMismatch`] or [`EtsError::InvalidParameter`].
    pub fn check_domain(&self, spec: &EtsSpec) -> Result<(), EtsError> {
        if !self.level.is_finite() {
            return Err(EtsError::InvalidParameter {
                name: "initial level".into(),
                value: self.level,
                requirement: "must be finite",
            });
        }
        match (spec.has_trend(), self.trend) {
            (true, None) => {
                return Err(EtsError::DimensionMismatch {
                    what: "initial trend state (the spec has a trend component)",
                    expected: 1,
                    got: 0,
                })
            }
            (false, Some(_)) => {
                return Err(EtsError::DimensionMismatch {
                    what: "initial trend state (the spec has no trend component)",
                    expected: 0,
                    got: 1,
                })
            }
            (true, Some(b)) => {
                if !b.is_finite() {
                    return Err(EtsError::InvalidParameter {
                        name: "initial trend".into(),
                        value: b,
                        requirement: "must be finite",
                    });
                }
                if spec.trend == Component::Multiplicative && b <= 0.0 {
                    return Err(EtsError::InvalidParameter {
                        name: "initial trend".into(),
                        value: b,
                        requirement: "a multiplicative trend is a growth ratio and must be \
                                      strictly positive",
                    });
                }
            }
            (false, None) => {}
        }
        match (spec.has_seasonal(), &self.seasonal) {
            (true, None) => {
                return Err(EtsError::DimensionMismatch {
                    what: "initial seasonal states (the spec has a seasonal component)",
                    expected: spec.m(),
                    got: 0,
                })
            }
            (false, Some(s)) => {
                return Err(EtsError::DimensionMismatch {
                    what: "initial seasonal states (the spec has no seasonal component)",
                    expected: 0,
                    got: s.len(),
                })
            }
            (true, Some(s)) => {
                if s.len() != spec.m() {
                    return Err(EtsError::DimensionMismatch {
                        what: "initial seasonal states (one per period)",
                        expected: spec.m(),
                        got: s.len(),
                    });
                }
                for (j, &x) in s.iter().enumerate() {
                    if !x.is_finite() {
                        return Err(EtsError::InvalidParameter {
                            name: format!("initial seasonal[{j}]"),
                            value: x,
                            requirement: "must be finite",
                        });
                    }
                    if spec.seasonal == Component::Multiplicative && x <= 0.0 {
                        return Err(EtsError::InvalidParameter {
                            name: format!("initial seasonal[{j}]"),
                            value: x,
                            requirement: "a multiplicative seasonal index must be strictly \
                                          positive",
                        });
                    }
                }
            }
            (false, None) => {}
        }
        if spec.trend == Component::Multiplicative && self.level <= 0.0 {
            return Err(EtsError::InvalidParameter {
                name: "initial level".into(),
                value: self.level,
                requirement: "a multiplicative trend divides by the level, which must be \
                              strictly positive",
            });
        }
        Ok(())
    }
}
