//! Maximum-likelihood estimation of one ETS model.
//!
//! The concentrated Gaussian likelihood of [`crate::filter`] is maximised
//! over the smoothing parameters — and, under the estimated
//! initialisation, the initial states — with `tsecon-optim`: a Nelder-Mead
//! search followed by a BFGS polish (central-difference gradients), the
//! better of the two kept (R's `ets` uses Nelder-Mead alone; statsmodels
//! L-BFGS-B). The smoothing parameters live in the traditional box
//! (`0 < alpha < 1`, `0 < beta < alpha`, `0 < gamma < 1 - alpha`,
//! `0.8 <= phi <= 0.98` — the bounds R's `ets` and statsmodels share;
//! statsmodels' numerical margins `1e-4` are used) through logistic
//! transforms of `alpha`, `beta* = beta / alpha`, `gamma* = gamma / (1 -
//! alpha)` and `phi`; the initial states are unconstrained, the level and
//! every additive state measured in units of `mean(|y|)` so the search is
//! scale-free. The `m` seasonal states carry `m - 1` free parameters (the
//! last is the normalisation: additive indices sum to zero, multiplicative
//! indices average one — R's convention; statsmodels pins one index at
//! `0`/`1` instead, an equivalent identification), and the parameter count
//! reported for the information criteria follows: smoothing parameters,
//! free initial states, and `sigma^2`.

use tsecon_optim::{
    minimize, BfgsOptions, Bounded, LbfgsOptions, Method, NelderMeadOptions, ObjectiveFn,
    OptimizeResult, Transform,
};

use crate::error::EtsError;
use crate::filter::{loglik_from_sums, recurse, smooth, Smoothed};
use crate::init::{heuristic_initial_states, starting_states};
use crate::spec::{Component, ErrorType, EtsParams, EtsSpec, EtsStates};

/// How the initial states are obtained.
#[derive(Debug, Clone, PartialEq)]
pub enum Initialization {
    /// Estimated jointly with the smoothing parameters, starting from the
    /// heuristic (Hyndman et al. 2008 section 2.6.1; R and statsmodels'
    /// default).
    Estimated,
    /// Fixed at the Hyndman (2008) heuristic values
    /// ([`heuristic_initial_states`]); only the smoothing parameters are
    /// estimated.
    Heuristic,
    /// Fixed at user-supplied values; only the smoothing parameters are
    /// estimated.
    Known(EtsStates),
}

impl Initialization {
    /// The name reported in results (`"estimated"`, `"heuristic"`,
    /// `"known"`).
    pub fn name(&self) -> &'static str {
        match self {
            Initialization::Estimated => "estimated",
            Initialization::Heuristic => "heuristic",
            Initialization::Known(_) => "known",
        }
    }
}

/// The numerical search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    /// L-BFGS and Nelder-Mead from the same start, then a BFGS polish from
    /// the better of the two; the best point found is kept (the default —
    /// the ETS likelihood can hold a boundary local optimum at
    /// `alpha -> 0` that traps one search and not the other).
    TwoStage,
    /// Nelder-Mead alone (R's choice).
    NelderMead,
    /// BFGS alone, central-difference gradients.
    Bfgs,
    /// L-BFGS alone, central-difference gradients.
    Lbfgs,
}

impl Optimizer {
    /// The name reported in results.
    pub fn name(self) -> &'static str {
        match self {
            Optimizer::TwoStage => "nelder_mead+bfgs",
            Optimizer::NelderMead => "nelder_mead",
            Optimizer::Bfgs => "bfgs",
            Optimizer::Lbfgs => "lbfgs",
        }
    }
}

/// Options of [`ets_fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct FitOptions {
    /// Initial-state treatment (default [`Initialization::Estimated`]).
    pub initialization: Initialization,
    /// Numerical search (default [`Optimizer::TwoStage`]).
    pub optimizer: Optimizer,
    /// Iteration budget per optimizer stage (`None`: the crate's default,
    /// `2000 + 500 * dim` Nelder-Mead iterations and 500 quasi-Newton
    /// iterations).
    pub max_iter: Option<usize>,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            initialization: Initialization::Estimated,
            optimizer: Optimizer::TwoStage,
            max_iter: None,
        }
    }
}

/// A fitted (or fixed-parameter-evaluated) ETS model.
#[derive(Debug, Clone, PartialEq)]
pub struct EtsFit {
    /// The specification.
    pub spec: EtsSpec,
    /// Smoothing parameters.
    pub params: EtsParams,
    /// Initial states (seasonal indices in time order; additive indices
    /// sum to zero, multiplicative ones average one under the estimated
    /// initialisation).
    pub initial_state: EtsStates,
    /// How the initial states were obtained.
    pub initialization: &'static str,
    /// One-step-ahead fitted values.
    pub fitted: Vec<f64>,
    /// Residuals (`y - f`, or `(y - f) / f` under multiplicative errors).
    pub resid: Vec<f64>,
    /// Level path.
    pub level: Vec<f64>,
    /// Trend path (`None` without a trend).
    pub trend: Option<Vec<f64>>,
    /// Seasonal path — the state updated at each `t` (`None` without a
    /// seasonal).
    pub seasonal: Option<Vec<f64>>,
    /// The state after the last observation (forecast anchor).
    pub final_state: EtsStates,
    /// Concentrated Gaussian log-likelihood.
    pub loglik: f64,
    /// `sigma^2 = mean(resid^2)`.
    pub sigma2: f64,
    /// Number of observations.
    pub nobs: usize,
    /// Parameters counted by the information criteria: smoothing
    /// parameters, free initial states (estimated initialisation only) and
    /// `sigma^2`.
    pub k_params: usize,
    /// `-2 loglik + 2 k`.
    pub aic: f64,
    /// `aic + 2 k (k + 1) / (n - k - 1)`.
    pub aicc: f64,
    /// `-2 loglik + k ln n`.
    pub bic: f64,
    /// Whether the search converged (a stage met its own stopping rule);
    /// `true` for a fixed-parameter evaluation.
    pub converged: bool,
    /// Optimizer iterations (all stages).
    pub n_iterations: usize,
    /// Objective evaluations (all stages).
    pub n_fevals: usize,
    /// The optimizer used (`"none"` for a fixed-parameter evaluation).
    pub optimizer: &'static str,
}

const BOX_LO: f64 = 1e-4;
const BOX_HI: f64 = 1.0 - 1e-4;
const PHI_LO: f64 = 0.8;
const PHI_HI: f64 = 0.98;

/// Which working-vector entries are free.
struct Layout {
    spec: EtsSpec,
    estimate_states: bool,
}

impl Layout {
    fn dim(&self) -> usize {
        self.spec.n_smoothing()
            + if self.estimate_states {
                self.spec.n_free_initial_states()
            } else {
                0
            }
    }
}

/// The negative mean log-likelihood in working coordinates, on the scaled
/// data.
struct NegLoglik<'a> {
    layout: Layout,
    y: &'a [f64],
    fixed_init: EtsStates,
    b01: Bounded,
    bphi: Bounded,
    ring: Vec<f64>,
}

impl NegLoglik<'_> {
    fn unpack(&self, z: &[f64]) -> Option<(EtsParams, EtsStates)> {
        let spec = &self.layout.spec;
        let mut k = 0;
        let mut one = [0.0];
        let mut next = |b: &Bounded| -> Option<f64> {
            b.forward(&z[k..k + 1], &mut one).ok()?;
            k += 1;
            Some(one[0])
        };
        let alpha = next(&self.b01)?;
        let beta = if spec.has_trend() {
            Some(alpha * next(&self.b01)?)
        } else {
            None
        };
        let gamma = if spec.has_seasonal() {
            Some((1.0 - alpha) * next(&self.b01)?)
        } else {
            None
        };
        let phi = if spec.damped {
            Some(next(&self.bphi)?)
        } else {
            None
        };
        let params = EtsParams {
            alpha,
            beta,
            gamma,
            phi,
        };
        let init = if self.layout.estimate_states {
            let mut idx = k;
            let level = z[idx];
            idx += 1;
            let trend = if spec.has_trend() {
                let b = z[idx];
                idx += 1;
                Some(b)
            } else {
                None
            };
            let seasonal = if spec.has_seasonal() {
                let m = spec.m();
                let mut s: Vec<f64> = z[idx..idx + m - 1].to_vec();
                let sum: f64 = s.iter().sum();
                let last = match spec.seasonal {
                    Component::Multiplicative => m as f64 - sum,
                    _ => -sum,
                };
                s.push(last);
                Some(s)
            } else {
                None
            };
            EtsStates {
                level,
                trend,
                seasonal,
            }
        } else {
            self.fixed_init.clone()
        };
        Some((params, init))
    }

    fn pack(&self, params: &EtsParams, init: &EtsStates) -> Result<Vec<f64>, EtsError> {
        let spec = &self.layout.spec;
        let mut z = Vec::with_capacity(self.layout.dim());
        let mut one = [0.0];
        let clamp = |v: f64, lo: f64, hi: f64| v.clamp(lo + 1e-6, hi - 1e-6);
        self.b01
            .inverse(&[clamp(params.alpha, BOX_LO, BOX_HI)], &mut one)?;
        z.push(one[0]);
        if let Some(b) = params.beta {
            let bs = if params.alpha > 0.0 {
                b / params.alpha
            } else {
                0.5
            };
            self.b01.inverse(&[clamp(bs, BOX_LO, BOX_HI)], &mut one)?;
            z.push(one[0]);
        }
        if let Some(g) = params.gamma {
            let gs = if params.alpha < 1.0 {
                g / (1.0 - params.alpha)
            } else {
                0.5
            };
            self.b01.inverse(&[clamp(gs, BOX_LO, BOX_HI)], &mut one)?;
            z.push(one[0]);
        }
        if let Some(p) = params.phi {
            self.bphi.inverse(&[clamp(p, PHI_LO, PHI_HI)], &mut one)?;
            z.push(one[0]);
        }
        if self.layout.estimate_states {
            z.push(init.level);
            if let Some(b) = init.trend {
                z.push(b);
            }
            if let Some(s) = &init.seasonal {
                let m = spec.m();
                z.extend_from_slice(&s[..m - 1]);
            }
        }
        Ok(z)
    }
}

impl ObjectiveFn for NegLoglik<'_> {
    fn value(&mut self, z: &[f64]) -> f64 {
        let Some((params, init)) = self.unpack(z) else {
            return f64::INFINITY;
        };
        let spec = self.layout.spec;
        if spec.trend == Component::Multiplicative
            && (init.level <= 0.0 || init.trend.is_some_and(|b| b <= 0.0))
        {
            return f64::INFINITY;
        }
        if spec.seasonal == Component::Multiplicative
            && init
                .seasonal
                .as_ref()
                .is_some_and(|s| s.iter().any(|&v| v <= 0.0))
        {
            return f64::INFINITY;
        }
        let Ok((sums, _)) = recurse(&spec, self.y, &params, &init, &mut self.ring, None) else {
            return f64::INFINITY;
        };
        match loglik_from_sums(self.y.len(), sums, spec.error == ErrorType::Multiplicative) {
            Some(ll) => -ll / self.y.len() as f64,
            None => f64::INFINITY,
        }
    }
}

/// Scales the additive states of `st` by `c` (multiplicative ratios are
/// scale-free).
fn scale_states(spec: &EtsSpec, st: &mut EtsStates, c: f64) {
    st.level *= c;
    if spec.trend == Component::Additive {
        if let Some(b) = st.trend.as_mut() {
            *b *= c;
        }
    }
    if spec.seasonal == Component::Additive {
        if let Some(s) = st.seasonal.as_mut() {
            s.iter_mut().for_each(|v| *v *= c);
        }
    }
}

fn pick_best(a: Option<OptimizeResult>, b: Option<OptimizeResult>) -> Option<OptimizeResult> {
    match (a, b) {
        (Some(x), Some(y)) => {
            let (best, other) = if y.f < x.f { (y, x) } else { (x, y) };
            Some(OptimizeResult {
                converged: best.converged || other.converged,
                iterations: best.iterations + other.iterations,
                fevals: best.fevals + other.fevals,
                gevals: best.gevals + other.gevals,
                ..best
            })
        }
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

fn run_search(
    obj: &mut NegLoglik<'_>,
    z0: &[f64],
    optimizer: Optimizer,
    max_iter: Option<usize>,
) -> Result<OptimizeResult, EtsError> {
    let dim = z0.len();
    let nm = Method::NelderMead(NelderMeadOptions {
        max_iter: Some(max_iter.unwrap_or(2000 + 500 * dim)),
        ..NelderMeadOptions::default()
    });
    // 1e-6 on the per-observation objective: the gradients are central
    // differences, whose noise floor sits near 1e-8, so the default 1e-8
    // flags correct optima as unconverged.
    let bfgs = Method::Bfgs(BfgsOptions {
        grad_tol: 1e-6,
        max_iter: Some(max_iter.unwrap_or(500)),
        ..BfgsOptions::default()
    });
    let lbfgs = Method::Lbfgs(LbfgsOptions {
        grad_tol: 1e-6,
        max_iter: Some(max_iter.unwrap_or(500)),
        ..LbfgsOptions::default()
    });
    let finite = |r: Result<OptimizeResult, _>| r.ok().filter(|r| r.f.is_finite());
    let res = match optimizer {
        Optimizer::TwoStage => {
            let quasi = finite(minimize(obj, z0, &lbfgs));
            let simplex = finite(minimize(obj, z0, &nm));
            let first = pick_best(quasi, simplex);
            let from = first.as_ref().map_or(z0, |r| r.x.as_slice()).to_vec();
            let second = finite(minimize(obj, &from, &bfgs));
            pick_best(first, second)
        }
        Optimizer::NelderMead => finite(minimize(obj, z0, &nm)),
        Optimizer::Bfgs => finite(minimize(obj, z0, &bfgs)),
        Optimizer::Lbfgs => finite(minimize(obj, z0, &lbfgs)),
    };
    res.ok_or(EtsError::Degenerate {
        what: "the likelihood search found no point with a finite likelihood \
               (the starting states are degenerate for these data)",
        index: 0,
    })
}

fn information_criteria(loglik: f64, n: usize, k: usize) -> (f64, f64, f64) {
    let nf = n as f64;
    let kf = k as f64;
    let aic = -2.0 * loglik + 2.0 * kf;
    let aicc = if nf - kf - 1.0 > 0.0 {
        aic + 2.0 * kf * (kf + 1.0) / (nf - kf - 1.0)
    } else {
        f64::INFINITY
    };
    let bic = -2.0 * loglik + kf * nf.ln();
    (aic, aicc, bic)
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    spec: EtsSpec,
    params: EtsParams,
    init: EtsStates,
    sm: Smoothed,
    initialization: &'static str,
    k_params: usize,
    search: Option<&OptimizeResult>,
    optimizer: &'static str,
) -> EtsFit {
    let n = sm.fitted.len();
    let (aic, aicc, bic) = information_criteria(sm.loglik, n, k_params);
    EtsFit {
        spec,
        params,
        initial_state: init,
        initialization,
        fitted: sm.fitted,
        resid: sm.resid,
        level: sm.level,
        trend: sm.trend,
        seasonal: sm.seasonal,
        final_state: sm.final_state,
        loglik: sm.loglik,
        sigma2: sm.sigma2,
        nobs: n,
        k_params,
        aic,
        aicc,
        bic,
        converged: search.is_none_or(|r| r.converged),
        n_iterations: search.map_or(0, |r| r.iterations),
        n_fevals: search.map_or(0, |r| r.fevals),
        optimizer,
    }
}

/// Evaluates `spec` on `y` at fixed smoothing parameters and initial
/// states — no optimisation (statsmodels `ETSModel(...).smooth(params)`
/// with a known initialisation). The information criteria count the
/// smoothing parameters and `sigma^2` (the states are given).
///
/// # Errors
///
/// As for [`crate::smooth`].
pub fn ets_at(
    spec: &EtsSpec,
    y: &[f64],
    params: &EtsParams,
    init: &EtsStates,
) -> Result<EtsFit, EtsError> {
    let sm = smooth(spec, y, params, init)?;
    let k = spec.n_smoothing() + 1;
    Ok(assemble(
        *spec,
        params.clone(),
        init.clone(),
        sm,
        "known",
        k,
        None,
        "none",
    ))
}

/// Fits `spec` to `y` by maximum likelihood (see the module docs).
///
/// # Errors
///
/// * [`EtsError::InvalidSpec`], [`EtsError::NonFinite`],
///   [`EtsError::NonPositiveData`] for a bad specification or data;
/// * [`EtsError::TooFewObservations`] when `n < k + 2` (the AICc needs
///   `n - k - 1 > 0`) or the initialisation heuristic needs more data;
/// * [`EtsError::Degenerate`] when no parameter point has a finite
///   likelihood (e.g. a constant series).
pub fn ets_fit(spec: &EtsSpec, y: &[f64], opts: &FitOptions) -> Result<EtsFit, EtsError> {
    spec.validate()?;
    crate::filter::check_data(spec, y)?;
    let n = y.len();
    let estimate_states = matches!(opts.initialization, Initialization::Estimated);
    let k_params = spec.n_smoothing()
        + if estimate_states {
            spec.n_free_initial_states()
        } else {
            0
        }
        + 1;
    if n < k_params + 2 {
        return Err(EtsError::TooFewObservations {
            needed: k_params + 2,
            got: n,
            what: format!(
                "fitting {} with {k_params} parameters (the AICc needs n - k - 1 > 0)",
                spec.name()
            ),
        });
    }
    // Starting / fixed states in original units.
    let start_states = match &opts.initialization {
        Initialization::Estimated => starting_states(spec, y)?,
        Initialization::Heuristic => heuristic_initial_states(spec, y)?,
        Initialization::Known(st) => {
            st.check_domain(spec)?;
            st.clone()
        }
    };
    // Scale-free search: additive states in units of mean |y|.
    let mean_abs = y.iter().map(|v| v.abs()).sum::<f64>() / n as f64;
    let c = if mean_abs.is_finite() && mean_abs > 0.0 {
        mean_abs
    } else {
        1.0
    };
    let ys: Vec<f64> = y.iter().map(|v| v / c).collect();
    let mut scaled_start = start_states.clone();
    scale_states(spec, &mut scaled_start, 1.0 / c);

    let start_params = EtsParams {
        alpha: 0.1,
        beta: spec.has_trend().then_some(0.01),
        gamma: spec.has_seasonal().then_some(0.01),
        phi: spec.damped.then_some(0.9782),
    };
    // Staged start for the joint problem: first the smoothing parameters
    // alone with the states held at the heuristic (a robust problem in at
    // most four dimensions), then the joint search from that point. Without
    // it the joint search can climb the alpha -> 1 ridge and miss the
    // interior optimum (observed on ETS(M,N,A) for the airline series).
    let start_params = if estimate_states {
        let mut pre = NegLoglik {
            layout: Layout {
                spec: *spec,
                estimate_states: false,
            },
            y: &ys,
            fixed_init: scaled_start.clone(),
            b01: Bounded::new(BOX_LO, BOX_HI)?,
            bphi: Bounded::new(PHI_LO, PHI_HI)?,
            ring: Vec::with_capacity(spec.m()),
        };
        let z0 = pre.pack(&start_params, &scaled_start)?;
        let nm = Method::NelderMead(NelderMeadOptions {
            max_iter: Some(2000),
            ..NelderMeadOptions::default()
        });
        match minimize(&mut pre, &z0, &nm)
            .ok()
            .filter(|r| r.f.is_finite())
        {
            Some(r) => pre.unpack(&r.x).map_or(start_params.clone(), |(p, _)| p),
            None => start_params,
        }
    } else {
        start_params
    };
    let layout = Layout {
        spec: *spec,
        estimate_states,
    };
    let mut obj = NegLoglik {
        layout,
        y: &ys,
        fixed_init: scaled_start.clone(),
        b01: Bounded::new(BOX_LO, BOX_HI)?,
        bphi: Bounded::new(PHI_LO, PHI_HI)?,
        ring: Vec::with_capacity(spec.m()),
    };
    let z0 = obj.pack(&start_params, &scaled_start)?;
    let search = run_search(&mut obj, &z0, opts.optimizer, opts.max_iter)?;
    let (params, mut init) = obj.unpack(&search.x).ok_or(EtsError::Degenerate {
        what: "the optimizer returned a non-finite point",
        index: 0,
    })?;
    if estimate_states {
        scale_states(spec, &mut init, c);
    } else {
        // Fixed states are reported exactly as supplied / computed, not
        // through the scale-and-unscale round trip.
        init = start_states;
    }
    let sm = smooth(spec, y, &params, &init)?;
    Ok(assemble(
        *spec,
        params,
        init,
        sm,
        match &opts.initialization {
            Initialization::Estimated => "estimated",
            Initialization::Heuristic => "heuristic",
            Initialization::Known(_) => "known",
        },
        k_params,
        Some(&search),
        opts.optimizer.name(),
    ))
}
