//! Automatic ARIMA order selection — the Hyndman-Khandakar (2008)
//! stepwise algorithm (`forecast::auto.arima`), built as a *composition*
//! over pieces that already ship: `d` from successive KPSS tests
//! ([`tsecon_diag::ndiffs`]), `D` from the STL seasonal-strength rule
//! ([`tsecon_diag::nsdiffs`]), and every candidate fit through
//! [`ArimaSpec::fit`] — exact Gaussian MLE on the state-space engine.
//! Nothing in this module estimates anything new; it only decides *which*
//! already-validated fit to report.
//!
//! The honest validation grade (stated up front, and in the model card):
//! the *candidate fits* are golden-pinned to statsmodels, but the
//! *selection loop itself* has no gating third-party reference. R's
//! `forecast::auto.arima` and Python's `pmdarima` famously disagree with
//! each other on real series (different fallback estimators, different
//! failure handling), so chasing either would pin an implementation
//! accident. The loop is graded on **Monte-Carlo order recovery** on
//! simulated DGPs with known orders, plus internal-consistency
//! invariants: the search is deterministic, every reported IC is
//! reproducible by refitting the reported order, and the best IC is the
//! minimum over the trace.

use core::fmt;

use tsecon_diag::{ndiffs, nsdiffs, NdiffsResult, NdiffsTest, NsdiffsResult};

use crate::error::ArimaError;
use crate::results::ArimaResults;
use crate::spec::ArimaSpec;

/// Which information criterion drives the selection.
///
/// All three share the statsmodels parameter count `k` (constant + AR +
/// MA + seasonal AR + seasonal MA + `sigma2`) and the effective sample
/// size `n` = observations remaining after differencing — the sample the
/// likelihood is actually computed on. Comparing criteria across
/// *different* `(d, D)` is meaningless (different data), which is exactly
/// why Hyndman-Khandakar fixes the differencing orders *before* the
/// order search; this module does the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionIc {
    /// Corrected AIC, `AIC + 2k(k+1)/(n - k - 1)` (Hurvich & Tsai 1989)
    /// — the `forecast::auto.arima` default; the correction matters
    /// precisely in the short samples where automatic selection is most
    /// tempted to overfit. Undefined (treated as inadmissible) when
    /// `n <= k + 1`.
    Aicc,
    /// Akaike information criterion `-2 loglik + 2k`.
    Aic,
    /// Bayesian information criterion `-2 loglik + k ln n`.
    Bic,
}

impl SelectionIc {
    /// The short code used in reports (`"aicc"`, `"aic"`, `"bic"`).
    pub fn code(self) -> &'static str {
        match self {
            SelectionIc::Aicc => "aicc",
            SelectionIc::Aic => "aic",
            SelectionIc::Bic => "bic",
        }
    }

    /// The criterion value for a completed fit; `None` when the AICc
    /// small-sample denominator `n - k - 1` is not strictly positive
    /// (the model has as many parameters as observations — inadmissible
    /// rather than a division blow-up).
    pub fn evaluate(self, res: &ArimaResults) -> Option<f64> {
        let n = res.nobs as f64;
        let k = res.k_params as f64;
        match self {
            SelectionIc::Aic => Some(res.aic),
            SelectionIc::Bic => Some(res.bic),
            SelectionIc::Aicc => {
                let denom = n - k - 1.0;
                if denom > 0.0 {
                    Some(res.aic + 2.0 * k * (k + 1.0) / denom)
                } else {
                    None
                }
            }
        }
    }
}

/// Options for [`auto_arima`], defaulting to the `forecast::auto.arima`
/// settings wherever the engine supports them.
///
/// Build with [`AutoArimaOptions::default`] and override fields:
///
/// ```
/// use tsecon_arima::AutoArimaOptions;
/// let opts = AutoArimaOptions {
///     seasonal_period: 12,
///     ..AutoArimaOptions::default()
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AutoArimaOptions {
    /// Seasonal period `s`: `0` for a non-seasonal search (the default),
    /// `>= 2` to search seasonal orders too. `1` is refused (a "season"
    /// of one observation is not a season).
    pub seasonal_period: usize,
    /// The information criterion minimized ([`SelectionIc::Aicc`] by
    /// default, matching R).
    pub ic: SelectionIc,
    /// `true` (default): the Hyndman-Khandakar stepwise neighborhood
    /// search. `false`: exhaustive grid over every admissible
    /// `(p, q, P, Q, constant)` — only sensible for small caps, and
    /// refused (with the count named) when the grid would exceed
    /// [`MAX_GRID_MODELS`] fits.
    pub stepwise: bool,
    /// Cap on the AR order `p` (default 5, R's `max.p`).
    pub max_p: usize,
    /// Cap on the MA order `q` (default 5, R's `max.q`).
    pub max_q: usize,
    /// Cap on the seasonal AR order `P` (default 2, R's `max.P`).
    pub max_seasonal_p: usize,
    /// Cap on the seasonal MA order `Q` (default 2, R's `max.Q`).
    pub max_seasonal_q: usize,
    /// Cap on `p + q + P + Q` (default 5, R's `max.order`). Exactly as
    /// in R, this cap applies **only to the exhaustive grid**
    /// (`stepwise: false`); the stepwise search ignores it — its own
    /// starting model `(2, d, 2)(1, D, 1)` already sums to 6.
    pub max_order: usize,
    /// Cap handed to the KPSS `d` sequence (default 2, R's `max.d`).
    pub max_d: usize,
    /// Cap handed to the seasonal-strength `D` sequence (default 1,
    /// R's `max.D`).
    pub max_seasonal_d: usize,
    /// Fix `d` instead of testing for it (`None`, the default, runs
    /// [`tsecon_diag::ndiffs`] with KPSS).
    pub fixed_d: Option<usize>,
    /// Fix `D` instead of testing for it (`None`, the default, runs
    /// [`tsecon_diag::nsdiffs`] when `seasonal_period >= 2`).
    pub fixed_seasonal_d: Option<usize>,
    /// Significance level for the KPSS sequence choosing `d` (default
    /// 0.05; the seasonal-strength rule choosing `D` is threshold-based
    /// and does not use it).
    pub alpha: f64,
    /// Stepwise safety budget: the search stops after this many candidate
    /// *fits* (default 94, R's `nmodels`) and reports
    /// [`AutoArimaResult::budget_exhausted`] instead of looping forever
    /// on a flat criterion surface.
    pub max_models: usize,
}

impl Default for AutoArimaOptions {
    fn default() -> Self {
        Self {
            seasonal_period: 0,
            ic: SelectionIc::Aicc,
            stepwise: true,
            max_p: 5,
            max_q: 5,
            max_seasonal_p: 2,
            max_seasonal_q: 2,
            max_order: 5,
            max_d: 2,
            max_seasonal_d: 1,
            fixed_d: None,
            fixed_seasonal_d: None,
            alpha: 0.05,
            max_models: 94,
        }
    }
}

/// Hard sanity cap on `max_p` / `max_q` (searching AR/MA orders beyond
/// this is never what a user means, and the root-admissibility guard is
/// only battle-tested for small degrees).
pub const MAX_REGULAR_CAP: usize = 12;
/// Hard sanity cap on `max_seasonal_p` / `max_seasonal_q`.
pub const MAX_SEASONAL_CAP: usize = 6;
/// The exhaustive grid (`stepwise: false`) refuses to fit more models
/// than this — use the stepwise search or lower the caps instead.
pub const MAX_GRID_MODELS: usize = 512;
/// A fitted model whose AR or MA polynomial (regular, or seasonal after
/// the `1/s` power-mapping of its root moduli) has a root with modulus
/// below this is *inadmissible*: it sits numerically on the unit circle,
/// its forecasts are fragile, and its near-cancelling-root cousins
/// produce spuriously good likelihoods. The value mirrors
/// `forecast::auto.arima`'s check that no inverse root exceeds `1/1.001`.
pub const ROOT_ADMISSIBILITY_THRESHOLD: f64 = 1.001;

/// Why a visited candidate did or did not become eligible for selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateStatus {
    /// The fit succeeded and the model is admissible; the recorded IC is
    /// finite and comparable.
    Ok,
    /// The fit succeeded but a fitted AR or MA root (regular or
    /// seasonal) has modulus below [`ROOT_ADMISSIBILITY_THRESHOLD`]:
    /// the model sits on the unit circle and is excluded from selection.
    NearUnitRoot,
    /// The fit succeeded but the AICc denominator `n - k - 1` is not
    /// strictly positive, so the criterion is undefined; excluded.
    IcUndefined,
    /// The fit itself failed (too few observations for the orders, or no
    /// finite likelihood was found); excluded.
    FitFailed,
}

impl CandidateStatus {
    /// The short code used in reports.
    pub fn code(self) -> &'static str {
        match self {
            CandidateStatus::Ok => "ok",
            CandidateStatus::NearUnitRoot => "near_unit_root",
            CandidateStatus::IcUndefined => "ic_undefined",
            CandidateStatus::FitFailed => "fit_failed",
        }
    }
}

/// One entry of the search trace: a candidate the search actually fit
/// (or tried to), with the criterion it scored. `d`, `D` and `s` are
/// shared by every candidate (fixed before the search) and live on
/// [`AutoArimaResult`], not here.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoArimaCandidate {
    /// AR order `p`.
    pub p: usize,
    /// MA order `q`.
    pub q: usize,
    /// Seasonal AR order `P` (always 0 in a non-seasonal search).
    pub seasonal_p: usize,
    /// Seasonal MA order `Q` (always 0 in a non-seasonal search).
    pub seasonal_q: usize,
    /// Whether the candidate includes a constant.
    pub constant: bool,
    /// The criterion value ([`AutoArimaResult::ic`] says which), present
    /// exactly when `status` is [`CandidateStatus::Ok`] or
    /// [`CandidateStatus::NearUnitRoot`] (recorded for the trace but not
    /// eligible in the latter case).
    pub ic: Option<f64>,
    /// Whether the candidate was eligible, and if not, why.
    pub status: CandidateStatus,
    /// The error message when `status` is [`CandidateStatus::FitFailed`].
    pub error: Option<String>,
}

/// The result of [`auto_arima`]: the selected fitted model, the evidence
/// that chose `d` and `D`, and the full search trace.
#[derive(Debug, Clone)]
pub struct AutoArimaResult {
    /// The selected model, fitted by exact MLE — the same object
    /// [`ArimaSpec::fit`] returns, so forecasting, residuals, and the
    /// parameter covariance all work on it directly. The selected orders
    /// are in `best.spec`.
    pub best: ArimaResults,
    /// The criterion value of the selected model (the minimum over every
    /// eligible candidate in `trace`).
    pub best_ic: f64,
    /// Which criterion was minimized.
    pub ic: SelectionIc,
    /// Whether the stepwise search (`true`) or the exhaustive grid
    /// (`false`) produced the result.
    pub stepwise: bool,
    /// The differencing order shared by every candidate.
    pub d: usize,
    /// The seasonal differencing order shared by every candidate.
    pub seasonal_d: usize,
    /// The seasonal period of the search (0 = non-seasonal).
    pub seasonal_period: usize,
    /// The KPSS evidence behind `d` (`None` when `d` was fixed by the
    /// caller).
    pub d_evidence: Option<NdiffsResult>,
    /// The seasonal-strength evidence behind `D` (`None` when `D` was
    /// fixed or the search is non-seasonal).
    pub seasonal_d_evidence: Option<NsdiffsResult>,
    /// Every candidate visited, in visit order (the audit trail: the
    /// reported best is the argmin over the eligible entries, and
    /// refitting any entry's orders reproduces its IC).
    pub trace: Vec<AutoArimaCandidate>,
    /// Number of candidate fits attempted (= `trace.len()`).
    pub n_models: usize,
    /// `true` when the stepwise search stopped on the
    /// [`AutoArimaOptions::max_models`] budget rather than on
    /// convergence; the reported best is the best found, not a certified
    /// local optimum of the move set.
    pub budget_exhausted: bool,
    /// Plain-language summary of what was selected and why.
    pub interpretation: String,
}

impl fmt::Display for AutoArimaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let spec = &self.best.spec;
        write!(
            f,
            "auto_arima: ARIMA({},{},{})",
            spec.p(),
            spec.d(),
            spec.q()
        )?;
        if self.seasonal_period >= 2 {
            write!(
                f,
                "({},{},{})[{}]",
                spec.seasonal_p(),
                spec.seasonal_d(),
                spec.seasonal_q(),
                self.seasonal_period
            )?;
        }
        write!(
            f,
            "{} — {} = {:.4}, {} models — {}",
            if spec.include_constant() {
                " with constant"
            } else {
                ""
            },
            self.ic.code(),
            self.best_ic,
            self.n_models,
            self.interpretation
        )
    }
}

// ------------------------------------------------------------ root guard

/// Minimum modulus of the roots of `P(z) = 1 + a_1 z + ... + a_n z^n`,
/// or `None` when the effective degree is zero (no roots — vacuously
/// admissible). Store AR coefficients as `a_k = -phi_k` and MA
/// coefficients as `a_k = +theta_k` (both polynomials are `1 -+ ...` in
/// the statsmodels sign conventions this crate uses).
///
/// Roots come from the Durand-Kerner (Weierstrass 1891 / Durand 1960 /
/// Kerner 1966) simultaneous iteration on the monic form — fully
/// deterministic (fixed starting configuration, fixed iteration budget),
/// which the search's determinism contract requires. Trailing
/// coefficients below `1e-10` in magnitude are trimmed first: they only
/// contribute roots of enormous modulus, which can never be the minimum,
/// and their presence makes the monic normalization ill-conditioned.
fn min_root_modulus(a: &[f64]) -> Option<f64> {
    let mut n = a.len();
    while n > 0 && a[n - 1].abs() < 1e-10 {
        n -= 1;
    }
    if n == 0 {
        return None;
    }
    let a = &a[..n];
    if n == 1 {
        // 1 + a_1 z = 0  =>  z = -1/a_1.
        return Some(1.0 / a[0].abs());
    }

    // Monic form: z^n + b_{n-1} z^{n-1} + ... + b_0, b_k = a_k / a_n
    // (with a_0 = 1 for the constant term).
    let an = a[n - 1];
    let mut b = vec![0.0; n];
    b[0] = 1.0 / an;
    for k in 1..n {
        b[k] = a[k - 1] / an;
    }

    // Complex helpers on (re, im) pairs.
    type C = (f64, f64);
    let mul = |x: C, y: C| -> C { (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0) };
    let sub = |x: C, y: C| -> C { (x.0 - y.0, x.1 - y.1) };
    let div = |x: C, y: C| -> C {
        let d = y.0 * y.0 + y.1 * y.1;
        ((x.0 * y.0 + x.1 * y.1) / d, (x.1 * y.0 - x.0 * y.1) / d)
    };
    let abs2 = |x: C| -> f64 { x.0 * x.0 + x.1 * x.1 };
    // Monic polynomial value at z (Horner).
    let eval = |z: C| -> C {
        let mut v: C = (1.0, 0.0);
        for k in (0..n).rev() {
            v = mul(v, z);
            v = (v.0 + b[k], v.1);
        }
        v
    };

    // Durand-Kerner from the standard non-real, non-equimodular start
    // (0.4 + 0.9i)^{k+1}.
    let seed: C = (0.4, 0.9);
    let mut x: Vec<C> = Vec::with_capacity(n);
    let mut acc: C = (1.0, 0.0);
    for _ in 0..n {
        acc = mul(acc, seed);
        x.push(acc);
    }
    for _ in 0..500 {
        let mut delta_max = 0.0f64;
        for i in 0..n {
            let mut denom: C = (1.0, 0.0);
            for j in 0..n {
                if j != i {
                    denom = mul(denom, sub(x[i], x[j]));
                }
            }
            if abs2(denom) == 0.0 {
                // Two iterates collided (measure-zero with this start):
                // nudge deterministically and continue.
                x[i] = (x[i].0 + 1e-6, x[i].1 + 1e-6);
                continue;
            }
            let step = div(eval(x[i]), denom);
            x[i] = sub(x[i], step);
            delta_max = delta_max.max(abs2(step).sqrt());
        }
        if delta_max < 1e-13 {
            break;
        }
    }
    x.into_iter()
        .map(|z| abs2(z).sqrt())
        .fold(None, |m: Option<f64>, v| {
            Some(m.map_or(v, |mv| mv.min(v)))
        })
}

/// Whether a completed fit is admissible: every fitted AR and MA root
/// (regular polynomials directly; seasonal polynomials through the
/// `1/s`-power mapping of `Phi(L^s)`'s roots onto `Phi(u)`'s) has
/// modulus at least [`ROOT_ADMISSIBILITY_THRESHOLD`]. Checking the
/// factor polynomials separately is *exactly* equivalent to checking the
/// multiplied-out polynomial: the roots of `phi(L)Phi(L^s)` are the
/// roots of `phi` together with the `s`-th roots of `Phi(u)`'s roots.
fn roots_admissible(res: &ArimaResults) -> bool {
    let s = res.spec.period();
    let neg = |c: &[f64]| -> Vec<f64> { c.iter().map(|v| -v).collect() };
    let regular_ok = |coefs: &[f64]| -> bool {
        min_root_modulus(coefs).is_none_or(|m| m >= ROOT_ADMISSIBILITY_THRESHOLD)
    };
    let seasonal_ok = |coefs: &[f64]| -> bool {
        min_root_modulus(coefs)
            .is_none_or(|m| m.powf(1.0 / s as f64) >= ROOT_ADMISSIBILITY_THRESHOLD)
    };
    regular_ok(&neg(res.ar()))
        && regular_ok(res.ma())
        && seasonal_ok(&neg(res.seasonal_ar()))
        && seasonal_ok(res.seasonal_ma())
}

// ------------------------------------------------------------ the search

/// A candidate's free orders (the search key); `d`, `D`, `s` are fixed
/// for the whole search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Key {
    p: usize,
    q: usize,
    sp: usize,
    sq: usize,
    constant: bool,
}

struct Search<'a> {
    y: &'a [f64],
    d: usize,
    seasonal_d: usize,
    s: usize,
    ic: SelectionIc,
    max_models: usize,
    visited: Vec<Key>,
    trace: Vec<AutoArimaCandidate>,
    best: Option<(Key, ArimaResults, f64)>,
    budget_exhausted: bool,
}

impl Search<'_> {
    fn spec_for(&self, key: Key) -> Result<ArimaSpec, ArimaError> {
        let spec = ArimaSpec::new(key.p, self.d, key.q)?.with_constant(key.constant);
        if self.s >= 2 {
            spec.seasonal(key.sp, self.seasonal_d, key.sq, self.s)
        } else {
            Ok(spec)
        }
    }

    /// Fit and score one candidate (skipping keys already visited),
    /// recording it in the trace; returns `true` when it became the new
    /// best. Every failure mode is recorded, never propagated: a
    /// candidate that cannot be fit is simply not selectable.
    fn consider(&mut self, key: Key) -> bool {
        if self.visited.contains(&key) {
            return false;
        }
        if self.trace.len() >= self.max_models {
            self.budget_exhausted = true;
            return false;
        }
        self.visited.push(key);
        let mut entry = AutoArimaCandidate {
            p: key.p,
            q: key.q,
            seasonal_p: key.sp,
            seasonal_q: key.sq,
            constant: key.constant,
            ic: None,
            status: CandidateStatus::FitFailed,
            error: None,
        };
        let fitted = self
            .spec_for(key)
            .and_then(|spec| spec.fit(self.y));
        let improved = match fitted {
            Err(e) => {
                entry.error = Some(e.to_string());
                false
            }
            Ok(res) => match self.ic.evaluate(&res) {
                None => {
                    entry.status = CandidateStatus::IcUndefined;
                    false
                }
                Some(ic) if !ic.is_finite() => {
                    entry.status = CandidateStatus::IcUndefined;
                    false
                }
                Some(ic) => {
                    if roots_admissible(&res) {
                        entry.status = CandidateStatus::Ok;
                        entry.ic = Some(ic);
                        let better = self
                            .best
                            .as_ref()
                            .is_none_or(|(_, _, best_ic)| ic < *best_ic);
                        if better {
                            self.best = Some((key, res, ic));
                        }
                        better
                    } else {
                        entry.status = CandidateStatus::NearUnitRoot;
                        entry.ic = Some(ic);
                        false
                    }
                }
            },
        };
        self.trace.push(entry);
        improved
    }
}

/// The stepwise neighborhood of the current best: each of `p`, `q`, `P`,
/// `Q` varied by +-1; `p` and `q` varied together; `P` and `Q` varied
/// together; and the constant toggled (when the differencing orders
/// leave a constant meaningful) — the Hyndman-Khandakar (2008, sec. 3.2)
/// move set, in a fixed documented order (regular moves, then seasonal,
/// then the constant), which is part of the determinism contract.
fn neighborhood(key: Key, opts: &AutoArimaOptions, allow_constant: bool) -> Vec<Key> {
    let mut out = Vec::with_capacity(17);
    let deltas: [(isize, isize); 8] = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
    ];
    let shift = |v: usize, dv: isize, cap: usize| -> Option<usize> {
        let nv = v as isize + dv;
        (nv >= 0 && nv as usize <= cap).then_some(nv as usize)
    };
    for (dp, dq) in deltas {
        if let (Some(p), Some(q)) = (shift(key.p, dp, opts.max_p), shift(key.q, dq, opts.max_q)) {
            out.push(Key { p, q, ..key });
        }
    }
    if opts.seasonal_period >= 2 {
        for (dp, dq) in deltas {
            if let (Some(sp), Some(sq)) = (
                shift(key.sp, dp, opts.max_seasonal_p),
                shift(key.sq, dq, opts.max_seasonal_q),
            ) {
                out.push(Key { sp, sq, ..key });
            }
        }
    }
    if allow_constant {
        out.push(Key {
            constant: !key.constant,
            ..key
        });
    }
    out
}

fn validate_options(opts: &AutoArimaOptions) -> Result<(), ArimaError> {
    if opts.seasonal_period == 1 {
        return Err(ArimaError::InvalidArgument {
            what: "seasonal_period = 1 has no meaning (a \"season\" of one observation \
                   is every observation). Pass 0 for a non-seasonal search, or the true \
                   period (12 monthly, 4 quarterly) for a seasonal one",
        });
    }
    if opts.max_p > MAX_REGULAR_CAP || opts.max_q > MAX_REGULAR_CAP {
        return Err(ArimaError::InvalidArgument {
            what: "max_p and max_q must be at most 12: automatic selection over higher \
                   AR/MA orders is never informative (a long-lag fit is either seasonal \
                   structure that belongs in (P, Q) or overfitting), and R's own default \
                   caps are 5",
        });
    }
    if opts.max_seasonal_p > MAX_SEASONAL_CAP || opts.max_seasonal_q > MAX_SEASONAL_CAP {
        return Err(ArimaError::InvalidArgument {
            what: "max_seasonal_p and max_seasonal_q must be at most 6: each seasonal \
                   order multiplies the state dimension by the period, and R's own \
                   default caps are 2",
        });
    }
    if opts.max_models == 0 {
        return Err(ArimaError::InvalidArgument {
            what: "max_models = 0 would forbid fitting any candidate; the R-parity \
                   default is 94",
        });
    }
    if opts.seasonal_period < 2 && opts.fixed_seasonal_d.is_some_and(|v| v > 0) {
        return Err(ArimaError::InvalidArgument {
            what: "a fixed seasonal differencing order D > 0 needs seasonal_period >= 2: \
                   seasonal differencing at no period is undefined",
        });
    }
    Ok(())
}

/// Automatic ARIMA order selection: the Hyndman-Khandakar (2008)
/// algorithm as implemented by `forecast::auto.arima`, on this crate's
/// exact-MLE engine.
///
/// The three stages, exactly as published:
///
/// 1. **`D`** (seasonal searches only): the STL seasonal-strength rule
///    via [`tsecon_diag::nsdiffs`] — one seasonal difference while the
///    Wang-Smith-Hyndman strength is at least 0.64, capped at
///    `max_seasonal_d`.
/// 2. **`d`**: successive KPSS tests via [`tsecon_diag::ndiffs`] *on the
///    seasonally differenced series* — difference while KPSS rejects
///    stationarity at `alpha`, capped at `max_d`. The evidence behind
///    both decisions is returned, not just the integers.
/// 3. **`(p, q, P, Q, constant)`**: minimize the chosen criterion
///    (AICc by default) at the now-fixed `(d, D)` — criteria are not
///    comparable across differencing orders, which is why the
///    differencing decisions come first and are never revisited. The
///    default is the stepwise search from the four Hyndman-Khandakar
///    starting models (`(2,2)(1,1)`, `(0,0)(0,0)`, `(1,0)(1,0)`,
///    `(0,1)(0,1)`, capped by the `max_*` options, plus the no-constant
///    null when a constant is allowed at all), repeatedly moving to the
///    first neighbor that improves the criterion — the Hyndman-Khandakar
///    move set in a fixed documented order: +-1 on each of `p`, `q`,
///    `P`, `Q` alone, `p` and `q` jointly, `P` and `Q` jointly, then
///    the constant toggled — until no neighbor improves or the
///    `max_models` budget is spent. `stepwise: false` fits the
///    full grid subject to `max_order` instead.
///
/// The constant is included in the starting models when `d + D <= 1`
/// (it is a mean for `d + D = 0` and a drift for `d + D = 1`, R's
/// `allowmean` / `allowdrift` defaults) and toggled as a search move;
/// for `d + D >= 2` no candidate carries a constant (a quadratic
/// deterministic trend is almost never what the data mean and R does the
/// same).
///
/// Admissibility guards: a fitted candidate whose AR or MA polynomial
/// has a root with modulus below [`ROOT_ADMISSIBILITY_THRESHOLD`] is
/// recorded in the trace but never selected (near-unit-root fits are
/// numerically fragile and their likelihoods flatter deceptively); a
/// candidate whose fit fails is recorded with its error and skipped —
/// failures steer the search rather than aborting it.
///
/// **No exogenous regressors** in this slice: the underlying engine has
/// no ARIMAX support yet, so there is no `xreg` argument to mirror.
///
/// Every candidate is fit by [`ArimaSpec::fit`] — exact Gaussian MLE,
/// deterministic — so the whole search is deterministic and every trace
/// entry is reproducible by refitting its orders.
///
/// # Errors
///
/// * [`ArimaError::InvalidArgument`] for `seasonal_period = 1`, caps
///   beyond the sanity limits, `max_models = 0`, a fixed `D > 0`
///   without a seasonal period, or (non-stepwise) a grid larger than
///   [`MAX_GRID_MODELS`];
/// * [`ArimaError::NonFinite`] when `y` contains NaN/infinity;
/// * [`ArimaError::Selection`] when the `d`/`D` advisors cannot run
///   (most often: too few observations for KPSS or for an STL fit at
///   the given period), or when **no** candidate produced an admissible
///   fit (the error names how many were tried and why the last one
///   failed).
pub fn auto_arima(y: &[f64], opts: &AutoArimaOptions) -> Result<AutoArimaResult, ArimaError> {
    validate_options(opts)?;
    if let Some(index) = y.iter().position(|v| !v.is_finite()) {
        return Err(ArimaError::NonFinite {
            what: "the series y",
            at: Some(index),
        });
    }
    let s = opts.seasonal_period;

    // --- Stage 1: D from seasonal strength (seasonal searches only). ---
    let (seasonal_d, seasonal_d_evidence) = if s >= 2 {
        match opts.fixed_seasonal_d {
            Some(fixed) => (fixed, None),
            None => {
                let r = nsdiffs(y, s, opts.alpha, opts.max_seasonal_d).map_err(|e| {
                    ArimaError::Selection {
                        what: format!(
                            "choosing the seasonal differencing order D via the \
                             seasonal-strength rule failed: {e}"
                        ),
                    }
                })?;
                let d = r.d;
                (d, Some(r))
            }
        }
    } else {
        (0, None)
    };

    // --- Stage 2: d from successive KPSS on the seasonally
    //     differenced series (the Hyndman-Khandakar order). ---
    let mut ys: Vec<f64> = y.to_vec();
    for _ in 0..seasonal_d {
        if ys.len() <= s {
            return Err(ArimaError::InsufficientObservations {
                needed: s + 1,
                got: ys.len(),
                nobs: y.len(),
                what: "seasonal differencing before the KPSS d-sequence (each seasonal \
                       difference drops a full period)",
            });
        }
        ys = ys.windows(s + 1).map(|w| w[s] - w[0]).collect();
    }
    let (d, d_evidence) = match opts.fixed_d {
        Some(fixed) => (fixed, None),
        None => {
            let r = ndiffs(&ys, NdiffsTest::Kpss, opts.alpha, opts.max_d).map_err(|e| {
                ArimaError::Selection {
                    what: format!(
                        "choosing the differencing order d via successive KPSS tests \
                         failed: {e}"
                    ),
                }
            })?;
            let dd = r.d;
            (dd, Some(r))
        }
    };

    // --- Stage 3: order search at fixed (d, D). ---
    let allow_constant = d + seasonal_d <= 1;
    let mut search = Search {
        y,
        d,
        seasonal_d,
        s,
        ic: opts.ic,
        max_models: opts.max_models,
        visited: Vec::new(),
        trace: Vec::new(),
        best: None,
        budget_exhausted: false,
    };

    let cap1 = |cap: usize| -> usize { 1.min(cap) };
    let seasonal = s >= 2;
    if opts.stepwise {
        // Hyndman-Khandakar starting models, capped by the max orders.
        let mut starts = vec![
            Key {
                p: 2.min(opts.max_p),
                q: 2.min(opts.max_q),
                sp: if seasonal { cap1(opts.max_seasonal_p) } else { 0 },
                sq: if seasonal { cap1(opts.max_seasonal_q) } else { 0 },
                constant: allow_constant,
            },
            Key {
                p: 0,
                q: 0,
                sp: 0,
                sq: 0,
                constant: allow_constant,
            },
            Key {
                p: cap1(opts.max_p),
                q: 0,
                sp: if seasonal { cap1(opts.max_seasonal_p) } else { 0 },
                sq: 0,
                constant: allow_constant,
            },
            Key {
                p: 0,
                q: cap1(opts.max_q),
                sp: 0,
                sq: if seasonal { cap1(opts.max_seasonal_q) } else { 0 },
                constant: allow_constant,
            },
        ];
        if allow_constant {
            // The no-constant null, so "no dynamics, no constant" is
            // always on the table (forecast does the same).
            starts.push(Key {
                p: 0,
                q: 0,
                sp: 0,
                sq: 0,
                constant: false,
            });
        }
        for key in starts {
            search.consider(key);
        }

        // Move to the first improving neighbor; restart the scan from
        // the new best; stop when no neighbor improves.
        'outer: while let Some((current, _, _)) = search.best.as_ref() {
            let current = *current;
            if search.budget_exhausted {
                break;
            }
            for key in neighborhood(current, opts, allow_constant) {
                if search.consider(key) {
                    continue 'outer;
                }
            }
            break;
        }
    } else {
        // Exhaustive grid, subject to max_order (mirroring R, which
        // applies max.order only to the non-stepwise search).
        let constants: &[bool] = if allow_constant {
            &[false, true]
        } else {
            &[false]
        };
        let max_sp = if seasonal { opts.max_seasonal_p } else { 0 };
        let max_sq = if seasonal { opts.max_seasonal_q } else { 0 };
        let mut grid = Vec::new();
        for p in 0..=opts.max_p {
            for q in 0..=opts.max_q {
                for sp in 0..=max_sp {
                    for sq in 0..=max_sq {
                        if p + q + sp + sq > opts.max_order {
                            continue;
                        }
                        for &constant in constants {
                            grid.push(Key {
                                p,
                                q,
                                sp,
                                sq,
                                constant,
                            });
                        }
                    }
                }
            }
        }
        if grid.len() > MAX_GRID_MODELS {
            return Err(ArimaError::InvalidArgument {
                what: "stepwise = false would fit more than 512 candidate models with \
                       these caps; use the stepwise search (the default), or lower \
                       max_p/max_q/max_seasonal_p/max_seasonal_q/max_order until the \
                       grid is small",
            });
        }
        search.max_models = grid.len().max(opts.max_models);
        for key in grid {
            search.consider(key);
        }
    }

    let Some((_, best, best_ic)) = search.best.take() else {
        let attempted = search.trace.len();
        let last_error = search
            .trace
            .iter()
            .rev()
            .find_map(|c| c.error.clone())
            .unwrap_or_else(|| "every candidate sat on the unit circle".to_owned());
        return Err(ArimaError::Selection {
            what: format!(
                "no admissible model among the {attempted} candidates tried (last \
                 failure: {last_error}). The series is probably too short for the \
                 differencing orders chosen — check d/D against the sample size, or \
                 fix d explicitly"
            ),
        });
    };

    let spec = best.spec;
    let n_ok = search
        .trace
        .iter()
        .filter(|c| c.status == CandidateStatus::Ok)
        .count();
    let seasonal_txt = if s >= 2 {
        format!(
            "({},{},{})[{}]",
            spec.seasonal_p(),
            spec.seasonal_d(),
            spec.seasonal_q(),
            s
        )
    } else {
        String::new()
    };
    let interpretation = format!(
        "Selected ARIMA({},{},{}){}{} by minimizing {} over {} candidates ({} admissible): \
         D {} d {}, then a {} search over (p, q{}, constant) at those fixed differencing \
         orders — information criteria are only comparable at equal differencing. The \
         trace lists every candidate tried; refitting any entry's orders reproduces its \
         criterion. Selection uncertainty is real: candidates within ~2 of the best {} \
         are near-ties, and the reported standard errors do not know a search happened.",
        spec.p(),
        spec.d(),
        spec.q(),
        seasonal_txt,
        if spec.include_constant() {
            " with constant"
        } else {
            " without constant"
        },
        opts.ic.code(),
        search.trace.len(),
        n_ok,
        if s >= 2 {
            if opts.fixed_seasonal_d.is_some() {
                format!("= {seasonal_d} was fixed by the caller,")
            } else {
                format!("= {seasonal_d} came from the seasonal-strength rule,")
            }
        } else {
            "= 0 (non-seasonal),".to_owned()
        },
        if opts.fixed_d.is_some() {
            format!("= {d} was fixed by the caller")
        } else {
            format!("= {d} from successive KPSS tests")
        },
        if opts.stepwise {
            "stepwise Hyndman-Khandakar"
        } else {
            "full-grid"
        },
        if s >= 2 { ", P, Q" } else { "" },
        opts.ic.code(),
    );

    Ok(AutoArimaResult {
        best,
        best_ic,
        ic: opts.ic,
        stepwise: opts.stepwise,
        d,
        seasonal_d,
        seasonal_period: s,
        d_evidence,
        seasonal_d_evidence,
        n_models: search.trace.len(),
        budget_exhausted: search.budget_exhausted,
        trace: search.trace,
        interpretation,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::min_root_modulus;

    #[test]
    fn root_modulus_degree_one() {
        // AR(1) phi = 0.5: P(z) = 1 - 0.5 z, root 2.
        let m = min_root_modulus(&[-0.5]).unwrap();
        assert!((m - 2.0).abs() < 1e-12, "{m}");
    }

    #[test]
    fn root_modulus_degree_two_real() {
        // (1 - 0.5 z)(1 - 0.25 z) = 1 - 0.75 z + 0.125 z^2: min root 2.
        let m = min_root_modulus(&[-0.75, 0.125]).unwrap();
        assert!((m - 2.0).abs() < 1e-8, "{m}");
    }

    #[test]
    fn root_modulus_degree_two_complex() {
        // z^2 - z + 0.5 scaled: AR(2) phi = (1.0, -0.5):
        // P(z) = 1 - z + 0.5 z^2, roots 1 +- i, modulus sqrt(2).
        let m = min_root_modulus(&[-1.0, 0.5]).unwrap();
        assert!((m - std::f64::consts::SQRT_2).abs() < 1e-8, "{m}");
    }

    #[test]
    fn root_modulus_near_unit() {
        // MA(1) theta = -0.999: P(z) = 1 - 0.999 z, root 1.001001...
        let m = min_root_modulus(&[-0.999]).unwrap();
        assert!((m - 1.0 / 0.999).abs() < 1e-12, "{m}");
    }

    #[test]
    fn root_modulus_degree_five() {
        // (1 - 0.9 z)(1 + 0.5 z)(1 - 0.4 z)(1 + 0.3 z)(1 - 0.2 z):
        // min-modulus root is 1/0.9.
        let factors: [f64; 5] = [0.9, -0.5, 0.4, -0.3, 0.2];
        let mut poly = vec![1.0];
        for r in factors {
            let mut next = vec![0.0; poly.len() + 1];
            for (i, &c) in poly.iter().enumerate() {
                next[i] += c;
                next[i + 1] -= c * r;
            }
            poly = next;
        }
        let m = min_root_modulus(&poly[1..]).unwrap();
        assert!((m - 1.0 / 0.9).abs() < 1e-7, "{m}");
    }

    #[test]
    fn root_modulus_zero_degree() {
        assert!(min_root_modulus(&[]).is_none());
        assert!(min_root_modulus(&[0.0, 0.0]).is_none());
    }
}
