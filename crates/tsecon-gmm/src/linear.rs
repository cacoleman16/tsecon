//! Linear instrumental-variables GMM: one-step, two-step efficient, and
//! iterated estimators with heteroskedasticity-robust and HAC weighting, the
//! robust (sandwich) parameter covariance, and the Hansen (1982) J-test of
//! over-identifying restrictions.
//!
//! # Model
//!
//! We estimate `k` parameters `beta` in the linear moment condition
//! `E[z_t (y_t - x_t' beta)] = 0`, where `x_t` are the `k` regressors
//! (some possibly endogenous), `z_t` are the `L >= k` instruments (the
//! included exogenous regressors instrument themselves, statsmodels/
//! linearmodels-style), and `y_t` is the response. Stacking observations,
//! `X` is `n x k`, `Z` is `n x L`, `y` is `n x 1`.
//!
//! The GMM estimator minimizes the criterion `gbar(beta)' W gbar(beta)`
//! where `gbar(beta) = Z'(y - X beta) / n` is the sample moment vector and
//! `W` is an `L x L` positive-definite weighting matrix. For linear moments
//! the minimizer is closed-form:
//!
//! ```text
//! beta_hat(W) = (X'Z W Z'X)^{-1} X'Z W Z'y.
//! ```
//!
//! # Weighting and the efficient two-step estimator
//!
//! The efficient GMM weight is the inverse of the moment-score covariance
//! `S = Avar(sqrt(n) gbar)`. It is unknown, so the two-step estimator
//! (Hansen 1982) plugs in a first-step estimate:
//!
//! * **Step 1** — `W1 = (Z'Z / n)^{-1}` (this makes step 1 numerically the
//!   two-stage least-squares estimator). Estimate `beta1`; residuals
//!   `u1 = y - X beta1`.
//! * **Step 2** — estimate the moment covariance `S(u1)` from the step-1
//!   residuals (see [`GmmWeight`]), set `W2 = S(u1)^{-1}`, and re-estimate
//!   `beta2`.
//!
//! # Covariance and the Hansen J-test — the exact linearmodels convention
//!
//! This crate reproduces `linearmodels` 7.0 `IVGMM(...).fit()` with the
//! default `weight_type="robust"` and `cov_type="robust"` to machine
//! precision. Two conventions matter and were pinned empirically against the
//! golden fixture (`fixtures/gmm.json`):
//!
//! * **Covariance** uses the *general* GMM sandwich, not the efficient
//!   simplification. With `G = Z'X / n` (the `L x k` moment Jacobian), the
//!   estimation weight `W` used in the final step, and the robust moment
//!   covariance `S` **recomputed at the final residuals**,
//!   ```text
//!   Cov(beta) = (1/n) (G' W G)^{-1} (G' W S W G) (G' W G)^{-1}.
//!   ```
//!   Because linearmodels keeps `W = S(u1)^{-1}` (the step-1 weight) while
//!   recomputing `S = S(u2)` at the step-2 residuals, `W != S^{-1}` exactly,
//!   so the sandwich does *not* collapse to `(G' W G)^{-1}/n`; using the
//!   collapsed form reproduces the golden `bse` only to ~5e-5, whereas the
//!   full sandwich matches to ~1e-17.
//! * **First-stage F** ([`FirstStageF`]) follows the *Stata* convention
//!   instead: an HC1 Wald divided by the number of excluded instruments and
//!   referred to `F(q, n - L)`, matching the sibling first-stage diagnostics
//!   elsewhere in this library. `linearmodels` reports the same test as an
//!   undivided HC0 Wald against `chi2(q)`; the two differ by the exact
//!   factor `q * n / (n - L)`, and the golden test pins that identity.
//! * **Hansen J** uses the *step-2 estimation weight* `W = S(u1)^{-1}`
//!   (the weight actually used to compute `beta2`), evaluated at the step-2
//!   residuals: `J = n * gbar(u2)' W gbar(u2)`. Recomputing `S` at `u2` for
//!   the J-statistic (as one might expect) reproduces the golden only to
//!   ~6e-4; the step-2 weight matches to ~3e-16. Under the null of correct
//!   over-identifying restrictions, `J ~ chi^2(L - k)`.
//!
//! See `tests/golden.rs` for the fixture check documenting these tolerances.

use tsecon_hac::{newey_west_maxlags, ols, Kernel, SeType};
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_stats::{chi2_sf, special::beta_inc};

use crate::error::GmmError;
use crate::matrix::{
    col_vec, inv_spd, mat_from_cols, mat_from_rowmajor, mat_to_rowmajor, solve_spd,
};

/// How the moment-score covariance `S = Avar(sqrt(n) gbar)` is estimated,
/// both for the efficient two-step weight `W = S^{-1}` and for the robust
/// sandwich covariance meat.
///
/// The moment scores are `g_t = z_t u_t` with `u_t` the estimation
/// residuals; `S` is the (long-run) covariance of `gbar = (1/n) sum_t g_t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GmmWeight {
    /// Heteroskedasticity-robust White (1980) covariance,
    /// `S = (1/n) sum_t g_t g_t' = (1/n) sum_t z_t z_t' u_t^2`. This is
    /// `linearmodels` `weight_type="robust"` (no small-sample degrees-of-
    /// freedom correction), the convention the golden fixture uses.
    Robust,
    /// Heteroskedasticity-and-autocorrelation-consistent (Newey-West 1987)
    /// covariance, `S = (1/n) [Gamma_0 + sum_{j>=1} w_j (Gamma_j + Gamma_j')]`
    /// with `Gamma_j = sum_{t>j} g_t g_{t-j}'` and kernel weights `w_j` from
    /// [`tsecon_hac::Kernel`] (the library's single kernel owner). Use for
    /// serially correlated moments (e.g. overlapping observations, forecast
    /// errors).
    ///
    /// # A zero bandwidth is rejected, not honored
    ///
    /// `bandwidth = 0` zeroes every lag `j >= 1` for every kernel, so `S`
    /// collapses to its lag-0 term — the White estimator, bit for bit. A
    /// caller who asked for HAC and silently received White has no
    /// serial-correlation robustness and no way to notice, so the estimators
    /// reject it with [`GmmError::HacBandwidthNoOp`] rather than compute it.
    /// Use [`GmmWeight::HacAuto`] to have the bandwidth chosen for you.
    ///
    /// # A positive bandwidth is not a fix for the coverage gap
    ///
    /// Rejecting the no-op restores a genuine HAC computation. It does not
    /// make the intervals cover. This library's own interval-coverage audit
    /// measured `iv_gmm(weight="hac")` under an AR(1) error with `phi = 0.8`
    /// at an explicit `bandwidth = 10` and found **0.868 ± 0.006 coverage
    /// against a nominal 0.95** — an 8.2-point shortfall
    /// (`docs/examples/interval-coverage.md`, Table 1). Serially correlated
    /// moments with a persistent error remain under-covered here at any
    /// bandwidth this crate offers; see [`GmmWeight::HacAuto`].
    Hac {
        /// The lag-weighting kernel.
        kernel: Kernel,
        /// Bandwidth in the [`tsecon_hac::Kernel::weight`] convention (for
        /// Bartlett/Parzen/truncated this is the lag-truncation `maxlags`).
        /// Must be strictly positive; see the type-level note above.
        bandwidth: f64,
    },
    /// HAC weighting with the bandwidth chosen by the documented automatic
    /// rule [`GmmWeight::auto_bandwidth`] (Newey & West 1994,
    /// `floor(4 * (n/100)^(2/9))`) instead of by the caller.
    ///
    /// The realized bandwidth is reported back in [`GmmFit::hac_bandwidth`],
    /// so an automatic choice is never invisible.
    ///
    /// # This does not close the measured coverage gap
    ///
    /// `HacAuto` is offered as a *default that is not zero*, not as a remedy.
    /// The interval-coverage audit measured `iv_gmm(weight="hac")` at
    /// **0.868 ± 0.006 against a nominal 0.95** under an AR(1) error with
    /// `phi = 0.8` at `T = 250` and an explicit `bandwidth = 10`
    /// (`docs/examples/interval-coverage.md`, Table 1). At that same
    /// `T = 250` this rule returns `floor(4 * 2.5^(2/9)) = 4` lags — *fewer*
    /// than the setting that produced 0.868, so it truncates the
    /// autocovariance sum sooner and there is no reason to expect it to do
    /// better. Switching from an explicit bandwidth to `HacAuto` should not
    /// be read as fixing the under-coverage; the gap is unresolved. Under
    /// persistent moments, treat a nominal-95% GMM interval as narrower than
    /// its label.
    HacAuto {
        /// The lag-weighting kernel.
        kernel: Kernel,
    },
}

impl GmmWeight {
    /// The automatic HAC lag truncation `floor(4 * (n/100)^(2/9))` used by
    /// [`GmmWeight::HacAuto`].
    ///
    /// This is the Newey & West (1994, *Review of Economic Studies* 61,
    /// §"a simple rule of thumb") Bartlett-kernel pilot rule — the same
    /// number statsmodels uses as the default `maxlags` for
    /// `cov_type="HAC"` and the one applied users can reproduce by hand.
    /// It delegates to [`tsecon_hac::newey_west_maxlags`], the library's
    /// single owner of the rule; GMM does not re-derive bandwidth formulas.
    ///
    /// **Why the deterministic rule and not the data-dependent plug-in.**
    /// [`tsecon_hac::newey_west_bandwidth`] implements the full Newey-West
    /// (1994) nonparametric plug-in, but it is defined for a *univariate*
    /// series, and the GMM moment vector is `L`-dimensional. Applying it
    /// here would force an undocumented extra convention (which scalarization
    /// of `g_t` the pilot is run on), and different packages choose
    /// differently. The rule of thumb has no such free choice, so an
    /// automatic bandwidth stays reproducible across implementations.
    ///
    /// The rule returns at least 1 for every `nobs >= 1`, so
    /// [`GmmWeight::HacAuto`] can never degenerate into the White estimator
    /// (`nobs = 0` is rejected earlier as an empty input). That is the whole
    /// of its guarantee: it is deterministic, reproducible, and nonzero. It
    /// is *not* tuned for coverage — see [`GmmWeight::HacAuto`] for the
    /// measured 0.868-against-0.95 shortfall it does not repair.
    #[must_use]
    pub fn auto_bandwidth(nobs: usize) -> f64 {
        newey_west_maxlags(nobs) as f64
    }

    /// Resolve to the concrete `(kernel, bandwidth)` actually used at sample
    /// size `nobs`, rejecting the silent HAC no-op. Called once per fit,
    /// before any estimation work, so a bad weighting request fails fast.
    fn resolve(self, nobs: usize) -> Result<ResolvedWeight, GmmError> {
        match self {
            Self::Robust => Ok(ResolvedWeight::Robust),
            Self::Hac { kernel, bandwidth } => {
                if !bandwidth.is_finite() || bandwidth < 0.0 {
                    return Err(GmmError::InvalidBandwidth { value: bandwidth });
                }
                if bandwidth == 0.0 {
                    return Err(GmmError::HacBandwidthNoOp {
                        kernel: kernel.name(),
                        suggested: newey_west_maxlags(nobs),
                    });
                }
                Ok(ResolvedWeight::Hac { kernel, bandwidth })
            }
            Self::HacAuto { kernel } => Ok(ResolvedWeight::Hac {
                kernel,
                bandwidth: Self::auto_bandwidth(nobs),
            }),
        }
    }
}

/// A [`GmmWeight`] with the bandwidth pinned down: what the moment-covariance
/// code actually consumes, after validation and automatic selection.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ResolvedWeight {
    Robust,
    Hac { kernel: Kernel, bandwidth: f64 },
}

impl ResolvedWeight {
    /// The realized HAC bandwidth, for reporting back to the caller.
    fn bandwidth(self) -> Option<f64> {
        match self {
            Self::Robust => None,
            Self::Hac { bandwidth, .. } => Some(bandwidth),
        }
    }
}

/// The Hansen (1982) J-test of over-identifying restrictions.
///
/// Present only when the model is over-identified (`L > k`); an exactly
/// identified model fits the moments exactly (`gbar = 0`), leaving no
/// restrictions to test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HansenJ {
    /// The J-statistic `n * gbar' W gbar` at the final estimate, with `W`
    /// the final-step estimation weight.
    pub stat: f64,
    /// Degrees of freedom `L - k` (number of over-identifying restrictions).
    pub dof: usize,
    /// The p-value `P(chi^2(dof) > stat)` (chi-squared survival function).
    pub pval: f64,
}

/// The first-stage strength diagnostic for one instrumented (endogenous)
/// regressor: the heteroskedasticity-robust F on the *excluded* instruments
/// in the first-stage regression of that regressor on the full instrument
/// set.
///
/// # Why this is reported at all
///
/// The IV-GMM sandwich standard errors are a first-order asymptotic
/// approximation that degrades as the instruments weaken, and the failure is
/// invisible in the output: an interval-coverage audit of this library found
/// nominal-95% GMM intervals covering 0.915 at a median first-stage F of
/// 10.5, and 0.839 with genuinely weak instruments (median F = 1.2), where
/// the median reported standard error was only 0.456 of the true sampling
/// standard deviation. The sibling IV entry points (`lp_iv`,
/// `lp_multiplier`, `proxy_svar`) all report a first-stage F; this one used
/// to report nothing, so the caller had no way to judge any of that.
///
/// **`F > 10` is a rule of thumb, not a safety threshold.** The audit
/// numbers above are exactly the regime where the rule is usually declared
/// satisfied. Treat a large F as necessary, not sufficient.
///
/// # With two or more endogenous regressors this is not a weak-identification test
///
/// Read literally, this statistic answers one question: *do the excluded
/// instruments jointly predict this one regressor?* With a single endogenous
/// regressor that is also the identification question, and the rule of thumb
/// applies. With **two or more**, it is not.
///
/// All of the per-regressor F's can sit comfortably above 10 while the system
/// is under-identified, because the instruments may predict only a single
/// common linear combination of the endogenous regressors: each regressor is
/// well explained, and yet the coefficients cannot be separated. Nothing in
/// this number will say so. The naive per-regressor F is silent about the one
/// failure mode that is specific to multiple endogenous regressors.
///
/// The statistics that do answer the question are the **Angrist-Pischke**
/// first-stage F (the per-regressor object: an F on the instruments after
/// partialling the *other* endogenous regressors out of both sides) and, for
/// the system, the **Cragg-Donald** minimum eigenvalue statistic — or
/// **Kleibergen-Paap** under heteroskedasticity or serial correlation —
/// referred to the Stock-Yogo critical values, which depend on the number of
/// endogenous regressors and instruments rather than on a fixed 10. This is
/// why `linearmodels` prints Shea's partial `R^2` next to its `f.stat` column
/// instead of leaving `f.stat` to be read alone.
///
/// None of those are implemented here yet. Until they are, with more than one
/// endogenous regressor treat this entry as a per-regressor *fit* summary and
/// not as evidence that the system is identified.
///
/// # Convention
///
/// The statistic is the Wald statistic on the `q` excluded-instrument
/// coefficients divided by `q`, using the HC1 (MacKinnon & White 1985)
/// robust covariance, referred to `F(q, n - L)`. It is pinned by a golden
/// fixture against `linearmodels` 7.0 (see below); it is *believed* to match
/// the robust first-stage F that Stata's `estat firststage` reports, since
/// that is the convention Stata uses for a Wald test after `regress, robust`,
/// but no fixture pins Stata and that claim is unverified here. The form is
/// self-consistent with the rest of this library: `tsecon-ident`'s
/// `proxy_svar` first-stage F is an HC1 squared robust t, and `tsecon-lp`'s is
/// a HAC squared robust t with `use_correction = true`.
///
/// `linearmodels` reports the same test in a different dress: its
/// `IV2SLS(...).fit(cov_type="robust").first_stage.diagnostics` `f.stat`
/// column is the **undivided** Wald statistic on the HC0 covariance,
/// referred to `chi2(q)`. The two are related exactly by
/// `f.stat = fstat * q * n / (n - L)`; `tests/golden.rs` pins this crate
/// against `linearmodels` through that identity. The HC1 form is the
/// smaller (more conservative) of the two, which is the right direction
/// given the audit's finding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FirstStageF {
    /// Index of the instrumented regressor within the `x_cols` the caller
    /// passed.
    pub regressor: usize,
    /// The robust first-stage F statistic (Wald on the excluded instruments,
    /// HC1, divided by the number of excluded instruments).
    pub fstat: f64,
    /// Numerator degrees of freedom `q`: the number of excluded instruments
    /// (instrument columns that are not also regressor columns).
    pub dof_num: usize,
    /// Denominator degrees of freedom `n - L` of the reference `F(q, n - L)`.
    pub dof_den: usize,
    /// The p-value `P(F(dof_num, dof_den) > fstat)`.
    pub pval: f64,
}

/// A fitted linear IV-GMM regression.
#[derive(Debug, Clone, PartialEq)]
pub struct GmmFit {
    /// Coefficient estimates `beta`, in the order the regressor columns were
    /// passed.
    pub params: Vec<f64>,
    /// Standard errors `sqrt(diag(Cov(beta)))`, one per parameter.
    pub bse: Vec<f64>,
    /// Parameter covariance `Cov(beta)`, `k x k` row-major (the robust GMM
    /// sandwich; see the module docs).
    pub cov: Vec<f64>,
    /// Residuals `u_t = y_t - x_t' beta`, length `n`.
    pub residuals: Vec<f64>,
    /// Number of observations `n`.
    pub nobs: usize,
    /// Number of moment conditions / instruments `L`.
    pub nmoments: usize,
    /// Number of parameters `k`.
    pub nparams: usize,
    /// Number of GMM estimation steps performed (weight updates + 1): 1 for
    /// one-step / 2SLS, 2 for the two-step estimator, and the iteration count
    /// for the iterated estimator.
    pub steps: usize,
    /// The Hansen J-test, present iff the model is over-identified (`L > k`).
    pub jtest: Option<HansenJ>,
    /// Robust first-stage F for each instrumented regressor, in `x_cols`
    /// order — the weak-instrument diagnostic; see [`FirstStageF`].
    ///
    /// A regressor column that also appears among the instruments is
    /// exogenous and instruments itself, so it gets no entry. The vector is
    /// therefore empty for a model with no endogenous regressors, and
    /// entries are also omitted wherever the diagnostic is undefined or not
    /// computable: no excluded instruments (`q = 0`), no residual degrees of
    /// freedom in the first stage (`n <= L`), a regressor the instruments
    /// reproduce numerically exactly, a rank-deficient instrument matrix, or
    /// a non-finite statistic. **A missing entry is not a failed fit** — the
    /// GMM estimate itself does not depend on this diagnostic, and a
    /// diagnostic is never allowed to fail an otherwise-valid estimation.
    /// Index by [`FirstStageF::regressor`], not by position.
    pub first_stage: Vec<FirstStageF>,
    /// The HAC lag truncation actually used for the moment covariance, or
    /// `None` under [`GmmWeight::Robust`]. Populated for
    /// [`GmmWeight::HacAuto`] too, so an automatically chosen bandwidth is
    /// always visible in the output.
    pub hac_bandwidth: Option<f64>,
}

/// The robust sandwich covariance (row-major `k x k`), the standard errors,
/// and the (optional, over-identified-only) Hansen J-test — the trio returned
/// by [`Design::cov_and_j`].
type CovAndJ = (Vec<f64>, Vec<f64>, Option<HansenJ>);

/// The assembled design in the several cross-product forms the estimators
/// reuse. Built once, then shared across steps.
struct Design {
    xmat: Mat<f64>, // n x k
    zmat: Mat<f64>, // n x L
    xz: Mat<f64>,   // k x L = X'Z
    zx: Mat<f64>,   // L x k = Z'X
    zy: Mat<f64>,   // L x 1 = Z'y
    zz: Mat<f64>,   // L x L = Z'Z
    y: Vec<f64>,
    n: usize,
    k: usize,
    l: usize,
    /// First-stage F per instrumented regressor. A function of (X, Z) only,
    /// so it is computed once here and shared by every estimator/weight.
    first_stage: Vec<FirstStageF>,
}

fn check_finite(xs: &[f64], what: &'static str) -> Result<(), GmmError> {
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(GmmError::NonFinite { what });
    }
    Ok(())
}

impl Design {
    /// Validate the inputs and assemble the cross-product matrices.
    fn build(x_cols: &[Vec<f64>], z_cols: &[Vec<f64>], y: &[f64]) -> Result<Self, GmmError> {
        let n = y.len();
        let k = x_cols.len();
        let l = z_cols.len();
        if n == 0 {
            return Err(GmmError::EmptyInput { what: "response y" });
        }
        if k == 0 {
            return Err(GmmError::EmptyInput {
                what: "regressor columns X",
            });
        }
        if l == 0 {
            return Err(GmmError::EmptyInput {
                what: "instrument columns Z",
            });
        }
        for col in x_cols.iter() {
            if col.len() != n {
                return Err(GmmError::DimensionMismatch {
                    what: "regressor column length vs y",
                    expected: n,
                    got: col.len(),
                });
            }
            check_finite(col, "regressor columns X")?;
        }
        for col in z_cols.iter() {
            if col.len() != n {
                return Err(GmmError::DimensionMismatch {
                    what: "instrument column length vs y",
                    expected: n,
                    got: col.len(),
                });
            }
            check_finite(col, "instrument columns Z")?;
        }
        check_finite(y, "response y")?;
        if l < k {
            return Err(GmmError::UnderIdentified {
                moments: l,
                params: k,
            });
        }
        if n <= k {
            return Err(GmmError::DegreesOfFreedom { n, k });
        }

        let xmat = mat_from_cols(x_cols, n);
        let zmat = mat_from_cols(z_cols, n);
        let ymat = col_vec(y);
        let xz = xmat.transpose() * &zmat; // k x L
        let zx = zmat.transpose() * &xmat; // L x k
        let zy = zmat.transpose() * &ymat; // L x 1
        let zz = zmat.transpose() * &zmat; // L x L
                                           // Infallible by construction: a broken diagnostic omits its entry, it
                                           // never fails the estimation. See `first_stage_f`.
        let first_stage = first_stage_f(x_cols, z_cols, n);
        Ok(Self {
            xmat,
            zmat,
            xz,
            zx,
            zy,
            zz,
            y: y.to_vec(),
            n,
            k,
            l,
            first_stage,
        })
    }

    /// The step-1 weight `W1 = (Z'Z / n)^{-1}` (makes step 1 the 2SLS
    /// estimator).
    fn initial_weight(&self) -> Result<Mat<f64>, GmmError> {
        let nf = self.n as f64;
        let zz_over_n = Mat::from_fn(self.l, self.l, |i, j| self.zz[(i, j)] / nf);
        inv_spd(zz_over_n.as_ref(), "step-1 weight (Z'Z/n)")
    }

    /// Closed-form point estimate `beta(W) = (X'Z W Z'X)^{-1} X'Z W Z'y`.
    fn point_estimate(&self, w: MatRef<'_, f64>) -> Result<Vec<f64>, GmmError> {
        let xzw = &self.xz * w; // k x L
        let a = &xzw * &self.zx; // k x k (symmetric PD)
        let b = &xzw * &self.zy; // k x 1
        let beta = solve_spd(
            a.as_ref(),
            b.as_ref(),
            "GMM normal equations X'Z W Z'X (weak/collinear instruments?)",
        )?;
        Ok((0..self.k).map(|i| beta[(i, 0)]).collect())
    }

    /// Residuals `u = y - X beta`.
    fn residuals(&self, beta: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|t| {
                let mut fit = 0.0;
                for (j, &b) in beta.iter().enumerate() {
                    fit += self.xmat[(t, j)] * b;
                }
                self.y[t] - fit
            })
            .collect()
    }

    /// Robust or HAC moment-score covariance `S` (`L x L`) at residuals `u`.
    fn moment_cov(&self, u: &[f64], weight: ResolvedWeight) -> Result<Mat<f64>, GmmError> {
        moment_cov(self.zmat.as_ref(), u, self.n, self.l, weight)
    }

    /// The robust GMM sandwich covariance and the Hansen J-test, given the
    /// final-step estimation weight `w`, the moment covariance `s` at the
    /// final residuals, and the final residuals `u` (for `gbar`).
    fn cov_and_j(
        &self,
        w: MatRef<'_, f64>,
        s: MatRef<'_, f64>,
        u: &[f64],
    ) -> Result<CovAndJ, GmmError> {
        let nf = self.n as f64;
        // G = Z'X / n  (L x k moment Jacobian).
        let g = Mat::from_fn(self.l, self.k, |i, j| self.zx[(i, j)] / nf);
        let gt = g.transpose();
        // bread = (G' W G)^{-1}.
        let gtw = gt * w; // k x L
        let bread_arg = &gtw * &g; // k x k
        let bread = inv_spd(
            bread_arg.as_ref(),
            "GMM sandwich bread G'WG (weak/collinear instruments?)",
        )?;
        // meat = G' W S W G.
        let gtws = &gtw * s; // k x L
        let gtwsw = &gtws * w; // k x L
        let meat = &gtwsw * &g; // k x k
                                // Cov = (1/n) bread meat bread.
        let bm = &bread * &meat;
        let cov_mat = Mat::from_fn(self.k, self.k, |i, j| {
            let mut acc = 0.0;
            for m in 0..self.k {
                acc += bm[(i, m)] * bread[(m, j)];
            }
            acc / nf
        });
        let cov = mat_to_rowmajor(cov_mat.as_ref());

        let mut bse = Vec::with_capacity(self.k);
        for i in 0..self.k {
            let v = cov[i * self.k + i];
            if v < 0.0 || !v.is_finite() {
                return Err(GmmError::SingularMatrix {
                    what: "GMM sandwich covariance diagonal (non-PSD moment covariance?)",
                });
            }
            bse.push(v.sqrt());
        }

        // Hansen J = n * gbar' W gbar, gbar = Z'u / n  =>  J = (Z'u)' W (Z'u) / n.
        let jtest = if self.l > self.k {
            let zu = self.zmat.transpose() * &col_vec(u); // L x 1
            let wzu = w * &zu; // L x 1
            let mut quad = 0.0;
            for i in 0..self.l {
                quad += zu[(i, 0)] * wzu[(i, 0)];
            }
            let stat = quad / nf;
            let dof = self.l - self.k;
            let pval = chi2_sf(stat, dof as f64)?;
            Some(HansenJ { stat, dof, pval })
        } else {
            None
        };
        Ok((cov, bse, jtest))
    }
}

/// Robust/HAC moment-score covariance `S` (`L x L`) of `gbar`, from scores
/// `g_t = z_t u_t`. Free function so the nonlinear driver could reuse it.
fn moment_cov(
    zmat: MatRef<'_, f64>,
    u: &[f64],
    n: usize,
    l: usize,
    weight: ResolvedWeight,
) -> Result<Mat<f64>, GmmError> {
    let nf = n as f64;
    // Scores g_t = z_t * u_t, stored n x L.
    let scores = Mat::from_fn(n, l, |t, j| zmat[(t, j)] * u[t]);

    match weight {
        ResolvedWeight::Robust => {
            // S = (1/n) sum_t g_t g_t'.
            Ok(Mat::from_fn(l, l, |i, j| {
                let mut acc = 0.0;
                for t in 0..n {
                    acc += scores[(t, i)] * scores[(t, j)];
                }
                acc / nf
            }))
        }
        ResolvedWeight::Hac { kernel, bandwidth } => {
            // The bandwidth was validated (finite, strictly positive) by
            // `GmmWeight::resolve` before any estimation started.
            // S = (1/n)[Gamma_0 + sum_{j>=1} w_j (Gamma_j + Gamma_j')].
            let mut s = vec![0.0_f64; l * l];
            for lag in 0..n {
                let wj = kernel.weight(lag, bandwidth);
                if lag > 0 && wj == 0.0 && kernel.truncates() {
                    break;
                }
                for t in lag..n {
                    for i in 0..l {
                        let gti = scores[(t, i)];
                        for j in 0..l {
                            let g = gti * scores[(t - lag, j)];
                            if lag == 0 {
                                s[i * l + j] += g;
                            } else {
                                s[i * l + j] += wj * g;
                                s[j * l + i] += wj * g;
                            }
                        }
                    }
                }
            }
            Ok(Mat::from_fn(l, l, |i, j| s[i * l + j] / nf))
        }
    }
}

/// Relative tolerance of [`same_column`], applied **per element**.
const SAME_COLUMN_RTOL: f64 = 1e-12;

/// Whether two design columns are the same variable.
///
/// The library's statsmodels-style convention is that included exogenous
/// regressors appear in *both* `x_cols` and `z_cols` — the caller passes the
/// same numbers twice — so bit equality is the normal case, and the tolerance
/// only absorbs re-derivation noise: a rebuilt constant column, a different
/// summation order, a value that made one extra round trip through the same
/// `f64` arithmetic. Every element must agree to
/// [`SAME_COLUMN_RTOL`] **relative to its own magnitude**,
/// `|a_i - b_i| <= 1e-12 * max(|a_i|, |b_i|)`.
///
/// # Why per-element relative, and what the heuristic still cannot do
///
/// An earlier version scaled one absolute tolerance by the column maximum
/// (`1e-12 * max_i |a_i|`). That is wrong in both directions and both were
/// observed:
///
/// * On a column with mixed magnitudes the tolerance is set by the largest
///   element, so it can swamp the small ones — a column whose maximum is
///   `1e13` gets an effective tolerance of `10` at every observation, and two
///   plainly different columns are declared the same variable. The endogenous
///   regressor is then mistaken for an exogenous one and silently gets **no**
///   first-stage F at all.
/// * A shared exogenous column perturbed by a *relative* `1e-9` (well below
///   the old absolute tolerance for small entries, well above it for large
///   ones) is reclassified as an excluded instrument. That does not merely
///   add a spurious row: the excluded-instrument count `q` changes, so the
///   genuinely endogenous regressor's own [`FirstStageF::dof_num`],
///   [`FirstStageF::dof_den`] and Wald form change with it. The caller gets a
///   wrong number on the row they care about.
///
/// The per-element form fixes both, but this is still a **heuristic on
/// numbers**, not a declaration of the exogenous/endogenous split (which this
/// crate's `(X, Z)` interface does not take; `linearmodels` does). Its
/// remaining limits, stated plainly:
///
/// * **A recomputed column is not recognized.** A column that is the same
///   variable but arrived through a lossier path — an `f32` round trip
///   (relative error `~6e-8`), a different algebraic route that loses more
///   than twelve digits — fails the test and is treated as excluded,
///   changing `q` and every reported degrees-of-freedom. Pass the *identical*
///   values in `x_cols` and `z_cols`.
/// * **Two distinct variables that agree to twelve relative digits
///   everywhere are indistinguishable** and are treated as one. They would be
///   collinear and the estimator would reject the design anyway, so this is
///   the benign direction.
/// * **It matches columns, not spans.** An exogenous regressor that is a
///   linear *combination* of instrument columns rather than one of them is
///   classified as endogenous, and gets a first-stage F it should not have.
fn same_column(a: &[f64], b: &[f64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() <= SAME_COLUMN_RTOL * x.abs().max(y.abs()))
}

/// Upper-tail p-value `P(F(d1, d2) > x)` via the regularized incomplete beta
/// function, `I_{d2/(d2 + d1 x)}(d2/2, d1/2)`.
///
/// Evaluating the upper tail directly (rather than `1 - cdf`) keeps the
/// small p-values of a strong first stage accurate; this mirrors the helper
/// in `tsecon-var::causality` and `tsecon-spectest::common`, which use the
/// same expression against the same `tsecon-stats` special function.
///
/// **NaN is an error, not a p-value of 1.** An earlier version folded NaN in
/// with `x <= 0.0` and returned `Ok(1.0)` for both, so a broken statistic came
/// back dressed as a confident "no evidence of a strong first stage". The
/// three cases are now distinguished: `x <= 0.0` (including `-inf`) is a
/// genuine `sf = 1`, `x = +inf` is a genuine `sf = 0`, and NaN is rejected.
fn f_sf(x: f64, d1: f64, d2: f64) -> Result<f64, GmmError> {
    if x.is_nan() {
        return Err(GmmError::NonFinite {
            what: "first-stage F statistic (NaN has no p-value)",
        });
    }
    if x <= 0.0 {
        return Ok(1.0);
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    Ok(beta_inc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * x))?)
}

/// The uncentered first-stage `R^2` above which the instruments are treated as
/// reproducing the regressor exactly: `RSS <= EXACT_FIT_R2_TOL * TSS`.
///
/// Scale-relative on purpose. The bit test it replaced
/// (`residuals.iter().all(|e| *e == 0.0)`) was dead code: the computed OLS
/// residuals of an exactly collinear regression are rounding noise of order
/// `eps * |x|`, never bit-zero. On the design `x = 2 + 3 z1 - 1.5 z2`, an
/// exact linear combination of `Z`, the measured `RSS/TSS` is `4.9e-32` and
/// the guard never fired — the fit returned `fstat = 1.19e33` with `pval = 0`,
/// a fabricated claim of infinitely strong instruments from the one diagnostic
/// whose job is to warn about weak ones.
///
/// `1e-12` sits twenty orders of magnitude above that measured `4.9e-32` (with
/// room for an ill-conditioned `Z`, whose noise floor grows like
/// `cond(Z)^2 eps^2`) and comfortably below any genuine first stage: the same
/// design plus a `1e-4` idiosyncratic component — already an implausibly good
/// first stage, uncentered `R^2 = 1 - 5.4e-10` — measures `5.4e-10` and keeps
/// its (astronomical, but real) F.
const EXACT_FIT_R2_TOL: f64 = 1e-12;

/// Robust first-stage F for every instrumented regressor; see
/// [`FirstStageF`] for the convention and why it is reported.
///
/// A regressor is treated as **instrumented (endogenous)** when its column
/// does not also appear among the instruments, and an instrument is
/// **excluded** when its column does not also appear among the regressors —
/// the only way to recover the exog/endog split from this crate's
/// `(X, Z)` interface, which (unlike `linearmodels`) does not take them as
/// separate arguments. See [`same_column`] for what "the same column" means
/// numerically and where the heuristic stops.
///
/// # A diagnostic never takes down the estimate
///
/// This function is infallible by construction: every way the first stage can
/// fail *skips that regressor's entry* rather than propagating an error. A
/// regressor is skipped when the diagnostic is not defined or not computable —
/// no excluded instruments (`q = 0`), no residual degrees of freedom in the
/// first stage (`n <= L`), a regressor the instruments reproduce to within
/// [`EXACT_FIT_R2_TOL`], a rank-deficient instrument matrix (the first-stage
/// OLS or its excluded block will not factor), or a non-finite statistic — and
/// the whole vector comes back empty when no regressor survives. A missing
/// entry is visible to the caller; a fabricated number would not be.
///
/// This is not a stylistic preference. Propagating those failures with `?`
/// made a *duplicated instrument column* fail the whole estimation, including
/// [`one_step_gmm`] with a caller-supplied weight — which never inverts `Z'Z`
/// and legitimately supported that design — and it reported the failure as a
/// singular `X'X` in an internal first-stage OLS the caller never asked for.
/// Where a rank-deficient `Z` genuinely does break an estimator, that
/// estimator still raises its own accurate error (e.g. the step-1 weight
/// `(Z'Z/n)` in [`two_stage_least_squares`] / [`two_step_gmm`]).
fn first_stage_f(x_cols: &[Vec<f64>], z_cols: &[Vec<f64>], n: usize) -> Vec<FirstStageF> {
    let l = z_cols.len();
    let excluded: Vec<usize> = (0..l)
        .filter(|&i| !x_cols.iter().any(|xc| same_column(xc, &z_cols[i])))
        .collect();
    let q = excluded.len();
    if q == 0 || n <= l {
        return Vec::new();
    }
    let dof_den = n - l;

    let mut out = Vec::new();
    for (j, xc) in x_cols.iter().enumerate() {
        // Included exogenous regressors instrument themselves: no first stage.
        if z_cols.iter().any(|zc| same_column(xc, zc)) {
            continue;
        }
        // First stage: OLS of the endogenous regressor on ALL instruments,
        // with the HC1 robust covariance (tsecon-hac owns OLS + robust
        // inference; GMM does not re-roll either). A rank-deficient Z makes
        // this fail; skip the diagnostic, do not fail the fit.
        let Ok(fit) = ols(xc, z_cols) else { continue };
        // A regressor the instruments reproduce (numerically) exactly has a
        // degenerate robust covariance — every score is rounding noise — so
        // the Wald statistic it would produce is noise divided by noise. The
        // test is scale-relative because the residuals of an exactly collinear
        // regression are ~1e-16, never bit-zero; see EXACT_FIT_R2_TOL.
        let rss: f64 = fit.residuals.iter().map(|e| e * e).sum();
        let tss: f64 = xc.iter().map(|v| v * v).sum();
        if rss <= EXACT_FIT_R2_TOL * tss {
            continue;
        }
        let Ok(inf) = fit.inference(SeType::Hc1) else {
            continue;
        };
        // Wald on the excluded-instrument coefficient block:
        // W = b_excl' V[excl, excl]^{-1} b_excl, and F = W / q.
        let block = Mat::from_fn(q, q, |a, b| inf.cov[excluded[a] * l + excluded[b]]);
        let Ok(block_inv) = inv_spd(
            block.as_ref(),
            "first-stage robust covariance of the excluded instruments \
             (collinear or duplicated instruments?)",
        ) else {
            continue;
        };
        let mut wald = 0.0;
        for a in 0..q {
            for b in 0..q {
                wald += fit.params[excluded[a]] * block_inv[(a, b)] * fit.params[excluded[b]];
            }
        }
        let fstat = wald / q as f64;
        // A non-finite statistic is not a diagnostic. Omit it rather than
        // report NaN/inf with whatever p-value falls out.
        if !fstat.is_finite() {
            continue;
        }
        let Ok(pval) = f_sf(fstat, q as f64, dof_den as f64) else {
            continue;
        };
        out.push(FirstStageF {
            regressor: j,
            fstat,
            dof_num: q,
            dof_den,
            pval,
        });
    }
    out
}

/// One-step linear GMM with a caller-supplied weighting matrix `weight`.
///
/// Estimates `beta = (X'Z W Z'X)^{-1} X'Z W Z'y` for the given `L x L`
/// weight `W` (row-major), then reports the robust GMM sandwich covariance
/// using `cov_weight` to estimate the moment covariance at the one-step
/// residuals, and the Hansen J-test when over-identified.
///
/// `x_cols` are the `k` regressor columns (include the constant explicitly,
/// statsmodels-style), `z_cols` the `L >= k` instrument columns (the included
/// exogenous regressors appear in both). `weight` must be `L x L` row-major.
///
/// # Errors
///
/// [`GmmError::EmptyInput`] for empty inputs; [`GmmError::DimensionMismatch`]
/// for mismatched column lengths or a mis-sized weight;
/// [`GmmError::UnderIdentified`] if `L < k`; [`GmmError::DegreesOfFreedom`]
/// if `n <= k`; [`GmmError::NonFinite`] on NaN/inf; [`GmmError::SingularMatrix`]
/// if the projected design or moment covariance is singular;
/// [`GmmError::HacBandwidthNoOp`] if `cov_weight` is
/// [`GmmWeight::Hac`] with `bandwidth = 0` (the White estimator in disguise).
pub fn one_step_gmm(
    x_cols: &[Vec<f64>],
    z_cols: &[Vec<f64>],
    y: &[f64],
    weight: &[f64],
    cov_weight: GmmWeight,
) -> Result<GmmFit, GmmError> {
    let d = Design::build(x_cols, z_cols, y)?;
    let rw = cov_weight.resolve(d.n)?;
    let w = mat_from_rowmajor(weight, d.l, "one-step GMM weight matrix (must be L x L)")?;
    finish(&d, w, 1, rw)
}

/// Two-stage least squares as one-step GMM with `W = (Z'Z / n)^{-1}`.
///
/// This is the exactly/over-identified 2SLS point estimator; the reported
/// covariance is the heteroskedasticity-robust ([`GmmWeight::Robust`])
/// sandwich (linearmodels `IV2SLS(...).fit(cov_type="robust")`). When the
/// model is exactly identified (`L == k`) the estimate coincides with the
/// simple IV estimator `(Z'X)^{-1} Z'y` for *any* weight.
///
/// # Errors
///
/// As [`one_step_gmm`].
pub fn two_stage_least_squares(
    x_cols: &[Vec<f64>],
    z_cols: &[Vec<f64>],
    y: &[f64],
) -> Result<GmmFit, GmmError> {
    let d = Design::build(x_cols, z_cols, y)?;
    let w1 = d.initial_weight()?;
    finish(&d, w1, 1, ResolvedWeight::Robust)
}

/// Two-step efficient linear IV-GMM (Hansen 1982).
///
/// Step 1 uses `W1 = (Z'Z / n)^{-1}` (2SLS); step 2 uses
/// `W2 = S(u1)^{-1}` with the moment covariance estimated from the step-1
/// residuals per `cov_weight`. The covariance and Hansen J follow the exact
/// linearmodels convention documented at the module level (the J-test uses
/// the step-2 weight `W2`, the covariance recomputes `S` at the step-2
/// residuals). With `cov_weight = GmmWeight::Robust` this reproduces
/// `linearmodels` `IVGMM(...).fit()` to machine precision.
///
/// # Errors
///
/// As [`one_step_gmm`], plus propagation from the moment-covariance inverse.
pub fn two_step_gmm(
    x_cols: &[Vec<f64>],
    z_cols: &[Vec<f64>],
    y: &[f64],
    cov_weight: GmmWeight,
) -> Result<GmmFit, GmmError> {
    let d = Design::build(x_cols, z_cols, y)?;
    let rw = cov_weight.resolve(d.n)?;
    // Step 1: 2SLS.
    let w1 = d.initial_weight()?;
    let beta1 = d.point_estimate(w1.as_ref())?;
    let u1 = d.residuals(&beta1);
    // Step 2: efficient weight from step-1 residuals.
    let s1 = d.moment_cov(&u1, rw)?;
    let w2 = inv_spd(s1.as_ref(), "step-2 GMM weight S(u1)")?;
    finish(&d, w2, 2, rw)
}

/// Iterated efficient GMM: repeat the (re-weight, re-estimate) loop until the
/// coefficient vector stops moving.
///
/// Starting from the two-step weight, each iteration recomputes
/// `W = S(u)^{-1}` at the current residuals and re-estimates `beta`, stopping
/// when the maximum absolute coefficient change falls below `tol` or
/// `max_iter` weight updates have been performed. At the fixed point
/// `W = S(u)^{-1}` exactly, so the sandwich covariance collapses to the
/// efficient `(G' W G)^{-1} / n`. On well-identified data this typically
/// equals the two-step estimate within a couple of iterations.
///
/// # Errors
///
/// As [`two_step_gmm`]; [`GmmError::InvalidArgument`] if `tol <= 0` or
/// `max_iter == 0`.
pub fn iterated_gmm(
    x_cols: &[Vec<f64>],
    z_cols: &[Vec<f64>],
    y: &[f64],
    cov_weight: GmmWeight,
    tol: f64,
    max_iter: usize,
) -> Result<GmmFit, GmmError> {
    if tol <= 0.0 || !tol.is_finite() {
        return Err(GmmError::InvalidArgument {
            what: "iterated GMM tolerance must be a positive finite number",
        });
    }
    if max_iter == 0 {
        return Err(GmmError::InvalidArgument {
            what: "iterated GMM max_iter must be at least 1",
        });
    }
    let d = Design::build(x_cols, z_cols, y)?;
    let rw = cov_weight.resolve(d.n)?;
    // Start at 2SLS.
    let w1 = d.initial_weight()?;
    let mut beta = d.point_estimate(w1.as_ref())?;
    let mut u = d.residuals(&beta);
    let mut steps = 1usize;
    let mut w = w1;
    for _ in 0..max_iter {
        let s = d.moment_cov(&u, rw)?;
        w = inv_spd(s.as_ref(), "iterated GMM weight S(u)")?;
        let beta_new = d.point_estimate(w.as_ref())?;
        steps += 1;
        let delta = beta
            .iter()
            .zip(beta_new.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        beta = beta_new;
        u = d.residuals(&beta);
        if delta < tol {
            break;
        }
    }
    // Covariance/J at the converged estimate, using the final weight `w`.
    let s_final = d.moment_cov(&u, rw)?;
    let (cov, bse, jtest) = d.cov_and_j(w.as_ref(), s_final.as_ref(), &u)?;
    Ok(GmmFit {
        params: beta,
        bse,
        cov,
        residuals: u,
        nobs: d.n,
        nmoments: d.l,
        nparams: d.k,
        steps,
        jtest,
        first_stage: d.first_stage.clone(),
        hac_bandwidth: rw.bandwidth(),
    })
}

/// Shared tail: estimate `beta` with the final weight `w`, compute residuals,
/// the moment covariance at those residuals, and the sandwich cov + J-test.
/// Used by the one-step and 2SLS entry points (where `w` is the final weight).
fn finish(
    d: &Design,
    w: Mat<f64>,
    steps: usize,
    cov_weight: ResolvedWeight,
) -> Result<GmmFit, GmmError> {
    let beta = d.point_estimate(w.as_ref())?;
    let u = d.residuals(&beta);
    let s = d.moment_cov(&u, cov_weight)?;
    let (cov, bse, jtest) = d.cov_and_j(w.as_ref(), s.as_ref(), &u)?;
    Ok(GmmFit {
        params: beta,
        bse,
        cov,
        residuals: u,
        nobs: d.n,
        nmoments: d.l,
        nparams: d.k,
        steps,
        jtest,
        first_stage: d.first_stage.clone(),
        hac_bandwidth: cov_weight.bandwidth(),
    })
}

#[cfg(test)]
mod tests {
    // The crate-level `warn(clippy::expect_used)` exists so library code never
    // panics on user input. In a test an unexpected `Err` *should* abort the
    // test, which is what `expect` does.
    #![allow(clippy::expect_used)]

    use super::*;

    /// `same_column` must compare **per element, relative to that element's
    /// own magnitude**. The rule it replaced scaled one absolute tolerance by
    /// the column maximum, `|a_i - b_i| <= 1e-12 * max(1, max_j |a_j|)`, which
    /// fails in both directions; both are pinned here.
    #[test]
    fn same_column_is_per_element_relative() {
        // Identical columns, including all-zero and mixed-sign.
        let a = vec![1.0, -2.5, 0.0, 1e13, 1e-13];
        assert!(same_column(&a, &a));
        assert!(same_column(&[0.0, 0.0], &[0.0, 0.0]));
        assert!(
            !same_column(&[1.0, 2.0], &[1.0, 2.0, 3.0]),
            "length differs"
        );

        // MERGE FAILURE (the silent one). A column whose maximum is 1e13 gave
        // the old rule an effective tolerance of 1e-12 * 1e13 = 10 at EVERY
        // observation. These two columns differ by 5 at each of the small
        // entries — plainly different variables — and were declared the same,
        // which makes an endogenous regressor look exogenous and silently
        // strips its first-stage F entirely.
        let big_a = vec![1e13_f64, 1.0, 2.0, 3.0];
        let big_b = vec![1e13_f64, 6.0, 7.0, 8.0];
        let old_tolerance = 1e-12 * big_a.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        assert!(
            big_a
                .iter()
                .zip(big_b.iter())
                .all(|(x, y)| (x - y).abs() <= old_tolerance),
            "the fixture must sit inside the OLD rule's blind spot (tol = {old_tolerance})"
        );
        assert!(
            !same_column(&big_a, &big_b),
            "columns differing by 5 at every small entry are not the same variable"
        );

        // SPLIT FAILURE (the loud one). The same *relative* discrepancy must
        // get the same answer at every magnitude. The old rule floored its
        // scale at 1.0, so a relative 1e-9 was absorbed on a small column and
        // rejected on a large one.
        for scale in [1e-6_f64, 1.0, 1e6, 1e12] {
            let base: Vec<f64> = (1..=8).map(|i| scale * i as f64).collect();
            let off: Vec<f64> = base.iter().map(|v| v * (1.0 + 1e-9)).collect();
            let near: Vec<f64> = base.iter().map(|v| v * (1.0 + 1e-14)).collect();
            assert!(
                !same_column(&base, &off),
                "relative 1e-9 is a different column at scale {scale}"
            );
            assert!(
                same_column(&base, &near),
                "relative 1e-14 is the same column at scale {scale}"
            );
        }

        // The comparison is symmetric: `max(|x|, |y|)`, not `|x|`, so argument
        // order cannot change the verdict.
        let p = vec![1.0, 2.0];
        let q = vec![1.0, 2.0 * (1.0 + 1e-9)];
        assert_eq!(same_column(&p, &q), same_column(&q, &p));
    }

    /// `f_sf` must distinguish NaN from a genuinely non-positive statistic.
    /// It used to fold both into `Ok(1.0)`, so a broken F came back dressed as
    /// a confident "no evidence against a weak first stage".
    #[test]
    fn f_sf_distinguishes_nan_from_a_zero_statistic() {
        // A NaN statistic has no p-value.
        assert!(matches!(
            f_sf(f64::NAN, 2.0, 100.0),
            Err(GmmError::NonFinite { .. })
        ));
        // Genuinely non-positive statistics keep sf = 1.
        for x in [0.0_f64, -1.0, f64::NEG_INFINITY] {
            assert_eq!(f_sf(x, 2.0, 100.0), Ok(1.0), "sf at x = {x}");
        }
        // A statistic of +inf is a p-value of exactly 0, not 1.
        assert_eq!(f_sf(f64::INFINITY, 2.0, 100.0), Ok(0.0));
        // And an ordinary value is a strictly interior probability that
        // decreases in the statistic.
        let mid = f_sf(3.0, 2.0, 100.0).expect("finite");
        let hi = f_sf(30.0, 2.0, 100.0).expect("finite");
        assert!(
            (0.0..1.0).contains(&mid) && hi > 0.0 && hi < mid,
            "{mid} {hi}"
        );
    }
}
