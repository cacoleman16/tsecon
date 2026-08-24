//! Generalized correlation dynamics: the cDCC (Aielli 2013) and ADCC
//! (Cappiello-Engle-Sheppard 2006) variants of the scalar DCC recursion,
//! and the Student-t second-stage likelihood.
//!
//! The plain DCC + Gaussian path stays in [`crate::dcc`] **untouched** so the
//! default estimator is bit-identical to earlier releases; everything
//! opt-in routes through the machinery here.
//!
//! # The three recursions
//!
//! All variants share `R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}` and
//! differ in how the auxiliary `Q_t` advances:
//!
//! ```text
//! DCC   (Engle 2002):    Q_t = (1-a-b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}
//! cDCC  (Aielli 2013):   Q_t = (1-a-b) S    + a z*_{t-1} z*_{t-1}' + b Q_{t-1},
//!                        z*_t = diag(Q_t)^{1/2} z_t
//! ADCC  (CES 2006):      Q_t = (1-a-b) Qbar - g Nbar
//!                              + a z_{t-1} z_{t-1}' + g n_{t-1} n_{t-1}' + b Q_{t-1},
//!                        n_t = min(z_t, 0)  (elementwise)
//! ```
//!
//! **Why cDCC exists.** In Engle's DCC the driving term has conditional mean
//! `E_{t-1}[z_t z_t'] = R_t`, *not* `Q_t` — the recursion is not a proper
//! GARCH-like process for `Q_t`, so targeting `Qbar` with the sample second
//! moment of `z_t` is **inconsistent** (the bias is small at typical
//! parameter values, which is why DCC survives in practice, but it does not
//! vanish as `T -> infinity`). Aielli's correction rescales the driver to
//! `z*_t = diag(Q_t)^{1/2} z_t`, for which `E_{t-1}[z*_t z*_t'] = Q_t`
//! exactly; the recursion becomes a well-defined process with unconditional
//! mean `S`, and targeting `S` by the (correlation-normalized) sample second
//! moment of `z*_t` is consistent. A convenient corollary: the diagonal
//! `q_{ii,t}` follows the *univariate* recursion
//! `q_{ii,t} = (1-a-b) + a q_{ii,t-1} z_{i,t-1}^2 + b q_{ii,t-1}` (because
//! `s_{ii} = 1`), so `z*` — and hence `S` — is computable from `(a, b)`
//! alone, no fixed-point iteration required.
//!
//! **ADCC.** The asymmetric term `g n_{t-1} n_{t-1}'` lets *joint negative*
//! news move correlations more than joint positive news (the documented
//! equity stylized fact). Targeting subtracts `g Nbar`
//! (`Nbar = (1/T) sum_t n_t n_t'`) from the intercept so the unconditional
//! level stays `Qbar`. The sufficient stationarity/positivity constraint is
//! `a + b + delta g < 1` with `delta = lambda_max(Qbar^{-1/2} Nbar
//! Qbar^{-1/2})` (Cappiello-Engle-Sheppard 2006): it keeps the intercept
//! `(1-a-b) Qbar - g Nbar` positive semi-definite, which — with the other
//! terms PSD — keeps every `Q_t` positive-definite. At `g = 0` ADCC **is**
//! DCC (the nesting test asserts this numerically).

use tsecon_linalg::faer::{Mat, MatRef, Side};
use tsecon_stats::special::ln_gamma;

use crate::error::MgarchError;
use crate::stage::UnivariateStage;
use crate::util::{cholesky, corr_from_cov, moment_matrix, quad_form};

/// Which scalar correlation recursion drives `Q_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DccVariant {
    /// Engle (2002) — the default; targeting is the (slightly inconsistent)
    /// sample second moment of `z_t`.
    Dcc,
    /// Aielli (2013) corrected DCC: driver `z*_t = diag(Q_t)^{1/2} z_t`,
    /// making correlation targeting consistent.
    Cdcc,
    /// Cappiello-Engle-Sheppard (2006) asymmetric DCC: adds
    /// `g n_{t-1} n_{t-1}'` with `n_t = min(z_t, 0)`.
    Adcc,
}

/// The second-stage innovation distribution for the correlation likelihood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrDist {
    /// Multivariate Gaussian — the default (QMLE).
    Normal,
    /// Standardized multivariate Student-t with `nu > 2` degrees of freedom
    /// estimated jointly with the correlation parameters in step 2 (step 1
    /// stays whatever the univariate [`tsecon_garch::GarchSpec`] says).
    StudentT,
}

/// Everything the second-stage objective needs from one walk of a variant's
/// `Q` recursion at fixed parameters.
pub(crate) struct DynamicsEval {
    /// `ln|R_t|` per `t`.
    pub ln_det_r: Vec<f64>,
    /// `z_t' R_t^{-1} z_t` per `t`.
    pub quad: Vec<f64>,
    /// The correlation path (only when `want_path`).
    pub r_path: Option<Vec<Mat<f64>>>,
    /// `Q_{T+1}` — the exact one-step-ahead auxiliary matrix.
    pub q_next: Mat<f64>,
    /// The targeting matrix actually used: `Qbar` (DCC/ADCC) or Aielli's
    /// `S` (cDCC, recomputed per `(a, b)`).
    pub target: Mat<f64>,
}

/// The scalar dynamic parameters of one recursion evaluation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DynParams {
    /// News coefficient.
    pub a: f64,
    /// Persistence coefficient.
    pub b: f64,
    /// Asymmetric news coefficient (`0.0` for non-ADCC variants).
    pub g: f64,
}

/// Walks the `Q` recursion of `variant` at `(a, b, g)` and evaluates the
/// per-`t` correlation terms.
///
/// `qbar` is the sample second moment of `z` (`moment_matrix`); for cDCC it
/// is only the *seed* — the actual target `S` is rebuilt from `z*` inside.
/// `nbar` must be `Some` exactly for ADCC.
///
/// # Errors
///
/// [`MgarchError::Linalg`] if some `R_t` cannot be factorized;
/// [`MgarchError::NonFinite`] if the cDCC diagonal recursion degenerates.
pub(crate) fn eval_dynamics(
    stage: &UnivariateStage,
    variant: DccVariant,
    qbar: MatRef<'_, f64>,
    nbar: Option<MatRef<'_, f64>>,
    params: DynParams,
    want_path: bool,
) -> Result<DynamicsEval, MgarchError> {
    let DynParams { a, b, g } = params;
    let k = stage.k;
    let t_obs = stage.nobs;

    // The driver series d_t and the targeting matrix.
    // DCC/ADCC: d_t = z_t, target = qbar. cDCC: d_t = z*_t, target = S.
    let (driver, target): (Vec<Vec<f64>>, Mat<f64>) = match variant {
        DccVariant::Cdcc => {
            let zstar = cdcc_driver(stage, a, b)?;
            let s = corr_from_cov(moment_matrix(&zstar, k).as_ref());
            (zstar, s)
        }
        DccVariant::Dcc | DccVariant::Adcc => (stage.z.clone(), qbar.to_owned()),
    };

    // Intercept matrix: (1-a-b) target, minus g Nbar for ADCC.
    let omega = 1.0 - a - b;
    let intercept: Mat<f64> = match (variant, nbar) {
        (DccVariant::Adcc, Some(nb)) => {
            Mat::from_fn(k, k, |i, j| omega * target[(i, j)] - g * nb[(i, j)])
        }
        (DccVariant::Adcc, None) => {
            return Err(MgarchError::NonFinite {
                what: "ADCC evaluation without Nbar (internal invariant)",
            })
        }
        _ => Mat::from_fn(k, k, |i, j| omega * target[(i, j)]),
    };

    let mut q: Mat<f64> = target.to_owned();
    let mut ln_det_r = Vec::with_capacity(t_obs);
    let mut quad = Vec::with_capacity(t_obs);
    let mut r_path = want_path.then(|| Vec::with_capacity(t_obs));
    for (z_t, d) in stage.z.iter().zip(driver.iter()) {
        let r = corr_from_cov(q.as_ref());
        let chol = cholesky(r.as_ref())?;
        ln_det_r.push(chol.log_det());
        quad.push(quad_form(chol.factor.as_ref(), z_t));
        if let Some(path) = r_path.as_mut() {
            path.push(r);
        }
        // Advance: Q_{t+1} = intercept + a d_t d_t' + [g n_t n_t'] + b Q_t.
        q = match variant {
            DccVariant::Adcc => Mat::from_fn(k, k, |i, j| {
                let ni = z_t[i].min(0.0);
                let nj = z_t[j].min(0.0);
                intercept[(i, j)] + a * d[i] * d[j] + g * ni * nj + b * q[(i, j)]
            }),
            _ => Mat::from_fn(k, k, |i, j| {
                intercept[(i, j)] + a * d[i] * d[j] + b * q[(i, j)]
            }),
        };
    }
    Ok(DynamicsEval {
        ln_det_r,
        quad,
        r_path,
        q_next: q,
        target,
    })
}

/// The cDCC driver `z*_t = diag(Q_t)^{1/2} z_t` from the closed univariate
/// diagonal recursion `q_{ii,t} = (1-a-b) + a z*^2_{i,t-1} + b q_{ii,t-1}`,
/// `q_{ii,0} = 1` (Aielli 2013 — computable because `s_{ii} = 1`).
fn cdcc_driver(stage: &UnivariateStage, a: f64, b: f64) -> Result<Vec<Vec<f64>>, MgarchError> {
    let k = stage.k;
    let omega = 1.0 - a - b;
    let mut qdiag = vec![1.0_f64; k];
    let mut zstar = Vec::with_capacity(stage.nobs);
    for t in 0..stage.nobs {
        let mut row = vec![0.0_f64; k];
        for i in 0..k {
            if !(qdiag[i].is_finite() && qdiag[i] > 0.0) {
                return Err(MgarchError::NonFinite {
                    what: "cDCC diagonal recursion (q_ii not finite-positive)",
                });
            }
            row[i] = qdiag[i].sqrt() * stage.z[t][i];
        }
        for i in 0..k {
            qdiag[i] = omega + a * row[i] * row[i] + b * qdiag[i];
        }
        zstar.push(row);
    }
    Ok(zstar)
}

/// `Nbar = (1/T) sum_t n_t n_t'` with `n_t = min(z_t, 0)` — the targeting
/// matrix of the ADCC asymmetric term.
pub(crate) fn asymmetric_moment(stage: &UnivariateStage) -> Mat<f64> {
    let n: Vec<Vec<f64>> = stage
        .z
        .iter()
        .map(|row| row.iter().map(|&z| z.min(0.0)).collect())
        .collect();
    moment_matrix(&n, stage.k)
}

/// The ADCC constraint scale `delta = lambda_max(Qbar^{-1/2} Nbar
/// Qbar^{-1/2})` (Cappiello-Engle-Sheppard 2006). Computed as the largest
/// eigenvalue of the symmetric PSD matrix `L^{-1} Nbar L^{-T}` with
/// `Qbar = L L'`.
///
/// # Errors
///
/// [`MgarchError::Linalg`] if `Qbar` cannot be factorized or the symmetric
/// eigensolver fails.
pub(crate) fn adcc_delta(qbar: MatRef<'_, f64>, nbar: MatRef<'_, f64>) -> Result<f64, MgarchError> {
    let k = qbar.nrows();
    let chol = cholesky(qbar)?;
    let l = &chol.factor;
    // W = L^{-1} Nbar: forward-substitute each column.
    let mut w = nbar.to_owned();
    forward_solve_in_place(l.as_ref(), &mut w);
    // M = W L^{-T} = (L^{-1} W')': forward-substitute the transpose.
    let mut wt = w.transpose().to_owned();
    forward_solve_in_place(l.as_ref(), &mut wt);
    let m_raw = wt.transpose().to_owned();
    // Symmetrize the float dust before the eigensolver.
    let m = Mat::from_fn(k, k, |i, j| 0.5 * (m_raw[(i, j)] + m_raw[(j, i)]));
    let eigen = m.self_adjoint_eigen(Side::Lower).map_err(|_| {
        MgarchError::Linalg(tsecon_linalg::LinalgError::EigenFailed {
            what: "ADCC constraint scale delta = lambda_max(Qbar^{-1/2} Nbar Qbar^{-1/2})",
        })
    })?;
    let s = eigen.S();
    let mut max = f64::NEG_INFINITY;
    for i in 0..k {
        max = max.max(s.column_vector()[i]);
    }
    // Nbar is PSD, so the spectrum is nonnegative up to float dust.
    Ok(max.max(0.0))
}

/// In-place forward substitution `X <- L^{-1} X` for lower-triangular `L`.
fn forward_solve_in_place(l: MatRef<'_, f64>, x: &mut Mat<f64>) {
    let n = l.nrows();
    for col in 0..x.ncols() {
        for i in 0..n {
            let mut s = x[(i, col)];
            for j in 0..i {
                s -= l[(i, j)] * x[(j, col)];
            }
            x[(i, col)] = s / l[(i, i)];
        }
    }
}

/// The full Gaussian log-likelihood from a recursion evaluation:
/// `-0.5 sum_t [ k ln(2 pi) + sum_i ln sigma2_{i,t} + ln|R_t| + quad_t ]`.
///
/// `ln_det_d2[t] = sum_i ln sigma2_{i,t}` is precomputed once per fit.
pub(crate) fn gaussian_loglik(k: usize, ln_det_d2: &[f64], eval: &DynamicsEval) -> f64 {
    let ln_2pi = (2.0 * core::f64::consts::PI).ln();
    let kf = k as f64;
    let mut ll = 0.0;
    for ((&ldd, &ldr), &qd) in ln_det_d2.iter().zip(&eval.ln_det_r).zip(&eval.quad) {
        ll += -0.5 * (kf * ln_2pi + ldd + ldr + qd);
    }
    ll
}

/// The full standardized multivariate Student-t log-likelihood from a
/// recursion evaluation:
///
/// ```text
/// sum_t [ C(nu) - 0.5 (sum_i ln sigma2_{i,t} + ln|R_t|)
///         - ((nu + k)/2) ln(1 + quad_t / (nu - 2)) ],
/// C(nu) = ln G((nu+k)/2) - ln G(nu/2) - (k/2) ln((nu-2) pi),
/// ```
///
/// the density of `eps_t = D_t z_t` when `z_t | F_{t-1}` follows the
/// unit-variance (standardized) multivariate t with correlation `R_t` and a
/// single common `nu > 2`.
pub(crate) fn student_t_loglik(k: usize, ln_det_d2: &[f64], eval: &DynamicsEval, nu: f64) -> f64 {
    let kf = k as f64;
    let c = ln_gamma(0.5 * (nu + kf))
        - ln_gamma(0.5 * nu)
        - 0.5 * kf * ((nu - 2.0) * core::f64::consts::PI).ln();
    let half_nuk = 0.5 * (nu + kf);
    let mut ll = 0.0;
    for ((&ldd, &ldr), &qd) in ln_det_d2.iter().zip(&eval.ln_det_r).zip(&eval.quad) {
        ll += c - 0.5 * (ldd + ldr) - half_nuk * (qd / (nu - 2.0)).ln_1p();
    }
    ll
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::dcc::dcc_full_loglik;
    use crate::stage::UnivariateStage;
    use tsecon_garch::{DistSpec, GarchSpec, MeanSpec, VolSpec};

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn uniform(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.uniform();
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn spec() -> GarchSpec {
        GarchSpec {
            mean: MeanSpec::Zero,
            vol: VolSpec::Garch { p: 1, q: 1 },
            dist: DistSpec::Normal,
        }
    }

    fn synthetic() -> Vec<Vec<f64>> {
        let mut rng = Rng(0x5EED_0DCC);
        let n = 400;
        let (mut s0, mut s1) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut v0, mut v1) = (1.0_f64, 1.0_f64);
        for _ in 0..n {
            let e0 = rng.normal();
            let common = rng.normal();
            let e1 = 0.6 * e0 + 0.8 * rng.normal();
            let x0 = v0.sqrt() * e0 + 0.2 * common;
            let x1 = v1.sqrt() * e1 + 0.2 * common;
            v0 = 0.05 + 0.1 * x0 * x0 + 0.85 * v0;
            v1 = 0.04 + 0.08 * x1 * x1 + 0.88 * v1;
            s0.push(x0);
            s1.push(x1);
        }
        vec![s0, s1]
    }

    fn stage() -> UnivariateStage {
        UnivariateStage::fit(&synthetic(), spec()).unwrap()
    }

    fn ln_det_d2(stage: &UnivariateStage) -> Vec<f64> {
        stage
            .sigma2
            .iter()
            .map(|row| row.iter().map(|s2| s2.ln()).sum())
            .collect()
    }

    /// Nesting invariant: the ADCC recursion at `g = 0` is numerically the
    /// DCC recursion — same Gaussian log-likelihood as the classic path to
    /// 1e-10 (relative) at a representative interior `(a, b)`.
    #[test]
    fn adcc_g0_matches_classic_dcc_loglik() {
        let st = stage();
        let qbar = moment_matrix(&st.z, st.k);
        let nbar = asymmetric_moment(&st);
        let (a, b) = (0.04, 0.9);
        let eval = eval_dynamics(
            &st,
            DccVariant::Adcc,
            qbar.as_ref(),
            Some(nbar.as_ref()),
            DynParams { a, b, g: 0.0 },
            false,
        )
        .unwrap();
        let ll_general = gaussian_loglik(st.k, &ln_det_d2(&st), &eval);
        let ll_classic = dcc_full_loglik(&st, qbar.as_ref(), a, b).unwrap();
        assert!(
            (ll_general - ll_classic).abs() <= 1e-10 * ll_classic.abs().max(1.0),
            "ADCC(g=0) {ll_general} vs classic DCC {ll_classic}"
        );
    }

    /// The generalized DCC evaluation reproduces the classic path too (the
    /// general machinery is only used off-default, but it must agree where
    /// they overlap).
    #[test]
    fn general_dcc_matches_classic_dcc_loglik() {
        let st = stage();
        let qbar = moment_matrix(&st.z, st.k);
        let (a, b) = (0.03, 0.95);
        let eval = eval_dynamics(
            &st,
            DccVariant::Dcc,
            qbar.as_ref(),
            None,
            DynParams { a, b, g: 0.0 },
            false,
        )
        .unwrap();
        let ll_general = gaussian_loglik(st.k, &ln_det_d2(&st), &eval);
        let ll_classic = dcc_full_loglik(&st, qbar.as_ref(), a, b).unwrap();
        assert!(
            (ll_general - ll_classic).abs() <= 1e-10 * ll_classic.abs().max(1.0),
            "general DCC {ll_general} vs classic {ll_classic}"
        );
    }

    /// cDCC at `a = b = 0`: the diagonal recursion stays at 1, so `z* = z`,
    /// `S = corr(Qbar)`, and `Q_t = S` for all `t` — exactly the CCC special
    /// case, hence exactly the DCC(0, 0) log-likelihood.
    #[test]
    fn cdcc_at_zero_equals_ccc_special_case() {
        let st = stage();
        let qbar = moment_matrix(&st.z, st.k);
        let eval = eval_dynamics(
            &st,
            DccVariant::Cdcc,
            qbar.as_ref(),
            None,
            DynParams {
                a: 0.0,
                b: 0.0,
                g: 0.0,
            },
            false,
        )
        .unwrap();
        let ll_cdcc0 = gaussian_loglik(st.k, &ln_det_d2(&st), &eval);
        let ll_dcc0 = dcc_full_loglik(&st, qbar.as_ref(), 0.0, 0.0).unwrap();
        assert!(
            (ll_cdcc0 - ll_dcc0).abs() <= 1e-10 * ll_dcc0.abs().max(1.0),
            "cDCC(0,0) {ll_cdcc0} vs DCC(0,0) {ll_dcc0}"
        );
        // And the cDCC target has an exactly-unit diagonal.
        for i in 0..st.k {
            assert!((eval.target[(i, i)] - 1.0).abs() <= 1e-15);
        }
    }

    /// The Student-t second-stage likelihood converges to the Gaussian one
    /// as `nu -> infinity` (checked at nu = 1e6, relative 1e-4).
    #[test]
    fn student_t_limits_to_gaussian() {
        let st = stage();
        let qbar = moment_matrix(&st.z, st.k);
        let eval = eval_dynamics(
            &st,
            DccVariant::Dcc,
            qbar.as_ref(),
            None,
            DynParams {
                a: 0.03,
                b: 0.95,
                g: 0.0,
            },
            false,
        )
        .unwrap();
        let ldd = ln_det_d2(&st);
        let ll_gauss = gaussian_loglik(st.k, &ldd, &eval);
        let ll_t = student_t_loglik(st.k, &ldd, &eval, 1.0e6);
        assert!(
            (ll_t - ll_gauss).abs() <= 1e-4 * ll_gauss.abs().max(1.0),
            "t(nu=1e6) {ll_t} vs Gaussian {ll_gauss}"
        );
        // And a genuinely fat-tailed nu is a *different* likelihood.
        let ll_t5 = student_t_loglik(st.k, &ldd, &eval, 5.0);
        assert!((ll_t5 - ll_gauss).abs() > 1e-3 * ll_gauss.abs().max(1.0));
    }

    /// `delta = lambda_max(Qbar^{-1/2} Qbar Qbar^{-1/2}) = 1` exactly when
    /// `Nbar` is taken to be `Qbar` itself — a closed-form check of the
    /// ADCC constraint-scale computation.
    #[test]
    fn adcc_delta_of_qbar_is_one() {
        let st = stage();
        let qbar = moment_matrix(&st.z, st.k);
        let delta = adcc_delta(qbar.as_ref(), qbar.as_ref()).unwrap();
        assert!((delta - 1.0).abs() <= 1e-10, "delta {delta}");
        // The real Nbar (negative parts only) is dominated by Qbar, so its
        // scale is strictly inside (0, 1] up to sampling noise.
        let nbar = asymmetric_moment(&st);
        let d = adcc_delta(qbar.as_ref(), nbar.as_ref()).unwrap();
        assert!(d > 0.0 && d < 1.5, "delta(Nbar) {d}");
    }
}
