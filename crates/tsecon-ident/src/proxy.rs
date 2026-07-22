//! Proxy SVAR / external-instrument identification (SVAR-IV): the
//! single-instrument, single-target-shock case (Stock & Watson 2018;
//! Mertens & Ravn 2013; Gertler & Karadi 2015; Montiel-Olea, Stock &
//! Watson 2021).
//!
//! # The identifying moment
//!
//! A reduced-form VAR leaves innovations `u_t` (`n`-vector) with
//! `u_t = H eps_t`, `E[eps eps'] = I`, so `Sigma_u = H H'`. An external
//! instrument (proxy) `m_t` is *relevant* for the target shock
//! (`E[m_t eps_{1t}] = phi != 0`) and *exogenous* to every other structural
//! shock (`E[m_t eps_{jt}] = 0`, `j != 1`). The residual-instrument
//! covariance is then
//!
//! ```text
//! gamma := E[m_t u_t] = H E[m_t eps_t] = H [phi; 0] = phi * h_col,
//! ```
//!
//! *exactly proportional* to the impact column `h_col` of the target shock.
//! The proportionality constant `phi` is the only thing left free, so the
//! impact column is point-identified up to a single scalar: normalizing on
//! the `norm_var` entry gives the scale-free **relative impact**
//! `rho = gamma / gamma[norm_var]` (with `rho[norm_var] = 1`).
//!
//! # Unit-effect normalization (Montiel-Olea-Stock-Watson)
//!
//! The MOSW baseline fixes the remaining scale and sign by the *unit-effect*
//! convention: scale the shock so that its impact on `norm_var` equals
//! `unit` (default `+1`). The impact vector is `b = unit * rho`, so
//! `b[norm_var] = unit` exactly and a positive shock raises `norm_var` by
//! `+unit` on impact. The full structural impulse response propagates `b`
//! through the reduced-form MA matrices: `irf_h = Psi_h b`.
//!
//! The **one-standard-deviation (unit-variance) normalization is
//! deliberately omitted**: the just-identified single-instrument case does
//! not point-identify that scale without over-identifying content (the
//! Mertens-Ravn quadratic needs a second moment), so v1 reports only the
//! unit-effect object.
//!
//! # Instrument strength
//!
//! The first-stage regression of the `norm_var` residual on the proxy gives
//! the effective-`F` (HC1-robust by default, mirroring the
//! Montiel-Olea-Pflueger effective `F` used by the local-projection IV path)
//! and the Stock-Watson (2018) **reliability** `= Corr(m, u_norm)^2` over the
//! overlap sample. An `F` below 10 flags a weak instrument.
//!
//! # Honest limitations
//!
//! * Just-identified: no over-identification test exists, and the
//!   exogeneity condition is untestable with a single instrument.
//! * v1 ships **no inference bands**. The naive residual/wild bootstrap is
//!   invalid for proxy SVARs (Jentsch & Lunsford 2019, AER); correct bands
//!   need their moving-block bootstrap and are a documented v2 extension.
//! * `reliability` is the first-stage `R^2` (Stock-Watson definition); the
//!   population `Corr(m, eps_1)^2` is not point-identified here.
//!
//! This module takes **reduced-form quantities** (residuals, the MA
//! matrices, `Sigma_u`) as inputs, exactly like the sign-restriction sampler
//! consumes a posterior rather than fitting a VAR, so the crate keeps its
//! boundary: `tsecon-ident` does not depend on `tsecon-var`. The caller (the
//! Python binding) fits the reduced-form VAR and passes the pieces in.

use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::IdentError;

/// The identified structural object from a single-instrument proxy SVAR.
///
/// Every vector is length `n` (the number of variables) unless noted;
/// `irf` is `(horizon + 1)` rows of length `n`. All quantities are the
/// unit-effect-normalized point estimate (no error bands in v1).
#[derive(Debug, Clone)]
pub struct ProxySvarResult {
    /// Structural impulse response `irf[h] = Psi_h b` of every variable to
    /// the identified shock, `h = 0..=horizon`. `irf[0]` is the impact
    /// vector `b`, with `irf[0][norm_var] == unit` exactly.
    pub irf: Vec<Vec<f64>>,
    /// Impact vector `b = unit * rho` (equals `irf[0]`).
    pub impact: Vec<f64>,
    /// Relative impact `rho = gamma / gamma[norm_var]`, scale-free with
    /// `rho[norm_var] == 1.0` exactly.
    pub relative_impact: Vec<f64>,
    /// First-stage instrument-strength `F` (HC1-robust when `robust_f`,
    /// classical otherwise). Below 10 signals a weak instrument.
    pub first_stage_f: f64,
    /// Stock-Watson reliability `= Corr(m, u_norm)^2` over the overlap, in
    /// `[0, 1]`.
    pub reliability: f64,
    /// The identifying moment `gamma = E[m_t u_t]` (a diagnostic; `impact`
    /// and `relative_impact` are derived from it).
    pub cov_um: Vec<f64>,
    /// Effective number of proxy observations used — `|O|`, the count of
    /// finite (non-NaN) proxy entries in the residual sample.
    pub n_proxy: usize,
    /// Minimum-variance estimate of the target structural shock over the
    /// full residual sample, `eps1_hat_t = (b' Sigma_u^{-1} u_t) /
    /// (b' Sigma_u^{-1} b)` (length `T`).
    pub shock: Vec<f64>,
}

/// Identify a single structural shock from an external instrument by the
/// SVAR-IV method of moments.
///
/// `u` is the `T x n` matrix of reduced-form VAR residuals; `proxy` is the
/// length-`T` instrument aligned to the residual rows, with non-finite
/// (NaN) entries dropped from the moments and first stage; `psi` is the
/// reduced-form MA sequence `Psi_0..Psi_H` (each `n x n`, `Psi_0 = I`);
/// `sigma_u` is the `n x n` residual covariance. `norm_var` selects the
/// variable whose impact is normalized to `unit`; `robust_f` chooses the
/// HC1-robust (default) versus classical first-stage `F`.
///
/// See the module documentation for the algebra and the honest limitations.
///
/// # Errors
///
/// * [`IdentError::Dimension`] if `proxy`, `psi`, or `sigma_u` do not match
///   the `T x n` residual shape;
/// * [`IdentError::RestrictionOutOfRange`] if `norm_var >= n`;
/// * [`IdentError::NonFinite`] if `u`, `sigma_u`, or `psi` contain a
///   NaN/infinity (the proxy may carry NaNs — they mark unavailability);
/// * [`IdentError::InvalidArgument`] if `unit` is zero or non-finite, the
///   proxy overlap has fewer than three finite observations, the proxy has
///   no variance over the overlap, the instrument has no first-stage
///   relevance for `norm_var` (`gamma[norm_var]` is zero), or `sigma_u` is
///   not positive definite.
#[allow(clippy::too_many_arguments)]
pub fn proxy_svar(
    u: MatRef<'_, f64>,
    proxy: &[f64],
    psi: &[Mat<f64>],
    sigma_u: MatRef<'_, f64>,
    norm_var: usize,
    unit: f64,
    robust_f: bool,
) -> Result<ProxySvarResult, IdentError> {
    let t = u.nrows();
    let n = u.ncols();

    if n == 0 || t == 0 {
        return Err(IdentError::InvalidArgument {
            what: "residual matrix u must have at least one row and one column",
        });
    }
    if proxy.len() != t {
        return Err(IdentError::Dimension {
            what: "proxy length must equal the number of residual rows T",
            expected: t,
            got: proxy.len(),
        });
    }
    if psi.is_empty() {
        return Err(IdentError::InvalidArgument {
            what: "psi must contain at least Psi_0 (the identity)",
        });
    }
    for ph in psi {
        if ph.nrows() != n || ph.ncols() != n {
            return Err(IdentError::Dimension {
                what: "every MA matrix Psi_h must be n x n",
                expected: n,
                got: if ph.nrows() != n {
                    ph.nrows()
                } else {
                    ph.ncols()
                },
            });
        }
    }
    if sigma_u.nrows() != n || sigma_u.ncols() != n {
        return Err(IdentError::Dimension {
            what: "sigma_u must be n x n",
            expected: n,
            got: if sigma_u.nrows() != n {
                sigma_u.nrows()
            } else {
                sigma_u.ncols()
            },
        });
    }
    if norm_var >= n {
        return Err(IdentError::RestrictionOutOfRange {
            what: "norm_var",
            index: norm_var,
            bound: n,
        });
    }
    if !unit.is_finite() || unit == 0.0 {
        return Err(IdentError::InvalidArgument {
            what: "unit (the impact size on norm_var) must be nonzero and finite",
        });
    }

    // Finiteness: residuals and covariance must be clean; the proxy may
    // carry NaNs (they mark observations outside the instrument's window).
    for j in 0..n {
        for i in 0..t {
            if !u[(i, j)].is_finite() {
                return Err(IdentError::NonFinite { what: "u" });
            }
        }
    }
    for j in 0..n {
        for i in 0..n {
            if !sigma_u[(i, j)].is_finite() {
                return Err(IdentError::NonFinite { what: "sigma_u" });
            }
        }
    }
    for ph in psi {
        for j in 0..n {
            for i in 0..n {
                if !ph[(i, j)].is_finite() {
                    return Err(IdentError::NonFinite { what: "psi" });
                }
            }
        }
    }

    // Overlap: the residual rows where the proxy is available (finite).
    let overlap: Vec<usize> = (0..t).filter(|&r| proxy[r].is_finite()).collect();
    let n_proxy = overlap.len();
    if n_proxy < 3 {
        return Err(IdentError::InvalidArgument {
            what: "proxy overlap has fewer than 3 finite observations; cannot run the first stage",
        });
    }
    let no = n_proxy as f64;

    // Means over the overlap.
    let mut mbar = 0.0;
    for &r in &overlap {
        mbar += proxy[r];
    }
    mbar /= no;
    let mut ubar = vec![0.0f64; n];
    for j in 0..n {
        let mut s = 0.0;
        for &r in &overlap {
            s += u[(r, j)];
        }
        ubar[j] = s / no;
    }

    // Identifying moment gamma_j = mean_O (m - mbar)(u_j - ubar_j).
    let mut gamma = vec![0.0f64; n];
    for j in 0..n {
        let mut s = 0.0;
        for &r in &overlap {
            s += (proxy[r] - mbar) * (u[(r, j)] - ubar[j]);
        }
        gamma[j] = s / no;
    }

    // gamma is a finite sum of finite products (u and the overlap proxy are
    // finite), so an exact zero is the only unrecoverable case here.
    let g_norm = gamma[norm_var];
    if g_norm == 0.0 {
        return Err(IdentError::InvalidArgument {
            what:
                "gamma[norm_var] is zero: the instrument has no first-stage relevance for norm_var",
        });
    }

    // Relative impact rho (rho[norm_var] == 1.0 exactly) and unit-effect
    // impact vector b (b[norm_var] == unit exactly).
    let rho: Vec<f64> = gamma.iter().map(|&g| g / g_norm).collect();
    let b: Vec<f64> = rho.iter().map(|&r| unit * r).collect();

    // First-stage OLS of the norm_var residual on [1, m] over the overlap.
    let ybar = ubar[norm_var];
    let mut smm = 0.0;
    let mut smy = 0.0;
    let mut syy = 0.0;
    for &r in &overlap {
        let md = proxy[r] - mbar;
        let yd = u[(r, norm_var)] - ybar;
        smm += md * md;
        smy += md * yd;
        syy += yd * yd;
    }
    // smm is a finite sum of squares; zero means a constant proxy over O.
    if smm == 0.0 {
        return Err(IdentError::InvalidArgument {
            what: "instrument has zero variance over the overlap; no first stage",
        });
    }
    let beta = smy / smm;
    let reliability = if syy > 0.0 {
        smy * smy / (smm * syy)
    } else {
        0.0
    };

    // Residuals of the first stage, and the two effective-F variants.
    let dof = (n_proxy - 2) as f64;
    let mut sse = 0.0;
    let mut meat = 0.0;
    for &r in &overlap {
        let md = proxy[r] - mbar;
        let e = (u[(r, norm_var)] - ybar) - beta * md;
        sse += e * e;
        meat += md * md * e * e;
    }
    let first_stage_f = if robust_f {
        // HC1-robust: squared robust t = Montiel-Olea-Pflueger effective F.
        let var_hc1 = (no / dof) * meat / (smm * smm);
        beta * beta / var_hc1
    } else {
        let s2 = sse / dof;
        beta * beta * smm / s2
    };

    // Structural IRF: irf_h = Psi_h b, h = 0..=H. With Psi_0 = I the impact
    // row equals b exactly (irf[0][norm_var] == unit).
    let mut irf: Vec<Vec<f64>> = Vec::with_capacity(psi.len());
    for ph in psi {
        let mut row = vec![0.0f64; n];
        for (i, slot) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for (k, &bk) in b.iter().enumerate() {
                s += ph[(i, k)] * bk;
            }
            *slot = s;
        }
        irf.push(row);
    }

    // Structural shock series over the full residual sample:
    // eps1_hat_t = (b' Sigma_u^{-1} u_t) / (b' Sigma_u^{-1} b). Let
    // w = Sigma_u^{-1} b (one SPD solve); then the numerator is w . u_t and
    // the denominator is w . b, since Sigma_u is symmetric.
    let w = spd_solve(sigma_u, &b)?;
    let mut denom = 0.0;
    for (bk, wk) in b.iter().zip(w.iter()) {
        denom += bk * wk;
    }
    // denom = b' Sigma_u^{-1} b is positive for a PD Sigma_u and b != 0
    // (b[norm_var] = unit != 0); the guard is belt-and-suspenders.
    if denom == 0.0 {
        return Err(IdentError::InvalidArgument {
            what: "b' Sigma_u^{-1} b is zero; cannot normalize the shock series",
        });
    }
    let mut shock = vec![0.0f64; t];
    for (r, slot) in shock.iter_mut().enumerate() {
        let mut s = 0.0;
        for (j, &wj) in w.iter().enumerate() {
            s += u[(r, j)] * wj;
        }
        *slot = s / denom;
    }

    Ok(ProxySvarResult {
        irf,
        impact: b,
        relative_impact: rho,
        first_stage_f,
        reliability,
        cov_um: gamma,
        n_proxy,
        shock,
    })
}

/// Solve the symmetric positive-definite system `A x = b` by a lower
/// Cholesky factorization and forward/backward substitution. `A` is read as
/// its lower triangle (the caller guarantees symmetry). Returns
/// [`IdentError::InvalidArgument`] if `A` is not positive definite.
fn spd_solve(a: MatRef<'_, f64>, b: &[f64]) -> Result<Vec<f64>, IdentError> {
    let n = b.len();
    // Lower Cholesky factor L with A = L L'.
    let mut l = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                // s is finite (sigma_u is pre-checked finite); a nonpositive
                // pivot means the matrix is not positive definite.
                if s <= 0.0 {
                    return Err(IdentError::InvalidArgument {
                        what: "sigma_u is not positive definite",
                    });
                }
                l[(i, j)] = s.sqrt();
            } else {
                l[(i, j)] = s / l[(j, j)];
            }
        }
    }
    // Forward solve L y = b.
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[(i, k)] * y[k];
        }
        y[i] = s / l[(i, i)];
    }
    // Backward solve L' x = y.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[(k, i)] * x[k];
        }
        x[i] = s / l[(i, i)];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny hand-checkable case: a rank-one, target-shock-only DGP so the
    /// residual-instrument covariance is EXACTLY proportional to the impact
    /// column and the normalized relative impact has no sampling error.
    #[test]
    fn exact_proxy_recovers_impact_column() -> Result<(), IdentError> {
        // u_t = eps0_t * hcol (only the target shock is active), so
        // gamma_j = hcol_j * phi * Var_sample(eps0) and, normalizing on
        // entry 0, rho = hcol / hcol[0] = [1, 0.5] exactly (the common
        // phi*Var factor cancels), regardless of the eps0 draws.
        let hcol = [2.0, 1.0];
        let eps0 = [1.0, -0.5, 0.3, 0.8, -0.2, 0.6, -0.9, 0.4];
        let tt = eps0.len();
        let n = 2;
        let mut u = Mat::<f64>::zeros(tt, n);
        for (r, &e) in eps0.iter().enumerate() {
            for i in 0..n {
                u[(r, i)] = hcol[i] * e;
            }
        }
        let phi = 3.0;
        let proxy: Vec<f64> = eps0.iter().map(|&e| phi * e).collect();
        // Psi_0 = I, Psi_1 = A (some MA matrix).
        let psi = vec![
            Mat::<f64>::identity(n, n),
            Mat::from_fn(n, n, |i, j| if i == j { 0.5 } else { 0.1 }),
        ];
        // A full-rank positive-definite covariance for the (unasserted here)
        // shock series; it need not match the rank-one residuals.
        let sigma = Mat::from_fn(n, n, |i, j| [[1.5, 0.2], [0.2, 1.0]][i][j]);

        let res = proxy_svar(u.as_ref(), &proxy, &psi, sigma.as_ref(), 0, 1.0, true)?;

        assert!((res.relative_impact[0] - 1.0).abs() < 1e-13);
        assert!((res.relative_impact[1] - 0.5).abs() < 1e-13);
        // Unit-effect impact: b[norm_var] == 1 exactly.
        assert_eq!(res.impact[0], 1.0);
        assert!((res.impact[1] - 0.5).abs() < 1e-13);
        // irf[0] == impact.
        assert_eq!(res.irf[0], res.impact);
        // irf[1] == Psi_1 b.
        let expected10 = 0.5 * res.impact[0] + 0.1 * res.impact[1];
        let expected11 = 0.1 * res.impact[0] + 0.5 * res.impact[1];
        assert!((res.irf[1][0] - expected10).abs() < 1e-13);
        assert!((res.irf[1][1] - expected11).abs() < 1e-13);
        assert_eq!(res.n_proxy, tt);
        // Reliability is a perfect fit here (u_0 is a deterministic multiple
        // of the proxy), so Corr(m, u_0)^2 == 1.
        assert!((res.reliability - 1.0).abs() < 1e-12);
        Ok(())
    }

    /// A NaN prefix in the proxy is dropped from the moments; `n_proxy`
    /// counts only the finite tail.
    #[test]
    fn nan_prefix_is_dropped() -> Result<(), IdentError> {
        let n = 2;
        let tt = 8;
        let mut u = Mat::<f64>::zeros(tt, n);
        for r in 0..tt {
            u[(r, 0)] = (r as f64) - 3.5;
            u[(r, 1)] = 0.5 * (r as f64) - 1.0;
        }
        let mut proxy = vec![0.0f64; tt];
        for (r, p) in proxy.iter_mut().enumerate() {
            *p = if r < 3 { f64::NAN } else { (r as f64) - 3.5 };
        }
        let psi = vec![Mat::<f64>::identity(n, n)];
        let sigma = Mat::<f64>::identity(n, n);
        let res = proxy_svar(u.as_ref(), &proxy, &psi, sigma.as_ref(), 0, 1.0, true)?;
        assert_eq!(res.n_proxy, tt - 3);
        Ok(())
    }

    /// SPD solve agrees with a hand-computed inverse on a 2x2 system.
    #[test]
    fn spd_solve_matches_by_hand() -> Result<(), IdentError> {
        // A = [[4,1],[1,3]], b = [1,2]. x = A^{-1} b.
        // det = 11; A^{-1} = 1/11 [[3,-1],[-1,4]]; x = 1/11 [3-2, -1+8] =
        // [1/11, 7/11].
        let a = Mat::from_fn(2, 2, |i, j| [[4.0, 1.0], [1.0, 3.0]][i][j]);
        let x = spd_solve(a.as_ref(), &[1.0, 2.0])?;
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-14);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-14);
        Ok(())
    }

    /// Dimension and domain guards fire.
    #[test]
    fn guards_fire() {
        let n = 2;
        let u = Mat::<f64>::from_fn(4, n, |i, _| i as f64);
        let psi = vec![Mat::<f64>::identity(n, n)];
        let sigma = Mat::<f64>::identity(n, n);
        // proxy length mismatch.
        assert!(matches!(
            proxy_svar(u.as_ref(), &[0.0; 3], &psi, sigma.as_ref(), 0, 1.0, true),
            Err(IdentError::Dimension { .. })
        ));
        // norm_var out of range.
        assert!(matches!(
            proxy_svar(u.as_ref(), &[0.0; 4], &psi, sigma.as_ref(), 5, 1.0, true),
            Err(IdentError::RestrictionOutOfRange { .. })
        ));
        // zero unit.
        assert!(matches!(
            proxy_svar(u.as_ref(), &[0.0; 4], &psi, sigma.as_ref(), 0, 0.0, true),
            Err(IdentError::InvalidArgument { .. })
        ));
        // all-NaN proxy -> empty overlap.
        assert!(matches!(
            proxy_svar(
                u.as_ref(),
                &[f64::NAN; 4],
                &psi,
                sigma.as_ref(),
                0,
                1.0,
                true
            ),
            Err(IdentError::InvalidArgument { .. })
        ));
    }
}
