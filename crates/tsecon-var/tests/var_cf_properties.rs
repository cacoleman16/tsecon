//! Seeded Monte-Carlo properties of the conditional forecast and the
//! residual diagnostics — the claims a golden match cannot make:
//!
//! * conditional-forecast coverage and calibration under the data-generating
//!   process (the conditioning cells are the realised future of one series,
//!   the free series must be covered at the nominal rate and its
//!   standardised errors must have unit variance);
//! * the minimum-norm characterisation of the implied shocks;
//! * Portmanteau size under white-noise residuals and power against an
//!   omitted lag; Jarque-Bera size under Gaussian innovations and power
//!   against Student-t(4) innovations;
//! * the residual-normality family added in 0.11.0 — the symmetric-root
//!   Jarque-Bera (JMulTi's "Doornik & Hansen (1994)" block) and the
//!   Doornik-Hansen (2008) omnibus: their invariances (and the one the
//!   symmetric root lacks), the null moments of the DH z-scores (the only
//!   guard on the kurtosis transform's common denominator, which cancels
//!   from the exact mean-matching identity), size on VAR residuals at
//!   T = 200 and T = 40, power against t(4) and skewed chi-squared(3)
//!   innovations, and the `None`-below-8-observations semantics of the
//!   bundle;
//! * refusals that name the offending parameter.

mod common;

use common::Lcg;
use tsecon_var::tsecon_linalg::faer::linalg::solvers::Solve;
use tsecon_var::tsecon_linalg::faer::{Mat, Side};
use tsecon_var::{
    doornik_hansen_test, normality_test_cholesky, normality_test_symmetric, Trend, VarSpec,
};

/// Innovation law of the simulator.
#[derive(Clone, Copy)]
enum Innov {
    /// Standard normal.
    Gauss,
    /// Student-t(4) scaled to unit variance.
    T4,
    /// Centred chi-squared(3) scaled to unit variance (skewed).
    Chi2,
}

/// VAR(p) simulator with intercept `c`, lag matrices `a`, innovation
/// `chol z` (or Student-t(4) scaled to unit variance when `t4`).
fn simulate(
    rng: &mut Lcg,
    n: usize,
    c: &[f64],
    a: &[Mat<f64>],
    chol: &Mat<f64>,
    t4: bool,
    start: Option<&Mat<f64>>,
) -> Mat<f64> {
    let innov = if t4 { Innov::T4 } else { Innov::Gauss };
    simulate_innov(rng, n, c, a, chol, innov, start)
}

/// [`simulate`] with an explicit innovation law (the Gaussian and t(4) draws
/// are exactly `simulate`'s).
fn simulate_innov(
    rng: &mut Lcg,
    n: usize,
    c: &[f64],
    a: &[Mat<f64>],
    chol: &Mat<f64>,
    innov: Innov,
    start: Option<&Mat<f64>>,
) -> Mat<f64> {
    let k = c.len();
    let p = a.len();
    let burn = if start.is_some() { 0 } else { 100 };
    let total = n + burn + p;
    let mut y = Mat::<f64>::zeros(total, k);
    if let Some(s) = start {
        for i in 0..p {
            for j in 0..k {
                y[(i, j)] = s[(i, j)];
            }
        }
    }
    let mut z = vec![0.0; k];
    for t in p..total {
        for zi in z.iter_mut() {
            let g = rng.gaussian();
            *zi = match innov {
                Innov::T4 => {
                    let chi: f64 = (0..4).map(|_| rng.gaussian().powi(2)).sum();
                    g / (chi / 4.0).sqrt() * (2.0f64 / 4.0).sqrt()
                }
                Innov::Gauss => g,
                Innov::Chi2 => {
                    let chi = g * g + rng.gaussian().powi(2) + rng.gaussian().powi(2);
                    (chi - 3.0) / 6.0f64.sqrt()
                }
            };
        }
        for r in 0..k {
            let mut v = c[r];
            for (i, ai) in a.iter().enumerate() {
                for l in 0..k {
                    v += ai[(r, l)] * y[(t - i - 1, l)];
                }
            }
            for l in 0..=r {
                v += chol[(r, l)] * z[l];
            }
            y[(t, r)] = v;
        }
    }
    y.submatrix(burn + p, 0, n, k).to_owned()
}

fn dgp_var1() -> (Vec<f64>, Vec<Mat<f64>>, Mat<f64>) {
    let c = vec![0.2, -0.1];
    let a = Mat::from_fn(2, 2, |i, j| [[0.5, 0.2], [0.1, 0.4]][i][j]);
    let chol = Mat::from_fn(2, 2, |i, j| [[1.0, 0.0], [0.5, 0.8]][i][j]);
    (c, vec![a], chol)
}

/// A k = 3 VAR(1) with unit-variance independent innovations.
fn dgp_var1_k3() -> (Vec<f64>, Vec<Mat<f64>>, Mat<f64>) {
    let c = vec![0.2, -0.1, 0.05];
    let a = Mat::from_fn(3, 3, |i, j| {
        [[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]][i][j]
    });
    let chol = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
    (c, vec![a], chol)
}

fn dgp_var2_k3() -> (Vec<f64>, Vec<Mat<f64>>, Mat<f64>) {
    let c = vec![0.5, -0.2, 0.1];
    let a1 = Mat::from_fn(3, 3, |i, j| {
        [[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]][i][j]
    });
    let a2 = Mat::from_fn(3, 3, |i, j| {
        [[0.1, 0.0, 0.05], [0.0, 0.1, 0.0], [0.05, 0.0, 0.1]][i][j]
    });
    let chol = Mat::from_fn(3, 3, |i, j| {
        [[0.8, 0.0, 0.0], [0.3, 0.6, 0.0], [-0.2, 0.25, 0.5]][i][j]
    });
    (c, vec![a1, a2], chol)
}

/// Coverage under the DGP: fit a VAR(1) on a long sample (T = 4000, so
/// coefficient error is negligible), then for 2000 replications draw the
/// true future path, condition on the realised path of series 0 over four
/// horizons, and check the 95% conditional interval for series 1. Measured
/// coverage must sit within Monte-Carlo error of 0.95 at every horizon, the
/// standardised errors must have unit variance, and the conditional band
/// must be strictly narrower than the unconditional one (Sigma_u has
/// correlation 0.53 here).
#[test]
fn conditional_forecast_coverage_under_the_dgp() {
    let (c, a, chol) = dgp_var1();
    let mut rng = Lcg::new(20260911);
    let y = simulate(&mut rng, 4000, &c, &a, &chol, false, None);
    let res = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(y.as_ref())
        .unwrap();
    let h = 4usize;
    let reps = 2000usize;
    let start = y.submatrix(y.nrows() - 1, 0, 1, 2).to_owned();
    let mut covered = vec![0usize; h];
    let mut covered_unc = vec![0usize; h];
    let mut sq_err = vec![0.0f64; h];
    let mut width_ratio = 0.0;
    let mut width_ratio_single = 0.0;
    let mut mean_maha = 0.0;
    for _ in 0..reps {
        let fut = simulate(&mut rng, h, &c, &a, &chol, false, Some(&start));
        let conds: Vec<Vec<Option<f64>>> = (0..h).map(|s| vec![Some(fut[(s, 0)]), None]).collect();
        let cf = res.conditional_forecast(h, &conds, 0.05).unwrap();
        let unc = res.forecast_interval(h, 0.05).unwrap();
        for s in 0..h {
            let v = fut[(s, 1)];
            if cf.lower[(s, 1)] <= v && v <= cf.upper[(s, 1)] {
                covered[s] += 1;
            }
            if unc.lower[(s, 1)] <= v && v <= unc.upper[(s, 1)] {
                covered_unc[s] += 1;
            }
            let e = (v - cf.point[(s, 1)]) / cf.se[(s, 1)];
            sq_err[s] += e * e;
            assert!(cf.se[(s, 1)] < cf.unconditional_se[(s, 1)]);
        }
        width_ratio += cf.se[(0, 1)] / cf.unconditional_se[(0, 1)];
        // Single-cell conditioning at h = 1 only, for the exact identity below.
        let cf1 = res
            .conditional_forecast(1, &[vec![Some(fut[(0, 0)]), None]], 0.05)
            .unwrap();
        width_ratio_single += cf1.se[(0, 1)] / cf1.unconditional_se[(0, 1)];
        mean_maha += cf.mahalanobis;
    }
    width_ratio /= reps as f64;
    width_ratio_single /= reps as f64;
    mean_maha /= reps as f64;
    let se_mc = (0.95f64 * 0.05 / reps as f64).sqrt(); // 0.0049
    println!(
        "conditional coverage: h=1 se ratio {width_ratio:.4} (four-cell path) / \
         {width_ratio_single:.4} (single cell), mean Mahalanobis {mean_maha:.3} \
         (reps {reps}, T 4000)"
    );
    for s in 0..h {
        let cov = covered[s] as f64 / reps as f64;
        let cov_u = covered_unc[s] as f64 / reps as f64;
        println!(
            "  h={}: conditional {cov:.4} unconditional {cov_u:.4} standardised-error variance {:.4}",
            s + 1,
            sq_err[s] / reps as f64
        );
        assert!(
            (cov - 0.95).abs() < 4.0 * se_mc + 0.005,
            "h={}: conditional coverage {cov} (unconditional {cov_u})",
            s + 1
        );
        assert!(
            (cov_u - 0.95).abs() < 4.0 * se_mc + 0.005,
            "h={}: unconditional {cov_u}",
            s + 1
        );
        let calib = sq_err[s] / reps as f64;
        assert!(
            (calib - 1.0).abs() < 0.12,
            "h={}: standardised error variance {calib}",
            s + 1
        );
    }
    // Conditioning on y_1 at h = 1 ALONE leaves y_2 with variance
    // sigma_22 (1 - rho^2): the se ratio is exactly sqrt(1 - rho_hat^2) at
    // the fitted covariance (near the DGP's sqrt(1 - 0.530^2) = 0.848).
    // Conditioning on the whole four-horizon path of y_1 tightens h = 1
    // further, because the later conditions inform the h = 1 innovation
    // through the dynamics (the smoother runs backwards).
    let s = &res.sigma_u;
    let rho_hat = s[(0, 1)] / (s[(0, 0)] * s[(1, 1)]).sqrt();
    let exact = (1.0 - rho_hat * rho_hat).sqrt();
    assert!(
        (width_ratio_single - exact).abs() < 1e-10,
        "single-cell h=1 se ratio {width_ratio_single} vs {exact}"
    );
    assert!(
        (exact - 0.848).abs() < 0.03,
        "fitted correlation far from the DGP: ratio {exact}"
    );
    assert!(
        width_ratio < width_ratio_single,
        "path conditioning {width_ratio} should tighten h=1 beyond {width_ratio_single}"
    );
    // The plausibility statistic of a path the model itself generated is a
    // chi2(4) draw: mean 4.
    assert!(
        (mean_maha - 4.0).abs() < 0.3,
        "mean Mahalanobis {mean_maha}"
    );
}

/// The implied shocks are the minimum-Sigma-norm solution of the
/// constraints: the orthogonalised shocks `eps*` lie in the row space of
/// the constraint operator, so every feasible alternative
/// `eps* + null-space component` has a strictly larger norm and the same
/// constrained cells; `P eps* = u*` and `eps*'eps* = mahalanobis` hold.
#[test]
fn implied_shocks_are_minimum_norm_and_feasible() {
    let (c, a, chol) = dgp_var2_k3();
    let mut rng = Lcg::new(7);
    let y = simulate(&mut rng, 300, &c, &a, &chol, false, None);
    let res = VarSpec::new(2, Trend::Constant)
        .unwrap()
        .fit(y.as_ref())
        .unwrap();
    let (k, h) = (3usize, 6usize);
    let conds: Vec<Vec<Option<f64>>> = vec![
        vec![None, Some(0.3), None],
        vec![None, Some(0.2), Some(-0.4)],
        vec![None, None, None],
        vec![Some(1.0), None, None],
    ];
    let cf = res.conditional_forecast(h, &conds, 0.05).unwrap();
    let m = cf.n_constrained;
    assert_eq!(m, 4);
    let n = h * k;

    // Constraint operator on the orthogonalised shocks: Rt = B (I ⊗ P).
    let psi = res.ma_rep(h - 1).unwrap();
    let p_chol = res.sigma_u.llt(Side::Lower).unwrap().L().to_owned();
    let mut cells = Vec::new();
    for (hh, row) in conds.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            if v.is_some() {
                cells.push((hh, j));
            }
        }
    }
    let mut rt = Mat::<f64>::zeros(m, n);
    for (i, &(hh, j)) in cells.iter().enumerate() {
        for s in 0..=hh {
            for l in 0..k {
                let mut acc = 0.0;
                for l2 in 0..k {
                    acc += psi[hh - s][(j, l2)] * p_chol[(l2, l)];
                }
                rt[(i, s * k + l)] = acc;
            }
        }
    }
    let eps = Mat::from_fn(n, 1, |i, _| cf.orth_shocks[(i / k, i % k)]);
    let u = Mat::from_fn(n, 1, |i, _| cf.shocks[(i / k, i % k)]);
    // P eps = u, block by block.
    for s in 0..h {
        for i in 0..k {
            let mut acc = 0.0;
            for l in 0..k {
                acc += p_chol[(i, l)] * eps[(s * k + l, 0)];
            }
            assert!((acc - u[(s * k + i, 0)]).abs() < 1e-12);
        }
    }
    let norm_star: f64 = (0..n).map(|i| eps[(i, 0)].powi(2)).sum();
    assert!((norm_star - cf.mahalanobis).abs() < 1e-10 * norm_star.max(1.0));

    // Random feasible alternatives eps* + (I - Rt'(Rt Rt')^{-1} Rt) eta.
    let gram = &rt * rt.transpose();
    let llt = gram.llt(Side::Lower).unwrap();
    for _ in 0..20 {
        let eta = Mat::from_fn(n, 1, |_, _| rng.gaussian());
        let proj = rt.transpose() * llt.solve(&(&rt * &eta));
        let null = &eta - &proj;
        let alt = &eps + &null;
        // Same constrained cells: Rt alt = Rt eps*.
        let d = &rt * &alt - &rt * &eps;
        for i in 0..m {
            assert!(d[(i, 0)].abs() < 1e-10);
        }
        // Orthogonality (the minimum-norm characterisation) and larger norm.
        let dot: f64 = (0..n).map(|i| eps[(i, 0)] * null[(i, 0)]).sum();
        assert!(
            dot.abs() < 1e-9,
            "eps* not orthogonal to the null space: {dot}"
        );
        let norm_alt: f64 = (0..n).map(|i| alt[(i, 0)].powi(2)).sum();
        assert!(norm_alt > norm_star);
        // And the path built from the alternative shocks hits the same
        // constrained cells.
        let mut path = cf.unconditional.clone();
        let alt_u = Mat::from_fn(h, k, |s, i| {
            (0..k)
                .map(|l| p_chol[(i, l)] * alt[(s * k + l, 0)])
                .sum::<f64>()
        });
        for hh in 0..h {
            for s in 0..=hh {
                for r in 0..k {
                    for l in 0..k {
                        path[(hh, r)] += psi[hh - s][(r, l)] * alt_u[(s, l)];
                    }
                }
            }
        }
        for &(hh, j) in &cells {
            assert!((path[(hh, j)] - cf.point[(hh, j)]).abs() < 1e-9);
        }
    }
}

/// Rejection frequencies at 5% over `reps` simulated fits.
struct Rates {
    /// Unadjusted Portmanteau.
    q: f64,
    /// Adjusted Portmanteau.
    adj: f64,
    /// Cholesky Jarque-Bera (statsmodels).
    jb: f64,
    /// Mean adjusted Portmanteau p-value.
    mean_p: f64,
    /// Symmetric-root Jarque-Bera (JMulTi "Doornik & Hansen (1994)").
    sym: f64,
    /// Doornik-Hansen (2008) omnibus.
    dh: f64,
    /// Doornik-Hansen (2008) skewness component `z1'z1 ~ chi2(k)`.
    dh_skew: f64,
}

#[allow(clippy::too_many_arguments)]
fn reject_rates(
    seed: u64,
    reps: usize,
    fit_p: usize,
    dgp: (Vec<f64>, Vec<Mat<f64>>, Mat<f64>),
    innov: Innov,
    n: usize,
    nlags: usize,
) -> Rates {
    let (c, a, chol) = dgp;
    let mut rng = Lcg::new(seed);
    let (mut rej_q, mut rej_adj, mut rej_jb, mut mean_p) = (0usize, 0usize, 0usize, 0.0);
    let (mut rej_sym, mut rej_dh, mut rej_dh_skew) = (0usize, 0usize, 0usize);
    for _ in 0..reps {
        let y = simulate_innov(&mut rng, n, &c, &a, &chol, innov, None);
        let res = VarSpec::new(fit_p, Trend::Constant)
            .unwrap()
            .fit(y.as_ref())
            .unwrap();
        let d = res.diagnostics(nlags).unwrap();
        if d.portmanteau.pvalue < 0.05 {
            rej_q += 1;
        }
        if d.portmanteau.adjusted_pvalue < 0.05 {
            rej_adj += 1;
        }
        if d.normality.pvalue < 0.05 {
            rej_jb += 1;
        }
        if d.normality_symmetric.unwrap().pvalue < 0.05 {
            rej_sym += 1;
        }
        let dh = d.doornik_hansen.unwrap();
        if dh.pvalue < 0.05 {
            rej_dh += 1;
        }
        if dh.skewness_pvalue < 0.05 {
            rej_dh_skew += 1;
        }
        mean_p += d.portmanteau.adjusted_pvalue;
    }
    let r = reps as f64;
    Rates {
        q: rej_q as f64 / r,
        adj: rej_adj as f64 / r,
        jb: rej_jb as f64 / r,
        mean_p: mean_p / r,
        sym: rej_sym as f64 / r,
        dh: rej_dh as f64 / r,
        dh_skew: rej_dh_skew as f64 / r,
    }
}

/// Size: on a correctly specified Gaussian VAR(1) (T = 200, k = 2, 1000
/// replications) both Portmanteau statistics and the Jarque-Bera test
/// reject near 5% — the unadjusted Q is known to be undersized at this T
/// (Lütkepohl 2005, section 4.4.3), the adjusted one closer to nominal.
/// The adjusted p-value averages 0.5 (uniform under the null). The
/// symmetric-root Jarque-Bera and the Doornik-Hansen (2008) test are
/// measured on the same fits (the draws are unchanged).
#[test]
fn portmanteau_and_jarque_bera_size_under_white_noise_residuals() {
    let r = reject_rates(101, 1000, 1, dgp_var1(), Innov::Gauss, 200, 8);
    let (q, adj, jb, mean_p) = (r.q, r.adj, r.jb, r.mean_p);
    println!(
        "size at 5% (T = 200, k = 2, nlags = 8, 1000 reps): Q {q:.4}, adjusted Q {adj:.4}, \
         Jarque-Bera {jb:.4}; mean adjusted p {mean_p:.4}; symmetric-root Jarque-Bera {:.4}, \
         Doornik-Hansen (2008) {:.4}",
        r.sym, r.dh
    );
    assert!((0.02..=0.08).contains(&q), "unadjusted Q size {q}");
    assert!((0.03..=0.08).contains(&adj), "adjusted Q size {adj}");
    assert!(
        adj >= q,
        "adjustment lowers the rejection rate: {adj} < {q}"
    );
    assert!((0.02..=0.08).contains(&jb), "Jarque-Bera size {jb}");
    assert!((0.45..=0.55).contains(&mean_p), "mean adjusted p {mean_p}");
    assert!(
        (0.02..=0.08).contains(&r.sym),
        "symmetric JB size {}",
        r.sym
    );
    assert!(
        (0.03..=0.075).contains(&r.dh),
        "Doornik-Hansen size {}",
        r.dh
    );
}

/// Size in a SMALL sample, where the asymptotic Jarque-Bera forms and the
/// finite-sample-transformed Doornik-Hansen (2008) test part ways: a
/// Gaussian k = 3 VAR(1) with T = 40 (2000 replications).
#[test]
fn normality_size_on_short_var_residuals() {
    let r = reject_rates(404, 2000, 1, dgp_var1_k3(), Innov::Gauss, 40, 6);
    println!(
        "normality size at 5% on VAR(1) residuals (T = 40, k = 3, 2000 reps): Cholesky \
         Jarque-Bera {:.4}, symmetric-root Jarque-Bera {:.4}, Doornik-Hansen (2008) {:.4}",
        r.jb, r.sym, r.dh
    );
    assert!(
        (0.03..=0.08).contains(&r.dh),
        "Doornik-Hansen size at T = 40: {}",
        r.dh
    );
    assert!(
        r.jb <= 0.08 && r.sym <= 0.08,
        "JB sizes {} / {}",
        r.jb,
        r.sym
    );
}

/// Power: fitting a VAR(1) to a VAR(2) DGP leaves lag-2 structure in the
/// residuals that the Portmanteau test detects; Student-t(4) innovations
/// are detected by Jarque-Bera and by Doornik-Hansen (2008); centred
/// chi-squared(3) innovations by the Doornik-Hansen skewness component.
/// (300 replications each.)
#[test]
fn portmanteau_and_jarque_bera_power() {
    let (c, a, chol) = dgp_var2_k3();
    // Strengthen the omitted lag so the misspecification is not subtle.
    let a2 = Mat::from_fn(3, 3, |i, j| a[1][(i, j)] + if i == j { 0.25 } else { 0.0 });
    let r = reject_rates(
        202,
        300,
        1,
        (c.clone(), vec![a[0].clone(), a2], chol.clone()),
        Innov::Gauss,
        300,
        6,
    );
    let (q, adj) = (r.q, r.adj);
    println!("power (T = 300, 300 reps): omitted lag Q {q:.4}, adjusted Q {adj:.4}");
    assert!(q > 0.9 && adj > 0.9, "Portmanteau power {q} / {adj}");
    let r = reject_rates(
        303,
        300,
        2,
        (c.clone(), a.clone(), chol.clone()),
        Innov::T4,
        300,
        6,
    );
    let jb = r.jb;
    println!(
        "power (T = 300, 300 reps): Jarque-Bera against t(4) {jb:.4}; symmetric-root \
         Jarque-Bera {:.4}; Doornik-Hansen (2008) {:.4}",
        r.sym, r.dh
    );
    assert!(jb > 0.9, "Jarque-Bera power against t(4) {jb}");
    assert!(r.dh > 0.9, "Doornik-Hansen power against t(4) {}", r.dh);
    let r = reject_rates(505, 300, 2, (c, a, chol), Innov::Chi2, 300, 6);
    println!(
        "power (T = 300, 300 reps) against centred chi2(3) innovations: Doornik-Hansen \
         skewness component {:.4}, omnibus {:.4}; Cholesky Jarque-Bera {:.4}",
        r.dh_skew, r.dh, r.jb
    );
    assert!(
        r.dh_skew > 0.9,
        "DH skewness power against chi2(3) {}",
        r.dh_skew
    );
}

/// The null moments of the Doornik-Hansen z-scores, univariate n = 50,
/// 20 000 replications: both near N(0, 1). `var z2` sits below 1 at this n
/// (0.9 measured, also by an independent NumPy transcription on other
/// seeds) — the gamma approximation's known small-sample undershoot — and
/// the band is set around it; this is the guard on the kurtosis transform's
/// common denominator `d`, which cancels from the exact identity.
#[test]
fn doornik_hansen_null_z_moments() {
    let mut rng = Lcg::new(6060);
    let (n, reps) = (50usize, 20_000usize);
    let (mut s1, mut s2, mut q1, mut q2) = (0.0, 0.0, 0.0, 0.0);
    for _ in 0..reps {
        let x = Mat::from_fn(n, 1, |_, _| rng.gaussian());
        let d = doornik_hansen_test(x.as_ref()).unwrap();
        let (z1, z2) = (d.z_skewness[0], d.z_kurtosis[0]);
        s1 += z1;
        s2 += z2;
        q1 += z1 * z1;
        q2 += z2 * z2;
    }
    let r = reps as f64;
    let (m1, m2) = (s1 / r, s2 / r);
    let (v1, v2) = (q1 / r - m1 * m1, q2 / r - m2 * m2);
    println!(
        "Doornik-Hansen null z-moments (n = 50, {reps} reps): mean z1 {m1:.4}, var z1 {v1:.4}, \
         mean z2 {m2:.4}, var z2 {v2:.4}"
    );
    assert!(m1.abs() < 0.03 && m2.abs() < 0.03, "means {m1} / {m2}");
    assert!((0.96..=1.04).contains(&v1), "var z1 {v1}");
    assert!((0.86..=1.02).contains(&v2), "var z2 {v2}");
}

/// Invariances on a seeded skewed panel: Doornik-Hansen (2008) is invariant
/// to column permutation, positive rescaling, sign flips and shifts; the
/// symmetric root to permutation, sign flips and shifts but NOT to
/// rescaling (it depends on units); the Cholesky form to sign flips, shifts
/// and rescaling but not to the column order. A sign flip of column j flips
/// `z1_j` and `b1_j` exactly.
#[test]
fn normality_invariances() {
    let mut rng = Lcg::new(77);
    let (t, k) = (150usize, 3usize);
    let mut x = Mat::from_fn(t, k, |_, _| {
        let g = rng.gaussian();
        g + 0.3 * (g * g - 1.0)
    });
    // Correlate the columns.
    for i in 0..t {
        x[(i, 1)] += 0.6 * x[(i, 0)];
        x[(i, 2)] += 0.4 * x[(i, 1)] - 0.2 * x[(i, 0)];
    }
    let close = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol * a.abs().max(1.0);
    let dh0 = doornik_hansen_test(x.as_ref()).unwrap();
    let sy0 = normality_test_symmetric(x.as_ref()).unwrap();
    let ch0 = normality_test_cholesky(x.as_ref()).unwrap();
    // Permutation.
    let perm = [2usize, 0, 1];
    let xp = Mat::from_fn(t, k, |i, j| x[(i, perm[j])]);
    let dhp = doornik_hansen_test(xp.as_ref()).unwrap();
    let syp = normality_test_symmetric(xp.as_ref()).unwrap();
    assert!(close(dhp.statistic, dh0.statistic, 1e-10));
    assert!(close(syp.statistic, sy0.statistic, 1e-10));
    for (j, &pj) in perm.iter().enumerate() {
        assert!(close(dhp.z_skewness[j], dh0.z_skewness[pj], 1e-10));
        assert!(close(dhp.z_kurtosis[j], dh0.z_kurtosis[pj], 1e-10));
    }
    let chp = normality_test_cholesky(xp.as_ref()).unwrap();
    assert!(
        !close(chp.statistic, ch0.statistic, 1e-6),
        "Cholesky is order-dependent"
    );
    // Positive rescaling by (100, 0.01, 3).
    let sc = [100.0, 0.01, 3.0];
    let xs = Mat::from_fn(t, k, |i, j| x[(i, j)] * sc[j]);
    let dhs = doornik_hansen_test(xs.as_ref()).unwrap();
    let sys = normality_test_symmetric(xs.as_ref()).unwrap();
    let chs = normality_test_cholesky(xs.as_ref()).unwrap();
    assert!(close(dhs.statistic, dh0.statistic, 1e-10));
    assert!(close(chs.statistic, ch0.statistic, 1e-10));
    println!(
        "rescaling by (100, 0.01, 3): DH {:.6} -> {:.6}, Cholesky JB {:.6} -> {:.6}, symmetric-root JB {:.6} -> {:.6}",
        dh0.statistic, dhs.statistic, ch0.statistic, chs.statistic, sy0.statistic, sys.statistic
    );
    assert!(
        (sys.statistic - sy0.statistic).abs() > 1.0,
        "the symmetric root should depend on units: {} vs {}",
        sys.statistic,
        sy0.statistic
    );
    // Sign flip of column 1.
    let xf = Mat::from_fn(t, k, |i, j| if j == 1 { -x[(i, j)] } else { x[(i, j)] });
    let dhf = doornik_hansen_test(xf.as_ref()).unwrap();
    let syf = normality_test_symmetric(xf.as_ref()).unwrap();
    let chf = normality_test_cholesky(xf.as_ref()).unwrap();
    assert!(close(dhf.statistic, dh0.statistic, 1e-12));
    assert!(close(syf.statistic, sy0.statistic, 1e-12));
    assert!(close(chf.statistic, ch0.statistic, 1e-12));
    for j in 0..k {
        let sgn = if j == 1 { -1.0 } else { 1.0 };
        assert!(close(dhf.z_skewness[j], sgn * dh0.z_skewness[j], 1e-12));
        assert!(close(dhf.sqrt_b1[j], sgn * dh0.sqrt_b1[j], 1e-12));
        assert!(close(dhf.z_kurtosis[j], dh0.z_kurtosis[j], 1e-12));
        assert!(close(
            syf.skewness_components[j],
            sgn * sy0.skewness_components[j],
            1e-12
        ));
        assert!(close(
            chf.skewness_components[j],
            sgn * ch0.skewness_components[j],
            1e-12
        ));
    }
    // Shift.
    let xh = Mat::from_fn(t, k, |i, j| x[(i, j)] + [5.0, -3.0, 0.25][j]);
    assert!(close(
        doornik_hansen_test(xh.as_ref()).unwrap().statistic,
        dh0.statistic,
        1e-10
    ));
    assert!(close(
        normality_test_symmetric(xh.as_ref()).unwrap().statistic,
        sy0.statistic,
        1e-10
    ));
    assert!(close(
        normality_test_cholesky(xh.as_ref()).unwrap().statistic,
        ch0.statistic,
        1e-10
    ));
}

/// The bundle never refuses on account of the added normality tests: a
/// VAR(1) with k = 2 and T = 7 effective observations (which works since
/// 0.10.0) returns `doornik_hansen == None`; at T = 8 it is `Some`. Series on
/// wildly different scales leave the unit-dependent symmetric root `None`
/// while everything else is computed.
#[test]
fn bundle_keeps_working_below_eight_observations_and_on_bad_scales() {
    let (c, a, chol) = dgp_var1();
    let mut rng = Lcg::new(31);
    let y = simulate(&mut rng, 9, &c, &a, &chol, false, None);
    let fit = |rows: usize| {
        VarSpec::new(1, Trend::Constant)
            .unwrap()
            .fit(y.submatrix(0, 0, rows, 2))
            .unwrap()
    };
    let r7 = fit(8);
    assert_eq!(r7.nobs, 7);
    let d7 = r7.diagnostics(2).unwrap();
    assert!(d7.doornik_hansen.is_none());
    assert!(d7.normality_symmetric.is_some());
    assert_eq!(d7.normality, r7.normality_test().unwrap());
    let e = r7.doornik_hansen_test().unwrap_err().to_string();
    assert!(e.contains("resid") && e.contains("8 rows"), "{e}");
    let r8 = fit(9);
    assert_eq!(r8.nobs, 8);
    let d8 = r8.diagnostics(2).unwrap();
    assert_eq!(d8.doornik_hansen, Some(r8.doornik_hansen_test().unwrap()));
    // Units: one series in 1e9, the other in 1e-9.
    let mut rng = Lcg::new(32);
    let y = simulate(&mut rng, 120, &c, &a, &chol, false, None);
    let ys = Mat::from_fn(120, 2, |i, j| y[(i, j)] * if j == 0 { 1e9 } else { 1e-9 });
    let rs = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(ys.as_ref())
        .unwrap();
    let ds = rs.diagnostics(5).unwrap();
    assert!(ds.normality_symmetric.is_none());
    let dh = ds.doornik_hansen.unwrap();
    let r0 = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(y.as_ref())
        .unwrap();
    let d0 = r0.diagnostics(5).unwrap();
    assert!((dh.statistic - d0.doornik_hansen.unwrap().statistic).abs() < 1e-8);
    assert!((ds.normality.statistic - d0.normality.statistic).abs() < 1e-8);
}

/// The stability verdict of the bundle agrees with `VarResults::is_stable`
/// and with both root readings, on a stable and on an explosive system.
#[test]
fn stability_flag_is_consistent() {
    let mut rng = Lcg::new(5);
    let (c, a, chol) = dgp_var1();
    let y = simulate(&mut rng, 300, &c, &a, &chol, false, None);
    let res = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(y.as_ref())
        .unwrap();
    let d = res.diagnostics(5).unwrap();
    assert!(d.is_stable && res.is_stable().unwrap());
    assert!(d.roots.iter().all(|r| *r > 1.0) && d.eigenvalue_moduli[0] < 1.0);
    assert!((d.roots.last().unwrap() * d.eigenvalue_moduli[0] - 1.0).abs() < 1e-12);
    // An explosive series: a random walk with drift twice, fitted as VAR(1).
    let mut z = Mat::<f64>::zeros(300, 2);
    for t in 1..300 {
        z[(t, 0)] = 1.05 * z[(t - 1, 0)] + 0.1 + rng.gaussian();
        z[(t, 1)] = 0.5 * z[(t - 1, 1)] + rng.gaussian();
    }
    let res_x = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(z.as_ref())
        .unwrap();
    let dx = res_x.diagnostics(5).unwrap();
    assert!(!dx.is_stable);
    assert!(dx.eigenvalue_moduli[0] > 1.0 && *dx.roots.last().unwrap() < 1.0);
}

/// Every refusal names the offending parameter and says what to pass.
#[test]
fn refusals_name_the_parameter() {
    let (c, a, chol) = dgp_var1();
    let mut rng = Lcg::new(9);
    let y = simulate(&mut rng, 200, &c, &a, &chol, false, None);
    let res = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(y.as_ref())
        .unwrap();
    let ok = vec![vec![Some(0.1), None]];
    assert!(res.conditional_forecast(3, &ok, 0.05).is_ok());

    let e = res
        .conditional_forecast(0, &ok, 0.05)
        .unwrap_err()
        .to_string();
    assert!(e.contains("steps = 0"), "{e}");
    let e = res
        .conditional_forecast(3, &ok, 1.5)
        .unwrap_err()
        .to_string();
    assert!(e.contains("alpha"), "{e}");
    let e = res
        .conditional_forecast(3, &[], 0.05)
        .unwrap_err()
        .to_string();
    assert!(e.contains("conditions is empty"), "{e}");
    let e = res
        .conditional_forecast(1, &[ok[0].clone(), ok[0].clone()], 0.05)
        .unwrap_err()
        .to_string();
    assert!(e.contains("conditions has more rows than steps"), "{e}");
    let e = res
        .conditional_forecast(3, &[vec![Some(0.1)]], 0.05)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("every row of conditions") && e.contains("expected 2, got 1"),
        "{e}"
    );
    let e = res
        .conditional_forecast(3, &[vec![None, Some(f64::NAN)]], 0.05)
        .unwrap_err()
        .to_string();
    assert!(e.contains("conditions constrains no cell"), "{e}");
    let e = res
        .conditional_forecast(3, &[vec![Some(f64::INFINITY), None]], 0.05)
        .unwrap_err()
        .to_string();
    assert!(e.contains("conditions contains an infinite value"), "{e}");
    // The memory budget refuses a steps typo instead of aborting. The check
    // runs BEFORE the steps x k grid of constrained flags is allocated, so a
    // horizon that would need terabytes is a teaching error, not an
    // allocator abort (the regression: `vec![vec![false; k]; steps]` used to
    // run first and `steps = 1 << 40` aborted the process).
    for bad in [20_000_000usize, 1usize << 40, usize::MAX] {
        let e = res
            .conditional_forecast(bad, &ok, 0.05)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("memory budget") && e.contains("steps"),
            "steps = {bad}: {e}"
        );
    }
    let e = res.portmanteau_test(1).unwrap_err().to_string();
    assert!(e.contains("nlags"), "{e}");
}
