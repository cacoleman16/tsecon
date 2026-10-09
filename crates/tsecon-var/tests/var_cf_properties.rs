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
//! * refusals that name the offending parameter;
//! * soft (Gaussian, stated-variance) conditions: signal-mode coverage of
//!   the true path under noisy readings, density-mode coverage under exact
//!   conditional draws from the true DGP (Matheron's rule), monotonicity in
//!   the stated sd, the sd -> 0 / sd -> infinity limits, the density
//!   scaling identity, and the `condition_sd` refusals.

mod common;

use common::Lcg;
use tsecon_var::tsecon_linalg::faer::linalg::solvers::Solve;
use tsecon_var::tsecon_linalg::faer::{Mat, Side};
use tsecon_var::{ConditionMode, SoftMode, Trend, VarSpec};

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
            *zi = if t4 {
                let chi: f64 = (0..4).map(|_| rng.gaussian().powi(2)).sum();
                g / (chi / 4.0).sqrt() * (2.0f64 / 4.0).sqrt()
            } else {
                g
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

fn reject_rates(
    seed: u64,
    reps: usize,
    fit_p: usize,
    dgp: (Vec<f64>, Vec<Mat<f64>>, Mat<f64>),
    t4: bool,
    n: usize,
    nlags: usize,
) -> (f64, f64, f64, f64) {
    let (c, a, chol) = dgp;
    let mut rng = Lcg::new(seed);
    let (mut rej_q, mut rej_adj, mut rej_jb, mut mean_p) = (0usize, 0usize, 0usize, 0.0);
    for _ in 0..reps {
        let y = simulate(&mut rng, n, &c, &a, &chol, t4, None);
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
        mean_p += d.portmanteau.adjusted_pvalue;
    }
    let r = reps as f64;
    (
        rej_q as f64 / r,
        rej_adj as f64 / r,
        rej_jb as f64 / r,
        mean_p / r,
    )
}

/// Size: on a correctly specified Gaussian VAR(1) (T = 200, k = 2, 1000
/// replications) both Portmanteau statistics and the Jarque-Bera test
/// reject near 5% — the unadjusted Q is known to be undersized at this T
/// (Lütkepohl 2005, section 4.4.3), the adjusted one closer to nominal.
/// The adjusted p-value averages 0.5 (uniform under the null).
#[test]
fn portmanteau_and_jarque_bera_size_under_white_noise_residuals() {
    let (q, adj, jb, mean_p) = reject_rates(101, 1000, 1, dgp_var1(), false, 200, 8);
    println!(
        "size at 5% (T = 200, k = 2, nlags = 8, 1000 reps): Q {q:.4}, adjusted Q {adj:.4}, \
         Jarque-Bera {jb:.4}; mean adjusted p {mean_p:.4}"
    );
    assert!((0.02..=0.08).contains(&q), "unadjusted Q size {q}");
    assert!((0.03..=0.08).contains(&adj), "adjusted Q size {adj}");
    assert!(
        adj >= q,
        "adjustment lowers the rejection rate: {adj} < {q}"
    );
    assert!((0.02..=0.08).contains(&jb), "Jarque-Bera size {jb}");
    assert!((0.45..=0.55).contains(&mean_p), "mean adjusted p {mean_p}");
}

/// Power: fitting a VAR(1) to a VAR(2) DGP leaves lag-2 structure in the
/// residuals that the Portmanteau test detects; Student-t(4) innovations
/// are detected by Jarque-Bera. (300 replications each.)
#[test]
fn portmanteau_and_jarque_bera_power() {
    let (c, a, chol) = dgp_var2_k3();
    // Strengthen the omitted lag so the misspecification is not subtle.
    let a2 = Mat::from_fn(3, 3, |i, j| a[1][(i, j)] + if i == j { 0.25 } else { 0.0 });
    let (q, adj, _, _) = reject_rates(
        202,
        300,
        1,
        (c.clone(), vec![a[0].clone(), a2], chol.clone()),
        false,
        300,
        6,
    );
    println!("power (T = 300, 300 reps): omitted lag Q {q:.4}, adjusted Q {adj:.4}");
    assert!(q > 0.9 && adj > 0.9, "Portmanteau power {q} / {adj}");
    let (_, _, jb, _) = reject_rates(303, 300, 2, (c, a, chol), true, 300, 6);
    println!("power (T = 300, 300 reps): Jarque-Bera against t(4) {jb:.4}");
    assert!(jb > 0.9, "Jarque-Bera power against t(4) {jb}");
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

// ------------------------------------------------------- soft conditions

/// The true `G = R Σ B'` (`n x m`) and `Occ = B Σ B'` (`m x m`) of the
/// VAR(1) DGP for conditioned cells `(h, j)`: `Ψ_h = A^h`, `Σ = chol chol'`.
/// Used to draw from the exact density-conditional distribution without
/// touching the fitted model (which would make the coverage check circular).
fn true_gain_pieces(
    a: &Mat<f64>,
    chol: &Mat<f64>,
    steps: usize,
    cells: &[(usize, usize)],
) -> (Mat<f64>, Mat<f64>) {
    let k = a.nrows();
    let n = steps * k;
    let m = cells.len();
    let sigma = chol * chol.transpose();
    let mut psi = vec![Mat::<f64>::identity(k, k)];
    for h in 1..steps {
        psi.push(a * &psi[h - 1]);
    }
    // R (n x n): block (h, s) = Psi_{h-s}.
    let mut r = Mat::<f64>::zeros(n, n);
    for h in 0..steps {
        for s in 0..=h {
            for i in 0..k {
                for l in 0..k {
                    r[(h * k + i, s * k + l)] = psi[h - s][(i, l)];
                }
            }
        }
    }
    let big_s = Mat::from_fn(n, n, |i, j| {
        if i / k == j / k {
            sigma[(i % k, j % k)]
        } else {
            0.0
        }
    });
    let b = Mat::from_fn(m, n, |i, c| r[(cells[i].0 * k + cells[i].1, c)]);
    let w = &big_s * b.transpose();
    (&r * &w, &b * &w)
}

/// SIGNAL coverage under the DGP: the readings are the realised path of
/// series 0 over four horizons PLUS independent N(0, sd^2) noise (sd 0.3,
/// 0.4, 0.5, 0.6). The signal-mode 95% interval must cover the TRUE value
/// at every cell — the four read cells and the four free ones — at the
/// nominal rate, the standardised errors must have unit variance, and the
/// Mahalanobis statistic of the readings is a chi2(4) draw (mean 4).
#[test]
fn soft_signal_coverage_under_the_dgp() {
    let (c, a, chol) = dgp_var1();
    let mut rng = Lcg::new(20261009);
    let y = simulate(&mut rng, 4000, &c, &a, &chol, false, None);
    let res = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(y.as_ref())
        .unwrap();
    let (h, reps) = (4usize, 2000usize);
    let sds = [0.3, 0.4, 0.5, 0.6];
    let start = y.submatrix(y.nrows() - 1, 0, 1, 2).to_owned();
    let mut covered = vec![[0usize; 2]; h];
    let mut sq_err = vec![[0.0f64; 2]; h];
    let mut mean_maha = 0.0;
    let sd_grid: Vec<Vec<Option<f64>>> = sds.iter().map(|&s| vec![Some(s), None]).collect();
    for _ in 0..reps {
        let fut = simulate(&mut rng, h, &c, &a, &chol, false, Some(&start));
        let conds: Vec<Vec<Option<f64>>> = (0..h)
            .map(|s| vec![Some(fut[(s, 0)] + sds[s] * rng.gaussian()), None])
            .collect();
        let cf = res
            .conditional_forecast_soft(h, &conds, &sd_grid, SoftMode::Signal, 0.05)
            .unwrap();
        assert_eq!(cf.mode, ConditionMode::Signal);
        for s in 0..h {
            for j in 0..2 {
                let v = fut[(s, j)];
                if cf.lower[(s, j)] <= v && v <= cf.upper[(s, j)] {
                    covered[s][j] += 1;
                }
                let e = (v - cf.point[(s, j)]) / cf.se[(s, j)];
                sq_err[s][j] += e * e;
            }
        }
        mean_maha += cf.mahalanobis;
    }
    mean_maha /= reps as f64;
    let se_mc = (0.95f64 * 0.05 / reps as f64).sqrt();
    println!("soft signal coverage (reps {reps}, T 4000, sd {sds:?} on series 0), mean Mahalanobis {mean_maha:.3}");
    for s in 0..h {
        let cov = [
            covered[s][0] as f64 / reps as f64,
            covered[s][1] as f64 / reps as f64,
        ];
        let calib = [sq_err[s][0] / reps as f64, sq_err[s][1] / reps as f64];
        println!(
            "  h={}: read cell {:.4} free cell {:.4}; standardised-error variance {:.4} / {:.4}",
            s + 1,
            cov[0],
            cov[1],
            calib[0],
            calib[1]
        );
        for j in 0..2 {
            assert!(
                (cov[j] - 0.95).abs() < 4.0 * se_mc + 0.005,
                "h={} series {j}: signal coverage {}",
                s + 1,
                cov[j]
            );
            assert!(
                (calib[j] - 1.0).abs() < 0.12,
                "h={} series {j}: {}",
                s + 1,
                calib[j]
            );
        }
    }
    assert!(
        (mean_maha - 4.0).abs() < 0.3,
        "mean Mahalanobis {mean_maha}"
    );
}

/// DENSITY coverage by exact conditional simulation (Matheron's rule) with
/// the TRUE DGP's MA matrices: draw the scenario's cells `c ~ N(v, sd^2)`,
/// draw an unconditional future `ỹ` from the DGP, and set
/// `y = ỹ + G Occ^{-1} (c - B ỹ)` — a draw from the
/// Andersson-Palmqvist-Waggoner distribution under the true model. The
/// fitted model's density-mode 95% intervals must cover it at the nominal
/// rate at every cell (the soft cells: `c` itself; the free ones: the
/// conditional response).
#[test]
fn soft_density_coverage_by_exact_conditional_simulation() {
    let (c, a, chol) = dgp_var1();
    let mut rng = Lcg::new(20261010);
    let y = simulate(&mut rng, 4000, &c, &a, &chol, false, None);
    let res = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(y.as_ref())
        .unwrap();
    let (h, reps, k) = (4usize, 2000usize, 2usize);
    let sds = [0.3, 0.4, 0.5, 0.6];
    let offsets = [0.8, 0.5, 0.3, 0.1];
    let unc = res.forecast(h).unwrap();
    let v: Vec<f64> = (0..h).map(|s| unc[(s, 0)] + offsets[s]).collect();
    let conds: Vec<Vec<Option<f64>>> = v.iter().map(|&x| vec![Some(x), None]).collect();
    let sd_grid: Vec<Vec<Option<f64>>> = sds.iter().map(|&s| vec![Some(s), None]).collect();
    let cf = res
        .conditional_forecast_soft(h, &conds, &sd_grid, SoftMode::Density, 0.05)
        .unwrap();
    assert_eq!(cf.mode, ConditionMode::Density);
    let cells: Vec<(usize, usize)> = (0..h).map(|s| (s, 0)).collect();
    let (g_true, occ_true) = true_gain_pieces(&a[0], &chol, h, &cells);
    // K = G Occ^{-1} (n x m), through the transpose solve Occ K' = G'.
    let k_gain = occ_true
        .llt(Side::Lower)
        .unwrap()
        .solve(g_true.transpose())
        .transpose()
        .to_owned();
    let start = y.submatrix(y.nrows() - 1, 0, 1, 2).to_owned();
    let mut covered = vec![[0usize; 2]; h];
    let mut sq_err = vec![[0.0f64; 2]; h];
    for _ in 0..reps {
        let draw_c: Vec<f64> = (0..h).map(|s| v[s] + sds[s] * rng.gaussian()).collect();
        let fut = simulate(&mut rng, h, &c, &a, &chol, false, Some(&start));
        let gap: Vec<f64> = (0..h)
            .map(|i| draw_c[i] - fut[(cells[i].0, cells[i].1)])
            .collect();
        for s in 0..h {
            for j in 0..k {
                let mut yv = fut[(s, j)];
                for (i, g) in gap.iter().enumerate() {
                    yv += k_gain[(s * k + j, i)] * g;
                }
                if cf.lower[(s, j)] <= yv && yv <= cf.upper[(s, j)] {
                    covered[s][j] += 1;
                }
                let e = (yv - cf.point[(s, j)]) / cf.se[(s, j)];
                sq_err[s][j] += e * e;
            }
        }
    }
    // The DGP's own density-conditional se at each cell, from the same
    // pieces: MSE - G Occ^{-1} G' + K Om K' on the diagonal.
    let sigma_true = &chol * chol.transpose();
    let mut mse_true = Mat::<f64>::zeros(k, k);
    let mut psi_h = Mat::<f64>::identity(k, k);
    let mut true_se = vec![[0.0f64; 2]; h];
    for s in 0..h {
        mse_true = &mse_true + &psi_h * &sigma_true * psi_h.transpose();
        psi_h = &a[0] * &psi_h;
        for (j, slot) in true_se[s].iter_mut().enumerate() {
            let row = s * k + j;
            let shrink: f64 = (0..h).map(|i| g_true[(row, i)] * k_gain[(row, i)]).sum();
            let spread: f64 = (0..h).map(|i| (k_gain[(row, i)] * sds[i]).powi(2)).sum();
            *slot = (mse_true[(j, j)] - shrink + spread).max(0.0).sqrt();
        }
    }
    let se_mc = (0.95f64 * 0.05 / reps as f64).sqrt();
    println!("soft density coverage (reps {reps}, T 4000, sd {sds:?} on series 0, exact conditional draws under the DGP)");
    for s in 0..h {
        let cov = [
            covered[s][0] as f64 / reps as f64,
            covered[s][1] as f64 / reps as f64,
        ];
        let calib = [sq_err[s][0] / reps as f64, sq_err[s][1] / reps as f64];
        println!(
            "  h={}: soft cell {:.4} free cell {:.4}; standardised-error variance {:.4} / {:.4}; \
             fitted/true se {:.4} / {:.4}",
            s + 1,
            cov[0],
            cov[1],
            calib[0],
            calib[1],
            cf.se[(s, 0)] / true_se[s][0],
            cf.se[(s, 1)] / true_se[s][1]
        );
        // The soft cell's se is the stated sd under both the DGP and the fit.
        assert!((cf.se[(s, 0)] / true_se[s][0] - 1.0).abs() < 1e-12);
        for j in 0..k {
            assert!(
                (cov[j] - 0.95).abs() < 4.0 * se_mc + 0.005,
                "h={} series {j}: density coverage {}",
                s + 1,
                cov[j]
            );
            assert!(
                (calib[j] - 1.0).abs() < 0.12,
                "h={} series {j}: {}",
                s + 1,
                calib[j]
            );
        }
    }
}

fn soft_fixture_model() -> tsecon_var::VarResults {
    let (c, a, chol) = dgp_var2_k3();
    let mut rng = Lcg::new(11);
    let y = simulate(&mut rng, 300, &c, &a, &chol, false, None);
    VarSpec::new(2, Trend::Constant)
        .unwrap()
        .fit(y.as_ref())
        .unwrap()
}

/// Two soft cells (series 1 at h = 1 and h = 3) and a hard one (series 2 at
/// h = 2), with the soft standard deviations given.
type Grid = Vec<Vec<Option<f64>>>;

fn soft_scenario(sd1: f64, sd3: f64) -> (Grid, Grid) {
    let conds = vec![
        vec![None, Some(0.4), None],
        vec![None, None, Some(-0.3)],
        vec![None, Some(0.2), None],
    ];
    let sd = vec![
        vec![None, Some(sd1), None],
        vec![None, None, Some(0.0)],
        vec![None, Some(sd3), None],
    ];
    (conds, sd)
}

/// Monotonicity: a signal-mode reading is informative but not exact (its
/// cell's se is below both its sd and its unconditional se), a noisier
/// reading never tightens any cell (Blackwell: the signal is garbled), and
/// stating a density spread never tightens any cell below the hard forecast.
#[test]
fn soft_condition_monotonicity() {
    let res = soft_fixture_model();
    let (h, k) = (6usize, 3usize);
    let (conds, _) = soft_scenario(0.0, 0.0);
    let hard = res.conditional_forecast(h, &conds, 0.05).unwrap();
    let grid = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0];
    let mut prev_sig: Option<tsecon_var::ConditionalForecast> = None;
    for &s1 in &grid {
        let (conds, sd) = soft_scenario(s1, 0.25);
        let sig = res
            .conditional_forecast_soft(h, &conds, &sd, SoftMode::Signal, 0.05)
            .unwrap();
        let den = res
            .conditional_forecast_soft(h, &conds, &sd, SoftMode::Density, 0.05)
            .unwrap();
        assert_eq!((sig.n_soft, den.n_soft, sig.n_constrained), (2, 2, 3));
        for (cell, sdv) in [((0usize, 1usize), s1), ((2, 1), 0.25)] {
            let se = sig.se[cell];
            assert!(
                se < sdv && se < sig.unconditional_se[cell],
                "sd {s1}: se {se}"
            );
            assert_eq!(den.se[cell].to_bits(), sdv.to_bits());
        }
        for hh in 0..h {
            for j in 0..k {
                assert!(
                    den.se[(hh, j)] >= hard.se[(hh, j)] * (1.0 - 1e-12),
                    "sd {s1}: density se ({hh},{j}) below the hard se"
                );
                if let Some(p) = &prev_sig {
                    assert!(
                        sig.se[(hh, j)] >= p.se[(hh, j)] * (1.0 - 1e-12),
                        "sd {s1}: signal se ({hh},{j}) fell when the reading got noisier"
                    );
                }
            }
        }
        prev_sig = Some(sig);
    }
}

/// Limits. sd -> 0 is the hard forecast in both modes (every quantity
/// within 1e-12); a signal reading with sd = 1e8 carries no information (the
/// unconditional forecast, to 1e-10); and scaling every stated sd by c in
/// density mode scales exactly the added variance term by c^2.
#[test]
fn soft_condition_limits_and_scaling() {
    let res = soft_fixture_model();
    let h = 6usize;
    let (conds, _) = soft_scenario(0.0, 0.0);
    let hard = res.conditional_forecast(h, &conds, 0.05).unwrap();
    let close = |a: &Mat<f64>, b: &Mat<f64>, tol: f64, what: &str| {
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                let d = (a[(i, j)] - b[(i, j)]).abs();
                assert!(
                    d <= tol * b[(i, j)].abs().max(1.0),
                    "{what}[({i},{j})]: {d:e}"
                );
            }
        }
    };
    let (_, tiny) = soft_scenario(1e-13, 1e-13);
    for mode in [SoftMode::Signal, SoftMode::Density] {
        let s = res
            .conditional_forecast_soft(h, &conds, &tiny, mode, 0.05)
            .unwrap();
        let what = format!("sd 1e-13 {mode:?}");
        for (key, a, b) in [
            ("point", &s.point, &hard.point),
            ("se", &s.se, &hard.se),
            ("shocks", &s.shocks, &hard.shocks),
            ("lower", &s.lower, &hard.lower),
        ] {
            close(a, b, 1e-12, &format!("{what} {key}"));
        }
        for hh in 0..h {
            close(
                &s.cov[hh],
                &hard.cov[hh],
                1e-12,
                &format!("{what} cov[{hh}]"),
            );
        }
        assert!((s.mahalanobis - hard.mahalanobis).abs() < 1e-12);
    }
    // Every cell soft with sd = 1e8: the readings say nothing.
    let conds_all = vec![
        vec![None, Some(0.4), None],
        vec![Some(9.0), None, Some(-0.3)],
    ];
    let huge = vec![
        vec![None, Some(1e8), None],
        vec![Some(1e8), None, Some(1e8)],
    ];
    let s = res
        .conditional_forecast_soft(h, &conds_all, &huge, SoftMode::Signal, 0.05)
        .unwrap();
    close(&s.point, &s.unconditional, 1e-10, "sd 1e8 point");
    close(&s.se, &s.unconditional_se, 1e-10, "sd 1e8 se");
    assert!(s.mahalanobis < 1e-12, "{}", s.mahalanobis);
    // Density scaling: cov(c sd) - cov_hard = c^2 (cov(sd) - cov_hard).
    let (_, sd1) = soft_scenario(0.3, 0.2);
    let (_, sd3) = soft_scenario(0.9, 0.6000000000000001);
    let d1 = res
        .conditional_forecast_soft(h, &conds, &sd1, SoftMode::Density, 0.05)
        .unwrap();
    let d3 = res
        .conditional_forecast_soft(h, &conds, &sd3, SoftMode::Density, 0.05)
        .unwrap();
    for hh in 0..h {
        let add1 = &d1.cov[hh] - &hard.cov[hh];
        let add3 = &d3.cov[hh] - &hard.cov[hh];
        let nine = Mat::from_fn(3, 3, |i, j| 9.0 * add1[(i, j)]);
        close(&add3, &nine, 1e-10, &format!("density scaling cov[{hh}]"));
    }
    // The path, shocks and Mahalanobis statistic do not depend on sd.
    assert_eq!(d1.point, hard.point);
    assert_eq!(d3.shocks, hard.shocks);
    assert_eq!(d3.mahalanobis.to_bits(), hard.mahalanobis.to_bits());
}

/// Rust-level refusals of `condition_sd`, each naming it (the binding adds
/// the cell indices), and the all-zero grid that is NOT a refusal.
#[test]
fn soft_refusals_name_condition_sd() {
    let res = soft_fixture_model();
    let conds = vec![vec![None, Some(0.4), None], vec![None, None, Some(-0.3)]];
    let ok = vec![vec![None, Some(0.2), None], vec![None, None, None]];
    assert!(res
        .conditional_forecast_soft(4, &conds, &ok, SoftMode::Signal, 0.05)
        .is_ok());
    let bad = |sd: Vec<Vec<Option<f64>>>| {
        res.conditional_forecast_soft(4, &conds, &sd, SoftMode::Density, 0.05)
            .unwrap_err()
            .to_string()
    };
    let e = bad(vec![vec![None, Some(-0.1), None], vec![None, None, None]]);
    assert!(
        e.contains("condition_sd = -0.1") && e.contains("finite standard deviation >= 0"),
        "{e}"
    );
    let e = bad(vec![
        vec![None, Some(f64::INFINITY), None],
        vec![None, None, None],
    ]);
    assert!(e.contains("condition_sd = inf"), "{e}");
    let e = bad(vec![vec![None, Some(1e200), None], vec![None, None, None]]);
    assert!(e.contains("condition_sd") && e.contains("square"), "{e}");
    let e = bad(vec![
        vec![Some(0.5), Some(0.2), None],
        vec![None, None, None],
    ]);
    assert!(
        e.contains("condition_sd = 0.5") && e.contains("free cell"),
        "{e}"
    );
    let e = bad(vec![vec![None, Some(0.2), None]]);
    assert!(
        e.contains("condition_sd must have exactly one row per row of conditions")
            && e.contains("expected 2, got 1"),
        "{e}"
    );
    let e = bad(vec![vec![None, Some(0.2)], vec![None, None, None]]);
    assert!(
        e.contains("every row of condition_sd") && e.contains("expected 3, got 2"),
        "{e}"
    );
    // conditions are checked first, with their own messages.
    let e = res
        .conditional_forecast_soft(
            4,
            &[vec![None, Some(f64::INFINITY), None]],
            &[vec![None, Some(-1.0), None]],
            SoftMode::Signal,
            0.05,
        )
        .unwrap_err()
        .to_string();
    assert!(e.contains("conditions contains an infinite value"), "{e}");
    let e = res
        .conditional_forecast_soft(1usize << 40, &conds, &ok, SoftMode::Signal, 0.05)
        .unwrap_err()
        .to_string();
    assert!(e.contains("memory budget") && e.contains("steps"), "{e}");
    // 0, -0, None and NaN all mean hard: the hard path, mode Hard.
    let zero = vec![
        vec![None, Some(-0.0), Some(f64::NAN)],
        vec![Some(0.0), None, Some(0.0)],
    ];
    let r = res
        .conditional_forecast_soft(4, &conds, &zero, SoftMode::Signal, 0.05)
        .unwrap();
    assert_eq!((r.mode, r.n_soft), (ConditionMode::Hard, 0));
    assert_eq!(r.mode.as_str(), "hard");
}
