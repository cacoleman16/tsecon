//! Property / invariant tests beyond the golden fixtures.

mod common;

use common::{as_f64_vec, assert_slice_close, load_fixture, Lcg};
use faer::linalg::solvers::Solve;
use faer::Mat;
use tsecon_linalg::{
    ar_psi_weights, autocovariances_biased, balance, balanced_eigenvalues, companion_from_ar,
    companion_from_var, is_stable, jittered_cholesky, levinson_durbin, levinson_durbin_from_series,
    solve_discrete_lyapunov, spectral_radius, symmetrize, toeplitz_solve, LinalgError,
};

/// Random matrix with entries in (-1, 1), rescaled to a target spectral
/// radius.
fn random_stable(rng: &mut Lcg, n: usize, target_rho: f64) -> Mat<f64> {
    loop {
        let m = Mat::from_fn(n, n, |_, _| rng.symmetric());
        let rho = spectral_radius(m.as_ref()).unwrap();
        if rho > 1e-8 {
            let scale = target_rho / rho;
            return Mat::from_fn(n, n, |i, j| m[(i, j)] * scale);
        }
        // Nilpotent draw (probability ~0); redraw.
    }
}

/// Random symmetric positive definite matrix `B B' + I`.
fn random_spd(rng: &mut Lcg, n: usize) -> Mat<f64> {
    let b = Mat::from_fn(n, n, |_, _| rng.symmetric());
    &b * b.transpose() + Mat::<f64>::identity(n, n)
}

/// Frobenius norm of a matrix.
fn fro(m: &Mat<f64>) -> f64 {
    m.as_ref().norm_l2()
}

// ---------------------------------------------------------------- Lyapunov

/// Residual invariant: ||X - A X A' - Q||_F / ||X||_F < 1e-12 on random
/// stable transition matrices of assorted sizes and spectral radii.
#[test]
fn lyapunov_residual_random_stable() {
    let mut rng = Lcg::new(20260716);
    for n in 1..=8usize {
        for &target_rho in &[0.1, 0.5, 0.9, 0.97] {
            let a = random_stable(&mut rng, n, target_rho);
            let q = random_spd(&mut rng, n);
            let x = solve_discrete_lyapunov(a.as_ref(), q.as_ref()).unwrap();
            let residual = &x - &a * &x * a.transpose() - &q;
            let rel = fro(&residual) / fro(&x);
            assert!(
                rel < 1e-12,
                "n={n} rho={target_rho}: relative residual {rel:e}"
            );
        }
    }
}

/// The solution of a symmetric-Q equation is exactly symmetric.
#[test]
fn lyapunov_solution_symmetric() {
    let mut rng = Lcg::new(7);
    let a = random_stable(&mut rng, 5, 0.8);
    let q = random_spd(&mut rng, 5);
    let x = solve_discrete_lyapunov(a.as_ref(), q.as_ref()).unwrap();
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(x[(i, j)], x[(j, i)], "asymmetry at ({i},{j})");
        }
    }
}

/// A unit root or explosive transition must error, never spin.
#[test]
fn lyapunov_rejects_unstable() {
    let q = Mat::<f64>::identity(1, 1);
    for &val in &[1.0, 1.2, -1.0] {
        let a = Mat::from_fn(1, 1, |_, _| val);
        match solve_discrete_lyapunov(a.as_ref(), q.as_ref()) {
            Err(LinalgError::Unstable { spectral_radius }) => {
                assert!((spectral_radius - val.abs()).abs() < 1e-12);
            }
            other => panic!("expected Unstable error for a={val}, got {other:?}"),
        }
    }
}

/// AR(1) closed form: X = q / (1 - a^2).
#[test]
fn lyapunov_ar1_closed_form() {
    let a = Mat::from_fn(1, 1, |_, _| 0.9);
    let q = Mat::from_fn(1, 1, |_, _| 2.5);
    let x = solve_discrete_lyapunov(a.as_ref(), q.as_ref()).unwrap();
    let expected = 2.5 / (1.0 - 0.81);
    assert!((x[(0, 0)] - expected).abs() < 1e-12 * expected);
}

/// Shape and finiteness violations are reported as errors.
#[test]
fn lyapunov_input_validation() {
    let a = Mat::<f64>::zeros(2, 3);
    let q = Mat::<f64>::identity(2, 2);
    assert!(matches!(
        solve_discrete_lyapunov(a.as_ref(), q.as_ref()),
        Err(LinalgError::NotSquare { .. })
    ));
    let a = Mat::<f64>::zeros(3, 3);
    assert!(matches!(
        solve_discrete_lyapunov(a.as_ref(), q.as_ref()),
        Err(LinalgError::DimensionMismatch { .. })
    ));
    let a = Mat::from_fn(2, 2, |_, _| f64::NAN);
    assert!(matches!(
        solve_discrete_lyapunov(a.as_ref(), q.as_ref()),
        Err(LinalgError::NonFinite { .. })
    ));
}

// ---------------------------------------------------- Levinson / Toeplitz

/// Levinson-Durbin AR coefficients equal the direct dense solution of the
/// Yule-Walker system Gamma phi = gamma (faer LU solver) at every order.
#[test]
fn levinson_matches_dense_yule_walker() {
    let fx = load_fixture("diagnostics.json");
    let nile = as_f64_vec(&fx["nile"]);
    let max_order = 10;
    let acov = autocovariances_biased(&nile, max_order).unwrap();
    let ld = levinson_durbin(&acov, max_order).unwrap();

    for p in 1..=max_order {
        let gamma_mat = Mat::from_fn(p, p, |i, j| acov[i.abs_diff(j)]);
        let rhs = Mat::from_fn(p, 1, |i, _| acov[i + 1]);
        let phi_dense = gamma_mat.partial_piv_lu().solve(&rhs);
        let dense: Vec<f64> = (0..p).map(|i| phi_dense[(i, 0)]).collect();
        assert_slice_close(
            &ld.ar_coefs[p - 1],
            &dense,
            1e-10,
            &format!("Yule-Walker order {p}"),
        );
    }
}

/// Innovation variances are positive, decreasing, and consistent with the
/// PACF product formula v_p = gamma(0) * prod (1 - pacf_m^2).
#[test]
fn levinson_variance_invariants() {
    let fx = load_fixture("diagnostics.json");
    let nile = as_f64_vec(&fx["nile"]);
    let ld = levinson_durbin_from_series(&nile, 10).unwrap();
    let v = &ld.innovation_variance;
    let mut product = v[0];
    for m in 1..v.len() {
        assert!(v[m] > 0.0, "v[{m}] must be positive");
        assert!(v[m] <= v[m - 1] + 1e-12, "v must be nonincreasing");
        product *= 1.0 - ld.pacf[m] * ld.pacf[m];
        assert!(
            (v[m] - product).abs() <= 1e-9 * product,
            "product formula at order {m}"
        );
    }
}

/// A non-positive-definite autocovariance sequence errors instead of
/// returning garbage (|gamma(1)| > gamma(0) implies |pacf| > 1).
#[test]
fn levinson_rejects_invalid_acov() {
    assert!(matches!(
        levinson_durbin(&[1.0, 1.5], 1),
        Err(LinalgError::NotPositiveDefinite { .. })
    ));
    assert!(matches!(
        levinson_durbin(&[0.0, 0.0], 1),
        Err(LinalgError::NotPositiveDefinite { .. })
    ));
    assert!(matches!(
        levinson_durbin(&[1.0], 1),
        Err(LinalgError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        levinson_durbin(&[], 0),
        Err(LinalgError::EmptyInput { .. })
    ));
    assert!(matches!(
        levinson_durbin(&[1.0, f64::NAN], 1),
        Err(LinalgError::NonFinite { .. })
    ));
}

/// Order zero is legal and returns the trivial decomposition.
#[test]
fn levinson_order_zero() {
    let ld = levinson_durbin(&[3.0], 0).unwrap();
    assert!(ld.ar_coefs.is_empty());
    assert_eq!(ld.pacf, vec![1.0]);
    assert_eq!(ld.innovation_variance, vec![3.0]);
    assert_eq!(ld.ar_coefs_final(), &[] as &[f64]);
    assert_eq!(ld.innovation_variance_final(), 3.0);
}

/// toeplitz_solve equals the dense faer LU solution on random SPD Toeplitz
/// systems (AR(1) autocovariance columns are always positive definite).
#[test]
fn toeplitz_matches_dense_solve() {
    let mut rng = Lcg::new(42);
    for n in 1..=12usize {
        let rho = 1.8 * rng.uniform() - 0.9; // (-0.9, 0.9)
        let sigma2 = 0.5 + 2.0 * rng.uniform();
        let col: Vec<f64> = (0..n).map(|h| sigma2 * rho.powi(h as i32)).collect();
        let rhs: Vec<f64> = (0..n).map(|_| rng.symmetric() * 3.0).collect();

        let x = toeplitz_solve(&col, &rhs).unwrap();

        let t = Mat::from_fn(n, n, |i, j| col[i.abs_diff(j)]);
        let b = Mat::from_fn(n, 1, |i, _| rhs[i]);
        let dense = t.partial_piv_lu().solve(&b);
        let dense: Vec<f64> = (0..n).map(|i| dense[(i, 0)]).collect();
        assert_slice_close(&x, &dense, 1e-10, &format!("toeplitz n={n} rho={rho}"));
    }
}

/// Solving against the first column of the identity-scaled Toeplitz
/// reproduces e_1 scaling; degenerate/invalid inputs error.
#[test]
fn toeplitz_edge_cases() {
    // 1x1 system.
    let x = toeplitz_solve(&[4.0], &[2.0]).unwrap();
    assert!((x[0] - 0.5).abs() < 1e-15);

    assert!(matches!(
        toeplitz_solve(&[], &[]),
        Err(LinalgError::EmptyInput { .. })
    ));
    assert!(matches!(
        toeplitz_solve(&[1.0, 0.5], &[1.0]),
        Err(LinalgError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        toeplitz_solve(&[-1.0, 0.0], &[1.0, 1.0]),
        Err(LinalgError::NotPositiveDefinite { .. })
    ));
    // |r_1| > r_0: indefinite Toeplitz matrix.
    assert!(matches!(
        toeplitz_solve(&[1.0, 2.0], &[1.0, 1.0]),
        Err(LinalgError::NotPositiveDefinite { .. })
    ));
}

// ------------------------------------------------------------- companion

/// Psi-weights of a stable AR(1) are phi^k.
#[test]
fn psi_weights_ar1_geometric() {
    let phi = 0.7;
    let psi = ar_psi_weights(&[phi], 50).unwrap();
    assert_eq!(psi.len(), 51);
    for (k, &w) in psi.iter().enumerate() {
        let expected = phi.powi(k as i32);
        assert!(
            (w - expected).abs() <= 1e-13 * expected.abs().max(1.0),
            "psi[{k}] = {w} vs {expected}"
        );
    }
}

/// AR(2) psi-weights satisfy the defining convolution identity
/// phi(L) psi(L) = 1 (coefficients of L^j vanish for j >= 1).
#[test]
fn psi_weights_invert_ar_polynomial() {
    let phi = [1.2, -0.35];
    let psi = ar_psi_weights(&phi, 30).unwrap();
    assert_eq!(psi[0], 1.0);
    for j in 1..=30usize {
        let mut coef = psi[j];
        for i in 1..=j.min(2) {
            coef -= phi[i - 1] * psi[j - i];
        }
        assert!(coef.abs() < 1e-14, "convolution coefficient at lag {j}");
    }
}

/// Companion eigenvalue of an AR(1) equals phi; AR(2) companion
/// eigenvalues equal the known lag-polynomial roots.
#[test]
fn companion_eigenvalues() {
    let c1 = companion_from_ar(&[0.7]).unwrap();
    assert_eq!(c1.nrows(), 1);
    assert_eq!(c1[(0, 0)], 0.7);
    let rho = spectral_radius(c1.as_ref()).unwrap();
    assert!((rho - 0.7).abs() < 1e-14);

    // x_t = 1.2 x_{t-1} - 0.35 x_{t-2}: roots of z^2 - 1.2 z + 0.35 are
    // z = 0.7 and z = 0.5.
    let c2 = companion_from_ar(&[1.2, -0.35]).unwrap();
    let mut moduli: Vec<f64> = c2
        .eigenvalues()
        .unwrap()
        .iter()
        .map(|z| z.re.hypot(z.im))
        .collect();
    moduli.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((moduli[0] - 0.5).abs() < 1e-12);
    assert!((moduli[1] - 0.7).abs() < 1e-12);
}

/// The VAR(1) companion of scalar blocks reduces to the AR companion, and
/// a VAR(2) companion has the documented block layout.
#[test]
fn companion_var_layout() {
    let a1 = Mat::from_fn(2, 2, |i, j| [[0.5, 0.1], [0.0, 0.3]][i][j]);
    let a2 = Mat::from_fn(2, 2, |i, j| [[0.2, 0.0], [0.05, -0.1]][i][j]);
    let c = companion_from_var(&[a1.as_ref(), a2.as_ref()]).unwrap();
    assert_eq!((c.nrows(), c.ncols()), (4, 4));
    // Top blocks.
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(c[(i, j)], a1[(i, j)]);
            assert_eq!(c[(i, 2 + j)], a2[(i, j)]);
        }
    }
    // Identity block and zero block.
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(c[(2 + i, j)], if i == j { 1.0 } else { 0.0 });
            assert_eq!(c[(2 + i, 2 + j)], 0.0);
        }
    }

    // k = 1 reduces to the scalar companion.
    let m1 = Mat::from_fn(1, 1, |_, _| 0.6);
    let m2 = Mat::from_fn(1, 1, |_, _| 0.2);
    let cv = companion_from_var(&[m1.as_ref(), m2.as_ref()]).unwrap();
    let ca = companion_from_ar(&[0.6, 0.2]).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(cv[(i, j)], ca[(i, j)]);
        }
    }
}

/// Stability check honours the tolerance argument.
#[test]
fn stability_tolerance() {
    let c = companion_from_ar(&[0.99]).unwrap();
    assert!(is_stable(c.as_ref(), 0.0).unwrap());
    assert!(!is_stable(c.as_ref(), 0.02).unwrap());
    let unit = companion_from_ar(&[1.0]).unwrap();
    assert!(!is_stable(unit.as_ref(), 0.0).unwrap());
    assert!(matches!(
        is_stable(c.as_ref(), -0.1),
        Err(LinalgError::InvalidArgument { .. })
    ));
    assert!(matches!(
        is_stable(c.as_ref(), 1.0),
        Err(LinalgError::InvalidArgument { .. })
    ));
}

/// A stationary AR fitted by Levinson-Durbin yields a stable companion
/// matrix (cross-module invariant).
#[test]
fn levinson_ar_is_stable() {
    let fx = load_fixture("diagnostics.json");
    let nile = as_f64_vec(&fx["nile"]);
    let ld = levinson_durbin_from_series(&nile, 10).unwrap();
    let c = companion_from_ar(ld.ar_coefs_final()).unwrap();
    assert!(is_stable(c.as_ref(), 0.0).unwrap());
}

// --------------------------------------------------------------- hygiene

/// symmetrize returns the exact symmetric part and is idempotent.
#[test]
fn symmetrize_properties() {
    let mut rng = Lcg::new(3);
    let m = Mat::from_fn(4, 4, |_, _| rng.symmetric());
    let s = symmetrize(m.as_ref()).unwrap();
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(s[(i, j)], s[(j, i)]);
            let expected = 0.5 * (m[(i, j)] + m[(j, i)]);
            assert!((s[(i, j)] - expected).abs() < 1e-16);
        }
    }
    let s2 = symmetrize(s.as_ref()).unwrap();
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(s[(i, j)], s2[(i, j)]);
        }
    }
    assert!(matches!(
        symmetrize(Mat::<f64>::zeros(2, 3).as_ref()),
        Err(LinalgError::NotSquare { .. })
    ));
}

/// jittered_cholesky: clean factorization of an SPD matrix uses no jitter
/// and reconstructs the input; L is lower triangular.
#[test]
fn jittered_cholesky_clean_path() {
    let mut rng = Lcg::new(11);
    let m = random_spd(&mut rng, 5);
    let res = jittered_cholesky(m.as_ref()).unwrap();
    assert_eq!(res.jitter, 0.0);
    assert_eq!(res.attempts, 1);
    let rebuilt = &res.factor * res.factor.transpose();
    for i in 0..5 {
        for j in 0..5 {
            assert!(
                (rebuilt[(i, j)] - m[(i, j)]).abs() < 1e-12 * fro(&m),
                "reconstruction at ({i},{j})"
            );
            if j > i {
                assert_eq!(res.factor[(i, j)], 0.0, "upper triangle must be zero");
            }
        }
    }
    // log-det agrees with the closed form for a diagonal matrix.
    let d = Mat::from_fn(3, 3, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
    let ld = jittered_cholesky(d.as_ref()).unwrap().log_det();
    assert!((ld - (6.0f64).ln()).abs() < 1e-14);
}

/// A singular PSD matrix triggers the jitter ladder and reports the jitter
/// used; the factor reproduces the matrix up to the reported jitter.
#[test]
fn jittered_cholesky_jitter_path() {
    // rank-1 PSD matrix vv' (singular, needs jitter).
    let v = [1.0, 2.0, 3.0];
    let m = Mat::from_fn(3, 3, |i, j| v[i] * v[j]);
    let res = jittered_cholesky(m.as_ref()).unwrap();
    assert!(res.jitter > 0.0, "singular matrix must need jitter");
    assert!(res.attempts > 1);
    let rebuilt = &res.factor * res.factor.transpose();
    for i in 0..3 {
        for j in 0..3 {
            let target = m[(i, j)] + if i == j { res.jitter } else { 0.0 };
            assert!(
                (rebuilt[(i, j)] - target).abs() <= 1e-8,
                "reconstruction at ({i},{j})"
            );
        }
    }
}

/// A genuinely indefinite matrix exhausts the bounded ladder and errors.
#[test]
fn jittered_cholesky_exhausts_on_indefinite() {
    let m = Mat::from_fn(2, 2, |i, j| if i == j { [1.0, -1.0][i] } else { 0.0 });
    match jittered_cholesky(m.as_ref()) {
        Err(LinalgError::JitterExhausted {
            attempts,
            max_jitter,
        }) => {
            assert!(attempts >= 2);
            assert!(max_jitter > 0.0);
            // Ladder is bounded: max jitter is 1e-8 * scale (scale = 1 here).
            assert!(max_jitter <= 1e-8 * 1.5);
        }
        other => panic!("expected JitterExhausted, got {other:?}"),
    }
}

/// An asymmetric input is symmetrized before factorization.
#[test]
fn jittered_cholesky_symmetrizes() {
    let m = Mat::from_fn(2, 2, |i, j| [[4.0, 0.9], [1.1, 3.0]][i][j]);
    let res = jittered_cholesky(m.as_ref()).unwrap();
    let rebuilt = &res.factor * res.factor.transpose();
    assert!((rebuilt[(0, 1)] - 1.0).abs() < 1e-14); // 0.5 (0.9 + 1.1)
    assert!((rebuilt[(1, 0)] - 1.0).abs() < 1e-14);
}

// ------------------------------------------------- cross-module coherence

/// The stationary variance of an AR(1) from the Lyapunov solver agrees
/// with the psi-weight series sum sigma2 * sum_k psi_k^2.
#[test]
fn lyapunov_agrees_with_psi_expansion() {
    let phi = 0.8;
    let sigma2 = 1.7;
    let a = companion_from_ar(&[phi]).unwrap();
    let q = Mat::from_fn(1, 1, |_, _| sigma2);
    let x = solve_discrete_lyapunov(a.as_ref(), q.as_ref()).unwrap();

    let psi = ar_psi_weights(&[phi], 400).unwrap();
    let series: f64 = psi.iter().map(|w| sigma2 * w * w).sum();
    assert!(
        (x[(0, 0)] - series).abs() < 1e-10,
        "{} vs {series}",
        x[(0, 0)]
    );
}

// ---------------------------------------------------------------- balancing

/// A random VAR(p) lag-matrix set with radius `target`, each matrix put
/// through the similarity `S A S^{-1}` with `S = diag(scales)` — the
/// coefficients of the same VAR with series `i` in units `scales[i]`.
fn scaled_var(rng: &mut Lcg, k: usize, p: usize, target: f64, scales: &[f64]) -> Vec<Mat<f64>> {
    loop {
        let a: Vec<Mat<f64>> = (0..p)
            .map(|l| Mat::from_fn(k, k, |_, _| rng.symmetric() * 0.5 / (l + 1) as f64))
            .collect();
        let refs: Vec<_> = a.iter().map(|m| m.as_ref()).collect();
        let rho = spectral_radius(companion_from_var(&refs).unwrap().as_ref()).unwrap();
        if rho < 1e-6 {
            continue;
        }
        let s = target / rho;
        return a
            .iter()
            .enumerate()
            .map(|(l, m)| {
                let f = s.powi(l as i32 + 1);
                Mat::from_fn(k, k, |i, j| m[(i, j)] * f * scales[i] / scales[j])
            })
            .collect();
    }
}

fn sorted_moduli(a: &Mat<f64>) -> Vec<f64> {
    let mut m: Vec<f64> = balanced_eigenvalues(a.as_ref())
        .unwrap()
        .iter()
        .map(|z| z.re.hypot(z.im))
        .collect();
    m.sort_by(|x, y| y.partial_cmp(x).unwrap());
    m
}

/// The balancing is an EXACT similarity: every factor is a power of two,
/// and the balanced matrix equals `D^{-1} A D` bit for bit.
#[test]
fn balancing_is_an_exact_power_of_two_similarity() {
    let mut rng = Lcg::new(2016);
    for (k, p, spread) in [
        (3usize, 2usize, 1e7f64),
        (4, 3, 1e6),
        (2, 4, 1e3),
        (5, 1, 1e10),
    ] {
        let scales: Vec<f64> = (0..k)
            .map(|i| spread.powf(1.0 - 2.0 * i as f64 / (k - 1) as f64))
            .collect();
        let a = scaled_var(&mut rng, k, p, 0.95, &scales);
        let refs: Vec<_> = a.iter().map(|m| m.as_ref()).collect();
        let comp = companion_from_var(&refs).unwrap();
        let (b, d) = balance(comp.as_ref()).unwrap();
        let n = comp.nrows();
        assert!(
            d.iter().any(|&f| f != 1.0),
            "k={k} p={p}: nothing was balanced"
        );
        for &f in &d {
            assert!(
                f > 0.0 && f.log2().fract() == 0.0,
                "factor {f} is not a power of two"
            );
        }
        for j in 0..n {
            for i in 0..n {
                let exact = comp[(i, j)] / d[i] * d[j];
                assert_eq!(
                    b[(i, j)].to_bits(),
                    exact.to_bits(),
                    "k={k} p={p} ({i},{j})"
                );
            }
        }
    }
}

/// Rescaling the series of a VAR is a diagonal similarity of its
/// companion matrix, so the eigenvalue moduli must not move. With units up
/// to 1e+-7 apart they agree with the unscaled solve to 1e-12 relative
/// (the unbalanced solve was off by 1e-3 at 1e+-7).
#[test]
fn companion_moduli_are_invariant_to_the_units_of_the_series() {
    let mut rng = Lcg::new(77);
    for (k, p) in [(3usize, 2usize), (4, 3), (2, 6)] {
        let ones = vec![1.0; k];
        let a = scaled_var(&mut rng, k, p, 0.9995, &ones);
        let refs: Vec<_> = a.iter().map(|m| m.as_ref()).collect();
        let base = sorted_moduli(&companion_from_var(&refs).unwrap());
        for spread in [1e3f64, 1e5, 1e7] {
            let s: Vec<f64> = (0..k)
                .map(|i| spread.powf(1.0 - 2.0 * i as f64 / (k - 1) as f64))
                .collect();
            let scaled: Vec<Mat<f64>> = a
                .iter()
                .map(|m| Mat::from_fn(k, k, |i, j| m[(i, j)] * s[i] / s[j]))
                .collect();
            let refs: Vec<_> = scaled.iter().map(|m| m.as_ref()).collect();
            let comp = companion_from_var(&refs).unwrap();
            let got = sorted_moduli(&comp);
            for (g, e) in got.iter().zip(&base) {
                assert!(
                    (g - e).abs() <= 1e-12 * e,
                    "k={k} p={p} spread {spread:e}: {g:.17e} vs {e:.17e}"
                );
            }
            let rho = spectral_radius(comp.as_ref()).unwrap();
            assert!((rho - base[0]).abs() <= 1e-12 * base[0]);
            assert!(is_stable(comp.as_ref(), 0.0).unwrap());
        }
    }
}

/// A zero column of `A_p` makes a zero companion column; the balancing
/// leaves that index alone (LAPACK skips it) while scaling the others. (The
/// mpmath golden `var2_k3_zero_column` pins the eigenvalues of such a
/// companion.)
#[test]
fn balancing_skips_an_isolated_index_and_refuses_bad_shapes() {
    let a1 = Mat::from_fn(2, 2, |i, j| [[0.5, 3e6], [2e-7, 0.3]][i][j]);
    let a2 = Mat::from_fn(2, 2, |i, j| [[0.1, 0.0], [1e-8, 0.0]][i][j]);
    let comp = companion_from_var(&[a1.as_ref(), a2.as_ref()]).unwrap();
    let (_, d) = balance(comp.as_ref()).unwrap();
    // Index 3 (lag-2 copy of series 1) has a zero column: never scaled.
    assert_eq!(d[3], 1.0);
    assert!(d[..3].iter().any(|&f| f != 1.0), "{d:?}");

    let rect = Mat::<f64>::zeros(2, 3);
    assert!(matches!(
        balance(rect.as_ref()),
        Err(LinalgError::NotSquare { .. })
    ));
    let empty = Mat::<f64>::zeros(0, 0);
    assert!(matches!(
        balanced_eigenvalues(empty.as_ref()),
        Err(LinalgError::EmptyInput { .. })
    ));
    let mut nan = Mat::<f64>::identity(2, 2);
    nan[(0, 1)] = f64::NAN;
    assert!(matches!(
        balance(nan.as_ref()),
        Err(LinalgError::NonFinite { .. })
    ));
}
