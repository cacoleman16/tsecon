//! Golden-value tests against the JSON fixtures produced by
//! NumPy/SciPy/statsmodels (see fixtures/generate_fixtures.py).

mod common;

use common::{as_f64_vec, as_mat, assert_mat_close, assert_slice_close, load_fixture};
use tsecon_linalg::{
    balanced_eigenvalues, companion_from_var, is_stable, levinson_durbin_from_series,
    solve_discrete_lyapunov, spectral_radius, toeplitz_solve,
};

/// `toeplitz_solve` matches `scipy.linalg.solve_toeplitz` to 1e-10.
#[test]
fn golden_toeplitz_solve() {
    let fx = load_fixture("linalg.json");
    let block = &fx["toeplitz_solve"];
    let first_col = as_f64_vec(&block["first_col"]);
    let rhs = as_f64_vec(&block["rhs"]);
    let expected = as_f64_vec(&block["x"]);

    let x = toeplitz_solve(&first_col, &rhs).unwrap();
    assert_slice_close(&x, &expected, 1e-10, "toeplitz_solve");
}

/// `solve_discrete_lyapunov` matches `scipy.linalg.solve_discrete_lyapunov`
/// to 1e-10 on both fixture cases (2x2 hand-picked and 4x4 random stable).
#[test]
fn golden_discrete_lyapunov() {
    let fx = load_fixture("linalg.json");
    let cases = fx["discrete_lyapunov"].as_array().unwrap();
    assert_eq!(cases.len(), 2, "fixture should hold two Lyapunov cases");
    for (idx, case) in cases.iter().enumerate() {
        let a = as_mat(&case["a"]);
        let q = as_mat(&case["q"]);
        let expected = as_mat(&case["x"]);
        let x = solve_discrete_lyapunov(a.as_ref(), q.as_ref()).unwrap();
        assert_mat_close(&x, &expected, 1e-10, &format!("discrete_lyapunov[{idx}]"));
    }
}

/// Levinson-Durbin on the Nile series matches
/// `statsmodels.tsa.stattools.levinson_durbin(y, nlags=10, isacov=False)`:
/// biased autocovariances of the demeaned series, order-10 AR coefficients,
/// the PACF sequence, and the final innovation variance.
#[test]
fn golden_levinson_durbin_nile() {
    let fx = load_fixture("diagnostics.json");
    let nile = as_f64_vec(&fx["nile"]);
    let block = &fx["levinson_durbin_10"];
    let expected_ar = as_f64_vec(&block["ar_coefs"]);
    let expected_pacf = as_f64_vec(&block["pacf"]);
    let expected_sigma2 = block["sigma2_final"].as_f64().unwrap();

    let ld = levinson_durbin_from_series(&nile, 10).unwrap();

    assert_slice_close(ld.ar_coefs_final(), &expected_ar, 1e-12, "ar_coefs");
    assert_slice_close(&ld.pacf, &expected_pacf, 1e-12, "pacf");
    // sigma2 is ~1.9e4; compare at 1e-12 relative.
    let sigma2 = ld.innovation_variance_final();
    assert!(
        (sigma2 - expected_sigma2).abs() <= 1e-12 * expected_sigma2.abs(),
        "sigma2_final: {sigma2} vs {expected_sigma2}"
    );

    // Shape conventions: pacf has order + 1 entries with pacf[0] = 1,
    // innovation variances start at gamma(0), one AR vector per order.
    assert_eq!(ld.pacf.len(), 11);
    assert_eq!(ld.pacf[0], 1.0);
    assert_eq!(ld.innovation_variance.len(), 11);
    assert_eq!(ld.ar_coefs.len(), 10);
    for (m, coefs) in ld.ar_coefs.iter().enumerate() {
        assert_eq!(coefs.len(), m + 1, "order {} coefficient count", m + 1);
    }
    // The last coefficient at each order is the PACF at that lag.
    for m in 1..=10 {
        assert_eq!(ld.ar_coefs[m - 1][m - 1], ld.pacf[m]);
    }
}

/// Companion eigenvalue moduli of badly scaled VARs (entries spanning up to
/// `3e-15 .. 8e13`) match a 60-digit mpmath solve of the exact float64
/// companion at 1e-12 relative (fixtures/companion_roots.json; an exactly
/// zero eigenvalue is compared against the radius). Before the balancing,
/// `seed21_var2_scaled` lost the third decimal of its near-unit roots.
#[test]
fn golden_balanced_companion_eigenvalues() {
    let fx = load_fixture("companion_roots.json");
    let cases = fx["companions"].as_array().unwrap();
    assert_eq!(cases.len(), 4, "fixture should hold four companion cases");
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let coefs: Vec<_> = case["coefs"]
            .as_array()
            .unwrap()
            .iter()
            .map(as_mat)
            .collect();
        let refs: Vec<_> = coefs.iter().map(|m| m.as_ref()).collect();
        let comp = companion_from_var(&refs).unwrap();
        let expected = as_f64_vec(&case["moduli"]);
        let radius = case["spectral_radius"].as_f64().unwrap();
        let mut moduli: Vec<f64> = balanced_eigenvalues(comp.as_ref())
            .unwrap()
            .iter()
            .map(|z| z.re.hypot(z.im))
            .collect();
        moduli.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(moduli.len(), expected.len(), "{name}");
        for (i, (m, e)) in moduli.iter().zip(&expected).enumerate() {
            let scale = if *e > 0.0 { *e } else { radius };
            assert!(
                (m - e).abs() <= 1e-12 * scale,
                "{name}: modulus {i} = {m:.17e}, mpmath {e:.17e}"
            );
        }
        let rho = spectral_radius(comp.as_ref()).unwrap();
        assert!(
            (rho - radius).abs() <= 1e-12 * radius,
            "{name}: spectral radius {rho:.17e}, mpmath {radius:.17e}"
        );
        assert!(is_stable(comp.as_ref(), 0.0).unwrap(), "{name}");
    }
}
