//! Property and invariant tests: OLS orthogonality, IRFs of a diagonal
//! VAR(1), FEVD row sums, a seeded simulate-estimate round trip, and
//! error-path validation.

mod common;

use common::{as_mat, load_fixture, Lcg};
use tsecon_linalg::faer::Mat;
use tsecon_var::{ma_rep, select_order, Trend, VarError, VarSpec};

/// OLS residuals are orthogonal to every regressor: rebuilding the
/// design matrix Z of the fixture VAR(2)-with-constant fit, `Z'U` is
/// zero to roundoff (the normal equations).
#[test]
fn residuals_orthogonal_to_regressors() {
    let fx = load_fixture("var.json");
    let data = as_mat(&fx["data_100dlog_gdp_cons_inv"]);
    let res = VarSpec::new(2, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap();

    let (p, k, t_eff) = (2usize, res.neqs, res.nobs);
    let m = 1 + k * p;
    let z = Mat::from_fn(t_eff, m, |t, c| {
        if c == 0 {
            1.0
        } else {
            let lag = (c - 1) / k + 1;
            let var = (c - 1) % k;
            data[(p + t - lag, var)]
        }
    });
    let ztu = z.transpose() * &res.resid;
    for i in 0..m {
        for j in 0..k {
            assert!(
                ztu[(i, j)].abs() < 1e-7,
                "Z'U[({i},{j})] = {} not ~ 0",
                ztu[(i, j)]
            );
        }
    }
}

/// The MA representation of a diagonal VAR(1) is the elementwise powers
/// of the diagonal: `Psi_h = A^h` with `A = diag(a_1, a_2)`.
#[test]
fn ma_rep_of_diagonal_var1_is_diagonal_powers() {
    let a = Mat::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => 0.5,
        (1, 1) => -0.3,
        _ => 0.0,
    });
    let psi = ma_rep(&[a], 8).unwrap();
    assert_eq!(psi.len(), 9);
    for (h, m) in psi.iter().enumerate() {
        let e00 = 0.5f64.powi(h as i32);
        let e11 = (-0.3f64).powi(h as i32);
        assert!((m[(0, 0)] - e00).abs() < 1e-14, "Psi_{h}[0,0]");
        assert!((m[(1, 1)] - e11).abs() < 1e-14, "Psi_{h}[1,1]");
        assert!(m[(0, 1)].abs() < 1e-14 && m[(1, 0)].abs() < 1e-14, "Psi_{h} off-diagonal");
    }
}

/// FEVD shares are a probability decomposition: every (variable,
/// horizon) row is non-negative and sums to 1 to roundoff.
#[test]
fn fevd_rows_sum_to_one() {
    let fx = load_fixture("var.json");
    let data = as_mat(&fx["data_100dlog_gdp_cons_inv"]);
    let res = VarSpec::new(2, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap();
    let fevd = res.fevd(12).unwrap();
    for (i, m) in fevd.decomp.iter().enumerate() {
        for h in 0..m.nrows() {
            let mut sum = 0.0;
            for j in 0..m.ncols() {
                assert!(m[(h, j)] >= 0.0, "fevd[{i}][({h},{j})] negative");
                sum += m[(h, j)];
            }
            assert!((sum - 1.0).abs() < 1e-12, "fevd[{i}] row {h} sums to {sum}");
        }
    }
}

/// Simulate a stable VAR(1) with constant from seeded Gaussian noise and
/// re-estimate it: the coefficients, intercept, and innovation
/// covariance land near the truth (loose tolerances, fixed seed), and
/// the estimated system is stable.
#[test]
fn simulated_stable_var1_round_trips() {
    let a = [[0.5, 0.1], [-0.2, 0.3]];
    let c = [0.4, -0.2];
    let (n, burn) = (4000usize, 200usize);
    let mut rng = Lcg::new(20260716);
    let mut y = Mat::<f64>::zeros(n, 2);
    let mut prev = [0.0f64, 0.0f64];
    for t in 0..(n + burn) {
        let cur = [
            c[0] + a[0][0] * prev[0] + a[0][1] * prev[1] + rng.gaussian(),
            c[1] + a[1][0] * prev[0] + a[1][1] * prev[1] + rng.gaussian(),
        ];
        if t >= burn {
            y[(t - burn, 0)] = cur[0];
            y[(t - burn, 1)] = cur[1];
        }
        prev = cur;
    }

    let res = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(y.as_ref())
        .unwrap();
    assert!(res.is_stable().unwrap());
    for i in 0..2 {
        assert!(
            (res.intercept[i] - c[i]).abs() < 0.15,
            "intercept[{i}] = {} vs {}",
            res.intercept[i],
            c[i]
        );
        for (j, &a_ij) in a[i].iter().enumerate() {
            assert!(
                (res.coefs[0][(i, j)] - a_ij).abs() < 0.08,
                "A[({i},{j})] = {} vs {a_ij}",
                res.coefs[0][(i, j)]
            );
            let eye = f64::from(u8::from(i == j));
            assert!(
                (res.sigma_u[(i, j)] - eye).abs() < 0.15,
                "sigma_u[({i},{j})] = {} vs {eye}",
                res.sigma_u[(i, j)]
            );
        }
    }
}

/// Validation errors: empty specs, short samples, bad horizons and
/// alphas, and malformed causality index sets are rejected (never
/// panics).
#[test]
fn error_paths() {
    assert!(matches!(
        VarSpec::new(0, Trend::None),
        Err(VarError::InvalidArgument { .. })
    ));

    let short = Mat::<f64>::zeros(5, 3);
    let spec = VarSpec::new(2, Trend::Constant).unwrap();
    assert!(matches!(
        spec.fit(short.as_ref()),
        Err(VarError::InsufficientObservations { .. })
    ));

    let nan = Mat::from_fn(30, 2, |i, j| if (i, j) == (3, 1) { f64::NAN } else { 0.1 });
    assert!(matches!(
        spec.fit(nan.as_ref()),
        Err(VarError::NonFinite { .. })
    ));

    let fx = load_fixture("var.json");
    let data = as_mat(&fx["data_100dlog_gdp_cons_inv"]);
    let res = spec.fit(data.as_ref()).unwrap();
    assert!(matches!(
        res.forecast(0),
        Err(VarError::InvalidArgument { .. })
    ));
    assert!(matches!(
        res.forecast_interval(4, 0.0),
        Err(VarError::InvalidArgument { .. })
    ));
    assert!(matches!(
        res.fevd(0),
        Err(VarError::InvalidArgument { .. })
    ));
    assert!(matches!(
        res.test_causality(&[0], &[]),
        Err(VarError::InvalidArgument { .. })
    ));
    assert!(matches!(
        res.test_causality(&[0], &[1, 1]),
        Err(VarError::InvalidArgument { .. })
    ));
    assert!(matches!(
        res.test_causality(&[0], &[7]),
        Err(VarError::Dimension { .. })
    ));
    assert!(matches!(
        select_order(data.as_ref(), 0, Trend::Constant),
        Err(VarError::InvalidArgument { .. })
    ));
}
