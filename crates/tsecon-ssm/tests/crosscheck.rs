//! Cross-check and invariant tests: univariate filter vs the independent
//! matrix (Joseph-form) filter, smoother consistency, initialization
//! behavior, and builder/filter validation errors.

mod common;

use common::Lcg;
use tsecon_linalg::faer::Mat;
use tsecon_ssm::{Initialization, LinearGaussianSSM, SsmError};

/// A well-conditioned 2-state, 2-observation model with known
/// initialization and diagonal H (so both filter paths accept it).
fn two_state_model() -> LinearGaussianSSM {
    LinearGaussianSSM::builder(2, 2, 2)
        .z(Mat::from_fn(2, 2, |i, j| [[1.0, 0.0], [0.5, 1.0]][i][j]))
        .h(Mat::from_fn(2, 2, |i, j| [[0.4, 0.0], [0.0, 0.9]][i][j]))
        .t(Mat::from_fn(2, 2, |i, j| [[0.7, 0.2], [0.1, 0.5]][i][j]))
        .r(Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 }))
        .q(Mat::from_fn(2, 2, |i, j| [[0.5, 0.1], [0.1, 0.3]][i][j]))
        .obs_intercept(vec![0.3, -0.1])
        .state_intercept(vec![0.05, -0.02])
        .initialization(Initialization::Known {
            a1: vec![0.3, -0.2],
            p1: Mat::from_fn(2, 2, |i, j| [[1.0, 0.2], [0.2, 0.8]][i][j]),
        })
        .build()
        .unwrap()
}

/// Synthetic observations from the seeded LCG (roughly matched to the
/// model's scale; exact distribution is irrelevant for the cross-check).
fn synthetic_obs(n: usize, p: usize, seed: u64) -> Mat<f64> {
    let mut rng = Lcg::new(seed);
    Mat::from_fn(n, p, |_, _| 2.0 * rng.symmetric())
}

/// The univariate (sequential) filter and the matrix (Joseph) filter are
/// algebraically identical on a known-initialization model with diagonal
/// H and complete data: log-likelihood, filtered and predicted moments
/// agree to 1e-10.
#[test]
fn univariate_matches_matrix_filter() {
    let model = two_state_model();
    let y = synthetic_obs(60, 2, 20260716);

    let uni = model.filter(y.as_ref()).unwrap();
    let mat = model.filter_matrix(y.as_ref()).unwrap();

    assert_eq!(uni.d_diffuse, 0);
    let ll_scale = mat.loglik.abs().max(1.0);
    assert!(
        (uni.loglik - mat.loglik).abs() <= 1e-10 * ll_scale,
        "loglik: univariate {} vs matrix {}",
        uni.loglik,
        mat.loglik
    );

    for t in 0..60 {
        for j in 0..2 {
            assert!(
                (uni.filtered_state[t][j] - mat.filtered_state[t][j]).abs() <= 1e-10,
                "filtered_state[{t}][{j}]: {} vs {}",
                uni.filtered_state[t][j],
                mat.filtered_state[t][j]
            );
            assert!(
                (uni.predicted_state[t][j] - mat.predicted_state[t][j]).abs() <= 1e-10,
                "predicted_state[{t}][{j}]"
            );
            for i in 0..2 {
                assert!(
                    (uni.filtered_state_cov[t][(i, j)] - mat.filtered_state_cov[t][(i, j)]).abs()
                        <= 1e-10,
                    "filtered_state_cov[{t}][({i},{j})]"
                );
                assert!(
                    (uni.predicted_state_cov[t][(i, j)] - mat.predicted_state_cov[t][(i, j)]).abs()
                        <= 1e-10,
                    "predicted_state_cov[{t}][({i},{j})]"
                );
            }
        }
    }
}

/// Partially and fully missing periods: the univariate filter (which
/// skips NaN elements one at a time) agrees with the matrix filter
/// (which subsets the observation vector) on likelihood and moments.
#[test]
fn univariate_matches_matrix_filter_with_missing() {
    let model = two_state_model();
    let mut y = synthetic_obs(40, 2, 42);
    y[(5, 0)] = f64::NAN; // partially missing
    y[(11, 1)] = f64::NAN;
    y[(20, 0)] = f64::NAN; // fully missing period
    y[(20, 1)] = f64::NAN;

    let uni = model.filter(y.as_ref()).unwrap();
    let mat = model.filter_matrix(y.as_ref()).unwrap();

    let ll_scale = mat.loglik.abs().max(1.0);
    assert!(
        (uni.loglik - mat.loglik).abs() <= 1e-10 * ll_scale,
        "loglik with missing: {} vs {}",
        uni.loglik,
        mat.loglik
    );
    for t in 0..40 {
        for j in 0..2 {
            assert!(
                (uni.filtered_state[t][j] - mat.filtered_state[t][j]).abs() <= 1e-10,
                "filtered_state[{t}][{j}] with missing"
            );
        }
    }
}

/// At the final period the smoother has no future information, so the
/// smoothed moments must equal the filtered moments.
#[test]
fn smoothed_equals_filtered_at_final_period() {
    let model = two_state_model();
    let y = synthetic_obs(50, 2, 7);
    let so = model.smooth(y.as_ref()).unwrap();
    let last = 49;
    for j in 0..2 {
        assert!(
            (so.smoothed_state[last][j] - so.filter.filtered_state[last][j]).abs() <= 1e-10,
            "smoothed_state[{last}][{j}] vs filtered"
        );
        for i in 0..2 {
            assert!(
                (so.smoothed_state_cov[last][(i, j)] - so.filter.filtered_state_cov[last][(i, j)])
                    .abs()
                    <= 1e-10,
                "smoothed_state_cov[{last}][({i},{j})] vs filtered"
            );
        }
    }
}

/// Stationary initialization solves the discrete Lyapunov equation:
/// P_1 = T P_1 T' + R Q R', and a_1 = (I - T)^{-1} c.
#[test]
fn stationary_initialization_moments() {
    let model = LinearGaussianSSM::ar(&[0.6, -0.2], 1.44, 1.5).unwrap();
    let init = model.initial_state().unwrap();
    assert!(!init.has_diffuse());
    // Unconditional mean of the first state is c / (1 - ar1 - ar2).
    let mean = 1.5 / (1.0 - 0.6 + 0.2);
    assert!((init.a1[0] - mean).abs() < 1e-12, "a1[0] = {}", init.a1[0]);
    // Second (Harvey companion) state: phi2 * mean.
    assert!((init.a1[1] - (-0.2) * mean).abs() < 1e-12);
    // Lyapunov residual P - T P T' - R Q R' = 0.
    let t = model.t().at(0);
    let p1 = &init.p_star;
    let tpt = t * p1.as_ref() * t.transpose();
    for i in 0..2 {
        for j in 0..2 {
            let rqr = if (i, j) == (0, 0) { 1.44 } else { 0.0 };
            let resid = p1[(i, j)] - tpt[(i, j)] - rqr;
            assert!(resid.abs() < 1e-10, "Lyapunov residual [{i},{j}] = {resid}");
        }
    }
}

/// Mixed initialization: a diffuse random-walk state alongside a
/// stationary AR(1) state. The diffuse block gets a unit P_inf entry, the
/// stationary block its Lyapunov variance, and the filter's diffuse
/// period lasts exactly one observation (rank of P_inf is 1).
#[test]
fn mixed_initialization_diffuse_and_stationary_blocks() {
    let model = LinearGaussianSSM::builder(1, 2, 2)
        .z(Mat::from_fn(1, 2, |_, _| 1.0))
        .h(Mat::from_fn(1, 1, |_, _| 0.5))
        .t(Mat::from_fn(2, 2, |i, j| [[1.0, 0.0], [0.0, 0.5]][i][j]))
        .r(Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 }))
        .q(Mat::from_fn(2, 2, |i, j| [[0.2, 0.0], [0.0, 0.3]][i][j]))
        .initialization(Initialization::Mixed {
            diffuse: vec![true, false],
        })
        .build()
        .unwrap();

    let init = model.initial_state().unwrap();
    assert!(init.has_diffuse());
    assert_eq!(init.p_inf[(0, 0)], 1.0);
    assert_eq!(init.p_inf[(1, 1)], 0.0);
    assert_eq!(init.p_star[(0, 0)], 0.0);
    // AR(1) unconditional variance: q / (1 - phi^2) = 0.3 / 0.75.
    assert!((init.p_star[(1, 1)] - 0.4).abs() < 1e-12);

    let y = synthetic_obs(30, 1, 99);
    let out = model.filter(y.as_ref()).unwrap();
    assert_eq!(out.d_diffuse, 1);
    assert!(out.loglik.is_finite());
    // After the diffuse period the diffuse covariance is exactly gone.
    let p_inf_after = &out.predicted_diffuse_state_cov[1];
    for i in 0..2 {
        for j in 0..2 {
            assert!(p_inf_after[(i, j)].abs() < 1e-12);
        }
    }

    // The smoother runs through the mixed diffuse period and the smoothed
    // covariance stays symmetric PSD-ish (diagonal nonnegative).
    let so = model.smooth(y.as_ref()).unwrap();
    for t in 0..30 {
        assert!(so.smoothed_state_cov[t][(0, 0)] >= -1e-12);
        assert!(so.smoothed_state_cov[t][(1, 1)] >= -1e-12);
        assert!(
            (so.smoothed_state_cov[t][(0, 1)] - so.smoothed_state_cov[t][(1, 0)]).abs() < 1e-12
        );
    }
}

/// A model that feeds a diffuse state into a "stationary" one has no
/// valid mixed initialization and must be rejected at filter time.
#[test]
fn mixed_initialization_rejects_diffuse_feedback() {
    let model = LinearGaussianSSM::builder(1, 2, 2)
        .z(Mat::from_fn(1, 2, |_, _| 1.0))
        .h(Mat::from_fn(1, 1, |_, _| 0.5))
        .t(Mat::from_fn(2, 2, |i, j| {
            // T[1][0] != 0: the stationary state is driven by the diffuse one.
            [[1.0, 0.0], [0.4, 0.5]][i][j]
        }))
        .r(Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 }))
        .q(Mat::from_fn(2, 2, |i, j| if i == j { 0.2 } else { 0.0 }))
        .initialization(Initialization::Mixed {
            diffuse: vec![true, false],
        })
        .build()
        .unwrap();
    assert!(matches!(
        model.initial_state(),
        Err(SsmError::InvalidArgument { .. })
    ));
}

/// Builder and filter validation errors surface as the right variants.
#[test]
fn validation_errors() {
    // Missing matrix.
    let err = LinearGaussianSSM::builder(1, 1, 1)
        .z(Mat::from_fn(1, 1, |_, _| 1.0))
        .build()
        .unwrap_err();
    assert!(matches!(err, SsmError::MissingMatrix { .. }));

    // Dimension mismatch.
    let err = LinearGaussianSSM::builder(1, 2, 1)
        .z(Mat::from_fn(1, 1, |_, _| 1.0)) // should be 1 x 2
        .h(Mat::from_fn(1, 1, |_, _| 1.0))
        .t(Mat::from_fn(2, 2, |i, j| if i == j { 0.5 } else { 0.0 }))
        .r(Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 0.0 }))
        .q(Mat::from_fn(1, 1, |_, _| 1.0))
        .build()
        .unwrap_err();
    assert!(matches!(err, SsmError::Dimension { .. }));

    // Indefinite H rejected.
    let err = LinearGaussianSSM::builder(1, 1, 1)
        .z(Mat::from_fn(1, 1, |_, _| 1.0))
        .h(Mat::from_fn(1, 1, |_, _| -1.0))
        .t(Mat::from_fn(1, 1, |_, _| 0.5))
        .r(Mat::from_fn(1, 1, |_, _| 1.0))
        .q(Mat::from_fn(1, 1, |_, _| 1.0))
        .build()
        .unwrap_err();
    assert!(matches!(err, SsmError::NotPsd { what: "H" }));

    // Non-diagonal H: build succeeds, univariate path refuses, matrix
    // path accepts.
    let model = LinearGaussianSSM::builder(2, 1, 1)
        .z(Mat::from_fn(2, 1, |_, _| 1.0))
        .h(Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.3 }))
        .t(Mat::from_fn(1, 1, |_, _| 0.5))
        .r(Mat::from_fn(1, 1, |_, _| 1.0))
        .q(Mat::from_fn(1, 1, |_, _| 1.0))
        .initialization(Initialization::Known {
            a1: vec![0.0],
            p1: Mat::from_fn(1, 1, |_, _| 1.0),
        })
        .build()
        .unwrap();
    assert!(!model.h_is_diagonal());
    let y = synthetic_obs(5, 2, 1);
    assert!(matches!(
        model.filter(y.as_ref()),
        Err(SsmError::NonDiagonalH)
    ));
    assert!(model.filter_matrix(y.as_ref()).is_ok());

    // Matrix filter refuses exact-diffuse initialization.
    let ll = LinearGaussianSSM::local_level(1.0, 0.5).unwrap();
    let y1 = synthetic_obs(5, 1, 2);
    assert!(matches!(
        ll.filter_matrix(y1.as_ref()),
        Err(SsmError::DiffuseNotSupported { .. })
    ));

    // Nonstationary AR under stationary initialization fails at filter
    // time with a wrapped linalg instability error.
    let unit_root = LinearGaussianSSM::ar(&[1.0], 1.0, 0.0).unwrap();
    assert!(matches!(
        unit_root.filter(y1.as_ref()),
        Err(SsmError::Linalg(_))
    ));

    // Infinities in y are rejected (NaN means missing; inf is an error).
    let mut y_bad = synthetic_obs(5, 1, 3);
    y_bad[(2, 0)] = f64::INFINITY;
    assert!(matches!(
        ll.filter(y_bad.as_ref()),
        Err(SsmError::NonFinite { .. })
    ));
}
