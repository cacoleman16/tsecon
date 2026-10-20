//! Golden and property tests for the fixed-parameter DFM smoother
//! (`smooth_fixed`, `smooth_standardized`) against `fixtures/dfm_smooth.json`.
//!
//! * `dynamic_factor_r2_p2` — statsmodels `DynamicFactor(k_factors=2,
//!   factor_order=2, error_order=0).smooth(params)` on a ragged panel:
//!   log-likelihood, smoothed states and smoothed state covariances.
//! * `kalman_smoother_q` — the same panel with a non-identity
//!   factor-innovation covariance (statsmodels' generic `KalmanSmoother`).
//! * `standardized_r1_p3` — `smooth_standardized` with non-trivial
//!   `center`/`scale`: the standardized-panel smoother output, the edge factor
//!   and the de-standardized edge nowcast.
//!
//! * `singular_cases` — zero idiosyncratic variances observed together (a
//!   singular observation density): the log-likelihood is the density on
//!   the support, the value with the redundant column set to NaN, against
//!   statsmodels on that reduced panel and an exact (rational / 80-digit
//!   decimal) Kalman filter, under several column orders.
//!
//! statsmodels' filter and the crate's univariate sequential filter agree to
//! roundoff, not bitwise; the gates are 1e-8 relative on the log-likelihood
//! and 1e-8 absolute on the moments (achieved over the three pinned cases:
//! log-likelihood 5.2e-13 relative, states 2.6e-10 and covariances 1.2e-10
//! absolute, both on the standardized panel; see
//! `achieved_agreement_is_far_inside_the_gates`).

mod common;

use common::{as_mat, as_mat_nan, as_vec, load_fixture};
use serde_json::Value;
use tsecon_linalg::faer::Mat;
use tsecon_nowcast::{smooth_fixed, smooth_standardized, DfmParams, NowcastError, Nowcaster};

const FIXTURE: &str = "dfm_smooth.json";

fn case(name: &str) -> Value {
    let fx = load_fixture(FIXTURE);
    fx["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("case {name} missing"))
        .clone()
}

fn params_of(c: &Value) -> DfmParams {
    DfmParams {
        loadings: as_mat(&c["loadings"]),
        factor_ar: as_mat(&c["factor_ar"]),
        factor_cov: as_mat(&c["factor_cov"]),
        idiosyncratic: as_vec(&c["idiosyncratic"]),
    }
}

/// Max deviations of the smoother output from the case's reference:
/// (relative loglik, abs state, abs state cov).
fn deviations(c: &Value, loglik: f64, state: &[Vec<f64>], cov: &[Mat<f64>]) -> (f64, f64, f64) {
    let ll = c["loglik"].as_f64().unwrap();
    let dev_ll = (loglik - ll).abs() / ll.abs().max(1.0);
    let ref_state = as_mat(&c["smoothed_state"]);
    assert_eq!(state.len(), ref_state.nrows());
    let mut dev_s = 0.0f64;
    for (t, row) in state.iter().enumerate() {
        assert_eq!(row.len(), ref_state.ncols());
        for (j, v) in row.iter().enumerate() {
            dev_s = dev_s.max((v - ref_state[(t, j)]).abs());
        }
    }
    let ref_cov = c["smoothed_state_cov"].as_array().unwrap();
    assert_eq!(cov.len(), ref_cov.len());
    let mut dev_c = 0.0f64;
    for (t, m) in cov.iter().enumerate() {
        let r = as_mat(&ref_cov[t]);
        assert_eq!((m.nrows(), m.ncols()), (r.nrows(), r.ncols()));
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                dev_c = dev_c.max((m[(i, j)] - r[(i, j)]).abs());
            }
        }
    }
    (dev_ll, dev_s, dev_c)
}

fn assert_case_raw(name: &str) -> (f64, f64, f64) {
    let c = case(name);
    let params = params_of(&c);
    let y = as_mat_nan(&c["panel"]);
    let out = smooth_fixed(&params, y.as_ref()).unwrap();
    let dev = deviations(&c, out.loglik, &out.smoothed_state, &out.smoothed_state_cov);
    assert!(dev.0 <= 1e-8, "{name} loglik rel dev {:e}", dev.0);
    assert!(dev.1 <= 1e-8, "{name} smoothed_state abs dev {:e}", dev.1);
    assert!(
        dev.2 <= 1e-8,
        "{name} smoothed_state_cov abs dev {:e}",
        dev.2
    );
    // The factors are the leading r entries of each smoothed state.
    let r = params.n_factors();
    for (f, s) in out.smoothed_factors.iter().zip(&out.smoothed_state) {
        assert_eq!(f.as_slice(), &s[..r]);
    }
    dev
}

#[test]
fn dynamic_factor_two_factors_var2_ragged_matches_statsmodels() {
    assert_case_raw("dynamic_factor_r2_p2");
}

#[test]
fn non_identity_factor_cov_matches_generic_kalman_smoother() {
    assert_case_raw("kalman_smoother_q");
}

#[test]
fn standardized_smoother_and_edge_nowcast_match_reference() {
    let c = case("standardized_r1_p3");
    let params = params_of(&c);
    let x = as_mat_nan(&c["panel"]);
    let center = as_vec(&c["center"]);
    let scale = as_vec(&c["scale"]);
    let res = smooth_standardized(&params, x.as_ref(), &center, &scale).unwrap();
    let sm = &res.smoothing;
    let dev = deviations(&c, sm.loglik, &sm.smoothed_state, &sm.smoothed_state_cov);
    assert!(dev.0 <= 1e-8, "loglik rel dev {:e}", dev.0);
    assert!(dev.1 <= 1e-8, "state abs dev {:e}", dev.1);
    assert!(dev.2 <= 1e-8, "state cov abs dev {:e}", dev.2);
    let edge = as_vec(&c["edge_factor"]);
    for (a, e) in res.edge_factor.iter().zip(&edge) {
        assert!((a - e).abs() <= 1e-8, "edge factor {a} vs {e}");
    }
    let nowcast = as_vec(&c["nowcast"]);
    for (i, (a, e)) in res.values.iter().zip(&nowcast).enumerate() {
        assert!(
            (a - e).abs() <= 1e-8 * e.abs().max(1.0),
            "nowcast[{i}] {a} vs {e}"
        );
    }
    // The edge factor is the last smoothed factor, and the nowcast is the
    // documented de-standardization of it (same operations, bitwise).
    assert_eq!(&res.edge_factor, sm.smoothed_factors.last().unwrap());
    for i in 0..center.len() {
        let mut acc = 0.0;
        for (k, f) in res.edge_factor.iter().enumerate() {
            acc += params.loadings[(i, k)] * f;
        }
        assert_eq!(
            res.values[i].to_bits(),
            (center[i] + scale[i] * acc).to_bits()
        );
    }
}

#[test]
fn achieved_agreement_is_far_inside_the_gates() {
    // Measured on the committed fixture (x86-64): loglik 1.7e-13 / 5.2e-13 /
    // 3.5e-13 relative, smoothed states 6.0e-11 / 2.8e-11 / 2.6e-10 and
    // their covariances 1.7e-11 / 8.6e-12 / 1.2e-10 absolute for
    // dynamic_factor_r2_p2 / kalman_smoother_q / standardized_r1_p3 — the
    // worst over the three, which the docs quote, is the standardized case.
    // A drift towards the 1e-8 gates would mean a real change in the
    // recursion, not roundoff.
    let mut devs = Vec::new();
    for name in ["dynamic_factor_r2_p2", "kalman_smoother_q"] {
        devs.push((name, assert_case_raw(name)));
    }
    let c = case("standardized_r1_p3");
    let x = as_mat_nan(&c["panel"]);
    let res = smooth_standardized(
        &params_of(&c),
        x.as_ref(),
        &as_vec(&c["center"]),
        &as_vec(&c["scale"]),
    )
    .unwrap();
    let sm = &res.smoothing;
    let dev = deviations(&c, sm.loglik, &sm.smoothed_state, &sm.smoothed_state_cov);
    devs.push(("standardized_r1_p3", dev));
    for (name, (ll, s, cv)) in devs {
        eprintln!("{name}: loglik rel {ll:e}, state abs {s:e}, state cov abs {cv:e}");
        assert!(
            ll < 1e-11 && s < 1e-9 && cv < 1e-9,
            "{name}: {ll:e} {s:e} {cv:e}"
        );
    }
}

/// The singular-density cases of `singular_cases`, one per column order:
/// the case's parameters and panel with the columns in `order`, the index
/// (in the permuted panel) of the redundant column, and the order's entry.
fn singular_orders(name: &str) -> Vec<(DfmParams, Mat<f64>, usize, Value)> {
    let fx = load_fixture(FIXTURE);
    let c = fx["singular_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("singular case {name} missing"))
        .clone();
    let params = params_of(&c);
    let y = as_mat_nan(&c["panel"]);
    c["orders"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| {
            let order: Vec<usize> = o["order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let p = DfmParams {
                loadings: Mat::from_fn(order.len(), params.n_factors(), |i, j| {
                    params.loadings[(order[i], j)]
                }),
                factor_ar: params.factor_ar.clone(),
                factor_cov: params.factor_cov.clone(),
                idiosyncratic: order.iter().map(|&k| params.idiosyncratic[k]).collect(),
            };
            let yo = Mat::from_fn(y.nrows(), order.len(), |t, i| y[(t, order[i])]);
            let redundant = o["redundant"].as_u64().unwrap() as usize;
            (p, yo, redundant, o.clone())
        })
        .collect()
}

/// **Zero idiosyncratic variances observed together.** When more than `r`
/// series with zero idiosyncratic variance are observed in one period, the
/// first pins the factor and the others are redundant: the log-likelihood
/// is the density on the support, the value with the redundant column set
/// to NaN — pinned against statsmodels on that reduced panel (1e-10) and an
/// exact rational / 80-digit decimal Kalman filter (1e-12), and bitwise
/// against the crate's own smoother on the reduced panel (a skipped element
/// leaves the recursion untouched). Before the roundoff test on `F_*`
/// (tsecon-ssm `TOLERANCE_CANCEL`) the redundant element's dust variance
/// entered the sum: the two-period case returned `+18.657` against
/// `-1.737`, and orders `[0, 1, 2, 3]`, `[0, 1, 3, 2]`, `[0, 3, 1, 2]` and
/// `[0, 3, 2, 1]` of the T = 100 panel `-275.617` / `-336.902` against
/// `-293.151` / `-355.055`.
#[test]
fn zero_variance_series_observed_together_give_the_density_on_the_support() {
    for name in ["zero_idio_two_periods", "zero_idio_pair_t100"] {
        for (params, y, redundant, o) in singular_orders(name) {
            let order = &o["order"];
            let full = smooth_fixed(&params, y.as_ref()).unwrap();
            let mut yr = y.clone();
            for t in 0..yr.nrows() {
                yr[(t, redundant)] = f64::NAN;
            }
            let reduced = smooth_fixed(&params, yr.as_ref()).unwrap();
            assert_eq!(
                full.loglik.to_bits(),
                reduced.loglik.to_bits(),
                "{name} {order}: {} vs {} with the redundant column NaN",
                full.loglik,
                reduced.loglik
            );
            assert_eq!(
                full.smoothed_state, reduced.smoothed_state,
                "{name} {order}"
            );
            let ll_sm = o["loglik"].as_f64().unwrap();
            let ll_exact = o["loglik_exact"].as_f64().unwrap();
            let rel_sm = (full.loglik - ll_sm).abs() / ll_sm.abs();
            let rel_exact = (full.loglik - ll_exact).abs() / ll_exact.abs();
            assert!(
                rel_sm <= 1e-10,
                "{name} {order}: loglik {} vs statsmodels {ll_sm}",
                full.loglik
            );
            assert!(
                rel_exact <= 1e-12,
                "{name} {order}: loglik {} vs exact {ll_exact}",
                full.loglik
            );
            let ref_state = as_mat(&o["smoothed_state"]);
            for (t, row) in full.smoothed_state.iter().enumerate() {
                for (j, v) in row.iter().enumerate() {
                    assert!(
                        (v - ref_state[(t, j)]).abs() <= 1e-9,
                        "{name} {order}: state[{t}][{j}]"
                    );
                }
            }
            let ref_cov = o["smoothed_state_cov"].as_array().unwrap();
            for (t, m) in full.smoothed_state_cov.iter().enumerate() {
                let r = as_mat(&ref_cov[t]);
                for i in 0..m.nrows() {
                    for j in 0..m.ncols() {
                        assert!(
                            (m[(i, j)] - r[(i, j)]).abs() <= 1e-9,
                            "{name} {order}: cov[{t}]"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn unit_moments_reduce_to_smooth_fixed_bitwise() {
    let c = case("dynamic_factor_r2_p2");
    let params = params_of(&c);
    let y = as_mat_nan(&c["panel"]);
    let n = params.n_series();
    let a = smooth_fixed(&params, y.as_ref()).unwrap();
    let b = smooth_standardized(&params, y.as_ref(), &vec![0.0; n], &vec![1.0; n]).unwrap();
    assert_eq!(a.loglik.to_bits(), b.smoothing.loglik.to_bits());
    for (sa, sb) in a.smoothed_state.iter().zip(&b.smoothing.smoothed_state) {
        for (u, v) in sa.iter().zip(sb) {
            assert_eq!(u.to_bits(), v.to_bits());
        }
    }
    for (ma, mb) in a
        .smoothed_state_cov
        .iter()
        .zip(&b.smoothing.smoothed_state_cov)
    {
        for i in 0..ma.nrows() {
            for j in 0..ma.ncols() {
                assert_eq!(ma[(i, j)].to_bits(), mb[(i, j)].to_bits());
            }
        }
    }
}

#[test]
fn nowcast_panel_is_smooth_standardized_at_the_fitted_model_bitwise() {
    // Both estimation routes: the round trip a vintage re-smooth relies on.
    let c = case("dynamic_factor_r2_p2");
    let y = as_mat_nan(&c["panel"]);
    // The balanced training block ends before the interior gap at row 70.
    let train = Mat::from_fn(70, y.ncols(), |i, j| y[(i, j)]);
    let fits = [
        Nowcaster::fit_two_step(train.as_ref(), 2, 2).unwrap(),
        Nowcaster::fit_mle(train.as_ref(), 1).unwrap(),
    ];
    for nc in &fits {
        let a = nc.nowcast_panel(y.as_ref()).unwrap();
        let b = smooth_standardized(nc.params(), y.as_ref(), nc.center(), nc.scale()).unwrap();
        assert_eq!(a.smoothing.loglik.to_bits(), b.smoothing.loglik.to_bits());
        assert_eq!(a.edge_factor, b.edge_factor);
        for (u, v) in a.values.iter().zip(&b.values) {
            assert_eq!(u.to_bits(), v.to_bits());
        }
    }
}

#[test]
fn smoothed_state_cov_is_symmetric_psd_and_shrinks_with_data() {
    let c = case("kalman_smoother_q");
    let params = params_of(&c);
    let y = as_mat_nan(&c["panel"]);
    let out = smooth_fixed(&params, y.as_ref()).unwrap();
    for (t, m) in out.smoothed_state_cov.iter().enumerate() {
        for i in 0..m.nrows() {
            assert!(m[(i, i)] >= 0.0, "t={t}: negative variance {}", m[(i, i)]);
            for j in 0..i {
                let scale = m[(i, i)].abs().max(m[(j, j)].abs()).max(1e-300);
                assert!(
                    (m[(i, j)] - m[(j, i)]).abs() <= 1e-12 * scale,
                    "t={t}: asymmetric ({i},{j})"
                );
                // Cauchy-Schwarz on every 2x2 principal minor.
                assert!(m[(i, j)] * m[(i, j)] <= m[(i, i)] * m[(j, j)] * (1.0 + 1e-10));
            }
        }
    }
    // A period with every series observed is more precisely smoothed than
    // the half-released last period (same model, fewer observations there).
    let r = params.n_factors();
    let tt = out.smoothed_state_cov.len();
    for k in 0..r {
        let full = out.smoothed_state_cov[tt - 5][(k, k)];
        let edge = out.smoothed_state_cov[tt - 1][(k, k)];
        assert!(
            edge > full,
            "factor {k}: edge var {edge} <= interior var {full}"
        );
    }
}

#[test]
fn affine_rescaling_of_the_panel_with_its_moments_is_invariant() {
    // Rescaling the raw panel and its moments together leaves z, hence the
    // whole smoother output, unchanged up to the roundoff of forming z.
    let c = case("standardized_r1_p3");
    let params = params_of(&c);
    let x = as_mat_nan(&c["panel"]);
    let center = as_vec(&c["center"]);
    let scale = as_vec(&c["scale"]);
    let base = smooth_standardized(&params, x.as_ref(), &center, &scale).unwrap();
    let (a, b) = (-7.25, 4.0); // b a power of two: z is bit-identical
    let x2 = Mat::from_fn(x.nrows(), x.ncols(), |i, j| a + b * x[(i, j)]);
    let center2: Vec<f64> = center.iter().map(|v| a + b * v).collect();
    let scale2: Vec<f64> = scale.iter().map(|v| b * v).collect();
    let res = smooth_standardized(&params, x2.as_ref(), &center2, &scale2).unwrap();
    assert!((res.smoothing.loglik - base.smoothing.loglik).abs() < 1e-9);
    for (u, v) in res.edge_factor.iter().zip(&base.edge_factor) {
        assert!((u - v).abs() < 1e-9);
    }
    for (i, (u, v)) in res.values.iter().zip(&base.values).enumerate() {
        assert!(
            ((u - a) / b - v).abs() < 1e-9 * v.abs().max(1.0),
            "nowcast[{i}]"
        );
    }
}

#[test]
fn malformed_moments_are_refused_by_name() {
    let c = case("standardized_r1_p3");
    let params = params_of(&c);
    let x = as_mat_nan(&c["panel"]);
    let center = as_vec(&c["center"]);
    let scale = as_vec(&c["scale"]);
    let n = center.len();
    let msg = |e: NowcastError| e.to_string();
    let e = smooth_standardized(&params, x.as_ref(), &center[..n - 1], &scale).unwrap_err();
    assert!(msg(e).contains("center"));
    let e = smooth_standardized(&params, x.as_ref(), &center, &scale[..n - 1]).unwrap_err();
    assert!(msg(e).contains("scale"));
    let mut bad = center.clone();
    bad[1] = f64::NAN;
    let e = smooth_standardized(&params, x.as_ref(), &bad, &scale).unwrap_err();
    assert!(msg(e).contains("center"));
    for v in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        let mut bad = scale.clone();
        bad[2] = v;
        let e = smooth_standardized(&params, x.as_ref(), &center, &bad).unwrap_err();
        assert!(msg(e).contains("scale"), "scale = {v}");
    }
    let empty = Mat::<f64>::zeros(0, n);
    assert!(matches!(
        smooth_standardized(&params, empty.as_ref(), &center, &scale),
        Err(NowcastError::EmptyInput { .. })
    ));
    let wide = Mat::<f64>::zeros(4, n + 1);
    assert!(matches!(
        smooth_standardized(&params, wide.as_ref(), &center, &scale),
        Err(NowcastError::DimensionMismatch { .. })
    ));
}
