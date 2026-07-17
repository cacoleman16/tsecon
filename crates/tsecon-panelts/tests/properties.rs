//! Property / behavioural tests for the heterogeneous-panel estimators — the
//! statistical invariances that no single golden value pins:
//!
//! * **the raison d'être of CCE:** under a common factor loaded into both `y`
//!   and `x`, plain mean group is materially biased for the true mean slope
//!   while CCE-MG is close (Pesaran 2006);
//! * with no common factor, MG recovers the true mean slope and CCE-MG agrees;
//! * `pvalues` / `conf_int` are internally consistent;
//! * the validation layer rejects malformed panels.

use tsecon_panelts::{cce_mean_group, mean_group, PanelTsError, PanelUnit};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

fn gaussian(stream: &mut Stream) -> f64 {
    let u = stream.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// Simulate a heterogeneous panel `y_it = a_i + b_i x_it + gamma_i f_t + e_it`
/// with `x_it = mu_i + delta_i f_t + v_it`. When `factor_on` is false the
/// common factor is switched off, so plain MG is unbiased.
fn simulate(seed: u64, n: usize, t: usize, b_mean: f64, factor_on: bool) -> Vec<PanelUnit> {
    let mut s = Stream::new(seed);
    // One common factor path shared by every unit.
    let f: Vec<f64> = (0..t).map(|_| gaussian(&mut s)).collect();

    (0..n)
        .map(|_| {
            let a_i = 0.5 + gaussian(&mut s);
            let b_i = b_mean + 0.30 * gaussian(&mut s); // unit-specific slope
            let gamma_i = if factor_on {
                1.0 + 0.5 * gaussian(&mut s)
            } else {
                0.0
            };
            let delta_i = if factor_on {
                0.7 + 0.3 * gaussian(&mut s)
            } else {
                0.0
            };
            let mu_i = gaussian(&mut s);

            let mut x = vec![0.0_f64; t];
            let mut y = vec![0.0_f64; t];
            for tt in 0..t {
                let v = gaussian(&mut s);
                let e = 0.6 * gaussian(&mut s);
                let x_it = mu_i + delta_i * f[tt] + v;
                x[tt] = x_it;
                y[tt] = a_i + b_i * x_it + gamma_i * f[tt] + e;
            }
            PanelUnit::new(y, vec![x])
        })
        .collect()
}

#[test]
fn cce_beats_mg_under_common_factor() {
    let b_mean = 1.0;
    let units = simulate(2026_0717, 60, 80, b_mean, true);

    let mg = mean_group(&units).expect("MG fits");
    let cce = cce_mean_group(&units).expect("CCE-MG fits");

    let mg_bias = (mg.coef[0] - b_mean).abs();
    let cce_bias = (cce.coef[0] - b_mean).abs();

    // The factor induces a large MG bias (~0.4+); CCE removes most of it.
    assert!(
        mg_bias > 0.20,
        "expected a sizeable MG bias under the factor, got {mg_bias:e}"
    );
    assert!(
        cce_bias < 0.15,
        "CCE-MG should be close to the truth, got bias {cce_bias:e}"
    );
    assert!(
        cce_bias < 0.4 * mg_bias,
        "CCE bias {cce_bias:e} should be far below MG bias {mg_bias:e}"
    );
}

#[test]
fn mg_unbiased_without_factor() {
    let b_mean = 1.0;
    let units = simulate(99, 80, 90, b_mean, false);

    let mg = mean_group(&units).expect("MG fits");
    let cce = cce_mean_group(&units).expect("CCE-MG fits");

    // With no common factor plain MG is already consistent.
    assert!((mg.coef[0] - b_mean).abs() < 0.08, "MG coef {}", mg.coef[0]);
    // Augmenting with (now-irrelevant) cross-section averages leaves it close.
    assert!(
        (cce.coef[0] - b_mean).abs() < 0.12,
        "CCE coef {}",
        cce.coef[0]
    );
}

#[test]
fn multiple_regressors_and_inference_are_consistent() {
    // Two regressors, small panel, factor on: just check internal consistency.
    let mut s = Stream::new(7);
    let (n, t) = (25usize, 60usize);
    let f: Vec<f64> = (0..t).map(|_| gaussian(&mut s)).collect();
    let units: Vec<PanelUnit> = (0..n)
        .map(|_| {
            let b = [1.5 + 0.3 * gaussian(&mut s), -0.8 + 0.25 * gaussian(&mut s)];
            let g = 1.0 + 0.5 * gaussian(&mut s);
            let d = [0.7 + 0.3 * gaussian(&mut s), 0.4 + 0.3 * gaussian(&mut s)];
            let mu = [gaussian(&mut s), gaussian(&mut s)];
            let mut x0 = vec![0.0; t];
            let mut x1 = vec![0.0; t];
            let mut y = vec![0.0; t];
            for tt in 0..t {
                x0[tt] = mu[0] + d[0] * f[tt] + gaussian(&mut s);
                x1[tt] = mu[1] + d[1] * f[tt] + gaussian(&mut s);
                y[tt] = 0.5 + b[0] * x0[tt] + b[1] * x1[tt] + g * f[tt] + 0.6 * gaussian(&mut s);
            }
            PanelUnit::new(y, vec![x0, x1])
        })
        .collect();

    let fit = cce_mean_group(&units).expect("CCE-MG fits");
    assert_eq!(fit.k, 2);
    assert_eq!(fit.n_units, n);
    assert_eq!(fit.coef_per_unit.len(), n);

    // t = coef / se exactly.
    for j in 0..2 {
        assert!((fit.tstat[j] - fit.coef[j] / fit.se[j]).abs() < 1e-12);
        assert!(fit.se[j] > 0.0);
    }

    // p-values in [0, 1]; a wider CI covers the point estimate and is nested.
    let p = fit.pvalues().expect("pvalues");
    let ci95 = fit.conf_int(0.95).expect("ci95");
    let ci99 = fit.conf_int(0.99).expect("ci99");
    for j in 0..2 {
        assert!((0.0..=1.0).contains(&p[j]));
        assert!(ci95[j].0 <= fit.coef[j] && fit.coef[j] <= ci95[j].1);
        assert!(ci99[j].0 <= ci95[j].0 && ci95[j].1 <= ci99[j].1); // 99% ⊇ 95%
    }
}

#[test]
fn rejects_too_few_units() {
    let u = PanelUnit::new(vec![1.0, 2.0, 3.0], vec![vec![0.1, 0.2, 0.3]]);
    assert!(matches!(
        mean_group(std::slice::from_ref(&u)),
        Err(PanelTsError::TooFewUnits { n: 1 })
    ));
    assert!(matches!(
        mean_group(&[]),
        Err(PanelTsError::TooFewUnits { n: 0 })
    ));
}

#[test]
fn rejects_no_regressors() {
    let units = vec![
        PanelUnit::new(vec![1.0, 2.0, 3.0], vec![]),
        PanelUnit::new(vec![1.0, 2.0, 3.0], vec![]),
    ];
    assert!(matches!(
        mean_group(&units),
        Err(PanelTsError::NoRegressors { unit: 0 })
    ));
}

#[test]
fn rejects_inconsistent_regressor_count() {
    let units = vec![
        PanelUnit::new(vec![1.0, 2.0, 3.0, 4.0], vec![vec![0.1, 0.2, 0.3, 0.4]]),
        PanelUnit::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]],
        ),
    ];
    assert!(matches!(
        mean_group(&units),
        Err(PanelTsError::InconsistentRegressors {
            unit: 1,
            expected: 1,
            got: 2
        })
    ));
}

#[test]
fn rejects_ragged_unit() {
    let units = vec![
        PanelUnit::new(vec![1.0, 2.0, 3.0, 4.0], vec![vec![0.1, 0.2, 0.3, 0.4]]),
        PanelUnit::new(vec![1.0, 2.0, 3.0, 4.0], vec![vec![0.1, 0.2, 0.3]]),
    ];
    assert!(matches!(
        mean_group(&units),
        Err(PanelTsError::RaggedUnit {
            unit: 1,
            column: 0,
            expected: 4,
            got: 3
        })
    ));
}

#[test]
fn cce_rejects_unbalanced_panel() {
    let units = vec![
        PanelUnit::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]],
        ),
        PanelUnit::new(vec![1.0, 2.0, 3.0, 4.0], vec![vec![0.1, 0.2, 0.3, 0.4]]),
    ];
    assert!(matches!(
        cce_mean_group(&units),
        Err(PanelTsError::UnbalancedPanel {
            unit: 1,
            expected: 5,
            got: 4
        })
    ));
}

#[test]
fn propagates_ols_failure() {
    // Collinear design (a constant regressor duplicates the internal intercept)
    // makes the per-unit OLS singular; the error is wrapped with the unit index.
    let col = vec![2.0_f64; 6];
    let units = vec![
        PanelUnit::new(vec![1.0, 2.0, 1.5, 2.5, 1.0, 3.0], vec![col.clone()]),
        PanelUnit::new(vec![1.0, 2.0, 1.5, 2.5, 1.0, 3.0], vec![col]),
    ];
    assert!(matches!(
        mean_group(&units),
        Err(PanelTsError::Ols { unit: 0, .. })
    ));
}
