//! Scale-equivariance of the Markov-switching AR estimator.
//!
//! The Hamilton (1989) switching-mean autoregression is a location-scale
//! model: replacing `y` by `c * y` (`c > 0`) reproduces the same fit with
//! means scaled by `c`, innovation variances by `c^2`, and the AR
//! coefficients, the transition matrix, the smoothed regime probabilities
//! and the convergence verdict left *exactly* where they were. A change of
//! measurement unit — payrolls in thousands versus in persons — must not
//! change the estimate.
//!
//! An absolute ridge or variance floor inside the M-step breaks that: the
//! M-step normal equations for the means carry `1 / sigma^2`, so their
//! entries shrink like `c^-2`, and a fixed `1e-10` added to the diagonal
//! grows in relative terms by `c^2`. These tests sweep several decades,
//! because a single-scale test cannot see this class of bug.

mod common;

use common::SplitMix64;
use tsecon_regime::{FitResult, MarkovSwitchingAr, MsarParams, MsarSpec};

const SPEC: MsarSpec = MsarSpec {
    k_regimes: 2,
    order: 1,
    switching_ar: false,
    switching_variance: true,
};

/// Simulated monthly change in US nonfarm payrolls, in persons: a persistent
/// two-state chain with an expansion mean of +150,000/month (sd 80,000) and
/// a recession mean of -400,000/month (sd 250,000).
fn payrolls_persons(n: usize) -> Vec<f64> {
    let mut rng = SplitMix64(2);
    let mu = [150_000.0_f64, -400_000.0];
    let sd = [80_000.0_f64, 250_000.0];
    let exit = [0.02_f64, 0.10]; // P(leave expansion), P(leave recession)
    let phi = 0.3;

    let mut y = vec![0.0; n];
    let mut s = 0usize;
    y[0] = mu[s] + sd[s] * rng.normal();
    for t in 1..n {
        let prev = s;
        if rng.uniform() < exit[s] {
            s = 1 - s;
        }
        y[t] = mu[s] + phi * (y[t - 1] - mu[prev]) + sd[s] * rng.normal();
    }
    y
}

/// The quantile-based default start used by the `markov_switching_ar`
/// Python entry point, so the sweep exercises the same path a user hits.
/// It is itself equivariant: quantiles scale by `c`, the sample variance by
/// `c^2`, and the AR and transition starts are scale-free.
fn default_start(y: &[f64], k: usize, order: usize) -> MsarParams {
    let mut sorted = y.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let means: Vec<f64> = (0..k)
        .map(|r| sorted[((r as f64 + 0.5) / k as f64 * n as f64) as usize % n])
        .collect();
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let var_y = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / n as f64;
    let transition: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| if i == j { 0.9 } else { 0.1 / (k as f64 - 1.0) })
                .collect()
        })
        .collect();
    MsarParams::new(transition, means, vec![vec![0.1; order]], vec![var_y; k])
        .expect("valid default start")
}

fn fit_scaled(y: &[f64], c: f64) -> FitResult {
    let ys: Vec<f64> = y.iter().map(|v| v * c).collect();
    let model = MarkovSwitchingAr::new(&ys, SPEC).expect("model");
    let start = default_start(&ys, SPEC.k_regimes, SPEC.order);
    model.fit(&start, 500, 1e-8).expect("fit")
}

/// Asserts `|a - e| <= tol * max(|e|, floor)`.
fn close(actual: f64, expected: f64, tol: f64, floor: f64, what: &str, c: f64) {
    let scale = expected.abs().max(floor);
    let rel = (actual - expected).abs() / scale;
    assert!(
        rel <= tol,
        "c = {c:e}: {what}: {actual:e} vs {expected:e} (rel {rel:e} > tol {tol:e})"
    );
}

/// The estimate is equivariant across six decades of measurement unit:
/// means scale by `c`, variances by `c^2`, and the transition matrix, AR
/// coefficients, log-likelihood increment (hence `converged` and
/// `iterations`), and smoothed probabilities do not move at all.
#[test]
fn markov_switching_ar_is_scale_equivariant() {
    let y = payrolls_persons(720);
    let base = fit_scaled(&y, 1.0);

    for &c in &[1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3] {
        let f = fit_scaled(&y, c);

        for (i, (&m, &m0)) in f.params.means().iter().zip(base.params.means()).enumerate() {
            close(m / c, m0, 1e-8, 1.0, &format!("mean[{i}] / c"), c);
        }
        for (i, (&v, &v0)) in f
            .params
            .variances()
            .iter()
            .zip(base.params.variances())
            .enumerate()
        {
            close(
                v / (c * c),
                v0,
                1e-8,
                1.0,
                &format!("variance[{i}] / c^2"),
                c,
            );
        }
        let (p, p0) = (
            f.params.transition_matrix(),
            base.params.transition_matrix(),
        );
        for i in 0..SPEC.k_regimes {
            for j in 0..SPEC.k_regimes {
                close(p[i][j], p0[i][j], 1e-8, 1e-3, &format!("P[{i}][{j}]"), c);
            }
        }
        assert_eq!(
            f.converged, base.converged,
            "c = {c:e}: converged flag must not depend on the measurement unit"
        );
        assert_eq!(
            f.iterations, base.iterations,
            "c = {c:e}: iteration count must not depend on the measurement unit"
        );
        // The log-likelihood shifts by the Jacobian -n ln c and nothing else.
        let n = (y.len() - SPEC.order) as f64;
        close(
            f.loglik + n * c.ln(),
            base.loglik,
            1e-9,
            1.0,
            "loglik + n ln c",
            c,
        );
        for (t, (row, row0)) in f
            .smoothed_prob
            .iter()
            .zip(base.smoothed_prob.iter())
            .enumerate()
        {
            for i in 0..SPEC.k_regimes {
                close(
                    row[i],
                    row0[i],
                    1e-6,
                    1e-3,
                    &format!("smoothed_prob[{t}][{i}]"),
                    c,
                );
            }
        }
    }
}

/// A constant series has no scale for the variance floor to be relative
/// *to* — its sample variance is zero — and it identifies no regimes, so
/// `fit` rejects it rather than returning the arbitrary answer a fixed
/// absolute floor used to manufacture. Scoring fixed parameters on it is
/// still well defined, so `filter` keeps working.
#[test]
fn constant_series_is_rejected_by_fit_but_not_by_filter() {
    let y = vec![3.5_f64; 50];
    let model = MarkovSwitchingAr::new(&y, SPEC).expect("model");
    let params = MsarParams::new(
        vec![vec![0.9, 0.1], vec![0.1, 0.9]],
        vec![3.0, 4.0],
        vec![vec![0.2]],
        vec![1.0, 1.0],
    )
    .expect("params");

    assert!(
        model.filter(&params).expect("filter").loglik.is_finite(),
        "a constant series still has a well-defined fixed-parameter likelihood"
    );
    let err = model.fit(&params, 100, 1e-8).expect_err("fit must reject");
    assert!(
        format!("{err}").contains("positive sample variance"),
        "unexpected error: {err}"
    );
}

/// The same sweep on a small, well-separated series with a *shared*
/// variance and no switching variance block, so the shared-variance floor
/// and the pooled AR normal equations are exercised too.
///
/// The sweep reaches down to `c = 1e-7`, where the innovation variance is
/// around `5e-15` and an absolute `1e-12` floor on it would bind — the
/// estimate would stop scaling with the data entirely.
#[test]
fn shared_variance_msar_is_scale_equivariant() {
    const SHARED: MsarSpec = MsarSpec {
        k_regimes: 2,
        order: 2,
        switching_ar: false,
        switching_variance: false,
    };
    let mut rng = SplitMix64(0x5EED_5CA1_E000_0001);
    let n = 400usize;
    let mu = [-2.0_f64, 3.0];
    let mut y = vec![0.0; n];
    let mut s = 0usize;
    y[0] = mu[s] + rng.normal();
    y[1] = mu[s] + rng.normal();
    for t in 2..n {
        let prev = s;
        if rng.uniform() > 0.95 {
            s = 1 - s;
        }
        y[t] =
            mu[s] + 0.4 * (y[t - 1] - mu[prev]) - 0.2 * (y[t - 2] - mu[prev]) + 0.7 * rng.normal();
    }

    let fit_at = |c: f64| {
        let ys: Vec<f64> = y.iter().map(|v| v * c).collect();
        let model = MarkovSwitchingAr::new(&ys, SHARED).expect("model");
        let mut start = default_start(&ys, SHARED.k_regimes, SHARED.order);
        // Shared variance: one block.
        start = MsarParams::new(
            start.transition_matrix(),
            start.means().to_vec(),
            vec![vec![0.1; SHARED.order]],
            vec![start.variances()[0]],
        )
        .expect("shared start");
        model.fit(&start, 500, 1e-9).expect("fit")
    };

    let base = fit_at(1.0);
    for &c in &[1e-7, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4] {
        let f = fit_at(c);
        for (i, (&m, &m0)) in f.params.means().iter().zip(base.params.means()).enumerate() {
            close(m / c, m0, 1e-8, 1.0, &format!("shared mean[{i}] / c"), c);
        }
        close(
            f.params.variances()[0] / (c * c),
            base.params.variances()[0],
            1e-8,
            1.0,
            "shared variance / c^2",
            c,
        );
        let (p, p0) = (
            f.params.transition_matrix(),
            base.params.transition_matrix(),
        );
        for i in 0..SHARED.k_regimes {
            for j in 0..SHARED.k_regimes {
                close(p[i][j], p0[i][j], 1e-8, 1e-3, &format!("P[{i}][{j}]"), c);
            }
        }
        assert_eq!(f.converged, base.converged, "c = {c:e}: converged flag");
        assert_eq!(f.iterations, base.iterations, "c = {c:e}: iteration count");
    }
}
