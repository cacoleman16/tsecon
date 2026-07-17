//! Golden-value tests against the `statsmodels` fixture
//! (`fixtures/regime.json`,
//! `MarkovAutoregression(k_regimes=2, order=1, switching_ar=False,
//! switching_variance=True)`): the fixed-parameter Hamilton-filter
//! log-likelihood and the filtered and smoothed regime-1 probabilities.

mod common;

use common::{as_f64_vec, assert_abs_close, assert_rel_close, load_fixture};
use tsecon_regime::{MarkovSwitchingAr, MsarParams, MsarSpec};

const SPEC: MsarSpec = MsarSpec {
    k_regimes: 2,
    order: 1,
    switching_ar: false,
    switching_variance: true,
};

/// Builds the fixture's fixed parameters. `params [p00, p10, const0,
/// const1, sigma2_0, sigma2_1, ar1]` with the column-stochastic transition
/// `P[i][j] = P(S_t = i | S_{t-1} = j)`: column 0 is `(p00, 1 - p00)`,
/// column 1 is `(p10, 1 - p10)`.
fn fixed_params() -> MsarParams {
    let p00 = 0.95;
    let p10 = 0.09999999999999998;
    MsarParams::new(
        vec![vec![p00, p10], vec![1.0 - p00, 1.0 - p10]],
        vec![-1.0, 1.5],
        vec![vec![0.5]],
        vec![0.6400000000000001, 1.44],
    )
    .unwrap()
}

/// The Hamilton-filter log-likelihood at the fixed parameters matches
/// `statsmodels` to 1e-6 relative — this pins the switching-mean AR
/// recursion, the switching-variance likelihood, the steady-state
/// initialization, and every constant. (Measured agreement is <= 1e-14.)
#[test]
fn golden_loglike_fixed() {
    let fx = load_fixture("regime.json");
    let y = as_f64_vec(&fx["y"]);
    let model = MarkovSwitchingAr::new(&y, SPEC).unwrap();
    let ll = model.loglike(&fixed_params()).unwrap();
    let expected = fx["loglike_fixed"].as_f64().unwrap();
    assert_rel_close(ll, expected, 1e-6, "loglike_fixed");
}

/// The filtered regime-1 probabilities match the fixture to 1e-6.
#[test]
fn golden_filtered_prob() {
    let fx = load_fixture("regime.json");
    let y = as_f64_vec(&fx["y"]);
    let model = MarkovSwitchingAr::new(&y, SPEC).unwrap();
    let out = model.filter(&fixed_params()).unwrap();
    let expected = as_f64_vec(&fx["filtered_prob_regime1"]);
    assert_eq!(out.filtered_prob.len(), expected.len());
    for (t, &e) in expected.iter().enumerate() {
        assert_abs_close(
            out.filtered_prob[t][1],
            e,
            1e-6,
            &format!("filtered_prob[{t}]"),
        );
    }
}

/// The smoothed regime-1 probabilities from the Kim (1994) smoother match
/// the fixture to 1e-6, and the smoother reproduces the same
/// log-likelihood as the filter.
#[test]
fn golden_smoothed_prob() {
    let fx = load_fixture("regime.json");
    let y = as_f64_vec(&fx["y"]);
    let model = MarkovSwitchingAr::new(&y, SPEC).unwrap();
    let out = model.smooth(&fixed_params()).unwrap();
    let expected = as_f64_vec(&fx["smoothed_prob_regime1"]);
    assert_eq!(out.smoothed_prob.len(), expected.len());
    for (t, &e) in expected.iter().enumerate() {
        assert_abs_close(
            out.smoothed_prob[t][1],
            e,
            1e-6,
            &format!("smoothed_prob[{t}]"),
        );
    }
    assert_rel_close(
        out.loglik,
        fx["loglike_fixed"].as_f64().unwrap(),
        1e-6,
        "smooth loglik",
    );
}

/// EM recovers the fixture parameters: from a perturbed start it reaches a
/// log-likelihood no worse than the fixed-parameter value (up to a small
/// slack) and recovers the two regime means to within ~0.3 (as an
/// unordered set, since regime labels are only identified up to
/// permutation). Local optima mean we assert improvement + rough recovery,
/// not an exact optimum.
#[test]
fn em_parameter_recovery() {
    let fx = load_fixture("regime.json");
    let y = as_f64_vec(&fx["y"]);
    let loglike_fixed = fx["loglike_fixed"].as_f64().unwrap();
    let model = MarkovSwitchingAr::new(&y, SPEC).unwrap();

    // A deliberately off start: symmetric-ish transitions, means pulled in,
    // equal variances.
    let start = MsarParams::new(
        vec![vec![0.8, 0.2], vec![0.2, 0.8]],
        vec![-0.5, 0.7],
        vec![vec![0.3]],
        vec![1.0, 1.0],
    )
    .unwrap();

    let fit = model.fit(&start, 500, 1e-8).unwrap();

    // Monotone EM must at least reach the fixed-parameter likelihood.
    assert!(
        fit.loglik >= loglike_fixed - 1.0,
        "EM loglik {} should reach fixed {} (slack 1.0)",
        fit.loglik,
        loglike_fixed
    );

    // Means recovered as an unordered set within 0.3.
    let means = fit.params.means();
    let (lo, hi) = if means[0] <= means[1] {
        (means[0], means[1])
    } else {
        (means[1], means[0])
    };
    assert_abs_close(lo, -1.0, 0.3, "recovered low mean");
    assert_abs_close(hi, 1.5, 0.3, "recovered high mean");

    // Every smoothed row is a probability distribution.
    for row in &fit.smoothed_prob {
        let s: f64 = row.iter().sum();
        assert_abs_close(s, 1.0, 1e-9, "smoothed row sums to 1");
    }
}
