//! Property tests for the Markov-switching machinery: transition rows sum
//! to one, two identical regimes give ~0.5 smoothed probabilities, and a
//! strongly separated two-regime simulation is classified accurately.

mod common;

use common::{assert_abs_close, SplitMix64};
use tsecon_regime::{classify, MarkovSwitchingAr, MsarParams, MsarSpec};

const SPEC: MsarSpec = MsarSpec {
    k_regimes: 2,
    order: 1,
    switching_ar: false,
    switching_variance: true,
};

/// Every column of the estimated transition matrix is a probability
/// distribution (non-negative, sums to one): the M-step normalizes expected
/// transition counts by destination-regime occupancy.
#[test]
fn estimated_transition_columns_are_stochastic() {
    let mut rng = SplitMix64(0xC0FF_EE12_3456_789A);
    let y: Vec<f64> = (0..300).map(|_| rng.normal()).collect();
    let model = MarkovSwitchingAr::new(&y, SPEC).unwrap();
    let start = MsarParams::new(
        vec![vec![0.7, 0.3], vec![0.3, 0.7]],
        vec![-0.5, 0.5],
        vec![vec![0.1]],
        vec![1.0, 1.0],
    )
    .unwrap();
    let fit = model.fit(&start, 100, 1e-8).unwrap();
    let p = fit.params.transition_matrix();
    let k = 2;
    for j in 0..k {
        let mut col = 0.0;
        for row in p.iter().take(k) {
            assert!(row[j] >= -1e-12, "transition prob must be non-negative");
            col += row[j];
        }
        assert_abs_close(col, 1.0, 1e-9, "transition column sums to 1");
    }
}

/// When the two regimes are made identical (equal means, AR, and variances)
/// under a symmetric transition matrix, the observations carry no
/// information to separate them, so every smoothed probability sits at the
/// stationary 0.5.
#[test]
fn identical_regimes_give_half_probabilities() {
    let mut rng = SplitMix64(0x1234_5678_9ABC_DEF0);
    let y: Vec<f64> = (0..200).map(|_| 0.5 * rng.normal()).collect();
    let model = MarkovSwitchingAr::new(&y, SPEC).unwrap();
    // Identical regimes, symmetric transition.
    let params = MsarParams::new(
        vec![vec![0.7, 0.3], vec![0.3, 0.7]],
        vec![0.2, 0.2],
        vec![vec![0.4]],
        vec![0.8, 0.8],
    )
    .unwrap();
    let out = model.smooth(&params).unwrap();
    for row in &out.smoothed_prob {
        assert_abs_close(row[0], 0.5, 1e-9, "identical-regime smoothed prob");
        assert_abs_close(row[1], 0.5, 1e-9, "identical-regime smoothed prob");
    }
    // Filtered probabilities collapse to 0.5 as well.
    for row in &out.filtered_prob {
        assert_abs_close(row[1], 0.5, 1e-9, "identical-regime filtered prob");
    }
}

/// A strongly separated two-regime simulation (means far apart, small
/// variances, persistent chain) is recovered by EM and classified with high
/// accuracy against the true regime path.
#[test]
fn separated_regimes_classified_accurately() {
    let mut rng = SplitMix64(0xBADC_0FFE_E0DD_F00D);
    let n = 600usize;
    let phi = 0.3;
    let mu = [-6.0, 6.0];
    let sd = [0.4_f64, 0.4];
    // Persistent chain: P(stay) = 0.97 in each regime.
    let stay = 0.97;

    let mut state = vec![0usize; n];
    let mut y = vec![0.0; n];
    let mut s = 0usize;
    // Presample.
    y[0] = mu[s] + sd[s] * rng.normal();
    state[0] = s;
    for t in 1..n {
        let prev = s;
        if rng.uniform() > stay {
            s = 1 - s;
        }
        state[t] = s;
        // Hamilton switching-mean AR(1).
        y[t] = mu[s] + phi * (y[t - 1] - mu[prev]) + sd[s] * rng.normal();
    }

    let model = MarkovSwitchingAr::new(&y, SPEC).unwrap();
    let start = MsarParams::new(
        vec![vec![0.9, 0.1], vec![0.1, 0.9]],
        vec![-3.0, 3.0],
        vec![vec![0.0]],
        vec![1.0, 1.0],
    )
    .unwrap();
    let fit = model.fit(&start, 500, 1e-9).unwrap();

    // Classified path aligns with the truth (allowing a global label swap).
    let pred = classify(&fit.smoothed_prob);
    // The smoothed path covers t = order..n, i.e. true states state[1..].
    let truth = &state[SPEC.order..];
    assert_eq!(pred.len(), truth.len());

    let agree = |flip: bool| -> f64 {
        let hits = pred
            .iter()
            .zip(truth.iter())
            .filter(|(&p, &t)| (if flip { 1 - p } else { p }) == t)
            .count();
        hits as f64 / pred.len() as f64
    };
    let accuracy = agree(false).max(agree(true));
    assert!(
        accuracy >= 0.95,
        "separated-regime classification accuracy {accuracy} should be >= 0.95"
    );
}
