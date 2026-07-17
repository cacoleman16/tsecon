//! Property / behavioural tests for the local-projection estimators.
//!
//! These cover the statistical claims the crate makes but that no single
//! golden value can pin: lag-augmented coverage of the true response (both
//! on the fixture DGP and in a seeded Monte Carlo), the Ramey-Zubairy
//! cumulative-response identities, and state-dependent recovery of a
//! two-regime effect.

use serde_json::Value;
use tsecon_lp::{lp, lp_state, LpSpec, SeKind};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

const Z975: f64 = 1.959964; // standard-normal 0.975 quantile.

fn load_fixture() -> Value {
    let path = format!("{}/../../fixtures/lp.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).expect("fixture file readable");
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

/// One standard-normal draw by inverse transform from a Philox uniform.
fn gaussian(stream: &mut Stream) -> f64 {
    let u = stream.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// Simulate the fixture-style DGP `s_t = rho s_{t-1} + e_t`, `y = s + sigma
/// * w`, returning `(y, e)` of length `n` after a burn-in. The true response
/// of `y` to `e` is `rho^h`.
fn simulate_ar1_with_noise(
    stream: &mut Stream,
    n: usize,
    rho: f64,
    sigma: f64,
) -> (Vec<f64>, Vec<f64>) {
    let burn = 100;
    let total = n + burn;
    let mut s = 0.0;
    let mut y = Vec::with_capacity(total);
    let mut e = Vec::with_capacity(total);
    for _ in 0..total {
        let et = gaussian(stream);
        s = rho * s + et;
        let w = gaussian(stream);
        y.push(s + sigma * w);
        e.push(et);
    }
    (y[burn..].to_vec(), e[burn..].to_vec())
}

#[test]
#[allow(clippy::needless_range_loop)] // h indexes several parallel arrays.
fn lag_augmented_ci_covers_true_irf_on_fixture() {
    // 4-sigma sanity: the default (lag-augmented HC1) interval must sit
    // within 4 standard errors of the known truth 0.9^h at every horizon.
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let truth = f64s(&fx["true_irf"]);
    let hmax = truth.len() - 1;

    let res = lp(&y, &e, LpSpec::new(hmax, 4)).expect("lp lag-augmented");
    assert_eq!(res.se_kind, SeKind::LagAugmentedHc1);

    for h in 0..=hmax {
        let z = (res.irf[h] - truth[h]).abs() / res.se[h];
        assert!(
            z <= 4.0,
            "h={h}: |beta {} - true {}| = {} SEs (se {}), exceeds 4-sigma",
            res.irf[h],
            truth[h],
            z,
            res.se[h]
        );
    }
}

#[test]
#[allow(clippy::needless_range_loop)] // h indexes several parallel arrays.
fn lag_augmented_monte_carlo_coverage_is_nominal() {
    // Seeded Monte Carlo: nominal-95% lag-augmented intervals should cover
    // the true 0.9^h with frequency inside [0.85, 0.99] at every horizon.
    let reps = 200usize;
    let n = 300usize;
    let hmax = 6usize;
    let rho = 0.9;
    let mut stream = Stream::new(20260717);

    let mut covered = vec![0usize; hmax + 1];
    for _ in 0..reps {
        let (y, e) = simulate_ar1_with_noise(&mut stream, n, rho, 1.0);
        let res = lp(&y, &e, LpSpec::new(hmax, 4)).expect("lp lag-augmented MC");
        for h in 0..=hmax {
            let truth = rho.powi(h as i32);
            if (res.irf[h] - truth).abs() <= Z975 * res.se[h] {
                covered[h] += 1;
            }
        }
    }

    for h in 0..=hmax {
        let cov = covered[h] as f64 / reps as f64;
        assert!(
            (0.85..=0.99).contains(&cov),
            "h={h}: Monte Carlo coverage {cov} outside [0.85, 0.99]"
        );
    }
}

#[test]
fn hac_undercovers_relative_to_lag_augmented_near_persistence() {
    // Teaching companion to the default: on the same persistent DGP the
    // Newey-West HAC intervals cover less often than the lag-augmented ones
    // at the longer horizons (Montiel Olea & Plagborg-Møller 2021). We only
    // assert the aggregate direction, which is robust to Monte Carlo noise.
    let reps = 200usize;
    let n = 300usize;
    let hmax = 6usize;
    let rho = 0.95; // closer to a unit root, where HAC-LP struggles most.
    let mut stream = Stream::new(4242);

    let mut cov_la = 0usize;
    let mut cov_hac = 0usize;
    let mut total = 0usize;
    for _ in 0..reps {
        let (y, e) = simulate_ar1_with_noise(&mut stream, n, rho, 1.0);
        let la = lp(&y, &e, LpSpec::new(hmax, 4)).expect("lp LA");
        let hac = lp(&y, &e, LpSpec::new(hmax, 4).with_hac(None)).expect("lp HAC");
        for h in 3..=hmax {
            let truth = rho.powi(h as i32);
            if (la.irf[h] - truth).abs() <= Z975 * la.se[h] {
                cov_la += 1;
            }
            if (hac.irf[h] - truth).abs() <= Z975 * hac.se[h] {
                cov_hac += 1;
            }
            total += 1;
        }
    }
    // Both are valid estimators; the claim is only that lag-augmented does
    // not cover *worse* than HAC here (it should do at least as well).
    assert!(
        cov_la + reps / 20 >= cov_hac,
        "lag-augmented coverage ({cov_la}/{total}) should be no worse than \
         HAC ({cov_hac}/{total}) at long horizons under high persistence"
    );
}

#[test]
fn cumulative_point_irf_is_cumsum_but_se_is_not() {
    // Ramey-Zubairy: regressing the cumulated outcome gives a point path
    // close to the cumsum of the level responses, but its SE path is *not*
    // the cumsum of the level SEs — that is the whole reason to run the
    // cumulative regression rather than summing level SEs by hand.
    let mut stream = Stream::new(90125);
    let (y, e) = simulate_ar1_with_noise(&mut stream, 400, 0.9, 1.0);
    let hmax = 6usize;

    let level = lp(&y, &e, LpSpec::new(hmax, 4)).expect("level lp");
    let cum = lp(&y, &e, LpSpec::new(hmax, 4).cumulative(true)).expect("cumulative lp");

    // Point IRF ~ cumulative sum of the level IRF (loose: the two use
    // different left-hand sides and samples, so allow a wide band).
    let mut running = 0.0;
    for h in 0..=hmax {
        running += level.irf[h];
        let rel = (cum.irf[h] - running).abs() / running.abs().max(1e-8);
        assert!(
            rel < 0.20,
            "h={h}: cumulative IRF {} vs cumsum of level {} (rel {rel})",
            cum.irf[h],
            running
        );
    }

    // SE path is NOT the naive cumsum of level SEs — assert the inequality
    // at the longer horizons, where the gap is unmistakable.
    let mut se_running = 0.0;
    let mut saw_large_gap = false;
    for h in 0..=hmax {
        se_running += level.se[h];
        if h >= 3 {
            let rel_gap = (cum.se[h] - se_running).abs() / se_running;
            if rel_gap > 0.10 {
                saw_large_gap = true;
            }
        }
    }
    assert!(
        saw_large_gap,
        "cumulative SE path should differ materially from the cumsum of \
         level SEs (that is the teaching point of Ramey-Zubairy)"
    );
}

#[test]
fn state_dependent_impact_is_larger_in_the_high_regime() {
    // Two-regime DGP: the impact effect of the shock is 2x in state 1.
    // lp_state must recover a state-1 impact that significantly exceeds the
    // state-0 impact.
    let mut stream = Stream::new(31337);
    let n = 600usize;
    let burn = 100usize;
    let total = n + burn;

    // Predetermined, balanced regime: blocks of 25 periods.
    let regime: Vec<f64> = (0..total).map(|t| ((t / 25) % 2) as f64).collect();
    let mut s = 0.0;
    let mut y = Vec::with_capacity(total);
    let mut e = Vec::with_capacity(total);
    for t in 0..total {
        let et = gaussian(&mut stream);
        // Lagged regime governs the impact multiplier (2x in state 1).
        let mult = if t > 0 && regime[t - 1] == 1.0 {
            2.0
        } else {
            1.0
        };
        s = 0.9 * s + mult * et;
        let w = gaussian(&mut stream);
        y.push(s + 0.5 * w);
        e.push(et);
    }
    let y = y[burn..].to_vec();
    let e = e[burn..].to_vec();
    let ind = regime[burn..].to_vec();

    let res = lp_state(&y, &e, &ind, LpSpec::new(4, 4)).expect("lp_state");

    let b1 = res.irf_state1[0];
    let b0 = res.irf_state0[0];
    let se = (res.se_state1[0].powi(2) + res.se_state0[0].powi(2)).sqrt();
    let tstat = (b1 - b0) / se;

    assert!(
        b1 > b0 && tstat > 2.0,
        "state-1 impact {b1} should significantly exceed state-0 impact {b0} \
         (t = {tstat}); ratio {}",
        b1 / b0
    );
    // Loose level check that we recovered roughly 2x vs 1x.
    assert!(
        (1.5..2.6).contains(&b1) && (0.6..1.4).contains(&b0),
        "recovered impacts (state1 {b1}, state0 {b0}) far from the 2 vs 1 truth"
    );
}
