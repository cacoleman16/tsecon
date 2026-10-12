//! Audit round 16: size arguments that used to abort the process (or name an
//! engine variable) are refused by name, with the largest admissible value,
//! before anything is allocated.
//!
//! * `setar_test` / `threshold_var_test` `n_boot` beyond the shared
//!   `MAX_TEST_REPLICATIONS` (10^6) cap: `n_boot = 2^31` allocated one RNG
//!   stream per replication (154 GB) and aborted.
//! * `star` `n_gamma * n_c` beyond `STAR_MAX_GRID_CELLS` (2^20): a 2^31 grid
//!   dimension requested a 16 GiB SSR surface and aborted.
//! * `MarkovSwitchingAr` `k_regimes` below 2 or `k_regimes^(order + 1)` past
//!   the 4096-state cap: the refusals now name `k_regimes` / `order` and
//!   state the largest admissible value of each.

use std::time::Instant;

use tsecon_bootstrap::{WildWeights, MAX_TEST_REPLICATIONS};
use tsecon_regime::{
    setar_test, star, threshold_var_test, MsarSpec, RegimeError, StarModel, STAR_MAX_GRID_CELLS,
};
use tsecon_rng::Stream;

fn ar1(t: usize, seed: u64) -> Vec<f64> {
    let mut s = Stream::new(seed);
    let mut y = vec![0.0_f64; t];
    for i in 1..t {
        y[i] = 0.5 * y[i - 1] + WildWeights::Normal.draw(&mut s);
    }
    y
}

fn var2(t: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = Stream::new(seed);
    let mut y = vec![vec![0.0_f64; 2]; t];
    for i in 1..t {
        let (a, b) = (y[i - 1][0], y[i - 1][1]);
        y[i] = vec![
            0.5 * a + 0.1 * b + WildWeights::Normal.draw(&mut s),
            0.2 * a + 0.3 * b + WildWeights::Normal.draw(&mut s),
        ];
    }
    y
}

#[test]
fn bootstrap_tests_refuse_n_boot_beyond_the_cap_before_any_work() {
    assert_eq!(MAX_TEST_REPLICATIONS, 1_000_000);
    let y = ar1(200, 1);
    let v = var2(200, 2);
    for n_boot in [MAX_TEST_REPLICATIONS + 1, 1 << 31, 1 << 40] {
        let t0 = Instant::now();
        let e1 = setar_test(&y, 1, 1, 0.15, true, n_boot, 0).unwrap_err();
        let e2 = threshold_var_test(&v, 1, 0, 1, 0.10, true, 20, n_boot, 0).unwrap_err();
        assert!(
            t0.elapsed().as_secs_f64() < 5.0,
            "refused before any replication"
        );
        for e in [e1, e2] {
            assert!(
                matches!(e, RegimeError::InvalidParameter { name: "n_boot", .. }),
                "{e:?}"
            );
            let msg = e.to_string();
            assert!(
                msg.contains(&format!("n_boot = {n_boot}")) && msg.contains("n_boot <= 1000000"),
                "{msg}"
            );
        }
    }
    // A valid count is untouched (and still runs).
    assert_eq!(setar_test(&y, 1, 1, 0.15, true, 9, 0).unwrap().n_boot, 9);
}

#[test]
fn star_refuses_a_grid_beyond_the_cell_budget_before_allocating() {
    assert_eq!(STAR_MAX_GRID_CELLS, 1 << 20);
    let y = ar1(300, 3);
    let lstar = StarModel::Lstar;
    for (n_gamma, n_c) in [
        (1 << 31, 5),
        (5, 1 << 31),
        (1 << 40, 5),
        (1025, 1024),
        (usize::MAX, 3),
    ] {
        let err = star(&y, 1, &[1], lstar, 0.15, true, n_gamma, n_c).unwrap_err();
        assert_eq!(
            err,
            RegimeError::GridTooLarge {
                n_gamma,
                n_c,
                max_cells: STAR_MAX_GRID_CELLS
            }
        );
        let msg = err.to_string();
        assert!(
            msg.contains("n_gamma") && msg.contains("n_c") && msg.contains("1048576"),
            "{msg}"
        );
    }
    let msg = star(&y, 1, &[1], lstar, 0.15, true, 1 << 31, 5)
        .unwrap_err()
        .to_string();
    assert!(
        msg.contains("at n_c = 5 the largest admissible n_gamma is 209715"),
        "{msg}"
    );
    let msg = star(&y, 1, &[1], lstar, 0.15, true, 1025, 1024)
        .unwrap_err()
        .to_string();
    assert!(
        msg.contains("largest admissible n_gamma is 1024")
            && msg.contains("largest admissible n_c is 1023"),
        "{msg}"
    );
    // The budget itself is admissible: a 2^20-cell request passes the check
    // and reaches the next one (here the sample-length check, so the test
    // does not evaluate a million cells).
    let short = ar1(6, 4);
    let err = star(&short, 1, &[1], lstar, 0.15, true, 1024, 1024).unwrap_err();
    assert!(
        matches!(err, RegimeError::InsufficientData { .. }),
        "{err:?}"
    );
}

#[test]
fn markov_refusals_name_k_regimes_and_order_with_the_largest_admissible_values() {
    let spec = |k_regimes, order| MsarSpec {
        k_regimes,
        order,
        switching_ar: false,
        switching_variance: true,
    };
    for k in [0, 1] {
        let err = spec(k, 1).expanded_states().unwrap_err();
        assert!(
            matches!(
                err,
                RegimeError::InvalidParameter {
                    name: "k_regimes",
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(err.to_string().contains("k_regimes >= 2"), "{err}");
    }
    // The cap's boundary: 64^2 = 2^12 = 4096 states fit, one more does not.
    assert_eq!(spec(64, 1).expanded_states().unwrap(), 4096);
    assert_eq!(spec(2, 11).expanded_states().unwrap(), 4096);
    let cases = [
        (
            65,
            1,
            "at order = 1 the largest admissible k_regimes is 64",
            // order = 0 fits 65 states, but `fit` refuses order = 0, so it
            // is not offered.
            "no order >= 1 is admissible at k_regimes = 65",
        ),
        (
            100,
            2,
            "at order = 2 the largest admissible k_regimes is 16",
            "no order >= 1 is admissible at k_regimes = 100 (order = 1 admits at most 64 \
             regimes)",
        ),
        (
            2,
            12,
            "no k_regimes >= 2 is admissible at order = 12",
            "at k_regimes = 2 the largest admissible order is 11",
        ),
        (
            3,
            20,
            "no k_regimes >= 2 is admissible at order = 20",
            "at k_regimes = 3 the largest admissible order is 6",
        ),
        (
            1 << 31,
            1,
            "at order = 1 the largest admissible k_regimes is 64",
            "no order >= 1 is admissible at k_regimes = 2147483648",
        ),
    ];
    for (k, p, a, b) in cases {
        let err = spec(k, p).expanded_states().unwrap_err();
        assert_eq!(
            err,
            RegimeError::StateSpaceTooLarge {
                k_regimes: k,
                order: p,
                max_states: 4096
            }
        );
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("k_regimes = {k} and order = {p}")),
            "{msg}"
        );
        assert!(msg.contains(a) && msg.contains(b), "{msg}");
    }
}

/// Every bound the state-space refusal offers is exact, checked against
/// `expanded_states` itself over a grid that crosses the cap on both axes:
/// the quoted largest `k_regimes` fits at that `order` and one more does
/// not, the quoted largest `order` (never 0, which `fit` refuses) fits at
/// that `k_regimes` and one more does not, and "no order >= 1" is said
/// exactly when `k_regimes^2 > 4096`.
#[test]
fn markov_state_space_bounds_quoted_in_the_refusal_are_exact() {
    let fits = |k_regimes: usize, order: usize| {
        MsarSpec {
            k_regimes,
            order,
            switching_ar: false,
            switching_variance: true,
        }
        .expanded_states()
        .is_ok()
    };
    let number_after = |msg: &str, key: &str| -> Option<usize> {
        let rest = &msg[msg.find(key)? + key.len()..];
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok()
    };
    let mut refused = 0;
    for k in (2..=70).chain([4095, 4096, 4097, 1 << 20]) {
        for order in 0..=13 {
            if fits(k, order) {
                continue;
            }
            refused += 1;
            let msg = MsarSpec {
                k_regimes: k,
                order,
                switching_ar: false,
                switching_variance: true,
            }
            .expanded_states()
            .unwrap_err()
            .to_string();
            match number_after(
                &msg,
                &format!("at order = {order} the largest admissible k_regimes is "),
            ) {
                Some(km) => assert!(fits(km, order) && !fits(km + 1, order), "{msg}"),
                None => assert!(
                    !fits(2, order) && msg.contains("no k_regimes >= 2 is admissible"),
                    "{msg}"
                ),
            }
            match number_after(
                &msg,
                &format!("at k_regimes = {k} the largest admissible order is "),
            ) {
                Some(pm) => assert!(pm >= 1 && fits(k, pm) && !fits(k, pm + 1), "{msg}"),
                None => assert!(
                    !fits(k, 1)
                        && msg.contains(&format!("no order >= 1 is admissible at k_regimes = {k}")),
                    "{msg}"
                ),
            }
        }
    }
    assert!(
        refused > 500,
        "the grid crosses the cap ({refused} refusals)"
    );
}
