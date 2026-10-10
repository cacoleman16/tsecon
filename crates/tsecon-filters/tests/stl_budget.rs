//! The STL/MSTL iteration budget: `inner_iter`, `outer_iter` and MSTL's
//! `iterate` are fixed loop counts that never stop early, so an absurd count
//! used to run for hours (audit round 16, A8: 2^31 inner passes is about a
//! day). Counts whose LOESS work exceeds `STL_WORK_BUDGET` are now refused
//! by name before any smoothing, quoting the largest admissible value —
//! checked against the documented work formula, transcribed here
//! independently of the crate's arithmetic — while the default counts (at
//! most `STL_FREE_PASSES` inner passes per fit) are never refused.

use tsecon_filters::{
    mstl, stl, stl_pass_work, FiltersError, MstlParams, StlParams, STL_FREE_PASSES, STL_WORK_BUDGET,
};

fn monthly(n: usize) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let t = t as f64;
            10.0 + 0.05 * t + (t * std::f64::consts::TAU / 12.0).sin() + 0.1 * (t * 0.37).cos()
        })
        .collect()
}

fn two_seasonal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let t = t as f64;
            5.0 + (t * std::f64::consts::TAU / 7.0).sin()
                + 0.5 * (t * std::f64::consts::TAU / 30.0).cos()
                + 0.01 * t
        })
        .collect()
}

/// The documented units of one inner pass: `(n / sj + 1 + 3 P) min(ns,
/// ceil(n / P)) + (n / lj + 1) min(nl, n) + (n / tj + 1) min(nt, n) + 12 n`,
/// with unit jumps.
fn pass_units(n: u128, p: u128, ns: u128, nl: u128, nt: u128) -> u128 {
    let points = n + 1;
    (points + 3 * p) * ns.min(n.div_ceil(p)) + points * nl.min(n) + points * nt.min(n) + 12 * n
}

fn iteration_refusal(e: FiltersError) -> (&'static str, usize, usize, u128, String) {
    let text = e.to_string();
    match e {
        FiltersError::IterationBudget {
            param,
            value,
            max,
            work,
            ..
        } => (param, value, max, work, text),
        other => panic!("expected IterationBudget, got {other:?}"),
    }
}

#[test]
fn pass_work_matches_the_documented_formula() {
    let y = monthly(96);
    let r = stl(&y, 12, &StlParams::default()).unwrap();
    // statsmodels defaults at period 12: seasonal 7, trend 23, low_pass 13.
    assert_eq!(
        (r.config.seasonal, r.config.trend, r.config.low_pass),
        (7, 23, 13)
    );
    assert_eq!(stl_pass_work(96, &r.config), pass_units(96, 12, 7, 13, 23));
}

#[test]
fn absurd_inner_iter_is_refused_by_name() {
    let y = monthly(96);
    let params = StlParams {
        inner_iter: Some(1 << 31),
        outer_iter: Some(0),
        ..Default::default()
    };
    let (param, value, max, work, text) = iteration_refusal(stl(&y, 12, &params).unwrap_err());
    let units = pass_units(96, 12, 7, 13, 23);
    assert_eq!((param, value), ("inner_iter", 1 << 31));
    assert_eq!(work, (1u128 << 31) * units);
    assert_eq!(max as u128, (STL_WORK_BUDGET / units).max(STL_FREE_PASSES));
    assert!(
        text.starts_with("stl: inner_iter = 2147483648 is past the STL work budget"),
        "{text}"
    );
    assert!(
        text.contains(&format!("pass inner_iter <= {max}")),
        "{text}"
    );
    // The quoted maximum is the last count within the budget.
    assert!((max as u128) * units <= STL_WORK_BUDGET);
    assert!((max as u128 + 1) * units > STL_WORK_BUDGET);
}

#[test]
fn absurd_outer_iter_is_refused_by_name() {
    let y = monthly(96);
    for outer in [1usize << 31, 1 << 40] {
        let params = StlParams {
            inner_iter: Some(2),
            outer_iter: Some(outer),
            ..Default::default()
        };
        let (param, value, max, _, text) = iteration_refusal(stl(&y, 12, &params).unwrap_err());
        let units = pass_units(96, 12, 7, 13, 23);
        assert_eq!((param, value), ("outer_iter", outer));
        // 2 (outer + 1) units <= budget.
        assert_eq!(max as u128, STL_WORK_BUDGET / (2 * units) - 1);
        assert!(
            text.contains(&format!("pass outer_iter <= {max}")),
            "{text}"
        );
    }
}

#[test]
fn default_and_moderate_counts_run() {
    let y = monthly(96);
    for params in [
        StlParams::default(),
        StlParams {
            robust: true,
            ..Default::default()
        },
        // Past the free passes but well within the budget.
        StlParams {
            inner_iter: Some(40),
            outer_iter: Some(3),
            ..Default::default()
        },
    ] {
        assert!(stl(&y, 12, &params).is_ok(), "{params:?}");
    }
}

#[test]
fn mstl_names_iterate_inner_or_outer() {
    let y = two_seasonal(240);
    let base = MstlParams::default();
    // iterate = 2^31 rounds over two periods.
    let p = MstlParams {
        iterate: 1 << 31,
        ..base.clone()
    };
    let (param, value, max, _, text) = iteration_refusal(mstl(&y, &[7, 30], &p).unwrap_err());
    assert_eq!((param, value), ("iterate", 1 << 31));
    assert!(max >= 2, "the default round count is admissible");
    assert!(text.starts_with("mstl: iterate = 2147483648"), "{text}");
    // inner_iter / outer_iter forwarded to every STL pass.
    let p = MstlParams {
        stl: StlParams {
            inner_iter: Some(1 << 31),
            outer_iter: Some(0),
            ..Default::default()
        },
        ..base.clone()
    };
    let (param, ..) = iteration_refusal(mstl(&y, &[7, 30], &p).unwrap_err());
    assert_eq!(param, "inner_iter");
    let p = MstlParams {
        stl: StlParams {
            inner_iter: Some(2),
            outer_iter: Some(1 << 40),
            ..Default::default()
        },
        ..base.clone()
    };
    let (param, ..) = iteration_refusal(mstl(&y, &[7, 30], &p).unwrap_err());
    assert_eq!(param, "outer_iter");
    // A single surviving period runs one round whatever iterate says
    // (statsmodels semantics), so a huge iterate costs nothing there.
    let p = MstlParams {
        iterate: 1 << 31,
        ..base
    };
    let r = mstl(&y, &[7], &p).expect("one round");
    assert_eq!(r.iterate, 1);
}
