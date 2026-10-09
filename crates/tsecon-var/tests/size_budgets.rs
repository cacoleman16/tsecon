//! Size budgets of the VAR engines (audit round 16).
//!
//! Every horizon, step, replication and simulation count is checked against
//! the shared budget before anything it sizes is allocated, and refused by
//! the Python keyword with the largest value that fits. Each refusal below
//! is for a count that, before the budget, made a single allocation of tens
//! of gigabytes or more and aborted the process; returning at all, with a
//! `SizeBudget` error, proves nothing of that size was requested.

mod common;

use common::Lcg;
use tsecon_linalg::faer::Mat;
use tsecon_var::budget::{check_horizon, BUDGET_VALUES};
use tsecon_var::forecast::ForecastBandScope;
use tsecon_var::irf_asymptotic::{
    check_irf_asymptotic_budget, irf_asymptotic_critical_values, irf_asymptotic_se, BandMethod,
    IrfBandScope,
};
use tsecon_var::irf_bootstrap::{bootstrap_irf_bands_simultaneous, bootstrap_irf_budget_values};
use tsecon_var::{bootstrap_irf_bands, ma_rep, Trend, VarError, VarResults, VarSpec};

const HUGE: [usize; 3] = [1 << 31, 1 << 40, usize::MAX];

fn var_data(n: usize, k: usize, seed: u64) -> Mat<f64> {
    let mut rng = Lcg::new(seed);
    let mut y = Mat::<f64>::zeros(n, k);
    for t in 1..n {
        for i in 0..k {
            let mut v = rng.gaussian();
            for j in 0..k {
                v += if i == j { 0.5 } else { 0.1 / k as f64 } * y[(t - 1, j)];
            }
            y[(t, i)] = v;
        }
    }
    y
}

fn fit(lags: usize, k: usize) -> VarResults {
    VarSpec {
        lags,
        trend: Trend::Constant,
    }
    .fit(var_data(200, k, 7).as_ref())
    .unwrap()
}

/// The keyword named and the largest value reported by a size refusal.
fn named<T: std::fmt::Debug>(r: Result<T, VarError>) -> (&'static str, Option<usize>, String) {
    match r {
        Err(e @ VarError::SizeBudget { .. }) => {
            let msg = e.to_string();
            match e {
                VarError::SizeBudget { name, max, .. } => (name, max, msg),
                _ => unreachable!(),
            }
        }
        other => panic!("expected a SizeBudget refusal, got {other:?}"),
    }
}

#[test]
fn every_moving_average_expansion_refuses_an_absurd_horizon_by_name() {
    let r = fit(2, 2);
    let (_, max, _) = named(check_horizon(2, 1 << 31).map(|_| ()));
    let max = max.unwrap();
    for h in HUGE {
        for (name, got, msg) in [
            named(ma_rep(&r.coefs, h)),
            named(r.ma_rep(h)),
            named(r.orth_ma_rep(h)),
            named(r.irf(h).map(|_| ())),
            named(r.fevd(h).map(|_| ())),
        ] {
            assert_eq!(name, "horizon", "{msg}");
            assert!(msg.starts_with("horizon:") && msg.contains("2^28"), "{msg}");
            // fevd counts `horizon` matrices, the others `horizon + 1`.
            assert!(got == Some(max) || got == Some(max + 1), "{msg}");
        }
    }
    // The boundary is exact and a valid horizon is untouched.
    assert!(r.ma_rep(max).is_ok());
    assert!(r.ma_rep(max + 1).is_err());
    assert_eq!(r.irf(12).unwrap().irfs.len(), 13);
    // A VAR(0) has its own allocation path; it is budgeted as well.
    let r0 = fit(0, 2);
    assert_eq!(named(r0.ma_rep(1 << 40)).0, "horizon");
    assert_eq!(r0.ma_rep(4).unwrap().len(), 5);
}

#[test]
fn forecast_paths_name_their_own_keyword() {
    let r = fit(2, 2);
    for steps in HUGE {
        for (name, _, msg) in [
            named(r.forecast_cov(steps)),
            named(r.forecast_interval(steps, 0.05)),
        ] {
            assert_eq!(name, "steps", "{msg}");
            assert!(
                msg.starts_with("steps:") && !msg.contains("got horizon"),
                "{msg}"
            );
        }
    }
}

#[test]
fn a_simultaneous_forecast_band_budgets_its_joint_covariance_and_simulation() {
    let r = fit(2, 2);
    // The joint covariance is (steps k)^2 values and cubic work in steps.
    for steps in [5_000, 1 << 31] {
        let (name, max, msg) = named(r.forecast_interval_simultaneous(
            steps,
            0.05,
            BandMethod::Sidak,
            ForecastBandScope::All,
            1,
            100_000,
        ));
        assert_eq!(name, "steps", "{msg}");
        let max = max.unwrap();
        assert!(max < steps && max > 100, "{msg}");
    }
    // An absurd simulation count: refused naming band_n_sim, with the
    // largest count that fits; that count is admitted (checked without
    // running it) and the default is untouched.
    for n_sim in [1 << 31, 1 << 40, usize::MAX] {
        let (name, max, msg) = named(r.forecast_interval_simultaneous(
            8,
            0.05,
            BandMethod::SupT,
            ForecastBandScope::All,
            1,
            n_sim,
        ));
        assert_eq!(name, "band_n_sim", "{msg}");
        assert_eq!(
            max,
            Some(tsecon_stats::simultaneous::sup_t_max_n_sim(1, 16))
        );
    }
    assert!(r
        .forecast_interval_simultaneous(8, 0.05, BandMethod::SupT, ForecastBandScope::All, 1, 5_000)
        .is_ok());
    let msg = r
        .forecast_interval_simultaneous(8, 0.05, BandMethod::SupT, ForecastBandScope::All, 1, 1)
        .unwrap_err()
        .to_string();
    assert!(msg.starts_with("band_n_sim must be at least 2"), "{msg}");
}

#[test]
fn the_irf_bootstrap_refuses_its_working_set_by_name() {
    let data = var_data(200, 2, 3);
    // An absurd replication count: named n_boot, with the largest count that
    // fits at this horizon (its working set at the boundary is in budget).
    for n_boot in [1 << 31, 1 << 40, usize::MAX] {
        let r = bootstrap_irf_bands(
            data.as_ref(),
            2,
            Trend::Constant,
            12,
            true,
            false,
            0.1,
            n_boot,
            0,
            false,
        );
        let (name, max, msg) = named(r);
        assert_eq!(name, "n_boot", "{msg}");
        assert!(msg.starts_with("n_boot and horizon:"), "{msg}");
        let max = max.unwrap();
        let at = |n| bootstrap_irf_budget_values(2, 2, 12, n, false, None).unwrap();
        assert!(at(max) <= BUDGET_VALUES && at(max + 1) > BUDGET_VALUES);
    }
    // An absurd horizon: no replication count fits, so horizon is named.
    for horizon in [1 << 31, 1 << 40, usize::MAX] {
        let r = bootstrap_irf_bands_simultaneous(
            data.as_ref(),
            2,
            Trend::Constant,
            horizon,
            true,
            false,
            0.1,
            50,
            0,
            true,
            BandMethod::SupT,
            IrfBandScope::All,
        );
        let (name, max, msg) = named(r);
        assert_eq!(name, "horizon", "{msg}");
        assert!(max.is_some_and(|m| m > 0 && m < horizon), "{msg}");
    }
    // Each replication is counted at its real size: a flat cube of
    // (horizon + 1) k^2 values plus a fixed overhead, not padded matrices.
    let per = bootstrap_irf_budget_values(2, 2, 12, 2, false, None).unwrap()
        - bootstrap_irf_budget_values(2, 2, 12, 1, false, None).unwrap();
    assert_eq!(per, 13 * 4 + 48 + 1);
}

#[test]
fn the_delta_method_bands_name_the_term_that_is_over() {
    let r = fit(2, 2);
    // Quadratic Jacobian work: named horizon, never band_scope / band_n_sim.
    for horizon in [5_000, 1 << 31, usize::MAX] {
        let (name, max, msg) = named(irf_asymptotic_se(&r, horizon, true, false));
        assert_eq!(name, "horizon", "{msg}");
        assert!(
            !msg.contains("band_scope") && !msg.contains("band_n_sim"),
            "{msg}"
        );
        let max = max.unwrap();
        assert!(check_irf_asymptotic_budget(2, 2, max, None, 0).is_ok());
        assert!(check_irf_asymptotic_budget(2, 2, max + 1, None, 0).is_err());
    }
    // A valid horizon is untouched.
    assert_eq!(irf_asymptotic_se(&r, 12, true, false).unwrap().len(), 13);
    // Only band_n_sim is out of range: it leads the refusal.
    for n_sim in [1 << 31, 1 << 40, usize::MAX] {
        let (name, max, msg) = named(irf_asymptotic_critical_values(
            &r,
            6,
            true,
            false,
            0.1,
            BandMethod::SupT,
            IrfBandScope::Horizon,
            1,
            n_sim,
        ));
        assert_eq!(name, "band_n_sim", "{msg}");
        assert!(msg.starts_with("band_n_sim:"), "{msg}");
        let max = max.unwrap();
        let band = Some((BandMethod::SupT, IrfBandScope::Horizon));
        assert!(check_irf_asymptotic_budget(2, 2, 6, band, max).is_ok());
        assert!(check_irf_asymptotic_budget(2, 2, 6, band, max + 1).is_err());
    }
    // A family covariance too large: named horizon and band_scope.
    let (name, _, msg) = named(irf_asymptotic_critical_values(
        &r,
        4_000,
        true,
        false,
        0.1,
        BandMethod::Sidak,
        IrfBandScope::All,
        1,
        100_000,
    ));
    assert_eq!(name, "horizon", "{msg}");
    assert!(msg.starts_with("horizon and band_scope:"), "{msg}");
    let msg = irf_asymptotic_critical_values(
        &r,
        6,
        true,
        false,
        0.1,
        BandMethod::SupT,
        IrfBandScope::Horizon,
        1,
        1,
    )
    .unwrap_err()
    .to_string();
    assert!(msg.starts_with("band_n_sim must be at least 2"), "{msg}");
}
