//! Input-validation and API-contract tests: the teaching errors and the
//! shape of every returned result.

use tsecon_diag::{AdfLagSelection, AdfRegression};
use tsecon_panelroot::{
    panel_unit_root, PanelRootDetail, PanelRootError, PanelRootOpts, PanelRootTest,
};
use tsecon_stats::ContinuousDist;

fn balanced(n: usize, t: usize, seed: u64) -> Vec<Vec<f64>> {
    // A simple deterministic pseudo-random walk (no RNG crate needed for the
    // validation paths; the numeric goldens live in golden.rs / properties.rs).
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    (0..n)
        .map(|_| {
            let mut acc = 0.0;
            (0..t)
                .map(|_| {
                    acc += next();
                    acc
                })
                .collect()
        })
        .collect()
}

const OPTS: PanelRootOpts = PanelRootOpts {
    lrv_kernel: tsecon_hac::Kernel::Bartlett,
    lrv_bandwidth: None,
};

#[test]
fn too_few_units_rejected() {
    let one = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]];
    let err = panel_unit_root(
        &one,
        PanelRootTest::Fisher,
        AdfRegression::Constant,
        AdfLagSelection::Fixed(0),
        &OPTS,
    )
    .unwrap_err();
    assert!(matches!(err, PanelRootError::TooFewUnits { n: 1 }));
    assert!(err.to_string().contains("at least 2"));
}

#[test]
fn ips_rejects_no_constant() {
    let panel = balanced(4, 40, 1);
    let err = panel_unit_root(
        &panel,
        PanelRootTest::Ips,
        AdfRegression::NoConstant,
        AdfLagSelection::Fixed(1),
        &OPTS,
    )
    .unwrap_err();
    assert!(matches!(err, PanelRootError::IpsNoConstant));
    assert!(err.to_string().contains("Im-Pesaran-Shin"));
}

#[test]
fn llc_rejects_unbalanced() {
    let mut panel = balanced(3, 40, 2);
    panel[2].truncate(35); // make it ragged
    let err = panel_unit_root(
        &panel,
        PanelRootTest::Llc,
        AdfRegression::Constant,
        AdfLagSelection::Fixed(1),
        &OPTS,
    )
    .unwrap_err();
    match err {
        PanelRootError::UnbalancedForLlc {
            unit,
            expected,
            got,
        } => {
            assert_eq!((unit, expected, got), (2, 40, 35));
        }
        other => panic!("expected UnbalancedForLlc, got {other:?}"),
    }
}

#[test]
fn ips_and_fisher_accept_unbalanced() {
    let mut panel = balanced(4, 45, 3);
    panel[1].truncate(38);
    panel[3].truncate(40);
    for test in [PanelRootTest::Ips, PanelRootTest::Fisher] {
        let r = panel_unit_root(
            &panel,
            test,
            AdfRegression::Constant,
            AdfLagSelection::Fixed(1),
            &OPTS,
        )
        .expect("unbalanced ips/fisher ok");
        assert_eq!(r.n_units, 4);
        assert_eq!(r.per_unit_nobs[1], 38 - 1 - 1);
        assert!(r.statistic.is_finite() && r.p_value.is_finite());
    }
}

#[test]
fn non_finite_rejected() {
    let mut panel = balanced(3, 40, 4);
    panel[1][10] = f64::NAN;
    let err = panel_unit_root(
        &panel,
        PanelRootTest::Fisher,
        AdfRegression::Constant,
        AdfLagSelection::Fixed(1),
        &OPTS,
    )
    .unwrap_err();
    assert!(matches!(err, PanelRootError::NonFinite { unit: 1 }));
}

#[test]
fn result_shapes_and_details() {
    let panel = balanced(5, 50, 5);
    // Fisher
    let f = panel_unit_root(
        &panel,
        PanelRootTest::Fisher,
        AdfRegression::Constant,
        AdfLagSelection::Fixed(1),
        &OPTS,
    )
    .unwrap();
    assert_eq!(f.test, PanelRootTest::Fisher);
    assert_eq!(f.per_unit_tstat.len(), 5);
    assert_eq!(f.per_unit_pvalue.len(), 5);
    assert_eq!(f.per_unit_lags.len(), 5);
    assert_eq!(f.per_unit_nobs.len(), 5);
    assert!(matches!(f.detail, PanelRootDetail::Fisher { .. }));
    // Clamped p-values are in (0, 1).
    assert!(f.per_unit_pvalue.iter().all(|&p| p > 0.0 && p < 1.0));

    // IPS
    let i = panel_unit_root(
        &panel,
        PanelRootTest::Ips,
        AdfRegression::ConstantTrend,
        AdfLagSelection::Fixed(1),
        &OPTS,
    )
    .unwrap();
    match i.detail {
        PanelRootDetail::Ips { t_bar } => assert!(t_bar.is_finite()),
        _ => panic!("expected Ips detail"),
    }
    // IPS p-value is Phi(statistic).
    let phi = tsecon_stats::StdNormal.cdf(i.statistic);
    assert!((phi - i.p_value).abs() < 1e-12);

    // LLC
    let l = panel_unit_root(
        &panel,
        PanelRootTest::Llc,
        AdfRegression::Constant,
        AdfLagSelection::Fixed(1),
        &OPTS,
    )
    .unwrap();
    match l.detail {
        PanelRootDetail::Llc {
            delta_hat,
            t_delta,
            s_n,
            t_bar_periods,
        } => {
            assert!(delta_hat.is_finite() && t_delta.is_finite());
            assert!(s_n > 0.0);
            assert_eq!(t_bar_periods, (50 - 1 - 1) as f64);
        }
        _ => panic!("expected Llc detail"),
    }
}

#[test]
fn llc_bandwidth_option_changes_statistic() {
    let panel = balanced(6, 60, 6);
    let base = panel_unit_root(
        &panel,
        PanelRootTest::Llc,
        AdfRegression::Constant,
        AdfLagSelection::Fixed(1),
        &OPTS,
    )
    .unwrap();
    let wide = PanelRootOpts {
        lrv_kernel: tsecon_hac::Kernel::Bartlett,
        lrv_bandwidth: Some(0.0),
    };
    let alt = panel_unit_root(
        &panel,
        PanelRootTest::Llc,
        AdfRegression::Constant,
        AdfLagSelection::Fixed(1),
        &wide,
    )
    .unwrap();
    // A zero-bandwidth LRV (short-run variance only) generally shifts s_n and
    // hence the bias-adjusted statistic.
    assert!((base.statistic - alt.statistic).abs() > 1e-9);
}
