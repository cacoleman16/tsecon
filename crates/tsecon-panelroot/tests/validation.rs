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

/// The reproducer from the interval/robustness audit: a 12 x 150 ARMA(1,1)
/// panel `y_t = 0.9 y_{t-1} + e_t + 0.4 e_{t-1}`. Persistent enough that the
/// unit-weighted (truncated-kernel) autocovariance sum goes negative for at
/// least one unit.
fn arma11_panel(seed: u64, n: usize, t: usize) -> Vec<Vec<f64>> {
    use tsecon_rng::Stream;
    use tsecon_stats::StdNormal;
    let mut s = Stream::new(seed);
    let mut gaussian = move || {
        let u = s.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
        StdNormal.ppf(u).expect("ppf on interior point")
    };
    (0..n)
        .map(|_| {
            let burn = 100;
            let mut y = 0.0;
            let mut eprev = gaussian();
            let mut out = Vec::with_capacity(t);
            for k in 0..(t + burn) {
                let e = gaussian();
                y = 0.9 * y + e + 0.4 * eprev;
                eprev = e;
                if k >= burn {
                    out.push(y);
                }
            }
            out
        })
        .collect()
}

/// A non-PSD long-run variance must be an error, never a NaN statistic.
///
/// Before the fix `lrv(...).sqrt()` on a negative estimate produced NaN,
/// which flowed into `s_n`, the bias term, `t*_delta` and the p-value, and
/// `panel_unit_root` returned `Ok` with `statistic = p_value = NaN`.
#[test]
fn llc_truncated_kernel_errors_instead_of_returning_nan() {
    let panel = arma11_panel(0xB41_5001, 12, 150);
    let trunc = PanelRootOpts {
        lrv_kernel: tsecon_hac::Kernel::Truncated,
        lrv_bandwidth: None,
    };
    let err = panel_unit_root(
        &panel,
        PanelRootTest::Llc,
        AdfRegression::Constant,
        AdfLagSelection::Fixed(1),
        &trunc,
    )
    .unwrap_err();
    match err {
        PanelRootError::NonPsdLongRunVariance {
            unit,
            kernel,
            value,
        } => {
            assert!(unit < 12, "unit index {unit} out of range");
            assert_eq!(kernel, "truncated");
            assert!(
                value <= 0.0,
                "the reported estimate {value} should be the non-positive one"
            );
        }
        other => panic!("expected NonPsdLongRunVariance, got {other:?}"),
    }
    let msg = err.to_string();
    assert!(msg.contains("truncated"), "message must name the kernel");
    assert!(
        msg.contains("bartlett") && msg.contains("parzen") && msg.contains("qs"),
        "message must suggest the PSD kernels: {msg}"
    );

    // The PSD kernels are unaffected: same panel, finite statistic.
    for k in [
        tsecon_hac::Kernel::Bartlett,
        tsecon_hac::Kernel::Parzen,
        tsecon_hac::Kernel::QuadraticSpectral,
    ] {
        let r = panel_unit_root(
            &panel,
            PanelRootTest::Llc,
            AdfRegression::Constant,
            AdfLagSelection::Fixed(1),
            &PanelRootOpts {
                lrv_kernel: k,
                lrv_bandwidth: None,
            },
        )
        .unwrap_or_else(|e| panic!("{k:?} should succeed, got {e}"));
        assert!(
            r.statistic.is_finite() && r.p_value.is_finite(),
            "{k:?} produced statistic={} p={}",
            r.statistic,
            r.p_value
        );
    }
}

/// No LLC path may return a non-finite statistic while reporting success —
/// the failure mode the guard closes. Swept over many seeds and both
/// deterministic cases so a single lucky draw cannot hide it.
#[test]
fn llc_never_returns_nan_successfully() {
    let mut errored = 0usize;
    let mut ok = 0usize;
    for seed in 0..40u64 {
        let panel = arma11_panel(0xB41_5100 + seed, 8, 120);
        for reg in [AdfRegression::Constant, AdfRegression::ConstantTrend] {
            for k in [
                tsecon_hac::Kernel::Truncated,
                tsecon_hac::Kernel::Bartlett,
                tsecon_hac::Kernel::Parzen,
                tsecon_hac::Kernel::QuadraticSpectral,
            ] {
                let out = panel_unit_root(
                    &panel,
                    PanelRootTest::Llc,
                    reg,
                    AdfLagSelection::Fixed(1),
                    &PanelRootOpts {
                        lrv_kernel: k,
                        lrv_bandwidth: None,
                    },
                );
                match out {
                    Ok(r) => {
                        ok += 1;
                        assert!(
                            r.statistic.is_finite() && r.p_value.is_finite(),
                            "seed {seed}, {reg:?}, {k:?}: Ok with statistic={} p={}",
                            r.statistic,
                            r.p_value
                        );
                    }
                    Err(e) => {
                        errored += 1;
                        assert!(
                            matches!(e, PanelRootError::NonPsdLongRunVariance { .. }),
                            "seed {seed}, {reg:?}, {k:?}: unexpected error {e}"
                        );
                    }
                }
            }
        }
    }
    // The sweep must actually exercise both arms, or it proves nothing.
    assert!(errored > 0, "no truncated-kernel failure was triggered");
    assert!(ok > 0, "no successful fit in the sweep");
}

/// The guard is a sign test, not a tolerance, so it must behave identically
/// however the data is scaled — and the LLC statistic itself is invariant to
/// a positive rescaling of `y` (both `sigma_y` and `sigma_eps` scale, so
/// `s_i`, `etil` and `vtil` do not).
#[test]
fn llc_guard_and_statistic_are_scale_invariant() {
    let base = arma11_panel(0xB41_5201, 10, 140);
    let mut reference: Option<f64> = None;
    for scale in [1e-8_f64, 1e-4, 1.0, 1e4, 1e8] {
        let panel: Vec<Vec<f64>> = base
            .iter()
            .map(|u| u.iter().map(|v| v * scale).collect())
            .collect();
        // The non-PSD guard fires at every scale, never a NaN.
        let err = panel_unit_root(
            &panel,
            PanelRootTest::Llc,
            AdfRegression::Constant,
            AdfLagSelection::Fixed(1),
            &PanelRootOpts {
                lrv_kernel: tsecon_hac::Kernel::Truncated,
                lrv_bandwidth: None,
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, PanelRootError::NonPsdLongRunVariance { .. }),
            "scale {scale:e}: expected the non-PSD error, got {err}"
        );

        // And the Bartlett statistic is unchanged by the rescaling.
        let r = panel_unit_root(
            &panel,
            PanelRootTest::Llc,
            AdfRegression::Constant,
            AdfLagSelection::Fixed(1),
            &OPTS,
        )
        .unwrap_or_else(|e| panic!("scale {scale:e}: {e}"));
        assert!(r.statistic.is_finite());
        match reference {
            None => reference = Some(r.statistic),
            Some(t0) => assert!(
                (r.statistic - t0).abs() < 1e-6 * t0.abs().max(1.0),
                "scale {scale:e}: t* = {} vs {t0} at scale 1",
                r.statistic
            ),
        }
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
