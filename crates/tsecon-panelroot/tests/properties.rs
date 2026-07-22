//! Monte-Carlo property tests — the honest calibration check.
//!
//! Golden fixtures pin the *algebra* (and, via `plm_anchor`, an external
//! implementation); these seeded simulations establish that the algebra is
//! the *statistically correct* one — in particular that every test rejects
//! in the right tail and is roughly calibrated. No external panel-root
//! engine runs inside `cargo test`, so this is what guards against sign /
//! tail-orientation errors and gross table-transcription mistakes.
//!
//! * SIZE: on iid random-walk panels (the null) each test keeps its nominal
//!   ~5% rejection rate within asymptotic slack.
//! * POWER: on stationary AR(1) panels (the alternative) each test rejects
//!   with high probability — confirming both calibration and tail direction.
//!
//! All randomness is the library's seeded Philox stream (`tsecon_rng`), so
//! the rates below are reproducible run to run.

use tsecon_diag::{AdfLagSelection, AdfRegression};
use tsecon_panelroot::{panel_unit_root, PanelRootOpts, PanelRootTest};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

fn gaussian(s: &mut Stream) -> f64 {
    let u = s.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// A random-walk panel (the unit-root null): each unit is a cumulative sum
/// of iid standard-normal innovations.
fn random_walk_panel(s: &mut Stream, n: usize, t: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| {
            let mut y = Vec::with_capacity(t);
            let mut acc = 0.0;
            for _ in 0..t {
                acc += gaussian(s);
                y.push(acc);
            }
            y
        })
        .collect()
}

/// A stationary AR(1) panel (the alternative): `y_t = rho y_{t-1} + e_t`
/// started from its stationary draw.
fn ar1_panel(s: &mut Stream, n: usize, t: usize, rho: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| {
            let mut y = Vec::with_capacity(t);
            let mut prev = gaussian(s) / (1.0 - rho * rho).sqrt();
            y.push(prev);
            for _ in 1..t {
                prev = rho * prev + gaussian(s);
                y.push(prev);
            }
            y
        })
        .collect()
}

fn rejection_rate(
    seed: u64,
    reps: usize,
    n: usize,
    t: usize,
    test: PanelRootTest,
    stationary: Option<f64>,
) -> f64 {
    let mut s = Stream::new(seed);
    let mut rejects = 0usize;
    for _ in 0..reps {
        let panel = match stationary {
            None => random_walk_panel(&mut s, n, t),
            Some(rho) => ar1_panel(&mut s, n, t, rho),
        };
        let r = panel_unit_root(
            &panel,
            test,
            AdfRegression::Constant,
            AdfLagSelection::Fixed(1),
            &PanelRootOpts::default(),
        )
        .expect("panel_unit_root ok");
        if r.p_value < 0.05 {
            rejects += 1;
        }
    }
    rejects as f64 / reps as f64
}

#[test]
fn size_is_near_nominal() {
    let reps = 600;
    // Per-test upper bounds. Fisher and IPS are well calibrated at N=10,
    // T=50; LLC is known to over-reject in small samples (its bias-adjusted
    // pooled t is only asymptotically N(0,1)) — plm::purtest shows the same
    // ~0.13-0.15 empirical size here, so the crate inherits it and the bound
    // is widened accordingly rather than masking a genuine property.
    let bounds: [(PanelRootTest, &str, f64, f64); 3] = [
        (PanelRootTest::Fisher, "fisher", 0.02, 0.12),
        (PanelRootTest::Ips, "ips", 0.015, 0.12),
        (PanelRootTest::Llc, "llc", 0.02, 0.20),
    ];
    for (test, tag, lo, hi) in bounds {
        let rate = rejection_rate(0xB41_2001 + tag.len() as u64, reps, 10, 50, test, None);
        eprintln!("SIZE {tag}: {rate:.3}");
        assert!(
            (lo..=hi).contains(&rate),
            "{tag} empirical size {rate:.3} outside [{lo}, {hi}] (nominal 0.05)"
        );
    }
}

#[test]
fn power_is_high_against_stationary_ar1() {
    let reps = 600;
    for (test, tag) in [
        (PanelRootTest::Fisher, "fisher"),
        (PanelRootTest::Ips, "ips"),
        (PanelRootTest::Llc, "llc"),
    ] {
        let rate = rejection_rate(0xB41_3001 + tag.len() as u64, reps, 10, 50, test, Some(0.8));
        eprintln!("POWER {tag} (rho=0.8): {rate:.3}");
        assert!(
            rate > 0.80,
            "{tag} empirical power {rate:.3} <= 0.80 against a stationary AR(1)"
        );
    }
}

/// A strongly stationary panel drives every statistic hard into its
/// rejection tail: very negative IPS/LLC statistics, a large Maddala-Wu P.
#[test]
fn tail_orientation() {
    let mut s = Stream::new(0xB41_4001);
    let panel = ar1_panel(&mut s, 12, 60, 0.2);
    let opts = PanelRootOpts::default();
    let ips = panel_unit_root(
        &panel,
        PanelRootTest::Ips,
        AdfRegression::Constant,
        AdfLagSelection::Fixed(1),
        &opts,
    )
    .unwrap();
    let llc = panel_unit_root(
        &panel,
        PanelRootTest::Llc,
        AdfRegression::Constant,
        AdfLagSelection::Fixed(1),
        &opts,
    )
    .unwrap();
    let fis = panel_unit_root(
        &panel,
        PanelRootTest::Fisher,
        AdfRegression::Constant,
        AdfLagSelection::Fixed(1),
        &opts,
    )
    .unwrap();
    assert!(
        ips.statistic < -3.0,
        "IPS should be strongly negative, got {}",
        ips.statistic
    );
    assert!(
        llc.statistic < -3.0,
        "LLC should be strongly negative, got {}",
        llc.statistic
    );
    assert!(
        fis.p_value < 1e-3,
        "Fisher should reject hard, got p={}",
        fis.p_value
    );
    assert!(ips.p_value < 1e-3 && llc.p_value < 1e-3);
}
