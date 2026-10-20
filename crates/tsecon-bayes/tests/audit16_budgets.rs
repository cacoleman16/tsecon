//! Audit round 16: size budgets and the `scale_ar` refusal of the Bayesian
//! layer.
//!
//! A horizon, draw count, chain count or grid size of `2^31` or `2^40` used
//! to reach an infallible allocation (`Vec::with_capacity`, a faer matrix)
//! and abort the process. Each is now refused as [`BayesError::TooLarge`],
//! naming the keyword and quoting the largest value that fits, before
//! anything is allocated — so these run in-process. A sample too short for
//! the AR(`scale_ar`) scale regressions is [`BayesError::ScaleArTooLong`],
//! naming `scale_ar` (through `with_scale_ar`), and `bvar_hierarchical`
//! raises it before its search instead of reporting a non-converged
//! optimization.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::{as_mat, load_fixture};
use tsecon_bayes::budget::{check_memory, irf_path_values, mat_values, SizeArg};
use tsecon_bayes::{
    bvar_hierarchical, bvar_ssvs, cholesky_irf, BayesError, HierarchicalConfig, MinnesotaNiwPrior,
    SsvsConfig, MAX_N_GRID,
};
use tsecon_linalg::faer::Mat;
use tsecon_rng::Stream;

const BIG: usize = 1 << 31;
const HUGE: usize = 1 << 40;

fn data() -> Mat<f64> {
    as_mat(&load_fixture("bvar_niw.json")["data"])
}

fn too_large(e: BayesError) -> (&'static str, usize, Option<usize>) {
    let msg = e.to_string();
    assert!(msg.contains("before anything is allocated"), "{msg}");
    match e {
        BayesError::TooLarge {
            arg, value, max, ..
        } => (arg, value, max),
        other => panic!("expected TooLarge, got {other:?}"),
    }
}

#[test]
fn cholesky_irf_refuses_an_absurd_horizon_and_quotes_the_boundary() {
    let x = data();
    let post = MinnesotaNiwPrior::new(x.as_ref(), 2, 100.0, 0.2, 1.0, 0.0)
        .unwrap()
        .posterior(x.as_ref())
        .unwrap();
    let b = post.b_bar().to_owned();
    let sigma = post.sigma_posterior_mean().unwrap();
    for h in [BIG, HUGE] {
        let (arg, value, max) =
            too_large(cholesky_irf(b.as_ref(), sigma.as_ref(), 2, h).unwrap_err());
        assert_eq!((arg, value), ("horizon", h));
        // The quoted value is the exact boundary of the documented count.
        let n = x.ncols();
        let cost = |h: usize| irf_path_values(n, h, 0) + mat_values(n * 2) * 3;
        let m = max.unwrap();
        assert!(cost(m) <= 1 << 28 && cost(m + 1) > 1 << 28, "max = {m}");
    }
    // An ordinary horizon is untouched.
    assert_eq!(
        cholesky_irf(b.as_ref(), sigma.as_ref(), 2, 40)
            .unwrap()
            .len(),
        41
    );
}

#[test]
fn irf_draws_names_n_draws_or_horizon() {
    let x = data();
    let post = MinnesotaNiwPrior::new(x.as_ref(), 2, 100.0, 0.2, 1.0, 0.0)
        .unwrap()
        .posterior(x.as_ref())
        .unwrap();
    let mut s = Stream::new(0);
    for d in [BIG, HUGE] {
        let (arg, value, max) = too_large(post.irf_draws(d, 8, &mut s).unwrap_err());
        assert_eq!((arg, value), ("n_draws", d));
        assert!(max.unwrap() > 1000, "{max:?}");
    }
    let (arg, _, _) = too_large(post.irf_draws(50, BIG, &mut s).unwrap_err());
    assert_eq!(arg, "horizon");
    assert_eq!(post.irf_draws(3, 4, &mut s).unwrap().len(), 3);
}

#[test]
fn ssvs_names_n_draws_n_chains_or_horizon_before_spawning() {
    let x = data();
    let cfg = |n_draws, n_chains, horizon| SsvsConfig {
        lags: 1,
        n_draws,
        burn: 50,
        horizon,
        n_chains,
        ..SsvsConfig::default()
    };
    for (c, want) in [
        (cfg(HUGE, 1, 4), "n_draws"),
        (cfg(BIG, 1, 4), "n_draws"),
        (cfg(200, BIG, 4), "n_chains"),
        (cfg(200, HUGE, 4), "n_chains"),
        (cfg(200, 1, HUGE), "horizon"),
    ] {
        let (arg, _, max) = too_large(bvar_ssvs(x.as_ref(), &c, 0).unwrap_err());
        assert_eq!(arg, want);
        assert!(max.is_some());
    }
    // lags = 0 names the keyword.
    let e = bvar_ssvs(
        x.as_ref(),
        &SsvsConfig {
            lags: 0,
            ..cfg(200, 1, 4)
        },
        0,
    )
    .unwrap_err()
    .to_string();
    assert!(
        e.starts_with("lags = 0") && !e.contains("lag length p"),
        "{e}"
    );
}

#[test]
fn hierarchical_caps_n_grid_by_name() {
    let x = data();
    for n_grid in [MAX_N_GRID + 1, BIG, HUGE] {
        let cfg = HierarchicalConfig {
            n_grid,
            ..HierarchicalConfig::default()
        };
        let (arg, value, max) = too_large(bvar_hierarchical(x.as_ref(), 2, &cfg).unwrap_err());
        assert_eq!((arg, value, max), ("n_grid", n_grid, Some(MAX_N_GRID)));
    }
}

#[test]
fn a_short_sample_names_scale_ar_with_the_largest_supported_order() {
    let x = data();
    let short = Mat::from_fn(9, x.ncols(), |i, j| x[(i, j)]);
    // 9 rows support 2 * scale_ar + 2 <= 9: scale_ar <= 3.
    let e =
        MinnesotaNiwPrior::with_scale_ar(short.as_ref(), 1, 100.0, 0.2, 1.0, 0.0, 4).unwrap_err();
    assert_eq!(
        e,
        BayesError::ScaleArTooLong {
            scale_ar: 4,
            needed: 10,
            got: 9,
            max: Some(3)
        }
    );
    assert!(e.to_string().contains("pass scale_ar <= 3"), "{e}");
    // An absurd order is refused by name, with checked arithmetic.
    let e = MinnesotaNiwPrior::with_scale_ar(x.as_ref(), 1, 100.0, 0.2, 1.0, 0.0, usize::MAX)
        .unwrap_err();
    assert!(matches!(
        e,
        BayesError::ScaleArTooLong {
            needed: usize::MAX,
            ..
        }
    ));
    // Callers without a scale_ar keyword still see a short sample.
    assert!(matches!(
        MinnesotaNiwPrior::new(short.as_ref(), 1, 100.0, 0.2, 1.0, 0.0),
        Err(BayesError::InsufficientObservations {
            needed: 10,
            got: 9,
            ..
        })
    ));
    // bvar_hierarchical reports it before the search (it used to surface as
    // a "poorly conditioned" NoConvergence).
    let e = bvar_hierarchical(short.as_ref(), 1, &HierarchicalConfig::default()).unwrap_err();
    assert!(
        matches!(
            e,
            BayesError::ScaleArTooLong {
                scale_ar: 4,
                max: Some(3),
                ..
            }
        ),
        "{e:?}"
    );
    let cfg = HierarchicalConfig {
        scale_ar: BIG,
        ..HierarchicalConfig::default()
    };
    assert!(matches!(
        bvar_hierarchical(x.as_ref(), 2, &cfg).unwrap_err(),
        BayesError::ScaleArTooLong { scale_ar: BIG, .. }
    ));
}

#[test]
fn the_budget_check_is_exact_at_its_boundary() {
    // A cost with a known boundary: 10 values per unit, budget 2^28.
    let cost = |v: &[usize]| (v[0] as u128) * 10;
    let max = (1usize << 28) / 10;
    assert!(check_memory("w", &[SizeArg::new("k", max, 1)], cost).is_ok());
    match check_memory("w", &[SizeArg::new("k", max + 1, 1)], cost) {
        Err(BayesError::TooLarge { max: Some(m), .. }) => assert_eq!(m, max),
        other => panic!("{other:?}"),
    }
}
