//! Audit round 16: the threshold-VECM grid and the Hansen-Seo bootstrap
//! refuse absurd sizes by name, with the largest admissible value, before
//! anything is allocated.
//!
//! * `hansen_seo_test` `n_boot` beyond the shared `MAX_TEST_REPLICATIONS`
//!   (10^6) cap: `n_boot = 2^31` allocated one RNG stream per replication
//!   (154 GB) and aborted the process.
//! * `threshold_vecm` `n_grid_beta * min(n_grid_gamma, n)` beyond
//!   `TVECM_MAX_GRID_CELLS` (2^20): `n_grid_beta = 2^31` requested a 16 GiB
//!   beta grid and aborted.

use std::time::Instant;

use tsecon_bootstrap::{WildWeights, MAX_TEST_REPLICATIONS};
use tsecon_coint::{hansen_seo_test, threshold_vecm, CointError, TVECM_MAX_GRID_CELLS};
use tsecon_linalg::faer::Mat;
use tsecon_rng::Stream;

/// A bivariate cointegrated system, `n x 2`: `x` a random walk and
/// `y = 1.5 x + noise`.
fn coint2(n: usize, seed: u64) -> Mat<f64> {
    let mut s = Stream::new(seed);
    let mut x = 0.0_f64;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        x += WildWeights::Normal.draw(&mut s);
        rows.push([1.5 * x + WildWeights::Normal.draw(&mut s), x]);
    }
    Mat::from_fn(n, 2, |i, j| rows[i][j])
}

#[test]
fn hansen_seo_refuses_n_boot_beyond_the_cap_before_any_work() {
    let data = coint2(200, 11);
    for n_boot in [MAX_TEST_REPLICATIONS + 1, 1 << 31, 1 << 40] {
        let t0 = Instant::now();
        let err = hansen_seo_test(data.as_ref(), 1, 0.05, 30, n_boot, 0, None).unwrap_err();
        assert!(
            t0.elapsed().as_secs_f64() < 5.0,
            "refused before any replication"
        );
        assert!(matches!(err, CointError::InvalidArgument { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("n_boot must be at most 1000000") && msg.contains("n_boot <= 1000000"),
            "{msg}"
        );
    }
    let ok = hansen_seo_test(data.as_ref(), 1, 0.05, 30, 9, 0, None).unwrap();
    assert_eq!(ok.n_boot, 9);
}

#[test]
fn threshold_vecm_refuses_a_beta_grid_beyond_the_cell_budget() {
    assert_eq!(TVECM_MAX_GRID_CELLS, 1 << 20);
    let data = coint2(200, 12);
    // n = 200 - (k_ar_diff + 1) = 198 usable rows bound the gamma grid.
    for (n_grid_gamma, n_grid_beta, gamma_cells) in [
        (20, 1usize << 31, 20usize),
        (20, 1 << 40, 20),
        (300, 5297, 198),
        (20, usize::MAX, 20),
    ] {
        let err = threshold_vecm(
            data.as_ref(),
            1,
            0.05,
            n_grid_gamma,
            n_grid_beta,
            10.0,
            None,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, CointError::InvalidSpec { .. }), "{err:?}");
        assert!(
            msg.contains(&format!("n_grid_beta = {n_grid_beta}"))
                && msg.contains(&format!(
                    "at n_grid_gamma = {n_grid_gamma} the largest admissible n_grid_beta is {}",
                    TVECM_MAX_GRID_CELLS / gamma_cells
                )),
            "{msg}"
        );
    }
    // A beta grid that is never built (beta_span = 0 fixes beta at the
    // linear estimate) is not refused: the budget prices the grid searched.
    let fit = threshold_vecm(data.as_ref(), 1, 0.05, 20, 1 << 31, 0.0, None).unwrap();
    assert_eq!(fit.beta_grid.len(), 1);
}
