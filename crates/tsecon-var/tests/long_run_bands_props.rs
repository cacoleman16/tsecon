//! Properties and Monte Carlo coverage of the Blanchard-Quah long-run SVAR
//! bands ([`tsecon_var::long_run_bootstrap_bands`],
//! [`tsecon_var::long_run_asymptotic_bands`]).
//!
//! The golden (`long_run_bands_golden.rs`) pins the numbers to an independent
//! NumPy / statsmodels transcription. This file pins what the numbers must
//! *mean*:
//!
//! * the imposed long-run zeros come out with exactly zero width and zero
//!   standard error in both arms (identification is re-run inside every
//!   draw, so a zero restriction is a zero in every draw);
//! * without bias correction the point estimate is `long_run_svar`'s own
//!   `irf` (plus the anchors), bit for bit;
//! * the bootstrap is bit-identical at 1 and 4 rayon threads, and draws the
//!   same pseudo-samples as `var_irf_bands` at the same seed;
//! * cumulative bands are the percentiles of the per-draw running sums;
//! * the `impact_signs` anchor holds in every draw;
//! * the delta-method simultaneous band contains the pointwise band, is a
//!   pure function of `band_seed`, and collapses to it on a single cell;
//! * every allocation that scales with a user count is budgeted, both sides
//!   of the boundary, before anything is allocated; every refusal names its
//!   parameter.
//!
//! `mc_long_run_band_coverage` (`#[ignore]`) is the committed source of every
//! coverage number the model card and the replication page quote. Run it in
//! release:
//!
//! ```text
//! cargo test --release -p tsecon-var --test long_run_bands_props \
//!     mc_long_run_band_coverage -- --ignored --nocapture
//! ```
//!
//! Its output is committed at `lab/bq_bands/out/mc_coverage.txt`.
//! `mc_smoke_kilian_level_coverage` is the always-on, small version.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::{as_mat, load_fixture, Lcg};
use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_var::irf_asymptotic::{BandMethod, IrfBandScope};
use tsecon_var::long_run_bands::{
    asymptotic_budget, asymptotic_jacobian_work, bootstrap_budget_values,
    LONG_RUN_BANDS_BUDGET_VALUES, LONG_RUN_BANDS_DELTA_WORK,
};
use tsecon_var::{
    bootstrap_irf_bands, long_run_asymptotic_bands, long_run_bootstrap_bands, LongRunBandSpec,
    LongRunBands, Trend, VarResults, VarSpec,
};

// ---------------------------------------------------------------- helpers

fn bq_data() -> Mat<f64> {
    as_mat(&load_fixture("long_run_svar_bands.json")["bq_data"])
}

fn var3_data() -> Mat<f64> {
    as_mat(&load_fixture("long_run_svar_bands.json")["var3_data"])
}

fn fit(data: &Mat<f64>, lags: usize) -> VarResults {
    VarSpec::new(lags, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap()
}

/// The page's specification: VAR(8), demand anchored so output rises.
fn bq_spec(horizon: usize, n_boot: usize) -> LongRunBandSpec {
    let mut s = LongRunBandSpec::new(8, horizon);
    s.impact_signs = Some(vec![(0, 1)]);
    s.n_boot = n_boot;
    s.seed = 20_260_911;
    s
}

/// The permuted-restriction 3-variable design of the golden.
fn var3_spec(horizon: usize, n_boot: usize) -> LongRunBandSpec {
    let mut s = LongRunBandSpec::new(2, horizon);
    s.long_run_zeros = Some(vec![(0, 0), (1, 0), (0, 2)]);
    s.normalize_impact = true;
    s.n_boot = n_boot;
    s.seed = 7;
    s
}

fn bits(m: &Mat<f64>) -> Vec<u64> {
    let mut v = Vec::with_capacity(m.nrows() * m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            v.push(m[(i, j)].to_bits());
        }
    }
    v
}

fn cube_bits(c: &[Mat<f64>]) -> Vec<u64> {
    c.iter().flat_map(bits).collect()
}

fn bands_bits(b: &LongRunBands) -> Vec<u64> {
    let mut v = cube_bits(&b.point);
    v.extend(cube_bits(&b.se));
    v.extend(cube_bits(&b.lower));
    v.extend(cube_bits(&b.upper));
    v.extend(bits(&b.long_run_se));
    v.extend(bits(&b.long_run_lower));
    v.extend(bits(&b.long_run_upper));
    if let Some(lo) = &b.sim_lower {
        v.extend(cube_bits(lo));
        v.extend(cube_bits(b.sim_upper.as_ref().unwrap()));
    }
    v
}

/// NumPy "linear" percentile of an ascending slice (the summaries' rule).
fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Per-cell Efron percentiles of a set of cubes.
fn percentiles(draws: &[Vec<Mat<f64>>], alpha: f64) -> (Vec<Mat<f64>>, Vec<Mat<f64>>) {
    let hh = draws[0].len();
    let k = draws[0][0].nrows();
    let mut lo = vec![Mat::<f64>::zeros(k, k); hh];
    let mut up = vec![Mat::<f64>::zeros(k, k); hh];
    let mut vals = vec![0.0f64; draws.len()];
    for h in 0..hh {
        for i in 0..k {
            for j in 0..k {
                for (b, d) in draws.iter().enumerate() {
                    vals[b] = d[h][(i, j)];
                }
                vals.sort_by(f64::total_cmp);
                lo[h][(i, j)] = percentile_sorted(&vals, alpha / 2.0);
                up[h][(i, j)] = percentile_sorted(&vals, 1.0 - alpha / 2.0);
            }
        }
    }
    (lo, up)
}

/// The per-draw running sum over horizons, in `theta_cube`'s order.
fn cumsum(c: &[Mat<f64>]) -> Vec<Mat<f64>> {
    let mut out: Vec<Mat<f64>> = c.to_vec();
    for h in 1..out.len() {
        let prev = out[h - 1].clone();
        out[h] = &out[h] + &prev;
    }
    out
}

/// The population (or full-sample) Blanchard-Quah IRF with anchors.
fn bq_truth(
    coefs: &[Mat<f64>],
    sigma: MatRef<'_, f64>,
    horizon: usize,
    spec: &LongRunBandSpec,
) -> tsecon_ident::LongRunSvar {
    let refs: Vec<MatRef<'_, f64>> = coefs.iter().map(Mat::as_ref).collect();
    let mut r = tsecon_ident::long_run_svar(
        &refs,
        sigma,
        horizon,
        spec.long_run_zeros.as_deref(),
        spec.normalize_impact,
    )
    .unwrap();
    if let Some(a) = spec.impact_signs.as_deref() {
        tsecon_ident::apply_impact_signs(&mut r, a).unwrap();
    }
    r
}

fn err_text<T>(r: Result<T, tsecon_var::VarError>) -> String {
    match r {
        Ok(_) => panic!("expected a refusal"),
        Err(e) => e.to_string(),
    }
}

// ---------------------------------------------------------- self-test

/// A data set, a spec, and the long-run zeros the spec imposes.
type ZeroCase = (Mat<f64>, LongRunBandSpec, Vec<(usize, usize)>);

/// Identification is re-run inside every draw, so an imposed long-run zero is
/// a zero in every draw: zero width and zero standard error, in both arms. A
/// non-degenerate imposed cell would prove the identification had been
/// hoisted out of the loop.
#[test]
fn imposed_long_run_zeros_have_exactly_zero_width_in_both_arms() {
    let cases: Vec<ZeroCase> = vec![
        (bq_data(), bq_spec(6, 40), vec![(0, 1)]),
        (var3_data(), var3_spec(6, 40), vec![(0, 0), (1, 0), (0, 2)]),
    ];
    for (data, spec, zeros) in cases {
        let boot = long_run_bootstrap_bands(data.as_ref(), &spec).unwrap();
        let delta = long_run_asymptotic_bands(&fit(&data, spec.lags), &spec).unwrap();
        for b in [&boot, &delta] {
            for &(v, s) in &zeros {
                assert_eq!(b.long_run_point[(v, s)], 0.0, "point ({v},{s})");
                assert_eq!(b.long_run_se[(v, s)], 0.0, "se ({v},{s})");
                assert_eq!(b.long_run_lower[(v, s)], 0.0, "lower ({v},{s})");
                assert_eq!(b.long_run_upper[(v, s)], 0.0, "upper ({v},{s})");
            }
            // ... and only those: every free long-run cell has a real band.
            let k = b.long_run_se.nrows();
            for v in 0..k {
                for s in 0..k {
                    if !zeros.contains(&(v, s)) {
                        assert!(b.long_run_se[(v, s)] > 0.0, "free ({v},{s})");
                        assert!(b.long_run_upper[(v, s)] > b.long_run_lower[(v, s)]);
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------ point estimate

/// Without bias correction the reported point is `long_run_svar`'s `irf`
/// (with the anchors applied) bit for bit, and its cumulative point is the
/// running sum of that `irf` — in both arms.
#[test]
fn point_is_long_run_svar_bit_for_bit_without_bias_correction() {
    for (data, spec) in [
        (bq_data(), bq_spec(12, 20)),
        (var3_data(), var3_spec(12, 20)),
    ] {
        let res = fit(&data, spec.lags);
        let truth = bq_truth(&res.coefs, res.sigma_u.as_ref(), 12, &spec);
        for cumulative in [false, true] {
            let spec = LongRunBandSpec {
                cumulative,
                ..spec.clone()
            };
            let want = if cumulative {
                cumsum(&truth.irf)
            } else {
                truth.irf.clone()
            };
            let boot = long_run_bootstrap_bands(data.as_ref(), &spec).unwrap();
            let delta = long_run_asymptotic_bands(&res, &spec).unwrap();
            for b in [&boot, &delta] {
                assert_eq!(cube_bits(&b.point), cube_bits(&want), "cum={cumulative}");
                assert_eq!(bits(&b.impact_point), bits(&truth.impact));
                assert_eq!(bits(&b.long_run_point), bits(&truth.long_run));
            }
        }
    }
}

/// With bias correction the point moves (it is the map at the Kilian-corrected
/// coefficients), and the delta-method arm refuses the flag by name.
#[test]
fn bias_correction_moves_the_point_and_is_refused_by_the_delta_arm() {
    let data = bq_data();
    let mut spec = bq_spec(8, 30);
    spec.bias_correct = true;
    let bc = long_run_bootstrap_bands(data.as_ref(), &spec).unwrap();
    let res = fit(&data, 8);
    let plain = bq_truth(&res.coefs, res.sigma_u.as_ref(), 8, &spec);
    assert_ne!(cube_bits(&bc.point), cube_bits(&plain.irf));
    assert!(bc.bias_correct);
    let msg = err_text(long_run_asymptotic_bands(&res, &spec));
    assert!(msg.contains("bias_correct"), "{msg}");
}

// ------------------------------------------------------ reproducibility

#[test]
fn bootstrap_is_bit_identical_at_one_and_four_threads() {
    let data = var3_data();
    let mut spec = var3_spec(6, 48);
    spec.bias_correct = true;
    spec.band = Some((BandMethod::SupT, IrfBandScope::Shock));
    spec.keep_draws = true;
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let a = pool1.install(|| long_run_bootstrap_bands(data.as_ref(), &spec).unwrap());
    let b = pool4.install(|| long_run_bootstrap_bands(data.as_ref(), &spec).unwrap());
    assert_eq!(bands_bits(&a), bands_bits(&b));
    let da: Vec<u64> = a
        .irf_draws
        .as_ref()
        .unwrap()
        .iter()
        .flat_map(|c| cube_bits(c))
        .collect();
    let db: Vec<u64> = b
        .irf_draws
        .as_ref()
        .unwrap()
        .iter()
        .flat_map(|c| cube_bits(c))
        .collect();
    assert_eq!(da, db);
    // The seed is live.
    let other =
        long_run_bootstrap_bands(data.as_ref(), &LongRunBandSpec { seed: 8, ..spec }).unwrap();
    assert_ne!(cube_bits(&a.lower), cube_bits(&other.lower));
}

/// One engine, one stream: at the same `(seed, n_boot)` the long-run bands
/// and `var_irf_bands` refit the VAR on the same pseudo-samples. Each draw's
/// reduced-form MA weights are recovered from its structural responses as
/// `Psi_h = Theta_h B^{-1}` (`B = Theta_0`), and their percentiles must be
/// `var_irf_bands(orth=False)`'s — plain and Kilian-corrected.
#[test]
fn draws_the_same_pseudo_samples_as_var_irf_bands() {
    let data = var3_data();
    for bias_correct in [false, true] {
        let mut spec = var3_spec(5, 60);
        spec.bias_correct = bias_correct;
        spec.keep_draws = true;
        let lr = long_run_bootstrap_bands(data.as_ref(), &spec).unwrap();
        let psi_draws: Vec<Vec<Mat<f64>>> = lr
            .irf_draws
            .as_ref()
            .unwrap()
            .iter()
            .map(|cube| {
                let binv = cube[0].partial_piv_lu().inverse();
                cube.iter().map(|th| th * &binv).collect()
            })
            .collect();
        let (lo, up) = percentiles(&psi_draws, spec.alpha);
        let v = bootstrap_irf_bands(
            data.as_ref(),
            2,
            Trend::Constant,
            5,
            false,
            false,
            spec.alpha,
            60,
            spec.seed,
            bias_correct,
        )
        .unwrap();
        for h in 0..=5 {
            for i in 0..3 {
                for j in 0..3 {
                    for (got, want) in [
                        (lo[h][(i, j)], v.lower[h][(i, j)]),
                        (up[h][(i, j)], v.upper[h][(i, j)]),
                    ] {
                        assert!(
                            (got - want).abs() <= 1e-12 + 1e-9 * want.abs(),
                            "bc={bias_correct} h={h} ({i},{j}): {got} vs {want}"
                        );
                    }
                }
            }
        }
    }
}

/// Cumulation happens per draw, before the percentiles: the cumulative band
/// is exactly the percentile band of the running sums of the plain draws.
#[test]
fn cumulative_bands_are_percentiles_of_the_cumulated_draws() {
    let data = bq_data();
    let mut spec = bq_spec(10, 50);
    spec.keep_draws = true;
    let plain = long_run_bootstrap_bands(data.as_ref(), &spec).unwrap();
    let cum = long_run_bootstrap_bands(
        data.as_ref(),
        &LongRunBandSpec {
            cumulative: true,
            ..spec.clone()
        },
    )
    .unwrap();
    let summed: Vec<Vec<Mat<f64>>> = plain
        .irf_draws
        .as_ref()
        .unwrap()
        .iter()
        .map(|c| cumsum(c))
        .collect();
    let kept: Vec<u64> = cum
        .irf_draws
        .as_ref()
        .unwrap()
        .iter()
        .flat_map(|c| cube_bits(c))
        .collect();
    let mine: Vec<u64> = summed.iter().flat_map(|c| cube_bits(c)).collect();
    assert_eq!(
        kept, mine,
        "irf_draws of a cumulative call are the cumulated draws"
    );
    let (lo, up) = percentiles(&summed, spec.alpha);
    assert_eq!(cube_bits(&cum.lower), cube_bits(&lo));
    assert_eq!(cube_bits(&cum.upper), cube_bits(&up));
    // The long-run draws do not depend on cumulation.
    assert_eq!(
        plain
            .long_run_draws
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(bits)
            .collect::<Vec<_>>(),
        cum.long_run_draws
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(bits)
            .collect::<Vec<_>>()
    );
    assert_eq!(plain.irf_draws.as_ref().unwrap().len(), 50);
}

/// The anchor is re-applied inside every draw: output's impact response to
/// demand is non-negative in all of them, and on the Blanchard-Quah data no
/// draw's anchor decides differently from the point's.
#[test]
fn the_impact_sign_anchor_holds_in_every_draw() {
    let data = bq_data();
    let mut spec = bq_spec(4, 80);
    spec.keep_draws = true;
    let b = long_run_bootstrap_bands(data.as_ref(), &spec).unwrap();
    for (n, cube) in b.irf_draws.as_ref().unwrap().iter().enumerate() {
        assert!(cube[0][(0, 1)] >= 0.0, "draw {n}: {}", cube[0][(0, 1)]);
    }
    assert!(b.point[0][(0, 1)] > 0.0);
    assert_eq!(b.sign_anchor_mismatches, Some(vec![0]));
    // No anchors -> no mismatch count.
    let none = long_run_bootstrap_bands(
        data.as_ref(),
        &LongRunBandSpec {
            impact_signs: None,
            ..spec
        },
    )
    .unwrap();
    assert_eq!(none.sign_anchor_mismatches, None);
}

/// A near-unit-root sample produces explosive raw refits, and `min_root` is
/// the full-sample inverse companion radius (the `var_fit` convention).
#[test]
fn unstable_draws_are_counted_and_min_root_is_the_inverse_radius() {
    let a1 = [[1.25, 0.05], [0.0, 0.9]];
    let a2 = [[-0.27, 0.0], [0.0, 0.08]];
    let y = simulate_var2(
        &a1,
        &a2,
        &[[1.0, 0.3], [0.3, 0.8]],
        70,
        300,
        &mut Lcg::new(11),
    );
    let mut spec = LongRunBandSpec::new(2, 6);
    spec.n_boot = 200;
    let b = long_run_bootstrap_bands(y.as_ref(), &spec).unwrap();
    let n_unstable = b.n_unstable_draws.unwrap();
    assert!(n_unstable > 0 && n_unstable < 200, "{n_unstable}");
    let res = fit(&y, 2);
    let moduli = res.roots_moduli().unwrap();
    let min_root = moduli.iter().copied().fold(f64::INFINITY, f64::min);
    assert_eq!(b.min_root.to_bits(), min_root.to_bits());
    assert!(b.min_root > 1.0);
    let d = long_run_asymptotic_bands(&res, &spec).unwrap();
    assert_eq!(d.n_unstable_draws, None);
    assert_eq!(d.min_root.to_bits(), min_root.to_bits());
}

// --------------------------------------------- delta-method sup-t band

#[test]
fn delta_simultaneous_band_contains_the_pointwise_band_and_shares_its_centre() {
    let data = bq_data();
    let res = fit(&data, 8);
    for scope in [
        IrfBandScope::Horizon,
        IrfBandScope::Shock,
        IrfBandScope::All,
    ] {
        for method in [BandMethod::SupT, BandMethod::Sidak, BandMethod::Bonferroni] {
            let mut spec = bq_spec(12, 2);
            spec.band = Some((method, scope));
            spec.band_n_sim = 4000;
            let b = long_run_asymptotic_bands(&res, &spec).unwrap();
            let plain =
                long_run_asymptotic_bands(&res, &LongRunBandSpec { band: None, ..spec }).unwrap();
            assert_eq!(cube_bits(&b.point), cube_bits(&plain.point));
            assert_eq!(cube_bits(&b.se), cube_bits(&plain.se));
            assert_eq!(cube_bits(&b.lower), cube_bits(&plain.lower));
            let (slo, sup) = (b.sim_lower.as_ref().unwrap(), b.sim_upper.as_ref().unwrap());
            let cv = b.simultaneous.as_ref().unwrap();
            for h in 0..=12 {
                for i in 0..2 {
                    for j in 0..2 {
                        assert!(slo[h][(i, j)] <= b.lower[h][(i, j)] + 1e-15);
                        assert!(sup[h][(i, j)] >= b.upper[h][(i, j)] - 1e-15);
                        assert!(cv.values[i][j] >= cv.pointwise);
                    }
                }
            }
            // n_cells_used counts exactly the cells with a positive delta se
            // (the family covariance's diagonal is the se's own variance).
            for i in 0..2 {
                for j in 0..2 {
                    let positive = match scope {
                        IrfBandScope::Horizon => {
                            (0..=12).filter(|&h| b.se[h][(i, j)] > 0.0).count()
                        }
                        IrfBandScope::Shock => (0..=12)
                            .map(|h| (0..2).filter(|&r| b.se[h][(r, j)] > 0.0).count())
                            .sum(),
                        IrfBandScope::All => (0..=12)
                            .map(|h| {
                                (0..2)
                                    .map(|r| (0..2).filter(|&c| b.se[h][(r, c)] > 0.0).count())
                                    .sum::<usize>()
                            })
                            .sum(),
                    };
                    assert_eq!(cv.n_cells_used[i][j], positive, "{scope:?} ({i},{j})");
                }
            }
        }
    }
}

#[test]
fn delta_sup_t_is_a_pure_function_of_band_seed() {
    let data = var3_data();
    let res = fit(&data, 2);
    let mut spec = var3_spec(8, 2);
    spec.band = Some((BandMethod::SupT, IrfBandScope::Horizon));
    spec.band_n_sim = 5000;
    let a = long_run_asymptotic_bands(&res, &spec).unwrap();
    let b = long_run_asymptotic_bands(&res, &spec).unwrap();
    assert_eq!(bands_bits(&a), bands_bits(&b));
    let c = long_run_asymptotic_bands(
        &res,
        &LongRunBandSpec {
            band_seed: 1,
            ..spec.clone()
        },
    )
    .unwrap();
    assert_ne!(
        a.simultaneous.as_ref().unwrap().values,
        c.simultaneous.as_ref().unwrap().values
    );
    // The bootstrap arm reads neither band_seed nor band_n_sim.
    let mut bs = var3_spec(4, 30);
    bs.band = Some((BandMethod::SupT, IrfBandScope::Horizon));
    let x = long_run_bootstrap_bands(data.as_ref(), &bs).unwrap();
    let y = long_run_bootstrap_bands(
        data.as_ref(),
        &LongRunBandSpec {
            band_seed: 99,
            band_n_sim: 7,
            ..bs
        },
    )
    .unwrap();
    assert_eq!(bands_bits(&x), bands_bits(&y));
}

#[test]
fn a_single_cell_delta_family_collapses_to_the_pointwise_band() {
    let res = fit(&bq_data(), 8);
    for method in [BandMethod::SupT, BandMethod::Sidak, BandMethod::Bonferroni] {
        let mut spec = bq_spec(0, 2);
        spec.band = Some((method, IrfBandScope::Horizon));
        let b = long_run_asymptotic_bands(&res, &spec).unwrap();
        let cv = b.simultaneous.as_ref().unwrap();
        assert_eq!(cv.n_cells, 1);
        let tol = if method == BandMethod::SupT {
            0.05
        } else {
            0.0
        };
        for i in 0..2 {
            for j in 0..2 {
                assert!((cv.values[i][j] - cv.pointwise).abs() <= tol, "{method:?}");
            }
        }
    }
}

// ------------------------------------------------------------- budgets

/// The bootstrap budget is exact at its boundary, refused one replication
/// past it naming `n_boot` and `horizon`, and an absurd count is refused
/// before any allocation (an attempted 2^64-value allocation would abort the
/// process, so returning at all proves nothing was allocated).
#[test]
fn bootstrap_budget_is_checked_at_its_boundary_before_allocating() {
    let data = bq_data();
    // The budget is affine in n_boot: a fixed reporting term plus a per-draw
    // term (the padded k x k draw matrices).
    let at = |n_boot: usize| {
        let s = LongRunBandSpec {
            n_boot,
            ..bq_spec(40, 2)
        };
        bootstrap_budget_values(&s, 2).unwrap()
    };
    let per = at(2) - at(1);
    let fixed = at(1) - per;
    // A 2 x 2 matrix occupies 8 * 2 + 8 doubles (64-byte column padding plus
    // the header), not 4: the budget counts storage, not cells.
    assert!(per >= 41 * 24, "{per}");
    let n_max = (LONG_RUN_BANDS_BUDGET_VALUES - fixed) / per;
    let mut spec = bq_spec(40, 2);
    spec.n_boot = n_max;
    assert!(bootstrap_budget_values(&spec, 2).unwrap() <= LONG_RUN_BANDS_BUDGET_VALUES);
    spec.n_boot = n_max + 1;
    assert!(bootstrap_budget_values(&spec, 2).unwrap() > LONG_RUN_BANDS_BUDGET_VALUES);
    let msg = err_text(long_run_bootstrap_bands(data.as_ref(), &spec));
    assert!(
        msg.contains("n_boot") && msg.contains("horizon") && msg.contains("2^28"),
        "{msg}"
    );
    for (n_boot, horizon) in [(usize::MAX, 40), (1000, usize::MAX), (1 << 40, 1 << 40)] {
        let s = LongRunBandSpec {
            n_boot,
            horizon,
            ..bq_spec(0, 2)
        };
        let msg = err_text(long_run_bootstrap_bands(data.as_ref(), &s));
        assert!(msg.contains("n_boot") && msg.contains("horizon"), "{msg}");
    }
    // Keeping the draws counts the caller's copy, so the boundary moves.
    let keep = LongRunBandSpec {
        n_boot: n_max,
        keep_draws: true,
        ..bq_spec(40, 2)
    };
    assert!(bootstrap_budget_values(&keep, 2).unwrap() > LONG_RUN_BANDS_BUDGET_VALUES);
}

#[test]
fn asymptotic_budget_is_checked_at_its_boundary_before_allocating() {
    let res = fit(&bq_data(), 8);
    // band_scope = "all": the joint covariance is ((H + 1) k^2)^2.
    let mut spec = bq_spec(1, 2);
    spec.band = Some((BandMethod::Sidak, IrfBandScope::All));
    let mut h = 1usize;
    while asymptotic_budget(
        &LongRunBandSpec {
            horizon: h + 1,
            ..spec.clone()
        },
        2,
    )
    .unwrap()
    .0 <= LONG_RUN_BANDS_BUDGET_VALUES
    {
        h += 1;
    }
    spec.horizon = h;
    assert!(asymptotic_budget(&spec, 2).unwrap().0 <= LONG_RUN_BANDS_BUDGET_VALUES);
    spec.horizon = h + 1;
    let msg = err_text(long_run_asymptotic_bands(&res, &spec));
    assert!(
        msg.contains("horizon") && msg.contains("band_scope"),
        "{msg}"
    );
    // A narrower family admits the same horizon.
    let narrow = LongRunBandSpec {
        band: Some((BandMethod::Sidak, IrfBandScope::Horizon)),
        ..spec.clone()
    };
    assert!(asymptotic_budget(&narrow, 2).unwrap().0 <= LONG_RUN_BANDS_BUDGET_VALUES);
    // Absurd horizons and simulation counts are refused, never allocated.
    for (horizon, n_sim) in [(usize::MAX, 100_000), (1 << 40, 100_000), (40, usize::MAX)] {
        let s = LongRunBandSpec {
            horizon,
            band_n_sim: n_sim,
            band: Some((BandMethod::SupT, IrfBandScope::All)),
            ..bq_spec(0, 2)
        };
        let msg = err_text(long_run_asymptotic_bands(&res, &s));
        assert!(
            msg.contains("horizon") && msg.contains("band_scope"),
            "{msg}"
        );
    }
}

/// The delta-method Jacobians are quadratic in `horizon`, so a horizon the
/// storage budget admits (pointwise, `k = 2`: thousands of periods) is still
/// refused, naming `horizon`, once their work passes the cap. Returning at
/// all proves the `G_h` sums were never started.
#[test]
fn asymptotic_jacobian_work_is_capped_quadratically_in_horizon() {
    let res = fit(&bq_data(), 8);
    let at = |horizon: usize| asymptotic_jacobian_work(&bq_spec(horizon, 2), 2);
    // horizon (horizon + 1) / 2 * k^2 * lags k^2, with k = 2 and lags = 8.
    assert_eq!(at(40), 40 * 41 / 2 * 4 * 32);
    let mut h = 1usize;
    while at(h + 1) <= LONG_RUN_BANDS_DELTA_WORK {
        h += 1;
    }
    let mut spec = bq_spec(h, 2);
    assert!(asymptotic_budget(&spec, 2).unwrap().0 <= LONG_RUN_BANDS_BUDGET_VALUES);
    spec.horizon = h + 1;
    assert!(asymptotic_budget(&spec, 2).unwrap().0 <= LONG_RUN_BANDS_BUDGET_VALUES);
    let msg = err_text(long_run_asymptotic_bands(&res, &spec));
    assert!(msg.contains("horizon") && msg.contains("2^28"), "{msg}");
    assert_eq!(at(usize::MAX), u128::MAX);
}

// ------------------------------------------------------------ refusals

#[test]
fn every_refusal_names_its_parameter() {
    let data = bq_data();
    let res = fit(&data, 8);
    let base = bq_spec(4, 20);
    let boot = |s: LongRunBandSpec| err_text(long_run_bootstrap_bands(data.as_ref(), &s));
    let delta = |s: LongRunBandSpec| err_text(long_run_asymptotic_bands(&res, &s));

    for alpha in [0.0, 1.0, -0.1, f64::NAN, f64::INFINITY] {
        let s = LongRunBandSpec {
            alpha,
            ..base.clone()
        };
        assert!(boot(s.clone()).contains("alpha"));
        assert!(delta(s).contains("alpha"));
    }
    for n_boot in [0, 1] {
        assert!(boot(LongRunBandSpec {
            n_boot,
            ..base.clone()
        })
        .contains("n_boot"));
    }
    let lags0 = LongRunBandSpec {
        lags: 0,
        ..base.clone()
    };
    assert!(boot(lags0.clone()).contains("lags"));
    assert!(delta(lags0).contains("lags"));
    for anchors in [vec![(2, 0)], vec![(0, 2)], vec![(0, 1), (1, 1)]] {
        let s = LongRunBandSpec {
            impact_signs: Some(anchors),
            ..base.clone()
        };
        assert!(boot(s.clone()).contains("impact_signs"));
        assert!(delta(s).contains("impact_signs"));
    }
    for zeros in [vec![(0, 0), (0, 1)], vec![(1, 0)], vec![(5, 0)]] {
        let s = LongRunBandSpec {
            long_run_zeros: Some(zeros),
            ..base.clone()
        };
        assert!(boot(s.clone()).contains("restrictions"));
        assert!(delta(s).contains("restrictions"));
    }
    let one_col = Mat::from_fn(data.nrows(), 1, |i, _| data[(i, 0)]);
    let msg = err_text(long_run_bootstrap_bands(one_col.as_ref(), &base));
    assert!(msg.contains("data") && msg.contains("2 variables"), "{msg}");
    let mut supt = base.clone();
    supt.band = Some((BandMethod::SupT, IrfBandScope::Horizon));
    supt.band_n_sim = 1;
    assert!(delta(supt).contains("band_n_sim"));
    // A non-finite sample is refused by the fit, before any draw.
    let mut bad = data.clone();
    bad[(10, 1)] = f64::NAN;
    assert!(long_run_bootstrap_bands(bad.as_ref(), &base).is_err());
    // A fitted VAR at other lags than the spec is a caller error.
    assert!(delta(LongRunBandSpec { lags: 4, ..base }).contains("lags"));
}

// ------------------------------------------------------- Monte Carlo

fn simulate_var2(
    a1: &[[f64; 2]; 2],
    a2: &[[f64; 2]; 2],
    sigma: &[[f64; 2]; 2],
    t: usize,
    burn: usize,
    rng: &mut Lcg,
) -> Mat<f64> {
    let c = [0.0, 0.0];
    let a = vec![
        Mat::from_fn(2, 2, |i, j| a1[i][j]),
        Mat::from_fn(2, 2, |i, j| a2[i][j]),
    ];
    let s = Mat::from_fn(2, 2, |i, j| sigma[i][j]);
    simulate_var(&a, &c, &s, t, burn, rng)
}

/// `y_t = c + sum_i A_i y_{t-i} + P e_t`, `P = chol(Sigma)`, `e_t` iid
/// standard normal; started at zero and burned in.
fn simulate_var(
    a: &[Mat<f64>],
    c: &[f64],
    sigma: &Mat<f64>,
    t: usize,
    burn: usize,
    rng: &mut Lcg,
) -> Mat<f64> {
    let k = sigma.nrows();
    let p = a.len();
    let l = sigma
        .llt(tsecon_linalg::faer::Side::Lower)
        .unwrap()
        .L()
        .to_owned();
    let n = t + burn + p;
    let mut y = Mat::<f64>::zeros(n, k);
    let mut e = vec![0.0f64; k];
    for tt in p..n {
        for v in e.iter_mut() {
            *v = rng.gaussian();
        }
        for r in 0..k {
            let mut v = c[r];
            for (i, ai) in a.iter().enumerate() {
                for cc in 0..k {
                    v += ai[(r, cc)] * y[(tt - 1 - i, cc)];
                }
            }
            for cc in 0..=r {
                v += l[(r, cc)] * e[cc];
            }
            y[(tt, r)] = v;
        }
    }
    Mat::from_fn(t, k, |i, j| y[(burn + p + i, j)])
}

/// One Monte Carlo design: a k = 2 DGP read as (growth, unemployment) with
/// the Blanchard-Quah scheme and the demand anchor `(0, 1)`.
struct Design {
    name: &'static str,
    coefs: Vec<Mat<f64>>,
    intercept: Vec<f64>,
    sigma: Mat<f64>,
    t: usize,
}

/// `Sigma_u = B B'` for the structural model whose long-run matrix is the
/// lower-triangular `lr`: `B = (I - A_1 - A_2) lr`, so the Blanchard-Quah
/// scheme identifies it exactly and the true responses are known.
fn sigma_from_long_run(coefs: &[Mat<f64>], lr: &Mat<f64>) -> Mat<f64> {
    let mut d = Mat::<f64>::identity(2, 2);
    for a in coefs {
        d = &d - a;
    }
    let b = &d * lr;
    &b * b.transpose()
}

/// The three designs. Both VAR(2)s load growth heavily on lagged
/// unemployment, so the anchored cell (output's impact response to demand)
/// is far from zero, as it is on the Blanchard-Quah data, and the anchor
/// decides the same way in essentially every draw.
fn designs() -> Vec<Design> {
    let m = |a: [[f64; 2]; 2]| Mat::from_fn(2, 2, |i, j| a[i][j]);
    let lr = m([[1.0, 0.0], [0.5, 1.5]]);
    let moderate = vec![m([[0.5, -0.4], [0.1, 0.5]]), m([[0.1, 0.1], [0.0, 0.1]])];
    let persistent = vec![
        m([[1.3, -0.3], [0.05, 1.2]]),
        m([[-0.33, 0.1], [0.0, -0.26]]),
    ];
    let page = fit(&bq_data(), 8);
    vec![
        Design {
            name: "moderate VAR(2), T = 200 (companion radius 0.65)",
            sigma: sigma_from_long_run(&moderate, &lr),
            coefs: moderate,
            intercept: vec![0.0, 0.0],
            t: 200,
        },
        Design {
            name: "persistent VAR(2), T = 200 (companion radius 0.95)",
            sigma: sigma_from_long_run(&persistent, &lr),
            coefs: persistent,
            intercept: vec![0.0, 0.0],
            t: 200,
        },
        Design {
            name: "the replication page's VAR(8) as truth, T = 202 (the page's sample)",
            coefs: page.coefs.clone(),
            intercept: page.intercept.clone(),
            sigma: page.sigma_u.clone(),
            t: bq_data().nrows(),
        },
    ]
}

/// Coverage counters for one arm: plain IRF, cumulative IRF, LR matrix.
struct Hits {
    plain: Vec<[[usize; 2]; 2]>,
    cum: Vec<[[usize; 2]; 2]>,
    lr: [[usize; 2]; 2],
}

impl Hits {
    fn new(hh: usize) -> Self {
        Hits {
            plain: vec![[[0; 2]; 2]; hh],
            cum: vec![[[0; 2]; 2]; hh],
            lr: [[0; 2]; 2],
        }
    }
}

fn covers(lo: f64, hi: f64, truth: f64) -> bool {
    lo <= truth && truth <= hi
}

fn score_cube(hits: &mut [[[usize; 2]; 2]], lo: &[Mat<f64>], up: &[Mat<f64>], truth: &[Mat<f64>]) {
    for (h, cell) in hits.iter_mut().enumerate() {
        for i in 0..2 {
            for j in 0..2 {
                if covers(lo[h][(i, j)], up[h][(i, j)], truth[h][(i, j)]) {
                    cell[i][j] += 1;
                }
            }
        }
    }
}

fn score_lr(hits: &mut [[usize; 2]; 2], lo: &Mat<f64>, up: &Mat<f64>, truth: &Mat<f64>) {
    for i in 0..2 {
        for j in 0..2 {
            if covers(lo[(i, j)], up[(i, j)], truth[(i, j)]) {
                hits[i][j] += 1;
            }
        }
    }
}

/// Hall (basic) band from Efron percentiles: `[2 point - upper, 2 point - lower]`.
fn hall(point: &[Mat<f64>], lo: &[Mat<f64>], up: &[Mat<f64>]) -> (Vec<Mat<f64>>, Vec<Mat<f64>>) {
    let l = point
        .iter()
        .zip(up)
        .map(|(p, u)| Mat::from_fn(2, 2, |i, j| 2.0 * p[(i, j)] - u[(i, j)]))
        .collect();
    let u = point
        .iter()
        .zip(lo)
        .map(|(p, l)| Mat::from_fn(2, 2, |i, j| 2.0 * p[(i, j)] - l[(i, j)]))
        .collect();
    (l, u)
}

struct McResult {
    reps: usize,
    efron: Hits,
    hall: Hits,
    kilian: Hits,
    delta: Hits,
    unstable_reps: [usize; 2],
    mismatch_reps: [usize; 2],
}

/// Runs `reps` Monte Carlo replications of one design. Each replication
/// scores four arms on the same sample: Efron percentiles, the Hall (basic)
/// band from the same draws, Efron with Kilian's bias correction, and the
/// delta method. Cumulative bootstrap bands are formed from the kept plain
/// draws (`cumulative_bands_are_percentiles_of_the_cumulated_draws` proves
/// that is the native cumulative band, bit for bit).
fn run_design(
    d: &Design,
    horizon: usize,
    reps: usize,
    n_boot: usize,
    seed: u64,
    arms: &[bool],
) -> McResult {
    let hh = horizon + 1;
    let mut spec = LongRunBandSpec::new(d.coefs.len(), horizon);
    spec.impact_signs = Some(vec![(0, 1)]);
    spec.n_boot = n_boot;
    spec.keep_draws = true;
    let truth = bq_truth(&d.coefs, d.sigma.as_ref(), horizon, &spec);
    let truth_cum = cumsum(&truth.irf);
    let mut out = McResult {
        reps: 0,
        efron: Hits::new(hh),
        hall: Hits::new(hh),
        kilian: Hits::new(hh),
        delta: Hits::new(hh),
        unstable_reps: [0; 2],
        mismatch_reps: [0; 2],
    };
    let mut rng = Lcg::new(seed);
    for rep in 0..reps {
        let y = simulate_var(&d.coefs, &d.intercept, &d.sigma, d.t, 300, &mut rng);
        spec.seed = rep as u64;
        let mut ok = true;
        let mut runs = Vec::new();
        for &bc in arms {
            match long_run_bootstrap_bands(
                y.as_ref(),
                &LongRunBandSpec {
                    bias_correct: bc,
                    ..spec.clone()
                },
            ) {
                Ok(b) => runs.push((bc, b)),
                Err(_) => ok = false,
            }
        }
        let res = VarSpec::new(d.coefs.len(), Trend::Constant)
            .unwrap()
            .fit(y.as_ref())
            .unwrap();
        let dspec = LongRunBandSpec {
            keep_draws: false,
            ..spec.clone()
        };
        let dp = long_run_asymptotic_bands(&res, &dspec);
        let dc = long_run_asymptotic_bands(
            &res,
            &LongRunBandSpec {
                cumulative: true,
                ..dspec
            },
        );
        if !ok || dp.is_err() || dc.is_err() {
            continue;
        }
        out.reps += 1;
        for (bc, b) in runs.iter() {
            let a = usize::from(*bc);
            let summed: Vec<Vec<Mat<f64>>> = b
                .irf_draws
                .as_ref()
                .unwrap()
                .iter()
                .map(|c| cumsum(c))
                .collect();
            let (clo, cup) = percentiles(&summed, spec.alpha);
            let hits = if a == 0 {
                &mut out.efron
            } else {
                &mut out.kilian
            };
            score_cube(&mut hits.plain, &b.lower, &b.upper, &truth.irf);
            score_cube(&mut hits.cum, &clo, &cup, &truth_cum);
            score_lr(
                &mut hits.lr,
                &b.long_run_lower,
                &b.long_run_upper,
                &truth.long_run,
            );
            if a == 0 {
                let (hl, hu) = hall(&b.point, &b.lower, &b.upper);
                let cpoint = cumsum(&b.point);
                let (chl, chu) = hall(&cpoint, &clo, &cup);
                score_cube(&mut out.hall.plain, &hl, &hu, &truth.irf);
                score_cube(&mut out.hall.cum, &chl, &chu, &truth_cum);
                let lp = &b.long_run_point;
                let lhl = Mat::from_fn(2, 2, |i, j| 2.0 * lp[(i, j)] - b.long_run_upper[(i, j)]);
                let lhu = Mat::from_fn(2, 2, |i, j| 2.0 * lp[(i, j)] - b.long_run_lower[(i, j)]);
                score_lr(&mut out.hall.lr, &lhl, &lhu, &truth.long_run);
            }
            if b.n_unstable_draws.unwrap() > 0 {
                out.unstable_reps[a] += 1;
            }
            if b.sign_anchor_mismatches.as_ref().unwrap()[0] > 0 {
                out.mismatch_reps[a] += 1;
            }
        }
        let (dp, dc) = (dp.unwrap(), dc.unwrap());
        score_cube(&mut out.delta.plain, &dp.lower, &dp.upper, &truth.irf);
        score_cube(&mut out.delta.cum, &dc.lower, &dc.upper, &truth_cum);
        score_lr(
            &mut out.delta.lr,
            &dp.long_run_lower,
            &dp.long_run_upper,
            &truth.long_run,
        );
    }
    out
}

fn rate(h: usize, n: usize) -> f64 {
    h as f64 / n as f64
}

/// The level objects: output LEVEL = cumulative row 0, unemployment = plain
/// row 1.
fn level_rate(hits: &Hits, h: usize, i: usize, j: usize, n: usize) -> f64 {
    if i == 0 {
        rate(hits.cum[h][0][j], n)
    } else {
        rate(hits.plain[h][1][j], n)
    }
}

fn print_design(d: &Design, r: &McResult, hs: &[usize], n_boot: usize) {
    let n = r.reps;
    println!("\n## {}", d.name);
    println!(
        "valid replications {n}, n_boot {n_boot}, alpha 0.10 (nominal 0.90), MC s.e. at 0.90 = {:.3}",
        (0.09 / n as f64).sqrt()
    );
    println!(
        "replications with any explosive raw refit: Efron {}, Kilian {}; with any anchor mismatch: Efron {}, Kilian {}",
        r.unstable_reps[0], r.unstable_reps[1], r.mismatch_reps[0], r.mismatch_reps[1]
    );
    let head: String = hs.iter().map(|h| format!(" h={h} |")).collect();
    let sep: String = hs.iter().map(|_| "---|").collect();
    let arms: [(&str, &Hits); 4] = [
        ("Efron", &r.efron),
        ("Hall", &r.hall),
        ("Efron+Kilian", &r.kilian),
        ("Delta", &r.delta),
    ];
    let shock = ["supply", "demand"];
    println!(
        "\n### Level objects (output LEVEL = cumulative growth row; unemployment = plain row)\n"
    );
    println!("| arm | cell |{head}");
    println!("|---|---|{sep}");
    for (name, hits) in arms {
        for (i, var) in ["output LEVEL", "unemployment"].iter().enumerate() {
            for (j, s) in shock.iter().enumerate() {
                let cells: String = hs
                    .iter()
                    .map(|&h| format!(" {:.3} |", level_rate(hits, h, i, j, n)))
                    .collect();
                println!("| {name} | {var} <- {s} |{cells}");
            }
        }
    }
    println!("\n### Plain IRF (growth row = output growth)\n");
    println!("| arm | cell |{head}");
    println!("|---|---|{sep}");
    for (name, hits) in arms {
        for (i, var) in ["growth", "unemployment"].iter().enumerate() {
            for (j, s) in shock.iter().enumerate() {
                let cells: String = hs
                    .iter()
                    .map(|&h| format!(" {:.3} |", rate(hits.plain[h][i][j], n)))
                    .collect();
                println!("| {name} | {var} <- {s} |{cells}");
            }
        }
    }
    println!("\n### Cumulative IRF\n");
    println!("| arm | cell |{head}");
    println!("|---|---|{sep}");
    for (name, hits) in arms {
        for (i, var) in ["cum growth", "cum unemployment"].iter().enumerate() {
            for (j, s) in shock.iter().enumerate() {
                let cells: String = hs
                    .iter()
                    .map(|&h| format!(" {:.3} |", rate(hits.cum[h][i][j], n)))
                    .collect();
                println!("| {name} | {var} <- {s} |{cells}");
            }
        }
        let avg: String = hs
            .iter()
            .map(|&h| {
                let s: usize = hits.cum[h].iter().flatten().sum();
                format!(" {:.3} |", s as f64 / (4 * n) as f64)
            })
            .collect();
        println!("| {name} | 4-cell average |{avg}");
    }
    println!("\n### Long-run matrix LR = C(1) B ([0][1] is the imposed zero)\n");
    println!("| arm | [0][0] | [0][1] | [1][0] | [1][1] |");
    println!("|---|---|---|---|---|");
    for (name, hits) in arms {
        println!(
            "| {name} | {:.3} | {:.3} | {:.3} | {:.3} |",
            rate(hits.lr[0][0], n),
            rate(hits.lr[0][1], n),
            rate(hits.lr[1][0], n),
            rate(hits.lr[1][1], n)
        );
    }
}

/// The committed Monte Carlo behind every coverage number the model card and
/// the replication page quote: three DGPs (a moderate VAR(2), a persistent
/// VAR(2), and the page's estimated VAR(8) taken as truth) x four arms
/// (Efron, Hall from the same draws, Efron + Kilian, delta) x every cell of
/// the plain, cumulative, level and long-run objects. Gaussian innovations;
/// 1000 replications x 299 bootstrap draws; nominal 90%.
#[test]
#[ignore = "Monte Carlo (minutes in release): run with --release --ignored --nocapture"]
fn mc_long_run_band_coverage() {
    const REPS: usize = 1000;
    const N_BOOT: usize = 299;
    const H: usize = 40;
    let hs = [0usize, 1, 2, 4, 8, 12, 20, 40];
    println!("# long_run_svar_bands coverage Monte Carlo (crates/tsecon-var/tests/long_run_bands_props.rs)");
    println!(
        "Blanchard-Quah scheme, impact_signs [(0, 1)], trend \"c\", lags = the DGP's order, \
         {REPS} replications x {N_BOOT} bootstrap draws, horizon {H}."
    );
    for (n, d) in designs().iter().enumerate() {
        let t0 = std::time::Instant::now();
        let r = run_design(d, H, REPS, N_BOOT, 0x5EED_0000 + n as u64, &[false, true]);
        print_design(d, &r, &hs, N_BOOT);
        println!("\n(wall time {:.1} s)", t0.elapsed().as_secs_f64());
    }
}

/// The always-on, small version: on the moderate VAR(2) at T = 200 (100
/// replications x 99 draws), Kilian-corrected level coverage averaged over
/// the four level cells at h = 0..8 sits in [0.80, 0.97] at nominal 0.90.
#[test]
fn mc_smoke_kilian_level_coverage() {
    let d = &designs()[0];
    let r = run_design(d, 8, 100, 99, 0x5EED_5A0E, &[true]);
    assert!(r.reps >= 95, "{} valid replications", r.reps);
    let mut acc = 0.0;
    let mut cells = 0.0;
    for h in 0..=8 {
        for i in 0..2 {
            for j in 0..2 {
                acc += level_rate(&r.kilian, h, i, j, r.reps);
                cells += 1.0;
            }
        }
    }
    let avg = acc / cells;
    assert!(
        (0.80..=0.97).contains(&avg),
        "Kilian level coverage {avg:.3}"
    );
    // The imposed zero is covered in every replication, in both arms run.
    for hits in [&r.kilian, &r.delta] {
        assert_eq!(hits.lr[0][1], r.reps);
    }
}
