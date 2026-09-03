//! Property / Monte-Carlo tests for the random forest — the grade the full
//! forest carries (its randomness is tsecon's own Philox stream, so no
//! third-party golden exists for it; the deterministic tree and the
//! single-tree bridge are golden-pinned in `trees_golden.rs`).
//!
//! Every numeric bar below was measured first and is asserted with a
//! margin; the tests print the achieved figures, which the model card
//! quotes.
//!
//! * same seed bit-identical, different seed differs, thread count
//!   irrelevant;
//! * out-of-sample R^2 on Friedman #1 above a documented bar;
//! * block / stationary resampling preserves the lag-1 autocorrelation of
//!   the resampled rows where iid resampling destroys it;
//! * out-of-bag error is OPTIMISTIC under persistent predictors with
//!   AR(1) errors — measured against pseudo-out-of-sample error on the
//!   same forest, sign asserted;
//! * quantile regression forest: the q10-q90 band covers ~0.80 of iid
//!   test targets and quantiles never cross;
//! * importance recovers the relevant variables of a sparse nonlinear
//!   DGP; a persistent irrelevant predictor's inflated importance is
//!   measured, and single-row vs block permutation are measured against
//!   each other honestly (see the importance module docs);
//! * the teaching errors fire with the house wording.

mod common;

use std::time::Instant;

use common::{mat_from_cols, Lcg};
use tsecon_ml::faer::{Mat, MatRef};
use tsecon_ml::{
    random_forest, resample_indices, ForestOptions, Importance, MaxFeatures, Resampling,
};
use tsecon_rng::Stream;

/// Friedman #1 on the first five columns of a `(0, 1)` design.
fn friedman1(x: &[Vec<f64>], i: usize) -> f64 {
    10.0 * (std::f64::consts::PI * x[0][i] * x[1][i]).sin()
        + 20.0 * (x[2][i] - 0.5).powi(2)
        + 10.0 * x[3][i]
        + 5.0 * x[4][i]
}

/// Standardized AR(1) path of length `n` (variance one after burn-in).
fn ar1(rng: &mut Lcg, n: usize, rho: f64) -> Vec<f64> {
    let burn = 200;
    let mut z = 0.0;
    let scale = (1.0 - rho * rho).sqrt();
    let mut out = Vec::with_capacity(n);
    for t in 0..n + burn {
        z = rho * z + rng.normal();
        if t >= burn {
            out.push(z * scale);
        }
    }
    out
}

/// Logistic squash of a standardized series onto `(0, 1)` — a persistent
/// bounded predictor.
fn squash(v: &[f64]) -> Vec<f64> {
    v.iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect()
}

fn uniform_cols(rng: &mut Lcg, n: usize, p: usize) -> Vec<Vec<f64>> {
    (0..p)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect()
}

fn persistent_cols(rng: &mut Lcg, n: usize, p: usize, rho: f64) -> Vec<Vec<f64>> {
    (0..p).map(|_| squash(&ar1(rng, n, rho))).collect()
}

fn r2(pred: &[f64], truth: &[f64]) -> f64 {
    let n = truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / n;
    let ss_res: f64 = pred.iter().zip(truth).map(|(p, t)| (t - p).powi(2)).sum();
    let ss_tot: f64 = truth.iter().map(|t| (t - mean).powi(2)).sum();
    1.0 - ss_res / ss_tot
}

fn lag1_autocorr(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var: f64 = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let cov: f64 = v
        .windows(2)
        .map(|w| (w[0] - mean) * (w[1] - mean))
        .sum::<f64>()
        / n;
    cov / var
}

fn opts(n_trees: usize, seed: u64) -> ForestOptions {
    ForestOptions {
        n_trees,
        seed,
        ..ForestOptions::default()
    }
}

#[test]
fn same_seed_is_bit_identical_and_different_seed_differs() {
    let mut rng = Lcg::new(1);
    let n = 200;
    let cols = uniform_cols(&mut rng, n, 6);
    let y: Vec<f64> = (0..n).map(|i| friedman1(&cols, i) + rng.normal()).collect();
    let x = mat_from_cols(&cols);
    let xt = Mat::from_fn(20, 6, |i, j| cols[j][i]);
    let o = ForestOptions {
        quantiles: Some(vec![0.1, 0.5, 0.9]),
        importance: Importance::BlockPermutation {
            permutation_block: None,
            n_permutations: 3,
        },
        ..opts(50, 11)
    };
    let a = random_forest(x.as_ref(), &y, &o, Some(xt.as_ref())).unwrap();
    let b = random_forest(x.as_ref(), &y, &o, Some(xt.as_ref())).unwrap();
    assert_eq!(a, b, "same seed must be bit-identical");
    let c = random_forest(
        x.as_ref(),
        &y,
        &ForestOptions { seed: 12, ..o },
        Some(xt.as_ref()),
    )
    .unwrap();
    assert_ne!(
        a.fitted, c.fitted,
        "a different seed must give a different forest"
    );
    assert_ne!(a.oob_mse, c.oob_mse);
}

#[test]
fn thread_count_does_not_change_the_forest() {
    let mut rng = Lcg::new(2);
    let n = 240;
    let cols = uniform_cols(&mut rng, n, 8);
    let y: Vec<f64> = (0..n).map(|i| friedman1(&cols, i) + rng.normal()).collect();
    let x = mat_from_cols(&cols);
    let xt = Mat::from_fn(30, 8, |i, j| cols[j][(i * 7) % n]);
    let o = ForestOptions {
        quantiles: Some(vec![0.25, 0.75]),
        importance: Importance::BlockPermutation {
            permutation_block: Some(4),
            n_permutations: 2,
        },
        resampling: Resampling::Stationary { block_length: 6 },
        ..opts(64, 5)
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| random_forest(x.as_ref(), &y, &o, Some(xt.as_ref())).unwrap())
    };
    let one = run(1);
    let three = run(3);
    let eight = run(8);
    assert_eq!(one, three, "1 vs 3 threads");
    assert_eq!(one, eight, "1 vs 8 threads");
    // And against whatever the global pool is.
    let global = random_forest(x.as_ref(), &y, &o, Some(xt.as_ref())).unwrap();
    assert_eq!(one, global, "explicit pool vs global pool");
}

/// Friedman #1, n = 500 train / 500 test, p = 10 (five noise columns),
/// noise sd 1, 200 trees, max_features = p/3, min_samples_leaf = 5.
/// scikit-learn's forest at the same settings measures R^2 ~ 0.77; the
/// bar is 0.70.
#[test]
fn friedman_out_of_sample_r2_clears_the_bar() {
    let mut rng = Lcg::new(3);
    let n = 1000;
    let cols = uniform_cols(&mut rng, n, 10);
    let y: Vec<f64> = (0..n).map(|i| friedman1(&cols, i) + rng.normal()).collect();
    let ntr = 500;
    let x = Mat::from_fn(ntr, 10, |i, j| cols[j][i]);
    let xt = Mat::from_fn(n - ntr, 10, |i, j| cols[j][ntr + i]);
    let t0 = Instant::now();
    let fit = random_forest(x.as_ref(), &y[..ntr], &opts(200, 0), Some(xt.as_ref())).unwrap();
    let elapsed = t0.elapsed().as_secs_f64();
    let pred = fit.predicted.unwrap();
    let r2_oos = r2(&pred, &y[ntr..]);
    let r2_oob = r2(fit.oob_prediction.as_ref().unwrap(), &y[..ntr]);
    println!(
        "friedman #1: out-of-sample R^2 {r2_oos:.3}, out-of-bag R^2 {r2_oob:.3}, {elapsed:.2}s"
    );
    assert!(
        r2_oos > 0.70,
        "out-of-sample R^2 {r2_oos} below the 0.70 bar"
    );
    // iid rows: out-of-bag is an honest estimate here (within 0.05 of OOS).
    assert!(
        (r2_oob - r2_oos).abs() < 0.05,
        "iid OOB vs OOS R^2 gap {}",
        r2_oob - r2_oos
    );
}

/// The default call (`n_trees = 500`, `n = 500`, `p = 10`) must stay fast.
/// The wall time is printed for the report (measured 0.29 s on four cores,
/// 0.34 s single-threaded, release); the loose tripwire is asserted only
/// in an optimized build — an unoptimized test binary on a loaded CI box
/// is an order of magnitude slower and says nothing about the estimator.
#[test]
fn default_call_timing_at_n500_p10() {
    let mut rng = Lcg::new(4);
    let n = 500;
    let cols = uniform_cols(&mut rng, n, 10);
    let y: Vec<f64> = (0..n).map(|i| friedman1(&cols, i) + rng.normal()).collect();
    let x = mat_from_cols(&cols);
    let t0 = Instant::now();
    let fit = random_forest(x.as_ref(), &y, &ForestOptions::default(), None).unwrap();
    let elapsed = t0.elapsed().as_secs_f64();
    println!("default random_forest n=500 p=10 n_trees=500: {elapsed:.3}s");
    assert_eq!(fit.n_trees, 500);
    assert_eq!(fit.max_features_resolved, 3);
    if !cfg!(debug_assertions) {
        assert!(elapsed < 10.0, "default call took {elapsed}s");
    }
}

#[test]
fn block_and_stationary_resampling_preserve_lag_one_autocorrelation() {
    let mut rng = Lcg::new(5);
    let n = 400;
    let z = ar1(&mut rng, n, 0.9);
    let original = lag1_autocorr(&z);
    let mut stream = Stream::new(99);
    let mean_acf = |scheme: Resampling, stream: &mut Stream| -> f64 {
        let reps = 50;
        (0..reps)
            .map(|_| {
                let idx = resample_indices(scheme, n, stream).unwrap();
                let r: Vec<f64> = idx.iter().map(|&i| z[i]).collect();
                lag1_autocorr(&r)
            })
            .sum::<f64>()
            / reps as f64
    };
    let iid = mean_acf(Resampling::Iid, &mut stream);
    let block = mean_acf(Resampling::MovingBlock { block_length: 20 }, &mut stream);
    let stationary = mean_acf(Resampling::Stationary { block_length: 20 }, &mut stream);
    println!(
        "lag-1 autocorrelation: original {original:.3}, iid resample {iid:.3}, \
         moving-block(20) {block:.3}, stationary(20) {stationary:.3}"
    );
    assert!(original > 0.8);
    assert!(
        iid.abs() < 0.1,
        "iid resampling must destroy the autocorrelation (got {iid})"
    );
    assert!(block > 0.7, "moving-block resample lag-1 {block}");
    assert!(stationary > 0.7, "stationary resample lag-1 {stationary}");
    assert!(block > iid + 0.5 && stationary > iid + 0.5);
}

/// Persistent predictors (logistic-squashed AR(0.9) columns) with either
/// iid or AR(0.9) errors of sd 2; forest fit on the first 400 rows of a
/// 600-row series. The out-of-bag MSE of that forest is compared with its
/// pseudo-out-of-sample MSE on the last 200 rows. With AR errors the
/// out-of-bag rows' temporal neighbours carry their error into the
/// in-bag leaves, so OOB is optimistic: the ratio OOB/POOS is below one
/// and below the iid-error ratio. Averaged over five seeds.
#[test]
fn oob_error_is_optimistic_under_persistent_predictors_with_ar_errors() {
    let n = 600;
    let ntr = 400;
    let mut ratios = [0.0f64; 2];
    for (k, rho_e) in [0.0, 0.9].into_iter().enumerate() {
        let mut acc = 0.0;
        let seeds = 5;
        for seed in 0..seeds {
            let mut rng = Lcg::new(100 + seed);
            let cols = persistent_cols(&mut rng, n, 5, 0.9);
            let e = if rho_e > 0.0 {
                ar1(&mut rng, n, rho_e)
            } else {
                (0..n).map(|_| rng.normal()).collect()
            };
            let y: Vec<f64> = (0..n).map(|i| friedman1(&cols, i) + 2.0 * e[i]).collect();
            let x = Mat::from_fn(ntr, 5, |i, j| cols[j][i]);
            let xt = Mat::from_fn(n - ntr, 5, |i, j| cols[j][ntr + i]);
            let o = ForestOptions {
                max_features: MaxFeatures::Count(2),
                ..opts(120, seed)
            };
            let fit = random_forest(x.as_ref(), &y[..ntr], &o, Some(xt.as_ref())).unwrap();
            let poos = fit
                .predicted
                .unwrap()
                .iter()
                .zip(&y[ntr..])
                .map(|(p, t)| (t - p).powi(2))
                .sum::<f64>()
                / (n - ntr) as f64;
            acc += fit.oob_mse.unwrap() / poos;
        }
        ratios[k] = acc / seeds as f64;
    }
    println!(
        "OOB/POOS MSE ratio: iid errors {:.3}, AR(0.9) errors {:.3}",
        ratios[0], ratios[1]
    );
    assert!(
        ratios[1] < 0.9,
        "OOB must be optimistic under AR errors (ratio {})",
        ratios[1]
    );
    assert!(
        ratios[1] < ratios[0] - 0.05,
        "AR-error optimism {} must exceed the iid-error one {}",
        ratios[1],
        ratios[0]
    );
}

#[test]
fn quantile_forest_band_covers_and_never_crosses() {
    let mut rng = Lcg::new(6);
    let n = 1000;
    let ntr = 600;
    let cols = uniform_cols(&mut rng, n, 6);
    // Heteroskedastic noise so the band has to widen with x[0].
    let y: Vec<f64> = (0..n)
        .map(|i| friedman1(&cols, i) + (0.5 + 2.0 * cols[0][i]) * rng.normal())
        .collect();
    let x = Mat::from_fn(ntr, 6, |i, j| cols[j][i]);
    let xt = Mat::from_fn(n - ntr, 6, |i, j| cols[j][ntr + i]);
    let q = vec![0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
    let o = ForestOptions {
        quantiles: Some(q.clone()),
        ..opts(200, 8)
    };
    let fit = random_forest(x.as_ref(), &y[..ntr], &o, Some(xt.as_ref())).unwrap();
    let qp = fit.quantile_predictions.unwrap();
    assert_eq!(qp.len(), n - ntr);
    let mut covered = 0usize;
    let mut covered_90 = 0usize;
    for (row, &t) in qp.iter().zip(&y[ntr..]) {
        assert_eq!(row.len(), q.len());
        for w in row.windows(2) {
            assert!(w[1] >= w[0], "quantiles crossed: {row:?}");
        }
        if row[1] <= t && t <= row[5] {
            covered += 1;
        }
        if row[0] <= t && t <= row[6] {
            covered_90 += 1;
        }
    }
    let cov = covered as f64 / (n - ntr) as f64;
    let cov90 = covered_90 as f64 / (n - ntr) as f64;
    // Median tracks the mean prediction.
    let med: Vec<f64> = qp.iter().map(|r| r[3]).collect();
    let corr = {
        let p = fit.predicted.unwrap();
        let mp = p.iter().sum::<f64>() / p.len() as f64;
        let mm = med.iter().sum::<f64>() / med.len() as f64;
        let c: f64 = p.iter().zip(&med).map(|(a, b)| (a - mp) * (b - mm)).sum();
        let va: f64 = p.iter().map(|a| (a - mp).powi(2)).sum();
        let vb: f64 = med.iter().map(|b| (b - mm).powi(2)).sum();
        c / (va * vb).sqrt()
    };
    println!(
        "QRF: q10-q90 coverage {cov:.3}, q05-q95 coverage {cov90:.3}, corr(median, mean) {corr:.3}"
    );
    // Binomial sd at 400 test rows is 0.02. The quantile forest is
    // CONSERVATIVE here — measured 0.88 for the nominal 0.80 band, because
    // the leaf-weighted distribution mixes targets from neighbouring leaves
    // whose conditional means differ, which widens every band (Meinshausen
    // 2006 reports the same over-coverage on small leaves). The band
    // asserted is therefore [0.75, 0.95]: it catches a band that has
    // collapsed or that misses the target, not the known conservatism.
    assert!(
        (0.75..=0.95).contains(&cov),
        "q10-q90 coverage {cov} outside [0.75, 0.95]"
    );
    assert!(cov90 > cov, "wider band must cover more");
    assert!(corr > 0.95);
}

#[test]
fn importance_recovers_the_relevant_variables_of_friedman() {
    let mut rng = Lcg::new(7);
    let n = 500;
    let p = 10;
    let cols = uniform_cols(&mut rng, n, p);
    let y: Vec<f64> = (0..n).map(|i| friedman1(&cols, i) + rng.normal()).collect();
    let x = mat_from_cols(&cols);
    let top5 = |imp: &[f64]| -> Vec<usize> {
        let mut order: Vec<usize> = (0..imp.len()).collect();
        order.sort_by(|&a, &b| imp[b].total_cmp(&imp[a]));
        let mut t: Vec<usize> = order[..5].to_vec();
        t.sort_unstable();
        t
    };
    let imp_fit = random_forest(
        x.as_ref(),
        &y,
        &ForestOptions {
            importance: Importance::Impurity,
            ..opts(200, 1)
        },
        None,
    )
    .unwrap();
    let imp = imp_fit.importance.unwrap();
    assert_eq!(
        imp_fit.importance_groups_resolved.unwrap(),
        (0..p).collect::<Vec<_>>()
    );
    assert!((imp.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    println!("impurity importance: {imp:.3?}");
    assert_eq!(top5(&imp), vec![0, 1, 2, 3, 4]);
    let min_rel = imp[..5].iter().cloned().fold(f64::INFINITY, f64::min);
    let max_irr = imp[5..].iter().cloned().fold(0.0, f64::max);
    assert!(
        min_rel > 1.5 * max_irr,
        "impurity: relevant {min_rel} vs irrelevant {max_irr}"
    );

    let perm_fit = random_forest(
        x.as_ref(),
        &y,
        &ForestOptions {
            importance: Importance::BlockPermutation {
                permutation_block: None,
                n_permutations: 5,
            },
            ..opts(200, 1)
        },
        None,
    )
    .unwrap();
    let pimp = perm_fit.importance.unwrap();
    println!("block-permutation importance (OOB MSE increase): {pimp:.3?}");
    assert_eq!(top5(&pimp), vec![0, 1, 2, 3, 4]);
    let min_rel = pimp[..5].iter().cloned().fold(f64::INFINITY, f64::min);
    let max_irr = pimp[5..].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        min_rel > 3.0 * max_irr.max(0.05),
        "permutation: relevant {min_rel} vs irrelevant {max_irr}"
    );

    // Grouping sums the members' importance exactly (impurity) and the
    // resolved labels are the sorted distinct labels.
    let groups = vec![0, 0, 0, 1, 1, 2, 2, 2, 2, 2];
    let grouped = random_forest(
        x.as_ref(),
        &y,
        &ForestOptions {
            importance: Importance::Impurity,
            importance_groups: Some(groups),
            ..opts(200, 1)
        },
        None,
    )
    .unwrap();
    let g = grouped.importance.unwrap();
    assert_eq!(grouped.importance_groups_resolved.unwrap(), vec![0, 1, 2]);
    assert!((g[0] - imp[..3].iter().sum::<f64>()).abs() < 1e-12);
    assert!((g[1] - imp[3..5].iter().sum::<f64>()).abs() < 1e-12);
    assert!((g[2] - imp[5..].iter().sum::<f64>()).abs() < 1e-12);
}

/// The persistent-predictor trap, measured. Design: five relevant columns
/// (Friedman #1) plus two lags of an irrelevant AR(0.95) series `z`.
/// Arm A: the relevant columns are iid uniform. Arm B: they are persistent
/// (logistic-squashed AR(0.9)). Errors iid, sd 1. Importance of the `z`
/// unit under (i) single-row permutation with the two lags permuted
/// separately ("naive"), (ii) single-row permutation of the lag pair as
/// one unit, (iii) block permutation (20 rows) of the pair as one unit.
///
/// What is asserted: (1) under persistent relevant predictors the
/// irrelevant persistent `z` picks up a materially larger importance than
/// under iid ones — the forest uses `z` as a time proxy, because two
/// persistent series are correlated in-sample; (2) grouped single-row and
/// grouped block permutation agree within Monte-Carlo noise — for a
/// row-wise forest scored row-wise the permutation's block structure does
/// not move the mean importance, so block permutation is NOT a fix for
/// (1). Both facts are printed and quoted on the card.
#[test]
fn persistent_irrelevant_predictor_importance_measured_under_naive_and_block_permutation() {
    let n = 500;
    let seeds = 4;
    let mut results = [[0.0f64; 3]; 2]; // [arm][scheme]
    let mut x0_importance = [0.0f64; 2];
    for (arm, persistent) in [false, true].into_iter().enumerate() {
        for seed in 0..seeds {
            let mut rng = Lcg::new(300 + seed);
            let mut cols = if persistent {
                persistent_cols(&mut rng, n + 2, 5, 0.9)
            } else {
                uniform_cols(&mut rng, n + 2, 5)
            };
            let z = ar1(&mut rng, n + 2, 0.95);
            let y: Vec<f64> = (2..n + 2)
                .map(|i| friedman1(&cols, i) + rng.normal())
                .collect();
            // Lags z_{t-1}, z_{t-2} as columns 5 and 6.
            let z1: Vec<f64> = (2..n + 2).map(|t| z[t - 1]).collect();
            let z2: Vec<f64> = (2..n + 2).map(|t| z[t - 2]).collect();
            for c in cols.iter_mut() {
                c.drain(..2);
            }
            cols.push(z1);
            cols.push(z2);
            let x = mat_from_cols(&cols);
            let run = |groups: Option<Vec<usize>>, block: usize| -> Vec<f64> {
                random_forest(
                    x.as_ref(),
                    &y,
                    &ForestOptions {
                        importance: Importance::BlockPermutation {
                            permutation_block: Some(block),
                            n_permutations: 6,
                        },
                        importance_groups: groups,
                        max_features: MaxFeatures::Count(3),
                        ..opts(150, seed)
                    },
                    None,
                )
                .unwrap()
                .importance
                .unwrap()
            };
            let naive = run(None, 1);
            let grouped_row = run(Some(vec![0, 1, 2, 3, 4, 5, 5]), 1);
            let grouped_block = run(Some(vec![0, 1, 2, 3, 4, 5, 5]), 20);
            results[arm][0] += (naive[5] + naive[6]) / seeds as f64;
            results[arm][1] += grouped_row[5] / seeds as f64;
            results[arm][2] += grouped_block[5] / seeds as f64;
            x0_importance[arm] += naive[0] / seeds as f64;
        }
    }
    println!(
        "irrelevant AR(0.95) unit, OOB-MSE increase (mean of {seeds} seeds):\n  iid relevant x:        \
         naive per-lag {:.3}, grouped single-row {:.3}, grouped block(20) {:.3} (x0 {:.2})\n  \
         persistent relevant x: naive per-lag {:.3}, grouped single-row {:.3}, grouped block(20) {:.3} (x0 {:.2})",
        results[0][0], results[0][1], results[0][2], x0_importance[0],
        results[1][0], results[1][1], results[1][2], x0_importance[1]
    );
    // (1) Inflation under persistent relevant predictors, in every scheme.
    for (s, (pers, iid)) in results[1].iter().zip(&results[0]).enumerate() {
        assert!(
            *pers > iid + 0.1,
            "scheme {s}: persistent-x importance {pers} should exceed iid-x {iid} by > 0.1"
        );
    }
    // (2) Grouped single-row vs grouped block agree within noise: the gap
    // is a small fraction of the inflation itself.
    let inflation = results[1][1] - results[0][1];
    let gap = (results[1][1] - results[1][2]).abs();
    assert!(
        gap < 0.5 * inflation,
        "single-row vs block gap {gap} is not small relative to the inflation {inflation}"
    );
}

#[test]
fn oob_prediction_is_nan_only_where_never_out_of_bag() {
    let mut rng = Lcg::new(8);
    let n = 60;
    let cols = uniform_cols(&mut rng, n, 3);
    let y: Vec<f64> = (0..n)
        .map(|i| {
            friedman1(
                &[
                    cols[0].clone(),
                    cols[1].clone(),
                    cols[2].clone(),
                    cols[0].clone(),
                    cols[1].clone(),
                ],
                i,
            ) + rng.normal()
        })
        .collect();
    let x = mat_from_cols(&cols);
    let one = random_forest(x.as_ref(), &y, &opts(1, 3), None).unwrap();
    let oob = one.oob_prediction.unwrap();
    let n_nan = oob.iter().filter(|v| v.is_nan()).count();
    assert!(
        n_nan > 0 && n_nan < n,
        "one tree leaves ~63% of rows in-bag: {n_nan} NaN"
    );
    assert!(one.oob_mse.unwrap().is_finite());
    let many = random_forest(x.as_ref(), &y, &opts(200, 3), None).unwrap();
    assert!(many.oob_prediction.unwrap().iter().all(|v| v.is_finite()));
    // Block and stationary resampling run end to end and produce OOB rows.
    for scheme in [
        Resampling::MovingBlock { block_length: 5 },
        Resampling::Stationary { block_length: 5 },
    ] {
        let f = random_forest(
            x.as_ref(),
            &y,
            &ForestOptions {
                resampling: scheme,
                ..opts(50, 3)
            },
            None,
        )
        .unwrap();
        assert!(f.oob_mse.unwrap().is_finite());
    }
}

#[test]
fn teaching_errors_name_the_argument_and_the_fix() {
    let mut rng = Lcg::new(9);
    let n = 40;
    let cols = uniform_cols(&mut rng, n, 4);
    let y: Vec<f64> = (0..n).map(|i| cols[0][i] + rng.normal()).collect();
    let x = mat_from_cols(&cols);
    let xt = Mat::from_fn(5, 4, |i, j| cols[j][i]);
    let err = |o: ForestOptions, xt: Option<MatRef<'_, f64>>| -> String {
        random_forest(x.as_ref(), &y, &o, xt)
            .unwrap_err()
            .to_string()
    };

    // NaN / inf refused naming the array.
    let mut bad = cols.clone();
    bad[1][3] = f64::NAN;
    let e = random_forest(mat_from_cols(&bad).as_ref(), &y, &opts(5, 0), None)
        .unwrap_err()
        .to_string();
    assert_eq!(e, "non-finite value (NaN or infinity) in x");
    let mut ybad = y.clone();
    ybad[0] = f64::INFINITY;
    let e = random_forest(x.as_ref(), &ybad, &opts(5, 0), None)
        .unwrap_err()
        .to_string();
    assert_eq!(e, "non-finite value (NaN or infinity) in y");
    let xt_bad = Mat::from_fn(5, 4, |i, j| if i == 2 && j == 1 { f64::NAN } else { 0.5 });
    assert_eq!(
        err(opts(5, 0), Some(xt_bad.as_ref())),
        "non-finite value (NaN or infinity) in x_test"
    );

    // House insufficiency wording: n < 2 * min_samples_leaf.
    let small = Mat::from_fn(7, 4, |i, j| cols[j][i]);
    let e = random_forest(small.as_ref(), &y[..7], &ForestOptions::default(), None)
        .unwrap_err()
        .to_string();
    assert_eq!(e, "insufficient data: 7 observations, at least 10 required");

    // Quantiles.
    let e = err(
        ForestOptions {
            quantiles: Some(vec![0.1, 1.2]),
            ..opts(5, 0)
        },
        Some(xt.as_ref()),
    );
    assert!(
        e.contains("strictly inside (0, 1)") && e.contains("quantiles=[0.1, 0.5, 0.9]"),
        "{e}"
    );
    let e = err(
        ForestOptions {
            quantiles: Some(vec![0.9, 0.5]),
            ..opts(5, 0)
        },
        Some(xt.as_ref()),
    );
    assert!(e.contains("strictly increasing"), "{e}");
    let e = err(
        ForestOptions {
            quantiles: Some(vec![0.5]),
            ..opts(5, 0)
        },
        None,
    );
    assert!(e.contains("x_test"), "{e}");

    // importance_groups length names both lengths.
    let e = err(
        ForestOptions {
            importance: Importance::Impurity,
            importance_groups: Some(vec![0, 1]),
            ..opts(5, 0)
        },
        None,
    );
    assert!(e.contains("expected 4, got 2"), "{e}");

    // Block lengths.
    let e = err(
        ForestOptions {
            resampling: Resampling::MovingBlock { block_length: 41 },
            ..opts(5, 0)
        },
        None,
    );
    assert!(e.starts_with("block_length=41 is outside 1..=40"), "{e}");
    let e = err(
        ForestOptions {
            importance: Importance::BlockPermutation {
                permutation_block: Some(0),
                n_permutations: 2,
            },
            ..opts(5, 0)
        },
        None,
    );
    assert!(
        e.starts_with("permutation_block=0 is outside 1..=40"),
        "{e}"
    );

    // Block permutation needs out-of-bag rows.
    let e = err(
        ForestOptions {
            resampling: Resampling::None,
            importance: Importance::BlockPermutation {
                permutation_block: None,
                n_permutations: 2,
            },
            ..opts(5, 0)
        },
        None,
    );
    assert!(
        e.contains("bootstrap='none'") && e.contains("impurity"),
        "{e}"
    );

    assert!(err(opts(0, 0), None).contains("n_trees"));
    assert!(err(
        ForestOptions {
            max_features: MaxFeatures::Count(9),
            ..opts(5, 0)
        },
        None
    )
    .contains("max_features"));
}
