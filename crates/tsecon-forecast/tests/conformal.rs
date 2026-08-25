//! Conformal-interval tests: exact hand-recomputed pins for the corrected
//! quantile and the split/ACI/EnbPI mechanics, the finite-sample coverage
//! guarantee verified by seeded Monte Carlo exactly where the `+1`
//! correction bites, leakage guards, reproducibility, and the teaching
//! errors on degenerate inputs.

use tsecon_forecast::{
    aci, ar_forecast, conformal_quantile, enbpi, enbpi_online, split_conformal,
    split_conformal_online, AciOptions, EnbpiOptions, ForecastError, SplitConformalOptions,
    SplitOnlineOptions,
};

/// Deterministic pseudo-random N(0,1)-ish draws from a splitmix-style
/// generator (sum of 4 uniforms, CLT-normalized) — seeded, dependency-free.
struct TestRng(u64);

impl TestRng {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        // Sum of 12 uniforms minus 6: mean 0, variance 1, symmetric.
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.uniform();
        }
        s - 6.0
    }
}

fn ar1_series(n: usize, phi: f64, sigma: f64, seed: u64) -> Vec<f64> {
    let mut rng = TestRng(seed);
    let mut y = Vec::with_capacity(n);
    let mut prev = 0.0;
    for _ in 0..n {
        prev = phi * prev + sigma * rng.normal();
        y.push(prev);
    }
    y
}

/// The naive base as a closure: last value repeated.
fn naive_base(train: &[f64], h: usize) -> Result<Vec<f64>, ForecastError> {
    Ok(vec![train[train.len() - 1]; h])
}

// ------------------------------------------------------------------ quantile

#[test]
fn conformal_quantile_exact_indices() {
    // m = 19, alpha = 0.1: k = ceil(20 * 0.9) = 18 -> 18th smallest.
    let scores: Vec<f64> = (1..=19).map(|v| v as f64).collect();
    assert_eq!(conformal_quantile(&scores, 0.1).unwrap(), 18.0);
    // m = 9, alpha = 0.1: k = ceil(10 * 0.9) = 9 (the float-noise case
    // 10 * 0.9 = 9.000000000000002 must NOT round up to 10).
    let nine: Vec<f64> = (1..=9).map(|v| v as f64).collect();
    assert_eq!(conformal_quantile(&nine, 0.1).unwrap(), 9.0);
    // m = 5, alpha = 0.5: k = ceil(6 * 0.5) = 3.
    let five: Vec<f64> = vec![10.0, 30.0, 20.0, 50.0, 40.0]; // order-free
    assert_eq!(conformal_quantile(&five, 0.5).unwrap(), 30.0);
    // m = 20, alpha = 0.05: k = ceil(21 * 0.95) = 20 -> the max.
    let twenty: Vec<f64> = (1..=20).map(|v| v as f64).collect();
    assert_eq!(conformal_quantile(&twenty, 0.05).unwrap(), 20.0);
}

#[test]
fn conformal_quantile_refuses_small_calibration_with_teaching_error() {
    let scores: Vec<f64> = (1..=5).map(|v| v as f64).collect();
    let err = conformal_quantile(&scores, 0.1).unwrap_err();
    match err {
        ForecastError::CalibrationTooSmall {
            n_calib, needed, ..
        } => {
            assert_eq!(n_calib, 5);
            assert_eq!(needed, 9); // ceil(0.9/0.1) = 9 supports alpha = 0.1
        }
        other => panic!("expected CalibrationTooSmall, got {other:?}"),
    }
    let msg = format!("{err}");
    assert!(
        msg.contains("ceil((m+1)(1-alpha))") && msg.contains("at least 9"),
        "error should teach the correction and the minimum, got: {msg}"
    );
    // Degenerate alpha values are their own error.
    assert!(matches!(
        conformal_quantile(&scores, 0.0),
        Err(ForecastError::InvalidAlpha { .. })
    ));
    assert!(matches!(
        conformal_quantile(&scores, 1.0),
        Err(ForecastError::InvalidAlpha { .. })
    ));
    assert!(matches!(
        conformal_quantile(&[1.0, f64::NAN], 0.5),
        Err(ForecastError::NonFinite { .. })
    ));
}

/// The exactness anchor at the primitive level: for iid continuous scores
/// the corrected quantile covers a fresh score with probability exactly
/// k/(m+1) >= 1 - alpha. Measured where the correction bites hardest.
#[test]
fn conformal_quantile_finite_sample_guarantee_monte_carlo() {
    let mut rng = TestRng(20260823);
    for &(m, alpha) in &[(19usize, 0.1), (9usize, 0.1), (39usize, 0.05)] {
        let k = ((m as f64 + 1.0) * (1.0 - alpha)).round(); // exact here
        let exact = k / (m as f64 + 1.0);
        assert!(exact >= 1.0 - alpha - 1e-12);
        let reps = 20_000;
        let mut covered = 0usize;
        for _ in 0..reps {
            let scores: Vec<f64> = (0..m).map(|_| rng.uniform()).collect();
            let q = conformal_quantile(&scores, alpha).unwrap();
            if rng.uniform() <= q {
                covered += 1;
            }
        }
        let cov = covered as f64 / reps as f64;
        let se = (exact * (1.0 - exact) / reps as f64).sqrt();
        assert!(
            (cov - exact).abs() < 5.0 * se,
            "m={m} alpha={alpha}: MC coverage {cov:.4} should match the \
             exact exchangeable value {exact:.4} (se {se:.4})"
        );
        // And in particular the guarantee direction: >= 1 - alpha.
        assert!(
            cov >= 1.0 - alpha - 5.0 * se,
            "m={m}: coverage {cov:.4} fell below the guaranteed {:.4}",
            1.0 - alpha
        );
    }
}

// ------------------------------------------------------------------- split

/// Everything about symmetric split conformal with a naive base is exactly
/// recomputable by hand; this is both a correctness pin and the leakage
/// guard (each score depends only on data up to its origin).
#[test]
fn split_conformal_naive_base_recomputed_exactly() {
    let y = ar1_series(80, 0.7, 1.0, 7);
    let n = y.len();
    let (h_max, m, alpha) = (2usize, 30usize, 0.2);
    let opts = SplitConformalOptions {
        horizon: h_max,
        alpha,
        calib: m,
        symmetric: true,
    };
    let r = split_conformal(&y, &opts, naive_base).unwrap();

    // Origins: the last m origins with all h_max targets in sample:
    // t = n - h_max - m .. n - 1 - h_max.
    for h in 1..=h_max {
        let mut scores = Vec::new();
        for t in (n - h_max - m)..=(n - 1 - h_max) {
            scores.push(y[t + h] - y[t]); // naive: h-step residual vs y[t]
        }
        assert_eq!(
            r.scores[h - 1],
            scores,
            "h={h} scores must be the naive residuals"
        );
        // Corrected quantile of absolute scores, by hand.
        let mut abs: Vec<f64> = scores.iter().map(|s| s.abs()).collect();
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k = ((m as f64 + 1.0) * (1.0 - alpha)).ceil() as usize; // 25
        assert_eq!(k, 25);
        let q = abs[k - 1];
        assert_eq!(r.q_upper[h - 1], q);
        assert_eq!(r.q_lower[h - 1], -q);
        assert_eq!(r.mean[h - 1], y[n - 1]); // naive forward forecast
        assert_eq!(r.lower[h - 1], y[n - 1] - q);
        assert_eq!(r.upper[h - 1], y[n - 1] + q);
    }
    assert_eq!(r.n_calib, m);
    assert!((r.level - 0.8).abs() < 1e-15);
    // finite_sample_level = k/(m+1) = 25/31.
    assert!((r.finite_sample_level - 25.0 / 31.0).abs() < 1e-12);
}

/// The leakage guard from the other direction: perturbing the last
/// observation only moves the pieces that may legally depend on it — the
/// scores whose *target* is y[n-1], and the forward forecast — and no
/// earlier score.
#[test]
fn split_conformal_scores_never_see_the_future() {
    let mut y = ar1_series(60, 0.5, 1.0, 11);
    let opts = SplitConformalOptions {
        horizon: 1,
        alpha: 0.2,
        calib: 20,
        symmetric: true,
    };
    let before = split_conformal(&y, &opts, naive_base).unwrap();
    let last = y.len() - 1;
    y[last] += 100.0;
    let after = split_conformal(&y, &opts, naive_base).unwrap();
    // All scores except the final one (target y[n-1]) are bit-identical.
    let m = opts.calib;
    assert_eq!(before.scores[0][..m - 1], after.scores[0][..m - 1]);
    assert_ne!(before.scores[0][m - 1], after.scores[0][m - 1]);
}

#[test]
fn split_conformal_asymmetric_calibrates_each_tail() {
    // Skewed residuals: a base that under-forecasts by construction.
    let y: Vec<f64> = (0..70).map(|t| t as f64).collect(); // pure trend
    let opts = SplitConformalOptions {
        horizon: 1,
        alpha: 0.1,
        calib: 40,
        symmetric: false,
    };
    // Naive base on a trend: residual y[t+1] - y[t] = 1 at every origin.
    let r = split_conformal(&y, &opts, naive_base).unwrap();
    // All 40 signed residuals are exactly +1: both offsets are +1, so the
    // asymmetric interval is [mean + 1, mean + 1] — it excludes the biased
    // point forecast, which is the mode's purpose.
    assert_eq!(r.q_lower[0], 1.0);
    assert_eq!(r.q_upper[0], 1.0);
    assert_eq!(r.mean[0], 69.0);
    assert_eq!(r.lower[0], 70.0);
    assert_eq!(r.upper[0], 70.0);
    // Symmetric mode on the same data straddles the point forecast.
    let sym = split_conformal(
        &y,
        &SplitConformalOptions {
            symmetric: true,
            ..opts
        },
        naive_base,
    )
    .unwrap();
    assert_eq!(sym.lower[0], 68.0);
    assert_eq!(sym.upper[0], 70.0);
}

#[test]
fn split_conformal_asymmetric_order_statistics_by_hand() {
    let y = ar1_series(100, 0.6, 1.0, 13);
    let (m, alpha) = (40usize, 0.2);
    let opts = SplitConformalOptions {
        horizon: 1,
        alpha,
        calib: m,
        symmetric: false,
    };
    let r = split_conformal(&y, &opts, naive_base).unwrap();
    let mut sorted = r.scores[0].clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k_up = ((m as f64 + 1.0) * (1.0 - alpha / 2.0)).ceil() as usize; // ceil(36.9)=37
    assert_eq!(k_up, 37);
    let k_lo = m + 1 - k_up; // 4
    assert_eq!(r.q_upper[0], sorted[k_up - 1]);
    assert_eq!(r.q_lower[0], sorted[k_lo - 1]);
    assert!(r.q_lower[0] <= r.q_upper[0]);
    // Exchangeable-case exact coverage (k_up - k_lo)/(m+1) = 33/41.
    assert!((r.finite_sample_level - 33.0 / 41.0).abs() < 1e-12);
}

#[test]
fn split_conformal_teaching_errors() {
    let y = ar1_series(60, 0.5, 1.0, 3);
    let base = SplitConformalOptions {
        horizon: 1,
        alpha: 0.1,
        calib: 20,
        symmetric: true,
    };
    // Calibration too small for alpha (needs 9 at alpha = 0.1).
    let err = split_conformal(&y, &SplitConformalOptions { calib: 5, ..base }, |t, h| {
        naive_base(t, h)
    })
    .unwrap_err();
    assert!(matches!(err, ForecastError::CalibrationTooSmall { .. }));
    // Asymmetric needs 2x-per-tail: calib = 12 supports symmetric 0.1 but
    // not the alpha/2 = 0.05 tails (needs 19).
    let err = split_conformal(
        &y,
        &SplitConformalOptions {
            calib: 12,
            symmetric: false,
            ..base
        },
        naive_base,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        ForecastError::CalibrationTooSmall { needed: 19, .. }
    ));
    // horizon = 0, bad alpha, NaN input, series too short.
    assert!(split_conformal(
        &y,
        &SplitConformalOptions { horizon: 0, ..base },
        naive_base
    )
    .is_err());
    assert!(split_conformal(
        &y,
        &SplitConformalOptions { alpha: 0.0, ..base },
        naive_base
    )
    .is_err());
    let mut bad = y.clone();
    bad[10] = f64::NAN;
    assert!(matches!(
        split_conformal(&bad, &base, naive_base),
        Err(ForecastError::NonFinite { .. })
    ));
    assert!(matches!(
        split_conformal(&y[..15], &base, naive_base),
        Err(ForecastError::SeriesTooShort { .. })
    ));
    // A forecaster that returns the wrong number of steps is caught.
    let err = split_conformal(&y, &base, |_t, h| Ok(vec![0.0; h + 1])).unwrap_err();
    assert!(matches!(err, ForecastError::ForecasterOutputLen { .. }));
    // A forecaster error propagates unchanged.
    let err = split_conformal(&y, &base, |_t, _h| {
        Err(ForecastError::BaseForecaster {
            message: "boom".into(),
        })
    })
    .unwrap_err();
    assert!(matches!(err, ForecastError::BaseForecaster { .. }));
}

// ------------------------------------------------------------------- online

#[test]
fn split_online_windows_recomputed_exactly() {
    let y = ar1_series(120, 0.6, 1.0, 17);
    let n = y.len();
    let (calib, n_eval, alpha) = (30usize, 20usize, 0.2);
    let opts = SplitOnlineOptions {
        horizon: 1,
        alpha,
        calib,
        n_eval,
        symmetric: true,
    };
    let r = split_conformal_online(&y, &opts, naive_base).unwrap();
    assert_eq!(r.origins.len(), n_eval);
    // Origins are the last n_eval of the p = calib + n_eval grid ending at
    // n - 1 - 1 (horizon 1).
    let p = calib + n_eval;
    let t0 = n - 1 - p; // first grid origin
    for (j, &t) in r.origins.iter().enumerate() {
        assert_eq!(t, t0 + calib + j);
        // Trailing window: the calib naive 1-step residuals from origins
        // t - calib .. t - 1.
        let mut abs: Vec<f64> = (t - calib..t).map(|u| (y[u + 1] - y[u]).abs()).collect();
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k = ((calib as f64 + 1.0) * (1.0 - alpha)).ceil() as usize; // 25
        let q = abs[k - 1];
        assert_eq!(r.mean[0][j], y[t]);
        assert_eq!(r.lower[0][j], y[t] - q);
        assert_eq!(r.upper[0][j], y[t] + q);
        let target = y[t + 1];
        assert_eq!(r.err[0][j], target < y[t] - q || target > y[t] + q);
    }
    let missed = r.err[0].iter().filter(|&&m| m).count() as f64;
    assert!((r.realized_coverage[0] - (1.0 - missed / n_eval as f64)).abs() < 1e-15);
    assert!(r.alpha_trajectory.is_none());
}

// --------------------------------------------------------------------- ACI

#[test]
fn aci_gamma_zero_is_rolling_split() {
    let y = ar1_series(150, 0.6, 1.0, 23);
    let (calib, n_eval, alpha) = (30usize, 30usize, 0.2);
    let a = aci(
        &y,
        &AciOptions {
            horizon: 1,
            alpha,
            gamma: 0.0,
            calib,
            n_eval,
        },
        naive_base,
    )
    .unwrap();
    let s = split_conformal_online(
        &y,
        &SplitOnlineOptions {
            horizon: 1,
            alpha,
            calib,
            n_eval,
            symmetric: true,
        },
        naive_base,
    )
    .unwrap();
    // With gamma = 0 the trajectory never moves and the online intervals
    // coincide with rolling split conformal exactly.
    let traj = a.online.alpha_trajectory.as_ref().unwrap();
    assert!(traj[0].iter().all(|&v| v == alpha));
    assert_eq!(a.online.lower, s.lower);
    assert_eq!(a.online.upper, s.upper);
    assert_eq!(a.online.err, s.err);
    assert_eq!(a.alpha_final[0], alpha);
}

#[test]
fn aci_trajectory_follows_the_published_recursion_exactly() {
    let y = ar1_series(200, 0.7, 1.0, 29);
    let (calib, n_eval, alpha, gamma) = (30usize, 60usize, 0.1, 0.05);
    let a = aci(
        &y,
        &AciOptions {
            horizon: 1,
            alpha,
            gamma,
            calib,
            n_eval,
        },
        naive_base,
    )
    .unwrap();
    let traj = &a.online.alpha_trajectory.as_ref().unwrap()[0];
    let err = &a.online.err[0];
    // Recompute alpha_{t+1} = alpha_t + gamma (alpha - err_t) with the
    // horizon-1 delay: the alpha used at step j absorbed errors 0..j-1.
    let mut expected = alpha;
    for j in 0..n_eval {
        assert!(
            (traj[j] - expected).abs() < 1e-14,
            "alpha_t at step {j}: {} vs recomputed {expected}",
            traj[j]
        );
        expected += gamma * (alpha - if err[j] { 1.0 } else { 0.0 });
    }
    assert!((a.alpha_final[0] - expected).abs() < 1e-14);
    // Realized coverage consistency.
    let missed = err.iter().filter(|&&m| m).count() as f64;
    assert!((a.online.realized_coverage[0] - (1.0 - missed / n_eval as f64)).abs() < 1e-15);
}

#[test]
fn aci_multi_step_updates_apply_with_delay_h() {
    let y = ar1_series(220, 0.6, 1.0, 31);
    let h_max = 3usize;
    let (calib, n_eval, alpha, gamma) = (40usize, 40usize, 0.1, 0.05);
    let a = aci(
        &y,
        &AciOptions {
            horizon: h_max,
            alpha,
            gamma,
            calib,
            n_eval,
        },
        naive_base,
    )
    .unwrap();
    for h in 1..=h_max {
        let traj = &a.online.alpha_trajectory.as_ref().unwrap()[h - 1];
        let err = &a.online.err[h - 1];
        // The alpha used at step j has absorbed errors 0..=j-h only.
        let mut expected = vec![alpha];
        let mut cur = alpha;
        let mut absorbed = 0usize;
        for j in 1..n_eval {
            while absorbed + h <= j {
                cur += gamma * (alpha - if err[absorbed] { 1.0 } else { 0.0 });
                absorbed += 1;
            }
            expected.push(cur);
        }
        for j in 0..n_eval {
            assert!(
                (traj[j] - expected[j]).abs() < 1e-14,
                "h={h} step {j}: {} vs {}",
                traj[j],
                expected[j]
            );
        }
    }
}

#[test]
fn aci_infinite_interval_when_alpha_t_collapses() {
    // A violent permanent level shift makes the naive-base miss repeatedly;
    // with a large gamma the recursion drives alpha_t below the level the
    // window supports and the interval must go infinite (err = 0), exactly
    // the published convention.
    let mut y = ar1_series(150, 0.3, 0.5, 37);
    let n = y.len();
    for v in y[n - 25..].iter_mut() {
        *v += 1000.0; // the shift
    }
    let a = aci(
        &y,
        &AciOptions {
            horizon: 1,
            alpha: 0.1,
            gamma: 0.5,
            calib: 30,
            n_eval: 40,
        },
        naive_base,
    )
    .unwrap();
    let lower = &a.online.lower[0];
    let upper = &a.online.upper[0];
    let err = &a.online.err[0];
    let inf_at: Vec<usize> = (0..lower.len())
        .filter(|&j| lower[j] == f64::NEG_INFINITY && upper[j] == f64::INFINITY)
        .collect();
    assert!(
        !inf_at.is_empty(),
        "repeated misses with gamma=0.5 must drive alpha_t into the \
         infinite-interval regime"
    );
    for &j in &inf_at {
        assert!(!err[j], "an infinite interval cannot miss");
    }
}

#[test]
fn aci_empty_interval_when_alpha_t_exceeds_one() {
    // Steady coverage with an enormous gamma pushes alpha_t past 1: the
    // interval degenerates to the point forecast and always misses.
    let y = ar1_series(150, 0.3, 0.5, 41);
    let a = aci(
        &y,
        &AciOptions {
            horizon: 1,
            alpha: 0.9,
            gamma: 2.0,
            calib: 30,
            n_eval: 30,
        },
        naive_base,
    )
    .unwrap();
    let traj = &a.online.alpha_trajectory.as_ref().unwrap()[0];
    let some_empty = (0..30).any(|j| {
        traj[j] >= 1.0
            && a.online.lower[0][j] == a.online.mean[0][j]
            && a.online.upper[0][j] == a.online.mean[0][j]
            && a.online.err[0][j]
    });
    assert!(
        some_empty,
        "alpha_t >= 1 must produce the degenerate empty interval that \
         always misses; trajectory head: {:?}",
        &traj[..5.min(traj.len())]
    );
}

#[test]
fn aci_teaching_errors() {
    let y = ar1_series(150, 0.5, 1.0, 43);
    let base = AciOptions {
        horizon: 1,
        alpha: 0.1,
        gamma: 0.005,
        calib: 30,
        n_eval: 20,
    };
    assert!(matches!(
        aci(
            &y,
            &AciOptions {
                gamma: -0.1,
                ..base
            },
            naive_base
        ),
        Err(ForecastError::InvalidConformalParam { what: "gamma", .. })
    ));
    assert!(matches!(
        aci(&y, &AciOptions { n_eval: 0, ..base }, naive_base),
        Err(ForecastError::InvalidConformalParam { what: "n_eval", .. })
    ));
    assert!(matches!(
        aci(&y, &AciOptions { calib: 5, ..base }, naive_base),
        Err(ForecastError::CalibrationTooSmall { .. })
    ));
}

// ------------------------------------------------------------------- EnbPI

#[test]
fn enbpi_is_seed_reproducible_and_seed_sensitive() {
    let y = ar1_series(150, 0.6, 1.0, 47);
    let opts = EnbpiOptions {
        horizon: 3,
        alpha: 0.1,
        lags: 2,
        n_boot: 25,
        seed: 20260823,
        optimize_beta: true,
        n_beta: 21,
    };
    let a = enbpi(&y, &opts).unwrap();
    let b = enbpi(&y, &opts).unwrap();
    assert_eq!(a, b, "same seed must be bit-identical");
    let c = enbpi(&y, &EnbpiOptions { seed: 1, ..opts }).unwrap();
    assert_ne!(
        a.lower, c.lower,
        "a different bootstrap seed must perturb the ensemble"
    );
    assert_eq!(a.mean.len(), 3);
    assert_eq!(a.n_calib + a.n_excluded, y.len() - opts.lags);
    for h in 0..3 {
        assert!(a.lower[h] < a.upper[h]);
        assert!(a.mean[h].is_finite());
    }
    assert!(a.beta.is_some());
    let beta = a.beta.unwrap();
    assert!((0.0..=opts.alpha).contains(&beta));
}

#[test]
fn enbpi_center_tracks_the_ar_fit() {
    // On a well-behaved AR(1), the LOO-aggregated ensemble center must sit
    // close to the plain AR least-squares forecast (they estimate the same
    // regression; the ensemble adds only resampling noise).
    let y = ar1_series(300, 0.7, 1.0, 53);
    let opts = EnbpiOptions {
        horizon: 1,
        alpha: 0.1,
        lags: 1,
        n_boot: 50,
        seed: 5,
        optimize_beta: false,
        n_beta: 21,
    };
    let e = enbpi(&y, &opts).unwrap();
    let ar = ar_forecast(&y, 1, 1).unwrap();
    assert!(
        (e.mean[0] - ar[0]).abs() < 0.15,
        "ensemble center {} vs AR forecast {}",
        e.mean[0],
        ar[0]
    );
    // Symmetric mode straddles the center.
    assert!(e.lower[0] < e.mean[0] && e.mean[0] < e.upper[0]);
    assert!(e.beta.is_none());
}

#[test]
fn enbpi_beta_search_never_widens() {
    // The width-minimizing beta interval is no wider than the beta = 0
    // interval [q_0, q_{1-alpha}] by construction.
    let y = ar1_series(200, 0.5, 1.0, 59);
    let opts = EnbpiOptions {
        horizon: 1,
        alpha: 0.1,
        lags: 1,
        n_boot: 25,
        seed: 2,
        optimize_beta: true,
        n_beta: 21,
    };
    let e = enbpi(&y, &opts).unwrap();
    let mut sorted = e.residuals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = sorted.len();
    let k = ((1.0 - 0.1) * m as f64).ceil() as usize;
    let beta0_width = sorted[k - 1] - sorted[0];
    let width = e.upper[0] - e.lower[0];
    assert!(
        width <= beta0_width + 1e-12,
        "beta search width {width} must not exceed the beta=0 width {beta0_width}"
    );
}

#[test]
fn enbpi_online_slides_the_window_and_scores_honestly() {
    let y = ar1_series(260, 0.6, 1.0, 61);
    let opts = EnbpiOptions {
        horizon: 1,
        alpha: 0.1,
        lags: 2,
        n_boot: 25,
        seed: 7,
        optimize_beta: true,
        n_beta: 21,
    };
    let r = enbpi_online(&y, &opts, 60, 1).unwrap();
    assert_eq!(r.err[0].len(), 60);
    assert_eq!(r.origins.len(), 60);
    // Coverage on a stationary AR(1) should be loosely near nominal; the
    // tight MC grades live in the Python suite.
    assert!(
        r.realized_coverage[0] > 0.7,
        "online coverage collapsed: {}",
        r.realized_coverage[0]
    );
    // err is consistent with the recorded bounds and realized targets:
    // target index = origin + 1 by the batch convention.
    for j in 0..60 {
        let target = y[r.origins[j] + 1];
        assert_eq!(
            r.err[0][j],
            target < r.lower[0][j] || target > r.upper[0][j]
        );
    }
    // Reproducible; batch > 1 also runs.
    let r2 = enbpi_online(&y, &opts, 60, 1).unwrap();
    assert_eq!(r, r2);
    let rb = enbpi_online(&y, &opts, 60, 5).unwrap();
    assert_eq!(rb.err[0].len(), 60);
}

#[test]
fn enbpi_teaching_errors() {
    let y = ar1_series(100, 0.5, 1.0, 67);
    let base = EnbpiOptions {
        horizon: 1,
        alpha: 0.1,
        lags: 1,
        n_boot: 25,
        seed: 0,
        optimize_beta: true,
        n_beta: 21,
    };
    assert!(matches!(
        enbpi(&y, &EnbpiOptions { lags: 0, ..base }),
        Err(ForecastError::InvalidConformalParam { what: "lags", .. })
    ));
    assert!(matches!(
        enbpi(&y, &EnbpiOptions { n_boot: 1, ..base }),
        Err(ForecastError::InvalidConformalParam { what: "n_boot", .. })
    ));
    assert!(matches!(
        enbpi(&y, &EnbpiOptions { n_beta: 1, ..base }),
        Err(ForecastError::InvalidConformalParam { what: "n_beta", .. })
    ));
    assert!(matches!(
        enbpi(&y[..3], &base),
        Err(ForecastError::SeriesTooShort { .. })
    ));
    // A constant series has a collinear lagged design: refuse with the
    // teaching error, not a crash or a zero-width fake interval.
    let flat = vec![5.0; 100];
    assert!(matches!(
        enbpi(&flat, &base),
        Err(ForecastError::SingularArDesign { .. })
    ));
    assert!(matches!(
        enbpi_online(&y, &base, 0, 1),
        Err(ForecastError::InvalidConformalParam { what: "n_eval", .. })
    ));
    assert!(matches!(
        enbpi_online(&y, &base, 10, 0),
        Err(ForecastError::InvalidConformalParam { what: "batch", .. })
    ));
}

// ------------------------------------------------------------- ar_forecast

#[test]
fn ar_forecast_recovers_an_exact_ar1_recursion() {
    // y follows y_t = 2 + 0.8 y_{t-1} EXACTLY: OLS has zero residual, so
    // the coefficients and every forecast step are exact.
    let mut y = vec![0.0f64];
    for _ in 1..30 {
        let prev = *y.last().unwrap();
        y.push(2.0 + 0.8 * prev);
    }
    let f = ar_forecast(&y, 1, 3).unwrap();
    let mut expect = *y.last().unwrap();
    for step in f.iter() {
        expect = 2.0 + 0.8 * expect;
        assert!(
            (step - expect).abs() < 1e-7,
            "forecast {step} vs exact recursion {expect}"
        );
    }
}

#[test]
fn ar_forecast_teaching_errors() {
    let y = ar1_series(50, 0.5, 1.0, 71);
    assert!(matches!(
        ar_forecast(&y, 0, 1),
        Err(ForecastError::InvalidConformalParam { what: "lags", .. })
    ));
    assert!(matches!(
        ar_forecast(&y, 1, 0),
        Err(ForecastError::InvalidSteps { .. })
    ));
    assert!(matches!(
        ar_forecast(&y[..3], 1, 1),
        Err(ForecastError::SeriesTooShort { .. })
    ));
    let flat = vec![1.0; 50];
    assert!(matches!(
        ar_forecast(&flat, 1, 1),
        Err(ForecastError::SingularArDesign { .. })
    ));
}
