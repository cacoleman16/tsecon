//! Property tests for the threshold-VAR generalized impulse responses —
//! the statistical content a transcription golden cannot prove, on a
//! strongly asymmetric two-regime VAR(1) DGP (k = 3, T = 600: persistent
//! and quiet below the threshold, transient and volatile above it):
//!
//! * **sign asymmetry** — `GIRF(+δ) + GIRF(−δ)` is far from zero relative
//!   to the Monte Carlo error;
//! * **size non-proportionality** — `GIRF(2δ) − 2·GIRF(δ)` likewise;
//! * **regime dependence** — low- and high-regime histories differ by
//!   more than the across-draw Monte Carlo error;
//! * **Monte Carlo convergence** — the history average approaches a
//!   high-draw reference at the `1/sqrt(n_draws)` rate;
//! * **antithetic variance reduction** — measured across seeds (run with
//!   `--nocapture` to see the ratio);
//! * **bit-identity** at any rayon thread count and across two fresh
//!   processes with the same seed;
//! * **the linear reduction** — with both regimes set equal the TVAR GIRF
//!   equals `Psi_h P e_j` from `tsecon_var::ma_rep` at 1e-12 for every
//!   history;
//! * **speed** — the roadmap's showcase configuration (T = 600, k = 3, 200
//!   histories, 500 draws, horizon 20) timed and printed (an indicative
//!   single-machine number, quoted in the model card);
//! * the documented **teaching errors**.

use std::time::Instant;

use tsecon_bootstrap::WildWeights;
use tsecon_regime::{threshold_var, tvar_girf, GirfRegime, RegimeError, TvarGirf, TvarGirfOptions};
use tsecon_rng::Stream;
use tsecon_var::tsecon_linalg::faer::Mat;
use tsecon_var::{ma_rep, GirfShock};

// ------------------------------------------------------------ the DGP

const C_LOW: [f64; 3] = [0.6, 0.2, 0.1];
const A_LOW: [[f64; 3]; 3] = [[0.7, 0.1, 0.0], [0.2, 0.5, 0.1], [0.0, 0.1, 0.6]];
const S_LOW: [[f64; 3]; 3] = [[0.25, 0.05, 0.0], [0.05, 0.25, 0.05], [0.0, 0.05, 0.25]];
const C_HIGH: [f64; 3] = [-0.6, -0.2, -0.1];
const A_HIGH: [[f64; 3]; 3] = [[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.1, 0.0, 0.3]];
const S_HIGH: [[f64; 3]; 3] = [[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]];

fn chol3(s: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut l = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let sum = s[i][j] - (0..j).map(|c| l[i][c] * l[j][c]).sum::<f64>();
            l[i][j] = if i == j { sum.sqrt() } else { sum / l[j][j] };
        }
    }
    l
}

/// Two-regime TVAR(1), regime by `y0_{t-1} <= 0`.
fn sim(stream: &mut Stream, t: usize) -> Vec<Vec<f64>> {
    let burn = 200;
    let (l_low, l_high) = (chol3(&S_LOW), chol3(&S_HIGH));
    let n = t + burn + 1;
    let mut y = vec![[0.0f64; 3]; n];
    for i in 1..n {
        let low = y[i - 1][0] <= 0.0;
        let (c, a, l) = if low {
            (&C_LOW, &A_LOW, &l_low)
        } else {
            (&C_HIGH, &A_HIGH, &l_high)
        };
        let z = [
            WildWeights::Normal.draw(stream),
            WildWeights::Normal.draw(stream),
            WildWeights::Normal.draw(stream),
        ];
        for r in 0..3 {
            let mut v = c[r];
            for cc in 0..3 {
                v += a[r][cc] * y[i - 1][cc] + l[r][cc] * z[cc];
            }
            y[i][r] = v;
        }
    }
    y[(burn + 1)..].iter().map(|r| r.to_vec()).collect()
}

fn data() -> Vec<Vec<f64>> {
    let mut stream = Stream::new(20260912);
    sim(&mut stream, 600)
}

fn opts(
    size: f64,
    n_draws: usize,
    seed: u64,
    regime: GirfRegime,
    histories: Option<usize>,
) -> TvarGirfOptions {
    TvarGirfOptions {
        shock: GirfShock::Orthogonal { var: 0, size },
        horizon: 12,
        n_draws,
        seed,
        antithetic: true,
        bands: (0.16, 0.84),
        regime,
        histories,
    }
}

/// Largest |a − b| / sqrt(se_a² + se_b²) over cells `h >= 1` (impact is
/// deterministic), plus the largest |a − b|.
fn max_t(a: &TvarGirf, b: &TvarGirf, sign_b: f64) -> (f64, f64) {
    let mut t_max = 0.0f64;
    let mut d_max = 0.0f64;
    for h in 1..a.girf.len() {
        for j in 0..a.girf[h].len() {
            let d = a.girf[h][j] + sign_b * b.girf[h][j];
            let se = (a.mc_se[h][j].powi(2) + b.mc_se[h][j].powi(2)).sqrt();
            t_max = t_max.max(d.abs() / se);
            d_max = d_max.max(d.abs());
        }
    }
    (t_max, d_max)
}

// ---------------------------------------------------- the nonlinear facts

#[test]
fn sign_asymmetry_size_nonproportionality_and_regime_dependence() {
    let y = data();
    let fit = threshold_var(&y, 1, 0, &[1], 0.10, true).expect("fit");
    assert!(
        fit.threshold.abs() < 0.3,
        "threshold {} not near 0",
        fit.threshold
    );

    let plus = tvar_girf(&fit, &y, &opts(1.0, 400, 1, GirfRegime::All, Some(120))).expect("girf");
    let minus = tvar_girf(&fit, &y, &opts(-1.0, 400, 1, GirfRegime::All, Some(120))).expect("girf");
    let double = tvar_girf(&fit, &y, &opts(2.0, 400, 1, GirfRegime::All, Some(120))).expect("girf");
    assert_eq!(plus.n_histories, 120);
    assert_eq!(
        plus.history_times, minus.history_times,
        "same seeded subsample"
    );

    // Sign: GIRF(+δ) + GIRF(−δ) would vanish in a linear model.
    let (t_sign, d_sign) = max_t(&plus, &minus, 1.0);
    // Size: GIRF(2δ) − 2·GIRF(δ) would vanish in a linear model.
    let mut two = plus.clone();
    for row in &mut two.girf {
        for v in row.iter_mut() {
            *v *= 2.0;
        }
    }
    for row in &mut two.mc_se {
        for v in row.iter_mut() {
            *v *= 2.0;
        }
    }
    let (t_size, d_size) = max_t(&double, &two, -1.0);

    // Regime: low- vs high-regime histories.
    let low = tvar_girf(&fit, &y, &opts(1.0, 400, 2, GirfRegime::Low, None)).expect("girf");
    let high = tvar_girf(&fit, &y, &opts(1.0, 400, 2, GirfRegime::High, None)).expect("girf");
    assert!(low.history_regimes.iter().all(|&r| r == 0));
    assert!(high.history_regimes.iter().all(|&r| r == 1));
    assert_eq!(low.n_histories + high.n_histories, y.len() - 1);
    let (t_regime, d_regime) = max_t(&low, &high, -1.0);
    let peak = plus
        .girf
        .iter()
        .flatten()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    println!(
        "tvar girf asymmetry (k = 3, T = 600, 400 antithetic draws, 120 histories): \
         peak |GIRF(+1)| = {peak:.4}; sign: max|GIRF(+1)+GIRF(-1)| = {d_sign:.4} \
         (max t = {t_sign:.1}); size: max|GIRF(2)-2GIRF(1)| = {d_size:.4} (max t = {t_size:.1}); \
         regime: max|low-high| = {d_regime:.4} (max t = {t_regime:.1}; {} low, {} high histories)",
        low.n_histories, high.n_histories
    );
    assert!(
        t_sign > 5.0,
        "sign asymmetry {d_sign} not beyond MC error (t = {t_sign})"
    );
    assert!(
        t_size > 5.0,
        "size non-proportionality {d_size} not beyond MC error (t = {t_size})"
    );
    assert!(
        t_regime > 5.0,
        "regime dependence {d_regime} not beyond MC error (t = {t_regime})"
    );
    // The per-regime averages of the "all" run are the same histories'
    // means split by regime, so they must bracket the overall average.
    let all = tvar_girf(&fit, &y, &opts(1.0, 100, 2, GirfRegime::All, None)).expect("girf");
    let gl = all.girf_low_regime.expect("low present");
    let gh = all.girf_high_regime.expect("high present");
    let (nl, nh) = (all.n_low_histories as f64, all.n_high_histories as f64);
    for h in 0..all.girf.len() {
        for j in 0..3 {
            let mix = (gl[h][j] * nl + gh[h][j] * nh) / (nl + nh);
            assert!((mix - all.girf[h][j]).abs() < 1e-12);
        }
    }
    // Across-history bands bracket the average.
    for h in 0..all.girf.len() {
        for j in 0..3 {
            assert!(
                all.lower[h][j] <= all.girf[h][j] + 1e-12
                    || all.upper[h][j] >= all.girf[h][j] - 1e-12
            );
            assert!(all.lower[h][j] <= all.upper[h][j]);
        }
    }
}

// ----------------------------------------------------------- Monte Carlo

fn rms(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut s = 0.0;
    let mut n = 0.0;
    for (ra, rb) in a.iter().zip(b) {
        for (&x, &y) in ra.iter().zip(rb) {
            s += (x - y) * (x - y);
            n += 1.0;
        }
    }
    (s / n).sqrt()
}

#[test]
fn history_average_converges_like_root_n_draws() {
    // The history subsample is keyed by `seed` (one seed drives everything),
    // so to vary only the draws the histories are fixed as every window of a
    // 31-row slice (30 histories).
    let y = data();
    let fit = threshold_var(&y, 1, 0, &[1], 0.10, true).expect("fit");
    let slice = &y[..31];
    let reference =
        tvar_girf(&fit, slice, &opts(1.0, 8192, 999, GirfRegime::All, None)).expect("ref");
    assert_eq!(reference.n_histories, 30);
    let mut errs = Vec::new();
    for n_draws in [32usize, 128, 512] {
        let mut acc = 0.0;
        let seeds = 6;
        for s in 0..seeds {
            let g = tvar_girf(
                &fit,
                slice,
                &opts(1.0, n_draws, 100 + s, GirfRegime::All, None),
            )
            .expect("girf");
            assert_eq!(g.history_times, reference.history_times);
            let e = rms(&g.girf, &reference.girf);
            acc += e * e;
        }
        errs.push((acc / seeds as f64).sqrt());
    }
    println!(
        "tvar girf MC convergence (30 histories, antithetic): rms error n=32 {:.4e}, n=128 {:.4e}, \
         n=512 {:.4e}; ratio 32/512 = {:.2} (1/sqrt(n) predicts 4)",
        errs[0],
        errs[1],
        errs[2],
        errs[0] / errs[2]
    );
    assert!(errs[0] > errs[1] && errs[1] > errs[2]);
    let ratio = errs[0] / errs[2];
    assert!(
        (2.5..=6.5).contains(&ratio),
        "32/512 error ratio {ratio} far from 4"
    );
}

#[test]
fn antithetic_pairs_reduce_the_variance_of_the_history_average() {
    // Histories fixed (every window of a 21-row slice) so only the draws
    // vary across seeds.
    let y = data();
    let fit = threshold_var(&y, 1, 0, &[1], 0.10, true).expect("fit");
    let slice = &y[..21];
    let seeds = 60u64;
    let n_draws = 64;
    let hh = 13;
    let mut plain = vec![Vec::new(); hh * 3];
    let mut anti = vec![Vec::new(); hh * 3];
    for s in 0..seeds {
        let mut o = opts(1.0, n_draws, 700 + s, GirfRegime::All, None);
        o.antithetic = false;
        let gp = tvar_girf(&fit, slice, &o).expect("girf");
        o.antithetic = true;
        let ga = tvar_girf(&fit, slice, &o).expect("girf");
        assert_eq!(gp.n_histories, 20);
        for h in 0..hh {
            for j in 0..3 {
                plain[h * 3 + j].push(gp.girf[h][j]);
                anti[h * 3 + j].push(ga.girf[h][j]);
            }
        }
    }
    let variance = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)
    };
    let (mut num, mut den) = (0.0, 0.0);
    for cell in 3..hh * 3 {
        num += variance(&anti[cell]);
        den += variance(&plain[cell]);
    }
    let ratio = num / den;
    println!(
        "tvar girf antithetic variance ratio ({n_draws} draws, 20 histories, {seeds} seeds, \
         h = 1..12): {ratio:.3} (1 = no reduction)"
    );
    // The measured number is quoted in the model card. Antithetic pairs
    // cancel the odd (linear) component of the paired difference in the
    // innovations; in a threshold model that component is small — the
    // difference is dominated by regime-crossing events, which are close
    // to even functions of the draw — so the reduction is modest at best.
    // The bound pins that the pairs do no harm beyond Monte Carlo noise.
    assert!(
        ratio < 1.15,
        "antithetic pairs materially increased variance: ratio {ratio}"
    );
}

// ---------------------------------------------------------- determinism

#[test]
fn girf_is_bit_identical_at_any_thread_count() {
    let y = data();
    let fit = threshold_var(&y, 1, 0, &[1], 0.10, true).expect("fit");
    let o = opts(1.0, 16, 5, GirfRegime::All, Some(40));
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool");
    let pool4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool");
    let g1 = pool1.install(|| tvar_girf(&fit, &y, &o).expect("girf"));
    let g4 = pool4.install(|| tvar_girf(&fit, &y, &o).expect("girf"));
    assert_eq!(g1, g4, "GIRF must not depend on the rayon thread count");
}

fn digest(g: &TvarGirf) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |v: f64| {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for arr in [
        &g.girf,
        &g.lower,
        &g.upper,
        &g.mc_se,
        &g.draw_sd,
        &g.draw_lower,
        &g.draw_upper,
    ] {
        for row in arr.iter() {
            for &v in row {
                feed(v);
            }
        }
    }
    for path in &g.per_history {
        for row in path {
            for &v in row {
                feed(v);
            }
        }
    }
    for &t in &g.history_times {
        feed(t as f64);
    }
    h
}

#[test]
fn process_replay_is_bit_identical() {
    let y = data();
    let fit = threshold_var(&y, 1, 0, &[1], 0.10, true).expect("fit");
    let g = tvar_girf(&fit, &y, &opts(-1.5, 32, 2026, GirfRegime::Low, Some(25))).expect("girf");
    let here = format!("{:016x}", digest(&g));
    if std::env::var_os("TSECON_GIRF_CHILD").is_some() {
        println!("GIRF_DIGEST {here}");
        return;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args(["process_replay_is_bit_identical", "--exact", "--nocapture"])
        .env("TSECON_GIRF_CHILD", "1")
        .env("RAYON_NUM_THREADS", "3")
        .output()
        .expect("spawn the test binary");
    assert!(
        out.status.success(),
        "child failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let child = text
        .lines()
        .find_map(|l| l.strip_prefix("GIRF_DIGEST "))
        .expect("child printed its digest")
        .trim()
        .to_string();
    assert_eq!(
        child, here,
        "a fresh process must reproduce the GIRF bit for bit"
    );
}

// ------------------------------------------------------ linear reduction

#[test]
fn equal_regimes_reduce_to_the_linear_closed_form() {
    let y = data();
    let fit = threshold_var(&y, 1, 0, &[1], 0.10, true).expect("fit");
    let mut linear = fit.clone();
    linear.coefs_high = linear.coefs_low.clone();
    linear.sigma_high = linear.sigma_low.clone();
    let k = 3;
    let a1 = Mat::from_fn(k, k, |r, c| linear.coefs_low[r][1 + c]);
    let psi = ma_rep(&[a1], 12).expect("ma_rep");
    let sigma = Mat::from_fn(k, k, |i, j| linear.sigma_low[i][j]);
    let l = sigma
        .llt(tsecon_var::tsecon_linalg::faer::Side::Lower)
        .expect("chol")
        .L()
        .to_owned();
    for (var, size, generalized) in [(0usize, 1.0, false), (2, -2.0, false), (1, 1.5, true)] {
        let mut o = opts(size, 4, 3, GirfRegime::All, Some(60));
        o.shock = if generalized {
            GirfShock::Generalized { var, size }
        } else {
            GirfShock::Orthogonal { var, size }
        };
        let g = tvar_girf(&linear, &y, &o).expect("girf");
        let sd = sigma[(var, var)].sqrt();
        for h in 0..=12 {
            for i in 0..k {
                let mut expected = 0.0;
                for c in 0..k {
                    let load = if generalized {
                        sigma[(c, var)] / sd
                    } else {
                        l[(c, var)]
                    };
                    expected += psi[h][(i, c)] * load;
                }
                expected *= size;
                assert!(
                    (g.girf[h][i] - expected).abs() <= 1e-12,
                    "var {var} h {h} i {i}: {} vs {expected}",
                    g.girf[h][i]
                );
                for path in &g.per_history {
                    assert!((path[h][i] - expected).abs() <= 1e-12);
                }
                assert!((g.upper[h][i] - g.lower[h][i]).abs() <= 1e-12);
                assert!(g.draw_sd[h][i].abs() <= 1e-12);
            }
        }
        // Both regimes are still visited by the histories, and both give
        // the same closed form.
        assert!(g.n_low_histories > 0 && g.n_high_histories > 0);
    }
}

// ----------------------------------------------------------------- speed

#[test]
fn showcase_configuration_timing() {
    // T = 600, k = 3, 200 histories, 500 antithetic draws, horizon 20: the
    // number quoted (as indicative) in the model card.
    let y = data();
    let t0 = Instant::now();
    let fit = threshold_var(&y, 1, 0, &[1], 0.10, true).expect("fit");
    let fit_s = t0.elapsed().as_secs_f64();
    let mut o = opts(1.0, 500, 42, GirfRegime::All, Some(200));
    o.horizon = 20;
    let t1 = Instant::now();
    let g = tvar_girf(&fit, &y, &o).expect("girf");
    let girf_s = t1.elapsed().as_secs_f64();
    assert_eq!(g.n_histories, 200);
    assert_eq!(g.n_draws, 500);
    println!(
        "tvar girf speed: T = 600, k = 3, 200 histories x 500 antithetic draws x horizon 20 \
         (2 paths, 4.2M simulated periods): fit {fit_s:.3} s, girf {girf_s:.3} s on {} rayon \
         threads",
        rayon::current_num_threads()
    );
    assert!(girf_s < 60.0, "girf took {girf_s} s");
}

// ------------------------------------------------------------ validation

#[test]
fn malformed_input_raises_teaching_errors() {
    let y = data();
    let fit = threshold_var(&y, 1, 0, &[1], 0.10, true).expect("fit");
    let ok = opts(1.0, 4, 0, GirfRegime::All, Some(10));

    assert!(matches!(
        tvar_girf(&fit, &y, &opts(1.0, 4, 0, GirfRegime::All, Some(0))),
        Err(RegimeError::InvalidParameter {
            name: "histories",
            ..
        })
    ));
    let mut o = ok.clone();
    o.shock = GirfShock::Orthogonal { var: 3, size: 1.0 };
    assert!(matches!(
        tvar_girf(&fit, &y, &o),
        Err(RegimeError::InvalidParameter {
            name: "shock_var",
            ..
        })
    ));
    let mut o = ok.clone();
    o.n_draws = 5;
    assert!(matches!(
        tvar_girf(&fit, &y, &o),
        Err(RegimeError::InvalidParameter {
            name: "n_draws",
            ..
        })
    ));
    o.n_draws = 0;
    assert!(matches!(
        tvar_girf(&fit, &y, &o),
        Err(RegimeError::InvalidParameter {
            name: "n_draws",
            ..
        })
    ));
    let mut o = ok.clone();
    o.bands = (0.9, 0.1);
    assert!(matches!(
        tvar_girf(&fit, &y, &o),
        Err(RegimeError::InvalidParameter { name: "bands", .. })
    ));
    let mut o = ok.clone();
    o.horizon = 5_000_000;
    assert!(matches!(
        tvar_girf(&fit, &y, &o),
        Err(RegimeError::InvalidParameter {
            name: "horizon",
            ..
        })
    ));

    let mut bad = y.clone();
    bad[7][2] = f64::NAN;
    assert!(matches!(
        tvar_girf(&fit, &bad, &ok),
        Err(RegimeError::NonFinite { .. })
    ));
    let narrow: Vec<Vec<f64>> = y.iter().map(|r| r[..2].to_vec()).collect();
    assert!(matches!(
        tvar_girf(&fit, &narrow, &ok),
        Err(RegimeError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        tvar_girf(&fit, &y[..1], &ok),
        Err(RegimeError::InsufficientData { .. })
    ));

    // A slice whose every window sits in the high regime holds no
    // low-regime history: refused, not an empty average.
    let start = (0..y.len() - 6)
        .find(|&a| (a..a + 5).all(|t| y[t][0] > fit.threshold))
        .expect("a run of high-regime rows exists");
    let slice = &y[start..start + 6];
    assert!(matches!(
        tvar_girf(&fit, slice, &opts(1.0, 4, 0, GirfRegime::Low, None)),
        Err(RegimeError::InvalidSpec { .. })
    ));
    let high_only = tvar_girf(&fit, slice, &opts(1.0, 4, 0, GirfRegime::All, None)).expect("girf");
    assert!(high_only.girf_low_regime.is_none());
    assert!(high_only.girf_high_regime.is_some());
    assert_eq!(high_only.n_histories, 5);
}
