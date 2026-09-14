//! Property tests for the Koop-Pesaran-Potter GIRF engine — what a golden
//! transcription cannot prove:
//!
//! * the exact **linear reduction**: on a fitted VAR the engine reproduces
//!   `Psi_h P e_j` / `Psi_h Sigma e_j / sqrt(sigma_jj)` from the crate's
//!   own `ma_rep` at 1e-12 for every history and every draw;
//! * **bit-identity** at any rayon thread count and across two fresh
//!   processes with the same seed (the child runs with a different
//!   `RAYON_NUM_THREADS`);
//! * on a nonlinear toy model: **Monte Carlo convergence** — the error of
//!   the history-average against a high-draw reference shrinks like
//!   `1/sqrt(n_draws)`; **antithetic variance reduction** measured across
//!   seeds (run with `--nocapture` to see the ratio);
//! * the documented **teaching errors** on malformed input, and refusal of
//!   an explosive model rather than a NaN response.

mod common;

use common::Lcg;
use tsecon_linalg::faer::Mat;
use tsecon_var::{
    girf, ma_rep, var_girf, Girf, GirfModel, GirfOptions, GirfShock, LinearVarModel, Trend,
    VarError, VarSpec,
};

// ------------------------------------------------------------- toy model

/// A two-variable, two-regime threshold VAR(1): regime by `y0_{t-1} <= 0`,
/// persistent-and-quiet below, transient-and-volatile above.
struct ToyTvar;

const A: [[[f64; 2]; 2]; 2] = [[[0.8, 0.1], [0.1, 0.7]], [[0.2, 0.0], [0.0, 0.3]]];
const C: [[f64; 2]; 2] = [[0.4, 0.1], [-0.4, -0.1]];
const SIGMA: [[[f64; 2]; 2]; 2] = [[[0.25, 0.05], [0.05, 0.25]], [[1.0, 0.3], [0.3, 1.0]]];

impl GirfModel for ToyTvar {
    fn n_vars(&self) -> usize {
        2
    }
    fn history_len(&self) -> usize {
        1
    }
    fn n_states(&self) -> usize {
        2
    }
    fn state(&self, history: &[f64]) -> usize {
        usize::from(history[0] > 0.0)
    }
    fn covariance(&self, state: usize) -> Vec<Vec<f64>> {
        SIGMA[state].iter().map(|r| r.to_vec()).collect()
    }
    fn step(&self, history: &[f64], innovation: &[f64], out: &mut [f64]) {
        let s = self.state(history);
        for r in 0..2 {
            out[r] = C[s][r] + A[s][r][0] * history[0] + A[s][r][1] * history[1] + innovation[r];
        }
    }
}

/// Histories: the last `n` states of a seeded simulation of the toy model.
fn toy_histories(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Lcg::new(seed);
    let model = ToyTvar;
    let mut y = vec![0.0, 0.0];
    let mut out = vec![0.0, 0.0];
    let mut hist = Vec::with_capacity(n);
    for t in 0..(200 + n) {
        let s = model.state(&y);
        let l = if s == 0 { 0.5 } else { 1.0 };
        let u = [l * rng.gaussian(), l * rng.gaussian()];
        model.step(&y, &u, &mut out);
        y.copy_from_slice(&out);
        if t >= 200 {
            hist.push(y.clone());
        }
    }
    hist
}

fn toy_opts(n_draws: usize, seed: u64, antithetic: bool) -> GirfOptions {
    GirfOptions {
        shock: GirfShock::Orthogonal { var: 0, size: 1.0 },
        horizon: 8,
        n_draws,
        seed,
        antithetic,
        bands: (0.16, 0.84),
    }
}

/// FNV-1a over the bit patterns of every path array (for cross-process
/// comparison).
fn digest(g: &Girf) -> u64 {
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
    h
}

// ---------------------------------------------------- linear VAR data

fn var_data(seed: u64, n: usize) -> Mat<f64> {
    let mut rng = Lcg::new(seed);
    let k = 3;
    let a1 = [[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]];
    let a2 = [[0.1, 0.0, 0.05], [0.0, 0.1, 0.0], [0.05, 0.0, 0.1]];
    let chol = [[0.8, 0.0, 0.0], [0.3, 0.6, 0.0], [-0.2, 0.25, 0.5]];
    let total = n + 100;
    let mut y = vec![[0.0f64; 3]; total];
    for t in 2..total {
        let z = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
        for r in 0..k {
            let mut v = 0.2 * (r as f64 + 1.0);
            for c in 0..k {
                v += a1[r][c] * y[t - 1][c] + a2[r][c] * y[t - 2][c];
                v += chol[r][c] * z[c];
            }
            y[t][r] = v;
        }
    }
    Mat::from_fn(n, k, |i, j| y[100 + i][j])
}

#[test]
fn linear_var_reduces_to_ma_rep_closed_forms_for_every_history_and_draw() {
    let data = var_data(11, 250);
    let res = VarSpec {
        lags: 2,
        trend: Trend::Constant,
    }
    .fit(data.as_ref())
    .expect("fit");
    let k = res.neqs;
    let horizon = 12;
    let psi = ma_rep(&res.coefs, horizon).expect("ma_rep");
    let p_chol = res
        .sigma_u
        .llt(tsecon_linalg::faer::Side::Lower)
        .expect("chol");
    let p_chol = p_chol.L().to_owned();
    for var in 0..k {
        for size in [1.0, -2.5] {
            for generalized in [false, true] {
                let shock = if generalized {
                    GirfShock::Generalized { var, size }
                } else {
                    GirfShock::Orthogonal { var, size }
                };
                let opts = GirfOptions {
                    shock,
                    horizon,
                    n_draws: 6,
                    seed: 5,
                    antithetic: true,
                    bands: (0.05, 0.95),
                };
                let g = var_girf(&res, &opts, Some(25)).expect("girf");
                assert_eq!(g.n_histories, 25);
                let sd = res.sigma_u[(var, var)].sqrt();
                for h in 0..=horizon {
                    for i in 0..k {
                        let mut expected = 0.0;
                        for c in 0..k {
                            let load = if generalized {
                                res.sigma_u[(c, var)] / sd
                            } else {
                                p_chol[(c, var)]
                            };
                            expected += psi[h][(i, c)] * load;
                        }
                        expected *= size;
                        assert!(
                            (g.girf[h][i] - expected).abs() <= 1e-12,
                            "var {var} size {size} gen {generalized} h {h} i {i}: {} vs {expected}",
                            g.girf[h][i]
                        );
                        for path in &g.per_history {
                            assert!((path[h][i] - expected).abs() <= 1e-12);
                        }
                        assert!(g.draw_sd[h][i].abs() <= 1e-12);
                        assert!(g.mc_se[h][i].abs() <= 1e-12);
                    }
                }
            }
        }
    }
}

#[test]
fn linear_model_from_explicit_matrices_matches_the_fitted_route() {
    let data = var_data(3, 200);
    let res = VarSpec {
        lags: 1,
        trend: Trend::None,
    }
    .fit(data.as_ref())
    .expect("fit");
    let k = res.neqs;
    let coefs: Vec<Vec<Vec<f64>>> = res
        .coefs
        .iter()
        .map(|a| {
            (0..k)
                .map(|i| (0..k).map(|j| a[(i, j)]).collect())
                .collect()
        })
        .collect();
    let sigma: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| res.sigma_u[(i, j)]).collect())
        .collect();
    let model = LinearVarModel::new(&coefs, vec![0.0; k], sigma).expect("model");
    // Two independent draws keep every summary finite (a single draw has
    // NaN Monte Carlo errors, and NaN != NaN would defeat the equality).
    let opts = GirfOptions {
        shock: GirfShock::Generalized { var: 2, size: 1.0 },
        horizon: 6,
        n_draws: 2,
        seed: 0,
        antithetic: false,
        bands: (0.16, 0.84),
    };
    let hist = tsecon_var::sample_histories(res.endog.as_ref(), 1);
    let a = girf(&model, &hist, &opts).expect("girf");
    let b = var_girf(&res, &opts, None).expect("var_girf");
    assert_eq!(a, b);
}

// ------------------------------------------------------- determinism

#[test]
fn nonlinear_model_is_bit_identical_at_any_thread_count() {
    let hist = toy_histories(24, 9);
    let opts = toy_opts(16, 77, true);
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool");
    let pool4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool");
    let g1 = pool1.install(|| girf(&ToyTvar, &hist, &opts).expect("girf"));
    let g4 = pool4.install(|| girf(&ToyTvar, &hist, &opts).expect("girf"));
    assert_eq!(g1, g4, "GIRF must not depend on the rayon thread count");
    // A different seed is a different answer (the seed is live).
    let g_other = girf(&ToyTvar, &hist, &toy_opts(16, 78, true)).expect("girf");
    assert_ne!(g1.girf, g_other.girf);
}

#[test]
fn process_replay_is_bit_identical() {
    // The same computation in this process and in a freshly spawned test
    // binary (with a different rayon thread count) must digest identically.
    let hist = toy_histories(20, 4);
    let g = girf(&ToyTvar, &hist, &toy_opts(32, 2024, true)).expect("girf");
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

// ------------------------------------------------------- Monte Carlo

/// Root-mean-square distance between two `[h][j]` paths.
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
    let hist = toy_histories(16, 21);
    let reference = girf(&ToyTvar, &hist, &toy_opts(16_384, 999, true)).expect("ref");
    let mut errs = Vec::new();
    for n_draws in [32usize, 128, 512] {
        let mut acc = 0.0;
        let seeds = 6;
        for s in 0..seeds {
            let g = girf(&ToyTvar, &hist, &toy_opts(n_draws, 100 + s, true)).expect("girf");
            let e = rms(&g.girf, &reference.girf);
            acc += e * e;
        }
        errs.push((acc / seeds as f64).sqrt());
    }
    println!(
        "girf MC convergence (toy TVAR, 16 histories): rms error n=32 {:.4e}, n=128 {:.4e}, \
         n=512 {:.4e}; ratio 32/512 = {:.2} (1/sqrt(n) predicts 4)",
        errs[0],
        errs[1],
        errs[2],
        errs[0] / errs[2]
    );
    assert!(
        errs[0] > errs[1] && errs[1] > errs[2],
        "error must fall with n_draws"
    );
    let ratio = errs[0] / errs[2];
    assert!(
        (2.5..=6.5).contains(&ratio),
        "32/512 error ratio {ratio} far from 4"
    );
}

#[test]
fn antithetic_pairs_reduce_the_variance_of_the_history_average() {
    let hist = toy_histories(12, 33);
    let seeds = 80u64;
    let n_draws = 64;
    let mut var_ratio_num = 0.0;
    let mut var_ratio_den = 0.0;
    let cells = 9 * 2;
    let mut plain = vec![Vec::new(); cells];
    let mut anti = vec![Vec::new(); cells];
    for s in 0..seeds {
        let gp = girf(&ToyTvar, &hist, &toy_opts(n_draws, 500 + s, false)).expect("girf");
        let ga = girf(&ToyTvar, &hist, &toy_opts(n_draws, 500 + s, true)).expect("girf");
        for h in 0..9 {
            for j in 0..2 {
                plain[h * 2 + j].push(gp.girf[h][j]);
                anti[h * 2 + j].push(ga.girf[h][j]);
            }
        }
    }
    let variance = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)
    };
    for cell in 0..cells {
        // Skip impact (h = 0): the paired difference there is the shock
        // itself, deterministic under both schemes.
        if cell < 2 {
            continue;
        }
        var_ratio_num += variance(&anti[cell]);
        var_ratio_den += variance(&plain[cell]);
    }
    let ratio = var_ratio_num / var_ratio_den;
    println!(
        "girf antithetic variance ratio (toy TVAR, {n_draws} draws, {seeds} seeds, h = 1..8): \
         {ratio:.3} (1 = no reduction)"
    );
    // Measured 0.93 on this seed set: the reduction is real but modest on
    // a threshold model, whose paired difference is dominated by
    // regime-crossing events that are not odd functions of the innovation
    // (the linear component antithetic pairs cancel is small). The bound
    // pins the direction; the model card quotes the number.
    assert!(
        ratio < 1.0,
        "antithetic pairs failed to reduce variance: ratio {ratio}"
    );
}

// ------------------------------------------------------- validation

struct Explosive;

impl GirfModel for Explosive {
    fn n_vars(&self) -> usize {
        1
    }
    fn history_len(&self) -> usize {
        1
    }
    fn n_states(&self) -> usize {
        1
    }
    fn state(&self, _: &[f64]) -> usize {
        0
    }
    fn covariance(&self, _: usize) -> Vec<Vec<f64>> {
        vec![vec![1.0]]
    }
    fn step(&self, history: &[f64], innovation: &[f64], out: &mut [f64]) {
        out[0] = 10.0 * history[0] + innovation[0];
    }
}

struct BadState;

impl GirfModel for BadState {
    fn n_vars(&self) -> usize {
        1
    }
    fn history_len(&self) -> usize {
        1
    }
    fn n_states(&self) -> usize {
        1
    }
    fn state(&self, _: &[f64]) -> usize {
        1
    }
    fn covariance(&self, _: usize) -> Vec<Vec<f64>> {
        vec![vec![1.0]]
    }
    fn step(&self, history: &[f64], innovation: &[f64], out: &mut [f64]) {
        out[0] = 0.5 * history[0] + innovation[0];
    }
}

#[test]
fn malformed_input_raises_teaching_errors() {
    let hist = toy_histories(4, 1);
    let ok = toy_opts(4, 0, true);

    assert!(matches!(
        girf(&ToyTvar, &[], &ok),
        Err(VarError::InvalidArgument { .. })
    ));
    assert!(matches!(
        girf(&ToyTvar, &[vec![1.0]], &ok),
        Err(VarError::Dimension {
            expected: 2,
            got: 1,
            ..
        })
    ));
    assert!(matches!(
        girf(&ToyTvar, &[vec![f64::NAN, 0.0]], &ok),
        Err(VarError::NonFinite { .. })
    ));

    let mut o = ok.clone();
    o.shock = GirfShock::Orthogonal { var: 2, size: 1.0 };
    assert!(matches!(
        girf(&ToyTvar, &hist, &o),
        Err(VarError::InvalidParameter {
            name: "shock_var",
            ..
        })
    ));
    o.shock = GirfShock::Generalized {
        var: 0,
        size: f64::INFINITY,
    };
    assert!(matches!(
        girf(&ToyTvar, &hist, &o),
        Err(VarError::InvalidParameter { name: "size", .. })
    ));
    o.shock = GirfShock::Raw(vec![1.0]);
    assert!(matches!(
        girf(&ToyTvar, &hist, &o),
        Err(VarError::Dimension { .. })
    ));

    let mut o = ok.clone();
    o.n_draws = 0;
    assert!(matches!(
        girf(&ToyTvar, &hist, &o),
        Err(VarError::InvalidParameter {
            name: "n_draws",
            ..
        })
    ));
    o.n_draws = 3;
    assert!(matches!(
        girf(&ToyTvar, &hist, &o),
        Err(VarError::InvalidParameter {
            name: "n_draws",
            ..
        })
    ));
    o.antithetic = false;
    assert!(
        girf(&ToyTvar, &hist, &o).is_ok(),
        "odd draws are fine without antithetic"
    );

    let mut o = ok.clone();
    o.horizon = 2_000_000;
    assert!(matches!(
        girf(&ToyTvar, &hist, &o),
        Err(VarError::InvalidParameter {
            name: "horizon",
            ..
        })
    ));
    let mut o = ok.clone();
    o.bands = (0.5, 0.5);
    assert!(matches!(
        girf(&ToyTvar, &hist, &o),
        Err(VarError::InvalidParameter { name: "bands", .. })
    ));
    o.bands = (-0.1, 0.9);
    assert!(matches!(
        girf(&ToyTvar, &hist, &o),
        Err(VarError::InvalidParameter { name: "bands", .. })
    ));
    // Absurd draw counts are refused before any allocation (the memory
    // budget, 0.10.0: the refusal now carries the counts and the budget
    // instead of a bare InvalidArgument).
    let mut o = ok.clone();
    o.n_draws = 1 << 40;
    let err = girf(&ToyTvar, &hist, &o).unwrap_err();
    assert!(matches!(err, VarError::MemoryBudget { .. }), "{err}");
    let text = err.to_string();
    for name in ["n_draws", "horizon", "histories", "budget"] {
        assert!(text.contains(name), "{text}");
    }

    // An explosive model overflows: refused, never a NaN response.
    let mut o = ok.clone();
    o.horizon = 400;
    o.shock = GirfShock::Orthogonal { var: 0, size: 1.0 };
    assert!(matches!(
        girf(&Explosive, &[vec![0.1]], &o),
        Err(VarError::NonFinite { .. })
    ));
    // A state index beyond n_states is caught.
    assert!(matches!(
        girf(&BadState, &[vec![0.1]], &ok),
        Err(VarError::InvalidArgument { .. })
    ));

    // The fitted-VAR route: histories = Some(0) and a VAR(0).
    let data = var_data(2, 60);
    let res = VarSpec {
        lags: 1,
        trend: Trend::Constant,
    }
    .fit(data.as_ref())
    .expect("fit");
    assert!(matches!(
        var_girf(&res, &ok, Some(0)),
        Err(VarError::InvalidArgument { .. })
    ));
    let res0 = VarSpec {
        lags: 0,
        trend: Trend::Constant,
    }
    .fit(data.as_ref())
    .expect("fit");
    assert!(matches!(
        var_girf(&res0, &ok, None),
        Err(VarError::InvalidArgument { .. })
    ));
}
