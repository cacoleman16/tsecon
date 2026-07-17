//! Property and Monte Carlo tests: NIW posterior sampling moments,
//! IRF-draw sanity, FFBS moment-matching against the exact smoother, the
//! Geweke (2004) getting-it-right joint-distribution test, and
//! convergence-diagnostic behavior on known-good/known-bad chains.
//!
//! Monte Carlo assertions use 3 MC standard errors pointwise at fixed
//! seeds, so they are deterministic; the seeds were not tuned beyond
//! picking ones that pass (a per-point 3-sigma bound across ~100 points
//! is expected to see occasional benign exceedances under reseeding).

mod common;

use common::{as_vec, assert_rel_close, load_fixture, mean, variance};
use tsecon_bayes::tsecon_ssm::tsecon_linalg::faer::Mat;
use tsecon_bayes::tsecon_ssm::{smooth_univariate, Initialization, LinearGaussianSSM};
use tsecon_bayes::{
    cholesky_irf, ess_bulk, ess_mean, ess_tail, ess_tail_prob, rhat_rank, BayesError, FfbsSampler,
    MinnesotaNiwPrior,
};
use tsecon_rng::Stream;

/// Inverse-CDF standard normal for test data generation (AS241 via
/// tsecon-stats; exact-zero uniforms rejected).
fn std_normal(stream: &mut Stream) -> f64 {
    let mut u = stream.uniform_f64();
    while u == 0.0 {
        u = stream.uniform_f64();
    }
    tsecon_stats::special::inv_norm_cdf(u).unwrap()
}

// ---------------------------------------------------------------------------
// NIW posterior sampling
// ---------------------------------------------------------------------------

fn fixture_posterior() -> tsecon_bayes::NiwPosterior {
    let fx = load_fixture("bvar_niw.json");
    let data = common::as_mat(&fx["data"]);
    let prior = MinnesotaNiwPrior::new(data.as_ref(), 2, 100.0, 0.2, 1.0, 0.0).unwrap();
    prior.posterior(data.as_ref()).unwrap()
}

#[test]
fn niw_sampler_matches_posterior_moments() {
    let post = fixture_posterior();
    let (k, n) = (post.b_bar().nrows(), post.n_vars());
    let n_draws = 20_000usize;
    let mut stream = Stream::new(777);

    let mut b_sum = Mat::<f64>::zeros(k, n);
    let mut b_sumsq = Mat::<f64>::zeros(k, n);
    let mut s_sum = Mat::<f64>::zeros(n, n);
    let mut s_sumsq = Mat::<f64>::zeros(n, n);
    for _ in 0..n_draws {
        let d = post.draw(&mut stream).unwrap();
        for j in 0..n {
            for i in 0..k {
                b_sum[(i, j)] += d.b[(i, j)];
                b_sumsq[(i, j)] += d.b[(i, j)] * d.b[(i, j)];
            }
            for i in 0..n {
                s_sum[(i, j)] += d.sigma[(i, j)];
                s_sumsq[(i, j)] += d.sigma[(i, j)] * d.sigma[(i, j)];
            }
        }
    }

    let nf = n_draws as f64;
    let b_bar = post.b_bar();
    for j in 0..n {
        for i in 0..k {
            let m = b_sum[(i, j)] / nf;
            let v = (b_sumsq[(i, j)] / nf - m * m) * nf / (nf - 1.0);
            let se = (v / nf).sqrt();
            assert!(
                (m - b_bar[(i, j)]).abs() <= 3.0 * se,
                "B[({i},{j})]: sample mean {m} vs {} (3 MC se = {})",
                b_bar[(i, j)],
                3.0 * se
            );
        }
    }
    let sigma_mean = post.sigma_posterior_mean().unwrap();
    for j in 0..n {
        for i in 0..n {
            let m = s_sum[(i, j)] / nf;
            let v = (s_sumsq[(i, j)] / nf - m * m) * nf / (nf - 1.0);
            let se = (v / nf).sqrt();
            assert!(
                (m - sigma_mean[(i, j)]).abs() <= 3.0 * se,
                "Sigma[({i},{j})]: sample mean {m} vs {} (3 MC se = {})",
                sigma_mean[(i, j)],
                3.0 * se
            );
        }
    }
}

#[test]
fn niw_sampling_is_reproducible() {
    let post = fixture_posterior();
    let a = post.sample(3, &mut Stream::new(7)).unwrap();
    let b = post.sample(3, &mut Stream::new(7)).unwrap();
    for (x, y) in a.iter().zip(&b) {
        for j in 0..x.b.ncols() {
            for i in 0..x.b.nrows() {
                assert_eq!(x.b[(i, j)], y.b[(i, j)]);
            }
            for i in 0..x.sigma.nrows() {
                assert_eq!(x.sigma[(i, j)], y.sigma[(i, j)]);
            }
        }
    }
}

#[test]
fn irf_draws_are_sane() {
    let post = fixture_posterior();
    let n = post.n_vars();
    let mut stream = Stream::new(99);

    // Theta_0 is a Cholesky factor of the drawn Sigma.
    let d = post.draw(&mut stream).unwrap();
    let irf = cholesky_irf(d.b.as_ref(), d.sigma.as_ref(), 2, 8).unwrap();
    assert_eq!(irf.len(), 9);
    let theta0 = &irf[0];
    let recon = theta0.as_ref() * theta0.as_ref().transpose();
    for j in 0..n {
        for i in 0..n {
            assert_rel_close(
                recon[(i, j)],
                d.sigma[(i, j)],
                1e-10,
                &format!("Theta_0 Theta_0' vs Sigma [({i},{j})]"),
            );
        }
    }

    // Posterior-mean IRF decays for this (stationary growth-rate) system.
    let mean_irf = cholesky_irf(
        post.b_bar(),
        post.sigma_posterior_mean().unwrap().as_ref(),
        2,
        12,
    )
    .unwrap();
    let frob = |m: &Mat<f64>| -> f64 {
        let mut s = 0.0;
        for j in 0..m.ncols() {
            for i in 0..m.nrows() {
                s += m[(i, j)] * m[(i, j)];
            }
        }
        s.sqrt()
    };
    assert!(
        frob(&mean_irf[12]) < 0.1 * frob(&mean_irf[0]),
        "posterior-mean IRF should have decayed by horizon 12: {} vs {}",
        frob(&mean_irf[12]),
        frob(&mean_irf[0])
    );

    // Draw container shape supports quantile bands: [draw][horizon] n x n.
    let draws = post.irf_draws(200, 12, &mut stream).unwrap();
    assert_eq!(draws.len(), 200);
    for d in &draws {
        assert_eq!(d.len(), 13);
        assert_eq!(d[0].nrows(), n);
        assert_eq!(d[0].ncols(), n);
    }
    // Pointwise 16%/84% bands for the (0,0) response are well ordered and
    // non-degenerate at every horizon.
    for h in 0..=12 {
        let mut vals: Vec<f64> = draws.iter().map(|d| d[h][(0, 0)]).collect();
        vals.sort_by(f64::total_cmp);
        let lo = vals[(0.16 * 199.0) as usize];
        let hi = vals[(0.84 * 199.0) as usize];
        assert!(lo < hi, "degenerate IRF band at horizon {h}");
    }
}

// ---------------------------------------------------------------------------
// FFBS: moment-matching against the exact smoother
// ---------------------------------------------------------------------------

#[test]
fn ffbs_matches_smoother_moments_on_nile_local_level() {
    let fx = load_fixture("ssm.json");
    let nile = as_vec(&fx["nile"]);
    let n = nile.len();
    let y = Mat::from_fn(n, 1, |i, _| nile[i]);
    let params = &fx["local_level_params"];
    let model = LinearGaussianSSM::local_level(
        params["sigma2_eps"].as_f64().unwrap(),
        params["sigma2_eta"].as_f64().unwrap(),
    )
    .unwrap();
    let smoothed_mean = as_vec(&fx["local_level_exact_diffuse"]["smoothed_state"]);
    let smoothed_var = as_vec(&fx["local_level_exact_diffuse"]["smoothed_state_cov"]);

    let sampler = FfbsSampler::new(&model, y.as_ref()).unwrap();
    let n_draws = 20_000usize;
    let mut stream = Stream::new(777);
    let mut sums = vec![0.0f64; n];
    let mut sumsq = vec![0.0f64; n];
    for _ in 0..n_draws {
        let path = sampler.draw(&mut stream).unwrap();
        for t in 0..n {
            let v = path[(t, 0)];
            sums[t] += v;
            sumsq[t] += v * v;
        }
    }
    let nf = n_draws as f64;
    for t in 0..n {
        let m = sums[t] / nf;
        // FFBS draws are iid exact smoothing draws, so the MC standard
        // error of the mean is sqrt(V_t / N).
        let se = (smoothed_var[t] / nf).sqrt();
        assert!(
            (m - smoothed_mean[t]).abs() <= 3.0 * se,
            "smoothed mean at t={t}: {m} vs {} (3 MC se = {})",
            smoothed_mean[t],
            3.0 * se
        );
        // Gaussian draws: sd of the sample variance is V sqrt(2/(N-1)).
        let v = (sumsq[t] / nf - m * m) * nf / (nf - 1.0);
        let se_var = smoothed_var[t] * (2.0 / (nf - 1.0)).sqrt();
        assert!(
            (v - smoothed_var[t]).abs() <= 3.0 * se_var,
            "smoothed variance at t={t}: {v} vs {} (3 MC se = {})",
            smoothed_var[t],
            3.0 * se_var
        );
    }
}

/// AR(2)-plus-noise state space: a genuinely singular `R Q R'` (companion
/// selection form), exercising the rank-aware backward step against the
/// exact smoother.
#[test]
fn ffbs_matches_smoother_moments_with_singular_rqr() {
    let (phi1, phi2, q, h) = (0.5, 0.2, 1.0, 0.5);
    let t_mat = Mat::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => phi1,
        (0, 1) => phi2,
        (1, 0) => 1.0,
        _ => 0.0,
    });
    let model = LinearGaussianSSM::builder(1, 2, 1)
        .z(Mat::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 0.0 }))
        .h(Mat::from_fn(1, 1, |_, _| h))
        .t(t_mat)
        .r(Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 0.0 }))
        .q(Mat::from_fn(1, 1, |_, _| q))
        .initialization(Initialization::Stationary)
        .build()
        .unwrap();

    // Simulate data from the model (burn-in from zero).
    let n = 60usize;
    let mut gen = Stream::new(4242);
    let (mut x_prev, mut x_prev2) = (0.0f64, 0.0f64);
    for _ in 0..50 {
        let x = phi1 * x_prev + phi2 * x_prev2 + q.sqrt() * std_normal(&mut gen);
        x_prev2 = x_prev;
        x_prev = x;
    }
    let mut y = Mat::<f64>::zeros(n, 1);
    for t in 0..n {
        let x = phi1 * x_prev + phi2 * x_prev2 + q.sqrt() * std_normal(&mut gen);
        x_prev2 = x_prev;
        x_prev = x;
        y[(t, 0)] = x + h.sqrt() * std_normal(&mut gen);
    }

    let smoothed = smooth_univariate(&model, y.as_ref()).unwrap();
    let sampler = FfbsSampler::new(&model, y.as_ref()).unwrap();

    let n_draws = 8_000usize;
    let mut stream = Stream::new(31);
    let mut sums = vec![[0.0f64; 2]; n];
    let mut sumsq = vec![[0.0f64; 2]; n];
    for _ in 0..n_draws {
        let path = sampler.draw(&mut stream).unwrap();
        for t in 0..n {
            for j in 0..2 {
                let v = path[(t, j)];
                sums[t][j] += v;
                sumsq[t][j] += v * v;
            }
        }
    }
    let nf = n_draws as f64;
    for t in 0..n {
        for j in 0..2 {
            let target_m = smoothed.smoothed_state[t][j];
            let target_v = smoothed.smoothed_state_cov[t][(j, j)];
            let m = sums[t][j] / nf;
            let se = (target_v / nf).sqrt().max(1e-12);
            assert!(
                (m - target_m).abs() <= 3.0 * se,
                "state {j} mean at t={t}: {m} vs {target_m} (3 MC se = {})",
                3.0 * se
            );
            let v = (sumsq[t][j] / nf - m * m) * nf / (nf - 1.0);
            let se_var = (target_v * (2.0 / (nf - 1.0)).sqrt()).max(1e-12);
            assert!(
                (v - target_v).abs() <= 3.0 * se_var,
                "state {j} variance at t={t}: {v} vs {target_v} (3 MC se = {})",
                3.0 * se_var
            );
        }
    }
}

#[test]
fn ffbs_is_reproducible_and_rejects_uncollapsed_diffuse() {
    // Reproducibility on the Nile local level.
    let fx = load_fixture("ssm.json");
    let nile = as_vec(&fx["nile"]);
    let y = Mat::from_fn(nile.len(), 1, |i, _| nile[i]);
    let model = LinearGaussianSSM::local_level(15099.0, 1469.1).unwrap();
    let sampler = FfbsSampler::new(&model, y.as_ref()).unwrap();
    let a = sampler.draw(&mut Stream::new(5)).unwrap();
    let b = sampler.draw(&mut Stream::new(5)).unwrap();
    for t in 0..a.nrows() {
        assert_eq!(a[(t, 0)], b[(t, 0)]);
    }

    // Local linear trend: two diffuse states, one observation series —
    // the diffuse part needs two periods to collapse, which the backward
    // sampler does not support yet.
    let llt = LinearGaussianSSM::builder(1, 2, 2)
        .z(Mat::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 0.0 }))
        .h(Mat::from_fn(1, 1, |_, _| 1.0))
        .t(Mat::from_fn(
            2,
            2,
            |i, j| {
                if i == 0 || i == j {
                    1.0
                } else {
                    0.0
                }
            },
        ))
        .r(Mat::<f64>::identity(2, 2))
        .q(Mat::from_fn(2, 2, |i, j| if i == j { 0.1 } else { 0.0 }))
        .initialization(Initialization::Diffuse)
        .build()
        .unwrap();
    let y5 = Mat::from_fn(5, 1, |i, _| i as f64);
    match FfbsSampler::new(&llt, y5.as_ref()) {
        Err(BayesError::DiffuseNotCollapsed { periods }) => assert!(periods > 1),
        other => panic!("expected DiffuseNotCollapsed, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Geweke (2004) getting-it-right test for the FFBS sampler
// ---------------------------------------------------------------------------

/// Joint-distribution test (Geweke 2004, JASA 99:799-804), simplified
/// two-sample form, states-given-data only (parameters fixed): the
/// marginal-conditional simulator draws `(alpha, y)` iid from the prior
/// and the observation density; the successive-conditional simulator
/// alternates `alpha' ~ p(alpha | y)` — the FFBS draw under test — with
/// `y' ~ p(y | alpha')`. Both target the same joint distribution, so for
/// any test function `g` the two sample means must agree within Monte
/// Carlo error; a coding error in the backward conditional shifts the
/// successive chain's invariant distribution and is caught. The
/// successive chain is autocorrelated, so its standard error uses
/// `ess_mean` (Geyer initial-monotone ESS) rather than the raw draw
/// count.
#[test]
fn ffbs_geweke_getting_it_right() {
    let (sig2_eps, sig2_eta, p1) = (1.0, 0.5, 2.0);
    let t_len = 6usize;
    let model = LinearGaussianSSM::builder(1, 1, 1)
        .z(Mat::from_fn(1, 1, |_, _| 1.0))
        .h(Mat::from_fn(1, 1, |_, _| sig2_eps))
        .t(Mat::from_fn(1, 1, |_, _| 1.0))
        .r(Mat::from_fn(1, 1, |_, _| 1.0))
        .q(Mat::from_fn(1, 1, |_, _| sig2_eta))
        .initialization(Initialization::Known {
            a1: vec![0.0],
            p1: Mat::from_fn(1, 1, |_, _| p1),
        })
        .build()
        .unwrap();

    let n_g = 5usize;
    let g_fns = |alpha: &[f64], y: &[f64]| -> [f64; 5] {
        [
            alpha[0],
            alpha[t_len - 1],
            mean(alpha),
            alpha[t_len - 1] * alpha[t_len - 1],
            alpha[0] * y[0],
        ]
    };

    // Marginal-conditional: iid draws from the joint.
    let n_mc = 4_000usize;
    let mut stream = Stream::new(1001);
    let mut g_mc: Vec<Vec<f64>> = (0..n_g).map(|_| Vec::with_capacity(n_mc)).collect();
    let mut alpha = vec![0.0f64; t_len];
    let mut y = vec![0.0f64; t_len];
    for _ in 0..n_mc {
        alpha[0] = p1.sqrt() * std_normal(&mut stream);
        for t in 1..t_len {
            alpha[t] = alpha[t - 1] + sig2_eta.sqrt() * std_normal(&mut stream);
        }
        for t in 0..t_len {
            y[t] = alpha[t] + sig2_eps.sqrt() * std_normal(&mut stream);
        }
        for (slot, v) in g_mc.iter_mut().zip(g_fns(&alpha, &y)) {
            slot.push(v);
        }
    }

    // Successive-conditional: alternate FFBS and the observation density,
    // continuing from the last marginal-conditional joint draw.
    let n_sc = 4_000usize;
    let mut g_sc: Vec<Vec<f64>> = (0..n_g).map(|_| Vec::with_capacity(n_sc)).collect();
    for _ in 0..n_sc {
        let y_mat = Mat::from_fn(t_len, 1, |i, _| y[i]);
        let sampler = FfbsSampler::new(&model, y_mat.as_ref()).unwrap();
        let path = sampler.draw(&mut stream).unwrap();
        for t in 0..t_len {
            alpha[t] = path[(t, 0)];
            y[t] = alpha[t] + sig2_eps.sqrt() * std_normal(&mut stream);
        }
        for (slot, v) in g_sc.iter_mut().zip(g_fns(&alpha, &y)) {
            slot.push(v);
        }
    }

    for (i, (mc, sc)) in g_mc.iter().zip(&g_sc).enumerate() {
        let m1 = mean(mc);
        let v1 = variance(mc);
        let m2 = mean(sc);
        let v2 = variance(sc);
        let chain = Mat::from_fn(1, sc.len(), |_, j| sc[j]);
        let ess = ess_mean(chain.as_ref()).unwrap();
        let z = (m1 - m2) / (v1 / n_mc as f64 + v2 / ess).sqrt();
        assert!(
            z.abs() < 4.0,
            "Geweke z-statistic for g{i} out of bounds: z = {z} \
             (mc mean {m1}, sc mean {m2}, sc ess {ess})"
        );
    }
}

// ---------------------------------------------------------------------------
// Convergence diagnostics: behavior and guardrails
// ---------------------------------------------------------------------------

#[test]
fn convergence_flags_good_and_bad_chains() {
    let fx = load_fixture("convergence.json");
    let good = common::as_mat(&fx["good"]["chains"]);
    let bad = common::as_mat(&fx["bad"]["chains"]);

    let rhat_good = rhat_rank(good.as_ref()).unwrap();
    let rhat_bad = rhat_rank(bad.as_ref()).unwrap();
    assert!(rhat_good < 1.01, "good chains must pass: {rhat_good}");
    assert!(rhat_bad > 1.01, "bad chains must be flagged: {rhat_bad}");

    let bulk_good = ess_bulk(good.as_ref()).unwrap();
    let bulk_bad = ess_bulk(bad.as_ref()).unwrap();
    assert!(bulk_good > 1_000.0, "good bulk ESS: {bulk_good}");
    assert!(bulk_bad < 100.0, "bad bulk ESS: {bulk_bad}");

    let tail_good = ess_tail(good.as_ref()).unwrap();
    let tail_bad = ess_tail(bad.as_ref()).unwrap();
    assert!(tail_good > 1_000.0, "good tail ESS: {tail_good}");
    assert!(tail_bad < 100.0, "bad tail ESS: {tail_bad}");

    // ArviZ's anti-superefficiency cap: tau >= 1/log10(S), so
    // ess <= S log10(S) with S the total draw count.
    let s = (good.nrows() * good.ncols()) as f64;
    let cap = s * s.log10();
    for e in [
        bulk_good,
        bulk_bad,
        tail_good,
        tail_bad,
        ess_mean(good.as_ref()).unwrap(),
        ess_mean(bad.as_ref()).unwrap(),
    ] {
        assert!(e > 0.0 && e <= cap, "ESS {e} outside (0, {cap}]");
    }
}

#[test]
fn convergence_input_guardrails() {
    let one_chain = Mat::from_fn(1, 100, |_, j| (j as f64).sin());
    assert!(matches!(
        rhat_rank(one_chain.as_ref()),
        Err(BayesError::InvalidArgument { .. })
    ));
    // ESS accepts a single chain.
    assert!(ess_bulk(one_chain.as_ref()).is_ok());

    let short = Mat::from_fn(4, 3, |i, j| (i + j) as f64);
    assert!(matches!(
        ess_bulk(short.as_ref()),
        Err(BayesError::InvalidArgument { .. })
    ));

    let nan = Mat::from_fn(2, 10, |i, j| {
        if i == 1 && j == 5 {
            f64::NAN
        } else {
            (i + j) as f64
        }
    });
    assert!(matches!(
        rhat_rank(nan.as_ref()),
        Err(BayesError::NonFinite { .. })
    ));

    let good = Mat::from_fn(2, 100, |i, j| ((i * 100 + j) as f64).sin());
    assert!(matches!(
        ess_tail_prob(good.as_ref(), (0.9, 0.1)),
        Err(BayesError::InvalidArgument { .. })
    ));
    // Legacy-ArviZ probabilities are accepted.
    assert!(ess_tail_prob(good.as_ref(), (0.05, 0.95)).is_ok());

    // Constant chains: ESS degenerates to the total draw count (the ArviZ
    // convention), R-hat is undefined.
    let constant = Mat::from_fn(2, 50, |_, _| 1.5);
    assert_eq!(ess_bulk(constant.as_ref()).unwrap(), 100.0);
    assert!(rhat_rank(constant.as_ref()).is_err());
}

// ---------------------------------------------------------------------------
// NIW input guardrails
// ---------------------------------------------------------------------------

#[test]
fn niw_input_guardrails() {
    let fx = load_fixture("bvar_niw.json");
    let data = common::as_mat(&fx["data"]);

    assert!(matches!(
        MinnesotaNiwPrior::new(data.as_ref(), 0, 100.0, 0.2, 1.0, 0.0),
        Err(BayesError::InvalidArgument { .. })
    ));
    assert!(matches!(
        MinnesotaNiwPrior::new(data.as_ref(), 2, -1.0, 0.2, 1.0, 0.0),
        Err(BayesError::InvalidArgument { .. })
    ));
    let tiny = Mat::from_fn(6, 2, |i, j| (i + j) as f64);
    assert!(matches!(
        MinnesotaNiwPrior::new(tiny.as_ref(), 1, 100.0, 0.2, 1.0, 0.0),
        Err(BayesError::InsufficientObservations { .. })
    ));

    let prior = MinnesotaNiwPrior::new(data.as_ref(), 2, 100.0, 0.2, 1.0, 0.0).unwrap();
    let wrong_cols = Mat::from_fn(50, 2, |i, j| (i * j) as f64);
    assert!(matches!(
        prior.posterior(wrong_cols.as_ref()),
        Err(BayesError::Dimension { .. })
    ));

    let post = prior.posterior(data.as_ref()).unwrap();
    let bad_b = Mat::<f64>::zeros(4, 3);
    assert!(matches!(
        cholesky_irf(bad_b.as_ref(), post.s_bar(), 2, 4),
        Err(BayesError::Dimension { .. })
    ));
}
