//! Internal validation of the sign-restriction sampler against a simulated
//! DGP with a known structural impact matrix (no external fixture exists for
//! this scheme). The checks are:
//!
//! * (a) coverage: for a DGP whose true impact matrix satisfies known signs,
//!   the identified-set min/max bands cover the true structural IRF;
//! * (b) infeasibility: restrictions the model can never satisfy produce
//!   zero acceptance, exposed by the diagnostics;
//! * (c) bit-exact reproducibility at a fixed seed;
//! * (d) accepted-set invariance to `max_tries` batching at a fixed seed.

use tsecon_bayes::{cholesky_irf, MinnesotaNiwPrior};
use tsecon_ident::{Sign, SignRestriction, SignRestrictionSet, SignSampler};
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_rng::Stream;
use tsecon_stats::special::inv_norm_cdf;

const N: usize = 3;

/// True VAR(1) autoregressive matrix `Phi` (stable) and lower-triangular
/// impact matrix `A0` with a strictly positive first column (shock 0 raises
/// every variable on impact). Because `A0` is lower triangular with a
/// positive diagonal, `chol(A0 A0') = A0`, so the true structural shock 0 is
/// exactly the first Cholesky shock and its IRF is `cholesky_irf` column 0.
fn true_params() -> ([[f64; N]; N], [[f64; N]; N]) {
    let phi = [[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]];
    let a0 = [[1.0, 0.0, 0.0], [0.5, 0.9, 0.0], [0.3, 0.2, 0.7]];
    (phi, a0)
}

/// One standard normal from the stream (inverse-CDF transform).
fn std_normal(s: &mut Stream) -> f64 {
    loop {
        let u = s.uniform_f64();
        if u > 0.0 {
            return inv_norm_cdf(u).expect("inv_norm_cdf");
        }
    }
}

/// Simulates `t_obs` observations of `y_t = Phi y_{t-1} + A0 eps_t` after a
/// burn-in, returning a `t_obs x N` data matrix.
fn simulate(t_obs: usize, seed: u64) -> Mat<f64> {
    let (phi, a0) = true_params();
    let burn = 200;
    let mut s = Stream::new(seed);
    let mut y = [0.0f64; N];
    let mut data = Mat::<f64>::zeros(t_obs, N);
    for t in 0..(burn + t_obs) {
        let eps: [f64; N] = std::array::from_fn(|_| std_normal(&mut s));
        let mut shock = [0.0f64; N];
        for i in 0..N {
            for k in 0..N {
                shock[i] += a0[i][k] * eps[k];
            }
        }
        let mut next = [0.0f64; N];
        for i in 0..N {
            let mut v = shock[i];
            for j in 0..N {
                v += phi[i][j] * y[j];
            }
            next[i] = v;
        }
        y = next;
        if t >= burn {
            for i in 0..N {
                data[(t - burn, i)] = y[i];
            }
        }
    }
    data
}

/// True structural IRF (column 0 = shock 0) to `horizon`, built from the
/// true parameters via `cholesky_irf`.
fn true_irf(horizon: usize) -> Vec<Mat<f64>> {
    let (phi, a0) = true_params();
    let mut b = Mat::<f64>::zeros(1 + N, N);
    for i in 0..N {
        for v in 0..N {
            b[(1 + v, i)] = phi[i][v];
        }
    }
    let sigma = Mat::from_fn(N, N, |i, j| {
        a0[i].iter().zip(a0[j].iter()).map(|(x, y)| x * y).sum()
    });
    cholesky_irf(b.as_ref(), sigma.as_ref(), 1, horizon).expect("true irf")
}

fn build_posterior(data: MatRef<'_, f64>) -> tsecon_bayes::NiwPosterior {
    // A deliberately loose prior so the T = 800 sample dominates and the
    // posterior concentrates near the true parameters.
    let prior = MinnesotaNiwPrior::new(data, 1, 100.0, 10.0, 1.0, 0.0).expect("prior");
    prior.posterior(data).expect("posterior")
}

/// Positive impact restrictions: shock 0 raises all three variables at
/// horizon 0 (the true sign pattern).
fn positive_impact_set(horizon: usize) -> SignRestrictionSet {
    SignRestrictionSet::new(
        vec![
            SignRestriction::at(0, 0, 0, Sign::Positive),
            SignRestriction::at(1, 0, 0, Sign::Positive),
            SignRestriction::at(2, 0, 0, Sign::Positive),
        ],
        N,
        horizon,
    )
    .expect("restriction set")
}

#[test]
fn identified_set_bands_cover_the_true_irf() {
    let horizon = 8;
    let data = simulate(800, 424242);
    let posterior = build_posterior(data.as_ref());
    let restr = positive_impact_set(horizon);

    let sampler = SignSampler::new(horizon, 600, 400).expect("sampler");
    let result = sampler.run(&posterior, &restr, 12345).expect("run");

    let diag = result.diagnostics();
    assert!(
        diag.accepted > 300,
        "expected a healthy accepted set, got {} of {}",
        diag.accepted,
        diag.posterior_draws_used
    );
    assert!(diag.acceptance_rate > 0.0);

    let truth = true_irf(horizon);
    let summary = result.summary();
    // Shock 0 is the restricted, sign-identified shock; its true IRF must
    // lie inside the identified-set envelope [min, max] at every horizon and
    // for every response variable.
    for (h, truth_h) in truth.iter().enumerate() {
        for i in 0..N {
            let t = truth_h[(i, 0)];
            let band = summary.point(i, 0, h).expect("band point");
            let scale = t.abs().max(1.0);
            let tol = 0.05 * scale;
            assert!(
                t >= band.min - tol && t <= band.max + tol,
                "coverage failure var {i} horizon {h}: true {t} not in [{}, {}]",
                band.min,
                band.max
            );
        }
    }
}

#[test]
fn infeasible_restrictions_yield_zero_acceptance() {
    let horizon = 4;
    let data = simulate(400, 424242);
    let posterior = build_posterior(data.as_ref());

    // Same cell required to be both positive and negative: no rotation and
    // no sign choice can ever satisfy this. This is the degenerate limit of
    // "restrictions contradicting the DGP" — acceptance is exactly zero and
    // the diagnostics expose it.
    let restr = SignRestrictionSet::new(
        vec![
            SignRestriction::at(0, 0, 0, Sign::Positive),
            SignRestriction::at(0, 0, 0, Sign::Negative),
        ],
        N,
        horizon,
    )
    .expect("restriction set");

    let n_draws = 40;
    let max_tries = 25;
    let sampler = SignSampler::new(horizon, n_draws, max_tries).expect("sampler");
    let result = sampler.run(&posterior, &restr, 999).expect("run");

    let diag = result.diagnostics();
    assert_eq!(
        diag.accepted, 0,
        "infeasible restrictions accepted something"
    );
    assert_eq!(diag.acceptance_rate, 0.0);
    // Every draw exhausted its full try budget without success.
    assert_eq!(diag.rotations_tried, n_draws * max_tries);
    assert!(result.draws().is_empty());
    // Summary cells are NaN when nothing was accepted.
    let band = result.summary().point(0, 0, 0).expect("band");
    assert!(band.min.is_nan() && band.max.is_nan());
}

#[test]
fn same_seed_is_bit_exact() {
    let horizon = 6;
    let data = simulate(400, 424242);
    let posterior = build_posterior(data.as_ref());
    let restr = positive_impact_set(horizon);
    let sampler = SignSampler::new(horizon, 200, 200).expect("sampler");

    let a = sampler.run(&posterior, &restr, 77).expect("run a");
    let b = sampler.run(&posterior, &restr, 77).expect("run b");

    assert_eq!(a.diagnostics().accepted, b.diagnostics().accepted);
    assert_eq!(
        a.diagnostics().rotations_tried,
        b.diagnostics().rotations_tried
    );
    assert_eq!(a.draws().len(), b.draws().len());
    for (da, db) in a.draws().iter().zip(b.draws().iter()) {
        for (ma, mb) in da.iter().zip(db.iter()) {
            for i in 0..N {
                for j in 0..N {
                    assert_eq!(ma[(i, j)].to_bits(), mb[(i, j)].to_bits());
                }
            }
        }
    }
}

#[test]
fn accepted_set_is_invariant_to_max_tries_batching() {
    let horizon = 6;
    let data = simulate(400, 424242);
    let posterior = build_posterior(data.as_ref());
    let restr = positive_impact_set(horizon);

    // Both budgets are large enough that (with the ~all-positive acceptance
    // rate here) every draw succeeds well within budget. The per-draw
    // substream design then guarantees each draw settles on the same first
    // accepted rotation regardless of the budget.
    let small = SignSampler::new(horizon, 200, 150)
        .expect("sampler")
        .run(&posterior, &restr, 2024)
        .expect("run small");
    let large = SignSampler::new(horizon, 200, 500)
        .expect("sampler")
        .run(&posterior, &restr, 2024)
        .expect("run large");

    assert_eq!(small.draws().len(), large.draws().len());
    assert_eq!(
        small.draws().len(),
        200,
        "some draw failed within the budget"
    );
    for (ds, dl) in small.draws().iter().zip(large.draws().iter()) {
        for (ms, ml) in ds.iter().zip(dl.iter()) {
            for i in 0..N {
                for j in 0..N {
                    assert_eq!(ms[(i, j)].to_bits(), ml[(i, j)].to_bits());
                }
            }
        }
    }

    // The pointwise median band is therefore identical too.
    for h in 0..=horizon {
        for i in 0..N {
            let bs = small.summary().point(i, 0, h).expect("band");
            let bl = large.summary().point(i, 0, h).expect("band");
            assert_eq!(bs.min.to_bits(), bl.min.to_bits());
            assert_eq!(bs.max.to_bits(), bl.max.to_bits());
        }
    }
}
