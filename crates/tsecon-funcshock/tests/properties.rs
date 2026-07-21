//! Property tests: invariants of the functional-shock estimators that hold
//! independently of any external reference, plus a seeded Monte Carlo of the
//! statistical claim (the scenario response recovers `integral B(m) delta(m)`
//! for a known functional `B`). Exact numbers live in `golden.rs`.

use std::f64::consts::PI;

use tsecon_funcshock::{
    flp, flp_scenario, functional_pca, fvar_scenario, scenario_response, scenario_weights,
};
use tsecon_rng::Stream;

/// Standard-normal draws via Box-Muller on the reproducible uniform stream.
fn normals(stream: &mut Stream, n: usize) -> Vec<f64> {
    (0..n)
        .map(|_| {
            let u1 = 1.0 - stream.uniform_f64(); // in (0, 1]
            let u2 = stream.uniform_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        })
        .collect()
}

fn ar1(stream: &mut Stream, n: usize, rho: f64, sd: f64) -> Vec<f64> {
    let e = normals(stream, n);
    let mut x = vec![0.0; n];
    x[0] = sd * e[0] / (1.0 - rho * rho).sqrt();
    for t in 1..n {
        x[t] = rho * x[t - 1] + sd * e[t];
    }
    x
}

/// A T x M curve panel driven by level/slope factors plus small noise.
fn curve_panel(stream: &mut Stream, t: usize, m: usize) -> Vec<Vec<f64>> {
    let f1 = ar1(stream, t, 0.6, 1.0);
    let f2 = ar1(stream, t, 0.4, 0.7);
    let noise = normals(stream, t * m);
    (0..t)
        .map(|tt| {
            (0..m)
                .map(|mm| {
                    let grid = mm as f64 / (m - 1).max(1) as f64;
                    f1[tt] + f2[tt] * (1.0 - grid) + 0.05 * noise[tt * m + mm]
                })
                .collect()
        })
        .collect()
}

#[test]
fn eigenfunctions_are_orthonormal_and_variance_splits() {
    let mut s = Stream::new(2026_0721);
    let t = 150;
    let m = 8;
    let curves = curve_panel(&mut s, t, m);
    let k = 4;
    let r = functional_pca(&curves, k).expect("fpca");

    // Orthonormality in the discrete inner product.
    for i in 0..k {
        for j in 0..k {
            let dot: f64 = r.eigenfunctions[i]
                .iter()
                .zip(r.eigenfunctions[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            let want = f64::from(u8::from(i == j));
            assert!(
                (dot - want).abs() < 1e-10,
                "<phi_{i}, phi_{j}> = {dot}, want {want}"
            );
        }
    }

    // Eigenvalues descending, nonnegative; explained shares in (0, 1],
    // descending, and summing to <= 1 (the trailing M-K carry the rest).
    for w in r.eigenvalues.windows(2) {
        assert!(w[0] >= w[1] - 1e-12, "eigenvalues not descending: {w:?}");
    }
    assert!(r.eigenvalues.iter().all(|l| *l >= -1e-12));
    let share_sum: f64 = r.explained.iter().sum();
    assert!(
        share_sum > 0.9,
        "2-factor DGP: leading 4 shares explain most"
    );
    assert!(share_sum <= 1.0 + 1e-12);
    assert!(r.total_variance > 0.0);

    // Score columns: mean ~ 0, population variance == eigenvalue, and
    // uncorrelated across components (diagonal covariance).
    let tf = t as f64;
    for a in 0..k {
        let mean: f64 = r.scores.iter().map(|row| row[a]).sum::<f64>() / tf;
        assert!(mean.abs() < 1e-10, "score {a} mean {mean}");
        for b in a..k {
            let cov: f64 = r.scores.iter().map(|row| row[a] * row[b]).sum::<f64>() / tf;
            let want = if a == b { r.eigenvalues[a] } else { 0.0 };
            assert!(
                (cov - want).abs() < 1e-9,
                "score cov({a},{b}) = {cov}, want {want}"
            );
        }
    }
}

#[test]
fn curves_reconstruct_from_full_rank_scores() {
    // With K = M the eigenfunctions span the grid: mean + sum_k s_k phi_k
    // reproduces every curve to machine precision.
    let mut s = Stream::new(7_070_707);
    let t = 60;
    let m = 6;
    let curves = curve_panel(&mut s, t, m);
    let r = functional_pca(&curves, m).expect("fpca");
    for (tt, curve) in curves.iter().enumerate() {
        for (mm, want) in curve.iter().enumerate() {
            let recon: f64 = r.mean_curve[mm]
                + (0..m)
                    .map(|k| r.scores[tt][k] * r.eigenfunctions[k][mm])
                    .sum::<f64>();
            assert!(
                (recon - want).abs() < 1e-9,
                "reconstruction[{tt}][{mm}]: {recon} vs {want}"
            );
        }
    }
}

#[test]
fn eigenfunction_scenario_reproduces_that_scores_coefficient_path_exactly() {
    // The EXACT reconstruction identity: a scenario equal to the j-th
    // eigenfunction has weights e_j (by orthonormality), so the scenario
    // response IS the beta_j path and the scenario SE IS sqrt(cov_jj),
    // bit-near-exactly.
    let mut s = Stream::new(31_415_926);
    let t = 200;
    let m = 7;
    let k = 3;
    let curves = curve_panel(&mut s, t, m);
    let fpca = functional_pca(&curves, k).expect("fpca");
    let e = normals(&mut s, t);
    let y: Vec<f64> = (0..t)
        .map(|tt| 0.4 * fpca.scores[tt][0] - 0.2 * fpca.scores[tt][1] + 0.3 * e[tt])
        .collect();
    let fit = flp(&y, &fpca.scores, 5, 2, None).expect("flp");

    for j in 0..k {
        let delta = fpca.eigenfunctions[j].clone();
        let w = scenario_weights(&fpca.eigenfunctions, &delta).expect("weights");
        for (i, wi) in w.iter().enumerate() {
            let want = f64::from(u8::from(i == j));
            assert!((wi - want).abs() < 1e-10, "w[{i}] = {wi}, want {want}");
        }
        let irf = flp_scenario(&fpca, &fit, &delta).expect("scenario");
        for h in 0..fit.betas.len() {
            assert!(
                (irf.response[h] - fit.betas[h][j]).abs() < 1e-10,
                "response[{h}] vs beta[{h}][{j}]"
            );
            let se_j = fit.covs[h][j * k + j].sqrt();
            assert!(
                (irf.se[h] - se_j).abs() < 1e-10,
                "se[{h}] vs sqrt(cov[{h}][{j},{j}])"
            );
        }
    }
}

#[test]
fn scenario_response_is_linear_in_the_scenario() {
    // response(a*d1 + b*d2) == a*response(d1) + b*response(d2): projection
    // and w'beta are both linear.
    let mut s = Stream::new(9_090_909);
    let t = 150;
    let m = 6;
    let k = 3;
    let curves = curve_panel(&mut s, t, m);
    let fpca = functional_pca(&curves, k).expect("fpca");
    let e = normals(&mut s, t);
    let y: Vec<f64> = (0..t).map(|tt| 0.5 * fpca.scores[tt][0] + e[tt]).collect();
    let fit = flp(&y, &fpca.scores, 4, 1, None).expect("flp");

    let d1: Vec<f64> = (0..m).map(|i| 1.0 + 0.1 * i as f64).collect();
    let d2: Vec<f64> = (0..m).map(|i| (0.9 * i as f64).sin()).collect();
    let combo: Vec<f64> = d1
        .iter()
        .zip(d2.iter())
        .map(|(a, b)| 2.0 * a - 0.5 * b)
        .collect();

    let r1 = flp_scenario(&fpca, &fit, &d1).expect("d1");
    let r2 = flp_scenario(&fpca, &fit, &d2).expect("d2");
    let rc = flp_scenario(&fpca, &fit, &combo).expect("combo");
    for h in 0..rc.response.len() {
        let want = 2.0 * r1.response[h] - 0.5 * r2.response[h];
        assert!(
            (rc.response[h] - want).abs() < 1e-9,
            "linearity at h={h}: {} vs {want}",
            rc.response[h]
        );
    }
}

#[test]
fn mc_flp_scenario_recovers_the_known_functional() {
    // DGP: y_{t+1} responds to the whole curve through a KNOWN functional
    // B(m): y_{t+1} = sum_m B[m] * Xc_t[m] + eps. The horizon-1 FLP scenario
    // response to delta must recover <B, delta> (delta lies in the factor
    // span up to the small measurement noise), within MC tolerance.
    let mut s = Stream::new(1_234_567);
    let t = 4000;
    let m = 10;
    let k = 3;
    let curves = curve_panel(&mut s, t, m);

    let b: Vec<f64> = (0..m).map(|i| 0.3 - 0.08 * i as f64).collect();
    let mean: Vec<f64> = (0..m)
        .map(|i| curves.iter().map(|c| c[i]).sum::<f64>() / t as f64)
        .collect();
    let e = normals(&mut s, t);
    let mut y = vec![0.0_f64; t];
    for tt in 1..t {
        let signal: f64 = (0..m)
            .map(|mm| b[mm] * (curves[tt - 1][mm] - mean[mm]))
            .sum();
        y[tt] = signal + 0.2 * e[tt];
    }

    let fpca = functional_pca(&curves, k).expect("fpca");
    let fit = flp(&y, &fpca.scores, 2, 1, None).expect("flp");

    // Scenario: a level shift plus a tilt (inside the level/slope span).
    let delta: Vec<f64> = (0..m)
        .map(|i| 1.0 + 0.5 * (1.0 - i as f64 / (m - 1) as f64))
        .collect();
    let irf = flp_scenario(&fpca, &fit, &delta).expect("scenario");
    let truth: f64 = b.iter().zip(delta.iter()).map(|(bi, di)| bi * di).sum();

    let got = irf.response[1];
    assert!(
        (got - truth).abs() < 0.05 * (1.0 + truth.abs()),
        "MC recovery at h=1: got {got}, truth {truth}"
    );
    // And the truth lies well inside the 4-sigma band.
    assert!(
        (got - truth).abs() < 4.0 * irf.se[1],
        "truth outside 4-sigma: got {got} +- {}, truth {truth}",
        irf.se[1]
    );
}

#[test]
fn fvar_impact_scores_equal_weights_and_match_flp_sign() {
    let mut s = Stream::new(55_555);
    let t = 400;
    let m = 8;
    let k = 3;
    let curves = curve_panel(&mut s, t, m);
    let fpca = functional_pca(&curves, k).expect("fpca");
    let e = normals(&mut s, t);
    let mut y = vec![0.0_f64; t];
    for tt in 1..t {
        y[tt] = 0.3 * y[tt - 1] + 0.6 * fpca.scores[tt][0] + 0.3 * e[tt];
    }

    let delta: Vec<f64> = (0..m).map(|i| 0.8 + 0.05 * i as f64).collect();
    let w = scenario_weights(&fpca.eigenfunctions, &delta).expect("weights");
    let r = fvar_scenario(&fpca.scores, &y, &w, 2, 6).expect("fvar");

    // Impact identity: the score block responds by exactly w at h = 0
    // (Theta_0 = P and P[..K,..K] z = w by construction).
    for (j, wj) in w.iter().enumerate().take(k) {
        assert!(
            (r.responses[0][j] - wj).abs() < 1e-10,
            "impact score {j}: {} vs {wj}",
            r.responses[0][j]
        );
    }
    assert_eq!(r.response_outcome.len(), 7);
    assert!(
        (r.implied_outcome_innovation - r.response_outcome[0]).abs() < 1e-12,
        "implied innovation is the impact outcome response"
    );
    // y loads positively on score 0 within the period in this DGP, and the
    // scenario weight on phi_0 is positive, so the impact response is > 0.
    assert!(w[0] > 0.0);
    assert!(r.response_outcome[0] > 0.0);
}

#[test]
fn negative_variance_is_rejected_but_hac_psd_is_accepted() {
    // scenario_response guards against a non-PSD covariance; the genuine
    // Bartlett-HAC covariances never trip it (checked across seeds).
    for seed in [1_u64, 2, 3] {
        let mut s = Stream::new(seed);
        let t = 120;
        let m = 5;
        let curves = curve_panel(&mut s, t, m);
        let fpca = functional_pca(&curves, 2).expect("fpca");
        let e = normals(&mut s, t);
        let y: Vec<f64> = (0..t).map(|tt| fpca.scores[tt][0] * 0.4 + e[tt]).collect();
        let fit = flp(&y, &fpca.scores, 6, 2, None).expect("flp");
        let delta = vec![1.0; m];
        let irf = flp_scenario(&fpca, &fit, &delta).expect("psd covariances accepted");
        assert!(irf.se.iter().all(|v| v.is_finite() && *v >= 0.0));

        // A hand-broken covariance is rejected with the teaching error.
        let mut broken = fit.clone();
        broken.covs[0] = vec![-1.0, 0.0, 0.0, -1.0];
        let err = scenario_response(&broken, &irf.weights).unwrap_err();
        assert!(
            err.to_string().contains("negative scenario variance"),
            "unexpected error: {err}"
        );
    }
}
