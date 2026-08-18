//! Seeded Monte-Carlo coverage for the weak-instrument-robust (Anderson-Rubin)
//! proxy-SVAR confidence sets, under a STRONG and a WEAK instrument.
//!
//! The weak arm is the whole point of the method, so it is the arm that has to
//! pass. A confidence set can look perfectly reasonable — bounded, tidy,
//! centred on the estimate — and still cover 84% of the time when the
//! instrument is weak; this library's own interval-coverage audit measured
//! exactly that for `iv_gmm`. The only way to know is to measure it.
//!
//! # Two families of arm
//!
//! The KNOWN-`Psi` arms feed the TRUE reduced-form innovations `u_t = H eps_t`
//! and the TRUE MA matrices computed from the known VAR(2) coefficients. That
//! is deliberate: [`ArVariance::Hc0`] estimates the variance of the
//! identification step only, so this is the setting in which it is supposed to
//! be right, and a shortfall there is a defect in the set itself.
//!
//! The ESTIMATED-`Psi` arms fit the VAR by OLS on simulated data, which is the
//! only configuration a real caller is ever in, and run each replication
//! twice: once with reduced-form uncertainty omitted and once with it
//! propagated through [`ArReducedForm`]. Omitting it is not a drift but a
//! collapse — measured `0.119` at `h = 8` with a strong instrument, against a
//! nominal `0.95` — and these arms pin both the collapse and its repair so
//! neither the code nor the numbers in the `proxy_ar` module docs can rot
//! silently.
//!
//! # The estimand does not move between the arms
//!
//! `gamma = E[m_t u_t] = phi * H[:, 0]`, so the relevance factor `phi` cancels
//! in `lambda = unit * (Psi_h gamma)_i / gamma_k`. Shrinking `phi` weakens the
//! instrument without touching the truth, which is what makes the two arms
//! directly comparable.

use tsecon_ident::proxy_ar::{
    proxy_ar_sets, psi_reduced_form_cov, psi_reduced_form_cov_mc, ArCritical, ArReducedForm, ArSet,
    ArVariance, ArVarianceSpec,
};
use tsecon_ident::IdentError;
use tsecon_linalg::faer::Mat;
use tsecon_rng::Stream;
use tsecon_stats::special::inv_norm_cdf;

const N: usize = 3;
const HORIZON: usize = 4;
const NORM_VAR: usize = 0;
const UNIT: f64 = 1.0;
const LEVEL: f64 = 0.95;
/// Lag order of the fitted VAR in the estimated-`Psi` arms — the same order
/// the DGP uses.
const LAGS: usize = 2;

/// Known invertible impact matrix; the target shock is column 0.
const H_TRUE: [[f64; N]; N] = [[1.0, 0.4, 0.2], [0.5, 1.2, 0.3], [0.3, 0.5, 0.9]];
const A1: [[f64; N]; N] = [[0.50, 0.10, 0.00], [0.00, 0.40, 0.10], [0.10, 0.00, 0.30]];
const A2: [[f64; N]; N] = [[0.10, 0.00, 0.00], [0.00, 0.10, 0.00], [0.00, 0.00, 0.10]];

/// One standard normal by inverse-CDF transform of the library stream.
fn std_normal(s: &mut Stream) -> f64 {
    loop {
        let u = s.uniform_f64();
        if u > 0.0 {
            if let Ok(z) = inv_norm_cdf(u) {
                return z;
            }
        }
    }
}

/// `Psi_0 = I`, `Psi_h = sum_{i=1..min(h,2)} Psi_{h-i} A_i`.
fn true_ma(horizon: usize) -> Vec<Mat<f64>> {
    let a: [[[f64; N]; N]; 2] = [A1, A2];
    let mut psi = vec![Mat::<f64>::identity(N, N)];
    for h in 1..=horizon {
        let mut acc = Mat::<f64>::zeros(N, N);
        for i in 1..=h.min(2) {
            for r in 0..N {
                for c in 0..N {
                    let mut s = 0.0;
                    for k in 0..N {
                        s += psi[h - i][(r, k)] * a[i - 1][k][c];
                    }
                    acc[(r, c)] += s;
                }
            }
        }
        psi.push(acc);
    }
    psi
}

/// The population `lambda_{i,h}`.
///
/// Built from `hcol / hcol[norm_var]` so the `(norm_var, h = 0)` truth is
/// EXACTLY `unit` — the same number the degenerate cell's point set carries,
/// which makes that cell's coverage a genuine exact-arithmetic check rather
/// than a rounding coin flip.
fn truth(psi: &[Mat<f64>]) -> Vec<Vec<f64>> {
    let hcol: [f64; N] = std::array::from_fn(|i| H_TRUE[i][0]);
    let ratio: [f64; N] = std::array::from_fn(|i| hcol[i] / hcol[NORM_VAR]);
    psi.iter()
        .map(|ph| {
            (0..N)
                .map(|i| {
                    let mut s = 0.0;
                    for (j, &r) in ratio.iter().enumerate() {
                        s += ph[(i, j)] * r;
                    }
                    UNIT * s
                })
                .collect()
        })
        .collect()
}

struct ArmResult {
    ar_coverage: f64,
    ar_coverage_by_h: Vec<f64>,
    wald_coverage: f64,
    bounded_fraction: f64,
    reps: usize,
}

/// One Monte-Carlo arm. `phi` is the proxy's relevance for the target shock;
/// `sig_nu` its measurement noise.
fn run_arm(seed: u64, reps: usize, t_obs: usize, phi: f64, sig_nu: f64) -> ArmResult {
    let psi = true_ma(HORIZON);
    let tru = truth(&psi);
    let z = inv_norm_cdf(0.5 + 0.5 * LEVEL).expect("normal quantile");

    let mut hit = vec![vec![0usize; N]; HORIZON + 1];
    let mut hit_wald = 0usize;
    let mut bounded = 0usize;
    let mut cells_total = 0usize;
    let mut used = 0usize;

    let mut stream = Stream::new(seed);
    for _ in 0..reps {
        let mut u = Mat::<f64>::zeros(t_obs, N);
        let mut proxy = vec![0.0f64; t_obs];
        for r in 0..t_obs {
            let eps: [f64; N] = std::array::from_fn(|_| std_normal(&mut stream));
            for i in 0..N {
                let mut s = 0.0;
                for (k, &e) in eps.iter().enumerate() {
                    s += H_TRUE[i][k] * e;
                }
                u[(r, i)] = s;
            }
            proxy[r] = phi * eps[0] + sig_nu * std_normal(&mut stream);
        }

        let res = match proxy_ar_sets(
            u.as_ref(),
            &proxy,
            &psi,
            NORM_VAR,
            UNIT,
            ArVariance::Hc0,
            ArCritical::Chi2 { level: LEVEL },
        ) {
            Ok(r) => r,
            Err(_) => continue,
        };
        used += 1;
        let no = res.n_proxy as f64;

        for (h, row) in res.cells.iter().enumerate() {
            for (i, cell) in row.iter().enumerate() {
                cells_total += 1;
                if cell.set.is_bounded() {
                    bounded += 1;
                }
                let target = tru[h][i];
                let covered = match cell.set {
                    ArSet::Point(p) => (p - target).abs() <= 1e-12 * (1.0 + p.abs()),
                    other => other.contains(target),
                };
                if covered {
                    hit[h][i] += 1;
                }
                // The delta-method Wald interval for the same ratio: the
                // variance FROZEN at the point estimate. This is the object
                // the AR set replaces; it is bounded in every replication, at
                // every instrument strength, which is exactly what Dufour
                // (1997) says cannot be valid.
                let v = cell.variance(cell.point).max(0.0);
                let se = (v / no).sqrt() / cell.q0.abs();
                if (target - cell.point).abs() <= z * se {
                    hit_wald += 1;
                }
            }
        }
    }

    let ncell = (HORIZON + 1) * N;
    let by_h: Vec<f64> = hit
        .iter()
        .map(|row| row.iter().sum::<usize>() as f64 / (N * used) as f64)
        .collect();
    let total: usize = hit.iter().flatten().sum();
    let out = ArmResult {
        ar_coverage: total as f64 / (ncell * used) as f64,
        ar_coverage_by_h: by_h,
        wald_coverage: hit_wald as f64 / cells_total as f64,
        bounded_fraction: bounded as f64 / cells_total as f64,
        reps: used,
    };
    // Visible with `cargo test -- --nocapture`. A coverage harness whose
    // numbers can only be read by making it fail is a harness nobody reads.
    eprintln!(
        "  arm(seed={seed:#x}, reps={}, T={t_obs}, phi={phi}, sig_nu={sig_nu}): \
         AR={:.4} Wald={:.4} bounded={:.3} by_h={:?}",
        out.reps,
        out.ar_coverage,
        out.wald_coverage,
        out.bounded_fraction,
        out.ar_coverage_by_h
            .iter()
            .map(|x| (x * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );
    out
}

/// STRONG INSTRUMENT. Coverage should sit at nominal, every set should be
/// bounded, and the Wald interval should be fine too — under strong
/// identification the two agree, which is why a strong-only harness proves
/// nothing about weak-instrument robustness.
#[test]
fn strong_instrument_arm_covers_at_nominal() {
    let arm = run_arm(0x5A1E_0001, 1200, 300, 1.0, 1.5);
    assert_eq!(arm.reps, 1200, "no replication should fail here");
    assert!(
        arm.ar_coverage > 0.93 && arm.ar_coverage < 0.97,
        "strong-arm AR coverage {:.4} is off nominal {LEVEL}; by horizon {:?}",
        arm.ar_coverage,
        arm.ar_coverage_by_h
    );
    assert_eq!(
        arm.bounded_fraction, 1.0,
        "a strong instrument must give bounded sets everywhere"
    );
    // Flat across horizons: with Psi known there is no horizon-dependent
    // omission, so a monotone decay here would be a real defect.
    for (h, &cov) in arm.ar_coverage_by_h.iter().enumerate() {
        assert!(
            cov > 0.93,
            "strong-arm coverage {cov:.4} at h={h} (all: {:?})",
            arm.ar_coverage_by_h
        );
    }
}

/// WEAK INSTRUMENT. This is the test that matters. Coverage must hold at
/// nominal even though the denominator moment is barely distinguishable from
/// zero — and the sets must be honest about it by coming back UNBOUNDED most
/// of the time. A procedure that returned bounded sets here could not be
/// valid (Dufour 1997), so "bounded fraction near 1" is a failure even if
/// coverage looks fine.
#[test]
fn weak_instrument_arm_covers_and_reports_unbounded_sets() {
    let arm = run_arm(0x5A1E_0002, 1200, 300, 0.06, 1.5);
    assert!(
        arm.reps > 1150,
        "too many failed replications: {}",
        arm.reps
    );
    assert!(
        arm.ar_coverage > 0.92 && arm.ar_coverage < 0.975,
        "weak-arm AR coverage {:.4} is off nominal {LEVEL}; by horizon {:?}",
        arm.ar_coverage,
        arm.ar_coverage_by_h
    );
    assert!(
        arm.bounded_fraction < 0.30,
        "weak-arm bounded fraction {:.3} is too high: a bounded set under weak \
         identification cannot have valid coverage",
        arm.bounded_fraction
    );
    // The Wald interval is bounded in 100% of cells by construction, and its
    // coverage misses nominal by more than the AR set's does. Which DIRECTION
    // it misses is DGP-dependent — with a near-zero denominator the
    // delta-method standard error explodes, so here it over-covers — but it is
    // wrong either way, and it is wrong while looking perfectly tidy.
    let ar_miss = (arm.ar_coverage - LEVEL).abs();
    let wald_miss = (arm.wald_coverage - LEVEL).abs();
    assert!(
        wald_miss > ar_miss,
        "weak arm: Wald coverage {:.4} is no worse than AR coverage {:.4}",
        arm.wald_coverage,
        arm.ar_coverage
    );
}

// ---------------------------------------------------------------------------
// ESTIMATED REDUCED FORM — the configuration every real caller is in
// ---------------------------------------------------------------------------

/// Gauss-Jordan inverse of a small square matrix with partial pivoting.
/// Written out here rather than pulled from a solver so the harness has no
/// dependency on the code path it is measuring.
fn inverse(a: &Mat<f64>) -> Mat<f64> {
    let n = a.nrows();
    let mut m = Mat::<f64>::from_fn(n, 2 * n, |i, j| {
        if j < n {
            a[(i, j)]
        } else {
            f64::from(u8::from(i + n == j))
        }
    });
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if m[(r, col)].abs() > m[(piv, col)].abs() {
                piv = r;
            }
        }
        assert!(m[(piv, col)] != 0.0, "singular matrix in the test harness");
        if piv != col {
            for j in 0..2 * n {
                let t = m[(col, j)];
                m[(col, j)] = m[(piv, j)];
                m[(piv, j)] = t;
            }
        }
        let d = m[(col, col)];
        for j in 0..2 * n {
            m[(col, j)] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[(r, col)];
            if f == 0.0 {
                continue;
            }
            for j in 0..2 * n {
                m[(r, j)] -= f * m[(col, j)];
            }
        }
    }
    Mat::from_fn(n, n, |i, j| m[(i, j + n)])
}

/// A simulated `y` path from the same VAR(2), plus the target structural
/// shock `eps_0` on the same dates, both trimmed of the burn-in.
fn simulate_levels(stream: &mut Stream, t_obs: usize) -> (Mat<f64>, Vec<f64>) {
    let burn = 200usize;
    let total = t_obs + burn;
    let mut y = Mat::<f64>::zeros(total, N);
    let mut shock = vec![0.0f64; total];
    for r in 0..total {
        let eps: [f64; N] = std::array::from_fn(|_| std_normal(stream));
        let mut u = [0.0f64; N];
        for (i, slot) in u.iter_mut().enumerate() {
            let mut s = 0.0;
            for (k, &e) in eps.iter().enumerate() {
                s += H_TRUE[i][k] * e;
            }
            *slot = s;
        }
        for i in 0..N {
            let mut s = u[i];
            if r >= 1 {
                for j in 0..N {
                    s += A1[i][j] * y[(r - 1, j)];
                }
            }
            if r >= 2 {
                for j in 0..N {
                    s += A2[i][j] * y[(r - 2, j)];
                }
            }
            y[(r, i)] = s;
        }
        shock[r] = eps[0];
    }
    let out = Mat::from_fn(t_obs, N, |i, j| y[(burn + i, j)]);
    (out, shock[burn..].to_vec())
}

/// OLS VAR(`LAGS`) with a constant. Returns residuals, `A_1..A_p`, and
/// `Cov(vec alpha_hat)` for the lag block in the `r = a*N + e` layout
/// [`psi_reduced_form_cov`] documents.
fn fit_var(y: &Mat<f64>) -> (Mat<f64>, Vec<Mat<f64>>, Mat<f64>) {
    let t_all = y.nrows();
    let t = t_all - LAGS;
    let k = 1 + N * LAGS;
    let z = Mat::from_fn(t, k, |r, c| {
        if c == 0 {
            1.0
        } else {
            let lag = (c - 1) / N + 1;
            let var = (c - 1) % N;
            y[(LAGS + r - lag, var)]
        }
    });
    let zz = Mat::from_fn(k, k, |i, j| {
        let mut s = 0.0;
        for r in 0..t {
            s += z[(r, i)] * z[(r, j)];
        }
        s
    });
    let zz_inv = inverse(&zz);
    let zy = Mat::from_fn(k, N, |i, e| {
        let mut s = 0.0;
        for r in 0..t {
            s += z[(r, i)] * y[(LAGS + r, e)];
        }
        s
    });
    let beta = Mat::from_fn(k, N, |i, e| {
        let mut s = 0.0;
        for l in 0..k {
            s += zz_inv[(i, l)] * zy[(l, e)];
        }
        s
    });
    let resid = Mat::from_fn(t, N, |r, e| {
        let mut s = y[(LAGS + r, e)];
        for i in 0..k {
            s -= z[(r, i)] * beta[(i, e)];
        }
        s
    });
    let dof = (t - k) as f64;
    let sigma_u = Mat::from_fn(N, N, |a, b| {
        let mut s = 0.0;
        for r in 0..t {
            s += resid[(r, a)] * resid[(r, b)];
        }
        s / dof
    });
    // coefs[l][(e, j)] is the coefficient of variable j at lag l+1 in eq e.
    let coefs: Vec<Mat<f64>> = (0..LAGS)
        .map(|l| Mat::from_fn(N, N, |e, j| beta[(1 + l * N + j, e)]))
        .collect();
    // cov_alpha[a*N + e, a2*N + e2] = zz_inv[1 + a, 1 + a2] * sigma_u[e, e2].
    let dim = LAGS * N * N;
    let cov_alpha = Mat::from_fn(dim, dim, |r, c| {
        zz_inv[(1 + r / N, 1 + c / N)] * sigma_u[(r % N, c % N)]
    });
    (resid, coefs, cov_alpha)
}

/// `Psi_h` from estimated coefficients.
fn ma_rep(coefs: &[Mat<f64>], horizon: usize) -> Vec<Mat<f64>> {
    let mut psi = vec![Mat::<f64>::identity(N, N)];
    for h in 1..=horizon {
        let mut acc = Mat::<f64>::zeros(N, N);
        for l in 1..=h.min(coefs.len()) {
            for r in 0..N {
                for c in 0..N {
                    let mut s = 0.0;
                    for q in 0..N {
                        s += psi[h - l][(r, q)] * coefs[l - 1][(q, c)];
                    }
                    acc[(r, c)] += s;
                }
            }
        }
        psi.push(acc);
    }
    psi
}

/// One estimated-`Psi` arm. `propagate` chooses whether reduced-form
/// uncertainty enters `V_hat(lam)`.
fn run_estimated_arm(
    seed: u64,
    reps: usize,
    t_obs: usize,
    phi: f64,
    sig_nu: f64,
    propagate: bool,
) -> ArmResult {
    let tru = truth(&true_ma(HORIZON));
    let z = inv_norm_cdf(0.5 + 0.5 * LEVEL).expect("normal quantile");
    let mut hit = vec![vec![0usize; N]; HORIZON + 1];
    let mut hit_wald = 0usize;
    let mut bounded = 0usize;
    let mut cells_total = 0usize;
    let mut used = 0usize;
    let mut stream = Stream::new(seed);

    for _ in 0..reps {
        let (y, shock) = simulate_levels(&mut stream, t_obs);
        // The proxy is aligned to the effective sample, exactly as a caller
        // would align it after dropping the presample rows.
        let proxy: Vec<f64> = (LAGS..t_obs)
            .map(|r| phi * shock[r] + sig_nu * std_normal(&mut stream))
            .collect();
        let (resid, coefs, cov_alpha) = fit_var(&y);
        let psi = ma_rep(&coefs, HORIZON);

        let spec = if propagate {
            // gamma has to come from the same overlap the sets use; recompute
            // it here rather than reaching into the result, so the input is
            // built exactly the way a caller builds it.
            let mut mbar = 0.0;
            for &m in &proxy {
                mbar += m;
            }
            mbar /= proxy.len() as f64;
            let gamma: Vec<f64> = (0..N)
                .map(|j| {
                    let mut ub = 0.0;
                    for r in 0..proxy.len() {
                        ub += resid[(r, j)];
                    }
                    ub /= proxy.len() as f64;
                    let mut s = 0.0;
                    for (r, &m) in proxy.iter().enumerate() {
                        s += (m - mbar) * (resid[(r, j)] - ub);
                    }
                    s / proxy.len() as f64
                })
                .collect();
            match psi_reduced_form_cov(&psi, &coefs, cov_alpha.as_ref(), &gamma, proxy.len()) {
                Ok(pv) => Some(pv),
                Err(_) => continue,
            }
        } else {
            None
        };
        let variance = match &spec {
            None => ArVarianceSpec::moment_only(ArVariance::Hc0),
            Some(pv) => ArVarianceSpec::with_reduced_form(
                ArVariance::Hc0,
                ArReducedForm {
                    psi_var: pv,
                    psi_gamma_cov: None,
                },
            ),
        };

        let res = match proxy_ar_sets(
            resid.as_ref(),
            &proxy,
            &psi,
            NORM_VAR,
            UNIT,
            variance,
            ArCritical::Chi2 { level: LEVEL },
        ) {
            Ok(r) => r,
            Err(_) => continue,
        };
        assert_eq!(res.reduced_form_uncertainty, propagate);
        assert_eq!(res.level.is_some(), propagate);
        used += 1;
        let no = res.n_proxy as f64;
        for (h, row) in res.cells.iter().enumerate() {
            for (i, cell) in row.iter().enumerate() {
                cells_total += 1;
                if cell.set.is_bounded() {
                    bounded += 1;
                }
                let target = tru[h][i];
                let covered = match cell.set {
                    ArSet::Point(p) => (p - target).abs() <= 1e-12 * (1.0 + p.abs()),
                    other => other.contains(target),
                };
                if covered {
                    hit[h][i] += 1;
                }
                let v = cell.variance(cell.point).max(0.0);
                let se = (v / no).sqrt() / cell.q0.abs();
                if (target - cell.point).abs() <= z * se {
                    hit_wald += 1;
                }
            }
        }
    }

    let ncell = (HORIZON + 1) * N;
    let by_h: Vec<f64> = hit
        .iter()
        .map(|row| row.iter().sum::<usize>() as f64 / (N * used) as f64)
        .collect();
    let total: usize = hit.iter().flatten().sum();
    let out = ArmResult {
        ar_coverage: total as f64 / (ncell * used) as f64,
        ar_coverage_by_h: by_h,
        wald_coverage: hit_wald as f64 / cells_total as f64,
        bounded_fraction: bounded as f64 / cells_total as f64,
        reps: used,
    };
    eprintln!(
        "  estimated-psi arm(seed={seed:#x}, reps={}, T={t_obs}, phi={phi}, \
         propagate={propagate}): AR={:.4} bounded={:.3} by_h={:?}",
        out.reps,
        out.ar_coverage,
        out.bounded_fraction,
        out.ar_coverage_by_h
            .iter()
            .map(|x| (x * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );
    out
}

/// Coverage excluding the `(NORM_VAR, h = 0)` cell, which is the point
/// `{unit}` and covers with probability exactly `1` by construction. Averaging
/// it in lifts the impact row above every other row and hides where a
/// horizon-dependent collapse starts.
fn coverage_excluding_degenerate(arm: &ArmResult) -> f64 {
    // by_h[0] averages N cells, one of which is the certain one.
    let n = N as f64;
    let h0 = (arm.ar_coverage_by_h[0] * n - 1.0) / (n - 1.0);
    let rest: f64 = arm.ar_coverage_by_h[1..].iter().sum();
    (h0 * (n - 1.0) + rest * n) / ((n - 1.0) + n * (HORIZON as f64))
}

/// THE DEFECT, PINNED. With the VAR estimated and reduced-form uncertainty
/// omitted, a STRONG instrument produces sets that are at nominal on impact
/// and then fall apart — this is the configuration that measured `0.119` at
/// `h = 8` in the fixture's Monte Carlo.
///
/// The assertion is deliberately on the BAD number. It is what stops the
/// module docs from claiming a collapse the code no longer has (or, worse,
/// from quietly ceasing to warn about one it still has), and it makes the
/// repair below a measured contrast rather than an assertion of faith.
#[test]
fn omitted_reduced_form_collapses_with_a_strong_instrument() {
    let arm = run_estimated_arm(0x5A1E_0010, 400, 300, 1.0, 1.5, false);
    assert!(arm.reps > 380, "too many failed replications: {}", arm.reps);
    assert_eq!(
        arm.bounded_fraction, 1.0,
        "a strong instrument gives bounded sets; the collapse is in their WIDTH"
    );
    let last = *arm.ar_coverage_by_h.last().expect("at least one horizon");
    assert!(
        last < 0.75,
        "the documented under-coverage is gone at h={HORIZON} (coverage {last:.4}, by horizon \
         {:?}). If the moment-only variance now covers, the module docs and the fixture table \
         are stale and must be re-measured",
        arm.ar_coverage_by_h
    );
    // Monotone decay with the horizon is the signature: impact is fine because
    // Psi_0 = I is not estimated.
    assert!(
        arm.ar_coverage_by_h[0] > 0.93,
        "impact coverage {:.4} should be unaffected — Psi_0 is the identity",
        arm.ar_coverage_by_h[0]
    );
    assert!(
        arm.ar_coverage_by_h[1] > last,
        "the shortfall should worsen with the horizon: {:?}",
        arm.ar_coverage_by_h
    );
}

/// THE REPAIR. Propagating reduced-form uncertainty on the same DGP restores
/// coverage to nominal at every horizon, and does not make the sets bounded
/// for the wrong reason.
#[test]
fn propagated_reduced_form_restores_coverage_with_a_strong_instrument() {
    let arm = run_estimated_arm(0x5A1E_0010, 400, 300, 1.0, 1.5, true);
    assert!(arm.reps > 380, "too many failed replications: {}", arm.reps);
    let excl = coverage_excluding_degenerate(&arm);
    assert!(
        excl > 0.92 && excl < 0.98,
        "propagated coverage {excl:.4} (excluding the degenerate cell) is off nominal {LEVEL}; \
         by horizon {:?}",
        arm.ar_coverage_by_h
    );
    for (h, &cov) in arm.ar_coverage_by_h.iter().enumerate() {
        assert!(
            cov > 0.90,
            "propagated coverage {cov:.4} at h={h} (all: {:?})",
            arm.ar_coverage_by_h
        );
    }
}

/// WEAK-IV ROBUSTNESS SURVIVES THE CORRECTION. The reduced-form term is
/// quadratic in `gamma`, so it must not turn the weak arm's unbounded sets
/// into bounded ones — a bounded set there could not have valid coverage
/// (Dufour 1997). Boundedness is decided by `A = T_O*q0^2 - c*v2`, which the
/// correction never touches, and this measures that end to end.
#[test]
fn propagated_reduced_form_leaves_the_weak_arm_unbounded() {
    let omitted = run_estimated_arm(0x5A1E_0011, 300, 300, 0.06, 1.5, false);
    let propagated = run_estimated_arm(0x5A1E_0011, 300, 300, 0.06, 1.5, true);
    assert_eq!(
        omitted.bounded_fraction, propagated.bounded_fraction,
        "the reduced-form correction changed which sets are bounded; it must enter v0 and v1 \
         only, never v2 or q0"
    );
    assert!(
        propagated.bounded_fraction < 0.30,
        "weak-arm bounded fraction {:.3} is too high",
        propagated.bounded_fraction
    );
    // Coverage stays at or above nominal; the correction is conservative here
    // because it turns some exterior sets into the whole line.
    assert!(
        propagated.ar_coverage > 0.93,
        "weak-arm coverage {:.4} fell below nominal after propagation",
        propagated.ar_coverage
    );
}

/// Bit-for-bit reproducibility at a fixed seed, and independence of the two
/// arms' streams.
#[test]
fn arms_are_reproducible() -> Result<(), IdentError> {
    let a = run_arm(0x5A1E_0003, 120, 200, 1.0, 1.5);
    let b = run_arm(0x5A1E_0003, 120, 200, 1.0, 1.5);
    assert_eq!(a.ar_coverage, b.ar_coverage);
    assert_eq!(a.bounded_fraction, b.bounded_fraction);
    let c = run_arm(0x5A1E_0004, 120, 200, 1.0, 1.5);
    assert!(
        a.ar_coverage != c.ar_coverage || a.wald_coverage != c.wald_coverage,
        "different seeds produced identical results"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// SECOND-ORDER (SIMULATION) REDUCED-FORM VARIANCE — audit round 6, finding 8
// ---------------------------------------------------------------------------

/// One fitted dataset for the second-order tests: a seeded card-DGP draw with
/// its OLS fit, its aligned strong proxy, and the moment `gamma` over the
/// proxy overlap.
#[allow(clippy::type_complexity)]
fn fitted_example(
    seed: u64,
    t_obs: usize,
) -> (Mat<f64>, Vec<f64>, Vec<Mat<f64>>, Mat<f64>, Vec<f64>) {
    let mut stream = Stream::new(seed);
    let (y, shock) = simulate_levels(&mut stream, t_obs);
    let proxy: Vec<f64> = (LAGS..t_obs)
        .map(|r| 1.0 * shock[r] + 1.5 * std_normal(&mut stream))
        .collect();
    let (resid, coefs, cov_alpha) = fit_var(&y);
    let mut mbar = 0.0;
    for &m in &proxy {
        mbar += m;
    }
    mbar /= proxy.len() as f64;
    let gamma: Vec<f64> = (0..N)
        .map(|j| {
            let mut ub = 0.0;
            for r in 0..proxy.len() {
                ub += resid[(r, j)];
            }
            ub /= proxy.len() as f64;
            let mut s = 0.0;
            for (r, &m) in proxy.iter().enumerate() {
                s += (m - mbar) * (resid[(r, j)] - ub);
            }
            s / proxy.len() as f64
        })
        .collect();
    (resid, proxy, coefs, cov_alpha, gamma)
}

/// The simulation variance is the exact propagation of the same Gaussian
/// coefficient uncertainty, so as that uncertainty shrinks the nonlinear
/// terms die and it must converge to the first-order delta method. Shrinking
/// `cov_alpha` by `eps^2` scales both variances by `eps^2`; the ratio of
/// diagonals must approach 1 within the sampler's own Monte-Carlo error.
#[test]
fn second_order_matches_delta_when_uncertainty_is_small() -> Result<(), IdentError> {
    let (_resid, proxy, coefs, cov_alpha, gamma) = fitted_example(0x5A1E_0020, 300);
    let horizon = 8;
    let psi = ma_rep(&coefs, horizon);
    let eps2 = 1e-12;
    let tiny = Mat::from_fn(cov_alpha.nrows(), cov_alpha.ncols(), |i, j| {
        eps2 * cov_alpha[(i, j)]
    });
    let delta = psi_reduced_form_cov(&psi, &coefs, tiny.as_ref(), &gamma, proxy.len())?;
    let mc = psi_reduced_form_cov_mc(
        horizon,
        &coefs,
        tiny.as_ref(),
        &gamma,
        proxy.len(),
        4096,
        0xD5EE_D001,
    )?;
    for h in 1..=horizon {
        for i in 0..N {
            let d = delta[h][(i, i)];
            let m = mc[h][(i, i)];
            assert!(
                (m / d - 1.0).abs() < 0.2,
                "h={h} i={i}: mc diag {m:.6e} vs delta diag {d:.6e} — the simulation \
                 variance must reduce to the delta method as the uncertainty vanishes"
            );
        }
    }
    Ok(())
}

/// Same seed, same matrices, bit for bit; a different seed moves them.
#[test]
fn second_order_is_deterministic_in_the_seed() -> Result<(), IdentError> {
    let (_resid, proxy, coefs, cov_alpha, gamma) = fitted_example(0x5A1E_0021, 250);
    let a = psi_reduced_form_cov_mc(6, &coefs, cov_alpha.as_ref(), &gamma, proxy.len(), 256, 7)?;
    let b = psi_reduced_form_cov_mc(6, &coefs, cov_alpha.as_ref(), &gamma, proxy.len(), 256, 7)?;
    let c = psi_reduced_form_cov_mc(6, &coefs, cov_alpha.as_ref(), &gamma, proxy.len(), 256, 8)?;
    let mut differs = false;
    for h in 0..=6 {
        for i in 0..N {
            for j in 0..N {
                assert_eq!(
                    a[h][(i, j)].to_bits(),
                    b[h][(i, j)].to_bits(),
                    "same seed must reproduce bit-for-bit at h={h}"
                );
                if a[h][(i, j)] != c[h][(i, j)] {
                    differs = true;
                }
            }
        }
    }
    assert!(differs, "a different seed produced identical matrices");
    Ok(())
}

/// The mechanism round 6 measured: at long horizons the exact propagation
/// carries the convexity of `alpha -> Psi_h` that the first-order delta
/// method drops, so on a fitted persistent system its diagonal exceeds the
/// delta diagonal — and by more at `h = 12` than at `h = 2`. Deterministic
/// given the seed.
#[test]
fn second_order_exceeds_delta_at_long_horizons() -> Result<(), IdentError> {
    let (_resid, proxy, coefs, cov_alpha, gamma) = fitted_example(0x5A1E_0022, 300);
    let horizon = 12;
    let psi = ma_rep(&coefs, horizon);
    let delta = psi_reduced_form_cov(&psi, &coefs, cov_alpha.as_ref(), &gamma, proxy.len())?;
    let mc = psi_reduced_form_cov_mc(
        horizon,
        &coefs,
        cov_alpha.as_ref(),
        &gamma,
        proxy.len(),
        2048,
        0xD5EE_D002,
    )?;
    let ratio_at = |h: usize| {
        let mut r = 0.0f64;
        for i in 0..N {
            r = r.max(mc[h][(i, i)] / delta[h][(i, i)]);
        }
        r
    };
    let early = ratio_at(2);
    let late = ratio_at(12);
    assert!(
        late > 1.05,
        "expected the second-order variance to exceed the delta at h=12; max ratio {late:.4}"
    );
    assert!(
        late > early,
        "the second-order excess must grow with the horizon: h=2 ratio {early:.4}, \
         h=12 ratio {late:.4}"
    );
    Ok(())
}

/// End to end: swapping the delta `psi_var` for the second-order one widens
/// the long-horizon intervals, leaves the boundedness decision bit-identical
/// (the correction enters `v0` only — never `q0` or `v2`), and leaves the
/// point estimates untouched.
#[test]
fn second_order_widens_sets_and_preserves_boundedness() -> Result<(), IdentError> {
    let (resid, proxy, coefs, cov_alpha, gamma) = fitted_example(0x5A1E_0023, 300);
    let horizon = 12;
    let psi = ma_rep(&coefs, horizon);
    let pv_delta = psi_reduced_form_cov(&psi, &coefs, cov_alpha.as_ref(), &gamma, proxy.len())?;
    let pv_mc = psi_reduced_form_cov_mc(
        horizon,
        &coefs,
        cov_alpha.as_ref(),
        &gamma,
        proxy.len(),
        512,
        0xD5EE_D003,
    )?;
    let run = |pv: &Vec<Mat<f64>>| {
        proxy_ar_sets(
            resid.as_ref(),
            &proxy,
            &psi,
            NORM_VAR,
            UNIT,
            ArVarianceSpec::with_reduced_form(
                ArVariance::Hc0,
                ArReducedForm {
                    psi_var: pv,
                    psi_gamma_cov: None,
                },
            ),
            ArCritical::Chi2 { level: LEVEL },
        )
    };
    let base = run(&pv_delta)?;
    let second = run(&pv_mc)?;
    assert_eq!(base.ar_bound_stat.to_bits(), second.ar_bound_stat.to_bits());
    assert_eq!(base.ar_bounded_all, second.ar_bounded_all);
    let mut wider = 0usize;
    for i in 0..N {
        let cb = &base.cells[horizon][i];
        let cs = &second.cells[horizon][i];
        assert_eq!(cb.point.to_bits(), cs.point.to_bits());
        if let (ArSet::Interval { lo: bl, hi: bh }, ArSet::Interval { lo: sl, hi: sh }) =
            (cb.set, cs.set)
        {
            if (sh - sl) > (bh - bl) {
                wider += 1;
            }
        }
    }
    assert!(
        wider >= 2,
        "the second-order variance should widen the h={horizon} intervals on this fitted \
         persistent system (wider in {wider}/3 cells)"
    );
    Ok(())
}

/// The sampler's argument contract: an odd or too-small draw count is refused
/// with a teaching error, never silently adjusted.
#[test]
fn second_order_rejects_bad_draw_counts() {
    let (_resid, proxy, coefs, cov_alpha, gamma) = fitted_example(0x5A1E_0024, 150);
    for draws in [0usize, 16, 31, 33, 255] {
        assert!(
            matches!(
                psi_reduced_form_cov_mc(
                    4,
                    &coefs,
                    cov_alpha.as_ref(),
                    &gamma,
                    proxy.len(),
                    draws,
                    0
                ),
                Err(IdentError::InvalidArgument { .. })
            ),
            "draws={draws} must be rejected"
        );
    }
}
