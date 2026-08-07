//! Property and Monte Carlo validation for `tsecon_stats::simultaneous`.
//!
//! The goldens in `simultaneous_golden.rs` check that this module computes the
//! documented formulas. This file checks the things a formula golden cannot:
//!
//!   1. the invariants a simultaneous critical value must satisfy — never
//!      below the pointwise value, increasing in the number of cells,
//!      collapsing to the pointwise value at one cell;
//!   2. the ordering Bonferroni >= Sidak >= sup-t under positive dependence,
//!      *measured* rather than assumed;
//!   3. degenerate (`se == 0`) cells;
//!   4. the point of the whole feature — a direct Monte Carlo measurement that
//!      a sup-t band attains its nominal *simultaneous* coverage while the
//!      pointwise band of the same nominal level does not.
//!
//! Everything random here runs off a seeded SplitMix64, so every number below
//! is reproducible. Run with `--nocapture` to see the measured coverages.

use tsecon_stats::simultaneous::{
    band, bonferroni_critical_value, pointwise_critical_value, required_uniforms,
    sidak_critical_value, std_errors_from_cov, sup_t_from_cov, sup_t_from_draws,
};
use tsecon_stats::special::inv_norm_cdf;

// ---------------------------------------------------------------------------
// Test-local deterministic RNG and linear algebra
// ---------------------------------------------------------------------------

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let z = z ^ (z >> 31);
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn uniforms(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = Self::new(seed);
        (0..n).map(|_| rng.uniform()).collect()
    }

    /// Standard normal via the same inverse transform the library uses.
    fn normal(&mut self) -> f64 {
        // Nudge off the 0.0 grid point exactly as the library does.
        let u = self.uniform() + 5.551_115_123_125_783e-17;
        inv_norm_cdf(u).expect("uniform in (0, 1)")
    }
}

/// Lower-triangular Cholesky factor of a positive-definite row-major matrix.
fn chol_lower(sigma: &[f64], k: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; k * k];
    for j in 0..k {
        let mut d = sigma[j * k + j];
        for m in 0..j {
            d -= l[j * k + m] * l[j * k + m];
        }
        assert!(d > 0.0, "test design matrix must be positive definite");
        let ljj = d.sqrt();
        l[j * k + j] = ljj;
        for i in (j + 1)..k {
            let mut s = sigma[i * k + j];
            for m in 0..j {
                s -= l[i * k + m] * l[j * k + m];
            }
            l[i * k + j] = s / ljj;
        }
    }
    l
}

fn mvn_draw(l: &[f64], k: usize, rng: &mut SplitMix64, out: &mut [f64]) {
    let z: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
    for (i, o) in out.iter_mut().enumerate() {
        let mut x = 0.0;
        for m in 0..=i {
            x += l[i * k + m] * z[m];
        }
        *o = x;
    }
}

// ---------------------------------------------------------------------------
// Test designs
// ---------------------------------------------------------------------------

/// IRF-shaped standard errors: a hump at short horizons, decaying after.
fn irf_se(k: usize) -> Vec<f64> {
    (0..k)
        .map(|h| {
            let h = h as f64;
            0.15 + 0.45 * (-0.5 * ((h - 2.0) / 4.0).powi(2)).exp()
        })
        .collect()
}

/// `Sigma_ij = rho^|i-j| * se_i * se_j` — a persistent IRF path.
fn ar1_cov(k: usize, rho: f64, se: &[f64]) -> Vec<f64> {
    let mut s = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            let d = (i as i64 - j as i64).unsigned_abs() as i32;
            s[i * k + j] = rho.powi(d) * se[i] * se[j];
        }
    }
    s
}

/// A multi-series forecast covariance: random-walk accumulation across
/// horizons (`cov = min(h, h') + 1`) times an equicorrelated cross-series
/// block. Cell order is series-major: `(series, horizon) -> s * H + h`.
fn forecast_cov(n_series: usize, n_horizon: usize, cross: f64) -> (usize, Vec<f64>) {
    let k = n_series * n_horizon;
    let mut s = vec![0.0; k * k];
    for a in 0..n_series {
        for h in 0..n_horizon {
            for b in 0..n_series {
                for g in 0..n_horizon {
                    let cross_term = if a == b { 1.0 } else { cross };
                    let horizon_term = (h.min(g) + 1) as f64 * 0.09;
                    s[(a * n_horizon + h) * k + (b * n_horizon + g)] = cross_term * horizon_term;
                }
            }
        }
    }
    (k, s)
}

// ---------------------------------------------------------------------------
// 1. Invariants
// ---------------------------------------------------------------------------

#[test]
fn every_route_is_at_least_the_pointwise_value() {
    let n_sim = 20_000;
    for &alpha in &[0.32, 0.10, 0.05, 0.01] {
        let z = pointwise_critical_value(alpha).unwrap();
        for k in [1usize, 2, 3, 5, 8, 13, 21, 40] {
            assert!(bonferroni_critical_value(alpha, k).unwrap() >= z);
            assert!(sidak_critical_value(alpha, k).unwrap() >= z);

            let se = irf_se(k);
            let sigma = ar1_cov(k, 0.85, &se);
            let u = SplitMix64::uniforms(11_000_001, required_uniforms(k, n_sim));
            let c_cov = sup_t_from_cov(&sigma, k, alpha, &u).unwrap();
            assert!(
                c_cov >= z,
                "sup_t_from_cov: alpha={alpha}, k={k}: {c_cov} < pointwise {z}"
            );

            // Same design, bootstrap route.
            let l = chol_lower(&sigma, k);
            let theta_hat = vec![0.5; k];
            let n_draws = 4_000;
            let mut rng = SplitMix64::new(11_000_002);
            let mut draws = vec![0.0; n_draws * k];
            let mut buf = vec![0.0; k];
            for b in 0..n_draws {
                mvn_draw(&l, k, &mut rng, &mut buf);
                for j in 0..k {
                    draws[b * k + j] = theta_hat[j] + buf[j];
                }
            }
            let c_boot = sup_t_from_draws(&draws, n_draws, &theta_hat, &se, alpha).unwrap();
            assert!(
                c_boot >= z,
                "sup_t_from_draws: alpha={alpha}, k={k}: {c_boot} < pointwise {z}"
            );
        }
    }
}

#[test]
fn closed_forms_increase_strictly_in_k() {
    for &alpha in &[0.32, 0.10, 0.05, 0.01] {
        let mut prev_bonf = f64::NEG_INFINITY;
        let mut prev_sidak = f64::NEG_INFINITY;
        for k in 1..=200usize {
            let bonf = bonferroni_critical_value(alpha, k).unwrap();
            let sidak = sidak_critical_value(alpha, k).unwrap();
            assert!(
                bonf > prev_bonf,
                "bonferroni not increasing at alpha={alpha}, k={k}"
            );
            assert!(
                sidak > prev_sidak,
                "sidak not increasing at alpha={alpha}, k={k}"
            );
            prev_bonf = bonf;
            prev_sidak = sidak;
        }
    }
}

/// Nested cells on one fixed draws matrix: adding a cell can only enlarge each
/// draw's maximum, so the quantile can only rise. This is an exact,
/// noise-free monotonicity statement.
#[test]
fn sup_t_from_draws_increases_in_k_on_nested_cells() {
    let k_max = 13;
    let se = irf_se(k_max);
    let sigma = ar1_cov(k_max, 0.85, &se);
    let l = chol_lower(&sigma, k_max);
    let theta_hat: Vec<f64> = (0..k_max).map(|h| 0.8 * 0.85f64.powi(h as i32)).collect();
    let n_draws = 5_000;
    let mut rng = SplitMix64::new(12_000_003);
    let mut full = vec![0.0; n_draws * k_max];
    let mut buf = vec![0.0; k_max];
    for b in 0..n_draws {
        mvn_draw(&l, k_max, &mut rng, &mut buf);
        for j in 0..k_max {
            full[b * k_max + j] = theta_hat[j] + buf[j];
        }
    }

    for &alpha in &[0.32, 0.10, 0.05] {
        let mut prev = f64::NEG_INFINITY;
        let mut first = 0.0;
        for k in 1..=k_max {
            // First k columns of each draw.
            let mut sub = vec![0.0; n_draws * k];
            for b in 0..n_draws {
                sub[b * k..(b + 1) * k].copy_from_slice(&full[b * k_max..b * k_max + k]);
            }
            let c = sup_t_from_draws(&sub, n_draws, &theta_hat[..k], &se[..k], alpha).unwrap();
            assert!(
                c >= prev,
                "alpha={alpha}: c({k}) = {c} < c({}) = {prev}",
                k - 1
            );
            if k == 1 {
                first = c;
            }
            prev = c;
        }
        assert!(
            prev > first + 0.2,
            "alpha={alpha}: c(13) = {prev} barely above c(1) = {first}"
        );
    }
}

/// Same nesting argument for the simulated route: the leading `k x k` block of
/// a Cholesky factor is the Cholesky factor of the leading `k x k` block, so
/// feeding the first `k` uniforms of each simulation block reproduces exactly
/// the same draws for the retained cells.
#[test]
fn sup_t_from_cov_increases_in_k_on_nested_cells() {
    let k_max = 13;
    let n_sim = 40_000;
    let se = irf_se(k_max);
    let sigma_full = ar1_cov(k_max, 0.85, &se);
    let u_full = SplitMix64::uniforms(13_000_004, required_uniforms(k_max, n_sim));

    for &alpha in &[0.32, 0.10, 0.05] {
        let mut prev = f64::NEG_INFINITY;
        let mut first = 0.0;
        for k in 1..=k_max {
            let mut sigma = vec![0.0; k * k];
            for i in 0..k {
                sigma[i * k..(i + 1) * k].copy_from_slice(&sigma_full[i * k_max..i * k_max + k]);
            }
            let mut u = vec![0.0; k * n_sim];
            for s in 0..n_sim {
                u[s * k..(s + 1) * k].copy_from_slice(&u_full[s * k_max..s * k_max + k]);
            }
            let c = sup_t_from_cov(&sigma, k, alpha, &u).unwrap();
            assert!(
                c >= prev,
                "alpha={alpha}: c({k}) = {c} < c({}) = {prev}",
                k - 1
            );
            if k == 1 {
                first = c;
            }
            prev = c;
        }
        assert!(
            prev > first + 0.2,
            "alpha={alpha}: c(13) = {prev} barely above c(1) = {first}"
        );
    }
}

/// The two sup-t routes are two estimators of the same population quantile, so
/// on a Gaussian design with the same `Sigma` they must agree up to Monte
/// Carlo error.
#[test]
fn the_two_sup_t_routes_agree_on_a_gaussian_design() {
    let k = 13;
    let se = irf_se(k);
    let sigma = ar1_cov(k, 0.85, &se);
    let l = chol_lower(&sigma, k);
    let theta_hat = vec![0.0; k];

    let n_draws = 200_000;
    let mut rng = SplitMix64::new(14_000_005);
    let mut draws = vec![0.0; n_draws * k];
    let mut buf = vec![0.0; k];
    for b in 0..n_draws {
        mvn_draw(&l, k, &mut rng, &mut buf);
        draws[b * k..(b + 1) * k].copy_from_slice(&buf);
    }
    let u = SplitMix64::uniforms(15_000_006, required_uniforms(k, 200_000));

    for &alpha in &[0.32, 0.10, 0.05] {
        let c_boot = sup_t_from_draws(&draws, n_draws, &theta_hat, &se, alpha).unwrap();
        let c_cov = sup_t_from_cov(&sigma, k, alpha, &u).unwrap();
        println!("alpha={alpha}: sup-t from draws {c_boot:.4}, from cov {c_cov:.4}");
        assert!(
            (c_boot - c_cov).abs() < 0.03,
            "alpha={alpha}: routes disagree, draws {c_boot} vs cov {c_cov}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. The measured ordering
// ---------------------------------------------------------------------------

/// Bonferroni >= Sidak >= sup-t is the expected ordering under positive
/// dependence: Bonferroni and Sidak ignore the correlation across cells, sup-t
/// exploits it. Measure it rather than assume it, on three dependence regimes.
#[test]
fn measured_ordering_bonferroni_ge_sidak_ge_sup_t() {
    let k = 13;
    let n_sim = 200_000;
    let se = irf_se(k);
    // The bool marks designs with strong enough positive dependence that sup-t
    // should beat Sidak by a visible margin, not merely tie it.
    let designs: Vec<(&str, Vec<f64>, bool)> = vec![
        ("persistent (rho = 0.95)", ar1_cov(k, 0.95, &se), true),
        ("moderate  (rho = 0.85)", ar1_cov(k, 0.85, &se), true),
        ("weak      (rho = 0.30)", ar1_cov(k, 0.30, &se), false),
        (
            "independent",
            {
                let mut s = vec![0.0; k * k];
                for i in 0..k {
                    s[i * k + i] = se[i] * se[i];
                }
                s
            },
            false,
        ),
    ];
    let u = SplitMix64::uniforms(16_000_007, required_uniforms(k, n_sim));

    for &alpha in &[0.10, 0.05] {
        let z = pointwise_critical_value(alpha).unwrap();
        let bonf = bonferroni_critical_value(alpha, k).unwrap();
        let sidak = sidak_critical_value(alpha, k).unwrap();
        assert!(
            bonf >= sidak,
            "alpha={alpha}: bonferroni {bonf} < sidak {sidak}"
        );
        println!(
            "alpha={alpha}, K={k}: pointwise {z:.4} | sidak {sidak:.4} | bonferroni {bonf:.4}"
        );
        for (name, sigma, strong_dependence) in &designs {
            let sup_t = sup_t_from_cov(sigma, k, alpha, &u).unwrap();
            println!("    sup-t {name:24} {sup_t:.4}");
            // Sidak is exact under independence, so allow a hair of simulation
            // slack in the independent design where the two coincide.
            assert!(
                sup_t <= sidak + 5e-3,
                "alpha={alpha}, {name}: sup-t {sup_t} exceeds sidak {sidak}"
            );
            assert!(
                sup_t <= bonf,
                "alpha={alpha}, {name}: sup-t {sup_t} exceeds bonferroni {bonf}"
            );
            assert!(sup_t >= z, "alpha={alpha}, {name}: sup-t {sup_t} below {z}");
            if *strong_dependence {
                assert!(
                    sup_t < sidak - 0.1,
                    "alpha={alpha}, {name}: sup-t {sup_t} should exploit the \
                     correlation and beat sidak {sidak} by a clear margin"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Degenerate cells
// ---------------------------------------------------------------------------

/// A cell pinned by a normalization has zero variance, so `Sigma` has a zero
/// row and column. It must contribute nothing to the maximum: the critical
/// value must equal the one from the problem with that cell deleted.
#[test]
fn a_pinned_cell_changes_nothing_in_the_cov_route() {
    let k = 13;
    let pinned = 6usize;
    let n_sim = 40_000;
    let se = irf_se(k);
    let mut sigma = ar1_cov(k, 0.85, &se);
    for i in 0..k {
        sigma[i * k + pinned] = 0.0;
        sigma[pinned * k + i] = 0.0;
    }
    let se_full = std_errors_from_cov(&sigma, k).unwrap();
    assert_eq!(se_full[pinned], 0.0);

    // The same problem with the pinned row/column deleted.
    let ks = k - 1;
    let keep: Vec<usize> = (0..k).filter(|&i| i != pinned).collect();
    let mut sigma_sub = vec![0.0; ks * ks];
    for (a, &i) in keep.iter().enumerate() {
        for (b, &j) in keep.iter().enumerate() {
            sigma_sub[a * ks + b] = sigma[i * k + j];
        }
    }

    let u_full = SplitMix64::uniforms(17_000_008, required_uniforms(k, n_sim));
    let mut u_sub = vec![0.0; ks * n_sim];
    for s in 0..n_sim {
        for (a, &j) in keep.iter().enumerate() {
            u_sub[s * ks + a] = u_full[s * k + j];
        }
    }

    for &alpha in &[0.32, 0.10, 0.05] {
        let c_full = sup_t_from_cov(&sigma, k, alpha, &u_full).unwrap();
        let c_sub = sup_t_from_cov(&sigma_sub, ks, alpha, &u_sub).unwrap();
        assert!(c_full.is_finite(), "pinned cell produced {c_full}");
        assert!(
            (c_full - c_sub).abs() < 1e-12,
            "alpha={alpha}: pinned {c_full} vs deleted {c_sub}"
        );
    }
}

#[test]
fn a_pinned_cell_gets_a_zero_width_band() {
    let theta_hat = vec![1.0, -0.5, 2.0];
    let se = vec![0.4, 0.0, 0.7];
    let b = band(&theta_hat, &se, 2.7).unwrap();
    assert_eq!(b.n_cells_used, 2);
    assert_eq!(b.lower[1], -0.5);
    assert_eq!(b.upper[1], -0.5);
    assert!((b.lower[0] - (1.0 - 2.7 * 0.4)).abs() < 1e-15);
    assert!((b.upper[2] - (2.0 + 2.7 * 0.7)).abs() < 1e-15);
    assert!(b.lower.iter().chain(b.upper.iter()).all(|v| v.is_finite()));
}

#[test]
fn an_all_degenerate_problem_is_an_error_not_a_nan() {
    let theta_hat = vec![1.0, 2.0];
    let se = vec![0.0, 0.0];
    let draws = vec![1.0, 2.0, 1.0, 2.0];
    assert!(sup_t_from_draws(&draws, 2, &theta_hat, &se, 0.1).is_err());
    let sigma = vec![0.0; 4];
    let u = SplitMix64::uniforms(1, 4);
    assert!(sup_t_from_cov(&sigma, 2, 0.1, &u).is_err());
}

// ---------------------------------------------------------------------------
// 4. Error surface
// ---------------------------------------------------------------------------

#[test]
fn invalid_inputs_are_rejected() {
    let theta_hat = vec![0.0, 0.0];
    let se = vec![1.0, 1.0];
    let draws = vec![0.1, -0.2, 0.3, 0.4];

    for bad_alpha in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
        assert!(pointwise_critical_value(bad_alpha).is_err());
        assert!(bonferroni_critical_value(bad_alpha, 3).is_err());
        assert!(sidak_critical_value(bad_alpha, 3).is_err());
        assert!(sup_t_from_draws(&draws, 2, &theta_hat, &se, bad_alpha).is_err());
    }
    assert!(bonferroni_critical_value(0.05, 0).is_err());
    assert!(sidak_critical_value(0.05, 0).is_err());

    // Length mismatches.
    assert!(sup_t_from_draws(&draws, 2, &theta_hat, &[1.0], 0.1).is_err());
    assert!(sup_t_from_draws(&draws, 3, &theta_hat, &se, 0.1).is_err());
    assert!(sup_t_from_draws(&draws, 1, &theta_hat, &se, 0.1).is_err());

    // Non-finite draws are rejected, not skipped.
    let poisoned = vec![0.1, -0.2, f64::NAN, 0.4];
    assert!(sup_t_from_draws(&poisoned, 2, &theta_hat, &se, 0.1).is_err());
    let infinite = vec![0.1, -0.2, f64::INFINITY, 0.4];
    assert!(sup_t_from_draws(&infinite, 2, &theta_hat, &se, 0.1).is_err());
    // Negative standard errors are a caller bug, not a degenerate cell.
    assert!(sup_t_from_draws(&draws, 2, &theta_hat, &[1.0, -1.0], 0.1).is_err());

    // Covariance route: shape, symmetry, definiteness, uniform range.
    let u = SplitMix64::uniforms(7, 4 * 100);
    assert!(sup_t_from_cov(&[1.0, 0.2, 0.2, 1.0], 2, 0.1, &u).is_ok());
    assert!(sup_t_from_cov(&[1.0, 0.2, 0.2], 2, 0.1, &u).is_err());
    assert!(sup_t_from_cov(&[1.0, 0.4, 0.2, 1.0], 2, 0.1, &u).is_err());
    assert!(sup_t_from_cov(&[1.0, 2.0, 2.0, 1.0], 2, 0.1, &u).is_err());
    assert!(sup_t_from_cov(&[1.0, 0.2, 0.2, 1.0], 2, 0.1, &[0.5; 3]).is_err());
    assert!(sup_t_from_cov(&[1.0, 0.2, 0.2, 1.0], 2, 0.1, &[0.5; 2]).is_err());
    assert!(sup_t_from_cov(&[1.0, 0.2, 0.2, 1.0], 2, 0.1, &[0.5, 0.5, 1.0, 0.5]).is_err());
    assert!(sup_t_from_cov(&[1.0, 0.2, 0.2, 1.0], 2, 0.1, &[0.5, 0.5, -0.1, 0.5]).is_err());

    // An exact-zero uniform is legal (`Stream::uniform_f64` can emit it) and
    // must not produce an infinite deviate.
    let c = sup_t_from_cov(&[1.0, 0.2, 0.2, 1.0], 2, 0.1, &[0.0, 0.5, 0.5, 0.5]).unwrap();
    assert!(c.is_finite(), "zero uniform produced {c}");

    assert!(band(&theta_hat, &se, f64::NAN).is_err());
    assert!(band(&theta_hat, &se, -1.0).is_err());
    assert!(band(&theta_hat, &[1.0], 2.0).is_err());
}

// ---------------------------------------------------------------------------
// 5. The Monte Carlo that justifies the feature
// ---------------------------------------------------------------------------

struct Coverage {
    simultaneous: f64,
    pointwise_cell0: f64,
}

/// Draw `n_rep` estimates from `N(theta_0, Sigma)` and measure how often the
/// band `theta_hat +/- c * se` covers `theta_0` at every cell at once.
///
/// With `theta_0 = 0` the band covers every cell iff
/// `max_k |theta_hat_k| / se_k <= c`, so one pass over the draws gives the
/// coverage of every candidate critical value.
fn measure_coverage(sigma: &[f64], k: usize, cs: &[f64], n_rep: usize, seed: u64) -> Vec<Coverage> {
    let l = chol_lower(sigma, k);
    let se = std_errors_from_cov(sigma, k).unwrap();
    let mut rng = SplitMix64::new(seed);
    let mut hits = vec![0usize; cs.len()];
    let mut hits_cell0 = vec![0usize; cs.len()];
    let mut buf = vec![0.0; k];
    for _ in 0..n_rep {
        mvn_draw(&l, k, &mut rng, &mut buf);
        let mut m: f64 = 0.0;
        for j in 0..k {
            let t = (buf[j] / se[j]).abs();
            if t > m {
                m = t;
            }
        }
        let t0 = (buf[0] / se[0]).abs();
        for (i, &c) in cs.iter().enumerate() {
            if m <= c {
                hits[i] += 1;
            }
            if t0 <= c {
                hits_cell0[i] += 1;
            }
        }
    }
    hits.iter()
        .zip(hits_cell0.iter())
        .map(|(&h, &h0)| Coverage {
            simultaneous: h as f64 / n_rep as f64,
            pointwise_cell0: h0 as f64 / n_rep as f64,
        })
        .collect()
}

/// The headline validation.
///
/// Both designs are *calibrated to the audit's measured failures*, so the
/// comparison is like for like rather than to an arbitrary correlation:
///
///   * Design A — 13 cells, nominal 90%, `rho = 0.956` chosen so the pointwise
///     band's simultaneous coverage reproduces the 72.2% the audit measured
///     for a nominal 90% pointwise IRF band over a 13-horizon path at T = 500.
///   * Design B — 45 cells (3 series x 15 horizons), nominal 95%, cross-series
///     correlation 0.3, chosen so the pointwise band's simultaneous coverage
///     reproduces the 48.1% the audit measured for nominal 95% marginal
///     `var_forecast` bands at T = 800 (the value the joint rate had *not*
///     converged past, having been 40.9% at T = 100).
///
/// On those same designs the sup-t band must hit its nominal level.
#[test]
fn monte_carlo_sup_t_attains_nominal_simultaneous_coverage() {
    let n_sim = 100_000;
    let n_rep = 50_000;

    let k_a = 13;
    let se_a = irf_se(k_a);
    let sigma_a = ar1_cov(k_a, 0.956, &se_a);
    let (k_b, sigma_b) = forecast_cov(3, 15, 0.3);

    for (label, k, sigma, alpha, audit) in [
        ("IRF path, K=13, nominal 90%", k_a, &sigma_a, 0.10, 0.722),
        (
            "forecast path, K=45, nominal 95%",
            k_b,
            &sigma_b,
            0.05,
            0.481,
        ),
    ] {
        let nominal = 1.0 - alpha;
        let u = SplitMix64::uniforms(18_000_009, required_uniforms(k, n_sim));
        let c_sup = sup_t_from_cov(sigma, k, alpha, &u).unwrap();
        let z = pointwise_critical_value(alpha).unwrap();
        let c_bonf = bonferroni_critical_value(alpha, k).unwrap();
        let c_sidak = sidak_critical_value(alpha, k).unwrap();

        let cov = measure_coverage(sigma, k, &[c_sup, z, c_bonf, c_sidak], n_rep, 18_000_010);
        let (sup, point, bonf, sidak) = (&cov[0], &cov[1], &cov[2], &cov[3]);

        println!("\n{label}  ({n_rep} replications; audit's pointwise joint rate {audit:.3})");
        println!("  critical value   simultaneous cov.   pointwise cov. (cell 0)");
        println!(
            "  sup-t      {c_sup:.4}        {:.4}              {:.4}",
            sup.simultaneous, sup.pointwise_cell0
        );
        println!(
            "  pointwise  {z:.4}        {:.4}              {:.4}",
            point.simultaneous, point.pointwise_cell0
        );
        println!(
            "  Sidak      {c_sidak:.4}        {:.4}              {:.4}",
            sidak.simultaneous, sidak.pointwise_cell0
        );
        println!(
            "  Bonferroni {c_bonf:.4}        {:.4}              {:.4}",
            bonf.simultaneous, bonf.pointwise_cell0
        );

        // The sup-t band hits its nominal simultaneous level. With 100k
        // replications the Monte Carlo standard error is under 0.001; 0.01
        // leaves room for the error in the simulated critical value itself.
        assert!(
            (sup.simultaneous - nominal).abs() < 0.01,
            "{label}: sup-t simultaneous coverage {:.4} off nominal {nominal}",
            sup.simultaneous
        );
        // The pointwise band of the same nominal level does not, and the
        // shortfall is the size the audit measured — this is the defect the
        // feature exists to fix.
        assert!(
            point.simultaneous < nominal - 0.05,
            "{label}: pointwise band simultaneous coverage {:.4} is not the \
             large shortfall the audit reports",
            point.simultaneous
        );
        assert!(
            (point.simultaneous - audit).abs() < 0.02,
            "{label}: design is meant to reproduce the audit's {audit:.3} \
             pointwise joint coverage, measured {:.4}",
            point.simultaneous
        );
        // Sanity: the pointwise band is correct *pointwise*.
        assert!(
            (point.pointwise_cell0 - nominal).abs() < 0.01,
            "{label}: pointwise band per-cell coverage {:.4} off nominal",
            point.pointwise_cell0
        );
        // The sup-t band over-covers any single cell, by construction.
        assert!(
            sup.pointwise_cell0 > nominal,
            "{label}: sup-t per-cell coverage {:.4} should exceed nominal",
            sup.pointwise_cell0
        );
        // The two fallbacks are conservative under this positive dependence.
        assert!(
            sidak.simultaneous >= nominal,
            "{label}: Sidak simultaneous coverage {:.4} below nominal",
            sidak.simultaneous
        );
        assert!(
            bonf.simultaneous >= sidak.simultaneous,
            "{label}: Bonferroni should be at least as conservative as Sidak"
        );
    }
}
