//! Simultaneous (sup-t) confidence bands for local-projection impulse
//! responses.
//!
//! Four things need evidence here, and the file is organised in that order:
//!
//! 1. **The pointwise output did not move.** Every band route must return the
//!    `lp` / `smooth_lp` point estimates and standard errors bit for bit —
//!    they are golden-gated against statsmodels and linearmodels, and this
//!    feature is not allowed to touch them.
//! 2. **The cross-horizon covariance is the same estimator as the reported
//!    standard errors.** `sqrt(diag(Sigma))` must reproduce `lp(...).se` on the
//!    lag-augmented path, and *exactly* reproduce `smooth_lp(...).se`. Anything
//!    else means the band's correlation structure came from a different
//!    estimator than its widths.
//! 3. **The multiplier behaves.** Ordering across the four routes, collapse to
//!    the pointwise value at `K = 1`, reproducibility from the seed.
//! 4. **The band actually covers the path.** A seeded Monte Carlo on a
//!    known-truth DGP, measuring joint coverage of the *whole* horizon path
//!    under each route, alongside the per-horizon marginal coverage that the
//!    pointwise band gets right. That last test is the acceptance test for the
//!    whole feature; the rest are invariants.

use serde_json::Value;
use tsecon_lp::{
    closed_form_band, lp, lp_band, lp_irf_cov, lp_iv, smooth_lp, smooth_lp_band, BandMethod,
    BandSpec, LpError, LpSpec, SeKind, SmoothLpSpec,
};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

fn load_fixture() -> Value {
    let path = format!("{}/../../fixtures/lp.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).expect("fixture file readable");
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn gaussian(stream: &mut Stream) -> f64 {
    let u = stream.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// `s_t = rho s_{t-1} + e_t`, `y_t = s_t + sigma w_t`, returned after a
/// burn-in. `e` is i.i.d. and independent of everything dated `t - 1` and
/// earlier, so the local-projection estimand is **exactly** `rho^h` at every
/// horizon whatever controls are used — which is what makes this a known-truth
/// design for a coverage measurement.
fn simulate(stream: &mut Stream, n: usize, rho: f64, sigma: f64) -> (Vec<f64>, Vec<f64>) {
    let burn = 100;
    let total = n + burn;
    let mut s = 0.0;
    let mut y = Vec::with_capacity(total);
    let mut e = Vec::with_capacity(total);
    for _ in 0..total {
        let et = gaussian(stream);
        s = rho * s + et;
        let w = gaussian(stream);
        y.push(s + sigma * w);
        e.push(et);
    }
    (y[burn..].to_vec(), e[burn..].to_vec())
}

/// Bitwise equality of two float paths. `assert_eq!` on `f64` would accept
/// `0.0 == -0.0`; nothing here should differ even by a sign bit.
fn assert_bit_identical(what: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{what}: length changed");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: entry {i} changed, {x} vs {y}"
        );
    }
}

// ---------------------------------------------------------------------------
// 1. The pointwise output did not move
// ---------------------------------------------------------------------------

#[test]
fn every_band_route_returns_the_lp_output_bit_for_bit() {
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let spec = LpSpec::new(12, 4);
    let plain = lp(&y, &e, spec).expect("plain lp");

    for method in [
        BandMethod::Pointwise,
        BandMethod::SupT,
        BandMethod::Sidak,
        BandMethod::Bonferroni,
    ] {
        let out = lp_band(&y, &e, spec, BandSpec::new(method, 0.10).with_n_sim(20_000))
            .unwrap_or_else(|err| panic!("lp_band({}) failed: {err}", method.label()));
        assert_bit_identical(&format!("{} irf", method.label()), &plain.irf, &out.lp.irf);
        assert_bit_identical(&format!("{} se", method.label()), &plain.se, &out.lp.se);
        assert_eq!(plain.nobs_per_h, out.lp.nobs_per_h);
        assert_eq!(plain.se_kind, out.lp.se_kind);
    }

    // Same promise on the HAC path, which is the statsmodels-parity one.
    let hac_spec = LpSpec::new(8, 4).with_hac(None);
    let plain_hac = lp(&y, &e, hac_spec).expect("plain hac lp");
    let out = lp_band(&y, &e, hac_spec, BandSpec::sup_t(0.10).with_n_sim(20_000))
        .expect("hac sup-t band");
    assert_bit_identical("hac irf", &plain_hac.irf, &out.lp.irf);
    assert_bit_identical("hac se", &plain_hac.se, &out.lp.se);
}

#[test]
fn smooth_lp_band_returns_the_smooth_lp_output_bit_for_bit() {
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let spec = SmoothLpSpec::new(12, 4).with_lambda(10.0);
    let plain = smooth_lp(&y, &e, &spec).expect("plain smooth_lp");
    let out = smooth_lp_band(&y, &e, &spec, BandSpec::sup_t(0.10).with_n_sim(20_000))
        .expect("smooth sup-t band");
    assert_bit_identical("smooth irf", &plain.irf, &out.smooth.irf);
    assert_bit_identical("smooth se", &plain.se, &out.smooth.se);
    assert_bit_identical("smooth theta", &plain.theta, &out.smooth.theta);
}

// ---------------------------------------------------------------------------
// 2. The covariance is the same estimator as the reported standard errors
// ---------------------------------------------------------------------------

#[test]
fn lag_augmented_covariance_diagonal_reproduces_the_reported_standard_errors() {
    // The load-bearing check. The band's *widths* come from `lp(...).se` and
    // its *shape* comes from `lp_irf_cov(...)`; if those two disagreed, the
    // correlation would be describing a different estimator than the widths.
    // On the lag-augmented path the two are algebraically identical (the
    // Frisch-Waugh-Lovell influence function squared and summed IS the HC1
    // variance), so only floating point separates them.
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);

    for (hmax, p) in [(12usize, 4usize), (6, 2), (3, 6)] {
        let spec = LpSpec::new(hmax, p);
        let res = lp(&y, &e, spec).expect("lp");
        let cov = lp_irf_cov(&y, &e, spec).expect("cov");
        assert_eq!(cov.se_kind, SeKind::LagAugmentedHc1);
        assert_eq!(cov.bandwidth, 0.0, "lag-augmented needs no kernel");
        let worst = cov
            .se
            .iter()
            .zip(&res.se)
            .map(|(a, b)| (a - b).abs() / b)
            .fold(0.0_f64, f64::max);
        println!("H={hmax} p={p}: max relative se gap {worst:.3e}");
        assert!(
            worst < 1e-10,
            "H={hmax} p={p}: sqrt(diag(Sigma)) differs from lp().se by {worst} \
             relative — the band's correlation and its widths are not the same \
             estimator"
        );
    }
}

#[test]
fn covariance_is_symmetric_psd_and_strongly_correlated_across_adjacent_horizons() {
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let spec = LpSpec::new(12, 4);
    let cov = lp_irf_cov(&y, &e, spec).expect("cov");
    let k = cov.horizons.len();
    assert_eq!(k, 13);
    assert_eq!(cov.cov.len(), k * k);

    // Exact symmetry: the shared sup-t routine checks it, and a row/column
    // mix-up is the classic silent bug it is guarding against.
    for i in 0..k {
        for j in 0..k {
            assert_eq!(
                cov.cov[i * k + j].to_bits(),
                cov.cov[j * k + i].to_bits(),
                "Sigma is not exactly symmetric at ({i},{j})"
            );
        }
    }

    // Correlations are real correlations, and adjacent horizons are positively
    // correlated — which is why the union-bound routes over-pay and sup-t does
    // not. (The h = 0 -> h = 1 pair is the weakest link on this fixture: the
    // impact residual is pure measurement noise, which the h = 1 residual does
    // not share, so the two are only mildly correlated. Nothing about the
    // construction assumes otherwise.)
    let mut adjacent = Vec::new();
    for i in 0..k {
        for j in 0..k {
            let r = cov.cov[i * k + j] / (cov.se[i] * cov.se[j]);
            assert!(
                r.abs() <= 1.0 + 1e-9,
                "correlation({i},{j}) = {r} is outside [-1, 1]"
            );
            if j == i + 1 {
                adjacent.push(r);
            }
        }
    }
    let adjacent_min = adjacent.iter().copied().fold(f64::INFINITY, f64::min);
    let adjacent_mean = adjacent.iter().sum::<f64>() / adjacent.len() as f64;
    println!("adjacent-horizon correlations: min {adjacent_min:.4}, mean {adjacent_mean:.4}");
    assert!(
        adjacent_min > 0.0,
        "no adjacent pair of a persistent IRF should be negatively correlated; \
         smallest adjacent correlation was {adjacent_min}"
    );
    assert!(
        adjacent_mean > 0.35,
        "adjacent horizons of a persistent IRF should be positively correlated \
         on average; mean adjacent correlation was {adjacent_mean}"
    );

    // Positive semi-definiteness, checked independently of the sup-t routine's
    // own Cholesky: every quadratic form on a pseudo-random set of directions.
    let mut stream = Stream::new(11_235_813);
    for _ in 0..64 {
        let v: Vec<f64> = (0..k).map(|_| gaussian(&mut stream)).collect();
        let mut q = 0.0;
        for i in 0..k {
            for j in 0..k {
                q += v[i] * cov.cov[i * k + j] * v[j];
            }
        }
        assert!(q >= -1e-18, "quadratic form {q} < 0: Sigma is not PSD");
    }
}

#[test]
fn smooth_lp_covariance_diagonal_is_exactly_the_reported_standard_error() {
    // Smooth LP is the one estimator that already had the joint covariance:
    // one spline coefficient vector serves every horizon, so `B V B'` is a
    // by-product. The diagonal is accumulated in the order the per-horizon
    // variance always was, so this is bit equality, not a tolerance.
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let res = smooth_lp(&y, &e, &SmoothLpSpec::new(12, 4).with_lambda(5.0)).expect("smooth_lp");
    let k = res.horizons.len();
    assert_eq!(res.cov.len(), k * k);
    for h in 0..k {
        assert_eq!(
            res.cov[h * k + h].sqrt().to_bits(),
            res.se[h].to_bits(),
            "h={h}: sqrt(cov diagonal) is not bit-identical to the reported se"
        );
        for g in 0..k {
            assert_eq!(
                res.cov[h * k + g].to_bits(),
                res.cov[g * k + h].to_bits(),
                "smooth covariance not exactly symmetric at ({h},{g})"
            );
        }
    }
}

#[test]
fn the_hac_path_uses_one_common_bandwidth_and_reports_the_drift() {
    // The documented compromise: with the default horizon-growing
    // `maxlags = h + p` every horizon has its own bandwidth, and a stacked
    // kernel estimator with per-cell bandwidths is not guaranteed PSD. So the
    // matrix uses one common bandwidth, the diagonal then drifts from the
    // reported per-horizon SEs, and the band says by how much.
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let (hmax, p) = (8usize, 4usize);

    let growing = LpSpec::new(hmax, p).with_hac(None);
    let cov = lp_irf_cov(&y, &e, growing).expect("hac cov");
    assert_eq!(cov.bandwidth, (hmax + p) as f64);
    assert_eq!(cov.se_kind, SeKind::HacBartlett);
    let out =
        lp_band(&y, &e, growing, BandSpec::sup_t(0.10).with_n_sim(20_000)).expect("hac sup-t band");
    let drift = out.band.cov_se_max_rel_diff.expect("drift reported");
    println!(
        "HAC, growing maxlags: common bandwidth {}, diagonal drift {drift:.4}",
        cov.bandwidth
    );
    assert!(
        drift > 1e-6,
        "with per-horizon bandwidths the diagonal must visibly differ from the \
         common-bandwidth one; got {drift}"
    );
    // The widths are still the reported ones, drift or no drift.
    let plain = lp(&y, &e, growing).expect("hac lp");
    for h in 0..=hmax {
        let half = out.band.critical_value * plain.se[h];
        assert!((out.band.upper[h] - (plain.irf[h] + half)).abs() < 1e-15);
    }

    // Fixing `maxlags` removes the compromise: every horizon already shares a
    // bandwidth, so the diagonal lines up again.
    let fixed = LpSpec::new(hmax, p).with_hac(Some(6));
    let fixed_res = lp(&y, &e, fixed).expect("fixed hac lp");
    let fixed_cov = lp_irf_cov(&y, &e, fixed).expect("fixed hac cov");
    assert_eq!(fixed_cov.bandwidth, 6.0);
    let worst = fixed_cov
        .se
        .iter()
        .zip(&fixed_res.se)
        .map(|(a, b)| (a - b).abs() / b)
        .fold(0.0_f64, f64::max);
    println!("HAC, fixed maxlags = 6: max relative se gap {worst:.3e}");
    assert!(
        worst < 1e-10,
        "with a fixed maxlags the covariance diagonal should reproduce the \
         reported HAC standard errors; worst relative gap {worst}"
    );
}

// ---------------------------------------------------------------------------
// 3. The multiplier behaves
// ---------------------------------------------------------------------------

#[test]
fn critical_values_are_ordered_pointwise_le_sup_t_le_sidak_le_bonferroni() {
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let spec = LpSpec::new(12, 4);
    let alpha = 0.10;

    let mut c = std::collections::BTreeMap::new();
    for method in [
        BandMethod::Pointwise,
        BandMethod::SupT,
        BandMethod::Sidak,
        BandMethod::Bonferroni,
    ] {
        let out = lp_band(
            &y,
            &e,
            spec,
            BandSpec::new(method, alpha).with_n_sim(100_000),
        )
        .expect("band");
        assert_eq!(out.band.method, method);
        assert_eq!(out.band.n_cells, 13);
        assert_eq!(out.band.n_cells_used, 13);
        c.insert(method.label(), out.band.critical_value);
    }
    println!("K=13, alpha=0.10 critical values: {c:?}");

    let z = c["pointwise"];
    let supt = c["sup-t"];
    let sidak = c["sidak"];
    let bonf = c["bonferroni"];
    assert!((z - 1.644_853_626_951_5).abs() < 1e-9, "pointwise z = {z}");
    assert!(
        supt >= z,
        "sup-t {supt} must never fall below pointwise {z}"
    );
    assert!(
        supt > z + 0.1,
        "sup-t {supt} is suspiciously close to the pointwise floor {z}; the \
         floor may be absorbing a covariance/se mismatch"
    );
    assert!(
        supt < sidak,
        "sup-t {supt} should beat Sidak {sidak} on a correlated IRF path"
    );
    assert!(sidak < bonf, "Sidak {sidak} should beat Bonferroni {bonf}");
    assert!((sidak - 2.648_98).abs() < 1e-3, "Sidak at K=13: {sidak}");
    assert!((bonf - 2.665_31).abs() < 1e-3, "Bonferroni at K=13: {bonf}");
}

#[test]
fn a_single_horizon_collapses_every_route_to_the_pointwise_band() {
    // K = 1 has no multiplicity to correct, so "simultaneous" must degrade to
    // the ordinary band. That makes it always safe to ask for a sup-t band.
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let spec = LpSpec::new(0, 4);
    let z = lp_band(&y, &e, spec, BandSpec::pointwise(0.05))
        .expect("pointwise")
        .band
        .critical_value;

    for method in [BandMethod::SupT, BandMethod::Sidak, BandMethod::Bonferroni] {
        let out = lp_band(
            &y,
            &e,
            spec,
            BandSpec::new(method, 0.05).with_n_sim(200_000),
        )
        .expect("K=1 band");
        assert_eq!(out.band.n_cells, 1);
        let gap = (out.band.critical_value - z).abs();
        println!(
            "K=1 {}: c = {}, gap {gap:.2e}",
            method.label(),
            out.band.critical_value
        );
        let tol = if method == BandMethod::SupT {
            0.05
        } else {
            1e-12
        };
        assert!(
            gap < tol,
            "{} at K=1 gave {} but the pointwise value is {z}",
            method.label(),
            out.band.critical_value
        );
    }
}

#[test]
fn the_sup_t_band_is_a_pure_function_of_its_seed() {
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let spec = LpSpec::new(12, 4);

    let a = lp_band(
        &y,
        &e,
        spec,
        BandSpec::sup_t(0.10).with_seed(7).with_n_sim(20_000),
    )
    .expect("band a");
    let b = lp_band(
        &y,
        &e,
        spec,
        BandSpec::sup_t(0.10).with_seed(7).with_n_sim(20_000),
    )
    .expect("band b");
    assert_eq!(
        a.band.critical_value.to_bits(),
        b.band.critical_value.to_bits(),
        "same seed must reproduce the critical value bit for bit"
    );
    assert_bit_identical("band upper", &a.band.upper, &b.band.upper);

    let d = lp_band(
        &y,
        &e,
        spec,
        BandSpec::sup_t(0.10).with_seed(8).with_n_sim(20_000),
    )
    .expect("band d");
    let jitter = (d.band.critical_value - a.band.critical_value).abs();
    println!("seed 7 vs 8 at n_sim=20k: |dc| = {jitter:.4}");
    assert!(
        jitter > 0.0,
        "a different seed should give a different draw"
    );
    assert!(
        jitter < 0.10,
        "simulation noise {jitter} at n_sim = 20,000 is larger than expected"
    );
}

#[test]
fn the_band_edges_are_the_reported_ses_times_the_multiplier() {
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let spec = LpSpec::new(12, 4);
    let out = lp_band(&y, &e, spec, BandSpec::sup_t(0.10).with_n_sim(20_000)).expect("band");
    for h in 0..=12 {
        let half = out.band.critical_value * out.lp.se[h];
        assert!((out.band.lower[h] - (out.lp.irf[h] - half)).abs() < 1e-15);
        assert!((out.band.upper[h] - (out.lp.irf[h] + half)).abs() < 1e-15);
        assert!(out.band.upper[h] > out.band.lower[h]);
    }
    // And the simultaneous band strictly contains the pointwise one.
    let pw = lp_band(&y, &e, spec, BandSpec::pointwise(0.10)).expect("pointwise");
    for h in 0..=12 {
        assert!(out.band.lower[h] < pw.band.lower[h]);
        assert!(out.band.upper[h] > pw.band.upper[h]);
    }
}

#[test]
fn closed_form_bands_serve_the_paths_that_have_no_covariance() {
    // lp_iv / lp_multiplier / lp_state have no cross-horizon covariance in
    // this crate, so sup-t is refused rather than faked, and the closed forms
    // are available on their reported paths.
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let x = f64s(&fx["x"]);
    let z = f64s(&fx["z"]);
    let iv = lp_iv(&y, &x, &z, LpSpec::new(6, 4)).expect("lp_iv");

    let sidak = closed_form_band(&iv.irf, &iv.se, BandSpec::sidak(0.10)).expect("sidak on lp_iv");
    assert_eq!(sidak.n_cells, 7);
    assert!(sidak.critical_value > sidak.pointwise_critical_value);
    assert!(sidak.cov.is_none() && sidak.cov_se_max_rel_diff.is_none());
    for h in 0..7 {
        assert!((sidak.upper[h] - (iv.irf[h] + sidak.critical_value * iv.se[h])).abs() < 1e-15);
    }

    let refused = closed_form_band(&iv.irf, &iv.se, BandSpec::sup_t(0.10));
    assert!(
        matches!(refused, Err(LpError::Band { .. })),
        "sup-t without a covariance must be refused, not faked: {refused:?}"
    );
    let msg = format!("{}", refused.unwrap_err());
    assert!(
        msg.contains("sup-t"),
        "the error should name the problem: {msg}"
    );
}

#[test]
fn band_misconfiguration_is_rejected_with_an_error_that_teaches() {
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let spec = LpSpec::new(4, 4);

    for alpha in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
        let bad = lp_band(&y, &e, spec, BandSpec::pointwise(alpha));
        assert!(
            matches!(bad, Err(LpError::Band { .. })),
            "alpha = {alpha} should be rejected"
        );
    }
    let too_few = lp_band(&y, &e, spec, BandSpec::sup_t(0.10).with_n_sim(1));
    assert!(matches!(too_few, Err(LpError::Band { .. })));
    let msg = format!("{}", too_few.unwrap_err());
    assert!(
        msg.contains("n_sim") && msg.contains("50,000"),
        "the n_sim error should say what a usable value looks like: {msg}"
    );

    let mismatched = closed_form_band(&[1.0, 2.0], &[0.1], BandSpec::sidak(0.10));
    assert!(matches!(mismatched, Err(LpError::LengthMismatch { .. })));
}

// ---------------------------------------------------------------------------
// 4. Does the band actually cover the path?
// ---------------------------------------------------------------------------

/// One Monte Carlo cell: joint and marginal coverage at a given sample size.
struct McResult {
    marginal: Vec<f64>,
    joint: std::collections::BTreeMap<&'static str, f64>,
    mean_c: std::collections::BTreeMap<&'static str, f64>,
}

const ROUTES: [&str; 4] = ["pointwise", "sup-t", "sidak", "bonferroni"];

fn monte_carlo(reps: usize, n: usize, hmax: usize, rho: f64, alpha: f64, seed: u64) -> McResult {
    // n_sim is deliberately small here: the critical value is averaged over
    // `reps` samples, so a per-sample simulation error of ~0.01 on a
    // multiplier of ~2.5 is far below the Monte Carlo noise in the coverage
    // rate itself. Production default is DEFAULT_BAND_N_SIM = 100_000.
    let n_sim = 4_000;
    let mut stream = Stream::new(seed);
    let truth: Vec<f64> = (0..=hmax).map(|h| rho.powi(h as i32)).collect();
    let mut marginal = vec![0usize; hmax + 1];
    let mut joint = std::collections::BTreeMap::new();
    let mut mean_c = std::collections::BTreeMap::new();
    for label in ROUTES {
        joint.insert(label, 0usize);
        mean_c.insert(label, 0.0_f64);
    }

    for rep in 0..reps {
        let (y, e) = simulate(&mut stream, n, rho, 1.0);
        let spec = LpSpec::new(hmax, 4);
        // One sup-t call gives the estimates, the pointwise multiplier, and
        // the sup-t multiplier; the closed forms need only K.
        let out = lp_band(
            &y,
            &e,
            spec,
            BandSpec::sup_t(alpha)
                .with_n_sim(n_sim)
                .with_seed(seed ^ (0x5EED_0000 + rep as u64)),
        )
        .expect("lp_band in the Monte Carlo");
        let sidak = closed_form_band(&out.lp.irf, &out.lp.se, BandSpec::sidak(alpha))
            .expect("sidak")
            .critical_value;
        let bonf = closed_form_band(&out.lp.irf, &out.lp.se, BandSpec::bonferroni(alpha))
            .expect("bonferroni")
            .critical_value;

        for (label, c) in [
            ("pointwise", out.band.pointwise_critical_value),
            ("sup-t", out.band.critical_value),
            ("sidak", sidak),
            ("bonferroni", bonf),
        ] {
            *mean_c.get_mut(label).expect("label") += c / reps as f64;
            let all = (0..=hmax).all(|h| (out.lp.irf[h] - truth[h]).abs() <= c * out.lp.se[h]);
            if all {
                *joint.get_mut(label).expect("label") += 1;
            }
        }
        let z = out.band.pointwise_critical_value;
        for h in 0..=hmax {
            if (out.lp.irf[h] - truth[h]).abs() <= z * out.lp.se[h] {
                marginal[h] += 1;
            }
        }
    }

    McResult {
        marginal: marginal.iter().map(|c| *c as f64 / reps as f64).collect(),
        joint: joint
            .into_iter()
            .map(|(k, v)| (k, v as f64 / reps as f64))
            .collect(),
        mean_c,
    }
}

#[test]
fn monte_carlo_joint_coverage_of_the_whole_horizon_path() {
    // The acceptance test for this feature. A nominal 90% band, a known-truth
    // DGP (the LP estimand is exactly rho^h at every horizon), two sample
    // sizes, and two questions asked of the same samples:
    //
    //   * marginal — does horizon h's interval cover beta_h? The pointwise
    //     band is built to get this right, and it roughly does.
    //   * joint    — does the band contain the WHOLE 13-horizon path? The
    //     pointwise band is not built to get this right, and it badly does
    //     not — and the failure does not shrink when T grows, because it is
    //     multiplicity, not inconsistency.
    //
    // Two sample sizes are run precisely to show that last point: if the
    // pointwise joint shortfall were a small-sample artefact it would close
    // between them, and it does not.
    let reps = 400usize;
    let hmax = 12usize;
    let rho = 0.9_f64;
    let alpha = 0.10;

    // `min_joint` is the floor each simultaneous route must clear at that
    // sample size. It is below the nominal 0.90 at T = 240 on purpose: the
    // *pointwise* intervals themselves run 2-5 points short at the long
    // horizons there (see the printed marginals), and a sup-t band inherits
    // whatever its pointwise standard errors get wrong — the multiplier can
    // only fix multiplicity. By T = 720 the marginals are on nominal and the
    // sup-t joint rate lands on nominal with them, which is the cleanest
    // statement of what this feature does and does not do.
    let mut cells = Vec::new();
    for (i, (n, min_joint)) in [(240usize, 0.74), (720usize, 0.83)].into_iter().enumerate() {
        let mc = monte_carlo(reps, n, hmax, rho, alpha, 20_260_807 + i as u64);
        println!(
            "\nMonte Carlo: reps={reps}, T={n}, K={} horizons, rho={rho}, nominal {:.0}% band",
            hmax + 1,
            100.0 * (1.0 - alpha)
        );
        let mut margins: Vec<String> = Vec::new();
        for h in 0..=hmax {
            margins.push(format!("{:.3}", mc.marginal[h]));
        }
        println!(
            "  pointwise MARGINAL coverage by horizon: [{}]",
            margins.join(", ")
        );
        for label in ROUTES {
            println!(
                "  JOINT coverage of the whole path, {label:>11}: {:.3}   (mean c = {:.3})",
                mc.joint[label], mc.mean_c[label]
            );
        }
        cells.push((n, min_joint, mc));
    }

    for (n, min_joint, mc) in &cells {
        // Marginal coverage of the pointwise band is in the right
        // neighbourhood at both sample sizes — the joint failure below is a
        // multiplicity problem, not a standard-error problem.
        for h in 0..=hmax {
            assert!(
                (0.79..=0.96).contains(&mc.marginal[h]),
                "T={n} h={h}: pointwise marginal coverage {} is far off the \
                 nominal 0.90, which would mean the standard errors are wrong \
                 rather than the multiplier",
                mc.marginal[h]
            );
        }
        // Joint coverage of the pointwise band is not in the right
        // neighbourhood. That gap is the whole reason this module exists.
        assert!(
            mc.joint["pointwise"] < 0.60,
            "T={n}: pointwise joint coverage {} was not materially below the \
             nominal 0.90 — the demonstration this feature rests on did not \
             reproduce",
            mc.joint["pointwise"]
        );
        // Every simultaneous route repairs most of it.
        for label in ["sup-t", "sidak", "bonferroni"] {
            assert!(
                mc.joint[label] > mc.joint["pointwise"] + 0.25,
                "T={n}: {label} joint coverage {} barely beats pointwise {}",
                mc.joint[label],
                mc.joint["pointwise"]
            );
            assert!(
                mc.joint[label] >= *min_joint,
                "T={n}: {label} joint coverage {} fell below {min_joint} \
                 against a nominal 0.90",
                mc.joint[label]
            );
        }
        // Sup-t is the tight one: same coverage target, smaller multiplier.
        assert!(
            mc.mean_c["sup-t"] < mc.mean_c["sidak"],
            "T={n}: mean sup-t multiplier {} should sit below Sidak's {}",
            mc.mean_c["sup-t"],
            mc.mean_c["sidak"]
        );
    }

    // The multiplicity gap does not close with the sample. Tripling T leaves
    // the pointwise joint rate far from nominal, while the simultaneous rate
    // converges on it: at T = 720 the pointwise *marginals* are on nominal and
    // the sup-t *joint* rate is on nominal with them, which is the sharpest
    // available statement that the residual gap at T = 240 comes from the
    // standard errors and not from the multiplier.
    let (n_small, _, small) = &cells[0];
    let (n_big, _, big) = &cells[1];
    println!(
        "\n  T={n_small} -> T={n_big}: pointwise joint {:.3} -> {:.3}; sup-t joint {:.3} -> {:.3}",
        small.joint["pointwise"], big.joint["pointwise"], small.joint["sup-t"], big.joint["sup-t"]
    );
    assert!(
        big.joint["pointwise"] < 0.60,
        "tripling T should not repair the pointwise joint rate ({} at T={n_big}); \
         if it did, the shortfall would be a small-sample artefact rather than \
         multiplicity",
        big.joint["pointwise"]
    );
    assert!(
        (big.joint["sup-t"] - (1.0 - alpha)).abs() < 0.07,
        "at T={n_big}, where the pointwise marginals are on nominal, the sup-t \
         joint rate {} should land on the nominal {}",
        big.joint["sup-t"],
        1.0 - alpha
    );
}
