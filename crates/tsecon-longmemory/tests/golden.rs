//! Documented-formula golden tests.
//!
//! `fixtures/longmemory.json` is produced by
//! `fixtures/generate_longmemory_fixtures.py`, which computes every published
//! quantity by literally writing the closed form in NumPy (no call to this
//! crate). Matching it proves the Rust reproduces the documented algebra:
//!
//! * fractional differencing / integration weights, filter, and exact
//!   round-trip inverse — to ~1e-12;
//! * the GPH log-periodogram regression `d`, its large-`m` closed form
//!   `pi/sqrt(24 m)` (the field `se_asymptotic`), and the OLS nonrobust slope
//!   SE — to ~1e-8;
//! * the Robinson (1995) local-Whittle minimizer `d` and its large-`m` closed
//!   form `1/(2 sqrt(m))` (the field `se_asymptotic`) — to ~1e-6.
//!
//! The headline `se` fields are the *bandwidth-exact* expressions those two
//! closed forms are limits of. The fixture predates them, so they are checked
//! here by writing the documented formula out longhand from `(n, m)` alone
//! — no periodogram, no OLS, nothing from the crate's code path — plus the
//! limit relation `se / se_asymptotic -> 1` as `m` grows.
//!
//! These pin the algebra only; the statistical recovery of a true `d`, and the
//! calibration of `se` against the realised sampling dispersion, is what
//! `properties.rs` establishes by Monte-Carlo.

use serde_json::Value;
use tsecon_longmemory::{frac_diff, frac_diff_weights, frac_integrate, gph, local_whittle};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/longmemory.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn g(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < tol || rel < tol,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e}"
    );
}

fn close_vec(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        close(*a, *e, tol, &format!("{what}[{i}]"));
    }
}

#[test]
fn fracdiff_weights_filter_and_inverse_match_documented_formula() {
    let fx = load();
    let cases = fx["fracdiff"]["cases"].as_array().expect("cases array");
    for case in cases {
        let d = g(&case["d"]);
        let nw = case["n_weights"].as_u64().expect("n_weights") as usize;
        let x = f64s(&case["x"]);

        let w = frac_diff_weights(d, nw).expect("weights");
        close_vec(
            &w,
            &f64s(&case["weights"]),
            1e-12,
            &format!("weights(d={d})"),
        );

        let fd = frac_diff(&x, d).expect("frac_diff");
        close_vec(
            &fd,
            &f64s(&case["frac_diff"]),
            1e-12,
            &format!("frac_diff(d={d})"),
        );

        let fi = frac_integrate(&x, d).expect("frac_integrate");
        close_vec(
            &fi,
            &f64s(&case["frac_integrate"]),
            1e-12,
            &format!("frac_integrate(d={d})"),
        );

        // The documented exact-inverse property, also pinned in the fixture.
        let rt = frac_integrate(&frac_diff(&x, d).expect("fd"), d).expect("fi");
        close_vec(
            &rt,
            &f64s(&case["roundtrip"]),
            1e-12,
            &format!("roundtrip(d={d})"),
        );
        // ...and it recovers the original series.
        close_vec(&rt, &x, 1e-10, &format!("roundtrip==x(d={d})"));
    }
}

/// The documented GPH and local-Whittle SE expressions written out longhand
/// from `(n, m)` alone, independent of the crate's estimation path.
///
/// Returns `(gph_se, gph_se_asymptotic, whittle_se, whittle_se_asymptotic)`.
fn documented_ses(n: usize, m: usize) -> (f64, f64, f64, f64) {
    let pi = std::f64::consts::PI;
    let lambdas: Vec<f64> = (1..=m)
        .map(|j| 2.0 * pi * (j as f64) / (n as f64))
        .collect();
    // GPH regressor R_j = -2 log(2 sin(lambda_j / 2)).
    let r: Vec<f64> = lambdas
        .iter()
        .map(|&l| -2.0 * (2.0 * (l / 2.0).sin()).ln())
        .collect();
    let r_bar = r.iter().sum::<f64>() / m as f64;
    let ss_r: f64 = r.iter().map(|&v| (v - r_bar) * (v - r_bar)).sum();
    // Local-Whittle nu_j = log lambda_j - mean(log lambda).
    let ll: Vec<f64> = lambdas.iter().map(|&l| l.ln()).collect();
    let ll_bar = ll.iter().sum::<f64>() / m as f64;
    let s_nu: f64 = ll.iter().map(|&v| (v - ll_bar) * (v - ll_bar)).sum();
    (
        (pi * pi / 6.0 / ss_r).sqrt(),
        pi / (24.0 * m as f64).sqrt(),
        1.0 / (2.0 * s_nu.sqrt()),
        1.0 / (2.0 * (m as f64).sqrt()),
    )
}

#[test]
fn gph_matches_documented_regression() {
    let fx = load();
    let s = &fx["semiparametric"];
    let x = f64s(&s["x"]);
    let m = s["m"].as_u64().expect("m") as usize;
    let fit = gph(&x, m).expect("gph");
    let e = &s["gph"];
    close(fit.d, g(&e["d"]), 1e-8, "gph.d");
    // The fixture's `se` is the large-m closed form, now reported separately.
    close(
        fit.se_asymptotic,
        g(&e["se"]),
        1e-12,
        "gph.se_asymptotic (pi/sqrt(24m))",
    );
    let (want_se, want_asym, _, _) = documented_ses(x.len(), m);
    close(fit.se_asymptotic, want_asym, 1e-14, "gph.se_asymptotic");
    close(
        fit.se,
        want_se,
        1e-12,
        "gph.se (sqrt((pi^2/6) / sum (R_j - Rbar)^2))",
    );
    // The headline SE is strictly wider than its large-m limit at this
    // bandwidth — the whole point of reporting it.
    assert!(
        fit.se > fit.se_asymptotic * 1.10,
        "gph.se = {:.6} is not materially wider than the asymptotic {:.6} at m = {m}",
        fit.se,
        fit.se_asymptotic
    );
    close(
        fit.se_regression,
        g(&e["se_regression"]),
        1e-8,
        "gph.se_regression",
    );
    // The intercept absorbs the periodogram's overall normalization, which
    // differs between the raw NumPy |rfft|^2 and tsecon-spectral's density
    // scaling, so it is intentionally NOT golden-matched. d and both SEs are
    // invariant to that constant and ARE matched above.
    assert!(fit.intercept.is_finite());
    assert_eq!(fit.m, m);
}

#[test]
fn local_whittle_matches_documented_minimizer() {
    let fx = load();
    let s = &fx["semiparametric"];
    let x = f64s(&s["x"]);
    let m = s["m"].as_u64().expect("m") as usize;
    let fit = local_whittle(&x, m).expect("local_whittle");
    let e = &s["whittle"];
    close(fit.d, g(&e["d"]), 1e-6, "whittle.d");
    // The fixture's `se` is the large-m closed form, now reported separately.
    close(
        fit.se_asymptotic,
        g(&e["se"]),
        1e-12,
        "whittle.se_asymptotic (1/(2 sqrt m))",
    );
    let (_, _, want_se, want_asym) = documented_ses(x.len(), m);
    close(fit.se_asymptotic, want_asym, 1e-14, "whittle.se_asymptotic");
    close(
        fit.se,
        want_se,
        1e-12,
        "whittle.se (1/(2 sqrt(sum nu_j^2)))",
    );
    assert!(
        fit.se > fit.se_asymptotic * 1.10,
        "whittle.se = {:.6} is not materially wider than the asymptotic {:.6} at m = {m}",
        fit.se,
        fit.se_asymptotic
    );
    assert_eq!(fit.m, m);
}

/// The reported `se` is a genuine refinement, not a reparametrisation: it
/// exceeds the textbook constant at every usable bandwidth, by a factor that
/// decays monotonically toward 1 as `m` grows. A test at one bandwidth cannot
/// see this — which is exactly how a constant SE passed review.
#[test]
fn the_exact_ses_converge_to_their_textbook_limits_as_m_grows() {
    let n = 65_536_usize;
    let x: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.017).sin() + 0.5).collect();
    let mut prev_gph = f64::INFINITY;
    let mut prev_lw = f64::INFINITY;
    for &m in &[16_usize, 32, 64, 256, 1024, 4096] {
        let (se_g, asym_g, se_w, asym_w) = documented_ses(n, m);
        let rg = se_g / asym_g;
        let rw = se_w / asym_w;
        assert!(
            rg > 1.0 && rw > 1.0,
            "m = {m}: ratios {rg:.4}, {rw:.4} <= 1"
        );
        assert!(
            rg < prev_gph && rw < prev_lw,
            "m = {m}: ratios {rg:.4}/{rw:.4} did not fall below {prev_gph:.4}/{prev_lw:.4}"
        );
        prev_gph = rg;
        prev_lw = rw;

        // ...and the crate reports exactly these on a real series.
        let fg = gph(&x, m).expect("gph");
        let fw = local_whittle(&x, m).expect("local_whittle");
        close(fg.se, se_g, 1e-12, &format!("gph.se(m={m})"));
        close(fg.se_asymptotic, asym_g, 1e-14, &format!("gph.asym(m={m})"));
        close(fw.se, se_w, 1e-12, &format!("whittle.se(m={m})"));
        close(fw.se_asymptotic, asym_w, 1e-14, &format!("lw.asym(m={m})"));
    }
    // By m = 4096 the correction is under 2%; at m = 16 it is over 30%.
    assert!(
        prev_gph < 1.02 && prev_lw < 1.02,
        "the exact SEs have not converged to their limits by m = 4096: {prev_gph:.4}, {prev_lw:.4}"
    );
}
