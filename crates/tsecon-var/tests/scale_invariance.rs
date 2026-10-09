//! The units of the series must not matter where theory says they cannot.
//!
//! Rescaling series `i` of a VAR by `s_i` is the diagonal similarity
//! `A_l -> S A_l S^{-1}` of every lag matrix: the roots, the stability
//! verdict and every stationarity decision are unchanged, and each response
//! of variable `i` is multiplied by `s_i`. The companion eigenvalue solve is
//! the one step whose ACCURACY depended on the units (series `1e14` apart
//! give a companion with entries spanning `3e-15 .. 8e13`); it now runs on
//! an exactly balanced matrix (`tsecon_linalg::balance`).
//!
//! * `golden_var_roots_on_badly_scaled_data` — the roots of a VAR(2) fitted
//!   to the audit's near-unit-root sample in units `(1e7, 1, 1e-7)` hit a
//!   60-digit mpmath solve (`fixtures/companion_roots.json`). The
//!   unbalanced solve gave `min_root = 0.997715` against `1.000887`, the
//!   wrong side of 1.
//! * `kilian_bands_and_unstable_draws_do_not_depend_on_units` — with
//!   `bias_correct`, the Kilian shrink (`delta -= 0.01` until the corrected
//!   companion radius is below 1) and `n_unstable_draws` are decided by that
//!   solve; both bootstrap engines now give the same counts and the same
//!   bands (row `i` divided by `s_i`) in either unit system. Before the
//!   balancing the bands moved by up to 1.25e-2 and `n_unstable_draws` went
//!   from 368 to 496.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::{as_mat, load_fixture};
use tsecon_linalg::faer::Mat;
use tsecon_var::{bootstrap_irf_bands, long_run_bootstrap_bands, LongRunBandSpec, Trend, VarSpec};

fn scaled_data() -> (Mat<f64>, Vec<f64>) {
    let fx = load_fixture("companion_roots.json");
    let block = &fx["var_fit_scaled"];
    let scale: Vec<f64> = block["scale"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    (as_mat(&block["data"]), scale)
}

/// `max |a_i / s_i - b| / max |b|` over a cube, row `i` of `a` in units `s_i`.
fn worst_rel(a: &[Mat<f64>], b: &[Mat<f64>], s: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (ma, mb) in a.iter().zip(b) {
        for j in 0..mb.ncols() {
            for i in 0..mb.nrows() {
                num = num.max((ma[(i, j)] / s[i] - mb[(i, j)]).abs());
                den = den.max(mb[(i, j)].abs());
            }
        }
    }
    num / den
}

#[test]
fn golden_var_roots_on_badly_scaled_data() {
    let fx = load_fixture("companion_roots.json");
    let block = &fx["var_fit_scaled"];
    let (data, _) = scaled_data();
    let lags = block["lags"].as_u64().unwrap() as usize;
    let res = VarSpec::new(lags, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap();
    let roots = res.roots_moduli().unwrap();
    let expected: Vec<f64> = block["roots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(roots.len(), expected.len());
    for (i, (r, e)) in roots.iter().zip(&expected).enumerate() {
        assert!(
            (r - e).abs() <= 1e-9 * e,
            "root {i}: {r:.15} vs mpmath {e:.15} (rel {:.2e})",
            (r - e).abs() / e
        );
    }
    let min_root = block["min_root"].as_f64().unwrap();
    assert!(min_root > 1.0 + 1e-4);
    assert!(res.is_stable().unwrap(), "a stable VAR read as unstable");
}

#[test]
fn kilian_bands_and_unstable_draws_do_not_depend_on_units() {
    let (scaled, s) = scaled_data();
    let unscaled = Mat::from_fn(scaled.nrows(), scaled.ncols(), |t, j| scaled[(t, j)] / s[j]);

    // Blanchard-Quah bands, Kilian-corrected.
    let mut spec = LongRunBandSpec::new(2, 4);
    spec.n_boot = 200;
    spec.seed = 3;
    spec.bias_correct = true;
    let a = long_run_bootstrap_bands(scaled.as_ref(), &spec).unwrap();
    let b = long_run_bootstrap_bands(unscaled.as_ref(), &spec).unwrap();
    assert_eq!(a.n_unstable_draws, b.n_unstable_draws);
    assert!((a.min_root - b.min_root).abs() <= 1e-12 * b.min_root);
    assert!(a.min_root > 1.0, "min_root {}", a.min_root);
    for (what, x, y) in [
        ("point", &a.point, &b.point),
        ("se", &a.se, &b.se),
        ("lower", &a.lower, &b.lower),
        ("upper", &a.upper, &b.upper),
    ] {
        let rel = worst_rel(x, y, &s);
        assert!(
            rel <= 1e-9,
            "long_run_bootstrap_bands {what}: rel change {rel:.2e}"
        );
    }

    // Cholesky IRF bands, Kilian-corrected.
    let a = bootstrap_irf_bands(
        scaled.as_ref(),
        2,
        Trend::Constant,
        4,
        true,
        false,
        0.1,
        200,
        3,
        true,
    )
    .unwrap();
    let b = bootstrap_irf_bands(
        unscaled.as_ref(),
        2,
        Trend::Constant,
        4,
        true,
        false,
        0.1,
        200,
        3,
        true,
    )
    .unwrap();
    for (what, x, y) in [
        ("point", &a.point, &b.point),
        ("se", &a.se, &b.se),
        ("lower", &a.lower, &b.lower),
        ("upper", &a.upper, &b.upper),
    ] {
        let rel = worst_rel(x, y, &s);
        assert!(
            rel <= 1e-9,
            "bootstrap_irf_bands {what}: rel change {rel:.2e}"
        );
    }
}
