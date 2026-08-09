//! Golden test for the Jentsch-Lunsford moving-block bootstrap bands
//! (`tsecon_var::proxy_bands`) against `fixtures/proxy_svar_bands.json`.
//!
//! # What kind of golden this is
//!
//! No external package implements JL moving-block bands for the proxy-SVAR
//! estimand — the R `svars` package has a moving-block bootstrap, but for
//! its own identification schemes, not this one — so there is no
//! third-party number to copy. This is therefore a **documented-formula
//! golden**: `fixtures/generate_proxy_svar_bands_fixtures.py` transcribes
//! the algorithm of `docs/roadmap/15-proxy-svar-bands.md` section A into
//! plain NumPy (with statsmodels' `VAR` cross-checking the reduced form) and
//! never imports `tsecon`, so agreement here is a cross-implementation
//! check rather than a restatement of one code path. That is a weaker
//! anchor than a statsmodels golden and it is labelled as such: it pins the
//! *arithmetic*, not the *theory*. The theory-level claims are carried by
//! the property and Monte-Carlo tests in `proxy_bands_props.rs`.
//!
//! # How the RNG is taken out of the comparison
//!
//! A NumPy transcription cannot reproduce this library's RNG, so the
//! fixture **pins the block starts**: a seeded `(B, N)` integer matrix is
//! written into the JSON, consumed by the generator, and fed to
//! [`tsecon_var::proxy_bands::proxy_svar_bands_from_starts`] here. The
//! randomness becomes a shared input; everything downstream of it —
//! position-wise centering, recursive reconstruction, re-estimation,
//! per-draw re-identification and re-normalization, both interval types —
//! is compared cell for cell.

mod common;

use common::{assert_rel_close, load_fixture};
use serde_json::Value;
use tsecon_linalg::faer::Mat;
use tsecon_var::proxy_bands::{
    position_centering, proxy_svar_bands_from_starts, ProxyBandMethod, ProxyBandSpec,
};
use tsecon_var::Trend;

/// Agreement with the NumPy transcription is essentially bit-level, and the
/// tolerance says so.
///
/// **Measured**, not assumed: the largest deviation over every pinned array
/// (`point`, both Hall endpoints, both Efron endpoints, `se`, all 150
/// `gamma_norm_draws`, the three scalar diagnostics) is `6.7e-16` — a few
/// ULP on `O(1)` quantities. faer's OLS and NumPy's `lstsq` take different
/// routes, and the bootstrap runs the whole chain (fit, MA recursion,
/// moment, ratio) inside every draw, but that does **not** visibly compound
/// here; a tolerance justified by compounding would be unearned. The 1e-10
/// used is therefore ~150,000x the observed deviation, and that margin is
/// headroom for a different SIMD width or BLAS reassociation on another
/// platform, not a claim about this one.
const RTOL: f64 = 1e-10;

struct Fx {
    /// The whole fixture document.
    fx: Value,
    /// The case being compared — the base case is spread at the top level,
    /// the sparse case is nested under `"sparse"`.
    case: Value,
    spec: ProxyBandSpec,
    data: Mat<f64>,
    proxy: Vec<f64>,
    starts: Vec<Vec<usize>>,
}

/// The healthy-proxy case: 72 of 78 dates available, every draw survives.
fn load() -> Fx {
    load_case(None)
}

/// The four-date case: draws genuinely fail, so the failure counters are
/// nonzero and the accounting assertions have something to be wrong about.
fn load_sparse() -> Fx {
    load_case(Some("sparse"))
}

fn load_case(name: Option<&str>) -> Fx {
    let fx = load_fixture("proxy_svar_bands.json");
    let case = match name {
        Some(k) => fx[k].clone(),
        None => fx.clone(),
    };
    let p = &fx["params"];
    let trend = match p["trend"].as_str().unwrap() {
        "c" => Trend::Constant,
        "n" => Trend::None,
        other => panic!("unknown trend {other:?}"),
    };
    let spec = ProxyBandSpec {
        lags: p["lags"].as_u64().unwrap() as usize,
        trend,
        horizon: p["horizon"].as_u64().unwrap() as usize,
        norm_var: p["norm_var"].as_u64().unwrap() as usize,
        unit: p["unit"].as_f64().unwrap(),
        alpha: p["alpha"].as_f64().unwrap(),
        // Per case: the two differ in block length, replication count and
        // proxy availability, and in nothing else.
        n_boot: case["n_boot"].as_u64().unwrap() as usize,
        seed: 0,
        method: ProxyBandMethod::MovingBlock,
        block_length: Some(case["block_length"].as_u64().unwrap() as usize),
        robust_f: true,
    };
    let rows = fx["data"].as_array().unwrap();
    let ncols = rows[0].as_array().unwrap().len();
    let data = Mat::from_fn(rows.len(), ncols, |i, j| {
        rows[i].as_array().unwrap()[j].as_f64().unwrap()
    });
    // JSON null marks a date where the instrument is unavailable.
    let proxy: Vec<f64> = case["proxy"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    let starts: Vec<Vec<usize>> = case["starts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    Fx {
        fx,
        case,
        spec,
        data,
        proxy,
        starts,
    }
}

/// Compare a `(H+1) x n` response array against the fixture.
fn check_array(actual: &[Vec<f64>], expected: &Value, what: &str) {
    let e = expected.as_array().unwrap();
    assert_eq!(actual.len(), e.len(), "{what}: horizon count");
    for (h, row) in actual.iter().enumerate() {
        let er = e[h].as_array().unwrap();
        assert_eq!(row.len(), er.len(), "{what}[{h}]: variable count");
        for (i, &v) in row.iter().enumerate() {
            assert_rel_close(
                v,
                er[i].as_f64().unwrap(),
                RTOL,
                &format!("{what}[{h}][{i}]"),
            );
        }
    }
}

/// Every pinned array of one case, compared cell for cell.
fn check_case(f: &Fx, what: &str) {
    let bands =
        proxy_svar_bands_from_starts(f.data.as_ref(), &f.proxy, &f.spec, &f.starts).unwrap();

    check_array(&bands.point, &f.case["point"], &format!("{what} point"));
    check_array(&bands.lower, &f.case["lower"], &format!("{what} lower"));
    check_array(&bands.upper, &f.case["upper"], &format!("{what} upper"));
    check_array(
        &bands.lower_efron,
        &f.case["lower_efron"],
        &format!("{what} lower_efron"),
    );
    check_array(
        &bands.upper_efron,
        &f.case["upper_efron"],
        &format!("{what} upper_efron"),
    );
    check_array(&bands.se, &f.case["se"], &format!("{what} se"));

    for (key, got) in [
        ("point_gamma_norm", bands.point_gamma_norm),
        ("point_first_stage_f", bands.point_first_stage_f),
        ("point_reliability", bands.point_reliability),
    ] {
        assert_rel_close(
            got,
            f.case[key].as_f64().unwrap(),
            RTOL,
            &format!("{what} {key}"),
        );
    }
    assert_eq!(
        bands.n_proxy,
        f.case["n_proxy"].as_u64().unwrap() as usize,
        "{what} n_proxy"
    );
}

/// The point estimate, both interval types, and the bootstrap standard
/// deviation reproduce the NumPy transcription cell for cell — on the
/// healthy-proxy case and on the four-date case, where 18 of 120 draws fail
/// and the quantiles are formed from the 102 that did not.
#[test]
fn golden_bands_match_the_numpy_transcription() {
    check_case(&load(), "base");
    check_case(&load_sparse(), "sparse");
}

/// The per-draw identifying moment matches draw by draw. This is the
/// joint-blocking detector: `gamma*[norm_var]` sits around the sample value
/// under joint blocking and would be centered at zero if `u` and `m` were
/// resampled independently, so pinning the whole sequence pins the pairing.
#[test]
fn golden_gamma_norm_draws_match_draw_by_draw() {
    let f = load();
    let bands =
        proxy_svar_bands_from_starts(f.data.as_ref(), &f.proxy, &f.spec, &f.starts).unwrap();
    let expected = f.case["gamma_norm_draws"].as_array().unwrap();
    assert_eq!(bands.gamma_norm_draws.len(), expected.len());
    for (r, e) in expected.iter().enumerate() {
        match e.as_f64() {
            Some(v) => assert_rel_close(
                bands.gamma_norm_draws[r],
                v,
                RTOL,
                &format!("gamma_norm_draws[{r}]"),
            ),
            // JSON null marks a failed draw.
            None => assert!(
                bands.gamma_norm_draws[r].is_nan(),
                "draw {r} should have failed"
            ),
        }
    }
    // And the draws really do sit around the sample value rather than zero.
    let finite: Vec<f64> = bands
        .gamma_norm_draws
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    assert!(
        (mean - bands.point_gamma_norm).abs() < 0.5 * bands.point_gamma_norm.abs(),
        "mean gamma*[norm_var] {mean} is far from the sample value {} — the signature of \
         independent (rather than joint) block indices",
        bands.point_gamma_norm
    );
}

/// `rho* = gamma*/gamma*[norm_var]` matches the reference draw by draw, and
/// `rho*[norm_var]` is exactly `1` in every surviving draw.
///
/// Spec Step 12 asks for `rho*` alongside `gamma*[norm_var]`, `F*` and
/// `reliability*`, and it is the scale-free one: it is what `gamma*[norm_var]`
/// means for the estimand, and it is the quantity in which the unimplemented
/// Jentsch-Lunsford proxy rescaling provably cancels. Pinning it against the
/// transcription is what stops it drifting into a decoration.
#[test]
fn golden_rho_draws_match_draw_by_draw() {
    for f in [load(), load_sparse()] {
        let bands =
            proxy_svar_bands_from_starts(f.data.as_ref(), &f.proxy, &f.spec, &f.starts).unwrap();
        let expected = f.case["rho_draws"].as_array().unwrap();
        assert_eq!(bands.rho_draws.len(), expected.len());
        for (r, row) in expected.iter().enumerate() {
            let er = row.as_array().unwrap();
            assert_eq!(bands.rho_draws[r].len(), er.len(), "rho_draws[{r}] width");
            for (i, e) in er.iter().enumerate() {
                match e.as_f64() {
                    Some(v) => assert_rel_close(
                        bands.rho_draws[r][i],
                        v,
                        RTOL,
                        &format!("rho_draws[{r}][{i}]"),
                    ),
                    None => assert!(
                        bands.rho_draws[r][i].is_nan(),
                        "rho_draws[{r}][{i}] should be NaN on a failed draw"
                    ),
                }
            }
            if bands.rho_draws[r][f.spec.norm_var].is_finite() {
                assert_eq!(
                    bands.rho_draws[r][f.spec.norm_var], 1.0,
                    "draw {r}: rho*[norm_var] must be exactly 1"
                );
            }
        }
    }
}

/// The position-specific (Künsch/BJT) centering terms match the fixture.
///
/// This is the term that a wrong implementation silently no-ops: centering
/// by the grand mean of `uhat` changes nothing at all, because OLS with an
/// intercept already forces `sum_t uhat_t = 0`. Pinning `ubar_s` and
/// `mbar_s` is what makes that bug visible.
#[test]
fn golden_position_centering_terms_match() {
    let f = load();
    let fit = tsecon_var::VarSpec::new(f.spec.lags, f.spec.trend)
        .unwrap()
        .fit(f.data.as_ref())
        .unwrap();
    let aligned = &f.proxy[..];
    let ell = f.spec.block_length.unwrap();
    let c = position_centering(fit.resid.as_ref(), aligned, ell).unwrap();

    let e_u = f.case["u_bar"].as_array().unwrap();
    assert_eq!(c.u_bar.nrows(), e_u.len(), "u_bar rows");
    for (s, e_row) in e_u.iter().enumerate() {
        let row = e_row.as_array().unwrap();
        for (j, e) in row.iter().enumerate() {
            assert_rel_close(
                c.u_bar[(s, j)],
                e.as_f64().unwrap(),
                RTOL,
                &format!("u_bar[{s}][{j}]"),
            );
        }
    }
    let e_m = f.case["m_bar"].as_array().unwrap();
    for (s, e) in e_m.iter().enumerate() {
        assert_rel_close(
            c.m_bar[s],
            e.as_f64().unwrap(),
            RTOL,
            &format!("m_bar[{s}]"),
        );
    }
    let e_c = f.case["m_count"].as_array().unwrap();
    for (s, e) in e_c.iter().enumerate() {
        assert_eq!(c.m_count[s], e.as_u64().unwrap() as usize, "m_count[{s}]");
    }

    // The grand mean of the residuals is ~0 by OLS, while the position-wise
    // means are orders of magnitude larger — the reason the position-wise
    // form is the one with content.
    let t = fit.resid.nrows();
    let grand = (0..fit.resid.ncols())
        .map(|j| ((0..t).map(|i| fit.resid[(i, j)]).sum::<f64>() / t as f64).abs())
        .fold(0.0f64, f64::max);
    let posn = (0..c.u_bar.nrows())
        .flat_map(|s| (0..c.u_bar.ncols()).map(move |j| (s, j)))
        .map(|(s, j)| c.u_bar[(s, j)].abs())
        .fold(0.0f64, f64::max);
    assert!(grand < 1e-12, "grand mean {grand:e} should be ~0 by OLS");
    assert!(
        posn > 1e6 * grand.max(1e-15),
        "position-wise mean {posn:e} vs grand mean {grand:e}: the end effect this step \
         corrects must be far larger than the grand mean, or the fix is a no-op"
    );
}

/// Failure accounting matches the reference **reason by reason**, on a case
/// where draws genuinely fail.
///
/// The healthy-proxy case has `n_failed = 0` with every counter at zero, so
/// checking it alone would assert `0 == 0` six times and prove nothing. The
/// sparse case — the same data and the same reduced form, with the
/// instrument available on four dates only — puts 18 of 120 draws on the
/// failure path: 17 retain fewer than three finite proxy entries once the
/// availability pattern is itself resampled, and 1 retains entries with no
/// variance at all. Both counts come from an independent implementation of
/// the same classification, so this pins the `|O*| >= 3` boundary and the
/// zero-variance branch rather than merely observing that nothing happened.
///
/// The base case is still checked, because "no draw fails on a healthy
/// proxy" is itself a claim worth pinning.
#[test]
fn golden_failure_accounting_matches() {
    let mut saw_failures = false;
    for (f, what) in [(load(), "base"), (load_sparse(), "sparse")] {
        let bands =
            proxy_svar_bands_from_starts(f.data.as_ref(), &f.proxy, &f.spec, &f.starts).unwrap();
        let want_failed = f.case["n_failed"].as_u64().unwrap() as usize;
        assert_eq!(bands.n_failed, want_failed, "{what} n_failed");
        assert_eq!(
            bands.n_used,
            f.case["n_used"].as_u64().unwrap() as usize,
            "{what} n_used"
        );
        assert_eq!(bands.n_used, bands.n_boot - bands.n_failed, "{what} n_used");
        assert_eq!(bands.n_failed, bands.failures.total(), "{what} total");

        let e = &f.case["failures"];
        let got = &bands.failures;
        for (key, got) in [
            ("too_few_proxy_obs", got.too_few_proxy_obs),
            ("zero_proxy_variance", got.zero_proxy_variance),
            ("near_zero_gamma_norm", got.near_zero_gamma_norm),
            ("refit_failed", got.refit_failed),
            ("identification_failed", got.identification_failed),
            ("non_finite", got.non_finite),
        ] {
            assert_eq!(
                got,
                e[key].as_u64().unwrap() as usize,
                "{what} failures.{key}"
            );
        }
        saw_failures |= want_failed > 0;
    }
    assert!(
        saw_failures,
        "no case in the fixture produces a failed draw, so the counters were only ever \
         compared against zero and this test has no teeth"
    );

    // The two counters the fixture cannot reach are covered elsewhere and
    // this says where, so a reader is not left thinking they are pinned
    // here: `non_finite` in `proxy_bands_props.rs`, and
    // `identification_failed` nowhere yet — the NumPy transcription has no
    // Cholesky step, so it cannot produce a non-PD sigma_u*.
    let sparse = load_sparse();
    assert!(
        sparse.case["failures"]["too_few_proxy_obs"]
            .as_u64()
            .unwrap()
            > 0
            && sparse.case["failures"]["zero_proxy_variance"]
                .as_u64()
                .unwrap()
                > 0,
        "the sparse case is meant to exercise more than one failure reason"
    );
}

/// The `h = 0` band for `norm_var` is degenerate at `unit` — zero width in
/// both interval types, in the crate and in the reference.
///
/// This is the cheapest available proof that the unit-effect normalization
/// is re-imposed **inside** every draw: `b*[norm_var] = unit` exactly when
/// `rho* = gamma*/gamma*[norm_var]` is recomputed per draw, and only then.
/// A nonzero width here means the normalization was hoisted out of the loop
/// (`b* = (unit/gammahat[norm_var]) * gamma*`), which over-covers with a
/// strong proxy and under-covers with a weak one.
#[test]
fn golden_impact_cell_of_norm_var_is_degenerate() {
    let f = load();
    assert_eq!(f.case["degenerate_cell_width"].as_f64().unwrap(), 0.0);
    let bands =
        proxy_svar_bands_from_starts(f.data.as_ref(), &f.proxy, &f.spec, &f.starts).unwrap();
    let nv = f.spec.norm_var;
    assert_eq!(bands.point[0][nv], f.spec.unit);
    assert_eq!(bands.lower[0][nv], f.spec.unit, "Hall lower");
    assert_eq!(bands.upper[0][nv], f.spec.unit, "Hall upper");
    assert_eq!(bands.lower_efron[0][nv], f.spec.unit, "Efron lower");
    assert_eq!(bands.upper_efron[0][nv], f.spec.unit, "Efron upper");
    assert_eq!(bands.se[0][nv], 0.0, "bootstrap SD at the degenerate cell");
    // Every other impact cell is genuinely uncertain, so the degeneracy is
    // specific to the normalization and not a dead bootstrap.
    for i in 0..bands.point[0].len() {
        if i != nv {
            assert!(
                bands.upper[0][i] - bands.lower[0][i] > 0.0,
                "impact band for variable {i} has zero width"
            );
        }
    }
}

/// The generator's own measurement of the wild bootstrap's frozen
/// identifying moment is exactly zero deviation across 200 draws — recorded
/// in the fixture so the claim in the module docs has a number behind it.
#[test]
fn golden_records_the_frozen_wild_moment() {
    let f = load();
    let w = &f.fx["wild"];
    assert_eq!(
        w["moment_max_deviation"].as_f64().unwrap(),
        0.0,
        "a common Rademacher draw must leave sum_t m_t u_t' bit-identical"
    );
    assert!(w["n_draws"].as_u64().unwrap() >= 200);
}
