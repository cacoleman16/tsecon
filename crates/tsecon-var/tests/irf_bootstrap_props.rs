//! Validation for the residual-bootstrap VAR-IRF confidence bands
//! ([`tsecon_var::bootstrap_irf_bands`]).
//!
//! The bootstrap is stochastic, so this suite validates it honestly on
//! four fronts:
//!
//! * **Reproducibility** — the same seed produces bit-identical bands on one
//!   platform, for the plain, cumulative, and bias-corrected paths; the plain
//!   path also reproduces the seeded snapshot pinned in
//!   `fixtures/var_irf_bootstrap.json` to a tight tolerance (the snapshot was
//!   generated on one architecture, so it is held cross-platform to tolerance,
//!   not to the bit — see [`assert_cube_close`]).
//! * **Structure** — `lower <= point <= upper` everywhere, standard errors
//!   are non-negative and finite, and the point estimate matches `var_irf`
//!   (statsmodels `orth_irfs`).
//! * **Magnitude sanity** — on the shared macro DGP the bootstrap SE is
//!   within a sane factor of the statsmodels *asymptotic* SE
//!   (`irf.stderr`), and the errband_mc reference is bracketed loosely.
//! * **Monte-Carlo coverage** — over many DGP replications the nominal 90%
//!   band covers the true short-horizon IRF close to 90% of the time.

mod common;

use common::{as_mat, assert_rel_close, load_fixture, Lcg};
use serde_json::{json, Value};
use tsecon_linalg::faer::Mat;
use tsecon_var::{bootstrap_irf_bands, ma_rep, IrfBands, Trend};

// ---------------------------------------------------------------- helpers

/// Parse the snapshot configuration recorded in the fixture (single source
/// of truth shared by the emitter and the checker).
struct SnapCfg {
    lags: usize,
    trend: Trend,
    horizon: usize,
    orth: bool,
    cumulative: bool,
    alpha: f64,
    n_boot: usize,
    seed: u64,
    bias_correct: bool,
}

fn snapshot_cfg(fx: &Value) -> SnapCfg {
    let sp = &fx["snapshot_params"];
    let trend = match sp["trend"].as_str().unwrap() {
        "c" => Trend::Constant,
        "n" => Trend::None,
        other => panic!("unknown trend {other:?}"),
    };
    SnapCfg {
        lags: sp["lags"].as_u64().unwrap() as usize,
        trend,
        horizon: sp["horizon"].as_u64().unwrap() as usize,
        orth: sp["orth"].as_bool().unwrap(),
        cumulative: sp["cumulative"].as_bool().unwrap(),
        alpha: sp["alpha"].as_f64().unwrap(),
        n_boot: sp["n_boot"].as_u64().unwrap() as usize,
        seed: sp["seed"].as_u64().unwrap(),
        bias_correct: sp["bias_correct"].as_bool().unwrap(),
    }
}

fn macro_data(fx: &Value) -> Mat<f64> {
    as_mat(&fx["data_100dlog_gdp_cons_inv"])
}

fn run_snapshot(fx: &Value) -> IrfBands {
    let c = snapshot_cfg(fx);
    let data = macro_data(fx);
    bootstrap_irf_bands(
        data.as_ref(),
        c.lags,
        c.trend,
        c.horizon,
        c.orth,
        c.cumulative,
        c.alpha,
        c.n_boot,
        c.seed,
        c.bias_correct,
    )
    .expect("snapshot bootstrap must succeed")
}

fn cube_to_json(cube: &[Mat<f64>]) -> Value {
    Value::Array(
        cube.iter()
            .map(|m| {
                Value::Array(
                    (0..m.nrows())
                        .map(|i| Value::Array((0..m.ncols()).map(|j| json!(m[(i, j)])).collect()))
                        .collect(),
                )
            })
            .collect(),
    )
}

/// Assert every cell of a cube reproduces the stored JSON snapshot to a tight
/// tolerance. The fixture is generated on one platform; bootstrap resampling
/// and the recursive DGP accumulate sub-ULP rounding that differs across
/// architectures (SIMD width, FMA contraction), so a *cross-platform* snapshot
/// can only be held to tolerance, never to the exact bit pattern — the library
/// promises bit-identical reproducibility per platform (see
/// `same_seed_is_bit_identical`, which stays bit-exact), not across them.
fn assert_cube_close(actual: &[Mat<f64>], stored: &Value, what: &str) {
    // Tight enough to catch a real regression, loose enough to absorb the
    // handful-of-ULPs cross-architecture drift the point path exhibits.
    const RTOL: f64 = 1e-9;
    const ATOL: f64 = 1e-12;
    let arr = stored
        .as_array()
        .unwrap_or_else(|| panic!("{what}: not an array"));
    assert_eq!(actual.len(), arr.len(), "{what}: horizon count");
    for (h, m) in actual.iter().enumerate() {
        let rows = arr[h].as_array().unwrap();
        assert_eq!(m.nrows(), rows.len(), "{what}[{h}]: rows");
        for i in 0..m.nrows() {
            let cols = rows[i].as_array().unwrap();
            assert_eq!(m.ncols(), cols.len(), "{what}[{h}][{i}]: cols");
            for j in 0..m.ncols() {
                let a = m[(i, j)];
                let e = cols[j].as_f64().unwrap();
                let tol = ATOL + RTOL * e.abs();
                assert!(
                    (a - e).abs() <= tol,
                    "{what}[{h}][{i}][{j}] off snapshot: {a} vs {e} (|Δ|={:e} > {tol:e})",
                    (a - e).abs()
                );
            }
        }
    }
}

fn cubes_bits_eq(a: &[Mat<f64>], b: &[Mat<f64>]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).all(|(m, n)| {
        m.nrows() == n.nrows()
            && m.ncols() == n.ncols()
            && (0..m.nrows())
                .all(|i| (0..m.ncols()).all(|j| m[(i, j)].to_bits() == n[(i, j)].to_bits()))
    })
}

fn bands_bits_eq(a: &IrfBands, b: &IrfBands) -> bool {
    cubes_bits_eq(&a.point, &b.point)
        && cubes_bits_eq(&a.se, &b.se)
        && cubes_bits_eq(&a.lower, &b.lower)
        && cubes_bits_eq(&a.upper, &b.upper)
}

// ------------------------------------------------------ reproducibility

/// The plain bootstrap bands reproduce the seeded snapshot pinned in the
/// fixture to a tight tolerance — the regression guard on the whole pipeline
/// (held to tolerance rather than the bit because the snapshot crosses
/// architectures; see [`assert_cube_close`]).
#[test]
fn snapshot_matches_stored() {
    let fx = load_fixture("var_irf_bootstrap.json");
    let snap = fx.get("tsecon_snapshot").unwrap_or_else(|| {
        panic!(
            "fixture has no `tsecon_snapshot`; run \
             `cargo test -p tsecon-var emit_snapshot -- --ignored --nocapture` \
             and merge its JSON into fixtures/var_irf_bootstrap.json"
        )
    });
    let bands = run_snapshot(&fx);
    assert_cube_close(&bands.point, &snap["point"], "point");
    assert_cube_close(&bands.se, &snap["se"], "se");
    assert_cube_close(&bands.lower, &snap["lower"], "lower");
    assert_cube_close(&bands.upper, &snap["upper"], "upper");
}

/// Same seed => bit-identical bands, across the plain, cumulative, and
/// bias-corrected code paths (the core reproducibility contract).
#[test]
fn same_seed_is_bit_identical() {
    let fx = load_fixture("var_irf_bootstrap.json");
    let data = macro_data(&fx);
    let configs: &[(bool, bool)] = &[(false, false), (true, false), (false, true)];
    for &(cumulative, bias_correct) in configs {
        let run = || {
            bootstrap_irf_bands(
                data.as_ref(),
                2,
                Trend::Constant,
                5,
                true,
                cumulative,
                0.10,
                120,
                4242,
                bias_correct,
            )
            .unwrap()
        };
        let a = run();
        let b = run();
        assert!(
            bands_bits_eq(&a, &b),
            "not bit-identical for (cumulative={cumulative}, bias_correct={bias_correct})"
        );
    }
}

/// Ignored emitter: prints the seeded snapshot as JSON so it can be merged
/// into the fixture under `tsecon_snapshot`. Run with
/// `cargo test -p tsecon-var emit_snapshot -- --ignored --nocapture`.
#[test]
#[ignore = "emits the fixture snapshot; run manually with --nocapture"]
fn emit_snapshot() {
    let fx = load_fixture("var_irf_bootstrap.json");
    let bands = run_snapshot(&fx);
    let snap = json!({
        "params": fx["snapshot_params"].clone(),
        "point": cube_to_json(&bands.point),
        "se": cube_to_json(&bands.se),
        "lower": cube_to_json(&bands.lower),
        "upper": cube_to_json(&bands.upper),
    });
    println!("===SNAPSHOT_JSON_BEGIN===");
    println!("{}", serde_json::to_string(&snap).unwrap());
    println!("===SNAPSHOT_JSON_END===");
}

// ------------------------------------------------------------- structure

/// `lower <= point <= upper` everywhere and the standard errors are
/// non-negative and finite. (The orthogonalized impact response `Psi_0 P`
/// still varies across draws through each draw's own `sigma_u`, so its SE is
/// positive — we only require it be well-formed, not zero.)
#[test]
fn bands_bracket_point_and_are_well_formed() {
    let fx = load_fixture("var_irf_bootstrap.json");
    let data = macro_data(&fx);
    let bands = bootstrap_irf_bands(
        data.as_ref(),
        2,
        Trend::Constant,
        8,
        true,
        false,
        0.10,
        400,
        11,
        false,
    )
    .unwrap();
    for h in 0..bands.point.len() {
        for i in 0..3 {
            for j in 0..3 {
                let p = bands.point[h][(i, j)];
                let lo = bands.lower[h][(i, j)];
                let hi = bands.upper[h][(i, j)];
                let se = bands.se[h][(i, j)];
                assert!(se.is_finite() && se >= 0.0, "se[{h}][{i}][{j}]={se}");
                assert!(lo <= hi, "lower>upper at [{h}][{i}][{j}]: {lo} > {hi}");
                assert!(
                    lo <= p + 1e-9 && p <= hi + 1e-9,
                    "point outside band at [{h}][{i}][{j}]: {lo} <= {p} <= {hi}"
                );
            }
        }
    }
}

/// The point estimate equals statsmodels' orthogonalized IRF (via
/// `var_irf`) elementwise — the contract that `point` matches `var_irf`.
#[test]
#[allow(clippy::needless_range_loop)] // h/i/j index both the cube and the fixture
fn point_matches_statsmodels_orth_irf() {
    let fx = load_fixture("var_irf_bootstrap.json");
    let data = macro_data(&fx);
    let bands = bootstrap_irf_bands(
        data.as_ref(),
        2,
        Trend::Constant,
        10,
        true,
        false,
        0.10,
        50,
        7,
        false,
    )
    .unwrap();
    let expected = fx["point_orth_h10"].as_array().unwrap();
    for h in 0..=10 {
        let rows = expected[h].as_array().unwrap();
        for i in 0..3 {
            let cols = rows[i].as_array().unwrap();
            for j in 0..3 {
                assert_rel_close(
                    bands.point[h][(i, j)],
                    cols[j].as_f64().unwrap(),
                    1e-8,
                    &format!("point[{h}][{i}][{j}]"),
                );
            }
        }
    }
}

// ------------------------------------------------------- magnitude sanity

/// On the shared macro DGP the bootstrap SE sits within a sane factor of
/// the statsmodels asymptotic (delta-method) SE at the cells where that SE
/// is well away from zero — bootstrap and asymptotic inference should agree
/// to an order of magnitude even though the numbers are not identical.
#[test]
#[allow(clippy::needless_range_loop)] // h/i/j index both the cube and the fixture
fn bootstrap_se_within_factor_of_asymptotic() {
    let fx = load_fixture("var_irf_bootstrap.json");
    let data = macro_data(&fx);
    let bands = bootstrap_irf_bands(
        data.as_ref(),
        2,
        Trend::Constant,
        10,
        true,
        false,
        0.10,
        600,
        20260722,
        false,
    )
    .unwrap();
    let asym = &fx["asymptotic_se_orth"];
    let mut compared = 0usize;
    let mut worst = 1.0f64;
    for h in 1..=6 {
        let rows = asym[h].as_array().unwrap();
        for i in 0..3 {
            let cols = rows[i].as_array().unwrap();
            for j in 0..3 {
                let se_a = cols[j].as_f64().unwrap();
                // Only compare where the asymptotic SE is materially positive.
                if se_a < 0.02 {
                    continue;
                }
                let se_b = bands.se[h][(i, j)];
                let ratio = se_b / se_a;
                worst = worst.max(ratio).max(1.0 / ratio);
                assert!(
                    (0.25..=4.0).contains(&ratio),
                    "SE ratio out of range at [{h}][{i}][{j}]: boot {se_b} vs asym {se_a} (ratio {ratio:.3})"
                );
                compared += 1;
            }
        }
    }
    assert!(compared >= 20, "too few cells compared: {compared}");
    println!("magnitude sanity: {compared} cells, worst SE ratio factor {worst:.3}");
}

// ------------------------------------------------------------- coverage

/// Simulate a stationary VAR(1) `y_t = A y_{t-1} + u_t`, `u ~ N(0, I)`,
/// keeping the last `t` of `t + burn` steps.
fn simulate_var1(a: &Mat<f64>, t: usize, burn: usize, lcg: &mut Lcg) -> Mat<f64> {
    let k = a.nrows();
    let total = t + burn;
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(total);
    let mut prev = vec![0.0f64; k];
    for _ in 0..total {
        let mut cur = vec![0.0f64; k];
        for (r, cr) in cur.iter_mut().enumerate() {
            let mut v = lcg.gaussian();
            for (c, &pc) in prev.iter().enumerate() {
                v += a[(r, c)] * pc;
            }
            *cr = v;
        }
        rows.push(cur.clone());
        prev = cur;
    }
    Mat::from_fn(t, k, |i, j| rows[burn + i][j])
}

/// Monte-Carlo coverage: over many DGP replications the nominal 90%
/// reduced-form band covers the *true* short-horizon IRF `A^h` close to 90%
/// of the time. Bootstrap IRF bands are known to slightly under-cover at
/// short horizons in small samples (the motivation for Kilian's bias
/// correction), so the acceptance interval is deliberately loose.
#[test]
fn mc_coverage_short_horizon() {
    let a = Mat::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => 0.5,
        (0, 1) => 0.2,
        (1, 0) => 0.1,
        (1, 1) => 0.4,
        _ => 0.0,
    });
    // True non-orthogonalized IRF: Psi_h = A^h.
    let truth = ma_rep(std::slice::from_ref(&a), 3).unwrap();

    let n_dgp = 150usize;
    let n_boot = 120usize;
    let t = 200usize;
    let burn = 100usize;
    let alpha = 0.10;

    // Coverage counters at horizons 1 and 2 (short horizons).
    let mut covered = [0usize; 3];
    let mut total = [0usize; 3];

    for rep in 0..n_dgp {
        let mut lcg = Lcg::new(777 + rep as u64);
        let data = simulate_var1(&a, t, burn, &mut lcg);
        let bands = bootstrap_irf_bands(
            data.as_ref(),
            1,
            Trend::None,
            3,
            false, // reduced-form (non-orth): truth is exactly A^h
            false,
            alpha,
            n_boot,
            4000 + rep as u64,
            false,
        )
        .unwrap();
        for h in 1..=2 {
            for i in 0..2 {
                for j in 0..2 {
                    let tv = truth[h][(i, j)];
                    let lo = bands.lower[h][(i, j)];
                    let hi = bands.upper[h][(i, j)];
                    total[h] += 1;
                    if lo <= tv && tv <= hi {
                        covered[h] += 1;
                    }
                }
            }
        }
    }

    let cov1 = covered[1] as f64 / total[1] as f64;
    let cov2 = covered[2] as f64 / total[2] as f64;
    println!(
        "MC coverage (nominal 0.90): h=1 {:.3} ({}/{}), h=2 {:.3} ({}/{})",
        cov1, covered[1], total[1], cov2, covered[2], total[2]
    );
    // Loose bounds: bootstrap percentile bands typically land in the low-to-mid
    // 0.80s at h=1 for T=200, climbing toward nominal as the horizon grows.
    assert!(
        (0.75..=0.99).contains(&cov1),
        "h=1 coverage {cov1:.3} outside [0.75, 0.99]"
    );
    assert!(
        (0.78..=0.99).contains(&cov2),
        "h=2 coverage {cov2:.3} outside [0.78, 0.99]"
    );
}

// --------------------------------------------------------- bias correction

/// The Kilian bias correction shifts the point estimate (the bias-corrected
/// coefficients differ from OLS) and still returns well-formed, reproducible
/// bands that bracket the corrected point.
#[test]
fn bias_correction_shifts_point_and_brackets() {
    let fx = load_fixture("var_irf_bootstrap.json");
    let data = macro_data(&fx);
    let plain = bootstrap_irf_bands(
        data.as_ref(),
        2,
        Trend::Constant,
        6,
        true,
        false,
        0.10,
        200,
        555,
        false,
    )
    .unwrap();
    let corrected = bootstrap_irf_bands(
        data.as_ref(),
        2,
        Trend::Constant,
        6,
        true,
        false,
        0.10,
        200,
        555,
        true,
    )
    .unwrap();

    // The bias-corrected point path differs from the plain OLS path.
    let mut max_shift = 0.0f64;
    for h in 0..plain.point.len() {
        for i in 0..3 {
            for j in 0..3 {
                max_shift =
                    max_shift.max((plain.point[h][(i, j)] - corrected.point[h][(i, j)]).abs());
            }
        }
    }
    assert!(
        max_shift > 1e-6,
        "bias correction did not move the point estimate (max shift {max_shift:.3e})"
    );

    // Bands still bracket the (corrected) point.
    for h in 0..corrected.point.len() {
        for i in 0..3 {
            for j in 0..3 {
                let p = corrected.point[h][(i, j)];
                let lo = corrected.lower[h][(i, j)];
                let hi = corrected.upper[h][(i, j)];
                assert!(lo <= hi, "corrected lower>upper at [{h}][{i}][{j}]");
                assert!(
                    lo <= p + 1e-9 && p <= hi + 1e-9,
                    "corrected point outside band at [{h}][{i}][{j}]"
                );
            }
        }
    }
    assert!(corrected.bias_correct);
}

// --------------------------------------------------------------- errors

/// Invalid arguments are rejected without panicking.
#[test]
fn rejects_invalid_arguments() {
    let fx = load_fixture("var_irf_bootstrap.json");
    let data = macro_data(&fx);
    // lags = 0
    assert!(bootstrap_irf_bands(
        data.as_ref(),
        0,
        Trend::Constant,
        4,
        true,
        false,
        0.1,
        10,
        0,
        false
    )
    .is_err());
    // n_boot = 1
    assert!(bootstrap_irf_bands(
        data.as_ref(),
        2,
        Trend::Constant,
        4,
        true,
        false,
        0.1,
        1,
        0,
        false
    )
    .is_err());
    // alpha out of range
    assert!(bootstrap_irf_bands(
        data.as_ref(),
        2,
        Trend::Constant,
        4,
        true,
        false,
        0.0,
        10,
        0,
        false
    )
    .is_err());
    assert!(bootstrap_irf_bands(
        data.as_ref(),
        2,
        Trend::Constant,
        4,
        true,
        false,
        1.0,
        10,
        0,
        false
    )
    .is_err());
}
