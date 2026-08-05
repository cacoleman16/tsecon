//! Golden-value tests against `fixtures/hac.json` (generated with
//! statsmodels 0.14.6; see `fixtures/generate_fixtures.py::gen_hac`) and
//! `fixtures/hc_robust.json` (see
//! `fixtures/generate_hc_robust_fixtures.py`).
//!
//! The regression block pins OLS params, nonrobust bse, and HAC bse /
//! tvalues over maxlags {4, 8, 12} x use_correction {true, false}; the
//! `lrv_nile_demeaned` block pins Bartlett and EWC long-run variances on
//! the demeaned Nile (loaded from `fixtures/diagnostics.json`). Spec
//! tolerance is 1e-10 relative; everything is asserted at that bound.
//!
//! `hc_robust.json` pins the whole HC ladder — nonrobust/HC0/HC1/HC2/HC3
//! bse, tvalues and full covariance matrices — against statsmodels
//! `fit(cov_type=...)` on two designs: a `T = 25` high-leverage one (the
//! DGP from the interval-coverage audit that motivated HC2/HC3) and a
//! calmer `k = 3` one. It also pins the hat-matrix diagonal that the
//! leverage weights are built from. Its third block, `near_singleton`,
//! deliberately breaks the 1e-10 tolerance: see
//! [`hc2_hc3_parity_degrades_as_the_leverage_complement_approaches_noise`].

use serde_json::Value;
use tsecon_hac::{ewc_lrv, lrv, newey_west_maxlags, ols, HacError, Kernel, SeType};

fn load(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
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

/// Relative comparison; falls back to absolute when the reference is 0.
fn assert_close(actual: f64, expected: f64, rtol: f64, ctx: &str) {
    if expected == 0.0 {
        assert!(
            actual.abs() <= rtol,
            "{ctx}: actual {actual}, expected 0 (atol {rtol})"
        );
    } else {
        let rel = ((actual - expected) / expected).abs();
        assert!(
            rel <= rtol,
            "{ctx}: actual {actual}, expected {expected}, rel err {rel:e} > {rtol:e}"
        );
    }
}

fn assert_all_close(actual: &[f64], expected: &[f64], rtol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, rtol, &format!("{ctx}[{i}]"));
    }
}

const TOL: f64 = 1e-10;

/// Design [const, x1, x2] exactly as the fixture assembles it.
fn regression_design(fx: &Value) -> (Vec<f64>, Vec<Vec<f64>>) {
    let reg = &fx["regression"];
    let y = f64s(&reg["y"]);
    let x1 = f64s(&reg["x1"]);
    let x2 = f64s(&reg["x2"]);
    let constant = vec![1.0; y.len()];
    (y, vec![constant, x1, x2])
}

fn demeaned_nile() -> Vec<f64> {
    let y = f64s(&load("diagnostics.json")["nile"]);
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    y.iter().map(|v| v - mean).collect()
}

#[test]
fn ols_params_and_nonrobust_bse_match_statsmodels() {
    let fx = load("hac.json");
    let (y, x) = regression_design(&fx);
    let fit = ols(&y, &x).unwrap();

    assert_all_close(
        &fit.params,
        &f64s(&fx["regression"]["ols_params"]),
        TOL,
        "ols params",
    );
    let inf = fit.inference(SeType::NonRobust).unwrap();
    assert_all_close(
        &inf.bse,
        &f64s(&fx["regression"]["ols_bse_nonrobust"]),
        TOL,
        "nonrobust bse",
    );
}

#[test]
fn hac_bse_and_tvalues_match_statsmodels_all_cases() {
    let fx = load("hac.json");
    let (y, x) = regression_design(&fx);
    let fit = ols(&y, &x).unwrap();

    let cases = fx["regression"]["hac_cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 6, "expected 3 maxlags x 2 corrections");
    for case in cases {
        let maxlags = case["maxlags"].as_u64().expect("maxlags") as f64;
        let use_correction = case["use_correction"].as_bool().expect("flag");
        let ctx = format!("HAC maxlags={maxlags} correction={use_correction}");

        let inf = fit
            .inference(SeType::Hac {
                kernel: Kernel::Bartlett,
                bandwidth: maxlags,
                use_correction,
            })
            .unwrap();
        assert_all_close(&inf.bse, &f64s(&case["bse"]), TOL, &format!("{ctx} bse"));
        assert_all_close(
            &inf.tvalues,
            &f64s(&case["tvalues"]),
            TOL,
            &format!("{ctx} tvalues"),
        );
    }
}

/// Rebuild one `hc_robust.json` block's design: `[const, x1(, x2)]`.
fn hc_design(block: &Value) -> (Vec<f64>, Vec<Vec<f64>>) {
    let y = f64s(&block["y"]);
    let mut cols = vec![vec![1.0; y.len()], f64s(&block["x1"])];
    if let Some(x2) = block.get("x2") {
        cols.push(f64s(x2));
    }
    (y, cols)
}

/// The whole ladder — bse, tvalues and the full covariance matrix — against
/// statsmodels `fit(cov_type=...)` on both fixture designs.
#[test]
fn hc_ladder_bse_tvalues_and_cov_match_statsmodels() {
    let fx = load("hc_robust.json");
    for name in ["small_high_leverage", "multi_regressor"] {
        let block = &fx[name];
        let (y, x) = hc_design(block);
        let fit = ols(&y, &x).unwrap();
        assert_all_close(
            &fit.params,
            &f64s(&block["params"]),
            TOL,
            &format!("{name} params"),
        );

        for (key, se_type) in [
            ("nonrobust", SeType::NonRobust),
            ("hc0", SeType::Hc0),
            ("hc1", SeType::Hc1),
            ("hc2", SeType::Hc2),
            ("hc3", SeType::Hc3),
        ] {
            let inf = fit.inference(se_type).unwrap();
            let ctx = format!("{name} {key}");
            assert_all_close(
                &inf.bse,
                &f64s(&block["bse"][key]),
                TOL,
                &format!("{ctx} bse"),
            );
            assert_all_close(
                &inf.tvalues,
                &f64s(&block["tvalues"][key]),
                TOL,
                &format!("{ctx} tvalues"),
            );
            assert_all_close(
                &inf.cov,
                &f64s(&block["cov"][key]),
                TOL,
                &format!("{ctx} cov"),
            );
        }
    }
}

/// The crate computes its own leverage internally, so this rebuilds the
/// HC2/HC3 covariance from *statsmodels'* hat diagonal with independent
/// linear algebra (Gauss-Jordan inverse, not the crate's Cholesky) and
/// requires the two to agree. `trace(H) = k` is checked alongside as a
/// sanity condition on the fixture itself.
#[test]
fn hc2_hc3_cov_rebuilt_from_the_statsmodels_hat_diagonal_agrees() {
    let fx = load("hc_robust.json");
    for name in ["small_high_leverage", "multi_regressor"] {
        let block = &fx[name];
        let h = f64s(&block["hat_diag"]);
        let (y, x) = hc_design(block);
        let k = x.len();
        assert_close(
            h.iter().sum(),
            k as f64,
            1e-12,
            &format!("{name} trace(H) = k"),
        );

        let fit = ols(&y, &x).unwrap();
        for (power, se_type) in [(1_i32, SeType::Hc2), (2, SeType::Hc3)] {
            let got = fit.inference(se_type).unwrap();
            let want = weighted_sandwich(&fit.residuals, &x, &h, power);
            assert_all_close(
                &got.cov,
                &want,
                1e-9,
                &format!("{name} HC{} cov rebuilt from fixture h", power + 1),
            );
        }
    }
}

/// `(X'X)^{-1} [sum_t u_t^2 (1-h_t)^{-power} x_t x_t'] (X'X)^{-1}`, built
/// from scratch here (dense Gauss-Jordan inverse) so the check does not go
/// through the crate's own sandwich code path.
fn weighted_sandwich(resid: &[f64], x: &[Vec<f64>], h: &[f64], power: i32) -> Vec<f64> {
    let n = resid.len();
    let k = x.len();
    let mut xtx = vec![0.0_f64; k * k];
    let mut meat = vec![0.0_f64; k * k];
    for t in 0..n {
        let w = resid[t] * resid[t] * (1.0 - h[t]).powi(-power);
        for i in 0..k {
            for j in 0..k {
                xtx[i * k + j] += x[i][t] * x[j][t];
                meat[i * k + j] += w * x[i][t] * x[j][t];
            }
        }
    }
    let bread = invert(&xtx, k);
    let mut tmp = vec![0.0_f64; k * k];
    let mut out = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            tmp[i * k + j] = (0..k).map(|l| meat[i * k + l] * bread[l * k + j]).sum();
        }
    }
    for i in 0..k {
        for j in 0..k {
            out[i * k + j] = (0..k).map(|l| bread[i * k + l] * tmp[l * k + j]).sum();
        }
    }
    out
}

/// Gauss-Jordan inverse with partial pivoting (test-only helper).
fn invert(a: &[f64], k: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0_f64; k * k];
    for i in 0..k {
        inv[i * k + i] = 1.0;
    }
    for col in 0..k {
        let pivot = (col..k)
            .max_by(|&r1, &r2| {
                m[r1 * k + col]
                    .abs()
                    .partial_cmp(&m[r2 * k + col].abs())
                    .expect("finite")
            })
            .expect("nonempty");
        for j in 0..k {
            m.swap(col * k + j, pivot * k + j);
            inv.swap(col * k + j, pivot * k + j);
        }
        let d = m[col * k + col];
        for j in 0..k {
            m[col * k + j] /= d;
            inv[col * k + j] /= d;
        }
        for row in 0..k {
            if row == col {
                continue;
            }
            let f = m[row * k + col];
            if f == 0.0 {
                continue;
            }
            for j in 0..k {
                m[row * k + j] -= f * m[col * k + j];
                inv[row * k + j] -= f * inv[col * k + j];
            }
        }
    }
    inv
}

/// Where statsmodels parity for HC2/HC3 actually stops, measured rather
/// than asserted away.
///
/// `near_singleton` is a dummy with `d[7] = 1`, `d[8] = eps`: shrinking
/// `eps` drives `h_7` toward 1 (the fixture walks `1 - h_7` from 9.6e-5
/// down to 0) while leaving `cond(X'X)` at ~47 — this is *cancellation in
/// the subtraction*, not ill-conditioning, so no conditioning-based guard
/// would see it. `h_7` carries an absolute error of a few ulp of 1, so
/// `1 - h_7` carries a relative error of order `eps_mach / (1 - h_7)`, and
/// the HC3 weight `(1-h_t)^-2` doubles it. Two implementations that each
/// compute the hat diagonal correctly therefore stop agreeing to the
/// crate's 1e-10 spec tolerance long before either one fails.
///
/// So this test asserts four things and no more: HC0, which never touches
/// leverage, keeps full 1e-10 accuracy on the same fits (isolating the
/// cause); above the crate's `LEVERAGE_FLOOR` HC2/HC3 still return, and
/// disagree only within the `eps_mach / (1 - h_t)` law rather than
/// arbitrarily; at the bottom of that band the disagreement is genuinely,
/// measurably worse than 1e-10 — which is why [`SeType::Hc2`]/
/// [`SeType::Hc3`] qualify their parity claim; and below the floor the crate
/// refuses by name in a cell where statsmodels still hands back a number.
#[test]
fn hc2_hc3_parity_degrades_as_the_leverage_complement_approaches_noise() {
    let fx = load("hc_robust.json");
    let block = &fx["near_singleton"];
    let y = f64s(&block["y"]);
    let x1 = f64s(&block["x1"]);

    // Not an ill-conditioned design — that is the whole point.
    let cond = block["cases"][0]["cond_xtx"].as_f64().expect("cond");
    assert!(
        cond < 1e3,
        "near_singleton cond(X'X) = {cond} should be tame"
    );

    // `LEVERAGE_FLOOR` is private, so the two sides of it are named here with
    // a decade of slack: the fixture must not park a case in the gap, where
    // statsmodels' `1 - h_7` and the crate's could land on opposite sides.
    const ABOVE_FLOOR: f64 = 1e-11;
    const BELOW_FLOOR: f64 = 1e-12;

    let mut refused_but_statsmodels_answered = 0_usize;
    let mut worst_in_band: f64 = 0.0;
    for case in block["cases"].as_array().expect("cases") {
        let eps = case["eps"].as_f64().expect("eps");
        let rest = case["one_minus_h7"].as_f64().expect("one_minus_h7");
        assert!(
            !(BELOW_FLOOR..ABOVE_FLOOR).contains(&rest),
            "eps={eps:e}: 1-h7={rest:e} straddles LEVERAGE_FLOOR; this test \
             cannot say whether the crate should answer or refuse"
        );
        let cols = vec![vec![1.0; y.len()], x1.clone(), f64s(&case["dummy"])];
        let fit = ols(&y, &cols).expect("design is solvable at every eps");
        let sm_finite = !case["bse"]["hc3"].is_null();

        if rest < BELOW_FLOOR {
            // The guard fires. Worth pinning that it fires even where
            // statsmodels still returns a (meaningless) finite number.
            if sm_finite {
                refused_but_statsmodels_answered += 1;
            }
            for se_type in [SeType::Hc2, SeType::Hc3] {
                let err = fit.inference(se_type).expect_err("must refuse");
                assert!(
                    matches!(err, HacError::FullLeverage { index: 7, .. }),
                    "eps={eps:e}: expected FullLeverage at obs 7, got {err:?}"
                );
            }
            // HC0/HC1 need no leverage, so they must survive the same fit.
            for se_type in [SeType::Hc0, SeType::Hc1] {
                let inf = fit.inference(se_type).expect("no leverage needed");
                assert!(inf.bse.iter().all(|s| s.is_finite()));
            }
            continue;
        }

        assert!(
            sm_finite,
            "eps={eps:e}: fixture should have a reference here"
        );

        // HC0 never touches leverage, so it must stay at full spec accuracy
        // no matter how close to 1 h_7 gets — this isolates the cause.
        let hc0 = fit.inference(SeType::Hc0).expect("HC0 needs no leverage");
        assert_all_close(
            &hc0.bse,
            &f64s(&case["bse"]["hc0"]),
            TOL,
            &format!("near_singleton eps={eps:e} hc0 bse"),
        );

        // The achievable relative agreement: a few ulp of h, divided by the
        // complement, doubled for HC3's squared weight. 100x headroom over
        // the measured worst case keeps this from being a coin flip.
        let budget = (100.0 * f64::EPSILON / rest).max(TOL);
        for (key, se_type, power) in [("hc2", SeType::Hc2, 1.0), ("hc3", SeType::Hc3, 2.0)] {
            let want = f64s(&case["bse"][key]);
            let got = fit
                .inference(se_type)
                .unwrap_or_else(|e| panic!("eps={eps:e}: {key} must still answer, got {e:?}"));
            assert!(got.bse.iter().all(|s| s.is_finite() && *s > 0.0));
            let rel = got
                .bse
                .iter()
                .zip(want.iter())
                .map(|(a, e)| ((a - e) / e).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                rel <= power * budget,
                "eps={eps:e} 1-h7={rest:e}: {key} rel err {rel:e} exceeds the \
                 eps/(1-h) budget {:e} — the cancellation law is wrong, or \
                 the leverage accumulation regressed",
                power * budget
            );
            if key == "hc3" {
                worst_in_band = worst_in_band.max(rel);
            }
        }
    }

    assert_eq!(
        refused_but_statsmodels_answered, 1,
        "the fixture should carry exactly one cell below the floor where \
         statsmodels still returns a finite (noise) standard error"
    );
    // The load-bearing half: the degradation is real, not a theoretical
    // worry. If this ever fails because agreement IMPROVED, the parity
    // qualification in the SeType docs can be tightened to match.
    assert!(
        worst_in_band > 1e-9,
        "HC3 agreed to {worst_in_band:e} across the whole band — better than \
         the docs admit; re-measure and tighten the SeType parity wording"
    );
}

#[test]
fn bartlett_lrv_on_demeaned_nile_matches_fixture() {
    let fx = load("hac.json");
    let z = demeaned_nile();
    for bw in [5_usize, 10, 20] {
        let expected = fx["lrv_nile_demeaned"]["bartlett"][bw.to_string()]
            .as_f64()
            .expect("bartlett value");
        let actual = lrv(&z, Kernel::Bartlett, bw as f64).unwrap();
        assert_close(actual, expected, TOL, &format!("bartlett lrv bw={bw}"));
    }
}

#[test]
fn ewc_lrv_on_demeaned_nile_matches_fixture() {
    let fx = load("hac.json");
    let z = demeaned_nile();
    for b in [4_usize, 8, 16] {
        let expected = fx["lrv_nile_demeaned"]["ewc"][b.to_string()]
            .as_f64()
            .expect("ewc value");
        let actual = ewc_lrv(&z, b).unwrap();
        assert_close(actual, expected, TOL, &format!("ewc lrv B={b}"));
    }
}

#[test]
fn newey_west_maxlags_rule_matches_fixture_integer() {
    let fx = load("hac.json");
    let z = demeaned_nile();
    let expected = fx["lrv_nile_demeaned"]["newey_west_auto_maxlags_floor_4_n100_2_9"]
        .as_u64()
        .expect("integer") as usize;
    assert_eq!(newey_west_maxlags(z.len()), expected);
}
