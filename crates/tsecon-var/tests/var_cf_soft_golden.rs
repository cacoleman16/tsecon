//! Golden tests for SOFT (Gaussian, stated-variance) conditions,
//! `VarResults::conditional_forecast_soft`, against
//! `fixtures/var_cf_soft.json`: the closed form transcribed in NumPy in both
//! readings (signal: Monti 2010; density: Andersson-Palmqvist-Waggoner
//! 2010) — every returned quantity; statsmodels `VARMAX` (signal: a
//! time-varying measurement-noise covariance; density: the hard smoother
//! plus an exact perturbation gain) — path, covariance and shocks, and its
//! future log-likelihood as a check on the Mahalanobis statistic; and the
//! APW Moore-Penrose structural form (density).
//!
//! Also here: the hard path is unchanged by the soft machinery. An all-`None`
//! or all-zero `condition_sd` reproduces `conditional_forecast` bit for bit
//! in the same run on every `var_cf.json` case (portable: both sides come
//! from the same binary), and an `#[ignore]`d fingerprint records the hard
//! output of the parent commit.

mod common;

use common::{as_mat, assert_mat_close, assert_rel_close, load_fixture};
use serde_json::Value;
use tsecon_linalg::faer::Mat;
use tsecon_var::{ConditionMode, ConditionalForecast, SoftMode, Trend, VarResults, VarSpec};

fn grid_of(v: &Value) -> Vec<Vec<Option<f64>>> {
    v.as_array()
        .expect("grid rows")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("grid row")
                .iter()
                .map(Value::as_f64)
                .collect()
        })
        .collect()
}

fn mode_of(case: &Value) -> SoftMode {
    match case["mode"].as_str().expect("mode") {
        "density" => SoftMode::Density,
        "signal" => SoftMode::Signal,
        other => panic!("unexpected mode {other}"),
    }
}

fn fit(series: &Value, case: &Value) -> VarResults {
    let data = as_mat(&series[case["series"].as_str().expect("series name")]);
    let trend = if case["trend"] == "c" {
        Trend::Constant
    } else {
        Trend::None
    };
    let p = case["p"].as_u64().expect("p") as usize;
    VarSpec::new(p, trend).unwrap().fit(data.as_ref()).unwrap()
}

fn soft_case(series: &Value, case: &Value) -> (VarResults, ConditionalForecast) {
    let res = fit(series, case);
    let steps = case["steps"].as_u64().expect("steps") as usize;
    let alpha = case["alpha"].as_f64().expect("alpha");
    let cf = res
        .conditional_forecast_soft(
            steps,
            &grid_of(&case["conditions"]),
            &grid_of(&case["condition_sd_input"]),
            mode_of(case),
            alpha,
        )
        .unwrap();
    (res, cf)
}

/// (`var_cf.json` series, soft cases).
fn cases() -> (Value, Vec<Value>) {
    let series = load_fixture("var_cf.json")["series"].clone();
    let fx = load_fixture("var_cf_soft.json");
    let cs = fx["cases"].as_array().expect("cases").clone();
    assert_eq!(cs.len(), 10);
    (series, cs)
}

fn label(case: &Value) -> String {
    format!(
        "{}/{}",
        case["name"].as_str().unwrap(),
        case["mode"].as_str().unwrap()
    )
}

/// Every returned quantity of every case reproduces the NumPy closed form
/// at 1e-10; `mode`, `n_soft`, `condition_sd`, `constrained` exactly.
#[test]
fn golden_closed_form_every_case() {
    let (series, cs) = cases();
    for case in &cs {
        let name = label(case);
        let (_, cf) = soft_case(&series, case);
        let steps = case["steps"].as_u64().unwrap() as usize;
        assert_eq!(cf.steps, steps, "{name}: steps");
        assert_eq!(cf.alpha, case["alpha"].as_f64().unwrap(), "{name}: alpha");
        assert_eq!(cf.mode.as_str(), case["mode"].as_str().unwrap(), "{name}");
        assert_eq!(
            cf.n_soft,
            case["n_soft"].as_u64().unwrap() as usize,
            "{name}: n_soft"
        );
        assert_eq!(
            cf.n_constrained,
            case["n_constrained"].as_u64().unwrap() as usize,
            "{name}: n_constrained"
        );
        for (key, mat) in [
            ("point", &cf.point),
            ("unconditional", &cf.unconditional),
            ("se", &cf.se),
            ("unconditional_se", &cf.unconditional_se),
            ("lower", &cf.lower),
            ("upper", &cf.upper),
            ("shocks", &cf.shocks),
            ("orth_shocks", &cf.orth_shocks),
        ] {
            assert_mat_close(mat, &case[key], 1e-10, &format!("{name}: {key}"));
        }
        assert_eq!(cf.cov.len(), steps, "{name}: cov length");
        for h in 0..steps {
            assert_mat_close(
                &cf.cov[h],
                &case["cov"][h],
                1e-10,
                &format!("{name}: cov[{h}]"),
            );
        }
        assert_rel_close(
            cf.mahalanobis,
            case["mahalanobis"].as_f64().unwrap(),
            1e-10,
            &format!("{name}: mahalanobis"),
        );
        assert_rel_close(
            cf.mahalanobis_pvalue,
            case["mahalanobis_pvalue"].as_f64().unwrap(),
            1e-10,
            &format!("{name}: mahalanobis_pvalue"),
        );
        let sd = as_mat(&case["condition_sd"]);
        for h in 0..steps {
            for j in 0..sd.ncols() {
                assert_eq!(
                    cf.condition_sd[(h, j)].to_bits(),
                    sd[(h, j)].to_bits(),
                    "{name}: condition_sd[({h},{j})]"
                );
                assert_eq!(
                    cf.constrained[h][j],
                    case["constrained"][h][j].as_bool().unwrap(),
                    "{name}: constrained[{h}][{j}]"
                );
            }
        }
    }
}

/// The statsmodels leg agrees with the crate at 1e-8 on the path, the
/// per-horizon covariance and the implied shocks; its future log-likelihood
/// `log N(r; 0, M)` equals `-0.5 (m ln 2 pi + ln det M + mahalanobis)` with
/// the crate's mahalanobis at 1e-8 (statsmodels' check on the statistic).
#[test]
fn golden_statsmodels_leg_every_case() {
    let (series, cs) = cases();
    for case in &cs {
        let name = label(case);
        let (_, cf) = soft_case(&series, case);
        assert_mat_close(
            &cf.point,
            &case["bgl_point"],
            1e-8,
            &format!("{name}: point vs VARMAX"),
        );
        assert_mat_close(
            &cf.shocks,
            &case["bgl_shocks"],
            1e-8,
            &format!("{name}: shocks vs VARMAX"),
        );
        for h in 0..cf.steps {
            assert_mat_close(
                &cf.cov[h],
                &case["bgl_cov"][h],
                1e-8,
                &format!("{name}: cov[{h}] vs VARMAX"),
            );
        }
        for key in ["point", "cov", "shocks"] {
            let dev = case["bgl_max_abs_dev"][key].as_f64().unwrap();
            assert!(
                dev < 1e-10,
                "{name}: generator-recorded {key} deviation {dev}"
            );
        }
        let m = cf.n_constrained as f64;
        let ln_det_m = case["ln_det_m"].as_f64().unwrap();
        let ll = -0.5 * (m * (2.0 * std::f64::consts::PI).ln() + ln_det_m + cf.mahalanobis);
        assert_rel_close(
            ll,
            case["bgl_llf_future"].as_f64().unwrap(),
            1e-8,
            &format!("{name}: future log-likelihood"),
        );
    }
}

/// Density mode: the Andersson-Palmqvist-Waggoner structural form
/// (Moore-Penrose, independent of the factor `P`) at 1e-8.
#[test]
fn golden_apw_structural_form_density_cases() {
    let (series, cs) = cases();
    let mut n = 0;
    for case in cs.iter().filter(|c| c["mode"] == "density") {
        let name = label(case);
        let (_, cf) = soft_case(&series, case);
        assert_mat_close(
            &cf.point,
            &case["apw_point"],
            1e-8,
            &format!("{name}: apw point"),
        );
        for h in 0..cf.steps {
            assert_mat_close(
                &cf.cov[h],
                &case["apw_cov"][h],
                1e-8,
                &format!("{name}: apw cov[{h}]"),
            );
        }
        n += 1;
    }
    assert_eq!(n, 5);
}

/// The exact cells. Density: every conditioned cell holds its value
/// bitwise, a soft cell's se is its sd bitwise, its variance is sd*sd and
/// its covariance with every other conditioned cell of its horizon exactly
/// 0. Both modes: hard cells (sd 0) are pinned bitwise with an exactly zero
/// row and column. Signal: a soft cell is NOT pinned and is tighter than
/// both its reading's sd and its unconditional se.
#[test]
fn exact_cells_in_both_modes() {
    let (series, cs) = cases();
    for case in &cs {
        let name = label(case);
        let (res, cf) = soft_case(&series, case);
        let k = res.neqs;
        let conds = grid_of(&case["conditions"]);
        let density = cf.mode == ConditionMode::Density;
        for (h, row) in conds.iter().enumerate() {
            for (j, c) in row.iter().enumerate() {
                let Some(v) = *c else { continue };
                let sd = cf.condition_sd[(h, j)];
                assert!(cf.constrained[h][j], "{name}: ({h},{j}) flagged");
                if sd == 0.0 || density {
                    assert_eq!(
                        cf.point[(h, j)].to_bits(),
                        v.to_bits(),
                        "{name}: point ({h},{j})"
                    );
                }
                if sd == 0.0 {
                    assert_eq!(cf.se[(h, j)], 0.0, "{name}: hard se ({h},{j})");
                    for l in 0..k {
                        assert_eq!(cf.cov[h][(j, l)], 0.0, "{name}: cov[{h}] row {j}");
                        assert_eq!(cf.cov[h][(l, j)], 0.0, "{name}: cov[{h}] col {j}");
                    }
                } else if density {
                    assert_eq!(
                        cf.se[(h, j)].to_bits(),
                        sd.to_bits(),
                        "{name}: se ({h},{j})"
                    );
                    assert_eq!(
                        cf.cov[h][(j, j)].to_bits(),
                        (sd * sd).to_bits(),
                        "{name}: var ({h},{j})"
                    );
                    for l in 0..k {
                        if l != j && cf.constrained[h][l] {
                            assert_eq!(cf.cov[h][(j, l)], 0.0, "{name}: cov[{h}][{j}][{l}]");
                            assert_eq!(cf.cov[h][(l, j)], 0.0, "{name}: cov[{h}][{l}][{j}]");
                        }
                    }
                } else {
                    assert_ne!(cf.point[(h, j)], v, "{name}: signal soft cell pinned");
                    // se^2 = sd^2 (1 - sd^2 [M^{-1}]_ii) < sd^2; at sd = 1e-9 the
                    // factor rounds to 1, so the bound is <= in floating point.
                    assert!(cf.se[(h, j)] <= sd, "{name}: signal se ({h},{j}) vs sd");
                    assert!(
                        cf.se[(h, j)] < cf.unconditional_se[(h, j)],
                        "{name}: signal se ({h},{j}) vs unconditional"
                    );
                }
            }
        }
    }
    // The two-soft-cell case: the soft x free cross term is left as
    // computed (not zeroed) — only the conditioned block is set.
    let case = cs
        .iter()
        .find(|c| c["name"] == "var2c_two_soft_same_h")
        .unwrap();
    let (_, cf) = soft_case(&series, case);
    assert!(cf.cov[1][(0, 2)].abs() > 1e-3, "{}", cf.cov[1][(0, 2)]);
    assert_eq!(cf.cov[1][(0, 1)], 0.0);
}

/// A single cell conditioned at its unconditional forecast with sd equal to
/// its unconditional se: the density forecast IS the unconditional one.
#[test]
fn density_identity_returns_the_unconditional_distribution() {
    let (series, cs) = cases();
    let case = cs
        .iter()
        .find(|c| c["name"] == "var2c_density_identity")
        .unwrap();
    let (res, cf) = soft_case(&series, case);
    let mse = res.forecast_cov(cf.steps).unwrap();
    assert_eq!(mse.len(), cf.steps);
    for (h, (got, want)) in cf.cov.iter().zip(&mse).enumerate() {
        for a in 0..res.neqs {
            assert!((cf.point[(h, a)] - cf.unconditional[(h, a)]).abs() < 1e-12);
            for c in 0..res.neqs {
                let d = (got[(a, c)] - want[(a, c)]).abs();
                assert!(
                    d < 1e-12 * want[(a, c)].abs().max(1.0),
                    "cov[{h}][{a}][{c}]: {d}"
                );
            }
        }
    }
    assert!(cf.mahalanobis.abs() < 1e-20);
}

fn assert_bitwise_equal(a: &ConditionalForecast, b: &ConditionalForecast, what: &str) {
    let mats: [(&str, &Mat<f64>, &Mat<f64>); 8] = [
        ("point", &a.point, &b.point),
        ("unconditional", &a.unconditional, &b.unconditional),
        ("se", &a.se, &b.se),
        ("unconditional_se", &a.unconditional_se, &b.unconditional_se),
        ("lower", &a.lower, &b.lower),
        ("upper", &a.upper, &b.upper),
        ("shocks", &a.shocks, &b.shocks),
        ("orth_shocks", &a.orth_shocks, &b.orth_shocks),
    ];
    for (key, x, y) in mats {
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert_eq!(
                    x[(i, j)].to_bits(),
                    y[(i, j)].to_bits(),
                    "{what}: {key}[({i},{j})]"
                );
            }
        }
    }
    for (h, (x, y)) in a.cov.iter().zip(&b.cov).enumerate() {
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert_eq!(x[(i, j)].to_bits(), y[(i, j)].to_bits(), "{what}: cov[{h}]");
            }
        }
    }
    assert_eq!(
        a.mahalanobis.to_bits(),
        b.mahalanobis.to_bits(),
        "{what}: mahalanobis"
    );
    assert_eq!(
        a.mahalanobis_pvalue.to_bits(),
        b.mahalanobis_pvalue.to_bits(),
        "{what}: mahalanobis_pvalue"
    );
    assert_eq!(a.constrained, b.constrained, "{what}: constrained");
    assert_eq!(a.n_constrained, b.n_constrained, "{what}: n_constrained");
    assert_eq!((a.steps, a.alpha.to_bits()), (b.steps, b.alpha.to_bits()));
    assert_eq!(a.mode, b.mode, "{what}: mode");
    assert_eq!(a.n_soft, b.n_soft, "{what}: n_soft");
}

/// SAME-RUN bit identity: an all-`None` and an all-zero `condition_sd`, in
/// either `SoftMode`, reproduce `conditional_forecast` bit for bit on all
/// six `var_cf.json` cases, with `mode = Hard`, `n_soft = 0` and a zero
/// `condition_sd`. (Identity with 0.10.0 itself is `var_cf_golden.rs`'s
/// unchanged 1e-10 pins plus the ignored fingerprint below.)
#[test]
fn no_soft_cell_is_the_hard_path_bit_for_bit() {
    let fx = load_fixture("var_cf.json");
    let cs = fx["cases"].as_array().unwrap();
    assert_eq!(cs.len(), 6);
    for case in cs {
        let name = case["name"].as_str().unwrap();
        let res = fit(&fx["series"], case);
        let steps = case["steps"].as_u64().unwrap() as usize;
        let alpha = case["alpha"].as_f64().unwrap();
        let conds = grid_of(&case["conditions"]);
        let hard = res.conditional_forecast(steps, &conds, alpha).unwrap();
        assert_eq!(hard.mode, ConditionMode::Hard);
        assert_eq!(hard.n_soft, 0);
        let none = vec![vec![None; res.neqs]; conds.len()];
        let zero = vec![vec![Some(0.0); res.neqs]; conds.len()];
        let nan = vec![vec![Some(f64::NAN); res.neqs]; conds.len()];
        for (sd_name, sd) in [("None", &none), ("zero", &zero), ("NaN", &nan)] {
            for mode in [SoftMode::Density, SoftMode::Signal] {
                let what = format!("{name}: {sd_name} sd, {mode:?}");
                let s = res
                    .conditional_forecast_soft(steps, &conds, sd, mode, alpha)
                    .unwrap();
                assert_bitwise_equal(&s, &hard, &what);
                for h in 0..steps {
                    for j in 0..res.neqs {
                        assert_eq!(s.condition_sd[(h, j)].to_bits(), 0.0f64.to_bits(), "{what}");
                    }
                }
            }
        }
    }
}

/// FNV-1a over `to_bits()` of `point`, `unconditional`, `se`,
/// `unconditional_se`, `lower`, `upper`, `shocks`, `orth_shocks`, `cov`,
/// `mahalanobis` and `mahalanobis_pvalue` of the six `var_cf.json` hard
/// cases, in that order.
fn hard_fingerprint() -> u64 {
    let fx = load_fixture("var_cf.json");
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut push = |x: f64| {
        for b in x.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
        }
    };
    for case in fx["cases"].as_array().unwrap() {
        let res = fit(&fx["series"], case);
        let steps = case["steps"].as_u64().unwrap() as usize;
        let alpha = case["alpha"].as_f64().unwrap();
        let cf = res
            .conditional_forecast(steps, &grid_of(&case["conditions"]), alpha)
            .unwrap();
        let mut push_mat = |m: &Mat<f64>| {
            for i in 0..m.nrows() {
                for j in 0..m.ncols() {
                    push(m[(i, j)]);
                }
            }
        };
        for m in [
            &cf.point,
            &cf.unconditional,
            &cf.se,
            &cf.unconditional_se,
            &cf.lower,
            &cf.upper,
            &cf.shocks,
            &cf.orth_shocks,
        ] {
            push_mat(m);
        }
        for c in &cf.cov {
            push_mat(c);
        }
        push(cf.mahalanobis);
        push(cf.mahalanobis_pvalue);
    }
    hash
}

/// The hard path is a pure code motion of the 0.10.0 body. This constant
/// was captured by running the same fingerprint against the unmodified
/// parent commit (`03de810`, x86-64 Linux) and reproduced after the
/// refactor. It is `#[ignore]`d by house policy (see
/// `pointwise_bitwise_baseline.rs`: bit-level snapshots are per platform,
/// because faer dispatches different SIMD kernels per CPU); run it with
/// `--ignored` on an x86-64 machine — a mismatch there is conclusive.
#[test]
#[ignore = "per-platform bit snapshot (x86-64); the same-run test above is the CI guard"]
fn hard_path_fingerprint_matches_the_parent_commit() {
    assert_eq!(hard_fingerprint(), 0xe757_5df3_c374_282e);
}
