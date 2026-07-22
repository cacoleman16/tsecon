//! Golden and property tests for the Fry-Pagan (2011) median-target SVAR
//! selection (`tsecon_ident::median_target`).
//!
//! `fixtures/fry_pagan_svar.json` is produced by
//! `fixtures/generate_fry_pagan_svar_fixtures.py`. The fixture STORES, as
//! input, a fixed set of `D` sign-normalized structural IRF draws and an
//! INDEPENDENT NumPy computation of the pointwise median, per-draw median-target
//! criterion `MT(d)`, and the winning index. Because the draws are the input,
//! the only thing under test is the pure-arithmetic SELECTION RULE:
//!
//! * **`golden_matches_numpy`** feeds the stored draws + target cells to
//!   [`median_target`] and bit-checks the returned `mt_index` (exactly) and
//!   `mt_statistic` / `median_irf` (to 1e-10) against the stored NumPy values —
//!   the cross-implementation golden.
//! * **`winner_is_in_the_accepted_set_and_minimizes`** recomputes `MT(d)` for
//!   every draw inside the test and confirms the returned draw attains the
//!   minimum and is a genuine member of the accepted set (the Fry-Pagan
//!   "single coherent model" property).
//! * consistency checks (median band is the pointwise median; determinism)
//!   round it out.

use serde_json::Value;
use tsecon_ident::median_target;
use tsecon_linalg::faer::Mat;

const TOL: f64 = 1e-10;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/fry_pagan_svar.json",
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

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(f64s).collect()
}

fn u(v: &Value) -> usize {
    v.as_u64().expect("uint") as usize
}

fn mat(v: &Value) -> Mat<f64> {
    let r = rows(v);
    let nr = r.len();
    let nc = r[0].len();
    Mat::from_fn(nr, nc, |i, j| r[i][j])
}

/// One draw's `[H+1]` IRF matrices.
fn theta_mats(v: &Value) -> Vec<Mat<f64>> {
    v.as_array().expect("array").iter().map(mat).collect()
}

/// The full `[D]` accepted set.
fn draw_set(v: &Value) -> Vec<Vec<Mat<f64>>> {
    v.as_array()
        .expect("array")
        .iter()
        .map(theta_mats)
        .collect()
}

fn target_cells(v: &Value) -> Vec<(usize, usize, usize)> {
    v.as_array()
        .expect("cells")
        .iter()
        .map(|c| {
            let t = c.as_array().expect("triple");
            (u(&t[0]), u(&t[1]), u(&t[2]))
        })
        .collect()
}

/// Independent (in-test) recomputation of `MT(d)` for one draw.
fn mt_of(draws: &[Vec<Mat<f64>>], cells: &[(usize, usize, usize)], d_idx: usize) -> f64 {
    let mut mt = 0.0;
    for &(i, j, h) in cells {
        let mut vals: Vec<f64> = draws.iter().map(|d| d[h][(i, j)]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        // Type-7 median.
        let len = vals.len();
        let pos = 0.5 * (len - 1) as f64;
        let lo = pos.floor() as usize;
        let med = if lo >= len - 1 {
            vals[len - 1]
        } else {
            vals[lo] + (pos - lo as f64) * (vals[lo + 1] - vals[lo])
        };
        let mean: f64 = vals.iter().sum::<f64>() / len as f64;
        let var: f64 = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / len as f64;
        let sd = var.sqrt();
        if sd > 0.0 {
            let z = (draws[d_idx][h][(i, j)] - med) / sd;
            mt += z * z;
        }
    }
    mt
}

fn scenarios(fx: &Value) -> &Vec<Value> {
    fx["scenarios"].as_array().expect("scenarios array")
}

#[test]
fn golden_matches_numpy() {
    let fx = load();
    let scenarios = scenarios(&fx);
    assert!(!scenarios.is_empty());
    for sc in scenarios {
        let name = sc["name"].as_str().unwrap_or("?");
        let draws = draw_set(&sc["draws"]);
        let cells = target_cells(&sc["target_cells"]);
        let out = median_target(&draws, &cells)
            .unwrap_or_else(|e| panic!("{name}: median_target failed: {e}"));

        let exp = &sc["expected"];
        // mt_index is an integer selection -> must match EXACTLY.
        assert_eq!(
            out.index,
            u(&exp["mt_index"]),
            "{name}: selected draw index differs from NumPy"
        );
        // mt_statistic and the median band to 1e-10.
        let exp_stat = exp["mt_statistic"].as_f64().expect("f64");
        assert!(
            (out.statistic - exp_stat).abs() < TOL,
            "{name}: mt_statistic {} vs expected {}",
            out.statistic,
            exp_stat
        );
        let exp_med = rows_of_theta(&exp["median_irf"]);
        assert_eq!(out.median_irf.len(), exp_med.len(), "{name} horizon length");
        for (h, want) in exp_med.iter().enumerate() {
            for (i, want_row) in want.iter().enumerate() {
                for (j, &wv) in want_row.iter().enumerate() {
                    let got = out.median_irf[h][(i, j)];
                    assert!(
                        (got - wv).abs() < TOL,
                        "{name}: median_irf[{h}][{i}][{j}] {got} vs {wv}"
                    );
                }
            }
        }
    }
}

/// `[H+1]` list of row-major matrices as nested `Vec`.
fn rows_of_theta(v: &Value) -> Vec<Vec<Vec<f64>>> {
    v.as_array().expect("theta").iter().map(rows).collect()
}

#[test]
fn winner_is_in_the_accepted_set_and_minimizes() {
    let fx = load();
    for sc in scenarios(&fx) {
        let name = sc["name"].as_str().unwrap_or("?");
        let draws = draw_set(&sc["draws"]);
        let cells = target_cells(&sc["target_cells"]);
        let out = median_target(&draws, &cells).expect("median_target");

        // (a) The winner index is a genuine member of the accepted set.
        assert!(
            out.index < draws.len(),
            "{name}: index out of the accepted set"
        );

        // (b) It attains the minimum criterion recomputed independently.
        let winner_mt = mt_of(&draws, &cells, out.index);
        for d in 0..draws.len() {
            assert!(
                mt_of(&draws, &cells, d) >= winner_mt - 1e-10,
                "{name}: draw {d} beats the selected winner {}",
                out.index
            );
        }
        // (c) The reported statistic equals that independent recomputation.
        assert!(
            (out.statistic - winner_mt).abs() < 1e-10,
            "{name}: reported statistic {} vs recomputed {}",
            out.statistic,
            winner_mt
        );
    }
}

#[test]
fn median_band_is_the_pointwise_median() {
    // The returned median_irf must equal, cell by cell, the median of the draws
    // at that cell (recomputed independently for every cell, not just targets).
    let fx = load();
    for sc in scenarios(&fx) {
        let name = sc["name"].as_str().unwrap_or("?");
        let draws = draw_set(&sc["draws"]);
        let cells = target_cells(&sc["target_cells"]);
        let out = median_target(&draws, &cells).expect("median_target");
        let n_h = draws[0].len();
        let n = draws[0][0].nrows();
        for h in 0..n_h {
            for i in 0..n {
                for j in 0..n {
                    let mut vals: Vec<f64> = draws.iter().map(|d| d[h][(i, j)]).collect();
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let len = vals.len();
                    let pos = 0.5 * (len - 1) as f64;
                    let lo = pos.floor() as usize;
                    let med = if lo >= len - 1 {
                        vals[len - 1]
                    } else {
                        vals[lo] + (pos - lo as f64) * (vals[lo + 1] - vals[lo])
                    };
                    assert!(
                        (out.median_irf[h][(i, j)] - med).abs() < 1e-12,
                        "{name}: median band mismatch at ({i},{j},{h})"
                    );
                }
            }
        }
    }
}

#[test]
fn is_bit_reproducible() {
    let fx = load();
    for sc in scenarios(&fx) {
        let draws = draw_set(&sc["draws"]);
        let cells = target_cells(&sc["target_cells"]);
        let a = median_target(&draws, &cells).expect("run a");
        let b = median_target(&draws, &cells).expect("run b");
        assert_eq!(a.index, b.index);
        assert_eq!(a.statistic.to_bits(), b.statistic.to_bits());
    }
}
