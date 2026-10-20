//! Shared loaders for the static-panel golden and property tests: the
//! datasets of `fixtures/panel_static.json` (the EmplUK inputs come from
//! `fixtures/panel_unbalanced.json`, bit-identical to `panel_fe`'s).

#![allow(dead_code)]

use serde_json::Value;
use tsecon_linalg::faer::Mat;
use tsecon_panel::PanelData;

pub fn load(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

/// A JSON row; `null` (a masked-out cell) becomes NaN.
pub fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

pub fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(f64s).collect()
}

pub fn to_mat(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

pub fn mask_of(v: &Value) -> Vec<Vec<bool>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() == 1)
                .collect()
        })
        .collect()
}

/// A raw panel: outcome, named regressors, mask.
pub struct Raw {
    pub y: Mat<f64>,
    pub regs: Vec<(String, Mat<f64>)>,
    pub mask: Vec<Vec<bool>>,
}

impl Raw {
    pub fn panel(&self) -> PanelData {
        PanelData::unbalanced(self.y.clone(), self.regs.clone(), &self.mask).expect("panel")
    }

    pub fn panel_without_regressors(&self) -> PanelData {
        PanelData::unbalanced(self.y.clone(), vec![], &self.mask).expect("panel")
    }

    pub fn n(&self) -> usize {
        self.y.nrows()
    }

    pub fn t(&self) -> usize {
        self.y.ncols()
    }
}

/// The EmplUK panel (log employment on log wage and log capital).
pub fn empluk() -> Raw {
    let fx = load("panel_unbalanced.json");
    let ds = &fx["datasets"]["empluk"];
    Raw {
        y: to_mat(&matrix(&ds["log_emp"])),
        regs: vec![
            ("log_wage".into(), to_mat(&matrix(&ds["log_wage"]))),
            ("log_capital".into(), to_mat(&matrix(&ds["log_capital"]))),
        ],
        mask: mask_of(&ds["mask"]),
    }
}

/// Each entity's mean of `x` over its observed cells, broadcast to N x T.
pub fn entity_means(x: &Mat<f64>, mask: &[Vec<bool>]) -> Mat<f64> {
    let (n, t) = (x.nrows(), x.ncols());
    let means: Vec<f64> = (0..n)
        .map(|i| {
            let (mut s, mut c) = (0.0, 0.0);
            for tt in 0..t {
                if mask[i][tt] {
                    s += x[(i, tt)];
                    c += 1.0;
                }
            }
            s / c
        })
        .collect();
    Mat::from_fn(n, t, |i, _| means[i])
}

/// A dataset of `panel_static.json` by name (EmplUK variants built from
/// `panel_unbalanced.json`).
pub fn dataset(fx: &Value, name: &str) -> Raw {
    match name {
        "empluk" => empluk(),
        "empluk_yeardummies" => {
            let mut raw = empluk();
            let (n, t) = (raw.n(), raw.t());
            for c in 1..t {
                raw.regs.push((
                    format!("d{c}"),
                    Mat::from_fn(n, t, |_, tt| if tt == c { 1.0 } else { 0.0 }),
                ));
            }
            raw
        }
        "empluk_mundlak" => {
            let mut raw = empluk();
            let mw = entity_means(&raw.regs[0].1, &raw.mask);
            let mk = entity_means(&raw.regs[1].1, &raw.mask);
            raw.regs.push(("mean_log_wage".into(), mw));
            raw.regs.push(("mean_log_capital".into(), mk));
            raw
        }
        other => {
            let ds = &fx["datasets"][other];
            let names: Vec<String> = ds["regressor_names"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            let regs = names
                .into_iter()
                .zip(ds["regressors"].as_array().unwrap())
                .map(|(nm, v)| (nm, to_mat(&matrix(v))))
                .collect();
            Raw {
                y: to_mat(&matrix(&ds["outcome"])),
                regs,
                mask: mask_of(&ds["mask"]),
            }
        }
    }
}

/// The panel a fixture case runs on (`regressors`: "all" or "none").
pub fn case_panel(fx: &Value, case: &Value) -> PanelData {
    let raw = dataset(fx, case["dataset"].as_str().unwrap());
    match case["regressors"].as_str().unwrap() {
        "none" => raw.panel_without_regressors(),
        _ => raw.panel(),
    }
}

pub fn assert_close(got: f64, want: f64, rtol: f64, floor: f64, what: &str) {
    let denom = want.abs().max(floor);
    let rel = ((got - want) / denom).abs();
    assert!(rel < rtol, "{what}: got {got}, want {want} (rel {rel:e})");
}
