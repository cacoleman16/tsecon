//! Properties of the static-panel menu (random effects, between, first
//! difference, Fama-MacBeth): invariances, cross-callable identities, the
//! documented divergences from linearmodels, and the refusals that name
//! their parameter.

mod static_common;

use static_common::{dataset, empluk, entity_means, f64s, load, matrix, Raw};
use tsecon_hac::{ols, SeType};
use tsecon_linalg::faer::Mat;
use tsecon_panel::{
    fama_macbeth, panel_between, panel_first_difference, panel_ols_fe, panel_random_effects,
    BetweenConfig, FamaMacBethSe, PanelData, PanelError, PanelSeType, RandomEffectsConfig,
    StaticSeType,
};

const MENU: [StaticSeType; 4] = [
    StaticSeType::NonRobust,
    StaticSeType::Hc1,
    StaticSeType::ClusterEntity,
    StaticSeType::DriscollKraay { bandwidth: 4.0 },
];

/// Minimal LCG so the tests depend on no RNG crate.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// A seeded panel `y = 1 + 0.7 x1 - 0.4 x2 + u_i + e` with a random effect
/// and an optional ragged mask (entry, exit, internal gaps).
fn seeded(n: usize, t: usize, seed: u64, ragged: bool) -> Raw {
    let mut rng = Lcg(seed);
    let u: Vec<f64> = (0..n).map(|_| rng.gaussian()).collect();
    let a: Vec<f64> = (0..n).map(|_| 0.5 * rng.gaussian()).collect();
    let x1 = Mat::from_fn(n, t, |i, _| a[i] + rng.gaussian());
    let x2 = Mat::from_fn(n, t, |_, _| 2.0 + rng.gaussian());
    let y = Mat::from_fn(n, t, |i, tt| {
        1.0 + 0.7 * x1[(i, tt)] - 0.4 * x2[(i, tt)] + u[i] + 0.8 * rng.gaussian()
    });
    let mask: Vec<Vec<bool>> = (0..n)
        .map(|i| {
            (0..t)
                .map(|tt| {
                    if !ragged {
                        return true;
                    }
                    let start = i % 3;
                    let end = t - (i % 2);
                    tt >= start && tt < end && (i * 7 + tt * 3) % 11 != 0
                })
                .collect()
        })
        .collect();
    Raw {
        y,
        regs: vec![("x1".into(), x1), ("x2".into(), x2)],
        mask,
    }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn cov_bits(m: &Mat<f64>) -> Vec<u64> {
    let k = m.nrows();
    (0..k)
        .flat_map(|a| (0..k).map(move |b| (a, b)))
        .map(|(a, b)| m[(a, b)].to_bits())
        .collect()
}

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (j, (x, y)) in a.iter().zip(b).enumerate() {
        let rel = (x - y).abs() / y.abs().max(1e-300);
        assert!(rel < tol, "{what}[{j}]: {x} vs {y} (rel {rel:e})");
    }
}

/// Every output of the four estimators, flattened (for bit / tolerance
/// comparisons between two panels that must give the same answer).
fn all_outputs(data: &PanelData) -> Vec<f64> {
    let mut out = Vec::new();
    for ss in [false, true] {
        let re = panel_random_effects(
            data,
            RandomEffectsConfig {
                constant: true,
                small_sample: ss,
            },
        )
        .unwrap();
        out.extend(&re.params);
        out.extend([re.sigma2_eps, re.sigma2_effects, re.rho]);
        for se in MENU {
            let inf = re.inference(se).unwrap();
            out.extend(&inf.bse);
            out.extend(cov_bits(&inf.cov).iter().map(|b| f64::from_bits(*b)));
        }
    }
    for rw in [false, true] {
        let be = panel_between(
            data,
            BetweenConfig {
                constant: true,
                reweight: rw,
            },
        )
        .unwrap();
        out.extend(&be.params);
        for se in &MENU[..3] {
            out.extend(&be.inference(*se).unwrap().bse);
        }
    }
    for c in [false, true] {
        let fd = panel_first_difference(data, c).unwrap();
        out.extend(&fd.params);
        for se in MENU {
            out.extend(&fd.inference(se).unwrap().bse);
        }
    }
    let fm = fama_macbeth(data, true).unwrap();
    out.extend(&fm.params);
    out.extend(&fm.inference(FamaMacBethSe::Standard).unwrap().bse);
    out.extend(
        &fm.inference(FamaMacBethSe::Hac { bandwidth: 2.0 })
            .unwrap()
            .bse,
    );
    out
}

#[test]
fn an_all_ones_mask_is_bit_identical_to_no_mask() {
    let raw = seeded(12, 9, 1, false);
    let a = PanelData::balanced(raw.y.clone(), raw.regs.clone()).unwrap();
    let b = raw.panel();
    assert!(b.is_balanced());
    assert_eq!(bits(&all_outputs(&a)), bits(&all_outputs(&b)));
}

#[test]
fn masked_out_cells_are_never_read() {
    let raw = seeded(14, 10, 2, true);
    let base = all_outputs(&raw.panel());
    for garbage in [f64::NAN, f64::INFINITY, 1e300, -7.0] {
        let mut dirty = Raw {
            y: raw.y.clone(),
            regs: raw.regs.clone(),
            mask: raw.mask.clone(),
        };
        for i in 0..raw.n() {
            for t in 0..raw.t() {
                if !raw.mask[i][t] {
                    dirty.y[(i, t)] = garbage;
                    for (_, m) in dirty.regs.iter_mut() {
                        m[(i, t)] = garbage;
                    }
                }
            }
        }
        assert_eq!(
            bits(&all_outputs(&dirty.panel())),
            bits(&base),
            "garbage {garbage}"
        );
    }
}

#[test]
fn an_entity_permutation_leaves_every_estimate_unchanged() {
    let raw = seeded(15, 8, 3, true);
    let perm: Vec<usize> = (0..raw.n()).map(|i| (i * 7 + 3) % raw.n()).collect();
    let permuted = Raw {
        y: Mat::from_fn(raw.n(), raw.t(), |i, t| raw.y[(perm[i], t)]),
        regs: raw
            .regs
            .iter()
            .map(|(nm, m)| {
                (
                    nm.clone(),
                    Mat::from_fn(raw.n(), raw.t(), |i, t| m[(perm[i], t)]),
                )
            })
            .collect(),
        mask: (0..raw.n()).map(|i| raw.mask[perm[i]].clone()).collect(),
    };
    let a = all_outputs(&raw.panel());
    let b = all_outputs(&permuted.panel());
    for (j, (x, y)) in a.iter().zip(&b).enumerate() {
        let scale = y.abs().max(1e-12);
        assert!((x - y).abs() / scale < 1e-12, "output {j}: {x} vs {y}");
    }
    // theta follows the entities
    let ta = panel_random_effects(&raw.panel(), RandomEffectsConfig::default()).unwrap();
    let tb = panel_random_effects(&permuted.panel(), RandomEffectsConfig::default()).unwrap();
    for (i, &p) in perm.iter().enumerate() {
        assert!((tb.theta[i] - ta.theta[p]).abs() < 1e-12);
    }
}

/// Pooled OLS of the stacked observed cells on `[1, x...]` (tsecon-hac).
fn pooled_ols_fit(raw: &Raw) -> tsecon_hac::OlsFit {
    let cells: Vec<(usize, usize)> = (0..raw.n())
        .flat_map(|i| (0..raw.t()).map(move |t| (i, t)))
        .filter(|&(i, t)| raw.mask[i][t])
        .collect();
    let y: Vec<f64> = cells.iter().map(|&c| raw.y[c]).collect();
    let mut cols = vec![vec![1.0; cells.len()]];
    for (_, m) in &raw.regs {
        cols.push(cells.iter().map(|&c| m[c]).collect());
    }
    ols(&y, &cols).unwrap()
}

#[test]
fn random_effects_with_a_truncated_effect_variance_is_pooled_ols() {
    let fx = load("panel_static.json");
    let raw = dataset(&fx, "re_truncated");
    let re = panel_random_effects(&raw.panel(), RandomEffectsConfig::default()).unwrap();
    assert_eq!(re.sigma2_effects, 0.0);
    assert!(re.theta.iter().all(|&t| t == 0.0));
    let po = pooled_ols_fit(&raw);
    close(&re.params, &po.params, 1e-12, "RE vs pooled params");
    let inf = re.inference(StaticSeType::Hc1).unwrap();
    let pinf = po.inference(SeType::Hc1).unwrap();
    close(&inf.bse, &pinf.bse, 1e-12, "RE vs pooled HC1 bse");
    let inf = re.inference(StaticSeType::NonRobust).unwrap();
    let pinf = po.inference(SeType::NonRobust).unwrap();
    close(&inf.bse, &pinf.bse, 1e-12, "RE vs pooled nonrobust bse");
}

/// The Mundlak / correlated-random-effects identity: random effects on
/// `[1, x, mean_i(x)]` (means over each entity's observed cells) gives the
/// fixed-effects slopes exactly, on unbalanced panels too.
fn mundlak_gap(raw: &Raw) -> f64 {
    let fe = panel_ols_fe(&raw.panel()).unwrap();
    let mut aug = Raw {
        y: raw.y.clone(),
        regs: raw.regs.clone(),
        mask: raw.mask.clone(),
    };
    for (nm, m) in &raw.regs {
        aug.regs
            .push((format!("mean_{nm}"), entity_means(m, &raw.mask)));
    }
    let re = panel_random_effects(&aug.panel(), RandomEffectsConfig::default()).unwrap();
    let k = raw.regs.len();
    re.params[1..=k]
        .iter()
        .zip(&fe.params)
        .map(|(a, b)| ((a - b) / b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn mundlak_random_effects_slopes_equal_the_fixed_effects_slopes() {
    let ragged = seeded(30, 10, 4, true);
    assert!(!ragged.panel().is_balanced());
    let gap = mundlak_gap(&ragged);
    assert!(gap < 1e-12, "ragged panel: max rel gap {gap:e}");
    let gap = mundlak_gap(&empluk());
    assert!(gap < 1e-12, "EmplUK: max rel gap {gap:e}");
    // The cluster-robust Wald on the mean coefficients reproduces the
    // fixture's (linearmodels') statistic.
    let fx = load("panel_static.json");
    let raw = dataset(&fx, "empluk_mundlak");
    let re = panel_random_effects(&raw.panel(), RandomEffectsConfig::default()).unwrap();
    let inf = re.inference(StaticSeType::ClusterEntity).unwrap();
    let (b1, b2) = (re.params[3], re.params[4]);
    let (v11, v12, v22) = (inf.cov[(3, 3)], inf.cov[(3, 4)], inf.cov[(4, 4)]);
    let det = v11 * v22 - v12 * v12;
    let wald = (b1 * b1 * v22 - 2.0 * b1 * b2 * v12 + b2 * b2 * v11) / det;
    let want = fx["mundlak"]["cluster_wald_mean_coefficients"]
        .as_f64()
        .unwrap();
    assert!(((wald - want) / want).abs() < 1e-9, "Wald {wald} vs {want}");
}

#[test]
fn at_two_periods_first_differences_equal_fixed_effects() {
    let raw = seeded(40, 2, 5, false);
    let data = raw.panel();
    let fd = panel_first_difference(&data, false).unwrap();
    let fe = panel_ols_fe(&data).unwrap();
    close(&fd.params, &fe.params, 1e-12, "FD vs FE slopes at T = 2");
    // ... and the cluster covariances agree up to the df conventions:
    // FD has m = N rows and K = k; FE n = 2N rows, n/(n-k) nested factor.
    let fdi = fd.inference(StaticSeType::ClusterEntity).unwrap();
    let fei = fe.inference(PanelSeType::ClusterEntity).unwrap();
    let (n, m, k) = (2.0 * 40.0, 40.0, 2.0);
    for j in 0..2 {
        let a = fdi.cov[(j, j)] / (m / (m - k));
        let b = fei.cov[(j, j)] / (n / (n - k));
        assert!(
            ((a - b) / b).abs() < 1e-10,
            "cluster meat/bread at T = 2: {a} vs {b}"
        );
    }
}

#[test]
fn small_sample_and_reweight_are_no_ops_when_every_entity_has_the_same_count() {
    // A fully balanced panel, and one whose mask drops a whole period for
    // everyone (so `is_balanced()` is false while every T_i is equal —
    // linearmodels' np.ptp(count) == 0 rule, not PanelData::is_balanced).
    let full = seeded(20, 7, 6, false);
    let mut hole = Raw {
        y: full.y.clone(),
        regs: full.regs.clone(),
        mask: full.mask.clone(),
    };
    for row in hole.mask.iter_mut() {
        row[3] = false;
    }
    assert!(!hole.panel().is_balanced());
    for raw in [&full, &hole] {
        let data = raw.panel();
        let a = panel_random_effects(
            &data,
            RandomEffectsConfig {
                constant: true,
                small_sample: false,
            },
        )
        .unwrap();
        let b = panel_random_effects(
            &data,
            RandomEffectsConfig {
                constant: true,
                small_sample: true,
            },
        )
        .unwrap();
        assert_eq!(bits(&a.params), bits(&b.params));
        assert_eq!(a.sigma2_effects.to_bits(), b.sigma2_effects.to_bits());
        let a = panel_between(
            &data,
            BetweenConfig {
                constant: true,
                reweight: false,
            },
        )
        .unwrap();
        let b = panel_between(
            &data,
            BetweenConfig {
                constant: true,
                reweight: true,
            },
        )
        .unwrap();
        assert_eq!(bits(&a.params), bits(&b.params));
        for se in &MENU[..3] {
            assert_eq!(
                cov_bits(&a.inference(*se).unwrap().cov),
                cov_bits(&b.inference(*se).unwrap().cov)
            );
        }
    }
}

#[test]
fn between_is_ols_on_the_entity_means() {
    let raw = seeded(25, 8, 7, true);
    let data = raw.panel();
    let be = panel_between(&data, BetweenConfig::default()).unwrap();
    let means: Vec<(f64, Vec<f64>)> = (0..raw.n())
        .filter(|&i| raw.mask[i].iter().any(|&m| m))
        .map(|i| {
            let ts: Vec<usize> = (0..raw.t()).filter(|&t| raw.mask[i][t]).collect();
            let c = ts.len() as f64;
            let yb = ts.iter().map(|&t| raw.y[(i, t)]).sum::<f64>() / c;
            let xb = raw
                .regs
                .iter()
                .map(|(_, m)| ts.iter().map(|&t| m[(i, t)]).sum::<f64>() / c)
                .collect();
            (yb, xb)
        })
        .collect();
    let y: Vec<f64> = means.iter().map(|m| m.0).collect();
    let mut cols = vec![vec![1.0; y.len()]];
    for j in 0..raw.regs.len() {
        cols.push(means.iter().map(|m| m.1[j]).collect());
    }
    let fit = ols(&y, &cols).unwrap();
    close(&be.params, &fit.params, 1e-12, "BE vs OLS on means");
    let h = fit.inference(SeType::Hc1).unwrap();
    close(
        &be.inference(StaticSeType::Hc1).unwrap().bse,
        &h.bse,
        1e-12,
        "BE hc1 vs ols hc1",
    );
    // clustering by entity with one row per entity is HC1
    close(
        &be.inference(StaticSeType::ClusterEntity).unwrap().bse,
        &h.bse,
        1e-12,
        "BE cluster vs hc1",
    );
    let nr = fit.inference(SeType::NonRobust).unwrap();
    close(
        &be.inference(StaticSeType::NonRobust).unwrap().bse,
        &nr.bse,
        1e-12,
        "BE nonrobust",
    );
}

#[test]
fn fama_macbeth_standard_error_is_the_dispersion_formula() {
    let raw = seeded(30, 16, 8, true);
    let fm = fama_macbeth(&raw.panel(), true).unwrap();
    let tv = fm.n_periods_used as f64;
    let used: Vec<&Vec<f64>> = fm
        .params_per_period
        .iter()
        .zip(&fm.periods_used)
        .filter(|(_, u)| **u)
        .map(|(b, _)| b)
        .collect();
    let std = fm.inference(FamaMacBethSe::Standard).unwrap();
    for j in 0..fm.params.len() {
        let mean = used.iter().map(|b| b[j]).sum::<f64>() / tv;
        assert!((mean - fm.params[j]).abs() < 1e-14);
        let var = used.iter().map(|b| (b[j] - mean).powi(2)).sum::<f64>() / (tv - 1.0);
        let se = (var / tv).sqrt();
        assert!(
            ((std.bse[j] - se) / se).abs() < 1e-12,
            "sd/sqrt(T_v) at {j}"
        );
    }
    // bandwidth 0 keeps only lag 0: the HAC is the standard SE
    let hac0 = fm.inference(FamaMacBethSe::Hac { bandwidth: 0.0 }).unwrap();
    assert_eq!(bits(&hac0.bse), bits(&std.bse));
}

#[test]
fn fama_macbeth_skips_a_period_with_fewer_entities_than_parameters() {
    let mut raw = seeded(12, 10, 9, false);
    for (i, row) in raw.mask.iter_mut().enumerate() {
        row[4] = i < 2; // 2 entities < K = 3
    }
    let fm = fama_macbeth(&raw.panel(), true).unwrap();
    assert!(!fm.periods_used[4]);
    assert!(fm.params_per_period[4].iter().all(|v| v.is_nan()));
    assert_eq!(fm.n_periods_used, 9);
    assert_eq!(fm.nobs, 12 * 9 + 2, "nobs counts every observed cell");
}

#[test]
fn first_differences_with_an_intercept_are_pooled_ols_on_the_differences() {
    let raw = seeded(18, 9, 10, true);
    let fd = panel_first_difference(&raw.panel(), true).unwrap();
    let mut dy = Vec::new();
    let mut cols = vec![Vec::new(), Vec::new(), Vec::new()];
    for i in 0..raw.n() {
        for t in 1..raw.t() {
            if raw.mask[i][t] && raw.mask[i][t - 1] {
                dy.push(raw.y[(i, t)] - raw.y[(i, t - 1)]);
                cols[0].push(1.0);
                for (j, (_, m)) in raw.regs.iter().enumerate() {
                    cols[j + 1].push(m[(i, t)] - m[(i, t - 1)]);
                }
            }
        }
    }
    let fit = ols(&dy, &cols).unwrap();
    assert_eq!(fd.nobs, dy.len());
    close(
        &fd.params,
        &fit.params,
        1e-12,
        "FD+const vs pooled OLS on differences",
    );
    close(
        &fd.inference(StaticSeType::Hc1).unwrap().bse,
        &fit.inference(SeType::Hc1).unwrap().bse,
        1e-12,
        "FD+const hc1",
    );
}

#[test]
fn first_differences_never_bridge_a_gap() {
    // Entity 0 misses period 3: the differences (0,3) and (0,4) do not
    // exist, and nothing differences period 4 against period 2.
    let mut raw = seeded(6, 7, 11, false);
    raw.mask[0][3] = false;
    let fd = panel_first_difference(&raw.panel(), false).unwrap();
    assert_eq!(fd.nobs, 6 * 6 - 2);
    assert_eq!(fd.n_entities, 6);
}

#[test]
fn a_period_nobody_observes_is_not_bridged_unlike_linearmodels() {
    // Documented divergence (fixtures/panel_static.json `divergences`):
    // with 1980 missing for every EmplUK firm linearmodels differences
    // ACROSS the hole (751 rows) and lags its Driscoll-Kraay scores over the
    // present periods; tsecon uses calendar adjacency (611 rows) and
    // calendar lags, pinned against the NumPy transcription.
    let fx = load("panel_static.json");
    let dv = &fx["divergences"]["empluk_without_1980"];
    let mut raw = empluk();
    let hole = dv["dropped_period"].as_u64().unwrap() as usize;
    for row in raw.mask.iter_mut() {
        row[hole] = false;
    }
    let data = raw.panel();
    let fd = panel_first_difference(&data, false).unwrap();
    assert_eq!(fd.nobs, dv["fd_calendar_nobs"].as_u64().unwrap() as usize);
    assert_eq!(fd.nobs, 611);
    assert_ne!(
        fd.nobs,
        dv["fd_linearmodels_nobs"].as_u64().unwrap() as usize
    );
    close(
        &fd.params,
        &f64s(&dv["fd_calendar_params"]),
        1e-10,
        "calendar FD params",
    );
    let lm = f64s(&dv["fd_linearmodels_params"]);
    assert!(
        (fd.params[1] - lm[1]).abs() > 0.05,
        "linearmodels bridges the hole"
    );

    let re = panel_random_effects(&data, RandomEffectsConfig::default()).unwrap();
    close(
        &re.params,
        &f64s(&dv["re_params"]),
        1e-10,
        "RE params (no divergence)",
    );
    let dk = re
        .inference(StaticSeType::DriscollKraay { bandwidth: 4.0 })
        .unwrap();
    close(
        &dk.bse,
        &f64s(&dv["re_dk_calendar_bse"]),
        1e-10,
        "calendar DK bse",
    );
    let cal = matrix(&dv["re_dk_calendar_cov"]);
    for a in 0..3 {
        for b in 0..3 {
            let s = (cal[a][a] * cal[b][b]).sqrt();
            assert!((dk.cov[(a, b)] - cal[a][b]).abs() / s < 1e-10);
        }
    }
    let lm_bse = f64s(&dv["re_dk_linearmodels_bse"]);
    assert!(
        (dk.bse[0] - lm_bse[0]).abs() / lm_bse[0] > 0.05,
        "compressed lags differ"
    );
}

fn msg(e: PanelError) -> String {
    e.to_string()
}

#[test]
fn refusals_name_their_parameter() {
    let raw = seeded(10, 6, 12, false);
    let data = raw.panel();
    let empty = PanelData::balanced(raw.y.clone(), vec![]).unwrap();
    let no_const = RandomEffectsConfig {
        constant: false,
        small_sample: false,
    };
    // k = 0 without a constant
    assert!(msg(panel_random_effects(&empty, no_const).unwrap_err()).starts_with("regressors"));
    assert!(msg(panel_between(
        &empty,
        BetweenConfig {
            constant: false,
            reweight: false
        }
    )
    .unwrap_err())
    .starts_with("regressors"));
    assert!(msg(fama_macbeth(&empty, false).unwrap_err()).starts_with("regressors"));
    assert!(msg(panel_first_difference(&empty, true).unwrap_err()).starts_with("regressors"));
    // k = 0 with a constant is legal for RE / BE / FM
    panel_random_effects(&empty, RandomEffectsConfig::default()).unwrap();
    panel_between(&empty, BetweenConfig::default()).unwrap();
    fama_macbeth(&empty, true).unwrap();

    // constant=true plus a ones column
    let mut ones = raw.regs.clone();
    ones.push(("ones".into(), Mat::from_fn(raw.n(), raw.t(), |_, _| 1.0)));
    let dup = PanelData::balanced(raw.y.clone(), ones).unwrap();
    for e in [
        panel_random_effects(&dup, RandomEffectsConfig::default()).unwrap_err(),
        panel_between(&dup, BetweenConfig::default()).unwrap_err(),
        fama_macbeth(&dup, true).unwrap_err(),
    ] {
        let m = msg(e);
        assert!(
            m.starts_with("regressors") && m.contains("constant=True"),
            "{m}"
        );
    }
    // ... legal as the intercept with constant=false
    panel_random_effects(&dup, no_const).unwrap();

    // a collinear final design
    let mut twice = raw.regs.clone();
    twice.push(("x1_again".into(), raw.regs[0].1.clone()));
    let col = PanelData::balanced(raw.y.clone(), twice).unwrap();
    assert!(
        msg(panel_random_effects(&col, RandomEffectsConfig::default()).unwrap_err())
            .starts_with("regressors")
    );
    assert!(
        msg(panel_between(&col, BetweenConfig::default()).unwrap_err()).starts_with("regressors")
    );
    assert!(msg(panel_first_difference(&col, false).unwrap_err()).starts_with("regressors"));

    // a time-invariant regressor in first differences
    let mut z = raw.regs.clone();
    z.push((
        "z".into(),
        Mat::from_fn(raw.n(), raw.t(), |i, _| 0.1 + i as f64 / 7.0),
    ));
    let zd = PanelData::balanced(raw.y.clone(), z).unwrap();
    let m = msg(panel_first_difference(&zd, false).unwrap_err());
    assert!(m.starts_with("regressors") && m.contains("panel_re"), "{m}");
    // ... which random effects estimates
    panel_random_effects(&zd, RandomEffectsConfig::default()).unwrap();

    // small_sample on a singular between moment matrix (EmplUK + year dummies)
    let fx = load("panel_static.json");
    let yd = dataset(&fx, "empluk_yeardummies").panel();
    let e = panel_random_effects(
        &yd,
        RandomEffectsConfig {
            constant: true,
            small_sample: true,
        },
    )
    .unwrap_err();
    assert!(matches!(e, PanelError::RankDeficient { .. }));
    assert!(msg(e).starts_with("small_sample"));
    panel_random_effects(&yd, RandomEffectsConfig::default()).unwrap();

    // RE: N <= K, and n - K - N + 1 <= 0
    let tiny = seeded(3, 5, 13, false).panel();
    assert!(
        msg(panel_random_effects(&tiny, RandomEffectsConfig::default()).unwrap_err())
            .starts_with("regressors")
    );
    assert!(
        msg(panel_between(&tiny, BetweenConfig::default()).unwrap_err()).starts_with("regressors")
    );
    let one_period = seeded(10, 1, 14, false).panel();
    assert!(
        msg(panel_random_effects(&one_period, RandomEffectsConfig::default()).unwrap_err())
            .starts_with("outcome")
    );
    assert!(msg(panel_first_difference(&one_period, false).unwrap_err()).starts_with("outcome"));
    // RE: sigma2_eps == 0 (the outcome is exactly linear in x within entities)
    let exact = Mat::from_fn(raw.n(), raw.t(), |i, t| {
        i as f64 + 2.0 * raw.regs[0].1[(i, t)]
    });
    let ex = PanelData::balanced(exact, vec![raw.regs[0].clone()]).unwrap();
    assert!(
        msg(panel_random_effects(&ex, RandomEffectsConfig::default()).unwrap_err())
            .starts_with("outcome")
    );
    // FD: m <= K
    let short = seeded(1, 3, 15, false).panel();
    assert!(msg(panel_first_difference(&short, false).unwrap_err()).starts_with("outcome"));

    // BE: Driscoll-Kraay names se_type
    let be = panel_between(&data, BetweenConfig::default()).unwrap();
    assert!(msg(be
        .inference(StaticSeType::DriscollKraay { bandwidth: 4.0 })
        .unwrap_err())
    .contains("se_type"));

    // FM: fewer than 2 usable periods; bandwidth + 1 > T_v; bad bandwidths
    let fm_short = seeded(10, 1, 16, false).panel();
    assert!(msg(fama_macbeth(&fm_short, true).unwrap_err()).starts_with("regressors"));
    let fm = fama_macbeth(&data, true).unwrap();
    assert_eq!(fm.n_periods_used, 6);
    let m = msg(fm
        .inference(FamaMacBethSe::Hac { bandwidth: 5.5 })
        .unwrap_err());
    assert!(m.starts_with("bandwidth"), "{m}");
    fm.inference(FamaMacBethSe::Hac { bandwidth: 5.0 }).unwrap();
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let m = msg(fm
            .inference(FamaMacBethSe::Hac { bandwidth: bad })
            .unwrap_err());
        assert!(
            m.starts_with("bandwidth") && !m.contains("Driscoll-Kraay"),
            "{m}"
        );
    }
    // DK bandwidth on RE / FD
    let re = panel_random_effects(&data, RandomEffectsConfig::default()).unwrap();
    let m = msg(re
        .inference(StaticSeType::DriscollKraay { bandwidth: -1.0 })
        .unwrap_err());
    assert!(m.contains("bandwidth"), "{m}");
}

#[test]
fn a_masked_panel_runs_every_estimator_without_panicking_on_degenerate_inputs() {
    // One observed cell per entity: RE has no within degrees of freedom,
    // FD has no differences, FM has one entity per period.
    let n = 6;
    let raw = seeded(n, n, 17, false);
    let mask: Vec<Vec<bool>> = (0..n).map(|i| (0..n).map(|t| t == i).collect()).collect();
    let data = PanelData::unbalanced(raw.y.clone(), raw.regs.clone(), &mask).unwrap();
    assert!(panel_random_effects(&data, RandomEffectsConfig::default()).is_err());
    assert!(panel_first_difference(&data, false).is_err());
    assert!(fama_macbeth(&data, true).is_err());
    let be = panel_between(&data, BetweenConfig::default()).unwrap();
    assert_eq!(be.nobs, n);
}

#[test]
fn the_fama_macbeth_rank_boundary_is_the_numpy_criterion() {
    // Documented divergence: period 0's cross-section [1, x1, x1 + 1e-14 z]
    // is rank 3 for LAPACK gelsy (linearmodels keeps the period) and rank 2
    // under the numpy singular-value criterion (tsecon skips it); every
    // other period agrees.
    let fx = load("panel_static.json");
    let dv = &fx["divergences"]["fm_rank_boundary"];
    assert_eq!(dv["period0_gelsy_rank"], 3);
    assert_eq!(dv["period0_numpy_rank"], 2);
    let lm_used: Vec<bool> = dv["linearmodels_periods_used"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    let raw = dataset(&fx, "fm_rank_boundary");
    let fm = fama_macbeth(&raw.panel(), true).unwrap();
    assert!(lm_used[0] && !fm.periods_used[0]);
    assert_eq!(fm.periods_used[1..], lm_used[1..]);
}

#[test]
fn first_differences_use_the_calendar_not_linearmodels_first_appearance_axis() {
    // Documented: linearmodels 7.0 orders its time axis by first appearance,
    // so EmplUK in plm's entity-major order (whose first firm lacks 1976)
    // is differenced across non-adjacent years; tsecon reads the calendar
    // layout and is independent of any row order.
    let fx = load("panel_static.json");
    let dv = &fx["divergences"]["fd_first_appearance_time_axis"];
    let fd = panel_first_difference(&empluk().panel(), false).unwrap();
    assert_eq!(fd.nobs, dv["calendar_nobs"].as_u64().unwrap() as usize);
    assert_eq!(fd.nobs, 891);
    assert_eq!(dv["linearmodels_nobs"], 808);
    close(
        &fd.params,
        &f64s(&dv["calendar_params"]),
        1e-10,
        "calendar FD",
    );
    let lm = f64s(&dv["linearmodels_params"]);
    assert!((fd.params[0] - lm[0]).abs() > 0.05);
}

#[test]
fn an_overflowing_scale_is_refused_not_clamped_to_a_zero_standard_error() {
    // Data of magnitude 1e300 square past f64::MAX: the covariance would be
    // non-finite, and clamping it would print a zero standard error.
    let raw = seeded(10, 6, 18, false);
    let huge = PanelData::balanced(
        Mat::from_fn(raw.n(), raw.t(), |i, t| raw.y[(i, t)] * 1e300),
        raw.regs.clone(),
    )
    .unwrap();
    let starts = |r: Result<Vec<f64>, PanelError>| match r {
        Ok(bse) => panic!("expected a refusal, got bse {bse:?}"),
        Err(e) => assert!(e.to_string().starts_with("outcome"), "{e}"),
    };
    starts(
        panel_random_effects(&huge, RandomEffectsConfig::default())
            .and_then(|f| f.inference(StaticSeType::ClusterEntity).map(|i| i.bse)),
    );
    for se in [
        StaticSeType::Hc1,
        StaticSeType::DriscollKraay { bandwidth: 2.0 },
    ] {
        starts(panel_first_difference(&huge, true).and_then(|f| f.inference(se).map(|i| i.bse)));
        starts(
            panel_between(&huge, BetweenConfig::default())
                .and_then(|f| f.inference(StaticSeType::Hc1).map(|i| i.bse)),
        );
    }
    starts(
        fama_macbeth(&huge, true).and_then(|f| f.inference(FamaMacBethSe::Standard).map(|i| i.bse)),
    );
}

#[test]
fn a_regressor_or_residual_outside_the_double_range_is_refused_by_name() {
    // Overflow on the regressor side: the up-front scale check names
    // `regressors` instead of letting the overflow surface as a spurious
    // rank, SVD or time-invariance refusal.
    // So does underflow (magnitude 1e-200): its squares flush to zero.
    let raw = seeded(10, 6, 19, false);
    let names_regressors = |e: PanelError| {
        let m = msg(e);
        assert!(m.starts_with("regressors: column 0"), "{m}");
        assert!(m.contains("magnitude"), "{m}");
    };
    for scale in [1e200, 1e-200] {
        let mut regs = raw.regs.clone();
        regs[0].1 = Mat::from_fn(raw.n(), raw.t(), |i, t| raw.regs[0].1[(i, t)] * scale);
        let x = PanelData::balanced(raw.y.clone(), regs).unwrap();
        names_regressors(panel_random_effects(&x, RandomEffectsConfig::default()).unwrap_err());
        names_regressors(panel_between(&x, BetweenConfig::default()).unwrap_err());
        names_regressors(panel_first_difference(&x, false).unwrap_err());
        names_regressors(fama_macbeth(&x, true).unwrap_err());
    }
    // Underflow: an outcome of magnitude 1e-170 squares to zero, so every
    // variance estimate would be an exact 0 (and around 1e-160 a subnormal
    // number with lost digits) beside non-zero coefficients.
    let tiny = PanelData::balanced(
        Mat::from_fn(raw.n(), raw.t(), |i, t| raw.y[(i, t)] * 1e-170),
        raw.regs.clone(),
    )
    .unwrap();
    let names_outcome = |e: PanelError| {
        let m = msg(e);
        assert!(m.starts_with("outcome") && m.contains("underflow"), "{m}");
    };
    names_outcome(panel_random_effects(&tiny, RandomEffectsConfig::default()).unwrap_err());
    names_outcome(panel_between(&tiny, BetweenConfig::default()).unwrap_err());
    names_outcome(panel_first_difference(&tiny, false).unwrap_err());
    names_outcome(
        fama_macbeth(&tiny, true)
            .and_then(|f| f.inference(FamaMacBethSe::Standard))
            .unwrap_err(),
    );
}

#[test]
fn a_single_cluster_or_a_single_score_period_is_refused_not_reported_as_zero() {
    // One entity: its score sum is X'e = 0 by the normal equations, so the
    // clustered covariance would be identically zero.
    let raw = seeded(1, 8, 20, false);
    let one = raw.panel();
    let fd = panel_first_difference(&one, false).unwrap();
    let m = msg(fd.inference(StaticSeType::ClusterEntity).unwrap_err());
    assert!(m.starts_with("se_type") && m.contains("one entity"), "{m}");
    // The kernel and White sandwiches are still well defined there.
    for se in [
        StaticSeType::Hc1,
        StaticSeType::DriscollKraay { bandwidth: 2.0 },
    ] {
        assert!(fd.inference(se).unwrap().bse.iter().all(|&s| s > 1e-3));
    }
    // Two periods: every difference falls in period 1, so the
    // Driscoll-Kraay meat is the single period's X'e = 0.
    let raw = seeded(12, 2, 21, false);
    let fd = panel_first_difference(&raw.panel(), false).unwrap();
    let m = msg(fd
        .inference(StaticSeType::DriscollKraay { bandwidth: 4.0 })
        .unwrap_err());
    assert!(m.starts_with("se_type") && m.contains("one period"), "{m}");
    assert!(fd
        .inference(StaticSeType::ClusterEntity)
        .unwrap()
        .bse
        .iter()
        .all(|&s| s > 1e-3));
}

/// An outcome constant within every entity has no within variation, so
/// sigma2_eps = 0 and random effects is undefined. With entity means that
/// are not exact in floating point the demeaned outcome is rounding noise,
/// and the old guard (within residual against within variation) compared
/// noise with noise: the call was refused naming `regressors`, or, without
/// the constant, accepted with sigma2_eps ~ 1e-31 and theta = 1. Both are
/// now refused naming `outcome`, like an exactly constant outcome.
#[test]
fn an_outcome_constant_within_every_entity_is_refused_naming_outcome() {
    let raw = seeded(50, 30, 31, false);
    let mut rng = Lcg(77);
    let levels: Vec<f64> = (0..50).map(|_| rng.gaussian()).collect();
    for (label, y) in [
        ("exact ones", Mat::from_fn(50, 30, |_, _| 1.0)),
        (
            "integer levels",
            Mat::from_fn(50, 30, |i, _| (i % 7) as f64),
        ),
        ("random levels", Mat::from_fn(50, 30, |i, _| levels[i])),
        (
            "random levels x 1e6",
            Mat::from_fn(50, 30, |i, _| 1e6 * levels[i]),
        ),
    ] {
        let data = PanelData::unbalanced(y, raw.regs.clone(), &raw.mask).unwrap();
        for constant in [true, false] {
            let cfg = RandomEffectsConfig {
                constant,
                small_sample: false,
            };
            let m = msg(panel_random_effects(&data, cfg).unwrap_err());
            assert!(
                m.starts_with("outcome:") && m.contains("constant within every entity"),
                "{label}, constant = {constant}: {m}"
            );
        }
    }
    // An outcome that varies within entities, even by little against its
    // level, still fits; the exact-within-fit refusal keeps its wording.
    let tiny = Mat::from_fn(50, 30, |i, t| {
        1e3 * levels[i] + 1e-6 * raw.regs[0].1[(i, t)] + 1e-6 * ((i * 31 + t * 17) % 13) as f64
    });
    let data = PanelData::unbalanced(tiny, raw.regs.clone(), &raw.mask).unwrap();
    assert!(panel_random_effects(&data, RandomEffectsConfig::default()).is_ok());
    let exact = Mat::from_fn(50, 30, |i, t| levels[i] + 0.5 * raw.regs[0].1[(i, t)]);
    let data = PanelData::unbalanced(exact, raw.regs.clone(), &raw.mask).unwrap();
    let m = msg(panel_random_effects(&data, RandomEffectsConfig::default()).unwrap_err());
    assert!(m.starts_with("outcome: the within residuals vanish"), "{m}");
}
