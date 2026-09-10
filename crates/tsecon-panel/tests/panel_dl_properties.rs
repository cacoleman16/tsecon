//! Property / behavioural tests for `panel_distributed_lag`: the seeded
//! Monte Carlo the linearmodels golden cannot prove (recovery of the
//! long-run impact and the measured coverage of the cumulative-effect
//! interval), the bit-identity with `panel_ols_fe` at `L = 0`,
//! relabelling invariance, determinism, and the refusals.
//!
//! Measured coverage table (the numbers the assertions below pin; the
//! same table is reported on the panel model card). 500 replications per
//! cell, seed 20260910 + cell index, `L = 2`, true lag polynomial
//! `beta = (1.0, -0.5, 0.2)` so the long-run impact is `0.7`; entity and
//! time effects swept out; nominal 95% normal intervals on the
//! cumulative effect.
//!
//! * Entity-clustered cells (T = 30): AR(1) errors with rho = 0.5 and
//!   entity-specific scales in [0.5, 1.5], regressor AR(1) around an
//!   entity mean, no cross-sectional dependence.
//! * Driscoll-Kraay cells (N = 25, bandwidth 4): a common AR(1) weather
//!   component in the regressor with heterogeneous loadings, and a
//!   common AR(1) factor in the errors with heterogeneous loadings (so
//!   the time effects absorb only the mean loading), plus AR(1)
//!   idiosyncratic noise.
//!
//! ```text
//!  covariance        cell            coverage   mean estimate   mean se
//!  cluster (entity)  N =  50, T = 30   0.928        0.7000       0.0642
//!  cluster (entity)  N = 200, T = 30   0.938        0.6998       0.0327
//!  Driscoll-Kraay    N = 25, T =  50   0.886        0.7020       0.0611
//!  Driscoll-Kraay    N = 25, T = 200   0.924        0.6978       0.0335
//! ```
//!
//! Reading it honestly: the point estimate is unbiased in every cell
//! (the mean sits on 0.70 to Monte Carlo precision), and the intervals
//! run 1-2 points short of nominal under clustering (the usual
//! finite-cluster shortfall, shrinking from N = 50 to N = 200) and
//! 6 points short for Driscoll-Kraay at T = 50 — the short-T kernel
//! caveat the module docs carry — recovering to 0.92 at T = 200. The
//! pinned lower bounds sit two Monte Carlo standard errors
//! (`sqrt(0.95 * 0.05 / 500) ~ 0.01`) below the measured values, so a
//! regression of the interval construction fails while sampling noise
//! does not.

use tsecon_linalg::faer::Mat;
use tsecon_panel::{
    panel_distributed_lag, panel_ols_fe, DistributedLagConfig, FixedEffects, PanelData, PanelError,
    PanelSeType,
};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

const SEED: u64 = 20260910;
const BETA: [f64; 3] = [1.0, -0.5, 0.2];
const TRUE_CUM: f64 = 0.7;
const REPS: usize = 500;

fn gaussian(stream: &mut Stream) -> f64 {
    let u = stream.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// AR(1) path of length `n` at stationarity via a 30-period burn-in.
fn ar1(stream: &mut Stream, n: usize, rho: f64) -> Vec<f64> {
    let burn = 30;
    let mut v = 0.0;
    let mut out = Vec::with_capacity(n);
    for t in 0..n + burn {
        v = rho * v + gaussian(stream);
        if t >= burn {
            out.push(v);
        }
    }
    out
}

/// Simulates `y_it = alpha_i + delta_t + sum_l beta_l x_{i,t-l} + e_it`
/// on `t_len` periods (the first two are only ever used as lags).
/// `cross_sectional` switches on the common weather component and the
/// common error factor of the Driscoll-Kraay cells.
fn simulate(stream: &mut Stream, n_ent: usize, t_len: usize, cross_sectional: bool) -> PanelData {
    let l = BETA.len() - 1;
    let alpha: Vec<f64> = (0..n_ent).map(|_| gaussian(stream)).collect();
    let delta: Vec<f64> = (0..t_len).map(|_| 0.5 * gaussian(stream)).collect();
    let w = if cross_sectional {
        ar1(stream, t_len, 0.5)
    } else {
        vec![0.0; t_len]
    };
    let f = if cross_sectional {
        ar1(stream, t_len, 0.5)
    } else {
        vec![0.0; t_len]
    };
    let mut x = vec![vec![0.0_f64; t_len]; n_ent];
    let mut y = vec![vec![0.0_f64; t_len]; n_ent];
    for i in 0..n_ent {
        let mu = 20.0 + 4.0 * gaussian(stream);
        let kappa = 1.0 + 0.5 * gaussian(stream);
        let lambda = gaussian(stream);
        let sigma = 0.5 + stream.uniform_f64();
        let v = ar1(stream, t_len, 0.3);
        let u = ar1(stream, t_len, if cross_sectional { 0.3 } else { 0.5 });
        for t in 0..t_len {
            x[i][t] = mu + kappa * w[t] + v[t];
        }
        for t in 0..t_len {
            let mut resp = 0.0;
            for (lag, b) in BETA.iter().enumerate() {
                if t >= lag {
                    resp += b * x[i][t - lag];
                }
            }
            let e = if cross_sectional {
                lambda * f[t] + u[t]
            } else {
                sigma * u[t]
            };
            y[i][t] = alpha[i] + delta[t] + resp + e;
        }
    }
    let _ = l;
    let outcome = Mat::from_fn(n_ent, t_len, |i, t| y[i][t]);
    let reg = Mat::from_fn(n_ent, t_len, |i, t| x[i][t]);
    PanelData::balanced(outcome, vec![("temp".to_string(), reg)]).expect("balanced panel")
}

fn cfg(se_type: PanelSeType) -> DistributedLagConfig {
    DistributedLagConfig::new(2, se_type)
}

/// One coverage cell: returns (coverage, mean estimate, mean se).
fn coverage_cell(cell: u64, n_ent: usize, t_len: usize, se_type: PanelSeType) -> (f64, f64, f64) {
    let cross = matches!(se_type, PanelSeType::DriscollKraay { .. });
    let mut stream = Stream::new(SEED + cell);
    let (mut covered, mut sum_est, mut sum_se) = (0usize, 0.0, 0.0);
    for _ in 0..REPS {
        let data = simulate(&mut stream, n_ent, t_len, cross);
        let res = panel_distributed_lag(&data, &cfg(se_type)).expect("fit");
        let (lo, hi) = (res.cumulative_ci_low[0][0], res.cumulative_ci_high[0][0]);
        if lo <= TRUE_CUM && TRUE_CUM <= hi {
            covered += 1;
        }
        sum_est += res.cumulative_effect[0][0];
        sum_se += res.cumulative_se[0][0];
    }
    let r = REPS as f64;
    (covered as f64 / r, sum_est / r, sum_se / r)
}

#[test]
fn cumulative_effect_recovers_the_long_run_impact() {
    // A large panel: the cumulative effect must sit within a tight
    // absolute band of the true long-run impact and within 3 SE of it.
    let mut stream = Stream::new(SEED);
    let data = simulate(&mut stream, 100, 60, false);
    let res = panel_distributed_lag(&data, &cfg(PanelSeType::ClusterEntity)).expect("fit");
    let cum = res.cumulative_effect[0][0];
    let se = res.cumulative_se[0][0];
    assert!(
        (cum - TRUE_CUM).abs() < 0.05,
        "cumulative {cum} vs true {TRUE_CUM}"
    );
    assert!(
        (cum - TRUE_CUM).abs() < 3.0 * se,
        "cumulative {cum} +/- {se} misses {TRUE_CUM}"
    );
    for (l, b) in BETA.iter().enumerate() {
        let got = res.lag_effects[0][0][l];
        assert!((got - b).abs() < 0.05, "lag {l}: {got} vs {b}");
    }
    assert_eq!(res.n_periods_used, 58);
    assert_eq!(res.nobs, 100 * 58);
    assert_eq!(res.names, ["temp_L0", "temp_L1", "temp_L2"]);
}

#[test]
fn coverage_entity_cluster_n50_n200() {
    let (c50, m50, s50) = coverage_cell(1, 50, 30, PanelSeType::ClusterEntity);
    let (c200, m200, s200) = coverage_cell(2, 200, 30, PanelSeType::ClusterEntity);
    eprintln!("cluster N=50  T=30: coverage {c50:.3} mean {m50:.4} mean se {s50:.4}");
    eprintln!("cluster N=200 T=30: coverage {c200:.3} mean {m200:.4} mean se {s200:.4}");
    // Measured on this seed: 0.928 and 0.938 (module docs). Bounds sit
    // two MC standard errors below the measured values.
    assert!(c50 >= 0.90, "N=50 cluster coverage {c50}");
    assert!(c200 >= 0.92, "N=200 cluster coverage {c200}");
    assert!((m50 - TRUE_CUM).abs() < 0.02 && (m200 - TRUE_CUM).abs() < 0.01);
}

#[test]
fn coverage_driscoll_kraay_t50_t200() {
    let dk = PanelSeType::DriscollKraay { bandwidth: 4.0 };
    let (c50, m50, s50) = coverage_cell(3, 25, 50, dk);
    let (c200, m200, s200) = coverage_cell(4, 25, 200, dk);
    eprintln!("dk N=25 T=50 : coverage {c50:.3} mean {m50:.4} mean se {s50:.4}");
    eprintln!("dk N=25 T=200: coverage {c200:.3} mean {m200:.4} mean se {s200:.4}");
    // Measured on this seed: 0.886 and 0.924 (module docs).
    assert!(c50 >= 0.86, "T=50 DK coverage {c50}");
    assert!(c200 >= 0.90, "T=200 DK coverage {c200}");
    assert!((m50 - TRUE_CUM).abs() < 0.03 && (m200 - TRUE_CUM).abs() < 0.015);
}

#[test]
fn lag_zero_entity_effects_reproduces_panel_fe_bit_identically() {
    let mut stream = Stream::new(SEED + 10);
    let data = simulate(&mut stream, 12, 24, false);
    let fe = panel_ols_fe(&data).expect("fe");
    for se_type in [
        PanelSeType::NonRobust,
        PanelSeType::ClusterEntity,
        PanelSeType::DriscollKraay { bandwidth: 4.0 },
    ] {
        let inf = fe.inference(se_type).expect("inference");
        let mut c = cfg(se_type);
        c.lags = 0;
        c.effects = FixedEffects::ENTITY;
        let dl = panel_distributed_lag(&data, &c).expect("dl");
        assert_eq!(dl.params, fe.params, "{se_type:?}: params");
        assert_eq!(dl.bse, inf.bse, "{se_type:?}: bse");
        assert_eq!(dl.tvalues, inf.tvalues, "{se_type:?}: tvalues");
        assert_eq!(dl.cov[(0, 0)], inf.cov[(0, 0)], "{se_type:?}: cov");
        assert_eq!(dl.nobs, fe.nobs);
        assert_eq!(dl.df_resid, fe.df_resid);
        assert_eq!(dl.cumulative_effect[0][0], fe.params[0]);
        assert_eq!(dl.cumulative_se[0][0], inf.bse[0]);
        assert_eq!(dl.names, ["temp_L0"]);
    }
}

fn permute_rows(data: &PanelData, perm: &[usize]) -> PanelData {
    let (n, t) = (data.n_entities(), data.n_periods());
    let y = data.outcome();
    let x = data.regressor(0).unwrap();
    PanelData::balanced(
        Mat::from_fn(n, t, |i, tt| y[(perm[i], tt)]),
        vec![(
            "temp".to_string(),
            Mat::from_fn(n, t, |i, tt| x[(perm[i], tt)]),
        )],
    )
    .unwrap()
}

fn permute_cols(data: &PanelData, perm: &[usize]) -> PanelData {
    let (n, t) = (data.n_entities(), data.n_periods());
    let y = data.outcome();
    let x = data.regressor(0).unwrap();
    PanelData::balanced(
        Mat::from_fn(n, t, |i, tt| y[(i, perm[tt])]),
        vec![(
            "temp".to_string(),
            Mat::from_fn(n, t, |i, tt| x[(i, perm[tt])]),
        )],
    )
    .unwrap()
}

fn assert_vec_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len());
    for (u, v) in a.iter().zip(b) {
        assert!(
            (u - v).abs() <= tol * v.abs().max(1e-12),
            "{what}: {u} vs {v}"
        );
    }
}

#[test]
fn invariant_to_entity_relabelling() {
    let mut stream = Stream::new(SEED + 11);
    let data = simulate(&mut stream, 15, 30, true);
    let perm: Vec<usize> = (0..15).rev().collect();
    let shuffled = permute_rows(&data, &perm);
    for se_type in [
        PanelSeType::NonRobust,
        PanelSeType::ClusterEntity,
        PanelSeType::DriscollKraay { bandwidth: 3.0 },
    ] {
        let mut c = cfg(se_type);
        c.effects = FixedEffects {
            entity: true,
            time: true,
            entity_trends: true,
        };
        let a = panel_distributed_lag(&data, &c).unwrap();
        let b = panel_distributed_lag(&shuffled, &c).unwrap();
        assert_vec_close(&a.params, &b.params, 1e-10, "params");
        assert_vec_close(&a.bse, &b.bse, 1e-10, "bse");
        assert_vec_close(&a.cumulative_se[0], &b.cumulative_se[0], 1e-10, "cum se");
    }
}

#[test]
fn invariant_to_time_relabelling_at_lag_zero() {
    // With L = 0 no lag structure exists, so permuting the periods must
    // leave the two-way fit unchanged (cluster and nonrobust SEs are
    // permutation-invariant in time; Driscoll-Kraay is not, by design).
    let mut stream = Stream::new(SEED + 12);
    let data = simulate(&mut stream, 15, 30, true);
    let perm: Vec<usize> = (0..30).map(|t| (t * 7) % 30).collect();
    let shuffled = permute_cols(&data, &perm);
    for se_type in [PanelSeType::NonRobust, PanelSeType::ClusterEntity] {
        let mut c = cfg(se_type);
        c.lags = 0;
        let a = panel_distributed_lag(&data, &c).unwrap();
        let b = panel_distributed_lag(&shuffled, &c).unwrap();
        assert_vec_close(&a.params, &b.params, 1e-10, "params");
        assert_vec_close(&a.bse, &b.bse, 1e-10, "bse");
    }
}

#[test]
fn deterministic_across_calls() {
    let mut stream = Stream::new(SEED + 13);
    let data = simulate(&mut stream, 10, 25, true);
    let mut c = cfg(PanelSeType::DriscollKraay { bandwidth: 4.0 });
    c.powers = 2;
    c.eval_points = Some(vec![15.0, 20.0, 25.0]);
    let a = panel_distributed_lag(&data, &c).unwrap();
    let b = panel_distributed_lag(&data, &c).unwrap();
    assert_eq!(a.params, b.params);
    assert_eq!(a.bse, b.bse);
    assert_eq!(a.cumulative_effect, b.cumulative_effect);
    assert_eq!(a.marginal_effect, b.marginal_effect);
    assert_eq!(a.marginal_se, b.marginal_se);
    assert_eq!(a.turning_point, b.turning_point);
    assert_eq!(a.turning_point_se, b.turning_point_se);
    // The quadratic block is consistent with its own pieces.
    let b1 = a.cumulative_effect[0][0];
    let b2 = a.cumulative_effect[0][1];
    let tp = a.turning_point.as_ref().unwrap()[0];
    assert!((tp - (-b1 / (2.0 * b2))).abs() < 1e-12);
    let me = a.marginal_effect.as_ref().unwrap();
    assert!((me[0][1] - (b1 + 2.0 * b2 * 20.0)).abs() < 1e-12);
    // The marginal effect vanishes at the turning point.
    let mut c2 = c.clone();
    c2.eval_points = Some(vec![tp]);
    let at_tp = panel_distributed_lag(&data, &c2).unwrap();
    assert!(at_tp.marginal_effect.as_ref().unwrap()[0][0].abs() < 1e-10);
}

#[test]
fn refusals_teach() {
    let mut stream = Stream::new(SEED + 14);
    let data = simulate(&mut stream, 8, 12, false);
    let ok = cfg(PanelSeType::ClusterEntity);

    // Too many lags: fewer than two periods would remain.
    let mut c = ok.clone();
    c.lags = 11;
    let err = panel_distributed_lag(&data, &c).unwrap_err();
    assert!(
        matches!(
            err,
            PanelError::InsufficientObservations {
                needed: 13,
                got: 12,
                ..
            }
        ),
        "{err}"
    );
    // L = T - 2 passes the period guard (two periods remain) and the
    // degrees-of-freedom guard then takes over: 11 lag columns plus
    // N + 2 - 1 = 9 absorbed effects exceed the n = 16 rows.
    c.lags = 10;
    assert!(matches!(
        panel_distributed_lag(&data, &c).unwrap_err(),
        PanelError::DegreesOfFreedomAbsorbed {
            n: 16,
            k: 11,
            n_absorbed: 9
        }
    ));
    c.lags = 2;
    assert!(
        panel_distributed_lag(&data, &c).is_ok(),
        "L = 2 on 8 x 12 fits"
    );
    c.lags = usize::MAX - 1;
    assert!(matches!(
        panel_distributed_lag(&data, &c).unwrap_err(),
        PanelError::InsufficientObservations { .. }
    ));

    // powers outside {1, 2}.
    let mut c = ok.clone();
    c.powers = 3;
    let err = panel_distributed_lag(&data, &c).unwrap_err();
    assert!(
        matches!(err, PanelError::InvalidArgument { .. }) && err.to_string().contains("powers")
    );

    // eval_points under powers = 1 is inert and refused.
    let mut c = ok.clone();
    c.eval_points = Some(vec![20.0]);
    let err = panel_distributed_lag(&data, &c).unwrap_err();
    assert!(err.to_string().contains("eval_points"), "{err}");

    // Empty / non-finite eval points under powers = 2.
    let mut c = ok.clone();
    c.powers = 2;
    c.eval_points = Some(vec![]);
    assert!(matches!(
        panel_distributed_lag(&data, &c).unwrap_err(),
        PanelError::InvalidArgument { .. }
    ));
    c.eval_points = Some(vec![f64::NAN]);
    assert!(matches!(
        panel_distributed_lag(&data, &c).unwrap_err(),
        PanelError::NonFinite { .. }
    ));

    // No effects at all, and trends without entity effects.
    let mut c = ok.clone();
    c.effects = FixedEffects {
        entity: false,
        time: false,
        entity_trends: false,
    };
    let err = panel_distributed_lag(&data, &c).unwrap_err();
    assert!(err.to_string().contains("no fixed effects"), "{err}");
    c.effects = FixedEffects {
        entity: false,
        time: true,
        entity_trends: true,
    };
    let err = panel_distributed_lag(&data, &c).unwrap_err();
    assert!(err.to_string().contains("entity_trends"), "{err}");

    // A common (entity-invariant) regressor is absorbed by time effects.
    let common = Mat::from_fn(8, 12, |_, t| (t as f64).sin());
    let absorbed = PanelData::balanced(
        Mat::from_fn(8, 12, |i, t| data.outcome()[(i, t)]),
        vec![("common".to_string(), common)],
    )
    .unwrap();
    let err = panel_distributed_lag(&absorbed, &ok).unwrap_err();
    assert!(matches!(err, PanelError::SingularDesign { .. }), "{err}");

    // NaN input is refused at the data boundary, as everywhere.
    let bad = PanelData::balanced(
        Mat::from_fn(4, 6, |i, t| if i == 1 && t == 2 { f64::NAN } else { 1.0 }),
        vec![],
    );
    assert!(matches!(bad.unwrap_err(), PanelError::NonFinite { .. }));

    // No regressors.
    let none = PanelData::balanced(Mat::from_fn(4, 6, |i, t| (i * 6 + t) as f64), vec![]).unwrap();
    assert!(matches!(
        panel_distributed_lag(&none, &ok).unwrap_err(),
        PanelError::InvalidArgument { .. }
    ));

    // Degrees of freedom under the general menu: a 2 x 3 panel with
    // two-way effects and one lag has n = 4 <= k + n_absorbed.
    let tiny = PanelData::balanced(
        Mat::from_fn(2, 3, |i, t| (i + t) as f64),
        vec![(
            "x".to_string(),
            Mat::from_fn(2, 3, |i, t| ((i * 3 + t) as f64).sqrt()),
        )],
    )
    .unwrap();
    let mut c = ok.clone();
    c.lags = 1;
    assert!(matches!(
        panel_distributed_lag(&tiny, &c).unwrap_err(),
        PanelError::DegreesOfFreedomAbsorbed { .. }
    ));
}
