//! Property / invariant tests for the structured-penalty and post-selection
//! slice, beyond the goldens in `structured_golden.rs`:
//!
//! * the group solver reduces exactly to the crate's (scikit-learn-pinned)
//!   `lasso` at `l1_ratio = 1`, with singleton groups at `l1_ratio = 0`,
//!   and — with custom singleton weights — to a column-rescaled `lasso`;
//! * `alpha_max` zeroes the fit just above and not just below, for every
//!   `l1_ratio` regime;
//! * the solver is exactly scale-equivariant (`X -> sX`, `alpha -> s
//!   alpha` gives `coef / s`) across many decades, which is what the
//!   dimensionless stopping rule buys;
//! * the `converged` honesty flag fires, and the returned KKT residual is
//!   an honest measurement when it does;
//! * relabeling groups or permuting columns does not change the answer;
//! * every teaching error fires with the promised category and wording;
//! * post-LASSO's refit satisfies the OLS normal equations on the support;
//! * **post-double-selection coverage, measured**: on a seeded design with
//!   autocorrelated errors and confounders that load strongly on the
//!   treatment but weakly on the outcome, the single-selection interval
//!   undercovers badly while the PDS interval tracks the infeasible oracle
//!   — both cells' numbers are what the model card quotes.

mod common;

use common::{as_f64_vec, as_mat, load_fixture, mat_from_cols, Lcg};
use tsecon_ml::faer::Mat;
use tsecon_ml::{
    group_lasso, group_lasso_alpha_max, lasso, pds_lasso, post_lasso, regularization_path,
    CoordDescentOptions, GroupWeights, MlError, PathOptions, PdsAlpha,
};

const TIGHT: CoordDescentOptions = CoordDescentOptions {
    tol: 1e-11,
    max_iter: 100_000,
};

/// The binding's default stopping controls.
const DEFAULT: CoordDescentOptions = CoordDescentOptions {
    tol: 1e-8,
    max_iter: 10_000,
};

fn blocks_design() -> (Mat<f64>, Vec<f64>, Vec<i64>) {
    let fx = load_fixture("structured.json");
    let d = &fx["group_lasso"]["designs"]["blocks"];
    let groups = d["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    (as_mat(&d["X"]), as_f64_vec(&d["y"]), groups)
}

/// A seeded `n x p` design (column-major) with mildly correlated columns
/// and a sparse target.
fn seeded_design(seed: u64, n: usize, p: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let mut cols: Vec<Vec<f64>> = (0..p)
        .map(|_| (0..n).map(|_| rng.normal()).collect())
        .collect();
    for j in 1..p {
        let (prev, cur) = cols.split_at_mut(j);
        for (c, &pv) in cur[0].iter_mut().zip(&prev[j - 1]) {
            *c += 0.4 * pv;
        }
    }
    let y: Vec<f64> = (0..n)
        .map(|i| 1.2 * cols[0][i] - 0.8 * cols[1][i] + 0.5 * cols[3][i] + 0.7 * rng.normal())
        .collect();
    (cols, y)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// `l1_ratio = 1` drops the group term, so the group solver must reproduce
/// `lasso` at 1e-8 whatever the grouping or weights — the reduction that
/// pins the objective's `1/(2n)` scaling to scikit-learn's transitively.
#[test]
fn l1_ratio_one_reproduces_lasso() {
    let (x, y, groups) = blocks_design();
    let scattered: Vec<i64> = (0..groups.len() as i64).map(|j| (j * 7) % 5 - 2).collect();
    let mut worst = 0.0f64;
    for &alpha in &[0.02, 0.05, 0.1, 0.3] {
        let base = lasso(x.as_ref(), &y, alpha, TIGHT).unwrap();
        for g in [&groups, &scattered] {
            let mut distinct = g.clone();
            distinct.sort_unstable();
            distinct.dedup();
            for w in [
                GroupWeights::SqrtSize,
                GroupWeights::Uniform,
                GroupWeights::Custom(vec![2.0; distinct.len()]),
            ] {
                let fit = group_lasso(x.as_ref(), &y, g, alpha, 1.0, &w, TIGHT).unwrap();
                assert!(fit.converged);
                let d = max_abs_diff(&fit.coef, &base.coef);
                assert!(d <= 1e-8, "alpha={alpha} {w:?}: max diff {d:e}");
                worst = worst.max(d);
            }
        }
    }
    println!("l1_ratio=1 reduction achieved max abs diff vs lasso: {worst:e}");
}

/// Singleton groups with `l1_ratio = 0` and unit weights are the LASSO
/// (the group norm of a scalar is its absolute value), at 1e-8; with custom
/// singleton weights `w_j` the solution equals the LASSO on the rescaled
/// columns `x_j / w_j` with coefficients divided by `w_j` — an exact
/// identity of the objective, so it pins the weight convention.
#[test]
fn singleton_groups_reproduce_lasso() {
    let (x, y, _) = blocks_design();
    let p = x.ncols();
    // Non-contiguous, non-consecutive labels: any distinct integers work.
    let labels: Vec<i64> = (0..p as i64).map(|j| 100 - 13 * j).collect();
    let mut worst = 0.0f64;
    for &alpha in &[0.03, 0.1, 0.25] {
        let base = lasso(x.as_ref(), &y, alpha, TIGHT).unwrap();
        for w in [GroupWeights::SqrtSize, GroupWeights::Uniform] {
            let fit = group_lasso(x.as_ref(), &y, &labels, alpha, 0.0, &w, TIGHT).unwrap();
            let d = max_abs_diff(&fit.coef, &base.coef);
            assert!(d <= 1e-8, "alpha={alpha} {w:?}: max diff {d:e}");
            worst = worst.max(d);
        }
        // Custom weights in ascending-label order. Labels descend with j,
        // so ascending label order is j = p-1, ..., 0.
        let w_by_j: Vec<f64> = (0..p).map(|j| 0.5 + 0.25 * j as f64).collect();
        let mut w_ascending: Vec<f64> = w_by_j.clone();
        w_ascending.reverse();
        let fit = group_lasso(
            x.as_ref(),
            &y,
            &labels,
            alpha,
            0.0,
            &GroupWeights::Custom(w_ascending),
            TIGHT,
        )
        .unwrap();
        let cols: Vec<Vec<f64>> = (0..p)
            .map(|j| (0..x.nrows()).map(|i| x[(i, j)] / w_by_j[j]).collect())
            .collect();
        let xr = mat_from_cols(&cols);
        let rescaled = lasso(xr.as_ref(), &y, alpha, TIGHT).unwrap();
        for (j, &wj) in w_by_j.iter().enumerate() {
            let d = (fit.coef[j] - rescaled.coef[j] / wj).abs();
            assert!(
                d <= 1e-8,
                "alpha={alpha} weighted singleton {j}: diff {d:e}"
            );
            worst = worst.max(d);
        }
    }
    println!("singleton reduction achieved max abs diff vs lasso: {worst:e}");
}

/// `alpha_max` is the boundary of the all-zero solution in every regime:
/// pure group (`l1_ratio = 0`), mixed, and pure `L1`.
#[test]
fn alpha_max_is_the_zero_boundary() {
    let (cols, y) = seeded_design(41, 120, 9);
    let x = mat_from_cols(&cols);
    let groups = [0, 0, 0, 1, 1, 2, 2, 2, 2];
    for &l1 in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        for w in [
            GroupWeights::SqrtSize,
            GroupWeights::Uniform,
            GroupWeights::Custom(vec![0.7, 1.9, 1.1]),
        ] {
            let am = group_lasso_alpha_max(x.as_ref(), &y, &groups, l1, &w).unwrap();
            assert!(am.is_finite() && am > 0.0);
            let above =
                group_lasso(x.as_ref(), &y, &groups, am * (1.0 + 1e-9), l1, &w, TIGHT).unwrap();
            assert!(
                above.coef.iter().all(|&b| b == 0.0),
                "l1={l1} {w:?}: not zero above"
            );
            assert_eq!(above.n_iter, 1, "zero solution is found in one sweep");
            assert!(above.converged && above.kkt_violation == 0.0);
            let below =
                group_lasso(x.as_ref(), &y, &groups, am * (1.0 - 1e-3), l1, &w, TIGHT).unwrap();
            assert!(
                below.coef.iter().any(|&b| b != 0.0),
                "l1={l1} {w:?}: zero below"
            );
            assert_eq!(below.alpha_max, am);
        }
    }
}

/// `X -> sX` with `alpha -> s alpha` leaves both penalty terms and the
/// data fit identical at `b / s`, so `coef * s` is an algebraic invariant.
/// Swept over sixteen decades at both shipped tolerances: an absolute
/// stopping rule would silently return the first prox step at large `s`.
#[test]
fn group_lasso_is_scale_equivariant() {
    let (cols, y) = seeded_design(7, 200, 8);
    let groups = [0, 0, 1, 1, 1, 2, 2, 3];
    let base_x = mat_from_cols(&cols);
    for opts in [TIGHT, DEFAULT] {
        for &(alpha1, l1) in &[(0.08, 0.0), (0.05, 0.5)] {
            let base = group_lasso(
                base_x.as_ref(),
                &y,
                &groups,
                alpha1,
                l1,
                &GroupWeights::SqrtSize,
                opts,
            )
            .unwrap();
            let mag = base.coef.iter().fold(0.0f64, |m, &b| m.max(b.abs()));
            assert!(mag > 0.5, "degenerate baseline");
            for &s in &[1e-8, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e8] {
                let scaled: Vec<Vec<f64>> = cols
                    .iter()
                    .map(|c| c.iter().map(|v| v * s).collect())
                    .collect();
                let x = mat_from_cols(&scaled);
                let fit = group_lasso(
                    x.as_ref(),
                    &y,
                    &groups,
                    alpha1 * s,
                    l1,
                    &GroupWeights::SqrtSize,
                    opts,
                )
                .unwrap();
                assert!(fit.converged);
                assert!(fit.max_rel_change <= opts.tol);
                for j in 0..base.coef.len() {
                    let d = (fit.coef[j] * s - base.coef[j]).abs();
                    assert!(
                        d <= 1e-7 * mag,
                        "tol={:e} l1={l1} s={s:e} coord {j}: {} vs {} (diff {d:e})",
                        opts.tol,
                        fit.coef[j] * s,
                        base.coef[j]
                    );
                }
            }
        }
    }
}

/// The honesty flag: with a sweep budget of one, a problem that needs more
/// returns `converged = false`, the last iterate, and a KKT residual that
/// is genuinely above the certified level (and agrees with an independent
/// re-evaluation through a warm-started continuation: running the solver
/// to convergence from scratch lowers the objective).
#[test]
fn converged_false_fires_and_is_honest() {
    let (cols, y) = seeded_design(3, 150, 10);
    let x = mat_from_cols(&cols);
    let groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
    let one = CoordDescentOptions {
        tol: 1e-11,
        max_iter: 1,
    };
    let cut = group_lasso(
        x.as_ref(),
        &y,
        &groups,
        0.02,
        0.5,
        &GroupWeights::SqrtSize,
        one,
    )
    .unwrap();
    assert!(!cut.converged, "one sweep must not certify this problem");
    assert_eq!(cut.n_iter, 1);
    assert!(cut.coef.iter().all(|b| b.is_finite()));
    let full = group_lasso(
        x.as_ref(),
        &y,
        &groups,
        0.02,
        0.5,
        &GroupWeights::SqrtSize,
        TIGHT,
    )
    .unwrap();
    assert!(full.converged);
    assert!(
        cut.kkt_violation > full.kkt_violation * 100.0,
        "truncated run's KKT residual {:e} is not honestly larger than the converged {:e}",
        cut.kkt_violation,
        full.kkt_violation
    );
    assert!(cut.objective >= full.objective);
}

/// Group labels are names, not positions: relabeling them (and permuting
/// the columns consistently) leaves the fit unchanged.
#[test]
fn relabeling_and_permuting_columns_is_invariant() {
    let (cols, y) = seeded_design(11, 100, 8);
    let x = mat_from_cols(&cols);
    let groups: [i64; 8] = [0, 0, 1, 1, 1, 2, 2, 3];
    let base = group_lasso(
        x.as_ref(),
        &y,
        &groups,
        0.05,
        0.3,
        &GroupWeights::SqrtSize,
        TIGHT,
    )
    .unwrap();
    // Relabel: 0 -> 42, 1 -> -5, 2 -> 7, 3 -> 1000 (order changes too).
    let relabeled: Vec<i64> = groups
        .iter()
        .map(|&g| match g {
            0 => 42,
            1 => -5,
            2 => 7,
            _ => 1000,
        })
        .collect();
    let fit = group_lasso(
        x.as_ref(),
        &y,
        &relabeled,
        0.05,
        0.3,
        &GroupWeights::SqrtSize,
        TIGHT,
    )
    .unwrap();
    assert!(max_abs_diff(&fit.coef, &base.coef) <= 1e-12);
    assert_eq!(fit.active_groups.len(), base.active_groups.len());
    // Permute columns: reverse order.
    let perm: Vec<usize> = (0..8).rev().collect();
    let pcols: Vec<Vec<f64>> = perm.iter().map(|&j| cols[j].clone()).collect();
    let pgroups: Vec<i64> = perm.iter().map(|&j| groups[j]).collect();
    let px = mat_from_cols(&pcols);
    let pfit = group_lasso(
        px.as_ref(),
        &y,
        &pgroups,
        0.05,
        0.3,
        &GroupWeights::SqrtSize,
        TIGHT,
    )
    .unwrap();
    for (k, &j) in perm.iter().enumerate() {
        assert!((pfit.coef[k] - base.coef[j]).abs() <= 1e-9);
    }
    assert_eq!(pfit.active_groups, base.active_groups);
}

#[test]
fn teaching_errors_fire_with_the_promised_wording() {
    let (cols, y) = seeded_design(5, 60, 6);
    let x = mat_from_cols(&cols);
    let groups = [0, 0, 1, 1, 2, 2];
    let sq = GroupWeights::SqrtSize;

    // groups length names both sizes.
    let e = group_lasso(x.as_ref(), &y, &[0, 0, 1], 0.1, 0.0, &sq, TIGHT).unwrap_err();
    assert_eq!(
        e,
        MlError::DimensionMismatch {
            what: "groups must carry one integer label per column of x",
            expected: 6,
            got: 3
        }
    );
    assert!(e.to_string().contains("groups") && e.to_string().contains("expected 6, got 3"));

    // l1_ratio domain.
    for bad in [-0.1, 1.5, f64::NAN] {
        let e = group_lasso(x.as_ref(), &y, &groups, 0.1, bad, &sq, TIGHT).unwrap_err();
        assert!(e.to_string().contains("l1_ratio must lie in [0, 1]"), "{e}");
    }
    // custom weights: wrong length, then non-positive.
    let e = group_lasso(
        x.as_ref(),
        &y,
        &groups,
        0.1,
        0.0,
        &GroupWeights::Custom(vec![1.0, 1.0]),
        TIGHT,
    )
    .unwrap_err();
    assert!(
        matches!(
            e,
            MlError::DimensionMismatch {
                expected: 3,
                got: 2,
                ..
            }
        ),
        "{e}"
    );
    assert!(e.to_string().contains("group_weights"));
    let e = group_lasso(
        x.as_ref(),
        &y,
        &groups,
        0.1,
        0.0,
        &GroupWeights::Custom(vec![1.0, 0.0, 1.0]),
        TIGHT,
    )
    .unwrap_err();
    assert!(e
        .to_string()
        .contains("group_weights must be finite and strictly positive"));

    // NaN / inf name the array.
    let mut bad_cols = cols.clone();
    bad_cols[2][7] = f64::NAN;
    let bx = mat_from_cols(&bad_cols);
    assert_eq!(
        group_lasso(bx.as_ref(), &y, &groups, 0.1, 0.0, &sq, TIGHT).unwrap_err(),
        MlError::NonFinite { what: "x" }
    );
    let mut bad_y = y.clone();
    bad_y[0] = f64::INFINITY;
    assert_eq!(
        post_lasso(x.as_ref(), &bad_y, 0.1, 1.0, TIGHT).unwrap_err(),
        MlError::NonFinite { what: "y" }
    );
    let d: Vec<f64> = cols[0].clone();
    assert_eq!(
        pds_lasso(&y, &bad_y, x.as_ref(), PdsAlpha::Bic, None, TIGHT).unwrap_err(),
        MlError::NonFinite { what: "d" }
    );
    assert!(matches!(
        pds_lasso(&y, &d[..10], x.as_ref(), PdsAlpha::Bic, None, TIGHT).unwrap_err(),
        MlError::DimensionMismatch {
            expected: 60,
            got: 10,
            ..
        }
    ));
    assert!(
        pds_lasso(&y, &d, x.as_ref(), PdsAlpha::Fixed(-1.0), None, TIGHT)
            .unwrap_err()
            .to_string()
            .contains("alpha must be finite and non-negative")
    );

    // Insufficiency, house wording: a square 5 x 5 design at a tiny alpha
    // selects every column, leaving the refit no residual degrees of
    // freedom.
    let (sq_cols, sq_y) = seeded_design(9, 5, 5);
    let sx = mat_from_cols(&sq_cols);
    let e = post_lasso(sx.as_ref(), &sq_y, 1e-9, 1.0, TIGHT).unwrap_err();
    match e {
        MlError::InsufficientData { got, needed, .. } => {
            assert_eq!(got, 5);
            assert_eq!(needed, 6);
        }
        other => panic!("expected InsufficientData, got {other:?}"),
    }
    assert!(e
        .to_string()
        .starts_with("insufficient data: 5 observations, at least 6 required"));
    // pds: 5 rows, treatment plus four selected controls already exhaust
    // the sample.
    let (nar_cols, nar_y) = seeded_design(10, 5, 4);
    let nx = mat_from_cols(&nar_cols);
    let nd: Vec<f64> = (0..5)
        .map(|i| nar_cols[0][i] + 0.3 * nar_cols[1][i] + 0.1)
        .collect();
    let e = pds_lasso(
        &nar_y,
        &nd,
        nx.as_ref(),
        PdsAlpha::Fixed(1e-9),
        Some(0),
        TIGHT,
    )
    .unwrap_err();
    assert!(
        e.to_string()
            .starts_with("insufficient data: 5 observations, at least "),
        "{e}"
    );

    // A treatment identical to a control it selects makes the final OLS
    // design `[d, x_0]` singular: the HAC engine's error surfaces wrapped,
    // not as a panic.
    let d0: Vec<f64> = cols[0].clone();
    let e = pds_lasso(&y, &d0, x.as_ref(), PdsAlpha::Fixed(1e-6), Some(2), TIGHT).unwrap_err();
    assert!(matches!(e, MlError::Hac(_)), "{e}");
    assert!(e.to_string().contains("HAC/OLS engine"));
}

/// The refit is OLS on the support: the normal equations
/// `X_S'(y - X_S b_S) = 0` hold to 1e-9, the off-support entries are
/// exactly zero, and `rss` is the residual sum of squares of that fit.
#[test]
fn post_lasso_refit_satisfies_the_normal_equations() {
    let (cols, y) = seeded_design(21, 150, 10);
    let x = mat_from_cols(&cols);
    for &(alpha, l1) in &[(0.05, 1.0), (0.2, 1.0), (0.1, 0.5)] {
        let fit = post_lasso(x.as_ref(), &y, alpha, l1, TIGHT).unwrap();
        assert!(!fit.support.is_empty());
        let resid: Vec<f64> = (0..y.len())
            .map(|i| y[i] - (0..10).map(|j| cols[j][i] * fit.coef_ols[j]).sum::<f64>())
            .collect();
        for &j in &fit.support {
            let g: f64 = cols[j].iter().zip(&resid).map(|(a, b)| a * b).sum();
            assert!(g.abs() <= 1e-9, "normal equation {j}: {g:e}");
            assert!(fit.coef_lasso[j] != 0.0);
        }
        for j in 0..10 {
            if !fit.support.contains(&j) {
                assert_eq!(fit.coef_ols[j], 0.0);
                assert_eq!(fit.coef_lasso[j], 0.0);
            }
        }
        let rss: f64 = resid.iter().map(|r| r * r).sum();
        assert!((fit.rss - rss).abs() <= 1e-9 * rss);
        assert_eq!(fit.n_selected, fit.support.len());
    }
}

// ---------------------------------------------------------------------------
// Post-double-selection: coverage, measured
// ---------------------------------------------------------------------------

fn ar1(rng: &mut Lcg, n: usize, rho: f64) -> Vec<f64> {
    let mut e = vec![0.0; n];
    e[0] = rng.normal() / (1.0 - rho * rho).sqrt();
    for t in 1..n {
        e[t] = rho * e[t - 1] + rng.normal();
    }
    e
}

fn standardize(c: &mut [f64]) {
    let n = c.len() as f64;
    let mean = c.iter().sum::<f64>() / n;
    let sd = (c.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    for v in c.iter_mut() {
        *v = (*v - mean) / sd;
    }
}

fn center(c: &mut [f64]) {
    let mean = c.iter().sum::<f64>() / c.len() as f64;
    for v in c.iter_mut() {
        *v -= mean;
    }
}

/// 95% HAC (Bartlett, `lags`, `n/(n-k)`) interval for the first column's
/// coefficient in OLS of `y` on `design`, normal critical value.
fn hac_ci(y: &[f64], design: &[Vec<f64>], lags: usize) -> (f64, f64) {
    let fit = tsecon_hac::ols(y, design).unwrap();
    let inf = fit
        .inference(tsecon_hac::SeType::Hac {
            kernel: tsecon_hac::Kernel::Bartlett,
            bandwidth: lags as f64,
            use_correction: true,
        })
        .unwrap();
    let b = fit.params[0];
    let se = inf.bse[0];
    (b - 1.959963984540054 * se, b + 1.959963984540054 * se)
}

#[derive(Debug, Clone, Copy)]
struct Coverage {
    pds: f64,
    single: f64,
    oracle: f64,
    mean_union: f64,
    reps: usize,
}

/// One Monte-Carlo cell. The DGP:
///
/// ```text
/// x_{tj} AR(1) with coefficient rho_x (p controls),
/// d_t = x_t' gamma + v_t,   y_t = tau d_t + x_t' beta + e_t,   tau = 1,
/// v_t, e_t independent AR(1) with coefficient rho_e,
/// gamma = (1, 1, -1, 1, 0, ...),  beta = (0.15, 0.15, -0.15, 0.15, 0.5, -0.5, 0, ...).
/// ```
///
/// Controls 0-3 are the trap: they drive `d` strongly and `y` weakly, so
/// selecting on the outcome equation alone drops them and the omitted
/// piece `sum_j beta_j gamma_j / var(d)` biases `tau` by about a standard
/// error. Three estimators per draw, all with the same HAC interval:
///
/// * `pds`: [`pds_lasso`] with `alpha = "bic"`, `hac_lags = None`;
/// * `single`: the BCH single-selection comparator — LASSO of `y` on `x`
///   with `d` unpenalized (concentrated out exactly by Frisch-Waugh: LASSO
///   of `M_d y` on `M_d x`) at its BIC pick, then OLS of `y` on `[d, x_S]`;
/// * `oracle`: OLS of `y` on `[d, x_0..x_5]` (the true support — infeasible).
fn pds_cell(
    seed: u64,
    n: usize,
    p: usize,
    rho_x: f64,
    rho_e: f64,
    reps: usize,
    cd: CoordDescentOptions,
) -> Coverage {
    let mut rng = Lcg::new(seed);
    let lags = tsecon_hac::newey_west_maxlags(n);
    let (mut hit_pds, mut hit_single, mut hit_oracle) = (0usize, 0usize, 0usize);
    let mut union_total = 0usize;
    let covers = |ci: (f64, f64)| ci.0 <= 1.0 && 1.0 <= ci.1;
    for _ in 0..reps {
        let mut cols: Vec<Vec<f64>> = (0..p).map(|_| ar1(&mut rng, n, rho_x)).collect();
        let v = ar1(&mut rng, n, rho_e);
        let e = ar1(&mut rng, n, rho_e);
        let gamma = [1.0, 1.0, -1.0, 1.0];
        let beta = [0.15, 0.15, -0.15, 0.15, 0.5, -0.5];
        let mut d = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 0..n {
            let xg: f64 = (0..4).map(|j| cols[j][t] * gamma[j]).sum();
            let xb: f64 = (0..6).map(|j| cols[j][t] * beta[j]).sum();
            d[t] = xg + v[t];
            y[t] = d[t] + xb + e[t];
        }
        for c in cols.iter_mut() {
            standardize(c);
        }
        center(&mut d);
        center(&mut y);
        let x = mat_from_cols(&cols);

        let fit = pds_lasso(&y, &d, x.as_ref(), PdsAlpha::Bic, None, cd).unwrap();
        assert_eq!(fit.hac_lags_resolved, lags);
        hit_pds += usize::from(covers(fit.conf_int));
        union_total += fit.n_controls_selected;

        // Single selection with d concentrated out.
        let dd: f64 = d.iter().map(|v| v * v).sum();
        let proj_y: f64 = d.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>() / dd;
        let yt: Vec<f64> = y.iter().zip(&d).map(|(a, b)| a - b * proj_y).collect();
        let xt: Vec<Vec<f64>> = cols
            .iter()
            .map(|c| {
                let pj: f64 = d.iter().zip(c).map(|(a, b)| a * b).sum::<f64>() / dd;
                c.iter().zip(&d).map(|(a, b)| a - b * pj).collect()
            })
            .collect();
        let xtm = mat_from_cols(&xt);
        let path = regularization_path(
            xtm.as_ref(),
            &yt,
            1.0,
            PathOptions {
                cd,
                ..PathOptions::default()
            },
        )
        .unwrap();
        let sel = &path.coefs[path.bic_best()];
        let mut design = vec![d.clone()];
        design.extend((0..p).filter(|&j| sel[j] != 0.0).map(|j| cols[j].clone()));
        hit_single += usize::from(covers(hac_ci(&y, &design, lags)));

        let mut oracle = vec![d.clone()];
        oracle.extend((0..6).map(|j| cols[j].clone()));
        hit_oracle += usize::from(covers(hac_ci(&y, &oracle, lags)));
    }
    Coverage {
        pds: hit_pds as f64 / reps as f64,
        single: hit_single as f64 / reps as f64,
        oracle: hit_oracle as f64 / reps as f64,
        mean_union: union_total as f64 / reps as f64,
        reps,
    }
}

/// **Coverage, measured — the always-on cell.** One modest cell (and a
/// looser selection-stage tolerance, which does not move a BIC support)
/// so the claim is asserted on every run in a debug build: the PDS
/// interval's coverage lies within the three-sigma Monte-Carlo band around
/// 0.95 and within that band of the infeasible oracle, while the
/// single-selection interval covers less than half the time. The two
/// larger cells quoted on the model card are
/// `pds_coverage_full_measurement` below.
#[test]
fn pds_covers_where_single_selection_does_not() {
    let quick = CoordDescentOptions {
        tol: 1e-6,
        max_iter: 10_000,
    };
    let c = pds_cell(20260905, 200, 16, 0.5, 0.3, 80, quick);
    let se = (0.95f64 * 0.05 / c.reps as f64).sqrt();
    println!(
        "PDS coverage (always-on cell, n=200 p=16 rho_x=0.5 rho_e=0.3): pds={:.3} \
         single={:.3} oracle={:.3} (MC se {se:.3}, mean |union| {:.1}, reps {})",
        c.pds, c.single, c.oracle, c.mean_union, c.reps
    );
    assert!(
        c.single < 0.5,
        "single selection covers {:.3}: the trap did not spring",
        c.single
    );
    assert!(c.single < c.pds - 6.0 * se);
    assert!(
        (c.pds - 0.95).abs() <= 3.0 * se,
        "PDS coverage {:.3} outside 0.95 +/- {:.3}",
        c.pds,
        3.0 * se
    );
    assert!((c.pds - c.oracle).abs() <= 3.0 * se);
}

/// **Coverage, measured — the two cells the model card quotes.** Ignored
/// by default because 600 replications of three regularization paths each
/// take minutes in a debug build; reproduce with
///
/// ```text
/// cargo test -p tsecon-ml --release --test structured_properties -- --ignored --nocapture
/// ```
///
/// * Cell A (`n = 400`, `p = 40`, `rho_x = 0.5`, `rho_e = 0.3`): the PDS
///   interval's coverage lies within the three-sigma Monte-Carlo band
///   around 0.95, the single-selection interval covers less than half the
///   time, and PDS is within Monte-Carlo noise of the oracle.
/// * Cell B (`n = 200`, `p = 40`, `rho_x = rho_e = 0.5`): the same
///   ordering; here even the **oracle's** Newey-West interval falls short of
///   0.95 — the well-known small-sample downward bias of the Bartlett HAC
///   estimator under this much persistence, which PDS inherits from the
///   HAC engine rather than adds to. The assertion is that PDS tracks the
///   oracle to within the band, not that either hits 0.95.
#[test]
#[ignore = "600-replication measurement; run with --release --ignored"]
fn pds_coverage_full_measurement() {
    let a = pds_cell(20260903, 400, 40, 0.5, 0.3, 300, DEFAULT);
    let b = pds_cell(20260904, 200, 40, 0.5, 0.5, 300, DEFAULT);
    for (name, c) in [("A rho_e=0.3 n=400", a), ("B rho_e=0.5 n=200", b)] {
        let se = (0.95f64 * 0.05 / c.reps as f64).sqrt();
        println!(
            "PDS coverage cell {name}: pds={:.3} single={:.3} oracle={:.3} \
             (MC se {se:.3}, mean |union| {:.1}, reps {})",
            c.pds, c.single, c.oracle, c.mean_union, c.reps
        );
    }
    let se = (0.95f64 * 0.05 / a.reps as f64).sqrt();
    assert!(a.single < 0.5);
    assert!(a.single < a.pds - 6.0 * se);
    assert!((a.pds - 0.95).abs() <= 3.0 * se, "PDS {:.3}", a.pds);
    assert!((a.pds - a.oracle).abs() <= 3.0 * se);

    let se_b = (0.95f64 * 0.05 / b.reps as f64).sqrt();
    assert!(b.single < 0.5);
    assert!(b.single < b.pds - 6.0 * se_b);
    assert!(
        (b.pds - b.oracle).abs() <= 3.0 * se_b,
        "PDS {:.3} oracle {:.3}",
        b.pds,
        b.oracle
    );
    assert!(b.pds > 0.8, "PDS coverage {:.3} collapsed", b.pds);
}
