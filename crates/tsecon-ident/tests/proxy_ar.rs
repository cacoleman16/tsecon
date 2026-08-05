//! Golden, grid-inversion and property tests for the weak-instrument-robust
//! (Anderson-Rubin) confidence sets for proxy-SVAR impulse responses.
//!
//! Three independent legs, matching the three the fixture generator uses:
//!
//! 1. GOLDEN. `fixtures/proxy_ar.json` is produced by
//!    `fixtures/generate_proxy_ar_fixtures.py`, whose reduced form comes from
//!    statsmodels and whose AR algebra is a plain-NumPy transcription of the
//!    moment condition — never from this crate. Reproducing the moment
//!    covariance, the quadratic coefficients, the set shapes and the
//!    endpoints is a genuine cross-implementation check.
//! 2. GRID INVERSION. The closed-form quadratic is proved against a
//!    brute-force scan that re-tests `AR(lam) <= c` directly at thousands of
//!    candidate values per cell, for every shape the set can take. This is
//!    the strongest available validation of the inversion and it needs no
//!    external reference at all.
//! 3. PROPERTIES that the algebra guarantees exactly: `unit`-equivariance,
//!    nesting in the confidence level, invariance to a NaN prefix on the
//!    proxy, the point estimate always lying in its own set, boundedness
//!    being all-or-nothing across the grid, and — the Wald detector — the
//!    sets being genuinely asymmetric about the point estimate.
//! 4. THE REDUCED-FORM CORRECTION, which is the difference between a set that
//!    covers `0.95` and one that covers `0.119`. `psi_reduced_form_cov` is
//!    checked against a NUMERICAL Jacobian built by perturbing VAR
//!    coefficients one at a time — no Kronecker product, no companion matrix,
//!    no shared code with the analytic route. The correction is then required
//!    to widen every set, to leave `A`, `q0` and `v2` bit-identical (so
//!    weak-instrument robustness cannot regress), and to vanish quadratically
//!    with `gamma`. `tests/proxy_ar_coverage.rs` measures what all that buys.
//!
//! Tolerance for the golden leg: `rtol = 1e-9`, `atol = 1e-11`. Only
//! faer-vs-NumPy summation order separates the two implementations.

use serde_json::Value;
use tsecon_ident::proxy_ar::{
    ar_cell, proxy_ar_sets, psi_reduced_form_cov, ArCell, ArCritical, ArMoments, ArReducedForm,
    ArSet, ArVariance, ArVarianceSpec,
};
use tsecon_ident::proxy_svar;
use tsecon_ident::IdentError;
use tsecon_linalg::faer::Mat;

const RTOL: f64 = 1e-9;
const ATOL: f64 = 1e-11;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/proxy_ar.json",
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

/// JSON `null` decodes to `NaN` — the proxy's unavailability mask.
fn proxy_f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| {
            if x.is_null() {
                f64::NAN
            } else {
                x.as_f64().expect("number")
            }
        })
        .collect()
}

fn mat(v: &Value) -> Mat<f64> {
    let rows: Vec<Vec<f64>> = v.as_array().expect("array").iter().map(f64s).collect();
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn psis(v: &Value) -> Vec<Mat<f64>> {
    v.as_array().expect("array").iter().map(mat).collect()
}

fn close(actual: f64, expected: f64, what: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= ATOL + RTOL * (1.0 + expected.abs()),
        "{what}: got {actual}, expected {expected} (abs err {err})"
    );
}

/// The inputs of one fixture case, decoded.
struct Case {
    name: String,
    u: Mat<f64>,
    proxy: Vec<f64>,
    psi: Vec<Mat<f64>>,
    norm_var: usize,
    unit: f64,
    hac_lags: usize,
    critical: ArCritical,
    /// `T_O * Cov(Psi_hat_h gamma)` per horizon, when the case propagates the
    /// reduced form.
    psi_var: Option<Vec<Mat<f64>>>,
    /// `T_O * Cov(Psi_hat_h gamma, gamma_hat)` per horizon, when the case
    /// carries a nonzero cross-covariance.
    psi_gamma_cov: Option<Vec<Mat<f64>>>,
    expected: Value,
    kinds_present: Vec<String>,
}

fn decode(case: &Value) -> Case {
    let critical = match case["critical"]["kind"].as_str().expect("kind") {
        "chi2" => ArCritical::Chi2 {
            level: case["critical"]["level"].as_f64().expect("level"),
        },
        "f" => ArCritical::F {
            level: case["critical"]["level"].as_f64().expect("level"),
        },
        "value" => ArCritical::Value(case["critical"]["value"].as_f64().expect("value")),
        other => panic!("unknown critical kind {other}"),
    };
    Case {
        name: case["name"].as_str().expect("name").to_string(),
        u: mat(&case["resid"]),
        proxy: proxy_f64s(&case["proxy_aligned"]),
        psi: psis(&case["psi"]),
        norm_var: case["norm_var"].as_u64().expect("norm_var") as usize,
        unit: case["unit"].as_f64().expect("unit"),
        hac_lags: case["variance"]["lags"].as_u64().expect("lags") as usize,
        critical,
        psi_var: case["reduced_form"]
            .as_object()
            .map(|rf| psis(&rf["psi_var"])),
        psi_gamma_cov: case["reduced_form"]
            .as_object()
            .filter(|rf| !rf["psi_gamma_cov"].is_null())
            .map(|rf| psis(&rf["psi_gamma_cov"])),
        expected: case["expected"].clone(),
        kinds_present: case["kinds_present"]
            .as_array()
            .expect("kinds")
            .iter()
            .map(|x| x.as_str().expect("str").to_string())
            .collect(),
    }
}

impl Case {
    fn variance(&self) -> ArVariance<'static> {
        if self.hac_lags == 0 {
            ArVariance::Hc0
        } else {
            ArVariance::HacBartlett {
                lags: self.hac_lags,
            }
        }
    }

    /// The full variance specification, including the reduced-form correction
    /// when the fixture case carries one.
    fn spec(&self) -> ArVarianceSpec<'_> {
        match &self.psi_var {
            None => ArVarianceSpec::moment_only(self.variance()),
            Some(pv) => ArVarianceSpec::with_reduced_form(
                self.variance(),
                ArReducedForm {
                    psi_var: pv,
                    psi_gamma_cov: self.psi_gamma_cov.as_deref(),
                },
            ),
        }
    }

    fn run(&self) -> Result<tsecon_ident::proxy_ar::ProxyArResult, IdentError> {
        proxy_ar_sets(
            self.u.as_ref(),
            &self.proxy,
            &self.psi,
            self.norm_var,
            self.unit,
            self.spec(),
            self.critical,
        )
    }
}

fn cases() -> Vec<Case> {
    load()["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(decode)
        .collect()
}

fn kind_str(set: &ArSet) -> &'static str {
    set.kind().as_str()
}

// ---------------------------------------------------------------------------
// LEG 1: golden
// ---------------------------------------------------------------------------

/// Every scalar the NumPy reference computes is reproduced: the moment
/// covariance, the shared boundedness statistic, the per-cell quadratic
/// coefficients, the set SHAPE, and the endpoints of that shape.
#[test]
fn matches_numpy_reference() -> Result<(), IdentError> {
    for case in cases() {
        let res = case.run()?;
        let exp = &case.expected;
        let tag = &case.name;

        assert_eq!(
            res.n_proxy,
            exp["n_proxy"].as_u64().expect("n_proxy") as usize,
            "[{tag}] n_proxy"
        );
        close(
            res.critical_value,
            exp["critical_value"].as_f64().expect("c"),
            &format!("[{tag}] critical_value"),
        );
        close(
            res.ar_bound_stat,
            exp["ar_bound_stat"].as_f64().expect("bstat"),
            &format!("[{tag}] ar_bound_stat"),
        );
        assert_eq!(
            res.ar_bounded_all,
            exp["ar_bounded_all"].as_bool().expect("bounded_all"),
            "[{tag}] ar_bounded_all"
        );
        close(
            res.first_stage_f,
            exp["first_stage_f"].as_f64().expect("F"),
            &format!("[{tag}] first_stage_f"),
        );
        assert_eq!(
            res.reduced_form_uncertainty,
            exp["reduced_form_uncertainty"].as_bool().expect("rf flag"),
            "[{tag}] reduced_form_uncertainty"
        );
        // `level` is claimed only when the sets earn it: a moment-only
        // variance conditions on the reduced form and must report `None`.
        match (res.level, exp["level"].as_f64()) {
            (None, None) => {}
            (Some(a), Some(b)) => close(a, b, &format!("[{tag}] level")),
            (a, b) => panic!("[{tag}] level: got {a:?}, expected {b:?}"),
        }
        for (j, &g) in f64s(&exp["cov_um"]).iter().enumerate() {
            close(res.cov_um[j], g, &format!("[{tag}] cov_um[{j}]"));
        }
        for (j, &bj) in f64s(&exp["impact"]).iter().enumerate() {
            close(res.impact[j], bj, &format!("[{tag}] impact[{j}]"));
        }
        let om = mat(&exp["omega"]);
        for i in 0..om.nrows() {
            for j in 0..om.ncols() {
                close(
                    res.omega[(i, j)],
                    om[(i, j)],
                    &format!("[{tag}] omega[{i},{j}]"),
                );
            }
        }

        let ecells = exp["cells"].as_array().expect("cells");
        assert_eq!(res.cells.len(), ecells.len(), "[{tag}] horizons");
        for (h, erow) in ecells.iter().enumerate() {
            let erow = erow.as_array().expect("row");
            assert_eq!(res.cells[h].len(), erow.len(), "[{tag}] n at h={h}");
            for (i, ec) in erow.iter().enumerate() {
                let c = &res.cells[h][i];
                let at = format!("[{tag}] cell (h={h}, i={i})");
                close(
                    c.point,
                    ec["point"].as_f64().expect("point"),
                    &(at.clone() + " point"),
                );
                close(c.a, ec["a"].as_f64().expect("a"), &(at.clone() + " A"));
                close(c.b, ec["b"].as_f64().expect("b"), &(at.clone() + " B"));
                close(c.c, ec["c"].as_f64().expect("c"), &(at.clone() + " C"));
                close(c.q1, ec["q1"].as_f64().expect("q1"), &(at.clone() + " q1"));
                close(c.q0, ec["q0"].as_f64().expect("q0"), &(at.clone() + " q0"));
                close(c.v0, ec["v0"].as_f64().expect("v0"), &(at.clone() + " v0"));
                close(c.v1, ec["v1"].as_f64().expect("v1"), &(at.clone() + " v1"));
                close(c.v2, ec["v2"].as_f64().expect("v2"), &(at.clone() + " v2"));
                assert_eq!(
                    c.excludes_zero,
                    ec["excludes_zero"].as_bool().expect("ez"),
                    "{at} excludes_zero"
                );
                assert_eq!(
                    kind_str(&c.set),
                    ec["kind"].as_str().expect("kind"),
                    "{at} kind"
                );
                // The endpoint payload of whatever shape it is. The reference
                // stores lo/hi with the same meaning the Rust variant does,
                // including "the rejected middle" for an exterior set.
                match c.set {
                    ArSet::Interval { lo, hi } | ArSet::Exterior { lo, hi } => {
                        close(lo, ec["lo"].as_f64().expect("lo"), &(at.clone() + " lo"));
                        close(hi, ec["hi"].as_f64().expect("hi"), &(at.clone() + " hi"));
                    }
                    ArSet::Point(p) => {
                        close(p, ec["lo"].as_f64().expect("lo"), &(at.clone() + " point"))
                    }
                    ArSet::RayBelow { hi } => {
                        close(hi, ec["hi"].as_f64().expect("hi"), &(at.clone() + " hi"))
                    }
                    ArSet::RayAbove { lo } => {
                        close(lo, ec["lo"].as_f64().expect("lo"), &(at.clone() + " lo"))
                    }
                    ArSet::Whole | ArSet::Empty => {}
                }
            }
        }
    }
    Ok(())
}

/// Running the fixture cases through `proxy_ar_sets` produces every shape the
/// taxonomy admits except `Empty`, which the never-empty result makes
/// unreachable here.
///
/// The shapes are collected from what the CODE returns, not from the
/// `kinds_present` list the fixture records — a test that only reads the
/// fixture would stay green if `proxy_ar_sets` lost a branch entirely. The
/// fixture's own list is then cross-checked against it, so the two must agree.
#[test]
fn every_shape_is_covered() -> Result<(), IdentError> {
    let mut seen: Vec<&'static str> = Vec::new();
    for case in cases() {
        let res = case.run()?;
        let mut here: Vec<&'static str> = Vec::new();
        for row in &res.cells {
            for cell in row {
                let k = kind_str(&cell.set);
                if !here.contains(&k) {
                    here.push(k);
                }
                if !seen.contains(&k) {
                    seen.push(k);
                }
            }
        }
        here.sort_unstable();
        let recorded: Vec<&str> = case.kinds_present.iter().map(String::as_str).collect();
        assert_eq!(
            here, recorded,
            "[{}] shapes the code produced disagree with the fixture's record",
            case.name
        );
    }
    for needed in [
        "interval",
        "point",
        "exterior",
        "whole",
        "ray_below",
        "ray_above",
    ] {
        assert!(
            seen.contains(&needed),
            "no fixture case makes proxy_ar_sets return the {needed} shape; saw {seen:?}"
        );
    }
    assert!(
        !seen.contains(&"empty"),
        "the never-empty result was violated: {seen:?}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// LEG 2: brute-force grid inversion
// ---------------------------------------------------------------------------

/// Prove the closed-form inversion by re-testing `AR(lam) <= c` directly.
///
/// For every cell of every fixture case, scan a fine grid of candidate values,
/// evaluate the statistic from its own moments, and demand that the accepted
/// grid points are exactly the ones the closed-form set claims. A
/// disagreement is tolerated only where the point sits on the boundary, judged
/// by the two sides of the inequality being equal to within `1e-10` of their
/// own magnitude — the relative scale matters, because at the knife edge the
/// whole quadratic is float noise.
///
/// This is the check that the four-case taxonomy is not merely internally
/// consistent but actually describes `{lam : AR(lam) <= c}`.
#[test]
fn grid_inversion_reproduces_every_set() -> Result<(), IdentError> {
    const NPTS: usize = 4001;
    let mut checked = 0usize;
    let mut points = 0usize;
    for case in cases() {
        let res = case.run()?;
        let no = res.n_proxy as f64;
        let crit = res.critical_value;
        for row in &res.cells {
            for cell in row {
                checked += 1;
                // Centre on the action: the point estimate and any endpoint.
                let (mut lo_a, mut hi_a) = (cell.point, cell.point);
                for x in [cell.set.endpoints().0, cell.set.endpoints().1]
                    .into_iter()
                    .chain(
                        cell.set
                            .excluded_middle()
                            .into_iter()
                            .flat_map(|(a, b)| [a, b]),
                    )
                {
                    if x.is_finite() {
                        lo_a = lo_a.min(x);
                        hi_a = hi_a.max(x);
                    }
                }
                let centre = 0.5 * (lo_a + hi_a);
                let half = (4.0 * (hi_a - lo_a)).max(5.0 * cell.point.abs().max(1.0));
                for g in 0..NPTS {
                    let lam = centre - half + 2.0 * half * (g as f64) / ((NPTS - 1) as f64);
                    let gg = cell.moment(lam);
                    let vv = cell.variance(lam);
                    // AR = 0/0 exactly at lam = unit in the (norm_var, h = 0)
                    // cell; a grid check must skip it rather than divide.
                    if vv.abs() <= 1e-14 * cell.v0.abs().max(cell.v2.abs()).max(1.0) {
                        continue;
                    }
                    points += 1;
                    let lhs = no * gg * gg;
                    let rhs = crit * vv;
                    let by_grid = lhs <= rhs;
                    let by_form = cell.set.contains(lam);
                    if by_grid == by_form {
                        continue;
                    }
                    let magn = lhs + rhs.abs();
                    assert!(
                        (lhs - rhs).abs() <= 1e-10 * magn.max(f64::MIN_POSITIVE),
                        "[{}] grid and closed form disagree away from a boundary at lam={lam}: \
                         kind={}, AR={} vs c={crit}",
                        case.name,
                        kind_str(&cell.set),
                        lhs / vv
                    );
                }
            }
        }
    }
    assert!(checked >= 100 && points >= 400_000, "grid check too thin");
    Ok(())
}

// ---------------------------------------------------------------------------
// LEG 3: exact properties of the algebra
// ---------------------------------------------------------------------------

/// The point estimate is bit-for-bit the one `proxy_svar` reports, so the set
/// and the IRF a user plots cannot drift apart.
#[test]
fn point_estimate_matches_proxy_svar_bitwise() -> Result<(), IdentError> {
    for case in cases() {
        let n = case.u.ncols();
        // proxy_svar needs a PD sigma_u only for its shock series, which is
        // irrelevant here; the identity is the cheapest valid choice.
        let sigma = Mat::<f64>::identity(n, n);
        let pt = proxy_svar(
            case.u.as_ref(),
            &case.proxy,
            &case.psi,
            sigma.as_ref(),
            case.norm_var,
            case.unit,
            true,
        )?;
        let ar = case.run()?;
        for h in 0..case.psi.len() {
            for i in 0..n {
                assert_eq!(
                    ar.cells[h][i].point, pt.irf[h][i],
                    "[{}] point estimate diverged from proxy_svar at (h={h}, i={i})",
                    case.name
                );
            }
        }
        // And the shared diagnostics agree exactly too.
        assert_eq!(ar.cov_um, pt.cov_um, "[{}] cov_um", case.name);
        assert_eq!(ar.impact, pt.impact, "[{}] impact", case.name);
        assert_eq!(
            ar.first_stage_f, pt.first_stage_f,
            "[{}] first_stage_f",
            case.name
        );
    }
    Ok(())
}

/// The point estimate is always inside its own set — the never-empty result.
/// A `Point` set is compared with a tolerance because it is the one shape
/// whose representative is reached through the quadratic's coefficients.
#[test]
fn point_estimate_is_always_in_its_own_set() -> Result<(), IdentError> {
    for case in cases() {
        let res = case.run()?;
        for (h, row) in res.cells.iter().enumerate() {
            for (i, cell) in row.iter().enumerate() {
                assert!(
                    !matches!(cell.set, ArSet::Empty),
                    "[{}] empty set at (h={h}, i={i}) — impossible in the 1x1 case",
                    case.name
                );
                let inside = match cell.set {
                    ArSet::Point(p) => (p - cell.point).abs() <= 1e-12 * (1.0 + p.abs()),
                    other => other.contains(cell.point),
                };
                assert!(
                    inside,
                    "[{}] point {} outside its own {} set at (h={h}, i={i})",
                    case.name,
                    cell.point,
                    kind_str(&cell.set)
                );
            }
        }
    }
    Ok(())
}

/// The impact response of the normalizing variable is the single point
/// `unit`, EXACTLY — the unit-effect normalization pins it by construction.
/// Reporting `Empty` there is the classic wrong branch.
#[test]
fn normalizing_impact_cell_is_the_point_unit() -> Result<(), IdentError> {
    let mut saw_point = false;
    let mut saw_whole = false;
    for case in cases() {
        let res = case.run()?;
        let cell = &res.cells[0][case.norm_var];
        match cell.set {
            ArSet::Point(p) => {
                assert_eq!(
                    p, case.unit,
                    "[{}] the (norm_var, h=0) point must be exactly `unit`",
                    case.name
                );
                assert!(res.ar_bounded_all, "[{}] point implies bounded", case.name);
                saw_point = true;
            }
            ArSet::Whole => {
                assert!(
                    !res.ar_bounded_all,
                    "[{}] whole line implies unbounded",
                    case.name
                );
                saw_whole = true;
            }
            other => panic!(
                "[{}] the (norm_var, h=0) cell must be a point or the whole line, got {other:?}",
                case.name
            ),
        }
    }
    assert!(saw_point && saw_whole, "both branches must be exercised");
    Ok(())
}

/// Boundedness is decided by one scalar shared by the whole grid, so it is
/// all-or-nothing — never "bounded at h=0, unbounded at h=8".
#[test]
fn boundedness_is_all_or_nothing() -> Result<(), IdentError> {
    for case in cases() {
        let res = case.run()?;
        let (bounded, total) = res.bounded_count();
        assert!(
            bounded == 0 || bounded == total,
            "[{}] mixed boundedness: {bounded} of {total}",
            case.name
        );
        assert_eq!(
            res.ar_bounded_all,
            bounded == total,
            "[{}] ar_bounded_all disagrees with the cells",
            case.name
        );
        // The shared leading coefficient really is shared.
        let a0 = res.cells[0][0].a;
        for row in &res.cells {
            for cell in row {
                assert_eq!(cell.a, a0, "[{}] A varies across cells", case.name);
                assert_eq!(cell.q0, res.cells[0][0].q0, "[{}] q0 varies", case.name);
                assert_eq!(cell.v2, res.cells[0][0].v2, "[{}] v2 varies", case.name);
            }
        }
    }
    Ok(())
}

/// THE WALD DETECTOR. A set built with the null imposed in the variance is
/// asymmetric about the point estimate; one built with a `lam`-independent
/// variance (the delta method frozen at `lam_hat`, or a bootstrap SD) is
/// exactly symmetric. A strong-instrument arm alone will never catch that
/// substitution — this test will.
#[test]
fn sets_are_asymmetric_about_the_point() -> Result<(), IdentError> {
    for case in cases() {
        let res = case.run()?;
        if !res.ar_bounded_all {
            continue;
        }
        let mut asym = 0usize;
        let mut total = 0usize;
        for row in &res.cells {
            for cell in row {
                if let ArSet::Interval { lo, hi } = cell.set {
                    total += 1;
                    let left = cell.point - lo;
                    let right = hi - cell.point;
                    if (left - right).abs() > 1e-6 * (left + right) {
                        asym += 1;
                    }
                }
            }
        }
        assert!(total > 0, "[{}] no bounded intervals to check", case.name);
        assert!(
            asym * 2 > total,
            "[{}] only {asym} of {total} intervals are asymmetric about the point estimate; \
             a symmetric set means the variance stopped depending on the tested value",
            case.name
        );
    }
    Ok(())
}

/// Scaling `unit` scales the whole set by the same factor, exactly — the
/// cheapest test that `unit` is applied once and to every term.
///
/// The reduced-form cases matter most here: [`ArReducedForm`] is deliberately
/// stated free of `unit` (it is `Cov(Psi_hat_h gamma)`, not of
/// `unit * Psi_hat_h gamma`), so the crate has to apply `unit^2` to the `v0`
/// correction and `unit` to the `v1` correction. Getting either power wrong
/// breaks this test and nothing else.
#[test]
fn sets_are_equivariant_in_unit() -> Result<(), IdentError> {
    for case in cases() {
        // The knife-edge cases pin the critical value to a statistic computed
        // at their own `unit`; rescaling there moves the knife edge, so the
        // comparison is run on the level-based cases.
        if matches!(case.critical, ArCritical::Value(_)) {
            continue;
        }
        let base = case.run()?;
        for &s in &[2.5f64, -1.5] {
            let scaled = proxy_ar_sets(
                case.u.as_ref(),
                &case.proxy,
                &case.psi,
                case.norm_var,
                case.unit * s,
                case.spec(),
                case.critical,
            )?;
            for (h, row) in base.cells.iter().enumerate() {
                for (i, cell) in row.iter().enumerate() {
                    let got = &scaled.cells[h][i];
                    let at = format!("[{}] (h={h}, i={i}) unit x {s}", case.name);
                    assert_eq!(
                        kind_str(&got.set),
                        kind_str(&cell.set),
                        "{at}: shape changed"
                    );
                    // s * [lo, hi] reverses order when s < 0, so compare the
                    // sorted images.
                    let (a0, b0) = cell.set.endpoints();
                    let (a1, b1) = got.set.endpoints();
                    let mut want = [a0 * s, b0 * s];
                    want.sort_by(|x, y| x.partial_cmp(y).expect("no NaN"));
                    let mut have = [a1, b1];
                    have.sort_by(|x, y| x.partial_cmp(y).expect("no NaN"));
                    for k in 0..2 {
                        if want[k].is_finite() {
                            assert!(
                                (have[k] - want[k]).abs() <= 1e-9 * (1.0 + want[k].abs()),
                                "{at}: endpoint {k} is {} not {}",
                                have[k],
                                want[k]
                            );
                        } else {
                            assert_eq!(have[k], want[k], "{at}: endpoint {k} infinity");
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Only the critical value moves with the level, and the quadratic is
/// monotone in it, so the sets are exactly nested: 90% inside 95% inside 99%.
#[test]
fn confidence_levels_are_nested() -> Result<(), IdentError> {
    for case in cases() {
        let mut prev: Option<Vec<Vec<ArCell>>> = None;
        for &level in &[0.90f64, 0.95, 0.99] {
            let res = proxy_ar_sets(
                case.u.as_ref(),
                &case.proxy,
                &case.psi,
                case.norm_var,
                case.unit,
                case.variance(),
                ArCritical::Chi2 { level },
            )?;
            if let Some(inner) = prev {
                for (h, row) in inner.iter().enumerate() {
                    for (i, small) in row.iter().enumerate() {
                        let big = &res.cells[h][i];
                        // Sample the smaller set and demand the larger one
                        // contains every sampled member.
                        for probe in sample_set(&small.set, small.point) {
                            assert!(
                                big.set.contains(probe)
                                    || matches!(big.set, ArSet::Point(p)
                                        if (p - probe).abs() <= 1e-9 * (1.0 + p.abs())),
                                "[{}] level {level} set does not contain the tighter set at \
                                 (h={h}, i={i}): probe {probe} outside {:?}",
                                case.name,
                                big.set
                            );
                        }
                    }
                }
            }
            prev = Some(res.cells);
        }
    }
    Ok(())
}

/// A handful of representative members of a set, for the nesting probe.
fn sample_set(set: &ArSet, point: f64) -> Vec<f64> {
    match *set {
        ArSet::Interval { lo, hi } => {
            let mid = 0.5 * (lo + hi);
            // Step just inside the endpoints: nesting is exact in exact
            // arithmetic, and the endpoints themselves are the only place
            // rounding can flip a strict comparison.
            let eps = 1e-9 * (hi - lo).max(1e-300);
            vec![lo + eps, mid, hi - eps, point]
        }
        ArSet::Point(p) => vec![p],
        ArSet::RayBelow { hi } => vec![hi - 1.0, hi - 1e3, point],
        ArSet::RayAbove { lo } => vec![lo + 1.0, lo + 1e3, point],
        ArSet::Exterior { lo, hi } => vec![lo - 1e-6, lo - 1.0, hi + 1e-6, hi + 1.0, point],
        ArSet::Whole => vec![point, point - 1e6, point + 1e6],
        ArSet::Empty => vec![],
    }
}

/// A NaN prefix on the proxy must leave the sets BIT-IDENTICAL to running on
/// the truncated sample: the AR statistic is scaled by the overlap count
/// `T_O`, never by the full residual length `T`. Scaling by `T` instead
/// shrinks every set by `sqrt(T/T_O)` and shows up as coverage that degrades
/// as the proxy's missing fraction rises.
#[test]
fn nan_prefix_leaves_the_sets_unchanged() -> Result<(), IdentError> {
    for case in cases() {
        let drop = 37usize;
        let t = case.u.nrows();
        let n = case.u.ncols();
        assert!(t > drop + 10);

        // (a) mask the first `drop` proxy entries, keeping every residual row.
        let mut masked = case.proxy.clone();
        for slot in masked.iter_mut().take(drop) {
            *slot = f64::NAN;
        }
        let with_mask = proxy_ar_sets(
            case.u.as_ref(),
            &masked,
            &case.psi,
            case.norm_var,
            case.unit,
            case.variance(),
            case.critical,
        )?;

        // (b) physically drop those rows.
        let u_short = Mat::from_fn(t - drop, n, |i, j| case.u[(i + drop, j)]);
        let p_short: Vec<f64> = case.proxy[drop..].to_vec();
        let truncated = proxy_ar_sets(
            u_short.as_ref(),
            &p_short,
            &case.psi,
            case.norm_var,
            case.unit,
            case.variance(),
            case.critical,
        )?;

        assert_eq!(with_mask.n_proxy, truncated.n_proxy, "[{}]", case.name);
        for (h, row) in with_mask.cells.iter().enumerate() {
            for (i, cell) in row.iter().enumerate() {
                let other = &truncated.cells[h][i];
                assert_eq!(
                    cell.set, other.set,
                    "[{}] set moved when the proxy prefix was masked rather than dropped \
                     at (h={h}, i={i})",
                    case.name
                );
                assert_eq!(cell.a, other.a, "[{}] A moved", case.name);
                assert_eq!(cell.point, other.point, "[{}] point moved", case.name);
            }
        }
    }
    Ok(())
}

/// `excludes_zero` is `C > 0`, which is exactly "zero is not in the set".
#[test]
fn excludes_zero_agrees_with_membership() -> Result<(), IdentError> {
    for case in cases() {
        let res = case.run()?;
        for (h, row) in res.cells.iter().enumerate() {
            for (i, cell) in row.iter().enumerate() {
                if matches!(cell.set, ArSet::Point(_)) {
                    continue; // a measure-zero set never literally contains 0
                }
                assert_eq!(
                    cell.excludes_zero,
                    !cell.set.contains(0.0),
                    "[{}] excludes_zero disagrees with membership at (h={h}, i={i}): {:?}",
                    case.name,
                    cell.set
                );
            }
        }
    }
    Ok(())
}

/// ONE TOLERANCE, TWO DECISIONS. At the boundedness knife edge `A` and `B` are
/// float residue and `solve_set` calls the set the whole real line whenever
/// `C <= tau_c`. Reading `C > 0.0` exactly for `excludes_zero` — as an
/// independent test of the numerator moment — then reports that a set
/// containing every real number excludes zero.
///
/// The moments below put `C` at `1e-13` with `tau_c = 4e-12`: inside the
/// tolerance the set uses, outside the one a bare `C > 0.0` would use. Every
/// quantity is a small power of two so `A` and `B` are exactly zero and the
/// branch is reached deterministically rather than by luck of rounding.
#[test]
fn excludes_zero_uses_the_same_tolerance_as_the_set() -> Result<(), IdentError> {
    let cell = ar_cell(
        4,
        4.0,
        ArMoments {
            q1: 1.0,
            q0: 1.0,
            // C = 4 - 4*v0 = 1e-13 > 0, but well inside tau_c = 4e-12.
            v0: 1.0 - 2.5e-14,
            v1: 1.0,
            v2: 1.0,
            point: 1.0,
        },
    )?;
    assert_eq!(cell.a, 0.0, "the knife edge was not reached");
    assert_eq!(cell.b, 0.0, "the degenerate branch was not reached");
    assert!(
        cell.c > 0.0,
        "C must be positive for this test to have teeth"
    );
    assert_eq!(cell.set, ArSet::Whole);
    assert!(
        !cell.excludes_zero,
        "a set that is the whole real line cannot exclude zero (C = {})",
        cell.c
    );
    assert!(cell.set.contains(0.0));
    Ok(())
}

/// Supplying the same `Omega` explicitly reproduces the built-in HC0 path
/// bit-for-bit — the route a moving-block-bootstrap covariance takes.
#[test]
fn supplied_omega_reproduces_hc0() -> Result<(), IdentError> {
    let case = &cases()[0];
    let base = case.run()?;
    let supplied = proxy_ar_sets(
        case.u.as_ref(),
        &case.proxy,
        &case.psi,
        case.norm_var,
        case.unit,
        ArVariance::Supplied(base.omega.as_ref()),
        case.critical,
    )?;
    for (h, row) in base.cells.iter().enumerate() {
        for (i, cell) in row.iter().enumerate() {
            assert_eq!(cell.set, supplied.cells[h][i].set, "(h={h}, i={i})");
            assert_eq!(cell.a, supplied.cells[h][i].a);
            assert_eq!(cell.b, supplied.cells[h][i].b);
            assert_eq!(cell.c, supplied.cells[h][i].c);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The reduced-form (Psi_h) correction
// ---------------------------------------------------------------------------

/// A fixture case that propagates the reduced form, and its moment-only twin
/// on the identical data.
fn reduced_form_pair() -> (Case, Case) {
    let all = cases();
    let with = all
        .iter()
        .find(|c| c.name == "strong_hc0_reduced_form")
        .expect("fixture has a reduced-form case");
    let without = all
        .iter()
        .find(|c| c.name == "strong_hc0")
        .expect("fixture has the moment-only twin");
    // Same data, same critical value, same normalization — only the variance
    // differs, which is what makes the comparisons below clean.
    assert_eq!(with.norm_var, without.norm_var);
    assert_eq!(with.unit, without.unit);
    assert_eq!(with.proxy.len(), without.proxy.len());
    (
        Case {
            name: with.name.clone(),
            u: with.u.clone(),
            proxy: with.proxy.clone(),
            psi: with.psi.clone(),
            norm_var: with.norm_var,
            unit: with.unit,
            hac_lags: with.hac_lags,
            critical: with.critical,
            psi_var: with.psi_var.clone(),
            psi_gamma_cov: with.psi_gamma_cov.clone(),
            expected: with.expected.clone(),
            kinds_present: with.kinds_present.clone(),
        },
        Case {
            name: without.name.clone(),
            u: without.u.clone(),
            proxy: without.proxy.clone(),
            psi: without.psi.clone(),
            norm_var: without.norm_var,
            unit: without.unit,
            hac_lags: without.hac_lags,
            critical: without.critical,
            psi_var: None,
            psi_gamma_cov: None,
            expected: without.expected.clone(),
            kinds_present: without.kinds_present.clone(),
        },
    )
}

/// THE BLOCKER'S TEST. Propagating reduced-form uncertainty must (a) widen
/// every set, (b) leave the boundedness switch — and therefore weak-instrument
/// robustness — untouched, and (c) flip the machine-readable flags.
///
/// (b) is the load-bearing half. `A = T_O*q0^2 - c*v2` is built from the
/// DENOMINATOR moment only, and the correction is a constant on `v0` and a
/// constant on `v1`. If a future edit ever routed part of it into `v2`, the
/// set would stop being weak-IV robust — it would stay bounded where Dufour
/// (1997) says no bounded set can be valid — and only this assertion would
/// notice.
#[test]
fn reduced_form_widens_the_set_without_touching_boundedness() -> Result<(), IdentError> {
    let (with, without) = reduced_form_pair();
    let a = with.run()?;
    let b = without.run()?;

    assert!(a.reduced_form_uncertainty, "the flag must be set");
    assert!(!b.reduced_form_uncertainty);
    assert_eq!(
        a.level,
        Some(0.95),
        "a propagated set earns its nominal level"
    );
    assert_eq!(
        b.level, None,
        "a moment-only set must not advertise a level it does not have"
    );

    // The boundedness switch is untouched, exactly.
    assert_eq!(a.ar_bound_stat, b.ar_bound_stat);
    assert_eq!(a.ar_bounded_all, b.ar_bounded_all);
    assert_eq!(a.first_stage_f, b.first_stage_f);
    for (h, row) in a.cells.iter().enumerate() {
        for (i, cell) in row.iter().enumerate() {
            let other = &b.cells[h][i];
            let at = format!("(h={h}, i={i})");
            assert_eq!(cell.a, other.a, "{at} A must not move");
            assert_eq!(cell.q0, other.q0, "{at} q0 must not move");
            assert_eq!(cell.q1, other.q1, "{at} q1 must not move");
            assert_eq!(cell.v2, other.v2, "{at} v2 must not move");
            assert_eq!(
                cell.point, other.point,
                "{at} the point estimate is the same"
            );
            // With no cross-covariance the correction is a nonnegative
            // constant on v0 alone, so V(lam) rises everywhere and the set
            // can only grow.
            assert!(
                cell.v0 >= other.v0,
                "{at} v0 fell: {} < {}",
                cell.v0,
                other.v0
            );
            assert_eq!(cell.v1, other.v1, "{at} v1 moves only with a cross term");
            if h > 0 {
                assert!(
                    cell.v0 > other.v0,
                    "{at} the correction vanished at an estimated horizon"
                );
            }
            for lam in sample_set(&other.set, other.point) {
                assert!(
                    cell.set.contains(lam) || matches!(cell.set, ArSet::Point(_)),
                    "{at} the corrected set dropped {lam}, which the moment-only set kept"
                );
            }
        }
    }
    // And the widening is not cosmetic: at the longest horizon the response is
    // small and reduced-form error dominates it.
    let hmax = a.cells.len() - 1;
    let widen: Vec<f64> = (0..a.cells[hmax].len())
        .map(|i| a.cells[hmax][i].set.width() / b.cells[hmax][i].set.width())
        .collect();
    assert!(
        widen.iter().all(|&r| r > 1.5),
        "reduced-form propagation barely moved the h={hmax} sets: {widen:?}"
    );
    Ok(())
}

/// The cross-covariance branch moves `v0` and `v1` by exactly the documented
/// amounts — `unit^2 * 2 * sum_j Psi_h[i,j] X_h[i,j]` and `unit * X_h[i,k]`.
///
/// The fixture's cross case pins this against NumPy; this pins the algebra
/// itself, so a sign error or a transposed index cannot hide behind a
/// regenerated fixture.
#[test]
fn cross_covariance_shifts_v0_and_v1_exactly() -> Result<(), IdentError> {
    let all = cases();
    let cross = all
        .iter()
        .find(|c| c.name == "strong_hc0_reduced_form_cross")
        .expect("fixture has a cross-covariance case");
    let plain = all
        .iter()
        .find(|c| c.name == "strong_hc0_reduced_form")
        .expect("fixture has the no-cross twin");
    let a = cross.run()?;
    let b = plain.run()?;
    let x = cross.psi_gamma_cov.as_ref().expect("cross matrices");
    let k = cross.norm_var;
    let unit = cross.unit;
    for (h, row) in a.cells.iter().enumerate() {
        for (i, cell) in row.iter().enumerate() {
            let other = &b.cells[h][i];
            let mut num = 0.0;
            for j in 0..cross.psi[h].ncols() {
                num += cross.psi[h][(i, j)] * x[h][(i, j)];
            }
            let want_v0 = other.v0 + unit * unit * 2.0 * num;
            let want_v1 = other.v1 + unit * x[h][(i, k)];
            let at = format!("(h={h}, i={i})");
            assert!(
                (cell.v0 - want_v0).abs() <= 1e-12 * (1.0 + want_v0.abs()),
                "{at} v0: {} not {want_v0}",
                cell.v0
            );
            assert!(
                (cell.v1 - want_v1).abs() <= 1e-12 * (1.0 + want_v1.abs()),
                "{at} v1: {} not {want_v1}",
                cell.v1
            );
        }
    }
    Ok(())
}

/// `psi_reduced_form_cov` against a NUMERICAL Jacobian.
///
/// The analytic route builds `G_h` from companion powers and Kronecker
/// products, where a transposed index or a wrong `vec` layout is invisible to
/// inspection. The reference here perturbs one VAR coefficient at a time,
/// recomputes `Psi_h gamma` from scratch through the MA recursion, and forms
/// the sandwich from the resulting finite-difference Jacobian — no Kronecker
/// product, no companion matrix, no shared code.
#[test]
fn psi_reduced_form_cov_matches_a_numerical_jacobian() -> Result<(), IdentError> {
    const N: usize = 3;
    const P: usize = 2;
    const H: usize = 5;
    let coefs = vec![
        Mat::from_fn(N, N, |i, j| 0.5 / (1.0 + i as f64) - 0.1 * (j as f64)),
        Mat::from_fn(N, N, |i, j| 0.08 * (i as f64) - 0.05 * (j as f64) + 0.04),
    ];
    let gamma = [0.7, -0.35, 0.2];
    let n_proxy = 240usize;
    let dim = P * N * N;
    // A deterministic PSD coefficient covariance: M M' / 400, full rank.
    let m = Mat::from_fn(dim, dim, |i, j| {
        (((i * 7 + j * 3) % 11) as f64 - 5.0) / 40.0 + f64::from(u8::from(i == j))
    });
    let cov_alpha = Mat::from_fn(dim, dim, |i, j| {
        let mut s = 0.0;
        for l in 0..dim {
            s += m[(i, l)] * m[(j, l)];
        }
        s / 400.0
    });

    let ma = |cs: &[Mat<f64>]| -> Vec<Mat<f64>> {
        let mut psi = vec![Mat::<f64>::identity(N, N)];
        for h in 1..=H {
            let mut acc = Mat::<f64>::zeros(N, N);
            for l in 1..=h.min(P) {
                for r in 0..N {
                    for c in 0..N {
                        let mut s = 0.0;
                        for q in 0..N {
                            s += psi[h - l][(r, q)] * cs[l - 1][(q, c)];
                        }
                        acc[(r, c)] += s;
                    }
                }
            }
            psi.push(acc);
        }
        psi
    };
    let psi = ma(&coefs);
    let got = psi_reduced_form_cov(&psi, &coefs, cov_alpha.as_ref(), &gamma, n_proxy)?;

    // Numerical Jacobian J[h][(i, r)] = d (Psi_h gamma)_i / d alpha_r, with
    // r = a*N + e, a = lag*N + variable (the regressor), e the equation, so
    // alpha_r lives at coefs[a / N][(e, a % N)].
    let eps = 1e-6;
    let mut jac = vec![Mat::<f64>::zeros(N, dim); H + 1];
    for r in 0..dim {
        let (a, e) = (r / N, r % N);
        let (lag, var) = (a / N, a % N);
        let mut up = coefs.clone();
        up[lag][(e, var)] += eps;
        let mut dn = coefs.clone();
        dn[lag][(e, var)] -= eps;
        let (pu, pd) = (ma(&up), ma(&dn));
        for h in 0..=H {
            for i in 0..N {
                let mut su = 0.0;
                let mut sd = 0.0;
                for (j, &g) in gamma.iter().enumerate() {
                    su += pu[h][(i, j)] * g;
                    sd += pd[h][(i, j)] * g;
                }
                jac[h][(i, r)] = (su - sd) / (2.0 * eps);
            }
        }
    }
    for h in 0..=H {
        for i in 0..N {
            for j in 0..N {
                let mut s = 0.0;
                for r in 0..dim {
                    for c in 0..dim {
                        s += jac[h][(i, r)] * cov_alpha[(r, c)] * jac[h][(j, c)];
                    }
                }
                let want = n_proxy as f64 * s;
                assert!(
                    (got[h][(i, j)] - want).abs() <= 1e-6 * (1.0 + want.abs()),
                    "h={h} ({i},{j}): analytic {} vs numerical {want}",
                    got[h][(i, j)]
                );
            }
        }
    }
    // Psi_0 = I is not estimated, so its correction is exactly zero.
    for i in 0..N {
        for j in 0..N {
            assert_eq!(got[0][(i, j)], 0.0);
        }
    }
    // The diagonal is a variance and the matrix is symmetric.
    for (h, cov) in got.iter().enumerate().skip(1) {
        for i in 0..N {
            assert!(cov[(i, i)] > 0.0, "h={h} i={i} has no variance");
            for j in 0..N {
                assert!((cov[(i, j)] - cov[(j, i)]).abs() < 1e-12);
            }
        }
    }
    Ok(())
}

/// The reduced-form correction is quadratic in `gamma`, so it dies with the
/// instrument's relevance. That is what keeps the set weak-IV robust; a term
/// scaled by the NORMALIZED impact `b` instead would not vanish and the set
/// would stay bounded where no bounded set can be valid.
#[test]
fn reduced_form_correction_vanishes_with_gamma() -> Result<(), IdentError> {
    const N: usize = 2;
    let coefs = vec![Mat::from_fn(N, N, |i, j| 0.4 - 0.1 * (i + j) as f64)];
    let mut psi = vec![Mat::<f64>::identity(N, N)];
    for h in 1..=4 {
        let mut acc = Mat::<f64>::zeros(N, N);
        for r in 0..N {
            for c in 0..N {
                let mut s = 0.0;
                for q in 0..N {
                    s += psi[h - 1][(r, q)] * coefs[0][(q, c)];
                }
                acc[(r, c)] = s;
            }
        }
        psi.push(acc);
    }
    let dim = N * N;
    let cov_alpha = Mat::from_fn(dim, dim, |i, j| if i == j { 0.01 } else { 0.002 });
    let base = [0.6f64, -0.25];
    let mut prev = f64::INFINITY;
    for &s in &[1.0f64, 0.1, 0.01] {
        let gamma: Vec<f64> = base.iter().map(|g| g * s).collect();
        let cov = psi_reduced_form_cov(&psi, &coefs, cov_alpha.as_ref(), &gamma, 200)?;
        let v = cov[4][(0, 0)];
        if prev.is_finite() {
            // Scaling gamma by 0.1 must scale the correction by 0.01.
            assert!(
                (v / prev - 0.01).abs() < 1e-9,
                "the correction is not quadratic in gamma: ratio {}",
                v / prev
            );
        }
        prev = v;
    }
    Ok(())
}

/// The new inputs are shape- and domain-checked rather than read blindly.
#[test]
fn reduced_form_guards_fire() {
    let case = &cases()[0];
    let n = case.u.ncols();
    let short = vec![Mat::<f64>::zeros(n, n); 2];
    assert!(matches!(
        proxy_ar_sets(
            case.u.as_ref(),
            &case.proxy,
            &case.psi,
            case.norm_var,
            case.unit,
            ArVarianceSpec::with_reduced_form(
                ArVariance::Hc0,
                ArReducedForm {
                    psi_var: &short,
                    psi_gamma_cov: None
                }
            ),
            case.critical,
        ),
        Err(IdentError::Dimension { .. })
    ));
    let mut negative = vec![Mat::<f64>::zeros(n, n); case.psi.len()];
    negative[1][(0, 0)] = -1.0;
    assert!(matches!(
        proxy_ar_sets(
            case.u.as_ref(),
            &case.proxy,
            &case.psi,
            case.norm_var,
            case.unit,
            ArVarianceSpec::with_reduced_form(
                ArVariance::Hc0,
                ArReducedForm {
                    psi_var: &negative,
                    psi_gamma_cov: None
                }
            ),
            case.critical,
        ),
        Err(IdentError::InvalidArgument { .. })
    ));
    let mut nan = vec![Mat::<f64>::zeros(n, n); case.psi.len()];
    nan[0][(0, 0)] = f64::NAN;
    assert!(matches!(
        proxy_ar_sets(
            case.u.as_ref(),
            &case.proxy,
            &case.psi,
            case.norm_var,
            case.unit,
            ArVarianceSpec::with_reduced_form(
                ArVariance::Hc0,
                ArReducedForm {
                    psi_var: &nan,
                    psi_gamma_cov: None
                }
            ),
            case.critical,
        ),
        Err(IdentError::NonFinite { .. })
    ));
    // A cross-covariance large enough to break V(lam) >= 0 is refused, not
    // silently turned into an "interval" where the statistic is negative.
    let big = vec![Mat::<f64>::from_fn(n, n, |_, _| 1e6); case.psi.len()];
    let zero = vec![Mat::<f64>::zeros(n, n); case.psi.len()];
    assert!(matches!(
        proxy_ar_sets(
            case.u.as_ref(),
            &case.proxy,
            &case.psi,
            case.norm_var,
            case.unit,
            ArVarianceSpec::with_reduced_form(
                ArVariance::Hc0,
                ArReducedForm {
                    psi_var: &zero,
                    psi_gamma_cov: Some(&big)
                }
            ),
            case.critical,
        ),
        Err(IdentError::InvalidArgument { .. })
    ));
    // And psi_reduced_form_cov's own shape guard.
    let coefs = vec![Mat::<f64>::zeros(n, n)];
    assert!(matches!(
        psi_reduced_form_cov(
            &case.psi,
            &coefs,
            Mat::<f64>::zeros(3, 3).as_ref(),
            &vec![0.0; n],
            10
        ),
        Err(IdentError::Dimension { .. })
    ));
}

/// An indefinite `Omega` breaks the quadratic-set structure and is refused
/// with an error that names the fix, rather than producing an "interval"
/// where the statistic is negative.
#[test]
fn indefinite_omega_is_refused() {
    let case = &cases()[0];
    let n = case.u.ncols();
    // Negate one diagonal entry: still symmetric, no longer PSD.
    let bad = Mat::from_fn(n, n, |i, j| {
        if i == j && i != case.norm_var {
            -1.0
        } else if i == j {
            1.0
        } else {
            0.9
        }
    });
    assert!(matches!(
        proxy_ar_sets(
            case.u.as_ref(),
            &case.proxy,
            &case.psi,
            case.norm_var,
            case.unit,
            ArVariance::Supplied(bad.as_ref()),
            case.critical,
        ),
        Err(IdentError::InvalidArgument { .. })
    ));
}

/// `ar_cell` is the same taxonomy in a reusable form, so a caller that has
/// corrected variance terms gets identical branch logic.
#[test]
fn ar_cell_reproduces_the_loop() -> Result<(), IdentError> {
    let case = &cases()[0];
    let res = case.run()?;
    for row in &res.cells {
        for cell in row {
            let rebuilt = ar_cell(
                res.n_proxy,
                res.critical_value,
                ArMoments {
                    q1: cell.q1,
                    q0: cell.q0,
                    v0: cell.v0,
                    v1: cell.v1,
                    v2: cell.v2,
                    point: cell.point,
                },
            )?;
            assert_eq!(rebuilt.set, cell.set);
            assert_eq!(rebuilt.a, cell.a);
            assert_eq!(rebuilt.b, cell.b);
            assert_eq!(rebuilt.c, cell.c);
            assert_eq!(rebuilt.excludes_zero, cell.excludes_zero);
        }
    }
    Ok(())
}

/// Domain and dimension guards fire with the right error variant.
#[test]
fn guards_fire() {
    let case = &cases()[0];
    let n = case.u.ncols();
    let ok = ArCritical::Chi2 { level: 0.95 };
    // proxy length mismatch.
    assert!(matches!(
        proxy_ar_sets(
            case.u.as_ref(),
            &case.proxy[1..],
            &case.psi,
            case.norm_var,
            case.unit,
            ArVariance::Hc0,
            ok
        ),
        Err(IdentError::Dimension { .. })
    ));
    // norm_var out of range.
    assert!(matches!(
        proxy_ar_sets(
            case.u.as_ref(),
            &case.proxy,
            &case.psi,
            n + 3,
            case.unit,
            ArVariance::Hc0,
            ok
        ),
        Err(IdentError::RestrictionOutOfRange { .. })
    ));
    // zero unit, out-of-range level, non-positive critical value, oversized
    // HAC bandwidth.
    for bad in [
        proxy_ar_sets(
            case.u.as_ref(),
            &case.proxy,
            &case.psi,
            case.norm_var,
            0.0,
            ArVariance::Hc0,
            ok,
        ),
        proxy_ar_sets(
            case.u.as_ref(),
            &case.proxy,
            &case.psi,
            case.norm_var,
            case.unit,
            ArVariance::Hc0,
            ArCritical::Chi2 { level: 1.0 },
        ),
        proxy_ar_sets(
            case.u.as_ref(),
            &case.proxy,
            &case.psi,
            case.norm_var,
            case.unit,
            ArVariance::Hc0,
            ArCritical::Value(-1.0),
        ),
        proxy_ar_sets(
            case.u.as_ref(),
            &case.proxy,
            &case.psi,
            case.norm_var,
            case.unit,
            ArVariance::HacBartlett { lags: 100_000 },
            ok,
        ),
    ] {
        assert!(matches!(bad, Err(IdentError::InvalidArgument { .. })));
    }
    // An all-NaN proxy leaves no overlap.
    let none = vec![f64::NAN; case.proxy.len()];
    assert!(matches!(
        proxy_ar_sets(
            case.u.as_ref(),
            &none,
            &case.psi,
            case.norm_var,
            case.unit,
            ArVariance::Hc0,
            ok
        ),
        Err(IdentError::InvalidArgument { .. })
    ));
}

/// RANDOMIZED never-empty sweep. The fixture cases pin the claim on four
/// datasets; this pins it on several hundred, across instrument strengths
/// from irrelevant to strong, dimensions, sample sizes and both signs of
/// `unit`. The set of a just-identified single-instrument problem contains its
/// own point estimate by construction — `A*lam_hat^2 + B*lam_hat + C =
/// -c*V(lam_hat) <= 0` — so `Empty` must never appear, and neither must the
/// negative-discriminant error that a non-PSD variance would trigger.
#[test]
fn never_empty_over_randomized_datasets() {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        // xorshift64*, adequate for shaking out branch coverage.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / ((1u64 << 53) as f64)
    };
    let mut normal = move || {
        // Box-Muller from two uniforms; exact distribution is irrelevant here.
        let (u1, u2) = (next().max(1e-12), next());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };

    let mut shapes: Vec<&'static str> = Vec::new();
    for trial in 0..300 {
        let n = 2 + trial % 3;
        let t = 20 + (trial * 7) % 120;
        let horizon = trial % 4;
        let norm_var = trial % n;
        let unit = if trial % 3 == 0 { -1.75 } else { 1.0 };
        // Relevance sweeps from exactly zero (an irrelevant instrument) up.
        let phi = (trial % 10) as f64 * 0.25;

        let u = Mat::from_fn(t, n, |_, _| normal());
        let proxy: Vec<f64> = (0..t)
            .map(|r| {
                let m = phi * u[(r, norm_var)] + normal();
                // Sprinkle unavailability, including interior gaps.
                if (r * 13 + trial) % 11 == 0 {
                    f64::NAN
                } else {
                    m
                }
            })
            .collect();
        let psi: Vec<Mat<f64>> = (0..=horizon)
            .map(|h| {
                if h == 0 {
                    Mat::<f64>::identity(n, n)
                } else {
                    Mat::from_fn(n, n, |_, _| 0.4 * normal())
                }
            })
            .collect();
        let variance = if trial % 4 == 3 {
            ArVariance::HacBartlett { lags: 2 }
        } else {
            ArVariance::Hc0
        };
        let critical = if trial % 5 == 4 {
            ArCritical::F { level: 0.90 }
        } else {
            ArCritical::Chi2 { level: 0.95 }
        };

        let res = match proxy_ar_sets(u.as_ref(), &proxy, &psi, norm_var, unit, variance, critical)
        {
            Ok(r) => r,
            // Only the documented degeneracies (too few finite proxy values,
            // a zero denominator moment) may refuse; a negative discriminant
            // or a PSD failure must not happen with the built-in estimators.
            Err(IdentError::InvalidArgument { what }) => {
                assert!(
                    what.contains("fewer than 3") || what.contains("no first-stage relevance"),
                    "trial {trial} refused for an unexpected reason: {what}"
                );
                continue;
            }
            Err(e) => panic!("trial {trial} failed: {e}"),
        };

        for (h, row) in res.cells.iter().enumerate() {
            for (i, cell) in row.iter().enumerate() {
                let k = kind_str(&cell.set);
                if !shapes.contains(&k) {
                    shapes.push(k);
                }
                assert_ne!(k, "empty", "trial {trial}: empty set at (h={h}, i={i})");
                let inside = match cell.set {
                    ArSet::Point(p) => (p - cell.point).abs() <= 1e-10 * (1.0 + p.abs()),
                    other => other.contains(cell.point),
                };
                assert!(
                    inside,
                    "trial {trial}: point {} outside its own {k} set at (h={h}, i={i})",
                    cell.point
                );
            }
        }
    }
    // The sweep must actually reach the unbounded regime, or it proves little.
    for needed in ["interval", "point", "exterior", "whole"] {
        assert!(
            shapes.contains(&needed),
            "randomized sweep never produced {needed}: {shapes:?}"
        );
    }
}
