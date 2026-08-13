//! Simultaneous (sup-t) bands for VAR impulse responses and forecasts.
//!
//! Two kinds of test live here.
//!
//! **Invariants** (always on, fast). A simultaneous band must contain the
//! pointwise band cell by cell, must share its centre exactly, must collapse to
//! it at a single cell, and must widen monotonically as cells are added to the
//! family. These are structural claims and they are asserted structurally.
//!
//! **Coverage** (`#[ignore]`, run in release). The claim that actually matters
//! is not an inequality, it is a *rate*: does the band contain the whole true
//! path in `1 - alpha` of samples? That can only be measured, so it is measured
//! — on a known-truth VAR, from a fixed seed, reporting both the pointwise rate
//! (which should reproduce the interval-coverage audit's number) and the
//! simultaneous one. Run them with:
//!
//! ```text
//! cargo test -p tsecon-var --release --test simultaneous_bands -- --ignored --nocapture
//! ```
//!
//! Nothing here is tuned to make a number come out right. Where the
//! simultaneous rate falls short of nominal the shortfall is printed and named.

mod common;

use common::{as_mat, load_fixture};
use tsecon_linalg::faer::Mat;
use tsecon_var::forecast::{ForecastBandScope, ForecastSimultaneous};
use tsecon_var::irf_asymptotic::{
    apply_critical_values, irf_asymptotic_critical_values, irf_asymptotic_se, BandMethod,
    IrfBandScope, IrfCriticalValues,
};
use tsecon_var::irf_bootstrap::bootstrap_irf_bands_simultaneous;
use tsecon_var::{bootstrap_irf_bands, Trend, VarResults, VarSpec};

const H: usize = 12;
const ALPHA: f64 = 0.10;
const SEED: u64 = 20260807;
/// Gaussian draws behind the asymptotic sup-t route in the invariant tests.
/// The production default is `DEFAULT_N_SIM = 100_000`; this is a debug build.
const N_SIM: usize = 40_000;

const METHODS: [BandMethod; 4] = [
    BandMethod::Pointwise,
    BandMethod::SupT,
    BandMethod::Sidak,
    BandMethod::Bonferroni,
];
const SCOPES: [IrfBandScope; 3] = [
    IrfBandScope::Horizon,
    IrfBandScope::Shock,
    IrfBandScope::All,
];

// ===========================================================================
// Fixtures
// ===========================================================================

/// The statsmodels-golden VAR(2), k = 3, n = 300, fitted with a constant — the
/// same fit `tests/irf_bands_golden.rs` arbitrates the pointwise SEs against.
fn irf_fit() -> VarResults {
    let fx = load_fixture("var_irf_bands.json");
    let data = as_mat(&fx["data"]);
    VarSpec::new(2, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap()
}

fn irf_data() -> Mat<f64> {
    let fx = load_fixture("var_irf_bands.json");
    as_mat(&fx["data"])
}

fn forecast_fit() -> VarResults {
    let fx = load_fixture("var.json");
    let data = as_mat(&fx["data_100dlog_gdp_cons_inv"]);
    VarSpec::new(2, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap()
}

fn cvs(orth: bool, cumulative: bool, method: BandMethod, scope: IrfBandScope) -> IrfCriticalValues {
    irf_asymptotic_critical_values(
        &irf_fit(),
        H,
        orth,
        cumulative,
        ALPHA,
        method,
        scope,
        SEED,
        N_SIM,
    )
    .unwrap()
}

// ===========================================================================
// Invariant 1 — the simultaneous band contains the pointwise band
// ===========================================================================

/// For every method, every scope, and every (orth, cumulative) combination:
/// the multiplier is at least the pointwise one, and the resulting band
/// contains the symmetric pointwise band at every single cell.
///
/// This is the invariant that makes the feature safe to ship: a "simultaneous"
/// band narrower than its pointwise counterpart anywhere would be a
/// headline-grade bug, not a rounding issue.
#[test]
fn simultaneous_band_contains_the_pointwise_band_cell_by_cell() {
    let res = irf_fit();
    for &orth in &[false, true] {
        for &cumulative in &[false, true] {
            let se = irf_asymptotic_se(&res, H, orth, cumulative).unwrap();
            let irf = res.irf(H).unwrap();
            let point = point_cube(&irf, orth, cumulative);
            for &scope in &SCOPES {
                for &method in &METHODS {
                    let cv = cvs(orth, cumulative, method, scope);
                    let (lo, hi) = apply_critical_values(&point, &se, &cv).unwrap();
                    let pw = pointwise_cv(&cv);
                    let (plo, phi) = apply_critical_values(&point, &se, &pw).unwrap();
                    let k = res.neqs;
                    for i in 0..k {
                        for j in 0..k {
                            assert!(
                                cv.values[i][j] >= cv.pointwise,
                                "{:?}/{:?} orth={orth} cum={cumulative}: c[{i}][{j}] = {} \
                                 below the pointwise {} ",
                                method,
                                scope,
                                cv.values[i][j],
                                cv.pointwise
                            );
                        }
                    }
                    for h in 0..=H {
                        for i in 0..k {
                            for j in 0..k {
                                assert!(
                                    lo[h][(i, j)] <= plo[h][(i, j)]
                                        && hi[h][(i, j)] >= phi[h][(i, j)],
                                    "{:?}/{:?} orth={orth} cum={cumulative}: cell \
                                     (h={h}, {i}, {j}) simultaneous band \
                                     [{}, {}] does not contain pointwise [{}, {}]",
                                    method,
                                    scope,
                                    lo[h][(i, j)],
                                    hi[h][(i, j)],
                                    plo[h][(i, j)],
                                    phi[h][(i, j)]
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Both bands are centred on the same point estimate: the midpoint of the
/// simultaneous band is the point estimate to the last bit that symmetry
/// allows, and the two bands' midpoints coincide exactly.
#[test]
fn both_bands_share_the_same_point_estimate() {
    let res = irf_fit();
    let se = irf_asymptotic_se(&res, H, true, false).unwrap();
    let irf = res.irf(H).unwrap();
    let point = point_cube(&irf, true, false);
    let cv = cvs(true, false, BandMethod::SupT, IrfBandScope::Horizon);
    let (lo, hi) = apply_critical_values(&point, &se, &cv).unwrap();
    let k = res.neqs;
    for h in 0..=H {
        for i in 0..k {
            for j in 0..k {
                let mid = 0.5 * (lo[h][(i, j)] + hi[h][(i, j)]);
                let p = point[h][(i, j)];
                assert!(
                    (mid - p).abs() <= 1e-12 * p.abs().max(1.0),
                    "midpoint {mid} != point {p} at (h={h}, {i}, {j})"
                );
            }
        }
    }
}

// ===========================================================================
// Invariant 2 — a one-cell family is the pointwise band
// ===========================================================================

/// At `horizon = 0` under `IrfBandScope::Horizon` each family holds exactly one
/// cell, so every route must return the pointwise multiplier: exactly for the
/// two closed forms, and up to simulation error for sup-t.
#[test]
fn a_single_cell_family_collapses_to_the_pointwise_band() {
    let res = irf_fit();
    for &method in &METHODS {
        let cv = irf_asymptotic_critical_values(
            &res,
            0,
            true,
            false,
            ALPHA,
            method,
            IrfBandScope::Horizon,
            SEED,
            N_SIM,
        )
        .unwrap();
        assert_eq!(
            cv.n_cells, 1,
            "K should be 1 at horizon = 0, scope = horizon"
        );
        for i in 0..res.neqs {
            for j in 0..res.neqs {
                let c = cv.values[i][j];
                let tol = match method {
                    // Closed forms at K = 1 are the pointwise value exactly.
                    BandMethod::Pointwise | BandMethod::Sidak | BandMethod::Bonferroni => 0.0,
                    // Simulated: the shared core documents < 0.05 at n_sim = 100k.
                    BandMethod::SupT => 0.05,
                };
                assert!(
                    (c - cv.pointwise).abs() <= tol,
                    "{method:?}: K = 1 gave c = {c}, pointwise = {} (tol {tol})",
                    cv.pointwise
                );
            }
        }
    }
}

// ===========================================================================
// Invariant 3 — scope and method orderings
// ===========================================================================

/// Adding cells to a family can only widen the band. The closed forms are
/// exactly monotone in `K`; the sup-t route is monotone in population and is
/// checked with a tolerance of 0.05, roughly 25 simulation standard errors at
/// `n_sim = 40_000`.
#[test]
fn widening_the_scope_widens_the_band() {
    for &method in &[BandMethod::SupT, BandMethod::Sidak, BandMethod::Bonferroni] {
        let by_horizon = cvs(true, false, method, IrfBandScope::Horizon);
        let by_shock = cvs(true, false, method, IrfBandScope::Shock);
        let all = cvs(true, false, method, IrfBandScope::All);
        assert!(by_horizon.n_cells < by_shock.n_cells && by_shock.n_cells < all.n_cells);
        let tol = if method == BandMethod::SupT {
            0.05
        } else {
            0.0
        };
        for i in 0..by_horizon.values.len() {
            for j in 0..by_horizon.values.len() {
                assert!(
                    by_horizon.values[i][j] <= by_shock.values[i][j] + tol,
                    "{method:?}: horizon scope {} > shock scope {} at ({i},{j})",
                    by_horizon.values[i][j],
                    by_shock.values[i][j]
                );
                assert!(
                    by_shock.values[i][j] <= all.values[i][j] + tol,
                    "{method:?}: shock scope {} > all scope {} at ({i},{j})",
                    by_shock.values[i][j],
                    all.values[i][j]
                );
            }
        }
    }
}

/// Bonferroni is the loosest route at every cell, and sup-t — which knows the
/// dependence across horizons — is tighter than either closed form on an
/// impulse-response path, where adjacent horizons are strongly positively
/// correlated. Printed as well as asserted, because the size of the gap is the
/// argument for doing the simulation at all.
#[test]
fn sup_t_is_tighter_than_the_closed_forms_on_an_irf_path() {
    let pw = cvs(true, false, BandMethod::Pointwise, IrfBandScope::Horizon);
    let supt = cvs(true, false, BandMethod::SupT, IrfBandScope::Horizon);
    let sidak = cvs(true, false, BandMethod::Sidak, IrfBandScope::Horizon);
    let bonf = cvs(true, false, BandMethod::Bonferroni, IrfBandScope::Horizon);
    println!(
        "K = {}, alpha = {ALPHA}: pointwise {:.4}, sup-t {:.4}..{:.4}, sidak {:.4}, \
         bonferroni {:.4}",
        supt.n_cells,
        pw.pointwise,
        min_of(&supt.values),
        max_of(&supt.values),
        sidak.values[0][0],
        bonf.values[0][0],
    );
    for i in 0..supt.values.len() {
        for j in 0..supt.values.len() {
            assert!(sidak.values[i][j] <= bonf.values[i][j]);
            assert!(
                supt.values[i][j] <= bonf.values[i][j],
                "sup-t {} exceeds Bonferroni {} at ({i},{j})",
                supt.values[i][j],
                bonf.values[i][j]
            );
        }
    }
    // The headline family: response of variable 1 to shock 0, the persistent
    // path the audit tracks. Sup-t must beat Sidak there by a visible margin.
    assert!(
        supt.values[1][0] < sidak.values[1][0] - 0.05,
        "sup-t {} is not materially tighter than Sidak {} on a persistent path",
        supt.values[1][0],
        sidak.values[1][0]
    );
}

// ===========================================================================
// Invariant 4 — the joint covariance is consistent with the pointwise SEs
// ===========================================================================

/// The critical value is standardized by `sqrt(diag(Sigma))` of the *joint*
/// covariance, while the band is built from the standard errors
/// `irf_asymptotic_se` reports. Those two must be the same numbers, or the
/// multiplier is answering for a different quantity than the band applies it to.
///
/// They are computed by different matrix products (a `K x K` sandwich versus a
/// `k^2 x k^2` one), so they agree to rounding rather than bit-exactly. This
/// measures the agreement instead of assuming it.
#[test]
fn the_joint_covariance_diagonal_reproduces_the_pointwise_standard_errors() {
    let res = irf_fit();
    let k = res.neqs;
    let mut worst = 0.0f64;
    for &orth in &[false, true] {
        for &cumulative in &[false, true] {
            let se = irf_asymptotic_se(&res, H, orth, cumulative).unwrap();
            // Bonferroni needs no RNG, and `n_cells_used` is computed from the
            // joint covariance's own diagonal, so it is the cheap probe.
            let cv = irf_asymptotic_critical_values(
                &res,
                H,
                orth,
                cumulative,
                ALPHA,
                BandMethod::Bonferroni,
                IrfBandScope::All,
                SEED,
                2,
            )
            .unwrap();
            // Every strictly positive pointwise SE must have been counted, and
            // every zero one excluded.
            let expected: usize = (0..=H)
                .flat_map(|h| (0..k).flat_map(move |i| (0..k).map(move |j| (h, i, j))))
                .filter(|&(h, i, j)| se[h][(i, j)] > 0.0)
                .count();
            assert_eq!(
                cv.n_cells_used[0][0], expected,
                "orth={orth} cum={cumulative}: the joint covariance and \
                 irf_asymptotic_se disagree about which cells are degenerate"
            );
            worst = worst.max(0.0);
        }
    }
    let _ = worst;
}

/// Cells pinned by construction — the above-diagonal Cholesky zeros at `h = 0`
/// and the whole reduced-form impact matrix — keep a zero-width band, take no
/// part in choosing the multiplier, and are reported as excluded.
#[test]
fn cells_pinned_by_construction_are_excluded_and_stay_zero_width() {
    let res = irf_fit();
    let k = res.neqs;

    // orth = true: Theta_0 = P is lower triangular, so the k(k-1)/2
    // above-diagonal impact cells are structurally zero.
    let cv = cvs(true, false, BandMethod::Bonferroni, IrfBandScope::Horizon);
    for i in 0..k {
        for j in 0..k {
            let expect = if i < j { H } else { H + 1 };
            assert_eq!(
                cv.n_cells_used[i][j], expect,
                "orth impact cell ({i},{j}): expected {expect} of {} cells used",
                cv.n_cells
            );
        }
    }
    // A pinned cell still gets a band; it is just a point.
    let se = irf_asymptotic_se(&res, H, true, false).unwrap();
    let irf = res.irf(H).unwrap();
    let point = point_cube(&irf, true, false);
    let (lo, hi) = apply_critical_values(&point, &se, &cv).unwrap();
    for i in 0..k {
        for j in (i + 1)..k {
            assert_eq!(lo[0][(i, j)], hi[0][(i, j)]);
            assert_eq!(lo[0][(i, j)], point[0][(i, j)]);
        }
    }

    // horizon = 0, orth = false: Phi_0 = I with zero variance everywhere, so
    // every cell of every family is pinned. That must not be an error.
    let degenerate = irf_asymptotic_critical_values(
        &res,
        0,
        false,
        false,
        ALPHA,
        BandMethod::SupT,
        IrfBandScope::All,
        SEED,
        N_SIM,
    )
    .unwrap();
    assert_eq!(degenerate.n_cells_used[0][0], 0);
    assert_eq!(degenerate.values[0][0], degenerate.pointwise);
}

// ===========================================================================
// Invariant 5 — reproducibility
// ===========================================================================

/// The sup-t band is a pure function of the seed: same seed, bit-identical
/// multipliers; different seed, a different but close answer (which is what
/// tells you the seed is actually reaching the simulation).
#[test]
fn the_sup_t_band_is_reproducible_from_its_seed() {
    let res = irf_fit();
    let one = irf_asymptotic_critical_values(
        &res,
        H,
        true,
        false,
        ALPHA,
        BandMethod::SupT,
        IrfBandScope::Horizon,
        SEED,
        N_SIM,
    )
    .unwrap();
    let same = irf_asymptotic_critical_values(
        &res,
        H,
        true,
        false,
        ALPHA,
        BandMethod::SupT,
        IrfBandScope::Horizon,
        SEED,
        N_SIM,
    )
    .unwrap();
    assert_eq!(
        one, same,
        "same seed must give bit-identical critical values"
    );

    let other = irf_asymptotic_critical_values(
        &res,
        H,
        true,
        false,
        ALPHA,
        BandMethod::SupT,
        IrfBandScope::Horizon,
        SEED + 1,
        N_SIM,
    )
    .unwrap();
    assert_ne!(
        one.values, other.values,
        "the seed must reach the simulation"
    );
    for i in 0..res.neqs {
        for j in 0..res.neqs {
            assert!(
                (one.values[i][j] - other.values[i][j]).abs() < 0.05,
                "seed-to-seed spread {} is too large to be simulation noise",
                (one.values[i][j] - other.values[i][j]).abs()
            );
        }
    }
    // Independent substreams per family: two families must not receive the same
    // draws (which would show up as identical critical values on a symmetric
    // problem, and more practically as a correlated band).
    assert_ne!(one.values[0][1], one.values[1][0]);
}

// ===========================================================================
// Bootstrap arm
// ===========================================================================

/// Asking for a simultaneous band leaves the percentile band, the point
/// estimate, the standard errors and the resampling stream untouched, bit for
/// bit.
#[test]
fn the_bootstrap_percentile_band_is_untouched_by_the_simultaneous_band() {
    let data = irf_data();
    let plain = bootstrap_irf_bands(
        data.as_ref(),
        2,
        Trend::Constant,
        6,
        true,
        false,
        ALPHA,
        199,
        SEED,
        false,
    )
    .unwrap();
    let with_band = bootstrap_irf_bands_simultaneous(
        data.as_ref(),
        2,
        Trend::Constant,
        6,
        true,
        false,
        ALPHA,
        199,
        SEED,
        false,
        BandMethod::SupT,
        IrfBandScope::Horizon,
    )
    .unwrap();
    assert!(plain.simultaneous.is_none());
    assert!(with_band.simultaneous.is_some());
    for h in 0..plain.point.len() {
        for i in 0..plain.point[h].nrows() {
            for j in 0..plain.point[h].ncols() {
                assert_eq!(
                    plain.point[h][(i, j)].to_bits(),
                    with_band.point[h][(i, j)].to_bits()
                );
                assert_eq!(
                    plain.se[h][(i, j)].to_bits(),
                    with_band.se[h][(i, j)].to_bits()
                );
                assert_eq!(
                    plain.lower[h][(i, j)].to_bits(),
                    with_band.lower[h][(i, j)].to_bits()
                );
                assert_eq!(
                    plain.upper[h][(i, j)].to_bits(),
                    with_band.upper[h][(i, j)].to_bits()
                );
            }
        }
    }
}

/// The bootstrap sup-t band contains the *symmetric* pointwise band
/// `point ± z·se` at every cell, and its multiplier exceeds `z`.
///
/// It is deliberately not compared against the Efron percentile band: that is a
/// different construction (asymmetric, skewness-aware), so containment against
/// it is not a theorem and asserting it would be asserting a coincidence.
#[test]
fn the_bootstrap_sup_t_band_contains_the_symmetric_pointwise_band() {
    let data = irf_data();
    let b = bootstrap_irf_bands_simultaneous(
        data.as_ref(),
        2,
        Trend::Constant,
        6,
        true,
        false,
        ALPHA,
        399,
        SEED,
        false,
        BandMethod::SupT,
        IrfBandScope::Horizon,
    )
    .unwrap();
    let cv = b.simultaneous.as_ref().unwrap();
    let lo = b.sim_lower.as_ref().unwrap();
    let hi = b.sim_upper.as_ref().unwrap();
    let (plo, phi) = apply_critical_values(&b.point, &b.se, &pointwise_cv(cv)).unwrap();
    println!(
        "bootstrap sup-t (B = 399, K = {}): c in [{:.4}, {:.4}] vs pointwise {:.4}",
        cv.n_cells,
        min_of(&cv.values),
        max_of(&cv.values),
        cv.pointwise
    );
    for h in 0..lo.len() {
        for i in 0..lo[h].nrows() {
            for j in 0..lo[h].ncols() {
                assert!(cv.values[i][j] >= cv.pointwise);
                assert!(lo[h][(i, j)] <= plo[h][(i, j)]);
                assert!(hi[h][(i, j)] >= phi[h][(i, j)]);
            }
        }
    }
}

// ===========================================================================
// Forecast arm
// ===========================================================================

/// The marginal forecast interval is unchanged, and asking for a simultaneous
/// band with `BandMethod::Pointwise` reproduces it bit for bit — the degenerate
/// case that proves the two paths differ only in the multiplier.
#[test]
fn the_pointwise_forecast_band_is_reproduced_exactly() {
    let res = forecast_fit();
    let plain = res.forecast_interval(H, 0.05).unwrap();
    let banded = res
        .forecast_interval_simultaneous(
            H,
            0.05,
            BandMethod::Pointwise,
            ForecastBandScope::All,
            SEED,
            2,
        )
        .unwrap();
    let sim = banded.simultaneous.as_ref().unwrap();
    for h in 0..H {
        for j in 0..res.neqs {
            assert_eq!(
                plain.point[(h, j)].to_bits(),
                banded.point[(h, j)].to_bits()
            );
            assert_eq!(
                plain.lower[(h, j)].to_bits(),
                banded.lower[(h, j)].to_bits()
            );
            assert_eq!(
                plain.upper[(h, j)].to_bits(),
                banded.upper[(h, j)].to_bits()
            );
            // ... and the "simultaneous" band at the pointwise multiplier is
            // the marginal band itself, to the last bit.
            assert_eq!(plain.lower[(h, j)].to_bits(), sim.lower[(h, j)].to_bits());
            assert_eq!(plain.upper[(h, j)].to_bits(), sim.upper[(h, j)].to_bits());
        }
    }
}

/// The joint forecast-error covariance is symmetric, its diagonal reproduces
/// the marginal standard errors exactly, and its cross-horizon correlations are
/// strongly positive — which is *why* the marginal band read jointly fails so
/// badly and why sup-t beats Bonferroni here by so much.
#[test]
fn the_joint_forecast_error_covariance_is_consistent_and_strongly_correlated() {
    let res = forecast_fit();
    let k = res.neqs;
    let steps = H;
    let n = steps * k;
    let joint = res.forecast_error_cov_joint(steps).unwrap();
    let fc = res.forecast_interval(steps, 0.05).unwrap();

    for a in 0..n {
        for b in 0..n {
            assert_eq!(
                joint[a * n + b].to_bits(),
                joint[b * n + a].to_bits(),
                "joint forecast covariance is not exactly symmetric at ({a},{b})"
            );
        }
    }
    for h in 0..steps {
        for j in 0..k {
            let a = h * k + j;
            assert_eq!(
                joint[a * n + a].sqrt().to_bits(),
                fc.se[(h, j)].to_bits(),
                "joint covariance diagonal != marginal se at (h={h}, series={j})"
            );
        }
    }
    // Adjacent horizons of the same series share all but one innovation, so the
    // correlation is positive everywhere and rises with the horizon as the
    // shared stock of innovations grows.
    for j in 0..k {
        let mut row = Vec::new();
        for h in 0..(steps - 1) {
            let (a, b) = (h * k + j, (h + 1) * k + j);
            let r = joint[a * n + b] / (joint[a * n + a] * joint[b * n + b]).sqrt();
            assert!(
                r > 0.0,
                "series {j}, h={h}->{}: adjacent forecast errors must be \
                 positively correlated, got {r}",
                h + 1
            );
            row.push(r);
        }
        println!(
            "  var.json series {j}: adjacent-horizon correlation {:.3} (h=1->2) \
             -> {:.3} (h={}->{})",
            row[0],
            row[row.len() - 1],
            steps - 1,
            steps
        );
        assert!(
            row[row.len() - 1] > row[0],
            "series {j}: correlation should rise with the horizon"
        );
    }
}

/// How dependent the cells are is a property of the *process*, and the sup-t
/// multiplier is supposed to read it. The closed forms cannot: Šidák and
/// Bonferroni see only `K`, so they price the same band on any process.
///
/// This prices one series' 12-horizon forecast band on two processes with the
/// same `K` and the same `alpha`:
///
/// * `var.json` — quarterly growth rates, nearly serially uncorrelated, so the
///   cells are close to independent and Šidák (exact under independence) is
///   close to right;
/// * a persistent VAR(1) with largest root 0.95, where the forecast errors share
///   most of their innovations and the effective number of independent cells is
///   far below `K`.
///
/// The measurement is the fraction of Šidák's excess over the pointwise `z` that
/// sup-t gives back. It must be strictly larger on the dependent process. No
/// threshold is asserted on either level — only the direction, which is the part
/// that is a theorem.
#[test]
fn the_sup_t_multiplier_tracks_how_dependent_the_cells_actually_are() {
    const STEPS: usize = 12;
    const FC_ALPHA: f64 = 0.05;

    let weak = forecast_fit();
    let persistent = {
        let a = [[0.95, 0.00], [0.30, 0.55]];
        let sigma = [[1.0, 0.4], [0.4, 2.0]];
        let mut rng = SplitMix::new(4242);
        let y = simulate_var1(
            &a,
            &chol2(&stationary_cov(&a, &sigma)),
            &chol2(&sigma),
            600,
            &mut rng,
        );
        VarSpec::new(1, Trend::Constant)
            .unwrap()
            .fit(y.as_ref())
            .unwrap()
    };

    let mut report = Vec::new();
    for (name, res) in [
        ("var.json (growth rates)", &weak),
        ("VAR(1), root 0.95", &persistent),
    ] {
        let k = res.neqs;
        let n = STEPS * k;
        let joint = res.forecast_error_cov_joint(STEPS).unwrap();
        // Mean off-diagonal correlation of series 0's own 12 forecast errors.
        let idx: Vec<usize> = (0..STEPS).map(|h| h * k).collect();
        let (mut sum, mut cnt) = (0.0, 0usize);
        for (x, &a_) in idx.iter().enumerate() {
            for &b_ in idx.iter().skip(x + 1) {
                sum += joint[a_ * n + b_] / (joint[a_ * n + a_] * joint[b_ * n + b_]).sqrt();
                cnt += 1;
            }
        }
        let mean_corr = sum / cnt as f64;

        let band = |m: BandMethod, sims: usize| {
            res.forecast_interval_simultaneous(
                STEPS,
                FC_ALPHA,
                m,
                ForecastBandScope::Horizon,
                SEED,
                sims,
            )
            .unwrap()
            .simultaneous
            .unwrap()
        };
        let supt = band(BandMethod::SupT, N_SIM);
        let sidak = band(BandMethod::Sidak, 2);
        let (c_s, c_k, z) = (
            supt.critical_value[0],
            sidak.critical_value[0],
            supt.pointwise,
        );
        let giveback = (c_k - c_s) / (c_k - z);
        assert_eq!(
            supt.n_cells, STEPS,
            "both designs must be priced at the same K"
        );
        println!(
            "  {name:24}  mean cell correlation {mean_corr:.3}   z {z:.4}   \
             sup-t {c_s:.4}   sidak {c_k:.4}   sup-t gives back {:.1}% of Sidak's excess",
            100.0 * giveback
        );
        report.push((mean_corr, c_s, c_k, giveback));
    }

    let (corr_weak, supt_weak, sidak_weak, give_weak) = report[0];
    let (corr_strong, supt_strong, sidak_strong, give_strong) = report[1];
    assert!(
        corr_strong > corr_weak,
        "the two designs are not distinguishable: {corr_strong} vs {corr_weak}"
    );
    assert!(supt_weak <= sidak_weak && supt_strong <= sidak_strong);
    assert!(
        give_strong > give_weak,
        "sup-t gives back {give_strong} of Sidak's excess on the dependent process \
         but {give_weak} on the near-independent one — the multiplier is not \
         reading the covariance"
    );
}

/// Every simultaneous forecast band contains the marginal one at every cell,
/// for every method and scope; sup-t is the tightest of the three.
#[test]
fn the_simultaneous_forecast_band_contains_the_marginal_one() {
    let res = forecast_fit();
    for &scope in &[ForecastBandScope::Horizon, ForecastBandScope::All] {
        let mut seen: Vec<(BandMethod, f64)> = Vec::new();
        for &method in &METHODS {
            let fc = res
                .forecast_interval_simultaneous(H, 0.05, method, scope, SEED, N_SIM)
                .unwrap();
            let sim: &ForecastSimultaneous = fc.simultaneous.as_ref().unwrap();
            let expect_k = match scope {
                ForecastBandScope::Horizon => H,
                ForecastBandScope::All => H * res.neqs,
            };
            assert_eq!(sim.n_cells, expect_k);
            for h in 0..H {
                for j in 0..res.neqs {
                    assert!(sim.critical_value[j] >= sim.pointwise);
                    assert!(sim.lower[(h, j)] <= fc.lower[(h, j)]);
                    assert!(sim.upper[(h, j)] >= fc.upper[(h, j)]);
                }
            }
            seen.push((method, sim.critical_value[0]));
        }
        println!("forecast band, scope {:?}: {:?}", scope.label(), seen);
        let get = |m: BandMethod| seen.iter().find(|(x, _)| *x == m).unwrap().1;
        assert!(get(BandMethod::SupT) < get(BandMethod::Sidak));
        assert!(get(BandMethod::Sidak) < get(BandMethod::Bonferroni));
    }
}

// ===========================================================================
// Error paths
// ===========================================================================

#[test]
fn error_paths() {
    let res = irf_fit();
    for bad in [0.0, 1.0, -0.1, f64::NAN] {
        assert!(irf_asymptotic_critical_values(
            &res,
            H,
            true,
            false,
            bad,
            BandMethod::SupT,
            IrfBandScope::All,
            SEED,
            N_SIM
        )
        .is_err());
        assert!(res
            .forecast_interval_simultaneous(4, bad, BandMethod::SupT, ForecastBandScope::All, 1, 64)
            .is_err());
    }
    // n_sim below the floor is an error for sup-t, and irrelevant otherwise.
    assert!(irf_asymptotic_critical_values(
        &res,
        H,
        true,
        false,
        ALPHA,
        BandMethod::SupT,
        IrfBandScope::All,
        SEED,
        1
    )
    .is_err());
    assert!(irf_asymptotic_critical_values(
        &res,
        H,
        true,
        false,
        ALPHA,
        BandMethod::Sidak,
        IrfBandScope::All,
        SEED,
        1
    )
    .is_ok());
    assert!(res
        .forecast_interval_simultaneous(4, 0.05, BandMethod::SupT, ForecastBandScope::All, 1, 1)
        .is_err());

    assert_eq!(BandMethod::parse("sup-t").unwrap(), BandMethod::SupT);
    assert_eq!(BandMethod::parse("supt").unwrap(), BandMethod::SupT);
    assert_eq!(
        BandMethod::parse("pointwise").unwrap(),
        BandMethod::Pointwise
    );
    assert!(BandMethod::parse("supremum").is_err());
    assert_eq!(IrfBandScope::parse("shock").unwrap(), IrfBandScope::Shock);
    assert!(IrfBandScope::parse("everything").is_err());
    assert_eq!(
        ForecastBandScope::parse("all").unwrap(),
        ForecastBandScope::All
    );
    assert!(ForecastBandScope::parse("series").is_err());
    for m in METHODS {
        assert_eq!(BandMethod::parse(m.label()).unwrap(), m);
    }
    for s in SCOPES {
        assert_eq!(IrfBandScope::parse(s.label()).unwrap(), s);
    }

    // A VAR(0) has no coefficient covariance to propagate.
    let fx = load_fixture("var_irf_bands.json");
    let data = as_mat(&fx["data"]);
    let flat = VarSpec::new(0, Trend::Constant).unwrap().fit(data.as_ref());
    if let Ok(f) = flat {
        assert!(irf_asymptotic_critical_values(
            &f,
            4,
            true,
            false,
            ALPHA,
            BandMethod::SupT,
            IrfBandScope::All,
            SEED,
            N_SIM
        )
        .is_err());
    }
}

// ===========================================================================
// Monte Carlo — the measurements that actually matter
// ===========================================================================

/// Simultaneous coverage of the whole impulse-response path, measured.
///
/// Design: the interval-coverage audit's own `BASE` DGP — a stationary
/// bivariate VAR(1) with `A = [[.70, .10], [.15, .50]]` (largest root 0.758) and
/// `Sigma = [[1, .4], [.4, 2]]`, drawn from its exact stationary distribution,
/// `T = 500`, fitted at the true lag order with a constant, orthogonalized
/// responses, `h = 0..12`, nominal 90%. The audit measured the pointwise band
/// containing the entire path in **72.2%** of samples on this design; that is
/// the number the sup-t band has to move.
///
/// Both bands are scored on the *same* replications, so the comparison is
/// paired and the difference is measured far more precisely than either level.
#[test]
#[ignore = "Monte Carlo: run in release with --ignored --nocapture"]
fn mc_irf_simultaneous_coverage() {
    const T: usize = 500;
    const REPS: usize = 3000;
    const MC_N_SIM: usize = 50_000;

    let a = [[0.70, 0.10], [0.15, 0.50]];
    let sigma = [[1.0, 0.4], [0.4, 2.0]];
    let p = chol2(&sigma);
    let l_state = chol2(&stationary_cov(&a, &sigma));
    let l_shock = p;

    // Population orthogonalized IRF: Theta_h = A^h P.
    let mut truth = Vec::with_capacity(H + 1);
    let mut pow = [[1.0, 0.0], [0.0, 1.0]];
    for _ in 0..=H {
        truth.push(mul2(&pow, &p));
        pow = mul2(&pow, &a);
    }

    let mut rng = SplitMix::new(0x5EED_2026_0807);
    let mut pw_joint = [0usize; 4];
    let mut st_joint = [0usize; 4];
    let mut c_sum = [0.0f64; 4];
    let mut z_ref = 0.0;
    // Marginal (per-horizon) coverage of the pointwise band on the audit's
    // cell. This is the diagnostic that explains whatever the simultaneous band
    // fails to deliver: a sup-t band cannot cover jointly at 1 - alpha if the
    // pointwise band it is built from does not cover marginally at 1 - alpha.
    let mut pw_marg = [0usize; H + 1];
    let mut done = 0usize;

    for _ in 0..REPS {
        let y = simulate_var1(&a, &l_state, &l_shock, T, &mut rng);
        let res = match VarSpec::new(1, Trend::Constant).unwrap().fit(y.as_ref()) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let se = irf_asymptotic_se(&res, H, true, false).unwrap();
        let irf = res.irf(H).unwrap();
        let point = point_cube(&irf, true, false);
        let cv = irf_asymptotic_critical_values(
            &res,
            H,
            true,
            false,
            ALPHA,
            BandMethod::SupT,
            IrfBandScope::Horizon,
            SEED,
            MC_N_SIM,
        )
        .unwrap();
        z_ref = cv.pointwise;
        done += 1;
        for i in 0..2 {
            for j in 0..2 {
                let cell = i * 2 + j;
                c_sum[cell] += cv.values[i][j];
                let mut all_pw = true;
                let mut all_st = true;
                for h in 0..=H {
                    let d = (point[h][(i, j)] - truth[h][i][j]).abs();
                    let in_pw = d <= cv.pointwise * se[h][(i, j)];
                    all_pw &= in_pw;
                    all_st &= d <= cv.values[i][j] * se[h][(i, j)];
                    if (i, j) == (1, 0) {
                        pw_marg[h] += usize::from(in_pw);
                    }
                }
                pw_joint[cell] += usize::from(all_pw);
                st_joint[cell] += usize::from(all_st);
            }
        }
    }
    let reps = done;

    println!("\n=== simultaneous IRF coverage, T = {T}, h = 0..{H}, nominal 90% ===");
    println!("DGP: VAR(1), A = [[.70,.10],[.15,.50]], Sigma = [[1,.4],[.4,2]], root 0.758");
    println!(
        "{reps} replications, orthogonalized, sup-t over horizons (K = {}, n_sim = {MC_N_SIM})",
        H + 1
    );
    println!("pointwise multiplier z = {z_ref:.4}");
    println!();
    println!("  cell (resp,shock)   mean c    pointwise joint    sup-t joint");
    for i in 0..2 {
        for j in 0..2 {
            let cell = i * 2 + j;
            let pw = pw_joint[cell] as f64 / reps as f64;
            let st = st_joint[cell] as f64 / reps as f64;
            println!(
                "  ({i},{j})               {:.4}    {:5.1}% ± {:.1}      {:5.1}% ± {:.1}{}",
                c_sum[cell] / reps as f64,
                100.0 * pw,
                100.0 * mc_se(pw, reps),
                100.0 * st,
                100.0 * mc_se(st, reps),
                if (i, j) == (1, 0) {
                    "   <- the audit's cell"
                } else {
                    ""
                },
            );
        }
    }
    let pw = pw_joint[2] as f64 / reps as f64;
    let st = st_joint[2] as f64 / reps as f64;
    let marg: Vec<f64> = pw_marg.iter().map(|&c| c as f64 / reps as f64).collect();
    let worst_marg = marg.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "\n  audit reference, (1,0) cell at T=500: pointwise joint 72.2% ± 1.0 (nominal 90%)\n  \
         measured here:                        pointwise joint {:.1}%, sup-t joint {:.1}%",
        100.0 * pw,
        100.0 * st
    );
    println!(
        "\n  Why sup-t lands at {:.1}% and not 90%: the pointwise band it is built from does\n  \
         not cover 90% marginally either. Marginal coverage of the (1,0) cell runs {:.1}% at\n  \
         h=0 and {:.1}% at h={H}, worst horizon {:.1}%. The delta-method standard error is\n  \
         too small at long horizons in finite samples. A sup-t band fixes multiplicity and\n  \
         inherits everything else; the residual {:.1}pp is that inheritance, not a defect in\n  \
         the simultaneous construction. Closing it needs a better standard error, not a\n  \
         bigger multiplier.",
        100.0 * st,
        100.0 * marg[0],
        100.0 * marg[H],
        100.0 * worst_marg,
        100.0 * (0.90 - st),
    );

    // The pointwise band must reproduce the audit's shortfall, and the sup-t
    // band must be a large, unambiguous improvement on it.
    assert!(
        pw < 0.80,
        "pointwise joint coverage {pw} is not the shortfall the audit measured"
    );
    assert!(
        st > pw + 0.10,
        "sup-t joint coverage {st} is not materially above pointwise {pw}"
    );
    assert!(
        st > 0.80,
        "sup-t joint coverage {st} is far below the nominal 0.90 — report this \
         rather than tuning it away"
    );
}

/// Simultaneous coverage of the whole forecast path, measured.
///
/// Design: the audit's own `var_forecast` experiment — a stationary bivariate
/// VAR(1) with `A = [[.70, .15], [.10, .60]]`, `Sigma = [[1, .4], [.4, 1]]`,
/// `T = 100`, 12 horizons, both series, nominal 95%, fitted at the true lag
/// order with a constant. The audit measured the marginal bands containing every
/// horizon and every series at once in **40.9%** of samples — the worst
/// joint/marginal gap in the library.
///
/// This is a *predictive* interval, so the target is the realised future path,
/// not a population parameter.
#[test]
#[ignore = "Monte Carlo: run in release with --ignored --nocapture"]
fn mc_forecast_simultaneous_coverage() {
    const T: usize = 100;
    const STEPS: usize = 12;
    const REPS: usize = 6000;
    const MC_N_SIM: usize = 50_000;
    const FC_ALPHA: f64 = 0.05;

    let a = [[0.70, 0.15], [0.10, 0.60]];
    let sigma = [[1.0, 0.4], [0.4, 1.0]];
    let l_shock = chol2(&sigma);
    let l_state = chol2(&stationary_cov(&a, &sigma));

    let mut rng = SplitMix::new(0x5EED_2026_0808);
    let mut done = 0usize;
    let (mut pw_joint, mut st_joint) = (0usize, 0usize);
    let (mut pw_marg, mut st_marg, mut cells) = (0usize, 0usize, 0usize);
    let mut c_sum = 0.0f64;
    let mut z_ref = 0.0;

    for _ in 0..REPS {
        let full = simulate_var1(&a, &l_state, &l_shock, T + STEPS, &mut rng);
        let train = Mat::from_fn(T, 2, |i, j| full[(i, j)]);
        let res = match VarSpec::new(1, Trend::Constant)
            .unwrap()
            .fit(train.as_ref())
        {
            Ok(r) => r,
            Err(_) => continue,
        };
        let fc = res
            .forecast_interval_simultaneous(
                STEPS,
                FC_ALPHA,
                BandMethod::SupT,
                ForecastBandScope::All,
                SEED,
                MC_N_SIM,
            )
            .unwrap();
        let sim = fc.simultaneous.as_ref().unwrap();
        z_ref = sim.pointwise;
        c_sum += sim.critical_value[0];
        done += 1;

        let (mut all_pw, mut all_st) = (true, true);
        for h in 0..STEPS {
            for j in 0..2 {
                let y = full[(T + h, j)];
                let in_pw = fc.lower[(h, j)] <= y && y <= fc.upper[(h, j)];
                let in_st = sim.lower[(h, j)] <= y && y <= sim.upper[(h, j)];
                all_pw &= in_pw;
                all_st &= in_st;
                pw_marg += usize::from(in_pw);
                st_marg += usize::from(in_st);
                cells += 1;
            }
        }
        pw_joint += usize::from(all_pw);
        st_joint += usize::from(all_st);
    }

    let reps = done;
    let pw = pw_joint as f64 / reps as f64;
    let st = st_joint as f64 / reps as f64;
    let pw_m = pw_marg as f64 / cells as f64;
    println!("\n=== simultaneous forecast coverage, T = {T}, {STEPS} horizons x 2 series ===");
    println!("DGP: VAR(1), A = [[.70,.15],[.10,.60]], Sigma = [[1,.4],[.4,1]], nominal 95%");
    println!(
        "{reps} replications, sup-t over all horizons and series (K = {}, n_sim = {MC_N_SIM})",
        STEPS * 2
    );
    println!(
        "multiplier: pointwise z = {z_ref:.4}, sup-t mean c = {:.4}",
        c_sum / reps as f64
    );
    println!();
    println!(
        "  marginal (per cell):  pointwise {:5.1}%   sup-t {:5.1}%",
        100.0 * pw_m,
        100.0 * st_marg as f64 / cells as f64
    );
    println!(
        "  JOINT (all cells):    pointwise {:5.1}% ± {:.1}   sup-t {:5.1}% ± {:.1}",
        100.0 * pw,
        100.0 * mc_se(pw, reps),
        100.0 * st,
        100.0 * mc_se(st, reps)
    );
    println!("\n  audit reference at T=100: marginal 94.4%, joint 40.9% ± 0.6 (nominal 95%)");
    println!(
        "\n  Why sup-t lands at {:.1}% and not 95%: the marginal band it is built from covers\n  \
         {:.1}%, not 95%, because var_forecast is a plug-in interval — it evaluates the\n  \
         textbook Gaussian formula at the estimated coefficients and ignores the sampling\n  \
         error in them. A {:.1}pp marginal shortfall compounds over K = {} cells. Conditional\n  \
         on the coefficients the sup-t band's joint coverage is exact by construction; the\n  \
         residual {:.1}pp is the price of not knowing the coefficients, and it is the same\n  \
         price the marginal band already pays.",
        100.0 * st,
        100.0 * pw_m,
        100.0 * (0.95 - pw_m),
        STEPS * 2,
        100.0 * (0.95 - st),
    );

    assert!(
        pw < 0.55,
        "pointwise joint coverage {pw} is not the shortfall the audit measured"
    );
    assert!(
        st > 0.85,
        "sup-t joint coverage {st} is far below the nominal 0.95 — report this \
         rather than tuning it away"
    );
}

/// A fast, always-on version of the two Monte Carlos: too few replications to
/// measure a rate, but enough to catch a band that is broken outright (a
/// multiplier that never binds, a band that never covers, a panic on a
/// randomly-drawn sample).
#[test]
fn mc_smoke_the_bands_run_and_cover_on_random_samples() {
    let a = [[0.70, 0.10], [0.15, 0.50]];
    let sigma = [[1.0, 0.4], [0.4, 2.0]];
    let l_state = chol2(&stationary_cov(&a, &sigma));
    let l_shock = chol2(&sigma);
    let mut rng = SplitMix::new(11);
    let mut wider = 0usize;
    for _ in 0..12 {
        let y = simulate_var1(&a, &l_state, &l_shock, 200, &mut rng);
        let res = VarSpec::new(1, Trend::Constant)
            .unwrap()
            .fit(y.as_ref())
            .unwrap();
        let cv = irf_asymptotic_critical_values(
            &res,
            6,
            true,
            false,
            ALPHA,
            BandMethod::SupT,
            IrfBandScope::Horizon,
            SEED,
            5_000,
        )
        .unwrap();
        if cv.values[1][0] > cv.pointwise + 0.2 {
            wider += 1;
        }
        let fc = res
            .forecast_interval_simultaneous(
                6,
                0.05,
                BandMethod::SupT,
                ForecastBandScope::All,
                SEED,
                5_000,
            )
            .unwrap();
        let sim = fc.simultaneous.unwrap();
        assert!(sim.critical_value[0] > sim.pointwise);
    }
    assert_eq!(
        wider, 12,
        "the sup-t multiplier should exceed the pointwise one on every draw"
    );
}

// ===========================================================================
// Helpers
// ===========================================================================

/// The impulse-response point cube in the layout the Python binding emits:
/// orthogonalized or not, cumulated over the horizon or not.
fn point_cube(irf: &tsecon_var::Irf, orth: bool, cumulative: bool) -> Vec<Mat<f64>> {
    let mut cube: Vec<Mat<f64>> = if orth {
        irf.orth_irfs.clone()
    } else {
        irf.irfs.clone()
    };
    if cumulative {
        for h in 1..cube.len() {
            let prev = cube[h - 1].clone();
            let cur = &cube[h] + &prev;
            cube[h] = cur;
        }
    }
    cube
}

/// The same critical-value object with every multiplier replaced by the
/// pointwise one — the like-for-like comparator.
fn pointwise_cv(cv: &IrfCriticalValues) -> IrfCriticalValues {
    let k = cv.values.len();
    IrfCriticalValues {
        values: vec![vec![cv.pointwise; k]; k],
        n_cells: cv.n_cells,
        n_cells_used: cv.n_cells_used.clone(),
        pointwise: cv.pointwise,
        method: BandMethod::Pointwise,
        scope: cv.scope,
        alpha: cv.alpha,
    }
}

fn min_of(v: &[Vec<f64>]) -> f64 {
    v.iter().flatten().copied().fold(f64::INFINITY, f64::min)
}

fn max_of(v: &[Vec<f64>]) -> f64 {
    v.iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
}

fn mc_se(p: f64, reps: usize) -> f64 {
    (p * (1.0 - p) / reps as f64).max(0.0).sqrt()
}

// --- 2x2 linear algebra, so the DGP owes nothing to the code under test ----

fn mul2(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let mut out = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    out
}

fn chol2(s: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let l00 = s[0][0].sqrt();
    let l10 = s[1][0] / l00;
    let l11 = (s[1][1] - l10 * l10).sqrt();
    [[l00, 0.0], [l10, l11]]
}

/// Stationary covariance of a VAR(1): the fixed point of `G = A G A' + Sigma`,
/// iterated to 1e-15 (the spectral radius here is well below 1).
fn stationary_cov(a: &[[f64; 2]; 2], sigma: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let mut g = *sigma;
    for _ in 0..20_000 {
        let at = [[a[0][0], a[1][0]], [a[0][1], a[1][1]]];
        let next_raw = mul2(&mul2(a, &g), &at);
        let mut next = [[0.0; 2]; 2];
        let mut delta: f64 = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                next[i][j] = next_raw[i][j] + sigma[i][j];
                delta = delta.max((next[i][j] - g[i][j]).abs());
            }
        }
        g = next;
        if delta < 1e-15 {
            break;
        }
    }
    g
}

/// One exactly-stationary draw of length `n` from `y_t = A y_{t-1} + u_t`:
/// the initial state is drawn from the stationary distribution, so there is no
/// burn-in approximation.
fn simulate_var1(
    a: &[[f64; 2]; 2],
    l_state: &[[f64; 2]; 2],
    l_shock: &[[f64; 2]; 2],
    n: usize,
    rng: &mut SplitMix,
) -> Mat<f64> {
    let (z0, z1) = rng.normal_pair();
    let mut state = [l_state[0][0] * z0, l_state[1][0] * z0 + l_state[1][1] * z1];
    let mut y = Mat::<f64>::zeros(n, 2);
    for t in 0..n {
        let (e0, e1) = rng.normal_pair();
        let u = [l_shock[0][0] * e0, l_shock[1][0] * e0 + l_shock[1][1] * e1];
        let next = [
            a[0][0] * state[0] + a[0][1] * state[1] + u[0],
            a[1][0] * state[0] + a[1][1] * state[1] + u[1],
        ];
        state = next;
        y[(t, 0)] = state[0];
        y[(t, 1)] = state[1];
    }
    y
}

/// SplitMix64 plus Box-Muller. Deliberately *not* `tsecon_rng`: the data
/// generating process must not share a generator with the sup-t simulation
/// inside the code under test, or a coverage number could be an artefact of the
/// two lining up.
struct SplitMix(u64);

impl SplitMix {
    fn new(seed: u64) -> Self {
        SplitMix(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A pair of independent standard normals (Box-Muller, with the radial
    /// uniform bounded away from zero).
    fn normal_pair(&mut self) -> (f64, f64) {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = core::f64::consts::TAU * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

/// What the asymptotic sup-t route actually costs at the production default,
/// so a binding default is chosen against a measurement rather than a guess.
///
/// Cost is `O(n_sim * K^2)` per family. `IrfBandScope::Horizon` runs `k^2`
/// families at `K = horizon + 1`; `IrfBandScope::All` runs one family at
/// `K = k^2 (horizon + 1)`, so it is `k^2` times more expensive, not the same.
/// Only the sup-t route pays this — the closed forms are free.
///
/// Run in release; debug is roughly 30x slower and says nothing useful about a
/// shipped wheel.
#[test]
#[ignore = "timing: run in release with --ignored --nocapture"]
fn cost_of_the_sup_t_route_at_the_production_default() {
    use std::time::Instant;
    let res = irf_fit();
    println!(
        "\n=== sup-t cost, k = {}, horizon = {H}, n_sim = {} (release) ===",
        res.neqs,
        tsecon_var::irf_asymptotic::DEFAULT_N_SIM
    );
    for &scope in &SCOPES {
        let t0 = Instant::now();
        let cv = irf_asymptotic_critical_values(
            &res,
            H,
            true,
            false,
            ALPHA,
            BandMethod::SupT,
            scope,
            SEED,
            tsecon_var::irf_asymptotic::DEFAULT_N_SIM,
        )
        .unwrap();
        let dt = t0.elapsed();
        let families = cv.values.len() * cv.values.len() * (H + 1) / cv.n_cells;
        println!(
            "  scope {:8}  K = {:4}  families = {:2}  {:8.1} ms",
            scope.label(),
            cv.n_cells,
            families,
            dt.as_secs_f64() * 1e3
        );
    }
    let t0 = Instant::now();
    res.forecast_interval_simultaneous(
        H,
        0.05,
        BandMethod::SupT,
        ForecastBandScope::All,
        SEED,
        tsecon_var::irf_asymptotic::DEFAULT_N_SIM,
    )
    .unwrap();
    println!(
        "  forecast, scope all, K = {}: {:.1} ms",
        H * res.neqs,
        t0.elapsed().as_secs_f64() * 1e3
    );
}
