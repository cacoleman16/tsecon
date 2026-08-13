//! The `converged` flag, and the data-scale invariance it depends on.
//!
//! Two properties are asserted together, because neither is worth much
//! alone.
//!
//! 1. **`converged` carries information in both directions.** It used to be
//!    a constant `false`: the Nelder-Mead default evaluation budget was
//!    sized for one run and then shared across the three runs
//!    `restarts: 2` asks for, so every fit — however well-posed —
//!    terminated on `MaxFevals`. A user gating on the flag discarded every
//!    valid estimate, and the flag could never warn about a fit that had
//!    genuinely failed.
//!
//! 2. **The estimator does not change character with the units of `y`.**
//!    The flag is only meaningful if a converged fit is a *good* fit, and
//!    the filtered variance used to be floored at an absolute `1e-12` — a
//!    variance in units of `y^2`. Quote the same returns in units where
//!    `Var(y)` falls near that constant and the whole filtered path pinned
//!    to the floor, so the optimizer converged onto a plateau that was an
//!    artifact of the constant rather than a maximum of the likelihood.

use serde_json::Value;
use tsecon_gas::{Density, GasError, GasModel};

fn load_fixture() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-gas.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("read fixture");
    serde_json::from_str(&text).expect("parse fixture")
}

fn as_f64_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

/// A correctly specified fit on 4000 simulated observations is exactly the
/// case `converged` exists to certify, under both densities.
#[test]
fn fit_reports_convergence_when_the_problem_is_well_posed() {
    let fx = load_fixture();
    for (case, density) in [
        ("sim_gaussian", Density::Gaussian),
        ("sim_student_t", Density::StudentT),
    ] {
        let y = as_f64_vec(&fx[case]["y"]);
        let res = GasModel::new(&y, density).unwrap().fit().unwrap();
        assert!(
            res.converged,
            "{case} under {}: a well-posed ML fit reported converged = false \
             (iterations {}, fevals {})",
            density.name(),
            res.iterations,
            res.fevals
        );
        // A convergence certificate on a fit that is not actually at an
        // optimum would be worse than no certificate: sanity-check the
        // answer the flag is vouching for.
        assert!(res.loglik.is_finite());
        assert!(res.params.omega > 0.0 && res.params.a >= 0.0);
        assert!((0.0..1.0).contains(&res.params.b));
    }
}

/// ...and the flag is not simply pinned to `true` in its place: a fit whose
/// optimum sits at an open boundary of the parameter space must report
/// failure.
///
/// Fitting the Student-t model to data that is Gaussian by construction is
/// exactly that case. The Gaussian is the `nu -> infinity` limit of the
/// standardized t, so the likelihood has no interior maximum in `nu`; the
/// working coordinate `ln(nu - 2)` runs off to infinity and the simplex can
/// never collapse around it, at any budget. The estimates of the other
/// parameters are fine — which is the point. `converged` reports the
/// optimizer's certificate, not the quality of the fit, and here there
/// honestly is none.
#[test]
fn fit_does_not_claim_convergence_at_an_open_boundary() {
    let fx = load_fixture();
    let y = as_f64_vec(&fx["sim_gaussian"]["y"]);
    let res = GasModel::new(&y, Density::StudentT).unwrap().fit().unwrap();

    assert!(
        !res.converged,
        "Student-t fitted to Gaussian data claimed convergence with \
         nu = {:e}",
        res.params.nu
    );
    // The diagnosis: nu has run away toward the Gaussian limit.
    assert!(
        res.params.nu > 1e3,
        "nu = {} — expected the dof to diverge on Gaussian data",
        res.params.nu
    );
    // And the run really did exhaust its budget chasing it.
    assert!(!res.converged && res.fevals > 0);

    // Same data, correctly specified: the flag flips. Same series, same
    // code path, opposite answer — that is what makes the flag a signal.
    let gaussian = GasModel::new(&y, Density::Gaussian).unwrap().fit().unwrap();
    assert!(gaussian.converged);
    // The t-fit found the same variance dynamics, so the non-convergence
    // really is about nu alone.
    assert!((gaussian.params.b - res.params.b).abs() < 1e-3);
}

/// A series with no scale is diagnosed, not fitted.
///
/// Regression test. An all-zero series used to come back
/// `converged = true` with `omega = 5e-324`, `variance[0] = 4.4e-323`,
/// `loglik = +70673` and `aic = -141340` — a certificate, a variance, and
/// the best information criterion in the library, all on a series with no
/// variance in it. The likelihood is unbounded above there (with every
/// `y_t = 0` the Gaussian log-density is `-0.5 (ln 2 pi + ln f_t)`, which
/// diverges as `f_t -> 0`), so the optimizer was not wrong to keep pushing
/// `omega` down; what it found was the smallest subnormal the variance
/// floor's `.max(f64::MIN_POSITIVE)` backstop would let it reach. The
/// backstop is what made the answer look finite and therefore fitted.
/// Refusing the input at construction says the true thing: no
/// maximum-likelihood estimate exists.
#[test]
fn degenerate_series_is_refused_rather_than_certified() {
    for density in [Density::Gaussian, Density::StudentT] {
        let zeros = vec![0.0; 500];
        let err = GasModel::new(&zeros, density).unwrap_err();
        assert!(
            matches!(err, GasError::DegenerateSeries { .. }),
            "all-zero series under {}: {err}",
            density.name()
        );
        // The message has to name the cause, not just the refusal.
        let msg = err.to_string();
        assert!(msg.contains("every observation is exactly zero"), "{msg}");
        assert!(msg.contains("mean(y^2)"), "{msg}");
        assert!(msg.contains("unbounded"), "{msg}");

        // The same verdict when the series is nonzero but squares away:
        // `mean(y^2)` is what the model measures, and it is zero here too.
        let underflowed = vec![1e-200_f64; 500];
        let err = GasModel::new(&underflowed, density).unwrap_err();
        assert!(matches!(err, GasError::DegenerateSeries { .. }));
        assert!(
            err.to_string().contains("squares to zero"),
            "underflow case should name underflow, not zeros: {err}"
        );
    }

    // And the guard is not a blanket refusal of small numbers: a series a
    // hundred orders below unit scale still has a second moment, so it
    // still fits — and to the same answer, since the estimator is scale
    // equivariant.
    let fx = load_fixture();
    let y = as_f64_vec(&fx["sim_gaussian"]["y"]);
    let base = GasModel::new(&y, Density::Gaussian).unwrap().fit().unwrap();
    let tiny: Vec<f64> = y.iter().map(|v| v * 1e-100).collect();
    let res = GasModel::new(&tiny, Density::Gaussian)
        .expect("mean(y^2) = 1e-200 is small, not degenerate")
        .fit()
        .unwrap();
    assert!(res.params.omega > 0.0 && res.variance.iter().all(|&v| v > 0.0));
    assert!(
        (res.params.b - base.params.b).abs() <= 1e-5,
        "b = {} at scale 1e-100 vs {} at unit scale",
        res.params.b,
        base.params.b
    );
}

/// Rescaling the data must rescale the answer, not change it.
///
/// `y -> c y` implies `omega -> c^2 omega` with `a`, `b` and `nu`
/// unchanged (they are dimensionless), and `loglik -> loglik - N ln c`
/// from the Jacobian. Sweeping `c` over twelve orders of magnitude is the
/// only way to see a scale-carrying constant hiding in the recursion: a
/// single-scale test passes no matter what that constant is.
#[test]
fn fit_is_scale_equivariant_across_twelve_orders_of_magnitude() {
    let fx = load_fixture();
    for (case, density) in [
        ("sim_gaussian", Density::Gaussian),
        ("sim_student_t", Density::StudentT),
    ] {
        let y = as_f64_vec(&fx[case]["y"]);
        let n = y.len() as f64;
        let base = GasModel::new(&y, density).unwrap().fit().unwrap();
        assert!(base.converged, "{case}: unit-scale fit did not converge");

        for c in [1e-6_f64, 1e-3, 1e3, 1e6] {
            let scaled: Vec<f64> = y.iter().map(|v| v * c).collect();
            let res = GasModel::new(&scaled, density).unwrap().fit().unwrap();

            assert!(
                res.converged,
                "{case} at scale {c:e}: converged = false ({} iterations, \
                 {} fevals)",
                res.iterations, res.fevals
            );
            assert!(
                (res.params.a - base.params.a).abs() <= 1e-4 * base.params.a,
                "{case} at scale {c:e}: a = {} vs {} — the score loading is \
                 dimensionless and must not move with the units of y",
                res.params.a,
                base.params.a
            );
            assert!(
                (res.params.b - base.params.b).abs() <= 1e-5,
                "{case} at scale {c:e}: b = {} vs {}",
                res.params.b,
                base.params.b
            );
            let omega_unscaled = res.params.omega / (c * c);
            assert!(
                (omega_unscaled - base.params.omega).abs() <= 1e-4 * base.params.omega,
                "{case} at scale {c:e}: omega/c^2 = {omega_unscaled:e} vs {:e}",
                base.params.omega
            );
            if density.needs_dof() {
                assert!(
                    (res.params.nu - base.params.nu).abs() <= 1e-3 * base.params.nu,
                    "{case} at scale {c:e}: nu = {} vs {}",
                    res.params.nu,
                    base.params.nu
                );
            }
            // The likelihood shifts by exactly the Jacobian term.
            let expected = base.loglik - n * c.ln();
            assert!(
                (res.loglik - expected).abs() <= 1e-8 * expected.abs().max(1.0),
                "{case} at scale {c:e}: loglik = {} vs the Jacobian-implied \
                 {expected}",
                res.loglik
            );
            // The filtered variance path carries the units too, so the
            // standardized residuals are scale-free.
            for (t, (&r, &b)) in res.std_resid.iter().zip(&base.std_resid).enumerate() {
                assert!(
                    (r - b).abs() <= 1e-5 * b.abs().max(1.0),
                    "{case} at scale {c:e}: std_resid[{t}] = {r} vs {b}"
                );
            }
        }
    }
}
