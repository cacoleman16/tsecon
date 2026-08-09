//! What the parameter covariance is *accurate to*, as opposed to what it
//! agrees with.
//!
//! `golden_bse.rs` compares against statsmodels, which pins parity but
//! cannot catch a defect the two implementations share, and cannot say
//! anything at all outside the fixtures' parameter range (every fixture
//! has `sigma2` in `[0.94, 2e4]`). The tests here compare against closed
//! forms instead, over ranges no fixture covers:
//!
//! * [`bse_is_scale_free_across_fourteen_decades_of_sigma2`] — the exact
//!   `se(c) = sqrt(sigma2/n)`, `se(sigma2) = sqrt(2 sigma2^2 / n)` of
//!   ARIMA(0,1,0)+c, swept over series scales spanning `sigma2` from
//!   9.8e-9 to 9.8e5. This is the test that would have caught the
//!   absolute-floored `sigma2` step, which was 4.6% wrong at
//!   `sigma2 = 9.8e-5` and silent about it.
//! * [`bse_is_exact_far_below_the_old_absolute_floor`] — the same closed
//!   forms plus the `loglik(c y) = loglik(y) - n ln c` identity, carried
//!   down to `sigma2 = 9.8e-19`. This is the test that would have caught
//!   the state-space filter's absolute variance floor, which discarded
//!   every observation below `sigma2 ~ 1e-10` and returned `loglik = 0.0`
//!   as a success.
//! * [`css_standard_errors_match_ols_on_the_conditional_sample`] — the
//!   CSS branch of the covariance dispatch, against the OLS closed form
//!   for the conditional likelihood. This one has teeth: the exact-MLE
//!   Hessian *fails* it by two orders of magnitude, so wiring the CSS arm
//!   to the wrong objective cannot pass.
//! * [`rank_guard_margins`] and
//!   [`an_unidentified_arma_never_reports_a_confident_standard_error`] —
//!   the conditioning guard, from both sides.

mod common;

use common::{as_vec, load_fixture, simulate_arma, Lcg};
use tsecon_arima::{ArimaError, ArimaSpec};

/// A seeded random walk with drift of length `t_obs`, at unit scale.
fn random_walk_with_drift(seed: u64, t_obs: usize) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    let mut y = Vec::with_capacity(t_obs);
    let mut level = 0.0;
    y.push(level);
    for _ in 1..t_obs {
        level += 0.4 + rng.gaussian();
        y.push(level);
    }
    y
}

/// Analytic exact MLE of ARIMA(0,1,0)+c: the mean and the divide-by-`n`
/// variance of the first differences.
fn rw_drift_mle(y: &[f64]) -> (f64, f64, f64) {
    let dx: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
    let n = dx.len() as f64;
    let c = dx.iter().sum::<f64>() / n;
    let s2 = dx.iter().map(|v| (v - c) * (v - c)).sum::<f64>() / n;
    (c, s2, n)
}

/// **The scale sweep.** For ARIMA(0,1,0) with a constant, both standard
/// errors have exact closed forms at the MLE,
///
/// ```text
/// se(c)      = sqrt(sigma2 / n),
/// se(sigma2) = sqrt(2 sigma2^2 / n),
/// ```
///
/// and multiplying the series by a constant multiplies `c` by it and
/// `sigma2` by its square, leaving both *relative* errors invariant. Any
/// scale dependence in the answer is therefore a defect in the
/// differentiation, not in the statistics — which is what makes this a
/// clean test of the step rule.
///
/// The band `sigma2 ~ [2.5e-5, 1e-3]` is the one that matters: it is
/// where daily log returns and rates-in-decimals land, and where the
/// absolute-floored step `h = eps^(1/4) max(|sigma2|, 0.1)` returned
/// 4.6e-2 relative error with no error and no NaN. Measured worst case
/// with the log-scale step, over the whole sweep: 4.6e-7.
///
/// The lower end used to stop at `sigma2 ~ 1e-8` for a reason outside this
/// crate: the state-space filter compared each prediction variance against
/// an *absolute* `1e-10` floor and dropped everything below it, so beneath
/// that the log-likelihood went numerically constant and this sweep could
/// not be extended. That floor is now relative to the variance's own scale
/// (`tsecon_ssm::filter::TOLERANCE_RANK`), so the likelihood is real all
/// the way down; [`bse_is_exact_far_below_the_old_absolute_floor`] carries
/// the sweep another ten decades to `sigma2 ~ 9.8e-19`.
#[test]
fn bse_is_scale_free_across_fourteen_decades_of_sigma2() {
    let base = random_walk_with_drift(7, 200);
    let spec = ArimaSpec::new(0, 1, 0).unwrap().with_constant(true);

    let mut worst = 0.0_f64;
    let mut worst_at = 0.0_f64;
    // scale 1e-4 .. 1e3 puts sigma2 in 1e-8 .. 1e6.
    for k in -3..=4 {
        let scale = 10f64.powi(-k);
        let y: Vec<f64> = base.iter().map(|v| v * scale).collect();
        let (c, s2, n) = rw_drift_mle(&y);
        let bse = spec.at_params(&y, &[c, s2]).unwrap().bse().unwrap();

        let want = [(s2 / n).sqrt(), (2.0 * s2 * s2 / n).sqrt()];
        for (i, (&got, &w)) in bse.iter().zip(&want).enumerate() {
            let rel = (got - w).abs() / w;
            if rel > worst {
                worst = rel;
                worst_at = s2;
            }
            assert!(
                rel <= 5e-6,
                "sigma2 = {s2:e}: bse[{i}] = {got} vs the closed form {w} (rel {rel:e})"
            );
        }
    }
    println!("sigma2 sweep: worst relative error {worst:e}, at sigma2 = {worst_at:e}");

    // The specific band the absolute-floored step got wrong. sigma2 here
    // is 9.8e-5, inside [2.5e-5, 1e-3]; the old rule returned 4.6e-2
    // relative error on se(sigma2) and reported success.
    let y: Vec<f64> = base.iter().map(|v| v * 1e-2).collect();
    let (c, s2, n) = rw_drift_mle(&y);
    assert!(
        (1e-5..1e-3).contains(&s2),
        "the regression case drifted out of its band: sigma2 = {s2:e}"
    );
    let bse = spec.at_params(&y, &[c, s2]).unwrap().bse().unwrap();
    let want = (2.0 * s2 * s2 / n).sqrt();
    assert!(
        (bse[1] - want).abs() <= 1e-6 * want,
        "se(sigma2) at sigma2 = {s2:e}: {} vs {want}",
        bse[1]
    );

    // Below the old absolute variance floor the likelihood used to go
    // numerically constant, and this assertion used to demand an error.
    // With the relative floor the likelihood is real there, so the
    // standard errors are simply correct — see
    // `bse_is_exact_far_below_the_old_absolute_floor` for the full sweep.
    let y: Vec<f64> = base.iter().map(|v| v * 1e-6).collect();
    let (c, s2, n) = rw_drift_mle(&y);
    assert!(
        s2 < 1e-10,
        "expected sigma2 under the old filter floor: {s2:e}"
    );
    let bse = spec
        .at_params(&y, &[c, s2])
        .unwrap()
        .bse()
        .expect("a sub-1e-10 sigma2 is filterable, not a flat likelihood");
    let want = (2.0 * s2 * s2 / n).sqrt();
    assert!(
        (bse[1] - want).abs() <= 5e-6 * want,
        "se(sigma2) at sigma2 = {s2:e}: {} vs the closed form {want}",
        bse[1]
    );
}

/// **The regression test for the silent `loglik = 0.0`.** The exact
/// log-likelihood of ARIMA(0,1,0)+c is a closed form in the series scale:
/// multiplying `y` by `c` multiplies the MLE `sigma2` by `c^2` and shifts
/// the log-likelihood by exactly `-n ln c`, where `n` is the number of
/// first differences. Nothing external is needed to check it.
///
/// Before the fix, the state-space filter compared each prediction
/// variance against an absolute `1e-10`. At `sigma2 ~ 1e-13` that
/// discarded *every* observation, and `arima` reported `loglik = 0.0` as a
/// success — a meaningless optimum indistinguishable from a real one. The
/// sweep here runs `sigma2` from `9.8e5` down to `9.8e-19`, twenty-four
/// decades, and asserts the likelihood sits on the closed-form line the
/// whole way. Under the old floor everything from `k = 5` on returned
/// exactly `0.0`, missing the line by thousands of nats.
///
/// The standard errors are checked alongside, because a likelihood that is
/// merely *nonzero* but wrong would still pass a nonzero-check.
#[test]
fn bse_is_exact_far_below_the_old_absolute_floor() {
    let base = random_walk_with_drift(7, 200);
    let spec = ArimaSpec::new(0, 1, 0).unwrap().with_constant(true);

    let (c0, _, n) = rw_drift_mle(&base);
    let (_, s2_0, _) = rw_drift_mle(&base);
    let ll0 = spec
        .loglike(&base, &[c0, s2_0])
        .expect("the unit-scale likelihood is well defined");

    let mut worst_ll = 0.0_f64;
    let mut worst_bse = 0.0_f64;
    // scale 1e-9 .. 1e3 puts sigma2 in 9.8e-19 .. 9.8e5.
    for k in -3..=9 {
        let scale = 10f64.powi(-k);
        let y: Vec<f64> = base.iter().map(|v| v * scale).collect();
        let (c, s2, _) = rw_drift_mle(&y);

        let ll = spec
            .loglike(&y, &[c, s2])
            .unwrap_or_else(|e| panic!("sigma2 = {s2:e}: loglike failed with {e}"));
        assert_ne!(
            ll, 0.0,
            "sigma2 = {s2:e}: loglik collapsed to exactly 0.0 — every \
             observation was discarded, which is the absolute-floor defect"
        );

        // loglik(scale * y) = loglik(y) - n ln(scale).
        let want_ll = ll0 - n * scale.ln();
        let rel_ll = (ll - want_ll).abs() / want_ll.abs().max(1.0);
        worst_ll = worst_ll.max(rel_ll);
        assert!(
            rel_ll <= 1e-12,
            "sigma2 = {s2:e}: loglik {ll} vs the closed form {want_ll} (rel {rel_ll:e})"
        );

        let bse = spec
            .at_params(&y, &[c, s2])
            .unwrap()
            .bse()
            .unwrap_or_else(|e| panic!("sigma2 = {s2:e}: bse failed with {e}"));
        let want = [(s2 / n).sqrt(), (2.0 * s2 * s2 / n).sqrt()];
        for (i, (&got, &w)) in bse.iter().zip(&want).enumerate() {
            let rel = (got - w).abs() / w;
            worst_bse = worst_bse.max(rel);
            assert!(
                rel <= 5e-6,
                "sigma2 = {s2:e}: bse[{i}] = {got} vs the closed form {w} (rel {rel:e})"
            );
        }
    }
    println!("24-decade sweep: worst loglik deviation {worst_ll:e}, worst bse {worst_bse:e}");
}

/// OLS of `y_t` on `[1, y_{t-1}, ..., y_{t-p}]` over `t = p..n`, with the
/// divide-by-`n_c` (maximum-likelihood, not degrees-of-freedom-corrected)
/// residual variance. Returns `(beta, se(beta), sigma2)`.
///
/// This is the closed form of the conditional-likelihood information for
/// a pure AR(p) with a constant: `SSR` is *exactly* quadratic in `beta`,
/// so the CSS objective's Hessian in the mean block is `X'X / sigma2` to
/// the last bit, with no approximation anywhere.
fn ols_conditional(y: &[f64], p: usize) -> (Vec<f64>, Vec<f64>, f64) {
    let n = y.len();
    let k = p + 1;
    let n_c = (n - p) as f64;
    let mut xtx = vec![0.0; k * k];
    let mut xty = vec![0.0; k];
    let row_at = |t: usize| {
        let mut row = vec![1.0];
        row.extend((0..p).map(|i| y[t - 1 - i]));
        row
    };
    for (t, &y_t) in y.iter().enumerate().skip(p) {
        let row = row_at(t);
        for a in 0..k {
            for b in 0..k {
                xtx[a * k + b] += row[a] * row[b];
            }
        }
        for a in 0..k {
            xty[a] += row[a] * y_t;
        }
    }
    let inv = invert_dense(&xtx, k);
    let beta: Vec<f64> = (0..k)
        .map(|a| (0..k).map(|b| inv[a * k + b] * xty[b]).sum())
        .collect();
    let ssr: f64 = (p..n)
        .map(|t| {
            let e = y[t] - row_at(t).iter().zip(&beta).map(|(r, b)| r * b).sum::<f64>();
            e * e
        })
        .sum();
    let sigma2 = ssr / n_c;
    let se = (0..k).map(|a| (sigma2 * inv[a * k + a]).sqrt()).collect();
    (beta, se, sigma2)
}

/// Gauss-Jordan inverse for the tiny `X'X` above (test-local, so the
/// reference does not borrow the crate's own linear algebra).
fn invert_dense(a: &[f64], n: usize) -> Vec<f64> {
    let w = 2 * n;
    let mut m = vec![0.0; n * w];
    for i in 0..n {
        m[i * w..i * w + n].copy_from_slice(&a[i * n..i * n + n]);
        m[i * w + n + i] = 1.0;
    }
    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&x, &y| {
                m[x * w + col]
                    .abs()
                    .partial_cmp(&m[y * w + col].abs())
                    .expect("finite")
            })
            .expect("nonempty");
        for c in 0..w {
            m.swap(col * w + c, pivot_row * w + c);
        }
        let pivot = m[col * w + col];
        assert!(pivot.abs() > 1e-12, "singular X'X in the test reference");
        for c in 0..w {
            m[col * w + c] /= pivot;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * w + col];
            for c in 0..w {
                m[r * w + c] -= f * m[col * w + c];
            }
        }
    }
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n..i * n + n].copy_from_slice(&m[i * w + n..i * w + w]);
    }
    inv
}

/// **The CSS guard, with teeth.** A CSS fit's standard errors must come
/// from the Hessian of the *conditional* likelihood — the objective CSS
/// actually maximized — and for a pure AR(p) with a constant that Hessian
/// is known in closed form:
///
/// ```text
/// Cov(c, phi) = sigma2 (X'X)^{-1},   Var(sigma2) = 2 sigma2^2 / n_c,
/// ```
///
/// `X` being the conditional design `[1, y_{t-1}, ..., y_{t-p}]` over
/// `t = p..n` and `sigma2 = SSR / n_c`. So `fit_css().bse()` must equal
/// OLS standard errors on that regression — to 1e-5 here, against a
/// measured worst case of 1.8e-7.
///
/// Why this is not the test it replaces. The old guard compared CSS
/// standard errors to *MLE* ones with a 10% tolerance and claimed to
/// detect the CSS arm being wired to the exact likelihood. It could not:
/// that mutation makes the two vectors agree **better**, so a
/// same-vs-same tolerance passes. The mutation was run — the whole crate
/// suite stayed green. Here the exact-likelihood Hessian misses the OLS
/// closed form by 1.4e-3 to 5.8e-3, two orders of magnitude outside the
/// gate, so the same mutation fails loudly. The `mle` leg below asserts
/// that separation directly, so the test cannot rot into a tautology
/// without saying so.
#[test]
fn css_standard_errors_match_ols_on_the_conditional_sample() {
    for (n, ar) in [(400usize, vec![0.6]), (400, vec![0.5, -0.3])] {
        let p = ar.len();
        let mut rng = Lcg::new(2026);
        let y = simulate_arma(&mut rng, n, 0.4, &ar, &[], 1.3);

        let spec = ArimaSpec::new(p, 0, 0).unwrap().with_constant(true);
        let css = spec.fit_css(&y).unwrap();
        let css_se = css.bse().unwrap();
        let (ols_beta, ols_se, ols_sigma2) = ols_conditional(&y, p);

        // The CSS optimum is the OLS solution; if it were not, comparing
        // curvature at two different points would be meaningless.
        for (i, (&got, &want)) in css.params().iter().zip(&ols_beta).enumerate() {
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1e-3),
                "p={p}: CSS did not land on the OLS solution: param[{i}] {got} vs {want}"
            );
        }
        assert!(
            (css.sigma2() - ols_sigma2).abs() <= 1e-9 * ols_sigma2,
            "p={p}: sigma2 {} vs SSR/n_c {ols_sigma2}",
            css.sigma2()
        );

        // The mean block: sigma2 (X'X)^{-1}.
        for (i, &want) in ols_se.iter().enumerate() {
            assert!(
                (css_se[i] - want).abs() <= 1e-5 * want,
                "p={p}: css bse[{i}] = {} vs the OLS closed form {want} (rel {:e})",
                css_se[i],
                (css_se[i] - want).abs() / want
            );
        }
        // The sigma2 slot: 2 sigma2^2 / n_c, with n_c the *conditional*
        // sample size — differentiating the exact likelihood would use
        // n instead, and would not be a stationary point in sigma2.
        let n_c = (n - p) as f64;
        let want_sigma2_se = (2.0 * ols_sigma2 * ols_sigma2 / n_c).sqrt();
        assert!(
            (css_se[p + 1] - want_sigma2_se).abs() <= 1e-5 * want_sigma2_se,
            "p={p}: css bse[sigma2] = {} vs {want_sigma2_se}",
            css_se[p + 1]
        );

        // The teeth: the exact-MLE Hessian is a *different* estimator and
        // must miss this closed form by far more than the gate above. If
        // this ever stopped holding, the test above would have stopped
        // discriminating and would need replacing, not loosening.
        let mle_se = spec.fit(&y).unwrap().bse().unwrap();
        let separation = ols_se
            .iter()
            .enumerate()
            .map(|(i, &want)| (mle_se[i] - want).abs() / want)
            .fold(0.0_f64, f64::max);
        assert!(
            separation > 1e-3,
            "p={p}: the exact-MLE standard errors are within {separation:e} of the CSS \
             closed form, so this test can no longer tell the two objectives apart"
        );
        println!(
            "css/ols gate 1e-5; exact-MLE Hessian misses by {separation:e} (p = {p}, n = {n})"
        );
    }
}

/// The conditioning guard from the accept side: every statsmodels golden
/// case must clear [`MIN_RCOND`](tsecon_arima::ParamCov::rcond) by a wide
/// margin, so a future change to the step rule cannot quietly turn a
/// working fit into a `CovarianceFailed`.
///
/// The Nile ARMA(1,1) is the binding case at `rcond = 5.1e-4` — an
/// AR(1) and an MA(1) term are nearly redundant on 100 observations, and
/// the evaluation point is not a stationary point of the likelihood. The
/// assertion is at 1e-5, i.e. it fails if that margin shrinks by 50x,
/// long before the fit itself would start erroring.
#[test]
fn rank_guard_margins() {
    // The binding case: the Nile ARMA(1,1) at the fixture's recorded
    // statsmodels parameters, which is the least well conditioned fit in
    // the whole golden set.
    let nile = as_vec(&load_fixture("diagnostics.json")["nile"]);
    let params = as_vec(&load_fixture("arima_bse.json")["cases"]["nile_arma11c"]["params"]);
    let nile_rcond = ArimaSpec::new(1, 0, 1)
        .unwrap()
        .with_constant(true)
        .at_params(&nile, &params)
        .unwrap()
        .param_cov()
        .unwrap()
        .rcond();
    assert!(
        nile_rcond > 1e-5,
        "the worst golden fit is down to rcond = {nile_rcond:e}, within 10x of the \
         threshold that would start refusing it"
    );

    let mut rng = Lcg::new(11);
    let y = simulate_arma(&mut rng, 300, 0.5, &[0.6], &[0.4], 1.2);
    let spec = ArimaSpec::new(1, 0, 1).unwrap().with_constant(true);
    let pc = spec.fit(&y).unwrap().param_cov().unwrap();
    assert!(
        pc.rcond() > 1e-3,
        "a healthy ARMA(1,1) fit is only conditioned to {:e}",
        pc.rcond()
    );
    assert!(pc.rcond() <= 1.0, "rcond = {:e} exceeds 1", pc.rcond());

    // The random walk with drift: the information is exactly block
    // diagonal at the MLE, so this is as well conditioned as it gets.
    let rw = random_walk_with_drift(7, 60);
    let (c, s2, _) = rw_drift_mle(&rw);
    let pc = ArimaSpec::new(0, 1, 0)
        .unwrap()
        .with_constant(true)
        .at_params(&rw, &[c, s2])
        .unwrap()
        .param_cov()
        .unwrap();
    assert!(
        pc.rcond() > 0.9,
        "block-diagonal information came back at rcond = {:e}",
        pc.rcond()
    );
}

/// The conditioning guard from the refuse side, on a model that is
/// unidentified *provably* rather than approximately.
///
/// An ARMA(1,1) with `theta = -phi` has cancelling polynomials: the
/// process is iid with mean `c / (1 - phi)`, so along the line
/// `{c = (1 - phi) mu, theta = -phi}` the exact likelihood is **constant**
/// — the test asserts that, bit for bit, as its own premise. `phi` is
/// therefore not identified by any sample, and the information matrix is
/// singular in that direction.
///
/// What the crate must never do there is return a small, confident-looking
/// standard error. It is allowed to fail two ways — `CovarianceFailed`
/// from the rank guard, or `NaN` from a negative variance — and which one
/// depends on the sign of the differencing noise that fills the flat
/// direction. Both are honest; a number is not.
#[test]
fn an_unidentified_arma_never_reports_a_confident_standard_error() {
    let mut rng = Lcg::new(4242);
    let y: Vec<f64> = (0..300).map(|_| 1.0 + rng.gaussian()).collect();
    let n = y.len() as f64;
    let mu = y.iter().sum::<f64>() / n;
    let s2 = y.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / n;
    let spec = ArimaSpec::new(1, 0, 1).unwrap().with_constant(true);

    // The premise: the likelihood really is flat along the line.
    let reference = spec
        .loglike(&y, &[(1.0 - 0.5) * mu, 0.5, -0.5, s2])
        .unwrap();
    for phi in [0.2_f64, 0.35, 0.65, 0.8] {
        let ll = spec
            .loglike(&y, &[(1.0 - phi) * mu, phi, -phi, s2])
            .unwrap();
        assert!(
            (ll - reference).abs() <= 1e-9 * reference.abs(),
            "phi = {phi}: loglik {ll} != {reference}, so this point is not unidentified \
             and the test premise is wrong"
        );
    }

    for phi in [0.2_f64, 0.35, 0.5, 0.65, 0.8] {
        let res = spec
            .at_params(&y, &[(1.0 - phi) * mu, phi, -phi, s2])
            .unwrap();
        match res.bse() {
            Err(ArimaError::CovarianceFailed { .. }) => {}
            Err(other) => panic!("phi = {phi}: unexpected error {other:?}"),
            Ok(bse) => {
                // sigma2 *is* identified here (the process is iid), so
                // only the [c, phi, theta] block must refuse to answer.
                assert!(
                    bse[..3].iter().any(|v| v.is_nan()),
                    "phi = {phi}: the unidentified block came back finite: {bse:?}"
                );
            }
        }
    }
}
