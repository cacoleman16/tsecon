//! The estimation budgets and the refusal wording of audit round 16.
//!
//! * `ArimaSpec::estimation_cost` against its documented closed form, and
//!   specifications past `ARIMA_WORK_BUDGET` / `ARIMA_BUDGET_VALUES`
//!   refused before anything is allocated (they used to run for hours or
//!   exhaust memory: at `p = 1000` one likelihood evaluation keeps 4.8 GB of
//!   filter covariances on 200 observations), with the largest state
//!   dimension that fits;
//! * the too-few-observations refusals naming the seasonal specification
//!   when the seasonal part is what consumes the series (H16-14), while the
//!   non-seasonal wording is unchanged.

mod common;

use common::{simulate_arma, Lcg};
use tsecon_arima::spec::{ARIMA_BUDGET_VALUES, ARIMA_WORK_BUDGET};
use tsecon_arima::{ArimaError, ArimaSpec, EstimationMethod};

fn series(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    simulate_arma(&mut rng, n, 0.0, &[0.5], &[], 1.0)
}

/// The documented count: `E = (4 n + 200) m^3`, `values = 3 (n + 1) m^2`,
/// `evals = 2 k (k + 2) + 1`.
fn closed_form(m: u128, k: u128, n: u128, css_lags: Option<u128>) -> (u128, u128) {
    let exact = (4 * n + 200) * m * m * m;
    let evals = 2 * k * (k + 2) + 1;
    let work = match css_lags {
        None => evals * exact,
        Some(lags) => evals * n * lags + exact,
    };
    (work, 3 * (n + 1) * m * m)
}

#[test]
fn estimation_cost_matches_the_documented_closed_form() {
    // SARIMA(1,0,0)(1,0,0)_199 with a constant: m = 200, k = 4.
    let spec = ArimaSpec::new(1, 0, 0)
        .unwrap()
        .with_constant(true)
        .seasonal(1, 0, 0, 199)
        .unwrap();
    assert_eq!(spec.state_dim(), 200);
    assert_eq!(
        spec.estimation_cost(200, EstimationMethod::ExactMle),
        closed_form(200, 4, 200, None)
    );
    // CSS: the recursion over p + s*P + q + s*Q + 1 = 201 lags, plus one
    // exact filter pass for the residuals.
    assert_eq!(
        spec.estimation_cost(200, EstimationMethod::Css),
        closed_form(200, 4, 200, Some(201))
    );
    let (fixed, _) = spec.estimation_cost(200, EstimationMethod::Fixed);
    assert_eq!(fixed, (4 * 200 + 200) * 200u128.pow(3));
    // MA-dominated state: m = q + s*Q + 1.
    let ma = ArimaSpec::new(0, 1, 2)
        .unwrap()
        .seasonal(0, 1, 1, 12)
        .unwrap();
    assert_eq!(ma.state_dim(), 15);
    assert_eq!(
        ma.estimation_cost(131, EstimationMethod::ExactMle),
        closed_form(15, 4, 131, None)
    );
}

/// Absurd specifications are refused before anything is allocated, naming
/// the orders and the largest state dimension that fits — the bisection on
/// the closed form.
#[test]
fn absurd_specifications_are_refused_by_name() {
    let y = series(200, 1);
    let cases: [(ArimaSpec, &str); 3] = [
        // The H16-14 refuter's 40-second timeouts.
        (
            ArimaSpec::new(1, 0, 0)
                .unwrap()
                .with_constant(true)
                .seasonal(1, 0, 0, 199)
                .unwrap(),
            "work",
        ),
        (
            ArimaSpec::new(1, 0, 0)
                .unwrap()
                .with_constant(true)
                .seasonal(50, 0, 0, 4)
                .unwrap(),
            "work",
        ),
        (
            ArimaSpec::new(150, 0, 0).unwrap().with_constant(true),
            "work",
        ),
    ];
    for (spec, _) in cases {
        let (work, _) = spec.estimation_cost(200, EstimationMethod::ExactMle);
        assert!(work > ARIMA_WORK_BUDGET, "{spec:?} must be past the budget");
        let err = spec.fit(&y).unwrap_err();
        let ArimaError::TooLarge {
            state_dim,
            k_params,
            n_eff,
            max_state_dim,
            ..
        } = err
        else {
            panic!("expected TooLarge, got {err:?}");
        };
        assert_eq!(
            (state_dim, k_params, n_eff),
            (spec.state_dim(), spec.k_params(), 200)
        );
        // The largest m whose closed-form cost fits both budgets.
        let fits = |m: u128| {
            let (w, v) = closed_form(m, k_params as u128, 200, None);
            w <= ARIMA_WORK_BUDGET && v <= ARIMA_BUDGET_VALUES
        };
        assert!(fits(max_state_dim as u128) && !fits(max_state_dim as u128 + 1));
        let msg = err.to_string();
        assert!(msg.contains("seasonal=(P, D, Q, s)"), "{msg}");
        assert!(
            msg.contains(&format!("can be at most {max_state_dim}:")),
            "{msg}"
        );
        assert!(
            msg.starts_with("the ARIMA orders (p, q and seasonal=(P, D, Q, s))"),
            "{msg}"
        );
        assert!(msg.contains("lower p or q"), "{msg}");
    }

    // The memory budget: p = 600 on 1000 observations would hold
    // 3 * 1001 * 600^2 = 1.08e9 values per likelihood evaluation.
    let long = series(1000, 2);
    let spec = ArimaSpec::new(600, 0, 0).unwrap();
    let err = spec.fit(&long).unwrap_err();
    assert!(
        matches!(err, ArimaError::TooLarge { exceeded, .. } if exceeded.starts_with("both")),
        "{err:?}"
    );
    // A single likelihood evaluation is refused on the memory count too.
    let mut params = vec![0.0; 601];
    params[600] = 1.0;
    assert!(matches!(
        spec.loglike(&long, &params),
        Err(ArimaError::TooLarge { .. })
    ));
    // CSS: the recursion is cheap but the residuals run the exact filter.
    let err = ArimaSpec::new(0, 0, 900)
        .unwrap()
        .fit_css(&series(2000, 3))
        .unwrap_err();
    assert!(matches!(err, ArimaError::TooLarge { .. }), "{err:?}");
}

/// Ordinary specifications are far inside the budgets: the airline model,
/// the auto_arima stepwise start on monthly data, a weekly SARMA with a
/// constant, and an AR(30) on 200 observations (measured at about two
/// minutes in `lab/arima_budget/out/arima_cost.txt`).
#[test]
fn ordinary_specifications_fit_inside_the_budgets() {
    let within = |spec: ArimaSpec, n_eff: usize| {
        let (w, v) = spec.estimation_cost(n_eff, EstimationMethod::ExactMle);
        w <= ARIMA_WORK_BUDGET && v <= ARIMA_BUDGET_VALUES
    };
    let airline = ArimaSpec::new(0, 1, 1)
        .unwrap()
        .seasonal(0, 1, 1, 12)
        .unwrap();
    assert!(within(airline, 131));
    let start = ArimaSpec::new(2, 1, 2)
        .unwrap()
        .seasonal(2, 1, 2, 12)
        .unwrap();
    assert!(within(start, 131));
    let weekly = ArimaSpec::new(1, 0, 1)
        .unwrap()
        .with_constant(true)
        .seasonal(1, 0, 1, 52)
        .unwrap();
    assert!(within(weekly, 1040));
    assert!(within(
        ArimaSpec::new(30, 0, 0).unwrap().with_constant(true),
        200
    ));
}

/// H16-14: when the seasonal part is what consumes the series, the
/// too-few-observations refusal names the seasonal specification instead of
/// telling the caller to "lower p, d, or q" (following that advice did not
/// help); the non-seasonal wording is unchanged.
#[test]
fn short_sample_refusals_name_the_seasonal_part() {
    let y = series(200, 4);
    // CSS: p + s*P = 200 presample values on 200 observations.
    let css = ArimaSpec::new(1, 0, 0)
        .unwrap()
        .with_constant(true)
        .seasonal(1, 0, 0, 199)
        .unwrap()
        .fit_css(&y)
        .unwrap_err();
    let msg = css.to_string();
    assert!(msg.contains("needs at least 205 observations"), "{msg}");
    assert!(msg.contains("seasonal=(P, D, Q, s)"), "{msg}");
    assert!(!msg.contains("Lower p, d, or q"), "{msg}");
    assert!(msg.contains("one residual degree of freedom"), "{msg}");
    // Differencing: D = 3 at s = 70 drops 210 of the 200 observations.
    let diff = ArimaSpec::new(0, 0, 0)
        .unwrap()
        .seasonal(0, 3, 0, 70)
        .unwrap()
        .fit(&y)
        .unwrap_err();
    let msg = diff.to_string();
    assert!(msg.contains("seasonal differencing order D"), "{msg}");
    assert!(msg.contains("seasonal=(P, D, Q, s)"), "{msg}");
    // Non-seasonal: unchanged.
    let plain = ArimaSpec::new(1, 0, 0).unwrap().fit(&y[..2]).unwrap_err();
    assert!(
        plain
            .to_string()
            .ends_with("Lower p, d, or q, or supply a longer series."),
        "{plain}"
    );
    let plain_css = ArimaSpec::new(1, 0, 0)
        .unwrap()
        .fit_css(&y[..3])
        .unwrap_err();
    assert!(
        plain_css
            .to_string()
            .ends_with("Lower p, d, or q, or supply a longer series."),
        "{plain_css}"
    );
}
