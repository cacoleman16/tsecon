//! Property tests: loading limits, cross-sectional reconstruction, the
//! optimal-lambda fit, Svensson nesting, dynamic-factor recovery, and
//! forward/yield consistency. Uses the fixture panel where real data helps.

use serde_json::Value;
use tsecon_termstructure::{
    fit_dynamic_ns, fit_nelson_siegel, fit_nelson_siegel_optimal_lambda, fit_svensson,
    nelson_siegel_loadings,
};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/termstructure.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture file readable");
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn f64_matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(f64s).collect()
}

#[test]
fn loading_short_end_limits_are_correct() {
    // As maturity -> 0 (with lambda fixed), the slope loading -> 1 and the
    // curvature loading -> 0. Check at a genuinely tiny maturity, where the
    // naive (1 - e^{-x})/x would suffer cancellation.
    let lambda = 0.0609;
    let tiny = 1e-9;
    let [level, slope, curv] = nelson_siegel_loadings(&[tiny], lambda).expect("loadings");
    assert_eq!(level[0], 1.0);
    assert!(
        (slope[0] - 1.0).abs() < 1e-8,
        "slope -> 1, got {}",
        slope[0]
    );
    assert!(curv[0].abs() < 1e-8, "curvature -> 0, got {}", curv[0]);

    // Long end: slope -> 0, curvature -> 0, level stays 1. The slope loading
    // decays like 1/(lambda t), so use a huge maturity to drive it below 1e-6.
    let big = 1e10;
    let [level_b, slope_b, curv_b] = nelson_siegel_loadings(&[big], lambda).expect("loadings");
    assert_eq!(level_b[0], 1.0);
    assert!(slope_b[0].abs() < 1e-6);
    assert!(curv_b[0].abs() < 1e-6);
}

#[test]
fn ns_reconstructs_fixture_curve_r2_above_090() {
    let fx = load();
    let maturities = f64s(&fx["maturities"]);
    let lambda = fx["lambda"].as_f64().expect("lambda");
    let yields = f64s(&fx["yields_date100"]);

    let fit = fit_nelson_siegel(&maturities, &yields, lambda).expect("fit");
    assert!(fit.rsquared > 0.9, "R^2 = {}", fit.rsquared);

    // The fitted curve is close to the observed one at every maturity.
    let fitted = fit.fitted(&maturities).expect("fitted");
    for (i, (&f, &y)) in fitted.iter().zip(yields.iter()).enumerate() {
        assert!((f - y).abs() < 0.25, "maturity {i}: fitted {f} vs {y}");
    }
}

#[test]
fn fixed_lambda_fits_whole_panel_well() {
    // Property: on the fixture panel, the fixed-lambda fit residuals are small
    // (R^2 > 0.9) for the great majority of dates.
    let fx = load();
    let maturities = f64s(&fx["maturities"]);
    let lambda = fx["lambda"].as_f64().expect("lambda");
    let panel = f64_matrix(&fx["yields_panel"]);

    let dyn_fit = fit_dynamic_ns(&panel, &maturities, lambda).expect("dynamic fit");
    let n = dyn_fit.rsquared.len();

    // The typical curve fits very well: the median R^2 exceeds 0.9 (a few
    // near-flat curves have a tiny total sum of squares and so a low R^2 even
    // with small residuals, which is why we use the median, not the minimum).
    let mut sorted = dyn_fit.rsquared.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite R^2"));
    let median = sorted[n / 2];
    assert!(median > 0.9, "median panel R^2 = {median}");

    // And the mean cross-sectional fit error is small in yield units.
    let mean_r2 = dyn_fit.rsquared.iter().sum::<f64>() / n as f64;
    assert!(mean_r2 > 0.9, "mean panel R^2 = {mean_r2}");
}

#[test]
fn optimal_lambda_fits_at_least_as_well_as_fixed() {
    let fx = load();
    let maturities = f64s(&fx["maturities"]);
    let lambda = fx["lambda"].as_f64().expect("lambda");
    let yields = f64s(&fx["yields_date100"]);

    let fixed = fit_nelson_siegel(&maturities, &yields, lambda).expect("fixed");
    let opt = fit_nelson_siegel_optimal_lambda(&maturities, &yields, lambda).expect("opt");

    assert!(opt.lambda > 0.0);
    // NLS minimizes SSR, i.e. maximizes R^2, so it cannot do worse (modulo a
    // tiny optimizer tolerance).
    assert!(
        opt.rsquared >= fixed.rsquared - 1e-6,
        "opt R^2 {} < fixed R^2 {}",
        opt.rsquared,
        fixed.rsquared
    );
    assert!(opt.rsquared > 0.9);
}

#[test]
fn svensson_nests_nelson_siegel_and_fits_at_least_as_well() {
    let fx = load();
    let maturities = f64s(&fx["maturities"]);
    let lambda = fx["lambda"].as_f64().expect("lambda");
    let yields = f64s(&fx["yields_date100"]);

    let ns = fit_nelson_siegel(&maturities, &yields, lambda).expect("ns");
    // A distinct second decay adds a genuine fourth column; the four-factor
    // least-squares fit cannot do worse than the three-factor one.
    let sv = fit_svensson(&maturities, &yields, lambda, 0.15).expect("svensson");
    assert!(
        sv.rsquared >= ns.rsquared - 1e-9,
        "svensson R^2 {} < ns R^2 {}",
        sv.rsquared,
        ns.rsquared
    );

    // Svensson yields reconstruct the curve well too.
    let fitted = sv.fitted(&maturities).expect("fitted");
    for (&f, &y) in fitted.iter().zip(yields.iter()) {
        assert!((f - y).abs() < 0.25);
    }
}

#[test]
fn dynamic_level_factor_tracks_long_yield() {
    // The level factor is the long-rate proxy: it should be highly correlated
    // with the longest-maturity yield across the panel.
    let fx = load();
    let maturities = f64s(&fx["maturities"]);
    let lambda = fx["lambda"].as_f64().expect("lambda");
    let panel = f64_matrix(&fx["yields_panel"]);
    let last_mat = maturities.len() - 1;

    let dyn_fit = fit_dynamic_ns(&panel, &maturities, lambda).expect("dynamic fit");
    let level = dyn_fit.level();
    let long_yield: Vec<f64> = panel.iter().map(|row| row[last_mat]).collect();

    let corr = correlation(&level, &long_yield);
    assert!(corr > 0.9, "level vs long yield correlation = {corr}");

    // And the fitted curves reconstruct the panel with small error.
    for (d, row) in panel.iter().enumerate() {
        let fitted = dyn_fit.fitted_curve(d).expect("fitted curve");
        for (i, (&f, &y)) in fitted.iter().zip(row.iter()).enumerate() {
            assert!(
                (f - y).abs() < 0.4,
                "date {d} maturity {i}: fitted {f} vs {y}"
            );
        }
    }
}

#[test]
fn dynamic_forecast_is_finite_and_curve_shaped() {
    let fx = load();
    let maturities = f64s(&fx["maturities"]);
    let lambda = fx["lambda"].as_f64().expect("lambda");
    let panel = f64_matrix(&fx["yields_panel"]);

    let dyn_fit = fit_dynamic_ns(&panel, &maturities, lambda).expect("dynamic fit");
    let fc = dyn_fit.forecast().expect("forecast");

    assert_eq!(fc.yields.len(), maturities.len());
    assert!(fc.factors.iter().all(|f| f.is_finite()));
    assert!(fc.yields.iter().all(|y| y.is_finite() && *y > 0.0));
    // The level factor is persistent (Diebold-Li): AR(1) phi near 1.
    assert!(
        fc.factor_ar1[0].phi > 0.5,
        "level phi = {}",
        fc.factor_ar1[0].phi
    );

    // The one-step forecast is close to the last observed curve (yields move
    // slowly month to month).
    let last = panel.last().expect("last date");
    for (i, (&y, &l)) in fc.yields.iter().zip(last.iter()).enumerate() {
        assert!(
            (y - l).abs() < 0.6,
            "maturity {i}: forecast {y} vs last {l}"
        );
    }
}

#[test]
fn ar1_recovers_simulated_dynamics() {
    // Simulate x_t = c + phi x_{t-1} + e_t with a fixed pseudo-random e_t and
    // check ar1_fit recovers (c, phi) well. Deterministic LCG noise keeps the
    // test dependency-free and reproducible.
    let (c_true, phi_true) = (0.5, 0.8);
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next_noise = || {
        // xorshift64 -> uniform(-0.5, 0.5) scaled small.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.2
    };
    let mut x = vec![c_true / (1.0 - phi_true)];
    for _ in 0..2000 {
        let prev = *x.last().expect("nonempty");
        x.push(c_true + phi_true * prev + next_noise());
    }

    let ar = tsecon_termstructure::ar1_fit(&x).expect("ar1");
    assert!((ar.phi - phi_true).abs() < 0.05, "phi = {}", ar.phi);
    assert!(
        (ar.unconditional_mean() - c_true / (1.0 - phi_true)).abs() < 0.1,
        "mean = {}",
        ar.unconditional_mean()
    );
}

#[test]
fn forward_and_yield_are_consistent() {
    // Consistency checks for the Nelson-Siegel forward curve:
    //  - at the short end, yield and forward both -> level + slope;
    //  - as maturity -> inf, both -> level;
    //  - the yield is the maturity-average of the forward, so
    //    y(t) * t = integral_0^t f(s) ds; verify by numeric quadrature.
    let fx = load();
    let maturities = f64s(&fx["maturities"]);
    let lambda = fx["lambda"].as_f64().expect("lambda");
    let yields = f64s(&fx["yields_date100"]);
    let fit = fit_nelson_siegel(&maturities, &yields, lambda).expect("fit");

    let [b0, b1, _b2] = fit.factors;

    // Short end.
    let y_short = fit.yield_at(1e-6).expect("y short");
    let f_short = fit.forward_at(1e-6).expect("f short");
    assert!((y_short - (b0 + b1)).abs() < 1e-3);
    assert!((f_short - (b0 + b1)).abs() < 1e-3);

    // Long end -> level.
    assert!((fit.yield_at(1e6).expect("y long") - b0).abs() < 1e-3);
    assert!((fit.forward_at(1e6).expect("f long") - b0).abs() < 1e-3);

    // Yield = average forward: y(t) * t ≈ ∫_0^t f(s) ds (trapezoidal).
    let t = 60.0_f64;
    let steps = 20_000;
    let h = t / steps as f64;
    let mut integral = 0.0;
    for k in 0..steps {
        let s0 = k as f64 * h;
        let s1 = (k + 1) as f64 * h;
        let f0 = fit.forward_at(s0.max(1e-12)).expect("f0");
        let f1 = fit.forward_at(s1).expect("f1");
        integral += 0.5 * (f0 + f1) * h;
    }
    let y_t = fit.yield_at(t).expect("y_t");
    assert!(
        (y_t * t - integral).abs() < 1e-3,
        "y*t {} vs integral {}",
        y_t * t,
        integral
    );
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        cov += (ai - ma) * (bi - mb);
        va += (ai - ma).powi(2);
        vb += (bi - mb).powi(2);
    }
    cov / (va.sqrt() * vb.sqrt())
}
