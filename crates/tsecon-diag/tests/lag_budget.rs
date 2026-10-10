//! The lag-design budgets of `arch_lm`, `adf`, `dfgls`, `ng_perron` and the
//! `zivot_andrews` lag search: an order the sample admits but whose design
//! or lag search no machine holds is refused by name, before any lag column
//! is built (it used to abort the process when the allocator refused, or run
//! for hours), and the refusal quotes the largest order that fits — checked
//! here against the documented work and size formulas, transcribed
//! independently of the crate's own arithmetic. Orders up to the default
//! rule's (Schwert's `ceil(12 (n/100)^(1/4))`, capped) are exempt from the
//! work budget: only the memory budget can refuse them.

use tsecon_diag::{
    adf, arch_lm, dfgls, lag_budget_exceeded, ng_perron, zivot_andrews, AdfLagSelection,
    AdfRegression, DfglsTrend, DiagError, LagLimit, NgPerronLagSelection, ZaLagSelection,
    ZaRegression, LAG_DESIGN_MAX_CELLS, LAG_WORK_BUDGET,
};

/// A seeded random walk (a SplitMix64 stream mapped to uniform steps).
fn walk(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut level = 0.0;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            level += (z >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            level
        })
        .collect()
}

/// Schwert's rule, the statsmodels/arch default search cap before capping.
fn schwert(n: usize) -> usize {
    (12.0 * (n as f64 / 100.0).powf(0.25)).ceil() as usize
}

/// The documented size of an ADF-type lag regression with `s` fixed
/// columns and `l` lagged differences on `n` observations: the design has
/// `(n - 1 - l) (s + l)` entries; one QR costs `rows k^2` multiply-adds, and
/// a search over `0..=l` costs `rows (s^2 + ... + (s + l)^2)` plus the
/// longest refit `(n - 1) (s + l)^2`.
fn adf_formula(n: usize, s: usize, l: usize, search: bool) -> (u128, u128) {
    let (n, s, l) = (n as u128, s as u128, l as u128);
    let rows = n - 1 - l;
    let k = s + l;
    let work = if search {
        rows * (s..=k).map(|j| j * j).sum::<u128>() + (n - 1) * k * k
    } else {
        rows * k * k
    };
    (rows * k, work)
}

/// The documented size of the ARCH-LM auxiliary regression: `(n - p) p`
/// entries and `(n - p) p^2 / 2 + p^3` multiply-adds.
fn arch_formula(n: usize, p: usize) -> (u128, u128) {
    let (n, p) = (n as u128, p as u128);
    ((n - p) * p, (n - p) * p * p / 2 + p * p * p)
}

/// Whether order `p` with size `(cells, work)` is admissible when orders up
/// to `free` are exempt from the work budget.
fn admissible(p: usize, free: usize, (cells, work): (u128, u128)) -> bool {
    cells <= LAG_DESIGN_MAX_CELLS && (p <= free || work <= LAG_WORK_BUDGET)
}

/// Unwrap a budget refusal: `(param, value, defaulted, max, message)`.
fn budget_refusal(
    r: Result<impl std::fmt::Debug, DiagError>,
) -> (&'static str, usize, bool, usize, String) {
    match r {
        Err(
            e @ DiagError::LagOrderTooLong {
                limit: LagLimit::Budget { .. },
                ..
            },
        ) => {
            let text = e.to_string();
            let DiagError::LagOrderTooLong {
                param,
                value,
                defaulted,
                max,
                ..
            } = e
            else {
                unreachable!()
            };
            (param, value, defaulted, max.expect("some order fits"), text)
        }
        other => panic!("expected a budget refusal, got {other:?}"),
    }
}

#[test]
fn budget_phrases_name_the_binding_budget() {
    assert_eq!(
        lag_budget_exceeded(LAG_DESIGN_MAX_CELLS, LAG_WORK_BUDGET),
        None
    );
    assert!(lag_budget_exceeded(LAG_DESIGN_MAX_CELLS + 1, 0)
        .unwrap()
        .starts_with("the memory budget"));
    assert!(lag_budget_exceeded(0, LAG_WORK_BUDGET + 1)
        .unwrap()
        .starts_with("the work budget"));
    assert!(lag_budget_exceeded(u128::MAX, u128::MAX)
        .unwrap()
        .starts_with("both"));
}

#[test]
fn arch_lm_refuses_an_order_whose_design_exceeds_the_budget() {
    let n = 1_000_000;
    let e = walk(n, 3);
    // nlags = 400000 satisfies n >= 2 nlags + 2, but its design is
    // 600000 x 400000 = 2.4e11 entries (1.9 TB).
    let (param, value, defaulted, max, text) = budget_refusal(arch_lm(&e, 400_000));
    assert_eq!((param, value, defaulted), ("nlags", 400_000, false));
    assert!(admissible(max, 0, arch_formula(n, max)), "max {max}");
    assert!(
        !admissible(max + 1, 0, arch_formula(n, max + 1)),
        "max {max}"
    );
    assert!(
        text.starts_with("arch_lm: nlags = 400000 needs a lag design of"),
        "{text}"
    );
    assert!(text.contains(&format!("pass nlags <= {max}")), "{text}");
}

#[test]
fn adf_refuses_a_fixed_lag_whose_design_exceeds_the_budget() {
    let n = 1_000_000;
    let y = walk(n, 5);
    let (param, value, defaulted, max, text) = budget_refusal(adf(
        &y,
        AdfRegression::Constant,
        AdfLagSelection::Fixed(400_000),
    ));
    assert_eq!((param, value, defaulted), ("maxlag", 400_000, false));
    let free = schwert(n);
    assert!(admissible(max, free, adf_formula(n, 2, max, false)));
    assert!(!admissible(
        max + 1,
        free,
        adf_formula(n, 2, max + 1, false)
    ));
    assert!(text.contains("memory budget of 2^26"), "{text}");
}

#[test]
fn adf_refuses_a_search_whose_work_exceeds_the_budget() {
    // Small enough for the design to fit memory, long enough for the search
    // (every order 0..=maxlag) to blow the work budget.
    let n = 20_000;
    let y = walk(n, 7);
    let (param, value, defaulted, max, text) = budget_refusal(adf(
        &y,
        AdfRegression::ConstantTrend,
        AdfLagSelection::Aic(Some(3000)),
    ));
    assert_eq!((param, value, defaulted), ("maxlag", 3000, false));
    let free = schwert(n);
    assert!(max > free, "the work budget binds past the default here");
    assert!(admissible(max, free, adf_formula(n, 3, max, true)));
    assert!(!admissible(max + 1, free, adf_formula(n, 3, max + 1, true)));
    assert!(text.contains("past the work budget"), "{text}");
}

#[test]
fn orders_up_to_the_default_are_exempt_from_the_work_budget() {
    // At n = 3e5 the default search (maxlag = 89) costs about 7.6e10
    // multiply-adds, past the work budget, but a default order is never
    // refused for work (its design fits memory). One lag more than the
    // default is charged, refused, and quotes the default as the largest.
    let n = 300_000;
    let y = walk(n, 19);
    let d = schwert(n);
    assert_eq!(d, 89);
    let (cells, work) = adf_formula(n, 2, d, true);
    assert!(work > LAG_WORK_BUDGET && cells <= LAG_DESIGN_MAX_CELLS);
    let (param, value, defaulted, max, text) = budget_refusal(adf(
        &y,
        AdfRegression::Constant,
        AdfLagSelection::Aic(Some(d + 1)),
    ));
    assert_eq!((param, value, defaulted, max), ("maxlag", d + 1, false, d));
    assert!(text.ends_with(&format!("pass maxlag <= {d}, or a shorter series")));
}

#[test]
fn adf_names_the_default_search_cap_when_its_design_exceeds_memory() {
    // The statsmodels default ceil(12 (n/100)^(1/4)) = 143 at n = 2e6: a
    // design of 2.9e8 entries, past the memory budget — refused naming the
    // default and the largest cap that fits.
    let n = 2_000_000;
    let y = walk(n, 9);
    let d = schwert(n);
    assert_eq!(d, 143);
    let (param, value, defaulted, max, text) =
        budget_refusal(adf(&y, AdfRegression::Constant, AdfLagSelection::Aic(None)));
    assert_eq!((param, value, defaulted), ("maxlag", d, true));
    assert!(admissible(max, d, adf_formula(n, 2, max, true)));
    assert!(!admissible(max + 1, d, adf_formula(n, 2, max + 1, true)));
    assert!(
        text.starts_with("adf: the default maxlag = 143 needs"),
        "{text}"
    );
    // Only the memory budget applies to the default order.
    assert!(text.contains("past the memory budget of 2^26"), "{text}");
}

#[test]
fn gls_tests_share_the_budget() {
    let n = 1_000_000;
    let y = walk(n, 11);
    let free = schwert(n);
    let (param, value, _, max, _) = budget_refusal(dfgls(
        &y,
        DfglsTrend::Constant,
        AdfLagSelection::Fixed(300_000),
    ));
    assert_eq!((param, value), ("lags", 300_000));
    assert!(admissible(max, free, adf_formula(n, 1, max, false)));
    assert!(!admissible(
        max + 1,
        free,
        adf_formula(n, 1, max + 1, false)
    ));
    let (param, value, _, max, _) = budget_refusal(ng_perron(
        &y,
        DfglsTrend::ConstantTrend,
        NgPerronLagSelection::Maic(Some(300_000)),
    ));
    assert_eq!((param, value), ("maxlag", 300_000));
    assert!(admissible(max, free, adf_formula(n, 1, max, true)));
    assert!(!admissible(max + 1, free, adf_formula(n, 1, max + 1, true)));
}

#[test]
fn zivot_andrews_lag_search_inherits_the_budget_under_its_own_name() {
    let n = 1_000_000;
    let y = walk(n, 13);
    let r = zivot_andrews(
        &y,
        ZaRegression::Constant,
        0.15,
        ZaLagSelection::Aic(Some(400_000)),
    );
    match r {
        Err(DiagError::LagOrderTooLong {
            what: "zivot_andrews",
            param: "maxlag",
            value: 400_000,
            limit: LagLimit::Budget { .. },
            max: Some(_),
            ..
        }) => {}
        other => panic!("expected the re-attributed budget refusal, got {other:?}"),
    }
}

#[test]
fn sample_refusal_quotes_the_budget_capped_maximum_on_a_long_series() {
    // Past the statsmodels bound on a series so long that the bound itself
    // is over budget: the quoted maximum is the largest order within both.
    let n = 1_000_000;
    let y = walk(n, 15);
    match adf(&y, AdfRegression::Constant, AdfLagSelection::Fixed(600_000)) {
        Err(
            e @ DiagError::LagOrderTooLong {
                limit:
                    LagLimit::Sample {
                        sample_max: Some(499_998),
                        ..
                    },
                max: Some(m),
                ..
            },
        ) => {
            let free = schwert(n);
            assert!(admissible(m, free, adf_formula(n, 2, m, false)));
            assert!(!admissible(m + 1, free, adf_formula(n, 2, m + 1, false)));
            let text = e.to_string();
            assert!(
                text.contains(&format!("pass maxlag <= {m} (orders {}", m + 1)),
                "{text}"
            );
        }
        other => panic!("expected a sample refusal, got {other:?}"),
    }
}

#[test]
fn ordinary_orders_are_untouched() {
    // A typical call never meets the budget: the same series, the same
    // answer as before (the goldens pin the numbers).
    let y = walk(500, 17);
    assert!(adf(&y, AdfRegression::Constant, AdfLagSelection::Aic(None)).is_ok());
    assert!(arch_lm(&y, 12).is_ok());
    assert!(dfgls(&y, DfglsTrend::Constant, AdfLagSelection::Aic(None)).is_ok());
    assert!(ng_perron(&y, DfglsTrend::Constant, NgPerronLagSelection::Maic(None)).is_ok());
}
