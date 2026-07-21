//! Monte-Carlo property tests — the statistical validation of the crate.
//! Golden fixtures pin the *algebra*; these seeded simulations establish
//! that the algebra is the *statistically correct* one.
//!
//! * (a) EXACTNESS: on random small-T DGPs the dynamic program returns the
//!   same optimal SSR and the same break dates as an in-test brute force
//!   built from `tsecon_hac::ols` segment regressions — a code path
//!   independent of the crate's recursive normal-equation engine.
//! * (b) RECOVERY: a two-break mean-shift DGP with strong breaks is
//!   selected as two breaks and both dates land within ±2 periods of the
//!   truth in almost every replication.
//! * (c) SIZE: on no-break data the sequential procedure keeps its
//!   nominal ~5% false-break rate.
//! * (d) CONSISTENCY: the sup-F statistic equals the maximum of a
//!   Chow-style F path computed independently from `tsecon_hac::ols`
//!   subsample regressions, date by date.
//!
//! All randomness is the library's seeded Philox stream (`tsecon_rng`);
//! the numbers below are reproducible run to run.

use tsecon_breaks::{bai_perron, hansen_supf_pvalue, sup_f_test, BaiPerronConfig};
use tsecon_hac::ols;
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

fn gaussian(s: &mut Stream) -> f64 {
    let u = s.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// Segment SSR via the library's OLS owner — independent of the crate's
/// incremental normal-equation engine.
fn ols_ssr(y: &[f64], x: &[Vec<f64>], start: usize, end: usize) -> f64 {
    let ys = &y[start..=end];
    let cols: Vec<Vec<f64>> = x.iter().map(|c| c[start..=end].to_vec()).collect();
    let fit = ols(ys, &cols).expect("segment OLS");
    fit.residuals.iter().map(|r| r * r).sum()
}

/// Exhaustive search over admissible m-break partitions.
fn brute_force(y: &[f64], x: &[Vec<f64>], h: usize, m: usize) -> (f64, Vec<usize>) {
    let mut best = (f64::INFINITY, Vec::new());
    let mut dates = vec![0_usize; m];
    #[allow(clippy::too_many_arguments)]
    fn rec(
        y: &[f64],
        x: &[Vec<f64>],
        h: usize,
        m: usize,
        level: usize,
        first: usize,
        dates: &mut Vec<usize>,
        best: &mut (f64, Vec<usize>),
    ) {
        let t = y.len();
        if level == m {
            let mut ssr = 0.0;
            let mut s = 0_usize;
            for &d in dates.iter() {
                ssr += ols_ssr(y, x, s, d);
                s = d + 1;
            }
            ssr += ols_ssr(y, x, s, t - 1);
            if ssr < best.0 {
                *best = (ssr, dates.clone());
            }
            return;
        }
        // date for regime `level` (0-indexed last obs), leaving room for
        // the remaining m - level - 1 breaks plus the final regime.
        let remaining = m - level; // breaks still to place including this one
        for d in first..=(t - 1 - remaining * h) {
            dates[level] = d;
            rec(y, x, h, m, level + 1, d + h, dates, best);
        }
    }
    rec(y, x, h, m, 0, h - 1, &mut dates, &mut best);
    (best.0, best.1)
}

#[test]
fn dp_equals_bruteforce_on_random_dgps() {
    // Target (a): 24 random DGPs, T = 30, q in {1, 2}, trim = 0.2 (h = 6).
    let mut s = Stream::new(0xB41_0001);
    for rep in 0..24 {
        let t = 30;
        let q2 = rep % 2 == 1;
        let x1: Vec<f64> = (0..t).map(|_| gaussian(&mut s)).collect();
        // A haphazard mix: drift, occasional shift, noise.
        let shift_at = 8 + (rep * 7) % 14;
        let y: Vec<f64> = (0..t)
            .map(|i| {
                let base = if i >= shift_at { 1.5 } else { 0.0 };
                let slope = if q2 { 0.8 * x1[i] } else { 0.0 };
                base + slope + gaussian(&mut s)
            })
            .collect();
        let x: Vec<Vec<f64>> = if q2 {
            vec![vec![1.0; t], x1]
        } else {
            vec![vec![1.0; t]]
        };
        let bp = bai_perron(
            &y,
            &x,
            BaiPerronConfig {
                max_breaks: 2,
                trim: 0.2,
            },
        )
        .expect("bai_perron");
        for m in 1..=2 {
            let (bssr, bdates) = brute_force(&y, &x, bp.h, m);
            let rel = (bp.ssr_path[m] - bssr).abs() / (1.0 + bssr);
            assert!(
                rel < 1e-8,
                "rep {rep} m={m}: DP SSR {} vs brute force {bssr}",
                bp.ssr_path[m]
            );
            assert_eq!(
                bp.break_dates_by_m[m - 1],
                bdates,
                "rep {rep} m={m}: DP dates must equal brute force"
            );
        }
    }
}

#[test]
fn two_break_dgp_is_recovered_within_two_periods() {
    // Target (b): T = 150, mean shifts 0 -> 3 -> -2 at dates 49 and 99.
    let mut s = Stream::new(0xB41_0002);
    let (t, d1, d2) = (150_usize, 49_usize, 99_usize);
    let reps = 120;
    let mut selected_two = 0;
    let mut dates_close = 0;
    for _ in 0..reps {
        let y: Vec<f64> = (0..t)
            .map(|i| {
                let mu = if i <= d1 {
                    0.0
                } else if i <= d2 {
                    3.0
                } else {
                    -2.0
                };
                mu + gaussian(&mut s)
            })
            .collect();
        let x = vec![vec![1.0; t]];
        let bp = bai_perron(
            &y,
            &x,
            BaiPerronConfig {
                max_breaks: 3,
                trim: 0.15,
            },
        )
        .expect("bai_perron");
        if bp.n_breaks == 2 {
            selected_two += 1;
            let e1 = bp.break_dates[0].abs_diff(d1);
            let e2 = bp.break_dates[1].abs_diff(d2);
            if e1 <= 2 && e2 <= 2 {
                dates_close += 1;
            }
        }
    }
    let sel_rate = selected_two as f64 / reps as f64;
    assert!(
        sel_rate >= 0.85,
        "two breaks selected in only {selected_two}/{reps} replications"
    );
    let close_rate = dates_close as f64 / selected_two.max(1) as f64;
    assert!(
        close_rate >= 0.90,
        "dates within ±2 in only {dates_close}/{selected_two} of the selected runs"
    );
}

#[test]
fn sequential_procedure_holds_size_on_no_break_data() {
    // Target (c): under H0 (no break) the false-break rate of the 5%
    // sequential procedure should sit near 5%; the band below is a
    // ~3-sigma Monte-Carlo envelope around the asymptotic level, wide
    // enough for finite-sample (T = 200) size distortion.
    let mut s = Stream::new(0xB41_0003);
    let t = 200;
    let reps = 300;
    let mut false_breaks = 0;
    for _ in 0..reps {
        let y: Vec<f64> = (0..t).map(|_| gaussian(&mut s)).collect();
        let x = vec![vec![1.0; t]];
        let bp = bai_perron(
            &y,
            &x,
            BaiPerronConfig {
                max_breaks: 2,
                trim: 0.15,
            },
        )
        .expect("bai_perron");
        if bp.n_breaks > 0 {
            false_breaks += 1;
        }
    }
    let rate = false_breaks as f64 / reps as f64;
    assert!(
        (0.005..=0.115).contains(&rate),
        "false-break rate {rate:.3} (= {false_breaks}/{reps}) is not ~5%"
    );
}

#[test]
fn sup_f_equals_independent_chow_path() {
    // Target (d): the crate's f_path must equal, date by date, the
    // Wald-form Chow statistic assembled from tsecon_hac::ols subsample
    // regressions — and the sup must be its maximum.
    let mut s = Stream::new(0xB41_0004);
    let t = 120;
    let x1: Vec<f64> = (0..t).map(|_| gaussian(&mut s)).collect();
    let y: Vec<f64> = (0..t)
        .map(|i| {
            let b = if i < 70 { 0.4 } else { 1.6 };
            0.3 + b * x1[i] + gaussian(&mut s)
        })
        .collect();
    let x = vec![vec![1.0; t], x1];
    let r = sup_f_test(&y, &x, 0.15).expect("sup_f_test");
    let q = x.len();
    let ssr0 = ols_ssr(&y, &x, 0, t - 1);
    let mut best = f64::NEG_INFINITY;
    let mut best_date = 0;
    for (i, &d) in r.dates.iter().enumerate() {
        let split = ols_ssr(&y, &x, 0, d) + ols_ssr(&y, &x, d + 1, t - 1);
        let f = (t - 2 * q) as f64 * (ssr0 - split) / split;
        let rel = (r.f_path[i] - f).abs() / (1.0 + f.abs());
        assert!(rel < 1e-8, "f_path[{i}] (date {d}): {} vs {f}", r.f_path[i]);
        if f > best {
            best = f;
            best_date = d;
        }
    }
    assert!(
        (r.stat - best).abs() / (1.0 + best) < 1e-10,
        "sup-F {} vs independent max {best}",
        r.stat
    );
    assert_eq!(r.break_date, best_date, "argmax date");
    assert!(r.p_value < 0.05, "a strong slope break must reject");
}

#[test]
fn structural_invariants_hold() {
    let mut s = Stream::new(0xB41_0005);
    let t = 150;
    let x1: Vec<f64> = (0..t).map(|_| gaussian(&mut s)).collect();
    let y: Vec<f64> = (0..t)
        .map(|i| {
            let mu = if i < 60 { 0.0 } else { 2.0 };
            mu + 0.5 * x1[i] + gaussian(&mut s)
        })
        .collect();
    let x = vec![vec![1.0; t], x1];
    let bp = bai_perron(
        &y,
        &x,
        BaiPerronConfig {
            max_breaks: 3,
            trim: 0.15,
        },
    )
    .expect("bai_perron");
    // ssr_path is nonincreasing in m.
    for m in 1..bp.ssr_path.len() {
        assert!(
            bp.ssr_path[m] <= bp.ssr_path[m - 1] + 1e-9,
            "ssr_path must be nonincreasing"
        );
    }
    // Every partition is sorted with regimes at least h long.
    for dates in &bp.break_dates_by_m {
        let mut prev: i64 = -1;
        for &d in dates {
            assert!(d as i64 - prev >= bp.h as i64, "regime shorter than h");
            prev = d as i64;
        }
        assert!(
            t as i64 - 1 - prev >= bp.h as i64 - 1,
            "last regime too short"
        );
    }
    // Regimes tile the sample and agree with the selected dates.
    assert_eq!(bp.regimes.len(), bp.n_breaks + 1);
    assert_eq!(bp.regimes[0].start, 0);
    assert_eq!(bp.regimes[bp.regimes.len() - 1].end, t - 1);
    for w in bp.regimes.windows(2) {
        assert_eq!(w[1].start, w[0].end + 1, "regimes must tile the sample");
    }
    // Selection consistency: rejections strictly before n_breaks, and a
    // non-rejection at l = n_breaks when the sequence continued.
    for l in 0..bp.n_breaks {
        assert!(bp.sup_f_seq[l] > bp.sup_f_crit[l]);
    }
    if bp.n_breaks < bp.sup_f_seq.len() {
        assert!(bp.sup_f_seq[bp.n_breaks] <= bp.sup_f_crit[bp.n_breaks]);
    }
    // CI sanity: intervals bracket the dates and 95% contains 90%.
    for ci in &bp.ci {
        assert!(ci.lower90 <= ci.date && ci.date <= ci.upper90);
        assert!(ci.lower95 <= ci.lower90 && ci.upper90 <= ci.upper95);
    }
}

#[test]
fn hansen_pvalue_is_monotone_and_bounded() {
    for q in 1..=10 {
        let mut prev = 1.0_f64;
        for i in 0..40 {
            let stat = 0.5 * i as f64;
            let p = hansen_supf_pvalue(stat, q, 0.15).expect("p-value");
            assert!((0.0..=1.0).contains(&p), "p in [0,1]");
            assert!(
                p <= prev + 1e-12,
                "p-value must be nonincreasing in the statistic (q={q})"
            );
            prev = p;
        }
    }
}
