//! Property tests for the realized-measure and HAR machinery. All
//! randomness is a deterministic xorshift so the suite is reproducible
//! without a third-party RNG dependency.

use tsecon_realized::{
    bipower_variation, bns_jump_ratio, har_rv, jump_component, realized_quarticity,
    realized_variance, tripower_quarticity, HarConfig,
};

/// Reproducible xorshift64* with Box-Muller standard normals.
struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng {
            state: seed | 1,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn uniform(&mut self) -> f64 {
        // 53-bit mantissa in (0, 1).
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        // Avoid exactly zero for the log in Box-Muller.
        u.max(f64::MIN_POSITIVE)
    }

    fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f64::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// All realized measures are nonnegative on arbitrary finite return data.
#[test]
fn measures_are_nonnegative() {
    let mut rng = Rng::new(0xA11CE);
    for _ in 0..200 {
        let n = 3 + (rng.next_u64() % 60) as usize;
        let r: Vec<f64> = (0..n).map(|_| 1.5 * rng.normal()).collect();
        assert!(realized_variance(&r).unwrap() >= 0.0);
        assert!(bipower_variation(&r).unwrap() >= 0.0);
        assert!(realized_quarticity(&r).unwrap() >= 0.0);
        assert!(tripower_quarticity(&r).unwrap() >= 0.0);
        assert!(jump_component(&r).unwrap() >= 0.0);
    }
}

/// With jumps present, realized variance (which captures the jumps)
/// exceeds bipower variation (which is jump-robust) on average.
#[test]
fn rv_exceeds_bv_on_average_for_jumpy_data() {
    let mut rng = Rng::new(0x11BE12);
    let days = 400;
    let per_day = 79;
    let mut sum_rv = 0.0;
    let mut sum_bv = 0.0;
    for _ in 0..days {
        let mut r: Vec<f64> = (0..per_day).map(|_| rng.normal()).collect();
        // Inject a jump on roughly a third of the days.
        if rng.uniform() < 0.33 {
            let idx = (rng.next_u64() as usize) % per_day;
            r[idx] += if rng.uniform() < 0.5 { 8.0 } else { -8.0 };
        }
        sum_rv += realized_variance(&r).unwrap();
        sum_bv += bipower_variation(&r).unwrap();
    }
    let mean_rv = sum_rv / days as f64;
    let mean_bv = sum_bv / days as f64;
    assert!(
        mean_rv > mean_bv,
        "mean RV {mean_rv} should exceed mean BV {mean_bv} under jumps"
    );
}

/// On a persistent AR(1) realized-variance series the HAR coefficient sum
/// `beta_d + beta_w + beta_m` recovers the RV persistence.
#[test]
fn har_coefficients_sum_near_persistence() {
    let mut rng = Rng::new(0x5EED5);
    let rho = 0.9_f64;
    let c = 0.5_f64;
    let sigma = 0.3_f64;
    let n = 3000;
    let mut rv = vec![c / (1.0 - rho); n];
    for t in 1..n {
        let mut v = c + rho * rv[t - 1] + sigma * rng.normal();
        if v < 1e-6 {
            v = 1e-6;
        }
        rv[t] = v;
    }
    let fit = har_rv(&rv, &HarConfig::default()).unwrap();
    let coef_sum = fit.params[1] + fit.params[2] + fit.params[3];
    assert!(
        (coef_sum - rho).abs() < 0.15,
        "HAR coefficient sum {coef_sum} should be near persistence {rho}"
    );
    assert!(coef_sum < 1.0, "persistent-but-stationary: coef sum < 1");
}

/// Regression pin of the HAR window definition itself (field report 0.5,
/// finding 2): the Corsi (2009) weekly/monthly aggregates INCLUDE the
/// daily lag — for target `t`, weekly = `mean(RV[t-5..t])` and monthly =
/// `mean(RV[t-22..t])`, both running through `RV_{t-1}`. Through 0.4.0 the
/// crate shifted both windows one day back (`mean(RV[t-6..t-1])` /
/// `mean(RV[t-23..t-1])`), excluding the daily lag, while citing Corsi.
///
/// The pin is by construction, not by stored numbers: `har_rv` must equal
/// an OLS on an independently built inclusive-window design to near
/// machine precision, and must *differ* detectably from the same OLS on
/// the excluding-window design (the two designs are verified to genuinely
/// differ on this series first, so the inequality assertion has teeth).
#[test]
fn har_windows_include_the_daily_lag() {
    use tsecon_hac::ols;

    let mut rng = Rng::new(0xC0251);
    let n = 220usize;
    // Varied, strictly positive, deterministic realized-variance series.
    let rv: Vec<f64> = (0..n)
        .map(|_| (1.0 + 0.5 * rng.normal()).powi(2) + 0.05)
        .collect();

    let cfg = HarConfig {
        use_correction: false,
        ..HarConfig::default()
    };
    let fit = har_rv(&rv, &cfg).unwrap();

    let first = cfg.start + 1;
    let rows = n - first;
    let mut y = Vec::with_capacity(rows);
    let mut daily = Vec::with_capacity(rows);
    let mut wk_inc = Vec::with_capacity(rows);
    let mut mo_inc = Vec::with_capacity(rows);
    let mut wk_exc = Vec::with_capacity(rows);
    let mut mo_exc = Vec::with_capacity(rows);
    for t in first..n {
        y.push(rv[t]);
        daily.push(rv[t - 1]);
        wk_inc.push(rv[t - 5..t].iter().sum::<f64>() / 5.0);
        mo_inc.push(rv[t - 22..t].iter().sum::<f64>() / 22.0);
        wk_exc.push(rv[t - 6..t - 1].iter().sum::<f64>() / 5.0);
        mo_exc.push(rv[t - 23..t - 1].iter().sum::<f64>() / 22.0);
    }

    // Including vs excluding RV_{t-1} gives detectably different weekly
    // means on this series — the distinguishing power of the test.
    let weekly_gap = wk_inc
        .iter()
        .zip(&wk_exc)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        weekly_gap > 0.05,
        "constructed series does not distinguish the window conventions \
         (max weekly gap {weekly_gap})"
    );

    let ones = vec![1.0; rows];
    let inclusive = ols(&y, &[ones.clone(), daily.clone(), wk_inc, mo_inc]).unwrap();
    let excluding = ols(&y, &[ones, daily, wk_exc, mo_exc]).unwrap();

    for (i, (&a, &e)) in fit.params.iter().zip(&inclusive.params).enumerate() {
        assert!(
            (a - e).abs() <= 1e-10,
            "param {i}: har_rv {a} vs inclusive-window OLS {e} — the HAR \
             design no longer matches the Corsi windows"
        );
    }
    let dist_to_old = fit
        .params
        .iter()
        .zip(&excluding.params)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        dist_to_old > 1e-3,
        "har_rv matches the pre-0.5 excluding-window design (max param \
         distance {dist_to_old}) — the window regression has regressed"
    );
}

/// The BNS ratio jump statistic flags a jump-injected day more strongly
/// than the same continuous path without the jump.
#[test]
fn jump_test_flags_injected_jump() {
    let mut rng = Rng::new(0x105E7);
    let per_day = 79;
    let mut flagged_more = 0;
    let trials = 100;
    for _ in 0..trials {
        let cont: Vec<f64> = (0..per_day).map(|_| rng.normal()).collect();
        let mut jumpy = cont.clone();
        let idx = (rng.next_u64() as usize) % per_day;
        jumpy[idx] += 9.0;
        let z_cont = bns_jump_ratio(&cont).unwrap();
        let z_jump = bns_jump_ratio(&jumpy).unwrap();
        if z_jump > z_cont {
            flagged_more += 1;
        }
    }
    // The jump should raise the statistic on the overwhelming majority of
    // draws (a single continuous path can occasionally have a large z, but
    // adding a 9-sigma jump essentially always increases it).
    assert!(
        flagged_more >= 95,
        "jump raised the statistic on only {flagged_more}/{trials} draws"
    );
}

/// `theta = pi^2/4 + pi - 5` — kept in the tests as an independent
/// transcription of the studentization constant.
const THETA: f64 =
    core::f64::consts::PI * core::f64::consts::PI / 4.0 + core::f64::consts::PI - 5.0;

/// The unadjusted BNS-2004 form of the ratio statistic — the pre-0.6
/// construction, rebuilt from the exported measures — for contrast with
/// the Huang-Tauchen form the crate now computes.
fn bns_2004_z(r: &[f64]) -> f64 {
    let rv = realized_variance(r).unwrap();
    let bv = bipower_variation(r).unwrap();
    let tq = tripower_quarticity(r).unwrap();
    let m = r.len() as f64;
    m.sqrt() * ((rv - bv) / rv) / (THETA * (tq / (bv * bv)).max(1.0)).sqrt()
}

/// The Huang-Tauchen (2005) form, rebuilt independently from the exported
/// BNS-2004 measures with the finite-sample `M/(M-1)` / `M/(M-2)` scalings.
fn huang_tauchen_z(r: &[f64]) -> f64 {
    let rv = realized_variance(r).unwrap();
    let m = r.len() as f64;
    let bv = bipower_variation(r).unwrap() * m / (m - 1.0);
    let tq = tripower_quarticity(r).unwrap() * m / (m - 2.0);
    m.sqrt() * ((rv - bv) / rv) / (THETA * (tq / (bv * bv)).max(1.0)).sqrt()
}

/// Regression pin of the round-9 finding: `bns_jump_ratio` is the Huang &
/// Tauchen (2005) statistic — the finite-sample `M/(M-1)` factor on bipower
/// variation and `M/(M-2)` on tripower quarticity applied INSIDE the test —
/// not the unadjusted BNS-2004 assembly it computed through 0.5.0. The pin
/// is by construction: the crate value must match an independent HT
/// transcription to near machine precision and must *differ* detectably
/// from the unadjusted form (verified to genuinely differ first, so the
/// inequality has teeth).
#[test]
fn bns_ratio_is_the_huang_tauchen_adjusted_statistic() {
    // The documented 7-return day from fixtures/realized.json.
    let small = [0.5, -0.3, 0.8, -1.2, 0.1, 0.4, -0.6];
    // And a seeded 78-bar (five-minute grid) day.
    let mut rng = Rng::new(0x1778);
    let day: Vec<f64> = (0..78).map(|_| 0.1 * rng.normal()).collect();

    for r in [&small[..], &day[..]] {
        let z = bns_jump_ratio(r).unwrap();
        let z_ht = huang_tauchen_z(r);
        let z_bns = bns_2004_z(r);
        assert!(
            (z - z_ht).abs() <= 1e-12,
            "crate z {z} vs independent Huang-Tauchen transcription {z_ht}"
        );
        assert!(
            (z_ht - z_bns).abs() > 0.05,
            "the two constructions do not separate on this day \
             (HT {z_ht} vs BNS-2004 {z_bns}) — the pin has no teeth"
        );
        assert!(
            (z - z_bns).abs() > 0.05,
            "crate z {z} matches the pre-0.6 unadjusted BNS-2004 assembly \
             {z_bns} — the Huang-Tauchen adjustment has regressed"
        );
    }
}

/// The measured decision flip that motivated the 0.6 fix: on a seeded
/// `M = 78` day with one modest (5-sigma-per-bar) jump, the unadjusted
/// BNS-2004 assembly reads z = 1.689391118323 while the Huang-Tauchen
/// statistic reads z = 1.564353999278 — the one-sided 5% call (1.645)
/// flips. Pins both numbers so the shift stays measured.
#[test]
fn huang_tauchen_adjustment_flips_a_marginal_five_percent_call() {
    let mut rng = Rng::new(0xBEEF);
    let m = 78usize;
    let mut r: Vec<f64> = (0..m).map(|_| 0.1 * rng.normal()).collect();
    r[40] += 0.5; // one 5-sigma bar on a sigma = 0.1 grid

    let z_new = bns_jump_ratio(&r).unwrap();
    let z_old = bns_2004_z(&r);
    assert!(
        (z_old - 1.689391118323).abs() < 1e-9,
        "unadjusted BNS-2004 z moved: {z_old}"
    );
    assert!(
        (z_new - 1.564353999278).abs() < 1e-9,
        "Huang-Tauchen z moved: {z_new}"
    );
    assert!(
        z_old > 1.645 && z_new < 1.645,
        "the marginal 5% decision no longer flips (old {z_old}, new {z_new})"
    );
}

/// Seeded null-size Monte Carlo for the corrected statistic at `M = 78`
/// (the five-minute US-equity grid): iid Gaussian days, one-sided 5% test
/// at 1.645. Measured rejection rate 0.053 over 4000 reps (the number the
/// realized-vol model card quotes); the asserted band is loose enough for
/// the seeded draw, tight enough to catch a mis-sized statistic.
#[test]
fn null_size_near_nominal_at_m78() {
    let mut rng = Rng::new(0x512E);
    let reps = 4000usize;
    let m = 78usize;
    let mut rejections = 0usize;
    for _ in 0..reps {
        let r: Vec<f64> = (0..m).map(|_| 0.01 * rng.normal()).collect();
        if bns_jump_ratio(&r).unwrap() > 1.645 {
            rejections += 1;
        }
    }
    let rate = rejections as f64 / reps as f64;
    assert!(
        (0.03..=0.08).contains(&rate),
        "one-sided 5% null rejection rate {rate} out of band at M = 78"
    );
}
