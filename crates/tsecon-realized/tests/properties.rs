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
