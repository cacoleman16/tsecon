// Regression: inv_gamma_p must be globally convergent for large shape `a`
// (SSVS draws Gamma with shape = gamma_a + T/2, which reaches the hundreds).
use tsecon_stats::special::{gamma_p, inv_gamma_p};

#[test]
fn inv_gamma_p_converges_and_is_accurate_for_large_a() {
    let mut worst = 0.0_f64;
    for &a in &[50.0, 200.0, 600.0, 750.0, 1000.0, 1500.0, 3000.0, 6000.0] {
        for i in 1..1000u32 {
            let p = i as f64 / 1000.0;
            let x = inv_gamma_p(a, p).expect("must converge");
            // round-trip: P(a, inv) == p to the resolution P(a, .) can deliver
            let resid = (gamma_p(a, x).unwrap() - p).abs();
            worst = worst.max(resid);
            assert!(resid < 1e-6, "a={a} p={p}: round-trip resid {resid:e}");
        }
    }
    println!("worst round-trip residual across grid: {worst:e}");
}
