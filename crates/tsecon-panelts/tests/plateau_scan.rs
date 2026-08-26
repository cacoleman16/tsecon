//! TEMPORARY diagnostic (removed before commit): where does the PMG
//! back-substitution's relative update plateau on the I(1) battery?
use tsecon_panelts::{pmg_with, PanelUnit};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

fn gaussian(stream: &mut Stream) -> f64 {
    let u = stream.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

fn battery_panel(seed: u64, kind: &str) -> Vec<PanelUnit> {
    let (n, t) = (10usize, 150usize);
    let mut s = Stream::new(seed);
    (0..n)
        .map(|_| {
            let mut x = vec![0.0_f64; t];
            let mut acc = 0.0_f64;
            for xv in x.iter_mut() {
                let e = gaussian(&mut s);
                if kind == "i0" {
                    *xv = e;
                } else {
                    acc += e;
                    *xv = acc;
                }
            }
            if kind == "i1x100" {
                for xv in x.iter_mut() {
                    *xv *= 100.0;
                }
            }
            let mut y = vec![0.0_f64; t];
            y[0] = x[0];
            for tt in 1..t {
                let dx = x[tt] - x[tt - 1];
                let dy = -0.3 * (y[tt - 1] - x[tt - 1]) + 0.2 * dx + 0.1 * gaussian(&mut s);
                y[tt] = y[tt - 1] + dy;
            }
            PanelUnit::new(y, vec![x])
        })
        .collect()
}

#[test]
fn scan_tolerances() {
    for kind in ["i1", "i0", "i1x100"] {
        for seed in 0..20u64 {
            let units = battery_panel(seed, kind);
            let mut line = format!("{kind} seed {seed:2}: ");
            for tol in [1e-12_f64, 1e-11, 1e-10, 1e-9] {
                match pmg_with(&units, tol, 1000) {
                    Ok(fit) => line.push_str(&format!(
                        "tol {tol:.0e} ok it {:3} th {:.15e} | ",
                        fit.iterations, fit.theta[0]
                    )),
                    Err(_) => line.push_str(&format!("tol {tol:.0e} FAIL | ")),
                }
            }
            println!("{line}");
        }
    }
}
