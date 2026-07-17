//! Shared helpers for the tsecon-optim integration tests.
#![allow(dead_code)]

/// Small deterministic LCG so tests need no RNG dependency.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493))
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / (1u64 << 53) as f64
    }

    /// Uniform in [lo, hi).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

/// Chained Rosenbrock: `sum_i 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2`
/// (More-Garbow-Hillstrom 1981); global minimum 0 at all ones.
pub fn rosenbrock(x: &[f64]) -> f64 {
    let mut f = 0.0;
    for i in 0..x.len() - 1 {
        let t = x[i + 1] - x[i] * x[i];
        f += 100.0 * t * t + (1.0 - x[i]) * (1.0 - x[i]);
    }
    f
}

/// Analytic gradient of [`rosenbrock`].
pub fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut g = vec![0.0; n];
    for i in 0..n - 1 {
        let t = x[i + 1] - x[i] * x[i];
        g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i]);
        g[i + 1] += 200.0 * t;
    }
    g
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Determinant by Gaussian elimination with partial pivoting (test-only,
/// small n).
pub fn det(mut m: Vec<Vec<f64>>) -> f64 {
    let n = m.len();
    let mut d = 1.0;
    for k in 0..n {
        let piv = (k..n)
            .max_by(|&a, &b| m[a][k].abs().total_cmp(&m[b][k].abs()))
            .unwrap();
        if m[piv][k] == 0.0 {
            return 0.0;
        }
        if piv != k {
            m.swap(piv, k);
            d = -d;
        }
        d *= m[k][k];
        for i in k + 1..n {
            let factor = m[i][k] / m[k][k];
            let (top, bottom) = m.split_at_mut(i);
            for (mij, &mkj) in bottom[0].iter_mut().zip(&top[k]).skip(k) {
                *mij -= factor * mkj;
            }
        }
    }
    d
}

/// Numerical Jacobian of a transform's forward map by central differences.
pub fn numerical_jacobian<T: tsecon_optim::Transform>(t: &T, z: &[f64]) -> Vec<Vec<f64>> {
    let n = z.len();
    let h = 1e-6;
    let mut jac = vec![vec![0.0; n]; n];
    let mut zp = z.to_vec();
    let mut tp = vec![0.0; n];
    let mut tm = vec![0.0; n];
    for j in 0..n {
        zp[j] = z[j] + h;
        t.forward(&zp, &mut tp).unwrap();
        zp[j] = z[j] - h;
        t.forward(&zp, &mut tm).unwrap();
        zp[j] = z[j];
        for i in 0..n {
            jac[i][j] = (tp[i] - tm[i]) / (2.0 * h);
        }
    }
    jac
}

// ---- Complex arithmetic + Durand-Kerner roots (test-only) ----

type C = (f64, f64);

fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn csub(a: C, b: C) -> C {
    (a.0 - b.0, a.1 - b.1)
}

fn cdiv(a: C, b: C) -> C {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

fn cabs(a: C) -> f64 {
    a.0.hypot(a.1)
}

/// Evaluates the monic polynomial `z^p + c[p-1] z^{p-1} + ... + c[0]`.
fn poly_eval(c: &[f64], z: C) -> C {
    let mut acc: C = (1.0, 0.0);
    for &ck in c.iter().rev() {
        acc = cmul(acc, z);
        acc = (acc.0 + ck, acc.1);
    }
    acc
}

/// All roots of the monic polynomial `z^p + c[p-1] z^{p-1} + ... + c[0]`
/// by Durand-Kerner iteration. Returns (roots, max residual |q(root)|).
pub fn durand_kerner(c: &[f64]) -> (Vec<C>, f64) {
    let p = c.len();
    let mut roots: Vec<C> = Vec::with_capacity(p);
    let mut cur: C = (1.0, 0.0);
    let seed: C = (0.4, 0.9);
    for _ in 0..p {
        cur = cmul(cur, seed);
        roots.push(cur);
    }
    for _ in 0..500 {
        let mut delta = 0.0f64;
        for i in 0..p {
            let mut denom: C = (1.0, 0.0);
            for j in 0..p {
                if j != i {
                    denom = cmul(denom, csub(roots[i], roots[j]));
                }
            }
            let step = cdiv(poly_eval(c, roots[i]), denom);
            roots[i] = csub(roots[i], step);
            delta = delta.max(cabs(step));
        }
        if delta < 1e-14 {
            break;
        }
    }
    let resid = roots
        .iter()
        .map(|&r| cabs(poly_eval(c, r)))
        .fold(0.0, f64::max);
    (roots, resid)
}

/// Maximum root modulus of the AR companion polynomial
/// `z^p - phi_1 z^{p-1} - ... - phi_p`; the AR process is stationary iff
/// this is < 1 (Hamilton 1994, prop. 1.1).
pub fn max_ar_root_modulus(phi: &[f64]) -> f64 {
    let p = phi.len();
    // Monic coefficients c[k] for z^k, k = 0..p-1: c[p-1-j] = -phi_{j+1}.
    let mut c = vec![0.0; p];
    for (j, &ph) in phi.iter().enumerate() {
        c[p - 1 - j] = -ph;
    }
    let (roots, resid) = durand_kerner(&c);
    assert!(
        resid < 1e-8,
        "Durand-Kerner did not converge (residual {resid:.3e})"
    );
    roots.iter().map(|&r| cabs(r)).fold(0.0, f64::max)
}
