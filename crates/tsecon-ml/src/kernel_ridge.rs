//! Kernel ridge regression — exact (dual Cholesky solve) and the
//! Rahimi-Recht random-Fourier-feature primal approximation.
//!
//! # Objective (scikit-learn convention)
//!
//! [`KernelRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)
//! minimizes, over a function `f` in the reproducing-kernel Hilbert space
//! of the kernel `k`,
//!
//! ```text
//! sum_i (y_i - f(x_i))^2 + alpha * ||f||_H^2 ,
//! ```
//!
//! whose representer solution `f(x) = sum_i a_i k(x, x_i)` has dual
//! coefficients
//!
//! ```text
//! (K + alpha I) a = y ,    K_ij = k(x_i, x_j) .
//! ```
//!
//! No `1/n` factor and no intercept — exactly the [`crate::ridge`]
//! objective with the design replaced by the kernel matrix (with the
//! linear kernel the two agree: `K = X X'`). scikit-learn solves the
//! system by Cholesky (`_solve_cholesky_kernel`) and so does this module;
//! where scikit-learn silently falls back to a least-squares solve when
//! `K + alpha I` is not positive definite, this module **refuses** with
//! [`MlError::NotPositiveDefinite`] naming `alpha` — the fallback answers a
//! different problem and a user should know.
//!
//! # Kernels (scikit-learn's exact parameterizations)
//!
//! ```text
//! rbf         k(x, y) = exp(-gamma ||x - y||_2^2)
//! laplacian   k(x, y) = exp(-gamma ||x - y||_1)
//! polynomial  k(x, y) = (gamma <x, y> + coef0)^degree
//! linear      k(x, y) = <x, y>
//! ```
//!
//! `gamma = None` resolves to scikit-learn's default `1 / n_features` for
//! the three `gamma` kernels; the linear kernel has no `gamma`, and an
//! explicitly supplied one is refused rather than silently ignored (the
//! library-wide inert-argument rule). `degree` and `coef0` act only on the
//! polynomial kernel and are refused at non-default values elsewhere.
//!
//! # Random Fourier features (Rahimi & Recht 2007)
//!
//! For the rbf kernel, Bochner's theorem gives `k(x, y) = E_w[ cos(w'x + b)
//! cos(w'y + b) ] * 2` with `w ~ N(0, 2 gamma I)` and `b ~ U[0, 2 pi)`, so
//! the `D`-dimensional feature map
//!
//! ```text
//! z(x) = sqrt(2 / D) [ cos(w_1'x + b_1), ..., cos(w_D'x + b_D) ]
//! ```
//!
//! satisfies `z(x)'z(y) -> k(x, y)` as `D -> infinity`. Ridge on `Z`
//! (the [`crate::ridge`] closed form, same objective) is then an `O(n D^2)`
//! primal approximation of the `O(n^3)` exact solve — the roadmap's
//! "tuning-light nonlinear baseline" for long samples. The draws come from
//! a [`tsecon_rng::Stream`] keyed by `seed` (Philox; bit-identical on every
//! platform), normals by Box-Muller. The approximation is **not**
//! golden-pinned (it is a Monte-Carlo object); the crate's property tests
//! pin seeded determinism and convergence to the exact fit as `D` grows.

use tsecon_linalg::faer::{Mat, MatRef, Side};
use tsecon_rng::Stream;

use crate::error::MlError;
use crate::ridge::ridge;
use crate::util::check_xy;

/// The kernel family, in scikit-learn's parameterization (see the
/// [module docs](self#kernels-scikit-learns-exact-parameterizations)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    /// `exp(-gamma ||x - y||_2^2)`.
    Rbf,
    /// `exp(-gamma ||x - y||_1)`.
    Laplacian,
    /// `(gamma <x, y> + coef0)^degree`.
    Polynomial,
    /// `<x, y>` (no `gamma`).
    Linear,
}

impl KernelType {
    /// The accepted kernel names, in the order the teaching error lists them.
    pub const ACCEPTED: &'static [&'static str] = &["rbf", "laplacian", "polynomial", "linear"];

    /// Parses a kernel name.
    ///
    /// # Errors
    ///
    /// [`MlError::InvalidValue`] listing the accepted names.
    pub fn parse(name: &str) -> Result<Self, MlError> {
        match name {
            "rbf" => Ok(Self::Rbf),
            "laplacian" => Ok(Self::Laplacian),
            "polynomial" => Ok(Self::Polynomial),
            "linear" => Ok(Self::Linear),
            other => Err(MlError::InvalidValue {
                what: format!(
                    "unknown kernel {other:?}; accepted values are {}",
                    quoted_list(Self::ACCEPTED)
                ),
            }),
        }
    }

    /// The kernel's name as accepted by [`KernelType::parse`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rbf => "rbf",
            Self::Laplacian => "laplacian",
            Self::Polynomial => "polynomial",
            Self::Linear => "linear",
        }
    }

    /// Whether the kernel takes a `gamma` (every family but the linear).
    pub fn uses_gamma(self) -> bool {
        !matches!(self, Self::Linear)
    }
}

/// Formats `["a", "b"]` as `"a", "b"` for teaching errors.
pub(crate) fn quoted_list(names: &[&str]) -> String {
    names
        .iter()
        .map(|s| format!("{s:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Configuration of a [`kernel_ridge`] fit.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRidgeOptions {
    /// Ridge penalty `alpha >= 0` on the RKHS norm (scikit-learn scale: no
    /// `1/n`).
    pub alpha: f64,
    /// Kernel family.
    pub kernel: KernelType,
    /// Kernel width / scale; `None` resolves to `1 / n_features`
    /// (scikit-learn's default). Must be `None` for the linear kernel.
    pub gamma: Option<f64>,
    /// Polynomial degree (polynomial kernel only; default `3`).
    pub degree: f64,
    /// Polynomial offset (polynomial kernel only; default `1`).
    pub coef0: f64,
    /// `Some(D)` switches to the `D`-feature random-Fourier-feature
    /// approximation (rbf kernel only); `None` is the exact dual solve.
    pub rff_features: Option<usize>,
    /// Seed of the feature draws (random-Fourier-feature mode only).
    pub seed: u64,
}

impl Default for KernelRidgeOptions {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            kernel: KernelType::Rbf,
            gamma: None,
            degree: 3.0,
            coef0: 1.0,
            rff_features: None,
            seed: 0,
        }
    }
}

/// Result of a [`kernel_ridge`] fit.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRidgeFit {
    /// Dual coefficients `a` (length `n`) — exact mode only.
    pub dual_coef: Option<Vec<f64>>,
    /// Primal coefficients on the random Fourier features (length `D`) —
    /// random-Fourier-feature mode only.
    pub coef: Option<Vec<f64>>,
    /// In-sample fitted values, length `n`.
    pub fitted: Vec<f64>,
    /// Predictions at the test rows, present only when `x_test` was given.
    pub predicted: Option<Vec<f64>>,
    /// The kernel family used.
    pub kernel: KernelType,
    /// The resolved `gamma` (`None` for the linear kernel).
    pub gamma: Option<f64>,
    /// The number of random Fourier features (`None` in exact mode).
    pub n_rff_features: Option<usize>,
}

/// The kernel matrix `K_ij = k(a_i, b_j)` between the rows of `a`
/// (`m x p`) and the rows of `b` (`q x p`).
///
/// `gamma` is the *resolved* width (ignored by the linear kernel);
/// `degree` and `coef0` act only on the polynomial kernel.
///
/// # Errors
///
/// [`MlError::DimensionMismatch`] if `a` and `b` have different column
/// counts.
pub fn kernel_matrix(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    kernel: KernelType,
    gamma: f64,
    degree: f64,
    coef0: f64,
) -> Result<Mat<f64>, MlError> {
    if a.ncols() != b.ncols() {
        return Err(MlError::DimensionMismatch {
            what: "kernel_matrix: both inputs must have the same number of columns",
            expected: a.ncols(),
            got: b.ncols(),
        });
    }
    let p = a.ncols();
    let (m, q) = (a.nrows(), b.nrows());
    let mut k = Mat::<f64>::zeros(m, q);
    for j in 0..q {
        for i in 0..m {
            let v = match kernel {
                KernelType::Rbf => {
                    let mut s = 0.0;
                    for c in 0..p {
                        let d = a[(i, c)] - b[(j, c)];
                        s += d * d;
                    }
                    (-gamma * s).exp()
                }
                KernelType::Laplacian => {
                    let mut s = 0.0;
                    for c in 0..p {
                        s += (a[(i, c)] - b[(j, c)]).abs();
                    }
                    (-gamma * s).exp()
                }
                KernelType::Polynomial => {
                    let mut s = 0.0;
                    for c in 0..p {
                        s += a[(i, c)] * b[(j, c)];
                    }
                    (gamma * s + coef0).powf(degree)
                }
                KernelType::Linear => {
                    let mut s = 0.0;
                    for c in 0..p {
                        s += a[(i, c)] * b[(j, c)];
                    }
                    s
                }
            };
            k[(i, j)] = v;
        }
    }
    Ok(k)
}

/// Validates the option block against the design and resolves `gamma`.
fn resolve_options(opts: &KernelRidgeOptions, p: usize) -> Result<Option<f64>, MlError> {
    if !opts.alpha.is_finite() || opts.alpha < 0.0 {
        return Err(MlError::InvalidValue {
            what: format!(
                "alpha={} must be finite and non-negative (the ridge penalty on the RKHS \
                 norm; alpha=0 is the interpolating fit, which is well posed only when the \
                 kernel matrix is itself positive definite)",
                opts.alpha
            ),
        });
    }
    if !opts.degree.is_finite() || opts.degree < 0.0 {
        return Err(MlError::InvalidValue {
            what: format!(
                "degree={} must be finite and non-negative (the polynomial kernel exponent)",
                opts.degree
            ),
        });
    }
    if !opts.coef0.is_finite() {
        return Err(MlError::InvalidValue {
            what: format!(
                "coef0={} must be finite (the polynomial kernel offset)",
                opts.coef0
            ),
        });
    }
    if opts.kernel != KernelType::Polynomial && (opts.degree != 3.0 || opts.coef0 != 1.0) {
        return Err(MlError::InvalidValue {
            what: format!(
                "degree={} / coef0={} have no effect under kernel={:?}: they parameterize the \
                 polynomial kernel (gamma <x, y> + coef0)^degree only and would be silently \
                 ignored. Pass kernel=\"polynomial\" for them to act, or leave them at their \
                 defaults (degree=3, coef0=1.0) with the {} kernel",
                opts.degree,
                opts.coef0,
                opts.kernel.as_str(),
                opts.kernel.as_str()
            ),
        });
    }
    let gamma = match (opts.kernel.uses_gamma(), opts.gamma) {
        (false, Some(g)) => {
            return Err(MlError::InvalidValue {
                what: format!(
                    "gamma={g} has no effect under kernel=\"linear\": the linear kernel is the \
                     plain inner product <x, y> with no width parameter, so gamma would be \
                     silently ignored. Drop gamma for the linear kernel, or pass kernel=\"rbf\", \
                     \"laplacian\" or \"polynomial\" for a kernel that uses it"
                ),
            })
        }
        (false, None) => None,
        (true, Some(g)) => {
            if !g.is_finite() || g <= 0.0 {
                return Err(MlError::InvalidValue {
                    what: format!(
                        "gamma={g} must be finite and positive (the kernel width: rbf \
                         exp(-gamma ||x - y||^2), laplacian exp(-gamma ||x - y||_1), \
                         polynomial (gamma <x, y> + coef0)^degree); leave gamma=None for \
                         scikit-learn's default 1 / n_features"
                    ),
                });
            }
            Some(g)
        }
        (true, None) => Some(1.0 / p as f64),
    };
    match opts.rff_features {
        Some(0) => {
            return Err(MlError::InvalidValue {
                what: "rff_features=0: the random-Fourier-feature approximation needs at \
                       least one feature (pass rff_features=None for the exact solve)"
                    .to_string(),
            })
        }
        Some(d) if opts.kernel != KernelType::Rbf => {
            return Err(MlError::InvalidValue {
                what: format!(
                    "rff_features={d} has no effect under kernel={:?}: random Fourier \
                     features (Rahimi & Recht 2007) sample the spectral measure of the \
                     shift-invariant rbf kernel and this implementation draws them for the \
                     rbf kernel only, so the argument would be silently ignored and the \
                     exact solve run instead. Pass kernel=\"rbf\" with rff_features, or drop \
                     rff_features for the exact {} solve",
                    opts.kernel.as_str(),
                    opts.kernel.as_str()
                ),
            })
        }
        Some(_) => {}
        None => {
            if opts.seed != 0 {
                return Err(MlError::InvalidValue {
                    what: format!(
                        "seed={} has no effect in exact mode (rff_features=None): the exact \
                         kernel ridge solve is deterministic and draws nothing, so the seed \
                         would be silently ignored. Pass rff_features=<D> for the seeded \
                         random-Fourier-feature approximation, or drop seed",
                        opts.seed
                    ),
                });
            }
        }
    }
    Ok(gamma)
}

/// Checks that an optional test design is finite and conformable.
fn check_x_test(x_test: Option<MatRef<'_, f64>>, p: usize) -> Result<(), MlError> {
    if let Some(xt) = x_test {
        if xt.ncols() != p {
            return Err(MlError::DimensionMismatch {
                what: "x_test must have the same number of columns as x",
                expected: p,
                got: xt.ncols(),
            });
        }
        for j in 0..xt.ncols() {
            for i in 0..xt.nrows() {
                if !xt[(i, j)].is_finite() {
                    return Err(MlError::NonFinite { what: "x_test" });
                }
            }
        }
    }
    Ok(())
}

/// `K v` for a dense `K`.
fn mat_vec(k: MatRef<'_, f64>, v: &[f64]) -> Vec<f64> {
    (0..k.nrows())
        .map(|i| (0..k.ncols()).map(|j| k[(i, j)] * v[j]).sum::<f64>())
        .collect()
}

/// One standard-normal draw by Box-Muller (two uniforms consumed; the
/// sine partner is discarded so the per-draw stream cost is fixed).
#[inline]
fn standard_normal(stream: &mut Stream) -> f64 {
    // uniform_f64 is on [0, 1); 1 - u is on (0, 1], keeping ln() finite.
    let u1 = 1.0 - stream.uniform_f64();
    let u2 = stream.uniform_f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

/// The random-Fourier-feature map `sqrt(2/D) cos(X W + b)` of the rows of
/// `x`, given the drawn frequencies `w` (`p x D`, column-major as
/// `w[j][c]`) and offsets `b`.
fn rff_features(x: MatRef<'_, f64>, w: &[Vec<f64>], b: &[f64]) -> Mat<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let d = b.len();
    let scale = (2.0 / d as f64).sqrt();
    Mat::from_fn(n, d, |i, j| {
        let mut s = b[j];
        for c in 0..p {
            s += x[(i, c)] * w[j][c];
        }
        scale * s.cos()
    })
}

/// Kernel ridge regression: `(K + alpha I) a = y` by Cholesky, or — with
/// `rff_features = Some(D)` — the Rahimi-Recht primal approximation on `D`
/// random Fourier features of the rbf kernel.
///
/// `x` is the `n x p` design, `y` the length-`n` target (no intercept is
/// fitted — center `y` if the kernel does not model a level; the
/// polynomial kernel with `coef0 > 0` and the rbf/laplacian kernels can
/// absorb one). `x_test` (`m x p`) adds `predicted`. See the
/// [module docs](self) for the objective, the kernels, and the feature
/// map.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on malformed inputs (`x_test` included);
/// * [`MlError::InvalidValue`] if `alpha` is negative, `gamma` is not
///   positive, `degree` is negative, `coef0` is not finite, or an argument
///   is supplied that the chosen mode cannot use (`gamma` with the linear
///   kernel, non-default `degree`/`coef0` with a non-polynomial kernel,
///   `rff_features` with a non-rbf kernel, `seed` in exact mode,
///   `rff_features = 0`);
/// * [`MlError::NotPositiveDefinite`] if `K + alpha I` fails its Cholesky
///   factorization (exact mode) — increase `alpha`;
/// * [`MlError::DecompositionFailed`] if the feature-space SVD fails
///   (random-Fourier-feature mode).
pub fn kernel_ridge(
    x: MatRef<'_, f64>,
    y: &[f64],
    x_test: Option<MatRef<'_, f64>>,
    opts: &KernelRidgeOptions,
) -> Result<KernelRidgeFit, MlError> {
    let (n, p) = check_xy(x, y)?;
    check_x_test(x_test, p)?;
    let gamma = resolve_options(opts, p)?;
    // Every kernel routine takes a resolved width; the linear kernel never
    // reads it.
    let g = gamma.unwrap_or(0.0);

    if let Some(d) = opts.rff_features {
        // Frequencies w_j ~ N(0, 2 gamma I): the spectral measure of
        // exp(-gamma ||x - y||^2) is Gaussian with variance 2 gamma per
        // coordinate. Offsets b_j ~ U[0, 2 pi). Draw order: for each
        // feature j, its p frequencies then its offset.
        let mut stream = Stream::new(opts.seed);
        let sd = (2.0 * g).sqrt();
        let mut w: Vec<Vec<f64>> = Vec::with_capacity(d);
        let mut b: Vec<f64> = Vec::with_capacity(d);
        for _ in 0..d {
            let wj: Vec<f64> = (0..p).map(|_| sd * standard_normal(&mut stream)).collect();
            w.push(wj);
            b.push(core::f64::consts::TAU * stream.uniform_f64());
        }
        let z = rff_features(x, &w, &b);
        let coef = ridge(z.as_ref(), y, opts.alpha)?;
        let fitted = mat_vec(z.as_ref(), &coef);
        let predicted = x_test.map(|xt| mat_vec(rff_features(xt, &w, &b).as_ref(), &coef));
        return Ok(KernelRidgeFit {
            dual_coef: None,
            coef: Some(coef),
            fitted,
            predicted,
            kernel: opts.kernel,
            gamma,
            n_rff_features: Some(d),
        });
    }

    let k = kernel_matrix(x, x, opts.kernel, g, opts.degree, opts.coef0)?;
    let mut k_alpha = k.clone();
    for i in 0..n {
        k_alpha[(i, i)] += opts.alpha;
    }
    let llt = k_alpha
        .as_ref()
        .llt(Side::Lower)
        .map_err(|_| MlError::NotPositiveDefinite {
            what: format!(
                "the kernel ridge system K + alpha*I (n = {n}) failed its Cholesky factorization \
             at alpha={}: the kernel matrix is numerically rank deficient (duplicate or \
             near-duplicate rows of x, a polynomial/linear kernel of rank at most p = {p}, \
             or an rbf/laplacian gamma so small that every row looks alike) and the penalty \
             is too small to make the system positive definite. Increase alpha (scikit-learn \
             would silently fall back to a least-squares solve of a different problem here; \
             tsecon refuses instead)",
                opts.alpha
            ),
        })?;
    let rhs = Mat::from_fn(n, 1, |i, _| y[i]);
    let sol = {
        use tsecon_linalg::faer::linalg::solvers::Solve;
        llt.solve(rhs.as_ref())
    };
    let dual: Vec<f64> = (0..n).map(|i| sol[(i, 0)]).collect();
    let fitted = mat_vec(k.as_ref(), &dual);
    let predicted = match x_test {
        Some(xt) => {
            let kt = kernel_matrix(xt, x, opts.kernel, g, opts.degree, opts.coef0)?;
            Some(mat_vec(kt.as_ref(), &dual))
        }
        None => None,
    };
    Ok(KernelRidgeFit {
        dual_coef: Some(dual),
        coef: None,
        fitted,
        predicted,
        kernel: opts.kernel,
        gamma,
        n_rff_features: None,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn small_design() -> (Mat<f64>, Vec<f64>) {
        // 8 rows, 2 columns; deterministic, no two rows alike.
        let x = Mat::from_fn(8, 2, |i, j| ((i * 3 + j * 7) % 11) as f64 / 3.0 - 1.5);
        let y: Vec<f64> = (0..8).map(|i| (i as f64 * 0.7).sin()).collect();
        (x, y)
    }

    #[test]
    fn linear_kernel_ridge_equals_primal_ridge_fit() {
        let (x, y) = small_design();
        let opts = KernelRidgeOptions {
            alpha: 0.5,
            kernel: KernelType::Linear,
            ..Default::default()
        };
        let fit = kernel_ridge(x.as_ref(), &y, None, &opts).unwrap();
        let beta = ridge(x.as_ref(), &y, 0.5).unwrap();
        for i in 0..8 {
            let primal: f64 = (0..2).map(|j| x[(i, j)] * beta[j]).sum();
            assert!((fit.fitted[i] - primal).abs() < 1e-12);
        }
        assert_eq!(fit.gamma, None);
        assert_eq!(fit.n_rff_features, None);
        assert!(fit.coef.is_none());
    }

    #[test]
    fn unknown_kernel_lists_accepted_names() {
        let e = KernelType::parse("cosine").unwrap_err();
        let msg = e.to_string();
        assert!(
            msg.contains("\"rbf\"") && msg.contains("\"linear\""),
            "{msg}"
        );
    }

    #[test]
    fn inert_arguments_are_refused() {
        let (x, y) = small_design();
        let base = KernelRidgeOptions::default();
        let cases = [
            KernelRidgeOptions {
                kernel: KernelType::Linear,
                gamma: Some(0.3),
                ..base.clone()
            },
            KernelRidgeOptions {
                degree: 2.0,
                ..base.clone()
            },
            KernelRidgeOptions {
                rff_features: Some(10),
                kernel: KernelType::Laplacian,
                ..base.clone()
            },
            KernelRidgeOptions {
                seed: 3,
                ..base.clone()
            },
            KernelRidgeOptions {
                rff_features: Some(0),
                ..base.clone()
            },
        ];
        for opts in &cases {
            let e = kernel_ridge(x.as_ref(), &y, None, opts).unwrap_err();
            assert!(matches!(e, MlError::InvalidValue { .. }), "{opts:?}: {e}");
        }
    }
}
