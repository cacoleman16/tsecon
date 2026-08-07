//! Simultaneous (sup-t) critical values for bands over many cells at once.
//!
//! # The problem this solves
//!
//! A *pointwise* band is a statement about one cell: for each horizon `k`
//! separately, `theta_hat_k ± z * se_k` covers `theta_k` with probability
//! `1 - alpha`. Read as a statement about a whole *path* — "the impulse
//! response function lies inside this band" — a pointwise band is badly
//! anti-conservative, and the shortfall does not vanish as the sample grows:
//! it is a multiplicity problem, not a consistency problem. The library's own
//! interval-coverage audit measured a nominal 90% pointwise IRF band
//! containing the entire 13-horizon path in 72.2% of samples at `T = 500`, and
//! nominal 95% marginal forecast bands covering every horizon and every series
//! at once in 40.9% of samples at `T = 100` and still only 48.1% at
//! `T = 800`.
//!
//! A *simultaneous* band replaces the pointwise `z` with a larger constant `c`
//! chosen so that `theta_hat_k ± c * se_k` covers **all** `K` cells jointly
//! with probability `1 - alpha`. The band keeps the pointwise standard errors
//! and only widens the multiplier, which is what makes it a drop-in change for
//! every band-producing routine in the workspace.
//!
//! # The four routes
//!
//! | route | needs | tightness | when to use |
//! |---|---|---|---|
//! | [`sup_t_from_draws`] | bootstrap/posterior draws | tightest | any bootstrap band |
//! | [`sup_t_from_cov`] | the delta-method covariance | tightest | asymptotic bands |
//! | [`sidak_critical_value`] | nothing but `K` | loose | fallback |
//! | [`bonferroni_critical_value`] | nothing but `K` | loosest | fallback |
//!
//! The sup-t construction — take the max over cells of the absolute
//! t-statistic and read off its `1 - alpha` quantile — is the method of
//! Montiel Olea and Plagborg-Møller, *Simultaneous confidence bands: Theory,
//! implementation, and an application to SVARs*. Both sup-t routes here
//! implement that construction; the difference is only where the draws of the
//! max-|t| statistic come from.
//!
//! # This crate stays RNG-free
//!
//! [`sup_t_from_cov`] needs random draws, but `tsecon-stats` takes **uniforms
//! from the caller** rather than depending on `tsecon-rng`. That follows the
//! convention already set by [`crate::ContinuousDist::sample_from_uniform`]:
//! this crate does inverse-transform sampling on uniforms handed in from
//! outside, and owns no generator state. Every caller that wants an
//! asymptotic simultaneous band (`tsecon-var`, `tsecon-lp`) already depends on
//! `tsecon-rng`, so nothing is gained by moving the dependency down here, and
//! keeping the foundational crate dependency-free keeps seeding and stream
//! discipline in one place — the caller's — instead of two.
//!
//! # Guarantees
//!
//! * Every route returns a critical value `>= pointwise_critical_value(alpha)`.
//!   The closed forms satisfy this analytically; the two sup-t routes are
//!   explicitly floored at the pointwise value, because the population sup-t
//!   quantile can never fall below it (the max of `K` absolute t-statistics
//!   dominates any single one) and a "simultaneous" band narrower than its
//!   pointwise counterpart would be a silent, headline-grade bug.
//! * At `K = 1` every route collapses to the pointwise critical value —
//!   exactly for the closed forms, up to simulation error for the sup-t
//!   routes.
//! * Cells with `se == 0` are excluded from the maximum. The proxy-SVAR
//!   normalization pins one cell by construction, giving it exactly zero
//!   standard error; such a cell carries no information about simultaneous
//!   coverage and would otherwise put `0/0` or `x/0` into the max. Its band is
//!   still reported, with zero width.

use crate::error::StatsError;
use crate::special::inv_norm_cdf;

/// Half of the spacing of the `[0, 1)` grid produced by
/// `tsecon_rng::Stream::uniform_f64` (`2^-54`). Uniforms are shifted by this
/// before inversion so that an exact `0.0` draw — possible, with probability
/// `2^-53` — maps to a finite normal deviate instead of `-inf`.
const HALF_ULP53: f64 = 5.551_115_123_125_783e-17;

/// Pivots below `PSD_ZERO_TOL * max(diag(Sigma))` are treated as an exactly
/// deficient direction of a positive semi-definite matrix.
const PSD_ZERO_TOL: f64 = 1e-12;

/// Pivots below `-PSD_NEG_TOL * max(diag(Sigma))` are reported as an
/// indefinite matrix rather than absorbed as rounding noise.
const PSD_NEG_TOL: f64 = 1e-8;

/// Relative tolerance for the symmetry check on a supplied covariance matrix.
const SYMMETRY_TOL: f64 = 1e-8;

/// A simultaneous band: one critical value applied to every cell's pointwise
/// standard error.
#[derive(Debug, Clone, PartialEq)]
pub struct SimultaneousBand {
    /// The multiplier `c` applied to every standard error.
    pub critical_value: f64,
    /// `theta_hat_k - c * se_k`, in the input cell order.
    pub lower: Vec<f64>,
    /// `theta_hat_k + c * se_k`, in the input cell order.
    pub upper: Vec<f64>,
    /// Number of cells with a strictly positive standard error. Cells with
    /// `se == 0` are pinned by construction: they get a zero-width band and
    /// took no part in choosing `critical_value`.
    pub n_cells_used: usize,
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

fn check_alpha(alpha: f64) -> Result<(), StatsError> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(StatsError::InvalidParameter {
            name: "alpha",
            value: alpha,
            requirement: "0 < alpha < 1",
        });
    }
    Ok(())
}

fn check_k(k: usize) -> Result<(), StatsError> {
    if k == 0 {
        return Err(StatsError::InvalidParameter {
            name: "k",
            value: 0.0,
            requirement: "k >= 1 (at least one cell)",
        });
    }
    Ok(())
}

/// Indices of the cells that can enter the maximum: finite, strictly positive
/// standard errors. Errors on a negative or non-finite `se`, and on an
/// all-degenerate input.
fn active_cells(se: &[f64]) -> Result<Vec<usize>, StatsError> {
    let mut active = Vec::with_capacity(se.len());
    for (j, &s) in se.iter().enumerate() {
        if !s.is_finite() || s < 0.0 {
            return Err(StatsError::Domain {
                name: "se",
                value: s,
                requirement: "every standard error finite and >= 0",
            });
        }
        if s > 0.0 {
            active.push(j);
        }
    }
    if active.is_empty() {
        return Err(StatsError::Domain {
            name: "se",
            value: 0.0,
            requirement: "at least one cell with a strictly positive standard error",
        });
    }
    Ok(active)
}

// ---------------------------------------------------------------------------
// Closed-form critical values
// ---------------------------------------------------------------------------

/// The ordinary two-sided pointwise critical value `Phi^{-1}(1 - alpha/2)`.
///
/// This is the baseline every simultaneous route must exceed, and the value
/// every route collapses to at `K = 1`. `alpha = 0.05` gives `1.959963985...`.
pub fn pointwise_critical_value(alpha: f64) -> Result<f64, StatsError> {
    check_alpha(alpha)?;
    inv_norm_cdf(1.0 - alpha / 2.0)
}

/// Bonferroni critical value: the pointwise value at level `alpha / K`, i.e.
/// `Phi^{-1}(1 - alpha / (2K))`.
///
/// Needs neither draws nor a covariance matrix, so it is always available —
/// the honest fallback when a routine can supply neither. It is valid under
/// *any* dependence across cells (it is a union bound), and correspondingly
/// conservative: for a persistent IRF path the cells are strongly positively
/// correlated and Bonferroni pays for a worst case that does not occur.
///
/// At `K = 1` this is exactly [`pointwise_critical_value`].
pub fn bonferroni_critical_value(alpha: f64, k: usize) -> Result<f64, StatsError> {
    check_alpha(alpha)?;
    check_k(k)?;
    inv_norm_cdf(1.0 - alpha / (2.0 * k as f64))
}

/// Šidák critical value: the pointwise value at per-cell level
/// `1 - (1 - alpha)^(1/K)`.
///
/// Slightly tighter than [`bonferroni_critical_value`] at every `K > 1`, and
/// **exact under independence across cells**.
///
/// That independence condition is essentially never met by anything this
/// library bands. An IRF path, a forecast path, and a local-projection
/// response are all smooth functions of a common parameter estimate; adjacent
/// horizons are strongly positively correlated, often above 0.9. Under that
/// positive dependence Šidák is *conservative*, not exact — it is a mild
/// improvement on Bonferroni (the gap is a few tenths of a percent of the
/// critical value at typical `K`), not a correct answer. Prefer a sup-t route
/// whenever draws or a covariance matrix exist; reach for Šidák only when
/// neither does.
///
/// At `K = 1` this is exactly [`pointwise_critical_value`]: the per-cell level
/// reduces to `alpha` algebraically, and the `k == 1` case is returned
/// directly so that no `ln1p`/`expm1` round-trip can perturb the last bit.
pub fn sidak_critical_value(alpha: f64, k: usize) -> Result<f64, StatsError> {
    check_alpha(alpha)?;
    check_k(k)?;
    if k == 1 {
        return pointwise_critical_value(alpha);
    }
    // 1 - (1 - alpha)^(1/k), evaluated as -expm1(ln1p(-alpha) / k) so that
    // small alpha keeps full relative precision.
    let per_cell = -((-alpha).ln_1p() / k as f64).exp_m1();
    inv_norm_cdf(1.0 - per_cell / 2.0)
}

// ---------------------------------------------------------------------------
// sup-t from draws
// ---------------------------------------------------------------------------

/// Sup-t critical value from draws of the estimand — the workhorse route.
///
/// For each draw `b`, form the maximum over cells of the absolute
/// t-statistic centred at the point estimate,
///
/// ```text
/// M_b = max_k | theta*_{b,k} - theta_hat_k | / se_k ,
/// ```
///
/// and return the `1 - alpha` empirical quantile of `M_1, ..., M_B`. The band
/// is `theta_hat_k ± c * se_k` (see [`band`]).
///
/// # Arguments
///
/// * `draws` — `n_draws * k` values, **draw-major**: draw `b`'s value for cell
///   `j` is `draws[b * k + j]`, where `k = theta_hat.len()`.
/// * `n_draws` — number of draws; must be `>= 2` and satisfy
///   `draws.len() == n_draws * k`.
/// * `theta_hat` — the point estimate, the centring used in the numerator.
///   Centring at the point estimate (rather than at the draw mean) is what
///   makes the resulting band centred at `theta_hat`.
/// * `se` — pointwise standard errors, same cell order. Cells with `se == 0`
///   are excluded from the maximum.
/// * `alpha` — `0 < alpha < 1`; `0.1` for a 90% band.
///
/// # Quantile convention
///
/// Linear interpolation between order statistics (`numpy.quantile`'s default,
/// Hyndman–Fan type 7). With `B = 999` and `alpha = 0.1` this reads off
/// between the 899th and 900th largest of the sorted maxima.
///
/// # Floor
///
/// The result is floored at [`pointwise_critical_value`]. The population sup-t
/// quantile is never below it, so the floor only ever binds on simulation
/// noise or on inconsistent inputs — if it binds by a visible margin, the
/// draws are tighter than the `se` you passed, which usually means the two
/// came from different estimators.
///
/// # Errors
///
/// Non-finite entries in `draws`, `theta_hat`, or `se` are rejected rather
/// than skipped: one `NaN` replication would otherwise poison a maximum
/// silently. Filter failed bootstrap replications before calling.
pub fn sup_t_from_draws(
    draws: &[f64],
    n_draws: usize,
    theta_hat: &[f64],
    se: &[f64],
    alpha: f64,
) -> Result<f64, StatsError> {
    check_alpha(alpha)?;
    let k = theta_hat.len();
    check_k(k)?;
    if se.len() != k {
        return Err(StatsError::InvalidParameter {
            name: "se.len",
            value: se.len() as f64,
            requirement: "se.len() == theta_hat.len()",
        });
    }
    if n_draws < 2 {
        return Err(StatsError::InvalidParameter {
            name: "n_draws",
            value: n_draws as f64,
            requirement: "n_draws >= 2",
        });
    }
    if draws.len() != n_draws * k {
        return Err(StatsError::InvalidParameter {
            name: "draws.len",
            value: draws.len() as f64,
            requirement: "draws.len() == n_draws * theta_hat.len()",
        });
    }
    for &t in theta_hat {
        if !t.is_finite() {
            return Err(StatsError::Domain {
                name: "theta_hat",
                value: t,
                requirement: "every point estimate finite",
            });
        }
    }
    let active = active_cells(se)?;

    let mut maxima = Vec::with_capacity(n_draws);
    for b in 0..n_draws {
        let row = &draws[b * k..(b + 1) * k];
        let mut m = 0.0f64;
        for &j in &active {
            let d = row[j];
            if !d.is_finite() {
                return Err(StatsError::Domain {
                    name: "draws",
                    value: d,
                    requirement: "every draw finite (filter failed replications first)",
                });
            }
            let t = ((d - theta_hat[j]) / se[j]).abs();
            if t > m {
                m = t;
            }
        }
        maxima.push(m);
    }

    let c = quantile_type7(&mut maxima, 1.0 - alpha);
    Ok(c.max(pointwise_critical_value(alpha)?))
}

// ---------------------------------------------------------------------------
// sup-t from a covariance matrix
// ---------------------------------------------------------------------------

/// Sup-t critical value from an asymptotic covariance matrix, by Gaussian
/// simulation.
///
/// For an asymptotic band there are no draws of the estimand, only the
/// delta-method covariance `Sigma` of `theta_hat`. Simulate
/// `x ~ N(0, Sigma)`, form the same max-|t| statistic
/// `M = max_k |x_k| / sqrt(Sigma_kk)`, and return its `1 - alpha` quantile.
/// The band is `theta_hat_k ± c * sqrt(Sigma_kk)` (see [`band`] and
/// [`std_errors_from_cov`]).
///
/// # Arguments
///
/// * `sigma` — `k * k` covariance, **row-major**: `Sigma_ij = sigma[i * k + j]`.
///   Must be symmetric (checked to a relative tolerance of `1e-8`) and
///   positive semi-definite. Singular `Sigma` is fine and expected: a cell
///   pinned by a normalization contributes a zero row and column.
/// * `k` — number of cells.
/// * `alpha` — `0 < alpha < 1`.
/// * `uniforms` — `n_sim * k` independent uniforms on `[0, 1)`, simulation-major
///   (simulation `s` uses `uniforms[s * k .. (s + 1) * k]`). Size the buffer
///   with [`required_uniforms`] and fill it from a seeded
///   `tsecon_rng::Stream::fill_uniform_f64`; the whole routine is then a pure
///   function of the seed. `n_sim >= 2` is required and `n_sim >= 100_000` is
///   recommended — this is a quantile deep in the tail of a maximum, and it is
///   cheap.
///
/// # Method
///
/// Uniforms are inverted to standard normals with the AS 241 quantile function
/// ([`crate::special::inv_norm_cdf`], ~1e-16 relative) and coloured by a
/// pivot-free Cholesky factor of `Sigma` that zeroes deficient directions, so
/// rank-deficient `Sigma` is handled exactly rather than regularized. Cells
/// with `Sigma_kk == 0` are excluded from the maximum.
///
/// # Floor
///
/// Floored at [`pointwise_critical_value`], as in [`sup_t_from_draws`].
pub fn sup_t_from_cov(
    sigma: &[f64],
    k: usize,
    alpha: f64,
    uniforms: &[f64],
) -> Result<f64, StatsError> {
    check_alpha(alpha)?;
    check_k(k)?;
    if sigma.len() != k * k {
        return Err(StatsError::InvalidParameter {
            name: "sigma.len",
            value: sigma.len() as f64,
            requirement: "sigma.len() == k * k (row-major)",
        });
    }
    if uniforms.len() % k != 0 {
        return Err(StatsError::InvalidParameter {
            name: "uniforms.len",
            value: uniforms.len() as f64,
            requirement: "uniforms.len() a multiple of k",
        });
    }
    let n_sim = uniforms.len() / k;
    if n_sim < 2 {
        return Err(StatsError::InvalidParameter {
            name: "n_sim",
            value: n_sim as f64,
            requirement: "at least 2 simulations (uniforms.len() >= 2 * k)",
        });
    }

    let se = std_errors_from_cov(sigma, k)?;
    let active = active_cells(&se)?;
    let l = cholesky_psd(sigma, k)?;

    let mut z = vec![0.0f64; k];
    let mut maxima = Vec::with_capacity(n_sim);
    for s in 0..n_sim {
        let block = &uniforms[s * k..(s + 1) * k];
        for (zj, &u) in z.iter_mut().zip(block) {
            *zj = normal_from_uniform(u)?;
        }
        let mut m = 0.0f64;
        for &i in &active {
            // Row i of L times z; L is lower triangular so only m <= i matter.
            let row = &l[i * k..i * k + i + 1];
            let mut x = 0.0f64;
            for (lim, &zm) in row.iter().zip(z.iter()) {
                x += lim * zm;
            }
            let t = (x / se[i]).abs();
            if t > m {
                m = t;
            }
        }
        maxima.push(m);
    }

    let c = quantile_type7(&mut maxima, 1.0 - alpha);
    Ok(c.max(pointwise_critical_value(alpha)?))
}

/// Number of uniforms [`sup_t_from_cov`] consumes for `n_sim` simulations over
/// `k` cells: simply `k * n_sim`. Provided so callers do not have to rederive
/// the layout.
#[must_use]
pub fn required_uniforms(k: usize, n_sim: usize) -> usize {
    k * n_sim
}

/// Pointwise standard errors `sqrt(diag(Sigma))` from a row-major `k * k`
/// covariance matrix.
///
/// Errors on a non-finite or negative diagonal entry. A zero diagonal entry is
/// returned as a zero standard error — a legitimate, pinned-by-construction
/// cell.
pub fn std_errors_from_cov(sigma: &[f64], k: usize) -> Result<Vec<f64>, StatsError> {
    check_k(k)?;
    if sigma.len() != k * k {
        return Err(StatsError::InvalidParameter {
            name: "sigma.len",
            value: sigma.len() as f64,
            requirement: "sigma.len() == k * k (row-major)",
        });
    }
    let mut se = Vec::with_capacity(k);
    for j in 0..k {
        let v = sigma[j * k + j];
        if !v.is_finite() || v < 0.0 {
            return Err(StatsError::Domain {
                name: "sigma_diagonal",
                value: v,
                requirement: "every variance finite and >= 0",
            });
        }
        se.push(v.sqrt());
    }
    Ok(se)
}

// ---------------------------------------------------------------------------
// Band assembly
// ---------------------------------------------------------------------------

/// Apply a critical value to a point estimate and its pointwise standard
/// errors: `theta_hat_k ± c * se_k`.
///
/// Works for any of the four routes — the whole point of the sup-t
/// construction is that only the multiplier changes. Cells with `se == 0` come
/// back with `lower == upper == theta_hat`, which is the correct band for a
/// quantity pinned by a normalization.
///
/// Errors if `critical_value` is negative or non-finite, if the lengths
/// disagree, or if any `theta_hat`/`se` entry is non-finite.
pub fn band(
    theta_hat: &[f64],
    se: &[f64],
    critical_value: f64,
) -> Result<SimultaneousBand, StatsError> {
    let k = theta_hat.len();
    check_k(k)?;
    if se.len() != k {
        return Err(StatsError::InvalidParameter {
            name: "se.len",
            value: se.len() as f64,
            requirement: "se.len() == theta_hat.len()",
        });
    }
    if !critical_value.is_finite() || critical_value < 0.0 {
        return Err(StatsError::Domain {
            name: "critical_value",
            value: critical_value,
            requirement: "finite and >= 0",
        });
    }
    let mut lower = Vec::with_capacity(k);
    let mut upper = Vec::with_capacity(k);
    let mut n_cells_used = 0usize;
    for j in 0..k {
        let (t, s) = (theta_hat[j], se[j]);
        if !t.is_finite() {
            return Err(StatsError::Domain {
                name: "theta_hat",
                value: t,
                requirement: "every point estimate finite",
            });
        }
        if !s.is_finite() || s < 0.0 {
            return Err(StatsError::Domain {
                name: "se",
                value: s,
                requirement: "every standard error finite and >= 0",
            });
        }
        if s > 0.0 {
            n_cells_used += 1;
        }
        lower.push(t - critical_value * s);
        upper.push(t + critical_value * s);
    }
    Ok(SimultaneousBand {
        critical_value,
        lower,
        upper,
        n_cells_used,
    })
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Invert a uniform on `[0, 1)` to a standard normal deviate.
///
/// The input is shifted by half the `2^-53` grid spacing before inversion.
/// `tsecon_rng::Stream::uniform_f64` can return exactly `0.0`, which would
/// invert to `-inf` and destroy the maximum; the midpoint shift keeps every
/// grid value strictly interior while moving it by less than an ulp.
fn normal_from_uniform(u: f64) -> Result<f64, StatsError> {
    // NaN fails `contains`, so it lands in the error branch.
    if !(0.0..1.0).contains(&u) {
        return Err(StatsError::Domain {
            name: "uniforms",
            value: u,
            requirement: "0 <= u < 1",
        });
    }
    let mut p = u + HALF_ULP53;
    if p >= 1.0 {
        // Largest double strictly below 1.
        p = 1.0 - f64::EPSILON / 2.0;
    }
    inv_norm_cdf(p)
}

/// Lower-triangular Cholesky factor of a positive semi-definite row-major
/// `k * k` matrix, with deficient directions zeroed.
///
/// For a PSD matrix a zero pivot forces the whole corresponding column of the
/// Schur complement to be zero, so writing zeros there is exact rather than a
/// regularization. That is what makes a normalization-pinned cell (zero row
/// and column of `Sigma`) work without perturbing the matrix.
fn cholesky_psd(sigma: &[f64], k: usize) -> Result<Vec<f64>, StatsError> {
    let mut scale = 0.0f64;
    for j in 0..k {
        let d = sigma[j * k + j];
        if !d.is_finite() || d < 0.0 {
            return Err(StatsError::Domain {
                name: "sigma_diagonal",
                value: d,
                requirement: "every variance finite and >= 0",
            });
        }
        if d > scale {
            scale = d;
        }
    }
    // Symmetry: a row/column-major mix-up is the most common caller bug here,
    // and it produces a plausible-looking wrong answer if it goes unchecked.
    let sym_tol = SYMMETRY_TOL * scale.max(f64::MIN_POSITIVE);
    for i in 0..k {
        for j in 0..i {
            let (a, b) = (sigma[i * k + j], sigma[j * k + i]);
            if !a.is_finite() || !b.is_finite() {
                return Err(StatsError::Domain {
                    name: "sigma",
                    value: if a.is_finite() { b } else { a },
                    requirement: "every entry finite",
                });
            }
            if (a - b).abs() > sym_tol {
                return Err(StatsError::Domain {
                    name: "sigma",
                    value: a - b,
                    requirement: "symmetric to 1e-8 relative to max(diag)",
                });
            }
        }
    }

    let zero_tol = PSD_ZERO_TOL * scale;
    let neg_tol = PSD_NEG_TOL * scale;
    let mut l = vec![0.0f64; k * k];
    for j in 0..k {
        let mut d = sigma[j * k + j];
        for m in 0..j {
            d -= l[j * k + m] * l[j * k + m];
        }
        if d > zero_tol {
            let ljj = d.sqrt();
            l[j * k + j] = ljj;
            for i in (j + 1)..k {
                let mut s = sigma[i * k + j];
                for m in 0..j {
                    s -= l[i * k + m] * l[j * k + m];
                }
                l[i * k + j] = s / ljj;
            }
        } else if d >= -neg_tol {
            // Deficient direction; column already zero.
        } else {
            return Err(StatsError::Domain {
                name: "sigma",
                value: d,
                requirement: "positive semi-definite (a Cholesky pivot was negative)",
            });
        }
    }
    Ok(l)
}

/// Empirical quantile with linear interpolation between order statistics
/// (Hyndman–Fan type 7, `numpy.quantile`'s default). Sorts `x` in place.
///
/// `x` must be non-empty; `p` in `[0, 1]`. Sorting uses `total_cmp` rather
/// than an unwrapped `partial_cmp` so that no input can turn this into a
/// panic — the crate's contract is that the non-test code path never panics.
fn quantile_type7(x: &mut [f64], p: f64) -> f64 {
    x.sort_by(f64::total_cmp);
    let n = x.len();
    if n == 1 {
        return x[0];
    }
    let h = (n - 1) as f64 * p;
    let lo = h.floor();
    let frac = h - lo;
    let lo = lo as usize;
    if lo + 1 >= n {
        return x[n - 1];
    }
    x[lo] + frac * (x[lo + 1] - x[lo])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_matches_numpy_type7_on_a_small_hand_case() {
        // numpy.quantile([1, 2, 3, 4], 0.9) == 3.7
        let mut x = [1.0, 2.0, 3.0, 4.0];
        assert!((quantile_type7(&mut x, 0.9) - 3.7).abs() < 1e-12);
        let mut x = [1.0, 2.0, 3.0, 4.0];
        assert!((quantile_type7(&mut x, 0.5) - 2.5).abs() < 1e-12);
        let mut x = [1.0, 2.0, 3.0, 4.0];
        assert!((quantile_type7(&mut x, 0.0) - 1.0).abs() < 1e-12);
        let mut x = [1.0, 2.0, 3.0, 4.0];
        assert!((quantile_type7(&mut x, 1.0) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_reproduces_a_psd_matrix_with_a_zero_row() {
        // 3 x 3 with cell 1 pinned (zero row/column), cells 0 and 2 correlated.
        let k = 3;
        let sigma = vec![1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 2.0];
        let l = cholesky_psd(&sigma, k).unwrap();
        for i in 0..k {
            for j in 0..k {
                let mut v = 0.0;
                for m in 0..k {
                    v += l[i * k + m] * l[j * k + m];
                }
                assert!(
                    (v - sigma[i * k + j]).abs() < 1e-12,
                    "L L^T[{i},{j}] = {v}, want {}",
                    sigma[i * k + j]
                );
            }
        }
    }

    #[test]
    fn cholesky_rejects_an_indefinite_matrix() {
        let sigma = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky_psd(&sigma, 2).is_err());
    }

    #[test]
    fn cholesky_rejects_an_asymmetric_matrix() {
        let sigma = vec![1.0, 0.4, 0.2, 1.0];
        assert!(cholesky_psd(&sigma, 2).is_err());
    }

    #[test]
    fn uniform_zero_inverts_to_a_finite_deviate() {
        let z = normal_from_uniform(0.0).unwrap();
        assert!(z.is_finite(), "u = 0 gave {z}");
        assert!(z < -8.0 && z > -9.0, "u = 0 gave {z}");
        assert!(normal_from_uniform(1.0).is_err());
        assert!(normal_from_uniform(-0.1).is_err());
        let hi = normal_from_uniform(1.0 - f64::EPSILON / 2.0).unwrap();
        assert!(hi.is_finite() && hi > 8.0, "u -> 1 gave {hi}");
    }
}
