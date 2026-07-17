//! The Haar-uniform rotation kernel: orthogonal draws uniform on the
//! orthogonal group `O(m)`, the primitive every rotation-based
//! identification scheme in the library is built on.
//!
//! # Why QR-of-Gaussian with a sign fix
//!
//! Filling an `m x m` matrix `A` with independent standard-normal entries
//! makes its distribution invariant under left- and right-multiplication by
//! any fixed orthogonal matrix (the Gaussian measure is rotationally
//! symmetric). Its QR factorization `A = Q R` therefore has an `Q` that is
//! *almost* Haar-uniform — but not quite: the factorization is unique only
//! once the sign of each diagonal entry of `R` is fixed, and an
//! off-the-shelf QR routine leaves those signs as an arbitrary,
//! non-uniform function of `A`. Taking the routine's `Q` directly silently
//! biases every downstream sign-restriction posterior (roadmap module 06,
//! implementation warning 1).
//!
//! The fix (Stewart 1980; Mezzadri 2007, "How to generate random matrices
//! from the classical compact groups"): let `Lambda = diag(sign(R_ii))`.
//! Then `A = (Q Lambda)(Lambda R)`, and `Lambda R` has a strictly positive
//! diagonal, so `Q Lambda` is *the* unique orthogonal factor of the
//! positive-diagonal QR — and that factor is exactly Haar-uniform. This is
//! an algebraic identity, independent of the QR routine's internal sign
//! conventions, so it applies verbatim to the Householder factorization
//! used here. (Uniform Givens angles are *not* Haar for `m >= 3`; only the
//! QR-of-Gaussian construction is.)
//!
//! # Determinism
//!
//! The `m^2` Gaussian entries are drawn column by column (all rows of
//! column 0, then column 1, ...), each an inverse-CDF transform of a
//! [`Stream`] uniform, so a given stream state produces a bit-identical
//! `Q`.

use tsecon_linalg::faer::Mat;
use tsecon_rng::Stream;
use tsecon_stats::special::inv_norm_cdf;

use crate::error::IdentError;

/// Retry budget for rejecting the (probability `2^-53`) exact-zero uniform
/// that the inverse normal CDF cannot map to a finite quantile.
const UNIFORM_RETRIES: usize = 128;

/// One standard-normal draw by inverse-CDF transform of a stream uniform
/// (Wichura AS241 `inv_norm_cdf`, ~1e-16 relative accuracy), rejecting the
/// exact 0 that [`Stream::uniform_f64`] can (with probability `2^-53`)
/// return.
///
/// TODO(phase0): share a ziggurat sampler with `tsecon-bayes` once the
/// shared RNG layer grows one; the inverse CDF is exact but slower.
fn std_normal(stream: &mut Stream) -> Result<f64, IdentError> {
    for _ in 0..UNIFORM_RETRIES {
        let u = stream.uniform_f64();
        if u > 0.0 {
            return Ok(inv_norm_cdf(u)?);
        }
    }
    Err(IdentError::NoConvergence {
        what: "positive uniform draw for a Gaussian Haar entry (stream returned 0 repeatedly)",
    })
}

/// A Haar-uniform orthogonal matrix `Q` of order `m`, drawn from the stream.
///
/// `Q` is uniform on the orthogonal group `O(m)` (both connected
/// components — `det(Q)` is `+1` or `-1` with equal probability, which is
/// the correct measure for SVAR rotation sampling, where a column sign flip
/// is a relabeling, not a distinct model). Columns are orthonormal to
/// machine precision.
///
/// Method: fill an `m x m` matrix with independent standard normals, take
/// its Householder QR factorization, and apply the Stewart/Mezzadri
/// `Q <- Q diag(sign(R_ii))` sign fix (see the module docs). The Gaussian
/// entries are drawn column by column for reproducibility.
///
/// # Errors
///
/// * [`IdentError::InvalidArgument`] if `m == 0`;
/// * [`IdentError::Stats`] if the inverse normal CDF fails on a stream
///   uniform;
/// * [`IdentError::NoConvergence`] if the uniform stream degenerates (not
///   observed in practice).
pub fn haar_rotation(m: usize, stream: &mut Stream) -> Result<Mat<f64>, IdentError> {
    if m == 0 {
        return Err(IdentError::InvalidArgument {
            what: "Haar rotation order m must be at least 1",
        });
    }

    // Gaussian source matrix, drawn column by column.
    let mut r = Mat::<f64>::zeros(m, m);
    for j in 0..m {
        for i in 0..m {
            r[(i, j)] = std_normal(stream)?;
        }
    }

    // Householder QR: overwrite `r` with R while accumulating Q = H_0 ...
    // H_{m-1}. After applying every reflector, A = H_0 ... H_{m-1} R, so
    // seeding Q = I and post-multiplying by each reflector in order builds
    // the orthogonal factor.
    let mut q = Mat::<f64>::identity(m, m);
    let mut v = vec![0.0f64; m];
    for k in 0..m {
        // Reflector on the sub-column r[k.., k].
        let mut norm_sq = 0.0;
        for i in k..m {
            norm_sq += r[(i, k)] * r[(i, k)];
        }
        let norm = norm_sq.sqrt();
        if norm == 0.0 {
            continue; // Sub-column already zero; nothing to reflect.
        }
        // alpha = -sign(r_kk) * norm chooses the reflection that avoids
        // cancellation; its sign is arbitrary and is corrected below.
        let alpha = if r[(k, k)] > 0.0 { -norm } else { norm };
        for slot in v.iter_mut().take(k) {
            *slot = 0.0;
        }
        v[k] = r[(k, k)] - alpha;
        for i in (k + 1)..m {
            v[i] = r[(i, k)];
        }
        let mut vnorm_sq = 0.0;
        for &vi in &v[k..m] {
            vnorm_sq += vi * vi;
        }
        if vnorm_sq == 0.0 {
            continue;
        }

        // R <- H R (columns k..m; earlier columns are already upper).
        for c in k..m {
            let mut dot = 0.0;
            for i in k..m {
                dot += v[i] * r[(i, c)];
            }
            let factor = 2.0 * dot / vnorm_sq;
            for i in k..m {
                r[(i, c)] -= factor * v[i];
            }
        }

        // Q <- Q H, i.e. Q <- Q - (2 / v'v) (Q v) v'.
        for row in 0..m {
            let mut qv = 0.0;
            for i in k..m {
                qv += q[(row, i)] * v[i];
            }
            let factor = 2.0 * qv / vnorm_sq;
            for i in k..m {
                q[(row, i)] -= factor * v[i];
            }
        }
    }

    // Stewart/Mezzadri sign fix: multiply column j of Q by sign(R_jj) so
    // the implied R has a positive diagonal and Q becomes Haar-uniform.
    // sign(0) is treated as +1 (a zero pivot has measure zero for Gaussian
    // input).
    for j in 0..m {
        if r[(j, j)] < 0.0 {
            for i in 0..m {
                q[(i, j)] = -q[(i, j)];
            }
        }
    }

    Ok(q)
}
