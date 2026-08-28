//! The Kamber-Morley-Wong (2018) Beveridge-Nelson filter.
//!
//! Kamber, Morley & Wong, "Intuitive and Reliable Estimates of the
//! Output Gap from a Beveridge-Nelson Filter", *Review of Economics and
//! Statistics* 100(3), 2018. The classic BN decomposition from a freely
//! estimated ARMA typically attributes almost everything to the trend
//! and leaves a tiny, noisy cycle; KMW instead fit an AR(`p`) to the
//! (demeaned) first differences **with the signal-to-noise ratio
//! `delta` pinned** — `delta = psi(1)^2 sigma_e^2 / sigma_{Delta y}^2`,
//! the share of forecast-error variance attributable to trend shocks —
//! which imposes a low-variance trend and produces the large,
//! persistent output gaps practitioners expect, while keeping the BN
//! trend definition (the long-horizon conditional expectation) intact.
//!
//! ## The algorithm (their baseline, transcribed from the authors' code)
//!
//! With `x_t` the (sample-mean-demeaned) first differences and
//! `rho = 1 - 1/sqrt(delta)` the implied sum of the AR coefficients:
//!
//! 1. pad `x` with `p + 2` zeros at the start (the unconditional-mean
//!    backcast — the "as in KMW2018" setting of the reference code);
//! 2. estimate the AR(`p`) in Dickey-Fuller form with the level
//!    coefficient **fixed at `rho`**: regress `x_t - rho x_{t-1}` on
//!    `[Delta x_{t-1}, ..., Delta x_{t-p+1}]` by Bayesian ridge with the
//!    shrinkage prior `psi_j ~ N(0, 0.5/j^2)` (the reference code's
//!    prior on the Dickey-Fuller coefficients) and the error variance
//!    `sigma^2` from the unrestricted zero-padded no-constant AR(`p`)
//!    OLS (`SSR/(T - p)`);
//! 3. map the Dickey-Fuller coefficients back to AR coefficients
//!    `phi_1..phi_p` (which sum to `rho` exactly);
//! 4. BN cycle from the companion form `F`:
//!    `cycle_t = -e1' F (I - F)^{-1} X_t` with state
//!    `X_t = [x_t, ..., x_{t-p+1}]` (zero-padded), the long-horizon
//!    forecastable component of future growth;
//! 5. `delta` itself is either fixed or selected by the paper's
//!    **amplitude-to-noise criterion**: grid `delta = d0, d0 + dt, ...`,
//!    stopping at the first local maximum of
//!    `var(cycle) / mean(residual^2)`;
//! 6. the fixed (non-dynamic) cycle standard error is
//!    `sqrt(e1' Phi Sigma_X Phi' e1)` with `Phi = F (I - F)^{-1}` and
//!    `Sigma_X` from the `vec`'d discrete Lyapunov equation
//!    `Sigma_X = F Sigma_X F' + Q`, `Q = e1 e1' sigma_c^2`, where
//!    `sigma_c^2` is the innovation variance from the *unpadded*
//!    AR(`p`)-with-constant OLS.
//!
//! The reference implementation is the authors' replication code
//! (bnfiltering.com; R conversion by Luke Hartigan of Ben Wong's MATLAB,
//! updated by James Morley et al.), run at its KMW-2018 baseline options
//! (`ib = FALSE`, `delta_select = 1`, `d0 = 0.01`, `dt = 0.0005`, fixed
//! error bands); the golden fixture `fixtures/bn_filters.json` pins this
//! crate to actual R runs of that code.

use crate::decomposition::{Alignment, Decomposition};
use crate::error::{check_finite, FiltersError};
use crate::lin::{cholesky_solve_spd, householder_lstsq, lu_solve};

/// Hard cap on the automatic `delta` grid search: the search errors out
/// rather than walking past `d0 + MAX_DELTA_STEPS * dt` (with the
/// reference grid, `delta > 25` — an order of magnitude beyond any
/// published application; KMW report `delta ~ 0.24` for US GDP).
const MAX_DELTA_STEPS: usize = 50_000;

/// How the signal-to-noise ratio `delta` of [`bn_filter`] is determined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BnDelta {
    /// The paper's automatic choice: grid search from `d0` in steps of
    /// `dt` for the **first local maximum of the amplitude-to-noise
    /// ratio** `var(cycle) / mean(residual^2)` (KMW 2018, section III.B;
    /// `delta_select = 1` with `d0 = 0.01`, `dt = 0.0005` in the
    /// reference code — [`BnDelta::auto`] gives exactly those values).
    Auto {
        /// Grid start (reference default `0.01`).
        d0: f64,
        /// Grid increment (reference default `0.0005`).
        dt: f64,
    },
    /// Impose a fixed `delta` (must be strictly positive).
    Fixed(f64),
}

impl BnDelta {
    /// The reference code's automatic grid: `d0 = 0.01`, `dt = 0.0005`.
    pub fn auto() -> Self {
        BnDelta::Auto {
            d0: 0.01,
            dt: 0.0005,
        }
    }
}

/// Result of the Kamber-Morley-Wong (2018) BN filter.
#[derive(Debug, Clone, PartialEq)]
pub struct BnFilterResult {
    /// Trend (`y - cycle`) and cycle, aligned to input observations
    /// `1, ..., n - 1` (`alignment.lost_start = 1`: the filter works on
    /// first differences, so the first observation carries no cycle
    /// estimate).
    pub decomposition: Decomposition,
    /// The signal-to-noise ratio actually used (selected or imposed).
    pub delta: f64,
    /// The AR(`p`) coefficients `phi_1..phi_p` of the demeaned growth
    /// process implied by the posterior; they sum to
    /// `rho = 1 - 1/sqrt(delta)` exactly.
    pub ar: Vec<f64>,
    /// The fixed (constant-across-`t`) cycle standard error of the
    /// reference code's non-dynamic error bands; the 95% band is
    /// `cycle +- 1.96 * cycle_se`.
    pub cycle_se: f64,
    /// The amplitude-to-noise ratio `var(cycle) / mean(residual^2)` at
    /// the returned `delta` (the criterion the automatic selection
    /// maximizes).
    pub amplitude_to_noise: f64,
    /// The drift removed from the first differences (the sample mean of
    /// `Delta y` when `demean = true`, `0.0` otherwise).
    pub drift: f64,
}

/// Kamber-Morley-Wong (2018) BN filter: the Beveridge-Nelson output-gap
/// estimator with the signal-to-noise ratio `delta` pinned by the
/// amplitude-to-noise criterion.
///
/// `p` is the AR lag order on the first differences (the paper's
/// baseline for quarterly US GDP is `p = 12`); `demean = true`
/// subtracts the sample mean of `Delta y` (the paper's baseline;
/// `false` = the reference code's `"nd"` no-drift option for series
/// already demeaned or without drift). See the [module
/// docs](self) for the algorithm and the reference implementation this
/// is pinned against.
///
/// Returns cycle and trend aligned to observations `1..n-1`
/// (`trend` is `y - cycle`, so the pair reconstructs `y` on that range
/// to within a final rounding), the `delta` used, the
/// implied AR coefficients, the fixed cycle standard error, and the
/// amplitude-to-noise ratio.
///
/// # Errors
///
/// * [`FiltersError::InvalidParameter`] — `p < 2` (the Dickey-Fuller
///   reparameterization needs at least one difference lag), or a
///   non-positive `delta`/`d0`/`dt`;
/// * [`FiltersError::SeriesTooShort`] — fewer than `2p + 3`
///   observations (the unpadded AR(`p`)-with-constant behind the cycle
///   standard error needs `T - p > p + 1` difference observations);
/// * [`FiltersError::NonFiniteInput`] — NaN/inf in `y`;
/// * [`FiltersError::Degenerate`] — the first differences are constant
///   (a linear ramp or flat line: the growth process the filter models
///   has zero variance), or the AR fit to the differences leaves zero
///   residual variance (an exactly deterministic `Delta y`);
/// * [`FiltersError::RankDeficient`] — a regression inside the filter is
///   singular, or the automatic grid search fails to find a local
///   maximum within the safety cap.
pub fn bn_filter(
    y: &[f64],
    p: usize,
    delta: BnDelta,
    demean: bool,
) -> Result<BnFilterResult, FiltersError> {
    if p < 2 {
        return Err(FiltersError::InvalidParameter {
            name: "p",
            value: p as f64,
            requirement: "an AR lag order >= 2: the Dickey-Fuller form that pins delta \
                          needs at least one difference lag (KMW use p = 12 for \
                          quarterly data)",
        });
    }
    let n = y.len();
    let needed = 2 * p + 3;
    if n < needed {
        return Err(FiltersError::SeriesTooShort {
            filter: "bn_filter",
            needed,
            got: n,
            why: "the AR(p)-with-constant regression behind the cycle standard error \
                  needs more than 2p + 1 first differences; lower p or supply a longer \
                  series",
        });
    }
    check_finite(y)?;

    // Demeaned first differences.
    let dy: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
    if dy.iter().all(|&v| v == dy[0]) {
        // Say what is actually constant: for a linear ramp (np.arange) or a
        // flat line it is the FIRST DIFFERENCES, and the BN filter models
        // variation in Delta y, not in the level (audit round 10: the old
        // refusal called a ramp "constant").
        return Err(FiltersError::Degenerate {
            what: "bn_filter: the first differences of the series are constant (the \
                   series is an exact linear ramp or a flat line), so the growth \
                   process the filter models has zero variance and no trend/cycle \
                   split exists. The BN decomposition works on the *changes* \
                   Delta y, which must have a stochastic component — check that \
                   the right column was passed (a time index or a deterministic \
                   trend often looks like this)",
        });
    }
    let drift = if demean {
        dy.iter().sum::<f64>() / dy.len() as f64
    } else {
        0.0
    };
    let x: Vec<f64> = dy.iter().map(|v| v - drift).collect();

    let delta_used = match delta {
        BnDelta::Fixed(d) => {
            if d <= 0.0 || !d.is_finite() {
                return Err(FiltersError::InvalidParameter {
                    name: "delta",
                    value: d,
                    requirement: "a strictly positive, finite signal-to-noise ratio",
                });
            }
            d
        }
        BnDelta::Auto { d0, dt } => {
            if d0 <= 0.0 || !d0.is_finite() {
                return Err(FiltersError::InvalidParameter {
                    name: "d0",
                    value: d0,
                    requirement: "a strictly positive, finite grid start",
                });
            }
            if dt <= 0.0 || !dt.is_finite() {
                return Err(FiltersError::InvalidParameter {
                    name: "dt",
                    value: dt,
                    requirement: "a strictly positive, finite grid increment",
                });
            }
            select_delta(&x, p, d0, dt)?
        }
    };

    let core = bn_filter_at_delta(&x, p, delta_used)?;

    // Fixed cycle standard error: innovation variance from the UNPADDED
    // AR(p)-with-constant OLS, state covariance from the vec'd discrete
    // Lyapunov equation of the companion form.
    let t_len = x.len();
    let rows = t_len - p;
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(p + 1);
    cols.push(vec![1.0; rows]);
    for k in 1..=p {
        cols.push((p..t_len).map(|t| x[t - k]).collect());
    }
    let rhs: Vec<f64> = x[p..].to_vec();
    let beta_c = householder_lstsq(
        cols.clone(),
        rhs.clone(),
        "bn_filter AR(p)-with-constant OLS",
    )?;
    let mut ssr = 0.0;
    for (r, &target) in rhs.iter().enumerate() {
        let mut fit = beta_c[0];
        for k in 1..=p {
            fit += beta_c[k] * cols[k][r];
        }
        let u = target - fit;
        ssr += u * u;
    }
    let sig2_c = ssr / ((rows - (p + 1)) as f64);

    // vec'd Lyapunov: (I - F (x) F) vec(Sigma_X) = vec(Q), Q = e1 e1' sig2_c.
    // Column-major vec as in the reference (Sigma_X is symmetric, and only
    // vec position 0 of Q is nonzero, so the orientation is also harmless).
    let f = companion(&core.ar);
    let p2 = p * p;
    let mut a = vec![0.0_f64; p2 * p2];
    for i in 0..p2 {
        a[i * p2 + i] = 1.0;
    }
    // kron(F, F)[(i1*p + i2), (j1*p + j2)] = F[i1][j1] * F[i2][j2].
    for i1 in 0..p {
        for i2 in 0..p {
            let row = i1 * p + i2;
            for j1 in 0..p {
                let f1 = f[i1 * p + j1];
                if f1 == 0.0 {
                    continue;
                }
                for j2 in 0..p {
                    a[row * p2 + (j1 * p + j2)] -= f1 * f[i2 * p + j2];
                }
            }
        }
    }
    let mut vec_q = vec![0.0_f64; p2];
    vec_q[0] = sig2_c;
    let vec_sigma = lu_solve(a, p2, vec_q, "bn_filter Lyapunov solve (I - F kron F)")?;
    // w' Sigma_X w with w = (I - F)^{-T} phi (so w' = e1' F (I - F)^{-1}).
    // Column-major vec: Sigma_X[i][j] = vec_sigma[j * p + i].
    let mut se2 = 0.0;
    for i in 0..p {
        for j in 0..p {
            se2 += core.w[i] * vec_sigma[j * p + i] * core.w[j];
        }
    }
    let cycle_se = se2.max(0.0).sqrt();

    // trend = y - cycle on the aligned range (observations 1..n-1).
    let trend: Vec<f64> = y[1..]
        .iter()
        .zip(core.cycle.iter())
        .map(|(yy, c)| yy - c)
        .collect();

    Ok(BnFilterResult {
        decomposition: Decomposition {
            trend: Some(trend),
            cycle: core.cycle,
            alignment: Alignment {
                lost_start: 1,
                lost_end: 0,
                input_len: n,
            },
        },
        delta: delta_used,
        ar: core.ar,
        cycle_se,
        amplitude_to_noise: core.amp_to_noise,
        drift,
    })
}

/// Core BN filter at a fixed `delta` on the (already demeaned)
/// difference series `x`: AR coefficients, cycle, companion weights, and
/// the amplitude-to-noise ratio. Transcribes `BN_Filter` of the
/// reference code with `ib = FALSE`.
struct BnCore {
    ar: Vec<f64>,
    cycle: Vec<f64>,
    /// `w = (I - F)^{-T} phi`, so `w' X_t = e1' F (I - F)^{-1} X_t`.
    w: Vec<f64>,
    amp_to_noise: f64,
}

fn bn_filter_at_delta(x: &[f64], p: usize, delta: f64) -> Result<BnCore, FiltersError> {
    let t_len = x.len();
    let rho = 1.0 - 1.0 / delta.sqrt();

    // Zero-padded series: p + 2 zeros, then x (the unconditional-mean
    // backcast of the reference code's ib = FALSE path).
    let pad = p + 2;
    let mut xp = vec![0.0_f64; pad + t_len];
    xp[pad..].copy_from_slice(x);
    let at = |t: usize, back: usize| xp[pad + t - back]; // x_{t-back}, zero-padded

    // sigma^2 from the unrestricted zero-padded no-constant AR(p) OLS.
    let mut cols_u: Vec<Vec<f64>> = Vec::with_capacity(p);
    for k in 1..=p {
        cols_u.push((0..t_len).map(|t| at(t, k)).collect());
    }
    let rhs_u: Vec<f64> = x.to_vec();
    let beta_u = householder_lstsq(cols_u.clone(), rhs_u.clone(), "bn_filter padded AR(p) OLS")?;
    let mut ssr_u = 0.0;
    for t in 0..t_len {
        let mut fit = 0.0;
        for k in 1..=p {
            fit += beta_u[k - 1] * cols_u[k - 1][t];
        }
        let u = x[t] - fit;
        ssr_u += u * u;
    }
    let sig2_ols = ssr_u / ((t_len - p) as f64);
    if sig2_ols <= 0.0 || !sig2_ols.is_finite() {
        return Err(FiltersError::Degenerate {
            what: "bn_filter: the AR(p) fit to the first differences has zero \
                   residual variance — Delta y follows an exact deterministic \
                   recursion, so the innovation variance behind the trend/cycle \
                   split is undefined; the BN filter needs a stochastic growth \
                   component in the differences",
        });
    }

    // Dickey-Fuller design on the padded differences, level coefficient
    // fixed at rho: regress (x_t - rho x_{t-1}) on [dx_{t-1}..dx_{t-p+1}],
    // Bayesian ridge with prior psi_j ~ N(0, 0.5/j^2).
    let m = p - 1; // difference lags
    let dx = |t: isize| -> f64 {
        // Delta x at padded position: x_t - x_{t-1}, both zero-padded.
        let i = pad as isize + t;
        if i <= 0 {
            0.0
        } else {
            xp[i as usize] - xp[(i - 1) as usize]
        }
    };
    // A_post = (V_prior^{-1} + X'X / sig2)^{-1} (X'y / sig2).
    let mut xtx = vec![0.0_f64; m * m];
    let mut xty = vec![0.0_f64; m];
    for t in 0..t_len {
        let ydf = at(t, 0) - rho * at(t, 1);
        for j in 0..m {
            let xj = dx(t as isize - (j as isize + 1));
            xty[j] += xj * ydf;
            for i in 0..=j {
                let xi = dx(t as isize - (i as isize + 1));
                xtx[i * m + j] += xi * xj;
            }
        }
    }
    for j in 0..m {
        for i in 0..j {
            xtx[j * m + i] = xtx[i * m + j];
        }
    }
    let mut a_mat = vec![0.0_f64; m * m];
    for i in 0..m {
        let jj = (i + 1) as f64;
        for j in 0..m {
            a_mat[i * m + j] = xtx[i * m + j] / sig2_ols;
        }
        a_mat[i * m + i] += jj * jj / 0.5; // V_prior^{-1} = diag(j^2 / 0.5)
    }
    let b_vec: Vec<f64> = xty.iter().map(|v| v / sig2_ols).collect();
    let psi = cholesky_solve_spd(&a_mat, m, &b_vec, "bn_filter ridge posterior")?;

    // Map Dickey-Fuller coefficients back to AR form (phi sums to rho).
    let mut phi = vec![0.0_f64; p];
    phi[p - 1] = -psi[m - 1];
    for i in (1..p - 1).rev() {
        let tail: f64 = phi[i + 1..].iter().sum();
        phi[i] = -psi[i - 1] - tail;
    }
    let tail: f64 = phi[1..].iter().sum();
    phi[0] = rho - tail;

    // Cycle: cycle_t = -e1' F (I-F)^{-1} X_t = -(w' X_t) with
    // (I - F)' w = phi.
    let f = companion(&phi);
    let mut imf_t = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..p {
            let v = if i == j { 1.0 } else { 0.0 } - f[i * p + j];
            imf_t[j * p + i] = v; // transpose
        }
    }
    let w = lu_solve(imf_t, p, phi.clone(), "bn_filter companion solve (I - F)")?;

    let mut cycle = Vec::with_capacity(t_len);
    for t in 0..t_len {
        let mut acc = 0.0;
        for (k, &wk) in w.iter().enumerate() {
            acc += wk * at(t, k);
        }
        cycle.push(-acc);
    }

    // Amplitude-to-noise: var(cycle, ddof = 1) / mean(residual^2), with
    // residuals from the padded design at the POSTERIOR coefficients.
    let mut msr = 0.0;
    for (t, &xt) in x.iter().enumerate() {
        let mut fit = 0.0;
        for k in 1..=p {
            fit += phi[k - 1] * at(t, k);
        }
        let u = xt - fit;
        msr += u * u;
    }
    msr /= t_len as f64;
    let cmean = cycle.iter().sum::<f64>() / t_len as f64;
    let cvar = cycle.iter().map(|c| (c - cmean) * (c - cmean)).sum::<f64>() / ((t_len - 1) as f64);
    let amp_to_noise = cvar / msr;

    Ok(BnCore {
        ar: phi,
        cycle,
        w,
        amp_to_noise,
    })
}

/// Companion matrix of `phi` (row-major `p x p`): first row `phi`,
/// subdiagonal identity.
fn companion(phi: &[f64]) -> Vec<f64> {
    let p = phi.len();
    let mut f = vec![0.0_f64; p * p];
    f[..p].copy_from_slice(phi);
    for i in 1..p {
        f[i * p + (i - 1)] = 1.0;
    }
    f
}

/// The paper's `delta` selection (`select_delta` with `delta_select = 1`):
/// walk the grid `d0, d0 + dt, ...` while the amplitude-to-noise ratio
/// increases; return the argument of its first local maximum.
fn select_delta(x: &[f64], p: usize, d0: f64, dt: f64) -> Result<f64, FiltersError> {
    let mut delta = d0;
    let mut best = bn_filter_at_delta(x, p, delta)?.amp_to_noise;
    for _ in 0..MAX_DELTA_STEPS {
        let cand = delta + dt;
        let next = bn_filter_at_delta(x, p, cand)?.amp_to_noise;
        if next > best {
            delta = cand;
            best = next;
        } else {
            return Ok(delta);
        }
    }
    Err(FiltersError::RankDeficient {
        what: "bn_filter delta selection: no local maximum of the amplitude-to-noise \
               ratio within the safety cap (delta walked past d0 + 50000*dt) — the \
               series is unlike anything the KMW criterion is meant for; impose a \
               fixed delta instead",
    })
}
