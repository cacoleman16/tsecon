//! Classic Beveridge-Nelson (1981) trend-cycle decomposition from an
//! ARIMA(p, 1, q) model.
//!
//! For `Delta y_t = mu + phi(L)^{-1} theta(L) eps_t` (stationary,
//! invertible ARMA growth), Beveridge & Nelson define the trend as the
//! long-horizon conditional expectation net of deterministic growth,
//!
//! ```text
//! tau_t = lim_{J->inf} ( E_t[y_{t+J}] - J*mu ),
//! ```
//!
//! which works out to a **random walk with drift** driven by the same
//! innovations as the series:
//!
//! ```text
//! tau_t = tau_{t-1} + mu + psi(1) eps_t,
//! psi(1) = theta(1)/phi(1) = (1 + theta_1 + ... + theta_q)
//!          / (1 - phi_1 - ... - phi_p),
//! ```
//!
//! the **long-run multiplier** (the cumulative impulse response of `y`
//! to `eps` — the permanent effect of a one-unit shock). The cycle is
//! everything else: `c_t = y_t - tau_t`, the (negative of the)
//! forecastable transitory component of future growth. This module
//! computes it in the companion form of Morley (2002, *Journal of
//! Applied Econometrics* 17):
//!
//! ```text
//! c_t = -e1' F (I - F)^{-1} X_t,
//! ```
//!
//! with `F` the companion matrix of the ARMA(p, q) on demeaned growth
//! `x_t = Delta y_t - mu` and state
//! `X_t = [x_t, ..., x_{t-p+1}, eps_t, ..., eps_{t-q+1}]'`; the
//! innovations `eps_t` come from the **conditional (zero-presample)
//! ARMA recursion**
//!
//! ```text
//! eps_t = x_t - sum_i phi_i x_{t-i} - sum_j theta_j eps_{t-j},
//! ```
//!
//! with `x_s = eps_s = 0` for `s < 0` — the CSS innovation convention,
//! which makes the identities hold in finite samples, not just
//! asymptotically: `tau_t + c_t = y_t` for every `t` (up to a final
//! float rounding — `tau` is stored as `y - c`), and
//! `Delta tau_t = mu + psi(1) eps_t` for `t >= 1` (the trend really is
//! a random walk with drift, observation by observation).
//!
//! Contrast with the Kamber-Morley-Wong `bn_filter` in
//! `tsecon-filters`: the classic decomposition here lets the data pick the
//! signal-to-noise ratio through the freely estimated ARMA — on US GDP
//! that attributes most variation to the trend and yields a small,
//! noisy cycle (the well-known result of Stock & Watson 1988 and Morley,
//! Nelson & Zivot 2003) — while the Kamber-Morley-Wong filter *pins*
//! the signal-to-noise ratio to recover a large, persistent output gap.
//! Both are exact BN decompositions; they differ in how the growth
//! model is disciplined.

use crate::error::ArimaError;
use crate::results::ArimaResults;
use crate::spec::ArimaSpec;

/// A Beveridge-Nelson trend-cycle decomposition; produced by
/// [`bn_decomposition`] (fit path) or [`bn_from_arma`] (fixed
/// coefficients).
///
/// `trend`, `cycle` and `innovations` are aligned to input observations
/// `1, ..., n - 1` (`lost_start = 1`: one observation is lost to the
/// differencing); `trend` is stored as `y - cycle`, so
/// `trend[t] + cycle[t]` recovers `y[t + 1]` to within a final rounding.
#[derive(Debug, Clone)]
pub struct BnDecomposition {
    /// BN trend `tau_t` — a random walk with drift `mu` and increments
    /// `mu + psi(1) eps_t`.
    pub trend: Vec<f64>,
    /// BN cycle `c_t = y_t - tau_t` (the transitory component).
    pub cycle: Vec<f64>,
    /// Conditional (zero-presample) ARMA innovations `eps_t` of the
    /// demeaned growth process — the shocks whose scaled cumulation is
    /// the stochastic part of the trend.
    pub innovations: Vec<f64>,
    /// The long-run multiplier `psi(1) = theta(1)/phi(1)` — the
    /// permanent effect of a unit innovation, and the number that
    /// separates trend from cycle.
    pub long_run_multiplier: f64,
    /// Deterministic drift `mu` of the trend (per period).
    pub drift: f64,
    /// AR coefficients `phi_1..phi_p` of the growth ARMA used.
    pub ar: Vec<f64>,
    /// MA coefficients `theta_1..theta_q` of the growth ARMA used.
    pub ma: Vec<f64>,
    /// Observations lost at the start (always `1`, from differencing).
    pub lost_start: usize,
    /// Length of the input series.
    pub input_len: usize,
    /// The fitted ARIMA(p, 1, q) results when [`bn_decomposition`]
    /// estimated the model; `None` for [`bn_from_arma`].
    pub results: Option<ArimaResults>,
}

/// Classic BN decomposition with the ARMA growth model estimated by this
/// crate's own exact-MLE ARIMA fit.
///
/// Fits `ARIMA(p, 1, q)` with a constant to `y` via [`ArimaSpec::fit`]
/// (exact Gaussian MLE, statsmodels `SARIMAX(order=(p, 1, q),
/// trend='c', simple_differencing=True)` conventions), recovers the
/// implied drift `mu = c / phi(1)` from the fitted intercept, and
/// computes the decomposition at the fitted coefficients via
/// [`bn_from_arma`]. The Morley-Nelson-Zivot (2003) specification for
/// US real GDP is `p = 2, q = 2`.
///
/// # Errors
///
/// Everything [`ArimaSpec::fit`] can raise, plus the
/// [`bn_from_arma`] guards — in particular
/// [`ArimaError::InvalidArgument`] when the fitted AR polynomial is
/// numerically on the unit circle (`phi(1) ~ 0`), where the long-run
/// multiplier — and with it the whole decomposition — is undefined.
pub fn bn_decomposition(y: &[f64], p: usize, q: usize) -> Result<BnDecomposition, ArimaError> {
    let spec = ArimaSpec::new(p, 1, q)?.with_constant(true);
    let results = spec.fit(y)?;
    let ar = results.ar().to_vec();
    let ma = results.ma().to_vec();
    let phi1 = 1.0 - ar.iter().sum::<f64>();
    if phi1.abs() < 1e-8 {
        return Err(ArimaError::InvalidArgument {
            what: "the fitted AR polynomial has phi(1) ~ 0 (a root numerically on the \
                   unit circle), so the long-run multiplier psi(1) = theta(1)/phi(1) \
                   and the BN trend are undefined; the growth series is likely \
                   over-differenced or needs a different (p, q)",
        });
    }
    // Fit-path-specific wording for the unit-circle guards: the user
    // supplied orders, not coefficients, so point at the orders.
    let threshold = crate::auto::ROOT_ADMISSIBILITY_THRESHOLD;
    let neg_ar: Vec<f64> = ar.iter().map(|v| -v).collect();
    if crate::auto::min_root_modulus(&neg_ar).is_some_and(|m| m < threshold) {
        return Err(ArimaError::InvalidArgument {
            what: "the fitted AR polynomial sits numerically on the unit circle, so the \
                   long-horizon expectation defining the BN trend does not converge; \
                   the growth series may need further differencing, or a smaller p",
        });
    }
    if crate::auto::min_root_modulus(&ma).is_some_and(|m| m < threshold) {
        return Err(ArimaError::InvalidArgument {
            what: "the fitted MA polynomial sits numerically on the unit circle (the \
                   classic MA-boundary pile-up, common when q is too large and AR/MA \
                   roots nearly cancel), so the BN innovation recursion is not \
                   reliable; lower q",
        });
    }
    let constant = results.constant().unwrap_or(0.0);
    let drift = constant / phi1;
    let mut out = bn_from_arma(y, drift, &ar, &ma)?;
    out.results = Some(results);
    Ok(out)
}

/// Classic BN decomposition at **fixed** ARMA(p, q) growth coefficients
/// (the documented Morley-2002 companion-form computation; see the
/// [module docs](self) for the formulas).
///
/// `ar`/`ma` are `phi_1..phi_p` / `theta_1..theta_q` in the statsmodels
/// sign convention (`x_t = mu + phi_1 x_{t-1} + ... + eps_t +
/// theta_1 eps_{t-1} + ...` with `x_t = Delta y_t`); `drift` is the
/// growth mean `mu` that is subtracted before the recursion (pass the
/// sample mean of `Delta y` for a quick mean-adjusted decomposition).
/// Use this to decompose at published coefficients, or via
/// [`bn_decomposition`] to estimate them with the library's ARIMA fit.
///
/// # Errors
///
/// * [`ArimaError::NonFinite`] — NaN/inf in `y`, `drift`, or the
///   coefficients;
/// * [`ArimaError::InsufficientObservations`] — fewer than 2
///   observations (no difference to decompose);
/// * [`ArimaError::InvalidArgument`] — an AR polynomial that is not
///   stationary or an MA polynomial that is not invertible (root
///   modulus below the crate's 1.001 unit-circle threshold): the
///   long-horizon expectation defining the trend does not converge for
///   the former, and the innovation recursion diverges for the latter.
pub fn bn_from_arma(
    y: &[f64],
    drift: f64,
    ar: &[f64],
    ma: &[f64],
) -> Result<BnDecomposition, ArimaError> {
    if let Some(at) = y.iter().position(|v| !v.is_finite()) {
        return Err(ArimaError::NonFinite {
            what: "the input series",
            at: Some(at),
        });
    }
    if !drift.is_finite() {
        return Err(ArimaError::NonFinite {
            what: "the drift",
            at: None,
        });
    }
    for (name, coefs) in [("the AR coefficients", ar), ("the MA coefficients", ma)] {
        if coefs.iter().any(|v| !v.is_finite()) {
            return Err(ArimaError::NonFinite {
                what: name,
                at: None,
            });
        }
    }
    if y.len() < 2 {
        return Err(ArimaError::InsufficientObservations {
            needed: 2,
            got: y.len(),
            nobs: y.len(),
            what: "one first difference — the BN decomposition works on growth rates",
        });
    }
    let threshold = crate::auto::ROOT_ADMISSIBILITY_THRESHOLD;
    let neg_ar: Vec<f64> = ar.iter().map(|v| -v).collect();
    if crate::auto::min_root_modulus(&neg_ar).is_some_and(|m| m < threshold) {
        return Err(ArimaError::InvalidArgument {
            what: "the AR polynomial has a root on or inside the unit circle (not \
                   stationary), so the long-horizon expectation that defines the BN \
                   trend does not converge; difference further or change the AR \
                   coefficients",
        });
    }
    if crate::auto::min_root_modulus(ma).is_some_and(|m| m < threshold) {
        return Err(ArimaError::InvalidArgument {
            what: "the MA polynomial has a root on or inside the unit circle (not \
                   invertible), so the innovation recursion behind the BN state \
                   diverges; use the invertible representation of the same \
                   autocovariances",
        });
    }

    let p = ar.len();
    let q = ma.len();
    let pp = p.max(1); // at least one growth slot in the state
    let mut phi = vec![0.0_f64; pp];
    phi[..p].copy_from_slice(ar);

    // Demeaned growth and the conditional innovation recursion.
    let x: Vec<f64> = y.windows(2).map(|w| w[1] - w[0] - drift).collect();
    let t_len = x.len();
    let mut eps = vec![0.0_f64; t_len];
    for t in 0..t_len {
        let mut acc = x[t];
        for (i, &ph) in phi.iter().enumerate() {
            if t > i {
                acc -= ph * x[t - i - 1];
            }
        }
        for (j, &th) in ma.iter().enumerate() {
            if t > j {
                acc -= th * eps[t - j - 1];
            }
        }
        eps[t] = acc;
    }

    // Companion matrix F (r x r, r = pp + q):
    //   first row [phi_1..phi_pp, theta_1..theta_q];
    //   identity shifts within the x-block and the eps-block.
    let r = pp + q;
    let mut f = vec![0.0_f64; r * r];
    f[..pp].copy_from_slice(&phi);
    f[pp..r].copy_from_slice(ma);
    for i in 1..pp {
        f[i * r + (i - 1)] = 1.0;
    }
    for j in 1..q {
        f[(pp + j) * r + (pp + j - 1)] = 1.0;
    }

    // w = (I - F)^{-T} f_row1, so w' X = e1' F (I - F)^{-1} X.
    let mut imf_t = vec![0.0_f64; r * r];
    for i in 0..r {
        for j in 0..r {
            let v = if i == j { 1.0 } else { 0.0 } - f[i * r + j];
            imf_t[j * r + i] = v;
        }
    }
    let f_row: Vec<f64> = f[..r].to_vec();
    let w = solve_lu(imf_t, r, f_row).ok_or(ArimaError::InvalidArgument {
        what: "(I - F) is numerically singular — the ARMA has a unit root the \
               admissibility guard did not catch; change the (p, q) orders",
    })?;

    // cycle_t = -(w' X_t), X_t = [x_t..x_{t-pp+1}, eps_t..eps_{t-q+1}].
    let mut cycle = Vec::with_capacity(t_len);
    for t in 0..t_len {
        let mut acc = 0.0;
        for i in 0..pp {
            if t >= i {
                acc += w[i] * x[t - i];
            }
        }
        for j in 0..q {
            if t >= j {
                acc += w[pp + j] * eps[t - j];
            }
        }
        cycle.push(-acc);
    }
    let trend: Vec<f64> = y[1..]
        .iter()
        .zip(cycle.iter())
        .map(|(yy, c)| yy - c)
        .collect();

    let theta1 = 1.0 + ma.iter().sum::<f64>();
    let phi1 = 1.0 - ar.iter().sum::<f64>();
    let long_run_multiplier = theta1 / phi1;

    Ok(BnDecomposition {
        trend,
        cycle,
        innovations: eps,
        long_run_multiplier,
        drift,
        ar: ar.to_vec(),
        ma: ma.to_vec(),
        lost_start: 1,
        input_len: y.len(),
        results: None,
    })
}

/// Partial-pivoting LU solve of `A x = b` (row-major `n x n`); `None` on
/// a numerically zero pivot.
fn solve_lu(mut a: Vec<f64>, n: usize, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let max_abs = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let tol = n as f64 * f64::EPSILON * max_abs;
    for col in 0..n {
        let mut piv = col;
        let mut piv_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > piv_val {
                piv = row;
                piv_val = v;
            }
        }
        if piv_val <= tol {
            return None;
        }
        if piv != col {
            for j in 0..n {
                a.swap(col * n + j, piv * n + j);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / d;
            if factor == 0.0 {
                continue;
            }
            a[row * n + col] = 0.0;
            for j in (col + 1)..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in (i + 1)..n {
            acc -= a[i * n + j] * x[j];
        }
        x[i] = acc / a[i * n + i];
    }
    Some(x)
}
