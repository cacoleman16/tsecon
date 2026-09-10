//! Python bindings for the JSZ canonical affine term-structure model of
//! `tsecon-termstructure` (Joslin-Singleton-Zhu 2011): `jsz_fit` (the
//! maximum-likelihood fit with the P-measure VAR concentrated out) and
//! `jsz_loadings` (the canonical Riccati recursions at given parameters).
//! Registered into `_core` through [`register`].

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{mat_to_vec2, to_faer, to_py, vec1};

/// JSZ canonical Gaussian affine term-structure model (Joslin, Singleton &
/// Zhu 2011) estimated by maximum likelihood.
///
/// Risk-neutral dynamics in the JSZ canonical form — the ordered
/// eigenvalues `lambda_q` (per period), one drift `k_inf_q`, short rate
/// `r_t = iota' X_t` — rotated onto `n_factors` observed yield portfolios
/// `P_t = w y_t` (default `w`: the first `n_factors` principal-component
/// loadings of the panel) that are priced WITHOUT error, the remaining
/// `M - n_factors` yield directions with iid error `sigma_e`. The P-measure
/// VAR(1) of the portfolios is concentrated out by OLS (exactly statsmodels
/// `VAR(1)`), `k_inf_q` and `sigma_e` are profiled analytically, and the
/// numerical search runs over `lambda_q` and the Cholesky factor of `sigma`
/// only: JSZ's recommended start (the eigenvalues of the OLS feedback
/// matrix) plus `n_starts - 1` seeded perturbations of the eigenvalue
/// pattern, each run by BFGS to a loose tolerance, the best basin polished
/// by BFGS then Nelder-Mead. The fit depends on `w` only through its row
/// space: any basis of the same portfolio space gives the same `llf`,
/// `fitted`, `lambda_q`, `k_inf_q` and `sigma_e`.
///
/// UNITS ARE LOAD-BEARING: `yields` is a `T x M` panel of ANNUALIZED,
/// continuously-compounded zero-coupon log yields in DECIMAL (0.05, not
/// 5.0); `maturities` are strictly ascending integer PERIODS (months for
/// monthly data; the one-period maturity need not be present);
/// `periods_per_year` converts to the per-period quantities the recursions
/// price. `lambda_q` and `k_inf_q` are per period (JSZ's convention);
/// everything in the portfolio rotation (`mu_p`, `phi_p`, `sigma`,
/// `k0_q_p`, `k1_q_p`, `a_p`, `b_p`) is in the annualized units of
/// `P_t = w y_t`; `sigma_e` and `rmse` in annualized-yield units.
///
/// Further arguments, with defaults: `n_factors` (3), `periods_per_year`
/// (12.0), `w` (None: PCA loadings; otherwise an `n_factors x M` array of
/// portfolio weights with linearly independent rows), `n_starts` (5, at
/// most 1000), `seed` (0; seeds the perturbed starts through `tsecon_rng`,
/// so the fit is a deterministic function of the inputs, `n_starts` and
/// `seed`). `seed` acts only when `n_starts > 1`, so passing it explicitly
/// with `n_starts=1` RAISES rather than being silently ignored.
///
/// Returns the Q parameters `lambda_q`, `k_inf_q`; `sigma` (the maximum-
/// likelihood innovation covariance of the portfolio VAR) and `sigma_e`;
/// the OLS P-measure VAR `mu_p`, `phi_p` with statsmodels-convention
/// standard errors `mu_p_se`, `phi_p_se` and the OLS residual covariance
/// `sigma_ols` (statsmodels `sigma_u_mle`; differs from `sigma` only through
/// the convexity term's weak dependence on the covariance); the risk-
/// neutral VAR in the portfolio rotation `k0_q_p`, `k1_q_p`; the market
/// prices of risk in ACM's units `lambda0 = mu_p - k0_q_p`, `lambda1 =
/// phi_p - k1_q_p`; yield coefficients in the portfolio rotation `a_p`,
/// `b_p` (`fitted = a_p + b_p P`, with `w b_p = I` and `w a_p = 0`) and for
/// the literal canonical latent state `a_x`, `b_x`; `fitted` yields
/// (`T x M`), `risk_neutral` yields — the same recursion re-run with the
/// P-measure VAR, exactly the `acm_term_premium` convention — and
/// `term_premium = fitted - risk_neutral`; `rmse` per maturity; the
/// portfolios `factors` (`T x n_factors`) and the weights `w` used; `llf`
/// (the log-likelihood of the yield panel, invariant to the basis of the
/// portfolio space; for an orthonormal `w` it is the JSZ replication code's
/// `llkP + llkQ`), `converged`, `n_iter`; and the echoed `maturities`,
/// `n_factors`, `periods_per_year`, `n_starts`, `seed`.
///
/// Two traps, stated plainly. (1) The likelihood is FLAT in the market
/// prices of risk: `lambda_q` and `k_inf_q` are pinned by the cross-section
/// to many digits, `mu_p`/`phi_p` by a T-observation VAR of persistent
/// factors — read `lambda0`/`lambda1` and the term premium's level with
/// `mu_p_se`/`phi_p_se` in mind. No standard errors are reported for the Q
/// parameters: a numerical Hessian of a profile likelihood near a unit root
/// is not an honest asymptotic covariance. (2) The level factor is nearly a
/// unit root under Q: `lambda_q[0]` is left unbounded, the recursions are
/// evaluated by recurrence so nothing degrades at or across 1, and an
/// estimate above 1 (explosive risk-neutral dynamics) is reported, not
/// hidden. On the real 1990-2007 GSW panel the surface is multimodal (three
/// basins; the JSZ start alone ends ~190 log-likelihood points below the
/// best), which is what `n_starts` is for.
///
/// Validation (fixtures/jsz.json): the Riccati recursions against a
/// documented-formula NumPy transcription at 1e-12; the portfolio VAR(1)
/// against statsmodels at 1e-9; the likelihood at stated parameters
/// against the documented formula at 1e-7 absolute (~1e-11 relative); the
/// MLE against a SciPy multi-start optimum (cross-optimizer target,
/// `lambda_q` within 1e-5) on a simulated canonical model — recovering
/// `lambda_q` to 9e-5, `k_inf_q` to 1.3%, `sigma_e` to 0.3% at T = 500 —
/// and on the real 1990-2007 GSW panel; the AFNS special case `lambda_q =
/// (1, e^-lam, e^-lam)` spans the Nelson-Siegel loadings exactly and its
/// convexity intercept converges at first order in the period length to the
/// CDR (2011) closed form of `afns_adjustment` (gap 9.8e-6 at monthly,
/// 6.1e-7 at 1/192 year). Properties: invariance to the basis of the
/// portfolio space (llf 1e-11, `lambda_q` 2e-10, fitted yields 1e-8),
/// exact pricing of the portfolios, determinism, and local optimality of
/// the maximized likelihood in every parameter direction.
///
/// Returned keys: `a_p`, `a_x`, `b_p`, `b_x`, `converged`, `factors`,
/// `fitted`, `k0_q_p`, `k1_q_p`, `k_inf_q`, `lambda0`, `lambda1`,
/// `lambda_q`, `llf`, `maturities`, `mu_p`, `mu_p_se`, `n_factors`,
/// `n_iter`, `n_starts`, `periods_per_year`, `phi_p`, `phi_p_se`,
/// `risk_neutral`, `rmse`, `seed`, `sigma`, `sigma_e`, `sigma_ols`,
/// `term_premium`, `w`.
#[pyfunction]
#[pyo3(signature = (yields, maturities, n_factors = 3, periods_per_year = 12.0, w = None, n_starts = 5, seed = None))]
#[allow(clippy::too_many_arguments)]
fn jsz_fit<'py>(
    py: Python<'py>,
    yields: PyReadonlyArray2<'py, f64>,
    maturities: Vec<usize>,
    n_factors: usize,
    periods_per_year: f64,
    w: Option<PyReadonlyArray2<'py, f64>>,
    n_starts: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    if n_starts == 1 && seed.is_some() {
        return Err(PyValueError::new_err(
            "seed was given but n_starts=1 ignores it: with a single start the search \
             runs only from JSZ's recommended starting values (the OLS eigenvalues), \
             so the seed of the perturbed starts cannot act; pass n_starts >= 2 or \
             drop seed",
        ));
    }
    let seed_used = seed.unwrap_or(0);
    let rows = mat_to_vec2(&to_faer(&yields));
    let w_rows: Option<Vec<Vec<f64>>> = w.as_ref().map(|m| mat_to_vec2(&to_faer(m)));
    let fit = tsecon_termstructure::fit_jsz(
        &rows,
        &maturities,
        n_factors,
        periods_per_year,
        w_rows.as_deref(),
        n_starts,
        seed_used,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("maturities", fit.maturities)?;
    d.set_item("n_factors", fit.n_factors)?;
    d.set_item("periods_per_year", fit.periods_per_year)?;
    d.set_item("n_starts", n_starts)?;
    d.set_item("seed", seed_used)?;
    d.set_item("w", fit.w)?;
    d.set_item("factors", fit.factors)?;
    d.set_item("lambda_q", fit.lambda_q.into_pyarray(py))?;
    d.set_item("k_inf_q", fit.k_inf_q)?;
    d.set_item("sigma", fit.sigma)?;
    d.set_item("sigma_e", fit.sigma_e)?;
    d.set_item("mu_p", fit.mu_p.into_pyarray(py))?;
    d.set_item("phi_p", fit.phi_p)?;
    d.set_item("mu_p_se", fit.mu_p_se.into_pyarray(py))?;
    d.set_item("phi_p_se", fit.phi_p_se)?;
    d.set_item("sigma_ols", fit.sigma_ols)?;
    d.set_item("k0_q_p", fit.k0_q_p.into_pyarray(py))?;
    d.set_item("k1_q_p", fit.k1_q_p)?;
    d.set_item("lambda0", fit.lambda0.into_pyarray(py))?;
    d.set_item("lambda1", fit.lambda1)?;
    d.set_item("a_p", fit.a_p.into_pyarray(py))?;
    d.set_item("b_p", fit.b_p)?;
    d.set_item("a_x", fit.a_x.into_pyarray(py))?;
    d.set_item("b_x", fit.b_x)?;
    d.set_item("fitted", fit.fitted)?;
    d.set_item("risk_neutral", fit.risk_neutral)?;
    d.set_item("term_premium", fit.term_premium)?;
    d.set_item("rmse", fit.rmse.into_pyarray(py))?;
    d.set_item("llf", fit.llf)?;
    d.set_item("converged", fit.converged)?;
    d.set_item("n_iter", fit.n_iter)?;
    Ok(d)
}

/// The JSZ canonical bond-loading recursions at given parameters.
///
/// Evaluates `A_{n+1} = A_n + K0' B_n + 1/2 B_n' sigma_x B_n`, `B_{n+1} =
/// K1' B_n - iota` from `A_0 = B_0 = 0` for the literal canonical form
/// `K0 = (k_inf_q, 0, ..., 0)'`, `K1 = J(lambda_q)` — diagonal, with a `1`
/// on the superdiagonal wherever two consecutive eigenvalues are exactly
/// equal (a Jordan block; the AFNS pattern `(1, rho, rho)` is one) — and
/// returns the per-maturity yield coefficients `a_x = -A_n / n *
/// periods_per_year` and `b_x = -B_n' / n` (`M x N`, dimensionless) so
/// that `y(n) = a_x + b_x X`. `lambda_q` must be ordered non-increasing
/// (per period, finite); `sigma_x` is the per-period `N x N` state
/// innovation covariance in the canonical basis; `maturities` strictly
/// ascending integer periods.
///
/// Further arguments, with defaults: `periods_per_year` (1.0) — it only
/// rescales the intercept `a_x` (pass 12.0 for annualized intercepts of a
/// monthly model); `lambda_q`, `k_inf_q` and `sigma_x` stay per period.
///
/// Validated against a documented-formula NumPy transcription at 1e-12
/// (distinct eigenvalues, the AFNS Jordan block, N = 2, N = 4 with an
/// interior tie), the Jordan-block closed form `b[n][2] = (1/n) sum_{j<n}
/// (j rho^{j-1} + rho^j)` at 1e-12, and the Nelson-Siegel / CDR (2011)
/// agreement of the AFNS case (see `jsz_fit`).
///
/// Returned keys: `a_x`, `b_x`, `k0_q`, `k1_q`, `maturities`.
#[pyfunction]
#[pyo3(signature = (lambda_q, k_inf_q, sigma_x, maturities, periods_per_year = 1.0))]
fn jsz_loadings<'py>(
    py: Python<'py>,
    lambda_q: PyReadonlyArray1<'py, f64>,
    k_inf_q: f64,
    sigma_x: PyReadonlyArray2<'py, f64>,
    maturities: Vec<usize>,
    periods_per_year: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let sig = mat_to_vec2(&to_faer(&sigma_x));
    let l = tsecon_termstructure::jsz_loadings(
        &vec1(&lambda_q),
        k_inf_q,
        &sig,
        &maturities,
        periods_per_year,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("maturities", l.maturities)?;
    d.set_item("a_x", l.a.into_pyarray(py))?;
    d.set_item("b_x", l.b)?;
    d.set_item("k0_q", l.k0_q.into_pyarray(py))?;
    d.set_item("k1_q", l.k1_q)?;
    Ok(d)
}

/// Adds the module's functions to the `_core` extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(jsz_fit, m)?)?;
    m.add_function(wrap_pyfunction!(jsz_loadings, m)?)?;
    Ok(())
}
