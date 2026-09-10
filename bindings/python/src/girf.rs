//! Python bindings for the Koop-Pesaran-Potter (1996) generalized
//! impulse-response engine of `tsecon-var`: `var_girf` (the linear VAR —
//! the engine's exact reduction to the closed-form IRFs) and
//! `threshold_var_girf` (regime-dependent GIRFs of the two-regime threshold
//! VAR through `tsecon-regime`). Registered into `_core` through
//! [`register`].

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{data_to_rows, to_py, var_results};

fn parse_shock(shock: &str, shock_var: usize, size: f64) -> PyResult<tsecon_var::GirfShock> {
    match shock {
        "orthogonal" => Ok(tsecon_var::GirfShock::Orthogonal {
            var: shock_var,
            size,
        }),
        "generalized" => Ok(tsecon_var::GirfShock::Generalized {
            var: shock_var,
            size,
        }),
        other => Err(PyValueError::new_err(format!(
            "unknown shock {other:?}; expected \"orthogonal\" (size standard deviations of \
             the shock_var-th Cholesky-orthogonalized innovation in the variable ordering — \
             var_irf(orth=True)'s shock) or \"generalized\" (Pesaran-Shin 1998: innovation \
             shock_var moved by size standard deviations, the others by their conditional \
             mean, no ordering)"
        ))),
    }
}

fn set_common<'py>(d: &Bound<'py, PyDict>, g: &tsecon_var::Girf) -> PyResult<()> {
    d.set_item("girf", g.girf.clone())?;
    d.set_item("lower", g.lower.clone())?;
    d.set_item("upper", g.upper.clone())?;
    d.set_item("per_history", g.per_history.clone())?;
    d.set_item("mc_se", g.mc_se.clone())?;
    d.set_item("draw_sd", g.draw_sd.clone())?;
    d.set_item("draw_lower", g.draw_lower.clone())?;
    d.set_item("draw_upper", g.draw_upper.clone())?;
    d.set_item("n_histories", g.n_histories)?;
    d.set_item("n_draws", g.n_draws)?;
    d.set_item("n_effective_draws", g.n_effective_draws)?;
    d.set_item("horizon", g.horizon)?;
    Ok(())
}

/// Generalized impulse responses (Koop-Pesaran-Potter 1996) of a linear
/// VAR(p) by simulation — the engine's exact reduction to the closed-form
/// impulse responses, exposed so the nonlinear GIRFs can be checked
/// against a linear benchmark on the same footing.
///
/// For every lag window of `data` (or a seeded subsample of `histories`
/// of them) and every future-innovation draw, the fitted VAR is simulated
/// forward twice with the same innovations — once with the shock added
/// to the impact-period innovation, once without — and the paired
/// difference is averaged (common random numbers; `antithetic` `(+z, -z)`
/// pairs when set, which needs an even `n_draws`). In a linear model the
/// paired difference is `Psi_h delta` for every draw and every history,
/// so the result carries no Monte Carlo noise and does not depend on
/// `n_draws` or `seed` beyond rounding; `mc_se`/`draw_sd` are zero to
/// rounding (below 1e-15), and `mc_se` is NaN below two effective draws —
/// at the default `n_draws=2` with `antithetic=True` there is exactly one,
/// so `mc_se` is NaN there (pass `n_draws=4` or `antithetic=False` for a
/// finite value).
///
/// `shock`: `"orthogonal"` — `size` standard deviations of the
/// `shock_var`-th Cholesky-orthogonalized innovation in the variable
/// ordering, so `girf[h]` equals `var_irf(orth=True)[h][:, shock_var] *
/// size` to 1e-12; `"generalized"` — the Pesaran-Shin (1998) shock
/// (innovation `shock_var` moved by `size` standard deviations, the
/// others by their conditional expectation given that move; no ordering
/// assumption), so `girf[h]` equals `Phi_h Sigma e_j / sqrt(sigma_jj) *
/// size` with `Sigma` the df-adjusted residual covariance. `size` may be
/// negative.
///
/// Returns `girf` (`[h][variable]`, `h = 0..horizon`, mean over
/// histories), `lower`/`upper` (the `bands` quantiles across histories —
/// zero width here), `per_history` (`[history][h][variable]`), `mc_se`
/// (Monte Carlo standard error of `girf`), `draw_sd` (across-draw spread
/// of one realized paired difference), `draw_lower`/`draw_upper` (mean
/// over histories of the per-history across-draw `bands` quantiles),
/// `n_histories`, `n_draws`, `n_effective_draws` (`n_draws / 2` under
/// antithetic), `horizon`, `shock` (echo), `shock_var`, `shock_size_used`
/// (the impact-period innovation to `shock_var` in raw units:
/// `size * P[j, j]` orthogonal, `size * sqrt(sigma_jj)` generalized), and
/// `shock_vector` (the full raw innovation added at impact).
///
/// Validation: statsmodels `VARResults.irf(orth=True)` and the
/// Pesaran-Shin closed form, both at 1e-12 with a single draw
/// (fixtures/girf.json); equality with `var_irf(orth=True)` is asserted
/// in the binding tests.
///
/// Further arguments, with defaults: `shock_var` (0), `size` (1.0),
/// `shock` ("orthogonal"), `horizon` (10), `n_draws` (2), `seed` (0),
/// `trend` ("c"), `antithetic` (True), `histories` (None = every lag
/// window; an int draws a seeded subsample of that many — a count at or
/// above the number of available windows uses all of them, reported in
/// `n_histories`), `bands` ((0.16, 0.84)).
#[pyfunction]
#[pyo3(signature = (data, p, shock_var = 0, size = 1.0, shock = "orthogonal", horizon = 10,
                    n_draws = 2, seed = 0, trend = "c", antithetic = true, histories = None,
                    bands = (0.16, 0.84)))]
#[allow(clippy::too_many_arguments)]
fn var_girf<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    p: usize,
    shock_var: usize,
    size: f64,
    shock: &str,
    horizon: usize,
    n_draws: usize,
    seed: u64,
    trend: &str,
    antithetic: bool,
    histories: Option<usize>,
    bands: (f64, f64),
) -> PyResult<Bound<'py, PyDict>> {
    let res = var_results(&data, p, trend)?;
    let opts = tsecon_var::GirfOptions {
        shock: parse_shock(shock, shock_var, size)?,
        horizon,
        n_draws,
        seed,
        antithetic,
        bands,
    };
    let g = tsecon_var::var_girf(&res, &opts, histories).map_err(to_py)?;
    let d = PyDict::new(py);
    set_common(&d, &g)?;
    d.set_item("shock", shock)?;
    d.set_item("shock_var", shock_var)?;
    d.set_item("shock_size_used", g.shock_by_state[0][shock_var])?;
    d.set_item("shock_vector", g.shock_by_state[0].clone())?;
    Ok(d)
}

/// Regime-dependent generalized impulse responses (Koop-Pesaran-Potter
/// 1996) of the two-regime threshold VAR: fit `threshold_var` with the
/// same `p`/`threshold_index`/`delay`|`delays`/`trim`/`constant`, then
/// simulate the fitted nonlinear system forward from the sample's actual
/// lag windows, regime-switching period by period.
///
/// Histories are every lag window `t >= max(p, delay)` of `data` (the
/// shock hits period `t`; its regime is decided by `data[t - delay,
/// threshold_index] <= threshold`), in time order; `regime="low"`/`"high"`
/// keeps only the windows whose shock-date regime is that one, and
/// `histories=m` a seeded subsample of `m` of the selected windows. For
/// each history and each of `n_draws` future-innovation draws the model is
/// simulated twice with the same standard-normal draws — the shocked path
/// adds the shock at impact — and the paired difference is averaged. At
/// every period each path reads its own regime from its own simulated
/// window and scales the common draw by the Cholesky factor of THAT
/// regime's ML residual covariance (`sigma_low`/`sigma_high`), so a path
/// that crosses the threshold switches both coefficients and innovation
/// covariance (the regime-by-regime draw scheme of Balke 2000; R tsDyn's
/// GIRF pools residuals because its TVAR fits one covariance). The impact
/// shock is scaled by the covariance of the regime the history is in at
/// the shock date: `"orthogonal"` — `size` standard deviations of the
/// `shock_var`-th Cholesky-orthogonalized innovation of that regime;
/// `"generalized"` — the Pesaran-Shin shock of that regime (`size *
/// Sigma_s[:, j] / sqrt(Sigma_s[j, j])`); the raw impact therefore differs
/// across regimes when their covariances do (see `shock_size_used`).
/// `antithetic` uses `(+z, -z)` pairs (even `n_draws`); the Monte Carlo
/// standard error is computed from the pair means.
///
/// Returns `girf` (`[h][variable]`, mean over the used histories),
/// `lower`/`upper` (the `bands` quantiles across the used histories — the
/// KPP history-conditional distribution), `per_history`
/// (`[history][h][variable]`), `mc_se` (Monte Carlo standard error of
/// `girf`, histories fixed; NaN below two effective draws), `draw_sd`
/// (across-draw spread of one realized paired difference), `draw_lower`/
/// `draw_upper` (mean over histories of the per-history across-draw
/// `bands` quantiles), `girf_low_regime`/`girf_high_regime` (means over
/// the used histories of each regime; `None` when the selection holds
/// none of that regime), `history_regimes` (0 low / 1 high per used
/// history), `history_times` (the shock date `t` of each), `n_histories`,
/// `n_low_histories`, `n_high_histories`, `n_draws`, `n_effective_draws`,
/// `horizon`, `shock` (echo), `shock_var`, `shock_size_used` (`[low,
/// high]` impact-period innovation to `shock_var` in raw units),
/// `shock_vector` (`[regime][variable]` raw innovation added at impact),
/// `regime` (echo), `threshold`, `delay`, `threshold_index`.
///
/// Reproducible: one Philox substream per (history, draw) spawned from
/// `seed`, bit-identical at any thread count and across processes.
/// Validation (honest grade): the engine's linear reduction is pinned at
/// 1e-12 against statsmodels and the Pesaran-Shin closed form
/// (`var_girf`); the regime-switching simulation is pinned at 1e-10
/// against an independent NumPy transcription of the documented engine
/// reproducing its random streams (fixtures/girf.json); sign asymmetry,
/// size non-proportionality, regime dependence, 1/sqrt(n_draws)
/// convergence and antithetic variance reduction are measured by seeded
/// Monte Carlo property tests (see the model card). No third-party TVAR
/// GIRF runs in the build container.
///
/// Further arguments, with defaults: `threshold_index` (0), `delay` (1),
/// `trim` (0.1), `delays` (None; a list searches the delay and overrides
/// `delay`), `constant` (True), `shock_var` (0), `size` (1.0), `shock`
/// ("orthogonal"), `horizon` (20), `n_draws` (500), `seed` (0), `regime`
/// ("all"), `histories` (None = every selected window; an int draws a seeded
/// subsample of that many — a count at or above the number of selected
/// windows uses all of them, reported in `n_histories`), `bands`
/// ((0.16, 0.84)), `antithetic` (True).
#[pyfunction]
#[pyo3(signature = (data, p, threshold_index = 0, delay = 1, trim = 0.10, delays = None,
                    constant = true, shock_var = 0, size = 1.0, shock = "orthogonal",
                    horizon = 20, n_draws = 500, seed = 0, regime = "all", histories = None,
                    bands = (0.16, 0.84), antithetic = true))]
#[allow(clippy::too_many_arguments)]
fn threshold_var_girf<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    p: usize,
    threshold_index: usize,
    delay: usize,
    trim: f64,
    delays: Option<Vec<usize>>,
    constant: bool,
    shock_var: usize,
    size: f64,
    shock: &str,
    horizon: usize,
    n_draws: usize,
    seed: u64,
    regime: &str,
    histories: Option<usize>,
    bands: (f64, f64),
    antithetic: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let regime_sel = match regime {
        "all" => tsecon_regime::GirfRegime::All,
        "low" => tsecon_regime::GirfRegime::Low,
        "high" => tsecon_regime::GirfRegime::High,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown regime {other:?}; expected \"all\" (every lag window; the per-regime \
                 averages come back as girf_low_regime/girf_high_regime), \"low\" (windows \
                 whose shock date has z_t <= threshold) or \"high\" (z_t > threshold)"
            )))
        }
    };
    let rows = data_to_rows(&data);
    let dl: Vec<usize> = delays.unwrap_or_else(|| vec![delay]);
    let opts = tsecon_regime::TvarGirfOptions {
        shock: parse_shock(shock, shock_var, size)?,
        horizon,
        n_draws,
        seed,
        antithetic,
        bands,
        regime: regime_sel,
        histories,
    };
    let (fit, g) =
        tsecon_regime::threshold_var_girf(&rows, p, threshold_index, &dl, trim, constant, &opts)
            .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("girf", g.girf)?;
    d.set_item("lower", g.lower)?;
    d.set_item("upper", g.upper)?;
    d.set_item("per_history", g.per_history)?;
    d.set_item("mc_se", g.mc_se)?;
    d.set_item("draw_sd", g.draw_sd)?;
    d.set_item("draw_lower", g.draw_lower)?;
    d.set_item("draw_upper", g.draw_upper)?;
    match g.girf_low_regime {
        Some(v) => d.set_item("girf_low_regime", v)?,
        None => d.set_item("girf_low_regime", py.None())?,
    }
    match g.girf_high_regime {
        Some(v) => d.set_item("girf_high_regime", v)?,
        None => d.set_item("girf_high_regime", py.None())?,
    }
    d.set_item("history_regimes", g.history_regimes)?;
    d.set_item("history_times", g.history_times)?;
    d.set_item("n_histories", g.n_histories)?;
    d.set_item("n_low_histories", g.n_low_histories)?;
    d.set_item("n_high_histories", g.n_high_histories)?;
    d.set_item("n_draws", g.n_draws)?;
    d.set_item("n_effective_draws", g.n_effective_draws)?;
    d.set_item("horizon", horizon)?;
    d.set_item("shock", shock)?;
    d.set_item("shock_var", shock_var)?;
    d.set_item(
        "shock_size_used",
        vec![
            g.shock_by_regime[0][shock_var],
            g.shock_by_regime[1][shock_var],
        ],
    )?;
    d.set_item("shock_vector", g.shock_by_regime)?;
    d.set_item("regime", regime)?;
    d.set_item("threshold", g.threshold)?;
    d.set_item("delay", g.delay)?;
    d.set_item("threshold_index", fit.threshold_index)?;
    Ok(d)
}

/// Adds the module's functions to the `_core` extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(var_girf, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_var_girf, m)?)?;
    Ok(())
}
