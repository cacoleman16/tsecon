//! Python bindings for the 0.10.0 additions to `tsecon-var`:
//! `var_conditional_forecast` (hard-path conditional forecasts, the
//! Doan-Litterman-Sims / Waggoner-Zha closed form, with the 0.11.0 Gaussian
//! soft conditions `condition_sd` / `soft`), `var_diagnostics`
//! (multivariate Portmanteau, multivariate Jarque-Bera — Cholesky and, since
//! 0.11.0, symmetric-root — the Doornik-Hansen 2008 test, stability roots)
//! and `var_select_order` (the lag-order selection that was validated in
//! Rust since 0.1 but never bound). Registered into `_core` through
//! [`register`].

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyIterator, PyString};
use tsecon_var::SoftMode;

use crate::{mat_to_vec2, parse_trend, py_type_name, to_py, var_results};

/// Conditional forecast of a VAR(p), hard or soft: the forecast of every
/// series when some cells of the future path are PINNED to given values, or
/// conditioned with a stated standard deviation — "inflation follows this
/// path for four quarters; what happens to output?"
///
/// `conditions` is a nested list with one row per horizon and one entry per
/// series (the data's column order): a number pins that `(horizon, series)`
/// cell, `None` (or `NaN`) leaves it free — so a NumPy array with NaN for the
/// free cells, or a pandas DataFrame with missing entries, works too. Rows
/// beyond `len(conditions)` up to `steps` are free, so a short list
/// conditions the near horizons only; `steps` defaults to `len(conditions)`.
/// At least one cell must be pinned (the all-free case is `var_forecast`).
/// COST: the guard on `steps` is a MEMORY budget (`steps * k` cells times the
/// constrained cells must stay inside 2^24 doubles), but the work is
/// QUADRATIC in `steps` — measured on a k = 3 VAR(2): 0.03 s at 1 000 steps,
/// 0.46 s at 4 000, 7.6 s at 16 000, 31 s at 32 000 (x4 per doubling), so a
/// `steps` the budget admits can still run for hours. Forecast horizons are
/// tens of periods in practice; treat five figures as a typo.
///
/// Method (Doan-Litterman-Sims 1984; Waggoner-Zha 1999): with the coefficients
/// treated as known and Gaussian innovations, the conditional path is the
/// Gaussian conditioning of the joint forecast-error distribution on the
/// pinned cells, equivalently the unconditional path plus the response to the
/// MINIMUM-NORM sequence of future shocks that delivers the conditions. It is
/// also exactly what a Kalman smoother returns when the free future cells are
/// set missing (Bańbura-Giannone-Lenza 2015) — the second, independent golden
/// leg in fixtures/var_cf.json (statsmodels `VARMAX(...).smooth`).
///
/// SOFT CONDITIONS (Gaussian, stated variance): `condition_sd` has the shape
/// of `conditions` (a nested list, a NaN array or a DataFrame with exactly
/// `len(conditions)` rows) and gives a conditioned cell a STANDARD DEVIATION
/// (not a variance) `> 0`; `0`, `None` or `NaN` keeps the cell hard, and a free
/// cell takes only `None`, `NaN` or `0`. Any sd > 0 REQUIRES `soft`, because
/// the two readings disagree on whether the cell's point is its value:
/// `soft="density"` (Andersson-Palmqvist-Waggoner 2010; "the scenario says
/// inflation ~ N(2, 0.5^2)") — the cell's forecast distribution IS
/// `N(value, sd^2)`: its point is the value and its se is sd exactly, the
/// path, shocks and `mahalanobis` are the hard ones at the central values
/// (so `mahalanobis` measures the central path only and ignores sd), and
/// the covariance gains `sum_i sd_i^2 x_i x_i'` through the hard
/// conditioning gain; `soft="signal"` (Monti 2010; "a nowcast with known
/// noise") — the value is a noisy reading `y + N(0, sd^2)` of the cell: the
/// point, se and shocks are the posterior given the readings (the cell is NOT
/// pinned and its se is below sd), and `mahalanobis` is
/// `r' (B Sigma B' + Omega)^{-1} r`, the plausibility of the readings, chi2
/// with `n_constrained` degrees of freedom. Hard cells (sd 0) are pinned in
/// both readings. Not provided: interval/range (truncated, Waggoner-Zha)
/// conditions and parameter uncertainty.
///
/// Returned keys, every path a `steps x k` nested list (row `h` = horizon
/// `h + 1`): `point` (the conditional mean path; hard cells — and every
/// conditioned cell under `soft="density"` — hold their condition exactly),
/// `unconditional` (the plain iterated forecast — equal to
/// `var_forecast(...)["point"]` bitwise), `cov` (`[h][i][j]`: the conditional
/// forecast-error covariance at each horizon, `k x k`; hard cells have a zero
/// row and column, a density-mode soft cell has `sd^2` and an exact 0 against
/// the other conditioned cells of its horizon), `se` (`sqrt(diag cov)`,
/// exactly 0 at hard cells and exactly sd at density-mode soft cells),
/// `unconditional_se` (the plain forecast's se, to read the variance
/// reduction), `lower`/`upper` (`point -/+ z_{1-alpha/2} se`, innovation
/// uncertainty only, coefficients treated as known — the `var_forecast`
/// convention), `shocks` (the implied reduced-form innovations `u*`, row `s`
/// = period `T+s+1`), `orth_shocks` (the same shocks orthogonalised by the
/// lower Cholesky factor of `sigma_u` in the data's column order — the
/// Waggoner-Zha structural shocks under a recursive ordering; ONLY this key
/// depends on the ordering), `constrained` (`steps x k` booleans: every
/// conditioned cell, hard or soft — a signal-mode soft cell is constrained
/// but NOT pinned; read it with `mode` and `condition_sd`), `n_constrained`,
/// `mahalanobis` (`r' (B Sigma B')^{-1} r`, the squared Sigma-norm of the
/// implied shocks: how far the conditions sit from the unconditional forecast
/// in the model's own metric; `B Sigma B' + Omega` under `soft="signal"`),
/// `mahalanobis_pvalue` (its `chi2(n_constrained)` tail probability — small
/// means the model must be pushed hard to deliver the scenario), `mode`
/// (`"hard"` when no cell has an sd above 0, else the `soft` reading),
/// `condition_sd` (`steps x k`: the sd applied at each soft cell, 0.0 at hard
/// and free cells), `n_soft` (the number of conditioned cells with sd > 0),
/// `steps`, `alpha`.
///
/// Validation: the closed form transcribed in NumPy (documented formula) and
/// statsmodels `VARMAX(...NaN future...).smooth(params)` (independent
/// package) — path, covariance and implied shocks — both pinned at 1e-10 /
/// 1e-8 in fixtures/var_cf.json on a seeded VAR(2) and a transformed US
/// macro system; the unconditional case equals `var_forecast` bitwise;
/// coverage of the free series under the DGP measured by seeded Monte Carlo
/// (see the model card). Soft conditions: fixtures/var_cf_soft.json pins both
/// readings' closed form at 1e-10, statsmodels VARMAX at 1e-8 (signal: a
/// time-varying measurement-noise covariance; density: the hard smoother
/// plus an exact perturbation gain) and, for density, the APW structural
/// form.
///
/// Further arguments, with defaults: `lags` (2), `trend` ("c"), `steps`
/// (None: `len(conditions)`), `alpha` (0.05), `condition_sd` (None: every
/// condition hard), `soft` (None; required — "density" or "signal" —
/// whenever any `condition_sd` is above 0, and refused otherwise).
#[pyfunction]
#[pyo3(signature = (data, conditions, lags = 2, trend = "c", steps = None, alpha = 0.05, condition_sd = None, soft = None))]
#[allow(clippy::too_many_arguments)]
fn var_conditional_forecast<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    conditions: &Bound<'py, PyAny>,
    lags: usize,
    trend: &str,
    steps: Option<usize>,
    alpha: f64,
    condition_sd: Option<&Bound<'py, PyAny>>,
    soft: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    // The two grids and `soft` are extracted here rather than by PyO3, so a
    // bad cell is named by its position and `soft="density"` (a str) is never
    // blamed for a str cell, nor `steps` for an int `soft`.
    let conditions = grid_arg(conditions, "conditions")?;
    let condition_sd = condition_sd
        .map(|o| grid_arg(o, "condition_sd"))
        .transpose()?;
    let soft_mode = parse_soft(soft)?;
    let r = var_results(&data, lags, trend)?;
    let steps = match steps {
        Some(s) => s,
        None => {
            if conditions.is_empty() {
                return Err(PyValueError::new_err(
                    "conditions is empty and steps is None: pass conditions as a nested \
                     list with one row per forecast horizon (None or NaN for a free cell, \
                     a number for a pinned one), or pass steps explicitly",
                ));
            }
            conditions.len()
        }
    };
    let n_soft = match &condition_sd {
        Some(sd) => check_condition_sd(sd, &conditions, r.neqs)?,
        None => 0,
    };
    let cf = match (soft_mode, n_soft) {
        (None, 0) => r.conditional_forecast(steps, &conditions, alpha),
        (Some((mode, _)), n) if n > 0 => r.conditional_forecast_soft(
            steps,
            &conditions,
            condition_sd.as_deref().unwrap_or_default(),
            mode,
            alpha,
        ),
        (None, n) => {
            return Err(PyValueError::new_err(format!(
                "condition_sd gives {n} cell{} a standard deviation > 0, so soft must say \
                 which reading you mean: soft='density' (the cell's forecast IS \
                 N(value, sd^2); Andersson-Palmqvist-Waggoner 2010) or soft='signal' \
                 (value is a noisy reading of the cell; Monti 2010)",
                if n == 1 { "" } else { "s" }
            )));
        }
        (Some((_, name)), _) => {
            let why = if condition_sd.is_none() {
                "condition_sd is None"
            } else {
                "condition_sd gives no cell a standard deviation > 0"
            };
            return Err(PyValueError::new_err(format!(
                "soft='{name}' has nothing to act on: {why}, so every condition is hard \
                 and soft would be ignored; drop soft, or give the cells you want soft a \
                 standard deviation > 0 in condition_sd"
            )));
        }
    }
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("point", mat_to_vec2(&cf.point))?;
    d.set_item("unconditional", mat_to_vec2(&cf.unconditional))?;
    let cov: Vec<Vec<Vec<f64>>> = cf.cov.iter().map(mat_to_vec2).collect();
    d.set_item("cov", cov)?;
    d.set_item("se", mat_to_vec2(&cf.se))?;
    d.set_item("unconditional_se", mat_to_vec2(&cf.unconditional_se))?;
    d.set_item("lower", mat_to_vec2(&cf.lower))?;
    d.set_item("upper", mat_to_vec2(&cf.upper))?;
    d.set_item("shocks", mat_to_vec2(&cf.shocks))?;
    d.set_item("orth_shocks", mat_to_vec2(&cf.orth_shocks))?;
    d.set_item("constrained", cf.constrained.clone())?;
    d.set_item("n_constrained", cf.n_constrained)?;
    d.set_item("mahalanobis", cf.mahalanobis)?;
    d.set_item("mahalanobis_pvalue", cf.mahalanobis_pvalue)?;
    d.set_item("mode", cf.mode.as_str())?;
    d.set_item("condition_sd", mat_to_vec2(&cf.condition_sd))?;
    d.set_item("n_soft", cf.n_soft)?;
    d.set_item("steps", cf.steps)?;
    d.set_item("alpha", cf.alpha)?;
    Ok(d)
}

/// `repr(obj)`, cut to 60 characters so a huge table cannot flood a message.
fn short_repr(obj: &Bound<'_, PyAny>) -> String {
    let s = obj
        .repr()
        .map(|r| r.to_string())
        .unwrap_or_else(|_| "<unprintable>".to_string());
    if s.chars().count() > 60 {
        format!("{}...", s.chars().take(57).collect::<String>())
    } else {
        s
    }
}

/// A double as Python's `repr` would show it: plain digits in
/// `[1e-4, 1e16)`, scientific notation outside (Rust's `{}` never switches,
/// so `-1e-300` would print 300 digits).
fn fmt_value(v: f64) -> String {
    let a = v.abs();
    if a == 0.0 || !a.is_finite() || (1e-4..1e16).contains(&a) {
        format!("{v}")
    } else {
        format!("{v:e}")
    }
}

fn is_text(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyBytes>()
}

/// `obj`'s length and an iterator over it when it is a sized, non-text
/// container. An unsized iterable (a generator, `itertools.repeat`) is
/// `None`: walking it to find a bad entry could run forever.
fn sized_iter<'py>(obj: &Bound<'py, PyAny>) -> Option<(usize, Bound<'py, PyIterator>)> {
    if is_text(obj) {
        return None;
    }
    let n = obj.len().ok()?;
    obj.try_iter().ok().map(|it| (n, it))
}

/// A `rows x k` grid of optional numbers (`conditions`, `condition_sd`): a
/// nested list, or the float64 array the coercion layer makes of a NumPy
/// array or DataFrame. When it does not extract, the first row or cell that
/// is not a sequence / a number-or-None is named by its position; the scan
/// walks sized containers only, at most their length.
fn grid_arg(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<Vec<Option<f64>>>> {
    if let Ok(v) = obj.extract::<Vec<Vec<Option<f64>>>>() {
        return Ok(v);
    }
    let shape = "a nested list with one row per forecast horizon and one entry per series \
                 (None or NaN for an empty cell), a 2-D NumPy array or a pandas DataFrame";
    let Some((n_rows, rows)) = sized_iter(obj) else {
        return Err(PyTypeError::new_err(format!(
            "var_conditional_forecast: {name}={} is of type {}, but {name} takes {shape}",
            short_repr(obj),
            py_type_name(obj)
        )));
    };
    for (h, row) in rows.take(n_rows).enumerate() {
        let row = row?;
        let Some((n_cells, cells)) = sized_iter(&row) else {
            return Err(PyTypeError::new_err(format!(
                "var_conditional_forecast: {name}[{h}]={} is of type {}, but every row of \
                 {name} must be a sequence with one entry per series ({name} takes {shape})",
                short_repr(&row),
                py_type_name(&row)
            )));
        };
        for (j, cell) in cells.take(n_cells).enumerate() {
            let cell = cell?;
            if cell.extract::<Option<f64>>().is_err() {
                return Err(PyTypeError::new_err(format!(
                    "var_conditional_forecast: {name}[{h}][{j}]={} is of type {}, but each \
                     cell of {name} takes a real number or None",
                    short_repr(&cell),
                    py_type_name(&cell)
                )));
            }
        }
    }
    // Every row and cell converts on its own: surface PyO3's own message.
    obj.extract::<Vec<Vec<Option<f64>>>>()
}

/// `soft`: None, "density" or "signal" (with the spelling, for messages).
fn parse_soft(soft: Option<&Bound<'_, PyAny>>) -> PyResult<Option<(SoftMode, String)>> {
    let Some(o) = soft else { return Ok(None) };
    let accepted = "soft='density' (the cell's forecast IS N(value, sd^2); \
                    Andersson-Palmqvist-Waggoner 2010) or soft='signal' (value is a noisy \
                    reading of the cell; Monti 2010), or soft=None when no condition_sd is \
                    above 0";
    let Ok(s) = o.extract::<String>() else {
        return Err(PyTypeError::new_err(format!(
            "var_conditional_forecast: soft={} is of type {}, but soft takes {accepted}",
            short_repr(o),
            py_type_name(o)
        )));
    };
    match s.as_str() {
        "density" => Ok(Some((SoftMode::Density, s))),
        "signal" => Ok(Some((SoftMode::Signal, s))),
        _ => Err(PyValueError::new_err(format!(
            "soft={} is not a soft-condition reading: pass {accepted}",
            short_repr(o)
        ))),
    }
}

/// Shape and cell checks of `condition_sd`, with the cell named; returns
/// the number of cells with a standard deviation above 0. A row of
/// `conditions` with the wrong length is left to the crate's own refusal.
fn check_condition_sd(
    sd: &[Vec<Option<f64>>],
    conditions: &[Vec<Option<f64>>],
    k: usize,
) -> PyResult<usize> {
    if sd.len() != conditions.len() {
        return Err(PyValueError::new_err(format!(
            "condition_sd has {} rows but conditions has {}: condition_sd must have exactly \
             one row per row of conditions (expected {}, got {}) — the standard deviation of \
             each conditioned cell, 0, None or NaN for a hard one and None or NaN at a free \
             cell",
            sd.len(),
            conditions.len(),
            conditions.len(),
            sd.len()
        )));
    }
    for (h, row) in sd.iter().enumerate() {
        if row.len() != k {
            return Err(PyValueError::new_err(format!(
                "condition_sd[{h}] has {} entries but the data has k = {k} series: every \
                 row of condition_sd must have exactly one entry per series, like the rows \
                 of conditions (expected {k}, got {})",
                row.len(),
                row.len()
            )));
        }
    }
    let mut n_soft = 0;
    for (h, row) in sd.iter().enumerate() {
        for (j, c) in row.iter().enumerate() {
            let Some(v) = *c else { continue };
            if v.is_nan() {
                continue;
            }
            if !(v.is_finite() && v >= 0.0) {
                return Err(PyValueError::new_err(format!(
                    "condition_sd[{h}][{j}] = {}: a standard deviation must be finite and \
                     >= 0 (0 or NaN means a hard condition)",
                    fmt_value(v)
                )));
            }
            if v > 0.0 && !(v * v).is_finite() {
                return Err(PyValueError::new_err(format!(
                    "condition_sd[{h}][{j}] = {v:e}: its square (the variance) overflows a \
                     double; a standard deviation must be below about 1.3e154"
                )));
            }
            let free = conditions[h].len() == k && conditions[h][j].is_none_or(f64::is_nan);
            if v > 0.0 && free {
                return Err(PyValueError::new_err(format!(
                    "condition_sd[{h}][{j}] = {} but conditions[{h}][{j}] is free: give the \
                     cell a value or drop the sd",
                    fmt_value(v)
                )));
            }
            // Counted even beside a mis-sized conditions row, so the call
            // reaches the crate's refusal of that row instead of calling
            // `soft` inert.
            if v > 0.0 {
                n_soft += 1;
            }
        }
    }
    Ok(n_soft)
}

/// Residual diagnostics of a fitted VAR(p) — the three checks every VAR
/// write-up reports: residual autocorrelation, residual normality, and
/// stability.
///
/// PORTMANTEAU (Hosking 1980; Lütkepohl 2005, section 4.4.3): with `C_i` the
/// lag-`i` autocovariances of the column-centred residuals, the unadjusted
/// `Q = T sum_{i=1}^{nlags} tr(C_i' C_0^{-1} C_i C_0^{-1})` and the
/// small-sample adjusted `T^2 sum_i tr(...) / (T - i)`, both `chi2(k^2
/// (nlags - p))` under white residuals; `nlags` must exceed the lag order
/// `lags`. Report the adjusted statistic: it corrects the unadjusted one's
/// small-sample undersize (the crate's seeded Monte Carlo at T = 200, k = 2,
/// nlags = 8, 1000 replications measures 4.6% for Q and 5.4% adjusted at a
/// nominal 5%, MC se 0.7 points; the gap widens as nlags grows relative to
/// T). Matches statsmodels `VARResults.test_whiteness(nlags,
/// adjusted=False/True)`.
///
/// NORMALITY — three statistics on the same column-centred residuals and ML
/// covariance (the same three `multivariate_normality` computes on any
/// residual matrix):
///
/// * multivariate Jarque-Bera (Lütkepohl 2005, section 4.5) with the LOWER
///   CHOLESKY factor — the statsmodels `test_normality` convention and
///   JMulTi's "Lütkepohl (1993)" block, so the components depend on the
///   column order, as orthogonalised impulse responses do: `skewness = T
///   b1'b1 / 6`, `kurtosis = T b2'b2 / 24` (each `chi2(k)`) add up to the
///   omnibus `jarque_bera` (`chi2(2k)`);
/// * the same statistic with the SYMMETRIC inverse root of the covariance and
///   no further transforms — what JMulTi prints under "Doornik & Hansen
///   (1994)" (all 16 bundled JMulTi outputs reproduced to their printed
///   precision): `jarque_bera_symmetric` and its parts; it does not depend on
///   the column order but DOES depend on the units of the series;
/// * the Doornik-Hansen (2008) omnibus test — the correlation-matrix
///   symmetric root plus D'Agostino's skewness and the Shenton-Bowman /
///   Wilson-Hilferty kurtosis transforms, so the z-scores are close to N(0, 1)
///   in small samples: `doornik_hansen` (`chi2(2k)`) and its parts. It is
///   invariant to the column order and to the units, and is the one to
///   report. Every `doornik_hansen*` key is None when `nobs < 8` (the
///   skewness transform degenerates at T = 7; this bundle never refuses on
///   its account), and also when the residual correlation matrix is
///   numerically singular; the `*_symmetric` keys are None when the residual
///   covariance's smallest eigenvalue is `<= 1e-12 x` its trace (series on
///   wildly different scales — rescale them to see it).
///
/// STABILITY: `roots` are the moduli of the reciprocal characteristic roots
/// in DESCENDING order (statsmodels `VARResults.roots`; stable iff the LAST
/// one exceeds 1), `eigenvalue_moduli` the companion eigenvalue moduli in
/// descending order (the first is the spectral radius; stable iff it is
/// below 1), and `is_stable` the verdict.
///
/// Returned keys: `portmanteau`, `portmanteau_adjusted`, `portmanteau_df`,
/// `portmanteau_pvalue`, `portmanteau_adjusted_pvalue`, `nlags`,
/// `jarque_bera`, `jarque_bera_pvalue`, `jarque_bera_df`, `skewness`,
/// `skewness_pvalue`, `kurtosis`, `kurtosis_pvalue`, `skewness_components`
/// (the per-series third moments `b1`), `kurtosis_components` (the
/// per-series excess fourth moments `b2`), `jarque_bera_symmetric`,
/// `jarque_bera_symmetric_pvalue` (df `jarque_bera_df`),
/// `skewness_symmetric`, `skewness_symmetric_pvalue`, `kurtosis_symmetric`,
/// `kurtosis_symmetric_pvalue`, `skewness_symmetric_components`,
/// `kurtosis_symmetric_components`, `doornik_hansen`,
/// `doornik_hansen_pvalue`, `doornik_hansen_df` (`2k`),
/// `doornik_hansen_skewness` (`z1'z1`, `chi2(k)`),
/// `doornik_hansen_skewness_pvalue`, `doornik_hansen_kurtosis` (`z2'z2`,
/// `chi2(k)`), `doornik_hansen_kurtosis_pvalue`,
/// `doornik_hansen_z_skewness`, `doornik_hansen_z_kurtosis` (the
/// per-component z-scores), `roots`, `eigenvalue_moduli`, `is_stable`,
/// `nobs` (the effective sample size `T` the residual tests use), `k`,
/// `lags`.
///
/// Validation: statsmodels `test_whiteness` / `test_normality` / `roots` /
/// `is_stable` pinned at 1e-10 (p-values also on the log scale) in
/// fixtures/var_diag.json on seeded VAR(2)/VAR(3)/underfit VAR(1) systems,
/// a Student-t-driven VAR(2) and a transformed US macro system; the
/// symmetric-root block against JMulTi's printed output (16 residual
/// matrices, 5e-5) and the generator's transcription (1e-10); the
/// Doornik-Hansen (2008) formulas transcribed at 1e-10 (z1 matched to
/// `scipy.stats.skewtest`); size and power measured by seeded Monte Carlo
/// (see the model card).
///
/// Further arguments, with defaults: `lags` (2), `trend` ("c"), `nlags`
/// (10).
#[pyfunction]
#[pyo3(signature = (data, lags = 2, trend = "c", nlags = 10))]
fn var_diagnostics<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    trend: &str,
    nlags: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = var_results(&data, lags, trend)?;
    let dg = r.diagnostics(nlags).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("portmanteau", dg.portmanteau.statistic)?;
    d.set_item("portmanteau_adjusted", dg.portmanteau.adjusted)?;
    d.set_item("portmanteau_df", dg.portmanteau.df)?;
    d.set_item("portmanteau_pvalue", dg.portmanteau.pvalue)?;
    d.set_item(
        "portmanteau_adjusted_pvalue",
        dg.portmanteau.adjusted_pvalue,
    )?;
    d.set_item("nlags", dg.portmanteau.nlags)?;
    d.set_item("jarque_bera", dg.normality.statistic)?;
    d.set_item("jarque_bera_pvalue", dg.normality.pvalue)?;
    d.set_item("jarque_bera_df", dg.normality.df)?;
    d.set_item("skewness", dg.normality.skewness)?;
    d.set_item("skewness_pvalue", dg.normality.skewness_pvalue)?;
    d.set_item("kurtosis", dg.normality.kurtosis)?;
    d.set_item("kurtosis_pvalue", dg.normality.kurtosis_pvalue)?;
    d.set_item(
        "skewness_components",
        dg.normality.skewness_components.clone(),
    )?;
    d.set_item(
        "kurtosis_components",
        dg.normality.kurtosis_components.clone(),
    )?;
    crate::var_normality::set_symmetric_keys(&d, dg.normality_symmetric.as_ref())?;
    crate::var_normality::set_doornik_hansen_keys(&d, dg.doornik_hansen.as_ref())?;
    d.set_item("roots", dg.roots.clone())?;
    d.set_item("eigenvalue_moduli", dg.eigenvalue_moduli.clone())?;
    d.set_item("is_stable", dg.is_stable)?;
    d.set_item("nobs", dg.nobs)?;
    d.set_item("k", dg.neqs)?;
    d.set_item("lags", dg.lags)?;
    Ok(d)
}

/// VAR lag-order selection by information criteria on a COMMON sample
/// (statsmodels `VAR.select_order` conventions; Lütkepohl 2005, section
/// 4.3): every candidate order `p` is fitted after dropping the first
/// `max_lags - p` rows, so all candidates share the same `n - max_lags`
/// effective observations and their criteria are comparable — which is why
/// this is not the same as reading `aic` off `var_fit` at each `lags`. The
/// candidates run from `p = 0` (an intercept-only baseline) with `trend="c"`
/// and from `p = 1` with `trend="n"`; ties go to the smaller order.
///
/// Returned keys: `aic`, `bic`, `hqic`, `fpe` (the selected order under each
/// criterion), `candidates` (the candidate orders, ascending), `aic_values`,
/// `bic_values`, `hqic_values`, `fpe_values` (one value per candidate),
/// `max_lags`, `trend`.
///
/// Validation: statsmodels `VAR.select_order` picks (fixtures/var.json,
/// since 0.1) and the full criterion tables (fixtures/var_diag.json) at 1e-8.
///
/// Further arguments, with defaults: `max_lags` (8), `trend` ("c").
#[pyfunction]
#[pyo3(signature = (data, max_lags = 8, trend = "c"))]
fn var_select_order<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    max_lags: usize,
    trend: &str,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_var::tsecon_linalg::faer::Mat;
    if max_lags == 0 {
        return Err(PyValueError::new_err(
            "max_lags = 0: the lag-order search needs at least one candidate order; pass \
             max_lags >= 1 (8 is a common quarterly choice)",
        ));
    }
    let tr = parse_trend(trend)?;
    let a = data.as_array();
    // `select_order` reserves one candidate per order up front, so a
    // `max_lags` typo would ask the allocator for terabytes before the
    // per-candidate estimation could refuse it. A candidate order at or
    // beyond the sample length can never be estimated anyway (it leaves
    // `n - max_lags <= 0` common observations), so refuse it here, naming
    // the parameter and the row count.
    if max_lags >= a.nrows() {
        return Err(PyValueError::new_err(format!(
            "max_lags = {max_lags} with only {} rows of data: every candidate order is \
             fitted on the common sample of n - max_lags = {} observations, so max_lags \
             must be smaller than the number of rows (and in practice far smaller — it \
             also has to leave more observations than k * max_lags + 1 coefficients per \
             equation); pass a smaller max_lags",
            a.nrows(),
            a.nrows() as i128 - max_lags as i128,
        )));
    }
    let m = Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let sel = tsecon_var::select_order(m.as_ref(), max_lags, tr).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("aic", sel.aic)?;
    d.set_item("bic", sel.bic)?;
    d.set_item("hqic", sel.hqic)?;
    d.set_item("fpe", sel.fpe)?;
    let cands: Vec<usize> = sel.candidates.iter().map(|c| c.lags).collect();
    d.set_item("candidates", cands)?;
    d.set_item(
        "aic_values",
        sel.candidates.iter().map(|c| c.aic).collect::<Vec<f64>>(),
    )?;
    d.set_item(
        "bic_values",
        sel.candidates.iter().map(|c| c.bic).collect::<Vec<f64>>(),
    )?;
    d.set_item(
        "hqic_values",
        sel.candidates.iter().map(|c| c.hqic).collect::<Vec<f64>>(),
    )?;
    d.set_item(
        "fpe_values",
        sel.candidates.iter().map(|c| c.fpe).collect::<Vec<f64>>(),
    )?;
    d.set_item("max_lags", max_lags)?;
    d.set_item("trend", trend)?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(var_conditional_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(var_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(var_select_order, m)?)?;
    Ok(())
}
