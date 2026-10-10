//! Python binding for the autoregressive-distributed-lag MIDAS regression of
//! `tsecon-midas` (`adl_midas`), validated in Rust since 0.1 and bound in
//! 0.11. The binding assembles the lagged-`y` columns from `y` itself and
//! trims every column to the rows that have them, so the caller passes the
//! same `(y, hf_lags)` pair `umidas` takes. Registered into `_core` through
//! [`register`].

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::vec1;

/// Largest accepted design, `nobs x (1 + P + K)` cells: `2^26` doubles is
/// 512 MiB per copy, the cap `ardl_bounds_test` puts on its design. The
/// lagged-`y` block is built here from a scalar `ar_lags`, so without the
/// cap an absurd `ar_lags` on a long series would allocate `P (n - P)`
/// cells before any check could fail.
const MAX_DESIGN_CELLS: u128 = 1 << 26;

/// Work budget, in units of `nobs * k^2` per HAC lag plus two for the OLS
/// fit and the sandwich (`k = 1 + P + K` coefficients, the HAC meat summing
/// `min(maxlags, nobs - 1)` lagged score cross-products of `k^2` terms each).
/// Measured at 0.29-1.65 ns per unit (`lab/audit/wave11/adl_midas_cost.py`;
/// a 160-row fit, 0.3 ms, comes out at 4.1), so a call at the cap takes
/// under 20 s; past it the call is refused
/// before any column is built, naming `maxlags` when the bandwidth is what
/// exceeds it and `ar_lags` otherwise.
const WORK_BUDGET: u128 = 1 << 33;

fn err(msg: String) -> PyErr {
    PyValueError::new_err(format!("adl_midas: {msg}"))
}

/// ADL-MIDAS: autoregressive-distributed-lag mixed-frequency regression
/// (Andreou, Ghysels & Kourtellos 2013) — U-MIDAS plus `ar_lags` lags of
/// the low-frequency target, fitted by OLS:
/// `y_t = const + sum_{p=1..P} rho_p y_{t-p} + sum_{k=1..K} b_k hf_lags[t, k-1] + u_t`
/// over the rows `t = P..n-1` (`P = ar_lags`), the first `P` rows being
/// used only as lags.
///
/// `y` is the length-`n` low-frequency target and `hf_lags` the `n x K`
/// stacked high-frequency lags aligned with it — the same pair `umidas`
/// takes (row `t` belongs to period `t`). Column 1 is high-frequency LAG 0,
/// the last high-frequency observation inside period `t`, column 2 the one
/// before it, and so on (the `stack_high_freq_lags` convention, most recent
/// first), which is why the coefficients are named `hf1..hfK` and not
/// `hf.L1..hf.LK`. The lagged-`y` columns are built here from `y` itself:
/// column `y.Lp` is `y[P - p : n - p]`.
///
/// Returned keys: `params` (`[const, y.L1..y.LP, hf1..hfK]`),
/// `param_names` (those labels), `bse`, `tvalues`, `rsquared` (centered),
/// `nobs` (`n - P`, the rows fitted), `residuals` (length `nobs`, aligned
/// with `y[P:]`), `first_index` (`P`: the index into `y` of the first
/// fitted row), `se_type` (echo) and `maxlags` (the HAC bandwidth actually
/// used; None under `"nonrobust"`).
///
/// `se_type`: `"hac"` (default) — Bartlett/Newey-West with the `n/(n - k)`
/// small-sample factor (statsmodels `cov_type="HAC"` with
/// `use_correction=True`) at `maxlags`, which defaults to the Newey-West
/// rule `floor(4 (nobs/100)^(2/9))` on the EFFECTIVE number of rows
/// `nobs = n - P` (the rows the regression actually has; `umidas` applies
/// the rule to its own rows, which here would be `n`); or `"nonrobust"`
/// (`sigma2 = RSS / (nobs - k)`), where an explicit `maxlags` is refused —
/// it would be ignored.
///
/// Refusals name the argument: `ar_lags = 0` (that model is `umidas`); a
/// row count of `hf_lags` different from `len(y)`, or zero columns; NaN or
/// infinity in `y` or `hf_lags`; too few rows for the `1 + P + K`
/// coefficients (`ar_lags`); `maxlags` under `"nonrobust"`; an unknown
/// `se_type`. A collinear design is refused by the OLS engine. COST: the
/// design `nobs x (1 + P + K)` is capped at 2^26 cells and the work,
/// about `nobs k^2 (min(maxlags, nobs - 1) + 2)` units with
/// `k = 1 + P + K` (no lag term under `"nonrobust"`), at 2^33 — measured at
/// 0.29-1.65 ns per unit, so a call at the cap takes under 20 s (11.3 s at
/// n = 100000, K = 6, maxlags = 1340, the cap itself;
/// `lab/audit/wave11/adl_midas_cost.py`). Past either cap the call is
/// refused before any column is built, naming `maxlags` when the bandwidth
/// alone exceeds the budget and `ar_lags` otherwise.
///
/// Validated against statsmodels 0.15.0 `OLS` on the same design built
/// independently with `pandas.shift` — params, residuals, R², and bse /
/// t-values under nonrobust and HAC (default and `maxlags=8`) at 1e-10
/// (achieved 6.9e-15) — `fixtures/adl_midas.json`. On persistent LEVEL
/// targets (a log-GDP level, an index near 100: the lagged-y columns nearly
/// collinear with the constant, `cond(X)` up to 1.4e5) the fit is pinned
/// against the exact rational least-squares solution of the same float64
/// design: params within 3e-13 and HAC bse within 1.3e-13 (statsmodels'
/// pinv params are within 1e-12 of it, its HAC bse 7.8e-10). The OLS
/// engine solves by Householder QR of the equilibrated design, so finite
/// inputs of any magnitude whose answer is representable return finite
/// params, bse and R².
///
/// Further arguments, with defaults: `ar_lags` (1), `se_type` ("hac"),
/// `maxlags` (None: the effective-rows Newey-West rule under "hac").
#[pyfunction]
#[pyo3(signature = (y, hf_lags, ar_lags = 1, se_type = "hac", maxlags = None))]
fn adl_midas<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    hf_lags: PyReadonlyArray2<'py, f64>,
    ar_lags: usize,
    se_type: &str,
    maxlags: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    if ar_lags == 0 {
        return Err(err(
            "ar_lags = 0 leaves no autoregressive term; that model is the unrestricted \
             MIDAS regression — call umidas(y, hf_lags) instead"
                .to_owned(),
        ));
    }
    let yv = vec1(&y);
    let n = yv.len();
    if n == 0 {
        return Err(err(
            "y is empty; pass the low-frequency target series".to_owned()
        ));
    }
    let a = hf_lags.as_array();
    let k = a.ncols();
    if a.nrows() != n {
        return Err(err(format!(
            "hf_lags has {} row(s) but y has {n} observations; hf_lags must be n x K with \
             row t holding the K high-frequency lags of period t (aligned with y)",
            a.nrows()
        )));
    }
    if k == 0 {
        return Err(err(
            "hf_lags has zero columns; pass at least one high-frequency lag column \
             (with no high-frequency regressor the model is a plain AR)"
                .to_owned(),
        ));
    }
    if let Some((i, v)) = yv.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(err(format!(
            "y contains a non-finite value ({v}) at index {i}; drop or impute it first"
        )));
    }
    if let Some(((i, j), v)) = a.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(err(format!(
            "hf_lags contains a non-finite value ({v}) at [{i}, {j}]; drop or impute it first"
        )));
    }
    // 1 + P + K coefficients need at least one residual degree of freedom
    // on the n - P fitted rows. Checked before any column is built, so an
    // absurd ar_lags is refused without allocating.
    let n_coef = 1u128 + ar_lags as u128 + k as u128;
    if ar_lags >= n || ((n - ar_lags) as u128) <= n_coef {
        return Err(err(format!(
            "ar_lags = {ar_lags} with K = {k} high-frequency column(s) leaves {} row(s) to \
             fit 1 + ar_lags + K = {n_coef} coefficients; ADL-MIDAS needs n - ar_lags > \
             1 + ar_lags + K (n = {n}). Lower ar_lags (or use fewer hf_lags columns), or \
             supply a longer sample",
            n.saturating_sub(ar_lags)
        )));
    }
    let nobs = n - ar_lags;
    let cells = nobs as u128 * n_coef;
    if cells > MAX_DESIGN_CELLS {
        return Err(err(format!(
            "ar_lags = {ar_lags} with K = {k} high-frequency column(s) builds an nobs x \
             (1 + ar_lags + K) = {nobs} x {n_coef} design of {cells} cells, beyond the 2^26 \
             (512 MiB) accepted. Lower ar_lags (or use fewer hf_lags columns), or fit a \
             shorter sample"
        )));
    }
    let (se, resolved) = match se_type {
        "nonrobust" => {
            if let Some(l) = maxlags {
                return Err(err(format!(
                    "maxlags = {l} has no effect under se_type=\"nonrobust\" (it is the HAC \
                     bandwidth); drop it, or pass se_type=\"hac\""
                )));
            }
            (tsecon_midas::SeType::NonRobust, None)
        }
        "hac" => {
            let l = maxlags.unwrap_or_else(|| tsecon_hac::newey_west_maxlags(nobs));
            (
                tsecon_midas::SeType::Hac {
                    kernel: tsecon_hac::Kernel::Bartlett,
                    bandwidth: l as f64,
                    use_correction: true,
                },
                Some(l),
            )
        }
        other => {
            return Err(err(format!(
                "unknown se_type {other:?}; expected \"hac\" or \"nonrobust\""
            )))
        }
    };

    // Cost guard: the OLS fit and sandwich plus, under HAC, one pass over
    // the scores per lag up to min(maxlags, nobs - 1) (the Bartlett weight
    // is zero past the bandwidth and the crate stops there).
    let per_lag = nobs as u128 * n_coef * n_coef;
    let lags_eff = resolved.map_or(0, |l| (l as u128).min(nobs as u128 - 1));
    let work = per_lag * (lags_eff + 2);
    if work > WORK_BUDGET {
        if maxlags.is_some() && per_lag * 2 <= WORK_BUDGET {
            return Err(err(format!(
                "maxlags = {} over nobs = {nobs} fitted rows and k = {n_coef} coefficients \
                 costs about nobs * k^2 * (min(maxlags, nobs - 1) + 2) = {work:.2e} work \
                 units, beyond the budget of 2^33 (about 20 s). Pass maxlags <= {} (the \
                 default Newey-West rule gives {})",
                maxlags.unwrap_or(0),
                WORK_BUDGET / per_lag - 2,
                tsecon_hac::newey_west_maxlags(nobs),
                work = work as f64
            )));
        }
        return Err(err(format!(
            "ar_lags = {ar_lags} with K = {k} high-frequency column(s) gives k = 1 + ar_lags + \
             K = {n_coef} coefficients over nobs = {nobs} fitted rows: the OLS fit and its \
             {se_type} covariance cost about nobs * k^2 * (lags + 2) = {work:.2e} work units, \
             beyond the budget of 2^33 (about 20 s). Lower ar_lags (or use fewer hf_lags \
             columns), or fit a shorter sample",
            work = work as f64
        )));
    }

    let p = ar_lags;
    let y_eff = yv[p..].to_vec();
    let y_lags: Vec<Vec<f64>> = (1..=p).map(|lag| yv[p - lag..n - lag].to_vec()).collect();
    let hf: Vec<Vec<f64>> = (0..k)
        .map(|j| a.column(j).iter().skip(p).copied().collect())
        .collect();
    let span = |stem: &str, last: usize| {
        if last == 1 {
            format!("{stem}1")
        } else {
            format!("{stem}1..{stem}{last}")
        }
    };
    let r = tsecon_midas::adl_midas(&y_eff, &y_lags, &hf, se).map_err(|e| match e {
        // The crate's OLS error reads "U-MIDAS OLS fit failed" (the variant
        // is shared with umidas) and names no argument; say which design
        // failed and which inputs make it collinear.
        tsecon_midas::MidasError::Ols(e) => err(format!(
            "the OLS fit of y[ar_lags:] on [const, {}, {}] failed: {e}. \
             A constant y makes its own lags collinear with the constant, and a constant \
             or duplicated hf_lags column does the same; drop the redundant hf_lags \
             column(s), or lower ar_lags",
            span("y.L", p),
            span("hf", k)
        )),
        e => err(e.to_string()),
    })?;

    let mut names = Vec::with_capacity(1 + p + k);
    names.push("const".to_owned());
    names.extend((1..=p).map(|i| format!("y.L{i}")));
    names.extend((1..=k).map(|j| format!("hf{j}")));

    let d = PyDict::new(py);
    d.set_item("params", r.params.into_pyarray(py))?;
    d.set_item("param_names", names)?;
    d.set_item("bse", r.bse.into_pyarray(py))?;
    d.set_item("tvalues", r.tvalues.into_pyarray(py))?;
    d.set_item("rsquared", r.rsquared)?;
    d.set_item("nobs", r.nobs)?;
    d.set_item("residuals", r.residuals.into_pyarray(py))?;
    d.set_item("first_index", p)?;
    d.set_item("se_type", se_type)?;
    d.set_item("maxlags", resolved)?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adl_midas, m)?)?;
    Ok(())
}
