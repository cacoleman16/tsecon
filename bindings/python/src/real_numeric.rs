//! The real-numeric element gate shared by the array arguments that take
//! real numbers: `impact` (`structural_fevd`, `historical_decomposition`),
//! `dfm_smooth`'s `loadings` / `factor_ar` / `factor_cov` and
//! `var_conditional_forecast`'s `conditions` / `condition_sd`.
//!
//! Every one of them ends in a float extraction that would accept a complex
//! number by calling its `__float__` — NumPy's drops the imaginary part with
//! a mere `ComplexWarning` — and compute on the real part as if nothing had
//! happened. The gate runs first and says why the argument is not real.

use pyo3::prelude::*;
use pyo3::types::PyComplex;

use crate::var_cf::short_repr;

/// Why a complex value is refused rather than read.
const COMPLEX: &str = "a complex value is not truncated to its real part";

/// The NumPy element kind (`dtype.kind`) of `obj`, when it has a dtype.
fn dtype_kind(obj: &Bound<'_, PyAny>) -> Option<(String, String)> {
    let dtype = obj.getattr("dtype").ok()?;
    let kind: String = dtype.getattr("kind").ok()?.extract().ok()?;
    let name = dtype
        .str()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| kind.clone());
    Some((kind, name))
}

/// True for a Python `complex` (NumPy's `complex128` subclasses it) or a
/// NumPy complex scalar of any width — not for a complex array of rank one
/// or more, which the row scan names by its dtype instead.
fn is_complex_scalar(cell: &Bound<'_, PyAny>) -> bool {
    if cell.is_instance_of::<PyComplex>() {
        return true;
    }
    let rank = cell
        .getattr("ndim")
        .and_then(|d| d.extract::<usize>())
        .unwrap_or(0);
    rank == 0 && matches!(dtype_kind(cell), Some((k, _)) if k == "c")
}

/// Why `obj` is not an array of real numbers, or `None` when it may be one
/// (the extraction that follows then has the last word, e.g. on a ragged
/// list or a string cell).
///
/// * an object with a NumPy dtype: kinds `b`, `i`, `u` and `f` pass; `c`
///   (complex) is refused; `O` (object) is scanned cell by cell when
///   `allow_object` (a nested list with `None` or `Decimal` entries arrives
///   that way), refused otherwise; every other kind (strings, datetimes,
///   bytes, records) is refused;
/// * a list or tuple of rows: scanned for complex cells, two levels deep
///   (a row that is an array is judged by its dtype).
///
/// The reason reads as the tail of `"{name} must hold real numbers, ..."`.
pub(crate) fn non_real_reason(
    obj: &Bound<'_, PyAny>,
    name: &str,
    allow_object: bool,
) -> PyResult<Option<String>> {
    if let Some((kind, dtype)) = dtype_kind(obj) {
        return Ok(match kind.as_str() {
            "b" | "i" | "u" | "f" => None,
            "c" => Some(format!("got dtype {dtype}: {COMPLEX}")),
            "O" if allow_object => scan_rows(obj, name)?,
            _ => Some(format!(
                "got dtype {dtype}: strings, datetimes or None entries are not read as numbers"
            )),
        });
    }
    if obj.is_instance_of::<pyo3::types::PyList>() || obj.is_instance_of::<pyo3::types::PyTuple>() {
        return scan_rows(obj, name);
    }
    Ok(None)
}

/// The first complex cell of a sequence of rows, named by its position.
fn scan_rows(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    let Ok(rows) = obj.try_iter() else {
        return Ok(None);
    };
    for (h, row) in rows.enumerate() {
        let row = row?;
        if is_complex_scalar(&row) {
            return Ok(Some(complex_cell(&format!("{name}[{h}]"), &row)));
        }
        if let Some((kind, dtype)) = dtype_kind(&row) {
            if kind == "c" {
                return Ok(Some(format!("{name}[{h}] has dtype {dtype}: {COMPLEX}")));
            }
            if kind != "O" {
                continue;
            }
        } else if !(row.is_instance_of::<pyo3::types::PyList>()
            || row.is_instance_of::<pyo3::types::PyTuple>())
        {
            continue;
        }
        let Ok(cells) = row.try_iter() else {
            continue;
        };
        for (j, cell) in cells.enumerate() {
            let cell = cell?;
            if is_complex_scalar(&cell) {
                return Ok(Some(complex_cell(&format!("{name}[{h}][{j}]"), &cell)));
            }
        }
    }
    Ok(None)
}

fn complex_cell(at: &str, cell: &Bound<'_, PyAny>) -> String {
    let ty = cell
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "complex".to_string());
    format!("{at} = {} is of type {ty}: {COMPLEX}", short_repr(cell))
}
