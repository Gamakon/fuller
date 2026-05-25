//! PyO3 bindings (Phase 1.5). Feature-gated behind `python`; absent from plain
//! `cargo build`/`test`. Exposes the denoise mutation operator to the Python
//! SR engine so it can be used as a GA mutation step.
//!
//! Built via `maturin develop`; importable as `from gamakAST import denoise`.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::extract::denoise as denoise_core;

/// Denoise a `Math` expression against training data.
///
/// Args:
///   expr: egglog `Math` surface-syntax string, e.g.
///         `(Add (Mul (Var "x") (Num 1.0)) (Mul (Num 0.0) (Var "y")))`.
///   rows: list of dicts mapping variable name -> value (one per data row).
///   tolerance: max relative R^2 loss vs the input's behaviour (default 1e-3).
///   k_variants: how many smallest equivalent forms to consider (default 64).
///
/// Returns a dict: {"expr": str, "cost": int, "changed": bool}. The chosen
/// (possibly smaller) equivalent expression, its structural cost, and whether
/// it shrank. Never raises on a normal un-simplifiable input — it returns the
/// input unchanged. Raises ValueError only on malformed egglog input.
#[pyfunction]
#[pyo3(signature = (expr, rows, tolerance = 1e-3, k_variants = 64))]
fn denoise(
    py: Python<'_>,
    expr: &str,
    rows: Vec<std::collections::HashMap<String, f64>>,
    tolerance: f64,
    k_variants: usize,
) -> PyResult<Py<PyDict>> {
    // Convert the Python rows (list of name->value dicts) into the core's
    // row format (Vec of (name, value) pairs).
    let core_rows: Vec<Vec<(String, f64)>> = rows
        .into_iter()
        .map(|m| m.into_iter().collect())
        .collect();

    let result = denoise_core(expr, &core_rows, tolerance, k_variants)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let out = PyDict::new_bound(py);
    out.set_item("expr", result.expr)?;
    out.set_item("cost", result.cost)?;
    out.set_item("changed", result.changed)?;
    Ok(out.into())
}

/// The native extension module. `module-name` in pyproject.toml is
/// `gamakAST._gamakast`, so this initialises `_gamakast`; the Python shim
/// re-exports from it.
#[pymodule]
fn _gamakast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(denoise, m)?)?;
    Ok(())
}
