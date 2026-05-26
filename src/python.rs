//! PyO3 bindings (Phase 1.5). Feature-gated behind `python`; absent from plain
//! `cargo build`/`test`. Exposes the denoise mutation operator to the Python
//! SR engine so it can be used as a GA mutation step.
//!
//! Built via `maturin develop`; importable as `from gamakAST import denoise`.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::extract::denoise as denoise_core;
use crate::karva::{karva_to_terms, terms_to_karva, FunctionSpec, PsetSpec, Token};

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

/// A karva token from Python, as a `(kind, value)` tuple:
///   ("func", "<token_name>") | ("var", "<name>") | ("num", <float>).
type PyToken = (String, PyObject);

/// Build a `PsetSpec` from Python pieces.
fn build_pset(
    variables: Vec<String>,
    functions: HashMap<String, (String, usize)>,
    rnc_values: Vec<f64>,
) -> PsetSpec {
    let functions = functions
        .into_iter()
        .map(|(name, (semantic_id, arity))| (name, FunctionSpec { semantic_id, arity }))
        .collect();
    PsetSpec { variables, functions, rnc_values }
}

/// Convert a list of `(kind, value)` tuples into `Vec<Token>`.
fn build_tokens(py: Python<'_>, raw: Vec<PyToken>) -> PyResult<Vec<Token>> {
    raw.into_iter()
        .map(|(kind, val)| match kind.as_str() {
            "func" => Ok(Token::Func(val.extract::<String>(py)?)),
            "var" => Ok(Token::Var(val.extract::<String>(py)?)),
            "num" => Ok(Token::Num(val.extract::<f64>(py)?)),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bad token kind {other:?} (want func/var/num)"
            ))),
        })
        .collect()
}

/// Denoise a karva chromosome directly: the SR engine's native interface.
///
/// Args:
///   head, tail: lists of (kind, value) tuples — kind in {"func","var","num"}.
///   variables: list of variable names in the pset.
///   functions: dict token_name -> (semantic_id, arity).
///   rnc_values: list of numeric constants.
///   rows: list of dicts var_name -> value (training data).
///   tolerance, k_variants, rng_seed: denoise + tail-padding controls.
///
/// Returns {"head": [...], "tail": [...], "changed": bool, "expr": str} where
/// head/tail are the denoised chromosome as (kind, value) tuples. On any
/// un-encodable input it returns the ORIGINAL head/tail unchanged (never
/// raises except on malformed pset/token data).
#[pyfunction]
#[pyo3(signature = (head, tail, variables, functions, rnc_values, rows,
                    tolerance = 1e-3, k_variants = 64, rng_seed = 0))]
#[allow(clippy::too_many_arguments)]
fn denoise_karva(
    py: Python<'_>,
    head: Vec<PyToken>,
    tail: Vec<PyToken>,
    variables: Vec<String>,
    functions: HashMap<String, (String, usize)>,
    rnc_values: Vec<f64>,
    rows: Vec<HashMap<String, f64>>,
    tolerance: f64,
    k_variants: usize,
    rng_seed: u64,
) -> PyResult<Py<PyDict>> {
    let pset = build_pset(variables, functions, rnc_values);
    let head_toks = build_tokens(py, head)?;
    let tail_toks = build_tokens(py, tail)?;

    // Helper: the "unchanged" result, from the original token vecs.
    let unchanged = |py: Python<'_>| -> PyResult<Py<PyDict>> {
        let out = PyDict::new_bound(py);
        out.set_item("head", tokens_to_py(py, &head_toks)?)?;
        out.set_item("tail", tokens_to_py(py, &tail_toks)?)?;
        out.set_item("changed", false)?;
        out.set_item("expr", py.None())?;
        Ok(out.into())
    };

    // karva -> Math. If it can't be encoded, return the original unchanged.
    let math = match karva_to_terms(&head_toks, &tail_toks, &pset) {
        Ok(m) => m,
        Err(_) => return unchanged(py),
    };

    let core_rows: Vec<Vec<(String, f64)>> =
        rows.into_iter().map(|m| m.into_iter().collect()).collect();

    let denoised = match denoise_core(&math, &core_rows, tolerance, k_variants) {
        Ok(d) if d.changed => d,
        _ => return unchanged(py),
    };

    // Math -> karva. A denoise can succeed yet produce a constructor whose
    // semantic_id has no token in the caller's pset (e.g. result is Abs but the
    // pset has no abs op). That is NOT "nothing to simplify" — surface it so the
    // caller knows the result was inexpressible rather than silently dropping a
    // real simplification. The chromosome is still returned unchanged (safe),
    // but flagged.
    let (new_head, new_tail) = match terms_to_karva(&denoised.expr, &pset, rng_seed) {
        Ok(ht) => ht,
        Err(why) => {
            let out = PyDict::new_bound(py);
            out.set_item("head", tokens_to_py(py, &head_toks)?)?;
            out.set_item("tail", tokens_to_py(py, &tail_toks)?)?;
            out.set_item("changed", false)?;
            out.set_item("expr", py.None())?;
            // Distinguishing signal: a simpler form existed but the pset can't
            // name it. The caller may want to add the missing op to its pset.
            out.set_item("inexpressible", format!("{} ({})", denoised.expr, why))?;
            return Ok(out.into());
        }
    };

    let out = PyDict::new_bound(py);
    out.set_item("head", tokens_to_py(py, &new_head)?)?;
    out.set_item("tail", tokens_to_py(py, &new_tail)?)?;
    out.set_item("changed", true)?;
    out.set_item("expr", denoised.expr)?;
    Ok(out.into())
}

/// Render `Vec<Token>` back to a list of (kind, value) tuples for Python.
fn tokens_to_py(py: Python<'_>, toks: &[Token]) -> PyResult<Vec<(String, PyObject)>> {
    toks.iter()
        .map(|t| match t {
            Token::Func(n) => Ok(("func".to_string(), n.into_py(py))),
            Token::Var(n) => Ok(("var".to_string(), n.into_py(py))),
            Token::Num(v) => Ok(("num".to_string(), v.into_py(py))),
        })
        .collect()
}

/// Physics-prior mutation GENERATOR (NOT denoise — these edits change the
/// function). One gene in -> a proliferation of physics-shaped candidate genes
/// out. Pure generation: NO data, NO evaluation, NO scoring — the caller
/// selects with HFF, and MUST gate `speculative=True` candidates on the
/// extrapolation objective (not holdout).
///
/// * `expr` — a Math s-expression.
/// * `paired_groups` — coordinate axes, e.g. [["x1","x2"],["y1","y2"]].
/// * `n` — max candidates to RETURN (>=1, default 10). All are generated
///   internally; if more than `n` exist, a uniform random `n` are sampled.
/// * `seed` — makes the random sample reproducible.
///
/// Each candidate is `{"expr", "rule", "speculative"}`.
#[pyfunction]
#[pyo3(signature = (expr, paired_groups, n = 10, seed = 0))]
fn physics_mutate(
    py: Python<'_>,
    expr: &str,
    paired_groups: Vec<Vec<String>>,
    n: usize,
    seed: u64,
) -> PyResult<Vec<Py<PyDict>>> {
    let cands = crate::physics::generate(expr, &paired_groups, n, seed)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    cands
        .into_iter()
        .map(|c| {
            let d = PyDict::new_bound(py);
            d.set_item("expr", c.expr)?;
            d.set_item("rule", c.rule)?;
            d.set_item("speculative", c.speculative)?;
            Ok(d.into())
        })
        .collect()
}

/// Physics-prior mutation GENERATOR, karva in / karva out — the interface the
/// SR engine uses. The engine selects a chromosome (by its own probability) and
/// passes its karva head/tail; this returns physics-shaped candidate
/// CHROMOSOMES (not Math strings).
///
/// One gene in -> a proliferation of physics-shaped candidate genes out. Pure
/// generation: NO data, NO evaluation, NO scoring. The caller selects with HFF
/// and MUST gate `speculative=True` candidates on the extrapolation objective.
///
/// Args (head/tail/variables/functions/rnc_values exactly as `denoise_karva`):
///   head, tail   — lists of ("func"|"var"|"num", value) tuples.
///   variables    — pset variable names.
///   functions    — dict token_name -> (semantic_id, arity).
///   rnc_values   — numeric constants (used when re-padding candidate tails).
///   paired_groups— coordinate axes, e.g. [["x1","x2"],["y1","y2"]].
///   n, seed      — max candidates returned (>=1) + reproducible sample.
///
/// Returns a list of {"head": [...], "tail": [...], "rule": str,
/// "speculative": bool}. Candidates whose form cannot be expressed in the
/// caller's pset (no token for a produced op) are skipped — so you only ever
/// get back chromosomes you can actually decode. Never raises on a normal gene.
#[pyfunction]
#[pyo3(signature = (head, tail, variables, functions, rnc_values, paired_groups,
                    n = 10, seed = 0))]
#[allow(clippy::too_many_arguments)]
fn physics_mutate_karva(
    py: Python<'_>,
    head: Vec<PyToken>,
    tail: Vec<PyToken>,
    variables: Vec<String>,
    functions: HashMap<String, (String, usize)>,
    rnc_values: Vec<f64>,
    paired_groups: Vec<Vec<String>>,
    n: usize,
    seed: u64,
) -> PyResult<Vec<Py<PyDict>>> {
    let pset = build_pset(variables, functions, rnc_values);
    let head_toks = build_tokens(py, head)?;
    let tail_toks = build_tokens(py, tail)?;

    // karva -> Math. If the input gene can't be encoded, there is nothing to
    // mutate; return an empty list (no candidates), not an error.
    let math = match karva_to_terms(&head_toks, &tail_toks, &pset) {
        Ok(m) => m,
        Err(_) => return Ok(Vec::new()),
    };

    let cands = crate::physics::generate(&math, &paired_groups, n, seed)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    // Convert each candidate Math back to a karva chromosome. Skip any that the
    // caller's pset can't name (same contract as denoise_karva's inexpressible
    // path — we don't hand back chromosomes they can't decode).
    let mut out = Vec::new();
    for c in cands {
        let (cand_head, cand_tail) = match terms_to_karva(&c.expr, &pset, seed) {
            Ok(ht) => ht,
            Err(_) => continue,
        };
        let d = PyDict::new_bound(py);
        d.set_item("head", tokens_to_py(py, &cand_head)?)?;
        d.set_item("tail", tokens_to_py(py, &cand_tail)?)?;
        d.set_item("rule", c.rule)?;
        d.set_item("speculative", c.speculative)?;
        out.push(d.into());
    }
    Ok(out)
}

/// The MASTER pset: every `(semantic_id, arity)` any gamakAST mutation
/// (denoise or physics-prior) can emit. The SR engine seeds its pset with one
/// token per entry UP FRONT, so every returned candidate is always expressible
/// — no candidate is ever dropped for lack of a token.
///
/// Returns a list of `(semantic_id, arity)` tuples. The engine maps each to its
/// own token name when building the `functions` dict it passes back in.
#[pyfunction]
fn master_pset() -> Vec<(String, usize)> {
    crate::karva::master_pset()
        .into_iter()
        .map(|(s, a)| (s.to_string(), a))
        .collect()
}

/// The native extension module. `module-name` in pyproject.toml is
/// `gamakAST._gamakast`, so this initialises `_gamakast`; the Python shim
/// re-exports from it.
#[pymodule]
fn _gamakast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(denoise, m)?)?;
    m.add_function(wrap_pyfunction!(denoise_karva, m)?)?;
    m.add_function(wrap_pyfunction!(physics_mutate, m)?)?;
    m.add_function(wrap_pyfunction!(physics_mutate_karva, m)?)?;
    m.add_function(wrap_pyfunction!(master_pset, m)?)?;
    Ok(())
}
