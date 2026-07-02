//! PyO3 bindings (Phase 1.5). Feature-gated behind `python`; absent from plain
//! `cargo build`/`test`. Exposes the denoise mutation operator to the Python
//! SR engine so it can be used as a GA mutation step.
//!
//! Built via `maturin develop`; importable as `from fuller import denoise`.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::extract::{
    denoise_assuming as denoise_core, denoise_candidates_assuming as denoise_candidates_core,
    eclass_extract_hff as eclass_extract_hff_core,
    eclass_extract_hff_instrumented as eclass_extract_hff_instrumented_core,
    eclass_variants as eclass_variants_core, EclassFamily,
};
use crate::karva::{
    karva_to_terms, terms_to_karva_sized, FunctionSpec, PsetSpec, Token,
};
use crate::parity::{proves_equal_assuming, Family};

/// Run a pure-Rust core computation with the GIL RELEASED and panics converted
/// to a normal `Err`.
///
/// * GIL: every core call here (egglog saturation, extraction, per-row eval)
///   can run for seconds; holding the GIL through it stalls every other
///   Python thread and signal handling. Only plain owned data crosses the
///   boundary, so releasing is safe.
/// * Panics: PyO3 converts a Rust panic into `pyo3_runtime.PanicException`,
///   which subclasses `BaseException` — it sails past the engine's
///   `except Exception` guards and can kill the run. egglog is not panic-free
///   on pathological programs, so catch the unwind and surface it as an
///   ordinary error string; each entry point then takes its normal error path
///   (return-unchanged or ValueError).
fn run_core<T: Send>(
    py: Python<'_>,
    f: impl FnOnce() -> Result<T, String> + Send,
) -> Result<T, String> {
    py.allow_threads(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|p| {
            let msg = if let Some(s) = p.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = p.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic payload".to_string()
            };
            Err(format!("internal panic (caught): {msg}"))
        })
    })
}

/// Denoise a `Math` expression against training data.
///
/// Args:
///   expr: egglog `Math` surface-syntax string, e.g.
///         `(Add (Mul (Var "x") (Num 1.0)) (Mul (Num 0.0) (Var "y")))`.
///   rows: list of dicts mapping variable name -> value (one per data row).
///   tolerance: max relative R^2 loss vs the input's behaviour (default 1e-3).
///   k_variants: how many smallest equivalent forms to consider (default 64).
///   positive_vars: variable names the CALLER knows are > 0 (e.g. from its
///       var_ranges) — unlocks guarded rewrites (Abs-shed, div-cancellation,
///       Pow2(Sqrt)). Sound: never assumed, only asserted by you.
///   nonzero_vars: variable names known != 0 (unlocks div/inv cancellation).
///
/// Returns a dict: {"expr": str, "cost": int, "changed": bool}. The chosen
/// (possibly smaller) equivalent expression, its structural cost, and whether
/// it shrank. Never raises on a normal un-simplifiable input — it returns the
/// input unchanged. Raises ValueError only on malformed egglog input or an
/// internal engine failure (Rust panics are caught and surfaced as ValueError,
/// never as a BaseException-derived PanicException).
#[pyfunction]
#[pyo3(signature = (expr, rows, tolerance = 1e-3, k_variants = 64,
                    positive_vars = vec![], nonzero_vars = vec![]))]
fn denoise(
    py: Python<'_>,
    expr: &str,
    rows: Vec<std::collections::HashMap<String, f64>>,
    tolerance: f64,
    k_variants: usize,
    positive_vars: Vec<String>,
    nonzero_vars: Vec<String>,
) -> PyResult<Py<PyDict>> {
    // Convert the Python rows (list of name->value dicts) into the core's
    // row format (Vec of (name, value) pairs).
    let core_rows: Vec<Vec<(String, f64)>> = rows
        .into_iter()
        .map(|m| m.into_iter().collect())
        .collect();

    let result = run_core(py, || {
        denoise_core(expr, &core_rows, tolerance, k_variants, &positive_vars, &nonzero_vars)
    })
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
                    tolerance = 1e-3, k_variants = 64, rng_seed = 0,
                    target_head_length = None,
                    positive_vars = vec![], nonzero_vars = vec![]))]
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
    target_head_length: Option<usize>,
    positive_vars: Vec<String>,
    nonzero_vars: Vec<String>,
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

    let denoised = match run_core(py, || {
        denoise_core(&math, &core_rows, tolerance, k_variants, &positive_vars, &nonzero_vars)
    }) {
        Ok(d) if d.changed => d,
        _ => return unchanged(py),
    };

    // Math -> karva. A denoise can succeed yet produce a constructor whose
    // semantic_id has no token in the caller's pset (e.g. result is Abs but the
    // pset has no abs op). That is NOT "nothing to simplify" — surface it so the
    // caller knows the result was inexpressible rather than silently dropping a
    // real simplification. The chromosome is still returned unchanged (safe),
    // but flagged.
    let (new_head, new_tail, oversized) =
        match terms_to_karva_sized(&denoised.expr, &pset, rng_seed, target_head_length) {
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

    // The denoised form's natural head exceeds the chromosome's head_length.
    // Truncating would change the term, so refuse: return unchanged + flagged.
    if oversized {
        let out = PyDict::new_bound(py);
        out.set_item("head", tokens_to_py(py, &head_toks)?)?;
        out.set_item("tail", tokens_to_py(py, &tail_toks)?)?;
        out.set_item("changed", false)?;
        out.set_item("expr", py.None())?;
        out.set_item("oversized", true)?;
        return Ok(out.into());
    }

    let out = PyDict::new_bound(py);
    out.set_item("head", tokens_to_py(py, &new_head)?)?;
    out.set_item("tail", tokens_to_py(py, &new_tail)?)?;
    out.set_item("changed", true)?;
    out.set_item("expr", denoised.expr)?;
    Ok(out.into())
}

/// Like `denoise_karva` but returns ALL candidates (equivalent forms +
/// pruned variants) — the engine scores them all via HFF and picks the
/// winner. No internal accept/reject: fuller proposes, HFF disposes.
///
/// Returns a list of dicts, each shaped exactly like a `denoise_karva`
/// success result:
///   {"head": [...], "tail": [...], "expr": Math str, "cost": int,
///    "is_original": bool, "oversized": bool, "inexpressible": optional str}
///
/// Engine usage (sketch):
///   cands = denoise_karva_candidates(head, tail, vars, fns, rnc, rows,
///                                      target_head_length=gene.head_length)
///   # Score each via _compute_raw_metrics + HFF; keep the lowest-angle one.
#[pyfunction]
#[pyo3(signature = (head, tail, variables, functions, rnc_values, rows,
                    k_variants = 64, rng_seed = 0,
                    target_head_length = None,
                    positive_vars = vec![], nonzero_vars = vec![]))]
fn denoise_karva_candidates(
    py: Python<'_>,
    head: Vec<PyToken>,
    tail: Vec<PyToken>,
    variables: Vec<String>,
    functions: HashMap<String, (String, usize)>,
    rnc_values: Vec<f64>,
    rows: Vec<HashMap<String, f64>>,
    k_variants: usize,
    rng_seed: u64,
    target_head_length: Option<usize>,
    positive_vars: Vec<String>,
    nonzero_vars: Vec<String>,
) -> PyResult<Vec<Py<PyDict>>> {
    let pset = build_pset(variables, functions, rnc_values);
    let head_toks = build_tokens(py, head)?;
    let tail_toks = build_tokens(py, tail)?;

    // karva -> Math. If it can't be encoded, return just the original.
    let math = match karva_to_terms(&head_toks, &tail_toks, &pset) {
        Ok(m) => m,
        Err(_) => {
            // Return single "original" candidate so caller has a uniform list.
            let d = PyDict::new_bound(py);
            d.set_item("head", tokens_to_py(py, &head_toks)?)?;
            d.set_item("tail", tokens_to_py(py, &tail_toks)?)?;
            d.set_item("expr", py.None())?;
            d.set_item("cost", 0u64)?;
            d.set_item("is_original", true)?;
            d.set_item("oversized", false)?;
            return Ok(vec![d.into()]);
        }
    };

    let core_rows: Vec<Vec<(String, f64)>> =
        rows.into_iter().map(|m| m.into_iter().collect()).collect();

    // On any core failure, return just the original candidate — same contract
    // as the un-encodable-karva path above (and as `denoise_karva`, which
    // returns unchanged). Raising here made this the ONE karva entry point
    // that could throw on an engine-internal failure.
    let candidates = match run_core(py, || {
        denoise_candidates_core(&math, &core_rows, k_variants, &positive_vars, &nonzero_vars)
    }) {
        Ok(c) => c,
        Err(_) => {
            let d = PyDict::new_bound(py);
            d.set_item("head", tokens_to_py(py, &head_toks)?)?;
            d.set_item("tail", tokens_to_py(py, &tail_toks)?)?;
            d.set_item("expr", py.None())?;
            d.set_item("cost", 0u64)?;
            d.set_item("is_original", true)?;
            d.set_item("oversized", false)?;
            return Ok(vec![d.into()]);
        }
    };

    let mut out: Vec<Py<PyDict>> = Vec::with_capacity(candidates.len());
    for c in candidates {
        let d = PyDict::new_bound(py);
        match terms_to_karva_sized(&c.expr, &pset, rng_seed, target_head_length) {
            Ok((new_head, new_tail, oversized)) => {
                if oversized {
                    // Skip oversized candidates — caller can't use them.
                    continue;
                }
                d.set_item("head", tokens_to_py(py, &new_head)?)?;
                d.set_item("tail", tokens_to_py(py, &new_tail)?)?;
                d.set_item("expr", c.expr)?;
                d.set_item("cost", c.cost)?;
                d.set_item("is_original", c.is_original)?;
                d.set_item("oversized", false)?;
            }
            Err(why) => {
                // Pset can't name this candidate — surface for diagnostics
                // and skip the karva fields (head/tail are None; callers must
                // check "inexpressible" — or head is None — before grafting).
                // Keys mirror the Ok branch so every dict has the same shape.
                d.set_item("head", py.None())?;
                d.set_item("tail", py.None())?;
                d.set_item("expr", c.expr)?;
                d.set_item("cost", c.cost)?;
                d.set_item("is_original", c.is_original)?;
                d.set_item("oversized", false)?;
                d.set_item("inexpressible", why)?;
            }
        }
        out.push(d.into());
    }
    Ok(out)
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
    let cands = run_core(py, || crate::physics::generate(expr, &paired_groups, n, seed))
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
                    n = 10, seed = 0, target_head_length = None))]
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
    target_head_length: Option<usize>,
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

    let cands = match run_core(py, || crate::physics::generate(&math, &paired_groups, n, seed)) {
        Ok(c) => c,
        // "Never raises on a normal gene": a core failure yields no candidates.
        Err(_) => return Ok(Vec::new()),
    };

    // Convert each candidate Math back to a karva chromosome. Skip any that the
    // caller's pset can't name (same contract as denoise_karva's inexpressible
    // path — we don't hand back chromosomes they can't decode).
    let mut out = Vec::new();
    for c in cands {
        let (cand_head, cand_tail, oversized) =
            match terms_to_karva_sized(&c.expr, &pset, seed, target_head_length) {
                Ok(ht) => ht,
                Err(_) => continue,
            };
        // A candidate whose natural head exceeds the chromosome's head_length
        // can't be grafted back without breaking GEP's uniform-head rule, and
        // truncation would change the form — so drop it (same skip contract as
        // an inexpressible candidate above).
        if oversized {
            continue;
        }
        let d = PyDict::new_bound(py);
        d.set_item("head", tokens_to_py(py, &cand_head)?)?;
        d.set_item("tail", tokens_to_py(py, &cand_tail)?)?;
        d.set_item("rule", c.rule)?;
        d.set_item("speculative", c.speculative)?;
        out.push(d.into());
    }
    Ok(out)
}

/// `snap_karva` — egglog-backed constant snapping (CR_snap_karva).
///
/// Karva in, **karva out** — same symmetry as denoise_karva / physics_mutate_karva.
/// Converts the chromosome to Math, proliferates constant-substituted variants
/// where a numeric atom matches a known constant (or composition) in the
/// generated lattice — VERIFIED in the e-graph, not merely numerically close —
/// then converts each back to karva.
///
/// Snapped forms introduce constant terminals (`pi`, `G`, …). Rather than make
/// the caller guess/register those, snap_karva **registers them itself** (as
/// pset terminals named by the constant) and returns, per candidate, both the
/// karva AND the mini-pset of constants it used, so the engine merges that and
/// decodes directly. (Resolves the karva-symmetry pushback.)
///
/// Each candidate: {"head", "tail", "expr" (Math, for reference), "cost",
/// "is_original", "constants": [{"name","value"}]}. The caller scores each on
/// holdout (R²-guard) and picks the winner — snap proposes, the data disposes.
#[pyfunction]
#[pyo3(signature = (head, tail, variables, functions, rnc_values, k_variants = 16, rel_tol = 1e-3, rng_seed = 0, target_head_length = None))]
fn snap_karva(
    py: Python<'_>,
    head: Vec<PyToken>,
    tail: Vec<PyToken>,
    variables: Vec<String>,
    functions: HashMap<String, (String, usize)>,
    rnc_values: Vec<f64>,
    k_variants: usize,
    rel_tol: f64,
    rng_seed: u64,
    target_head_length: Option<usize>,
) -> PyResult<Vec<Py<PyDict>>> {
    let pset = build_pset(variables.clone(), functions.clone(), rnc_values.clone());
    let head_toks = build_tokens(py, head)?;
    let tail_toks = build_tokens(py, tail)?;
    let math = match karva_to_terms(&head_toks, &tail_toks, &pset) {
        Ok(m) => m,
        Err(_) => return Ok(Vec::new()),
    };
    let cands = run_core(py, || crate::snap_karva::snap_variants(&math, k_variants, rel_tol))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let cvals = crate::snap_karva::constant_values();

    let mut out = Vec::new();
    for c in cands {
        // Build a pset augmented with the constants this candidate uses, as
        // variables (so terms_to_karva renders them as Var terminals).
        let mut vars_aug: Vec<String> = variables.clone();
        let mut consts_list: Vec<(String, f64)> = Vec::new();
        for name in &c.constants_used {
            if let Some(&v) = cvals.get(name) {
                if !vars_aug.contains(name) {
                    vars_aug.push(name.clone());
                }
                consts_list.push((name.clone(), v));
            }
        }
        let pset_aug = build_pset(vars_aug, functions.clone(), rnc_values.clone());
        let d = PyDict::new_bound(py);
        d.set_item("oversized", false)?;
        match terms_to_karva_sized(&c.expr, &pset_aug, rng_seed, target_head_length) {
            Ok((cand_head, cand_tail, oversized)) => {
                d.set_item("head", tokens_to_py(py, &cand_head)?)?;
                d.set_item("tail", tokens_to_py(py, &cand_tail)?)?;
                d.set_item("inexpressible", py.None())?;
                // Natural head longer than the chromosome's head_length —
                // grafting it back would break GEP's uniform-head rule, and
                // truncation would change the snapped form. Flag so the caller
                // drops it (per CR_karva_target_head_length).
                d.set_item("oversized", oversized)?;
            }
            Err(why) => {
                // A snap that uses an op the caller's pset lacks (e.g. the
                // composed 1/(4π) form needs `inv`/`mul`). Do NOT silently drop
                // it — surface the Math + the reason so the caller knows which
                // op to add to its pset (or pre-register master_pset). This was
                // the bug: composed snaps need inv/mul/pow, and a mul-only pset
                // dropped every one, leaving just the original.
                d.set_item("head", py.None())?;
                d.set_item("tail", py.None())?;
                d.set_item("inexpressible", why)?;
            }
        }
        d.set_item("expr", &c.expr)?;
        d.set_item("cost", c.cost)?;
        d.set_item("is_original", c.cost == 0)?;
        let consts_py: Vec<(String, f64)> = consts_list;
        d.set_item("constants", consts_py)?;
        out.push(d.into());
    }
    Ok(out)
}

/// The representation DOWN-FLIP: replace every named-constant terminal
/// (`pi`, `G`, `sqrt2`, ... — Var tokens whose name is in
/// `master_constants()`) with its numeric literal, in head and tail. Inverse
/// of `snap_karva` (the up-flip); the pair lets the same structure compete in
/// the population in constants-form and numeric-form, with selection deciding
/// which representation recovers the law. Behaviour-preserving,
/// deterministic, never raises on token data.
///
/// Returns {"head": [...], "tail": [...], "changed": bool,
///          "replaced": [names]} — `changed=False` (empty `replaced`) means
/// the chromosome was already fully numeric; skip the no-op mutant.
#[pyfunction]
fn concretize_karva(
    py: Python<'_>,
    head: Vec<PyToken>,
    tail: Vec<PyToken>,
) -> PyResult<Py<PyDict>> {
    let head_toks = build_tokens(py, head)?;
    let tail_toks = build_tokens(py, tail)?;
    let (new_head, new_tail, replaced) = crate::snap_karva::concretize(&head_toks, &tail_toks);
    let out = PyDict::new_bound(py);
    out.set_item("head", tokens_to_py(py, &new_head)?)?;
    out.set_item("tail", tokens_to_py(py, &new_tail)?)?;
    out.set_item("changed", !replaced.is_empty())?;
    out.set_item("replaced", replaced)?;
    Ok(out.into())
}

/// The canonical constant ATOMS the engine pre-registers as pset terminals
/// once per fit (symmetry with `master_pset`; determinism across snap_karva
/// calls). Returns [(name, value)] — e.g. [("G", 6.674e-11), ("pi", 3.14159…)].
/// Composed forms (1/(4π), 2π) are NOT here — they appear in snap candidates as
/// expressions built from these atoms, not as their own terminals.
#[pyfunction]
fn master_constants() -> Vec<(String, f64)> {
    crate::snap_karva::master_constants()
}

/// The full constant LATTICE: every `(value, math_sexpr, label)` snap can
/// recognise — base constants AND their composed forms (pi/2, 1/(4*pi),
/// 1/sqrt(2*pi), 2/sqrt(pi), ...). Read-once, deterministic, no file path
/// (the lattice is embedded in the crate). Mirrors master_pset /
/// master_constants. The engine uses this to drive snap recognition (and to
/// retire sympy.nsimplify): for a fitted scalar `x`, find the lattice entry
/// whose `value` matches within tolerance and adopt its `math` form.
///
/// `math` is a `Math` s-expression over constant `Var`s + integer `Num`s,
/// e.g. `(Div (Num 1.0) (Sqrt (Mul (Num 2.0) (Var "pi"))))` for 1/sqrt(2*pi).
#[pyfunction]
fn master_lattice() -> Vec<(f64, String, String)> {
    crate::snap_karva::lattice()
        .into_iter()
        .map(|e| (e.value, e.math, e.label))
        .collect()
}

/// Prove two `Math` s-expressions equal by equality saturation: insert both,
/// run the `family` rules to a bounded fixpoint, and check whether they land in
/// the same e-class. This is the SOUND equivalence oracle — a `true` is a real
/// proof under the ruleset, a `false` means "not proven equal" (NOT "proven
/// unequal"). Bounded iteration count per family, so it cannot hang the way
/// `sympy.simplify` does on junk transcendental towers.
///
/// Semantics note: "equal" means equal as REAL-domain identities (the rules'
/// theory), not pointwise-identical under f64/NaN evaluation. E.g. `x - x` and
/// `0` are proven equal, yet at a row where `x` is NaN the first evaluates NaN
/// and the second 0. Where pointwise behaviour on data matters, that is
/// `denoise`'s R²-gate job, not this oracle's.
///
/// Args:
///   input, target: egglog `Math` surface-syntax strings (e.g.
///       `(Mul (Var "x") (Num 1.0))` and `(Var "x")`).
///   family: "algebra" (default), "rational", "trig", or "wide" — which ruleset
///       to run. "wide" adds commutativity/associativity/distributivity, so it
///       proves reordered/re-associated forms equal (e.g. `x+y == y+x`) that the
///       collapse-only families miss; it is iteration-capped for boundedness.
///   nonzero_vars: variable names to assume `!= 0`, unlocking guarded
///       cancellation (`x/x -> 1`) for SCALE-constant equivalence. Sound for the
///       scale question (a ratio is only defined where the denominator is
///       nonzero). Default empty.
///
/// Returns bool. Raises ValueError on malformed egglog input.
#[pyfunction]
#[pyo3(signature = (input, target, family = "algebra", nonzero_vars = vec![]))]
fn proves_equal(
    py: Python<'_>,
    input: &str,
    target: &str,
    family: &str,
    nonzero_vars: Vec<String>,
) -> PyResult<bool> {
    let fam = match family {
        "algebra" => Family::Algebra,
        "rational" => Family::Rational,
        "trig" => Family::Trig,
        "wide" => Family::Wide,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "family must be \"algebra\", \"rational\", \"trig\", or \"wide\", got {other:?}"
            )))
        }
    };
    run_core(py, || proves_equal_assuming(input, target, fam, &nonzero_vars))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// The MASTER pset: every `(semantic_id, arity)` any fuller mutation
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

// ---------------------------------------------------------------------------
// Brainfuck simplifier bindings
// ---------------------------------------------------------------------------

/// Simplify a Brainfuck source string using equality saturation.
///
/// Returns a dict: `{"source": str, "op_count": int, "changed": bool}`.
/// - `source`: the simplified BF source string (same as input if no rule fires).
/// - `op_count`: number of BF ops (`+-<>.,[]`) in the simplified source.
/// - `changed`: True if the simplified form is strictly shorter than the input.
///
/// Never raises on normal input — returns the input unchanged on any internal
/// error. Raises ValueError only on truly malformed input (syntax error after
/// removing comments).
#[pyfunction]
fn bf_simplify(py: Python<'_>, source: &str) -> PyResult<Py<pyo3::types::PyDict>> {
    let result = crate::bf::extract::bf_simplify(source)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new_bound(py);
    out.set_item("source", result.source)?;
    out.set_item("op_count", result.op_count)?;
    out.set_item("changed", result.changed)?;
    Ok(out.into())
}

/// Convert a BF source string to its egglog Prog s-expression.
///
/// Non-BF characters are ignored (they are comments). Returns an error on
/// unmatched brackets.
#[pyfunction]
fn bf_parse(source: &str) -> PyResult<String> {
    crate::bf::parse::parse_bf(source)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Convert an egglog Prog s-expression back to BF source text.
#[pyfunction]
fn bf_unparse(sexpr: &str) -> PyResult<String> {
    crate::bf::parse::unparse_bf(sexpr)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Enumerate the equivalence class of a karva chromosome under a FULL rule
/// family (algebra+powers+distribute, or algebra+powers+trig) — the wide
/// saturation the tournament figure needs, NOT the bounded denoise subset.
///
/// `family` is "algebra" or "trig" (distribute and trig explode co-saturated, so
/// one is chosen). `iters` bounds the run schedule so a divergent rule can't peg
/// the machine. Returns `[(cost, Math s-expression)]` for each distinct variant,
/// for the caller to score with the pattern-metric library + HFF.
#[pyfunction]
#[pyo3(signature = (head, tail, variables, functions, rnc_values, family = "algebra", k = 64, iters = 12))]
fn eclass_variants(
    py: Python<'_>,
    head: Vec<PyToken>,
    tail: Vec<PyToken>,
    variables: Vec<String>,
    functions: HashMap<String, (String, usize)>,
    rnc_values: Vec<f64>,
    family: &str,
    k: usize,
    iters: u32,
) -> PyResult<Vec<(u64, String)>> {
    let pset = build_pset(variables, functions, rnc_values);
    let head_toks = build_tokens(py, head)?;
    let tail_toks = build_tokens(py, tail)?;
    let math = karva_to_terms(&head_toks, &tail_toks, &pset)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let fam = match family {
        "trig" => EclassFamily::Trig,
        "algebra" => EclassFamily::Algebra,
        "wide" => EclassFamily::Wide,
        "structural" => EclassFamily::Structural,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "family must be \"algebra\", \"trig\", \"wide\", or \"structural\", got {other:?}"
            )))
        }
    };
    run_core(py, || eclass_variants_core(&math, fam, k, iters))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Enumerate the equivalence class and RANK it by the CDF-corrected
/// hyperspherical-fitness angle (the `/pattern/{measure}` tournament) instead of
/// egglog's scalar tree cost — the opt-in HFF extractor. The per-e-class winner
/// is chosen by the angular measure-vector ordering as the extraction walk
/// proceeds (see `crate::score`).
///
/// Same karva-in interface as `eclass_variants`. Returns
/// `[(angle_percentile, Math s-expression)]`, best (lowest percentile) first.
#[pyfunction]
#[pyo3(signature = (head, tail, variables, functions, rnc_values, family = "algebra", k = 64, iters = 12, exclude_measures = vec![]))]
fn eclass_extract_hff(
    py: Python<'_>,
    head: Vec<PyToken>,
    tail: Vec<PyToken>,
    variables: Vec<String>,
    functions: HashMap<String, (String, usize)>,
    rnc_values: Vec<f64>,
    family: &str,
    k: usize,
    iters: u32,
    exclude_measures: Vec<String>,
) -> PyResult<Vec<(f64, String)>> {
    let pset = build_pset(variables, functions, rnc_values);
    let head_toks = build_tokens(py, head)?;
    let tail_toks = build_tokens(py, tail)?;
    let math = karva_to_terms(&head_toks, &tail_toks, &pset)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let fam = match family {
        "trig" => EclassFamily::Trig,
        "algebra" => EclassFamily::Algebra,
        "wide" => EclassFamily::Wide,
        "structural" => EclassFamily::Structural,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "family must be \"algebra\", \"trig\", \"wide\", or \"structural\", got {other:?}"
            )))
        }
    };
    // exclude_measures (default []) down-selects the /pattern/{measure} library —
    // drop the named rules to test which measures matter. [] runs all of them.
    run_core(py, || eclass_extract_hff_core(&math, fam, k, iters, &exclude_measures))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// INSTRUMENTED e-class tournament — `eclass_extract_hff` plus measured
/// behaviour columns. Each equivalent form is RUN on `rows_train` and on
/// `rows_val` (held out) and compared against the input's own predictions;
/// the TrueNorth vector is `[form measures | domain_mismatch, disagreement
/// (train) | domain_mismatch, disagreement (val)]`, all [0,1], 0 best.
///
/// Algebraically-equal forms differ measurably in f64: rounding divergence
/// (catastrophic cancellation), introduced/removed NaN/inf on the data
/// distribution. The val columns stop the rewrite CHOICE from overfitting the
/// profiling rows: the winner is the form whose measured behaviour is
/// cleanest and stays clean off the profiled set. Candidates that fail to
/// evaluate rank last.
///
/// Same karva-in interface as `eclass_extract_hff`, with `rows_train` /
/// `rows_val` as lists of {var: value} dicts. Returns [(angle, Math s-expr)],
/// best first. If a behaviour change must NEVER win regardless of form
/// (exact-recovery pipelines), hard-filter the adopted result with `denoise`'s
/// R^2 gate — the angle trades objectives; it does not enforce vetoes.
#[pyfunction]
#[pyo3(signature = (head, tail, variables, functions, rnc_values, rows_train, rows_val,
                    family = "algebra", k = 64, iters = 12, exclude_measures = vec![]))]
fn eclass_extract_hff_instrumented(
    py: Python<'_>,
    head: Vec<PyToken>,
    tail: Vec<PyToken>,
    variables: Vec<String>,
    functions: HashMap<String, (String, usize)>,
    rnc_values: Vec<f64>,
    rows_train: Vec<HashMap<String, f64>>,
    rows_val: Vec<HashMap<String, f64>>,
    family: &str,
    k: usize,
    iters: u32,
    exclude_measures: Vec<String>,
) -> PyResult<Vec<(f64, String)>> {
    let pset = build_pset(variables, functions, rnc_values);
    let head_toks = build_tokens(py, head)?;
    let tail_toks = build_tokens(py, tail)?;
    let math = karva_to_terms(&head_toks, &tail_toks, &pset)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let fam = match family {
        "trig" => EclassFamily::Trig,
        "algebra" => EclassFamily::Algebra,
        "wide" => EclassFamily::Wide,
        "structural" => EclassFamily::Structural,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "family must be \"algebra\", \"trig\", \"wide\", or \"structural\", got {other:?}"
            )))
        }
    };
    let core_tr: Vec<Vec<(String, f64)>> =
        rows_train.into_iter().map(|m| m.into_iter().collect()).collect();
    let core_va: Vec<Vec<(String, f64)>> =
        rows_val.into_iter().map(|m| m.into_iter().collect()).collect();
    run_core(py, || {
        eclass_extract_hff_instrumented_core(
            &math,
            fam,
            k,
            iters,
            &exclude_measures,
            &core_tr,
            &core_va,
        )
    })
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// The native extension module. `module-name` in pyproject.toml is
/// `fuller._fuller`, so this initialises `_fuller`; the Python shim
/// re-exports from it.
#[pymodule]
fn _fuller(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(denoise, m)?)?;
    m.add_function(wrap_pyfunction!(denoise_karva, m)?)?;
    m.add_function(wrap_pyfunction!(denoise_karva_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(physics_mutate, m)?)?;
    m.add_function(wrap_pyfunction!(physics_mutate_karva, m)?)?;
    m.add_function(wrap_pyfunction!(proves_equal, m)?)?;
    m.add_function(wrap_pyfunction!(master_pset, m)?)?;
    m.add_function(wrap_pyfunction!(master_constants, m)?)?;
    m.add_function(wrap_pyfunction!(master_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(eclass_variants, m)?)?;
    m.add_function(wrap_pyfunction!(eclass_extract_hff, m)?)?;
    m.add_function(wrap_pyfunction!(snap_karva, m)?)?;
    m.add_function(wrap_pyfunction!(concretize_karva, m)?)?;
    m.add_function(wrap_pyfunction!(eclass_extract_hff_instrumented, m)?)?;
    // Brainfuck simplifier
    m.add_function(wrap_pyfunction!(bf_simplify, m)?)?;
    m.add_function(wrap_pyfunction!(bf_parse, m)?)?;
    m.add_function(wrap_pyfunction!(bf_unparse, m)?)?;
    Ok(())
}
