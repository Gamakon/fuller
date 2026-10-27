//! fuller — an egglog-based bidirectional AST hub for symbolic expression
//! rewriting.
//!
//! fuller drives egglog 2.0 (Rust -> egglog -> saturate -> extract -> Rust)
//! over a real-domain `Math` datatype (`expr`), with algebra/power rulesets
//! (`ruleset`), a real-domain evaluator (`eval`), and the data-aware `denoise`
//! mutation operator (`extract`). The `python` feature exposes `denoise` to
//! the Python SR engine via PyO3 (`from fuller import denoise`).

/// Hard depth cap for expression trees crossing the crate boundary.
///
/// Every recursive parse/render/eval walk in the crate is bounded by this, so
/// a pathologically deep gene surfaces as a normal `Err` (and the Python API
/// returns the input unchanged) instead of overflowing the stack — a stack
/// overflow is a SIGSEGV abort that no never-raise guard can catch.
///
/// SIZING — the cap only protects if the guarded recursion fits the stack
/// BEFORE the check trips: the walk pushes up to `MAX_EXPR_DEPTH` native
/// frames and bails on the next one. The budget must hold in the worst case:
/// DEBUG builds (unoptimised frames run 1.5–2KB for the parse/render fns) on
/// the smallest default stack in play (2MB: Rust test threads and rayon
/// workers). 256 × ~2KB ≈ 512KB ≤ 2MB/2 — 4x headroom. (The original 1024
/// aborted `cargo test` at default stack: 1024 debug frames overran 2MB
/// before the guard could return Err.) Semantically 256 is generous: decoded
/// depth is bounded by head length, and a 256-deep unary chain is wallpaper
/// no real gene reaches — deeper inputs degrade safely to Err/unchanged.
pub const MAX_EXPR_DEPTH: usize = 256;

pub mod bf;
pub mod calibration;
pub mod clamp;
pub mod chrom_score;
pub mod eval;
pub mod expr;
pub mod extract;
pub mod extract_dag;
pub mod homeotic;
pub mod infix;
pub mod population_dag;
pub mod geneframe;
pub mod tsr_kingdom;
pub mod futhark_kingdom;
pub mod futhark_shapes;
pub mod futhark_shapes_cost;
pub mod futhark_shapes_egraph;
pub mod futhark_shapes_evolve;
pub mod futhark_shapes_gpu;
pub mod futhark_shapes_interp;
pub mod futhark_interp;
pub mod futhark_exec;
pub mod futhark_egraph;
pub mod kingdom_contract;
pub mod typed_chromosome;
pub mod typed_gep;
pub mod regex_typed;
#[cfg(test)]
mod typed_gep_tests;
/// The documentation index (`docs/index.html`) is generated; this is the check
/// that keeps it so.
#[cfg(test)]
mod docs_index;
// GPU batch evaluator. The module (opcodes, batch layout, WGSL source) always
// compiles so its tests run on any machine; only the wgpu device code is
// behind `feature = "gpu"`.
pub mod gpu_eval;
#[cfg(feature = "gpu")]
pub mod gpu_io;
#[cfg(test)]
pub(crate) mod gpu_test_lock;
pub mod karva;
pub mod pattern_match;
pub mod lint;
pub mod parity;
pub mod srbench_equiv;
pub mod physics;
pub mod regex;
pub mod ruleset;
pub mod score;
pub mod snap;
pub mod snap_karva;
#[cfg(feature = "wgsl")]
pub mod wgsl;

#[cfg(feature = "python")]
mod python;
