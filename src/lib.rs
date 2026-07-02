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
pub mod eval;
pub mod expr;
pub mod extract;
pub mod geneframe;
pub mod karva;
pub mod parity;
pub mod physics;
pub mod ruleset;
pub mod score;
pub mod snap;
pub mod snap_karva;

#[cfg(feature = "python")]
mod python;
