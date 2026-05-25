//! gamakAST — an egglog-based bidirectional AST hub for symbolic expression
//! rewriting.
//!
//! gamakAST drives egglog 2.0 (Rust -> egglog -> saturate -> extract -> Rust)
//! over a real-domain `Math` datatype (`expr`), with algebra/power rulesets
//! (`ruleset`), a real-domain evaluator (`eval`), and the data-aware `denoise`
//! mutation operator (`extract`). The `python` feature exposes `denoise` to
//! the Python SR engine via PyO3 (`from gamakAST import denoise`).

pub mod calibration;
pub mod eval;
pub mod expr;
pub mod extract;
pub mod ruleset;

#[cfg(feature = "python")]
mod python;
