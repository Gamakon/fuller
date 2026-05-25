//! gamakAST — an egglog-based bidirectional AST hub for symbolic expression
//! rewriting.
//!
//! Phase 1.0-1.2: this crate proves egglog 2.0 can be driven
//! Rust -> egglog -> saturate -> extract -> Rust (calibration), defines the
//! real-domain `Math` expression datatype (`expr`), and the pure-algebra
//! identity ruleset (`ruleset::identities`). The public `denoise` /
//! `karva_to_terms` / `terms_to_karva` surface from BRIEF.md is not
//! implemented yet; it lands in later phases (1.5).

pub mod calibration;
pub mod eval;
pub mod expr;
pub mod extract;
pub mod ruleset;
