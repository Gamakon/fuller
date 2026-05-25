//! gamakAST — an egglog-based bidirectional AST hub for symbolic expression
//! rewriting.
//!
//! Phase 1.0 (calibration) only: this crate currently exposes a minimal
//! boolean-algebra round-trip used to prove that egglog 2.0 can be driven
//! Rust -> egglog -> saturate -> extract -> Rust on this machine. The public
//! `denoise` / `karva_to_terms` / `terms_to_karva` surface from BRIEF.md is
//! not implemented yet; it lands in later phases.

pub mod calibration;
