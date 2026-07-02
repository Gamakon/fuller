//! Brainfuck simplifier module, mirroring the Math/egglog layer.
//!
//! The Brainfuck ("BF") language has 8 ops: `+`, `-`, `<`, `>`, `[`, `]`,
//! `.`, `,`. Programs are modelled as an egglog datatype (`Prog`) so that
//! equality saturation can find semantically equivalent but structurally
//! smaller programs. The layer structure mirrors the Math simplifier:
//!
//! - `expr.rs`    — BF_DATATYPE (egglog surface syntax for the Prog sort)
//! - `eval.rs`    — tape interpreter (Rust, semantic ground-truth)
//! - `ruleset.rs` — BF_RULESET (sound, acyclic rewrites)
//! - `parse.rs`   — BF source string <-> Prog s-expression
//! - `extract.rs` — `bf_simplify`: insert, bounded-saturate, extract

pub mod eval;
pub mod expr;
pub mod extract;
pub mod parse;
pub mod ruleset;

// Re-export the primary public API.
pub use eval::{run_bf, TapeResult};
pub use extract::{bf_simplify, Simplified};
pub use parse::{parse_bf, unparse_bf};
