//! Regex simplifier module, mirroring the Brainfuck/Math egglog layer.
//!
//! The regex kingdom in `../phylu` evolves data-quality assertion rules as
//! regexes. Evolved rules are bloated; this layer shrinks them by equality
//! saturation over an egglog datatype (`Re`), exactly as `bf_simplify` shrinks
//! Brainfuck. The layer structure mirrors the BF simplifier:
//!
//! - `expr.rs`    — RE_DATATYPE (egglog surface syntax for the `Re` sort)
//! - `ruleset.rs` — RE_RULESET (sound, acyclic, CONTRACTING rewrites)
//! - `parse.rs`   — phylu `ast_sexpr` S-expression <-> egglog `Re` s-expression
//! - `extract.rs` — `regex_simplify`: insert, bounded-saturate, extract
//!
//! # The S-expression format (the contract with phylu)
//!
//! Input and output are the SAME S-expression format phylu's
//! `gene_export::ast_sexpr` prints and `regex_sexpr_to_nodes` reads:
//!
//!   `(RegexConcat (RegexPlus (RegexDot)) (RegexCcDigit))`
//!
//! Op names are the fuller `Op` Debug names WITH the "Regex" prefix; a byte
//! literal is `(RegexLit (Num <byte>))`. `regex_simplify` takes that string and
//! returns the same format, simplified.
//!
//! # Soundness
//!
//! By construction, like BF: every rule is individually sound and CONTRACTING,
//! and tree extraction can only pick a member of the input's e-class. No DFA /
//! derivatives. Tests cross-check with the `regex` crate as an oracle.
//!
//! # Drawable-ops invariant
//!
//! The ruleset NEVER produces `RegexRepN`/`RegexRepUpto` or bare Integer leaves
//! — phylu's re-encoder (`write_back::grafts_of_nodes`) only maps drawable ops,
//! and a `Num` leaf there always means a Char byte. Keeping to drawable ops is
//! what lets a simplified rule round-trip back into a valid multi-typed gene.

pub mod expr;
pub mod extract;
pub mod parse;
pub mod ruleset;

pub use extract::{regex_simplify, Simplified};
pub use parse::{parse_regex, unparse_regex};
