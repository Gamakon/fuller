//! The regex `Re` datatype in egglog surface syntax.
//!
//! # Design
//!
//! A single recursive datatype `Re` — a flat tree where each constructor is a
//! regex AST node. A single sort (not Pattern/CharClass split) dodges egglog
//! 2.0's lack of mutual recursion across datatype declarations, exactly as the
//! BF `Prog` sort does. A byte literal rides as an `i64` payload (`Lit i64`),
//! mirroring BF's `AddN i64`.
//!
//!   Re :=
//!     (Empty)            — the empty pattern (matches "")       [RegexEmpty]
//!     (Dot)              — any one byte                          [RegexDot]
//!     (CcDigit)          — \d                                   [RegexCcDigit]
//!     (CcWord)           — \w                                   [RegexCcWord]
//!     (CcSpace)          — \s                                   [RegexCcSpace]
//!     (AnchorStart)      — ^                                     [RegexAnchorStart]
//!     (AnchorEnd)        — $                                     [RegexAnchorEnd]
//!     (Lit i64)          — one literal byte                      [RegexLit (Num b)]
//!     (Star Re)          — zero or more                          [RegexStar]
//!     (Plus Re)          — one or more                           [RegexPlus]
//!     (Opt Re)           — zero or one                           [RegexOpt]
//!     (ClassOf Re)       — a CharClass used as a Pattern         [RegexClassOf]
//!     (CcNegate Re)      — [^...] negation of a CharClass        [RegexCcNegate]
//!     (Concat Re Re)     — sequence                              [RegexConcat]
//!     (Alt Re Re)        — alternation |                         [RegexAlt]
//!     (CcRange Re Re)    — byte range a-z (operands are Lit)     [RegexCcRange]
//!     (CcUnion Re Re)    — union of two CharClasses              [RegexCcUnion]
//!
//! DELIBERATELY EXCLUDED: `RepN`/`RepUpto` (and any Integer leaf). phylu's
//! re-encoder only maps drawable ops and reads a `Num` leaf as a Char byte, so
//! the ruleset must never introduce counted repetition. (The arities match
//! `fuller::gpu_eval::Op::arity` for the regex ops.)

/// The `Re` datatype declaration for egglog.
pub const RE_DATATYPE: &str = r#"
(datatype Re
    (Empty)
    (Dot)
    (CcDigit)
    (CcWord)
    (CcSpace)
    (AnchorStart)
    (AnchorEnd)
    (Lit i64)
    (Star Re)
    (Plus Re)
    (Opt Re)
    (ClassOf Re)
    (CcNegate Re)
    (Concat Re Re)
    (Alt Re Re)
    (CcRange Re Re)
    (CcUnion Re Re))
"#;

#[cfg(test)]
mod tests {
    use super::RE_DATATYPE;
    use egglog::EGraph;

    #[test]
    fn re_datatype_loads() {
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(None, RE_DATATYPE).expect("RE_DATATYPE loads");
    }

    #[test]
    fn re_sample_program_parses() {
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(None, RE_DATATYPE).expect("datatype");
        // Concat(Plus(Dot), CcDigit)
        let prog = r#"(let __p (Concat (Plus (Dot)) (CcDigit)))"#;
        egraph.parse_and_run_program(None, prog).expect("program parses");
    }
}
