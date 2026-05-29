//! The Brainfuck `Prog` datatype in egglog surface syntax.
//!
//! # Design
//!
//! Programs are modelled as a single recursive datatype `Prog` — a flat
//! cons-list where each constructor is both the operation AND the container
//! for the rest of the program:
//!
//!   Prog :=
//!     (Nil)              — empty program
//!     (Inc  Prog)        — cell++ then rest (BF `+`)
//!     (Dec  Prog)        — cell-- then rest (BF `-`)
//!     (Left Prog)        — ptr-- then rest  (BF `<`)
//!     (Right Prog)       — ptr++ then rest  (BF `>`)
//!     (Out  Prog)        — output cell, rest (BF `.`)
//!     (In   Prog)        — read into cell, rest (BF `,`)
//!     (Loop Prog Prog)   — loop `body` while cell != 0, then `rest` (BF `[body]`)
//!     (AddN i64 Prog)    — aggregate Inc/Dec: net n, then rest
//!     (MoveN i64 Prog)   — aggregate Left/Right: net n, then rest
//!     (Clear Prog)       — set cell to 0, then rest (normal form of `[-]`/`[+]`)
//!
//! This avoids mutual recursion between `Prog` and `Op`, which egglog 2.0
//! does not support across separate datatype declarations.
//!
//! `AddN 0 rest` and `MoveN 0 rest` are no-ops equivalent to `rest` (erased by
//! rules). Positive `AddN k` means k net increments; negative means decrements.

/// The `Prog` datatype declaration for egglog.
pub const BF_DATATYPE: &str = r#"
(datatype Prog
    (Nil)
    (Inc Prog)
    (Dec Prog)
    (Left Prog)
    (Right Prog)
    (Out Prog)
    (In Prog)
    (Loop Prog Prog)
    (AddN i64 Prog)
    (MoveN i64 Prog)
    (Clear Prog))
"#;

#[cfg(test)]
mod tests {
    use super::BF_DATATYPE;
    use egglog::EGraph;

    #[test]
    fn bf_datatype_loads() {
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(None, BF_DATATYPE).expect("BF_DATATYPE loads");
    }

    #[test]
    fn bf_sample_program_parses() {
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(None, BF_DATATYPE).expect("datatype");
        // A few ops in a flat Prog: Inc (Right (Inc (Left (Dec (Nil)))))
        let prog = r#"(let __p (Inc (Right (Inc (Left (Dec (Nil)))))))"#;
        egraph.parse_and_run_program(None, prog).expect("program parses");
    }
}
