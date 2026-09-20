//! `collect`: what sympy does on CONSTRUCTION, spelled out one operand order at
//! a time.
//!
//! sympy's `simplify` has no pass that gathers the literals of `(3 + M) + 4`:
//! its `Add` and `Mul` are n-ary and unordered, so `Add.flatten` /
//! `Mul.flatten` (sympy core/add.py, core/mul.py) fold numeric coefficients and
//! like terms the moment an expression is built. A binary, ordered tree gets
//! none of that for free: every order in which a literal can sit across a term
//! is its own rule.
//!
//! These rules were read out of sympy's source by agents
//! (`docs/rule_proposals/sympy_{rational,powers,trig}.md`, 173 candidates),
//! judged mechanically — admitted at a level or refused, by evaluation
//! (`examples/judge_rules.rs`) — and then run over 113,444 expressions from
//! live runs. ONLY the ones that fired are here: a rule that never fires costs
//! matching time and buys nothing. Together they remove 7.8% more nodes than
//! the crate's rules did without them. The comment on each rule is its
//! measured fire count and the level the classifier gave it.

/// The `collect` ruleset.
pub const COLLECT_RULESET: &str = r#"
(ruleset collect)

; fired 739x on the live corpus; classified A/Rounding
(rewrite (Add (Sub q (Num b)) (Num a)) (Add q (Num (+ a (neg b)))) :ruleset collect)
; fired 366x on the live corpus; classified A/Rounding
(rewrite (Mul (ProtectedDiv (Num b) q) (Num a)) (ProtectedDiv (Num (* a b)) q) :ruleset collect)
; fired 364x on the live corpus; classified A/Finite
(rewrite (Sub a (Add a b)) (Neg b) :ruleset collect)
; fired 362x on the live corpus; classified A/Finite
(rewrite (Sub a (Sub a b)) b :ruleset collect)
; fired 280x on the live corpus; classified A/Rounding
(rewrite (Add (Num a) (Sub (Num b) q)) (Sub (Num (+ a b)) q) :ruleset collect)
; fired 228x on the live corpus; classified A/Finite
(rewrite (Sub (Sub a b) a) (Neg b) :ruleset collect)
; fired 219x on the live corpus; classified A/Rounding
(rewrite (Add (Num a) (Sub q (Num b))) (Add q (Num (+ a (neg b)))) :ruleset collect)
; fired 210x on the live corpus; classified A/Finite
(rewrite (Sub (Mul (Num a) x) x) (Mul (Num (+ a -1.0)) x) :ruleset collect)
; fired 191x on the live corpus; classified A/Rounding
(rewrite (Add (Num a) (Add q (Num b))) (Add q (Num (+ a b))) :ruleset collect)
; fired 180x on the live corpus; classified A/Rounding
(rewrite (Mul (Mul (Num b) p) (Num a)) (Mul (Num (* a b)) p) :ruleset collect)
; fired 150x on the live corpus; classified A/Rounding
(rewrite (Sub (Num a) (Add (Num b) q)) (Sub (Num (+ a (neg b))) q) :ruleset collect)
; fired 147x on the live corpus; classified A/Rounding
(rewrite (Sub (Num a) (Add q (Num b))) (Sub (Num (+ a (neg b))) q) :ruleset collect)
; fired 146x on the live corpus; classified A/Rounding
(rewrite (Mul (Num a) (ProtectedDiv (Num b) q)) (ProtectedDiv (Num (* a b)) q) :ruleset collect)
; fired 145x on the live corpus; classified A/Finite
(rewrite (Sub a (Add b a)) (Neg b) :ruleset collect)
; fired 145x on the live corpus; classified A/Rounding
(rewrite (Sub (Num a) (Sub q (Num b))) (Sub (Num (+ a b)) q) :ruleset collect)
; fired 142x on the live corpus; classified A/Rounding
(rewrite (Sub (Add q (Num b)) (Num a)) (Add q (Num (+ b (neg a)))) :ruleset collect)
; fired 132x on the live corpus; classified A/Rounding
(rewrite (Sub (Num a) (Sub (Num b) q)) (Add (Num (+ a (neg b))) q) :ruleset collect)
; fired 125x on the live corpus; classified A/Rounding
(rewrite (Mul (Num a) (Mul p (Num b))) (Mul (Num (* a b)) p) :ruleset collect)
; fired 115x on the live corpus; classified A/Rounding
(rewrite (Add (Add (Num b) q) (Num a)) (Add (Num (+ a b)) q) :ruleset collect)
; fired 114x on the live corpus; classified A/Rounding
(rewrite (Sub (Sub q (Num b)) (Num a)) (Sub q (Num (+ a b))) :ruleset collect)
; fired 94x on the live corpus; classified A/Rounding
(rewrite (Sub (Sub (Num b) q) (Num a)) (Sub (Num (+ b (neg a))) q) :ruleset collect)
; fired 92x on the live corpus; classified A/Rounding
(rewrite (Sub (Add (Num b) q) (Num a)) (Add (Num (+ b (neg a))) q) :ruleset collect)
; fired 89x on the live corpus; classified A/Rounding
(rewrite (Add (Sub (Num b) q) (Num a)) (Sub (Num (+ a b)) q) :ruleset collect)
; fired 71x on the live corpus; classified A/Bit
(rewrite (Neg (Mul x (Num a))) (Mul x (Num (neg a))) :ruleset collect)
; fired 61x on the live corpus; classified A/Bit
(rewrite (Mul (Num a) (Neg x)) (Mul (Num (neg a)) x) :ruleset collect)
; fired 60x on the live corpus; classified A/Bit
(rewrite (ProtectedDiv (Neg x) (Num a)) (ProtectedDiv x (Num (neg a))) :ruleset collect)
; fired 54x on the live corpus; classified A/Finite
(rewrite (Add (Mul x (Num a)) x) (Mul x (Num (+ a 1.0))) :ruleset collect)
; fired 48x on the live corpus; classified A/Finite
(rewrite (Add (Mul (Num a) x) x) (Mul (Num (+ a 1.0)) x) :ruleset collect)
; fired 40x on the live corpus; classified A/Bit
(rewrite (Mul (Neg x) (Num a)) (Mul (Num (neg a)) x) :ruleset collect)
; fired 35x on the live corpus; classified A/Bit
(rewrite (Neg (ProtectedDiv (Num a) x)) (ProtectedDiv (Num (neg a)) x) :ruleset collect)
; fired 28x on the live corpus; classified A/Bit
(rewrite (Neg (ProtectedDiv x (Num a))) (ProtectedDiv x (Num (neg a))) :ruleset collect)
; fired 27x on the live corpus; classified A/Finite
(rewrite (Sub x (Mul (Num a) x)) (Mul (Num (+ 1.0 (neg a))) x) :ruleset collect)
; fired 19x on the live corpus; classified A/Finite
(rewrite (Add x (Mul (Num a) x)) (Mul (Num (+ a 1.0)) x) :ruleset collect)
; fired 19x on the live corpus; classified A/Bit
(rewrite (Mul x (Pow2 x)) (Pow3 x) :ruleset collect)
; fired 19x on the live corpus; classified A/Rounding
(rewrite (Add (Mul (Sin a) (Cos b)) (Mul (Cos a) (Sin b))) (Sin (Add a b)) :ruleset collect)
; fired 16x on the live corpus; classified A/Bit
(rewrite (Neg (Mul (Num a) x)) (Mul (Num (neg a)) x) :ruleset collect)
; fired 15x on the live corpus; classified A/Finite
(rewrite (Sub (Mul a b) (Mul c a)) (Mul a (Sub b c)) :ruleset collect)
; fired 14x on the live corpus; classified A/Finite
(rewrite (Add (Mul b a) (Mul a c)) (Mul a (Add b c)) :ruleset collect)
; fired 14x on the live corpus; classified A/Rounding
(rewrite (Add (ProtectedDiv a c) (ProtectedDiv b c)) (ProtectedDiv (Add a b) c) :ruleset collect)
; fired 13x on the live corpus; classified A/Finite
(rewrite (Add (Mul a b) (Mul c a)) (Mul a (Add b c)) :ruleset collect)
; fired 12x on the live corpus; classified A/Rounding
(rewrite (Sub (Mul (Cos a) (Cos b)) (Mul (Sin a) (Sin b))) (Cos (Add a b)) :ruleset collect)
; fired 11x on the live corpus; classified A/Rounding
(rewrite (Sub (ProtectedDiv a c) (ProtectedDiv b c)) (ProtectedDiv (Sub a b) c) :ruleset collect)
; fired 10x on the live corpus; classified A/Bit
(rewrite (Mul (Pow2 x) x) (Pow3 x) :ruleset collect)
; fired 9x on the live corpus; classified A/Finite
(rewrite (Add (Mul a b) (Mul a c)) (Mul a (Add b c)) :ruleset collect)
; fired 6x on the live corpus; classified A/Finite
(rewrite (Sub (Mul a b) (Mul a c)) (Mul a (Sub b c)) :ruleset collect)
; fired 6x on the live corpus; classified A/Finite
(rewrite (Add (Mul b a) (Mul c a)) (Mul (Add b c) a) :ruleset collect)
; fired 5x on the live corpus; classified A/Finite
(rewrite (Sub (Mul b a) (Mul c a)) (Mul (Sub b c) a) :ruleset collect)
; fired 4x on the live corpus; classified A/Finite
(rewrite (Add (Mul x (Num a)) (Mul (Num b) x)) (Mul (Num (+ a b)) x) :ruleset collect)
; fired 4x on the live corpus; classified A/Rounding
(rewrite (Sub (Pow2 (Cos x)) (Pow2 (Sin x))) (Cos (Mul (Num 2.0) x)) :ruleset collect)
; fired 3x on the live corpus; classified A/Finite
(rewrite (Sub (Mul b a) (Mul a c)) (Mul a (Sub b c)) :ruleset collect)
; fired 2x on the live corpus; classified A/Finite
(rewrite (Sub (Mul (Num a) x) (Mul (Num b) x)) (Mul (Num (+ a (neg b))) x) :ruleset collect)
; fired 1x on the live corpus; classified A/Rounding
(rewrite (Mul x (Pow3 x)) (Pow2 (Pow2 x)) :ruleset collect)
"#;

#[cfg(test)]
mod tests {
    use super::COLLECT_RULESET;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use egglog::EGraph;

    /// The text is egglog's own: it must load there too, not only in the linter.
    #[test]
    fn egglog_loads_the_ruleset() {
        let mut e = EGraph::default();
        for prog in [MATH_DATATYPE, GUARD_RELATIONS, COLLECT_RULESET] {
            e.parse_and_run_program(None, prog).expect("program loads");
        }
    }
}
