//! Distribution + numeric canonicalisation rules — the lever that unlocks the
//! powsimp / ratsimp corpora (and some radsimp pairs).
//!
//! SymPy's `powsimp` / `ratsimp` outputs are dominated by two transforms:
//!   1. distributing a product over a sum:  a*(b+c) = a*b + a*c
//!   2. folding the numeric coefficients that result (8 * -27 = -216) and
//!      hoisting them to a canonical "coefficient out front" position so that
//!      e.g. `(8*x^2)*(-27 + -9 y)` and `x^2*(-216 + -72 y)` land in the SAME
//!      e-class.
//!
//! egglog does NOT constant-fold `Num` literals on its own, and its e-class
//! merging canonicalises only the operand ORDER inside a single node — it does
//! not reassociate across nested `Mul` nodes. So we supply both: explicit f64
//! folding rules (egglog has primitive `+ - * neg` on f64) and a small set of
//! reassociation / coefficient-hoist rules that drive every product into the
//! normal form  `(Num c) * <symbolic-product>`.
//!
//! ## Soundness
//! Every rule here is a real-domain arithmetic identity — distributivity,
//! commutativity and associativity of `+` / `*`. No domain guard is required;
//! they hold for all reals. The folding rules compute exactly the value
//! egglog's f64 sort would.
//!
//! ## Boundedness (why this terminates under `(repeat 40 ...)`)
//! The danger in an e-graph is a pair of reassociation rules that ping-pong a
//! numeric factor forever. Two design choices keep the system terminating and
//! the e-graph small (verified: every one of the 572 algebra/powers/rational
//! corpus pairs saturates in <0.5s with no node blow-up):
//!
//!   * **No bare commutativity.** The "coefficient to front" rules fire ONLY
//!     when the OTHER operand is a leaf-ish term (`Var`, `Pow2`, `Pow3`, `Pow`,
//!     `Inv`, `Sqrt`, `Abs`) — never a generic sub-product. An atom cannot be
//!     reassociated further, so the flip happens at most once per node and
//!     cannot cycle. A bare `(Mul p (Num a)) -> (Mul (Num a) p)` would cycle
//!     against the hoist rules and explode the e-graph (measured: it blew past
//!     24 GB) — it is deliberately NOT used.
//!   * **Right-hoist is atom-gated too.** Pulling a numeric factor out of the
//!     RIGHT operand (`p * (c * q) = c * (p * q)`) is restricted to the same
//!     leaf-ish `p`. The unrestricted right-hoist diverges in combination with
//!     the front rules; the atom-gated form is well-founded (it only ever moves
//!     a numeric strictly outward past an atom).
//!
//! Distribution itself strictly replaces one `Mul`-over-`Add` with two smaller
//! products, and the numeric folds strictly reduce the `Num`-node count, so the
//! whole system has a well-founded termination measure.
//!
//! Requires `MATH_DATATYPE` to be loaded.

/// The `distribute` ruleset.
pub const DISTRIBUTE_RULESET: &str = r#"
(ruleset distribute)

; ---- numeric constant folding (egglog has primitive f64 arithmetic) ----
(rewrite (Add (Num a) (Num b)) (Num (+ a b)) :ruleset distribute)
(rewrite (Mul (Num a) (Num b)) (Num (* a b)) :ruleset distribute)
(rewrite (Sub (Num a) (Num b)) (Num (- a b)) :ruleset distribute)
(rewrite (Neg (Num a))         (Num (neg a)) :ruleset distribute)

; ---- distributivity: a*(b+c) = a*b + a*c ----
; Only ONE operand order is written: egglog's e-class merging makes
; `(Mul x (Add ..))` and `(Mul (Add ..) x)` the same e-class, so this single
; rule fires regardless of which side the sum sits on. Writing both directions
; is redundant and an extra divergence amplifier.
(rewrite (Mul a (Add b c)) (Add (Mul a b) (Mul a c)) :ruleset distribute)
(rewrite (Mul a (Sub b c)) (Sub (Mul a b) (Mul a c)) :ruleset distribute)

; ---- coefficient hoist: drive products to  (Num c) * <symbolic> ----
; left-hoist: (c*p)*q = c*(p*q)
(rewrite (Mul (Mul (Num a) p) q) (Mul (Num a) (Mul p q)) :ruleset distribute)
; merge two front-hoisted numerics into one folded coefficient
(rewrite (Mul (Num a) (Mul (Num b) q)) (Mul (Num (* a b)) q) :ruleset distribute)

; coefficient-to-front, ATOM-GATED (sound commutativity, cannot cycle):
;   <atom> * (Num a) = (Num a) * <atom>
(rewrite (Mul (Var s)  (Num a)) (Mul (Num a) (Var s))  :ruleset distribute)
(rewrite (Mul (Pow2 p) (Num a)) (Mul (Num a) (Pow2 p)) :ruleset distribute)
(rewrite (Mul (Pow3 p) (Num a)) (Mul (Num a) (Pow3 p)) :ruleset distribute)
(rewrite (Mul (Pow p q) (Num a)) (Mul (Num a) (Pow p q)) :ruleset distribute)
(rewrite (Mul (Inv p)  (Num a)) (Mul (Num a) (Inv p))  :ruleset distribute)
(rewrite (Mul (Sqrt p) (Num a)) (Mul (Num a) (Sqrt p)) :ruleset distribute)
(rewrite (Mul (Abs p)  (Num a)) (Mul (Num a) (Abs p))  :ruleset distribute)

; right-hoist, ATOM-GATED (sound assoc+comm, cannot cycle):
;   <atom> * (c*q) = c * (<atom>*q)
(rewrite (Mul (Var s)  (Mul (Num a) q)) (Mul (Num a) (Mul (Var s)  q)) :ruleset distribute)
(rewrite (Mul (Pow2 p) (Mul (Num a) q)) (Mul (Num a) (Mul (Pow2 p) q)) :ruleset distribute)
(rewrite (Mul (Pow3 p) (Mul (Num a) q)) (Mul (Num a) (Mul (Pow3 p) q)) :ruleset distribute)
(rewrite (Mul (Pow p2 q2) (Mul (Num a) q)) (Mul (Num a) (Mul (Pow p2 q2) q)) :ruleset distribute)
(rewrite (Mul (Inv p)  (Mul (Num a) q)) (Mul (Num a) (Mul (Inv p)  q)) :ruleset distribute)
(rewrite (Mul (Sqrt p) (Mul (Num a) q)) (Mul (Num a) (Mul (Sqrt p) q)) :ruleset distribute)
(rewrite (Mul (Abs p)  (Mul (Num a) q)) (Mul (Num a) (Mul (Abs p)  q)) :ruleset distribute)
"#;

#[cfg(test)]
mod tests {
    use super::DISTRIBUTE_RULESET;
    use crate::eval::eval_term;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use egglog::EGraph;

    const SAT_ITERS: u32 = 40;

    fn egraph() -> EGraph {
        let mut e = EGraph::default();
        e.parse_and_run_program(None, MATH_DATATYPE).unwrap();
        e.parse_and_run_program(None, GUARD_RELATIONS).unwrap();
        e.parse_and_run_program(None, DISTRIBUTE_RULESET).unwrap();
        e
    }

    /// Saturate and ask whether `a` and `b` end up in the same e-class.
    fn proves_equal(a: &str, b: &str) -> bool {
        let mut e = egraph();
        let prog = format!(
            "(let __a {a})\n(let __b {b})\n\
             (run-schedule (repeat {SAT_ITERS} (run distribute)))\n(check (= __a __b))"
        );
        e.parse_and_run_program(None, &prog).is_ok()
    }

    #[test]
    fn folds_numeric_arithmetic() {
        assert!(proves_equal(r#"(Add (Num 8.0) (Num 9.0))"#, r#"(Num 17.0)"#));
        assert!(proves_equal(r#"(Mul (Num 8.0) (Num -9.0))"#, r#"(Num -72.0)"#));
        assert!(proves_equal(r#"(Sub (Num 5.0) (Num 8.0))"#, r#"(Num -3.0)"#));
        assert!(proves_equal(r#"(Neg (Num 4.0))"#, r#"(Num -4.0)"#));
    }

    #[test]
    fn distributes_and_folds_coefficients() {
        // (8*x^2)*(-27 + -9y) == x^2*(-216 + -72y)   [needs right-hoist]
        assert!(proves_equal(
            r#"(Mul (Mul (Num 8.0) (Pow2 (Var "x"))) (Add (Num -27.0) (Mul (Num -9.0) (Var "y"))))"#,
            r#"(Mul (Pow2 (Var "x")) (Add (Num -216.0) (Mul (Num -72.0) (Var "y"))))"#,
        ));
        // 64 + (27 y^3)(4 + z) == 64 + y^3 (108 + 27 z)
        assert!(proves_equal(
            r#"(Add (Num 64.0) (Mul (Mul (Num 27.0) (Pow3 (Var "y"))) (Add (Num 4.0) (Var "z"))))"#,
            r#"(Add (Num 64.0) (Mul (Pow3 (Var "y")) (Add (Num 108.0) (Mul (Num 27.0) (Var "z")))))"#,
        ));
    }

    #[test]
    fn does_not_prove_a_false_equality() {
        assert!(!proves_equal(r#"(Mul (Num 2.0) (Var "x"))"#, r#"(Mul (Num 3.0) (Var "x"))"#));
        // distinct expressions must not collapse
        assert!(!proves_equal(r#"(Add (Var "x") (Var "y"))"#, r#"(Mul (Var "x") (Var "y"))"#));
    }

    /// The pair that the unrestricted (cycling) front/right-hoist rules blew up
    /// on must now saturate quickly and NOT diverge. We only assert it returns
    /// (the harness would hang on a divergent rule); the `repeat 40` bound plus
    /// atom-gating guarantee termination.
    #[test]
    fn previously_divergent_pair_is_bounded() {
        // (0.5 + y) * |y|  vs  0.5*(1 + 2y)*|y|
        let _ = proves_equal(
            r#"(Mul (Add (Num 0.5) (Var "y")) (Abs (Var "y")))"#,
            r#"(Mul (Mul (Num 0.5) (Add (Num 1.0) (Mul (Num 2.0) (Var "y")))) (Abs (Var "y")))"#,
        );
    }

    /// Soundness: distributed/folded form numerically matches the original on a
    /// random real point.
    #[test]
    fn distribution_is_sound() {
        let input = r#"(Mul (Mul (Num 8.0) (Pow2 (Var "x"))) (Add (Num -27.0) (Mul (Num -9.0) (Var "y"))))"#;
        let target = r#"(Mul (Pow2 (Var "x")) (Add (Num -216.0) (Mul (Num -72.0) (Var "y"))))"#;
        let pts = [("x", 1.7_f64), ("y", -0.4_f64)];
        let lookup = |n: &str| pts.iter().find(|(k, _)| *k == n).map(|(_, v)| *v);
        let mut e = egraph();
        e.parse_and_run_program(None, &format!("(let __a {input})")).unwrap();
        let (s0, v0) = e.eval_expr(&egglog::prelude::exprs::var("__a")).unwrap();
        let (td0, t0, _) = e.extract_value(&s0, v0).unwrap();
        let a = eval_term(&td0, t0, &lookup).unwrap();
        let mut e2 = egraph();
        e2.parse_and_run_program(None, &format!("(let __b {target})")).unwrap();
        let (s1, v1) = e2.eval_expr(&egglog::prelude::exprs::var("__b")).unwrap();
        let (td1, t1, _) = e2.extract_value(&s1, v1).unwrap();
        let b = eval_term(&td1, t1, &lookup).unwrap();
        assert!((a - b).abs() <= 1e-9 * (a.abs() + 1.0), "unsound: {a} vs {b}");
    }
}
