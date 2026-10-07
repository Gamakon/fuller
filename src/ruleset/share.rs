//! THE REVERSE exp(a+b) -> exp(a)*exp(b) direction, for finding
//! equivalence-after-rewriting across a chromosome's genes
//! (`docs/PLAN_saturated_share_equivalence_v1.md`, `extract::maximal_shared_saturated`).
//!
//! `powers.rs` already has the FORWARD direction
//! (`(Mul (Exp a) (Exp b)) -> (Exp (Add a b))`) — but egglog only fires a
//! rule on a pattern that is literally present in the e-graph, and three
//! independently-asserted roots (`Exp(Add x y)`, `Exp(x)`, `Exp(y)`) never
//! construct a `Mul(Exp a, Exp b)` node on their own. Without this reverse
//! direction, those three roots never unify, confirmed by direct probe
//! against a live `egglog::EGraph` (not assumed from documentation).
//!
//! Deliberately NOT added to `powers` or loaded by `denoise`: this
//! direction is an EXPANDER (it can only grow an e-class, never shrink
//! it), and `denoise`'s `extract_variants` hangs on large classes built by
//! expander rules — the same reason `distribute` is excluded from
//! `denoise` today. Combined with `algebra`'s existing commutativity and
//! associativity on `Add`, this rule can expand `exp` of an n-term sum
//! into every partition of those terms — a real blowup risk a bounded
//! `repeat N` alone does not guard against (`repeat` bounds iteration
//! COUNT, not e-graph SIZE). Any caller loading this ruleset MUST also
//! guard e-graph size independently (via `EGraph::num_tuples()`, checked
//! between small `repeat` chunks) — see `extract::maximal_shared_saturated`.
pub const SHARE_RULESET: &str = r#"
(ruleset share)
(rewrite (Exp (Add a b)) (Mul (Exp a) (Exp b)) :ruleset share)
"#;

#[cfg(test)]
mod tests {
    use super::SHARE_RULESET;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use egglog::prelude::exprs as fs_exprs;
    use egglog::EGraph;

    /// THE NEGATIVE CASE, confirmed before this rule existed: three bare
    /// roots (as the motivating conversation posed it -- `exp(x+y)`,
    /// `exp(x)`, `exp(y)`, never a pre-built `Mul`) do NOT unify under
    /// `algebra`+`powers`+`sign` alone, because `powers`'s own rule is
    /// one-directional and nothing constructs the `Mul` pattern it
    /// matches on. If this assertion ever starts failing, something else
    /// in `algebra`/`powers`/`sign` changed to construct that `Mul` node
    /// independently, and this ruleset's whole justification needs
    /// re-checking, not just a changed expected value.
    #[test]
    fn three_bare_roots_do_not_unify_without_the_reverse_rule() {
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(None, MATH_DATATYPE).unwrap();
        egraph.parse_and_run_program(None, GUARD_RELATIONS).unwrap();
        egraph.parse_and_run_program(None, crate::ruleset::identities::ALGEBRA_RULESET).unwrap();
        egraph.parse_and_run_program(None, crate::ruleset::powers::POWERS_RULESET).unwrap();
        egraph.parse_and_run_program(None, crate::ruleset::sign::SIGN_RULESET).unwrap();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (let gene0 (Exp (Add (Var "x") (Var "y"))))
                (let gene1 (Exp (Var "x")))
                (let gene2 (Exp (Var "y")))
                (unstable-combined-ruleset probe_all guards algebra powers sign)
                (run-schedule (repeat 40 (run probe_all)))
                "#,
            )
            .unwrap();
        let (_s0, v0) = egraph.eval_expr(&fs_exprs::var("gene0")).unwrap();
        let (_s1, v1) = egraph.eval_expr(&fs_exprs::var("gene1")).unwrap();
        let (_s2, v2) = egraph.eval_expr(&fs_exprs::var("gene2")).unwrap();
        assert_ne!(v0, v1, "gene0 and gene1 must NOT unify without `share`'s reverse rule");
        assert_ne!(v0, v2, "gene0 and gene2 must NOT unify without `share`'s reverse rule");
    }

    /// THE POSITIVE CASE: the same three bare roots DO unify once `share`
    /// joins the combined ruleset -- gene0's e-class gains a `Mul` member
    /// whose two children resolve to exactly gene1's and gene2's classes.
    /// Confirmed by direct probe before this test existed, not assumed.
    #[test]
    fn three_bare_roots_unify_with_the_reverse_rule() {
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(None, MATH_DATATYPE).unwrap();
        egraph.parse_and_run_program(None, GUARD_RELATIONS).unwrap();
        egraph.parse_and_run_program(None, crate::ruleset::identities::ALGEBRA_RULESET).unwrap();
        egraph.parse_and_run_program(None, crate::ruleset::powers::POWERS_RULESET).unwrap();
        egraph.parse_and_run_program(None, crate::ruleset::sign::SIGN_RULESET).unwrap();
        egraph.parse_and_run_program(None, SHARE_RULESET).unwrap();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (let gene0 (Exp (Add (Var "x") (Var "y"))))
                (let gene1 (Exp (Var "x")))
                (let gene2 (Exp (Var "y")))
                (unstable-combined-ruleset probe_all2 guards algebra powers sign share)
                (run-schedule (repeat 40 (run probe_all2)))
                "#,
            )
            .unwrap();
        let (sort0, v0) = egraph.eval_expr(&fs_exprs::var("gene0")).unwrap();
        let (_s1, v1) = egraph.eval_expr(&fs_exprs::var("gene1")).unwrap();
        let (_s2, v2) = egraph.eval_expr(&fs_exprs::var("gene2")).unwrap();

        use egglog::SerializeConfig;
        let out = egraph.serialize(SerializeConfig {
            max_functions: None,
            max_calls_per_function: None,
            include_temporary_functions: false,
            root_eclasses: vec![(sort0.clone(), v0), (sort0.clone(), v1), (sort0.clone(), v2)],
        });
        let ser = out.egraph;
        let classes = ser.classes();
        let c0 = classes.get(&ser.root_eclasses[0]).expect("gene0's root class");
        assert_eq!(c0.nodes.len(), 2, "gene0's class should have exactly [Exp, Mul] members");
        let mul_node = c0.nodes.iter().map(|n| &ser.nodes[n]).find(|n| n.op == "Mul").expect("a Mul member in gene0's class");
        let child_classes: Vec<_> = mul_node.children.iter().map(|cid| ser.nodes[cid].eclass.clone()).collect();
        assert!(child_classes.contains(&ser.root_eclasses[1]), "Mul's children must resolve to gene1's class");
        assert!(child_classes.contains(&ser.root_eclasses[2]), "Mul's children must resolve to gene2's class");
    }
}
