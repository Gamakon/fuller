//! The linter's expression tree, its termination measure, and evaluation.
//!
//! The CPU reference works on a plain tree, so every form it produces is by
//! construction a K-expression once written out level by level. (The device
//! kernel's evaluator-private layout is a Phase 3 concern and is checked
//! against this.)

use egglog::TermDag;

use super::tables::op_name;
use crate::extract::PNode;
use crate::gpu_eval::Op;

#[derive(Debug, Clone)]
pub enum Tree {
    Num(f64),
    Var(String),
    App(Op, Vec<Tree>),
}

/// Structural equality. Literals compare by BIT PATTERN: `-0.0` and `0.0` are
/// different trees. A missed same-subtree match is sound; a false one is not.
impl PartialEq for Tree {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Tree::Num(a), Tree::Num(b)) => a.to_bits() == b.to_bits(),
            (Tree::Var(a), Tree::Var(b)) => a == b,
            (Tree::App(o1, k1), Tree::App(o2, k2)) => o1 == o2 && k1 == k2,
            _ => false,
        }
    }
}

/// The termination measure (spec §5), compared lexicographically:
/// `(node_count, #Pow, #Div, #ProtectedInv, M_neg)` where `M_neg` is the sum,
/// over `Neg` nodes, of the number of NON-`Neg` proper ancestors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Measure {
    pub nodes: usize,
    pub pow: usize,
    pub div: usize,
    pub protected_inv: usize,
    pub m_neg: usize,
}

impl Tree {
    pub(crate) fn from_pnode(p: &PNode) -> Result<Tree, String> {
        Ok(match p {
            PNode::Num(v) => Tree::Num(*v),
            PNode::Var(name) => Tree::Var(name.clone()),
            PNode::App(ctor, kids) => {
                let op = Op::from_math(ctor).ok_or_else(|| format!("unknown constructor {ctor}"))?;
                if op.arity() != kids.len() {
                    return Err(format!("{ctor} given {} children", kids.len()));
                }
                Tree::App(op, kids.iter().map(Tree::from_pnode).collect::<Result<_, _>>()?)
            }
        })
    }

    pub(crate) fn to_pnode(&self) -> PNode {
        match self {
            Tree::Num(v) => PNode::Num(*v),
            Tree::Var(name) => PNode::Var(name.clone()),
            Tree::App(op, kids) => {
                PNode::App(op_name(*op).to_string(), kids.iter().map(Tree::to_pnode).collect())
            }
        }
    }

    pub fn parse(math: &str) -> Result<Tree, String> {
        let p = crate::extract::parse_pnode(math).ok_or_else(|| format!("could not parse {math:?}"))?;
        Tree::from_pnode(&p)
    }

    pub fn to_math(&self) -> String {
        self.to_pnode().to_math()
    }

    pub fn node_count(&self) -> usize {
        match self {
            Tree::Num(_) | Tree::Var(_) => 1,
            Tree::App(_, kids) => 1 + kids.iter().map(Tree::node_count).sum::<usize>(),
        }
    }

    pub fn measure(&self) -> Measure {
        fn go(t: &Tree, non_neg_ancestors: usize, m: &mut Measure) {
            m.nodes += 1;
            if let Tree::App(op, kids) = t {
                match op {
                    Op::Pow => m.pow += 1,
                    Op::Div => m.div += 1,
                    Op::ProtectedInv => m.protected_inv += 1,
                    Op::Neg => m.m_neg += non_neg_ancestors,
                    _ => {}
                }
                let below = non_neg_ancestors + usize::from(*op != Op::Neg);
                for k in kids {
                    go(k, below, m);
                }
            }
        }
        let mut m = Measure { nodes: 0, pow: 0, div: 0, protected_inv: 0, m_neg: 0 };
        go(self, 0, &mut m);
        m
    }

    /// The subtree at `path` (child indices from the root).
    pub fn at(&self, path: &[u8]) -> Option<&Tree> {
        let mut t = self;
        for &i in path {
            match t {
                Tree::App(_, kids) => t = kids.get(i as usize)?,
                _ => return None,
            }
        }
        Some(t)
    }

    /// A copy with the subtree at `path` replaced.
    pub fn replaced(&self, path: &[u8], with: Tree) -> Tree {
        match (path.split_first(), self) {
            (None, _) => with,
            (Some((&i, rest)), Tree::App(op, kids)) => {
                let mut kids = kids.clone();
                if let Some(k) = kids.get_mut(i as usize) {
                    *k = k.replaced(rest, with);
                }
                Tree::App(*op, kids)
            }
            (Some(_), leaf) => leaf.clone(),
        }
    }

    /// Every position, in LEVEL ORDER — the order of a K-expression, and so the
    /// order in which the device kernel meets them.
    pub fn positions(&self) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = vec![Vec::new()];
        let mut next = 0usize;
        while next < out.len() {
            let path = out[next].clone();
            if let Some(Tree::App(_, kids)) = self.at(&path) {
                for i in 0..kids.len() {
                    let mut p = path.clone();
                    p.push(i as u8);
                    out.push(p);
                }
            }
            next += 1;
        }
        out
    }

    /// Evaluate on one row. Exactly the crate evaluator's semantics.
    pub fn eval(&self, row: &[(String, f64)]) -> Result<f64, String> {
        Compiled::new(self).eval(row)
    }
}

/// A tree lowered once to the evaluator's term form, for repeated evaluation.
pub struct Compiled {
    termdag: TermDag,
    root: egglog::TermId,
}

impl Compiled {
    pub fn new(tree: &Tree) -> Compiled {
        let mut termdag = TermDag::default();
        let root = crate::extract::pnode_to_term(&tree.to_pnode(), &mut termdag);
        Compiled { termdag, root }
    }

    pub fn eval(&self, row: &[(String, f64)]) -> Result<f64, String> {
        let consts = crate::snap_karva::constant_values();
        crate::eval::eval_term(&self.termdag, self.root, &|name: &str| {
            row.iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| *v)
                .or_else(|| consts.get(name).copied())
        })
        .map_err(|e| format!("{e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::Tree;

    fn t(s: &str) -> Tree {
        Tree::parse(s).unwrap()
    }

    #[test]
    fn round_trips_through_math_text() {
        let s = r#"(Sub (Neg (Var "a")) (Mul (Num 2.0) (Var "b")))"#;
        assert_eq!(t(s).to_math(), s);
        assert_eq!(t(s).node_count(), 6);
    }

    #[test]
    fn positions_are_level_order() {
        let tree = t(r#"(Add (Mul (Neg (Var "a")) (Var "b")) (Var "c"))"#);
        let ops: Vec<String> = tree
            .positions()
            .iter()
            .map(|p| tree.at(p).unwrap().to_math().chars().take(4).collect())
            .collect();
        assert_eq!(ops, ["(Add", "(Mul", "(Var", "(Neg", "(Var", "(Var"]);
    }

    /// The spec's v2 measure (sum of Neg depths) is RAISED by this float-out
    /// when `b` carries Negs of its own; M_neg is not. This is why the measure
    /// was replaced.
    #[test]
    fn m_neg_drops_where_sum_of_depths_rises() {
        fn sum_of_neg_depths(t: &Tree, depth: usize) -> usize {
            match t {
                Tree::App(op, kids) => {
                    let here = if *op == crate::gpu_eval::Op::Neg { depth } else { 0 };
                    here + kids.iter().map(|k| sum_of_neg_depths(k, depth + 1)).sum::<usize>()
                }
                _ => 0,
            }
        }
        let b = r#"(Mul (Neg (Var "u")) (Neg (Var "v")))"#;
        let before = t(&format!(r#"(Sub (Neg (Var "a")) {b})"#));
        let after = t(&format!(r#"(Neg (Add (Var "a") {b}))"#));
        assert!(sum_of_neg_depths(&after, 0) > sum_of_neg_depths(&before, 0));
        assert_eq!(before.measure().nodes, after.measure().nodes);
        assert_eq!(after.measure().m_neg + 1, before.measure().m_neg);
        assert!(after.measure() < before.measure());
    }

    #[test]
    fn negative_zero_is_a_different_tree() {
        assert_ne!(t("(Num 0.0)"), t("(Num -0.0)"));
    }
}
