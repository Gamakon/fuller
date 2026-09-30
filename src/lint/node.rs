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

    /// Infix text a standard parser reads (SRBench parses submissions with
    /// sympy's `parse_expr`), with NO algebra system involved in producing it.
    /// Protected operators are written as the engine's own symbolic map writes
    /// them: `ProtectedSqrt x` is `sqrt(Abs(x))`, `ProtectedLog x` is
    /// `log(Abs(x))`, `ProtectedExp` is `exp`, `ProtectedDiv a b` is `a/b`,
    /// `ProtectedInv x` is `1/x`, `ProtectedAsin x` / `ProtectedAcos x` are
    /// `asin(x)` / `acos(x)`. Every operand is parenthesised, so the text
    /// means exactly what the tree means whatever the reader's precedence.
    /// [`Tree::to_infix`], except that a protected operator is written as what it
    /// COMPUTES, a Piecewise sympy can read: `ProtectedDiv a b` is 0 where
    /// |b| < 1e-6, `ProtectedInv x` is 1 at x = 0, `ProtectedSqrt x` is 0 where x
    /// is not finite, `ProtectedAsin x` / `ProtectedAcos x` clamp x to [-1, 1]
    /// (and are 0 where x is not finite). `to_infix` writes them as a/b, 1/x,
    /// sqrt(Abs(x)), asin(x), acos(x) — a
    /// different function wherever those cases occur on the data (the Rust
    /// engine's Feynman I.50.26: chromosome R² 0.94, its a/b rendering 0.47).
    /// A caller that reports a model first turns every protected operator its
    /// DATA never triggers into the raw one (`evolve::engine::resolve_protected`),
    /// so what is left here is only what truly needs saying.
    pub fn to_infix_faithful(&self) -> String {
        match self {
            Tree::App(Op::ProtectedDiv, k) => {
                let (a, b) = (k[0].to_infix_faithful(), k[1].to_infix_faithful());
                format!("Piecewise((0, Abs({b}) < 1e-6), (({a}/{b}), True))")
            }
            Tree::App(Op::ProtectedInv, k) => {
                let a = k[0].to_infix_faithful();
                format!("Piecewise((1, Eq({a}, 0)), ((1/{a}), True))")
            }
            Tree::App(Op::ProtectedSqrt, k) => {
                let a = k[0].to_infix_faithful();
                format!("Piecewise((sqrt(Abs({a})), Abs({a}) < 1.7976931348623157e308), (0, True))")
            }
            Tree::App(op @ (Op::ProtectedAsin | Op::ProtectedAcos), k) => {
                let a = k[0].to_infix_faithful();
                let f = if *op == Op::ProtectedAsin { "arcsin" } else { "arccos" };
                // The clamp as a Piecewise, not Min/Max: numpy cannot execute sympy's
                // Min(1, Max(-1, array)) (it builds a ragged array and raises), and
                // the faithful form exists to be executed.
                format!("Piecewise(({f}(Piecewise((-1, {a} < -1), (1, {a} > 1), ({a}, True))), Abs({a}) < 1.7976931348623157e308), (0, True))")
            }
            Tree::App(op, k) => {
                // every other operator: the plain rendering, over faithful operands
                let plain = Tree::App(*op, k.iter().enumerate().map(|(i, _)| Tree::Var(format!("\u{1}{i}\u{2}"))).collect()).to_infix();
                k.iter().enumerate().fold(plain, |text, (i, kid)| text.replace(&format!("\u{1}{i}\u{2}"), &kid.to_infix_faithful()))
            }
            leaf => leaf.to_infix(),
        }
    }

    pub fn to_infix(&self) -> String {
        let one = |k: &[Tree]| k[0].to_infix();
        let two = |k: &[Tree]| (k[0].to_infix(), k[1].to_infix());
        // An integer-valued float prints as an INTEGER (`1`, not `1.0`). SRBench's
        // truth carries `sqrt(1 - v^2/c^2)` with an integer 1; our `1.0` made
        // `round_floats` leave `1.0` while the truth kept `1`, so `sym_diff` did
        // not cancel and a form that IS the law scored NO (the whole relativistic
        // family). Emitting ints as ints removes that false negative.
        // A SNAPPED CONSTANT PRINTS AS ITS SYMBOL. Snap corrects 1.4142129 to
        // the exact f64 sqrt(2), but printed as 1.4142135623730951 SRBench's
        // round_floats makes it 1.414 and sqrt(2)*sqrt(d^2/2) no longer cancels
        // to sqrt(d^2) (Feynman I.8.14: NO as a number, YES as `sqrt(2)`). Only
        // the exact library value (to 1e-12 relative) is spelled symbolically,
        // so a fitted number that merely happens to be close still prints as
        // the number it is.
        let symbolic = |v: f64| -> Option<&'static str> {
            let near = |c: f64| (v - c).abs() <= 1e-12 * c.abs();
            if near(std::f64::consts::PI) {
                Some("pi")
            } else if near(std::f64::consts::E) {
                Some("E")
            } else if near(std::f64::consts::SQRT_2) {
                Some("sqrt(2)")
            } else if near(std::f64::consts::TAU) {
                Some("(2*pi)")
            } else {
                None
            }
        };
        let num = |v: f64| -> String {
            if let Some(sym) = symbolic(v) {
                sym.to_string()
            } else if v.is_finite() && v == v.trunc() && v.abs() < 1e15 {
                format!("{}", v as i64)
            } else {
                format!("{v:?}")
            }
        };
        match self {
            Tree::Num(v) if *v < 0.0 => format!("({})", num(*v)),
            Tree::Num(v) => num(*v),
            Tree::Var(name) => name.clone(),
            Tree::App(op, k) => match op {
                Op::Add => { let (a, b) = two(k); format!("({a} + {b})") }
                Op::Sub => { let (a, b) = two(k); format!("({a} - {b})") }
                Op::Mul => { let (a, b) = two(k); format!("({a}*{b})") }
                Op::Div | Op::ProtectedDiv => { let (a, b) = two(k); format!("({a}/{b})") }
                Op::Pow => { let (a, b) = two(k); format!("({a}**{b})") }
                Op::Neg => format!("(-{})", one(k)),
                Op::Abs => format!("Abs({})", one(k)),
                Op::Sqrt => format!("sqrt({})", one(k)),
                Op::Log => format!("log({})", one(k)),
                Op::Exp | Op::ProtectedExp => format!("exp({})", one(k)),
                Op::Sin => format!("sin({})", one(k)),
                Op::Cos => format!("cos({})", one(k)),
                Op::Tan => format!("tan({})", one(k)),
                Op::Tanh => format!("tanh({})", one(k)),
                Op::Pow2 => format!("({}**2)", one(k)),
                Op::Pow3 => format!("({}**3)", one(k)),
                Op::Inv | Op::ProtectedInv => format!("(1/{})", one(k)),
                Op::ProtectedSqrt => format!("sqrt(Abs({}))", one(k)),
                Op::ProtectedLog => format!("log(Abs({}))", one(k)),
                // SRBench's truths come from metadata.yaml as NUMPY names: arcsin
                // and arccos are UNDEFINED sympy functions there, not sympy's
                // asin/acos, so a model spelled asin can never equal a truth
                // spelled arcsin (measured on Feynman I.30.5 and I.26.2: the
                // exact law is NO as asin, YES as arcsin). We spell them their way.
                Op::Asin | Op::ProtectedAsin => format!("arcsin({})", one(k)),
                Op::Acos | Op::ProtectedAcos => format!("arccos({})", one(k)),
                Op::Var | Op::Num => unreachable!("leaves are not applications"),
            },
        }
    }

    /// Would this form still MEAN anything after SRBench rounds it?
    ///
    /// `srbench/postprocessing/symbolic_utils.py::round_floats` rounds every
    /// float to 3 decimals and sends anything under 1e-4 to zero, BEFORE
    /// comparing our model with the law. A form can be numerically perfect and
    /// lose all its structure there: `-4096*y*tan(tan(0.000244*(x+y-2)))` is
    /// `-y*(x+y-2)` to eleven decimals, and rounding sends 0.000244 to 0, so
    /// what SRBench compares is `0`.
    ///
    /// The exact test needs the rounded tree evaluated; this is the cheap
    /// necessary condition that catches the whole class: a literal that ROUNDS
    /// TO ZERO while it is not zero has, by being there at all, been carrying
    /// something the rounded form cannot carry. Reported forms are small, so a
    /// walk costs nothing.
    pub fn dies_on_rounding(&self) -> bool {
        // SRBench sends |a| < 1e-4 to zero and ROUNDS everything else to three
        // decimals — and a literal such as 0.000244 is above the first
        // threshold yet still zero after the second. Both routes to zero count.
        fn vanishes(v: f64) -> bool {
            v != 0.0 && (v.abs() < 1e-4 || (v * 1000.0).round() == 0.0)
        }
        match self {
            Tree::Num(v) => vanishes(*v),
            Tree::Var(_) => false,
            Tree::App(_, kids) => kids.iter().any(Tree::dies_on_rounding),
        }
    }

    /// The EXACT round_floats test, on data: does applying SRBench's
    /// `round_floats` to this form's literals move its predictions off
    /// `reference` (the raw form's predictions) by more than `tol`?
    ///
    /// [`dies_on_rounding`] is the cheap structural necessary condition (a
    /// literal that VANISHES). It misses the class where rounding does not zero
    /// a literal but still DEGRADES it: `0.9*cos^2 + 0.1` written from a decimal
    /// coefficient rounds its distributed coefficients to `0.891/0.099` and no
    /// longer equals the law — the literal never vanished. This test catches
    /// that by rounding the constants ([`crate::srbench_equiv::snap_constants`],
    /// SRBench's rule verbatim) and re-evaluating. A form spelled with exact
    /// integers or `p/q` ratios rounds to itself and does NOT die; the decimal
    /// twin does — which is precisely why the rational spelling should win the
    /// reported-form sort. A rounding that produces a non-finite prediction
    /// (a denominator sent to zero) counts as dying.
    pub fn dies_on_rounding_on_data(&self, rows: &[Vec<(String, f64)>], reference: &[f64], tol: f64) -> bool {
        if rows.is_empty() || reference.is_empty() {
            return self.dies_on_rounding(); // no data: fall back to the structural check
        }
        self.rounding_drift_on_data(rows, reference).is_none_or(|drift| drift > tol)
    }

    /// HOW FAR SRBENCH'S ROUNDING MOVES THIS SPELLING ON THE DATA: the literals
    /// rounded as `round_floats` rounds them, the form re-evaluated on `rows`,
    /// and the mean squared drift from `reference` relative to the reference's
    /// variance. 0.0 is a spelling rounding cannot touch (an integer, a
    /// rational, a symbol); None when the rounded form does not evaluate.
    pub fn rounding_drift_on_data(&self, rows: &[Vec<(String, f64)>], reference: &[f64]) -> Option<f64> {
        let rounded = crate::srbench_equiv::snap_constants(&self.to_math());
        let pred = crate::extract::eval_expr_rows(&rounded, rows).ok()?;
        if pred.len() != reference.len() || pred.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let n = reference.len() as f64;
        let mean = reference.iter().sum::<f64>() / n;
        let var = reference.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let var = if var > 0.0 { var } else { 1.0 };
        let drift = pred.iter().zip(reference).map(|(p, r)| (p - r).powi(2)).sum::<f64>() / n / var;
        drift.is_finite().then_some(drift)
    }

    /// THE SPELLING SRBENCH CAN READ of a trig-of-inverse-trig composition.
    /// SRBench's truths carry `arcsin`/`arccos` as UNDEFINED sympy functions, so
    /// its simplify cannot see that `cos(arcsin(x))` is `sqrt(1 - x^2)` — and
    /// the search likes that spelling (Feynman II.13.23 was refused 7 times as
    /// `rho/cos(arcsin(v/c))` for `rho/sqrt(1 - v^2/c^2)`). This rewrites the
    /// compositions to their algebraic forms, which is a change of spelling, not
    /// of value, for |x| <= 1 (where the raw inverse functions are defined). The
    /// protected forms are left alone: past |x| = 1 they clamp, and the algebraic
    /// form would not.
    pub fn srbench_spelling(&self) -> Tree {
        match self {
            Tree::Num(_) | Tree::Var(_) => self.clone(),
            Tree::App(op, kids) => {
                let kids: Vec<Tree> = kids.iter().map(Tree::srbench_spelling).collect();
                let one_minus_sq = |x: &Tree| Tree::App(Op::Sqrt, vec![Tree::App(Op::Sub, vec![Tree::Num(1.0), Tree::App(Op::Pow2, vec![x.clone()])])]);
                if let [Tree::App(inner, k1)] = kids.as_slice() {
                    let x = &k1[0];
                    match (op, inner) {
                        (Op::Cos, Op::Asin) | (Op::Sin, Op::Acos) => return one_minus_sq(x),
                        (Op::Sin, Op::Asin) | (Op::Cos, Op::Acos) => return x.clone(),
                        (Op::Tan, Op::Asin) => return Tree::App(Op::Div, vec![x.clone(), one_minus_sq(x)]),
                        (Op::Tan, Op::Acos) => return Tree::App(Op::Div, vec![one_minus_sq(x), x.clone()]),
                        _ => {}
                    }
                }
                Tree::App(*op, kids)
            }
        }
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

    /// The data round_floats test: the DECIMAL spelling of shearflow2's law dies
    /// (its distributed coeffs degrade under round-to-3dp) while the RATIONAL
    /// spelling of the SAME law survives — so the sort that keys on this prefers
    /// the ratio, the form SRBench actually certifies.
    #[test]
    fn rational_spelling_survives_data_rounding_where_the_decimal_dies() {
        // rows over x_0, x_1; reference = the raw decimal form's predictions.
        let rows: Vec<Vec<(String, f64)>> = (0..40)
            .map(|i| {
                let x0 = 0.1 * i as f64;
                let x1 = 0.05 * i as f64 + 0.3;
                vec![("x_0".to_string(), x0), ("x_1".to_string(), x1)]
            })
            .collect();
        let decimal = r#"(Mul (Num -0.011111110940650105) (Mul (Sub (Num -9.0) (Pow2 (Mul (Num -9.0) (Cos (Var "x_1"))))) (Sin (Var "x_0"))))"#;
        let rational = r#"(Mul (Div (Num -1.0) (Num 90.0)) (Mul (Sub (Num -9.0) (Pow2 (Mul (Num -9.0) (Cos (Var "x_1"))))) (Sin (Var "x_0"))))"#;
        let reference = crate::extract::eval_expr_rows(decimal, &rows).unwrap();
        // The decimal DIES: rounding -0.0111 -> -0.011 shifts every prediction.
        assert!(t(decimal).dies_on_rounding_on_data(&rows, &reference, 1e-10), "decimal should die on rounding");
        // The rational SURVIVES: -1.0 and 90.0 round to themselves, drift 0.
        assert!(!t(rational).dies_on_rounding_on_data(&rows, &reference, 1e-10), "rational should survive rounding");
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
    fn infix_text_needs_no_algebra_system() {
        let e = t(r#"(Sub (ProtectedDiv (Var "a") (Pow2 (Var "b"))) (Mul (Num -2.5) (ProtectedSqrt (Neg (Var "c")))))"#);
        assert_eq!(e.to_infix(), "((a/(b**2)) - ((-2.5)*sqrt(Abs((-c)))))");
        // Integer-valued floats print as integers (3, not 3.0) — SRBench's truth
        // uses integer literals, and `1.0` vs `1` was a false-negative source.
        assert_eq!(t("(Num 3.0)").to_infix(), "3");
        assert_eq!(t("(Num 2.5)").to_infix(), "2.5");
        assert_eq!(t("(Num 1.0)").to_infix(), "1");
        // The exact library constants print as symbols; a nearby number does not.
        assert_eq!(t("(Num 1.4142135623730951)").to_infix(), "sqrt(2)");
        assert_eq!(t("(Num 3.141592653589793)").to_infix(), "pi");
        assert_eq!(t("(Num 2.718281828459045)").to_infix(), "E");
        assert_eq!(t("(Num 6.283185307179586)").to_infix(), "(2*pi)");
        assert_eq!(t("(Num 1.4142129717195346)").to_infix(), "1.4142129717195346");
        assert_eq!(t("(Mul (Num 3.141592653589793) (Var \"r\"))").to_infix(), "(pi*r)");
    }

    /// A rational spelling has zero rounding drift; a decimal that rounds has some.
    #[test]
    fn rounding_drift_is_zero_for_an_exact_spelling_and_positive_for_a_rounded_decimal() {
        let rows: Vec<Vec<(String, f64)>> = (0..20).map(|i| vec![("x".to_string(), 0.5 + i as f64 * 0.1)]).collect();
        let reference: Vec<f64> = rows.iter().map(|r| r[0].1 / 9.0).collect();
        let rational = t(r#"(Mul (Div (Num 1.0) (Num 9.0)) (Var "x"))"#);
        let decimal = t(r#"(Mul (Num 0.1111111111111111) (Var "x"))"#);
        let r = rational.rounding_drift_on_data(&rows, &reference).expect("evaluates");
        assert!(r < 1e-20, "an exact spelling drifts only by f64 noise: {r}");
        let d = decimal.rounding_drift_on_data(&rows, &reference).expect("evaluates");
        assert!(d > 0.0 && d < 1e-4, "0.111 against 1/9: {d}");
        assert!(!rational.dies_on_rounding_on_data(&rows, &reference, 1e-9));
        assert!(decimal.dies_on_rounding_on_data(&rows, &reference, 1e-9));
    }

    /// cos(arcsin x) becomes sqrt(1 - x^2) and friends; the protected forms and
    /// everything else stay; values agree where the raw functions are defined.
    #[test]
    fn trig_of_inverse_trig_is_spelled_algebraically_for_srbench() {
        let e = t(r#"(Div (Var "rho") (Cos (Asin (Div (Var "v") (Var "c")))))"#);
        assert_eq!(e.srbench_spelling().to_infix(), "(rho/sqrt((1 - ((v/c)**2))))");
        assert_eq!(t(r#"(Sin (Acos (Var "x")))"#).srbench_spelling().to_infix(), "sqrt((1 - (x**2)))");
        assert_eq!(t(r#"(Sin (Asin (Var "x")))"#).srbench_spelling().to_infix(), "x");
        assert_eq!(t(r#"(Tan (Asin (Var "x")))"#).srbench_spelling().to_infix(), "(x/sqrt((1 - (x**2))))");
        let untouched = t(r#"(Cos (ProtectedAsin (Var "x")))"#);
        assert_eq!(untouched.srbench_spelling(), untouched, "the clamped form is not the algebraic one past |x| = 1");
        let plain = t(r#"(Mul (Var "a") (Cos (Var "b")))"#);
        assert_eq!(plain.srbench_spelling(), plain);
        // Same value on a point inside the domain.
        let rows = vec![vec![("rho".to_string(), 2.0), ("v".to_string(), 0.6), ("c".to_string(), 1.0)]];
        let before = crate::extract::eval_expr_rows(&e.to_math(), &rows).unwrap()[0];
        let after = crate::extract::eval_expr_rows(&e.srbench_spelling().to_math(), &rows).unwrap()[0];
        assert!((before - after).abs() < 1e-12, "{before} vs {after}");
    }

    /// The inverse-trig ops: text round trip, the plain rendering (protected
    /// forms print as the plain function), and the faithful one, which writes
    /// the clamp and the non-finite case the protected forms really compute.
    #[test]
    fn inverse_trig_round_trips_and_prints() {
        for (ctor, plain) in [("Asin", "arcsin"), ("Acos", "arccos"), ("ProtectedAsin", "arcsin"), ("ProtectedAcos", "arccos")] {
            let s = format!(r#"({ctor} (Mul (Var "n") (Sin (Var "t"))))"#);
            let e = t(&s);
            assert_eq!(e.to_math(), s);
            assert_eq!(e.node_count(), 5);
            assert_eq!(e.to_infix(), format!("{plain}((n*sin(t)))"));
            let faithful = e.to_infix_faithful();
            if ctor.starts_with("Protected") {
                assert_eq!(
                    faithful,
                    format!("Piecewise(({plain}(Piecewise((-1, (n*sin(t)) < -1), (1, (n*sin(t)) > 1), ((n*sin(t)), True))), Abs((n*sin(t))) < 1.7976931348623157e308), (0, True))")
                );
            } else {
                assert_eq!(faithful, e.to_infix(), "a raw op is already what it computes");
            }
        }
        assert_eq!(t(r#"(Tan (Var "y"))"#).to_infix(), "tan(y)");
        // An operand's protected op is spelled out under a plain parent.
        assert_eq!(
            t(r#"(Sin (ProtectedAsin (Var "x")))"#).to_infix_faithful(),
            "sin(Piecewise((arcsin(Piecewise((-1, x < -1), (1, x > 1), (x, True))), Abs(x) < 1.7976931348623157e308), (0, True)))"
        );
    }

    /// On a row, the tree evaluates by the crate evaluator: clamp and NaN.
    #[test]
    fn inverse_trig_evaluates_like_the_crate_evaluator() {
        let at = |s: &str, x: f64| t(s).eval(&[("x".to_string(), x)]).unwrap();
        assert!(at(r#"(Asin (Var "x"))"#, 2.0).is_nan());
        assert_eq!(at(r#"(ProtectedAsin (Var "x"))"#, 2.0), std::f64::consts::FRAC_PI_2);
        assert_eq!(at(r#"(ProtectedAcos (Var "x"))"#, -3.0), std::f64::consts::PI);
        assert_eq!(at(r#"(Acos (Var "x"))"#, 1.0), 0.0);
    }

    #[test]
    fn negative_zero_is_a_different_tree() {
        assert_ne!(t("(Num 0.0)"), t("(Num -0.0)"));
    }
}
