//! The linter's tables: rules, guard rows, and what was refused.
//!
//! Spec §2. A rule is a theorem about specific symbols, so every rule knows
//! which symbols it mentions (`rule_symbols`), and the rules usable in a
//! kingdom are derived from the kingdom by an anti-join — never chosen by hand.

use std::collections::BTreeSet;

use crate::geneframe::{Symbol, Ty};
use crate::gpu_eval::Op;

/// `Math` constructor names, indexed by `Op` discriminant. Locked to
/// `Op::from_math` by a test, so the two cannot drift.
const OP_NAMES: [&str; 24] = [
    "Var", "Num", "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Sqrt", "Log", "Exp", "Sin", "Cos",
    "Tan", "Tanh", "Pow", "Pow2", "Pow3", "Inv", "ProtectedDiv", "ProtectedSqrt", "ProtectedLog",
    "ProtectedExp", "ProtectedInv",
];

/// The `Math` constructor name of an opcode.
pub fn op_name(op: Op) -> &'static str {
    OP_NAMES[op as usize]
}

/// Domain facts a guard can require, as a bitmask (spec §4, K1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Facts(pub u8);

impl Facts {
    pub const NONE: Facts = Facts(0);
    pub const POSITIVE: Facts = Facts(1);
    pub const NONZERO: Facts = Facts(2);
    pub const NONNEG: Facts = Facts(4);

    /// The fact a relation name asserts, if it is one of the fact relations.
    pub fn from_relation(name: &str) -> Option<Facts> {
        match name {
            "is-positive" => Some(Facts::POSITIVE),
            "is-nonzero" => Some(Facts::NONZERO),
            "is-nonneg" => Some(Facts::NONNEG),
            _ => None,
        }
    }

    pub fn contains(self, other: Facts) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn union(self, other: Facts) -> Facts {
        Facts(self.0 | other.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cmp {
    Lt,
    Gt,
    Ge,
}

impl Cmp {
    pub fn holds(self, v: f64, k: f64) -> bool {
        match self {
            Cmp::Lt => v < k,
            Cmp::Gt => v > k,
            Cmp::Ge => v >= k,
        }
    }
}

/// A numeric predicate on a literal bound by the pattern: `(< c 0.0)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pred {
    /// Index of the numeric metavariable.
    pub num: u8,
    pub cmp: Cmp,
    pub k: f64,
}

/// A pattern. Metavariables are numbered per rule; a repeated `Mv` index is
/// the same-subtree constraint (`Mul x x`).
#[derive(Debug, Clone, PartialEq)]
pub enum Pat {
    Op(Op, Vec<Pat>),
    Mv(u8),
    /// `(Num 1.0)` — this exact literal.
    NumExact(f64),
    /// `(Num a)` — any literal, bound to numeric metavariable `a`.
    NumMv(u8),
}

/// A literal computed from bound literals: `(neg c)`, `(* a b)`, `(+ a b)`.
#[derive(Debug, Clone, PartialEq)]
pub enum NumExpr {
    Const(f64),
    Mv(u8),
    Neg(Box<NumExpr>),
    Mul(Box<NumExpr>, Box<NumExpr>),
    Add(Box<NumExpr>, Box<NumExpr>),
}

impl NumExpr {
    pub fn eval(&self, nums: &[f64]) -> f64 {
        match self {
            NumExpr::Const(v) => *v,
            NumExpr::Mv(i) => nums[*i as usize],
            NumExpr::Neg(a) => -a.eval(nums),
            NumExpr::Mul(a, b) => a.eval(nums) * b.eval(nums),
            NumExpr::Add(a, b) => a.eval(nums) + b.eval(nums),
        }
    }

    /// True when the literal depends on a bound literal (it is COMPUTED, not
    /// written in the rule). These are excluded from the v1 device subset.
    pub fn is_computed(&self) -> bool {
        !matches!(self, NumExpr::Const(_))
    }
}

/// A replacement template.
#[derive(Debug, Clone, PartialEq)]
pub enum Tmpl {
    Op(Op, Vec<Tmpl>),
    Mv(u8),
    Num(NumExpr),
}

/// Where a rule sits in the termination order (spec §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Strictly reduces the node count.
    A,
    /// Keeps the node count; strictly reduces a later component of the measure.
    B,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    /// Source order across all loaded texts. Stable, so ties break the same
    /// way on every run.
    pub id: usize,
    pub ruleset: String,
    /// The egglog form this row was read from, for reports.
    pub text: String,
    pub lhs: Pat,
    pub rhs: Tmpl,
    pub preds: Vec<Pred>,
    /// Facts required of a metavariable's binding: `:when ((is-positive x))`.
    pub when: Vec<(u8, Facts)>,
    pub n_mv: u8,
    pub n_num: u8,
    pub order: Order,
    /// Exact only where every bound subterm is finite (class F): `Mul x 0 -> 0`
    /// is wrong at `x = inf`.
    pub finite_only: bool,
}

impl Rule {
    /// True when the template computes a literal from bound literals.
    pub fn computes_literal(&self) -> bool {
        fn go(t: &Tmpl) -> bool {
            match t {
                Tmpl::Op(_, kids) => kids.iter().any(go),
                Tmpl::Mv(_) => false,
                Tmpl::Num(e) => e.is_computed(),
            }
        }
        go(&self.rhs)
    }

    /// Every opcode the rule mentions, on EITHER side (spec §2: `Div 1 x ->
    /// Inv x` depends on `Inv` too).
    pub fn ops(&self) -> BTreeSet<u32> {
        fn pat(p: &Pat, out: &mut BTreeSet<u32>) {
            match p {
                Pat::Op(op, kids) => {
                    out.insert(*op as u32);
                    kids.iter().for_each(|k| pat(k, out));
                }
                Pat::NumExact(_) | Pat::NumMv(_) => {
                    out.insert(Op::Num as u32);
                }
                Pat::Mv(_) => {}
            }
        }
        fn tmpl(t: &Tmpl, out: &mut BTreeSet<u32>) {
            match t {
                Tmpl::Op(op, kids) => {
                    out.insert(*op as u32);
                    kids.iter().for_each(|k| tmpl(k, out));
                }
                Tmpl::Num(_) => {
                    out.insert(Op::Num as u32);
                }
                Tmpl::Mv(_) => {}
            }
        }
        let mut out = BTreeSet::new();
        pat(&self.lhs, &mut out);
        tmpl(&self.rhs, &mut out);
        out
    }

    /// `rule_symbols`: the `semantic_id` of every FUNCTION symbol the rule
    /// mentions. Literals and variables are terminals, not symbols.
    pub fn semantic_ids(&self) -> BTreeSet<&'static str> {
        self.ops()
            .into_iter()
            .filter_map(|o| OP_NAMES.get(o as usize))
            .filter_map(|name| crate::karva::math_ctor_to_semantic(name))
            .collect()
    }
}

/// One guard row: how a fact arises at a node (spec §2 `guard_seeds`,
/// `guard_propagation`, and fact implication).
#[derive(Debug, Clone, PartialEq)]
pub struct GuardRule {
    pub id: usize,
    pub text: String,
    /// The node's opcode, or `None` for a pure implication (`positive =>
    /// nonzero`).
    pub op: Option<Op>,
    /// Facts required of each child, by position. Empty mask = no requirement.
    pub child_req: Vec<Facts>,
    /// Predicates on the node's own literal (only when `op` is `Num`).
    pub num_preds: Vec<(Cmp, f64)>,
    /// Facts the node must already carry (implication).
    pub self_req: Facts,
    pub gives: Facts,
}

/// A form the reader understood and deliberately did not admit.
#[derive(Debug, Clone, PartialEq)]
pub struct Refused {
    pub ruleset: String,
    pub text: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default)]
pub struct Tables {
    pub rules: Vec<Rule>,
    pub guards: Vec<GuardRule>,
    pub refused: Vec<Refused>,
}

impl Tables {
    /// The rules usable in a kingdom: those whose every symbol is in it.
    /// Relational division, written as an anti-join (spec §2).
    pub fn usable<'a>(&'a self, kingdom: &[&Symbol]) -> Vec<&'a Rule> {
        let have: BTreeSet<&str> = kingdom.iter().map(|s| s.semantic_id.as_str()).collect();
        self.rules
            .iter()
            .filter(|r| r.semantic_ids().iter().all(|id| have.contains(id)))
            .collect()
    }

    /// Type-check every rule against the symbol rows: each function's child
    /// count must equal the symbol's input arity, and every input and output
    /// must be the same single type. Trivially true for Symbolic Regression
    /// (`F -> F`), and still run, so a mistyped row is a load error for any
    /// kingdom.
    pub fn type_check(&self, kingdom: &[&Symbol]) -> Result<(), String> {
        fn pat(p: &Pat, kingdom: &[&Symbol], rule: &Rule) -> Result<(), String> {
            if let Pat::Op(op, kids) = p {
                check(*op, kids.len(), kingdom, rule)?;
                kids.iter().try_for_each(|k| pat(k, kingdom, rule))?;
            }
            Ok(())
        }
        fn tmpl(t: &Tmpl, kingdom: &[&Symbol], rule: &Rule) -> Result<(), String> {
            if let Tmpl::Op(op, kids) = t {
                check(*op, kids.len(), kingdom, rule)?;
                kids.iter().try_for_each(|k| tmpl(k, kingdom, rule))?;
            }
            Ok(())
        }
        fn check(op: Op, n_kids: usize, kingdom: &[&Symbol], rule: &Rule) -> Result<(), String> {
            let Some(id) = crate::karva::math_ctor_to_semantic(op_name(op)) else {
                return Ok(()); // a terminal constructor, not a symbol row
            };
            let Some(sym) = kingdom.iter().find(|s| s.semantic_id == id) else {
                return Ok(()); // not in this kingdom: `usable` excludes the rule
            };
            if sym.arity.total_in() as usize != n_kids {
                return Err(format!(
                    "rule {} ({}): {id} takes {} inputs, the rule gives it {n_kids}",
                    rule.id,
                    rule.text,
                    sym.arity.total_in()
                ));
            }
            let single = |m: &std::collections::BTreeMap<Ty, u32>| m.len() <= 1;
            let in_ty = sym.arity.inputs.keys().next();
            let out_ty = sym.arity.outputs.keys().next();
            if !single(&sym.arity.inputs) || !single(&sym.arity.outputs) {
                return Err(format!(
                    "rule {} ({}): {id} is many-typed; per-instance arity resolution is not built",
                    rule.id, rule.text
                ));
            }
            if in_ty.is_some() && in_ty != out_ty {
                return Err(format!(
                    "rule {} ({}): {id} changes type; typed templates are not built",
                    rule.id, rule.text
                ));
            }
            Ok(())
        }
        for r in &self.rules {
            pat(&r.lhs, kingdom, r)?;
            tmpl(&r.rhs, kingdom, r)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{op_name, OP_NAMES};
    use crate::gpu_eval::Op;

    #[test]
    fn op_names_lockstep_with_from_math() {
        for (i, name) in OP_NAMES.iter().enumerate() {
            let op = Op::from_math(name).expect("every name is a constructor");
            assert_eq!(op as usize, i, "{name} is at the wrong index");
            assert_eq!(op_name(op), *name);
        }
    }
}
