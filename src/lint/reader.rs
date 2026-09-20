//! egglog rule text -> linter table rows.
//!
//! One written copy of every rule: the text egglog runs is the text read here.
//!
//! Accepted grammar (every form that occurs in the shipped rulesets):
//!
//! ```text
//! (ruleset NAME)                                   declaration, recorded
//! (relation NAME (Math))                           declaration, recorded
//! (rewrite PAT TMPL [:when (FACT..)] :ruleset R)   -> rule
//! (rule (PREMISE..) (ACTION..) :ruleset R)
//!     ACTION = (union e TMPL)                      -> rule   (premises: (= e PAT), (CMP n k))
//!     ACTION = (FACT-REL e)..                      -> guard  (premises: (= e (Op v..)), (FACT-REL v), (CMP n k))
//! PAT  = mv | (Num f64) | (Num mv) | (Op PAT..)
//! TMPL = mv | (Num NUM) | (Op TMPL..)
//! NUM  = f64 | mv | (neg NUM) | (* NUM NUM) | (+ NUM NUM)
//! CMP  = < | > | >=
//! ```
//!
//! Anything else is a LOAD ERROR. A form that is understood but cannot be a
//! linter row is REFUSED with a reason and reported — never dropped silently.

use std::collections::BTreeMap;

use super::sexp::{parse_all, Sexp};
use super::tables::{Cmp, Facts, GuardRule, NumExpr, Pat, Pred, Refused, Tmpl};
use crate::gpu_eval::Op;

/// Patterns and templates deeper than this do not fit the device's
/// 15-slot heap (spec §4).
pub const MAX_PATTERN_DEPTH: usize = 3;

/// A rule as read, before classification decides its order and exactness.
#[derive(Debug, Clone, PartialEq)]
pub struct Draft {
    pub ruleset: String,
    pub text: String,
    pub lhs: Pat,
    pub rhs: Tmpl,
    pub preds: Vec<Pred>,
    pub when: Vec<(u8, Facts)>,
    pub n_mv: u8,
    pub n_num: u8,
}

#[derive(Debug, Default)]
pub struct Read {
    pub drafts: Vec<Draft>,
    pub guards: Vec<GuardRule>,
    pub refused: Vec<Refused>,
}

enum Lowered {
    Declaration,
    Draft(Draft),
    Guard(GuardRule),
    Refused(String),
}

/// A reason a form cannot be a row. Distinct from a load error (`Err(String)`).
struct Refuse(String);

enum Fail {
    Refuse(Refuse),
    Error(String),
}

impl From<Refuse> for Fail {
    fn from(r: Refuse) -> Self {
        Fail::Refuse(r)
    }
}

/// Metavariable names of one rule, numbered in order of first appearance.
#[derive(Default)]
struct Names {
    math: Vec<String>,
    num: Vec<String>,
}

impl Names {
    fn math(&mut self, name: &str) -> u8 {
        index_of(&mut self.math, name)
    }
    fn num(&mut self, name: &str) -> u8 {
        index_of(&mut self.num, name)
    }
    fn bound_math(&self, name: &str) -> Option<u8> {
        self.math.iter().position(|n| n == name).map(|i| i as u8)
    }
    fn bound_num(&self, name: &str) -> Option<u8> {
        self.num.iter().position(|n| n == name).map(|i| i as u8)
    }
}

fn index_of(names: &mut Vec<String>, name: &str) -> u8 {
    match names.iter().position(|n| n == name) {
        Some(i) => i as u8,
        None => {
            names.push(name.to_string());
            (names.len() - 1) as u8
        }
    }
}

/// Read every form of every source text. `sources` is `(label, text)`.
pub fn read(sources: &[(&str, &str)]) -> Result<Read, String> {
    let mut out = Read::default();
    for (label, text) in sources {
        let forms = parse_all(text).map_err(|e| format!("{label}: {e}"))?;
        for form in &forms {
            let shown = show(form);
            let ruleset = ruleset_of(form).unwrap_or_else(|| (*label).to_string());
            match lower(form).map_err(|e| format!("{label}: {e}\n  in {shown}"))? {
                Lowered::Declaration => {}
                Lowered::Draft(mut d) => {
                    d.ruleset = ruleset;
                    d.text = shown;
                    out.drafts.push(d);
                }
                Lowered::Guard(mut g) => {
                    g.id = out.guards.len();
                    g.text = shown;
                    out.guards.push(g);
                }
                Lowered::Refused(reason) => out.refused.push(Refused {
                    ruleset,
                    text: shown,
                    reason,
                }),
            }
        }
    }
    Ok(out)
}

fn lower(form: &Sexp) -> Result<Lowered, String> {
    let result = match form.head() {
        Some("ruleset") | Some("relation") => return Ok(Lowered::Declaration),
        Some("rewrite") => lower_rewrite(form),
        Some("rule") => lower_rule(form),
        _ => return Err("unrecognised top-level form".to_string()),
    };
    match result {
        Ok(l) => Ok(l),
        Err(Fail::Refuse(Refuse(reason))) => Ok(Lowered::Refused(reason)),
        Err(Fail::Error(e)) => Err(e),
    }
}

fn lower_rewrite(form: &Sexp) -> Result<Lowered, Fail> {
    let items = form.list().unwrap_or(&[]);
    if items.len() < 3 {
        return Err(Fail::Error("rewrite needs a pattern and a template".to_string()));
    }
    let mut names = Names::default();
    let lhs = pattern(&items[1], &mut names, 0)?;
    let when = keyword(items, ":when")
        .map(|w| when_facts(w, &names))
        .transpose()?
        .unwrap_or_default();
    let rhs = template(&items[2], &names, 0)?;
    Ok(Lowered::Draft(draft(lhs, rhs, Vec::new(), when, &names)))
}

fn lower_rule(form: &Sexp) -> Result<Lowered, Fail> {
    let items = form.list().unwrap_or(&[]);
    let (Some(premises), Some(actions)) = (
        items.get(1).and_then(Sexp::list),
        items.get(2).and_then(Sexp::list),
    ) else {
        return Err(Fail::Error("rule needs a premise list and an action list".to_string()));
    };
    if actions.is_empty() {
        return Err(Fail::Error("rule has no action".to_string()));
    }
    if actions.iter().all(|a| a.head() == Some("union")) {
        if actions.len() != 1 {
            return Err(Fail::Error("rule with several unions".to_string()));
        }
        union_rule(premises, &actions[0])
    } else {
        guard_rule(premises, actions)
    }
}

/// `(rule ((= e PAT) (CMP n k)..) ((union e TMPL)))` — a rewrite with numeric
/// predicates on bound literals.
fn union_rule(premises: &[Sexp], action: &Sexp) -> Result<Lowered, Fail> {
    let act = action.list().unwrap_or(&[]);
    let (Some(target), Some(tmpl)) = (act.get(1).and_then(Sexp::atom), act.get(2)) else {
        return Err(Fail::Error("malformed union".to_string()));
    };
    let mut names = Names::default();
    let mut lhs = None;
    let mut raw_preds = Vec::new();
    let mut raw_facts = Vec::new();
    for p in premises {
        let items = p.list().unwrap_or(&[]);
        match p.head() {
            Some("=") => {
                if items.get(1).and_then(Sexp::atom) != Some(target) || lhs.is_some() {
                    return Err(Fail::Error("union rule needs exactly one (= e PAT)".to_string()));
                }
                lhs = Some(pattern(&items[2], &mut names, 0)?);
            }
            Some("<") | Some(">") | Some(">=") => raw_preds.push(p),
            Some(_) => raw_facts.push(p),
            None => return Err(Fail::Error("malformed premise".to_string())),
        }
    }
    let lhs = lhs.ok_or_else(|| Fail::Error("union rule has no (= e PAT)".to_string()))?;
    let mut preds = Vec::new();
    for p in raw_preds {
        let (cmp, name, k) = comparison(p)?;
        let num = names
            .bound_num(name)
            .ok_or_else(|| Fail::Error(format!("predicate on unbound literal `{name}`")))?;
        preds.push(Pred { num, cmp, k });
    }
    let when = when_facts(&Sexp::List(raw_facts.into_iter().cloned().collect()), &names)?;
    let rhs = template(tmpl, &names, 0)?;
    Ok(Lowered::Draft(draft(lhs, rhs, preds, when, &names)))
}

/// `(rule (PREMISE..) ((FACT e)..))` — a guard row: seed, propagation, or
/// implication.
fn guard_rule(premises: &[Sexp], actions: &[Sexp]) -> Result<Lowered, Fail> {
    let mut gives = Facts::NONE;
    let mut target: Option<&str> = None;
    for a in actions {
        let items = a.list().unwrap_or(&[]);
        let (Some(rel), Some(var)) = (a.head(), items.get(1).and_then(Sexp::atom)) else {
            return Err(Fail::Error("malformed guard action".to_string()));
        };
        let Some(fact) = Facts::from_relation(rel) else {
            return Err(Refuse(format!("derives `{rel}`, which is not a fact relation")).into());
        };
        if target.is_some_and(|t| t != var) {
            return Err(Fail::Error("guard actions name different nodes".to_string()));
        }
        target = Some(var);
        gives = gives.union(fact);
    }
    let target = target.ok_or_else(|| Fail::Error("guard rule has no action".to_string()))?;

    let mut op = None;
    let mut children: Vec<String> = Vec::new();
    let mut literal: Option<String> = None;
    let mut var_facts: BTreeMap<String, Facts> = BTreeMap::new();
    let mut raw_preds = Vec::new();
    for p in premises {
        let items = p.list().unwrap_or(&[]);
        match p.head() {
            Some("=") => {
                if items.get(1).and_then(Sexp::atom) != Some(target) || op.is_some() {
                    return Err(Fail::Error("guard rule needs at most one (= e (Op ..))".to_string()));
                }
                let shape = items.get(2).and_then(Sexp::list).unwrap_or(&[]);
                let ctor = shape.first().and_then(Sexp::atom).unwrap_or("");
                let this = Op::from_math(ctor)
                    .ok_or_else(|| Fail::Error(format!("unknown constructor `{ctor}`")))?;
                if this == Op::Var {
                    return Err(Refuse("guard over a bare variable".to_string()).into());
                }
                for kid in &shape[1..] {
                    let Some(name) = kid.atom() else {
                        return Err(Refuse("guard pattern deeper than one level".to_string()).into());
                    };
                    children.push(name.to_string());
                }
                if this == Op::Num {
                    literal = children.pop();
                } else if children.len() != this.arity() {
                    return Err(Fail::Error(format!("{ctor} given {} children", children.len())));
                }
                op = Some(this);
            }
            Some("<") | Some(">") | Some(">=") => raw_preds.push(p),
            Some(rel) => {
                let Some(fact) = Facts::from_relation(rel) else {
                    return Err(Refuse(format!("requires `{rel}`, which is not a fact relation")).into());
                };
                let Some(var) = items.get(1).and_then(Sexp::atom) else {
                    return Err(Fail::Error("malformed fact premise".to_string()));
                };
                let slot = var_facts.entry(var.to_string()).or_default();
                *slot = slot.union(fact);
            }
            None => return Err(Fail::Error("malformed premise".to_string())),
        }
    }

    let mut num_preds = Vec::new();
    for p in raw_preds {
        let (cmp, name, k) = comparison(p)?;
        if literal.as_deref() != Some(name) {
            return Err(Fail::Error(format!("predicate on `{name}`, which is not the node's literal")));
        }
        num_preds.push((cmp, k));
    }
    let self_req = var_facts.remove(target).unwrap_or_default();
    let child_req: Vec<Facts> = children
        .iter()
        .map(|c| var_facts.remove(c).unwrap_or_default())
        .collect();
    if let Some((stray, _)) = var_facts.into_iter().next() {
        return Err(Fail::Error(format!("fact premise on `{stray}`, which the rule never binds")));
    }
    if op.is_none() && self_req == Facts::NONE {
        return Err(Fail::Error("guard rule with nothing to match".to_string()));
    }
    Ok(Lowered::Guard(GuardRule {
        id: 0,
        text: String::new(),
        op,
        child_req,
        num_preds,
        self_req,
        gives,
    }))
}

fn draft(lhs: Pat, rhs: Tmpl, preds: Vec<Pred>, when: Vec<(u8, Facts)>, names: &Names) -> Draft {
    Draft {
        ruleset: String::new(),
        text: String::new(),
        lhs,
        rhs,
        preds,
        when,
        n_mv: names.math.len() as u8,
        n_num: names.num.len() as u8,
    }
}

fn pattern(s: &Sexp, names: &mut Names, depth: usize) -> Result<Pat, Fail> {
    if depth > MAX_PATTERN_DEPTH {
        return Err(Refuse(format!("pattern deeper than {MAX_PATTERN_DEPTH}")).into());
    }
    match s {
        Sexp::Atom(name) => Ok(Pat::Mv(names.math(name))),
        Sexp::Str(_) => Err(Fail::Error("string in a pattern".to_string())),
        Sexp::List(items) => {
            let ctor = items.first().and_then(Sexp::atom).unwrap_or("");
            let op = Op::from_math(ctor)
                .ok_or_else(|| Fail::Error(format!("unknown constructor `{ctor}`")))?;
            match op {
                Op::Var => Err(Refuse("pattern over a bare variable".to_string()).into()),
                Op::Num => {
                    let Some(arg) = items.get(1).and_then(Sexp::atom) else {
                        return Err(Fail::Error("malformed (Num ..) pattern".to_string()));
                    };
                    Ok(match arg.parse::<f64>() {
                        Ok(v) => Pat::NumExact(v),
                        Err(_) => Pat::NumMv(names.num(arg)),
                    })
                }
                _ => {
                    if items.len() - 1 != op.arity() {
                        return Err(Fail::Error(format!("{ctor} given {} children", items.len() - 1)));
                    }
                    let kids = items[1..]
                        .iter()
                        .map(|k| pattern(k, names, depth + 1))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Pat::Op(op, kids))
                }
            }
        }
    }
}

fn template(s: &Sexp, names: &Names, depth: usize) -> Result<Tmpl, Fail> {
    if depth > MAX_PATTERN_DEPTH {
        return Err(Refuse(format!("template deeper than {MAX_PATTERN_DEPTH}")).into());
    }
    match s {
        Sexp::Atom(name) => names
            .bound_math(name)
            .map(Tmpl::Mv)
            .ok_or_else(|| Fail::Error(format!("template uses unbound `{name}`"))),
        Sexp::Str(_) => Err(Fail::Error("string in a template".to_string())),
        Sexp::List(items) => {
            let ctor = items.first().and_then(Sexp::atom).unwrap_or("");
            let op = Op::from_math(ctor)
                .ok_or_else(|| Fail::Error(format!("unknown constructor `{ctor}`")))?;
            match op {
                Op::Var => Err(Refuse("template introduces a variable".to_string()).into()),
                Op::Num => {
                    let arg = items
                        .get(1)
                        .ok_or_else(|| Fail::Error("malformed (Num ..) template".to_string()))?;
                    Ok(Tmpl::Num(num_expr(arg, names)?))
                }
                _ => {
                    if items.len() - 1 != op.arity() {
                        return Err(Fail::Error(format!("{ctor} given {} children", items.len() - 1)));
                    }
                    let kids = items[1..]
                        .iter()
                        .map(|k| template(k, names, depth + 1))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Tmpl::Op(op, kids))
                }
            }
        }
    }
}

fn num_expr(s: &Sexp, names: &Names) -> Result<NumExpr, Fail> {
    match s {
        Sexp::Atom(a) => match a.parse::<f64>() {
            Ok(v) => Ok(NumExpr::Const(v)),
            Err(_) => names
                .bound_num(a)
                .map(NumExpr::Mv)
                .ok_or_else(|| Fail::Error(format!("literal uses unbound `{a}`"))),
        },
        Sexp::Str(_) => Err(Fail::Error("string in a literal".to_string())),
        Sexp::List(items) => {
            let arg = |i: usize| -> Result<Box<NumExpr>, Fail> {
                let item = items
                    .get(i)
                    .ok_or_else(|| Fail::Error("literal primitive is missing an argument".to_string()))?;
                Ok(Box::new(num_expr(item, names)?))
            };
            match (s.head(), items.len()) {
                (Some("neg"), 2) => Ok(NumExpr::Neg(arg(1)?)),
                (Some("*"), 3) => Ok(NumExpr::Mul(arg(1)?, arg(2)?)),
                (Some("+"), 3) => Ok(NumExpr::Add(arg(1)?, arg(2)?)),
                (other, _) => Err(Fail::Error(format!("unsupported literal primitive {other:?}"))),
            }
        }
    }
}

/// `((is-positive x) (is-nonzero y))` -> facts per metavariable.
fn when_facts(list: &Sexp, names: &Names) -> Result<Vec<(u8, Facts)>, Fail> {
    let mut out: Vec<(u8, Facts)> = Vec::new();
    for cond in list.list().unwrap_or(&[]) {
        let items = cond.list().unwrap_or(&[]);
        let (Some(rel), Some(var)) = (cond.head(), items.get(1).and_then(Sexp::atom)) else {
            return Err(Fail::Error("malformed :when condition".to_string()));
        };
        let Some(fact) = Facts::from_relation(rel) else {
            return Err(Refuse(format!("guarded by `{rel}`, which is not a fact relation")).into());
        };
        let mv = names
            .bound_math(var)
            .ok_or_else(|| Fail::Error(format!(":when on unbound `{var}`")))?;
        match out.iter_mut().find(|(m, _)| *m == mv) {
            Some((_, f)) => *f = f.union(fact),
            None => out.push((mv, fact)),
        }
    }
    Ok(out)
}

/// `(< c 0.0)` -> (Lt, "c", 0.0). The literal must be on the right.
fn comparison(p: &Sexp) -> Result<(Cmp, &str, f64), Fail> {
    let items = p.list().unwrap_or(&[]);
    let cmp = match p.head() {
        Some("<") => Cmp::Lt,
        Some(">") => Cmp::Gt,
        Some(">=") => Cmp::Ge,
        other => return Err(Fail::Error(format!("unsupported comparison {other:?}"))),
    };
    let (Some(name), Some(k)) = (
        items.get(1).and_then(Sexp::atom),
        items.get(2).and_then(Sexp::atom).and_then(|a| a.parse::<f64>().ok()),
    ) else {
        return Err(Fail::Error("comparison must be (CMP name literal)".to_string()));
    };
    Ok((cmp, name, k))
}

fn keyword<'a>(items: &'a [Sexp], key: &str) -> Option<&'a Sexp> {
    items
        .iter()
        .position(|i| i.atom() == Some(key))
        .and_then(|i| items.get(i + 1))
}

fn ruleset_of(form: &Sexp) -> Option<String> {
    keyword(form.list()?, ":ruleset")?.atom().map(str::to_string)
}

/// Render a form back to text, for reports and refusal messages.
pub fn show(s: &Sexp) -> String {
    match s {
        Sexp::Atom(a) => a.clone(),
        Sexp::Str(v) => format!("\"{v}\""),
        Sexp::List(items) => {
            format!("({})", items.iter().map(show).collect::<Vec<_>>().join(" "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(text: &str) -> Read {
        read(&[("test", text)]).expect("reads")
    }

    #[test]
    fn plain_rewrite() {
        let r = one("(rewrite (Mul x (Num 1.0)) x :ruleset algebra)");
        let d = &r.drafts[0];
        assert_eq!(d.ruleset, "algebra");
        assert_eq!(d.lhs, Pat::Op(Op::Mul, vec![Pat::Mv(0), Pat::NumExact(1.0)]));
        assert_eq!(d.rhs, Tmpl::Mv(0));
        assert_eq!((d.n_mv, d.n_num), (1, 0));
    }

    #[test]
    fn nonlinear_pattern_repeats_the_index() {
        let r = one("(rewrite (Sub x x) (Num 0.0) :ruleset algebra)");
        assert_eq!(r.drafts[0].lhs, Pat::Op(Op::Sub, vec![Pat::Mv(0), Pat::Mv(0)]));
    }

    #[test]
    fn when_facts_attach_to_metavariables() {
        let r = one("(rewrite (Mul (Exp a) (Exp b)) (Exp (Add a b)) :when ((is-positive a) (is-nonzero a) (is-positive b)) :ruleset powers)");
        assert_eq!(
            r.drafts[0].when,
            vec![(0, Facts::POSITIVE.union(Facts::NONZERO)), (1, Facts::POSITIVE)]
        );
    }

    #[test]
    fn computed_literals_nest() {
        let r = one("(rewrite (Pow3 (Num a)) (Num (* a (* a a))) :ruleset rational)");
        let Tmpl::Num(e) = &r.drafts[0].rhs else { panic!("not a literal") };
        assert_eq!(e.eval(&[2.0]), 8.0);
        assert!(e.is_computed());
    }

    #[test]
    fn union_rule_with_two_predicates() {
        let r = one(
            "(rule ((= e (ProtectedDiv x (Num k))) (< k 0.000001) (> k -0.000001)) ((union e (Num 0.0))) :ruleset algebra)",
        );
        let d = &r.drafts[0];
        assert_eq!(d.preds.len(), 2);
        assert_eq!(d.preds[0], Pred { num: 0, cmp: Cmp::Lt, k: 0.000001 });
        assert_eq!(d.rhs, Tmpl::Num(NumExpr::Const(0.0)));
    }

    #[test]
    fn guard_seed_with_two_conclusions() {
        let r = one("(rule ((= e (Exp x))) ((is-positive e) (is-nonzero e)) :ruleset guards)");
        let g = &r.guards[0];
        assert_eq!(g.op, Some(Op::Exp));
        assert_eq!(g.gives, Facts::POSITIVE.union(Facts::NONZERO));
        assert_eq!(g.child_req, vec![Facts::NONE]);
    }

    #[test]
    fn guard_propagation_and_implication_and_literal_seed() {
        let r = one(
            "(rule ((is-positive a) (is-positive b) (= m (Mul a b))) ((is-positive m)) :ruleset guards)\n\
             (rule ((is-positive m)) ((is-nonzero m)) :ruleset guards)\n\
             (rule ((= e (Num n)) (>= n 0.0)) ((is-nonneg e)) :ruleset guards)",
        );
        assert_eq!(r.guards[0].child_req, vec![Facts::POSITIVE, Facts::POSITIVE]);
        assert_eq!((r.guards[1].op, r.guards[1].self_req), (None, Facts::POSITIVE));
        assert_eq!(r.guards[2].num_preds, vec![(Cmp::Ge, 0.0)]);
        assert_eq!(r.guards[2].op, Some(Op::Num));
    }

    #[test]
    fn derived_structural_relations_are_refused_with_a_reason() {
        let r = one(
            "(relation leaf (Math))\n\
             (rule ((= m (Num a))) ((leaf m)) :ruleset rational)\n\
             (rewrite (Pow2 (Add a b)) (Add a b) :when ((leaf a)) :ruleset rational)",
        );
        assert!(r.drafts.is_empty() && r.guards.is_empty());
        assert_eq!(r.refused.len(), 2);
        assert!(r.refused[0].reason.contains("leaf"));
        assert!(r.refused[1].reason.contains("leaf"));
    }

    #[test]
    fn an_unknown_form_is_a_load_error_not_a_skip() {
        assert!(read(&[("t", "(birewrite (Add a b) (Add b a) :ruleset x)")]).is_err());
        assert!(read(&[("t", "(rewrite (Frob x) x :ruleset x)")]).is_err());
        assert!(read(&[("t", "(rewrite (Mul x (Num 1.0)) y :ruleset x)")]).is_err());
    }
}
