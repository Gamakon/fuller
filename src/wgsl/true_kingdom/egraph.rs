//! ST9: equivalent programs of an expressed tree of the true kingdom, made by fuller's e-graph (egglog).
//!
//! The tree enters the e-graph as terms of six constructors, one per arity (`N0` .. `N4` carry the row id, `K` a `?_T`
//! placeholder's row and its `C_T` index), so every node is a row of the table. The rules are generated from the table
//! and exist only where both sides are rows of the same types: commutativity of `arith.add`, `arith.mul`, `arith.min`,
//! `arith.max`; `fma(a, b, c)` with `add(mul(a, b), c)`; `mix(a, b, t)` with `add(a, mul(sub(b, a), t))`; `sub(a, b)` with
//! `add(a, neg(b))`; `mul(a, a)` with `pow(a, 2.0)` (a vector's 2.0 is the splat of the scalar constant); the scalar
//! times a vector with the splat of the scalar times the vector; each per lane count the table has. A rule whose rows
//! are missing or whose types differ is not made, and [`Rules::made`] lists the rules there are. Saturation is bounded
//! (`iterations`) and the e-graph is read back through egglog's own `serialize()`.
//!
//! Many programs are then drawn from the e-graph: at each e-class the original node is kept, or with probability `p`
//! another member of the class is taken (a member whose children lead back to a class on the current path is never
//! taken, so every program is finite). Every program drawn is a tree of rows, typed by construction; the caller writes
//! it into a gene by the section 5 write-back, and a rewrite whose write-back does not fit is a named defect.
//!
//! These are equal as real-number programs; their floating-point pictures can differ (`fma` rounds once, `pow` is not
//! `x * x` bit for bit). Fitness measures the picture of each; nothing here judges them.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use egglog::prelude::exprs;
use egglog::{EGraph, SerializeConfig};

use super::{Rng, RowKind, TrueKingdom};
use crate::typed_gep::Tree;

const DATATYPE: &str = "(datatype T (N0 i64) (N1 i64 T) (N2 i64 T T) (N3 i64 T T T) (N4 i64 T T T T) (K i64 i64))\n";

/// A pattern over rows: a variable, a row with its arguments, or a constant placeholder.
#[derive(Debug, Clone)]
enum Pat {
    Var(&'static str),
    Row(u32, Vec<Pat>),
    Const(u32, u32),
}

impl Pat {
    fn text(&self) -> String {
        match self {
            Pat::Var(v) => v.to_string(),
            Pat::Const(s, i) => format!("(K {s} {i})"),
            Pat::Row(s, kids) => {
                let mut t = format!("(N{} {s}", kids.len());
                for k in kids {
                    t.push(' ');
                    t.push_str(&k.text());
                }
                t.push(')');
                t
            }
        }
    }
}

/// The rules generated from a table.
#[derive(Debug, Clone)]
pub struct Rules {
    /// `(name, egglog command)`.
    rules: Vec<(String, String)>,
}

impl Rules {
    /// The rule names made from this table, in order.
    pub fn made(&self) -> Vec<String> {
        self.rules.iter().map(|(n, _)| n.clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Every rule whose rows exist with the same types on both sides.
    pub fn from_table(k: &TrueKingdom) -> Rules {
        let id = |name: &str| k.rows.iter().position(|r| r.name == name).map(|i| i as u32);
        let sig = |s: u32| (&k.rows[s as usize].args, k.rows[s as usize].out);
        // `name` is a row of `arity` arguments, each of type `t`, outputting `t`
        let uniform = |name: &str, arity: usize| -> Option<(u32, crate::geneframe::Ty)> {
            let s = id(name)?;
            let (args, out) = sig(s);
            (args.len() == arity && args.iter().all(|a| *a == out)).then_some((s, out))
        };
        let mut rules = Vec::new();
        let (a, b, c) = (Pat::Var("a"), Pat::Var("b"), Pat::Var("c"));
        let two = k.rows.iter().position(|r| r.name == "literal.real" && r.kind == RowKind::Constant).and_then(|lit| k.constants.get(&k.rows[lit].out).and_then(|v| v.iter().position(|t| t == "2.0")).map(|i| (lit as u32, i as u32)));
        for (sfx, lanes) in [("", 1u8), ("_v2", 2), ("_v3", 3), ("_v4", 4)] {
            let op = |o: &str, n: usize| uniform(&format!("arith.{o}{sfx}"), n);
            let (add, mul, sub, neg, min, max, fma, mix, pow) = (op("add", 2), op("mul", 2), op("sub", 2), op("neg", 1), op("min", 2), op("max", 2), op("fma", 3), op("mix", 3), op("pow", 2));
            for (name, row) in [("add", add), ("mul", mul), ("min", min), ("max", max)] {
                if let Some((s, _)) = row {
                    rules.push((format!("commute.{name}{sfx}"), format!("(rewrite {} {})", Pat::Row(s, vec![a.clone(), b.clone()]).text(), Pat::Row(s, vec![b.clone(), a.clone()]).text())));
                }
            }
            let same = |xs: &[Option<(u32, crate::geneframe::Ty)>]| -> Option<Vec<u32>> {
                let t = xs.first()?.as_ref()?.1;
                xs.iter().map(|x| x.filter(|(_, ty)| *ty == t).map(|(s, _)| s)).collect()
            };
            if let Some(v) = same(&[fma, add, mul]) {
                let (f, ad, mu) = (v[0], v[1], v[2]);
                rules.push((format!("fma{sfx}"), format!("(birewrite {} {})", Pat::Row(f, vec![a.clone(), b.clone(), c.clone()]).text(), Pat::Row(ad, vec![Pat::Row(mu, vec![a.clone(), b.clone()]), c.clone()]).text())));
            }
            if let Some(v) = same(&[mix, add, mul, sub]) {
                let (mx, ad, mu, su) = (v[0], v[1], v[2], v[3]);
                rules.push((format!("mix{sfx}"), format!("(birewrite {} {})", Pat::Row(mx, vec![a.clone(), b.clone(), c.clone()]).text(), Pat::Row(ad, vec![a.clone(), Pat::Row(mu, vec![Pat::Row(su, vec![b.clone(), a.clone()]), c.clone()])]).text())));
            }
            if let Some(v) = same(&[sub, add, neg]) {
                let (su, ad, ne) = (v[0], v[1], v[2]);
                rules.push((format!("sub{sfx}"), format!("(birewrite {} {})", Pat::Row(su, vec![a.clone(), b.clone()]).text(), Pat::Row(ad, vec![a.clone(), Pat::Row(ne, vec![b.clone()])]).text())));
            }
            // the constant 2.0 of this lane count: the scalar placeholder, or its splat
            let two_of = || -> Option<Pat> {
                let (lit, i) = two?;
                if lanes == 1 {
                    return Some(Pat::Const(lit, i));
                }
                let splat = id(&format!("shape.splat{lanes}"))?;
                let (args, out) = sig(splat);
                let want = mul.map(|(_, t)| t)?;
                (args.len() == 1 && args[0] == k.rows[lit as usize].out && out == want).then(|| Pat::Row(splat, vec![Pat::Const(lit, i)]))
            };
            if let (Some(v), Some(t2)) = (same(&[mul, pow]), two_of()) {
                let (mu, pw) = (v[0], v[1]);
                rules.push((format!("square{sfx}"), format!("(birewrite {} {})", Pat::Row(mu, vec![a.clone(), a.clone()]).text(), Pat::Row(pw, vec![a.clone(), t2]).text())));
            }
            if lanes > 1 {
                let smul = id(&format!("arith.mul_scalar_vec{sfx}"));
                let splat = id(&format!("shape.splat{lanes}"));
                if let (Some(sm), Some(sp), Some((mu, t))) = (smul, splat, mul) {
                    let (sargs, sout) = sig(sm);
                    let (pargs, pout) = sig(sp);
                    if sargs.len() == 2 && sargs[1] == t && sout == t && pargs.len() == 1 && pargs[0] == sargs[0] && pout == t {
                        rules.push((format!("scale{sfx}"), format!("(birewrite {} {})", Pat::Row(sm, vec![a.clone(), b.clone()]).text(), Pat::Row(mu, vec![Pat::Row(sp, vec![a.clone()]), b.clone()]).text())));
                    }
                }
            }
        }
        Rules { rules }
    }
}

/// The e-graph of one tree after bounded saturation, read back from egglog's `serialize()`.
pub struct Equivalents {
    /// The member nodes of each class, in a fixed order.
    classes: BTreeMap<String, Vec<Member>>,
    /// The class of every node of the original tree, by its pre-order position.
    original_class: Vec<String>,
    /// The original tree's node at each class it occupies (the first occurrence).
    original_member: HashMap<String, Member>,
    root: String,
    /// Classes with more than one member: the places a draw can change.
    pub choice_classes: usize,
    pub classes_total: usize,
    pub iterations: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Member {
    symbol: u32,
    const_index: Option<u32>,
    kids: Vec<String>,
}

/// The tree as an egglog term.
fn tree_text(t: &Tree) -> String {
    match t.const_index {
        Some(i) => format!("(K {} {i})", t.symbol),
        None => {
            let mut s = format!("(N{} {}", t.kids.len(), t.symbol);
            for c in &t.kids {
                s.push(' ');
                s.push_str(&tree_text(c));
            }
            s.push(')');
            s
        }
    }
}

impl Equivalents {
    /// Saturate `tree` under `rules` for at most `iterations` iterations (stopping early at `max_tuples`).
    pub fn build(k: &TrueKingdom, rules: &Rules, tree: &Tree, iterations: usize, max_tuples: usize) -> Result<Equivalents, String> {
        let mut eg = EGraph::default();
        eg.parse_and_run_program(None, DATATYPE).map_err(|e| format!("our software has a bug: the e-graph datatype: {e}"))?;
        for (name, r) in &rules.rules {
            eg.parse_and_run_program(None, &format!("{r}\n")).map_err(|e| format!("our software has a bug: rule {name}: {e}"))?;
        }
        eg.parse_and_run_program(None, &format!("(let root {})\n", tree_text(tree))).map_err(|e| format!("our software has a bug: the tree as a term: {e}"))?;
        let mut done = 0;
        while done < iterations {
            eg.parse_and_run_program(None, "(run 1)\n").map_err(|e| format!("our software has a bug: saturation: {e}"))?;
            done += 1;
            if eg.num_tuples() > max_tuples {
                break;
            }
        }
        let (sort, value) = eg.eval_expr(&exprs::var("root")).map_err(|e| format!("our software has a bug: the root: {e}"))?;
        let out = eg.serialize(SerializeConfig { max_functions: None, max_calls_per_function: None, include_temporary_functions: false, root_eclasses: vec![(sort, value)] });
        let ser = out.egraph;
        // the i64 literal of a child node
        let int_of = |nid: &egraph_serialize::NodeId| -> Result<u32, String> {
            let n = ser.nodes.get(nid).ok_or_else(|| format!("our software has a bug: serialized node {nid} is missing"))?;
            n.op.parse::<u32>().map_err(|e| format!("our software has a bug: {:?} is not a row id: {e}", n.op))
        };
        let mut classes: BTreeMap<String, BTreeSet<Member>> = BTreeMap::new();
        for n in ser.nodes.values() {
            let arity = match n.op.as_str() {
                "N0" => 0,
                "N1" => 1,
                "N2" => 2,
                "N3" => 3,
                "N4" => 4,
                "K" => usize::MAX,
                _ => continue,
            };
            let m = if arity == usize::MAX {
                Member { symbol: int_of(&n.children[0])?, const_index: Some(int_of(&n.children[1])?), kids: Vec::new() }
            } else {
                if n.children.len() != arity + 1 {
                    return Err(format!("our software has a bug: {} with {} children", n.op, n.children.len()));
                }
                let kids = n.children[1..].iter().map(|c| ser.nodes.get(c).map(|x| x.eclass.to_string()).ok_or_else(|| format!("our software has a bug: child {c} missing"))).collect::<Result<Vec<_>, _>>()?;
                Member { symbol: int_of(&n.children[0])?, const_index: None, kids }
            };
            if k.rows.get(m.symbol as usize).is_none_or(|r| r.args.len() != m.kids.len()) {
                return Err(format!("our software has a bug: the e-graph holds row {} with {} arguments", m.symbol, m.kids.len()));
            }
            classes.entry(n.eclass.to_string()).or_default().insert(m);
        }
        let root = ser.root_eclasses.first().map(|c| c.to_string()).ok_or("our software has a bug: no root class")?;
        // the class of each original node: its member key, bottom up
        let by_member: HashMap<Member, String> = classes.iter().flat_map(|(c, ms)| ms.iter().map(move |m| (m.clone(), c.clone()))).collect();
        let mut original_class = Vec::new();
        let mut original_member = HashMap::new();
        fn place(t: &Tree, by: &HashMap<Member, String>, out: &mut Vec<String>, first: &mut HashMap<String, Member>) -> Result<String, String> {
            let at = out.len();
            out.push(String::new());
            let kids = t.kids.iter().map(|c| place(c, by, out, first)).collect::<Result<Vec<_>, _>>()?;
            let m = Member { symbol: t.symbol, const_index: t.const_index, kids };
            let c = by.get(&m).cloned().ok_or_else(|| format!("our software has a bug: the original node {} is in no class", t.symbol))?;
            first.entry(c.clone()).or_insert(m);
            out[at] = c.clone();
            Ok(c)
        }
        let rc = place(tree, &by_member, &mut original_class, &mut original_member)?;
        if rc != root {
            return Err("our software has a bug: the original tree's root is not the root class".into());
        }
        let classes: BTreeMap<String, Vec<Member>> = classes.into_iter().map(|(c, ms)| (c, ms.into_iter().collect())).collect();
        let choice_classes = classes.values().filter(|ms| ms.len() > 1).count();
        Ok(Equivalents { classes_total: classes.len(), classes, original_class, original_member, root, choice_classes, iterations: done })
    }

    /// Classes of the original tree's nodes that have another member: where a draw can differ from the original.
    pub fn changeable_original_nodes(&self) -> usize {
        self.original_class.iter().filter(|c| self.classes.get(*c).is_some_and(|ms| ms.len() > 1)).count()
    }

    /// One program: at each class the original member is kept, or with probability `p` (always, at a class the
    /// original does not occupy) a member drawn uniformly from those that do not lead back to the current path.
    pub fn draw(&self, rng: &mut Rng, p: f64) -> Result<Tree, String> {
        let mut path = Vec::new();
        self.draw_class(&self.root, rng, p, &mut path)
    }

    fn draw_class(&self, class: &str, rng: &mut Rng, p: f64, path: &mut Vec<String>) -> Result<Tree, String> {
        let members = self.classes.get(class).ok_or_else(|| format!("our software has a bug: class {class} has no members"))?;
        path.push(class.to_string());
        let open: Vec<&Member> = members.iter().filter(|m| m.kids.iter().all(|c| !path.contains(c))).collect();
        if open.is_empty() {
            return Err(format!("our software has a bug: every member of class {class} leads back to the path"));
        }
        let original = self.original_member.get(class).filter(|m| open.contains(m));
        let m = match original {
            Some(o) if open.len() == 1 || rng.unit() >= p => o,
            _ => open[rng.below(open.len())],
        };
        let kids = m.kids.iter().map(|c| self.draw_class(c, rng, p, path)).collect::<Result<Vec<_>, _>>()?;
        path.pop();
        Ok(Tree { symbol: m.symbol, const_index: m.const_index, kids })
    }
}

/// Up to `want` distinct programs equal to `tree` and different from it, drawn with per-class change probability `p`
/// in at most `attempts` draws, in the order drawn.
pub fn variants(eq: &Equivalents, tree: &Tree, want: usize, p: f64, attempts: usize, seed: u64) -> Result<Vec<Tree>, String> {
    let mut rng = Rng(seed);
    let mut seen: BTreeSet<String> = BTreeSet::new();
    seen.insert(tree_text(tree));
    let mut out = Vec::new();
    for _ in 0..attempts {
        if out.len() >= want {
            break;
        }
        let t = eq.draw(&mut rng, p)?;
        if seen.insert(tree_text(&t)) {
            out.push(t);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_gep::{express, write_back, GeneLayout};
    use crate::wgsl::cuneus_originals::{image_dir, volumepassage};
    use crate::wgsl::true_kingdom::{random_gene, Share};

    fn typed(k: &TrueKingdom, t: &Tree) -> bool {
        let row = &k.rows[t.symbol as usize];
        row.args.len() == t.kids.len() && t.kids.iter().zip(&row.args).all(|(c, a)| k.rows[c.symbol as usize].out == *a && typed(k, c))
    }

    /// The rules of the volumepassage table: commutativity, fma, mix, sub, square and scale per lane count where the
    /// rows exist with one type; each rule is a row-to-row rewrite.
    #[test]
    fn the_rules_are_made_from_rows_of_the_same_types() {
        let k = TrueKingdom::load(&image_dir("volumepassage")).expect("table");
        let r = Rules::from_table(&k);
        let made = r.made();
        for want in ["commute.add", "commute.mul_v3", "fma", "fma_v3", "mix", "sub_v2", "square", "square_v3", "scale_v3"] {
            assert!(made.iter().any(|m| m == want), "{want} not made: {made:?}");
        }
        assert!(!made.iter().any(|m| m == "scale"), "no scalar form of the scalar-times-vector row");
    }

    /// The encoded volumepassage in the e-graph: the original is one of its programs, the draws are typed trees of
    /// rows different from the original, each written into a gene by the write-back and read back unchanged; a draw at
    /// `p = 0` is the original itself.
    #[test]
    fn equivalent_programs_of_volumepassage_are_typed_trees_that_write_back_into_a_gene() {
        let k = TrueKingdom::load(&image_dir("volumepassage")).expect("table");
        let c = k.relations().check().expect("check");
        let tree = volumepassage(&k).unwrap_or_else(|g| panic!("{g:?}"));
        let rules = Rules::from_table(&k);
        let eq = Equivalents::build(&k, &rules, &tree, 3, 2_000_000).expect("e-graph");
        assert!(eq.choice_classes > 0 && eq.changeable_original_nodes() > 10, "{} choice classes, {} changeable", eq.choice_classes, eq.changeable_original_nodes());
        assert_eq!(eq.draw(&mut Rng(1), 0.0).expect("draw"), tree, "p = 0 keeps every original member");
        let vs = variants(&eq, &tree, 12, 3.0 / eq.changeable_original_nodes() as f64, 400, 7).expect("variants");
        assert!(vs.len() >= 8, "{} variants", vs.len());
        let n = vs.iter().map(|t| t.level_order().iter().filter(|(x, _)| !x.kids.is_empty()).count()).max().unwrap_or(0).max(294);
        let layout = GeneLayout::from_program_size(&c, k.root, n).expect("layout");
        let cl = k.const_lens();
        let previous = random_gene(&c, &layout, &cl, &mut Rng(3), Share::Half, true);
        for v in &vs {
            assert!(typed(&k, v));
            assert_ne!(v, &tree);
            let gene = write_back(&c, &layout, &previous, v).expect("write-back");
            assert_eq!(&express(&c, &layout, &gene, &cl).expect("express").tree(), v);
            k.render(v).expect("render");
        }
        let again = variants(&eq, &tree, 12, 3.0 / eq.changeable_original_nodes() as f64, 400, 7).expect("variants");
        assert_eq!(again, vs, "deterministic for a seed");
    }
}
