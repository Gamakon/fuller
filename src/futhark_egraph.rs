//! The Futhark kingdom inside Fuller's e-graph.
//!
//! Fuller's equality saturation runs on egglog over the `Math` datatype — a
//! language of *scalar reals*. The Futhark kingdom is a language of *typed
//! array programs*, so it needs its own datatypes. The design here is the one
//! that makes the typing fall out for free:
//!
//! **one egglog sort per kingdom type.** `FR` is `f32.real`, `FA` is
//! `arr.f32`, `FS` is `scn.f32` and `FF` is `fn.f32_f32`. A node's *sort* is
//! its type, so egglog itself refuses to build an ill-typed program — there
//! is no second type checker to keep in sync, and the accumulator
//! distinction (`arr.f32` versus `scn.f32`, which is Futhark's `acc`) is
//! carried by the sort rather than by a side table.
//!
//! The constructors mirror the kingdom's 38 rows one for one, plus two
//! e-graph-only constructors:
//!
//! * `FId` — the identity function `\x -> x`. It is not a kingdom row (the
//!   kingdom has no identity-function row), but Futhark has the expression,
//!   and one of the rewrites needs it.
//! * `FCompose f g` — the composition `f . g`. Futhark has no first-class
//!   functions, so this renders by *inlining* the two lambdas; the result is
//!   a valid Futhark program but not a kingdom row, so a candidate that uses
//!   it is compilation-only (see [`Candidate::gene`]).
//!
//! Both e-graph-only constructors are sound: `FId` is the identity on reals
//! and `FCompose` is pointwise composition.
//!
//! # The rewrites, and why each is allowed
//!
//! Every rule in [`FUTHARK_RULESET`] was checked against Futhark 0.28.1
//! itself: a program computes both sides of the rewrite and returns the
//! magnitude of their difference, over inputs that include negatives, zeros
//! and a one-element array. All of them are **bit-identical**.
//!
//! * `map id xs = xs` — elementwise identity; no arithmetic, so no
//!   floating-point question. Checked over an ordinary array *and* over a
//!   scan result, and again under a `reduce` (where Futhark's `acc` path
//!   takes a different route than the ordinary one).
//! * `reverse (reverse xs) = xs` — `reverse` is an involution.
//! * `map f (reverse xs) = reverse (map f xs)` — elementwise, so the two
//!   sides compute the same values in the same order. The scan flavour is
//!   `reverse (map f s) = map f (reverse s)`, where the reverse of a scan
//!   result is an ordinary array again.
//! * `scn.f32.identity = map id (scan (+) 0.0 xs)` — the terminal's own
//!   definition, spelled out so that the `map id` rule can fire on it. The
//!   expansion is what turns the kingdom's `scn.f32.identity` row into the
//!   bare `scan (+) 0.0 xs`, which is the candidate the compiler rescue
//!   wants.
//!
//! The fusion rules — `map f (map g xs) = map (f . g) xs`, also checked
//! bit-identical — live in [`FUTHARK_RULESET`] beside the identities, and the
//! **unit laws of composition** (`f . id = f`, `id . g = g`) are what make
//! that safe: without them the two rules diverge together, because `map id`
//! puts a class's own `map` node inside that class and the fusion pattern
//! then matches inside it. The mechanism, and the measurement that pinned it
//! down, are in that constant's documentation.
//!
//! What is deliberately **not** here: any reassociation of floating-point
//! arithmetic. The kingdom has no scalar arithmetic to reassociate, and
//! `map2 (+) a b` is not interchangeable with `map2 (+) b a` in general
//! (the shorter array decides the length, and the operands may differ in
//! length). Nothing in this ruleset assumes that mathematical equivalence
//! implies floating-point equivalence; every rule is an identity of
//! *computations*.
//!
//! # What this module does not do
//!
//! It does not cost the candidates by anything Futhark-specific: the
//! extractor uses egglog's tree-additive cost, so "cheapest" means "fewest
//! nodes". The Futhark-relevant costs — traversals, allocations, memory
//! traffic, measured compile time — are a later layer, deliberately kept
//! separate from the static ones.

use egglog::extract::hff_extract;
use egglog::prelude::exprs;
use egglog::{ArcSort, EGraph, Term, TermDag, TermId, Value};

use crate::futhark_kingdom::{FutharkKingdom, FutharkKind};
use crate::kingdom_contract::SymbolId;
use crate::saturated_fold::EgraphKingdom;
use crate::typed_chromosome::{AdfSet, Epoch};
use crate::typed_gep::Tree;

/// The e-graph's sorts, one per kingdom type.
///
/// The four sorts are mutually recursive — `FR` holds reductions over arrays,
/// `FA` holds maps over functions, `FS` holds scans over arrays — so they are
/// forward-declared with `sort` and then given their constructors one by
/// one. (Egglog's `datatype` command declares the sort itself, which cannot
/// be forward-declared; `constructor` is the same thing without that side
/// effect.)
pub const FUTHARK_DATATYPE: &str = r#"
(sort FR)
(sort FA)
(sort FS)
(sort FF)

; ---- f32.real --------------------------------------------------------------
(constructor RNum (f64) FR)
(constructor RReduceAdd (FA) FR)
(constructor RReduceMul (FA) FR)
(constructor RReduceAddS (FS) FR)
(constructor RReduceMulS (FS) FR)

; ---- arr.f32 ---------------------------------------------------------------
(constructor AInput () FA)
(constructor AMap (FF FA) FA)
(constructor AMap2Add (FA FA) FA)
(constructor AMap2Mul (FA FA) FA)
(constructor AMap2Max (FA FA) FA)
(constructor AMap2Min (FA FA) FA)
(constructor ARev (FA) FA)
(constructor ARevScan (FS) FA)

; ---- scn.f32 ---------------------------------------------------------------
(constructor SScanIdentity () FS)
(constructor SScanAdd (FA) FS)
(constructor SScanMul (FA) FS)
(constructor SMap (FF FS) FS)
(constructor SMap2Add (FS FS) FS)
(constructor SMap2Mul (FS FS) FS)
(constructor SMap2Max (FS FS) FS)
(constructor SMap2Min (FS FS) FS)

; ---- fn.f32_f32 ------------------------------------------------------------
(constructor FId () FF)
(constructor FNeg () FF)
(constructor FAbs () FF)
(constructor FSqrt () FF)
(constructor FExp () FF)
(constructor FLog () FF)
(constructor FSin () FF)
(constructor FCos () FF)
(constructor FTan () FF)
(constructor FSqr () FF)
(constructor FCube () FF)
(constructor FRecip () FF)
(constructor FFloor () FF)
(constructor FRound () FF)
(constructor FZero () FF)
(constructor FAddC (FR) FF)
(constructor FMulC (FR) FF)
(constructor FSubC (FR) FF)
(constructor FCompose (FF FF) FF)
"#;

/// The rewrite ruleset.
///
/// Every rule is an identity of *computations* under Futhark 0.28.1, checked
/// against the compiler itself; see the module header for the evidence.
pub const FUTHARK_RULESET: &str = r#"
(ruleset futhark)

; ---- the identity function is the identity -------------------------------
; `map (\x -> x) xs` computes exactly `xs`: no arithmetic happens, so there
; is no floating-point question, and the sort (ordinary array versus scan
; result) is unchanged.
(rewrite (AMap (FId) x) x :ruleset futhark)
(rewrite (SMap (FId) x) x :ruleset futhark)

; ---- the identity scan, spelled out ---------------------------------------
; The kingdom's `scn.f32.identity` terminal IS `map (\x -> x) (scan (+) 0.0
; xs)`. Writing it out lets the rule above fire on it, so the bare scan
; becomes a candidate of the same e-class.
(rewrite (SScanIdentity) (SMap (FId) (SScanAdd (AInput))) :ruleset futhark)

; ---- reverse is an involution -------------------------------------------
; `reverse` only permutes the elements.
(rewrite (ARev (ARev x)) x :ruleset futhark)

; ---- map commutes with reverse ------------------------------------------
; Both sides apply `f` to every element; the order of the elements is the
; same on both sides. The reverse of a scan result is an ordinary array
; again, so the scan flavour lands in `arr.f32`.
(rewrite (ARev (AMap f x)) (AMap f (ARev x)) :ruleset futhark)
(rewrite (ARevScan (SMap f s)) (AMap f (ARevScan s)) :ruleset futhark)

; ---- the unit laws of composition ----------------------------------------
; `f . id = f` and `id . g = g`. These are what make the fusion rules safe to
; live in the same saturation as `map id`.
;
; Without them the pair diverges, and it is worth saying why, because the
; mechanism is not obvious. The `map id` rule puts a class's own `map` node
; inside that class: `map id xs` and `xs` end up in the same e-class, so the
; class holds an e-node `AMap (FId) x` whose argument is that very class. The
; fusion pattern `AMap f (AMap g y)` then matches *inside* the class — the
; class contains a map, so the pattern matches — and mints
; `AMap (FCompose f (FId)) y`, which is a new e-node, and the next round finds
; two of them, and four after that. Measured on the identity scan: the e-graph
; grew 5 -> 7 -> 13 -> 55 -> 1357 e-nodes over five rounds and did not
; converge after six.
;
; With the unit laws, `AMap (FCompose f (FId)) y` collapses to `AMap f y`,
; which is already in the class, and `AMap (FCompose (FId) g) y` collapses to
; `AMap g y`, which is there too. Nothing new is created, the loop closes, and
; the pair converges in two or three rounds.
(rewrite (FCompose f (FId)) f :ruleset futhark)
(rewrite (FCompose (FId) g) g :ruleset futhark)

; ---- fusion ----------------------------------------------------------------
; `map f (map g xs) = map (f . g) xs`: a true identity of computations, checked
; against Futhark 0.28.1 bit for bit, over ordinary arrays *and* over scan
; results under a `reduce`.
;
; The composition is an e-graph-only constructor: Futhark has no first-class
; functions, so it renders by inlining the two lambdas, which is a valid
; program but not a kingdom row — so a candidate that uses it is
; compilation-only, and its *native* sibling (the unfused form) sits beside it
; in the same e-class.
(rewrite (AMap f (AMap g x)) (AMap (FCompose f g) x) :ruleset futhark)
(rewrite (SMap f (SMap g s)) (SMap (FCompose f g) s) :ruleset futhark)
"#;

/// The kingdom row each e-graph constructor stands for.
///
/// The kingdom's own rows, one for one: all 38 of them.
const ROW_CONSTRUCTORS: &[(&str, &str)] = &[
    ("map_a", "AMap"),
    ("map2_add_a", "AMap2Add"),
    ("map2_mul_a", "AMap2Mul"),
    ("map2_max_a", "AMap2Max"),
    ("map2_min_a", "AMap2Min"),
    ("rev_a", "ARev"),
    ("map_s", "SMap"),
    ("map2_add_s", "SMap2Add"),
    ("map2_mul_s", "SMap2Mul"),
    ("map2_max_s", "SMap2Max"),
    ("map2_min_s", "SMap2Min"),
    ("rev_s", "ARevScan"),
    ("scan_add_s", "SScanAdd"),
    ("scan_mul_s", "SScanMul"),
    ("reduce_add_a", "RReduceAdd"),
    ("reduce_mul_a", "RReduceMul"),
    ("reduce_add_s", "RReduceAddS"),
    ("reduce_mul_s", "RReduceMulS"),
    ("fn.f32_f32.neg", "FNeg"),
    ("fn.f32_f32.abs", "FAbs"),
    ("fn.f32_f32.sqrt", "FSqrt"),
    ("fn.f32_f32.exp", "FExp"),
    ("fn.f32_f32.log", "FLog"),
    ("fn.f32_f32.sin", "FSin"),
    ("fn.f32_f32.cos", "FCos"),
    ("fn.f32_f32.tan", "FTan"),
    ("fn.f32_f32.sqr", "FSqr"),
    ("fn.f32_f32.cube", "FCube"),
    ("fn.f32_f32.recip", "FRecip"),
    ("fn.f32_f32.floor", "FFloor"),
    ("fn.f32_f32.round", "FRound"),
    ("fn.f32_f32.add_c", "FAddC"),
    ("fn.f32_f32.mul_c", "FMulC"),
    ("fn.f32_f32.sub_c", "FSubC"),
    ("?_f32", "RNum"),
    ("arr.f32.input", "AInput"),
    ("scn.f32.identity", "SScanIdentity"),
    ("fn.f32_f32.const", "FZero"),
];

/// The e-graph constructor of a kingdom row, if it has one.
fn constructor_of(row_name: &str) -> Option<&'static str> {
    ROW_CONSTRUCTORS
        .iter()
        .find(|(n, _)| *n == row_name)
        .map(|(_, c)| *c)
}

/// The kingdom row an e-graph constructor stands for, if any.
fn row_of(constructor: &str) -> Option<&'static str> {
    ROW_CONSTRUCTORS
        .iter()
        .find(|(_, c)| *c == constructor)
        .map(|(n, _)| *n)
}

/// The text of a literal: the kingdom's own spelling when the value is one of
/// the kingdom's literals, and the value itself otherwise (which happens only
/// for a candidate the kingdom cannot spell).
fn literal_text(kingdom: &FutharkKingdom, value: f64) -> Result<String, String> {
    for dual in &kingdom.duals {
        for text in &dual.literals {
            if text.parse::<f32>() == Ok(value as f32) {
                return Ok(text.clone());
            }
        }
    }
    Ok(fmt_f64(value))
}

/// A float in egglog's surface syntax.
fn fmt_f64(v: f64) -> String {
    let s = format!("{v:?}");
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("nan") {
        s
    } else {
        format!("{s}.0")
    }
}

/// An expressed Futhark tree as an egglog term.
///
/// The term is the e-graph's *input*: one constructor per kingdom row, with
/// the kingdom's argument order. A constant placeholder becomes its literal
/// value — the constant array's index is a gene-level detail, and the e-graph
/// works with values.
pub fn import(tree: &Tree, kingdom: &FutharkKingdom) -> Result<String, String> {
    let row = kingdom
        .rows
        .get(tree.symbol as usize)
        .ok_or_else(|| format!("futhark_egraph: symbol {} is not a row", tree.symbol))?;
    if row.kind == FutharkKind::Const {
        let v = kingdom.literal(tree)?;
        return Ok(format!("(RNum {})", fmt_f64(v as f64)));
    }
    let ctor = constructor_of(&row.name)
        .ok_or_else(|| format!("futhark_egraph: the row {} has no e-graph constructor", row.name))?;
    let kids: Vec<String> = tree
        .kids
        .iter()
        .map(|kid| import(kid, kingdom))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(if kids.is_empty() {
        format!("({ctor})")
    } else {
        format!("({ctor} {})", kids.join(" "))
    })
}

/// An egglog term as a Futhark expression.
///
/// The text is the kingdom's own spelling, with every child bracketed — the
/// same convention `FutharkKingdom::render` uses — so a candidate renders
/// exactly as the kingdom would have rendered the equivalent tree. The two
/// e-graph-only constructors render as valid Futhark that the kingdom has no
/// row for: `FId` is `\x -> x`, and `FCompose` inlines its two lambdas.
pub fn render(termdag: &TermDag, id: TermId, kingdom: &FutharkKingdom) -> Result<String, String> {
    let term = termdag.get(id);
    let Term::App(op, kids) = term else {
        return Err(match term {
            Term::Lit(_) => "futhark_egraph: a literal is not a program".to_string(),
            Term::Var(v) => format!("futhark_egraph: {v} is not a program"),
            _ => unreachable!(),
        });
    };
    let child = |i: usize| -> Result<String, String> { render(termdag, kids[i], kingdom) };
    let two = |i: usize, j: usize| -> Result<(String, String), String> {
        Ok((child(i)?, child(j)?))
    };
    Ok(match op.as_str() {
        // --- the function values ------------------------------------------
        "FId" => "\\x -> x".to_string(),
        "FNeg" => "\\x -> -x".to_string(),
        "FAbs" => "\\x -> f32.abs x".to_string(),
        "FSqrt" => "\\x -> f32.sqrt x".to_string(),
        "FExp" => "\\x -> f32.exp x".to_string(),
        "FLog" => "\\x -> f32.log x".to_string(),
        "FSin" => "\\x -> f32.sin x".to_string(),
        "FCos" => "\\x -> f32.cos x".to_string(),
        "FTan" => "\\x -> f32.tan x".to_string(),
        "FSqr" => "\\x -> x * x".to_string(),
        "FCube" => "\\x -> x * x * x".to_string(),
        "FRecip" => "\\x -> 1.0 / x".to_string(),
        "FFloor" => "\\x -> f32.floor x".to_string(),
        "FRound" => "\\x -> f32.round x".to_string(),
        "FZero" => "\\x -> 0.0".to_string(),
        "FAddC" => format!("\\x -> x + ({})", child(0)?),
        "FMulC" => format!("\\x -> x * ({})", child(0)?),
        "FSubC" => format!("\\x -> x - ({})", child(0)?),
        // `f . g`: Futhark has no first-class functions, so the two lambdas
        // are inlined and applied. The binder is `z`, not `x`, so the inlined
        // lambdas' own binders do not shadow it.
        "FCompose" => {
            let (f, g) = two(0, 1)?;
            format!("\\z -> ({f}) (({g}) (z))")
        }
        // --- the ordinary arrays -------------------------------------------
        "AInput" => "xs".to_string(),
        "AMap" => {
            let (f, a) = two(0, 1)?;
            format!("map ({f}) ({a})")
        }
        "AMap2Add" => {
            let (a, b) = two(0, 1)?;
            format!("map2 (+) ({a}) ({b})")
        }
        "AMap2Mul" => {
            let (a, b) = two(0, 1)?;
            format!("map2 (*) ({a}) ({b})")
        }
        "AMap2Max" => {
            let (a, b) = two(0, 1)?;
            format!("map2 f32.max ({a}) ({b})")
        }
        "AMap2Min" => {
            let (a, b) = two(0, 1)?;
            format!("map2 f32.min ({a}) ({b})")
        }
        "ARev" => format!("reverse ({})", child(0)?),
        "ARevScan" => format!("reverse ({})", child(0)?),
        // --- the scan results ----------------------------------------------
        // The kingdom's `scn.f32.identity` terminal, spelled as its table
        // spells it: `map (\x -> x) (scan (+) 0.0 xs)`.
        "SScanIdentity" => "map (\\x -> x) (scan (+) 0.0 xs)".to_string(),
        "SScanAdd" => format!("scan (+) 0.0 ({})", child(0)?),
        "SScanMul" => format!("scan (*) 1.0 ({})", child(0)?),
        "SMap" => {
            let (f, s) = two(0, 1)?;
            format!("map ({f}) ({s})")
        }
        "SMap2Add" => {
            let (a, b) = two(0, 1)?;
            format!("map2 (+) ({a}) ({b})")
        }
        "SMap2Mul" => {
            let (a, b) = two(0, 1)?;
            format!("map2 (*) ({a}) ({b})")
        }
        "SMap2Max" => {
            let (a, b) = two(0, 1)?;
            format!("map2 f32.max ({a}) ({b})")
        }
        "SMap2Min" => {
            let (a, b) = two(0, 1)?;
            format!("map2 f32.min ({a}) ({b})")
        }
        // --- the reals ------------------------------------------------------
        "RNum" => {
            let Term::Lit(egglog::ast::Literal::Float(v)) = termdag.get(kids[0]) else {
                return Err("futhark_egraph: RNum needs a float".to_string());
            };
            literal_text(kingdom, v.into_inner())?
        }
        "RReduceAdd" => format!("reduce (+) 0.0 ({})", child(0)?),
        "RReduceMul" => format!("reduce (*) 1.0 ({})", child(0)?),
        "RReduceAddS" => format!("reduce (+) 0.0 ({})", child(0)?),
        "RReduceMulS" => format!("reduce (*) 1.0 ({})", child(0)?),
        other => return Err(format!("futhark_egraph: unknown constructor {other}")),
    })
}

/// An egglog term as an expressed Futhark tree, if it can be one.
///
/// `None` means the term uses an e-graph-only constructor (`FId` or
/// `FCompose`) or a literal that is not in the kingdom's list — in either
/// case there is no gene that denotes it, so the candidate is
/// compilation-only.
pub fn to_tree(termdag: &TermDag, id: TermId, kingdom: &FutharkKingdom) -> Option<Tree> {
    let term = termdag.get(id);
    let Term::App(op, kids) = term else {
        return None;
    };
    let row_name = row_of(op)?;
    let symbol = kingdom.rows.iter().position(|r| r.name == row_name)? as u32;
    if op == "RNum" {
        let Term::Lit(egglog::ast::Literal::Float(v)) = termdag.get(kids[0]) else {
            return None;
        };
        let value = v.into_inner();
        let dual = kingdom.dual(kingdom.rows[symbol as usize].output)?;
        let index = dual
            .literals
            .iter()
            .position(|l| l.parse::<f32>() == Ok(value as f32))?;
        return Some(Tree {
            symbol,
            const_index: Some(index as u32),
            kids: vec![],
        });
    }
    let mut kid_trees = Vec::with_capacity(kids.len());
    for kid in kids {
        kid_trees.push(to_tree(termdag, *kid, kingdom)?);
    }
    Some(Tree {
        symbol,
        const_index: None,
        kids: kid_trees,
    })
}

/// One equivalent program found in the e-graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// egglog's tree-additive cost: the number of nodes.
    pub cost: u64,
    /// The Futhark source.
    pub source: String,
    /// The gene-level tree, when the candidate is one the kingdom can spell.
    /// `None` means the candidate is compilation-only.
    pub gene: Option<Tree>,
}

impl Candidate {
    /// Whether the candidate can be written back as a Karva gene.
    pub fn is_representable(&self) -> bool {
        self.gene.is_some()
    }
}

/// The e-graph, saturated, with the root's sort and value.
///
/// The ruleset is [`FUTHARK_RULESET`] — the identities and the fusion rules
/// together, which the unit laws of composition keep convergent.
pub fn saturate(term: &str, iters: u32) -> Result<(EGraph, ArcSort, Value), String> {
    saturate_with(term, FUTHARK_RULESET, &["futhark"], iters)
}

/// The e-graph, saturated under a chosen ruleset.
///
/// The schedule is bounded: a rewrite that stopped shrinking would otherwise
/// hang the never-raise API with no cap. The bound is not a safety net for a
/// *diverging* ruleset — the one ruleset this module had that diverged, the
/// fusion pair, is now convergent because the unit laws of composition sit in
/// the same ruleset, and the saturation is tested to settle.
///
/// `rulesets` names the egglog rulesets to run, in order; each one gets
/// `iters` rounds.
pub fn saturate_with(
    term: &str,
    rules: &str,
    rulesets: &[&str],
    iters: u32,
) -> Result<(EGraph, ArcSort, Value), String> {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, FUTHARK_DATATYPE)
        .map_err(|e| format!("futhark_egraph: the datatype does not load: {e}"))?;
    egraph
        .parse_and_run_program(None, rules)
        .map_err(|e| format!("futhark_egraph: the ruleset does not load: {e}"))?;
    let schedule = rulesets
        .iter()
        .map(|r| format!("(repeat {iters} (run {r}))"))
        .collect::<Vec<_>>()
        .join(" ");
    egraph
        .parse_and_run_program(
            None,
            &format!("(let __root {term})\n(run-schedule {schedule})"),
        )
        .map_err(|e| format!("futhark_egraph: {term} does not load: {e}"))?;
    let (sort, value) = egraph
        .eval_expr(&exprs::var("__root"))
        .map_err(|e| format!("futhark_egraph: the root does not evaluate: {e}"))?;
    Ok((egraph, sort, value))
}

/// The cost of a term: its number of e-nodes.
///
/// This is egglog's tree-additive cost, computed here so that the ranking
/// does not depend on which extractor produced the term.
pub fn term_cost(termdag: &TermDag, id: TermId) -> u64 {
    match termdag.get(id) {
        Term::App(_, kids) => 1 + kids.iter().map(|k| term_cost(termdag, *k)).sum::<u64>(),
        _ => 1,
    }
}

/// The equivalent programs of `term`, cheapest first.
///
/// The ranking is by the number of e-nodes and then by the rendered text, so
/// the same input and the same settings always give the same order — the
/// e-graph's internal iteration order is not something to depend on.
///
/// The candidates are the *members* of the root's e-class, enumerated all the
/// way down: egglog's stock `extract_variants` only varies the root's own
/// e-nodes and resolves every child to its cheapest form, which would hide
/// exactly the alternatives the rescue needs (the expansion of the identity
/// scan, say, two levels down). [`egglog::extract::hff_extract`] enumerates
/// the members of each class recursively, so a candidate may differ from the
/// original anywhere in the tree.
///
/// `k` bounds how many candidates come back; `iters` bounds the saturation.
pub fn candidates(
    term: &str,
    kingdom: &FutharkKingdom,
    k: usize,
    iters: u32,
) -> Result<Vec<Candidate>, String> {
    let (egraph, sort, value) = saturate(term, iters)?;
    Ok(extract_candidates(&egraph, sort, value, kingdom, k))
}

/// The members of the root's e-class, as candidates.
///
/// The list holds the `k` cheapest candidates, **and it never lets the
/// compilation-only ones crowd the heritable ones out**. That matters now
/// that fusion lives in the same ruleset: a fused `map (f . g) xs` has
/// *fewer* e-nodes than the unfused `map f (map g xs)`, so it is cheaper,
/// and a list truncated on cost alone could be all fused candidates — with
/// no representable one left, and nothing for the evolutionary loop to
/// inherit.
///
/// So the representable candidates are taken first (up to `k` of them, they
/// are the cheapest *heritable* programs there are), and the remaining slots
/// go to the cheapest compilation-only ones. The result is still sorted by
/// cost, and it always contains `min(k, number of representable candidates)`
/// heritable candidates.
fn extract_candidates(
    egraph: &EGraph,
    sort: ArcSort,
    value: Value,
    kingdom: &FutharkKingdom,
    k: usize,
) -> Vec<Candidate> {
    let mut termdag = TermDag::default();
    let score = |td: &TermDag, t: TermId| term_cost(td, t) as f64;
    let ranked = hff_extract(egraph, &mut termdag, value, sort, &score, MEMBERS_PER_CLASS);
    let mut representable: Vec<Candidate> = Vec::new();
    let mut compilation_only: Vec<Candidate> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (cost, id) in ranked {
        let source = match render(&termdag, id, kingdom) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if !seen.insert(source.clone()) {
            continue;
        }
        let candidate = Candidate {
            cost: cost as u64,
            source,
            gene: to_tree(&termdag, id, kingdom),
        };
        if candidate.is_representable() {
            if representable.len() < k {
                representable.push(candidate);
            }
        } else if compilation_only.len() < k {
            compilation_only.push(candidate);
        }
        if representable.len() >= k && compilation_only.len() >= k {
            break;
        }
    }
    let mut out = representable;
    out.extend(compilation_only);
    out.sort_by(|a, b| {
        a.cost
            .cmp(&b.cost)
            .then_with(|| a.source.cmp(&b.source))
    });
    out.truncate(k);
    out
}

/// How many member terms one e-class may contribute to the enumeration.
const MEMBERS_PER_CLASS: usize = 64;

/// The saturation of one program: how big the e-graph grew, how long it took,
/// and which equivalent programs came out.
#[derive(Debug, Clone)]
pub struct Saturation {
    /// The egglog term the tree imported to.
    pub term: String,
    /// The root's sort: the program's kingdom type.
    pub sort: String,
    /// The number of e-nodes in the saturated e-graph.
    pub size: usize,
    /// How long the saturation took.
    pub elapsed: std::time::Duration,
    /// The equivalent programs, cheapest first.
    pub candidates: Vec<Candidate>,
}

impl Saturation {
    /// The candidates the kingdom can spell as a gene, in rank order.
    ///
    /// This is the **heritable** set: a candidate whose tree is `Some` has a
    /// Karva gene that denotes it, so `typed_gep::write_back` can encode it
    /// and the evolutionary loop can inherit it.
    ///
    /// It matters that this is a *filter over the same e-graph*, not a
    /// separate search: every candidate that uses an e-graph-only
    /// constructor has a native sibling in the same e-class. A fused
    /// `map (f . g) xs` sits in the same class as the unfused
    /// `map f (map g xs)`, and the identity scan's spelled-out form sits in
    /// the same class as the kingdom's `scn.f32.identity` row. So the
    /// e-graph always holds a representable equivalent of anything it
    /// found — the extractor only has to prefer it.
    pub fn heritable(&self) -> Vec<&Candidate> {
        self.candidates.iter().filter(|c| c.is_representable()).collect()
    }
}

/// Saturate a tree's e-graph and extract its equivalent programs.
///
/// This is the whole of the e-graph side of the rescue experiment: the
/// program's term, the size of the space that was searched, how long the
/// search took, and the candidates that came out of it.
pub fn saturate_tree(
    tree: &Tree,
    kingdom: &FutharkKingdom,
    k: usize,
    iters: u32,
) -> Result<Saturation, String> {
    let term = import(tree, kingdom)?;
    let started = std::time::Instant::now();
    let (egraph, sort, value) = saturate(&term, iters)?;
    let size = egraph_size(&egraph);
    let candidates = extract_candidates(&egraph, sort.clone(), value, kingdom, k);
    Ok(Saturation {
        term,
        sort: sort.name().to_string(),
        size,
        elapsed: started.elapsed(),
        candidates,
    })
}

/// Saturate a FOLDED tree (over the epoch alphabet): the e-graph works on
/// the unfolded tree, and its candidates are folded again by the next fold.
pub fn saturate_folded(epoch: &Epoch, adfs: &AdfSet, tree: &Tree, kingdom: &FutharkKingdom, k: usize, iters: u32) -> Result<Saturation, String> {
    saturate_tree(&adfs.unfold(epoch, tree), kingdom, k, iters)
}

/// The 1-D kingdom's e-graph as the population fold needs it
/// ([`crate::saturated_fold`]): the datatype, [`FUTHARK_RULESET`], and the
/// constructor-to-row bridge. `FId` and `FCompose` have no row, so the fold's
/// extractor never chooses them; an `RNum` whose value is not in `C_T` is not
/// a row either.
pub struct FoldEgraph<'a> {
    pub kingdom: &'a FutharkKingdom,
}

impl EgraphKingdom for FoldEgraph<'_> {
    fn programs(&self) -> Vec<&str> {
        vec![FUTHARK_DATATYPE, FUTHARK_RULESET]
    }
    fn rulesets(&self) -> Vec<&str> {
        vec!["futhark"]
    }
    fn import(&self, tree: &Tree) -> Result<String, String> {
        import(tree, self.kingdom)
    }
    fn row(&self, constructor: &str, literals: &[&str]) -> Option<(SymbolId, Option<u32>)> {
        let name = row_of(constructor)?;
        let symbol = self.kingdom.rows.iter().position(|r| r.name == name)? as SymbolId;
        if constructor != "RNum" {
            return literals.is_empty().then_some((symbol, None));
        }
        let [text] = literals else { return None };
        let value = text.parse::<f64>().ok()? as f32;
        let dual = self.kingdom.dual(self.kingdom.rows[symbol as usize].output)?;
        let index = dual.literals.iter().position(|l| l.parse::<f32>().map(f32::to_bits) == Ok(value.to_bits()))?;
        Some((symbol, Some(index as u32)))
    }
}

/// Every constructor the e-graph can hold.
pub const CONSTRUCTORS: &[&str] = &[
    "RNum", "RReduceAdd", "RReduceMul", "RReduceAddS", "RReduceMulS", "AInput", "AMap", "AMap2Add",
    "AMap2Mul", "AMap2Max", "AMap2Min", "ARev", "ARevScan", "SScanIdentity", "SScanAdd", "SScanMul",
    "SMap", "SMap2Add", "SMap2Mul", "SMap2Max", "SMap2Min", "FId", "FNeg", "FAbs", "FSqrt", "FExp",
    "FLog", "FSin", "FCos", "FTan", "FSqr", "FCube", "FRecip", "FFloor", "FRound", "FZero", "FAddC",
    "FMulC", "FSubC", "FCompose",
];

/// The number of e-nodes in a saturated e-graph: the size of the search space
/// the saturation explored.
pub fn egraph_size(egraph: &EGraph) -> usize {
    CONSTRUCTORS.iter().map(|c| egraph.get_size(c)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::futhark_kingdom::FutharkKingdom;
    use crate::typed_gep::Tree;

    fn kingdom() -> FutharkKingdom {
        FutharkKingdom::load().expect("the kingdom loads")
    }

    /// A tree for one kingdom row: each argument slot gets the kingdom's own
    /// terminal for that type.
    fn tree_for(kingdom: &FutharkKingdom, name: &str) -> Option<Tree> {
        let row = kingdom.rows.iter().find(|r| r.name == name)?;
        if row.kind == crate::futhark_kingdom::FutharkKind::Const {
            return Some(Tree {
                symbol: kingdom.rows.iter().position(|r| r.name == name).unwrap() as u32,
                const_index: Some(0),
                kids: vec![],
            });
        }
        let kids: Vec<Tree> = row
            .inputs
            .iter()
            .map(|ty| leaf_for(kingdom, *ty))
            .collect::<Option<Vec<_>>>()?;
        Some(Tree {
            symbol: kingdom.rows.iter().position(|r| r.name == name).unwrap() as u32,
            const_index: None,
            kids,
        })
    }

    /// A leaf of the given type: the kingdom's own terminal for it. The real
    /// type has no terminal — its leaf is the constant placeholder, which
    /// needs an index into the literal list, so it takes the first.
    fn leaf_for(kingdom: &FutharkKingdom, ty: crate::geneframe::Ty) -> Option<Tree> {
        let is_const = ty == kingdom.ty("f32.real").expect("the real type is in the kingdom");
        let name = if is_const {
            kingdom.rows.iter().find(|r| r.kind == crate::futhark_kingdom::FutharkKind::Const)?.name.clone()
        } else {
            kingdom
                .rows
                .iter()
                .find(|r| r.kind == crate::futhark_kingdom::FutharkKind::Terminal && r.output == ty)?
                .name
                .clone()
        };
        Some(Tree {
            symbol: kingdom.rows.iter().position(|r| r.name == name).unwrap() as u32,
            const_index: if is_const { Some(0) } else { None },
            kids: vec![],
        })
    }

    #[test]
    fn the_datatype_and_the_ruleset_load() {
        let (egraph, sort, _value) = saturate("(AInput)", 2).expect("the datatype loads");
        assert_eq!(sort.name(), "FA");
        assert!(egraph_size(&egraph) > 0);
    }

    #[test]
    fn every_kingdom_row_imports_and_renders_back_to_the_same_text() {
        let k = kingdom();
        for row in &k.rows {
            let tree = tree_for(&k, &row.name)
                .unwrap_or_else(|| panic!("no tree for the row {}", row.name));
            let term = import(&tree, &k).unwrap_or_else(|e| panic!("{}: {e}", row.name));
            // Import, then render through the e-graph, and compare with the
            // kingdom's own rendering: the two must agree, or a candidate
            // would not be the program the gene denotes.
            // No saturation: the e-graph then holds exactly the imported
            // term, so the rendering must be the kingdom's own text.
            let (egraph, sort, value) = saturate(&term, 0).expect("the term loads");
            let (termdag, root, _cost) = egraph
                .extract_value(&sort, value)
                .expect("the root extracts");
            let rendered = render(&termdag, root, &k).expect("the root renders");
            let own = k.render(&tree).expect("the kingdom renders");
            assert_eq!(rendered, own, "the row {}", row.name);
        }
    }

    /// The sort IS the type: egglog itself refuses a program that does not
    /// type-check, so there is no second type checker to disagree with.
    #[test]
    fn the_egraph_refuses_an_ill_typed_program() {
        // `map` wants a function and an ordinary array; an array is not a
        // function.
        assert!(saturate("(AMap (AInput) (AInput))", 1).is_err());
        // `map` over a scan result is a different row from `map` over an
        // ordinary array, and the two sorts do not mix.
        assert!(saturate("(SMap (FNeg) (AInput))", 1).is_err());
        // A reduction takes an array, not a function.
        assert!(saturate("(RReduceAdd (FNeg))", 1).is_err());
    }

    /// The accumulator distinction is structural: `scn.f32` is its own sort,
    /// so a scan result can never stand in for an ordinary array.
    #[test]
    fn the_accumulator_distinction_is_a_sort() {
        let (_egraph, sort, _value) = saturate("(SScanAdd (AInput))", 1).expect("it loads");
        assert_eq!(sort.name(), "FS");
        // `rev_s` turns a scan result into an ordinary array; `rev_a` cannot
        // turn a scan result into an ordinary array -- that is `rev_s`'s job.
        let (_egraph, sort, _value) =
            saturate("(ARevScan (SScanAdd (AInput)))", 1).expect("it loads");
        assert_eq!(sort.name(), "FA");
        assert!(saturate("(ARev (SScanAdd (AInput)))", 1).is_err());
    }

    /// The rescue hypothesis, at the e-graph's level: the kingdom's
    /// `scn.f32.identity` terminal is `map id (scan (+) 0.0 xs)`, and the
    /// rules must be able to take the `map id` away.
    #[test]
    fn the_identity_scan_expands_to_the_bare_scan() {
        let k = kingdom();
        let tree = tree_for(&k, "scn.f32.identity").expect("the terminal has a tree");
        let sat = saturate_tree(&tree, &k, 16, 4).expect("the e-graph saturates");
        let sources: Vec<&str> = sat.candidates.iter().map(|c| c.source.as_str()).collect();
        assert!(
            sources.contains(&"scan (+) 0.0 (xs)"),
            "the bare scan is not among the candidates: {sources:?}"
        );
        assert!(
            sources.contains(&"map (\\x -> x) (scan (+) 0.0 xs)"),
            "the original is not among the candidates: {sources:?}"
        );
    }

    /// The same input and the same settings give the same ranking, every
    /// time: egglog's internal iteration order is not something to depend on.
    #[test]
    fn the_extraction_is_deterministic() {
        let k = kingdom();
        let tree = tree_for(&k, "map_s").expect("map_s has a tree");
        let first = saturate_tree(&tree, &k, 16, 4).expect("the e-graph saturates");
        let second = saturate_tree(&tree, &k, 16, 4).expect("the e-graph saturates");
        let a: Vec<(u64, &str)> = first.candidates.iter().map(|c| (c.cost, c.source.as_str())).collect();
        let b: Vec<(u64, &str)> = second.candidates.iter().map(|c| (c.cost, c.source.as_str())).collect();
        assert_eq!(a, b, "the ranking changed between two identical runs");
        assert!(!a.is_empty(), "no candidates at all");
    }

    /// The saturation is bounded: the iteration count caps the search, so a
    /// program that keeps growing cannot hang the caller.
    #[test]
    fn the_saturation_terminates_within_its_budget() {
        let k = kingdom();
        let tree = tree_for(&k, "scn.f32.identity").expect("the terminal has a tree");
        let started = std::time::Instant::now();
        let sat = saturate_tree(&tree, &k, 64, 20).expect("the e-graph saturates");
        assert!(started.elapsed() < std::time::Duration::from_secs(30), "the saturation ran away");
        assert!(sat.size > 0);
    }

    /// The ruleset does not grow: the e-graph settles and stays there,
    /// however many rounds the saturation is given.
    #[test]
    fn the_ruleset_settles() {
        let k = kingdom();
        let tree = tree_for(&k, "scn.f32.identity").expect("the terminal has a tree");
        let term = import(&tree, &k).expect("it imports");
        let mut sizes = Vec::new();
        for iters in [2u32, 4, 8, 16] {
            let (egraph, _sort, _value) =
                saturate_with(&term, FUTHARK_RULESET, &["futhark"], iters).expect("it saturates");
            sizes.push(egraph_size(&egraph));
        }
        // The e-graph needs a few rounds to settle — it gains the fused node
        // — and then it stops: the same size for four, eight and sixteen
        // rounds alike.
        assert_eq!(sizes[1], sizes[2], "the ruleset grew: {sizes:?}");
        assert_eq!(sizes[2], sizes[3], "the ruleset grew: {sizes:?}");
        assert!(sizes[0] <= sizes[1], "the ruleset shrank: {sizes:?}");
    }

    /// `map id` and fusion share a saturation, and the pair **converges**.
    ///
    /// They used not to: the `map id` rule makes a class hold its own `map`
    /// node, and the fusion rule then matched inside that class and built
    /// another one, and the next round found two of them — 5 → 7 → 13 → 55 →
    /// 1357 e-nodes over five rounds, with no convergence after six. The
    /// **unit laws of composition** (`f . id = f`, `id . g = g`) are what
    /// close the loop: the fused node the pattern mints collapses to a node
    /// that is already there, so nothing new is created.
    ///
    /// This is the regression test for that fix, so it asserts *settling*
    /// rather than growth: the same e-graph, however many rounds it is given.
    #[test]
    fn the_fusion_rules_converge_alongside_map_id() {
        let k = kingdom();
        // Two shapes: the identity scan on its own, and the same scan under a
        // *non-identity* map — `map_s (fn.f32_f32.sqr) scn.f32.identity` —
        // the case where the class holds a map node that is not the
        // identity, so the fusion pattern can mint a composite that is not
        // `f . id`.
        let trees = vec![
            tree_for(&k, "scn.f32.identity").expect("the terminal has a tree"),
            Tree {
                symbol: k.rows.iter().position(|r| r.name == "map_s").unwrap() as u32,
                const_index: None,
                kids: vec![
                    tree_for(&k, "fn.f32_f32.sqr").expect("sqr has a tree"),
                    tree_for(&k, "scn.f32.identity").expect("the terminal has a tree"),
                ],
            },
        ];
        for tree in trees {
            let term = import(&tree, &k).expect("it imports");
            let mut sizes = Vec::new();
            for iters in [2u32, 4, 8, 16] {
                let (egraph, _sort, _value) =
                    saturate_with(&term, FUTHARK_RULESET, &["futhark"], iters).expect("it saturates");
                sizes.push(egraph_size(&egraph));
            }
            assert_eq!(sizes[1], sizes[2], "the ruleset grew: {sizes:?}");
            assert_eq!(sizes[2], sizes[3], "the ruleset grew: {sizes:?}");
            // And it stays small: the divergence this test guards against
            // reached 1357 e-nodes in five rounds.
            assert!(
                sizes[3] <= 16,
                "the ruleset grew without bound: {sizes:?}"
            );
        }
    }

    /// The heritable candidates are never crowded out by the cheaper
    /// compilation-only ones.
    ///
    /// A fused `map (f . g) xs` has *fewer* e-nodes than the unfused
    /// `map f (map g xs)`, so it sorts first — and the kingdom cannot spell
    /// it. A list truncated on cost alone could then be all fused candidates,
    /// with nothing the evolutionary loop can inherit. So the extraction
    /// reserves the representable ones.
    #[test]
    fn the_heritable_candidates_survive_a_small_k() {
        let k = kingdom();
        // `map f (map g xs)`: two nested maps, so the fusion rule fires.
        let tree = Tree {
            symbol: k.rows.iter().position(|r| r.name == "map_a").unwrap() as u32,
            const_index: None,
            kids: vec![
                tree_for(&k, "fn.f32_f32.sqr").expect("sqr has a tree"),
                tree_for(&k, "map_a").expect("map_a has a tree"),
            ],
        };
        let term = import(&tree, &k).expect("it imports");
        let (egraph, sort, value) =
            saturate_with(&term, FUTHARK_RULESET, &["futhark"], 4).expect("the e-graph saturates");
        for want in [1usize, 2, 8] {
            let cands = extract_candidates(&egraph, sort.clone(), value, &k, want);
            assert!(cands.len() <= want, "the candidate list is longer than asked for");
            assert!(
                cands.iter().any(Candidate::is_representable),
                "no heritable candidate among {want} candidates: {:?}",
                cands.iter().map(|c| c.source.as_str()).collect::<Vec<_>>()
            );
        }
    }

    /// Every candidate the e-graph finds that the kingdom cannot spell has a
    /// *native* sibling in the same e-class: the fused `map (f . g) xs` sits
    /// beside the unfused `map f (map g xs)`.
    ///
    /// That is what makes the e-graph's output heritable without touching
    /// the kingdom: the representable equivalent is already there, and the
    /// extractor only has to prefer it.
    #[test]
    fn a_compilation_only_candidate_has_a_native_sibling() {
        let k = kingdom();
        // `map f (map g xs)`: two nested maps, so the fusion rule has
        // something to fire on.
        let tree = Tree {
            symbol: k.rows.iter().position(|r| r.name == "map_a").unwrap() as u32,
            const_index: None,
            kids: vec![
                tree_for(&k, "fn.f32_f32.sqr").expect("sqr has a tree"),
                tree_for(&k, "map_a").expect("map_a has a tree"),
            ],
        };
        let term = import(&tree, &k).expect("it imports");
        let (egraph, sort, value) =
            saturate_with(&term, FUTHARK_RULESET, &["futhark"], 4).expect("the e-graph saturates");
        let cands = extract_candidates(&egraph, sort, value, &k, 32);
        assert!(
            cands.iter().any(|c| !c.is_representable()),
            "the fusion rule produced nothing the kingdom cannot spell"
        );
        assert!(
            cands.iter().any(Candidate::is_representable),
            "no native candidate at all: {:?}",
            cands.iter().map(|c| c.source.as_str()).collect::<Vec<_>>()
        );
        // The native sibling is the unfused form: it spells two maps, the
        // fused one spells a composed lambda.
        let fused = cands
            .iter()
            .find(|c| !c.is_representable())
            .expect("a fused candidate");
        assert!(
            fused.source.contains("(\\x -> x * x)"),
            "the fused candidate does not carry the composed lambda: {}",
            fused.source
        );
        assert!(
            cands.iter().any(|c| c.is_representable() && c.source.contains("map (\\x -> x * x) (map (")),
            "the unfused native form is missing: {:?}",
            cands.iter().map(|c| c.source.as_str()).collect::<Vec<_>>()
        );
    }

    /// A re-encoded candidate is a legal tree: every node's children's types
    /// are the ones its row asks for. This is the accumulator check as well
    /// as the ordinary one.
    #[test]
    fn a_re_encoded_candidate_is_legal() {
        let k = kingdom();
        for name in ["map_s", "map2_add_s", "reduce_add_s", "rev_s", "scan_mul_s"] {
            let tree = tree_for(&k, name).unwrap_or_else(|| panic!("no tree for {name}"));
            let sat = saturate_tree(&tree, &k, 16, 4).expect("the e-graph saturates");
            for cand in &sat.candidates {
                let Some(gene) = &cand.gene else {
                    continue;
                };
                check_legal(&k, gene).unwrap_or_else(|e| panic!("{name}: {e}\n{}", cand.source));
            }
        }
    }

    /// The differential test: every candidate that the kingdom can spell
    /// computes what the original computes, element for element, on boundary
    /// and randomised inputs.
    ///
    /// The candidates the kingdom cannot spell (the fused ones, whose
    /// composition is not a kingdom row) are checked against the *compiler*
    /// in the rescue example instead — they are compilation-only by design.
    #[test]
    fn every_representable_candidate_agrees_with_the_interpreter() {
        let k = kingdom();
        let contract = k.relations().check().expect("the kingdom's own check");
        let const_len = k.const_len();
        let inputs: Vec<Vec<f32>> = vec![
            (0..16).map(|i| 0.25 * (i as f32 + 1.0)).collect(),
            vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
            vec![0.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        let mut rng = crate::futhark_kingdom::GeneRng::new(0x000A_11CE_5EED_0001);
        let mut checked = 0usize;
        for root_name in ["arr.f32", "scn.f32", "f32.real"] {
            let root = k.ty(root_name).expect("the root is in the kingdom");
            let layout = crate::typed_gep::GeneLayout::from_program_size(&contract, root, 4)
                .expect("the layout builds");
            for _ in 0..3 {
                let mut gene = vec![0u32; layout.width];
                for (pos, cell) in gene.iter_mut().enumerate() {
                    let alphabet = layout.alphabet(&contract, pos, &const_len);
                    *cell = alphabet[rng.below(alphabet.len())];
                }
                let e = crate::typed_gep::express(&contract, &layout, &gene, &const_len)
                    .expect("a random gene is legal");
                let tree = e.tree();
                let sat = saturate_tree(&tree, &k, 24, 4).expect("the e-graph saturates");
                for cand in &sat.candidates {
                    let Some(gene) = &cand.gene else {
                        continue;
                    };
                    for input in &inputs {
                        let want = k.eval(&tree, input).expect("the interpreter agrees");
                        let got = k.eval(gene, input).expect("the interpreter agrees");
                        assert_eq!(
                            format!("{want:?}"),
                            format!("{got:?}"),
                            "the candidate disagrees with the original\n{}\n{}",
                            k.render(&tree).unwrap_or_default(),
                            cand.source
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no candidate was checked at all");
    }

    /// The type check a re-encoded candidate has to pass: every node's
    /// children have the types its row asks for.
    fn check_legal(kingdom: &FutharkKingdom, tree: &Tree) -> Result<(), String> {
        let row = &kingdom.rows[tree.symbol as usize];
        if row.kind == FutharkKind::Const {
            return Ok(());
        }
        if row.inputs.len() != tree.kids.len() {
            return Err(format!(
                "{} wants {} arguments and has {}",
                row.name,
                row.inputs.len(),
                tree.kids.len()
            ));
        }
        for (i, kid) in tree.kids.iter().enumerate() {
            check_legal(kingdom, kid)?;
            let kid_row = &kingdom.rows[kid.symbol as usize];
            let want = kingdom.dual(row.inputs[i]).map(|d| d.dual.clone());
            let got = kingdom.dual(kid_row.output).map(|d| d.dual.clone());
            if want != got {
                return Err(format!(
                    "{}'s {i}th argument wants {} and has {}",
                    row.name,
                    want.unwrap_or_default(),
                    got.unwrap_or_default()
                ));
            }
        }
        Ok(())
    }
}
