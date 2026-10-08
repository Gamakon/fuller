//! One-off study: cross-gene subtree sharing in a real evolved population,
//! matched as a TYPED FUNCTION SHAPE — `Left(string, int)`, not `Left(x_0, 3)`.
//!
//! A candidate promoted into the master `SymbolTable` (`src/geneframe.rs`) is
//! one row: op name + a many-hot `Arity { inputs: BTreeMap<Ty,u32>, outputs }`.
//! That row cannot record WHICH terminal filled a slot (which `Var N`, which
//! literal) — only how many slots of each `Ty`. So two subtrees are the SAME
//! promotion candidate iff they are the same op tree with every LEAF
//! generalized to its `Ty`: `(Neg (Var 0))` and `(Neg (Var 3))` are the same
//! shape (`Neg(F)`); `(Pow2 (Num 13))` and `(Pow2 (Var 0))` are NOT (a literal
//! constant is baked into the row, a `Var` is a real input slot) — only the
//! shape with every leaf TRULY VARYING across the matched occurrences becomes
//! an input slot; see `shape()` / `leaf_ty()`.
//!
//! SR is single-typed (every leaf is `F`); the same `shape()` generalization
//! is kingdom-agnostic — a regex gene's `(Lit 48)`/`(Lit 57)` leaves are
//! `Char`, so `(CcRange (Lit 48) (Lit 57))` generalizes the same way.
//!
//! Reads a `GeneExport` population dump (phylu's `EVOLVE_EXPORT_POP=<path>.json`,
//! `ExportedGene { row, gene, genome, rnc, ast, rule }`) and, per individual,
//! reports cross-gene TYPED-SHAPE matches two ways:
//!   - IDENTICAL: shape-equal subtrees across different genes, size >= MIN_OPS.
//!   - EQUAL (saturated): after a bounded algebra+powers run, subtrees whose
//!     e-classes collide across genes but whose shapes differ.
//!
//! Population-wide: the recurring shapes, ranked by count.
//!
//! Usage: cargo run --release --example gene_share_study -- <population.json>

use egglog::prelude::exprs;
use egglog::EGraph;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;

use fuller::expr::{GUARD_RELATIONS, MATH_DATATYPE};
use fuller::ruleset::identities::ALGEBRA_RULESET;
use fuller::ruleset::powers::POWERS_RULESET;

/// Minimum COMBINED existing symbols (op_count) for a subtree to be a
/// function-discovery candidate. 1 = a single existing op over a terminal
/// (Pow2(F), Neg(F)) — normal use, not a combination. 2 is the floor for
/// "a combination of things in the symbol table folded into one symbol".
const MIN_OPS: usize = 2;
const SATURATE_ITERS: u32 = 40;

/// Only the fields this study reads; `genome`/`rnc`/`rule` are in the file
/// too (docs/GENE_EXPORT_FORMAT.md, phylu) but serde ignores unknown fields,
/// so they're left out here rather than carried as dead weight.
#[derive(Deserialize)]
struct ExportedGene {
    row: usize,
    gene: usize,
    ast: Option<String>,
}

/// The on-disk `GeneExport` file shape (docs/GENE_EXPORT_FORMAT.md, phylu):
/// a header plus a flat `genes` array. Only `genes` matters for this study.
#[derive(Deserialize)]
struct GeneExportFile {
    genes: Vec<ExportedGene>,
}

/// A parsed `Math` s-expression, enough to walk every subtree and print it back.
#[derive(Clone)]
enum Node {
    Leaf(String),
    App(String, Vec<Node>),
}

impl Node {
    fn to_sexpr(&self) -> String {
        match self {
            Node::Leaf(s) => s.clone(),
            Node::App(op, ch) => {
                let parts: Vec<String> = ch.iter().map(Node::to_sexpr).collect();
                format!("({op} {})", parts.join(" "))
            }
        }
    }

    /// The `Ty` a TERMINAL node (a `Num` or `Var` application, or a bare leaf
    /// token) would occupy in the master `SymbolTable`. SR's kingdom is
    /// single-typed (every op is `F` in/out, `geneframe::master_table`'s `sr`
    /// rows) — `(Num v)` and `(Var n)` both become an `F` slot, since either
    /// could fill that argument position in a promoted row. `None` if `self`
    /// is not a terminal (an internal op with real children stays concrete).
    fn terminal_ty(&self) -> Option<&'static str> {
        match self {
            Node::Leaf(_) => Some("F"), // every current SR leaf is F-typed.
            // `(Num v)` / `(Var n)` parse as a 1-child App (the value/index is
            // the child) — still terminals, not internal ops, for shape purposes.
            Node::App(op, ch) if (op == "Num" || op == "Var") && ch.len() == 1 => Some("F"),
            Node::App(_, _) => None,
        }
    }

    /// The TYPED SHAPE: every TERMINAL generalized to its `Ty` (the only thing
    /// a `SymbolTable` row can record about a slot), every internal op name
    /// kept concrete (that IS the function being matched — `Left`,
    /// `ProtectedSqrt`, `Concat`, ...). `(Neg (Var 0))` and `(Neg (Var 3))`
    /// both shape to `(Neg F)`; `(Pow2 (Num 13))` and `(Pow2 (Var 0))` both
    /// shape to `(Pow2 F)` too — whether a leaf is truly a VARYING slot
    /// (promote it) or a fixed literal worth keeping (don't) is decided later,
    /// across the matched occurrences, not here. This only answers "same shape?".
    fn shape(&self) -> String {
        if let Some(ty) = self.terminal_ty() {
            return ty.to_string();
        }
        match self {
            Node::App(op, ch) => {
                let parts: Vec<String> = ch.iter().map(Node::shape).collect();
                format!("({op} {})", parts.join(" "))
            }
            Node::Leaf(_) => unreachable!("terminal_ty() covers every Leaf"),
        }
    }

    /// How many COMBINED existing symbols (internal ops, not terminals) this
    /// subtree is made of. Function discovery wants op_count >= 2 — a single
    /// op over a terminal (`Pow2(F)`, `Neg(F)`) is just normal use of a symbol
    /// already in the table, not a new combination worth promoting.
    fn op_count(&self) -> usize {
        if self.terminal_ty().is_some() {
            return 0;
        }
        match self {
            Node::App(_, ch) => 1 + ch.iter().map(Node::op_count).sum::<usize>(),
            Node::Leaf(_) => unreachable!("terminal_ty() covers every Leaf"),
        }
    }

    /// Every subtree with at least `min_ops` COMBINED symbols (see `op_count`),
    /// including itself.
    fn subtrees<'a>(&'a self, min_ops: usize, out: &mut Vec<&'a Node>) {
        if self.op_count() >= min_ops {
            out.push(self);
        }
        if let Node::App(_, ch) = self {
            if self.terminal_ty().is_none() {
                for c in ch {
                    c.subtrees(min_ops, out);
                }
            }
        }
    }
}

/// Minimal s-expression parser for `Math` surface syntax: `(Op a b)`, `(Num 1.5)`,
/// `(Var "x")`. No validation beyond balanced parens — input is fuller's own
/// `ast_sexpr` output, trusted.
fn parse_sexpr(s: &str) -> Node {
    let toks = tokenize(s);
    let mut pos = 0;
    parse_one(&toks, &mut pos)
}

fn tokenize(s: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let mut cur = String::new();
    let mut in_str = false;
    for c in s.chars() {
        if in_str {
            cur.push(c);
            if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '(' | ')' => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
                toks.push(c.to_string());
            }
            '"' => {
                cur.push(c);
                in_str = true;
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    toks
}

fn parse_one(toks: &[String], pos: &mut usize) -> Node {
    assert_eq!(toks[*pos], "(", "expected '(' at {pos}");
    *pos += 1;
    let op = toks[*pos].clone();
    *pos += 1;
    let mut children = Vec::new();
    while toks[*pos] != ")" {
        if toks[*pos] == "(" {
            children.push(parse_one(toks, pos));
        } else {
            children.push(Node::Leaf(toks[*pos].clone()));
            *pos += 1;
        }
    }
    *pos += 1; // consume ')'
    if children.is_empty() {
        Node::Leaf(op)
    } else {
        Node::App(op, children)
    }
}

fn main() -> Result<(), String> {
    let path = env::args().nth(1).ok_or("usage: gene_share_study <population.json>")?;
    let raw = fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;
    let file: GeneExportFile = serde_json::from_str(&raw).map_err(|e| format!("parse {path}: {e}"))?;
    let genes = file.genes;

    // Group by individual (row).
    let mut by_row: HashMap<usize, Vec<&ExportedGene>> = HashMap::new();
    for g in &genes {
        by_row.entry(g.row).or_default().push(g);
    }

    let mut identical_hits: HashMap<String, usize> = HashMap::new();
    let mut saturated_only_hits: HashMap<String, usize> = HashMap::new();
    let mut n_individuals_with_sharing = 0usize;
    let mut n_individuals = 0usize;

    // Pass C accumulators (field-exact, within-individual).
    let mut exact_repeat_hits: HashMap<String, usize> = HashMap::new(); // sexpr -> how many individuals had a repeat
    let mut exact_max_repeats: HashMap<String, usize> = HashMap::new(); // sexpr -> highest repeat count seen in any one individual
    let mut n_individuals_with_exact_repeat = 0usize;
    let mut total_ops_saved = 0usize;

    for row_genes in by_row.values() {
        let parsed: Vec<(usize, Node)> = row_genes
            .iter()
            .filter_map(|g| g.ast.as_ref().map(|a| (g.gene, parse_sexpr(a))))
            .collect();
        if parsed.len() < 2 {
            continue; // nothing to share across
        }
        n_individuals += 1;

        // Pass A: identical TYPED SHAPE across DIFFERENT genes — `(Neg F)` from
        // gene 2's `(Neg (Var 0))` matches gene 5's `(Neg (Var 3))`.
        let mut seen: HashMap<String, Vec<usize>> = HashMap::new();
        for (gene_idx, tree) in &parsed {
            let mut subs = Vec::new();
            tree.subtrees(MIN_OPS, &mut subs);
            for s in subs {
                seen.entry(s.shape()).or_default().push(*gene_idx);
            }
        }
        let mut row_had_sharing = false;
        for (shape, genes_hit) in &seen {
            let mut uniq = genes_hit.clone();
            uniq.sort_unstable();
            uniq.dedup();
            if uniq.len() >= 2 {
                *identical_hits.entry(shape.clone()).or_default() += 1;
                row_had_sharing = true;
            }
        }

        // Pass B: saturate (bounded algebra+powers), then find e-class collisions
        // across genes that were NOT already caught by Pass A (different shapes,
        // same e-class after rewriting) — compared on the CONCRETE string, since
        // the e-class is a property of the actual term, not its generalized shape.
        if let Ok(hits) = saturated_cross_gene_hits(&parsed) {
            for sexpr in hits {
                let shape = parse_sexpr(&sexpr).shape();
                if !seen.contains_key(&shape) {
                    *saturated_only_hits.entry(sexpr).or_default() += 1;
                    row_had_sharing = true;
                }
            }
        }

        if row_had_sharing {
            n_individuals_with_sharing += 1;
        }

        // Pass C: FIELD-EXACT share equivalence (Neumann ≡S), scoped to THIS
        // individual's own genes — the question that matters for execution
        // cost, not function discovery. `Mul(Sqrt(17), Sqrt(3))` occurring 5
        // times in one chromosome's genes means 5 evaluations of the identical
        // value; computing it once and wiring every consumer to that one
        // result (a homeotic REF, `docs/Geneframe_Homeotic_Genes_design.md`)
        // is cheaper to EXECUTE regardless of whether it's ever promoted as a
        // new named symbol. Exact string match: same op, same literals, same
        // variable — the opposite generalization from shape().
        let mut exact: HashMap<String, usize> = HashMap::new(); // sexpr -> occurrence count, this individual
        for (_gene_idx, tree) in &parsed {
            let mut subs = Vec::new();
            tree.subtrees(MIN_OPS, &mut subs);
            for s in subs {
                *exact.entry(s.to_sexpr()).or_default() += 1;
            }
        }
        let mut row_saved_ops = 0usize;
        for (sexpr, count) in &exact {
            if *count >= 2 {
                let ops = parse_sexpr(sexpr).op_count();
                row_saved_ops += (*count - 1) * ops; // every repeat beyond the first is pure waste
                *exact_repeat_hits.entry(sexpr.clone()).or_default() += 1;
                if *count > *exact_max_repeats.get(sexpr).unwrap_or(&0) {
                    exact_max_repeats.insert(sexpr.clone(), *count);
                }
            }
        }
        if row_saved_ops > 0 {
            n_individuals_with_exact_repeat += 1;
            total_ops_saved += row_saved_ops;
        }
    }

    println!("individuals with >=2 genes: {n_individuals}");
    println!("individuals with cross-gene shape sharing (identical or saturated): {n_individuals_with_sharing}");
    println!();
    println!("-- identical TYPED SHAPES (no saturation), top 20 by population count --");
    println!("   (a shape is a candidate SymbolTable row: op names concrete, every leaf -> its Ty)");
    print_top(&identical_hits, 20);
    println!();
    println!("-- additional shape matches ONLY found after saturation, top 20 --");
    print_top(&saturated_only_hits, 20);
    println!();
    println!("-- FIELD-EXACT share equivalence (Neumann (s), within one individual's own genes) --");
    println!("   (same op, same literals, same variable -- a real repeated computation, not a shape)");
    println!(
        "individuals with >=1 within-individual exact repeat: {n_individuals_with_exact_repeat}/{n_individuals}"
    );
    println!("total redundant op-evaluations across the population (sum of (count-1)*op_count per repeat): {total_ops_saved}");
    println!("top exact subtrees by (max repeats seen in one individual), top 20:");
    let mut v: Vec<(&String, &usize)> = exact_max_repeats.iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (sexpr, max_repeats) in v.into_iter().take(20) {
        let individuals_hit = exact_repeat_hits.get(sexpr).copied().unwrap_or(0);
        println!("  max {max_repeats:>3} in one individual, seen in {individuals_hit:>4} individuals  {sexpr}");
    }

    Ok(())
}

fn print_top(counts: &HashMap<String, usize>, n: usize) {
    let mut v: Vec<(&String, &usize)> = counts.iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (sexpr, count) in v.into_iter().take(n) {
        println!("{count:>5}  {sexpr}");
    }
}

/// Assert every gene's tree as a separate root, saturate bounded algebra+powers,
/// and return the canonical (smallest) string for every subtree size whose
/// e-class is hit by more than one gene.
fn saturated_cross_gene_hits(parsed: &[(usize, Node)]) -> Result<Vec<String>, String> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, MATH_DATATYPE).map_err(|e| format!("datatype: {e}"))?;
    egraph.parse_and_run_program(None, GUARD_RELATIONS).map_err(|e| format!("guards: {e}"))?;
    egraph
        .parse_and_run_program(None, &format!("{ALGEBRA_RULESET}\n{POWERS_RULESET}"))
        .map_err(|e| format!("ruleset: {e}"))?;

    // Bind every subtree of every gene to its own named root so we can read
    // back its post-saturation e-class via eval_expr.
    let mut names: Vec<(usize, String, String)> = Vec::new(); // (gene_idx, sexpr, let-name)
    let mut letters = String::new();
    for (gene_idx, tree) in parsed {
        let mut subs = Vec::new();
        tree.subtrees(MIN_OPS, &mut subs);
        for s in subs {
            let name = format!("__g{}_{}", gene_idx, names.len());
            letters.push_str(&format!("(let {name} {})\n", s.to_sexpr()));
            names.push((*gene_idx, s.to_sexpr(), name));
        }
    }
    if names.is_empty() {
        return Ok(Vec::new());
    }
    egraph.parse_and_run_program(None, &letters).map_err(|e| format!("bind subtrees: {e}"))?;
    egraph
        .parse_and_run_program(
            None,
            &format!("(run-schedule (repeat {SATURATE_ITERS} (run algebra) (run powers)))"),
        )
        .map_err(|e| format!("saturate: {e}"))?;

    let mut by_class: HashMap<egglog::Value, Vec<(usize, String)>> = HashMap::new();
    for (gene_idx, sexpr, name) in &names {
        let (_sort, value) = egraph
            .eval_expr(&exprs::var(name))
            .map_err(|e| format!("eval {name}: {e}"))?;
        by_class.entry(value).or_default().push((*gene_idx, sexpr.clone()));
    }

    let mut hits = Vec::new();
    for members in by_class.values() {
        let mut genes_hit: Vec<usize> = members.iter().map(|(g, _)| *g).collect();
        genes_hit.sort_unstable();
        genes_hit.dedup();
        if genes_hit.len() >= 2 {
            // Report the shortest string among the class as the canonical form.
            let canon = members.iter().map(|(_, s)| s.clone()).min_by_key(String::len).unwrap();
            hits.push(canon);
        }
    }
    Ok(hits)
}
