//! Population-level share-equivalence: how many of a population's individuals
//! and genes are distinct, before and after rewriting, and how much repeated
//! computation the whole population carries.
//!
//! Reads phylu's `population_genes.tsv` (columns `individual island gene math`,
//! `math` = fuller `Math` s-expression) and reports:
//!
//! 1. EXACT, population-wide: distinct genes / genes, distinct chromosomes /
//!    individuals (per island too), and for every subtree with >= 1 operator
//!    the occurrence count across the population, bucketed by operator count,
//!    with the redundant operator-evaluations `sum((count - 1) * ops)` that a
//!    population-wide shared evaluation would remove.
//! 2. AFTER REWRITING: every DISTINCT gene is canonicalised on its own with
//!    `fuller::extract::smallest_form` (bounded algebra+powers then rational,
//!    each family in its own e-graph), and the same counts are taken on the
//!    canonical forms. This is the scalable "normalise, then hash" route; it
//!    finds the equivalences the normal form exposes and no others.
//!
//! Progress and results go to stdout; the caller tees them to a log.
//!
//! Usage: cargo run --release --example population_share_classes -- <population_genes.tsv>

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::time::Instant;

use fuller::extract::smallest_form;

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

    fn is_terminal(&self) -> bool {
        match self {
            Node::Leaf(_) => true,
            Node::App(op, ch) => (op == "Num" || op == "Var") && ch.len() == 1,
        }
    }

    fn op_count(&self) -> usize {
        if self.is_terminal() {
            return 0;
        }
        match self {
            Node::App(_, ch) => 1 + ch.iter().map(Node::op_count).sum::<usize>(),
            Node::Leaf(_) => 0,
        }
    }

    /// Every non-terminal subtree, including self.
    fn subtrees<'a>(&'a self, out: &mut Vec<&'a Node>) {
        if self.is_terminal() {
            return;
        }
        out.push(self);
        if let Node::App(_, ch) = self {
            for c in ch {
                c.subtrees(out);
            }
        }
    }
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

fn parse_one(toks: &[String], pos: &mut usize) -> Result<Node, String> {
    let open = toks.get(*pos).ok_or("unexpected end")?;
    if open != "(" {
        *pos += 1;
        return Ok(Node::Leaf(open.clone()));
    }
    *pos += 1;
    let op = toks.get(*pos).ok_or("missing operator")?.clone();
    *pos += 1;
    let mut children = Vec::new();
    loop {
        let t = toks.get(*pos).ok_or("unbalanced parentheses")?;
        if t == ")" {
            *pos += 1;
            break;
        }
        children.push(parse_one(toks, pos)?);
    }
    if children.is_empty() {
        Ok(Node::Leaf(op))
    } else {
        Ok(Node::App(op, children))
    }
}

/// egglog types `(Num 56)` as i64 and refuses it where f64 is wanted; phylu's
/// export prints whole-number literals without a decimal point. Spell every
/// `Num` literal as f64 so `smallest_form` accepts the gene. Presentation only.
fn float_literals(n: &Node) -> Node {
    match n {
        Node::App(op, ch) if op == "Num" && ch.len() == 1 => {
            if let Node::Leaf(v) = &ch[0] {
                if !v.contains('.') && !v.contains('e') && !v.contains("inf") && !v.contains("NaN") {
                    return Node::App(op.clone(), vec![Node::Leaf(format!("{v}.0"))]);
                }
            }
            n.clone()
        }
        Node::App(op, ch) => Node::App(op.clone(), ch.iter().map(float_literals).collect()),
        Node::Leaf(_) => n.clone(),
    }
}

fn parse_sexpr(s: &str) -> Result<Node, String> {
    let toks = tokenize(s);
    let mut pos = 0;
    let node = parse_one(&toks, &mut pos)?;
    if pos != toks.len() {
        return Err(format!("trailing tokens in {s:?}"));
    }
    Ok(node)
}

struct Gene {
    individual: usize,
    island: String,
    sexpr: String,
}

struct Stats {
    label: String,
    genes: usize,
    distinct_genes: usize,
    individuals: usize,
    distinct_chromosomes: usize,
    per_island: BTreeMap<String, (usize, usize)>,
    /// op_count -> (occurrences, distinct)
    buckets: BTreeMap<usize, (usize, usize)>,
    /// Distinct non-terminal subtrees across the population: the node count of
    /// the population hash-consed into one DAG. `total_ops - dag_nodes` is the
    /// operator evaluations a population-wide shared evaluation would remove.
    dag_nodes: usize,
    total_ops: usize,
    top_shared: Vec<(usize, usize, String)>, // (redundant ops, count, sexpr)
}

fn analyse(label: &str, genes: &[Gene]) -> Result<Stats, String> {
    let mut distinct_genes: HashSet<&str> = HashSet::new();
    let mut by_individual: HashMap<(usize, &str), Vec<&str>> = HashMap::new();
    let mut occurrences: HashMap<String, (usize, usize)> = HashMap::new(); // sexpr -> (count, ops)
    let mut total_ops = 0usize;
    for g in genes {
        distinct_genes.insert(g.sexpr.as_str());
        by_individual.entry((g.individual, g.island.as_str())).or_default().push(g.sexpr.as_str());
        let tree = parse_sexpr(&g.sexpr)?;
        total_ops += tree.op_count();
        let mut subs = Vec::new();
        tree.subtrees(&mut subs);
        for s in subs {
            let e = occurrences.entry(s.to_sexpr()).or_insert((0, s.op_count()));
            e.0 += 1;
        }
    }
    let mut chromosomes: HashSet<Vec<&str>> = HashSet::new();
    let mut per_island: BTreeMap<String, (HashSet<Vec<&str>>, usize)> = BTreeMap::new();
    for ((_, island), mut gs) in by_individual.clone() {
        gs.sort_unstable();
        chromosomes.insert(gs.clone());
        let e = per_island.entry(island.to_string()).or_default();
        e.0.insert(gs);
        e.1 += 1;
    }
    let mut buckets: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    let mut top: Vec<(usize, usize, String)> = Vec::new();
    for (sexpr, (count, ops)) in &occurrences {
        let b = buckets.entry(*ops).or_default();
        b.0 += count;
        b.1 += 1;
        if *count >= 2 {
            // Redundant evaluations of this subtree taken as a whole tree;
            // overlaps with its own children, so it ranks, it does not sum.
            top.push(((count - 1) * ops, *count, sexpr.clone()));
        }
    }
    let dag_nodes = occurrences.len();
    top.sort_by(|a, b| b.0.cmp(&a.0).then(a.2.cmp(&b.2)));
    top.truncate(25);
    Ok(Stats {
        label: label.to_string(),
        genes: genes.len(),
        distinct_genes: distinct_genes.len(),
        individuals: by_individual.len(),
        distinct_chromosomes: chromosomes.len(),
        per_island: per_island.into_iter().map(|(k, (set, n))| (k, (set.len(), n))).collect(),
        buckets,
        dag_nodes,
        total_ops,
        top_shared: top,
    })
}

fn report(s: &Stats) {
    println!("== {} ==", s.label);
    println!("genes {}  distinct genes {}  ({:.1}% distinct)", s.genes, s.distinct_genes, pct(s.distinct_genes, s.genes));
    println!(
        "individuals {}  distinct chromosomes {}  ({:.1}% distinct)",
        s.individuals,
        s.distinct_chromosomes,
        pct(s.distinct_chromosomes, s.individuals)
    );
    for (island, (distinct, n)) in &s.per_island {
        println!("  island {island}: {n} individuals, {distinct} distinct chromosomes ({:.1}%)", pct(*distinct, *n));
    }
    println!("total operator evaluations per generation (one per node): {}", s.total_ops);
    println!("subtree size   occurrences   distinct");
    for (ops, (occ, distinct)) in &s.buckets {
        println!("{ops:>12} {occ:>13} {distinct:>10}");
    }
    println!(
        "population as one DAG: {} distinct subtrees against {} tree nodes; shared evaluation removes {} operator evaluations ({:.1}% of the population's work)",
        s.dag_nodes,
        s.total_ops,
        s.total_ops.saturating_sub(s.dag_nodes),
        pct(s.total_ops.saturating_sub(s.dag_nodes), s.total_ops)
    );
    println!("top shared subtrees by (count-1)*ops, whole-tree view (overlapping, ranks only):");
    for (red, count, sexpr) in &s.top_shared {
        println!("  {red:>7} {count:>6}  {sexpr}");
    }
    println!();
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * a as f64 / b as f64
    }
}

fn main() -> Result<(), String> {
    let path = env::args().nth(1).ok_or("usage: population_share_classes <population_genes.tsv>")?;
    let raw = fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;
    let mut lines = raw.lines();
    let header = lines.next().ok_or("empty file")?;
    let cols: Vec<&str> = header.split('\t').collect();
    let col = |name: &str| cols.iter().position(|c| *c == name).ok_or(format!("missing column {name}"));
    let (ci, cis, cm) = (col("individual")?, col("island")?, col("math")?);
    let mut genes = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        genes.push(Gene {
            individual: f[ci].parse().map_err(|e| format!("individual: {e}"))?,
            island: f[cis].to_string(),
            sexpr: f[cm].to_string(),
        });
    }
    println!("read {} genes from {path}", genes.len());
    let t0 = Instant::now();
    let exact = analyse("EXACT (as decoded, no rewriting)", &genes)?;
    report(&exact);
    println!("exact pass took {:.2} s\n", t0.elapsed().as_secs_f64());

    // Canonicalise every DISTINCT gene once; map each gene to its canonical form.
    let inputs: Vec<String> = (0..16).map(|i| format!("x_{i}")).collect();
    let mut distinct: Vec<&str> = genes.iter().map(|g| g.sexpr.as_str()).collect::<HashSet<_>>().into_iter().collect();
    distinct.sort_unstable();
    println!("canonicalising {} distinct genes with smallest_form ...", distinct.len());
    let t1 = Instant::now();
    let mut canon: HashMap<&str, String> = HashMap::new();
    let (mut shrank, mut failed, mut ops_before, mut ops_after) = (0usize, 0usize, 0usize, 0usize);
    for (i, g) in distinct.iter().enumerate() {
        let spelled = float_literals(&parse_sexpr(g)?).to_sexpr();
        match smallest_form(&spelled, &inputs, &[], &[]) {
            Ok(sf) => {
                if sf.cost < sf.input_cost {
                    shrank += 1;
                }
                ops_before += parse_sexpr(g)?.op_count();
                ops_after += parse_sexpr(&sf.expr)?.op_count();
                canon.insert(g, sf.expr);
            }
            Err(e) => {
                failed += 1;
                if failed <= 5 {
                    println!("  smallest_form failed on {g}: {e}");
                }
                canon.insert(g, (*g).to_string());
            }
        }
        if (i + 1) % 250 == 0 {
            println!(
                "  {}/{} canonicalised, {:.1} s elapsed, {} shrank, {} failed",
                i + 1,
                distinct.len(),
                t1.elapsed().as_secs_f64(),
                shrank,
                failed
            );
        }
    }
    println!(
        "canonicalisation: {} distinct genes in {:.1} s ({:.1} ms each); {} shrank, {} failed; operators {} -> {} ({:.1}% fewer)",
        distinct.len(),
        t1.elapsed().as_secs_f64(),
        1000.0 * t1.elapsed().as_secs_f64() / distinct.len().max(1) as f64,
        shrank,
        failed,
        ops_before,
        ops_after,
        pct(ops_before.saturating_sub(ops_after), ops_before)
    );
    let canon_genes: Vec<Gene> = genes
        .iter()
        .map(|g| Gene { individual: g.individual, island: g.island.clone(), sexpr: canon[g.sexpr.as_str()].clone() })
        .collect();
    let after = analyse("AFTER REWRITING (smallest_form canonical forms)", &canon_genes)?;
    report(&after);
    println!(
        "equivalence classes: distinct genes {} -> {} ({} merged by rewriting); distinct chromosomes {} -> {}",
        exact.distinct_genes,
        after.distinct_genes,
        exact.distinct_genes.saturating_sub(after.distinct_genes),
        exact.distinct_chromosomes,
        after.distinct_chromosomes
    );
    println!("total {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}
