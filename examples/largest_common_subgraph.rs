//! CLI wrapper around `fuller::extract::maximal_shared` (the detection
//! itself now lives there as a library function — phylu's `fold_to_dag`
//! calls it directly). This example reads a `GeneExport` JSON population
//! dump, finds the individual with the most maximal-match savings, and
//! prints a report.
//!
//! Usage: cargo run --release --example largest_common_subgraph -- <population.json> [row]

use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;

use fuller::extract::maximal_shared;

#[derive(Deserialize)]
struct ExportedGene {
    row: usize,
    gene: usize,
    ast: Option<String>,
}

#[derive(Deserialize)]
struct GeneExportFile {
    genes: Vec<ExportedGene>,
}

/// phylu's exported `ast` renders two leaf shapes egglog's `Math` datatype
/// does not accept as written: an integer-valued `Num` literal bare ("-72",
/// needs "-72.0" — `Num` takes a syntactic f64) and a `Var`'s column index
/// as a bare integer ("0", needs the string "x_0" — `Var` takes a `String`).
/// Reparse and re-render both leaf forms; every other token passes through.
/// (`maximal_shared` itself takes already-correctly-rendered `Math` strings
/// — phylu's live callers use `nodes_to_math_named`, which already renders
/// `(Var "x_0")` correctly, so this workaround is specific to reading the
/// raw JSON export this example parses, not a property of the library call.)
fn ensure_float_literals(ast: &str) -> String {
    let toks = tokenize(ast);
    let mut out = String::new();
    let mut i = 0;
    while i < toks.len() {
        if toks[i] == "(" && toks.get(i + 1).map(String::as_str) == Some("Num") {
            let v = &toks[i + 2];
            let as_float = if v.contains('.') || v.contains('e') || v.contains('E') { v.clone() } else { format!("{v}.0") };
            out.push_str(&format!("(Num {as_float})"));
            i += 4; // "(" "Num" value ")"
        } else if toks[i] == "(" && toks.get(i + 1).map(String::as_str) == Some("Var") {
            let col = &toks[i + 2];
            out.push_str(&format!("(Var \"x_{col}\")"));
            i += 4; // "(" "Var" column ")"
        } else {
            out.push_str(&toks[i]);
            out.push(' ');
            i += 1;
        }
    }
    out
}

fn tokenize(s: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' | ')' => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
                toks.push(c.to_string());
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

const MIN_OPS: usize = 1;

struct Report {
    row: usize,
    n_genes: usize,
    matches: Vec<fuller::extract::Match>,
    total_savings: usize,
    baseline_ops: usize,
}

fn total_op_count(ast: &str) -> usize {
    // A quick, local op-count over the Math s-expr text itself (Num/Var
    // leaves excluded) -- same definition maximal_shared uses internally,
    // kept separate here since the library only reports matches, not a
    // whole-gene baseline (callers who need one, like this report, compute
    // it themselves from the same genes they already have).
    let toks = tokenize(ast);
    let mut depth = 0i32;
    let mut ops = 0usize;
    let mut i = 0;
    let mut skip_until: i32 = -1;
    while i < toks.len() {
        if toks[i] == "(" {
            depth += 1;
            if i + 1 < toks.len() && (toks[i + 1] == "Num" || toks[i + 1] == "Var") {
                skip_until = depth - 1; // mark: don't count this one or descend further meaningfully
                ops += 0; // Num/Var themselves are not ops
            } else {
                ops += 1;
            }
        } else if toks[i] == ")" {
            depth -= 1;
            if depth == skip_until {
                skip_until = -1;
            }
        }
        i += 1;
    }
    ops
}

fn analyze(row: usize, row_genes: &[&ExportedGene]) -> Result<Option<Report>, String> {
    let mut sorted_genes: Vec<&&ExportedGene> = row_genes.iter().collect();
    sorted_genes.sort_by_key(|g| g.gene);
    let genes: Vec<String> = sorted_genes.iter().filter_map(|g| g.ast.as_ref().map(|a| ensure_float_literals(a))).collect();
    if genes.len() < 2 {
        return Ok(None);
    }
    let matches = maximal_shared(&genes, MIN_OPS)?;
    let total_savings: usize = matches.iter().map(|m| (m.sites.len() - 1) * m.internal_op_count).sum();
    let baseline_ops: usize = genes.iter().map(|g| total_op_count(g)).sum();
    Ok(Some(Report { row, n_genes: genes.len(), matches, total_savings, baseline_ops }))
}

fn main() -> Result<(), String> {
    let path = env::args().nth(1).ok_or("usage: largest_common_subgraph <population.json> [row]")?;
    let raw = fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;
    let file: GeneExportFile = serde_json::from_str(&raw).map_err(|e| format!("parse {path}: {e}"))?;

    let mut by_row: HashMap<usize, Vec<&ExportedGene>> = HashMap::new();
    for g in &file.genes {
        by_row.entry(g.row).or_default().push(g);
    }

    let report = if let Some(row_arg) = env::args().nth(2) {
        let row: usize = row_arg.parse().map_err(|_| format!("not a row number: {row_arg}"))?;
        let row_genes = by_row.get(&row).ok_or_else(|| format!("no such row: {row}"))?;
        analyze(row, row_genes)?.ok_or_else(|| format!("row {row} has no maximal shared subgraph"))?
    } else {
        let mut best: Option<Report> = None;
        let mut n_tried = 0usize;
        for (&row, row_genes) in &by_row {
            if row_genes.iter().filter(|g| g.ast.is_some()).count() < 2 {
                continue;
            }
            n_tried += 1;
            if let Some(report) = analyze(row, row_genes)? {
                let better = best.as_ref().map(|b| report.total_savings > b.total_savings).unwrap_or(true);
                if better {
                    best = Some(report);
                }
            }
        }
        best.ok_or_else(|| format!("no individual with a maximal shared subgraph found among {n_tried} multi-gene individuals checked"))?
    };

    println!("individual: row {}, {} genes", report.row, report.n_genes);
    println!("maximal shared subgraphs (>= {MIN_OPS} internal op(s), site-counted): {}", report.matches.len());
    for m in &report.matches {
        let savings = (m.sites.len() - 1) * m.internal_op_count;
        println!("\n  {} independent sites, {} internal op(s), saves {savings} op-evaluations if folded:", m.sites.len(), m.internal_op_count);
        println!("    sites: {:?}", m.sites);
    }
    println!("\ncurrent (unfolded) total op count for this chromosome: {}", report.baseline_ops);
    println!(
        "total savings if EVERY maximal match above were folded: {} ops ({:.1}% of the chromosome's current op count)",
        report.total_savings,
        100.0 * report.total_savings as f64 / report.baseline_ops.max(1) as f64
    );

    Ok(())
}
