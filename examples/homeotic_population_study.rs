//! The homeotic fold on a real population: every chromosome of a phylu
//! `population_genes.tsv` dump (columns `individual island gene math`) is
//! folded, checked to unfold exactly, and encoded at the layout's head
//! length. Reports how many chromosomes fill a tail slot, how many slots, the
//! operators removed, what was skipped and why, and what the saturated finder
//! adds over the exact one.
//!
//! Usage: cargo run --release --example homeotic_population_study -- <population_genes.tsv> [head_len] [tail_slots]

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::time::Instant;

use fuller::extract::{maximal_shared, maximal_shared_saturated};
use fuller::homeotic::{canonical, encode, fold, unfold, FoldOptions};
use fuller::karva::{master_pset, FunctionSpec, PsetSpec};

fn pset(vars: &[String]) -> PsetSpec {
    let mut functions = HashMap::new();
    for (name, arity) in master_pset() {
        functions.insert(name.to_string(), FunctionSpec { semantic_id: name.to_string(), arity });
    }
    PsetSpec { variables: vars.to_vec(), functions, rnc_values: vec![] }
}

/// (individual, input genes, folded head, folded tail) — the first fold shown.
type Example = (usize, Vec<String>, Vec<String>, Vec<String>);

fn op_count(s: &str) -> usize {
    // One operator per opening paren that is not a Num or Var leaf.
    s.matches('(').count() - s.matches("(Num ").count() - s.matches("(Var ").count()
}

fn main() -> Result<(), String> {
    let path = env::args().nth(1).ok_or("usage: homeotic_population_study <population_genes.tsv> [head_len] [tail_slots]")?;
    let head_len: usize = env::args().nth(2).map(|v| v.parse().map_err(|e| format!("head_len: {e}"))).transpose()?.unwrap_or(34);
    let tail_slots: usize = env::args().nth(3).map(|v| v.parse().map_err(|e| format!("tail_slots: {e}"))).transpose()?.unwrap_or(3);
    let raw = fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;
    let mut lines = raw.lines();
    let cols: Vec<&str> = lines.next().ok_or("empty")?.split('\t').collect();
    let col = |n: &str| cols.iter().position(|c| *c == n).ok_or(format!("missing column {n}"));
    let (ci, cm) = (col("individual")?, col("math")?);
    let mut by_individual: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        by_individual.entry(f[ci].parse().map_err(|e| format!("individual: {e}"))?).or_default().push(f[cm].to_string());
    }
    let mut vars: Vec<String> = Vec::new();
    for g in by_individual.values().flatten() {
        for v in g.split("(Var \"").skip(1) {
            let name = v.split('"').next().unwrap_or("").to_string();
            if !vars.contains(&name) {
                vars.push(name);
            }
        }
    }
    vars.sort();
    let pset = pset(&vars);
    println!("{} chromosomes, head {head_len}, tail slots {tail_slots}, variables {vars:?}", by_individual.len());

    let opts = FoldOptions { tail_slots, max_definition_ops: Some(head_len) };
    let t0 = Instant::now();
    let mut exact_fills: BTreeMap<usize, usize> = BTreeMap::new(); // slots filled -> chromosomes
    let (mut exact_with_fill, mut exact_ops_removed, mut exact_skipped, mut unfold_fail, mut encode_fail, mut encode_ok) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut sat_extra_fills = 0usize;
    let mut sat_with_fill = 0usize;
    let mut sat_errors = 0usize;
    let mut sat_time = 0.0f64;
    let mut example: Option<Example> = None;
    // What was shared: definition -> chromosomes sharing it; whole-gene copies vs subtrees;
    // cross-gene vs within-gene; a per-chromosome table.
    let mut definitions: BTreeMap<String, usize> = BTreeMap::new();
    let (mut whole_gene_defs, mut subtree_defs, mut cross_gene, mut within_gene) = (0usize, 0usize, 0usize, 0usize);
    let mut per_chromosome: Vec<String> = vec!["individual\tfilled\tops_before\tops_after\twhole_gene\tdefinitions".to_string()];
    for (id, genes) in &by_individual {
        let matches = maximal_shared(genes, 2)?;
        let folded = fold(genes, &matches, opts)?;
        *exact_fills.entry(folded.filled).or_default() += 1;
        exact_skipped += folded.skipped;
        if folded.filled > 0 {
            exact_with_fill += 1;
            let before: usize = genes.iter().map(|g| op_count(g)).sum();
            let after: usize = folded.head.iter().chain(folded.tail.iter()).map(|g| op_count(g)).sum();
            exact_ops_removed += before.saturating_sub(after);
            let mut whole = 0usize;
            let mut defs = Vec::new();
            for (slot, def) in folded.tail.iter().take(folded.filled).enumerate() {
                *definitions.entry(def.clone()).or_default() += 1;
                let canon_genes: Vec<String> = genes.iter().map(|g| canonical(g)).collect::<Result<_, _>>()?;
                let reference = format!("(Var \"href{slot}\")");
                let is_whole = canon_genes.iter().any(|g| g == def);
                if is_whole {
                    whole += 1;
                    whole_gene_defs += 1;
                } else {
                    subtree_defs += 1;
                }
                let genes_reading = folded.head.iter().filter(|h| h.contains(&reference)).count()
                    + folded.tail.iter().take(folded.filled).filter(|t| t.contains(&reference)).count();
                if genes_reading >= 2 {
                    cross_gene += 1;
                } else {
                    within_gene += 1;
                }
                defs.push(def.clone());
            }
            per_chromosome.push(format!("{id}\t{}\t{before}\t{after}\t{whole}\t{}", folded.filled, defs.join(" | ")));
            let want: Vec<String> = genes.iter().map(|g| canonical(g)).collect::<Result<_, _>>()?;
            if unfold(&folded)? != want {
                unfold_fail += 1;
            }
            match encode(&folded, &pset, 7013, head_len) {
                Ok(_) => encode_ok += 1,
                Err(_) => encode_fail += 1,
            }
            if example.is_none() {
                example = Some((*id, genes.clone(), folded.head.clone(), folded.tail.clone()));
            }
        }
        // The saturated finder, on the same chromosome: what rewriting adds.
        let ts = Instant::now();
        match maximal_shared_saturated(genes, 2) {
            Ok((sm, _, chosen)) => {
                let sf = fold(&chosen, &sm, opts)?;
                if sf.filled > 0 {
                    sat_with_fill += 1;
                }
                if sf.filled > folded.filled {
                    sat_extra_fills += 1;
                }
            }
            Err(_) => sat_errors += 1,
        }
        sat_time += ts.elapsed().as_secs_f64();
    }
    let n = by_individual.len();
    println!("exact finder: chromosomes with >=1 tail fill {exact_with_fill}/{n}; fills histogram {exact_fills:?}; skipped matches {exact_skipped}");
    println!("operators removed across the population by folding {exact_ops_removed}; unfold mismatches {unfold_fail}; encode ok {encode_ok}, encode failures {encode_fail}");
    println!("saturated finder: chromosomes with >=1 fill {sat_with_fill}/{n}; chromosomes with MORE fills than exact {sat_extra_fills}; errors {sat_errors}; {:.1} ms per chromosome", 1000.0 * sat_time / n as f64);
    if let Some((id, genes, head, tail)) = example {
        println!("example, individual {id}:");
        for g in &genes {
            println!("  in   {g}");
        }
        for g in &head {
            println!("  head {g}");
        }
        for g in &tail {
            println!("  tail {g}");
        }
    }
    println!("definitions: whole-gene copies {whole_gene_defs}, subtrees {subtree_defs}; read by >=2 genes {cross_gene}, read within one gene only {within_gene}");
    let mut ranked: Vec<(&String, &usize)> = definitions.iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    println!("top shared definitions (chromosomes sharing it, definition):");
    for (def, n) in ranked.iter().take(15) {
        println!("  {n:>5}  {def}");
    }
    let table = format!("{}.homeotic_folds.tsv", path.trim_end_matches(".tsv"));
    fs::write(&table, per_chromosome.join("\n") + "\n").map_err(|e| format!("write {table}: {e}"))?;
    println!("per-chromosome table: {table}");
    println!("total {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}
