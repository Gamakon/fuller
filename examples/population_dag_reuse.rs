//! Task 3's measurement (`docs/PLAN_population_dag.md`): over consecutive
//! generation pairs exported by phylu (`population.g<N>.json`, the gene
//! export format), fold generation g into the population DAG, evaluate
//! it over the law's rows (every memo a miss), then fold g+1 into the
//! SAME DAG and evaluate it: the memo hits, the node evaluations avoided
//! against a fresh evaluation of g+1, the host milliseconds of each, and
//! the bit-identity of the memoised against the fresh values.
//!
//!   cargo run --release --example population_dag_reuse -- <data.tsv> <rows> <dir with population.g*.json>

use std::collections::BTreeMap;
use std::time::Instant;

use fuller::population_dag::PopulationDag;

fn load_rows(path: &str, n: usize) -> Result<Vec<Vec<(String, f64)>>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().ok_or("empty")?.split('\t').collect();
    // The exported genes name variables by column index among the
    // non-target columns, in order: `(Var 0)` is the first feature.
    let features: Vec<usize> = header.iter().enumerate().filter(|(_, h)| **h != "target").map(|(i, _)| i).collect();
    let mut rows = Vec::new();
    for line in lines.take(n) {
        let cells: Vec<&str> = line.split('\t').collect();
        let row: Vec<(String, f64)> = features.iter().enumerate().map(|(k, &i)| (k.to_string(), cells[i].parse::<f64>().unwrap_or(f64::NAN))).collect();
        rows.push(row);
    }
    Ok(rows)
}

fn load_genes(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    Ok(json["genes"].as_array().ok_or("no genes")?.iter().filter_map(|g| g["ast"].as_str().map(str::to_string)).collect())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [data, n_rows, dir] = args.as_slice() else {
        return Err("usage: population_dag_reuse <data.tsv> <rows> <dir>".into());
    };
    let rows = load_rows(data, n_rows.parse().map_err(|e| format!("rows: {e}"))?)?;
    let mut files: BTreeMap<u32, String> = BTreeMap::new();
    for e in std::fs::read_dir(dir).map_err(|e| format!("{dir}: {e}"))?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if let Some(g) = name.strip_prefix("population.g").and_then(|s| s.strip_suffix(".json")).and_then(|s| s.parse::<u32>().ok()) {
            files.insert(g, e.path().to_string_lossy().to_string());
        }
    }
    println!("{} rows of {data}; {} exports in {dir}", rows.len(), files.len());
    println!("{:>11} {:>7} {:>8} {:>8} {:>11} {:>10} {:>10} {:>9} {:>9} {:>9}  identical", "pair", "nodes", "homeotic", "new", "interpreter", "evaluated", "avoided", "hits", "fresh ms", "memo ms");
    let mut sums = (0u64, 0u64, 0f64, 0f64, 0usize, 0u64);
    for (&g, path) in &files {
        let Some(next) = files.get(&(g + 1)) else { continue };
        let mut dag = PopulationDag::new();
        let roots = dag.fold(&load_genes(path)?)?;
        let t = Instant::now();
        let _ = dag.eval(&roots, &rows, 1)?;
        let _first_ms = t.elapsed().as_secs_f64() * 1e3;
        // The next generation into the same DAG.
        let genes1 = load_genes(next)?;
        let roots1 = dag.fold(&genes1)?;
        let report = dag.report();
        let t = Instant::now();
        let (memo_values, stats) = dag.eval(&roots1, &rows, 1)?;
        let memo_ms = t.elapsed().as_secs_f64() * 1e3;
        // The same generation, fresh.
        let mut fresh = PopulationDag::new();
        let roots_f = fresh.fold(&genes1)?;
        let t = Instant::now();
        let (fresh_values, fresh_stats) = fresh.eval(&roots_f, &rows, 1)?;
        let fresh_ms = t.elapsed().as_secs_f64() * 1e3;
        let identical = memo_values.iter().zip(&fresh_values).all(|(a, b)| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
        // What an interpreter does: every operator of every gene, occurrences and all.
        let interpreter: u64 = roots1.iter().map(|&r| u64::from(dag.nodes[r as usize].ops)).sum();
        println!(
            "{:>5}/{:<5} {:>7} {:>8} {:>8} {:>11} {:>10} {:>10} {:>9} {:>9.1} {:>9.1}  {}",
            g,
            g + 1,
            report.nodes_total,
            report.homeotic,
            report.homeotic_new,
            interpreter,
            stats.evaluated,
            stats.avoided,
            stats.hits,
            fresh_ms,
            memo_ms,
            if identical { "yes" } else { "NO" }
        );
        sums.0 += stats.evaluated;
        sums.1 += stats.avoided;
        sums.2 += fresh_ms;
        sums.3 += memo_ms;
        sums.4 += 1;
        sums.5 += interpreter;
        if !identical {
            return Err(format!("pair {g}/{}: memoised values differ from fresh ones", g + 1));
        }
        let _ = fresh_stats;
    }
    if sums.4 > 0 {
        let work = sums.0 + sums.1;
        println!("\nover {} pairs: an interpreter evaluates {} operators per generation; the shared DAG {} (one per distinct subtree, {:.1} % of the interpreter's); with last generation's memos {} ({:.1} %), the memos avoiding {:.1} % of the DAG's own work; host ms per generation {:.1} fresh, {:.1} with memos", sums.4, sums.5 / sums.4 as u64, work / sums.4 as u64, 100.0 * work as f64 / sums.5 as f64, sums.0 / sums.4 as u64, 100.0 * sums.0 as f64 / sums.5 as f64, 100.0 * sums.1 as f64 / work as f64, sums.2 / sums.4 as f64, sums.3 / sums.4 as f64);
    }
    Ok(())
}
