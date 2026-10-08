//! Read WGSL kernels into the kingdom's roots and report what the chromosome
//! would need: roots by kind, nodes per root (the head length a root needs
//! under the generic encoder), roots past the device's node limit, the
//! `class.instance` rows used and the naga nodes the table has no row for,
//! the per-kernel terminals, and the raw repeats across roots.
//!
//!   cargo run --release --features wgsl --example wgsl_read_kernel -- [--sexpr] <file.wgsl>...
//!
//! Then, per function, the chromosome: the roots as head genes, repeated
//! subtrees folded into a homeotic tail (exact repeats, conservative load
//! rule), every gene encoded as Karva through the generic pair and decoded
//! back (`KARVA ok`), with the head length each function needs.
//!
//! `--sexpr` also prints every root's s-expression. A kernel that is assembled
//! from several files at build time (`mix64.wgsl` with `vary.wgsl`, `splice.wgsl`
//! with `kernel.wgsl`) is given as `a.wgsl+b.wgsl`: the files are concatenated
//! in that order, as `concat!(include_str!(..))` does.

use std::collections::BTreeMap;

use fuller::gpu_eval::MAX_NODES;
use fuller::wgsl::{chromosome, read, ChromosomeOptions, RootKind};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let show_sexpr = args.iter().any(|a| a == "--sexpr");
    let paths: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if paths.is_empty() {
        return Err("usage: wgsl_read_kernel [--sexpr] <file.wgsl>...".into());
    }
    println!(
        "{:<18} {:<20} {:>5} {:>5} {:>5} {:>4} {:>4} {:>5} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "file", "function", "roots", "store", "cond", "init", "ret", "args", "nodes", "max", "p90", ">lim", "shared", "ldshr"
    );
    let mut all_unknown: BTreeMap<String, usize> = BTreeMap::new();
    let mut all_known: BTreeMap<String, usize> = BTreeMap::new();
    for path in &paths {
        let mut src = String::new();
        for part in path.split('+') {
            src.push_str(&std::fs::read_to_string(part).map_err(|e| format!("read {part}: {e}"))?);
        }
        let kernel = read(&src).map_err(|e| format!("{path}: {e}"))?;
        let file: Vec<String> = path.split('+').map(|p| std::path::Path::new(p).file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default()).collect();
        let file = file.join("+");
        for f in &kernel.functions {
            let mut by_kind = [0usize; 5];
            let mut sizes: Vec<usize> = Vec::new();
            for r in &f.roots {
                let k = match r.kind {
                    RootKind::Store { .. } => 0,
                    RootKind::Condition { .. } => 1,
                    RootKind::Init { .. } => 2,
                    RootKind::Return => 3,
                    RootKind::Argument { .. } => 4,
                };
                by_kind[k] += 1;
                sizes.push(r.tree.size());
            }
            sizes.sort_unstable();
            let total: usize = sizes.iter().sum();
            let max = sizes.last().copied().unwrap_or(0);
            let p90 = if sizes.is_empty() { 0 } else { sizes[(sizes.len() * 9 / 10).min(sizes.len() - 1)] };
            let over = sizes.iter().filter(|&&n| n > MAX_NODES).count();
            println!(
                "{file:<18} {:<20} {:>5} {:>5} {:>5} {:>4} {:>4} {:>5} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6}",
                f.name,
                f.roots.len(),
                by_kind[0],
                by_kind[1],
                by_kind[2],
                by_kind[3],
                by_kind[4],
                total,
                max,
                p90,
                over,
                f.shared_handles.len(),
                f.shared_loads
            );
            if !f.unread_statements.is_empty() {
                println!("    unread statements: {:?}", f.unread_statements);
            }
            if show_sexpr {
                for r in &f.roots {
                    println!("    [{:?}] {}", r.kind, r.tree.to_sexpr());
                }
            }
        }
        println!("    {:<20} {:>5} {:>6} {:>6} {:>7} {:>5} {:>7} {:>5}  oversized per head length", "chromosome", "genes", "shared", "filled", "refused", "head", "fits@", "karva");
        for f in &kernel.functions {
            match chromosome(f, &ChromosomeOptions::default()) {
                Ok(c) => {
                    let over: Vec<String> = c.oversized_at.iter().map(|(h, n)| format!("{h}:{n}")).collect();
                    println!(
                        "    {:<20} {:>5} {:>6} {:>6} {:>7} {:>5} {:>7} {:>5}  {}",
                        c.function,
                        c.folded.head.len() + c.folded.tail.len(),
                        c.matches,
                        c.folded.filled,
                        c.refused_loads,
                        c.head_needed,
                        c.genes.as_ref().map(|(h, _)| h.to_string()).unwrap_or_else(|| "none".into()),
                        if c.genes.is_some() { "ok" } else { "-" },
                        over.join(" ")
                    );
                }
                Err(e) => println!("    {:<20} KARVA FAILED: {e}", f.name),
            }
        }
        for (name, n) in kernel.functions_used() {
            let known = !name.starts_with("naga.");
            let bucket = if known { &mut all_known } else { &mut all_unknown };
            *bucket.entry(name).or_default() += n;
        }
        let terminals = kernel.terminals();
        println!("    terminals ({}): {}", terminals.len(), terminals.iter().cloned().collect::<Vec<_>>().join(" "));
    }
    let total_known: usize = all_known.values().sum();
    let total_unknown: usize = all_unknown.values().sum();
    println!(
        "\nrows used: {} distinct class.instance names over {} nodes; {} nodes ({:.1} %) on {} naga nodes the table has no row for",
        all_known.len(),
        total_known,
        total_unknown,
        100.0 * total_unknown as f64 / (total_known + total_unknown).max(1) as f64,
        all_unknown.len()
    );
    for (name, n) in &all_unknown {
        println!("    no row: {name:<40} {n:>6}");
    }
    let mut known: Vec<(&String, &usize)> = all_known.iter().collect();
    known.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    println!("top rows:");
    for (name, n) in known.iter().take(25) {
        println!("    {name:<40} {n:>6}");
    }
    Ok(())
}
