//! Read WGSL kernels into the kingdom's roots and report what the chromosome
//! would need: roots by kind, nodes per root (the head length a root needs
//! under the generic encoder), roots past the device's node limit, the
//! `class.instance` rows used and the naga nodes the table has no row for,
//! the per-kernel terminals, and the raw repeats across roots.
//!
//!   cargo run --release --features wgsl --example wgsl_read_kernel -- [--sexpr] [--dump <out.json>] <file.wgsl>...
//!
//! `--rebuilt <dir>` writes each kernel's rebuilt WGSL text (the device
//! parity run compiles it beside the original). `--dump <out.json>` writes
//! every function's chromosome (pset, head
//! length, tail slots, Karva genes as tokens) and the kingdom's size, the
//! artefact phylu uploads and decodes on the device.
//!
//! Then, per function, the chromosome: the roots as head genes, repeated
//! subtrees folded into a homeotic tail (exact repeats, conservative load
//! rule), every gene encoded as Karva through the generic pair and decoded
//! back (`KARVA ok`), with the head length each function needs.
//!
//! Last, the naga round trip per kernel: every function's genes DECODED from
//! Karva, unfolded, rebuilt into a naga module, validated, written as WGSL,
//! read again and compared root by root (`ROUND_TRIP ok`, or the first
//! difference).
//!
//! `--sexpr` also prints every root's s-expression. A kernel that is assembled
//! from several files at build time (`mix64.wgsl` with `vary.wgsl`, `splice.wgsl`
//! with `kernel.wgsl`) is given as `a.wgsl+b.wgsl`: the files are concatenated
//! in that order, as `concat!(include_str!(..))` does.

use std::collections::BTreeMap;

use fuller::gpu_eval::MAX_NODES;
use fuller::homeotic::{unfold, FoldedChromosome};
use fuller::karva::karva_to_terms_generic;
use fuller::wgsl::{chromosome, chromosome_typed, infer_function, read, rebuild_folded, round_trip, ChromosomeOptions, RootKind, WgslKingdom};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let show_sexpr = args.iter().any(|a| a == "--sexpr");
    let dump_path = args.iter().position(|a| a == "--dump").and_then(|i| args.get(i + 1).cloned());
    let rebuilt_dir = args.iter().position(|a| a == "--rebuilt").and_then(|i| args.get(i + 1).cloned());
    let folded_dir = args.iter().position(|a| a == "--rebuilt-folded").and_then(|i| args.get(i + 1).cloned());
    // --variants <dir>: every subset of the ENTRY POINT's filled tail slots as
    // its own folded rebuild, <dir>/v<mask>/<file>, the share-or-recompute
    // population of docs/PLAN_wgsl_superopt.md §2.1 (helper functions keep
    // every share). Written only for a file with one entry point.
    let variants_dir = args.iter().position(|a| a == "--variants").and_then(|i| args.get(i + 1).cloned());
    let paths: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| !(a.starts_with("--") || (*i > 0 && (args[i - 1] == "--dump" || args[i - 1] == "--rebuilt" || args[i - 1] == "--rebuilt-folded" || args[i - 1] == "--variants"))))
        .map(|(_, a)| a)
        .collect();
    let mut dump: Vec<serde_json::Value> = Vec::new();
    if paths.is_empty() {
        return Err("usage: wgsl_read_kernel [--sexpr] <file.wgsl>...".into());
    }
    println!(
        "{:<18} {:<20} {:>5} {:>5} {:>5} {:>4} {:>4} {:>5} {:>4} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "file", "function", "roots", "store", "cond", "init", "ret", "args", "let", "nodes", "max", "p90", ">lim", "shared", "ldshr"
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
            let mut by_kind = [0usize; 6];
            let mut sizes: Vec<usize> = Vec::new();
            for r in &f.roots {
                let k = match r.kind {
                    RootKind::Store { .. } => 0,
                    RootKind::Condition { .. } => 1,
                    RootKind::Init { .. } => 2,
                    RootKind::Return => 3,
                    RootKind::Argument { .. } => 4,
                    RootKind::Let { .. } => 5,
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
                "{file:<18} {:<20} {:>5} {:>5} {:>5} {:>4} {:>4} {:>5} {:>4} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6}",
                f.name,
                f.roots.len(),
                by_kind[0],
                by_kind[1],
                by_kind[2],
                by_kind[3],
                by_kind[4],
                by_kind[5],
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
        println!("    {:<20} {:>5} {:>6} {:>6} {:>7} {:>7} {:>5} {:>7} {:>5}  oversized per head length", "chromosome", "genes", "shared", "filled", "refused", "lineage", "head", "fits@", "karva");
        for f in &kernel.functions {
            match chromosome(f, &ChromosomeOptions::default()) {
                Ok(c) => {
                    let over: Vec<String> = c.oversized_at.iter().map(|(h, n)| format!("{h}:{n}")).collect();
                    println!(
                        "    {:<20} {:>5} {:>6} {:>6} {:>7} {:>7} {:>5} {:>7} {:>5}  {}",
                        c.function,
                        c.folded.head.len() + c.folded.tail.len(),
                        c.matches,
                        c.folded.filled,
                        c.refused.values().sum::<usize>(),
                        c.refused_lineage,
                        c.head_needed,
                        c.genes.as_ref().map(|(h, _)| h.to_string()).unwrap_or_else(|| "none".into()),
                        if c.genes.is_some() { "ok" } else { "-" },
                        over.join(" ")
                    );
                }
                Err(e) => println!("    {:<20} KARVA FAILED: {e}", f.name),
            }
        }
        // Form inference: duals by use, conflicts, texts read under two forms.
        let kingdom = WgslKingdom::load();
        println!("    {:<20} {:>6} {:>6} {:>9} {:>8} {:>7} {:>9}  untyped by slot", "inference", "nodes", "typed", "conflicts", "no dual", "no row", "2-form");
        for f in &kernel.functions {
            let t = infer_function(f, &kingdom);
            let st = &t.stats;
            let by_slot: Vec<String> = st.untyped_by_slot.iter().map(|(s, n)| format!("{s}:{n}")).collect();
            println!("    {:<20} {:>6} {:>6} {:>9} {:>8} {:>7} {:>9}  {}", f.name, st.nodes, st.typed, st.conflicts, st.no_dual, st.no_row, st.conflicting_texts, by_slot.join(" "));
        }
        // The chromosome dump, if asked: tokens as the engine would upload them.
        if dump_path.is_some() {
            let kingdom = WgslKingdom::load();
            for f in &kernel.functions {
                if let Ok(c) = chromosome_typed(f, &kingdom, &ChromosomeOptions::default()) {
                    if let Some((h, genes)) = &c.genes {
                        let tok = |t: &fuller::karva::Token| match t {
                            fuller::karva::Token::Func(n) => serde_json::json!({ "f": n }),
                            fuller::karva::Token::Var(n) => serde_json::json!({ "v": n }),
                            fuller::karva::Token::Num(v) => serde_json::json!({ "n": v }),
                        };
                        let mut functions: Vec<serde_json::Value> = c.pset.functions.iter().map(|(n, spec)| serde_json::json!({ "name": n, "arity": spec.arity })).collect();
                        functions.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
                        dump.push(serde_json::json!({
                            "file": file,
                            "function": f.name,
                            "head_length": h,
                            "tail_length": h * 3 + 1,
                            "k": 4,
                            "head_genes": c.folded.head.len(),
                            "tail_slots": c.folded.tail.len(),
                            "tail_filled": c.folded.filled,
                            "pset": { "variables": c.pset.variables, "functions": functions, "rnc_values": c.pset.rnc_values },
                            "genes": genes.iter().zip(&c.ty_codes).map(|((head, tail), codes)| serde_json::json!({ "head": head.iter().map(tok).collect::<Vec<_>>(), "tail": tail.iter().map(tok).collect::<Vec<_>>(), "ty_codes": codes })).collect::<Vec<_>>(),
                            "refused_forms": c.refused_forms,
                            "refused_lineage": c.refused_lineage,
                            "refused": c.refused,
                            "placements": c.placements.iter().map(|p| p.as_ref().map(|p| serde_json::json!({ "path": p.path, "after_let": p.after_let }))).collect::<Vec<_>>(),
                            "roots": c.roots,
                            "folded": { "head": c.folded.head, "tail": c.folded.tail },
                        }));
                    }
                }
            }
        }
        // The naga round trip, from the decoded genes of every function.
        let mut roots_all: Vec<Vec<String>> = Vec::new();
        let mut trip_err: Option<String> = None;
        let mut chromosomes_all = Vec::new();
        for f in &kernel.functions {
            match chromosome(f, &ChromosomeOptions::default()) {
                Ok(c) => match c.genes.as_ref() {
                    Some((_, genes)) => {
                        let decoded: Result<Vec<String>, String> = genes.iter().map(|(h, t)| karva_to_terms_generic(h, t, &c.pset)).collect();
                        match decoded {
                            Ok(d) => {
                                let n_head = c.folded.head.len();
                                let folded = FoldedChromosome { head: d[..n_head].to_vec(), tail: d[n_head..].to_vec(), filled: c.folded.filled, skipped: c.folded.skipped };
                                match unfold(&folded) {
                                    Ok(r) => roots_all.push(r),
                                    Err(e) => trip_err = trip_err.or(Some(format!("{}: unfold: {e}", f.name))),
                                }
                                // The folded rebuild takes the DECODED genes with
                                // their placements: what the device will run.
                                let mut decoded_c = c.clone();
                                decoded_c.folded = folded;
                                chromosomes_all.push(decoded_c);
                            }
                            Err(e) => trip_err = trip_err.or(Some(format!("{}: decode: {e}", f.name))),
                        }
                    }
                    None => trip_err = trip_err.or(Some(format!("{}: no head length fits", f.name))),
                },
                Err(e) => trip_err = trip_err.or(Some(format!("{}: {e}", f.name))),
            }
        }
        if let (Some(dir), None) = (&variants_dir, &trip_err) {
            let entries: Vec<usize> = kernel.functions.iter().enumerate().filter(|(_, f)| f.entry_point).map(|(i, _)| i).collect();
            if entries.len() != 1 {
                return Err(format!("--variants needs one entry point, {path} has {}", entries.len()));
            }
            let e = entries[0];
            let filled: Vec<usize> = (0..chromosomes_all[e].folded.tail.len()).filter(|&s| chromosomes_all[e].folded.tail[s] != fuller::homeotic::EMPTY_SLOT).collect();
            println!("    VARIANTS: {} filled slots in {} → {} subsets", filled.len(), kernel.functions[e].name, 1usize << filled.len());
            for mask in 0..(1usize << filled.len()) {
                let mut cs = chromosomes_all.clone();
                // Unfold the slots NOT in the subset, highest slot first so a
                // definition that reads a later slot is substituted after it.
                for (bit, &slot) in filled.iter().enumerate().rev() {
                    if mask & (1 << bit) == 0 {
                        cs[e].folded = fuller::homeotic::unfold_slot(&cs[e].folded, slot)?;
                    }
                }
                let r = rebuild_folded(&kernel, &cs).map_err(|e| format!("variant {mask}: {e}"))?;
                let vdir = format!("{dir}/v{mask:02}");
                std::fs::create_dir_all(&vdir).map_err(|e| format!("mkdir {vdir}: {e}"))?;
                let out = format!("{vdir}/{}", file.replace('+', "_"));
                std::fs::write(&out, &r.wgsl).map_err(|e| format!("write {out}: {e}"))?;
            }
            println!("    VARIANTS written to {dir}/v00 .. v{:02}", (1usize << filled.len()) - 1);
        }
        // The folded rebuild: every shared definition a `let` at its
        // placement; validated and written, no structural gate (the roots
        // differ by design), proven by the device (the oracle, the sweep).
        if let (Some(dir), None) = (&folded_dir, &trip_err) {
            match rebuild_folded(&kernel, &chromosomes_all) {
                Ok(r) => {
                    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {dir}: {e}"))?;
                    let out = format!("{dir}/{}", file.replace('+', "_"));
                    std::fs::write(&out, &r.wgsl).map_err(|e| format!("write {out}: {e}"))?;
                    let defs = r.wgsl.matches("let def_").count();
                    println!("    FOLDED rebuild ok ({} functions, {} bytes, {defs} shared definitions emitted) written to {out}", kernel.functions.len(), r.wgsl.len());
                }
                Err(e) => println!("    FOLDED rebuild FAILED: {e}"),
            }
        }
        match trip_err {
            Some(e) => println!("    ROUND_TRIP FAILED before the rebuild: {e}"),
            None => match round_trip(&kernel, &roots_all) {
                Ok(r) => {
                    println!("    ROUND_TRIP ok ({} functions rebuilt, {} bytes of WGSL written and read back)", kernel.functions.len(), r.wgsl.len());
                    // The rebuilt text, for the device run against the original.
                    if let Some(dir) = &rebuilt_dir {
                        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {dir}: {e}"))?;
                        let out = format!("{dir}/{}", file.replace('+', "_"));
                        std::fs::write(&out, &r.wgsl).map_err(|e| format!("write {out}: {e}"))?;
                        println!("    rebuilt WGSL written to {out}");
                    }
                }
                Err(e) => println!("    ROUND_TRIP FAILED: {e}"),
            },
        }
        for (name, n) in kernel.functions_used() {
            let known = !name.starts_with("naga.");
            let bucket = if known { &mut all_known } else { &mut all_unknown };
            *bucket.entry(name).or_default() += n;
        }
        let terminals = kernel.terminals();
        println!("    terminals ({}): {}", terminals.len(), terminals.iter().cloned().collect::<Vec<_>>().join(" "));
    }
    if let Some(path) = dump_path {
        let kingdom = fuller::wgsl::WgslKingdom::load();
        let out = serde_json::json!({
            "kingdom": { "name": fuller::wgsl::WGSL, "duals": kingdom.duals.len(), "instances": kingdom.rows.len(), "k_max": fuller::gpu_eval::K_MAX, "tables": "kingdoms/wgsl/{types,functions}.tsv; fuller::wgsl::WgslKingdom::load()" },
            "functions": dump,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?).map_err(|e| format!("write {path}: {e}"))?;
        println!("dumped {} chromosomes to {path}", out["functions"].as_array().map_or(0, |a| a.len()));
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
