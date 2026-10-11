//! The drafted kernels into the safeWGSL grammar (`src/wgsl/safe_reader.rs`):
//! for every `kNN.wgsl` in a directory, the safe tree or the refusal by
//! name; a converted tree is rendered, validated by naga and written as
//! `<out>/kNN.safe.wgsl` with its s-expression beside it.
//!
//! With `--karva <dir>` every converted kernel is also encoded as Karva
//! genes (`src/wgsl/safe_karva.rs`) at one head length and tail-slot count
//! for the whole population, and written as `<dir>/kNN.json` in the dump
//! form phylu's typed loader reads; `--head <n>` fixes the head length
//! (default 32), the slot count is the largest any kernel needs, rounded up
//! to a multiple of eight.
//!
//!   cargo run --release --features wgsl --example wgsl_safe_convert -- <dir> [--out <dir>] [--karva <dir> [--head <n>] [--slots <n>]]

use std::path::PathBuf;

use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::render::{cost, gram_shipped_safe_tree, validate_text, Tree};
use fuller::wgsl::safe_karva::{dump_file, encode, head_needed, render_tree, split};
use fuller::wgsl::safe_reader::{read_safe, statement_outline};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or("usage: wgsl_safe_convert <dir> [--out <dir>]")?);
    let mut out = dir.join("safe");
    let mut outline: Option<String> = None;
    let mut karva: Option<PathBuf> = None;
    let mut head_length = 32usize;
    // --slots: the tail slot count forced (a population's layout must hold
    // kernels larger than any draft, 2026-10-11); 0 = the smallest that fits.
    let mut forced_slots = 0usize;
    while let Some(a) = args.next() {
        if a == "--out" {
            out = PathBuf::from(args.next().ok_or("--out needs a value")?);
        }
        if a == "--karva" {
            karva = Some(PathBuf::from(args.next().ok_or("--karva needs a value")?));
        }
        if a == "--slots" {
            forced_slots = args.next().ok_or("--slots needs a value")?.parse().map_err(|e| format!("--slots: {e}"))?;
            continue;
        }
        if a == "--head" {
            head_length = args.next().ok_or("--head needs a value")?.parse().map_err(|e| format!("--head: {e}"))?;
        }
        if a == "--outline" {
            outline = Some(args.next().ok_or("--outline needs a kernel id")?);
        }
    }
    if let Some(id) = outline {
        let src = std::fs::read_to_string(dir.join(format!("{id}.wgsl"))).map_err(|e| e.to_string())?;
        for l in statement_outline(&src)? {
            println!("{l}");
        }
        return Ok(());
    }
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let kingdom = WgslKingdom::load_safe();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
    files.sort();
    let (mut converted, mut valid) = (0, 0);
    let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();
    let mut trees: Vec<(String, Tree)> = Vec::new();
    println!("{:<6} {:<10} {:>6} {:<8} reason / notes", "kernel", "converted", "nodes", "valid");
    for f in &files {
        let id = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let src = std::fs::read_to_string(f).map_err(|e| e.to_string())?;
        match read_safe(&src, &kingdom) {
            Err(e) => {
                let key: String = e.split(':').next_back().unwrap_or(&e).trim().chars().take(60).collect();
                *reasons.entry(key).or_default() += 1;
                println!("{id:<6} {:<10} {:>6} {:<8} {}", "no", "-", "-", &e[..e.len().min(1600)]);
            }
            Ok(r) => {
                converted += 1;
                let (text, ok) = match render_tree(&r.tree, &kingdom) {
                    Ok(o) => match validate_text(&o.text) {
                        Ok(()) => (o.text, "yes".to_string()),
                        Err(e) => {
                            if std::env::var_os("SAFE_CONVERT_ERRORS").is_some() {
                                eprintln!("{id}: {e}");
                            }
                            (o.text, format!("INVALID: {}", e.lines().find(|l| l.contains("error")).unwrap_or("")))
                        }
                    },
                    Err(e) => (String::new(), format!("RENDER: {e}")),
                };
                if ok == "yes" {
                    valid += 1;
                }
                std::fs::write(out.join(format!("{id}.safe.wgsl")), &text).map_err(|e| e.to_string())?;
                std::fs::write(out.join(format!("{id}.safe.sexpr")), r.tree.to_sexpr()).map_err(|e| e.to_string())?;
                println!("{id:<6} {:<10} {:>6} {:<8} {}", "yes", r.tree.size(), ok, r.notes.join("; "));
                if ok == "yes" {
                    trees.push((id.clone(), r.tree));
                }
            }
        }
    }
    println!("\n{} of {} converted, {} valid; refusals:", converted, files.len(), valid);
    for (k, v) in &reasons {
        println!("  {v:3}  {k}");
    }
    let Some(karva) = karva else { return Ok(()) };
    std::fs::create_dir_all(&karva).map_err(|e| e.to_string())?;
    // The layout: tail slots any kernel needs at each head length, then the
    // chosen head length's count rounded up to a multiple of eight.
    println!("\n{:<6} {:>8} {:>10}  slots needed per kernel", "head", "max", "mean");
    let mut chosen_slots = 0usize;
    for h in [16usize, 24, 32, 48, 64] {
        let needs: Vec<usize> = trees.iter().map(|(_, t)| split(t, h).map(|s| s.tail.len()).unwrap_or(usize::MAX)).collect();
        let max = needs.iter().copied().max().unwrap_or(0);
        let mean = needs.iter().sum::<usize>() as f64 / needs.len().max(1) as f64;
        println!("{h:<6} {max:>8} {mean:>10.1}");
        if h == head_length {
            chosen_slots = max.div_ceil(8) * 8;
        }
    }
    if forced_slots > 0 {
        chosen_slots = forced_slots;
    }
    if chosen_slots == 0 {
        let max = trees.iter().map(|(_, t)| split(t, head_length).map(|s| s.tail.len()).unwrap_or(usize::MAX)).max().unwrap_or(0);
        chosen_slots = max.div_ceil(8) * 8;
    }
    println!("\nlayout: head {head_length}, tail {}, {} tail slots, {} genes a kernel", head_length * 3 + 1, chosen_slots, chosen_slots + 1);
    let shipped = cost(&gram_shipped_safe_tree(&kingdom)?);
    println!("shipped safe tree: cost {:.0} per item, {:.0} base", shipped.per_item, shipped.base);
    println!("{:<6} {:>6} {:>6} {:>10} {:>10} {:>8} {:>10}", "kernel", "nodes", "slots", "head need", "per item", "base", "x shipped");
    for (id, t) in &trees {
        let c = encode(t, head_length, chosen_slots, 7013).map_err(|e| format!("{id}: {e}"))?;
        let v = dump_file(&format!("{id}.wgsl"), &kingdom, &c);
        std::fs::write(karva.join(format!("{id}.json")), serde_json::to_string(&v).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let k = cost(t);
        println!("{id:<6} {:>6} {:>6} {:>10} {:>10.0} {:>8.0} {:>10.1}", t.size(), c.folded.filled, head_needed(t), k.per_item, k.base, k.per_item / shipped.per_item);
    }
    println!("{} dumps written to {}", trees.len(), karva.display());
    Ok(())
}
