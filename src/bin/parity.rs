//! Parity runner: score gamakAST against SymPy corpora.
//!
//!   cargo run --release --bin parity -- parity/corpus/powsimp.jsonl ...
//!
//! Each file is JSONL lines `{"input": <math>, "target": <math>}` produced by
//! parity/gen_corpus.py. Prints per-file and overall parity %.

use std::fs;

use gamakast::parity::{score_with, Family, Pair};

/// Minimal extraction of the two string fields from a JSONL line, without a
/// JSON dependency: find "input":"..." and "target":"..." with escape handling.
fn field(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":");
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let q = rest.find('"')? + 1;
    let bytes = rest.as_bytes();
    let mut i = q;
    let mut out = String::new();
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '\\' && i + 1 < bytes.len() {
            // unescape \" and \\
            let nxt = bytes[i + 1] as char;
            out.push(nxt);
            i += 2;
            continue;
        }
        if c == '"' {
            return Some(out);
        }
        out.push(c);
        i += 1;
    }
    None
}

fn load(path: &str) -> Result<Vec<Pair>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut pairs = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let input = field(line, "input").ok_or_else(|| format!("no input in: {line}"))?;
        let target = field(line, "target").ok_or_else(|| format!("no target in: {line}"))?;
        pairs.push(Pair { input, target });
    }
    Ok(pairs)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: parity <corpus.jsonl> [more.jsonl ...]".into());
    }
    let (mut g_total, mut g_matched) = (0usize, 0usize);
    println!("{:<24} {:>7} {:>7} {:>8}", "module", "matched", "total", "parity");
    println!("{}", "-".repeat(50));
    for path in &args {
        let pairs = load(path)?;
        let name = path.rsplit('/').next().unwrap_or(path).trim_end_matches(".jsonl");
        // The trig corpus needs the trig family; everything else uses algebra.
        // (distribute and trig explode the e-graph if run together, so they are
        // scored with the family appropriate to the corpus.)
        let family = if name.contains("trig") { Family::Trig } else { Family::Algebra };
        let rep = score_with(&pairs, family);
        g_total += rep.total;
        g_matched += rep.matched;
        println!("{:<24} {:>7} {:>7} {:>7.1}%", name, rep.matched, rep.total, rep.pct());
    }
    println!("{}", "-".repeat(50));
    let pct = if g_total == 0 { 0.0 } else { 100.0 * g_matched as f64 / g_total as f64 };
    println!("{:<24} {:>7} {:>7} {:>7.1}%", "OVERALL", g_matched, g_total, pct);
    Ok(())
}
