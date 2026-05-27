//! Parity runner: score gamakAST against SymPy corpora.
//!
//!   cargo run --release --bin parity -- parity/corpus/powsimp.jsonl ...
//!
//! Each file is JSONL lines `{"input": <math>, "target": <math>}` produced by
//! parity/gen_corpus.py. Prints per-file and overall parity %.

use std::fs;

use gamakast::parity::{proves_equal_with, score_with, Family, Pair};

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

fn family_for(name: &str) -> Family {
    // distribute and trig explode the e-graph if run together, so the family is
    // selected to match each corpus.
    if name.contains("trig") {
        Family::Trig
    } else if name.contains("ratsimp") || name.contains("radsimp") {
        Family::Rational
    } else {
        Family::Algebra
    }
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `--dump-misses`: print every unmatched (input -> target) pair instead of
    // scoring. Feeds the rule-mining agents real failure cases, not guesses.
    let dump_misses = args.first().map(|s| s == "--dump-misses").unwrap_or(false);
    if dump_misses {
        args.remove(0);
    }
    if args.is_empty() {
        return Err("usage: parity [--dump-misses] <corpus.jsonl> [more.jsonl ...]".into());
    }
    if dump_misses {
        for path in &args {
            let pairs = load(path)?;
            let name = path.rsplit('/').next().unwrap_or(path).trim_end_matches(".jsonl");
            let family = family_for(name);
            for p in &pairs {
                if !proves_equal_with(&p.input, &p.target, family).unwrap_or(false) {
                    println!("{name}\tIN  {}\n{name}\tTGT {}\n", p.input, p.target);
                }
            }
        }
        return Ok(());
    }
    let (mut g_total, mut g_matched) = (0usize, 0usize);
    println!("{:<24} {:>7} {:>7} {:>8}", "module", "matched", "total", "parity");
    println!("{}", "-".repeat(50));
    for path in &args {
        let pairs = load(path)?;
        let name = path.rsplit('/').next().unwrap_or(path).trim_end_matches(".jsonl");
        let rep = score_with(&pairs, family_for(name));
        g_total += rep.total;
        g_matched += rep.matched;
        println!("{:<24} {:>7} {:>7} {:>7.1}%", name, rep.matched, rep.total, rep.pct());
    }
    println!("{}", "-".repeat(50));
    let pct = if g_total == 0 { 0.0 } else { 100.0 * g_matched as f64 / g_total as f64 };
    println!("{:<24} {:>7} {:>7} {:>7.1}%", "OVERALL", g_matched, g_total, pct);
    Ok(())
}
