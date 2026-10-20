//! How much of the harvested corpus the kingdom reads: for every file of
//! `tests/corpus/harvest/sources.tsv` that validates under naga and has a
//! compute entry point, the safe reader's verdict (a tree, or the reason),
//! split into dawn and the other repositories, then the commonest reasons.
//! `--rows` also prints one line per file (`path`, `read` or the reason).
//!
//!   cargo run --release --features wgsl --example wgsl_harvest_read -- [--rows] [tests/corpus/harvest]

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::render::validate_text;
use fuller::wgsl::safe_karva::render_tree_with;
use fuller::wgsl::safe_reader::read_safe;

/// A reason with its numbers and quoted names folded, so one cause counts once.
fn kind(reason: &str) -> String {
    // (The statement paths and the names of the helpers inlined on the way are
    // where the failure was, not what it was: the last cause counts.)
    let first = reason.lines().next().unwrap_or("");
    let first = first.rsplit("]: ").next().unwrap_or(first);
    let first = first.rsplit("inlining ").next().unwrap_or(first);
    let mut out = String::new();
    let mut quoted = false;
    for c in first.chars() {
        match c {
            '`' | '"' | '\'' => {
                if !quoted {
                    out.push('X');
                }
                quoted = !quoted;
            }
            _ if quoted => {}
            c if c.is_ascii_digit() => {
                if !out.ends_with('N') {
                    out.push('N');
                }
            }
            c => out.push(c),
        }
    }
    out.chars().take(110).collect()
}

/// The text with a `// override <name> <value>` header for each pipeline
/// constant it leaves without a default (the pipeline's value is not in the
/// file): 64 for a size or a count of threads, 1 for any other.
fn with_override_headers(src: &str) -> String {
    let mut headers = String::new();
    let mut rest = src;
    while let Some(i) = rest.find("override ") {
        let after = &rest[i + "override ".len()..];
        let end = after.find(';').unwrap_or(after.len());
        let name: String = after.trim_start().chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        if !name.is_empty() && end < after.len() && !after[..end].contains('=') {
            let lower = name.to_lowercase();
            let v = if ["size", "workgroup", "threads", "wg"].iter().any(|k| lower.contains(k)) { 64 } else { 1 };
            headers.push_str(&format!("// override {name} {v}\n"));
        }
        rest = after;
    }
    format!("{headers}{src}")
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rows = args.iter().any(|a| a == "--rows");
    let dir = PathBuf::from(args.iter().find(|a| !a.starts_with("--")).map_or("tests/corpus/harvest", String::as_str));
    let table = std::fs::read_to_string(dir.join("sources.tsv")).map_err(|e| e.to_string())?;
    let kingdom = WgslKingdom::load_safe();
    // [validating with a compute entry, read, rendered and validated] by group.
    let mut count: BTreeMap<&str, [usize; 3]> = BTreeMap::new();
    let mut reasons: BTreeMap<&str, BTreeMap<String, (usize, String)>> = BTreeMap::new();
    for line in table.lines().skip(1) {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() < 10 || c[9] != "ok" || c[6] != "true" {
            continue;
        }
        let group = if c[0] == "dawn" { "dawn" } else { "non-dawn" };
        let path = dir.join(c[0]).join(c[2]);
        let Ok(src) = std::fs::read_to_string(&path) else { continue };
        let src = with_override_headers(&src);
        let n = count.entry(group).or_default();
        n[0] += 1;
        let verdict = catch_unwind(AssertUnwindSafe(|| {
            let r = read_safe(&src, &kingdom)?;
            // (A module that declares aggregate types reads in the kingdom extended by them.)
            let k = r.kingdom.as_ref().unwrap_or(&kingdom);
            let text = render_tree_with(&r.tree, k, &r.role)?.text;
            validate_text(&text)?;
            Ok::<(), String>(())
        }))
        .unwrap_or_else(|_| Err("panic in the reader".into()));
        let n = count.entry(group).or_default();
        match &verdict {
            Ok(()) => {
                n[1] += 1;
                n[2] += 1;
                if rows {
                    println!("{}/{}\tread", c[0], c[2]);
                }
            }
            Err(e) => {
                let slot = reasons.entry(group).or_default().entry(kind(e)).or_insert((0, format!("{}/{}", c[0], c[2])));
                slot.0 += 1;
                if rows {
                    println!("{}/{}\t{}", c[0], c[2], e.lines().next().unwrap_or(""));
                }
            }
        }
    }
    for (g, n) in &count {
        println!("{g}: {} validate with a compute entry, {} read, {} rendered and validated", n[0], n[1], n[2]);
    }
    let mut all: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for m in reasons.values() {
        for (k, (n, f)) in m {
            all.entry(k.clone()).or_insert((0, f.clone())).0 += n;
        }
    }
    let mut top: Vec<_> = all.into_iter().collect();
    top.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    println!("reasons a read fails, commonest first:");
    for (k, (n, f)) in top.iter().take(25) {
        println!("{n:6}  {k}   e.g. {f}");
    }
    Ok(())
}
