//! Parse and validate every `.wgsl` under a directory with naga's own front
//! end and the kingdom's validator, one TSV row per file on stdout:
//!
//!   path  parse  validate  kingdom  needs_substitution  reason  entry_count  compute_entries
//!
//! `parse` is naga's WGSL front end alone, `validate` is `Target::validate`
//! (the device's capabilities and limits), `kingdom` is the reader's scan
//! (`naga_shape::supported`: does every construct have a row). Each is `ok`
//! or `FAIL`, with the first error in `reason` for the first failing stage.
//! A file that carries a preprocessor directive or a template placeholder
//! (Bevy `#import`, compute-toy `#storage`, `{{name}}`) is `needs_substitution`
//! with the construct named; it is recorded, not dropped.
//!
//!   cargo run --release --features wgsl --example wgsl_corpus_parse -- tests/corpus/harvest > parse.tsv

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

/// What one source text came to.
#[derive(Debug, PartialEq, Eq)]
struct Row {
    parse: bool,
    validate: bool,
    kingdom: bool,
    substitution: Option<String>,
    reason: String,
    entries: usize,
    compute: usize,
}

/// The first preprocessor directive or template placeholder in `src`.
fn substitution(src: &str) -> Option<String> {
    for line in src.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix('#') {
            let word: String = rest.trim_start().chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            return Some(format!("preprocessor directive #{word}"));
        }
        if t.contains("{{") && t.contains("}}") {
            return Some("template placeholder {{...}}".into());
        }
        if t.contains("${") {
            return Some("template placeholder ${...}".into());
        }
    }
    None
}

fn check(src: &str) -> Row {
    let mut row = Row { parse: false, validate: false, kingdom: false, substitution: substitution(src), reason: String::new(), entries: 0, compute: 0 };
    let module = match naga::front::wgsl::parse_str(src) {
        Ok(m) => m,
        Err(e) => {
            row.reason = e.emit_to_string(src).lines().next().unwrap_or("").to_string();
            row.compute = src.matches("@compute").count();
            return row;
        }
    };
    row.parse = true;
    row.entries = module.entry_points.len();
    row.compute = module.entry_points.iter().filter(|ep| ep.stage == naga::ShaderStage::Compute).count();
    match fuller::wgsl::target::Target::current().validate(&module, src) {
        Ok(_) => row.validate = true,
        Err(e) => row.reason = e.lines().next().unwrap_or("").to_string(),
    }
    match fuller::wgsl::naga_shape::supported(&module) {
        Ok(()) => row.kingdom = true,
        Err(e) => {
            if row.reason.is_empty() {
                row.reason = e.lines().next().unwrap_or("").to_string();
            }
        }
    }
    row
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?.map(|e| e.map(|e| e.path())).collect::<Result<_, _>>().map_err(|e| e.to_string())?;
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk(&p, out)?;
        } else if p.extension().is_some_and(|x| x == "wgsl") {
            out.push(p);
        }
    }
    Ok(())
}

fn ok(b: bool) -> &'static str {
    if b {
        "ok"
    } else {
        "FAIL"
    }
}

fn main() -> Result<(), String> {
    let root = std::env::args().nth(1).ok_or("usage: wgsl_corpus_parse <dir>")?;
    let root = PathBuf::from(root);
    let mut files = Vec::new();
    walk(&root, &mut files)?;
    println!("path\tparse\tvalidate\tkingdom\tneeds_substitution\treason\tentry_count\tcompute_entries");
    for f in files {
        let bytes = std::fs::read(&f).map_err(|e| format!("{}: {e}", f.display()))?;
        let rel = f.strip_prefix(&root).unwrap_or(&f).display().to_string();
        let row = match String::from_utf8(bytes) {
            Ok(src) => catch_unwind(AssertUnwindSafe(|| check(&src))).unwrap_or_else(|_| Row { parse: false, validate: false, kingdom: false, substitution: None, reason: "naga panicked".into(), entries: 0, compute: 0 }),
            Err(_) => Row { parse: false, validate: false, kingdom: false, substitution: None, reason: "not utf-8".into(), entries: 0, compute: 0 },
        };
        let reason = row.reason.replace(['\t', '\n'], " ");
        println!("{rel}\t{}\t{}\t{}\t{}\t{reason}\t{}\t{}", ok(row.parse), ok(row.validate), ok(row.kingdom), row.substitution.as_deref().unwrap_or(""), row.entries, row.compute);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: &str = "@group(0) @binding(0) var<storage, read_write> a: array<u32>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) id: vec3<u32>) { a[id.x] = id.x; }\n";

    #[test]
    fn a_compute_kernel_parses_validates_and_counts() {
        let r = check(KERNEL);
        assert!(r.parse && r.validate && r.kingdom, "{r:?}");
        assert_eq!((r.entries, r.compute, r.substitution), (1, 1, None));
    }

    #[test]
    fn a_syntax_error_fails_parse_and_still_counts_compute_text() {
        let r = check("@compute @workgroup_size(1) fn main( {");
        assert!(!r.parse && !r.validate && r.compute == 1 && !r.reason.is_empty(), "{r:?}");
    }

    #[test]
    fn a_preprocessor_line_is_recorded_as_needing_substitution() {
        let r = check(&format!("#import bevy_pbr::mesh_view_bindings\n{KERNEL}"));
        assert_eq!(r.substitution.as_deref(), Some("preprocessor directive #import"));
        assert!(!r.parse);
    }
}
