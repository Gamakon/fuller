//! THE READING SNAPSHOT (2026-10-19, the naga/wgpu 30 upgrade,
//! docs/PLAN_naga_wgpu_30.md section 3; Codex's review: a count of kernels
//! read cannot show that roots, holes or trees stayed the same). For every
//! committed kernel, one JSON line: the scaffold reader's roots per function
//! (kind, path, point, tree) and the safe reader's tree and interface, or
//! each one's refusal. `tests/fixtures/reading_snapshot_naga020.jsonl` holds
//! each line naga 0.20.0 read as its digest ([`digest`]: 1,942 kernels, 55 MB
//! of readings in full); `examples/wgsl_reading_snapshot` writes either form
//! (`--full` the readings, for finding a difference). The test below reads
//! every kernel again and compares, line for line.

use std::path::{Path, PathBuf};

use crate::wgsl::loader::WgslKingdom;

/// The kernels the snapshot covers, by their path from the crate root, in a
/// fixed order: the model-reply corpus, the safe corpus, gen0's drafts and
/// v24's reference (all tracked).
pub fn kernels(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for dir in ["tests/corpus/model_replies", "tests/corpus/safewgsl", "logs/HGP/001/gen0"] {
        let mut files: Vec<PathBuf> = std::fs::read_dir(root.join(dir)).map_err(|e| format!("{dir}: {e}"))?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
        files.sort();
        out.extend(files.into_iter().map(|p| p.strip_prefix(root).map(Path::to_path_buf).unwrap_or(p)));
    }
    out.push(PathBuf::from("problems/v24_ingest_small/reference.wgsl"));
    Ok(out)
}

/// FNV-1a over a reading's JSON text: the same reading, the same digest.
fn fnv(text: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// A line as its digest: the file, and per reader its reading's digest, or
/// `refused` (a refusal's words quote naga's errors, so only the fact is
/// kept).
pub fn digest(line: &serde_json::Value) -> serde_json::Value {
    let part = |v: &serde_json::Value| if v.get("refused").is_some() { serde_json::Value::String("refused".into()) } else { serde_json::Value::String(fnv(&v.to_string())) };
    serde_json::json!({ "file": line["file"], "scaffold": part(&line["scaffold"]), "safe": part(&line["safe"]) })
}

/// One kernel's line.
pub fn line(file: &str, text: &str, kingdom: &WgslKingdom) -> serde_json::Value {
    let scaffold = match crate::wgsl::reader::read(text) {
        Ok(k) => serde_json::json!({ "functions": k.functions.iter().map(|f| serde_json::json!({
            "name": f.name,
            "entry_point": f.entry_point,
            "roots": f.roots.iter().map(|r| serde_json::json!({ "kind": format!("{:?}", r.kind), "path": r.path, "point": r.point, "tree": r.tree.to_sexpr() })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>() }),
        Err(e) => serde_json::json!({ "refused": e }),
    };
    let safe = match crate::wgsl::safe_reader::read_safe(text, kingdom) {
        Ok(r) => serde_json::json!({ "tree": r.tree.to_sexpr(), "role": format!("{:?}", r.role) }),
        Err(e) => serde_json::json!({ "refused": e }),
    };
    serde_json::json!({ "file": file, "scaffold": scaffold, "safe": safe })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EVERY COMMITTED KERNEL READS AS NAGA 0.20 READ IT: roots, paths,
    /// points and trees of the scaffold reader, the safe reader's tree and
    /// interface. A refusal must stay a refusal (its words may differ: they
    /// quote naga's errors). Every line that differs is printed, then the
    /// count asserted zero.
    #[test]
    fn every_kernel_reads_as_the_snapshot_says() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let kingdom = WgslKingdom::load_safe();
        let saved = std::fs::read_to_string(root.join("tests/fixtures/reading_snapshot_naga020.jsonl")).expect("the snapshot");
        let saved: Vec<serde_json::Value> = saved.lines().map(|l| serde_json::from_str(l).expect("a snapshot line")).collect();
        let files = kernels(root).expect("the kernels");
        assert_eq!(files.len(), saved.len(), "the kernels and the snapshot's lines");
        let refused = |v: &serde_json::Value| v.get("refused").is_some();
        // (A difference is found with the example's `--full` readings of both builds.)
        let mut differ = Vec::new();
        for (f, want) in files.iter().zip(&saved) {
            let name = f.display().to_string();
            assert_eq!(want["file"], serde_json::Value::String(name.clone()), "the snapshot's order");
            let text = std::fs::read_to_string(root.join(f)).expect("a kernel");
            let full = line(&name, &text, &kingdom);
            let got = digest(&full);
            for part in ["scaffold", "safe"] {
                if got[part] != want[part] {
                    let now = if refused(&full[part]) { full[part]["refused"].to_string().chars().take(160).collect::<String>() } else { "read".into() };
                    differ.push(format!("{name} {part}: {} / {} ({now})", want[part], got[part]));
                }
            }
        }
        for d in &differ {
            println!("DIFFER {d}");
        }
        println!("reading snapshot: {} of {} kernels as naga 0.20 read them", files.len() - differ.len(), files.len());
        assert!(differ.is_empty(), "{} readings differ from the snapshot", differ.len());
    }
}
