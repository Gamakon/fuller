//! THE READING SNAPSHOT (2026-10-19, the naga/wgpu 30 upgrade,
//! docs/PLAN_naga_wgpu_30.md section 3; Codex's review: a count of kernels
//! read cannot show that roots, holes or trees stayed the same). For every
//! committed kernel, one JSON line: the scaffold reader's roots per function
//! (kind, path, point, tree) and the safe reader's tree and interface (its
//! store guard, the emitter's text, left out), or
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

/// The scaffold reader's names carry naga's arena indices (`let.x@12`,
/// `local.i@3`, `call.f@40`), which move whenever naga's lowering appends
/// another expression or local (naga 30's short-circuit locals, its constant
/// folding), and a compose names its type by the type arena's index
/// (`naga.Compose.type7`, which moves when naga 30 adds a type): each
/// `@<number>` and each `.type<number>` is renumbered by first appearance in
/// the kernel's reading, so two readings that name the same things in the
/// same order compare equal.
fn renumbered(text: &str) -> String {
    // (Keyed by the whole name, `local.i@3` and `let.i@3` apart.)
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut out = String::with_capacity(text.len());
    let mut word = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '@' {
            if c.is_alphanumeric() || c == '.' || c == '_' || c == '#' {
                word.push(c);
            } else {
                word.clear();
            }
            out.push(c);
            continue;
        }
        let mut digits = String::new();
        while let Some(d) = chars.peek().copied().filter(char::is_ascii_digit) {
            digits.push(d);
            chars.next();
        }
        out.push('@');
        if digits.is_empty() {
            word.push('@');
            continue;
        }
        let key = format!("{word}@{digits}");
        let n = seen.len();
        let k = *seen.entry(key).or_insert(n);
        out.push_str(&format!("#{k}"));
        word.push_str(&format!("@#{k}"));
    }
    // The types, `Compose.type<number>`, apart from the arena names above.
    let mut types: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut renamed = String::with_capacity(out.len());
    let mut rest = out.as_str();
    while let Some(i) = rest.find("Compose.type") {
        let (head, tail) = rest.split_at(i + "Compose.type".len());
        renamed.push_str(head);
        let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
        if digits.is_empty() {
            rest = tail;
            continue;
        }
        let n = types.len();
        renamed.push_str(&format!("#{}", *types.entry(digits.clone()).or_insert(n)));
        rest = &tail[digits.len()..];
    }
    renamed.push_str(rest);
    renamed
}

/// A role as the snapshot recorded it (naga 0.20, 2026-10-19 morning): the
/// fields without their types, and none of the interface's later growth: the
/// structs and the other uniform blocks, the rows of a runtime-sized array's
/// length (`len.<binding>`), of a struct binding's leaves, of another uniform
/// block's fields and of an array or matrix field of the first. What naga
/// read of a kernel is what the digest holds.
fn legacy_role(role: &crate::wgsl::render::Role) -> String {
    // (Written field by field, as the derived Debug of the earlier Role did.)
    struct Legacy<'a> {
        role: &'a crate::wgsl::render::Role,
        prologue: String,
        per_kernel: Vec<&'a crate::wgsl::render::PerKernel>,
    }
    struct LegacyField<'a>(&'a crate::wgsl::render::Field);
    impl std::fmt::Debug for LegacyField<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Field").field("name", &self.0.name).field("role", &self.0.role).finish()
        }
    }
    impl std::fmt::Debug for Legacy<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let r = self.role;
            let fields: Vec<LegacyField> = r.fields.iter().map(LegacyField).collect();
            f.debug_struct("Role")
                .field("name", &r.name)
                .field("prologue", &self.prologue)
                .field("items", &r.items)
                .field("per_kernel", &self.per_kernel)
                .field("bindings", &r.bindings)
                .field("uniform_var", &r.uniform_var)
                .field("uniform_struct", &r.uniform_struct)
                .field("fields", &fields)
                .field("input", &r.input)
                .field("output", &r.output)
                .field("guard", &"")
                .field("entry", &r.entry)
                .field("entries", &r.entries)
                .finish()
        }
    }
    let later_var = |n: &str, prefix: &str| role.more_uniforms.iter().any(|u| n.strip_prefix(prefix).is_some_and(|r| r.starts_with(&format!("{}_", u.var))));
    let later_field = |n: &str| {
        role.fields.iter().filter(|f| f.ty.starts_with("array<") || f.ty.starts_with("mat")).any(|f| n == format!("load.{}_{}", role.uniform_var, f.name))
    };
    let per_kernel = role.per_kernel.iter().filter(|p| {
        let n = p.name.as_str();
        let leaf = ["load.", "store.", "atomic."].iter().any(|pre| n.strip_prefix(pre).is_some_and(|r| r.contains('.')));
        !(n.starts_with("len.") || leaf || later_var(n, "load.") || later_var(n, "uniform.") || later_field(n))
    });
    // (The prologue's struct lines as the Gram's u32 fields wrote them.)
    let prologue = role.prologue.lines().filter(|l| !l.starts_with("struct ") || role.fields.iter().all(|f| f.ty == "u32") && l.starts_with(&format!("struct {} ", role.uniform_struct))).collect::<Vec<_>>().join("\n");
    format!("{:?}", Legacy { role, prologue, per_kernel: per_kernel.collect() })
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
        Ok(r) => serde_json::json!({ "tree": r.tree.to_sexpr(), "role": legacy_role(&r.role) }),
        Err(e) => serde_json::json!({ "refused": e }),
    };
    let scaffold: serde_json::Value = serde_json::from_str(&renumbered(&scaffold.to_string())).expect("the renumbered reading is JSON");
    serde_json::json!({ "file": file, "scaffold": scaffold, "safe": safe })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EVERY COMMITTED KERNEL READS AS NAGA 0.20 READ IT: roots, paths,
    /// points and trees of the scaffold reader, the safe reader's tree and
    /// interface. A refusal must stay a refusal (its words may differ: they
    /// quote naga's errors). The changes naga 30 makes right are recorded,
    /// kernel by kernel, in tests/fixtures/reading_snapshot_naga30_changes.tsv
    /// (2026-10-19: 160 safe readings that are the same tree plus the zero
    /// sets of a loop-local var naga 30 zeroes at each pass, 165 scaffold
    /// readings with those zero stores, draft_i32's FirstLeadingBit,
    /// draft_matrices refused); every other line must be equal, and every
    /// recorded line must still differ.
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
        let recorded: std::collections::BTreeSet<(String, String)> = std::fs::read_to_string(root.join("tests/fixtures/reading_snapshot_naga30_changes.tsv")).expect("the recorded changes").lines().skip(1).filter_map(|l| {
            let mut c = l.split('\t');
            Some((c.next()?.to_string(), c.next()?.to_string()))
        }).collect();
        let mut seen = std::collections::BTreeSet::new();
        let mut differ = Vec::new();
        for (f, want) in files.iter().zip(&saved) {
            let name = f.display().to_string();
            assert_eq!(want["file"], serde_json::Value::String(name.clone()), "the snapshot's order");
            let text = std::fs::read_to_string(root.join(f)).expect("a kernel");
            let full = line(&name, &text, &kingdom);
            let got = digest(&full);
            for part in ["scaffold", "safe"] {
                if got[part] != want[part] && recorded.contains(&(name.clone(), part.to_string())) {
                    seen.insert((name.clone(), part.to_string()));
                    continue;
                }
                if got[part] != want[part] {
                    let now = if refused(&full[part]) { full[part]["refused"].to_string().chars().take(160).collect::<String>() } else { "read".into() };
                    differ.push(format!("{name} {part}: {} / {} ({now})", want[part], got[part]));
                }
            }
        }
        for d in &differ {
            println!("DIFFER {d}");
        }
        println!("reading snapshot: {} readings recorded as naga 30's changes, {} differ otherwise, of {} kernels", seen.len(), differ.len(), files.len());
        assert!(differ.is_empty(), "{} readings differ from the snapshot", differ.len());
        let stale: Vec<_> = recorded.difference(&seen).collect();
        assert!(stale.is_empty(), "recorded changes that no longer differ: {stale:?}");
    }
}
