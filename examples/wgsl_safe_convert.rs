//! The drafted kernels into the safeWGSL grammar (`src/wgsl/safe_reader.rs`):
//! for every `kNN.wgsl` in a directory, the safe tree or the refusal by
//! name; a converted tree is rendered, validated by naga and written as
//! `<out>/kNN.safe.wgsl` with its s-expression beside it.
//!
//!   cargo run --release --features wgsl --example wgsl_safe_convert -- <dir> [--out <dir>]

use std::path::PathBuf;

use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::render::{gram_role, validate_text, Renderer};
use fuller::wgsl::safe_reader::{read_safe, statement_outline};
use fuller::wgsl::table::FunctionTable;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or("usage: wgsl_safe_convert <dir> [--out <dir>]")?);
    let mut out = dir.join("safe");
    let mut outline: Option<String> = None;
    while let Some(a) = args.next() {
        if a == "--out" {
            out = PathBuf::from(args.next().ok_or("--out needs a value")?);
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
    let (table, role) = (FunctionTable::safe(), gram_role());
    let rr = Renderer::new(&kingdom, &table, &role);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
    files.sort();
    let (mut converted, mut valid) = (0, 0);
    let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();
    println!("{:<6} {:<10} {:>6} {:<8} reason / notes", "kernel", "converted", "nodes", "valid");
    for f in &files {
        let id = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let src = std::fs::read_to_string(f).map_err(|e| e.to_string())?;
        match read_safe(&src, &kingdom) {
            Err(e) => {
                let key: String = e.split(':').next_back().unwrap_or(&e).trim().chars().take(60).collect();
                *reasons.entry(key).or_default() += 1;
                println!("{id:<6} {:<10} {:>6} {:<8} {}", "no", "-", "-", &e[..e.len().min(600)]);
            }
            Ok(r) => {
                converted += 1;
                let (text, ok) = match rr.kernel(&r.tree) {
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
            }
        }
    }
    println!("\n{} of {} converted, {} valid; refusals:", converted, files.len(), valid);
    for (k, v) in &reasons {
        println!("  {v:3}  {k}");
    }
    Ok(())
}
