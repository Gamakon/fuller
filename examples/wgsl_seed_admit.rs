//! Whether a problem's seed kernels enter feasible on the host, no GPU: each file is read into the safe
//! grammar, checked by the legality checker under the problem's `safety.contract` (the layer 7 mode the fit
//! runs, `--layer7 strict|arena|arena+rows`), encoded into a typed gene and expressed, rendered and
//! validated. One line a kernel: `admitted`, or the stage and the reason.
//!
//!   cargo run --release --features wgsl --example wgsl_seed_admit -- --problem problems/v24_ingest5b [--layer7 arena+rows] <seed.wgsl or dir>...

use std::path::{Path, PathBuf};

use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::safe_karva::render_tree_with;
use fuller::wgsl::typed_gene::Alphabets;

fn files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path).map(|rd| rd.flatten().map(|e| e.path()).collect()).unwrap_or_default();
        entries.sort();
        for p in entries {
            files(&p, out);
        }
    } else if path.extension().is_some_and(|x| x == "wgsl") {
        out.push(path.to_path_buf());
    }
}

fn main() -> Result<(), String> {
    let mut problem = None;
    let mut layer7 = "strict".to_string();
    let mut inputs = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--problem" => problem = Some(args.next().ok_or("--problem needs a directory")?),
            "--layer7" => layer7 = args.next().ok_or("--layer7 needs a mode")?,
            _ => inputs.push(a),
        }
    }
    let dir = problem.ok_or("--problem <dir> is required")?;
    let mut contract = fuller::wgsl::flow::contract_of(Path::new(&dir))?;
    contract = match layer7.as_str() {
        "strict" => contract,
        "arena" => contract.with_arena_mode(true),
        "arena+rows" => contract.with_arena_mode(true).with_rows_mode(true),
        other => return Err(format!("--layer7 {other}: strict, arena or arena+rows")),
    };
    let kingdom = WgslKingdom::load_safe();
    let mut alphabets = Alphabets::default();
    let mut list = Vec::new();
    for i in &inputs {
        files(Path::new(i), &mut list);
    }
    let mut admitted = 0;
    for f in &list {
        let text = std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        let verdict = (|| -> Result<(), String> {
            let read = fuller::wgsl::safe_reader::read_safe(&text, &kingdom).map_err(|e| format!("read: {e}"))?;
            let k = read.kingdom.as_ref().unwrap_or(&kingdom);
            let under = fuller::wgsl::knob::Under { role: read.role.clone(), contract: Some(contract.clone()) };
            fuller::wgsl::knob::check_legal_under(&read.tree, k, &under).map_err(|(stage, e)| format!("{}: {e}", stage.name()))?;
            let a = alphabets.of(k, &read.role).as_ref().map_err(|g| format!("alphabet: {}: {}", g.reason, g.detail))?;
            let enc = fuller::wgsl::typed_gene::encode(a, k, &read.tree).map_err(|g| format!("encode: {}: {}", g.reason, g.detail))?;
            let back = fuller::wgsl::typed_gene::express(a, k, &enc).map_err(|g| format!("express: {}: {}", g.reason, g.detail))?;
            let rendered = render_tree_with(&back, k, &read.role).map_err(|e| format!("render: {e}"))?;
            fuller::wgsl::render::validate_text(&rendered.text).map_err(|e| format!("validate: {e}"))?;
            fuller::wgsl::knob::check_legal_under(&back, k, &under).map_err(|(stage, e)| format!("after the gene, {}: {e}", stage.name()))?;
            Ok(())
        })();
        match verdict {
            Ok(()) => {
                admitted += 1;
                println!("{}\tadmitted", f.display());
            }
            Err(e) => println!("{}\t{}", f.display(), e.replace('\n', " ")),
        }
    }
    println!("# {admitted} of {} admitted (layer 7 {layer7})", list.len());
    Ok(())
}
