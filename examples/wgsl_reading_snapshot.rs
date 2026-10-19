//! Writes the reading snapshot (`fuller::wgsl::naga_snapshot`): one JSON line
//! per committed kernel, to stdout, as its digest, or in full with `--full`.
//!
//!   cargo run --release --features wgsl --example wgsl_reading_snapshot > tests/fixtures/reading_snapshot_naga020.jsonl
//!   cargo run --release --features wgsl --example wgsl_reading_snapshot -- --full > readings.jsonl

use std::path::Path;

fn main() -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let kingdom = fuller::wgsl::loader::WgslKingdom::load_safe();
    let full = std::env::args().any(|a| a == "--full");
    for f in fuller::wgsl::naga_snapshot::kernels(root)? {
        let text = std::fs::read_to_string(root.join(&f)).map_err(|e| format!("{}: {e}", f.display()))?;
        let line = fuller::wgsl::naga_snapshot::line(&f.display().to_string(), &text, &kingdom);
        println!("{}", if full { line } else { fuller::wgsl::naga_snapshot::digest(&line) });
    }
    Ok(())
}
