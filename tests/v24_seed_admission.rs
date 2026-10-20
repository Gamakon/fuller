//! The seeds run 5 admitted are still admitted (no GPU). Run 5 (fuller 9d770424, `logs/V24/run5/ledger.tsv`) entered
//! 15 of its 32 seeds feasible under `problems/v24_ingest5b` at layer 7 `arena+rows` (ledger gen 0 row r is
//! `seed<r-1>`, matched by the workgroup arrays and entry grids of `logs/V24/run5/kernels/g000_r<r>.wgsl`; rows 0 and
//! 33 are not seeds of this directory). Run 6c lost 10 of them to a rule change: `dc74bbed` made a u32 literal past
//! 2^24 exact, so a hash multiply (2654435761u) tripped `index_wraps`. This test needs the `wraps_up` fix in `check.rs`
//! (mtgep-seeds) to pass and fails on main until it lands. Each kernel is read into the safe grammar, checked under the
//! problem's contract exactly as `wgsl_fit` checks a gene, encoded into a typed gene of its own interface, expressed,
//! rendered, validated by naga and checked again. A rule change that loses one fails here by name.
#![cfg(feature = "wgsl")]

use std::path::Path;

use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::safe_karva::render_tree_with;
use fuller::wgsl::typed_gene::Alphabets;

/// The run 5 seeds (`tests/fixtures/v24_seeds5/<name>.wgsl`, copies of `logs/V24/run5/seeds`).
const ADMITTED: [&str; 15] = [
    "seed03", "seed04", "seed05", "seed06", "seed07", "seed08", "seed11", "seed12", "seed13", "seed16", "seed23", "seed24", "seed25", "seed26", "seed27",
];

#[test]
fn the_seeds_run_5_admitted_are_still_admitted() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let contract = fuller::wgsl::flow::contract_of(&root.join("problems/v24_ingest5b")).expect("the problem's contract").with_arena_mode(true).with_rows_mode(true);
    let kingdom = WgslKingdom::load_safe();
    let mut alphabets = Alphabets::default();
    let mut lost = Vec::new();
    for name in ADMITTED {
        let text = std::fs::read_to_string(root.join(format!("tests/fixtures/v24_seeds5/{name}.wgsl"))).unwrap_or_else(|e| panic!("{name}: {e}"));
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
        if let Err(e) = verdict {
            lost.push(format!("{name}: {}", e.chars().take(300).collect::<String>()));
        }
    }
    assert!(lost.is_empty(), "{} of {} seeds run 5 admitted are lost:\n{}", lost.len(), ADMITTED.len(), lost.join("\n"));
}
