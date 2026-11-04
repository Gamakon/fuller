//! THE CHECKLIST, CHECKED (`docs/CHECKLIST_futhark_problem.md`, `docs/PLAN_futhark_kingdom.md` step 4): what every
//! Futhark problem of the one runner gets. The cheap claims are checked here for every problem of [`PROBLEM_NAMES`]:
//! the problem loads (its view, its kingdom, its signature, an interpreter that covers every row), it has a default
//! fitness scheme of its own kind, its reference and planted slowdown build from their s-expressions and fit the
//! layout at the problem's program size, its reference and planted slowdown are exact on every stored case through
//! the host interpreter (a problem scored on its own objective, k-means, stores no answers: its control is that the
//! seed's run finishes on every fitness case through the host interpreter), it has a timing input of the declared
//! shape; and the checklist's table has a column for every problem and a row for every item, every cell filled, a
//! "not yet" cell with its reason.

use super::*;

/// The items of the checklist, in the table's order (one row each).
pub const ITEMS: [&str; 20] = [
    "problem entry (`futhark_sig::problem`)",
    "view (`view.tsv`, the master)",
    "interpreter covers every row",
    "default fitness scheme",
    "slope check (`--check`)",
    "quality weight (stage 2, one flat vector)",
    "reference exact, host interpreter",
    "reference exact, compiled",
    "planted slowdown, equal and recoverable",
    "stored cases (train, held, test)",
    "timing input",
    "interleaved one-upload timing",
    "timing-case output check",
    "ring canvas",
    "operator mix with per-operator logging",
    "stream, killme.trigger, checkpoints",
    "hall of fame, ledger with lineage",
    "aa_gate.tsv",
    "static cost (speed input where a scheme asks)",
    "measured GPU cost",
];

/// The checklist's table: one column a problem, one row an item, every cell filled.
#[test]
fn the_checklist_table_has_every_problem_and_every_item() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/CHECKLIST_futhark_problem.md");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let cells = |line: &str| line.trim().trim_matches('|').split('|').map(str::trim).map(str::to_string).collect::<Vec<_>>();
    let header: Vec<String> = std::iter::once("item".to_string()).chain(PROBLEM_NAMES.iter().map(|s| s.to_string())).collect();
    let start = text.lines().position(|l| cells(l) == header).unwrap_or_else(|| panic!("the checklist has no table headed {header:?}"));
    let rows: Vec<Vec<String>> = text.lines().skip(start + 2).take_while(|l| l.trim_start().starts_with('|')).map(cells).collect();
    assert_eq!(rows.iter().map(|r| r[0].as_str()).collect::<Vec<_>>(), ITEMS.to_vec(), "the rows are the items, in order");
    for r in &rows {
        assert_eq!(r.len(), header.len(), "the row {:?} has {} cells", r[0], r.len());
        for (p, cell) in PROBLEM_NAMES.iter().zip(&r[1..]) {
            assert!(!cell.is_empty(), "{p}: the cell {:?} is empty", r[0]);
            if cell.starts_with("not yet") {
                assert!(cell.len() > "not yet: ".len() && cell.contains(':'), "{p}: the cell {:?} says not yet without its reason", r[0]);
            }
        }
    }
    // Every "not yet" is counted in the summary line, so the count cannot drift from the table.
    let not_yet = rows.iter().flat_map(|r| &r[1..]).filter(|c| c.starts_with("not yet")).count();
    assert!(text.contains(&format!("**{not_yet} cells say not yet**")), "the summary line counts {not_yet} not-yet cells");
}
