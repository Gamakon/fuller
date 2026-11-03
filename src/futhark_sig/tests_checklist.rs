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
use crate::fitness::{scheme_for, scheme_for_problem, Family};
use crate::typed_gep::fits;

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

#[test]
fn every_problem_loads_has_a_default_scheme_and_its_programs_are_exact_on_the_host() {
    let scheme = scheme_for(Family::FutharkSig, None).unwrap();
    assert_eq!(scheme.name, "futhark-slope4-qqs", "the family's default is its newest scheme");
    assert!(scheme.fault.is_none() && scheme.is_flat(), "no default carries the offset (Andrew, 2026-11-03)");
    let mut report = Vec::new();
    for name in PROBLEM_NAMES {
        let p = problem(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(p.name, name);
        let k = load(&p.instance).unwrap_or_else(|e| panic!("{name}: the kingdom does not load: {e}"));
        let sig = signature(&p.instance).unwrap_or_else(|e| panic!("{name}: the signature: {e}"));
        // The interpreter refuses a row it has no meaning for, so this is "every row of the view is covered".
        let interp = Interp::new(&k, &sig).unwrap_or_else(|e| panic!("{name}: the interpreter: {e}"));
        let contract = k.relations().check().unwrap_or_else(|d| panic!("{name}: the contract: {d:?}"));
        let layout = GeneLayout::from_program_size(&contract, k.roots()[0], p.program_size).unwrap_or_else(|e| panic!("{name}: the layout at N = {}: {e}", p.program_size));
        let cases = load_cases(&p).unwrap_or_else(|e| panic!("{name}: the cases: {e}"));
        assert!(cases.iter().any(|c| c.split == Split::Train) && cases.iter().any(|c| c.split == Split::Held), "{name}: training and held-out cases");
        // The problem's own default scheme is of its kind, and a scheme of the other kind is refused by name.
        let own = scheme_for_problem(&p, None).unwrap_or_else(|e| panic!("{name}: the default scheme: {e}"));
        assert_eq!(own.name, p.fitness);
        assert!(own.fault.is_none(), "{name}: a default is never an old scheme");
        assert_eq!(cases.iter().all(|c| c.output.is_some()), p.cases_have_answers(), "{name}: the cases carry answers exactly when the problem is scored against answers");
        let other = if own.is_converged() { "futhark-slope4-gated" } else { "futhark-kmeans-converged" };
        let refused = scheme_for_problem(&p, Some(other)).unwrap_err();
        assert!(refused.contains("not of") && refused.contains(name), "{name}: {refused}");
        // The timing input: the problem's own, of the declared shape.
        assert!(p.timing_n > 0, "{name}: no timing input");
        let stdin = timing_input(&p).unwrap_or_else(|e| panic!("{name}: the timing input: {e}")).unwrap_or_else(|| panic!("{name}: no timing input recipe"));
        let inputs = read_binary(&stdin).unwrap();
        let sizes = sig.sizes(&inputs).unwrap_or_else(|e| panic!("{name}: the timing input is not of the declared shape: {e}"));
        match p.seed {
            Seed::Program { .. } | Seed::Translated { .. } => {
                let mut exact = 0;
                for (what, text) in [("reference", p.reference().unwrap()), ("planted slowdown", p.planted().unwrap())] {
                    let t = build(&k, text).unwrap_or_else(|e| panic!("{name}: the {what} does not build: {e}"));
                    assert!(function_count(&k, &t) < p.program_size, "{name}: the {what} has {} functions, the program size is {}", function_count(&k, &t), p.program_size);
                    fits(&contract, &layout, &t).unwrap_or_else(|e| panic!("{name}: the {what} does not fit N = {}: {e}", p.program_size));
                    if own.is_converged() {
                        // The seed's control (the runner's): its run CONVERGES on every fitness case, compiled with
                        // `futhark c` (the dbpedia cases are the compiled program's: the host interpreter is a minute a
                        // step on them).
                        use crate::futhark_sig::kmeans_converged::{converge, step_of_exe, StepTimer};
                        if !std::process::Command::new("futhark").arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
                            panic!("futhark 0.28.1 is needed for this test");
                        }
                        let exe = compile_c(&program(&k, &sig, &t).unwrap(), &std::env::temp_dir().join(format!("futhark_sig_checklist_{name}_{}_{}", what.replace(' ', "_"), std::process::id()))).unwrap();
                        for c in cases.iter().filter(|c| c.split.in_fitness()) {
                            let r = converge(&c.inputs[0], &c.inputs[1], step_of_exe(&exe, &c.inputs[0], StepTimer::FutharkC)).unwrap();
                            assert!(r.converged, "{name}: the {what}'s run on {} does not converge: {} iterations, finished {}: {}", c.name, r.iterations, r.finished, r.failure);
                            exact += 1;
                        }
                    } else {
                        for c in &cases {
                            assert_eq!(interp.eval(&t, &c.inputs).unwrap_or_else(|e| panic!("{name}: the {what} on {}: {e}", c.name)), *c.answer().unwrap(), "{name}: the {what} on {}", c.name);
                            exact += 1;
                        }
                    }
                }
                // The reference's work at the timing input is under the problem's timing bound.
                let reference = build(&k, p.reference().unwrap()).unwrap();
                let work = interp.work(&reference, &sizes).unwrap();
                assert!(work <= timing_work_cap(work), "{name}: the reference is over its own timing bound");
                report.push(format!("{name}: {} rows, {} cases, reference and planted slowdown {} ({exact} runs), timing input {sizes:?}, work {work}", k.rows.len(), cases.len(), if own.is_converged() { "run to the stopping point on every fitness case" } else { "exact on all" }));
            }
            Seed::NotYetTranslated { .. } => {
                // THE SEED IS NOT YET TRANSLATED (k-means, plan 4c): the slot says so plainly, both accessors refuse with
                // that text, and the problem is the one scored on its objective (no answers to be exact on).
                let state = p.seed_state();
                assert!(state.contains("not yet translated") && state.contains("UltraDim"), "{name}: {state}");
                assert_eq!(p.reference().unwrap_err(), state);
                assert_eq!(p.planted().unwrap_err(), state);
                assert!(own.is_converged(), "{name}: a problem with no seed is scored on its objective, never against answers");
                report.push(format!("{name}: {} rows, {} cases (no answers), {state}; timing input {sizes:?}", k.rows.len(), cases.len()));
            }
        }
    }
    for line in &report {
        eprintln!("{line}");
    }
    assert_eq!(report.len(), PROBLEM_NAMES.len());
}

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
