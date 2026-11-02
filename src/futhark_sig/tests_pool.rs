//! THE POOLS ACROSS THE SET CONTEXTS (`docs/PLAN_futhark_kingdom.md` 3g): every program of the committed pools
//! (`kingdoms/futhark/sig/<problem>/pool/{bench,llm,alternatives}.tsv`), recorded with its text and its values on the
//! stored cases under the stack contexts (`tests/fixtures/futhark_pool_values/`, `examples/futhark_pool_check`), reads
//! into the set-context kingdom from its old text (`build_stack_named`), computes the same values on every stored
//! case (host interpreter), is written in the committed pool with its new text, and expresses through a gene of the
//! runner's default program size when it is of the root type.

use super::*;
use crate::futhark_rings::{build_at, load_pool};
use crate::typed_gep::{express, fits, write_back};

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/futhark_pool_values").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The binder names of the stack contexts and of the set contexts, mapped one to the other in a text: the old text
/// with every old variable name replaced by its new one must be the new text, so the two differ only in binder
/// names.
fn names_only(old: &str, new: &str) -> bool {
    let tok = |t: &str| t.replace('(', " ( ").replace(')', " ) ").split_whitespace().map(str::to_string).collect::<Vec<_>>();
    let (a, b) = (tok(old), tok(new));
    let is_old_var = |x: &str| x.strip_prefix(['i', 'j', 's']).is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
    let is_new_var = |x: &str| x.strip_prefix(['i', 's']).is_some_and(|r| r.trim_start_matches(|c: char| c.is_ascii_digit()).starts_with('_'));
    a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x == y || (is_old_var(x) && is_new_var(y)))
}

#[test]
fn every_pool_program_reads_with_the_same_values_and_expresses() {
    let mut report = Vec::new();
    for pname in ["histogram", "pathfinder"] {
        let p = problem(pname).unwrap();
        let k = load(&p.instance).unwrap();
        let sig = signature(&p.instance).unwrap();
        let interp = Interp::new(&k, &sig).unwrap();
        let contract = k.relations().check().unwrap();
        let root = k.roots()[0];
        let layout = GeneLayout::from_program_size(&contract, root, DEFAULT_PROGRAM_SIZE).unwrap();
        let const_len = k.const_len();
        let cases = load_cases(&p).unwrap();
        let mut rng = GeneRng::new(0x9001_0002);
        for file in ["bench", "llm", "alternatives"] {
            let committed = load_pool(&Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("kingdoms/futhark/sig/{pname}/pool/{file}.tsv"))).unwrap();
            let old = fixture(&format!("{pname}.{file}.tsv"));
            let old: Vec<Vec<&str>> = old.lines().skip(1).map(|l| l.split('\t').collect()).collect();
            assert_eq!(old.len(), committed.len(), "{pname} {file}: the pool and the record differ in length");
            let (mut same, mut lost, mut at_root, mut expressed) = (0, Vec::new(), 0, 0);
            for (o, e) in old.iter().zip(&committed) {
                let (seq, dual, text, values) = (o[0], o[1], o[2], o[3]);
                assert_eq!(dual, e.dual, "{pname} {file} {seq}: the dual changed");
                let t = match build_stack_named(&k, text, dual) {
                    Ok(t) => t,
                    Err(err) => {
                        lost.push(format!("{seq}: {err}"));
                        continue;
                    }
                };
                assert_eq!(format!("{:016x}", values_fingerprint(&interp, &t, &cases)), values, "{pname} {file} {seq}: the values changed: {text}");
                assert_eq!(e.sexp, sexp(&k, &t), "{pname} {file} {seq}: the committed pool's text is not the program's");
                assert!(names_only(text, &e.sexp), "{pname} {file} {seq}: the old and new texts differ in more than binder names:\n{text}\n{}", e.sexp);
                assert_eq!(build_at(&k, &e.sexp, dual).unwrap(), t, "{pname} {file} {seq}: the new text does not read back");
                same += 1;
                if k.ty(dual) == Some(root) {
                    at_root += 1;
                    fits(&contract, &layout, &t).unwrap_or_else(|x| panic!("{pname} {file} {seq}: does not fit N = {DEFAULT_PROGRAM_SIZE}: {x}"));
                    let g = write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut rng).unwrap(), &t).unwrap();
                    assert_eq!(express(&contract, &layout, &g, &const_len).unwrap().tree(), t, "{pname} {file} {seq}: does not express");
                    expressed += 1;
                }
            }
            eprintln!("{pname} {file}: {} programs, {same} read with the same values, {at_root} of the root type, {expressed} express; lost {}: {lost:?}", committed.len(), lost.len());
            report.push((pname, file, committed.len(), same, expressed, lost.len()));
        }
    }
    // Measured 2026-11-02 (docs/PLAN_futhark_kingdom.md 3g): (problem, file, programs, read with the same values,
    // expressed at N = 32, lost to shadowing).
    assert_eq!(report, [("histogram", "bench", 10, 10, 1, 0), ("histogram", "llm", 48, 48, 48, 0), ("histogram", "alternatives", 577, 577, 577, 0), ("pathfinder", "bench", 9, 9, 0, 0), ("pathfinder", "llm", 45, 45, 45, 0), ("pathfinder", "alternatives", 153, 153, 153, 0)]);
}
