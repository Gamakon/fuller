//! THE POOL CHECK (`docs/PLAN_futhark_kingdom.md` 3g): every program of the committed pools
//! (`kingdoms/futhark/sig/<problem>/pool/{bench,llm,alternatives}.tsv`) read into its problem's kingdom, expressed
//! through a gene of the runner's default program size when it is a program of the root type, and its values on the
//! problem's stored cases fingerprinted (`fuller::futhark_sig::values_fingerprint`, the host interpreter; CPU only).
//!
//! ```text
//! cargo run --release --example futhark_pool_check                 # counts per problem and file
//! cargo run --release --example futhark_pool_check -- --write DIR  # also write DIR/<problem>.<file>.tsv
//! cargo run --release --example futhark_pool_check -- --rename     # rewrite the pools' stack-context names
//! ```
//!
//! The fixture `tests/fixtures/futhark_pool_values/` was written by `--write` before contexts became sets: each
//! program's text in the stack-context kingdom and its values fingerprint, which `futhark_sig::tests_pool` checks.
//! `--rename` (run once, 2026-11-02) rewrote the committed pools from that fixture's texts: each program read by
//! `fuller::futhark_sig::build_stack_named` and written back in the set-context names (a shadowed program would be
//! named and left out); the records of the readers (`*_programs.tsv`, `bench_functions.tsv`) are rewritten alike.

use std::path::{Path, PathBuf};

use fuller::futhark_rings::{build_at, load_pool};
use fuller::futhark_sig::{build_stack_named, load, load_cases, problem, random_gene, sexp, signature, values_fingerprint, Interp, DEFAULT_PROGRAM_SIZE};
use fuller::futhark_table::GeneRng;
use fuller::typed_gep::{express, fits, write_back, GeneLayout};

fn main() {
    if let Err(e) = run() {
        eprintln!("futhark_pool_check: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let out: Option<PathBuf> = args.iter().position(|a| a == "--write").map(|i| args.get(i + 1).map(PathBuf::from).ok_or("--write DIR")).transpose()?;
    if let Some(d) = &out {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let root_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    if args.iter().any(|a| a == "--rename") {
        return rename(root_dir);
    }
    // `--convert PROBLEM`: each line of stdin, a program of the root type in the stack-context names, printed in the
    // set-context names (or the reason it does not convert).
    if let Some(i) = args.iter().position(|a| a == "--convert") {
        let k = load(&problem(args.get(i + 1).ok_or("--convert PROBLEM")?)?.instance)?;
        let root = k.dual(k.roots()[0]).ok_or("no root dual")?.dual.clone();
        for line in std::io::stdin().lines() {
            let line = line.map_err(|e| e.to_string())?;
            match build_stack_named(&k, &line, &root) {
                Ok(t) => println!("{}", sexp(&k, &t)),
                Err(e) => println!("ERROR {e}"),
            }
        }
        return Ok(());
    }
    for pname in ["histogram", "pathfinder"] {
        let p = problem(pname)?;
        let k = load(&p.instance)?;
        let sig = signature(&p.instance)?;
        let interp = Interp::new(&k, &sig)?;
        let contract = k.relations().check().map_err(|d| format!("{pname}: the kingdom check: {:?}", &d[..d.len().min(3)]))?;
        let root = k.roots()[0];
        let layout = GeneLayout::from_program_size(&contract, root, DEFAULT_PROGRAM_SIZE).map_err(|e| e.to_string())?;
        let const_len = k.const_len();
        let cases = load_cases(&p)?;
        let mut rng = GeneRng::new(0x9001_0001);
        for file in ["bench", "llm", "alternatives"] {
            let entries = load_pool(&root_dir.join(format!("kingdoms/futhark/sig/{pname}/pool/{file}.tsv")))?;
            let (mut read, mut at_root, mut expressed) = (0, 0, 0);
            let mut lines = String::from("seq\tdual\tsexp\tvalues\n");
            for (i, e) in entries.iter().enumerate() {
                let t = match build_at(&k, &e.sexp, &e.dual) {
                    Ok(t) => t,
                    Err(err) => {
                        println!("{pname} {file} {}: does not read: {err}", i + 1);
                        lines.push_str(&format!("{}\t{}\t{}\tunread\n", i + 1, e.dual, e.sexp));
                        continue;
                    }
                };
                read += 1;
                if k.ty(&e.dual) == Some(root) {
                    at_root += 1;
                    let ok = fits(&contract, &layout, &t).is_ok()
                        && write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut rng)?, &t).ok().and_then(|g| express(&contract, &layout, &g, &const_len).ok()).map(|x| x.tree()) == Some(t.clone());
                    expressed += usize::from(ok);
                    if !ok {
                        println!("{pname} {file} {}: reads but does not express at N = {DEFAULT_PROGRAM_SIZE}", i + 1);
                    }
                }
                lines.push_str(&format!("{}\t{}\t{}\t{:016x}\n", i + 1, e.dual, e.sexp, values_fingerprint(&interp, &t, &cases)));
            }
            println!("{pname} {file}: {} programs, {read} read, {at_root} of the root type, {expressed} of them express at N = {DEFAULT_PROGRAM_SIZE}", entries.len());
            if let Some(d) = &out {
                std::fs::write(d.join(format!("{pname}.{file}.tsv")), lines).map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}

/// `--rename`: the committed pools and their records, rewritten in the set-context names from their stack-context
/// texts (the pool files from the fixture, column `sexp`; a record's `sexp_or_construct` of a `read` row).
fn rename(root_dir: &Path) -> Result<(), String> {
    for pname in ["histogram", "pathfinder"] {
        let k = load(&problem(pname)?.instance)?;
        let dir = root_dir.join(format!("kingdoms/futhark/sig/{pname}/pool"));
        // The pool files: (file, the column of the dual, the column of the text, the status column if any).
        for (file, dual_col, text_col, status_col) in [("bench.tsv", 4, 5, None), ("llm.tsv", 4, 5, None), ("alternatives.tsv", 4, 5, None), ("seed.tsv", 4, 5, None), ("bench_functions.tsv", 3, 4, Some(2)), ("llm_programs.tsv", 3, 4, Some(2)), ("alternatives_programs.tsv", 3, 4, Some(2))] {
            let path = dir.join(file);
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut out = String::with_capacity(text.len());
            let (mut renamed, mut lost) = (0, 0);
            for (i, line) in text.lines().enumerate() {
                let mut f: Vec<String> = line.split('\t').map(str::to_string).collect();
                let is_program = i > 0 && status_col.is_none_or(|c| f.get(c).map(String::as_str) == Some("read"));
                if is_program {
                    match build_stack_named(&k, &f[text_col], &f[dual_col]) {
                        Ok(t) => {
                            f[text_col] = sexp(&k, &t);
                            renamed += 1;
                        }
                        Err(e) => {
                            println!("{pname} {file} line {}: LOST, {e}: {}", i + 1, f[text_col]);
                            lost += 1;
                            continue;
                        }
                    }
                }
                out.push_str(&f.join("\t"));
                out.push('\n');
            }
            std::fs::write(&path, out).map_err(|e| format!("{}: {e}", path.display()))?;
            println!("{pname} {file}: {renamed} programs renamed, {lost} lost");
        }
    }
    Ok(())
}
