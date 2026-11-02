//! THE POOL CHECK (`docs/PLAN_futhark_kingdom.md` 3g): every program of the committed pools
//! (`kingdoms/futhark/sig/<problem>/pool/{bench,llm,alternatives}.tsv`) read into its problem's kingdom, expressed
//! through a gene of the runner's default program size when it is a program of the root type, and its values on the
//! problem's stored cases fingerprinted (`fuller::futhark_sig::values_fingerprint`, the host interpreter; CPU only).
//!
//! ```text
//! cargo run --release --example futhark_pool_check                 # counts per problem and file
//! cargo run --release --example futhark_pool_check -- --write DIR  # also write DIR/<problem>.<file>.tsv
//! ```
//!
//! The fixture `tests/fixtures/futhark_pool_values/` was written by `--write` before contexts became sets: each
//! program's text in the stack-context kingdom and its values fingerprint, which `futhark_sig::tests_pool` checks.

use std::path::{Path, PathBuf};

use fuller::futhark_rings::{build_at, load_pool};
use fuller::futhark_sig::{load, load_cases, problem, random_gene, signature, values_fingerprint, Interp, DEFAULT_PROGRAM_SIZE};
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
