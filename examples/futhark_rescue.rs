//! The compiler rescue: Fuller's e-graph finds an equivalent program that the
//! Futhark compiler CAN compile, for a program it cannot.
//!
//!   cargo run --release --features wgsl --example futhark_rescue -- [options]
//!
//!   --root NAME     the program's root type (default f32.real)
//!   --size N        the program size (default 4)
//!   --count N       how many random genes to draw (default 12)
//!   --index N       which of them is the pathological one (default 3)
//!   --candidates N  how many equivalent programs to extract (default 32)
//!   --iters N       the saturation bound (default 4)
//!   --limit S       the compile wall-clock limit in seconds (default 120)
//!   --keep          keep the compiled programs on disk
//!
//! The procedure is the specification's:
//!
//!   1. reproduce the compilation timeout;
//!   2. build the e-graph of the pathological program;
//!   3. extract a bounded number of distinct equivalent candidates;
//!   4. compile each with the wall-clock limit;
//!   5. execute the ones that compiled;
//!   6. compare against the independent interpreter.
//!
//! Everything is recorded: the gene, the e-graph size, the saturation time,
//! each candidate's hash, cost, compile time and verdict, and whether the
//! program's output agrees with the interpreter's.
//!
//! The genes are drawn with the same generator and the same seed as the sweep
//! that found the pathological program, so the program this reproduces is the
//! same program the sweep found.

use std::collections::BTreeMap;
use std::time::Duration;

use fuller::futhark_egraph;
use fuller::futhark_exec::{compile_program, run_compiled, Verdict};
use fuller::futhark_interp::Value;
use fuller::futhark_kingdom::{FutharkKingdom, GeneRng};
use fuller::geneframe::Ty;
use fuller::kingdom_contract::KingdomContract;
use fuller::typed_gep::{self, GeneLayout, Tree};

/// The seed the sweep that found the pathological program used.
const SEED: u64 = 0x5EED_1234_ABCD_0001;

/// The Futhark program around an expression.
fn program(scalar_root: bool, expr: &str) -> String {
    let result = if scalar_root { "f32" } else { "[]f32" };
    format!("-- a program of the Futhark-style kingdom\nentry main (xs: []f32) : {result} = {expr}\n")
}

/// A short, stable hash of a candidate's source: the record needs to name
/// candidates, and a hash is shorter than the program.
fn hash(text: &str) -> String {
    // FNV-1a, 64 bit: not a cryptographic hash, just a name.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

/// The gene's cells, as the record wants them.
fn gene_text(gene: &[u32]) -> String {
    gene.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")
}

fn main() -> Result<(), String> {
    let mut root_name = "f32.real".to_string();
    let mut size = 4usize;
    let mut count = 12usize;
    let mut index = 3usize;
    let mut want = 32usize;
    let mut iters = 4u32;
    let mut limit = 120u64;
    let mut keep = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--root" => root_name = args.next().ok_or("--root needs a value")?,
            "--size" => size = args.next().ok_or("--size needs a value")?.parse().map_err(|_| "--size needs a number")?,
            "--count" => count = args.next().ok_or("--count needs a value")?.parse().map_err(|_| "--count needs a number")?,
            "--index" => index = args.next().ok_or("--index needs a value")?.parse().map_err(|_| "--index needs a number")?,
            "--candidates" => want = args.next().ok_or("--candidates needs a value")?.parse().map_err(|_| "--candidates needs a number")?,
            "--iters" => iters = args.next().ok_or("--iters needs a value")?.parse().map_err(|_| "--iters needs a number")?,
            "--limit" => limit = args.next().ok_or("--limit needs a value")?.parse().map_err(|_| "--limit needs a number")?,
            "--keep" => keep = true,
            other => return Err(format!("unknown option {other}")),
        }
    }

    let k = FutharkKingdom::load()?;
    let contract: KingdomContract = k.relations().check().map_err(|d| d.join("; "))?;
    let root: Ty = k.ty(&root_name).ok_or_else(|| format!("no root {root_name}"))?;
    let layout = GeneLayout::from_program_size(&contract, root, size).map_err(|e| format!("{e:?}"))?;
    let const_len: BTreeMap<Ty, u32> = k.const_len();

    // --- the pathological gene, regenerated from its seed -------------------
    let mut rng = GeneRng::new(SEED);
    let mut gene = vec![0u32; layout.width];
    let mut chosen: Option<(Vec<u32>, Tree)> = None;
    for i in 0..count {
        for (pos, cell) in gene.iter_mut().enumerate() {
            let alphabet = layout.alphabet(&contract, pos, &const_len);
            if alphabet.is_empty() {
                return Err(format!("cell {pos} has an empty alphabet"));
            }
            *cell = alphabet[rng.below(alphabet.len())];
        }
        let e = typed_gep::express(&contract, &layout, &gene, &const_len)
            .map_err(|defect| format!("gene {i} was refused: {defect:?}"))?;
        if i == index {
            chosen = Some((gene.clone(), e.tree()));
        }
    }
    let (gene, tree) = chosen.ok_or_else(|| format!("no gene {index} among {count}"))?;
    let expr = k.render(&tree)?;
    let scalar = root_name == "f32.real";
    let source = program(scalar, &expr);

    println!("== the pathological program ==");
    println!("root          {root_name}, program size {size}, gene width {}", layout.width);
    println!("gene index    {index} of {count}, seed {SEED:#x}");
    println!("gene          [{}]", gene_text(&gene));
    println!("expression    {expr}");
    println!("hash          {}", hash(&expr));

    let dir = std::env::temp_dir().join("futhark_rescue").join(format!("p{}", std::process::id()));

    // --- 1. reproduce the timeout -------------------------------------------
    println!("\n== 1. the original, compiled ==");
    let (verdict, elapsed) = compile_program(&source, &dir, Duration::from_secs(limit));
    match &verdict {
        Verdict::Compiled => println!("compiled in {elapsed:?}"),
        Verdict::Refused(m) => println!("refused after {elapsed:?}: {}", m.lines().next().unwrap_or("")),
        Verdict::TimedOut => println!("did not converge: killed after {elapsed:?}"),
    }

    // --- 2-3. the e-graph and its candidates --------------------------------
    println!("\n== 2. the e-graph ==");
    let sat = futhark_egraph::saturate_tree(&tree, &k, want, iters)?;
    println!("term          {}", sat.term);
    println!("sort          {}", sat.sort);
    println!(
        "e-graph       {} e-nodes, saturated in {:?} ({} iterations)",
        sat.size, sat.elapsed, iters
    );
    println!("\n== 3. {} equivalent candidates ==", sat.candidates.len());

    // --- 4-6. compile, run, compare ----------------------------------------
    println!("\n== 4. each candidate, compiled and run ==");
    let input: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.25).collect();
    let expected = k.eval(&tree, &input)?;
    let want_value: Vec<f32> = match &expected {
        Value::Arr(a) => a.clone(),
        Value::Real(r) => vec![*r],
        Value::Fun(_) => return Err("the root of a program is not a function".to_string()),
    };

    let mut rescued: Option<String> = None;
    let mut compiled_ok = 0usize;
    let mut timed_out = 0usize;
    let mut refused = 0usize;
    let mut wrong = 0usize;

    for (rank, cand) in sat.candidates.iter().enumerate() {
        let cand_program = program(scalar, &cand.source);
        let cand_dir = dir.join(format!("c{rank}"));
        let (verdict, elapsed) = compile_program(&cand_program, &cand_dir, Duration::from_secs(limit));
        let verdict_text = match &verdict {
            Verdict::Compiled => {
                compiled_ok += 1;
                "compiled".to_string()
            }
            Verdict::Refused(m) => {
                refused += 1;
                format!("refused: {}", m.lines().next().unwrap_or(""))
            }
            Verdict::TimedOut => {
                timed_out += 1;
                "TIMED OUT".to_string()
            }
        };
        let mut run_text = String::new();
        if verdict == Verdict::Compiled {
            match run_compiled(scalar, &input, &cand_dir) {
                Ok(got) => {
                    let same = got.len() == want_value.len()
                        && got.iter().zip(want_value.iter()).all(|(a, b)| {
                            a == b || (a.is_nan() && b.is_nan())
                        });
                    if same {
                        run_text = format!("runs, agrees with the interpreter ({got:?})");
                        if rescued.is_none() && cand.source != expr {
                            rescued = Some(cand.source.clone());
                        }
                    } else {
                        wrong += 1;
                        run_text = format!("runs, DISAGREES with the interpreter: {got:?} vs {want_value:?}");
                    }
                }
                Err(e) => run_text = format!("compiled but did not run: {}", e.lines().next().unwrap_or("")),
            }
        }
        println!(
            "{rank:>3}. cost {:>3}  hash {}  gene {:<5}  {verdict_text:<28} {:>10.3?}  {run_text}",
            cand.cost,
            hash(&cand.source),
            if cand.is_representable() { "yes" } else { "no" },
            elapsed,
        );
        println!("      {}", cand.source);
    }

    println!("\n== the record ==");
    println!("original      compiled: no (the compiler did not converge within {limit}s)");
    println!(
        "candidates    {}, of which {compiled_ok} compiled, {refused} were refused, {timed_out} timed out",
        sat.candidates.len()
    );
    println!("correctness   {compiled_ok} compiled candidates ran; {wrong} disagreed with the interpreter");
    match &rescued {
        Some(src) => {
            println!("rescue        an equivalent candidate compiles and runs correctly:");
            println!("              {src}");
            println!("              hash {}", hash(src));
        }
        None => println!("rescue        NONE: no equivalent candidate compiled and ran correctly"),
    }
    if keep {
        println!("kept          {}", dir.display());
    } else {
        let _ = std::fs::remove_dir_all(&dir);
    }
    Ok(())
}
