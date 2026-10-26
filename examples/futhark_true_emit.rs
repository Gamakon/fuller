//! The Futhark-style kingdom, end to end: random genes in, Futhark source
//! out, and (with `--compile`) the real Futhark compiler's verdict.
//!
//!   cargo run --release --features wgsl --example futhark_true_emit -- [options]
//!
//!   --root arr|f32   the program's root type (default arr)
//!   --size N         the program size, the number of function nodes (default 4)
//!   --count N        how many random genes to draw (default 100)
//!   --compile        run `futhark c` on each emitted program
//!   --show N         print the first N emitted programs
//!
//! What this proves, in order:
//!
//!   1. **Legality by construction.** Every random gene validates and
//!      expresses: no repair, no refusal, no substitution.
//!   2. **The renderer.** The expressed tree becomes Futhark source.
//!   3. **Futhark accepts it.** The emitted program type-checks and compiles
//!      with the real Futhark compiler -- so the kingdom's typing and
//!      Futhark's agree.
//!
//! The genetic representation is the Karva gene throughout. The Futhark text
//! is a rendering of the expressed tree, regenerated every time; nothing
//! reads it back into the genome.

use std::collections::BTreeMap;
use std::process::Command;

use fuller::futhark_exec::compile_and_run;
use fuller::futhark_kingdom::{FutharkKingdom, GeneRng};
use fuller::geneframe::Ty;
use fuller::kingdom_contract::KingdomContract;
use fuller::typed_gep::{self, GeneLayout};

/// The Futhark program around an expression: the input array and the result
/// type.
///
/// The input is UNSIZED, so the same program shape serves every root, and the
/// generated C API is the same too (`struct futhark_f32_1d`).
fn program(scalar_root: bool, expr: &str) -> String {
    let result = if scalar_root { "f32" } else { "[]f32" };
    format!("-- a program of the Futhark-style kingdom\nentry main (xs: []f32) : {result} = {expr}\n")
}

/// The type of a root by name.
fn root_ty(k: &FutharkKingdom, name: &str) -> Ty {
    k.ty(name).unwrap_or_else(|| panic!("no root {name}"))
}

fn main() -> Result<(), String> {
    let mut root_name = "arr.f32".to_string();
    let mut size = 4usize;
    let mut count = 100usize;
    let mut compile = false;
    let mut check = false;
    let mut show = 3usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--root" => root_name = args.next().ok_or("--root needs a value")?,
            "--size" => size = args.next().ok_or("--size needs a value")?.parse().map_err(|_| "--size needs a number")?,
            "--count" => count = args.next().ok_or("--count needs a value")?.parse().map_err(|_| "--count needs a number")?,
            "--compile" => compile = true,
            "--check" => check = true,
            "--show" => show = args.next().ok_or("--show needs a value")?.parse().map_err(|_| "--show needs a number")?,
            other => return Err(format!("unknown option {other}")),
        }
    }

    let k = FutharkKingdom::load()?;
    let contract: KingdomContract = k.relations().check().map_err(|d| d.join("; "))?;
    let root = root_ty(&k, &root_name);
    let layout = GeneLayout::from_program_size(&contract, root, size).map_err(|e| format!("{e:?}"))?;
    let const_len: BTreeMap<Ty, u32> = k.const_len();

    println!(
        "kingdom Futhark: {} types, {} rows, root {root_name}, program size {size}, gene width {}",
        k.duals.len(),
        k.rows.len(),
        layout.width
    );

    let mut rng = GeneRng::new(0x5EED_1234_ABCD_0001);
    let mut expressed = 0usize;
    let mut rendered = 0usize;
    let mut compiled = 0usize;
    let mut checked = 0usize;
    let mut timed_out = 0usize;
    let mut compile_errors: Vec<String> = Vec::new();
    let mut shown = 0usize;

    for _ in 0..count {
        // A random gene: every cell a symbol from its own domain's alphabet.
        let mut gene = vec![0u32; layout.width];
        for (pos, cell) in gene.iter_mut().enumerate() {
            let alphabet = layout.alphabet(&contract, pos, &const_len);
            if alphabet.is_empty() {
                return Err(format!("cell {pos} has an empty alphabet"));
            }
            *cell = alphabet[rng.below(alphabet.len())];
        }
        // Validate and express. A defect here is a bug in the kingdom, not in
        // the gene: the design promises every gene is legal.
        let e = match typed_gep::express(&contract, &layout, &gene, &const_len) {
            Ok(e) => e,
            Err(defect) => {
                return Err(format!("a random gene was refused: {defect:?}\ngene: {gene:?}"));
            }
        };
        expressed += 1;
        let tree = e.tree();
        let expr = k.render(&tree)?;
        rendered += 1;
        let scalar = root_name == "f32.real";
        let source = program(scalar, &expr);

        if shown < show {
            shown += 1;
            println!("\n--- gene {shown} ---\n{}", source.trim_end());
        }

        if check {
            // What the gene MEANS, from the interpreter, against what the
            // COMPILED Futhark program produces on the same input.
            let input: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.25).collect();
            let expected = k.eval(&tree, &input)?;
            // A PRIVATE directory per process: the compiled program, its C
            // driver and the executable are all named after the program, so
            // two runs sharing a directory race on the same files.
            let dir = std::env::temp_dir()
                .join("futhark_true_emit")
                .join(format!("check{}", std::process::id()));
            let got = match compile_and_run(&source, scalar, &input, &dir) {
                Ok(got) => got,
                Err(e) => {
                    // A program the compiler does not converge on is a data
                    // point, not a reason to abandon the run: the gene is
                    // legal, the compiler is the thing that gives up.
                    if e.contains("no answer within") {
                        timed_out += 1;
                        println!("\n--- the compiler did not converge ---\n{}", source.trim_end());
                        continue;
                    }
                    return Err(e);
                }
            };
            let (want, label): (Vec<f32>, &str) = match &expected {
                fuller::futhark_interp::Value::Arr(a) => (a.clone(), "array"),
                fuller::futhark_interp::Value::Real(r) => (vec![*r], "real"),
                fuller::futhark_interp::Value::Fun(_) => {
                    return Err("the root of a program is not a function".to_string())
                }
            };
            if got.len() != want.len() {
                return Err(format!("the program produced {} values, the interpreter {}", got.len(), want.len()));
            }
            for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                let same = if a == b {
                    // Equal, including the two infinities: `inf - inf` is
                    // NaN, so the tolerance test below cannot see it.
                    true
                } else if a.is_nan() && b.is_nan() {
                    true
                } else {
                    let scale = a.abs().max(b.abs()).max(1.0);
                    (a - b).abs() <= 1e-4 * scale
                };
                if !same {
                    return Err(format!(
                        "value {i} of the {label}: futhark says {a}, the interpreter says {b}\nprogram: {}",
                        source.lines().last().unwrap_or("")
                    ));
                }
            }
            checked += 1;
        }

        if compile {
            let dir = std::env::temp_dir().join("futhark_true_emit");
            std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
            let path = dir.join(format!("p{shown}_{}.fut", rng.next_u64()));
            std::fs::write(&path, &source).map_err(|e| format!("write: {e}"))?;
            let out = Command::new("futhark")
                .args(["c", "--library", path.to_str().unwrap(), "-o", "/tmp/futhark_true_emit.out"])
                .output();
            match out {
                Ok(o) if o.status.success() => compiled += 1,
                Ok(o) => {
                    let first = String::from_utf8_lossy(&o.stderr).lines().next().unwrap_or("futhark failed").to_string();
                    compile_errors.push(format!("{first}\n      in: {}", source.lines().last().unwrap_or("")));
                }
                Err(e) => compile_errors.push(format!("could not run futhark: {e}")),
            }
            let _ = std::fs::remove_file(&path);
        }
    }

    println!("\n{expressed} random genes expressed, {rendered} rendered");
    if check {
        println!("{checked} of {count} agree with the interpreter, element for element");
        if timed_out > 0 {
            println!("{timed_out} of {count} did not converge: the compiler was killed");
        }
    }
    if compile {
        println!("{compiled} of {count} compiled with futhark c");
        for e in compile_errors.iter().take(5) {
            println!("  compile error: {e}");
        }
    }
    Ok(())
}
