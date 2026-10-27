//! The stored cases of a Futhark signature problem (`fuller::futhark_sig`): random inputs at the declared sizes and
//! the published program's exact outputs on them, computed by the problem's yardstick (the published kernel) through
//! `futhark c`, written in Futhark's binary format under the problem's corpus directory with `cases.tsv`.
//!
//!   cargo run --release --example futhark_problem_cases -- --problem pathfinder
//!
//! Pathfinder: walls are uniform integers 0 .. 9 (the published benchmark's walls are its Sobol numbers modulo 10).
//! Training sizes 64x64 (two seeds), 256x256 and 16x100; held-out sizes 128x512, 512x128, 1x37 and 37x1 (rows x
//! cols), the last two the edges of the signature (one row: the output is row 0; one column: every neighbour clamps).

use fuller::futhark_kingdom::GeneRng;
use fuller::futhark_sig::{compile_c, corpus_dir, problem, run_exe, write_binary, Array};

fn main() {
    if let Err(e) = real_main() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let name = args.iter().position(|a| a == "--problem").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "pathfinder".into());
    let p = problem(&name)?;
    if p.name != "pathfinder" {
        return Err(format!("no case recipe for {name}"));
    }
    let dir = corpus_dir(&p);
    let build = std::env::temp_dir().join(format!("futhark_problem_cases_{}", std::process::id()));
    let exe = compile_c(p.yardstick, &build)?;
    let cases: [(&str, &str, usize, usize, u64); 8] = [
        ("train_64x64_s1", "train", 64, 64, 1),
        ("train_64x64_s2", "train", 64, 64, 2),
        ("train_256x256_s3", "train", 256, 256, 3),
        ("train_16x100_s4", "train", 16, 100, 4),
        ("held_128x512_s5", "held", 128, 512, 5),
        ("held_512x128_s6", "held", 512, 128, 6),
        ("held_1x37_s7", "held", 1, 37, 7),
        ("held_37x1_s8", "held", 37, 1, 8),
    ];
    let mut index = String::from("# name\tsplit\trows\tcols\tseed (walls uniform 0..9; outputs by yardstick.fut through futhark c)\n");
    for (case, split, rows, cols, seed) in cases {
        let mut rng = GeneRng::new(0x9A7F_1DE5 ^ seed);
        let walls = Array { dims: vec![rows, cols], data: (0..rows * cols).map(|_| rng.below(10) as i32).collect() };
        let stdin = write_binary(&[&walls]);
        let out = run_exe(&exe, &stdin)?;
        if out.dims != vec![cols] {
            return Err(format!("{case}: the yardstick gave {:?}", out.dims));
        }
        std::fs::write(dir.join(format!("{case}.in")), &stdin).map_err(|e| e.to_string())?;
        std::fs::write(dir.join(format!("{case}.out")), write_binary(&[&out])).map_err(|e| e.to_string())?;
        index.push_str(&format!("{case}\t{split}\t{rows}\t{cols}\t{seed}\n"));
        println!("{case}: {rows}x{cols}, output min {} max {}", out.data.iter().min().unwrap_or(&0), out.data.iter().max().unwrap_or(&0));
    }
    std::fs::write(dir.join("cases.tsv"), index).map_err(|e| e.to_string())?;
    Ok(())
}
