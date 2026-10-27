//! The stored cases of a Futhark signature problem (`fuller::futhark_sig`): random inputs at the declared sizes and
//! the published program's exact outputs on them, computed by the problem's yardstick (the published kernel) through
//! `futhark c`, written in Futhark's binary format under the problem's corpus directory with `cases.tsv`.
//!
//!   cargo run --release --example futhark_problem_cases -- --problem pathfinder
//!
//! Pathfinder: walls are uniform integers 0 .. 9 (the published benchmark's walls are its Sobol numbers modulo 10).
//! Training sizes 64x64 (two seeds), 256x256 and 16x100; held-out sizes 128x512, 512x128, 1x37 and 37x1 (rows x
//! cols), the last two the edges of the signature (one row: the output is row 0; one column: every neighbour clamps).
//!
//! The micro benchmarks (`--problem reduce|scan|histogram`): values uniform over the whole `i32` range, as
//! futhark-benchmarks' `random input { [n]i32 }` (so sums wrap, as Futhark's `i32` does). Reduce and scan: training
//! sizes n = 64 (two seeds) and 1000, held out n = 777 and the edge n = 1. Histogram (`k`, `is`, `vs`): training
//! (k, n) = (10, 500) (two seeds) and (37, 1000), held out (100, 777) and the edge (1, 13). Every size is at most 1024,
//! so a gene's work (`max(D) ** 2` a node at the family's depth) stays under the work guard.

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
    let dir = corpus_dir(&p);
    let build = std::env::temp_dir().join(format!("futhark_problem_cases_{}", std::process::id()));
    let exe = compile_c(p.yardstick, &build)?;
    if p.name != "pathfinder" {
        return micro(&name, &dir, &exe);
    }
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

/// A value uniform over the whole `i32` range.
fn any_i32(rng: &mut GeneRng) -> i32 {
    rng.next_u64() as u32 as i32
}

/// The micro benchmarks' cases: (name, split, k, n, seed); `k` is the histogram's bin count (unused otherwise).
fn micro(name: &str, dir: &std::path::Path, exe: &std::path::Path) -> Result<(), String> {
    let cases: Vec<(&str, &str, usize, usize, u64)> = match name {
        "reduce" | "scan" => vec![
            ("train_n64_s1", "train", 0, 64, 1),
            ("train_n64_s2", "train", 0, 64, 2),
            ("train_n1000_s3", "train", 0, 1000, 3),
            ("held_n777_s4", "held", 0, 777, 4),
            ("held_n1_s5", "held", 0, 1, 5),
        ],
        "histogram" => vec![
            ("train_k10_n500_s1", "train", 10, 500, 1),
            ("train_k10_n500_s2", "train", 10, 500, 2),
            ("train_k37_n1000_s3", "train", 37, 1000, 3),
            ("held_k100_n777_s4", "held", 100, 777, 4),
            ("held_k1_n13_s5", "held", 1, 13, 5),
        ],
        other => return Err(format!("no case recipe for {other}")),
    };
    let mut index = format!("# name\tsplit\tk\tn\tseed ({name}: i32 values uniform over the whole range; outputs by yardstick.fut through futhark c)\n");
    for (case, split, k, n, seed) in cases {
        let mut rng = GeneRng::new(0x51C_3E7A ^ seed);
        let xs = Array { dims: vec![n], data: (0..n).map(|_| any_i32(&mut rng)).collect() };
        let stdin = if name == "histogram" {
            let vs = Array { dims: vec![n], data: (0..n).map(|_| any_i32(&mut rng)).collect() };
            write_binary(&[&Array { dims: vec![], data: vec![k as i32] }, &xs, &vs])
        } else {
            write_binary(&[&xs])
        };
        let out = run_exe(exe, &stdin)?;
        let want: Vec<usize> = match name {
            "reduce" => vec![],
            "scan" => vec![n],
            _ => vec![k],
        };
        if out.dims != want {
            return Err(format!("{case}: the yardstick gave {:?}, the signature says {want:?}", out.dims));
        }
        std::fs::write(dir.join(format!("{case}.in")), &stdin).map_err(|e| e.to_string())?;
        std::fs::write(dir.join(format!("{case}.out")), write_binary(&[&out])).map_err(|e| e.to_string())?;
        index.push_str(&format!("{case}\t{split}\t{k}\t{n}\t{seed}\n"));
        println!("{case}: k {k}, n {n}, output {} values, min {} max {}", out.data.len(), out.data.iter().min().unwrap_or(&0), out.data.iter().max().unwrap_or(&0));
    }
    std::fs::write(dir.join("cases.tsv"), index).map_err(|e| e.to_string())?;
    Ok(())
}
