//! The stored cases of a Futhark signature problem (`fuller::futhark_sig`): random inputs at the declared sizes and
//! the published program's exact outputs on them, computed by the problem's yardstick (the published kernel) through
//! `futhark c`, written in Futhark's binary format under the problem's corpus directory with `cases.tsv`.
//!
//!   cargo run --release --example futhark_problem_cases -- --problem pathfinder
//!
//! Three splits (docs/PLAN_futhark_fitness_fixes.md, sections C and G): `train` and `held` cases are the fitness
//! (the held-out cases its validation column); `test` cases are never seen by selection and are where an exactness
//! claim is made.
//!
//! Pathfinder: walls are uniform integers 0 .. 9 (the published benchmark's walls are its Sobol numbers modulo 10).
//! Training sizes 64x64 (two seeds), 256x256, 16x100, 32x32, 48x80, 100x40, 8x8 and the stepping stones 2x50 and 3x40
//! (one and two steps of the loop, which a partly right program can solve exactly); held-out sizes 128x512, 512x128,
//! 5x200, 1x37 and 37x1 (rows x cols), the last two the edges of the signature (one row: the output is row 0; one
//! column: every neighbour clamps); test sizes 200x200, 64x300, 2x7 and 300x3.
//!
//! The micro benchmarks (`--problem reduce|scan|histogram`): values uniform over the whole `i32` range, as
//! futhark-benchmarks' `random input { [n]i32 }` (so sums wrap, as Futhark's `i32` does). Reduce and scan: training
//! sizes n = 64 (two seeds) and 1000, held out n = 777 and the edge n = 1. Histogram (`k`, `is`, `vs`): the study's
//! cases (whole-range indices and values: (k, n) = (10, 500) twice and (37, 1000) training, (100, 777) and the edge
//! (1, 13) held out), and STEPPING STONES a partly right program can solve exactly (fault 4 of the review): indices
//! already in `[0, k)`, k = 1, 2 and 3, values all 1 (a count), small values (no wrap, so an error's size means
//! something). Every size is at most 1024, so a gene's work (`max(D) ** 2` a node at the family's depth) stays under
//! the work guard. The study's cases are drawn first and in the same order, so they are byte-identical to before.

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
    let cases: [(&str, &str, usize, usize, u64); 19] = [
        ("train_64x64_s1", "train", 64, 64, 1),
        ("train_64x64_s2", "train", 64, 64, 2),
        ("train_256x256_s3", "train", 256, 256, 3),
        ("train_16x100_s4", "train", 16, 100, 4),
        ("train_32x32_s9", "train", 32, 32, 9),
        ("train_48x80_s10", "train", 48, 80, 10),
        ("train_100x40_s11", "train", 100, 40, 11),
        ("train_2x50_s12", "train", 2, 50, 12),
        ("train_3x40_s13", "train", 3, 40, 13),
        ("train_8x8_s14", "train", 8, 8, 14),
        ("held_128x512_s5", "held", 128, 512, 5),
        ("held_512x128_s6", "held", 512, 128, 6),
        ("held_5x200_s15", "held", 5, 200, 15),
        ("held_1x37_s7", "held", 1, 37, 7),
        ("held_37x1_s8", "held", 37, 1, 8),
        ("test_200x200_s16", "test", 200, 200, 16),
        ("test_64x300_s17", "test", 64, 300, 17),
        ("test_2x7_s18", "test", 2, 7, 18),
        ("test_300x3_s19", "test", 300, 3, 19),
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

/// How a histogram case's indices are drawn: over the whole `i32` range, or already in `[0, k)`.
#[derive(Clone, Copy)]
enum Ix {
    Any,
    InRange,
}

/// How a histogram case's values are drawn: over the whole `i32` range, uniform in `0 ..= hi`, or all 1 (a count).
#[derive(Clone, Copy)]
enum Vals {
    Any,
    Upto(u64),
    One,
}

/// The micro benchmarks' cases: (name, split, k, n, seed, indices, values); `k` is the histogram's bin count (unused
/// otherwise; reduce and scan draw their values over the whole range).
fn micro(name: &str, dir: &std::path::Path, exe: &std::path::Path) -> Result<(), String> {
    let cases: Vec<(&str, &str, usize, usize, u64, Ix, Vals)> = match name {
        "reduce" | "scan" => vec![
            ("train_n64_s1", "train", 0, 64, 1, Ix::Any, Vals::Any),
            ("train_n64_s2", "train", 0, 64, 2, Ix::Any, Vals::Any),
            ("train_n1000_s3", "train", 0, 1000, 3, Ix::Any, Vals::Any),
            ("held_n777_s4", "held", 0, 777, 4, Ix::Any, Vals::Any),
            ("held_n1_s5", "held", 0, 1, 5, Ix::Any, Vals::Any),
        ],
        "histogram" => vec![
            ("train_k10_n500_s1", "train", 10, 500, 1, Ix::Any, Vals::Any),
            ("train_k10_n500_s2", "train", 10, 500, 2, Ix::Any, Vals::Any),
            ("train_k37_n1000_s3", "train", 37, 1000, 3, Ix::Any, Vals::Any),
            ("train_inrange_k10_n500_s6", "train", 10, 500, 6, Ix::InRange, Vals::Upto(9)),
            ("train_inrange_count_k16_n300_s7", "train", 16, 300, 7, Ix::InRange, Vals::One),
            ("train_k1_n50_s8", "train", 1, 50, 8, Ix::Any, Vals::Upto(9)),
            ("train_k2_n200_s9", "train", 2, 200, 9, Ix::Any, Vals::Upto(9)),
            ("train_k3_n300_s10", "train", 3, 300, 10, Ix::Any, Vals::Upto(9)),
            ("train_count_k10_n500_s11", "train", 10, 500, 11, Ix::Any, Vals::One),
            ("train_small_k10_n500_s12", "train", 10, 500, 12, Ix::Any, Vals::Upto(9)),
            ("train_small_k24_n700_s13", "train", 24, 700, 13, Ix::Any, Vals::Upto(99)),
            ("held_k100_n777_s4", "held", 100, 777, 4, Ix::Any, Vals::Any),
            ("held_k1_n13_s5", "held", 1, 13, 5, Ix::Any, Vals::Any),
            ("held_small_k7_n300_s14", "held", 7, 300, 14, Ix::Any, Vals::Upto(9)),
            ("held_inrange_k30_n400_s15", "held", 30, 400, 15, Ix::InRange, Vals::Any),
            ("test_k50_n1000_s16", "test", 50, 1000, 16, Ix::Any, Vals::Any),
            ("test_small_k13_n100_s17", "test", 13, 100, 17, Ix::Any, Vals::Upto(9)),
            ("test_k3_n31_s18", "test", 3, 31, 18, Ix::Any, Vals::Any),
            ("test_count_k5_n64_s19", "test", 5, 64, 19, Ix::Any, Vals::One),
        ],
        other => return Err(format!("no case recipe for {other}")),
    };
    let mut index = format!("# name\tsplit\tk\tn\tseed\tindices\tvalues ({name}: i32 drawn as the last two columns say, `any` over the whole range; outputs by yardstick.fut through futhark c)\n");
    for (case, split, k, n, seed, ix, vals) in cases {
        let mut rng = GeneRng::new(0x51C_3E7A ^ seed);
        let xs = Array { dims: vec![n], data: (0..n).map(|_| match ix {
            Ix::Any => any_i32(&mut rng),
            Ix::InRange => rng.below(k) as i32,
        }).collect() };
        let stdin = if name == "histogram" {
            let vs = Array { dims: vec![n], data: (0..n).map(|_| match vals {
                Vals::Any => any_i32(&mut rng),
                Vals::Upto(hi) => rng.below(hi as usize + 1) as i32,
                Vals::One => 1,
            }).collect() };
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
        let ixs = match ix {
            Ix::Any => "any".to_string(),
            Ix::InRange => "0..k-1".to_string(),
        };
        let vls = match vals {
            Vals::Any => "any".to_string(),
            Vals::Upto(hi) => format!("0..{hi}"),
            Vals::One => "1".to_string(),
        };
        index.push_str(&format!("{case}\t{split}\t{k}\t{n}\t{seed}\t{ixs}\t{vls}\n"));
        println!("{case}: k {k}, n {n}, output {} values, min {} max {}", out.data.len(), out.data.iter().min().unwrap_or(&0), out.data.iter().max().unwrap_or(&0));
    }
    std::fs::write(dir.join("cases.tsv"), index).map_err(|e| e.to_string())?;
    Ok(())
}
