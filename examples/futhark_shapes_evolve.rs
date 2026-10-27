//! The shape kingdom's evolution loop, and what it wins.
//!
//! The kernel is the one the kingdom's vocabulary can make *expensive*:
//! the product of two matrices, computed **twice** — once directly and once
//! the transposed way, then added. The two products are the same matrix, but
//! the compiler cannot see that: they are different expressions, so its CSE
//! leaves both, and the GPU program allocates twice as much.
//!
//! The search's job is to find a program that computes the same value for
//! less. The e-graph's **transposed-product** rule — `(Y' . X')' = X . Y` —
//! is what does it: it rewrites the second operand into the first, and the
//! compiler's own CSE then does the rest.
//!
//! The cost that decides is **measured on the GPU**, from the compiled GPU
//! program: the device memory it allocates, the shared memory the tiler
//! asks for, and the number of kernels.
//!
//! ```text
//! cargo run --release --features wgsl --example futhark_shapes_evolve
//! ```

use std::collections::BTreeMap;
use std::time::Duration;

use fuller::futhark_exec::{run_text_program_on, Backend};
use fuller::futhark_kingdom::FutharkKingdom;
use fuller::futhark_shapes::program;
use fuller::futhark_shapes_cost::{program_cost, score_at, CostTable};
use fuller::futhark_shapes_evolve::{evolve, Config};
use fuller::futhark_shapes_gpu::measure;
use fuller::futhark_shapes_interp::{format_matrix, parse_matrix, Mat, MValue};
use fuller::geneframe::Ty;
use fuller::typed_gep::{fits, GeneLayout, Tree};

/// A tree of one row over the given trees.
fn tree_with(k: &FutharkKingdom, name: &str, kids: Vec<Tree>) -> Tree {
    let symbol = k
        .rows
        .iter()
        .position(|r| r.name == name)
        .unwrap_or_else(|| panic!("the kingdom has no row {name}")) as u32;
    Tree { symbol, const_index: None, kids }
}

/// A leaf: a row with no children.
fn leaf(k: &FutharkKingdom, name: &str) -> Tree {
    tree_with(k, name, Vec::new())
}

/// The reference kernel: the product of two matrices, computed twice — once
/// directly and once the transposed way — and the two added.
///
/// ```text
/// map2 (add) (A . B) (transpose ((B' . A')))
/// ```
///
/// The second product is the same matrix as the first, but the compiler
/// cannot see that: the two are different expressions, so its CSE leaves
/// both, and the GPU program does the work twice.
fn reference(k: &FutharkKingdom) -> Tree {
    let a = leaf(k, "mat.n.m.input");
    let b = leaf(k, "mat.m.p.input");
    let direct = tree_with(
        k,
        "matmul@mat.n.mxmat.m.p",
        vec![a.clone(), b.clone()],
    );
    let transposed = tree_with(
        k,
        "transpose@mat.p.n",
        vec![tree_with(
            k,
            "matmul@mat.p.mxmat.m.n",
            vec![
                tree_with(k, "transpose@mat.m.p", vec![b.clone()]),
                tree_with(k, "transpose@mat.n.m", vec![a.clone()]),
            ],
        )],
    );
    tree_with(
        k,
        "map2_rows@mat.n.p",
        vec![
            leaf(k, "fn.arr2_f32_arr_f32_arr_f32.add"),
            direct,
            transposed,
        ],
    )
}

/// The inputs: one matrix per class, with `-0.0` first in each — the sign of
/// a zero is where the compiled program and the source semantics part
/// company.
fn inputs(k: &FutharkKingdom, sizes: &BTreeMap<String, usize>) -> (BTreeMap<Ty, Mat>, String) {
    inputs_from(k, sizes, 0x1234_5678_9ABC_DEF0)
}

/// The inputs drawn from a seed: one matrix per class, with `-0.0` first in
/// each — the sign of a zero is where the compiled program and the source
/// semantics part company.
fn inputs_from(
    k: &FutharkKingdom,
    sizes: &BTreeMap<String, usize>,
    seed: u64,
) -> (BTreeMap<Ty, Mat>, String) {
    let mut map: BTreeMap<Ty, Mat> = BTreeMap::new();
    let mut text = String::new();
    let mut rng: u64 = seed;
    for d in &k.duals {
        if d.slot != "mat" {
            continue;
        }
        let dims: Vec<usize> = d.form.split('.').map(|r| *sizes.get(r).unwrap_or(&1)).collect();
        let mut data = Vec::with_capacity(dims[0] * dims[1]);
        for _ in 0..dims[0] * dims[1] {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            data.push(((rng >> 40) as f32) / 16777216.0 * 16.0 - 8.0);
        }
        if !data.is_empty() {
            data[0] = -0.0;
        }
        let mat = Mat::new(dims[0], dims[1], data).expect("the shape matches the values");
        text.push_str(&format_matrix(&mat));
        let ty = k.ty(&d.dual).expect("a dual's own type");
        map.insert(ty, mat);
    }
    (map, text)
}

/// The smallest program size whose layout holds the reference.
fn layout_for(k: &FutharkKingdom, reference: &Tree) -> Result<(GeneLayout, usize), String> {
    let contract = k.relations().check().map_err(|bad| bad.join("; "))?;
    let root = k.ty("mat.n.p").ok_or("the shape kingdom has no mat.n.p class")?;
    for size in 3..=12 {
        let layout = GeneLayout::from_program_size(&contract, root, size)
            .map_err(|e| format!("the layout at size {size}: {e}"))?;
        match fits(&contract, &layout, reference) {
            Ok(()) => return Ok((layout, size)),
            Err(e) => {
                if size <= 6 {
                    println!("  size {size}: {e}");
                }
            }
        }
    }
    Err("the reference does not fit any layout up to size 12".to_string())
}

fn main() {
    let mut generations = 8usize;
    let mut population = 32usize;
    let mut seed = 0x5EED_1234_ABCD_0010u64;
    let mut measure_gpu = true;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--generations" => {
                generations = args[i + 1].parse().expect("--generations takes a number");
                i += 2;
            }
            "--population" => {
                population = args[i + 1].parse().expect("--population takes a number");
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().expect("--seed takes a number");
                i += 2;
            }
            "--no-gpu" => {
                measure_gpu = false;
                i += 1;
            }
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
    }

    let k = FutharkKingdom::load_shapes().expect("the shape tables load");
    let root = k.ty("mat.n.p").expect("the root class");
    let table = CostTable::load(&k).expect("the cost table loads");
    let reference_tree = reference(&k);
    let (layout, size) = layout_for(&k, &reference_tree).expect("a layout holds the reference");
    let mut cfg = Config { generations, population, seed, measure_gpu, ..Config::default() };
    cfg.program_size = size;
    // A REALISTIC size: at 2x3x4 the fixed overhead of a GPU program (the
    // counters, the tiling parameters) swamps the traffic, and the win the
    // search finds is invisible. At 64x64x64 the traffic is what the cost
    // measures.
    cfg.sizes = [("m".to_string(), 64usize), ("n".to_string(), 64), ("p".to_string(), 64)]
        .into_iter()
        .collect();

    println!("== the kernel ==");
    println!("reference     the product of two matrices, computed TWICE: once directly");
    println!("              and once the transposed way, and the two added.");
    println!("program size  {size}   gene width {}", layout.width);
    println!("population    {population}   generations {generations}   seed {seed:#x}");

    let (input_matrices, input_text) = inputs(&k, &cfg.sizes);
    // The suite the correctness is decided on: the same shapes, drawn
    // again, so a program that coincides with the reference on one sample
    // does not get in.
    let mut suite = vec![input_matrices.clone()];
    for i in 1..cfg.samples {
        suite.push(inputs_from(&k, &cfg.sizes, i as u64).0);
    }
    let reference_source = program(&k, root, &reference_tree, &cfg.sizes).expect("the program renders");
    println!("\n-- the reference's source --\n{}", reference_source.trim());

    // The reference's own costs.
    let reference_cost = program_cost(&reference_tree, &k, &table).expect("the cost is computed");
    let reference_score = score_at(&reference_cost, &cfg.sizes).expect("the score is computed");
    println!("\n== the reference ==");
    println!("cheap cost    traffic {}   alloc {}   depth {}", reference_score.traffic, reference_score.alloc, reference_score.depth);

    let dir = std::env::temp_dir().join("futhark_shapes_evolve_example");
    if measure_gpu {
        let measured = measure(&reference_source, &dir).expect("the GPU cost is measured");
        println!(
            "GPU cost      {} device allocations ({} of a computed size), {} bytes literal, {} shared, {} kernels   tiled: {}",
            measured.cost.device_allocs,
            measured.cost.computed_allocs,
            measured.cost.device_bytes,
            measured.cost.shared_allocs,
            measured.cost.kernels,
            measured.is_tiled()
        );
    }

    // The loop.
    println!("\n== the search ==");
    let report = evolve(&reference_tree, &suite, &cfg).expect("the loop runs");
    println!(
        "{:>4}  {:>7}  {:>10}  {:>8}  {:>9}  {:>10}  {:>9}  {:>7}  {:>9}",
        "gen", "correct", "traffic", "alloc", "polished", "improved", "measured", "tiled", "no-gpu"
    );
    for generation in &report.generations {
        let best = generation.best.map(|s| (s.traffic, s.alloc)).unwrap_or((0, 0));
        println!(
            "{:>4}  {:>7}  {:>10}  {:>8}  {:>9}  {:>10}  {:>9}  {:>7}  {:>9}",
            generation.number,
            generation.correct,
            best.0,
            best.1,
            generation.polished,
            generation.improved,
            generation.measured,
            generation.tiled,
            generation.not_runnable
        );
    }

    let Some(best) = report.best.clone() else {
        println!("\nno correct program was found");
        return;
    };
    println!("\n== the best ==");
    println!("cheap cost    traffic {}   alloc {}   depth {}", best.score.traffic, best.score.alloc, best.score.depth);
    let best_source = program(&k, root, &best.tree, &cfg.sizes).expect("the program renders");
    println!("-- the best's source --\n{}", best_source.trim());

    if measure_gpu {
        let measured = measure(&best_source, &dir).expect("the GPU cost is measured");
        println!(
            "GPU cost      {} device allocations ({} of a computed size), {} bytes literal, {} shared, {} kernels   tiled: {}",
            measured.cost.device_allocs,
            measured.cost.computed_allocs,
            measured.cost.device_bytes,
            measured.cost.shared_allocs,
            measured.cost.kernels,
            measured.is_tiled()
        );
        if let Some(reference_gpu) = report.reference_gpu {
            let before = reference_gpu.cost.device_allocs;
            let after = measured.cost.device_allocs;
            println!(
                "the win       {} device allocations -> {} ({:+.1}%)",
                before,
                after,
                100.0 * (after as f64 - before as f64) / before.max(1) as f64
            );
        }
    }

    // And the winner, run on the GPU for real.
    println!("\n== the winner, on the GPU ==");
    match run_text_program_on(Backend::OpenCl, &best_source, &input_text, &dir, Duration::from_secs(180)) {
        Ok(out) => {
            let got = parse_matrix(&out).expect("the output is a matrix");
            let want = match k.eval_matrices(&best.tree, &input_matrices).expect("the interpreter evaluates") {
                MValue::Mat(m) => m,
                other => panic!("the value is not a matrix: {other:?}"),
            };
            let exact = got.data.iter().zip(want.data.iter()).filter(|(a, b)| a == b).count();
            let max_diff = got
                .data
                .iter()
                .zip(want.data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("the GPU ran the program: {} elements, {} agree exactly", got.data.len(), exact);
            if max_diff > 0.0 {
                println!(
                    "the largest difference from the interpreter: {max_diff} (the GPU contracts the multiply-add, the interpreter does not)"
                );
            } else {
                println!("the GPU's values are the interpreter's, bit for bit");
            }
        }
        Err(e) => println!("the GPU run failed: {e}"),
    }

    // The determinism check: the same seed, the same run.
    let again = evolve(&reference_tree, &suite, &cfg).expect("the loop runs again");
    let same = again
        .best
        .as_ref()
        .zip(report.best.as_ref())
        .map(|(a, b)| a.gene == b.gene)
        .unwrap_or(false);
    println!("\n== determinism ==\nre-running with the same seed gives the same best: {same}");

}
