//! A whole symbolic-regression fit in Rust, no Python:
//!
//!   cargo run --release --features gpu --example evolve_fit -- data.tsv [seed] [seconds] [max_rows] [population] [all|split] [cleanse_rate] [test.tsv] [harvests]
//!
//! `data.tsv`: tab-separated, a header, the target in the column named
//! `target` (a PMLB dataset, gunzipped). SRBench's 75/25 split is mimicked with
//! our own shuffle; from the 75%, `max_rows` random rows are split 80/20 into
//! train and validation. `population` is split 3:1 into the intake and champion
//! islands (default 800). A sixth argument `all` means the file is ALREADY the
//! training part (a harness made the split): every row is used, none held back.
//! Prints what the search did, where the time went, the
//! model and its R² on the unseen 25%.

use fuller::chrom_score::Splits;
use fuller::evolve::engine::{evaluate_math, final_form, resolve_protected, Config, Data, Engine};
use fuller::evolve::{below, draw};
use fuller::lint::node::Tree;

fn shuffled(n: usize, seed: u32, stream: u32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, below(draw(seed, 0, i as u32, 0, stream), i as u32 + 1) as usize);
    }
    order
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: evolve_fit data.tsv [seed] [seconds] [max_rows]");
    let seed: u32 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(7001);
    let seconds: f64 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(30.0);
    let max_rows: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(5000);

    let text = std::fs::read_to_string(path).expect("read the data file");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("a header line").split('\t').collect();
    let target = header.iter().position(|h| *h == "target").expect("a column named target");
    let rows: Vec<Vec<f64>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').map(|v| v.trim().parse::<f64>().expect("a number")).collect())
        .collect();
    let names: Vec<String> = (0..header.len() - 1).map(|i| format!("x_{i}")).collect();
    let inputs = |r: &Vec<f64>| -> Vec<f64> { r.iter().enumerate().filter(|(i, _)| *i != target).map(|(_, v)| *v).collect() };

    let order = shuffled(rows.len(), seed, 200);
    let use_all = args.get(5).is_some_and(|a| a == "all");
    let n_fit = if use_all { rows.len() } else { rows.len() * 3 / 4 };
    let (fit_rows, test_rows) = order.split_at(n_fit);
    let used: Vec<usize> = shuffled(fit_rows.len(), seed, 201).into_iter().take(max_rows).map(|i| fit_rows[i]).collect();
    let n_train = used.len() * 4 / 5;
    let splits = Splits { n_train, n_val: used.len() - n_train, n_extrap: 0 };
    let x: Vec<f32> = used.iter().flat_map(|&r| inputs(&rows[r])).map(|v| v as f32).collect();
    let y: Vec<f64> = used.iter().map(|&r| rows[r][target]).collect();

    let mut config = Config::srbench(seed);
    if let Some(population) = args.get(4).and_then(|a| a.parse::<u32>().ok()) {
        config.pop_intake = population / 4 * 3;
        config.pop_champion = population - config.pop_intake;
    }
    // Experiment knobs come by environment so the positional arguments stay put:
    //   EVOLVE_RNC_LO / EVOLVE_RNC_HI   the range a gene's random constants are drawn from
    //   EVOLVE_RESTARTS                 split the time into this many independent searches
    let env = |k: &str| std::env::var(k).ok();
    if let (Some(lo), Some(hi)) = (env("EVOLVE_RNC_LO").and_then(|v| v.parse().ok()), env("EVOLVE_RNC_HI").and_then(|v| v.parse().ok())) {
        config.rnc_lo = lo;
        config.rnc_hi = hi;
    }
    let restarts: u32 = env("EVOLVE_RESTARTS").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    config.cleanse = args.get(6).and_then(|a| a.parse().ok()).unwrap_or(0.0);
    config.harvests = args.get(8).and_then(|a| a.parse().ok()).unwrap_or(0);
    // RESTARTS: the same seconds as one search, spent as several independent
    // ones (seeds derived from the fit's seed). The first that meets the stop bar
    // ends the fit; otherwise the one with the best validation error is reported.
    // Whether a problem is solved in 6 s depends heavily on the draw; this buys
    // more draws for the same time.
    config.max_seconds = seconds / f64::from(restarts);
    let mut kept: Option<(Engine, fuller::evolve::engine::FitResult)> = None;
    let (mut generations, mut individuals) = (0u32, 0u64);
    for k in 0..restarts {
        let mut c = config.clone();
        c.seed = seed.wrapping_add(k.wrapping_mul(1_000_003));
        let mut engine = Engine::new(c, Data { names: names.clone(), x: x.clone(), y: y.clone(), splits }).expect("engine");
        let out = engine.fit().expect("fit");
        generations += out.generations;
        individuals += out.individuals;
        let done = out.stopped_by != "time" && out.stopped_by != "n_gen";
        if kept.as_ref().is_none_or(|(_, best)| out.best.one_minus_r2[1] < best.best.one_minus_r2[1]) {
            kept = Some((engine, out));
        }
        if done {
            break;
        }
    }
    let (engine, mut out) = kept.expect("at least one search ran");
    out.generations = generations;
    out.individuals = individuals;

    let t = &out.timing;
    println!("dataset {path}\nrows: train {n_train}, validation {}, unseen test {}", splits.n_val, test_rows.len());
    println!(
        "population {}+{} | {} generations in {:.1} s = {:.1} ms per generation | {} individuals = {:.0} per second | stopped by {}",
        engine.islands[0].hi, engine.islands[1].hi - engine.islands[1].lo, out.generations, out.seconds,
        1e3 * out.seconds / f64::from(out.generations.max(1)), out.individuals, out.individuals as f64 / out.seconds, out.stopped_by
    );
    println!(
        "seconds: variation {:.2} | read-back {:.2} | decode {:.2} | evaluate {:.2} | link+scale+metrics {:.2} | HFF {:.2} | pump {:.2}",
        t.vary, t.read, t.decode, t.evaluate, t.score, t.hff, t.pump
    );
    println!("genes evaluated {} (unique per generation), over the 64-node limit {}", out.unique_genes, out.oversized_genes);
    println!("1 - R²: train {:.3e}, validation {:.3e}", out.best.one_minus_r2[0], out.best.one_minus_r2[1]);

    // fuller's final form, the data as judge: the fit rows (train + validation)
    // decide which inputs are positive and which prunes change nothing.
    let fit_rows: Vec<Vec<(String, f64)>> = used.iter().map(|&r| names.iter().cloned().zip(inputs(&rows[r])).collect()).collect();
    // First say what the protected operators actually do on this data; then tidy.
    let resolved = resolve_protected(&out.math, &fit_rows).unwrap_or_else(|_| out.math.clone());
    let tidied = final_form(&resolved, &names, &fit_rows).unwrap_or(resolved);
    let tidy = Tree::parse(&tidied).expect("the final form parses");
    println!("model: {}", tidy.to_infix_faithful());
    println!("GENERATIONS\t{}\t{}\t{:.3e}\t{}", out.generations, out.stopped_by, out.best.one_minus_r2[1], out.harvested);
    println!("MODEL_INFIX\t{}", tidy.to_infix_faithful());
    println!("RAW_MATH\t{}", out.math);
    // A harness that made the split itself may hand over its test rows: the
    // RAW chromosome's R² on them (f64, fuller's evaluator), so the harness can
    // check that the string it reports computes the same thing.
    if let Some(test_path) = args.get(7) {
        let text = std::fs::read_to_string(test_path).expect("read the test file");
        let held: Vec<Vec<f64>> = text.lines().skip(1).filter(|l| !l.trim().is_empty())
            .map(|l| l.split('\t').map(|v| v.trim().parse::<f64>().expect("a number")).collect()).collect();
        let cols: Vec<Vec<(String, f64)>> = held.iter().map(|r| names.iter().cloned().zip(inputs(r)).collect()).collect();
        if let Ok(pred) = evaluate_math(&out.math, &cols) {
            let truth: Vec<f64> = held.iter().map(|r| r[target]).collect();
            let mean = truth.iter().sum::<f64>() / truth.len() as f64;
            let ss_res: f64 = pred.iter().zip(&truth).map(|(p, t)| (p - t).powi(2)).sum();
            let ss_tot: f64 = truth.iter().map(|t| (t - mean).powi(2)).sum();
            println!("CHROMOSOME_TEST_R2\t{:e}", 1.0 - ss_res / ss_tot);
        }
    }
    if test_rows.is_empty() {
        return;
    }

    // R² on the 25% the search never saw, by fuller's own f64 evaluator.
    let columns: Vec<Vec<(String, f64)>> = test_rows.iter().map(|&r| names.iter().cloned().zip(inputs(&rows[r])).collect()).collect();
    match evaluate_math(&out.math, &columns) {
        Ok(pred) => {
            let truth: Vec<f64> = test_rows.iter().map(|&r| rows[r][target]).collect();
            let mean = truth.iter().sum::<f64>() / truth.len() as f64;
            let ss_res: f64 = pred.iter().zip(&truth).map(|(p, t)| (p - t).powi(2)).sum();
            let ss_tot: f64 = truth.iter().map(|t| (t - mean).powi(2)).sum();
            println!("R² on the unseen test rows: {:.6}", 1.0 - ss_res / ss_tot);
        }
        Err(e) => println!("test R² not computed: {e}"),
    }
}
