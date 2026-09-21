//! A whole symbolic-regression fit in Rust, no Python:
//!
//!   cargo run --release --features gpu --example evolve_fit -- data.tsv [seed] [seconds] [max_rows] [population] [all|split] [cleanse_rate]
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
use fuller::evolve::engine::{evaluate_math, Config, Data, Engine};
use fuller::evolve::{below, draw};
use fuller::lint::node::Tree;
use fuller::lint::{lint, Options};
use fuller::lint::tables::Tables;

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
    config.max_seconds = seconds;
    if let Some(population) = args.get(4).and_then(|a| a.parse::<u32>().ok()) {
        config.pop_intake = population / 4 * 3;
        config.pop_champion = population - config.pop_intake;
    }
    config.cleanse = args.get(6).and_then(|a| a.parse().ok()).unwrap_or(0.0);
    let mut engine = Engine::new(config, Data { names: names.clone(), x, y, splits }).expect("engine");
    let out = engine.fit().expect("fit");

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

    let tables = Tables::standard().expect("fuller's rule tables");
    let tidy = lint(&tables, &out.math, &names, &Options::default()).map(|o| o.best).unwrap_or_else(|_| Tree::parse(&out.math).expect("math"));
    println!("model: {}", tidy.to_infix());
    println!("GENERATIONS\t{}\t{}\t{:.3e}", out.generations, out.stopped_by, out.best.one_minus_r2[1]);
    println!("MODEL_INFIX\t{}", tidy.to_infix());
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
