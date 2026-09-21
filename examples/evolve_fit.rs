//! A whole symbolic-regression fit in Rust, no Python:
//!
//!   cargo run --release --features gpu --example evolve_fit -- data.tsv [seed] [seconds] [max_rows] [population] [all|split] [cleanse_rate] [test.tsv]
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
use fuller::evolve::umap2d::embed_2d;
use fuller::evolve::{below, draw, smogd, smote};
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
    let mut x: Vec<f32> = used.iter().flat_map(|&r| inputs(&rows[r])).map(|v| v as f32).collect();
    let mut y: Vec<f64> = used.iter().map(|&r| rows[r][target]).collect();
    // EDGE VALIDATION (EVOLVE_EDGE=file.tsv, same columns): rows the harness held
    // out as the most isolated of what it was given. They are never fitted on;
    // their errors are separate HFF objectives and the stop bar must hold on
    // them too. A law holds at the edges; an interior fit does not.
    let mut splits = splits;
    if let Some(edge_path) = std::env::var("EVOLVE_EDGE").ok().filter(|p| !p.is_empty()) {
        let text = std::fs::read_to_string(&edge_path).expect("read the edge file");
        for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let row: Vec<f64> = line.split('\t').map(|v| v.trim().parse::<f64>().expect("a number")).collect();
            x.extend(inputs(&row).into_iter().map(|v| v as f32));
            y.push(row[target]);
            splits.n_extrap += 1;
        }
    }

    // SMOGD (EVOLVE_SMOGD=1): the third block is GENERATED here, from the rows we
    // were given and nothing else — a 2D UMAP of the inputs, ManifoldGridSampler's
    // adaptive grid, and in every cell new rows drawn from their nearest real
    // rows. Noisy on purpose: they rank individuals in the tournaments (three HFF
    // objectives) and never decide that a fit is exact.
    let smogd_on = std::env::var("EVOLVE_SMOGD").is_ok_and(|v| v == "1");
    if smogd_on {
        let started = std::time::Instant::now();
        let real: Vec<f64> = used.iter().flat_map(|&r| inputs(&rows[r])).collect();
        let real_y: Vec<f64> = used.iter().map(|&r| rows[r][target]).collect();
        let embedding = embed_2d(&real, names.len(), u64::from(seed)).expect("the 2D embedding");
        // EVOLVE_SMOGD_NOISE: the multiplier on the neighbours' variance (1 = the original).
        let noise_multiplier = std::env::var("EVOLVE_SMOGD_NOISE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let params = smogd::Params { noise_multiplier, ..smogd::Params::defaults() };
        let (synth_x, synth_y) = smogd::rows(&real, &real_y, names.len(), &embedding, seed, &params);
        println!("SMOGD\t{}\trows from {} real rows in {:.1} s, noise x{noise_multiplier}", synth_y.len(), used.len(), started.elapsed().as_secs_f64());
        x.extend(synth_x.iter().map(|v| *v as f32));
        splits.n_extrap += synth_y.len();
        y.extend(synth_y);
    }
    // SMOTE (EVOLVE_SMOTE=1): rows on the segment between a real row and one of
    // its 5 nearest — no noise. They join SMOGD's in the third block, as many of
    // them as SMOGD placed (the block's error is row-weighted, so this is 50/50);
    // alone, one for every 20 real rows.
    let smote_on = std::env::var("EVOLVE_SMOTE").is_ok_and(|v| v == "1");
    if smote_on {
        let started = std::time::Instant::now();
        let real: Vec<f64> = used.iter().flat_map(|&r| inputs(&rows[r])).collect();
        let real_y: Vec<f64> = used.iter().map(|&r| rows[r][target]).collect();
        let count = if splits.n_extrap > 0 { splits.n_extrap } else { used.len() / 20 };
        let (synth_x, synth_y) = smote::rows(&real, &real_y, names.len(), count, seed);
        println!("SMOTE\t{}\trows from {} real rows in {:.1} s", synth_y.len(), used.len(), started.elapsed().as_secs_f64());
        x.extend(synth_x.iter().map(|v| *v as f32));
        splits.n_extrap += synth_y.len();
        y.extend(synth_y);
    }

    let mut config = Config::srbench(seed);
    config.smogd = smogd_on || smote_on;
    //   EVOLVE_HFF_LOG_TRAIN / _VAL / _BLOCK3 = 1   that block enters HFF on the log scale
    //   EVOLVE_HFF_LOG=1                            validation and the third block both
    let on = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
    let both = on("EVOLVE_HFF_LOG");
    config.log_scale = [on("EVOLVE_HFF_LOG_TRAIN"), both || on("EVOLVE_HFF_LOG_VAL"), both || on("EVOLVE_HFF_LOG_BLOCK3")];
    //   EVOLVE_HFF_NO_VAL=1             validation is left out of HFF: train + block three
    config.hff_without_validation = std::env::var("EVOLVE_HFF_NO_VAL").is_ok_and(|v| v == "1");
    //   EVOLVE_TOWER=1                  the tower objective (t_depth) joins HFF
    config.tower = std::env::var("EVOLVE_TOWER").is_ok_and(|v| v == "1");
    if let Some(population) = args.get(4).and_then(|a| a.parse::<u32>().ok()) {
        config.pop_intake = population / 4 * 3;
        config.pop_champion = population - config.pop_intake;
    }
    //   EVOLVE_VHEAD_EVERY / _START     the GROWING HEAD: the virtual head starts at START
    //                                   (12) and gains a position every EVERY generations
    //                                   (0 = off: the whole head from the start)
    config.vhead_every = std::env::var("EVOLVE_VHEAD_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if let Some(start) = std::env::var("EVOLVE_VHEAD_START").ok().and_then(|v| v.parse().ok()) {
        config.vhead_start = start;
    }
    //   EVOLVE_STOP_LOG10_P             the stop bar's p-value half (the engine's default is -19;
    //                                   "inf" switches it off)
    if let Some(bar) = std::env::var("EVOLVE_STOP_LOG10_P").ok().and_then(|v| v.parse::<f64>().ok()) {
        config.stop_log10_p = bar;
    }
    //   EVOLVE_BALANCED_TOURNAMENTS=1   the tournaments rank on hff's BALANCED pole (for
    //                                   diversity); the hall of fame, the stop bar and the
    //                                   report stay on TrueNorth
    config.balanced_tournaments = std::env::var("EVOLVE_BALANCED_TOURNAMENTS").is_ok_and(|v| v == "1");
    //   EVOLVE_HOF_FILE                 the hall of fame's best is appended here at every report
    config.hof_path = std::env::var("EVOLVE_HOF_FILE").ok().filter(|p| !p.is_empty());
    //   EVOLVE_PROGRESS_EVERY           a progress line on stderr every N generations (0 = none)
    config.progress_every = std::env::var("EVOLVE_PROGRESS_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    //   EVOLVE_COMPOUNDS=1              the compound functions (sqrt|a+-b|, 1/sqrt|a+-b|, 1/(a+-b)) join
    //                                   the symbol table — for the race's second pass
    config.compounds = std::env::var("EVOLVE_COMPOUNDS").is_ok_and(|v| v == "1");
    //   EVOLVE_GENES                    genes per chromosome (the engine's default is 3)
    if let Some(genes) = std::env::var("EVOLVE_GENES").ok().and_then(|v| v.parse::<u32>().ok()) {
        config.n_genes = genes;
    }
    //   EVOLVE_PUMP_EVERY               the pump's beat in generations (the engine's default is 4)
    if let Some(beat) = std::env::var("EVOLVE_PUMP_EVERY").ok().and_then(|v| v.parse::<u32>().ok()) {
        config.pump_every = beat;
    }
    //   EVOLVE_HEAD                     a gene's head length (the engine's default is 34)
    if let Some(head) = std::env::var("EVOLVE_HEAD").ok().and_then(|v| v.parse::<u32>().ok()) {
        config.head = head;
    }
    //   EVOLVE_POP_CHAMPION             the champion island's size; the `population`
    //                                   argument is then the INTAKE island's size
    //                                   (1500 + 1500 rather than the 3:1 split, which
    //                                   dates from when a generation was slow)
    if let Some(champion) = std::env::var("EVOLVE_POP_CHAMPION").ok().and_then(|v| v.parse::<u32>().ok()) {
        if let Some(intake) = args.get(4).and_then(|a| a.parse::<u32>().ok()) {
            config.pop_intake = intake;
        }
        config.pop_champion = champion;
    }
    // Experiment knobs come by environment so the positional arguments stay put:
    //   EVOLVE_RNC_LO / EVOLVE_RNC_HI   the range a gene's random constants are drawn from
    //   EVOLVE_RESTARTS                 split the time into this many independent searches
    let env = |k: &str| std::env::var(k).ok();
    if let (Some(lo), Some(hi)) = (env("EVOLVE_RNC_LO").and_then(|v| v.parse().ok()), env("EVOLVE_RNC_HI").and_then(|v| v.parse().ok())) {
        config.rnc_lo = lo;
        config.rnc_hi = hi;
    }
    //   EVOLVE_REDUNDANCY=1             leave-one-gene-out redundancy as an HFF objective
    //   EVOLVE_MAX_GENERATIONS          stop by GENERATIONS (give a long time cap): a
    //                                   comparison that does not depend on how busy the device is
    config.redundancy = env("EVOLVE_REDUNDANCY").is_some_and(|v| v == "1");
    if let Some(g) = env("EVOLVE_MAX_GENERATIONS").and_then(|v| v.parse().ok()) {
        config.max_generations = g;
    }
    let restarts: u32 = env("EVOLVE_RESTARTS").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    config.cleanse = args.get(6).and_then(|a| a.parse().ok()).unwrap_or(0.0);
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
    // A SYNTHETIC third block (SMOGD / SMOTE) is no judge: real rows only.
    let judged = if config.smogd { splits.n_train + splits.n_val } else { splits.total() };
    let fit_rows: Vec<Vec<(String, f64)>> =
        x.chunks(names.len()).take(judged).map(|r| names.iter().cloned().zip(r.iter().map(|v| f64::from(*v))).collect()).collect();
    // First say what the protected operators actually do on this data; then tidy.
    let resolved = resolve_protected(&out.math, &fit_rows).unwrap_or_else(|_| out.math.clone());
    let tidied = final_form(&resolved, &names, &fit_rows).unwrap_or(resolved);
    let tidy = Tree::parse(&tidied).expect("the final form parses");
    println!("model: {}", tidy.to_infix_faithful());
    println!("GENERATIONS\t{}\t{}\t{:.3e}", out.generations, out.stopped_by, out.best.one_minus_r2[1]);
    // What evolution selected on: the model's HFF fitness (the tournaments rank on
    // it; smaller is better), and 1-R² on each block — train, validation, and the
    // third block (SMOGD or edge rows; "-" when there is none).
    let third = if splits.n_extrap > 0 { format!("{:.3e}", out.best.one_minus_r2[2]) } else { "-".to_string() };
    println!("HFF\t{:.6}\t{:.3e}\t{:.3e}\t{third}", out.best.fitness, out.best.one_minus_r2[0], out.best.one_minus_r2[1]);
    // The same three blocks as MSE: 1-R² is MSE over the block's variance of y, so
    // MSE = (1-R²) x var(y) on that block — the engine's own definition (Caps).
    let blocks = [(0, splits.n_train), (splits.n_train, splits.n_train + splits.n_val), (splits.n_train + splits.n_val, y.len())];
    let mse: Vec<String> = blocks
        .iter()
        .zip(out.best.one_minus_r2)
        .map(|(&(lo, hi), omr2)| {
            if hi <= lo {
                return "-".to_string();
            }
            let mean = y[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
            let var = y[lo..hi].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (hi - lo) as f64;
            format!("{:.3e}", omr2 * var)
        })
        .collect();
    println!("MSE\t{}\t{}\t{}", mse[0], mse[1], mse[2]);
    println!("TOWER\t{}\t{}", out.best.t_depth, if config.tower { "in HFF" } else { "reported only" });
    // The HFF angle as a p-value (hff's beta-CDF): how likely a random point on the
    // m-objective sphere is to sit this near the pole.
    let (p_value, log10_p) = fuller::evolve::engine::hff_p_value(out.best.fitness, engine.hff_dimensions());
    println!("PVALUE\t{p_value:.3e}\t{log10_p:.2}\t{}", engine.hff_dimensions());
    println!("MODEL_INFIX\t{}", tidy.to_infix_faithful());
    // The same model with every protected operator written as the ordinary one —
    // the FUNCTION, without the execution guard. It is what goes to SRBench, which
    // only compares it symbolically with the law and never executes it (accuracy
    // comes from predict(), the chromosome). MODEL_INFIX stays the faithful form:
    // the harness executes THAT one to prove the report computes the chromosome.
    println!("MODEL_PLAIN\t{}", tidy.to_infix());
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
