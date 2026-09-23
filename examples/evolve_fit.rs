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
//!
//! THE RUN CARD ([`fuller::evolve::card`]). Every run WRITES one: the whole
//! `Config`, the dataset and its splits, the synthetic third block, the git
//! commit of this binary and what the engine derived — above all how many HFF
//! objectives it ended up with. It goes to `EVOLVE_CARD_OUT`, or to `card.json`
//! beside the telemetry stream when there is one.
//!
//! `EVOLVE_CARD=path/to/card.json` runs FROM a card. The environment still wins
//! over it, so a card is a starting point to vary from, and the card that run
//! writes records what ACTUALLY ran. Precedence, highest first: environment,
//! positional argument, card, built-in default. An absent card changes nothing.

use fuller::chrom_score::Splits;
use fuller::evolve::card::{self, Card, Code, DataCard, Derived, IslandSpan, Run, Synthetic};
use fuller::evolve::engine::{evaluate_math, final_form_reporting, resolve_protected, Config, Data, Engine, Lane};
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
    // THE RUN CARD IS READ FIRST, before a single default is taken, because it
    // is what every default falls back to. Absent (the ordinary case) it is
    // None and nothing below sees any difference. A card that is NAMED and
    // cannot be read is a stopped run, never a silent fall-through to the
    // built-in defaults: "use card 26" must not quietly become "use no card".
    let card: Option<Card> = std::env::var("EVOLVE_CARD").ok().filter(|p| !p.is_empty()).map(|p| {
        let c = Card::load(&p).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("CARD\tfrom {p}\tid {}\tcommit {}{}", c.card_id, c.code.git_commit, if c.code.git_dirty { " (dirty)" } else { "" });
        c
    });
    let path = args.first().cloned().or_else(|| card.as_ref().map(|c| c.data.path.clone()));
    let path = &path.expect("usage: evolve_fit data.tsv [seed] [seconds] [max_rows] (or EVOLVE_CARD=card.json)");
    // POSITIONAL, and easy to get wrong: arg 1 is the SEED and arg 2 is the
    // seconds. Passing a big number meaning "no time cap" in slot 1 sets the
    // seed and leaves the cap at its 30 s default, which silently truncated
    // several long runs. EVOLVE_SEED and EVOLVE_SECONDS say which is which.
    let seed: u32 = std::env::var("EVOLVE_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| args.get(1).and_then(|a| a.parse().ok()))
        .or_else(|| card.as_ref().map(|c| c.run.seed))
        .unwrap_or(7001);
    let seconds: f64 = std::env::var("EVOLVE_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| args.get(2).and_then(|a| a.parse().ok()))
        .or_else(|| card.as_ref().map(|c| c.run.budget_seconds))
        .unwrap_or(30.0);
    eprintln!("BUDGET\tseed {seed}\t{seconds:.0} s");
    let max_rows: usize =
        args.get(3).and_then(|a| a.parse().ok()).or_else(|| card.as_ref().map(|c| c.data.max_rows)).unwrap_or(5000);

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
    let use_all = args.get(5).map_or_else(|| card.as_ref().is_some_and(|c| c.data.use_all), |a| a == "all");
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
    let edge = std::env::var("EVOLVE_EDGE").ok().filter(|p| !p.is_empty()).or_else(|| card.as_ref().and_then(|c| c.data.edge_path.clone()));
    if let Some(edge_path) = edge.clone() {
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
    let smogd_on = std::env::var("EVOLVE_SMOGD").map_or_else(|_| card.as_ref().is_some_and(|c| c.synthetic.smogd), |v| v == "1");
    // EVOLVE_SMOGD_NOISE: the multiplier on the neighbours' variance (1 = the
    // original). Read here rather than inside the branch so the card records it
    // whether or not SMOGD ran.
    let noise_multiplier: f64 = std::env::var("EVOLVE_SMOGD_NOISE")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| card.as_ref().map(|c| c.synthetic.smogd_noise))
        .unwrap_or(1.0);
    if smogd_on {
        let started = std::time::Instant::now();
        let real: Vec<f64> = used.iter().flat_map(|&r| inputs(&rows[r])).collect();
        let real_y: Vec<f64> = used.iter().map(|&r| rows[r][target]).collect();
        let embedding = embed_2d(&real, names.len(), u64::from(seed)).expect("the 2D embedding");
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
    let smote_on = std::env::var("EVOLVE_SMOTE").map_or_else(|_| card.as_ref().is_some_and(|c| c.synthetic.smote), |v| v == "1");
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

    // THE CONFIG: the card's when there is one, the engine's defaults when there
    // is not, and then THE ENVIRONMENT OVERRIDES on top of either — so a card is
    // a starting point to vary from and never a cage. Every knob that was read
    // here by name now lives in `card::apply_env`, which is a pure function of
    // an injected lookup: that is what lets a test assert "the environment wins"
    // and "an empty environment changes nothing" without racing on process env.
    //
    // A card's `seed` is NOT taken here. `Config::seed` is per-RESTART (derived
    // below from the fit's seed), and the fit's seed has already been resolved
    // from the environment, the positional argument and the card in that order.
    let mut config = match card.as_ref() {
        Some(c) => Config { seed, ..c.config.clone() },
        None => Config::srbench(seed),
    };
    // A THIRD BLOCK IS IN HFF WHEN ONE WAS GENERATED. Not a card field of its
    // own: `synthetic.smogd` / `.smote` say what MADE it, and this says what the
    // engine does with it, which is decided by whether the rows exist.
    config.smogd = smogd_on || smote_on;
    // THE POSITIONAL `population`, arg 4: one number split 3:1. It goes on
    // BEFORE `apply_env` so the named EVOLVE_POP_INTAKE / _CHAMPION still win,
    // which is the order the block this replaces had.
    if let Some(population) = args.get(4).and_then(|a| a.parse::<u32>().ok()) {
        config.pop_intake = population / 4 * 3;
        config.pop_champion = population - config.pop_intake;
    }
    // EVOLVE_POP_CHAMPION re-reads arg 4 as the INTAKE's size, so
    // `... 1500 ... EVOLVE_POP_CHAMPION=1500` means 1500 + 1500 rather than the
    // 3:1 split. Kept, and kept here with the other positional handling.
    if std::env::var("EVOLVE_POP_CHAMPION").is_ok() {
        if let Some(intake) = args.get(4).and_then(|a| a.parse::<u32>().ok()) {
            config.pop_intake = intake;
        }
    }
    card::apply_env(&mut config, &|k| std::env::var(k).ok());
    eprintln!("POPULATION\t{} intake + {} champion", config.pop_intake, config.pop_champion);
    // THE OUTPUT PATHS, which are never carded (see `Config`'s own note: a
    // replay carrying them truncates the original run's telemetry stream and
    // resumes its checkpoint). Where a run writes is the caller's to say, every
    // time.
    //   EVOLVE_HOF_FILE                 the hall of fame's best is appended here at every report
    //   EVOLVE_CHECKPOINT_DIR           five rotating slots, so a fit killed at any moment
    //                                   resumes from the beat before it
    //   EVOLVE_CHECKPOINT_EVERY         how often, in SECONDS (default 60 when a directory is
    //                                   given; 0 = only at the end)
    //   EVOLVE_GENEALOGY_FILE           THE GENEALOGY LOG: every individual gets an identity, an
    //                                   age and a lineage, and this file takes the best of every
    //                                   generation, every arrival, and the winner's whole chain
    //   EVOLVE_TELEMETRY_FILE           THE TELEMETRY STREAM, a versioned JSONL record per
    //                                   progress beat, which `hff-watch --follow` repaints from
    //   EVOLVE_TELEMETRY_RUN_ID         what the stream calls this run
    config.hof_path = std::env::var("EVOLVE_HOF_FILE").ok().filter(|p| !p.is_empty());
    config.checkpoint_dir = std::env::var("EVOLVE_CHECKPOINT_DIR").ok().filter(|p| !p.is_empty());
    if config.checkpoint_dir.is_some() {
        config.checkpoint_every_seconds =
            std::env::var("EVOLVE_CHECKPOINT_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(60.0);
    }
    if let Some(m) = std::env::var("EVOLVE_COHORT_MERGE").ok().and_then(|v| v.parse::<u32>().ok()) {
        config.cohort_merge = m;
    }
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
    //   EVOLVE_POP_INTAKE               the intake island's size, by name. The intake
    //                                   used to come ONLY from positional argument 4,
    //                                   so a caller setting EVOLVE_POP_INTAKE got the
    //                                   default 600 and no complaint — a 200,000-row
    //                                   run reported itself as 600 + 100,000.
    if let Some(intake) = std::env::var("EVOLVE_POP_INTAKE").ok().and_then(|v| v.parse::<u32>().ok()) {
        config.pop_intake = intake;
    }
    if let Some(champion) = std::env::var("EVOLVE_POP_CHAMPION").ok().and_then(|v| v.parse::<u32>().ok()) {
        if let Some(intake) = args.get(4).and_then(|a| a.parse::<u32>().ok()) {
            config.pop_intake = intake;
        }
        config.pop_champion = champion;
    }
    eprintln!("POPULATION\t{} intake + {} champion", config.pop_intake, config.pop_champion);
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
    //   EVOLVE_PAIRS                    pairs of islands (intake + champion); the island
    //                                   sizes are ONE pair's, so the population is this
    //                                   many times as large (default 1)
    //   EVOLVE_CROSS_EVERY              THE CROSS STEP's beat: every this many generations
    //                                   each intake island takes in the best of the other
    //                                   pairs' champion islands (default 0 = never)
    //   EVOLVE_K_MIGRANTS               how many each champion island sends (default 3)
    if let Some(n) = env("EVOLVE_PAIRS").and_then(|v| v.parse().ok()) {
        config.n_pairs = n;
    }
    if let Some(n) = env("EVOLVE_CROSS_EVERY").and_then(|v| v.parse().ok()) {
        config.cross_every = n;
    }
    if let Some(n) = env("EVOLVE_K_MIGRANTS").and_then(|v| v.parse().ok()) {
        config.k_migrants = n;
    }
    //   EVOLVE_LANES                    THE SWIM LANES: one rule set per island pair, as
    //                                   `name:explore:recombine[:cleanse]` separated by
    //                                   commas — "general:1:1,explorer:3:0.5". There must
    //                                   be exactly EVOLVE_PAIRS of them. `explore` scales
    //                                   the point-mutation and transposition rates,
    //                                   `recombine` the three crossovers, and the lane's
    //                                   name is reported as having found the law.
    if let Some(spec) = env("EVOLVE_LANES") {
        let mut lanes = Vec::new();
        for one in spec.split(',').filter(|s| !s.trim().is_empty()) {
            let f: Vec<&str> = one.split(':').collect();
            let num = |i: usize, what: &str| -> f64 {
                f.get(i)
                    .unwrap_or_else(|| panic!("EVOLVE_LANES: lane {one:?} has no {what}; it is name:explore:recombine[:cleanse]"))
                    .parse()
                    .unwrap_or_else(|e| panic!("EVOLVE_LANES: lane {one:?} {what}: {e}"))
            };
            lanes.push(Lane {
                name: (*f.first().expect("a lane needs a name")).to_string(),
                explore: num(1, "explore"),
                recombine: num(2, "recombine"),
                cleanse: f.get(3).map(|v| v.parse().expect("EVOLVE_LANES: cleanse")),
            });
        }
        eprintln!("LANES {}", lanes.iter().map(|l| format!("{} x{}/{}", l.name, l.explore, l.recombine)).collect::<Vec<_>>().join(" | "));
        config.lanes = Some(lanes);
    }
    //   EVOLVE_SNAP_EVERY               SNAP WINNERS' beat in generations (0 = off, the default):
    //                                   kept snapped forms are written back into the genes
    //   EVOLVE_SNAP_TOP_K               rows per island, by fitness, that are snap winners (0 = all)
    if let Some(n) = env("EVOLVE_SNAP_EVERY").and_then(|v| v.parse().ok()) {
        config.snap_every = n;
    }
    if let Some(n) = env("EVOLVE_SNAP_TOP_K").and_then(|v| v.parse().ok()) {
        config.snap_top_k = n;
    }
    //   EVOLVE_FOLD_EVERY               THE FOLD OPERATOR's beat in generations (0 = off): the
    //                                   winners of every island have their near-constant subtrees
    //                                   collapsed to the constants they are, and the clean gene is
    //                                   written back into its own row
    //   EVOLVE_FOLD_TOP_K               rows per island, by fitness, the fold examines (0 = all)
    if let Some(n) = env("EVOLVE_FOLD_EVERY").and_then(|v| v.parse().ok()) {
        config.fold_every = n;
    }
    if let Some(n) = env("EVOLVE_FOLD_TOP_K").and_then(|v| v.parse().ok()) {
        config.fold_top_k = n;
    }
    //   EVOLVE_GENEALOGY_FILE           THE GENEALOGY LOG: every individual of the fit gets an
    //                                   IDENTITY, an AGE (generations since its genotype entered
    //                                   the population) and a LINEAGE, and this file takes the
    //                                   best of every generation, every arrival (pump, cross,
    //                                   snap) and the winner's chain back to its founder. Unset
    //                                   = off, and the engine is what it was, bit for bit.
    config.genealogy_path = env("EVOLVE_GENEALOGY_FILE").filter(|p| !p.is_empty());
    //   EVOLVE_TELEMETRY_FILE           THE TELEMETRY STREAM: a versioned JSONL record per
    //                                   progress beat — the global state, every island's rows
    //                                   and best HFF, and every cohort's split by island —
    //                                   which `hff-watch --follow` repaints from. The prose
    //                                   log is unchanged; this is a SECOND stream, and it is
    //                                   the production API (never parse the prose log).
    //                                   Unset = off, and the engine is what it was, bit for
    //                                   bit. It rides on EVOLVE_PROGRESS_EVERY, so that is
    //                                   defaulted to 10 here when a stream is asked for and
    //                                   no beat was given — a caller who asks to watch a fit
    //                                   should not get an empty file because of a second knob.
    //   EVOLVE_TELEMETRY_RUN_ID         what the stream calls this run (default
    //                                   `<dataset>-seed<seed>`)
    config.telemetry_path = env("EVOLVE_TELEMETRY_FILE").filter(|p| !p.is_empty());
    config.telemetry_run_id = env("EVOLVE_TELEMETRY_RUN_ID").filter(|p| !p.is_empty());
    config.telemetry_dataset = std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned());
    // The stream rides on the progress report, so a caller who asks to watch a
    // fit should not get an empty file because of a second knob.
    if config.telemetry_path.is_some() && config.progress_every == 0 {
        config.progress_every = 10;
    }
    if let Some(lanes) = &config.lanes {
        eprintln!("LANES {}", lanes.iter().map(|l| format!("{} x{}/{}", l.name, l.explore, l.recombine)).collect::<Vec<_>>().join(" | "));
    }
    //   EVOLVE_RESTARTS                 split the time into this many independent searches
    let restarts: u32 = std::env::var("EVOLVE_RESTARTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| card.as_ref().map(|c| c.run.restarts))
        .unwrap_or(1)
        .max(1);
    // THE CLEANSING MUTATION's rate, positional argument 6 over the card's.
    if let Some(rate) = args.get(6).and_then(|a| a.parse().ok()) {
        config.cleanse = rate;
    }
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
        // THE RUN CARD, written at the START of the fit — before `fit()`, so a
        // run killed at any moment has already said how it was configured. It
        // goes out from the FIRST restart, whose engine is the one the card's
        // settings describe; the later restarts differ only in their derived
        // seed, which the card names as `run.seed`.
        //
        // Written HERE and not earlier because this is where the DERIVED facts
        // exist: `hff_dimensions()`, the island spans and the real population
        // are the engine's own numbers, and taking them from it is what keeps
        // the card from drifting into a parallel calculation of its own.
        if k == 0 {
            let written = Card {
                card_version: card::CARD_VERSION,
                // THE CARD'S NAME, in the order that makes "use card 26" work: the
                // run's telemetry id when it has one (which is the log directory's
                // tag), then the card this run came FROM, then dataset-and-seed.
                card_id: config
                    .telemetry_run_id
                    .clone()
                    .or_else(|| card.as_ref().map(|c| c.card_id.clone()))
                    .unwrap_or_else(|| format!("{}-seed{seed}", config.telemetry_dataset.clone().unwrap_or_else(|| "run".into()))),
                written_utc: fuller::evolve::telemetry::now_utc(),
                code: Code::of_this_binary(),
                data: DataCard {
                    path: path.clone(),
                    max_rows,
                    use_all,
                    edge_path: edge.clone(),
                    test_path: args.get(7).cloned(),
                    n_train: splits.n_train,
                    n_val: splits.n_val,
                    n_third: splits.n_extrap,
                    n_test: test_rows.len(),
                },
                synthetic: Synthetic { smogd: smogd_on, smote: smote_on, smogd_noise: noise_multiplier },
                run: Run { seed, budget_seconds: seconds, restarts },
                config: config.clone(),
                derived: Derived {
                    hff_objectives: engine.hff_dimensions(),
                    population: engine.layout.pop,
                    islands: engine.islands.iter().map(|i| IslandSpan { lo: i.lo, hi: i.hi }).collect(),
                    live_cohorts: card::live_cohorts(config.cohort_merge, config.pump_every),
                },
                // A run writes its card BEFORE it starts, so it has nothing to
                // say yet. Findings are appended when it ends.
                notes: Vec::new(),
            };
            // WHERE IT GOES: `EVOLVE_CARD_OUT` when it is given, otherwise
            // `card.json` beside the telemetry stream — the run's own output
            // directory, which is where its other outputs already are. A run
            // with neither writes no file rather than dropping one into
            // whatever directory it was launched from, and says so.
            // BESIDE THE RUN ALWAYS, and in the library as well.
            //
            // `card.json` next to the telemetry stream is the copy that matters:
            // "when we run something, I want to see the specification of what we
            // ran BESIDE THE LOG FILES" (Andrew). It is written whenever there
            // is a run directory to write it into, and `EVOLVE_CARD_OUT` no
            // longer MOVES it -- pointing that at a card library used to take
            // the card away from the run it describes, which is the opposite of
            // the point.
            let beside = config
                .telemetry_path
                .as_ref()
                .and_then(|t| std::path::Path::new(t).parent().map(|d| d.join("card.json").to_string_lossy().into_owned()));
            let library = std::env::var("EVOLVE_CARD_OUT").ok().filter(|p| !p.is_empty());
            let mut wrote_any = false;
            for path in [beside, library].into_iter().flatten() {
                // A card that cannot be written must not kill a fit that is
                // otherwise ready to run: the fit is the expensive thing and the
                // card is a record of it.
                match written.write(&path) {
                    Ok(()) => {
                        eprintln!("CARD\twritten {path}\tid {}\t{} HFF objectives", written.card_id, written.derived.hff_objectives);
                        wrote_any = true;
                    }
                    Err(e) => eprintln!("CARD\tNOT WRITTEN to {path}: {e}"),
                }
            }
            if !wrote_any {
                eprintln!("CARD\tnot written (no telemetry file to sit beside, and no EVOLVE_CARD_OUT)");
            }
        }
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
    let (mut engine, mut out) = kept.expect("at least one search ran");
    out.generations = generations;
    out.individuals = individuals;

    let t = &out.timing;
    println!("dataset {path}\nrows: train {n_train}, validation {}, unseen test {}", splits.n_val, test_rows.len());
    println!(
        "population {} x ({}+{}) = {} | {} generations in {:.1} s = {:.1} ms per generation | {} individuals = {:.0} per second | stopped by {}",
        engine.pairs().len(), engine.islands[0].hi, engine.islands[1].hi - engine.islands[1].lo, engine.layout.pop, out.generations, out.seconds,
        1e3 * out.seconds / f64::from(out.generations.max(1)), out.individuals, out.individuals as f64 / out.seconds, out.stopped_by
    );
    println!(
        "seconds: variation {:.2} | read-back {:.2} | decode {:.2} | evaluate {:.2} | link+scale+metrics {:.2} | HFF {:.2} | pump {:.2} | cross {:.2}",
        t.vary, t.read, t.decode, t.evaluate, t.score, t.hff, t.pump, t.cross
    );
    if config.snap_every > 0 {
        println!("seconds: snap {:.2} ({:.3} per beat in the snap step itself)", t.snap, out.snap.seconds / out.snap.beats.max(1) as f64);
        println!("{}", out.snap.line());
        println!("{}", out.snap.detail());
    }
    // THE FOLD OPERATOR: what it cost the search, per beat, and what it bought —
    // the head positions it gave back to genes that were spending them on blobs
    // holding one number.
    if config.fold_every > 0 {
        println!("seconds: fold {:.2} ({:.3} per beat in the fold step itself)", t.fold, out.fold.seconds / out.fold.beats.max(1) as f64);
        println!("{}", out.fold.line());
    }
    // THE BEAM: beats, mutants, how many beat their original, the best log10 p
    // before and after, and the seconds it cost — plus which wraps earned their
    // place and what the float zone held.
    if config.beam_every > 0 {
        println!("seconds: beam {:.2} ({:.3} per beat)", t.beam, out.beam.seconds / out.beam.beats.max(1) as f64);
        println!("{}", out.beam.line());
        println!("{}", out.beam.wrap_detail());
        println!("{}", out.beam.float_line());
    }
    println!("genes evaluated {} (unique per generation), over the 64-node limit {}", out.unique_genes, out.oversized_genes);
    // TYPED TRANSCENDENTAL DEPTH's two measurements. Printed on BOTH arms: the
    // depth histogram of an untyped run is the control the typed one is read
    // against, so a line that only appeared with the knob on would leave the
    // A/B with nothing to compare.
    println!(
        "TYPED\tceiling={}\trefused={}\tdepths={}",
        config.typed_depth.map_or_else(|| "off".to_string(), |c| c.to_string()),
        out.typed_refused,
        out.depth_histogram.iter().enumerate().map(|(d, n)| format!("{d}:{n}")).collect::<Vec<_>>().join(" ")
    );
    println!("1 - R²: train {:.3e}, validation {:.3e}", out.best.one_minus_r2[0], out.best.one_minus_r2[1]);

    // fuller's final form, the data as judge: the fit rows (train + validation)
    // decide which inputs are positive and which prunes change nothing.
    // A SYNTHETIC third block (SMOGD / SMOTE) is no judge: real rows only.
    let judged = if config.smogd { splits.n_train + splits.n_val } else { splits.total() };
    let fit_rows: Vec<Vec<(String, f64)>> =
        x.chunks(names.len()).take(judged).map(|r| names.iter().cloned().zip(r.iter().map(|v| f64::from(*v))).collect()).collect();
    // First say what the protected operators actually do on this data; then tidy.
    let resolved = resolve_protected(&out.math, &fit_rows).unwrap_or_else(|_| out.math.clone());
    // The reduction sizes its tolerance against how well the model actually fits:
    // a term worth less than a tenth of the model's own error is not the law.
    // AND WHAT THE ROUNDING GENERATOR FOUND on the way: the folds go onto the
    // telemetry stream (after `run_end` — this is where the fold happens) so the
    // viewer's discoveries panel can show them beside snap's substitutions.
    // AND THE LEAVE-ONE-OUT's drops beside them: the two are complements, and a
    // panel shown one without the other is shown half the tidy. The summary goes
    // out even when the final form FAILED — a viewer must be able to tell "it ran
    // and found nothing" from "it never ran", and both are a zero.
    let tidy_out = final_form_reporting(&resolved, &names, &fit_rows, Some(out.best.one_minus_r2[0]))
        .unwrap_or_else(|_| fuller::evolve::engine::Tidied { form: resolved.clone(), ..Default::default() });
    let (tidied, folds, drops) = (tidy_out.form, tidy_out.folds, tidy_out.drops);
    for f in &folds {
        println!("FOLD\t{}\t{:.9}\t{}", f.nodes, f.value, f.infix);
    }
    for r in &drops {
        println!("REDUCE\t{}\t{:.9}\t{}", r.nodes, r.held, r.infix);
    }
    engine.report_tidy(out.generations, &folds, &drops);
    // ... and the data guided rewrites once more: fuller's linter can WRITE a shape
    // they cover (it turns Add (Neg (Log b)) (Log a) into Sub (Log a) (Log b)), and a
    // form that only appears after the tidy must not slip past them.
    let tidied = resolve_protected(&tidied, &fit_rows).unwrap_or(tidied);
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
    // Unique genes evaluated, how many of them were dropped as over the 64-node
    // limit, and the individuals they stood for.
    println!("GENES\t{}\t{}\t{}", out.unique_genes, out.oversized_genes, out.individuals);
    // THE WINNER'S LINEAGE: how old it was in generations, the generation its line
    // began at, the mechanism that put its founder into the population, and its id.
    // Only when EVOLVE_GENEALOGY_FILE is set.
    if let Some(m) = out.lineage {
        println!("LINEAGE\t{}\t{}\t{}\t{}", m.age, m.founder_generation, m.founder_origin, m.id);
        println!("GENEALOGY\t{}\t{}\t{}\t{:.2}", out.genealogy_minted, out.genealogy_lines, out.genealogy_bytes, out.timing.genealogy);
    }
    // THE FINAL POPULATION's ages (min, median, max, mean) over every row and over
    // the best ten, and how many distinct lines the best fifty descend from — the
    // diversity number a decision about age layers would rest on.
    if let Some(a) = out.population_ages {
        println!("AGES\t{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\t{:.2}", a.all.0, a.all.1, a.all.2, a.all.3, a.best_10.0, a.best_10.1, a.best_10.2, a.best_10.3);
        println!("FOUNDERS\t{}\t{}", a.founders_best_50, a.founders_all);
    }
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
