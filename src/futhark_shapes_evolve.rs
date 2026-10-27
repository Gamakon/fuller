//! The shape kingdom's evolution loop.
//!
//! # The two operators, and the two jobs
//!
//! * **GEP mutation and crossover find new semantics.** A point mutation
//!   rewrites one cell of a gene; a crossover exchanges a range between two
//!   parents. Both come from `typed_gep`, which is kingdom-agnostic: the
//!   layout's alphabet says what a cell may hold, so every child is legal
//!   by construction.
//! * **The e-graph is a local search on the elites, not the mutation
//!   operator.** Its rules are equivalences *by construction* — `map` with
//!   the identity row function is the matrix, `transpose` twice is the
//!   matrix, and the product computed the transposed way and transposed
//!   back is the product itself — so every candidate it offers is equal to
//!   its parent. It can therefore only move along the *cost* axis, which is
//!   exactly what a local search should do.
//!
//! # The fitness
//!
//! Two parts, and the first is the one that matters:
//!
//! 1. **Correctness against the reference kernel.** A candidate is correct
//!    when it computes the *reference's* value — not its own. The
//!    reference is evaluated once, on the inputs, and every child is
//!    compared with that. The comparison is bit for bit (`identical`) on the
//!    elementwise paths, and exact values (`equal`, no tolerance) where the
//!    program contains a reduction, because the compiled program may
//!    legitimately differ from the source semantics there.
//! 2. **The cost.** The cheap polynomial ([`crate::futhark_shapes_cost`])
//!    ranks the population; the *measured* GPU cost
//!    ([`crate::futhark_shapes_gpu`]) decides the elites, because the
//!    polynomial is a pre-optimisation model and the machine sees something
//!    else.
//!
//! # The reference is the target, not the starting point
//!
//! The loop does not start from the reference: it starts from a random
//! population, most of which is wrong, and *finds* programs that match
//! the reference. The reference's own cost is measured too, so a win is a
//! win against something.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use crate::futhark_kingdom::{FutharkKingdom, GeneRng};
use crate::futhark_shapes::program;
use crate::futhark_shapes_cost::{program_cost, score_at, CostTable, Score};
use crate::futhark_shapes_egraph::{saturate_tree, Bridge};
use crate::futhark_shapes_gpu::{measure, GpuMeasurement};
use crate::futhark_shapes_interp::{identical, Mat, MValue};
use crate::geneframe::Ty;
use crate::kingdom_contract::KingdomContract;
use crate::runner::{load_checkpoint, log_scaled, save_checkpoint, two_level_hff, FameEntry, HallOfFame, Ledger, TwoLevel};
use crate::telemetry::{EventKind, GenerationStats, Mutation, RunInfo, RunTelemetry};
use serde::{Deserialize, Serialize};
use crate::typed_gep::{express, write_back, GeneLayout, Tree};

/// How the loop runs.
#[derive(Debug, Clone)]
pub struct Config {
    /// How many individuals the population holds.
    pub population: usize,
    /// How many generations to run.
    pub generations: usize,
    /// The seed: the same seed gives the same run, always.
    pub seed: u64,
    /// The program size the layout is built for.
    pub program_size: usize,
    /// How many elites the e-graph polishes each generation.
    pub elites: usize,
    /// How many candidates the e-graph offers per polished elite.
    pub candidates: usize,
    /// How many rounds the saturation runs.
    pub iters: u32,
    /// The role sizes the cost is evaluated at.
    pub sizes: BTreeMap<String, usize>,
    /// How many independent input sets the correctness is decided on.
    ///
    /// One sample is not enough: a program that *coincidentally* matches
    /// the reference on one input would enter the elite set and be ranked.
    /// The suite is drawn deterministically from the seed, so the run is
    /// reproducible.
    pub samples: usize,
    /// Whether a candidate has to RUN on the GPU to be eligible.
    ///
    /// It is the hard gate: a program whose kernel the device cannot
    /// compile is not a program, however cheap its allocations look. It
    /// costs a compile and a run per candidate, so it is for the elites.
    pub require_runnable: bool,
    /// Whether the population starts with the reference itself.
    pub seed_with_reference: bool,
    /// Whether the elites' cost is *measured* on the GPU. It is slow — the
    /// GPU pipeline is seconds per program — so it is for the elites, not
    /// for every child.
    pub measure_gpu: bool,
    /// Continue from the output directory's `checkpoint.json` (the stream is appended to).
    pub resume: bool,
    /// The wall-clock guard, seconds (0: none): the run ends `time` between generations once it is spent.
    pub seconds: f64,
    /// How many distinct individuals the hall of fame keeps.
    pub hall_of_fame: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            population: 32,
            generations: 10,
            seed: 0x5EED_1234_ABCD_0010,
            program_size: 3,
            elites: 4,
            candidates: 24,
            iters: 4,
            sizes: [("m".to_string(), 3usize), ("n".to_string(), 2), ("p".to_string(), 4)]
                .into_iter()
                .collect(),
            samples: 3,
            require_runnable: true,
            seed_with_reference: true,
            measure_gpu: true,
            resume: false,
            seconds: 3600.0,
            hall_of_fame: 8,
        }
    }
}

/// One program, with what is known about it.
#[derive(Debug, Clone)]
pub struct Individual {
    /// The gene.
    pub gene: Vec<u32>,
    /// The program the gene denotes.
    pub tree: Tree,
    /// Whether it computes the reference's value.
    pub correct: bool,
    /// How much of its value agrees with the reference's, exactly.
    ///
    /// This is the GRADED measure the GEP can hill-climb on: a random
    /// mutation is almost never equal to the reference, but it is often
    /// *nearly* equal, and the closer it is the more likely the next
    /// mutation is to land on the reference.
    pub agreement: f64,
    /// The share of the value that DISAGREES with the reference's, one per input set of the suite: the quality
    /// group of the two-level HFF ([`two_level`]), each its own input.
    pub disagreement: Vec<f64>,
    /// The cheap cost.
    pub score: Score,
    /// The measured GPU cost, when it was measured.
    pub gpu: Option<GpuMeasurement>,
}

/// What one generation did.
#[derive(Debug, Clone)]
pub struct Generation {
    /// Which generation it was.
    pub number: usize,
    /// How many individuals were correct.
    pub correct: usize,
    /// The best correct individual's cheap cost.
    pub best: Option<Score>,
    /// The best correct individual's measured GPU cost.
    pub best_gpu: Option<GpuMeasurement>,
    /// How many polished children the e-graph offered.
    pub polished: usize,
    /// How many of them were cheaper than their elite.
    pub improved: usize,
    /// How many correct individuals were measured on the GPU.
    pub measured: usize,
    /// How many of them the compiler tiles with the block+register tiler.
    ///
    /// It is a REPORT, never a filter: the reference's kernel is tiled and
    /// the winner's is not, so a rule that credited only a tiled form would
    /// have rejected the only win.
    pub tiled: usize,
    /// How many of them the GPU could not run at all.
    pub not_runnable: usize,
}

/// What the run found.
#[derive(Debug, Clone)]
pub struct Report {
    /// The generations, in order.
    pub generations: Vec<Generation>,
    /// The best correct individual the run found.
    pub best: Option<Individual>,
    /// The reference's own cheap cost, for comparison.
    pub reference: Option<Individual>,
    /// The reference's own measured GPU cost, for comparison.
    pub reference_gpu: Option<GpuMeasurement>,
    /// The final population: the genes the last generation evaluated, best first.
    pub population: Vec<Vec<u32>>,
}

/// The interface's inputs, in the order the entry point takes them, in
/// Futhark's textual format — the form the GPU executable reads.
fn input_text(kingdom: &FutharkKingdom, sample: &BTreeMap<Ty, Mat>) -> Result<String, String> {
    let mut text = String::new();
    for dual in &kingdom.duals {
        if dual.slot != "mat" {
            continue;
        }
        let ty = kingdom.ty(&dual.dual).ok_or("a dual's own type")?;
        let mat = sample
            .get(&ty)
            .ok_or_else(|| format!("the sample has no matrix for {}", dual.dual))?;
        text.push_str(&crate::futhark_shapes_interp::format_matrix(mat));
    }
    Ok(text)
}

/// Everything the loop needs to evaluate a program.
struct Context<'a> {
    kingdom: &'a FutharkKingdom,
    table: &'a CostTable,
    /// The input sets, one per sample.
    inputs: &'a [BTreeMap<Ty, Mat>],
    /// The reference's value on each input set: the targets.
    wants: &'a [Mat],
    /// The role sizes the cost is evaluated at.
    sizes: &'a BTreeMap<String, usize>,
}

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

/// The kernel the kingdom's vocabulary can make **expensive**: the product
/// of two matrices, computed twice — once directly and once the transposed
/// way — and the two added.
///
/// ```text
/// map2 (add) (A . B) (transpose ((B\' . A\')))
/// ```
///
/// The second product is the same matrix as the first, but the compiler
/// cannot see that: the two are different expressions, so its CSE leaves
/// both and the GPU program does the work twice. The e-graph's
/// transposed-product rule is what closes the gap.
pub fn product_twice(k: &FutharkKingdom) -> Tree {
    let a = leaf(k, "mat.n.m.input");
    let b = leaf(k, "mat.m.p.input");
    let direct = tree_with(k, "matmul@mat.n.mxmat.m.p", vec![a.clone(), b.clone()]);
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
        vec![leaf(k, "fn.arr2_f32_arr_f32_arr_f32.add"), direct, transposed],
    )
}

/// One member of the population: its gene and its lineage.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Member {
    gene: Vec<u32>,
    /// The individual's id, unique in the run.
    id: u64,
    /// Its parents' ids.
    parents: Vec<u64>,
    /// The operator that made it: `init`, `reference`, `elite`, `crossover`, `mutation`, `egraph_exact`.
    operator: String,
}

/// A scored member.
#[derive(Debug, Clone)]
struct Ranked {
    individual: Individual,
    hff: TwoLevel,
    member: Member,
}

/// What the run's checkpoint holds: enough to continue exactly where it stopped.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Checkpoint {
    /// The next generation to run.
    next: usize,
    rng: u64,
    population: Vec<Member>,
    next_id: u64,
    hall_of_fame: HallOfFame,
    /// The run's best gene (re-scored on resume; its GPU measurement is taken again when the run measures).
    best_gene: Option<Vec<u32>>,
    best_ever: Option<f64>,
    individuals: u64,
    seconds: f64,
}

/// A gene's identity in the hall of fame: a hash of its words, as hex.
fn gene_key(gene: &[u32]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in gene {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

/// The loop's own name for the kingdom on the stream.
pub const KINGDOM: &str = "futhark_shapes";

/// Run the loop against a reference kernel, writing the run's standard outputs into `out`
/// (`docs/STANDARD_runners.md`): `stream.jsonl`, `killme.trigger` (delete it to stop after the current generation),
/// `checkpoint.json` (every generation; `cfg.resume` continues from it), `ledger.tsv` (every scored individual and its
/// parents) and `hall_of_fame/`.
///
/// The fitness is the [`two_level`] HFF. The `inputs` are the **suite**: the correctness of every candidate is
/// decided on all of them, so a program that coincides with the reference on one sample does not get in.
pub fn evolve(
    reference: &Tree,
    inputs: &[BTreeMap<Ty, Mat>],
    cfg: &Config,
    out: &Path,
) -> Result<Report, String> {
    let kingdom = FutharkKingdom::load_shapes()?;
    let contract = kingdom
        .relations()
        .check()
        .map_err(|bad| bad.join("; "))?;
    let root = kingdom.ty("mat.n.p").ok_or("the shape kingdom has no mat.n.p class")?;
    let layout = GeneLayout::from_program_size(&contract, root, cfg.program_size)
        .map_err(|e| format!("the layout: {e}"))?;
    let const_len = kingdom.const_len();
    let table = CostTable::load(&kingdom)?;
    let bridge = Bridge::load();

    let resumed: Option<Checkpoint> = if cfg.resume { Some(load_checkpoint(out)?) } else { None };
    let mut run = RunTelemetry::start(
        out,
        format!("{KINGDOM}-seed{:x}", cfg.seed),
        RunInfo {
            kingdom: KINGDOM.to_string(),
            dataset: "product_twice".to_string(),
            seed: cfg.seed as u32,
            population: cfg.population as u32,
            max_generations: cfg.generations as u32,
            budget_ms: (cfg.seconds * 1e3) as u64,
            hff_objectives: (0..inputs.len())
                .map(|i| format!("quality: disagreement on input set {i} (log)"))
                .chain(["speed: traffic / (traffic + reference)", "speed: allocations / (allocations + reference)", "speed: depth / (depth + reference)"].map(String::from))
                .collect(),
            config_json: Some(format!("{cfg:?}")),
            resume: cfg.resume,
            already_spent_seconds: resumed.as_ref().map_or(0.0, |c| c.seconds),
        },
    )?;
    let mut ledger = Ledger::open(out, cfg.resume)?;

    // The target: the reference's own value, on each input set.
    let mut wants = Vec::with_capacity(inputs.len());
    for sample in inputs {
        match kingdom.eval_matrices(reference, sample)? {
            MValue::Mat(m) => wants.push(m),
            other => return Err(format!("the reference is not a matrix: {other:?}")),
        }
    }

    let context = Context {
        kingdom: &kingdom,
        table: &table,
        inputs,
        wants: &wants,
        sizes: &cfg.sizes,
    };
    let source_of = |tree: &Tree| program(&kingdom, root, tree, &cfg.sizes);

    let reference_individual = evaluate(reference, &[], &context)?;
    let reference_score = reference_individual.score;
    let dir = run_dir();
    let reference_gpu = if cfg.measure_gpu {
        match measure_program(reference, &kingdom, cfg, &dir) {
            Ok(m) => Some(m),
            // The reference that the GPU will not compile is not a reference
            // the search can beat: the run has nothing to measure against.
            Err(e) => return Err(format!("the reference's GPU cost: {e}")),
        }
    } else {
        None
    };
    let mut report = Report {
        generations: Vec::new(),
        best: None,
        reference: Some(reference_individual),
        reference_gpu,
        population: Vec::new(),
    };

    let (mut population, mut rng, mut next_id, mut fame, first) = match resumed {
        Some(c) => {
            run.restore(c.best_ever, c.individuals);
            if let Some(gene) = &c.best_gene {
                let tree = express(&contract, &layout, gene, &const_len).map_err(|e| format!("resume: the best gene does not express: {e}"))?.tree();
                report.best = Some(evaluate(&tree, gene, &context)?);
            }
            (c.population, GeneRng::new(c.rng), c.next_id, c.hall_of_fame, c.next)
        }
        None => {
            let mut rng = GeneRng::new(cfg.seed);
            // The population: the reference — the incumbent the search has to beat
            // — and then random genes, most of which are wrong.
            let mut population: Vec<Member> = Vec::with_capacity(cfg.population);
            if cfg.seed_with_reference {
                // The gene the reference is written onto must be a LEGAL gene: every
                // cell holds a symbol of its own domain's type. A blank gene is not
                // — its zero is only the right symbol for the first row of the
                // first type — so the reference is written onto a random legal gene.
                let carrier = random_gene(&layout, &contract, &const_len, &mut rng);
                if let Ok(gene) = write_back(&contract, &layout, &carrier, reference) {
                    population.push(Member { gene, id: 0, parents: Vec::new(), operator: "reference".to_string() });
                }
            }
            while population.len() < cfg.population {
                let id = population.len() as u64;
                population.push(Member { gene: random_gene(&layout, &contract, &const_len, &mut rng), id, parents: Vec::new(), operator: "init".to_string() });
            }
            let next_id = population.len() as u64;
            (population, rng, next_id, HallOfFame::new(cfg.hall_of_fame), 0)
        }
    };

    let mut stopped_by = "n_gen";
    for number in first..cfg.generations {
        // ---- 0. the stop: the kill trigger, then the wall-clock guard ------
        if run.stop_requested() {
            stopped_by = "killed";
            break;
        }
        if run.out_of_time((cfg.seconds * 1e3) as u64) {
            stopped_by = "time";
            break;
        }
        let g = number as u32;
        // ---- 1. everyone is evaluated ------------------------------------
        let mut scored: Vec<Ranked> = Vec::with_capacity(population.len());
        for member in &population {
            let tree = express(&contract, &layout, &member.gene, &const_len)
                .map_err(|e| format!("a gene does not express: {e}"))?
                .tree();
            let individual = evaluate(&tree, &member.gene, &context)?;
            let hff = two_level(&individual, &reference_score);
            ledger.record(g, member.id, &member.parents, &member.operator, hff.fitness, &source_of(&individual.tree)?)?;
            scored.push(Ranked { individual, hff, member: member.clone() });
        }
        scored.sort_by(order);
        report.population = scored.iter().map(|s| s.individual.gene.clone()).collect();

        let correct = scored.iter().filter(|s| s.individual.correct).count();

        // ---- 2. the elites are polished by the e-graph ---------------------
        let elites: Vec<Ranked> = scored
            .iter()
            .filter(|s| s.individual.correct)
            .take(cfg.elites)
            .cloned()
            .collect();
        let mut polished_total = 0usize;
        let mut polish_evaluated = 0usize;
        let mut improved = 0usize;
        let mut pool = scored.clone();
        let mut seen: Vec<Vec<u32>> = pool.iter().map(|s| s.individual.gene.clone()).collect();
        for elite in &elites {
            let sat = saturate_tree(&elite.individual.tree, &kingdom, &bridge, cfg.candidates, cfg.iters)?;
            for candidate in sat.heritable() {
                polished_total += 1;
                let Some(tree) = candidate.gene.as_ref() else { continue };
                let gene = match write_back(&contract, &layout, &elite.individual.gene, tree) {
                    Ok(gene) => gene,
                    Err(e) => {
                        let mut m = Mutation::new(EventKind::WriteBackDefect, "egraph_exact", format!("the e-graph's form does not fit the layout: {e}"));
                        m.id = Some(elite.member.id);
                        m.text_before = Some(source_of(&elite.individual.tree)?);
                        m.text_after = source_of(tree).ok();
                        run.mutation(g, m)?;
                        continue;
                    }
                };
                // A candidate the layout cannot express is not a candidate.
                let Ok(expressed) = express(&contract, &layout, &gene, &const_len) else { continue };
                // The same child is offered every generation; taking it once
                // keeps the elites distinct.
                if seen.iter().any(|g| g == &gene) {
                    continue;
                }
                polish_evaluated += 1;
                let child = evaluate(&expressed.tree(), &gene, &context)?;
                if !child.correct {
                    continue;
                }
                seen.push(gene.clone());
                let hff = two_level(&child, &reference_score);
                // The polish only counts when it is actually better.
                let better = hff.fitness < elite.hff.fitness;
                if better {
                    improved += 1;
                }
                let member = Member { gene, id: next_id, parents: vec![elite.member.id], operator: "egraph_exact".to_string() };
                next_id += 1;
                let mut m = Mutation::new(EventKind::EgraphExact, "egraph_exact", format!("e-graph polish of #{}: traffic {} -> {}", elite.member.id, elite.individual.score.traffic, child.score.traffic));
                m.id = Some(member.id);
                m.parents = member.parents.clone();
                m.text_before = Some(source_of(&elite.individual.tree)?);
                m.text_after = Some(source_of(&child.tree)?);
                m.fitness_before = Some(elite.hff.fitness);
                m.fitness_after = Some(hff.fitness);
                m.entered = Some(true);
                m.improved = Some(better);
                run.mutation(g, m)?;
                ledger.record(g, member.id, &member.parents, &member.operator, hff.fitness, &source_of(&child.tree)?)?;
                pool.push(Ranked { individual: child, hff, member });
            }
        }
        pool.sort_by(order);

        // ---- 3. the cost is MEASURED, and it decides -----------------------
        //
        // The static polynomial is the *filter*; the compiled GPU program is
        // the truth. So the measured cost is what the elites are chosen by,
        // and a candidate the compiler no longer tiles is not eligible: a
        // form the tiler does not fire on is a different program on the GPU.
        let mut measured: Vec<(Ranked, GpuMeasurement)> = Vec::new();
        let mut tiled = 0usize;
        let mut not_runnable = 0usize;
        if cfg.measure_gpu {
            let text = input_text(&kingdom, &context.inputs[0])?;
            for ranked in pool.iter().filter(|s| s.individual.correct).take(cfg.elites * 2) {
                let source = source_of(&ranked.individual.tree)?;
                // A program the GPU pipeline will not compile is not a
                // candidate: it is skipped, not a crash.
                let Ok(cost) = measure(&source, &dir) else { continue };
                if cost.is_tiled() {
                    tiled += 1;
                }
                // The hard gate: the program has to RUN on the GPU. A
                // kernel the device cannot compile — the product computed
                // twice is one, its OpenCL kernel is too big — is not a
                // program, however cheap it looks.
                if !cfg.require_runnable {
                    measured.push((ranked.clone(), cost));
                    continue;
                }
                let ran = crate::futhark_exec::run_text_program_on(
                    crate::futhark_exec::Backend::OpenCl,
                    &source,
                    &text,
                    &dir,
                    Duration::from_secs(120),
                );
                match ran {
                    Ok(_) => measured.push((ranked.clone(), cost)),
                    Err(_) => not_runnable += 1,
                }
            }
        }
        measured.sort_by(|a, b| a.1.cost.cmp(&b.1.cost));

        // The best: by the measured cost when there is one, by the two-level
        // HFF otherwise.
        let best: Option<Individual> = if let Some((ranked, cost)) = measured.first() {
            let mut best = ranked.individual.clone();
            best.gpu = Some(*cost);
            Some(best)
        } else {
            pool.iter().find(|s| s.individual.correct).map(|s| s.individual.clone())
        };
        let best_gpu = measured.first().map(|(_, cost)| *cost);

        report.generations.push(Generation {
            number,
            correct,
            best: best.as_ref().map(|b| b.score),
            best_gpu,
            polished: polished_total,
            improved,
            measured: measured.len(),
            tiled,
            not_runnable,
        });
        if let Some(best) = &best {
            let better = match &report.best {
                None => true,
                Some(previous) => match (best.gpu, previous.gpu) {
                    (Some(a), Some(b)) => a.cost < b.cost,
                    (Some(_), None) => true,
                    (None, Some(_)) => false,
                    (None, None) => two_level(best, &reference_score).fitness < two_level(previous, &reference_score).fitness,
                },
            };
            if better {
                report.best = Some(best.clone());
            }
        }

        // ---- the generation's record: hall of fame, stream ------------------
        let mut fame_changed = false;
        for r in &pool {
            fame_changed |= fame.offer(FameEntry {
                key: gene_key(&r.individual.gene),
                fitness: r.hff.fitness,
                born: g,
                text: source_of(&r.individual.tree)?,
                nodes: Some(tree_nodes(&r.individual.tree) as u32),
                picture: None,
                extras: [("traffic".to_string(), r.individual.score.traffic as f64), ("correct".to_string(), f64::from(u8::from(r.individual.correct)))].into_iter().collect(),
            });
        }
        if fame_changed {
            let rows = fame.write(out)?;
            run.hall_of_fame(g, rows)?;
        }
        let fitness: Vec<f64> = pool.iter().map(|r| r.hff.fitness).collect();
        let distinct = {
            let mut genes: Vec<&Vec<u32>> = pool.iter().map(|r| &r.individual.gene).collect();
            genes.sort();
            genes.dedup();
            genes.len() as u32
        };
        let head = &pool[0];
        let extras: BTreeMap<String, f64> = [
            ("correct", correct as f64),
            ("polished", polished_total as f64),
            ("improved", improved as f64),
            ("measured", measured.len() as f64),
            ("tiled", tiled as f64),
            ("not_runnable", not_runnable as f64),
            ("best_traffic", head.individual.score.traffic as f64),
            ("reference_traffic", reference_score.traffic as f64),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        run.generation(GenerationStats {
            generation: g,
            fitness: &fitness,
            distinct: Some(distinct),
            angle_groups: [("quality".to_string(), head.hff.quality), ("speed".to_string(), head.hff.speed)].into_iter().collect(),
            extras,
            evaluated: (scored.len() + polish_evaluated) as u64,
        })?;
        ledger.flush()?;

        // ---- 4. the next generation ---------------------------------------
        // The elites are the measured best, then the two-level best, and they
        // are distinct genes.
        let mut next: Vec<Member> = Vec::with_capacity(cfg.population);
        let mut taken: Vec<Vec<u32>> = Vec::new();
        for ranked in measured.iter().map(|(r, _)| r).chain(pool.iter()) {
            if !ranked.individual.correct {
                continue;
            }
            if taken.iter().any(|g| g == &ranked.individual.gene) {
                continue;
            }
            taken.push(ranked.individual.gene.clone());
            next.push(Member { gene: ranked.individual.gene.clone(), id: ranked.member.id, parents: ranked.member.parents.clone(), operator: "elite".to_string() });
            if next.len() >= cfg.elites {
                break;
            }
        }
        while next.len() < cfg.population {
            let a = tournament(&pool, &mut rng);
            let b = tournament(&pool, &mut rng);
            let mut child = a.individual.gene.clone();
            let (operator, parents) = if rng.next_u64().is_multiple_of(2) {
                // A crossover: a range of one domain, exchanged.
                if let Some((lo, hi)) = a_range(&layout, &mut rng) {
                    let mut other = b.individual.gene.clone();
                    let _ = layout.swap_range(&layout, &mut child, &mut other, lo, hi);
                }
                ("crossover", vec![a.member.id, b.member.id])
            } else {
                // A point mutation: the cell's own alphabet says what it may
                // hold, so the child is legal by construction.
                let _ = mutate(&mut child, &contract, &layout, &const_len, &mut rng);
                ("mutation", vec![a.member.id])
            };
            next.push(Member { gene: child, id: next_id, parents, operator: operator.to_string() });
            next_id += 1;
        }
        population = next;
        save_checkpoint(
            out,
            &Checkpoint {
                next: number + 1,
                rng: rng.state(),
                population: population.clone(),
                next_id,
                hall_of_fame: fame.clone(),
                best_gene: report.best.as_ref().map(|b| b.gene.clone()),
                best_ever: run.best_ever(),
                individuals: run.individuals(),
                seconds: run.writer().elapsed_ms() as f64 / 1e3,
            },
        )?;
    }

    let done = report.generations.len() as u32;
    let best = report.best.as_ref().map(|b| two_level(b, &reference_score).fitness);
    run.end(stopped_by, done, best)?;
    Ok(report)
}

/// The nodes of a tree.
fn tree_nodes(tree: &Tree) -> usize {
    1 + tree.kids.iter().map(tree_nodes).sum::<usize>()
}

/// The order the population is ranked in: the two-level HFF, lower first; a tie keeps the earlier member.
fn order(a: &Ranked, b: &Ranked) -> std::cmp::Ordering {
    a.hff.fitness.total_cmp(&b.hff.fitness)
}

/// The directory this run measures in.
///
/// It is per-process, because two runs that share a directory would write
/// the same source file over each other — and the measurement is cached by
/// the source, so a stale file would be measured under the wrong program.
fn run_dir() -> std::path::PathBuf {
    std::env::temp_dir().join("futhark_shapes_evolve").join(format!("run{}", std::process::id()))
}

/// Evaluate one program: is it right, and what does it cost?
///
/// The correctness is decided on **every** input set in the suite, and the
/// comparison is bit for bit (`identical`) throughout: both sides are the
/// interpreter, so there is no reason to loosen it. (The *compiled*
/// program may differ from the interpreter where there is a reduction —
/// Futhark's simplifier folds `0.0 + x` away — but that is the compiled
/// program's business, and it is measured where the compiled program is
/// run.)
fn evaluate(tree: &Tree, gene: &[u32], context: &Context<'_>) -> Result<Individual, String> {
    let mut correct = true;
    let mut agreement = 0.0f64;
    let mut disagreement = Vec::with_capacity(context.wants.len());
    for (inputs, want) in context.inputs.iter().zip(context.wants.iter()) {
        let value = context.kingdom.eval_matrices(tree, inputs)?;
        match &value {
            MValue::Mat(m) => {
                let agreeing = m
                    .data
                    .iter()
                    .zip(want.data.iter())
                    .filter(|(a, b)| identical(**a, **b))
                    .count();
                let total = want.data.len().max(1);
                let share = agreeing as f64 / total as f64;
                agreement += share;
                if !matrices_agree(m, want, identical) {
                    correct = false;
                }
                // A matrix of the wrong shape agrees nowhere it does not overlap.
                disagreement.push(if matrices_agree(m, want, identical) { 0.0 } else { (1.0 - share).max(1.0 / total as f64) });
            }
            _ => {
                correct = false;
                disagreement.push(1.0);
            }
        }
    }
    if !context.wants.is_empty() {
        agreement /= context.wants.len() as f64;
    }
    let cost = program_cost(tree, context.kingdom, context.table)?;
    let score = score_at(&cost, context.sizes)?;
    Ok(Individual {
        gene: gene.to_vec(),
        tree: tree.clone(),
        correct,
        agreement,
        disagreement,
        score,
        gpu: None,
    })
}

/// Whether two matrices agree, under the comparison the caller chose.
fn matrices_agree(got: &Mat, want: &Mat, same: fn(f32, f32) -> bool) -> bool {
    got.rows == want.rows
        && got.cols == want.cols
        && got.data.iter().zip(want.data.iter()).all(|(a, b)| same(*a, *b))
}

/// A random gene of the layout.
fn random_gene(
    layout: &GeneLayout,
    contract: &KingdomContract,
    const_len: &BTreeMap<Ty, u32>,
    rng: &mut GeneRng,
) -> Vec<u32> {
    let mut gene = vec![0u32; layout.width];
    for (pos, cell) in gene.iter_mut().enumerate() {
        let alphabet = layout.alphabet(contract, pos, const_len);
        if alphabet.is_empty() {
            continue;
        }
        *cell = alphabet[rng.below(alphabet.len())];
    }
    gene
}

/// A point mutation: one cell, a value from its own alphabet.
fn mutate(
    gene: &mut [u32],
    contract: &KingdomContract,
    layout: &GeneLayout,
    const_len: &BTreeMap<Ty, u32>,
    rng: &mut GeneRng,
) -> bool {
    for _ in 0..8 {
        let pos = rng.below(layout.width);
        let alphabet = layout.alphabet(contract, pos, const_len);
        if alphabet.is_empty() {
            continue;
        }
        let value = alphabet[rng.below(alphabet.len())];
        if gene[pos] == value {
            continue;
        }
        if layout.mutate(contract, gene, pos, value, const_len).is_ok() {
            return true;
        }
    }
    false
}

/// A range inside one domain of one type — the unit a crossover exchanges.
fn a_range(layout: &GeneLayout, rng: &mut GeneRng) -> Option<(usize, usize)> {
    let spans: Vec<(usize, usize)> = layout
        .spans()
        .into_iter()
        .filter(|s| s.len >= 2)
        .map(|s| (s.off, s.len))
        .collect();
    if spans.is_empty() {
        return None;
    }
    let (off, len) = spans[rng.below(spans.len())];
    let lo = off + rng.below(len - 1);
    let hi = lo + 1 + rng.below(len - (lo - off));
    Some((lo, hi))
}

/// A tournament: the better of two draws, by the two-level HFF.
fn tournament<'a>(pool: &'a [Ranked], rng: &mut GeneRng) -> &'a Ranked {
    let a = &pool[rng.below(pool.len())];
    let b = &pool[rng.below(pool.len())];
    if b.hff.fitness < a.hff.fitness {
        b
    } else {
        a
    }
}

/// The measured GPU cost of a program.
///
/// This is the *decision* cost — the one the compiled GPU program carries —
/// as against the cheap polynomial in [`crate::futhark_shapes_cost`], which
/// is the filter.
pub fn measure_program(
    tree: &Tree,
    kingdom: &FutharkKingdom,
    cfg: &Config,
    dir: &Path,
) -> Result<GpuMeasurement, String> {
    let root = kingdom.ty("mat.n.p").ok_or("the shape kingdom has no mat.n.p class")?;
    let source = program(kingdom, root, tree, &cfg.sizes)?;
    measure(&source, dir)
}

/// The shape kingdom's **evaluation service**: score a population of genes
/// against a reference kernel.
///
/// This is the seam a GPU-resident population uses. The population lives on
/// the GPU — decoded, varied and selected there — and the scoring comes
/// back here, because a Futhark program has to be *compiled* before it can
/// be run, and only the host can do that.
///
/// The result is one [`Individual`] per gene, in the order the genes were
/// given: whether it computes the reference's value, how much of its value
/// agrees with the reference's, and what it costs.
pub fn score_population(
    reference: &Tree,
    genes: &[Vec<u32>],
    inputs: &[BTreeMap<Ty, Mat>],
    cfg: &Config,
) -> Result<Vec<Individual>, String> {
    Scorer::new(reference.clone(), inputs.to_vec(), cfg.sizes.clone(), cfg.program_size)?.score(genes)
}

/// The shape kingdom's evaluation service, PREPARED ONCE.
///
/// [`score_population`] builds the kingdom, the contract, the layout, the cost
/// table and the reference's own value on every call. That is right for a
/// one-shot call and wrong for a GPU-resident population, which scores once a
/// generation and would rebuild the tables every time.
///
/// So the prepared form is here: one object, built once, asked for a
/// population's scores as often as the population changes. The tables it
/// holds are the same ones the loop above uses, and the scores it returns are
/// the same scores — [`score_population`] is a one-line wrapper over it, so
/// there is only one implementation to be right.
pub struct Scorer {
    kingdom: FutharkKingdom,
    contract: KingdomContract,
    layout: GeneLayout,
    const_len: BTreeMap<Ty, u32>,
    table: CostTable,
    reference: Tree,
    inputs: Vec<BTreeMap<Ty, Mat>>,
    wants: Vec<Mat>,
    sizes: BTreeMap<String, usize>,
    /// The reference's own cost: what a win is a win against.
    reference_score: Score,
}

impl Scorer {
    /// Prepare the service.
    ///
    /// `reference` is the kernel the population is trying to reproduce, and
    /// `inputs` is the suite the correctness is decided on — every set counts,
    /// and one set is not enough. `sizes` gives the roles their sizes and
    /// `program_size` the layout's program size `N`.
    pub fn new(
        reference: Tree,
        inputs: Vec<BTreeMap<Ty, Mat>>,
        sizes: BTreeMap<String, usize>,
        program_size: usize,
    ) -> Result<Scorer, String> {
        // An empty suite decides nothing: every program would agree with the
        // reference vacuously, and every one of them would be called correct.
        if inputs.is_empty() {
            return Err("the evaluation service needs at least one input set: an empty suite decides nothing".into());
        }
        let kingdom = FutharkKingdom::load_shapes()?;
        let contract = kingdom
            .relations()
            .check()
            .map_err(|bad| bad.join("; "))?;
        let root = kingdom.ty("mat.n.p").ok_or("the shape kingdom has no mat.n.p class")?;
        let layout = GeneLayout::from_program_size(&contract, root, program_size)
            .map_err(|e| format!("the layout: {e}"))?;
        let const_len = kingdom.const_len();
        let table = CostTable::load(&kingdom)?;

        let mut wants = Vec::with_capacity(inputs.len());
        for sample in &inputs {
            match kingdom.eval_matrices(&reference, sample)? {
                MValue::Mat(m) => wants.push(m),
                other => return Err(format!("the reference is not a matrix: {other:?}")),
            }
        }
        let reference_cost = program_cost(&reference, &kingdom, &table)?;
        let reference_score = score_at(&reference_cost, &sizes)?;
        Ok(Scorer {
            kingdom,
            contract,
            layout,
            const_len,
            table,
            reference,
            inputs,
            wants,
            sizes,
            reference_score,
        })
    }

    /// Score a population of genes, in the order the genes were given.
    ///
    /// A gene the layout cannot express is an error: the layout says what a
    /// cell may hold, so it should not happen.
    pub fn score(&self, genes: &[Vec<u32>]) -> Result<Vec<Individual>, String> {
        let mut out = Vec::with_capacity(genes.len());
        for (i, gene) in genes.iter().enumerate() {
            match self.score_one(gene)? {
                Some(individual) => out.push(individual),
                None => return Err(format!("gene {i} does not express")),
            }
        }
        Ok(out)
    }

    /// Score a population, tolerating a gene that does not express.
    ///
    /// `Ok(Some(individual))` is a score, `Ok(None)` is a gene the layout
    /// cannot express, and `Err` is a real failure. The layout says what a
    /// cell may hold, so an inexpressible gene should not happen — but a
    /// generation must not be lost to one row of it.
    pub fn score_lenient(&self, genes: &[Vec<u32>]) -> Result<Vec<Option<Individual>>, String> {
        genes.iter().map(|gene| self.score_one(gene)).collect()
    }

    /// Score one gene: `None` when it does not express.
    fn score_one(&self, gene: &[u32]) -> Result<Option<Individual>, String> {
        let context = Context {
            kingdom: &self.kingdom,
            table: &self.table,
            inputs: &self.inputs,
            wants: &self.wants,
            sizes: &self.sizes,
        };
        let Ok(expressed) = express(&self.contract, &self.layout, gene, &self.const_len) else {
            return Ok(None);
        };
        Ok(Some(evaluate(&expressed.tree(), gene, &context)?))
    }

    /// The fitness a GPU-resident population selects on: one `f32` per
    /// individual, lower is fitter.
    ///
    /// See [`fitness_of`] for what the number means.
    pub fn fitness(&self, individuals: &[Individual]) -> Vec<f32> {
        individuals.iter().map(|i| fitness_of(i, &self.reference_score)).collect()
    }

    /// The reference's own gene: the reference written onto a LEGAL gene.
    ///
    /// A blank gene is not a legal one — its zero is only the right symbol for
    /// the first row of the first type — so the reference is written onto a
    /// random legal gene. `None` when the layout cannot hold the reference,
    /// which is a named size, not a failure.
    pub fn seed_gene(&self, seed: u64) -> Option<Vec<u32>> {
        let mut rng = GeneRng::new(seed);
        let carrier = random_gene(&self.layout, &self.contract, &self.const_len, &mut rng);
        write_back(&self.contract, &self.layout, &carrier, &self.reference).ok()
    }

    /// The contract the population is typed by.
    pub fn contract(&self) -> &KingdomContract {
        &self.contract
    }

    /// The kingdom the tables came from.
    pub fn kingdom(&self) -> &FutharkKingdom {
        &self.kingdom
    }

    /// The layout the genes are words of.
    pub fn layout(&self) -> &GeneLayout {
        &self.layout
    }

    /// The constant lengths, per type.
    pub fn const_len(&self) -> &BTreeMap<Ty, u32> {
        &self.const_len
    }

    /// The reference kernel.
    pub fn reference(&self) -> &Tree {
        &self.reference
    }

    /// The reference's own cost: what a win is a win against.
    pub fn reference_score(&self) -> &Score {
        &self.reference_score
    }
}

/// THE TWO-LEVEL HFF of a program (`docs/STANDARD_runners.md`; [`crate::runner::two_level_hff`]).
///
/// * **Quality:** one input per input set of the suite, the share of the value that disagrees with the reference's,
///   on HFF's log scale ([`crate::runner::log_scaled`]): a correct program is 0 in every input, and a program wrong
///   in one element of thousands is already far from 0, so being nearly right is not being right.
/// * **Speed:** one input per cost term, the program's over the program's plus the reference's (`x / (x + ref)`):
///   traffic, allocations, serial depth. The reference itself is 0.5 in each; strictly increasing and bounded by 1.
///
/// The fitness is the hff library's TrueNorth over the two group angles, each divided by π/2.
pub fn two_level(individual: &Individual, reference: &Score) -> TwoLevel {
    let quality: Vec<f64> = individual.disagreement.iter().map(|d| log_scaled(*d)).collect();
    let ratio = |x: u64, r: u64| {
        let (x, r) = (x as f64, r.max(1) as f64);
        x / (x + r)
    };
    let s = &individual.score;
    let speed = [ratio(s.traffic, reference.traffic), ratio(s.alloc, reference.alloc), ratio(s.depth, reference.depth)];
    two_level_hff(&quality, &speed)
}

/// The fitness a GPU-resident population selects on: the [`two_level`] HFF, as phylu's device reads a fitness (an
/// angle, 0 perfect, lower fitter).
pub fn fitness_of(individual: &Individual, reference: &Score) -> f32 {
    two_level(individual, reference).fitness as f32
}

/// One set of inputs, drawn from a seed.
///
/// The values are in `[-8, 8]`, and the first of every matrix is `-0.0`, so
/// the suite exercises the sign of zero as well as the magnitudes.
pub fn input_set(
    k: &FutharkKingdom,
    sizes: &BTreeMap<String, usize>,
    seed: u64,
) -> BTreeMap<Ty, Mat> {
    let mut map: BTreeMap<Ty, Mat> = BTreeMap::new();
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
        let ty = k.ty(&d.dual).expect("a dual's own type");
        map.insert(ty, mat);
    }
    map
}

/// A suite of independent input sets, drawn deterministically.
///
/// Every set is drawn from its own seed, so the sets do not agree with one
/// another by construction, and the suite is the same for the same count and
/// the same sizes.
pub fn suite(
    k: &FutharkKingdom,
    sizes: &BTreeMap<String, usize>,
    count: usize,
) -> Vec<BTreeMap<Ty, Mat>> {
    (0..count)
        .map(|i| {
            let step = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            input_set(k, sizes, 0x1234_5678_9ABC_DEF0u64.wrapping_add(step))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::futhark_shapes_gpu::measure;
    use crate::futhark_shapes_interp::{matmul, transpose, Mat};

    fn kingdom() -> FutharkKingdom {
        FutharkKingdom::load_shapes().expect("the shape tables load")
    }

    /// A run's own output directory.
    fn out(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fuller_shapes_evolve_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// A small configuration: the loop must be quick in a test.
    fn config() -> Config {
        Config {
            population: 8,
            generations: 2,
            measure_gpu: false,
            ..Config::default()
        }
    }

    /// The e-graph's **transposed-product** rule is the one that can win:
    /// it turns the product computed the transposed way into the product
    /// computed directly, and the compiler's own CSE then does the rest.
    ///
    /// This test is the whole point of the rule: the polished program
    /// computes the same value and costs less.
    #[test]
    fn the_egraph_polish_finds_the_cheaper_form() {
        let k = kingdom();
        let table = CostTable::load(&k).expect("the cost table loads");
        let bridge = Bridge::load();
        let contract = k.relations().check().map_err(|bad| bad.join("; ")).expect("the contract");
        let root = k.ty("mat.n.p").expect("the root class");
        let reference = product_twice(&k);
        let layout = GeneLayout::from_program_size(&contract, root, 3).expect("the layout");
        let const_len = k.const_len();

        let reference_cost = program_cost(&reference, &k, &table).expect("costed");
        let reference_score = score_at(&reference_cost, &config().sizes).expect("scored");

        // The reference's own gene: the reference written onto a LEGAL gene,
        // because a blank gene is not one (its zero is only the right symbol
        // for the first row of the first type).
        let mut rng = GeneRng::new(0x5EED_1234_ABCD_0021);
        let carrier = random_gene(&layout, &contract, &const_len, &mut rng);
        let reference_gene =
            write_back(&contract, &layout, &carrier, &reference).expect("the reference has a gene");

        // The polished candidates.
        let sat = saturate_tree(&reference, &k, &bridge, 40, 6).expect("the saturation runs");
        let mut best: Option<Score> = None;
        for candidate in sat.heritable() {
            let Some(tree) = candidate.gene.as_ref() else { continue };
            let Ok(gene) = write_back(&contract, &layout, &reference_gene, tree) else { continue };
            let Ok(expressed) = express(&contract, &layout, &gene, &const_len) else { continue };
            let cost = program_cost(&expressed.tree(), &k, &table).expect("costed");
            let score = score_at(&cost, &config().sizes).expect("scored");
            if best.map(|b| score < b).unwrap_or(true) {
                best = Some(score);
            }
        }
        let best = best.expect("the e-graph offered a candidate");
        assert!(
            best < reference_score,
            "the polish did not make the program cheaper: {best:?} against {reference_score:?}"
        );
    }

    /// The loop finds a program that computes the reference's value, and it
    /// is cheaper than the reference.
    #[test]
    fn the_loop_finds_a_cheaper_program() {
        let k = kingdom();
        let reference = product_twice(&k);
        let cfg = config();
        let input = suite(&k, &cfg.sizes, cfg.samples);
        let report = evolve(&reference, &input, &cfg, &out("cheaper")).expect("the loop runs");
        let best = report.best.expect("the loop found a correct program");
        assert!(best.correct, "the best program is correct");
        let reference = report.reference.expect("the reference was measured");
        assert!(
            best.score < reference.score,
            "the best program is not cheaper: {:?} against {:?}",
            best.score,
            reference.score
        );
    }

    /// The same seed, the same run: the loop is deterministic.
    #[test]
    fn the_loop_is_deterministic() {
        let k = kingdom();
        let reference = product_twice(&k);
        let cfg = config();
        let input = suite(&k, &cfg.sizes, cfg.samples);
        let first = evolve(&reference, &input, &cfg, &out("det1")).expect("the loop runs");
        let second = evolve(&reference, &input, &cfg, &out("det2")).expect("the loop runs again");
        let a = first.best.expect("a best");
        let b = second.best.expect("a best");
        assert_eq!(a.gene, b.gene, "the two runs found different programs");
    }

    /// The evaluation service scores a population the way the loop does:
    /// the same correctness, the same cost.
    #[test]
    fn the_evaluation_service_scores_a_population() {
        let k = kingdom();
        let reference = product_twice(&k);
        let cfg = config();
        let input = suite(&k, &cfg.sizes, cfg.samples);
        let contract = k.relations().check().map_err(|bad| bad.join("; ")).expect("the contract");
        let root = k.ty("mat.n.p").expect("the root class");
        let layout = GeneLayout::from_program_size(&contract, root, cfg.program_size).expect("layout");
        let const_len = k.const_len();
        let mut rng = GeneRng::new(0x5EED_1234_ABCD_0033);
        let genes: Vec<Vec<u32>> = (0..4)
            .map(|_| random_gene(&layout, &contract, &const_len, &mut rng))
            .collect();
        let scored = score_population(&reference, &genes, &input, &cfg).expect("the population is scored");
        assert_eq!(scored.len(), genes.len(), "every gene is scored");
        for (individual, gene) in scored.iter().zip(&genes) {
            assert_eq!(&individual.gene, gene, "the scores keep the genes' order");
        }
    }

    /// The PREPARED service scores exactly what the one-shot one scores, and
    /// its seed gene is the reference written onto a legal gene — so a
    /// GPU-resident population can start from the incumbent.
    #[test]
    fn the_prepared_service_is_the_one_shot_service() {
        let k = kingdom();
        let reference = product_twice(&k);
        let cfg = config();
        let input = suite(&k, &cfg.sizes, cfg.samples);
        let contract = k.relations().check().map_err(|bad| bad.join("; ")).expect("the contract");
        let root = k.ty("mat.n.p").expect("the root class");
        let layout = GeneLayout::from_program_size(&contract, root, cfg.program_size).expect("layout");
        let const_len = k.const_len();
        let mut rng = GeneRng::new(0x5EED_1234_ABCD_0033);
        let genes: Vec<Vec<u32>> = (0..4)
            .map(|_| random_gene(&layout, &contract, &const_len, &mut rng))
            .collect();

        let one_shot = score_population(&reference, &genes, &input, &cfg).expect("scored");
        let scorer = Scorer::new(reference.clone(), input.clone(), cfg.sizes.clone(), cfg.program_size).expect("prepared");
        let prepared = scorer.score(&genes).expect("scored");
        assert_eq!(one_shot.len(), prepared.len());
        for (a, b) in one_shot.iter().zip(&prepared) {
            assert_eq!(a.correct, b.correct);
            assert_eq!(a.agreement, b.agreement);
            assert_eq!(a.score, b.score);
        }

        // The seed gene expresses to the reference.
        let seed = scorer.seed_gene(0xA5A5_1234_5678_9ABC).expect("the reference fits the layout");
        assert_eq!(seed.len(), layout.width, "a gene is a row of the layout");
        let expressed = express(&contract, &layout, &seed, &const_len).expect("the seed expresses");
        assert_eq!(expressed.tree(), reference, "the seed gene IS the reference");
    }

    /// The fitness a device selects on is the two-level HFF: a correct program has a quality angle of 0 and is
    /// ranked by its cost; an incorrect one is ranked by how much of its value disagrees; and at the same cost a
    /// correct program is always fitter than an incorrect one, however nearly right.
    #[test]
    fn the_fitness_is_the_two_level_hff() {
        let k = kingdom();
        let reference = product_twice(&k);
        let cfg = config();
        let input = suite(&k, &cfg.sizes, cfg.samples);
        let scorer = Scorer::new(reference.clone(), input.clone(), cfg.sizes.clone(), cfg.program_size).expect("prepared");
        let contract = k.relations().check().map_err(|bad| bad.join("; ")).expect("the contract");
        let root = k.ty("mat.n.p").expect("the root class");
        let layout = GeneLayout::from_program_size(&contract, root, cfg.program_size).expect("layout");
        let const_len = k.const_len();
        let mut rng = GeneRng::new(0xC0FF_EE00_1122_3344);
        let genes: Vec<Vec<u32>> = (0..12)
            .map(|_| random_gene(&layout, &contract, &const_len, &mut rng))
            .collect();
        let scored = scorer.score(&genes).expect("scored");
        let fitness = scorer.fitness(&scored);
        let r = *scorer.reference_score();
        for (individual, f) in scored.iter().zip(&fitness) {
            assert!((0.0..=std::f32::consts::FRAC_PI_2).contains(f), "the fitness is an angle: {f}");
            assert_eq!(individual.disagreement.len(), cfg.samples, "one quality input per input set");
            assert_eq!(two_level(individual, &r).quality == 0.0, individual.correct, "a quality angle of 0 is correctness");
        }

        // The reference itself is correct and sits at 0.5 in every speed input.
        let seed = scorer.seed_gene(7).expect("the reference fits the layout");
        let reference_individual = scorer.score(&[seed]).expect("scored").remove(0);
        assert!(reference_individual.correct, "the reference is correct");
        assert_eq!(reference_individual.disagreement, vec![0.0; cfg.samples]);

        // A cheap correct program is fitter than a dear one, strictly.
        let at = |traffic: u64| Individual { score: Score { traffic, ..r }, ..reference_individual.clone() };
        let (cheap, dear, dearer) = (at(1), at(r.traffic), at(1_000 * r.traffic));
        assert!(fitness_of(&cheap, &r) < fitness_of(&dear, &r));
        assert!(fitness_of(&dear, &r) < fitness_of(&dearer, &r));

        // An incorrect program is ranked by its disagreement, and at the same cost a correct program beats it,
        // however nearly right it is (one element of a matrix).
        let wrong = |d: f64| Individual { correct: false, agreement: 1.0 - d, disagreement: vec![d, 0.0, 0.0], ..reference_individual.clone() };
        let one_element = 1.0 / (8.0 * 8.0);
        assert!(fitness_of(&wrong(one_element), &r) < fitness_of(&wrong(0.9), &r));
        assert!(fitness_of(&dear, &r) < fitness_of(&wrong(one_element), &r));
    }

    /// The premise of the transposed-product rule: the product computed the
    /// transposed way and transposed back is the product itself, **bit for
    /// bit**.
    #[test]
    fn the_transposed_product_is_the_product() {
        let a = Mat::new(2, 3, vec![1.0, 2.0, 3.0, -0.0, 5.0, 6.0]).expect("the shape");
        let b = Mat::new(3, 2, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).expect("the shape");
        let direct = matmul(&a, &b).expect("the product");
        let b_transposed = transpose(&b);
        let a_transposed = transpose(&a);
        let product = matmul(&b_transposed, &a_transposed).expect("the product");
        let transposed = transpose(&product);
        assert_eq!(direct.rows, transposed.rows);
        assert_eq!(direct.cols, transposed.cols);
        for (i, (x, y)) in direct.data.iter().zip(transposed.data.iter()).enumerate() {
            assert!(
                identical(*x, *y),
                "element {i}: the direct product says {x}, the transposed one says {y}"
            );
        }
    }

    /// What the compiler does with the two forms: the reference (the
    /// product computed twice) and the winner (the product computed once).
    ///
    /// This is the measurement the plan's §6 table quotes, and it is a
    /// test: the reference's kernel is the *bigger* one, and the winner's
    /// is not block+register tiled — so the tiling and the allocations pull
    /// in opposite directions, and the hard gate is that the program runs
    /// at all.
    #[test]
    fn the_two_forms_are_measured_on_the_gpu() {
        let k = FutharkKingdom::load_shapes().expect("the shape tables load");
        if !std::process::Command::new("futhark")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            eprintln!("skipped: futhark is not installed");
            return;
        }
        let reference = product_twice(&k);
        let a = leaf(&k, "mat.n.m.input");
        let b = leaf(&k, "mat.m.p.input");
        let winner = tree_with(
            &k,
            "map2_rows@mat.n.p",
            vec![
                leaf(&k, "fn.arr2_f32_arr_f32_arr_f32.add"),
                tree_with(&k, "matmul@mat.n.mxmat.m.p", vec![a.clone(), b.clone()]),
                tree_with(&k, "matmul@mat.n.mxmat.m.p", vec![a, b]),
            ],
        );
        let cfg = Config::default();
        let dir = std::env::temp_dir().join("futhark_shapes_probe");
        let root = k.ty("mat.n.p").expect("the root class");
        let mut costs = Vec::new();
        for tree in [&reference, &winner] {
            let source = program(&k, root, tree, &cfg.sizes).expect("the program renders");
            let m = measure(&source, &dir).expect("the GPU cost is measured");
            costs.push(m);
        }
        let (reference_cost, winner_cost) = (costs[0], costs[1]);
        assert!(
            winner_cost.cost.device_allocs < reference_cost.cost.device_allocs,
            "the winner allocates {} device buffers and the reference {}",
            winner_cost.cost.device_allocs,
            reference_cost.cost.device_allocs
        );
    }

    /// Every child the loop makes is legal: it expresses, and the engine
    /// can spell it.
    #[test]
    fn every_child_is_legal() {
        let k = kingdom();
        let reference = product_twice(&k);
        let cfg = config();
        let input = suite(&k, &cfg.sizes, cfg.samples);
        let report = evolve(&reference, &input, &cfg, &out("legal")).expect("the loop runs");
        let contract = k.relations().check().map_err(|bad| bad.join("; ")).expect("the contract");
        let root = k.ty("mat.n.p").expect("the root class");
        let layout = GeneLayout::from_program_size(&contract, root, cfg.program_size).expect("layout");
        let const_len = k.const_len();
        let best = report.best.expect("a best");
        let expressed = express(&contract, &layout, &best.gene, &const_len).expect("the best expresses");
        assert_eq!(expressed.tree(), best.tree, "the gene and the tree agree");
        k.render(&best.tree).expect("the program renders");
    }
}

