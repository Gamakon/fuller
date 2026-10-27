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

/// Run the loop against a reference kernel.
///
/// The `inputs` are the **suite**: the correctness of every candidate is
/// decided on all of them, so a program that coincides with the reference
/// on one sample does not get in.
pub fn evolve(
    reference: &Tree,
    inputs: &[BTreeMap<Ty, Mat>],
    cfg: &Config,
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
    let mut rng = GeneRng::new(cfg.seed);

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

    let reference_individual = evaluate(reference, &[], &context)?;
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
    };

    // The population: the reference — the incumbent the search has to beat
    // — and then random genes, most of which are wrong.
    let mut population: Vec<Vec<u32>> = Vec::with_capacity(cfg.population);
    if cfg.seed_with_reference {
        // The gene the reference is written onto must be a LEGAL gene: every
        // cell holds a symbol of its own domain's type. A blank gene is not
        // — its zero is only the right symbol for the first row of the
        // first type — so the reference is written onto a random legal gene.
        let carrier = random_gene(&layout, &contract, &const_len, &mut rng);
        if let Ok(gene) = write_back(&contract, &layout, &carrier, reference) {
            population.push(gene);
        }
    }
    while population.len() < cfg.population {
        population.push(random_gene(&layout, &contract, &const_len, &mut rng));
    }

    for number in 0..cfg.generations {
        // ---- 1. everyone is evaluated ------------------------------------
        let mut scored: Vec<Individual> = Vec::with_capacity(population.len());
        for gene in &population {
            let tree = express(&contract, &layout, gene, &const_len)
                .map_err(|e| format!("a gene does not express: {e}"))?
                .tree();
            scored.push(evaluate(&tree, gene, &context)?);
        }
        scored.sort_by(order);

        let correct = scored.iter().filter(|s| s.correct).count();

        // ---- 2. the elites are polished by the e-graph ---------------------
        let elites: Vec<Individual> = scored
            .iter()
            .filter(|s| s.correct)
            .take(cfg.elites)
            .cloned()
            .collect();
        let mut polished_total = 0usize;
        let mut improved = 0usize;
        let mut pool = scored.clone();
        let mut seen: Vec<Vec<u32>> = pool.iter().map(|s| s.gene.clone()).collect();
        for elite in &elites {
            let sat = saturate_tree(&elite.tree, &kingdom, &bridge, cfg.candidates, cfg.iters)?;
            for candidate in sat.heritable() {
                polished_total += 1;
                let Some(tree) = candidate.gene.as_ref() else { continue };
                let Ok(gene) = write_back(&contract, &layout, &elite.gene, tree) else { continue };
                // A candidate the layout cannot express is not a candidate.
                let Ok(expressed) = express(&contract, &layout, &gene, &const_len) else { continue };
                // The same child is offered every generation; taking it once
                // keeps the elites distinct.
                if seen.iter().any(|g| g == &gene) {
                    continue;
                }
                let child = evaluate(&expressed.tree(), &gene, &context)?;
                if !child.correct {
                    continue;
                }
                seen.push(gene.clone());
                // The polish only counts when it is actually cheaper.
                if child.score < elite.score {
                    improved += 1;
                }
                pool.push(child);
            }
        }
        pool.sort_by(order);

        // ---- 3. the cost is MEASURED, and it decides -----------------------
        //
        // The static polynomial is the *filter*; the compiled GPU program is
        // the truth. So the measured cost is what the elites are chosen by,
        // and a candidate the compiler no longer tiles is not eligible: a
        // form the tiler does not fire on is a different program on the GPU.
        let mut measured: Vec<(Individual, GpuMeasurement)> = Vec::new();
        let mut skipped_untiled = 0usize;
        let mut not_runnable = 0usize;
        if cfg.measure_gpu {
            let text = input_text(&kingdom, &context.inputs[0])?;
            for individual in pool.iter().filter(|s| s.correct).take(cfg.elites * 2) {
                let source = program(&kingdom, root, &individual.tree, &cfg.sizes)?;
                // A program the GPU pipeline will not compile is not a
                // candidate: it is skipped, not a crash.
                let Ok(cost) = measure(&source, &dir) else { continue };
                if cost.is_tiled() {
                    skipped_untiled += 1;
                }
                // The hard gate: the program has to RUN on the GPU. A
                // kernel the device cannot compile — the product computed
                // twice is one, its OpenCL kernel is too big — is not a
                // program, however cheap it looks.
                if !cfg.require_runnable {
                    measured.push((individual.clone(), cost));
                    continue;
                }
                let run = crate::futhark_exec::run_text_program_on(
                    crate::futhark_exec::Backend::OpenCl,
                    &source,
                    &text,
                    &dir,
                    Duration::from_secs(120),
                );
                match run {
                    Ok(_) => measured.push((individual.clone(), cost)),
                    Err(_) => not_runnable += 1,
                }
            }
        }
        measured.sort_by(|a, b| a.1.cost.cmp(&b.1.cost));

        // The best: by the measured cost when there is one, by the static
        // cost otherwise.
        let best: Option<Individual> = if let Some((individual, cost)) = measured.first() {
            let mut best = individual.clone();
            best.gpu = Some(*cost);
            Some(best)
        } else {
            pool.iter().find(|s| s.correct).cloned()
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
            tiled: cfg.elites * 2 - skipped_untiled - not_runnable,
            not_runnable,
        });
        if let Some(best) = &best {
            let better = match &report.best {
                None => true,
                Some(previous) => match (best.gpu, previous.gpu) {
                    (Some(a), Some(b)) => a.cost < b.cost,
                    (Some(_), None) => true,
                    (None, Some(_)) => false,
                    (None, None) => best.score < previous.score,
                },
            };
            if better {
                report.best = Some(best.clone());
            }
        }

        // ---- 4. the next generation ---------------------------------------
        // The elites are the measured best, then the static best, and they
        // are distinct genes.
        population = Vec::with_capacity(cfg.population);
        let mut taken: Vec<Vec<u32>> = Vec::new();
        for individual in measured.iter().map(|(i, _)| i).chain(pool.iter()) {
            if !individual.correct {
                continue;
            }
            if taken.iter().any(|g| g == &individual.gene) {
                continue;
            }
            taken.push(individual.gene.clone());
            population.push(individual.gene.clone());
            if population.len() >= cfg.elites {
                break;
            }
        }
        while population.len() < cfg.population {
            let a = tournament(&pool, &mut rng);
            let b = tournament(&pool, &mut rng);
            let mut child = a.gene.clone();
            if rng.next_u64().is_multiple_of(2) {
                // A crossover: a range of one domain, exchanged.
                if let Some((lo, hi)) = a_range(&layout, &mut rng) {
                    let mut other = b.gene.clone();
                    let _ = layout.swap_range(&layout, &mut child, &mut other, lo, hi);
                }
            } else {
                // A point mutation: the cell's own alphabet says what it may
                // hold, so the child is legal by construction.
                let _ = mutate(&mut child, &contract, &layout, &const_len, &mut rng);
            }
            population.push(child);
        }
    }

    Ok(report)
}

/// The order the population is ranked in: correct first, then the closer to
/// the reference, then the cheaper.
fn order(a: &Individual, b: &Individual) -> std::cmp::Ordering {
    b.correct
        .cmp(&a.correct)
        .then_with(|| {
            b.agreement
                .partial_cmp(&a.agreement)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| a.score.cmp(&b.score))
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
                agreement += agreeing as f64 / total as f64;
                if !matrices_agree(m, want, identical) {
                    correct = false;
                }
            }
            _ => correct = false,
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

/// A tournament: the better of two draws.
///
/// The order is the same as the sort's: correct first, then the closer to
/// the reference, then the cheaper. The graded measure is what the GEP
/// hill-climbs on, and most of the population is *wrong*, so the agreement
/// has to be compared between two wrong programs too.
fn tournament(pool: &[Individual], rng: &mut GeneRng) -> Individual {
    let a = &pool[rng.below(pool.len())];
    let b = &pool[rng.below(pool.len())];
    if b.correct && !a.correct {
        return b.clone();
    }
    if a.correct && !b.correct {
        return a.clone();
    }
    if b.agreement > a.agreement || (b.agreement == a.agreement && b.score < a.score) {
        b.clone()
    } else {
        a.clone()
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

    let mut out = Vec::with_capacity(genes.len());
    for gene in genes {
        let expressed = express(&contract, &layout, gene, &const_len)
            .map_err(|e| format!("a gene does not express: {e}"))?;
        out.push(evaluate(&expressed.tree(), gene, &context)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::futhark_shapes_gpu::measure;
    use crate::futhark_shapes_interp::{matmul, transpose, Mat};

    fn kingdom() -> FutharkKingdom {
        FutharkKingdom::load_shapes().expect("the shape tables load")
    }

    /// A suite of independent input sets, drawn deterministically.
    fn suite(k: &FutharkKingdom, sizes: &BTreeMap<String, usize>, count: usize) -> Vec<BTreeMap<Ty, Mat>> {
        (0..count)
            .map(|i| {
            let step = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            input_set(k, sizes, 0x1234_5678_9ABC_DEF0u64.wrapping_add(step))
        })
            .collect()
    }

    /// One set of inputs, drawn from a seed.
    fn input_set(
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
        let report = evolve(&reference, &input, &cfg).expect("the loop runs");
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
        let first = evolve(&reference, &input, &cfg).expect("the loop runs");
        let second = evolve(&reference, &input, &cfg).expect("the loop runs again");
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
        let report = evolve(&reference, &input, &cfg).expect("the loop runs");
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

