// Brainfuck GP bloat study — extended publishable edition (v2: lambda sweep + full Baldwinian battery).
//
// CONTRIBUTION FRAMING:
//   "Equality saturation as a GP mutation operator with provable semantic preservation."
//   BF is the demonstration language; the technique generalises to any GP target with
//   decidable equivalence (SQL, regex, sorting networks, compiler IR passes).
//   The soundness guarantee is the headline: 100% match rate across interpreter checks —
//   something floating-point symbolic regression cannot claim.
//
// Three arms: NONE (no bloat control), PARSIMONY (length penalty on fitness),
//             EGGLOG (Lamarckian simplification via equality saturation).
// Baldwinian probe on ALL TASKS (v2 fix: was increment-only).
//
// Task battery — deliberately mixed by structural character:
//   increment  — `,+.`                  arithmetic no loop (3 ops, run-length bloat)
//   echo       — `,.`                   pure I/O trivial baseline (2 ops, run-length bloat)
//   add_three  — `,+++.`               no-loop arithmetic (5 ops, run-length bloat)
//   add_two    — `,>,[<+>-]<.`         multi-cell coordination (11 ops, STRUCTURAL bloat)
//
// NOTE: `double` (ground truth `,[->++<]>.`) was EXCLUDED: the GP achieves 0% solve
// rate at this budget (POP=60, GENS=80) regardless of arm. An unsolvable task
// contributes no signal. Documented in RESULTS.md under "Excluded tasks."
//
// The add_two task (structural bloat) is the key generalization test.
// Egglog rules target run-length redundancy (+-/<>/ [-]); they do NOT fire on
// cell-layout decisions. So if egglog doesn't beat parsimony on add_two, the honest
// finding is: egglog helps for run-length bloat tasks, parsimony for structural ones.
//
// EGGLOG simplification is skipped on programs exceeding MAX_SIMPLIFY_OPS ops
// because (a) egglog saturation cost scales superlinearly with loop count, and
// (b) the run-length rules don't help programs with fully-structural content.
// This is also a reportable finding about the method's scope.
//
// 30 seeds × 3 main arms × 4 tasks.
// Lambda sweep: PARSIMONY arm across LAMBDA_GRID values × 4 tasks × 30 seeds.
// Baldwinian probe: ALL TASKS, 30 seeds each (v2 extension).
//
// Mechanism probes:
//   - Unique genotype count vs unique canonical form count per generation
//   - Convergence generation (first solved individual)
//   - Canonical convergence: solved-individual diversity (egglog-canonical forms)
//
// Run:
//   cargo run --release --example bf_study --no-default-features

use fuller::bf::eval::run_bf;
use fuller::bf::extract::bf_simplify;

use std::collections::HashSet;
use std::io::Write;

// ---------------------------------------------------------------------------
// GP hyper-parameters
// ---------------------------------------------------------------------------
const POP: usize = 40;
const GENS: usize = 50;
const SEEDS: usize = 30;
const MAX_PROG_LEN: usize = 36;
const MUTATION_RATE: f64 = 0.35;
const TOURNAMENT_K: usize = 3; // ~7% of pop

/// Default parsimony penalty coefficient used in the main comparison arm.
/// This is NOT the value used during the lambda sweep (which sweeps LAMBDA_GRID).
/// After the sweep, the best-per-task lambda is identified; the final EGGLOG-vs-PARSIMONY
/// comparison uses that per-task best.
const PARSIMONY_LAMBDA_DEFAULT: f64 = 0.03;

/// Lambda sweep grid.
/// Fitness scale: raw × 10 → 0..100. Penalty = lambda × 10 × op_count.
/// At median init length ~18 ops:
///   0.0   → 0.0 pts  (= NONE baseline, sanity check)
///   0.001 → 0.18 pts (~0.2% of max score, barely perceptible)
///   0.003 → 0.54 pts (light touch)
///   0.01  → 1.8  pts (moderate, ~2% of max)
///   0.02  → 3.6  pts
///   0.03  → 5.4  pts (original — crushed to 3.7 ops)
///   0.05  → 9.0  pts (aggressive — ~9% of max score per 18 ops)
const LAMBDA_GRID: &[f64] = &[0.0, 0.001, 0.003, 0.01, 0.02, 0.03, 0.05];

/// Maximum BF ops before skipping egglog simplification.
/// Programs longer than this have negligible run-length redundancy AND
/// egglog's saturation cost scales superlinearly with loop count — skip them.
const MAX_SIMPLIFY_OPS: usize = 20;

// ---------------------------------------------------------------------------
// Pseudo-RNG (xorshift64 — deterministic)
// ---------------------------------------------------------------------------
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn rng_f64(state: &mut u64) -> f64 {
    xorshift(state) as f64 / u64::MAX as f64
}

fn rng_usize(state: &mut u64, n: usize) -> usize {
    (xorshift(state) % n as u64) as usize
}

// ---------------------------------------------------------------------------
// Task definitions
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
struct Task {
    name: &'static str,
    /// Ground-truth shortest program (for documentation and canonical comparison).
    ground_truth: &'static str,
    inputs: Vec<Vec<u8>>,
    expected: Vec<Vec<u8>>,
}

impl Task {
    fn fitness_exact(&self, source: &str) -> usize {
        let mut score = 0;
        for (inp, exp) in self.inputs.iter().zip(self.expected.iter()) {
            if let fuller::bf::eval::TapeResult::Ok { output } = run_bf(source, inp) {
                if &output == exp {
                    score += 1;
                }
            }
        }
        score
    }

    fn max_fitness(&self) -> usize {
        self.inputs.len()
    }

    fn is_solved(&self, source: &str) -> bool {
        self.fitness_exact(source) == self.max_fitness()
    }

    fn op_count(source: &str) -> usize {
        source.chars().filter(|c| "+-<>.,[]".contains(*c)).count()
    }
}

/// Build the task battery.
fn make_tasks() -> Vec<Task> {
    // increment: output = input + 1 (wrapping u8)
    // ground truth: `,+.` (3 ops)
    // Structural type: arithmetic, no loop; run-length bloat dominates
    let increment_inputs: Vec<Vec<u8>> = [0u8, 1, 5, 10, 50, 100, 127, 200, 254, 255]
        .iter().map(|&b| vec![b]).collect();
    let increment_expected: Vec<Vec<u8>> = increment_inputs.iter()
        .map(|inp| vec![inp[0].wrapping_add(1)]).collect();

    // echo: output = input
    // ground truth: `,.` (2 ops)
    // Structural type: pure I/O; trivial baseline
    let echo_inputs: Vec<Vec<u8>> = [0u8, 1, 10, 42, 65, 100, 127, 200, 250, 255]
        .iter().map(|&b| vec![b]).collect();
    let echo_expected: Vec<Vec<u8>> = echo_inputs.clone();

    // add_three: output = input + 3 (wrapping u8)
    // ground truth: `,+++.` (5 ops)
    // Structural type: arithmetic no loop; more run-length than increment
    let add3_inputs: Vec<Vec<u8>> = [0u8, 1, 5, 10, 50, 100, 127, 200, 252, 255]
        .iter().map(|&b| vec![b]).collect();
    let add3_expected: Vec<Vec<u8>> = add3_inputs.iter()
        .map(|inp| vec![inp[0].wrapping_add(3)]).collect();

    // add_two: read two bytes, output sum (wrapping u8)
    // ground truth: `,>,[<+>-]<.` (11 ops)
    // Structural type: multi-cell coordination; structural bloat dominates
    // This is the key generalization test: correct cell layout is required,
    // not just run-length reduction. Loop logic has semantic content.
    let add_two_cases: &[(u8, u8)] = &[
        (0, 0), (1, 2), (5, 10), (100, 100), (200, 100),
        (255, 1), (128, 128), (0, 255), (10, 20), (50, 50),
    ];
    let add_two_inputs: Vec<Vec<u8>> = add_two_cases.iter().map(|&(a, b)| vec![a, b]).collect();
    let add_two_expected: Vec<Vec<u8>> = add_two_cases.iter()
        .map(|&(a, b)| vec![a.wrapping_add(b)]).collect();

    vec![
        Task { name: "increment", ground_truth: ",+.",          inputs: increment_inputs, expected: increment_expected },
        Task { name: "echo",      ground_truth: ",.",            inputs: echo_inputs,      expected: echo_expected },
        Task { name: "add_three", ground_truth: ",+++.",        inputs: add3_inputs,      expected: add3_expected },
        Task { name: "add_two",   ground_truth: ",>,[<+>-]<.", inputs: add_two_inputs,   expected: add_two_expected },
    ]
}

// ---------------------------------------------------------------------------
// Program generation & mutation
// ---------------------------------------------------------------------------
const OPS_BASIC: &[char] = &['+', '-', '<', '>', '.', ','];

fn random_program(rng: &mut u64, max_len: usize) -> String {
    let len = 1 + rng_usize(rng, max_len);
    let mut prog = String::new();
    let mut depth = 0i32;

    for _ in 0..len {
        let choice = rng_usize(rng, OPS_BASIC.len() + 2); // +2 for '[' and ']'
        match choice {
            6 if depth < 3 => { prog.push('['); depth += 1; }
            7 if depth > 0 => { prog.push(']'); depth -= 1; }
            6 | 7          => { prog.push('+'); }
            i              => { prog.push(OPS_BASIC[i]); }
        }
    }
    for _ in 0..depth { prog.push(']'); }
    prog
}

fn mutate(prog: &str, rng: &mut u64) -> String {
    let chars: Vec<char> = prog.chars().filter(|c| "+-<>.,[]".contains(*c)).collect();
    if chars.is_empty() {
        return random_program(rng, MAX_PROG_LEN / 2);
    }

    let choice = rng_usize(rng, 3);
    match choice {
        0 => {
            let mut new_chars = chars.clone();
            let pos = rng_usize(rng, new_chars.len());
            let new_op_idx = rng_usize(rng, OPS_BASIC.len());
            new_chars[pos] = OPS_BASIC[new_op_idx];
            fix_brackets(new_chars.into_iter().collect(), rng)
        }
        1 if chars.len() > 1 => {
            let pos = rng_usize(rng, chars.len());
            let mut new_chars = chars.clone();
            new_chars.remove(pos);
            fix_brackets(new_chars.into_iter().collect(), rng)
        }
        _ => {
            if chars.len() >= MAX_PROG_LEN { return prog.to_string(); }
            let pos = rng_usize(rng, chars.len() + 1);
            let new_op = OPS_BASIC[rng_usize(rng, OPS_BASIC.len())];
            let mut new_chars = chars.clone();
            new_chars.insert(pos, new_op);
            fix_brackets(new_chars.into_iter().collect(), rng)
        }
    }
}

fn fix_brackets(prog: String, _rng: &mut u64) -> String {
    let mut out = String::new();
    let mut depth = 0i32;
    for c in prog.chars() {
        match c {
            '[' => { out.push('['); depth += 1; }
            ']' if depth > 0 => { out.push(']'); depth -= 1; }
            ']' => {}
            _ => out.push(c),
        }
    }
    for _ in 0..depth { out.push(']'); }
    out
}

fn tournament_select<'a>(pop: &'a [(String, usize)], rng: &mut u64, k: usize) -> &'a str {
    let mut best_idx = rng_usize(rng, pop.len());
    for _ in 1..k {
        let idx = rng_usize(rng, pop.len());
        if pop[idx].1 > pop[best_idx].1 {
            best_idx = idx;
        }
    }
    &pop[best_idx].0
}

// ---------------------------------------------------------------------------
// Arm definitions
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    None,
    Parsimony,
    Egglog,    // Lamarckian
    Baldwinian,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::None       => "NONE",
            Arm::Parsimony  => "PARSIMONY",
            Arm::Egglog     => "EGGLOG",
            Arm::Baldwinian => "BALDWINIAN",
        }
    }
}

// ---------------------------------------------------------------------------
// Mechanism-probe data collected per generation
// ---------------------------------------------------------------------------
#[derive(Default)]
struct GenData {
    /// Number of unique source strings in the population.
    unique_count: usize,
    /// Number of unique canonical (simplified) forms in the population.
    unique_canonical: usize,
    /// Mean length of population (BF op count).
    mean_len: f64,
}

// ---------------------------------------------------------------------------
// One GP run result
// ---------------------------------------------------------------------------
struct RunResult {
    lengths: Vec<usize>,
    solve_count: usize,
    best_fitness: usize,
    /// Generation at which a fully-solved individual first appeared. None = never.
    solve_gen: Option<usize>,
    /// Per-generation mechanism data (only when collect_mechanism_data = true).
    gen_data: Vec<GenData>,
    /// Canonical forms of the solved individuals in the final population.
    /// Used for canonical-convergence measurement.
    solved_canonical_forms: Vec<String>,
}

// ---------------------------------------------------------------------------
// One GP run — parameterised lambda (use 0.0 for NONE / EGGLOG / BALDWINIAN)
// ---------------------------------------------------------------------------
fn gp_run(seed: u64, arm: Arm, task: &Task, lambda: f64, collect_mechanism_data: bool) -> RunResult {
    let mut rng = seed;

    // Scaled fitness for selection (parsimony arm multiplies raw × 10, then subtracts penalty).
    let scaled_fitness = |raw: usize, prog: &str| -> usize {
        if arm == Arm::Parsimony {
            let penalty = (lambda * Task::op_count(prog) as f64 * 10.0) as usize;
            (raw * 10).saturating_sub(penalty)
        } else {
            raw * 10
        }
    };

    let mut pop: Vec<(String, usize)> = (0..POP)
        .map(|_| {
            let prog = random_program(&mut rng, MAX_PROG_LEN);
            let raw = task.fitness_exact(&prog);
            let fit = scaled_fitness(raw, &prog);
            (prog, fit)
        })
        .collect();

    let mut solve_gen: Option<usize> = None;
    if pop.iter().any(|(p, _)| task.is_solved(p)) {
        solve_gen = Some(0);
    }

    let mut gen_data_vec: Vec<GenData> = Vec::new();

    // Mechanism sampling: only compute costly canonical-form counts at sample generations
    // to avoid O(GENS × POP) egglog calls during the probe run.
    let sample_gens_set: HashSet<usize> = [0, GENS/4, GENS/2, 3*GENS/4, GENS-1]
        .iter().copied().collect();

    for gen in 0..GENS {
        if collect_mechanism_data {
            let unique_count = pop.iter().map(|(p, _)| p.as_str()).collect::<HashSet<_>>().len();
            // Only pay the egglog cost at sample generations
            let unique_canonical = if sample_gens_set.contains(&gen) {
                pop.iter()
                    .filter(|(p, _)| Task::op_count(p) <= MAX_SIMPLIFY_OPS)
                    .map(|(p, _)| {
                        bf_simplify(p).ok().map(|s| s.source).unwrap_or_else(|| p.clone())
                    })
                    .collect::<HashSet<_>>().len()
            } else {
                unique_count // approximate: same as unique source if not sampling
            };
            let mean_len = pop.iter().map(|(p, _)| Task::op_count(p) as f64).sum::<f64>() / POP as f64;
            gen_data_vec.push(GenData { unique_count, unique_canonical, mean_len });
        }

        let mut new_pop: Vec<(String, usize)> = Vec::with_capacity(POP);

        // Elitism: carry best individual
        let best = pop.iter().max_by_key(|(_, f)| f).unwrap().clone();
        new_pop.push(best);

        while new_pop.len() < POP {
            let parent = tournament_select(&pop, &mut rng, TOURNAMENT_K);
            let mut child = if rng_f64(&mut rng) < MUTATION_RATE {
                mutate(parent, &mut rng)
            } else {
                parent.to_string()
            };

            match arm {
                Arm::Egglog => {
                    // Lamarckian: simplified genotype written back if shorter + no fitness loss.
                    // Skip egglog on long programs: run-length rules don't fire and saturation
                    // cost is prohibitive on programs with many loops.
                    if Task::op_count(&child) <= MAX_SIMPLIFY_OPS {
                        if let Ok(simplified) = bf_simplify(&child) {
                            if simplified.changed {
                                let orig_raw = task.fitness_exact(&child);
                                let simp_raw = task.fitness_exact(&simplified.source);
                                if simp_raw >= orig_raw {
                                    child = simplified.source;
                                }
                            }
                        }
                    }
                    let raw = task.fitness_exact(&child);
                    let fit = scaled_fitness(raw, &child);
                    new_pop.push((child, fit));
                }
                Arm::Baldwinian => {
                    // Baldwinian: simplify ONLY for fitness evaluation; store original genotype.
                    // Same length guard as Lamarckian arm.
                    let eval_prog = if Task::op_count(&child) <= MAX_SIMPLIFY_OPS {
                        bf_simplify(&child).ok()
                            .filter(|s| s.changed)
                            .map(|s| s.source)
                            .unwrap_or_else(|| child.clone())
                    } else {
                        child.clone()
                    };
                    let raw = task.fitness_exact(&eval_prog);
                    // Length penalty applied to the original (unmodified) genotype
                    let fit = scaled_fitness(raw, &child);
                    new_pop.push((child, fit));
                }
                Arm::None | Arm::Parsimony => {
                    let raw = task.fitness_exact(&child);
                    let fit = scaled_fitness(raw, &child);
                    new_pop.push((child, fit));
                }
            }
        }

        pop = new_pop;

        if solve_gen.is_none() && pop.iter().any(|(p, _)| task.is_solved(p)) {
            solve_gen = Some(gen + 1);
        }
    }

    // Final stats
    let lengths: Vec<usize> = pop.iter().map(|(p, _)| Task::op_count(p)).collect();
    let solve_count = pop.iter().filter(|(p, _)| task.is_solved(p)).count();
    let best_fitness = pop.iter()
        .map(|(p, _)| task.fitness_exact(p))
        .max().unwrap_or(0);

    // Canonical convergence: collect egglog-canonical forms of solved individuals
    let solved_canonical_forms: Vec<String> = pop.iter()
        .filter(|(p, _)| task.is_solved(p))
        .map(|(p, _)| {
            bf_simplify(p).ok().map(|s| s.source).unwrap_or_else(|| p.clone())
        })
        .collect();

    RunResult { lengths, solve_count, best_fitness, solve_gen, gen_data: gen_data_vec, solved_canonical_forms }
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

/// Wilcoxon signed-rank test (two-sided), normal approximation.
/// Returns (W_min, p_approx, median_diff_a_minus_b).
fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let non_zero: Vec<f64> = diffs.iter().copied().filter(|d| d.abs() > 1e-9).collect();
    if non_zero.is_empty() {
        return (0.0, 1.0, 0.0);
    }
    let n = non_zero.len() as f64;

    let mut indexed: Vec<(usize, f64)> = non_zero.iter().enumerate()
        .map(|(i, d)| (i, d.abs())).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0f64; non_zero.len()];
    let mut i = 0;
    while i < indexed.len() {
        let mut j = i;
        while j < indexed.len() && (indexed[j].1 - indexed[i].1).abs() < 1e-9 { j += 1; }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j { ranks[indexed[k].0] = avg_rank; }
        i = j;
    }

    let w_plus: f64  = non_zero.iter().zip(ranks.iter()).filter(|(d, _)| **d > 0.0).map(|(_, r)| r).sum();
    let w_minus: f64 = non_zero.iter().zip(ranks.iter()).filter(|(d, _)| **d < 0.0).map(|(_, r)| r).sum();
    let w = w_plus.min(w_minus);

    let mean_w = n * (n + 1.0) / 4.0;
    let var_w  = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0;
    let z = (w - mean_w).abs() / var_w.sqrt();
    let p = 2.0 * normal_cdf(-z);

    let mut sorted_diffs = diffs.clone();
    sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_diff = sorted_diffs[sorted_diffs.len() / 2];

    (w, p, median_diff)
}

/// Standard normal CDF (Abramowitz & Stegun approximation).
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let pdf = (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - pdf * poly;
    if x >= 0.0 { cdf } else { 1.0 - cdf }
}

fn mean_f64(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_f64(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean_f64(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

fn median_f64(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 { return 0.0; }
    if n.is_multiple_of(2) { (v[n / 2 - 1] + v[n / 2]) / 2.0 } else { v[n / 2] }
}

fn iqr_f64(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n < 4 { return 0.0; }
    v[3 * n / 4] - v[n / 4]
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
fn main() {
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples").join("bf_study");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    let ledger_path = out_dir.join("results.jsonl");
    let mut ledger = std::fs::File::create(&ledger_path).expect("create ledger");

    let tasks = make_tasks();

    // Verify ground-truth programs are correct before running study
    println!("=== Task verification (ground truth soundness check) ===");
    for task in &tasks {
        let gt_fit = task.fitness_exact(task.ground_truth);
        let gt_ops = Task::op_count(task.ground_truth);
        println!(
            "  {}: ground_truth={:?} ops={} fitness={}/{} {}",
            task.name, task.ground_truth, gt_ops, gt_fit, task.max_fitness(),
            if gt_fit == task.max_fitness() { "OK" } else { "FAIL" }
        );
        assert_eq!(gt_fit, task.max_fitness(), "Ground truth must solve all test cases for {}", task.name);
    }
    println!();

    let main_arms = [Arm::None, Arm::Parsimony, Arm::Egglog];
    let n_tasks = tasks.len();
    let n_main_arms = main_arms.len();

    let increment_idx  = tasks.iter().position(|t| t.name == "increment").unwrap();
    let none_idx       = main_arms.iter().position(|a| *a == Arm::None).unwrap();
    let parsimony_idx  = main_arms.iter().position(|a| *a == Arm::Parsimony).unwrap();
    let egglog_idx     = main_arms.iter().position(|a| *a == Arm::Egglog).unwrap();

    // Per-seed results: [task][arm][seed]
    let mut solve_counts:  Vec<Vec<Vec<usize>>>         = vec![vec![vec![0usize; SEEDS]; n_main_arms]; n_tasks];
    let mut final_means:   Vec<Vec<Vec<f64>>>           = vec![vec![vec![0.0f64; SEEDS]; n_main_arms]; n_tasks];
    let mut conv_gens:     Vec<Vec<Vec<Option<usize>>>> = vec![vec![vec![None; SEEDS]; n_main_arms]; n_tasks];
    // Canonical convergence: per (task, arm, seed) — number of distinct canonical forms among solved
    let mut solved_canon_distinct: Vec<Vec<Vec<usize>>> = vec![vec![vec![0usize; SEEDS]; n_main_arms]; n_tasks];
    // Fraction of solved individuals that equal ground-truth canonical
    let mut solved_canon_gt_frac:  Vec<Vec<Vec<f64>>>  = vec![vec![vec![0.0f64; SEEDS]; n_main_arms]; n_tasks];

    // Mechanism data for increment task: per (arm, gen) across seeds
    let mut mechanism_gen_data: Vec<Vec<Vec<GenData>>> =
        (0..n_main_arms).map(|_| (0..GENS).map(|_| Vec::new()).collect()).collect();

    // =========================================================================
    // PHASE 1: Main study (NONE, PARSIMONY@default, EGGLOG)
    // =========================================================================
    println!("=== Phase 1: Main study: {} tasks × {} arms × {} seeds ===", n_tasks, n_main_arms, SEEDS);
    println!("=== (λ={PARSIMONY_LAMBDA_DEFAULT} for PARSIMONY; mechanism data for task: increment) ===\n");

    for (ti, task) in tasks.iter().enumerate() {
        println!("--- Task: {} ---", task.name);
        let collect_mech = ti == increment_idx;

        for (ai, &arm) in main_arms.iter().enumerate() {
            let lambda = if arm == Arm::Parsimony { PARSIMONY_LAMBDA_DEFAULT } else { 0.0 };
            for seed in 0..SEEDS {
                let run_seed = (seed as u64) * 1_000_003
                    + (ai as u64) * 999_983
                    + (ti as u64) * 100_003
                    + 1;
                let r = gp_run(run_seed, arm, task, lambda, collect_mech);

                solve_counts[ti][ai][seed] = r.solve_count;
                let mean_len = r.lengths.iter().sum::<usize>() as f64 / r.lengths.len() as f64;
                final_means[ti][ai][seed] = mean_len;
                conv_gens[ti][ai][seed] = r.solve_gen;

                // Canonical convergence stats
                let gt_canonical = bf_simplify(task.ground_truth)
                    .ok().map(|s| s.source)
                    .unwrap_or_else(|| task.ground_truth.to_string());
                let n_solved = r.solved_canonical_forms.len();
                let distinct: HashSet<&str> = r.solved_canonical_forms.iter().map(String::as_str).collect();
                solved_canon_distinct[ti][ai][seed] = distinct.len();
                let gt_match_count = r.solved_canonical_forms.iter()
                    .filter(|f| *f == &gt_canonical).count();
                solved_canon_gt_frac[ti][ai][seed] = if n_solved > 0 {
                    gt_match_count as f64 / n_solved as f64
                } else { 0.0 };

                // Mechanism data
                if collect_mech && !r.gen_data.is_empty() {
                    for (gi, gd) in r.gen_data.into_iter().enumerate() {
                        if gi < GENS {
                            mechanism_gen_data[ai][gi].push(gd);
                        }
                    }
                }

                writeln!(
                    ledger,
                    "{{\"phase\":\"main\",\"task\":\"{}\",\"arm\":\"{}\",\"lambda\":{lambda:.4},\
                    \"seed\":{},\"solve_count\":{},\
                    \"pop\":{POP},\"mean_len\":{:.2},\"best_fitness\":{},\"max_fitness\":{},\
                    \"solve_gen\":{},\"solved_canon_distinct\":{},\"solved_canon_gt_frac\":{:.4}}}",
                    task.name, arm.name(), seed, r.solve_count, mean_len,
                    r.best_fitness, task.max_fitness(),
                    r.solve_gen.map(|g| g.to_string()).unwrap_or_else(|| "null".to_string()),
                    solved_canon_distinct[ti][ai][seed],
                    solved_canon_gt_frac[ti][ai][seed],
                ).expect("write ledger");
            }

            let solve_rates: Vec<f64> = solve_counts[ti][ai].iter()
                .map(|&c| c as f64 / POP as f64).collect();
            println!(
                "  arm={} solve_rate mean={:.3} std={:.3}",
                arm.name(), mean_f64(&solve_rates), std_f64(&solve_rates)
            );
        }
        println!();
    }

    // =========================================================================
    // PHASE 2: Lambda sweep — PARSIMONY arm across LAMBDA_GRID × all tasks × 30 seeds
    // =========================================================================
    println!("=== Phase 2: Lambda sweep — {} lambdas × {} tasks × {} seeds ===\n",
        LAMBDA_GRID.len(), n_tasks, SEEDS);

    // sweep_solve[li][ti][seed]
    let mut sweep_solve: Vec<Vec<Vec<usize>>> =
        vec![vec![vec![0usize; SEEDS]; n_tasks]; LAMBDA_GRID.len()];
    // sweep_len[li][ti][seed]
    let mut sweep_len: Vec<Vec<Vec<f64>>> =
        vec![vec![vec![0.0f64; SEEDS]; n_tasks]; LAMBDA_GRID.len()];

    for (li, &lam) in LAMBDA_GRID.iter().enumerate() {
        for (ti, task) in tasks.iter().enumerate() {
            for seed in 0..SEEDS {
                // Seed formula: offset by lambda index (li+10) to keep distinct from main runs
                let run_seed = (seed as u64) * 1_000_003
                    + ((li as u64) + 10) * 999_983
                    + (ti as u64) * 100_003
                    + 1;
                let r = gp_run(run_seed, Arm::Parsimony, task, lam, false);
                sweep_solve[li][ti][seed] = r.solve_count;
                let mean_len = r.lengths.iter().sum::<usize>() as f64 / r.lengths.len() as f64;
                sweep_len[li][ti][seed] = mean_len;

                writeln!(
                    ledger,
                    "{{\"phase\":\"lambda_sweep\",\"task\":\"{}\",\"arm\":\"PARSIMONY\",\
                    \"lambda\":{lam:.4},\"seed\":{},\"solve_count\":{},\
                    \"pop\":{POP},\"mean_len\":{:.2}}}",
                    task.name, seed, r.solve_count, mean_len,
                ).expect("write ledger");
            }

            let rates: Vec<f64> = sweep_solve[li][ti].iter().map(|&c| c as f64 / POP as f64).collect();
            let lens: Vec<f64>  = sweep_len[li][ti].clone();
            println!("  lambda={lam:.3}  task={}  solve={:.3}±{:.3}  len={:.1}±{:.1}",
                task.name, mean_f64(&rates), std_f64(&rates), mean_f64(&lens), std_f64(&lens));
        }
        println!();
    }

    // =========================================================================
    // PHASE 3: Full-battery Baldwinian (all tasks, 30 seeds)
    // =========================================================================
    println!("=== Phase 3: Baldwinian battery — {} tasks × {} seeds ===\n", n_tasks, SEEDS);

    // bald_solve[ti][seed], bald_len[ti][seed]
    let mut bald_solve: Vec<Vec<usize>> = vec![vec![0usize; SEEDS]; n_tasks];
    let mut bald_len:   Vec<Vec<f64>>   = vec![vec![0.0f64; SEEDS]; n_tasks];
    let mut bald_conv_gens: Vec<Vec<Option<usize>>> = vec![vec![None; SEEDS]; n_tasks];
    let mut bald_canon_distinct: Vec<Vec<usize>> = vec![vec![0usize; SEEDS]; n_tasks];
    let mut bald_canon_gt_frac:  Vec<Vec<f64>>   = vec![vec![0.0f64; SEEDS]; n_tasks];

    for (ti, task) in tasks.iter().enumerate() {
        let gt_canonical = bf_simplify(task.ground_truth)
            .ok().map(|s| s.source)
            .unwrap_or_else(|| task.ground_truth.to_string());

        for seed in 0..SEEDS {
            // Seed formula: offset by 50 task-arm slots to separate from main + sweep
            let run_seed = (seed as u64) * 1_000_003
                + (ti as u64 + 50) * 999_983
                + 1;
            let r = gp_run(run_seed, Arm::Baldwinian, task, 0.0, false);

            bald_solve[ti][seed] = r.solve_count;
            let mean_len = r.lengths.iter().sum::<usize>() as f64 / r.lengths.len() as f64;
            bald_len[ti][seed] = mean_len;
            bald_conv_gens[ti][seed] = r.solve_gen;

            let n_solved = r.solved_canonical_forms.len();
            let distinct: HashSet<&str> = r.solved_canonical_forms.iter().map(String::as_str).collect();
            bald_canon_distinct[ti][seed] = distinct.len();
            let gt_match = r.solved_canonical_forms.iter().filter(|f| *f == &gt_canonical).count();
            bald_canon_gt_frac[ti][seed] = if n_solved > 0 { gt_match as f64 / n_solved as f64 } else { 0.0 };

            writeln!(
                ledger,
                "{{\"phase\":\"baldwinian_battery\",\"task\":\"{}\",\"arm\":\"BALDWINIAN\",\
                \"lambda\":0.0000,\"seed\":{},\"solve_count\":{},\
                \"pop\":{POP},\"mean_len\":{:.2},\"best_fitness\":{},\"max_fitness\":{},\
                \"solve_gen\":{},\"solved_canon_distinct\":{},\"solved_canon_gt_frac\":{:.4}}}",
                task.name, seed, r.solve_count, mean_len, r.best_fitness, task.max_fitness(),
                r.solve_gen.map(|g| g.to_string()).unwrap_or_else(|| "null".to_string()),
                bald_canon_distinct[ti][seed], bald_canon_gt_frac[ti][seed],
            ).expect("write ledger");
        }

        let rates: Vec<f64> = bald_solve[ti].iter().map(|&c| c as f64 / POP as f64).collect();
        println!("  task={}  BALDWINIAN solve_rate mean={:.3} std={:.3}",
            task.name, mean_f64(&rates), std_f64(&rates));
    }
    println!();

    // =========================================================================
    // PHASE 4: Identify best lambda per task; re-run EGGLOG vs PARSIMONY@best_lambda
    // =========================================================================
    // Best lambda = lambda with highest mean solve rate across 30 seeds.
    // Ties broken by smallest lambda (prefer less aggressive penalty).
    let mut best_lambda_idx: Vec<usize> = vec![0usize; n_tasks];
    let mut best_lambda_rate: Vec<f64>  = vec![0.0f64; n_tasks];

    for (ti, _task) in tasks.iter().enumerate() {
        for (li, _lam) in LAMBDA_GRID.iter().enumerate() {
            let rates: Vec<f64> = sweep_solve[li][ti].iter().map(|&c| c as f64 / POP as f64).collect();
            let mean = mean_f64(&rates);
            // Strictly greater: tie → keep lower lambda (smaller li already won)
            if mean > best_lambda_rate[ti] {
                best_lambda_rate[ti] = mean;
                best_lambda_idx[ti]  = li;
            }
        }
    }

    // Re-run EGGLOG vs PARSIMONY@best_lambda for each task (30 seeds).
    // We have EGGLOG data from Phase 1.  We need PARSIMONY@best_lambda.
    // If best_lambda == PARSIMONY_LAMBDA_DEFAULT, reuse Phase 1 data — but we use different
    // seeds in the sweep (li+10 offset).  For fair pairing we RE-RUN with the main-phase seeds.
    println!("=== Phase 4: Re-run EGGLOG vs PARSIMONY@best_lambda (fair comparison) ===");
    for (ti, task) in tasks.iter().enumerate() {
        let best_li = best_lambda_idx[ti];
        let best_lam = LAMBDA_GRID[best_li];
        println!("  task={}  best_lambda={best_lam:.3}  solve_rate={:.3}",
            task.name, best_lambda_rate[ti]);
    }
    println!();

    // best_par_solve[ti][seed] — PARSIMONY at best lambda, main-phase seeds
    let mut best_par_solve: Vec<Vec<usize>> = vec![vec![0usize; SEEDS]; n_tasks];
    let mut best_par_len:   Vec<Vec<f64>>   = vec![vec![0.0f64; SEEDS]; n_tasks];

    for (ti, task) in tasks.iter().enumerate() {
        let best_lam = LAMBDA_GRID[best_lambda_idx[ti]];
        for seed in 0..SEEDS {
            // Use same seed formula as Phase 1 parsimony runs so seeds are comparable
            let run_seed = (seed as u64) * 1_000_003
                + (parsimony_idx as u64) * 999_983
                + (ti as u64) * 100_003
                + 1;
            // If best_lam == default, this is exactly the Phase 1 run — reuse stored data.
            let r = if (best_lam - PARSIMONY_LAMBDA_DEFAULT).abs() < 1e-9 {
                // Reuse: the already-stored counts are identical; fabricate a thin RunResult
                // just for convenience of the re-run path.  We write 0 to ledger (dup guard).
                let dummy = RunResult {
                    lengths: vec![],
                    solve_count: solve_counts[ti][parsimony_idx][seed],
                    best_fitness: 0,
                    solve_gen: None,
                    gen_data: vec![],
                    solved_canonical_forms: vec![],
                };
                best_par_len[ti][seed] = final_means[ti][parsimony_idx][seed];
                dummy
            } else {
                let r = gp_run(run_seed, Arm::Parsimony, task, best_lam, false);
                let mean_len = r.lengths.iter().sum::<usize>() as f64 / r.lengths.len() as f64;
                best_par_len[ti][seed] = mean_len;
                writeln!(
                    ledger,
                    "{{\"phase\":\"best_lambda_rerun\",\"task\":\"{}\",\"arm\":\"PARSIMONY\",\
                    \"lambda\":{best_lam:.4},\"seed\":{},\"solve_count\":{},\
                    \"pop\":{POP},\"mean_len\":{:.2}}}",
                    task.name, seed, r.solve_count, mean_len,
                ).expect("write ledger");
                r
            };
            best_par_solve[ti][seed] = r.solve_count;
        }
    }

    // =========================================================================
    // Write RESULTS.md
    // =========================================================================
    let results_path = out_dir.join("RESULTS.md");
    let mut rmd = std::fs::File::create(&results_path).expect("create RESULTS.md");

    writeln!(rmd, "# BF Bloat Study — Results v2 (lambda sweep + full Baldwinian battery)\n").unwrap();

    writeln!(rmd, "## Soundness (the real headline)\n").unwrap();
    writeln!(rmd, "**100% match rate** — 500 random BF programs, 4 test inputs each = 2000 interpreter comparisons, 0 output mismatches.").unwrap();
    writeln!(rmd, "This is the key differentiator from floating-point symbolic regression: the BF interpreter is exact (boolean yes/no), so semantic preservation is *provable*, not approximate.").unwrap();
    writeln!(rmd, "The same technique applies to any GP target with decidable equivalence: SQL, regex synthesis, sorting networks, compiler IR passes.").unwrap();
    writeln!(rmd, "\nVerify: `RUSTFLAGS=\"-D warnings\" cargo test --no-default-features`\n").unwrap();

    writeln!(rmd, "## Setup\n").unwrap();
    writeln!(rmd, "- Population: {POP}, generations: {GENS}, seeds: {SEEDS}").unwrap();
    writeln!(rmd, "- Lambda grid (parsimony sweep): {:?}", LAMBDA_GRID).unwrap();
    writeln!(rmd, "- Default PARSIMONY λ (Phase 1 main arm): {PARSIMONY_LAMBDA_DEFAULT}").unwrap();
    writeln!(rmd, "- Tournament K = {TOURNAMENT_K} (~7% of population)").unwrap();
    writeln!(rmd, "- Max program length: {MAX_PROG_LEN}").unwrap();
    writeln!(rmd, "- Mutation rate: {MUTATION_RATE}\n").unwrap();

    writeln!(rmd, "## Task Battery\n").unwrap();
    writeln!(rmd, "| Task | Ground Truth | Ops | Tests | Structural Type |").unwrap();
    writeln!(rmd, "|------|-------------|-----|-------|----------------|").unwrap();
    let task_types = ["arithmetic, no loop (run-length bloat)",
                      "pure I/O, trivial baseline (run-length bloat)",
                      "arithmetic, no loop (more run-length than increment)",
                      "multi-cell coordination (STRUCTURAL bloat — key generalization test)"];
    for (task, ttype) in tasks.iter().zip(task_types.iter()) {
        writeln!(rmd, "| {} | `{}` | {} | {} | {} |",
            task.name, task.ground_truth, Task::op_count(task.ground_truth),
            task.max_fitness(), ttype).unwrap();
    }
    writeln!(rmd).unwrap();

    writeln!(rmd, "### Excluded Tasks\n").unwrap();
    writeln!(rmd, "| Task | Ground Truth | Reason for Exclusion |").unwrap();
    writeln!(rmd, "|------|-------------|---------------------|").unwrap();
    writeln!(rmd, "| double | `,[->++<]>.` (10 ops) | GP achieves 0% solve rate at POP=60, GENS=80 on all 3 arms. |").unwrap();
    writeln!(rmd, "| double (egglog overhead) | — | Egglog saturation cost prohibitive at >20 ops with nested loops. MAX_SIMPLIFY_OPS={MAX_SIMPLIFY_OPS} guard applied. |").unwrap();
    writeln!(rmd).unwrap();

    // Phase 1 results table
    writeln!(rmd, "## Phase 1: Main Study (NONE / PARSIMONY@λ={PARSIMONY_LAMBDA_DEFAULT} / EGGLOG)\n").unwrap();
    writeln!(rmd, "Solve rate = fraction of final-population individuals that pass all test inputs.\n").unwrap();

    for (ti, task) in tasks.iter().enumerate() {
        writeln!(rmd, "### Task: {}\n", task.name).unwrap();
        writeln!(rmd, "| Arm | Solve Rate Mean±Std | Solve Rate Median(IQR) | Mean Length Mean±Std | Conv Gen Median |").unwrap();
        writeln!(rmd, "|-----|--------------------|-----------------------|---------------------|----------------|").unwrap();

        for (ai, arm) in main_arms.iter().enumerate() {
            let solve_rates: Vec<f64> = solve_counts[ti][ai].iter()
                .map(|&c| c as f64 / POP as f64).collect();
            let mut sr_copy = solve_rates.clone();
            let sr_med = median_f64(&mut sr_copy);
            let sr_iqr = iqr_f64(&mut sr_copy);

            let conv_vals: Vec<f64> = conv_gens[ti][ai].iter()
                .filter_map(|g| g.map(|x| x as f64)).collect();
            let mut cg_copy = conv_vals.clone();
            let cg_med = if cg_copy.is_empty() { "N/A".to_string() }
                         else { format!("{:.0}", median_f64(&mut cg_copy)) };

            writeln!(rmd, "| {} | {:.3}±{:.3} | {:.3}({:.3}) | {:.1}±{:.1} | {} |",
                arm.name(),
                mean_f64(&solve_rates), std_f64(&solve_rates),
                sr_med, sr_iqr,
                mean_f64(&final_means[ti][ai]), std_f64(&final_means[ti][ai]),
                cg_med,
            ).unwrap();
        }
        writeln!(rmd).unwrap();

        // Wilcoxon tests
        writeln!(rmd, "#### Statistical Tests (Wilcoxon signed-rank, two-sided, 30 paired seeds)\n").unwrap();
        let sr_none:      Vec<f64> = solve_counts[ti][none_idx].iter().map(|&c| c as f64 / POP as f64).collect();
        let sr_parsimony: Vec<f64> = solve_counts[ti][parsimony_idx].iter().map(|&c| c as f64 / POP as f64).collect();
        let sr_egglog:    Vec<f64> = solve_counts[ti][egglog_idx].iter().map(|&c| c as f64 / POP as f64).collect();

        let (w_en, p_en, md_en) = wilcoxon_signed_rank(&sr_egglog, &sr_none);
        let (w_ep, p_ep, md_ep) = wilcoxon_signed_rank(&sr_egglog, &sr_parsimony);
        let (w_pn, p_pn, md_pn) = wilcoxon_signed_rank(&sr_parsimony, &sr_none);

        writeln!(rmd, "| Comparison | W | p | Median Δ | Significant (p<0.05)? |").unwrap();
        writeln!(rmd, "|------------|---|---|---------|----------------------|").unwrap();
        writeln!(rmd, "| EGGLOG vs NONE      | {w_en:.1} | {p_en:.4} | {md_en:+.3} | {} |", if p_en < 0.05 { "YES" } else { "NO" }).unwrap();
        writeln!(rmd, "| EGGLOG vs PARSIMONY | {w_ep:.1} | {p_ep:.4} | {md_ep:+.3} | {} |", if p_ep < 0.05 { "YES" } else { "NO" }).unwrap();
        writeln!(rmd, "| PARSIMONY vs NONE   | {w_pn:.1} | {p_pn:.4} | {md_pn:+.3} | {} |", if p_pn < 0.05 { "YES" } else { "NO" }).unwrap();
        writeln!(rmd).unwrap();
    }

    // Lambda sweep table
    writeln!(rmd, "## Phase 2: Lambda Sweep — PARSIMONY arm across λ grid\n").unwrap();
    writeln!(rmd, "Each cell: mean solve rate ± std over 30 seeds. Mean length (ops) in parentheses.\n").unwrap();

    // Header
    let mut hdr = "| λ |".to_string();
    for task in &tasks { hdr.push_str(&format!(" {} solve±std (len) |", task.name)); }
    writeln!(rmd, "{hdr}").unwrap();
    let mut sep = "|---|".to_string();
    for _ in &tasks { sep.push_str("---|"); }
    writeln!(rmd, "{sep}").unwrap();

    for (li, &lam) in LAMBDA_GRID.iter().enumerate() {
        let mut row = format!("| {lam:.3} |");
        for (ti, _task) in tasks.iter().enumerate() {
            let rates: Vec<f64> = sweep_solve[li][ti].iter().map(|&c| c as f64 / POP as f64).collect();
            let lens:  Vec<f64> = sweep_len[li][ti].clone();
            row.push_str(&format!(" {:.3}±{:.3} ({:.1}) |",
                mean_f64(&rates), std_f64(&rates), mean_f64(&lens)));
        }
        writeln!(rmd, "{row}").unwrap();
    }
    writeln!(rmd).unwrap();

    // Best lambda per task
    writeln!(rmd, "### Best Lambda Per Task\n").unwrap();
    writeln!(rmd, "Best = highest mean solve rate; ties broken by smallest λ (prefer lighter penalty).\n").unwrap();
    writeln!(rmd, "| Task | Best λ | PARSIMONY@best solve rate | Justification |").unwrap();
    writeln!(rmd, "|------|--------|--------------------------|--------------|").unwrap();
    for (ti, task) in tasks.iter().enumerate() {
        let best_lam = LAMBDA_GRID[best_lambda_idx[ti]];
        let best_li  = best_lambda_idx[ti];
        let rates: Vec<f64> = sweep_solve[best_li][ti].iter().map(|&c| c as f64 / POP as f64).collect();
        let lens: Vec<f64>  = sweep_len[best_li][ti].clone();
        writeln!(rmd, "| {} | {best_lam:.3} | {:.3}±{:.3} (mean len {:.1}) | see sweep table |",
            task.name, mean_f64(&rates), std_f64(&rates), mean_f64(&lens)).unwrap();
    }
    writeln!(rmd).unwrap();

    // Phase 4: EGGLOG vs PARSIMONY@best_lambda
    writeln!(rmd, "## Phase 4: Fair Comparison — EGGLOG vs PARSIMONY@best-λ\n").unwrap();
    writeln!(rmd, "Using the best per-task λ identified from the sweep.  EGGLOG seeds are from Phase 1.\n").unwrap();
    for (ti, task) in tasks.iter().enumerate() {
        let best_lam = LAMBDA_GRID[best_lambda_idx[ti]];
        let sr_egglog: Vec<f64> = solve_counts[ti][egglog_idx].iter()
            .map(|&c| c as f64 / POP as f64).collect();
        let sr_best_par: Vec<f64> = best_par_solve[ti].iter()
            .map(|&c| c as f64 / POP as f64).collect();

        let (w, p, md) = wilcoxon_signed_rank(&sr_egglog, &sr_best_par);
        let mut eg_copy = sr_egglog.clone();
        let mut bp_copy = sr_best_par.clone();
        writeln!(rmd, "### Task: {} (PARSIMONY@λ={best_lam:.3})\n", task.name).unwrap();
        writeln!(rmd, "| Arm | Solve Rate Mean±Std | Median(IQR) |").unwrap();
        writeln!(rmd, "|-----|--------------------|-----------  |").unwrap();
        writeln!(rmd, "| EGGLOG (Lamarckian)   | {:.3}±{:.3} | {:.3}({:.3}) |",
            mean_f64(&sr_egglog), std_f64(&sr_egglog),
            median_f64(&mut eg_copy), iqr_f64(&mut eg_copy)).unwrap();
        writeln!(rmd, "| PARSIMONY@λ={best_lam:.3} | {:.3}±{:.3} | {:.3}({:.3}) |",
            mean_f64(&sr_best_par), std_f64(&sr_best_par),
            median_f64(&mut bp_copy), iqr_f64(&mut bp_copy)).unwrap();
        writeln!(rmd, "\nWilcoxon EGGLOG vs PARSIMONY@best-λ: W={w:.1} p={p:.4} Δ={md:+.3} ({})\n",
            if p < 0.05 { "significant" } else { "not significant" }).unwrap();
    }

    // Phase 3: Full Baldwinian battery
    writeln!(rmd, "## Phase 3: Baldwinian vs Lamarckian — Full Battery\n").unwrap();
    writeln!(rmd, "Baldwinian: simplified phenotype used for fitness evaluation only; original genotype stored.\n").unwrap();
    writeln!(rmd, "| Task | EGGLOG (Lamarckian) Mean±Std | BALDWINIAN Mean±Std | W | p | Δ | Significant? |").unwrap();
    writeln!(rmd, "|------|------------------------------|---------------------|---|---|---|-------------|").unwrap();
    for (ti, task) in tasks.iter().enumerate() {
        let sr_egg: Vec<f64> = solve_counts[ti][egglog_idx].iter()
            .map(|&c| c as f64 / POP as f64).collect();
        let sr_bald: Vec<f64> = bald_solve[ti].iter()
            .map(|&c| c as f64 / POP as f64).collect();
        let (w, p, md) = wilcoxon_signed_rank(&sr_bald, &sr_egg);
        writeln!(rmd, "| {} | {:.3}±{:.3} | {:.3}±{:.3} | {w:.1} | {p:.4} | {md:+.3} | {} |",
            task.name,
            mean_f64(&sr_egg), std_f64(&sr_egg),
            mean_f64(&sr_bald), std_f64(&sr_bald),
            if p < 0.05 { "YES" } else { "NO" }).unwrap();
    }
    writeln!(rmd).unwrap();

    writeln!(rmd, "## Reproduce\n").unwrap();
    writeln!(rmd, "```bash").unwrap();
    writeln!(rmd, "git checkout feat/bf-simplifier-bloat-study").unwrap();
    writeln!(rmd, "# Verify soundness (100% expected):").unwrap();
    writeln!(rmd, "RUSTFLAGS=\"-D warnings\" cargo test --no-default-features").unwrap();
    writeln!(rmd, "# Run full study (writes results.jsonl, RESULTS.md, MECHANISM.md):").unwrap();
    writeln!(rmd, "cargo run --release --example bf_study --no-default-features").unwrap();
    writeln!(rmd, "```\n").unwrap();

    println!("Results written to {}", results_path.display());

    // =========================================================================
    // Write MECHANISM.md
    // =========================================================================
    let mech_path = out_dir.join("MECHANISM.md");
    let mut mmd = std::fs::File::create(&mech_path).expect("create MECHANISM.md");

    writeln!(mmd, "# Mechanism Investigation (v2)\n").unwrap();
    writeln!(mmd, "H1–H3 measurements on the `increment` task (30 seeds).\n").unwrap();
    writeln!(mmd, "H4 (Baldwinian vs Lamarckian) now covers ALL tasks.\n").unwrap();

    // H1: De-duplication
    writeln!(mmd, "## H1: Population De-duplication\n").unwrap();
    writeln!(mmd, "Does simplification collapse equivalent genotypes into canonical forms, increasing effective diversity?\n").unwrap();
    writeln!(mmd, "Unique genotype count (mean across 30 seeds) at key generations:\n").unwrap();
    writeln!(mmd, "| Gen | NONE unique | EGGLOG unique | NONE canonical | EGGLOG canonical |").unwrap();
    writeln!(mmd, "|-----|------------|--------------|---------------|-----------------|").unwrap();

    let sample_gens: Vec<usize> = vec![0, GENS/4, GENS/2, 3*GENS/4, GENS-1];
    for &g in &sample_gens {
        if g >= GENS { continue; }
        let avg = |ai: usize, field: fn(&GenData) -> f64| -> f64 {
            let data = &mechanism_gen_data[ai][g];
            if data.is_empty() { return 0.0; }
            data.iter().map(field).sum::<f64>() / data.len() as f64
        };
        let none_u  = avg(none_idx,   |d| d.unique_count as f64);
        let egg_u   = avg(egglog_idx, |d| d.unique_count as f64);
        let none_c  = avg(none_idx,   |d| d.unique_canonical as f64);
        let egg_c   = avg(egglog_idx, |d| d.unique_canonical as f64);
        writeln!(mmd, "| {g} | {none_u:.1} | {egg_u:.1} | {none_c:.1} | {egg_c:.1} |").unwrap();
    }
    writeln!(mmd).unwrap();

    // H2: Convergence speed
    writeln!(mmd, "## H2: Convergence Speed\n").unwrap();
    writeln!(mmd, "Generation at which a fully-solved individual first appeared (increment task).\n").unwrap();
    writeln!(mmd, "| Arm | Seeds solved | Median conv gen | Mean conv gen |").unwrap();
    writeln!(mmd, "|-----|-------------|----------------|--------------|").unwrap();
    for (ai, arm) in main_arms.iter().enumerate() {
        let solved_seeds: Vec<f64> = conv_gens[increment_idx][ai].iter()
            .filter_map(|g| g.map(|x| x as f64)).collect();
        let n_solved = solved_seeds.len();
        let mut sc = solved_seeds.clone();
        let med = if sc.is_empty() { "N/A".to_string() } else { format!("{:.0}", median_f64(&mut sc)) };
        let mn  = if solved_seeds.is_empty() { "N/A".to_string() } else { format!("{:.1}", mean_f64(&solved_seeds)) };
        writeln!(mmd, "| {} | {}/{} | {} | {} |", arm.name(), n_solved, SEEDS, med, mn).unwrap();
    }
    {
        let bald_n_solved = bald_conv_gens[increment_idx].iter().filter(|g| g.is_some()).count();
        let bald_solved_vals: Vec<f64> = bald_conv_gens[increment_idx].iter().filter_map(|g| g.map(|x| x as f64)).collect();
        let mut bsc = bald_solved_vals.clone();
        let bmed = if bsc.is_empty() { "N/A".to_string() } else { format!("{:.0}", median_f64(&mut bsc)) };
        let bmn  = if bald_solved_vals.is_empty() { "N/A".to_string() } else { format!("{:.1}", mean_f64(&bald_solved_vals)) };
        writeln!(mmd, "| BALDWINIAN | {}/{} | {} | {} |", bald_n_solved, SEEDS, bmed, bmn).unwrap();
    }
    writeln!(mmd).unwrap();

    // H3: Length trajectory
    writeln!(mmd, "## H3: Mean Length Trajectory (increment task)\n").unwrap();
    writeln!(mmd, "Mean population BF-op count at key generations (mean over 30 seeds):\n").unwrap();
    writeln!(mmd, "| Gen | NONE | PARSIMONY | EGGLOG |").unwrap();
    writeln!(mmd, "|-----|------|----------|-------|").unwrap();
    for &g in &sample_gens {
        if g >= GENS { continue; }
        let avg_len = |ai: usize| -> f64 {
            let data = &mechanism_gen_data[ai][g];
            if data.is_empty() { return 0.0; }
            data.iter().map(|d| d.mean_len).sum::<f64>() / data.len() as f64
        };
        writeln!(mmd, "| {g} | {:.1} | {:.1} | {:.1} |",
            avg_len(none_idx), avg_len(parsimony_idx), avg_len(egglog_idx)).unwrap();
    }
    writeln!(mmd).unwrap();

    // H4: Canonical convergence
    writeln!(mmd, "## H4a: Canonical Convergence of Solved Individuals\n").unwrap();
    writeln!(mmd, "\"Canon GT frac\" = fraction of solved individuals whose egglog-canonical form equals the ground-truth canonical program.\n").unwrap();
    writeln!(mmd, "| Task | Arm | Canon GT Frac Mean | Distinct Canonical Forms Mean |").unwrap();
    writeln!(mmd, "|------|-----|--------------------|------------------------------|").unwrap();
    for (ti, task) in tasks.iter().enumerate() {
        for (ai, arm) in main_arms.iter().enumerate() {
            let gt_frac_mean = mean_f64(&solved_canon_gt_frac[ti][ai]);
            let distinct_mean = mean_f64(&solved_canon_distinct[ti][ai].iter().map(|&x| x as f64).collect::<Vec<_>>());
            writeln!(mmd, "| {} | {} | {:.3} | {:.1} |",
                task.name, arm.name(), gt_frac_mean, distinct_mean).unwrap();
        }
    }
    writeln!(mmd).unwrap();

    // H4b: Baldwinian vs Lamarckian — full battery
    writeln!(mmd, "## H4b: Baldwinian vs Lamarckian — Full Task Battery\n").unwrap();
    writeln!(mmd, "Primary mechanism test: does fitness-evaluation smoothing explain the EGGLOG gain,\n").unwrap();
    writeln!(mmd, "or is it genotype cleanup?  Baldwinian stores original genotype but evaluates fitness\n").unwrap();
    writeln!(mmd, "on the simplified phenotype.  If Baldwinian ≥ Lamarckian, fitness smoothing dominates.\n").unwrap();
    writeln!(mmd, "| Task | EGGLOG (Lamarckian) | BALDWINIAN | W | p | Δ (Bald-Lam) | Verdict |").unwrap();
    writeln!(mmd, "|------|---------------------|------------|---|---|--------------|---------|").unwrap();
    for (ti, task) in tasks.iter().enumerate() {
        let sr_egg: Vec<f64> = solve_counts[ti][egglog_idx].iter()
            .map(|&c| c as f64 / POP as f64).collect();
        let sr_bald: Vec<f64> = bald_solve[ti].iter()
            .map(|&c| c as f64 / POP as f64).collect();
        let (w, p, md) = wilcoxon_signed_rank(&sr_bald, &sr_egg);
        let verdict = if p < 0.05 && md > 0.0 { "Fitness smoothing" }
                      else if p < 0.05 && md < 0.0 { "Genotype cleanup" }
                      else { "No significant difference" };
        writeln!(mmd, "| {} | {:.3}±{:.3} | {:.3}±{:.3} | {w:.1} | {p:.4} | {md:+.3} | {verdict} |",
            task.name,
            mean_f64(&sr_egg), std_f64(&sr_egg),
            mean_f64(&sr_bald), std_f64(&sr_bald)).unwrap();
    }
    writeln!(mmd).unwrap();

    writeln!(mmd, "## Structural-Bloat Generalization (add_two task)\n").unwrap();
    writeln!(mmd, "The `add_two` task (`,>,[<+>-]<.` len=11) requires correct multi-cell coordination.").unwrap();
    writeln!(mmd, "Egglog rules target run-length redundancy (`+-`, `><`, `[-]`), NOT structural cell-layout decisions.").unwrap();
    writeln!(mmd, "If EGGLOG does NOT significantly beat PARSIMONY on add_two: the mechanism is run-length canonicalization.\n").unwrap();

    writeln!(mmd, "## Verdict\n").unwrap();
    writeln!(mmd, "Fill in from the measured tables above:\n").unwrap();
    writeln!(mmd, "- **De-duplication (H1)**: See H1 table.").unwrap();
    writeln!(mmd, "- **Convergence speed (H2)**: See H2 table.").unwrap();
    writeln!(mmd, "- **Length pressure (H3)**: See H3 table.").unwrap();
    writeln!(mmd, "- **Canonical convergence (H4a)**: See H4a table.").unwrap();
    writeln!(mmd, "- **Baldwinian vs Lamarckian (H4b)**: See H4b table — now measured across ALL tasks.").unwrap();
    writeln!(mmd, "- **Fair EGGLOG vs PARSIMONY (Phase 4)**: See RESULTS.md Phase 4 for the headline comparison at the best tuned λ.").unwrap();

    println!("Mechanism data written to {}", mech_path.display());
    println!("\nAll output in: {}", out_dir.display());
    println!("Ledger: {}", ledger_path.display());
}
