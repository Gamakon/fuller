// Brainfuck GP bloat study — Rust example.
//
// Task: "increment" — read one byte, output that byte + 1 (wrapping).
// The ground-truth is `,+.` (3 ops).
//
// Arms:
//   OFF: plain GP, no simplification.
//   ON:  same GP, but each offspring is passed through bf_simplify before
//        fitness evaluation. The simplified form replaces the offspring if
//        it is shorter AND passes the fitness check.
//
// Evolved program set: 10 seeds × 2 arms × (pop=30, gen=34) ≈ 1 020 evals.
//
// Measures (written to JSONL ledger):
//   - Final op-count distribution per arm.
//   - Solve rate (correctness across 10 test inputs) per arm.
//
// Run:
//   cargo run --release --example bf_bloat_study --no-default-features
//
// (Note: this example is in examples/bf_bloat/ but compiled as a single
//  binary; Cargo requires example files to be directly in examples/ or in
//  examples/<name>/main.rs. The build rule in Cargo.toml names this binary
//  "bf_bloat_study".)

use fuller::bf::eval::run_bf;
use fuller::bf::extract::bf_simplify;

use std::io::Write;

// ---------------------------------------------------------------------------
// GP parameters
// ---------------------------------------------------------------------------
const POP: usize = 30;
const GENS: usize = 34; // POP * GENS ≈ 1020 evals per seed
const SEEDS: usize = 10;
const MAX_PROG_LEN: usize = 24; // max BF ops in a random program
const MUTATION_RATE: f64 = 0.3;
const TOURNAMENT_K: usize = 4;

// Task: increment — output = input byte + 1 (wrapping u8)
const TEST_INPUTS: &[u8] = &[0, 1, 5, 10, 50, 100, 127, 200, 254, 255];

// ---------------------------------------------------------------------------
// Pseudo-RNG (xorshift64 — deterministic, fast)
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
// Fitness
// ---------------------------------------------------------------------------
/// Count how many of the TEST_INPUTS produce the correct increment output.
/// Programs that exceed the step limit on any input score 0 (they're non-halting).
/// Maximum score = TEST_INPUTS.len().
fn fitness(source: &str) -> usize {
    let mut score = 0;
    for &b in TEST_INPUTS {
        let expected = b.wrapping_add(1);
        match run_bf(source, &[b]) {
            fuller::bf::eval::TapeResult::Ok { output } if output == vec![expected] => {
                score += 1;
            }
            _ => {}
        }
    }
    score
}

fn is_solved(source: &str) -> bool {
    fitness(source) == TEST_INPUTS.len()
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
            6 if depth < 2 => { prog.push('['); depth += 1; }
            7 if depth > 0 => { prog.push(']'); depth -= 1; }
            6 | 7 => { prog.push('+'); } // fallback to safe op
            i => { prog.push(OPS_BASIC[i]); }
        }
    }
    // Close open brackets
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
            // Point change: replace one op
            let mut new_chars = chars.clone();
            let pos = rng_usize(rng, new_chars.len());
            let new_op_idx = rng_usize(rng, OPS_BASIC.len());
            new_chars[pos] = OPS_BASIC[new_op_idx];
            let candidate: String = new_chars.into_iter().collect();
            // Re-validate brackets
            fix_brackets(candidate, rng)
        }
        1 if chars.len() > 1 => {
            // Delete one op
            let pos = rng_usize(rng, chars.len());
            let mut new_chars = chars.clone();
            new_chars.remove(pos);
            fix_brackets(new_chars.into_iter().collect(), rng)
        }
        _ => {
            // Insert one op
            if chars.len() >= MAX_PROG_LEN { return prog.to_string(); }
            let pos = rng_usize(rng, chars.len() + 1);
            let new_op = OPS_BASIC[rng_usize(rng, OPS_BASIC.len())];
            let mut new_chars = chars.clone();
            new_chars.insert(pos, new_op);
            fix_brackets(new_chars.into_iter().collect(), rng)
        }
    }
}

/// Close any open brackets; remove dangling close brackets. Simple bracket fixer.
fn fix_brackets(prog: String, _rng: &mut u64) -> String {
    let mut out = String::new();
    let mut depth = 0i32;
    for c in prog.chars() {
        match c {
            '[' => { out.push('['); depth += 1; }
            ']' if depth > 0 => { out.push(']'); depth -= 1; }
            ']' => {} // skip dangling close bracket
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
// One GP run
// ---------------------------------------------------------------------------
struct RunResult {
    /// Op counts of final population.
    lengths: Vec<usize>,
    /// Number of solved individuals in final population.
    solve_count: usize,
    /// Best fitness achieved.
    best_fitness: usize,
}

fn gp_run(seed: u64, use_simplifier: bool) -> RunResult {
    let mut rng = seed;

    // Initialise population: (source, fitness)
    let mut pop: Vec<(String, usize)> = (0..POP)
        .map(|_| {
            let prog = random_program(&mut rng, MAX_PROG_LEN);
            let fit = fitness(&prog);
            (prog, fit)
        })
        .collect();

    for _gen in 0..GENS {
        let mut new_pop: Vec<(String, usize)> = Vec::with_capacity(POP);

        // Elitism: carry over the best individual
        let best = pop.iter().max_by_key(|(_, f)| f).unwrap().clone();
        new_pop.push(best);

        while new_pop.len() < POP {
            let parent = tournament_select(&pop, &mut rng, TOURNAMENT_K);
            let mut child = if rng_f64(&mut rng) < MUTATION_RATE {
                mutate(parent, &mut rng)
            } else {
                parent.to_string()
            };

            // Simplifier arm: pass through bf_simplify
            if use_simplifier {
                if let Ok(simplified) = bf_simplify(&child) {
                    if simplified.changed {
                        // Keep simplified form if it doesn't lose fitness
                        let orig_fit = fitness(&child);
                        let simp_fit = fitness(&simplified.source);
                        if simp_fit >= orig_fit {
                            child = simplified.source;
                        }
                    }
                }
            }

            let fit = fitness(&child);
            new_pop.push((child, fit));
        }

        pop = new_pop;
    }

    let lengths: Vec<usize> = pop
        .iter()
        .map(|(p, _)| p.chars().filter(|c| "+-<>.,[]".contains(*c)).count())
        .collect();
    let solve_count = pop.iter().filter(|(p, _)| is_solved(p)).count();
    let best_fitness = pop.iter().map(|(_, f)| *f).max().unwrap_or(0);

    RunResult { lengths, solve_count, best_fitness }
}

// ---------------------------------------------------------------------------
// Main: run both arms, write ledger
// ---------------------------------------------------------------------------
fn main() {
    let out_dir = std::path::Path::new(
        env!("CARGO_MANIFEST_DIR")
    ).join("examples").join("bf_bloat");
    std::fs::create_dir_all(&out_dir).expect("create output dir");
    let ledger_path = out_dir.join("results.jsonl");
    let mut ledger = std::fs::File::create(&ledger_path).expect("create ledger");

    println!("BF Bloat Study: task=increment, pop={POP}, gens={GENS}, seeds={SEEDS}");
    println!("Ledger: {}", ledger_path.display());
    println!();

    let mut all_off_lengths: Vec<usize> = Vec::new();
    let mut all_on_lengths: Vec<usize> = Vec::new();
    let mut off_solve_total = 0usize;
    let mut on_solve_total = 0usize;
    let total_individuals = SEEDS * POP;

    for seed in 0u64..SEEDS as u64 {
        for &use_simplifier in &[false, true] {
            let arm = if use_simplifier { "ON" } else { "OFF" };
            // Different initial seeds per arm so they're independent
            let run_seed = seed * 1_000_003 + if use_simplifier { 999_983 } else { 1 };
            let r = gp_run(run_seed, use_simplifier);

            let len_mean: f64 = r.lengths.iter().sum::<usize>() as f64 / r.lengths.len() as f64;
            let len_max = r.lengths.iter().copied().max().unwrap_or(0);
            let len_median = {
                let mut sorted = r.lengths.clone();
                sorted.sort_unstable();
                sorted[sorted.len() / 2]
            };

            println!(
                "seed={seed:2} arm={arm}: solve={}/{POP}  len mean={:.1} median={len_median} max={len_max}  best_fit={}/{}",
                r.solve_count, len_mean, r.best_fitness, TEST_INPUTS.len()
            );

            // Write JSONL record
            writeln!(
                ledger,
                "{{\"seed\":{seed},\"arm\":\"{arm}\",\"solve_count\":{},\"pop\":{POP},\
                \"len_mean\":{:.2},\"len_median\":{len_median},\"len_max\":{len_max},\
                \"best_fitness\":{},\"task\":\"increment\"}}",
                r.solve_count, len_mean, r.best_fitness
            ).expect("write ledger");

            if use_simplifier {
                all_on_lengths.extend(r.lengths.iter().copied());
                on_solve_total += r.solve_count;
            } else {
                all_off_lengths.extend(r.lengths.iter().copied());
                off_solve_total += r.solve_count;
            }
        }
    }

    // Summary
    println!();
    println!("=== SUMMARY ===");

    let summarize = |name: &str, lengths: &[usize], solves: usize| {
        let mean = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
        let mut sorted = lengths.to_vec();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2];
        let max = sorted.last().copied().unwrap_or(0);
        let solve_pct = solves as f64 / total_individuals as f64 * 100.0;
        println!(
            "{name}: len mean={mean:.1} median={median} max={max}  \
             solve_count={solves}/{total_individuals} ({solve_pct:.1}%)"
        );
    };

    summarize("OFF (no simplifier)", &all_off_lengths, off_solve_total);
    summarize("ON  (with simplifier)", &all_on_lengths, on_solve_total);

    // Write summary record to ledger
    {
        let off_mean = all_off_lengths.iter().sum::<usize>() as f64 / all_off_lengths.len() as f64;
        let on_mean = all_on_lengths.iter().sum::<usize>() as f64 / all_on_lengths.len() as f64;
        let mut sorted_off = all_off_lengths.clone();
        sorted_off.sort_unstable();
        let mut sorted_on = all_on_lengths.clone();
        sorted_on.sort_unstable();
        writeln!(
            ledger,
            "{{\"summary\":true,\"off_len_mean\":{:.2},\"off_len_median\":{},\"off_len_max\":{},\
            \"on_len_mean\":{:.2},\"on_len_median\":{},\"on_len_max\":{},\
            \"off_solve_total\":{off_solve_total},\"on_solve_total\":{on_solve_total},\
            \"total_individuals\":{total_individuals}}}",
            off_mean,
            sorted_off[sorted_off.len() / 2],
            sorted_off.last().copied().unwrap_or(0),
            on_mean,
            sorted_on[sorted_on.len() / 2],
            sorted_on.last().copied().unwrap_or(0),
        ).expect("write summary");
    }

    println!();
    println!("Results written to {}", ledger_path.display());
}
