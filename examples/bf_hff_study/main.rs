// bf_hff_study — HFF/validation-in-fitness study for Brainfuck GP.
//
// CONTRIBUTION FRAMING:
//   Validation-in-fitness via Hyperspherical Fitness Functions (HFF) as multi-objective
//   selection signal for discrete, decidable-equivalence GP (Brainfuck).
//   Shows the HFF/val pattern that beats overfit in SR also transfers to BF —
//   a memoriser that fools single-objective fitness is caught by HFF_VAL.
//
// Arms:
//   NONE        — raw err_train, single-objective baseline (lower=better)
//   HFF_TRAIN   — k=1 HFF wrapper (isolates projection effect from val signal)
//   HFF_VAL     — k=2: [err_train, err_val] via TrueNorth
//   HFF_EXTRAP  — k=3: [err_train, err_val, err_extrap] via TrueNorth
//
// Tasks:
//   increment  — `,+.`             (arithmetic no loop)
//   echo       — `,.`              (pure I/O baseline)
//   add_three  — `,+++.`          (more run-length than increment)
//   add_two    — `,>,[<+>-]<.`   (multi-cell coordination; structural bloat)
//
// Per task: TRAIN / VAL / EXTRAP input sets, generated deterministically from per-task seeds.
// 30 seeds × 4 arms × 4 tasks.
//
// Memoriser-attack test: a hand-constructed BF program that scores 10/10 on TRAIN
// and near-0/10 on VAL is injected; verifies HFF_VAL ranks it worse than ground truth.
//
// Run:
//   cargo run --release --example bf_hff_study --no-default-features

use fuller::bf::eval::run_bf;
use ndarray::Array1;

use std::io::Write;

// ---------------------------------------------------------------------------
// GP hyper-parameters (equal to bf_study for cross-study comparability)
// ---------------------------------------------------------------------------
const POP: usize = 40;
const GENS: usize = 50;
const SEEDS: usize = 30;
const MAX_PROG_LEN: usize = 36;
const MUTATION_RATE: f64 = 0.35;
const TOURNAMENT_K: usize = 3; // ~7% of pop

// Input set sizes per split.
const TRAIN_SIZE: usize = 10;
const VAL_SIZE: usize = 10;
const EXTRAP_SIZE: usize = 10;
const TRAIN_SIZE_MULTI: usize = 10;
const VAL_SIZE_MULTI: usize = 10;
const EXTRAP_SIZE_MULTI: usize = 10;

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
// HFF projection (TrueNorth) — LOWER is BETTER
//
// Input vec is already in [0,1] (error fractions); no normalisation needed.
// Perfect [0,...,0] → angle 0. Worst [1,...,1] → large angle.
// ---------------------------------------------------------------------------
fn hff_truenorth(errors: &[f64]) -> f64 {
    let arr = Array1::from_vec(errors.to_vec());
    let k = errors.len();
    hff_core::core_functions::calculate_single_hyperspherical_fitness_f64_with_method(
        &arr, k, false, None, "truenorth",
    )
}

// ---------------------------------------------------------------------------
// Arm definitions
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    None,
    HffTrain,
    HffVal,
    HffExtrap,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::None      => "NONE",
            Arm::HffTrain  => "HFF_TRAIN",
            Arm::HffVal    => "HFF_VAL",
            Arm::HffExtrap => "HFF_EXTRAP",
        }
    }
}

// ---------------------------------------------------------------------------
// Task (with three disjoint input sets)
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
struct Task {
    name: &'static str,
    ground_truth: &'static str,
    train_inputs:    Vec<Vec<u8>>,
    train_expected:  Vec<Vec<u8>>,
    val_inputs:      Vec<Vec<u8>>,
    val_expected:    Vec<Vec<u8>>,
    extrap_inputs:   Vec<Vec<u8>>,
    extrap_expected: Vec<Vec<u8>>,
    extrap_note: &'static str,
}

impl Task {
    fn acc(source: &str, inputs: &[Vec<u8>], expected: &[Vec<u8>]) -> f64 {
        let correct = inputs.iter().zip(expected.iter())
            .filter(|(inp, exp)| {
                matches!(run_bf(source, inp),
                    fuller::bf::eval::TapeResult::Ok { output } if &output == *exp)
            })
            .count();
        correct as f64 / inputs.len() as f64
    }

    fn train_acc(&self, src: &str) -> f64 { Self::acc(src, &self.train_inputs, &self.train_expected) }
    fn val_acc(&self,   src: &str) -> f64 { Self::acc(src, &self.val_inputs,   &self.val_expected) }
    fn extrap_acc(&self,src: &str) -> f64 { Self::acc(src, &self.extrap_inputs,&self.extrap_expected) }

    fn is_solved_oracle(&self, src: &str) -> bool {
        self.train_acc(src) >= 1.0 - 1e-9
            && self.val_acc(src) >= 1.0 - 1e-9
            && self.extrap_acc(src) >= 1.0 - 1e-9
    }

    fn is_solved_train(&self, src: &str) -> bool {
        self.train_acc(src) >= 1.0 - 1e-9
    }

    /// HFF fitness score for an arm (LOWER = BETTER).
    fn hff_score(&self, src: &str, arm: Arm) -> f64 {
        let et = 1.0 - self.train_acc(src);
        match arm {
            Arm::None     => et,
            Arm::HffTrain => hff_truenorth(&[et]),
            Arm::HffVal   => hff_truenorth(&[et, 1.0 - self.val_acc(src)]),
            Arm::HffExtrap => hff_truenorth(&[et, 1.0 - self.val_acc(src), 1.0 - self.extrap_acc(src)]),
        }
    }
}

// ---------------------------------------------------------------------------
// Task construction — deterministic from per-task seeds
// ---------------------------------------------------------------------------

/// Sample n distinct u8 values from [lo, hi] (inclusive) via Fisher-Yates.
fn sample_bytes(rng: &mut u64, n: usize, lo: u8, hi: u8) -> Vec<u8> {
    let range_size = (hi as usize) - (lo as usize) + 1;
    if range_size <= n {
        return (lo..=hi).collect();
    }
    let mut pool: Vec<u8> = (lo..=hi).collect();
    for i in 0..n {
        let j = i + rng_usize(rng, pool.len() - i);
        pool.swap(i, j);
    }
    pool[..n].to_vec()
}

fn make_tasks() -> Vec<Task> {
    // ---- increment ----
    let (inc_tr, inc_va, inc_ex) = {
        let mut rng: u64 = 0xBEEF_CAFE_1234_5678;
        let train = sample_bytes(&mut rng, TRAIN_SIZE, 0, 199);
        let train_set: std::collections::HashSet<u8> = train.iter().copied().collect();
        let mut vpool: Vec<u8> = (0u8..=199).filter(|b| !train_set.contains(b)).collect();
        let mut val = Vec::new();
        for i in 0..VAL_SIZE.min(vpool.len()) {
            let j = i + rng_usize(&mut rng, vpool.len() - i);
            vpool.swap(i, j);
            val.push(vpool[i]);
        }
        let extrap = sample_bytes(&mut rng, EXTRAP_SIZE, 200, 255);
        (train, val, extrap)
    };
    let increment = Task {
        name: "increment", ground_truth: ",+.",
        train_inputs:    inc_tr.iter().map(|&b| vec![b]).collect(),
        train_expected:  inc_tr.iter().map(|&b| vec![b.wrapping_add(1)]).collect(),
        val_inputs:      inc_va.iter().map(|&b| vec![b]).collect(),
        val_expected:    inc_va.iter().map(|&b| vec![b.wrapping_add(1)]).collect(),
        extrap_inputs:   inc_ex.iter().map(|&b| vec![b]).collect(),
        extrap_expected: inc_ex.iter().map(|&b| vec![b.wrapping_add(1)]).collect(),
        extrap_note: "Near-wrap regime [200..255]. All 256 inputs are semantically uniform for increment; extrap is a held-out edge-value chunk, not truly OOD.",
    };

    // ---- echo ----
    let (echo_tr, echo_va, echo_ex) = {
        let mut rng: u64 = 0xDEAD_BEEF_ABCD_EF01;
        let train = sample_bytes(&mut rng, TRAIN_SIZE, 0, 199);
        let train_set: std::collections::HashSet<u8> = train.iter().copied().collect();
        let mut vpool: Vec<u8> = (0u8..=199).filter(|b| !train_set.contains(b)).collect();
        let mut val = Vec::new();
        for i in 0..VAL_SIZE.min(vpool.len()) {
            let j = i + rng_usize(&mut rng, vpool.len() - i);
            vpool.swap(i, j);
            val.push(vpool[i]);
        }
        let extrap = sample_bytes(&mut rng, EXTRAP_SIZE, 200, 255);
        (train, val, extrap)
    };
    let echo = Task {
        name: "echo", ground_truth: ",.",
        train_inputs:    echo_tr.iter().map(|&b| vec![b]).collect(),
        train_expected:  echo_tr.iter().map(|&b| vec![b]).collect(),
        val_inputs:      echo_va.iter().map(|&b| vec![b]).collect(),
        val_expected:    echo_va.iter().map(|&b| vec![b]).collect(),
        extrap_inputs:   echo_ex.iter().map(|&b| vec![b]).collect(),
        extrap_expected: echo_ex.iter().map(|&b| vec![b]).collect(),
        extrap_note: "Held-out high-end bytes [200..255]. Echo is semantically uniform; extrap is a held-out split, not OOD.",
    };

    // ---- add_three ----
    let (a3_tr, a3_va, a3_ex) = {
        let mut rng: u64 = 0xFEED_FACE_0102_0304;
        let train = sample_bytes(&mut rng, TRAIN_SIZE, 0, 199);
        let train_set: std::collections::HashSet<u8> = train.iter().copied().collect();
        let mut vpool: Vec<u8> = (0u8..=199).filter(|b| !train_set.contains(b)).collect();
        let mut val = Vec::new();
        for i in 0..VAL_SIZE.min(vpool.len()) {
            let j = i + rng_usize(&mut rng, vpool.len() - i);
            vpool.swap(i, j);
            val.push(vpool[i]);
        }
        let extrap = sample_bytes(&mut rng, EXTRAP_SIZE, 200, 255);
        (train, val, extrap)
    };
    let add_three = Task {
        name: "add_three", ground_truth: ",+++.",
        train_inputs:    a3_tr.iter().map(|&b| vec![b]).collect(),
        train_expected:  a3_tr.iter().map(|&b| vec![b.wrapping_add(3)]).collect(),
        val_inputs:      a3_va.iter().map(|&b| vec![b]).collect(),
        val_expected:    a3_va.iter().map(|&b| vec![b.wrapping_add(3)]).collect(),
        extrap_inputs:   a3_ex.iter().map(|&b| vec![b]).collect(),
        extrap_expected: a3_ex.iter().map(|&b| vec![b.wrapping_add(3)]).collect(),
        extrap_note: "Near-wrap regime [200..255]; values where add-3 crosses the u8 wrap boundary.",
    };

    // ---- add_two ----
    let (a2_tr, a2_va, a2_ex) = {
        let mut rng: u64 = 0xCAFE_BABE_8899_AABB;
        let mut train_pairs: Vec<(u8, u8)> = Vec::new();
        while train_pairs.len() < TRAIN_SIZE_MULTI {
            let a = rng_usize(&mut rng, 128) as u8;
            let b = rng_usize(&mut rng, 128) as u8;
            if !train_pairs.contains(&(a, b)) { train_pairs.push((a, b)); }
        }
        let mut val_pairs: Vec<(u8, u8)> = Vec::new();
        while val_pairs.len() < VAL_SIZE_MULTI {
            let a = rng_usize(&mut rng, 128) as u8;
            let b = rng_usize(&mut rng, 128) as u8;
            if !train_pairs.contains(&(a, b)) && !val_pairs.contains(&(a, b)) {
                val_pairs.push((a, b));
            }
        }
        let mut extrap_pairs: Vec<(u8, u8)> = Vec::new();
        while extrap_pairs.len() < EXTRAP_SIZE_MULTI {
            let a = (rng_usize(&mut rng, 128) + 128) as u8;
            let b = (rng_usize(&mut rng, 128) + 128) as u8;
            if !extrap_pairs.contains(&(a, b)) { extrap_pairs.push((a, b)); }
        }
        (train_pairs, val_pairs, extrap_pairs)
    };
    let add_two = Task {
        name: "add_two", ground_truth: ",>,[<+>-]<.",
        train_inputs:    a2_tr.iter().map(|&(a, b)| vec![a, b]).collect(),
        train_expected:  a2_tr.iter().map(|&(a, b)| vec![a.wrapping_add(b)]).collect(),
        val_inputs:      a2_va.iter().map(|&(a, b)| vec![a, b]).collect(),
        val_expected:    a2_va.iter().map(|&(a, b)| vec![a.wrapping_add(b)]).collect(),
        extrap_inputs:   a2_ex.iter().map(|&(a, b)| vec![a, b]).collect(),
        extrap_expected: a2_ex.iter().map(|&(a, b)| vec![a.wrapping_add(b)]).collect(),
        extrap_note: "Large-magnitude pairs [128..255]×[128..255]; sums cross the u8 wrap boundary. Genuinely OOD relative to train domain [0..127]².",
    };

    vec![increment, echo, add_three, add_two]
}

// ---------------------------------------------------------------------------
// Program generation & mutation (identical to bf_study)
// ---------------------------------------------------------------------------
const OPS_BASIC: &[char] = &['+', '-', '<', '>', '.', ','];

fn random_program(rng: &mut u64, max_len: usize) -> String {
    let len = 1 + rng_usize(rng, max_len);
    let mut prog = String::new();
    let mut depth = 0i32;
    for _ in 0..len {
        let choice = rng_usize(rng, OPS_BASIC.len() + 2);
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
    if chars.is_empty() { return random_program(rng, MAX_PROG_LEN / 2); }
    match rng_usize(rng, 3) {
        0 => {
            let mut nc = chars.clone();
            let pos = rng_usize(rng, nc.len());
            nc[pos] = OPS_BASIC[rng_usize(rng, OPS_BASIC.len())];
            fix_brackets(nc.into_iter().collect(), rng)
        }
        1 if chars.len() > 1 => {
            let pos = rng_usize(rng, chars.len());
            let mut nc = chars.clone();
            nc.remove(pos);
            fix_brackets(nc.into_iter().collect(), rng)
        }
        _ => {
            if chars.len() >= MAX_PROG_LEN { return prog.to_string(); }
            let pos = rng_usize(rng, chars.len() + 1);
            let op = OPS_BASIC[rng_usize(rng, OPS_BASIC.len())];
            let mut nc = chars.clone();
            nc.insert(pos, op);
            fix_brackets(nc.into_iter().collect(), rng)
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

/// Tournament selection: LOWER HFF score wins.
fn tournament_select_hff<'a>(pop: &'a [(String, f64)], rng: &mut u64, k: usize) -> &'a str {
    let mut best = rng_usize(rng, pop.len());
    for _ in 1..k {
        let idx = rng_usize(rng, pop.len());
        if pop[idx].1 < pop[best].1 { best = idx; }
    }
    &pop[best].0
}

// ---------------------------------------------------------------------------
// One GP run
// ---------------------------------------------------------------------------
struct RunResult {
    train_solve_count:  usize,
    oracle_solve_count: usize,
    mean_val_acc:       f64,
    mean_extrap_acc:    f64,
    mean_drift:         f64,
    train_solve_gen:    Option<usize>,
}

fn gp_run(seed: u64, arm: Arm, task: &Task) -> RunResult {
    let mut rng = seed;

    let mut pop: Vec<(String, f64)> = (0..POP)
        .map(|_| {
            let prog = random_program(&mut rng, MAX_PROG_LEN);
            let score = task.hff_score(&prog, arm);
            (prog, score)
        })
        .collect();

    let mut train_solve_gen: Option<usize> = None;
    if pop.iter().any(|(p, _)| task.is_solved_train(p)) { train_solve_gen = Some(0); }

    for gen in 0..GENS {
        let mut new_pop: Vec<(String, f64)> = Vec::with_capacity(POP);
        let best = pop.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().clone();
        new_pop.push(best);
        while new_pop.len() < POP {
            let parent = tournament_select_hff(&pop, &mut rng, TOURNAMENT_K);
            let child = if rng_f64(&mut rng) < MUTATION_RATE {
                mutate(parent, &mut rng)
            } else {
                parent.to_string()
            };
            let score = task.hff_score(&child, arm);
            new_pop.push((child, score));
        }
        pop = new_pop;
        if train_solve_gen.is_none() && pop.iter().any(|(p, _)| task.is_solved_train(p)) {
            train_solve_gen = Some(gen + 1);
        }
    }

    let train_solve_count  = pop.iter().filter(|(p, _)| task.is_solved_train(p)).count();
    let oracle_solve_count = pop.iter().filter(|(p, _)| task.is_solved_oracle(p)).count();
    let mean_val_acc    = pop.iter().map(|(p, _)| task.val_acc(p)).sum::<f64>() / POP as f64;
    let mean_extrap_acc = pop.iter().map(|(p, _)| task.extrap_acc(p)).sum::<f64>() / POP as f64;
    let mean_drift      = pop.iter()
        .map(|(p, _)| task.train_acc(p) - task.val_acc(p))
        .sum::<f64>() / POP as f64;

    RunResult { train_solve_count, oracle_solve_count, mean_val_acc, mean_extrap_acc, mean_drift, train_solve_gen }
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------
fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let non_zero: Vec<f64> = diffs.iter().copied().filter(|d| d.abs() > 1e-9).collect();
    if non_zero.is_empty() { return (0.0, 1.0, 0.0); }
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

    let w_plus:  f64 = non_zero.iter().zip(ranks.iter()).filter(|(d, _)| **d > 0.0).map(|(_, r)| r).sum();
    let w_minus: f64 = non_zero.iter().zip(ranks.iter()).filter(|(d, _)| **d < 0.0).map(|(_, r)| r).sum();
    let w = w_plus.min(w_minus);
    let mean_w = n * (n + 1.0) / 4.0;
    let var_w  = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0;
    let z = (w - mean_w).abs() / var_w.sqrt();
    let p = 2.0 * normal_cdf(-z);

    let mut sd = diffs.clone();
    sd.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_diff = sd[sd.len() / 2];
    (w, p, median_diff)
}

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
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Memoriser construction
//
// Generates a BF program that outputs the correct answer for each (k, v) case
// and produces NO output for any other input. For the increment task, this
// means train_acc=1.0 and val_acc≈0.0 (no output = wrong).
//
// Tape layout: c0=input (preserved), c1=scratch, c2=copy_temp.
//
// For each case (k, v), starting with ptr at c0:
//   1. Non-destructive copy c0 -> c1 via c2:
//        >[-]>[-]<<          zero c1 and c2 (ptr back at c0)
//        [->>+>+<<<]         c0 → c2 and c3, c0 zeroed (note: >> from c0 = c2, >+>>> = c3)
//      Wait -- `[->>+>+<<<]`: inside body, ptr goes c0 -> c2 (+1), c2 -> c3 (+1 more),
//      then c3 -> c0 (<<<). So this copies c0 into c2 and c3, zeroes c0.
//        >>>[-<<<+>>>]<<<    c3 → c0 restore (ptr at c0); c2 holds copy.
//
//   Correction: after the copy above, c2=input, c3=0 (consumed), c0=restored.
//   So c2 is our comparison cell.
//
//   2. Subtract k from c2 (the scratch copy):
//        >> k×'-' <<         ptr at c2, subtract k, ptr back to c0
//
//   3. Test c2 == 0: use c3 as flag.
//        >>>+<<<             c3 = 1 (flag set); ptr at c0
//        >>                  ptr at c2
//        [>-<-]              drain c2: each step decrements c3 and c2
//                            after: if c2 was 0, c3 still 1 (match); else c3 = 0
//        >                   ptr at c3
//        [                   if c3 == 1 (match):
//          >[-]              zero c4
//          v×'+'             c4 = v
//          .                 output v
//          [-]               zero c4
//          <[-]              zero c3 (exit loop)
//        ]
//        <<                  ptr back at c1 area... actually ptr at c2
//        <<                  ptr at c0; c1=0,c2=0,c3=0 ready for next case.
//
// After all cases, the program halts with no output for non-train inputs.
// ---------------------------------------------------------------------------
pub fn build_memoriser_increment(cases: &[(u8, u8)]) -> String {
    // Tape layout: c0=input(preserved), c1=unused, c2=copy_scratch, c3=is-zero flag, c4=output_temp.
    // ptr always starts at c0 for each case.
    //
    // For each case (k, v):
    //  1. Zero c2, c3, c4:  >>[-]>[-]>[-]<<<<
    //  2. Copy c0 -> c2 (via c3 as temp), preserving c0:
    //       [->>+>+<<<]      c0 → c2 and c3; c0 zeroed (ptr ends at c0)
    //       >>>[-<<<+>>>]<<< c3 → c0 restore (ptr at c0); c2=input_copy, c3=0
    //  3. Subtract k from c2: >> k×'-' <<
    //  4. Set c3 = 1 (assume match): >>>+<<<
    //  5. Test c2 == 0 (is-zero pattern):
    //       >>                ptr at c2
    //       [                 while c2 != 0:
    //         >[-]<           zero c3 (no match)
    //         [-]             drain c2 to zero
    //       ]                 c2=0; c3=1 iff c2 was originally 0
    //  6. Check c3 (match flag):
    //       >                 ptr at c3
    //       [                 if match:
    //         >[-]            zero c4
    //         v×'+'           c4 = v
    //         .               output v
    //         [-]             zero c4
    //         <[-]            zero c3 (exit loop)
    //       ]
    //  7. Return to c0:  <<<  (ptr: c3 -> c0, since c1 is unused; go c3->c2->c1->c0 = <<<)

    let mut bf = String::new();
    bf.push(','); // read input to c0

    for &(k, v) in cases {
        // 1. Zero c2, c3, c4.
        bf.push_str(">>[-]>[-]>[-]<<<<"); // ptr at c0

        // 2. Copy c0 -> c2 (and c3 temp), restore c0 from c3.
        bf.push_str("[->>+>+<<<]");        // c0→c2,c3; c0=0; ptr at c0
        bf.push_str(">>>[-<<<+>>>]<<<");   // c3→c0 restore; c2=copy, c3=0; ptr at c0

        // 3. Subtract k from c2.
        bf.push_str(">>");                 // ptr at c2
        for _ in 0..k { bf.push('-'); }
        bf.push_str("<<");                 // ptr at c0

        // 4. Set c3 = 1.
        bf.push_str(">>>");               // ptr at c3
        bf.push('+');
        bf.push_str("<<<");               // ptr at c0

        // 5. Is-zero test on c2.
        bf.push_str(">>");                 // ptr at c2
        bf.push('[');                     // while c2 != 0:
        bf.push_str(">[-]<");             //   zero c3 (no match)
        bf.push_str("[-]");               //   drain c2 to 0
        bf.push(']');                     // end; c2=0; c3=1 iff c2 was 0

        // 6. Check flag and output.
        bf.push('>');                     // ptr at c3
        bf.push('[');                     // if match (c3=1):
        bf.push('>');                     //   ptr at c4
        bf.push_str("[-]");               //   zero c4
        for _ in 0..v { bf.push('+'); }   //   c4 = v
        bf.push('.');                     //   output v
        bf.push_str("[-]");               //   zero c4
        bf.push('<');                     //   ptr at c3
        bf.push_str("[-]");               //   zero c3 (exit loop)
        bf.push(']');

        // 7. Return to c0 (ptr at c3; c3 is at offset +3 from c0).
        bf.push_str("<<<");               // ptr at c0; ready for next case.
    }
    bf
}

// ---------------------------------------------------------------------------
// Memoriser attack
// ---------------------------------------------------------------------------

/// Run the memoriser attack for the increment task.
/// Returns (mem_train_acc, mem_val_acc, mem_hff_val_score, gt_hff_val_score, verdict).
fn memoriser_attack(task: &Task) -> (f64, f64, f64, f64, String) {
    let train_cases: Vec<(u8, u8)> = task.train_inputs.iter()
        .zip(task.train_expected.iter())
        .map(|(inp, exp)| (inp[0], exp[0]))
        .collect();

    let memoriser = build_memoriser_increment(&train_cases);

    let mem_train_acc = task.train_acc(&memoriser);
    let mem_val_acc   = task.val_acc(&memoriser);
    let mem_hff_val   = hff_truenorth(&[1.0 - mem_train_acc, 1.0 - mem_val_acc]);

    let gt = task.ground_truth;
    let gt_hff_val = hff_truenorth(&[1.0 - task.train_acc(gt), 1.0 - task.val_acc(gt)]);

    let none_verdict = if (1.0 - mem_train_acc) <= 0.0 + 1e-9 {
        "NONE is fooled: memoriser err_train=0.0 wins single-objective selection"
    } else {
        "NONE not fooled (unexpected — check memoriser construction)"
    };

    let hff_verdict = if mem_hff_val > gt_hff_val + 1e-9 {
        "HFF_VAL DEFEATS memoriser: ground truth has strictly lower HFF angle (PASS)"
    } else if (mem_hff_val - gt_hff_val).abs() <= 1e-9 {
        "HFF_VAL ties memoriser (both have same train+val error profile)"
    } else {
        "HFF_VAL FAILS to defeat memoriser: memoriser wins HFF_VAL tournament (DESIGN FLAW)"
    };

    let verdict = format!(
        "Memoriser: train_acc={:.3}  val_acc={:.3}  HFF_VAL_score={:.6}\n\
         Ground truth `{}`: HFF_VAL_score={:.6}\n\
         NONE:    {}\n\
         HFF_VAL: {}\n",
        mem_train_acc, mem_val_acc, mem_hff_val,
        task.ground_truth, gt_hff_val,
        none_verdict, hff_verdict,
    );

    (mem_train_acc, mem_val_acc, mem_hff_val, gt_hff_val, verdict)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
fn main() {
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples").join("bf_hff_study");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    let ledger_path = out_dir.join("results_hff.jsonl");
    let mut ledger = std::fs::File::create(&ledger_path).expect("create ledger");

    let tasks = make_tasks();

    // =========================================================================
    // Ground-truth verification
    // =========================================================================
    println!("=== Ground-truth verification ===");
    for task in &tasks {
        let tr = task.train_acc(task.ground_truth);
        let va = task.val_acc(task.ground_truth);
        let ex = task.extrap_acc(task.ground_truth);
        println!(
            "  {}: gt={:?}  train={:.3} val={:.3} extrap={:.3}  {}",
            task.name, task.ground_truth, tr, va, ex,
            if tr >= 1.0 - 1e-9 && va >= 1.0 - 1e-9 && ex >= 1.0 - 1e-9 { "OK" } else { "FAIL" }
        );
        assert!(
            tr >= 1.0 - 1e-9 && va >= 1.0 - 1e-9 && ex >= 1.0 - 1e-9,
            "Ground truth must solve all splits for {}",
            task.name
        );
    }
    println!();

    // Print split contents for reproducibility.
    println!("=== Input split summary ===");
    for task in &tasks {
        println!("  task={}", task.name);
        println!("    TRAIN ({} inputs):  {:?}", task.train_inputs.len(), task.train_inputs);
        println!("    VAL   ({} inputs):  {:?}", task.val_inputs.len(), task.val_inputs);
        println!("    EXTRAP ({} inputs): {:?}", task.extrap_inputs.len(), task.extrap_inputs);
        println!("    extrap_note: {}", task.extrap_note);
    }
    println!();

    // =========================================================================
    // Memoriser attack test (increment task)
    // =========================================================================
    println!("=== Memoriser Attack Test (increment task) ===");
    let increment_task = tasks.iter().find(|t| t.name == "increment").unwrap();
    let (mem_train_acc, mem_val_acc, mem_hff_val, gt_hff_val, attack_verdict) =
        memoriser_attack(increment_task);
    println!("{}", attack_verdict);

    let attack_pass = mem_hff_val > gt_hff_val + 1e-9;

    // =========================================================================
    // Phase 1: Main study — 4 arms × 4 tasks × 30 seeds
    // =========================================================================
    let main_arms = [Arm::None, Arm::HffTrain, Arm::HffVal, Arm::HffExtrap];
    let n_tasks = tasks.len();
    let n_arms  = main_arms.len();

    let mut train_solves:  Vec<Vec<Vec<usize>>> = vec![vec![vec![0usize; SEEDS]; n_arms]; n_tasks];
    let mut oracle_solves: Vec<Vec<Vec<usize>>> = vec![vec![vec![0usize; SEEDS]; n_arms]; n_tasks];
    let mut val_accs:      Vec<Vec<Vec<f64>>>   = vec![vec![vec![0.0; SEEDS]; n_arms]; n_tasks];
    let mut extrap_accs:   Vec<Vec<Vec<f64>>>   = vec![vec![vec![0.0; SEEDS]; n_arms]; n_tasks];
    let mut drifts:        Vec<Vec<Vec<f64>>>   = vec![vec![vec![0.0; SEEDS]; n_arms]; n_tasks];

    println!("=== Phase 1: {} tasks × {} arms × {} seeds ===\n", n_tasks, n_arms, SEEDS);

    for (ti, task) in tasks.iter().enumerate() {
        println!("--- Task: {} ---", task.name);
        for (ai, &arm) in main_arms.iter().enumerate() {
            for seed in 0..SEEDS {
                // Seed formula distinct from bf_study seeds.
                let run_seed = (seed as u64) * 1_000_003
                    + (ai as u64 + 7) * 999_983
                    + (ti as u64 + 3) * 100_003
                    + 0xDEAD_BEEF;
                let r = gp_run(run_seed, arm, task);

                train_solves[ti][ai][seed]  = r.train_solve_count;
                oracle_solves[ti][ai][seed] = r.oracle_solve_count;
                val_accs[ti][ai][seed]      = r.mean_val_acc;
                extrap_accs[ti][ai][seed]   = r.mean_extrap_acc;
                drifts[ti][ai][seed]        = r.mean_drift;

                writeln!(
                    ledger,
                    "{{\"phase\":\"main\",\"task\":\"{}\",\"arm\":\"{}\",\
                    \"seed\":{},\"train_solve_count\":{},\"oracle_solve_count\":{},\
                    \"mean_val_acc\":{:.4},\"mean_extrap_acc\":{:.4},\"mean_drift\":{:.4},\
                    \"pop\":{POP},\"solve_gen\":{},\
                    \"train_size\":{},\"val_size\":{},\"extrap_size\":{}}}",
                    task.name, arm.name(), seed,
                    r.train_solve_count, r.oracle_solve_count,
                    r.mean_val_acc, r.mean_extrap_acc, r.mean_drift,
                    r.train_solve_gen.map(|g| g.to_string()).unwrap_or_else(|| "null".to_string()),
                    task.train_inputs.len(), task.val_inputs.len(), task.extrap_inputs.len(),
                ).expect("write ledger");
            }

            let sr: Vec<f64> = train_solves[ti][ai].iter().map(|&c| c as f64 / POP as f64).collect();
            let va: Vec<f64> = val_accs[ti][ai].clone();
            let dr: Vec<f64> = drifts[ti][ai].clone();
            println!(
                "  arm={} train_solve={:.3}±{:.3} val_acc={:.3}±{:.3} drift={:.3}±{:.3}",
                arm.name(), mean_f64(&sr), std_f64(&sr), mean_f64(&va), std_f64(&va),
                mean_f64(&dr), std_f64(&dr)
            );
        }
        println!();
    }

    // =========================================================================
    // Write RESULTS_HFF.md
    // =========================================================================
    let results_path = out_dir.join("RESULTS_HFF.md");
    let mut rmd = std::fs::File::create(&results_path).expect("create RESULTS_HFF.md");

    writeln!(rmd, "# BF HFF Validation Study — Results\n").unwrap();
    writeln!(rmd, "Validation-in-fitness via HFF (TrueNorth) for Brainfuck GP.\n").unwrap();

    writeln!(rmd, "## Setup\n").unwrap();
    writeln!(rmd, "- Population: {POP}, generations: {GENS}, seeds: {SEEDS}").unwrap();
    writeln!(rmd, "- Tournament K = {TOURNAMENT_K} (~7% of pop)").unwrap();
    writeln!(rmd, "- Max program length: {MAX_PROG_LEN}").unwrap();
    writeln!(rmd, "- Mutation rate: {MUTATION_RATE}").unwrap();
    writeln!(rmd, "- TRAIN={TRAIN_SIZE}, VAL={VAL_SIZE}, EXTRAP={EXTRAP_SIZE} inputs per task").unwrap();
    writeln!(rmd, "\nArms:\n\
        - **NONE**: err_train (lower=better), single-objective baseline\n\
        - **HFF_TRAIN**: k=1 TrueNorth([err_train])\n\
        - **HFF_VAL**: k=2 TrueNorth([err_train, err_val])\n\
        - **HFF_EXTRAP**: k=3 TrueNorth([err_train, err_val, err_extrap])").unwrap();
    writeln!(rmd).unwrap();

    writeln!(rmd, "## Input Split Scheme\n").unwrap();
    writeln!(rmd, "Deterministic per-task seeds. All splits disjoint.\n").unwrap();
    writeln!(rmd, "| Task | Train N | Val N | Extrap N | Extrap Regime |").unwrap();
    writeln!(rmd, "|------|---------|-------|----------|--------------|").unwrap();
    for task in &tasks {
        writeln!(rmd, "| {} | {} | {} | {} | {} |",
            task.name, task.train_inputs.len(), task.val_inputs.len(),
            task.extrap_inputs.len(), task.extrap_note).unwrap();
    }
    writeln!(rmd).unwrap();

    // Memoriser attack
    writeln!(rmd, "## Memoriser Attack Test (increment task)\n").unwrap();
    writeln!(rmd, "A hand-constructed BF program (`build_memoriser_increment`) that outputs the\n\
        correct answer for every TRAIN input and produces no output (wrong) for any other input.\n\
        This is the gold-standard test for whether HFF_VAL catches input-set overfit.\n").unwrap();
    writeln!(rmd, "```\n{}\n```\n", attack_verdict).unwrap();
    writeln!(rmd, "**Overall verdict: {}**\n",
        if attack_pass { "PASS — HFF_VAL defeats the memoriser" }
        else { "FAIL — HFF_VAL does NOT defeat the memoriser (investigate)" }).unwrap();
    writeln!(rmd, "- Memoriser: train_acc={:.3}, val_acc={:.3}, HFF_VAL_score={:.6}",
        mem_train_acc, mem_val_acc, mem_hff_val).unwrap();
    writeln!(rmd, "- Ground truth: HFF_VAL_score={:.6}", gt_hff_val).unwrap();
    writeln!(rmd).unwrap();

    // Per-task results
    writeln!(rmd, "## Phase 1: Main Study Results\n").unwrap();
    writeln!(rmd, "Metrics per arm, mean±std over {SEEDS} seeds.\n").unwrap();
    writeln!(rmd, "- **train_solve_rate**: fraction of final population with train_acc=1.0").unwrap();
    writeln!(rmd, "- **oracle_solve_rate**: fraction with train AND val AND extrap accuracy = 1.0").unwrap();
    writeln!(rmd, "- **mean_val_acc**: mean validation accuracy in final population").unwrap();
    writeln!(rmd, "- **mean_drift**: mean (train_acc - val_acc); higher = more overfit").unwrap();
    writeln!(rmd).unwrap();

    let none_ai   = main_arms.iter().position(|a| *a == Arm::None).unwrap();
    let htrain_ai = main_arms.iter().position(|a| *a == Arm::HffTrain).unwrap();
    let hval_ai   = main_arms.iter().position(|a| *a == Arm::HffVal).unwrap();
    let hext_ai   = main_arms.iter().position(|a| *a == Arm::HffExtrap).unwrap();

    for (ti, task) in tasks.iter().enumerate() {
        writeln!(rmd, "### Task: {}\n", task.name).unwrap();
        writeln!(rmd, "| Arm | TrainSolve Mean±Std | OracleSolve Mean±Std | ValAcc Mean±Std | Drift Mean±Std |").unwrap();
        writeln!(rmd, "|-----|--------------------|-----------------------|----------------|---------------|").unwrap();

        for (ai, arm) in main_arms.iter().enumerate() {
            let sr: Vec<f64> = train_solves[ti][ai].iter().map(|&c| c as f64 / POP as f64).collect();
            let os: Vec<f64> = oracle_solves[ti][ai].iter().map(|&c| c as f64 / POP as f64).collect();
            let va: Vec<f64> = val_accs[ti][ai].clone();
            let dr: Vec<f64> = drifts[ti][ai].clone();
            writeln!(rmd, "| {} | {:.3}±{:.3} | {:.3}±{:.3} | {:.3}±{:.3} | {:.3}±{:.3} |",
                arm.name(),
                mean_f64(&sr), std_f64(&sr), mean_f64(&os), std_f64(&os),
                mean_f64(&va), std_f64(&va), mean_f64(&dr), std_f64(&dr),
            ).unwrap();
        }
        writeln!(rmd).unwrap();

        writeln!(rmd, "#### Wilcoxon signed-rank tests (30 paired seeds, two-sided)\n").unwrap();
        writeln!(rmd, "| Comparison | W | p | Median Δ | p<0.05? |").unwrap();
        writeln!(rmd, "|------------|---|---|---------|--------|").unwrap();

        let sr_none:   Vec<f64> = train_solves[ti][none_ai].iter().map(|&c| c as f64 / POP as f64).collect();
        let sr_htrain: Vec<f64> = train_solves[ti][htrain_ai].iter().map(|&c| c as f64 / POP as f64).collect();
        let sr_hval:   Vec<f64> = train_solves[ti][hval_ai].iter().map(|&c| c as f64 / POP as f64).collect();
        let sr_hext:   Vec<f64> = train_solves[ti][hext_ai].iter().map(|&c| c as f64 / POP as f64).collect();
        let va_none:   Vec<f64> = val_accs[ti][none_ai].clone();
        let va_hval:   Vec<f64> = val_accs[ti][hval_ai].clone();
        let va_hext:   Vec<f64> = val_accs[ti][hext_ai].clone();

        let (w1, p1, m1) = wilcoxon_signed_rank(&sr_hval, &sr_htrain);
        let (w2, p2, m2) = wilcoxon_signed_rank(&sr_hval, &sr_none);
        let (w3, p3, m3) = wilcoxon_signed_rank(&sr_hext, &sr_hval);
        let (w4, p4, m4) = wilcoxon_signed_rank(&va_hval, &va_none);
        let (w5, p5, m5) = wilcoxon_signed_rank(&va_hext, &va_hval);

        let sig = |p: f64| if p < 0.05 { "YES" } else { "NO" };
        writeln!(rmd, "| HFF_VAL vs HFF_TRAIN (train_solve)  | {w1:.1} | {p1:.4} | {m1:+.3} | {} |", sig(p1)).unwrap();
        writeln!(rmd, "| HFF_VAL vs NONE (train_solve)       | {w2:.1} | {p2:.4} | {m2:+.3} | {} |", sig(p2)).unwrap();
        writeln!(rmd, "| HFF_EXTRAP vs HFF_VAL (train_solve) | {w3:.1} | {p3:.4} | {m3:+.3} | {} |", sig(p3)).unwrap();
        writeln!(rmd, "| HFF_VAL vs NONE (val_acc)           | {w4:.1} | {p4:.4} | {m4:+.3} | {} |", sig(p4)).unwrap();
        writeln!(rmd, "| HFF_EXTRAP vs HFF_VAL (val_acc)     | {w5:.1} | {p5:.4} | {m5:+.3} | {} |", sig(p5)).unwrap();
        writeln!(rmd).unwrap();
    }

    // Extrap accuracy table
    writeln!(rmd, "## Extrap Accuracy Summary\n").unwrap();
    writeln!(rmd, "Mean extrap_acc in final population (higher = better generalisation).\n").unwrap();
    writeln!(rmd, "| Task | NONE | HFF_TRAIN | HFF_VAL | HFF_EXTRAP |").unwrap();
    writeln!(rmd, "|------|------|-----------|---------|------------|").unwrap();
    for (ti, task) in tasks.iter().enumerate() {
        let fmt = |ai: usize| {
            let v: Vec<f64> = extrap_accs[ti][ai].clone();
            format!("{:.3}±{:.3}", mean_f64(&v), std_f64(&v))
        };
        writeln!(rmd, "| {} | {} | {} | {} | {} |",
            task.name, fmt(none_ai), fmt(htrain_ai), fmt(hval_ai), fmt(hext_ai)).unwrap();
    }
    writeln!(rmd).unwrap();

    writeln!(rmd, "## Reproduce\n").unwrap();
    writeln!(rmd, "```bash").unwrap();
    writeln!(rmd, "git checkout feat/bf-hff-validation").unwrap();
    writeln!(rmd, "RUSTFLAGS=\"-D warnings\" cargo test --no-default-features").unwrap();
    writeln!(rmd, "cargo run --release --example bf_hff_study --no-default-features").unwrap();
    writeln!(rmd, "```\n").unwrap();

    println!("Results written to {}", results_path.display());
    println!("Ledger: {}", ledger_path.display());
    println!("\nMEMORISER ATTACK: {}", if attack_pass { "PASS" } else { "FAIL" });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hff_truenorth_orientation() {
        // Perfect [0, 0] should score lower (better) than worst [1, 1].
        let perfect = hff_truenorth(&[0.0, 0.0]);
        let worst   = hff_truenorth(&[1.0, 1.0]);
        assert!(perfect < worst, "perfect {} should be lower than worst {}", perfect, worst);

        // Memoriser-like [0.0, 1.0] should score worse than generaliser [0.0, 0.0].
        let memoriser = hff_truenorth(&[0.0, 1.0]);
        let generaliser = hff_truenorth(&[0.0, 0.0]);
        assert!(
            memoriser > generaliser,
            "memoriser-like [0,1]={:.6} should score worse than generaliser [0,0]={:.6}",
            memoriser, generaliser
        );
    }

    #[test]
    fn ground_truth_solves_all_splits() {
        let tasks = make_tasks();
        for task in &tasks {
            assert!(task.train_acc(task.ground_truth) >= 1.0 - 1e-9,
                "{} ground truth fails train", task.name);
            assert!(task.val_acc(task.ground_truth) >= 1.0 - 1e-9,
                "{} ground truth fails val", task.name);
            assert!(task.extrap_acc(task.ground_truth) >= 1.0 - 1e-9,
                "{} ground truth fails extrap", task.name);
        }
    }

    #[test]
    fn splits_are_disjoint() {
        let tasks = make_tasks();
        for task in &tasks {
            let train_set: std::collections::HashSet<Vec<u8>> =
                task.train_inputs.iter().cloned().collect();
            let val_set: std::collections::HashSet<Vec<u8>> =
                task.val_inputs.iter().cloned().collect();
            let extrap_set: std::collections::HashSet<Vec<u8>> =
                task.extrap_inputs.iter().cloned().collect();
            let tv: std::collections::HashSet<_> = train_set.intersection(&val_set).collect();
            assert!(tv.is_empty(), "{}: train ∩ val non-empty: {:?}", task.name, tv);
            let te: std::collections::HashSet<_> = train_set.intersection(&extrap_set).collect();
            assert!(te.is_empty(), "{}: train ∩ extrap non-empty: {:?}", task.name, te);
        }
    }

    #[test]
    fn memoriser_attack_increment() {
        let tasks = make_tasks();
        let task = tasks.iter().find(|t| t.name == "increment").unwrap();
        let (mem_train_acc, mem_val_acc, mem_hff_val, gt_hff_val, _verdict) =
            memoriser_attack(task);

        assert!(
            mem_train_acc >= 1.0 - 1e-9,
            "Memoriser must have train_acc=1.0; got {:.4}",
            mem_train_acc
        );
        assert!(
            mem_val_acc < 0.5,
            "Memoriser must have val_acc < 0.5 (fails most val inputs); got {:.4}",
            mem_val_acc
        );
        assert!(
            gt_hff_val < mem_hff_val,
            "HFF_VAL must rank ground truth better than memoriser: {:.6} < {:.6}",
            gt_hff_val, mem_hff_val
        );
    }

    #[test]
    fn memoriser_fails_val_inputs() {
        let tasks = make_tasks();
        let task = tasks.iter().find(|t| t.name == "increment").unwrap();
        let train_cases: Vec<(u8, u8)> = task.train_inputs.iter()
            .zip(task.train_expected.iter())
            .map(|(inp, exp)| (inp[0], exp[0]))
            .collect();
        let memoriser = build_memoriser_increment(&train_cases);

        let train_set: std::collections::HashSet<u8> =
            task.train_inputs.iter().map(|v| v[0]).collect();

        // At least one val input should get the wrong output.
        let any_wrong = task.val_inputs.iter().any(|vi| {
            if train_set.contains(&vi[0]) { return false; }
            let expected = vec![vi[0].wrapping_add(1)];
            match run_bf(&memoriser, vi) {
                fuller::bf::eval::TapeResult::Ok { output } => output != expected,
                _ => true,
            }
        });
        assert!(any_wrong, "Memoriser should fail at least one non-train val input");
    }
}
