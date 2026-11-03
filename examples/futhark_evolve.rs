//! THE GENERAL FUTHARK RUNNER: evolve a program for a published benchmark of the Futhark kingdom
//! (`fuller::futhark_sig`: a problem is a view over the master table), from its stored cases alone (data in, the
//! published program's exact outputs out; HFF 0 is exact). A new benchmark is a problem folder and a problem entry,
//! never a new runner (`fuller::futhark_sig::PROBLEM_NAMES`: reduce, scan, histogram, pathfinder, kmeans, and since
//! step 4 of `docs/PLAN_futhark_kingdom.md` A7, the matrix kingdom, `--problem a7`, which the deprecated
//! `examples/futhark_a7_evolve.rs` ran on a loop of its own). What every problem gets: `docs/CHECKLIST_futhark_problem.md`.
//! The study it serves: `docs/STUDY_evolving_benchmarks.md`.
//!
//!   cargo run --release --features gpu,wgsl --example futhark_evolve -- --problem pathfinder --regime R \
//!       --evaluations 6000 --minutes 60 --seed 1 --out /private/tmp/claude-501/fbench/pathfinder_R_s1
//!   evowatch /private/tmp/claude-501/fbench/pathfinder_R_s1
//!
//! Regimes (the study's):
//! * `R`, from random: a random population (the root cell a function of the root type), tournament selection,
//!   low-change mutation, homologous crossover, MARRIAGE (typed subtree crossover between two parents whose expressed
//!   trees differ, both offspring written back), the PUMP (every `--pump-every` generations the e-graph variants of
//!   the best enter) and the SATURATED FOLD in the loop (every `--fold-every` generations, the jointly extracted forms
//!   of the population enter);
//! * `C`, the random-search control: a fresh random gene per evaluation, the same budget, no selection;
//! * with `--stop-at-exact` (R and C only), a discovery run stops at the first program exact on every case and
//!   reports that evaluation (`early_stop`): it answers H1 and H2 and never enters stage 2, so speed is measured
//!   in P alone;
//! * `P`, from the published program toward faster: the reference gene (`--start published`) or the planted slowdown
//!   (`--start planted`, the control), its e-graph variants and the saturated fold's forms make the population; it
//!   starts in stage 2 (the reference is exact).
//!
//! FITNESS, a NAMED SCHEME (`--fitness NAME`, `fuller::fitness`, family `futhark_sig`; `docs/FITNESS_SCHEMES.md`),
//! stated in the header, the stream's `run_start` and `report.tsv`. The default, `futhark-slope4-qqs`, is described
//! below; its stage 2 is ONE FLAT VECTOR (Andrew's ruling of 2026-11-03: the HFF angle is the final fitness and
//! nothing manipulates it): the quality inputs entered `--quality-weight` times (default
//! `fitness::ANSWER_QUALITY_WEIGHT`, measured) beside the timing slots, TrueNorth over the whole. The old schemes,
//! `futhark-slope4-gated` (stage 2 gated by an offset `π/2 + θ_quality`, the runs of 2026-10-29 to 2026-11-03),
//! `futhark-slope4` (stage 2 ungated, the 2026-10-29 reruns) and `futhark-study7` (the study's seven inputs, stage 2
//! ungated, the 2026-10-27/28 study), are NOT TO BE RUN AS IS: each has a known fault, named in its header line;
//! redefine the fitness function before using one. Staged (`fuller::runner::FitnessStage`,
//! `fuller::fitness::Scheme::staged_weighted`):
//! * stage 1, quality only, no timing: for every stored case, training and held-out alike, the inputs of
//!   `futhark_sig::quality_inputs` in the order of `futhark_sig::QUALITY_PARTS` (plan section B, 2026-10-29), each its
//!   own, linear: the share of cells not exact; the two shape inputs, `(1 - r)/2` for Spearman's and for Pearson's
//!   correlation with the correct output (absent when the correct output is constant); the shares of cells off by
//!   over 50 % and over 1 % of the correct value's magnitude. (The study of 2026-10-27 used seven inputs, now
//!   `QualityPart::Study`; its pilot used three on HFF's log scale, which put every inexact program within 0.03 of 1
//!   and ranked its whole population alike for 26 generations.)
//! * stage 2, once the best is exact on every case (a `stage_change` event, never back): the flat HFF of that
//!   quality group (repeated `--quality-weight` times) and the speed group: the program's TIMING SLOTS (`fuller::runner::TimingSlots`, `--pairs` of
//!   them, 11 by default), interleaved timed pairs against the published program (the yardstick) on the largest
//!   training case, kept by the program's text and filled before it is scored, so a program re-scored in a later
//!   generation reads the same pairs and is never timed again; each pair is its own speed input, `r/(1+r)` with `r`
//!   the candidate's GPU time over the yardstick's, never averaged. (Three pairs a candidate, before 2026-10-27
//!   21:00Z, ranked one program at 1.287 in the run and 1.000 in an 11-pair re-time.)
//!   A pair is the candidate and the published program INTERLEAVED run by run on one upload of the timing input, a
//!   warm-up each and then at least 1 s a side (`fuller::futhark_batch::pair_runs`: the runs from the slower side's
//!   run time, at least 3; `Step::Pair`, the driver's `p` command), on
//!   every evaluation path (`--eval single` in a timing process of the candidate's own); each side's time is its
//!   fastest run's sum of GPU-timed passes. (Until 2026-10-31 each run re-uploaded the input and the GPU's clock
//!   stepped down in the gaps: the A/A gate failed on every path.) The A/A noise
//!   floor (the yardstick against itself, 11 pairs) is measured at the start; in P the published program is also
//!   timed against itself THROUGH THE SLOT PATH (compiled as a candidate, `--pairs` pairs), the in-run A/A. A speed
//!   claim counts only when the champion's 11-pair re-time at the end has its median below the least MEDIAN of the
//!   published program re-timed `--aa-retimes` times by the champion's own path (default `runner::AA_RETIMES`, a
//!   5 % false "beat"; fault 13: until 2026-11-01 it was the least single ratio of the two A/A sets) and at least 9
//!   of 11 pairs faster (`speed_claim_beats_noise_floor` in the report).
//!
//! THE K-MEANS SCHEME, `futhark-kmeans-better` (Andrew's specification of 2026-11-03, `docs/KMEANS_FITNESS.md`;
//! k-means's own default, `Problem::fitness`): the runs below, and per case THREE quality inputs, each the cliff
//! (`fitness::quality_cliff`, `--quality-cliff`, default 0.005) of x = the candidate relative to the seed (the final
//! cosine inertia; the final mean log10 percentile; the convergence 1 + (k - k_ref), k the further iterations the
//! SEED's step needs from the candidate's final centroids: the seed's server stepped on from them on the c backend,
//! the `k` command's further entry on the device), and the two speed inputs below; ONE FLAT VECTOR at m =
//! `--quality-weight` (default 3), its angle the fitness, its HFF percentile F(theta) and d reported beside it in the
//! log lines, `per_generation.tsv` and the report. `futhark-kmeans-qqs` (below) is NOT TO BE RUN AS IS: it rewards
//! early stopping, r/(1+r) compressing a quality loss (the first k-means run).
//!
//! THE SCHEME BEFORE IT, `futhark-kmeans-qqs` (`docs/KMEANS_FITNESS.md`; an answer-scored scheme is refused for
//! k-means: a k-means step has no exact answer): the candidate and the seed are
//! each run to UltraDim's stopping rule on every fitness case (`futhark_sig::kmeans_converged::converge`: the harness
//! assigns, the program steps, the Euclidean inertia's change under 1e-4 stops; the 100-iteration cap is UltraDim's
//! `converged: false`, a slow finished run), interleaved `--pairs` times (default 2 for this scheme, so that neither
//! side always runs after the other: the candidate first in the even pairs), and the inputs a case are against THE
//! SEED'S RUN OF THAT INTERLEAVING: the quality group (cosine inertia ratio; the ratio of the seed's mean log10
//! angular percentile to the candidate's over the points off their centroid) and the speed group (THE PROGRAM'S OWN
//! TIME to the stopping point, the shim's GPU-timed passes or `futhark c`'s runtime, summed over the iterations, the
//! harness's assignment and the processes excluded; the iterations ratio). The fitness is ONE FLAT VECTOR
//! (`fitness::qqs`): every case's quality inputs entered `--quality-weight` times (default 2), its speed inputs once,
//! TrueNorth over the whole, and that angle is the fitness: no gate, no offset. No timing input and no timed A/A sets;
//! the in-run A/A is the seed judged as a candidate against itself, on both backends, and the spread of its speed
//! inputs around 1/2 is the noise floor a speed claim must clear (the report's `seed_aa_speed_floor`). ONE PROCESS A
//! PROGRAM: on the c backend the compiled program is a Futhark server (`futhark c --server`,
//! `kmeans_converged::Server`), kept alive across its iterations and its cases with each case's points loaded once
//! (until 2026-11-03 every iteration was its own process with the points down the pipe: about two minutes a
//! candidate); on the GPU backend each iteration is still one process of the shim's executable, which has no server
//! mode. The work guard of this scheme is the problem's own: a candidate's work model on a fitness case over 10 x the
//! seed's on that case is not run and takes the worst score (never a stop of the run). The control at the start is
//! that the compiled seed's run CONVERGES on every fitness case. THE TEST CASE (`test_n1000_k32`, never a fitness
//! input) is run every generation the champion changes, champion and seed interleaved, and at the end:
//! `OUT/test_case.tsv` and the report carry the four inputs and the raw measures of both. k-means's seed is
//! UltraDim's own kernels translated construct for construct (`futhark_sig::kmeans_seed`, `docs/kmeans_seed.md`;
//! `Seed::Translated`), on the CPU 2026-11-03; the GPU has not run it. The retired `futhark-kmeans-quality-first`
//! (the two-level HFF gated by an offset) and `futhark-kmeans-converged` (five inputs, the harness's clock), both
//! NOT TO BE RUN AS IS, still run through the same judge with their own inputs.
//!
//! The BUDGET is `--evaluations` fitness evaluations (an individual scored, a cache hit included, so R and C are
//! counted alike); the wall clock (`--minutes`) is a guard. The work guard: every candidate's exact node executions
//! on every case (`Interp::work`) are computed before it runs, and one over the per-submit cap stops the run as a
//! misconfigured experiment, named.
//!
//! Backends: `--backend gpu` (the default) compiles each candidate with the branch compiler's webgpu backend, links
//! it against the shim and runs it on our wgpu/Metal path; `--backend c` compiles with `futhark c` and runs on the
//! host, for host-only measurement; it has no GPU timing, so stage 2 cannot be entered with it (an error, named).
//!
//! EVALUATION, `--eval batch` (the default since 2026-10-28) or `--eval single`:
//! * batch (`fuller::futhark_batch`): every program of a generation not seen before is one entry point of ONE
//!   Futhark program, compiled once as a library and run in ONE process by a driver that sets the device up once,
//!   runs each quality entry on every case and, in stage 2, each timed entry's pairs against the published program's
//!   entry in the same process (the same scheme: `--pairs` pairs, each its own speed input, the first
//!   process-equivalent of a fresh program discarded). A program the branch compiler refuses is found by halving the
//!   batch, logged in `bugs.log` and scored worst; the rest are scored from their batches. The A/A sets and the
//!   champion's re-time run in batch processes too. The results are taken when each program is scored, so selection,
//!   the counts and the bugs are the single path's (checked bit for bit, `docs/PLAN_futhark_wgpu_host.md` §13);
//! * rust (`fuller::futhark_rust`, `docs/PLAN_futhark_rust_wgpu.md`): the same batch, compiled by our Futhark fork's
//!   rust-wgpu backend into ONE Rust module (WGSL kernels and a host over wgpu-rs), built by `rustc` into a dylib and
//!   run IN THIS PROCESS by the runtime `futhark-rust-rt` on one device: no C, no shim, no process. The same commands,
//!   records and halving as batch; what is scored is the code that ships. In the ledger its `eval` is `rust`, its
//!   `futhark_ms` the fork compiler and its `cc_ms` rustc (no port, check or link stage);
//! * single: each program its own compile and its own processes, the path before.
//!
//! Where the time goes: every ledger line carries its evaluation's path, its batch's size and each stage's
//! milliseconds (render, futhark, port, cc, check, link, start, device, pipelines, runs, timing; in batch mode its
//! share of its batch), then its quality and speed inputs exactly; each generation's stage seconds and evaluation
//! rate are in the stream's extras (`stage_*_s`, `gen_seconds`, `evaluations_per_second`) and in `OUT/stages.tsv`.
//!
//! The standard (`docs/STANDARD_runners.md`): the stream (`OUT/stream.jsonl`), the staged two-level HFF,
//! checkpoints (`OUT/checkpoint.json`, `--resume`), the kill trigger (`OUT/killme.trigger`), the guards, the hall of
//! fame (`OUT/hall_of_fame/`), the ledger with lineage (`OUT/ledger.tsv`), mutation events (`egraph_exact`, `pump`,
//! `saturated_fold`, `marriage`, `stage_change`, and a `note` for each child of a row operator, its operator
//! `invert_rows`, `transpose_rows`, `cross_rows`, `invert_rel`, `transpose_rel` or `cross_rel`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use fuller::futhark_table::{FutharkKingdom, GeneRng};
use fuller::futhark_exec::output_with_timeout_bytes;
use fuller::futhark_batch::{Host, Inputs, KmeansSide, Member, Outcome, Program, Step, PAIR_SIDE_SECONDS};
use fuller::fitness::{Definition, Scheme};
use fuller::futhark_sig::kmeans_converged::{converge, converge_interleaved, converged_work_cap, further_iterations, kmeans_inputs, kmeans_inputs_better, kmeans_inputs_five, run_of_k_payload, sides_of_runs, step_of_exe, step_of_server, ConvergedRun, Server, StepTimer, Stepped, INPUT_NAMES_FIVE, KMEANS_CASES, QUALITY_NAMES, QUALITY_NAMES_BETTER, SPEED_NAMES, WORK_CAP_TIMES_SEED};
use fuller::futhark_sig::{build, function_count, cases_fingerprint, compare, quality_fingerprint_of, compile_c, compile_c_server, load, load_cases, problem, program, program_cases, quality_input_names_of, quality_inputs_of, random_gene, read_binary, sexp, signature, tree_distance, variants, Array, Bridge, Case, FoldEgraph, Instance, Interp, QualityPart, Signature, Split};
use fuller::futhark_wgpu_host::{compile_program_prebuilt, compile_program_prebuilt_staged, prebuild_shim, run_program_bytes, run_program_bytes_staged, CompileStages, RunStages};
use fuller::geneframe::Ty;
use fuller::kingdom_contract::KingdomContract;
use fuller::population_fold::AdfTable;
use fuller::runner::{interleaved_runs, write_aa_gate, AA_RETIMES, median, parse_ops, pick_op, save_checkpoint, slope_check, speed_verdict, FameEntry, FitnessStage, HallOfFame, Ledger, LedgerLine, OperatorTally, TimingSlots};
use fuller::saturated_fold::{fold_saturated, Budget};
use fuller::telemetry::{EventKind, GenerationStats, Mutation, RunInfo, RunTelemetry};
use fuller::typed_gep::{express, write_back, GeneLayout, Tree};
use fuller::futhark_operators::{cross_rel_once, cross_rows_coding_once, cross_rows_once, crossover_once, invert_once, invert_rel_once, invert_rows_coding_once, invert_rows_once, transpose_once, transpose_rel_once, transpose_rows_coding_once, transpose_rows_once};
use fuller::futhark_rings::{build_at, grow, load_pool, published_source, Rings, Source};
use fuller::wgsl::oracle::device::CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    arg(args, name).map_or(Ok(default), |v| v.parse::<T>().map_err(|_| format!("{name}: {v} is not a number")))
}

/// UTC wall time `HH:MM:SSZ`.
fn utc() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) % 86_400;
    format!("{:02}:{:02}:{:02}Z", s / 3600, (s / 60) % 60, s % 60)
}


/// The hash of a program's text, as its identity in the stream and the hall of fame.
fn text_hash(s: &str) -> String {
    text_hash_bytes(s.as_bytes())
}

/// FNV-1a of bytes, 16 hex digits.
fn text_hash_bytes(s: &[u8]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Gpu,
    C,
}

/// How the new programs of a generation are evaluated (`--eval`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalMode {
    /// Each program its own compile and its own processes (the path before 2026-10-28).
    Single,
    /// Every new program of a generation in one compiled program and one process (`fuller::futhark_batch`).
    Batch,
    /// The batch, compiled by the fork's rust-wgpu backend into a Rust module and run IN THIS PROCESS by the runtime
    /// (`fuller::futhark_rust`): WGSL kernels and a Rust host over wgpu-rs, no C, no shim, no process.
    Rust,
}

impl EvalMode {
    /// A generation's new programs are evaluated together (`batch` or `rust`).
    fn batched(self) -> bool {
        matches!(self, EvalMode::Batch | EvalMode::Rust)
    }
    /// The path's name, in the ledger.
    fn label(self) -> &'static str {
        match self {
            EvalMode::Single => "single",
            EvalMode::Batch => "batch",
            EvalMode::Rust => "rust",
        }
    }
}

/// Where an evaluation's wall time went, in seconds: rendering the program text, the Futhark compiler, the WGSL port,
/// `cc`, the surface check, the link, the processes' start, device set-up and pipeline creation, the runs on the
/// stored cases, and the speed timing (stage 2). In batch mode a program's share of its batch: the batch's shared
/// stages divided among its programs, its own runs and timing.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Stages {
    render: f64,
    futhark: f64,
    port: f64,
    cc: f64,
    check: f64,
    link: f64,
    start: f64,
    device: f64,
    pipelines: f64,
    runs: f64,
    timing: f64,
}

impl Stages {
    const NAMES: [&'static str; 11] = ["render", "futhark", "port", "cc", "check", "link", "start", "device", "pipelines", "runs", "timing"];

    fn values(&self) -> [f64; 11] {
        [self.render, self.futhark, self.port, self.cc, self.check, self.link, self.start, self.device, self.pipelines, self.runs, self.timing]
    }

    fn of(v: [f64; 11]) -> Stages {
        Stages { render: v[0], futhark: v[1], port: v[2], cc: v[3], check: v[4], link: v[5], start: v[6], device: v[7], pipelines: v[8], runs: v[9], timing: v[10] }
    }

    fn add(&mut self, o: &Stages) {
        let (a, b) = (self.values(), o.values());
        *self = Stages::of(std::array::from_fn(|i| a[i] + b[i]));
    }

    fn scaled(&self, f: f64) -> Stages {
        let a = self.values();
        Stages::of(std::array::from_fn(|i| a[i] * f))
    }

    fn compiled(c: &CompileStages) -> Stages {
        Stages { futhark: c.futhark, port: c.port, cc: c.cc, check: c.check, link: c.link, ..Stages::default() }
    }

    fn ran(r: &RunStages) -> Stages {
        Stages { start: r.start, device: r.device, pipelines: r.pipelines, runs: r.runs, ..Stages::default() }
    }

    fn total(&self) -> f64 {
        self.values().iter().sum()
    }
}

/// The steps of a program's timed pairs (`missing` of them) in a batch: the process-equivalent the per-candidate path
/// discards for a freshly compiled executable (`fresh`), then each pair INTERLEAVED run by run on one upload of the
/// timing input (`Step::Pair`, `runner::interleaved_runs`), the side that goes first alternating from pair to pair.
fn pair_steps(missing: usize, fresh: bool, runs: usize) -> Vec<Step> {
    let mut v = Vec::with_capacity(missing + 1);
    if fresh {
        v.push(Step::Time { yardstick: false, runs });
    }
    for i in 0..missing {
        v.push(Step::Pair { yard_first: interleaved_runs(i, 1)[0], itself: false });
    }
    v
}

/// The ratios candidate / yardstick of [`pair_steps`]' outcomes: an error anywhere is the timing's error, as one
/// process's error is on the per-candidate path.
fn pair_ratios(outs: &[Outcome], missing: usize, fresh: bool) -> Result<Vec<f64>, String> {
    let mut o = outs.iter();
    if fresh {
        o.next().ok_or("our software has a bug: no discarded timing")?.fastest_ns()?;
    }
    (0..missing).map(|_| o.next().ok_or("our software has a bug: a pair cut short")?.pair_ratio()).collect()
}

/// The input the speed is timed on.
#[derive(Debug, Clone)]
struct TimingInput {
    name: String,
    stdin: Vec<u8>,
    sizes: BTreeMap<String, usize>,
    /// The published program's output on it, when the problem has its own timing input.
    output: Option<Vec<u8>>,
}

impl TimingInput {
    /// The problem's own timing input (`futhark_sig::timing_input`, `Problem::timing_n`) with the published
    /// program's output on it (the yardstick through `futhark c`), or the largest training case.
    /// `with_output` false (the k-means scheme, whose speed is measured in the run and whose timing-case check is not
    /// bit for bit) leaves the published program's output out.
    fn of(p: &fuller::futhark_sig::Problem, sig: &Signature, cases: &[Case], build_dir: &Path, with_output: bool) -> Result<TimingInput, String> {
        if let Some(stdin) = fuller::futhark_sig::timing_input(p)? {
            let inputs = read_binary(&stdin)?;
            let output = if with_output {
                let exe = compile_c(p.yardstick, build_dir)?;
                Some(fuller::futhark_sig::write_binary(&[&fuller::futhark_sig::run_exe(&exe, &stdin)?]))
            } else {
                None
            };
            return Ok(TimingInput { name: format!("{}'s timing input", p.name), sizes: sig.sizes(&inputs)?, stdin, output });
        }
        let i = (0..cases.len()).filter(|i| cases[*i].split == Split::Train).max_by_key(|i| cases[*i].output.as_ref().map_or(0, Array::len) * cases[*i].inputs.iter().map(|a| a.len()).sum::<usize>()).ok_or("no training case")?;
        Ok(TimingInput { name: cases[i].name.clone(), stdin: cases[i].stdin.clone(), sizes: sig.sizes(&cases[i].inputs)?, output: None })
    }

    fn label(&self) -> String {
        let sizes: Vec<String> = self.sizes.iter().map(|(k, v)| format!("{k} = {v}")).collect();
        format!("{} ({})", self.name, sizes.join(", "))
    }
}

/// The A/A sets of the batch path: the published program against itself, the published program as a member against
/// it, and single timings of the published program (ns).
struct AaSets {
    noise: Vec<f64>,
    inrun: Vec<f64>,
    rounds: Vec<f64>,
}

/// The ledger's columns after the text: how the program was evaluated, its batch's size, each stage's milliseconds.
/// Then the fitness inputs, each exactly (Rust's shortest round-trip form, comma-separated): the quality inputs in case
/// order and the speed inputs (stage 2), so two runs' inputs can be compared bit for bit; then the program's
/// behaviour (the hash of its outputs on every case) and whether every output is all zeros (0 or 1).
const LEDGER_STAGE_COLUMNS: [&str; 17] = ["eval", "batch_size", "render_ms", "futhark_ms", "port_ms", "cc_ms", "check_ms", "link_ms", "start_ms", "device_ms", "pipelines_ms", "runs_ms", "timing_ms", "quality_inputs", "speed_inputs", "behaviour", "all_zero"];

/// What a program measured on the stored cases (independent of the stage).
#[derive(Debug, Clone)]
struct Quality {
    /// Three raw measurements a fitness case (differing share, mean, max), in case order: exactness.
    raw: Vec<[f64; 3]>,
    /// The quality inputs (`futhark_sig::quality_inputs`, the parts of `futhark_sig::QUALITY_PARTS` a case), in case order.
    inputs: Vec<f64>,
    exact_train: usize,
    exact_held: usize,
    /// The test cases it is exact on (reported only: a test case is never a fitness input).
    exact_test: usize,
    /// The most node executions on any case.
    work: u128,
    /// The outputs on every case, in case order (empty for a program that did not run).
    outputs: Vec<Array>,
    /// The program's BEHAVIOUR: the hash of its outputs on every case (two programs with the same outputs are one
    /// behaviour, whatever their texts); every program that did not run is the one behaviour `failed`.
    behaviour: String,
}

impl Quality {
    fn perfect(&self) -> bool {
        self.raw.iter().all(|r| *r == [0.0; 3])
    }
    fn inputs(&self) -> Vec<f64> {
        self.inputs.clone()
    }
    /// Every output is all zeros (the dead point random programs crowd at, plan section H).
    fn all_zero(&self) -> bool {
        !self.outputs.is_empty() && self.outputs.iter().all(|a| a.values().all(|x| x == 0.0))
    }
    /// A legal gene's program that did not compile or run (our software has a bug, logged): wrong everywhere.
    fn worst(cases: &[Case], parts: &[QualityPart]) -> Quality {
        let fit: Vec<&Case> = cases.iter().filter(|c| c.split.in_fitness()).collect();
        let inputs = fit.iter().flat_map(|c| vec![1.0; c.output.as_ref().map_or(0, |o| quality_input_names_of(parts, o).len())]).collect();
        Quality { raw: vec![[1.0; 3]; fit.len()], inputs, exact_train: 0, exact_held: 0, exact_test: 0, work: 0, outputs: Vec::new(), behaviour: "failed".into() }
    }
    /// The quality of a program's outputs on the cases: the fitness's inputs and exactness from the training and
    /// held-out cases alone; the test cases only counted (`exact_test`).
    fn of(outs: Vec<Array>, cases: &[Case], work: u128, parts: &[QualityPart]) -> Quality {
        let mut q = Quality { raw: Vec::with_capacity(cases.len()), inputs: Vec::new(), exact_train: 0, exact_held: 0, exact_test: 0, work, outputs: Vec::new(), behaviour: String::new() };
        for (got, c) in outs.iter().zip(cases) {
            let Some(want) = &c.output else { continue };
            let r = compare(got, want);
            if c.split.in_fitness() {
                q.inputs.extend(quality_inputs_of(parts, got, want));
                q.raw.push(r);
            }
            if r == [0.0; 3] {
                match c.split {
                    Split::Train => q.exact_train += 1,
                    Split::Held => q.exact_held += 1,
                    Split::Test => q.exact_test += 1,
                }
            }
        }
        q.behaviour = text_hash_bytes(&fuller::futhark_sig::write_binary(&outs.iter().collect::<Vec<_>>()));
        q.outputs = outs;
        q
    }
}

/// An individual's fitness in the stage it was scored in.
#[derive(Debug, Clone)]
struct Eval {
    fitness: f64,
    theta_q: f64,
    theta_s: f64,
    quality: Quality,
    /// The speed inputs (stage 2) and the median ratio candidate / yardstick.
    speed: Vec<f64>,
    ratio: f64,
}

#[derive(Clone)]
struct Individual {
    id: u64,
    parents: Vec<u64>,
    born: u32,
    operator: String,
    gene: Vec<u32>,
    tree: Tree,
    source: String,
    eval: Option<Eval>,
}

impl Individual {
    fn fit(&self) -> f64 {
        self.eval.as_ref().map_or(f64::INFINITY, |e| e.fitness)
    }
    /// The identity diversity is kept on (plan section D, faults 6 and 21): in stage 1 the program's BEHAVIOUR (its
    /// outputs), so two programs with the same outputs are one parent, one elite; in stage 2 every program is exact
    /// and has the same outputs, and its text is the identity, as before.
    fn identity(&self, stage: FitnessStage) -> &str {
        match (&self.eval, stage.measures_speed()) {
            (Some(e), false) => e.quality.behaviour.as_str(),
            _ => self.source.as_str(),
        }
    }
}

/// A batched program's quality outputs (or the failure that names it), its share of its batch's stages, and the
/// batch's size.
type Batched = (Result<Vec<Array>, String>, Stages, usize);

/// The fitness machinery: the cases, the yardstick's executable, the caches by program text.
struct Judge<'a> {
    k: &'a FutharkKingdom,
    sig: &'a Signature,
    /// Every case's input, in case order, as one stdin for the quality program.
    all_stdin: Vec<u8>,
    interp: Interp<'a>,
    cases: Vec<Case>,
    /// THE FITNESS SCHEME (`--fitness`, `fuller::fitness`, family `futhark_sig`): the quality parts and the stage-2
    /// rule.
    scheme: &'static Scheme,
    /// The scheme's quality parts (`Scheme::quality_parts`).
    parts: &'static [QualityPart],
    sizes: Vec<BTreeMap<String, usize>>,
    /// The input the speed is timed on: the problem's own large timing input (never a quality case) or, without one,
    /// the largest training case.
    timing: TimingInput,
    /// The problem's timing bound (`futhark_sig::timing_work_cap`): a program whose work at the timing input is over it
    /// is never timed.
    timing_cap: u128,
    /// Programs over the work cap at the timing size, not timed (no speed credit).
    over_cap_untimed: u64,
    backend: Backend,
    runs: usize,
    yard_exe: Option<PathBuf>,
    /// The shim, built once at the start: every candidate links with plain cc, no cargo, no lock.
    shim: PathBuf,
    work_dir: PathBuf,
    quality_cache: HashMap<String, Quality>,
    /// The timing slots, kept by program text (`--pairs` of them, each its own speed input).
    slots: TimingSlots,
    /// FITNESS EVALUATIONS, the budget's unit (plan section E, faults 7 and 8): programs not scored before, by text.
    /// An elite carried over, a repeat and a pump child equal to a program already scored are free.
    evaluations: u64,
    /// Every individual scored, repeats included (reported beside `evaluations`).
    scored: u64,
    /// Programs compiled and run (cache misses).
    compiled: u64,
    hits: u64,
    bugs: u64,
    /// Of the bugs, the branch compiler's "Cannot handle SetMem statement" (a known compiler limitation).
    setmem: u64,
    /// Of the bugs, Futhark's internal compiler errors (each a kingdom bug, logged with its program).
    ice: u64,
    bug_log: PathBuf,
    max_work: u128,
    mode: EvalMode,
    /// The batch host (`--eval batch`).
    host: Option<Host<'a>>,
    /// `--eval single`'s TIMING HOST: a candidate's timed pairs are its own library build and its own process, the
    /// batch driver's interleaved pairs on one upload (`Step::Pair`), so the three paths time alike. Measured
    /// 2026-10-31: Futhark's own command line re-uploads the timing input every run, and the GPU's clock stepped down
    /// in the gaps (the runner's A/A gate, 9 of 22 pairs within 5 % on histogram). Futhark's command line still gives
    /// the quality outputs.
    timer: Option<Host<'a>>,
    /// Batch mode: the quality outputs of the programs batched this generation, by text, with each one's share of its
    /// batch's stages and the batch's size; taken when the program is scored (so the counts, the bugs and the ledger
    /// are the per-candidate path's).
    batched_q: HashMap<String, Batched>,
    /// Batch mode: the timed pairs of the programs batched this generation, by text, with each one's timing seconds.
    batched_t: HashMap<String, (Result<Vec<f64>, String>, f64)>,
    /// The stages of this generation (the batches' and the single evaluations' wall time).
    gen_stages: Stages,
    /// Batches halved although no program in them was at fault (a limit of the batch, logged in `bugs.log`).
    batch_splits: u64,
    /// The last evaluation's path (`single`, `batch`, `hit`), its batch's size and its stages, for the ledger.
    last: (&'static str, usize, Stages),
    /// THE K-MEANS SCHEME's judge (`futhark-kmeans-better`, or the retired `futhark-kmeans-qqs`,
    /// `futhark-kmeans-quality-first` and `futhark-kmeans-converged`), when that is the scheme.
    converged: Option<ConvergedJudge>,
    /// The k-means scheme's speed group of every program scored, by text (its quality group is in `quality_cache`).
    converged_speed: HashMap<String, Vec<f64>>,
    /// The k-means scheme's candidates over the work cap on a fitness case, not run, scored worst.
    over_cap_unrun: u64,
    /// The k-means scheme's candidate runs that did not finish (a NaN or infinite centroid, a wrong shape, a step that
    /// failed): the candidate's worst score on that case, not a bug of our software.
    unfinished: u64,
    /// THE REPETITION COUNT of a flat scheme (`--quality-weight`): the quality inputs are entered this many times in
    /// the one vector the HFF reads (`fitness::qqs`); 1 for a scheme that is not flat.
    quality_weight: usize,
    /// THE CLIFF'S WIDTH ε of `futhark-kmeans-better` (`--quality-cliff`, `fitness::quality_cliff`); unread by every
    /// other scheme.
    quality_cliff: f64,
    /// `--kmeans-step process` (the c backend): one process an iteration instead of a server, for the measurement.
    step_process: bool,
    /// The resident-sample path: the runs of the programs batched this generation, by text, with each one's share of
    /// the batch's stages (`Judge::prefetch`), taken when the program is scored.
    batched_k: HashMap<String, (BatchedKmeans, Stages)>,
    /// The resident sample's upload seconds this generation (0 once resident) and the runs that found it resident.
    upload_seconds: f64,
    resident_hits: u64,
}

/// HOW A COMPILED K-MEANS PROGRAM IS STEPPED: on the GPU one process of the shim's executable an iteration (the
/// program's own time its GPU timestamps); on the c backend a Futhark server kept alive across the iterations and the
/// cases (`kmeans_converged::Server`, the program's own time the call's `runtime:`).
enum Stepper {
    Exe(PathBuf, StepTimer),
    Server(Server),
}

/// A step on a case, as `converge` takes it, boxed (the two steppers' closures differ in type).
type BoxedStep<'s> = Box<dyn FnMut(&Array) -> Result<Stepped, String> + 's>;

impl Stepper {
    /// The step on a case, as `converge` takes it; a server loads the case's points once.
    fn step<'s>(&'s mut self, case: &'s Case) -> Result<BoxedStep<'s>, String> {
        match self {
            Stepper::Exe(exe, timer) => Ok(Box::new(step_of_exe(exe, &case.inputs[0], *timer))),
            Stepper::Server(s) => {
                s.load_points(&case.name, &case.inputs[0])?;
                Ok(Box::new(step_of_server(s, &case.name)))
            }
        }
    }
}

/// The k-means scheme's state: the seed's compiled program (a server on the c backend, kept for the whole run) and
/// its run on every fitness case at the start (the control and the report; every input is against the seed's run
/// interleaved with the candidate's), the work cap of every fitness case (`converged_work_cap` of the seed's work on
/// it), how many interleaved pairs each candidate is run against the seed, and the test cases (never a fitness
/// input) the champion is run on.
struct ConvergedJudge {
    /// The seed's stepper (`--eval single`); none on the resident-sample path, where the seed is an entry of every
    /// generation's module.
    seed: Option<Stepper>,
    /// The seed's run on each fitness case at the start, in case order (`Case::split.in_fitness()`).
    seed_runs: Vec<ConvergedRun>,
    /// The work cap of each fitness case, in case order.
    work_caps: Vec<u128>,
    pairs: usize,
    /// The shared points files of the servers (`Server::start`'s `points_dir`).
    points_dir: PathBuf,
    /// THE RESIDENT-SAMPLE PATH (`--eval rust`; Andrew, 2026-11-03): every stored case's row range `(start, n)` of
    /// the one sample, in case order, checked at the start against the case's points. The sample is uploaded once a
    /// session and kept on the device; a generation's programs are one module; each candidate's cases run against
    /// the resident sample interleaved with the seed's entry (`futhark_batch::Step::Kmeans`).
    cuts: Option<Vec<(usize, usize)>>,
}

/// A k-means program's runs on the resident-sample path, by its text: the (candidate, seed) runs a case, or the
/// program's failure (a compile the module refused, found by the batch's halving).
type BatchedKmeans = Result<Vec<(ConvergedRun, ConvergedRun)>, String>;

/// THE CHAMPION ON A TEST CASE (the round-2 review's N2): the candidate's and the seed's runs of one interleaving on
/// a case that is never a fitness input, with the candidate's four inputs against that seed run.
struct TestCaseRun {
    case: String,
    candidate: ConvergedRun,
    seed: ConvergedRun,
    quality: Vec<f64>,
    speed: Vec<f64>,
}

impl TestCaseRun {
    /// One line: the inputs, then the raw measures of both sides.
    fn line(&self) -> String {
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
        let raw = |r: &ConvergedRun| format!("inertia {:.5} log10p {:.2} (all {:.2}, {} at centroid) seconds {:.4} iterations {} converged {} finished {}", r.inertia, r.mean_log10_p, r.mean_log10_p_all, r.coincident, r.seconds, r.iterations, r.converged, r.finished);
        format!("{}: quality [{}] speed [{}]; champion {}; seed {}", self.case, fmt(&self.quality), fmt(&self.speed), raw(&self.candidate), raw(&self.seed))
    }
}

impl<'a> Judge<'a> {
    /// The fitness scheme's quality parts.
    fn parts(&self) -> &'static [QualityPart] {
        self.parts
    }

    /// A candidate's quality program compiled per candidate (`--eval single`), with each stage's wall time (the c
    /// backend's compile is all `futhark`).
    fn compile_staged(&self, source: &str, dir: &Path) -> Result<(PathBuf, Stages), String> {
        match self.backend {
            Backend::Gpu => compile_program_prebuilt_staged(source, dir, &self.shim).map(|(e, c)| (e, Stages::compiled(&c))),
            Backend::C => {
                let t = Instant::now();
                let e = compile_c(source, dir)?;
                Ok((e, Stages { futhark: t.elapsed().as_secs_f64(), ..Stages::default() }))
            }
        }
    }

    /// Run the quality program (every case in one process) and read the outputs, one a case, with the process's
    /// stages.
    fn run_all(&self, exe: &Path) -> Result<(Vec<Array>, Stages), String> {
        let (raw, stages) = match self.backend {
            Backend::Gpu => {
                let (out, report, rs) = run_program_bytes_staged(exe, &self.all_stdin, false, 1)?;
                if report.backend != "Metal" {
                    return Err(format!("the program ran on {}, not Metal", report.backend));
                }
                (out, Stages::ran(&rs))
            }
            Backend::C => {
                let t = Instant::now();
                let out = output_with_timeout_bytes(std::process::Command::new(exe).arg("-b"), std::time::Duration::from_secs(120), Some(&self.all_stdin))?;
                if !out.status.success() {
                    return Err(format!("{} failed:\n{}", exe.display(), String::from_utf8_lossy(&out.stderr)));
                }
                (out.stdout, Stages { runs: t.elapsed().as_secs_f64(), ..Stages::default() })
            }
        };
        let v = read_binary(&raw)?;
        if v.len() != self.cases.len() {
            return Err(format!("the quality program wrote {} values for {} cases", v.len(), self.cases.len()));
        }
        Ok((v, stages))
    }

    /// The most node executions of a tree on any stored case.
    fn work_of(&self, tree: &Tree) -> Result<u128, String> {
        let mut work = 0u128;
        for s in &self.sizes {
            work = work.max(self.interp.work(tree, s)?);
        }
        Ok(work)
    }

    /// BATCH MODE: every program of `items` that will need a compile this generation — its quality not yet known,
    /// or (in a stage with speed) its timing slots not yet filled — in one batch (`fuller::futhark_batch`): one
    /// compile and one process (more only when the halving must find a program the compiler refuses). The results are
    /// kept by text and taken when each program is scored, so selection, the counts and the bugs are the
    /// per-candidate path's. Programs over the work cap are left out: scoring them stops the run, as before.
    fn prefetch(&mut self, items: &[(&Tree, &str)], stage: FitnessStage) -> Result<(), String> {
        if !self.mode.batched() {
            return Ok(());
        }
        if self.converged.is_some() {
            return self.prefetch_kmeans(items);
        }
        let t0 = Instant::now();
        let mut seen: HashSet<&str> = HashSet::new();
        let mut members: Vec<Member> = Vec::new();
        let mut keys: Vec<(String, bool, usize, bool)> = Vec::new();
        for (tree, source) in items {
            if !seen.insert(source) {
                continue;
            }
            let mut steps = Vec::new();
            let quality = !self.quality_cache.contains_key(*source) && !self.batched_q.contains_key(*source) && self.work_of(tree)? <= CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT;
            if quality {
                steps.push(Step::Quality);
            }
            let (mut missing, mut fresh) = (0, false);
            if stage.measures_speed() && self.backend == Backend::Gpu && !self.batched_t.contains_key(*source) {
                let m = self.slots.missing(source);
                if m > 0 && self.interp.work(tree, &self.timing.sizes)? <= self.timing_cap {
                    (missing, fresh) = (m, m == self.slots.slots());
                    steps.extend(pair_steps(missing, fresh, self.runs));
                }
            }
            if steps.is_empty() {
                continue;
            }
            members.push(Member { program: Program::Tree(tree), steps });
            keys.push((source.to_string(), quality, missing, fresh));
        }
        if members.is_empty() {
            return Ok(());
        }
        let host = self.host.as_ref().ok_or("our software has a bug: batch mode with no batch host")?;
        let r = host.run(&members)?;
        let n = members.len();
        let c = &r.cost;
        // The batch's wall time: the compiles (a halving's failed ones in `futhark`) and the processes (their start,
        // device set-up, shader and pipeline creation, and the rest). The commands' time is each member's own: its
        // quality run in `runs`, its pairs in `timing`, the pipelines created inside them in `pipelines`.
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        let (own_q, own_t) = (sum(&c.member_quality_seconds), sum(&c.member_timing_seconds));
        let (pipe_q, pipe_t) = (sum(&c.member_quality_pipeline_seconds), sum(&c.member_timing_pipeline_seconds));
        let mut shared = Stages::compiled(&c.compile);
        shared.render = c.render;
        shared.futhark += c.failed_compile_seconds;
        shared.start = c.run.start;
        shared.device = c.run.device;
        shared.pipelines = (c.run.pipelines - pipe_q - pipe_t).max(0.0);
        shared.runs = (c.run.runs - (own_q - pipe_q) - (own_t - pipe_t)).max(0.0);
        let mut whole = shared;
        whole.pipelines = c.run.pipelines;
        whole.timing = (own_t - pipe_t).max(0.0);
        whole.runs = (c.run.runs - whole.timing).max(0.0);
        self.gen_stages.add(&whole);
        self.batch_splits += c.splits.len() as u64;
        for why in &c.splits {
            // Not a candidate's bug (every program of the halves was scored): our software's limit on a batch, named.
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.bug_log) {
                let _ = writeln!(f, "==== {} our software has a bug: a batch had to be halved (no candidate is charged; the halves were scored)\n{why}", utc());
            }
        }
        let share = shared.scaled(1.0 / n as f64);
        for (i, (key, quality, missing, fresh)) in keys.into_iter().enumerate() {
            let outs = &r.outcomes[i];
            let mut mine = share;
            mine.runs += c.member_quality_seconds[i] - c.member_quality_pipeline_seconds[i];
            mine.pipelines += c.member_quality_pipeline_seconds[i] + c.member_timing_pipeline_seconds[i];
            mine.timing = c.member_timing_seconds[i] - c.member_timing_pipeline_seconds[i];
            let mut at = 0;
            if quality {
                let q = match &outs[0] {
                    Outcome::Quality(v) => Ok(v.clone()),
                    Outcome::Failed(e) => Err(format!("{e}\n(in a batch of {n}: the halving found this program at fault)")),
                    other => return Err(format!("our software has a bug: a quality step's outcome {other:?}")),
                };
                // The timing seconds are taken with the pairs (`evaluate`), never twice.
                let mut qs = mine;
                qs.timing = 0.0;
                self.batched_q.insert(key.clone(), (q, qs, n));
                at = 1;
            }
            if missing > 0 {
                self.batched_t.insert(key, (pair_ratios(&outs[at..], missing, fresh), mine.timing));
            }
        }
        println!("# {} batch: {n} programs, {} compile(s) ({} failed, {} not finished within the limit), {} process(es), {} halving(s), {:.1} s", utc(), c.compiles, c.failed_compiles, c.unfinished_compiles, c.processes, c.splits.len(), t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// THE RESIDENT-SAMPLE PATH'S GENERATION (`--eval rust`; Andrew, 2026-11-03): every program of `items` not scored
    /// before and under the work cap on every fitness case, in ONE module with the seed's entry, run against the
    /// resident sample on every fitness case `pairs` times interleaved with the seed; the runs kept by text and taken
    /// when the program is scored (`evaluate_converged`). The batch's stages are shared out as the quality batch's
    /// are; the upload (once a session) and the resident hits are counted for the generation's report.
    fn prefetch_kmeans(&mut self, items: &[(&Tree, &str)]) -> Result<(), String> {
        let t0 = Instant::now();
        let (caps, pairs) = self.converged.as_ref().map(|cj| (cj.work_caps.clone(), cj.pairs)).ok_or("our software has a bug: a k-means batch with no converged judge")?;
        let fit: Vec<usize> = (0..self.cases.len()).filter(|i| self.cases[*i].split.in_fitness()).collect();
        let mut seen: HashSet<&str> = HashSet::new();
        let mut programs: Vec<Program> = Vec::new();
        let mut keys: Vec<String> = Vec::new();
        for (tree, source) in items {
            if !seen.insert(source) || self.quality_cache.contains_key(*source) || self.batched_k.contains_key(*source) {
                continue;
            }
            let over = fit.iter().enumerate().any(|(slot, i)| self.interp.work(tree, &self.sizes[*i]).is_ok_and(|w| w > caps[slot]));
            if over {
                continue;
            }
            programs.push(Program::Tree(tree));
            keys.push(source.to_string());
        }
        if programs.is_empty() {
            return Ok(());
        }
        let (runs, c) = self.kmeans_batch(&programs, &fit, pairs)?;
        let n = programs.len();
        let mut shared = Stages::compiled(&c.compile);
        shared.render = c.render;
        shared.futhark += c.failed_compile_seconds;
        shared.start = c.run.start;
        shared.device = c.run.device;
        let own: f64 = c.member_timing_seconds.iter().sum();
        shared.pipelines = c.run.pipelines;
        shared.runs = (c.run.runs - own).max(0.0);
        let mut whole = shared;
        whole.timing = own;
        self.gen_stages.add(&whole);
        self.upload_seconds += c.upload_seconds;
        self.resident_hits += c.resident_hits as u64;
        self.batch_splits += c.splits.len() as u64;
        for why in &c.splits {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.bug_log) {
                let _ = writeln!(f, "==== {} our software has a bug: a batch had to be halved (no candidate is charged; the halves were scored)\n{why}", utc());
            }
        }
        let share = shared.scaled(1.0 / n as f64);
        for (i, (key, r)) in keys.into_iter().zip(runs).enumerate() {
            let mut mine = share;
            mine.timing = c.member_timing_seconds[i] - c.member_timing_pipeline_seconds[i];
            mine.pipelines += c.member_timing_pipeline_seconds[i];
            self.batched_k.insert(key, (r, mine));
        }
        println!("# {} k-means batch (the resident sample): {n} programs in one module, {} compile(s) ({} failed, {} not finished within the limit, {} compiled with the fusion bound), {} run(s), {} halving(s); compile {:.1} s, upload {:.3} s ({} upload(s), {} resident), run {:.1} s; {:.1} s in all", utc(), c.compiles, c.failed_compiles, c.unfinished_compiles, c.bounded_fusion_compiles, c.processes, c.splits.len(), c.compile.futhark + c.compile.cc, c.upload_seconds, c.uploads, c.resident_hits, c.run.total, t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// BATCH MODE: the A/A sets in one process: the published program against itself (a discarded pair, then 11),
    /// and, for P, the published program as a member (wrapped as a candidate's text is) against it through the
    /// slots' scheme (`inrun_pairs`, the first process-equivalent discarded); with `rounds`, that many single timings
    /// of the published program (the gate's). Returns (noise, in-run A/A, rounds in ns).
    fn aa_batch(&self, yardstick: &str, inrun_pairs: usize, rounds: usize) -> Result<AaSets, String> {
        let host = self.timing_host()?;
        // A discarded pair, then 11, each interleaved run by run (`Step::Pair`, the published program against itself).
        let mut steps: Vec<Step> = (0..12).map(|i| Step::Pair { yard_first: interleaved_runs(i, 1)[0], itself: true }).collect();
        steps.extend(vec![Step::Time { yardstick: true, runs: self.runs }; rounds]);
        let mut members = vec![Member { program: Program::Text(yardstick), steps }];
        if inrun_pairs > 0 {
            members.push(Member { program: Program::Text(yardstick), steps: pair_steps(inrun_pairs, true, self.runs) });
        }
        let r = host.run(&members)?;
        let o = &r.outcomes[0];
        o[0].pair_ratio()?;
        let noise = o[1..12].iter().map(Outcome::pair_ratio).collect::<Result<Vec<_>, _>>()?;
        let rounds = o[12..].iter().map(Outcome::fastest_ns).collect::<Result<Vec<_>, _>>()?;
        let inrun = if inrun_pairs > 0 { pair_ratios(&r.outcomes[1], inrun_pairs, true)? } else { Vec::new() };
        Ok(AaSets { noise, inrun, rounds })
    }

    /// THE A/A MEDIANS the speed verdict reads (`runner::speed_verdict`, fault 13): the PUBLISHED program re-timed
    /// exactly as the champion is (`champion_batch`, 11 pairs, a process each), `retimes` times. Like for like: the
    /// same statistic (a median of 11) by the same path, so under "no faster" the champion's median is one more
    /// draw of the same distribution and falls below all `retimes` of them with probability `1/(retimes + 1)`.
    /// Measured 2026-11-01, histogram: the published program as the champion gave median 0.9987 with 11 of 11
    /// pairs faster against yardstick-against-itself medians 0.9994-1.0012 (a false "beat"), because each timing
    /// process carries its own offset of about 0.1 %: so the A/A sets the run timed at its start (another path)
    /// are not used here.
    fn aa_medians(&self, reference: &Tree, retimes: usize) -> Result<Vec<f64>, String> {
        (0..retimes).map(|_| self.champion_batch(reference, 11).map(|(r, _)| median(&r))).collect()
    }

    /// BATCH MODE: the champion's re-time (`pairs`, the first process-equivalent discarded) and its output on the
    /// timing input, in one process.
    fn champion_batch(&self, tree: &Tree, pairs: usize) -> Result<(Vec<f64>, Array), String> {
        let host = self.timing_host()?;
        let mut steps = pair_steps(pairs, true, self.runs);
        steps.push(Step::Output);
        let r = host.run(&[Member { program: Program::Tree(tree), steps }])?;
        let o = &r.outcomes[0];
        let out = match o.last() {
            Some(Outcome::Output(a)) => a.clone(),
            other => return Err(format!("the champion's output on the timing input: {other:?}")),
        };
        Ok((pair_ratios(&o[..o.len() - 1], pairs, true)?, out))
    }

    /// The host that times pairs: the batch host (`--eval batch`, `--eval rust`) or the single path's timing host.
    fn timing_host(&self) -> Result<&Host<'a>, String> {
        match self.mode {
            EvalMode::Single => self.timer.as_ref().ok_or_else(|| "our software has a bug: --eval single times on the GPU with no timing host (the c backend has no GPU timing)".to_string()),
            EvalMode::Batch | EvalMode::Rust => self.host.as_ref().ok_or_else(|| "our software has a bug: batch mode with no batch host".to_string()),
        }
    }

    /// `--eval single`: one candidate's `missing` timed pairs (`fresh`: its first process-equivalent discarded) in its
    /// own build and process, with the build's stages.
    fn time_single(&self, tree: &Tree, missing: usize, fresh: bool) -> Result<(Vec<f64>, Stages), String> {
        let r = self.timing_host()?.run(&[Member { program: Program::Tree(tree), steps: pair_steps(missing, fresh, self.runs) }])?;
        let st = Stages::compiled(&r.cost.compile);
        Ok((pair_ratios(&r.outcomes[0], missing, fresh)?, st))
    }

    fn log_bug(&mut self, msg: &str, source: &str) {
        self.bugs += 1;
        // The branch compiler's own limitation, its own category: a SetMem statement in a GPU kernel (the gap of
        // docs/PROPOSAL_memory_views_setmem.md).
        if msg.contains("Cannot handle SetMem statement") {
            self.setmem += 1;
        }
        if msg.to_lowercase().contains("internal compiler error") {
            self.ice += 1;
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.bug_log) {
            let _ = writeln!(f, "==== {} our software has a bug: a legal gene's program failed\n{msg}\n---- program\n{source}", utc());
        }
    }

    /// The work guard, then the quality program's outputs: its own compile and process (single mode), or its batch's
    /// (batch mode, `Judge::prefetch`).
    fn quality(&mut self, tree: &Tree, source: &str) -> Result<Quality, String> {
        let work = self.work_of(tree)?;
        if work > CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT {
            return Err(format!("THE WORK GUARD: a candidate runs {work} node executions on a case, over the per-submit cap {CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT}; the experiment's sizes are misconfigured for this layout\n{source}"));
        }
        self.max_work = self.max_work.max(work);
        let t = Instant::now();
        let batch = program_cases(self.k, self.sig, tree, self.cases.len())?;
        let render = t.elapsed().as_secs_f64();
        let got = match self.mode {
            EvalMode::Single => {
                let dir = self.work_dir.join("quality");
                let mut st = Stages { render, ..Stages::default() };
                let r = self.compile_staged(&batch, &dir).and_then(|(exe, cs)| {
                    st.add(&cs);
                    let (o, rs) = self.run_all(&exe)?;
                    st.add(&rs);
                    Ok(o)
                });
                self.gen_stages.add(&st);
                self.last = ("single", 1, st);
                r
            }
            EvalMode::Batch | EvalMode::Rust => {
                let (r, st, n) = self.batched_q.remove(source).ok_or_else(|| format!("our software has a bug: a program scored in batch mode that no batch held\n{source}"))?;
                self.last = (self.mode.label(), n, st);
                r
            }
        };
        match got {
            Ok(o) => Ok(Quality::of(o, &self.cases, work, self.parts())),
            Err(e) => {
                self.log_bug(&e, &batch);
                Ok(Quality::worst(&self.cases, self.parts()))
            }
        }
    }

    /// The k-means scheme's inputs a fitness case: (quality, speed) counts.
    fn converged_widths(&self) -> (usize, usize) {
        match self.scheme.definition {
            Definition::FutharkConvergedGated | Definition::FutharkConvergedQqs => (QUALITY_NAMES.len(), SPEED_NAMES.len()),
            Definition::FutharkKmeansBetter => (QUALITY_NAMES_BETTER.len(), SPEED_NAMES.len()),
            _ => (INPUT_NAMES_FIVE.len(), 0),
        }
    }

    /// The two input groups of one case under the run's k-means scheme (`futhark-kmeans-better`'s three cliff inputs
    /// and two speed inputs, `kmeans_inputs_better`; the flat and gated schemes' two and two, `kmeans_inputs`), the
    /// candidate's run against the seed's of the same interleaving.
    fn case_inputs(&self, cand: &ConvergedRun, seed: &ConvergedRun) -> Result<fuller::futhark_sig::kmeans_converged::Inputs, String> {
        if self.scheme.uses_cliff() {
            kmeans_inputs_better(cand, seed, self.quality_cliff)
        } else {
            Ok(kmeans_inputs(cand, seed))
        }
    }

    /// The k-means scheme's compiled candidate: on the c backend a server (`futhark c --server`, started here and
    /// kept for the candidate's cases), on the GPU the shim's executable, with the compile's stages.
    fn compile_converged(&self, source: &str, dir: &Path) -> Result<(Stepper, Stages), String> {
        match self.backend {
            Backend::Gpu => compile_program_prebuilt_staged(source, dir, &self.shim).map(|(e, c)| (Stepper::Exe(e, StepTimer::Gpu), Stages::compiled(&c))),
            Backend::C if self.step_process => {
                // `--kmeans-step process`: the path before 2026-11-03, one process an iteration, kept for the
                // throughput measurement beside the server.
                let t = Instant::now();
                let exe = compile_c(source, dir)?;
                Ok((Stepper::Exe(exe, StepTimer::FutharkC), Stages { futhark: t.elapsed().as_secs_f64(), ..Stages::default() }))
            }
            Backend::C => {
                let t = Instant::now();
                let exe = compile_c_server(source, dir)?;
                let points_dir = self.converged.as_ref().map_or_else(|| self.work_dir.join("points"), |cj| cj.points_dir.clone());
                let s = Server::start(&exe, &dir.join("server"), &points_dir)?;
                Ok((Stepper::Server(s), Stages { futhark: t.elapsed().as_secs_f64(), ..Stages::default() }))
            }
        }
    }

    /// The candidate and the seed run INTERLEAVED on `case`, `pairs` times, the seed's run of that interleaving the
    /// yardstick; the seed's run not converging beside a candidate is our bug. With `further` (`futhark-kmeans-better`)
    /// the seed's step is then run on from each side's final centroids (`kmeans_converged::further_iterations`): a
    /// finished side's `further_iterations`.
    fn converged_pair(cj: &mut ConvergedJudge, candidate: &mut Stepper, case: &Case, further: bool) -> Result<(ConvergedRun, ConvergedRun), String> {
        let (points, start) = (&case.inputs[0], &case.inputs[1]);
        let pairs = cj.pairs;
        let seed_stepper = cj.seed.as_mut().ok_or("our software has a bug: a stepped pair on the resident-sample path")?;
        let (mut cand, mut seed) = converge_interleaved(points, start, pairs, candidate.step(case)?, seed_stepper.step(case)?)?;
        if !seed.converged {
            return Err(format!("our software has a bug: the seed's run beside a candidate on {} did not converge ({} iterations, finished {}: {})", case.name, seed.iterations, seed.finished, seed.failure));
        }
        if further {
            seed.further_iterations = Some(further_iterations(points, &seed.centroids, seed_stepper.step(case)?)?);
            if cand.finished {
                cand.further_iterations = Some(further_iterations(points, &cand.centroids, seed_stepper.step(case)?)?);
            }
        }
        Ok((cand, seed))
    }

    /// THE RESIDENT-SAMPLE BATCH (`--eval rust`): every program of `members` with a k-means step on each case of
    /// `case_idx` (`pairs` interleaved pairs against the seed's entry), one module, one run of the in-process host on
    /// the resident sample; each program's (candidate, seed) runs a case, in `case_idx` order, read from the `k`
    /// records (`run_of_k_payload`, the medians of the pairs as `converge_interleaved` takes them), or its failure. A
    /// seed side that failed or did not converge is our bug.
    fn kmeans_batch(&mut self, members: &[Program], case_idx: &[usize], pairs: usize) -> Result<(Vec<BatchedKmeans>, fuller::futhark_batch::Cost), String> {
        let cuts = self.converged.as_ref().and_then(|cj| cj.cuts.clone()).ok_or("our software has a bug: a resident-sample batch with no cuts")?;
        let host = self.host.as_ref().ok_or("our software has a bug: the resident-sample path with no in-process host")?;
        let further = self.scheme.uses_cliff();
        let steps: Vec<Step> = case_idx.iter().map(|i| Step::Kmeans { case: *i, start: cuts[*i].0, n: cuts[*i].1, pairs, further }).collect();
        let ms: Vec<Member> = members.iter().map(|p| Member { program: *p, steps: steps.clone() }).collect();
        let r = host.run(&ms)?;
        let mut out = Vec::with_capacity(members.len());
        for outs in &r.outcomes {
            let mut runs = Vec::with_capacity(case_idx.len());
            let mut failed: Option<String> = None;
            for (o, i) in outs.iter().zip(case_idx) {
                let c = &self.cases[*i];
                let sides = match o {
                    Outcome::Kmeans(v) => v,
                    Outcome::Failed(e) => {
                        failed = Some(e.clone());
                        break;
                    }
                    other => return Err(format!("our software has a bug: a k-means step's outcome {other:?}")),
                };
                let (mut cand, mut seed) = (Vec::new(), Vec::new());
                for KmeansSide { candidate, result } in sides {
                    match result {
                        Ok((payload, ns)) => {
                            let run = run_of_k_payload(&c.inputs[0], &c.inputs[1], payload, *ns)?;
                            if *candidate { cand.push(run) } else { seed.push(run) }
                        }
                        // The seed's continuation from the candidate's final centroids failing is the seed's
                        // program failing: our bug, never the candidate's.
                        Err(e) if e.contains("the further entry") => return Err(format!("our software has a bug: the seed's run on from a side's final centroids on {} failed in the host: {e}", c.name)),
                        // The candidate's side failing in the host (a panic, a lost device) is its run not finishing.
                        Err(e) if *candidate => cand.push(run_of_k_payload(&c.inputs[0], &c.inputs[1], &fuller::futhark_sig::kmeans_converged::k_payload(0, false, false, 0.0, 0.0, 0.0, format!("the step failed in the host: {e}").as_bytes()), 0)?),
                        Err(e) => return Err(format!("our software has a bug: the seed's side on {} failed in the host: {e}", c.name)),
                    }
                }
                let (cand, seed) = sides_of_runs(cand, seed)?;
                if !seed.converged {
                    return Err(format!("our software has a bug: the seed's run beside a candidate on {} did not converge ({} iterations, finished {}: {})", c.name, seed.iterations, seed.finished, seed.failure));
                }
                if further && (seed.further_iterations.is_none() || (cand.finished && cand.further_iterations.is_none())) {
                    return Err(format!("our software has a bug: a k-means record on {} without the further iterations its command asked for", c.name));
                }
                runs.push((cand, seed));
            }
            out.push(match failed { Some(e) => Err(e), None => Ok(runs) });
        }
        Ok((out, r.cost))
    }

    /// THE CHAMPION ON THE TEST CASES (the round-2 review's N2): the program compiled again (its build directory
    /// is every candidate's) and run interleaved with the seed on every case that is not a fitness input, with its
    /// four inputs against that seed run; on the resident-sample path its own module and run. Reported, never a
    /// fitness input.
    fn converged_on_test(&mut self, tree: &Tree, source: &str) -> Result<Vec<TestCaseRun>, String> {
        let test: Vec<usize> = (0..self.cases.len()).filter(|i| !self.cases[*i].split.in_fitness()).collect();
        let mut out = Vec::new();
        if self.mode == EvalMode::Rust {
            let pairs = self.converged.as_ref().map_or(1, |cj| cj.pairs);
            let (runs, _) = self.kmeans_batch(&[Program::Tree(tree)], &test, pairs)?;
            let runs = runs.into_iter().next().ok_or("our software has a bug: a test batch of no member")??;
            for ((cand, seed), i) in runs.into_iter().zip(&test) {
                let inp = self.case_inputs(&cand, &seed)?;
                out.push(TestCaseRun { case: self.cases[*i].name.clone(), candidate: cand, seed, quality: inp.quality, speed: inp.speed });
            }
            return Ok(out);
        }
        let (mut candidate, _) = self.compile_converged(source, &self.work_dir.join("test_case"))?;
        let further = self.scheme.uses_cliff();
        let mut pairs = Vec::with_capacity(test.len());
        {
            let cj = self.converged.as_mut().ok_or("our software has a bug: a test-case run with no converged judge")?;
            for i in &test {
                pairs.push(Judge::converged_pair(cj, &mut candidate, &self.cases[*i], further)?);
            }
        }
        for ((cand, seed), i) in pairs.into_iter().zip(&test) {
            let inp = self.case_inputs(&cand, &seed)?;
            out.push(TestCaseRun { case: self.cases[*i].name.clone(), candidate: cand, seed, quality: inp.quality, speed: inp.speed });
        }
        Ok(out)
    }

    /// THE K-MEANS SCHEME's evaluation (`docs/KMEANS_FITNESS.md`), returning the quality group (in `Quality::inputs`)
    /// and the speed group. THE WORK GUARD first (the review's F6): the candidate's work model on each fitness case
    /// against `converged_work_cap` of the seed's on that case; over it the candidate is not run and takes the worst
    /// score (`over_cap_unrun`), never a stop of the run. Then the plain program compiled once (its own directory,
    /// `--eval single`), and on every fitness case the candidate and the seed run to UltraDim's stopping rule
    /// interleaved `pairs` times, the inputs a case against THE SEED'S RUN OF THAT INTERLEAVING (F5), the program's
    /// own time read by the judge's timer (F4). A program that fails to compile is a bug of our software (logged) and
    /// scores worst; a run that does not finish is the candidate's (a NaN, a wrong shape: `unfinished`, worst on that
    /// case) and the run goes on; the seed's run not converging beside a candidate is our bug and stops the run. The
    /// quality's `outputs` are the final centroids a case (the behaviour), its exactness counts 0 (there is no exact
    /// answer).
    fn evaluate_converged(&mut self, tree: &Tree, source: &str) -> Result<(Quality, Vec<f64>), String> {
        let caps = self.converged.as_ref().ok_or("our software has a bug: a converged evaluation with no converged judge")?.work_caps.clone();
        let fit: Vec<usize> = (0..self.cases.len()).filter(|i| self.cases[*i].split.in_fitness()).collect();
        let (qw, sw) = self.converged_widths();
        let worst = |work: u128, behaviour: &str| (Quality { raw: vec![[1.0; 3]; fit.len()], inputs: vec![1.0; qw * fit.len()], exact_train: 0, exact_held: 0, exact_test: 0, work, outputs: Vec::new(), behaviour: behaviour.into() }, vec![1.0; sw * fit.len()]);
        let mut work = 0u128;
        for (slot, i) in fit.iter().enumerate() {
            let w = self.interp.work(tree, &self.sizes[*i])?;
            work = work.max(w);
            if w > caps[slot] {
                self.over_cap_unrun += 1;
                println!("# {} over the work cap on {}: {w} node executions, the cap {} ({WORK_CAP_TIMES_SEED} x the seed's); not run, the worst score", utc(), self.cases[*i].name, caps[slot]);
                self.last = ("hit", 0, Stages::default());
                return Ok(worst(work, "over the work cap"));
            }
        }
        self.max_work = self.max_work.max(work);
        let t = Instant::now();
        let dir = self.work_dir.join("converged");
        let mut st = Stages::default();
        let definition = self.scheme.definition;
        // The runs a case: the resident-sample path's batch (this generation's module), or the candidate's own
        // compile and its stepped pairs against the seed.
        let runs: Result<Vec<(ConvergedRun, ConvergedRun)>, String> = if self.mode == EvalMode::Rust {
            let (r, bs) = self.batched_k.remove(source).ok_or_else(|| format!("our software has a bug: a k-means program scored on the resident-sample path that no batch held\n{source}"))?;
            st.add(&bs);
            r
        } else {
            match self.compile_converged(source, &dir) {
                Ok((mut candidate, cs)) => {
                    st.add(&cs);
                    let t_runs = Instant::now();
                    let further = self.scheme.uses_cliff();
                    let cj = self.converged.as_mut().ok_or("our software has a bug: a converged evaluation with no converged judge")?;
                    let mut runs = Vec::with_capacity(fit.len());
                    for i in &fit {
                        runs.push(Judge::converged_pair(cj, &mut candidate, &self.cases[*i], further)?);
                    }
                    st.runs = t_runs.elapsed().as_secs_f64();
                    Ok(runs)
                }
                Err(e) => Err(e),
            }
        };
        let (q, speed) = match runs {
            Ok(runs) => {
                let mut quality = Vec::with_capacity(qw * fit.len());
                let mut speed = Vec::with_capacity(sw * fit.len());
                let mut outputs = Vec::with_capacity(fit.len());
                let mut unfinished = 0;
                for (cand, seed) in &runs {
                    if !cand.finished {
                        unfinished += 1;
                    }
                    match definition {
                        Definition::FutharkConvergedGated | Definition::FutharkConvergedQqs | Definition::FutharkKmeansBetter => {
                            let inputs = self.case_inputs(cand, seed)?;
                            quality.extend(inputs.quality);
                            speed.extend(inputs.speed);
                        }
                        _ => quality.extend(kmeans_inputs_five(cand, seed)),
                    }
                    outputs.push(Array::f32s(vec![cand.centroids.len(), cand.centroids.first().map_or(0, Vec::len)], cand.centroids.concat()));
                }
                self.unfinished += unfinished;
                let behaviour = text_hash_bytes(&fuller::futhark_sig::write_binary(&outputs.iter().collect::<Vec<_>>()));
                (Quality { raw: vec![[1.0; 3]; fit.len()], inputs: quality, exact_train: 0, exact_held: 0, exact_test: 0, work, outputs, behaviour }, speed)
            }
            Err(e) => {
                self.log_bug(&e, source);
                worst(work, "failed")
            }
        };
        if self.mode == EvalMode::Rust {
            self.last = ("rust", 1, st);
        } else {
            st.timing = t.elapsed().as_secs_f64() - st.total();
            self.gen_stages.add(&st);
            self.last = ("single", 1, st);
        }
        Ok((q, speed))
    }

    /// Score one individual in a stage.
    fn evaluate(&mut self, tree: &Tree, source: &str, stage: FitnessStage) -> Result<Eval, String> {
        self.scored += 1;
        self.last = ("hit", 0, Stages::default());
        if self.converged.is_some() {
            // The k-means scheme: the fitness of the two groups by the scheme's rule (`Scheme::converged_fitness`); a
            // program scored before reads its cached groups.
            let eval = |f: fuller::runner::TwoLevel, q: Quality, speed: Vec<f64>| Eval { fitness: f.fitness, theta_q: f.quality, theta_s: f.speed, quality: q, speed, ratio: f64::NAN };
            if let Some(q) = self.quality_cache.get(source) {
                self.hits += 1;
                let speed = self.converged_speed.get(source).cloned().ok_or("our software has a bug: a k-means program with a cached quality group and no speed group")?;
                let f = self.scheme.converged_fitness(&q.inputs, &speed, self.quality_weight)?;
                return Ok(eval(f, q.clone(), speed));
            }
            self.evaluations += 1;
            self.compiled += 1;
            let (q, speed) = self.evaluate_converged(tree, source)?;
            let f = self.scheme.converged_fitness(&q.inputs, &speed, self.quality_weight)?;
            self.quality_cache.insert(source.to_string(), q.clone());
            self.converged_speed.insert(source.to_string(), speed.clone());
            return Ok(eval(f, q, speed));
        }
        let q = if let Some(q) = self.quality_cache.get(source) {
            self.hits += 1;
            q.clone()
        } else {
            self.evaluations += 1;
            self.compiled += 1;
            let q = self.quality(tree, source)?;
            self.quality_cache.insert(source.to_string(), q.clone());
            q
        };
        let mut speed = Vec::new();
        let mut ratio = f64::NAN;
        if stage.measures_speed() {
            if self.backend == Backend::C {
                return Err("stage 2 times on the GPU: the c backend cannot enter it (run with --backend gpu)".into());
            }
            let missing = self.slots.missing(source);
            let timing_work = self.interp.work(tree, &self.timing.sizes)?;
            if missing > 0 && timing_work > self.timing_cap {
                // Over the per-submit cap at the timing size: never dispatched (the work guard), no speed credit.
                self.over_cap_untimed += 1;
                self.slots.add(source, &vec![f64::INFINITY; missing])?;
            } else if missing > 0 {
                // The timed program is the plain one (the declared signature, one case); its empty slots are filled.
                let got = match self.mode {
                    EvalMode::Single => {
                        let t = Instant::now();
                        let r = self.time_single(tree, missing, missing == self.slots.slots());
                        let mut st = match &r {
                            Ok((_, cs)) => *cs,
                            Err(_) => Stages::default(),
                        };
                        st.timing = t.elapsed().as_secs_f64() - st.total();
                        self.gen_stages.add(&st);
                        self.last.2.add(&st);
                        r.map(|(r, _)| r)
                    }
                    EvalMode::Batch | EvalMode::Rust => {
                        let (r, secs) = self.batched_t.remove(source).ok_or_else(|| format!("our software has a bug: a program timed in batch mode that no batch held\n{source}"))?;
                        if self.last.0 == "hit" {
                            self.last.0 = self.mode.label();
                        }
                        self.last.2.timing += secs;
                        r
                    }
                };
                let r = match got {
                    Ok(r) => r,
                    Err(err) => {
                        self.log_bug(&format!("timing: {err}"), source);
                        vec![f64::INFINITY; missing]
                    }
                };
                self.slots.add(source, &r)?;
            }
            let ratios = self.slots.full(source)?.to_vec();
            speed = ratios.iter().map(|r| if r.is_finite() { r / (1.0 + r) } else { 1.0 }).collect();
            ratio = median(&ratios);
        }
        let t = self.scheme.staged_weighted(stage, &q.inputs(), &speed, self.quality_weight)?;
        Ok(Eval { fitness: t.fitness, theta_q: t.quality, theta_s: t.speed, quality: q, speed, ratio })
    }
}

/// The genetic side: the contract, the layout and the variation operators.
struct Genetics<'a> {
    k: &'a FutharkKingdom,
    contract: &'a KingdomContract,
    layout: &'a GeneLayout,
    const_len: &'a BTreeMap<Ty, u32>,
    /// THE RING CANVAS (`--start llm-ring`, plan section H): every fresh gene is a canvas of these rings, not random.
    canvas: Option<&'a Rings>,
    /// `--mutation ring-walk` (plan section I.1): ring walk replaces point mutation.
    ring_walk: bool,
}

impl Genetics<'_> {
    /// A FRESH gene: a ring canvas with `--start llm-ring` (`fuller::futhark_rings::Rings::canvas`), else a random
    /// gene. Every fresh gene of a run comes from here: generation 0's padding, C's draws, a resumed tree's padding,
    /// the slope check's draws.
    fn random_gene(&self, rng: &mut GeneRng) -> Result<Vec<u32>, String> {
        match self.canvas {
            Some(r) => r.canvas(self.contract, self.layout, rng),
            None => random_gene(self.contract, self.layout, self.const_len, rng),
        }
    }

    /// THE MUTATION of the loop (`--mutation`): `point`, today's point mutation of `m` coding cells; `ring-walk`, one
    /// ring walk of one type's head or tail window by `k` in `1 ..= m` positions (`Rings::walk`), returned so the
    /// caller records it as a `ring_walk` event.
    fn mutation(&self, gene: &mut [u32], m: usize, rng: &mut GeneRng) -> Result<Option<fuller::futhark_rings::Walk>, String> {
        if !self.ring_walk {
            self.mutate(gene, m, rng)?;
            return Ok(None);
        }
        let rings = self.canvas.ok_or("our software has a bug: ring walk with no rings")?;
        let coding = self.coding_cells(gene)?;
        rings.walk(self.layout, gene, &coding, m, rng)
    }

    /// The event of a ring walk on a child.
    fn walk_event(&self, w: &fuller::futhark_rings::Walk, child: &Individual, parent: u64) -> Mutation {
        let mut m = Mutation::new(EventKind::RingWalk, "ring_walk", format!("{} {} advanced {} along its ring", self.k.dual(w.ty).map(|d| d.dual.as_str()).unwrap_or(""), w.domain.name(), w.k));
        m.id = Some(child.id);
        m.parents = vec![parent];
        m.text_after = Some(sexp(self.k, &child.tree));
        m
    }

    /// The operator name of a fresh gene in the ledger.
    fn fresh_label(&self) -> &'static str {
        if self.canvas.is_some() {
            "canvas"
        } else {
            "random"
        }
    }

    fn tree(&self, gene: &[u32]) -> Result<Tree, String> {
        Ok(express(self.contract, self.layout, gene, self.const_len).map_err(|e| format!("a gene does not express: {e}"))?.tree())
    }

    /// The cells the walk reads: every expressed node's symbol cell and, for an expressed constant, its Dc cell.
    fn coding_cells(&self, gene: &[u32]) -> Result<Vec<usize>, String> {
        fuller::futhark_rings::coding_cells(self.contract, self.layout, gene, self.const_len)
    }

    /// LOW-CHANGE mutation: `m` expressed cells, each set to another member of its own alphabet.
    fn mutate(&self, gene: &mut [u32], m: usize, rng: &mut GeneRng) -> Result<(), String> {
        fuller::futhark_rings::point_mutate(self.contract, self.layout, self.const_len, gene, m, rng)
    }

    /// THE ONE-EDIT NEIGHBOURS of a gene: every expressed cell set to every other member of its own alphabet (the
    /// point mutations of [`Genetics::mutate`]), each child's tree once, the gene's own tree left out. Returns each
    /// with the cell it changed.
    fn neighbours(&self, gene: &[u32]) -> Result<Vec<(usize, Tree)>, String> {
        let own = self.tree(gene)?;
        let mut seen: HashSet<Tree> = HashSet::new();
        let mut out = Vec::new();
        for pos in self.coding_cells(gene)? {
            for v in self.layout.alphabet(self.contract, pos, self.const_len) {
                if v == gene[pos] {
                    continue;
                }
                let mut g = gene.to_vec();
                self.layout.mutate(self.contract, &mut g, pos, v, self.const_len).map_err(|e| format!("a neighbour: {e}"))?;
                let t = self.tree(&g)?;
                if t != own && seen.insert(t.clone()) {
                    out.push((pos, t));
                }
            }
        }
        Ok(out)
    }

    /// Homologous crossover: a range of the domain that holds one of `a`'s expressed cells, exchanged between `a`
    /// and `b` (both children kept).
    fn crossover(&self, a: &mut [u32], b: &mut [u32], rng: &mut GeneRng) -> Result<(), String> {
        crossover_once(self.contract, self.layout, self.const_len, a, b, rng)
    }

    /// MARRIAGE, typed subtree crossover: the domain is every pair (node of A, node of B) of one output type, not the
    /// two roots, not identical subtrees, whose two offspring both write back into the parents' genes; one pair is
    /// drawn uniformly. Returns the two offspring genes, or None for an empty domain.
    fn marry(&self, a: &Individual, b: &Individual, rng: &mut GeneRng) -> Result<Option<Offspring>, String> {
        let na = a.tree.level_order();
        let nb = b.tree.level_order();
        let mut domain = Vec::new();
        for (sa, pa) in &na {
            for (sb, pb) in &nb {
                if (pa.is_empty() && pb.is_empty()) || sa == sb || self.k.rows[sa.symbol as usize].output != self.k.rows[sb.symbol as usize].output {
                    continue;
                }
                domain.push((pa.clone(), pb.clone(), (*sa).clone(), (*sb).clone()));
            }
        }
        // Uniform over the fitting pairs of the WHOLE domain (plan section F, fault 19): the domain in a random order
        // (Fisher-Yates), the first pair whose two offspring both write back.
        for i in (1..domain.len()).rev() {
            domain.swap(i, rng.below(i + 1));
        }
        for (pa, pb, sa, sb) in domain {
            let ca = replace(&a.tree, &pa, &sb);
            let cb = replace(&b.tree, &pb, &sa);
            if let (Ok(ga), Ok(gb)) = (write_back(self.contract, self.layout, &a.gene, &ca), write_back(self.contract, self.layout, &b.gene, &cb)) {
                return Ok(Some((ga, gb)));
            }
        }
        Ok(None)
    }
}

/// A marriage's two offspring genes.
type Offspring = (Vec<u32>, Vec<u32>);

/// `t` with the subtree at `path` replaced.
fn replace(t: &Tree, path: &[u32], with: &Tree) -> Tree {
    match path.split_first() {
        None => with.clone(),
        Some((i, rest)) => {
            let mut kids = t.kids.clone();
            kids[*i as usize] = replace(&kids[*i as usize], rest, with);
            Tree { symbol: t.symbol, const_index: t.const_index, kids }
        }
    }
}

fn tree_json(t: &Tree) -> serde_json::Value {
    serde_json::json!([t.symbol, t.const_index, t.kids.iter().map(tree_json).collect::<Vec<_>>()])
}

fn tree_of_json(v: &serde_json::Value) -> Result<Tree, String> {
    let a = v.as_array().filter(|a| a.len() == 3).ok_or("a checkpoint tree is not [symbol, const, kids]")?;
    Ok(Tree {
        symbol: a[0].as_u64().ok_or("a checkpoint symbol")? as u32,
        const_index: a[1].as_u64().map(|c| c as u32),
        kids: a[2].as_array().ok_or("checkpoint kids")?.iter().map(tree_of_json).collect::<Result<_, _>>()?,
    })
}

/// The running executable's identity (its path and the hash of its bytes), so every seed of a comparison can be
/// shown to have run on one binary (plan section G, fault 17).
fn binary_identity() -> String {
    match std::env::current_exe() {
        Ok(p) => format!("{} {}", p.display(), std::fs::read(&p).map(|b| text_hash_bytes(&b)).unwrap_or_else(|e| format!("unreadable: {e}"))),
        Err(e) => format!("unknown: {e}"),
    }
}

/// The tag of the random-search control's own random stream (regime C): `seed ^ REGIME_C_TAG`.
const REGIME_C_TAG: u64 = 0x5245_4749_4D45_5F43;

/// The tag of the slope check's own random stream (`--check`): its genes are never a run's generation 0.
const CHECK_TAG: u64 = 0x534C_4F50_4543_4845;

/// THE RING CANVAS of a run (`--start llm-ring`, plan section H): the pool's programs (`DIR/bench.tsv`, the published
/// benchmarks' functions, then `DIR/llm.tsv`, the topic programs, written by `examples/futhark_pool.rs`) of the chosen
/// sources, in order, into the rings; the problem's own published file is left out (leave one out: its R and C runs
/// and its measurements must not have the answer in the canvas); the empty rings filled (`Rings::complete`). Writes
/// `OUT/rings.tsv` (what went in, every fill and gap). Returns the rings and the TASK ALTERNATIVES (`DIR/alternatives.tsv`,
/// Sonnet's programs for the benchmark's own task, Andrew's clarification of 2026-10-29): the population's material,
/// never the rings', in order, those of the root type that fit the layout.
fn ring_canvas(k: &FutharkKingdom, contract: &KingdomContract, layout: &GeneLayout, dir: &Path, pname: &str, sources: &[Source], out: &Path) -> Result<(Rings, Vec<Tree>), String> {
    let mut entries = Vec::new();
    for f in ["bench.tsv", "llm.tsv"] {
        if dir.join(f).is_file() {
            entries.extend(load_pool(&dir.join(f))?);
        }
    }
    let excluded = published_source(pname).unwrap_or("");
    let mut rings = Rings::default();
    let (mut used, mut left_out, mut by_source) = (0usize, 0usize, BTreeMap::<&str, usize>::new());
    for e in &entries {
        if !sources.contains(&e.source) {
            continue;
        }
        if e.source == Source::Bench && e.origin == excluded {
            left_out += 1;
            continue;
        }
        let t = build_at(k, &e.sexp, &e.dual).map_err(|x| format!("the pool line {} {} {}: {x}", e.source.name(), e.origin, e.function))?;
        rings.add(contract, &t);
        used += 1;
        *by_source.entry(e.source.name()).or_default() += 1;
    }
    let root = k.dual(layout.root).map(|d| d.dual.clone()).unwrap_or_default();
    let (mut alternatives, mut unfit) = (Vec::new(), 0usize);
    if dir.join("alternatives.tsv").is_file() {
        for e in load_pool(&dir.join("alternatives.tsv"))? {
            if e.dual != root {
                continue;
            }
            let t = build_at(k, &e.sexp, &e.dual).map_err(|x| format!("the alternative {}: {x}", e.origin))?;
            if fuller::typed_gep::fits(contract, layout, &t).is_ok() {
                alternatives.push(t);
            } else {
                unfit += 1;
            }
        }
    }
    let (full, fills) = rings.complete(k, contract, layout)?;
    let (by_class, by_slot) = (fills.iter().filter(|f| f.from == "class").count(), fills.iter().filter(|f| f.from != "class").count());
    let mut tsv = format!("key\tvalue\npool\t{}\nsources\t{}\nprograms\t{used}\n", dir.display(), sources.iter().map(|s| s.name()).collect::<Vec<_>>().join(","));
    for (s, n) in &by_source {
        tsv.push_str(&format!("programs_{s}\t{n}\n"));
    }
    tsv.push_str(&format!("left_out_published_file\t{excluded} ({left_out} functions)\ntask_alternatives_that_fit\t{}\ntask_alternatives_too_big_for_the_layout\t{unfit}\nrings_filled_from_their_class\t{by_class}\nrings_filled_from_their_slot_family_class_gap\t{by_slot}\n", alternatives.len()));
    for f in &fills {
        tsv.push_str(&format!("fill\t{} {} from {} ({} symbols)\n", f.dual, f.ring.name(), f.from, f.len));
    }
    std::fs::write(out.join("rings.tsv"), &tsv).map_err(|e| e.to_string())?;
    println!("# {} ring canvas: {used} pool programs ({by_source:?}), {left_out} left out ({excluded}), {by_class} rings filled from their class, {by_slot} from their slot family (class gaps); {} task alternatives fit the layout, {unfit} too big", utc(), alternatives.len());
    Ok((full, alternatives))
}

/// GENERATION 0 of `--start llm-ring` in a discovery run (R, C; Andrew's clarification of 2026-10-29): Sonnet's task
/// alternatives only, in the pool's order, each on its own ring canvas, with NO NEAR COPIES. The bar, stated before
/// the population is built and recorded in `OUT/population.tsv`: a tree edit distance of at least `bar` to every
/// other individual and to the reference (leave one out: an alternative within the bar of the published program is
/// the answer planted), and an output (on every case, the GPU path) no other individual has. An alternative below the
/// bar is thrown away and the next one taken; too few distinct alternatives is an error asking for more, never a
/// random fill.
fn alternatives_population(run: &mut Run, sig: &Signature, alternatives: &[Tree], reference: &Tree, pop: usize, bar: usize, rng: &mut GeneRng) -> Result<Vec<Individual>, String> {
    let (mut near_reference, mut near_other, mut same_output) = (0usize, 0usize, 0usize);
    let mut taken: Vec<Tree> = Vec::new();
    let mut behaviours: HashSet<String> = HashSet::new();
    let mut out: Vec<Individual> = Vec::new();
    let mut lines = String::from("alternative\tverdict\tdistance_to_reference\tleast_distance_to_an_individual\tprogram\n");
    let mut next = 0usize;
    while out.len() < pop && next < alternatives.len() {
        // Candidates by distance, a batch at a time; then their outputs on the run's path.
        let mut batch: Vec<(usize, Tree)> = Vec::new();
        while batch.len() + out.len() < pop && next < alternatives.len() {
            let t = &alternatives[next];
            let dr = tree_distance(t, reference);
            let d_other = taken.iter().chain(batch.iter().map(|b| &b.1)).map(|o| tree_distance(t, o)).min().unwrap_or(usize::MAX);
            let verdict = if dr < bar {
                near_reference += 1;
                "thrown: near the reference"
            } else if d_other < bar {
                near_other += 1;
                "thrown: near another individual"
            } else {
                batch.push((next, t.clone()));
                "kept by distance"
            };
            lines.push_str(&format!("{next}\t{verdict}\t{dr}\t{}\t{}\n", if d_other == usize::MAX { "-".to_string() } else { d_other.to_string() }, sexp(run.gen.k, t)));
            next += 1;
        }
        let sources: Vec<String> = batch.iter().map(|b| program(run.gen.k, sig, &b.1)).collect::<Result<_, _>>()?;
        let items: Vec<(&Tree, &str)> = batch.iter().zip(&sources).map(|(b, s)| (&b.1, s.as_str())).collect();
        run.judge.prefetch(&items, FitnessStage::Quality)?;
        for ((i, t), src) in batch.iter().zip(&sources) {
            let e = run.judge.evaluate(t, src, FitnessStage::Quality)?;
            if !behaviours.insert(e.quality.behaviour.clone()) {
                same_output += 1;
                lines.push_str(&format!("{i}\tthrown: the output of another individual\t-\t-\t{}\n", sexp(run.gen.k, t)));
                continue;
            }
            taken.push(t.clone());
            let g = write_back(run.gen.contract, run.gen.layout, &run.gen.random_gene(rng)?, t).map_err(|e| format!("an alternative's write-back: {e}"))?;
            let ind = run.fresh(sig, vec![], "llm_alternative", g, 0)?;
            let mut m = Mutation::new(EventKind::ModelRewrite, "llm_alternative", format!("generation 0: Sonnet's task alternative {i} on a ring canvas"));
            m.id = Some(ind.id);
            m.text_after = Some(sexp(run.gen.k, &ind.tree));
            m.entered = Some(true);
            run.tel.mutation(0, m)?;
            out.push(ind);
        }
    }
    let summary = format!("bar_least_tree_edit_distance\t{bar}\nbar_distinct_outputs\tyes\nalternatives_seen\t{next}\nkept\t{}\nthrown_near_the_reference\t{near_reference}\nthrown_near_another_individual\t{near_other}\nthrown_same_output\t{same_output}\n", out.len());
    std::fs::write(run.out.join("population.tsv"), format!("{summary}\n{lines}")).map_err(|e| e.to_string())?;
    let msg = format!("generation 0 from Sonnet's task alternatives: {} kept of {next} seen; thrown: {near_reference} near the reference, {near_other} near another individual (tree edit distance under {bar}), {same_output} with another's output", out.len());
    println!("# {} {msg}", utc());
    run.tel.note(0, msg)?;
    if out.len() < pop {
        return Err(format!("only {} distinct task alternatives of {} wanted (bar: distance {bar}, distinct outputs; OUT/population.tsv): Sonnet must write more, different alternatives (examples/futhark_pool.rs llm)", out.len(), pop));
    }
    Ok(out)
}

/// GENERATION 0 FROM ONE SEED (`--gen0 fable-mutants`, Andrew, 2026-10-29): Fable's program for the task (`DIR/seed.tsv`,
/// cached so every seed and both regimes start from it; leave one out: `examples/futhark_pool.rs seed` keeps it at
/// least 5 tree edits from the published program) written onto a ring canvas is individual 1; then each mutant is the
/// run's mutation operator (`--mutation`, `Genetics::mutation`, m in 1 ..= 4) applied once to a member drawn
/// uniformly, kept only if its text and its tree differ from every member's (`fuller::futhark_rings::mutants`), until
/// the population is full. Logs the seed program, the mutants tried and kept and each member's tree edit distance
/// from the seed (`OUT/population.tsv`); every kept mutant is a mutation event.
/// With `keep_seed` false (`--gen0 fable-mutants-only`, Andrew, 2026-10-29: the winner dominated), the seed is grown
/// and logged but left out of generation 0, so the population is `pop` mutants and no seed.
fn fable_mutants_population(run: &mut Run, sig: &Signature, dir: &Path, pop: usize, keep_seed: bool, rng: &mut GeneRng) -> Result<Vec<Individual>, String> {
    if run.gen.canvas.is_none() {
        return Err("--gen0 fable-mutants writes the seed onto a ring canvas: it needs --start llm-ring".into());
    }
    let e = load_pool(&dir.join("seed.tsv"))?.into_iter().next().ok_or("seed.tsv holds no program")?;
    let seed = build_at(run.gen.k, &e.sexp, &e.dual)?;
    fuller::typed_gep::fits(run.gen.contract, run.gen.layout, &seed).map_err(|x| format!("the seed program does not fit the layout: {x}"))?;
    let g0 = write_back(run.gen.contract, run.gen.layout, &run.gen.random_gene(rng)?, &seed).map_err(|x| format!("the seed's write-back: {x}"))?;
    let spec = fuller::futhark_rings::MutantSpec { contract: run.gen.contract, layout: run.gen.layout, const_len: run.gen.const_len, pop: pop + usize::from(!keep_seed), max_tries: 200 * pop };
    let m = {
        let gen = &run.gen;
        let mut mutate = |g: &mut Vec<u32>, r: &mut GeneRng| -> Result<(), String> {
            let m = 1 + r.below(4);
            gen.mutation(g, m, r).map(|_| ())
        };
        let render = |t: &Tree| program(gen.k, sig, t);
        fuller::futhark_rings::mutants(&spec, g0, &mut mutate, &render, rng)?
    };
    let operator = if run.gen.ring_walk { "ring_walk" } else { "mutation" };
    let mut lines = format!("seed_program\t{}\nseed_origin\t{}\noperator\t{operator}\nmutants_tried\t{}\nmutants_kept\t{}\n\nmember\tparent\tdistance_to_seed\tprogram\n", e.sexp, e.origin, m.tried, m.genes.len() - 1);
    let mut out: Vec<Individual> = Vec::with_capacity(pop);
    for (i, g) in m.genes.iter().enumerate() {
        let parents: Vec<u64> = m.parents[i].map(|p| vec![out[p].id]).unwrap_or_default();
        let ind = run.fresh(sig, parents.clone(), if i == 0 { "fable_seed" } else { "gen0_mutant" }, g.clone(), 0)?;
        let d = tree_distance(&ind.tree, &seed);
        lines.push_str(&format!("{i}\t{}\t{d}\t{}\n", m.parents[i].map_or("-".to_string(), |p| p.to_string()), sexp(run.gen.k, &ind.tree)));
        let mut ev = Mutation::new(if i == 0 { EventKind::ModelRewrite } else if run.gen.ring_walk { EventKind::RingWalk } else { EventKind::Note }, if i == 0 { "fable_seed" } else { "gen0_mutant" }, if i == 0 { format!("generation 0: the seed program ({}) on a ring canvas", e.origin) } else { format!("generation 0: a {operator} mutant of member {}, {d} tree edits from the seed", m.parents[i].unwrap_or(0)) });
        ev.id = Some(ind.id);
        ev.parents = parents;
        ev.text_after = Some(sexp(run.gen.k, &ind.tree));
        ev.entered = Some(true);
        run.tel.mutation(0, ev)?;
        out.push(ind);
    }
    if !keep_seed {
        out.remove(0);
        lines.push_str("\nthe seed (member 0) is LEFT OUT of generation 0 (--gen0 fable-mutants-only)\n");
    }
    std::fs::write(run.out.join("population.tsv"), &lines).map_err(|x| x.to_string())?;
    let msg = format!("generation 0 from the seed program and its {operator} mutants: {} kept of {} tried (no two members with one text or one tree); the seed {}", m.genes.len() - 1, m.tried, if keep_seed { "is in" } else { "is LEFT OUT" });
    println!("# {} {msg}\n# seed: {}", utc(), e.sexp);
    run.tel.note(0, msg)?;
    Ok(out)
}

// THE WORK BOUND FOR TIMING, one for every evaluation path: `fuller::futhark_sig::TIMING_WORK_CAP` (its basis there).
use fuller::futhark_sig::TIMING_WORK_CAP;

/// THE OPERATOR MIX (`--ops`, Andrew, 2026-10-31): every child draws ONE operator by its probability, so all of them
/// run in the same generation and each child's operator is recorded. `marriage` is the share of children made by
/// marriage (as `--marriage`); the others are drawn for each remaining child. The ROW operators (`invert-rows`,
/// `transpose-rows`, `cross-rows`: Andrew, 2026-11-01) act on rows across every type's column; the old ones, inside
/// one type's span, are unchanged beside them.
/// The RELATIVE-row operators (`invert-rel`, `transpose-rel`, `cross-rel`) cut every column at its own row. The AIMED
/// row operators (`invert-rows-coding`, `transpose-rows-coding`, `cross-rows-coding`; `fuller::futhark_operators`)
/// are the row operators with their rows drawn around a coding cell.
const OPS: [&str; 15] = ["point", "invert", "transpose", "crossover", "walk", "marriage", "invert-rows", "transpose-rows", "cross-rows", "invert-rel", "transpose-rel", "cross-rel", "invert-rows-coding", "transpose-rows-coding", "cross-rows-coding"];

/// The mutation event of a row operator's child (`docs/STANDARD_runners.md`): the operator, what it did, the child,
/// its parents and the texts before and after.
fn row_event(k: &FutharkKingdom, operator: &str, what: &str, child: &Individual, before: &Tree) -> Mutation {
    let mut m = Mutation::new(EventKind::Note, operator, what.to_string());
    m.id = Some(child.id);
    m.parents = child.parents.clone();
    m.text_before = Some(sexp(k, before));
    m.text_after = Some(sexp(k, &child.tree));
    m
}

const MEASURE_TAG: u64 = 0x4D45_4153_5552_4521;

/// The operators `--measure` measures: the old ones inside one type's span (`invert`, `transpose`, `crossover`) and
/// the row operators across every column (`invert-rows`, `transpose-rows`, `cross-rows`; `invert-rel`, `transpose-rel`,
/// `cross-rel` at each column's own relative row), on the same parents and
/// seeds, so old and new compare.
const MEASURE_KINDS: [&str; 15] = ["grow", "point", "walk", "invert", "transpose", "crossover", "invert-rows", "transpose-rows", "cross-rows", "invert-rel", "transpose-rel", "cross-rel", "invert-rows-coding", "transpose-rows-coding", "cross-rows-coding"];

/// THE OPERATOR MEASUREMENT (`--measure KIND`, plan sections H and I), on the run's evaluation path (the GPU): `count`
/// children, each ONE application of the operator to a parent, the parents Sonnet's task alternatives in order
/// (`ring_canvas`), none within `--distinct-edits` of the reference or of another parent, each written onto a fresh gene (a ring canvas with `--start llm-ring`, else random padding).
/// KIND: `grow` (`fuller::futhark_rings::grow`: a coding head cell given more arguments, so cells the padding holds
/// are read), `point` (today's point mutation, `Genetics::mutate`, one cell), `walk` (`Rings::walk`, ring walk, k at
/// most 4; it needs the rings), `invert`, `transpose`, `crossover` (inside one type's span) and `invert-rows`,
/// `transpose-rows`, `cross-rows` (the row operators) and `invert-rel`, `transpose-rel`, `cross-rel` (the relative-row
/// operators); a crossover's partner is the next parent, its child the
/// parent's side. Measured, the children's: the share whose output is all zeros; the share with an
/// output not seen before in the measurement (parents' included); the share better than their parent (stage-1
/// fitness); the share equal to their parent's output; distinct outputs. Writes `OUT/measure.tsv` (one line a child)
/// and `OUT/measure_summary.tsv`; every child is a mutation event (`ring_walk` for the walk).
fn measure_command(run: &mut Run, sig: &Signature, kind: &str, parents: &[Tree], count: usize, chunk: usize, seed: u64) -> Result<(), String> {
    if parents.is_empty() {
        return Err("--measure: the pool has no model-written program that fits the layout".into());
    }
    if !MEASURE_KINDS.contains(&kind) {
        return Err(format!("--measure {kind}: one of {}", MEASURE_KINDS.join(", ")));
    }
    if kind == "walk" && run.gen.canvas.is_none() {
        return Err("--measure walk needs the rings: --start llm-ring".into());
    }
    let t0 = Instant::now();
    let (mut pad, mut op) = (GeneRng::new(seed ^ MEASURE_TAG), GeneRng::new(seed ^ MEASURE_TAG ^ 1));
    let mut children: Vec<(usize, Tree, String)> = Vec::with_capacity(count);
    let mut skipped = 0usize;
    let mut i = 0usize;
    while children.len() < count {
        let pi = i % parents.len();
        i += 1;
        let mut g = write_back(run.gen.contract, run.gen.layout, &run.gen.random_gene(&mut pad)?, &parents[pi]).map_err(|e| format!("a parent's write-back: {e}"))?;
        let coding = fuller::futhark_rings::coding_cells(run.gen.contract, run.gen.layout, &g, run.gen.const_len)?;
        let what = match kind {
            "grow" => grow(run.gen.contract, run.gen.layout, &mut g, &coding, run.gen.const_len, &mut op).map(|(p, a, b)| format!("cell {p}: {} -> {}", run.gen.k.rows[a as usize].name, run.gen.k.rows[b as usize].name)),
            "point" => {
                run.gen.mutate(&mut g, 1, &mut op)?;
                Some("one coding cell".to_string())
            }
            "invert" => invert_once(run.gen.layout, &mut g, &coding, &mut op)?,
            "transpose" => transpose_once(run.gen.contract, run.gen.layout, &mut g, &coding, &mut op)?,
            "invert-rows" => invert_rows_once(run.gen.layout, &mut g, &mut op)?,
            "transpose-rows" => transpose_rows_once(run.gen.contract, run.gen.layout, &mut g, &mut op)?,
            "invert-rel" => invert_rel_once(run.gen.layout, &mut g, &mut op)?,
            "transpose-rel" => transpose_rel_once(run.gen.contract, run.gen.layout, &mut g, &mut op)?,
            "invert-rows-coding" => invert_rows_coding_once(run.gen.layout, &mut g, &coding, &mut op)?,
            "transpose-rows-coding" => transpose_rows_coding_once(run.gen.contract, run.gen.layout, &mut g, &coding, &mut op)?,
            "crossover" | "cross-rows" | "cross-rel" | "cross-rows-coding" => {
                // The partner: the next parent, on its own fresh padding; the child is this parent's side.
                let mut h = write_back(run.gen.contract, run.gen.layout, &run.gen.random_gene(&mut pad)?, &parents[(pi + 1) % parents.len()]).map_err(|e| format!("a partner's write-back: {e}"))?;
                if kind == "cross-rows" {
                    cross_rows_once(run.gen.layout, &mut g, &mut h, &mut op)?
                } else if kind == "cross-rows-coding" {
                    cross_rows_coding_once(run.gen.contract, run.gen.layout, run.gen.const_len, &mut g, &mut h, &mut op)?
                } else if kind == "cross-rel" {
                    cross_rel_once(run.gen.layout, &mut g, &mut h, &mut op)?
                } else {
                    run.gen.crossover(&mut g, &mut h, &mut op)?;
                    Some(format!("a range of one span exchanged with parent {}", (pi + 1) % parents.len()))
                }
            }
            _ => run.gen.canvas.ok_or("no rings")?.walk(run.gen.layout, &mut g, &coding, 4, &mut op)?.map(|w| format!("{} {} by {}", run.gen.k.dual(w.ty).map(|d| d.dual.as_str()).unwrap_or(""), w.domain.name(), w.k)),
        };
        match what {
            Some(w) => children.push((pi, run.gen.tree(&g)?, w)),
            None => {
                skipped += 1;
                if skipped > 100 * count {
                    return Err(format!("--measure {kind}: no parent admits the operator"));
                }
            }
        }
    }
    let padding = if run.gen.canvas.is_some() { "ring canvas" } else { "random padding" };
    println!("# {} measure {kind} on {padding}: {count} children of {} parents ({skipped} parents the operator could not change)", utc(), parents.len());
    // Score the parents, then the children, on the run's path.
    let mut progs: Vec<&Tree> = parents.iter().collect();
    progs.extend(children.iter().map(|c| &c.1));
    let sources: Vec<String> = progs.iter().map(|t| program(run.gen.k, sig, t)).collect::<Result<_, _>>()?;
    let mut scored: Vec<(f64, Quality)> = Vec::with_capacity(progs.len());
    // THE INTERIM RESULTS (Rules.md, logging: decision-critical, so a run stopped early still teaches): after every
    // batch, the running shares over the children scored so far, printed and appended to OUT/measure_progress.tsv.
    let np0 = parents.len();
    let mut iseen: HashSet<String> = HashSet::new();
    let (mut i_done, mut i_zero, mut i_new, mut i_better, mut i_same) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut i_best = f64::INFINITY;
    let bugs0 = run.judge.bugs;
    let mut progress = String::from("utc\tseconds\tchildren_scored\tof\tunseen_output\tbetter_than_parent\tsame_output_as_parent\tall_zero\tbest_child_fitness\tbest_parent_fitness\tcompile_failures\n");
    for idx in (0..progs.len()).collect::<Vec<_>>().chunks(chunk.max(1)) {
        let items: Vec<(&Tree, &str)> = idx.iter().map(|i| (progs[*i], sources[*i].as_str())).collect();
        run.judge.prefetch(&items, FitnessStage::Quality)?;
        for &i in idx {
            let e = run.judge.evaluate(progs[i], &sources[i], FitnessStage::Quality)?;
            scored.push((e.fitness, e.quality));
        }
        if scored.len() < np0 {
            continue;
        }
        if iseen.is_empty() {
            iseen = scored[..np0].iter().map(|s| s.1.behaviour.clone()).collect();
            let bp = scored[..np0].iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
            println!("# {} measure: the {np0} parents scored; best parent fitness {bp:.4}, {} distinct parent outputs", utc(), iseen.len());
        }
        while np0 + i_done < scored.len() {
            let (f, q) = &scored[np0 + i_done];
            let (pf, pq) = &scored[children[i_done].0];
            i_new += usize::from(iseen.insert(q.behaviour.clone()));
            i_zero += usize::from(q.all_zero());
            i_better += usize::from(f < pf);
            i_same += usize::from(q.behaviour == pq.behaviour);
            i_best = i_best.min(*f);
            i_done += 1;
        }
        if i_done > 0 {
            let n = i_done as f64;
            let bp = scored[..np0].iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
            let fails = run.judge.bugs - bugs0;
            let secs = t0.elapsed().as_secs_f64();
            println!(
                "# {} measure {kind}: {i_done}/{} children | unseen {:.1} % | better than parent {:.1} % | same as parent {:.1} % | all zeros {:.1} % | best child {i_best:.4} (best parent {bp:.4}) | compile failures {fails} | {secs:.0} s",
                utc(), children.len(), 100.0 * i_new as f64 / n, 100.0 * i_better as f64 / n, 100.0 * i_same as f64 / n, 100.0 * i_zero as f64 / n
            );
            progress.push_str(&format!("{}\t{secs:.0}\t{i_done}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{i_best:.6}\t{bp:.6}\t{fails}\n", utc(), children.len(), i_new as f64 / n, i_better as f64 / n, i_same as f64 / n, i_zero as f64 / n));
            std::fs::write(run.out.join("measure_progress.tsv"), &progress).map_err(|e| e.to_string())?;
        }
    }
    let np = parents.len();
    let mut seen: HashSet<String> = scored[..np].iter().map(|s| s.1.behaviour.clone()).collect();
    let (mut zero, mut unseen, mut better, mut same) = (0usize, 0usize, 0usize, 0usize);
    let mut distinct: HashSet<&str> = HashSet::new();
    let mut tsv = String::from("child	parent	edit	fitness	parent_fitness	all_zero	unseen	same_as_parent	behaviour	program
");
    for (j, (pi, tree, what)) in children.iter().enumerate() {
        let (f, q) = &scored[np + j];
        let (pf, pq) = &scored[*pi];
        let is_new = seen.insert(q.behaviour.clone());
        zero += usize::from(q.all_zero());
        unseen += usize::from(is_new);
        better += usize::from(f < pf);
        same += usize::from(q.behaviour == pq.behaviour);
        distinct.insert(q.behaviour.as_str());
        tsv.push_str(&format!("{j}	{pi}	{what}	{f:?}	{pf:?}	{}	{}	{}	{}	{}
", u8::from(q.all_zero()), u8::from(is_new), u8::from(q.behaviour == pq.behaviour), q.behaviour, sexp(run.gen.k, tree)));
        let mut m = Mutation::new(if kind == "walk" { EventKind::RingWalk } else { EventKind::Note }, &format!("measure_{kind}"), format!("{what}: fitness {pf:.4} -> {f:.4}{}", if is_new { ", an unseen output" } else { "" }));
        m.fitness_before = Some(*pf);
        m.fitness_after = Some(*f);
        m.improved = Some(f < pf);
        m.text_before = Some(sexp(run.gen.k, &parents[*pi]));
        m.text_after = Some(sexp(run.gen.k, tree));
        run.tel.mutation(0, m)?;
    }
    std::fs::write(run.out.join("measure.tsv"), &tsv).map_err(|e| e.to_string())?;
    let n = children.len().max(1) as f64;
    let parent_zero = scored[..np].iter().filter(|s| s.1.all_zero()).count();
    let rep = format!(
        "problem	{}
operator	{kind}
padding	{padding}
eval	{}
parents	{np}
parents_all_zero	{parent_zero}
children	{}
parents_the_operator_could_not_change	{skipped}
all_zero_share	{:.4}
unseen_output_share	{:.4}
better_than_parent_share	{:.4}
same_output_as_parent_share	{:.4}
distinct_outputs	{}
seconds	{:.1}
",
        run.inst.name,
        run.judge.mode.label(),
        children.len(),
        zero as f64 / n,
        unseen as f64 / n,
        better as f64 / n,
        same as f64 / n,
        distinct.len(),
        t0.elapsed().as_secs_f64()
    );
    std::fs::write(run.out.join("measure_summary.tsv"), &rep).map_err(|e| e.to_string())?;
    print!("{rep}");
    Ok(())
}

/// The first line of a check's stored outputs (`check_outputs.bin`).
const CHECK_OUTPUTS_MAGIC: &str = "fuller-futhark-check-outputs v1";

/// THE SLOPE CHECK (`--check`, docs/PLAN_futhark_fitness_fixes.md section A), a gate before any run: the fitness
/// (stage 1, the quality angle) of every one-edit neighbour of the reference gene and of `n_random` random genes,
/// scored on the run's evaluation path (the GPU with `--eval rust` or `--eval batch`), each beside its tree edit
/// distance to the reference ([`tree_distance`]). The verdict is `fuller::runner::slope_check`'s. It writes
/// `OUT/check.tsv` (one line a program), `OUT/check_summary.tsv` (the verdict and the identities of the cases and
/// of the quality inputs' formula, which a run's `--gate` compares) and `OUT/check_outputs.bin` (every program's
/// outputs on every case, as the GPU gave them). With `--check-replay FILE` the outputs are read from such a file
/// (the same cases, checked by their fingerprint) instead of run again, so a change to the quality inputs alone is
/// re-checked on the GPU's own outputs.
fn slope_check_command(run: &mut Run, sig: &Signature, reference: &Tree, chunk: usize, n_random: usize, seed: u64, replay: Option<&Path>, program_size: usize) -> Result<(), String> {
    let t0 = Instant::now();
    let fp_cases = cases_fingerprint(&run.judge.cases);
    let stored: Option<HashMap<String, Vec<Array>>> = match replay {
        Some(f) => Some(read_check_outputs(f, &fp_cases)?),
        None => None,
    };
    let mut rng = GeneRng::new(seed ^ CHECK_TAG);
    let ref_gene = write_back(run.gen.contract, run.gen.layout, &run.gen.random_gene(&mut rng)?, reference).map_err(|e| format!("the reference's write-back: {e}"))?;
    let mut progs: Vec<(&'static str, Option<usize>, Tree)> = vec![("reference", None, reference.clone())];
    for (pos, t) in run.gen.neighbours(&ref_gene)? {
        progs.push(("neighbour", Some(pos), t));
    }
    let n_neigh = progs.len() - 1;
    for _ in 0..n_random {
        let g = run.gen.random_gene(&mut rng)?;
        progs.push(("random", None, run.gen.tree(&g)?));
    }
    println!("# {} slope check: the reference, {n_neigh} one-edit neighbours of its gene, {n_random} random genes; program size N = {program_size}; {}", utc(), if replay.is_some() { "outputs replayed from the stored GPU outputs" } else { "scored on the run's evaluation path" });
    let sources: Vec<String> = progs.iter().map(|p| program(run.gen.k, sig, &p.2)).collect::<Result<_, _>>()?;
    let mut scored: Vec<(f64, Quality)> = Vec::with_capacity(progs.len());
    for (ci, idx) in (0..progs.len()).collect::<Vec<_>>().chunks(chunk.max(1)).enumerate() {
        if stored.is_none() {
            let items: Vec<(&Tree, &str)> = idx.iter().map(|i| (&progs[*i].2, sources[*i].as_str())).collect();
            run.judge.prefetch(&items, FitnessStage::Quality)?;
        }
        for &i in idx {
            let (tree, source) = (&progs[i].2, &sources[i]);
            let e = match &stored {
                Some(m) => {
                    let text = sexp(run.gen.k, tree);
                    let outs = m.get(&text).ok_or_else(|| format!("the stored outputs have no program {text}"))?;
                    let q = if outs.is_empty() { Quality::worst(&run.judge.cases, run.judge.parts()) } else { Quality::of(outs.clone(), &run.judge.cases, run.judge.work_of(tree)?, run.judge.parts()) };
                    (run.judge.scheme.staged(FitnessStage::Quality, &q.inputs, &[])?.fitness, q)
                }
                None => {
                    let e = run.judge.evaluate(tree, source, FitnessStage::Quality)?;
                    (e.fitness, e.quality)
                }
            };
            scored.push(e);
        }
        if ci % 10 == 0 {
            println!("# {} slope check: {} of {} programs scored, {:.0} s", utc(), scored.len(), progs.len(), t0.elapsed().as_secs_f64());
        }
    }
    if scored[0].0 != 0.0 {
        return Err(format!("our software has a bug: the reference scores {} in the slope check", scored[0].0));
    }
    // The table, one line a program; the stored outputs.
    let mut tsv = String::from("kind\tcell\tfitness\tdistance\tnodes\texact_train\texact_held\tall_zero\tbehaviour\tprogram\n");
    let mut bin: Vec<u8> = format!("{CHECK_OUTPUTS_MAGIC}\t{fp_cases}\n").into_bytes();
    let mut written: HashSet<String> = HashSet::new();
    let (mut neigh, mut rand) = (Vec::new(), Vec::new());
    for (p, (f, q)) in progs.iter().zip(&scored) {
        let d = tree_distance(&p.2, reference);
        let text = sexp(run.gen.k, &p.2);
        tsv.push_str(&format!("{}\t{}\t{f:?}\t{d}\t{}\t{}\t{}\t{}\t{}\t{text}\n", p.0, p.1.map_or("-".to_string(), |c| c.to_string()), p.2.level_order().len(), q.exact_train, q.exact_held, u8::from(q.all_zero()), q.behaviour));
        if written.insert(text.clone()) {
            let raw = if q.outputs.is_empty() { Vec::new() } else { fuller::futhark_sig::write_binary(&q.outputs.iter().collect::<Vec<_>>()) };
            bin.extend_from_slice(text.as_bytes());
            bin.push(b'\n');
            bin.extend_from_slice(&(raw.len() as u64).to_le_bytes());
            bin.extend_from_slice(&raw);
        }
        match p.0 {
            "neighbour" => neigh.push((*f, d)),
            "random" => rand.push((*f, d)),
            _ => {}
        }
    }
    std::fs::write(run.out.join("check.tsv"), &tsv).map_err(|e| e.to_string())?;
    if stored.is_none() {
        std::fs::write(run.out.join("check_outputs.bin"), &bin).map_err(|e| e.to_string())?;
    }
    let c = slope_check(&neigh, &rand)?;
    let random_q: Vec<&Quality> = progs.iter().zip(&scored).filter(|(p, _)| p.0 == "random").map(|(_, s)| &s.1).collect();
    let behaviours: HashSet<&str> = random_q.iter().map(|q| q.behaviour.as_str()).collect();
    let zeros = random_q.iter().filter(|q| q.all_zero()).count();
    let distinct_random: HashSet<&Tree> = progs.iter().filter(|p| p.0 == "random").map(|p| &p.2).collect();
    let verdict = if c.pass { "pass" } else { "fail" };
    let rep = format!(
        "problem\t{}\nprogram_size\t{program_size}\neval\t{}\nreplayed\t{}\ncases\t{}\ncases_fingerprint\t{fp_cases}\nquality_fingerprint\t{}\nquality_inputs\t{}\nneighbours_inexact\t{}\nneighbours_exact\t{}\nrandom_genes\t{}\nrandom_distinct_programs\t{}\nrandom_distinct_outputs\t{}\nrandom_all_zero_share\t{:.4}\nmedian_neighbour\t{:.6}\nmedian_random\t{:.6}\nbest_random\t{:.6}\nrandom_beaten_by_median_neighbour\t{:.4}\nrequired_random_beaten\t{}\nneighbours_above_median_random\t{:.4}\nneighbours_above_best_random\t{:.4}\nauc_neighbour_beats_random\t{:.4}\nspearman_fitness_distance\t{:.4}\nspearman_fitness_distance_random_only\t{:.4}\nseconds\t{:.1}\nverdict\t{verdict}\n",
        run.inst.name,
        run.judge.mode.label(),
        if replay.is_some() { "yes" } else { "no" },
        run.judge.cases.len(),
        quality_fingerprint_of(run.judge.parts()),
        run.judge.cases.iter().map(|k| k.output.as_ref().map_or(0, |o| quality_input_names_of(run.judge.parts(), o).len())).sum::<usize>(),
        c.neighbours,
        c.exact_neighbours,
        c.random,
        distinct_random.len(),
        behaviours.len(),
        zeros as f64 / random_q.len().max(1) as f64,
        c.median_neighbour,
        c.median_random,
        c.best_random,
        c.random_beaten_by_median_neighbour,
        fuller::runner::SLOPE_RANDOM_BEATEN,
        c.neighbours_above_median_random,
        c.neighbours_above_best_random,
        c.auc,
        c.fdc,
        c.fdc_random,
        t0.elapsed().as_secs_f64(),
    );
    std::fs::write(run.out.join("check_summary.tsv"), &rep).map_err(|e| e.to_string())?;
    print!("{rep}");
    run.tel.note(0, format!("slope check: {verdict}; the median neighbour beats {:.1} % of random genes, Spearman(fitness, distance) {:.3}", 100.0 * c.random_beaten_by_median_neighbour, c.fdc))?;
    Ok(())
}

/// The stored outputs of a check (`check_outputs.bin`), by program text; a file of other cases is an error.
fn read_check_outputs(path: &Path, fp_cases: &str) -> Result<HashMap<String, Vec<Array>>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let nl = raw.iter().position(|b| *b == b'\n').ok_or("the stored outputs have no header")?;
    let head = String::from_utf8_lossy(&raw[..nl]).to_string();
    if head != format!("{CHECK_OUTPUTS_MAGIC}\t{fp_cases}") {
        return Err(format!("{}: stored for other cases or another format ({head}); these cases are {fp_cases}", path.display()));
    }
    let mut out = HashMap::new();
    let mut i = nl + 1;
    while i < raw.len() {
        let e = i + raw[i..].iter().position(|b| *b == b'\n').ok_or("a stored program with no end of line")?;
        let text = String::from_utf8_lossy(&raw[i..e]).to_string();
        let n = u64::from_le_bytes(raw.get(e + 1..e + 9).ok_or("a stored program cut short")?.try_into().map_err(|_| "a length")?) as usize;
        let body = raw.get(e + 9..e + 9 + n).ok_or("a stored program's outputs cut short")?;
        out.insert(text, if n == 0 { Vec::new() } else { read_binary(body)? });
        i = e + 9 + n;
    }
    Ok(out)
}

/// THE GATE (`--gate DIR`): a discovery run (R, C) starts only after the slope check of the same problem, program
/// size, cases and quality inputs passed (`DIR/check_summary.tsv`).
fn gate_passed(dir: &Path, problem: &str, program_size: usize, fp_cases: &str, parts: &[QualityPart]) -> Result<(), String> {
    let path = dir.join("check_summary.tsv");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("the gate {}: {e}", path.display()))?;
    let get = |k: &str| text.lines().find_map(|l| l.strip_prefix(&format!("{k}\t"))).unwrap_or("").to_string();
    let want = [("problem", problem.to_string()), ("program_size", program_size.to_string()), ("cases_fingerprint", fp_cases.to_string()), ("quality_fingerprint", quality_fingerprint_of(parts)), ("verdict", "pass".to_string())];
    for (k, v) in want {
        if get(k) != v {
            return Err(format!("the gate {}: {k} is {:?}, this run needs {v:?}; a discovery run starts only after the slope check passes for its problem, program size, cases and quality inputs (--check)", path.display(), get(k)));
        }
    }
    Ok(())
}

const CHART: &str = r##"
import sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
g = [int(r["generation"]) for r in rows]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(g, [float(r["best_hff"]) for r in rows], label="best HFF", color="#1f6feb", linewidth=2)
ax.plot(g, [float(r["median_hff"]) for r in rows], label="median HFF", color="#8b949e")
ax.plot(g, [float(r["best_theta_q"]) for r in rows], label="best's quality angle", color="#2da44e", linestyle="--")
ax.plot(g, [float(r["best_theta_s"]) for r in rows], label="best's speed angle", color="#bf8700", linestyle=":")
st = [i for i in range(1, len(rows)) if rows[i]["stage"] != rows[i - 1]["stage"]]
for i in st:
    ax.axvline(g[i], color="#cf222e", linewidth=1, linestyle="--", label="stage 2 (quality and speed)")
ax.set_ylim(0, 1.6)
ax.set_xlabel("generation")
ax.set_ylabel("HFF (0 best, pi/2 worst)")
ax.set_title(sys.argv[3])
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=120)
"##;

/// Everything the run carries between generations.
struct Run<'a> {
    inst: Instance,
    out: PathBuf,
    gen: Genetics<'a>,
    judge: Judge<'a>,
    tel: RunTelemetry,
    ledger: Ledger,
    next_id: u64,
    hof: HallOfFame,
    stage: FitnessStage,
    /// The evaluation at which an individual was first exact on every case (training and held-out).
    first_exact: Option<(u64, u32, u64)>,
    /// The first exact program's test cases exact (of the test set, never in the fitness).
    first_exact_test: Option<usize>,
    /// THE CHAMPION: the best individual scored in the current stage, the hall of fame's rank 1 (plan section G,
    /// faults 15 and 16): every champion field of the report and its re-time come from it, never the last batch.
    champion: Option<Individual>,
}

impl Run<'_> {
    fn fresh(&mut self, sig: &Signature, parents: Vec<u64>, operator: &str, gene: Vec<u32>, born: u32) -> Result<Individual, String> {
        let tree = self.gen.tree(&gene)?;
        let source = program(self.gen.k, sig, &tree)?;
        self.next_id += 1;
        Ok(Individual { id: self.next_id, parents, born, operator: operator.to_string(), gene, tree, source, eval: None })
    }

    /// Score an individual in the current stage, record it in the ledger and the hall of fame.
    fn score(&mut self, generation: u32, ind: &mut Individual) -> Result<(), String> {
        let e = self.judge.evaluate(&ind.tree, &ind.source, self.stage)?;
        if e.quality.perfect() && self.first_exact.is_none() {
            self.first_exact = Some((self.judge.evaluations, generation, ind.id));
            self.first_exact_test = Some(e.quality.exact_test);
            self.tel.note(generation, format!("first exact program on every case (training and held-out): evaluation {}, generation {generation}, id {}", self.judge.evaluations, ind.id))?;
            println!("# {} FIRST EXACT on every case at evaluation {}, generation {generation}: {}", utc(), self.judge.evaluations, sexp(self.gen.k, &ind.tree));
        }
        let text = sexp(self.gen.k, &ind.tree);
        let (path, batch, st) = self.judge.last;
        let mut cols = vec![path.to_string(), batch.to_string()];
        cols.extend(st.values().iter().map(|v| format!("{:.1}", v * 1e3)));
        let exact = |v: &[f64]| if v.is_empty() { "-".to_string() } else { v.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>().join(",") };
        cols.push(exact(&e.quality.inputs));
        cols.push(exact(&e.speed));
        cols.push(e.quality.behaviour.clone());
        cols.push(u8::from(e.quality.all_zero()).to_string());
        self.ledger.record_line(&LedgerLine { generation, id: ind.id, parents: &ind.parents, operator: &ind.operator, fitness: e.fitness, text: &text, extra: &cols })?;
        let mut extras = BTreeMap::new();
        extras.insert("theta_quality".to_string(), e.theta_q);
        extras.insert("theta_speed".to_string(), e.theta_s);
        extras.insert("exact_train".to_string(), e.quality.exact_train as f64);
        extras.insert("exact_held".to_string(), e.quality.exact_held as f64);
        if e.ratio.is_finite() {
            extras.insert("speed_ratio".to_string(), e.ratio);
        }
        extras.insert("stage".to_string(), f64::from(self.stage.number()));
        self.hof.offer(FameEntry { key: text_hash(&ind.source), fitness: e.fitness, born: ind.born, text: format!("{text}\n\n{}", ind.source), nodes: Some(ind.tree.level_order().len() as u32), picture: None, extras });
        ind.eval = Some(e);
        self.offer_champion(ind);
        Ok(())
    }

    /// The champion of the stage: strictly better replaces it (ties keep the earlier, as the hall of fame does).
    fn offer_champion(&mut self, ind: &Individual) {
        if self.champion.as_ref().is_none_or(|c| ind.fit() < c.fit()) {
            self.champion = Some(ind.clone());
        }
    }

    fn checkpoint(&self, generation: u32, population: &[Individual], seconds: f64, regime: &str, rng_state: u64) -> Result<(), String> {
        let ind = |i: &Individual| serde_json::json!({"id": i.id, "parents": i.parents, "born": i.born, "operator": i.operator, "tree": tree_json(&i.tree)});
        let v = serde_json::json!({
            "problem": self.inst.name, "regime": regime, "generation": generation, "seconds": seconds, "next_id": self.next_id,
            "evaluations": self.judge.evaluations, "scored": self.judge.scored, "compiled": self.judge.compiled, "hits": self.judge.hits, "bugs": self.judge.bugs, "setmem": self.judge.setmem,
            "stage": self.stage, "first_exact": self.first_exact, "best_ever": self.tel.best_ever(), "rng": rng_state,
            "population": population.iter().map(ind).collect::<Vec<_>>(), "hof": self.hof,
        });
        save_checkpoint(&self.out, &v)
    }
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let pname = arg(&args, "--problem").unwrap_or_else(|| "pathfinder".into());
    // THE SLOPE CHECK (plan section A): a gate before any discovery run, not a run (regime "A").
    // THE OPERATOR MEASUREMENT (`--measure grow|point|walk`, plan sections H and I): a measurement, not a run, like
    // the slope check (regime "A", no timing).
    let measure = arg(&args, "--measure");
    let p = problem(&pname)?;
    // THE SEED: a problem whose prototype is not yet translated has no reference program and stops here, plainly
    // (`futhark_sig::Seed::NotYetTranslated`; no problem is in that state since k-means's translation, plan 4c (d)).
    let (reference_text, planted_text) = (p.reference()?, p.planted()?);
    println!("# {}", p.seed_state());
    // THE FITNESS SCHEME (`--fitness NAME`, `fuller::fitness`; default the problem's own, `Problem::fitness`: the
    // newest scheme of its kind).
    let scheme = fuller::fitness::scheme_for_problem(&p, arg(&args, "--fitness").as_deref())?;
    // The k-means scheme has no quality parts (nothing is compared with an answer): its judge is `evaluate_converged`.
    let parts: &'static [QualityPart] = if scheme.is_converged() { &[] } else { scheme.quality_parts()? };
    println!("# {}", scheme.card_line());
    let check = args.iter().any(|a| a == "--check") || measure.is_some();
    let regime = match arg(&args, "--regime") {
        Some(_) if check => return Err("--check is not a run: it takes no --regime".into()),
        Some(r) => r,
        None if check => "A".to_string(),
        None => return Err("--regime R|C|P (or --check, the slope check)".into()),
    };
    if !check && !["R", "C", "P"].contains(&regime.as_str()) {
        return Err(format!("--regime {regime}: R (from random), C (random-search control) or P (from the published program)"));
    }
    // R and C start from random genes; only P has a start program (the record said "published" for R and C until
    // 2026-10-28, a default of the flag, never used by them: their generation 0 is all `random` in every ledger).
    let start = match (regime.as_str(), arg(&args, "--start")) {
        ("P", None) => "published".to_string(),
        ("P", Some(s)) if s == "published" || s == "planted" => s,
        ("P", Some(s)) => return Err(format!("--start {s}: published or planted")),
        (_, None) => "random".to_string(),
        // R, C and the measurements (plan section H): generation 0 is the pool's model-written programs, each on a
        // ring canvas, and every fresh gene is a ring canvas (`--pool DIR`, `--ring-sources bench|llm|both`).
        (_, Some(s)) if s == "llm-ring" => s,
        (r, Some(s)) => return Err(format!("--start {s}: regime {r} starts from random genes or llm-ring; published and planted are for P")),
    };
    let backend = match arg(&args, "--backend").as_deref().unwrap_or("gpu") {
        "gpu" => Backend::Gpu,
        "c" => Backend::C,
        o => return Err(format!("--backend {o}: gpu or c")),
    };
    // A generation's new programs in one compiled program and one process (batch, the default since 2026-10-28), or
    // each program its own compile and processes (single).
    let mode = match arg(&args, "--eval").as_deref().unwrap_or("batch") {
        "batch" => EvalMode::Batch,
        "rust" => EvalMode::Rust,
        "single" => EvalMode::Single,
        o => return Err(format!("--eval {o}: batch, rust or single")),
    };
    if scheme.is_converged() && mode == EvalMode::Batch {
        return Err(format!("the fitness scheme {} runs UltraDim's loop: it takes --eval single (one process or server a program) or --eval rust (the resident sample on the device, the rust path's in-process host); the C batch driver evaluates fixed steps only", scheme.name));
    }
    let minutes: f64 = num(&args, "--minutes", 60.0)?;
    let budget: u64 = num(&args, "--evaluations", 6000)?;
    let pop: usize = num(&args, "--pop", 24)?;
    let seed: u64 = num(&args, "--seed", 1)?;
    // The k-means scheme runs the candidate and the seed to the stopping point `--pairs` times each on every fitness
    // case (a pair is two full runs, about 20 iterations each): its default is TWO pairs, the least that interleaves
    // (with one pair the candidate always runs first and the seed second: the round-2 review's N4); the timed-pairs
    // schemes keep 11.
    let pairs: usize = num(&args, "--pairs", if scheme.is_converged() { 2 } else { 11 })?;
    if pairs == 0 {
        return Err("--pairs 0: stage 2 needs at least one timing slot".into());
    }
    // THE QUALITY WEIGHT of a flat scheme (`--quality-weight`, Andrew's ruling of 2026-11-03): the quality inputs
    // entered this many times in the one vector the HFF reads; the scheme's own default when not given (2 for
    // k-means, the measured `fitness::ANSWER_QUALITY_WEIGHT` for the answer-scored scheme). A scheme that is not
    // flat has none, and giving it one is an error that says so.
    let quality_weight: usize = num(&args, "--quality-weight", scheme.default_quality_weight())?;
    // THE K-MEANS STEP on the c backend (`--kmeans-step server|process`): a Futhark server a program (the default
    // since 2026-11-03) or one process an iteration (the path before, kept for the throughput measurement).
    let step_process = match arg(&args, "--kmeans-step").as_deref().unwrap_or("server") {
        "server" => false,
        "process" => true,
        o => return Err(format!("--kmeans-step {o}: server or process")),
    };
    if step_process && (backend != Backend::C || !scheme.is_converged()) {
        return Err("--kmeans-step process is the c backend's k-means path (the GPU path is one process an iteration already)".into());
    }
    if scheme.is_flat() && quality_weight == 0 {
        return Err("--quality-weight 0: the quality inputs would not be entered at all (1 is no repetition)".into());
    }
    if !scheme.is_flat() && arg(&args, "--quality-weight").is_some() {
        return Err(format!("--quality-weight: the fitness scheme {} is not a flat scheme and has no quality weight", scheme.name));
    }
    if scheme.is_flat() {
        println!("# {} quality weight m = {quality_weight}: every quality input is entered {quality_weight} times in the one vector the HFF reads, the speed inputs once", utc());
    }
    // THE QUALITY CLIFF of `futhark-kmeans-better` (`--quality-cliff`, Andrew's specification of 2026-11-03): a quality
    // input is 1/2 at the reference program's value and 1 at a loss of this fraction or more. Only that scheme has one.
    let quality_cliff: f64 = num(&args, "--quality-cliff", fuller::fitness::DEFAULT_QUALITY_CLIFF)?;
    if !scheme.uses_cliff() && arg(&args, "--quality-cliff").is_some() {
        return Err(format!("--quality-cliff: the fitness scheme {} has no quality cliff (futhark-kmeans-better has)", scheme.name));
    }
    if !(quality_cliff > 0.0 && quality_cliff.is_finite()) {
        return Err(format!("--quality-cliff {quality_cliff}: the cliff's width must be positive and finite"));
    }
    if scheme.uses_cliff() {
        println!("# {} quality cliff eps = {quality_cliff}: each quality input is x/2 for x <= 1 (x the candidate relative to the reference program) and 1/2 + 1/2 min(1, (x - 1)/{quality_cliff}) above; the HFF percentile F(theta) = I_(sin^2 theta)((d - 1)/2, 1/2) is reported beside every angle", utc());
    }
    let runs: usize = num(&args, "--runs", 20)?;
    let elites: usize = num(&args, "--elites", 2)?;
    let pump_every: u32 = num(&args, "--pump-every", 5)?;
    let fold_every: u32 = num(&args, "--fold-every", 10)?;
    let ops: Option<Vec<(String, f64)>> = arg(&args, "--ops").map(|t| parse_ops(&t, &OPS)).transpose()?;
    let marriage: f64 = match &ops {
        Some(o) => o.iter().find(|x| x.0 == "marriage").map_or(0.0, |x| x.1),
        None => num(&args, "--marriage", 0.3)?,
    };
    // THE MUTATION (plan section I, one operator at a time): `point` (the baseline) or `ring-walk` (needs the rings).
    let ring_walk = match arg(&args, "--mutation").as_deref().unwrap_or("point") {
        "point" => false,
        "ring-walk" => true,
        o => return Err(format!("--mutation {o}: point or ring-walk")),
    };
    let checkpoint_every: u32 = num(&args, "--checkpoint-every", 5)?;
    let resume = args.iter().any(|a| a == "--resume");
    // Discovery runs (R, C) answer H1 and H2: they stop at the first program exact on every case, never timing.
    let stop_at_exact = args.iter().any(|a| a == "--stop-at-exact");
    // The timing gate (P): measure the two A/A sets and stop, before any search.
    let aa_only = args.iter().any(|a| a == "--aa-only");
    // The A/A re-times of the speed verdict (fault 13): a false "beat" of the published program has probability
    // 1/(n + 1), so the default (runner::AA_RETIMES) is 5 %.
    let aa_retimes: usize = arg(&args, "--aa-retimes").map(|v| v.parse().map_err(|_| format!("--aa-retimes {v}"))).transpose()?.unwrap_or(AA_RETIMES);
    if aa_retimes == 0 {
        return Err("--aa-retimes must be at least 1".into());
    }
    // THE RELATIVE SCORE of one program (`--rescore SEXP`, regime P): its quality inputs on every stored case beside
    // the published program's (all 0), and, when exact, its 11-pair re-time against the published program next to
    // both A/A sets, with the verdict (beat / matched / exact but slower / not reached). Writes OUT/rescore.tsv.
    let check_random: usize = num(&args, "--check-random", 1400)?;
    let check_replay = arg(&args, "--check-replay").map(PathBuf::from);
    let rescore = arg(&args, "--rescore");
    if rescore.is_some() && regime != "P" {
        return Err("--rescore needs --regime P (its in-run A/A set is the noise floor beside the re-time)".into());
    }
    if stop_at_exact && scheme.is_converged() {
        return Err(format!("--stop-at-exact: the fitness scheme {} has no exact program to stop at (k-means has no exact answer)", scheme.name));
    }
    if stop_at_exact && regime == "P" {
        return Err("--stop-at-exact is for the discovery regimes R and C; P starts exact and measures speed".into());
    }
    let out = PathBuf::from(arg(&args, "--out").ok_or("--out DIR")?);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let checkpoint: Option<serde_json::Value> = if resume { Some(fuller::runner::load_checkpoint(&out)?) } else { None };
    let spent = checkpoint.as_ref().and_then(|c| c["seconds"].as_f64()).unwrap_or(0.0);
    let limit = std::time::Duration::from_secs_f64((minutes * 60.0 - spent).max(0.0));

    let inst = p.instance;
    let k = load(&inst)?;
    let sig = signature(&inst)?;
    let interp = Interp::new(&k, &sig)?;
    let contract = k.relations().check().map_err(|d| format!("the kingdom check: {:?}", &d[..d.len().min(3)]))?;
    let reference = build(&k, reference_text)?;
    let planted = build(&k, planted_text)?;
    // THE PROGRAM SIZE is an experiment parameter (`--program-size`, default the problem's `program_size`), never the
    // reference's size; the run card records it. The reference and the planted slowdown must fit the layout.
    let program_size: usize = num(&args, "--program-size", p.program_size)?;
    let start_tree = if start == "planted" { planted.clone() } else { reference.clone() };
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], program_size).map_err(|e| e.to_string())?;
    for (what, t) in [("reference", &reference), ("planted slowdown", &planted)] {
        fuller::typed_gep::fits(&contract, &layout, t).map_err(|e| format!("the {what} ({} functions) does not fit the layout of program size N = {program_size}: {e}", function_count(&k, t)))?;
    }
    let const_len = k.const_len();
    let (canvas, pool_programs) = if start == "llm-ring" || measure.is_some() {
        let dir = PathBuf::from(arg(&args, "--pool").ok_or("--start llm-ring and --measure need --pool DIR (bench.tsv and llm.tsv, examples/futhark_pool.rs)")?);
        let sources = Source::set(arg(&args, "--ring-sources").as_deref().unwrap_or("both"))?;
        let (r, progs) = ring_canvas(&k, &contract, &layout, &dir, &pname, &sources, &out)?;
        // A measurement with random padding (no `--start llm-ring`) takes its parents from the pool, not its rings.
        (if start == "llm-ring" { Some(r) } else { None }, progs)
    } else {
        (None, Vec::new())
    };
    if ring_walk && canvas.is_none() {
        return Err("--mutation ring-walk walks the rings: it needs --start llm-ring --pool DIR".into());
    }
    // Every stored case: the fitness's (training and held-out) and the TEST SET (plan section G), run in the same
    // quality program but never a fitness input, so selection never sees it; an exactness claim is made on it.
    let cases = load_cases(&p)?;
    let n_test = cases.iter().filter(|c| !c.split.in_fitness()).count();
    let sizes: Vec<BTreeMap<String, usize>> = cases.iter().map(|c| sig.sizes(&c.inputs)).collect::<Result<_, _>>()?;
    if (regime == "R" || regime == "C") && scheme.is_converged() {
        // The slope check scores neighbours of the reference against stored answers; k-means has none, so the
        // scheme has no slope check and a discovery run has no gate to pass. Said plainly, not refused.
        println!("# {} no gate: the k-means scheme has no slope check (no stored answers), so a discovery run starts without one", utc());
    } else if regime == "R" || regime == "C" {
        let g = arg(&args, "--gate").ok_or("--gate DIR: a discovery run (R, C) starts only after the slope check passes (--check --out DIR)")?;
        gate_passed(Path::new(&g), &pname, program_size, &cases_fingerprint(&cases), scheme.quality_parts()?)?;
        println!("# {} the gate: the slope check in {g} passed for these cases and quality inputs", utc());
    }
    let all_stdin: Vec<u8> = cases.iter().flat_map(|c| c.stdin.clone()).collect();
    let timing = TimingInput::of(&p, &sig, &cases, &out.join("timing_build"), !scheme.is_converged())?;
    let timing_cap = fuller::futhark_sig::timing_work_cap(interp.work(&reference, &timing.sizes)?);
    println!("# {} timing bound: {timing_cap} node executions at the timing input (the published program's work {}, the constant {TIMING_WORK_CAP})", utc(), interp.work(&reference, &timing.sizes)?);
    // THE STATIC COST (the matrix kingdom, A7: `fuller::futhark_cost`), reported at the timing input's sizes; a
    // signature problem has none.
    let timing_sizes = timing.sizes.clone();
    let static_cost = |t: &Tree| -> Result<String, String> { Ok(fuller::futhark_sig::static_cost(&k, t, &timing_sizes)?.map_or("-".to_string(), |s| format!("traffic {} alloc {} depth {}", s.traffic, s.alloc, s.depth))) };
    let reference_static = static_cost(&reference)?;
    if reference_static != "-" {
        println!("# {} static cost of the published program at the timing input: {reference_static}", utc());
    }
    println!(
        "# {} futhark_evolve: problem {pname}, regime {regime}{}, budget {budget} evaluations, guard {minutes} min, population {pop}, seed {seed}, backend {backend:?}, eval {mode:?}, program size N = {program_size} ({} words a gene); {} cases ({} held out, {} test cases never in the fitness), timing on {}{}",
        utc(),
        if regime == "P" { format!(" from the {start} program") } else { String::new() },
        layout.width,
        cases.len(),
        cases.iter().filter(|c| c.split == Split::Held).count(),
        n_test,
        timing.label(),
        if resume { "; RESUMED" } else { "" }
    );
    if scheme.is_converged() {
        // Control (the k-means scheme): the seed's COMPILED run converges on every fitness case (below, the converged
        // judge). The host interpreter is not run on the dbpedia cases (about 4e7 node executions a second: one step
        // of 1,000 x 1,536 x 16 is a minute); its agreement with the compiled program is the kingdom's tests' on the
        // small case (`tests_kmeans_seed`).
        println!("# {} control: the seed's run is checked compiled (the host interpreter is not run on cases of this size)", utc());
    } else {
        // Control: the reference gene is exact on every case (interpreted), so the kingdom expresses the published program.
        for c in &cases {
            if interp.eval(&reference, &c.inputs)? != *c.answer()? {
                return Err(format!("our software has a bug: the reference gene is not exact on {}", c.name));
            }
        }
        println!("# {} control: the reference gene is exact on all {} cases (host interpreter)", utc(), cases.len());
    }

    let mut objectives: Vec<String> = if scheme.definition == Definition::FutharkConvergedQqs || scheme.definition == Definition::FutharkKmeansBetter {
        // The one flat vector's columns, in its order: every case's quality inputs, the whole group m times, then
        // every case's speed inputs once.
        let fit = || cases.iter().filter(|c| c.split.in_fitness());
        let qnames: &[&str] = if scheme.uses_cliff() { &QUALITY_NAMES_BETTER } else { &QUALITY_NAMES };
        let quality: Vec<String> = fit().flat_map(|c| qnames.iter().map(|m| format!("quality {} {m}", c.name)).collect::<Vec<_>>()).collect();
        (0..quality_weight).flat_map(|i| quality.iter().map(move |q| if quality_weight > 1 { format!("{q} (copy {} of {quality_weight})", i + 1) } else { q.clone() })).chain(fit().flat_map(|c| SPEED_NAMES.iter().map(|m| format!("speed {} {m}", c.name)).collect::<Vec<_>>())).collect()
    } else if scheme.definition == Definition::FutharkConvergedGated {
        let fit = || cases.iter().filter(|c| c.split.in_fitness());
        fit().flat_map(|c| QUALITY_NAMES.iter().map(|m| format!("quality {} {m}", c.name)).collect::<Vec<_>>()).chain(fit().flat_map(|c| SPEED_NAMES.iter().map(|m| format!("speed {} {m}", c.name)).collect::<Vec<_>>())).collect()
    } else if scheme.is_converged() {
        cases.iter().filter(|c| c.split.in_fitness()).flat_map(|c| INPUT_NAMES_FIVE.iter().map(|m| format!("quality {} {m}", c.name)).collect::<Vec<_>>()).collect()
    } else {
        cases.iter().filter(|c| c.split.in_fitness()).flat_map(|c| c.output.as_ref().map_or_else(Vec::new, |o| quality_input_names_of(parts, o)).into_iter().map(|m| format!("quality {} {m}", c.name)).collect::<Vec<_>>()).collect()
    };
    if !scheme.is_converged() {
        for i in 0..pairs {
            objectives.push(format!("speed pair {i} r/(1+r) (stage 2)"));
        }
    }
    let card = serde_json::json!({"problem": pname, "regime": regime, "start": start, "backend": format!("{backend:?}"), "eval": format!("{mode:?}"), "evaluations": budget, "minutes": minutes, "pop": pop, "seed": seed, "stop_at_exact": stop_at_exact, "pairs": pairs, "runs": runs, "elites": elites, "pump_every": pump_every, "fold_every": fold_every, "marriage": marriage, "program_size": program_size, "program_size_given": args.iter().any(|a| a == "--program-size"), "program_size_default": p.program_size, "gene_words": layout.width, "fitness": scheme.name, "fitness_commit": scheme.commit, "fitness_definition": scheme.description, "quality_weight": scheme.is_flat().then_some(quality_weight), "quality_cliff": scheme.uses_cliff().then_some(quality_cliff), "kmeans_step": if step_process { "process" } else { "server" }});
    let tel = RunTelemetry::start_with_fitness_start(
        &out,
        format!("futhark-{pname}-{regime}-seed{seed}"),
        RunInfo {
            kingdom: "futhark_sig".into(),
            dataset: format!("{pname}: {} cases", cases.len()),
            seed: seed as u32,
            population: pop as u32,
            max_generations: u32::MAX,
            budget_ms: (minutes * 60_000.0) as u64,
            hff_objectives: objectives,
            config_json: Some(card.to_string()),
            resume,
            already_spent_seconds: spent,
        },
        scheme.start_with(quality_weight, quality_cliff),
    )?;
    if mode == EvalMode::Rust && backend != Backend::Gpu {
        return Err("--eval rust runs on the GPU: it takes --backend gpu".into());
    }
    // The shim is built once, here (under the cargo lock); no candidate's link runs cargo or takes the lock. The rust
    // path has no shim and no C: its runtime is built once instead, into the run's own directory.
    let with_shim = backend == Backend::Gpu && mode != EvalMode::Rust;
    let shim = if with_shim { prebuild_shim(&out.join("shim"))? } else { out.join("shim") };
    let yard_exe = if with_shim && !scheme.is_converged() { Some(compile_program_prebuilt(p.yardstick, &out.join("yardstick_build"), &shim)?) } else { None };
    // The other side of every pair: the published program, or for k-means THE SEED (the reference gene's program:
    // the yardstick of its fitness), whose range entry every generation's module carries.
    let ref_source = program(&k, &sig, &reference)?;
    let host = match mode {
        EvalMode::Rust => {
            let rt = fuller::futhark_rust::prebuild_rt(&out.join("rust_rt"))?;
            let b = fuller::futhark_batch::Backend::Rust { rt };
            // A batch whose compile does not finish within the host's limit has its source saved under the run's
            // bugs/ (`futhark_batch::COMPILE_LIMIT_FLOOR`; the split is logged in bugs.log as any halving).
            Some(Host::new(&k, &sig, cases.len(), b, &out.join("rust_build"), Inputs { cases_stdin: &all_stdin, timing_stdin: Some(timing.stdin.as_slice()), yardstick: Some(if scheme.is_converged() { ref_source.as_str() } else { p.yardstick }) })?.with_bugs_dir(&out.join("bugs")))
        }
        EvalMode::Batch => {
            let b = if backend == Backend::Gpu { fuller::futhark_batch::Backend::Gpu { shim: shim.clone() } } else { fuller::futhark_batch::Backend::C };
            let timing_stdin = if backend == Backend::Gpu { Some(timing.stdin.as_slice()) } else { None };
            Some(Host::new(&k, &sig, cases.len(), b, &out.join("batch_build"), Inputs { cases_stdin: &all_stdin, timing_stdin, yardstick: Some(p.yardstick) })?.with_bugs_dir(&out.join("bugs")))
        }
        EvalMode::Single => None,
    };
    let timer = if mode == EvalMode::Single && backend == Backend::Gpu && !scheme.is_converged() {
        Some(Host::new(&k, &sig, cases.len(), fuller::futhark_batch::Backend::Gpu { shim: shim.clone() }, &out.join("timing_build"), Inputs { cases_stdin: &all_stdin, timing_stdin: Some(timing.stdin.as_slice()), yardstick: Some(p.yardstick) })?)
    } else {
        None
    };
    let mut run = Run {
        inst,
        out: out.clone(),
        gen: Genetics { k: &k, contract: &contract, layout: &layout, const_len: &const_len, canvas: canvas.as_ref(), ring_walk },
        judge: Judge { k: &k, sig: &sig, all_stdin, interp, cases, scheme, parts, sizes, timing, timing_cap, over_cap_untimed: 0, backend, runs, yard_exe, shim: shim.clone(), work_dir: out.join("candidate_build"), quality_cache: HashMap::new(), slots: TimingSlots::new(pairs), evaluations: 0, scored: 0, compiled: 0, hits: 0, bugs: 0, setmem: 0, ice: 0, bug_log: out.join("bugs.log"), max_work: 0, mode, host, timer, batched_q: HashMap::new(), batched_t: HashMap::new(), gen_stages: Stages::default(), batch_splits: 0, last: ("hit", 0, Stages::default()), converged: None, converged_speed: HashMap::new(), over_cap_unrun: 0, unfinished: 0, quality_weight, quality_cliff, step_process, batched_k: HashMap::new(), upload_seconds: 0.0, resident_hits: 0 },
        tel,
        ledger: Ledger::open_with(&out, resume, &LEDGER_STAGE_COLUMNS)?,
        next_id: 0,
        hof: HallOfFame::new(5),
        stage: if regime == "P" { FitnessStage::QualityAndSpeed } else { FitnessStage::Quality },
        first_exact: None,
        first_exact_test: None,
        champion: None,
    };

    // The A/A noise floor: the yardstick against itself (one warm-up pair discarded, then 11).
    let mut noise = Vec::new();
    // Batch mode: both A/A sets (and the gate's single timings) in one batch process, as every candidate is timed.
    let mut batch_aa: Option<AaSets> = None;
    if backend == Backend::Gpu && !check && !scheme.is_converged() {
        // Every path times its pairs with the driver's interleaved pairs (`Step::Pair`): the batch process, this
        // process (rust), or the single path's own timing process.
        let a = run.judge.aa_batch(p.yardstick, if regime == "P" { pairs } else { 0 }, if aa_only { 3 } else { 0 })?;
        noise = a.noise.clone();
        batch_aa = Some(a);
        let msg = format!("A/A noise floor{}, the published program against itself, 11 interleaved pairs of at least {PAIR_SIDE_SECONDS} s a side on {}: ratio median {:.4}, min {:.4}, max {:.4}", if mode.batched() { " (in one batch process)" } else { " (in one timing process)" }, run.judge.timing.label(), median(&noise), noise.iter().cloned().fold(f64::INFINITY, f64::min), noise.iter().cloned().fold(0.0, f64::max));
        println!("# {} {msg}", utc());
        run.tel.note(0, msg)?;
    }
    if let (Some(want), Some(y), false) = (&run.judge.timing.output, &run.judge.yard_exe, check) {
        // Control: the yardstick's GPU output on the timing input is the stored one (futhark c).
        let (got, _) = run_program_bytes(y, &run.judge.timing.stdin, false, 1)?;
        if read_binary(&got)? != read_binary(want)? {
            return Err("our software has a bug: the yardstick on the GPU does not give futhark c's output on the timing input".into());
        }
        println!("# {} control: the yardstick on the GPU gives futhark c's output on {} (GPU)", utc(), run.judge.timing.label());
    }
    if check_replay.is_some() && measure.is_none() {
        // A REPLAYED check runs nothing on the GPU: the stored outputs are the GPU's, the reference's own among them,
        // and the check stops with our bug if the reference does not score 0 on them.
        slope_check_command(&mut run, &sig, &reference, pop, check_random, seed, check_replay.as_deref(), program_size)?;
        run.tel.end("early_stop", 0, None)?;
        return Ok(());
    }
    if scheme.is_converged() {
        // THE K-MEANS SCHEME's yardstick: the seed compiled once (the run's own directory), run to the stopping point on
        // every fitness case with the program's own time read as every candidate's will be; a run that does not
        // CONVERGE is our bug. Each candidate is then run interleaved with it, and scored against that run. The work
        // cap of every fitness case is the seed's work on it times `WORK_CAP_TIMES_SEED`.
        let points_dir = out.join("candidate_build").join("points");
        let fit: Vec<usize> = (0..run.judge.cases.len()).filter(|i| run.judge.cases[*i].split.in_fitness()).collect();
        let t_control = Instant::now();
        let (seed_stepper, seed_runs, cuts) = if mode == EvalMode::Rust {
            // THE RESIDENT-SAMPLE PATH: every stored case is a row range of the sample (the timing input's first
            // value), checked here against the case's points; the seed's control is the seed as a member against
            // the seed's entry, one pair, on every fitness case.
            let sample = read_binary(&run.judge.timing.stdin)?.into_iter().next().ok_or("the timing input holds no sample")?;
            let mut cuts = Vec::with_capacity(run.judge.cases.len());
            for c in &run.judge.cases {
                let cut = KMEANS_CASES.iter().chain(std::iter::once(&fuller::futhark_sig::kmeans_converged::KMEANS_TIMING_CUT)).find(|x| x.name == c.name).ok_or_else(|| format!("the case {} is not a declared row range of the sample (kmeans_converged::KMEANS_CASES)", c.name))?;
                let d = sample.dims[1];
                if c.inputs[0].dims != [cut.end - cut.start, d] || c.inputs[0].data != sample.data[cut.start * d..cut.end * d] {
                    return Err(format!("our software has a bug: the case {} is not the sample's rows {}..{}", c.name, cut.start, cut.end));
                }
                cuts.push((cut.start, cut.end - cut.start));
            }
            run.judge.converged = Some(ConvergedJudge { seed: None, seed_runs: Vec::new(), work_caps: Vec::new(), pairs, points_dir: points_dir.clone(), cuts: Some(cuts.clone()) });
            let (runs, c) = run.judge.kmeans_batch(&[Program::Text(&ref_source)], &fit, 1)?;
            let runs = runs.into_iter().next().ok_or("our software has a bug: the seed's control batch of no member")??;
            println!("# {} control: the seed's module compiled in {:.1} s, the sample uploaded in {:.3} s ({} MB), the seed's {} runs in {:.1} s", utc(), c.compile.futhark + c.compile.cc, c.upload_seconds, sample.data.len() * 4 / 1_000_000, 2 * fit.len(), c.run.total);
            (None, runs.into_iter().map(|(_, seed)| seed).collect::<Vec<_>>(), Some(cuts))
        } else {
            let (mut seed_stepper, _) = run.judge.compile_converged(&ref_source, &out.join("seed_build"))?;
            let mut seed_runs = Vec::new();
            for i in &fit {
                let c = &run.judge.cases[*i];
                seed_runs.push(converge(&c.inputs[0], &c.inputs[1], seed_stepper.step(c)?)?);
            }
            (Some(seed_stepper), seed_runs, None)
        };
        let mut work_caps = Vec::new();
        for (i, r) in fit.iter().zip(&seed_runs) {
            let c = &run.judge.cases[*i];
            if !r.converged {
                return Err(format!("our software has a bug: the compiled seed's run does not converge on {} ({backend:?}, {mode:?}): {} iterations, finished {}: {}", c.name, r.iterations, r.finished, r.failure));
            }
            let seed_work = run.judge.interp.work(&reference, &run.judge.sizes[*i])?;
            println!("# {} control: the compiled seed on {}: {} iterations to the stopping point, the program's own {:.3} s ({:.3} s wall), cosine inertia {:.5}, mean log10 p {:.2} ({:.2} over every point, {} at their centroid); work {seed_work}, the cap {}", utc(), c.name, r.iterations, r.seconds, r.wall_seconds, r.inertia, r.mean_log10_p, r.mean_log10_p_all, r.coincident, converged_work_cap(seed_work));
            work_caps.push(converged_work_cap(seed_work));
        }
        println!("# {} control: the seed's {} runs took {:.1} s ({})", utc(), seed_runs.len(), t_control.elapsed().as_secs_f64(), match (backend, step_process, mode) { (_, _, EvalMode::Rust) => "the resident sample on the device, the in-process host", (Backend::C, false, _) => "one Futhark server, the points loaded once a case", (Backend::C, true, _) => "one process of the futhark c executable an iteration (--kmeans-step process)", (Backend::Gpu, _, _) => "one process of the shim's executable an iteration" });
        run.judge.converged = Some(ConvergedJudge { seed: seed_stepper, seed_runs, work_caps, pairs, points_dir, cuts });
    } else {
        run.judge.prefetch(&[(&reference, ref_source.as_str())], FitnessStage::Quality)?;
        let ref_q = run.judge.quality(&reference, &ref_source)?;
        if !ref_q.perfect() {
            return Err(format!("our software has a bug: the compiled reference is not exact ({:?})", ref_q.raw));
        }
        println!("# {} control: the compiled reference gene is exact on every case ({backend:?}); work {} node executions at most", utc(), ref_q.work);
    }
    std::fs::write(out.join("reference.fut"), &ref_source).map_err(|e| e.to_string())?;
    if let Some(kind) = &measure {
        let count: usize = num(&args, "--count", 1400)?;
        // The parents: the task alternatives with no near copies (the bar of generation 0, by distance alone).
        let bar: usize = num(&args, "--distinct-edits", 5)?;
        let mut parents: Vec<Tree> = Vec::new();
        if let Some(dir) = arg(&args, "--parents-from") {
            // The parents: the distinct programs of the LAST generation of an earlier run (its ledger), so the
            // operator is measured on real, partly correct elites.
            let ledger = std::fs::read_to_string(PathBuf::from(&dir).join("ledger.tsv")).map_err(|e| format!("--parents-from {dir}: {e}"))?;
            let rows: Vec<Vec<&str>> = ledger.lines().skip(1).map(|l| l.split('\t').collect()).filter(|f: &Vec<&str>| f.len() > 5).collect();
            let last = rows.iter().filter_map(|f| f[0].parse::<u64>().ok()).max().ok_or("--parents-from: an empty ledger")?;
            let dual = k.dual(layout.root).map(|d| d.dual.clone()).ok_or("the root has no dual")?;
            let mut texts: Vec<&str> = rows.iter().filter(|f| f[0].parse::<u64>().ok() == Some(last)).map(|f| f[5]).collect();
            texts.sort_unstable();
            texts.dedup();
            // A ledger written before 2026-11-02 names variables by frame number (`j1`, `s2`); the stack-named reader
            // gives each its class name (`fuller::futhark_sig::build_stack_named`).
            let mut stack_named = 0;
            for t in texts {
                let tree = match build_at(&k, t, &dual) {
                    Ok(tree) => tree,
                    Err(e) => {
                        stack_named += 1;
                        fuller::futhark_sig::build_stack_named(&k, t, &dual).map_err(|e2| format!("--parents-from: {t}: {e}; as a stack-named program: {e2}"))?
                    }
                };
                parents.push(tree);
            }
            println!("# {} measure: {} parents, the distinct programs of generation {last} of {dir}{}", utc(), parents.len(), if stack_named > 0 { format!("; {stack_named} read by their stack names") } else { String::new() });
        }
        for t in pool_programs.iter().filter(|_| arg(&args, "--parents-from").is_none()) {
            if tree_distance(t, &reference) >= bar && parents.iter().all(|o| tree_distance(t, o) >= bar) {
                parents.push(t.clone());
            }
        }
        if arg(&args, "--parents-from").is_none() {
            println!("# {} measure: {} parents of {} task alternatives (tree edit distance at least {bar} to the reference and to each other)", utc(), parents.len(), pool_programs.len());
        }
        measure_command(&mut run, &sig, kind, &parents, count, pop, seed)?;
        run.tel.end("early_stop", 0, None)?;
        return Ok(());
    }
    if check {
        slope_check_command(&mut run, &sig, &reference, pop, check_random, seed, check_replay.as_deref(), program_size)?;
        run.tel.end("early_stop", 0, None)?;
        return Ok(());
    }
    // The in-run A/A (P): the published program, compiled as a candidate, through the timing slots. The converged
    // scheme has no timing slots: its A/A is the seed judged as a candidate against itself (below, the control).
    let mut inrun_aa: Vec<f64> = Vec::new();
    if regime == "P" && backend == Backend::Gpu && !scheme.is_converged() {
        let r = batch_aa.as_ref().map(|a| a.inrun.clone()).ok_or("our software has a bug: the in-run A/A with no A/A sets")?;
        run.judge.slots.add(p.yardstick, &r)?;
        inrun_aa = run.judge.slots.full(p.yardstick)?.to_vec();
        let msg = format!("in-run A/A{}, the published program as a candidate through the timing slots, {pairs} pairs: ratio median {:.4}, min {:.4}, max {:.4}", if mode.batched() { " (in the batch process)" } else { " (in the timing process)" }, median(&inrun_aa), inrun_aa.iter().cloned().fold(f64::INFINITY, f64::min), inrun_aa.iter().cloned().fold(0.0, f64::max));
        println!("# {} {msg}", utc());
        run.tel.note(0, msg)?;
    }
    // THE K-MEANS SCHEME'S A/A, on both backends (the round-2 review's N3): the seed judged as a candidate against the
    // seed through the same interleaving every candidate gets. Equal runs give 1/2 on every quality input (a quality
    // input off 1/2 is a non-deterministic reduction, written down here); the spread of the seconds and iterations
    // inputs around 1/2 is THE NOISE FLOOR of this scheme, which a champion's speed claim must clear.
    let mut aa_quality: Vec<f64> = Vec::new();
    let mut aa_speed: Vec<f64> = Vec::new();
    if scheme.is_converged() && !check {
        let source = program(&k, &sig, &reference)?;
        let t_aa = Instant::now();
        run.judge.prefetch(&[(&reference, source.as_str())], FitnessStage::Quality)?;
        let e = run.judge.evaluate(&reference, &source, FitnessStage::Quality)?;
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(", ");
        let worst = |v: &[f64]| v.iter().map(|x| (x - 0.5).abs()).fold(0.0, f64::max);
        let msg = format!("in-run A/A (k-means): the seed as a candidate against itself through {pairs} interleaved pair(s) on every fitness case, {:.1} s: fitness {:.4}; quality inputs [{}], worst distance from 1/2 {:.2e}{}; speed inputs [{}], worst distance from 1/2 {:.4} (the noise floor of a speed claim)", t_aa.elapsed().as_secs_f64(), e.fitness, fmt(&e.quality.inputs), worst(&e.quality.inputs), if worst(&e.quality.inputs) > 0.0 { " (A QUALITY INPUT OFF 1/2: the program's reduction is not deterministic on this backend)" } else { "" }, fmt(&e.speed), worst(&e.speed));
        println!("# {} {msg}", utc());
        run.tel.note(0, msg)?;
        aa_quality = e.quality.inputs.clone();
        aa_speed = e.speed.clone();
    }
    // The A/A gate's evidence, on every run that timed its A/A sets (every evaluation path: batch, single, rust).
    if let Some(a) = &batch_aa {
        let rounds_us: Vec<f64> = a.rounds.iter().map(|ns| ns / 1e3).collect();
        write_aa_gate(&out, &pname, &run.judge.timing.label(), &rounds_us, &noise, &inrun_aa)?;
    }
    if let Some(text) = &rescore {
        let tree = build(&k, text)?;
        let source = program(&k, &sig, &tree)?;
        run.judge.prefetch(&[(&tree, source.as_str())], FitnessStage::Quality)?;
        let q = run.judge.quality(&tree, &source)?;
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
        let mut rep = format!("problem\t{pname}\nprogram\t{}\nexact_on_every_case\t{}\nexact_train\t{}\nexact_held\t{}\n", sexp(&k, &tree), if q.perfect() { "yes" } else { "no" }, q.exact_train, q.exact_held);
        rep.push_str(&format!("{}\n", run.judge.scheme.card_line()));
        // Each case's inputs by name (a report: the fitness takes every input on its own).
        let mut at = 0;
        for c in run.judge.cases.iter().filter(|c| c.split.in_fitness()) {
            let names = quality_input_names_of(run.judge.parts(), c.answer()?);
            let vals = &q.inputs[at..at + names.len()];
            at += names.len();
            let cells: Vec<String> = names.iter().zip(vals).map(|(n, v)| format!("{n}={v:.6}")).collect();
            rep.push_str(&format!("quality\t{}\t{:?}\t{}\n", c.name, c.split, cells.join("\t")));
        }
        rep.push_str(&format!("published_quality_inputs\tall 0\nexact_test\t{} of {}\n", q.exact_test, run.judge.cases.iter().filter(|c| !c.split.in_fitness()).count()));
        let all: Vec<f64> = noise.iter().chain(inrun_aa.iter()).cloned().collect();
        let floor = all.iter().cloned().fold(f64::INFINITY, f64::min);
        let ceiling = all.iter().cloned().fold(0.0, f64::max);
        rep.push_str(&format!("timing_input\t{}\nnoise_floor_ratios_published_vs_itself\t{}\ninrun_aa_ratios\t{}\nnoise_floor_least_ratio\t{floor:.4}\nnoise_floor_greatest_ratio\t{ceiling:.4}\n", run.judge.timing.label(), fmt(&noise), fmt(&inrun_aa)));
        let verdict = if q.perfect() && backend == Backend::Gpu {
            let (ratios, got) = run.judge.champion_batch(&tree, 11)?;
            let m = median(&ratios);
            rep.push_str(&format!("champion_vs_published_ratios\t{}\nchampion_vs_published_median\t{m:.4}\npairs_faster\t{} of {}\n", fmt(&ratios), ratios.iter().filter(|r| **r < 1.0).count(), ratios.len()));
            if let Some(want) = &run.judge.timing.output {
                rep.push_str(&format!("champion_exact_on_timing_input\t{}\n", if got == fuller::futhark_sig::read_one(want)? { "yes" } else { "NO" }));
            }
            let medians = run.judge.aa_medians(&reference, aa_retimes)?;
            rep.push_str(&format!("aa_set_medians\t{}\n", fmt(&medians)));
            speed_verdict(&ratios, &medians).text()
        } else {
            "not reached (not exact)"
        };
        rep.push_str(&format!("verdict\t{verdict}\n"));
        std::fs::write(out.join("rescore.tsv"), &rep).map_err(|e| e.to_string())?;
        print!("{rep}");
        run.tel.end("early_stop", 0, None)?;
        return Ok(());
    }
    if aa_only {
        if regime != "P" {
            return Err("--aa-only measures the two A/A sets of regime P".into());
        }
        if batch_aa.is_none() {
            return Err("--aa-only times on the GPU (--backend gpu)".into());
        }
        // Written above, with the timing rounds: the gate's file is the report.
        let rep = std::fs::read_to_string(out.join("aa_gate.tsv")).map_err(|e| format!("aa_gate.tsv: {e}"))?;
        print!("{rep}");
        run.tel.end("early_stop", 0, None)?;
        return Ok(());
    }

    let bridge = Bridge::load(&run.inst)?;
    // Each regime its own random stream (plan section E, fault 10): R and C with one seed never share generation 0.
    let stream = if regime == "C" { seed ^ REGIME_C_TAG } else { seed };
    let mut rng0 = GeneRng::new(stream);
    let mut generation: u32;
    let mut cands: Vec<Individual> = Vec::new();
    if let Some(c) = &checkpoint {
        generation = c["generation"].as_u64().ok_or("checkpoint generation")? as u32 + 1;
        run.next_id = c["next_id"].as_u64().ok_or("checkpoint next_id")?;
        run.judge.evaluations = c["evaluations"].as_u64().unwrap_or(0);
        run.judge.scored = c["scored"].as_u64().unwrap_or(0);
        run.judge.compiled = c["compiled"].as_u64().unwrap_or(0);
        run.judge.hits = c["hits"].as_u64().unwrap_or(0);
        run.judge.bugs = c["bugs"].as_u64().unwrap_or(0);
        run.judge.setmem = c["setmem"].as_u64().unwrap_or(0);
        run.stage = serde_json::from_value(c["stage"].clone()).map_err(|e| format!("checkpoint stage: {e}"))?;
        run.first_exact = serde_json::from_value(c["first_exact"].clone()).map_err(|e| format!("checkpoint first_exact: {e}"))?;
        run.hof = serde_json::from_value(c["hof"].clone()).map_err(|e| format!("checkpoint hof: {e}"))?;
        run.tel.restore(c["best_ever"].as_f64(), run.judge.evaluations);
        let mut rng = GeneRng::new(seed ^ 0x5245_5355_4d45);
        for i in c["population"].as_array().ok_or("checkpoint population")? {
            let tree = tree_of_json(&i["tree"])?;
            let gene = write_back(&contract, &layout, &run.gen.random_gene(&mut rng)?, &tree).map_err(|e| format!("re-encoding a checkpoint tree: {e}"))?;
            cands.push(Individual {
                id: i["id"].as_u64().unwrap_or(0),
                parents: i["parents"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default(),
                born: i["born"].as_u64().unwrap_or(0) as u32,
                operator: "resumed".into(),
                source: program(&k, &sig, &tree)?,
                gene,
                tree,
                eval: None,
            });
        }
        println!("# {} resumed at generation {generation}: {} individuals, {} evaluations spent, stage {}", utc(), cands.len(), run.judge.evaluations, run.stage.number());
    } else {
        generation = 0;
        if regime == "P" {
            let start_gene = write_back(&contract, &layout, &run.gen.random_gene(&mut rng0)?, &start_tree).map_err(|e| format!("the start program's write-back: {e}"))?;
            let s = run.fresh(&sig, vec![], &start, start_gene.clone(), 0)?;
            let sid = s.id;
            let ssrc = s.source.clone();
            cands.push(s);
            let vs = variants(&k, &bridge, &start_tree, 3, 32, pop * 2)?;
            println!("# {} e-graph: {} variants of the {start} program", utc(), vs.len());
            for v in &vs {
                let g = write_back(&contract, &layout, &start_gene, v).map_err(|e| format!("a variant's write-back: {e}"))?;
                let ind = run.fresh(&sig, vec![sid], "egraph_exact", g, 0)?;
                let mut m = Mutation::new(EventKind::EgraphExact, "egraph_variant", format!("an e-graph variant of the {start} program (exact identities of wrapping i32 and min/max)"));
                m.id = Some(ind.id);
                m.parents = vec![sid];
                m.text_before = Some(ssrc.clone());
                m.text_after = Some(ind.source.clone());
                run.tel.mutation(0, m)?;
                cands.push(ind);
            }
            let trees: Vec<Tree> = cands.iter().map(|c| c.tree.clone()).collect();
            let genes: Vec<Vec<u32>> = cands.iter().map(|c| c.gene.clone()).collect();
            let mut table = AdfTable::default();
            let fe = FoldEgraph { kingdom: &k, bridge: &bridge };
            let fold = fold_saturated(&mut table, &fe, &contract, &layout, &const_len, &genes, &trees, &Budget::default())?;
            let mut m = Mutation::new(EventKind::SaturatedFold, "saturated_fold", format!("{} shared parts over {} genes; {} forms differ", fold.used.len(), genes.len(), fold.reformed));
            m.count = Some(fold.used.len() as u32);
            m.nodes = Some(fold.reformed as u32);
            run.tel.mutation(0, m)?;
            for (i, f) in fold.forms.iter().enumerate() {
                if !cands.iter().any(|c| &c.tree == f) {
                    if let Ok(g) = write_back(&contract, &layout, &start_gene, f) {
                        let parent = cands[i].id;
                        let ind = run.fresh(&sig, vec![parent], "saturated_fold", g, 0)?;
                        cands.push(ind);
                    }
                }
            }
            // Keep the start and the smallest variants (the e-graph orders them by size).
            cands.truncate(pop);
        } else if matches!(arg(&args, "--gen0").as_deref(), Some("fable-mutants" | "fable-mutants-only")) {
            // Generation 0 from ONE seed program (Fable's, cached in the pool) and its mutants by the run's operator.
            let dir = PathBuf::from(arg(&args, "--pool").ok_or("--gen0 fable-mutants needs --pool DIR (seed.tsv: examples/futhark_pool.rs seed)")?);
            let keep_seed = arg(&args, "--gen0").as_deref() == Some("fable-mutants");
            cands = fable_mutants_population(&mut run, &sig, &dir, pop, keep_seed, &mut rng0)?;
        } else if canvas.is_some() && arg(&args, "--gen0").as_deref() != Some("canvas") {
            // Generation 0 of `--start llm-ring` (R, C): Sonnet's task alternatives, distinct, on ring canvases. With
            // `--gen0 canvas` it is fresh ring-canvas genes instead (the branch below), the alternatives not used.
            let bar: usize = num(&args, "--distinct-edits", 5)?;
            cands = alternatives_population(&mut run, &sig, &pool_programs, &reference, pop, bar, &mut rng0)?;
        } else {
            while cands.len() < pop {
                let g = run.gen.random_gene(&mut rng0)?;
                let ind = run.fresh(&sig, vec![], run.gen.fresh_label(), g, 0)?;
                cands.push(ind);
            }
        }
    }

    let mut series = if resume { std::fs::read_to_string(out.join("series.tsv")).unwrap_or_default() } else { String::new() };
    if series.is_empty() {
        series = String::from("generation\tutc\tseconds\tevaluations\tcompiled\tcache_hits\tbugs\tstage\tbest_hff\tbest_theta_q\tbest_theta_s\tmedian_hff\tbest_ratio\tbest_exact_train\tbest_exact_held\tbest_nodes\tbest_work\tdistinct\tdistinct_outputs\tscored\tgen_seconds\n");
    }
    let mut stages_tsv = if resume { std::fs::read_to_string(out.join("stages.tsv")).unwrap_or_default() } else { String::new() };
    let mut population: Vec<Individual> = Vec::new();
    let mut stopped_by = "early_stop";
    let mut last_generation = generation;
    // The best the pump last fired on.
    let mut last_pumped: Option<String> = None;
    let mut per_gen = if resume { std::fs::read_to_string(out.join("per_generation.tsv")).unwrap_or_default() } else { String::new() };
    let mut prev_so_far: Option<f64> = per_gen.lines().last().and_then(|l| l.split('\t').nth(2)).and_then(|v| v.parse().ok());
    // THE TEST CASE (k-means, the round-2 review's N2): the champion run on every test case, interleaved with the
    // seed, in every generation its text changes; `OUT/test_case.tsv` keeps the learning curve, `last_test` the
    // latest for the report. Never a fitness input.
    let mut test_tsv = if resume { std::fs::read_to_string(out.join("test_case.tsv")).unwrap_or_default() } else { String::new() };
    let mut last_tested: Option<String> = None;
    let mut last_test: Vec<TestCaseRun> = Vec::new();
    loop {
        if run.tel.stop_requested() {
            stopped_by = "killed";
            break;
        }
        if run.judge.evaluations >= budget {
            stopped_by = "n_gen";
            break;
        }
        let gen_start = Instant::now();
        run.judge.gen_stages = Stages::default();
        let evals_before = run.judge.evaluations;
        // ---- evaluate (batch mode: every new program of the generation in one batch first) ----
        {
            let items: Vec<(&Tree, &str)> = cands.iter().map(|c| (&c.tree, c.source.as_str())).collect();
            run.judge.prefetch(&items, run.stage)?;
        }
        let mut next = Vec::with_capacity(cands.len());
        let mut timed_out = false;
        let mut exact_stop = false;
        for mut c in std::mem::take(&mut cands) {
            if started.elapsed() > limit {
                timed_out = true;
                break;
            }
            if run.judge.evaluations >= budget {
                break;
            }
            run.score(generation, &mut c)?;
            next.push(c);
            if stop_at_exact && run.first_exact.is_some() {
                exact_stop = true;
                break;
            }
        }
        if timed_out {
            stopped_by = "time";
        }
        if exact_stop {
            stopped_by = "early_stop";
            println!("# {} --stop-at-exact: stopping at the first exact program (evaluation {})", utc(), run.judge.evaluations);
        }
        if next.is_empty() {
            break;
        }
        if !population.is_empty() {
            // Every generation, each operator's children and how many beat their best parent (decision-critical).
            let prev: HashMap<u64, f64> = population.iter().map(|i| (i.id, i.fit())).collect();
            let mut by = OperatorTally::default();
            for c in next.iter().filter(|c| c.operator != "elite") {
                let pf = c.parents.iter().filter_map(|p| prev.get(p)).cloned().fold(f64::INFINITY, f64::min);
                by.add(&c.operator, c.fit() < pf);
            }
            println!("# {} gen {generation} operators: {}", utc(), by.line());
        }
        population = next;
        population.sort_by(|a, b| a.fit().total_cmp(&b.fit()));
        // ---- the stage: quality only until the best is exact on every case, then quality and speed ----
        let best_perfect = population[0].eval.as_ref().is_some_and(|e| e.quality.perfect());
        let new_stage = run.stage.next(best_perfect);
        if new_stage != run.stage && !timed_out && !exact_stop {
            if backend == Backend::C {
                return Err("the best is exact on every case: stage 2 needs GPU timing, which the c backend does not have (run with --backend gpu)".into());
            }
            let msg = format!("stage 1 -> 2 at generation {generation}, evaluation {}: the best is exact on every case (training and held-out); fitness is now the two-level HFF with speed", run.judge.evaluations);
            println!("# {} {msg}", utc());
            run.tel.writer().event(generation, EventKind::StageChange, msg, None, Some(2.0))?;
            run.stage = new_stage;
            {
                let items: Vec<(&Tree, &str)> = population.iter().map(|c| (&c.tree, c.source.as_str())).collect();
                run.judge.prefetch(&items, run.stage)?;
            }
            // The population is re-scored in stage 2 so that selection compares like with like (not new evaluations).
            // (Their quality is known, so none of them is a new evaluation.)
            // One hall of fame a stage (fault 16): stage 1's is kept in hall_of_fame_stage1/, and stage 2's starts
            // from the re-scored population, so its rank 1 is the fastest exact program, not the first one.
            let rows = run.hof.write_in(&out, "hall_of_fame_stage1")?;
            run.tel.note(generation, format!("stage 1's hall of fame ({} entries) is in hall_of_fame_stage1/", rows.len()))?;
            run.hof = HallOfFame::new(5);
            run.champion = None;
            for ind in population.iter_mut() {
                let e = run.judge.evaluate(&ind.tree, &ind.source, run.stage)?;
                ind.eval = Some(e.clone());
                let text = sexp(&k, &ind.tree);
                run.hof.offer(FameEntry { key: text_hash(&ind.source), fitness: e.fitness, born: ind.born, text: format!("{text}\n\n{}", ind.source), nodes: Some(ind.tree.level_order().len() as u32), picture: None, extras: BTreeMap::new() });
                run.offer_champion(ind);
            }
            population.sort_by(|a, b| a.fit().total_cmp(&b.fit()));
        }
        let fits: Vec<f64> = population.iter().map(Individual::fit).collect();
        let distinct = {
            let mut s: Vec<&str> = population.iter().map(|i| i.source.as_str()).collect();
            s.sort_unstable();
            s.dedup();
            s.len()
        };
        let distinct_outputs = population.iter().filter_map(|i| i.eval.as_ref().map(|e| e.quality.behaviour.as_str())).collect::<HashSet<_>>().len();
        let best = population[0].clone();
        let be = best.eval.clone().ok_or("an unscored best")?;
        let mut groups = BTreeMap::new();
        groups.insert("quality".to_string(), be.theta_q);
        groups.insert("speed".to_string(), be.theta_s);
        let mut extras = BTreeMap::new();
        extras.insert("stage".to_string(), f64::from(run.stage.number()));
        extras.insert("median_hff".to_string(), median(&fits));
        extras.insert("exact_train".to_string(), be.quality.exact_train as f64);
        extras.insert("exact_held".to_string(), be.quality.exact_held as f64);
        if be.ratio.is_finite() {
            extras.insert("speed_ratio".to_string(), be.ratio);
        }
        extras.insert("evaluations".to_string(), run.judge.evaluations as f64);
        extras.insert("scored".to_string(), run.judge.scored as f64);
        extras.insert("distinct_outputs".to_string(), distinct_outputs as f64);
        extras.insert("compiled".to_string(), run.judge.compiled as f64);
        extras.insert("cache_hits".to_string(), run.judge.hits as f64);
        extras.insert("bugs".to_string(), run.judge.bugs as f64);
        extras.insert("max_work".to_string(), run.judge.max_work as f64);
        // Where the generation's wall time went (seconds a stage) and the evaluation rate.
        let gen_wall = gen_start.elapsed().as_secs_f64();
        let gen_evals = run.judge.evaluations - evals_before;
        for (name, v) in Stages::NAMES.iter().zip(run.judge.gen_stages.values()) {
            extras.insert(format!("stage_{name}_s"), v);
        }
        extras.insert("stage_all_s".to_string(), run.judge.gen_stages.total());
        extras.insert("gen_seconds".to_string(), gen_wall);
        extras.insert("evaluations_per_second".to_string(), gen_evals as f64 / gen_wall.max(1e-9));
        extras.insert("batched".to_string(), f64::from(u8::from(mode.batched())));
        extras.insert("batch_halvings".to_string(), run.judge.batch_splits as f64);
        if mode == EvalMode::Rust && scheme.is_converged() {
            extras.insert("upload_s".to_string(), run.judge.upload_seconds);
            extras.insert("resident_hits".to_string(), run.judge.resident_hits as f64);
            run.judge.upload_seconds = 0.0;
            run.judge.resident_hits = 0;
        }
        let before = run.tel.individuals();
        run.tel.generation(GenerationStats { generation, fitness: &fits, distinct: Some(distinct as u32), angle_groups: groups, extras, evaluated: run.judge.evaluations.saturating_sub(before) })?;
        let gen_seconds = gen_start.elapsed().as_secs_f64();
        series.push_str(&format!(
            "{generation}\t{}\t{:.1}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}\t{}\t{}\t{}\t{}\t{distinct}\t{distinct_outputs}\t{}\t{gen_seconds:.2}\n",
            utc(),
            spent + started.elapsed().as_secs_f64(),
            run.judge.evaluations,
            run.judge.compiled,
            run.judge.hits,
            run.judge.bugs,
            run.stage.number(),
            be.fitness,
            be.theta_q,
            be.theta_s,
            median(&fits),
            be.ratio,
            be.quality.exact_train,
            be.quality.exact_held,
            best.tree.level_order().len(),
            be.quality.work,
            run.judge.scored,
        ));
        std::fs::write(out.join("series.tsv"), &series).map_err(|e| e.to_string())?;
        // PER GENERATION (Andrew, 2026-10-29: whether the search learns, generation by generation): the best-so-far
        // is the stage's champion (the hall of fame's rank 1), the improvement its fall since the last generation.
        let so_far = run.champion.as_ref().map_or(be.fitness, Individual::fit);
        let zero_share = population.iter().filter(|i| i.eval.as_ref().is_some_and(|e| e.quality.all_zero())).count() as f64 / population.len() as f64;
        // THE HFF PERCENTILE beside the angle (`futhark-kmeans-better`, Andrew's specification of 2026-11-03): F(theta)
        // over the d inputs of the one flat vector (every k-means individual's vector has the same d).
        let percentile = if scheme.uses_cliff() {
            let d = hff_dimensions(&be, quality_weight);
            Some((fuller::fitness::hff_percentile(so_far, d)?, fuller::fitness::hff_percentile(be.fitness, d)?, d))
        } else {
            None
        };
        if per_gen.is_empty() {
            per_gen = String::from("generation\tevaluations\tbest_so_far_hff\tmedian_hff\tbest_this_generation_hff\tdistinct_outputs\tall_zero_share\timprovement_in_best_so_far");
            per_gen.push_str(if percentile.is_some() { "\tbest_so_far_hff_percentile\tbest_this_generation_hff_percentile\thff_dimensions_d\n" } else { "\n" });
        }
        per_gen.push_str(&format!("{generation}\t{}\t{so_far:.6}\t{:.6}\t{:.6}\t{distinct_outputs}\t{zero_share:.4}\t{:.6}", run.judge.evaluations, median(&fits), be.fitness, prev_so_far.map_or(0.0, |p| p - so_far)));
        per_gen.push_str(&percentile.map_or("\n".to_string(), |(a, b, d)| format!("\t{a:.6e}\t{b:.6e}\t{d}\n")));
        prev_so_far = Some(so_far);
        std::fs::write(out.join("per_generation.tsv"), &per_gen).map_err(|e| e.to_string())?;
        if stages_tsv.is_empty() {
            stages_tsv = format!("generation\tutc\teval\tevaluations\tgen_seconds\tevaluations_per_second\t{}\tall\n", Stages::NAMES.join("\t"));
        }
        stages_tsv.push_str(&format!(
            "{generation}\t{}\t{mode:?}\t{gen_evals}\t{gen_seconds:.3}\t{:.4}\t{}\t{:.3}\n",
            utc(),
            gen_evals as f64 / gen_seconds.max(1e-9),
            run.judge.gen_stages.values().iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join("\t"),
            run.judge.gen_stages.total()
        ));
        std::fs::write(out.join("stages.tsv"), &stages_tsv).map_err(|e| e.to_string())?;
        println!(
            "{} gen {generation}: stage {}, best HFF {:.5} (q {:.5}, s {:.5}, ratio {:.3}), exact {}+{} of {}, median {:.5}, {distinct} distinct texts, {distinct_outputs} distinct outputs; {} evaluations ({} compiled, {} hits, {} bugs); {gen_seconds:.1} s",
            utc(),
            run.stage.number(),
            be.fitness,
            be.theta_q,
            be.theta_s,
            be.ratio,
            be.quality.exact_train,
            be.quality.exact_held,
            run.judge.cases.iter().filter(|c| c.split.in_fitness()).count(),
            median(&fits),
            run.judge.evaluations,
            run.judge.compiled,
            run.judge.hits,
            run.judge.bugs
        );
        if scheme.is_converged() {
            // THE K-MEANS LINES A GENERATION (Andrew): the HFF of the best and its four measures against the seed on
            // every fitness case (each r/(1+r), 1/2 equal to the seed, below better), then the champion on the test
            // case when its text changed.
            let (qw, sw) = run.judge.converged_widths();
            let names: Vec<&str> = run.judge.cases.iter().filter(|c| c.split.in_fitness()).map(|c| c.name.as_str()).collect();
            let per_case: Vec<String> = names
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    let q = &be.quality.inputs[i * qw..(i + 1) * qw];
                    let s = &be.speed[i * sw..(i + 1) * sw];
                    format!("{n} q [{}] s [{}]", q.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" "), s.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" "))
                })
                .collect();
            match percentile {
                Some((_, p, d)) => println!("# {} gen {generation} best HFF {:.5} (HFF percentile F {p:.4e} at d = {d}) against the seed (quality: cosine inertia, percentile, convergence, each the cliff of x; speed: seconds, iterations): {}", utc(), be.fitness, per_case.join("; ")),
                None => println!("# {} gen {generation} best HFF {:.5} against the seed (cosine inertia, percentile; seconds, iterations): {}", utc(), be.fitness, per_case.join("; ")),
            }
            let champion = run.champion.clone().unwrap_or_else(|| best.clone());
            if last_tested.as_deref() != Some(champion.source.as_str()) && run.judge.cases.iter().any(|c| !c.split.in_fitness()) {
                let t_test = Instant::now();
                let rows = run.judge.converged_on_test(&champion.tree, &champion.source)?;
                for r in &rows {
                    println!("# {} gen {generation} champion (id {}, HFF {:.5}) on the test case {} ({:.1} s)", utc(), champion.id, champion.fit(), r.line(), t_test.elapsed().as_secs_f64());
                    if test_tsv.is_empty() {
                        test_tsv = String::from("generation\tevaluations\tchampion_id\tchampion_hff\tcase\tquality_inputs\tspeed_inputs\tchampion_inertia\tchampion_log10p\tchampion_log10p_all\tchampion_seconds\tchampion_iterations\tchampion_finished\tseed_inertia\tseed_log10p\tseed_log10p_all\tseed_seconds\tseed_iterations\n");
                    }
                    let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(",");
                    test_tsv.push_str(&format!(
                        "{generation}\t{}\t{}\t{:.6}\t{}\t{}\t{}\t{:.6}\t{:.4}\t{:.4}\t{:.6}\t{}\t{}\t{:.6}\t{:.4}\t{:.4}\t{:.6}\t{}\n",
                        run.judge.evaluations, champion.id, champion.fit(), r.case, fmt(&r.quality), fmt(&r.speed), r.candidate.inertia, r.candidate.mean_log10_p, r.candidate.mean_log10_p_all, r.candidate.seconds, r.candidate.iterations, r.candidate.finished, r.seed.inertia, r.seed.mean_log10_p, r.seed.mean_log10_p_all, r.seed.seconds, r.seed.iterations
                    ));
                }
                std::fs::write(out.join("test_case.tsv"), &test_tsv).map_err(|e| e.to_string())?;
                last_tested = Some(champion.source.clone());
                last_test = rows;
            }
        }
        let rng_state = stream.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ u64::from(generation + 1);
        if generation.is_multiple_of(checkpoint_every.max(1)) {
            let rows = run.hof.write(&out)?;
            run.tel.hall_of_fame(generation, rows)?;
            run.checkpoint(generation, &population, spent + started.elapsed().as_secs_f64(), &regime, rng_state)?;
        }
        run.ledger.flush()?;
        last_generation = generation;
        if timed_out || exact_stop {
            break;
        }

        // ---- the next generation; its random numbers come from (seed, generation), so a resume continues it ----
        let mut rng = GeneRng::new(rng_state);
        let mut children: Vec<Individual> = Vec::with_capacity(pop);
        if regime == "C" {
            // The random-search control: a fresh random gene per evaluation, no selection.
            while children.len() < pop {
                let g = run.gen.random_gene(&mut rng)?;
                let ind = run.fresh(&sig, vec![], run.gen.fresh_label(), g, generation + 1)?;
                children.push(ind);
            }
            cands = children;
            generation += 1;
            continue;
        }
        // At most one elite an identity (a behaviour in stage 1): an elite is carried over, never a new evaluation.
        let stage = run.stage;
        let mut taken: Vec<String> = Vec::new();
        for ind in &population {
            if children.len() >= elites {
                break;
            }
            if !taken.iter().any(|t| t == ind.identity(stage)) {
                taken.push(ind.identity(stage).to_string());
                let mut e = ind.clone();
                e.operator = "elite".into();
                children.push(e);
            }
        }
        let g1 = generation + 1;
        if pump_every > 0 && generation > 0 && generation.is_multiple_of(pump_every) && last_pumped.as_deref() != Some(population[0].source.as_str()) {
            // The pump: the e-graph variants of the best enter, only when the best has changed since the last pump
            // (plan section F, fault 9: on a plateau it sent the same variants again), and only variants not scored
            // before (a repeat would take a child's place for nothing).
            let b = population[0].clone();
            last_pumped = Some(b.source.clone());
            for v in variants(&k, &bridge, &b.tree, 2, 16, 4)? {
                if children.len() >= pop {
                    break;
                }
                if run.judge.quality_cache.contains_key(&program(&k, &sig, &v)?) {
                    continue;
                }
                if let Ok(g) = write_back(&contract, &layout, &b.gene, &v) {
                    let ind = run.fresh(&sig, vec![b.id], "pump_egraph", g, g1)?;
                    let mut m = Mutation::new(EventKind::Pump, "egraph_variant", "the pump: an e-graph variant of the best".into());
                    m.id = Some(ind.id);
                    m.parents = vec![b.id];
                    m.text_before = Some(sexp(&k, &b.tree));
                    m.text_after = Some(sexp(&k, &ind.tree));
                    m.fitness_before = Some(b.fit());
                    m.entered = Some(true);
                    run.tel.mutation(generation, m)?;
                    children.push(ind);
                }
            }
        }
        if fold_every > 0 && generation > 0 && generation.is_multiple_of(fold_every) {
            // The saturated share-equivalence fold over the population: its jointly extracted forms enter.
            let trees: Vec<Tree> = population.iter().map(|c| c.tree.clone()).collect();
            let genes: Vec<Vec<u32>> = population.iter().map(|c| c.gene.clone()).collect();
            let mut table = AdfTable::default();
            let fe = FoldEgraph { kingdom: &k, bridge: &bridge };
            let fold = fold_saturated(&mut table, &fe, &contract, &layout, &const_len, &genes, &trees, &Budget::default())?;
            let mut entered = 0u32;
            for (i, f) in fold.forms.iter().enumerate() {
                if children.len() >= pop {
                    break;
                }
                if f != &trees[i] && !children.iter().any(|c| &c.tree == f) {
                    if let Ok(g) = write_back(&contract, &layout, &genes[i], f) {
                        let parent = population[i].id;
                        let ind = run.fresh(&sig, vec![parent], "saturated_fold", g, g1)?;
                        children.push(ind);
                        entered += 1;
                    }
                }
            }
            let mut m = Mutation::new(EventKind::SaturatedFold, "saturated_fold", format!("{} shared parts over {} genes; {} forms differ, {entered} entered", fold.used.len(), genes.len(), fold.reformed));
            m.count = Some(fold.used.len() as u32);
            m.nodes = Some(fold.reformed as u32);
            run.tel.mutation(generation, m)?;
        }
        let tournament = |rng: &mut GeneRng| -> usize {
            let mut best = rng.below(population.len());
            for _ in 0..2 {
                let c = rng.below(population.len());
                if population[c].fit() < population[best].fit() {
                    best = c;
                }
            }
            best
        };
        // Marriage: pairs of parents whose expressed trees differ, two offspring each.
        let n_pairs = ((pop.saturating_sub(children.len())) as f64 * marriage / 2.0).floor() as usize;
        let (mut married, mut no_partner, mut no_operator) = (0u32, 0u32, 0u32);
        for _ in 0..n_pairs {
            let a = tournament(&mut rng);
            let pool: Vec<usize> = (0..population.len()).filter(|i| population[*i].identity(stage) != population[a].identity(stage)).collect();
            if pool.is_empty() {
                no_partner += 1;
                continue;
            }
            let mut b = pool[rng.below(pool.len())];
            for _ in 0..2 {
                let c = pool[rng.below(pool.len())];
                if population[c].fit() < population[b].fit() {
                    b = c;
                }
            }
            match run.gen.marry(&population[a], &population[b], &mut rng)? {
                Some((ga, gb)) => {
                    let (ia, ib) = (population[a].id, population[b].id);
                    for g in [ga, gb] {
                        let ind = run.fresh(&sig, vec![ia, ib], "marriage", g, g1)?;
                        children.push(ind);
                    }
                    married += 1;
                }
                None => no_operator += 1,
            }
        }
        if n_pairs > 0 {
            let mut m = Mutation::new(EventKind::Marriage, "subtree", format!("{married} pairs made {} offspring (typed subtree crossover, parents of different outputs in stage 1, texts in stage 2); {no_partner} without a partner, {no_operator} with an empty domain", 2 * married));
            m.count = Some(married);
            run.tel.mutation(generation, m)?;
        }
        // Half the rest from crossover (both children kept; each then mutated or not, a separate recorded choice:
        // plan section F, fault 20), half from mutation alone. A crossover pair's parents differ in identity.
        while children.len() < pop {
            let a = tournament(&mut rng);
            let m_of = |rng: &mut GeneRng| if regime == "P" { 1 + rng.below(2) } else { 1 + rng.below(4) };
            if let Some(o) = &ops {
                // One operator drawn by its probability among the non-marriage operators.
                let pick = pick_op(o, "marriage", rng.below(1_000_000) as f64 / 1e6).unwrap_or("point");
                if pick == "cross-rows" || pick == "cross-rel" || pick == "cross-rows-coding" {
                    // ROW CROSSOVER: a partner of another identity (as crossover's), both children kept, each its
                    // own mutation event.
                    let mut b = tournament(&mut rng);
                    for _ in 0..8 {
                        if population[b].identity(stage) != population[a].identity(stage) {
                            break;
                        }
                        b = tournament(&mut rng);
                    }
                    let (mut ga, mut gb) = (population[a].gene.clone(), population[b].gene.clone());
                    let name = match pick {
                        "cross-rows" => "cross_rows",
                        "cross-rows-coding" => "cross_rows_coding",
                        _ => "cross_rel",
                    };
                    let what = if population[b].identity(stage) == population[a].identity(stage) {
                        None
                    } else if pick == "cross-rows" {
                        cross_rows_once(run.gen.layout, &mut ga, &mut gb, &mut rng)?
                    } else if pick == "cross-rows-coding" {
                        cross_rows_coding_once(run.gen.contract, run.gen.layout, run.gen.const_len, &mut ga, &mut gb, &mut rng)?
                    } else {
                        cross_rel_once(run.gen.layout, &mut ga, &mut gb, &mut rng)?
                    };
                    match what {
                        Some(what) => {
                            for (g, before) in [(ga, a), (gb, b)] {
                                if children.len() >= pop {
                                    break;
                                }
                                let ind = run.fresh(&sig, vec![population[a].id, population[b].id], name, g, g1)?;
                                run.tel.mutation(generation, row_event(run.gen.k, name, &what, &ind, &population[before].tree))?;
                                children.push(ind);
                            }
                        }
                        None => {
                            // No partner of another identity (or a one-row table): the child is mutation alone, as
                            // crossover's is.
                            let mut gene = population[a].gene.clone();
                            let m = m_of(&mut rng);
                            let walk = run.gen.mutation(&mut gene, m, &mut rng)?;
                            let ind = run.fresh(&sig, vec![population[a].id], if ring_walk { "ring_walk" } else { "mutation" }, gene, g1)?;
                            if let Some(w) = walk {
                                run.tel.mutation(generation, run.gen.walk_event(&w, &ind, population[a].id))?;
                            }
                            children.push(ind);
                        }
                    }
                    continue;
                }
                if pick != "crossover" {
                    let mut gene = population[a].gene.clone();
                    let coding = fuller::futhark_rings::coding_cells(run.gen.contract, run.gen.layout, &gene, run.gen.const_len)?;
                    let done = match pick {
                        "point" => {
                            let m = m_of(&mut rng);
                            run.gen.mutate(&mut gene, m, &mut rng)?;
                            true
                        }
                        "invert" => invert_once(run.gen.layout, &mut gene, &coding, &mut rng)?.is_some(),
                        "transpose" => transpose_once(run.gen.contract, run.gen.layout, &mut gene, &coding, &mut rng)?.is_some(),
                        "invert-rows" | "transpose-rows" | "invert-rel" | "transpose-rel" | "invert-rows-coding" | "transpose-rows-coding" => {
                            let (what, name) = match pick {
                                "invert-rows" => (invert_rows_once(run.gen.layout, &mut gene, &mut rng)?, "invert_rows"),
                                "invert-rows-coding" => (invert_rows_coding_once(run.gen.layout, &mut gene, &coding, &mut rng)?, "invert_rows_coding"),
                                "transpose-rows-coding" => (transpose_rows_coding_once(run.gen.contract, run.gen.layout, &mut gene, &coding, &mut rng)?, "transpose_rows_coding"),
                                "transpose-rows" => (transpose_rows_once(run.gen.contract, run.gen.layout, &mut gene, &mut rng)?, "transpose_rows"),
                                "invert-rel" => (invert_rel_once(run.gen.layout, &mut gene, &mut rng)?, "invert_rel"),
                                _ => (transpose_rel_once(run.gen.contract, run.gen.layout, &mut gene, &mut rng)?, "transpose_rel"),
                            };
                            if let Some(what) = what {
                                let ind = run.fresh(&sig, vec![population[a].id], name, gene, g1)?;
                                run.tel.mutation(generation, row_event(run.gen.k, name, &what, &ind, &population[a].tree))?;
                                children.push(ind);
                            }
                            continue;
                        }
                        _ => {
                            let rings = run.gen.canvas.ok_or("--ops walk needs the rings: --start llm-ring")?;
                            match rings.walk(run.gen.layout, &mut gene, &coding, 4, &mut rng)? {
                                Some(w) => {
                                    let ind = run.fresh(&sig, vec![population[a].id], "ring_walk", gene, g1)?;
                                    run.tel.mutation(generation, run.gen.walk_event(&w, &ind, population[a].id))?;
                                    children.push(ind);
                                    continue;
                                }
                                None => false,
                            }
                        }
                    };
                    if done {
                        let name = match pick {
                            "point" => "mutation",
                            "invert" => "inversion",
                            _ => "transposition",
                        };
                        let ind = run.fresh(&sig, vec![population[a].id], name, gene, g1)?;
                        children.push(ind);
                    }
                    continue;
                }
            }
            if ops.is_some() || rng.below(2) == 0 {
                let mut b = tournament(&mut rng);
                for _ in 0..8 {
                    if population[b].identity(stage) != population[a].identity(stage) {
                        break;
                    }
                    b = tournament(&mut rng);
                }
                if population[b].identity(stage) != population[a].identity(stage) {
                    let (mut ga, mut gb) = (population[a].gene.clone(), population[b].gene.clone());
                    run.gen.crossover(&mut ga, &mut gb, &mut rng)?;
                    for g in [ga, gb] {
                        if children.len() >= pop {
                            break;
                        }
                        let mut g = g;
                        let (op, walk) = if rng.below(2) == 0 {
                            let m = m_of(&mut rng);
                            let w = run.gen.mutation(&mut g, m, &mut rng)?;
                            (if ring_walk { "crossover+ring_walk" } else { "crossover+mutation" }, w)
                        } else {
                            ("crossover", None)
                        };
                        let ind = run.fresh(&sig, vec![population[a].id, population[b].id], op, g, g1)?;
                        if let Some(w) = walk {
                            run.tel.mutation(generation, run.gen.walk_event(&w, &ind, population[a].id))?;
                        }
                        children.push(ind);
                    }
                    continue;
                }
            }
            let mut gene = population[a].gene.clone();
            let m = m_of(&mut rng);
            let walk = run.gen.mutation(&mut gene, m, &mut rng)?;
            let ind = run.fresh(&sig, vec![population[a].id], if ring_walk { "ring_walk" } else { "mutation" }, gene, g1)?;
            if let Some(w) = walk {
                run.tel.mutation(generation, run.gen.walk_event(&w, &ind, population[a].id))?;
            }
            children.push(ind);
        }
        children.truncate(pop);
        cands = children;
        generation += 1;
    }
    if population.is_empty() {
        return Err("no generation completed within the guard".into());
    }
    let seconds = spent + started.elapsed().as_secs_f64();
    run.checkpoint(last_generation, &population, seconds, &regime, 0)?;
    let rows = run.hof.write(&out)?;
    run.tel.hall_of_fame(last_generation, rows)?;
    println!("# {} stopped ({stopped_by}) after generation {last_generation}, {:.1} min; {} evaluations ({} compiled, {} cache hits, {} bugs); {:.2} evaluations a second", utc(), seconds / 60.0, run.judge.evaluations, run.judge.compiled, run.judge.hits, run.judge.bugs, run.judge.evaluations as f64 / seconds.max(1e-9));

    // ---- the report: every champion field from the champion, the hall of fame's rank 1 ----
    let best = run.champion.clone().unwrap_or_else(|| population[0].clone());
    let be = best.eval.clone().ok_or("an unscored best")?;
    std::fs::write(out.join("best.fut"), &best.source).map_err(|e| e.to_string())?;
    let mut rep = format!(
        "problem\t{pname}\nregime\t{regime}\nstart\t{start}\nseed\t{seed}\nbackend\t{backend:?}\neval\t{mode:?}\nstopped_by\t{stopped_by}\nlast_generation\t{last_generation}\nminutes\t{:.2}\nevaluations\t{}\nbudget\t{budget}\ncompiled\t{}\ncache_hits\t{}\nbugs\t{}\nbugs_setmem_compiler_limitation\t{}\nbatch_halvings\t{}\nevaluations_per_second\t{:.3}\nstage\t{}\nfirst_exact_evaluation\t{}\nfirst_exact_generation\t{}\nbest_hff\t{:.6}\nbest_theta_quality\t{:.6}\nbest_theta_speed\t{:.6}\nbest_exact_train\t{}\nbest_exact_held\t{}\nbest_nodes\t{}\nreference_nodes\t{}\nbest_work\t{}\nreference_work\t{}\nbest_program\t{}\nmax_work_seen\t{}\nscored\t{}\nbest_exact_test\t{} of {n_test}\nfirst_exact_test\t{}\nbugs_internal_compiler_error\t{}\nbinary\t{}\n",
        seconds / 60.0,
        run.judge.evaluations,
        run.judge.compiled,
        run.judge.hits,
        run.judge.bugs,
        run.judge.setmem,
        run.judge.batch_splits,
        run.judge.evaluations as f64 / seconds.max(1e-9),
        run.stage.number(),
        run.first_exact.map_or("-".to_string(), |f| f.0.to_string()),
        run.first_exact.map_or("-".to_string(), |f| f.1.to_string()),
        be.fitness,
        be.theta_q,
        be.theta_s,
        be.quality.exact_train,
        be.quality.exact_held,
        best.tree.level_order().len(),
        reference.level_order().len(),
        be.quality.work,
        run.judge.work_of(&reference)?,
        sexp(&k, &best.tree),
        run.judge.max_work,
        run.judge.scored,
        be.quality.exact_test,
        run.first_exact_test.map_or("-".to_string(), |t| format!("{t} of {n_test}")),
        run.judge.ice,
        binary_identity(),
    );
    // The static cost (the matrix kingdom; `-` for a signature problem), at the timing input's sizes.
    rep.push_str(&format!("reference_static_cost\t{reference_static}\nbest_static_cost\t{}\n", static_cost(&best.tree)?));
    if backend == Backend::Gpu {
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
        rep.push_str(&format!("noise_floor_ratios_published_vs_itself\t{}\nnoise_floor_median\t{:.4}\n", fmt(&noise), median(&noise)));
        if !inrun_aa.is_empty() {
            rep.push_str(&format!("inrun_aa_ratios\t{}\ninrun_aa_median\t{:.4}\n", fmt(&inrun_aa), median(&inrun_aa)));
        }
        rep.push_str(&format!("timing_input\t{}\ntiming_cap\t{}\nover_cap_untimed\t{}\n", run.judge.timing.label(), run.judge.timing_cap, run.judge.over_cap_untimed));
        rep.push_str(&format!("best_speed_inputs\t{}\nbest_speed_ratio_median\t{:.4}\n", fmt(&be.speed), be.ratio));
        if be.quality.perfect() {
            let (ratios, got) = run.judge.champion_batch(&best.tree, 11)?;
            if let Some(want) = &run.judge.timing.output {
                rep.push_str(&format!("champion_exact_on_timing_input\t{}\n", if got == fuller::futhark_sig::read_one(want)? { "yes" } else { "NO" }));
            }
            rep.push_str(&format!("best_vs_published_pairs\t11\nbest_vs_published_ratios\t{}\nbest_vs_published_median\t{:.4}\nfaster_rounds\t{}\n", fmt(&ratios), median(&ratios), ratios.iter().filter(|r| **r < 1.0).count()));
            // A speed claim counts only below the noise floor (the least MEDIAN of `--aa-retimes` re-times of the
            // published program by the champion's own path, fault 13) AND with at least 9 of 11 pairs faster (a sign
            // test, one-sided p = 0.033): the rule of the study plan, 2026-10-28 13:10Z, after a quiet A/A set beside
            // a noisy re-time made the published program itself "faster".
            let medians = run.judge.aa_medians(&reference, aa_retimes)?;
            let floor = medians.iter().cloned().fold(f64::INFINITY, f64::min);
            let v = speed_verdict(&ratios, &medians).text();
            rep.push_str(&format!("aa_set_medians\t{}\nnoise_floor_least_aa_median\t{floor:.4}\nspeed_claim_beats_noise_floor\t{}\nverdict\t{v}\n", fmt(&medians), if v.starts_with("beat") { "yes" } else { "no" }));
        }
    }
    if scheme.is_converged() {
        // The k-means scheme's own lines: the candidates not run (over the work cap), the runs that did not finish,
        // and the champion's two input groups.
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
        if scheme.uses_cliff() {
            let d = hff_dimensions(&be, quality_weight);
            rep.push_str(&format!("quality_cliff\t{quality_cliff}\nhff_dimensions_d\t{d}\nbest_hff\t{:.6}\nbest_hff_percentile\t{:.6e}\n", be.fitness, fuller::fitness::hff_percentile(be.fitness, d)?));
            if let Some(h) = run.hof.entries().first() {
                rep.push_str(&format!("best_ever_hff_percentile\t{:.6e}\n", fuller::fitness::hff_percentile(h.fitness, d)?));
            }
        }
        rep.push_str(&format!("quality_weight\t{}\npairs\t{pairs}\nover_work_cap_not_run\t{}\nruns_not_finished\t{}\nbest_quality_inputs\t{}\nbest_speed_inputs\t{}\n", if scheme.is_flat() { quality_weight.to_string() } else { "-".to_string() }, run.judge.over_cap_unrun, run.judge.unfinished, fmt(&be.quality.inputs), fmt(&be.speed)));
        // The seed's runs at the start (the control), a case: iterations, the program's own seconds, cosine inertia,
        // mean log10 p over the points off their centroid, the coincident count.
        if let Some(cj) = &run.judge.converged {
            let seed_lines: Vec<String> = run.judge.cases.iter().filter(|c| c.split.in_fitness()).zip(&cj.seed_runs).map(|(c, r)| format!("{}:{}:{:.4}:{:.5}:{:.2}:{}", c.name, r.iterations, r.seconds, r.inertia, r.mean_log10_p, r.coincident)).collect();
            rep.push_str(&format!("seed_runs_at_start\t{}\n", seed_lines.join(",")));
        }
        // THE NOISE FLOOR (N3) and the champion's margin: the seed against itself through the same interleaving gave
        // these inputs; the floor is the widest distance of its speed inputs from 1/2. The champion's seconds claim is
        // its widest seconds input below 1/2 over the cases (the seconds input is the first of each case's speed
        // pair); it beats the floor when every seconds input is below 1/2 - floor.
        if !aa_speed.is_empty() {
            let (_, sw) = run.judge.converged_widths();
            let floor = aa_speed.iter().map(|x| (x - 0.5).abs()).fold(0.0, f64::max);
            let seconds: Vec<f64> = be.speed.iter().step_by(sw.max(1)).copied().collect();
            let slowest = seconds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            rep.push_str(&format!("seed_aa_quality_inputs\t{}\nseed_aa_speed_inputs\t{}\nseed_aa_speed_floor\t{floor:.4}\nbest_seconds_inputs\t{}\nbest_seconds_margin_below_half\t{:.4}\nspeed_claim_beats_noise_floor\t{}\n", fmt(&aa_quality), fmt(&aa_speed), fmt(&seconds), 0.5 - slowest, if seconds.iter().all(|s| *s < 0.5 - floor) { "yes" } else { "no" }));
        }
        // THE TEST CASE (N2): the champion and the seed on every case that is never a fitness input, the latest run.
        for r in &last_test {
            rep.push_str(&format!("test_case\t{}\n", r.line()));
        }
    }
    // The best-ever program (the hall of fame's rank 1). For C, which has no selection, the best above is only the
    // best of the last random batch, and the best-ever is the run's champion (measured 2026-10-28: pathfinder C seed
    // 2's last batch was at 1.207, its best-ever at 1.076).
    if let Some(h) = run.hof.entries().first() {
        rep.push_str(&format!("best_ever_hff\t{:.6}\nbest_ever_born\t{}\nbest_ever_program\t{}\n", h.fitness, h.born, h.text.lines().next().unwrap_or("")));
    }
    rep.push_str(&format!("{}\n", run.judge.scheme.card_line()));
    std::fs::write(out.join("report.tsv"), &rep).map_err(|e| e.to_string())?;
    print!("{rep}");
    let chart = std::process::Command::new("python3")
        .arg("-I")
        .arg("-c")
        .arg(CHART)
        .arg(out.join("series.tsv"))
        .arg(out.join("chart_hff_vs_generation.png"))
        .arg(format!("{pname}, regime {regime}, seed {seed}: HFF against generation"))
        .output()
        .map_err(|e| format!("python3: {e}"))?;
    if !chart.status.success() {
        return Err(format!("the chart failed: {}", String::from_utf8_lossy(&chart.stderr)));
    }
    run.tel.end(stopped_by, last_generation, Some(be.fitness))?;
    println!("# {} done: {}", utc(), out.display());
    Ok(())
}

/// THE LENGTH d OF A K-MEANS INDIVIDUAL'S FLAT VECTOR (`fitness::qqs`): its quality inputs entered `quality_weight`
/// times and its speed inputs once, the `d` of its HFF percentile.
fn hff_dimensions(e: &Eval, quality_weight: usize) -> usize {
    e.quality.inputs.len() * quality_weight + e.speed.len()
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
