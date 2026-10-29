//! THE GENERAL FUTHARK RUNNER: evolve a program for a published benchmark in the signature family of the shape
//! kingdom (`fuller::futhark_sig`), from its stored cases alone (data in, the published program's exact outputs out;
//! HFF 0 is exact). A new benchmark is a problem folder and a problem entry, never a new runner. The study it serves:
//! `docs/STUDY_evolving_benchmarks.md`.
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
//! FITNESS, staged (`fuller::runner::FitnessStage`, `staged_hff`):
//! * stage 1, quality only, no timing: for every stored case, training and held-out alike, the seven inputs of
//!   `futhark_sig::quality_inputs`, each its own, linear: the share of cells not exact; the shares off by over 1 %,
//!   10 % and 50 %; the mean and largest error over the correct output's range (`e/(1+e)`); the share of neighbouring
//!   pairs in the wrong order. (The pilot of 2026-10-27 used three inputs on HFF's log scale, which put every inexact
//!   program within 0.03 of 1, and ranked its whole population alike for 26 generations.)
//! * stage 2, once the best is exact on every case (a `stage_change` event, never back): the two-level HFF of that
//!   quality group and the speed group: the program's TIMING SLOTS (`fuller::runner::TimingSlots`, `--pairs` of
//!   them, 11 by default), interleaved timed pairs against the published program (the yardstick) on the largest
//!   training case, kept by the program's text and filled before it is scored, so a program re-scored in a later
//!   generation reads the same pairs and is never timed again; each pair is its own speed input, `r/(1+r)` with `r`
//!   the candidate's GPU time over the yardstick's, never averaged. (Three pairs a candidate, before 2026-10-27
//!   21:00Z, ranked one program at 1.287 in the run and 1.000 in an 11-pair re-time.)
//!   Each side of a pair is one process running the program `--runs` times (Futhark's `-r`, after its warm-up run);
//!   its time is the fastest run's sum of GPU-timed passes (the GPU's clock ramps within a process). The A/A noise
//!   floor (the yardstick against itself, 11 pairs) is measured at the start; in P the published program is also
//!   timed against itself THROUGH THE SLOT PATH (compiled as a candidate, `--pairs` pairs), the in-run A/A. A speed
//!   claim counts only when the champion's 11-pair re-time at the end has its median below the least ratio of both
//!   A/A sets (`speed_claim_beats_noise_floor` in the report).
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
//! `saturated_fold`, `marriage`, `stage_change`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use fuller::futhark_kingdom::{FutharkKingdom, GeneRng};
use fuller::futhark_exec::output_with_timeout_bytes;
use fuller::futhark_batch::{Host, Inputs, Member, Outcome, Program, Step};
use fuller::futhark_sig::{build, cases_fingerprint, compare, quality_fingerprint, compile_c, load, load_cases, problem, program, program_cases, quality_input_names, quality_inputs, random_gene, read_binary, sexp, signature, tree_distance, variants, Array, Bridge, Case, FoldEgraph, Instance, Interp, Signature, Split};
use fuller::futhark_wgpu_host::{compile_program_prebuilt, compile_program_prebuilt_staged, prebuild_shim, run_program_bytes, run_program_bytes_staged, CompileStages, RunStages};
use fuller::geneframe::Ty;
use fuller::kingdom_contract::KingdomContract;
use fuller::population_fold::AdfTable;
use fuller::runner::{save_checkpoint, slope_check, staged_hff, FameEntry, FitnessStage, HallOfFame, Ledger, LedgerLine, TimingSlots};
use fuller::saturated_fold::{fold_saturated, Budget};
use fuller::telemetry::{EventKind, GenerationStats, Mutation, RunInfo, RunTelemetry};
use fuller::typed_gep::{express, write_back, Domain, GeneLayout, Tree};
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

fn median(v: &[f64]) -> f64 {
    let mut s: Vec<f64> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
    s.sort_by(|a, b| a.total_cmp(b));
    match s.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => s[n / 2],
        n => 0.5 * (s[n / 2 - 1] + s[n / 2]),
    }
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
/// discards for a freshly compiled executable (`fresh`), then each pair, its order alternating, as `Judge::time_pairs`.
fn pair_steps(missing: usize, fresh: bool, runs: usize) -> Vec<Step> {
    let own = Step::Time { yardstick: false, runs };
    let yard = Step::Time { yardstick: true, runs };
    let mut v = Vec::with_capacity(2 * missing + 1);
    if fresh {
        v.push(own);
    }
    for i in 0..missing {
        if i % 2 == 0 {
            v.extend([own, yard]);
        } else {
            v.extend([yard, own]);
        }
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
    let mut r = Vec::with_capacity(missing);
    for i in 0..missing {
        let a = o.next().ok_or("our software has a bug: a pair cut short")?.fastest_ns()?;
        let b = o.next().ok_or("our software has a bug: a pair cut short")?.fastest_ns()?;
        let (c, y) = if i % 2 == 0 { (a, b) } else { (b, a) };
        r.push(c / y);
    }
    Ok(r)
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
    fn of(p: &fuller::futhark_sig::Problem, sig: &Signature, cases: &[Case], build_dir: &Path) -> Result<TimingInput, String> {
        if let Some(stdin) = fuller::futhark_sig::timing_input(p)? {
            let inputs = read_binary(&stdin)?;
            let exe = compile_c(p.yardstick, build_dir)?;
            let output = fuller::futhark_sig::run_exe(&exe, &stdin)?;
            return Ok(TimingInput { name: format!("{}'s timing input", p.name), sizes: sig.sizes(&inputs)?, stdin, output: Some(fuller::futhark_sig::write_binary(&[&output])) });
        }
        let i = (0..cases.len()).filter(|i| cases[*i].split == Split::Train).max_by_key(|i| cases[*i].output.data.len() * cases[*i].inputs.iter().map(|a| a.data.len()).sum::<usize>()).ok_or("no training case")?;
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
    /// The quality inputs (`futhark_sig::quality_inputs`, seven a case), in case order.
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
        !self.outputs.is_empty() && self.outputs.iter().all(|a| a.data.iter().all(|x| *x == 0))
    }
    /// A legal gene's program that did not compile or run (our software has a bug, logged): wrong everywhere.
    fn worst(cases: &[Case]) -> Quality {
        let fit: Vec<&Case> = cases.iter().filter(|c| c.split.in_fitness()).collect();
        let inputs = fit.iter().flat_map(|c| vec![1.0; quality_input_names(&c.output).len()]).collect();
        Quality { raw: vec![[1.0; 3]; fit.len()], inputs, exact_train: 0, exact_held: 0, exact_test: 0, work: 0, outputs: Vec::new(), behaviour: "failed".into() }
    }
    /// The quality of a program's outputs on the cases: the fitness's inputs and exactness from the training and
    /// held-out cases alone; the test cases only counted (`exact_test`).
    fn of(outs: Vec<Array>, cases: &[Case], work: u128) -> Quality {
        let mut q = Quality { raw: Vec::with_capacity(cases.len()), inputs: Vec::new(), exact_train: 0, exact_held: 0, exact_test: 0, work, outputs: Vec::new(), behaviour: String::new() };
        for (got, c) in outs.iter().zip(cases) {
            let r = compare(got, &c.output);
            if c.split.in_fitness() {
                q.inputs.extend(quality_inputs(got, &c.output));
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
    sizes: Vec<BTreeMap<String, usize>>,
    /// The input the speed is timed on: the problem's own large timing input (never a quality case) or, without one,
    /// the largest training case.
    timing: TimingInput,
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
}

impl<'a> Judge<'a> {
    fn compile(&self, source: &str, dir: &Path) -> Result<PathBuf, String> {
        self.compile_staged(source, dir).map(|(e, _)| e)
    }

    /// [`Judge::compile`] with each stage's wall time (the c backend's compile is all `futhark`).
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
                if m > 0 && self.interp.work(tree, &self.timing.sizes)? <= CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT {
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
        println!("# {} batch: {n} programs, {} compile(s) ({} failed), {} process(es), {} halving(s), {:.1} s", utc(), c.compiles, c.failed_compiles, c.processes, c.splits.len(), t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// BATCH MODE: the A/A sets in one process: the published program against itself (a discarded pair, then 11),
    /// and, for P, the published program as a member (wrapped as a candidate's text is) against it through the
    /// slots' scheme (`inrun_pairs`, the first process-equivalent discarded); with `rounds`, that many single timings
    /// of the published program (the gate's). Returns (noise, in-run A/A, rounds in ns).
    fn aa_batch(&self, yardstick: &str, inrun_pairs: usize, rounds: usize) -> Result<AaSets, String> {
        let host = self.host.as_ref().ok_or("our software has a bug: batch mode with no batch host")?;
        let y = Step::Time { yardstick: true, runs: self.runs };
        let mut steps = vec![y; 2 + 2 * 11];
        steps.extend(vec![y; rounds]);
        let mut members = vec![Member { program: Program::Text(yardstick), steps }];
        if inrun_pairs > 0 {
            members.push(Member { program: Program::Text(yardstick), steps: pair_steps(inrun_pairs, true, self.runs) });
        }
        let r = host.run(&members)?;
        let o = &r.outcomes[0];
        let mut noise = Vec::with_capacity(11);
        for i in 0..11 {
            noise.push(o[2 + 2 * i].fastest_ns()? / o[3 + 2 * i].fastest_ns()?);
        }
        let rounds = o[24..].iter().map(Outcome::fastest_ns).collect::<Result<Vec<_>, _>>()?;
        let inrun = if inrun_pairs > 0 { pair_ratios(&r.outcomes[1], inrun_pairs, true)? } else { Vec::new() };
        Ok(AaSets { noise, inrun, rounds })
    }

    /// BATCH MODE: the champion's re-time (`pairs`, the first process-equivalent discarded) and its output on the
    /// timing input, in one process.
    fn champion_batch(&self, tree: &Tree, pairs: usize) -> Result<(Vec<f64>, Array), String> {
        let host = self.host.as_ref().ok_or("our software has a bug: batch mode with no batch host")?;
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

    /// One process's time on the timing case: the fastest of `runs` runs' GPU-timed passes, in nanoseconds.
    fn time_once(&self, exe: &Path) -> Result<f64, String> {
        let (_, report) = run_program_bytes(exe, &self.timing.stdin, true, self.runs)?;
        if report.dispatches == 0 && report.passes.is_empty() && report.backend == "Metal" {
            // No GPU work at all (a constant output: measured 2026-10-27, 18 of reduce's first 48 programs): the GPU
            // time is not a measure of it, so it earns no speed credit (ratio infinite, speed input 1), never a bug.
            return Ok(f64::INFINITY);
        }
        report.fastest_run_ns(self.runs)
    }

    /// The pairs of a FRESHLY COMPILED executable: one process of it run first and discarded, then the pairs. Measured
    /// 2026-10-27 on reduce's timing input: the first process of a new executable took 3.16x the yardstick's time
    /// (the in-run A/A's first pair), the next 10 pairs 0.994 to 1.010.
    fn time_fresh(&self, exe: &Path, pairs: usize) -> Result<Vec<f64>, String> {
        self.time_once(exe)?;
        self.time_pairs(exe, pairs)
    }

    /// Interleaved pairs, candidate against the yardstick, the order alternating: the ratios candidate / yardstick.
    fn time_pairs(&self, exe: &Path, pairs: usize) -> Result<Vec<f64>, String> {
        let yard = self.yard_exe.as_ref().ok_or("our software has a bug: timing with no yardstick (the c backend has no GPU timing)")?;
        let mut out = Vec::with_capacity(pairs);
        for i in 0..pairs {
            let (c, y) = if i % 2 == 0 {
                let c = self.time_once(exe)?;
                (c, self.time_once(yard)?)
            } else {
                let y = self.time_once(yard)?;
                (self.time_once(exe)?, y)
            };
            out.push(c / y);
        }
        Ok(out)
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
            Ok(o) => Ok(Quality::of(o, &self.cases, work)),
            Err(e) => {
                self.log_bug(&e, &batch);
                Ok(Quality::worst(&self.cases))
            }
        }
    }

    /// Score one individual in a stage.
    fn evaluate(&mut self, tree: &Tree, source: &str, stage: FitnessStage) -> Result<Eval, String> {
        self.scored += 1;
        self.last = ("hit", 0, Stages::default());
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
            if missing > 0 && timing_work > CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT {
                // Over the per-submit cap at the timing size: never dispatched (the work guard), no speed credit.
                self.over_cap_untimed += 1;
                self.slots.add(source, &vec![f64::INFINITY; missing])?;
            } else if missing > 0 {
                // The timed program is the plain one (the declared signature, one case); its empty slots are filled.
                let got = match self.mode {
                    EvalMode::Single => {
                        let t = Instant::now();
                        let r = self.compile_staged(source, &self.work_dir.join("timed")).and_then(|(e, cs)| if missing == self.slots.slots() { self.time_fresh(&e, missing).map(|r| (r, cs)) } else { self.time_pairs(&e, missing).map(|r| (r, cs)) });
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
        let t = staged_hff(stage, &q.inputs(), &speed)?;
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
        let e = express(self.contract, self.layout, gene, self.const_len).map_err(|e| format!("a gene does not express: {e}"))?;
        let mut q: BTreeMap<Ty, usize> = BTreeMap::new();
        let mut out = Vec::with_capacity(e.nodes.len() * 2);
        for n in &e.nodes {
            out.push(n.pos);
            if n.const_index.is_some() {
                let qq = q.entry(n.ty).or_default();
                let frame = self.layout.frame(n.ty).ok_or("a node of a type with no frame")?;
                out.push(frame.dc_off.ok_or("a constant with no Dc")? + *qq);
                *qq += 1;
            }
        }
        Ok(out)
    }

    /// LOW-CHANGE mutation: `m` expressed cells, each set to another member of its own alphabet.
    fn mutate(&self, gene: &mut [u32], m: usize, rng: &mut GeneRng) -> Result<(), String> {
        for _ in 0..m {
            let cells = self.coding_cells(gene)?;
            for _ in 0..16 {
                let pos = cells[rng.below(cells.len())];
                let a = self.layout.alphabet(self.contract, pos, self.const_len);
                if a.len() < 2 {
                    continue;
                }
                let v = a[rng.below(a.len())];
                if v != gene[pos] {
                    self.layout.mutate(self.contract, gene, pos, v, self.const_len).map_err(|e| format!("mutation: {e}"))?;
                    break;
                }
            }
        }
        Ok(())
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
        let cells = self.coding_cells(a)?;
        let pos = cells[rng.below(cells.len())];
        let span = self.layout.spans().into_iter().find(|s| s.off <= pos && pos < s.off + s.len).ok_or("a cell in no span")?;
        if span.domain == Domain::Dc && span.len < 2 {
            return Ok(());
        }
        let lo = span.off + rng.below(pos - span.off + 1);
        let hi = (pos + 1 + rng.below(span.off + span.len - pos)).min(span.off + span.len);
        self.layout.swap_range(self.layout, a, b, lo, hi).map_err(|e| format!("crossover: {e}"))
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

/// THE VERDICT of an exact champion's re-time against the published program (one function for the run's report and
/// `--rescore`, plan section G, fault 12): "beat" needs the median below the least A/A ratio AND at least 9 of 11
/// pairs faster (a sign test, one-sided p = 0.033; the study plan's rule of 2026-10-28 13:10Z); "exact but slower"
/// the median above the greatest A/A ratio; otherwise "matched".
fn speed_verdict(ratios: &[f64], aa: &[f64]) -> &'static str {
    let floor = aa.iter().cloned().fold(f64::INFINITY, f64::min);
    let ceiling = aa.iter().cloned().fold(0.0, f64::max);
    let m = median(ratios);
    let faster = ratios.iter().filter(|r| **r < 1.0).count();
    if m < floor && faster * 11 >= 9 * ratios.len() {
        "beat (exact and faster: the median below the least A/A ratio and at least 9 of 11 pairs faster)"
    } else if m > ceiling {
        "exact but slower (the median above the greatest A/A ratio)"
    } else {
        "matched (exact, within the A/A noise)"
    }
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
fn fable_mutants_population(run: &mut Run, sig: &Signature, dir: &Path, pop: usize, rng: &mut GeneRng) -> Result<Vec<Individual>, String> {
    if run.gen.canvas.is_none() {
        return Err("--gen0 fable-mutants writes the seed onto a ring canvas: it needs --start llm-ring".into());
    }
    let e = load_pool(&dir.join("seed.tsv"))?.into_iter().next().ok_or("seed.tsv holds no program")?;
    let seed = build_at(run.gen.k, &e.sexp, &e.dual)?;
    fuller::typed_gep::fits(run.gen.contract, run.gen.layout, &seed).map_err(|x| format!("the seed program does not fit the layout: {x}"))?;
    let g0 = write_back(run.gen.contract, run.gen.layout, &run.gen.random_gene(rng)?, &seed).map_err(|x| format!("the seed's write-back: {x}"))?;
    let spec = fuller::futhark_rings::MutantSpec { contract: run.gen.contract, layout: run.gen.layout, const_len: run.gen.const_len, pop, max_tries: 200 * pop };
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
    std::fs::write(run.out.join("population.tsv"), &lines).map_err(|x| x.to_string())?;
    let msg = format!("generation 0 from the seed program and its {operator} mutants: {} kept of {} tried (no two members with one text or one tree)", m.genes.len() - 1, m.tried);
    println!("# {} {msg}\n# seed: {}", utc(), e.sexp);
    run.tel.note(0, msg)?;
    Ok(out)
}

const MEASURE_TAG: u64 = 0x4D45_4153_5552_4521;

/// THE OPERATOR MEASUREMENT (`--measure KIND`, plan sections H and I), on the run's evaluation path (the GPU): `count`
/// children, each ONE application of the operator to a parent, the parents Sonnet's task alternatives in order
/// (`ring_canvas`), none within `--distinct-edits` of the reference or of another parent, each written onto a fresh gene (a ring canvas with `--start llm-ring`, else random padding).
/// KIND: `grow` (`fuller::futhark_rings::grow`: a coding head cell given more arguments, so cells the padding holds
/// are read), `point` (today's point mutation, `Genetics::mutate`, one cell), `walk` (`Rings::walk`, ring walk, k at
/// most 4; it needs the rings). Measured, the children's: the share whose output is all zeros; the share with an
/// output not seen before in the measurement (parents' included); the share better than their parent (stage-1
/// fitness); the share equal to their parent's output; distinct outputs. Writes `OUT/measure.tsv` (one line a child)
/// and `OUT/measure_summary.tsv`; every child is a mutation event (`ring_walk` for the walk).
fn measure_command(run: &mut Run, sig: &Signature, kind: &str, parents: &[Tree], count: usize, chunk: usize, seed: u64) -> Result<(), String> {
    if parents.is_empty() {
        return Err("--measure: the pool has no model-written program that fits the layout".into());
    }
    if !["grow", "point", "walk"].contains(&kind) {
        return Err(format!("--measure {kind}: grow, point or walk"));
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
    for (ci, idx) in (0..progs.len()).collect::<Vec<_>>().chunks(chunk.max(1)).enumerate() {
        let items: Vec<(&Tree, &str)> = idx.iter().map(|i| (progs[*i], sources[*i].as_str())).collect();
        run.judge.prefetch(&items, FitnessStage::Quality)?;
        for &i in idx {
            let e = run.judge.evaluate(progs[i], &sources[i], FitnessStage::Quality)?;
            scored.push((e.fitness, e.quality));
        }
        if ci % 10 == 0 {
            println!("# {} measure: {} of {} programs scored, {:.0} s", utc(), scored.len(), progs.len(), t0.elapsed().as_secs_f64());
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
                    let q = if outs.is_empty() { Quality::worst(&run.judge.cases) } else { Quality::of(outs.clone(), &run.judge.cases, run.judge.work_of(tree)?) };
                    (staged_hff(FitnessStage::Quality, &q.inputs, &[])?.fitness, q)
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
        quality_fingerprint(),
        run.judge.cases.iter().map(|k| quality_input_names(&k.output).len()).sum::<usize>(),
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
fn gate_passed(dir: &Path, problem: &str, program_size: usize, fp_cases: &str) -> Result<(), String> {
    let path = dir.join("check_summary.tsv");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("the gate {}: {e}", path.display()))?;
    let get = |k: &str| text.lines().find_map(|l| l.strip_prefix(&format!("{k}\t"))).unwrap_or("").to_string();
    let want = [("problem", problem.to_string()), ("program_size", program_size.to_string()), ("cases_fingerprint", fp_cases.to_string()), ("quality_fingerprint", quality_fingerprint()), ("verdict", "pass".to_string())];
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
    let minutes: f64 = num(&args, "--minutes", 60.0)?;
    let budget: u64 = num(&args, "--evaluations", 6000)?;
    let pop: usize = num(&args, "--pop", 24)?;
    let seed: u64 = num(&args, "--seed", 1)?;
    let pairs: usize = num(&args, "--pairs", 11)?;
    if pairs == 0 {
        return Err("--pairs 0: stage 2 needs at least one timing slot".into());
    }
    let runs: usize = num(&args, "--runs", 20)?;
    let elites: usize = num(&args, "--elites", 2)?;
    let pump_every: u32 = num(&args, "--pump-every", 5)?;
    let fold_every: u32 = num(&args, "--fold-every", 10)?;
    let marriage: f64 = num(&args, "--marriage", 0.3)?;
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
    // THE RELATIVE SCORE of one program (`--rescore SEXP`, regime P): its quality inputs on every stored case beside
    // the published program's (all 0), and, when exact, its 11-pair re-time against the published program next to
    // both A/A sets, with the verdict (beat / matched / exact but slower / not reached). Writes OUT/rescore.tsv.
    let check_random: usize = num(&args, "--check-random", 1400)?;
    let check_replay = arg(&args, "--check-replay").map(PathBuf::from);
    let rescore = arg(&args, "--rescore");
    if rescore.is_some() && regime != "P" {
        return Err("--rescore needs --regime P (its in-run A/A set is the noise floor beside the re-time)".into());
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

    let p = problem(&pname)?;
    let inst = p.instance;
    let k = load(&inst)?;
    let sig = signature(&inst)?;
    let interp = Interp::new(&k, &sig)?;
    let contract = k.relations().check().map_err(|d| format!("the kingdom check: {:?}", &d[..d.len().min(3)]))?;
    let reference = build(&k, p.reference)?;
    let planted = build(&k, p.planted)?;
    let n_funcs = reference.level_order().iter().filter(|(t, _)| !k.rows[t.symbol as usize].inputs.is_empty()).count();
    let program_size: usize = num(&args, "--program-size", n_funcs)?;
    let start_tree = if start == "planted" { planted.clone() } else { reference.clone() };
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], program_size).map_err(|e| e.to_string())?;
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
    if regime == "R" || regime == "C" {
        let g = arg(&args, "--gate").ok_or("--gate DIR: a discovery run (R, C) starts only after the slope check passes (--check --out DIR)")?;
        gate_passed(Path::new(&g), &pname, program_size, &cases_fingerprint(&cases))?;
        println!("# {} the gate: the slope check in {g} passed for these cases and quality inputs", utc());
    }
    let all_stdin: Vec<u8> = cases.iter().flat_map(|c| c.stdin.clone()).collect();
    let timing = TimingInput::of(&p, &sig, &cases, &out.join("timing_build"))?;
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
    // Control: the reference gene is exact on every case (interpreted), so the kingdom expresses the published program.
    for c in &cases {
        if interp.eval(&reference, &c.inputs)? != c.output {
            return Err(format!("our software has a bug: the reference gene is not exact on {}", c.name));
        }
    }
    println!("# {} control: the reference gene is exact on all {} cases (host interpreter)", utc(), cases.len());

    let mut objectives: Vec<String> = cases.iter().filter(|c| c.split.in_fitness()).flat_map(|c| quality_input_names(&c.output).into_iter().map(|m| format!("quality {} {m}", c.name)).collect::<Vec<_>>()).collect();
    for i in 0..pairs {
        objectives.push(format!("speed pair {i} r/(1+r) (stage 2)"));
    }
    let card = serde_json::json!({"problem": pname, "regime": regime, "start": start, "backend": format!("{backend:?}"), "eval": format!("{mode:?}"), "evaluations": budget, "minutes": minutes, "pop": pop, "seed": seed, "stop_at_exact": stop_at_exact, "pairs": pairs, "runs": runs, "elites": elites, "pump_every": pump_every, "fold_every": fold_every, "marriage": marriage, "program_size": program_size, "gene_words": layout.width, "fitness": "staged: TrueNorth(quality) until exact on every case, then TrueNorth(theta_q/(pi/2), theta_s/(pi/2))"});
    let tel = RunTelemetry::start(
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
    )?;
    if mode == EvalMode::Rust && backend != Backend::Gpu {
        return Err("--eval rust runs on the GPU: it takes --backend gpu".into());
    }
    // The shim is built once, here (under the cargo lock); no candidate's link runs cargo or takes the lock. The rust
    // path has no shim and no C: its runtime is built once instead, into the run's own directory.
    let with_shim = backend == Backend::Gpu && mode != EvalMode::Rust;
    let shim = if with_shim { prebuild_shim(&out.join("shim"))? } else { out.join("shim") };
    let yard_exe = if with_shim { Some(compile_program_prebuilt(p.yardstick, &out.join("yardstick_build"), &shim)?) } else { None };
    let host = match mode {
        EvalMode::Rust => {
            let rt = fuller::futhark_rust::prebuild_rt(&out.join("rust_rt"))?;
            let b = fuller::futhark_batch::Backend::Rust { rt };
            Some(Host::new(&k, &sig, cases.len(), b, &out.join("rust_build"), Inputs { cases_stdin: &all_stdin, timing_stdin: Some(timing.stdin.as_slice()), yardstick: Some(p.yardstick) })?)
        }
        EvalMode::Batch => {
            let b = if backend == Backend::Gpu { fuller::futhark_batch::Backend::Gpu { shim: shim.clone() } } else { fuller::futhark_batch::Backend::C };
            let timing_stdin = if backend == Backend::Gpu { Some(timing.stdin.as_slice()) } else { None };
            Some(Host::new(&k, &sig, cases.len(), b, &out.join("batch_build"), Inputs { cases_stdin: &all_stdin, timing_stdin, yardstick: Some(p.yardstick) })?)
        }
        EvalMode::Single => None,
    };
    let mut run = Run {
        inst,
        out: out.clone(),
        gen: Genetics { k: &k, contract: &contract, layout: &layout, const_len: &const_len, canvas: canvas.as_ref(), ring_walk },
        judge: Judge { k: &k, sig: &sig, all_stdin, interp, cases, sizes, timing, over_cap_untimed: 0, backend, runs, yard_exe, shim: shim.clone(), work_dir: out.join("candidate_build"), quality_cache: HashMap::new(), slots: TimingSlots::new(pairs), evaluations: 0, scored: 0, compiled: 0, hits: 0, bugs: 0, setmem: 0, ice: 0, bug_log: out.join("bugs.log"), max_work: 0, mode, host, batched_q: HashMap::new(), batched_t: HashMap::new(), gen_stages: Stages::default(), batch_splits: 0, last: ("hit", 0, Stages::default()) },
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
    if backend == Backend::Gpu && !check {
        match mode {
            EvalMode::Single => {
                let y = run.judge.yard_exe.clone().ok_or("no yardstick")?;
                run.judge.time_pairs(&y, 1)?;
                noise = run.judge.time_pairs(&y, 11)?;
            }
            EvalMode::Batch | EvalMode::Rust => {
                let a = run.judge.aa_batch(p.yardstick, if regime == "P" { pairs } else { 0 }, if aa_only { 3 } else { 0 })?;
                noise = a.noise.clone();
                batch_aa = Some(a);
            }
        }
        let msg = format!("A/A noise floor{}, the published program against itself, 11 pairs of {runs} runs on {}: ratio median {:.4}, min {:.4}, max {:.4}", if mode.batched() { " (in one batch process)" } else { "" }, run.judge.timing.label(), median(&noise), noise.iter().cloned().fold(f64::INFINITY, f64::min), noise.iter().cloned().fold(0.0, f64::max));
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
    let ref_source = program(&k, &sig, &reference)?;
    if check_replay.is_some() && measure.is_none() {
        // A REPLAYED check runs nothing on the GPU: the stored outputs are the GPU's, the reference's own among them,
        // and the check stops with our bug if the reference does not score 0 on them.
        slope_check_command(&mut run, &sig, &reference, pop, check_random, seed, check_replay.as_deref(), program_size)?;
        run.tel.end("early_stop", 0, None)?;
        return Ok(());
    }
    run.judge.prefetch(&[(&reference, ref_source.as_str())], FitnessStage::Quality)?;
    let ref_q = run.judge.quality(&reference, &ref_source)?;
    if !ref_q.perfect() {
        return Err(format!("our software has a bug: the compiled reference is not exact ({:?})", ref_q.raw));
    }
    println!("# {} control: the compiled reference gene is exact on every case ({backend:?}); work {} node executions at most", utc(), ref_q.work);
    std::fs::write(out.join("reference.fut"), &ref_source).map_err(|e| e.to_string())?;
    if let Some(kind) = &measure {
        let count: usize = num(&args, "--count", 1400)?;
        // The parents: the task alternatives with no near copies (the bar of generation 0, by distance alone).
        let bar: usize = num(&args, "--distinct-edits", 5)?;
        let mut parents: Vec<Tree> = Vec::new();
        for t in &pool_programs {
            if tree_distance(t, &reference) >= bar && parents.iter().all(|o| tree_distance(t, o) >= bar) {
                parents.push(t.clone());
            }
        }
        println!("# {} measure: {} parents of {} task alternatives (tree edit distance at least {bar} to the reference and to each other)", utc(), parents.len(), pool_programs.len());
        measure_command(&mut run, &sig, kind, &parents, count, pop, seed)?;
        run.tel.end("early_stop", 0, None)?;
        return Ok(());
    }
    if check {
        slope_check_command(&mut run, &sig, &reference, pop, check_random, seed, check_replay.as_deref(), program_size)?;
        run.tel.end("early_stop", 0, None)?;
        return Ok(());
    }
    // The in-run A/A (P): the published program, compiled as a candidate, through the timing slots.
    let mut inrun_aa: Vec<f64> = Vec::new();
    if regime == "P" && backend == Backend::Gpu {
        let r = match &batch_aa {
            Some(a) => a.inrun.clone(),
            None => {
                let exe = run.judge.compile(p.yardstick, &out.join("candidate_build").join("timed"))?;
                run.judge.time_fresh(&exe, pairs)?
            }
        };
        run.judge.slots.add(p.yardstick, &r)?;
        inrun_aa = run.judge.slots.full(p.yardstick)?.to_vec();
        let msg = format!("in-run A/A{}, the published program as a candidate through the timing slots, {pairs} pairs: ratio median {:.4}, min {:.4}, max {:.4}", if mode.batched() { " (in the batch process)" } else { "" }, median(&inrun_aa), inrun_aa.iter().cloned().fold(f64::INFINITY, f64::min), inrun_aa.iter().cloned().fold(0.0, f64::max));
        println!("# {} {msg}", utc());
        run.tel.note(0, msg)?;
    }
    if let Some(text) = &rescore {
        let tree = build(&k, text)?;
        let source = program(&k, &sig, &tree)?;
        run.judge.prefetch(&[(&tree, source.as_str())], FitnessStage::Quality)?;
        let q = run.judge.quality(&tree, &source)?;
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
        let mut rep = format!("problem\t{pname}\nprogram\t{}\nexact_on_every_case\t{}\nexact_train\t{}\nexact_held\t{}\n", sexp(&k, &tree), if q.perfect() { "yes" } else { "no" }, q.exact_train, q.exact_held);
        // Each case's inputs by name (a report: the fitness takes every input on its own).
        let mut at = 0;
        for c in run.judge.cases.iter().filter(|c| c.split.in_fitness()) {
            let names = quality_input_names(&c.output);
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
            let exe = compile_program_prebuilt(&source, &out.join("rescore_build"), &shim)?;
            let ratios = run.judge.time_fresh(&exe, 11)?;
            let m = median(&ratios);
            rep.push_str(&format!("champion_vs_published_ratios\t{}\nchampion_vs_published_median\t{m:.4}\npairs_faster\t{} of {}\n", fmt(&ratios), ratios.iter().filter(|r| **r < 1.0).count(), ratios.len()));
            if let Some(want) = &run.judge.timing.output {
                let (got, _) = run_program_bytes(&exe, &run.judge.timing.stdin, false, 1)?;
                rep.push_str(&format!("champion_exact_on_timing_input\t{}\n", if read_binary(&got)? == read_binary(want)? { "yes" } else { "NO" }));
            }
            speed_verdict(&ratios, &all)
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
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
        let all: Vec<f64> = noise.iter().chain(inrun_aa.iter()).cloned().collect();
        let within = all.iter().filter(|r| (0.95..=1.05).contains(*r)).count();
        let (one, unit) = match &batch_aa {
            Some(a) => (a.rounds.clone(), "rounds_in_one_batch_process"),
            None => {
                let y = run.judge.yard_exe.clone().ok_or("no yardstick")?;
                ((0..3).map(|_| run.judge.time_once(&y)).collect::<Result<_, _>>()?, "processes")
            }
        };
        let rep = format!("problem\t{pname}\ntiming_input\t{}\npublished_gpu_us_fastest_run_3_{unit}\t{}\nnoise_floor_ratios_published_vs_itself\t{}\ninrun_aa_ratios\t{}\nwithin_5_percent\t{within} of {}\ngate\t{}\n", run.judge.timing.label(), one.iter().map(|ns| format!("{:.1}", ns / 1e3)).collect::<Vec<_>>().join(","), fmt(&noise), fmt(&inrun_aa), all.len(), if within == all.len() && !all.is_empty() { "pass" } else { "fail" });
        std::fs::write(out.join("aa_gate.tsv"), &rep).map_err(|e| e.to_string())?;
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
            let vs = variants(&run.inst, &k, &bridge, &start_tree, 3, 32, pop * 2)?;
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
            let fe = FoldEgraph { inst: &run.inst, kingdom: &k, bridge: &bridge };
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
        } else if arg(&args, "--gen0").as_deref() == Some("fable-mutants") {
            // Generation 0 from ONE seed program (Fable's, cached in the pool) and its mutants by the run's operator.
            let dir = PathBuf::from(arg(&args, "--pool").ok_or("--gen0 fable-mutants needs --pool DIR (seed.tsv: examples/futhark_pool.rs seed)")?);
            cands = fable_mutants_population(&mut run, &sig, &dir, pop, &mut rng0)?;
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
        if per_gen.is_empty() {
            per_gen = String::from("generation\tevaluations\tbest_so_far_hff\tmedian_hff\tbest_this_generation_hff\tdistinct_outputs\tall_zero_share\timprovement_in_best_so_far\n");
        }
        per_gen.push_str(&format!("{generation}\t{}\t{so_far:.6}\t{:.6}\t{:.6}\t{distinct_outputs}\t{zero_share:.4}\t{:.6}\n", run.judge.evaluations, median(&fits), be.fitness, prev_so_far.map_or(0.0, |p| p - so_far)));
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
            for v in variants(&run.inst, &k, &bridge, &b.tree, 2, 16, 4)? {
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
            let fe = FoldEgraph { inst: &run.inst, kingdom: &k, bridge: &bridge };
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
            if rng.below(2) == 0 {
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
        ref_q.work,
        sexp(&k, &best.tree),
        run.judge.max_work,
        run.judge.scored,
        be.quality.exact_test,
        run.first_exact_test.map_or("-".to_string(), |t| format!("{t} of {n_test}")),
        run.judge.ice,
        binary_identity(),
    );
    if backend == Backend::Gpu {
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
        rep.push_str(&format!("noise_floor_ratios_published_vs_itself\t{}\nnoise_floor_median\t{:.4}\n", fmt(&noise), median(&noise)));
        if !inrun_aa.is_empty() {
            rep.push_str(&format!("inrun_aa_ratios\t{}\ninrun_aa_median\t{:.4}\n", fmt(&inrun_aa), median(&inrun_aa)));
        }
        rep.push_str(&format!("timing_input\t{}\nover_cap_untimed\t{}\n", run.judge.timing.label(), run.judge.over_cap_untimed));
        rep.push_str(&format!("best_speed_inputs\t{}\nbest_speed_ratio_median\t{:.4}\n", fmt(&be.speed), be.ratio));
        if be.quality.perfect() {
            let (ratios, got) = match mode {
                EvalMode::Single => {
                    let exe = compile_program_prebuilt(&best.source, &out.join("best_build"), &shim)?;
                    let ratios = run.judge.time_fresh(&exe, 11)?;
                    let (got, _) = run_program_bytes(&exe, &run.judge.timing.stdin, false, 1)?;
                    (ratios, fuller::futhark_sig::read_one(&got)?)
                }
                EvalMode::Batch | EvalMode::Rust => run.judge.champion_batch(&best.tree, 11)?,
            };
            if let Some(want) = &run.judge.timing.output {
                rep.push_str(&format!("champion_exact_on_timing_input\t{}\n", if got == fuller::futhark_sig::read_one(want)? { "yes" } else { "NO" }));
            }
            rep.push_str(&format!("best_vs_published_pairs\t11\nbest_vs_published_ratios\t{}\nbest_vs_published_median\t{:.4}\nfaster_rounds\t{}\n", fmt(&ratios), median(&ratios), ratios.iter().filter(|r| **r < 1.0).count()));
            // A speed claim counts only below the noise floor (the least ratio of both A/A sets) AND with at least 9
            // of 11 pairs faster (a sign test, one-sided p = 0.033): the rule of the study plan, 2026-10-28 13:10Z,
            // after a quiet A/A set beside a noisy re-time made the published program itself "faster".
            let all: Vec<f64> = noise.iter().chain(inrun_aa.iter()).cloned().collect();
            let floor = all.iter().cloned().fold(f64::INFINITY, f64::min);
            let v = speed_verdict(&ratios, &all);
            rep.push_str(&format!("noise_floor_least_ratio\t{floor:.4}\nspeed_claim_beats_noise_floor\t{}\nverdict\t{v}\n", if v.starts_with("beat") { "yes" } else { "no" }));
        }
    }
    // The best-ever program (the hall of fame's rank 1). For C, which has no selection, the best above is only the
    // best of the last random batch, and the best-ever is the run's champion (measured 2026-10-28: pathfinder C seed
    // 2's last batch was at 1.207, its best-ever at 1.076).
    if let Some(h) = run.hof.entries().first() {
        rep.push_str(&format!("best_ever_hff\t{:.6}\nbest_ever_born\t{}\nbest_ever_program\t{}\n", h.fitness, h.born, h.text.lines().next().unwrap_or("")));
    }
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

fn main() {
    if let Err(e) = real_main() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
