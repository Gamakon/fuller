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
use fuller::futhark_sig::{build, compare, compile_c, load, load_cases, problem, program, program_cases, quality_inputs, random_gene, read_binary, sexp, QUALITY_INPUTS, signature, variants, Array, Bridge, Case, FoldEgraph, Instance, Interp, Signature, Split};
use fuller::futhark_wgpu_host::{compile_program_prebuilt, compile_program_prebuilt_staged, prebuild_shim, run_program_bytes, run_program_bytes_staged, CompileStages, RunStages};
use fuller::geneframe::Ty;
use fuller::kingdom_contract::KingdomContract;
use fuller::population_fold::AdfTable;
use fuller::runner::{save_checkpoint, staged_hff, FameEntry, FitnessStage, HallOfFame, Ledger, LedgerLine, TimingSlots};
use fuller::saturated_fold::{fold_saturated, Budget};
use fuller::telemetry::{EventKind, GenerationStats, Mutation, RunInfo, RunTelemetry};
use fuller::typed_gep::{express, write_back, Domain, GeneLayout, Tree};
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
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
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
/// order and the speed inputs (stage 2), so two runs' inputs can be compared bit for bit.
const LEDGER_STAGE_COLUMNS: [&str; 15] = ["eval", "batch_size", "render_ms", "futhark_ms", "port_ms", "cc_ms", "check_ms", "link_ms", "start_ms", "device_ms", "pipelines_ms", "runs_ms", "timing_ms", "quality_inputs", "speed_inputs"];

/// What a program measured on the stored cases (independent of the stage).
#[derive(Debug, Clone)]
struct Quality {
    /// Three raw measurements a case (differing share, mean, max), in case order: exactness.
    raw: Vec<[f64; 3]>,
    /// The quality inputs (`futhark_sig::quality_inputs`, seven a case), in case order.
    inputs: Vec<f64>,
    exact_train: usize,
    exact_held: usize,
    /// The most node executions on any case.
    work: u128,
}

impl Quality {
    fn perfect(&self) -> bool {
        self.raw.iter().all(|r| *r == [0.0; 3])
    }
    fn inputs(&self) -> Vec<f64> {
        self.inputs.clone()
    }
    /// A legal gene's program that did not compile or run (our software has a bug, logged): wrong everywhere.
    fn worst(cases: usize) -> Quality {
        Quality { raw: vec![[1.0; 3]; cases], inputs: vec![1.0; cases * QUALITY_INPUTS.len()], exact_train: 0, exact_held: 0, work: 0 }
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
    /// Fitness evaluations (the budget's unit): every individual scored, cache hits included.
    evaluations: u64,
    /// Programs compiled and run (cache misses).
    compiled: u64,
    hits: u64,
    bugs: u64,
    /// Of the bugs, the branch compiler's "Cannot handle SetMem statement" (a known compiler limitation).
    setmem: u64,
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
        let outs = match got {
            Ok(o) => o,
            Err(e) => {
                self.log_bug(&e, &batch);
                return Ok(Quality::worst(self.cases.len()));
            }
        };
        let mut q = Quality { raw: Vec::with_capacity(self.cases.len()), inputs: Vec::new(), exact_train: 0, exact_held: 0, work };
        for (i, got) in outs.iter().enumerate() {
            let r = compare(got, &self.cases[i].output);
            q.inputs.extend(quality_inputs(got, &self.cases[i].output));
            if r == [0.0; 3] {
                match self.cases[i].split {
                    Split::Train => q.exact_train += 1,
                    Split::Held => q.exact_held += 1,
                }
            }
            q.raw.push(r);
        }
        Ok(q)
    }

    /// Score one individual in a stage.
    fn evaluate(&mut self, tree: &Tree, source: &str, stage: FitnessStage) -> Result<Eval, String> {
        self.evaluations += 1;
        self.last = ("hit", 0, Stages::default());
        let q = if let Some(q) = self.quality_cache.get(source) {
            self.hits += 1;
            q.clone()
        } else {
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
}

impl Genetics<'_> {
    fn random_gene(&self, rng: &mut GeneRng) -> Result<Vec<u32>, String> {
        random_gene(self.contract, self.layout, self.const_len, rng)
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

    /// Homologous crossover: a range of the domain that holds one of `a`'s expressed cells, exchanged with `b`.
    fn crossover(&self, a: &mut [u32], b: &[u32], rng: &mut GeneRng) -> Result<(), String> {
        let cells = self.coding_cells(a)?;
        let pos = cells[rng.below(cells.len())];
        let span = self.layout.spans().into_iter().find(|s| s.off <= pos && pos < s.off + s.len).ok_or("a cell in no span")?;
        if span.domain == Domain::Dc && span.len < 2 {
            return Ok(());
        }
        let lo = span.off + rng.below(pos - span.off + 1);
        let hi = (pos + 1 + rng.below(span.off + span.len - pos)).min(span.off + span.len);
        let mut other = b.to_vec();
        self.layout.swap_range(self.layout, a, &mut other, lo, hi).map_err(|e| format!("crossover: {e}"))
    }

    /// MARRIAGE, typed subtree crossover: the domain is every pair (node of A, node of B) of one output type, not the
    /// two roots, not identical subtrees, whose two offspring both write back into the parents' genes (enumerated
    /// before the draw); one pair is drawn uniformly. Returns the two offspring genes, or None for an empty domain.
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
        let mut fitting = Vec::new();
        for (pa, pb, sa, sb) in domain {
            let ca = replace(&a.tree, &pa, &sb);
            let cb = replace(&b.tree, &pb, &sa);
            if let (Ok(ga), Ok(gb)) = (write_back(self.contract, self.layout, &a.gene, &ca), write_back(self.contract, self.layout, &b.gene, &cb)) {
                fitting.push((ga, gb));
                if fitting.len() >= 256 {
                    break;
                }
            }
        }
        if fitting.is_empty() {
            return Ok(None);
        }
        Ok(Some(fitting.swap_remove(rng.below(fitting.len()))))
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
        Ok(())
    }

    fn checkpoint(&self, generation: u32, population: &[Individual], seconds: f64, regime: &str, rng_state: u64) -> Result<(), String> {
        let ind = |i: &Individual| serde_json::json!({"id": i.id, "parents": i.parents, "born": i.born, "operator": i.operator, "tree": tree_json(&i.tree)});
        let v = serde_json::json!({
            "problem": self.inst.name, "regime": regime, "generation": generation, "seconds": seconds, "next_id": self.next_id,
            "evaluations": self.judge.evaluations, "compiled": self.judge.compiled, "hits": self.judge.hits, "bugs": self.judge.bugs, "setmem": self.judge.setmem,
            "stage": self.stage, "first_exact": self.first_exact, "best_ever": self.tel.best_ever(), "rng": rng_state,
            "population": population.iter().map(ind).collect::<Vec<_>>(), "hof": self.hof,
        });
        save_checkpoint(&self.out, &v)
    }
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let pname = arg(&args, "--problem").unwrap_or_else(|| "pathfinder".into());
    let regime = arg(&args, "--regime").ok_or("--regime R|C|P")?;
    if !["R", "C", "P"].contains(&regime.as_str()) {
        return Err(format!("--regime {regime}: R (from random), C (random-search control) or P (from the published program)"));
    }
    // R and C start from random genes; only P has a start program (the record said "published" for R and C until
    // 2026-10-28, a default of the flag, never used by them: their generation 0 is all `random` in every ledger).
    let start = match (regime.as_str(), arg(&args, "--start")) {
        ("P", None) => "published".to_string(),
        ("P", Some(s)) if s == "published" || s == "planted" => s,
        ("P", Some(s)) => return Err(format!("--start {s}: published or planted")),
        (_, None) => "random".to_string(),
        (r, Some(s)) => return Err(format!("--start {s}: regime {r} starts from random genes; --start is for P")),
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
    let checkpoint_every: u32 = num(&args, "--checkpoint-every", 5)?;
    let resume = args.iter().any(|a| a == "--resume");
    // Discovery runs (R, C) answer H1 and H2: they stop at the first program exact on every case, never timing.
    let stop_at_exact = args.iter().any(|a| a == "--stop-at-exact");
    // The timing gate (P): measure the two A/A sets and stop, before any search.
    let aa_only = args.iter().any(|a| a == "--aa-only");
    // THE RELATIVE SCORE of one program (`--rescore SEXP`, regime P): its quality inputs on every stored case beside
    // the published program's (all 0), and, when exact, its 11-pair re-time against the published program next to
    // both A/A sets, with the verdict (beat / matched / exact but slower / not reached). Writes OUT/rescore.tsv.
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
    let cases = load_cases(&p)?;
    let sizes: Vec<BTreeMap<String, usize>> = cases.iter().map(|c| sig.sizes(&c.inputs)).collect::<Result<_, _>>()?;
    let all_stdin: Vec<u8> = cases.iter().flat_map(|c| c.stdin.clone()).collect();
    let timing = TimingInput::of(&p, &sig, &cases, &out.join("timing_build"))?;
    println!(
        "# {} futhark_evolve: problem {pname}, regime {regime}{}, budget {budget} evaluations, guard {minutes} min, population {pop}, seed {seed}, backend {backend:?}, eval {mode:?}, program size N = {program_size} ({} words a gene); {} cases ({} held out), timing on {}{}",
        utc(),
        if regime == "P" { format!(" from the {start} program") } else { String::new() },
        layout.width,
        cases.len(),
        cases.iter().filter(|c| c.split == Split::Held).count(),
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

    let mut objectives: Vec<String> = cases.iter().flat_map(|c| QUALITY_INPUTS.map(|m| format!("quality {} {m}", c.name))).collect();
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
        gen: Genetics { k: &k, contract: &contract, layout: &layout, const_len: &const_len },
        judge: Judge { k: &k, sig: &sig, all_stdin, interp, cases, sizes, timing, over_cap_untimed: 0, backend, runs, yard_exe, shim: shim.clone(), work_dir: out.join("candidate_build"), quality_cache: HashMap::new(), slots: TimingSlots::new(pairs), evaluations: 0, compiled: 0, hits: 0, bugs: 0, setmem: 0, bug_log: out.join("bugs.log"), max_work: 0, mode, host, batched_q: HashMap::new(), batched_t: HashMap::new(), gen_stages: Stages::default(), batch_splits: 0, last: ("hit", 0, Stages::default()) },
        tel,
        ledger: Ledger::open_with(&out, resume, &LEDGER_STAGE_COLUMNS)?,
        next_id: 0,
        hof: HallOfFame::new(5),
        stage: if regime == "P" { FitnessStage::QualityAndSpeed } else { FitnessStage::Quality },
        first_exact: None,
    };

    // The A/A noise floor: the yardstick against itself (one warm-up pair discarded, then 11).
    let mut noise = Vec::new();
    // Batch mode: both A/A sets (and the gate's single timings) in one batch process, as every candidate is timed.
    let mut batch_aa: Option<AaSets> = None;
    if backend == Backend::Gpu {
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
    if let (Some(want), Some(y)) = (&run.judge.timing.output, &run.judge.yard_exe) {
        // Control: the yardstick's GPU output on the timing input is the stored one (futhark c).
        let (got, _) = run_program_bytes(y, &run.judge.timing.stdin, false, 1)?;
        if read_binary(&got)? != read_binary(want)? {
            return Err("our software has a bug: the yardstick on the GPU does not give futhark c's output on the timing input".into());
        }
        println!("# {} control: the yardstick on the GPU gives futhark c's output on {} (GPU)", utc(), run.judge.timing.label());
    }
    let ref_source = program(&k, &sig, &reference)?;
    run.judge.prefetch(&[(&reference, ref_source.as_str())], FitnessStage::Quality)?;
    let ref_q = run.judge.quality(&reference, &ref_source)?;
    if !ref_q.perfect() {
        return Err(format!("our software has a bug: the compiled reference is not exact ({:?})", ref_q.raw));
    }
    println!("# {} control: the compiled reference gene is exact on every case ({backend:?}); work {} node executions at most", utc(), ref_q.work);
    std::fs::write(out.join("reference.fut"), &ref_source).map_err(|e| e.to_string())?;
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
        rep.push_str(&format!("quality_inputs\tcase\tsplit\t{}\n", QUALITY_INPUTS.join("\t")));
        let n = QUALITY_INPUTS.len();
        for (i, c) in run.judge.cases.iter().enumerate() {
            rep.push_str(&format!("quality\t{}\t{:?}\t{}\n", c.name, c.split, q.inputs[i * n..(i + 1) * n].iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join("\t")));
        }
        let cases = run.judge.cases.len() as f64;
        let means: Vec<String> = (0..n).map(|j| format!("{:.6}", (0..run.judge.cases.len()).map(|i| q.inputs[i * n + j]).sum::<f64>() / cases)).collect();
        rep.push_str(&format!("quality_mean_over_cases\t{}\npublished_quality_inputs\tall 0\n", means.join("\t")));
        let all: Vec<f64> = noise.iter().chain(inrun_aa.iter()).cloned().collect();
        let floor = all.iter().cloned().fold(f64::INFINITY, f64::min);
        let ceiling = all.iter().cloned().fold(0.0, f64::max);
        rep.push_str(&format!("timing_input\t{}\nnoise_floor_ratios_published_vs_itself\t{}\ninrun_aa_ratios\t{}\nnoise_floor_least_ratio\t{floor:.4}\nnoise_floor_greatest_ratio\t{ceiling:.4}\n", run.judge.timing.label(), fmt(&noise), fmt(&inrun_aa)));
        let verdict = if q.perfect() && backend == Backend::Gpu {
            let exe = compile_program_prebuilt(&source, &out.join("rescore_build"), &shim)?;
            let ratios = run.judge.time_fresh(&exe, 11)?;
            let m = median(&ratios);
            rep.push_str(&format!("champion_vs_published_ratios\t{}\nchampion_vs_published_median\t{m:.4}\n", fmt(&ratios)));
            if let Some(want) = &run.judge.timing.output {
                let (got, _) = run_program_bytes(&exe, &run.judge.timing.stdin, false, 1)?;
                rep.push_str(&format!("champion_exact_on_timing_input\t{}\n", if read_binary(&got)? == read_binary(want)? { "yes" } else { "NO" }));
            }
            if m < floor {
                "beat (exact and faster: the median below the least A/A ratio)"
            } else if m <= ceiling {
                "matched (exact, within the A/A noise)"
            } else {
                "exact but slower (the median above the greatest A/A ratio)"
            }
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
    let mut rng0 = GeneRng::new(seed);
    let mut generation: u32;
    let mut cands: Vec<Individual> = Vec::new();
    if let Some(c) = &checkpoint {
        generation = c["generation"].as_u64().ok_or("checkpoint generation")? as u32 + 1;
        run.next_id = c["next_id"].as_u64().ok_or("checkpoint next_id")?;
        run.judge.evaluations = c["evaluations"].as_u64().unwrap_or(0);
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
        } else {
            while cands.len() < pop {
                let g = run.gen.random_gene(&mut rng0)?;
                let ind = run.fresh(&sig, vec![], "random", g, 0)?;
                cands.push(ind);
            }
        }
    }

    let mut series = if resume { std::fs::read_to_string(out.join("series.tsv")).unwrap_or_default() } else { String::new() };
    if series.is_empty() {
        series = String::from("generation\tutc\tseconds\tevaluations\tcompiled\tcache_hits\tbugs\tstage\tbest_hff\tbest_theta_q\tbest_theta_s\tmedian_hff\tbest_ratio\tbest_exact_train\tbest_exact_held\tbest_nodes\tbest_work\tdistinct\tgen_seconds\n");
    }
    let mut stages_tsv = if resume { std::fs::read_to_string(out.join("stages.tsv")).unwrap_or_default() } else { String::new() };
    let mut population: Vec<Individual> = Vec::new();
    let mut stopped_by = "early_stop";
    let mut last_generation = generation;
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
            for ind in population.iter_mut() {
                let e = run.judge.evaluate(&ind.tree, &ind.source, run.stage)?;
                run.judge.evaluations -= 1;
                ind.eval = Some(e);
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
            "{generation}\t{}\t{:.1}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}\t{}\t{}\t{}\t{}\t{distinct}\t{gen_seconds:.2}\n",
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
        ));
        std::fs::write(out.join("series.tsv"), &series).map_err(|e| e.to_string())?;
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
            "{} gen {generation}: stage {}, best HFF {:.5} (q {:.5}, s {:.5}, ratio {:.3}), exact {}+{} of {}, median {:.5}, {distinct} distinct; {} evaluations ({} compiled, {} hits, {} bugs); {gen_seconds:.1} s",
            utc(),
            run.stage.number(),
            be.fitness,
            be.theta_q,
            be.theta_s,
            be.ratio,
            be.quality.exact_train,
            be.quality.exact_held,
            run.judge.cases.len(),
            median(&fits),
            run.judge.evaluations,
            run.judge.compiled,
            run.judge.hits,
            run.judge.bugs
        );
        let rng_state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ u64::from(generation + 1);
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
                let ind = run.fresh(&sig, vec![], "random", g, generation + 1)?;
                children.push(ind);
            }
            cands = children;
            generation += 1;
            continue;
        }
        let mut taken: Vec<String> = Vec::new();
        for ind in &population {
            if children.len() >= elites {
                break;
            }
            if !taken.contains(&ind.source) {
                taken.push(ind.source.clone());
                let mut e = ind.clone();
                e.operator = "elite".into();
                children.push(e);
            }
        }
        let g1 = generation + 1;
        if pump_every > 0 && generation > 0 && generation.is_multiple_of(pump_every) {
            // The pump: the e-graph variants of the best enter.
            let b = population[0].clone();
            for v in variants(&run.inst, &k, &bridge, &b.tree, 2, 16, 4)? {
                if children.len() >= pop {
                    break;
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
            let pool: Vec<usize> = (0..population.len()).filter(|i| population[*i].source != population[a].source).collect();
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
            let mut m = Mutation::new(EventKind::Marriage, "subtree", format!("{married} pairs made {} offspring (typed subtree crossover, distinct parents); {no_partner} without a partner, {no_operator} with an empty domain", 2 * married));
            m.count = Some(married);
            run.tel.mutation(generation, m)?;
        }
        while children.len() < pop {
            let a = tournament(&mut rng);
            let mut gene = population[a].gene.clone();
            let mut parents = vec![population[a].id];
            let mut op = String::from("mutation");
            if rng.below(2) == 0 {
                let mut b = tournament(&mut rng);
                for _ in 0..8 {
                    if population[b].source != population[a].source {
                        break;
                    }
                    b = tournament(&mut rng);
                }
                if population[b].source != population[a].source {
                    run.gen.crossover(&mut gene, &population[b].gene, &mut rng)?;
                    parents.push(population[b].id);
                    op = "crossover+mutation".into();
                }
            }
            let m = if regime == "P" { 1 + rng.below(2) } else { 1 + rng.below(4) };
            run.gen.mutate(&mut gene, m, &mut rng)?;
            let ind = run.fresh(&sig, parents, &op, gene, g1)?;
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

    // ---- the report ----
    let best = population[0].clone();
    let be = best.eval.clone().ok_or("an unscored best")?;
    std::fs::write(out.join("best.fut"), &best.source).map_err(|e| e.to_string())?;
    let mut rep = format!(
        "problem\t{pname}\nregime\t{regime}\nstart\t{start}\nseed\t{seed}\nbackend\t{backend:?}\neval\t{mode:?}\nstopped_by\t{stopped_by}\nlast_generation\t{last_generation}\nminutes\t{:.2}\nevaluations\t{}\nbudget\t{budget}\ncompiled\t{}\ncache_hits\t{}\nbugs\t{}\nbugs_setmem_compiler_limitation\t{}\nbatch_halvings\t{}\nevaluations_per_second\t{:.3}\nstage\t{}\nfirst_exact_evaluation\t{}\nfirst_exact_generation\t{}\nbest_hff\t{:.6}\nbest_theta_quality\t{:.6}\nbest_theta_speed\t{:.6}\nbest_exact_train\t{}\nbest_exact_held\t{}\nbest_nodes\t{}\nreference_nodes\t{}\nbest_work\t{}\nreference_work\t{}\nbest_program\t{}\nmax_work_seen\t{}\n",
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
            // A speed claim counts only below the noise floor: the least ratio of both A/A sets.
            let floor = noise.iter().chain(inrun_aa.iter()).cloned().fold(f64::INFINITY, f64::min);
            rep.push_str(&format!("noise_floor_least_ratio\t{floor:.4}\nspeed_claim_beats_noise_floor\t{}\n", if median(&ratios) < floor { "yes" } else { "no" }));
        }
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
