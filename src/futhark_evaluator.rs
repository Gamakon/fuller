//! THE FUTHARK EVALUATOR: the library object that scores Futhark programs for an evolution run (plan step 5a,
//! `docs/PLAN_futhark_kingdom.md` 5a.2). fuller does not evolve code (Andrew's ruling of 2026-11-03): the generation
//! loop, selection and the population are the caller's (phylu's engine; until X1 the deprecated runner
//! `examples/futhark_evolve.rs`). This module holds what scores and varies an individual, moved out of that runner
//! unchanged (task F2):
//!
//! * [`Judge`]: the fitness machinery of every fitness scheme of the family `futhark_sig` (the cases, the batch host,
//!   the timing slots, the caches by program text, the k-means paths: [`ConvergedJudge`], the resident sample, the
//!   test case), with its types ([`Quality`], [`Eval`], [`Stages`], [`TimingInput`], [`AaSets`], [`TestCaseRun`],
//!   [`pair_steps`], [`pair_ratios`], [`LEDGER_STAGE_COLUMNS`], [`hff_dimensions`]);
//! * [`Genetics`]: the contract, the layout and the variation operators of one individual (mutation, the ring walk,
//!   crossover, marriage, the one-edit neighbours);
//! * [`FutharkEvaluator`]: the owned object a run holds for its whole life ([`FutharkEvaluator::new`] from an
//!   [`EvaluatorConfig`]), with the controls at the start ([`FutharkEvaluator::controls`]) and the names of the HFF
//!   input vector's columns ([`FutharkEvaluator::input_names`]);
//! * THE SEAM, [`FutharkEvaluator::evaluate_generation`] (task F3a, the answer-scored problems): a call's rows scored
//!   in one compiled module, each row's HFF input vector and the scheme's angle ([`RowScore`]), the stage change
//!   inside the call, the call's new evaluations and cost ([`GenerationScores`], [`GenerationCost`]), the run file
//!   checked before every batch; its messages go to the caller ([`Messages`]);
//! * [`FreshRequest`] and [`FreshIndividual`]: the types of the engine's request for new individuals (the start, a
//!   pump's refill, a fold beat's forms) and of what is returned; `FutharkEvaluator::fresh` is task F5.
//!
//! The seam for k-means (F3b), the report of a generation (F4), the state (F7) and the final report (F6) are the next
//! tasks. Every candidate's work on a case is bounded by the per-submit work cap
//! (`futhark_sig::CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT`, which `wgsl::work_guard` re-exports); the module needs no
//! feature.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::fitness::{Definition, Scheme};
use crate::futhark_batch::{Host, Inputs, KmeansSide, Member, Outcome, Program, Step, PAIR_SIDE_SECONDS};
use crate::futhark_exec::output_with_timeout_bytes;
use crate::futhark_rings::{build_at, load_pool, published_source, Rings, Source};
use crate::futhark_sig::kmeans_converged::{converge, converge_interleaved, converged_work_cap, further_iterations, kmeans_inputs, kmeans_inputs_better, kmeans_inputs_five, run_of_k_payload, sides_of_runs, step_of_exe, step_of_server, ConvergedRun, Server, StepTimer, Stepped, INPUT_NAMES_FIVE, KMEANS_CASES, QUALITY_NAMES, QUALITY_NAMES_BETTER, SPEED_NAMES, WORK_CAP_TIMES_SEED};
use crate::futhark_sig::{build, cases_fingerprint, compare, compile_c, compile_c_server, function_count, load, load_cases, problem, program, program_cases, quality_fingerprint_of, quality_input_names_of, quality_inputs_of, random_gene, read_binary, sexp, signature, Array, Case, Instance, Interp, Problem, QualityPart, Signature, Split, TIMING_WORK_CAP};
use crate::futhark_table::{FutharkKingdom, GeneRng};
use crate::futhark_wgpu_host::{compile_program_prebuilt, compile_program_prebuilt_staged, prebuild_shim, run_program_bytes, run_program_bytes_staged, CompileStages, RunStages};
use crate::geneframe::Ty;
use crate::kingdom_contract::KingdomContract;
use crate::runner::{interleaved_runs, median, FitnessStage, TimingSlots};
use crate::telemetry::{EventKind, Mutation};
use crate::typed_gep::{express, write_back, GeneLayout, Tree};
use crate::futhark_operators::crossover_once;
use crate::futhark_sig::CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT;

/// UTC wall time `HH:MM:SSZ`.
pub fn utc() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) % 86_400;
    format!("{:02}:{:02}:{:02}Z", s / 3600, (s / 60) % 60, s % 60)
}

/// WHERE THE EVALUATOR'S MESSAGES GO (the F2 code review, note 3): the library never prints. Each message is one line,
/// its UTC time first; it goes to the caller's sink when one is set (`FutharkEvaluator::set_messages`: the engine's
/// log, the deprecated runner's console), else it is held until the caller takes it
/// (`FutharkEvaluator::take_messages`) or sets a sink, which first receives every held line in order.
/// The caller's sink for the evaluator's messages: one call a line.
pub type Sink = Box<dyn FnMut(&str)>;

#[derive(Default)]
pub struct Messages {
    sink: Option<Sink>,
    held: Vec<String>,
}

impl Messages {
    /// One message, stamped with the UTC time.
    pub fn say(&mut self, msg: &str) {
        let line = format!("{} {msg}", utc());
        match &mut self.sink {
            Some(f) => f(&line),
            None => self.held.push(line),
        }
    }

    /// The caller's sink from now on; the held lines go to it first, in order.
    pub fn set_sink(&mut self, mut sink: Sink) {
        for l in self.held.drain(..) {
            sink(&l);
        }
        self.sink = Some(sink);
    }

    /// The held lines, in order (none when a sink is set: they went to it).
    pub fn take(&mut self) -> Vec<String> {
        std::mem::take(&mut self.held)
    }
}

/// FNV-1a of bytes, 16 hex digits.
pub fn text_hash_bytes(s: &[u8]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Gpu,
    C,
}

/// How the new programs of a generation are evaluated (`--eval`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMode {
    /// Each program its own compile and its own processes (the path before 2026-10-28).
    Single,
    /// Every new program of a generation in one compiled program and one process (`crate::futhark_batch`).
    Batch,
    /// The batch, compiled by the fork's rust-wgpu backend into a Rust module and run IN THIS PROCESS by the runtime
    /// (`crate::futhark_rust`): WGSL kernels and a Rust host over wgpu-rs, no C, no shim, no process.
    Rust,
}

impl EvalMode {
    /// A generation's new programs are evaluated together (`batch` or `rust`).
    pub fn batched(self) -> bool {
        matches!(self, EvalMode::Batch | EvalMode::Rust)
    }
    /// The path's name, in the ledger.
    pub fn label(self) -> &'static str {
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
pub struct Stages {
    pub render: f64,
    pub futhark: f64,
    pub port: f64,
    pub cc: f64,
    pub check: f64,
    pub link: f64,
    pub start: f64,
    pub device: f64,
    pub pipelines: f64,
    pub runs: f64,
    pub timing: f64,
}

impl Stages {
    pub const NAMES: [&'static str; 11] = ["render", "futhark", "port", "cc", "check", "link", "start", "device", "pipelines", "runs", "timing"];

    pub fn values(&self) -> [f64; 11] {
        [self.render, self.futhark, self.port, self.cc, self.check, self.link, self.start, self.device, self.pipelines, self.runs, self.timing]
    }

    pub fn of(v: [f64; 11]) -> Stages {
        Stages { render: v[0], futhark: v[1], port: v[2], cc: v[3], check: v[4], link: v[5], start: v[6], device: v[7], pipelines: v[8], runs: v[9], timing: v[10] }
    }

    pub fn add(&mut self, o: &Stages) {
        let (a, b) = (self.values(), o.values());
        *self = Stages::of(std::array::from_fn(|i| a[i] + b[i]));
    }

    pub fn scaled(&self, f: f64) -> Stages {
        let a = self.values();
        Stages::of(std::array::from_fn(|i| a[i] * f))
    }

    pub fn compiled(c: &CompileStages) -> Stages {
        Stages { futhark: c.futhark, port: c.port, cc: c.cc, check: c.check, link: c.link, ..Stages::default() }
    }

    pub fn ran(r: &RunStages) -> Stages {
        Stages { start: r.start, device: r.device, pipelines: r.pipelines, runs: r.runs, ..Stages::default() }
    }

    pub fn total(&self) -> f64 {
        self.values().iter().sum()
    }
}

/// The steps of a program's timed pairs (`missing` of them) in a batch: the process-equivalent the per-candidate path
/// discards for a freshly compiled executable (`fresh`), then each pair INTERLEAVED run by run on one upload of the
/// timing input (`Step::Pair`, `runner::interleaved_runs`), the side that goes first alternating from pair to pair.
pub fn pair_steps(missing: usize, fresh: bool, runs: usize) -> Vec<Step> {
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
pub fn pair_ratios(outs: &[Outcome], missing: usize, fresh: bool) -> Result<Vec<f64>, String> {
    let mut o = outs.iter();
    if fresh {
        o.next().ok_or("our software has a bug: no discarded timing")?.fastest_ns()?;
    }
    (0..missing).map(|_| o.next().ok_or("our software has a bug: a pair cut short")?.pair_ratio()).collect()
}

/// The input the speed is timed on.
#[derive(Debug, Clone)]
pub struct TimingInput {
    pub name: String,
    pub stdin: Vec<u8>,
    pub sizes: BTreeMap<String, usize>,
    /// The published program's output on it, when the problem has its own timing input.
    pub output: Option<Vec<u8>>,
}

impl TimingInput {
    /// The problem's own timing input (`futhark_sig::timing_input`, `Problem::timing_n`) with the published
    /// program's output on it (the yardstick through `futhark c`), or the largest training case.
    /// `with_output` false (the k-means scheme, whose speed is measured in the run and whose timing-case check is not
    /// bit for bit) leaves the published program's output out.
    pub fn of(p: &crate::futhark_sig::Problem, sig: &Signature, cases: &[Case], build_dir: &Path, with_output: bool) -> Result<TimingInput, String> {
        if let Some(stdin) = crate::futhark_sig::timing_input(p)? {
            let inputs = read_binary(&stdin)?;
            let output = if with_output {
                let exe = compile_c(p.yardstick, build_dir)?;
                Some(crate::futhark_sig::write_binary(&[&crate::futhark_sig::run_exe(&exe, &stdin)?]))
            } else {
                None
            };
            return Ok(TimingInput { name: format!("{}'s timing input", p.name), sizes: sig.sizes(&inputs)?, stdin, output });
        }
        let i = (0..cases.len()).filter(|i| cases[*i].split == Split::Train).max_by_key(|i| cases[*i].output.as_ref().map_or(0, Array::len) * cases[*i].inputs.iter().map(|a| a.len()).sum::<usize>()).ok_or("no training case")?;
        Ok(TimingInput { name: cases[i].name.clone(), stdin: cases[i].stdin.clone(), sizes: sig.sizes(&cases[i].inputs)?, output: None })
    }

    pub fn label(&self) -> String {
        let sizes: Vec<String> = self.sizes.iter().map(|(k, v)| format!("{k} = {v}")).collect();
        format!("{} ({})", self.name, sizes.join(", "))
    }
}

/// The A/A sets of the batch path: the published program against itself, the published program as a member against
/// it, and single timings of the published program (ns).
#[derive(Debug, Clone, Default)]
pub struct AaSets {
    pub noise: Vec<f64>,
    pub inrun: Vec<f64>,
    pub rounds: Vec<f64>,
}

/// The ledger's columns after the text: how the program was evaluated, its batch's size, each stage's milliseconds.
/// Then the fitness inputs, each exactly (Rust's shortest round-trip form, comma-separated): the quality inputs in case
/// order and the speed inputs (stage 2), so two runs' inputs can be compared bit for bit; then the program's
/// behaviour (the hash of its outputs on every case) and whether every output is all zeros (0 or 1).
pub const LEDGER_STAGE_COLUMNS: [&str; 17] = ["eval", "batch_size", "render_ms", "futhark_ms", "port_ms", "cc_ms", "check_ms", "link_ms", "start_ms", "device_ms", "pipelines_ms", "runs_ms", "timing_ms", "quality_inputs", "speed_inputs", "behaviour", "all_zero"];

/// What a program measured on the stored cases (independent of the stage).
#[derive(Debug, Clone)]
pub struct Quality {
    /// Three raw measurements a fitness case (differing share, mean, max), in case order: exactness.
    pub raw: Vec<[f64; 3]>,
    /// The quality inputs (`futhark_sig::quality_inputs`, the parts of `futhark_sig::QUALITY_PARTS` a case), in case order.
    pub inputs: Vec<f64>,
    pub exact_train: usize,
    pub exact_held: usize,
    /// The test cases it is exact on (reported only: a test case is never a fitness input).
    pub exact_test: usize,
    /// The most node executions on any case.
    pub work: u128,
    /// The outputs on every case, in case order (empty for a program that did not run).
    pub outputs: Vec<Array>,
    /// The program's BEHAVIOUR: the hash of its outputs on every case (two programs with the same outputs are one
    /// behaviour, whatever their texts); every program that did not run is the one behaviour `failed`.
    pub behaviour: String,
}

impl Quality {
    pub fn perfect(&self) -> bool {
        self.raw.iter().all(|r| *r == [0.0; 3])
    }
    pub fn inputs(&self) -> Vec<f64> {
        self.inputs.clone()
    }
    /// Every output is all zeros (the dead point random programs crowd at, plan section H).
    pub fn all_zero(&self) -> bool {
        !self.outputs.is_empty() && self.outputs.iter().all(|a| a.values().all(|x| x == 0.0))
    }
    /// A legal gene's program that did not compile or run (our software has a bug, logged): wrong everywhere.
    pub fn worst(cases: &[Case], parts: &[QualityPart]) -> Quality {
        let fit: Vec<&Case> = cases.iter().filter(|c| c.split.in_fitness()).collect();
        let inputs = fit.iter().flat_map(|c| vec![1.0; c.output.as_ref().map_or(0, |o| quality_input_names_of(parts, o).len())]).collect();
        Quality { raw: vec![[1.0; 3]; fit.len()], inputs, exact_train: 0, exact_held: 0, exact_test: 0, work: 0, outputs: Vec::new(), behaviour: "failed".into() }
    }
    /// The quality of a program's outputs on the cases: the fitness's inputs and exactness from the training and
    /// held-out cases alone; the test cases only counted (`exact_test`).
    pub fn of(outs: Vec<Array>, cases: &[Case], work: u128, parts: &[QualityPart]) -> Quality {
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
        q.behaviour = text_hash_bytes(&crate::futhark_sig::write_binary(&outs.iter().collect::<Vec<_>>()));
        q.outputs = outs;
        q
    }
}

/// An individual's fitness in the stage it was scored in.
#[derive(Debug, Clone)]
pub struct Eval {
    pub fitness: f64,
    pub theta_q: f64,
    pub theta_s: f64,
    pub quality: Quality,
    /// The speed inputs (stage 2) and the median ratio candidate / yardstick.
    pub speed: Vec<f64>,
    pub ratio: f64,
}



/// A batched program's quality outputs (or the failure that names it), its share of its batch's stages, and the
/// batch's size.
pub type Batched = (Result<Vec<Array>, String>, Stages, usize);

/// The fitness machinery: the cases, the yardstick's executable, the caches by program text. It owns what it reads
/// (shared handles: plan step 5a, F1).
pub struct Judge {
    pub k: Arc<FutharkKingdom>,
    pub sig: Arc<Signature>,
    /// Every case's input, in case order, as one stdin for the quality program.
    pub all_stdin: Vec<u8>,
    pub interp: Interp,
    pub cases: Vec<Case>,
    /// THE FITNESS SCHEME (`--fitness`, `crate::fitness`, family `futhark_sig`): the quality parts and the stage-2
    /// rule.
    pub scheme: &'static Scheme,
    /// The scheme's quality parts (`Scheme::quality_parts`).
    pub parts: &'static [QualityPart],
    pub sizes: Vec<BTreeMap<String, usize>>,
    /// The input the speed is timed on: the problem's own large timing input (never a quality case) or, without one,
    /// the largest training case.
    pub timing: TimingInput,
    /// The problem's timing bound (`futhark_sig::timing_work_cap`): a program whose work at the timing input is over it
    /// is never timed.
    pub timing_cap: u128,
    /// Programs over the work cap at the timing size, not timed (no speed credit).
    pub over_cap_untimed: u64,
    pub backend: Backend,
    pub runs: usize,
    pub yard_exe: Option<PathBuf>,
    /// The shim, built once at the start: every candidate links with plain cc, no cargo, no lock.
    pub shim: PathBuf,
    pub work_dir: PathBuf,
    pub quality_cache: HashMap<String, Quality>,
    /// The timing slots, kept by program text (`--pairs` of them, each its own speed input).
    pub slots: TimingSlots,
    /// FITNESS EVALUATIONS, the budget's unit (plan section E, faults 7 and 8): programs not scored before, by text.
    /// An elite carried over, a repeat and a pump child equal to a program already scored are free.
    pub evaluations: u64,
    /// Every individual scored, repeats included (reported beside `evaluations`).
    pub scored: u64,
    /// Programs compiled and run (cache misses).
    pub compiled: u64,
    pub hits: u64,
    pub bugs: u64,
    /// Of the bugs, the branch compiler's "Cannot handle SetMem statement" (a known compiler limitation).
    pub setmem: u64,
    /// Of the bugs, Futhark's internal compiler errors (each a kingdom bug, logged with its program).
    pub ice: u64,
    pub bug_log: PathBuf,
    pub max_work: u128,
    pub mode: EvalMode,
    /// The batch host (`--eval batch`).
    pub host: Option<Host>,
    /// `--eval single`'s TIMING HOST: a candidate's timed pairs are its own library build and its own process, the
    /// batch driver's interleaved pairs on one upload (`Step::Pair`), so the three paths time alike. Measured
    /// 2026-10-31: Futhark's own command line re-uploads the timing input every run, and the GPU's clock stepped down
    /// in the gaps (the runner's A/A gate, 9 of 22 pairs within 5 % on histogram). Futhark's command line still gives
    /// the quality outputs.
    pub timer: Option<Host>,
    /// Batch mode: the quality outputs of the programs batched this generation, by text, with each one's share of its
    /// batch's stages and the batch's size; taken when the program is scored (so the counts, the bugs and the ledger
    /// are the per-candidate path's).
    pub batched_q: HashMap<String, Batched>,
    /// Batch mode: the timed pairs of the programs batched this generation, by text, with each one's timing seconds.
    pub batched_t: HashMap<String, (Result<Vec<f64>, String>, f64)>,
    /// The stages of this generation (the batches' and the single evaluations' wall time).
    pub gen_stages: Stages,
    /// Batches halved although no program in them was at fault (a limit of the batch, logged in `bugs.log`).
    pub batch_splits: u64,
    /// The last evaluation's path (`single`, `batch`, `hit`), its batch's size and its stages, for the ledger.
    pub last: (&'static str, usize, Stages),
    /// THE K-MEANS SCHEME's judge (`futhark-kmeans-better`, or the retired `futhark-kmeans-qqs`,
    /// `futhark-kmeans-quality-first` and `futhark-kmeans-converged`), when that is the scheme.
    pub converged: Option<ConvergedJudge>,
    /// The k-means scheme's speed group of every program scored, by text (its quality group is in `quality_cache`).
    pub converged_speed: HashMap<String, Vec<f64>>,
    /// The k-means scheme's candidates over the work cap on a fitness case, not run, scored worst.
    pub over_cap_unrun: u64,
    /// The k-means scheme's candidate runs that did not finish (a NaN or infinite centroid, a wrong shape, a step that
    /// failed): the candidate's worst score on that case, not a bug of our software.
    pub unfinished: u64,
    /// THE REPETITION COUNT of a flat scheme (`--quality-weight`): the quality inputs are entered this many times in
    /// the one vector the HFF reads (`fitness::qqs`); 1 for a scheme that is not flat.
    pub quality_weight: usize,
    /// THE CLIFF'S WIDTH ε of `futhark-kmeans-better` (`--quality-cliff`, `fitness::quality_cliff`); unread by every
    /// other scheme.
    pub quality_cliff: f64,
    /// `--kmeans-step process` (the c backend): one process an iteration instead of a server, for the measurement.
    pub step_process: bool,
    /// The resident-sample path: the runs of the programs batched this generation, by text, with each one's share of
    /// the batch's stages (`Judge::prefetch`), taken when the program is scored.
    pub batched_k: HashMap<String, (BatchedKmeans, Stages)>,
    /// The resident sample's upload seconds this generation (0 once resident) and the runs that found it resident.
    pub upload_seconds: f64,
    pub resident_hits: u64,
    /// The messages of the judge and the evaluator ([`Messages`]): the caller's, never printed here.
    pub messages: Messages,
    /// Modules compiled (a batch's compiles, or one a program on `--eval single`), compiles that failed, and compiles
    /// the compiler did not finish within the host's limit: the run's totals, for each call's [`GenerationCost`].
    pub modules: u64,
    pub failed_compiles: u64,
    pub unfinished_compiles: u64,
}

/// HOW A COMPILED K-MEANS PROGRAM IS STEPPED: on the GPU one process of the shim's executable an iteration (the
/// program's own time its GPU timestamps); on the c backend a Futhark server kept alive across the iterations and the
/// cases (`kmeans_converged::Server`, the program's own time the call's `runtime:`).
pub enum Stepper {
    Exe(PathBuf, StepTimer),
    Server(Server),
}

/// A step on a case, as `converge` takes it, boxed (the two steppers' closures differ in type).
pub type BoxedStep<'s> = Box<dyn FnMut(&Array) -> Result<Stepped, String> + 's>;

impl Stepper {
    /// The step on a case, as `converge` takes it; a server loads the case's points once.
    pub fn step<'s>(&'s mut self, case: &'s Case) -> Result<BoxedStep<'s>, String> {
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
pub struct ConvergedJudge {
    /// The seed's stepper (`--eval single`); none on the resident-sample path, where the seed is an entry of every
    /// generation's module.
    pub seed: Option<Stepper>,
    /// The seed's run on each fitness case at the start, in case order (`Case::split.in_fitness()`).
    pub seed_runs: Vec<ConvergedRun>,
    /// The work cap of each fitness case, in case order.
    pub work_caps: Vec<u128>,
    pub pairs: usize,
    /// The shared points files of the servers (`Server::start`'s `points_dir`).
    pub points_dir: PathBuf,
    /// THE RESIDENT-SAMPLE PATH (`--eval rust`; Andrew, 2026-11-03): every stored case's row range `(start, n)` of
    /// the one sample, in case order, checked at the start against the case's points. The sample is uploaded once a
    /// session and kept on the device; a generation's programs are one module; each candidate's cases run against
    /// the resident sample interleaved with the seed's entry (`futhark_batch::Step::Kmeans`).
    pub cuts: Option<Vec<(usize, usize)>>,
}

/// A k-means program's runs on the resident-sample path, by its text: the (candidate, seed) runs a case, or the
/// program's failure (a compile the module refused, found by the batch's halving).
pub type BatchedKmeans = Result<Vec<(ConvergedRun, ConvergedRun)>, String>;

/// THE CHAMPION ON A TEST CASE (the round-2 review's N2): the candidate's and the seed's runs of one interleaving on
/// a case that is never a fitness input, with the candidate's four inputs against that seed run.
pub struct TestCaseRun {
    pub case: String,
    pub candidate: ConvergedRun,
    pub seed: ConvergedRun,
    pub quality: Vec<f64>,
    pub speed: Vec<f64>,
}

impl TestCaseRun {
    /// One line: the inputs, then the raw measures of both sides.
    pub fn line(&self) -> String {
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
        let raw = |r: &ConvergedRun| format!("inertia {:.5} log10p {:.2} (all {:.2}, {} at centroid) seconds {:.4} iterations {} converged {} finished {}", r.inertia, r.mean_log10_p, r.mean_log10_p_all, r.coincident, r.seconds, r.iterations, r.converged, r.finished);
        format!("{}: quality [{}] speed [{}]; champion {}; seed {}", self.case, fmt(&self.quality), fmt(&self.speed), raw(&self.candidate), raw(&self.seed))
    }
}

impl Judge {
    /// The fitness scheme's quality parts.
    pub fn parts(&self) -> &'static [QualityPart] {
        self.parts
    }

    /// A candidate's quality program compiled per candidate (`--eval single`), with each stage's wall time (the c
    /// backend's compile is all `futhark`).
    pub fn compile_staged(&self, source: &str, dir: &Path) -> Result<(PathBuf, Stages), String> {
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
    pub fn run_all(&self, exe: &Path) -> Result<(Vec<Array>, Stages), String> {
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
    pub fn work_of(&self, tree: &Tree) -> Result<u128, String> {
        let mut work = 0u128;
        for s in &self.sizes {
            work = work.max(self.interp.work(tree, s)?);
        }
        Ok(work)
    }

    /// BATCH MODE: every program of `items` that will need a compile this generation — its quality not yet known,
    /// or (in a stage with speed) its timing slots not yet filled — in one batch (`crate::futhark_batch`): one
    /// compile and one process (more only when the halving must find a program the compiler refuses). The results are
    /// kept by text and taken when each program is scored, so selection, the counts and the bugs are the
    /// per-candidate path's. Programs over the work cap are left out: scoring them stops the run, as before.
    pub fn prefetch(&mut self, items: &[(&Tree, &str)], stage: FitnessStage) -> Result<(), String> {
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
        self.modules += c.compiles as u64;
        self.failed_compiles += c.failed_compiles as u64;
        self.unfinished_compiles += c.unfinished_compiles as u64;
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
        self.messages.say(&format!("batch: {n} programs, {} compile(s) ({} failed, {} not finished within the limit), {} process(es), {} halving(s), {:.1} s", c.compiles, c.failed_compiles, c.unfinished_compiles, c.processes, c.splits.len(), t0.elapsed().as_secs_f64()));
        Ok(())
    }

    /// THE RESIDENT-SAMPLE PATH'S GENERATION (`--eval rust`; Andrew, 2026-11-03): every program of `items` not scored
    /// before and under the work cap on every fitness case, in ONE module with the seed's entry, run against the
    /// resident sample on every fitness case `pairs` times interleaved with the seed; the runs kept by text and taken
    /// when the program is scored (`evaluate_converged`). The batch's stages are shared out as the quality batch's
    /// are; the upload (once a session) and the resident hits are counted for the generation's report.
    pub fn prefetch_kmeans(&mut self, items: &[(&Tree, &str)]) -> Result<(), String> {
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
        self.messages.say(&format!("k-means batch (the resident sample): {n} programs in one module, {} compile(s) ({} failed, {} not finished within the limit, {} compiled with the fusion bound), {} run(s), {} halving(s); compile {:.1} s, upload {:.3} s ({} upload(s), {} resident), run {:.1} s; {:.1} s in all", c.compiles, c.failed_compiles, c.unfinished_compiles, c.bounded_fusion_compiles, c.processes, c.splits.len(), c.compile.futhark + c.compile.cc, c.upload_seconds, c.uploads, c.resident_hits, c.run.total, t0.elapsed().as_secs_f64()));
        Ok(())
    }

    /// BATCH MODE: the A/A sets in one process: the published program against itself (a discarded pair, then 11),
    /// and, for P, the published program as a member (wrapped as a candidate's text is) against it through the
    /// slots' scheme (`inrun_pairs`, the first process-equivalent discarded); with `rounds`, that many single timings
    /// of the published program (the gate's). Returns (noise, in-run A/A, rounds in ns).
    pub fn aa_batch(&self, yardstick: &str, inrun_pairs: usize, rounds: usize) -> Result<AaSets, String> {
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
    pub fn aa_medians(&self, reference: &Tree, retimes: usize) -> Result<Vec<f64>, String> {
        (0..retimes).map(|_| self.champion_batch(reference, 11).map(|(r, _)| median(&r))).collect()
    }

    /// BATCH MODE: the champion's re-time (`pairs`, the first process-equivalent discarded) and its output on the
    /// timing input, in one process.
    pub fn champion_batch(&self, tree: &Tree, pairs: usize) -> Result<(Vec<f64>, Array), String> {
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
    pub fn timing_host(&self) -> Result<&Host, String> {
        match self.mode {
            EvalMode::Single => self.timer.as_ref().ok_or_else(|| "our software has a bug: --eval single times on the GPU with no timing host (the c backend has no GPU timing)".to_string()),
            EvalMode::Batch | EvalMode::Rust => self.host.as_ref().ok_or_else(|| "our software has a bug: batch mode with no batch host".to_string()),
        }
    }

    /// `--eval single`: one candidate's `missing` timed pairs (`fresh`: its first process-equivalent discarded) in its
    /// own build and process, with the build's stages.
    pub fn time_single(&mut self, tree: &Tree, missing: usize, fresh: bool) -> Result<(Vec<f64>, Stages), String> {
        let r = self.timing_host()?.run(&[Member { program: Program::Tree(tree), steps: pair_steps(missing, fresh, self.runs) }])?;
        self.modules += r.cost.compiles as u64;
        self.failed_compiles += r.cost.failed_compiles as u64;
        self.unfinished_compiles += r.cost.unfinished_compiles as u64;
        let st = Stages::compiled(&r.cost.compile);
        Ok((pair_ratios(&r.outcomes[0], missing, fresh)?, st))
    }

    pub fn log_bug(&mut self, msg: &str, source: &str) {
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
    pub fn quality(&mut self, tree: &Tree, source: &str) -> Result<Quality, String> {
        let work = self.work_of(tree)?;
        if work > CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT {
            return Err(format!("THE WORK GUARD: a candidate runs {work} node executions on a case, over the per-submit cap {CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT}; the experiment's sizes are misconfigured for this layout\n{source}"));
        }
        self.max_work = self.max_work.max(work);
        let t = Instant::now();
        let batch = program_cases(&self.k, &self.sig, tree, self.cases.len())?;
        let render = t.elapsed().as_secs_f64();
        let got = match self.mode {
            EvalMode::Single => {
                let dir = self.work_dir.join("quality");
                let mut st = Stages { render, ..Stages::default() };
                let r = match self.compile_staged(&batch, &dir) {
                    Ok((exe, cs)) => {
                        self.modules += 1;
                        st.add(&cs);
                        self.run_all(&exe).map(|(o, rs)| {
                            st.add(&rs);
                            o
                        })
                    }
                    Err(e) => {
                        self.failed_compiles += 1;
                        Err(e)
                    }
                };
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
    pub fn converged_widths(&self) -> (usize, usize) {
        match self.scheme.definition {
            Definition::FutharkConvergedGated | Definition::FutharkConvergedQqs => (QUALITY_NAMES.len(), SPEED_NAMES.len()),
            Definition::FutharkKmeansBetter => (QUALITY_NAMES_BETTER.len(), SPEED_NAMES.len()),
            _ => (INPUT_NAMES_FIVE.len(), 0),
        }
    }

    /// The two input groups of one case under the run's k-means scheme (`futhark-kmeans-better`'s three cliff inputs
    /// and two speed inputs, `kmeans_inputs_better`; the flat and gated schemes' two and two, `kmeans_inputs`), the
    /// candidate's run against the seed's of the same interleaving.
    pub fn case_inputs(&self, cand: &ConvergedRun, seed: &ConvergedRun) -> Result<crate::futhark_sig::kmeans_converged::Inputs, String> {
        if self.scheme.uses_cliff() {
            kmeans_inputs_better(cand, seed, self.quality_cliff)
        } else {
            Ok(kmeans_inputs(cand, seed))
        }
    }

    /// The k-means scheme's compiled candidate: on the c backend a server (`futhark c --server`, started here and
    /// kept for the candidate's cases), on the GPU the shim's executable, with the compile's stages.
    pub fn compile_converged(&self, source: &str, dir: &Path) -> Result<(Stepper, Stages), String> {
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
    pub fn converged_pair(cj: &mut ConvergedJudge, candidate: &mut Stepper, case: &Case, further: bool) -> Result<(ConvergedRun, ConvergedRun), String> {
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
    pub fn kmeans_batch(&mut self, members: &[Program], case_idx: &[usize], pairs: usize) -> Result<(Vec<BatchedKmeans>, crate::futhark_batch::Cost), String> {
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
                        Err(e) if *candidate => cand.push(run_of_k_payload(&c.inputs[0], &c.inputs[1], &crate::futhark_sig::kmeans_converged::k_payload(0, false, false, 0.0, 0.0, 0.0, format!("the step failed in the host: {e}").as_bytes()), 0)?),
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
    pub fn converged_on_test(&mut self, tree: &Tree, source: &str) -> Result<Vec<TestCaseRun>, String> {
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
    pub fn evaluate_converged(&mut self, tree: &Tree, source: &str) -> Result<(Quality, Vec<f64>), String> {
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
                self.messages.say(&format!("over the work cap on {}: {w} node executions, the cap {} ({WORK_CAP_TIMES_SEED} x the seed's); not run, the worst score", self.cases[*i].name, caps[slot]));
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
                let behaviour = text_hash_bytes(&crate::futhark_sig::write_binary(&outputs.iter().collect::<Vec<_>>()));
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
    pub fn evaluate(&mut self, tree: &Tree, source: &str, stage: FitnessStage) -> Result<Eval, String> {
        self.scored += 1;
        self.last = ("hit", 0, Stages::default());
        if self.converged.is_some() {
            // The k-means scheme: the fitness of the two groups by the scheme's rule (`Scheme::converged_fitness`); a
            // program scored before reads its cached groups.
            let eval = |f: crate::runner::TwoLevel, q: Quality, speed: Vec<f64>| Eval { fitness: f.fitness, theta_q: f.quality, theta_s: f.speed, quality: q, speed, ratio: f64::NAN };
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
            let missing = self.slots.missing(source);
            // The c backend has no GPU timing: a program with an empty timing slot cannot be scored in stage 2 on it.
            // One whose slots are all filled (the in-run A/A, a restored state) needs no timing.
            if missing > 0 && self.backend == Backend::C {
                return Err("stage 2 times on the GPU: the c backend cannot enter it (run with --backend gpu)".into());
            }
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

/// The genetic side: the contract, the layout and the variation operators. It owns what it reads (shared handles:
/// plan step 5a, F1).
pub struct Genetics {
    pub k: Arc<FutharkKingdom>,
    pub contract: Arc<KingdomContract>,
    pub layout: Arc<GeneLayout>,
    pub const_len: Arc<BTreeMap<Ty, u32>>,
    /// THE RING CANVAS (`--start llm-ring`, plan section H): every fresh gene is a canvas of these rings, not random.
    pub canvas: Option<Arc<Rings>>,
    /// `--mutation ring-walk` (plan section I.1): ring walk replaces point mutation.
    pub ring_walk: bool,
}

impl Genetics {
    /// A FRESH gene: a ring canvas with `--start llm-ring` (`crate::futhark_rings::Rings::canvas`), else a random
    /// gene. Every fresh gene of a run comes from here: generation 0's padding, C's draws, a resumed tree's padding,
    /// the slope check's draws.
    pub fn random_gene(&self, rng: &mut GeneRng) -> Result<Vec<u32>, String> {
        match &self.canvas {
            Some(r) => r.canvas(&self.contract, &self.layout, rng),
            None => random_gene(&self.contract, &self.layout, &self.const_len, rng),
        }
    }

    /// THE MUTATION of the loop (`--mutation`): `point`, today's point mutation of `m` coding cells; `ring-walk`, one
    /// ring walk of one type's head or tail window by `k` in `1 ..= m` positions (`Rings::walk`), returned so the
    /// caller records it as a `ring_walk` event.
    pub fn mutation(&self, gene: &mut [u32], m: usize, rng: &mut GeneRng) -> Result<Option<crate::futhark_rings::Walk>, String> {
        if !self.ring_walk {
            self.mutate(gene, m, rng)?;
            return Ok(None);
        }
        let rings = self.canvas.as_ref().ok_or("our software has a bug: ring walk with no rings")?;
        let coding = self.coding_cells(gene)?;
        rings.walk(&self.layout, gene, &coding, m, rng)
    }

    /// The event of a ring walk on a child (its id and its tree) of `parent`.
    pub fn walk_event(&self, w: &crate::futhark_rings::Walk, child_id: u64, child_tree: &Tree, parent: u64) -> Mutation {
        let mut m = Mutation::new(EventKind::RingWalk, "ring_walk", format!("{} {} advanced {} along its ring", self.k.dual(w.ty).map(|d| d.dual.as_str()).unwrap_or(""), w.domain.name(), w.k));
        m.id = Some(child_id);
        m.parents = vec![parent];
        m.text_after = Some(sexp(&self.k, child_tree));
        m
    }

    /// The operator name of a fresh gene in the ledger.
    pub fn fresh_label(&self) -> &'static str {
        if self.canvas.is_some() {
            "canvas"
        } else {
            "random"
        }
    }

    pub fn tree(&self, gene: &[u32]) -> Result<Tree, String> {
        Ok(express(&self.contract, &self.layout, gene, &self.const_len).map_err(|e| format!("a gene does not express: {e}"))?.tree())
    }

    /// The cells the walk reads: every expressed node's symbol cell and, for an expressed constant, its Dc cell.
    pub fn coding_cells(&self, gene: &[u32]) -> Result<Vec<usize>, String> {
        crate::futhark_rings::coding_cells(&self.contract, &self.layout, gene, &self.const_len)
    }

    /// LOW-CHANGE mutation: `m` expressed cells, each set to another member of its own alphabet.
    pub fn mutate(&self, gene: &mut [u32], m: usize, rng: &mut GeneRng) -> Result<(), String> {
        crate::futhark_rings::point_mutate(&self.contract, &self.layout, &self.const_len, gene, m, rng)
    }

    /// THE ONE-EDIT NEIGHBOURS of a gene: every expressed cell set to every other member of its own alphabet (the
    /// point mutations of [`Genetics::mutate`]), each child's tree once, the gene's own tree left out. Returns each
    /// with the cell it changed.
    pub fn neighbours(&self, gene: &[u32]) -> Result<Vec<(usize, Tree)>, String> {
        let own = self.tree(gene)?;
        let mut seen: HashSet<Tree> = HashSet::new();
        let mut out = Vec::new();
        for pos in self.coding_cells(gene)? {
            for v in self.layout.alphabet(&self.contract, pos, &self.const_len) {
                if v == gene[pos] {
                    continue;
                }
                let mut g = gene.to_vec();
                self.layout.mutate(&self.contract, &mut g, pos, v, &self.const_len).map_err(|e| format!("a neighbour: {e}"))?;
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
    pub fn crossover(&self, a: &mut [u32], b: &mut [u32], rng: &mut GeneRng) -> Result<(), String> {
        crossover_once(&self.contract, &self.layout, &self.const_len, a, b, rng)
    }

    /// MARRIAGE, typed subtree crossover: the domain is every pair (node of A, node of B) of one output type, not the
    /// two roots, not identical subtrees, whose two offspring both write back into the parents' genes; one pair is
    /// drawn uniformly. Each parent is its expressed tree and its gene. Returns the two offspring genes, or None for
    /// an empty domain.
    pub fn marry(&self, a: (&Tree, &[u32]), b: (&Tree, &[u32]), rng: &mut GeneRng) -> Result<Option<Offspring>, String> {
        let ((a_tree, a_gene), (b_tree, b_gene)) = (a, b);
        let na = a_tree.level_order();
        let nb = b_tree.level_order();
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
            let ca = replace(a_tree, &pa, &sb);
            let cb = replace(b_tree, &pb, &sa);
            if let (Ok(ga), Ok(gb)) = (write_back(&self.contract, &self.layout, a_gene, &ca), write_back(&self.contract, &self.layout, b_gene, &cb)) {
                return Ok(Some((ga, gb)));
            }
        }
        Ok(None)
    }
}

/// A marriage's two offspring genes.
pub type Offspring = (Vec<u32>, Vec<u32>);

/// `t` with the subtree at `path` replaced.
pub fn replace(t: &Tree, path: &[u32], with: &Tree) -> Tree {
    match path.split_first() {
        None => with.clone(),
        Some((i, rest)) => {
            let mut kids = t.kids.clone();
            kids[*i as usize] = replace(&kids[*i as usize], rest, with);
            Tree { symbol: t.symbol, const_index: t.const_index, kids }
        }
    }
}

/// THE RING CANVAS of a run (`--start llm-ring`, plan section H): the pool's programs (`DIR/bench.tsv`, the published
/// benchmarks' functions, then `DIR/llm.tsv`, the topic programs, written by `examples/futhark_pool.rs`) of the chosen
/// sources, in order, into the rings; the problem's own published file is left out (leave one out: its R and C runs
/// and its measurements must not have the answer in the canvas); the empty rings filled (`Rings::complete`). Writes
/// `OUT/rings.tsv` (what went in, every fill and gap). Returns the rings and the TASK ALTERNATIVES (`DIR/alternatives.tsv`,
/// Sonnet's programs for the benchmark's own task, Andrew's clarification of 2026-10-29): the population's material,
/// never the rings', in order, those of the root type that fit the layout.
pub fn ring_canvas(k: &FutharkKingdom, contract: &KingdomContract, layout: &GeneLayout, dir: &Path, pname: &str, sources: &[Source], out: &Path, messages: &mut Messages) -> Result<(Rings, Vec<Tree>), String> {
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
            if crate::typed_gep::fits(contract, layout, &t).is_ok() {
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
    messages.say(&format!("ring canvas: {used} pool programs ({by_source:?}), {left_out} left out ({excluded}), {by_class} rings filled from their class, {by_slot} from their slot family (class gaps); {} task alternatives fit the layout, {unfit} too big", alternatives.len()));
    Ok((full, alternatives))
}

/// THE GATE (`--gate DIR`): a discovery run (R, C) starts only after the slope check of the same problem, program
/// size, cases and quality inputs passed (`DIR/check_summary.tsv`).
pub fn gate_passed(dir: &Path, problem: &str, program_size: usize, fp_cases: &str, parts: &[QualityPart]) -> Result<(), String> {
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

/// THE LENGTH d OF A K-MEANS INDIVIDUAL'S FLAT VECTOR (`fitness::qqs`): its quality inputs entered `quality_weight`
/// times and its speed inputs once, the `d` of its HFF percentile.
pub fn hff_dimensions(e: &Eval, quality_weight: usize) -> usize {
    e.quality.inputs.len() * quality_weight + e.speed.len()
}

/// The run's regime (`docs/STUDY_evolving_benchmarks.md`): R from random, C the random-search control, P from the
/// published program toward faster (P times the in-run A/A at the start).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    R,
    C,
    P,
}

/// THE RING CANVAS of a run (plan section H, [`ring_canvas`]): the pool directory (`bench.tsv`, `llm.tsv`,
/// `alternatives.tsv`, written by `examples/futhark_pool.rs`), the sources that enter the rings, and whether every
/// fresh gene is a canvas of the rings (`--start llm-ring`); without that the pool gives the task alternatives only
/// (the operator measurement's parents).
#[derive(Debug, Clone)]
pub struct RingConfig {
    pub pool: PathBuf,
    pub sources: Vec<Source>,
    pub canvas: bool,
}

/// What a run's evaluator is made from (plan 5a.2). `None` fields take the problem's or the scheme's own default.
#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    /// One of `futhark_sig::PROBLEM_NAMES`.
    pub problem: String,
    /// A `fuller::fitness` scheme of the family `futhark_sig`; None: the problem's own default (`Problem::fitness`).
    pub fitness: Option<String>,
    /// The repetition count of a flat scheme; None: the scheme's default. Given for a scheme that is not flat it is
    /// an error.
    pub quality_weight: Option<usize>,
    /// The cliff's width of `futhark-kmeans-better`; None: `fitness::DEFAULT_QUALITY_CLIFF`. Given for another
    /// scheme it is an error.
    pub quality_cliff: Option<f64>,
    pub backend: Backend,
    pub eval: EvalMode,
    /// The timing slots of a program (the answer-scored schemes) or the interleaved pairs of a k-means candidate.
    pub pairs: usize,
    /// The runs of one timed side (`Step::Time`).
    pub runs: usize,
    /// The program size N of the gene layout; None: the problem's own (`Problem::program_size`).
    pub program_size: Option<usize>,
    /// The run's directory: `bugs.log`, `bugs/`, the build directories, `rings.tsv`.
    pub out: PathBuf,
    /// The c backend's k-means step as one process an iteration instead of a Futhark server (the measurement).
    pub kmeans_step_process: bool,
    /// THE GATE: the slope check's directory a discovery run must have passed (`check_summary.tsv`, [`gate_passed`]).
    pub gate: Option<PathBuf>,
    /// The ring canvas and the task alternatives (the pool), when the run uses them.
    pub rings: Option<RingConfig>,
    /// The ring walk replaces point mutation (it needs the ring canvas).
    pub ring_walk: bool,
}

/// What the controls at the start time and run (plan 5a.2, `FutharkEvaluator::controls`).
#[derive(Debug, Clone, Copy)]
pub struct ControlRequest {
    /// The run's regime; None for a measurement that is not a run (the slope check, the operator measurement): no A/A
    /// sets, no check of the timing input's output, no in-run A/A.
    pub regime: Option<Regime>,
    /// Single timings of the published program for the A/A gate (`--aa-only`: 3; else 0).
    pub aa_rounds: usize,
}

/// The controls' measurements, for the run's report and its gate file.
#[derive(Debug, Clone, Default)]
pub struct Controls {
    /// The A/A noise floor: the published program against itself, 11 interleaved pairs (GPU only).
    pub noise: Vec<f64>,
    /// The A/A sets as timed (the noise floor, the in-run A/A, the gate's single timings), when timed.
    pub aa: Option<AaSets>,
    /// The in-run A/A of regime P: the published program as a candidate through the timing slots.
    pub inrun_aa: Vec<f64>,
    /// The k-means scheme's A/A: the reference program judged as a candidate against itself, its quality and speed
    /// inputs (the speed inputs' spread around 1/2 is the noise floor of a speed claim).
    pub kmeans_aa_quality: Vec<f64>,
    pub kmeans_aa_speed: Vec<f64>,
    /// The run's notes of generation 0, in order (the stream's `note` events).
    pub notes: Vec<String>,
}

/// What the engine asks new individuals for (plan 5a.2; `FutharkEvaluator::fresh` is task F5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshPurpose {
    /// The start population.
    Start,
    /// One pair's pump refill.
    PumpRefill,
    /// The fold beat's forms.
    FoldBeat,
}

/// The engine's request for new individuals (plan 5a.2).
#[derive(Debug, Clone)]
pub struct FreshRequest {
    pub purpose: FreshPurpose,
    /// The pair whose intake is refilled.
    pub pair: usize,
    /// The rows the engine will place (13 a pump at intake 16): the most that may be returned.
    pub rows: usize,
    /// The champion island's best of that pair.
    pub best: Option<Vec<u32>>,
    pub generation: u32,
}

/// One new individual for the engine to place (plan 5a.2; phylu's widened `ExternalFresh` imports it, P2a).
#[derive(Debug, Clone)]
pub struct FreshIndividual {
    pub genome: Vec<u32>,
    /// "reference", "egraph_variant", "saturated_fold", "pump_mutant", "random".
    pub origin: String,
    /// Rows of the population the request saw.
    pub parents: Vec<usize>,
    /// The stream's mutation event (`pump`, `saturated_fold`, `egraph_exact`, ...).
    pub event: Mutation,
    /// A text not scored before: a budgeted evaluation when it is scored.
    pub new_text: bool,
}

/// What one [`FutharkEvaluator::evaluate_generation`] call cost (plan 5a.2): its wall time by stage (the batches' and
/// the single evaluations', [`Stages`], in seconds) and their sums, the modules compiled, the compiles that failed or
/// did not finish within the host's limit, the batches halved, the resident sample's upload and resident hits.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationCost {
    pub stages: Stages,
    /// The Futhark compiler, the WGSL port, `cc`, the surface check and the link (s).
    pub compile_seconds: f64,
    /// The processes' start, device set-up, pipeline creation and the runs on the stored cases (s).
    pub run_seconds: f64,
    /// The speed timing of stage 2 (s).
    pub timing_seconds: f64,
    pub upload_seconds: f64,
    /// Modules compiled: one a call on `--eval batch` and `--eval rust` (two when the stage changes and stage 2 times
    /// programs, more when a batch is halved), one a new program on `--eval single`.
    pub modules: u64,
    pub failed_compiles: u64,
    pub unfinished_compiles: u64,
    pub halvings: u64,
    pub resident_hits: u64,
    /// The call's wall time (s).
    pub seconds: f64,
}

/// ONE ROW's score (plan 5a.2): its program, its HFF input vector, the scheme's angle of it and the measures behind it.
#[derive(Debug, Clone)]
pub struct RowScore {
    /// The program as the ledger writes it (the kingdom's s-expression), rendered once a distinct gene.
    pub text: String,
    /// Its Futhark source (the declared signature): the key of every cache.
    pub source: String,
    /// THE HFF INPUT VECTOR as the scheme builds it: in stage 1 the quality group; in stage 2 of a flat scheme the
    /// quality group `quality_weight` times and the speed group once (`fitness::flat_vector`); in stage 2 of a
    /// retired two-level scheme (gated, ungated) the two groups once each, whose angle is not taken over one vector.
    pub inputs: Vec<f64>,
    /// The scheme's angle (`fuller::fitness`): THE number the engine's selection reads.
    pub fitness: f64,
    pub theta_quality: f64,
    pub theta_speed: f64,
    /// F(fitness) at d = `inputs.len()` (`fitness::hff_percentile`) when the angle is TrueNorth over `inputs` (stage 1,
    /// and stage 2 of a flat scheme); None for a two-level angle.
    pub percentile: Option<f64>,
    pub d: usize,
    /// The quality inputs in case order and the speed inputs (stage 2: one a timing slot, `r/(1+r)`).
    pub quality: Vec<f64>,
    pub speed: Vec<f64>,
    /// The median ratio candidate / yardstick of the timing slots (stage 2; NaN in stage 1).
    pub ratio: f64,
    /// Exact cases: training, held-out, test (the test cases are never a fitness input).
    pub exact: [usize; 3],
    /// Exact on every fitness case (training and held-out).
    pub perfect: bool,
    pub behaviour: String,
    pub all_zero: bool,
    /// This row's program was not scored before: a budgeted evaluation (one a distinct new text, its first row).
    pub new_evaluation: bool,
    /// The deprecated runner's ledger columns after the text ([`LEDGER_STAGE_COLUMNS`], [`ledger_columns`]).
    pub ledger: Vec<String>,
}

/// One call's scores (plan 5a.2).
#[derive(Debug, Clone)]
pub struct GenerationScores {
    /// One a row handed over, in row order; when `stopped`, the rows scored before the run file went.
    pub rows: Vec<RowScore>,
    /// The stage the rows were scored in.
    pub stage: FitnessStage,
    /// The best row became exact on every case: every row was scored again in stage 2 within this call.
    pub stage_changed: bool,
    /// The run file (`OUT/killme.trigger`) was gone before a batch: the engine ends the run as `killed`.
    pub stopped: bool,
    /// This call's new evaluations (what the engine charges to the budget: programs not scored before, by text; a
    /// repeat, an elite or a program scored by an earlier call is free) and the run's total after it.
    pub evaluations_new: u64,
    pub evaluations_total: u64,
    pub cost: GenerationCost,
}

/// A batch finished inside a call ([`FutharkEvaluator::evaluate_generation_with`]): its number in the call (from 1),
/// the stage it scored for, and the rows scored so far in that stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchDone {
    pub batch: usize,
    pub stage: FitnessStage,
    pub rows_scored: usize,
}

/// THE LEDGER COLUMNS of a scored row after its text, as the deprecated runner writes them ([`LEDGER_STAGE_COLUMNS`]):
/// the evaluation path, its batch's size, each stage's milliseconds (one decimal), the quality and speed inputs each
/// exactly (Rust's shortest round-trip form, comma-separated, `-` for none), the behaviour, and whether every output
/// is all zeros.
pub fn ledger_columns(last: &(&'static str, usize, Stages), e: &Eval) -> Vec<String> {
    let (path, batch, st) = last;
    let mut cols = vec![path.to_string(), batch.to_string()];
    cols.extend(st.values().iter().map(|v| format!("{:.1}", v * 1e3)));
    let exact = |v: &[f64]| if v.is_empty() { "-".to_string() } else { v.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>().join(",") };
    cols.push(exact(&e.quality.inputs));
    cols.push(exact(&e.speed));
    cols.push(e.quality.behaviour.clone());
    cols.push(u8::from(e.quality.all_zero()).to_string());
    cols
}

/// A distinct gene of a call, expressed and rendered once.
struct Rendered {
    tree: Tree,
    text: String,
    source: String,
}

/// THE FUTHARK EVALUATOR (plan 5a.2): one owned object a run holds for its whole life. It holds the problem, the
/// reference program, the [`Judge`] and the [`Genetics`], each owning what it reads (F1); nothing in it borrows.
pub struct FutharkEvaluator {
    problem: Problem,
    program_size: usize,
    reference: Tree,
    planted: Tree,
    reference_source: String,
    /// The task alternatives of the pool that fit the layout ([`ring_canvas`]), in the pool's order.
    alternatives: Vec<Tree>,
    out: PathBuf,
    /// The controls ran ([`FutharkEvaluator::controls`]): they run once a run.
    controls_ran: bool,
    /// THE FITNESS STAGE of [`FutharkEvaluator::evaluate_generation`]: stage 1 (quality alone) from the start, stage 2
    /// from the start in regime P (set by the controls), and from the call in which the best row is exact on every
    /// case on; a stage never goes back.
    stage: FitnessStage,
    /// The fitness machinery. Public for the deprecated runner (`examples/futhark_evolve.rs`), which drives it
    /// directly until X1; the engine's seam is the evaluator's methods.
    pub judge: Judge,
    /// The genetic side. Public for the deprecated runner, as [`FutharkEvaluator::judge`].
    pub gen: Genetics,
}

impl FutharkEvaluator {
    /// The evaluator of a problem under a fitness scheme on an evaluation path: the kingdom, the signature, the
    /// contract and the layout loaded; the reference program and the planted slowdown built and fitted to the layout;
    /// the ring canvas when configured; the stored cases (fitness and test) and the gate checked; the timing input
    /// with its work bound; the shim, the yardstick and the hosts built. The configuration is checked first: every
    /// combination the judge cannot score is an error that names it.
    pub fn new(cfg: &EvaluatorConfig) -> Result<FutharkEvaluator, String> {
        let p = problem(&cfg.problem)?;
        let (reference_text, planted_text) = (p.reference()?, p.planted()?);
        let scheme = crate::fitness::scheme_for_problem(&p, cfg.fitness.as_deref())?;
        // The k-means scheme has no quality parts (nothing is compared with an answer): its judge is
        // `evaluate_converged`.
        let parts: &'static [QualityPart] = if scheme.is_converged() { &[] } else { scheme.quality_parts()? };
        let (backend, mode) = (cfg.backend, cfg.eval);
        if scheme.is_converged() && mode == EvalMode::Batch {
            return Err(format!("the fitness scheme {} runs UltraDim's loop: it takes --eval single (one process or server a program) or --eval rust (the resident sample on the device, the rust path's in-process host); the C batch driver evaluates fixed steps only", scheme.name));
        }
        if cfg.pairs == 0 {
            return Err("--pairs 0: stage 2 needs at least one timing slot".into());
        }
        let quality_weight = cfg.quality_weight.unwrap_or_else(|| scheme.default_quality_weight());
        if cfg.kmeans_step_process && (backend != Backend::C || !scheme.is_converged()) {
            return Err("--kmeans-step process is the c backend's k-means path (the GPU path is one process an iteration already)".into());
        }
        if scheme.is_flat() && quality_weight == 0 {
            return Err("--quality-weight 0: the quality inputs would not be entered at all (1 is no repetition)".into());
        }
        if !scheme.is_flat() && cfg.quality_weight.is_some() {
            return Err(format!("--quality-weight: the fitness scheme {} is not a flat scheme and has no quality weight", scheme.name));
        }
        let quality_cliff = cfg.quality_cliff.unwrap_or(crate::fitness::DEFAULT_QUALITY_CLIFF);
        if !scheme.uses_cliff() && cfg.quality_cliff.is_some() {
            return Err(format!("--quality-cliff: the fitness scheme {} has no quality cliff (futhark-kmeans-better has)", scheme.name));
        }
        if !(quality_cliff > 0.0 && quality_cliff.is_finite()) {
            return Err(format!("--quality-cliff {quality_cliff}: the cliff's width must be positive and finite"));
        }
        if mode == EvalMode::Rust && backend != Backend::Gpu {
            return Err("--eval rust runs on the GPU: it takes --backend gpu".into());
        }
        if cfg.ring_walk && !cfg.rings.as_ref().is_some_and(|r| r.canvas) {
            return Err("--mutation ring-walk walks the rings: it needs --start llm-ring --pool DIR".into());
        }
        let out = cfg.out.clone();
        std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;

        let inst = p.instance;
        let k = load(&inst)?;
        let sig = signature(&inst)?;
        let interp = Interp::new(&k, &sig)?;
        let contract = Arc::new(k.relations().check().map_err(|d| format!("the kingdom check: {:?}", &d[..d.len().min(3)]))?);
        let reference = build(&k, reference_text)?;
        let planted = build(&k, planted_text)?;
        // THE PROGRAM SIZE is an experiment parameter (default the problem's `program_size`), never the reference's
        // size. The reference and the planted slowdown must fit the layout.
        let program_size = cfg.program_size.unwrap_or(p.program_size);
        let layout = Arc::new(GeneLayout::from_program_size(&contract, k.roots()[0], program_size).map_err(|e| e.to_string())?);
        for (what, t) in [("reference", &reference), ("planted slowdown", &planted)] {
            crate::typed_gep::fits(&contract, &layout, t).map_err(|e| format!("the {what} ({} functions) does not fit the layout of program size N = {program_size}: {e}", function_count(&k, t)))?;
        }
        let const_len = Arc::new(k.const_len());
        // The messages of the construction, held until the caller sets its sink or takes them.
        let mut messages = Messages::default();
        let (canvas, alternatives) = match &cfg.rings {
            Some(rc) => {
                let (r, progs) = ring_canvas(&k, &contract, &layout, &rc.pool, &cfg.problem, &rc.sources, &out, &mut messages)?;
                // Without the canvas (a measurement on random padding) the pool gives the parents, not the rings.
                (if rc.canvas { Some(Arc::new(r)) } else { None }, progs)
            }
            None => (None, Vec::new()),
        };
        // Every stored case: the fitness's (training and held-out) and the TEST SET (plan section G), run in the same
        // quality program but never a fitness input, so selection never sees it.
        let cases = load_cases(&p)?;
        let sizes: Vec<BTreeMap<String, usize>> = cases.iter().map(|c| sig.sizes(&c.inputs)).collect::<Result<_, _>>()?;
        if let Some(g) = &cfg.gate {
            gate_passed(g, &cfg.problem, program_size, &cases_fingerprint(&cases), scheme.quality_parts()?)?;
            messages.say(&format!("the gate: the slope check in {} passed for these cases and quality inputs", g.display()));
        }
        let all_stdin: Vec<u8> = cases.iter().flat_map(|c| c.stdin.clone()).collect();
        let timing = TimingInput::of(&p, &sig, &cases, &out.join("timing_build"), !scheme.is_converged())?;
        let reference_work = interp.work(&reference, &timing.sizes)?;
        let timing_cap = crate::futhark_sig::timing_work_cap(reference_work);
        messages.say(&format!("timing bound: {timing_cap} node executions at the timing input (the published program's work {reference_work}, the constant {TIMING_WORK_CAP})"));
        // The shim is built once, here (under the cargo lock); no candidate's link runs cargo or takes the lock. The
        // rust path has no shim and no C: its runtime is built once instead, into the run's own directory.
        let with_shim = backend == Backend::Gpu && mode != EvalMode::Rust;
        let shim = if with_shim { prebuild_shim(&out.join("shim"))? } else { out.join("shim") };
        let yard_exe = if with_shim && !scheme.is_converged() { Some(compile_program_prebuilt(p.yardstick, &out.join("yardstick_build"), &shim)?) } else { None };
        // The other side of every pair: the published program, or for k-means THE REFERENCE PROGRAM (the yardstick
        // of its fitness), whose range entry every generation's module carries.
        let reference_source = program(&k, &sig, &reference)?;
        let host = match mode {
            EvalMode::Rust => {
                let rt = crate::futhark_rust::prebuild_rt(&out.join("rust_rt"))?;
                let b = crate::futhark_batch::Backend::Rust { rt };
                // A batch whose compile does not finish within the host's limit has its source saved under the run's
                // bugs/ (`futhark_batch::COMPILE_LIMIT_FLOOR`; the split is logged in bugs.log as any halving).
                Some(Host::new(&k, &sig, cases.len(), b, &out.join("rust_build"), Inputs { cases_stdin: &all_stdin, timing_stdin: Some(timing.stdin.as_slice()), yardstick: Some(if scheme.is_converged() { reference_source.as_str() } else { p.yardstick }) })?.with_bugs_dir(&out.join("bugs")))
            }
            EvalMode::Batch => {
                let b = if backend == Backend::Gpu { crate::futhark_batch::Backend::Gpu { shim: shim.clone() } } else { crate::futhark_batch::Backend::C };
                let timing_stdin = if backend == Backend::Gpu { Some(timing.stdin.as_slice()) } else { None };
                Some(Host::new(&k, &sig, cases.len(), b, &out.join("batch_build"), Inputs { cases_stdin: &all_stdin, timing_stdin, yardstick: Some(p.yardstick) })?.with_bugs_dir(&out.join("bugs")))
            }
            EvalMode::Single => None,
        };
        let timer = if mode == EvalMode::Single && backend == Backend::Gpu && !scheme.is_converged() {
            Some(Host::new(&k, &sig, cases.len(), crate::futhark_batch::Backend::Gpu { shim: shim.clone() }, &out.join("timing_build"), Inputs { cases_stdin: &all_stdin, timing_stdin: Some(timing.stdin.as_slice()), yardstick: Some(p.yardstick) })?)
        } else {
            None
        };
        let gen = Genetics { k: Arc::clone(&k), contract, layout, const_len, canvas, ring_walk: cfg.ring_walk };
        let judge = Judge {
            k,
            sig,
            all_stdin,
            interp,
            cases,
            scheme,
            parts,
            sizes,
            timing,
            timing_cap,
            over_cap_untimed: 0,
            backend,
            runs: cfg.runs,
            yard_exe,
            shim,
            work_dir: out.join("candidate_build"),
            quality_cache: HashMap::new(),
            slots: TimingSlots::new(cfg.pairs),
            evaluations: 0,
            scored: 0,
            compiled: 0,
            hits: 0,
            bugs: 0,
            setmem: 0,
            ice: 0,
            bug_log: out.join("bugs.log"),
            max_work: 0,
            mode,
            host,
            timer,
            batched_q: HashMap::new(),
            batched_t: HashMap::new(),
            gen_stages: Stages::default(),
            batch_splits: 0,
            last: ("hit", 0, Stages::default()),
            converged: None,
            converged_speed: HashMap::new(),
            over_cap_unrun: 0,
            unfinished: 0,
            quality_weight,
            quality_cliff,
            step_process: cfg.kmeans_step_process,
            batched_k: HashMap::new(),
            upload_seconds: 0.0,
            resident_hits: 0,
            messages,
            modules: 0,
            failed_compiles: 0,
            unfinished_compiles: 0,
        };
        Ok(FutharkEvaluator { problem: p, program_size, reference, planted, reference_source, alternatives, out, controls_ran: false, stage: FitnessStage::Quality, judge, gen })
    }

    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    pub fn instance(&self) -> &Instance {
        &self.problem.instance
    }

    pub fn scheme(&self) -> &'static Scheme {
        self.judge.scheme
    }

    pub fn kingdom(&self) -> &Arc<FutharkKingdom> {
        &self.gen.k
    }

    pub fn signature(&self) -> &Arc<Signature> {
        &self.judge.sig
    }

    pub fn contract(&self) -> &KingdomContract {
        &self.gen.contract
    }

    pub fn layout(&self) -> &GeneLayout {
        &self.gen.layout
    }

    pub fn const_len(&self) -> &BTreeMap<Ty, u32> {
        &self.gen.const_len
    }

    pub fn program_size(&self) -> usize {
        self.program_size
    }

    /// The repetition count of the quality inputs (a flat scheme's; the scheme's default when not configured).
    pub fn quality_weight(&self) -> usize {
        self.judge.quality_weight
    }

    pub fn quality_cliff(&self) -> f64 {
        self.judge.quality_cliff
    }

    /// The timing slots of a program (or the interleaved pairs of a k-means candidate).
    pub fn pairs(&self) -> usize {
        self.judge.slots.slots()
    }

    /// The published program written as a gene of the kingdom (the reference program).
    pub fn reference(&self) -> &Tree {
        &self.reference
    }

    /// The planted slowdown (a control of regime P).
    pub fn planted(&self) -> &Tree {
        &self.planted
    }

    /// The reference program's Futhark source (the declared signature).
    pub fn reference_source(&self) -> &str {
        &self.reference_source
    }

    /// The pool's task alternatives that fit the layout (empty without a ring configuration).
    pub fn alternatives(&self) -> &[Tree] {
        &self.alternatives
    }

    /// THE NAMES OF THE HFF INPUT VECTOR'S COLUMNS, in its order (the stream's `run_start`, the hall of fame's
    /// columns): for the flat k-means schemes every fitness case's quality inputs, the whole group `quality_weight`
    /// times, then every case's speed inputs once; for the gated k-means scheme quality then speed; for the retired
    /// five-input scheme its five a case; for the answer-scored schemes the quality inputs a fitness case, then the
    /// speed pairs (stage 2).
    /// The caller's sink for the evaluator's messages ([`Messages`]); the lines held so far go to it first.
    pub fn set_messages(&mut self, sink: Sink) {
        self.judge.messages.set_sink(sink);
    }

    /// The messages held so far (none once a sink is set), in order.
    pub fn take_messages(&mut self) -> Vec<String> {
        self.judge.messages.take()
    }

    pub fn input_names(&self) -> Vec<String> {
        let (scheme, cases, quality_weight) = (self.judge.scheme, &self.judge.cases, self.judge.quality_weight);
        let fit = || cases.iter().filter(|c| c.split.in_fitness());
        let mut names: Vec<String> = if scheme.definition == Definition::FutharkConvergedQqs || scheme.definition == Definition::FutharkKmeansBetter {
            let qnames: &[&str] = if scheme.uses_cliff() { &QUALITY_NAMES_BETTER } else { &QUALITY_NAMES };
            let quality: Vec<String> = fit().flat_map(|c| qnames.iter().map(|m| format!("quality {} {m}", c.name)).collect::<Vec<_>>()).collect();
            (0..quality_weight).flat_map(|i| quality.iter().map(move |q| if quality_weight > 1 { format!("{q} (copy {} of {quality_weight})", i + 1) } else { q.clone() })).chain(fit().flat_map(|c| SPEED_NAMES.iter().map(|m| format!("speed {} {m}", c.name)).collect::<Vec<_>>())).collect()
        } else if scheme.definition == Definition::FutharkConvergedGated {
            fit().flat_map(|c| QUALITY_NAMES.iter().map(|m| format!("quality {} {m}", c.name)).collect::<Vec<_>>()).chain(fit().flat_map(|c| SPEED_NAMES.iter().map(|m| format!("speed {} {m}", c.name)).collect::<Vec<_>>())).collect()
        } else if scheme.is_converged() {
            fit().flat_map(|c| INPUT_NAMES_FIVE.iter().map(|m| format!("quality {} {m}", c.name)).collect::<Vec<_>>()).collect()
        } else {
            fit().flat_map(|c| c.output.as_ref().map_or_else(Vec::new, |o| quality_input_names_of(self.judge.parts, o)).into_iter().map(|m| format!("quality {} {m}", c.name)).collect::<Vec<_>>()).collect()
        };
        if !scheme.is_converged() {
            for i in 0..self.judge.slots.slots() {
                names.push(format!("speed pair {i} r/(1+r) (stage 2)"));
            }
        }
        names
    }

    /// THE INTERPRETED CONTROL: the reference gene is exact on every case under the host interpreter, so the kingdom
    /// expresses the published program. The k-means scheme's reference program is checked compiled instead
    /// ([`FutharkEvaluator::controls`]): the host interpreter is not run on cases of that size.
    pub fn control_interpreted(&mut self) -> Result<(), String> {
        if self.judge.scheme.is_converged() {
            // The host interpreter is not run on the dbpedia cases (about 4e7 node executions a second: one step of
            // 1,000 x 1,536 x 16 is a minute); its agreement with the compiled program is the kingdom's tests' on the
            // small case (`tests_kmeans_seed`).
            self.judge.messages.say("control: the seed's run is checked compiled (the host interpreter is not run on cases of this size)");
            return Ok(());
        }
        for c in &self.judge.cases {
            if self.judge.interp.eval(&self.reference, &c.inputs)? != *c.answer()? {
                return Err(format!("our software has a bug: the reference gene is not exact on {}", c.name));
            }
        }
        self.judge.messages.say(&format!("control: the reference gene is exact on all {} cases (host interpreter)", self.judge.cases.len()));
        Ok(())
    }

    /// THE CONTROLS AT THE START (plan 5a.2), in order: the interpreted control; on the GPU for a run, the A/A sets
    /// (the published program against itself, 11 interleaved pairs; in P the in-run A/A pairs; `aa_rounds` single
    /// timings) and the yardstick's GPU output on the timing input against `futhark c`'s; then the compiled control:
    /// for the k-means scheme the reference program's run converges on every fitness case (its runs and the work caps
    /// kept in the judge; on the resident-sample path every case checked as a row range of the sample), else the
    /// compiled reference program is exact on every case; in P the in-run A/A enters the timing slots; for the
    /// k-means scheme in a run, the reference program judged as a candidate against itself (a budgeted evaluation).
    /// ONCE A RUN (the F2 code review, note 2): a second call would add the in-run A/A to the timing slots again and
    /// judge the k-means A/A again as a budgeted evaluation, so it is an error that says so.
    pub fn controls(&mut self, req: &ControlRequest) -> Result<Controls, String> {
        if self.controls_ran {
            return Err("our software has a bug: the controls were asked for a second time; they run once a run (a second call would add the in-run A/A to the timing slots again and judge the k-means A/A again)".into());
        }
        self.controls_ran = true;
        // Regime P starts from the published program, exact: it is scored on quality and speed from the start.
        if req.regime == Some(Regime::P) {
            self.stage = FitnessStage::QualityAndSpeed;
        }
        self.control_interpreted()?;
        let mut c = Controls::default();
        let converged = self.judge.scheme.is_converged();
        let (backend, mode) = (self.judge.backend, self.judge.mode);
        let timed = req.regime.is_some();
        let p_regime = req.regime == Some(Regime::P);
        let pairs = self.judge.slots.slots();
        if backend == Backend::Gpu && timed && !converged {
            // Every path times its pairs with the driver's interleaved pairs (`Step::Pair`): the batch process, this
            // process (rust), or the single path's own timing process.
            let a = self.judge.aa_batch(self.problem.yardstick, if p_regime { pairs } else { 0 }, req.aa_rounds)?;
            c.noise = a.noise.clone();
            let noise = &c.noise;
            let msg = format!("A/A noise floor{}, the published program against itself, 11 interleaved pairs of at least {PAIR_SIDE_SECONDS} s a side on {}: ratio median {:.4}, min {:.4}, max {:.4}", if mode.batched() { " (in one batch process)" } else { " (in one timing process)" }, self.judge.timing.label(), median(noise), noise.iter().cloned().fold(f64::INFINITY, f64::min), noise.iter().cloned().fold(0.0, f64::max));
            self.judge.messages.say(&msg);
            c.notes.push(msg);
            c.aa = Some(a);
        }
        if let (Some(want), Some(y), true) = (&self.judge.timing.output, &self.judge.yard_exe, timed) {
            // Control: the yardstick's GPU output on the timing input is the stored one (futhark c).
            let (got, _) = run_program_bytes(y, &self.judge.timing.stdin, false, 1)?;
            if read_binary(&got)? != read_binary(want)? {
                return Err("our software has a bug: the yardstick on the GPU does not give futhark c's output on the timing input".into());
            }
            self.judge.messages.say(&format!("control: the yardstick on the GPU gives futhark c's output on {} (GPU)", self.judge.timing.label()));
        }
        if converged {
            self.control_converged()?;
        } else {
            let (reference, source) = (self.reference.clone(), self.reference_source.clone());
            self.judge.prefetch(&[(&reference, source.as_str())], FitnessStage::Quality)?;
            let ref_q = self.judge.quality(&reference, &source)?;
            if !ref_q.perfect() {
                return Err(format!("our software has a bug: the compiled reference is not exact ({:?})", ref_q.raw));
            }
            self.judge.messages.say(&format!("control: the compiled reference gene is exact on every case ({backend:?}); work {} node executions at most", ref_q.work));
        }
        // The in-run A/A (P): the published program, compiled as a candidate, through the timing slots. The converged
        // scheme has no timing slots: its A/A is the reference program judged as a candidate against itself (below).
        if p_regime && backend == Backend::Gpu && !converged {
            let r = c.aa.as_ref().map(|a| a.inrun.clone()).ok_or("our software has a bug: the in-run A/A with no A/A sets")?;
            self.judge.slots.add(self.problem.yardstick, &r)?;
            c.inrun_aa = self.judge.slots.full(self.problem.yardstick)?.to_vec();
            let inrun = &c.inrun_aa;
            let msg = format!("in-run A/A{}, the published program as a candidate through the timing slots, {pairs} pairs: ratio median {:.4}, min {:.4}, max {:.4}", if mode.batched() { " (in the batch process)" } else { " (in the timing process)" }, median(inrun), inrun.iter().cloned().fold(f64::INFINITY, f64::min), inrun.iter().cloned().fold(0.0, f64::max));
            self.judge.messages.say(&msg);
            c.notes.push(msg);
        }
        // THE K-MEANS SCHEME'S A/A, on both backends (the round-2 review's N3): the reference program judged as a
        // candidate against itself through the same interleaving every candidate gets. Equal runs give 1/2 on every
        // quality input (a quality input off 1/2 is a non-deterministic reduction, written down here); the spread of
        // the seconds and iterations inputs around 1/2 is THE NOISE FLOOR of this scheme.
        if converged && timed {
            let (reference, source) = (self.reference.clone(), self.reference_source.clone());
            let t_aa = Instant::now();
            self.judge.prefetch(&[(&reference, source.as_str())], FitnessStage::Quality)?;
            let e = self.judge.evaluate(&reference, &source, FitnessStage::Quality)?;
            let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(", ");
            let worst = |v: &[f64]| v.iter().map(|x| (x - 0.5).abs()).fold(0.0, f64::max);
            let msg = format!("in-run A/A (k-means): the seed as a candidate against itself through {pairs} interleaved pair(s) on every fitness case, {:.1} s: fitness {:.4}; quality inputs [{}], worst distance from 1/2 {:.2e}{}; speed inputs [{}], worst distance from 1/2 {:.4} (the noise floor of a speed claim)", t_aa.elapsed().as_secs_f64(), e.fitness, fmt(&e.quality.inputs), worst(&e.quality.inputs), if worst(&e.quality.inputs) > 0.0 { " (A QUALITY INPUT OFF 1/2: the program's reduction is not deterministic on this backend)" } else { "" }, fmt(&e.speed), worst(&e.speed));
            self.judge.messages.say(&msg);
            c.notes.push(msg);
            c.kmeans_aa_quality = e.quality.inputs.clone();
            c.kmeans_aa_speed = e.speed.clone();
        }
        Ok(c)
    }

    /// THE SEAM (plan 5a.2, task F3a): every row the engine hands over, in row order, scored, with no output of its
    /// own ([`FutharkEvaluator::evaluate_generation_with`] reports each batch). A call is a call: the engine calls it
    /// after variation and again after a pump or fold beat, and nothing here assumes one call a generation.
    pub fn evaluate_generation(&mut self, genes: &[&[u32]], generation: u32) -> Result<GenerationScores, String> {
        self.evaluate_generation_with(genes, generation, &mut |_| {})
    }

    /// [`FutharkEvaluator::evaluate_generation`], with `after_batch` called after every batch (the caller's progress
    /// line). The answer-scored problems (reduce, scan, histogram, pathfinder, a7); k-means is task F3b.
    ///
    /// Each distinct gene is expressed from its head cells and rendered once (a gene that does not express is a bug
    /// of our software and an error). Every program not scored before, and in stage 2 every program with an empty
    /// timing slot, is evaluated in ONE compiled module a call on `--eval batch` and `--eval rust` (`Judge::prefetch`),
    /// or its own compile on `--eval single`; a program scored before, in this call or an earlier one, is a cache hit
    /// and costs no evaluation. THE RUN FILE (`OUT/killme.trigger`) is checked before every batch: when it is gone the
    /// call returns the rows scored so far with `stopped`. THE STAGE: when the best row of stage 1 (the least angle,
    /// the first on a tie, as the deprecated runner sorts) is exact on every case, the stage becomes 2 and every row
    /// is scored again in stage 2 within the call (a second module when its programs are timed), and the rows
    /// returned are the stage-2 scores; a run file gone before that second batch returns the stage-1 rows, the stage
    /// unchanged.
    pub fn evaluate_generation_with(&mut self, genes: &[&[u32]], generation: u32, after_batch: &mut dyn FnMut(&BatchDone)) -> Result<GenerationScores, String> {
        if self.judge.scheme.is_converged() {
            return Err(format!("evaluate_generation for the k-means scheme {} is task F3b (plan 5a.3: the resident sample's prefetch_kmeans and evaluate_converged, the champion's test case), not built yet", self.judge.scheme.name));
        }
        let t0 = Instant::now();
        let before = (self.judge.evaluations, self.judge.gen_stages, self.judge.modules, self.judge.failed_compiles, self.judge.unfinished_compiles, self.judge.batch_splits, self.judge.upload_seconds, self.judge.resident_hits);
        let mut index: HashMap<&[u32], usize> = HashMap::new();
        let mut distinct: Vec<Rendered> = Vec::new();
        let mut row_of: Vec<usize> = Vec::with_capacity(genes.len());
        for (r, g) in genes.iter().enumerate() {
            let i = match index.get(g) {
                Some(i) => *i,
                None => {
                    let tree = self.gen.tree(g).map_err(|e| format!("our software has a bug: row {r} of generation {generation}: {e}"))?;
                    let source = program(&self.gen.k, &self.judge.sig, &tree)?;
                    let text = sexp(&self.gen.k, &tree);
                    distinct.push(Rendered { tree, text, source });
                    index.insert(g, distinct.len() - 1);
                    distinct.len() - 1
                }
            };
            row_of.push(i);
        }
        let mut batches = 0usize;
        let (mut rows, mut stopped) = self.score_rows(&distinct, &row_of, self.stage, &mut batches, after_batch)?;
        let mut stage_changed = false;
        if !stopped {
            let best = (0..rows.len()).min_by(|a, b| rows[*a].fitness.total_cmp(&rows[*b].fitness).then(a.cmp(b)));
            let next = self.stage.next(best.is_some_and(|b| rows[b].perfect));
            if next != self.stage {
                self.judge.messages.say(&format!("stage 1 -> 2 at generation {generation}, evaluation {}: the best is exact on every case (training and held-out); every row is scored again in stage 2 within this call", self.judge.evaluations));
                let (again, s2) = self.score_rows(&distinct, &row_of, next, &mut batches, after_batch)?;
                if s2 {
                    stopped = true;
                } else {
                    self.stage = next;
                    rows = again;
                    stage_changed = true;
                }
            }
        }
        let (e0, st0, m0, f0, u0, h0, up0, rh0) = before;
        let (a, b) = (self.judge.gen_stages.values(), st0.values());
        let stages = Stages::of(std::array::from_fn(|i| a[i] - b[i]));
        let cost = GenerationCost {
            stages,
            compile_seconds: stages.futhark + stages.port + stages.cc + stages.check + stages.link,
            run_seconds: stages.start + stages.device + stages.pipelines + stages.runs,
            timing_seconds: stages.timing,
            upload_seconds: self.judge.upload_seconds - up0,
            modules: self.judge.modules - m0,
            failed_compiles: self.judge.failed_compiles - f0,
            unfinished_compiles: self.judge.unfinished_compiles - u0,
            halvings: self.judge.batch_splits - h0,
            resident_hits: self.judge.resident_hits - rh0,
            seconds: t0.elapsed().as_secs_f64(),
        };
        Ok(GenerationScores { rows, stage: self.stage, stage_changed, stopped, evaluations_new: self.judge.evaluations - e0, evaluations_total: self.judge.evaluations, cost })
    }

    /// Whether a program needs a batch in `stage`: its quality not known, or in stage 2 an empty timing slot.
    fn needs_batch(&self, source: &str, stage: FitnessStage) -> bool {
        !self.judge.quality_cache.contains_key(source) || (stage.measures_speed() && self.judge.slots.missing(source) > 0)
    }

    /// Every row scored in `stage`, in row order; the run file checked before every batch. Returns the rows and
    /// whether the run file was gone (the rows then those scored before it went).
    fn score_rows(&mut self, distinct: &[Rendered], row_of: &[usize], stage: FitnessStage, batches: &mut usize, after_batch: &mut dyn FnMut(&BatchDone)) -> Result<(Vec<RowScore>, bool), String> {
        let run_file = self.out.join(crate::run_file::RUN_FILE);
        let batched = self.judge.mode.batched();
        if batched && row_of.iter().any(|i| self.needs_batch(&distinct[*i].source, stage)) {
            if !run_file.exists() {
                return Ok((Vec::new(), true));
            }
            let compiles = self.judge.modules + self.judge.failed_compiles;
            let items: Vec<(&Tree, &str)> = row_of.iter().map(|i| (&distinct[*i].tree, distinct[*i].source.as_str())).collect();
            self.judge.prefetch(&items, stage)?;
            if self.judge.modules + self.judge.failed_compiles > compiles {
                *batches += 1;
                after_batch(&BatchDone { batch: *batches, stage, rows_scored: 0 });
            }
        }
        let mut rows = Vec::with_capacity(row_of.len());
        for i in row_of {
            let r = &distinct[*i];
            let own_batch = !batched && self.needs_batch(&r.source, stage);
            if own_batch && !run_file.exists() {
                return Ok((rows, true));
            }
            let evaluations = self.judge.evaluations;
            let e = self.judge.evaluate(&r.tree, &r.source, stage)?;
            rows.push(self.row_score(r, &e, stage, self.judge.evaluations > evaluations)?);
            if own_batch {
                *batches += 1;
                after_batch(&BatchDone { batch: *batches, stage, rows_scored: rows.len() });
            }
        }
        Ok((rows, false))
    }

    /// A scored row: the HFF input vector the scheme's angle is taken over, the angle's percentile, the ledger's
    /// columns from the judge's last evaluation.
    fn row_score(&self, r: &Rendered, e: &Eval, stage: FitnessStage, new_evaluation: bool) -> Result<RowScore, String> {
        let q = &e.quality;
        let (inputs, one_vector) = match stage {
            FitnessStage::Quality => (q.inputs.clone(), true),
            FitnessStage::QualityAndSpeed => match self.judge.scheme.stage_two()? {
                crate::fitness::StageTwo::Flat => (crate::fitness::flat_vector(&q.inputs, &e.speed, self.judge.quality_weight), true),
                crate::fitness::StageTwo::Ungated | crate::fitness::StageTwo::Gated => (crate::fitness::flat_vector(&q.inputs, &e.speed, 1), false),
            },
        };
        let d = inputs.len();
        let percentile = if one_vector { Some(crate::fitness::hff_percentile(e.fitness, d)?) } else { None };
        Ok(RowScore {
            text: r.text.clone(),
            source: r.source.clone(),
            inputs,
            fitness: e.fitness,
            theta_quality: e.theta_q,
            theta_speed: e.theta_s,
            percentile,
            d,
            quality: q.inputs.clone(),
            speed: e.speed.clone(),
            ratio: e.ratio,
            exact: [q.exact_train, q.exact_held, q.exact_test],
            perfect: q.perfect(),
            behaviour: q.behaviour.clone(),
            all_zero: q.all_zero(),
            new_evaluation,
            ledger: ledger_columns(&self.judge.last, e),
        })
    }

    /// The fitness stage of the next call.
    pub fn stage(&self) -> FitnessStage {
        self.stage
    }

    /// THE K-MEANS SCHEME's yardstick: the reference program compiled once (the run's own directory), run to the
    /// stopping point on every fitness case with the program's own time read as every candidate's will be; a run that
    /// does not CONVERGE is our bug. Each candidate is then run interleaved with it, and scored against that run. The
    /// work cap of every fitness case is the reference program's work on it times `WORK_CAP_TIMES_SEED`.
    fn control_converged(&mut self) -> Result<(), String> {
        let (backend, mode, step_process) = (self.judge.backend, self.judge.mode, self.judge.step_process);
        let pairs = self.judge.slots.slots();
        let points_dir = self.out.join("candidate_build").join("points");
        let fit: Vec<usize> = (0..self.judge.cases.len()).filter(|i| self.judge.cases[*i].split.in_fitness()).collect();
        let t_control = Instant::now();
        let (seed_stepper, seed_runs, cuts) = if mode == EvalMode::Rust {
            // THE RESIDENT-SAMPLE PATH: every stored case is a row range of the sample (the timing input's first
            // value), checked here against the case's points; the control is the reference program as a member
            // against its own entry, one pair, on every fitness case.
            let sample = read_binary(&self.judge.timing.stdin)?.into_iter().next().ok_or("the timing input holds no sample")?;
            let mut cuts = Vec::with_capacity(self.judge.cases.len());
            for c in &self.judge.cases {
                let cut = KMEANS_CASES.iter().chain(std::iter::once(&crate::futhark_sig::kmeans_converged::KMEANS_TIMING_CUT)).find(|x| x.name == c.name).ok_or_else(|| format!("the case {} is not a declared row range of the sample (kmeans_converged::KMEANS_CASES)", c.name))?;
                let d = sample.dims[1];
                if c.inputs[0].dims != [cut.end - cut.start, d] || c.inputs[0].data != sample.data[cut.start * d..cut.end * d] {
                    return Err(format!("our software has a bug: the case {} is not the sample's rows {}..{}", c.name, cut.start, cut.end));
                }
                cuts.push((cut.start, cut.end - cut.start));
            }
            self.judge.converged = Some(ConvergedJudge { seed: None, seed_runs: Vec::new(), work_caps: Vec::new(), pairs, points_dir: points_dir.clone(), cuts: Some(cuts.clone()) });
            let source = self.reference_source.clone();
            let (runs, c) = self.judge.kmeans_batch(&[Program::Text(&source)], &fit, 1)?;
            let runs = runs.into_iter().next().ok_or("our software has a bug: the seed's control batch of no member")??;
            self.judge.messages.say(&format!("control: the seed's module compiled in {:.1} s, the sample uploaded in {:.3} s ({} MB), the seed's {} runs in {:.1} s", c.compile.futhark + c.compile.cc, c.upload_seconds, sample.data.len() * 4 / 1_000_000, 2 * fit.len(), c.run.total));
            (None, runs.into_iter().map(|(_, seed)| seed).collect::<Vec<_>>(), Some(cuts))
        } else {
            let (mut seed_stepper, _) = self.judge.compile_converged(&self.reference_source, &self.out.join("seed_build"))?;
            let mut seed_runs = Vec::new();
            for i in &fit {
                let c = &self.judge.cases[*i];
                seed_runs.push(converge(&c.inputs[0], &c.inputs[1], seed_stepper.step(c)?)?);
            }
            (Some(seed_stepper), seed_runs, None)
        };
        let mut work_caps = Vec::new();
        for (i, r) in fit.iter().zip(&seed_runs) {
            let c = &self.judge.cases[*i];
            if !r.converged {
                return Err(format!("our software has a bug: the compiled seed's run does not converge on {} ({backend:?}, {mode:?}): {} iterations, finished {}: {}", c.name, r.iterations, r.finished, r.failure));
            }
            let seed_work = self.judge.interp.work(&self.reference, &self.judge.sizes[*i])?;
            self.judge.messages.say(&format!("control: the compiled seed on {}: {} iterations to the stopping point, the program's own {:.3} s ({:.3} s wall), cosine inertia {:.5}, mean log10 p {:.2} ({:.2} over every point, {} at their centroid); work {seed_work}, the cap {}", c.name, r.iterations, r.seconds, r.wall_seconds, r.inertia, r.mean_log10_p, r.mean_log10_p_all, r.coincident, converged_work_cap(seed_work)));
            work_caps.push(converged_work_cap(seed_work));
        }
        self.judge.messages.say(&format!("control: the seed's {} runs took {:.1} s ({})", seed_runs.len(), t_control.elapsed().as_secs_f64(), match (backend, step_process, mode) { (_, _, EvalMode::Rust) => "the resident sample on the device, the in-process host", (Backend::C, false, _) => "one Futhark server, the points loaded once a case", (Backend::C, true, _) => "one process of the futhark c executable an iteration (--kmeans-step process)", (Backend::Gpu, _, _) => "one process of the shim's executable an iteration" }));
        self.judge.converged = Some(ConvergedJudge { seed: seed_stepper, seed_runs, work_caps, pairs, points_dir, cuts });
        Ok(())
    }
}

#[cfg(test)]
mod tests;
