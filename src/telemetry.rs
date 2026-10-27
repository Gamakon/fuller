//! THE TELEMETRY STREAM — the one format every evolution run in fuller and phylu
//! writes to `OUT/stream.jsonl`, and the one the watchers read
//! (`evowatch OUT` in a terminal, `evowatch OUT --web PORT` in a browser).
//!
//! It lives in fuller because fuller and phylu are one system: phylu depends on
//! fuller, so a runner in either crate writes the stream through these types,
//! and phylu's `evolve::telemetry` re-exports them rather than keeping a copy.
//! A runner with a generation loop creates a [`RunTelemetry`] to run at all
//! (`docs/STANDARD_runners.md`): the stream, the kill trigger and the run's
//! end are then the same for every kingdom.
//!
//! The engine's human log is a scrolling prose report: it prints the state of
//! the search as sentences, three times over, and an operator watching a
//! 450-second fit cannot read it. That log stays exactly as it is — it is what
//! an offline diagnosis reads. This module adds a SECOND, separate stream: one
//! compact JSON object per line, versioned, numeric, written at the beat the
//! progress report already runs on. `evowatch` repaints from it.
//!
//! Three rules shape every decision here, and they are the reason the module
//! looks the way it does.
//!
//! **It rides along.** Nothing in here scans the population. A snapshot is built
//! from what `Engine::report` has ALREADY read — the generation's fitness vector,
//! the cohort labels the cohort report read back, the `Scored` best. A new pass
//! over 200,000 rows to fill a screen would be the telemetry paying for itself
//! out of the fit's budget, which is exactly the trade the brief forbids.
//!
//! **Off is off.** With no path configured the engine allocates nothing, reads
//! no clock and opens no file: `Config::telemetry_path = None` leaves the fit
//! bit-for-bit what it was, and a test asserts it (`the_telemetry_changes_no_bit_
//! of_the_population`).
//!
//! **Numbers are numbers.** A metric that is not defined is JSON `null`, never
//! the string `"—"` — the dash belongs to the viewer, which knows the terminal
//! it is drawing into. `f64::NAN` and the infinities serialise as `null` through
//! serde_json, which is the behaviour we want: a cohort whose every row failed to
//! score has no best HFF, and `null` says so. Counts of those rows are carried
//! EXPLICITLY (`nan_rows`), because a JSON number cannot be NaN and a reader must
//! not have to infer how many rows went missing.
//!
//! This module is deliberately free of engine types and of the `gpu` feature: it
//! is schema, a writer and a reader. The engine converts into it; the viewer
//! converts out of it; neither needs the other to build.

use std::collections::BTreeMap;
use std::io::Write;

use serde::{Deserialize, Serialize};

/// The stream's version. A reader that finds a number it does not know must say
/// so rather than guess at the fields: bumping this is how a breaking change to
/// the record shape is announced.
pub const SCHEMA_VERSION: u32 = 1;

/// One line of the stream. `type` is the tag, so a reader matches on it before
/// it looks at anything else, and an unknown tag is skipped rather than fatal —
/// which is what lets a later engine add a record kind without breaking a viewer
/// that is already deployed.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Record {
    RunStart(RunStart),
    Snapshot(Snapshot),
    Event(Event),
    Model(Model),
    /// The hall of fame as it stands, written when it changes (a WGSL fit:
    /// `true_wgsl_fit`). An older viewer counts it as a bad line and goes on.
    HallOfFame(HallOfFame),
    RunEnd(RunEnd),
}

impl Record {
    /// The fields every record carries, whatever its kind — so a reader can
    /// check monotonicity and staleness without a match on the tag.
    pub fn header(&self) -> &Header {
        match self {
            Record::RunStart(r) => &r.header,
            Record::Snapshot(r) => &r.header,
            Record::Event(r) => &r.header,
            Record::Model(r) => &r.header,
            Record::HallOfFame(r) => &r.header,
            Record::RunEnd(r) => &r.header,
        }
    }
}

/// What is on EVERY record: which stream it belongs to, where it sits in that
/// stream, and when it was written.
///
/// `seq` is the stream's own counter and increases by one per record with no
/// gaps, so a reader that sees a jump knows a line was lost (a rotation, a
/// truncated tail) rather than guessing from timestamps. `generation` is the
/// fit's clock and may repeat — several events can belong to one generation —
/// which is why the two are separate numbers and not one.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Header {
    pub schema_version: u32,
    pub run_id: String,
    pub seq: u64,
    /// RFC-3339, UTC, seconds — written by the engine so a recorded file can be
    /// read months later and still say when it ran.
    pub timestamp_utc: String,
    pub elapsed_ms: u64,
    pub generation: u32,
}

/// The first line of a stream: everything about the run that does not change, so
/// a viewer attaching at any point has the frame's fixed furniture without
/// waiting for a snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunStart {
    #[serde(flatten)]
    pub header: Header,
    #[serde(default)]
    pub dataset: String,
    #[serde(default)]
    pub seed: u32,
    /// The whole population, `layout.pop` — the number the invariant checks sum
    /// to. It is NOT `pop_intake + pop_champion`: the float zone's rows are real
    /// intake rows and are in it.
    #[serde(default)]
    pub population: u32,
    #[serde(default)]
    pub n_pairs: u32,
    #[serde(default)]
    pub pop_intake: u32,
    #[serde(default)]
    pub pop_champion: u32,
    #[serde(default)]
    pub float_zone: u32,
    /// Rows in the train, validation and third (edge / SMOGD) blocks.
    #[serde(default)]
    pub n_train: usize,
    #[serde(default)]
    pub n_val: usize,
    #[serde(default)]
    pub n_extrap: usize,
    /// The time cap in milliseconds; the budget bar's denominator.
    #[serde(default)]
    pub budget_ms: u64,
    #[serde(default)]
    pub max_generations: u32,
    /// `cohort_merge`: 0 means VIRTUAL ALPS is off and there are no cohorts to
    /// table. The viewer says so rather than drawing an empty table.
    #[serde(default)]
    pub cohort_merge: u32,
    #[serde(default)]
    pub pump_every: u32,
    #[serde(default)]
    pub progress_every: u32,
    /// THE STOP BAR'S p HALF, as the engine holds it (`Config::stop_log10_p`):
    /// the confirmed model's HFF angle as a log10 p-value must be at most this.
    ///
    /// It is on the wire because A VIEWER MUST NOT OWN A THRESHOLD THE ENGINE
    /// OWNS. The default moves as the measurement improves and a run can
    /// override it, so a screen carrying its own -19 would colour one run's
    /// p-value against another run's bar the day either changes.
    ///
    /// None is the honest silence of a stream written before this field existed,
    /// or of a run with the half switched off (`f64::INFINITY`, which is not a
    /// bar any p can clear). The viewer draws the value in ordinary ink and says
    /// the bar is not there, rather than supplying one.
    #[serde(default)]
    pub stop_log10_p: Option<f64>,
    /// THE STOP BAR'S OTHER HALF (`Config::stop_one_minus_r2`): validation — and
    /// edge, when there is one — 1 - R² must be at most this. Carried beside the
    /// p half because BOTH must pass for the fit to call a model a law, and a
    /// stream that published only one of them would invite a reader to conclude
    /// from half a bar.
    #[serde(default)]
    pub stop_one_minus_r2: Option<f64>,
    /// THE WHOLE `Config`, as it was serialised into the run card, carried in
    /// the stream so the viewer can show every setting — not only the dozen
    /// fields above, but every editor switch (fold, snap, beam, typed_depth,
    /// tower, redundancy, …). It is a JSON string rather than the typed struct
    /// so telemetry does not depend on the engine's `Config` type and cannot
    /// drift from it: the engine serialises the same value the card does, once.
    /// `None` on a stream written before this field existed; the viewer then
    /// shows the header fields it does have and says the card is unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_json: Option<String>,
    /// THE KINGDOM this run draws from (`Config::kingdom`): `"Regex"` for a regex
    /// fit, absent/empty for symbolic regression. The viewer reads it to label
    /// the run as a regex run (and to read `Global::fn_mass`/`fp_mass`/`strokes`
    /// instead of the SR `mse`/`r2`/`p`). `None` on a stream written before this
    /// field existed — the viewer then treats it as SR, as it always did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kingdom: Option<String>,
    /// THE HFF OBJECTIVES BY NAME, in column order — what each dimension HFF ranks
    /// on actually measures. The run card's HFF panel shows these so a user sees
    /// WHAT the search is optimising, for both SR and regex. Empty on a stream
    /// written before this field existed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hff_objectives: Vec<String>,
    /// THE WGSL RUN'S FIXED FACTS (`true_wgsl_fit`): the interface, the problem,
    /// the gene's words, the program size and the HFF input count. `None` on
    /// every other kingdom's stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wgsl: Option<WgslStart>,
}

/// What a WGSL evolution run declares once, at its start.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct WgslStart {
    /// The kingdom instance the genes are drawn from (its table directory).
    pub interface: String,
    /// The shader or problem the scorer measures against.
    pub problem: String,
    /// Words of one gene row (the layout's width).
    pub gene_words: u32,
    /// The layout's program size `N`.
    pub program_size: u32,
    /// The number of HFF inputs a kernel is scored on (None: the scorer did not say).
    #[serde(default)]
    pub hff_inputs: Option<u32>,
}

/// The repainting frame: the state of the search at one beat.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    #[serde(flatten)]
    pub header: Header,
    pub budget_ms: u64,
    pub global: Global,
    pub islands: Vec<IslandRow>,
    /// Cohorts summed over every island — the table the mockup draws. Empty when
    /// cohorts are off.
    pub global_cohorts: Vec<CohortRow>,
    /// The `model` record holding the expression this snapshot's best belongs to,
    /// by its `seq`. None until the first model has been written. The full
    /// expression is NEVER on a snapshot: it is hundreds of characters and it
    /// changes far more rarely than the numbers do.
    #[serde(default)]
    pub model_ref: Option<u64>,
    /// Pumps run since the previous snapshot — the denominator for "did a fresh
    /// line survive a pump beat", and the reason the viewer can say `—` for
    /// survival honestly rather than calling a quiet window zero.
    #[serde(default)]
    pub pumps_since: u32,
}

/// The run as a whole at this beat.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Global {
    /// The best HFF angle in the population now, by TrueNorth. LOWER IS BETTER,
    /// everywhere in this schema and everywhere in the viewer.
    pub best_hff: Option<f64>,
    /// The best HFF the fit has EVER held (the hall of fame's), which is not the
    /// same number: a champion can leave the population. Reported separately so
    /// a current best is never mistaken for a record.
    pub best_ever_hff: Option<f64>,
    /// The mean over rows that scored at all.
    pub avg_hff: Option<f64>,
    pub mse_train: Option<f64>,
    pub r2_train: Option<f64>,
    pub r2_val: Option<f64>,
    /// The third block's R² — edge or SMOGD rows. None when there is no third block.
    pub r2_third: Option<f64>,
    /// The best model's HFF angle as a p-value, log10.
    pub log10_p: Option<f64>,
    /// The tower height of the best model: how deep its nesting goes.
    #[serde(default)]
    pub t_depth: u32,
    /// The virtual head in force at this generation — the room a gene may use.
    #[serde(default)]
    pub vhead: u32,
    /// Rows that produced no score at all. A JSON number cannot be NaN, so the
    /// count is carried and the reader is never left to infer it from a gap.
    #[serde(default)]
    pub nan_rows: u32,
    /// THE REGEX KINGDOM's Rule-Golf objectives, when this is a regex fit; `None`
    /// for symbolic regression. `mse_train`/`r2_*`/`log10_p` above are SR-shaped
    /// and mean nothing for a regex, so a regex viewer reads these instead:
    /// `fn_mass` = missed positives / positives, `fp_mass` = matched negatives /
    /// negatives, `strokes` = the NFA instruction count (the golf cost). Additive
    /// with serde defaults, so every existing SR stream still parses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fn_mass: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp_mass: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strokes: Option<u32>,
    /// THE WGSL KINGDOMS' numbers (a kernel per individual,
    /// `examples/wgsl_fit.rs`), boxed so the snapshot record stays the size
    /// it was; `None` on every other kingdom's stream. Additive with serde
    /// defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wgsl: Option<Box<WgslGlobal>>,
    /// POPULATION DIVERSITY over the whole population (`evolve::diversity`),
    /// emitted only when `Config::diversity` is on — a stream from a run with
    /// it off, or an older stream, carries none of these and reads `None`.
    /// `distinct_genes` is over expressed genes (`genes` of them);
    /// `distinct_chromosomes` over all rows, a chromosome being the multiset
    /// of its gene hashes; `max_gene_copies` the most-copied expressed gene.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub genes: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distinct_genes: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distinct_chromosomes: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_gene_copies: Option<u32>,
    /// CANDIDATE EVALUATIONS so far, by source (`evolve::evaluations`), and
    /// the running rate per generation (total / generations so far). Always
    /// emitted by the engine — a counter, not a switch; `Option` only so an
    /// older stream without it still parses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluations: Option<EvalCounts>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluations_per_generation: Option<f64>,
    /// THE GENERIC VIEW's numbers, which any kingdom fills ([`RunTelemetry::generation`]): the median HFF over the
    /// rows that scored, the best individual's angle per HFF group (two-level HFF: `quality`, `speed`), and the
    /// runner's own measurements by name. Each is omitted from the wire when unset, so every stream written
    /// before they existed is byte for byte what it was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub median_hff: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub angle_groups: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extras: BTreeMap<String, f64>,
}

/// THE WGSL KINGDOMS' global numbers, one generation of a kernel-per-individual
/// fit.
///
/// `evaluated` is the rows the scorer measured this generation (the rest were
/// carried over from their parents or shared a kernel text with a row scored
/// before them, `memo_hits`). The true multi-typed fit (`true_wgsl_fit`) adds
/// the size of the expressed trees, the population's diversity, the mutation
/// modes that made the next population, the generation's wall and stage
/// clocks, kernels the device failed to run, and `extras`: whatever the
/// scorer measures that the core does not know about (an image fit's
/// picture-exact share and GPU ms, a cache's hits), as name and number.
///
/// The legacy fit (`examples/wgsl_fit.rs`) wrote `feasible_rows`, `accepted`,
/// `attempts`, `best_speedups` and `stages`; they are kept, optional, so its
/// streams still parse, and are omitted from the wire when not set.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct WgslGlobal {
    #[serde(default, skip_serializing_if = "is_zero")]
    pub feasible_rows: u32,
    #[serde(default)]
    pub evaluated: u32,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub accepted: u32,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_speedups: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stages: BTreeMap<String, u32>,
    /// Expressed nodes of the population's trees: the median and the largest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nodes_median: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nodes_max: Option<u32>,
    /// Distinct expressed trees and distinct fitness values in the population,
    /// and the median overlap of sampled pairs of trees (1 = the same tree).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distinct_trees: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distinct_fitness: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlap_median: Option<f64>,
    /// The shares of the mutation modes that make the NEXT population.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modes: Option<WgslModes>,
    /// The diversity dial (None: the schedule is fixed) and the LEAF mode's exponent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dial: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leaf_alpha: Option<f64>,
    /// The generation's wall clock and the milliseconds of each stage, by name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stage_ms: BTreeMap<String, f64>,
    /// Rows this generation that reused the score of a row with the same kernel text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memo_hits: Option<u32>,
    /// Kernels the device failed to run: this generation, and the run so far.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_failures: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_failures_total: Option<u32>,
    /// Rows the pump replaced this generation (0 when it did not fire).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pumped: Option<u32>,
    /// The scorer's own measurements of the generation, by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extras: BTreeMap<String, f64>,
    /// THE IN-LOOP OPTIMISERS of the generation (`true_wgsl_fit::OptSpec`), by name (`cleanse`, `egraph_exact`,
    /// `egraph_real`, `hoist`, `snap`, `trips`, `fold`): what each made, entered, improved and could not write.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub optimisers: BTreeMap<String, OptTally>,
    /// `let`s the share pass bound over the kernels scored this generation (None: the pass is off).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub share_lets: Option<u32>,
}

/// One optimiser's generation: new programs it made, children that entered the population (scored), children better
/// than the individual they were made from, and rewrites it could not write (named defects in the log).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct OptTally {
    pub made: u32,
    pub entered: u32,
    pub improved: u32,
    pub defects: u32,
}

/// The mutation modes' shares (they sum to 1).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct WgslModes {
    pub low: f64,
    pub middle: f64,
    pub leaf: f64,
    pub high: f64,
}

fn is_zero(v: &u32) -> bool {
    *v == 0
}

/// The hall of fame at one moment, best first.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HallOfFame {
    #[serde(flatten)]
    pub header: Header,
    pub rows: Vec<HofRow>,
}

/// One entry of the hall of fame.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct HofRow {
    /// 1 is the best.
    pub rank: u32,
    pub hff: Option<f64>,
    /// The generation the entry was found at.
    pub born: u32,
    /// The expressed tree's hash, as hex: the entry's identity across records.
    pub tree: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nodes: Option<u32>,
    /// The kernel's GPU time, ms, when the scorer measured it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_ms: Option<f64>,
    /// A picture of the entry's output, when the runner wrote one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub picture: Option<String>,
    /// The runner's own measurements of the entry (e.g. differing channels), by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extras: BTreeMap<String, f64>,
}

/// One island's row. The brief's example had these `null` because the supplied
/// log could not recover them; the engine can, from a slice of the fitness
/// vector, so nothing here is null that the engine knows.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IslandRow {
    /// `intake-0`, `champion-0`, `intake-1`, … — pair p is islands 2p and 2p+1.
    /// The brief's example writes the bare `intake` / `champion` of a
    /// single-pair run; the suffix is what makes the id unique once there is
    /// more than one pair, and a reader treats it as an opaque label.
    pub id: String,
    /// None when the producer did not say. The brief's example record has no
    /// `kind`, and a viewer that cannot tell intake from champion should show
    /// the id it was given rather than guess.
    #[serde(default)]
    pub kind: Option<IslandKind>,
    #[serde(default)]
    pub pair: u32,
    pub rows: u32,
    pub best_hff: Option<f64>,
    #[serde(default)]
    pub avg_hff: Option<f64>,
    /// Rows that produced no score. `None` = NOT EMITTED, which is a different
    /// thing from zero: a viewer must not turn a silence into "100% of rows
    /// scored". The engine always fills it; the brief's example does not.
    #[serde(default)]
    pub nan_rows: Option<u32>,
    /// THIS ISLAND's diversity (same definitions as `Global`'s), emitted only
    /// when `Config::diversity` is on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub genes: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distinct_genes: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distinct_chromosomes: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_gene_copies: Option<u32>,
    /// THIS ISLAND's cohort split. Per the brief's invariant, a cohort's island
    /// counts sum to its global count — which is what makes the global table's
    /// rows attributable instead of a total nobody can place.
    pub cohorts: Vec<CohortRow>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IslandKind {
    Intake,
    Champion,
}

/// A cohort as it stands: the pump beat a line arrived on, how many rows carry
/// that label, and the best of them.
///
/// A cohort label is INHERITED LINEAGE MEMBERSHIP, not a count of independent
/// lines: 80,000 rows of c100 are 80,000 descendants of whatever the pump put in
/// at generation 100, which may be far fewer lines. `birth_generation` is the
/// label itself (cohorts are keyed on the beat), kept as its own field so a
/// reader never has to know that.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CohortRow {
    pub id: u32,
    pub birth_generation: u32,
    pub rows: u32,
    pub best_hff: Option<f64>,
    /// Rows of this cohort that produced no score.
    #[serde(default)]
    pub nan_rows: u32,
}

/// Something that happened, written the moment it did rather than waiting for a
/// beat. Events are low-volume and MUST NOT be dropped when the writer lags:
/// a snapshot that is late is merely stale, but a missed extinction is a hole in
/// the record that nothing later can fill.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Event {
    #[serde(flatten)]
    pub header: Header,
    pub kind: EventKind,
    /// One line for a human, already written — the viewer prints it as given.
    pub message: String,
    /// The cohort the event is about, when it is about one.
    pub cohort: Option<u32>,
    /// The number the event turns on, when it has one (a new best's HFF, a
    /// merged cohort's label). Never a percentage: the viewer computes those.
    pub value: Option<f64>,
    /// WHAT A DISCOVERY REPLACED, as a number — the literal snap found, or the
    /// value a near-constant subtree was folded to. `value` above carries what it
    /// BECAME, so before and after are two numbers and the viewer subtracts them
    /// rather than parsing `message` back into arithmetic ("numbers are numbers",
    /// at the top of this file).
    #[serde(default)]
    pub before: Option<f64>,
    /// What it became, as a FORM: `pi`, `2*pi`, `sqrt(2)` for a snap. A fold's
    /// after is a number and lives in `value`; its `detail` is the subtree that
    /// went.
    #[serde(default)]
    pub detail: Option<String>,
    /// How many nodes the discovery removed — a fold's whole point, and the
    /// number that makes "thirteen operators wearing a 1" readable as a size.
    #[serde(default)]
    pub nodes: Option<u32>,
    /// The population row a snap was written into, when the event is about one.
    #[serde(default)]
    pub row: Option<u32>,
    /// THE MUTATION EVENT's fields (`docs/STANDARD_runners.md`, "Mutation events"): every mutation outside plain
    /// GEP variation is one event, written through [`Writer::mutation`] or [`RunTelemetry::mutation`]. Each field
    /// is omitted when unset, so every event written before they existed is byte for byte what it was.
    ///
    /// The operator's own name (`egraph_exact`, `cleanse`, …): finer than `kind` where one kind has variants.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,
    /// The individual's lineage: the ids of its parents (the runner's own individual ids).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parents: Vec<u64>,
    /// The individual's own id, when the runner gives individuals ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    /// Before and after, as the kingdom's own text (a subtree, a formula, a rule, a kernel line), each at most
    /// [`EVENT_TEXT_LIMIT`] characters; `text_file` names the side file holding both in full when either was cut.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_before: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_after: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_file: Option<String>,
    /// The individual's HFF before and after the mutation, where known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fitness_before: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fitness_after: Option<f64>,
    /// Whether the child entered the population, and whether it was better than the individual it was made from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entered: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub improved: Option<bool>,
    /// A batched event stands for this many mutations (×N); absent means one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// The global best HFF fell.
    NewBest,
    /// A cohort label appeared that the previous snapshot did not hold.
    CohortBorn,
    /// A cohort label the previous snapshot held has no rows left.
    CohortExtinct,
    /// Cohorts at or past `cohort_merge` are ONE band. This says how the
    /// displayed IDs aggregate from here on, because a merge changes the
    /// classification rule and a table that silently re-labels is a lie.
    CohortMerge,
    /// The fit's stop bar was met.
    EarlyStop,
    /// SNAP WROTE A NAMED CONSTANT INTO A GENE: a literal the search had fitted
    /// numerically is now `pi` (or `2*pi`, or `sqrt(2)`) and breeds as that
    /// token. Rare — it is the last branch of a four-stage pipeline — and
    /// therefore exactly the kind of thing an event is for.
    Snap,
    /// THE ROUNDING GENERATOR FOLDED A NEAR-CONSTANT SUBTREE: a subtree whose
    /// whole range across the rows was under a percent of its own value, replaced
    /// by that value.
    ///
    /// A fold arrives from TWO places and they are the same finding. The pump's
    /// fold operator sends one the moment it writes a folded gene back into the
    /// population, mid-fit, with the `row` it landed in; the final form sends the
    /// ones it made on the reported model, after `run_end`. Both say "this
    /// subtree was secretly a constant".
    Fold,
    /// THE LEAVE-ONE-OUT DROPPED A SUBTREE THE DATA CANNOT SEE: a subtree held at
    /// its mean on the rows without the model's predictions moving, so it was
    /// carrying nothing and was removed permanently.
    ///
    /// The COMPLEMENT of a fold and not a repeat of it. A fold says the subtree
    /// had one value — it was a constant in a costume, and the constant is kept.
    /// A reduction says the subtree had no effect on the OUTPUT at all, whatever
    /// it computed, so nothing of it is kept. A panel that showed one and not the
    /// other would be showing half the tidy.
    ///
    /// These arrive after `run_end` only: the reduction runs over the egglog
    /// candidate forms in the final form, which is the caller's step once the fit
    /// has returned, and there is no honest earlier moment for them.
    Reduce,
    /// TYPED SNAP SWITCHED A SUBTREE FOR A LIBRARY SHAPE: a subtree whose values
    /// matched a typed pattern (the Lorentz factor, a Gaussian, …) up to scale —
    /// the ratio subtree/pattern was flat — replaced by the clean pattern and
    /// landed for the tournament. The shape-level analogue of a snap: snap writes
    /// a named CONSTANT, typed snap writes a named SHAPE.
    TypedSnap,
    /// REGEX SIMPLIFY FOLDED A CLEANED RULE BACK INTO THE POPULATION: a bloated
    /// evolved regex was algebraically shrunk by fuller's `regex_simplify` (sound,
    /// equivalence-preserving rewrites), re-encoded into a valid multi-typed Karva
    /// gene and landed in its island's worst row for the tournament to judge. The
    /// `what`/`became` of its Discovery hold the before/after rule; `nodes` the
    /// node count after. The regex kingdom's analogue of SR's fold.
    RegexSimplify,
    /// THE PUMP REPLACED THE WORST ROWS (a WGSL fit): `value` is how many, the
    /// message says from where (the hall of fame, given genes, random genes).
    Pump,
    /// THE DEVICE FAILED TO RUN A KERNEL (a WGSL fit). By design this should
    /// not occur: the message names the generation, the row and where its gene
    /// was saved.
    DeviceFailure,
    /// THE MUTATION EVENTS (`docs/STANDARD_runners.md`): every mutation outside plain GEP variation. `fold`,
    /// `snap` and `pump` above are three of them; these are the rest. The in-loop cleanse (dead code out).
    Cleanse,
    /// The e-graph's exact rewrite (an equality by construction).
    EgraphExact,
    /// The e-graph's value-changing rewrite (real-domain rules that move the value).
    EgraphReal,
    /// A hoist (a subtree lifted out of a loop or a branch).
    Hoist,
    /// A trip-count rewrite.
    Trips,
    /// The ADF fold: shared parts made automatically defined functions (`count` ADFs created, genes rewritten).
    AdfFold,
    /// The saturated share-equivalence fold (ADFs created, genes rewritten, the shared part).
    SaturatedFold,
    /// `let` sharing: a repeated subexpression bound once.
    LetShare,
    /// The marriage step (two parents' subtrees joined by the kingdom's own rule).
    Marriage,
    /// Subtree crossover.
    SubtreeCrossover,
    /// The LEAF mutation mode.
    Leaf,
    /// A model rewrite: an LLM's edit of an individual.
    ModelRewrite,
    /// A rewrite the gene could not hold: written back with a defect (the message names it).
    WriteBackDefect,
    /// Something the engine wants an operator to see that is not one of the above.
    #[default]
    Note,
}

/// The full expression, written rarely. A snapshot points at one of these by
/// `seq` (`Snapshot::model_ref`) rather than carrying hundreds of characters
/// every beat.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    #[serde(flatten)]
    pub header: Header,
    pub hff: Option<f64>,
    /// The generation the model was FOUND at, which is earlier than the
    /// generation it is reported at.
    pub found_generation: u32,
    /// The faithful form: every protected operator written as itself, so the
    /// string computes what the chromosome computes.
    pub infix_protected: String,
    /// The same model with the protected operators written as ordinary ones —
    /// what a symbolic comparison gets. It is NOT executable on every input.
    pub infix_plain: String,
    /// The engine's own `Math` s-expression.
    pub raw_math: String,
    pub t_depth: u32,
}

/// The last line. A finished run must be visibly different from a stale one, and
/// this is the difference: a viewer that has seen a `run_end` says FINISHED, and
/// one that has merely stopped receiving says STALE.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunEnd {
    #[serde(flatten)]
    pub header: Header,
    /// `time`, `n_gen`, `early_stop` — the engine's own word.
    pub stopped_by: String,
    pub generations: u32,
    pub individuals: u64,
    /// THE CONFIRMED f64 RESCORE, which is NOT the last snapshot's
    /// `global.best_hff`. A snapshot's numbers come from the device's f32
    /// ranking scores; the fit's answer is re-scored in f64 at the end
    /// (`Engine::confirm`), and the two differ in the last digits — and by more
    /// than that when the hall of fame's winner is put back into a row. A viewer
    /// showing both must not read the difference as a bug.
    pub best_hff: Option<f64>,
    /// The WHOLE fit's seconds, across every stop and resume — the same clock
    /// the header's `elapsed_ms` carries, not this process's alone.
    pub seconds: f64,
    /// THE HOLDOUT RESULT (regex kingdom): the champion scored once, at fit-end,
    /// on the untouched holdout block. `fn`/`fp` are rates on [0,1]; `pos`/`neg`
    /// are the block's class counts. `None` when there is no holdout block. The
    /// viewer shows this as the honest final test when the run is FINISHED.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holdout_fn: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holdout_fp: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holdout_pos: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holdout_neg: Option<usize>,
}

// ---------------------------------------------------------------------------
// The writer
// ---------------------------------------------------------------------------

/// The stream's writer: it owns the file, the sequence counter and the clock the
/// `elapsed_ms` of every record is measured from.
///
/// Writes are SYNCHRONOUS, one `writeln!` and one flush per record. The brief
/// asks for a bounded channel to a single writer thread, and that is the right
/// shape at a high beat; at the beat this actually runs on — the progress
/// report's, at most once a second — a record is a few hundred bytes and the
/// write is tens of microseconds against generations that cost milliseconds.
/// A thread would buy nothing measurable and would add a shutdown ordering
/// problem (the `run_end` must not be lost when the fit ends), so the deviation
/// is deliberate and measured. If the beat ever rises, the channel goes here and
/// nothing outside this file changes.
///
/// Flushing every line is what makes `evowatch` work at all: an
/// unflushed line is not a line, and a viewer tailing a buffered file would show
/// a frame that is minutes old and call it live.
pub struct Writer {
    file: std::fs::File,
    run_id: String,
    seq: u64,
    started: std::time::Instant,
    /// THE SECONDS THE FIT SPENT BEFORE THIS PROCESS STARTED, from a checkpoint.
    /// `started` clocks this process, and a resumed run has two of them, so a
    /// header measured from `started` alone restarts the viewer's clock at zero
    /// and puts a run 15 s in at 0 ms against a budget bar it is a quarter of
    /// the way through. `elapsed_ms` is the WHOLE fit's, across every stop and
    /// start, exactly as `generation` is — and it is the same quantity the fit
    /// loop spends its budget against and writes into its checkpoints.
    already_spent: f64,
    /// When the last snapshot went out, so the ≥1 s throttle the brief asks for
    /// can be applied without a second clock.
    last_snapshot: Option<std::time::Instant>,
    /// The `seq` of the most recent `model` record, for `Snapshot::model_ref`.
    last_model: Option<u64>,
}

/// How long a snapshot waits behind the one before it. The brief's cadence is
/// "each 1 second or 10 generations, whichever is LATER" — the generation half
/// is the caller's beat (`progress_every`), and this is the second half: a fit
/// running hundreds of generations a second must not write hundreds of frames a
/// second at a screen that repaints four times a second.
const SNAPSHOT_MIN_GAP: std::time::Duration = std::time::Duration::from_millis(1000);

/// The floor, when a caller asks for a faster pulse through
/// `HFF_TELEMETRY_MIN_MS`. A snapshot costs a line of JSON and an O(population)
/// walk over `gen.fitness`, which the fit already holds on the host — the cohort
/// labels, the one part that came from the device, are now read only when the
/// pump has moved them. So the pulse can be raised to whatever a screen can use;
/// the viewer repaints at 2-4 Hz, so past about 250 ms the frames are written to
/// be thrown away.
fn snapshot_min_gap() -> std::time::Duration {
    match std::env::var("HFF_TELEMETRY_MIN_MS").ok().and_then(|v| v.parse::<u64>().ok()) {
        Some(ms) => std::time::Duration::from_millis(ms),
        None => SNAPSHOT_MIN_GAP,
    }
}

impl Writer {
    /// Open the stream and write its `run_start`. Truncates: a run owns its file,
    /// as the hall of fame and the genealogy log do.
    ///
    /// **`HFF_TELEMETRY_APPEND=1` opens it for APPEND instead**, and that is what
    /// makes a CASCADE watchable. A sweep over the 133 runs one fit per law, each
    /// lasting a minute; with a file per law the viewer is pointed at a stream
    /// that dies sixty seconds later, and following a corpse through a 133-law
    /// run is no way to watch it. Appending puts every law in ONE file: each fit
    /// writes its own `run_start`, and the viewer's `Tailer` — which follows a
    /// path by byte offset and reads a fresh `run_start` as a new run — rolls
    /// onto the next law on its own. One path, live from the first law to the
    /// last, with no symlink and no helper process to go stale.
    ///
    /// Truncating stays the default because a single fit owning its file is the
    /// right behaviour for a single fit, and a sweep that silently appended to a
    /// previous sweep's stream would be worse than either.
    pub fn create(path: &str, run_id: String, start: RunStartFields) -> Result<Writer, String> {
        let append = std::env::var("HFF_TELEMETRY_APPEND").is_ok_and(|v| v == "1");
        Self::open(path, run_id, start, append)
    }

    /// Open the stream for APPEND: a resumed fit keeps the records of the run
    /// it resumes and adds its own `run_start` after them.
    pub fn append(path: &str, run_id: String, start: RunStartFields) -> Result<Writer, String> {
        Self::open(path, run_id, start, true)
    }

    fn open(path: &str, run_id: String, start: RunStartFields, append: bool) -> Result<Writer, String> {
        let file = if append {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| format!("telemetry file {path}: {e}"))?
        } else {
            std::fs::File::create(path).map_err(|e| format!("telemetry file {path}: {e}"))?
        };
        let mut w = Writer {
            file,
            run_id,
            seq: 0,
            started: std::time::Instant::now(),
            already_spent: start.already_spent_seconds,
            last_snapshot: None,
            last_model: None,
        };
        let header = w.header(0);
        let record = Record::RunStart(RunStart {
            header,
            dataset: start.dataset,
            seed: start.seed,
            population: start.population,
            n_pairs: start.n_pairs,
            pop_intake: start.pop_intake,
            pop_champion: start.pop_champion,
            float_zone: start.float_zone,
            n_train: start.n_train,
            n_val: start.n_val,
            n_extrap: start.n_extrap,
            budget_ms: start.budget_ms,
            max_generations: start.max_generations,
            cohort_merge: start.cohort_merge,
            pump_every: start.pump_every,
            progress_every: start.progress_every,
            stop_log10_p: start.stop_log10_p,
            stop_one_minus_r2: start.stop_one_minus_r2,
            config_json: start.config_json,
            kingdom: start.kingdom,
            hff_objectives: start.hff_objectives,
            wgsl: start.wgsl,
        });
        w.write(&record)?;
        Ok(w)
    }

    /// The `seq` the next record will carry (a side file is named by it).
    pub fn next_seq(&self) -> u64 {
        self.seq
    }

    /// A MUTATION EVENT (`docs/STANDARD_runners.md`). The text is written as given: [`RunTelemetry::mutation`]
    /// cuts it to [`EVENT_TEXT_LIMIT`] and keeps the full text in a side file.
    pub fn mutation(&mut self, generation: u32, m: Mutation) -> Result<(), String> {
        let header = self.header(generation);
        self.write(&Record::Event(Event {
            header,
            kind: m.kind,
            message: m.message,
            cohort: None,
            value: m.fitness_after,
            before: None,
            detail: None,
            nodes: m.nodes,
            row: m.row,
            operator: Some(m.operator),
            parents: m.parents,
            id: m.id,
            text_before: m.text_before,
            text_after: m.text_after,
            text_file: m.text_file,
            fitness_before: m.fitness_before,
            fitness_after: m.fitness_after,
            entered: m.entered,
            improved: m.improved,
            count: m.count,
        }))
    }

    /// The run's id — the viewer shows it, and a rotation is detected by it
    /// changing.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// The fit's total wall clock, in milliseconds: what was spent before this
    /// process plus what this process has spent. `as u64` saturates rather than
    /// wrapping, so a checkpoint carrying an absurd `elapsed_seconds` gives a
    /// clamped number instead of a panic or a small wrong one.
    pub fn elapsed_ms(&self) -> u64 {
        ((self.already_spent + self.started.elapsed().as_secs_f64()) * 1e3) as u64
    }

    /// The next header, and the counter advanced. Every record gets one, so the
    /// sequence has no gaps by construction.
    fn header(&mut self, generation: u32) -> Header {
        let seq = self.seq;
        self.seq += 1;
        Header {
            schema_version: SCHEMA_VERSION,
            run_id: self.run_id.clone(),
            seq,
            timestamp_utc: now_utc(),
            elapsed_ms: self.elapsed_ms(),
            generation,
        }
    }

    fn write(&mut self, record: &Record) -> Result<(), String> {
        let line = serde_json::to_string(record).map_err(|e| format!("telemetry: {e}"))?;
        writeln!(self.file, "{line}").map_err(|e| format!("telemetry write: {e}"))?;
        self.file.flush().map_err(|e| format!("telemetry flush: {e}"))
    }

    /// Whether a snapshot at this moment would be inside the minimum gap. The
    /// caller asks BEFORE it builds one, so a dropped frame costs nothing: the
    /// per-island and per-cohort reductions are never run for a frame that will
    /// not be written. `force` is the last report of a fit, which always goes out.
    pub fn snapshot_due(&self, force: bool) -> bool {
        force || self.last_snapshot.is_none_or(|t| t.elapsed() >= snapshot_min_gap())
    }

    /// Write a snapshot. `body` is everything but the header — the caller builds
    /// it from the scans it has already made.
    pub fn snapshot(&mut self, generation: u32, budget_ms: u64, body: SnapshotBody) -> Result<(), String> {
        let header = self.header(generation);
        let model_ref = self.last_model;
        self.last_snapshot = Some(std::time::Instant::now());
        self.write(&Record::Snapshot(Snapshot {
            header,
            budget_ms,
            global: body.global,
            islands: body.islands,
            global_cohorts: body.global_cohorts,
            model_ref,
            pumps_since: body.pumps_since,
        }))
    }

    pub fn event(&mut self, generation: u32, kind: EventKind, message: String, cohort: Option<u32>, value: Option<f64>) -> Result<(), String> {
        let header = self.header(generation);
        self.write(&Record::Event(Event { header, kind, message, cohort, value, before: None, detail: None, nodes: None, row: None, ..Event::default() }))
    }

    /// A DISCOVERY: a snap or a fold, with its before and after carried as the
    /// numbers and forms they are rather than only as prose. The `message` is
    /// still written, because the events pane and `--dump` print it and a reader
    /// that knows none of the new fields still sees what happened.
    pub fn discovery(&mut self, generation: u32, kind: EventKind, message: String, d: Discovery) -> Result<(), String> {
        let header = self.header(generation);
        self.write(&Record::Event(Event {
            header,
            kind,
            message,
            cohort: None,
            value: d.after,
            before: d.before,
            detail: d.detail,
            nodes: d.nodes,
            row: d.row,
            ..Event::default()
        }))
    }

    /// Write a model and remember its `seq`, so the snapshots after it point at it.
    pub fn model(&mut self, generation: u32, model: ModelFields) -> Result<(), String> {
        let header = self.header(generation);
        self.last_model = Some(header.seq);
        self.write(&Record::Model(Model {
            header,
            hff: finite(model.hff),
            found_generation: model.found_generation,
            infix_protected: model.infix_protected,
            infix_plain: model.infix_plain,
            raw_math: model.raw_math,
            t_depth: model.t_depth,
        }))
    }

    /// Write the hall of fame as it stands, best first.
    pub fn hall_of_fame(&mut self, generation: u32, rows: Vec<HofRow>) -> Result<(), String> {
        let header = self.header(generation);
        self.write(&Record::HallOfFame(HallOfFame { header, rows }))
    }

    /// `holdout` is `Some((fn_rate, fp_rate, n_pos, n_neg))` for a regex fit with
    /// a holdout block, else `None`.
    pub fn run_end(&mut self, generation: u32, stopped_by: &str, generations: u32, individuals: u64, best_hff: Option<f64>, seconds: f64, holdout: Option<(f64, f64, usize, usize)>) -> Result<(), String> {
        let header = self.header(generation);
        self.write(&Record::RunEnd(RunEnd {
            header,
            stopped_by: stopped_by.to_string(),
            generations,
            individuals,
            best_hff: best_hff.and_then(finite),
            seconds,
            holdout_fn: holdout.map(|h| h.0),
            holdout_fp: holdout.map(|h| h.1),
            holdout_pos: holdout.map(|h| h.2),
            holdout_neg: holdout.map(|h| h.3),
        }))
    }
}

/// The most characters of before or after text an event carries; the rest goes to a side file.
pub const EVENT_TEXT_LIMIT: usize = 240;

/// One mutation outside plain GEP variation, as a runner reports it ([`Writer::mutation`],
/// [`RunTelemetry::mutation`]).
#[derive(Clone, Debug)]
pub struct Mutation {
    pub kind: EventKind,
    /// The operator's own name.
    pub operator: String,
    /// One line for a human.
    pub message: String,
    pub row: Option<u32>,
    pub id: Option<u64>,
    pub parents: Vec<u64>,
    pub text_before: Option<String>,
    pub text_after: Option<String>,
    /// Set by [`RunTelemetry::mutation`] when the text was cut.
    pub text_file: Option<String>,
    pub fitness_before: Option<f64>,
    pub fitness_after: Option<f64>,
    pub entered: Option<bool>,
    pub improved: Option<bool>,
    pub nodes: Option<u32>,
    pub count: Option<u32>,
}

impl Mutation {
    /// A mutation of `kind` by `operator`, every other field unset.
    pub fn new(kind: EventKind, operator: &str, message: String) -> Mutation {
        Mutation {
            kind,
            operator: operator.to_string(),
            message,
            row: None,
            id: None,
            parents: Vec::new(),
            text_before: None,
            text_after: None,
            text_file: None,
            fitness_before: None,
            fitness_after: None,
            entered: None,
            improved: None,
            nodes: None,
            count: None,
        }
    }
}

/// `text` cut to [`EVENT_TEXT_LIMIT`] characters (with an ellipsis), and whether it was cut.
pub fn cut_text(text: &str) -> (String, bool) {
    if text.chars().count() <= EVENT_TEXT_LIMIT {
        return (text.to_string(), false);
    }
    let mut cut: String = text.chars().take(EVENT_TEXT_LIMIT - 1).collect();
    cut.push('…');
    (cut, true)
}

/// The fixed facts a `run_start` carries, as one argument rather than fifteen.
#[derive(Default)]
pub struct RunStartFields {
    pub dataset: String,
    pub seed: u32,
    pub population: u32,
    pub n_pairs: u32,
    pub pop_intake: u32,
    pub pop_champion: u32,
    pub float_zone: u32,
    pub n_train: usize,
    pub n_val: usize,
    pub n_extrap: usize,
    pub budget_ms: u64,
    pub max_generations: u32,
    pub cohort_merge: u32,
    pub pump_every: u32,
    pub progress_every: u32,
    /// What a resumed fit had already spent when this process picked it up, so
    /// every `elapsed_ms` this writer stamps is the whole fit's clock. 0 for a
    /// fit starting from nothing.
    pub already_spent_seconds: f64,
    /// The stop bar, both halves, straight off the config. `None` for a half
    /// that is switched off — an infinite p bar or a negative 1 - R² bar is not
    /// a bar, and the engine maps both to the silence they are.
    pub stop_log10_p: Option<f64>,
    pub stop_one_minus_r2: Option<f64>,
    /// The whole `Config` as JSON, for the viewer's card overlay. See
    /// [`RunStart::config_json`].
    pub config_json: Option<String>,
    /// `Config::kingdom` — `"Regex"` for a regex fit, else empty/SR. See
    /// [`RunStart::kingdom`].
    pub kingdom: Option<String>,
    /// The HFF objectives by name, in column order. See [`RunStart::hff_objectives`].
    pub hff_objectives: Vec<String>,
    /// A WGSL run's fixed facts. See [`RunStart::wgsl`].
    pub wgsl: Option<WgslStart>,
}

/// A snapshot without its header — what the engine builds.
pub struct SnapshotBody {
    pub global: Global,
    pub islands: Vec<IslandRow>,
    pub global_cohorts: Vec<CohortRow>,
    pub pumps_since: u32,
}

/// A discovery's typed half — the fields that make `before -> after` a pair of
/// values instead of a sentence, as one argument rather than five.
#[derive(Clone, Debug, Default)]
pub struct Discovery {
    pub before: Option<f64>,
    pub after: Option<f64>,
    pub detail: Option<String>,
    pub nodes: Option<u32>,
    pub row: Option<u32>,
}

pub struct ModelFields {
    pub hff: f64,
    pub found_generation: u32,
    pub infix_protected: String,
    pub infix_plain: String,
    pub raw_math: String,
    pub t_depth: u32,
}

/// A number, or nothing. Infinity is what an empty minimum reduces to and NaN is
/// what a row that would not score produces; neither is a measurement, and both
/// become JSON `null` so a reader never has to guess which sentinel it is
/// looking at.
pub fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

/// The wall clock as RFC-3339 UTC to the second, computed from the epoch by hand.
///
/// A date library for one timestamp would be a dependency the crate does not
/// otherwise need, and the arithmetic here is the whole of it: days since 1970
/// converted to a civil date by Howard Hinnant's `civil_from_days`, which is
/// exact for every date this will ever see.
pub fn now_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (y, m, d) = civil_from_days(days as i64);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Days since the Unix epoch as (year, month, day). Hinnant's algorithm: shift
/// the era so March is the first month, which makes the leap day the last day of
/// the year and the whole thing branch-free.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ---------------------------------------------------------------------------
// The run: what every generation loop creates to run
// ---------------------------------------------------------------------------

/// What a runner says about itself once, at its start, for [`RunTelemetry::start`].
#[derive(Clone, Debug, Default)]
pub struct RunInfo {
    /// The kingdom's name, as the watchers show it (`futhark_shapes`, `wgsl`, `Regex`, …). A kingdom the watcher
    /// has no view of its own for gets the generic view, so a new name needs no watcher change.
    pub kingdom: String,
    /// The problem or dataset the run measures against.
    pub dataset: String,
    pub seed: u32,
    /// The individuals a generation holds.
    pub population: u32,
    /// The generation bound (the watcher's progress bar).
    pub max_generations: u32,
    /// The wall-clock cap in milliseconds (0: none).
    pub budget_ms: u64,
    /// What each HFF input measures, in column order.
    pub hff_objectives: Vec<String>,
    /// The runner's whole configuration as JSON, for the watcher's card.
    pub config_json: Option<String>,
    /// A resumed run APPENDS to its stream and starts its clock where the checkpoint left it.
    pub resume: bool,
    /// The seconds the run spent before this process (from its checkpoint); 0 for a fresh run.
    pub already_spent_seconds: f64,
}

/// One generation, as the generic view reads it. The fitness is HFF per individual (lower is better, 0 best,
/// π/2 worst); a NaN or infinite entry is an individual that did not score.
#[derive(Clone, Debug, Default)]
pub struct GenerationStats<'a> {
    pub generation: u32,
    pub fitness: &'a [f64],
    /// Distinct individuals (by gene) in the generation, when the runner counts them.
    pub distinct: Option<u32>,
    /// The best individual's angle per HFF group (two-level HFF), by group name.
    pub angle_groups: BTreeMap<String, f64>,
    /// The runner's own measurements, by name.
    pub extras: BTreeMap<String, f64>,
    /// Candidate evaluations this generation (summed into the run's `individuals`).
    pub evaluated: u64,
}

/// THE RUN'S STREAM AND KILL TRIGGER. A generation loop takes one of these to run at all, so a runner cannot skip
/// the stream: [`RunTelemetry::start`] creates `OUT/stream.jsonl` (and `OUT/killme.trigger`), every generation is a
/// snapshot, and the run's end is written by [`RunTelemetry::end`] — or, when the loop unwinds without reaching it,
/// by `Drop`, as `stopped_by: "error"`, so a run that died is never mistaken for a stale live one.
pub struct RunTelemetry {
    writer: Writer,
    dir: std::path::PathBuf,
    run_file: crate::run_file::RunFile,
    best_ever: Option<f64>,
    individuals: u64,
    last_generation: u32,
    ended: bool,
}

/// The stream's file name inside a run's output directory.
pub const STREAM_FILE: &str = "stream.jsonl";

impl RunTelemetry {
    /// Open `out/stream.jsonl` (creating `out`) and write the `run_start`; write `out/killme.trigger`.
    pub fn start(out: &std::path::Path, run_id: String, info: RunInfo) -> Result<RunTelemetry, String> {
        let fields = RunStartFields {
            dataset: info.dataset,
            seed: info.seed,
            population: info.population,
            budget_ms: info.budget_ms,
            max_generations: info.max_generations,
            already_spent_seconds: info.already_spent_seconds,
            config_json: info.config_json,
            kingdom: Some(info.kingdom),
            hff_objectives: info.hff_objectives,
            ..RunStartFields::default()
        };
        RunTelemetry::start_with(out, run_id, fields, info.resume)
    }

    /// [`RunTelemetry::start`] with a `run_start` the caller fills in full (a kingdom with fixed facts of its own,
    /// such as the WGSL fits' `wgsl` block). `append` is a resume.
    pub fn start_with(out: &std::path::Path, run_id: String, fields: RunStartFields, append: bool) -> Result<RunTelemetry, String> {
        std::fs::create_dir_all(out).map_err(|e| format!("telemetry directory {}: {e}", out.display()))?;
        let path = out.join(STREAM_FILE);
        let path_text = path.to_str().ok_or_else(|| format!("telemetry path {} is not UTF-8", path.display()))?;
        let writer = if append { Writer::append(path_text, run_id, fields) } else { Writer::create(path_text, run_id, fields) }?;
        let run_file = crate::run_file::RunFile::create(&out.join(crate::run_file::RUN_FILE))?;
        Ok(RunTelemetry { writer, dir: out.to_path_buf(), run_file, best_ever: None, individuals: 0, last_generation: 0, ended: false })
    }

    /// The run's output directory.
    pub fn dir(&self) -> &std::path::Path {
        &self.dir
    }

    /// The stream's path.
    pub fn stream_path(&self) -> std::path::PathBuf {
        self.dir.join(STREAM_FILE)
    }

    /// Whether the operator asked the run to stop: its `killme.trigger` was deleted. A loop checks this between
    /// generations and ends with [`RunTelemetry::end`]`("killed", …)`.
    pub fn stop_requested(&self) -> bool {
        !self.run_file.present()
    }

    /// THE WALL-CLOCK GUARD: whether the run has spent its `budget_ms` (never, with no budget). A loop checks this
    /// between generations and ends with [`RunTelemetry::end`]`("time", …)`.
    pub fn out_of_time(&self, budget_ms: u64) -> bool {
        budget_ms > 0 && self.writer.elapsed_ms() >= budget_ms
    }

    /// The writer, for the records a kingdom writes beyond the generic ones (events, discoveries, models, a
    /// snapshot with islands and cohorts).
    pub fn writer(&mut self) -> &mut Writer {
        &mut self.writer
    }

    /// The best HFF the run has held.
    pub fn best_ever(&self) -> Option<f64> {
        self.best_ever
    }

    /// Restore what a checkpoint carried (a resume): the best ever and the individuals evaluated so far.
    pub fn restore(&mut self, best_ever: Option<f64>, individuals: u64) {
        self.best_ever = best_ever;
        self.individuals = individuals;
    }

    /// The candidate evaluations the run has counted.
    pub fn individuals(&self) -> u64 {
        self.individuals
    }

    /// Write one generation as a snapshot: best, mean and median HFF, unscored rows, best ever, diversity, angle
    /// groups and extras, as one island holding the whole population. Every generation is written (the runners
    /// that use this run at most a few generations a second).
    pub fn generation(&mut self, g: GenerationStats<'_>) -> Result<(), String> {
        let mut scored: Vec<f64> = g.fitness.iter().copied().filter(|v| v.is_finite()).collect();
        scored.sort_by(f64::total_cmp);
        let rows = g.fitness.len() as u32;
        let nan_rows = rows - scored.len() as u32;
        let best = scored.first().copied();
        let mean = (!scored.is_empty()).then(|| scored.iter().sum::<f64>() / scored.len() as f64);
        let median = (!scored.is_empty()).then(|| {
            let n = scored.len();
            if n % 2 == 1 {
                scored[n / 2]
            } else {
                0.5 * (scored[n / 2 - 1] + scored[n / 2])
            }
        });
        if let Some(b) = best {
            self.best_ever = Some(self.best_ever.map_or(b, |e| e.min(b)));
        }
        self.individuals += g.evaluated;
        self.last_generation = g.generation;
        let global = Global {
            best_hff: best,
            best_ever_hff: self.best_ever,
            avg_hff: mean,
            nan_rows,
            genes: g.distinct.map(|_| rows),
            distinct_genes: g.distinct,
            median_hff: median,
            angle_groups: g.angle_groups,
            extras: g.extras,
            ..Global::default()
        };
        let island = IslandRow {
            id: "population".to_string(),
            kind: None,
            pair: 0,
            rows,
            best_hff: best,
            avg_hff: mean,
            nan_rows: Some(nan_rows),
            genes: None,
            distinct_genes: None,
            distinct_chromosomes: None,
            max_gene_copies: None,
            cohorts: Vec::new(),
        };
        self.writer.snapshot(g.generation, 0, SnapshotBody { global, islands: vec![island], global_cohorts: Vec::new(), pumps_since: 0 })
    }

    /// Write the hall of fame as it stands, best first.
    pub fn hall_of_fame(&mut self, generation: u32, rows: Vec<HofRow>) -> Result<(), String> {
        self.writer.hall_of_fame(generation, rows)
    }

    /// A MUTATION EVENT: the before/after text cut to [`EVENT_TEXT_LIMIT`], the full text in
    /// `OUT/event_text/<seq>.txt` when either was cut.
    pub fn mutation(&mut self, generation: u32, mut m: Mutation) -> Result<(), String> {
        let before = m.text_before.as_deref().map(cut_text);
        let after = m.text_after.as_deref().map(cut_text);
        if before.as_ref().is_some_and(|b| b.1) || after.as_ref().is_some_and(|a| a.1) {
            let dir = self.dir.join("event_text");
            std::fs::create_dir_all(&dir).map_err(|e| format!("event text {}: {e}", dir.display()))?;
            let file = dir.join(format!("{}.txt", self.writer.next_seq()));
            let full = format!("before:\n{}\n\nafter:\n{}\n", m.text_before.as_deref().unwrap_or(""), m.text_after.as_deref().unwrap_or(""));
            std::fs::write(&file, full).map_err(|e| format!("event text {}: {e}", file.display()))?;
            m.text_file = Some(format!("event_text/{}.txt", self.writer.next_seq()));
        }
        m.text_before = before.map(|b| b.0);
        m.text_after = after.map(|a| a.0);
        self.writer.mutation(generation, m)
    }

    /// A note for the operator (an event of kind `note`).
    pub fn note(&mut self, generation: u32, message: String) -> Result<(), String> {
        self.writer.event(generation, EventKind::Note, message, None, None)
    }

    /// The run's last record. `stopped_by` is the runner's own word: `n_gen`, `time`, `killed`, `early_stop`.
    pub fn end(mut self, stopped_by: &str, generations: u32, best_hff: Option<f64>) -> Result<(), String> {
        self.ended = true;
        let seconds = self.writer.elapsed_ms() as f64 / 1e3;
        let best = best_hff.or(self.best_ever);
        self.writer.run_end(self.last_generation, stopped_by, generations, self.individuals, best, seconds, None)?;
        self.run_file.remove();
        Ok(())
    }
}

impl Drop for RunTelemetry {
    /// A loop that unwound without [`RunTelemetry::end`] (an error returned through `?`, a panic) ended in error,
    /// and the stream says so. A failure to write it is printed: there is no caller left to return it to.
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        let seconds = self.writer.elapsed_ms() as f64 / 1e3;
        if let Err(e) = self.writer.run_end(self.last_generation, "error", self.last_generation, self.individuals, self.best_ever, seconds, None) {
            eprintln!("telemetry: the run ended in error and its run_end could not be written: {e}");
        }
        self.run_file.remove();
    }
}

// ---------------------------------------------------------------------------
// The reader
// ---------------------------------------------------------------------------

/// What a line turned out to be. A stream being read live is a file somebody
/// else is appending to, so the reader must distinguish a line that is BAD from
/// a line that is not finished yet — the first is skipped and counted, the
/// second is kept and completed by the next read.
#[derive(Clone, Debug)]
pub enum Parsed {
    Ok(Box<Record>),
    /// The line was complete and was not a record we understand: a truncated
    /// tail from a crash, a rotation's leftovers, a record kind from a newer
    /// engine. Counted, reported, and never fatal.
    Bad(String),
}

/// A tailing reader over a line-delimited stream.
///
/// It holds a byte offset rather than a file handle so a rotation — the file
/// getting SHORTER than what has been read, or replaced — is detectable at the
/// next poll: the engine truncates its telemetry file at `create`, so a second
/// fit writing to the same path is exactly that case, and a viewer that kept
/// reading from the old offset would show a frame stitched from two runs.
///
/// The partial-line buffer is the other half: `read_to_end` on a file being
/// appended to lands mid-line as a matter of course, and that tail is held until
/// the bytes that finish it arrive.
pub struct Tailer {
    path: std::path::PathBuf,
    offset: u64,
    /// BYTES, not a `String`. A read lands mid-line as a matter of course, and
    /// on a file being appended to it can land mid-CHARACTER: decoding the chunk
    /// before the line is whole would turn a split multi-byte character into a
    /// permanent replacement char and lose that record. The bytes are held and
    /// decoded once the newline that ends the line has arrived.
    partial: Vec<u8>,
    /// Bumped when the file shrank or vanished and reading restarted — the
    /// viewer shows it, because a silent restart looks like a stall.
    pub rotations: u32,
    pub bad_lines: u32,
}

impl Tailer {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Tailer {
        Tailer { path: path.into(), offset: 0, partial: Vec::new(), rotations: 0, bad_lines: 0 }
    }

    /// Everything that has arrived since the last call. Returns an empty vector
    /// when there is nothing new — including when the file does not exist yet,
    /// which is the ordinary case for a viewer started before its fit.
    pub fn poll(&mut self) -> Vec<Parsed> {
        use std::io::{Read, Seek, SeekFrom};
        let Ok(mut file) = std::fs::File::open(&self.path) else { return Vec::new() };
        let len = file.metadata().map_or(0, |m| m.len());
        // SHORTER than what we have read means the file was replaced under us.
        // Start again from the top, and say so.
        if len < self.offset {
            self.offset = 0;
            self.partial.clear();
            self.rotations += 1;
        }
        if len == self.offset {
            return Vec::new();
        }
        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            return Vec::new();
        }
        let mut buf = Vec::new();
        let Ok(read) = file.read_to_end(&mut buf) else { return Vec::new() };
        self.offset += read as u64;
        self.partial.extend_from_slice(&buf);
        let mut out = Vec::new();
        // A line is only a line once its terminator has arrived. Whatever is
        // after the last newline is next poll's problem — including a
        // multi-byte character the read cut in half, which is why the decode
        // happens HERE, per whole line, and not on the chunk above.
        while let Some(at) = self.partial.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.partial.drain(..=at).collect();
            let text = String::from_utf8_lossy(&line);
            let line = text.trim_end_matches(['\n', '\r']);
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Record>(line) {
                Ok(r) => out.push(Parsed::Ok(Box::new(r))),
                Err(e) => {
                    self.bad_lines += 1;
                    out.push(Parsed::Bad(format!("{e}")));
                }
            }
        }
        out
    }
}

/// Parse a whole recorded stream. `--file` uses this; `--follow` uses `Tailer`.
/// Both end up in the same `WatchState`, which is the brief's requirement that
/// live and replay share one parser and one state machine.
pub fn parse_stream(text: &str) -> (Vec<Record>, u32) {
    let mut bad = 0;
    let records = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| match serde_json::from_str::<Record>(l) {
            Ok(r) => Some(r),
            Err(_) => {
                bad += 1;
                None
            }
        })
        .collect();
    (records, bad)
}

// ---------------------------------------------------------------------------
// The invariants
// ---------------------------------------------------------------------------

/// What a snapshot must be true of, per the brief. The engine checks these on
/// the way out (in its tests) and the viewer can check them on the way in; both
/// call the same function, so there is one definition of "this frame is
/// coherent" rather than two that drift.
///
/// The HFF comparison carries a tolerance on purpose: `global.best_hff` comes
/// from the f64 `Scored` the report ranks on, and the island minima come from
/// the device's f32 fitness vector. They are the same quantity measured at two
/// precisions, and demanding bit equality of them would be demanding that f32
/// and f64 agree.
pub fn check_snapshot(s: &Snapshot, population: u32, cohorts_on: bool) -> Vec<String> {
    let mut problems = Vec::new();
    let island_rows: u32 = s.islands.iter().map(|i| i.rows).sum();
    if island_rows != population {
        problems.push(format!("island rows sum to {island_rows}, population is {population}"));
    }
    if cohorts_on {
        let cohort_rows: u32 = s.global_cohorts.iter().map(|c| c.rows).sum();
        if cohort_rows != population {
            problems.push(format!("cohort rows sum to {cohort_rows}, population is {population}"));
        }
        // Each cohort's island counts sum to its global count — the check that
        // makes the global table attributable. ONLY when a split was emitted at
        // all: a producer that cannot recover which island a cohort's rows sat
        // on (the brief's own example record) emits `cohorts: []`, and holding
        // that to a sum it never claimed would turn an honest silence into a
        // failed invariant.
        if s.islands.iter().any(|i| !i.cohorts.is_empty()) {
            let mut per_island: BTreeMap<u32, u32> = BTreeMap::new();
            for island in &s.islands {
                for c in &island.cohorts {
                    *per_island.entry(c.id).or_default() += c.rows;
                }
            }
            for c in &s.global_cohorts {
                let seen = per_island.get(&c.id).copied().unwrap_or(0);
                if seen != c.rows {
                    problems.push(format!("cohort {} has {} rows globally but {seen} over the islands", c.id, c.rows));
                }
            }
        }
    }
    // The global best is the best island's best. Relative tolerance, because the
    // two numbers are an f64 and a reduction over f32s.
    if let Some(global) = s.global.best_hff {
        let island = s.islands.iter().filter_map(|i| i.best_hff).fold(f64::INFINITY, f64::min);
        if island.is_finite() && (global - island).abs() > 1e-5 * global.abs().max(island.abs()).max(1e-30) {
            problems.push(format!("global best {global:e} is not the minimum island best {island:e}"));
        }
    }
    problems
}

// ---------------------------------------------------------------------------
// Candidate evaluations
// ---------------------------------------------------------------------------

/// What submitted a candidate for scoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalSource {
    /// Variation's offspring, the pump's immigrants and promotions, the first
    /// generation: the population itself.
    Main,
    /// `Engine::fold_winners` (and the regex kingdom's simplify editor, which
    /// is its fold: fuller's rewrite landed in the pool).
    Fold,
    /// `Engine::snap_winners` and `subexpr_winners` (the constant lattice).
    Snap,
    /// The typed snap (residual and subtree modes).
    TypedSnap,
    /// The beam: its scored mutants, and the wrap it lands.
    Beam,
    /// The share editor (FoldToDag). Not in the live loop yet: always 0.
    Share,
    /// fuller's post-search candidate generation, after the fit. 0 inside a
    /// fit; the law runner fills it from the kitchen sink's report.
    PostSearch,
}

/// Cumulative candidate evaluations by source.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalCounts {
    pub main: u64,
    pub fold: u64,
    pub snap: u64,
    pub typed_snap: u64,
    pub beam: u64,
    pub share: u64,
    pub post_search: u64,
}

impl EvalCounts {
    pub fn total(&self) -> u64 {
        self.main + self.fold + self.snap + self.typed_snap + self.beam + self.share + self.post_search
    }

    pub fn add(&mut self, source: EvalSource, n: u64) {
        let slot = match source {
            EvalSource::Main => &mut self.main,
            EvalSource::Fold => &mut self.fold,
            EvalSource::Snap => &mut self.snap,
            EvalSource::TypedSnap => &mut self.typed_snap,
            EvalSource::Beam => &mut self.beam,
            EvalSource::Share => &mut self.share,
            EvalSource::PostSearch => &mut self.post_search,
        };
        *slot += n;
    }

    /// `self + other`, for summing restarts.
    pub fn plus(&self, other: &EvalCounts) -> EvalCounts {
        EvalCounts {
            main: self.main + other.main,
            fold: self.fold + other.fold,
            snap: self.snap + other.snap,
            typed_snap: self.typed_snap + other.typed_snap,
            beam: self.beam + other.beam,
            share: self.share + other.share,
            post_search: self.post_search + other.post_search,
        }
    }

    /// One tab-separated line for a run log: `total per main fold snap
    /// typed_snap beam share post_search cap`, the shape the law runner parses.
    pub fn log_line(&self, per: u64, cap: u64) -> String {
        format!(
            "EVALUATIONS\t{}\t{per}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{cap}",
            self.total(),
            self.main,
            self.fold,
            self.snap,
            self.typed_snap,
            self.beam,
            self.share,
            self.post_search
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fuller_telemetry_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn info() -> RunInfo {
        RunInfo { kingdom: "test_kingdom".to_string(), dataset: "d".to_string(), seed: 3, population: 4, max_generations: 2, ..RunInfo::default() }
    }

    /// A run writes run_start, a snapshot a generation with the generic numbers, a hall of fame and run_end, and
    /// its kill trigger is there while it runs and gone after.
    #[test]
    fn a_run_writes_the_generic_stream() {
        let dir = scratch("generic");
        let mut t = RunTelemetry::start(&dir, "r".to_string(), info()).expect("the run starts");
        assert!(dir.join(crate::run_file::RUN_FILE).exists() && !t.stop_requested());
        let mut groups = BTreeMap::new();
        groups.insert("quality".to_string(), 0.25);
        for g in 0..=2u32 {
            let fitness = [0.5 - 0.1 * f64::from(g), 0.75, f64::NAN, 1.0];
            t.generation(GenerationStats { generation: g, fitness: &fitness, distinct: Some(3), angle_groups: groups.clone(), evaluated: 4, ..GenerationStats::default() })
                .expect("a generation");
        }
        t.hall_of_fame(2, vec![HofRow { rank: 1, hff: Some(0.3), born: 2, tree: "ab".to_string(), ..HofRow::default() }]).expect("hof");
        t.end("n_gen", 3, None).expect("the end");
        assert!(!dir.join(crate::run_file::RUN_FILE).exists(), "the trigger is removed at the end");
        let (records, bad) = parse_stream(&std::fs::read_to_string(dir.join(STREAM_FILE)).expect("the stream"));
        assert_eq!(bad, 0);
        let Record::RunStart(s) = &records[0] else { panic!("run_start first") };
        assert_eq!(s.kingdom.as_deref(), Some("test_kingdom"));
        let snaps: Vec<&Snapshot> = records.iter().filter_map(|r| if let Record::Snapshot(s) = r { Some(s) } else { None }).collect();
        assert_eq!(snaps.len(), 3);
        let last = &snaps[2].global;
        assert!((last.best_hff.expect("best") - 0.3).abs() < 1e-12);
        assert_eq!((last.nan_rows, last.distinct_genes, last.genes), (1, Some(3), Some(4)));
        assert_eq!(last.median_hff, Some(0.75));
        assert_eq!(last.angle_groups.get("quality"), Some(&0.25));
        assert!(check_snapshot(snaps[2], 4, false).is_empty());
        let Some(Record::RunEnd(e)) = records.last() else { panic!("run_end last") };
        assert_eq!((e.stopped_by.as_str(), e.individuals), ("n_gen", 12));
    }

    /// A loop that unwinds without `end` leaves a stream that says it ended in error.
    #[test]
    fn a_run_dropped_without_its_end_ended_in_error() {
        let dir = scratch("dropped");
        {
            let _t = RunTelemetry::start(&dir, "r".to_string(), info()).expect("the run starts");
        }
        let (records, _) = parse_stream(&std::fs::read_to_string(dir.join(STREAM_FILE)).expect("the stream"));
        let Some(Record::RunEnd(e)) = records.last() else { panic!("run_end last") };
        assert_eq!(e.stopped_by, "error");
    }

    /// Deleting the trigger is seen by the loop.
    #[test]
    fn deleting_the_trigger_requests_the_stop() {
        let dir = scratch("kill");
        let t = RunTelemetry::start(&dir, "r".to_string(), info()).expect("the run starts");
        std::fs::remove_file(dir.join(crate::run_file::RUN_FILE)).expect("deleted");
        assert!(t.stop_requested());
        t.end("killed", 0, None).expect("the end");
    }

    /// A mutation event carries its operator, lineage, texts and fitness; a long text is cut and kept in full in
    /// a side file; an event that is not a mutation carries none of the new fields.
    #[test]
    fn a_mutation_event_carries_its_texts_and_keeps_the_long_one_whole() {
        let dir = scratch("mutation");
        let mut t = RunTelemetry::start(&dir, "r".to_string(), info()).expect("the run starts");
        let long = "x".repeat(EVENT_TEXT_LIMIT + 50);
        let mut m = Mutation::new(EventKind::EgraphExact, "egraph_exact", "rewrite".to_string());
        m.row = Some(2);
        m.parents = vec![7, 9];
        m.text_before = Some(long.clone());
        m.text_after = Some("y".to_string());
        m.fitness_before = Some(0.5);
        m.fitness_after = Some(0.25);
        m.entered = Some(true);
        m.improved = Some(true);
        m.count = Some(3);
        t.mutation(0, m).expect("the event");
        t.note(0, "a note".to_string()).expect("a note");
        t.end("n_gen", 0, None).expect("the end");
        let text = std::fs::read_to_string(dir.join(STREAM_FILE)).expect("the stream");
        let (records, bad) = parse_stream(&text);
        assert_eq!(bad, 0);
        let events: Vec<&Event> = records.iter().filter_map(|r| if let Record::Event(e) = r { Some(e) } else { None }).collect();
        let e = events[0];
        assert_eq!((e.kind, e.operator.as_deref(), e.parents.clone(), e.count, e.improved), (EventKind::EgraphExact, Some("egraph_exact"), vec![7, 9], Some(3), Some(true)));
        assert_eq!(e.text_before.as_ref().map(|t| t.chars().count()), Some(EVENT_TEXT_LIMIT));
        let side = std::fs::read_to_string(dir.join(e.text_file.as_ref().expect("a side file"))).expect("the side file");
        assert!(side.contains(&long), "the side file holds the full text");
        let line = text.lines().find(|l| l.contains("a note")).expect("the note's line");
        assert!(!line.contains("operator") && !line.contains("parents") && !line.contains("text_"), "{line}");
    }

    /// The generic fields are omitted when unset: an old snapshot serialises as it did.
    #[test]
    fn the_generic_fields_are_additive() {
        let text = serde_json::to_string(&Global::default()).expect("serialises");
        assert!(!text.contains("median_hff") && !text.contains("angle_groups") && !text.contains("extras"), "{text}");
    }
}
