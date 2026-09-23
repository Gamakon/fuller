//! THE TELEMETRY STREAM — the machine-readable half of what a fit says about
//! itself, and the only production API for watching one run.
//!
//! The engine's human log is a scrolling prose report: it prints the state of
//! the search as sentences, three times over, and an operator watching a
//! 450-second fit cannot read it. That log stays exactly as it is — it is what
//! an offline diagnosis reads. This module adds a SECOND, separate stream: one
//! compact JSON object per line, versioned, numeric, written at the beat the
//! progress report already runs on. `hff-watch` repaints from it.
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    /// by that value. These arrive AFTER `run_end` — the fold runs in the final
    /// form, once the fit is over — which is the honest place for them and not a
    /// gap in the stream.
    Fold,
    /// Something the engine wants an operator to see that is not one of the above.
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
    pub seconds: f64,
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
/// Flushing every line is what makes `hff-watch --follow` work at all: an
/// unflushed line is not a line, and a viewer tailing a buffered file would show
/// a frame that is minutes old and call it live.
pub struct Writer {
    file: std::fs::File,
    run_id: String,
    seq: u64,
    started: std::time::Instant,
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
    pub fn create(path: &str, run_id: String, start: RunStartFields) -> Result<Writer, String> {
        let file = std::fs::File::create(path).map_err(|e| format!("telemetry file {path}: {e}"))?;
        let mut w = Writer { file, run_id, seq: 0, started: std::time::Instant::now(), last_snapshot: None, last_model: None };
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
        });
        w.write(&record)?;
        Ok(w)
    }

    /// The run's id — the viewer shows it, and a rotation is detected by it
    /// changing.
    pub fn run_id(&self) -> &str {
        &self.run_id
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
            elapsed_ms: self.started.elapsed().as_millis() as u64,
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
        self.write(&Record::Event(Event { header, kind, message, cohort, value, before: None, detail: None, nodes: None, row: None }))
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

    pub fn run_end(&mut self, generation: u32, stopped_by: &str, generations: u32, individuals: u64, best_hff: Option<f64>, seconds: f64) -> Result<(), String> {
        let header = self.header(generation);
        self.write(&Record::RunEnd(RunEnd {
            header,
            stopped_by: stopped_by.to_string(),
            generations,
            individuals,
            best_hff: best_hff.and_then(finite),
            seconds,
        }))
    }
}

/// The fixed facts a `run_start` carries, as one argument rather than fifteen.
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
fn civil_from_days(z: i64) -> (i64, u32, u32) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// THE FIXTURE: a trimmed recording of a real fit (seed 7014, bacres1), kept
    /// verbatim so a change to the record shape has to be made deliberately —
    /// this file is the schema's revision test. It has a `run_start`, several
    /// snapshots, events, a model and a `run_end`, and its last two lines are a
    /// GARBAGE line and a TRUNCATED one: a stream that ends badly is the ordinary
    /// case (a killed fit) and must cost the reader nothing but a count.
    const FIXTURE: &str = include_str!("../../tests/fixtures/telemetry_v1.jsonl");

    #[test]
    fn the_fixture_parses_and_the_bad_lines_are_only_counted() {
        let (records, bad) = parse_stream(FIXTURE);
        assert_eq!(bad, 2, "the fixture's two deliberate bad lines");
        assert!(records.len() > 5, "the fixture has a stream in it");
        assert!(matches!(records.first(), Some(Record::RunStart(_))), "a stream opens with run_start");
        assert!(records.iter().any(|r| matches!(r, Record::Snapshot(_))));
        assert!(records.iter().any(|r| matches!(r, Record::Model(_))));
        assert!(records.iter().any(|r| matches!(r, Record::RunEnd(_))), "and closes with run_end");
        for r in &records {
            assert_eq!(r.header().schema_version, SCHEMA_VERSION, "every record carries the version");
        }
    }

    /// THE BRIEF'S OWN EXAMPLE RECORD, pasted verbatim from the design brief.
    ///
    /// It is the line a later consumer — a gRPC viewer, somebody's script —
    /// will start from, so it has to parse as written: every field this schema
    /// added beyond it is `#[serde(default)]`, and the ones it cannot know
    /// (which island a cohort sat on, how many rows failed to score) come back
    /// as the silences they are rather than as zeros.
    const BRIEF_EXAMPLE: &str = r#"{"schema_version":1,"type":"snapshot","run_id":"bacres1-seed7015","seq":17,"timestamp_utc":"2026-09-23T00:00:00Z","elapsed_ms":376000,"generation":170,"budget_ms":450000,"global":{"best_hff":0.0004078684,"avg_hff":0.3721728,"mse_train":0.0000019019,"r2_train":0.9999997090,"r2_val":0.9999996989,"log10_p":-17.42},"islands":[{"id":"intake","rows":100000,"best_hff":null,"cohorts":[]},{"id":"champion","rows":100000,"best_hff":null,"cohorts":[]}],"global_cohorts":[{"id":0,"birth_generation":0,"rows":120000,"best_hff":0.0004079},{"id":100,"birth_generation":100,"rows":80000,"best_hff":0.005356}]}"#;

    #[test]
    fn the_briefs_own_example_record_parses() {
        let record: Record = serde_json::from_str(BRIEF_EXAMPLE).expect("the brief's example record must parse");
        let Record::Snapshot(s) = record else { panic!("not a snapshot") };
        assert_eq!(s.header.generation, 170);
        assert_eq!(s.header.seq, 17);
        assert_eq!(s.global.best_hff, Some(0.0004078684));
        assert_eq!(s.islands.len(), 2);
        assert_eq!(s.islands[0].id, "intake");
        // The fields the brief's record does not carry come back as SILENCE,
        // not as zero: the island has no best, no kind and no unscored count.
        assert_eq!(s.islands[0].best_hff, None);
        assert_eq!(s.islands[0].kind, None);
        assert_eq!(s.islands[0].nan_rows, None, "a missing count must not read as zero");
        assert!(s.islands[0].cohorts.is_empty());
        assert_eq!(s.global_cohorts.len(), 2);
        assert_eq!((s.global_cohorts[1].id, s.global_cohorts[1].rows), (100, 80000));
        // And it passes the invariants: the island rows sum to the population,
        // the cohort rows sum to the population, and the per-cohort island check
        // is SKIPPED because no split was emitted — holding a producer to a sum
        // it never claimed would turn an honest silence into a failure.
        let problems = check_snapshot(&s, 200_000, true);
        assert!(problems.is_empty(), "the brief's own record fails our invariants: {problems:?}");
    }

    #[test]
    fn the_sequence_and_the_generations_only_go_forwards() {
        let (records, _) = parse_stream(FIXTURE);
        let mut seq = None;
        let mut generation = 0;
        for r in &records {
            let h = r.header();
            if let Some(previous) = seq {
                assert!(h.seq > previous, "seq went backwards at {}", h.seq);
            }
            seq = Some(h.seq);
            assert!(h.generation >= generation, "generation went backwards at {}", h.generation);
            generation = h.generation;
        }
    }

    /// THE BRIEF'S INVARIANTS, over every snapshot of a real recording: island
    /// rows sum to the population, cohort rows sum to the population, each
    /// cohort's island counts sum to its global count, and the global best is the
    /// minimum island best.
    #[test]
    fn every_snapshot_of_the_fixture_is_coherent() {
        let (records, _) = parse_stream(FIXTURE);
        let start = records
            .iter()
            .find_map(|r| match r {
                Record::RunStart(s) => Some(s.clone()),
                _ => None,
            })
            .expect("the fixture's run_start");
        let mut checked = 0;
        for r in &records {
            if let Record::Snapshot(s) = r {
                let problems = check_snapshot(s, start.population, start.cohort_merge > 0);
                assert!(problems.is_empty(), "generation {}: {problems:?}", s.header.generation);
                checked += 1;
            }
        }
        assert!(checked >= 3, "the fixture has snapshots to check");
    }

    /// A metric that is not defined is `null` on the wire, and `null` comes back
    /// as None — never the string "—", which belongs to the viewer.
    #[test]
    fn undefined_metrics_are_null_and_never_a_dash() {
        assert_eq!(finite(f64::NAN), None);
        assert_eq!(finite(f64::INFINITY), None);
        assert_eq!(finite(0.5), Some(0.5));
        let row = CohortRow { id: 7, birth_generation: 7, rows: 3, best_hff: finite(f64::INFINITY), nan_rows: 3 };
        let json = serde_json::to_string(&row).expect("serialise");
        assert!(json.contains("\"best_hff\":null"), "{json}");
        assert!(!json.contains('—'), "the dash is the viewer's, not the schema's");
        let back: CohortRow = serde_json::from_str(&json).expect("round trip");
        assert_eq!(back.best_hff, None);
        assert_eq!(back.nan_rows, 3);
    }

    /// A REAL RECORDING OF A FIT THAT DISCOVERED THINGS: trimmed from a
    /// 30-second bacres1 run (seed 7014) that made 166 snap substitutions and
    /// one near-constant fold. [`FIXTURE`] predates both kinds, so it cannot
    /// check that they are ordered, parse or carry their numbers — this one can,
    /// and it is a recording rather than a hand-built stream, so a change to the
    /// record shape has to be made against what the engine actually writes.
    const DISCOVERIES: &str = include_str!("../../tests/fixtures/telemetry_v1_discoveries.jsonl");

    /// THE STREAM'S GENERATIONS ONLY GO FORWARDS, over a recording that has
    /// discoveries in it.
    ///
    /// A snap is stamped with THE GENERATION IT HAPPENED AT, which is earlier
    /// than the beat that reports it — snap fires every 20 generations and a
    /// snapshot lands every 30-odd. Writing them inside the snapshot's throttled
    /// block put `snap gen 940` after `cohort_born gen 960` and every one of a
    /// run's 166 snap events went backwards. They are written before the beat's
    /// own records now, which is also where an event belongs: it must not wait
    /// for a frame.
    #[test]
    fn a_recording_with_discoveries_in_it_still_only_goes_forwards() {
        let (records, bad) = parse_stream(DISCOVERIES);
        assert_eq!(bad, 0, "the recording is clean");
        let (mut seq, mut generation) = (None, 0);
        for r in &records {
            let h = r.header();
            if let Some(previous) = seq {
                assert!(h.seq > previous, "seq went backwards at {}", h.seq);
            }
            seq = Some(h.seq);
            assert!(h.generation >= generation, "generation went backwards at seq {}: {} < {generation}", h.seq, h.generation);
            generation = h.generation;
            assert_eq!(h.schema_version, SCHEMA_VERSION);
        }
        // It really holds both kinds, and their detail survived the round trip.
        let kinds: Vec<EventKind> = records.iter().filter_map(|r| match r {
            Record::Event(e) => Some(e.kind),
            _ => None,
        }).collect();
        assert!(kinds.contains(&EventKind::Snap), "no snap in the recording");
        assert!(kinds.contains(&EventKind::Fold), "no fold in the recording");
        let snap = records.iter().find_map(|r| match r {
            Record::Event(e) if e.kind == EventKind::Snap => Some(e.clone()),
            _ => None,
        }).expect("a snap");
        assert!(snap.before.is_some_and(f64::is_finite), "a snap with no literal: {snap:?}");
        assert!(snap.detail.as_ref().is_some_and(|d| !d.is_empty()), "a snap with no form: {snap:?}");
        assert!(snap.row.is_some(), "a snap with no row: {snap:?}");
        let fold = records.iter().find_map(|r| match r {
            Record::Event(e) if e.kind == EventKind::Fold => Some(e.clone()),
            _ => None,
        }).expect("a fold");
        assert!(fold.nodes.is_some_and(|n| n > 0), "a fold that removed nothing: {fold:?}");
        // THE FOLD COMES AFTER `run_end`, by design: it runs in the final form,
        // which is the caller's step once the fit has returned.
        let ended = records.iter().position(|r| matches!(r, Record::RunEnd(_))).expect("a run_end");
        let at = records.iter().position(|r| matches!(r, Record::Event(e) if e.kind == EventKind::Fold)).expect("a fold");
        assert!(at > ended, "the fold was written before run_end");

        // And every snapshot of it is coherent, by the same invariants.
        let start = records.iter().find_map(|r| match r {
            Record::RunStart(s) => Some(s.clone()),
            _ => None,
        }).expect("a run_start");
        for r in &records {
            if let Record::Snapshot(s) = r {
                let problems = check_snapshot(s, start.population, start.cohort_merge > 0);
                assert!(problems.is_empty(), "generation {}: {problems:?}", s.header.generation);
            }
        }
    }

    /// A DISCOVERY CARRIES ITS BEFORE AND AFTER AS VALUES, not only as prose —
    /// "numbers are numbers", so a viewer never parses `message` back into
    /// arithmetic. And every one of the fields it added is optional, so the
    /// fixtures written before they existed still parse and the schema version
    /// did not have to move.
    #[test]
    fn a_discovery_carries_its_before_and_after_and_adds_only_optional_fields() {
        // An event WITHOUT any of the new fields — every event ever written
        // before them — parses, and they come back as the silences they are.
        let old = r#"{"schema_version":1,"type":"event","run_id":"r","seq":3,"timestamp_utc":"2026-09-23T00:00:00Z","elapsed_ms":10,"generation":5,"kind":"new_best","message":"global best HFF 1e-3","cohort":null,"value":0.001}"#;
        let record: Record = serde_json::from_str(old).expect("an event without the discovery fields must still parse");
        let Record::Event(e) = record else { panic!("not an event") };
        assert_eq!((e.before, e.detail.clone(), e.nodes, e.row), (None, None, None, None));
        assert_eq!(e.value, Some(0.001));
        assert_eq!(SCHEMA_VERSION, 1, "optional fields must not have moved the version");

        // And a snap round-trips with both halves of the substitution as numbers.
        let snap = Event {
            header: Header { schema_version: SCHEMA_VERSION, run_id: "r".into(), seq: 4, timestamp_utc: now_utc(), elapsed_ms: 11, generation: 6 },
            kind: EventKind::Snap,
            message: "snap: 3.142857143 -> pi (row 7, gene 0)".into(),
            cohort: None,
            value: finite(std::f64::consts::PI),
            before: finite(22.0 / 7.0),
            detail: Some("pi".into()),
            nodes: None,
            row: Some(7),
        };
        let json = serde_json::to_string(&Record::Event(snap)).expect("serialise");
        assert!(json.contains("\"kind\":\"snap\""), "{json}");
        let Record::Event(back) = serde_json::from_str::<Record>(&json).expect("round trip") else { panic!() };
        assert_eq!(back.before, Some(22.0 / 7.0));
        assert_eq!(back.value, Some(std::f64::consts::PI));
        assert_eq!(back.detail.as_deref(), Some("pi"));
        assert_eq!(back.row, Some(7));

        // A fold's after is its value and its detail is the subtree that went.
        let fold = Event {
            header: Header { schema_version: SCHEMA_VERSION, run_id: "r".into(), seq: 5, timestamp_utc: now_utc(), elapsed_ms: 12, generation: 6 },
            kind: EventKind::Fold,
            message: "fold: 14 nodes [tanh(exp(cos(log(x))))] -> 1.000000000".into(),
            cohort: None,
            value: finite(1.0),
            before: None,
            detail: Some("tanh(exp(cos(log(x))))".into()),
            nodes: Some(14),
            row: None,
        };
        let json = serde_json::to_string(&Record::Event(fold)).expect("serialise");
        assert!(json.contains("\"kind\":\"fold\""), "{json}");
        let Record::Event(back) = serde_json::from_str::<Record>(&json).expect("round trip") else { panic!() };
        assert_eq!((back.nodes, back.value), (Some(14), Some(1.0)));
    }

    /// The reader holds a partial line until the bytes that finish it arrive, and
    /// a rotation (the file getting shorter) restarts it rather than stitching
    /// two runs into one frame.
    #[test]
    fn the_tailer_buffers_a_partial_line_and_notices_a_rotation() {
        let dir = std::env::temp_dir().join("fuller-telemetry-tailer");
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let path = dir.join("stream.jsonl");
        let first = FIXTURE.lines().next().expect("the run_start line");
        // Half a line: nothing is a record yet.
        let (head, tail) = first.split_at(first.len() / 2);
        std::fs::write(&path, head).expect("write the head");
        let mut tailer = Tailer::new(&path);
        assert!(tailer.poll().is_empty(), "half a line is not a line");
        std::fs::write(&path, format!("{first}\n")).expect("write the whole line");
        let got = tailer.poll();
        assert_eq!(got.len(), 1, "the line completed");
        assert!(matches!(got[0], Parsed::Ok(_)));
        assert_eq!(tail.len() + head.len(), first.len());
        // A shorter file is a new run: read it from the top again.
        std::fs::write(&path, "\n").expect("truncate");
        tailer.poll();
        assert_eq!(tailer.rotations, 1, "the rotation was noticed");
        std::fs::remove_file(&path).expect("remove the stream");
        std::fs::remove_dir(&dir).expect("remove its directory");
    }

    /// A READ THAT LANDS MID-CHARACTER loses nothing. A stream being appended to
    /// can be read with a multi-byte character cut in half, and decoding the
    /// chunk rather than the finished line would turn it into a replacement
    /// character permanently — the record would parse, with a corrupted string
    /// in it, which is worse than a bad line because nothing would say so.
    #[test]
    fn a_read_that_splits_a_character_loses_nothing() {
        let dir = std::env::temp_dir().join("fuller-telemetry-split-char");
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let path = dir.join("stream.jsonl");
        // A dataset name with a multi-byte character in it.
        let w = Writer::create(
            path.to_str().expect("a path"),
            "run-é".to_string(),
            RunStartFields {
                dataset: "données-µ.tsv".to_string(),
                seed: 1,
                population: 4,
                n_pairs: 1,
                pop_intake: 2,
                pop_champion: 2,
                float_zone: 0,
                n_train: 1,
                n_val: 1,
                n_extrap: 0,
                budget_ms: 1,
                max_generations: 1,
                cohort_merge: 0,
                pump_every: 1,
                progress_every: 1,
            },
        )
        .expect("a stream");
        // The run_start is flushed as it is written, so the file is whole here;
        // the writer is dropped so nothing holds the handle while the test
        // rewrites the file under it.
        assert_eq!(w.run_id(), "run-é");
        drop(w);
        let whole = std::fs::read(&path).expect("the stream's bytes");
        // Cut inside the last multi-byte character before the newline.
        let cut = whole.iter().rposition(|b| *b >= 0x80).expect("a multi-byte character");
        std::fs::write(&path, &whole[..cut + 1]).expect("half a character");
        let mut tailer = Tailer::new(&path);
        assert!(tailer.poll().is_empty(), "half a line is not a line");
        std::fs::write(&path, &whole).expect("the rest");
        let got = tailer.poll();
        assert_eq!(got.len(), 1);
        match &got[0] {
            Parsed::Ok(r) => match r.as_ref() {
                Record::RunStart(s) => assert_eq!(s.dataset, "données-µ.tsv", "the split character was corrupted"),
                other => panic!("{other:?}"),
            },
            Parsed::Bad(e) => panic!("the split character made the line unparseable: {e}"),
        }
        assert_eq!(tailer.bad_lines, 0);
        std::fs::remove_file(&path).expect("remove the stream");
        std::fs::remove_dir(&dir).expect("remove its directory");
    }

    /// A malformed line is counted and skipped; the reader keeps going.
    #[test]
    fn a_malformed_line_is_counted_not_fatal() {
        let dir = std::env::temp_dir().join("fuller-telemetry-malformed");
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let path = dir.join("stream.jsonl");
        let good = FIXTURE.lines().next().expect("the run_start line");
        std::fs::write(&path, format!("{good}\n{{not json\n{good}\n")).expect("write");
        let mut tailer = Tailer::new(&path);
        let got = tailer.poll();
        assert_eq!(got.len(), 3);
        assert_eq!(tailer.bad_lines, 1);
        assert!(matches!(got[0], Parsed::Ok(_)) && matches!(got[1], Parsed::Bad(_)) && matches!(got[2], Parsed::Ok(_)));
        std::fs::remove_file(&path).expect("remove the stream");
        std::fs::remove_dir(&dir).expect("remove its directory");
    }

    /// The timestamp is a real RFC-3339 date, not a fudged one: the epoch and a
    /// known later second, by hand.
    #[test]
    fn the_timestamp_is_rfc_3339_utc() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        // A leap day, which is where a hand-rolled calendar goes wrong.
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        let now = now_utc();
        assert_eq!(now.len(), 20, "{now}");
        assert!(now.ends_with('Z') && now.contains('T'), "{now}");
    }
}
