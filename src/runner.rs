//! THE RUNNER STANDARD'S SHARED PIECES (`docs/STANDARD_runners.md`), one home for every evolution runner in fuller
//! and phylu: the two-level HFF, the checkpoint, the ledger and the hall of fame. The stream and the kill trigger are
//! [`crate::telemetry::RunTelemetry`] and [`crate::run_file`].

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::telemetry::HofRow;

// ---------------------------------------------------------------------------
// Two-level HFF
// ---------------------------------------------------------------------------

/// The scaled error HFF's log scale calls zero (phylu's engine uses the same floor).
pub const HFF_LOG_FLOOR: f64 = 1e-12;

/// One individual's two-level HFF: the fitness selection ranks on, and the two group angles it is made of (radians,
/// each in `[0, π/2]` for inputs in `[0, 1]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoLevel {
    pub fitness: f64,
    pub quality: f64,
    pub speed: f64,
}

/// An input on `[0, 1]` on HFF's log scale: `1e-12` and below is 0, 1 is 1.
pub fn log_scaled(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    if x <= HFF_LOG_FLOOR {
        0.0
    } else {
        1.0 + x.log10() / -HFF_LOG_FLOOR.log10()
    }
}

/// THE TWO-LEVEL HFF (Andrew, 2026-10-22): the hff library's TrueNorth over the quality inputs gives `θ_quality`,
/// over the speed inputs `θ_speed`, and the fitness is TrueNorth over `(θ_quality/(π/2), θ_speed/(π/2))`. Every
/// measurement is its own input inside its group, nothing averaged, so the input counts do not decide the balance.
/// The inputs are on `[0, 1]`, 0 best; the caller scales them (a quality input usually with [`log_scaled`], so any
/// error at all weighs). An empty group's angle is 0. A non-finite input makes that group's angle π (TrueNorth's
/// own answer for an input it cannot place).
pub fn two_level_hff(quality: &[f64], speed: &[f64]) -> TwoLevel {
    let angle = |x: &[f64]| if x.is_empty() { 0.0 } else { crate::score::truenorth_angle(x) };
    let quality = angle(quality);
    let speed = angle(speed);
    let half_pi = std::f64::consts::FRAC_PI_2;
    let fitness = crate::score::truenorth_angle(&[quality / half_pi, speed / half_pi]);
    TwoLevel { fitness, quality, speed }
}

/// THE FITNESS STAGES (Andrew, 2026-10-27): evolve on QUALITY ONLY until the picture (the answer) is perfect, and
/// only then add performance. Stage 1 ranks on the quality group's angle alone, with no speed inputs and no timing
/// runs; stage 2 is the two-level HFF. The switch is made for the whole population once the best individual is
/// perfect on every quality case (training and held-out), to the runner's stated tolerance, and is logged as a
/// `stage_change` event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
pub enum FitnessStage {
    /// Stage 1: the quality group alone.
    Quality,
    /// Stage 2: the two-level HFF of the quality and speed groups.
    QualityAndSpeed,
}

impl FitnessStage {
    /// 1 or 2, for the stream's extras.
    pub fn number(self) -> u32 {
        match self {
            FitnessStage::Quality => 1,
            FitnessStage::QualityAndSpeed => 2,
        }
    }

    /// Whether this stage measures speed at all (stage 1 runs no timing).
    pub fn measures_speed(self) -> bool {
        self == FitnessStage::QualityAndSpeed
    }

    /// The stage after this generation: stage 2 once the best individual is perfect on every quality case; a
    /// stage never goes back.
    pub fn next(self, best_is_perfect: bool) -> FitnessStage {
        if best_is_perfect {
            FitnessStage::QualityAndSpeed
        } else {
            self
        }
    }
}

/// The fitness of a stage: stage 1 is TrueNorth over the quality inputs (the speed inputs must be empty, and the
/// speed angle is 0); stage 2 is [`two_level_hff`].
pub fn staged_hff(stage: FitnessStage, quality: &[f64], speed: &[f64]) -> Result<TwoLevel, String> {
    match stage {
        FitnessStage::Quality => {
            if !speed.is_empty() {
                return Err("our software has a bug: speed inputs in stage 1 (quality only)".into());
            }
            let q = if quality.is_empty() { 0.0 } else { crate::score::truenorth_angle(quality) };
            Ok(TwoLevel { fitness: q, quality: q, speed: 0.0 })
        }
        FitnessStage::QualityAndSpeed => Ok(two_level_hff(quality, speed)),
    }
}

/// THE TIMING SLOTS (the scheme of phylu `evolve::image_fitness::TimingStore`): every timed pair a program has had,
/// kept by its text, up to `slots` of them. A program scored again adds pairs to its old ones (never replaces them),
/// and its HFF reads exactly `slots` speed inputs, one a pair: the scorer times [`TimingSlots::missing`] more pairs
/// before it scores, so no slot is padded and no mean or median is taken before HFF. A full program is never timed
/// again, so re-scoring it across generations gives the same speed inputs.
#[derive(Debug, Clone, Default)]
pub struct TimingSlots {
    slots: usize,
    by_text: std::collections::HashMap<String, Vec<f64>>,
}

impl TimingSlots {
    pub fn new(slots: usize) -> TimingSlots {
        TimingSlots { slots, by_text: std::collections::HashMap::new() }
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// The pairs `text` still needs to fill its slots.
    pub fn missing(&self, text: &str) -> usize {
        self.slots - self.by_text.get(text).map_or(0, Vec::len)
    }

    /// Add newly timed pairs (the ratios candidate / yardstick) of `text`; more than its empty slots is our bug.
    pub fn add(&mut self, text: &str, ratios: &[f64]) -> Result<(), String> {
        if ratios.len() > self.missing(text) {
            return Err(format!("our software has a bug: {} new timed pairs for a program with {} empty slots", ratios.len(), self.missing(text)));
        }
        self.by_text.entry(text.to_string()).or_default().extend_from_slice(ratios);
        Ok(())
    }

    /// The ratios of `text` when every slot is filled; a program with an empty slot is our software's bug here.
    pub fn full(&self, text: &str) -> Result<&[f64], String> {
        match self.by_text.get(text) {
            Some(r) if r.len() == self.slots => Ok(r),
            r => Err(format!("our software has a bug: a program scored with {} of {} timing slots filled", r.map_or(0, Vec::len), self.slots)),
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// The checkpoint's file name inside a run's output directory.
pub const CHECKPOINT_FILE: &str = "checkpoint.json";

/// Write `state` as `out/checkpoint.json`, atomically (a temporary file renamed over the old one), so a run killed
/// mid-write leaves the previous checkpoint whole.
pub fn save_checkpoint<T: Serialize>(out: &Path, state: &T) -> Result<(), String> {
    let path = out.join(CHECKPOINT_FILE);
    let tmp = out.join(format!("{CHECKPOINT_FILE}.tmp"));
    let text = serde_json::to_string(state).map_err(|e| format!("checkpoint: {e}"))?;
    std::fs::write(&tmp, text).map_err(|e| format!("checkpoint {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("checkpoint {}: {e}", path.display()))
}

/// Read `out/checkpoint.json`. A resume with no checkpoint is an error: resuming nothing would silently start again.
pub fn load_checkpoint<T: DeserializeOwned>(out: &Path) -> Result<T, String> {
    let path = out.join(CHECKPOINT_FILE);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("resume: {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("resume: {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Ledger
// ---------------------------------------------------------------------------

/// The ledger's file name inside a run's output directory.
pub const LEDGER_FILE: &str = "ledger.tsv";

/// THE LEDGER AND LINEAGE: one line per scored individual — the generation, its id, its parents' ids, the operator
/// that made it, its fitness and its text — so any individual can be traced to where it came from.
pub struct Ledger {
    file: std::fs::File,
}

impl Ledger {
    /// Open `out/ledger.tsv`: truncated with its header for a fresh run, appended to on a resume.
    pub fn open(out: &Path, resume: bool) -> Result<Ledger, String> {
        let path = out.join(LEDGER_FILE);
        let fresh = !resume || !path.exists();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(!fresh)
            .truncate(fresh)
            .open(&path)
            .map_err(|e| format!("ledger {}: {e}", path.display()))?;
        if fresh {
            writeln!(file, "generation\tid\tparents\toperator\tfitness\ttext").map_err(|e| format!("ledger: {e}"))?;
        }
        Ok(Ledger { file })
    }

    /// One individual. Tabs and newlines in the text are written as spaces.
    pub fn record(&mut self, generation: u32, id: u64, parents: &[u64], operator: &str, fitness: f64, text: &str) -> Result<(), String> {
        let parents = parents.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
        let text = text.replace(['\t', '\n', '\r'], " ");
        writeln!(self.file, "{generation}\t{id}\t{}\t{operator}\t{fitness}\t{text}", if parents.is_empty() { "-" } else { &parents }).map_err(|e| format!("ledger: {e}"))
    }

    /// Flush at the end of a generation.
    pub fn flush(&mut self) -> Result<(), String> {
        self.file.flush().map_err(|e| format!("ledger: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Hall of fame
// ---------------------------------------------------------------------------

/// One entry of the hall of fame.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct FameEntry {
    /// The entry's identity: individuals with the same key are one entry (the kingdom's text, a gene's hash).
    pub key: String,
    pub fitness: f64,
    pub born: u32,
    /// The kingdom's own text of the entry (a program, a formula, a kernel).
    pub text: String,
    pub nodes: Option<u32>,
    /// A picture of the entry's output, when the kingdom draws one (a path under the run's directory).
    pub picture: Option<String>,
    pub extras: BTreeMap<String, f64>,
}

/// THE HALL OF FAME: the best `capacity` distinct individuals the run has seen, best first, each written as text
/// (`out/hall_of_fame/rank<N>.txt`) with its picture beside it when it has one.
#[derive(Clone, Debug, Default, Serialize, serde::Deserialize)]
pub struct HallOfFame {
    capacity: usize,
    entries: Vec<FameEntry>,
}

impl HallOfFame {
    pub fn new(capacity: usize) -> HallOfFame {
        HallOfFame { capacity: capacity.max(1), entries: Vec::new() }
    }

    /// Offer an individual; returns whether the hall of fame changed. A key already in it keeps the better fitness.
    /// Ties keep the earlier entry, so the order is deterministic.
    pub fn offer(&mut self, entry: FameEntry) -> bool {
        if !entry.fitness.is_finite() {
            return false;
        }
        if let Some(at) = self.entries.iter().position(|e| e.key == entry.key) {
            if entry.fitness >= self.entries[at].fitness {
                return false;
            }
            self.entries.remove(at);
        } else if self.entries.len() >= self.capacity && self.entries.last().is_some_and(|w| entry.fitness >= w.fitness) {
            return false;
        }
        let at = self.entries.iter().position(|e| entry.fitness < e.fitness).unwrap_or(self.entries.len());
        self.entries.insert(at, entry);
        self.entries.truncate(self.capacity);
        true
    }

    /// The entries, best first.
    pub fn entries(&self) -> &[FameEntry] {
        &self.entries
    }

    /// Write every entry's text to `out/hall_of_fame/rank<N>.txt` and return the stream's rows.
    pub fn write(&self, out: &Path) -> Result<Vec<HofRow>, String> {
        let dir: PathBuf = out.join("hall_of_fame");
        std::fs::create_dir_all(&dir).map_err(|e| format!("hall of fame {}: {e}", dir.display()))?;
        let mut rows = Vec::with_capacity(self.entries.len());
        for (i, e) in self.entries.iter().enumerate() {
            let rank = i as u32 + 1;
            let file = dir.join(format!("rank{rank}.txt"));
            std::fs::write(&file, format!("fitness {}\nborn {}\nkey {}\n\n{}\n", e.fitness, e.born, e.key, e.text)).map_err(|e| format!("hall of fame {}: {e}", file.display()))?;
            rows.push(HofRow { rank, hff: Some(e.fitness), born: e.born, tree: e.key.clone(), nodes: e.nodes, gpu_ms: None, picture: e.picture.clone(), extras: e.extras.clone() });
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {

    /// The slots fill, accumulate and stop: a new program misses every slot, pairs added in two batches are kept in
    /// order, a full program reads the same ratios again and takes no more.
    #[test]
    fn timing_slots_accumulate_to_the_cap_and_a_full_program_reads_the_same() {
        let mut s = TimingSlots::new(11);
        assert_eq!(s.missing("p"), 11);
        assert!(s.full("p").is_err(), "0 of 11 slots is not a score");
        s.add("p", &[1.0, 0.9, 1.1]).unwrap();
        assert_eq!(s.missing("p"), 8);
        assert!(s.full("p").is_err(), "3 of 11 slots is not a score");
        s.add("p", &[1.0; 8]).unwrap();
        assert_eq!(s.missing("p"), 0);
        let r = s.full("p").unwrap().to_vec();
        assert_eq!(r.len(), 11);
        assert_eq!(&r[..3], &[1.0, 0.9, 1.1]);
        assert_eq!(s.full("p").unwrap(), &r[..]);
        assert!(s.add("p", &[1.0]).is_err(), "a full program takes no more pairs");
        assert_eq!(s.missing("q"), 11, "slots are kept by text");
    }

    #[test]
    fn stage_one_ranks_on_quality_alone_and_the_switch_is_one_way() {
        let q = [0.2, 0.0, 0.4];
        let one = staged_hff(FitnessStage::Quality, &q, &[]).unwrap();
        assert_eq!(one.fitness, crate::score::truenorth_angle(&q));
        assert_eq!((one.quality, one.speed), (one.fitness, 0.0));
        assert!(staged_hff(FitnessStage::Quality, &q, &[0.5]).is_err(), "stage 1 takes no speed inputs");
        let two = staged_hff(FitnessStage::QualityAndSpeed, &q, &[0.5, 0.5]).unwrap();
        assert_eq!(two, two_level_hff(&q, &[0.5, 0.5]));
        assert_eq!(FitnessStage::Quality.next(false), FitnessStage::Quality);
        assert_eq!(FitnessStage::Quality.next(true), FitnessStage::QualityAndSpeed);
        assert_eq!(FitnessStage::QualityAndSpeed.next(false), FitnessStage::QualityAndSpeed);
        assert_eq!((FitnessStage::Quality.number(), FitnessStage::QualityAndSpeed.number()), (1, 2));
        assert!(!FitnessStage::Quality.measures_speed() && FitnessStage::QualityAndSpeed.measures_speed());
    }

    use super::*;

    /// A perfect individual is 0; each group's error moves its own angle; worse in either group is a worse fitness.
    #[test]
    fn the_two_level_hff_ranks_each_group_on_its_own() {
        let perfect = two_level_hff(&[0.0, 0.0], &[0.0]);
        assert_eq!(perfect.fitness, 0.0);
        let slow = two_level_hff(&[0.0, 0.0], &[0.6]);
        let slower = two_level_hff(&[0.0, 0.0], &[0.8]);
        assert!(slow.quality == 0.0 && slow.speed > 0.0 && slower.fitness > slow.fitness);
        let wrong = two_level_hff(&[0.5, 0.0], &[0.6]);
        assert!(wrong.quality > 0.0 && wrong.fitness > slow.fitness);
        assert_eq!(log_scaled(1e-13), 0.0);
        assert_eq!(log_scaled(1.0), 1.0);
    }

    #[test]
    fn the_hall_of_fame_keeps_the_best_distinct_entries() {
        let e = |key: &str, fitness: f64| FameEntry { key: key.to_string(), fitness, born: 0, text: key.to_string(), nodes: None, picture: None, extras: BTreeMap::new() };
        let mut h = HallOfFame::new(2);
        assert!(h.offer(e("a", 0.5)));
        assert!(h.offer(e("b", 0.3)));
        assert!(!h.offer(e("c", 0.9)), "worse than a full hall of fame");
        assert!(!h.offer(e("a", 0.6)), "the same key, worse");
        assert!(h.offer(e("a", 0.1)));
        assert_eq!(h.entries().iter().map(|x| x.key.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        let dir = std::env::temp_dir().join(format!("fuller_hof_{}", std::process::id()));
        let rows = h.write(&dir).expect("written");
        assert_eq!((rows[0].rank, rows.len()), (1, 2));
        assert!(std::fs::read_to_string(dir.join("hall_of_fame/rank1.txt")).expect("text").contains("fitness 0.1"));
    }

    #[test]
    fn a_checkpoint_round_trips_and_a_ledger_appends_on_resume() {
        let dir = std::env::temp_dir().join(format!("fuller_ckpt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        save_checkpoint(&dir, &(3u32, vec![1u32, 2])).expect("saved");
        let back: (u32, Vec<u32>) = load_checkpoint(&dir).expect("loaded");
        assert_eq!(back, (3, vec![1, 2]));
        let mut l = Ledger::open(&dir, false).expect("ledger");
        l.record(0, 1, &[], "init", 0.5, "a\tb").expect("line");
        l.flush().expect("flush");
        drop(l);
        let mut l = Ledger::open(&dir, true).expect("ledger again");
        l.record(1, 2, &[1], "mutation", 0.4, "c").expect("line");
        l.flush().expect("flush");
        let text = std::fs::read_to_string(dir.join(LEDGER_FILE)).expect("ledger text");
        assert_eq!(text.lines().count(), 3, "{text}");
        assert!(text.contains("0\t1\t-\tinit\t0.5\ta b") && text.contains("1\t2\t1\tmutation"));
    }
}
