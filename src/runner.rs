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
/// speed angle is 0); stage 2 is [`two_level_hff`], GATED BY CORRECTNESS (Andrew, 2026-10-29): an exact program
/// (every quality input 0) scores the two-level angle, in `[0, π/2)`; an inexact one scores `π/2 + θ_quality`, so
/// no inexact program, however fast, ranks above an exact one, and inexact programs still rank by quality.
pub fn staged_hff(stage: FitnessStage, quality: &[f64], speed: &[f64]) -> Result<TwoLevel, String> {
    match stage {
        FitnessStage::Quality => {
            if !speed.is_empty() {
                return Err("our software has a bug: speed inputs in stage 1 (quality only)".into());
            }
            let q = if quality.is_empty() { 0.0 } else { crate::score::truenorth_angle(quality) };
            Ok(TwoLevel { fitness: q, quality: q, speed: 0.0 })
        }
        FitnessStage::QualityAndSpeed => {
            let t = two_level_hff(quality, speed);
            if quality.iter().all(|x| *x == 0.0) {
                Ok(t)
            } else {
                Ok(TwoLevel { fitness: std::f64::consts::FRAC_PI_2 + t.quality, ..t })
            }
        }
    }
}

/// THE TIMING SLOTS (the scheme of phylu `evolve::image_fitness::TimingStore`): every timed pair a program has had,
/// kept by its text, up to `slots` of them. A program scored again adds pairs to its old ones (never replaces them),
/// and its HFF reads exactly `slots` speed inputs, one a pair: the scorer times [`TimingSlots::missing`] more pairs
/// before it scores, so no slot is padded and no mean or median is taken before HFF. A full program is never timed
/// again, so re-scoring it across generations gives the same speed inputs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, serde::Deserialize)]
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
// The speed verdict, the operator mix and the operator tally (every runner that times or mixes operators)
// ---------------------------------------------------------------------------

/// The median of the values of `v` that are not NaN (NaN when there is none); an infinity counts.
pub fn median(v: &[f64]) -> f64 {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    s.sort_by(f64::total_cmp);
    match s.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => s[n / 2],
        n => 0.5 * (s[n / 2 - 1] + s[n / 2]),
    }
}

/// THE VERDICT of an exact program's re-time against the published one (`docs/PLAN_futhark_fitness_fixes.md` section
/// G, fault 12): ratios are the program's time over the published program's, one an interleaved pair; `aa` every
/// ratio of the A/A sets (the published program against itself).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeedVerdict {
    /// Faster: the median below the least A/A ratio AND at least 9 of 11 pairs faster (a sign test, one-sided
    /// p = 0.033; the study plan's rule of 2026-10-28 13:10Z).
    Beat,
    /// Within the A/A noise.
    Matched,
    /// The median above the greatest A/A ratio.
    Slower,
}

impl SpeedVerdict {
    /// The verdict as the reports write it.
    pub fn text(self) -> &'static str {
        match self {
            SpeedVerdict::Beat => "beat (exact and faster: the median below the least A/A ratio and at least 9 of 11 pairs faster)",
            SpeedVerdict::Slower => "exact but slower (the median above the greatest A/A ratio)",
            SpeedVerdict::Matched => "matched (exact, within the A/A noise)",
        }
    }
}

/// [`SpeedVerdict`] of `ratios` against the A/A ratios `aa`. "At least 9 of 11" scales with the count: at least
/// `9/11` of the pairs below 1.
pub fn speed_verdict(ratios: &[f64], aa: &[f64]) -> SpeedVerdict {
    let floor = aa.iter().copied().fold(f64::INFINITY, f64::min);
    let ceiling = aa.iter().copied().fold(0.0, f64::max);
    let m = median(ratios);
    let faster = ratios.iter().filter(|r| **r < 1.0).count();
    if m < floor && faster * 11 >= 9 * ratios.len() {
        SpeedVerdict::Beat
    } else if m > ceiling {
        SpeedVerdict::Slower
    } else {
        SpeedVerdict::Matched
    }
}

/// THE OPERATOR MIX (`--ops NAME=P,...`, Andrew, 2026-10-31): every child draws ONE operator by its probability, so
/// all of them run in the same generation and each child's operator is recorded. `allowed` are the runner's
/// operators; the probabilities are finite, at least 0, and sum to 1.
pub fn parse_ops(text: &str, allowed: &[&str]) -> Result<Vec<(String, f64)>, String> {
    let mut v: Vec<(String, f64)> = Vec::new();
    for part in text.split(',') {
        let (name, w) = part.split_once('=').ok_or_else(|| format!("--ops {part}: want NAME=PROBABILITY"))?;
        if !allowed.contains(&name) {
            return Err(format!("--ops: {name} is not an operator of this runner ({})", allowed.join(", ")));
        }
        if v.iter().any(|x| x.0 == name) {
            return Err(format!("--ops: {name} is named twice"));
        }
        let w: f64 = w.parse().map_err(|_| format!("--ops {part}: {w} is not a number"))?;
        if !(w >= 0.0 && w.is_finite()) {
            return Err(format!("--ops {part}: a probability must be finite and at least 0"));
        }
        v.push((name.to_string(), w));
    }
    let total: f64 = v.iter().map(|x| x.1).sum();
    if (total - 1.0).abs() > 1e-6 {
        return Err(format!("--ops: the probabilities sum to {total}, not 1"));
    }
    Ok(v)
}

/// The operator a child draws from the mix with `u` uniform on `[0, 1)`, among the operators other than `exclude`
/// (marriage, whose share of the children is set apart) with a probability above 0, renormalised; `None` when there
/// is none. `u` at the top of the range (rounding) takes the last such operator.
pub fn pick_op<'a>(mix: &'a [(String, f64)], exclude: &str, u: f64) -> Option<&'a str> {
    let rest: Vec<&(String, f64)> = mix.iter().filter(|x| x.0 != exclude && x.1 > 0.0).collect();
    let total: f64 = rest.iter().map(|x| x.1).sum();
    let mut r = u * total;
    for x in &rest {
        if r < x.1 {
            return Some(x.0.as_str());
        }
        r -= x.1;
    }
    rest.last().map(|x| x.0.as_str())
}

/// THE OPERATOR TALLY of one generation (decision-critical, logged every generation): each operator's children and
/// how many of them beat their best parent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperatorTally {
    by: BTreeMap<String, (usize, usize)>,
}

impl OperatorTally {
    /// One child of `op`; `beat` when it is strictly better than its best parent.
    pub fn add(&mut self, op: &str, beat: bool) {
        let e = self.by.entry(op.to_string()).or_default();
        e.0 += 1;
        e.1 += usize::from(beat);
    }

    /// `(operator, children, children better than their best parent)`, by operator name.
    pub fn entries(&self) -> Vec<(String, usize, usize)> {
        self.by.iter().map(|(k, (n, b))| (k.clone(), *n, *b)).collect()
    }

    /// `point 9 (2 better), crossover 8 (1 better)`: the log's form (`-` when no child was made).
    pub fn line(&self) -> String {
        if self.by.is_empty() {
            return "-".to_string();
        }
        self.by.iter().map(|(k, (n, b))| format!("{k} {n} ({b} better)")).collect::<Vec<_>>().join(", ")
    }

    /// `point=9/2;crossover=8/1`: one TSV cell (`-` when no child was made).
    pub fn cell(&self) -> String {
        if self.by.is_empty() {
            return "-".to_string();
        }
        self.by.iter().map(|(k, (n, b))| format!("{k}={n}/{b}")).collect::<Vec<_>>().join(";")
    }
}

// ---------------------------------------------------------------------------
// The slope check (docs/PLAN_futhark_fitness_fixes.md, section A)
// ---------------------------------------------------------------------------

/// Ranks from 1, ties sharing their average rank.
pub fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| v[*a].total_cmp(&v[*b]));
    let mut out = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for k in &idx[i..=j] {
            out[*k] = r;
        }
        i = j + 1;
    }
    out
}

/// Pearson's correlation; `None` when either side is constant or the lengths differ or are under 2.
pub fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    (sxx > 0.0 && syy > 0.0).then(|| (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0))
}

/// Spearman's rank correlation (Pearson's on the average ranks); `None` as [`pearson`].
pub fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    pearson(&ranks(x), &ranks(y))
}

/// THE SLOPE CHECK's verdict: whether a fitness (lower is better, 0 exact) has a slope toward the target, from the
/// fitness and the tree distance to the target of every one-edit neighbour of the target's gene and of random genes.
/// It passes only if BOTH:
///
/// * the neighbours rank above the random genes: the median inexact neighbour is better than at least
///   [`SLOPE_RANDOM_BEATEN`] of the random genes (ties count as not beaten), so a step from the target looks better
///   than nearly every random program;
/// * the fitness-distance correlation, Spearman's over every program of the check, is positive (further is worse).
///
/// A neighbour exact on every case (an equal program) has fitness 0 whatever the fitness is; it is in the
/// correlation but not in the neighbours' ranking.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct SlopeCheck {
    pub neighbours: usize,
    pub exact_neighbours: usize,
    pub random: usize,
    pub median_neighbour: f64,
    pub median_random: f64,
    pub best_random: f64,
    /// The share of random genes strictly worse than the median inexact neighbour.
    pub random_beaten_by_median_neighbour: f64,
    /// The shares of inexact neighbours strictly better than the median random gene and than the best one.
    pub neighbours_above_median_random: f64,
    pub neighbours_above_best_random: f64,
    /// The probability that an inexact neighbour is better than a random gene, ties counted a half.
    pub auc: f64,
    /// Spearman(fitness, distance) over every program, and over the random genes alone.
    pub fdc: f64,
    pub fdc_random: f64,
    pub pass: bool,
}

/// The share of random genes the median neighbour must beat ([`SlopeCheck`]): 95 %, a step from the target better
/// than 19 of 20 random programs. (Set at 99 % before any measurement and changed on 2026-10-28 after A1 to A3 were
/// seen: many neighbours give exactly an output some random gene also gives, and a fitness of the outputs must tie
/// them; on histogram 70 of 126 do, so about 99.4 % is the most any fitness can reach.)
pub const SLOPE_RANDOM_BEATEN: f64 = 0.95;

/// [`SlopeCheck`] of `(fitness, distance)` pairs: the target's one-edit neighbours and the random genes.
pub fn slope_check(neighbours: &[(f64, usize)], random: &[(f64, usize)]) -> Result<SlopeCheck, String> {
    let inexact: Vec<f64> = neighbours.iter().map(|n| n.0).filter(|f| *f > 0.0).collect();
    let rnd: Vec<f64> = random.iter().map(|r| r.0).collect();
    if inexact.is_empty() || rnd.is_empty() {
        return Err(format!("the slope check needs inexact neighbours and random genes ({} and {})", inexact.len(), rnd.len()));
    }
    if inexact.iter().chain(&rnd).any(|f| !f.is_finite()) {
        return Err("our software has a bug: a fitness in the slope check is not finite".into());
    }
    let median = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        let n = s.len();
        if n % 2 == 1 {
            s[n / 2]
        } else {
            0.5 * (s[n / 2 - 1] + s[n / 2])
        }
    };
    let share = |n: usize, of: usize| n as f64 / of as f64;
    let (mn, mr) = (median(&inexact), median(&rnd));
    let best = rnd.iter().cloned().fold(f64::INFINITY, f64::min);
    let beaten = share(rnd.iter().filter(|r| **r > mn).count(), rnd.len());
    let mut wins = 0.0;
    for a in &inexact {
        for b in &rnd {
            wins += if a < b { 1.0 } else if a == b { 0.5 } else { 0.0 };
        }
    }
    let all: Vec<(f64, usize)> = neighbours.iter().chain(random).copied().collect();
    let fit: Vec<f64> = all.iter().map(|p| p.0).collect();
    let dist: Vec<f64> = all.iter().map(|p| p.1 as f64).collect();
    let fdc = spearman(&fit, &dist).unwrap_or(0.0);
    let fdc_random = spearman(&rnd, &random.iter().map(|r| r.1 as f64).collect::<Vec<_>>()).unwrap_or(0.0);
    Ok(SlopeCheck {
        neighbours: inexact.len(),
        exact_neighbours: neighbours.len() - inexact.len(),
        random: rnd.len(),
        median_neighbour: mn,
        median_random: mr,
        best_random: best,
        random_beaten_by_median_neighbour: beaten,
        neighbours_above_median_random: share(inexact.iter().filter(|n| **n < mr).count(), inexact.len()),
        neighbours_above_best_random: share(inexact.iter().filter(|n| **n < best).count(), inexact.len()),
        auc: wins / (inexact.len() * rnd.len()) as f64,
        fdc,
        fdc_random,
        pass: beaten >= SLOPE_RANDOM_BEATEN && fdc > 0.0,
    })
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

/// One line of the ledger ([`Ledger::record_line`]).
pub struct LedgerLine<'a> {
    pub generation: u32,
    pub id: u64,
    pub parents: &'a [u64],
    pub operator: &'a str,
    pub fitness: f64,
    pub text: &'a str,
    /// The values of the extra columns ([`Ledger::open_with`]), in order.
    pub extra: &'a [String],
}

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

    /// [`Ledger::open`] with `extra` columns after the text (a runner's own measurements of each individual, such as
    /// the time each stage of its evaluation took); every line then carries them ([`Ledger::record_with`]).
    pub fn open_with(out: &Path, resume: bool, extra: &[&str]) -> Result<Ledger, String> {
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
            let more: String = extra.iter().map(|c| format!("\t{c}")).collect();
            writeln!(file, "generation\tid\tparents\toperator\tfitness\ttext{more}").map_err(|e| format!("ledger: {e}"))?;
        }
        Ok(Ledger { file })
    }

    /// One individual. Tabs and newlines in the text are written as spaces.
    pub fn record(&mut self, generation: u32, id: u64, parents: &[u64], operator: &str, fitness: f64, text: &str) -> Result<(), String> {
        self.record_line(&LedgerLine { generation, id, parents, operator, fitness, text, extra: &[] })
    }

    /// One individual with the values of the extra columns of [`Ledger::open_with`], in order.
    pub fn record_line(&mut self, l: &LedgerLine) -> Result<(), String> {
        let parents = l.parents.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
        let text = l.text.replace(['\t', '\n', '\r'], " ");
        let more: String = l.extra.iter().map(|v| format!("\t{}", v.replace(['\t', '\n', '\r'], " "))).collect();
        writeln!(self.file, "{}\t{}\t{}\t{}\t{}\t{text}{more}", l.generation, l.id, if parents.is_empty() { "-" } else { &parents }, l.operator, l.fitness).map_err(|e| format!("ledger: {e}"))
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
        self.write_in(out, "hall_of_fame")
    }

    /// [`HallOfFame::write`] into `out/<name>/` (a staged runner keeps one hall of fame a stage).
    pub fn write_in(&self, out: &Path, name: &str) -> Result<Vec<HofRow>, String> {
        let dir: PathBuf = out.join(name);
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

    /// "Beat" needs both the median below the least A/A ratio and 9 of 11 pairs faster; a median below the floor with
    /// only 8 of 11 faster is "matched"; a median above the greatest A/A ratio is "slower".
    #[test]
    fn the_speed_verdict_needs_the_median_below_the_floor_and_nine_of_eleven_faster() {
        let aa = [0.97, 1.0, 1.02, 0.99];
        let nine = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1];
        assert_eq!(speed_verdict(&nine, &aa), SpeedVerdict::Beat);
        let eight = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1, 1.1];
        assert_eq!(speed_verdict(&eight, &aa), SpeedVerdict::Matched, "the median is below the floor but only 8 of 11 are faster");
        let near = [0.98; 11];
        assert_eq!(speed_verdict(&near, &aa), SpeedVerdict::Matched, "11 of 11 faster but the median inside the A/A noise");
        assert_eq!(speed_verdict(&[1.1; 11], &aa), SpeedVerdict::Slower);
        assert!(SpeedVerdict::Beat.text().starts_with("beat"));
        assert_eq!(median(&[3.0, f64::NAN, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[f64::INFINITY, 1.0, 2.0]), 2.0, "an untimed ratio (infinite) counts");
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn the_operator_mix_parses_draws_by_probability_and_the_tally_counts_children_that_beat_their_parent() {
        let ops = ["point", "invert", "transpose", "crossover", "marriage"];
        let mix = parse_ops("point=0.5,invert=0.1,transpose=0.1,crossover=0.2,marriage=0.1", &ops).unwrap();
        assert_eq!(mix.len(), 5);
        assert!(parse_ops("point=0.5,invert=0.1", &ops).is_err(), "sums to 0.6");
        assert!(parse_ops("walk=1", &ops).is_err(), "not an operator");
        assert!(parse_ops("point=0.5,point=0.5", &ops).is_err(), "named twice");
        assert!(parse_ops("point=-1,invert=2", &ops).is_err(), "negative");
        // without marriage the rest renormalise to 0.5/0.9, 0.1/0.9, 0.1/0.9, 0.2/0.9
        assert_eq!(pick_op(&mix, "marriage", 0.0), Some("point"));
        assert_eq!(pick_op(&mix, "marriage", 0.55), Some("point"));
        assert_eq!(pick_op(&mix, "marriage", 0.56), Some("invert"));
        assert_eq!(pick_op(&mix, "marriage", 0.999_999), Some("crossover"));
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for i in 0..9000 {
            *counts.entry(pick_op(&mix, "marriage", (i as f64 + 0.5) / 9000.0).unwrap()).or_default() += 1;
        }
        assert_eq!(counts.get("point"), Some(&5000));
        assert_eq!(counts.get("crossover"), Some(&2000));
        assert!(pick_op(&[("marriage".to_string(), 1.0)], "marriage", 0.3).is_none());
        let mut t = OperatorTally::default();
        assert_eq!((t.line(), t.cell()), ("-".to_string(), "-".to_string()));
        t.add("point", true);
        t.add("point", false);
        t.add("crossover", false);
        assert_eq!(t.line(), "crossover 1 (0 better), point 2 (1 better)");
        assert_eq!(t.cell(), "crossover=1/0;point=2/1");
        assert_eq!(t.entries()[1], ("point".to_string(), 2, 1));
    }

    #[test]
    fn in_stage_two_an_exact_slow_program_ranks_above_every_inexact_fast_one() {
        let slow_exact = staged_hff(FitnessStage::QualityAndSpeed, &[0.0, 0.0], &[1.0; 11]).unwrap();
        for wrong in [[1e-9, 0.0], [0.5, 0.5], [1.0, 1.0]] {
            let fast_wrong = staged_hff(FitnessStage::QualityAndSpeed, &wrong, &[0.0; 11]).unwrap();
            assert!(slow_exact.fitness < fast_wrong.fitness, "{wrong:?}: {} !< {}", slow_exact.fitness, fast_wrong.fitness);
        }
        let near = staged_hff(FitnessStage::QualityAndSpeed, &[0.1, 0.0], &[0.0; 11]).unwrap();
        let far = staged_hff(FitnessStage::QualityAndSpeed, &[0.9, 0.9], &[0.0; 11]).unwrap();
        assert!(near.fitness < far.fitness, "inexact programs still rank by quality");
    }

    #[test]
    fn stage_one_ranks_on_quality_alone_and_the_switch_is_one_way() {
        let q = [0.2, 0.0, 0.4];
        let one = staged_hff(FitnessStage::Quality, &q, &[]).unwrap();
        assert_eq!(one.fitness, crate::score::truenorth_angle(&q));
        assert_eq!((one.quality, one.speed), (one.fitness, 0.0));
        assert!(staged_hff(FitnessStage::Quality, &q, &[0.5]).is_err(), "stage 1 takes no speed inputs");
        let exact = [0.0, 0.0, 0.0];
        let two = staged_hff(FitnessStage::QualityAndSpeed, &exact, &[0.5, 0.5]).unwrap();
        assert_eq!(two, two_level_hff(&exact, &[0.5, 0.5]));
        let inexact = staged_hff(FitnessStage::QualityAndSpeed, &q, &[0.5, 0.5]).unwrap();
        assert_eq!(inexact.fitness, std::f64::consts::FRAC_PI_2 + crate::score::truenorth_angle(&q));
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

    /// Ranks share ties; Spearman is 1 for any increasing map, -1 for a decreasing one, and `None` for a constant.
    #[test]
    fn the_rank_statistics_are_the_textbook_ones() {
        assert_eq!(ranks(&[3.0, 1.0, 3.0, 2.0]), vec![3.5, 1.0, 3.5, 2.0]);
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(spearman(&x, &[1.0, 8.0, 27.0, 64.0, 125.0]), Some(1.0));
        assert_eq!(spearman(&x, &[5.0, 4.0, 3.0, 2.0, 1.0]), Some(-1.0));
        assert_eq!(spearman(&x, &[2.0; 5]), None);
        let r = pearson(&[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0]).unwrap();
        assert!((r - 0.981_980_506_061_965_7).abs() < 1e-12, "{r}");
    }

    /// The slope check passes a fitness whose neighbours beat the random genes and whose fitness grows with the
    /// distance, and fails one whose neighbours sit in the random genes' band or that improves with distance.
    #[test]
    fn the_slope_check_passes_a_slope_and_fails_a_plateau_or_a_reversed_slope() {
        let random: Vec<(f64, usize)> = (0..200).map(|i| (1.0 + i as f64 / 1000.0, 10 + i / 10)).collect();
        let near: Vec<(f64, usize)> = (0..20).map(|i| (0.3 + i as f64 / 100.0, 1)).chain([(0.0, 1)]).collect();
        let c = slope_check(&near, &random).unwrap();
        assert!(c.pass && c.exact_neighbours == 1 && c.neighbours == 20, "{c:?}");
        assert_eq!((c.random_beaten_by_median_neighbour, c.neighbours_above_best_random, c.auc), (1.0, 1.0, 1.0));
        assert!(c.fdc > 0.5 && c.fdc_random > 0.9, "{c:?}");
        // The neighbours in the random genes' band: the median neighbour beats about half of them.
        let plateau: Vec<(f64, usize)> = (0..20).map(|i| (1.0 + i as f64 / 100.0, 1)).collect();
        let c = slope_check(&plateau, &random).unwrap();
        assert!(!c.pass && c.random_beaten_by_median_neighbour < 0.6, "{c:?}");
        // Better with distance (the review's fault 1), however good the neighbours are.
        let reversed: Vec<(f64, usize)> = (0..200).map(|i| (1.0 - i as f64 / 1000.0, 10 + i / 10)).collect();
        let far: Vec<(f64, usize)> = (0..20).map(|i| (0.5 + i as f64 / 100.0, 60)).collect();
        let c = slope_check(&far, &reversed).unwrap();
        assert!(!c.pass && c.fdc < 0.0, "{c:?}");
        assert!(slope_check(&[(0.0, 1)], &random).is_err(), "no inexact neighbour is no check");
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
