//! THE RUN CARD — the whole configuration of a fit, written at its START.
//!
//! Andrew: "I think we need to generate a run card. This is the parameters used
//! to start the run ... you can just go 'use card 26 and run this'. And then
//! when you're doing that, it produces a card. And so everybody just talks about
//! the cards after a while instead of the configurations one by one."
//!
//! WHAT THIS IS FOR, in one measurement. The best result on record is 75 of 133
//! laws. Finding out what produced it meant grepping the git log and then
//! reading settings a human had written into prose in `docs/EXPERIMENTS.md` —
//! and they turned out to be population 800 + 400, pump 100, cohort merge
//! 10,000, nothing like the 100,000 + 100,000, pump 20, merge 100 being run the
//! next day. FOUR RUNS were spent before anyone knew the configuration had
//! drifted that far. A card is the file that makes that a `diff`.
//!
//! THE CARD IS WRITTEN BY THE BINARY, not by the launcher script, and that is
//! the whole reliability argument: the script knows what it EXPORTED, the binary
//! knows what it is actually RUNNING after every environment override, every
//! positional argument and every default. `logs/bacres1_test.sh` used to echo a
//! hand-kept list that once said "pump 33" while the binary ran 20 — a line that
//! gets believed.
//!
//! WHAT IS ON A CARD
//!
//! * `config` — the WHOLE [`Config`], by `#[derive(Serialize)]`. Not a chosen
//!   subset: a hand-listed subset drifts from the engine the first time a knob
//!   is added, which is the failure this exists to end. (The telemetry stream's
//!   `run_start` carries about fifteen fields and so is not a record of how a
//!   run was configured; the card does not change it.)
//! * What `Config` genuinely DOES NOT HOLD but which changes the result: the
//!   dataset and its splits, SMOGD and SMOTE and SMOGD's noise multiplier (all
//!   of which live in `examples/evolve_fit.rs`, because the synthetic rows are
//!   generated from a dataset the engine never sees), and the restart count.
//! * `code` — the GIT COMMIT THE BINARY WAS BUILT FROM (see `build.rs`), so a
//!   card names the code as well as the settings.
//! * `derived` — what the engine WORKS OUT and a reader cannot infer: above all
//!   THE NUMBER OF HFF OBJECTIVES. The runs that recovered laws had SIX and the
//!   ones that did not had NINE, and no existing output said so. It is a
//!   function of the third block, `hff_without_validation`, `log_scale`,
//!   `redundancy` and `tower` together — five settings, so nobody reads it off
//!   the config by eye. Also the island layout and the real population.
//!
//! WHAT IS NOT ON A CARD: the six OUTPUT PATHS of [`Config`], which are
//! `#[serde(skip)]` there. See that type's own note — replaying them truncates
//! the original run's telemetry stream and resumes its checkpoint.
//!
//! REPLAY: `EVOLVE_CARD=path/to/card.json`. The card is the STARTING POINT and
//! the environment still WINS over it, so a card is something to vary from; the
//! card that run then writes records what ACTUALLY ran, overrides included.
//! Round-trip is the property that matters and [`tests`] asserts it.

use super::engine::{Config, Lane};
use serde::{Deserialize, Serialize};

/// The card format's version. A reader that does not know a version should say
/// so rather than guess at the fields.
pub const CARD_VERSION: u32 = 1;

/// A whole run's configuration: everything needed to start the same fit again.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Card {
    pub card_version: u32,
    /// THE SHORT STABLE NAME a card is talked about by — Andrew's "use card 26".
    /// It is the run's TAG, not a counter: a sequence number cannot be minted
    /// without a lock, and with several worktrees fitting at once two runs would
    /// both be card 26. A tag is chosen by whoever starts the run, is already
    /// what the log directory is called, and is already what the telemetry run
    /// id is set to.
    pub card_id: String,
    pub written_utc: String,
    pub code: Code,
    pub data: DataCard,
    pub synthetic: Synthetic,
    pub run: Run,
    /// THE WHOLE CONFIG, derived — never a hand-copied subset.
    pub config: Config,
    /// What the engine worked out from the config and the data together.
    pub derived: Derived,
}

/// The code the binary was built from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Code {
    /// Baked in at COMPILE time (`build.rs`), because that is the only reading
    /// of "the commit of this binary" that stays true: a `git rev-parse` at run
    /// time names whatever HEAD the checkout is on when the fit starts, which
    /// with several worktrees committing through a day is routinely not the
    /// source that was built.
    pub git_commit: String,
    /// Whether that tree had uncommitted changes, so the commit does not by
    /// itself identify the source.
    pub git_dirty: bool,
}

impl Code {
    /// What this binary was built from.
    pub fn of_this_binary() -> Code {
        Code {
            git_commit: env!("FULLER_GIT_COMMIT").to_string(),
            git_dirty: env!("FULLER_GIT_DIRTY") == "true",
        }
    }
}

/// The dataset and how it was cut. `Config` holds none of this — the engine is
/// handed columns and never sees a file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataCard {
    pub path: String,
    /// The most rows taken from the fitting part before the train/validation cut.
    pub max_rows: usize,
    /// The sixth positional argument `all`: the file IS already the training
    /// part, so no 25% is held back.
    pub use_all: bool,
    /// `EVOLVE_EDGE`, the harness's isolated held-out rows, when there was one.
    pub edge_path: Option<String>,
    /// A held-out test file the harness passed as positional argument 7.
    pub test_path: Option<String>,
    /// THE ROW COUNTS, which identify the cut as no path can.
    pub n_train: usize,
    pub n_val: usize,
    /// The third block: edge rows plus whatever SMOGD and SMOTE generated.
    pub n_third: usize,
    /// The unseen 25%, 0 when `use_all`.
    pub n_test: usize,
}

/// The synthetic third block. Generated in `evolve_fit` from the rows it was
/// handed, so these cannot be engine settings: they only exist once a dataset
/// does. `Config::smogd` says only that a third block is in HFF, never which of
/// the two made it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Synthetic {
    pub smogd: bool,
    pub smote: bool,
    /// The multiplier on the neighbours' variance (1 = the original).
    pub smogd_noise: f64,
}

/// The run's own frame, outside `Config` because a restart makes several configs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Run {
    /// The fit's seed. Each restart derives its own from it.
    pub seed: u32,
    /// The WHOLE time budget. `Config::max_seconds` is this divided by the
    /// restarts, so the two differ whenever there is more than one.
    pub budget_seconds: f64,
    /// The time split into this many independent searches (1 = one search).
    pub restarts: u32,
}

/// WHAT THE ENGINE WORKED OUT — the numbers a reader wants and cannot get from
/// the config by eye. Recorded, never read back: a replay recomputes them, and
/// a card whose derived numbers disagree with the engine's is a card from
/// different code, which `code.git_commit` already says.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Derived {
    /// HOW MANY OBJECTIVES HFF RANKS ON. The whole reason `derived` exists: the
    /// runs that recovered laws had six and the ones that did not had nine, and
    /// it is a function of five separate settings, so nobody spots it by eye.
    pub hff_objectives: usize,
    /// Every row the engine holds — `n_pairs * (pop_intake + float_zone +
    /// pop_champion)`, which is not `pop_intake + pop_champion`.
    pub population: u32,
    /// Each island as `lo..hi`, in the engine's own order: intake, champion,
    /// intake, champion, one pair after another.
    pub islands: Vec<IslandSpan>,
    /// `cohort_merge / pump_every` — how many cohorts VIRTUAL ALPS keeps apart
    /// at once, which is the number with a measurement behind it rather than the
    /// age. 0 when cohorts are off.
    pub live_cohorts: u32,
}

/// One island's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IslandSpan {
    pub lo: u32,
    pub hi: u32,
}

impl Card {
    /// Read a card. The error says which file and why, because a mistyped
    /// `EVOLVE_CARD` must not look like a run that simply took its defaults.
    ///
    /// # Errors
    /// When the file cannot be read or is not a card this version understands.
    pub fn load(path: &str) -> Result<Card, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("run card {path}: {e}"))?;
        let card: Card = serde_json::from_str(&text).map_err(|e| format!("run card {path}: {e}"))?;
        if card.card_version != CARD_VERSION {
            return Err(format!("run card {path}: version {}, this binary writes and reads {CARD_VERSION}", card.card_version));
        }
        Ok(card)
    }

    /// Write the card, creating the directory if it is not there.
    ///
    /// PRETTY, and that is a requirement rather than a courtesy: a card is read
    /// by people and `diff a.json b.json` is the thing we most want from two of
    /// them. serde writes a struct's fields in declaration order, so two cards
    /// from one binary line up and a plain `diff` shows only what changed.
    ///
    /// # Errors
    /// When the directory cannot be made or the file cannot be written.
    pub fn write(&self, path: &str) -> Result<(), String> {
        if let Some(dir) = std::path::Path::new(path).parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| format!("run card directory {}: {e}", dir.display()))?;
        }
        let mut text = serde_json::to_string_pretty(self).map_err(|e| format!("run card encode: {e}"))?;
        text.push('\n');
        std::fs::write(path, text).map_err(|e| format!("run card {path}: {e}"))
    }
}

/// `cohort_merge / pump_every`: the live cohort count, the number with the
/// measurement behind it. A beat of 0 is not a beat, so there are no bands.
pub fn live_cohorts(cohort_merge: u32, pump_every: u32) -> u32 {
    if pump_every == 0 {
        return 0;
    }
    cohort_merge / pump_every
}

/// THE ENVIRONMENT OVERRIDES, as one function over a config.
///
/// `env` is INJECTED rather than read from the process, and that is what makes
/// the whole card testable: "the environment wins over the card" and "an empty
/// environment leaves the card alone" are both assertions about this function,
/// with no global state for concurrent tests to race on.
///
/// EVERY OVERRIDE IS CONDITIONAL — `if let Some(v) = env(..)`, never
/// `unwrap_or(literal)`. That is what makes a card round-trip: an unconditional
/// assignment would quietly put `tower` back to false on a replay of a run that
/// had it on. With no card the behaviour is unchanged, because each literal that
/// was being assigned is the same value [`Config::srbench`] already holds.
///
/// Booleans keep their `v == "1"` reading, so `EVOLVE_TOWER=0` still means
/// explicitly off rather than "unset".
pub fn apply_env(config: &mut Config, env: &dyn Fn(&str) -> Option<String>) {
    let flag = |k: &str| env(k).map(|v| v == "1");
    // GENERIC over the target type, so the same reader serves a u32 beat, an
    // i32 constant bound and an f64 fraction and each call site's field decides
    // which. A closure cannot be generic — it would be pinned to whichever type
    // it was first used at, which is the error this replaced.
    fn num<T: std::str::FromStr>(env: &dyn Fn(&str) -> Option<String>, k: &str) -> Option<T> {
        env(k).and_then(|v| v.parse().ok())
    }

    //   EVOLVE_HFF_LOG_TRAIN / _VAL / _BLOCK3 = 1   that block enters HFF on the log scale
    //   EVOLVE_HFF_LOG=1                            validation and the third block both
    //
    // A block is on the log scale when its OWN variable says so or when
    // EVOLVE_HFF_LOG does; either being unset leaves the card's own answer.
    let both = flag("EVOLVE_HFF_LOG");
    let log = |k: &str, wide: Option<bool>, was: bool| -> bool {
        match (flag(k), wide) {
            (None, None) => was,
            (own, w) => own.unwrap_or(false) || w.unwrap_or(false),
        }
    };
    config.log_scale = [
        log("EVOLVE_HFF_LOG_TRAIN", None, config.log_scale[0]),
        log("EVOLVE_HFF_LOG_VAL", both, config.log_scale[1]),
        log("EVOLVE_HFF_LOG_BLOCK3", both, config.log_scale[2]),
    ];
    //   EVOLVE_HFF_NO_VAL=1             validation is left out of HFF: train + block three
    if let Some(v) = flag("EVOLVE_HFF_NO_VAL") {
        config.hff_without_validation = v;
    }
    //   EVOLVE_TOWER=1                  the tower objective (t_depth) joins HFF
    if let Some(v) = flag("EVOLVE_TOWER") {
        config.tower = v;
    }
    //   EVOLVE_VHEAD_EVERY / _START     the GROWING HEAD: the virtual head starts at START
    //                                   (12) and gains a position every EVERY generations
    //                                   (0 = off: the whole head from the start)
    if let Some(n) = num(env, "EVOLVE_VHEAD_EVERY") {
        config.vhead_every = n;
    }
    if let Some(n) = num(env, "EVOLVE_VHEAD_START") {
        config.vhead_start = n;
    }
    //   EVOLVE_STOP_LOG10_P             the stop bar's p-value half (the engine's default is -19;
    //                                   "inf" switches it off)
    if let Some(bar) = num(env, "EVOLVE_STOP_LOG10_P") {
        config.stop_log10_p = bar;
    }
    //   EVOLVE_BALANCED_TOURNAMENTS=1   the tournaments rank on hff's BALANCED pole (for
    //                                   diversity); the hall of fame, the stop bar and the
    //                                   report stay on TrueNorth
    if let Some(v) = flag("EVOLVE_BALANCED_TOURNAMENTS") {
        config.balanced_tournaments = v;
    }
    //   EVOLVE_HFF_ON_HOST=1            the HFF candidate walk runs on the HOST, the way it did
    //                                   before the kernel existed. For a parity check or a
    //                                   bisect, not for a benchmark: measured at 238 s of a
    //                                   450 s fit at population 200,000.
    if let Some(v) = flag("EVOLVE_HFF_ON_HOST") {
        config.hff_on_host = v;
    }
    //   EVOLVE_CHAMPION_TOURNAMENT      the CHAMPION island's tournament, as a fraction of it
    //                                   (unset = the same as the intake's). The champion island
    //                                   is an open knockout, and there pressure becomes a lock:
    //                                   measured on bacres1 at 2,000 + 2,000, cohort 160 held
    //                                   96.9% of it for 19,000 generations while being only 6%
    //                                   better than its nearest challenger.
    if let Some(f) = num(env, "EVOLVE_CHAMPION_TOURNAMENT") {
        config.champion_tournament_fraction = Some(f);
    }
    //   EVOLVE_ARRIVAL_CHILDREN         how many children a promoted line gets, each crossed
    //                                   with its own randomly drawn champion (0 = no band, the
    //                                   promotion takes its chances in the tournament)
    if let Some(n) = num(env, "EVOLVE_ARRIVAL_CHILDREN") {
        config.arrival_children = n;
    }
    //   EVOLVE_CHAMPION_ALPS=1          VIRTUAL ALPS on the CHAMPION island too, so a promoted
    //                                   line meets its own generation there rather than the
    //                                   incumbent. Off = the open knockout, which is what
    //                                   recovered strogatz_bacres1 twice and is still default.
    if let Some(v) = flag("EVOLVE_CHAMPION_ALPS") {
        config.champion_open_fight = !v;
    }
    //   EVOLVE_CHAMPION_ELITES          the champion island's elites (unset = the intake's).
    //                                   Elites are copied unmutated and skip crossover, so on a
    //                                   converged island they are the incumbent's safest rows.
    if let Some(n) = num(env, "EVOLVE_CHAMPION_ELITES") {
        config.champion_elites = Some(n);
    }
    //   EVOLVE_CHAMPION_COHORT_MERGE    VIRTUAL ALPS' merge age on the champion island alone
    //                                   (unset = the intake's, 0 = cohorts off there). Only bites
    //                                   when EVOLVE_CHAMPION_ALPS=1, since an open knockout
    //                                   ignores cohorts whatever their age.
    if let Some(n) = num(env, "EVOLVE_CHAMPION_COHORT_MERGE") {
        config.champion_cohort_merge = Some(n);
    }
    //   EVOLVE_PROGRESS_EVERY           a progress line on stderr every N generations (0 = none)
    if let Some(n) = num(env, "EVOLVE_PROGRESS_EVERY") {
        config.progress_every = n;
    }
    //   EVOLVE_COMPOUNDS=1              the compound functions (sqrt|a+-b|, 1/sqrt|a+-b|, 1/(a+-b)) join
    //                                   the symbol table — for the race's second pass
    if let Some(v) = flag("EVOLVE_COMPOUNDS") {
        config.compounds = v;
    }
    //   EVOLVE_GENES                    genes per chromosome (the engine's default is 3)
    if let Some(n) = num(env, "EVOLVE_GENES") {
        config.n_genes = n;
    }
    //   EVOLVE_GENE_SUBSETS=1           THE DYNAMIC GENE-SUBSET CHOICE: a chromosome is scored
    //                                   under every non-empty subset of its genes and keeps the
    //                                   best (at most 3 genes)
    if let Some(v) = flag("EVOLVE_GENE_SUBSETS") {
        config.gene_subsets = v;
    }
    //   EVOLVE_COHORT_MERGE             VIRTUAL ALPS — "couples from the same century".
    //                                   Each row carries the pump beat its line arrived
    //                                   on, inherited by every descendant, and a
    //                                   tournament prefers a candidate of the SAME
    //                                   cohort however fit the others are. 0 = off.
    if let Some(m) = num(env, "EVOLVE_COHORT_MERGE") {
        config.cohort_merge = m;
    }
    //   EVOLVE_PUMP_EVERY               the pump's beat in generations
    if let Some(beat) = num(env, "EVOLVE_PUMP_EVERY") {
        config.pump_every = beat;
    }
    //   EVOLVE_HEAD                     a gene's head length (the engine's default is 34)
    if let Some(head) = num(env, "EVOLVE_HEAD") {
        config.head = head;
    }
    //   EVOLVE_RNC_LO / EVOLVE_RNC_HI   the range a gene's random constants are drawn from
    if let (Some(lo), Some(hi)) = (num(env, "EVOLVE_RNC_LO"), num(env, "EVOLVE_RNC_HI")) {
        config.rnc_lo = lo;
        config.rnc_hi = hi;
    }
    //   EVOLVE_REDUNDANCY=1             leave-one-gene-out redundancy as an HFF objective
    if let Some(v) = flag("EVOLVE_REDUNDANCY") {
        config.redundancy = v;
    }
    //   EVOLVE_MAX_GENERATIONS          stop by GENERATIONS (give a long time cap): a
    //                                   comparison that does not depend on how busy the device is
    if let Some(g) = num(env, "EVOLVE_MAX_GENERATIONS") {
        config.max_generations = g;
    }
    //   EVOLVE_PAIRS                    pairs of islands (intake + champion); the island
    //                                   sizes are ONE pair's, so the population is this
    //                                   many times as large (default 1)
    //   EVOLVE_CROSS_EVERY              THE CROSS STEP's beat: every this many generations
    //                                   each intake island takes in the best of the other
    //                                   pairs' champion islands (default 0 = never)
    //   EVOLVE_K_MIGRANTS               how many each champion island sends (default 3)
    if let Some(n) = num(env, "EVOLVE_PAIRS") {
        config.n_pairs = n;
    }
    if let Some(n) = num(env, "EVOLVE_CROSS_EVERY") {
        config.cross_every = n;
    }
    if let Some(n) = num(env, "EVOLVE_K_MIGRANTS") {
        config.k_migrants = n;
    }
    //   EVOLVE_LANES                    THE SWIM LANES: one rule set per island pair, as
    //                                   `name:explore:recombine[:cleanse]` separated by
    //                                   commas — "general:1:1,explorer:3:0.5". There must
    //                                   be exactly EVOLVE_PAIRS of them.
    if let Some(spec) = env("EVOLVE_LANES") {
        config.lanes = Some(parse_lanes(&spec));
    }
    //   EVOLVE_SNAP_EVERY               SNAP WINNERS' beat in generations (0 = off):
    //                                   kept snapped forms are written back into the genes
    //   EVOLVE_SNAP_TOP_K               rows per island, by fitness, that are snap winners (0 = all)
    if let Some(n) = num(env, "EVOLVE_SNAP_EVERY") {
        config.snap_every = n;
    }
    if let Some(n) = num(env, "EVOLVE_SNAP_TOP_K") {
        config.snap_top_k = n;
    }
    //   EVOLVE_BEAM_EVERY               THE BEAM's beat in generations (0 = off)
    //   EVOLVE_BEAM_WIDTH               how many mutants a beat generates (default 2000)
    //   EVOLVE_BEAM_WRAPS=0             switch the FUNCTIONAL mutations off
    //   EVOLVE_BEAM_TREE=1              switch the TREE mutations ON
    //   EVOLVE_FLOAT_ZONE               THE FLOAT ZONE: extra intake rows a beam survivor is
    //                                   APPENDED into (0 = off)
    if let Some(n) = num(env, "EVOLVE_BEAM_EVERY") {
        config.beam_every = n;
    }
    if let Some(n) = num(env, "EVOLVE_BEAM_WIDTH") {
        config.beam_width = n;
    }
    if let Some(v) = flag("EVOLVE_BEAM_WRAPS") {
        config.beam_wraps = v;
    }
    if let Some(v) = flag("EVOLVE_BEAM_TREE") {
        config.beam_tree = v;
    }
    if let Some(n) = num(env, "EVOLVE_FLOAT_ZONE") {
        config.float_zone = n;
    }
    //   EVOLVE_POP_INTAKE / EVOLVE_POP_CHAMPION   the two islands' sizes, by name.
    //
    // THE POSITIONAL `population` ARGUMENT is not here: it is a 3:1 split of one
    // number and belongs with the other positional arguments in `evolve_fit`,
    // which applies it BEFORE this so a named variable still wins.
    if let Some(n) = num(env, "EVOLVE_POP_INTAKE") {
        config.pop_intake = n;
    }
    if let Some(n) = num(env, "EVOLVE_POP_CHAMPION") {
        config.pop_champion = n;
    }
}

/// `EVOLVE_LANES`, as `name:explore:recombine[:cleanse]` separated by commas.
///
/// # Panics
/// When a lane has no name, or a number it must have is missing or unparseable:
/// a mistyped lane is a silently different experiment, which is worse than a
/// stopped run.
pub fn parse_lanes(spec: &str) -> Vec<Lane> {
    let mut lanes = Vec::new();
    for one in spec.split(',').filter(|s| !s.trim().is_empty()) {
        let f: Vec<&str> = one.split(':').collect();
        let num = |i: usize, what: &str| -> f64 {
            f.get(i)
                .unwrap_or_else(|| panic!("EVOLVE_LANES: lane {one:?} has no {what}; it is name:explore:recombine[:cleanse]"))
                .parse()
                .unwrap_or_else(|e| panic!("EVOLVE_LANES: lane {one:?} {what}: {e}"))
        };
        lanes.push(Lane {
            name: (*f.first().expect("a lane needs a name")).to_string(),
            explore: num(1, "explore"),
            recombine: num(2, "recombine"),
            cleanse: f.get(3).map(|v| v.parse().expect("EVOLVE_LANES: cleanse")),
        });
    }
    lanes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A config that differs from the default in every KIND of field, so a
    /// round-trip failure in any of them shows up: numbers, floats, bools, the
    /// bool array, both Option flavours and the lane table.
    fn varied() -> Config {
        Config {
            pop_intake: 12_345,
            pop_champion: 6_789,
            tournament_fraction: 0.11,
            champion_tournament_fraction: Some(0.03),
            arrival_children: 7,
            champion_open_fight: false,
            champion_elites: Some(9),
            champion_cohort_merge: Some(40),
            head: 21,
            pump_every: 100,
            cohort_merge: 10_000,
            n_pairs: 3,
            cross_every: 5,
            log_scale: [true, false, true],
            hff_without_validation: true,
            tower: true,
            redundancy: true,
            smogd: true,
            balanced_tournaments: true,
            compounds: true,
            gene_subsets: true,
            beam_tree: true,
            float_zone: 500,
            lanes: Some(vec![
                Lane { name: "general".into(), explore: 1.0, recombine: 1.0, cleanse: None },
                Lane { name: "explorer".into(), explore: 3.0, recombine: 0.5, cleanse: Some(0.25) },
                Lane { name: "still".into(), explore: 0.5, recombine: 2.0, cleanse: None },
            ]),
            ..Config::srbench(7014)
        }
    }

    fn card_of(config: Config) -> Card {
        Card {
            card_version: CARD_VERSION,
            card_id: "a_test".into(),
            written_utc: "2026-09-23T00:00:00Z".into(),
            code: Code { git_commit: "abc123".into(), git_dirty: false },
            data: DataCard {
                path: "/tmp/bacres1.tsv".into(),
                max_rows: 5_000,
                use_all: false,
                edge_path: None,
                test_path: None,
                n_train: 400,
                n_val: 100,
                n_third: 250,
                n_test: 167,
            },
            synthetic: Synthetic { smogd: true, smote: true, smogd_noise: 1.5 },
            run: Run { seed: 7014, budget_seconds: 86_400.0, restarts: 1 },
            config,
            derived: Derived {
                hff_objectives: 6,
                population: 57_402,
                islands: vec![IslandSpan { lo: 0, hi: 12_345 }, IslandSpan { lo: 12_345, hi: 19_134 }],
                live_cohorts: 100,
            },
        }
    }

    /// THE PROPERTY THAT MATTERS: a card written and read back is the same card,
    /// and above all the same CONFIG. Every knob, not a chosen subset.
    #[test]
    fn a_card_round_trips() {
        let card = card_of(varied());
        let dir = std::env::temp_dir().join("fuller_card_round_trip");
        std::fs::create_dir_all(&dir).expect("the temp directory");
        let path = dir.join("card.json");
        let path = path.to_str().expect("a utf-8 path");
        card.write(path).expect("write the card");
        let back = Card::load(path).expect("read the card");
        assert_eq!(back, card, "a card read back is the card that was written");
        assert_eq!(back.config, varied(), "every config field survives the round trip");
        std::fs::remove_file(path).ok();
    }

    /// AN EMPTY ENVIRONMENT LEAVES THE CARD ALONE. This is the assertion that
    /// catches an `unwrap_or(literal)` creeping back into the override block: one
    /// of those would put the field back to the engine's default on every replay.
    #[test]
    fn an_empty_environment_changes_nothing() {
        let mut config = varied();
        apply_env(&mut config, &|_| None);
        assert_eq!(config, varied(), "no environment variable set changes no field");
    }

    /// THE ENVIRONMENT WINS OVER THE CARD, so a card is a starting point to vary
    /// from rather than a cage.
    #[test]
    fn the_environment_wins_over_the_card() {
        let mut config = varied();
        let set = |k: &str| -> Option<String> {
            Some(match k {
                "EVOLVE_PUMP_EVERY" => "20",
                "EVOLVE_COHORT_MERGE" => "100",
                "EVOLVE_POP_INTAKE" => "2000",
                "EVOLVE_POP_CHAMPION" => "2000",
                "EVOLVE_TOWER" => "0",
                "EVOLVE_CHAMPION_ALPS" => "1",
                "EVOLVE_LANES" => "one:2:0.5",
                _ => return None,
            }
            .to_string())
        };
        apply_env(&mut config, &set);
        assert_eq!(config.pump_every, 20);
        assert_eq!(config.cohort_merge, 100);
        assert_eq!((config.pop_intake, config.pop_champion), (2_000, 2_000));
        assert!(!config.tower, "EVOLVE_TOWER=0 is explicitly off, not unset");
        assert!(!config.champion_open_fight, "EVOLVE_CHAMPION_ALPS=1 turns the open knockout off");
        assert_eq!(config.lanes, Some(vec![Lane { name: "one".into(), explore: 2.0, recombine: 0.5, cleanse: None }]));
        // Everything the environment did NOT name is still the card's.
        assert_eq!(config.head, 21);
        assert_eq!(config.float_zone, 500);
        assert_eq!(config.log_scale, [true, false, true]);
    }

    /// THE STOP BAR'S INFINITY survives JSON. serde_json writes a non-finite
    /// float as `null` and refuses to read `null` into an `f64`, so without the
    /// `finite_or_word` codec a card from a run with `EVOLVE_STOP_LOG10_P=inf`
    /// would be written and then never load.
    #[test]
    fn an_infinite_stop_bar_survives_json() {
        for bar in [f64::INFINITY, f64::NEG_INFINITY, -19.0] {
            let config = Config { stop_log10_p: bar, ..Config::srbench(1) };
            let text = serde_json::to_string(&config).expect("encode");
            let back: Config = serde_json::from_str(&text).expect("decode");
            assert_eq!(back.stop_log10_p, bar, "the bar {bar} round trips");
        }
    }

    /// THE OUTPUT PATHS ARE NOT ON A CARD: replaying them would truncate the
    /// original run's telemetry stream and resume its checkpoint directory.
    #[test]
    fn the_output_paths_are_not_carded() {
        let config = Config {
            telemetry_path: Some("logs/bacres1_test/stream.jsonl".into()),
            checkpoint_dir: Some("logs/bacres1_test/checkpoint".into()),
            hof_path: Some("logs/bacres1_test/hof.tsv".into()),
            genealogy_path: Some("logs/bacres1_test/genealogy.tsv".into()),
            telemetry_run_id: Some("bacres1_test".into()),
            telemetry_dataset: Some("bacres1.tsv".into()),
            ..Config::srbench(1)
        };
        let text = serde_json::to_string(&config).expect("encode");
        // The VALUES, not the field names: `checkpoint_every_seconds` is a
        // carded number and its name legitimately contains "checkpoint", so the
        // strings looked for here are the directory parts no card may carry.
        for path in ["stream.jsonl", "logs/bacres1_test", "hof.tsv", "genealogy.tsv", "\"bacres1_test\"", "bacres1.tsv"] {
            assert!(!text.contains(path), "{path} must not reach a card: a replay would write over the original run");
        }
        let back: Config = serde_json::from_str(&text).expect("decode");
        assert_eq!(back.telemetry_path, None, "a loaded card names no output path");
        assert_eq!(back.checkpoint_dir, None);
        assert_eq!(back.hof_path, None);
        assert_eq!(back.genealogy_path, None);
        assert_eq!(back.telemetry_run_id, None);
        assert_eq!(back.telemetry_dataset, None);
    }

    /// A CARD REFUSES A VERSION IT DOES NOT KNOW rather than reading the fields
    /// it happens to recognise.
    #[test]
    fn a_card_from_another_version_is_refused() {
        let mut card = card_of(Config::srbench(1));
        card.card_version = CARD_VERSION + 7;
        let dir = std::env::temp_dir().join("fuller_card_version");
        std::fs::create_dir_all(&dir).expect("the temp directory");
        let path = dir.join("card.json");
        let path = path.to_str().expect("a utf-8 path");
        card.write(path).expect("write");
        let err = Card::load(path).expect_err("a future version is refused");
        assert!(err.contains("version"), "the error says what is wrong: {err}");
        std::fs::remove_file(path).ok();
    }

    /// THE LIVE COHORT COUNT, the number VIRTUAL ALPS' measurement is about.
    #[test]
    fn live_cohorts_is_the_ratio() {
        assert_eq!(live_cohorts(100, 20), 5, "the ratio that recovered a law");
        assert_eq!(live_cohorts(10_000, 100), 100);
        assert_eq!(live_cohorts(0, 20), 0, "cohorts off");
        assert_eq!(live_cohorts(100, 0), 0, "no beat is no bands, and no division by zero");
    }

    /// A card is PRETTY and its fields come out in declaration order, so two
    /// cards from one binary line up under a plain `diff`.
    #[test]
    fn a_card_is_readable_and_diffable() {
        let a = serde_json::to_string_pretty(&card_of(varied())).expect("encode");
        let b = serde_json::to_string_pretty(&card_of(Config { pump_every: 20, ..varied() })).expect("encode");
        assert!(a.contains("\n  \"card_id\""), "pretty, with a field to a line");
        let (la, lb): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
        assert_eq!(la.len(), lb.len(), "two cards of the same shape have the same lines");
        let differing: Vec<usize> = (0..la.len()).filter(|&i| la[i] != lb[i]).collect();
        assert_eq!(differing.len(), 1, "one changed knob is one changed line, not a reordering");
        assert!(la[differing[0]].contains("pump_every"), "and it is the line that changed");
    }
}
