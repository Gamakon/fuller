//! THE CHECKPOINT: stop a fit and start it again as though it never stopped.
//!
//! The benchmark is 133 laws at ten seeds and eight hours a fit — 1,330 runs, a
//! month of continuous GPU, on a laptop that has other work to do. So the fit
//! must survive being killed and resumed, repeatedly, and the result must not
//! depend on when it was interrupted. If it does, the run is not reproducible
//! and nothing built on it can be published.
//!
//! The bar is therefore EXACT, not approximate: a fit run for 2T seconds must
//! equal a fit run for T, killed, resumed, and run for T more — same model, same
//! generation, same hall of fame, bit for bit. Anything left out of the
//! checkpoint shows up as a divergence, which is why the test is worth more than
//! the code.
//!
//! **The generator needs no state of its own.** Every draw is a pure function of
//! `(seed, generation, row, slot, stream)`, so the generation counter IS the
//! stream position. That is what makes an exact resume achievable at all: there
//! is no hidden RNG to capture, and a resumed generation 4,001 draws exactly what
//! an uninterrupted one would.
//!
//! **Five slots, rotating.** A checkpoint of a 200,000-row population is tens of
//! megabytes; keeping five and overwriting the oldest bounds the disk while
//! leaving room to fall back if the newest is torn by a kill mid-write. A slot is
//! written to a temporary file and renamed, so a reader never sees half a file.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::vary::Generation;
use super::{Layout, Population};

/// How many rotating slots a checkpoint directory keeps.
pub const SLOTS: usize = 5;
/// Bumped whenever the saved shape changes, so an old file is refused rather
/// than read as something it is not.
pub const FORMAT_VERSION: u32 = 1;

/// Everything the fit loop mutates, and nothing it does not.
///
/// The config is NOT here: a resume is given its config by the caller, and the
/// two are checked to agree on the things that would make the resume a different
/// search (`seed`, the layout, the island sizes). That way a typo in the
/// population size is refused instead of silently starting a new search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub format_version: u32,
    /// What the resume must be asked to continue, or it is not this search.
    pub seed: u32,
    pub layout: Layout,
    pub pop_intake: u32,
    pub pop_champion: u32,
    pub n_pairs: u32,

    /// Where the search had got to. The generator is keyed on this.
    pub generation: u32,
    /// Seconds already spent, so `T + T` is `2T` and not `2T + overhead`.
    pub elapsed_seconds: f64,

    /// The population itself.
    pub genome: Vec<u32>,
    pub rnc: Vec<f32>,
    pub wrapper_id: Vec<u32>,
    pub fitness: Vec<f32>,

    /// VIRTUAL ALPS: the cohort label of every row. Empty when cohorts are off.
    pub cohorts: Vec<u32>,

    /// THE ARRIVAL BAND of each island, in island order: how many rows above
    /// the elites the last pump beat filled with promotions.
    ///
    /// This is state the pump writes and the NEXT generation's variation reads,
    /// so a fit resumed between the two would breed that band as ordinary rows
    /// and diverge from a run that was never stopped. Empty in a checkpoint
    /// written before the band existed, which reads as no arrivals — the
    /// behaviour those checkpoints were taken under.
    #[serde(default)]
    pub arrivals: Vec<u32>,

    /// HFF's frozen per-column maxima. Recomputing these on resume would rescale
    /// every objective and change what the search prefers.
    pub col_max: Option<[f64; 9]>,

    /// The best the fit has seen, which may no longer be in the population.
    pub hof: Option<SavedHof>,

    /// Counters the fit reports at the end.
    pub unique_genes: u64,
    pub oversized_genes: u64,
    pub individuals: u64,
}

/// The hall of fame as it goes to disk: the winner's genes and what it scored,
/// without the parts that can be recomputed from them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedHof {
    pub generation: u32,
    pub math: String,
    pub genome: Vec<u32>,
    pub rnc: Vec<f32>,
    pub wrapper_id: u32,
    pub fitness: f64,
    pub one_minus_r2: [f64; 3],
    pub t_depth: u32,
    pub linker: usize,
    pub wrapper: usize,
    pub a: f64,
    pub b: f64,
    pub selection: f64,
    pub genes: u32,
}

impl Checkpoint {
    /// The population, as the engine holds it.
    pub fn population(&self) -> Population {
        Population {
            layout: self.layout,
            genome: self.genome.clone(),
            rnc: self.rnc.clone(),
            wrapper_id: self.wrapper_id.clone(),
        }
    }

    /// The generation, as the fit loop holds it.
    pub fn generation_state(&self) -> Generation {
        Generation { pop: self.population(), fitness: self.fitness.clone() }
    }

    /// Would resuming this checkpoint under `other` be the SAME search? A resume
    /// that quietly changes the seed or the population size is a new run wearing
    /// an old name, and the whole point of the checkpoint is that it is not.
    pub fn agrees_with(&self, seed: u32, layout: Layout, pop_intake: u32, pop_champion: u32, n_pairs: u32) -> Result<(), String> {
        if self.format_version != FORMAT_VERSION {
            return Err(format!("checkpoint is format {} and this engine writes {FORMAT_VERSION}", self.format_version));
        }
        let mismatch = |what: &str, was: String, now: String| Err(format!("checkpoint {what} is {was}, the resume asks for {now}"));
        if self.seed != seed {
            return mismatch("seed", self.seed.to_string(), seed.to_string());
        }
        if self.layout != layout {
            return mismatch("layout", format!("{:?}", self.layout), format!("{layout:?}"));
        }
        if (self.pop_intake, self.pop_champion, self.n_pairs) != (pop_intake, pop_champion, n_pairs) {
            return mismatch(
                "islands",
                format!("{}+{} x {}", self.pop_intake, self.pop_champion, self.n_pairs),
                format!("{pop_intake}+{pop_champion} x {n_pairs}"),
            );
        }
        Ok(())
    }
}

/// A directory of rotating checkpoint slots.
pub struct Slots {
    dir: PathBuf,
}

impl Slots {
    /// The directory is created if it is not there.
    pub fn open(dir: impl AsRef<Path>) -> Result<Slots, String> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| format!("checkpoint directory {}: {e}", dir.display()))?;
        Ok(Slots { dir })
    }

    fn slot_path(&self, n: usize) -> PathBuf {
        self.dir.join(format!("slot_{n}.json"))
    }

    /// Write `cp` to the oldest slot, so the newest four survive a kill during
    /// this write. The file is written beside its slot and renamed, which is
    /// atomic on every filesystem we run on: a reader sees the old slot or the
    /// new one, never a half-written file.
    pub fn save(&self, cp: &Checkpoint) -> Result<PathBuf, String> {
        let oldest = (0..SLOTS)
            .min_by_key(|&n| {
                std::fs::metadata(self.slot_path(n))
                    .and_then(|m| m.modified())
                    .ok()
                    // A slot that does not exist is the oldest there is.
                    .map_or(std::time::SystemTime::UNIX_EPOCH, |t| t)
            })
            .unwrap_or(0);
        let path = self.slot_path(oldest);
        let tmp = path.with_extension("writing");
        let bytes = serde_json::to_vec(cp).map_err(|e| format!("checkpoint encode: {e}"))?;
        {
            let mut f = std::fs::File::create(&tmp).map_err(|e| format!("checkpoint {}: {e}", tmp.display()))?;
            f.write_all(&bytes).map_err(|e| format!("checkpoint {}: {e}", tmp.display()))?;
            f.sync_all().map_err(|e| format!("checkpoint {}: {e}", tmp.display()))?;
        }
        std::fs::rename(&tmp, &path).map_err(|e| format!("checkpoint rename {}: {e}", path.display()))?;
        Ok(path)
    }

    /// The newest slot that reads back whole, or None when the directory holds
    /// nothing usable. A torn or truncated slot is SKIPPED rather than fatal —
    /// that is what the other four are for — and the reason is returned so a
    /// caller can say so rather than silently starting from generation 0.
    pub fn newest(&self) -> (Option<Checkpoint>, Vec<String>) {
        let mut found: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
        for n in 0..SLOTS {
            let p = self.slot_path(n);
            if let Ok(t) = std::fs::metadata(&p).and_then(|m| m.modified()) {
                found.push((t, p));
            }
        }
        found.sort_by(|a, b| b.0.cmp(&a.0));
        let mut skipped = Vec::new();
        for (_, p) in found {
            match std::fs::read(&p).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice::<Checkpoint>(&b).map_err(|e| e.to_string())) {
                Ok(cp) => return (Some(cp), skipped),
                Err(e) => skipped.push(format!("{}: {e}", p.display())),
            }
        }
        (None, skipped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy(generation: u32) -> Checkpoint {
        Checkpoint {
            format_version: FORMAT_VERSION,
            seed: 7014,
            layout: Layout::for_arity(8, 1, 6, 2, 2),
            pop_intake: 6,
            pop_champion: 2,
            n_pairs: 1,
            generation,
            elapsed_seconds: f64::from(generation) * 0.5,
            genome: vec![1, 2, 3],
            rnc: vec![0.5],
            wrapper_id: vec![0; 8],
            fitness: vec![0.25; 8],
            cohorts: vec![0; 8],
            arrivals: vec![0, 0],
            col_max: Some([1.0; 9]),
            hof: None,
            unique_genes: 10,
            oversized_genes: 0,
            individuals: 80,
        }
    }

    #[test]
    fn a_saved_checkpoint_reads_back_as_what_was_written() {
        let dir = std::env::temp_dir().join(format!("fuller-cp-{}", std::process::id()));
        let slots = Slots::open(&dir).expect("open");
        slots.save(&toy(42)).expect("save");
        let (back, skipped) = slots.newest();
        let back = back.expect("a checkpoint");
        assert!(skipped.is_empty(), "{skipped:?}");
        assert_eq!(back.generation, 42);
        assert_eq!(back.elapsed_seconds, 21.0);
        assert_eq!(back.fitness.len(), 8);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// FIVE SLOTS, and the newest wins. Six saves must leave five files, and the
    /// one that reads back must be the last one written.
    #[test]
    fn the_oldest_slot_is_the_one_overwritten() {
        let dir = std::env::temp_dir().join(format!("fuller-cp-rot-{}", std::process::id()));
        let slots = Slots::open(&dir).expect("open");
        for g in 1..=(SLOTS as u32 + 1) {
            slots.save(&toy(g)).expect("save");
            // The rotation picks by modification time, which has a coarse
            // resolution on some filesystems; a moment between saves keeps the
            // order unambiguous.
            std::thread::sleep(std::time::Duration::from_millis(12));
        }
        let files = std::fs::read_dir(&dir).expect("dir").filter_map(Result::ok).filter(|e| e.path().extension().is_some_and(|x| x == "json")).count();
        assert_eq!(files, SLOTS, "the rotation kept {files} slots");
        let (back, _) = slots.newest();
        assert_eq!(back.expect("a checkpoint").generation, SLOTS as u32 + 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A TORN SLOT IS SKIPPED, not fatal. That is what the other four are for,
    /// and a run killed mid-write must still resume from the beat before.
    #[test]
    fn a_torn_slot_falls_back_to_the_one_before_it() {
        let dir = std::env::temp_dir().join(format!("fuller-cp-torn-{}", std::process::id()));
        let slots = Slots::open(&dir).expect("open");
        slots.save(&toy(7)).expect("save");
        std::thread::sleep(std::time::Duration::from_millis(12));
        let newer = slots.save(&toy(8)).expect("save");
        // Half a file, as a kill during a write would leave if the rename were
        // not atomic.
        std::fs::write(&newer, b"{\"format_version\":1,\"seed\":70").expect("tear it");
        let (back, skipped) = slots.newest();
        assert_eq!(back.expect("a checkpoint").generation, 7, "the torn slot was not skipped");
        assert_eq!(skipped.len(), 1, "the skip was not reported: {skipped:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A RESUME THAT CHANGES THE SEARCH IS REFUSED. Quietly continuing under a
    /// different seed or population would produce a run that is reproducible
    /// only by accident.
    #[test]
    fn a_resume_under_different_settings_is_refused() {
        let cp = toy(3);
        let l = cp.layout;
        assert!(cp.agrees_with(7014, l, 6, 2, 1).is_ok());
        let wrong_seed = cp.agrees_with(7015, l, 6, 2, 1).expect_err("a different seed must be refused");
        assert!(wrong_seed.contains("seed"), "{wrong_seed}");
        let wrong_pop = cp.agrees_with(7014, l, 60, 2, 1).expect_err("a different population must be refused");
        assert!(wrong_pop.contains("islands"), "{wrong_pop}");
    }
}
