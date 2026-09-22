//! IDENTITY, AGE and LINEAGE for every individual of a fit — ALPS's measurement
//! half (Hornby's Age-Layered Population Structure), and nothing of its algorithm:
//! this module WATCHES the population, it never changes who breeds.
//!
//! Every row of every generation carries
//!   * an `id`, unique within a fit and stable for the SAME genotype;
//!   * an `age` in generations since that genotype entered the population — an
//!     offspring is its parent's age plus one, a fresh random individual is 0;
//!   * a FOUNDER: the id, generation and [`Origin`] of the individual at the head
//!     of its line, carried forward row by row so "how old is the winner and where
//!     did its line come from" is answerable without holding every edge.
//!
//! The edges themselves — who came from whom, one record per minted id — are kept
//! in a `Vec<Edge>` indexed BY the id, so the ancestry walk is a chain of array
//! reads. See [`Genealogy::walk`].
//!
//! The lineage edge is not re-derived here. `vary::select` (and `vary.wgsl`'s
//! `select_main`, which `device.rs` proves bit-identical to it) already computes,
//! for every row of the next generation, the row of this one it is cloned from;
//! [`Genealogy::advance`] is handed that `parent` array and does the bookkeeping.

use std::fmt;

use super::vary::Island;

/// WHICH MECHANISM put a row into the population. Every path in the engine that
/// writes a row names one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Origin {
    /// The initial draw: generation 0, `Engine::fit`'s `dev.init`.
    Init,
    /// Ordinary variation: cloned from a parent by the tournaments and mutated.
    Vary,
    /// An elite, copied unchanged into its island's first rows by the same kernel.
    Elite,
    /// THE PUMP: an intake island's best, copied over a champion island's worst.
    PumpPromote,
    /// THE PUMP: a new random individual filling the refilled intake.
    PumpRefill,
    /// THE PUMP: one of the intake's best fifth, kept in place through the refill.
    PumpKeep,
    /// THE CROSS STEP: a migrant from another pair's champion island.
    CrossArrival,
    /// THE CROSS STEP: a new random individual filling what the migrants left.
    CrossFresh,
    /// THE CROSS STEP: one of the intake's best fifth, kept through the step.
    CrossKeep,
    /// SNAP's write-back rewrote a gene of this row in place.
    SnapWriteback,
    /// A MUTATION BEAM survivor appended to the intake island. Reserved: the beam
    /// is another agent's work and nothing here writes this yet.
    BeamAppend,
}

impl Origin {
    /// The tab-separated log's word for this mechanism.
    pub fn as_str(self) -> &'static str {
        match self {
            Origin::Init => "init",
            Origin::Vary => "vary",
            Origin::Elite => "elite",
            Origin::PumpPromote => "pump_promote",
            Origin::PumpRefill => "pump_refill",
            Origin::PumpKeep => "pump_keep",
            Origin::CrossArrival => "cross_arrival",
            Origin::CrossFresh => "cross_fresh",
            Origin::CrossKeep => "cross_keep",
            Origin::SnapWriteback => "snap_writeback",
            Origin::BeamAppend => "beam_append",
        }
    }

    /// Whether this mechanism STARTS a line: a founder's origin is one of these.
    /// A line begins where a genotype entered the population from outside it —
    /// the initial draw, or one of the refills that draw new random individuals.
    pub fn is_arrival(self) -> bool {
        matches!(self, Origin::Init | Origin::PumpRefill | Origin::CrossFresh | Origin::BeamAppend)
    }
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One minted individual's record, indexed by its own id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    /// The individual this one was cloned from, or `u64::MAX` for a founder.
    pub parent: u64,
    /// THE OTHER ANCESTOR: the id of the row this one was mated with by crossover,
    /// or `u64::MAX`. Crossover works over consecutive offspring PAIRS, so an
    /// offspring that took part has two ancestors — the row it was cloned from
    /// (`parent`) and its mate.
    pub other: u64,
    /// The generation in which this id was minted.
    pub generation: u32,
    /// Generations since this line's genotype entered the population.
    pub age: u32,
    /// The id at the head of this line.
    pub founder: u64,
    /// The generation that founder was minted in.
    pub founder_generation: u32,
    /// The mechanism that put that founder into the population.
    pub founder_origin: Origin,
    /// The mechanism that put THIS individual into its row.
    pub origin: Origin,
}

impl Edge {
    /// A founder: the head of its own line.
    fn founder_edge(id: u64, generation: u32, origin: Origin) -> Edge {
        Edge {
            parent: u64::MAX,
            other: u64::MAX,
            generation,
            age: 0,
            founder: id,
            founder_generation: generation,
            founder_origin: origin,
            origin,
        }
    }
}

/// What a row carries: its identity and its line, kept beside `Population`'s
/// `genome` / `rnc` / `wrapper_id` and moved by exactly the same steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowMark {
    pub id: u64,
    pub age: u32,
    pub founder: u64,
    pub founder_generation: u32,
    pub founder_origin: Origin,
}

/// One row of the genealogy log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    pub generation: u32,
    pub row: u32,
    pub id: u64,
    pub parent: u64,
    pub other: u64,
    pub age: u32,
    pub founder: u64,
    pub founder_generation: u32,
    pub founder_origin: Origin,
    pub origin: Origin,
}

/// The identities of a population, and the edges of everything minted so far.
///
/// `rows` is parallel to the population: `rows[r]` is what sits in row `r` right
/// now. `edges[id]` is that individual's record — ids are minted from a counter
/// that starts at 0 for each fit, so the id IS the index.
pub struct Genealogy {
    rows: Vec<RowMark>,
    edges: Vec<Edge>,
    next: u64,
}

impl Genealogy {
    /// The initial draw: every row a founder of its own line, age 0.
    pub fn init(pop: u32) -> Genealogy {
        let mut g = Genealogy { rows: Vec::with_capacity(pop as usize), edges: Vec::with_capacity(pop as usize), next: 0 };
        for _ in 0..pop {
            let mark = g.mint_founder(0, Origin::Init);
            g.rows.push(mark);
        }
        g
    }

    /// A new id, and the edge that goes with it.
    fn mint(&mut self, edge: Edge) -> u64 {
        let id = self.next;
        self.next += 1;
        debug_assert_eq!(self.edges.len() as u64, id, "edges are indexed by id");
        self.edges.push(edge);
        id
    }

    /// A fresh individual: its own founder, age 0.
    fn mint_founder(&mut self, generation: u32, origin: Origin) -> RowMark {
        let id = self.next;
        let id = self.mint(Edge::founder_edge(id, generation, origin));
        RowMark { id, age: 0, founder: id, founder_generation: generation, founder_origin: origin }
    }

    /// A row's mark as it stands.
    pub fn row(&self, r: usize) -> RowMark {
        self.rows[r]
    }

    /// How many rows are tracked.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// How many ids have been minted in this fit: the edge table's length, and so
    /// the memory it costs.
    pub fn minted(&self) -> u64 {
        self.next
    }

    /// The edge of an id, if it was minted in this fit.
    pub fn edge(&self, id: u64) -> Option<Edge> {
        self.edges.get(id as usize).copied()
    }

    /// Every row's mark, in row order.
    pub fn marks(&self) -> &[RowMark] {
        &self.rows
    }

    /// The record of the individual now in row `r`, for the log: its `origin` is
    /// the mechanism that MINTED it — how the individual came to exist.
    pub fn record(&self, generation: u32, r: usize) -> Record {
        let origin = self.edges[self.rows[r].id as usize].origin;
        self.record_as(generation, r, origin)
    }

    /// The same, but naming the EVENT being logged rather than the minting. A
    /// keeper and a snap write-back are not new individuals — no id is minted for
    /// them, so their edge still says how they were born — and the log would
    /// otherwise be unable to say that the pump kept this row or that snap
    /// rewrote it. The identity, age and line reported are the row's own.
    pub fn record_as(&self, generation: u32, r: usize, origin: Origin) -> Record {
        let mark = self.rows[r];
        let edge = self.edges[mark.id as usize];
        Record {
            generation,
            row: r as u32,
            id: mark.id,
            parent: edge.parent,
            other: edge.other,
            age: mark.age,
            founder: mark.founder,
            founder_generation: mark.founder_generation,
            founder_origin: mark.founder_origin,
            origin,
        }
    }

    /// THE ANCESTRY WALK: an id, then its parent, then its parent's, back to the
    /// founder — whose `parent` is `u64::MAX` and whose origin is an arrival.
    /// The chain is newest first, each step its own `(id, edge)`.
    pub fn walk(&self, id: u64) -> Vec<(u64, Edge)> {
        let mut chain = Vec::new();
        let mut at = id;
        while let Some(edge) = self.edges.get(at as usize).copied() {
            chain.push((at, edge));
            if edge.parent == u64::MAX {
                break;
            }
            at = edge.parent;
        }
        chain
    }

    /// ONE GENERATION of variation, from the `parent` array the selection kernel
    /// wrote: for every row of the next generation, the row of this one it is
    /// cloned from.
    ///
    /// An island's first `elites` rows are its elites, copied unchanged by the
    /// mutate kernel — they KEEP their id and their founder and age by one, the
    /// ALPS rule for a genotype that survived another generation. Every other row
    /// is an offspring: a NEW id, its parent's age plus one, its parent's founder.
    ///
    /// `crossover` mates consecutive offspring PAIRS, so an offspring's MATE is
    /// the other row of its pair; it is recorded unconditionally as the
    /// `other` ancestor, whether or not a crossover coin came up — the log then
    /// names who the row was mated WITH, and the three coins that decide whether
    /// any tokens moved are the engine's own draws.
    pub fn advance(&mut self, parent: &[u32], islands: &[Island], generation: u32) {
        let mut next: Vec<RowMark> = Vec::with_capacity(self.rows.len());
        // Row r of the next generation reads row parent[r] of THIS one, so the
        // marks of this generation must all be read before any is overwritten.
        let now = self.rows.clone();
        for isl in islands {
            for r in isl.lo..isl.hi {
                let src = now[parent[r as usize] as usize];
                if r < isl.lo + isl.elites {
                    // THE ELITE: the same genotype, copied unchanged, one generation
                    // older. It keeps the id it entered the population with — no new
                    // individual exists, so no id is minted and its edge (which still
                    // names the mechanism that put the genotype there) is unchanged.
                    // Its age lives on the ROW, which is where the ageing happens.
                    next.push(RowMark { id: src.id, age: src.age + 1, ..src });
                    continue;
                }
                let mate = mate_of(*isl, r).map(|m| now[m as usize].id).unwrap_or(u64::MAX);
                let id = self.mint(Edge {
                    parent: src.id,
                    other: mate,
                    generation,
                    age: src.age + 1,
                    founder: src.founder,
                    founder_generation: src.founder_generation,
                    founder_origin: src.founder_origin,
                    origin: Origin::Vary,
                });
                next.push(RowMark { id, age: src.age + 1, founder: src.founder, founder_generation: src.founder_generation, founder_origin: src.founder_origin });
            }
        }
        self.rows = next;
    }

    /// THE PUMP's promotion: the genotype in row `from` is COPIED over row `to`,
    /// and stays where it was — so the copy is a new individual of the same line,
    /// at its parent's age (no generation has passed).
    pub fn promote(&mut self, from: usize, to: usize, generation: u32) {
        let src = self.rows[from];
        let id = self.mint(Edge {
            parent: src.id,
            other: u64::MAX,
            generation,
            age: src.age,
            founder: src.founder,
            founder_generation: src.founder_generation,
            founder_origin: src.founder_origin,
            origin: Origin::PumpPromote,
        });
        self.rows[to] = RowMark { id, age: src.age, ..src };
    }

    /// A REFILL's moves: a row of `before` lands in row `to`. A keeper or an
    /// arrival is the same genotype in another place and no generation has passed,
    /// so its age does not change; an arrival is a CLONE (its source stays in its
    /// champion island) and gets a new id, while a keeper that moves within its own
    /// island is the same individual and keeps its id.
    pub fn moved(&mut self, marks_before: &[RowMark], from: usize, to: usize, origin: Origin, generation: u32) {
        let src = marks_before[from];
        let mark = match origin {
            Origin::PumpKeep | Origin::CrossKeep => src,
            _ => {
                let id = self.mint(Edge {
                    parent: src.id,
                    other: u64::MAX,
                    generation,
                    age: src.age,
                    founder: src.founder,
                    founder_generation: src.founder_generation,
                    founder_origin: src.founder_origin,
                    origin,
                });
                RowMark { id, ..src }
            }
        };
        self.rows[to] = mark;
    }

    /// A refill's FRESH row: a new random individual, its own founder, age 0.
    pub fn fresh(&mut self, to: usize, origin: Origin, generation: u32) {
        self.rows[to] = self.mint_founder(generation, origin);
    }

    /// SNAP's write-back: a gene of row `r` was rewritten in place. The guard only
    /// keeps a form that computes the same model within its R² tolerance, so this
    /// is a REPAIR of the individual, not reproduction — the row keeps its id, its
    /// age and its line, and the log records the event.
    pub fn snap(&mut self, r: usize, generation: u32) -> Record {
        let mark = self.rows[r];
        Record {
            generation,
            row: r as u32,
            id: mark.id,
            parent: mark.id,
            other: u64::MAX,
            age: mark.age,
            founder: mark.founder,
            founder_generation: mark.founder_generation,
            founder_origin: mark.founder_origin,
            origin: Origin::SnapWriteback,
        }
    }

    /// The marks as they stand, to be read before a step rewrites rows.
    pub fn snapshot(&self) -> Vec<RowMark> {
        self.rows.clone()
    }

    /// THE AGE DISTRIBUTION of a set of rows: (min, median, max, mean).
    pub fn ages(&self, rows: impl IntoIterator<Item = usize>) -> Option<(u32, u32, u32, f64)> {
        let mut v: Vec<u32> = rows.into_iter().map(|r| self.rows[r].age).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_unstable();
        let mean = v.iter().map(|&a| f64::from(a)).sum::<f64>() / v.len() as f64;
        Some((v[0], v[v.len() / 2], v[v.len() - 1], mean))
    }

    /// HOW MANY DISTINCT FOUNDERS a set of rows descends from — the diversity
    /// number ALPS bears on.
    pub fn distinct_founders(&self, rows: impl IntoIterator<Item = usize>) -> usize {
        let mut v: Vec<u64> = rows.into_iter().map(|r| self.rows[r].founder).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    }
}

/// THE GENEALOGY LOG: a header, then one tab-separated row per record, appended —
/// the idiom of `Config::hof_path`'s hall-of-fame file.
///
/// Writing every row of every generation is not affordable (a population of 3,000
/// over 1,000 generations is 3 million rows per fit), so the log is SELECTIVE:
///
///   * the best individual by HFF, every generation (`kind` = `best`);
///   * every ARRIVAL — any row whose origin is not `vary`: the promotions, the
///     keepers, the migrants and the snap write-backs (`kind` = `arrival`). The
///     initial draw and the fresh refills are thousands of identical-shaped rows,
///     so a batch of consecutive fresh ids is ONE line (`kind` = `batch`) whose
///     `id` and `parent` columns are the first and last id of the batch — the
///     individuals are still each identifiable, at a hundredth of the bytes;
///   * when a fit ends, the winner's whole chain back to its founder
///     (`kind` = `lineage_of_winner`), newest first.
pub struct GenealogyLog {
    file: std::fs::File,
    pub lines: u64,
    pub bytes: u64,
}

/// The log's columns.
pub const GENEALOGY_HEADER: &str =
    "kind\tgeneration\trow\tid\tparent\tother\tage\tfounder\tfounder_gen\tfounder_origin\torigin\n";

impl GenealogyLog {
    /// Create (or truncate) the file and write its header.
    pub fn create(path: &str) -> Result<GenealogyLog, String> {
        use std::io::Write;
        let mut file = std::fs::File::create(path).map_err(|e| format!("genealogy file {path}: {e}"))?;
        file.write_all(GENEALOGY_HEADER.as_bytes()).map_err(|e| format!("genealogy file {path}: {e}"))?;
        Ok(GenealogyLog { file, lines: 0, bytes: GENEALOGY_HEADER.len() as u64 })
    }

    fn write(&mut self, line: &str) -> Result<(), String> {
        use std::io::Write;
        self.file.write_all(line.as_bytes()).map_err(|e| format!("genealogy file: {e}"))?;
        self.lines += 1;
        self.bytes += line.len() as u64;
        Ok(())
    }

    /// One record.
    pub fn record(&mut self, kind: &str, r: &Record) -> Result<(), String> {
        let line = format!(
            "{kind}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.generation,
            r.row,
            r.id,
            signed(r.parent),
            signed(r.other),
            r.age,
            r.founder,
            r.founder_generation,
            r.founder_origin,
            r.origin
        );
        self.write(&line)
    }

    /// A RUN of fresh individuals drawn in one step: rows `row_lo ..= row_hi` took
    /// the consecutive ids `id_lo ..= id_hi`, all age 0, all founders of their own
    /// lines. One line for what would otherwise be thousands.
    pub fn batch(&mut self, generation: u32, row_lo: u32, row_hi: u32, id_lo: u64, id_hi: u64, origin: Origin) -> Result<(), String> {
        let line = format!(
            "batch\t{generation}\t{row_lo}-{row_hi}\t{id_lo}\t{id_hi}\t-1\t0\t{id_lo}\t{generation}\t{origin}\t{origin}\n"
        );
        self.write(&line)
    }

    /// THE WINNER'S CHAIN, newest first, when a fit ends: each step's own id, so
    /// the `parent` column links one line to the next. The `row` column is the
    /// step's distance from the winner (0 = the winner itself).
    pub fn lineage_of_winner(&mut self, chain: &[(u64, Edge)]) -> Result<(), String> {
        for (step, (id, edge)) in chain.iter().enumerate() {
            let line = format!(
                "lineage_of_winner\t{}\t{step}\t{id}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                edge.generation,
                signed(edge.parent),
                signed(edge.other),
                edge.age,
                edge.founder,
                edge.founder_generation,
                edge.founder_origin,
                edge.origin
            );
            self.write(&line)?;
        }
        Ok(())
    }
}

/// `u64::MAX` (no such ancestor) as `-1`, so the column stays numeric.
fn signed(id: u64) -> String {
    if id == u64::MAX {
        "-1".to_string()
    } else {
        id.to_string()
    }
}

/// The row an offspring is MATED with by `vary::crossover`: the offspring of an
/// island are paired `(lo + elites, lo + elites + 1)`, `(.. + 2, .. + 3)` and so on
/// (`validate` guarantees an even count). An elite has no mate.
pub fn mate_of(island: Island, row: u32) -> Option<u32> {
    let first = island.lo + island.elites;
    if row < first {
        return None;
    }
    let k = row - first;
    Some(if k % 2 == 0 { row + 1 } else { row - 1 }).filter(|&m| m < island.hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn islands() -> Vec<Island> {
        vec![Island { lo: 0, hi: 20, elites: 2, tournsize: 3 }, Island { lo: 20, hi: 30, elites: 2, tournsize: 3 }]
    }

    #[test]
    fn a_new_population_is_thirty_founders_each_its_own_line() {
        let g = Genealogy::init(30);
        assert_eq!(g.len(), 30);
        assert_eq!(g.minted(), 30);
        assert_eq!(g.distinct_founders(0..30), 30, "every row founds its own line");
        for r in 0..30 {
            let m = g.row(r);
            assert_eq!((m.age, m.founder, m.founder_generation, m.founder_origin), (0, m.id, 0, Origin::Init));
            let chain = g.walk(m.id);
            assert_eq!(chain.len(), 1, "a founder's chain is itself");
            assert_eq!(chain[0].0, m.id);
            assert_eq!(chain[0].1.parent, u64::MAX);
        }
    }

    #[test]
    fn an_elite_keeps_its_id_and_ages_by_one_and_an_offspring_is_a_new_id_at_its_parents_age_plus_one() {
        let isl = islands();
        let mut g = Genealogy::init(30);
        let before = g.snapshot();
        // Every row cloned from row 7 of its island's range, so the parentage is known.
        let parent: Vec<u32> = (0..30).map(|r| if r < 20 { 7 } else { 27 }).collect();
        g.advance(&parent, &isl, 1);
        // the two elites of island 0 keep the ids that were in rows 0 and 1
        for r in 0..2 {
            assert_eq!(g.row(r).id, before[7].id, "an elite is cloned from the selected row and keeps ITS id");
            assert_eq!(g.row(r).age, before[7].age + 1, "an elite ages by one");
        }
        for r in 2..20 {
            let m = g.row(r);
            assert_ne!(m.id, before[7].id, "an offspring is a new individual");
            assert_eq!(m.age, before[7].age + 1);
            assert_eq!(m.founder, before[7].founder, "it keeps its parent's line");
            let edge = g.edge(m.id).expect("an edge");
            assert_eq!((edge.parent, edge.origin), (before[7].id, Origin::Vary));
        }
        // ages accumulate: ten generations of the same parentage is age ten
        for generation in 2..=10u32 {
            g.advance(&parent, &isl, generation);
        }
        assert_eq!(g.row(5).age, 10);
        assert_eq!(g.row(0).age, 10, "the elite line is as old as the fit");
    }

    #[test]
    fn an_offspring_records_the_row_it_was_mated_with_as_its_other_ancestor() {
        let isl = islands();
        let mut g = Genealogy::init(30);
        let before = g.snapshot();
        let parent: Vec<u32> = (0..30).collect();
        g.advance(&parent, &isl, 1);
        // island 0's offspring start at row 2 and are paired (2,3), (4,5), ...
        assert_eq!(mate_of(isl[0], 2), Some(3));
        assert_eq!(mate_of(isl[0], 3), Some(2));
        assert_eq!(mate_of(isl[0], 0), None, "an elite has no mate");
        let edge = g.edge(g.row(2).id).expect("an edge");
        assert_eq!(edge.other, before[3].id, "the mate's id is the other ancestor");
        let edge = g.edge(g.row(3).id).expect("an edge");
        assert_eq!(edge.other, before[2].id);
        // an elite's edge is not minted at all — the row keeps the id it had
        assert_eq!(g.edge(g.row(0).id).expect("an edge").origin, Origin::Init);
    }

    #[test]
    fn the_pumps_refill_is_age_zero_and_its_promotion_carries_the_line() {
        let mut g = Genealogy::init(30);
        let parent: Vec<u32> = (0..30).collect();
        for generation in 1..=5u32 {
            g.advance(&parent, &islands(), generation);
        }
        assert_eq!(g.row(9).age, 5);
        let promoted_from = g.row(9);
        g.promote(9, 25, 5);
        let to = g.row(25);
        assert_ne!(to.id, promoted_from.id, "the promotion is a clone: the source stays where it was");
        assert_eq!(to.age, promoted_from.age, "no generation passed, so no ageing");
        assert_eq!(to.founder, promoted_from.founder, "it carries its line into the champion island");
        assert_eq!(g.edge(to.id).expect("an edge").origin, Origin::PumpPromote);
        // the refill
        let before = g.snapshot();
        g.moved(&before, 9, 0, Origin::PumpKeep, 5);
        assert_eq!(g.row(0).id, before[9].id, "a keeper is the same individual in another row");
        assert_eq!(g.row(0).age, 5);
        for r in 1..20 {
            g.fresh(r, Origin::PumpRefill, 5);
        }
        for r in 1..20 {
            let m = g.row(r);
            assert_eq!((m.age, m.founder, m.founder_generation, m.founder_origin), (0, m.id, 5, Origin::PumpRefill), "a refilled row is new");
        }
    }

    #[test]
    fn a_cross_arrival_is_a_clone_of_another_pairs_champion_and_keeps_its_age() {
        let mut g = Genealogy::init(30);
        let parent: Vec<u32> = (0..30).collect();
        for generation in 1..=7u32 {
            g.advance(&parent, &islands(), generation);
        }
        let before = g.snapshot();
        g.moved(&before, 25, 3, Origin::CrossArrival, 7);
        let m = g.row(3);
        assert_ne!(m.id, before[25].id, "the migrant is a clone; the champion island is not touched");
        assert_eq!((m.age, m.founder), (before[25].age, before[25].founder));
        assert_eq!(g.edge(m.id).expect("an edge").origin, Origin::CrossArrival);
        g.fresh(4, Origin::CrossFresh, 7);
        assert_eq!(g.row(4).age, 0);
        assert_eq!(g.row(4).founder_origin, Origin::CrossFresh);
    }

    #[test]
    fn a_snap_write_back_keeps_the_individual_and_records_the_event() {
        let mut g = Genealogy::init(30);
        let parent: Vec<u32> = (0..30).collect();
        g.advance(&parent, &islands(), 1);
        let before = g.row(5);
        let record = g.snap(5, 1);
        assert_eq!(g.row(5), before, "a repaired gene is the same individual");
        assert_eq!((record.id, record.age, record.origin), (before.id, before.age, Origin::SnapWriteback));
    }

    #[test]
    fn the_walk_ends_at_a_founder_whose_origin_is_an_arrival() {
        let isl = islands();
        let mut g = Genealogy::init(30);
        let parent: Vec<u32> = (0..30).map(|r| if r < 20 { 4 } else { 24 }).collect();
        for generation in 1..=12u32 {
            g.advance(&parent, &isl, generation);
            // a pump beat: the intake keeps one row and refills the rest
            if generation % 4 == 0 {
                let before = g.snapshot();
                g.promote(2, 29, generation);
                g.moved(&before, 2, 0, Origin::PumpKeep, generation);
                for r in 1..20 {
                    g.fresh(r, Origin::PumpRefill, generation);
                }
            }
        }
        // the champion island's line runs back to the initial draw
        let chain = g.walk(g.row(24).id);
        let (founder_id, founder) = *chain.last().expect("a founder");
        assert_eq!(founder.parent, u64::MAX);
        assert!(founder.origin.is_arrival(), "a line begins at an arrival, not at {}", founder.origin);
        assert_eq!(founder.origin, Origin::Init);
        assert_eq!(founder_id, g.row(24).founder, "the carried founder is the one the walk reaches");
        assert_eq!(founder.founder_generation, g.row(24).founder_generation);
        // every step of the chain links to the next
        for pair in chain.windows(2) {
            assert_eq!(pair[0].1.parent, pair[1].0, "the chain is linked parent to child");
        }
        // and a refilled intake row's line begins at the pump beat that drew it
        let refilled = g.row(5);
        assert_eq!(refilled.founder_origin, Origin::PumpRefill);
        let chain = g.walk(refilled.id);
        assert_eq!(chain.last().expect("a founder").1.origin, Origin::PumpRefill);
        assert_eq!(chain.last().expect("a founder").1.generation, 12, "drawn at the last pump beat");
    }

    #[test]
    fn every_row_of_a_generation_holds_a_different_id() {
        let isl = islands();
        let mut g = Genealogy::init(30);
        for generation in 1..=30u32 {
            // a parent array that repeats rows, as real tournaments do
            let parent: Vec<u32> = (0..30).map(|r| if r < 20 { (r * 7) % 20 } else { 20 + (r * 3) % 10 }).collect();
            g.advance(&parent, &isl, generation);
            let mut ids: Vec<u64> = (0..30).map(|r| g.row(r).id).collect();
            ids.sort_unstable();
            let n = ids.len();
            ids.dedup();
            assert_eq!(ids.len(), n, "generation {generation}: two rows share an id");
        }
    }

    #[test]
    fn the_age_distribution_and_the_founder_count_read_the_population() {
        let isl = islands();
        let mut g = Genealogy::init(30);
        let parent: Vec<u32> = (0..30).map(|r| if r < 20 { 0 } else { 20 }).collect();
        for generation in 1..=6u32 {
            g.advance(&parent, &isl, generation);
        }
        assert_eq!(g.ages(0..30), Some((6, 6, 6, 6.0)), "one line, all the same age");
        assert_eq!(g.distinct_founders(0..30), 2, "island 0 descends from row 0, island 1 from row 20");
        for r in 10..20 {
            g.fresh(r, Origin::PumpRefill, 6);
        }
        let (min, _, max, _) = g.ages(0..30).expect("ages");
        assert_eq!((min, max), (0, 6), "the refill puts age-0 material beside the old");
        assert_eq!(g.distinct_founders(0..30), 12, "ten new lines beside the two");
    }
}
