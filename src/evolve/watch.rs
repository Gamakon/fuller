//! `hff-watch`'s STATE MACHINE — the screen's whole model, with no terminal in it.
//!
//! The brief requires that a replayed file and a followed live stream go through
//! ONE parser and ONE state machine, and this is that machine. `src/bin/
//! hff_watch.rs` is the terminal: it polls for records, feeds them in here, asks
//! for rows and draws them. Everything that decides WHAT is on screen — the sort,
//! the selection, the gains, the sparkline windows, whether the run is live,
//! stale or finished — is here, where a test can reach it without a tty.
//!
//! Two ideas run through all of it.
//!
//! **HFF is lower-is-better.** Every comparison, colour, bar and sparkline in
//! this file obeys that, and none of them treats HFF as a percentage: improvement
//! is drawn in LOG SPACE, because a fit goes from 1e-1 to 1e-5 and a linear bar
//! of that is a bar that is empty and then full with nothing in between.
//!
//! **A gap is a gap.** `NEW`, `EXTINCT` and `MERGED` are words on the screen, not
//! numbers: a cohort that has just appeared has no previous best, and printing
//! `+0.0%` or `—0.0%` for it would be inventing a measurement. The same rule
//! sends every undefined metric to a literal dash rather than a zero.

use std::collections::BTreeMap;

use super::telemetry::{CohortRow, IslandRow, Parsed, Record, RunEnd, RunStart, Snapshot};

/// How many snapshots of history a cohort keeps for its sparkline and its gain.
/// Enough to see a young cohort's whole climb at a 10-generation beat, small
/// enough that a long fit's memory is bounded — a viewer must not grow without
/// limit while it watches.
pub const HISTORY: usize = 240;

/// How many snapshots an EXTINCT cohort keeps its row in the table for.
///
/// Watching a cohort die is the point of the table, so it must not simply vanish
/// between redraws — but a 450-second fit at a pump beat of 4 buries the living
/// rows under thousands of dead ones within a minute, which is worse. Three
/// beats is long enough to read "EXTINCT" and gone before it crowds anything.
pub const EXTINCT_LINGER: usize = 3;

/// How many snapshots after its death a cohort's HISTORY is kept. Longer than
/// the row lingers, because a cohort can come back — a pump can refill rows with
/// an old label — and its trace should resume rather than restart. Past this it
/// is dropped: the history map is keyed by every cohort the stream has EVER
/// shown, and a long fit mints thousands of them.
pub const HISTORY_LINGER: usize = 30;

/// How stale a live stream may get before the screen says so. Four times the
/// writer's own minimum gap: a fit that is merely between beats is not stale,
/// and one that has stopped writing for eight seconds is.
pub const STALE_AFTER: std::time::Duration = std::time::Duration::from_secs(8);

/// What the run is doing, as the header badge says it.
///
/// FINISHED and STALE are deliberately different states and not one "not live":
/// the brief requires that a finished run look different from telemetry that has
/// merely stopped arriving, because the operator's next action is different. A
/// finished run is a result; a stale one is a question about the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Liveness {
    /// Records are arriving, or this is a replay being stepped through.
    Live,
    /// A `run_end` was seen: the fit is over and this is its final state.
    Finished,
    /// Nothing has arrived for `STALE_AFTER` and no `run_end` came. The last
    /// good frame is still on screen, and it is marked.
    Stale,
}

/// How the cohort table is ordered. `s` cycles it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sort {
    /// Best HFF, lowest first — lower is better, so this is the leader board.
    Best,
    /// Gain over the last window, largest first. THE SORT THAT MATTERS: a young
    /// cohort improving fast sits far down the Best table behind a converged
    /// elder, and this is what brings it up where an operator can see it.
    Gain,
    Rows,
}

impl Sort {
    pub fn next(self) -> Sort {
        match self {
            Sort::Best => Sort::Gain,
            Sort::Gain => Sort::Rows,
            Sort::Rows => Sort::Best,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Sort::Best => "best HFF",
            Sort::Gain => "gain",
            Sort::Rows => "rows",
        }
    }
}

/// A cohort's gain over the last window, or the reason it has none.
///
/// `Gain::New` and `Gain::Extinct` are WORDS on the screen. The brief is
/// explicit: show them "explicitly rather than assigning a bogus numerical
/// delta", and this type is how that is made impossible to get wrong — there is
/// no number in those variants to print by accident.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gain {
    /// `100 × (previous_best / current_best − 1)%`. Signed: negative means the
    /// cohort's current best is WORSE than it was, which happens when the row
    /// holding the best was refilled.
    Percent(f64),
    /// The cohort appeared in this snapshot and has no previous best.
    New,
    /// It was here last snapshot and has no rows now.
    Extinct,
    /// Its label is at or past `cohort_merge`, so it is part of the merged band
    /// and its identity is an aggregate — a gain on it is a gain on the band.
    Merged,
    /// It has rows but none of them scored, so there is no best to compare.
    Unscored,
}

/// One cohort's history: enough to draw a sparkline and compute a gain, and
/// nothing else.
#[derive(Clone, Debug, Default)]
pub struct CohortHistory {
    /// Best HFF at each snapshot the cohort was alive for, oldest first, capped
    /// at `HISTORY`.
    pub bests: Vec<Option<f64>>,
    /// The snapshot generation each entry belongs to, so a sparkline can state
    /// the window it spans rather than implying one.
    pub generations: Vec<u32>,
    /// The lowest HFF this cohort has EVER held, which is not its current best:
    /// the brief asks for best-ever separately so a current best is never
    /// mistaken for a record.
    pub best_ever: Option<f64>,
    /// The snapshot generation the cohort was first seen at. The engine's label
    /// IS the pump beat it was born on, but a viewer attaching mid-run sees a
    /// cohort for the first time later than that, and both facts are useful.
    pub first_seen: u32,
    /// True once the cohort has been seen and then vanished.
    pub extinct: bool,
    /// How many snapshots it has been extinct for. It is a COUNT OF SNAPSHOTS
    /// rather than of generations because the table's lingering is measured in
    /// redraws an operator sees, not in the fit's own clock — a fit that runs
    /// ten times faster should not bury its dead ten times sooner.
    pub extinct_for: usize,
}

/// A row of the cohort table, ready to draw.
#[derive(Clone, Debug)]
pub struct CohortView {
    pub id: u32,
    pub birth_generation: u32,
    pub rows: u32,
    pub best_hff: Option<f64>,
    pub best_ever: Option<f64>,
    pub gain: Gain,
    pub nan_rows: u32,
    /// The bars, lowest-HFF-is-highest, over `spark_span`.
    pub spark: Vec<u64>,
    /// The generations the sparkline spans, as (first, last). The brief requires
    /// a sparkline to state its window.
    pub spark_span: Option<(u32, u32)>,
    pub extinct: bool,
}

/// The whole screen's model.
pub struct WatchState {
    /// The fixed facts. None until a `run_start` has been seen — a viewer that
    /// attaches to a stream mid-run and never gets one says so instead of
    /// inventing a population to check invariants against.
    pub start: Option<RunStart>,
    /// THE LAST GOOD FRAME. A malformed line, a partial one, or a stream that
    /// stops never touches this: it is only ever replaced by a snapshot that
    /// parsed whole. That is the brief's "does not ... corrupt the last complete
    /// frame", enforced by the type — there is no path that half-updates it.
    pub snapshot: Option<Snapshot>,
    /// The snapshot before the current one, which is what a gain is measured
    /// against and what a birth or an extinction is a difference from.
    pub previous: Option<Snapshot>,
    pub end: Option<RunEnd>,
    /// Per cohort, its history. Keyed by ID, so it survives the table reordering.
    pub history: BTreeMap<u32, CohortHistory>,
    /// The events the stream has sent, newest last, capped.
    pub events: Vec<(u32, String)>,
    /// The most recent `model` record, for the `m` viewer.
    pub model: Option<super::telemetry::Model>,
    /// THE SELECTION, held BY ID. The brief: "Keep the selected cohort by ID when
    /// rows reorder. Never auto-scroll away from a selected row." An index would
    /// mean the selection slides under the operator every time the sort changes
    /// or a cohort is born, which is precisely the failure this avoids.
    pub selected: Option<u32>,
    /// Which island is focused (`Tab`). An index into the snapshot's islands.
    pub island: usize,
    /// THE TABLE'S SOURCE: false is the global cohort totals, true is the
    /// SELECTED ISLAND's own split. The brief: "selection focuses an island" and
    /// "`g` shows global cohorts". The per-island split is the whole reason the
    /// engine fills the brief's `null` island values, so the screen has to be
    /// able to show it — a viewer that only ever draws the global totals has
    /// thrown that work away.
    ///
    /// Gain and the sparklines stay keyed on the GLOBAL cohort id and are
    /// measured against the global history, which the table's title says, so a
    /// focused gain is never read as an island-local one.
    pub island_focus: bool,
    pub sort: Sort,
    /// `/`'s filter over cohort IDs and island names; empty means everything.
    pub filter: String,
    /// True while `/` is being typed into.
    pub filtering: bool,
    /// `Space`: the redraw is paused. It pauses the SCREEN and nothing else —
    /// records still accumulate, and the fit is never touched.
    pub paused: bool,
    /// Malformed lines seen, and rotations. Shown, because a viewer that is
    /// silently skipping input is a viewer that is lying.
    pub bad_lines: u32,
    pub rotations: u32,
    /// When the last record arrived, for staleness. None for a `--file` replay,
    /// which is never stale: it is a recording, and a recording is as fresh as
    /// it will ever be.
    pub last_record: Option<std::time::Instant>,
    /// Whether this is a live follow (staleness applies) or a replay.
    pub following: bool,
}

impl WatchState {
    pub fn new(following: bool) -> WatchState {
        WatchState {
            start: None,
            snapshot: None,
            previous: None,
            end: None,
            history: BTreeMap::new(),
            events: Vec::new(),
            model: None,
            selected: None,
            island: 0,
            island_focus: false,
            sort: Sort::Best,
            filter: String::new(),
            filtering: false,
            paused: false,
            bad_lines: 0,
            rotations: 0,
            last_record: None,
            following,
        }
    }

    /// Feed one parsed line in. A `Parsed::Bad` is counted and otherwise IGNORED:
    /// it does not clear the frame, does not reset the history, does not move the
    /// selection. That is the whole of the brief's robustness requirement, and it
    /// is one branch because the frame is only ever written by a whole record.
    pub fn apply(&mut self, parsed: Parsed) {
        match parsed {
            Parsed::Bad(_) => {
                self.bad_lines += 1;
            }
            Parsed::Ok(record) => {
                self.last_record = Some(std::time::Instant::now());
                self.apply_record(*record);
            }
        }
    }

    pub fn apply_record(&mut self, record: Record) {
        match record {
            Record::RunStart(s) => {
                // A run_start after a run has already been seen is a NEW run in
                // the same file: the history belongs to the old one and keeping
                // it would stitch two fits into one sparkline.
                if self.start.is_some() {
                    let following = self.following;
                    let (sort, filter) = (self.sort, std::mem::take(&mut self.filter));
                    let (bad, rot) = (self.bad_lines, self.rotations);
                    *self = WatchState { sort, filter, bad_lines: bad, rotations: rot, ..WatchState::new(following) };
                }
                self.start = Some(s);
            }
            Record::Snapshot(s) => {
                self.record_history(&s);
                self.previous = self.snapshot.take();
                self.snapshot = Some(s);
                // Nothing is selected yet: start on the best cohort. After that
                // the selection is the operator's and is never moved by a new
                // snapshot, however the table reorders.
                if self.selected.is_none() {
                    self.selected = self.rows().first().map(|r| r.id);
                }
            }
            Record::Event(e) => {
                self.events.push((e.header.generation, e.message));
                // Bounded: a long fit's events must not grow the viewer without
                // limit, and only the recent ones are ever on screen.
                let overflow = self.events.len().saturating_sub(200);
                self.events.drain(..overflow);
            }
            Record::Model(m) => self.model = Some(m),
            Record::RunEnd(e) => self.end = Some(e),
        }
    }

    /// Fold a snapshot into the per-cohort history. Cohorts that have gone get a
    /// `None` entry rather than a gap, so a sparkline's bars line up with the
    /// generations beside them and a cohort's disappearance is visible in its own
    /// trace instead of being silently compacted away.
    fn record_history(&mut self, s: &Snapshot) {
        let generation = s.header.generation;
        let present: std::collections::BTreeSet<u32> = s.global_cohorts.iter().map(|c| c.id).collect();
        for c in &s.global_cohorts {
            let h = self.history.entry(c.id).or_insert_with(|| CohortHistory { first_seen: generation, ..CohortHistory::default() });
            // A cohort that comes back — a pump refilling rows with an old label
            // — resumes its trace rather than starting a new one.
            h.extinct = false;
            h.extinct_for = 0;
            h.bests.push(c.best_hff);
            h.generations.push(generation);
            if let Some(v) = c.best_hff {
                h.best_ever = Some(h.best_ever.map_or(v, |b: f64| b.min(v)));
            }
            let overflow = h.bests.len().saturating_sub(HISTORY);
            h.bests.drain(..overflow);
            h.generations.drain(..overflow);
        }
        for (id, h) in &mut self.history {
            if !present.contains(id) {
                h.extinct = true;
                h.extinct_for += 1;
                h.bests.push(None);
                h.generations.push(generation);
                let overflow = h.bests.len().saturating_sub(HISTORY);
                h.bests.drain(..overflow);
                h.generations.drain(..overflow);
            }
        }
        // THE MAP IS BOUNDED. It is keyed by every cohort the stream has ever
        // shown, and a long fit mints one per pump beat — thousands of them, each
        // with its own history vector. A cohort long dead is dropped; one that
        // comes back before then keeps its trace.
        self.history.retain(|_, h| h.extinct_for <= HISTORY_LINGER);
    }

    pub fn liveness(&self) -> Liveness {
        if self.end.is_some() {
            return Liveness::Finished;
        }
        match self.last_record {
            Some(t) if self.following && t.elapsed() > STALE_AFTER => Liveness::Stale,
            _ => Liveness::Live,
        }
    }

    /// Whether cohorts are on at all. `cohort_merge = 0` means VIRTUAL ALPS is
    /// off and every row carries the same label, and a table of one row called
    /// "c0: everything" is worse than saying so.
    pub fn cohorts_on(&self) -> bool {
        self.start.as_ref().is_some_and(|s| s.cohort_merge > 0)
    }

    /// The cohort table's rows, filtered and sorted.
    ///
    /// The sort is TOTAL and deterministic: ties break on the ID, so the same
    /// snapshot always produces the same order and a row does not jitter between
    /// redraws because two cohorts share a best.
    pub fn rows(&self) -> Vec<CohortView> {
        let Some(s) = self.snapshot.as_ref() else { return Vec::new() };
        let merge = self.start.as_ref().map_or(u32::MAX, |r| r.cohort_merge);
        // The focused island's own split, or the global totals. The per-island
        // rows are the brief's filled-in nulls, and this is where they reach the
        // screen.
        let source: &[CohortRow] = match self.island_focus.then(|| s.islands.get(self.island)).flatten() {
            Some(island) => &island.cohorts,
            None => &s.global_cohorts,
        };
        let mut rows: Vec<CohortView> = source.iter().filter(|c| self.matches(c)).map(|c| self.view(c, merge)).collect();
        // A cohort that was alive last snapshot and has no rows now still gets a
        // row, marked EXTINCT, for a few beats: watching a cohort DIE is the
        // point of the table, and a row that simply vanishes between redraws
        // shows nothing. Only in the GLOBAL view — a cohort leaving one island
        // has not died, it has moved, and marking that EXTINCT would be a lie.
        for (&id, h) in self.history.iter().filter(|_| !self.island_focus) {
            if h.extinct && h.extinct_for <= EXTINCT_LINGER && !rows.iter().any(|r| r.id == id) && self.matches_id(id) {
                rows.push(CohortView {
                    id,
                    birth_generation: id,
                    rows: 0,
                    best_hff: None,
                    best_ever: h.best_ever,
                    gain: Gain::Extinct,
                    nan_rows: 0,
                    spark: spark_of(&h.bests),
                    spark_span: span_of(h),
                    extinct: true,
                });
            }
        }
        match self.sort {
            // Lower HFF first — lower is better. A cohort with no best at all
            // sorts last, because "unknown" is not "excellent".
            Sort::Best => rows.sort_by(|a, b| {
                a.best_hff.unwrap_or(f64::INFINITY).total_cmp(&b.best_hff.unwrap_or(f64::INFINITY)).then(a.id.cmp(&b.id))
            }),
            // Largest gain first. A cohort with no gain (NEW, EXTINCT, MERGED,
            // unscored) sorts below every cohort that has one: it is not a zero,
            // and it must not sit among the regressions as though it were.
            Sort::Gain => rows.sort_by(|a, b| {
                let key = |g: &Gain| match g {
                    Gain::Percent(v) => *v,
                    _ => f64::NEG_INFINITY,
                };
                key(&b.gain).total_cmp(&key(&a.gain)).then(a.id.cmp(&b.id))
            }),
            Sort::Rows => rows.sort_by(|a, b| b.rows.cmp(&a.rows).then(a.id.cmp(&b.id))),
        }
        rows
    }

    fn matches(&self, c: &CohortRow) -> bool {
        self.matches_id(c.id)
    }

    fn matches_id(&self, id: u32) -> bool {
        if self.filter.is_empty() {
            return true;
        }
        let needle = self.filter.trim().trim_start_matches('c');
        format!("{id}").contains(needle)
    }

    fn view(&self, c: &CohortRow, merge: u32) -> CohortView {
        let h = self.history.get(&c.id);
        let previous = self
            .previous
            .as_ref()
            .and_then(|p| p.global_cohorts.iter().find(|q| q.id == c.id))
            .and_then(|q| q.best_hff);
        let gain = match (c.best_hff, previous) {
            _ if c.id >= merge && merge > 0 => Gain::Merged,
            (None, _) => Gain::Unscored,
            // No previous best of its own: the cohort is new to this stream.
            (Some(_), None) => Gain::New,
            // 100 x (previous / current - 1): positive when the current best is
            // SMALLER, which is what improvement means for HFF.
            (Some(now), Some(before)) if now > 0.0 => Gain::Percent(100.0 * (before / now - 1.0)),
            (Some(_), Some(_)) => Gain::Unscored,
        };
        CohortView {
            id: c.id,
            birth_generation: c.birth_generation,
            rows: c.rows,
            best_hff: c.best_hff,
            best_ever: h.and_then(|h| h.best_ever),
            gain,
            nan_rows: c.nan_rows,
            spark: h.map(|h| spark_of(&h.bests)).unwrap_or_default(),
            spark_span: h.and_then(span_of),
            extinct: false,
        }
    }

    /// The selected cohort's row, if it is still in the table. When it is not —
    /// it went extinct, or the filter excludes it — the SELECTION IS NOT MOVED:
    /// this returns None and the detail pane says why. Snapping to row 0 would be
    /// auto-scrolling away from the selection, which the brief forbids.
    pub fn selected_row(&self) -> Option<CohortView> {
        let id = self.selected?;
        self.rows().into_iter().find(|r| r.id == id)
    }

    /// Move the selection one row down (`j`) or up (`k`). Operates on the CURRENT
    /// order, and lands on an ID — so the next reorder keeps it.
    pub fn move_selection(&mut self, down: bool) {
        let rows = self.rows();
        if rows.is_empty() {
            return;
        }
        let at = self.selected.and_then(|id| rows.iter().position(|r| r.id == id));
        let next = match (at, down) {
            (None, _) => 0,
            (Some(i), true) => (i + 1).min(rows.len() - 1),
            (Some(i), false) => i.saturating_sub(1),
        };
        self.selected = Some(rows[next].id);
    }

    /// Move the island focus, wrapping (`Tab` / `Shift-Tab`).
    pub fn move_island(&mut self, forward: bool) {
        let n = self.snapshot.as_ref().map_or(0, |s| s.islands.len());
        if n == 0 {
            return;
        }
        self.island = if forward { (self.island + 1) % n } else { (self.island + n - 1) % n };
    }

    pub fn islands(&self) -> &[IslandRow] {
        self.snapshot.as_ref().map_or(&[], |s| s.islands.as_slice())
    }

    /// HOW MANY GENERATIONS A GAIN SPANS. A snapshot is written at the progress
    /// beat but no more often than once a second, so the window between two of
    /// them is whatever the fit's speed made it — 45 generations on a fast fit,
    /// 10 on a slow one. The column header states this number rather than
    /// implying a fixed "per 10 generations", which would be wrong on most runs.
    pub fn gain_window(&self) -> Option<u32> {
        let now = self.snapshot.as_ref()?.header.generation;
        let before = self.previous.as_ref()?.header.generation;
        Some(now.saturating_sub(before))
    }

    /// The global best's change since the previous snapshot, as a gain — the same
    /// signed percentage the cohort table uses, so one number means one thing
    /// everywhere on the screen.
    pub fn global_gain(&self) -> Gain {
        let now = self.snapshot.as_ref().and_then(|s| s.global.best_hff);
        let before = self.previous.as_ref().and_then(|s| s.global.best_hff);
        match (now, before) {
            (Some(now), Some(before)) if now > 0.0 => Gain::Percent(100.0 * (before / now - 1.0)),
            (Some(_), None) => Gain::New,
            _ => Gain::Unscored,
        }
    }

    /// THE SEARCH PULSE, as three labelled meters and NO composite.
    ///
    /// The brief permits this: "A simpler first version can ship without a
    /// composite, showing three labelled meters instead." That is what is here,
    /// and it is the honest version. A single 0–100 number that mixes improvement
    /// rate, survival and coverage reads as a probability of success to everyone
    /// who sees it, which is exactly what the brief says it is not — and the
    /// weights (45/35/20) are a choice nobody has measured. Three meters cannot
    /// be misread that way: each one is its own measurement with its own name.
    ///
    /// Each is `(label, value in [0,1] or None, the reason it is None)`.
    pub fn pulse(&self, island: usize) -> [(&'static str, Option<f64>, &'static str); 3] {
        let s = self.snapshot.as_ref();
        let row = s.and_then(|s| s.islands.get(island));
        // IMPROVEMENT, in LOG SPACE. A fit's HFF falls by orders of magnitude, so
        // a linear rate is a meter that is pinned at one end for the whole run.
        // The reference pace is one decade per hundred generations, stated here
        // rather than buried: it is a scale for a meter, not a claim about what
        // a healthy fit does.
        let no_island_best = row.is_some_and(|r| r.best_hff.is_none());
        let improvement = match (row.and_then(|r| r.best_hff), self.previous.as_ref().and_then(|p| p.islands.get(island)).and_then(|r| r.best_hff)) {
            (Some(now), Some(before)) if now > 0.0 && before > 0.0 => {
                let gens = s.map_or(0, |s| s.header.generation).saturating_sub(self.previous.as_ref().map_or(0, |p| p.header.generation));
                let decades = (before / now).log10();
                (gens > 0).then(|| (decades / (f64::from(gens) / 100.0) / REFERENCE_DECADES_PER_100_GEN).clamp(0.0, 1.0))
            }
            _ => None,
        };
        // FRESH-LINE SURVIVAL after a pump beat. The engine does not yet publish
        // per-island survival counts, and `pumps_since = 0` means there was no
        // pump in this window at all — which the brief says explicitly must show
        // `—` rather than be called zero. Both cases are honest dashes here.
        let survival_reason = match s.map(|s| s.pumps_since) {
            Some(0) => "no pump in this window",
            _ => "not instrumented",
        };
        // FINITE COVERAGE: the fraction of the island's rows that produced a
        // score at all. It is the PROTECTED evaluator's coverage, and it is
        // labelled as such — the brief warns that the submitted/plain
        // expression's finite fraction is a DIFFERENT metric, and this screen
        // does not claim to show that one.
        //
        // `nan_rows: None` is NOT EMITTED, and coverage is then None too. A
        // producer that never counted the unscored rows would otherwise be
        // reported as "100% scored" — a fabricated island metric on a stream
        // that said nothing, which is the first thing the brief forbids.
        let coverage = row.and_then(|r| {
            let nan = r.nan_rows?;
            let scored = r.rows.saturating_sub(nan);
            Some(f64::from(scored) / f64::from(r.rows.max(1)))
        });
        [
            (
                "improvement",
                improvement,
                if no_island_best { "island best not emitted" } else { "no previous window" },
            ),
            ("fresh lines", None, survival_reason),
            ("scored rows", coverage, "unscored rows not emitted"),
        ]
    }
}

/// The improvement meter's full-scale mark: one decade of HFF per hundred
/// generations. It is a SCALE, stated and adjustable, not a target — the brief
/// asks for a reference pace measured from real runs and kept explicit, and this
/// is the placeholder with its name on it until that measurement exists.
pub const REFERENCE_DECADES_PER_100_GEN: f64 = 1.0;

/// A sparkline's bars from a history of bests: LOWER HFF IS A HIGHER BAR.
///
/// Normalised over `-log10(hff)` across the window, because HFF spans orders of
/// magnitude and a linear normalisation would draw a run that went from 1e-1 to
/// 1e-5 as one step. A window with only one distinct value is drawn flat at
/// mid-height rather than full: a single measurement is not a record high.
pub fn spark_of(bests: &[Option<f64>]) -> Vec<u64> {
    let logs: Vec<Option<f64>> = bests.iter().map(|b| b.filter(|v| *v > 0.0).map(|v| -v.log10())).collect();
    let present: Vec<f64> = logs.iter().flatten().copied().collect();
    if present.is_empty() {
        return Vec::new();
    }
    let (lo, hi) = present.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
    let span = hi - lo;
    logs.iter()
        .map(|v| match v {
            // A gap in the trace is a zero bar: the cohort had no best then, and
            // carrying the last value forward would draw a life it did not live.
            None => 0,
            Some(v) if span <= 0.0 => 4,
            Some(v) => 1 + ((v - lo) / span * 7.0).round() as u64,
        })
        .collect()
}

/// The generations a sparkline spans, so it can say so. None when there is
/// nothing to draw.
pub fn span_of(h: &CohortHistory) -> Option<(u32, u32)> {
    Some((*h.generations.first()?, *h.generations.last()?))
}

/// A number for the screen, or a dash. Every undefined metric goes through here,
/// so the dash is in ONE place and no format string can print a zero for a thing
/// that was never measured.
pub fn or_dash(v: Option<f64>, digits: usize) -> String {
    v.map_or_else(|| "—".to_string(), |v| format!("{v:.digits$e}"))
}

/// A gain as the screen says it: a signed percentage, or the WORD that explains
/// why there is no percentage.
pub fn gain_text(g: Gain) -> String {
    match g {
        Gain::Percent(v) if v >= 0.0 => format!("+{v:.1}%"),
        Gain::Percent(v) => format!("{v:.1}%"),
        Gain::New => "NEW".to_string(),
        Gain::Extinct => "EXTINCT".to_string(),
        Gain::Merged => "MERGED".to_string(),
        Gain::Unscored => "—".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolve::telemetry::parse_stream;

    const FIXTURE: &str = include_str!("../../tests/fixtures/telemetry_v1.jsonl");

    fn replayed() -> WatchState {
        let (records, bad) = parse_stream(FIXTURE);
        let mut state = WatchState::new(false);
        state.bad_lines = bad;
        for r in records {
            state.apply_record(r);
        }
        state
    }

    /// A recorded stream replays into a complete frame, and the two deliberate
    /// bad lines are counted rather than fatal.
    #[test]
    fn the_fixture_replays_into_a_frame() {
        let state = replayed();
        assert!(state.start.is_some(), "the run_start was read");
        assert!(state.snapshot.is_some(), "a frame was drawn");
        assert_eq!(state.bad_lines, 2);
        assert_eq!(state.liveness(), Liveness::Finished, "the fixture ends with run_end");
        assert!(!state.rows().is_empty(), "the cohort table has rows");
        assert!(state.model.is_some(), "the model viewer has something to show");
        assert!(!state.events.is_empty());
    }

    /// A malformed line does not touch the last good frame. The frame before and
    /// after a garbage line is the same frame, bit for bit in the numbers that
    /// matter.
    #[test]
    fn a_malformed_line_does_not_corrupt_the_frame() {
        let mut state = replayed();
        let before = state.snapshot.as_ref().map(|s| (s.header.seq, s.global.best_hff, s.global_cohorts.len()));
        let rows_before = state.rows().len();
        let selected = state.selected;
        for _ in 0..5 {
            state.apply(Parsed::Bad("not json".to_string()));
        }
        let after = state.snapshot.as_ref().map(|s| (s.header.seq, s.global.best_hff, s.global_cohorts.len()));
        assert_eq!(before, after, "a bad line moved the frame");
        assert_eq!(state.rows().len(), rows_before);
        assert_eq!(state.selected, selected, "a bad line moved the selection");
        assert_eq!(state.bad_lines, 7);
    }

    /// THE SELECTION IS AN ID. It survives every sort, and it survives a new
    /// snapshot reordering the table under it — which is the brief's "keep the
    /// selected cohort by ID when rows reorder; never auto-scroll away".
    #[test]
    fn the_selection_survives_a_reorder_and_a_new_snapshot() {
        let (records, _) = parse_stream(FIXTURE);
        let mut state = WatchState::new(false);
        let snapshots: Vec<_> = records.iter().filter(|r| matches!(r, Record::Snapshot(_))).cloned().collect();
        assert!(snapshots.len() >= 3, "the fixture has snapshots to reorder");
        for r in records.iter().take_while(|r| !matches!(r, Record::RunEnd(_))).cloned() {
            state.apply_record(r);
        }
        // Pick a cohort that is NOT the best one, so a snap-to-top would show.
        let rows = state.rows();
        let target = rows.last().expect("a last row").id;
        state.selected = Some(target);
        for sort in [Sort::Gain, Sort::Rows, Sort::Best, Sort::Gain] {
            state.sort = sort;
            let _ = state.rows();
            assert_eq!(state.selected, Some(target), "the sort moved the selection");
        }
        // And a new snapshot, which reorders the table, leaves it alone.
        state.apply_record(snapshots.last().expect("a snapshot").clone());
        assert_eq!(state.selected, Some(target), "a snapshot moved the selection");
    }

    /// Gain is `100 x (previous / current - 1)`, positive when HFF FELL, and a
    /// cohort with no previous best is the WORD `NEW`, never a number.
    #[test]
    fn gain_is_signed_and_a_new_cohort_is_a_word() {
        assert_eq!(gain_text(Gain::Percent(0.0)), "+0.0%");
        // HFF halved: 100 x (2/1 - 1) = +100%.
        let halved = Gain::Percent(100.0 * (2e-3 / 1e-3 - 1.0));
        assert_eq!(gain_text(halved), "+100.0%");
        // HFF doubled: a regression, and it is signed.
        let doubled = Gain::Percent(100.0 * (1e-3 / 2e-3 - 1.0));
        assert_eq!(gain_text(doubled), "-50.0%");
        assert_eq!(gain_text(Gain::New), "NEW");
        assert_eq!(gain_text(Gain::Extinct), "EXTINCT");
        assert_eq!(gain_text(Gain::Merged), "MERGED");
        assert_eq!(gain_text(Gain::Unscored), "—");
        // No number hides inside the words.
        for g in [Gain::New, Gain::Extinct, Gain::Merged, Gain::Unscored] {
            assert!(!gain_text(g).contains('%'), "{g:?} printed a percentage");
        }
    }

    /// SORTING BY GAIN brings an improving young cohort above a converged elder
    /// with a far better absolute HFF — the acceptance check the brief names.
    #[test]
    fn sorting_by_gain_lifts_an_improving_cohort_above_a_better_one() {
        let mut state = replayed();
        state.sort = Sort::Best;
        let by_best = state.rows();
        state.sort = Sort::Gain;
        let by_gain = state.rows();
        assert_eq!(by_best.len(), by_gain.len(), "the sort changed the rows");
        // The gain order is a real reordering, and it is by gain: every row with
        // a percentage sits above every row without one, and the percentages
        // descend.
        let percents: Vec<f64> = by_gain.iter().filter_map(|r| match r.gain {
            Gain::Percent(v) => Some(v),
            _ => None,
        }).collect();
        assert!(percents.windows(2).all(|w| w[0] >= w[1]), "gains are not descending: {percents:?}");
        let first_wordy = by_gain.iter().position(|r| !matches!(r.gain, Gain::Percent(_)));
        if let Some(at) = first_wordy {
            assert!(by_gain[at..].iter().all(|r| !matches!(r.gain, Gain::Percent(_))), "a gain sank below a NEW");
        }
        // And by best, HFF ascends — lower is better, so the leader is first.
        let bests: Vec<f64> = by_best.iter().map(|r| r.best_hff.unwrap_or(f64::INFINITY)).collect();
        assert!(bests.windows(2).all(|w| w[0] <= w[1]), "best order is not lower-first: {bests:?}");
    }

    /// LOWER HFF IS A HIGHER BAR, and the scale is log — a run from 1e-1 to 1e-5
    /// is drawn as a climb, not as one step.
    #[test]
    fn the_sparkline_is_log_space_and_lower_hff_is_taller() {
        let falling = [Some(1e-1), Some(1e-2), Some(1e-3), Some(1e-4), Some(1e-5)];
        let bars = spark_of(&falling);
        assert_eq!(bars.len(), 5);
        assert!(bars.windows(2).all(|w| w[0] < w[1]), "a falling HFF must rise: {bars:?}");
        assert_eq!((bars[0], bars[4]), (1, 8), "the window's ends are the scale's ends: {bars:?}");
        // Evenly spaced in log space means evenly spaced bars — the linear test.
        assert_eq!(bars, vec![1, 3, 5, 6, 8], "{bars:?}");
        // A gap is a zero bar, not a carried-forward value.
        let gapped = spark_of(&[Some(1e-2), None, Some(1e-4)]);
        assert_eq!(gapped[1], 0);
        // One distinct value is flat at mid-height, not a full bar.
        assert_eq!(spark_of(&[Some(1e-3), Some(1e-3)]), vec![4, 4]);
        assert!(spark_of(&[None, None]).is_empty(), "nothing to draw is nothing drawn");
    }

    /// A FINISHED run and a STALE one are different states. A replay is never
    /// stale: a recording is as fresh as it will ever be.
    #[test]
    fn finished_stale_and_live_are_three_different_things() {
        let finished = replayed();
        assert_eq!(finished.liveness(), Liveness::Finished);
        // The same stream WITHOUT its run_end, followed live and just arrived:
        // live, not finished.
        let (records, _) = parse_stream(FIXTURE);
        let mut live = WatchState::new(true);
        for r in records.into_iter().filter(|r| !matches!(r, Record::RunEnd(_))) {
            live.apply(Parsed::Ok(Box::new(r)));
        }
        assert_eq!(live.liveness(), Liveness::Live);
        // Wind the clock back past the threshold: stale, and the frame is STILL
        // there — a stale viewer shows its last good frame, marked.
        live.last_record = Some(std::time::Instant::now() - STALE_AFTER - std::time::Duration::from_secs(1));
        assert_eq!(live.liveness(), Liveness::Stale);
        assert!(live.snapshot.is_some(), "a stale viewer keeps its last good frame");
        // A replay of the same records is never stale.
        let mut replay = WatchState::new(false);
        replay.last_record = Some(std::time::Instant::now() - STALE_AFTER * 10);
        replay.snapshot = live.snapshot.clone();
        assert_eq!(replay.liveness(), Liveness::Live);
    }

    /// THE SEARCH PULSE ships as three labelled meters and no composite, and a
    /// window with no pump in it shows a dash WITH ITS REASON rather than a zero.
    #[test]
    fn the_pulse_is_three_meters_and_a_quiet_window_is_a_dash() {
        let state = replayed();
        let pulse = state.pulse(0);
        assert_eq!(pulse.len(), 3);
        assert_eq!(pulse.iter().map(|(l, _, _)| *l).collect::<Vec<_>>(), ["improvement", "fresh lines", "scored rows"]);
        // Survival is not instrumented and says so; it is never a zero.
        assert_eq!(pulse[1].1, None);
        assert!(!pulse[1].2.is_empty(), "a dash without a reason is not honest");
        // Coverage is a real fraction of the island's rows.
        let coverage = pulse[2].1.expect("the island's scored fraction");
        assert!((0.0..=1.0).contains(&coverage), "{coverage}");
    }

    /// A second `run_start` in one file is a NEW run: the history is dropped
    /// rather than stitched onto the old one's sparklines.
    #[test]
    fn a_second_run_start_begins_a_new_run() {
        let mut state = replayed();
        assert!(!state.history.is_empty());
        let (records, _) = parse_stream(FIXTURE);
        let start = records.iter().find(|r| matches!(r, Record::RunStart(_))).expect("a run_start").clone();
        state.apply_record(start);
        assert!(state.history.is_empty(), "the old run's history was carried over");
        assert!(state.snapshot.is_none(), "the old run's frame was carried over");
        assert!(state.end.is_none(), "the old run's ending was carried over");
        assert_eq!(state.liveness(), Liveness::Live, "a new run is not finished");
    }

    /// The selection is not moved when the cohort it names leaves the table: the
    /// row is gone, the selection is not.
    #[test]
    fn a_selection_that_leaves_the_table_is_not_reassigned() {
        let mut state = replayed();
        state.selected = Some(999_999);
        assert!(state.selected_row().is_none(), "there is no such cohort");
        assert_eq!(state.selected, Some(999_999), "the viewer reassigned the selection");
        // And the filter excluding it does not move it either.
        state.selected = state.rows().first().map(|r| r.id);
        let kept = state.selected;
        state.filter = "999999".to_string();
        assert!(state.rows().is_empty());
        assert_eq!(state.selected, kept);
    }

    /// AN EXTINCT ROW LINGERS A FEW BEATS AND THEN GOES. Watching a cohort die
    /// is the point of the table; burying the living under every cohort the fit
    /// has ever minted is not.
    #[test]
    fn an_extinct_row_lingers_and_then_stops_crowding_the_table() {
        let mut state = replayed();
        let alive: Vec<u32> = state.snapshot.as_ref().expect("a frame").global_cohorts.iter().map(|c| c.id).collect();
        let dead = state.rows().iter().filter(|r| r.extinct).count();
        assert!(dead <= state.history.len(), "sanity");
        // Feed snapshots holding ONLY the first cohort: everything else dies.
        let mut s = state.snapshot.as_ref().expect("a frame").clone();
        s.global_cohorts.retain(|c| c.id == alive[0]);
        for _ in 0..(EXTINCT_LINGER + 1) {
            s.header.generation += 10;
            s.header.seq += 1;
            state.apply_record(Record::Snapshot(s.clone()));
        }
        let rows = state.rows();
        assert!(rows.iter().all(|r| !r.extinct), "an extinct row outstayed EXTINCT_LINGER: {rows:?}");
        // And the HISTORY map is bounded too, past HISTORY_LINGER.
        for _ in 0..(HISTORY_LINGER + 2) {
            s.header.generation += 10;
            s.header.seq += 1;
            state.apply_record(Record::Snapshot(s.clone()));
        }
        assert_eq!(state.history.len(), 1, "the history map grew without bound: {} entries", state.history.len());
    }

    /// FOCUSING AN ISLAND shows that island's own cohort split, which is the
    /// whole point of the engine filling the brief's `null` island values. `g`
    /// puts the table back to the global totals.
    #[test]
    fn focusing_an_island_shows_its_own_split_and_g_restores_the_totals() {
        let mut state = replayed();
        let global: u32 = state.rows().iter().map(|r| r.rows).sum();
        let population = state.start.as_ref().expect("a run_start").population;
        // The global table's rows sum to the population (the extinct rows carry
        // zero, so they do not disturb it).
        assert_eq!(global, population);
        assert!(state.islands().len() >= 2);
        for island in 0..state.islands().len() {
            state.island = island;
            state.island_focus = true;
            let rows = state.rows();
            let sum: u32 = rows.iter().map(|r| r.rows).sum();
            let expect = state.islands()[island].rows;
            assert_eq!(sum, expect, "island {island}'s split does not sum to its rows");
            // A focused table never invents an EXTINCT row: a cohort leaving one
            // island has moved, not died.
            assert!(rows.iter().all(|r| !r.extinct), "a focused table marked a cohort extinct");
        }
        // `g`'s effect: back to the global totals.
        state.island_focus = false;
        assert_eq!(state.rows().iter().map(|r| r.rows).sum::<u32>(), population);
    }

    /// THE GAIN WINDOW is measured, not assumed. A snapshot is written at the
    /// progress beat but no more often than once a second, so the window is
    /// whatever the fit's speed made it, and the screen states that number.
    #[test]
    fn the_gain_window_is_the_measured_gap_between_snapshots() {
        let state = replayed();
        let window = state.gain_window().expect("two snapshots");
        let now = state.snapshot.as_ref().expect("a frame").header.generation;
        let before = state.previous.as_ref().expect("the previous frame").header.generation;
        assert_eq!(window, now - before);
        assert!(window > 0, "two snapshots one generation apart is not a window");
        // A viewer with only one snapshot has no window and says so.
        let mut one = WatchState::new(false);
        one.apply_record(Record::Snapshot(state.snapshot.clone().expect("a frame")));
        assert_eq!(one.gain_window(), None);
    }

    /// THE MOCKUP'S RUN, as a telemetry stream: the gen 10–170 checkpoints the
    /// design brief's interactive mockup replays, with the island values left
    /// exactly as the brief has them — `best_hff: null`, `cohorts: []`, because
    /// the supplied log could not recover which island those rows occupied.
    const MOCKUP: &str = include_str!("../../tests/fixtures/mockup_bacres1_seed7015.jsonl");

    fn mockup_upto(generation: u32) -> WatchState {
        let (records, bad) = parse_stream(MOCKUP);
        assert_eq!(bad, 0);
        let mut state = WatchState::new(false);
        for r in records.into_iter().take_while(|r| r.header().generation <= generation) {
            state.apply_record(r);
        }
        state
    }

    /// THE BRIEF'S FIRST ACCEPTANCE CHECK, on the brief's own numbers: c0 holds
    /// 200,000 rows through gen 90; at gen 100 it is 120,000 with c100 at
    /// 80,000; and c100 improves to 5.356e-3 by gen 170.
    #[test]
    fn the_mockups_run_replays_the_cohort_progression_the_brief_describes() {
        let at_90 = mockup_upto(90);
        let rows = at_90.rows();
        assert_eq!(rows.len(), 1, "c0 is the only cohort through gen 90");
        assert_eq!((rows[0].id, rows[0].rows), (0, 200_000));
        let at_100 = mockup_upto(100);
        let rows = at_100.rows();
        assert_eq!(rows.len(), 2, "c100 appears at gen 100");
        let by_id = |rs: &[CohortView], id: u32| rs.iter().find(|r| r.id == id).expect("the cohort").clone();
        assert_eq!(by_id(&rows, 0).rows, 120_000);
        assert_eq!(by_id(&rows, 100).rows, 80_000);
        // A cohort that has just appeared is the WORD NEW, not a percentage.
        assert_eq!(by_id(&rows, 100).gain, Gain::New);
        let at_170 = mockup_upto(170);
        let rows = at_170.rows();
        assert_eq!(by_id(&rows, 100).best_hff, Some(0.005356), "c100 improved to 5.356e-3 by gen 170");
        assert_eq!(by_id(&rows, 0).best_hff, Some(0.0004079));
        assert_eq!(at_170.snapshot.as_ref().expect("a frame").global.best_hff, Some(0.0004078684));
    }

    /// AND ON THAT RUN, GAIN SORT LIFTS c100 ABOVE c0 — the brief's second
    /// acceptance check — although c100's absolute HFF is an order of magnitude
    /// worse. The selection stays on c100 across the sort and the next snapshot.
    #[test]
    fn on_the_mockups_run_gain_lifts_c100_above_the_better_c0() {
        let mut state = mockup_upto(110);
        state.sort = Sort::Best;
        let by_best = state.rows();
        assert_eq!(by_best[0].id, 0, "by best HFF, c0 leads");
        state.sort = Sort::Gain;
        let by_gain = state.rows();
        assert_eq!(by_gain[0].id, 100, "by gain, the improving c100 leads");
        // It really is the worse one that was lifted.
        let c100 = by_gain[0].best_hff.expect("c100's best");
        let c0 = by_gain.iter().find(|r| r.id == 0).expect("c0").best_hff.expect("c0's best");
        assert!(c100 > c0, "c100 {c100:e} should be the WORSE absolute HFF than c0 {c0:e}");
        // The selection follows c100 by ID through the sort AND the next
        // snapshot, which is the rest of that acceptance check.
        state.selected = Some(100);
        for sort in [Sort::Best, Sort::Rows, Sort::Gain] {
            state.sort = sort;
            let _ = state.rows();
            assert_eq!(state.selected, Some(100));
        }
        let (records, _) = parse_stream(MOCKUP);
        for r in records.into_iter().filter(|r| r.header().generation > 110) {
            state.apply_record(r);
            assert_eq!(state.selected, Some(100), "a snapshot moved the selection off c100");
        }
        assert_eq!(state.rows().iter().find(|r| r.id == 100).expect("c100").best_hff, Some(0.005356));
    }

    /// NO PANEL PRETENDS TO KNOW WHAT THE STREAM DID NOT SAY. On the mockup's
    /// run the islands carry no best and no cohort split, and the pulse must
    /// report that as a dash WITH ITS REASON — not as "0% improvement" and
    /// certainly not as "100% of rows scored", which is a fabricated island
    /// metric on a stream that emitted nothing.
    #[test]
    fn a_stream_with_no_island_metrics_fabricates_none() {
        let state = mockup_upto(170);
        let islands = state.islands();
        assert_eq!(islands.len(), 2);
        assert!(islands.iter().all(|i| i.best_hff.is_none() && i.cohorts.is_empty() && i.nan_rows.is_none()));
        for island in 0..islands.len() {
            let pulse = state.pulse(island);
            for (label, value, reason) in pulse {
                assert_eq!(value, None, "{label} was invented from a stream that did not emit it");
                assert!(!reason.is_empty(), "{label} dashed without saying why");
            }
        }
        // And the engine's own stream, which DOES emit them, still shows them —
        // the silence is the producer's, not a blanket refusal to draw.
        let ours = replayed();
        let coverage = ours.pulse(0)[2].1;
        assert!(coverage.is_some(), "a stream that emits unscored counts must show coverage");
    }

    /// Every undefined number is a dash in ONE place, and a dash is never a zero.
    #[test]
    fn an_undefined_metric_is_a_dash_and_never_a_zero() {
        assert_eq!(or_dash(None, 3), "—");
        assert_eq!(or_dash(Some(0.0), 3), "0.000e0");
        assert_ne!(or_dash(None, 3), or_dash(Some(0.0), 3));
    }
}
