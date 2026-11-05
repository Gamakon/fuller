//! The SIGNATURE FAMILY of the Futhark kingdom: a Futhark kingdom whose interface is a program's signature, in the
//! direction of "any Futhark program: inputs and outputs are any typed values". A problem's kingdom is a VIEW over
//! the master Futhark table (`kingdoms/futhark/master/`, [`crate::futhark_view`]): its specification
//! (`kingdoms/futhark/sig/<problem>/view.tsv`) is written by `kingdoms/futhark/sniff.py` from the problem's Futhark
//! source, and the view filters, binds and closes the master's rows (`docs/PLAN_futhark_kingdom.md`, step 3).
//!
//! A specification gives its inputs and output as typed arrays over named dimensions — Rodinia's pathfinder is
//! `walls: [rows][cols]i32 -> [cols]i32` — and the view derives the rest, every rule of a legal program a type:
//! a class (`a.rows.cols`, `a.cols`, `ix.cols`, `i32`) in a context (the binders around the node: `tabulate`, a
//! `loop` over a dimension carrying a state, the same loop from 1). An index is in range by its type (its terminals
//! are `0`, `D - 1` and the counters of binders over `D`; its functions clamp), a variable is read only in a context
//! that binds it, and a binder, a reduction or a `replicate` exists only above the family's depth, so a node runs at
//! most `max(D) ** depth` times. Every gene expresses a program Futhark accepts; the tests here check that Futhark
//! agrees and that the compiled program computes what the host interpreter says.
//!
//! What this module holds: the loader and the declared signature, Futhark's binary value format for the cases (typed:
//! `i32`, or `f32` as its raw bits, with the comparison policy on [`Array`]), the program around an expressed tree
//! (the entry point binding the interface), the host interpreter (the meaning of a gene, a test oracle, exact
//! arithmetic as Futhark's: `i32` wraps, `f32` is IEEE with `fmin`/`fmax` and `i64.f32` as measured through
//! `futhark c`), the exact work of a tree on given sizes, a
//! builder that writes a program given as an s-expression of the family's rows as a tree, the problem registry (the
//! instance, its reference written as a gene, the published program as the speed yardstick, and its stored cases),
//! and the instance's e-graph (one sort a type, constants by their pool index).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use egglog::extract::hff_extract;
use egglog::prelude::exprs;
use egglog::{EGraph, Term, TermDag, TermId};

use crate::futhark_interpreter::{row2_fn, row_fn, Row2Fn, RowFn};
use crate::futhark_table::{FutharkKind, FutharkKingdom, GeneRng};
use crate::futhark_view::{var_name, Class, Ctx, Shape, View as TableView, ViewSpec};
use crate::geneframe::Ty;
use crate::kingdom_contract::{KingdomContract, SymbolId};
use crate::saturated_fold::EgraphKingdom;
use crate::typed_gep::{Domain, GeneLayout, Tree};

/// One instance of the family: a problem's VIEW over the master Futhark table (`kingdoms/futhark/master/`), given by
/// its specification (`kingdoms/futhark/sig/<problem>/view.tsv`, written by `kingdoms/futhark/sniff.py` from the
/// problem's Futhark source). Its tables, signature and e-graph are computed at runtime from the master and the
/// specification ([`crate::futhark_view`]), once a process. The tables are not committed: `View::write` writes them
/// for the generator's witnesses, and each view's text is pinned by its fingerprint.
#[derive(Debug, Clone, Copy)]
pub struct Instance {
    pub name: &'static str,
    pub spec: &'static str,
}

impl Instance {
    /// The instance's view (computed once a process, shared by every caller).
    pub fn view(&self) -> Result<&'static TableView, String> {
        crate::futhark_view::cached(self.spec).map_err(|e| format!("{}: the view does not load: {e}", self.name))
    }
}

/// Rodinia pathfinder: `walls: [rows][cols]i32 -> [cols]i32`.
pub const PATHFINDER: Instance = Instance { name: "pathfinder", spec: include_str!("../kingdoms/futhark/sig/pathfinder/view.tsv") };

/// The study's micro benchmark reduce (`micro/reduce.fut`, `sum_i32`): `xs: [n]i32 -> i32`.
pub const REDUCE: Instance = Instance { name: "reduce", spec: include_str!("../kingdoms/futhark/sig/reduce/view.tsv") };

/// The study's micro benchmark scan (`micro/scan.fut`, `sum_i32`): `xs: [n]i32 -> [n]i32`, inclusive prefix sums.
pub const SCAN: Instance = Instance { name: "scan", spec: include_str!("../kingdoms/futhark/sig/scan/view.tsv") };

/// The study's histogram (`micro/reduce_by_index.fut`, `sum_i32`): `(k: i32) (is: [n]i32) (vs: [n]i32) -> [k]i32`,
/// the sum of the values in each bin `is % k`.
pub const HISTOGRAM: Instance = Instance { name: "histogram", spec: include_str!("../kingdoms/futhark/sig/histogram/view.tsv") };

/// The ruleset name the generated rules are in.
pub const RULESET: &str = "futhark_sig";

/// A problem: a signature instance, its published program written as a gene of the kingdom (an s-expression of the
/// family's rows, for [`build`]), the published program itself as the speed yardstick (a Futhark source whose `main`
/// has the declared signature), and the directory of its stored cases.
#[derive(Debug, Clone, Copy)]
pub struct Problem {
    pub name: &'static str,
    pub instance: Instance,
    /// THE SEED: the prototype's program written as a gene (the reference) with its planted slowdown, or the explicit
    /// state that the prototype is not yet translated ([`Seed`]). Read through [`Problem::reference`] and
    /// [`Problem::planted`], which name the state plainly instead of a program.
    pub seed: Seed,
    pub yardstick: &'static str,
    /// THE PROBLEM'S DEFAULT FITNESS SCHEME (`fuller::fitness::SCHEMES`, by name): the newest scheme of its kind. The
    /// i32 problems and A7 are scored against their stored answers (`futhark-slope4-qqs`); k-means is scored on its
    /// clustering objective run to UltraDim's stopping rule (`futhark-kmeans-better`, `docs/KMEANS_FITNESS.md`),
    /// because a k-means step has no exact answer. `--fitness` may name another scheme of the same kind only
    /// ([`crate::fitness::scheme_for_problem`]).
    pub fitness: &'static str,
    /// Relative to the crate's root.
    pub corpus: &'static str,
    /// THE TIMING INPUT's size (`n`; 0: time on the largest training case). A micro benchmark is timed on its own
    /// large input ([`timing_input`]), used only for timing and never as a quality case: measured 2026-10-27, reduce
    /// timed on its 1,000-element training case gave A/A ratios in three clusters (0.30, 1.00, 3.3), the GPU's clock
    /// states on a 10-30 us kernel, so selection in stage 2 chose on noise. The micro benchmarks use 15 x 2^20: until
    /// series patch 0011 (2026-11-02) a map over more than 65,535 x 256 elements could not be dispatched (wgpu allows
    /// 65,535 workgroups a dimension; measured 2026-10-28 at n = 2^24, "[65536, 1, 1] must be less or equal to 65535");
    /// the compiler now folds such a grid into two dimensions, and the size is kept so the timings stay comparable.
    pub timing_n: usize,
    /// THE PROGRAM SIZE `N` of the problem's gene layout when `--program-size` is not given: an experiment parameter
    /// above the reference and the planted slowdown ([`DEFAULT_PROGRAM_SIZE`] for the i32 benchmarks; k-means's
    /// reference has 72 functions, the membership written twice, so its layout is [`KMEANS_PROGRAM_SIZE`]).
    pub program_size: usize,
}

/// The state of a problem's reference slot.
#[derive(Debug, Clone, Copy)]
pub enum Seed {
    /// The prototype's program as a gene (an s-expression of the kingdom's rows, for [`build`]), and THE PLANTED
    /// SLOWDOWN (a control of regime P): an equal program that does the work twice, which the search must recover
    /// from. For pathfinder: the elementwise least of the reference and its re-associated twin (the `min`s grouped
    /// the other way), which Futhark cannot merge and the e-graph can (associativity puts the twins in one class, and
    /// `min a a = a` removes the `map2`).
    Program { reference: &'static str, planted: &'static str },
    /// The prototype's program and its planted slowdown as genes whose text is assembled once from parts (k-means's
    /// seed, `docs/kmeans_seed.md`): the same as [`Seed::Program`] with the texts made at first use.
    Translated { reference: fn() -> &'static str, planted: fn() -> &'static str },
    /// THE SEED IS NOT YET TRANSLATED: the prototype is named, the translation is planned, and until it lands the
    /// problem has no reference program. The runner stops with this text; no program stands in (k-means was in this
    /// state from the review of 2026-11-03 until its translation landed, `docs/PLAN_futhark_kingdom.md` 4c).
    NotYetTranslated { prototype: &'static str, plan: &'static str, pool: &'static str },
}

impl Problem {
    /// The reference program's text, or the plain statement that the seed is not yet translated.
    pub fn reference(&self) -> Result<&'static str, String> {
        match self.seed {
            Seed::Program { reference, .. } => Ok(reference),
            Seed::Translated { reference, .. } => Ok(reference()),
            Seed::NotYetTranslated { .. } => Err(self.seed_state()),
        }
    }

    /// The planted slowdown's text, or the plain statement that the seed is not yet translated.
    pub fn planted(&self) -> Result<&'static str, String> {
        match self.seed {
            Seed::Program { planted, .. } => Ok(planted),
            Seed::Translated { planted, .. } => Ok(planted()),
            Seed::NotYetTranslated { .. } => Err(self.seed_state()),
        }
    }

    /// One sentence on the seed's state, for the runner's header, the checklist and an error.
    pub fn seed_state(&self) -> String {
        match self.seed {
            Seed::Program { .. } | Seed::Translated { .. } => format!("{}: the seed is the prototype's program as a gene", self.name),
            Seed::NotYetTranslated { prototype, plan, pool } => format!("{}: the seed is not yet translated -- the prototype is {prototype}; the translation is {plan}; until it lands the problem has no reference program and cannot be run (the textbook program is a pool alternative at {pool}, never the seed)", self.name),
        }
    }

    /// Whether the stored cases carry an answer (`<name>.out`) the program is scored against. A problem scored on its
    /// own objective (the k-means schemes, `futhark-kmeans-better`) stores inputs only: there is no exact
    /// answer to store.
    pub fn cases_have_answers(&self) -> bool {
        !crate::fitness::scheme(self.fitness).is_ok_and(|s| s.is_converged())
    }
}

/// Pathfinder's published algorithm (`rodinia/pathfinder/pathfinder.fut`) as a gene: the state starts as row 0 of
/// the walls; for each later row `i` (1 .. rows-1), each column `j` becomes the wall plus the least of the state's
/// `j`, `j - 1` and `j + 1`, the published order of the two `min`s. The published kernel skips a neighbour that is off
/// the edge; here the index is clamped, so the off-edge neighbour is `j` itself, which the least already holds.
pub const PATHFINDER_REFERENCE: &str = "(loop1.rows (row.rows.cols walls zero)
  (tab (add (get.cols (row.rows.cols walls i_rows) i_cols)
            (min (min (get.cols s_a_cols i_cols) (get.cols s_a_cols (prev i_cols))) (get.cols s_a_cols (next i_cols))))))";

pub const PATHFINDER_PLANTED: &str = "(map2.min
  (loop1.rows (row.rows.cols walls zero)
    (tab (add (get.cols (row.rows.cols walls i_rows) i_cols)
              (min (min (get.cols s_a_cols i_cols) (get.cols s_a_cols (prev i_cols))) (get.cols s_a_cols (next i_cols))))))
  (loop1.rows (row.rows.cols walls zero)
    (tab (add (get.cols (row.rows.cols walls i_rows) i_cols)
              (min (get.cols s_a_cols i_cols) (min (get.cols s_a_cols (prev i_cols)) (get.cols s_a_cols (next i_cols))))))))";

pub const PATHFINDER_PROBLEM: Problem = Problem {
    name: "pathfinder",
    instance: PATHFINDER,
    seed: Seed::Program { reference: PATHFINDER_REFERENCE, planted: PATHFINDER_PLANTED },
    yardstick: include_str!("../tests/corpus/futhark_problems/pathfinder/yardstick.fut"),
    fitness: "futhark-slope4-qqs",
    corpus: "tests/corpus/futhark_problems/pathfinder",
    timing_n: PATHFINDER_TIMING_ROWS * 100_000,
    program_size: DEFAULT_PROGRAM_SIZE,
};

/// Reduce's published program, `i32.sum` (`reduce (+) 0`), as a gene.
pub const REDUCE_REFERENCE: &str = "(sum.n xs)";
/// Reduce's planted slowdown: the sum of the elementwise least of the input and itself (a `map2` Futhark keeps; the
/// e-graph's `map2 min a a = a` removes it).
pub const REDUCE_PLANTED: &str = "(sum.n (map2.min xs xs))";

pub const REDUCE_PROBLEM: Problem = Problem {
    name: "reduce",
    instance: REDUCE,
    seed: Seed::Program { reference: REDUCE_REFERENCE, planted: REDUCE_PLANTED },
    yardstick: include_str!("../tests/corpus/futhark_problems/reduce/yardstick.fut"),
    fitness: "futhark-slope4-qqs",
    corpus: "tests/corpus/futhark_problems/reduce",
    timing_n: 15 << 20,
    program_size: DEFAULT_PROGRAM_SIZE,
};

/// Scan's published program, `scan (+) 0`, as a gene.
pub const SCAN_REFERENCE: &str = "(scan.add xs)";
/// Scan's planted slowdown: the scan of the elementwise greatest of the input and itself.
pub const SCAN_PLANTED: &str = "(scan.add (map2.max xs xs))";

pub const SCAN_PROBLEM: Problem = Problem {
    name: "scan",
    instance: SCAN,
    seed: Seed::Program { reference: SCAN_REFERENCE, planted: SCAN_PLANTED },
    yardstick: include_str!("../tests/corpus/futhark_problems/scan/yardstick.fut"),
    fitness: "futhark-slope4-qqs",
    corpus: "tests/corpus/futhark_problems/scan",
    timing_n: 15 << 20,
    program_size: DEFAULT_PROGRAM_SIZE,
};

/// The histogram's published program, `hist (+) 0 k (map (% k) is) vs`, as a gene: the bins are the indices modulo
/// `k`, tabulated over `n`, and the values are summed into them.
pub const HISTOGRAM_REFERENCE: &str = "(hist.add.n (tab (mod.k (get.n is i_n))) vs)";
/// The histogram's planted slowdown: the values replaced by the elementwise least of the values and themselves.
pub const HISTOGRAM_PLANTED: &str = "(hist.add.n (tab (mod.k (get.n is i_n))) (map2.min vs vs))";

pub const HISTOGRAM_PROBLEM: Problem = Problem {
    name: "histogram",
    instance: HISTOGRAM,
    seed: Seed::Program { reference: HISTOGRAM_REFERENCE, planted: HISTOGRAM_PLANTED },
    yardstick: include_str!("../tests/corpus/futhark_problems/histogram/yardstick.fut"),
    fitness: "futhark-slope4-qqs",
    corpus: "tests/corpus/futhark_problems/histogram",
    timing_n: 15 << 20,
    program_size: DEFAULT_PROGRAM_SIZE,
};

/// THE K-MEANS STEP: `(points: [n][d]f32) (centres: [k][d]f32) -> [k][d]f32` (`docs/PLAN_futhark_kingdom.md`, step
/// 4; the seed is UltraDim's own kernels, 4c (d), `docs/kmeans_seed.md`). The view is at master version 5, depth 3
/// (a point's nearest centre is a loop over `k` inside the tabulate over `n`, with the distance's loop over `d`
/// inside; a cell's compensated sum is a loop over `n` inside a tabulate over `d` inside the tabulate of rows over
/// `k`), with ONE KERNEL BOUNDARY, `lets a.n`: the assignments the first kernel writes and the second reads. Its one
/// element is `f32`: the assignments and the counts are integers held exactly in f32 (below 2^24). The faithful
/// two-element view (`i32` assignments, `lets a.i32.n`) is measured in `tests_kmeans_seed`: 131,825 types without the
/// kernel boundary and 237,325 with it, over the 65,536 type codes, so it does not load (the measurement and the
/// decision are in `docs/kmeans_seed.md`).
pub const KMEANS: Instance = Instance { name: "kmeans", spec: include_str!("../kingdoms/futhark/sig/kmeans/view.tsv") };

/// THE SEED: UltraDim's three k-means kernels (`lib/projection-voting-kmeans/src/gpu_kmeans.rs`: `compute_distances`,
/// `assign_points`, `update_centroids`) translated construct for construct, `docs/kmeans_seed.md` line by line. The
/// kernel boundary `let` names the assignments (`s_a_n`, the `[n]` buffer `assign_points` writes): a tabulate over the
/// points of the argmin over the centres as the pair (best distance, best index) from the first centre's distance, a
/// later centre winning only when its distance is strictly smaller (`d - best < 0`), each distance `sqrt` of the loop
/// over `d` accumulating `(v - c) * (v - c)` from 0 (the distance kernel, computed where it is read: see the document).
/// The body is `update_centroids`: for each (cluster, coordinate) cell a loop over the points carrying the TRIPLE
/// (sum, compensation, count), stepping only a point whose assignment is the cluster (`|a - j| - 1 < 0`) by
/// `y = x - c; t = s + y; c = (t - s) - y; s = t; count + 1`, then the sum over the count when the count is positive
/// (`-count < 0`), else the previous centre. The loop is written three times (the kingdom has no `let` below the top;
/// the values are one, and the compiler's CSE computes it once); the distance of the centre under test is written
/// twice (the compare and the replacement). `KMEANS_SEED_ASSIGN`, `KMEANS_SEED_UPDATE` and `KMEANS_SEED_CELL_LOOP` are
/// its parts; [`kmeans_seed`] assembles the program text.
pub const KMEANS_SEED_ASSIGN: &str = "(tab (snd.f32.f32 (loop1.k (pair (sqrt (loop.d 0 (add s_f32 (mul (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres zero) i_d)) (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres zero) i_d)))))) 0)
  (psel (sub DIST s_f32) (pair DIST (ival.k i_k)) (pair s_f32 s2_f32)))))";
/// The distance of the centre under test (`i_k`) inside the argmin loop, where the loop over `d` binds the third f32.
pub const KMEANS_SEED_DIST: &str = "(sqrt (loop.d 0 (add s3_f32 (mul (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres i_k) i_d)) (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres i_k) i_d))))))";
pub const KMEANS_SEED_UPDATE: &str = "(tabrows (tab (sel (neg (thd.f32.f32.f32 LOOP)) (div (fst.f32.f32.f32 LOOP) (thd.f32.f32.f32 LOOP)) (get.d (row.k.d centres i_k) i_d))))";
pub const KMEANS_SEED_CELL_LOOP: &str = "(loop.n zero (psel (sub (abs (sub (get.n s_a_n i_n) (ival.k i_k))) 1)
    (triple (add s_f32 (sub (get.d (row.n.d points i_n) i_d) s2_f32))
            (sub (sub (add s_f32 (sub (get.d (row.n.d points i_n) i_d) s2_f32)) s_f32) (sub (get.d (row.n.d points i_n) i_d) s2_f32))
            (add s3_f32 1))
    (triple s_f32 s2_f32 s3_f32)))";

/// The seed program as a gene: `(let <assignments> <update>)`.
pub fn kmeans_seed() -> String {
    format!("(let {} {})", KMEANS_SEED_ASSIGN.replace("DIST", KMEANS_SEED_DIST), KMEANS_SEED_UPDATE.replace("LOOP", KMEANS_SEED_CELL_LOOP))
}

/// The seed's planted slowdown (a control of regime P): the assignments computed twice and the elementwise least of
/// the two taken (`map2 f32.min a a`, a traversal Futhark keeps and the e-graph's `map2.min a a = a` removes).
pub fn kmeans_planted() -> String {
    let assign = KMEANS_SEED_ASSIGN.replace("DIST", KMEANS_SEED_DIST);
    format!("(let (map2.min {assign} {assign}) {})", KMEANS_SEED_UPDATE.replace("LOOP", KMEANS_SEED_CELL_LOOP))
}

/// The seed's text, made once (the problem registry holds a `&'static str`).
pub fn kmeans_seed_static() -> &'static str {
    static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    TEXT.get_or_init(kmeans_seed)
}

/// The planted slowdown's text, made once.
pub fn kmeans_planted_static() -> &'static str {
    static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    TEXT.get_or_init(kmeans_planted)
}

/// THE TEXTBOOK K-MEANS STEP as a gene: the published `find_nearest_point`, then `centroids_of`. A POOL ALTERNATIVE
/// (`kingdoms/futhark/sig/kmeans/pool/`), NOT the seed: it stood in the reference slot until 2026-11-03, when the
/// review (`docs/PLAN_futhark_kingdom.md` 4c) found the seed swapped -- the seed is UltraDim's own kernels
/// ([`kmeans_seed`], `docs/kmeans_seed.md`). Each point's membership is a loop
/// over the centres carrying the nearest's index (as an `f32`, the view's one element; `clamp` makes it the centre's
/// row index) from 0, kept when its distance is below the centre's and replaced by the centre otherwise (the
/// published fold's `if p1.1 < p2.1 then p1 else p2`: on a tie, and on a NaN distance, the later centre); a distance
/// is the published `euclid_dist_2`, the left-fold sum from 0 of the squared differences in order. The cluster sums
/// are the elementwise add histogram of the points' rows at their memberships (`hist (map2 (+))`), each row divided
/// by its cluster's count, a count of 0 taken as 1 (the published `if n == 0 then 1 else n`): the counts are the add
/// histogram of ones, at least 1. The membership is written twice (the kingdom has no `let`; the values are one).
pub const KMEANS_TEXTBOOK: &str = "(rowdiv (histrows.add.n (tab (loop.k 0 (sel (sub (sum.d (tab (mul (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres (clamp s_f32)) i_d)) (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres (clamp s_f32)) i_d))))) (sum.d (tab (mul (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres i_k) i_d)) (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres i_k) i_d)))))) s_f32 (ival.k i_k)))) points) (map2.max (replicate 1) (hist.add.n (tab (loop.k 0 (sel (sub (sum.d (tab (mul (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres (clamp s_f32)) i_d)) (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres (clamp s_f32)) i_d))))) (sum.d (tab (mul (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres i_k) i_d)) (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres i_k) i_d)))))) s_f32 (ival.k i_k)))) (replicate 1))))";

/// The textbook step's planted slowdown (a test of the e-graph on this view, as the seeded problems' planted
/// slowdowns are): the ones the counts are a histogram of, replaced by the elementwise least of the ones and themselves
/// (`map2 min a a = a`, which the e-graph removes). The planted form of the other problems, the work twice, would
/// write the membership a third time (108 functions, past [`KMEANS_PROGRAM_SIZE`]).
pub const KMEANS_TEXTBOOK_PLANTED: &str = "(rowdiv (histrows.add.n (tab (loop.k 0 (sel (sub (sum.d (tab (mul (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres (clamp s_f32)) i_d)) (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres (clamp s_f32)) i_d))))) (sum.d (tab (mul (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres i_k) i_d)) (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres i_k) i_d)))))) s_f32 (ival.k i_k)))) points) (map2.max (replicate 1) (hist.add.n (tab (loop.k 0 (sel (sub (sum.d (tab (mul (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres (clamp s_f32)) i_d)) (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres (clamp s_f32)) i_d))))) (sum.d (tab (mul (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres i_k) i_d)) (sub (get.d (row.n.d points i_n) i_d) (get.d (row.k.d centres i_k) i_d)))))) s_f32 (ival.k i_k)))) (map2.min (replicate 1) (replicate 1)))))";

/// The k-means problem. Its seed is a `Seed::Translated`: the program text is assembled from its parts once
/// ([`kmeans_seed_static`]), so the registry stays a constant.
pub const KMEANS_PROBLEM: Problem = Problem {
    name: "kmeans",
    instance: KMEANS,
    seed: Seed::Translated { reference: kmeans_seed_static, planted: kmeans_planted_static },
    yardstick: include_str!("../tests/corpus/futhark_problems/kmeans/yardstick.fut"),
    fitness: "futhark-kmeans-better",
    corpus: "tests/corpus/futhark_problems/kmeans",
    timing_n: KMEANS_TIMING_N,
    program_size: KMEANS_PROGRAM_SIZE,
};

/// K-means's program size: the seed is 131 functions (the cell's loop written three times, the centre's distance
/// twice: the kingdom has no `let` below the top) and its planted slowdown 183, so 192 holds both; the textbook pool
/// programs (72 and 74) fit too. The gene at this size is measured in `tests_kmeans_seed` (27.6 million words).
pub const KMEANS_PROGRAM_SIZE: usize = 192;

/// A7, THE MATRIX KINGDOM AS A PROBLEM OF THE ONE RUNNER (`docs/PLAN_futhark_kingdom.md`, step 4; the product
/// decision in `docs/PLAN_futhark_product.md`): the master's matrix family over the roles `n`, `m`, `p`, rooted at
/// `[n][p]` (`kingdoms/futhark/view.tsv`, the same view `FutharkKingdom::load` and the deprecated
/// `examples/futhark_a7_evolve.rs` load). Its signature is the interface's nine matrices, one a class, in the view's
/// class order (`m_n_n: [n][n]f32`, ... `m_p_p: [p][p]f32`), and its output `[n][p]f32`.
pub const A7: Instance = Instance { name: "a7", spec: include_str!("../kingdoms/futhark/view.tsv") };

/// A7's published program, the product of two matrices computed TWICE (once directly, once the transposed way) and the
/// two added, as a gene. A matrix row is named by its argument classes (`matmul@mat.n.mxmat.m.p`) or its input class
/// (`transpose@mat.n.m`, whose output is `mat.m.n`), so the s-expression names those rows in full ([`build`]).
pub const A7_REFERENCE: &str = "(map2_rows fn.arr2_f32_arr_f32_arr_f32.add
  (matmul@mat.n.mxmat.m.p mat.n.m.input mat.m.p.input)
  (transpose@mat.p.n (matmul@mat.p.mxmat.m.n (transpose@mat.m.p mat.m.p.input) (transpose@mat.n.m mat.n.m.input))))";

/// A7's winner (`docs/measurements/2026-10-27_futhark_shapes_A7.md`): the e-graph's transposed-product rule,
/// `(Y' . X')' = X . Y`, rewrites the second operand into the first, and the compiler's CSE computes the product once.
pub const A7_WINNER: &str = "(map2_rows fn.arr2_f32_arr_f32_arr_f32.add
  (matmul@mat.n.mxmat.m.p mat.n.m.input mat.m.p.input)
  (matmul@mat.n.mxmat.m.p mat.n.m.input mat.m.p.input))";

/// A7's planted slowdown: the reference under a `map` of the identity row function, a traversal Futhark keeps and the
/// e-graph's `map id = id` rule removes.
pub const A7_PLANTED: &str = "(map_rows fn.arr_f32_arr_f32.id
  (map2_rows fn.arr2_f32_arr_f32_arr_f32.add
    (matmul@mat.n.mxmat.m.p mat.n.m.input mat.m.p.input)
    (transpose@mat.p.n (matmul@mat.p.mxmat.m.n (transpose@mat.m.p mat.m.p.input) (transpose@mat.n.m mat.n.m.input)))))";

pub const A7_PROBLEM: Problem = Problem {
    name: "a7",
    instance: A7,
    seed: Seed::Program { reference: A7_REFERENCE, planted: A7_PLANTED },
    yardstick: include_str!("../tests/corpus/futhark_problems/a7/yardstick.fut"),
    fitness: "futhark-slope4-qqs",
    corpus: "tests/corpus/futhark_problems/a7",
    timing_n: A7_TIMING_N,
    program_size: DEFAULT_PROGRAM_SIZE,
};

// ---- The timing inputs' shapes: the problems' data (the timing inputs themselves are drawn by phylu's
// `futhark_eval::measure`, plan 5b) ----

/// The rows of pathfinder's timing input (the published medium data set's); its columns are `timing_n / rows`.
pub const PATHFINDER_TIMING_ROWS: usize = 100;

/// The side of A7's timing input, every role: n = m = p = 256, so the two products are 2 x 256^3 multiply-adds and the
/// traffic, not a GPU program's fixed overhead, is what the time measures (the product plan's 64^3 measurement had the
/// reference's OpenCL kernel too big for the device; the webgpu backend is what this runner times). The values are
/// uniform in [-8, 8) with the first of every matrix `-0.0`, as A7's correctness suite draws them
/// (`fuller::futhark::input_set`).
pub const A7_TIMING_N: usize = 256;

/// K-MEANS'S TIMING INPUT'S SHAPE: the whole dbpedia sample, n = 5,000 rows of d = 1,536, with k = 256 starting
/// centres (n / k about 20, as the fitness cases keep it; the advisor's review of 2026-11-03: UltraDim's k = 1,024
/// belongs with its n = 8,192, and at n = 5,000 it gives 5 points a cluster and a percentile that counts singletons).
pub const KMEANS_TIMING_N: usize = 5000;

pub const KMEANS_TIMING_D: usize = 1536;

pub const KMEANS_TIMING_K: usize = 256;

/// THE PROGRAM SIZE `N` of a run's gene layout when `--program-size` is not given: an experiment parameter, fixed and
/// well above every reference (pathfinder's is 13 functions, histogram's 4, reduce's and scan's 1), as SR and regex
/// fix theirs. Before 2026-11-02 the default was the reference's own function count, so reduce's and scan's layouts
/// were the root's head of one cell and nothing with two functions could be expressed
/// (`docs/reviews/2026-11-01_futhark_kingdom_design_fable.md`, F3). The run card records the size used.
pub const DEFAULT_PROGRAM_SIZE: usize = 32;

/// The number of functions (nodes with arguments) of a tree: the program size it needs at least.
pub fn function_count(k: &FutharkKingdom, t: &Tree) -> usize {
    t.level_order().iter().filter(|(n, _)| !k.rows[n.symbol as usize].inputs.is_empty()).count()
}

/// A problem by name.
pub fn problem(name: &str) -> Result<Problem, String> {
    match name {
        "pathfinder" => Ok(PATHFINDER_PROBLEM),
        "reduce" => Ok(REDUCE_PROBLEM),
        "scan" => Ok(SCAN_PROBLEM),
        "histogram" => Ok(HISTOGRAM_PROBLEM),
        "kmeans" => Ok(KMEANS_PROBLEM),
        "a7" => Ok(A7_PROBLEM),
        other => Err(format!("no Futhark problem {other} ({})", PROBLEM_NAMES.join(", "))),
    }
}

/// Every problem of the runner, by name, in the order the checklist lists them (`docs/CHECKLIST_futhark_problem.md`).
pub const PROBLEM_NAMES: [&str; 6] = ["reduce", "scan", "histogram", "pathfinder", "kmeans", "a7"];

/// Node executions one submit may be bounded by, on this machine (Apple M3 Max, the laptop that also drives the
/// display). Basis: the evaluator throughput measured earlier, about 17 G node evaluations per second (phylu tsr
/// speed report), times 0.1 s (a submit stays well under the second the display can wait), divided by a safety
/// factor of 10 for a kernel slower than the evaluator: 17e9 * 0.1 / 10 = 1.7e8. It bounds a Futhark candidate's work
/// on a stored case (`futhark_evaluator`) and every WGSL submit (`wgsl::work_guard`, which re-exports it); it lives
/// here, ungated, so the Futhark evaluator needs no `wgsl` feature.
pub const CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT: u128 = 170_000_000;

/// Load an instance's kingdom: its view's table, as a SHARED HANDLE, so a host, an interpreter or an evaluator that
/// holds it owns what it reads and can outlive the caller's scope (plan step 5a, F1).
pub fn load(inst: &Instance) -> Result<Arc<FutharkKingdom>, String> {
    inst.view()?.kingdom(&format!("Futhark Sig {}", inst.name)).map(Arc::new)
}

/// THE STATIC COST of a program of the matrix kingdom (A7; `crate::futhark_cost`, the generator's `costs.tsv`: a
/// polynomial in the roles): its traffic, allocations and serial depth at the given sizes. `None` for a kingdom with no
/// cost table (the signature family). It is reported, and available as a speed input where a fitness scheme asks for it
/// (the `shapes` schemes of `crate::futhark_search` do; the `futhark_sig` schemes time the GPU instead).
pub fn static_cost(k: &FutharkKingdom, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<Option<crate::futhark_cost::Score>, String> {
    if !k.duals.iter().any(|d| d.slot == "mat") {
        return Ok(None);
    }
    let table = crate::futhark_cost::CostTable::load(k)?;
    Ok(Some(crate::futhark_cost::score_at(&crate::futhark_cost::program_cost(tree, k, &table)?, sizes)?))
}

// ---- the declared signature -------------------------------------------------------------------------------------

/// A typed array of the signature: its name, element type and dimension roles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub elem: String,
    pub dims: Vec<String>,
}

/// The declared signature, read from the view's specification (`view.tsv`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    pub inputs: Vec<Param>,
    pub output: Param,
    /// The family's binder depth: a node runs at most `max(D) ** depth` times.
    pub depth: u32,
    /// Every dimension role, in order of first appearance.
    pub dims: Vec<String>,
    /// The SIZE INPUTS: dimensions given as an `i32` value parameter of the entry point (a `size` line of the
    /// specification; the parameter is in `inputs`, a scalar), rather than by an array's shape.
    pub size_inputs: Vec<String>,
    /// The view's element types, in the specification's order (`futhark_view::ViewSpec::elements`): the classes'
    /// texts carry an array's element when there are several (`a.f32.n`), and a program may hold values of any of
    /// them.
    pub elements: Vec<String>,
}

/// The signature of an instance: a signature view's from its specification ([`signature_of`]), a matrix view's from
/// its kingdom ([`matrix_signature`]); a shared handle, as [`load`]'s kingdom.
pub fn signature(inst: &Instance) -> Result<Arc<Signature>, String> {
    let view = inst.view()?;
    match view.spec.shape {
        Shape::Signature => signature_of(&view.spec).map_err(|e| format!("{}: {e}", inst.name)),
        Shape::Matrix => matrix_signature(&view.kingdom(inst.name)?, &view.spec).map_err(|e| format!("{}: {e}", inst.name)),
    }
    .map(Arc::new)
}

/// THE MATRIX VIEW'S SIGNATURE (A7): one input a matrix class, in the kingdom's class order, named by its input
/// terminal's spelling (`m_n_m`, the Futhark variable the program binds) with the class's roles as its dimensions; the
/// output the root class; the dims the specification's roles; depth 0 (the matrix family has no binders), no size inputs.
pub fn matrix_signature(k: &FutharkKingdom, spec: &ViewSpec) -> Result<Signature, String> {
    if spec.shape != Shape::Matrix {
        return Err("the specification is not a matrix view".into());
    }
    let elem = spec.element.clone();
    let mut inputs = Vec::new();
    for d in k.duals.iter().filter(|d| d.slot == "mat") {
        let ty = k.ty(&d.dual).ok_or_else(|| format!("our software has a bug: the dual {} has no type", d.dual))?;
        let term = k.rows.iter().find(|r| r.kind == FutharkKind::Terminal && r.output == ty).ok_or_else(|| format!("the matrix class {} has no input terminal", d.dual))?;
        inputs.push(Param { name: term.spelling.clone(), elem: elem.clone(), dims: d.form.split('.').map(str::to_string).collect() });
    }
    let root = k.roots();
    let [root] = root.as_slice() else { return Err(format!("the matrix view has {} roots, wants 1", root.len())) };
    let root = k.dual(*root).ok_or("our software has a bug: the root has no dual")?;
    if spec.roles.is_empty() {
        return Err("the matrix specification names no roles".into());
    }
    Ok(Signature { inputs, output: Param { name: "main".into(), elem: elem.clone(), dims: root.form.split('.').map(str::to_string).collect() }, depth: 0, dims: spec.roles.clone(), size_inputs: Vec::new(), elements: vec![elem] })
}

/// The signature a view specification declares: its inputs (a size input as an `i32` scalar), its output, its depth
/// and its dimensions.
pub fn signature_of(spec: &ViewSpec) -> Result<Signature, String> {
    if spec.shape != Shape::Signature {
        return Err("the specification is not a signature view".into());
    }
    let param = |p: &crate::futhark_view::SpecInput| Param { name: p.name.clone(), elem: if p.elem == "size" { "i32".into() } else { p.elem.clone() }, dims: p.dims.clone() };
    if spec.dims.is_empty() {
        return Err("the specification has no dims".into());
    }
    Ok(Signature {
        inputs: spec.inputs.iter().map(param).collect(),
        output: spec.output.as_ref().map(param).ok_or("the specification has no output")?,
        depth: spec.depth,
        dims: spec.dims.clone(),
        size_inputs: spec.sizes.clone(),
        elements: spec.elements.clone(),
    })
}

impl Signature {
    /// The sizes of every dimension role, bound from the inputs (each input's shape must have its declared rank;
    /// a role named twice must have one size; every size is at least 1, the family's precondition).
    pub fn sizes(&self, inputs: &[Array]) -> Result<BTreeMap<String, usize>, String> {
        if inputs.len() != self.inputs.len() {
            return Err(format!("{} inputs, the signature has {}", inputs.len(), self.inputs.len()));
        }
        let mut out = BTreeMap::new();
        for (p, a) in self.inputs.iter().zip(inputs) {
            if a.elem.name() != p.elem {
                return Err(format!("the input {} is {}, declared {}", p.name, a.elem.name(), p.elem));
            }
            if self.size_inputs.contains(&p.name) {
                let n = match (a.dims.as_slice(), a.data.as_slice()) {
                    ([], [n]) if (*n as i32) >= 1 => *n as usize,
                    _ => return Err(format!("the size input {} is {:?} {:?}, not one i32 of at least 1", p.name, a.dims, a.values().collect::<Vec<_>>())),
                };
                if let Some(old) = out.insert(p.name.clone(), n) {
                    if old != n {
                        return Err(format!("the dimension {} is {old} and {n}", p.name));
                    }
                }
                continue;
            }
            if a.dims.len() != p.dims.len() {
                return Err(format!("the input {} has rank {}, declared {}", p.name, a.dims.len(), p.dims.len()));
            }
            for (d, &n) in p.dims.iter().zip(&a.dims) {
                if n == 0 {
                    return Err(format!("the input {} has an empty dimension {d}: every dimension is at least 1", p.name));
                }
                if let Some(old) = out.insert(d.clone(), n) {
                    if old != n {
                        return Err(format!("the dimension {d} is {old} and {n}"));
                    }
                }
            }
        }
        Ok(out)
    }

    /// The dimensions bound by an array's shape (the entry point's size parameters): every one but the size inputs.
    pub fn shape_dims(&self) -> impl Iterator<Item = &String> {
        self.dims.iter().filter(|d| !self.size_inputs.contains(d))
    }

    /// A parameter's or the output's Futhark type, its dimensions suffixed with `i` (a case of the quality program;
    /// empty for the plain program); a size input's dimension is `i64.i32` of its parameter.
    pub fn futhark_type(&self, p: &Param, i: &str) -> String {
        let dim = |d: &String| if self.size_inputs.contains(d) { format!("[i64.i32 {d}{i}]") } else { format!("[{d}{i}]") };
        format!("{}{}", p.dims.iter().map(dim).collect::<String>(), p.elem)
    }
}

// ---- values: Futhark's binary format ----------------------------------------------------------------------------

/// An element type of the signature family's values: `i32` (Futhark's wrapping integer arithmetic) or `f32` (IEEE
/// single precision, held as its raw bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Elem {
    I32,
    F32,
}

impl Elem {
    pub fn name(self) -> &'static str {
        match self {
            Elem::I32 => "i32",
            Elem::F32 => "f32",
        }
    }

    /// The element of a Futhark type name; another element is an error that names it.
    pub fn of(name: &str) -> Result<Elem, String> {
        match name {
            "i32" => Ok(Elem::I32),
            "f32" => Ok(Elem::F32),
            other => Err(format!("the element {other} is not one of the signature family's (i32, f32)")),
        }
    }

    /// The type tag of Futhark's binary format (four bytes, space-padded).
    fn tag(self) -> &'static [u8; 4] {
        match self {
            Elem::I32 => b" i32",
            Elem::F32 => b" f32",
        }
    }

    fn of_tag(tag: &[u8]) -> Option<Elem> {
        [Elem::I32, Elem::F32].into_iter().find(|e| e.tag() == tag)
    }
}

/// A typed array of any rank (a scalar has no dims), row-major: its element and each value's RAW 32 BITS (an `i32`
/// as itself, an `f32` as its IEEE bits), exactly as Futhark's binary format writes them.
///
/// THE COMPARISON POLICY (`==`, the exactness of every case): two arrays agree when their dims and element agree
/// and every value agrees BIT FOR BIT, with one exception: any NaN agrees with any NaN. An `f32` NaN's payload is
/// not a value: Futhark prints every NaN as `f32.nan`, and which payload an operation keeps (of two NaN operands, of
/// `-x`) is the hardware's choice, so no program can be held to it. `-0.0` and `+0.0` are DIFFERENT values: different
/// bits, and Futhark prints `-0.0`. This is what the stored cases' bytes (`futhark c -b`) compare as, what the
/// k-means exact gate of `futhark_rust` compares (bytes), and `futhark_interpreter::identical`; the `i32` comparison
/// is unchanged (every bit pattern is one value).
#[derive(Debug, Clone)]
pub struct Array {
    pub dims: Vec<usize>,
    pub elem: Elem,
    /// The raw 32-bit words, row-major.
    pub data: Vec<u32>,
}

impl PartialEq for Array {
    fn eq(&self, o: &Array) -> bool {
        self.dims == o.dims && self.elem == o.elem && self.data.len() == o.data.len() && self.data.iter().zip(&o.data).all(|(a, b)| same_value(self.elem, *a, *b))
    }
}

impl Eq for Array {}

/// Whether two raw words are one value under the comparison policy: the same bits, or two NaNs.
pub fn same_value(elem: Elem, a: u32, b: u32) -> bool {
    a == b || (elem == Elem::F32 && f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan())
}

/// A raw word as an `f64`, exactly (every `i32` and every `f32` is one).
pub fn word_value(elem: Elem, w: u32) -> f64 {
    match elem {
        Elem::I32 => f64::from(w as i32),
        Elem::F32 => f64::from(f32::from_bits(w)),
    }
}

impl Array {
    /// An `i32` array.
    pub fn i32s(dims: Vec<usize>, data: Vec<i32>) -> Array {
        Array { dims, elem: Elem::I32, data: data.into_iter().map(|x| x as u32).collect() }
    }

    /// An `f32` array, each value as its bits.
    pub fn f32s(dims: Vec<usize>, data: Vec<f32>) -> Array {
        Array { dims, elem: Elem::F32, data: data.into_iter().map(f32::to_bits).collect() }
    }

    /// An `i32` scalar.
    pub fn scalar_i32(x: i32) -> Array {
        Array::i32s(Vec::new(), vec![x])
    }

    /// The number of values.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Every value as an `f64` (exact for both elements), in order.
    pub fn values(&self) -> impl Iterator<Item = f64> + '_ {
        self.data.iter().map(move |w| word_value(self.elem, *w))
    }

    /// The `i`-th value as an `f64`.
    pub fn value(&self, i: usize) -> f64 {
        word_value(self.elem, self.data[i])
    }

    /// The `i32` values of an `i32` array; an `f32` array is an error.
    pub fn i32_values(&self) -> Result<Vec<i32>, String> {
        match self.elem {
            Elem::I32 => Ok(self.data.iter().map(|w| *w as i32).collect()),
            Elem::F32 => Err("an f32 array has no i32 values".into()),
        }
    }

    /// The `f32` values of an `f32` array; an `i32` array is an error.
    pub fn f32_values(&self) -> Result<Vec<f32>, String> {
        match self.elem {
            Elem::F32 => Ok(self.data.iter().map(|w| f32::from_bits(*w)).collect()),
            Elem::I32 => Err("an i32 array has no f32 values".into()),
        }
    }

    /// Whether the `i`-th values of two arrays agree under the comparison policy.
    pub fn same_at(&self, o: &Array, i: usize) -> bool {
        self.elem == o.elem && same_value(self.elem, self.data[i], o.data[i])
    }
}

/// Values in Futhark's binary format (`b`, version 2, rank, the element's tag, the dims as `i64`, the data), one after
/// another: what a Futhark executable reads on stdin for its entry point's parameters, in order.
pub fn write_binary(values: &[&Array]) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        out.extend_from_slice(&[b'b', 2, v.dims.len() as u8]);
        out.extend_from_slice(v.elem.tag());
        for &d in &v.dims {
            out.extend_from_slice(&(d as i64).to_le_bytes());
        }
        for x in &v.data {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// Every value of a stream in Futhark's binary format (`i32` and `f32`; whitespace between values is skipped).
pub fn read_binary(raw: &[u8]) -> Result<Vec<Array>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    loop {
        while i < raw.len() && raw[i].is_ascii_whitespace() {
            i += 1;
        }
        if i == raw.len() {
            return Ok(out);
        }
        let head = raw.get(i..i + 7).ok_or("a value ends inside its header")?;
        if head[0] != b'b' || head[1] != 2 {
            return Err(format!("not Futhark's binary format: {:?}", &head[..2]));
        }
        let elem = Elem::of_tag(&head[3..7]).ok_or_else(|| format!("a value that is not i32 or f32: {:?}", String::from_utf8_lossy(&head[3..7])))?;
        let rank = head[2] as usize;
        i += 7;
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            let d = raw.get(i..i + 8).ok_or("a value ends inside its shape")?;
            let n = i64::from_le_bytes(d.try_into().map_err(|_| "a shape that is not 8 bytes")?);
            dims.push(usize::try_from(n).map_err(|_| format!("a negative dimension {n}"))?);
            i += 8;
        }
        let n: usize = dims.iter().product();
        let data = raw.get(i..i + 4 * n).ok_or("a value ends inside its data")?;
        out.push(Array { dims, elem, data: data.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() });
        i += 4 * n;
    }
}

/// The one value of a program's output.
pub fn read_one(raw: &[u8]) -> Result<Array, String> {
    let mut v = read_binary(raw)?;
    if v.len() != 1 {
        return Err(format!("the program wrote {} values, wants 1", v.len()));
    }
    Ok(v.remove(0))
}

// ---- the program ------------------------------------------------------------------------------------------------

/// The whole program: the entry point with the declared signature (the dimension roles as size parameters, a size
/// input as an `i32` parameter rebound as the size `i64.i32` of it), the zero array of every rank-1 class bound once,
/// around the rendered tree.
pub fn program(k: &FutharkKingdom, sig: &Signature, tree: &Tree) -> Result<String, String> {
    program_named(k, sig, tree, "main")
}

/// [`program`] with its entry point named `entry`: the text is [`program`]'s but for the name.
pub fn program_named(k: &FutharkKingdom, sig: &Signature, tree: &Tree, entry: &str) -> Result<String, String> {
    Ok(format!("{}{}", prelude(k), program_named_bare(k, sig, tree, entry)?))
}

/// THE PRELUDE a program of this kingdom needs before its entry: the `matmul` helper
/// (`crate::futhark::MATMUL_HELPER`) when a row spells `matmul` (the matrix family: Futhark 0.28.1 has no `matmul`),
/// else nothing. A batch program (`crate::futhark_batch`) holds many programs of one kingdom and emits the prelude
/// ONCE, before their bare texts: a definition repeated is a Futhark error.
pub fn prelude(k: &FutharkKingdom) -> String {
    if k.rows.iter().any(|r| r.spelling.starts_with("matmul ")) {
        format!("{}\n", crate::futhark::MATMUL_HELPER)
    } else {
        String::new()
    }
}

/// The entry point's parameter list and result type (`[n][m] (xs: [n][m]i32) : [m]i32`): the declared signature.
fn entry_header(sig: &Signature) -> String {
    let mut s = sig.shape_dims().map(|d| format!("[{d}]")).collect::<String>();
    for p in &sig.inputs {
        s.push_str(&format!(" ({}: {})", p.name, sig.futhark_type(p, "")));
    }
    format!("{s} : {}", sig.futhark_type(&sig.output, ""))
}

/// The body of a program: the size inputs rebound as sizes, the zero array of every rank-1 class a row spells
/// (`zeros_n`, or `zeros_f32_n` in a view of several elements: the signature family's `zeros` rows, each named by
/// its class; the matrix family spells none), in dimension order then element order, then the rendered tree.
fn program_body(k: &FutharkKingdom, sig: &Signature, tree: &Tree) -> Result<String, String> {
    let root = k.rows.get(tree.symbol as usize).ok_or("the root is not a row")?;
    if !k.dual(root.output).is_some_and(|d| d.root) {
        return Err(format!("the root {} is not the declared output", root.name));
    }
    let mut s = String::new();
    for d in &sig.size_inputs {
        s.push_str(&format!("  let {d} = i64.i32 {d}\n"));
    }
    let mut zeros: Vec<(usize, usize, String)> = Vec::new();
    for r in k.rows.iter().filter(|r| r.kind == FutharkKind::Terminal && r.spelling.starts_with("zeros_")) {
        let dual = &k.dual(r.output).ok_or("a row with no dual")?.dual;
        let class = Class::parse(dual.split(':').next().unwrap_or(""), &sig.elements, None)?;
        let (Some(elem), [dim]) = (class.elem(), class.dims()) else { return Err(format!("our software has a bug: the zero array {} is not of a rank-1 class", r.name)) };
        let d = sig.dims.iter().position(|x| x == dim).ok_or_else(|| format!("our software has a bug: the zero array {} is over no dimension of the signature", r.name))?;
        let e = sig.elements.iter().position(|x| x == elem).ok_or_else(|| format!("our software has a bug: the zero array {} is of no element of the signature", r.name))?;
        let line = format!("  let {} = replicate {dim} 0{elem}\n", r.spelling);
        if !zeros.iter().any(|(_, _, l)| *l == line) {
            zeros.push((d, e, line));
        }
    }
    zeros.sort();
    for (_, _, line) in zeros {
        s.push_str(&line);
    }
    // A body with no `let` (the matrix family) is the expression alone.
    s.push_str(&format!("  {}{}\n", if s.is_empty() { "" } else { "in " }, k.render(tree)?));
    Ok(s)
}

/// [`program_named`] WITHOUT the kingdom's [`prelude`]: a member of a batch program, which emits the prelude once.
pub fn program_named_bare(k: &FutharkKingdom, sig: &Signature, tree: &Tree, entry: &str) -> Result<String, String> {
    Ok(format!("entry {entry} {} =\n{}", entry_header(sig), program_body(k, sig, tree)?))
}

/// An entry point `entry` with the declared signature that returns `callee` applied to its parameters, in order (its
/// output's sizes left to the callee): how a program given as text, wrapped in a module so its own definitions cannot
/// clash with another copy's, becomes a member of a batch (`crate::futhark_batch`).
pub fn entry_calling(sig: &Signature, entry: &str, callee: &str) -> String {
    let mut s = format!("entry {entry} {}", sig.shape_dims().map(|d| format!("[{d}]")).collect::<String>());
    for p in &sig.inputs {
        s.push_str(&format!(" ({}: {})", p.name, sig.futhark_type(p, "")));
    }
    let args: Vec<&str> = sig.inputs.iter().map(|p| p.name.as_str()).collect();
    format!("{s} : {}{} =\n  {callee} {}\n", "[]".repeat(sig.output.dims.len()), sig.output.elem, args.join(" "))
}

/// THE QUALITY PROGRAM: the tree as a function of the declared signature (`#[noinline]`, so it is compiled once), and
/// an entry point that applies it to `n` cases at once, each with its own sizes, returning the `n` outputs. A GPU
/// process costs about 0.25 s to start whatever it runs (measured 2026-10-27, M3 Max), so one process for every case
/// instead of one a case is what makes the evaluation rate. The function's text is [`program`]'s body unchanged;
/// the timed program is [`program`] itself.
pub fn program_cases(k: &FutharkKingdom, sig: &Signature, tree: &Tree, n: usize) -> Result<String, String> {
    program_cases_named(k, sig, tree, n, "kernel", "main")
}

/// [`program_cases`] with its function named `kernel` and its entry point `entry` (a member of a batch program,
/// `crate::futhark_batch`): the text is [`program_cases`]'s but for the two names.
pub fn program_cases_named(k: &FutharkKingdom, sig: &Signature, tree: &Tree, n: usize, kernel: &str, entry: &str) -> Result<String, String> {
    Ok(format!("{}{}", prelude(k), program_cases_named_bare(k, sig, tree, n, kernel, entry)?))
}

/// [`program_cases_named`] WITHOUT the kingdom's [`prelude`]: a member of a batch program, which emits the prelude once.
pub fn program_cases_named_bare(k: &FutharkKingdom, sig: &Signature, tree: &Tree, n: usize, kernel: &str, entry: &str) -> Result<String, String> {
    if n == 0 {
        return Err("a quality program of no cases".into());
    }
    let body = program_body(k, sig, tree)?;
    let ty = |p: &Param, i: usize| sig.futhark_type(p, &i.to_string());
    let mut s = format!("#[noinline]\ndef {kernel} {} =\n{body}\nentry {entry} ", entry_header(sig));
    for i in 0..n {
        s.push_str(&sig.shape_dims().map(|d| format!("[{d}{i}]")).collect::<String>());
    }
    for i in 0..n {
        for p in &sig.inputs {
            s.push_str(&format!(" ({}{i}: {})", p.name, ty(p, i)));
        }
    }
    let outs: Vec<String> = (0..n).map(|i| ty(&sig.output, i)).collect();
    let calls: Vec<String> = (0..n).map(|i| format!("{kernel} {}", sig.inputs.iter().map(|p| format!("{}{i}", p.name)).collect::<Vec<_>>().join(" "))).collect();
    s.push_str(&format!(" : ({}) =\n  ({})\n", outs.join(", "), calls.join(", ")));
    Ok(s)
}

// ---- the host interpreter ---------------------------------------------------------------------------------------

/// What a row does, parsed once from its name and its output class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Add,
    Sub,
    Mul,
    Min,
    Max,
    Neg,
    Abs,
    /// `f32` only: IEEE division.
    Div,
    /// `f32` only: the square root (NaN below zero).
    Sqrt,
    Sel,
    Get,
    Sum,
    Minimum,
    Maximum,
    /// An index as a value of the element.
    Ival(Elem),
    /// The size of a dimension as a value of the element.
    DimVal(usize, Elem),
    Const,
    /// A bound variable (a loop state or an index): its slot, one a variable name of the kingdom's contexts
    /// ([`var_name`]: a class and the variable's ordinal in that class).
    Var(usize),
    Prev,
    Next(usize),
    Clamp(usize),
    Zero,
    Last(usize),
    /// A tabulate over a dimension (its size), binding the index slot; its element.
    Tab(usize, usize, Elem),
    /// A tabulate of ROWS over a dimension (the first of its rank-2 class), binding the index slot; its body is a row.
    TabRows(usize, usize),
    Replicate(usize, Elem),
    /// An elementwise operation of two arrays of one class (`map2`): add, sub, mul, min, max.
    Map2(Bin),
    /// The inclusive scan of an array (the `scan` family): add, min, max.
    Scan(Bin),
    /// The histogram into a dimension's array of values at indices (the `hist` family): add, min, max; its element.
    Hist(Bin, usize, Elem),
    /// The elementwise add histogram of ROWS: into `[K][D]` (the two dimensions) of `[D2][D]` rows at `[D2]` indices.
    HistRows(usize, usize, Elem),
    /// Each row of a `[K][D]` array divided by its `[K]` scalar (`f32` only).
    RowDiv,
    /// A value modulo a dimension's size (the `mod` family), Futhark's `%`: in `[0, D)`.
    Mod(usize),
    /// The row of an array.
    Row,
    Input(usize),
    Zeros(usize, Elem),
    /// A loop over a dimension binding an index slot and its state's slots: one for a scalar or an array state, two
    /// or three for a tuple state, which is DESTRUCTURED into its components' slots (`arity` of `states` are used);
    /// `from1` runs its counter over 1 .. D-1.
    Loop { dim: usize, index: usize, states: [usize; 3], arity: usize, from1: bool },
    /// THE TUPLE ROWS (master versions 4 and 5): a pair `({0}, {1})` or a triple `({0}, {1}, {2})`.
    Tuple,
    /// A projection: `{0}.0`, `{0}.1` or `{0}.2`.
    Proj(usize),
    /// `if {0} < 0 then {1} else {2}` over two tuples.
    PSel,
    /// The zero tuple: `arity` components of the elements.
    ZeroTuple([Elem; 3], usize),
    /// THE KERNEL BOUNDARY (master version 5): `let {S.var} = {0} in {1}`, the bound array's slot.
    Let(usize),
    /// A conversion into the element: `f32.i32` (exact below 2^24, else rounded to nearest), or the guarded
    /// `i32.f32` (a NaN is 0; the value clamped into i32's range, then truncated toward zero).
    Cvt(Elem),
    /// THE MATRIX FAMILY (A7). `transpose` of a rank-2 array.
    Transpose,
    /// The `matmul` helper: a result cell is the left fold from `0.0` of the products of a row of the first and a
    /// column of the second, in order (`crate::futhark_interpreter::matmul`).
    Matmul,
    /// `map f A`: a row function over the rows of a matrix.
    MapRows,
    /// `map2 f A B`: a binary row function over the pairs of rows of two matrices of one class.
    Map2Rows,
    /// A row function as a value (the matrix family's function classes).
    RowFn(RowFn),
    /// A binary row function as a value.
    Row2Fn(Row2Fn),
}

/// The elementwise operations of `map2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bin {
    Add,
    Sub,
    Mul,
    Min,
    Max,
}

impl Bin {
    fn of(name: &str) -> Option<Bin> {
        match name {
            "add" => Some(Bin::Add),
            "sub" => Some(Bin::Sub),
            "mul" => Some(Bin::Mul),
            "min" => Some(Bin::Min),
            "max" => Some(Bin::Max),
            _ => None,
        }
    }

    /// Whether Futhark's `scan` and `hist` rows exist for it (`add`, `min`, `max`: the ones with a neutral element).
    fn reduces(self) -> bool {
        matches!(self, Bin::Add | Bin::Min | Bin::Max)
    }
}

/// Futhark's `f32.min` (the C library's `fmin`, measured through `futhark c` 2026-11-03): a NaN loses to any number,
/// and of the two zeros `-0.0` is the smaller.
fn fmin(x: f32, y: f32) -> f32 {
    if x.is_nan() {
        y
    } else if y.is_nan() || x < y {
        x
    } else if y < x {
        y
    } else if x.is_sign_negative() {
        x
    } else {
        y
    }
}

/// Futhark's `f32.max` (`fmax`): a NaN loses to any number, and of the two zeros `+0.0` is the larger.
fn fmax(x: f32, y: f32) -> f32 {
    if x.is_nan() {
        y
    } else if y.is_nan() || x > y {
        x
    } else if y > x {
        y
    } else if x.is_sign_positive() {
        x
    } else {
        y
    }
}

/// Futhark's `i64.f32` (measured through `futhark c` 2026-11-03): truncation toward zero; a NaN, an infinity or a
/// value outside `i64`'s range is 0.
fn f32_to_i64(x: f32) -> i64 {
    if x.is_nan() || x.is_infinite() || x.abs() >= 9_223_372_036_854_775_808.0 {
        0
    } else {
        x as i64
    }
}

/// A scalar of the interpreter: the element's arithmetic is Futhark's (`i32` wraps; `f32` is IEEE, with Futhark's
/// `min`, `max` and conversions as measured).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Num {
    I(i32),
    F(f32),
}

impl Num {
    fn from_word(elem: Elem, w: u32) -> Num {
        match elem {
            Elem::I32 => Num::I(w as i32),
            Elem::F32 => Num::F(f32::from_bits(w)),
        }
    }

    fn word(self) -> u32 {
        match self {
            Num::I(x) => x as u32,
            Num::F(x) => x.to_bits(),
        }
    }

    fn elem(self) -> Elem {
        match self {
            Num::I(_) => Elem::I32,
            Num::F(_) => Elem::F32,
        }
    }

    fn zero(elem: Elem) -> Num {
        Num::from_word(elem, 0)
    }

    /// An index or a size as a value: `{E}.i64`.
    fn of_i64(elem: Elem, i: i64) -> Num {
        match elem {
            Elem::I32 => Num::I(i as i32),
            Elem::F32 => Num::F(i as f32),
        }
    }

    /// A literal of a pool (`-1`, `0.5`; the element's suffix already removed).
    fn parse(elem: Elem, text: &str) -> Result<Num, String> {
        Ok(match elem {
            Elem::I32 => Num::I(text.parse().map_err(|e| format!("the literal {text}: {e}"))?),
            Elem::F32 => Num::F(text.parse().map_err(|e| format!("the literal {text}: {e}"))?),
        })
    }

    /// The neutral element Futhark's `scan` and `hist` start from (`{E}.highest` is `i32.MAX`, `f32.inf`).
    fn neutral(elem: Elem, b: Bin) -> Num {
        match (elem, b) {
            (Elem::I32, Bin::Min) => Num::I(i32::MAX),
            (Elem::I32, Bin::Max) => Num::I(i32::MIN),
            (Elem::F32, Bin::Min) => Num::F(f32::INFINITY),
            (Elem::F32, Bin::Max) => Num::F(f32::NEG_INFINITY),
            (_, Bin::Add | Bin::Sub | Bin::Mul) => Num::zero(elem),
        }
    }

    fn bin(self, b: Bin, o: Num) -> Result<Num, String> {
        Ok(match (self, o) {
            (Num::I(x), Num::I(y)) => Num::I(match b {
                Bin::Add => x.wrapping_add(y),
                Bin::Sub => x.wrapping_sub(y),
                Bin::Mul => x.wrapping_mul(y),
                Bin::Min => x.min(y),
                Bin::Max => x.max(y),
            }),
            (Num::F(x), Num::F(y)) => Num::F(match b {
                Bin::Add => x + y,
                Bin::Sub => x - y,
                Bin::Mul => x * y,
                Bin::Min => fmin(x, y),
                Bin::Max => fmax(x, y),
            }),
            _ => return Err(format!("our software has a bug: {self:?} and {o:?} are of two elements")),
        })
    }

    fn neg(self) -> Num {
        match self {
            Num::I(x) => Num::I(x.wrapping_neg()),
            Num::F(x) => Num::F(-x),
        }
    }

    fn abs(self) -> Num {
        match self {
            Num::I(x) => Num::I(x.wrapping_abs()),
            Num::F(x) => Num::F(x.abs()),
        }
    }

    /// `{0} < 0{E}` (a NaN is not).
    fn is_negative(self) -> bool {
        match self {
            Num::I(x) => x < 0,
            Num::F(x) => x < 0.0,
        }
    }

    fn f32_only(self, what: &str) -> Result<f32, String> {
        match self {
            Num::F(x) => Ok(x),
            Num::I(_) => Err(format!("our software has a bug: {what} on an i32")),
        }
    }

    /// The `clamp` row: a value clamped into `[0, n)`. `i32`: `i64.max 0 (i64.min (n - 1) (i64.i32 x))`. `f32`: a NaN
    /// is 0, else the value clamped into `[0, n - 1]` as an `f32` before the conversion (`f32.i64 (n - 1)` may round),
    /// then clamped again as an `i64` (the master row M0056's spelling, mirrored).
    fn clamp(self, n: usize) -> i64 {
        let top = n as i64 - 1;
        match self {
            Num::I(x) => i64::from(x).min(top).max(0),
            Num::F(x) => {
                let i = if x.is_nan() { 0 } else { f32_to_i64(fmax(0.0, fmin(top as f32, x))) };
                i.min(top).max(0)
            }
        }
    }

    /// The `hist` rows' index of a value into `[0, n)`: `i32`: `i64.i32 x`; `f32`: a NaN is -1, else the value
    /// clamped into `[-1, n]` as an `f32` then truncated (the master rows M0057-M0059 and M0065, mirrored). Outside
    /// `[0, n)` Futhark's `hist` ignores it: `None`.
    fn hist_index(self, n: usize) -> Option<usize> {
        let i = match self {
            Num::I(x) => i64::from(x),
            Num::F(x) => {
                if x.is_nan() {
                    -1
                } else {
                    f32_to_i64(fmax(-1.0, fmin(n as f32, x)))
                }
            }
        };
        usize::try_from(i).ok().filter(|i| *i < n)
    }

    /// The conversion rows (master version 4). Into `f32`: Futhark's `f32.i32`, exact below 2^24 and rounded to
    /// nearest above (Rust's `as`). Into `i32`: the guarded `i32.f32` of M0076, mirrored: a NaN is 0, the value is
    /// clamped into `[-2^31, 2147483520]` (the largest `f32` below 2^31) as an `f32`, then truncated toward zero.
    fn convert(self, into: Elem) -> Num {
        match (self, into) {
            (Num::I(x), Elem::F32) => Num::F(x as f32),
            (Num::F(x), Elem::I32) => Num::I(if x.is_nan() { 0 } else { fmax(-2_147_483_648.0, fmin(2_147_483_520.0, x)) as i32 }),
            (n, _) => n,
        }
    }
}

/// A program's work on given sizes ([`Interp::work_parts`]): its node executions, and its loops' state traffic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Work {
    /// Node executions, the cells reductions and traversals read or write included: the measure
    /// [`CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT`] is calibrated in (node evaluations) and bounds on a stored case, as it
    /// did every case before 2026-11-04.
    pub nodes: u128,
    /// The cells of the loops' array states copied a trip: part of a run's time, which the timing bound reads at the
    /// timing input, never the per-submit cap (the F3a replay of 2026-11-04: a histogram program the runner scored
    /// on every case, 9 s for its runs, is 1.1e9 with them on a case of n = 1000 and under the cap without them).
    pub state: u128,
}

impl Work {
    /// Both parts: a run's work, the timing bound's measure.
    pub fn total(&self) -> u128 {
        self.nodes + self.state
    }
}

/// THE INTERPRETER of one instance: a master 5 view's ([`Interp::new`]) or a master 6 view's ([`Interp::new6`], plan
/// section 7 task F3: the meaning read from G3's decoding, the adapters interpreted, sizes from the properties). The
/// two answer the same calls ([`Interp::eval`], [`Interp::work`], [`Interp::work_parts`]) with the same signatures, so
/// phylu's callers build unchanged and take a master 6 interpreter when their kingdom is one. It OWNS the kingdom and
/// the signature it reads (shared handles), so it lives as long as its holder.
pub struct Interp(Masters);

/// Which master an interpreter reads (master 5 stays loadable until plan task D1).
enum Masters {
    Five(Interp5),
    Six(Box<interp6::Interp6>),
}

impl Interp {
    /// The interpreter of a master 5 kingdom and its signature; it keeps a clone of each handle.
    pub fn new(k: &Arc<FutharkKingdom>, sig: &Arc<Signature>) -> Result<Interp, String> {
        Interp5::new(k, sig).map(|i| Interp(Masters::Five(i)))
    }

    /// The interpreter of a master 6 kingdom whose cases are `sig`'s (its problem's signature: the inputs by name, the
    /// sizes they bind); it keeps a clone of each handle.
    pub fn new6(k: &Arc<crate::futhark_table::Kingdom6>, sig: &Arc<Signature>) -> Result<Interp, String> {
        interp6::Interp6::new(k, sig).map(|i| Interp(Masters::Six(Box::new(i))))
    }

    /// The value of an expressed tree on the inputs: the program's output, as Futhark computes it. A master 6 tree is
    /// decoded first (passes 2 to 4), and every value is checked against its properties as Futhark checks its sizes.
    pub fn eval(&self, tree: &Tree, inputs: &[Array]) -> Result<Array, String> {
        match &self.0 {
            Masters::Five(i) => i.eval(tree, inputs),
            Masters::Six(i) => i.eval(tree, inputs),
        }
    }

    /// The node executions of a tree on given sizes, the loops' state traffic included ([`Work::total`]).
    pub fn work(&self, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<u128, String> {
        match &self.0 {
            Masters::Five(i) => i.work(tree, sizes),
            Masters::Six(i) => i.work(tree, sizes),
        }
    }

    /// The work of a tree on given sizes, in two parts ([`Work`]); a master 6 tree's adapters are charged in
    /// [`Work::nodes`], the per-submit cap's measure.
    pub fn work_parts(&self, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<Work, String> {
        match &self.0 {
            Masters::Five(i) => i.work_parts(tree, sizes),
            Masters::Six(i) => i.work_parts(tree, sizes),
        }
    }

    /// THE STATIC COST of a tree at `sizes`: master 5's for the matrix kingdom ([`static_cost`]; `None` for its
    /// signature views), master 6's for every view (`crate::futhark_cost::program_cost6`, adapters charged).
    pub fn static_cost(&self, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<Option<crate::futhark_cost::Score>, String> {
        match &self.0 {
            Masters::Five(i) => static_cost(&i.k, tree, sizes),
            Masters::Six(i) => i.static_cost(tree, sizes).map(Some),
        }
    }

    /// The bound on any tree of a layout: its most nodes, each run at most `max(D) ** depth` times and each costing at
    /// most `1 + max(D) ** 2`, on the given largest dimension.
    pub fn layout_bound(&self, layout: &GeneLayout, max_dim: usize) -> u128 {
        let depth = match &self.0 {
            Masters::Five(i) => i.sig.depth,
            Masters::Six(i) => i.depth(),
        };
        let d = max_dim as u128;
        layout.max_nodes() as u128 * d.pow(depth) * (1 + d * d)
    }
}

/// Master 5's interpreter: each row's meaning, parsed once.
struct Interp5 {
    k: Arc<FutharkKingdom>,
    sig: Arc<Signature>,
    ops: Vec<Op>,
    literals: Vec<Option<Vec<Num>>>,
    /// The number of variable slots (one a variable name of the kingdom's contexts).
    slots: usize,
}

/// An array value as a view: shared raw words, an offset, the dims, the element.
#[derive(Debug, Clone)]
struct View {
    data: Rc<Vec<u32>>,
    off: usize,
    dims: Vec<usize>,
    elem: Elem,
}

impl View {
    fn of(data: Vec<u32>, dims: Vec<usize>, elem: Elem) -> View {
        View { data: Rc::new(data), off: 0, dims, elem }
    }
    fn len(&self) -> usize {
        self.dims.iter().product()
    }
    fn slice(&self) -> &[u32] {
        &self.data[self.off..self.off + self.len()]
    }
    fn num(&self, w: u32) -> Num {
        Num::from_word(self.elem, w)
    }
}

#[derive(Debug, Clone)]
enum Val {
    S(Num),
    I(i64),
    A(View),
    /// A tuple (a pair or a triple).
    Tup(Vec<Val>),
    /// A row function (the matrix family).
    F(RowFn),
    /// A binary row function (the matrix family).
    F2(Row2Fn),
}

/// The bound variables by slot (a binder sets its slots and restores the outer values after).
struct Env {
    sizes: Vec<usize>,
    inputs: Vec<View>,
    vars: Vec<Option<Val>>,
}

impl Interp5 {
    /// The interpreter of a kingdom and its signature; it keeps a clone of each handle.
    pub fn new(k: &Arc<FutharkKingdom>, sig: &Arc<Signature>) -> Result<Interp5, String> {
        let dim = |d: &str| sig.dims.iter().position(|x| x == d).ok_or_else(|| format!("no dimension {d}"));
        // A dual's class, read through the class grammar with the view's elements.
        let class_of = |dual: &str| -> Result<Class, String> { Class::parse(dual.split(':').next().unwrap_or(""), &sig.elements, None).map_err(|e| format!("{dual}: {e}")) };
        let elem_of = |c: &Class| -> Result<Elem, String> { Elem::of(c.elem().ok_or_else(|| format!("the class {} has no element", c.text(sig.elements.len() > 1)))?) };
        // The dimensions of an array class (`a.k.d` -> [k, d]), by index.
        let class_dims = |dual: &str| -> Result<Vec<usize>, String> {
            let c = class_of(dual)?;
            if c.dims().is_empty() {
                return Err(format!("{dual} is not an array class"));
            }
            c.dims().iter().map(|d| dim(d)).collect()
        };
        let class_dim = |dual: &str| -> Result<usize, String> {
            match class_of(dual)? {
                Class::Index(d) => dim(&d),
                Class::Array { dims, .. } if dims.len() == 1 => dim(&dims[0]),
                _ => Err(format!("{dual} is not an index or a rank-1 class")),
            }
        };
        // THE VARIABLE SLOTS: one a variable name of the kingdom's contexts (a class and an ordinal, `var_name`: the
        // indices over each dimension, the loop states of every class a context binds), read from the duals.
        let mut slot_of: BTreeMap<String, usize> = BTreeMap::new();
        for d in &k.duals {
            if let Some(ctx) = Ctx::of_dual(&d.dual)? {
                for (c, n) in &ctx.classes {
                    for ord in 1..=*n {
                        let next = slot_of.len();
                        slot_of.entry(var_name(c, ord)).or_insert(next);
                    }
                }
            }
        }
        // A binder's slot: the next ordinal of its class in its own (output's) context, `more` further on.
        let binder_slot = |out: &str, cls: &str, more: u32| -> Result<usize, String> {
            let ctx = Ctx::of_dual(out)?.ok_or_else(|| format!("our software has a bug: a binder {out} outside a context"))?;
            let name = var_name(cls, ctx.count(cls) + 1 + more);
            slot_of.get(&name).copied().ok_or_else(|| format!("our software has a bug: no slot for the variable {name} of {out}"))
        };
        let multi = sig.elements.len() > 1;
        let mut ops = Vec::with_capacity(k.rows.len());
        let mut literals = Vec::with_capacity(k.duals.len());
        for d in &k.duals {
            literals.push(if d.literals.is_empty() {
                None
            } else {
                let elem = Elem::of(&d.slot).map_err(|e| format!("the type {} has a literal pool: {e}", d.dual))?;
                Some(d.literals.iter().map(|l| Num::parse(elem, l.trim_end_matches(elem.name()))).collect::<Result<Vec<_>, _>>()?)
            });
        }
        for r in &k.rows {
            let out = &k.dual(r.output).ok_or("a row with no dual")?.dual;
            let (op, _) = r.name.split_once('@').unwrap_or((r.name.as_str(), ""));
            let out_class = out.split(':').next().unwrap_or("");
            // The row's element: its output's, for a scalar or an array.
            let out_elem = || -> Result<Elem, String> { elem_of(&class_of(out)?) };
            // A row named by its element where its output class has none (`clamp.f32`, `psel.i32` in a view of
            // several elements; `clamp`, `psel` in a view of one): the op without the element.
            let bare = |head: &str| -> bool { op == head || (multi && op.strip_prefix(head).and_then(|rest| rest.strip_prefix('.')).is_some_and(|e| sig.elements.iter().any(|x| x == e))) };
            let o = match op {
                "add" => Op::Add,
                "sub" => Op::Sub,
                "mul" => Op::Mul,
                "min" => Op::Min,
                "max" => Op::Max,
                "neg" => Op::Neg,
                "abs" => Op::Abs,
                "div" => Op::Div,
                "sqrt" => Op::Sqrt,
                "sel" => Op::Sel,
                "prev" => Op::Prev,
                "next" => Op::Next(class_dim(out)?),
                _ if bare("clamp") => Op::Clamp(class_dim(out)?),
                "zero" => match class_of(out)? {
                    Class::Index(_) => Op::Zero,
                    Class::Tuple(parts) if (2..=3).contains(&parts.len()) => {
                        let mut elems = [Elem::I32; 3];
                        for (slot, p) in elems.iter_mut().zip(&parts) {
                            *slot = elem_of(p)?;
                        }
                        Op::ZeroTuple(elems, parts.len())
                    }
                    _ => return Err(format!("our software has a bug: the zero row {} is not an index's, a pair's or a triple's", r.name)),
                },
                "last" => Op::Last(class_dim(out)?),
                "tab" => {
                    let d = class_dim(out)?;
                    Op::Tab(d, binder_slot(out, &format!("ix.{}", sig.dims[d]), 0)?, out_elem()?)
                }
                "tabrows" => {
                    let ds = class_dims(out)?;
                    let [kd, _] = ds.as_slice() else { return Err(format!("our software has a bug: the tabrows row {} is not of a rank-2 class", r.name)) };
                    Op::TabRows(*kd, binder_slot(out, &format!("ix.{}", sig.dims[*kd]), 0)?)
                }
                "rowdiv" => Op::RowDiv,
                "replicate" => Op::Replicate(class_dim(out)?, out_elem()?),
                "zeros" => Op::Zeros(class_dim(out)?, out_elem()?),
                "pair" | "triple" => Op::Tuple,
                _ if bare("psel") => Op::PSel,
                // The kernel boundary binds its first argument's class: the next variable of that class in its
                // context (the let's output's).
                "let" => {
                    let bound = &k.dual(*r.inputs.first().ok_or_else(|| format!("our software has a bug: the let {} has no bound argument", r.name))?).ok_or("a let with no dual")?.dual;
                    Op::Let(binder_slot(out, bound.split(':').next().unwrap_or(""), 0)?)
                }
                _ if r.kind == FutharkKind::Const => Op::Const,
                _ if op.starts_with("fst.") => Op::Proj(0),
                _ if op.starts_with("snd.") => Op::Proj(1),
                _ if op.starts_with("thd.") => Op::Proj(2),
                _ if op.starts_with("cvt.") => Op::Cvt(out_elem()?),
                _ if op.starts_with("get.") => Op::Get,
                _ if op.starts_with("sum.") => Op::Sum,
                _ if op.starts_with("minimum.") => Op::Minimum,
                _ if op.starts_with("maximum.") => Op::Maximum,
                _ if op.starts_with("ival.") => Op::Ival(out_elem()?),
                _ if op.starts_with("row.") => Op::Row,
                _ if op.starts_with("map2.") => Op::Map2(Bin::of(&op[5..]).ok_or_else(|| format!("our software has a bug: the map2 row {}", r.name))?),
                _ if op.starts_with("scan.") => Op::Scan(Bin::of(&op[5..]).filter(|b| b.reduces()).ok_or_else(|| format!("our software has a bug: the scan row {}", r.name))?),
                _ if op.starts_with("histrows.add.") => {
                    let ds = class_dims(out)?;
                    let [kd, dd] = ds.as_slice() else { return Err(format!("our software has a bug: the histrows row {} is not of a rank-2 class", r.name)) };
                    Op::HistRows(*kd, *dd, out_elem()?)
                }
                _ if op.starts_with("hist.") => {
                    let b = op.split('.').nth(1).and_then(Bin::of).filter(|b| b.reduces()).ok_or_else(|| format!("our software has a bug: the histogram row {}", r.name))?;
                    Op::Hist(b, class_dim(out)?, out_elem()?)
                }
                _ if op.starts_with("mod.") => Op::Mod(dim(&op[4..])?),
                // A size input's row is its dimension as a value, not an input array.
                _ if sig.size_inputs.iter().any(|d| d == op) => Op::DimVal(dim(op)?, out_elem()?),
                _ if op.starts_with("loop1.") || op.starts_with("loop.") => {
                    let (from1, d) = match op.strip_prefix("loop1.") {
                        Some(d) => (true, d),
                        None => (false, &op[5..]),
                    };
                    // A tuple state is destructured: the loop binds its components, a later one further on for every
                    // earlier one of its class (`(s_f32, s2_f32, s_i32)`).
                    let (states, arity) = match class_of(out)? {
                        Class::Tuple(parts) if (2..=3).contains(&parts.len()) => {
                            let texts: Vec<String> = parts.iter().map(|p| p.text(multi)).collect();
                            let mut slots = [0usize; 3];
                            for (i, t) in texts.iter().enumerate() {
                                slots[i] = binder_slot(out, t, texts[..i].iter().filter(|x| *x == t).count() as u32)?;
                            }
                            (slots, parts.len())
                        }
                        Class::Tuple(parts) => return Err(format!("our software has a bug: the loop {} carries a tuple of {} components", r.name, parts.len())),
                        _ => ([binder_slot(out, out_class, 0)?, 0, 0], 1),
                    };
                    Op::Loop { dim: dim(d)?, index: binder_slot(out, &format!("ix.{d}"), 0)?, states, arity, from1 }
                }
                _ if r.kind == FutharkKind::Terminal && slot_of.contains_key(op) => Op::Var(slot_of[op]),
                // The matrix family (A7): its operators, its row functions (function rows of no arguments, and the
                // constant terminals), and its inputs, named by their spelling (`mat.n.m.input` spells `m_n_m`).
                "transpose" => Op::Transpose,
                "matmul" => Op::Matmul,
                "map_rows" => Op::MapRows,
                "map2_rows" => Op::Map2Rows,
                _ if row_fn(&r.name).is_some() => Op::RowFn(row_fn(&r.name).ok_or("a row function")?),
                _ if row2_fn(&r.name).is_some() => Op::Row2Fn(row2_fn(&r.name).ok_or("a binary row function")?),
                _ if r.kind == FutharkKind::Terminal && sig.inputs.iter().any(|p| p.name == r.spelling) => Op::Input(sig.inputs.iter().position(|p| p.name == r.spelling).ok_or("an input")?),
                _ => {
                    if let Some(i) = sig.inputs.iter().position(|p| p.name == op) {
                        Op::Input(i)
                    } else if let Ok(d) = dim(op) {
                        Op::DimVal(d, out_elem()?)
                    } else {
                        return Err(format!("our software has a bug: the row {} has no meaning in the interpreter", r.name));
                    }
                }
            };
            ops.push(o);
        }
        Ok(Interp5 { k: Arc::clone(k), sig: Arc::clone(sig), ops, literals, slots: slot_of.len() })
    }

    /// The value of an expressed tree on the inputs: the program's output, as Futhark computes it.
    pub fn eval(&self, tree: &Tree, inputs: &[Array]) -> Result<Array, String> {
        let sizes_map = self.sig.sizes(inputs)?;
        let sizes: Vec<usize> = self.sig.dims.iter().map(|d| sizes_map[d]).collect();
        let mut env = Env { inputs: inputs.iter().map(|a| View::of(a.data.clone(), a.dims.clone(), a.elem)).collect(), vars: vec![None; self.slots], sizes };
        match self.at(tree, &mut env)? {
            Val::A(v) => Ok(Array { dims: v.dims.clone(), elem: v.elem, data: v.slice().to_vec() }),
            Val::S(x) => Ok(Array { dims: vec![], elem: x.elem(), data: vec![x.word()] }),
            Val::I(_) => Err("our software has a bug: a program whose value is an index".into()),
            Val::Tup(_) => Err("our software has a bug: a program whose value is a tuple (a tuple root is not built)".into()),
            Val::F(_) | Val::F2(_) => Err("our software has a bug: a program whose value is a row function".into()),
        }
    }

    fn s(&self, t: &Tree, env: &mut Env) -> Result<Num, String> {
        match self.at(t, env)? {
            Val::S(x) => Ok(x),
            other => Err(format!("our software has a bug: a scalar slot holds {other:?}")),
        }
    }
    fn i(&self, t: &Tree, env: &mut Env) -> Result<i64, String> {
        match self.at(t, env)? {
            Val::I(x) => Ok(x),
            other => Err(format!("our software has a bug: an index slot holds {other:?}")),
        }
    }
    fn a(&self, t: &Tree, env: &mut Env) -> Result<View, String> {
        match self.at(t, env)? {
            Val::A(x) => Ok(x),
            other => Err(format!("our software has a bug: an array slot holds {other:?}")),
        }
    }
    fn tuple(&self, t: &Tree, env: &mut Env) -> Result<Vec<Val>, String> {
        match self.at(t, env)? {
            Val::Tup(v) => Ok(v),
            other => Err(format!("our software has a bug: a tuple slot holds {other:?}")),
        }
    }

    /// A left fold over an array's values from a neutral element, in order (Futhark's sequential `reduce`).
    fn fold(&self, v: &View, b: Bin) -> Result<Num, String> {
        let mut acc = Num::neutral(v.elem, b);
        for w in v.slice() {
            acc = acc.bin(b, v.num(*w))?;
        }
        Ok(acc)
    }

    fn at(&self, t: &Tree, env: &mut Env) -> Result<Val, String> {
        let op = *self.ops.get(t.symbol as usize).ok_or("a symbol that is not a row")?;
        let kid = |i: usize| t.kids.get(i).ok_or_else(|| format!("{} lacks argument {i}", self.k.rows[t.symbol as usize].name));
        Ok(match op {
            Op::Add => Val::S(self.s(kid(0)?, env)?.bin(Bin::Add, self.s(kid(1)?, env)?)?),
            Op::Sub => Val::S(self.s(kid(0)?, env)?.bin(Bin::Sub, self.s(kid(1)?, env)?)?),
            Op::Mul => Val::S(self.s(kid(0)?, env)?.bin(Bin::Mul, self.s(kid(1)?, env)?)?),
            Op::Min => Val::S(self.s(kid(0)?, env)?.bin(Bin::Min, self.s(kid(1)?, env)?)?),
            Op::Max => Val::S(self.s(kid(0)?, env)?.bin(Bin::Max, self.s(kid(1)?, env)?)?),
            Op::Neg => Val::S(self.s(kid(0)?, env)?.neg()),
            Op::Abs => Val::S(self.s(kid(0)?, env)?.abs()),
            Op::Div => Val::S(Num::F(self.s(kid(0)?, env)?.f32_only("div")? / self.s(kid(1)?, env)?.f32_only("div")?)),
            Op::Sqrt => Val::S(Num::F(self.s(kid(0)?, env)?.f32_only("sqrt")?.sqrt())),
            Op::Sel => {
                // Futhark evaluates only the branch taken.
                let c = self.s(kid(0)?, env)?;
                Val::S(if c.is_negative() { self.s(kid(1)?, env)? } else { self.s(kid(2)?, env)? })
            }
            Op::PSel => {
                let c = self.s(kid(0)?, env)?;
                self.at(if c.is_negative() { kid(1)? } else { kid(2)? }, env)?
            }
            Op::Tuple => Val::Tup(t.kids.iter().map(|c| self.at(c, env)).collect::<Result<Vec<_>, _>>()?),
            Op::Proj(i) => {
                let mut v = self.tuple(kid(0)?, env)?;
                if i >= v.len() {
                    return Err(format!("our software has a bug: the projection {i} of a tuple of {}", v.len()));
                }
                v.swap_remove(i)
            }
            Op::ZeroTuple(elems, arity) => Val::Tup(elems[..arity].iter().map(|e| Val::S(Num::zero(*e))).collect()),
            Op::Let(slot) => {
                // The bound expression runs once, then the body with the variable in scope; the outer value of the
                // slot (none at the top) is restored after.
                let bound = self.at(kid(0)?, env)?;
                let outer = env.vars[slot].replace(bound);
                let out = self.at(kid(1)?, env);
                env.vars[slot] = outer;
                out?
            }
            Op::Cvt(into) => Val::S(self.s(kid(0)?, env)?.convert(into)),
            Op::Get => {
                let a = self.a(kid(0)?, env)?;
                let i = self.i(kid(1)?, env)?;
                let n = a.len();
                if a.dims.len() != 1 || i < 0 || i as usize >= n {
                    return Err(format!("our software has a bug: the index {i} is out of range for {:?}", a.dims));
                }
                Val::S(a.num(a.data[a.off + i as usize]))
            }
            Op::Sum => Val::S(self.fold(&self.a(kid(0)?, env)?, Bin::Add)?),
            Op::Minimum => Val::S(self.fold(&self.a(kid(0)?, env)?, Bin::Min)?),
            Op::Maximum => Val::S(self.fold(&self.a(kid(0)?, env)?, Bin::Max)?),
            Op::Ival(elem) => Val::S(Num::of_i64(elem, self.i(kid(0)?, env)?)),
            Op::DimVal(d, elem) => Val::S(Num::of_i64(elem, env.sizes[d] as i64)),
            Op::Const => {
                let row = &self.k.rows[t.symbol as usize];
                let ty = match row.output {
                    Ty::Futhark(i) => i as usize,
                    _ => return Err("a constant outside the family".into()),
                };
                let pool = self.literals[ty].as_ref().ok_or("a constant of a type with no pool")?;
                Val::S(*pool.get(t.const_index.ok_or("a constant with no index")? as usize).ok_or("a constant index past the pool")?)
            }
            Op::Var(k) => env.vars.get(k).cloned().flatten().ok_or_else(|| format!("our software has a bug: the variable of slot {k} is read unbound"))?,
            Op::Prev => Val::I((self.i(kid(0)?, env)? - 1).max(0)),
            Op::Next(d) => Val::I((self.i(kid(0)?, env)? + 1).min(env.sizes[d] as i64 - 1)),
            Op::Clamp(d) => Val::I(self.s(kid(0)?, env)?.clamp(env.sizes[d])),
            Op::Zero => Val::I(0),
            Op::Last(d) => Val::I(env.sizes[d] as i64 - 1),
            Op::Tab(d, slot, elem) => {
                let n = env.sizes[d];
                let mut out = Vec::with_capacity(n);
                let outer = env.vars[slot].take();
                for j in 0..n {
                    env.vars[slot] = Some(Val::I(j as i64));
                    out.push(self.s(kid(0)?, env)?.word());
                }
                env.vars[slot] = outer;
                Val::A(View::of(out, vec![n], elem))
            }
            Op::TabRows(d, slot) => {
                let n = env.sizes[d];
                let mut out = Vec::new();
                let mut cols = None;
                let outer = env.vars[slot].take();
                for j in 0..n {
                    env.vars[slot] = Some(Val::I(j as i64));
                    let row = self.a(kid(0)?, env)?;
                    if row.dims.len() != 1 || cols.is_some_and(|(c, e)| c != row.dims[0] || e != row.elem) {
                        return Err(format!("our software has a bug: a tabulate of rows whose row {j} is {:?}", row.dims));
                    }
                    cols = Some((row.dims[0], row.elem));
                    out.extend_from_slice(row.slice());
                }
                env.vars[slot] = outer;
                let (cols, elem) = cols.ok_or("our software has a bug: a tabulate of rows over an empty dimension")?;
                Val::A(View::of(out, vec![n, cols], elem))
            }
            Op::Replicate(d, elem) => {
                let n = env.sizes[d];
                let x = self.s(kid(0)?, env)?.word();
                Val::A(View::of(vec![x; n], vec![n], elem))
            }
            Op::Map2(b) => {
                let x = self.a(kid(0)?, env)?;
                let y = self.a(kid(1)?, env)?;
                if x.dims != y.dims || x.dims.len() != 1 || x.elem != y.elem {
                    return Err(format!("our software has a bug: map2 over {:?} and {:?}", x.dims, y.dims));
                }
                let out: Vec<u32> = x.slice().iter().zip(y.slice()).map(|(p, q)| x.num(*p).bin(b, y.num(*q)).map(Num::word)).collect::<Result<_, _>>()?;
                Val::A(View::of(out, x.dims.clone(), x.elem))
            }
            Op::Scan(b) => {
                let x = self.a(kid(0)?, env)?;
                let mut acc = Num::neutral(x.elem, b);
                let mut out = Vec::with_capacity(x.len());
                for w in x.slice() {
                    acc = acc.bin(b, x.num(*w))?;
                    out.push(acc.word());
                }
                Val::A(View::of(out, x.dims.clone(), x.elem))
            }
            Op::Hist(b, d, elem) => {
                let is = self.a(kid(0)?, env)?;
                let vs = self.a(kid(1)?, env)?;
                if is.dims != vs.dims || is.dims.len() != 1 || vs.elem != elem {
                    return Err(format!("our software has a bug: a histogram of {:?} at {:?}", vs.dims, is.dims));
                }
                let n = env.sizes[d];
                let mut out = vec![Num::neutral(elem, b); n];
                for (i, v) in is.slice().iter().zip(vs.slice()) {
                    // Futhark's hist ignores an index outside [0, n).
                    if let Some(i) = is.num(*i).hist_index(n) {
                        out[i] = out[i].bin(b, vs.num(*v))?;
                    }
                }
                Val::A(View::of(out.into_iter().map(Num::word).collect(), vec![n], elem))
            }
            Op::HistRows(kd, dd, elem) => {
                let is = self.a(kid(0)?, env)?;
                let vs = self.a(kid(1)?, env)?;
                let (k, d) = (env.sizes[kd], env.sizes[dd]);
                if is.dims.len() != 1 || vs.dims.len() != 2 || vs.dims[0] != is.dims[0] || vs.dims[1] != d || vs.elem != elem {
                    return Err(format!("our software has a bug: a histogram of rows {:?} at {:?} into [{k}][{d}]", vs.dims, is.dims));
                }
                let mut out = vec![Num::zero(elem); k * d];
                let rows = vs.slice();
                for (i, idx) in is.slice().iter().enumerate() {
                    if let Some(b) = is.num(*idx).hist_index(k) {
                        for e in 0..d {
                            out[b * d + e] = out[b * d + e].bin(Bin::Add, vs.num(rows[i * d + e]))?;
                        }
                    }
                }
                Val::A(View::of(out.into_iter().map(Num::word).collect(), vec![k, d], elem))
            }
            Op::RowDiv => {
                let m = self.a(kid(0)?, env)?;
                let s = self.a(kid(1)?, env)?;
                if m.dims.len() != 2 || s.dims.len() != 1 || s.dims[0] != m.dims[0] {
                    return Err(format!("our software has a bug: the rows of {:?} divided by {:?}", m.dims, s.dims));
                }
                let d = m.dims[1];
                let mut out = Vec::with_capacity(m.len());
                for (r, sw) in s.slice().iter().enumerate() {
                    let div = s.num(*sw).f32_only("rowdiv")?;
                    for w in &m.slice()[r * d..(r + 1) * d] {
                        out.push((m.num(*w).f32_only("rowdiv")? / div).to_bits());
                    }
                }
                Val::A(View::of(out, m.dims.clone(), m.elem))
            }
            Op::Mod(d) => match self.s(kid(0)?, env)? {
                Num::I(x) => Val::S(Num::I(x.rem_euclid(env.sizes[d] as i32))),
                other => return Err(format!("our software has a bug: mod on {other:?}")),
            },
            Op::Row => {
                let m = self.a(kid(0)?, env)?;
                let i = self.i(kid(1)?, env)?;
                if m.dims.len() < 2 || i < 0 || i as usize >= m.dims[0] {
                    return Err(format!("our software has a bug: the row {i} is out of range for {:?}", m.dims));
                }
                let dims = m.dims[1..].to_vec();
                let stride: usize = dims.iter().product();
                Val::A(View { data: m.data.clone(), off: m.off + i as usize * stride, dims, elem: m.elem })
            }
            Op::Input(i) => Val::A(env.inputs[i].clone()),
            Op::Zeros(d, elem) => Val::A(View::of(vec![0; env.sizes[d]], vec![env.sizes[d]], elem)),
            Op::Loop { dim, index, states, arity, from1 } => {
                let mut state = self.at(kid(0)?, env)?;
                let n = env.sizes[dim];
                let trips = if from1 { n.saturating_sub(1) } else { n };
                let slots = &states[..arity];
                let outer_index = env.vars[index].take();
                let outer: Vec<Option<Val>> = slots.iter().map(|s| env.vars[*s].take()).collect();
                for t in 0..trips {
                    env.vars[index] = Some(Val::I((t + usize::from(from1)) as i64));
                    match (arity, state) {
                        // A tuple state is destructured into its components' variables.
                        (2..=3, Val::Tup(parts)) if parts.len() == arity => {
                            for (s, p) in slots.iter().zip(parts) {
                                env.vars[*s] = Some(p);
                            }
                        }
                        (2..=3, other) => return Err(format!("our software has a bug: a tuple loop's state is {other:?}")),
                        (_, s) => env.vars[slots[0]] = Some(s),
                    }
                    state = self.at(kid(1)?, env)?;
                }
                env.vars[index] = outer_index;
                for (s, v) in slots.iter().zip(outer) {
                    env.vars[*s] = v;
                }
                state
            }
            Op::RowFn(f) => Val::F(f),
            Op::Row2Fn(f) => Val::F2(f),
            Op::Transpose => {
                let m = self.a(kid(0)?, env)?;
                let [rows, cols] = m.dims[..] else { return Err(format!("our software has a bug: a transpose of {:?}", m.dims)) };
                let src = m.slice();
                let mut out = vec![0u32; rows * cols];
                for i in 0..rows {
                    for j in 0..cols {
                        out[j * rows + i] = src[i * cols + j];
                    }
                }
                Val::A(View::of(out, vec![cols, rows], m.elem))
            }
            Op::Matmul => {
                // The helper's arithmetic (`crate::futhark_interpreter::matmul`): `acc + a * b` in f32 from 0.0, a
                // multiply then an add, in order; the compiled program is a left fold too (measured bit-identical).
                let a = self.a(kid(0)?, env)?;
                let b = self.a(kid(1)?, env)?;
                let ([n, m], [m2, p]) = (&a.dims[..], &b.dims[..]) else { return Err(format!("our software has a bug: a matmul of {:?} and {:?}", a.dims, b.dims)) };
                if m != m2 {
                    return Err(format!("our software has a bug: a matmul whose inner dimensions are {m} and {m2}"));
                }
                let (n, m, p) = (*n, *m, *p);
                let (sa, sb) = (a.slice(), b.slice());
                let mut out = Vec::with_capacity(n * p);
                for i in 0..n {
                    for j in 0..p {
                        let mut acc = 0.0f32;
                        for kk in 0..m {
                            acc += f32::from_bits(sa[i * m + kk]) * f32::from_bits(sb[kk * p + j]);
                        }
                        out.push(acc.to_bits());
                    }
                }
                Val::A(View::of(out, vec![n, p], Elem::F32))
            }
            Op::MapRows => {
                let f = match self.at(kid(0)?, env)? {
                    Val::F(f) => f,
                    other => return Err(format!("our software has a bug: a row function slot holds {other:?}")),
                };
                let m = self.a(kid(1)?, env)?;
                let [rows, cols] = m.dims[..] else { return Err(format!("our software has a bug: a map over the rows of {:?}", m.dims)) };
                let src = m.slice();
                let mut out = Vec::with_capacity(rows * cols);
                for i in 0..rows {
                    let row: Vec<f32> = src[i * cols..(i + 1) * cols].iter().map(|w| f32::from_bits(*w)).collect();
                    out.extend(f.apply(&row).into_iter().map(f32::to_bits));
                }
                Val::A(View::of(out, vec![rows, cols], Elem::F32))
            }
            Op::Map2Rows => {
                let f = match self.at(kid(0)?, env)? {
                    Val::F2(f) => f,
                    other => return Err(format!("our software has a bug: a binary row function slot holds {other:?}")),
                };
                let a = self.a(kid(1)?, env)?;
                let b = self.a(kid(2)?, env)?;
                if a.dims != b.dims || a.dims.len() != 2 {
                    return Err(format!("our software has a bug: a map2 over the rows of {:?} and {:?}", a.dims, b.dims));
                }
                let (rows, cols) = (a.dims[0], a.dims[1]);
                let (sa, sb) = (a.slice(), b.slice());
                let mut out = Vec::with_capacity(rows * cols);
                for i in 0..rows {
                    let x: Vec<f32> = sa[i * cols..(i + 1) * cols].iter().map(|w| f32::from_bits(*w)).collect();
                    let y: Vec<f32> = sb[i * cols..(i + 1) * cols].iter().map(|w| f32::from_bits(*w)).collect();
                    out.extend(f.apply(&x, &y)?.into_iter().map(f32::to_bits));
                }
                Val::A(View::of(out, vec![rows, cols], Elem::F32))
            }
        })
    }

    /// The node executions of a tree on given sizes, the loops' state traffic included ([`Work::total`] of
    /// [`Interp::work_parts`]): the timing bound's measure at the timing input's sizes.
    pub fn work(&self, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<u128, String> {
        self.work_parts(tree, sizes).map(|w| w.total())
    }

    /// The work of a tree on given sizes, in two parts. [`Work::nodes`]: every node once, times the trips of the
    /// binders it is inside (a binder's body is its last argument); a reduction, a `map2` and a `replicate` also cost
    /// their array's length, a histogram of rows and a division of rows their `[K][D]` cells. A pair row costs one, as
    /// a scalar's does. [`Work::state`]: a LOOP whose state holds arrays costs their cells on every trip, the state
    /// written each trip (a tuple's arrays summed, a scalar nothing): the G3 GPU check of 2026-11-04 found histogram
    /// mutants `(loop1.k X (loop.k (hist.add.n vs zeros) zeros))`, whose `[n]` state is rewritten k x k times, counted
    /// at 1 a trip (3.2e9, under the 5.9e10 bound) and timed for minutes; with the state's cells they are 1.6e11.
    /// Two loops write no state (the rule of the F3a replay, 2026-11-04): one Futhark's simplifier removes, its body
    /// the loop's own state or its start (`(loop.n vs vs)`: Futhark's loop-invariant merge variables), and a
    /// component whose body is an operation that writes a fresh array and is charged its cells already (a `map2`, a
    /// scan, a histogram, a tabulate, a `replicate`, the matrix operations), which writes the state in place. A body
    /// that is a variable, an input, `zeros`, a row, a selection or another loop is copied into the state each trip.
    pub fn work_parts(&self, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<Work, String> {
        let n: Vec<u128> = self.sig.dims.iter().map(|d| sizes.get(d).map(|&x| x as u128).ok_or_else(|| format!("no size for {d}"))).collect::<Result<_, _>>()?;
        fn walk(me: &Interp5, t: &Tree, mult: u128, n: &[u128], w: &mut Work) -> Result<(), String> {
            let op = me.ops[t.symbol as usize];
            // The cells of a node's matrix class (`mat.X.Y`: the product of its roles' sizes).
            let cells = |node: &Tree| -> Result<u128, String> {
                let d = me.k.dual(me.k.rows[node.symbol as usize].output).ok_or("a row with no dual")?;
                d.form.split('.').map(|r| me.sig.dims.iter().position(|x| x == r).map(|i| n[i]).ok_or_else(|| format!("the role {r} has no size"))).product()
            };
            let (body_mult, own) = match op {
                Op::Tab(d, ..) | Op::TabRows(d, _) => (n[d], 1),
                Op::Loop { dim, from1, .. } => (n[dim].saturating_sub(u128::from(from1)), 1),
                Op::Replicate(d, _) | Op::Hist(_, d, _) => (1, 1 + n[d]),
                Op::HistRows(kd, dd, _) => (1, 1 + n[kd] * n[dd]),
                // The matrix family: a traversal costs its output's cells; the product its n x m x p multiply-adds (the
                // output's cells times the inner role, the first argument's second role).
                Op::Transpose | Op::MapRows | Op::Map2Rows => (1, 1 + cells(t)?),
                Op::Matmul => {
                    let a = t.kids.first().ok_or("a matmul with no first argument")?;
                    let inner = me.k.dual(me.k.rows[a.symbol as usize].output).ok_or("a row with no dual")?.form.split('.').nth(1).ok_or("a matmul of a non-matrix")?.to_string();
                    let m = me.sig.dims.iter().position(|x| *x == inner).map(|i| n[i]).ok_or_else(|| format!("the role {inner} has no size"))?;
                    (1, 1 + cells(t)? * m)
                }
                _ => (1, 1),
            };
            w.nodes += mult * own;
            if let Op::Loop { dim, index, states, arity, from1 } = op {
                w.state += mult * n[dim].saturating_sub(u128::from(from1)) * state_cells(me, t, &states[..arity], index, n)?;
            }
            // The array reduced, or the rows histogrammed (the second argument) or divided, costs its class's cells.
            let costed = match op {
                Op::Sum | Op::Minimum | Op::Maximum | Op::Map2(_) | Op::Scan(_) | Op::Hist(..) | Op::RowDiv => t.kids.first(),
                Op::HistRows(..) => t.kids.get(1),
                _ => None,
            };
            if let Some(kid) = costed {
                let dual = &me.k.dual(me.k.rows[kid.symbol as usize].output).ok_or("a row with no dual")?.dual;
                let class = Class::parse(dual.split(':').next().unwrap_or(""), &me.sig.elements, None)?;
                if class.dims().is_empty() {
                    return Err("a reduction of a non-array".into());
                }
                let mut cells = 1u128;
                for d in class.dims() {
                    cells *= n[me.sig.dims.iter().position(|x| x == d).ok_or("a reduction over an unknown dimension")?];
                }
                w.nodes += mult * cells;
            }
            let last = t.kids.len().saturating_sub(1);
            for (i, kid) in t.kids.iter().enumerate() {
                let binder = matches!(op, Op::Tab(..) | Op::TabRows(..) | Op::Loop { .. }) && i == last;
                walk(me, kid, if binder { mult * body_mult } else { mult }, n, w)?;
            }
            Ok(())
        }
        // Whether a subtree reads one of the slots (a loop's state or index).
        fn reads(me: &Interp5, t: &Tree, slots: &[usize]) -> bool {
            matches!(me.ops[t.symbol as usize], Op::Var(s) if slots.contains(&s)) || t.kids.iter().any(|k| reads(me, k, slots))
        }
        // Whether two subtrees are one Futhark expression: their rows' spellings (a row of one spelling may be several
        // rows, one a context), their constants and their children.
        fn same(me: &Interp5, a: &Tree, b: &Tree) -> bool {
            me.k.rows[a.symbol as usize].spelling == me.k.rows[b.symbol as usize].spelling && a.const_index == b.const_index && a.kids.len() == b.kids.len() && a.kids.iter().zip(&b.kids).all(|(x, y)| same(me, x, y))
        }
        // An operation that writes a fresh array and is charged its cells already: it writes a loop's state in place.
        fn writes_fresh(op: Op) -> bool {
            matches!(op, Op::Map2(_) | Op::Scan(_) | Op::Hist(..) | Op::HistRows(..) | Op::Replicate(..) | Op::Tab(..) | Op::TabRows(..) | Op::RowDiv | Op::Transpose | Op::Matmul | Op::MapRows | Op::Map2Rows)
        }
        // The cells a loop's state is written a trip: its arrays' (a tuple's summed, component by component when the
        // body is a tuple row), 0 for a scalar or an index, for a body Futhark removes the loop of, and for a component
        // an operation writes fresh.
        fn state_cells(me: &Interp5, t: &Tree, own: &[usize], index: usize, n: &[u128]) -> Result<u128, String> {
            let (start, body) = match &t.kids[..] {
                [start, body] => (start, body),
                _ => return Err("our software has a bug: a loop without a start and a body".into()),
            };
            let mut bound = own.to_vec();
            bound.push(index);
            let own_state = own.len() == 1 && matches!(me.ops[body.symbol as usize], Op::Var(s) if s == own[0]);
            if own_state || (same(me, body, start) && !reads(me, body, &bound)) {
                return Ok(0);
            }
            let dual = &me.k.dual(me.k.rows[t.symbol as usize].output).ok_or("a row with no dual")?.dual;
            fn cells(c: &Class, body: Option<&Tree>, me: &Interp5, n: &[u128]) -> Result<u128, String> {
                if body.is_some_and(|b| writes_fresh(me.ops[b.symbol as usize])) {
                    return Ok(0);
                }
                match c {
                    Class::Tuple(parts) => {
                        let kids = body.filter(|b| matches!(me.ops[b.symbol as usize], Op::Tuple) && b.kids.len() == parts.len());
                        parts.iter().enumerate().map(|(i, p)| cells(p, kids.map(|b| &b.kids[i]), me, n)).sum()
                    }
                    Class::Array { dims, .. } => dims.iter().map(|d| me.sig.dims.iter().position(|x| x == d).map(|i| n[i]).ok_or_else(|| format!("a loop's state over an unknown dimension {d}"))).product(),
                    _ => Ok(0),
                }
            }
            cells(&Class::parse(dual.split(':').next().unwrap_or(""), &me.sig.elements, None)?, Some(body), me, n)
        }
        let mut w = Work::default();
        walk(self, tree, 1, &n, &mut w)?;
        Ok(w)
    }
}

// ---- the builder ------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

fn parse_sexp(text: &str) -> Result<Sexp, String> {
    let spaced = text.replace('(', " ( ").replace(')', " ) ");
    let tokens: Vec<&str> = spaced.split_whitespace().collect();
    let mut pos = 0;
    let out = parse_at(&tokens, &mut pos)?;
    if pos != tokens.len() {
        return Err(format!("text after the expression: {:?}", &tokens[pos..tokens.len().min(pos + 5)]));
    }
    Ok(out)
}

fn parse_at(tokens: &[&str], pos: &mut usize) -> Result<Sexp, String> {
    let t = *tokens.get(*pos).ok_or("the expression ends early")?;
    *pos += 1;
    match t {
        "(" => {
            let mut items = Vec::new();
            loop {
                match tokens.get(*pos) {
                    None => return Err("an unclosed (".into()),
                    Some(&")") => {
                        *pos += 1;
                        return Ok(Sexp::List(items));
                    }
                    Some(_) => items.push(parse_at(tokens, pos)?),
                }
            }
        }
        ")" => Err("an unopened )".into()),
        _ => Ok(Sexp::Atom(t.to_string())),
    }
}

/// A program written as an s-expression of the family's rows, as a tree of the root type. A head `op` in a slot of
/// dual `T` is the row `op@T` (its arguments' duals are the row's own), an atom is the terminal `atom@T`, and an
/// integer is the literal of `T`'s pool. A construct with no row is an error that names it, never a substitute.
pub fn build(k: &FutharkKingdom, text: &str) -> Result<Tree, String> {
    let root = *k.roots().first().ok_or("the kingdom has no root")?;
    let by_name: HashMap<&str, u32> = k.rows.iter().enumerate().map(|(i, r)| (r.name.as_str(), i as u32)).collect();
    build_at(k, &by_name, &parse_sexp(text)?, root)
}

fn build_at(k: &FutharkKingdom, by_name: &HashMap<&str, u32>, e: &Sexp, ty: Ty) -> Result<Tree, String> {
    let dual = k.dual(ty).ok_or("a slot of no dual")?;
    // `op@T` for a slot of dual `T`; else `op` is a row's FULL name whose output is `T` (the matrix family names a
    // row by its arguments or its input class: `matmul@mat.n.mxmat.m.p`, `transpose@mat.n.m`, `mat.n.m.input`).
    let row = |op: &str| -> Result<u32, String> {
        if let Some(s) = by_name.get(format!("{op}@{}", dual.dual).as_str()) {
            return Ok(*s);
        }
        match by_name.get(op) {
            Some(s) if k.rows[*s as usize].output == ty => Ok(*s),
            Some(s) => Err(format!("the row {op} is of {}, not {}", k.dual(k.rows[*s as usize].output).map_or("?", |d| d.dual.as_str()), dual.dual)),
            None => Err(format!("the kingdom has no row {op}@{}", dual.dual)),
        }
    };
    match e {
        Sexp::Atom(a) => {
            // A literal: a number (an integer or, in an f32 pool, a decimal), matched by value against the pool.
            let numeric = a.strip_prefix('-').unwrap_or(a).chars().next().is_some_and(|c| c.is_ascii_digit());
            if let Some(v) = numeric.then(|| a.parse::<f64>().ok()).flatten() {
                let i = dual.literals.iter().position(|l| l.trim_end_matches(dual.slot.as_str()).parse::<f64>() == Ok(v)).ok_or_else(|| format!("the literal {a} is not in the pool of {}", dual.dual))?;
                let sym = by_name.get(format!("?_{}", dual.dual).as_str()).copied().ok_or_else(|| format!("{} has no constant", dual.dual))?;
                return Ok(Tree { symbol: sym, const_index: Some(i as u32), kids: vec![] });
            }
            Ok(Tree { symbol: row(a)?, const_index: None, kids: vec![] })
        }
        Sexp::List(items) => {
            let Some(Sexp::Atom(head)) = items.first() else { return Err(format!("a list with no head in {}", dual.dual)) };
            let sym = row(head)?;
            let inputs = k.rows[sym as usize].inputs.clone();
            if inputs.len() != items.len() - 1 {
                return Err(format!("{head}@{} takes {} arguments, given {}", dual.dual, inputs.len(), items.len() - 1));
            }
            let kids = items[1..].iter().zip(inputs).map(|(a, t)| build_at(k, by_name, a, t)).collect::<Result<Vec<_>, _>>()?;
            Ok(Tree { symbol: sym, const_index: None, kids })
        }
    }
}

/// THE STACK-NAMED READER: a program written in the variable names of the stack contexts (before 2026-11-02: a
/// variable named by its binder's frame number, outermost 1 — a tabulate's `j1`, a loop's `i2` and `s2`), as the
/// pools, fixtures, reports and halls of fame of that time hold them, read as a tree of this kingdom at the dual
/// `dual`. Each old name is the variable of its frame's class, now named by its class and its ordinal among the
/// frames of that class around it ([`crate::futhark_view::var_name`]), so every old program has its new name.
pub fn build_stack_named(k: &FutharkKingdom, text: &str, dual: &str) -> Result<Tree, String> {
    let ty = k.ty(dual).ok_or_else(|| format!("the kingdom has no type {dual}"))?;
    let by_name: HashMap<&str, u32> = k.rows.iter().enumerate().map(|(i, r)| (r.name.as_str(), i as u32)).collect();
    let mut frames: Vec<Vec<(String, String)>> = Vec::new();
    stack_named_at(k, &by_name, &parse_sexp(text)?, ty, &mut frames)
}

fn stack_named_at(k: &FutharkKingdom, by_name: &HashMap<&str, u32>, e: &Sexp, ty: Ty, frames: &mut Vec<Vec<(String, String)>>) -> Result<Tree, String> {
    let dual = k.dual(ty).ok_or("a slot of no dual")?;
    match e {
        Sexp::Atom(a) => {
            let old = a.strip_prefix(['i', 'j', 's']).and_then(|n| n.parse::<usize>().ok()).filter(|n| *n >= 1 && *n <= frames.len());
            let Some(kk) = old else { return build_at(k, by_name, e, ty) };
            let cls = frames[kk - 1].iter().find(|(n, _)| n == a).map(|(_, c)| c.clone()).ok_or_else(|| format!("{a} is not bound by frame {kk}"))?;
            // Its ordinal: the frames up to its own that bind its class.
            let ord = frames[..kk].iter().filter(|f| f.iter().any(|(_, c)| *c == cls)).count() as u32;
            build_at(k, by_name, &Sexp::Atom(var_name(&cls, ord)), ty)
        }
        Sexp::List(items) => {
            let Some(Sexp::Atom(head)) = items.first() else { return Err(format!("a list with no head in {}", dual.dual)) };
            let sym = by_name.get(format!("{head}@{}", dual.dual).as_str()).copied().ok_or_else(|| format!("the kingdom has no row {head}@{}", dual.dual))?;
            let inputs = k.rows[sym as usize].inputs.clone();
            if inputs.len() != items.len() - 1 {
                return Err(format!("{head}@{} takes {} arguments, given {}", dual.dual, inputs.len(), items.len() - 1));
            }
            let cls = dual.dual.split(':').next().unwrap_or("").to_string();
            let kk = frames.len() + 1;
            let frame: Option<Vec<(String, String)>> = if head == "tab" {
                Some(vec![(format!("j{kk}"), format!("ix.{}", cls.strip_prefix("a.").ok_or("a tabulate of a non-array")?))])
            } else if let Some(d) = head.strip_prefix("loop1.").or_else(|| head.strip_prefix("loop.")) {
                Some(vec![(format!("s{kk}"), cls.clone()), (format!("i{kk}"), format!("ix.{d}"))])
            } else {
                None
            };
            let last = inputs.len().saturating_sub(1);
            let mut kids = Vec::with_capacity(inputs.len());
            for (i, (a, t)) in items[1..].iter().zip(inputs).enumerate() {
                match (&frame, i == last) {
                    (Some(f), true) => {
                        frames.push(f.clone());
                        let r = stack_named_at(k, by_name, a, t, frames);
                        frames.pop();
                        kids.push(r?);
                    }
                    _ => kids.push(stack_named_at(k, by_name, a, t, frames)?),
                }
            }
            Ok(Tree { symbol: sym, const_index: None, kids })
        }
    }
}

/// A tree as the builder's s-expression (the inverse of [`build`]): for the hall of fame and the reports.
pub fn sexp(k: &FutharkKingdom, t: &Tree) -> String {
    let row = &k.rows[t.symbol as usize];
    // The head is `op` when the row is `op@<its output's dual>` (every row of a signature view), else the row's full
    // name (the matrix family), which [`build`] reads back.
    let out = k.dual(row.output).map_or("", |d| d.dual.as_str());
    let op = match row.name.split_once('@') {
        Some((o, d)) if d == out => o,
        _ => row.name.as_str(),
    };
    if row.kind == FutharkKind::Const {
        let lit = k.dual(row.output).and_then(|d| t.const_index.and_then(|i| d.literals.get(i as usize)).map(|l| l.trim_end_matches(d.slot.as_str()).to_string()));
        return lit.unwrap_or_else(|| "?".into());
    }
    if t.kids.is_empty() {
        return op.to_string();
    }
    format!("({op} {})", t.kids.iter().map(|c| sexp(k, c)).collect::<Vec<_>>().join(" "))
}

// ---- random genes -----------------------------------------------------------------------------------------------

/// A random gene of the layout: each head cell a function with probability one half and a terminal otherwise (each
/// uniform over its kind; a cell whose alphabet has one kind draws from it), tail, Dc and Dv cells uniform; the root cell
/// is a function of the root type (phylu's `root_function`: a root that is a terminal is a constant program).
pub fn random_gene(contract: &KingdomContract, layout: &GeneLayout, const_len: &BTreeMap<Ty, u32>, rng: &mut GeneRng) -> Result<Vec<u32>, String> {
    let mut gene = vec![0u32; layout.width];
    for span in layout.spans() {
        let a = layout.alphabet(contract, span.off, const_len);
        if a.is_empty() {
            return Err(format!("our software has a bug: the span at {} has an empty alphabet", span.off));
        }
        // Only a head cell holds a symbol to split by kind: a Dc cell's words are pool indices, a Dv cell's index words.
        let (f, t): (Vec<u32>, Vec<u32>) = if span.domain == Domain::Head { a.iter().partition(|s| !contract.is_terminal(**s)) } else { (Vec::new(), Vec::new()) };
        let split = !f.is_empty() && !t.is_empty();
        for cell in &mut gene[span.off..span.off + span.len] {
            *cell = if split {
                let pick = if rng.below(2) == 0 { &f } else { &t };
                pick[rng.below(pick.len())]
            } else {
                a[rng.below(a.len())]
            };
        }
    }
    let pos = layout.frame(layout.root).and_then(|f| f.row_pos(0)).ok_or("the root has no first row")?;
    let a = layout.alphabet(contract, pos, const_len);
    let f: Vec<u32> = a.into_iter().filter(|s| !contract.is_terminal(*s)).collect();
    if !f.is_empty() {
        gene[pos] = f[rng.below(f.len())];
    }
    Ok(gene)
}

// ---- the stored cases and the executables -------------------------------------------------------------------------

/// Which part of the fitness a case is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    /// Held out from the training sizes; a fitness input all the same (Andrew's rule: the quality group holds the
    /// held-out cases, its validation column).
    Held,
    /// THE TEST SET: never a fitness input, never seen by selection; an exactness claim is made on it (fault 11 of
    /// `docs/reviews/2026-10-28_futhark_evolve_faults.md`).
    Test,
}

impl Split {
    /// Whether the case is a fitness input (training or held-out).
    pub fn in_fitness(self) -> bool {
        self != Split::Test
    }
}

/// One stored case: the inputs and, for a problem scored against answers, the published program's output on them.
#[derive(Debug, Clone)]
pub struct Case {
    pub name: String,
    pub split: Split,
    pub inputs: Vec<Array>,
    /// THE STORED ANSWER (`<name>.out`): the published program's output, for a problem scored against it
    /// ([`Problem::cases_have_answers`]); `None` for a problem scored on its own objective (k-means: the fitness runs
    /// the program to UltraDim's stopping rule and measures the clustering, `docs/KMEANS_FITNESS.md`). Read through
    /// [`Case::answer`] where an answer is required.
    pub output: Option<Array>,
    /// The inputs in Futhark's binary format, as the executable reads them.
    pub stdin: Vec<u8>,
}

impl Case {
    /// The stored answer, or the plain statement that this case has none.
    pub fn answer(&self) -> Result<&Array, String> {
        self.output.as_ref().ok_or_else(|| format!("the case {} has no stored answer: its problem is scored on its own objective (docs/KMEANS_FITNESS.md), not against an output", self.name))
    }
}

/// The case directory of a problem.
pub fn corpus_dir(p: &Problem) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(p.corpus)
}

/// The stored cases (`cases.tsv`: name, split, then `<name>.in` and, for a problem scored against answers, `<name>.out`
/// in Futhark's binary format; a problem scored on its objective has no `.out`, and one present is an error).
pub fn load_cases(p: &Problem) -> Result<Vec<Case>, String> {
    let answers = p.cases_have_answers();
    let dir = corpus_dir(p);
    let index = std::fs::read_to_string(dir.join("cases.tsv")).map_err(|e| format!("{}: {e}", dir.join("cases.tsv").display()))?;
    let mut out = Vec::new();
    for line in index.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        let [name, split, ..] = f.as_slice() else { return Err(format!("the case line {line:?}")) };
        let split = match *split {
            "train" => Split::Train,
            "held" => Split::Held,
            "test" => Split::Test,
            other => return Err(format!("the case {name} has the split {other}")),
        };
        let stdin = std::fs::read(dir.join(format!("{name}.in"))).map_err(|e| format!("{name}.in: {e}"))?;
        let inputs = read_binary(&stdin)?;
        let out_path = dir.join(format!("{name}.out"));
        let output = if answers {
            Some(read_one(&std::fs::read(&out_path).map_err(|e| format!("{name}.out: {e}"))?)?)
        } else if out_path.exists() {
            return Err(format!("{}: {} is scored on its own objective and stores no answers, but {name}.out exists", out_path.display(), p.name));
        } else {
            None
        };
        out.push(Case { name: name.to_string(), split, inputs, output, stdin });
    }
    Ok(out)
}

/// THE VALUES FINGERPRINT of a tree on a problem's stored cases: FNV-1a (64 bits) over, per case in order, its
/// name, then the interpreter's output (its dims and data, little-endian) or the error text. Two trees with the same
/// fingerprint compute the same value on every stored case (to the hash); it pins a pool program's meaning across a
/// change of the kingdom's names (`docs/PLAN_futhark_kingdom.md` 3g).
pub fn values_fingerprint(interp: &Interp, tree: &Tree, cases: &[Case]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for c in cases {
        eat(c.name.as_bytes());
        eat(&[0]);
        match interp.eval(tree, &c.inputs) {
            Ok(a) => {
                eat(b"ok");
                for d in &a.dims {
                    eat(&(*d as u64).to_le_bytes());
                }
                for x in &a.data {
                    eat(&x.to_le_bytes());
                }
            }
            Err(e) => {
                eat(b"error");
                eat(e.as_bytes());
            }
        }
        eat(&[0]);
    }
    h
}

/// Compile a program with Futhark's sequential C backend (`futhark c`, the released compiler) into `dir`.
pub fn compile_c(source: &str, dir: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let fut = dir.join("prog.fut");
    std::fs::write(&fut, source).map_err(|e| format!("write {}: {e}", fut.display()))?;
    let exe = dir.join("prog_c");
    let out = crate::futhark_exec::output_with_timeout(
        std::process::Command::new("futhark").arg("c").arg(&fut).arg("-o").arg(&exe),
        std::time::Duration::from_secs(120),
        None,
    )?;
    if !out.status.success() {
        return Err(format!("futhark c failed:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(exe)
}

/// Run a compiled executable (either backend's) on a case's binary input and read its one output value.
pub fn run_exe(exe: &Path, stdin: &[u8]) -> Result<Array, String> {
    let out = crate::futhark_exec::output_with_timeout_bytes(std::process::Command::new(exe).arg("-b"), std::time::Duration::from_secs(120), Some(stdin))?;
    if !out.status.success() {
        return Err(format!("{} failed:\n{}", exe.display(), String::from_utf8_lossy(&out.stderr)));
    }
    read_one(&out.stdout)
}

/// What a program's output is against the expected one, each on `[0, 1]`, 0 exact: the share of elements that
/// differ; the mean absolute error over the expected values' mean magnitude, and the largest absolute error over
/// their largest magnitude, each mapped `e -> e / (1 + e)`. An output of another shape differs everywhere (1, 1, 1).
pub fn compare(got: &Array, want: &Array) -> [f64; 3] {
    if got.dims != want.dims || got.elem != want.elem {
        return [1.0; 3];
    }
    let n = want.len().max(1) as f64;
    let (mut differ, mut sum, mut max) = (0usize, 0f64, 0f64);
    for i in 0..want.len() {
        let d = gap(got, want, i);
        differ += usize::from(!got.same_at(want, i));
        sum += d;
        max = max.max(d);
    }
    let mean_mag = (want.values().map(f64::abs).sum::<f64>() / n).max(1.0);
    let max_mag = want.values().map(f64::abs).fold(1.0, f64::max);
    [differ as f64 / n, squash(sum / n / mean_mag), squash(max / max_mag)]
}

/// A cell's error: 0 when the two values agree under the comparison policy (the same bits, or two NaNs), else the
/// absolute difference of the values (0 for the two zeros, which differ but are no distance apart); a difference
/// that is not a number (one NaN, two infinities of one sign) is infinite. Exact for `i32` (as before the values
/// were typed), where a cell differs exactly when its error is above 0.
fn gap(got: &Array, want: &Array, i: usize) -> f64 {
    if got.same_at(want, i) {
        return 0.0;
    }
    let d = (got.value(i) - want.value(i)).abs();
    if d.is_nan() {
        f64::INFINITY
    } else {
        d
    }
}

/// `e -> e / (1 + e)` on `[0, 1]`; an infinite error is 1.
fn squash(e: f64) -> f64 {
    if e.is_finite() {
        e / (1.0 + e)
    } else {
        1.0
    }
}

/// The sign of a step between two cells (up, down, flat); a step that is not a number is flat.
fn step_sign(d: f64) -> i8 {
    if d > 0.0 {
        1
    } else if d < 0.0 {
        -1
    } else {
        0
    }
}

/// One kind of quality input ([`QUALITY_PARTS`] lists the ones the fitness uses, in order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityPart {
    /// One input a case: the share of cells not exactly right (the one exactness input).
    Wrong,
    /// One input a cell: `ln(1 + |d|) / ln(1 + M)` (at most 1), `d` the cell's error and `M` the largest magnitude of
    /// the correct output (at least 1). Graded, no threshold, and an error's size counts relative to the answer's
    /// magnitude, never to its range (fault 3).
    CellLog,
    /// One input a cell: `e / (1 + e)` with `e = |d| / m`, `m` the mean magnitude of the correct output (at least 1).
    CellLevel,
    /// One input a neighbouring pair of cells: 1 if the step between them has the wrong sign (up, down, flat) against
    /// the correct output's, else 0. Structure, not level: it credits a program whose output rises and falls where the
    /// answer does (fault 3). Absent for a one-cell output.
    Step,
    /// Two inputs a case whose correct output has at least two distinct values: `(1 - r) / 2` for Spearman's and for
    /// Pearson's correlation of the output with the correct one (a constant output has no shape: 0.5 each).
    Shape,
    /// One input a case: the share of neighbouring pairs of cells whose step has the wrong sign (up, down, flat)
    /// against the correct output's; 0 for a one-cell output; absent, as the shape inputs are, when the correct output
    /// is constant over two or more cells (it has no order to agree with, and "is the output flat" would only repeat
    /// the exactness share).
    Order,
    /// One input a case: the share of cells off by over this percentage of the correct value's magnitude (at least
    /// 1).
    Outside(u8),
    /// The study's seven inputs a case (2026-10-27): the share of cells not exact; the shares off by over 1 %, 10 %
    /// and 50 % of the correct value's magnitude; the mean and largest error over the correct output's range, each
    /// `e / (1 + e)`; the share of neighbouring pairs in the wrong order (0 for one cell).
    Study,
}

/// THE QUALITY INPUTS' FORMULA: the parts each case contributes, in order (plan section B, chosen 2026-10-29 by the
/// slope check on the GPU's outputs; `docs/PLAN_futhark_fitness_fixes.md` and
/// `docs/measurements/2026-10-29_futhark_slope_check.tsv` record every formula tried). The exactness share, the two
/// shape inputs and the shares off by over 50 % and 1 %. Of every combination of the measured parts it is the one that
/// passes the check on BOTH benchmarks (histogram: the median neighbour beats 97.4 % of random genes, Spearman
/// +0.079; pathfinder: 99.0 %, +0.402), does not reward the all-zeros output (68.6 % and 84.1 % of the non-zero
/// wrong random outputs score better than it; under the study's seven inputs 28.8 % on histogram), and keeps the
/// partial-correctness test's margins on every pathfinder case (`tests::the_quality_inputs_give_partial_correctness_a_slope`).
/// Graded per-cell inputs were measured and lose: histogram's whole-range cases give hundreds of cells of unrelated
/// large numbers that drown the exactness signal, and on pathfinder a level-matching constant beats the
/// near-neighbours.
pub const QUALITY_PARTS: &[QualityPart] = &[QualityPart::Wrong, QualityPart::Shape, QualityPart::Outside(50), QualityPart::Outside(1)];

/// The names of [`quality_inputs`] against a correct output `want`, in order: their number depends on the case
/// (one input a cell or a step).
pub fn quality_input_names(want: &Array) -> Vec<String> {
    quality_input_names_of(QUALITY_PARTS, want)
}

/// [`quality_input_names`] for the parts `parts` (a fitness scheme's, `crate::fitness::Scheme::quality_parts`).
pub fn quality_input_names_of(parts: &[QualityPart], want: &Array) -> Vec<String> {
    let n = want.len();
    let distinct = want.data.iter().collect::<std::collections::BTreeSet<_>>().len();
    let mut out = Vec::new();
    for part in parts {
        match part {
            QualityPart::Wrong => out.push("wrong_share".to_string()),
            QualityPart::CellLog => out.extend((0..n).map(|i| format!("cell{i}_log_error"))),
            QualityPart::CellLevel => out.extend((0..n).map(|i| format!("cell{i}_level_error"))),
            QualityPart::Step => out.extend((1..n).map(|i| format!("step{i}_sign_wrong"))),
            QualityPart::Shape if distinct >= 2 => out.extend(["rank_disagreement".to_string(), "linear_disagreement".to_string()]),
            QualityPart::Shape => {}
            QualityPart::Order if n >= 2 && distinct < 2 => {}
            QualityPart::Order => out.push("order_wrong".to_string()),
            QualityPart::Outside(p) => out.push(format!("outside_{p}pct")),
            QualityPart::Study => out.extend(STUDY_INPUTS.iter().map(|s| s.to_string())),
        }
    }
    out
}

/// The names of [`QualityPart::Study`]'s inputs.
const STUDY_INPUTS: [&str; 7] = ["wrong_share", "outside_1pct", "outside_10pct", "outside_50pct", "mean_abs_over_range", "max_abs_over_range", "order_wrong"];

/// [`QualityPart::Study`]'s inputs, exactly the study's (checked against its ledgers).
fn study_inputs(got: &Array, want: &Array) -> [f64; 7] {
    let n = want.len().max(1) as f64;
    let lo = want.values().fold(f64::INFINITY, f64::min);
    let hi = want.values().fold(f64::NEG_INFINITY, f64::max);
    let range = if want.is_empty() { 0.0 } else { hi - lo };
    let mag = want.values().map(f64::abs).fold(0.0, f64::max);
    let scale = if range > 0.0 { range } else { mag }.max(1.0);
    let (mut wrong, mut o1, mut o10, mut o50, mut sum, mut max) = (0.0, 0.0, 0.0, 0.0, 0.0, 0f64);
    for i in 0..want.len() {
        let d = gap(got, want, i);
        let rel = d / want.value(i).abs().max(1.0);
        wrong += f64::from(u8::from(!got.same_at(want, i)));
        o1 += f64::from(u8::from(rel > 0.01));
        o10 += f64::from(u8::from(rel > 0.10));
        o50 += f64::from(u8::from(rel > 0.50));
        sum += d;
        max = max.max(d);
    }
    let steps = want.len().saturating_sub(1);
    let order = if steps == 0 {
        0.0
    } else {
        let sign = |a: &Array, j: usize| step_sign(a.value(j + 1) - a.value(j));
        (0..steps).filter(|&j| sign(got, j) != sign(want, j)).count() as f64 / steps as f64
    };
    [wrong / n, o1 / n, o10 / n, o50 / n, squash(sum / n / scale), squash(max / scale), order]
}

/// THE QUALITY INPUTS of one case, each its own input on `[0, 1]`, 0 exact, in the order of
/// [`quality_input_names`]: the parts of [`QUALITY_PARTS`]. Every cell and every step is its own input; nothing is
/// averaged before HFF but the one exactness share. An output of the wrong shape is 1 everywhere.
pub fn quality_inputs(got: &Array, want: &Array) -> Vec<f64> {
    quality_inputs_of(QUALITY_PARTS, got, want)
}

/// [`quality_inputs`] for the parts `parts` (a fitness scheme's, `crate::fitness::Scheme::quality_parts`).
pub fn quality_inputs_of(parts: &[QualityPart], got: &Array, want: &Array) -> Vec<f64> {
    let names = quality_input_names_of(parts, want).len();
    if got.dims != want.dims || got.elem != want.elem {
        return vec![1.0; names];
    }
    let n = want.len();
    let d: Vec<f64> = (0..n).map(|i| gap(got, want, i)).collect();
    let big = want.values().map(f64::abs).fold(1.0, f64::max);
    let mean = (want.values().map(f64::abs).sum::<f64>() / n.max(1) as f64).max(1.0);
    let distinct = want.data.iter().collect::<std::collections::BTreeSet<_>>().len();
    let mut out = Vec::with_capacity(names);
    for part in parts {
        match part {
            QualityPart::Wrong => out.push((0..n).filter(|i| !got.same_at(want, *i)).count() as f64 / n.max(1) as f64),
            QualityPart::CellLog => out.extend(d.iter().map(|x| (x.ln_1p() / big.ln_1p()).min(1.0))),
            QualityPart::CellLevel => out.extend(d.iter().map(|x| squash(x / mean))),
            QualityPart::Step => {
                let sign = |a: &Array, j: usize| step_sign(a.value(j + 1) - a.value(j));
                out.extend((0..n.saturating_sub(1)).map(|j| f64::from(u8::from(sign(got, j) != sign(want, j)))));
            }
            QualityPart::Shape if distinct >= 2 => {
                let g: Vec<f64> = got.values().collect();
                let w: Vec<f64> = want.values().collect();
                // A value that is not finite (an f32 NaN or infinity) has no rank or level to correlate: the answer
                // agrees with itself (0), any other output is the worst (1) on both.
                let finite = g.iter().chain(&w).all(|x| x.is_finite());
                for r in [crate::runner::spearman(&g, &w), crate::runner::pearson(&g, &w)] {
                    out.push(if finite { r.map_or(0.5, |r| (1.0 - r) / 2.0) } else if got == want { 0.0 } else { 1.0 });
                }
            }
            QualityPart::Shape => {}
            QualityPart::Order if n >= 2 && distinct < 2 => {}
            QualityPart::Order => out.push(study_inputs(got, want)[6]),
            QualityPart::Outside(p) => out.push((0..n).filter(|i| d[*i] / want.value(*i).abs().max(1.0) > f64::from(*p) / 100.0).count() as f64 / n.max(1) as f64),
            QualityPart::Study => out.extend(study_inputs(got, want)),
        }
    }
    out
}

// ---- the slope check's measures (docs/PLAN_futhark_fitness_fixes.md, section A) -----------------------------------

/// THE TREE EDIT DISTANCE between two expressed trees (Zhang and Shasha, 1989): the least number of node relabels,
/// deletions and insertions, each of cost 1, that turn one tree into the other. A node's label is its row and, for a
/// constant, its pool index. The distance of a program to the reference gene's tree is the slope check's distance.
pub fn tree_distance(a: &Tree, b: &Tree) -> usize {
    /// A tree in postorder: each node's label, its leftmost leaf's postorder index, and the keyroots (the highest
    /// node of each leftmost leaf), ascending.
    struct Post {
        labels: Vec<(u32, Option<u32>)>,
        leftmost: Vec<usize>,
        keyroots: Vec<usize>,
    }
    fn walk(t: &Tree, labels: &mut Vec<(u32, Option<u32>)>, leftmost: &mut Vec<usize>) -> usize {
        let mut first = None;
        for k in &t.kids {
            let i = walk(k, labels, leftmost);
            first.get_or_insert(leftmost[i]);
        }
        let me = labels.len();
        labels.push((t.symbol, t.const_index));
        leftmost.push(first.unwrap_or(me));
        me
    }
    fn post(t: &Tree) -> Post {
        let (mut labels, mut leftmost) = (Vec::new(), Vec::new());
        walk(t, &mut labels, &mut leftmost);
        let mut seen = std::collections::HashSet::new();
        let mut keyroots: Vec<usize> = (0..labels.len()).rev().filter(|i| seen.insert(leftmost[*i])).collect();
        keyroots.sort_unstable();
        Post { labels, leftmost, keyroots }
    }
    let (pa, pb) = (post(a), post(b));
    let (na, nb) = (pa.labels.len(), pb.labels.len());
    let mut tree = vec![vec![0usize; nb]; na];
    for &i in &pa.keyroots {
        for &j in &pb.keyroots {
            let (li, lj) = (pa.leftmost[i], pb.leftmost[j]);
            let (m, n) = (i - li + 2, j - lj + 2);
            let mut forest = vec![vec![0usize; n]; m];
            for x in 1..m {
                forest[x][0] = forest[x - 1][0] + 1;
            }
            for y in 1..n {
                forest[0][y] = forest[0][y - 1] + 1;
            }
            for x in 1..m {
                for y in 1..n {
                    let (u, v) = (li + x - 1, lj + y - 1);
                    let edit = (forest[x - 1][y] + 1).min(forest[x][y - 1] + 1);
                    if pa.leftmost[u] == li && pb.leftmost[v] == lj {
                        forest[x][y] = edit.min(forest[x - 1][y - 1] + usize::from(pa.labels[u] != pb.labels[v]));
                        tree[u][v] = forest[x][y];
                    } else {
                        forest[x][y] = edit.min(forest[pa.leftmost[u] - li][pb.leftmost[v] - lj] + tree[u][v]);
                    }
                }
            }
        }
    }
    tree[na - 1][nb - 1]
}

/// FNV-1a over bytes, as 16 hex digits.
fn fnv(parts: &[&[u8]]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in parts {
        for b in p.iter().chain(&[0xff]) {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

/// The identity of a problem's stored cases: every case's name, split, inputs and output.
pub fn cases_fingerprint(cases: &[Case]) -> String {
    let mut parts: Vec<Vec<u8>> = Vec::new();
    for c in cases {
        parts.push(format!("{}\t{:?}", c.name, c.split).into_bytes());
        parts.push(c.stdin.clone());
        parts.push(c.output.as_ref().map_or_else(|| b"no answer".to_vec(), |o| write_binary(&[o])));
    }
    fnv(&parts.iter().map(Vec::as_slice).collect::<Vec<_>>())
}

/// The identity of the quality inputs' formula: their values, exactly, on fixed outputs against fixed answers (a
/// level shift, a reversal, a constant, a shape mismatch, a near miss), so any change to the formula changes it.
pub fn quality_fingerprint() -> String {
    quality_fingerprint_of(QUALITY_PARTS)
}

/// [`quality_fingerprint`] of the parts `parts` (a fitness scheme's).
pub fn quality_fingerprint_of(parts: &[QualityPart]) -> String {
    let w: Vec<i32> = vec![3, -40, 900, 7, 7, 0, 123_456];
    let want = Array::i32s(vec![7], w.clone());
    let probes = [
        Array::i32s(vec![7], w.iter().map(|x| x + 5).collect()),
        Array::i32s(vec![7], w.iter().rev().copied().collect()),
        Array::i32s(vec![7], vec![0; 7]),
        Array::i32s(vec![6], vec![1; 6]),
        Array::i32s(vec![7], vec![3, -40, 901, 7, 7, 0, 123_456]),
    ];
    let text: String = probes.iter().flat_map(|g| quality_inputs_of(parts, g, &want)).map(|x| format!("{x:?},")).collect();
    fnv(&[text.as_bytes()])
}

// ---- the instance's e-graph -------------------------------------------------------------------------------------

/// The row <-> constructor mapping (`egraph.rows`).
#[derive(Debug, Clone)]
pub struct Bridge {
    ctor_of_row: BTreeMap<String, String>,
    row_of_ctor: BTreeMap<String, String>,
    /// The view's e-graph datatype and ruleset, the programs the mapping is of.
    pub datatype: &'static str,
    pub rules: &'static str,
    /// The name of the ruleset the rules are in ([`RULESET`] for a signature view; the matrix view's generated
    /// `futhark_shape`).
    pub ruleset: &'static str,
}

impl Bridge {
    /// Read an instance's mapping; a repeated row or constructor is an error (the generator's witness refuses one). A
    /// signature view's e-graph is the view's own; the matrix view's (A7) is the generator's
    /// (`kingdoms/futhark/egraph.*`, `crate::futhark_saturation`), written from the same view and kept there by
    /// `gen_tables.py --check`.
    pub fn load(inst: &Instance) -> Result<Bridge, String> {
        let view = inst.view()?;
        let (rows, datatype, rules, ruleset): (&'static str, &'static str, &'static str, &'static str) = match view.spec.shape {
            Shape::Signature => (&view.egraph_rows, &view.datatype, &view.rules, RULESET),
            Shape::Matrix => (crate::futhark_saturation::SHAPE_ROWS, crate::futhark_saturation::SHAPE_DATATYPE, crate::futhark_saturation::SHAPE_RULESET, crate::futhark_saturation::SHAPE_RULESET_NAME),
        };
        let mut ctor_of_row = BTreeMap::new();
        let mut row_of_ctor = BTreeMap::new();
        for line in rows.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 3 {
                return Err(format!("the mapping line {line:?} is malformed"));
            }
            if ctor_of_row.insert(f[0].to_string(), f[1].to_string()).is_some() || row_of_ctor.insert(f[1].to_string(), f[0].to_string()).is_some() {
                return Err(format!("the mapping line {line:?} repeats a row or a constructor"));
            }
        }
        Ok(Bridge { ctor_of_row, row_of_ctor, datatype, rules, ruleset })
    }
}

/// An expressed tree as an egglog term; a constant placeholder is its constructor applied to the pool index.
pub fn import(tree: &Tree, k: &FutharkKingdom, bridge: &Bridge) -> Result<String, String> {
    let row = k.rows.get(tree.symbol as usize).ok_or_else(|| format!("symbol {} is not a row", tree.symbol))?;
    let ctor = bridge.ctor_of_row.get(&row.name).ok_or_else(|| format!("the row {} has no constructor", row.name))?;
    if row.kind == FutharkKind::Const {
        return Ok(format!("({ctor} {})", tree.const_index.ok_or_else(|| format!("{} has no constant index", row.name))?));
    }
    let kids = tree.kids.iter().map(|t| import(t, k, bridge)).collect::<Result<Vec<_>, _>>()?;
    Ok(if kids.is_empty() { format!("({ctor})") } else { format!("({ctor} {})", kids.join(" ")) })
}

/// An egglog term as a tree of the kingdom's rows; `None` when it is not one.
pub fn to_tree(termdag: &TermDag, id: TermId, k: &FutharkKingdom, bridge: &Bridge, symbols: &HashMap<String, u32>) -> Option<Tree> {
    let Term::App(op, kids) = termdag.get(id) else { return None };
    let symbol = *symbols.get(bridge.row_of_ctor.get(op.as_str())?)?;
    if k.rows[symbol as usize].kind == FutharkKind::Const {
        let [kid] = kids.as_slice() else { return None };
        let Term::Lit(egglog::ast::Literal::Int(i)) = termdag.get(*kid) else { return None };
        return Some(Tree { symbol, const_index: Some(u32::try_from(*i).ok()?), kids: vec![] });
    }
    let kids = kids.iter().map(|t| to_tree(termdag, *t, k, bridge, symbols)).collect::<Option<Vec<_>>>()?;
    Some(Tree { symbol, const_index: None, kids })
}

fn term_cost(termdag: &TermDag, id: TermId) -> u64 {
    match termdag.get(id) {
        Term::App(_, kids) => 1 + kids.iter().map(|k| term_cost(termdag, *k)).sum::<u64>(),
        _ => 1,
    }
}

/// The e-graph variants of a tree: `iters` iterations of the instance's ruleset, the ranked members of the root's
/// class (HFF extraction, `members` a class), up to `n` distinct trees other than the tree itself, smallest first and
/// then by their rendered text, so the order is deterministic. Every rule is an identity of wrapping `i32`
/// arithmetic and of `min`/`max`, so every variant computes the same values.
pub fn variants(k: &FutharkKingdom, bridge: &Bridge, tree: &Tree, iters: u32, members: usize, n: usize) -> Result<Vec<Tree>, String> {
    let term = import(tree, k, bridge)?;
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, bridge.datatype).map_err(|e| format!("the datatype does not load: {e}"))?;
    egraph.parse_and_run_program(None, bridge.rules).map_err(|e| format!("the ruleset does not load: {e}"))?;
    egraph.parse_and_run_program(None, &format!("(let __root {term})\n(run-schedule (repeat {iters} (run {})))", bridge.ruleset)).map_err(|e| format!("the term does not load: {e}"))?;
    let (sort, value) = egraph.eval_expr(&exprs::var("__root")).map_err(|e| format!("the root does not evaluate: {e}"))?;
    let mut termdag = TermDag::default();
    let score = |td: &TermDag, t: TermId| term_cost(td, t) as f64;
    let ranked = hff_extract(&egraph, &mut termdag, value, sort, &score, members);
    let symbols: HashMap<String, u32> = k.rows.iter().enumerate().map(|(i, r)| (r.name.clone(), i as u32)).collect();
    let mut out: Vec<(u64, String, Tree)> = Vec::new();
    for (cost, id) in ranked {
        let Some(t) = to_tree(&termdag, id, k, bridge, &symbols) else { continue };
        if &t == tree {
            continue;
        }
        let text = k.render(&t)?;
        if out.iter().any(|(_, s, _)| s == &text) {
            continue;
        }
        out.push((cost as u64, text, t));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    out.truncate(n);
    Ok(out.into_iter().map(|(_, _, t)| t).collect())
}

/// The instance's e-graph as the saturated fold needs it: the bridge carries the view's datatype, rules and ruleset.
pub struct FoldEgraph<'a> {
    pub kingdom: &'a FutharkKingdom,
    pub bridge: &'a Bridge,
}

impl EgraphKingdom for FoldEgraph<'_> {
    fn programs(&self) -> Vec<&str> {
        vec![self.bridge.datatype, self.bridge.rules]
    }
    fn rulesets(&self) -> Vec<&str> {
        vec![self.bridge.ruleset]
    }
    fn import(&self, tree: &Tree) -> Result<String, String> {
        import(tree, self.kingdom, self.bridge)
    }
    fn row(&self, constructor: &str, literals: &[&str]) -> Option<(SymbolId, Option<u32>)> {
        let name = self.bridge.row_of_ctor.get(constructor)?;
        let symbol = self.kingdom.rows.iter().position(|r| &r.name == name)? as SymbolId;
        if self.kingdom.rows[symbol as usize].kind == FutharkKind::Const {
            let [text] = literals else { return None };
            return Some((symbol, Some(text.parse().ok()?)));
        }
        literals.is_empty().then_some((symbol, None))
    }
}

// ---- the reference programs as master 6 trees (plan section 7, task F2a) ------------------------------------------
//
// Every reference program (the four benchmarks' references and planted slowdowns, A7's reference, winner and planted
// slowdown, k-means's reference, planted slowdown, textbook form and its planted slowdown) written by hand as a tree
// of its master 6 view (`kingdoms/futhark/sig/<problem>/m6/`), beside its master 5 s-expression above. The reader
// reads no k-means program today, so the conversion is by hand. Each tree's expected text is its master 5 rendering
// ([`program`], unchanged), stored in `tests/fixtures/futhark_m6_references.txt` for task F2's renderer to reproduce
// byte for byte.
//
// A master 6 s-expression names each function by its view symbol (`tab@a.i32`) and each leaf as one of: a terminal
// symbol (`zero@i64`, `add@fn.arr2_f32_arr_f32_arr_f32`), a literal of the slot's data type's pool (`0`, a `const@T`
// cell whose Dc index is the literal's place in the pool), or a VARIABLE by name: a root variable of the view's
// environment (`points`, `n`) or a name a binder above gives the variables it binds, listed in Binding row order in
// brackets after the binder's symbol (`(tab@a.i32[i] n ...)`, `(loop1@a.i32[s,i] rows ...)`), the innermost binder
// first. A variable leaf becomes the `var.T` of the slot's data type, and its index word is the one G3's decoder says
// names that variable in the leaf's environment ([`crate::typed_environment::Environment::index_of`]): never a number
// computed by hand.

/// One reference program in both masters: the problem (and the name of its master 6 view), the form (`reference`,
/// `planted`, `winner`, `textbook`, `textbook_planted`), its master 5 s-expression and its master 6 s-expression.
#[derive(Debug, Clone, Copy)]
pub struct Reference6 {
    pub problem: &'static str,
    pub form: &'static str,
    pub master5: fn() -> String,
    pub master6: fn() -> String,
}

impl Reference6 {
    /// `<problem>.<form>`, the reference's key in the fixture.
    pub fn key(&self) -> String {
        format!("{}.{}", self.problem, self.form)
    }
}

/// Reduce's reference, master 6.
pub const REDUCE_REFERENCE6: &str = "(sum@i32 xs)";
pub const REDUCE_PLANTED6: &str = "(sum@i32 (map2.min@a.i32 xs xs))";
/// Scan's reference, master 6.
pub const SCAN_REFERENCE6: &str = "(scan.add@a.i32 xs)";
pub const SCAN_PLANTED6: &str = "(scan.add@a.i32 (map2.max@a.i32 xs xs))";
/// The histogram's reference, master 6: the bins' size `k` and the tabulate's `n` are `var.size` leaves.
pub const HISTOGRAM_REFERENCE6: &str = "(hist.add@a.i32 k (tab@a.i32[i] n (mod@i32 k (get@i32 is i))) vs)";
pub const HISTOGRAM_PLANTED6: &str = "(hist.add@a.i32 k (tab@a.i32[i] n (mod@i32 k (get@i32 is i))) (map2.min@a.i32 vs vs))";
/// Pathfinder's reference, master 6: the loop from 1 over `rows` binds its state `s` and its counter `i`, the
/// tabulate over `cols` its index `j`.
pub const PATHFINDER_REFERENCE6: &str = "(loop1@a.i32[s,i] rows (row@a.i32 walls zero@i64)
  (tab@a.i32[j] cols (add@i32 (get@i32 (row@a.i32 walls i) j)
                      (min@i32 (min@i32 (get@i32 s j) (get@i32 s (prev@i64 j))) (get@i32 s (next@i64 j))))))";
pub const PATHFINDER_PLANTED6: &str = "(map2.min@a.i32
  (loop1@a.i32[s,i] rows (row@a.i32 walls zero@i64)
    (tab@a.i32[j] cols (add@i32 (get@i32 (row@a.i32 walls i) j)
                        (min@i32 (min@i32 (get@i32 s j) (get@i32 s (prev@i64 j))) (get@i32 s (next@i64 j))))))
  (loop1@a.i32[s,i] rows (row@a.i32 walls zero@i64)
    (tab@a.i32[j] cols (add@i32 (get@i32 (row@a.i32 walls i) j)
                        (min@i32 (get@i32 s j) (min@i32 (get@i32 s (prev@i64 j)) (get@i32 s (next@i64 j))))))))";
/// A7's reference, master 6: every matrix is `[][]f32`, and the interface's inputs are root variables.
pub const A7_REFERENCE6: &str = "(map2_rows@m.f32 add@fn.arr2_f32_arr_f32_arr_f32
  (matmul@m.f32 m_n_m m_m_p)
  (transpose@m.f32 (matmul@m.f32 (transpose@m.f32 m_m_p) (transpose@m.f32 m_n_m))))";
pub const A7_WINNER6: &str = "(map2_rows@m.f32 add@fn.arr2_f32_arr_f32_arr_f32
  (matmul@m.f32 m_n_m m_m_p)
  (matmul@m.f32 m_n_m m_m_p))";
pub const A7_PLANTED6: &str = "(map_rows@m.f32 id@fn.arr_f32_arr_f32
  (map2_rows@m.f32 add@fn.arr2_f32_arr_f32_arr_f32
    (matmul@m.f32 m_n_m m_m_p)
    (transpose@m.f32 (matmul@m.f32 (transpose@m.f32 m_m_p) (transpose@m.f32 m_n_m)))))";

/// [`KMEANS_SEED_ASSIGN`], master 6: the tabulate over the points binds `i`; the argmin loop from 1 over the centres
/// binds the best distance `best`, its centre `bi` and the counter `c`; the first centre's distance loop over `d`
/// binds `acc` and `j`.
pub const KMEANS_ASSIGN6: &str = "(tab@a.f32[i] n (snd.f32.f32@f32 (loop1@f32.f32[best,bi,c] k (pair@f32.f32 (sqrt@f32 (loop@f32[acc,j] d 0 (add@f32 acc (mul@f32 (sub@f32 (get@f32 (row@a.f32 points i) j) (get@f32 (row@a.f32 centres zero@i64) j)) (sub@f32 (get@f32 (row@a.f32 points i) j) (get@f32 (row@a.f32 centres zero@i64) j)))))) 0)
  (psel@f32.f32 (sub@f32 DIST best) (pair@f32.f32 DIST (ival@f32 c)) (pair@f32.f32 best bi)))))";
/// [`KMEANS_SEED_DIST`], master 6: the distance of centre `c`, its loop over `d` binding `acc2` and `j2`.
pub const KMEANS_DIST6: &str = "(sqrt@f32 (loop@f32[acc2,j2] d 0 (add@f32 acc2 (mul@f32 (sub@f32 (get@f32 (row@a.f32 points i) j2) (get@f32 (row@a.f32 centres c) j2)) (sub@f32 (get@f32 (row@a.f32 points i) j2) (get@f32 (row@a.f32 centres c) j2))))))";
/// [`KMEANS_SEED_UPDATE`], master 6: the rows over the clusters bind `ck`, the tabulate over the coordinates `cj`.
pub const KMEANS_UPDATE6: &str = "(tabrows@m.f32[ck] k d (tab@a.f32[cj] d (sel@f32 (neg@f32 (thd.f32.f32.f32@f32 LOOP)) (div@f32 (fst.f32.f32.f32@f32 LOOP) (thd.f32.f32.f32@f32 LOOP)) (get@f32 (row@a.f32 centres ck) cj))))";
/// [`KMEANS_SEED_CELL_LOOP`], master 6: the loop over the points binds the triple (`sum`, `comp`, `count`) and the
/// point `p`; `assign` is the kernel boundary's `let`.
pub const KMEANS_CELL_LOOP6: &str = "(loop@f32.f32.f32[sum,comp,count,p] n zero@f32.f32.f32 (psel@f32.f32.f32 (sub@f32 (abs@f32 (sub@f32 (get@f32 assign p) (ival@f32 ck))) 1)
    (triple@f32.f32.f32 (add@f32 sum (sub@f32 (get@f32 (row@a.f32 points p) cj) comp))
            (sub@f32 (sub@f32 (add@f32 sum (sub@f32 (get@f32 (row@a.f32 points p) cj) comp)) sum) (sub@f32 (get@f32 (row@a.f32 points p) cj) comp))
            (add@f32 count 1))
    (triple@f32.f32.f32 sum comp count)))";

/// The k-means reference program, master 6: `(let <assignments> <update>)`, the `let` binding `assign`.
pub fn kmeans_reference6() -> String {
    format!("(let.a.f32@m.f32[assign] {} {})", KMEANS_ASSIGN6.replace("DIST", KMEANS_DIST6), KMEANS_UPDATE6.replace("LOOP", KMEANS_CELL_LOOP6))
}

/// The k-means planted slowdown, master 6: the assignments twice and their elementwise least.
pub fn kmeans_planted6() -> String {
    let assign = KMEANS_ASSIGN6.replace("DIST", KMEANS_DIST6);
    format!("(let.a.f32@m.f32[assign] (map2.min@a.f32 {assign} {assign}) {})", KMEANS_UPDATE6.replace("LOOP", KMEANS_CELL_LOOP6))
}

/// [`KMEANS_TEXTBOOK`]'s membership, master 6: the tabulate over the points binds `i`; the loop over the centres binds
/// the nearest centre so far `m` (an `f32`) and the counter `c`; each distance's tabulate over `d` binds `j`.
pub const KMEANS_TEXTBOOK_MEMBERSHIP6: &str = "(tab@a.f32[i] n (loop@f32[m,c] k 0 (sel@f32 (sub@f32 (sum@f32 (tab@a.f32[j] d (mul@f32 (sub@f32 (get@f32 (row@a.f32 points i) j) (get@f32 (row@a.f32 centres (clamp@i64 k m)) j)) (sub@f32 (get@f32 (row@a.f32 points i) j) (get@f32 (row@a.f32 centres (clamp@i64 k m)) j))))) (sum@f32 (tab@a.f32[j] d (mul@f32 (sub@f32 (get@f32 (row@a.f32 points i) j) (get@f32 (row@a.f32 centres c) j)) (sub@f32 (get@f32 (row@a.f32 points i) j) (get@f32 (row@a.f32 centres c) j)))))) m (ival@f32 c))))";

/// [`KMEANS_TEXTBOOK`], master 6, with `ONES` the ones the counts are a histogram of.
fn kmeans_textbook6_with(ones: &str) -> String {
    let m = KMEANS_TEXTBOOK_MEMBERSHIP6;
    format!("(rowdiv@m.f32 (histrows.add@m.f32 k d {m} points) (map2.max@a.f32 (replicate@a.f32 k 1) (hist.add@a.f32 k {m} {ones})))")
}

/// The textbook k-means step, master 6.
pub fn kmeans_textbook6() -> String {
    kmeans_textbook6_with("(replicate@a.f32 n 1)")
}

/// The textbook step's planted slowdown, master 6.
pub fn kmeans_textbook_planted6() -> String {
    kmeans_textbook6_with("(map2.min@a.f32 (replicate@a.f32 n 1) (replicate@a.f32 n 1))")
}

/// EVERY REFERENCE PROGRAM in both masters, in the problem registry's order ([`PROBLEM_NAMES`]).
pub const REFERENCES6: [Reference6; 15] = [
    Reference6 { problem: "reduce", form: "reference", master5: || REDUCE_REFERENCE.to_string(), master6: || REDUCE_REFERENCE6.to_string() },
    Reference6 { problem: "reduce", form: "planted", master5: || REDUCE_PLANTED.to_string(), master6: || REDUCE_PLANTED6.to_string() },
    Reference6 { problem: "scan", form: "reference", master5: || SCAN_REFERENCE.to_string(), master6: || SCAN_REFERENCE6.to_string() },
    Reference6 { problem: "scan", form: "planted", master5: || SCAN_PLANTED.to_string(), master6: || SCAN_PLANTED6.to_string() },
    Reference6 { problem: "histogram", form: "reference", master5: || HISTOGRAM_REFERENCE.to_string(), master6: || HISTOGRAM_REFERENCE6.to_string() },
    Reference6 { problem: "histogram", form: "planted", master5: || HISTOGRAM_PLANTED.to_string(), master6: || HISTOGRAM_PLANTED6.to_string() },
    Reference6 { problem: "pathfinder", form: "reference", master5: || PATHFINDER_REFERENCE.to_string(), master6: || PATHFINDER_REFERENCE6.to_string() },
    Reference6 { problem: "pathfinder", form: "planted", master5: || PATHFINDER_PLANTED.to_string(), master6: || PATHFINDER_PLANTED6.to_string() },
    Reference6 { problem: "kmeans", form: "reference", master5: kmeans_seed, master6: kmeans_reference6 },
    Reference6 { problem: "kmeans", form: "planted", master5: kmeans_planted, master6: kmeans_planted6 },
    Reference6 { problem: "kmeans", form: "textbook", master5: || KMEANS_TEXTBOOK.to_string(), master6: kmeans_textbook6 },
    Reference6 { problem: "kmeans", form: "textbook_planted", master5: || KMEANS_TEXTBOOK_PLANTED.to_string(), master6: kmeans_textbook_planted6 },
    Reference6 { problem: "a7", form: "reference", master5: || A7_REFERENCE.to_string(), master6: || A7_REFERENCE6.to_string() },
    Reference6 { problem: "a7", form: "winner", master5: || A7_WINNER.to_string(), master6: || A7_WINNER6.to_string() },
    Reference6 { problem: "a7", form: "planted", master5: || A7_PLANTED.to_string(), master6: || A7_PLANTED6.to_string() },
];

/// THE EXPECTED TEXTS: every reference's master 5 rendering ([`master5_rendering`]), each after a line
/// `==== <problem>.<form> ====`, in [`REFERENCES6`]'s order. Task F2's renderer must reproduce each byte for byte.
pub const REFERENCES6_MASTER5_TEXTS: &str = include_str!("../tests/fixtures/futhark_m6_references.txt");

/// The fixture's separator line for `key`.
pub fn references6_separator(key: &str) -> String {
    format!("==== {key} ====\n")
}

/// The expected text (the stored master 5 rendering) of the reference `key` (`<problem>.<form>`).
pub fn expected_text6(key: &str) -> Option<&'static str> {
    let sep = references6_separator(key);
    let start = REFERENCES6_MASTER5_TEXTS.find(&sep)? + sep.len();
    let rest = &REFERENCES6_MASTER5_TEXTS[start..];
    Some(&rest[..rest.find("\n==== ").map_or(rest.len(), |e| e + 1)])
}

/// A reference's master 5 rendering: its s-expression built in its master 5 view and rendered by [`program`],
/// unchanged.
pub fn master5_rendering(r: &Reference6) -> Result<String, String> {
    let inst = problem(r.problem)?.instance;
    let k = load(&inst)?;
    let sig = signature(&inst)?;
    program(&k, &sig, &build(&k, &(r.master5)())?)
}

/// A problem's master 6 kingdom: its committed view (`kingdoms/futhark/sig/<problem>/m6/`) over the committed master 6.
pub fn kingdom6(problem: &str) -> Result<crate::futhark_table::Kingdom6, String> {
    crate::futhark_view::master6::committed_kingdom(problem)
}

/// The length of each data type's constant pool `C_T` in a master 6 kingdom: its literal pool, for the data types
/// with a `const` row.
pub fn const_len6(k: &crate::futhark_table::Kingdom6) -> BTreeMap<Ty, u32> {
    k.const_len()
}

/// A parsed master 6 s-expression: its head (a symbol, or a leaf's name) with the binder's variable names, and its
/// arguments.
#[derive(Debug, Clone)]
struct Sexp6 {
    head: String,
    labels: Vec<String>,
    kids: Vec<Sexp6>,
}

fn parse_sexp6(text: &str) -> Result<Sexp6, String> {
    let spaced = text.replace('(', " ( ").replace(')', " ) ");
    let tokens: Vec<&str> = spaced.split_whitespace().collect();
    let mut pos = 0;
    let e = parse6_at(&tokens, &mut pos)?;
    if pos != tokens.len() {
        return Err(format!("our software has a bug: the master 6 s-expression has text after its end at token {pos}"));
    }
    Ok(e)
}

fn parse6_at(tokens: &[&str], pos: &mut usize) -> Result<Sexp6, String> {
    let bad = |what: &str| format!("our software has a bug: the master 6 s-expression {what}");
    let t = *tokens.get(*pos).ok_or_else(|| bad("ends early"))?;
    *pos += 1;
    if t == ")" {
        return Err(bad("has an unopened ')'"));
    }
    if t != "(" {
        if t.contains('[') {
            return Err(bad(&format!("names the variables of the leaf {t}, which binds none")));
        }
        return Ok(Sexp6 { head: t.to_string(), labels: Vec::new(), kids: Vec::new() });
    }
    let h = *tokens.get(*pos).ok_or_else(|| bad("ends after '('"))?;
    *pos += 1;
    let (head, labels) = match h.split_once('[') {
        Some((name, rest)) => {
            let names = rest.strip_suffix(']').ok_or_else(|| bad(&format!("has the binder {h} with no closing ']'")))?;
            let labels: Vec<String> = names.split(',').map(str::to_string).collect();
            if labels.iter().any(String::is_empty) {
                return Err(bad(&format!("has the binder {h} with an empty variable name")));
            }
            if let Some(l) = labels.iter().enumerate().find_map(|(i, l)| labels[..i].contains(l).then_some(l)) {
                return Err(bad(&format!("has the binder {h}, which names {l} twice")));
            }
            (name.to_string(), labels)
        }
        None => (h.to_string(), Vec::new()),
    };
    let mut kids = Vec::new();
    while tokens.get(*pos) != Some(&")") {
        if *pos >= tokens.len() {
            return Err(bad(&format!("leaves ({head} unclosed")));
        }
        kids.push(parse6_at(tokens, pos)?);
    }
    *pos += 1;
    Ok(Sexp6 { head, labels, kids })
}

/// What the first pass of [`build6`] records: each binder's bound argument's path with its variables' names, each
/// variable leaf's path with its name, and the path of each `var@T` leaf written by its own symbol name (the leaf that
/// reads `T`'s constant, nothing of `T` being in scope).
#[derive(Default)]
struct Names6 {
    binders: Vec<(Vec<u32>, Vec<String>)>,
    leaves: Vec<(Vec<u32>, String)>,
    constants: Vec<Vec<u32>>,
}

fn build6_at(k: &crate::futhark_table::Kingdom6, e: &Sexp6, ty: Ty, path: Vec<u32>, names: &mut Names6) -> Result<Tree, String> {
    use crate::geneframe::Role;
    let bad = |what: String| format!("our software has a bug: {}'s master 6 tree: {what}", k.kingdom);
    let ty_name = |t: Ty| k.data_type(t).map_or_else(|| format!("{t:?}"), |d| d.name.clone());
    let by_name = k.symbol_named(&e.head);
    if !e.kids.is_empty() || !e.labels.is_empty() {
        let s = by_name.ok_or_else(|| bad(format!("no symbol {}", e.head)))?;
        if k.output(s) != ty {
            return Err(bad(format!("{} returns {}, its slot takes {}", e.head, ty_name(k.output(s)), ty_name(ty))));
        }
        let inputs = k.inputs(s).to_vec();
        if inputs.len() != e.kids.len() {
            return Err(bad(format!("{} takes {} inputs, given {}", e.head, inputs.len(), e.kids.len())));
        }
        if !e.labels.is_empty() {
            let (pos, count) = k.binds(s)?.ok_or_else(|| bad(format!("{} names its variables but binds none", e.head)))?;
            if count != e.labels.len() {
                return Err(bad(format!("{} binds {count} variables, {} named", e.head, e.labels.len())));
            }
            let mut at = path.clone();
            at.push(pos);
            names.binders.push((at, e.labels.clone()));
        }
        let mut kids = Vec::with_capacity(e.kids.len());
        for (i, (kid, kty)) in e.kids.iter().zip(&inputs).enumerate() {
            let mut p = path.clone();
            p.push(i as u32);
            kids.push(build6_at(k, kid, *kty, p, names)?);
        }
        return Ok(Tree { symbol: s, const_index: None, kids });
    }
    if let Some(s) = by_name {
        // A variable terminal by its own name (`var@i64`): the leaf that reads its data type's constant, which the
        // second pass checks (nothing of the type is in scope there).
        if k.role(s) == Role::Variable && k.output(s) == ty {
            names.constants.push(path);
            return Ok(Tree { symbol: s, const_index: Some(0), kids: Vec::new() });
        }
        // A leaf symbol takes no input: a terminal, or a function value of no argument (`add@fn.arr2_f32_...`).
        if k.role(s) != Role::Function || !k.inputs(s).is_empty() || k.output(s) != ty {
            return Err(bad(format!("the leaf {} is not a terminal of {}", e.head, ty_name(ty))));
        }
        return Ok(Tree { symbol: s, const_index: None, kids: Vec::new() });
    }
    let of_role = |role: Role| k.symbols().find(|r| r.role == role && k.output(r.symbol as SymbolId) == ty).map(|r| r.symbol as SymbolId);
    if let Some(s) = of_role(Role::Constant) {
        if let Some(i) = (0..k.literals(s).len()).find(|i| k.literal_text(s, *i).as_deref() == Some(e.head.as_str())) {
            return Ok(Tree { symbol: s, const_index: Some(i as u32), kids: Vec::new() });
        }
    }
    let s = of_role(Role::Variable).ok_or_else(|| bad(format!("the leaf {} is no symbol, no literal of {} and no variable: {} has no var row", e.head, ty_name(ty), ty_name(ty))))?;
    names.leaves.push((path, e.head.clone()));
    Ok(Tree { symbol: s, const_index: Some(0), kids: Vec::new() })
}

fn tree_at_mut<'a>(t: &'a mut Tree, path: &[u32]) -> Option<&'a mut Tree> {
    match path.split_first() {
        None => Some(t),
        Some((i, rest)) => tree_at_mut(t.kids.get_mut(*i as usize)?, rest),
    }
}

/// THE MASTER 6 BUILDER: the tree of a master 6 s-expression (the syntax above [`Reference6`]) in kingdom `k`, every
/// variable leaf a `var.T` whose index word names, in the leaf's environment as G3's decoder builds it, the variable
/// the leaf names: the innermost binder above that names it, else the root variable of that name. A leaf named by the
/// variable terminal's own symbol name (`var@i64`) is that terminal reading its data type's constant, as the decoder
/// reads a `var.T` with nothing of `T` in scope ([`sexp6`] writes it so); where a variable of `T` is in scope it is an
/// error, since the leaf would read that variable. A name in scope nowhere, of another data type, or a malformed text
/// is an error. Every word is checked by decoding the tree again.
pub fn build6(k: &crate::futhark_table::Kingdom6, text: &str) -> Result<Tree, String> {
    build6_named(k, text).map(|(tree, _)| tree)
}

/// Each variable leaf's path and the name a master 6 text gives it, in the text's order.
type NamedLeaves6 = Vec<(Vec<u32>, String)>;

/// [`build6`], with its variable leaves' names.
fn build6_named(k: &crate::futhark_table::Kingdom6, text: &str) -> Result<(Tree, NamedLeaves6), String> {
    use crate::typed_environment::{Binder, Choice};
    let bad = |what: String| format!("our software has a bug: {}'s master 6 tree: {what}", k.kingdom);
    let mut names = Names6::default();
    let mut tree = build6_at(k, &parse_sexp6(text)?, k.entry.output, Vec::new(), &mut names)?;
    let decoding = k.decode(&tree)?;
    let mut wanted: Vec<(Vec<u32>, Binder)> = Vec::with_capacity(names.leaves.len());
    for (path, name) in &names.leaves {
        let leaf = decoding.node(path).ok_or_else(|| bad(format!("no node at {path:?}")))?;
        let binder = names.binders.iter().filter(|(at, labels)| path.starts_with(at) && labels.contains(name)).max_by_key(|(at, _)| at.len());
        let variable = match binder {
            Some((at, labels)) => {
                let env = &decoding.node(at).ok_or_else(|| bad(format!("no node at {at:?}")))?.environment;
                let bound: Vec<_> = env.variables.iter().filter(|v| v.binder.depth == env.depth).collect();
                let i = labels.iter().position(|l| l == name).ok_or_else(|| bad(format!("{name} is not a label")))?;
                *bound.get(i).ok_or_else(|| bad(format!("the binder at {at:?} binds {} variables, not {}", bound.len(), labels.len())))?
            }
            None => decoding.root.environment.variables.iter().find(|v| v.name.as_deref() == Some(name.as_str())).ok_or_else(|| bad(format!("{name} is bound nowhere above {path:?} and is no root variable")))?,
        };
        if variable.data_type != leaf.data_type {
            return Err(bad(format!("{name} at {path:?} is a {:?}, its slot takes {:?}", variable.data_type, leaf.data_type)));
        }
        let word = leaf.environment.index_of(leaf.data_type, variable.binder).ok_or_else(|| bad(format!("{name} is not in scope at {path:?}")))?;
        tree_at_mut(&mut tree, path).ok_or_else(|| bad(format!("no node at {path:?}")))?.const_index = Some(word);
        wanted.push((path.clone(), variable.binder));
    }
    let again = k.decode(&tree)?;
    for path in &names.constants {
        let node = again.node(path).ok_or_else(|| bad(format!("no node at {path:?}")))?;
        let got = node.choice;
        if got != Some(Choice::Constant) {
            let name = k.name(node.symbol);
            return Err(bad(format!("{name} at {path:?} names its data type's constant, but a variable of the type is in scope there and the leaf reads {got:?}")));
        }
    }
    for (path, binder) in wanted {
        let got = again.node(&path).and_then(|n| n.choice);
        if got != Some(Choice::Variable(binder)) {
            return Err(bad(format!("the variable at {path:?} reads as {got:?}, not {binder:?}")));
        }
    }
    Ok((tree, names.leaves))
}

/// THE MASTER 6 TEXT FORM of a tree, the s-expression [`build6`] reads (the gene's text in ledgers, halls of fame and
/// checkpoints): a symbol by name, a constant by its pool literal, a root variable by name, a bound variable
/// `v<depth>_<ordinal>` named at its binder (`(tab@a.i32[v1_0] n ..)`), and a `var.T` with nothing of `T` in scope,
/// which reads `T`'s constant, by its own symbol name (`var@i64`). `build6(k, &sexp6(k, t)?)` decodes as `t` does.
pub fn sexp6(k: &crate::futhark_table::Kingdom6, tree: &Tree) -> Result<String, String> {
    let decoding = k.decode(tree)?;
    let mut s = String::new();
    sexp6_at(k, &decoding.root, &mut s)?;
    Ok(s)
}

fn sexp6_at(k: &crate::futhark_table::Kingdom6, n: &crate::typed_environment::Node, s: &mut String) -> Result<(), String> {
    use crate::geneframe::Role;
    use crate::typed_environment::Choice;
    let bad = |what: String| format!("our software has a bug: {}'s master 6 text form: {what}", k.kingdom);
    let name = k.name(n.symbol).to_string();
    match (k.role(n.symbol), n.choice) {
        (Role::Constant, Some(Choice::Dc(i))) => s.push_str(&k.literal_text(n.symbol, i as usize).ok_or_else(|| bad(format!("{name} has no literal {i}")))?),
        (Role::Variable, Some(Choice::Variable(b))) => {
            let v = n.environment.variables.iter().find(|v| v.binder == b).ok_or_else(|| bad(format!("the variable {b:?} is not in scope")))?;
            match &v.name {
                Some(vname) if b.depth == 0 => s.push_str(vname),
                _ => s.push_str(&format!("v{}_{}", b.depth, b.ordinal)),
            }
        }
        (Role::Variable, Some(Choice::Constant)) => s.push_str(&name),
        (_, choice) if n.kids.is_empty() => {
            if choice.is_some() {
                return Err(bad(format!("the leaf {name} reads {choice:?}")));
            }
            s.push_str(&name);
        }
        _ => {
            s.push('(');
            s.push_str(&name);
            if let Some((pos, count)) = k.binds(n.symbol)? {
                let depth = n.kids.get(pos as usize).map(|kid| kid.environment.depth).ok_or_else(|| bad(format!("{name} binds at its argument {pos}, which it has not")))?;
                let labels: Vec<String> = (0..count).map(|o| format!("v{depth}_{o}")).collect();
                s.push_str(&format!("[{}]", labels.join(",")));
            }
            for kid in &n.kids {
                s.push(' ');
                sexp6_at(k, kid, s)?;
            }
            s.push(')');
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_references6;
#[cfg(test)]
mod tests_micro;
#[cfg(test)]
mod tests_view;

#[cfg(test)]
mod tests_kmeans;
#[cfg(test)]
mod tests_kmeans_seed;
#[cfg(test)]
mod tests_a7;
#[cfg(test)]
mod tests_checklist;

mod interp6;
#[cfg(test)]
mod tests_interp6;
