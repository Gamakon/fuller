//! The SIGNATURE FAMILY of the Futhark kingdom: a Futhark kingdom whose interface is a program's signature, in the
//! direction of "any Futhark program: inputs and outputs are any typed values". A problem's kingdom is a VIEW over
//! the master Futhark table (`kingdoms/futhark/master/6/`, [`crate::futhark_view::master6`]): its specification
//! (`kingdoms/futhark/sig/<problem>/m6/view.tsv`) is written by `kingdoms/futhark/sniff.py` from the problem's Futhark
//! source, and its rows (the kingdom `futhark.<problem>` of the master tables, `kingdoms/master/`) by
//! `kingdoms/futhark/view.py` through `write_rows.py`, both committed and checked byte for byte
//! (`docs/PLAN_futhark_kingdom.md` section 7 and 7.7 K0).
//!
//! A gene's types are data types (`i32`, `[]i32`, `[][]f32`, `i64`, `size`, tuples, function types); an array's length
//! and an index's bound are properties, scope is the environment the decoder builds, and where a requirement fails the
//! rendered program carries an adapter (`docs/DESIGN_futhark_types_and_properties.md`). Every gene expresses a program
//! Futhark accepts; the tests here check that Futhark agrees and that the compiled program computes what the host
//! interpreter says.
//!
//! What this module holds: the problem registry (the reference written as a gene, the published program as the speed
//! yardstick, and its stored cases), the declared signature (read from the view's specification), Futhark's binary
//! value format for the cases (typed: `i32`, or `f32` as its raw bits, with the comparison policy on [`Array`]), the
//! host interpreter (`interp6`: the meaning of a gene, a test oracle, exact arithmetic as Futhark's: `i32` wraps, `f32`
//! is IEEE with `fmin`/`fmax` and `i64.f32` as measured through `futhark c`), the exact work of a tree on given sizes,
//! the quality measures, and a builder that writes a program given as an s-expression of the view's symbols as a tree
//! ([`build6`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use crate::futhark_table::GeneRng;
use crate::futhark_view::master6::ViewSpec6;
use crate::geneframe::Ty;
use crate::kingdom_contract::{KingdomContract, SymbolId};
use crate::typed_gep::{Domain, GeneLayout, Tree};

/// A problem: its name (the name of its view, `kingdoms/futhark/sig/<name>/m6/`), its published program written as a
/// gene of the view (an s-expression of the view's symbols, for [`build6`]), the published program itself as the
/// speed yardstick (a Futhark source whose `main` has the declared signature), and the directory of its stored cases.
#[derive(Debug, Clone, Copy)]
pub struct Problem {
    pub name: &'static str,
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
    /// above the reference and the planted slowdown ([`DEFAULT_PROGRAM_SIZE`] for the i32 benchmarks and A7; k-means's
    /// is [`KMEANS_PROGRAM_SIZE`]).
    pub program_size: usize,
}

/// The state of a problem's reference slot.
#[derive(Debug, Clone, Copy)]
pub enum Seed {
    /// The prototype's program as a gene (an s-expression of the view's symbols, for [`build6`]), and THE PLANTED
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
/// `j`, `j - 1` and `j + 1`, the published order of the two `min`s ([`PATHFINDER_REFERENCE6`]). The published kernel
/// skips a neighbour that is off the edge; here the index is clamped, so the off-edge neighbour is `j` itself, which
/// the least already holds. Its planted slowdown is the elementwise least of the reference and its re-associated twin
/// (the `min`s grouped the other way), which Futhark cannot merge and the e-graph can.
pub const PATHFINDER_PROBLEM: Problem = Problem {
    name: "pathfinder",
    seed: Seed::Program { reference: PATHFINDER_REFERENCE6, planted: PATHFINDER_PLANTED6 },
    yardstick: include_str!("../tests/corpus/futhark_problems/pathfinder/yardstick.fut"),
    fitness: "futhark-slope4-qqs",
    corpus: "tests/corpus/futhark_problems/pathfinder",
    timing_n: PATHFINDER_TIMING_ROWS * 100_000,
    program_size: DEFAULT_PROGRAM_SIZE,
};

/// Reduce (the study's micro benchmark, `micro/reduce.fut`, `sum_i32`: `xs: [n]i32 -> i32`): its published program,
/// `i32.sum`, as a gene; its planted slowdown the sum of the elementwise least of the input and itself (a `map2`
/// Futhark keeps; the e-graph's `map2 min a a = a` removes it).
pub const REDUCE_PROBLEM: Problem = Problem {
    name: "reduce",
    seed: Seed::Program { reference: REDUCE_REFERENCE6, planted: REDUCE_PLANTED6 },
    yardstick: include_str!("../tests/corpus/futhark_problems/reduce/yardstick.fut"),
    fitness: "futhark-slope4-qqs",
    corpus: "tests/corpus/futhark_problems/reduce",
    timing_n: 15 << 20,
    program_size: DEFAULT_PROGRAM_SIZE,
};

/// Scan (`micro/scan.fut`, `sum_i32`: `xs: [n]i32 -> [n]i32`, inclusive prefix sums): `scan (+) 0` as a gene; its
/// planted slowdown the scan of the elementwise greatest of the input and itself.
pub const SCAN_PROBLEM: Problem = Problem {
    name: "scan",
    seed: Seed::Program { reference: SCAN_REFERENCE6, planted: SCAN_PLANTED6 },
    yardstick: include_str!("../tests/corpus/futhark_problems/scan/yardstick.fut"),
    fitness: "futhark-slope4-qqs",
    corpus: "tests/corpus/futhark_problems/scan",
    timing_n: 15 << 20,
    program_size: DEFAULT_PROGRAM_SIZE,
};

/// The histogram (`micro/reduce_by_index.fut`, `sum_i32`: `(k: i32) (is: [n]i32) (vs: [n]i32) -> [k]i32`, the sum of
/// the values in each bin `is % k`): `hist (+) 0 k (map (% k) is) vs` as a gene, the bins the indices modulo `k`
/// tabulated over `n`; its planted slowdown the values replaced by the elementwise least of the values and themselves.
pub const HISTOGRAM_PROBLEM: Problem = Problem {
    name: "histogram",
    seed: Seed::Program { reference: HISTOGRAM_REFERENCE6, planted: HISTOGRAM_PLANTED6 },
    yardstick: include_str!("../tests/corpus/futhark_problems/histogram/yardstick.fut"),
    fitness: "futhark-slope4-qqs",
    corpus: "tests/corpus/futhark_problems/histogram",
    timing_n: 15 << 20,
    program_size: DEFAULT_PROGRAM_SIZE,
};

/// THE K-MEANS STEP: `(points: [n][d]f32) (centres: [k][d]f32) -> [k][d]f32` (`docs/PLAN_futhark_kingdom.md`, step
/// 4; the seed is UltraDim's own kernels, 4c (d), `docs/kmeans_seed.md`), at depth 3 with ONE KERNEL BOUNDARY, a `let`
/// of the assignments the first kernel writes and the second reads. Its seed is a `Seed::Translated`: the program text
/// is assembled from its parts once ([`kmeans_reference6_static`]), so the registry stays a constant.
pub const KMEANS_PROBLEM: Problem = Problem {
    name: "kmeans",
    seed: Seed::Translated { reference: kmeans_reference6_static, planted: kmeans_planted6_static },
    yardstick: include_str!("../tests/corpus/futhark_problems/kmeans/yardstick.fut"),
    fitness: "futhark-kmeans-better",
    corpus: "tests/corpus/futhark_problems/kmeans",
    timing_n: KMEANS_TIMING_N,
    program_size: KMEANS_PROGRAM_SIZE,
};

/// K-means's program size: its seed was 131 functions (the cell's loop written three times, the centre's distance
/// twice) and its planted slowdown 183 with the shape-class kingdom of before master 6, so 192 holds both; the
/// textbook pool programs fit too.
pub const KMEANS_PROGRAM_SIZE: usize = 192;

/// A7, THE MATRIX KINGDOM AS A PROBLEM OF THE ONE RUNNER (`docs/PLAN_futhark_kingdom.md`, step 4; the product
/// decision in `docs/PLAN_futhark_product.md`): the master's matrix family over the sizes `n`, `m`, `p`, rooted at
/// `[n][p]` (`kingdoms/futhark/sig/a7/m6/`). Its signature is the interface's nine matrices (`m_n_n: [n][n]f32`, ...
/// `m_p_p: [p][p]f32`), and its output `[n][p]f32`. Its published program is the product of two matrices computed
/// TWICE (once directly, once the transposed way) and the two added ([`A7_REFERENCE6`]); its planted slowdown the
/// reference under a `map` of the identity row function, a traversal Futhark keeps and the e-graph's `map id = id`
/// rule removes.
pub const A7_PROBLEM: Problem = Problem {
    name: "a7",
    seed: Seed::Program { reference: A7_REFERENCE6, planted: A7_PLANTED6 },
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
/// uniform in [-8, 8) with the first of every matrix `-0.0`.
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

// ---- the declared signature -------------------------------------------------------------------------------------

/// A typed array of the signature: its name, element type and dimension roles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub elem: String,
    pub dims: Vec<String>,
}

/// The declared signature, read from the view's specification (`m6/view.tsv`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    pub inputs: Vec<Param>,
    pub output: Param,
    /// The view's binder depth: a node runs at most `max(D) ** depth` times (0 in a view with no work bound, A7).
    pub depth: u32,
    /// Every size, in order of first appearance.
    pub dims: Vec<String>,
    /// The SIZE INPUTS: sizes given as an `i32` value parameter of the entry point (a `size` line of the specification
    /// with the element `i32`; the parameter is in `inputs`, a scalar, before the arrays), rather than by an array's
    /// shape.
    pub size_inputs: Vec<String>,
    /// The view's element types, in the specification's order ([`ViewSpec6::elements`]).
    pub elements: Vec<String>,
}

/// The signature of a problem's committed view ([`signature_of`]); a shared handle, as the kingdom's.
pub fn signature(problem: &str) -> Result<Arc<Signature>, String> {
    let (_, spec) = crate::futhark_view::master6::COMMITTED6.iter().find(|c| c.0 == problem).ok_or_else(|| format!("no Futhark problem {problem} ({})", PROBLEM_NAMES.join(", ")))?;
    signature_of(&ViewSpec6::parse(spec)?).map(Arc::new).map_err(|e| format!("{problem}: {e}"))
}

/// The signature a view specification declares, as its entry point takes it ([`crate::futhark_table::Kingdom6::frame`]):
/// the size inputs (each an `i32` scalar), then the input arrays in order; the output; the depth (0 with no work
/// bound); the sizes in order of first appearance.
pub fn signature_of(spec: &ViewSpec6) -> Result<Signature, String> {
    let sizes: Vec<_> = spec.env.iter().filter(|l| l.kind == "size").collect();
    if sizes.is_empty() {
        return Err("the specification has no sizes".into());
    }
    let size_inputs: Vec<String> = sizes.iter().filter(|l| !l.elem.is_empty()).map(|l| l.name.clone()).collect();
    let mut inputs: Vec<Param> = size_inputs.iter().map(|n| Param { name: n.clone(), elem: "i32".into(), dims: Vec::new() }).collect();
    inputs.extend(spec.env.iter().filter(|l| l.kind == "input").map(|l| Param { name: l.name.clone(), elem: l.elem.clone(), dims: l.dims.clone() }));
    let (name, elem, dims) = &spec.output;
    Ok(Signature {
        inputs,
        output: Param { name: name.clone(), elem: elem.clone(), dims: dims.clone() },
        depth: spec.depth.unwrap_or(0),
        dims: sizes.iter().map(|l| l.name.clone()).collect(),
        size_inputs,
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
/// k-means exact gate of `futhark_rust` compares (bytes); the `i32` comparison
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

// ---- the host interpreter ---------------------------------------------------------------------------------------

/// The elementwise operations of `map2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bin {
    Add,
    Sub,
    Mul,
    Min,
    Max,
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

/// THE INTERPRETER of one problem's view (plan section 7 task F3, `interp6`: the meaning read from G3's decoding, the
/// adapters interpreted, sizes from the properties). It OWNS the kingdom and the signature it reads (shared handles),
/// so it lives as long as its holder.
pub struct Interp(Box<interp6::Interp6>);

impl Interp {
    /// The interpreter of a kingdom whose cases are `sig`'s (its problem's signature: the inputs by name, the sizes
    /// they bind); it keeps a clone of each handle.
    pub fn new6(k: &Arc<crate::futhark_table::Kingdom6>, sig: &Arc<Signature>) -> Result<Interp, String> {
        interp6::Interp6::new(k, sig).map(|i| Interp(Box::new(i)))
    }

    /// The value of an expressed tree on the inputs: the program's output, as Futhark computes it. The tree is decoded
    /// first (passes 2 to 4), and every value is checked against its properties as Futhark checks its sizes.
    pub fn eval(&self, tree: &Tree, inputs: &[Array]) -> Result<Array, String> {
        self.0.eval(tree, inputs)
    }

    /// The node executions of a tree on given sizes, the loops' state traffic included ([`Work::total`]).
    pub fn work(&self, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<u128, String> {
        self.0.work(tree, sizes)
    }

    /// The work of a tree on given sizes, in two parts ([`Work`]); its adapters are charged in [`Work::nodes`], the
    /// per-submit cap's measure.
    pub fn work_parts(&self, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<Work, String> {
        self.0.work_parts(tree, sizes)
    }

    /// THE STATIC COST of a tree at `sizes` (`crate::futhark_cost::program_cost6`, adapters charged).
    pub fn static_cost(&self, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<crate::futhark_cost::Score, String> {
        self.0.static_cost(tree, sizes)
    }

    /// The bound on any tree of a layout: its most nodes, each run at most `max(D) ** depth` times and each costing at
    /// most `1 + max(D) ** 2`, on the given largest dimension.
    pub fn layout_bound(&self, layout: &GeneLayout, max_dim: usize) -> u128 {
        let d = max_dim as u128;
        layout.max_nodes() as u128 * d.pow(self.0.depth()) * (1 + d * d)
    }
}

/// A row function, `[]f32 -> []f32` (the matrix family's function data types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowFn {
    /// `\xs -> xs`
    Id,
    /// `\xs -> map (+1.0) xs`
    Add1,
    /// `\xs -> map (*2.0) xs`
    Scale2,
    /// `\xs -> map (\_ -> 0.0) xs` — the constant row function, whose every element is zero. It is spelled as a
    /// *constant* and not as an addition of zero on purpose: Futhark's simplifier folds `x + 0` away, so a row spelled
    /// `map (+0.0)` would compile to the identity and differ from it only in the sign of a zero. Spelled as a
    /// constant, it differs from the identity by VALUE.
    Zero,
}

/// A binary row function, `[]f32 -> []f32 -> []f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Row2Fn {
    /// `\xs _ys -> xs`
    Id2,
    /// `\xs _ys -> map (+1.0) xs`
    Add1_2,
    /// `\xs _ys -> map (\_ -> 0.0) xs` — the constant binary row function.
    Zero2,
    /// `\xs ys -> map2 (+) xs ys`
    Add2,
}

impl RowFn {
    /// Apply the function to a row.
    fn apply(self, row: &[f32]) -> Vec<f32> {
        match self {
            RowFn::Id => row.to_vec(),
            RowFn::Add1 => row.iter().map(|x| x + 1.0).collect(),
            RowFn::Scale2 => row.iter().map(|x| x * 2.0).collect(),
            RowFn::Zero => vec![0.0; row.len()],
        }
    }
}

impl Row2Fn {
    /// Apply the function to a pair of rows. Futhark's `map2` requires the two arrays to have the same length, and so
    /// does this: the second operand is *checked*, not ignored.
    fn apply(self, x: &[f32], y: &[f32]) -> Result<Vec<f32>, String> {
        if x.len() != y.len() {
            return Err(format!("a binary row function needs two rows of the same length, got {} and {}", x.len(), y.len()));
        }
        Ok(match self {
            Row2Fn::Id2 => x.to_vec(),
            Row2Fn::Add1_2 => x.iter().map(|v| v + 1.0).collect(),
            Row2Fn::Zero2 => vec![0.0; x.len()],
            Row2Fn::Add2 => x.iter().zip(y.iter()).map(|(u, v)| u + v).collect(),
        })
    }
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
        // The exit status names a kill by a signal (the machine under load), which leaves no compiler message.
        return Err(format!("futhark c failed ({}):\n{}", out.status, String::from_utf8_lossy(&out.stderr)));
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

// ---- the reference programs (plan section 7, task F2a) --------------------------------------------------------------
//
// Every reference program (the four benchmarks' references and planted slowdowns, A7's reference, winner and planted
// slowdown, k-means's reference, planted slowdown, textbook form and its planted slowdown) written by hand as a tree
// of its view (`kingdoms/futhark/sig/<problem>/m6/`). Each tree's expected text is stored in
// `tests/fixtures/futhark_m6_references.txt` (the rendering of the shape-class kingdom of before master 6, frozen as
// text), which the renderer reproduces byte for byte.
//
// An s-expression names each function by its view symbol (`tab@a.i32`) and each leaf as one of: a terminal
// symbol (`zero@i64`, `add@fn.arr2_f32_arr_f32_arr_f32`), a literal of the slot's data type's pool (`0`, a `const@T`
// cell whose Dc index is the literal's place in the pool), or a VARIABLE by name: a root variable of the view's
// environment (`points`, `n`) or a name a binder above gives the variables it binds, listed in Binding row order in
// brackets after the binder's symbol (`(tab@a.i32[i] n ...)`, `(loop1@a.i32[s,i] rows ...)`), the innermost binder
// first. A variable leaf becomes the `var.T` of the slot's data type, and its index word is the one G3's decoder says
// names that variable in the leaf's environment ([`crate::typed_environment::Environment::index_of`]): never a number
// computed by hand.

/// One reference program: the problem (the name of its view), the form (`reference`, `planted`, `winner`,
/// `textbook`, `textbook_planted`) and its s-expression.
#[derive(Debug, Clone, Copy)]
pub struct Reference6 {
    pub problem: &'static str,
    pub form: &'static str,
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

/// THE K-MEANS SEED: UltraDim's three k-means kernels (`lib/projection-voting-kmeans/src/gpu_kmeans.rs`:
/// `compute_distances`, `assign_points`, `update_centroids`) translated construct for construct, `docs/kmeans_seed.md`
/// line by line. The kernel boundary `let` names the assignments (`assign`, the `[n]` buffer `assign_points` writes):
/// a tabulate over the points of the argmin over the centres as the pair (best distance, best index) from the first
/// centre's distance, a later centre winning only when its distance is strictly smaller (`d - best < 0`), each
/// distance `sqrt` of the loop over `d` accumulating `(v - c) * (v - c)` from 0. The body is `update_centroids`: for
/// each (cluster, coordinate) cell a loop over the points carrying the TRIPLE (sum, compensation, count), stepping only
/// a point whose assignment is the cluster (`|a - j| - 1 < 0`) by `y = x - c; t = s + y; c = (t - s) - y; s = t;
/// count + 1`, then the sum over the count when the count is positive (`-count < 0`), else the previous centre.
///
/// The assignments: the tabulate over the points binds `i`; the argmin loop from 1 over the centres
/// binds the best distance `best`, its centre `bi` and the counter `c`; the first centre's distance loop over `d`
/// binds `acc` and `j`.
pub const KMEANS_ASSIGN6: &str = "(tab@a.f32[i] n (snd.f32.f32@f32 (loop1@f32.f32[best,bi,c] k (pair@f32.f32 (sqrt@f32 (loop@f32[acc,j] d 0 (add@f32 acc (mul@f32 (sub@f32 (get@f32 (row@a.f32 points i) j) (get@f32 (row@a.f32 centres zero@i64) j)) (sub@f32 (get@f32 (row@a.f32 points i) j) (get@f32 (row@a.f32 centres zero@i64) j)))))) 0)
  (psel@f32.f32 (sub@f32 DIST best) (pair@f32.f32 DIST (ival@f32 c)) (pair@f32.f32 best bi)))))";
/// The k-means seed's distance of centre `c`, its loop over `d` binding `acc2` and `j2`.
pub const KMEANS_DIST6: &str = "(sqrt@f32 (loop@f32[acc2,j2] d 0 (add@f32 acc2 (mul@f32 (sub@f32 (get@f32 (row@a.f32 points i) j2) (get@f32 (row@a.f32 centres c) j2)) (sub@f32 (get@f32 (row@a.f32 points i) j2) (get@f32 (row@a.f32 centres c) j2))))))";
/// The k-means seed's update: the rows over the clusters bind `ck`, the tabulate over the coordinates `cj`.
pub const KMEANS_UPDATE6: &str = "(tabrows@m.f32[ck] k d (tab@a.f32[cj] d (sel@f32 (neg@f32 (thd.f32.f32.f32@f32 LOOP)) (div@f32 (fst.f32.f32.f32@f32 LOOP) (thd.f32.f32.f32@f32 LOOP)) (get@f32 (row@a.f32 centres ck) cj))))";
/// The k-means seed's cell: the loop over the points binds the triple (`sum`, `comp`, `count`) and the
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

/// The k-means reference's text, made once (the problem registry holds a `&'static str`).
pub fn kmeans_reference6_static() -> &'static str {
    static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    TEXT.get_or_init(kmeans_reference6)
}

/// The k-means planted slowdown's text, made once.
pub fn kmeans_planted6_static() -> &'static str {
    static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    TEXT.get_or_init(kmeans_planted6)
}

/// THE TEXTBOOK K-MEANS STEP (the published `find_nearest_point`, then `centroids_of`; a POOL ALTERNATIVE, never the
/// seed)'s membership: the tabulate over the points binds `i`; the loop over the centres binds
/// the nearest centre so far `m` (an `f32`) and the counter `c`; each distance's tabulate over `d` binds `j`.
pub const KMEANS_TEXTBOOK_MEMBERSHIP6: &str = "(tab@a.f32[i] n (loop@f32[m,c] k 0 (sel@f32 (sub@f32 (sum@f32 (tab@a.f32[j] d (mul@f32 (sub@f32 (get@f32 (row@a.f32 points i) j) (get@f32 (row@a.f32 centres (clamp@i64 k m)) j)) (sub@f32 (get@f32 (row@a.f32 points i) j) (get@f32 (row@a.f32 centres (clamp@i64 k m)) j))))) (sum@f32 (tab@a.f32[j] d (mul@f32 (sub@f32 (get@f32 (row@a.f32 points i) j) (get@f32 (row@a.f32 centres c) j)) (sub@f32 (get@f32 (row@a.f32 points i) j) (get@f32 (row@a.f32 centres c) j)))))) m (ival@f32 c))))";

/// The textbook k-means step, with `ONES` the ones the counts are a histogram of.
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

/// EVERY REFERENCE PROGRAM, in the problem registry's order ([`PROBLEM_NAMES`]).
pub const REFERENCES6: [Reference6; 15] = [
    Reference6 { problem: "reduce", form: "reference", master6: || REDUCE_REFERENCE6.to_string() },
    Reference6 { problem: "reduce", form: "planted", master6: || REDUCE_PLANTED6.to_string() },
    Reference6 { problem: "scan", form: "reference", master6: || SCAN_REFERENCE6.to_string() },
    Reference6 { problem: "scan", form: "planted", master6: || SCAN_PLANTED6.to_string() },
    Reference6 { problem: "histogram", form: "reference", master6: || HISTOGRAM_REFERENCE6.to_string() },
    Reference6 { problem: "histogram", form: "planted", master6: || HISTOGRAM_PLANTED6.to_string() },
    Reference6 { problem: "pathfinder", form: "reference", master6: || PATHFINDER_REFERENCE6.to_string() },
    Reference6 { problem: "pathfinder", form: "planted", master6: || PATHFINDER_PLANTED6.to_string() },
    Reference6 { problem: "kmeans", form: "reference", master6: kmeans_reference6 },
    Reference6 { problem: "kmeans", form: "planted", master6: kmeans_planted6 },
    Reference6 { problem: "kmeans", form: "textbook", master6: kmeans_textbook6 },
    Reference6 { problem: "kmeans", form: "textbook_planted", master6: kmeans_textbook_planted6 },
    Reference6 { problem: "a7", form: "reference", master6: || A7_REFERENCE6.to_string() },
    Reference6 { problem: "a7", form: "winner", master6: || A7_WINNER6.to_string() },
    Reference6 { problem: "a7", form: "planted", master6: || A7_PLANTED6.to_string() },
];

/// THE EXPECTED TEXTS: every reference's stored rendering, each after a line `==== <problem>.<form> ====`, in
/// [`REFERENCES6`]'s order. The renderer reproduces each byte for byte.
pub const REFERENCES6_TEXTS: &str = include_str!("../tests/fixtures/futhark_m6_references.txt");

/// The fixture's separator line for `key`.
pub fn references6_separator(key: &str) -> String {
    format!("==== {key} ====\n")
}

/// The expected text (the stored rendering) of the reference `key` (`<problem>.<form>`).
pub fn expected_text6(key: &str) -> Option<&'static str> {
    let sep = references6_separator(key);
    let start = REFERENCES6_TEXTS.find(&sep)? + sep.len();
    let rest = &REFERENCES6_TEXTS[start..];
    Some(&rest[..rest.find("\n==== ").map_or(rest.len(), |e| e + 1)])
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
mod tests_checklist;

mod interp6;
#[cfg(test)]
mod tests_interp6;
