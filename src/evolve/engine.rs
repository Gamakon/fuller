//! Step 3 — a whole fit in Rust: no Python anywhere in the generation loop.
//!
//! One generation: the variation kernels (`vary.wgsl`) produce the next
//! population on the device; its unevaluated genes are decoded to evaluator
//! nodes, deduplicated and run over the resident rows by the evaluator
//! (`gpu_eval`); every (chromosome, linker, wrapper) candidate is linked,
//! wrapped, scaled and scored (`chrom_score`); HFF ranks the candidates over
//! train, validation and extrapolation errors; the best candidate's fitness goes
//! back to the device for the next selection.
//!
//! In this step decoding, scoring and HFF run on the host in Rust (scoring is
//! already parallel), the pump rearranges rows on the host, and the genome
//! crosses the bus once a generation. Later steps of the plan move those onto
//! the device; none of them puts Python back.

use std::collections::HashMap;
use std::time::Instant;

use super::device::EvolveDevice;
use super::score::{GpuScorer, CANDIDATES, WIDTH};
use super::vary::{GenParams, Generation, Island, Rates};
use super::{InitParams, Layout, SymbolCodes};
use crate::chrom_score::{score_chromosomes, Linker, ScoreSpec, Splits, Wrapper, METRIC_WIDTH, SCORE_WIDTH};
use crate::gpu_eval::{ExprBatch, GpuEvaluator, GpuNode, Op, MAX_NODES};

/// What a token id means to the evaluator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Symbol {
    Function(Op),
    /// A data column.
    Input(u32),
    /// A fixed numeric terminal.
    Constant(f32),
    /// The "?" placeholder: the n-th one in a gene's expression reads
    /// `rnc[dc[n]]` — geppy's Dc domain.
    Rnc,
}

#[derive(Clone, Debug)]
pub struct SymbolTable {
    pub symbols: Vec<Symbol>,
    /// Terminals that may be carried but never drawn (named constants).
    pub withheld: Vec<bool>,
}

impl SymbolTable {
    /// The SRBench entry's wide primitive set over `n_inputs` columns, plus "?".
    pub fn wide(n_inputs: u32) -> SymbolTable {
        let functions = [
            Op::Add, Op::Sub, Op::Mul, Op::ProtectedDiv, Op::Sin, Op::Cos, Op::Tanh, Op::ProtectedSqrt,
            Op::ProtectedLog, Op::ProtectedExp, Op::Neg, Op::Abs, Op::Pow2, Op::Pow3, Op::ProtectedInv,
            Op::Tan, Op::ProtectedAsin, Op::ProtectedAcos,
        ];
        let mut symbols: Vec<Symbol> = functions.iter().map(|&op| Symbol::Function(op)).collect();
        symbols.extend((0..n_inputs).map(Symbol::Input));
        symbols.push(Symbol::Rnc);
        let withheld = vec![false; symbols.len()];
        SymbolTable { symbols, withheld }
    }

    pub fn arity(&self, id: u32) -> u32 {
        match self.symbols[id as usize] {
            Symbol::Function(op) => op.arity() as u32,
            _ => 0,
        }
    }

    pub fn codes(&self) -> SymbolCodes {
        let ids = 0..self.symbols.len() as u32;
        SymbolCodes {
            arity: ids.clone().map(|i| self.arity(i)).collect(),
            sample_functions: ids.clone().filter(|&i| self.arity(i) > 0).collect(),
            sample_terminals: ids.filter(|&i| self.arity(i) == 0 && !self.withheld[i as usize]).collect(),
            rnc_id: self.symbols.iter().position(|s| *s == Symbol::Rnc).map(|i| i as u32),
        }
    }

    pub fn max_arity(&self) -> u32 {
        (0..self.symbols.len() as u32).map(|i| self.arity(i)).max().unwrap_or(1)
    }
}

/// A gene's expressed tree as evaluator nodes, in level order (so every child
/// index is greater than its parent's, which is what the evaluator needs).
/// `None`: the expression does not close, or a "?" has no Dc entry.
pub fn decode_gene(gene: &[u32], rnc: &[f32], layout: Layout, table: &SymbolTable) -> Option<Vec<GpuNode>> {
    let ht = (layout.head + layout.tail) as usize;
    let (mut need, mut n) = (1i64, 0usize);
    while need > 0 && n < ht {
        need += i64::from(table.arity(gene[n])) - 1;
        n += 1;
    }
    if need > 0 {
        return None;
    }
    let mut nodes = Vec::with_capacity(n);
    let (mut child, mut n_rnc) = (1u32, 0usize);
    for &id in &gene[..n] {
        let node = match table.symbols[id as usize] {
            Symbol::Function(op) => {
                let a = op.arity() as u32;
                let node = GpuNode { op: op as u32, arg0: child, arg1: if a == 2 { child + 1 } else { 0 }, konst: 0.0 };
                child += a;
                node
            }
            Symbol::Input(col) => GpuNode { op: Op::Var as u32, arg0: col, arg1: 0, konst: 0.0 },
            Symbol::Constant(v) => GpuNode { op: Op::Num as u32, arg0: 0, arg1: 0, konst: v },
            Symbol::Rnc => {
                let k = *gene.get(ht + n_rnc)? as usize;
                n_rnc += 1;
                GpuNode { op: Op::Num as u32, arg0: 0, arg1: 0, konst: *rnc.get(k)? }
            }
        };
        nodes.push(node);
    }
    Some(nodes)
}

/// The nodes as a fuller `Math` expression, inputs named `names[col]`.
pub fn nodes_to_math(nodes: &[GpuNode], at: usize, names: &[String]) -> String {
    let node = nodes[at];
    if node.op == Op::Var as u32 {
        return format!("(Var \"{}\")", names[node.arg0 as usize]);
    }
    if node.op == Op::Num as u32 {
        return format!("(Num {:?})", f64::from(node.konst));
    }
    let op = OPS.iter().find(|op| **op as u32 == node.op).copied().unwrap_or(Op::Add);
    let first = nodes_to_math(nodes, node.arg0 as usize, names);
    if op.arity() == 2 {
        format!("({op:?} {first} {})", nodes_to_math(nodes, node.arg1 as usize, names))
    } else {
        format!("({op:?} {first})")
    }
}

const OPS: [Op; 26] = [
    Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Neg, Op::Abs, Op::Sqrt, Op::Log, Op::Exp, Op::Sin, Op::Cos,
    Op::Tan, Op::Tanh, Op::Pow, Op::Pow2, Op::Pow3, Op::Inv, Op::ProtectedDiv, Op::ProtectedSqrt,
    Op::ProtectedLog, Op::ProtectedExp, Op::ProtectedInv, Op::Asin, Op::Acos, Op::ProtectedAsin,
    Op::ProtectedAcos,
];

/// The data of one fit: rows are train, then validation, then edge
/// (extrapolation) — the order `chrom_score` expects.
pub struct Data {
    pub names: Vec<String>,
    /// Row-major, `names.len()` wide.
    pub x: Vec<f32>,
    pub y: Vec<f64>,
    pub splits: Splits,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u32,
    pub pop_intake: u32,
    pub pop_champion: u32,
    pub tournament_fraction: f64,
    pub elites: u32,
    pub n_genes: u32,
    pub head: u32,
    pub n_rnc: u32,
    pub rnc_lo: i32,
    pub rnc_hi: i32,
    pub pump_every: u32,
    /// The cleansing mutation's rate per row (0 = off).
    pub cleanse: f64,
    /// Redundancy as an HFF objective: the scoring kernel's leave-one-gene-out
    /// score in [0, 1] (0 = every varying gene carries part of the fit; towards 1
    /// = genes whose removal costs nothing). Measured on finished models, a law's
    /// parts each cost a quarter to a half of the fit and a refined-noise model's
    /// typical part 0.3%. Off by default until an A/B keeps it.
    pub redundancy: bool,
    /// The third block of rows is SMOGD's: synthetic and noisy on purpose. Its
    /// errors rank individuals in the tournaments (three HFF objectives); it never
    /// decides that a fit is exact — the stop bar stays on the real validation rows.
    pub smogd: bool,
    /// Which blocks enter HFF on the log scale — see `hff_truenorth` — in the
    /// order train, validation, third (SMOGD / SMOTE / edge). Each is its own
    /// choice: validation may be log while the noisy third block stays linear.
    pub log_scale: [bool; 3],
    /// Block two (validation) is left OUT of HFF: a random cut of the same rows as
    /// train, its errors mirror train's and only dilute the third block's. HFF is
    /// then train + block three (hff's `METRIC_NAMES_TRAIN_ONLY`). Validation
    /// still decides the stop bar and is still reported.
    pub hff_without_validation: bool,
    /// The TOWER objective: [`tower_penalty`] of the chromosome's [`t_depth`] joins
    /// HFF, so a tournament prefers the individual that is not a tower of nested
    /// functions when the errors cannot tell them apart.
    pub tower: bool,
    /// Report the best individual on stderr every this many generations (0 = never):
    /// a long fit is watched as it runs, not read when it ends.
    pub progress_every: u32,
    /// THE GROWING HEAD. The population starts with a virtual head of `vhead_start`
    /// and gains one position every `vhead_every` generations, up to the physical
    /// head: short expressions are searched first, and a gene that is already good
    /// keeps evolving as the room grows — raising the virtual head changes no
    /// expression. `vhead_every` = 0: off, the whole head from the start.
    pub vhead_start: u32,
    pub vhead_every: u32,
    /// THE HALL OF FAME's file: at every progress report the best individual the
    /// fit has EVER held is appended to it — generation found, scores, and the
    /// model as plain infix. None = no file (the hall of fame is still kept).
    pub hof_path: Option<String>,
    /// BALANCED-POLE TOURNAMENTS, for diversity. Selection (the tournaments, and
    /// with them the pump's promotions) ranks on hff's BALANCED pole — the angle
    /// from (1/sqrt m, ..) — which rewards even trade-offs and so keeps individuals
    /// alive that TrueNorth would drop. Everything that JUDGES a model stays on
    /// TrueNorth: the hall of fame, the stop bar, the report. The balanced pole is
    /// banned as a fitness (it scores a uniformly mediocre point as perfect); here
    /// it only chooses who breeds. Off by default.
    pub balanced_tournaments: bool,
    pub max_generations: u32,
    pub max_seconds: f64,
    /// Stop when validation (and edge, when there is one) 1 - R² is this small.
    pub stop_one_minus_r2: f64,
}

impl Config {
    /// The SRBench entry's settings.
    pub fn srbench(seed: u32) -> Config {
        Config {
            seed,
            pop_intake: 600,
            pop_champion: 200,
            tournament_fraction: 0.07,
            elites: 2,
            n_genes: 3,
            // 34: the longest of SRBench's true laws needs a head of 29 written whole in
            // ONE gene (median 8, 90% within 17); 48 only left room for towers.
            head: 34,
            n_rnc: 10,
            rnc_lo: -100,
            rnc_hi: 100,
            pump_every: 4,
            cleanse: 0.0,
            redundancy: false,
            smogd: false,
            log_scale: [false; 3],
            hff_without_validation: false,
            tower: false,
            progress_every: 0,
            vhead_start: 12,
            vhead_every: 0,
            hof_path: None,
            balanced_tournaments: false,
            // Kept after a two-seed A/B (7012: 46 -> 47, 7013: 44 -> 45, no losses).
            max_generations: 1500,
            max_seconds: 30.0,
            stop_one_minus_r2: 1e-10,
        }
    }
}

/// The best candidate of an individual.
#[derive(Clone, Copy, Debug)]
pub struct Scored {
    pub fitness: f64,
    pub linker: usize,
    pub wrapper: usize,
    pub a: f64,
    pub b: f64,
    /// 1 - R² on train, validation, edge.
    pub one_minus_r2: [f64; 3],
    /// The chromosome's tower height: [`t_depth`], the largest over its genes.
    pub t_depth: u32,
    /// What the TOURNAMENTS rank on: `fitness` (TrueNorth), or the balanced-pole
    /// angle when `Config::balanced_tournaments` is on.
    pub selection: f64,
}

/// THE HALL OF FAME: the best individual a fit has ever held (lowest HFF), when it
/// appeared, and what it computes. The scores are the device's f32 ranking scores.
#[derive(Clone, Debug)]
pub struct HallOfFame {
    pub generation: u32,
    pub best: Scored,
    pub math: String,
    /// The winner itself — its genes, constants and wrapper — so it can be
    /// confirmed in f64 and reported even after it has left the population
    /// (under balanced tournaments the TrueNorth best is not an elite).
    pub genome: Vec<u32>,
    pub rnc: Vec<f32>,
    pub wrapper_id: u32,
}

#[derive(Clone, Debug)]
pub struct Timing {
    pub vary: f64,
    pub read: f64,
    pub decode: f64,
    pub evaluate: f64,
    pub score: f64,
    pub hff: f64,
    pub pump: f64,
}

pub struct FitResult {
    pub generations: u32,
    pub seconds: f64,
    pub individuals: u64,
    pub unique_genes: u64,
    pub oversized_genes: u64,
    pub stopped_by: &'static str,
    pub best: Scored,
    pub math: String,
    pub timing: Timing,
}

const LINKERS: [Linker; 3] = [Linker::AVG, Linker::MUL, Linker::ADD];
const WRAPPERS: [Wrapper; 3] = [Wrapper::Identity, Wrapper::LogAbs, Wrapper::SqrtAbs];
const LINKER_NAMES: [&str; 3] = ["avgval", "mulval", "addval"];

/// An HFF angle as a P-VALUE: the probability that a point drawn uniformly on the
/// `m`-objective hypersphere lies within `theta` of the pole — hff's own
/// `higd::cdf_beta_correction`, `I_{sin² θ}((m-1)/2, 1/2)`. Returned as
/// `(p, log10 p)`; the log comes from hff's log-space routine, so it stays
/// finite in the deep left tail where `p` itself underflows to 0.
pub fn hff_p_value(theta: f64, m: usize) -> (f64, f64) {
    let p = hff_core::higd::cdf_beta_correction(theta, m);
    (p, hff_core::higd::log_cdf_beta_correction(theta, m) / std::f64::consts::LN_10)
}

/// TRANSCENDENTAL NESTING DEPTH of a decoded gene: the most transcendental
/// functions met on any path from the root to a leaf — exp, log, sin, cos, tan,
/// tanh, asin, acos, Abs, sqrt (and their protected forms), and a `Pow` whose exponent is not
/// a whole number. `+ - * /`, negation, `1/x` and whole powers do not count, so a
/// long FLAT law scores 0: this is not parsimony. Measured on 1,230 labelled fits
/// (`docs/BRAINSTORM_tower_detector.md`): every one of SRBench's 133 true laws is
/// at most 2; three quarters of the accurate-but-wrong models are 3 or more.
///
/// One forward scan: a gene's nodes are in level order, so a node's depth is
/// known before its children are reached.
pub fn t_depth(nodes: &[GpuNode]) -> u32 {
    let counts = |i: usize| -> u32 {
        let n = &nodes[i];
        let whole = |c: usize| nodes.get(c).is_some_and(|e| e.op == Op::Num as u32 && e.konst.fract() == 0.0);
        let t = [
            Op::Abs, Op::Sqrt, Op::Log, Op::Exp, Op::Sin, Op::Cos, Op::Tan, Op::Tanh, Op::ProtectedSqrt, Op::ProtectedLog,
            Op::ProtectedExp, Op::Asin, Op::Acos, Op::ProtectedAsin, Op::ProtectedAcos,
        ];
        u32::from(t.iter().any(|&o| o as u32 == n.op) || (n.op == Op::Pow as u32 && !whole(n.arg1 as usize)))
    };
    if nodes.is_empty() {
        return 0;
    }
    let binary = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Pow, Op::ProtectedDiv];
    let leaf = [Op::Var, Op::Num];
    let mut depth = vec![0u32; nodes.len()];
    depth[0] = counts(0);
    for i in 0..nodes.len() {
        let n = &nodes[i];
        if leaf.iter().any(|&o| o as u32 == n.op) {
            continue;
        }
        let kids = if binary.iter().any(|&o| o as u32 == n.op) { vec![n.arg0, n.arg1] } else { vec![n.arg0] };
        for k in kids.into_iter().map(|k| k as usize).filter(|&k| k < nodes.len()) {
            depth[k] = depth[i] + counts(k);
        }
    }
    depth.into_iter().max().unwrap_or(0)
}

/// The tower objective on [0, 1]: a FREE ZONE up to a depth of 2, then a quarter
/// per level, 1 from a depth of 6. The free zone is room to explore: a law may
/// need its sqrt or its log (every SRBench true law is at most 2 deep), and
/// charged from depth 0 the search settles for a flat polynomial instead — it
/// rebuilt m c^2 + m v^2 / 2 for m c^2 / sqrt(1 - v^2/c^2) and never tried a root.
pub fn tower_penalty(t_depth: u32) -> f64 {
    (f64::from(t_depth.saturating_sub(2)) / 4.0).min(1.0)
}

/// Which of the nine objectives `[mse x3, 1-R2 x3, mae x3]` (blocks train,
/// validation, third) feed HFF, and which of those are log-scaled.
fn hff_columns(n_extrap: usize, without_validation: bool, log_scale: [bool; 3]) -> Vec<(usize, bool)> {
    let mut columns = Vec::new();
    for metric in 0..3 {
        for (block, &log) in log_scale.iter().enumerate() {
            let absent = (block == 2 && n_extrap == 0) || (block == 1 && without_validation);
            if !absent {
                columns.push((3 * metric + block, log));
            }
        }
    }
    columns
}

/// The logbook's header: one row per report under it (min and avg are HFF fitness,
/// lower is fitter; the R² are the best individual's).
const REPORT_HEADER: &str = "    gen    secs  head      min_hff      avg_hff    mse_train       r2_train     r2_val_bl2     r2_val_bk3  t_depth   log10_p";

/// The objectives as HFF sees them: each on [0, 1] by its frozen range, the
/// log-scaled ones stretched. None when one is not finite.
fn hff_scaled(objectives: &[f64], col_max: &[f64], log_scaled: &[bool]) -> Option<Vec<f64>> {
    let mut scaled = Vec::with_capacity(objectives.len());
    for ((v, max), log) in objectives.iter().zip(col_max).zip(log_scaled) {
        if !v.is_finite() {
            return None;
        }
        let mut x = if *max > 0.0 { (v / max).min(1.0) } else { 0.0 };
        if *log {
            x = if x <= HFF_LOG_FLOOR { 0.0 } else { 1.0 + x.log10() / -HFF_LOG_FLOOR.log10() };
        }
        scaled.push(x);
    }
    Some(scaled)
}

/// The same objectives seen from hff's BALANCED pole (its own function, not a
/// copy): the angle from (1/sqrt m, .., 1/sqrt m). For choosing who breeds ONLY —
/// see `Config::balanced_tournaments`.
fn hff_balanced(objectives: &[f64], col_max: &[f64], log_scaled: &[bool]) -> f64 {
    let Some(scaled) = hff_scaled(objectives, col_max, log_scaled) else { return std::f64::consts::PI };
    let m = scaled.len();
    hff_core::core_functions::calculate_single_hyperspherical_fitness_f64_with_method(&ndarray::Array1::from(scaled), m, false, None, "balanced")
}

/// The scaled error that HFF's log scale calls zero.
const HFF_LOG_FLOOR: f64 = 1e-12;

/// HFF, TrueNorth: objectives scaled into [0, 1] by frozen ranges, the angle
/// from the all-zero pole — `acos(1 - min(sum(x^2) / m, 1))` (hff_core
/// `true_north_cos_theta`: with the pole at (0, .., 0, 1) the cosine is the
/// energy score alone).
///
/// An objective marked `log_scaled` is first stretched, `1 + log10(x) / 12`
/// (so 1e-12 and below is 0, 1 stays 1): every factor of ten in the error counts
/// the same. Squared as it stands, an error of 5e-3 weighs 3e-5 and one of 2e-4
/// weighs 4e-8 — both nothing, and a tournament cannot tell a fake at 1e-3 from a
/// law at 1e-12. On the log scale they sit at 0.75 and 0.
fn hff_truenorth(objectives: &[f64], col_max: &[f64], log_scaled: &[bool]) -> f64 {
    let m = objectives.len() as f64;
    let Some(scaled) = hff_scaled(objectives, col_max, log_scaled) else { return std::f64::consts::PI };
    let energy: f64 = scaled.iter().map(|x| x * x).sum();
    let cos_theta = (1.0 - (energy / m).min(1.0)).clamp(-1.0, 1.0);
    if cos_theta > 1.0 - f64::EPSILON {
        0.0
    } else {
        cos_theta.acos()
    }
}

struct Caps {
    /// The constant model's error on each split: var(y), and mean |y - mean|.
    var: [f64; 3],
    mad: [f64; 3],
}

impl Caps {
    fn of(y: &[f64], s: Splits) -> Caps {
        let ranges = [(0, s.n_train), (s.n_train, s.n_train + s.n_val), (s.n_train + s.n_val, s.total())];
        let (mut var, mut mad) = ([0.0; 3], [0.0; 3]);
        for (k, (lo, hi)) in ranges.into_iter().enumerate() {
            if hi > lo {
                let part = &y[lo..hi];
                let mean = part.iter().sum::<f64>() / part.len() as f64;
                var[k] = part.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / part.len() as f64;
                mad[k] = part.iter().map(|v| (v - mean).abs()).sum::<f64>() / part.len() as f64;
            }
        }
        Caps { var, mad }
    }

    /// `[mse x3, 1-R² x3, mae x3]`, each capped at the constant model's error: a
    /// model worse than predicting the mean is useless, and how much worse
    /// carries no information (uncapped, one exploding individual froze a range
    /// at 1e27 and the objective stopped discriminating).
    fn objectives(&self, s: &[f64], n_extrap: usize) -> ([f64; 9], [f64; 3]) {
        let mse = [s[2], s[3], s[5]];
        let mae = [s[6], s[7], s[8]];
        let mut o = [0.0; 9];
        let mut omr2 = [0.0; 3];
        for k in 0..3 {
            if k == 2 && n_extrap == 0 {
                continue;
            }
            omr2[k] = if self.var[k] > 0.0 { mse[k] / self.var[k] } else { f64::INFINITY };
            o[k] = mse[k].min(self.var[k]);
            o[3 + k] = omr2[k].min(1.0);
            o[6 + k] = mae[k].min(self.mad[k]);
        }
        (o, omr2)
    }
}

/// DATA GUIDED REWRITES (Andrew's name for them): rewrites fuller's final form may
/// make because the DATA says they hold on every row — never identities in
/// general, always exact on the rows the model was selected on, and checked: the
/// rewritten model must predict what the original predicts.
///
/// `Abs e` where `e` keeps one sign on every row becomes `e` or `Neg e`, and
/// every protected operator the DATA never triggers becomes the raw one:
/// `ProtectedDiv a b` with |b| >= 1e-6 on every row is `Div a b`, with |b| < 1e-6
/// on every row it is 0; `ProtectedInv x` with x never 0 is `Inv x`;
/// `ProtectedAsin x` / `ProtectedAcos x` with x in [-1, 1] on every row are
/// `Asin x` / `Acos x`. One that is
/// triggered on SOME rows stays protected, and `Tree::to_infix_faithful` writes
/// it out as the Piecewise it is. Decided on `rows`, the rows the model was
/// selected on — the engine's counterpart of hff's `symbolic_protected_div`.
pub fn resolve_protected(math: &str, rows: &[Vec<(String, f64)>]) -> Result<String, String> {
    use crate::lint::node::Tree;
    fn has_input(t: &Tree) -> bool {
        match t {
            Tree::Var(_) => true,
            Tree::Num(_) => false,
            Tree::App(_, kids) => kids.iter().any(has_input),
        }
    }
    /// A subtree that takes ONE value on every row is that constant: a dead term,
    /// however it is built — a clamp that always fires, |asin| of an argument that
    /// is beyond +-1 with either sign (strogatz glider2), x/x.
    fn constant_on_data(t: &Tree, rows: &[Vec<(String, f64)>]) -> Option<Tree> {
        if !matches!(t, Tree::App(..)) || !has_input(t) {
            return None;
        }
        let v = evaluate_math(&t.to_math(), rows).ok()?;
        let (lo, hi) = v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| (lo.min(*x), hi.max(*x)));
        let steady = !v.is_empty() && v.iter().all(|x| x.is_finite()) && hi - lo <= 1e-12 * lo.abs().max(hi.abs()).max(1.0);
        steady.then(|| Tree::Num((lo + hi) / 2.0))
    }
    fn go(t: &Tree, rows: &[Vec<(String, f64)>]) -> Tree {
        let rewritten = specific(t, rows);
        constant_on_data(&rewritten, rows).unwrap_or(rewritten)
    }
    fn specific(t: &Tree, rows: &[Vec<(String, f64)>]) -> Tree {
        let Tree::App(op, kids) = t else { return t.clone() };
        let kids: Vec<Tree> = kids.iter().map(|k| go(k, rows)).collect();
        let values = |k: &Tree| evaluate_math(&k.to_math(), rows).unwrap_or_default();
        match op {
            Op::ProtectedDiv => {
                let b = values(&kids[1]);
                if !b.is_empty() && b.iter().all(|v| v.is_finite() && v.abs() >= 1e-6) {
                    return Tree::App(Op::Div, kids);
                }
                if !b.is_empty() && b.iter().all(|v| v.abs() < 1e-6) {
                    return Tree::Num(0.0);
                }
            }
            Op::ProtectedInv => {
                let a = values(&kids[0]);
                if !a.is_empty() && a.iter().all(|v| v.is_finite() && *v != 0.0) {
                    return Tree::App(Op::Inv, kids);
                }
            }
            // sqrt|x| unless x is not finite (then 0): on data where the argument
            // is always finite it IS sqrt(Abs(x)), and must be written so — left
            // protected it prints as a Piecewise no scorer can match to a law.
            Op::ProtectedSqrt => {
                let a = values(&kids[0]);
                if !a.is_empty() && a.iter().all(|v| v.is_finite()) {
                    // ... and the Abs goes too where the argument keeps one sign.
                    return Tree::App(Op::Sqrt, vec![go(&Tree::App(Op::Abs, kids), rows)]);
                }
            }
            // asin / acos of the argument clamped to [-1, 1]: where the argument
            // never leaves [-1, 1] the clamp does nothing and it IS the raw function
            // (feynman I.26.2, asin(n sin t), is reported as the law and not as a
            // Piecewise over a Min and a Max).
            // An inverse of its own function is LINEAR in the argument on whichever
            // branch the data keeps the argument in: on [k pi, (k+1) pi], acos(cos e)
            // is e - k pi (k even) or (k+1) pi - e (k odd); on [k pi - pi/2, k pi +
            // pi/2], asin(sin e) is (-1)^k (e - k pi). (strogatz barmag2 hid its law
            // behind acos(cos x) with x in [pi, 2 pi] on every row: 2 pi - x.)
            Op::ProtectedAsin | Op::ProtectedAcos | Op::Asin | Op::Acos
                if matches!((op, &kids[0]), (Op::ProtectedAsin | Op::Asin, Tree::App(Op::Sin, _)) | (Op::ProtectedAcos | Op::Acos, Tree::App(Op::Cos, _))) =>
            {
                use std::f64::consts::{FRAC_PI_2, PI};
                let Tree::App(_, inner) = &kids[0] else { return Tree::App(*op, kids) };
                let e = values(&inner[0]);
                let asin = matches!(op, Op::ProtectedAsin | Op::Asin);
                if let Some(first) = e.first().filter(|v| v.is_finite()) {
                    // the branch the first row is on; every row must be on it
                    let k = if asin { (first / PI).round() } else { (first / PI).floor() };
                    let (lo, hi) = if asin { (k * PI - FRAC_PI_2, k * PI + FRAC_PI_2) } else { (k * PI, (k + 1.0) * PI) };
                    if k.abs() <= 1e6 && e.iter().all(|v| v.is_finite() && *v >= lo && *v <= hi) {
                        let arg = inner[0].clone();
                        let odd = k.rem_euclid(2.0) == 1.0;
                        return match (asin, odd, k == 0.0) {
                            (_, _, true) => arg,
                            (true, false, _) => Tree::App(Op::Sub, vec![arg, Tree::Num(k * PI)]),
                            (true, true, _) => Tree::App(Op::Sub, vec![Tree::Num(k * PI), arg]),
                            (false, false, _) => Tree::App(Op::Sub, vec![arg, Tree::Num(k * PI)]),
                            (false, true, _) => Tree::App(Op::Sub, vec![Tree::Num((k + 1.0) * PI), arg]),
                        };
                    }
                }
                if matches!(op, Op::ProtectedAsin | Op::ProtectedAcos) {
                    // sin and cos never leave [-1, 1]: the clamp cannot fire.
                    return Tree::App(if *op == Op::ProtectedAsin { Op::Asin } else { Op::Acos }, kids);
                }
            }
            Op::ProtectedAsin | Op::ProtectedAcos => {
                let a = values(&kids[0]);
                let asin = *op == Op::ProtectedAsin;
                if !a.is_empty() && a.iter().all(|v| v.is_finite() && v.abs() <= 1.0) {
                    return Tree::App(if asin { Op::Asin } else { Op::Acos }, kids);
                }
                // The clamp fires on EVERY row: the node is a constant — asin(1), or
                // asin(-1) — and must be written as one, as an always-zero protected
                // division is. Left as arcsin(x) it hides an exact law behind a dead
                // term (feynman III.12.43 came out as 0.159*h*n + 0.159*arcsin(n) - 0.25
                // with n > 1 on every row: 0.159 * pi/2 = 0.25, the law exactly).
                use std::f64::consts::{FRAC_PI_2, PI};
                if !a.is_empty() && a.iter().all(|v| v.is_finite() && *v >= 1.0) {
                    return Tree::Num(if asin { FRAC_PI_2 } else { 0.0 });
                }
                if !a.is_empty() && a.iter().all(|v| v.is_finite() && *v <= -1.0) {
                    return Tree::Num(if asin { -FRAC_PI_2 } else { PI });
                }
            }
            // |e| where e keeps ONE SIGN on every row is e, or -e. Left as Abs, an
            // exact law is reported in a form no scorer matches to it (feynman
            // II.11.27: 3*n*eps*Ef / Abs(n - 3/alpha), the law but for the Abs).
            Op::Abs => {
                let a = values(&kids[0]);
                if !a.is_empty() && a.iter().all(|v| v.is_finite() && *v >= 0.0) {
                    return kids[0].clone();
                }
                if !a.is_empty() && a.iter().all(|v| v.is_finite() && *v <= 0.0) {
                    return Tree::App(Op::Neg, kids);
                }
            }
            _ => {}
        }
        Tree::App(*op, kids)
    }
    let rewritten = go(&Tree::parse(math)?, rows).to_math();
    // CHECKED: a data guided rewrite may not move the model's predictions. Where
    // the original is finite on every row, the rewritten form must agree with it
    // to FINAL_FORM_AGREE; if it does not, the original stands.
    let (before, after) = (evaluate_math(math, rows)?, evaluate_math(&rewritten, rows)?);
    if before.iter().all(|v| v.is_finite()) && !before.is_empty() {
        let n = before.len() as f64;
        let mean = before.iter().sum::<f64>() / n;
        let var = before.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let drift = before.iter().zip(&after).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n;
        if drift.is_nan() || drift > FINAL_FORM_AGREE * var.max(f64::MIN_POSITIVE) {
            return Ok(math.to_string());
        }
    }
    Ok(rewritten)
}

/// How closely a tidied form must predict what the model predicts: mean squared
/// difference over the rows, relative to the model's variance.
pub const FINAL_FORM_AGREE: f64 = 1e-10;

/// fuller tidies the reported model, the data as judge: the linter is told which
/// inputs are positive / non-zero on every row and given rows to judge prunes
/// on; every candidate form is executed on ALL of `rows`, and the smallest one
/// that predicts what `math` predicts to [`FINAL_FORM_AGREE`] is the final form.
/// A snap that moves a constant by 1e-4 does not pass that bar — it is the
/// reporting harness's decision, judged on its own terms, not this function's.
/// Returns `math` itself when nothing smaller agrees.
pub fn final_form(math: &str, names: &[String], rows: &[Vec<(String, f64)>]) -> Result<String, String> {
    use crate::lint::tables::{Exactness, Tables};
    static TABLES: std::sync::OnceLock<Result<Tables, String>> = std::sync::OnceLock::new();
    let tables = TABLES.get_or_init(Tables::standard).as_ref().map_err(|e| format!("lint tables: {e}"))?;
    let reference = evaluate_math(math, rows)?;
    let n = reference.len() as f64;
    let mean = reference.iter().sum::<f64>() / n;
    let var = reference.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    // NaN counts as "not usable" on both tests.
    if var.is_nan() || var <= 0.0 || reference.iter().any(|v| !v.is_finite()) {
        return Ok(math.to_string());
    }
    let all = |test: &dyn Fn(f64) -> bool| -> Vec<String> {
        names.iter().filter(|name| rows.iter().all(|r| r.iter().any(|(k, v)| k == *name && test(*v)))).cloned().collect()
    };
    let facts = crate::lint::DataFacts {
        rows: &rows[..rows.len().min(256)],
        positive_vars: all(&|v| v > 0.0),
        nonzero_vars: all(&|v| v != 0.0),
    };
    let candidates = crate::lint::forms(tables, math, names, Exactness::Finite, 8, facts)?;
    let mut best: Option<(usize, String, String)> = None;
    for (tree, _) in candidates {
        let form = tree.to_math();
        let Ok(pred) = evaluate_math(&form, rows) else { continue };
        let drift = pred.iter().zip(&reference).map(|(p, r)| (p - r).powi(2)).sum::<f64>() / n / var;
        if drift.is_nan() || drift > FINAL_FORM_AGREE {
            continue;
        }
        let key = (tree.node_count(), tree.to_infix(), form);
        if best.as_ref().is_none_or(|b| (key.0, &key.1) < (b.0, &b.1)) {
            best = Some(key);
        }
    }
    Ok(best.map_or(math.to_string(), |b| b.2))
}

/// A `Math` expression on rows of named values, in f64, by fuller's own
/// evaluator: how a model is scored on data the search never saw.
pub fn evaluate_math(math: &str, rows: &[Vec<(String, f64)>]) -> Result<Vec<f64>, String> {
    crate::extract::eval_expr_rows(math, rows)
}

pub struct Engine {
    pub config: Config,
    pub table: SymbolTable,
    pub layout: Layout,
    pub islands: Vec<Island>,
    dev: EvolveDevice,
    evaluator: GpuEvaluator,
    scorer: GpuScorer,
    data: Data,
    caps: Caps,
    col_max: Option<[f64; 9]>,
    scored: Vec<Option<Scored>>,
}

impl Engine {
    pub fn new(config: Config, data: Data) -> Result<Engine, String> {
        let table = SymbolTable::wide(data.names.len() as u32);
        let pop = config.pop_intake + config.pop_champion;
        let layout = Layout::for_arity(pop, config.n_genes, config.head, table.max_arity(), config.n_rnc);
        let tourn = |n: u32| ((config.tournament_fraction * f64::from(n)).round() as u32).max(2);
        let islands = vec![
            Island { lo: 0, hi: config.pop_intake, elites: config.elites, tournsize: tourn(config.pop_intake) },
            Island { lo: config.pop_intake, hi: pop, elites: config.elites, tournsize: tourn(config.pop_champion) },
        ];
        super::vary::validate(layout, &islands)?;
        if data.y.len() != data.splits.total() || data.x.len() != data.y.len() * data.names.len() {
            return Err("data: x, y and the splits do not agree".into());
        }
        let dev = EvolveDevice::new(layout, &table.codes())?;
        let evaluator = GpuEvaluator::new(&data.x, data.names.len())?;
        let scorer = GpuScorer::new(&evaluator, &data.y, data.splits)?;
        let caps = Caps::of(&data.y, data.splits);
        Ok(Engine { scored: vec![None; pop as usize], config, table, layout, islands, dev, evaluator, scorer, data, caps, col_max: None })
    }

    /// Score every unevaluated row of `gen`; returns (unique genes, oversized).
    fn evaluate(&mut self, gen: &mut Generation, timing: &mut Timing) -> Result<(u64, u64), String> {
        let l = self.layout;
        let (width, nr, g_n) = (l.gene_width() as usize, l.n_rnc as usize, l.n_genes as usize);
        let t = Instant::now();
        let rows: Vec<usize> = (0..l.pop as usize).filter(|&r| gen.fitness[r].is_nan()).collect();
        let mut index: HashMap<Vec<u32>, usize> = HashMap::new();
        let mut batch = ExprBatch::new();
        let mut gene_ok: Vec<bool> = Vec::new();
        let mut gene_tower: Vec<u32> = Vec::new();
        let mut chromosomes: Vec<Vec<usize>> = Vec::with_capacity(rows.len());
        let mut oversized = 0u64;
        for &r in &rows {
            let mut genes = Vec::with_capacity(g_n);
            for g in 0..g_n {
                let tokens = &gen.pop.genome[(r * g_n + g) * width..(r * g_n + g + 1) * width];
                let consts = &gen.pop.rnc[(r * g_n + g) * nr..(r * g_n + g + 1) * nr];
                let nodes = decode_gene(tokens, consts, l, &self.table);
                let fits = nodes.as_ref().is_some_and(|n| n.len() <= MAX_NODES);
                let signature: Vec<u32> = match &nodes {
                    Some(n) if fits => n.iter().flat_map(|x| [x.op, x.arg0, x.arg1, x.konst.to_bits()]).collect(),
                    _ => vec![u32::MAX, (r * g_n + g) as u32],
                };
                let next = gene_ok.len();
                let at = *index.entry(signature).or_insert(next);
                if at == next {
                    if fits {
                        batch.push(nodes.as_deref().unwrap_or(&[]));
                    } else {
                        // keeps gene indices and batch slots aligned
                        batch.push(&[GpuNode { op: Op::Num as u32, arg0: 0, arg1: 0, konst: 0.0 }]);
                        oversized += u64::from(nodes.is_some());
                    }
                    gene_ok.push(fits);
                    gene_tower.push(nodes.as_deref().map_or(0, t_depth));
                }
                genes.push(at);
            }
            chromosomes.push(genes);
        }
        timing.decode += t.elapsed().as_secs_f64();

        // Evaluate and score in one trip: the predictions stay on the device and
        // only the candidates' metrics come back (f32 — they RANK; see `confirm`).
        let t = Instant::now();
        let scores: Vec<f64> = if rows.is_empty() {
            Vec::new()
        } else {
            self.scorer.eval_and_score(&self.evaluator, &batch, &gene_ok, &chromosomes)?.into_iter().map(f64::from).collect()
        };
        timing.evaluate += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let per = CANDIDATES;
        let n_ex = self.data.splits.n_extrap;
        if self.col_max.is_none() {
            // Frozen at the first evaluation, as the engine freezes them at
            // generation 0: the largest capped value of each objective.
            let mut max = [0.0f64; 9];
            for c in scores.chunks(WIDTH).filter(|c| c[0].is_finite()) {
                let (o, _) = self.caps.objectives(c, n_ex);
                for k in 0..9 {
                    max[k] = max[k].max(o[k]);
                }
            }
            self.col_max = Some(max);
        }
        let col_max = self.col_max.unwrap_or([1.0; 9]);
        let columns = hff_columns(n_ex, self.config.hff_without_validation, self.config.log_scale);
        for (i, &r) in rows.iter().enumerate() {
            let mut best: Option<Scored> = None;
            let tower = chromosomes[i].iter().map(|&g| gene_tower[g]).max().unwrap_or(0);
            for c in 0..per {
                let s = &scores[(i * per + c) * WIDTH..(i * per + c + 1) * WIDTH];
                if !s[0].is_finite() {
                    continue;
                }
                let (o, omr2) = self.caps.objectives(s, n_ex);
                let mut used: Vec<f64> = columns.iter().map(|&(k, _)| o[k]).collect();
                let mut maxes: Vec<f64> = columns.iter().map(|&(k, _)| col_max[k]).collect();
                let mut logs: Vec<bool> = columns.iter().map(|&(_, log)| log).collect();
                if self.config.redundancy {
                    used.push(s[9].clamp(0.0, 1.0));    // already on [0, 1]: its range is its scale
                    maxes.push(1.0);
                    logs.push(false);
                }
                if self.config.tower {
                    used.push(tower_penalty(tower));    // on [0, 1] by construction
                    maxes.push(1.0);
                    logs.push(false);
                }
                let fitness = hff_truenorth(&used, &maxes, &logs);
                if best.is_none_or(|b| fitness < b.fitness) {
                    // TrueNorth chooses the candidate and judges it; the balanced pole,
                    // when it is on, only decides who breeds.
                    let selection = if self.config.balanced_tournaments { hff_balanced(&used, &maxes, &logs) } else { fitness };
                    best = Some(Scored { fitness, linker: c / WRAPPERS.len(), wrapper: c % WRAPPERS.len(), a: s[0], b: s[1], one_minus_r2: omr2, t_depth: tower, selection });
                }
            }
            self.scored[r] = best;
            gen.fitness[r] = best.map_or(std::f32::consts::PI, |b| b.selection as f32);
        }
        timing.hff += t.elapsed().as_secs_f64();
        Ok((gene_ok.len() as u64, oversized))
    }

    /// The pump: the intake's best two replace the champion island's worst two;
    /// the intake keeps its best fifth (one of each distinct row) and the rest is
    /// refilled with new random individuals.
    fn pump(&mut self, gen: &mut Generation, generation: u32) {
        let l = self.layout;
        let row_w = (l.n_genes * l.gene_width()) as usize;
        let rnc_w = (l.n_genes * l.n_rnc) as usize;
        let (intake, champion) = (self.islands[0], self.islands[1]);
        let by_fitness = |rows: std::ops::Range<u32>, fitness: &[f32]| {
            let mut v: Vec<u32> = rows.collect();
            v.sort_by(|&a, &b| fitness[a as usize].total_cmp(&fitness[b as usize]).then(a.cmp(&b)));
            v
        };
        let copy_row = |gen: &mut Generation, scored: &mut Vec<Option<Scored>>, from: usize, to: usize| {
            gen.pop.genome.copy_within(from * row_w..(from + 1) * row_w, to * row_w);
            gen.pop.rnc.copy_within(from * rnc_w..(from + 1) * rnc_w, to * rnc_w);
            gen.pop.wrapper_id[to] = gen.pop.wrapper_id[from];
            gen.fitness[to] = gen.fitness[from];
            scored[to] = scored[from];
        };
        let best_intake: Vec<u32> = by_fitness(intake.lo..intake.hi, &gen.fitness).into_iter().filter(|&r| !gen.fitness[r as usize].is_nan()).collect();
        let worst_champion: Vec<u32> = by_fitness(champion.lo..champion.hi, &gen.fitness).into_iter().rev().take(2).collect();
        for (&from, &to) in best_intake.iter().zip(&worst_champion) {
            copy_row(gen, &mut self.scored, from as usize, to as usize);
        }
        let keep = ((f64::from(intake.hi - intake.lo) * 0.20).round() as usize).max(1);
        let mut seen: Vec<&[u32]> = Vec::new();
        let mut keepers: Vec<u32> = Vec::new();
        for &r in &best_intake {
            let row = &gen.pop.genome[r as usize * row_w..(r as usize + 1) * row_w];
            if keepers.len() < keep && !seen.contains(&row) {
                seen.push(row);
                keepers.push(r);
            }
        }
        let snapshot = gen.clone();
        let scored_before = self.scored.clone();
        for (slot, &from) in keepers.iter().enumerate() {
            let (to, from) = (intake.lo as usize + slot, from as usize);
            gen.pop.genome[to * row_w..(to + 1) * row_w].copy_from_slice(&snapshot.pop.genome[from * row_w..(from + 1) * row_w]);
            gen.pop.rnc[to * rnc_w..(to + 1) * rnc_w].copy_from_slice(&snapshot.pop.rnc[from * rnc_w..(from + 1) * rnc_w]);
            gen.pop.wrapper_id[to] = snapshot.pop.wrapper_id[from];
            gen.fitness[to] = snapshot.fitness[from];
            self.scored[to] = scored_before[from];
        }
        // Fresh rows: the CPU reference of the init kernel, keyed by this
        // generation, so a refill is reproducible.
        let first = intake.lo + keepers.len() as u32;
        let fresh = super::init(l, &self.table.codes(), &InitParams {
            seed: self.config.seed, generation, rnc_lo: self.config.rnc_lo, rnc_hi: self.config.rnc_hi, n_wrappers: WRAPPERS.len() as u32,
            vhead: self.vhead_at(generation),     // the pump's fresh rows are born at today's virtual head
        });
        if let Ok(fresh) = fresh {
            for r in first as usize..intake.hi as usize {
                gen.pop.genome[r * row_w..(r + 1) * row_w].copy_from_slice(&fresh.genome[r * row_w..(r + 1) * row_w]);
                gen.pop.rnc[r * rnc_w..(r + 1) * rnc_w].copy_from_slice(&fresh.rnc[r * rnc_w..(r + 1) * rnc_w]);
                gen.pop.wrapper_id[r] = fresh.wrapper_id[r];
                gen.fitness[r] = f32::NAN;
                self.scored[r] = None;
            }
        }
    }

    /// The row's best candidate re-scored in f64 by `chrom_score`, the
    /// definition: what may stop a fit or be reported. The device's f32 metrics
    /// only rank.
    fn confirm(&self, gen: &Generation, row: usize) -> Result<Option<Scored>, String> {
        let l = self.layout;
        let (width, nr, g_n) = (l.gene_width() as usize, l.n_rnc as usize, l.n_genes as usize);
        let mut batch = ExprBatch::new();
        let mut gene_ok = Vec::with_capacity(g_n);
        let mut tower = 0u32;
        for g in 0..g_n {
            let tokens = &gen.pop.genome[(row * g_n + g) * width..(row * g_n + g + 1) * width];
            let consts = &gen.pop.rnc[(row * g_n + g) * nr..(row * g_n + g + 1) * nr];
            match decode_gene(tokens, consts, l, &self.table).filter(|n| n.len() <= MAX_NODES) {
                Some(nodes) => {
                    tower = tower.max(t_depth(&nodes));
                    batch.push(&nodes);
                    gene_ok.push(true);
                }
                None => {
                    batch.push(&[GpuNode { op: Op::Num as u32, arg0: 0, arg1: 0, konst: 0.0 }]);
                    gene_ok.push(false);
                }
            }
        }
        let preds = self.evaluator.eval(&batch)?;
        let spec = ScoreSpec { linkers: LINKERS.to_vec(), wrappers: WRAPPERS.to_vec(), splits: self.data.splits, linear_scaling: true };
        let scores = score_chromosomes(&preds, &gene_ok, &[(0..g_n).collect()], &self.data.y, &spec)?;
        let n_ex = self.data.splits.n_extrap;
        let col_max = self.col_max.unwrap_or([1.0; 9]);
        let mut best: Option<Scored> = None;
        for c in 0..CANDIDATES {
            let s = &scores[c * SCORE_WIDTH..c * SCORE_WIDTH + METRIC_WIDTH];
            if !s[0].is_finite() {
                continue;
            }
            let (o, omr2) = self.caps.objectives(s, n_ex);
            let columns = hff_columns(n_ex, self.config.hff_without_validation, self.config.log_scale);
            let (used, maxes): (Vec<f64>, Vec<f64>) = columns.iter().map(|&(k, _)| (o[k], col_max[k])).unzip();
            let (mut used, mut maxes) = (used, maxes);
            let mut logs: Vec<bool> = columns.iter().map(|&(_, log)| log).collect();
            if self.config.tower {
                used.push(tower_penalty(tower));
                maxes.push(1.0);
                logs.push(false);
            }
            let fitness = hff_truenorth(&used, &maxes, &logs);
            if best.is_none_or(|b| fitness < b.fitness) {
                best = Some(Scored { fitness, linker: c / WRAPPERS.len(), wrapper: c % WRAPPERS.len(), a: s[0], b: s[1], one_minus_r2: omr2, t_depth: tower, selection: fitness });
            }
        }
        Ok(best)
    }

    /// The virtual head at `generation` — see `Config::vhead_every`. 0 = the whole
    /// head (what `InitParams` / `GenParams` take for "the ordinary gene").
    pub fn vhead_at(&self, generation: u32) -> u32 {
        let c = &self.config;
        if c.vhead_every == 0 {
            return 0;
        }
        (c.vhead_start + generation / c.vhead_every).clamp(2, self.layout.head)
    }

    /// THE HALL OF FAME's entry rule: the best individual by TrueNorth, if it beats
    /// the one held.
    fn remember(&self, hof: &mut Option<HallOfFame>, gen: &Generation, generation: u32) {
        let Some((row, b)) = self.best(gen) else { return };
        if hof.as_ref().is_some_and(|h| h.best.fitness <= b.fitness) {
            return;
        }
        let l = self.layout;
        let (row_w, rnc_w) = ((l.n_genes * l.gene_width()) as usize, (l.n_genes * l.n_rnc) as usize);
        *hof = Some(HallOfFame {
            generation,
            best: b,
            math: self.math_of(gen, row, &b),
            genome: gen.pop.genome[row * row_w..(row + 1) * row_w].to_vec(),
            rnc: gen.pop.rnc[row * rnc_w..(row + 1) * rnc_w].to_vec(),
            wrapper_id: gen.pop.wrapper_id[row],
        });
    }

    /// One row of the logbook, and the hall of fame's best appended to its file.
    fn report(&self, generation: u32, seconds: f64, gen: &Generation, hof: Option<&HallOfFame>) -> Result<(), String> {
        let fitness: Vec<f64> = gen.fitness.iter().filter(|f| !f.is_nan()).map(|f| f64::from(*f)).collect();
        let avg = fitness.iter().sum::<f64>() / fitness.len().max(1) as f64;
        let Some((_, b)) = self.best(gen) else { return Ok(()) };
        let third = |s: &Scored| if self.data.splits.n_extrap > 0 { format!("{:.10}", 1.0 - s.one_minus_r2[2]) } else { "-".to_string() };
        let (_, log10_p) = hff_p_value(b.fitness, self.hff_dimensions());
        let head = match self.vhead_at(generation) { 0 => self.layout.head, v => v };
        eprintln!(
            "{generation:>7}{seconds:>8.0}{head:>6}{:>13.6e}{avg:>13.6e}{:>13.4e}{:>15.10}{:>15.10}{:>15}{:>9}{log10_p:>10.2}",
            b.fitness, b.one_minus_r2[0] * self.caps.var[0], 1.0 - b.one_minus_r2[0], 1.0 - b.one_minus_r2[1], third(&b), b.t_depth
        );
        if let (Some(path), Some(h)) = (&self.config.hof_path, hof) {
            use std::io::Write;
            let model = crate::lint::node::Tree::parse(&h.math).map_or_else(|_| h.math.clone(), |t| t.to_infix());
            let (_, hof_log10_p) = hff_p_value(h.best.fitness, self.hff_dimensions());
            let mut file = std::fs::OpenOptions::new().append(true).open(path).map_err(|e| format!("hall of fame file {path}: {e}"))?;
            writeln!(
                file,
                "{generation}\t{}\t{:.6e}\t{hof_log10_p:.2}\t{:.4e}\t{:.10}\t{:.10}\t{}\t{}\t{model}",
                h.generation, h.best.fitness, h.best.one_minus_r2[0] * self.caps.var[0], 1.0 - h.best.one_minus_r2[0], 1.0 - h.best.one_minus_r2[1], third(&h.best), h.best.t_depth
            )
            .map_err(|e| format!("hall of fame file {path}: {e}"))?;
        }
        Ok(())
    }

    /// How many objectives HFF has in this fit: the dimension of its sphere.
    pub fn hff_dimensions(&self) -> usize {
        hff_columns(self.data.splits.n_extrap, self.config.hff_without_validation, self.config.log_scale).len()
            + usize::from(self.config.redundancy)
            + usize::from(self.config.tower)
    }

    fn best(&self, gen: &Generation) -> Option<(usize, Scored)> {
        (0..self.layout.pop as usize)
            .filter_map(|r| self.scored[r].map(|s| (r, s)))
            .filter(|(r, _)| !gen.fitness[*r].is_nan())
            .min_by(|a, b| a.1.fitness.total_cmp(&b.1.fitness).then(a.0.cmp(&b.0)))
    }

    /// `a * WRAPPER(LINKER(genes)) + b` as a fuller `Math` expression.
    pub fn math_of(&self, gen: &Generation, row: usize, s: &Scored) -> String {
        let l = self.layout;
        let (width, nr, g_n) = (l.gene_width() as usize, l.n_rnc as usize, l.n_genes as usize);
        let genes: Vec<String> = (0..g_n)
            .map(|g| {
                let tokens = &gen.pop.genome[(row * g_n + g) * width..(row * g_n + g + 1) * width];
                let consts = &gen.pop.rnc[(row * g_n + g) * nr..(row * g_n + g + 1) * nr];
                decode_gene(tokens, consts, l, &self.table).map_or("(Num 0.0)".to_string(), |n| nodes_to_math(&n, 0, &self.data.names))
            })
            .collect();
        let mut body = genes[0].clone();
        for g in &genes[1..] {
            body = if LINKER_NAMES[s.linker] == "mulval" { format!("(Mul {body} {g})") } else { format!("(Add {body} {g})") };
        }
        if LINKER_NAMES[s.linker] == "avgval" && g_n > 1 {
            body = format!("(Div {body} (Num {:?}))", g_n as f64);
        }
        body = match WRAPPERS[s.wrapper] {
            Wrapper::LogAbs => format!("(Log (Abs {body}))"),
            Wrapper::SqrtAbs => format!("(Sqrt (Abs {body}))"),
            _ => body,
        };
        format!("(Add (Mul (Num {:?}) {body}) (Num {:?}))", s.a, s.b)
    }

    pub fn fit(&mut self) -> Result<FitResult, String> {
        let c = self.config.clone();
        let started = Instant::now();
        let mut timing = Timing { vary: 0.0, read: 0.0, decode: 0.0, evaluate: 0.0, score: 0.0, hff: 0.0, pump: 0.0 };
        let rates = Rates::with_cleanse(self.layout, c.cleanse);
        self.dev.init(&InitParams { seed: c.seed, generation: 0, rnc_lo: c.rnc_lo, rnc_hi: c.rnc_hi, n_wrappers: WRAPPERS.len() as u32, vhead: self.vhead_at(0) })?;
        let mut gen = self.dev.read_generation()?;
        gen.fitness.fill(f32::NAN);
        let (mut unique, mut oversized) = self.evaluate(&mut gen, &mut timing)?;
        let mut individuals = u64::from(self.layout.pop);
        self.dev.write_fitness(&gen.fitness)?;
        let (mut generation, mut stopped_by) = (0u32, "n_gen");
        let mut hof: Option<HallOfFame> = None;
        self.remember(&mut hof, &gen, 0);
        if c.progress_every > 0 {
            eprintln!("{REPORT_HEADER}");
        }
        if let Some(path) = &c.hof_path {
            std::fs::write(path, "reported_at_gen\tfound_at_gen\thff\tlog10_p\tmse_train\tr2_train\tr2_val_bl2\tr2_val_bk3\tt_depth\tmodel\n").map_err(|e| format!("hall of fame file {path}: {e}"))?;
        }
        while generation < c.max_generations {
            if started.elapsed().as_secs_f64() > c.max_seconds {
                stopped_by = "time";
                break;
            }
            generation += 1;
            let t = Instant::now();
            self.dev.vary(&self.islands, &rates, &GenParams { seed: c.seed, generation, rnc_lo: c.rnc_lo, rnc_hi: c.rnc_hi, vhead: self.vhead_at(generation) })?;
            // Only elites arrive evaluated: the kernel puts the j-th fittest row
            // of an island (ties to the lower row) in the island's j-th row.
            let mut carried: Vec<(usize, Option<Scored>)> = Vec::new();
            for isl in &self.islands {
                let mut order: Vec<u32> = (isl.lo..isl.hi).collect();
                let key = |r: u32| { let f = gen.fitness[r as usize]; if f.is_nan() { f32::MAX } else { f.min(f32::MAX) } };
                order.sort_by(|&a, &b| key(a).total_cmp(&key(b)).then(a.cmp(&b)));
                for (j, &from) in order.iter().take(isl.elites as usize).enumerate() {
                    carried.push((isl.lo as usize + j, self.scored[from as usize]));
                }
            }
            timing.vary += t.elapsed().as_secs_f64();
            let t = Instant::now();
            gen = self.dev.read_generation()?;
            timing.read += t.elapsed().as_secs_f64();
            self.scored.fill(None);
            for (to, s) in carried {
                self.scored[to] = s;
            }
            let (u, o) = self.evaluate(&mut gen, &mut timing)?;
            unique += u;
            oversized += o;
            individuals += u64::from(self.layout.pop);
            // THE HALL OF FAME, before anything can end the fit: the winner of the
            // generation that meets the bar belongs in it too.
            self.remember(&mut hof, &gen, generation);
            // The device's f32 metrics cannot resolve 1e-10; they can say "this
            // one is worth confirming". The f64 re-score decides.
            if let Some((row, ranked)) = self.best(&gen) {
                if ranked.one_minus_r2[1] <= 1e-5 {
                    if let Some(s) = self.confirm(&gen, row)? {
                        let edge_ok = c.smogd || self.data.splits.n_extrap == 0 || s.one_minus_r2[2] <= c.stop_one_minus_r2;
                        if s.one_minus_r2[1] <= c.stop_one_minus_r2 && edge_ok {
                            stopped_by = "early_stop";
                            break;
                        }
                    }
                }
            }
            // THE PUMP, on its beat: the islands are where the diversity comes from.
            let t = Instant::now();
            if c.pump_every > 0 && generation % c.pump_every == 0 {
                self.pump(&mut gen, generation);
                let (u, o) = self.evaluate(&mut gen, &mut timing)?;
                unique += u;
                oversized += o;
                self.dev.write_population(&gen.pop)?;
            }
            timing.pump += t.elapsed().as_secs_f64();
            self.dev.write_fitness(&gen.fitness)?;
            if c.progress_every > 0 && generation % c.progress_every == 0 {
                self.report(generation, started.elapsed().as_secs_f64(), &gen, hof.as_ref())?;
            }
        }
        // The last row of the logbook: however the fit ended, its final state is
        // reported and the hall of fame's best is in the file.
        if c.progress_every > 0 && (stopped_by != "n_gen" || generation % c.progress_every != 0) {
            self.report(generation, started.elapsed().as_secs_f64(), &gen, hof.as_ref())?;
        }
        let (mut row, mut ranked) = self.best(&gen).ok_or("no individual could be scored")?;
        // Under balanced tournaments the TrueNorth best is not an elite and may have
        // left the population: the HALL OF FAME's winner goes back into a row, to be
        // confirmed in f64 and reported like any other.
        if let Some(h) = hof.as_ref().filter(|h| c.balanced_tournaments && h.best.fitness < ranked.fitness) {
            let l = self.layout;
            let (row_w, rnc_w) = ((l.n_genes * l.gene_width()) as usize, (l.n_genes * l.n_rnc) as usize);
            gen.pop.genome[row * row_w..(row + 1) * row_w].copy_from_slice(&h.genome);
            gen.pop.rnc[row * rnc_w..(row + 1) * rnc_w].copy_from_slice(&h.rnc);
            gen.pop.wrapper_id[row] = h.wrapper_id;
            ranked = h.best;
            row = row.min(l.pop as usize - 1);
        }
        let best = self.confirm(&gen, row)?.unwrap_or(ranked);
        Ok(FitResult {
            generations: generation,
            seconds: started.elapsed().as_secs_f64(),
            individuals,
            unique_genes: unique,
            oversized_genes: oversized,
            stopped_by,
            math: self.math_of(&gen, row, &best),
            best,
            timing,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_abs_whose_argument_keeps_one_sign_is_resolved_on_the_data() {
        // rows(): x_0 in 1..5, x_1 in 2..3, x_2 in 1.5..3.5.
        let positive = resolve_protected(r#"(Abs (Add (Var "x_0") (Var "x_1")))"#, &rows()).unwrap();
        assert_eq!(positive, r#"(Add (Var "x_0") (Var "x_1"))"#);
        // II.11.27's shape: the argument is negative on every row, so |e| = -e.
        let negative = resolve_protected(r#"(Div (Var "x_2") (Abs (Sub (Var "x_0") (Num 50.0))))"#, &rows()).unwrap();
        assert_eq!(negative, r#"(Div (Var "x_2") (Neg (Sub (Var "x_0") (Num 50.0))))"#);
        // One that changes sign on the data stays an Abs.
        let mixed = r#"(Abs (Sub (Var "x_0") (Num 3.0)))"#;
        assert_eq!(resolve_protected(mixed, &rows()).unwrap(), mixed);
        // And the predictions do not move.
        let before = evaluate_math(r#"(Div (Var "x_2") (Abs (Sub (Var "x_0") (Num 50.0))))"#, &rows()).unwrap();
        let after = evaluate_math(&negative, &rows()).unwrap();
        assert!(before.iter().zip(&after).all(|(a, b)| (a - b).abs() <= 1e-15 * a.abs()));
    }

    /// A gene's nodes from a `Math` string, laid out parents-before-children as
    /// `decode_gene` lays them out.
    fn nodes_of(math: &str) -> Vec<GpuNode> {
        use crate::lint::node::Tree;
        fn op_of(t: &Tree) -> (Op, Vec<&Tree>, f32, u32) {
            match t {
                Tree::Num(v) => (Op::Num, vec![], *v as f32, 0),
                Tree::Var(name) => (Op::Var, vec![], 0.0, name.trim_start_matches("x_").parse().unwrap_or(0)),
                Tree::App(op, kids) => (Op::from_math(&format!("{op:?}")).expect("an evaluator op"), kids.iter().collect(), 0.0, 0),
            }
        }
        let tree = Tree::parse(math).expect("parses");
        let mut queue = std::collections::VecDeque::from([&tree]);
        let mut out: Vec<GpuNode> = Vec::new();
        let mut next = 1u32;
        while let Some(t) = queue.pop_front() {
            let (op, kids, konst, var) = op_of(t);
            let (arg0, arg1) = match kids.len() {
                0 => (var, 0),
                1 => (next, 0),
                _ => (next, next + 1),
            };
            next += kids.len() as u32;
            out.push(GpuNode { op: op as u32, arg0, arg1, konst });
            queue.extend(kids);
        }
        out
    }

    #[test]
    fn t_depth_counts_nested_transcendentals_and_nothing_else() {
        // m c^2 / sqrt(1 - v^2/c^2): one transcendental on the deepest path.
        let law = r#"(Div (Mul (Var "x_0") (Pow2 (Var "x_2"))) (Sqrt (Sub (Num 1.0) (Div (Pow2 (Var "x_1")) (Pow2 (Var "x_2"))))))"#;
        assert_eq!(t_depth(&nodes_of(law)), 1);
        // A long flat law: no transcendental at all.
        let flat = r#"(Div (Mul (Mul (Var "x_0") (Var "x_1")) (Mul (Var "x_2") (Pow3 (Var "x_3")))) (Add (Var "x_0") (Inv (Neg (Var "x_1")))))"#;
        assert_eq!(t_depth(&nodes_of(flat)), 0);
        // exp(tanh(log|cos(sqrt x)|)): five, the protected log counting once.
        let tower = r#"(Exp (Tanh (ProtectedLog (Cos (Sqrt (Var "x_0"))))))"#;
        assert_eq!(t_depth(&nodes_of(tower)), 5);
        // The deepest PATH counts, not the total: sin x + cos x is 1.
        assert_eq!(t_depth(&nodes_of(r#"(Add (Sin (Var "x_0")) (Cos (Var "x_0")))"#)), 1);
        // A whole power is not transcendental; a fractional one is.
        assert_eq!(t_depth(&nodes_of(r#"(Pow (Var "x_0") (Num 4.0))"#)), 0);
        assert_eq!(t_depth(&nodes_of(r#"(Pow (Var "x_0") (Num 1.5))"#)), 1);
        assert_eq!(t_depth(&[]), 0);
        // feynman I.26.2, asin(n sin t): two, as the engine samples it and as
        // it is reported. tan and the inverse-trig ops each count once.
        assert_eq!(t_depth(&nodes_of(r#"(ProtectedAsin (Mul (Var "x_0") (Sin (Var "x_1"))))"#)), 2);
        assert_eq!(t_depth(&nodes_of(r#"(Asin (Mul (Var "x_0") (Sin (Var "x_1"))))"#)), 2);
        for op in ["Tan", "Asin", "Acos", "ProtectedAsin", "ProtectedAcos"] {
            assert_eq!(t_depth(&nodes_of(&format!(r#"(Mul (Var "x_0") ({op} (Var "x_1")))"#))), 1, "{op}");
        }
        // strogatz shearflow1, cos(x) cot(y) = cos(x) / tan(y): one.
        assert_eq!(t_depth(&nodes_of(r#"(ProtectedDiv (Cos (Var "x_0")) (Tan (Var "x_1")))"#)), 1);
    }

    /// The wide table samples tan and the protected inverse-trig functions, a
    /// population drawn from it keeps the structural rules, and a gene that
    /// holds one decodes to the Math constructor of the same name.
    #[test]
    fn the_wide_table_samples_tan_and_protected_inverse_trig() {
        let table = SymbolTable::wide(3);
        let codes = table.codes();
        codes.validate().unwrap();
        let new = [Op::Tan, Op::ProtectedAsin, Op::ProtectedAcos];
        let ids: Vec<u32> = new
            .iter()
            .map(|&op| table.symbols.iter().position(|s| *s == Symbol::Function(op)).unwrap_or_else(|| panic!("{op:?} is not in the wide table")) as u32)
            .collect();
        assert!(ids.iter().all(|id| codes.sample_functions.contains(id) && codes.arity[*id as usize] == 1));
        // The raw inverse-trig ops are NaN outside [-1, 1]: never sampled.
        assert!(!table.symbols.contains(&Symbol::Function(Op::Asin)) && !table.symbols.contains(&Symbol::Function(Op::Acos)));

        let layout = Layout::for_arity(400, 2, 12, table.max_arity(), 5);
        let p = crate::evolve::InitParams { seed: 11, generation: 0, rnc_lo: -10, rnc_hi: 10, n_wrappers: 3, vhead: 0 };
        let pop = crate::evolve::init(layout, &codes, &p).unwrap();
        pop.check(&codes).unwrap();
        assert_eq!(pop, crate::evolve::init(layout, &codes, &p).unwrap(), "same seed, same population");

        let names = names();
        let (width, n_rnc) = (layout.gene_width() as usize, layout.n_rnc as usize);
        let mut met = [false; 3];
        for (g, gene) in pop.genome.chunks(width).enumerate() {
            let Some(nodes) = decode_gene(gene, &pop.rnc[g * n_rnc..(g + 1) * n_rnc], layout, &table) else { continue };
            let math = nodes_to_math(&nodes, 0, &names);
            for (k, op) in new.iter().enumerate() {
                if nodes.iter().any(|n| n.op == *op as u32) {
                    assert!(math.contains(&format!("({op:?} ")), "{op:?} is in the nodes but not in {math}");
                    crate::lint::node::Tree::parse(&math).unwrap_or_else(|e| panic!("{math}: {e}"));
                    met[k] = true;
                }
            }
        }
        assert_eq!(met, [true; 3], "a population of 800 genes never expressed one of {new:?}");
    }

    #[test]
    fn the_p_value_is_hffs_own_and_shrinks_with_the_angle_and_the_dimension() {
        let (p, log10_p) = hff_p_value(0.379637, 10);
        assert!((p - hff_core::higd::cdf_beta_correction(0.379637, 10)).abs() == 0.0);
        assert!((log10_p - p.log10()).abs() < 1e-9, "{log10_p} vs {}", p.log10());
        // Nearer the pole is rarer; so is the same angle on a bigger sphere.
        assert!(hff_p_value(0.1, 10).0 < hff_p_value(0.3, 10).0);
        assert!(hff_p_value(0.3, 10).0 < hff_p_value(0.3, 4).0);
        // The deep tail: p underflows, its log does not.
        let (tiny, log_tiny) = hff_p_value(1e-40, 10);
        assert!(tiny == 0.0 || tiny < 1e-300);
        assert!(log_tiny.is_finite() && log_tiny < -300.0, "{log_tiny}");
    }

    #[test]
    fn the_tower_penalty_leaves_a_free_zone_up_to_depth_two() {
        assert_eq!([0, 1, 2, 3, 4, 5, 6, 9].map(tower_penalty), [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
    }

    #[test]
    fn the_hff_columns_are_the_blocks_asked_for() {
        let k = |c: Vec<(usize, bool)>| c.into_iter().map(|(i, _)| i).collect::<Vec<_>>();
        assert_eq!(k(hff_columns(0, false, [false; 3])), vec![0, 1, 3, 4, 6, 7]);          // train + validation (as before)
        assert_eq!(k(hff_columns(50, false, [false; 3])), vec![0, 1, 2, 3, 4, 5, 6, 7, 8]); // all nine (as before)
        assert_eq!(k(hff_columns(50, true, [false; 3])), vec![0, 2, 3, 5, 6, 8]);           // train + block three
        assert_eq!(k(hff_columns(0, true, [false; 3])), vec![0, 3, 6]);                     // train alone
        // The blocks' log scale is for blocks two and three, not train ...
        assert_eq!(hff_columns(50, true, [false, true, true]), vec![(0, false), (2, true), (3, false), (5, true), (6, false), (8, true)]);
        // ... unless train is asked for: alone in HFF, it is all the error there is.
        assert_eq!(hff_columns(0, true, [true, false, false]), vec![(0, true), (3, true), (6, true)]);
        // Each block is its own choice: validation log, the third block linear.
        assert_eq!(hff_columns(50, false, [false, true, false])[..3], [(0, false), (1, true), (2, false)]);
    }

    #[test]
    fn the_balanced_pole_is_hffs_own_and_ranks_differently_from_truenorth() {
        let (max, lin) = ([1.0, 1.0, 1.0], [false, false, false]);
        // An even, mediocre trade-off against a lopsided but better one.
        let (even, lopsided) = ([0.3, 0.3, 0.3], [0.01, 0.01, 0.4]);
        assert!(hff_truenorth(&lopsided, &max, &lin) < hff_truenorth(&even, &max, &lin), "TrueNorth prefers the smaller errors");
        assert!(hff_balanced(&even, &max, &lin) < hff_balanced(&lopsided, &max, &lin), "the balanced pole prefers the even trade-off");
        // It IS hff's function on the same scaled objectives.
        let direct = hff_core::core_functions::calculate_single_hyperspherical_fitness_f64_with_method(&ndarray::Array1::from(even.to_vec()), 3, false, None, "balanced");
        assert_eq!(hff_balanced(&even, &max, &lin), direct);
        assert_eq!(hff_balanced(&[f64::NAN, 0.1, 0.1], &max, &lin), std::f64::consts::PI);
    }

    #[test]
    fn the_log_scale_separates_small_errors_the_square_cannot() {
        let max = [1.0, 1.0];
        let (law, fake, poor) = ([1e-14, 1e-14], [2e-4, 5e-3], [2e-2, 0.3]);
        let linear = |o: &[f64; 2]| hff_truenorth(o, &max, &[false, false]);
        let log = |o: &[f64; 2]| hff_truenorth(o, &max, &[false, true]);
        // Both keep the order law < fake < poor ...
        assert!(linear(&law) <= linear(&fake) && linear(&fake) < linear(&poor));
        assert!(log(&law) < log(&fake) && log(&fake) < log(&poor));
        // ... but only the log scale puts the fake a long way from the law.
        assert!(linear(&fake) - linear(&law) < 0.01);
        assert!(log(&fake) - log(&law) > 0.5);
        // The floor and the ceiling: 1e-12 and below is 0, 1 is 1.
        assert_eq!(log(&[0.0, 1e-13]), 0.0);
        assert_eq!(log(&[0.0, 1.0]), linear(&[0.0, 1.0]));
    }

    fn rows() -> Vec<Vec<(String, f64)>> {
        (0..200)
            .map(|i| {
                let t = f64::from(i);
                vec![("x_0".to_string(), 1.0 + t * 0.02), ("x_1".to_string(), 2.0 + (t * 0.37).sin().abs()), ("x_2".to_string(), 1.5 + t * 0.01)]
            })
            .collect()
    }

    fn names() -> Vec<String> {
        vec!["x_0".to_string(), "x_1".to_string(), "x_2".to_string()]
    }

    /// Feynman I.12.4 as the engine found it: the law times a huge constant, plus
    /// a term eleven orders of magnitude smaller. The final form drops the dead
    /// term — and keeps predicting what the model predicts.
    #[test]
    fn the_final_form_prunes_a_term_the_data_cannot_see() {
        let model = r#"(Mul (Num -6.9e-11) (Div (Mul (Var "x_0") (Sub (Var "x_1") (Div (Num 1153767966.0) (Pow2 (Var "x_2"))))) (Var "x_1")))"#;
        let tidy = final_form(model, &names(), &rows()).unwrap();
        let (before, after) = (crate::lint::node::Tree::parse(model).unwrap(), crate::lint::node::Tree::parse(&tidy).unwrap());
        assert!(after.node_count() < before.node_count(), "{tidy}");
        let (want, got) = (evaluate_math(model, &rows()).unwrap(), evaluate_math(&tidy, &rows()).unwrap());
        let mean = want.iter().sum::<f64>() / want.len() as f64;
        let var = want.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / want.len() as f64;
        let drift = want.iter().zip(&got).map(|(w, g)| (w - g).powi(2)).sum::<f64>() / want.len() as f64 / var;
        assert!(drift <= FINAL_FORM_AGREE, "drift {drift}");
    }

    /// Every column positive on the data: |x| is x, and sqrt((a/b)^2) is a/b.
    #[test]
    fn the_final_form_uses_what_the_data_says_about_signs() {
        let model = r#"(Add (Mul (Num 2.0) (ProtectedSqrt (Abs (Div (Pow2 (Var "x_0")) (Pow2 (Var "x_1")))))) (Num 0.5))"#;
        let tidy = final_form(model, &names(), &rows()).unwrap();
        assert!(!tidy.contains("Abs") && !tidy.contains("Sqrt"), "{tidy}");
    }

    /// Feynman I.50.26 as the Rust engine found it: a divisor exp(-x*y)^3 that is
    /// below 1e-6 on every row. The protected divide is 0 there, not x / 1e-33.
    #[test]
    fn a_divisor_the_data_always_finds_tiny_is_zero() {
        let model = r#"(Add (Var "x_0") (ProtectedDiv (Var "x_1") (Pow3 (Exp (Neg (Mul (Num 30.0) (Var "x_1")))))))"#;
        let resolved = resolve_protected(model, &rows()).unwrap();
        assert_eq!(resolved, r#"(Add (Var "x_0") (Num 0.0))"#);
        assert_eq!(evaluate_math(model, &rows()).unwrap(), evaluate_math(&resolved, &rows()).unwrap());
    }

    #[test]
    fn a_divisor_the_data_never_finds_tiny_is_a_plain_division_and_a_mixed_one_is_spelled_out() {
        let never = resolve_protected(r#"(ProtectedDiv (Var "x_0") (Var "x_1"))"#, &rows()).unwrap();
        assert_eq!(never, r#"(Div (Var "x_0") (Var "x_1"))"#);
        // x_2 - 1.5 is exactly 0 on the first row only
        let mixed = resolve_protected(r#"(ProtectedDiv (Var "x_0") (Sub (Var "x_2") (Num 1.5)))"#, &rows()).unwrap();
        assert!(mixed.contains("ProtectedDiv"));
        let text = crate::lint::node::Tree::parse(&mixed).unwrap().to_infix_faithful();
        assert!(text.starts_with("Piecewise((0, Abs((x_2 - 1.5)) < 1e-6)"), "{text}");
    }

    #[test]
    fn a_root_whose_argument_is_always_finite_is_a_plain_root() {
        let resolved = resolve_protected(r#"(ProtectedSqrt (Sub (Var "x_0") (Var "x_1")))"#, &rows()).unwrap();
        assert_eq!(resolved, r#"(Sqrt (Abs (Sub (Var "x_0") (Var "x_1"))))"#);
        let overflow = resolve_protected(r#"(ProtectedSqrt (Exp (Mul (Num 400.0) (Var "x_0"))))"#, &rows()).unwrap();
        assert!(overflow.contains("ProtectedSqrt"), "{overflow}");
    }

    /// rows(): x_0 / 10 stays in [0.1, 0.5], so the clamp never acts and the
    /// protected form is the raw function; x_0 - 3 leaves [-1, 1] on some rows.
    #[test]
    fn an_inverse_trig_whose_argument_stays_in_the_domain_is_the_raw_function() {
        for (protected, raw, plain) in [("ProtectedAsin", "Asin", "asin"), ("ProtectedAcos", "Acos", "acos")] {
            let inside = format!(r#"({protected} (Mul (Num 0.1) (Mul (Var "x_0") (Sin (Var "x_1")))))"#);
            let resolved = resolve_protected(&inside, &rows()).unwrap();
            assert_eq!(resolved, format!(r#"({raw} (Mul (Num 0.1) (Mul (Var "x_0") (Sin (Var "x_1")))))"#));
            assert_eq!(evaluate_math(&inside, &rows()).unwrap(), evaluate_math(&resolved, &rows()).unwrap());
            let text = crate::lint::node::Tree::parse(&resolved).unwrap().to_infix_faithful();
            assert_eq!(text, format!("{plain}((0.1*(x_0*sin(x_1))))"));

            // x_0 - 3 runs from -2 to 1.98: clamped on some rows, so it stays
            // protected and is spelled out.
            let outside = format!(r#"({protected} (Sub (Var "x_0") (Num 3.0)))"#);
            assert_eq!(resolve_protected(&outside, &rows()).unwrap(), outside);
            let text = crate::lint::node::Tree::parse(&outside).unwrap().to_infix_faithful();
            assert!(text.starts_with(&format!("Piecewise(({plain}(Piecewise((-1, (x_0 - 3.0) < -1), (1, (x_0 - 3.0) > 1)")), "{text}");

            // exp(400 x_0) is huge-but-finite on the first rows (clamped to 1) and
            // +inf on the rest (answered with 0). acos gives 0 BOTH ways — one value
            // on every row, so the data guided rewrite writes it; asin gives pi/2
            // and then 0 — two values, so it stays protected.
            let overflow = format!(r#"({protected} (Exp (Mul (Num 400.0) (Var "x_0"))))"#);
            let expected = if protected == "ProtectedAcos" { "(Num 0.0)".to_string() } else { overflow.clone() };
            assert_eq!(resolve_protected(&overflow, &rows()).unwrap(), expected);
        }
    }

    /// rows(): x_0 runs from 1 to 5, so x_0 >= 1 and -x_0 <= -1 on EVERY row — the
    /// clamp always fires and the node is a constant, to be written as one.
    #[test]
    fn an_inverse_trig_whose_clamp_always_fires_is_its_constant() {
        use std::f64::consts::{FRAC_PI_2, PI};
        let above = r#"(Var "x_0")"#;
        let below = r#"(Neg (Var "x_0"))"#;
        for (expr, value) in [
            (format!("(ProtectedAsin {above})"), FRAC_PI_2),
            (format!("(ProtectedAsin {below})"), -FRAC_PI_2),
            (format!("(ProtectedAcos {above})"), 0.0),
            (format!("(ProtectedAcos {below})"), PI),
        ] {
            let resolved = resolve_protected(&expr, &rows()).unwrap();
            assert_eq!(resolved, format!("(Num {value:?})"), "{expr}");
            assert!(evaluate_math(&expr, &rows()).unwrap().iter().all(|v| (v - value).abs() < 1e-15), "{expr}");
        }
        // feynman III.12.43 as the engine found it: the law, plus a dead protected
        // arcsin that is pi/2 on every row, plus the constant that cancels it.
        let found = r#"(Add (Add (Mul (Num 0.1591549428374411) (Mul (Var "x_0") (Var "x_1"))) (Mul (Num 0.1591549428374411) (ProtectedAsin (Var "x_0")))) (Num -0.2499999951846772))"#;
        let resolved = resolve_protected(found, &rows()).unwrap();
        assert!(!resolved.contains("Asin"), "{resolved}");
        let before = evaluate_math(found, &rows()).unwrap();
        let after = evaluate_math(&resolved, &rows()).unwrap();
        assert!(before.iter().zip(&after).all(|(a, b)| (a - b).abs() < 1e-12), "the resolved form predicts something else");
        let tidy = final_form(&resolved, &names(), &rows()).unwrap();
        assert!(!tidy.contains("Asin") && tidy.contains("x_0") && tidy.contains("x_1"), "{tidy}");
    }

    /// Data guided rewrites: a subtree with ONE value on every row is that constant
    /// (strogatz glider2's shape), and acos(cos e) = e where the data keeps e in
    /// [0, pi] (strogatz barmag2's). What the data does not support is left alone.
    #[test]
    fn data_guided_rewrites_fold_dead_terms_and_principal_range_inverses() {
        // 1000*(x_0 - 3.01) is beyond +-1 with BOTH signs over rows(): the clamped
        // asin is +-pi/2, its absolute value pi/2 on every row.
        let dead = r#"(Add (Var "x_1") (Sqrt (Abs (ProtectedAsin (Mul (Num 1000.0) (Sub (Var "x_0") (Num 3.01)))))))"#;
        let resolved = resolve_protected(dead, &rows()).unwrap();
        assert!(!resolved.contains("Asin") && resolved.contains("x_1"), "{resolved}");
        let (a, b) = (evaluate_math(dead, &rows()).unwrap(), evaluate_math(&resolved, &rows()).unwrap());
        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-12));
        // x/x is 1.
        assert_eq!(resolve_protected(r#"(Div (Var "x_0") (Var "x_0"))"#, &rows()).unwrap(), "(Num 1.0)");
        // A live term is not touched.
        let live = r#"(Add (Var "x_1") (Sin (Var "x_0")))"#;
        assert_eq!(resolve_protected(live, &rows()).unwrap(), live);

        // 0.5*x_2 stays in [0.75, 1.75], inside [0, pi] and inside [-pi/2, pi/2]... of
        // which only the first holds for the whole of x_2 itself (it reaches 3.49 > pi).
        let inside = r#"(Mul (Num 0.5) (Var "x_2"))"#;
        assert_eq!(resolve_protected(&format!("(ProtectedAcos (Cos {inside}))"), &rows()).unwrap(), inside);
        assert_eq!(resolve_protected(r#"(ProtectedAsin (Sin (Mul (Num 0.2) (Var "x_2"))))"#, &rows()).unwrap(), r#"(Mul (Num 0.2) (Var "x_2"))"#);
        let outside = r#"(ProtectedAcos (Cos (Var "x_2")))"#;
        assert_eq!(resolve_protected(outside, &rows()).unwrap(), r#"(Acos (Cos (Var "x_2")))"#, "x_2 straddles pi: not linear, but the clamp cannot fire");
        // Any branch: x_2 + 2 runs 3.5..5.49, inside [pi, 2 pi] -> acos(cos e) = 2 pi - e
        // (strogatz barmag2); x_2 + 3.5 runs 5.0..6.99, inside [2 pi - pi/2, 2 pi + pi/2]
        // -> asin(sin e) = e - 2 pi. The predictions do not move.
        for (expr, wanted) in [
            (r#"(ProtectedAcos (Cos (Add (Var "x_2") (Num 2.0))))"#, format!(r#"(Sub (Num {:?}) (Add (Var "x_2") (Num 2.0)))"#, 2.0 * std::f64::consts::PI)),
            (r#"(ProtectedAsin (Sin (Add (Var "x_2") (Num 3.5))))"#, format!(r#"(Sub (Add (Var "x_2") (Num 3.5)) (Num {:?}))"#, 2.0 * std::f64::consts::PI)),
        ] {
            let resolved = resolve_protected(expr, &rows()).unwrap();
            assert_eq!(resolved, wanted);
            let (a, b) = (evaluate_math(expr, &rows()).unwrap(), evaluate_math(&resolved, &rows()).unwrap());
            assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-12), "{expr}");
        }
    }

    /// A term that matters stays.
    #[test]
    fn the_final_form_keeps_a_term_the_data_can_see() {
        let model = r#"(Add (Mul (Var "x_0") (Var "x_1")) (Sin (Var "x_2")))"#;
        let tidy = final_form(model, &names(), &rows()).unwrap();
        assert!(tidy.contains("Sin") && tidy.contains("x_0") && tidy.contains("x_1"), "{tidy}");
    }
}
