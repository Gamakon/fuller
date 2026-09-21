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

const OPS: [Op; 22] = [
    Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Neg, Op::Abs, Op::Sqrt, Op::Log, Op::Exp, Op::Sin, Op::Cos,
    Op::Tan, Op::Tanh, Op::Pow, Op::Pow2, Op::Pow3, Op::Inv, Op::ProtectedDiv, Op::ProtectedSqrt,
    Op::ProtectedLog, Op::ProtectedExp, Op::ProtectedInv,
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
    /// Blocks two and three (validation; SMOGD / SMOTE / edge) enter HFF on the
    /// log scale — see `hff_truenorth`. The train block stays linear.
    pub log_scale_blocks: bool,
    /// Block two (validation) is left OUT of HFF: a random cut of the same rows as
    /// train, its errors mirror train's and only dilute the third block's. HFF is
    /// then train + block three (hff's `METRIC_NAMES_TRAIN_ONLY`). Validation
    /// still decides the stop bar and is still reported.
    pub hff_without_validation: bool,
    /// The TRAIN block enters HFF on the log scale too. With validation and the
    /// third block out of HFF, train is all the error there is: linear, a fake at
    /// 1 - R² = 6e-6 and the law at 1e-14 are the same zero to a tournament.
    pub log_scale_train: bool,
    /// The TOWER objective: [`tower_penalty`] of the chromosome's [`t_depth`] joins
    /// HFF, so a tournament prefers the individual that is not a tower of nested
    /// functions when the errors cannot tell them apart.
    pub tower: bool,
    /// Harvest and regrow: a model that reaches the stop bar is put in a parking
    /// lot, it and its structural relatives are removed from the population, and
    /// the search goes on to grow another — up to this many (0 = stop at the
    /// first, as before). The one reported is the SMALLEST after fuller's final
    /// form; two harvests with the same final form end the fit.
    pub harvests: u32,
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
            log_scale_blocks: false,
            hff_without_validation: false,
            log_scale_train: false,
            tower: false,
            // Kept after a two-seed A/B (7012: 46 -> 47, 7013: 44 -> 45, no losses).
            harvests: 4,
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
    /// Models parked by harvest-and-regrow (0 when it is off).
    pub harvested: usize,
    pub best: Scored,
    pub math: String,
    pub timing: Timing,
}

const LINKERS: [Linker; 3] = [Linker::AVG, Linker::MUL, Linker::ADD];
const WRAPPERS: [Wrapper; 3] = [Wrapper::Identity, Wrapper::LogAbs, Wrapper::SqrtAbs];
const LINKER_NAMES: [&str; 3] = ["avgval", "mulval", "addval"];

/// TRANSCENDENTAL NESTING DEPTH of a decoded gene: the most transcendental
/// functions met on any path from the root to a leaf — exp, log, sin, cos, tan,
/// tanh, Abs, sqrt (and their protected forms), and a `Pow` whose exponent is not
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
        let t = [Op::Abs, Op::Sqrt, Op::Log, Op::Exp, Op::Sin, Op::Cos, Op::Tan, Op::Tanh, Op::ProtectedSqrt, Op::ProtectedLog, Op::ProtectedExp];
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

/// The tower objective on [0, 1]: SMOOTH, a sixth per level from 0 — depth 0 is 0,
/// 1 is 1/6, 2 is 2/6 ... 6 and deeper is 1. No step and no free zone: at equal
/// error the shallower individual always wins the tournament.
pub fn tower_penalty(t_depth: u32) -> f64 {
    (f64::from(t_depth) / 6.0).min(1.0)
}

/// Which of the nine objectives `[mse x3, 1-R2 x3, mae x3]` (blocks train,
/// validation, third) feed HFF, and which of those are log-scaled.
fn hff_columns(n_extrap: usize, without_validation: bool, log_scale_blocks: bool, log_scale_train: bool) -> Vec<(usize, bool)> {
    let mut columns = Vec::new();
    for metric in 0..3 {
        for block in 0..3 {
            let absent = (block == 2 && n_extrap == 0) || (block == 1 && without_validation);
            if !absent {
                columns.push((3 * metric + block, if block == 0 { log_scale_train } else { log_scale_blocks }));
            }
        }
    }
    columns
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
    let mut energy = 0.0;
    for ((v, max), log) in objectives.iter().zip(col_max).zip(log_scaled) {
        if !v.is_finite() {
            return std::f64::consts::PI;
        }
        let mut x = if *max > 0.0 { (v / max).min(1.0) } else { 0.0 };
        if *log {
            x = if x <= HFF_LOG_FLOOR { 0.0 } else { 1.0 + x.log10() / -HFF_LOG_FLOOR.log10() };
        }
        energy += x * x;
    }
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

/// `Abs e` where `e` keeps one sign on every row becomes `e` or `Neg e`, and
/// every protected operator the DATA never triggers becomes the raw one:
/// `ProtectedDiv a b` with |b| >= 1e-6 on every row is `Div a b`, with |b| < 1e-6
/// on every row it is 0; `ProtectedInv x` with x never 0 is `Inv x`. One that is
/// triggered on SOME rows stays protected, and `Tree::to_infix_faithful` writes
/// it out as the Piecewise it is. Decided on `rows`, the rows the model was
/// selected on — the engine's counterpart of hff's `symbolic_protected_div`.
pub fn resolve_protected(math: &str, rows: &[Vec<(String, f64)>]) -> Result<String, String> {
    use crate::lint::node::Tree;
    fn go(t: &Tree, rows: &[Vec<(String, f64)>]) -> Tree {
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
    Ok(go(&Tree::parse(math)?, rows).to_math())
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
        let columns = hff_columns(n_ex, self.config.hff_without_validation, self.config.log_scale_blocks, self.config.log_scale_train);
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
                    best = Some(Scored { fitness, linker: c / WRAPPERS.len(), wrapper: c % WRAPPERS.len(), a: s[0], b: s[1], one_minus_r2: omr2, t_depth: tower });
                }
            }
            self.scored[r] = best;
            gen.fitness[r] = best.map_or(std::f32::consts::PI, |b| b.fitness as f32);
        }
        timing.hff += t.elapsed().as_secs_f64();
        Ok((gene_ok.len() as u64, oversized))
    }

    /// The pump: the intake's best two replace the champion island's worst two;
    /// the intake keeps its best fifth (one of each distinct row) and the rest is
    /// refilled with new random individuals.
    fn pump(&mut self, gen: &mut Generation, generation: u32, vacate: Option<&[bool]>) {
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
        // A HARVEST is a pump event. The rows to vacate (the parked model's
        // structural relatives, on either island) lose their fitness first, so
        // the pump's own sort puts them last: the champion island's vacated rows
        // are the first the promotion overwrites, and no vacated row is kept.
        let vacated = |r: u32| vacate.is_some_and(|v| v[r as usize]);
        for r in 0..l.pop {
            if vacated(r) {
                gen.fitness[r as usize] = f32::NAN;
                self.scored[r as usize] = None;
            }
        }
        let best_intake: Vec<u32> = by_fitness(intake.lo..intake.hi, &gen.fitness).into_iter().filter(|&r| !gen.fitness[r as usize].is_nan()).collect();
        let worst_champion: Vec<u32> = by_fitness(champion.lo..champion.hi, &gen.fitness).into_iter().rev().take(2).collect();
        let mut promoted: Vec<u32> = Vec::new();
        for (&from, &to) in best_intake.iter().zip(&worst_champion) {
            copy_row(gen, &mut self.scored, from as usize, to as usize);
            promoted.push(to);
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
        });
        if let Ok(fresh) = fresh {
            // The intake's refill, and every vacated champion row the promotion
            // did not take.
            let refill = (first..intake.hi).chain((champion.lo..champion.hi).filter(|&r| vacated(r) && !promoted.contains(&r)));
            for r in refill.map(|r| r as usize) {
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
            let columns = hff_columns(n_ex, self.config.hff_without_validation, self.config.log_scale_blocks, self.config.log_scale_train);
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
                best = Some(Scored { fitness, linker: c / WRAPPERS.len(), wrapper: c % WRAPPERS.len(), a: s[0], b: s[1], one_minus_r2: omr2, t_depth: tower });
            }
        }
        Ok(best)
    }

    /// Every row the fit was given (train, validation and edge) as named values,
    /// for fuller's final form: a tidied model must hold on the edge rows too.
    fn fit_rows(&self) -> Vec<Vec<(String, f64)>> {
        let d = self.data.names.len();
        // A synthetic third block (SMOGD / SMOTE) is no judge of a form: its noisy
        // inputs can leave the real range, and one negative made-up value would
        // unmake "this input is positive". Real rows only.
        let s = self.data.splits;
        let n = if self.config.smogd { s.n_train + s.n_val } else { s.total() };
        (0..n).map(|r| self.data.names.iter().cloned().zip(self.data.x[r * d..(r + 1) * d].iter().map(|v| f64::from(*v))).collect()).collect()
    }

    /// What a gene COMPUTES, as a key: its expressed nodes with constants resolved.
    fn gene_key(&self, gen: &Generation, row: usize, g: usize) -> Option<Vec<u32>> {
        let l = self.layout;
        let (width, nr, g_n) = (l.gene_width() as usize, l.n_rnc as usize, l.n_genes as usize);
        let tokens = &gen.pop.genome[(row * g_n + g) * width..(row * g_n + g + 1) * width];
        let consts = &gen.pop.rnc[(row * g_n + g) * nr..(row * g_n + g + 1) * nr];
        decode_gene(tokens, consts, l, &self.table).map(|n| n.iter().flat_map(|x| [x.op, x.arg0, x.arg1, x.konst.to_bits()]).collect())
    }

    /// The harvested row's structural relatives: every row that is the same
    /// genome, or carries one of its genes of five nodes or more (the junk lives
    /// in the big genes; a bare terminal is a building block everyone shares).
    /// The pump vacates them.
    fn relatives(&self, gen: &Generation, harvested: usize) -> Vec<bool> {
        let g_n = self.layout.n_genes as usize;
        let keys: Vec<Option<Vec<u32>>> = (0..g_n).map(|g| self.gene_key(gen, harvested, g)).collect();
        let big: Vec<&Vec<u32>> = keys.iter().flatten().filter(|k| k.len() >= 5 * 4).collect();
        (0..self.layout.pop as usize)
            .map(|r| {
                let mine: Vec<Option<Vec<u32>>> = (0..g_n).map(|g| self.gene_key(gen, r, g)).collect();
                mine == keys || mine.iter().flatten().any(|k| big.contains(&k))
            })
            .collect()
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
        self.dev.init(&InitParams { seed: c.seed, generation: 0, rnc_lo: c.rnc_lo, rnc_hi: c.rnc_hi, n_wrappers: WRAPPERS.len() as u32 })?;
        let mut gen = self.dev.read_generation()?;
        gen.fitness.fill(f32::NAN);
        let (mut unique, mut oversized) = self.evaluate(&mut gen, &mut timing)?;
        let mut individuals = u64::from(self.layout.pop);
        self.dev.write_fitness(&gen.fitness)?;
        let (mut generation, mut stopped_by) = (0u32, "n_gen");
        let mut archive: Vec<(usize, String, String, Scored)> = Vec::new();
        while generation < c.max_generations {
            if started.elapsed().as_secs_f64() > c.max_seconds {
                stopped_by = "time";
                break;
            }
            generation += 1;
            let t = Instant::now();
            self.dev.vary(&self.islands, &rates, &GenParams { seed: c.seed, generation, rnc_lo: c.rnc_lo, rnc_hi: c.rnc_hi })?;
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
            // The device's f32 metrics cannot resolve 1e-10; they can say "this
            // one is worth confirming". The f64 re-score decides.
            let mut harvest_row: Option<usize> = None;
            if let Some((row, ranked)) = self.best(&gen) {
                if ranked.one_minus_r2[1] <= 1e-5 {
                    if let Some(s) = self.confirm(&gen, row)? {
                        let edge_ok = c.smogd || self.data.splits.n_extrap == 0 || s.one_minus_r2[2] <= c.stop_one_minus_r2;
                        if s.one_minus_r2[1] <= c.stop_one_minus_r2 && edge_ok {
                            if c.harvests == 0 {
                                stopped_by = "early_stop";
                                break;
                            }
                            // HARVEST. Park it; if an earlier harvest reduced
                            // to the same final form, two independent growths
                            // agree and the fit is over.
                            let raw = self.math_of(&gen, row, &s);
                            let tidy = final_form(&raw, &self.data.names, &self.fit_rows()).unwrap_or_else(|_| raw.clone());
                            let nodes = crate::lint::node::Tree::parse(&tidy).map_or(usize::MAX, |t| t.node_count());
                            let agreed = archive.iter().any(|a: &(usize, String, String, Scored)| a.1 == tidy);
                            archive.push((nodes, tidy, raw, s));
                            if agreed || archive.len() as u32 >= c.harvests {
                                stopped_by = if agreed { "converged" } else { "harvested" };
                                break;
                            }
                            harvest_row = Some(row);
                        }
                    }
                }
            }
            // THE PUMP, on its beat — and at once on a harvest, where it also
            // vacates the parked model's relatives: REGROW is the pump's refill.
            let t = Instant::now();
            if harvest_row.is_some() || (c.pump_every > 0 && generation % c.pump_every == 0) {
                let vacate = harvest_row.map(|row| self.relatives(&gen, row));
                self.pump(&mut gen, generation, vacate.as_deref());
                let (u, o) = self.evaluate(&mut gen, &mut timing)?;
                unique += u;
                oversized += o;
                self.dev.write_population(&gen.pop)?;
            }
            timing.pump += t.elapsed().as_secs_f64();
            self.dev.write_fitness(&gen.fitness)?;
        }
        let harvested = archive.len();
        // The parking lot decides, when there is one: every model in it met the
        // bar, so PARSIMONY picks — the fewest nodes after fuller's final form.
        if let Some((_, _, raw, s)) = archive.into_iter().min_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1))) {
            return Ok(FitResult {
                generations: generation,
                seconds: started.elapsed().as_secs_f64(),
                individuals,
                unique_genes: unique,
                oversized_genes: oversized,
                stopped_by,
                harvested,
                math: raw,
                best: s,
                timing,
            });
        }
        let (row, ranked) = self.best(&gen).ok_or("no individual could be scored")?;
        let best = self.confirm(&gen, row)?.unwrap_or(ranked);
        Ok(FitResult {
            generations: generation,
            seconds: started.elapsed().as_secs_f64(),
            individuals,
            unique_genes: unique,
            oversized_genes: oversized,
            stopped_by,
            harvested,
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
    }

    #[test]
    fn the_tower_penalty_is_smooth_a_sixth_per_level() {
        let got = [0, 1, 2, 3, 4, 5, 6, 9].map(tower_penalty);
        let want = [0.0, 1.0 / 6.0, 2.0 / 6.0, 0.5, 4.0 / 6.0, 5.0 / 6.0, 1.0, 1.0];
        assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-15), "{got:?}");
        // Every level costs the same: no step.
        assert!((1..=6).all(|t| (tower_penalty(t) - tower_penalty(t - 1) - 1.0 / 6.0).abs() < 1e-15));
    }

    #[test]
    fn the_hff_columns_are_the_blocks_asked_for() {
        let k = |c: Vec<(usize, bool)>| c.into_iter().map(|(i, _)| i).collect::<Vec<_>>();
        assert_eq!(k(hff_columns(0, false, false, false)), vec![0, 1, 3, 4, 6, 7]);          // train + validation (as before)
        assert_eq!(k(hff_columns(50, false, false, false)), vec![0, 1, 2, 3, 4, 5, 6, 7, 8]); // all nine (as before)
        assert_eq!(k(hff_columns(50, true, false, false)), vec![0, 2, 3, 5, 6, 8]);           // train + block three
        assert_eq!(k(hff_columns(0, true, false, false)), vec![0, 3, 6]);                     // train alone
        // The blocks' log scale is for blocks two and three, not train ...
        assert_eq!(hff_columns(50, true, true, false), vec![(0, false), (2, true), (3, false), (5, true), (6, false), (8, true)]);
        // ... unless train is asked for: alone in HFF, it is all the error there is.
        assert_eq!(hff_columns(0, true, false, true), vec![(0, true), (3, true), (6, true)]);
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

    /// A term that matters stays.
    #[test]
    fn the_final_form_keeps_a_term_the_data_can_see() {
        let model = r#"(Add (Mul (Var "x_0") (Var "x_1")) (Sin (Var "x_2")))"#;
        let tidy = final_form(model, &names(), &rows()).unwrap();
        assert!(tidy.contains("Sin") && tidy.contains("x_0") && tidy.contains("x_1"), "{tidy}");
    }
}
