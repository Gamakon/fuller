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
use super::genealogy::{Genealogy, GenealogyLog, Origin, PopulationAges, RowMark};
use super::score::{GpuScorer, WIDTH};
use super::vary::{GenParams, Generation, Island, Rates};
use super::write_back::{gene_form, model_form, write_back, SnapCounts, WriteBack};
use super::{InitParams, Layout, Population, SymbolCodes};
use crate::chrom_score::{gene_linker_combinations, score_gene_subsets, GeneLinker, Linker, ScoreSpec, Splits, Wrapper, METRIC_WIDTH, SCORE_WIDTH};
use crate::gpu_eval::{ExprBatch, GpuEvaluator, GpuNode, Op, MAX_NODES};

/// What a token id means to the evaluator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Symbol {
    Function(Op),
    /// A COMPOUND function: one symbol of arity 2 that a gene can pick as it picks
    /// any other, expanded into ordinary nodes when the gene is decoded.
    Compound(Compound),
    /// A data column.
    Input(u32),
    /// A fixed numeric terminal.
    Constant(f32),
    /// The "?" placeholder: the n-th one in a gene's expression reads
    /// `rnc[dc[n]]` — geppy's Dc domain.
    Rnc,
    /// A NAMED CONSTANT of the lattice (pi, e, sqrt2, hbar ...): `named[k]` of the
    /// table. Withheld — never drawn; it enters a gene only by snap's write-back.
    Named(u32),
}

/// COMPOUND FUNCTIONS (Andrew's idea): "construct a function called sum, and then
/// another with a sum under a 1/root, and set up the arity to fill these functions
/// properly ... then it is a matter of the genes to find when to use it." Each is
/// a shape the search measurably never builds from single operators — a sum under
/// a root (1 of 18 such laws solved), the 1/sqrt(1 - v^2/c^2) family (0 of 9), a
/// sum in a denominator (4 of 19) — offered as ONE symbol. A compound is a MACRO:
/// `decode_gene` expands it into nodes that already exist, protected like every
/// sampled operator, so the evaluator, fuller's Math, the linter and the printers
/// see nothing new, and the data guided rewrites turn the protected forms plain
/// where the data allows. They thin the search like any added function, so they
/// are a switch (`Config::compounds`) meant for the race's SECOND pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compound {
    /// sqrt|a + b|
    SqrtSum,
    /// sqrt|a - b|
    SqrtDiff,
    /// 1 / sqrt|a + b|
    InvSqrtSum,
    /// 1 / sqrt|a - b|   (the Lorentz factor is `InvSqrtDiff(1, (v/c)^2)`)
    InvSqrtDiff,
    /// 1 / (a + b)
    InvSum,
    /// 1 / (a - b)
    InvDiff,
}

impl Compound {
    pub const ALL: [Compound; 6] = [Compound::SqrtSum, Compound::SqrtDiff, Compound::InvSqrtSum, Compound::InvSqrtDiff, Compound::InvSum, Compound::InvDiff];

    /// The operators applied to `(a, b)`, innermost first.
    pub fn expansion(self) -> &'static [Op] {
        match self {
            Compound::SqrtSum => &[Op::Add, Op::ProtectedSqrt],
            Compound::SqrtDiff => &[Op::Sub, Op::ProtectedSqrt],
            Compound::InvSqrtSum => &[Op::Add, Op::ProtectedSqrt, Op::ProtectedInv],
            Compound::InvSqrtDiff => &[Op::Sub, Op::ProtectedSqrt, Op::ProtectedInv],
            Compound::InvSum => &[Op::Add, Op::ProtectedInv],
            Compound::InvDiff => &[Op::Sub, Op::ProtectedInv],
        }
    }
}

#[derive(Clone, Debug)]
pub struct SymbolTable {
    pub symbols: Vec<Symbol>,
    /// Terminals that may be carried but never drawn (named constants).
    pub withheld: Vec<bool>,
    /// The lattice's named constants, by NAME INDEX (`SnapTable::names`' order):
    /// the name, and its f64 value from `snap_karva::constant_values`. The index,
    /// never the name, identifies one: a data column may itself be called `c`.
    pub named: Vec<(String, f64)>,
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
        SymbolTable { symbols, withheld, named: Vec::new() }
    }

    /// The wide set plus the compound functions. They are appended AFTER every
    /// existing symbol, so no id of the wide set moves.
    pub fn with_compounds(mut self) -> SymbolTable {
        self.symbols.extend(Compound::ALL.map(Symbol::Compound));
        self.withheld.resize(self.symbols.len(), false);
        self
    }

    /// The lattice's named constants join the table as WITHHELD terminals, in the
    /// lattice's own order (`names` is `SnapTable::names`, taken at run time). They
    /// are appended AFTER every existing symbol — the wide set, then the compounds
    /// when they are on, then these — so no id moves. A name the crate has no value
    /// for is an error.
    pub fn with_named(mut self, names: &[String]) -> Result<SymbolTable, String> {
        let known = crate::snap_karva::constant_values();
        for (k, name) in names.iter().enumerate() {
            let value = *known.get(name).ok_or_else(|| format!("{name}: the lattice names it, constant_values does not"))?;
            self.symbols.push(Symbol::Named(k as u32));
            self.named.push((name.clone(), value));
        }
        self.withheld.resize(self.symbols.len(), true);
        Ok(self)
    }

    /// The id of named constant `k`, if the table carries it.
    pub fn named_id(&self, k: u32) -> Option<u32> {
        self.symbols.iter().position(|s| *s == Symbol::Named(k)).map(|i| i as u32)
    }

    /// The id of function `op`, if the table carries it.
    pub fn function_id(&self, op: Op) -> Option<u32> {
        self.symbols.iter().position(|s| *s == Symbol::Function(op)).map(|i| i as u32)
    }

    /// The f64 values of the named constants, by name index: what
    /// [`nodes_to_math_named`] prints.
    pub fn named_values(&self) -> Vec<f64> {
        self.named.iter().map(|(_, v)| *v).collect()
    }

    pub fn arity(&self, id: u32) -> u32 {
        match self.symbols[id as usize] {
            Symbol::Function(op) => op.arity() as u32,
            Symbol::Compound(_) => 2,
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
    // 1. The gene's own tree, by gene position: position i's children are the next
    //    unclaimed positions (Karva), and the n-th "?" IN GENE ORDER reads dc[n].
    let (mut child, mut n_rnc) = (1usize, 0usize);
    let mut kids: Vec<(usize, usize)> = Vec::with_capacity(n);
    let mut konst = vec![0.0f32; n];
    for (i, &id) in gene[..n].iter().enumerate() {
        let a = table.arity(id) as usize;
        kids.push((if a >= 1 { child } else { 0 }, if a == 2 { child + 1 } else { 0 }));
        child += a;
        if table.symbols[id as usize] == Symbol::Rnc {
            let k = *gene.get(ht + n_rnc)? as usize;
            n_rnc += 1;
            konst[i] = *rnc.get(k)?;
        }
    }
    // 2. Level order over the EXPANDED tree. A queue entry is a gene position, or an
    //    inner operator of a compound still to be written above a gene position. For
    //    a gene with no compound this is the gene's own order, node for node.
    enum Todo {
        Gene(usize),
        /// the compound at this gene position, with this many of its operators
        /// still to write (the outermost is written first)
        Inner(usize, usize),
    }
    let mut nodes: Vec<GpuNode> = Vec::with_capacity(n + 8);
    let mut queue = std::collections::VecDeque::from([Todo::Gene(0)]);
    let mut next = 1u32;
    while let Some(todo) = queue.pop_front() {
        let (pos, left) = match todo {
            Todo::Gene(pos) => (pos, None),
            Todo::Inner(pos, left) => (pos, Some(left)),
        };
        let node = match (table.symbols[gene[pos] as usize], left) {
            (Symbol::Compound(c), left) => {
                let ops = c.expansion();
                let left = left.unwrap_or(ops.len());
                let op = ops[left - 1];
                if left == 1 {
                    // the innermost operator takes the compound's two arguments
                    queue.push_back(Todo::Gene(kids[pos].0));
                    queue.push_back(Todo::Gene(kids[pos].1));
                    next += 2;
                    GpuNode { op: op as u32, arg0: next - 2, arg1: next - 1, konst: 0.0 }
                } else {
                    queue.push_back(Todo::Inner(pos, left - 1));
                    next += 1;
                    GpuNode { op: op as u32, arg0: next - 1, arg1: 0, konst: 0.0 }
                }
            }
            (Symbol::Function(op), _) => {
                let a = op.arity() as u32;
                queue.push_back(Todo::Gene(kids[pos].0));
                if a == 2 {
                    queue.push_back(Todo::Gene(kids[pos].1));
                }
                next += a;
                GpuNode { op: op as u32, arg0: next - a, arg1: if a == 2 { next - 1 } else { 0 }, konst: 0.0 }
            }
            (Symbol::Input(col), _) => GpuNode { op: Op::Var as u32, arg0: col, arg1: 0, konst: 0.0 },
            (Symbol::Constant(v), _) => GpuNode { op: Op::Num as u32, arg0: 0, arg1: 0, konst: v },
            (Symbol::Rnc, _) => GpuNode { op: Op::Num as u32, arg0: 0, arg1: 0, konst: konst[pos] },
            // Stage 2's convention: the f32 value the evaluator reads, and the name's
            // index + 1 in `arg0` (no reader of a `Num` looks there), so a decoded
            // gene that carries pi is known to carry it.
            (Symbol::Named(k), _) => GpuNode { op: Op::Num as u32, arg0: k + 1, arg1: 0, konst: table.named.get(k as usize)?.1 as f32 },
        };
        nodes.push(node);
    }
    Some(nodes)
}

/// The nodes as a fuller `Math` expression, inputs named `names[col]`.
pub fn nodes_to_math(nodes: &[GpuNode], at: usize, names: &[String]) -> String {
    nodes_to_math_named(nodes, at, names, &[])
}

/// [`nodes_to_math`] for genes that may carry named constants: a `Num` whose
/// `arg0` is 1 + a name index is written as that constant's f64 VALUE, `named[k]`
/// (`SymbolTable::named_values`) — `(Num 3.141592653589793)`, never the f32 the
/// device reads and never `(Var "pi")`: `evaluate_math` binds names from the data
/// columns, and a column may itself be called `c` or `h`.
pub fn nodes_to_math_named(nodes: &[GpuNode], at: usize, names: &[String], named: &[f64]) -> String {
    let node = nodes[at];
    if node.op == Op::Var as u32 {
        return format!("(Var \"{}\")", names[node.arg0 as usize]);
    }
    if node.op == Op::Num as u32 {
        let value = (node.arg0 as usize).checked_sub(1).and_then(|k| named.get(k)).copied().unwrap_or(f64::from(node.konst));
        return format!("(Num {value:?})");
    }
    let op = OPS.iter().find(|op| **op as u32 == node.op).copied().unwrap_or(Op::Add);
    let first = nodes_to_math_named(nodes, node.arg0 as usize, names, named);
    if op.arity() == 2 {
        format!("({op:?} {first} {})", nodes_to_math_named(nodes, node.arg1 as usize, names, named))
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
    /// PAIRS of islands: each pair is an intake island and its champion island, and
    /// THE PUMP works inside a pair. `pop_intake` and `pop_champion` are the sizes of
    /// ONE pair's islands (as each deme has its own size in the notebook), so the
    /// population is `n_pairs * (pop_intake + pop_champion)`. 1 = the single pair.
    pub n_pairs: u32,
    /// THE CROSS STEP's beat: every this many generations each intake island takes
    /// in the best of the OTHER pairs' champion islands (0 = never). The SRBench
    /// entry's beat is 5, just off the pump's 4 — on ONE pair (it never sets
    /// `wrapper_islands`), where the step is a keep-the-fifth refill. The notebook
    /// also holds the step back until `gen > 30` and runs it every generation once
    /// `gen > n_gen - 10`. Neither is here: the warm-up was sized for a Python fit
    /// of a few hundred generations, and a fit stopped by time has no `n_gen` to
    /// count down to. The beat is the whole rule.
    pub cross_every: u32,
    /// How many of its best each champion island sends in a cross step.
    pub k_migrants: u32,
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
    /// The COMPOUND functions join the symbol table — see [`Compound`]. For the
    /// race's second pass; off by default.
    pub compounds: bool,
    pub max_generations: u32,
    pub max_seconds: f64,
    /// Stop when validation (and edge, when there is one) 1 - R² is this small.
    pub stop_one_minus_r2: f64,
    /// THE STOP BAR's second half: the confirmed model's HFF angle as a p-value must
    /// be at most this (log10). -19: on development seed 7013 every real law that
    /// met the 1 - R² bar sat at -19.4 or below (most at -21 to -22) and the one fake
    /// that slipped under it (strogatz bacres2, validation 1 - R² 1.0e-11) sat at
    /// -17.55 — the clearest signal we have. A law a little above the bar is not
    /// lost: the fit simply keeps evolving and reports its best. Measured with 4
    /// objectives (train + t_depth); p depends on how many objectives HFF has.
    /// `f64::INFINITY` switches this half off.
    pub stop_log10_p: f64,
    /// THE DYNAMIC GENE-SUBSET CHOICE (Andrew: "make the linker dynamic and it
    /// could decide on the number of genes"): a chromosome is scored under every
    /// NON-EMPTY SUBSET of its genes — each single gene by itself, each pair and
    /// the whole under each linker, 15 gene-linker combinations for 3 genes — and
    /// keeps the best, as it already keeps the best linker and wrapper. A law that
    /// is one nested structure no longer needs the other genes to evolve into
    /// exact do-nothing values. At most 3 genes; off = all the genes, always.
    pub gene_subsets: bool,
    /// SNAP WINNERS' beat (`hff_sr_engine.py::_apply_snap_to_winners`): every this
    /// many generations the chosen rows' genes go through snap — match, graft, the
    /// guard on the TRAIN rows judging the whole MODEL — and every kept form is
    /// WRITTEN BACK into its gene (`write_back.rs`). 0 = off, the default.
    pub snap_every: u32,
    /// How many rows of each island, by fitness, are snap winners; 0 = every
    /// evaluated row (the notebook's `snap_winners_top_k = 0`).
    pub snap_top_k: u32,
    /// `_snap_op.py`'s `rel_tol`: how near a lattice entry a folded constant must be.
    pub snap_rel_tol: f64,
    /// `_snap_op.py`'s `r2_drop_tol`: how much of the model's train R² a snap may cost.
    pub snap_r2_drop: f64,
    /// THE GENEALOGY LOG's file — ALPS's measurement half. None (the default) is
    /// OFF and the engine is what it was, bit for bit: nothing is tracked, no
    /// buffer is read back and no file is written. With a path, every individual
    /// of the fit carries an IDENTITY, an AGE and a LINEAGE
    /// ([`super::genealogy`]) and the file takes a header and then, appended: the
    /// best individual of every generation, every ARRIVAL (a row whose origin is
    /// not ordinary variation — a promotion, a keeper, a migrant, a snap
    /// write-back), one line per batch of fresh random individuals, and when the
    /// fit ends the winner's whole chain back to its founder.
    pub genealogy_path: Option<String>,
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
            n_pairs: 1,
            cross_every: 0,
            k_migrants: 3,
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
            compounds: false,
            // Kept after a two-seed A/B (7012: 46 -> 47, 7013: 44 -> 45, no losses).
            max_generations: 1500,
            max_seconds: 30.0,
            stop_one_minus_r2: 1e-10,
            stop_log10_p: -19.0,
            gene_subsets: false,
            snap_every: 0,
            snap_top_k: 0,
            snap_rel_tol: 1e-3,
            snap_r2_drop: crate::lint::snap_guard::R2_DROP_TOL,
            genealogy_path: None,
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
    /// Which genes the model USES: bit g is the chromosome's g-th gene. All of
    /// them unless `Config::gene_subsets` chose fewer; a single gene has no linker.
    pub genes: u32,
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
    /// WHO IT WAS: its identity, age and line at the generation it was remembered.
    /// None when the genealogy is off.
    pub mark: Option<RowMark>,
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
    pub cross: f64,
    pub snap: f64,
    /// What the genealogy costs: the parent read-back and the log's writes. 0.0
    /// when `Config::genealogy_path` is None.
    pub genealogy: f64,
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
    /// What snap did (all zero when `Config::snap_every` is 0).
    pub snap: SnapCounts,
    /// THE WINNER'S LINEAGE: its identity, how old it was in generations, and the
    /// generation and mechanism its line began at. None when the genealogy is off.
    pub lineage: Option<RowMark>,
    /// THE FINAL POPULATION's ages, and how many lines its best rows descend
    /// from — the diversity number. None when the genealogy is off.
    pub population_ages: Option<PopulationAges>,
    /// How many ids the fit minted, and how many lines and bytes the log took.
    pub genealogy_minted: u64,
    pub genealogy_lines: u64,
    pub genealogy_bytes: u64,
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
/// What the logbook row and the hall of fame's file gain when the genealogy is on:
/// the best individual's AGE, and the generation and mechanism its line began at.
const LINEAGE_HEADER: &str = "      age  found_gen   found_origin";

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
/// `Asin x` / `Acos x`; `ProtectedLog x` with x never 0 is `Log (Abs x)`;
/// `ProtectedExp x` that never overflows is `Exp x`. And three forms a symbolic
/// scorer does not see through: `Log` of a product with an `Exp u` factor, the
/// rest of it c > 0 on every row, is `u + log c`; `Log a -/+ Log b` with a, b > 0
/// on every row is `Log (Div a b)` / `Log (Mul a b)`, a negation going inside
/// as `Log (Div b a)`; `Sin` / `Cos` of `e + k*pi/2`, the constant exact to
/// 1e-9, is the function a quarter turn on; `Tan` of an arcsin of a `u` the data
/// keeps inside |u| < 1 is `u / sqrt(1 - u^2)`; `Sqrt` of a square is the
/// absolute value, and of an `(a+b)/(a-b)` quotient the data keeps positive is
/// the family written in the ratio, `(1 + b/a)/sqrt(1 - (b/a)^2)`; and a factor
/// standing on both sides of a product's line, never 0 on the data, cancels.
/// One that is
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
    /// `t` on every row, when it is finite on every row and passes `test` there.
    fn on_every_row(t: &Tree, rows: &[Vec<(String, f64)>], test: impl Fn(f64) -> bool) -> bool {
        evaluate_math(&t.to_math(), rows).is_ok_and(|v| !v.is_empty() && v.iter().all(|x| x.is_finite() && test(*x)))
    }
    /// The factors of a product / quotient, the `Exp u` factors apart: `u` goes to
    /// `above` or `below` the line, every other factor to `num` or `den`.
    fn factors<'t>(t: &'t Tree, inverted: bool, above: &mut Vec<&'t Tree>, below: &mut Vec<&'t Tree>, num: &mut Vec<&'t Tree>, den: &mut Vec<&'t Tree>) {
        match t {
            Tree::App(Op::Mul, k) => k.iter().for_each(|f| factors(f, inverted, above, below, num, den)),
            Tree::App(Op::Div, k) => {
                factors(&k[0], inverted, above, below, num, den);
                factors(&k[1], !inverted, above, below, num, den);
            }
            Tree::App(Op::Inv, k) => factors(&k[0], !inverted, above, below, num, den),
            Tree::App(Op::Exp, k) => (if inverted { below } else { above }).push(&k[0]),
            _ => (if inverted { den } else { num }).push(t),
        }
    }
    fn product(of: &[&Tree]) -> Option<Tree> {
        of.iter().map(|f| (*f).clone()).reduce(|a, b| Tree::App(Op::Mul, vec![a, b]))
    }
    /// The factors of a product / quotient, `Exp` kept as an ordinary factor:
    /// what is above the line and what is below it. The counterpart of `factors`
    /// for cancelling, where an exponent must NOT be hoisted out of the product.
    fn over_and_under<'t>(t: &'t Tree, inverted: bool, num: &mut Vec<&'t Tree>, den: &mut Vec<&'t Tree>) {
        match t {
            Tree::App(Op::Mul, k) => k.iter().for_each(|f| over_and_under(f, inverted, num, den)),
            Tree::App(Op::Div, k) => {
                over_and_under(&k[0], inverted, num, den);
                over_and_under(&k[1], !inverted, num, den);
            }
            Tree::App(Op::Inv, k) => over_and_under(&k[0], !inverted, num, den),
            _ => (if inverted { den } else { num }).push(t),
        }
    }
    /// A FACTOR ON BOTH SIDES OF THE LINE cancels — where the data says it is
    /// finite and never 0, since a/a is not 1 at 0. The general form of the x/x
    /// `constant_on_data` already folds: feynman II.13.23's tan(asin(v/c))/v
    /// becomes (v/c)/sqrt(1 - v^2/c^2)/v, and the law is only there once the v
    /// and the c cancel. `None` when nothing cancels, so a product that is
    /// already in its lowest terms is left exactly as it was written.
    fn cancelled(t: &Tree, rows: &[Vec<(String, f64)>]) -> Option<Tree> {
        let (mut num, mut den) = (vec![], vec![]);
        over_and_under(t, false, &mut num, &mut den);
        let mut cut = false;
        let mut i = 0;
        while i < num.len() {
            // A literal is cancelled by the folds, not here; and 0 never cancels.
            let live = !matches!(num[i], Tree::Num(_)) && on_every_row(num[i], rows, |v| v != 0.0);
            match den.iter().position(|d| *d == num[i]).filter(|_| live) {
                Some(j) => {
                    den.remove(j);
                    num.remove(i);
                    cut = true;
                }
                None => i += 1,
            }
        }
        if !cut {
            return None;
        }
        // The literals lead the rebuilt product, so a scale and a folded constant
        // end up adjacent and the rational folds can gather them.
        num.sort_by_key(|f| usize::from(!matches!(f, Tree::Num(_))));
        Some(match (product(&num), product(&den)) {
            (None, None) => Tree::Num(1.0),
            (Some(n), None) => n,
            (None, Some(d)) => Tree::App(Op::Inv, vec![d]),
            (Some(n), Some(d)) => Tree::App(Op::Div, vec![n, d]),
        })
    }
    /// log(c * exp u) = u + log c for real u and c > 0: `Log arg` where `Exp u` is
    /// a FACTOR of `arg` and the rest of it, c, is positive on every row. sympy
    /// will not cancel log(exp u) without knowing u is real (feynman II.10.9 was
    /// the law behind 0.2*log(exp(5 u)/5) + 0.3218.., and scored as wrong).
    fn log_of_exp_factor(arg: &Tree, rows: &[Vec<(String, f64)>]) -> Option<Tree> {
        let (mut above, mut below, mut num, mut den) = (vec![], vec![], vec![], vec![]);
        factors(arg, false, &mut above, &mut below, &mut num, &mut den);
        if above.is_empty() && below.is_empty() {
            return None;
        }
        // exp never overflows and never underflows to 0 on the data: the log is finite.
        if !on_every_row(&Tree::App(Op::Log, vec![arg.clone()]), rows, |_| true) {
            return None;
        }
        let rest = match (product(&num), product(&den)) {
            (None, None) => None,
            (Some(n), None) => Some((n, true)),
            (None, Some(d)) => Some((d, false)),
            (Some(n), Some(d)) => Some((Tree::App(Op::Div, vec![n, d]), true)),
        };
        let mut sum: Option<Tree> = None;
        for u in above {
            sum = Some(sum.map_or_else(|| u.clone(), |s| Tree::App(Op::Add, vec![s, u.clone()])));
        }
        for u in below {
            sum = Some(sum.map_or_else(|| Tree::App(Op::Neg, vec![u.clone()]), |s| Tree::App(Op::Sub, vec![s, u.clone()])));
        }
        let sum = sum?;
        let Some((c, added)) = rest else { return Some(sum) };
        // A numeric constant is a plain fact; anything else the data must say.
        let log_c = match &c {
            Tree::Num(v) if *v > 0.0 => Tree::Num(v.ln()),
            _ if on_every_row(&c, rows, |v| v > 0.0) => Tree::App(Op::Log, vec![c]),
            _ => return None,
        };
        Some(Tree::App(if added { Op::Add } else { Op::Sub }, vec![sum, log_c]))
    }
    /// `t` with a `Log (Div a b)` factor turned over to `Log (Div b a)` — which
    /// negates `t`: log(b/a) = -log(a/b) where the quotient is positive on the data.
    fn turned_over(t: &Tree, rows: &[Vec<(String, f64)>]) -> Option<Tree> {
        match t {
            Tree::App(Op::Log, k) => match &k[0] {
                Tree::App(Op::Div, q) if on_every_row(&k[0], rows, |v| v > 0.0) => Some(Tree::App(Op::Log, vec![Tree::App(Op::Div, vec![q[1].clone(), q[0].clone()])])),
                _ => None,
            },
            Tree::App(Op::Mul, k) => (0..k.len()).find_map(|i| {
                let mut kids = k.clone();
                kids[i] = turned_over(&k[i], rows)?;
                Some(Tree::App(Op::Mul, kids))
            }),
            _ => None,
        }
    }
    /// `t` with a `Neg` factor taken off — which negates `t`.
    fn without_neg(t: &Tree) -> Option<Tree> {
        match t {
            Tree::App(Op::Neg, k) => Some(k[0].clone()),
            // a negative numeric factor carries the sign as well as a Neg does (the
            // fitted scale a of a * f(x) + b is where I.44.4's minus sign lived)
            Tree::Num(c) if *c < 0.0 => Some(Tree::Num(-c)),
            Tree::App(Op::Mul, k) => (0..k.len()).find_map(|i| {
                let mut kids = k.clone();
                kids[i] = without_neg(&k[i])?;
                Some(Tree::App(Op::Mul, kids))
            }),
            _ => None,
        }
    }
    /// `sqrt((a + b)/(a - b))` = `(1 + b/a) / sqrt(1 - (b/a)^2)`, where the data
    /// says a > 0 and a > |b| on every row — so both a + b and a - b are positive
    /// and the two sides are the same positive number. The RIGHT-HAND side is the
    /// bigger form: it is the canonical one for the sqrt(1 - v^2/c^2) family, which
    /// SRBench writes in the ratio b/a (feynman I.34.14's law is
    /// omega_0 (1 + v/c)/sqrt(1 - v^2/c^2), found as omega_0 sqrt((c+v)/(c-v)) and
    /// scored as wrong — sympy will not reconcile the two without positivity).
    /// `None` unless the quotient is a FACTOR of the radicand with that shape.
    fn doppler(radicand: &Tree, rows: &[Vec<(String, f64)>]) -> Option<(Tree, Vec<Tree>)> {
        let (mut num, mut den) = (vec![], vec![]);
        over_and_under(radicand, false, &mut num, &mut den);
        for i in 0..num.len() {
            let Tree::App(Op::Add, s) = num[i] else { continue };
            for j in 0..den.len() {
                let Tree::App(Op::Sub, d) = den[j] else { continue };
                // a - b below, a + b above, written either way round.
                let (a, b) = (&d[0], &d[1]);
                if !((s[0] == *a && s[1] == *b) || (s[1] == *a && s[0] == *b)) {
                    continue;
                }
                if !on_every_row(a, rows, |v| v > 0.0) || !on_every_row(den[j], rows, |v| v > 0.0) || !on_every_row(num[i], rows, |v| v > 0.0) {
                    continue;
                }
                let beta = Tree::App(Op::Div, vec![b.clone(), a.clone()]);
                let over = Tree::App(Op::Add, vec![Tree::Num(1.0), beta.clone()]);
                let under = Tree::App(Op::Sqrt, vec![Tree::App(Op::Sub, vec![Tree::Num(1.0), Tree::App(Op::Pow2, vec![beta])])]);
                // What else was in the radicand keeps ONE root of its own — over the
                // line, never as sqrt(1/x): a scorer reading the string with no
                // positivity does not take that for 1/sqrt(x).
                let others: Vec<&Tree> = num.iter().enumerate().filter(|(k, _)| *k != i).map(|(_, f)| *f).collect();
                let unders: Vec<&Tree> = den.iter().enumerate().filter(|(k, _)| *k != j).map(|(_, f)| *f).collect();
                if !unders.is_empty() {
                    return None;
                }
                return Some((Tree::App(Op::Div, vec![over, under]), others.into_iter().cloned().collect()));
            }
        }
        None
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
            // sqrt(e^2) = |e| for every real e — and the Abs then sheds the sign the
            // data settles. Written out as sqrt(x^2) an exact law is reported in a
            // form no scorer matches to it.
            Op::Sqrt if matches!(&kids[0], Tree::App(Op::Pow2, _)) => {
                let Tree::App(_, square) = &kids[0] else { return Tree::App(*op, kids) };
                return go(&Tree::App(Op::Abs, vec![square[0].clone()]), rows);
            }
            // ... and the root of a (a+b)/(a-b) quotient is the family's own form.
            Op::Sqrt => {
                if let Some((family, rest)) = doppler(&kids[0], rows) {
                    let whole = rest.into_iter().fold(family, |acc, f| Tree::App(Op::Mul, vec![go(&Tree::App(Op::Sqrt, vec![f]), rows), acc]));
                    return whole;
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
            // log|x| unless x is 0 or not finite (then +inf): where the data never
            // finds that, it IS log(Abs(x)) — and the Abs goes where x keeps one sign.
            Op::ProtectedLog => {
                if on_every_row(&kids[0], rows, |v| v != 0.0) {
                    return go(&Tree::App(Op::Log, vec![Tree::App(Op::Abs, kids)]), rows);
                }
            }
            // exp unless the argument is not finite: where the data never overflows
            // it, it IS exp.
            Op::ProtectedExp => {
                if on_every_row(&Tree::App(Op::ProtectedExp, kids.clone()), rows, |_| true) {
                    return Tree::App(Op::Exp, kids);
                }
            }
            Op::Log => {
                if let Some(sum) = log_of_exp_factor(&kids[0], rows) {
                    return go(&sum, rows);
                }
            }
            // log a - log b = log(a/b), log a + log b = log(a*b), where the data says
            // a > 0 and b > 0 on every row. sympy will not merge logs without knowing
            // that (feynman I.44.4 was the law, written -(log V1 - log V2), and scored
            // as wrong).
            Op::Sub | Op::Add => {
                // A log term and its sign: `Log a` is (a, +), `Neg (Log a)` is (a, -).
                // The chromosome writes the difference whichever way it grew — feynman
                // I.44.4 came as Add (Neg (Log V2)) (Log V1), not as a Sub.
                let term = |k: &Tree| match k {
                    Tree::App(Op::Log, a) if on_every_row(&a[0], rows, |v| v > 0.0) => Some((a[0].clone(), true)),
                    Tree::App(Op::Neg, n) => match &n[0] {
                        Tree::App(Op::Log, a) if on_every_row(&a[0], rows, |v| v > 0.0) => Some((a[0].clone(), false)),
                        _ => None,
                    },
                    _ => None,
                };
                if let (Some((a, plus_a)), Some((b, plus_b))) = (term(&kids[0]), term(&kids[1])) {
                    // the second term's sign as it enters the sum
                    let plus_b = if *op == Op::Sub { !plus_b } else { plus_b };
                    let log = |arg: Tree| Tree::App(Op::Log, vec![arg]);
                    let merged = match (plus_a, plus_b) {
                        (true, true) => log(Tree::App(Op::Mul, vec![a, b])),
                        (true, false) => log(Tree::App(Op::Div, vec![a, b])),
                        (false, true) => log(Tree::App(Op::Div, vec![b, a])),
                        (false, false) => Tree::App(Op::Neg, vec![log(Tree::App(Op::Mul, vec![a, b]))]),
                    };
                    return go(&merged, rows);
                }
            }
            // ... and the sign is put INSIDE the log: -log(a/b) is log(b/a), so the
            // reported form carries no leading negation.
            Op::Neg => {
                if let Some(turned) = turned_over(&kids[0], rows) {
                    return turned;
                }
            }
            Op::Mul if kids.len() == 2 => {
                for (i, j) in [(0, 1), (1, 0)] {
                    if let (Some(plain), Some(turned)) = (without_neg(&kids[i]), turned_over(&kids[j], rows)) {
                        let mut both = [plain, turned];
                        if i == 1 {
                            both.swap(0, 1);
                        }
                        return Tree::App(Op::Mul, both.to_vec());
                    }
                }
                if let Some(lowest) = cancelled(&Tree::App(*op, kids.clone()), rows) {
                    return lowest;
                }
            }
            Op::Div | Op::Inv => {
                if let Some(lowest) = cancelled(&Tree::App(*op, kids.clone()), rows) {
                    return lowest;
                }
            }
            // tan(asin u) = u / sqrt(1 - u^2), for |u| < 1 — the shape the
            // sqrt(1 - v^2/c^2) family is found in. The data supplies the condition:
            // where |u| < 1 on every row the arcsin is on its principal branch, its
            // cosine is the positive root, and the quotient is an identity. At
            // |u| = 1 both sides are infinite and the equality says nothing, so the
            // bound is STRICT. A protected arcsin whose clamp fires on some row is
            // left alone — its argument is not u there (feynman II.13.23 was reported
            // as tan(asin(v/c))/v, which no scorer matches to the law's root).
            Op::Tan if matches!(&kids[0], Tree::App(Op::ProtectedAsin | Op::Asin, _)) => {
                let Tree::App(_, inner) = &kids[0] else { return Tree::App(*op, kids) };
                let u = &inner[0];
                if on_every_row(u, rows, |v| v.abs() < 1.0) {
                    let root = Tree::App(Op::Sqrt, vec![Tree::App(Op::Sub, vec![Tree::Num(1.0), Tree::App(Op::Pow2, vec![u.clone()])])]);
                    return Tree::App(Op::Div, vec![u.clone(), root]);
                }
            }
            // A PHASE SHIFT by a quarter turn: sin(e + pi/2) is cos e, cos(e + pi/2)
            // is -sin e, and so on round the circle — where the constant IS k*pi/2 to
            // machine precision (the engine produces exactly f64 pi/2: a clamped
            // protected arcsin resolved to its constant). Rounded to 3 decimals by a
            // scorer, sin(y + 1.571) is not cos(y) (strogatz glider2).
            Op::Sin | Op::Cos => {
                use std::f64::consts::FRAC_PI_2;
                let shift = match &kids[0] {
                    Tree::App(Op::Add, s) => match (&s[0], &s[1]) {
                        (e, Tree::Num(c)) | (Tree::Num(c), e) => Some((e, *c)),
                        _ => None,
                    },
                    Tree::App(Op::Sub, s) => match (&s[0], &s[1]) {
                        (e, Tree::Num(c)) => Some((e, -*c)),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some((e, c)) = shift {
                    let k = (c / FRAC_PI_2).round();
                    if k != 0.0 && k.abs() <= 1e6 && (c - k * FRAC_PI_2).abs() <= 1e-9 {
                        // sin, cos, -sin, -cos: a quarter turn on from each is the next.
                        let quarter = (k.rem_euclid(4.0) as usize + usize::from(*op == Op::Cos)) % 4;
                        let turned = Tree::App(if quarter.is_multiple_of(2) { Op::Sin } else { Op::Cos }, vec![e.clone()]);
                        return if quarter < 2 { turned } else { Tree::App(Op::Neg, vec![turned]) };
                    }
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

/// Set in the generation that keys THE CROSS STEP's fresh rows; no fit runs this
/// many generations, so the key is never a pump's.
const CROSS_KEY: u32 = 1 << 31;

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
    /// Snap's resident parts; None when `Config::snap_every` is 0.
    snap: Option<SnapState>,
    /// IDENTITY, AGE and LINEAGE; None when `Config::genealogy_path` is None, and
    /// then nothing in the fit loop touches it and the engine is what it was.
    lineage: Option<LineageState>,
}

/// What the genealogy keeps for the length of a fit: the identities of the
/// population, and the file they are written to.
struct LineageState {
    tracker: Genealogy,
    log: GenealogyLog,
}

/// What a refill tells the genealogy: the marks as they stood BEFORE the step
/// (its rows move over one another, so the sources must be read first), the
/// generation, and the three words for this step's kinds of row — the pump's or
/// the cross step's. `before` is None when the genealogy is off, and then every
/// call through here does nothing.
struct Marks {
    before: Option<Vec<RowMark>>,
    generation: u32,
    /// A row of the intake's best fifth, kept through the step.
    keep: Origin,
    /// A row copied in from elsewhere: a promotion, or a migrant.
    arrive: Origin,
    /// A new random individual.
    fresh: Origin,
}

/// What the snap step keeps between beats: the lattice, the linter's literal
/// codes (`encode` reads them), the guard's f64 rows, and the counts.
struct SnapState {
    table: crate::lint::snap_table::SnapTable,
    kernel_tables: crate::lint::device::KernelTables,
    guard_data: crate::lint::snap_guard::GuardData,
    counts: SnapCounts,
}

impl Engine {
    pub fn new(config: Config, data: Data) -> Result<Engine, String> {
        let wide = SymbolTable::wide(data.names.len() as u32);
        let table = if config.compounds { wide.with_compounds() } else { wide };
        // The named constants join the table only when snap can write one: with the
        // switch off the symbol table is the one it always was.
        let lattice = if config.snap_every > 0 { Some(crate::lint::snap_table::SnapTable::standard()?) } else { None };
        let table = match &lattice {
            Some(lattice) => table.with_named(&lattice.names)?,
            None => table,
        };
        if config.n_pairs == 0 {
            return Err("n_pairs: a population is at least one pair of islands".into());
        }
        let pair = config.pop_intake + config.pop_champion;
        let pop = config.n_pairs * pair;
        let layout = Layout::for_arity(pop, config.n_genes, config.head, table.max_arity(), config.n_rnc);
        let tourn = |n: u32| ((config.tournament_fraction * f64::from(n)).round() as u32).max(2);
        // Pair p is islands 2p (intake) and 2p + 1 (champion), one pair after another.
        let islands: Vec<Island> = (0..config.n_pairs)
            .flat_map(|p| {
                let lo = p * pair;
                [
                    Island { lo, hi: lo + config.pop_intake, elites: config.elites, tournsize: tourn(config.pop_intake) },
                    Island { lo: lo + config.pop_intake, hi: lo + pair, elites: config.elites, tournsize: tourn(config.pop_champion) },
                ]
            })
            .collect();
        super::vary::validate(layout, &islands)?;
        if data.y.len() != data.splits.total() || data.x.len() != data.y.len() * data.names.len() {
            return Err("data: x, y and the splits do not agree".into());
        }
        let dev = EvolveDevice::new(layout, &table.codes())?;
        let evaluator = GpuEvaluator::new(&data.x, data.names.len())?;
        let scorer = GpuScorer::new(&evaluator, &data.y, data.splits)?;
        let caps = Caps::of(&data.y, data.splits);
        let snap = match lattice {
            Some(table) => {
                let tables = crate::lint::tables::Tables::standard()?;
                let packed = crate::lint::pack::pack(&tables.rules.iter().collect::<Vec<_>>());
                let kernel_tables = crate::lint::device::KernelTables::new(&packed.rules, packed.codes, &tables.guards)?;
                let x: Vec<f64> = data.x.iter().map(|v| f64::from(*v)).collect();
                let guard_data = crate::lint::snap_guard::GuardData::train(data.names.clone(), x, data.y.clone(), data.splits)?;
                Some(SnapState { table, kernel_tables, guard_data, counts: SnapCounts::default() })
            }
            None => None,
        };
        Ok(Engine { scored: vec![None; pop as usize], config, table, layout, islands, dev, evaluator, scorer, data, caps, col_max: None, snap, lineage: None })
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
        let combinations = self.combinations()?;
        let scores: Vec<f64> = if rows.is_empty() {
            Vec::new()
        } else {
            self.scorer.eval_and_score_with(&self.evaluator, &batch, &gene_ok, &chromosomes, &combinations)?.into_iter().map(f64::from).collect()
        };
        timing.evaluate += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let per = combinations.len() * WRAPPERS.len();
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
            for c in 0..per {
                let s = &scores[(i * per + c) * WIDTH..(i * per + c + 1) * WIDTH];
                if !s[0].is_finite() {
                    continue;
                }
                // The tower is over the candidate's USED genes: an unused gene's costs nothing.
                let combination = combinations[c / WRAPPERS.len()];
                let tower = combination.positions().map(|g| gene_tower[chromosomes[i][g]]).max().unwrap_or(0);
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
                    best = Some(Scored { fitness, linker: combination.linker, wrapper: c % WRAPPERS.len(), a: s[0], b: s[1], one_minus_r2: omr2, t_depth: tower, selection, genes: combination.genes });
                }
            }
            self.scored[r] = best;
            gen.fitness[r] = best.map_or(std::f32::consts::PI, |b| b.selection as f32);
        }
        timing.hff += t.elapsed().as_secs_f64();
        Ok((gene_ok.len() as u64, oversized))
    }

    /// THE CANDIDATES' gene-linker combinations (`chrom_score::gene_linker_combinations`):
    /// all the genes under each linker, or every non-empty subset when
    /// `Config::gene_subsets` is on — an error for more than 3 genes then.
    fn combinations(&self) -> Result<Vec<GeneLinker>, String> {
        gene_linker_combinations(self.layout.n_genes as usize, LINKERS.len(), self.config.gene_subsets)
    }

    /// The pairs of the population: (intake island, champion island).
    pub fn pairs(&self) -> Vec<(Island, Island)> {
        self.islands.chunks_exact(2).map(|p| (p[0], p[1])).collect()
    }

    /// An island's evaluated rows, fittest first (ties to the lower row).
    fn by_fitness(island: Island, fitness: &[f32]) -> Vec<u32> {
        let mut v: Vec<u32> = (island.lo..island.hi).filter(|&r| !fitness[r as usize].is_nan()).collect();
        v.sort_by(|&a, &b| fitness[a as usize].total_cmp(&fitness[b as usize]).then(a.cmp(&b)));
        v
    }

    /// The rows an intake island keeps when it is refilled: its best fifth, one of
    /// each distinct genome, fittest first.
    fn keepers(&self, intake: Island, gen: &Generation) -> Vec<u32> {
        let row_w = (self.layout.n_genes * self.layout.gene_width()) as usize;
        let keep = ((f64::from(intake.hi - intake.lo) * 0.20).round() as usize).max(1);
        let mut seen: Vec<&[u32]> = Vec::new();
        let mut keepers: Vec<u32> = Vec::new();
        for r in Self::by_fitness(intake, &gen.fitness) {
            let row = &gen.pop.genome[r as usize * row_w..(r as usize + 1) * row_w];
            if keepers.len() < keep && !seen.contains(&row) {
                seen.push(row);
                keepers.push(r);
            }
        }
        keepers
    }

    /// Rewrite an intake island: `keepers` (rows of `before`, with their scores)
    /// first, then `arrivals` (rows of `before`, to be evaluated again), then rows
    /// of `fresh` — new random individuals, unevaluated. Arrivals the island has no
    /// room for are left out.
    fn refill(&mut self, gen: &mut Generation, intake: Island, before: &Generation, keepers: &[u32], arrivals: &[u32], fresh: &Population, marks: &Marks) -> Result<(), String> {
        let l = self.layout;
        let row_w = (l.n_genes * l.gene_width()) as usize;
        let rnc_w = (l.n_genes * l.n_rnc) as usize;
        let scored_before = self.scored.clone();
        let room = (intake.hi - intake.lo) as usize;
        let sources = keepers.iter().map(|&r| (r, true)).chain(arrivals.iter().map(|&r| (r, false))).take(room);
        let mut to = intake.lo as usize;
        for (from, evaluated) in sources {
            let from = from as usize;
            gen.pop.genome[to * row_w..(to + 1) * row_w].copy_from_slice(&before.pop.genome[from * row_w..(from + 1) * row_w]);
            gen.pop.rnc[to * rnc_w..(to + 1) * rnc_w].copy_from_slice(&before.pop.rnc[from * rnc_w..(from + 1) * rnc_w]);
            gen.pop.wrapper_id[to] = before.pop.wrapper_id[from];
            gen.fitness[to] = if evaluated { before.fitness[from] } else { f32::NAN };
            self.scored[to] = if evaluated { scored_before[from] } else { None };
            // A keeper is the same individual in another row of its own island; an
            // arrival is a CLONE of a champion row that stays where it is.
            self.track_move(marks, from, to, if evaluated { marks.keep } else { marks.arrive })?;
            to += 1;
        }
        let fresh_from = to;
        for r in to..intake.hi as usize {
            gen.pop.genome[r * row_w..(r + 1) * row_w].copy_from_slice(&fresh.genome[r * row_w..(r + 1) * row_w]);
            gen.pop.rnc[r * rnc_w..(r + 1) * rnc_w].copy_from_slice(&fresh.rnc[r * rnc_w..(r + 1) * rnc_w]);
            gen.pop.wrapper_id[r] = fresh.wrapper_id[r];
            gen.fitness[r] = f32::NAN;
            self.scored[r] = None;
        }
        self.track_fresh(fresh_from, intake.hi as usize, marks.fresh, marks.generation)
    }

    /// What a refill needs from the genealogy: the marks as they stood BEFORE the
    /// step (rows move over one another), the generation, and which origin each
    /// kind of row takes — the pump's words or the cross step's.
    /// THE BEST INDIVIDUAL of a generation, one line in the log — who it is, how
    /// old and what it descends from. Nothing when the genealogy is off.
    fn log_best(&mut self, generation: u32, gen: &Generation, timing: &mut Timing) -> Result<(), String> {
        // Off costs nothing at all: not the clock, not the search for the best row.
        if self.lineage.is_none() {
            return Ok(());
        }
        let t = Instant::now();
        let row = self.best(gen).map(|(r, _)| r);
        if let (Some(l), Some(row)) = (self.lineage.as_mut(), row) {
            let record = l.tracker.record(generation, row);
            l.log.record("best", &record)?;
        }
        timing.genealogy += t.elapsed().as_secs_f64();
        Ok(())
    }

    /// The marks as they stand, with the words a step uses for its kinds of row.
    fn marks(&self, generation: u32, keep: Origin, arrive: Origin, fresh: Origin) -> Marks {
        Marks { before: self.lineage.as_ref().map(|l| l.tracker.snapshot()), generation, keep, arrive, fresh }
    }

    /// One row of a refill: the same individual moved (a keeper), or a clone of a
    /// row that stays where it is (a promotion, a migrant). Either way it is an
    /// ARRIVAL in the log — a row this generation did not breed.
    fn track_move(&mut self, marks: &Marks, from: usize, to: usize, origin: Origin) -> Result<(), String> {
        let (Some(l), Some(before)) = (self.lineage.as_mut(), marks.before.as_ref()) else { return Ok(()) };
        l.tracker.moved(before, from, to, origin, marks.generation);
        // A keeper mints no id — it is the same individual in another row — so the
        // log is told the EVENT, not how that individual was originally born.
        let record = l.tracker.record_as(marks.generation, to, origin);
        l.log.record("arrival", &record)
    }

    /// A run of fresh random individuals: one batch line in the log, not one per row.
    fn track_fresh(&mut self, from: usize, to: usize, origin: Origin, generation: u32) -> Result<(), String> {
        let Some(l) = self.lineage.as_mut() else { return Ok(()) };
        if from >= to {
            return Ok(());
        }
        let id_lo = l.tracker.minted();
        for r in from..to {
            l.tracker.fresh(r, origin, generation);
        }
        let id_hi = l.tracker.minted() - 1;
        l.log.batch(generation, from as u32, to as u32 - 1, id_lo, id_hi, origin)
    }

    /// New random individuals for a refill: the CPU reference of the init kernel,
    /// keyed by `key`, so a refill is reproducible. A row of it is taken whole, at
    /// its own place, so every island of every pair draws different individuals.
    fn fresh(&self, key: u32, generation: u32) -> Result<Population, String> {
        super::init(self.layout, &self.table.codes(), &InitParams {
            seed: self.config.seed, generation: key, rnc_lo: self.config.rnc_lo, rnc_hi: self.config.rnc_hi, n_wrappers: WRAPPERS.len() as u32,
            vhead: self.vhead_at(generation),     // fresh rows are born at today's virtual head
        })
    }

    /// The pump, in every pair: the intake's best two replace its champion
    /// island's worst two; the intake keeps its best fifth (one of each distinct
    /// row) and the rest is refilled with new random individuals.
    fn pump(&mut self, gen: &mut Generation, generation: u32) -> Result<(), String> {
        // Fresh rows are keyed by this generation.
        let fresh = self.fresh(generation, generation)?;
        for (intake, champion) in self.pairs() {
            self.pump_pair(gen, intake, champion, &fresh, generation)?;
        }
        Ok(())
    }

    /// The pump inside one pair.
    fn pump_pair(&mut self, gen: &mut Generation, intake: Island, champion: Island, fresh: &Population, generation: u32) -> Result<(), String> {
        let l = self.layout;
        let row_w = (l.n_genes * l.gene_width()) as usize;
        let rnc_w = (l.n_genes * l.n_rnc) as usize;
        let best_intake = Self::by_fitness(intake, &gen.fitness);
        // The champion island's worst first; an unevaluated row is the worst of all.
        let mut worst_champion: Vec<u32> = (champion.lo..champion.hi).collect();
        worst_champion.sort_by(|&a, &b| gen.fitness[b as usize].total_cmp(&gen.fitness[a as usize]).then(b.cmp(&a)));
        for (&from, &to) in best_intake.iter().zip(worst_champion.iter().take(2)) {
            let (from, to) = (from as usize, to as usize);
            gen.pop.genome.copy_within(from * row_w..(from + 1) * row_w, to * row_w);
            gen.pop.rnc.copy_within(from * rnc_w..(from + 1) * rnc_w, to * rnc_w);
            gen.pop.wrapper_id[to] = gen.pop.wrapper_id[from];
            gen.fitness[to] = gen.fitness[from];
            self.scored[to] = self.scored[from];
            // THE PROMOTION: the intake row is copied over a champion row and stays
            // where it is, so the copy is a new individual of the same line.
            if let Some(l) = self.lineage.as_mut() {
                l.tracker.promote(from, to, generation);
                let record = l.tracker.record(generation, to);
                l.log.record("arrival", &record)?;
            }
        }
        let keepers = self.keepers(intake, gen);
        let before = gen.clone();
        let marks = self.marks(generation, Origin::PumpKeep, Origin::PumpPromote, Origin::PumpRefill);
        self.refill(gen, intake, &before, &keepers, &[], fresh, &marks)
    }

    /// THE CROSS STEP, between pairs (the notebook's `_migrate_pump_cross`). First
    /// every champion island names its best `k_migrants` — from the population as
    /// it stands, before any intake is rewritten. Then every intake island keeps
    /// its best fifth (one of each distinct row) and the rest is filled FIRST with
    /// the migrants of the OTHER pairs — never its own pair's — in pair order, then
    /// champion rank, and THEN with new random individuals. Arrivals are clones
    /// and are evaluated again; champion islands are not touched.
    ///
    /// Stated rules: a champion island smaller than `k_migrants` sends all it has;
    /// an intake with no room for every arrival takes the first that fit (the
    /// notebook's `cross_pool[:n_to_fill]`). With one pair there are no other
    /// champions and the step is a keep-the-fifth refill, as it is in the notebook.
    ///
    /// The notebook re-pins arrivals to the receiving pair's WRAPPER class. There
    /// is nothing here for that to act on: this engine scores every chromosome
    /// under all wrappers and keeps its best, so no pair has a wrapper class.
    ///
    /// Fresh rows are keyed by the generation with [`CROSS_KEY`] set: when the pump
    /// and the cross step share a beat, the cross step's new individuals are not
    /// the ones the pump has just drawn for the same rows.
    fn cross(&mut self, gen: &mut Generation, generation: u32) -> Result<(), String> {
        let fresh = self.fresh(generation | CROSS_KEY, generation)?;
        let pairs = self.pairs();
        let migrants: Vec<Vec<u32>> =
            pairs.iter().map(|&(_, champion)| Self::by_fitness(champion, &gen.fitness).into_iter().take(self.config.k_migrants as usize).collect()).collect();
        let before = gen.clone();
        let marks = self.marks(generation, Origin::CrossKeep, Origin::CrossArrival, Origin::CrossFresh);
        for (p, &(intake, _)) in pairs.iter().enumerate() {
            let keepers = self.keepers(intake, &before);
            let arrivals: Vec<u32> = migrants.iter().enumerate().filter(|(q, _)| *q != p).flat_map(|(_, m)| m.iter().copied()).collect();
            self.refill(gen, intake, &before, &keepers, &arrivals, &fresh, &marks)?;
        }
        Ok(())
    }

    /// The row's best candidate re-scored in f64 by `chrom_score`, the
    /// definition: what may stop a fit or be reported. The device's f32 metrics
    /// only rank.
    fn confirm(&self, gen: &Generation, row: usize) -> Result<Option<Scored>, String> {
        let l = self.layout;
        let (width, nr, g_n) = (l.gene_width() as usize, l.n_rnc as usize, l.n_genes as usize);
        let mut batch = ExprBatch::new();
        let mut gene_ok = Vec::with_capacity(g_n);
        let mut gene_tower = Vec::with_capacity(g_n);
        for g in 0..g_n {
            let tokens = &gen.pop.genome[(row * g_n + g) * width..(row * g_n + g + 1) * width];
            let consts = &gen.pop.rnc[(row * g_n + g) * nr..(row * g_n + g + 1) * nr];
            match decode_gene(tokens, consts, l, &self.table).filter(|n| n.len() <= MAX_NODES) {
                Some(nodes) => {
                    gene_tower.push(t_depth(&nodes));
                    batch.push(&nodes);
                    gene_ok.push(true);
                }
                None => {
                    batch.push(&[GpuNode { op: Op::Num as u32, arg0: 0, arg1: 0, konst: 0.0 }]);
                    gene_ok.push(false);
                    gene_tower.push(0);
                }
            }
        }
        let preds = self.evaluator.eval(&batch)?;
        let spec = ScoreSpec { linkers: LINKERS.to_vec(), wrappers: WRAPPERS.to_vec(), splits: self.data.splits, linear_scaling: true };
        // The same candidates as `evaluate`, in the same order, in f64.
        let combinations = self.combinations()?;
        let scores = score_gene_subsets(&preds, &gene_ok, &[(0..g_n).collect()], &self.data.y, &spec, &combinations)?;
        let n_ex = self.data.splits.n_extrap;
        let col_max = self.col_max.unwrap_or([1.0; 9]);
        let mut best: Option<Scored> = None;
        for c in 0..combinations.len() * WRAPPERS.len() {
            let s = &scores[c * SCORE_WIDTH..c * SCORE_WIDTH + METRIC_WIDTH];
            if !s[0].is_finite() {
                continue;
            }
            let combination = combinations[c / WRAPPERS.len()];
            let tower = combination.positions().map(|g| gene_tower[g]).max().unwrap_or(0);
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
                best = Some(Scored { fitness, linker: combination.linker, wrapper: c % WRAPPERS.len(), a: s[0], b: s[1], one_minus_r2: omr2, t_depth: tower, selection: fitness, genes: combination.genes });
            }
        }
        Ok(best)
    }

    /// SNAP WINNERS (`hff_sr_engine.py::_apply_snap_to_winners`, per gene as
    /// `_snap_op.py::snap_individual`): the best `snap_top_k` evaluated rows of
    /// every island (0 = all of them). Each gene with a folded constant that is
    /// not a whole number is ONE expression for the snap pipeline — the row's whole
    /// model `a * WRAPPER(LINKER(genes)) + b` with its fitted a and b, that gene's
    /// constants offered and every other literal withheld — so the guard judges
    /// the MODEL on the train rows. Match, graft and guard run on the evaluator's
    /// device; what is written back comes from `Guarded::decisions`, the host's
    /// f64 word, never from the device's verdicts. A row whose genome changed is
    /// left unevaluated. Returns how many rows changed; nothing raises into
    /// evolution — a gene that cannot take its form is unchanged and COUNTED.
    fn snap_winners(&mut self, gen: &mut Generation, generation: u32) -> Result<u64, String> {
        use crate::lint::snap_graft::{variant_sites, SnapGraft};
        use crate::lint::snap_guard::{SnapGuard, Verdict};
        use crate::lint::snap_table::SnapKernel;
        let started = Instant::now();
        let Some(state) = self.snap.as_ref() else { return Ok(0) };
        let l = self.layout;
        let (width, nr, g_n) = (l.gene_width() as usize, l.n_rnc as usize, l.n_genes as usize);
        let codes = self.table.codes();
        let mut counts = SnapCounts { beats: 1, ..SnapCounts::default() };
        let mut winners: Vec<u32> = Vec::new();
        for island in &self.islands {
            let ranked = Self::by_fitness(*island, &gen.fitness);
            let take = if self.config.snap_top_k == 0 { ranked.len() } else { self.config.snap_top_k as usize };
            winners.extend(ranked.into_iter().take(take));
        }
        // One expression per (row, gene) with something to offer.
        let mut exprs = Vec::new();
        let mut offered: Vec<Vec<bool>> = Vec::new();
        let mut sites: Vec<Vec<Option<usize>>> = Vec::new();
        let mut owner: Vec<(usize, usize)> = Vec::new();
        for &r in &winners {
            let r = r as usize;
            counts.rows += 1;
            let gene = |g: usize| (&gen.pop.genome[(r * g_n + g) * width..(r * g_n + g + 1) * width], &gen.pop.rnc[(r * g_n + g) * nr..(r * g_n + g + 1) * nr]);
            let decodes = |g: usize| decode_gene(gene(g).0, gene(g).1, l, &self.table).is_some_and(|n| n.len() <= MAX_NODES);
            let forms: Option<Vec<_>> = (0..g_n).map(|g| gene_form(gene(g).0, gene(g).1, l, &self.table, &codes, true).filter(|_| decodes(g))).collect();
            let (Some(s), Some(forms)) = (self.scored[r], forms) else {
                counts.rows_unscored += 1;
                continue;
            };
            for g in 0..g_n {
                counts.genes_examined += 1;
                if forms[g].offered() == 0 {
                    continue;
                }
                counts.genes_with_literal += 1;
                let genes: Vec<_> = forms.iter().enumerate().map(|(k, f)| if k == g { f.clone() } else { f.withheld() }).collect();
                let (flat, at) = model_form(&genes, LINKER_NAMES[s.linker], WRAPPERS[s.wrapper], s.a, s.b).flatten(&self.data.names);
                if flat.nodes.len() > MAX_NODES {
                    counts.model_oversize += 1;
                    continue;
                }
                counts.literals_offered += at.iter().flatten().count() as u64;
                offered.push(at.iter().map(Option::is_some).collect());
                exprs.push(flat);
                sites.push(at);
                owner.push((r, g));
            }
        }
        let mut changed: Vec<usize> = Vec::new();
        if !exprs.is_empty() {
            let guarded = {
                let kernel = SnapKernel::on_device(self.evaluator.device(), self.evaluator.queue(), state.table.clone())?;
                let graft = SnapGraft::new(&kernel)?;
                let mut guard = SnapGuard::new(&self.evaluator, &kernel, &graft, state.guard_data.clone())?;
                guard.r2_drop_tol = self.config.snap_r2_drop;
                let (guarded, blocks) =
                    guard.run_resident_offered(&exprs, Some(&offered), &state.kernel_tables, self.config.snap_rel_tol, crate::gpu_eval::MAX_GROUPS_PER_DIM)?;
                // The write-back below is the host's: the device's blocks are not read.
                if let Some(blocks) = blocks {
                    blocks.destroy();
                    self.evaluator.device().poll(wgpu::Maintain::Poll);
                }
                guarded
            };
            counts.band_overflows = guarded.band_overflows as u64;
            counts.device_kept = guarded.device_kept as u64;
            counts.refused_f64 = (guarded.refused_literal + guarded.refused_r2) as u64;
            let vhead = super::virtual_head(self.vhead_at(generation), l)?;
            for (e, expr) in exprs.iter().enumerate() {
                let hits = &guarded.hits[e];
                for hit in hits.iter().flatten() {
                    counts.literals_matched += 1;
                    counts.matched_by_family[state.table.family(hit.entry) as usize] += 1;
                }
                let decision = &guarded.decisions[e];
                let Some(slot) = decision.slot.filter(|_| decision.status == Verdict::Kept) else { continue };
                let grafts: Vec<_> = variant_sites(expr, hits, slot)
                    .into_iter()
                    .map(|i| Ok((sites[e][i].ok_or("the guard kept a variant that grafts a literal snap did not offer")?, hits[i].ok_or("a grafted site with no hit")?)))
                    .collect::<Result<_, String>>()?;
                let (r, g) = owner[e];
                let tokens = &mut gen.pop.genome[(r * g_n + g) * width..(r * g_n + g + 1) * width];
                let consts = &mut gen.pop.rnc[(r * g_n + g) * nr..(r * g_n + g + 1) * nr];
                let status = write_back(tokens, consts, l, vhead, &self.table, &grafts, &state.table);
                counts.count(status);
                if status == WriteBack::Grafted && !changed.contains(&r) {
                    changed.push(r);
                }
            }
        }
        for &r in &changed {
            gen.fitness[r] = f32::NAN;
            self.scored[r] = None;
            // The guard keeps only a form that computes the same model within its
            // R² tolerance, so the write-back REPAIRS the individual: it keeps its
            // id, its age and its line, and the log records that it happened.
            if let Some(l) = self.lineage.as_mut() {
                let record = l.tracker.snap(r, generation);
                l.log.record("arrival", &record)?;
            }
        }
        counts.rows_changed = changed.len() as u64;
        counts.seconds = started.elapsed().as_secs_f64();
        if let Some(state) = self.snap.as_mut() {
            state.counts.add(&counts);
        }
        Ok(changed.len() as u64)
    }

    /// The population as the device holds it: after a fit, the last generation
    /// with every write-back in it.
    pub fn population(&self) -> Result<Population, String> {
        self.dev.read()
    }

    /// The snap counts of this engine's fits so far.
    pub fn snap_counts(&self) -> SnapCounts {
        self.snap.as_ref().map_or_else(SnapCounts::default, |s| s.counts.clone())
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
            mark: self.lineage.as_ref().map(|l| l.tracker.row(row)),
        });
    }

    /// One row of the logbook, and the hall of fame's best appended to its file.
    fn report(&self, generation: u32, seconds: f64, gen: &Generation, hof: Option<&HallOfFame>) -> Result<(), String> {
        let fitness: Vec<f64> = gen.fitness.iter().filter(|f| !f.is_nan()).map(|f| f64::from(*f)).collect();
        let avg = fitness.iter().sum::<f64>() / fitness.len().max(1) as f64;
        let Some((best_row, b)) = self.best(gen) else { return Ok(()) };
        let third = |s: &Scored| if self.data.splits.n_extrap > 0 { format!("{:.10}", 1.0 - s.one_minus_r2[2]) } else { "-".to_string() };
        let (_, log10_p) = hff_p_value(b.fitness, self.hff_dimensions());
        let head = match self.vhead_at(generation) { 0 => self.layout.head, v => v };
        // The lineage columns come after everything that was already reported, so a
        // harness that reads the old ones by position still finds them.
        let lineage = match self.lineage.as_ref().map(|l| l.tracker.row(best_row)) {
            // The origin is a word, so it is left-aligned under its heading; the
            // two numbers are right-aligned like every other column.
            Some(m) => format!("{:>9}{:>11}   {:<12}", m.age, m.founder_generation, m.founder_origin),
            None => String::new(),
        };
        eprintln!(
            "{generation:>7}{seconds:>8.0}{head:>6}{:>13.6e}{avg:>13.6e}{:>13.4e}{:>15.10}{:>15.10}{:>15}{:>9}{log10_p:>10.2}{lineage}",
            b.fitness, b.one_minus_r2[0] * self.caps.var[0], 1.0 - b.one_minus_r2[0], 1.0 - b.one_minus_r2[1], third(&b), b.t_depth
        );
        if let (Some(path), Some(h)) = (&self.config.hof_path, hof) {
            use std::io::Write;
            let model = crate::lint::node::Tree::parse(&h.math).map_or_else(|_| h.math.clone(), |t| t.to_infix());
            let (_, hof_log10_p) = hff_p_value(h.best.fitness, self.hff_dimensions());
            let mut file = std::fs::OpenOptions::new().append(true).open(path).map_err(|e| format!("hall of fame file {path}: {e}"))?;
            // The hall of fame's own winner: its age and line as it was when the
            // fit remembered it, after the model, so the old columns do not move.
            let lineage = match h.mark {
                Some(m) => format!("\t{}\t{}\t{}", m.age, m.founder_generation, m.founder_origin),
                None => String::new(),
            };
            writeln!(
                file,
                "{generation}\t{}\t{:.6e}\t{hof_log10_p:.2}\t{:.4e}\t{:.10}\t{:.10}\t{}\t{}\t{model}{lineage}",
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

    /// `a * WRAPPER(LINKER(genes)) + b` as a fuller `Math` expression — over the
    /// genes the model USES (`Scored::genes`): a single gene stands alone, with no
    /// linker; avg divides by the number of used genes, as `chrom_score` does.
    pub fn math_of(&self, gen: &Generation, row: usize, s: &Scored) -> String {
        let l = self.layout;
        let (width, nr, g_n) = (l.gene_width() as usize, l.n_rnc as usize, l.n_genes as usize);
        let genes: Vec<String> = (0..g_n)
            .filter(|g| s.genes >> g & 1 == 1)
            .map(|g| {
                let tokens = &gen.pop.genome[(row * g_n + g) * width..(row * g_n + g + 1) * width];
                let consts = &gen.pop.rnc[(row * g_n + g) * nr..(row * g_n + g + 1) * nr];
                decode_gene(tokens, consts, l, &self.table).map_or("(Num 0.0)".to_string(), |n| nodes_to_math_named(&n, 0, &self.data.names, &self.table.named_values()))
            })
            .collect();
        let mut body = genes[0].clone();
        for g in &genes[1..] {
            body = if LINKER_NAMES[s.linker] == "mulval" { format!("(Mul {body} {g})") } else { format!("(Add {body} {g})") };
        }
        if LINKER_NAMES[s.linker] == "avgval" && genes.len() > 1 {
            body = format!("(Div {body} (Num {:?}))", genes.len() as f64);
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
        let mut timing = Timing { vary: 0.0, read: 0.0, decode: 0.0, evaluate: 0.0, score: 0.0, hff: 0.0, pump: 0.0, cross: 0.0, snap: 0.0, genealogy: 0.0 };
        let rates = Rates::with_cleanse(self.layout, c.cleanse);
        self.dev.init(&InitParams { seed: c.seed, generation: 0, rnc_lo: c.rnc_lo, rnc_hi: c.rnc_hi, n_wrappers: WRAPPERS.len() as u32, vhead: self.vhead_at(0) })?;
        let mut gen = self.dev.read_generation()?;
        gen.fitness.fill(f32::NAN);
        let (mut unique, mut oversized) = self.evaluate(&mut gen, &mut timing)?;
        let mut individuals = u64::from(self.layout.pop);
        self.dev.write_fitness(&gen.fitness)?;
        let (mut generation, mut stopped_by) = (0u32, "n_gen");
        // THE GENEALOGY, when it is on: the initial draw is the population's
        // founders, one line each, all age 0. A fit starts its own count, as the
        // hall of fame's file starts its own.
        self.lineage = match &c.genealogy_path {
            Some(path) => {
                let t = Instant::now();
                let mut log = GenealogyLog::create(path)?;
                let tracker = Genealogy::init(self.layout.pop);
                log.batch(0, 0, self.layout.pop - 1, 0, u64::from(self.layout.pop) - 1, Origin::Init)?;
                timing.genealogy += t.elapsed().as_secs_f64();
                Some(LineageState { tracker, log })
            }
            None => None,
        };
        let mut hof: Option<HallOfFame> = None;
        self.remember(&mut hof, &gen, 0);
        self.log_best(0, &gen, &mut timing)?;
        if c.progress_every > 0 {
            eprintln!("{REPORT_HEADER}{}", if c.genealogy_path.is_some() { LINEAGE_HEADER } else { "" });
        }
        if let Some(path) = &c.hof_path {
            std::fs::write(path, if c.genealogy_path.is_some() {
                "reported_at_gen\tfound_at_gen\thff\tlog10_p\tmse_train\tr2_train\tr2_val_bl2\tr2_val_bk3\tt_depth\tmodel\tage\tfounder_gen\tfounder_origin\n"
            } else {
                "reported_at_gen\tfound_at_gen\thff\tlog10_p\tmse_train\tr2_train\tr2_val_bl2\tr2_val_bk3\tt_depth\tmodel\n"
            }).map_err(|e| format!("hall of fame file {path}: {e}"))?;
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
            // THE LINEAGE EDGE, from the kernel that computed it: for every row of
            // the generation just made, the row of the one before it was cloned
            // from. Read back only when the genealogy is on.
            if self.lineage.is_some() {
                let t = Instant::now();
                let parent = self.dev.read_parent()?;
                let islands = self.islands.clone();
                if let Some(l) = self.lineage.as_mut() {
                    l.tracker.advance(&parent, &islands, generation);
                }
                timing.genealogy += t.elapsed().as_secs_f64();
            }
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
            self.log_best(generation, &gen, &mut timing)?;
            // The device's f32 metrics cannot resolve 1e-10; they can say "this
            // one is worth confirming". The f64 re-score decides.
            if let Some((row, ranked)) = self.best(&gen) {
                if ranked.one_minus_r2[1] <= 1e-5 {
                    if let Some(s) = self.confirm(&gen, row)? {
                        let edge_ok = c.smogd || self.data.splits.n_extrap == 0 || s.one_minus_r2[2] <= c.stop_one_minus_r2;
                        // `confirm` scores TrueNorth over the error blocks and t_depth
                        // (not redundancy): that is the sphere its angle lives on.
                        let (_, log10_p) = hff_p_value(s.fitness, self.hff_dimensions() - usize::from(c.redundancy));
                        if s.one_minus_r2[1] <= c.stop_one_minus_r2 && edge_ok && log10_p <= c.stop_log10_p {
                            stopped_by = "early_stop";
                            break;
                        }
                    }
                }
            }
            // SNAP WINNERS, on its beat: the rows just scored, before the pump moves
            // them and before they breed. A row whose gene was written back is scored
            // again, as the pump's fresh rows are.
            if c.snap_every > 0 && generation % c.snap_every == 0 {
                let t = Instant::now();
                if self.snap_winners(&mut gen, generation)? > 0 {
                    let (u, o) = self.evaluate(&mut gen, &mut timing)?;
                    unique += u;
                    oversized += o;
                    self.remember(&mut hof, &gen, generation);
                    self.dev.write_population(&gen.pop)?;
                }
                timing.snap += t.elapsed().as_secs_f64();
            }
            // THE PUMP, on its beat: the islands are where the diversity comes from.
            let t = Instant::now();
            let crossed = c.cross_every > 0 && generation % c.cross_every == 0;
            if c.pump_every > 0 && generation % c.pump_every == 0 {
                self.pump(&mut gen, generation)?;
                let (u, o) = self.evaluate(&mut gen, &mut timing)?;
                unique += u;
                oversized += o;
                if !crossed {
                    self.dev.write_population(&gen.pop)?;
                }
            }
            timing.pump += t.elapsed().as_secs_f64();
            // THE CROSS STEP, on its own beat, and AFTER the pump when they share one
            // (the notebook's order): it ranks the intake islands the pump has just
            // refilled and evaluated. The population goes back to the device once.
            let t = Instant::now();
            if crossed {
                self.cross(&mut gen, generation)?;
                let (u, o) = self.evaluate(&mut gen, &mut timing)?;
                unique += u;
                oversized += o;
                self.dev.write_population(&gen.pop)?;
            }
            timing.cross += t.elapsed().as_secs_f64();
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
        // Whether the row reported is the hall of fame's winner written over a row
        // of the population, or the population's own best. The genealogy needs to
        // know: the row it was written over belongs to somebody else.
        let mut hof_restored = false;
        if let Some(h) = hof.as_ref().filter(|h| c.balanced_tournaments && h.best.fitness < ranked.fitness) {
            let l = self.layout;
            let (row_w, rnc_w) = ((l.n_genes * l.gene_width()) as usize, (l.n_genes * l.n_rnc) as usize);
            gen.pop.genome[row * row_w..(row + 1) * row_w].copy_from_slice(&h.genome);
            gen.pop.rnc[row * rnc_w..(row + 1) * rnc_w].copy_from_slice(&h.rnc);
            gen.pop.wrapper_id[row] = h.wrapper_id;
            ranked = h.best;
            row = row.min(l.pop as usize - 1);
            hof_restored = true;
        }
        let best = self.confirm(&gen, row)?.unwrap_or(ranked);
        // THE WINNER'S LINEAGE: its mark, and its whole chain back to its founder
        // appended to the log. When the hall of fame's winner was put back into a
        // row (balanced tournaments), the lineage reported is the one it had when
        // it was remembered — the row it was written over is somebody else.
        let mut winner = None;
        let mut population_ages = None;
        let (mut minted, mut lines, mut bytes) = (0u64, 0u64, 0u64);
        if self.lineage.is_some() {
            let t = Instant::now();
            // THE FINAL POPULATION: its ages, and how many lines its best rows come
            // from. Ranked fittest first over the rows that were scored.
            let mut ranked: Vec<usize> = (0..self.layout.pop as usize).filter(|&r| self.scored[r].is_some() && !gen.fitness[r].is_nan()).collect();
            ranked.sort_by(|&a, &b| self.scored[a].map_or(f64::MAX, |s| s.fitness).total_cmp(&self.scored[b].map_or(f64::MAX, |s| s.fitness)).then(a.cmp(&b)));
            population_ages = self.lineage.as_ref().and_then(|l| l.tracker.population_ages(&ranked));
            if let (Some(l), Some(ages)) = (self.lineage.as_mut(), population_ages) {
                let line = l.tracker.population_record(generation, &ages);
                l.log.line(&line)?;
            }
            // The hall of fame's mark is the winner's as it stood when it was
            // remembered; only use it when that winner was actually put back.
            winner = match hof.as_ref().filter(|_| hof_restored).and_then(|h| h.mark) {
                Some(mark) => Some(mark),
                None => self.lineage.as_ref().map(|l| l.tracker.row(row)),
            };
            if let Some(l) = self.lineage.as_mut() {
                if let Some(mark) = winner {
                    let chain = l.tracker.walk(mark.id);
                    l.log.lineage_of_winner(&chain)?;
                }
                (minted, lines, bytes) = (l.tracker.minted(), l.log.lines, l.log.bytes);
            }
            timing.genealogy += t.elapsed().as_secs_f64();
        }
        Ok(FitResult {
            lineage: winner,
            population_ages,
            genealogy_minted: minted,
            genealogy_lines: lines,
            genealogy_bytes: bytes,
            generations: generation,
            seconds: started.elapsed().as_secs_f64(),
            individuals,
            unique_genes: unique,
            oversized_genes: oversized,
            stopped_by,
            math: self.math_of(&gen, row, &best),
            best,
            timing,
            snap: self.snap.as_ref().map_or_else(SnapCounts::default, |s| s.counts.clone()),
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

    /// `decode_gene` as it was before compounds: the gene's own order IS level order.
    fn decode_gene_in_gene_order(gene: &[u32], rnc: &[f32], layout: Layout, table: &SymbolTable) -> Option<Vec<GpuNode>> {
        let ht = (layout.head + layout.tail) as usize;
        let (mut need, mut n) = (1i64, 0usize);
        while need > 0 && n < ht {
            need += i64::from(table.arity(gene[n])) - 1;
            n += 1;
        }
        if need > 0 {
            return None;
        }
        let (mut child, mut n_rnc) = (1u32, 0usize);
        let mut nodes = Vec::new();
        for &id in &gene[..n] {
            nodes.push(match table.symbols[id as usize] {
                Symbol::Function(op) => {
                    let a = op.arity() as u32;
                    child += a;
                    GpuNode { op: op as u32, arg0: child - a, arg1: if a == 2 { child - 1 } else { 0 }, konst: 0.0 }
                }
                Symbol::Input(col) => GpuNode { op: Op::Var as u32, arg0: col, arg1: 0, konst: 0.0 },
                Symbol::Constant(v) => GpuNode { op: Op::Num as u32, arg0: 0, arg1: 0, konst: v },
                Symbol::Rnc => {
                    let k = *gene.get(ht + n_rnc)? as usize;
                    n_rnc += 1;
                    GpuNode { op: Op::Num as u32, arg0: 0, arg1: 0, konst: *rnc.get(k)? }
                }
                Symbol::Compound(_) | Symbol::Named(_) => return None,
            });
        }
        Some(nodes)
    }

    #[test]
    fn a_gene_without_a_compound_decodes_exactly_as_it_always_did() {
        let table = SymbolTable::wide(3);
        let layout = Layout::for_arity(300, 3, 34, table.max_arity(), 10);
        let pop = crate::evolve::init(layout, &table.codes(), &crate::evolve::InitParams { seed: 9, generation: 0, rnc_lo: -5, rnc_hi: 5, n_wrappers: 3, vhead: 0 }).unwrap();
        let (width, nr) = (layout.gene_width() as usize, layout.n_rnc as usize);
        let mut decoded = 0;
        for (g, gene) in pop.genome.chunks(width).enumerate() {
            let rnc = &pop.rnc[g * nr..(g + 1) * nr];
            let (now, before) = (decode_gene(gene, rnc, layout, &table), decode_gene_in_gene_order(gene, rnc, layout, &table));
            assert_eq!(now.is_some(), before.is_some(), "gene {g}");
            if let (Some(now), Some(before)) = (now, before) {
                assert_eq!(now.len(), before.len(), "gene {g}");
                assert!(now.iter().zip(&before).all(|(a, b)| (a.op, a.arg0, a.arg1, a.konst.to_bits()) == (b.op, b.arg0, b.arg1, b.konst.to_bits())), "gene {g}");
                decoded += 1;
            }
        }
        assert!(decoded > 800, "{decoded}");
    }

    #[test]
    fn a_compound_expands_into_ordinary_nodes_and_computes_what_it_says() {
        let table = SymbolTable::wide(3).with_compounds();
        let id = |wanted: Symbol| table.symbols.iter().position(|s| *s == wanted).unwrap() as u32;
        // no id of the wide set moved, and the compounds are functions of two arguments
        assert!(SymbolTable::wide(3).symbols.iter().zip(&table.symbols).all(|(a, b)| a == b));
        assert!(Compound::ALL.iter().all(|&c| table.arity(id(Symbol::Compound(c))) == 2));
        let layout = Layout::for_arity(1, 1, 8, table.max_arity(), 4);
        let (ht, width) = ((layout.head + layout.tail) as usize, layout.gene_width() as usize);
        // m c^2 / sqrt(1 - v^2/c^2) with x_0 = m, x_1 = v, x_2 = c, as ONE gene:
        //   Mul( Mul(x_0, Pow2 x_2),  InvSqrtDiff( ?=1 , Pow2(ProtectedDiv(x_1, x_2)) ) )
        // level order: Mul | Mul InvSqrtDiff | x_0 Pow2 ? Pow2 | x_2 PDiv | x_1 x_2
        let mut gene = vec![id(Symbol::Input(0)); width];
        let karva = [
            Symbol::Function(Op::Mul), Symbol::Function(Op::Mul), Symbol::Compound(Compound::InvSqrtDiff), Symbol::Input(0), Symbol::Function(Op::Pow2),
            Symbol::Rnc, Symbol::Function(Op::Pow2), Symbol::Input(2), Symbol::Function(Op::ProtectedDiv), Symbol::Input(1), Symbol::Input(2),
        ];
        for (slot, symbol) in karva.iter().enumerate() {
            gene[slot] = id(*symbol);
        }
        gene[ht] = 2;                                   // the first "?" reads rnc[2]
        let nodes = decode_gene(&gene, &[9.0, 9.0, 1.0, 9.0], layout, &table).expect("closes");
        assert_eq!(nodes.len(), 11 + 2, "the compound wrote three operators for one symbol");
        assert!(nodes.iter().enumerate().all(|(i, n)| n.op == Op::Var as u32 || n.op == Op::Num as u32 || (n.arg0 as usize > i && (n.arg1 == 0 || n.arg1 as usize > i))), "child index > parent index");
        let names: Vec<String> = (0..3).map(|i| format!("x_{i}")).collect();
        let math = nodes_to_math(&nodes, 0, &names);
        assert_eq!(math, r#"(Mul (Mul (Var "x_0") (Pow2 (Var "x_2"))) (ProtectedInv (ProtectedSqrt (Sub (Num 1.0) (Pow2 (ProtectedDiv (Var "x_1") (Var "x_2")))))))"#);
        // and on data it IS the law: m = 2, v = 3, c = 5  ->  2 * 25 / sqrt(1 - 9/25) = 62.5
        let row = vec![vec![("x_0".to_string(), 2.0), ("x_1".to_string(), 3.0), ("x_2".to_string(), 5.0)]];
        assert!((evaluate_math(&math, &row).unwrap()[0] - 62.5).abs() < 1e-9);
        assert_eq!(t_depth(&nodes), 1, "one root on the deepest path, as the law has");
        // every compound, against its definition
        for c in Compound::ALL {
            let mut g = vec![id(Symbol::Input(0)); width];
            g[0] = id(Symbol::Compound(c));
            g[1] = id(Symbol::Input(0));
            g[2] = id(Symbol::Input(1));
            let m = nodes_to_math(&decode_gene(&g, &[0.0; 4], layout, &table).unwrap(), 0, &names);
            let (a, b) = (7.0f64, 3.0f64);
            let got = evaluate_math(&m, &[vec![("x_0".to_string(), a), ("x_1".to_string(), b)]]).unwrap()[0];
            let want = match c {
                Compound::SqrtSum => (a + b).sqrt(),
                Compound::SqrtDiff => (a - b).sqrt(),
                Compound::InvSqrtSum => 1.0 / (a + b).sqrt(),
                Compound::InvSqrtDiff => 1.0 / (a - b).sqrt(),
                Compound::InvSum => 1.0 / (a + b),
                Compound::InvDiff => 1.0 / (a - b),
            };
            assert!((got - want).abs() < 1e-12, "{c:?}: {got} vs {want} from {m}");
        }
    }

    #[test]
    fn a_population_with_compounds_keeps_the_rules_and_decodes() {
        let table = SymbolTable::wide(4).with_compounds();
        let layout = Layout::for_arity(400, 3, 34, table.max_arity(), 10);
        let pop = crate::evolve::init(layout, &table.codes(), &crate::evolve::InitParams { seed: 3, generation: 0, rnc_lo: -5, rnc_hi: 5, n_wrappers: 3, vhead: 0 }).unwrap();
        pop.check(&table.codes()).unwrap();
        let (width, nr) = (layout.gene_width() as usize, layout.n_rnc as usize);
        let (mut closed, mut with_compound) = (0, 0);
        for (g, gene) in pop.genome.chunks(width).enumerate() {
            if let Some(nodes) = decode_gene(gene, &pop.rnc[g * nr..(g + 1) * nr], layout, &table) {
                closed += 1;
                assert!(nodes.iter().enumerate().all(|(i, n)| n.op == Op::Var as u32 || n.op == Op::Num as u32 || (n.arg0 as usize > i && (n.arg1 == 0 || n.arg1 as usize > i))));
                with_compound += usize::from(gene.iter().take(34).any(|&id| matches!(table.symbols[id as usize], Symbol::Compound(_))));
            }
        }
        assert!(closed > 1000 && with_compound > 300, "{closed} closed, {with_compound} with a compound");
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

    /// A RELATIVISTIC block: x_0 a density, x_1 a speed and x_2 a speed of light
    /// with x_1 / x_2 in [0.05, 0.95] on every row — the beta the sqrt(1 - v^2/c^2)
    /// family lives on, strictly inside the arcsin's domain. rows() straddles
    /// |x_1/x_2| = 1 and is left as it is.
    fn rows_beta() -> Vec<Vec<(String, f64)>> {
        (0..200)
            .map(|i| {
                let t = f64::from(i);
                let c = 2.0 + t * 0.01;
                vec![("x_0".to_string(), 1.0 + t * 0.02), ("x_1".to_string(), c * (0.05 + t * 0.0045)), ("x_2".to_string(), c)]
            })
            .collect()
    }

    /// feynman I.34.14's columns: x_0 the speed of light, x_1 a speed below it,
    /// x_2 a frequency — so x_0 > |x_1| > 0 and both x_0 +- x_1 are positive on
    /// every row. rows() has x_0 - x_1 changing sign and is left as it is.
    fn rows_doppler() -> Vec<Vec<(String, f64)>> {
        (0..200)
            .map(|i| {
                let t = f64::from(i);
                let c = 4.0 + t * 0.01;
                vec![("x_0".to_string(), c), ("x_1".to_string(), c * (0.05 + t * 0.0045)), ("x_2".to_string(), 1.0 + t * 0.02)]
            })
            .collect()
    }

    /// rows() and two more positive columns, x_3 and x_4; rows() itself is not changed.
    fn rows5() -> Vec<Vec<(String, f64)>> {
        rows()
            .into_iter()
            .enumerate()
            .map(|(i, mut row)| {
                let t = i as f64;
                row.push(("x_3".to_string(), 0.5 + t * 0.03));
                row.push(("x_4".to_string(), 4.0 + (t * 0.11).cos()));
                row
            })
            .collect()
    }

    fn names5() -> Vec<String> {
        (0..5).map(|i| format!("x_{i}")).collect()
    }

    /// Mean squared difference of the two models over `rows`, relative to the
    /// first one's variance: the measure FINAL_FORM_AGREE is a bound on.
    fn drift(model: &str, other: &str, rows: &[Vec<(String, f64)>]) -> f64 {
        let (want, got) = (evaluate_math(model, rows).unwrap(), evaluate_math(other, rows).unwrap());
        let n = want.len() as f64;
        let mean = want.iter().sum::<f64>() / n;
        let var = want.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        want.iter().zip(&got).map(|(w, g)| (w - g).powi(2)).sum::<f64>() / n / var
    }

    /// feynman II.10.9 as the engine found it (seed 7013): the law inside
    /// 0.2*log(exp(5 u)/5) + 0.3218.. — log(exp(5 u)/5) is 5 u - log 5 and the
    /// constants cancel. The final form has no log and no exp left.
    #[test]
    fn a_log_of_an_exp_factor_is_its_exponent_plus_the_log_of_the_rest() {
        let u = r#"(Div (Mul (Div (Num 5.0) (Add (Num 1.0) (Var "x_2"))) (Var "x_0")) (Var "x_1"))"#;
        for (log, exp) in [("ProtectedLog", "ProtectedExp"), ("Log", "Exp")] {
            let found = format!("(Add (Mul (Num 0.200000001133779) ({log} (Div ({exp} {u}) (Num 5.0)))) (Num 0.32188757398128187))");
            let resolved = resolve_protected(&found, &rows()).unwrap();
            assert_eq!(resolved, format!("(Add (Mul (Num 0.200000001133779) (Sub {u} (Num {:?}))) (Num 0.32188757398128187))", 5.0_f64.ln()));
            let tidy = final_form(&resolved, &names(), &rows()).unwrap();
            assert!(!tidy.contains("Log") && !tidy.contains("Exp"), "{tidy}");
            assert!(drift(&found, &tidy, &rows()) <= FINAL_FORM_AGREE, "{tidy}");
        }
        // c * exp u, exp u alone, c / exp u, and c a positive column of the data.
        for (expr, wanted) in [
            (r#"(Log (Mul (Num 3.0) (Exp (Var "x_0"))))"#, format!(r#"(Add (Var "x_0") (Num {:?}))"#, 3.0_f64.ln())),
            (r#"(Log (Exp (Var "x_0")))"#, r#"(Var "x_0")"#.to_string()),
            (r#"(Log (Div (Var "x_1") (Exp (Var "x_0"))))"#, r#"(Add (Neg (Var "x_0")) (Log (Var "x_1")))"#.to_string()),
        ] {
            let resolved = resolve_protected(expr, &rows()).unwrap();
            assert_eq!(resolved, wanted);
            assert!(drift(expr, &resolved, &rows()) <= FINAL_FORM_AGREE, "{expr}");
        }
        // x_0 - 3.01 changes sign on the data: the factor is not positive, and the
        // form stands — raw (its log is not a number on half the rows) and protected
        // (log|.|: the Abs stays, and the exp stays inside it).
        let mixed = r#"(Log (Mul (Exp (Var "x_1")) (Sub (Var "x_0") (Num 3.01))))"#;
        assert_eq!(resolve_protected(mixed, &rows()).unwrap(), mixed);
        let guarded = resolve_protected(r#"(ProtectedLog (Mul (Exp (Var "x_1")) (Sub (Var "x_0") (Num 3.01))))"#, &rows()).unwrap();
        assert_eq!(guarded, r#"(Log (Abs (Mul (Exp (Var "x_1")) (Sub (Var "x_0") (Num 3.01)))))"#);
        // exp(400 x_0) overflows on the data: the protected exp stays.
        let overflow = r#"(Log (ProtectedExp (Mul (Num 400.0) (Var "x_0"))))"#;
        assert_eq!(resolve_protected(overflow, &rows()).unwrap(), overflow);
    }

    /// feynman I.44.4 as the engine found it (seed 7013): -(x_1 x_0 x_2 (log|x_3| -
    /// log|x_4|)), every column positive. One log of a quotient, and no negation.
    #[test]
    fn a_difference_of_logs_is_one_log_of_a_quotient_with_the_sign_inside() {
        for (x3, x4) in [(r#"(ProtectedLog (Var "x_3"))"#, r#"(ProtectedLog (Var "x_4"))"#), (r#"(Log (Abs (Var "x_3")))"#, r#"(Log (Abs (Var "x_4")))"#)] {
            let found = format!(r#"(Neg (Mul (Var "x_1") (Mul (Var "x_0") (Mul (Var "x_2") (Sub {x3} {x4})))))"#);
            let resolved = resolve_protected(&found, &rows5()).unwrap();
            assert_eq!(resolved, r#"(Mul (Var "x_1") (Mul (Var "x_0") (Mul (Var "x_2") (Log (Div (Var "x_4") (Var "x_3"))))))"#);
            let tidy = final_form(&resolved, &names5(), &rows5()).unwrap();
            assert_eq!(tidy.matches("Log").count(), 1, "{tidy}");
            assert!(tidy.contains(r#"(Log (Div (Var "x_4") (Var "x_3")))"#) && !tidy.contains("Neg") && !tidy.contains("Num"), "{tidy}");
            assert!(drift(&found, &tidy, &rows5()) <= FINAL_FORM_AGREE, "{tidy}");
        }
        // A sum of logs is the log of the product; a Neg factor goes into the log.
        assert_eq!(resolve_protected(r#"(Add (Log (Var "x_3")) (Log (Var "x_4")))"#, &rows5()).unwrap(), r#"(Log (Mul (Var "x_3") (Var "x_4")))"#);
        let factor = resolve_protected(r#"(Mul (Mul (Neg (Var "x_0")) (Var "x_1")) (Sub (Log (Var "x_3")) (Log (Var "x_4"))))"#, &rows5()).unwrap();
        assert_eq!(factor, r#"(Mul (Mul (Var "x_0") (Var "x_1")) (Log (Div (Var "x_4") (Var "x_3"))))"#);
        // x_0 - 3.01 is not positive on every row: the raw logs are not merged, and
        // the protected one is merged only as the log|.| it is.
        let mixed = r#"(Sub (Log (Sub (Var "x_0") (Num 3.01))) (Log (Var "x_4")))"#;
        assert_eq!(resolve_protected(mixed, &rows5()).unwrap(), mixed);
        let guarded = resolve_protected(r#"(Sub (ProtectedLog (Sub (Var "x_0") (Num 3.01))) (Log (Var "x_4")))"#, &rows5()).unwrap();
        assert_eq!(guarded, r#"(Log (Div (Abs (Sub (Var "x_0") (Num 3.01))) (Var "x_4")))"#);
    }

    /// strogatz glider2 as the engine found it (seed 7013): x - sin(y + pi/2)/x,
    /// the pi/2 exactly f64's. It is x - cos(y)/x, and is written so.
    #[test]
    fn a_phase_shift_by_a_quarter_turn_is_the_other_function() {
        use std::f64::consts::{FRAC_PI_2, PI};
        let found = r#"(Add (Sub (Var "x_0") (Div (Sin (Add (Num 1.5707963267948966) (Var "x_1"))) (Var "x_0"))) (Num -1.2788959224963037e-7))"#;
        let resolved = resolve_protected(found, &rows()).unwrap();
        let wanted = r#"(Add (Sub (Var "x_0") (Div (Cos (Var "x_1")) (Var "x_0"))) (Num -1.2788959224963037e-7))"#;
        assert_eq!(resolved, crate::lint::node::Tree::parse(wanted).unwrap().to_math());
        let tidy = final_form(&resolved, &names(), &rows()).unwrap();
        assert!(tidy.contains(r#"(Cos (Var "x_1"))"#) && !tidy.contains("Sin") && !tidy.contains("1.57"), "{tidy}");
        assert!(drift(found, &tidy, &rows()) <= FINAL_FORM_AGREE, "{tidy}");
        // Round the circle, the constant on either side or subtracted.
        let e = r#"(Var "x_1")"#;
        for (expr, wanted) in [
            (format!("(Sin (Add {e} (Num {FRAC_PI_2:?})))"), format!("(Cos {e})")),
            (format!("(Sin (Sub {e} (Num {FRAC_PI_2:?})))"), format!("(Neg (Cos {e}))")),
            (format!("(Cos (Add (Num {FRAC_PI_2:?}) {e}))"), format!("(Neg (Sin {e}))")),
            (format!("(Cos (Sub {e} (Num {FRAC_PI_2:?})))"), format!("(Sin {e})")),
            (format!("(Sin (Add {e} (Num {PI:?})))"), format!("(Neg (Sin {e}))")),
            (format!("(Cos (Add {e} (Num {PI:?})))"), format!("(Neg (Cos {e}))")),
            (format!("(Sin (Add {e} (Num {:?})))", 5.0 * FRAC_PI_2), format!("(Cos {e})")),
            (format!("(Cos (Sub {e} (Num {:?})))", 4.0 * FRAC_PI_2), format!("(Cos {e})")),
        ] {
            let resolved = resolve_protected(&expr, &rows()).unwrap();
            assert_eq!(resolved, wanted, "{expr}");
            let (a, b) = (evaluate_math(&expr, &rows()).unwrap(), evaluate_math(&resolved, &rows()).unwrap());
            assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-12), "{expr}");
        }
        // 1e-6 from pi/2 is not pi/2: the form stands.
        let near = format!("(Sin (Add {e} (Num {:?})))", FRAC_PI_2 + 1e-6);
        assert_eq!(resolve_protected(&near, &rows()).unwrap(), near);
    }

    /// feynman II.13.23's shape: tan(asin(v/c)) is the sqrt(1 - v^2/c^2) family's
    /// signature, and rewriting it cancels the /x_1 and leaves the law's root.
    #[test]
    fn a_tangent_of_an_arcsine_is_the_quotient_by_the_root_where_the_data_stays_in_the_domain() {
        let u = r#"(ProtectedDiv (Var "x_1") (Var "x_2"))"#;
        // Both spellings of the arcsin, on a beta strictly inside the domain.
        for asin in ["ProtectedAsin", "Asin"] {
            let expr = format!("(Tan ({asin} {u}))");
            let resolved = resolve_protected(&expr, &rows_beta()).unwrap();
            assert_eq!(
                resolved,
                r#"(Div (Div (Var "x_1") (Var "x_2")) (Sqrt (Sub (Num 1.0) (Pow2 (Div (Var "x_1") (Var "x_2"))))))"#,
                "{expr}"
            );
            assert!(drift(&expr, &resolved, &rows_beta()) <= FINAL_FORM_AGREE, "{expr}");
        }

        // THE CHROMOSOME, verbatim as the engine found it (seed 7014, pass2_s13).
        // The tan(asin ..) goes, the /x_1 cancels against it, and the law's root is
        // what is left — under the spurious scale the fit carries.
        let found = r#"(Mul (Num -4.371138630895453e-8) (Mul (Mul (Var "x_0") (ProtectedDiv (Num 1.633123935319537e16) (ProtectedInv (Var "x_2")))) (ProtectedDiv (Tan (ProtectedAsin (ProtectedDiv (Var "x_1") (Var "x_2")))) (Var "x_1"))))"#;
        let resolved = resolve_protected(found, &rows_beta()).unwrap();
        assert!(!resolved.contains("Tan") && !resolved.contains("Asin"), "{resolved}");
        assert!(drift(found, &resolved, &rows_beta()) <= FINAL_FORM_AGREE, "the predictions moved: {resolved}");
        let tidy = final_form(&resolved, &names(), &rows_beta()).unwrap();
        let tidy = resolve_protected(&tidy, &rows_beta()).unwrap();
        // no tan, no asin, and x_1 appears ONCE — only inside the root.
        assert!(!tidy.contains("Tan") && !tidy.contains("Asin"), "{tidy}");
        assert_eq!(tidy.matches(r#"(Var "x_1")"#).count(), 1, "the /x_1 did not cancel: {tidy}");
        assert!(tidy.contains("Sqrt"), "{tidy}");
        assert!(drift(found, &tidy, &rows_beta()) <= FINAL_FORM_AGREE, "{tidy}");

        // NEGATIVE: rows() runs x_1/x_2 from 0.61 to 1.95, so the arcsin is clamped
        // on some rows and the identity does not hold there. The form stands.
        let clamped = format!("(Tan (ProtectedAsin {u}))");
        let stood = resolve_protected(&clamped, &rows()).unwrap();
        assert!(stood.contains("Tan") && stood.contains("Asin"), "{stood}");
    }

    /// A factor on both sides of the line cancels — but only where the data says it
    /// is never 0, and a product already in its lowest terms is left exactly as it
    /// was written.
    #[test]
    fn a_factor_on_both_sides_of_the_line_cancels_where_the_data_says_it_is_never_zero() {
        let live = r#"(Div (Mul (Var "x_0") (Var "x_1")) (Var "x_1"))"#;
        assert_eq!(resolve_protected(live, &rows()).unwrap(), r#"(Var "x_0")"#);
        // Through an Inv — the divisor written the other way — with the literal put
        // first in the rebuilt product, where the rational folds can reach it.
        let through = r#"(Mul (Mul (Var "x_0") (Var "x_1")) (Inv (Mul (Num 3.0) (Var "x_1"))))"#;
        assert_eq!(resolve_protected(through, &rows()).unwrap(), r#"(Div (Var "x_0") (Num 3.0))"#);

        // NEGATIVE: x_2 - 1.5 is exactly 0 on the first row, so it does not cancel.
        let zero = r#"(Div (Mul (Var "x_0") (Sub (Var "x_2") (Num 1.5))) (Sub (Var "x_2") (Num 1.5)))"#;
        assert_eq!(resolve_protected(zero, &rows()).unwrap(), zero);
        // NEGATIVE: nothing to cancel — the product is written back as it was.
        let lowest = r#"(Mul (Var "x_0") (Var "x_1"))"#;
        assert_eq!(resolve_protected(lowest, &rows()).unwrap(), lowest);
    }

    /// sqrt(e^2) is |e| for every real e, and the Abs then sheds the sign the data
    /// settles — or stays where the data does not settle it.
    #[test]
    fn a_root_of_a_square_is_the_absolute_value() {
        let positive = r#"(Sqrt (Pow2 (Add (Var "x_0") (Var "x_1"))))"#;
        assert_eq!(resolve_protected(positive, &rows()).unwrap(), r#"(Add (Var "x_0") (Var "x_1"))"#);
        // NEGATIVE for the Abs: x_0 - 3 changes sign over rows(), so the Abs stays.
        let mixed = r#"(Sqrt (Pow2 (Sub (Var "x_0") (Num 3.0))))"#;
        assert_eq!(resolve_protected(mixed, &rows()).unwrap(), r#"(Abs (Sub (Var "x_0") (Num 3.0)))"#);
        assert!(drift(mixed, &resolve_protected(mixed, &rows()).unwrap(), &rows()) <= FINAL_FORM_AGREE);
    }

    /// feynman I.34.14 as the engine found it (seed 7014, pass2_s13), rebuilt from
    /// the fit's `fuller_model` `sqrt(((x_1 + x_0)/(x_0 - x_1))*(x_2**2))`: the root
    /// of the quotient is the law written in the ratio v/c, the form SRBench takes.
    #[test]
    fn a_root_of_a_sum_over_a_difference_is_the_family_written_in_the_ratio() {
        let found = r#"(Sqrt (Mul (Div (Add (Var "x_1") (Var "x_0")) (Sub (Var "x_0") (Var "x_1"))) (Pow2 (Var "x_2"))))"#;
        let resolved = resolve_protected(found, &rows_doppler()).unwrap();
        // x_2 comes out of the root as itself (positive on the data), and what is
        // left is (1 + v/c)/sqrt(1 - (v/c)^2).
        let beta = r#"(Div (Var "x_1") (Var "x_0"))"#;
        assert_eq!(
            resolved,
            format!(r#"(Mul (Var "x_2") (Div (Add (Num 1.0) {beta}) (Sqrt (Sub (Num 1.0) (Pow2 {beta})))))"#),
            "{resolved}"
        );
        assert!(drift(found, &resolved, &rows_doppler()) <= FINAL_FORM_AGREE, "the predictions moved");
        // The reported string is the law's own shape, with no root left in a
        // denominator's argument and no (c+v)/(c-v) quotient.
        let text = crate::lint::node::Tree::parse(&resolved).unwrap().to_infix();
        assert_eq!(text, "(x_2*((1.0 + (x_1/x_0))/sqrt((1.0 - ((x_1/x_0)**2)))))", "{text}");

        // The PROTECTED spelling a chromosome carries: the ProtectedSqrt arm builds
        // the raw root itself, so the rewrite only reaches it on the second pass —
        // the one the engine runs after the final form. It must land the same way.
        let protected = r#"(ProtectedSqrt (Mul (ProtectedDiv (Add (Var "x_1") (Var "x_0")) (Sub (Var "x_0") (Var "x_1"))) (Pow2 (Var "x_2"))))"#;
        let once = resolve_protected(protected, &rows_doppler()).unwrap();
        let tidy = final_form(&once, &names(), &rows_doppler()).unwrap();
        let twice = resolve_protected(&tidy, &rows_doppler()).unwrap();
        assert_eq!(crate::lint::node::Tree::parse(&twice).unwrap().to_infix(), text, "{twice}");
        assert!(drift(protected, &twice, &rows_doppler()) <= FINAL_FORM_AGREE, "{twice}");

        // NEGATIVE: over rows() the difference x_0 - x_1 changes sign, so a + b and
        // a - b are not both positive and the two forms are not the same number.
        let stood = resolve_protected(found, &rows()).unwrap();
        assert!(stood.contains(r#"(Sub (Var "x_0") (Var "x_1"))"#), "{stood}");
        assert!(!stood.contains("(Num 1.0)"), "the rewrite fired where the data does not support it: {stood}");
    }

    /// feynman I.44.4 EXACTLY as the chromosome wrote it (RAW_MATH of the seed-7013
    /// one-gene fit): the difference of logs is an Add with a Neg inside, and the
    /// minus sign of the law lives in the fitted scale. It must come out as ONE log
    /// of a quotient with no negation in front — the form SRBench's scorer accepts.
    #[test]
    fn a_difference_of_logs_is_merged_however_the_chromosome_wrote_it() {
        let found = r#"(Add (Mul (Num -1.0000000480226603) (Mul (Var "x_1") (Mul (Var "x_0") (Mul (Var "x_2") (Add (Neg (ProtectedLog (Var "x_4"))) (ProtectedLog (Var "x_3"))))))) (Num 1.1673999131267543e-7))"#;
        let resolved = resolve_protected(found, &rows5()).unwrap();
        assert_eq!(resolved.matches("(Log ").count(), 1, "{resolved}");
        assert!(resolved.contains(r#"(Log (Div (Var "x_4") (Var "x_3")))"#), "{resolved}");
        assert!(!resolved.contains("Neg") && !resolved.contains("(Num -1."), "a negation is still in front: {resolved}");
        assert!(drift(found, &resolved, &rows5()) <= FINAL_FORM_AGREE, "the predictions moved");
        // every signed pairing
        for (expr, wanted) in [
            (r#"(Add (Log (Var "x_3")) (Neg (Log (Var "x_4"))))"#, r#"(Log (Div (Var "x_3") (Var "x_4")))"#),
            (r#"(Sub (Neg (Log (Var "x_3"))) (Neg (Log (Var "x_4"))))"#, r#"(Log (Div (Var "x_4") (Var "x_3")))"#),
            (r#"(Sub (Log (Var "x_3")) (Neg (Log (Var "x_4"))))"#, r#"(Log (Mul (Var "x_3") (Var "x_4")))"#),
        ] {
            assert_eq!(resolve_protected(expr, &rows5()).unwrap(), wanted, "{expr}");
        }
    }

    /// A term that matters stays.
    #[test]
    fn the_final_form_keeps_a_term_the_data_can_see() {
        let model = r#"(Add (Mul (Var "x_0") (Var "x_1")) (Sin (Var "x_2")))"#;
        let tidy = final_form(model, &names(), &rows()).unwrap();
        assert!(tidy.contains("Sin") && tidy.contains("x_0") && tidy.contains("x_1"), "{tidy}");
    }

    // -----------------------------------------------------------------------
    // The islands: pairs, the pump, the cross step.
    // -----------------------------------------------------------------------

    /// A gene written by hand: `symbols` in the head, the tail filled with input 0
    /// and the Dc domain with zeros.
    fn hand_gene(symbols: &[Symbol], layout: Layout, table: &SymbolTable) -> Vec<u32> {
        let id = |wanted: Symbol| table.symbols.iter().position(|s| *s == wanted).expect("a symbol of the table") as u32;
        let mut gene = vec![id(Symbol::Input(0)); (layout.head + layout.tail) as usize];
        for (slot, symbol) in gene.iter_mut().zip(symbols) {
            *slot = id(*symbol);
        }
        gene.extend(std::iter::repeat_n(0, layout.tail as usize));
        gene
    }

    /// Row 0 of a drawn generation overwritten with hand-written genes, and its
    /// fitness cleared so `evaluate` scores it.
    fn plant(engine: &Engine, genes: &[Vec<Symbol>]) -> Generation {
        let mut gen = drawn_generation(engine, 3);
        let width = engine.layout.gene_width() as usize;
        for (g, symbols) in genes.iter().enumerate() {
            gen.pop.genome[g * width..(g + 1) * width].copy_from_slice(&hand_gene(symbols, engine.layout, &engine.table));
        }
        gen.fitness[0] = f32::NAN;
        gen
    }

    fn fresh_timing() -> Timing {
        Timing { vary: 0.0, read: 0.0, decode: 0.0, evaluate: 0.0, score: 0.0, hff: 0.0, pump: 0.0, cross: 0.0, snap: 0.0, genealogy: 0.0 }
    }

    /// THE DYNAMIC GENE-SUBSET CHOICE. Gene 1 alone IS the law y = x0 * x1; gene 0
    /// is junk (cos x1) and gene 2 a tower (sin sin sin exp x0, t_depth 4). With
    /// the switch on `evaluate` picks {1}: 1 - R² ~ 0, `genes == 0b010`, the tower
    /// in gene 2 costs nothing, `math_of` prints one gene and no linker, and
    /// `confirm` (f64) agrees on every count. With the switch off the same row
    /// must use all three genes and cannot be exact.
    #[test]
    fn a_chromosome_whose_one_gene_is_the_law_uses_that_gene_alone() {
        let data = {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for i in 0..60u32 {
                let row = [1.0 + f64::from(i % 7) * 0.5, 2.0 + f64::from(i % 5) * 0.25, 1.5 + f64::from(i % 11) * 0.2];
                x.extend(row.iter().map(|v| *v as f32));
                y.push(row[0] * row[1]);
            }
            Data { names: names(), x, y, splits: Splits { n_train: 40, n_val: 20, n_extrap: 0 } }
        };
        let (mul, sin, cos, exp) = (Symbol::Function(Op::Mul), Symbol::Function(Op::Sin), Symbol::Function(Op::Cos), Symbol::Function(Op::ProtectedExp));
        let (x0, x1) = (Symbol::Input(0), Symbol::Input(1));
        let genes = vec![vec![cos, x1], vec![mul, x0, x1], vec![sin, sin, sin, exp, x0]];
        for (on, want_genes) in [(true, 0b010u32), (false, 0b111u32)] {
            let config = Config { gene_subsets: on, tower: true, ..toy_config(30, 10) };
            let mut engine = Engine::new(config, Data { names: data.names.clone(), x: data.x.clone(), y: data.y.clone(), splits: data.splits }).expect("engine");
            let mut gen = plant(&engine, &genes);
            engine.evaluate(&mut gen, &mut fresh_timing()).expect("evaluate");
            let ranked = engine.scored[0].expect("row 0 scored");
            let confirmed = engine.confirm(&gen, 0).expect("confirm").expect("row 0 confirmed");
            assert_eq!((ranked.genes, confirmed.genes), (want_genes, want_genes), "switch {on}: {ranked:?} / {confirmed:?}");
            let math = engine.math_of(&gen, 0, &confirmed);
            if on {
                assert!(ranked.one_minus_r2[1] < 1e-6 && confirmed.one_minus_r2[1] < 1e-12, "{ranked:?} / {confirmed:?}");
                assert_eq!((ranked.t_depth, confirmed.t_depth), (0, 0), "the tower in gene 2 is not used");
                assert_eq!(ranked.wrapper, 0);
                assert!(!math.contains("Cos") && !math.contains("Sin") && !math.contains("Exp"), "{math}");
                assert!(math.contains(r#"(Mul (Var "x_0") (Var "x_1"))"#) && !math.contains("(Div "), "{math}");
                assert_eq!(math.matches("(Add ").count(), 1, "only the offset's Add: {math}");
            } else {
                assert!(ranked.one_minus_r2[1] > 1e-4 && confirmed.one_minus_r2[1] > 1e-4, "{ranked:?} / {confirmed:?}");
                assert_eq!((ranked.t_depth, confirmed.t_depth), (4, 4));
                assert!(math.contains("Cos") && math.contains("Sin"), "{math}");
            }
            // The fitness of row 0 is what the tournaments will rank on.
            assert!((f64::from(gen.fitness[0]) - ranked.selection).abs() < 1e-6);
        }
    }

    /// A law that needs the SUM of two genes, y = x0 * x1 + x2, with a junk third:
    /// the pair {0, 1} wins. Under linear scaling avg and add of the same genes are
    /// the same model, so either linker may be reported — never mul, never the
    /// junk gene; `math_of` prints the two genes and, under avg, their divisor 2.
    #[test]
    fn a_law_that_is_a_sum_of_two_genes_picks_the_pair() {
        let (mul, sin) = (Symbol::Function(Op::Mul), Symbol::Function(Op::Sin));
        let (x0, x1, x2) = (Symbol::Input(0), Symbol::Input(1), Symbol::Input(2));
        let genes = vec![vec![mul, x0, x1], vec![x2], vec![sin, x0]];
        let config = Config { gene_subsets: true, ..toy_config(30, 10) };
        let mut engine = Engine::new(config, toy_data()).expect("engine");
        let mut gen = plant(&engine, &genes);
        engine.evaluate(&mut gen, &mut fresh_timing()).expect("evaluate");
        let ranked = engine.scored[0].expect("row 0 scored");
        let confirmed = engine.confirm(&gen, 0).expect("confirm").expect("row 0 confirmed");
        assert_eq!((ranked.genes, confirmed.genes), (0b011, 0b011), "{ranked:?} / {confirmed:?}");
        assert!(confirmed.one_minus_r2[1] < 1e-12 && ranked.one_minus_r2[1] < 1e-6, "{ranked:?} / {confirmed:?}");
        assert!(matches!(LINKER_NAMES[confirmed.linker], "avgval" | "addval"), "{confirmed:?}");
        let math = engine.math_of(&gen, 0, &confirmed);
        assert!(math.contains(r#"(Add (Mul (Var "x_0") (Var "x_1")) (Var "x_2"))"#) && !math.contains("Sin"), "{math}");
        assert_eq!(math.contains("(Div "), LINKER_NAMES[confirmed.linker] == "avgval", "{math}");
        if math.contains("(Div ") {
            assert!(math.contains("(Num 2.0)"), "avg divides by the USED genes: {math}");
        }
    }

    /// Four genes with the subset choice on is an error, not a fallback.
    #[test]
    fn four_genes_with_the_subset_choice_on_is_an_error() {
        let config = Config { gene_subsets: true, n_genes: 4, ..toy_config(30, 10) };
        let mut engine = Engine::new(config, toy_data()).expect("engine");
        let mut gen = drawn_generation(&engine, 3);
        gen.fitness[0] = f32::NAN;
        let err = engine.evaluate(&mut gen, &mut fresh_timing()).expect_err("an error");
        assert!(err.contains("at most 3 genes"), "{err}");
    }

    /// y = x_0 * x_1 + x_2 on a small grid: train, then validation.
    fn toy_data() -> Data {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for i in 0..60u32 {
            let row = [1.0 + f64::from(i % 7) * 0.5, 2.0 + f64::from(i % 5) * 0.25, 1.5 + f64::from(i % 11) * 0.2];
            x.extend(row.iter().map(|v| *v as f32));
            y.push(row[0] * row[1] + row[2]);
        }
        Data { names: names(), x, y, splits: Splits { n_train: 40, n_val: 20, n_extrap: 0 } }
    }

    fn toy_config(pop_intake: u32, pop_champion: u32) -> Config {
        Config { pop_intake, pop_champion, head: 8, ..Config::srbench(7013) }
    }

    /// A generation with every row evaluated: a seeded population and a fitness
    /// drawn per row (distinct streams of the engine's own generator).
    fn drawn_generation(engine: &Engine, seed: u32) -> Generation {
        let p = InitParams { seed, generation: 0, rnc_lo: -100, rnc_hi: 100, n_wrappers: WRAPPERS.len() as u32, vhead: 0 };
        let pop = crate::evolve::init(engine.layout, &engine.table.codes(), &p).expect("init");
        let fitness = (0..engine.layout.pop).map(|r| crate::evolve::below(crate::evolve::draw(seed, 0, r, 0, 99), 1_000_000) as f32 * 1e-6).collect();
        Generation { pop, fitness }
    }

    /// FNV-1a over everything a host step can change.
    fn digest(gen: &Generation) -> u64 {
        let words = gen.pop.genome.iter().copied()
            .chain(gen.pop.rnc.iter().map(|v| v.to_bits()))
            .chain(gen.pop.wrapper_id.iter().copied())
            .chain(gen.fitness.iter().map(|v| v.to_bits()));
        words.fold(0xcbf2_9ce4_8422_2325u64, |h, w| (h ^ u64::from(w)).wrapping_mul(0x0000_0100_0000_01b3))
    }

    /// One pair and no cross step is the engine as it was: the same two islands,
    /// and the pump's effect on a fixed generation, beat after beat, is the digest
    /// recorded from the single-pair engine at commit 5ed2b79 (before pairs existed).
    #[test]
    fn one_pair_without_a_cross_step_is_the_engine_as_it_was() {
        let engine = Engine::new(Config::srbench(1), toy_data()).expect("engine");
        assert_eq!(engine.islands, vec![
            Island { lo: 0, hi: 600, elites: 2, tournsize: 42 },
            Island { lo: 600, hi: 800, elites: 2, tournsize: 14 },
        ]);
        let mut engine = Engine::new(toy_config(60, 20), toy_data()).expect("engine");
        let mut gen = drawn_generation(&engine, 11);
        let mut seen = Vec::new();
        for generation in [4u32, 8, 12] {
            engine.pump(&mut gen, generation).expect("the pump");
            seen.push(digest(&gen));
            // what `evaluate` would do to the fresh rows, without a device in the way
            for (r, f) in gen.fitness.iter_mut().enumerate() {
                if f.is_nan() {
                    *f = crate::evolve::below(crate::evolve::draw(11, generation, r as u32, 0, 99), 1_000_000) as f32 * 1e-6;
                }
            }
        }
        assert_eq!(seen, vec![290130392017734542, 18034047553287649106, 11794650284444892442], "the pump on one pair");
    }

    /// A row whole: its genes and its constants.
    fn row_of(gen: &Generation, r: u32) -> (Vec<u32>, Vec<u32>) {
        let l = gen.pop.layout;
        let (row_w, rnc_w, r) = ((l.n_genes * l.gene_width()) as usize, (l.n_genes * l.n_rnc) as usize, r as usize);
        (gen.pop.genome[r * row_w..(r + 1) * row_w].to_vec(), gen.pop.rnc[r * rnc_w..(r + 1) * rnc_w].iter().map(|v| v.to_bits()).collect())
    }

    /// Three pairs of a 20-row intake island and a 10-row champion island, every
    /// row evaluated, with a fitness that is known: inside an island it FALLS as the
    /// row rises, so an island's best row is its last.
    fn three_pairs() -> (Engine, Generation) {
        let engine = Engine::new(Config { n_pairs: 3, ..toy_config(20, 10) }, toy_data()).expect("engine");
        let mut gen = drawn_generation(&engine, 23);
        for isl in &engine.islands {
            for r in isl.lo..isl.hi {
                gen.fitness[r as usize] = 0.5 + (isl.hi - r) as f32 * 0.01 + isl.lo as f32 * 0.0001;
            }
        }
        (engine, gen)
    }

    #[test]
    fn pairs_tile_the_population_and_every_island_keeps_its_own_tournament() {
        let engine = Engine::new(Config { n_pairs: 3, ..Config::srbench(1) }, toy_data()).expect("engine");
        assert_eq!(engine.layout.pop, 2400, "the sizes are one pair's: 3 x (600 + 200)");
        assert_eq!(engine.islands.len(), 6);
        let mut next = 0;
        for (p, (intake, champion)) in engine.pairs().into_iter().enumerate() {
            assert_eq!((intake.lo, intake.hi, champion.lo, champion.hi), (next, next + 600, next + 600, next + 800), "pair {p}");
            assert_eq!((intake.tournsize, champion.tournsize), (42, 14), "pair {p}: 7% of each island");
            assert_eq!((intake.elites, champion.elites), (2, 2), "pair {p}");
            next = champion.hi;
        }
        assert_eq!(next, engine.layout.pop, "no gap, no overlap, nothing left over");
        crate::evolve::vary::validate(engine.layout, &engine.islands).expect("the variation kernels accept the layout");
        assert!(Engine::new(Config { n_pairs: 0, ..Config::srbench(1) }, toy_data()).is_err(), "no pairs is no population");
    }

    #[test]
    fn the_pump_works_inside_each_pair_and_nowhere_else() {
        let (mut engine, mut gen) = three_pairs();
        let before = gen.clone();
        engine.pump(&mut gen, 4).expect("the pump");
        let fresh = engine.fresh(4, 4).expect("fresh");
        for (p, (intake, champion)) in engine.pairs().into_iter().enumerate() {
            // The intake's best two are its last two rows; the champion island's worst
            // two are its first two, the worst of all first.
            let promoted = [(intake.hi - 1, champion.lo), (intake.hi - 2, champion.lo + 1)];
            for (from, to) in promoted {
                assert_eq!(row_of(&gen, to), row_of(&before, from), "pair {p}: row {from} is promoted to row {to}");
                assert_eq!(gen.fitness[to as usize], before.fitness[from as usize], "pair {p}: a promoted row keeps its fitness");
            }
            for r in champion.lo + 2..champion.hi {
                assert_eq!(row_of(&gen, r), row_of(&before, r), "pair {p}: the rest of the champion island is untouched");
                assert_eq!(gen.fitness[r as usize], before.fitness[r as usize]);
            }
            // Nobody else's intake reaches this champion island.
            for (q, (other, _)) in engine.pairs().into_iter().enumerate() {
                if q != p {
                    for to in [champion.lo, champion.lo + 1] {
                        assert!((other.lo..other.hi).all(|r| row_of(&before, r) != row_of(&gen, to)), "pair {p}: row {to} came from pair {q}");
                    }
                }
            }
            // The best fifth (4 of 20) stays, fittest first; the rest is fresh and unevaluated.
            for slot in 0..4 {
                assert_eq!(row_of(&gen, intake.lo + slot), row_of(&before, intake.hi - 1 - slot), "pair {p}: keeper {slot}");
                assert!(!gen.fitness[(intake.lo + slot) as usize].is_nan());
            }
            for r in intake.lo + 4..intake.hi {
                let width = (engine.layout.n_genes * engine.layout.gene_width()) as usize;
                assert_eq!(row_of(&gen, r).0, fresh.genome[r as usize * width..(r as usize + 1) * width], "pair {p}: row {r} is a new individual");
                assert!(gen.fitness[r as usize].is_nan() && engine.scored[r as usize].is_none(), "pair {p}: row {r} is unevaluated");
            }
        }
        // Every pair draws its own new individuals.
        let pairs = engine.pairs();
        assert_ne!(row_of(&gen, pairs[0].0.lo + 4), row_of(&gen, pairs[1].0.lo + 4));
    }

    #[test]
    fn the_cross_step_sends_each_intake_the_other_pairs_champions() {
        let (mut engine, mut gen) = three_pairs();
        // Intake 0's two best rows are the same individual: only one of it is kept.
        let pairs = engine.pairs();
        let (twin_from, twin_to) = ((pairs[0].0.hi - 1) as usize, (pairs[0].0.hi - 2) as usize);
        let (row_w, rnc_w) = ((engine.layout.n_genes * engine.layout.gene_width()) as usize, (engine.layout.n_genes * engine.layout.n_rnc) as usize);
        gen.pop.genome.copy_within(twin_from * row_w..(twin_from + 1) * row_w, twin_to * row_w);
        gen.pop.rnc.copy_within(twin_from * rnc_w..(twin_from + 1) * rnc_w, twin_to * rnc_w);
        let before = gen.clone();
        engine.cross(&mut gen, 5).expect("the cross step");
        let fresh = engine.fresh(5 | CROSS_KEY, 5).expect("fresh");
        for (p, &(intake, champion)) in pairs.iter().enumerate() {
            // Its best fifth, distinct, fittest first, with the fitness it had.
            let kept: Vec<u32> = if p == 0 { vec![intake.hi - 1, intake.hi - 3, intake.hi - 4, intake.hi - 5] } else { (1..=4).map(|k| intake.hi - k).collect() };
            for (slot, &from) in kept.iter().enumerate() {
                let to = intake.lo + slot as u32;
                assert_eq!(row_of(&gen, to), row_of(&before, from), "pair {p}: keeper {slot}");
                assert_eq!(gen.fitness[to as usize], before.fitness[from as usize], "pair {p}: a keeper keeps its fitness");
            }
            // Then the OTHER pairs' three best champions: pair order, then rank.
            let arrivals: Vec<u32> = pairs.iter().enumerate().filter(|(q, _)| *q != p).flat_map(|(_, &(_, c))| (1..=3).map(move |k| c.hi - k)).collect();
            assert_eq!(arrivals.len(), 6);
            for (slot, &from) in arrivals.iter().enumerate() {
                let to = intake.lo + 4 + slot as u32;
                assert_eq!(row_of(&gen, to), row_of(&before, from), "pair {p}: arrival {slot} is champion row {from}");
                assert!(gen.fitness[to as usize].is_nan() && engine.scored[to as usize].is_none(), "pair {p}: an arrival is evaluated again");
            }
            // Never its own pair's champions.
            for r in intake.lo..intake.hi {
                assert!((champion.lo..champion.hi).all(|c| row_of(&before, c) != row_of(&gen, r)), "pair {p}: row {r} is one of its own champions");
            }
            // The rest: new individuals, unevaluated.
            for r in intake.lo + 10..intake.hi {
                assert_eq!(row_of(&gen, r).0, fresh.genome[r as usize * row_w..(r as usize + 1) * row_w], "pair {p}: row {r} is a new individual");
                assert!(gen.fitness[r as usize].is_nan() && engine.scored[r as usize].is_none());
            }
            // Champion islands are not touched.
            for r in champion.lo..champion.hi {
                assert_eq!(row_of(&gen, r), row_of(&before, r), "pair {p}: champion row {r}");
                assert_eq!(gen.fitness[r as usize], before.fitness[r as usize]);
            }
        }
        // Deterministic: the same step on the same generation, again.
        let (mut again_engine, _) = three_pairs();
        let mut again = before.clone();
        again_engine.cross(&mut again, 5).expect("the cross step");
        assert_eq!(digest(&again), digest(&gen));
        assert_eq!(again.pop, gen.pop);
    }

    /// `k_migrants` beyond a champion island: it sends all it has. An intake with
    /// no room for every arrival: the first that fit (pair order, then rank).
    #[test]
    fn the_cross_step_truncates_what_does_not_fit() {
        let mut engine = Engine::new(Config { n_pairs: 3, k_migrants: 10, ..toy_config(6, 4) }, toy_data()).expect("engine");
        let mut gen = drawn_generation(&engine, 29);
        for isl in &engine.islands {
            for r in isl.lo..isl.hi {
                gen.fitness[r as usize] = 0.5 + (isl.hi - r) as f32 * 0.01;
            }
        }
        let before = gen.clone();
        engine.cross(&mut gen, 5).expect("the cross step");
        let pairs = engine.pairs();
        // Intake 0 (6 rows) keeps 1 (a fifth of 6, rounded) and has room for 5 of the
        // 8 arrivals: pair 1's four champions, best first, then pair 2's best.
        let intake = pairs[0].0;
        assert_eq!(row_of(&gen, intake.lo), row_of(&before, intake.hi - 1));
        let arrivals = [pairs[1].1.hi - 1, pairs[1].1.hi - 2, pairs[1].1.hi - 3, pairs[1].1.hi - 4, pairs[2].1.hi - 1];
        for (slot, &from) in arrivals.iter().enumerate() {
            let to = intake.lo + 1 + slot as u32;
            assert_eq!(row_of(&gen, to), row_of(&before, from), "arrival {slot}");
            assert!(gen.fitness[to as usize].is_nan());
        }
        assert_eq!(intake.lo + 1 + arrivals.len() as u32, intake.hi, "the island is full: no fresh row");
    }

    /// With one pair there are no other champions: the cross step is the
    /// keep-the-fifth refill. And on a beat it shares with the pump, its new
    /// individuals are not the ones the pump drew for the same rows.
    #[test]
    fn the_cross_step_draws_its_own_new_individuals() {
        let mut engine = Engine::new(toy_config(60, 20), toy_data()).expect("engine");
        let mut pumped = drawn_generation(&engine, 31);
        let mut crossed = pumped.clone();
        let before = pumped.clone();
        engine.pump(&mut pumped, 20).expect("the pump");
        engine.cross(&mut crossed, 20).expect("the cross step");
        let (intake, champion) = engine.pairs()[0];
        for r in champion.lo..champion.hi {
            assert_eq!(row_of(&crossed, r), row_of(&before, r), "the cross step promotes nobody");
        }
        for r in intake.lo..intake.lo + 12 {
            assert_eq!(row_of(&crossed, r), row_of(&pumped, r), "the same best fifth");
        }
        for r in intake.lo + 12..intake.hi {
            assert_ne!(row_of(&crossed, r), row_of(&pumped, r), "row {r}: the pump's new individual, drawn again");
            assert!(crossed.fitness[r as usize].is_nan());
        }
    }

    /// Three pairs, the pump every 4 and the cross step every 5, end to end on the
    /// device. With the stop bar out of reach the fit runs 22 generations — five
    /// pumps, four cross steps, and generation 20 where they share a beat — holds
    /// y = x_0 * x_1 + x_2 at the end, and its hall of fame's file and the returned
    /// model tell the same story; the same fit again is the same fit. With the stop
    /// bar in place it stops on the law.
    #[test]
    fn three_pairs_with_the_cross_step_find_a_simple_law() {
        let dir = std::env::temp_dir().join(format!("fuller_pairs_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a directory for the hall of fame");
        let path = dir.join("hof.tsv");
        let config = Config {
            n_pairs: 3,
            cross_every: 5,
            progress_every: 22,
            hof_path: Some(path.to_string_lossy().into_owned()),
            max_generations: 22,
            max_seconds: 3600.0,
            stop_one_minus_r2: -1.0,
            ..toy_config(200, 100)
        };
        let fit = |config: &Config| {
            let mut engine = Engine::new(config.clone(), toy_data()).expect("engine");
            assert_eq!(engine.layout.pop, 900);
            engine.fit().expect("fit")
        };
        let out = fit(&config);
        assert_eq!((out.stopped_by, out.generations), ("n_gen", 22));
        assert_eq!(out.individuals, 900 * 23);
        assert!(out.timing.pump > 0.0 && out.timing.cross > 0.0, "both steps ran: {:?}", out.timing);
        assert!(out.best.one_minus_r2[0] <= 1e-10 && out.best.one_minus_r2[1] <= 1e-10, "{:?}: {}", out.best.one_minus_r2, out.math);
        // The model computes the law on rows the fit never saw.
        let held: Vec<Vec<(String, f64)>> = (0..20).map(|i| names().into_iter().zip([0.3 + f64::from(i), 7.0 - 0.2 * f64::from(i), 4.5]).collect()).collect();
        let predicted = evaluate_math(&out.math, &held).expect("the model evaluates");
        for (row, got) in held.iter().zip(&predicted) {
            let want = row[0].1 * row[1].1 + row[2].1;
            assert!((got - want).abs() <= 1e-6 * want.abs().max(1.0), "{got} is not {want}: {}", out.math);
        }
        // The hall of fame's last line is the generation the fit ended on, and its
        // best is the returned model's (the file holds the f32 ranking scores).
        let text = std::fs::read_to_string(&path).expect("the hall of fame file");
        let last: Vec<&str> = text.lines().last().expect("a line").split('\t').collect();
        assert_eq!(last[0].parse::<u32>().expect("reported_at_gen"), out.generations);
        assert!(last[1].parse::<u32>().expect("found_at_gen") <= out.generations);
        let (hff, r2_val): (f64, f64) = (last[2].parse().expect("hff"), last[6].parse().expect("r2_val_bl2"));
        assert!(1.0 - r2_val <= 1e-5, "the hall of fame's best is not the law: {last:?}");
        assert!((hff - out.best.fitness).abs() <= 1e-3, "hall of fame {hff} against the confirmed {}", out.best.fitness);
        // Deterministic, pump and cross step included.
        let again = fit(&config);
        assert_eq!((again.math, again.unique_genes), (out.math, out.unique_genes));
        // And with the stop bar in place the fit ends on the law.
        let stopped = fit(&Config { stop_one_minus_r2: 1e-10, max_generations: 400, ..config });
        assert_eq!(stopped.stopped_by, "early_stop", "after {} generations: {}", stopped.generations, stopped.math);
        std::fs::remove_file(&path).expect("remove the hall of fame file");
        std::fs::remove_dir(&dir).expect("remove its directory");
    }

    // -----------------------------------------------------------------------
    // THE GENEALOGY: identity, age and lineage — ALPS's measurement half.
    // -----------------------------------------------------------------------

    /// A scratch path of this test's own.
    fn scratch(name: &str) -> (std::path::PathBuf, String) {
        let dir = std::env::temp_dir().join(format!("fuller-genealogy-{name}"));
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let path = dir.join("genealogy.tsv");
        (dir, path.to_string_lossy().into_owned())
    }

    /// A config that exercises every arrival mechanism in a few generations: the
    /// pump on its beat, the cross step on its own, over three pairs.
    fn tracked_config(genealogy_path: Option<String>) -> Config {
        Config {
            n_pairs: 3,
            pump_every: 3,
            cross_every: 5,
            max_generations: 12,
            max_seconds: 120.0,
            stop_one_minus_r2: 0.0,
            stop_log10_p: f64::NEG_INFINITY,
            genealogy_path,
            ..toy_config(30, 10)
        }
    }

    /// THE OFF-BY-DEFAULT PROOF: with `genealogy_path = None` the engine is what it
    /// was, bit for bit. The same fit with the log ON must leave the population
    /// identical — same genome, constants, wrappers and fitness bits at the end,
    /// and the same model — so nothing the tracking does can reach the search.
    #[test]
    fn the_genealogy_changes_no_bit_of_the_population() {
        let (dir, path) = scratch("bit-identical");
        let run = |config: Config| {
            let mut engine = Engine::new(config, toy_data()).expect("engine");
            let out = engine.fit().expect("fit");
            let pop = engine.population().expect("the population");
            let words = pop.genome.iter().copied()
                .chain(pop.rnc.iter().map(|v| v.to_bits()))
                .chain(pop.wrapper_id.iter().copied());
            let digest = words.fold(0xcbf2_9ce4_8422_2325u64, |h, w| (h ^ u64::from(w)).wrapping_mul(0x0000_0100_0000_01b3));
            (digest, out)
        };
        let (off, out_off) = run(tracked_config(None));
        let (on, out_on) = run(tracked_config(Some(path.clone())));
        assert_eq!(off, on, "the genealogy moved a bit of the population");
        assert_eq!(out_off.math, out_on.math, "the genealogy changed the model");
        assert_eq!(out_off.generations, out_on.generations);
        assert_eq!(out_off.unique_genes, out_on.unique_genes);
        assert_eq!(out_off.best.fitness.to_bits(), out_on.best.fitness.to_bits());
        assert!(out_off.lineage.is_none(), "off means nothing is tracked");
        assert!(out_on.lineage.is_some(), "on means the winner has a lineage");
        assert_eq!(out_off.timing.genealogy, 0.0, "off costs nothing");
        std::fs::remove_file(&path).expect("remove the log");
        std::fs::remove_dir(&dir).expect("remove its directory");
    }

    /// The log's rows, parsed back: (kind, generation, row, id, parent, other, age,
    /// founder, founder_gen, founder_origin, origin).
    fn log_rows(path: &str) -> Vec<Vec<String>> {
        let text = std::fs::read_to_string(path).expect("the genealogy log");
        let mut lines = text.lines();
        assert_eq!(
            lines.next().expect("a header"),
            super::super::genealogy::GENEALOGY_HEADER.trim_end(),
            "the log's header"
        );
        lines.filter(|l| !l.trim().is_empty()).map(|l| l.split('\t').map(str::to_string).collect()).collect()
    }

    /// TWO RUNS OF THE SAME SEED give identical ids, ages and origins: the ids come
    /// from a counter advanced in a fixed row order inside deterministic loops, so
    /// they are as reproducible as the population itself.
    #[test]
    fn the_same_seed_is_the_same_genealogy() {
        let (dir, path) = scratch("deterministic");
        let second = path.replace("genealogy.tsv", "again.tsv");
        let run = |p: &str| {
            let mut engine = Engine::new(tracked_config(Some(p.to_string())), toy_data()).expect("engine");
            let out = engine.fit().expect("fit");
            (log_rows(p), out.lineage.expect("a lineage"), out.genealogy_minted)
        };
        let (a, mark_a, minted_a) = run(&path);
        let (b, mark_b, minted_b) = run(&second);
        assert_eq!(a, b, "the same seed wrote a different genealogy");
        assert_eq!(mark_a, mark_b, "the winner's identity is not reproducible");
        assert_eq!(minted_a, minted_b);
        assert!(!a.is_empty(), "the log is empty");
        std::fs::remove_file(&path).expect("remove");
        std::fs::remove_file(&second).expect("remove");
        std::fs::remove_dir(&dir).expect("remove its directory");
    }

    /// THE ORIGINS the log must name, on a real fit: the initial draw as a batch at
    /// generation 0, the pump's promotions and its refills on the pump's beat, and
    /// the cross step's arrivals on its own. Every line's founder origin is an
    /// ARRIVAL — a line never begins at `vary`.
    #[test]
    fn the_log_names_every_mechanism_that_put_a_row_into_the_population() {
        let (dir, path) = scratch("origins");
        let mut engine = Engine::new(tracked_config(Some(path.clone())), toy_data()).expect("engine");
        let out = engine.fit().expect("fit");
        let rows = log_rows(&path);
        let origin_of = |r: &Vec<String>| r[10].clone();
        let kinds: Vec<String> = rows.iter().map(|r| r[0].clone()).collect();
        assert!(kinds.iter().any(|k| k == "best"), "the best of a generation is logged");
        assert!(kinds.iter().any(|k| k == "batch"), "fresh individuals are logged as batches");
        assert!(kinds.iter().any(|k| k == "arrival"), "arrivals are logged");
        assert!(kinds.iter().any(|k| k == "lineage_of_winner"), "the winner's chain is appended");
        // generation 0 is the initial draw, one batch line over the whole population
        let init: Vec<&Vec<String>> = rows.iter().filter(|r| r[0] == "batch" && r[1] == "0").collect();
        assert_eq!(init.len(), 1, "the initial draw is ONE line");
        assert_eq!(origin_of(init[0]), "init");
        assert_eq!(init[0][3], "0");
        assert_eq!(init[0][4], (engine.layout.pop - 1).to_string(), "the batch covers every row");
        let origins: Vec<String> = rows.iter().filter(|r| r[0] != "lineage_of_winner").map(origin_of).collect();
        for want in ["pump_promote", "pump_refill", "pump_keep", "cross_arrival", "cross_fresh", "cross_keep"] {
            assert!(origins.iter().any(|o| o == want), "the log never says {want}: {:?}", {
                let mut u = origins.clone();
                u.sort();
                u.dedup();
                u
            });
        }
        // a pump beat's promotions and refills carry that generation
        let promotions: Vec<&Vec<String>> = rows.iter().filter(|r| origin_of(r) == "pump_promote").collect();
        assert!(!promotions.is_empty());
        for p in &promotions {
            assert_eq!(p[1].parse::<u32>().expect("a generation") % 3, 0, "a promotion off the pump's beat: {p:?}");
            assert_eq!(p[0], "arrival");
        }
        for r in rows.iter().filter(|r| origin_of(r) == "pump_refill" || origin_of(r) == "cross_fresh") {
            assert_eq!(r[6], "0", "a fresh individual is age 0: {r:?}");
            assert_eq!(r[3], r[7], "a fresh individual founds its own line: {r:?}");
        }
        // NO line begins at ordinary variation. The `population` summary is not a
        // record — its columns are its own key-value pairs — so it is left out.
        for r in rows.iter().filter(|r| r[0] != "population") {
            let founder_origin = &r[9];
            assert!(
                ["init", "pump_refill", "cross_fresh", "beam_append"].contains(&founder_origin.as_str()),
                "a line begins at {founder_origin}, which is not an arrival: {r:?}"
            );
        }
        // the winner's chain ends at a founder, and it is the one the mark carries
        let chain: Vec<&Vec<String>> = rows.iter().filter(|r| r[0] == "lineage_of_winner").collect();
        assert!(!chain.is_empty());
        let mark = out.lineage.expect("a lineage");
        assert_eq!(chain[0][3], mark.id.to_string(), "the chain starts at the winner");
        let last = chain.last().expect("a founder");
        assert_eq!(last[4], "-1", "the founder has no parent");
        assert_eq!(last[3], mark.founder.to_string(), "the walk reaches the carried founder");
        assert_eq!(last[1], mark.founder_generation.to_string());
        assert_eq!(last[10], mark.founder_origin.to_string());
        // each step of the chain links to the next
        for pair in chain.windows(2) {
            assert_eq!(pair[0][4], pair[1][3], "the chain is not linked: {:?} -> {:?}", pair[0], pair[1]);
        }
        std::fs::remove_file(&path).expect("remove");
        std::fs::remove_dir(&dir).expect("remove its directory");
    }

    /// THE AGE RULES on a real fit: every id of the last generation is distinct, a
    /// pump beat leaves age-0 material, no row is older than the fit, and every
    /// row's carried founder is the one its walk reaches.
    #[test]
    fn the_age_rules_hold_over_a_real_fit() {
        let (dir, path) = scratch("ages");
        let mut engine = Engine::new(tracked_config(Some(path.clone())), toy_data()).expect("engine");
        engine.fit().expect("fit");
        let l = engine.lineage.as_ref().expect("the genealogy");
        // every row of the last generation holds a different id
        let mut ids: Vec<u64> = l.tracker.marks().iter().map(|m| m.id).collect();
        let n = ids.len();
        assert_eq!(n, engine.layout.pop as usize);
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "two rows of the last generation share an id");
        // no age exceeds the fit's generations, and the refills put age-0 material in
        let (min, _, max, _) = l.tracker.ages(0..n).expect("ages");
        assert_eq!(min, 0, "a pump beat near the end leaves age-0 rows");
        assert!(max <= 12, "a row is older than the fit: {max}");
        // every row's carried founder is reachable by the walk, and is a founder
        for r in 0..n {
            let mark = l.tracker.row(r);
            let chain = l.tracker.walk(mark.id);
            let (founder_id, founder) = *chain.last().expect("a founder");
            assert_eq!(founder_id, mark.founder, "row {r}: the carried founder is not the walk's");
            assert_eq!(founder.parent, u64::MAX);
            assert!(founder.origin.is_arrival(), "row {r} descends from {}", founder.origin);
            assert_eq!(founder.generation, mark.founder_generation);
        }
        std::fs::remove_file(&path).expect("remove");
        std::fs::remove_dir(&dir).expect("remove its directory");
    }

    /// SNAP's write-back keeps the individual: the row's id and age do not change,
    /// and the event is logged as `snap_writeback`.
    #[test]
    fn a_snap_write_back_is_the_same_individual_in_the_log() {
        let (dir, path) = scratch("snap");
        let config = Config { snap_every: 2, snap_top_k: 4, ..tracked_config(Some(path.clone())) };
        let mut engine = Engine::new(config, toy_data()).expect("engine");
        engine.fit().expect("fit");
        let rows = log_rows(&path);
        let snapped: Vec<&Vec<String>> = rows.iter().filter(|r| r[10] == "snap_writeback").collect();
        // Snap may find nothing to graft on this toy law; when it does, the rule holds.
        for r in &snapped {
            assert_eq!(r[0], "arrival");
            assert_eq!(r[3], r[4], "a repaired row is its own parent: it is the same individual");
            assert_eq!(r[5], "-1", "a repair has no second ancestor");
        }
        assert!(engine.snap_counts().beats > 0, "the snap beat ran");
        std::fs::remove_file(&path).expect("remove");
        std::fs::remove_dir(&dir).expect("remove its directory");
    }

    /// THE FINAL POPULATION's ages and diversity: reported on the fit, written into
    /// the log as its own line, and consistent with the tracker's own rows.
    #[test]
    fn a_fit_reports_the_final_populations_ages_and_how_many_lines_its_best_rows_come_from() {
        let (dir, path) = scratch("population-ages");
        let mut engine = Engine::new(tracked_config(Some(path.clone())), toy_data()).expect("engine");
        let out = engine.fit().expect("fit");
        let ages = out.population_ages.expect("the final population's ages");
        let pop = engine.layout.pop as usize;
        // the whole population's ages bracket the best ten's
        assert!(ages.all.0 <= ages.best_10.0 && ages.best_10.2 <= ages.all.2, "{ages:?}");
        assert!(ages.all.0 <= ages.all.1 && ages.all.1 <= ages.all.2, "{ages:?}");
        assert!(ages.all.2 <= out.generations, "a row older than the fit: {ages:?}");
        assert!(ages.founders_best_50 >= 1 && ages.founders_best_50 <= 50);
        assert!(ages.founders_all >= ages.founders_best_50, "{ages:?}");
        assert!(ages.founders_all <= pop);
        // the tracker agrees
        let l = engine.lineage.as_ref().expect("the genealogy");
        assert_eq!(l.tracker.ages(0..pop).expect("ages"), ages.all);
        assert_eq!(l.tracker.distinct_founders(0..pop), ages.founders_all);
        // and the log carries it, so the study file stands on its own
        let text = std::fs::read_to_string(&path).expect("the log");
        let line = text.lines().find(|l| l.starts_with("population\t")).expect("a population line");
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f[2], "all");
        assert_eq!(f[3].parse::<u32>().expect("min"), ages.all.0);
        assert_eq!(f[5].parse::<u32>().expect("max"), ages.all.2);
        assert_eq!(f[7], "best10");
        assert_eq!(f[12], "founders_best50");
        assert_eq!(f[13].parse::<usize>().expect("founders"), ages.founders_best_50);
        assert_eq!(f[15].parse::<usize>().expect("founders_all"), ages.founders_all);
        std::fs::remove_file(&path).expect("remove");
        std::fs::remove_dir(&dir).expect("remove its directory");
    }
}
