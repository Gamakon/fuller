//! THE IN-LOOP OPTIMISERS OF THE TRUE WGSL KINGDOM (`docs/PLAN_true_gep_wgsl.md` ST10): the rewrites the SR engine runs
//! on its winners (cleanse, the e-graph, snap), made for kernels. Each takes an expressed tree of the table's rows and
//! returns another tree of the table's rows, so the per-type write-back (`MultiTypedGEP.md` section 5) puts it into a
//! gene that expresses a legal kernel by construction; nothing here judges a kernel, fitness does.
//!
//! * [`Optimisers::cleanse`]: dead work. A write to a register, a uniform register, a private array or a workgroup
//!   array (the `reg.set_*`, `ureg.set*`, `parr.set_*`, `wg.set_*` rows: arg 0 names the variable) is dead when no later
//!   read made by a live statement uses its value: per write, by strong liveness backwards over the statements (ST12;
//!   to ST11 per variable, so a variable live anywhere kept every write of it). A dead write whose other arguments have
//!   no effect becomes its statement type's empty terminal (`{}`); then every statement left empty (a block of empties, an `if`
//!   over empties with an effect-free condition, a `_ = x;` of an effect-free `x`, a counted loop over an empty body
//!   whose counter nothing else reads) becomes the empty terminal too. A write spelled with a barrier (`wg.put_*`) is
//!   never removed: its barriers may order other memory. An output store (every `store.<b>_O*`/`_L*`/`_C*` row, the
//!   texture's `textureStore` rows among them, and the epilogue's terminals) is live by its row and never removed.
//! * THE EPILOGUE RULE: [`Optimisers::new`] fails unless every row giving an epilogue type (`stmt.O*`, `block.O*`)
//!   stores the primary output on every path, so no draw, crossover or rewrite at an epilogue cell has a row that skips
//!   the store in its domain; [`Optimisers::missing_store`] checks one tree.
//! * [`Optimisers::simplify`]: table-row rewrites of scalar float arithmetic, bottom up to a fixed point: constant
//!   folding (`lit + lit`, `-lit`), `x * 1`, `x / 1`, `x - 0`, `-(-x)` (exact), `x + 0`, `x * 0` (equal for finite
//!   `x`), strength reduction `x / c` to `x * (1 / c)` (exact when `c` is a power of two). A folded value must be an
//!   entry of the literal's `C_T` (append-only for the epoch); a value it does not hold is counted, never invented.
//! * [`Optimisers::egraph`]: [`simplify`](Optimisers::simplify), then fuller's e-graph (`true_kingdom::egraph`, the
//!   ST9 rules plus [`Rules::work`]) and its cheapest program under [`Optimisers::cost`], then simplify again; with
//!   `exact` only the rules whose claim is [`Claim::Exact`].
//! * [`Optimisers::hoist`]: loop-invariant code motion where the table can spell it. A subtree inside a counted loop's
//!   body whose every row is context-free (`true_share::context_free_rows`: no register, counter or written memory) is
//!   computed once into a fresh register before the loop (`reg.set_<x>` at the loop's own rung, in
//!   `stmt.block(block.seq2(set, loop))`) and read inside it by `reg.get_<x>`.
//! * [`Optimisers::snap`]: a float literal within `rel_tol` of a library constant (`fuller::snap::default_constants`)
//!   but not equal to it takes the `C_T` entry that IS that constant; a constant `C_T` does not hold is a named defect.
//! * [`Optimisers::fewer_trips`]: one counted loop with a literal bound gets the next smaller literal of its `C_T`. This
//!   changes the picture: a VARIATION for fitness to judge, labelled [`Optimiser::Trips`], never an equivalence.

use std::collections::{BTreeMap, BTreeSet};

use crate::geneframe::Ty;
use crate::typed_gep::Tree;
use crate::wgsl::true_kingdom::egraph::{Claim, Equivalents, Rules};
use crate::wgsl::true_kingdom::{Rng, RowKind, TrueKingdom};
use crate::wgsl::true_share::context_free_rows;

/// Text in a row's spelling that makes the row act beyond computing its value.
const EFFECTS: &[&str] = &[
    "Barrier",
    "workgroupUniformLoad",
    "atomicAdd",
    "atomicSub",
    "atomicMax",
    "atomicMin",
    "atomicAnd",
    "atomicOr",
    "atomicXor",
    "atomicExchange",
    "atomicCompareExchange",
    "atomicStore",
    "textureStore",
    "discard",
    "return",
    "break",
    "continue",
];

/// The optimisers, as the evolution loop and its log name them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Optimiser {
    Cleanse,
    EgraphExact,
    EgraphReal,
    Hoist,
    Snap,
    Trips,
}

impl Optimiser {
    pub const ALL: [Optimiser; 6] = [Optimiser::Cleanse, Optimiser::EgraphExact, Optimiser::EgraphReal, Optimiser::Hoist, Optimiser::Snap, Optimiser::Trips];

    pub fn name(self) -> &'static str {
        match self {
            Optimiser::Cleanse => "cleanse",
            Optimiser::EgraphExact => "egraph_exact",
            Optimiser::EgraphReal => "egraph_real",
            Optimiser::Hoist => "hoist",
            Optimiser::Snap => "snap",
            Optimiser::Trips => "trips",
        }
    }

    /// The lineage operator code (the marriage step's are 1 to 5).
    pub fn code(self) -> u32 {
        match self {
            Optimiser::Cleanse => 6,
            Optimiser::EgraphExact => 7,
            Optimiser::EgraphReal => 8,
            Optimiser::Hoist => 9,
            Optimiser::Snap => 10,
            Optimiser::Trips => 11,
        }
    }

    pub fn from_code(code: u32) -> Option<Optimiser> {
        Optimiser::ALL.into_iter().find(|o| o.code() == code)
    }

    /// What a rewrite of this optimiser claims: cleanse, the exact e-graph and the hoist keep every value; the real
    /// e-graph is a real-number equality; snap and fewer trips are variations.
    pub fn claim(self) -> OptClaim {
        match self {
            Optimiser::Cleanse | Optimiser::EgraphExact | Optimiser::Hoist => OptClaim::Exact,
            Optimiser::EgraphReal => OptClaim::Real,
            Optimiser::Snap | Optimiser::Trips => OptClaim::Variation,
        }
    }
}

/// What an optimiser's child claims about its picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptClaim {
    /// The same values bit for bit, by IEEE arithmetic.
    Exact,
    /// Equal as real-number programs.
    Real,
    /// A different program for fitness to judge.
    Variation,
}

/// What one rewrite did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Counts {
    /// Cleanse: dead writes removed; statements left empty and replaced by the empty terminal.
    pub dead_writes: usize,
    pub empties: usize,
    /// Simplify: rewrites applied; folds whose value the literal's `C_T` does not hold.
    pub simplified: usize,
    pub folds_missing: usize,
    /// Hoist: subtrees moved out of loops; loops the table cannot spell a hoist at (no set row at the rung, no free
    /// register).
    pub hoisted: usize,
    pub hoist_unspellable: usize,
    /// Snap: literals snapped; near misses whose exact constant `C_T` does not hold (named in `defects`).
    pub snapped: usize,
    /// Snap: library constants appended to a `C_T` that did not hold them.
    pub appended: usize,
    /// Named defects: a rewrite the table cannot spell.
    pub defects: Vec<String>,
}

/// A float literal row and the f32 value of each entry of its `C_T`.
#[derive(Debug, Clone)]
struct FloatConst {
    row: u32,
    values: Vec<Option<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
}

/// A scalar float binary row: its kind, output and argument types.
#[derive(Debug, Clone)]
struct BinRow {
    kind: BinKind,
    out: Ty,
    args: [Ty; 2],
}

/// How a statement row moves liveness backwards (cleanse's per-write analysis).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Flow {
    /// A write row: arg 0 names the variable; `kills` when it replaces the whole variable (a register), not when it
    /// writes one element (a private or workgroup array).
    Write { kills: bool },
    /// Text with no control flow and each statement argument spelled exactly once: the arguments run once, in the
    /// order of their placeholders (the argument indices in spelling order, value arguments included).
    Straight(Vec<usize>),
    /// A loop, an `if`, or any other row: its statement arguments run in any order, any number of times, possibly
    /// none (a counted loop's own `break` leaves to after the loop, which this covers).
    Any,
    /// Text that leaves the statement early (`break`, `continue`, `return`, `discard` outside a counted loop's own
    /// exit): every variable is live before it.
    Jump,
    /// An output store (a `store.<b>_*` row, the texture store among them, or an epilogue terminal): live by its row
    /// whatever is read after it, every value it takes is read, and it is never removed.
    Store,
}

/// An epilogue statement type (`stmt.O<d>_f<k>`, `block.O<d>_f<k>`): every path through a statement of one of these
/// ends in the primary output's store (`MultiTypedGEP_s9` batch 1).
fn is_epilogue_name(d: &str) -> bool {
    d.strip_prefix("stmt.O").or_else(|| d.strip_prefix("block.O")).is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()))
}

/// The argument placeholders of a spelling in spelling order: `(index, has a field)` for `{N}` and `{N:F}`.
fn placeholders(s: &str) -> Vec<(usize, bool)> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(open) = s[i..].find('{') {
        let at = i + open;
        let Some(close) = s[at..].find('}') else { break };
        let inner = &s[at + 1..at + close];
        let (a, f) = inner.split_once(':').map_or((inner, None), |(a, f)| (a, Some(f)));
        if !a.is_empty() && a.bytes().all(|c| c.is_ascii_digit()) && f.is_none_or(|f| !f.is_empty() && f.bytes().all(|c| c.is_ascii_digit())) {
            if let Ok(n) = a.parse::<usize>() {
                out.push((n, f.is_some()));
            }
            i = at + close + 1;
        } else {
            i = at + 1;
        }
    }
    out
}

/// The identifier words of a text.
fn words(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).filter(|w| !w.is_empty())
}

/// The state of one cleanse's liveness pass: the table, every variable the tree names, the path of the statement being
/// analysed, the dead writes found (by path) and the dead writes the table cannot remove.
struct Liveness<'a> {
    k: &'a TrueKingdom,
    universe: BTreeSet<u32>,
    path: Vec<usize>,
    dead: BTreeSet<Vec<usize>>,
    defects: BTreeSet<String>,
}

/// A register family: the register type, its read row, its terminals, its write row at each statement type.
#[derive(Debug, Clone)]
struct RegFamily {
    get: u32,
    terminals: Vec<u32>,
    set: BTreeMap<Ty, u32>,
}

/// The optimisers of one table, every row family read from its rows.
#[derive(Debug, Clone)]
pub struct Optimisers {
    /// Statement-like types (`stmt.*`, `block.*`, `kernel.*`): never a value.
    statement: BTreeSet<Ty>,
    /// The empty terminal (`{}`) of each statement type that has one.
    empty: BTreeMap<Ty, u32>,
    /// Per row: a variable's name (a register, private array or workgroup array terminal).
    name_terminal: Vec<bool>,
    /// Per row: a workgroup array's name (memory other invocations of the workgroup read).
    wg_name: Vec<bool>,
    /// Per row: how a statement row moves liveness (cleanse); a write row (arg 0 the variable written, no barrier in
    /// its spelling) is [`Flow::Write`].
    flow: Vec<Flow>,
    /// Per row: its spelling acts beyond its value, or it outputs a statement.
    effect: Vec<bool>,
    /// Per row: an output store, by its row: the `store.<b>_O*`, `_L*`, `_C*` rows of every output (the texture's
    /// `textureStore` rows among them) and the epilogue's terminals `own.block_O*` / `own.stmt_O*`, which spell the
    /// primary output's store. Cleanse treats each as live and never removes it.
    store: Vec<bool>,
    /// The epilogue statement types ([`is_epilogue_name`]).
    epilogue: BTreeSet<Ty>,
    /// Per row: a pure sequence (`block.one`, `block.seq*`, `stmt.block`, `stmt.lift`): statement arguments and no text
    /// but braces.
    container: Vec<bool>,
    /// Per row: an `if` (value conditions, statement bodies).
    is_if: Vec<bool>,
    /// Per row: `_ = x;`.
    discard: Vec<bool>,
    /// Per row: a counted loop with a statement body (not a reduction).
    loop_body: Vec<bool>,
    context_free: Vec<bool>,
    consts: BTreeMap<Ty, FloatConst>,
    bins: BTreeMap<u32, BinRow>,
    /// `(kind, arg types)` -> row, to find `mul` of the shape of a `div`.
    bin_by_shape: BTreeMap<(u8, Ty, Ty), u32>,
    /// Scalar `neg` rows: row -> type.
    neg: BTreeMap<u32, Ty>,
    /// Value type -> its register family.
    regs: BTreeMap<Ty, RegFamily>,
    /// Statement type -> (`stmt.block_X`, `block.seq2_X`).
    seq: BTreeMap<Ty, (u32, u32)>,
    /// Per row: functions count 1 and the costly ones more ([`Optimisers::cost`]).
    row_cost: Vec<u64>,
    rules_all: Rules,
    rules_exact: Rules,
}

fn bug(what: impl std::fmt::Display) -> String {
    format!("our software has a bug: {what}")
}

/// A spelling with every `{N}` / `{N:F}` / `{v}` placeholder removed.
fn without_placeholders(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'{' {
            if let Some(close) = s[i..].find('}') {
                let inner = &s[i + 1..i + close];
                let (a, f) = inner.split_once(':').map_or((inner, None), |(a, f)| (a, Some(f)));
                if inner == "v" || (!a.is_empty() && a.bytes().all(|c| c.is_ascii_digit()) && f.is_none_or(|f| !f.is_empty() && f.bytes().all(|c| c.is_ascii_digit()))) {
                    i += close + 1;
                    continue;
                }
            }
        }
        let ch = s[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

impl Optimisers {
    /// The optimisers of table `k`; our bug, named, when a row giving an epilogue type can skip the primary output's
    /// store ([`Optimisers::check_epilogue_rows`]).
    pub fn new(k: &TrueKingdom) -> Result<Optimisers, String> {
        let o = Optimisers::of_rows(k);
        o.check_epilogue_rows(k)?;
        Ok(o)
    }

    /// Every row family read from the table, before the epilogue rule is checked.
    fn of_rows(k: &TrueKingdom) -> Optimisers {
        let n = k.rows.len();
        let dual = |t: Ty| k.type_name(t);
        let statement: BTreeSet<Ty> = k.types.iter().map(|d| d.ty()).filter(|t| {
            let d = dual(*t);
            d.starts_with("stmt.") || d.starts_with("block.") || d.starts_with("kernel.")
        }).collect();
        let is_name_ty = |t: Ty| {
            let d = dual(t);
            d.starts_with("reg<") || d.starts_with("ureg<") || d.starts_with("parr<") || d.starts_with("wgmem<")
        };
        let mut empty = BTreeMap::new();
        for (i, r) in k.rows.iter().enumerate() {
            if r.kind == RowKind::Terminal && statement.contains(&r.out) && r.spelling.trim() == "{}" {
                empty.entry(r.out).or_insert(i as u32);
            }
        }
        let name_terminal: Vec<bool> = k.rows.iter().map(|r| r.kind == RowKind::Terminal && is_name_ty(r.out)).collect();
        let effect: Vec<bool> = k.rows.iter().map(|r| statement.contains(&r.out) || EFFECTS.iter().any(|e| without_placeholders(&r.spelling).contains(e))).collect();
        let epilogue: BTreeSet<Ty> = statement.iter().copied().filter(|t| is_epilogue_name(&dual(*t))).collect();
        let store: Vec<bool> = k
            .rows
            .iter()
            .map(|r| {
                let stores = r.name.starts_with("store.") && r.kind == RowKind::Function && statement.contains(&r.out) && r.args.iter().all(|a| !statement.contains(a));
                let own_epilogue = (r.name.starts_with("own.block_O") || r.name.starts_with("own.stmt_O")) && r.kind == RowKind::Terminal && epilogue.contains(&r.out);
                stores || own_epilogue
            })
            .collect();
        let write: Vec<bool> = k
            .rows
            .iter()
            .map(|r| {
                let named = ["reg.set_", "ureg.set", "parr.set_", "wg.set_"].iter().any(|p| r.name.starts_with(p));
                named && r.kind == RowKind::Function && statement.contains(&r.out) && r.args.first().is_some_and(|a| is_name_ty(*a)) && !r.spelling.contains("Barrier")
            })
            .collect();
        let container: Vec<bool> = k
            .rows
            .iter()
            .map(|r| {
                let text = without_placeholders(&r.spelling);
                r.kind == RowKind::Function && statement.contains(&r.out) && !r.args.is_empty() && r.args.iter().all(|a| statement.contains(a)) && text.chars().all(|c| c.is_whitespace() || c == '{' || c == '}')
            })
            .collect();
        let wg_name: Vec<bool> = k.rows.iter().map(|r| r.kind == RowKind::Terminal && dual(r.out).starts_with("wgmem<")).collect();
        let flow: Vec<Flow> = (0..n)
            .map(|i| {
                let r = &k.rows[i];
                if store[i] {
                    return Flow::Store;
                }
                if write[i] {
                    let d = dual(r.args[0]);
                    return Flow::Write { kills: d.starts_with("reg<") || d.starts_with("ureg<") };
                }
                if k.loops[i].is_some() {
                    return Flow::Any;
                }
                let text = without_placeholders(&r.spelling);
                if words(&text).any(|w| matches!(w, "break" | "continue" | "return" | "discard")) {
                    return Flow::Jump;
                }
                if words(&text).any(|w| matches!(w, "if" | "else" | "loop" | "for" | "while" | "switch" | "continuing")) {
                    return Flow::Any;
                }
                let order = placeholders(&r.spelling);
                let once = r.args.iter().enumerate().filter(|(_, a)| statement.contains(a)).all(|(j, _)| order.iter().filter(|(p, _)| *p == j).count() == 1 && order.iter().any(|(p, f)| *p == j && !f));
                if once && order.iter().all(|(p, _)| *p < r.args.len()) {
                    Flow::Straight(order.into_iter().map(|(p, _)| p).collect())
                } else {
                    Flow::Any
                }
            })
            .collect();
        let is_if: Vec<bool> = k.rows.iter().map(|r| r.name.starts_with("stmt.if") && r.kind == RowKind::Function && statement.contains(&r.out)).collect();
        let discard: Vec<bool> = k.rows.iter().map(|r| r.name.starts_with("stmt.discard") && r.kind == RowKind::Function && r.args.iter().all(|a| !statement.contains(a))).collect();
        let loop_body: Vec<bool> = (0..n)
            .map(|i| {
                let r = &k.rows[i];
                k.loops[i].as_ref().is_some_and(|l| !l.own_work) && statement.contains(&r.out) && r.args.iter().any(|a| statement.contains(a))
            })
            .collect();
        // float literal rows: every C_T entry parses as a float (`f32.*` in the old tables, `f32` in the rows kingdom)
        let mut consts = BTreeMap::new();
        for (i, r) in k.rows.iter().enumerate() {
            if r.kind != RowKind::Constant || !(dual(r.out) == "f32" || dual(r.out).starts_with("f32.")) {
                continue;
            }
            let Some(texts) = k.constants.get(&r.out) else { continue };
            let values: Vec<Option<f32>> = texts.iter().map(|t| t.trim().parse::<f64>().ok().map(|v| v as f32)).collect();
            if values.iter().all(Option::is_some) {
                consts.entry(r.out).or_insert(FloatConst { row: i as u32, values });
            }
        }
        let mut bins = BTreeMap::new();
        let mut bin_by_shape = BTreeMap::new();
        let mut neg = BTreeMap::new();
        for (i, r) in k.rows.iter().enumerate() {
            let base = r.name.split("_bare").next().unwrap_or("");
            let kind = match base {
                "arith.add" => Some(BinKind::Add),
                "arith.sub" => Some(BinKind::Sub),
                "arith.mul" => Some(BinKind::Mul),
                "arith.div" => Some(BinKind::Div),
                _ => None,
            };
            if let Some(kind) = kind {
                if r.args.len() == 2 && consts.contains_key(&r.out) && r.args.iter().all(|a| consts.contains_key(a)) {
                    bins.insert(i as u32, BinRow { kind, out: r.out, args: [r.args[0], r.args[1]] });
                    bin_by_shape.insert((kind as u8, r.args[0], r.args[1]), i as u32);
                }
            }
            if r.name == "arith.neg" && r.args.len() == 1 && r.args[0] == r.out && consts.contains_key(&r.out) {
                neg.insert(i as u32, r.out);
            }
        }
        // register families
        let mut regs: BTreeMap<Ty, RegFamily> = BTreeMap::new();
        let mut reg_of_value: BTreeMap<Ty, Ty> = BTreeMap::new();
        for (i, r) in k.rows.iter().enumerate() {
            if r.name.starts_with("reg.get_") && r.kind == RowKind::Function && r.args.len() == 1 && dual(r.args[0]).starts_with("reg<") && !statement.contains(&r.out) {
                let terminals: Vec<u32> = k.rows.iter().enumerate().filter(|(_, t)| t.kind == RowKind::Terminal && t.out == r.args[0]).map(|(j, _)| j as u32).collect();
                if !terminals.is_empty() && !regs.contains_key(&r.out) {
                    reg_of_value.insert(r.out, r.args[0]);
                    regs.insert(r.out, RegFamily { get: i as u32, terminals, set: BTreeMap::new() });
                }
            }
        }
        for (i, r) in k.rows.iter().enumerate() {
            if r.name.starts_with("reg.set_") && r.kind == RowKind::Function && r.args.len() == 2 && statement.contains(&r.out) {
                if let Some(f) = regs.get_mut(&r.args[1]) {
                    if reg_of_value.get(&r.args[1]) == Some(&r.args[0]) {
                        f.set.entry(r.out).or_insert(i as u32);
                    }
                }
            }
        }
        // stmt.block_X and block.seq2_X for every statement type stmt.X
        let mut seq = BTreeMap::new();
        let id = |name: &str| k.rows.iter().position(|r| r.name == name).map(|i| i as u32);
        for t in &statement {
            let d = dual(*t);
            let Some(x) = d.strip_prefix("stmt.") else { continue };
            if let (Some(sb), Some(s2)) = (id(&format!("stmt.block_{x}")), id(&format!("block.seq2_{x}"))) {
                let (rb, r2) = (&k.rows[sb as usize], &k.rows[s2 as usize]);
                if rb.out == *t && rb.args.len() == 1 && rb.args[0] == r2.out && r2.args == [*t, *t] {
                    seq.insert(*t, (sb, s2));
                }
            }
        }
        let row_cost: Vec<u64> = k
            .rows
            .iter()
            .map(|r| {
                if r.kind != RowKind::Function {
                    return 1;
                }
                let base = r.name.split("_bare").next().unwrap_or("");
                let base = base.trim_end_matches("_v2").trim_end_matches("_v3").trim_end_matches("_v4");
                match base {
                    "arith.pow" | "arith.exp" | "arith.exp2" | "arith.log" | "arith.log2" | "arith.sin" | "arith.cos" | "arith.tan" | "arith.tanh" | "arith.asin" | "arith.acos" | "arith.atan" | "arith.atan2" | "arith.sqrt" | "arith.inverse_sqrt" => 8,
                    "arith.div" => 4,
                    _ => 1,
                }
            })
            .collect();
        let rules_all = Rules::work(k);
        let rules_exact = rules_all.exact_only();
        Optimisers { statement, empty, name_terminal, wg_name, flow, effect, store, epilogue, container, is_if, discard, loop_body, context_free: context_free_rows(k), consts, bins, bin_by_shape, neg, regs, seq, row_cost, rules_all, rules_exact }
    }

    /// THE EPILOGUE RULE OVER THE TABLE: every row that gives an epilogue type stores the primary output on every
    /// path, so the rows of an epilogue type (the domain every draw, crossover and rewrite at an epilogue cell takes
    /// from) hold no tree that skips the store. A row passes when it is an output store ([`Flow::Store`]); a sequence
    /// that runs one of its epilogue arguments exactly once; an `if` with an `else` whose every statement argument is
    /// an epilogue; or a counted loop whose every statement argument is an epilogue and whose trip bound is a literal
    /// type every producer of which is at least 1. Anything else (a discard, a one-armed `if`, a jump, a row with no
    /// epilogue argument) is our bug, named.
    fn check_epilogue_rows(&self, k: &TrueKingdom) -> Result<(), String> {
        let mut bad = Vec::new();
        for (i, r) in k.rows.iter().enumerate() {
            if !self.epilogue.contains(&r.out) || self.store[i] {
                continue;
            }
            let stmts: Vec<Ty> = r.args.iter().copied().filter(|a| self.statement.contains(a)).collect();
            let all_epilogue = !stmts.is_empty() && stmts.iter().all(|a| self.epilogue.contains(a));
            let ok = match &self.flow[i] {
                Flow::Straight(_) => r.kind == RowKind::Function && r.args.iter().any(|a| self.epilogue.contains(a)),
                Flow::Any if k.loops[i].is_some() => all_epilogue && self.bound_at_least_one(k, i),
                Flow::Any => all_epilogue && self.is_if[i] && words(&without_placeholders(&r.spelling)).any(|w| w == "else"),
                Flow::Store | Flow::Write { .. } | Flow::Jump => false,
            };
            if !ok {
                bad.push(format!("{} ({}: {})", r.name, k.type_name(r.out), r.spelling));
            }
        }
        if bad.is_empty() {
            return Ok(());
        }
        Err(bug(format!("{} row(s) give an epilogue type and do not store the primary output on every path: {}", bad.len(), bad.join("; "))))
    }

    /// A loop row's trip bound is a literal argument whose every producer is at least 1: its `C_T` entries, its
    /// terminals' spellings, and its function rows clamped below at `1u`.
    fn bound_at_least_one(&self, k: &TrueKingdom, row: usize) -> bool {
        let Some(arg) = k.loops[row].as_ref().and_then(|l| l.literal_arg) else { return false };
        let ty = k.rows[row].args[arg];
        let at_least_one = |s: &str| s.trim().trim_end_matches('u').parse::<u64>().is_ok_and(|v| v >= 1);
        k.rows.iter().filter(|p| p.out == ty).all(|p| match p.kind {
            RowKind::Constant => k.constants.get(&ty).is_some_and(|c| !c.is_empty() && c.iter().all(|v| at_least_one(v))),
            RowKind::Terminal => at_least_one(&p.spelling),
            _ => p.spelling.trim_end().ends_with(", 1u, 64u)"),
        })
    }

    /// `Some(reason)` when the expressed tree's epilogue can end without the primary output's store: an epilogue
    /// statement stores when it is a store row, a sequence one of whose epilogue arguments stores, an `if`/`else` or
    /// counted loop whose every epilogue argument stores (a loop's literal bound at least 1). A kingdom with no
    /// epilogue type (no `block.O`) has nothing to check.
    pub fn missing_store(&self, k: &TrueKingdom, tree: &Tree) -> Option<String> {
        if self.epilogue.is_empty() {
            return None;
        }
        let root = &k.rows[tree.symbol as usize];
        let epilogues: Vec<&Tree> = tree.kids.iter().filter(|c| self.epilogue.contains(&k.rows[c.symbol as usize].out)).collect();
        if epilogues.is_empty() {
            return Some(format!("the root {} takes no epilogue", root.name));
        }
        epilogues.into_iter().find_map(|e| (!self.stores(k, e)).then(|| format!("the epilogue {} can end without the primary output's store", k.rows[e.symbol as usize].name)))
    }

    fn stores(&self, k: &TrueKingdom, t: &Tree) -> bool {
        let s = t.symbol as usize;
        if self.store[s] {
            return true;
        }
        let o: Vec<&Tree> = t.kids.iter().filter(|c| self.epilogue.contains(&k.rows[c.symbol as usize].out)).collect();
        if o.is_empty() {
            return false;
        }
        match &self.flow[s] {
            Flow::Straight(_) => o.iter().any(|c| self.stores(k, c)),
            Flow::Any => {
                let trips = match k.loops[s].as_ref() {
                    None => self.is_if[s],
                    Some(l) => l.literal_arg.and_then(|i| t.kids.get(i)).is_some_and(|b| self.literal_at_least_one(k, b)),
                };
                trips && o.iter().all(|c| self.stores(k, c))
            }
            Flow::Store | Flow::Write { .. } | Flow::Jump => false,
        }
    }

    /// A loop bound node that is at least 1: a `C_T` literal, a terminal or a function row clamped below at `1u`.
    fn literal_at_least_one(&self, k: &TrueKingdom, b: &Tree) -> bool {
        let r = &k.rows[b.symbol as usize];
        let at_least_one = |s: &str| s.trim().trim_end_matches('u').parse::<u64>().is_ok_and(|v| v >= 1);
        match r.kind {
            RowKind::Constant => b.const_index.and_then(|i| k.constants.get(&r.out)?.get(i as usize).cloned()).is_some_and(|v| at_least_one(&v)),
            RowKind::Terminal => at_least_one(&r.spelling),
            _ => r.spelling.trim_end().ends_with(", 1u, 64u)"),
        }
    }

    /// The cost a node's row adds to a program: 1, 4 for a division, 8 for `pow`, `exp`, `log`, `sqrt` and the
    /// trigonometric rows. The e-graph's cheapest program minimises the sum over the tree.
    pub fn cost(&self, symbol: u32) -> u64 {
        self.row_cost.get(symbol as usize).copied().unwrap_or(1)
    }

    /// The rule names and claims the e-graph step uses.
    pub fn rules(&self) -> Vec<(String, Claim)> {
        self.rules_all.claimed()
    }

    fn effect_free(&self, t: &Tree) -> bool {
        !self.effect[t.symbol as usize] && t.kids.iter().all(|c| self.effect_free(c))
    }

    // ---- cleanse

    /// Dead work removed (module doc). `None` when nothing is dead or empty.
    ///
    /// A WRITE IS DEAD when no later read that a live statement makes uses its value (strong liveness, the faint
    /// variables of the dataflow literature): liveness runs backwards over the statements, a write of a variable not
    /// live after it is dead and reads nothing (so a value fed only to dead writes, or only to itself around a loop, is
    /// dead too), a register write kills its variable, an array element write does not, a loop or `if` runs its bodies
    /// any number of times in any order (iterated to its fixpoint), a statement that leaves early makes every variable
    /// live, and a workgroup array read anywhere is live everywhere (other invocations read it). Each dead write found
    /// this way is removed where it stands, so a variable written dead in one place and live in another keeps its live
    /// writes.
    pub fn cleanse(&self, k: &TrueKingdom, tree: &Tree) -> Result<Option<(Tree, Counts)>, String> {
        let mut universe: BTreeSet<u32> = BTreeSet::new();
        let mut stack = vec![tree];
        while let Some(t) = stack.pop() {
            if self.name_terminal[t.symbol as usize] {
                universe.insert(t.symbol);
            }
            stack.extend(t.kids.iter());
        }
        let exit: BTreeSet<u32> = universe.iter().copied().filter(|v| self.wg_name[*v as usize]).collect();
        let mut cx = Liveness { k, universe, path: Vec::new(), dead: BTreeSet::new(), defects: BTreeSet::new() };
        self.live_in(&mut cx, tree, &exit, true);
        let Liveness { dead, defects, .. } = cx;
        // the variables (declared names) of every row that is not a loop: a loop's counter is read only through these
        let mut named_outside: BTreeSet<&str> = BTreeSet::new();
        let mut stack = vec![tree];
        while let Some(t) = stack.pop() {
            if k.loops[t.symbol as usize].is_none() {
                named_outside.extend(k.rows[t.symbol as usize].declares.iter().map(String::as_str));
            }
            stack.extend(t.kids.iter());
        }
        let mut counts = Counts { defects: defects.into_iter().collect(), ..Counts::default() };
        let out = self.sweep(k, tree, &dead, &mut Vec::new(), &named_outside, &mut counts)?;
        if out == *tree {
            return Ok(None);
        }
        Ok(Some((out, counts)))
    }

    /// The variables read by a value subtree (every name terminal in it).
    fn reads_into(&self, t: &Tree, live: &mut BTreeSet<u32>) {
        if self.name_terminal[t.symbol as usize] {
            live.insert(t.symbol);
        }
        for c in &t.kids {
            self.reads_into(c, live);
        }
    }

    /// The variables live before statement `t` given those live after it (`out`). With `mark`, every dead write is
    /// recorded in `cx.dead` by its path of argument indices from the root (`cx.path` is `t`'s).
    fn live_in(&self, cx: &mut Liveness<'_>, t: &Tree, out: &BTreeSet<u32>, mark: bool) -> BTreeSet<u32> {
        let k = cx.k;
        let s = t.symbol as usize;
        let stmt = |c: &Tree| self.statement.contains(&k.rows[c.symbol as usize].out);
        let flow = if matches!(self.flow[s], Flow::Write { .. }) && !t.kids.first().is_some_and(|v| self.name_terminal[v.symbol as usize]) { &Flow::Any } else { &self.flow[s] };
        match flow {
            Flow::Write { kills } => {
                let var = t.kids[0].symbol;
                if !out.contains(&var) && t.kids[1..].iter().all(|c| self.effect_free(c)) {
                    let row = &k.rows[s];
                    if self.empty.contains_key(&row.out) {
                        if mark {
                            cx.dead.insert(cx.path.clone());
                        }
                        return out.clone();
                    }
                    cx.defects.insert(format!("cleanse: a dead write {} has no empty terminal of {}", row.name, k.type_name(row.out)));
                }
                let mut live = out.clone();
                if *kills {
                    live.remove(&var);
                }
                for c in &t.kids[1..] {
                    self.reads_into(c, &mut live);
                }
                live
            }
            Flow::Jump => cx.universe.clone(),
            Flow::Store => {
                // an output store is live by its row: never marked dead, and everything its values read is live
                let mut live = out.clone();
                for c in &t.kids {
                    self.reads_into(c, &mut live);
                }
                live
            }
            Flow::Straight(order) => {
                let mut live = out.clone();
                for &a in order.iter().rev() {
                    let c = &t.kids[a];
                    if stmt(c) {
                        cx.path.push(a);
                        live = self.live_in(cx, c, &live, mark);
                        cx.path.pop();
                    } else {
                        self.reads_into(c, &mut live);
                    }
                }
                live
            }
            Flow::Any => {
                // the bodies' exit: after the statement, before any body, or after a value is read
                let mut values = BTreeSet::new();
                for c in t.kids.iter().filter(|c| !stmt(c)) {
                    self.reads_into(c, &mut values);
                }
                let mut exit: BTreeSet<u32> = out.union(&values).copied().collect();
                loop {
                    let mut next = exit.clone();
                    for (a, c) in t.kids.iter().enumerate().filter(|(_, c)| stmt(c)) {
                        cx.path.push(a);
                        next.extend(self.live_in(cx, c, &exit, false));
                        cx.path.pop();
                    }
                    if next == exit {
                        break;
                    }
                    exit = next;
                }
                if mark {
                    for (a, c) in t.kids.iter().enumerate().filter(|(_, c)| stmt(c)) {
                        cx.path.push(a);
                        self.live_in(cx, c, &exit, true);
                        cx.path.pop();
                    }
                }
                exit
            }
        }
    }

    fn is_empty_terminal(&self, t: &Tree) -> bool {
        t.kids.is_empty() && self.empty.values().any(|e| *e == t.symbol)
    }

    fn sweep(&self, k: &TrueKingdom, t: &Tree, dead: &BTreeSet<Vec<usize>>, path: &mut Vec<usize>, named_outside: &BTreeSet<&str>, counts: &mut Counts) -> Result<Tree, String> {
        let s = t.symbol as usize;
        let row = &k.rows[s];
        // an output store stays as it is, by its row
        if self.store[s] {
            if dead.contains(path.as_slice()) {
                return Err(bug(format!("cleanse marked the output store {} dead", row.name)));
            }
            return Ok(t.clone());
        }
        // a dead write (found by `live_in`, so its value has no effect and its type has an empty terminal)
        if dead.contains(path.as_slice()) {
            let e = self.empty.get(&row.out).ok_or_else(|| bug(format!("cleanse marked {} dead, but {} has no empty terminal", row.name, k.type_name(row.out))))?;
            counts.dead_writes += 1;
            return Ok(Tree { symbol: *e, const_index: None, kids: Vec::new() });
        }
        let mut kids = Vec::with_capacity(t.kids.len());
        for (a, c) in t.kids.iter().enumerate() {
            if self.statement.contains(&k.rows[c.symbol as usize].out) {
                path.push(a);
                kids.push(self.sweep(k, c, dead, path, named_outside, counts)?);
                path.pop();
            } else {
                kids.push(c.clone());
            }
        }
        let rebuilt = Tree { symbol: t.symbol, const_index: t.const_index, kids };
        if self.is_empty_terminal(&rebuilt) || !self.statement.contains(&row.out) {
            return Ok(rebuilt);
        }
        let stmt_kid = |c: &Tree| self.statement.contains(&k.rows[c.symbol as usize].out);
        let all_empty = |ks: &[Tree]| ks.iter().filter(|c| stmt_kid(c)).all(|c| self.is_empty_terminal(c));
        let values_pure = |ks: &[Tree]| ks.iter().filter(|c| !stmt_kid(c)).all(|c| self.effect_free(c));
        let empty_now = if self.container[s] || self.is_if[s] {
            all_empty(&rebuilt.kids) && values_pure(&rebuilt.kids)
        } else if self.discard[s] {
            values_pure(&rebuilt.kids)
        } else if self.loop_body[s] {
            all_empty(&rebuilt.kids) && values_pure(&rebuilt.kids) && !row.declares.iter().any(|d| named_outside.contains(d.as_str()))
        } else {
            false
        };
        if empty_now {
            if let Some(e) = self.empty.get(&row.out) {
                counts.empties += 1;
                return Ok(Tree { symbol: *e, const_index: None, kids: Vec::new() });
            }
        }
        Ok(rebuilt)
    }

    // ---- simplify (table-row rewrites of scalar float arithmetic)

    fn value_of(&self, t: &Tree) -> Option<(Ty, f32)> {
        let i = t.const_index? as usize;
        self.consts.iter().find(|(_, c)| c.row == t.symbol).and_then(|(ty, c)| c.values.get(i).copied().flatten().map(|v| (*ty, v)))
    }

    /// The literal of type `ty` holding exactly `v` (bit for bit), if its `C_T` has it.
    fn literal(&self, ty: Ty, v: f32) -> Option<Tree> {
        let c = self.consts.get(&ty)?;
        let i = c.values.iter().position(|x| x.is_some_and(|x| x.to_bits() == v.to_bits()))?;
        Some(Tree { symbol: c.row, const_index: Some(i as u32), kids: Vec::new() })
    }

    /// The scalar float rewrites (module doc), bottom up to a fixed point; with `exact` only the exact ones.
    pub fn simplify(&self, k: &TrueKingdom, tree: &Tree, exact: bool, counts: &mut Counts) -> Tree {
        let mut t = tree.clone();
        loop {
            let before = counts.simplified;
            t = self.simplify_once(k, &t, exact, counts);
            if counts.simplified == before {
                return t;
            }
        }
    }

    fn simplify_once(&self, k: &TrueKingdom, t: &Tree, exact: bool, counts: &mut Counts) -> Tree {
        let kids: Vec<Tree> = t.kids.iter().map(|c| self.simplify_once(k, c, exact, counts)).collect();
        let node = Tree { symbol: t.symbol, const_index: t.const_index, kids };
        let out_of = |x: &Tree| k.rows[x.symbol as usize].out;
        if let Some(ty) = self.neg.get(&node.symbol) {
            let x = &node.kids[0];
            if self.neg.contains_key(&x.symbol) && out_of(&x.kids[0]) == *ty {
                counts.simplified += 1;
                return x.kids[0].clone();
            }
            if let Some((_, v)) = self.value_of(x) {
                match self.literal(*ty, -v) {
                    Some(l) => {
                        counts.simplified += 1;
                        return l;
                    }
                    None => counts.folds_missing += 1,
                }
            }
            return node;
        }
        let Some(b) = self.bins.get(&node.symbol) else { return node };
        let (x, y) = (&node.kids[0], &node.kids[1]);
        let (vx, vy) = (self.value_of(x).map(|v| v.1), self.value_of(y).map(|v| v.1));
        let keep = |c: &Tree| out_of(c) == b.out;
        let rewritten = (|| -> Option<Option<Tree>> {
            if let (Some(a), Some(c)) = (vx, vy) {
                let (r, is_exact) = match b.kind {
                    BinKind::Add => (a + c, true),
                    BinKind::Sub => (a - c, true),
                    BinKind::Mul => (a * c, true),
                    BinKind::Div => (a / c, false),
                };
                if (is_exact || !exact) && r.is_finite() {
                    // None inside: a fold whose value C_T does not hold
                    return Some(self.literal(b.out, r));
                }
            }
            let is = |v: Option<f32>, w: f32| v.is_some_and(|v| v.to_bits() == w.to_bits());
            match b.kind {
                BinKind::Mul => {
                    if is(vy, 1.0) && keep(x) {
                        return Some(Some(x.clone()));
                    }
                    if is(vx, 1.0) && keep(y) {
                        return Some(Some(y.clone()));
                    }
                    if !exact && (is(vx, 0.0) || is(vy, 0.0)) && self.effect_free(x) && self.effect_free(y) {
                        if let Some(z) = self.literal(b.out, 0.0) {
                            return Some(Some(z));
                        }
                    }
                }
                BinKind::Div if is(vy, 1.0) && keep(x) => return Some(Some(x.clone())),
                BinKind::Sub if is(vy, 0.0) && keep(x) => return Some(Some(x.clone())),
                BinKind::Add if !exact => {
                    if is(vy, 0.0) && keep(x) {
                        return Some(Some(x.clone()));
                    }
                    if is(vx, 0.0) && keep(y) {
                        return Some(Some(y.clone()));
                    }
                }
                _ => {}
            }
            // strength reduction: x / c -> x * (1 / c), in a mul row of the same shape
            if b.kind == BinKind::Div {
                if let Some(c) = vy.filter(|c| *c != 0.0 && c.is_finite()) {
                    let inv = 1.0 / c;
                    let power_of_two = c.is_normal() && c.to_bits() & 0x007f_ffff == 0 && inv.is_normal();
                    if power_of_two || !exact {
                        if let (Some(mul), Some(l)) = (self.bin_by_shape.get(&(BinKind::Mul as u8, b.args[0], b.args[1])), self.literal(b.args[1], inv)) {
                            return Some(Some(Tree { symbol: *mul, const_index: None, kids: vec![x.clone(), l] }));
                        }
                    }
                }
            }
            None
        })();
        match rewritten {
            Some(None) => {
                counts.folds_missing += 1;
                node
            }
            Some(Some(t)) => {
                counts.simplified += 1;
                t
            }
            None => node,
        }
    }

    // ---- the e-graph

    /// Simplify, the e-graph's cheapest program, simplify (module doc). `None` when the program is the input.
    pub fn egraph(&self, k: &TrueKingdom, tree: &Tree, exact: bool, iterations: usize, max_tuples: usize) -> Result<Option<(Tree, Counts)>, String> {
        let mut counts = Counts::default();
        let pre = self.simplify(k, tree, exact, &mut counts);
        let rules = if exact { &self.rules_exact } else { &self.rules_all };
        let eq = Equivalents::build(k, rules, &pre, iterations, max_tuples)?;
        let cheap = eq.cheapest(&|s| self.cost(s))?;
        let out = self.simplify(k, &cheap, exact, &mut counts);
        if out == *tree {
            return Ok(None);
        }
        Ok(Some((out, counts)))
    }

    // ---- hoist

    /// Loop-invariant subtrees moved before their outermost loop (module doc). `None` when nothing moves.
    pub fn hoist(&self, k: &TrueKingdom, tree: &Tree) -> Result<Option<(Tree, Counts)>, String> {
        let mut used: BTreeSet<u32> = BTreeSet::new();
        let mut stack = vec![tree];
        while let Some(t) = stack.pop() {
            used.insert(t.symbol);
            stack.extend(t.kids.iter());
        }
        let mut counts = Counts::default();
        let out = self.hoist_in(k, tree, &mut used, &mut counts)?;
        if out == *tree {
            return Ok(None);
        }
        Ok(Some((out, counts)))
    }

    fn functions_in(&self, k: &TrueKingdom, t: &Tree) -> usize {
        usize::from(k.rows[t.symbol as usize].kind == RowKind::Function) + t.kids.iter().map(|c| self.functions_in(k, c)).sum::<usize>()
    }

    fn invariant(&self, t: &Tree) -> bool {
        self.context_free.get(t.symbol as usize).copied().unwrap_or(false) && t.kids.iter().all(|c| self.invariant(c))
    }

    /// The maximal invariant subtrees of `t` worth a register at statement type `at` (two or more functions, a
    /// register family with a write row at `at`).
    fn candidates<'t>(&self, k: &TrueKingdom, t: &'t Tree, at: Ty, out: &mut Vec<&'t Tree>) {
        let ty = k.rows[t.symbol as usize].out;
        if !self.statement.contains(&ty) && self.invariant(t) && self.functions_in(k, t) >= 2 && self.regs.get(&ty).is_some_and(|f| f.set.contains_key(&at)) {
            if !out.contains(&t) {
                out.push(t);
            }
            return;
        }
        for c in &t.kids {
            self.candidates(k, c, at, out);
        }
    }

    fn hoist_in(&self, k: &TrueKingdom, t: &Tree, used: &mut BTreeSet<u32>, counts: &mut Counts) -> Result<Tree, String> {
        let s = t.symbol as usize;
        if self.loop_body[s] {
            let row = &k.rows[s];
            let at = row.out;
            let lr = k.loops[s].as_ref().ok_or_else(|| bug("a loop row without its loop"))?;
            let mut found: Vec<&Tree> = Vec::new();
            for (i, c) in t.kids.iter().enumerate() {
                if lr.inside.get(i).copied().unwrap_or(false) {
                    self.candidates(k, c, at, &mut found);
                }
            }
            if !found.is_empty() {
                if !self.seq.contains_key(&at) {
                    counts.hoist_unspellable += 1;
                } else {
                    let mut binds: Vec<(Tree, u32, Ty)> = Vec::new();
                    for v in found {
                        let ty = k.rows[v.symbol as usize].out;
                        let fam = &self.regs[&ty];
                        let Some(reg) = fam.terminals.iter().copied().find(|r| !used.contains(r)) else {
                            counts.hoist_unspellable += 1;
                            counts.defects.push(format!("hoist: no free register of {} at {}", k.type_name(ty), k.type_name(at)));
                            continue;
                        };
                        used.insert(reg);
                        binds.push((v.clone(), reg, ty));
                    }
                    if !binds.is_empty() {
                        fn replace(t: &Tree, binds: &[(Tree, u32, Ty)], o: &Optimisers) -> Tree {
                            if let Some((_, reg, ty)) = binds.iter().find(|(v, _, _)| v == t) {
                                return Tree { symbol: o.regs[ty].get, const_index: None, kids: vec![Tree { symbol: *reg, const_index: None, kids: Vec::new() }] };
                            }
                            Tree { symbol: t.symbol, const_index: t.const_index, kids: t.kids.iter().map(|c| replace(c, binds, o)).collect() }
                        }
                        let kids: Vec<Tree> = t.kids.iter().enumerate().map(|(i, c)| if lr.inside.get(i).copied().unwrap_or(false) { replace(c, &binds, self) } else { c.clone() }).collect();
                        let mut stmt = Tree { symbol: t.symbol, const_index: t.const_index, kids };
                        let (sb, s2) = self.seq[&at];
                        for (v, reg, ty) in binds.iter().rev() {
                            let set = Tree { symbol: self.regs[ty].set[&at], const_index: None, kids: vec![Tree { symbol: *reg, const_index: None, kids: Vec::new() }, v.clone()] };
                            stmt = Tree { symbol: sb, const_index: None, kids: vec![Tree { symbol: s2, const_index: None, kids: vec![set, stmt] }] };
                            counts.hoisted += 1;
                        }
                        return Ok(stmt);
                    }
                }
            }
        }
        let kids = t.kids.iter().map(|c| self.hoist_in(k, c, used, counts)).collect::<Result<Vec<_>, _>>()?;
        Ok(Tree { symbol: t.symbol, const_index: t.const_index, kids })
    }

    // ---- snap

    /// Float literals near a library constant ([`snap_library`]) snapped to that constant (module doc). A literal
    /// that already is a library value is left alone; of the values within `rel_tol` the nearest is taken. Its
    /// `C_T` entry is used when `C_T` holds it bit for bit, and otherwise appended to `C_T` (append-only for the run's
    /// epoch; the caller rebuilds what reads `C_T`). `None` when nothing snaps.
    pub fn snap(&self, k: &mut TrueKingdom, tree: &Tree, rel_tol: f64) -> (Option<Tree>, Counts) {
        let library = snap_library();
        let mut counts = Counts::default();
        fn index_of(k: &TrueKingdom, ty: Ty, v: f32) -> Option<u32> {
            k.constants.get(&ty)?.iter().position(|t| t.trim().parse::<f64>().ok().is_some_and(|x| (x as f32).to_bits() == v.to_bits())).map(|i| i as u32)
        }
        fn go(o: &Optimisers, k: &mut TrueKingdom, t: &Tree, lib: &[(String, f64)], tol: f64, counts: &mut Counts) -> Tree {
            if let Some((ty, v)) = o.value_of(t) {
                let is_library = lib.iter().any(|(_, s)| (*s as f32).to_bits() == v.to_bits());
                if v.is_finite() && v != 0.0 && !is_library {
                    let v64 = f64::from(v);
                    let near = lib.iter().filter(|(_, s)| (v64 - s).abs() <= tol * s.abs()).min_by(|a, b| (v64 - a.1).abs().total_cmp(&(v64 - b.1).abs()));
                    if let Some((_, s)) = near {
                        let exact = *s as f32;
                        let i = match index_of(k, ty, exact) {
                            Some(i) => i,
                            None => {
                                let v = k.constants.entry(ty).or_default();
                                v.push(format!("{exact:?}"));
                                counts.appended += 1;
                                v.len() as u32 - 1
                            }
                        };
                        counts.snapped += 1;
                        return Tree { symbol: t.symbol, const_index: Some(i), kids: Vec::new() };
                    }
                }
            }
            Tree { symbol: t.symbol, const_index: t.const_index, kids: t.kids.iter().map(|c| go(o, k, c, lib, tol, counts)).collect() }
        }
        let out = go(self, k, tree, &library, rel_tol, &mut counts);
        (if out == *tree { None } else { Some(out) }, counts)
    }

    // ---- fewer loop trips

    /// The counted loops of `tree` whose bound is a literal with a smaller entry in its `C_T`, in pre-order.
    pub fn trip_sites(&self, k: &TrueKingdom, tree: &Tree) -> usize {
        let mut n = 0;
        let mut stack = vec![tree];
        while let Some(t) = stack.pop() {
            if self.lower_bound(k, t).is_some() {
                n += 1;
            }
            stack.extend(t.kids.iter());
        }
        n
    }

    /// For a loop node with a literal bound `n`: (argument, the `C_T` index of the largest literal below `n`, at least 1).
    fn lower_bound(&self, k: &TrueKingdom, t: &Tree) -> Option<(usize, u32)> {
        let l = k.loops.get(t.symbol as usize)?.as_ref()?;
        let i = l.literal_arg?;
        let kid = t.kids.get(i)?;
        let row = &k.rows[kid.symbol as usize];
        let c = kid.const_index?;
        if row.kind != RowKind::Constant {
            return None;
        }
        let texts = k.constants.get(&row.out)?;
        let n: u64 = texts.get(c as usize)?.trim_end_matches('u').parse().ok()?;
        let parsed = |x: &String| x.trim_end_matches('u').parse::<u64>().ok();
        texts.iter().enumerate().filter_map(|(j, x)| parsed(x).map(|v| (j, v))).filter(|(_, v)| *v >= 1 && *v < n).max_by_key(|(_, v)| *v).map(|(j, _)| (i, j as u32))
    }

    /// One loop, drawn by `rng` among [`trip_sites`](Optimisers::trip_sites), with its literal bound lowered to the next
    /// smaller literal. A variation, not an equivalence.
    pub fn fewer_trips(&self, k: &TrueKingdom, tree: &Tree, rng: &mut Rng) -> Option<Tree> {
        let sites = self.trip_sites(k, tree);
        if sites == 0 {
            return None;
        }
        self.fewer_trips_at(k, tree, rng.below(sites))
    }

    /// Site `site` (pre-order among [`trip_sites`](Optimisers::trip_sites)) with its literal bound lowered.
    pub fn fewer_trips_at(&self, k: &TrueKingdom, tree: &Tree, site: usize) -> Option<Tree> {
        fn go(o: &Optimisers, k: &TrueKingdom, t: &Tree, pick: usize, seen: &mut usize) -> Tree {
            if let Some((i, j)) = o.lower_bound(k, t) {
                if *seen == pick {
                    *seen += 1;
                    let mut kids = t.kids.clone();
                    kids[i].const_index = Some(j);
                    return Tree { symbol: t.symbol, const_index: t.const_index, kids };
                }
                *seen += 1;
            }
            Tree { symbol: t.symbol, const_index: t.const_index, kids: t.kids.iter().map(|c| go(o, k, c, pick, seen)).collect() }
        }
        let mut seen = 0;
        let out = go(self, k, tree, site, &mut seen);
        (out != *tree).then_some(out)
    }
}

/// The snap library: pi, e, sqrt 2, tau, 1/sqrt 2, ln 2, 2 pi, pi/2, and every rational p/q with q <= 12 and
/// |p/q| <= 64. The literals of a WGSL kernel are entries of a finite `C_T`, so a constant is a near miss only against
/// a short exact value it could have been (0.59719 against 3/5, 1.60475 against 8/5).
pub fn snap_library() -> Vec<(String, f64)> {
    use std::f64::consts::{E, FRAC_1_SQRT_2, FRAC_PI_2, LN_2, PI, SQRT_2, TAU};
    let mut out: Vec<(String, f64)> = [("pi", PI), ("e", E), ("sqrt2", SQRT_2), ("tau", TAU), ("1/sqrt2", FRAC_1_SQRT_2), ("ln2", LN_2), ("pi/2", FRAC_PI_2)].iter().map(|(n, v)| (n.to_string(), *v)).collect();
    let mut seen: BTreeSet<u64> = out.iter().map(|(_, v)| v.to_bits()).collect();
    for q in 1..=12i64 {
        for p in -64 * q..=64 * q {
            let v = p as f64 / q as f64;
            if p != 0 && seen.insert(v.to_bits()) {
                out.push((format!("{p}/{q}"), v));
            }
        }
    }
    out
}

/// THE RUN'S EPOCH (`MultiTypedGEP.md` section 6): the kingdom plus one ADF reference terminal `A_d` for every ADF the
/// run's fold has made, append-only, so `A_d` is symbol `n_base + d` for the whole run and a gene written in an earlier
/// epoch reads the same in every later one. An ADF is an immutable expanded tree of base rows (its root is never the
/// kingdom's root); a gene holding `A_d` expresses its tree by inlining.
#[derive(Debug, Clone)]
pub struct RunEpoch {
    base: crate::kingdom_contract::KingdomContract,
    root: Ty,
    epoch: crate::typed_chromosome::Epoch,
    /// By id: the expanded body, its root type, its level.
    bodies: Vec<(Tree, Ty, u32)>,
    index: std::collections::HashMap<Tree, u32>,
}

impl RunEpoch {
    pub fn new(base: &crate::kingdom_contract::KingdomContract, root: Ty) -> Result<RunEpoch, String> {
        let epoch = crate::typed_chromosome::Epoch::new(base, &[], &[]).map_err(|e| e.to_string())?;
        Ok(RunEpoch { base: base.clone(), root, epoch, bodies: Vec::new(), index: std::collections::HashMap::new() })
    }

    /// The epoch alphabet as a contract: genes are laid out, drawn, varied and expressed against this.
    pub fn contract(&self) -> &crate::kingdom_contract::KingdomContract {
        self.epoch.kingdom()
    }

    pub fn adfs(&self) -> usize {
        self.bodies.len()
    }

    fn n_base(&self) -> usize {
        self.epoch.n_base()
    }

    /// The tree with every `A_d` replaced by its body: base rows only.
    pub fn inline(&self, t: &Tree) -> Tree {
        let n = self.n_base() as u32;
        if t.symbol >= n {
            return self.bodies[(t.symbol - n) as usize].0.clone();
        }
        Tree { symbol: t.symbol, const_index: t.const_index, kids: t.kids.iter().map(|c| self.inline(c)).collect() }
    }

    /// A base tree with every subtree that is an ADF's body replaced by its reference, top down (the largest first).
    pub fn fold_known(&self, t: &Tree) -> Tree {
        if let Some(id) = self.index.get(t) {
            return Tree { symbol: self.n_base() as u32 + id, const_index: None, kids: Vec::new() };
        }
        Tree { symbol: t.symbol, const_index: t.const_index, kids: t.kids.iter().map(|c| self.fold_known(c)).collect() }
    }

    /// THE FOLD of a population (`trees`: base trees): every subtree holding a function, not rooted at the kingdom's
    /// root, with two or more DAG edges in two or more individuals becomes an ADF (an existing body keeps its id); the
    /// epoch is rebuilt with every ADF made so far. Returns how many ADFs are new. Each individual's gene is then
    /// [`fold_known`](RunEpoch::fold_known) of its tree, written back: a removal against the gene's own tree, because
    /// every reference the gene held is to an ADF that still exists.
    pub fn fold(&mut self, trees: &[Tree]) -> Result<usize, String> {
        let mut ids: std::collections::HashMap<&Tree, usize> = std::collections::HashMap::new();
        let mut info: Vec<(&Tree, u64, u64, usize)> = Vec::new(); // tree, functions, edges, individuals
        let base = &self.base;
        fn walk<'t>(t: &'t Tree, base: &crate::kingdom_contract::KingdomContract, ids: &mut std::collections::HashMap<&'t Tree, usize>, info: &mut Vec<(&'t Tree, u64, u64, usize)>, seen: &mut BTreeSet<usize>) -> (usize, u64) {
            let mut f = u64::from(!base.is_terminal(t.symbol));
            if let Some(&id) = ids.get(t) {
                info[id].2 += 1;
                if seen.insert(id) {
                    info[id].3 += 1;
                }
                return (id, info[id].1);
            }
            for c in &t.kids {
                f += walk(c, base, ids, info, seen).1;
            }
            let id = info.len();
            info.push((t, f, 1, 1));
            ids.insert(t, id);
            seen.insert(id);
            (id, f)
        }
        for t in trees {
            let mut seen = BTreeSet::new();
            walk(t, base, &mut ids, &mut info, &mut seen);
        }
        let mut made = 0;
        // post-order ids: a body after everything it holds, so a lower ADF has a lower id
        for (t, f, edges, individuals) in &info {
            let qualifies = *f >= 1 && *edges >= 2 && *individuals >= 2 && base.output(t.symbol) != self.root;
            if qualifies && !self.index.contains_key(*t) {
                fn deepest(t: &Tree, index: &std::collections::HashMap<Tree, u32>, bodies: &[(Tree, Ty, u32)], top: bool) -> Option<u32> {
                    if !top {
                        if let Some(id) = index.get(t) {
                            return Some(bodies[*id as usize].2);
                        }
                    }
                    t.kids.iter().filter_map(|c| deepest(c, index, bodies, false)).max()
                }
                let level = deepest(t, &self.index, &self.bodies, true).map_or(0, |l| l + 1);
                let id = self.bodies.len() as u32;
                self.bodies.push(((*t).clone(), base.output(t.symbol), level));
                self.index.insert((*t).clone(), id);
                made += 1;
            }
        }
        if made > 0 {
            let decls: Vec<crate::typed_chromosome::AdfDecl> = self.bodies.iter().enumerate().map(|(i, (_, root, level))| crate::typed_chromosome::AdfDecl { id: i as u64, root: *root, level: *level }).collect();
            self.epoch = crate::typed_chromosome::Epoch::new(&self.base, &[], &decls).map_err(|e| e.to_string())?;
            for i in 0..self.bodies.len() {
                if self.epoch.adf_terminal(i as u64) != Some((self.n_base() + i) as u32) {
                    return Err(bug(format!("ADF {i} is not symbol n_base + {i} in the rebuilt epoch")));
                }
            }
        }
        Ok(made)
    }
}
