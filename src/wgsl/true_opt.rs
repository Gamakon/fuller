//! THE IN-LOOP OPTIMISERS OF THE TRUE WGSL KINGDOM (`docs/PLAN_true_gep_wgsl.md` ST10): the rewrites the SR engine runs
//! on its winners (cleanse, the e-graph, snap), made for kernels. Each takes an expressed tree of the table's rows and
//! returns another tree of the table's rows, so the per-type write-back (`MultiTypedGEP.md` section 5) puts it into a
//! gene that expresses a legal kernel by construction; nothing here judges a kernel, fitness does.
//!
//! * [`Optimisers::cleanse`]: dead work. A write to a register, a uniform register, a private array or a workgroup
//!   array (the `reg.set_*`, `ureg.set*`, `parr.set_*`, `wg.set_*` rows: arg 0 names the variable) is dead when no read of
//!   that variable reaches anything a live statement uses: the live variables are those read outside any write's value,
//!   closed under "read in the value of a write to a live variable". A dead write whose other arguments have no effect
//!   becomes its statement type's empty terminal (`{}`); then every statement left empty (a block of empties, an `if`
//!   over empties with an effect-free condition, a `_ = x;` of an effect-free `x`, a counted loop over an empty body
//!   whose counter nothing else reads) becomes the empty terminal too. A write spelled with a barrier (`wg.put_*`) is
//!   never removed: its barriers may order other memory.
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
    /// Per row: a write whose arg 0 is the variable written and whose spelling has no barrier.
    write: Vec<bool>,
    /// Per row: its spelling acts beyond its value, or it outputs a statement.
    effect: Vec<bool>,
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
    pub fn new(k: &TrueKingdom) -> Result<Optimisers, String> {
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
        let is_if: Vec<bool> = k.rows.iter().map(|r| r.name.starts_with("stmt.if") && r.kind == RowKind::Function && statement.contains(&r.out)).collect();
        let discard: Vec<bool> = k.rows.iter().map(|r| r.name.starts_with("stmt.discard") && r.kind == RowKind::Function && r.args.iter().all(|a| !statement.contains(a))).collect();
        let loop_body: Vec<bool> = (0..n)
            .map(|i| {
                let r = &k.rows[i];
                k.loops[i].as_ref().is_some_and(|l| !l.own_work) && statement.contains(&r.out) && r.args.iter().any(|a| statement.contains(a))
            })
            .collect();
        // float literal rows: every C_T entry parses as a float
        let mut consts = BTreeMap::new();
        for (i, r) in k.rows.iter().enumerate() {
            if r.kind != RowKind::Constant || !dual(r.out).starts_with("f32.") {
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
        Ok(Optimisers { statement, empty, name_terminal, write, effect, container, is_if, discard, loop_body, context_free: context_free_rows(k), consts, bins, bin_by_shape, neg, regs, seq, row_cost, rules_all, rules_exact })
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
    pub fn cleanse(&self, k: &TrueKingdom, tree: &Tree) -> Result<Option<(Tree, Counts)>, String> {
        // reads: (owner write's variable or none, variable read)
        let mut reads: Vec<(Option<u32>, u32)> = Vec::new();
        fn walk(o: &Optimisers, t: &Tree, owner: Option<u32>, reads: &mut Vec<(Option<u32>, u32)>) {
            if o.write[t.symbol as usize] && t.kids.first().is_some_and(|v| o.name_terminal[v.symbol as usize]) {
                let var = t.kids[0].symbol;
                for c in &t.kids[1..] {
                    walk(o, c, Some(owner.unwrap_or(var)), reads);
                }
                return;
            }
            if o.name_terminal[t.symbol as usize] {
                reads.push((owner, t.symbol));
            }
            for c in &t.kids {
                walk(o, c, owner, reads);
            }
        }
        walk(self, tree, None, &mut reads);
        let mut live: BTreeSet<u32> = reads.iter().filter(|(o, _)| o.is_none()).map(|(_, v)| *v).collect();
        loop {
            let before = live.len();
            for (o, v) in &reads {
                if let Some(w) = o {
                    if live.contains(w) {
                        live.insert(*v);
                    }
                }
            }
            if live.len() == before {
                break;
            }
        }
        // the variables (declared names) of every row that is not a loop: a loop's counter is read only through these
        let mut named_outside: BTreeSet<&str> = BTreeSet::new();
        let mut stack = vec![tree];
        while let Some(t) = stack.pop() {
            if k.loops[t.symbol as usize].is_none() {
                named_outside.extend(k.rows[t.symbol as usize].declares.iter().map(String::as_str));
            }
            stack.extend(t.kids.iter());
        }
        let mut counts = Counts::default();
        let out = self.sweep(k, tree, &live, &named_outside, &mut counts)?;
        if out == *tree {
            return Ok(None);
        }
        Ok(Some((out, counts)))
    }

    fn is_empty_terminal(&self, t: &Tree) -> bool {
        t.kids.is_empty() && self.empty.values().any(|e| *e == t.symbol)
    }

    fn sweep(&self, k: &TrueKingdom, t: &Tree, live: &BTreeSet<u32>, named_outside: &BTreeSet<&str>, counts: &mut Counts) -> Result<Tree, String> {
        let s = t.symbol as usize;
        let row = &k.rows[s];
        // a dead write
        if self.write[s] && t.kids.first().is_some_and(|v| self.name_terminal[v.symbol as usize] && !live.contains(&v.symbol)) && t.kids[1..].iter().all(|c| self.effect_free(c)) {
            if let Some(e) = self.empty.get(&row.out) {
                counts.dead_writes += 1;
                return Ok(Tree { symbol: *e, const_index: None, kids: Vec::new() });
            }
            counts.defects.push(format!("cleanse: a dead write {} has no empty terminal of {}", row.name, k.type_name(row.out)));
        }
        let kids: Vec<Tree> = t.kids.iter().map(|c| if self.statement.contains(&k.rows[c.symbol as usize].out) { self.sweep(k, c, live, named_outside, counts) } else { Ok(c.clone()) }).collect::<Result<_, _>>()?;
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

    /// Float literals near a library constant snapped to the `C_T` entry that is it (module doc). `None` when nothing
    /// snaps; a near miss `C_T` cannot spell is a named defect in the counts.
    pub fn snap(&self, k: &TrueKingdom, tree: &Tree, rel_tol: f64) -> (Option<Tree>, Counts) {
        let library = crate::snap::default_constants();
        let mut counts = Counts::default();
        fn go(o: &Optimisers, k: &TrueKingdom, t: &Tree, lib: &[(&str, f64)], tol: f64, counts: &mut Counts) -> Tree {
            if let Some((ty, v)) = o.value_of(t) {
                let v64 = f64::from(v);
                for (name, s) in lib {
                    for target in [*s, -*s] {
                        let exact = target as f32;
                        if v.to_bits() != exact.to_bits() && (v64 - target).abs() <= tol * target.abs() {
                            match o.literal(ty, exact) {
                                Some(l) => {
                                    counts.snapped += 1;
                                    return l;
                                }
                                None => {
                                    let d = format!("snap: {v} is within {tol} of {}{name}, which C_T of {} does not hold", if target < 0.0 { "-" } else { "" }, k.type_name(ty));
                                    if !counts.defects.contains(&d) {
                                        counts.defects.push(d);
                                    }
                                }
                            }
                        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_gep::{express, write_back, GeneLayout};
    use crate::wgsl::cuneus_originals::{image_dir, nebula, volumepassage};
    use crate::wgsl::true_kingdom::{random_gene, Share};

    fn assert_typed(k: &TrueKingdom, t: &Tree) {
        let row = &k.rows[t.symbol as usize];
        assert_eq!(row.args.len(), t.kids.len(), "{}", row.name);
        for (i, kid) in t.kids.iter().enumerate() {
            let kr = &k.rows[kid.symbol as usize];
            assert_eq!(kr.out, row.args[i], "{} argument {i} is {} ({}), the signature asks for {}", row.name, kr.name, k.type_name(kr.out), k.type_name(row.args[i]));
            assert_typed(k, kid);
        }
    }

    fn node(k: &TrueKingdom, name: &str, value: Option<&str>, kids: Vec<Tree>) -> Tree {
        let i = k.rows.iter().position(|r| r.name == name).unwrap_or_else(|| panic!("no row {name}"));
        let const_index = value.map(|v| k.constants[&k.rows[i].out].iter().position(|t| t == v).unwrap_or_else(|| panic!("{name}: no constant {v}")) as u32);
        Tree { symbol: i as u32, const_index, kids }
    }

    fn count(k: &TrueKingdom, t: &Tree, pred: &dyn Fn(&str) -> bool) -> usize {
        usize::from(pred(&k.rows[t.symbol as usize].name)) + t.kids.iter().map(|c| count(k, c, pred)).sum::<usize>()
    }

    fn functions(t: &Tree, k: &TrueKingdom) -> usize {
        usize::from(!k.rows[t.symbol as usize].args.is_empty()) + t.kids.iter().map(|c| functions(c, k)).sum::<usize>()
    }

    /// The tree written into a gene at its own program size reads back unchanged; it renders and validates in naga.
    fn writes_back_and_validates(k: &TrueKingdom, t: &Tree) {
        let c = k.relations().check().expect("check");
        let layout = GeneLayout::from_program_size(&c, k.root, functions(t, k)).expect("layout");
        let cl = k.const_lens();
        let previous = random_gene(&c, &layout, &cl, &mut Rng(5), Share::Half, true);
        let gene = write_back(&c, &layout, &previous, t).expect("write-back");
        assert_eq!(&express(&c, &layout, &gene, &cl).expect("express").tree(), t);
        let src = k.render(t).expect("render");
        let module = crate::wgsl::naga_shape::parse_unscanned(&src).expect("naga parse");
        crate::wgsl::target::Target::current().validate(&module, &src).expect("naga validate");
    }

    /// The first statement of a type with an f32 register write row and a two-statement block, `S`, replaced by
    /// `{ w; S }` where `w` writes `reg` (a planted dead write), or with `empty` the type's empty terminal `{}` in its place.
    fn plant_dead_write(k: &TrueKingdom, t: &Tree, o: &Optimisers, reg: u32, empty: bool, done: &mut bool) -> Tree {
        let ty = k.rows[t.symbol as usize].out;
        let f32_ty = k.rows.iter().find(|r| r.name == "literal.real").expect("literal.real").out;
        if !*done && k.type_name(ty).starts_with("stmt.") {
            if let (Some(set), Some((sb, s2)), Some(e)) = (o.regs.get(&f32_ty).and_then(|f| f.set.get(&ty)), o.seq.get(&ty), o.empty.get(&ty)) {
                *done = true;
                let w = if empty { Tree { symbol: *e, const_index: None, kids: Vec::new() } } else { Tree { symbol: *set, const_index: None, kids: vec![Tree { symbol: reg, const_index: None, kids: Vec::new() }, node(k, "literal.real", Some("2.0"), vec![])] } };
                return Tree { symbol: *sb, const_index: None, kids: vec![Tree { symbol: *s2, const_index: None, kids: vec![w, t.clone()] }] };
            }
        }
        Tree { symbol: t.symbol, const_index: t.const_index, kids: t.kids.iter().map(|c| plant_dead_write(k, c, o, reg, empty, done)).collect() }
    }

    /// The encoded volumepassage's own dead work goes (the shipped shader's `i`, written twice and never read); a dead
    /// write planted in it is removed too (the cleansed planted tree is the cleansed original), a write read only by a
    /// dead write is dead, every live write is kept, and the result is typed, writes back and validates.
    #[test]
    fn cleanse_removes_a_planted_dead_write_and_keeps_the_live_ones() {
        let k = TrueKingdom::load(&image_dir("volumepassage")).expect("table");
        let o = Optimisers::new(&k).expect("optimisers");
        let tree = volumepassage(&k).unwrap_or_else(|g| panic!("{g:?}"));
        let base = o.cleanse(&k, &tree).expect("cleanse").map_or(tree.clone(), |(t, _)| t);
        let writes = |t: &Tree| count(&k, t, &|n: &str| n.starts_with("reg.set_"));
        // the shipped shader's own dead work: `var i = 0.0;` and `i += 1.0;` in the march, `i` never read
        assert_eq!(writes(&base), writes(&tree) - 2, "the original's two writes of its unread `i` are dead, every other write is live");
        let reg = k.rows.iter().position(|r| r.name == "reg.rf63").expect("rf63") as u32;
        let plant = |t: &Tree, reg: u32, empty: bool| {
            let mut done = false;
            let p = plant_dead_write(&k, t, &o, reg, empty, &mut done);
            assert!(done, "a statement with an f32 write row and a two-statement block");
            p
        };
        let planted = plant(&tree, reg, false);
        let with_empty = plant(&tree, reg, true);
        let want = o.cleanse(&k, &with_empty).expect("cleanse").map_or(with_empty.clone(), |(t, _)| t);
        assert_eq!(writes(&planted), writes(&tree) + 1);
        let (cleansed, counts) = o.cleanse(&k, &planted).expect("cleanse").expect("the planted write is dead");
        assert!(counts.dead_writes >= 3, "the planted write and the original's two: {counts:?}");
        assert_eq!(cleansed, want, "the planted write is removed and nothing else differs");
        assert_typed(&k, &cleansed);
        writes_back_and_validates(&k, &cleansed);
        // a write read only by a dead write is dead too: rf62 = 2.0; rf63 = rf62
        let f32_ty = k.rows.iter().find(|r| r.name == "literal.real").expect("lit").out;
        let get = o.regs[&f32_ty].get;
        let r62 = k.rows.iter().position(|r| r.name == "reg.rf62").expect("rf62") as u32;
        let mut second = plant(&plant(&tree, r62, false), reg, false);
        // make the rf63 write read rf62
        fn reread(t: &mut Tree, reg: u32, get: u32, r62: u32) -> bool {
            if t.kids.first().is_some_and(|v| v.symbol == reg) && t.kids.len() == 2 {
                t.kids[1] = Tree { symbol: get, const_index: None, kids: vec![Tree { symbol: r62, const_index: None, kids: Vec::new() }] };
                return true;
            }
            t.kids.iter_mut().any(|c| reread(c, reg, get, r62))
        }
        assert!(reread(&mut second, reg, get, r62));
        let (_, counts) = o.cleanse(&k, &second).expect("cleanse").expect("dead chain");
        assert_eq!(counts.dead_writes, 4, "both writes of the dead chain and the original's two: {counts:?}");
        // a planted write of a register the kernel reads is live: plant a write of the first f32 register the cleansed
        // original still writes
        fn first_write(k: &TrueKingdom, t: &Tree) -> Option<u32> {
            if k.rows[t.symbol as usize].name.starts_with("reg.set_f_") {
                return t.kids.first().map(|v| v.symbol);
            }
            t.kids.iter().find_map(|c| first_write(k, c))
        }
        let live_reg = first_write(&k, &base).expect("a live f32 write");
        let live = plant(&tree, live_reg, false);
        let kept = o.cleanse(&k, &live).expect("cleanse").map_or(live.clone(), |(t, _)| t);
        assert_eq!(writes(&kept), writes(&base) + 1, "the planted write of a read register stays");
    }

    /// Simplify on small trees: `x * 1` is `x`, `2 + 1` is the literal 3, `x / 2` is `x * 0.5` (exact), `x / 3` keeps
    /// its division under exact (3 is not a power of two).
    #[test]
    fn simplify_folds_constants_and_removes_identities() {
        let k = TrueKingdom::load(&image_dir("volumepassage")).expect("table");
        let o = Optimisers::new(&k).expect("optimisers");
        let x = node(&k, "reg.get_f", None, vec![node(&k, "reg.rf0", None, vec![])]);
        let lit = |v: &str| node(&k, "literal.real", Some(v), vec![]);
        let mut c = Counts::default();
        assert_eq!(o.simplify(&k, &node(&k, "arith.mul", None, vec![x.clone(), lit("1.0")]), true, &mut c), x);
        assert_eq!(o.simplify(&k, &node(&k, "arith.add", None, vec![lit("2.0"), lit("1.0")]), true, &mut c), lit("3.0"));
        assert_eq!(o.simplify(&k, &node(&k, "arith.div", None, vec![x.clone(), lit("2.0")]), true, &mut c), node(&k, "arith.mul", None, vec![x.clone(), lit("0.5")]));
        let third = node(&k, "arith.div", None, vec![x.clone(), lit("3.0")]);
        assert_eq!(o.simplify(&k, &third, true, &mut c), third, "3 is not a power of two");
        assert_eq!(o.simplify(&k, &node(&k, "arith.neg", None, vec![node(&k, "arith.neg", None, vec![x.clone()])]), true, &mut c), x);
        assert_eq!(o.simplify(&k, &node(&k, "arith.add", None, vec![x.clone(), lit("0.0")]), true, &mut c), node(&k, "arith.add", None, vec![x.clone(), lit("0.0")]), "x + 0 is not exact (-0)");
        assert_eq!(o.simplify(&k, &node(&k, "arith.add", None, vec![x.clone(), lit("0.0")]), false, &mut c), x);
        // the bare twin: x * bare 1.0
        if k.rows.iter().any(|r| r.name == "arith.mul_bare1") {
            assert_eq!(o.simplify(&k, &node(&k, "arith.mul_bare1", None, vec![x.clone(), node(&k, "bare.real", Some("1.0"), vec![])]), true, &mut c), x);
        }
    }

    /// The e-graph step on both encoded originals: exact and real cheapest programs are typed trees of rows that cost
    /// no more than the original, write back into a gene and validate in naga.
    #[test]
    fn the_cheapest_programs_of_both_originals_are_typed_and_write_back() {
        for (shader, build) in [("volumepassage", volumepassage as fn(&TrueKingdom) -> Result<Tree, Vec<String>>), ("nebula", nebula)] {
            let k = TrueKingdom::load(&image_dir(shader)).expect("table");
            let o = Optimisers::new(&k).expect("optimisers");
            let tree = build(&k).unwrap_or_else(|g| panic!("{g:?}"));
            let cost = |t: &Tree| -> u64 {
                fn c(o: &Optimisers, t: &Tree) -> u64 {
                    o.cost(t.symbol) + t.kids.iter().map(|k| c(o, k)).sum::<u64>()
                }
                c(&o, t)
            };
            for exact in [true, false] {
                match o.egraph(&k, &tree, exact, 3, 2_000_000).expect("e-graph") {
                    Some((t, counts)) => {
                        assert_typed(&k, &t);
                        assert!(cost(&t) <= cost(&tree), "{shader} exact {exact}: cost {} against {}", cost(&t), cost(&tree));
                        writes_back_and_validates(&k, &t);
                        eprintln!("{shader} exact {exact}: {} -> {} nodes, cost {} -> {}, {counts:?}", tree.level_order().len(), t.level_order().len(), cost(&tree), cost(&t));
                    }
                    None => eprintln!("{shader} exact {exact}: the original is already the cheapest program"),
                }
            }
        }
    }

    /// Hoist on the encoded volumepassage: invariant subtrees leave the march, each read inside by its register; the
    /// tree is typed, writes back and validates; on the hoisted tree hoist finds nothing more.
    #[test]
    fn hoist_moves_invariant_subtrees_out_of_the_march() {
        let k = TrueKingdom::load(&image_dir("volumepassage")).expect("table");
        let o = Optimisers::new(&k).expect("optimisers");
        let tree = volumepassage(&k).unwrap_or_else(|g| panic!("{g:?}"));
        let Some((t, counts)) = o.hoist(&k, &tree).expect("hoist") else {
            eprintln!("volumepassage: no invariant subtree of two functions in a loop");
            return;
        };
        assert!(counts.hoisted >= 1, "{counts:?}");
        assert_typed(&k, &t);
        writes_back_and_validates(&k, &t);
        assert_eq!(o.hoist(&k, &t).expect("hoist again").map(|x| x.1.hoisted), None, "nothing left to hoist");
        eprintln!("volumepassage hoist: {counts:?}, {} -> {} nodes", tree.level_order().len(), t.level_order().len());
    }

    /// Snap on a table whose `C_T` holds a near-pi literal: it becomes the exact pi of `C_T`; with no exact entry, the
    /// near miss is a named defect and nothing changes.
    #[test]
    fn snap_moves_a_near_constant_to_the_exact_one_or_names_the_missing_constant() {
        let mut k = TrueKingdom::load(&image_dir("volumepassage")).expect("table");
        let lit = k.rows.iter().position(|r| r.name == "literal.real").expect("literal.real");
        let ty = k.rows[lit].out;
        k.constants.get_mut(&ty).expect("C_T").push("3.1416".to_string());
        let o = Optimisers::new(&k).expect("optimisers");
        let near = Tree { symbol: lit as u32, const_index: Some(k.constants[&ty].len() as u32 - 1), kids: Vec::new() };
        let x = node(&k, "arith.mul", None, vec![node(&k, "reg.get_f", None, vec![node(&k, "reg.rf0", None, vec![])]), near.clone()]);
        let (snapped, counts) = o.snap(&k, &x, 1e-3);
        let pi = k.constants[&ty].iter().position(|t| t.parse::<f64>().ok().map(|v| v as f32) == Some(std::f32::consts::PI)).expect("pi in C_T");
        assert_eq!(snapped.expect("snapped").kids[1].const_index, Some(pi as u32));
        assert_eq!(counts.snapped, 1);
        let only_pi = k.constants[&ty][pi].clone();
        k.constants.get_mut(&ty).expect("C_T")[pi] = "3.0000001".to_string();
        let o2 = Optimisers::new(&k).expect("optimisers");
        let (none, counts) = o2.snap(&k, &x, 1e-3);
        assert!(none.is_none() && counts.defects.iter().any(|d| d.contains("pi")), "{counts:?} (C_T had {only_pi})");
    }

    /// Fewer trips: one of the encoded volumepassage's literal-bound loops gets the next smaller literal, every other
    /// node unchanged; the work bound falls.
    #[test]
    fn fewer_trips_lowers_one_literal_bound() {
        let k = TrueKingdom::load(&image_dir("volumepassage")).expect("table");
        let o = Optimisers::new(&k).expect("optimisers");
        let tree = volumepassage(&k).unwrap_or_else(|g| panic!("{g:?}"));
        assert!(o.trip_sites(&k, &tree) >= 2, "the march and the noise");
        let t = o.fewer_trips(&k, &tree, &mut Rng(1)).expect("a lowered loop");
        let a = tree.level_order();
        let b = t.level_order();
        assert_eq!(a.len(), b.len());
        let differ = a.iter().zip(&b).filter(|(x, y)| (x.0.symbol, x.0.const_index) != (y.0.symbol, y.0.const_index)).count();
        assert_eq!(differ, 1);
        assert!(k.tree_work_bound(&t).expect("bound") < k.tree_work_bound(&tree).expect("bound"));
    }

    /// EVERY REWRITE THAT CLAIMS EQUIVALENCE KEEPS THE PICTURE, on the device: the encoded volumepassage and nebula,
    /// each through the cleanse (with a planted dead write), the exact e-graph, the hoist, all three in turn, and the
    /// share pass's lets, drawn at 64 x 64 and 60 x 44, give 0 differing channels against the encoded original.
    #[cfg(feature = "gpu")]
    #[test]
    fn every_exact_rewrite_draws_the_encoded_picture_on_the_device() {
        use crate::wgsl::cuneus_originals::case_bindings;
        use crate::wgsl::image::{compare, ImageKernel};
        use crate::wgsl::oracle::device::{Device, Fuel};
        let dev = Device::new().expect("device");
        for (shader, build) in [("volumepassage", volumepassage as fn(&TrueKingdom) -> Result<Tree, Vec<String>>), ("nebula", nebula)] {
            let k = TrueKingdom::load(&image_dir(shader)).expect("table");
            let o = Optimisers::new(&k).expect("optimisers");
            let tree = build(&k).unwrap_or_else(|g| panic!("{g:?}"));
            let reg = k.rows.iter().position(|r| r.name == "reg.rf63").expect("rf63") as u32;
            let mut done = false;
            let planted = plant_dead_write(&k, &tree, &o, reg, false, &mut done);
            let cleansed = o.cleanse(&k, &planted).expect("cleanse").expect("dead").0;
            let exact = o.egraph(&k, &tree, true, 3, 2_000_000).expect("e-graph").map_or(tree.clone(), |x| x.0);
            let hoisted = o.hoist(&k, &tree).expect("hoist").map_or(tree.clone(), |x| x.0);
            let all = {
                let a = o.cleanse(&k, &tree).expect("c").map_or(tree.clone(), |x| x.0);
                let b = o.egraph(&k, &a, true, 3, 2_000_000).expect("e").map_or(a.clone(), |x| x.0);
                o.hoist(&k, &b).expect("h").map_or(b.clone(), |x| x.0)
            };
            let mut texts = vec![("encoded", k.render(&tree).expect("render"))];
            for (name, t) in [("cleanse", &cleansed), ("egraph_exact", &exact), ("hoist", &hoisted), ("all three", &all)] {
                assert_typed(&k, t);
                texts.push((name, k.render(t).expect("render")));
            }
            let share = crate::wgsl::true_share::Share::load(&k).expect("share rows");
            texts.push(("share lets", share.render(&k, &share.share(&k, &tree).expect("share").tree).expect("render")));
            let fuel = Fuel::Tree { per_invocation: [&tree, &cleansed, &exact, &hoisted, &all].iter().map(|t| k.tree_work_bound(t).expect("bound")).max().unwrap_or(0) };
            let entry = "main";
            for (w, h) in [(64u32, 64u32), (60, 44)] {
                let bindings = case_bindings(shader, 5.0, w, h).expect("bindings");
                let want = ImageKernel::compile(&dev, "encoded", &texts[0].1, entry, &bindings, w, h, fuel.clone()).expect("encoded").run(0, 1).expect("run").picture;
                for (name, src) in &texts[1..] {
                    let got = ImageKernel::compile(&dev, name, src, entry, &bindings, w, h, fuel.clone()).unwrap_or_else(|e| panic!("{shader} {name}: {e}")).run(0, 1).expect("run").picture;
                    let d = compare(&got, &want).expect("same size");
                    eprintln!("{shader} {w}x{h} {name}: {} differing channels", d.differing_channels);
                    assert_eq!(d.differing_channels, 0, "{shader} {w}x{h} {name}: {d:?}");
                }
            }
        }
    }
}
