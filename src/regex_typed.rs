//! THE REGEX KINGDOM AS ROWS of the gene frame's fixed schema (`docs/DESIGN_geneframe.md` 3.3 and 8.2;
//! `docs/PLAN_futhark_kingdom.md` 7.7, task K2), on the multi-typed GEP core (`nucleotable/docs/MultiTypedGEP.md`,
//! revision 4; `phylu/docs/PLAN_regex_on_typed_gep.md`).
//!
//! The kingdom `regex` of the one set of master tables (`kingdoms/master/`, every table with a `kingdom` column; task
//! K0), written by `kingdoms/regex/gen_tables.py` (never by hand; its `--check` is a test), holds the four kinds of
//! row and nothing else: four data types (`Pattern`, `CharClass`, `Char`, `I`, the last
//! carrying `value`), the POSIX ERE alphabet as Symbol and Argument rows, no Flow row (a regex carries no property),
//! and ONE Guard row: the counted repetition `regex_rep_range(CharClass, I lo, I hi)` is legal only when
//! `lo <= hi` (POSIX rejects `a{5,3}`), `must`, repaired at the node by rendering the bounds exchanged. Every regex
//! gene is valid by construction (a `head` and a `tail` per data type, no substitution); the one repair exchanges two
//! leaves and drops nothing.
//!
//! This module owns what is specific to regex and not to the core:
//!
//! * the kingdom's rows ([`RegexRows::load`]), with or without the literal-Pattern terminal `?_Pat`
//!   ([`PATTERN_LITERAL`]), kept or dropped as a view keeps the rows a problem has;
//! * the constant pools `C_Char`, `C_Pat` and `C_I`, which are data of the kingdom ([`const_len`], [`pool_value`]);
//! * THE DEVICE'S RULES ([`RegexRows::device_rules`]): each Guard row compiled into the one closed form the device
//!   applies (exchange two constant inputs when the first's pool value exceeds the second's), and a row outside that
//!   form is "our software has a bug" at load, never a silent approximation;
//! * the emission of an expressed tree (the walk's level-order nodes) as the level-order [`GpuNode`] array the regex
//!   compiler reads ([`emit_nodes`]), the device's rules applied; the device twin is
//!   `phylu/src/evolve/regex_emit.wgsl`, built from the same tables ([`EmitTables`]);
//! * THE HOST DECODE ([`RegexRows::decode_gene`]): the same gene read through the rows as rows, the guard's condition
//!   evaluated by the formula engine over the leaves' pool values and the repair applied as the rows say; the tests
//!   require it equal to the device's rules, gene for gene.
//!
//! There is no coercion in the compiler's input: a `Char` is a `?_Char` placeholder read from `Dc_Char` and resolved
//! through `C_Char`, an `I` the same through `C_I`, and a literal byte in a `Pattern` slot is `?_Pat` through `C_Pat`;
//! each is emitted as a bare `Num` node whose `konst` is the pool value, the one form the compiler's literal-byte test
//! already accepts.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::geneframe::{master, Entry, KingdomTables, Owner, PropertyExpr, Reference, Relation, Site, Strength, Ty, REGEX};
use crate::gpu_eval::{GpuNode, Op};
use crate::kingdom_contract::{relations_from_tables, KingdomContract, KingdomRelations, SymbolId};
use crate::typed_environment::Decoder;
use crate::typed_gep::{express, GeneLayout, Tree};

/// The kingdom's name in the master tables (`kingdoms/master/kingdoms.tsv`).
pub const KINGDOM: &str = "regex";

/// `C_Char` and `C_Pat`: the printable ASCII bytes `0x20..=0x7e`, in byte order,
/// so neighbouring indices are neighbouring bytes and a point mutation of an
/// index moves a literal to a nearby one. The datasets are ASCII.
pub const CHAR_POOL_LEN: u32 = 95;
/// First byte of the printable pool.
pub const CHAR_POOL_BASE: u32 = 0x20;
/// `C_I`: the repeat counts `0..=8`, the compiler's clamp
/// (`regex_compile::MAX_REPEAT`). The count bounds the instructions a repetition
/// emits (`k` for `rep_n`, `2k` for `rep_upto` and `rep_range`), which [`capacity`] adds to the
/// layout's worst-case program so the compiler's buffers are sized to hold it.
pub const COUNT_POOL_LEN: u32 = 9;
/// The widest pool, the stride of a flat device pool table.
pub const POOL_STRIDE: u32 = CHAR_POOL_LEN;

/// The name of the literal-Pattern terminal the `pattern_literal` switch keeps.
pub const PATTERN_LITERAL: &str = "?_Pat";

/// The counted range `{lo,hi}`, the one guarded symbol.
pub const REP_RANGE: &str = "regex_rep_range";

fn bug(what: String) -> String {
    format!("our software has a bug: {what}")
}

/// The table's data types, in the order the table declares them.
fn data_type(i: usize) -> Result<Ty, String> {
    [Ty::Pattern, Ty::CharClass, Ty::Char, Ty::I].get(i).copied().ok_or_else(|| bug(format!("the regex table declares data type {i}; the kingdom has four")))
}

/// The kingdom's rows as the master renders them (the named text form), without the `?_Pat` rows when the kingdom
/// has no literal Pattern.
fn table_text(pattern_literal: bool) -> Result<String, String> {
    let names_pat = |l: &str| {
        let f: Vec<&str> = l.split('\t').collect();
        f.get(1) == Some(&PATTERN_LITERAL) || (f.first() == Some(&"constant") && f.get(2) == Some(&PATTERN_LITERAL))
    };
    let text = master::select(KINGDOM)?.named_text()?;
    Ok(text.lines().filter(|l| pattern_literal || !names_pat(l)).collect::<Vec<_>>().join("\n"))
}

/// The regex kingdom's relations for the typed core, read from the rows: the alphabet plus the placeholders `?_Char`
/// and `?_I`. With `pattern_literal` the kingdom has `?_Pat`, a constant-capable `Pattern` terminal whose `C_Pat`
/// value is the byte it matches, so a one-byte pattern is a leaf (`lit(?_Char)` costs a function and a `Char` slot).
/// Without it a `Pattern` tail holds only `.`, `^`, `$` and the empty pattern, and every literal costs `lit`.
pub fn typed_regex_relations(pattern_literal: bool) -> KingdomRelations {
    let (tables, entry) = table_text(pattern_literal).and_then(|t| KingdomTables::parse_named(REGEX, &t, &data_type)).unwrap_or_else(|e| panic!("{e}"));
    relations_from_tables(&tables, vec![entry.output])
}

/// ONE SYMBOL'S RULE AS THE DEVICE READS IT, compiled from its Guard rows: exchange inputs `a` and `b` (both leaves of a
/// data type whose only symbols are constants) when the pool value of `a` exceeds the pool value of `b`. `None` for a
/// symbol with no guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRule {
    pub swap: Option<(u32, u32)>,
}

impl DeviceRule {
    /// The rule as one table word: 0 for none, else `(a + 1) << 4 | (b + 1)`.
    pub fn word(self) -> u32 {
        self.swap.map_or(0, |(a, b)| ((a + 1) << 4) | (b + 1))
    }

    /// The rule a table word holds.
    pub fn from_word(w: u32) -> DeviceRule {
        DeviceRule { swap: (w != 0).then(|| ((w >> 4) - 1, (w & 15) - 1)) }
    }
}

/// The loaded regex kingdom: its rows, entry, contract and decoder, built once.
#[derive(Debug, Clone)]
pub struct RegexRows {
    pub pattern_literal: bool,
    pub tables: Arc<KingdomTables>,
    pub entry: Entry,
    pub contract: Arc<KingdomContract>,
    pub decoder: Decoder,
}

impl RegexRows {
    /// Load the rows, with or without the literal-Pattern terminal: the table is read and checked, the contract built
    /// and checked, the decoder built, the entry checked, and every Guard row compiled into the device's form.
    pub fn load(pattern_literal: bool) -> Result<RegexRows, String> {
        RegexRows::from_text(pattern_literal, &table_text(pattern_literal)?)
    }

    /// [`Self::load`] over a table text (the tests' planted faults).
    pub fn from_text(pattern_literal: bool, text: &str) -> Result<RegexRows, String> {
        let (tables, entry) = KingdomTables::parse_named(REGEX, text, &data_type)?;
        tables.check().map_err(|d| d.join("\n"))?;
        let contract = relations_from_tables(&tables, vec![entry.output]).check().map_err(|d| d.join("\n"))?;
        let (tables, contract) = (Arc::new(tables), Arc::new(contract));
        let decoder = Decoder::from_arcs(Arc::clone(&tables), Arc::clone(&contract)).map_err(|d| d.join("\n"))?;
        decoder.check_entry(&entry).map_err(|d| d.join("\n"))?;
        let rows = RegexRows { pattern_literal, tables, entry, contract, decoder };
        rows.device_rules()?;
        Ok(rows)
    }

    /// The symbol named `name`.
    pub fn symbol(&self, name: &str) -> Option<SymbolId> {
        (0..self.contract.n_symbols() as SymbolId).find(|s| self.contract.name(*s) == name)
    }

    /// The gene's layout for a program of `n` functions, rooted at `Pattern`.
    pub fn layout(&self, n: usize) -> Result<GeneLayout, String> {
        GeneLayout::from_program_size(&self.contract, Ty::Pattern, n).map_err(|e| bug(format!("the regex layout at program size {n}: {e:?}")))
    }

    /// THE DEVICE'S RULES, one per symbol id, compiled from the Guard rows: each guard must be a `must` comparison
    /// `in<a>.value <= in<b>.value` over two inputs of a data type whose every symbol is a constant (the device reads
    /// a leaf's pool value), with one unconditional repair at the node whose template is the symbol's default spelling
    /// with `in<a>.value` and `in<b>.value` exchanged.
    ///
    /// # Errors
    /// "our software has a bug" naming the first row outside that form.
    pub fn device_rules(&self) -> Result<Vec<DeviceRule>, String> {
        (0..self.contract.n_symbols() as SymbolId).map(|s| self.device_rule(s).map_err(|e| bug(format!("regex symbol {s} ({}): {e}", self.contract.name(s))))).collect()
    }

    fn device_rule(&self, s: SymbolId) -> Result<DeviceRule, String> {
        let id = i64::from(s);
        let mut rule = DeviceRule { swap: None };
        for g in self.tables.guards.iter().filter(|g| g.owner == Owner::Symbol(id)) {
            if g.strength != Strength::Must {
                return Err(format!("guard {}: the device applies must guards only", g.order));
            }
            let (a, b) = match g.condition.comparison() {
                Some((Relation::Le, PropertyExpr::Input { position: a, property: pa }, PropertyExpr::Input { position: b, property: pb })) if pa == "value" && pb == "value" && a != b => (*a, *b),
                _ => return Err(format!("guard {}: {} is not `in<a>.value <= in<b>.value`", g.order, g.condition)),
            };
            let sig = self.contract.signature(s);
            for k in [a, b] {
                let ty = *sig.get(k as usize).ok_or_else(|| format!("guard {}: input {k}, which the symbol has not", g.order))?;
                let leaves_only = self.contract.functions_of(ty).is_empty() && self.contract.terminals_of(ty).iter().all(|t| self.contract.is_constant(*t));
                if !leaves_only {
                    return Err(format!("guard {}: input {k} is a {ty:?}, whose symbols are not all constants, so the device cannot read its value", g.order));
                }
            }
            let spelling = self.tables.symbol(id).and_then(|row| row.spelling("")).ok_or_else(|| format!("guard {}: the symbol has no default spelling to exchange the bounds of", g.order))?;
            let exchanged = spelling.replace(&format!("in{a}.value"), "\u{1}").replace(&format!("in{b}.value"), &format!("in{a}.value")).replace('\u{1}', &format!("in{b}.value"));
            match g.repairs.as_slice() {
                [r] if r.site == Site::Node && r.unconditional() && r.template == exchanged => {}
                _ => return Err(format!("guard {}: not one unconditional repair at the node spelled as the symbol with in{a}.value and in{b}.value exchanged ({exchanged:?})", g.order)),
            }
            if rule.swap.replace((a, b)).is_some() {
                return Err(format!("guard {}: a second rule on one symbol", g.order));
            }
        }
        Ok(rule)
    }

    /// THE HOST DECODE of one gene through the rows as rows (the twin the device's rules are tested against): Karva's
    /// walk over `layout`, then at every node each `must` Guard row's condition evaluated by the formula engine with
    /// `in<k>.value` read as the leaf's pool value, and where it does not hold, its repair applied (the node's two
    /// bounds exchanged, which is what rendering the repair's template does). The result is the compiler's node array.
    ///
    /// # Errors
    /// A gene that does not fit the layout, or a guard the formula engine cannot decide: our software's bug.
    pub fn decode_gene(&self, layout: &GeneLayout, gene: &[u32], const_len: &BTreeMap<Ty, u32>) -> Result<Vec<GpuNode>, String> {
        let e = express(&self.contract, layout, gene, const_len).map_err(|d| bug(format!("the regex gene does not express: {d}")))?;
        let mut tree = e.tree();
        self.repair(&mut tree)?;
        let tables = EmitTables::new(self).map_err(|d| d.to_string())?;
        emit(&tables, &level_order(&tree), false).map_err(|d| d.to_string())
    }

    /// Pass 4 of the decoder on a tree with no properties: the must guards, read generically, each repair the exchange
    /// of the two inputs its template names.
    fn repair(&self, t: &mut Tree) -> Result<(), String> {
        for k in &mut t.kids {
            self.repair(k)?;
        }
        let id = i64::from(t.symbol);
        let bounds = self.tables.bounds();
        let rules = self.device_rules()?;
        for g in self.tables.guards.iter().filter(|g| g.owner == Owner::Symbol(id) && g.strength == Strength::Must) {
            let read = |r: &Reference| -> Option<PropertyExpr> {
                let Reference::Input { position, property: "value" } = r else { return None };
                let kid = t.kids.get(*position as usize)?;
                let v = pool_value(self.contract.output(kid.symbol), kid.const_index?)?;
                Some(PropertyExpr::Literal(i64::from(v)))
            };
            let condition = g.condition.substitute(&read, &bounds);
            if condition.truth(&bounds).holds() {
                continue;
            }
            let (a, b) = rules[t.symbol as usize].swap.ok_or_else(|| bug(format!("the guard {} fails and its repair is not the exchange of two inputs", g.condition)))?;
            t.kids.swap(a as usize, b as usize);
        }
        Ok(())
    }
}

/// A tree in the walk's level order (a node's inputs follow it, contiguous, in argument order).
pub fn level_order(t: &Tree) -> Vec<TypedNode> {
    let mut out = Vec::new();
    let mut queue = std::collections::VecDeque::from([(t, None, None)]);
    while let Some((n, parent_step, arg_index)) = queue.pop_front() {
        out.push(TypedNode { symbol: n.symbol, parent_step, arg_index, const_index: n.const_index });
        let step = out.len();
        queue.extend(n.kids.iter().enumerate().map(|(a, k)| (k, Some(step), Some(a as u32))));
    }
    out
}

/// `|C_T|` for every constant-capable type of the regex kingdom, the
/// `const_len` map the layouts and the walk take.
pub fn const_len(k: &KingdomContract) -> BTreeMap<Ty, u32> {
    k.types().iter().copied().filter(|t| k.constant_capable(*t)).map(|t| (t, pool_len(t))).collect()
}

/// The length of `C_T`; zero for a type with no pool.
pub fn pool_len(ty: Ty) -> u32 {
    match ty {
        Ty::Char | Ty::Pattern => CHAR_POOL_LEN,
        Ty::I => COUNT_POOL_LEN,
        _ => 0,
    }
}

/// `C_T[index]`, the value a placeholder of type `ty` with Dc word `index`
/// stands for: a byte for `Char` and `Pattern`, a count for `I`.
pub fn pool_value(ty: Ty, index: u32) -> Option<u32> {
    if index >= pool_len(ty) {
        return None;
    }
    Some(match ty {
        Ty::Char | Ty::Pattern => CHAR_POOL_BASE + index,
        _ => index,
    })
}

/// The inverse of [`pool_value`] for a byte: the `C_Char` index of `byte`.
pub fn char_index(byte: u8) -> Option<u32> {
    let b = u32::from(byte);
    (CHAR_POOL_BASE..CHAR_POOL_BASE + CHAR_POOL_LEN).contains(&b).then(|| b - CHAR_POOL_BASE)
}

/// The compiler opcode of a regex symbol name; a constant placeholder is a
/// bare `Num` (the literal-byte or count leaf).
pub fn op_of_name(name: &str) -> Option<Op> {
    Some(match name {
        "regex_dot" => Op::RegexDot,
        "regex_anchor_start" => Op::RegexAnchorStart,
        "regex_anchor_end" => Op::RegexAnchorEnd,
        "regex_empty" => Op::RegexEmpty,
        "regex_cc_digit" => Op::RegexCcDigit,
        "regex_cc_word" => Op::RegexCcWord,
        "regex_cc_space" => Op::RegexCcSpace,
        "regex_star" => Op::RegexStar,
        "regex_plus" => Op::RegexPlus,
        "regex_opt" => Op::RegexOpt,
        "regex_concat" => Op::RegexConcat,
        "regex_alt" => Op::RegexAlt,
        "regex_rep_n" => Op::RegexRepN,
        "regex_rep_upto" => Op::RegexRepUpto,
        REP_RANGE => Op::RegexRepRange,
        "regex_class_of" => Op::RegexClassOf,
        "regex_lit" => Op::RegexLit,
        "regex_cc_range" => Op::RegexCcRange,
        "regex_cc_union" => Op::RegexCcUnion,
        "regex_cc_negate" => Op::RegexCcNegate,
        "?_Char" | "?_I" | PATTERN_LITERAL => Op::Num,
        _ => return None,
    })
}

/// One expressed node as the walk reports it, level order from step 1. The
/// host walk (`typed_gep::Node`) and the device walk (`WalkNode`) both carry
/// these four fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypedNode {
    pub symbol: SymbolId,
    /// The step (from 1) of the parent; `None` for the root.
    pub parent_step: Option<usize>,
    pub arg_index: Option<u32>,
    /// The `C_T` index an expressed placeholder read from its Dc.
    pub const_index: Option<u32>,
}

/// A fault in the kingdom tables or the walk, reported and never repaired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitDefect {
    /// A symbol of the kingdom has no compiler opcode.
    NoOpcode { symbol: SymbolId },
    /// A Guard row outside the device's form.
    Rule(String),
    /// A placeholder was expressed without a Dc index, or past its pool.
    BadConstant { step: usize, index: Option<u32> },
    /// The walk did not report a parent's children as consecutive steps in
    /// argument order, which the level order guarantees.
    ChildrenNotContiguous { step: usize },
    /// The first node is not a parentless root, or a later node is parentless.
    NotARootedTree { step: usize },
    /// A guarded node's bound is not a leaf with a pool value.
    BoundNotALeaf { step: usize },
    Empty,
}

impl fmt::Display for EmitDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmitDefect::Rule(e) => write!(f, "{e}"),
            EmitDefect::NoOpcode { symbol } => write!(f, "our software has a bug: regex symbol {symbol} has no compiler opcode"),
            EmitDefect::BadConstant { step, index } => write!(f, "our software has a bug: node {step} is a placeholder with pool index {index:?}, outside its pool"),
            EmitDefect::ChildrenNotContiguous { step } => write!(f, "our software has a bug: the children of node {step} are not consecutive steps in argument order"),
            EmitDefect::NotARootedTree { step } => write!(f, "our software has a bug: node {step} breaks the rooted level-order tree"),
            EmitDefect::BoundNotALeaf { step } => write!(f, "our software has a bug: node {step} is guarded and a bound of it is not a leaf with a pool value"),
            EmitDefect::Empty => write!(f, "our software has a bug: an expressed gene with no nodes"),
        }
    }
}

/// What the regex compiler's buffers must hold for every gene a layout can
/// express: they are sized from this, never from a fixed constant, so a gene is
/// never dropped for length at any program size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capacity {
    /// The most nodes an expressed gene can have: every function row of every
    /// head (`sum_T h_T`) plus every leaf (`L`).
    pub nodes: usize,
    /// The most VM instructions its program can have, `Match` included.
    pub insts: usize,
}

/// Instructions one `Pattern` function can emit, at most. A class operand and
/// an `I` operand emit none of their own; `rep_upto` and `rep_range` emit `2k`, `rep_n` `k`.
fn pattern_function_cost(name: &str) -> usize {
    let max_count = (COUNT_POOL_LEN - 1) as usize;
    match name {
        "regex_rep_upto" | REP_RANGE => 2 * max_count,
        "regex_rep_n" => max_count,
        "regex_alt" | "regex_star" => 2,
        "regex_concat" => 0,
        _ => 1,
    }
}

/// [`Capacity`] of a gene rooted at `Pattern` with head lengths `heads`: nodes
/// from the node bound, instructions from the `Pattern` frame alone (every
/// instruction is emitted by a `Pattern` node; class, `Char` and `I` nodes emit
/// none of their own).
pub fn capacity(k: &KingdomContract, heads: &BTreeMap<Ty, usize>) -> Result<Capacity, String> {
    let b = k.bounds(Ty::Pattern, heads)?;
    let p = b.get(Ty::Pattern).ok_or_else(|| "our software has a bug: the regex kingdom has no Pattern frame".to_string())?;
    let function_cost = k.functions_of(Ty::Pattern).iter().map(|s| pattern_function_cost(k.name(*s))).max().unwrap_or(1);
    let leaf_cost = 1;
    Ok(Capacity { nodes: heads.values().sum::<usize>() + b.l, insts: p.h * function_cost + p.t * leaf_cost + 1 })
}

/// The per-symbol tables of emission, the data both the host [`emit_nodes`]
/// and the device kernel read: indexed by `SymbolId`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitTables {
    /// `Op as u32` of each symbol.
    pub sym_op: Vec<u32>,
    /// `Ty::code()` of each symbol's output type.
    pub sym_ty_code: Vec<u32>,
    /// `1` for a constant placeholder, else `0`.
    pub sym_const: Vec<u32>,
    /// The pool of each symbol's type: `pool_len` (0 for no pool).
    pub sym_pool_len: Vec<u32>,
    /// Each symbol's [`DeviceRule`] as its table word (0 for none).
    pub sym_rule: Vec<u32>,
    /// `pool_value(ty, i)` for `i` in `0..POOL_STRIDE`, one row per symbol
    /// (zero where the symbol has no pool); `sym * POOL_STRIDE + i`.
    pub pool: Vec<u32>,
}

impl EmitTables {
    pub fn new(rows: &RegexRows) -> Result<EmitTables, EmitDefect> {
        let k = &rows.contract;
        let rules = rows.device_rules().map_err(EmitDefect::Rule)?;
        let n = k.n_symbols();
        let mut t = EmitTables { sym_op: vec![0; n], sym_ty_code: vec![0; n], sym_const: vec![0; n], sym_pool_len: vec![0; n], sym_rule: vec![0; n], pool: vec![0; n * POOL_STRIDE as usize] };
        for s in 0..n as SymbolId {
            let op = op_of_name(k.name(s)).ok_or(EmitDefect::NoOpcode { symbol: s })?;
            let ty = k.output(s);
            let i = s as usize;
            t.sym_op[i] = op as u32;
            t.sym_ty_code[i] = ty.code();
            t.sym_rule[i] = rules[i].word();
            if k.is_constant(s) {
                t.sym_const[i] = 1;
                t.sym_pool_len[i] = pool_len(ty);
                for j in 0..pool_len(ty) {
                    t.pool[i * POOL_STRIDE as usize + j as usize] = pool_value(ty, j).unwrap_or(0);
                }
            }
        }
        Ok(t)
    }
}

/// The expressed tree as the level-order [`GpuNode`] array the regex compiler
/// reads: `op`, the first child's index `arg0`, the node's output `ty_code`,
/// and for a placeholder the pool value as `konst`, with the device's rules applied (a guarded node's two bounds
/// exchanged when the first exceeds the second). The node order is the
/// walk's, so the result is level order and a node's children are contiguous.
pub fn emit_nodes(t: &EmitTables, nodes: &[TypedNode]) -> Result<Vec<GpuNode>, EmitDefect> {
    emit(t, nodes, true)
}

fn emit(t: &EmitTables, nodes: &[TypedNode], rules: bool) -> Result<Vec<GpuNode>, EmitDefect> {
    if nodes.is_empty() {
        return Err(EmitDefect::Empty);
    }
    let mut out: Vec<GpuNode> = Vec::with_capacity(nodes.len());
    let mut kids: Vec<usize> = vec![0; nodes.len()];
    for (i, n) in nodes.iter().enumerate() {
        let step = i + 1;
        match n.parent_step {
            None if i != 0 => return Err(EmitDefect::NotARootedTree { step }),
            Some(p) if i == 0 || p == 0 || p >= step => return Err(EmitDefect::NotARootedTree { step }),
            _ => {}
        }
        if let (Some(p), Some(a)) = (n.parent_step, n.arg_index) {
            // The a-th child of node p sits right after its (a-1)-th.
            let want = if a == 0 { None } else { Some(i - 1) };
            let ok = match want {
                None => kids[p - 1] == 0,
                Some(prev) => nodes[prev].parent_step == Some(p) && nodes[prev].arg_index == Some(a - 1),
            };
            if !ok {
                return Err(EmitDefect::ChildrenNotContiguous { step: p });
            }
            if a == 0 {
                kids[p - 1] = i;
            }
        }
        let s = n.symbol as usize;
        let konst = if t.sym_const[s] == 1 {
            match n.const_index {
                Some(ix) if ix < t.sym_pool_len[s] => t.pool[s * POOL_STRIDE as usize + ix as usize] as f32,
                other => return Err(EmitDefect::BadConstant { step, index: other }),
            }
        } else {
            0.0
        };
        out.push(GpuNode { op: t.sym_op[s], arg0: 0, ty_code: t.sym_ty_code[s], konst });
    }
    for (node, first) in out.iter_mut().zip(&kids) {
        node.arg0 = *first as u32;
    }
    if rules {
        // THE DEVICE'S RULE: the two bounds of a guarded node exchanged when the first exceeds the second (the twin
        // of `regex_emit.wgsl`, which it is checked against word for word).
        for i in 0..nodes.len() {
            let Some((a, b)) = DeviceRule::from_word(t.sym_rule[nodes[i].symbol as usize]).swap else { continue };
            let (ca, cb) = (kids[i] + a as usize, kids[i] + b as usize);
            let leaf = |c: usize| c < out.len() && c > i && t.sym_const[nodes[c].symbol as usize] == 1;
            if !(leaf(ca) && leaf(cb)) {
                return Err(EmitDefect::BoundNotALeaf { step: i + 1 });
            }
            if out[ca].konst > out[cb].konst {
                let (ka, kb) = (out[ca].konst, out[cb].konst);
                out[ca].konst = kb;
                out[cb].konst = ka;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
