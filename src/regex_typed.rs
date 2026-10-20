//! The regex kingdom on the multi-typed GEP core (`nucleotable/docs/MultiTypedGEP.md`,
//! revision 4; `phylu/docs/PLAN_regex_on_typed_gep.md`).
//!
//! Every regex gene is valid by construction: a `head` and a `tail` per output
//! type (`Pattern`, `CharClass`, `Char`, `I`), no repair. This module owns what
//! is specific to regex and not to the core:
//!
//! * the kingdom's relations, [`typed_regex_relations`], which are
//!   [`crate::kingdom_contract::regex_relations`] plus an optional literal
//!   Pattern terminal `?_Pat`;
//! * the constant pools `C_Char`, `C_Pat` and `C_I`, which are data of the
//!   kingdom ([`const_len`], [`pool_value`]);
//! * the emission of an expressed tree (the walk's level-order nodes) as the
//!   level-order [`GpuNode`] array the regex compiler reads ([`emit_nodes`]);
//!   the device twin is `phylu/src/evolve/regex_emit.wgsl`, built from the
//!   same tables ([`EmitTables`]).
//!
//! There is no coercion in the compiler's input: a `Char` is a `?_Char`
//! placeholder read from `Dc_Char` and resolved through `C_Char`, an `I` the
//! same through `C_I`, and a literal byte in a `Pattern` slot is `?_Pat`
//! through `C_Pat`; each is emitted as a bare `Num` node whose `konst` is the
//! pool value, the one form the compiler's literal-byte test already accepts.

use std::collections::BTreeMap;
use std::fmt;

use crate::geneframe::{regex_rows, Ty, REGEX};
use crate::gpu_eval::{GpuNode, Op};
use crate::kingdom_contract::{relations_from_rows, KingdomContract, KingdomRelations, SymbolId};

/// `C_Char` and `C_Pat`: the printable ASCII bytes `0x20..=0x7e`, in byte order,
/// so neighbouring indices are neighbouring bytes and a point mutation of an
/// index moves a literal to a nearby one. The datasets are ASCII.
pub const CHAR_POOL_LEN: u32 = 95;
/// First byte of the printable pool.
pub const CHAR_POOL_BASE: u32 = 0x20;
/// `C_I`: the repeat counts `0..=4`. The count bounds the instructions a
/// repetition emits (`k` for `rep_n`, `2k` for `rep_upto`), and the layout's
/// worst-case program must fit the compiler's `MAX_INST` for every gene it can
/// hold ([`max_insts`]); counts to 4 cover the formats the datasets have
/// (`\d{1,3}`, `[A-Z]{2}`) and keep N = 12 inside the budget.
pub const COUNT_POOL_LEN: u32 = 5;
/// The widest pool, the stride of a flat device pool table.
pub const POOL_STRIDE: u32 = CHAR_POOL_LEN;

/// The name of the literal-Pattern terminal added by [`typed_regex_relations`].
pub const PATTERN_LITERAL: &str = "?_Pat";

/// The typed core's regex rows: [`regex_rows`] with one change. A counted
/// repetition takes a `CharClass`, `rep_n(CharClass, I)` and
/// `rep_upto(CharClass, I)`, not a `Pattern`. Expanding a repetition of an
/// arbitrary sub-pattern multiplies the instruction count with nesting, which no
/// fixed program buffer holds for every gene a layout can express, and a gene
/// that overflowed would have to be dropped. A class folds to one 256-bit set, so
/// a repetition of a class emits at most `2k` instructions however large the
/// class tree is: the bound is a property of the types, and it covers the
/// formats the kingdom is for (`\d{3}`, `[A-Z]{1,2}`). A repeated group is
/// written with `star`, `plus` and `opt`, which loop rather than copy.
pub fn typed_regex_rows() -> Vec<(String, Vec<Ty>, Ty)> {
    regex_rows()
        .iter()
        .map(|(sem, _, ins, out)| {
            let ins = match *sem {
                "regex_rep_n" | "regex_rep_upto" => vec![Ty::CharClass, Ty::I],
                _ => ins.to_vec(),
            };
            (sem.to_string(), ins, *out)
        })
        .collect()
}

/// The regex kingdom's relations for the typed core: [`typed_regex_rows`] plus
/// the placeholders `?_Char` and `?_I`. With `pattern_literal` the kingdom has
/// `?_Pat`, a constant-capable `Pattern` terminal whose `C_Pat` value is the
/// byte it matches, so a one-byte pattern is a leaf (`lit(?_Char)` costs a
/// function and a `Char` slot). Without it a `Pattern` tail holds only `.`, `^`,
/// `$` and the empty pattern, and every literal costs `lit`.
pub fn typed_regex_relations(pattern_literal: bool) -> KingdomRelations {
    let mut rows = typed_regex_rows();
    rows.push(("?_Char".to_string(), Vec::new(), Ty::Char));
    rows.push(("?_I".to_string(), Vec::new(), Ty::I));
    let mut constants = vec!["?_Char", "?_I"];
    if pattern_literal {
        rows.push((PATTERN_LITERAL.to_string(), Vec::new(), Ty::Pattern));
        constants.push(PATTERN_LITERAL);
    }
    relations_from_rows(REGEX, vec![Ty::Pattern], &rows, &constants)
}

/// `|C_T|` for every constant-capable type of the regex kingdom, the
/// `const_len` map the layouts and the walk take.
pub fn const_len(k: &KingdomContract) -> BTreeMap<Ty, u32> {
    k.types()
        .iter()
        .copied()
        .filter(|t| k.constant_capable(*t))
        .map(|t| (t, pool_len(t)))
        .collect()
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
    /// A placeholder was expressed without a Dc index, or past its pool.
    BadConstant { step: usize, index: Option<u32> },
    /// The walk did not report a parent's children as consecutive steps in
    /// argument order, which the level order guarantees.
    ChildrenNotContiguous { step: usize },
    /// The first node is not a parentless root, or a later node is parentless.
    NotARootedTree { step: usize },
    Empty,
}

impl fmt::Display for EmitDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "our software has a bug: ")?;
        match self {
            EmitDefect::NoOpcode { symbol } => write!(f, "regex symbol {symbol} has no compiler opcode"),
            EmitDefect::BadConstant { step, index } => write!(f, "node {step} is a placeholder with pool index {index:?}, outside its pool"),
            EmitDefect::ChildrenNotContiguous { step } => write!(f, "the children of node {step} are not consecutive steps in argument order"),
            EmitDefect::NotARootedTree { step } => write!(f, "node {step} breaks the rooted level-order tree"),
            EmitDefect::Empty => write!(f, "an expressed gene with no nodes"),
        }
    }
}

/// What the regex compiler must hold for every gene a layout can express.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capacity {
    /// The most nodes an expressed gene can have: every function row of every
    /// head (`sum_T h_T`) plus every leaf (`L`).
    pub nodes: usize,
    /// The most VM instructions its program can have, `Match` included.
    pub insts: usize,
}

/// The compiler's limits: `MAX_NODES` nodes and `MAX_INST = 2 * MAX_NODES + 1`
/// instructions (`regex_compile.wgsl`).
pub const MAX_INST: usize = 2 * crate::gpu_eval::MAX_NODES + 1;

/// Instructions one `Pattern` function can emit, at most. A class operand and
/// an `I` operand emit none of their own; `rep_upto` emits `2k`, `rep_n` `k`.
fn pattern_function_cost(name: &str) -> usize {
    let max_count = (COUNT_POOL_LEN - 1) as usize;
    match name {
        "regex_rep_upto" => 2 * max_count,
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

/// `Err` when a layout of these heads can express a gene the compiler does not
/// hold. A configuration check made once at setup: no gene is dropped for
/// length at run time because none can exceed the limits.
pub fn check_capacity(k: &KingdomContract, heads: &BTreeMap<Ty, usize>) -> Result<(), String> {
    let c = capacity(k, heads)?;
    let (node_cap, inst_cap) = (crate::gpu_eval::MAX_NODES, MAX_INST);
    if c.nodes > node_cap || c.insts > inst_cap {
        return Err(format!(
            "our software has a bug: this layout (heads guarantee a program size of {}) can express {} nodes and {} instructions, the regex compiler holds {node_cap} and {inst_cap}; the largest program size that fits is {}",
            k.guaranteed_program_size(Ty::Pattern, heads),
            c.nodes,
            c.insts,
            largest_program_size(k)
        ));
    }
    Ok(())
}

/// The largest program size `N` whose heads fit the compiler's limits
/// ([`check_capacity`]).
pub fn largest_program_size(k: &KingdomContract) -> usize {
    let fits = |n: usize| {
        k.heads_from_program_size(Ty::Pattern, n)
            .and_then(|h| capacity(k, &h))
            .is_ok_and(|c| c.nodes <= crate::gpu_eval::MAX_NODES && c.insts <= MAX_INST)
    };
    (1..).take_while(|n| fits(*n)).last().unwrap_or(0)
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
    /// `pool_value(ty, i)` for `i` in `0..POOL_STRIDE`, one row per symbol
    /// (zero where the symbol has no pool); `sym * POOL_STRIDE + i`.
    pub pool: Vec<u32>,
}

impl EmitTables {
    pub fn new(k: &KingdomContract) -> Result<EmitTables, EmitDefect> {
        let n = k.n_symbols();
        let mut t = EmitTables { sym_op: vec![0; n], sym_ty_code: vec![0; n], sym_const: vec![0; n], sym_pool_len: vec![0; n], pool: vec![0; n * POOL_STRIDE as usize] };
        for s in 0..n as SymbolId {
            let op = op_of_name(k.name(s)).ok_or(EmitDefect::NoOpcode { symbol: s })?;
            let ty = k.output(s);
            let i = s as usize;
            t.sym_op[i] = op as u32;
            t.sym_ty_code[i] = ty.code();
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
/// and for a placeholder the pool value as `konst`. The node order is the
/// walk's, so the result is level order and a node's children are contiguous.
pub fn emit_nodes(t: &EmitTables, nodes: &[TypedNode]) -> Result<Vec<GpuNode>, EmitDefect> {
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
    for (node, first) in out.iter_mut().zip(kids) {
        node.arg0 = first as u32;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_gep::{express, Domain, GeneLayout};

    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) % n as u64) as usize
        }
    }

    fn kingdom(lit: bool) -> KingdomContract {
        typed_regex_relations(lit).check().expect("the typed regex kingdom passes the check")
    }

    fn random_gene(k: &KingdomContract, layout: &GeneLayout, rng: &mut Rng) -> Vec<u32> {
        let cl = const_len(k);
        let mut gene = vec![0u32; layout.width];
        for s in layout.spans() {
            for cell in &mut gene[s.off..s.off + s.len] {
                *cell = match s.domain {
                    Domain::Head => {
                        let a: Vec<SymbolId> = (0..k.n_symbols() as SymbolId).filter(|x| k.output(*x) == s.ty).collect();
                        a[rng.below(a.len())]
                    }
                    Domain::Tail => {
                        let ts = k.terminals_of(s.ty);
                        ts[rng.below(ts.len())]
                    }
                    Domain::Dc => rng.below(cl[&s.ty] as usize) as u32,
                };
            }
        }
        gene
    }

    #[test]
    fn both_variants_pass_the_kingdom_check() {
        for lit in [false, true] {
            let k = kingdom(lit);
            assert_eq!(k.roots(), [Ty::Pattern]);
            assert_eq!(k.constant_capable(Ty::Pattern), lit);
            for t in [Ty::Char, Ty::I] {
                assert!(k.constant_capable(t), "{t:?}");
            }
            let cl = const_len(&k);
            assert_eq!(cl.get(&Ty::Char), Some(&CHAR_POOL_LEN));
            assert_eq!(cl.get(&Ty::I), Some(&COUNT_POOL_LEN));
            assert_eq!(cl.contains_key(&Ty::Pattern), lit);
            EmitTables::new(&k).expect("every symbol has an opcode");
        }
    }

    /// The kingdom check figures at the program sizes the plan names; printed
    /// for the plan and asserted on the closed-form heads.
    /// The compiler holds 128 nodes and 257 instructions (`gpu_eval::MAX_NODES`,
    /// `MAX_INST`), which fixes the largest regex layout at N = 12: 124 nodes and
    /// 209 instructions. N = 13 gives 135 nodes.
    #[test]
    fn the_compiler_limits_fix_the_program_size() {
        for lit in [false, true] {
            let k = kingdom(lit);
            let cap = |n: usize| capacity(&k, &k.heads_from_program_size(Ty::Pattern, n).expect("heads")).expect("capacity");
            assert_eq!(cap(12), Capacity { nodes: 124, insts: 209 });
            assert_eq!(cap(13), Capacity { nodes: 135, insts: 227 });
            assert_eq!(largest_program_size(&k), 12);
            assert!(check_capacity(&k, &k.heads_from_program_size(Ty::Pattern, 12).expect("heads")).is_ok());
            let e = check_capacity(&k, &k.heads_from_program_size(Ty::Pattern, 13).expect("heads")).expect_err("13 does not fit");
            assert!(e.starts_with("our software has a bug: ") && e.contains("largest program size that fits is 12"), "{e}");
        }
    }

    /// A repetition takes a class, so its cost is bounded by its count.
    #[test]
    fn a_repetition_takes_a_class() {
        let k = kingdom(true);
        for name in ["regex_rep_n", "regex_rep_upto"] {
            let s = (0..k.n_symbols() as SymbolId).find(|s| k.name(*s) == name).expect(name);
            assert_eq!(k.signature(s), [Ty::CharClass, Ty::I], "{name}");
        }
    }

    #[test]
    fn regex_bound_table() {
        for lit in [false, true] {
            let k = kingdom(lit);
            for n in [8usize, 16, 28, 40] {
                let h = k.heads_from_program_size(Ty::Pattern, n).expect("heads");
                let b = k.bounds(Ty::Pattern, &h).expect("bounds");
                // a_Pattern = 2 (concat), a_CharClass = 2 (union), a_Char = 2 (range), a_I = 1.
                assert_eq!((h[&Ty::Pattern], h[&Ty::CharClass], h[&Ty::Char], h[&Ty::I]), (2 * n - 1, 2 * (n - 1), 2 * (n - 1), n - 1));
                assert_eq!(k.guaranteed_program_size(Ty::Pattern, &h), n);
                let mut line = format!("pattern_literal={lit} N={n} L={} width={}", b.l, b.width());
                for t in &b.types {
                    line.push_str(&format!(" | {:?} h{} P{} t{} d{}{}", t.ty, t.h, t.p, t.t, t.d, if t.constant { "" } else { "-" }));
                }
                println!("{line}");
            }
        }
    }

    /// Every random gene expresses, and emission gives a level-order tree whose
    /// ops, arities, types and constants agree with the kingdom, node for node.
    #[test]
    fn emission_matches_the_kingdom_on_random_genes() {
        let mut rng = Rng(7013);
        for lit in [false, true] {
            let k = kingdom(lit);
            let t = EmitTables::new(&k).expect("tables");
            let cl = const_len(&k);
            for n in [1usize, 3, 8, 12] {
                let layout = GeneLayout::from_program_size(&k, Ty::Pattern, n).expect("layout");
                for _ in 0..2000 {
                    let gene = random_gene(&k, &layout, &mut rng);
                    let e = express(&k, &layout, &gene, &cl).expect("a valid-by-construction gene expresses");
                    let cap = capacity(&k, &k.heads_from_program_size(Ty::Pattern, n).expect("heads")).expect("capacity");
                    assert!(e.nodes.len() <= cap.nodes, "a layout of size {n} has at most {} nodes", cap.nodes);
                    let typed: Vec<TypedNode> = e.nodes.iter().map(|x| TypedNode { symbol: x.symbol, parent_step: x.parent_step, arg_index: x.arg_index, const_index: x.const_index }).collect();
                    let g = emit_nodes(&t, &typed).expect("emits");
                    assert_eq!(g.len(), e.nodes.len());
                    assert_eq!(g[0].ty_code, Ty::Pattern.code());
                    for (i, (node, x)) in g.iter().zip(&e.nodes).enumerate() {
                        let arity = k.signature(x.symbol).len();
                        assert_eq!(node.op, op_of_name(k.name(x.symbol)).map(|o| o as u32).expect("op"));
                        assert_eq!(node.ty_code, x.ty.code());
                        for a in 0..arity {
                            let c = node.child(a);
                            assert!(c > i && c < g.len(), "child {a} of node {i} is forward and in range");
                            assert_eq!(e.nodes[c].parent_step, Some(i + 1));
                            assert_eq!(e.nodes[c].arg_index, Some(a as u32));
                            assert_eq!(e.nodes[c].ty, k.signature(x.symbol)[a], "the child fills the slot's type");
                        }
                        if k.is_constant(x.symbol) {
                            assert_eq!(node.konst as u32, pool_value(x.ty, x.const_index.expect("index")).expect("in pool"));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_broken_walk_is_a_defect_not_a_repair() {
        let k = kingdom(true);
        let t = EmitTables::new(&k).expect("tables");
        let concat = (0..k.n_symbols() as SymbolId).find(|s| k.name(*s) == "regex_concat").expect("concat");
        let dot = (0..k.n_symbols() as SymbolId).find(|s| k.name(*s) == "regex_dot").expect("dot");
        let pat = (0..k.n_symbols() as SymbolId).find(|s| k.name(*s) == PATTERN_LITERAL).expect("?_Pat");
        let n = |symbol, parent_step, arg_index, const_index| TypedNode { symbol, parent_step, arg_index, const_index };
        assert!(matches!(emit_nodes(&t, &[]), Err(EmitDefect::Empty)));
        // A placeholder with no pool index.
        assert!(matches!(emit_nodes(&t, &[n(pat, None, None, None)]), Err(EmitDefect::BadConstant { .. })));
        // A placeholder index past the pool.
        assert!(matches!(emit_nodes(&t, &[n(pat, None, None, Some(CHAR_POOL_LEN))]), Err(EmitDefect::BadConstant { .. })));
        // concat(dot, dot) with the second child skipped to argument 2.
        let bad = [n(concat, None, None, None), n(dot, Some(1), Some(0), None), n(dot, Some(1), Some(2), None)];
        assert!(matches!(emit_nodes(&t, &bad), Err(EmitDefect::ChildrenNotContiguous { .. })));
        // A second root.
        assert!(matches!(emit_nodes(&t, &[n(dot, None, None, None), n(dot, None, None, None)]), Err(EmitDefect::NotARootedTree { .. })));
        let ok = [n(concat, None, None, None), n(dot, Some(1), Some(0), None), n(pat, Some(1), Some(1), Some(char_index(b'A').expect("A")))];
        let g = emit_nodes(&t, &ok).expect("emits");
        assert_eq!((g[0].op, g[0].arg0, g[1].op, g[2].op, g[2].konst as u32), (Op::RegexConcat as u32, 1, Op::RegexDot as u32, Op::Num as u32, 65));
    }
}
