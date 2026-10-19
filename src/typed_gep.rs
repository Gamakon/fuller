//! The multi-typed GEP genome layout and its host expression
//! (`nucleotable/docs/MultiTypedGEP.md` sections 2 to 4).
//!
//! A gene is one head followed by one tail for each type that has a frame,
//! followed by one Dc for each constant-capable frame type. A slot of type `T`
//! reads the next head row whose symbol outputs `T`, then the next row of
//! `T`'s tail. A tail holds terminals of its type only, so a slot cannot read a
//! symbol of another type. There is no repair, substitution or fallback: a
//! cell that does not fit is a [`LayoutDefect`], "our software has a bug: ...".
//!
//! [`express`] is the host reference. It is the test reference the GPU decoder
//! must match bit for bit, not a search path.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;

use crate::geneframe::Ty;
use crate::kingdom_contract::{KingdomContract, SymbolId};

/// The three domains of a gene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Domain {
    Head,
    Tail,
    Dc,
}

impl Domain {
    pub fn name(self) -> &'static str {
        match self {
            Domain::Head => "head",
            Domain::Tail => "tail",
            Domain::Dc => "dc",
        }
    }
}

/// The domain-local part of a row key
/// `(chromosome_id, gene_id, domain, domain_type, domain_seq_id)`: the type is
/// `None` for the head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainKey {
    pub domain: Domain,
    pub domain_type: Option<Ty>,
    pub domain_seq_id: usize,
}

/// One type's tail and Dc inside a gene.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub ty: Ty,
    pub tail_off: usize,
    pub tail_len: usize,
    /// Present for a constant-capable type; as long as the tail.
    pub dc_off: Option<usize>,
}

/// Where everything is in a gene of head length `h` rooted at `root`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneLayout {
    pub h: usize,
    pub root: Ty,
    pub frames: Vec<Frame>,
    pub width: usize,
}

impl GeneLayout {
    /// Lay a gene out from the checked kingdom: head, the tail of each frame
    /// type (root first, then ascending code), then the Dc of each
    /// constant-capable frame type in the same order.
    pub fn new(k: &KingdomContract, h: usize, root: Ty) -> Result<GeneLayout, LayoutDefect> {
        if !k.roots().contains(&root) {
            return Err(LayoutDefect::RootNotInKingdom { root });
        }
        let types = k.frame_types(root);
        let mut off = h;
        let mut frames = Vec::new();
        for ty in &types {
            let tail_len = k.tail_len(h, *ty, root);
            frames.push(Frame { ty: *ty, tail_off: off, tail_len, dc_off: None });
            off += tail_len;
        }
        for f in &mut frames {
            if k.constant_capable(f.ty) {
                f.dc_off = Some(off);
                off += f.tail_len;
            }
        }
        Ok(GeneLayout { h, root, frames, width: off })
    }

    pub fn frame(&self, ty: Ty) -> Option<&Frame> {
        self.frames.iter().find(|f| f.ty == ty)
    }

    /// The domain-local key of the cell at `pos`.
    pub fn key(&self, pos: usize) -> DomainKey {
        if pos < self.h {
            return DomainKey { domain: Domain::Head, domain_type: None, domain_seq_id: pos };
        }
        for f in &self.frames {
            if pos >= f.tail_off && pos < f.tail_off + f.tail_len {
                return DomainKey { domain: Domain::Tail, domain_type: Some(f.ty), domain_seq_id: pos - f.tail_off };
            }
            if let Some(d) = f.dc_off {
                if pos >= d && pos < d + f.tail_len {
                    return DomainKey { domain: Domain::Dc, domain_type: Some(f.ty), domain_seq_id: pos - d };
                }
            }
        }
        unreachable!("position {pos} is outside a gene of width {}", self.width)
    }

    /// The gene as rows of the geneframe relation: head and tail rows carry a
    /// `symbol`, Dc rows a `dc_index` and never a symbol.
    pub fn rows(&self, chromosome_id: u64, gene_id: u64, gene: &[u32]) -> Vec<GeneRow> {
        (0..self.width)
            .map(|pos| {
                let key = self.key(pos);
                let (symbol, dc_index) = if key.domain == Domain::Dc { (None, Some(gene[pos])) } else { (Some(gene[pos]), None) };
                GeneRow { chromosome_id, gene_id, key, symbol, dc_index }
            })
            .collect()
    }
}

/// One row of the geneframe relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneRow {
    pub chromosome_id: u64,
    pub gene_id: u64,
    pub key: DomainKey,
    pub symbol: Option<SymbolId>,
    pub dc_index: Option<u32>,
}

/// A cell the layout or the kingdom cannot express. Every one is a software
/// bug to report; none is repaired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutDefect {
    RootNotInKingdom { root: Ty },
    GeneLength { want: usize, got: usize },
    UnknownSymbol { pos: usize, symbol: u32 },
    /// A slot of `ty` found its frame (head rows then tail) exhausted.
    TailRanOut { ty: Ty, cursor: usize },
    /// A tail cell read for a slot of `ty` holds a symbol of another type.
    ForeignSymbol { ty: Ty, symbol: SymbolId, pos: usize },
    /// A tail cell read holds a function; a tail holds terminals of its type only.
    FunctionInTail { ty: Ty, symbol: SymbolId, pos: usize },
    /// A slot asks for a type with no frame.
    NoFrame { ty: Ty },
    /// More expressed placeholders of `ty` than its Dc holds.
    DcRanOut { ty: Ty, q: usize },
    /// A Dc word names a constant past the end of `C_T`.
    DcIndexPastConstants { ty: Ty, index: u32, len: u32 },
}

impl fmt::Display for LayoutDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "our software has a bug: ")?;
        match self {
            LayoutDefect::RootNotInKingdom { root } => write!(f, "root type {root:?} is not a root of this kingdom"),
            LayoutDefect::GeneLength { want, got } => write!(f, "a gene has {got} words, its layout has {want}"),
            LayoutDefect::UnknownSymbol { pos, symbol } => write!(f, "cell {pos} holds symbol {symbol}, which is not in the kingdom"),
            LayoutDefect::TailRanOut { ty, cursor } => write!(f, "the frame of type {ty:?} ran out at cursor {cursor}"),
            LayoutDefect::ForeignSymbol { ty, symbol, pos } => write!(f, "cell {pos} read for a slot of type {ty:?} holds symbol {symbol} of another type"),
            LayoutDefect::FunctionInTail { ty, pos, .. } => write!(f, "a function in tail {ty:?} at position {pos}"),
            LayoutDefect::NoFrame { ty } => write!(f, "a slot asks for type {ty:?}, which has no frame"),
            LayoutDefect::DcRanOut { ty, q } => write!(f, "the Dc of type {ty:?} ran out at placeholder {q}"),
            LayoutDefect::DcIndexPastConstants { ty, index, len } => write!(f, "a Dc word of type {ty:?} is {index}, C_T has {len} constants"),
        }
    }
}

/// One expressed node, in level order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    /// Level-order number from 1.
    pub step: usize,
    /// The type of the slot this node fills, which is its symbol's output.
    pub ty: Ty,
    /// The slot's rank within its type's frame (head rows, then the tail).
    pub frame_k: usize,
    pub symbol: SymbolId,
    pub depth: usize,
    pub parent_step: Option<usize>,
    pub arg_index: Option<u32>,
    /// The `C_T` index an expressed `?_T` read from its Dc.
    pub const_index: Option<u32>,
    /// The gene position the symbol was read from.
    pub pos: usize,
}

/// How far into each tail and Dc a walk went: the measured side of the bound.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reach {
    /// Tail cells read, per type.
    pub tail_used: BTreeMap<Ty, usize>,
    /// Expressed placeholders, per type.
    pub q_used: BTreeMap<Ty, usize>,
}

/// The expressed tree and how far its walk reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expressed {
    pub nodes: Vec<Node>,
    pub reach: Reach,
}

struct Slot {
    ty: Ty,
    depth: usize,
    parent_step: Option<usize>,
    arg_index: Option<u32>,
}

/// Express a gene: the FIFO walk of the design, section 3. `const_len` is the
/// length of each type's constant array `C_T`.
pub fn express(k: &KingdomContract, layout: &GeneLayout, gene: &[u32], const_len: &BTreeMap<Ty, u32>) -> Result<Expressed, LayoutDefect> {
    if gene.len() != layout.width {
        return Err(LayoutDefect::GeneLength { want: layout.width, got: gene.len() });
    }
    // Each type's head rows, in head order.
    let mut head_rows: BTreeMap<Ty, Vec<usize>> = BTreeMap::new();
    for (pos, w) in gene[..layout.h].iter().enumerate() {
        if *w as usize >= k.n_symbols() {
            return Err(LayoutDefect::UnknownSymbol { pos, symbol: *w });
        }
        head_rows.entry(k.output(*w)).or_default().push(pos);
    }
    let mut k_t: BTreeMap<Ty, usize> = BTreeMap::new();
    let mut q_t: BTreeMap<Ty, usize> = BTreeMap::new();
    let mut reach = Reach::default();
    let mut nodes: Vec<Node> = Vec::new();
    let mut queue = VecDeque::from([Slot { ty: layout.root, depth: 0, parent_step: None, arg_index: None }]);
    while let Some(slot) = queue.pop_front() {
        let frame = layout.frame(slot.ty).ok_or(LayoutDefect::NoFrame { ty: slot.ty })?;
        let heads = head_rows.get(&slot.ty).map_or(&[][..], Vec::as_slice);
        let cursor = k_t.entry(slot.ty).or_insert(0);
        let frame_k = *cursor;
        *cursor += 1;
        let pos = if frame_k < heads.len() {
            heads[frame_k]
        } else {
            let t = frame_k - heads.len();
            if t >= frame.tail_len {
                return Err(LayoutDefect::TailRanOut { ty: slot.ty, cursor: frame_k });
            }
            let used = reach.tail_used.entry(slot.ty).or_default();
            *used = (*used).max(t + 1);
            frame.tail_off + t
        };
        let symbol = gene[pos];
        if symbol as usize >= k.n_symbols() {
            return Err(LayoutDefect::UnknownSymbol { pos, symbol });
        }
        if k.output(symbol) != slot.ty {
            return Err(LayoutDefect::ForeignSymbol { ty: slot.ty, symbol, pos });
        }
        if pos >= frame.tail_off && !k.is_terminal(symbol) {
            return Err(LayoutDefect::FunctionInTail { ty: slot.ty, symbol, pos });
        }
        let step = nodes.len() + 1;
        let const_index = if k.is_constant(symbol) {
            let q = *q_t.entry(slot.ty).or_default();
            let dc_off = frame.dc_off.ok_or(LayoutDefect::DcRanOut { ty: slot.ty, q })?;
            if q >= frame.tail_len {
                return Err(LayoutDefect::DcRanOut { ty: slot.ty, q });
            }
            let index = gene[dc_off + q];
            let len = const_len.get(&slot.ty).copied().unwrap_or(0);
            if index >= len {
                return Err(LayoutDefect::DcIndexPastConstants { ty: slot.ty, index, len });
            }
            q_t.insert(slot.ty, q + 1);
            reach.q_used.insert(slot.ty, q + 1);
            Some(index)
        } else {
            None
        };
        nodes.push(Node {
            step,
            ty: slot.ty,
            frame_k,
            symbol,
            depth: slot.depth,
            parent_step: slot.parent_step,
            arg_index: slot.arg_index,
            const_index,
            pos,
        });
        for (i, ty) in k.signature(symbol).iter().enumerate() {
            queue.push_back(Slot { ty: *ty, depth: slot.depth + 1, parent_step: Some(step), arg_index: Some(i as u32) });
        }
    }
    Ok(Expressed { nodes, reach })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kingdom_contract::{relations_from_rows, SymbolId};

    /// Symbols of the fixtures, by id.
    const LEFT: SymbolId = 0;
    const S: SymbolId = 1;
    const I: SymbolId = 2;

    /// The design's two-type fixture: `left:S(S,I)`, terminals `s:S`, `i:I`.
    fn two_type() -> KingdomContract {
        relations_from_rows(
            "two-type",
            vec![Ty::S],
            &[
                ("left".to_string(), vec![Ty::S, Ty::I], Ty::S),
                ("s".to_string(), vec![], Ty::S),
                ("i".to_string(), vec![], Ty::I),
            ],
            &[],
        )
        .check()
        .expect("valid")
    }

    fn no_constants() -> BTreeMap<Ty, u32> {
        BTreeMap::new()
    }

    /// The design's table, section 3: head `[left, i, s]` expresses
    /// `left(s, i)`, reading head positions 0, 2, 1.
    #[test]
    fn the_two_type_example_reads_the_head_out_of_physical_order() {
        let k = two_type();
        let layout = GeneLayout::new(&k, 3, Ty::S).expect("layout");
        assert_eq!((layout.width, layout.frame(Ty::S).map(|f| f.tail_len), layout.frame(Ty::I).map(|f| f.tail_len)), (10, Some(4), Some(3)));
        let mut gene = vec![LEFT, I, S];
        gene.extend([S; 4]);
        gene.extend([I; 3]);
        let e = express(&k, &layout, &gene, &no_constants()).expect("expresses");
        let row = |n: &Node| format!("{} {:?} k{} sym{} d{} parent{:?} arg{:?} pos{}", n.step, n.ty, n.frame_k, n.symbol, n.depth, n.parent_step, n.arg_index, n.pos);
        let table: Vec<String> = e.nodes.iter().map(row).collect();
        assert_eq!(
            table,
            vec![
                "1 S k0 sym0 d0 parentNone argNone pos0",
                "2 S k1 sym1 d1 parentSome(1) argSome(0) pos2",
                "3 I k0 sym2 d1 parentSome(1) argSome(1) pos1",
            ]
        );
        assert!(e.reach.tail_used.is_empty(), "the head covered every slot");
    }

    #[test]
    fn a_slot_past_the_head_reads_the_next_row_of_its_own_tail() {
        let k = two_type();
        let layout = GeneLayout::new(&k, 3, Ty::S).expect("layout");
        // head: left left left. The three S slots take the three heads, the fourth takes tail S; the I slots take tail I.
        let mut gene = vec![LEFT, LEFT, LEFT];
        gene.extend([S, S, S, S]);
        gene.extend([I, I, I]);
        let e = express(&k, &layout, &gene, &no_constants()).expect("expresses");
        assert_eq!(e.nodes.len(), 3 + 1 + 3, "3 lefts, one S leaf from the tail, 3 I leaves");
        assert_eq!(e.reach.tail_used.get(&Ty::S), Some(&1));
        assert_eq!(e.reach.tail_used.get(&Ty::I), Some(&3));
    }

    #[test]
    fn keys_are_domain_local_and_cover_every_cell_once() {
        let k = two_type();
        let layout = GeneLayout::new(&k, 3, Ty::S).expect("layout");
        let keys: Vec<DomainKey> = (0..layout.width).map(|p| layout.key(p)).collect();
        assert_eq!(keys[0], DomainKey { domain: Domain::Head, domain_type: None, domain_seq_id: 0 });
        assert_eq!(keys[3], DomainKey { domain: Domain::Tail, domain_type: Some(Ty::S), domain_seq_id: 0 });
        assert_eq!(keys[7], DomainKey { domain: Domain::Tail, domain_type: Some(Ty::I), domain_seq_id: 0 });
        assert_eq!(keys[9], DomainKey { domain: Domain::Tail, domain_type: Some(Ty::I), domain_seq_id: 2 });
        let rows = layout.rows(4, 1, &[0; 10]);
        assert!(rows.iter().all(|r| r.symbol.is_some() && r.dc_index.is_none()), "no Dc in this kingdom");
    }

    /// A kingdom with a constant: `add:F(F,F)`, `x:F`, `?:F`.
    fn with_constant() -> KingdomContract {
        relations_from_rows(
            "c",
            vec![Ty::F],
            &[("add".to_string(), vec![Ty::F, Ty::F], Ty::F), ("x".to_string(), vec![], Ty::F), ("?".to_string(), vec![], Ty::F)],
            &["?"],
        )
        .check()
        .expect("valid")
    }

    #[test]
    fn dc_rows_hold_an_index_and_never_a_symbol() {
        let k = with_constant();
        let layout = GeneLayout::new(&k, 2, Ty::F).expect("layout");
        assert_eq!(layout.width, 2 + 3 + 3);
        let rows = layout.rows(0, 0, &[0, 2, 1, 1, 1, 7, 8, 9]);
        let dc: Vec<_> = rows.iter().filter(|r| r.key.domain == Domain::Dc).collect();
        assert_eq!(dc.len(), 3);
        assert!(dc.iter().all(|r| r.symbol.is_none() && r.dc_index.is_some() && r.key.domain_type == Some(Ty::F)));
    }

    /// Only an EXPRESSED placeholder reads the Dc. Head `[add, ?, x]`:
    /// root add reads head 0; its two F slots read head 1 (`?`) and head 2 (`x`).
    /// A `?` further down the head is not expressed and moves nothing.
    #[test]
    fn only_an_expressed_placeholder_advances_the_dc() {
        let k = with_constant();
        let layout = GeneLayout::new(&k, 4, Ty::F).expect("layout");
        let (add, x, q) = (0, 1, 2);
        // head: add ? x ?   tail: x x x x x (t = 4*1+1 = 5)   dc: 3 0 0 0 0 | 1 2 ...
        let mut gene = vec![add, q, x, q];
        gene.extend([x; 5]);
        gene.extend([3, 0, 0, 0, 0]);
        let consts = BTreeMap::from([(Ty::F, 4)]);
        let e = express(&k, &layout, &gene, &consts).expect("expresses");
        let c: Vec<Option<u32>> = e.nodes.iter().map(|n| n.const_index).collect();
        assert_eq!(c, vec![None, Some(3), None], "the first expressed ? reads Dc[0] = 3");
        assert_eq!(e.reach.q_used.get(&Ty::F), Some(&1), "the second ? in the head was never expressed");
    }

    #[test]
    fn a_dc_word_past_the_constants_is_a_defect() {
        let k = with_constant();
        let layout = GeneLayout::new(&k, 1, Ty::F).expect("layout");
        let gene = vec![2, 1, 1, 9, 0];
        let err = express(&k, &layout, &gene, &BTreeMap::from([(Ty::F, 4)])).expect_err("defect");
        assert_eq!(err, LayoutDefect::DcIndexPastConstants { ty: Ty::F, index: 9, len: 4 });
        assert!(err.to_string().starts_with("our software has a bug: "));
        let none = express(&k, &layout, &[2, 1, 1, 0, 0], &no_constants()).expect_err("an empty C_T");
        assert!(matches!(none, LayoutDefect::DcIndexPastConstants { len: 0, .. }));
    }

    #[test]
    fn a_foreign_terminal_in_a_tail_is_a_defect_not_a_substitution() {
        let k = two_type();
        let layout = GeneLayout::new(&k, 3, Ty::S).expect("layout");
        // left left left; S tail starts with an `i` (type I).
        let mut gene = vec![LEFT, LEFT, LEFT, I, S, S, S];
        gene.extend([I; 3]);
        let err = express(&k, &layout, &gene, &no_constants()).expect_err("defect");
        assert_eq!(err, LayoutDefect::ForeignSymbol { ty: Ty::S, symbol: I, pos: 3 });
    }

    #[test]
    fn a_function_in_a_tail_is_a_defect_even_when_its_type_matches() {
        let k = two_type();
        let layout = GeneLayout::new(&k, 3, Ty::S).expect("layout");
        // left left left; the S tail starts with `left`, which outputs S but is a function.
        let mut gene = vec![LEFT, LEFT, LEFT, LEFT, S, S, S];
        gene.extend([I; 3]);
        let err = express(&k, &layout, &gene, &no_constants()).expect_err("defect");
        assert_eq!(err, LayoutDefect::FunctionInTail { ty: Ty::S, symbol: LEFT, pos: 3 });
        assert_eq!(err.to_string(), "our software has a bug: a function in tail S at position 3");
    }

    #[test]
    fn a_symbol_outside_the_kingdom_and_a_wrong_length_are_defects() {
        let k = two_type();
        let layout = GeneLayout::new(&k, 3, Ty::S).expect("layout");
        let mut gene = vec![99, S, S];
        gene.extend([S; 4]);
        gene.extend([I; 3]);
        assert_eq!(express(&k, &layout, &gene, &no_constants()), Err(LayoutDefect::UnknownSymbol { pos: 0, symbol: 99 }));
        assert_eq!(express(&k, &layout, &gene[..9], &no_constants()), Err(LayoutDefect::GeneLength { want: 10, got: 9 }));
    }

    #[test]
    fn an_exhausted_frame_is_a_defect() {
        // A layout built by hand with a tail shorter than t_S: the walk must
        // report it rather than wrap or pad.
        let k = two_type();
        let mut layout = GeneLayout::new(&k, 3, Ty::S).expect("layout");
        layout.frames[0].tail_len = 0;
        layout.width -= 4;
        let layout = GeneLayout { frames: vec![layout.frames[0].clone(), Frame { tail_off: 3, ..layout.frames[1].clone() }], ..layout };
        let mut gene = vec![LEFT, LEFT, LEFT];
        gene.extend([I; 3]);
        let err = express(&k, &layout, &gene, &no_constants()).expect_err("defect");
        assert_eq!(err, LayoutDefect::TailRanOut { ty: Ty::S, cursor: 3 });
    }

    #[test]
    fn a_root_outside_the_kingdom_is_a_defect() {
        let k = two_type();
        assert_eq!(GeneLayout::new(&k, 3, Ty::I), Err(LayoutDefect::RootNotInKingdom { root: Ty::I }));
    }
}
