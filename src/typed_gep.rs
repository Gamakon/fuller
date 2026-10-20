//! The multi-typed GEP genome layout and its host expression
//! (`nucleotable/docs/MultiTypedGEP.md`, revision 4, sections 2 to 5).
//!
//! A gene has, for every type `T` its root reaches, a `head_T`, a `tail_T` and
//! (for a constant-capable type) a `Dc_T`. A slot requiring `T` reads the next
//! unused row of `T`'s frame, `head_T` then `tail_T`; `head_T` holds symbols
//! whose output is `T`, `tail_T` terminals whose output is `T`. The lengths
//! come from the kingdom's bound table ([`KingdomContract::bounds`]) for a
//! declared program size `N` or explicit `h_T`, and belong to the layout, never
//! to a gene record. Physical order is every head, then every tail, then every
//! Dc, each in frame order (the root's type first, then ascending code), so a
//! one-type gene is exactly the physical Karva head, tail and Dc.
//!
//! There is no repair, substitution or fallback: a cell that does not fit is a
//! [`LayoutDefect`], "our software has a bug: ...". Variation draws from
//! domains enumerated from the layout ([`GeneLayout::alphabet`],
//! [`GeneLayout::spans`], [`Layout::homologous_pairs`]); the per-type
//! write-back ([`write_back`]) is the one encoder every rewrite uses.
//!
//! [`express`] is the host reference. It is the test reference the GPU decoder
//! must match bit for bit, not a search path.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;

use crate::geneframe::Ty;
use crate::kingdom_contract::{GeneBounds, KingdomContract, SymbolId};

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
/// `(chromosome_id, gene_id, domain, domain_type, domain_seq_id)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainKey {
    pub domain: Domain,
    pub domain_type: Ty,
    pub domain_seq_id: usize,
}

/// One type's head, tail and Dc inside a gene.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub ty: Ty,
    pub head_off: usize,
    /// `h_T`.
    pub head_len: usize,
    pub tail_off: usize,
    /// `t_T`.
    pub tail_len: usize,
    /// Present for a constant-capable type.
    pub dc_off: Option<usize>,
    /// `d_T`; zero without a Dc.
    pub dc_len: usize,
}

impl Frame {
    /// The gene position of row `k` of this frame (`head_T` then `tail_T`), or
    /// `None` past its end.
    pub fn row_pos(&self, k: usize) -> Option<usize> {
        if k < self.head_len {
            Some(self.head_off + k)
        } else if k - self.head_len < self.tail_len {
            Some(self.tail_off + (k - self.head_len))
        } else {
            None
        }
    }
}

/// A contiguous run of one domain of one type: the unit every operator stays in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub domain: Domain,
    pub ty: Ty,
    pub off: usize,
    pub len: usize,
}

/// Where everything is in a gene rooted at `root`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneLayout {
    pub root: Ty,
    /// The declared program size `N`, if the heads were derived from one.
    pub program_size: Option<usize>,
    pub frames: Vec<Frame>,
    pub width: usize,
}

impl GeneLayout {
    /// Heads from the declared program size `N` (design section 2.1).
    pub fn from_program_size(k: &KingdomContract, root: Ty, n: usize) -> Result<GeneLayout, LayoutDefect> {
        if !k.roots().contains(&root) {
            return Err(LayoutDefect::RootNotInKingdom { root });
        }
        let heads = k.heads_from_program_size(root, n).map_err(LayoutDefect::Contract)?;
        let bounds = k.bounds(root, &heads).map_err(LayoutDefect::Contract)?;
        Ok(Self::build(root, Some(n), &bounds))
    }

    /// Explicit head lengths `h_T`; [`GeneLayout::guaranteed_program_size`]
    /// reports the largest `N` they hold.
    pub fn from_heads(k: &KingdomContract, root: Ty, heads: &BTreeMap<Ty, usize>) -> Result<GeneLayout, LayoutDefect> {
        if !k.roots().contains(&root) {
            return Err(LayoutDefect::RootNotInKingdom { root });
        }
        let bounds = k.bounds(root, heads).map_err(LayoutDefect::Contract)?;
        Ok(Self::build(root, None, &bounds))
    }

    fn build(root: Ty, program_size: Option<usize>, b: &GeneBounds) -> GeneLayout {
        let mut frames: Vec<Frame> = Vec::with_capacity(b.types.len());
        let mut off = 0;
        for t in &b.types {
            frames.push(Frame { ty: t.ty, head_off: off, head_len: t.h, tail_off: 0, tail_len: t.t, dc_off: None, dc_len: 0 });
            off += t.h;
        }
        for f in &mut frames {
            f.tail_off = off;
            off += f.tail_len;
        }
        for (f, t) in frames.iter_mut().zip(&b.types) {
            if t.constant {
                f.dc_off = Some(off);
                f.dc_len = t.d;
                off += t.d;
            }
        }
        GeneLayout { root, program_size, frames, width: off }
    }

    pub fn frame(&self, ty: Ty) -> Option<&Frame> {
        self.frames.iter().find(|f| f.ty == ty)
    }

    /// `h_T` for every frame.
    pub fn head_lens(&self) -> BTreeMap<Ty, usize> {
        self.frames.iter().map(|f| (f.ty, f.head_len)).collect()
    }

    /// The largest program size the heads guarantee.
    pub fn guaranteed_program_size(&self, k: &KingdomContract) -> usize {
        k.guaranteed_program_size(self.root, &self.head_lens())
    }

    /// Two genes are homologous when the layout gives them the same root and
    /// the same `h_T`; the tails and Dcs then agree too.
    pub fn homologous(&self, other: &GeneLayout) -> bool {
        self.root == other.root && self.head_lens() == other.head_lens()
    }

    /// Every non-empty domain, in physical order.
    pub fn spans(&self) -> Vec<Span> {
        let mut v = Vec::new();
        for f in &self.frames {
            v.push(Span { domain: Domain::Head, ty: f.ty, off: f.head_off, len: f.head_len });
        }
        for f in &self.frames {
            v.push(Span { domain: Domain::Tail, ty: f.ty, off: f.tail_off, len: f.tail_len });
        }
        for f in &self.frames {
            if let Some(d) = f.dc_off {
                v.push(Span { domain: Domain::Dc, ty: f.ty, off: d, len: f.dc_len });
            }
        }
        v.retain(|s| s.len > 0);
        v
    }

    /// The domain-local key of the cell at `pos`.
    pub fn key(&self, pos: usize) -> DomainKey {
        for s in self.spans() {
            if pos >= s.off && pos < s.off + s.len {
                return DomainKey { domain: s.domain, domain_type: s.ty, domain_seq_id: pos - s.off };
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

    /// The values a mutation may write at `pos`: any symbol outputting `T` in
    /// `head_T`, a terminal outputting `T` in `tail_T`, an index below `C_T` in
    /// `Dc_T`. Enumerated before the draw, so a drawn child is valid.
    pub fn alphabet(&self, k: &KingdomContract, pos: usize, const_len: &BTreeMap<Ty, u32>) -> Vec<u32> {
        let key = self.key(pos);
        let t = key.domain_type;
        match key.domain {
            Domain::Head => (0..k.n_symbols() as SymbolId).filter(|s| k.output(*s) == t).collect(),
            Domain::Tail => k.terminals_of(t).to_vec(),
            Domain::Dc => (0..const_len.get(&t).copied().unwrap_or(0)).collect(),
        }
    }

    /// The span holding `[lo, hi)`, if one does.
    fn span_of(&self, lo: usize, hi: usize) -> Result<Span, LayoutDefect> {
        self.spans().into_iter().find(|s| lo >= s.off && hi <= s.off + s.len && lo < hi).ok_or(LayoutDefect::RangeAcrossDomains { lo, hi })
    }

    /// Point mutation: `value` must be in the alphabet of the cell.
    pub fn mutate(&self, k: &KingdomContract, gene: &mut [u32], pos: usize, value: u32, const_len: &BTreeMap<Ty, u32>) -> Result<(), LayoutDefect> {
        if !self.alphabet(k, pos, const_len).contains(&value) {
            return Err(LayoutDefect::NotInAlphabet { pos, value });
        }
        gene[pos] = value;
        Ok(())
    }

    /// Homologous recombination: exchange `[lo, hi)` between two genes. Both
    /// layouts must be homologous and the range inside one domain of one type.
    pub fn swap_range(&self, other: &GeneLayout, a: &mut [u32], b: &mut [u32], lo: usize, hi: usize) -> Result<(), LayoutDefect> {
        if !self.homologous(other) {
            return Err(LayoutDefect::NotHomologous);
        }
        self.span_of(lo, hi)?;
        a[lo..hi].swap_with_slice(&mut b[lo..hi]);
        Ok(())
    }

    /// Inversion of `[lo, hi)`, inside one domain of one type.
    pub fn invert(&self, gene: &mut [u32], lo: usize, hi: usize) -> Result<(), LayoutDefect> {
        self.span_of(lo, hi)?;
        gene[lo..hi].reverse();
        Ok(())
    }
}

/// The genes of one layout id: the root and domain lengths of every gene id
/// (`layout_genes`, `layout_heads`). A gene record carries neither.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub layout_id: u64,
    genes: Vec<GeneLayout>,
}

impl Layout {
    /// The gene id of each layout is its index.
    pub fn new(layout_id: u64, genes: Vec<GeneLayout>) -> Layout {
        Layout { layout_id, genes }
    }

    pub fn n_genes(&self) -> usize {
        self.genes.len()
    }

    pub fn gene(&self, gene_id: u64) -> Option<&GeneLayout> {
        self.genes.get(gene_id as usize)
    }

    pub fn genes(&self) -> &[GeneLayout] {
        &self.genes
    }

    /// The ordered pairs `(from, to)`, `from != to`, of homologous gene ids:
    /// the only gene transpositions and whole-gene crossovers the layout
    /// permits. Empty with one gene per root and head set.
    pub fn homologous_pairs(&self) -> Vec<(u64, u64)> {
        let mut v = Vec::new();
        for (i, a) in self.genes.iter().enumerate() {
            for (j, b) in self.genes.iter().enumerate() {
                if i != j && a.homologous(b) {
                    v.push((i as u64, j as u64));
                }
            }
        }
        v
    }

    /// `g_T`: how many genes of the layout have root `ty` (the genic terminals of `ty`).
    pub fn genes_rooted_at(&self, ty: Ty) -> usize {
        self.genes.iter().filter(|g| g.root == ty).count()
    }

    /// Gene transposition: copy gene `from` over gene `to`. Only a homologous
    /// pair is in the domain.
    pub fn copy_gene(&self, genes: &mut [Vec<u32>], from: u64, to: u64) -> Result<(), LayoutDefect> {
        if !self.homologous_pairs().contains(&(from, to)) {
            return Err(LayoutDefect::NotHomologous);
        }
        let src = genes[from as usize].clone();
        genes[to as usize] = src;
        Ok(())
    }

    /// Whole-gene recombination: exchange gene `g` between two parents of this layout.
    pub fn exchange_gene(&self, a: &mut [Vec<u32>], b: &mut [Vec<u32>], g: u64) -> Result<(), LayoutDefect> {
        if self.gene(g).is_none() {
            return Err(LayoutDefect::NotHomologous);
        }
        std::mem::swap(&mut a[g as usize], &mut b[g as usize]);
        Ok(())
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
    /// A bound-table or head derivation the kingdom contract refused (already spelled).
    Contract(String),
    GeneLength { want: usize, got: usize },
    UnknownSymbol { pos: usize, symbol: u32 },
    /// A slot of `ty` found its frame (head rows then tail) exhausted.
    TailRanOut { ty: Ty, cursor: usize },
    /// A head or tail cell of type `ty` holds a symbol whose output is another type.
    ForeignSymbol { ty: Ty, symbol: SymbolId, pos: usize },
    /// A tail cell holds a function; a tail holds terminals of its type only.
    FunctionInTail { ty: Ty, symbol: SymbolId, pos: usize },
    /// A slot asks for a type with no frame.
    NoFrame { ty: Ty },
    /// More expressed placeholders of `ty` than its Dc holds.
    DcRanOut { ty: Ty, q: usize },
    /// A Dc word names a constant past the end of `C_T`.
    DcIndexPastConstants { ty: Ty, index: u32, len: u32 },
    /// A write outside the alphabet of its cell.
    NotInAlphabet { pos: usize, value: u32 },
    /// An exchange between genes the layout does not call homologous.
    NotHomologous,
    /// An operator range that is not inside one domain of one type.
    RangeAcrossDomains { lo: usize, hi: usize },
    /// A tree to encode whose node `step` fills a slot of another type.
    TreeNotTyped { step: usize, want: Ty, got: Ty },
    /// A tree to encode whose node `step` has the wrong number of children.
    TreeArity { step: usize, want: usize, got: usize },
    /// The last `ty` function of the tree sits at rank `need - 1`, past `head_T` (`h`).
    HeadTooShort { ty: Ty, need: usize, h: usize },
    /// The tree has more `ty` nodes than `head_T` and `tail_T` hold.
    TailTooShort { ty: Ty, need: usize, rows: usize },
    /// The tree has more `?_T` than `Dc_T` holds.
    DcTooShort { ty: Ty, need: usize, d: usize },
    /// A `?_T` node with no `C_T` index.
    MissingConstIndex { step: usize },
}

impl fmt::Display for LayoutDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let LayoutDefect::Contract(m) = self {
            return write!(f, "{m}");
        }
        write!(f, "our software has a bug: ")?;
        match self {
            LayoutDefect::Contract(_) => Ok(()),
            LayoutDefect::RootNotInKingdom { root } => write!(f, "root type {root:?} is not a root of this kingdom"),
            LayoutDefect::GeneLength { want, got } => write!(f, "a gene has {got} words, its layout has {want}"),
            LayoutDefect::UnknownSymbol { pos, symbol } => write!(f, "cell {pos} holds symbol {symbol}, which is not in the kingdom"),
            LayoutDefect::TailRanOut { ty, cursor } => write!(f, "the frame of type {ty:?} ran out at cursor {cursor}"),
            LayoutDefect::ForeignSymbol { ty, symbol, pos } => write!(f, "cell {pos} of a domain of type {ty:?} holds symbol {symbol}, whose output is another type"),
            LayoutDefect::FunctionInTail { ty, pos, .. } => write!(f, "a function in tail {ty:?} at position {pos}"),
            LayoutDefect::NoFrame { ty } => write!(f, "a slot asks for type {ty:?}, which has no frame"),
            LayoutDefect::DcRanOut { ty, q } => write!(f, "the Dc of type {ty:?} ran out at placeholder {q}"),
            LayoutDefect::DcIndexPastConstants { ty, index, len } => write!(f, "a Dc word of type {ty:?} is {index}, C_T has {len} constants"),
            LayoutDefect::NotInAlphabet { pos, value } => write!(f, "value {value} is not in the alphabet of cell {pos}"),
            LayoutDefect::NotHomologous => write!(f, "an exchange between genes the layout does not call homologous"),
            LayoutDefect::RangeAcrossDomains { lo, hi } => write!(f, "range [{lo}, {hi}) is not inside one domain of one type"),
            LayoutDefect::TreeNotTyped { step, want, got } => write!(f, "tree node {step} fills a slot of type {want:?} and outputs {got:?}"),
            LayoutDefect::TreeArity { step, want, got } => write!(f, "tree node {step} has {got} children, its symbol takes {want}"),
            LayoutDefect::HeadTooShort { ty, need, h } => write!(f, "the tree needs {need} head rows of type {ty:?}, h_T is {h}"),
            LayoutDefect::TailTooShort { ty, need, rows } => write!(f, "the tree has {need} nodes of type {ty:?}, head and tail hold {rows}"),
            LayoutDefect::DcTooShort { ty, need, d } => write!(f, "the tree has {need} placeholders of type {ty:?}, Dc holds {d}"),
            LayoutDefect::MissingConstIndex { step } => write!(f, "tree node {step} is a constant placeholder with no C_T index"),
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

/// How far into each frame and Dc a walk went: the measured side of the bound.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reach {
    /// Frame rows read (`head_T` and `tail_T` together), per type.
    pub frame_used: BTreeMap<Ty, usize>,
    /// Tail rows read, per type.
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

/// An expression as a tree of symbols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tree {
    pub symbol: SymbolId,
    pub const_index: Option<u32>,
    pub kids: Vec<Tree>,
}

impl Expressed {
    pub fn tree(&self) -> Tree {
        let mut kids: Vec<Vec<(u32, usize)>> = vec![Vec::new(); self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            if let (Some(p), Some(a)) = (n.parent_step, n.arg_index) {
                kids[p - 1].push((a, i));
            }
        }
        fn build(e: &Expressed, kids: &[Vec<(u32, usize)>], i: usize) -> Tree {
            let mut ks = kids[i].clone();
            ks.sort_unstable();
            Tree { symbol: e.nodes[i].symbol, const_index: e.nodes[i].const_index, kids: ks.into_iter().map(|(_, c)| build(e, kids, c)).collect() }
        }
        build(self, &kids, 0)
    }
}

impl Tree {
    /// The nodes in level order (the order of the walk), with the path of
    /// argument indices from the root to each.
    pub fn level_order(&self) -> Vec<(&Tree, Vec<u32>)> {
        let mut out: Vec<(&Tree, Vec<u32>)> = Vec::new();
        let mut queue = VecDeque::from([(self, Vec::new())]);
        while let Some((t, path)) = queue.pop_front() {
            for (i, c) in t.kids.iter().enumerate() {
                let mut p = path.clone();
                p.push(i as u32);
                queue.push_back((c, p));
            }
            out.push((t, path));
        }
        out
    }

    fn replace_at(&self, path: &[u32], with: &Tree) -> Tree {
        match path.split_first() {
            None => with.clone(),
            Some((i, rest)) => {
                let mut kids = self.kids.clone();
                kids[*i as usize] = kids[*i as usize].replace_at(rest, with);
                Tree { symbol: self.symbol, const_index: self.const_index, kids }
            }
        }
    }
}

struct Slot {
    ty: Ty,
    depth: usize,
    parent_step: Option<usize>,
    arg_index: Option<u32>,
}

fn check_symbol(k: &KingdomContract, pos: usize, w: u32) -> Result<SymbolId, LayoutDefect> {
    if w as usize >= k.n_symbols() {
        Err(LayoutDefect::UnknownSymbol { pos, symbol: w })
    } else {
        Ok(w)
    }
}

/// Every cell of a gene against its domain: a head cell outputs its domain's
/// type, a tail cell is a terminal that does, a Dc word indexes `C_T`. A cell
/// that is not read by the walk is still checked: the domain alphabets are the
/// definition of a valid gene.
pub fn validate(k: &KingdomContract, layout: &GeneLayout, gene: &[u32], const_len: &BTreeMap<Ty, u32>) -> Result<(), LayoutDefect> {
    if gene.len() != layout.width {
        return Err(LayoutDefect::GeneLength { want: layout.width, got: gene.len() });
    }
    for s in layout.spans() {
        for (i, w) in gene[s.off..s.off + s.len].iter().enumerate() {
            let pos = s.off + i;
            match s.domain {
                Domain::Head | Domain::Tail => {
                    let sym = check_symbol(k, pos, *w)?;
                    if k.output(sym) != s.ty {
                        return Err(LayoutDefect::ForeignSymbol { ty: s.ty, symbol: sym, pos });
                    }
                    if s.domain == Domain::Tail && !k.is_terminal(sym) {
                        return Err(LayoutDefect::FunctionInTail { ty: s.ty, symbol: sym, pos });
                    }
                }
                Domain::Dc => {
                    let len = const_len.get(&s.ty).copied().unwrap_or(0);
                    if *w >= len {
                        return Err(LayoutDefect::DcIndexPastConstants { ty: s.ty, index: *w, len });
                    }
                }
            }
        }
    }
    Ok(())
}

/// Express a gene: the FIFO walk of the design, section 3. `const_len` is the
/// length of each type's constant array `C_T`.
pub fn express(k: &KingdomContract, layout: &GeneLayout, gene: &[u32], const_len: &BTreeMap<Ty, u32>) -> Result<Expressed, LayoutDefect> {
    validate(k, layout, gene, const_len)?;
    let mut k_t: BTreeMap<Ty, usize> = BTreeMap::new();
    let mut q_t: BTreeMap<Ty, usize> = BTreeMap::new();
    let mut reach = Reach::default();
    let mut nodes: Vec<Node> = Vec::new();
    let mut queue = VecDeque::from([Slot { ty: layout.root, depth: 0, parent_step: None, arg_index: None }]);
    while let Some(slot) = queue.pop_front() {
        let frame = layout.frame(slot.ty).ok_or(LayoutDefect::NoFrame { ty: slot.ty })?;
        let cursor = k_t.entry(slot.ty).or_insert(0);
        let frame_k = *cursor;
        *cursor += 1;
        let pos = frame.row_pos(frame_k).ok_or(LayoutDefect::TailRanOut { ty: slot.ty, cursor: frame_k })?;
        reach.frame_used.insert(slot.ty, frame_k + 1);
        if frame_k >= frame.head_len {
            reach.tail_used.insert(slot.ty, frame_k - frame.head_len + 1);
        }
        let symbol = gene[pos];
        let step = nodes.len() + 1;
        let const_index = if k.is_constant(symbol) {
            let q = *q_t.entry(slot.ty).or_default();
            let dc_off = frame.dc_off.ok_or(LayoutDefect::DcRanOut { ty: slot.ty, q })?;
            if q >= frame.dc_len {
                return Err(LayoutDefect::DcRanOut { ty: slot.ty, q });
            }
            q_t.insert(slot.ty, q + 1);
            reach.q_used.insert(slot.ty, q + 1);
            Some(gene[dc_off + q])
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

// ---------------------------------------------------------------------------
// The per-type write-back (design section 5)
// ---------------------------------------------------------------------------

/// The cells a tree occupies in a layout: `(position, word)`.
struct Placement {
    cells: Vec<(usize, u32)>,
}

/// `j_T`: one more than the rank among `T` nodes (level order) of the last `T`
/// function, zero with none. A tree is encodable exactly when `j_T <= h_T`.
pub fn required_heads(k: &KingdomContract, tree: &Tree) -> BTreeMap<Ty, usize> {
    let mut rank: BTreeMap<Ty, usize> = BTreeMap::new();
    let mut j: BTreeMap<Ty, usize> = BTreeMap::new();
    for (t, _) in tree.level_order() {
        let ty = k.output(t.symbol);
        let r = rank.entry(ty).or_default();
        if !t.kids.is_empty() {
            j.insert(ty, *r + 1);
        }
        *r += 1;
    }
    j
}

fn place(k: &KingdomContract, layout: &GeneLayout, tree: &Tree) -> Result<Placement, LayoutDefect> {
    let order = tree.level_order();
    // Typing: the root has the layout's root type, each child the type its slot requires.
    if k.output(tree.symbol) != layout.root {
        return Err(LayoutDefect::TreeNotTyped { step: 1, want: layout.root, got: k.output(tree.symbol) });
    }
    let step_of: BTreeMap<Vec<u32>, usize> = order.iter().enumerate().map(|(i, (_, path))| (path.clone(), i + 1)).collect();
    for (i, (t, path)) in order.iter().enumerate() {
        let sig = k.signature(t.symbol);
        if sig.len() != t.kids.len() {
            return Err(LayoutDefect::TreeArity { step: i + 1, want: sig.len(), got: t.kids.len() });
        }
        for (a, c) in t.kids.iter().enumerate() {
            if k.output(c.symbol) != sig[a] {
                let mut child = path.clone();
                child.push(a as u32);
                return Err(LayoutDefect::TreeNotTyped { step: step_of[&child], want: sig[a], got: k.output(c.symbol) });
            }
        }
    }
    for (ty, need) in required_heads(k, tree) {
        let f = layout.frame(ty).ok_or(LayoutDefect::NoFrame { ty })?;
        if need > f.head_len {
            return Err(LayoutDefect::HeadTooShort { ty, need, h: f.head_len });
        }
    }
    let mut rank: BTreeMap<Ty, usize> = BTreeMap::new();
    let mut q: BTreeMap<Ty, usize> = BTreeMap::new();
    let mut cells = Vec::new();
    for (i, (t, _)) in order.iter().enumerate() {
        let ty = k.output(t.symbol);
        let f = layout.frame(ty).ok_or(LayoutDefect::NoFrame { ty })?;
        let r = rank.entry(ty).or_default();
        let pos = f.row_pos(*r).ok_or(LayoutDefect::TailTooShort { ty, need: *r + 1, rows: f.head_len + f.tail_len })?;
        *r += 1;
        cells.push((pos, t.symbol));
        if k.is_constant(t.symbol) {
            let n = q.entry(ty).or_default();
            let idx = t.const_index.ok_or(LayoutDefect::MissingConstIndex { step: i + 1 })?;
            let off = f.dc_off.filter(|_| *n < f.dc_len).ok_or(LayoutDefect::DcTooShort { ty, need: *n + 1, d: f.dc_len })?;
            cells.push((off + *n, idx));
            *n += 1;
        }
    }
    Ok(Placement { cells })
}

/// Whether `tree` is encodable in `layout` (`j_T <= h_T` for every type, and
/// its nodes and placeholders inside the tails and Dcs).
pub fn fits(k: &KingdomContract, layout: &GeneLayout, tree: &Tree) -> Result<(), LayoutDefect> {
    place(k, layout, tree).map(|_| ())
}

/// The write-back of a rewritten expressed tree (fold, cleanse, snap,
/// promotion): the walk of section 3 in reverse. The `T` nodes in level order
/// are rows `0, 1, 2, ...` of `T`'s frame, so the first `h_T` are `head_T` and
/// the rest `tail_T`; `Dc_T` is rebuilt in the order the `?_T` appear; rows the
/// walk does not read keep their `previous` contents (Karva's non-coding
/// region, deterministic).
pub fn write_back(k: &KingdomContract, layout: &GeneLayout, previous: &[u32], tree: &Tree) -> Result<Vec<u32>, LayoutDefect> {
    if previous.len() != layout.width {
        return Err(LayoutDefect::GeneLength { want: layout.width, got: previous.len() });
    }
    let mut gene = previous.to_vec();
    for (pos, w) in place(k, layout, tree)?.cells {
        gene[pos] = w;
    }
    Ok(gene)
}

/// Promotion lifts a same-type child into its parent's place. Its domain on a
/// gene is the `(level-order step, argument)` pairs whose promoted tree is
/// encodable, enumerated before the draw.
pub fn promotions(k: &KingdomContract, layout: &GeneLayout, tree: &Tree) -> Vec<(usize, u32)> {
    let mut out = Vec::new();
    for (i, (t, _)) in tree.level_order().iter().enumerate() {
        for (a, c) in t.kids.iter().enumerate() {
            if k.output(c.symbol) == k.output(t.symbol) && fits(k, layout, &promote(tree, i + 1, a as u32)).is_ok() {
                out.push((i + 1, a as u32));
            }
        }
    }
    out
}

/// The tree with node `step` (level order from 1) replaced by its child `arg`.
pub fn promote(tree: &Tree, step: usize, arg: u32) -> Tree {
    let order = tree.level_order();
    let (node, path) = &order[step - 1];
    tree.replace_at(path, &node.kids[arg as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kingdom_contract::{relations_from_rows, SymbolId};

    /// Symbols of the fixtures, by id.
    const LEFT: SymbolId = 0;
    const S: SymbolId = 1;
    const I: SymbolId = 2;
    const Z_S: SymbolId = 3;
    const Z_I: SymbolId = 4;

    /// The design's two-type fixture: `left:S(S,I)`, terminals `s:S`, `i:I`, and
    /// the polymorphic spelling `z` as `z:S` and `z:I`.
    fn two_type() -> KingdomContract {
        relations_from_rows(
            "two-type",
            vec![Ty::S],
            &[
                ("left".to_string(), vec![Ty::S, Ty::I], Ty::S),
                ("s".to_string(), vec![], Ty::S),
                ("i".to_string(), vec![], Ty::I),
                ("z".to_string(), vec![], Ty::S),
                ("z".to_string(), vec![], Ty::I),
            ],
            &[],
        )
        .check()
        .expect("valid")
    }

    fn heads(pairs: &[(Ty, usize)]) -> BTreeMap<Ty, usize> {
        pairs.iter().copied().collect()
    }

    /// `h_S = 2`, `h_I = 1`: head_S, head_I, tail_S (1), tail_I (1).
    fn design_layout(k: &KingdomContract) -> GeneLayout {
        GeneLayout::from_heads(k, Ty::S, &heads(&[(Ty::S, 2), (Ty::I, 1)])).expect("layout")
    }

    fn no_constants() -> BTreeMap<Ty, u32> {
        BTreeMap::new()
    }

    fn row(n: &Node) -> String {
        format!("{} {:?} k{} sym{} d{} parent{:?} arg{:?} pos{}", n.step, n.ty, n.frame_k, n.symbol, n.depth, n.parent_step, n.arg_index, n.pos)
    }

    /// The design's table (section 3): `head_S = (left, s)`, `head_I = (i)`
    /// expresses `left(s, i)`; the I slot reads `head_I`, not a shared stream.
    #[test]
    fn the_design_example_reads_each_type_from_its_own_head() {
        let k = two_type();
        let layout = design_layout(&k);
        assert_eq!(layout.width, 2 + 1 + 1 + 1);
        assert_eq!(layout.guaranteed_program_size(&k), 2);
        let gene = vec![LEFT, S, I, S, I];
        let e = express(&k, &layout, &gene, &no_constants()).expect("expresses");
        let table: Vec<String> = e.nodes.iter().map(row).collect();
        assert_eq!(
            table,
            vec!["1 S k0 sym0 d0 parentNone argNone pos0", "2 S k1 sym1 d1 parentSome(1) argSome(0) pos1", "3 I k0 sym2 d1 parentSome(1) argSome(1) pos2"]
        );
        assert!(e.reach.tail_used.is_empty(), "the heads covered every slot");
    }

    /// The design's second gene: `head_S = (left, left)`, `tail_S = (s)`,
    /// `head_I = (i)`, `tail_I = (z)` is `left(left(s, z), i)`, with the
    /// polymorphic `z` of output I.
    #[test]
    fn the_design_second_gene_uses_the_tails_and_the_polymorphic_spelling() {
        let k = two_type();
        let layout = design_layout(&k);
        let e = express(&k, &layout, &[LEFT, LEFT, I, S, Z_I], &no_constants()).expect("expresses");
        let t = e.tree();
        let leaf = |symbol| Tree { symbol, const_index: None, kids: vec![] };
        assert_eq!(t, Tree { symbol: LEFT, const_index: None, kids: vec![Tree { symbol: LEFT, const_index: None, kids: vec![leaf(S), leaf(Z_I)] }, leaf(I)] });
        assert_eq!((e.reach.tail_used.get(&Ty::S), e.reach.tail_used.get(&Ty::I)), (Some(&1), Some(&1)));
        assert_ne!(Z_S, Z_I, "one symbol_id per output type");
    }

    #[test]
    fn keys_are_domain_local_and_cover_every_cell_once() {
        let k = two_type();
        let layout = design_layout(&k);
        let key = |domain, ty, seq| DomainKey { domain, domain_type: ty, domain_seq_id: seq };
        let keys: Vec<DomainKey> = (0..layout.width).map(|p| layout.key(p)).collect();
        assert_eq!(
            keys,
            vec![
                key(Domain::Head, Ty::S, 0),
                key(Domain::Head, Ty::S, 1),
                key(Domain::Head, Ty::I, 0),
                key(Domain::Tail, Ty::S, 0),
                key(Domain::Tail, Ty::I, 0)
            ]
        );
        let rows = layout.rows(4, 1, &[0; 5]);
        assert!(rows.iter().all(|r| r.symbol.is_some() && r.dc_index.is_none()), "no Dc in this kingdom");
    }

    #[test]
    fn heads_from_a_program_size_and_the_largest_size_they_hold() {
        let k = two_type();
        for n in 1..12usize {
            let l = GeneLayout::from_program_size(&k, Ty::S, n).expect("layout");
            assert_eq!(l.program_size, Some(n));
            assert_eq!(l.guaranteed_program_size(&k), n);
            assert_eq!(l.head_lens(), heads(&[(Ty::S, n), (Ty::I, n - 1)]));
        }
        assert_eq!(GeneLayout::from_program_size(&k, Ty::S, 0).expect_err("zero").to_string(), "our software has a bug: two-type: a layout needs a program size of at least one function");
        assert_eq!(GeneLayout::from_program_size(&k, Ty::I, 3), Err(LayoutDefect::RootNotInKingdom { root: Ty::I }));
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
        let layout = GeneLayout::from_heads(&k, Ty::F, &heads(&[(Ty::F, 2)])).expect("layout");
        assert_eq!(layout.width, 2 + 3 + 3);
        let rows = layout.rows(0, 0, &[0, 2, 1, 1, 1, 0, 1, 2]);
        let dc: Vec<_> = rows.iter().filter(|r| r.key.domain == Domain::Dc).collect();
        assert_eq!(dc.len(), 3);
        assert!(dc.iter().all(|r| r.symbol.is_none() && r.dc_index.is_some() && r.key.domain_type == Ty::F));
    }

    /// Only an EXPRESSED placeholder reads the Dc. Head `[add, ?, x, ?]`:
    /// root add reads head 0; its two F slots read head 1 (`?`) and head 2 (`x`).
    /// A `?` further down the head is not expressed and moves nothing.
    #[test]
    fn only_an_expressed_placeholder_advances_the_dc() {
        let k = with_constant();
        let layout = GeneLayout::from_heads(&k, Ty::F, &heads(&[(Ty::F, 4)])).expect("layout");
        let (add, x, q) = (0, 1, 2);
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
        let layout = GeneLayout::from_heads(&k, Ty::F, &heads(&[(Ty::F, 1)])).expect("layout");
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
        let layout = design_layout(&k);
        let err = express(&k, &layout, &[LEFT, S, I, I, I], &no_constants()).expect_err("defect");
        assert_eq!(err, LayoutDefect::ForeignSymbol { ty: Ty::S, symbol: I, pos: 3 });
    }

    /// A head cell whose output is not its domain's type is a defect even when
    /// the walk never reads it.
    #[test]
    fn a_symbol_of_the_wrong_output_in_a_head_is_a_defect_read_or_not() {
        let k = two_type();
        let layout = design_layout(&k);
        let err = express(&k, &layout, &[LEFT, I, I, S, I], &no_constants()).expect_err("an I in head_S");
        assert_eq!(err, LayoutDefect::ForeignSymbol { ty: Ty::S, symbol: I, pos: 1 });
        let err = express(&k, &layout, &[S, S, S, S, I], &no_constants()).expect_err("an S in head_I, never read");
        assert_eq!(err, LayoutDefect::ForeignSymbol { ty: Ty::I, symbol: S, pos: 2 });
    }

    #[test]
    fn a_function_in_a_tail_is_a_defect_even_when_its_type_matches() {
        let k = two_type();
        let layout = design_layout(&k);
        let err = express(&k, &layout, &[LEFT, LEFT, I, LEFT, I], &no_constants()).expect_err("defect");
        assert_eq!(err, LayoutDefect::FunctionInTail { ty: Ty::S, symbol: LEFT, pos: 3 });
        assert_eq!(err.to_string(), "our software has a bug: a function in tail S at position 3");
    }

    #[test]
    fn a_symbol_outside_the_kingdom_and_a_wrong_length_are_defects() {
        let k = two_type();
        let layout = design_layout(&k);
        assert_eq!(express(&k, &layout, &[99, S, I, S, I], &no_constants()), Err(LayoutDefect::UnknownSymbol { pos: 0, symbol: 99 }));
        assert_eq!(express(&k, &layout, &[LEFT, S, I, S], &no_constants()), Err(LayoutDefect::GeneLength { want: 5, got: 4 }));
    }

    #[test]
    fn an_exhausted_frame_is_a_defect() {
        // A layout built by hand with a tail shorter than t_S: the walk must
        // report it rather than wrap or pad.
        let k = two_type();
        let layout = GeneLayout::from_heads(&k, Ty::S, &heads(&[(Ty::S, 1), (Ty::I, 1)])).expect("layout");
        assert_eq!(layout.width, 3, "head_S, head_I, tail_S");
        let mut short = layout.clone();
        short.frames[0].tail_len = 0;
        short.width = 2;
        let err = express(&k, &short, &[LEFT, I], &no_constants()).expect_err("defect");
        assert_eq!(err, LayoutDefect::TailRanOut { ty: Ty::S, cursor: 1 });
    }

    #[test]
    fn a_root_outside_the_kingdom_is_a_defect() {
        let k = two_type();
        assert_eq!(GeneLayout::from_heads(&k, Ty::I, &heads(&[(Ty::I, 3)])), Err(LayoutDefect::RootNotInKingdom { root: Ty::I }));
        assert!(matches!(GeneLayout::from_heads(&k, Ty::S, &heads(&[(Ty::S, 3), (Ty::B, 1)])), Err(LayoutDefect::Contract(_))));
    }

    // -- operators stay in their domain --------------------------------------

    #[test]
    fn mutation_alphabets_are_per_domain() {
        let k = two_type();
        let layout = design_layout(&k);
        let a = |pos| layout.alphabet(&k, pos, &no_constants());
        assert_eq!(a(0), vec![LEFT, S, Z_S], "head_S: every symbol outputting S");
        assert_eq!(a(2), vec![I, Z_I], "head_I: every symbol outputting I");
        assert_eq!(a(3), vec![S, Z_S], "tail_S: terminals outputting S");
        assert_eq!(a(4), vec![I, Z_I]);
        let mut g = vec![LEFT, S, I, S, I];
        assert_eq!(layout.mutate(&k, &mut g, 2, S, &no_constants()), Err(LayoutDefect::NotInAlphabet { pos: 2, value: S }));
        assert_eq!(g, vec![LEFT, S, I, S, I], "a refused write changes nothing");
        layout.mutate(&k, &mut g, 2, Z_I, &no_constants()).expect("in the alphabet");
        assert_eq!(g[2], Z_I);
    }

    #[test]
    fn range_operators_stay_inside_one_domain_and_homologous_genes() {
        let k = two_type();
        let layout = design_layout(&k);
        let mut a = vec![LEFT, S, I, S, I];
        let mut b = vec![S, LEFT, Z_I, Z_S, Z_I];
        layout.swap_range(&layout, &mut a, &mut b, 0, 2).expect("head_S");
        assert_eq!((&a[..2], &b[..2]), (&[S, LEFT][..], &[LEFT, S][..]));
        assert_eq!(layout.swap_range(&layout, &mut a, &mut b, 1, 3), Err(LayoutDefect::RangeAcrossDomains { lo: 1, hi: 3 }));
        assert_eq!(layout.invert(&mut a, 0, 3), Err(LayoutDefect::RangeAcrossDomains { lo: 0, hi: 3 }));
        layout.invert(&mut a, 0, 2).expect("head_S");
        assert_eq!(&a[..2], &[LEFT, S]);
        let other = GeneLayout::from_heads(&k, Ty::S, &heads(&[(Ty::S, 3), (Ty::I, 1)])).expect("layout");
        assert!(!layout.homologous(&other));
        let mut c = vec![0; other.width];
        assert_eq!(layout.swap_range(&other, &mut a, &mut c, 0, 1), Err(LayoutDefect::NotHomologous));
    }

    #[test]
    fn gene_moves_are_only_between_homologous_ids() {
        let k = two_type();
        let g = |h_s| GeneLayout::from_heads(&k, Ty::S, &heads(&[(Ty::S, h_s), (Ty::I, 1)])).expect("layout");
        let layout = Layout::new(7, vec![g(2), g(3), g(2), g(3), g(4)]);
        assert_eq!(layout.homologous_pairs(), vec![(0, 2), (1, 3), (2, 0), (3, 1)]);
        assert_eq!(layout.genes_rooted_at(Ty::S), 5);
        let mut genes: Vec<Vec<u32>> = layout.genes().iter().map(|l| vec![S; l.width]).collect();
        genes[0][0] = LEFT;
        layout.copy_gene(&mut genes, 0, 2).expect("homologous");
        assert_eq!(genes[2][0], LEFT);
        assert_eq!(layout.copy_gene(&mut genes, 0, 1), Err(LayoutDefect::NotHomologous));
        assert_eq!(layout.copy_gene(&mut genes, 4, 0), Err(LayoutDefect::NotHomologous), "a lone gene id has no pair");
        let solo = Layout::new(8, vec![g(2)]);
        assert!(solo.homologous_pairs().is_empty());
    }

    // -- the write-back ------------------------------------------------------

    /// Five-function one-type kingdom of the promotion example: `P(2) Q(1)
    /// Z(3) W(1) F(2)` and the terminals `a`.
    fn promo_kingdom() -> KingdomContract {
        let f = |n: &str, a: usize| (n.to_string(), vec![Ty::F; a], Ty::F);
        relations_from_rows("promo", vec![Ty::F], &[f("P", 2), f("Q", 1), f("Z", 3), f("W", 1), f("F", 2), f("a", 0)], &[]).check().expect("valid")
    }

    fn t(symbol: SymbolId, kids: Vec<Tree>) -> Tree {
        Tree { symbol, const_index: None, kids }
    }

    /// The design's example: `P(Q(Z(a,b,c)), W(F(u,v)))` has j = 5; promoting
    /// Q to Z gives j = 7, so its place in the domain depends on `h`.
    #[test]
    fn promotion_is_in_the_domain_exactly_when_the_promoted_tree_fits() {
        let k = promo_kingdom();
        let (p, q, z, w, f, a) = (0, 1, 2, 3, 4, 5);
        let a_ = || t(a, vec![]);
        let tree = t(p, vec![t(q, vec![t(z, vec![a_(), a_(), a_()])]), t(w, vec![t(f, vec![a_(), a_()])])]);
        assert_eq!(required_heads(&k, &tree), BTreeMap::from([(Ty::F, 5)]));
        let promoted = promote(&tree, 2, 0);
        assert_eq!(promoted, t(p, vec![t(z, vec![a_(), a_(), a_()]), t(w, vec![t(f, vec![a_(), a_()])])]));
        assert_eq!(required_heads(&k, &promoted), BTreeMap::from([(Ty::F, 7)]));
        for h in 5..12usize {
            let layout = GeneLayout::from_heads(&k, Ty::F, &heads(&[(Ty::F, h)])).expect("layout");
            assert!(fits(&k, &layout, &tree).is_ok(), "h = {h}: the original fits");
            let dom = promotions(&k, &layout, &tree);
            assert_eq!(dom.contains(&(2, 0)), h >= 7, "h = {h}: promoting Q to Z");
        }
        let small = GeneLayout::from_heads(&k, Ty::F, &heads(&[(Ty::F, 6)])).expect("layout");
        assert_eq!(fits(&k, &small, &promoted), Err(LayoutDefect::HeadTooShort { ty: Ty::F, need: 7, h: 6 }));
    }

    #[test]
    fn write_back_round_trips_through_the_walk_and_keeps_unread_rows() {
        let k = two_type();
        let layout = design_layout(&k);
        let tree = t(LEFT, vec![t(LEFT, vec![t(S, vec![]), t(Z_I, vec![])]), t(I, vec![])]);
        let previous = vec![S, S, Z_I, Z_S, I];
        let gene = write_back(&k, &layout, &previous, &tree).expect("encodes");
        assert_eq!(gene, vec![LEFT, LEFT, I, S, Z_I]);
        assert_eq!(express(&k, &layout, &gene, &no_constants()).expect("expresses").tree(), tree);
        // A smaller tree overwrites only the rows it reads.
        let small = t(LEFT, vec![t(S, vec![]), t(I, vec![])]);
        let g2 = write_back(&k, &layout, &gene, &small).expect("encodes");
        assert_eq!(g2, vec![LEFT, S, I, S, Z_I], "rows past the read ones keep their previous contents");
        assert_eq!(express(&k, &layout, &g2, &no_constants()).expect("expresses").tree(), small);
    }

    #[test]
    fn write_back_rebuilds_the_dc_in_expression_order() {
        let k = with_constant();
        let layout = GeneLayout::from_heads(&k, Ty::F, &heads(&[(Ty::F, 3)])).expect("layout");
        let q = |i| Tree { symbol: 2, const_index: Some(i), kids: vec![] };
        let tree = t(0, vec![q(3), t(0, vec![t(1, vec![]), q(1)])]);
        let gene = write_back(&k, &layout, &vec![1; layout.width], &tree).expect("encodes");
        let consts = BTreeMap::from([(Ty::F, 5)]);
        let e = express(&k, &layout, &gene, &consts).expect("expresses");
        assert_eq!(e.tree(), tree);
        assert_eq!(e.reach.q_used.get(&Ty::F), Some(&2));
    }

    #[test]
    fn write_back_defects_are_named() {
        let k = two_type();
        let layout = design_layout(&k);
        let prev = vec![S; layout.width];
        let deep = t(LEFT, vec![t(LEFT, vec![t(LEFT, vec![t(S, vec![]), t(I, vec![])]), t(I, vec![])]), t(I, vec![])]);
        assert_eq!(write_back(&k, &layout, &prev, &deep), Err(LayoutDefect::HeadTooShort { ty: Ty::S, need: 3, h: 2 }));
        let wrong = t(LEFT, vec![t(I, vec![]), t(I, vec![])]);
        assert_eq!(write_back(&k, &layout, &prev, &wrong), Err(LayoutDefect::TreeNotTyped { step: 2, want: Ty::S, got: Ty::I }));
        let arity = t(LEFT, vec![t(S, vec![])]);
        assert_eq!(write_back(&k, &layout, &prev, &arity), Err(LayoutDefect::TreeArity { step: 1, want: 2, got: 1 }));
        assert_eq!(write_back(&k, &layout, &prev[..4], &t(S, vec![])), Err(LayoutDefect::GeneLength { want: 5, got: 4 }));
    }
}
