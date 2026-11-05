//! The multi-typed GEP genome layout and its host expression
//! (`nucleotable/docs/MultiTypedGEP.md`, revision 4, sections 2 to 5).
//!
//! A gene has, for every type `T` its root reaches, a `head_T`, a `tail_T`,
//! (for a constant-capable type) a `Dc_T` and (for a type with a variable
//! terminal `var.T`, plan 7 task G2) a `Dv_T`. A slot requiring `T` reads the next
//! unused row of `T`'s frame, `head_T` then `tail_T`; `head_T` holds symbols
//! whose output is `T`, `tail_T` terminals whose output is `T`. The lengths
//! come from the kingdom's bound table ([`KingdomContract::bounds`]) for a
//! declared program size `N` or explicit `h_T`, and belong to the layout, never
//! to a gene record. Physical order is every head, then every tail, then every
//! Dc, then every Dv, each in frame order (the root's type first, then ascending
//! code), so a one-type gene is exactly the physical Karva head, tail and Dc, and
//! a kingdom with no `var.T` lays out exactly as before the Dv lane existed.
//!
//! `Dv_T` is a lane beside `head_T` and `tail_T`: word `r` of `Dv_T` belongs
//! to frame row `r` (the cell [`Frame::row_pos`] gives), whatever symbol that
//! cell holds, and is read only when the cell expresses as `var.T`. It is never
//! consumed in order as `Dc_T` is, so a `var.T` takes no `Dc_T` word and the Dc
//! lengths are unchanged. Every operator that moves a head or tail cell moves its
//! Dv word with it; an operator applied to a Dv span alone is a defect, since it
//! would part a word from its cell. An index word
//! is any `u32` below [`DV_WORDS`]; the decoder (G3) reads it modulo the
//! variables in scope, so no draw or edit needs the environment.
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

/// The four domains of a gene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Domain {
    Head,
    Tail,
    Dc,
    /// The variable terminals' index words, one a frame row of a type with a `var.T`.
    Dv,
}

/// An index word of `Dv_T` is any value below this (design 4, plan 7.1).
pub const DV_WORDS: u32 = 1 << 16;

/// The words [`GeneLayout::alphabet`] lists for a `Dv_T` cell, `0 .. DV_NEIGHBOURS`: the one-edit neighbours of an
/// index word. The decoder reads a word modulo the `c` variables in scope, so words `0 .. c` name every variable once
/// and these 64 reach every variable of any scope of up to 64, whatever the word held before; listing all
/// [`DV_WORDS`] would give 65,535 neighbours of which at most `c` are different programs. A random draw or a point
/// edit of a Dv word takes any word below [`DV_WORDS`] uniformly ([`GeneLayout::in_alphabet`] is the legal set).
pub const DV_NEIGHBOURS: u32 = 64;

impl Domain {
    pub fn name(self) -> &'static str {
        match self {
            Domain::Head => "head",
            Domain::Tail => "tail",
            Domain::Dc => "dc",
            Domain::Dv => "dv",
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
    /// Present for a type with a variable terminal `var.T`.
    pub dv_off: Option<usize>,
    /// `h_T + t_T` with a Dv; zero without.
    pub dv_len: usize,
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

    /// The gene position of the Dv word of frame row `k`, or `None` without a Dv or past its end.
    pub fn dv_pos(&self, k: usize) -> Option<usize> {
        self.dv_off.filter(|_| k < self.dv_len).map(|o| o + k)
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
            frames.push(Frame { ty: t.ty, head_off: off, head_len: t.h, tail_off: 0, tail_len: t.t, dc_off: None, dc_len: 0, dv_off: None, dv_len: 0 });
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
        for (f, t) in frames.iter_mut().zip(&b.types) {
            if t.variable {
                f.dv_off = Some(off);
                f.dv_len = t.dv();
                off += t.dv();
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

    /// Frame rows of the gene, `sum_T (h_T + t_T)`: every expressed node reads
    /// exactly one, so no gene expresses more nodes than this.
    pub fn frame_rows(&self) -> usize {
        self.frames.iter().map(|f| f.head_len + f.tail_len).sum()
    }

    /// The per-thread node scratch an evaluator needs for this layout, derived
    /// from its own bound ([`crate::gpu_eval::max_nodes_for_rows`]); never
    /// below `gpu_eval::MAX_NODES`.
    pub fn max_nodes(&self) -> usize {
        crate::gpu_eval::max_nodes_for_rows(self.frame_rows())
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
        for f in &self.frames {
            if let Some(d) = f.dv_off {
                v.push(Span { domain: Domain::Dv, ty: f.ty, off: d, len: f.dv_len });
            }
        }
        v.retain(|s| s.len > 0);
        v
    }

    /// The gene position of the Dv word parallel to the first cell of `s`, a head or tail span of a type with a Dv
    /// lane: the word of position `p` of `s` is at `lane + (p - s.off)`. `None` for any other span.
    pub fn lane(&self, s: &Span) -> Option<usize> {
        let f = self.frame(s.ty)?;
        let base = f.dv_off?;
        match s.domain {
            Domain::Head => Some(base),
            Domain::Tail => Some(base + f.head_len),
            Domain::Dc | Domain::Dv => None,
        }
    }

    /// The Dv word parallel to the head or tail cell at `pos`, if its type has a Dv lane.
    pub fn dv_of_cell(&self, pos: usize) -> Option<usize> {
        let s = self.spans().into_iter().find(|s| pos >= s.off && pos < s.off + s.len)?;
        self.lane(&s).map(|l| l + (pos - s.off))
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
    /// `symbol`, Dc rows a `dc_index` and Dv rows a `dv_index`, never a symbol.
    pub fn rows(&self, chromosome_id: u64, gene_id: u64, gene: &[u32]) -> Vec<GeneRow> {
        (0..self.width)
            .map(|pos| {
                let key = self.key(pos);
                let w = Some(gene[pos]);
                let (symbol, dc_index, dv_index) = match key.domain {
                    Domain::Head | Domain::Tail => (w, None, None),
                    Domain::Dc => (None, w, None),
                    Domain::Dv => (None, None, w),
                };
                GeneRow { chromosome_id, gene_id, key, symbol, dc_index, dv_index }
            })
            .collect()
    }

    /// The values a mutation may write at `pos`: any symbol outputting `T` in
    /// `head_T`, a terminal outputting `T` in `tail_T`, an index below `C_T` in
    /// `Dc_T`. Enumerated before the draw, so a drawn child is valid. For a
    /// `Dv_T` cell it lists the words below [`DV_NEIGHBOURS`] only, the bounded
    /// neighbours of an index word; every word below [`DV_WORDS`] is legal there
    /// ([`GeneLayout::in_alphabet`]), and a draw takes any of them.
    pub fn alphabet(&self, k: &KingdomContract, pos: usize, const_len: &BTreeMap<Ty, u32>) -> Vec<u32> {
        let key = self.key(pos);
        let t = key.domain_type;
        match key.domain {
            Domain::Head => (0..k.n_symbols() as SymbolId).filter(|s| k.output(*s) == t).collect(),
            Domain::Tail => k.terminals_of(t).to_vec(),
            Domain::Dc => (0..const_len.get(&t).copied().unwrap_or(0)).collect(),
            Domain::Dv => (0..DV_NEIGHBOURS).collect(),
        }
    }

    /// Whether `value` is in the alphabet of the cell at `pos` ([`GeneLayout::alphabet`] without enumerating it); for a
    /// `Dv_T` cell, whether it is a legal index word (below [`DV_WORDS`], a superset of the listed neighbours).
    pub fn in_alphabet(&self, k: &KingdomContract, pos: usize, value: u32, const_len: &BTreeMap<Ty, u32>) -> bool {
        let key = self.key(pos);
        let t = key.domain_type;
        match key.domain {
            Domain::Head => (value as usize) < k.n_symbols() && k.output(value) == t,
            Domain::Tail => k.terminals_of(t).contains(&value),
            Domain::Dc => value < const_len.get(&t).copied().unwrap_or(0),
            Domain::Dv => value < DV_WORDS,
        }
    }

    /// The span holding `[lo, hi)`, if one does.
    fn span_of(&self, lo: usize, hi: usize) -> Result<Span, LayoutDefect> {
        self.spans().into_iter().find(|s| lo >= s.off && hi <= s.off + s.len && lo < hi).ok_or(LayoutDefect::RangeAcrossDomains { lo, hi })
    }

    /// The span holding `[lo, hi)` for an operator that MOVES cells: a Dv span alone is a defect (its words would part
    /// from their cells), and the result carries the Dv lane parallel to the range, if the span has one.
    fn moving_span(&self, lo: usize, hi: usize) -> Result<(Span, Option<usize>), LayoutDefect> {
        let s = self.span_of(lo, hi)?;
        if s.domain == Domain::Dv {
            return Err(LayoutDefect::LaneMovedAlone { lo, hi });
        }
        Ok((s, self.lane(&s).map(|l| l + (lo - s.off))))
    }

    /// Point mutation: `value` must be in the alphabet of the cell. An edit of a Dv word changes no symbol, and an
    /// edit of a cell leaves its Dv word.
    pub fn mutate(&self, k: &KingdomContract, gene: &mut [u32], pos: usize, value: u32, const_len: &BTreeMap<Ty, u32>) -> Result<(), LayoutDefect> {
        if !self.in_alphabet(k, pos, value, const_len) {
            return Err(LayoutDefect::NotInAlphabet { pos, value });
        }
        gene[pos] = value;
        Ok(())
    }

    /// Homologous recombination: exchange `[lo, hi)` between two genes, with the cells' Dv words. Both
    /// layouts must be homologous and the range inside one domain of one type.
    pub fn swap_range(&self, other: &GeneLayout, a: &mut [u32], b: &mut [u32], lo: usize, hi: usize) -> Result<(), LayoutDefect> {
        if !self.homologous(other) {
            return Err(LayoutDefect::NotHomologous);
        }
        let (_, lane) = self.moving_span(lo, hi)?;
        a[lo..hi].swap_with_slice(&mut b[lo..hi]);
        if let Some(l) = lane {
            a[l..l + hi - lo].swap_with_slice(&mut b[l..l + hi - lo]);
        }
        Ok(())
    }

    /// Inversion of `[lo, hi)`, inside one domain of one type, the cells' Dv words with them.
    pub fn invert(&self, gene: &mut [u32], lo: usize, hi: usize) -> Result<(), LayoutDefect> {
        let (_, lane) = self.moving_span(lo, hi)?;
        gene[lo..hi].reverse();
        if let Some(l) = lane {
            gene[l..l + hi - lo].reverse();
        }
        Ok(())
    }

    /// A row operator applied to the Dv table moves words without their cells: a defect whatever the words hold, so
    /// it is reported before any column is looked at (rows `[lo, hi)` of the first Dv column name the range).
    fn not_dv(&self, domain: Domain, lo: usize, hi: usize) -> Result<(), LayoutDefect> {
        match self.columns(Domain::Dv).first() {
            Some(c) if domain == Domain::Dv => Err(LayoutDefect::LaneMovedAlone { lo: c.off + lo, hi: c.off + hi }),
            _ => Ok(()),
        }
    }

    /// Whether reversing `[lo, hi)` (with its Dv words) would change the gene.
    fn reversal_changes(&self, gene: &[u32], s: &Span, lo: usize, hi: usize) -> bool {
        let differs = |cells: &[u32]| cells.iter().zip(cells.iter().rev()).any(|(a, b)| a != b);
        differs(&gene[lo..hi]) || self.lane(s).is_some_and(|l| differs(&gene[l + (lo - s.off)..l + (hi - s.off)]))
    }

    /// GEP's insertion-sequence transposition inside one domain of one type: the run `[src, src + len)` is copied
    /// to `dst`, the cells from `dst` on shift towards the span's end and the overflow falls off it. The run and
    /// `dst` must lie in one span. The cells' Dv words move with them. Returns whether a cell or word changed.
    pub fn transpose(&self, gene: &mut [u32], src: usize, len: usize, dst: usize) -> Result<bool, LayoutDefect> {
        let (s, _) = self.moving_span(src, src + len)?;
        if dst < s.off || dst >= s.off + s.len {
            return Err(LayoutDefect::RangeAcrossDomains { lo: dst, hi: dst + 1 });
        }
        let shift = |gene: &mut [u32], base: usize| {
            let before = gene[base..base + s.len].to_vec();
            let mut span = before.clone();
            let run: Vec<u32> = gene[base + (src - s.off)..base + (src - s.off) + len].to_vec();
            span.splice(dst - s.off..dst - s.off, run);
            span.truncate(s.len);
            gene[base..base + s.len].copy_from_slice(&span);
            span != before
        };
        let cells = shift(gene, s.off);
        let words = self.lane(&s).is_some_and(|l| shift(gene, l));
        Ok(cells || words)
    }

    // ---- THE ROW TABLE (Andrew, 2026-11-01): a domain of the gene is a table, row = position, column = type ----
    //
    // Column `T` of domain `D` is `D_T`; its row `r` is the cell `off_T + r` while `r < len_T`. The columns are
    // ragged: a column takes part in a row only where it has that row. Every row operator below moves a symbol only
    // within its own column, so a child is legal by construction, and with one type it is the one-type operator.

    /// The columns of `domain`'s table, in physical order: one non-empty span a type.
    pub fn columns(&self, domain: Domain) -> Vec<Span> {
        self.spans().into_iter().filter(|s| s.domain == domain).collect()
    }

    /// The rows of `domain`'s table: its longest column.
    pub fn domain_rows(&self, domain: Domain) -> usize {
        self.columns(domain).iter().map(|s| s.len).max().unwrap_or(0)
    }

    /// INVERSION, row-wise: rows `[lo, hi)` reversed in every column of `domain`, each column over the part of
    /// `[lo, hi)` it has. Returns the columns that changed.
    pub fn invert_rows(&self, gene: &mut [u32], domain: Domain, lo: usize, hi: usize) -> Result<usize, LayoutDefect> {
        let rows = self.domain_rows(domain);
        self.not_dv(domain, lo, hi)?;
        if lo >= hi || hi > rows {
            return Err(LayoutDefect::RowsOutsideTable { domain: Some(domain), lo, hi, rows });
        }
        let mut changed = 0;
        for c in self.columns(domain) {
            let h = hi.min(c.len);
            if h <= lo + 1 {
                continue;
            }
            if self.reversal_changes(gene, &c, c.off + lo, c.off + h) {
                self.invert(gene, c.off + lo, c.off + h)?;
                changed += 1;
            }
        }
        Ok(changed)
    }

    /// TRANSPOSITION, row-wise (GEP's IS; RIS with `dst = 0` and `src` from [`GeneLayout::ris_row`]): in every
    /// column of `domain`, the run of rows `[src, src + len)` (the part the column has) is copied to row `dst`, the
    /// rows from `dst` on shift down and the overflow falls off the column's end ([`GeneLayout::transpose`]). A
    /// column without row `src` or row `dst` is unchanged. Returns the columns that changed.
    pub fn transpose_rows(&self, gene: &mut [u32], domain: Domain, src: usize, len: usize, dst: usize) -> Result<usize, LayoutDefect> {
        let rows = self.domain_rows(domain);
        if len == 0 || src >= rows || dst >= rows {
            return Err(LayoutDefect::RowsOutsideTable { domain: Some(domain), lo: src.min(dst), hi: src.max(dst) + len.max(1), rows });
        }
        let mut changed = 0;
        for c in self.columns(domain) {
            if src >= c.len || dst >= c.len {
                continue;
            }
            let run = len.min(c.len - src);
            changed += usize::from(self.transpose(gene, c.off + src, run, c.off + dst)?);
        }
        Ok(changed)
    }

    /// RIS's start: the first row at or after `from` where the ROOT's head column holds a function, if one does.
    pub fn ris_row(&self, k: &KingdomContract, gene: &[u32], from: usize) -> Option<usize> {
        let f = self.frame(self.root)?;
        (from..f.head_len).find(|r| !k.signature(gene[f.head_off + r]).is_empty())
    }

    /// The rows of the WHOLE gene's table: the longest column, a column of type `T` being `head_T`, then `tail_T`,
    /// then `Dc_T` ([`GeneLayout::column_pos`]).
    pub fn gene_rows(&self) -> usize {
        self.frames.iter().map(|f| f.head_len + f.tail_len + f.dc_len).max().unwrap_or(0)
    }

    /// The gene position of row `r` of `f`'s whole column (`head_T`, `tail_T`, `Dc_T`), or `None` past its end.
    pub fn column_pos(f: &Frame, r: usize) -> Option<usize> {
        match f.row_pos(r) {
            Some(p) => Some(p),
            None => {
                let d = r - f.head_len - f.tail_len;
                f.dc_off.filter(|_| d < f.dc_len).map(|o| o + d)
            }
        }
    }

    /// CROSSOVER, row-wise: GEP's one-point (`cut2 = None`: rows `cut1..`) or two-point (rows `[cut1, cut2)`)
    /// recombination at row indices of the whole gene's table, the rows exchanged between `a` and `b` in every
    /// column. A column is `head_T`, `tail_T`, `Dc_T` in that order, so a cut spans head and tail (and Dc) together,
    /// as the classic recombination does over the linear gene; with one type the column IS the linear gene and this
    /// is the classic operator. A head or tail row's Dv word goes with it. Both layouts must be homologous. Returns
    /// the cells exchanged, Dv words not counted.
    pub fn crossover_rows(&self, other: &GeneLayout, a: &mut [u32], b: &mut [u32], cut1: usize, cut2: Option<usize>) -> Result<usize, LayoutDefect> {
        if !self.homologous(other) {
            return Err(LayoutDefect::NotHomologous);
        }
        let rows = self.gene_rows();
        let hi = cut2.unwrap_or(rows);
        if cut1 >= hi || hi > rows {
            return Err(LayoutDefect::RowsOutsideTable { domain: None, lo: cut1, hi, rows });
        }
        let mut cells = 0;
        for f in &self.frames {
            for r in cut1..hi {
                match Self::column_pos(f, r) {
                    Some(p) => {
                        std::mem::swap(&mut a[p], &mut b[p]);
                        if let Some(d) = f.dv_pos(r) {
                            std::mem::swap(&mut a[d], &mut b[d]);
                        }
                        cells += 1;
                    }
                    None => break,
                }
            }
        }
        Ok(cells)
    }

    // ---- THE RELATIVE ROWS (2026-11-01): every column takes part in every application ----
    //
    // A fraction `f` of a column is a fixed-point number of `FRAC_ONE` parts; column `T` applies it at its own row
    // `floor(f * len_T)` ([`frac_row`]). With one type, [`frac_of_row`] gives the fraction of any row exactly, so each
    // relative operator is the absolute one, and so the one-type operator.

    /// RELATIVE INVERSION: in every column of `domain`, its rows `[frac_row(lo, len), frac_row(hi, len))` reversed.
    /// `lo < hi <= FRAC_ONE`. Returns the columns that changed.
    pub fn invert_rel(&self, gene: &mut [u32], domain: Domain, lo: u64, hi: u64) -> Result<usize, LayoutDefect> {
        if lo >= hi || hi > FRAC_ONE {
            return Err(LayoutDefect::FractionOutsideUnit { lo, hi });
        }
        if let Some(c) = self.columns(Domain::Dv).first().filter(|_| domain == Domain::Dv) {
            return Err(LayoutDefect::LaneMovedAlone { lo: c.off + frac_row(lo, c.len), hi: c.off + frac_row(hi, c.len) });
        }
        let mut changed = 0;
        for c in self.columns(domain) {
            let (l, h) = (frac_row(lo, c.len), frac_row(hi, c.len));
            if h <= l + 1 {
                continue;
            }
            if self.reversal_changes(gene, &c, c.off + l, c.off + h) {
                self.invert(gene, c.off + l, c.off + h)?;
                changed += 1;
            }
        }
        Ok(changed)
    }

    /// RELATIVE TRANSPOSITION (IS; RIS with `dst = 0` and `src` the fraction of [`GeneLayout::ris_row`]'s row in the
    /// root's head column, [`frac_of_row`]): in every column of `domain`, the run of rows from `frac_row(src, len)` to
    /// `frac_row(end, len)` (at least one row, at most to the column's end) copied to row `frac_row(dst, len)`, the
    /// rest shifted down and the overflow off the column's end. `src, dst < FRAC_ONE`, `src <= end <= FRAC_ONE`.
    /// Returns the columns that changed.
    pub fn transpose_rel(&self, gene: &mut [u32], domain: Domain, src: u64, end: u64, dst: u64) -> Result<usize, LayoutDefect> {
        if src >= FRAC_ONE || dst >= FRAC_ONE || end < src || end > FRAC_ONE {
            return Err(LayoutDefect::FractionOutsideUnit { lo: src.min(dst), hi: end.max(dst) });
        }
        let mut changed = 0;
        for c in self.columns(domain) {
            let (s, d) = (frac_row(src, c.len), frac_row(dst, c.len));
            let e = frac_row(end, c.len).clamp(s + 1, c.len);
            changed += usize::from(self.transpose(gene, c.off + s, e - s, c.off + d)?);
        }
        Ok(changed)
    }

    /// RELATIVE CROSSOVER: one-point (`cut2 = None`) or two-point, each column of the whole gene (`head_T`, `tail_T`,
    /// `Dc_T`, as [`GeneLayout::crossover_rows`]) exchanging its rows `[frac_row(cut1, len), frac_row(cut2, len))`
    /// (to its end for one-point), a head or tail row's Dv word with it. `cut1 < cut2 <= FRAC_ONE`. Returns the cells
    /// exchanged, Dv words not counted.
    pub fn crossover_rel(&self, other: &GeneLayout, a: &mut [u32], b: &mut [u32], cut1: u64, cut2: Option<u64>) -> Result<usize, LayoutDefect> {
        if !self.homologous(other) {
            return Err(LayoutDefect::NotHomologous);
        }
        let hi = cut2.unwrap_or(FRAC_ONE);
        if cut1 >= hi || hi > FRAC_ONE {
            return Err(LayoutDefect::FractionOutsideUnit { lo: cut1, hi });
        }
        let mut cells = 0;
        for f in &self.frames {
            let len = f.head_len + f.tail_len + f.dc_len;
            for r in frac_row(cut1, len)..frac_row(hi, len) {
                let p = Self::column_pos(f, r).expect("a row below the column's length");
                std::mem::swap(&mut a[p], &mut b[p]);
                if let Some(d) = f.dv_pos(r) {
                    std::mem::swap(&mut a[d], &mut b[d]);
                }
                cells += 1;
            }
        }
        Ok(cells)
    }
}

/// The fixed-point one of a relative row: a fraction is `f / FRAC_ONE`.
pub const FRAC_ONE: u64 = 1 << 32;

/// The row of a column of `len` rows at fraction `f`: `floor(f * len / FRAC_ONE)`, exact in integers.
pub fn frac_row(f: u64, len: usize) -> usize {
    ((u128::from(f) * len as u128) >> 32) as usize
}

/// The smallest fraction whose row in a column of `len` rows is `r` (`r <= len`): `ceil(r * FRAC_ONE / len)`, so
/// `frac_row(frac_of_row(r, len), len) == r`.
pub fn frac_of_row(r: usize, len: usize) -> u64 {
    ((r as u128 * u128::from(FRAC_ONE)).div_ceil(len as u128)) as u64
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
    /// A Dv row's index word.
    pub dv_index: Option<u32>,
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
    /// A row operator's rows `[lo, hi)` outside a table of `rows` rows (`domain`'s, or the whole gene's, `None`, for
    /// a crossover), or empty.
    RowsOutsideTable { domain: Option<Domain>, lo: usize, hi: usize, rows: usize },
    /// A relative row operator's fractions (parts of `FRAC_ONE`) out of order or past one.
    FractionOutsideUnit { lo: u64, hi: u64 },
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
    /// A `var.T` node with no index word.
    MissingVariableIndex { step: usize },
    /// A `var.T` of a type whose frame has no Dv lane.
    NoDvLane { ty: Ty },
    /// A Dv word at or past [`DV_WORDS`].
    DvWordPastRange { pos: usize, word: u32 },
    /// An operator that moves cells applied to a Dv span alone, which would part its words from their cells.
    LaneMovedAlone { lo: usize, hi: usize },
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
            LayoutDefect::RowsOutsideTable { domain, lo, hi, rows } => write!(f, "rows [{lo}, {hi}) are empty or outside the {} table of {rows} rows", domain.map_or("gene", Domain::name)),
            LayoutDefect::FractionOutsideUnit { lo, hi } => write!(f, "fractions {lo} and {hi} (of {FRAC_ONE}) are out of order or past one"),
            LayoutDefect::TreeNotTyped { step, want, got } => write!(f, "tree node {step} fills a slot of type {want:?} and outputs {got:?}"),
            LayoutDefect::TreeArity { step, want, got } => write!(f, "tree node {step} has {got} children, its symbol takes {want}"),
            LayoutDefect::HeadTooShort { ty, need, h } => write!(f, "the tree needs {need} head rows of type {ty:?}, h_T is {h}"),
            LayoutDefect::TailTooShort { ty, need, rows } => write!(f, "the tree has {need} nodes of type {ty:?}, head and tail hold {rows}"),
            LayoutDefect::DcTooShort { ty, need, d } => write!(f, "the tree has {need} placeholders of type {ty:?}, Dc holds {d}"),
            LayoutDefect::MissingConstIndex { step } => write!(f, "tree node {step} is a constant placeholder with no C_T index"),
            LayoutDefect::MissingVariableIndex { step } => write!(f, "tree node {step} is a variable terminal with no index word"),
            LayoutDefect::NoDvLane { ty } => write!(f, "a variable terminal of type {ty:?}, whose frame has no Dv lane"),
            LayoutDefect::DvWordPastRange { pos, word } => write!(f, "the Dv word at {pos} is {word}, at or past {DV_WORDS}"),
            LayoutDefect::LaneMovedAlone { lo, hi } => write!(f, "range [{lo}, {hi}) is a Dv lane moved without its cells"),
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
    /// The `C_T` index an expressed `?_T` read from its Dc, or the index word an expressed `var.T` read from its
    /// Dv (the symbol's [`KingdomContract::is_variable`] says which).
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tree {
    pub symbol: SymbolId,
    /// A `?_T`'s `C_T` index, or a `var.T`'s Dv index word ([`KingdomContract::is_variable`]); `None` otherwise.
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
                Domain::Dv => {
                    if *w >= DV_WORDS {
                        return Err(LayoutDefect::DvWordPastRange { pos, word: *w });
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
        } else if k.is_variable(symbol) {
            Some(gene[frame.dv_pos(frame_k).ok_or(LayoutDefect::NoDvLane { ty: slot.ty })?])
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

/// THE EXPRESSED NODES WALKED WITHOUT BUILDING THEM: [`express`]'s FIFO walk over the same cells, in the same order,
/// with the same defects, that keeps only each slot's type. It calls `visit` with each node's symbol and, for a
/// constant, its Dc word, and for a variable terminal, its Dv word (the two words that decide the node), and stops once it has walked `limit + 1` nodes. Returns
/// the nodes walked: the expressed tree's node count when that is at most `limit`, else `limit + 1`. A size guard
/// counts with it before any tree or text is built (`FutharkEvaluator`'s size bound, the F3a code review), and keys
/// an oversized gene by a hash of every node's words (`limit` = `usize::MAX - 1`).
pub fn walk_expressed(k: &KingdomContract, layout: &GeneLayout, gene: &[u32], const_len: &BTreeMap<Ty, u32>, limit: usize, visit: &mut dyn FnMut(SymbolId, Option<u32>)) -> Result<usize, LayoutDefect> {
    validate(k, layout, gene, const_len)?;
    let mut k_t: BTreeMap<Ty, usize> = BTreeMap::new();
    let mut q_t: BTreeMap<Ty, usize> = BTreeMap::new();
    let mut queue = VecDeque::from([layout.root]);
    let mut walked = 0usize;
    while let Some(ty) = queue.pop_front() {
        if walked > limit {
            break;
        }
        let frame = layout.frame(ty).ok_or(LayoutDefect::NoFrame { ty })?;
        let cursor = k_t.entry(ty).or_insert(0);
        let frame_k = *cursor;
        *cursor += 1;
        let pos = frame.row_pos(frame_k).ok_or(LayoutDefect::TailRanOut { ty, cursor: frame_k })?;
        let symbol = gene[pos];
        let const_word = if k.is_constant(symbol) {
            let q = *q_t.entry(ty).or_default();
            let dc_off = frame.dc_off.ok_or(LayoutDefect::DcRanOut { ty, q })?;
            if q >= frame.dc_len {
                return Err(LayoutDefect::DcRanOut { ty, q });
            }
            q_t.insert(ty, q + 1);
            Some(gene[dc_off + q])
        } else if k.is_variable(symbol) {
            Some(gene[frame.dv_pos(frame_k).ok_or(LayoutDefect::NoDvLane { ty })?])
        } else {
            None
        };
        walked += 1;
        visit(symbol, const_word);
        queue.extend(k.signature(symbol).iter().copied());
    }
    Ok(walked)
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
        let row = *r;
        let pos = f.row_pos(row).ok_or(LayoutDefect::TailTooShort { ty, need: row + 1, rows: f.head_len + f.tail_len })?;
        *r += 1;
        cells.push((pos, t.symbol));
        if k.is_variable(t.symbol) {
            let w = t.const_index.ok_or(LayoutDefect::MissingVariableIndex { step: i + 1 })?;
            let d = f.dv_pos(row).ok_or(LayoutDefect::NoDvLane { ty })?;
            if w >= DV_WORDS {
                return Err(LayoutDefect::DvWordPastRange { pos: d, word: w });
            }
            cells.push((d, w));
        }
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
/// the rest `tail_T`; `Dc_T` is rebuilt in the order the `?_T` appear; a `var.T`'s word goes to its row of `Dv_T`; rows the
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

    // -- every existing kingdom lays out and expresses as before the Dv lane (plan 7 task G2) ----------------------

    /// FNV-1a over 64-bit words: a fingerprint that is the same in every process.
    fn fnv(h: &mut u64, w: u64) {
        for b in w.to_le_bytes() {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }

    /// A fingerprint of a kingdom's layouts (every root at N = 1, 8, 32: width and every frame's offsets and lengths)
    /// and of 200 random genes a root at N = 8 expressed (every node's symbol, word and position). The golden values
    /// below were taken on the code before the Dv lane existed (fuller e526a62f). Together the two tests below cover
    /// 22 kingdoms: 19 in a default build, the three WGSL ones with `--features wgsl`.
    fn layout_and_words_fingerprint(r: &crate::kingdom_contract::KingdomRelations, const_len: &BTreeMap<Ty, u32>) -> u64 {
        let k = r.check().unwrap_or_else(|d| panic!("{}: {:?}", r.kingdom, d.first()));
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut rng = 0x5eed_u64;
        let mut next = move |n: usize| {
            rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) % n as u64) as usize
        };
        for root in k.roots().to_vec() {
            for n in [1usize, 8, 32] {
                let l = GeneLayout::from_program_size(&k, root, n).unwrap_or_else(|e| panic!("{}: {e}", r.kingdom));
                for w in [u64::from(root.code()), n as u64, l.width as u64] {
                    fnv(&mut h, w);
                }
                for f in &l.frames {
                    assert_eq!((f.dv_off, f.dv_len), (None, 0), "{}: no Dv lane", r.kingdom);
                    for w in [u64::from(f.ty.code()), f.head_off as u64, f.head_len as u64, f.tail_off as u64, f.tail_len as u64, f.dc_off.map_or(u64::MAX, |o| o as u64), f.dc_len as u64] {
                        fnv(&mut h, w);
                    }
                }
                if n != 8 {
                    continue;
                }
                for _ in 0..200 {
                    let mut gene = vec![0u32; l.width];
                    for s in l.spans() {
                        for cell in &mut gene[s.off..s.off + s.len] {
                            *cell = match s.domain {
                                Domain::Head => {
                                    let (f, t) = (k.functions_of(s.ty), k.terminals_of(s.ty));
                                    let i = next(f.len() + t.len());
                                    if i < f.len() {
                                        f[i]
                                    } else {
                                        t[i - f.len()]
                                    }
                                }
                                Domain::Tail => k.terminals_of(s.ty)[next(k.terminals_of(s.ty).len())],
                                Domain::Dc => next(const_len.get(&s.ty).copied().unwrap_or(1).max(1) as usize) as u32,
                                Domain::Dv => panic!("{}: an existing kingdom has a Dv lane", r.kingdom),
                            };
                        }
                    }
                    let e = express(&k, &l, &gene, const_len).unwrap_or_else(|d| panic!("{}: {d}", r.kingdom));
                    for node in &e.nodes {
                        for w in [u64::from(node.symbol), node.const_index.map_or(u64::MAX, u64::from), node.pos as u64] {
                            fnv(&mut h, w);
                        }
                    }
                }
            }
        }
        h
    }

    fn consts_of(r: &crate::kingdom_contract::KingdomRelations) -> BTreeMap<Ty, u32> {
        let k = r.check().expect("checks");
        k.types().iter().filter(|t| k.constant_capable(**t)).map(|t| (*t, 5)).collect()
    }

    #[test]
    fn every_existing_kingdom_lays_out_and_expresses_word_for_word_as_before_the_dv_lane() {
        use crate::kingdom_contract::{regex_relations, sr_relations, tsr_relations};
        let mut kingdoms: Vec<(String, crate::kingdom_contract::KingdomRelations)> = vec![("sr".into(), sr_relations()), ("regex".into(), regex_relations())];
        kingdoms.push(("typed regex".into(), crate::regex_typed::typed_regex_relations(false)));
        kingdoms.push(("typed regex, pattern literal".into(), crate::regex_typed::typed_regex_relations(true)));
        for v in [crate::tsr_kingdom::TsrVariant::Single, crate::tsr_kingdom::TsrVariant::Order, crate::tsr_kingdom::TsrVariant::Depth, crate::tsr_kingdom::TsrVariant::OrderDepth] {
            kingdoms.push((format!("tsr {}", v.dir()), tsr_relations(v, 3).expect("TSR tables load")));
        }
        kingdoms.push(("futhark".into(), crate::futhark_table::FutharkKingdom::load().expect("Futhark kingdom").relations()));
        kingdoms.push(("futhark 1-D".into(), crate::futhark_table::FutharkKingdom::load_1d().expect("1-D kingdom").relations()));
        for inst in [crate::futhark_sig::REDUCE, crate::futhark_sig::SCAN, crate::futhark_sig::HISTOGRAM, crate::futhark_sig::PATHFINDER, crate::futhark_sig::A7] {
            kingdoms.push((format!("futhark view {}", inst.name), crate::futhark_sig::load(&inst).expect("view loads").relations()));
        }
        for inst in [crate::futhark_pixel::VOLUMEPASSAGE, crate::futhark_pixel::CUNEUS, crate::futhark_pixel::IMAGE] {
            kingdoms.push((format!("futhark pixel {}", inst.name), crate::futhark_pixel::load(&inst).expect("pixel view loads").relations()));
        }
        #[cfg(feature = "wgsl")]
        {
            kingdoms.push(("wgsl".into(), crate::kingdom_contract::wgsl_relations()));
            kingdoms.push(("safewgsl".into(), crate::kingdom_contract::safewgsl_relations()));
            let t = crate::wgsl::true_kingdom::TrueKingdom::load(&crate::wgsl::true_kingdom::default_dir()).expect("true WGSL table");
            kingdoms.push(("true wgsl".into(), t.relations()));
        }
        let golden: BTreeMap<&str, u64> = BTreeMap::from([
            ("sr", 0x5fea_40a2_cad6_8e36),
            ("regex", 0x7fa3_1d59_5356_0275),
            ("typed regex", 0x41d5_f0a6_e494_878a),
            ("typed regex, pattern literal", 0x9839_d2a6_1760_f1a1),
            ("tsr single", 0x6be9_289b_4aa6_c90b),
            ("tsr order", 0x63b6_0c17_542c_37b9),
            ("tsr depth", 0x8890_feef_ab56_ab14),
            ("tsr order_depth", 0x5872_1638_c12a_7f88),
            ("futhark", 0x6c96_d430_8287_1fa2),
            ("futhark 1-D", 0x4d21_23d5_b0e4_8a3c),
            ("futhark view reduce", 0xfd25_3707_45fa_2ed3),
            ("futhark view scan", 0xd334_75fe_624f_a773),
            ("futhark view histogram", 0x3d88_a218_edc9_6378),
            ("futhark view pathfinder", 0x143c_67e8_b7d1_73ea),
            // Added by the G2 code review's finding 7, each taken at e526a62f (the code before the Dv lane) by a
            // copy of this fingerprint run there; the 14 above reproduced in that run. A7's view is the matrix
            // kingdom's table, so it fingerprints as "futhark".
            ("futhark view a7", 0x6c96_d430_8287_1fa2),
            ("futhark pixel volumepassage", 0xe842_9cbb_b21a_e29f),
            ("futhark pixel cuneus", 0x1172_991c_3c36_edd0),
            ("futhark pixel image", 0x0ed5_4ee4_76aa_24d4),
            ("wgsl", 0xaca5_c7c1_66fa_1914),
            ("safewgsl", 0x0513_9731_15ba_6f78),
            ("true wgsl", 0x4119_d902_a24d_25aa),
        ]);
        let mut wrong = Vec::new();
        for (name, r) in &kingdoms {
            let k = r.check().expect("checks");
            assert!(k.types().iter().all(|t| !k.variable_capable(*t)), "{name}: no existing kingdom has a variable terminal");
            let got = layout_and_words_fingerprint(r, &consts_of(r));
            if golden.get(name.as_str()) != Some(&got) {
                wrong.push(format!("{name}: 0x{got:016x}"));
            }
        }
        assert!(wrong.is_empty(), "layouts or expressed words changed: {wrong:?}");
    }

    /// The k-means view (14,359 types) as the test above, in a test of its own because it is slow (about 95 s in a
    /// release build): not ignored, timed, its seconds printed, as the repository runs its other k-means checks.
    /// Golden taken at e526a62f, as the five added above.
    #[test]
    fn the_kmeans_view_lays_out_and_expresses_word_for_word_as_before_the_dv_lane() {
        let t = std::time::Instant::now();
        let r = crate::futhark_sig::load(&crate::futhark_sig::KMEANS).expect("view loads").relations();
        let k = r.check().expect("checks");
        assert!(k.types().iter().all(|t| !k.variable_capable(*t)), "k-means: no variable terminal");
        let got = layout_and_words_fingerprint(&r, &consts_of(&r));
        eprintln!("the k-means view's fingerprint: {:.1} s", t.elapsed().as_secs_f64());
        assert_eq!(got, 0x9aaf_6e28_47c5_32ea, "the k-means view's layouts or expressed words changed: 0x{got:016x}");
    }

    // -- the variable terminal `var.T` and its Dv lane (plan 7 task G2) ------------------------------------------

    /// The toy kingdom of G2: `pair:S(S,I)`, `inc:I(I)`, terminals `s:S`, `var.S:S`, `i:I`, `var.I:I`, and the
    /// constant placeholder `?:I`. `variables` false gives the same rows with no variable terminal.
    fn toy_relations(variables: bool) -> crate::kingdom_contract::KingdomRelations {
        let r = relations_from_rows(
            "toy var.T",
            vec![Ty::S],
            &[
                ("pair".to_string(), vec![Ty::S, Ty::I], Ty::S),
                ("inc".to_string(), vec![Ty::I], Ty::I),
                ("s".to_string(), vec![], Ty::S),
                ("var.S".to_string(), vec![], Ty::S),
                ("i".to_string(), vec![], Ty::I),
                ("var.I".to_string(), vec![], Ty::I),
                ("?".to_string(), vec![], Ty::I),
            ],
            &["?"],
        );
        if variables {
            r.with_variables(&["var.S", "var.I"]).expect("two terminals")
        } else {
            r
        }
    }
    const PAIR: SymbolId = 0;
    const INC: SymbolId = 1;
    const SS: SymbolId = 2;
    const VAR_S: SymbolId = 3;
    const II: SymbolId = 4;
    const VAR_I: SymbolId = 5;
    const QI: SymbolId = 6;

    fn toy() -> KingdomContract {
        toy_relations(true).check().expect("the toy kingdom checks")
    }

    fn toy_consts() -> BTreeMap<Ty, u32> {
        BTreeMap::from([(Ty::I, 4)])
    }

    fn leaf_w(symbol: SymbolId, w: u32) -> Tree {
        Tree { symbol, const_index: Some(w), kids: vec![] }
    }

    /// A gene of `l` drawn valid by layout from `seed`, every Dv word a distinct tag (its own position), so a test can
    /// tell which cell a word came from.
    fn toy_gene(k: &KingdomContract, l: &GeneLayout, seed: u64, tag_base: u32) -> Vec<u32> {
        let mut state = seed;
        let mut next = |n: usize| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) % n as u64) as usize
        };
        let mut gene = vec![0u32; l.width];
        for s in l.spans() {
            for (p, cell) in gene.iter_mut().enumerate().skip(s.off).take(s.len) {
                *cell = match s.domain {
                    Domain::Head => {
                        let a = l.alphabet(k, p, &toy_consts());
                        a[next(a.len())]
                    }
                    Domain::Tail => k.terminals_of(s.ty)[next(k.terminals_of(s.ty).len())],
                    Domain::Dc => next(4) as u32,
                    Domain::Dv => tag_base + p as u32,
                };
            }
        }
        gene
    }

    #[test]
    fn the_contract_flags_variable_terminals_apart_from_constants() {
        let k = toy();
        assert!(k.is_variable(VAR_S) && k.is_variable(VAR_I) && !k.is_constant(VAR_I) && !k.is_variable(QI) && k.is_constant(QI));
        assert!(k.takes_word(VAR_I) && k.takes_word(QI) && !k.takes_word(II) && !k.takes_word(SS) && !k.takes_word(PAIR));
        assert!(k.variable_capable(Ty::S) && k.variable_capable(Ty::I));
        assert_eq!((k.variable_of(Ty::S), k.variable_of(Ty::I)), (Some(VAR_S), Some(VAR_I)));
        assert!(!k.constant_capable(Ty::S) && k.constant_capable(Ty::I));
        let plain = toy_relations(false).check().expect("checks");
        assert!(!plain.variable_capable(Ty::S) && plain.variable_of(Ty::I).is_none());
    }

    #[test]
    fn a_variable_terminal_that_is_not_a_plain_terminal_is_a_defect() {
        let defect = |r: crate::kingdom_contract::KingdomRelations, needle: &str| {
            let d = r.check().expect_err("defect");
            assert!(d.iter().all(|m| m.starts_with("our software has a bug: ")) && d.iter().any(|m| m.contains(needle)), "{needle}: {d:?}");
        };
        let mut with_args = toy_relations(true);
        with_args.variables.insert(PAIR);
        defect(with_args, "variable terminal pair takes arguments");
        let mut both = toy_relations(true);
        both.variables.insert(QI);
        defect(both, "both a constant placeholder and a variable terminal");
        let mut two = toy_relations(true);
        two.variables.insert(II);
        defect(two, "has two variable terminals");
        let mut unknown = toy_relations(true);
        unknown.variables.insert(99);
        defect(unknown, "not a symbol");
        assert!(toy_relations(false).with_variables(&["pair"]).expect_err("a function").starts_with("our software has a bug: "));
        assert!(toy_relations(false).with_variables(&["nothing"]).is_err());
    }

    /// The Dv lane: one word a frame row (`h_T + t_T`) of each variable-capable type, after every Dc; the heads, tails
    /// and Dcs where they are without variables, so the frames are sized as today and the gene gains one word a cell.
    #[test]
    fn the_dv_lane_has_one_word_a_cell_after_the_dcs_and_moves_nothing_else() {
        let (k, plain) = (toy(), toy_relations(false).check().expect("checks"));
        for n in [1usize, 2, 5, 17] {
            let (l, p) = (GeneLayout::from_program_size(&k, Ty::S, n).expect("layout"), GeneLayout::from_program_size(&plain, Ty::S, n).expect("layout"));
            assert_eq!(l.frames.len(), p.frames.len());
            for (f, g) in l.frames.iter().zip(&p.frames) {
                assert_eq!((f.ty, f.head_off, f.head_len, f.tail_off, f.tail_len, f.dc_off, f.dc_len), (g.ty, g.head_off, g.head_len, g.tail_off, g.tail_len, g.dc_off, g.dc_len), "N = {n}");
                assert_eq!(f.dv_len, f.head_len + f.tail_len, "N = {n}: one word a frame row");
                assert!(f.dv_off.expect("a Dv") >= p.width, "N = {n}: the Dv lanes follow every Dc");
            }
            assert_eq!(l.width, p.width + l.frame_rows(), "N = {n}: one index word a cell");
            let b = k.bounds(Ty::S, &l.head_lens()).expect("bounds");
            assert_eq!(b.width(), l.width);
            assert!(b.types.iter().all(|t| t.variable && t.dv() == t.h + t.t));
            let dv: Vec<Span> = l.columns(Domain::Dv);
            assert_eq!(dv.iter().map(|s| s.len).sum::<usize>(), l.frame_rows());
            for s in &dv {
                assert_eq!(l.key(s.off).domain, Domain::Dv);
            }
            let rows = l.rows(1, 0, &vec![7; l.width]);
            assert_eq!(rows.iter().filter(|r| r.key.domain == Domain::Dv).count(), l.frame_rows());
            assert!(rows.iter().filter(|r| r.key.domain == Domain::Dv).all(|r| r.symbol.is_none() && r.dc_index.is_none() && r.dv_index == Some(7)));
            // Each head and tail cell's parallel word is its own frame row of Dv_T.
            for f in &l.frames {
                for r in 0..f.head_len + f.tail_len {
                    assert_eq!(l.dv_of_cell(f.row_pos(r).expect("row")), f.dv_pos(r));
                }
            }
        }
    }

    /// A tree with variables writes back and expresses to itself; each `var.T` reads the word of its own row of
    /// `Dv_T` and no Dc word, so the `?` after it still reads `Dc_I[0]`; `walk_expressed` hands the same words.
    #[test]
    fn a_tree_with_variables_writes_back_and_expresses_to_itself() {
        let k = toy();
        let l = GeneLayout::from_program_size(&k, Ty::S, 4).expect("layout");
        // pair(pair(var.S#9, ?#2), inc(var.I#3)): the I nodes in level order are inc, ?, var.I; and
        // pair(pair(var.S#9, ?#2), var.I#3): var.I comes before the ?, which must still read Dc_I[0].
        let first = Tree { symbol: PAIR, const_index: None, kids: vec![Tree { symbol: PAIR, const_index: None, kids: vec![leaf_w(VAR_S, 9), leaf_w(QI, 2)] }, Tree { symbol: INC, const_index: None, kids: vec![leaf_w(VAR_I, 3)] }] };
        let previous = toy_gene(&k, &l, 5, 100);
        let g1 = write_back(&k, &l, &previous, &first).expect("encodes");
        assert_eq!(express(&k, &l, &g1, &toy_consts()).expect("expresses").tree(), first);
        let tree = Tree { symbol: PAIR, const_index: None, kids: vec![Tree { symbol: PAIR, const_index: None, kids: vec![leaf_w(VAR_S, 9), leaf_w(QI, 2)] }, leaf_w(VAR_I, 3)] };
        let gene = write_back(&k, &l, &previous, &tree).expect("encodes");
        let e = express(&k, &l, &gene, &toy_consts()).expect("expresses");
        assert_eq!(e.tree(), tree);
        assert_eq!(e.reach.q_used.get(&Ty::I), Some(&1), "only the ? read the Dc");
        let fi = l.frame(Ty::I).expect("I");
        assert_eq!(gene[fi.dc_off.expect("Dc_I")], 2, "the ? reads Dc_I[0]: the var.I took no Dc word");
        for node in &e.nodes {
            if k.is_variable(node.symbol) {
                let f = l.frame(node.ty).expect("frame");
                assert_eq!(node.const_index, Some(gene[f.dv_pos(node.frame_k).expect("a Dv word")]), "the word of its own row");
            } else if !k.is_constant(node.symbol) {
                assert_eq!(node.const_index, None);
            }
        }
        // Rows the tree does not use keep their previous words, Dv included.
        let used: std::collections::BTreeSet<usize> = e.nodes.iter().flat_map(|n| [Some(n.pos), l.dv_of_cell(n.pos).filter(|_| k.is_variable(n.symbol))]).flatten().collect();
        let dc0 = fi.dc_off.expect("Dc_I");
        for p in (0..l.width).filter(|p| !used.contains(p) && *p != dc0) {
            assert_eq!(gene[p], previous[p], "cell {p} is not the tree's");
        }
        let mut seen = Vec::new();
        walk_expressed(&k, &l, &gene, &toy_consts(), usize::MAX - 1, &mut |s, w| seen.push((s, w))).expect("walks");
        assert_eq!(seen, e.nodes.iter().map(|n| (n.symbol, n.const_index)).collect::<Vec<_>>());
        // A var.T with no word, or a word past the range, does not encode.
        let bare = Tree { symbol: PAIR, const_index: None, kids: vec![Tree { symbol: VAR_S, const_index: None, kids: vec![] }, leaf_w(QI, 0)] };
        assert_eq!(write_back(&k, &l, &previous, &bare), Err(LayoutDefect::MissingVariableIndex { step: 2 }));
        let wide = Tree { symbol: PAIR, const_index: None, kids: vec![leaf_w(VAR_S, DV_WORDS), leaf_w(QI, 0)] };
        assert!(matches!(write_back(&k, &l, &previous, &wide), Err(LayoutDefect::DvWordPastRange { word: DV_WORDS, .. })));
    }

    /// An edit of a Dv word changes no symbol, only the word its `var.T` reads; a word at or past the range is not in
    /// the alphabet, and a gene holding one does not validate.
    #[test]
    fn an_edit_of_a_dv_word_changes_no_symbol() {
        let k = toy();
        let l = GeneLayout::from_program_size(&k, Ty::S, 6).expect("layout");
        let mut checked = 0;
        for seed in 0..200u64 {
            let gene = toy_gene(&k, &l, seed, 0);
            let before = express(&k, &l, &gene, &toy_consts()).expect("expresses");
            for s in l.columns(Domain::Dv) {
                for p in s.off..s.off + s.len {
                    let mut g = gene.clone();
                    l.mutate(&k, &mut g, p, 40_000, &toy_consts()).expect("in the alphabet");
                    let after = express(&k, &l, &g, &toy_consts()).expect("expresses");
                    let syms = |e: &Expressed| e.nodes.iter().map(|n| (n.symbol, n.pos)).collect::<Vec<_>>();
                    assert_eq!(syms(&after), syms(&before), "seed {seed}: a Dv edit at {p} changed a symbol");
                    let reader = before.nodes.iter().position(|n| k.is_variable(n.symbol) && l.dv_of_cell(n.pos) == Some(p));
                    for (i, (a, b)) in after.nodes.iter().zip(&before.nodes).enumerate() {
                        assert_eq!(a.const_index, if Some(i) == reader { Some(40_000) } else { b.const_index });
                    }
                    checked += usize::from(reader.is_some());
                }
            }
        }
        assert!(checked > 100, "only {checked} edits reached an expressed var.T");
        let s = l.columns(Domain::Dv)[0];
        assert!(l.in_alphabet(&k, s.off, DV_WORDS - 1, &toy_consts()) && !l.in_alphabet(&k, s.off, DV_WORDS, &toy_consts()));
        let mut g = toy_gene(&k, &l, 1, 0);
        assert_eq!(l.mutate(&k, &mut g, s.off, DV_WORDS, &toy_consts()), Err(LayoutDefect::NotInAlphabet { pos: s.off, value: DV_WORDS }));
        g[s.off] = DV_WORDS;
        assert_eq!(express(&k, &l, &g, &toy_consts()), Err(LayoutDefect::DvWordPastRange { pos: s.off, word: DV_WORDS }));
    }

    /// Every head and tail cell's (symbol, Dv word) pair, keyed by the word's tag: a row operator that keeps every
    /// pair keeps every word with its cell.
    fn pairs_by_tag(l: &GeneLayout, gene: &[u32]) -> BTreeMap<u32, u32> {
        let mut m = BTreeMap::new();
        for s in l.spans().into_iter().filter(|s| matches!(s.domain, Domain::Head | Domain::Tail)) {
            for p in s.off..s.off + s.len {
                m.insert(gene[l.dv_of_cell(p).expect("a Dv word")], gene[p]);
            }
        }
        m
    }

    fn assert_cells_keep_their_words(l: &GeneLayout, tagged: &BTreeMap<u32, u32>, gene: &[u32], what: &str) {
        for s in l.spans().into_iter().filter(|s| matches!(s.domain, Domain::Head | Domain::Tail)) {
            for p in s.off..s.off + s.len {
                let w = gene[l.dv_of_cell(p).expect("a Dv word")];
                assert_eq!(tagged.get(&w), Some(&gene[p]), "{what}: cell {p} holds {} beside the word of another cell", gene[p]);
            }
        }
    }

    /// After inversion, IS and RIS transposition and one- and two-point crossover, absolute and relative, every cell
    /// (so every `var.T` cell) keeps its Dv word, and the gene still expresses. A Dv span moved alone is a defect.
    #[test]
    fn every_row_operator_moves_a_cell_and_its_dv_word_together() {
        let k = toy();
        let l = GeneLayout::from_program_size(&k, Ty::S, 7).expect("layout");
        let mut moved = 0usize;
        for seed in 0..300u64 {
            let (a0, b0) = (toy_gene(&k, &l, seed, 0), toy_gene(&k, &l, seed + 10_000, 20_000));
            let mut tagged = pairs_by_tag(&l, &a0);
            tagged.extend(pairs_by_tag(&l, &b0));
            let x = seed as usize;
            for domain in [Domain::Head, Domain::Tail] {
                let rows = l.domain_rows(domain);
                let (lo, hi) = (x % rows, (x % rows) + 1 + (x / 7) % (rows - x % rows));
                let mut g = a0.clone();
                if hi > lo {
                    moved += l.invert_rows(&mut g, domain, lo, hi).expect("inversion");
                    assert_cells_keep_their_words(&l, &tagged, &g, "inversion");
                }
                let mut g = a0.clone();
                moved += l.transpose_rows(&mut g, domain, x % rows, 1 + x % 3, (x / 3) % rows).expect("IS");
                assert_cells_keep_their_words(&l, &tagged, &g, "IS");
                if let (Domain::Head, Some(src)) = (domain, l.ris_row(&k, &a0, x % rows)) {
                    let mut g = a0.clone();
                    moved += l.transpose_rows(&mut g, domain, src, 1 + x % 3, 0).expect("RIS");
                    assert_cells_keep_their_words(&l, &tagged, &g, "RIS");
                    express(&k, &l, &g, &toy_consts()).expect("RIS child expresses");
                }
                let f = |v: usize| (v as u64 * 0x1234_5679) % FRAC_ONE;
                let (fl, fh) = (f(x).min(f(x + 1)), f(x).max(f(x + 1)));
                if fl < fh {
                    let mut g = a0.clone();
                    moved += l.invert_rel(&mut g, domain, fl, fh).expect("relative inversion");
                    assert_cells_keep_their_words(&l, &tagged, &g, "relative inversion");
                }
                let mut g = a0.clone();
                moved += l.transpose_rel(&mut g, domain, fl, fh, f(x + 2)).expect("relative IS");
                assert_cells_keep_their_words(&l, &tagged, &g, "relative IS");
            }
            let rows = l.gene_rows();
            let cut = 1 + x % (rows - 1);
            for cut2 in [None, Some(cut + 1 + (x / 5) % (rows - cut))] {
                let (mut a, mut b) = (a0.clone(), b0.clone());
                if cut2.is_some_and(|c| c > rows) {
                    continue;
                }
                l.crossover_rows(&l, &mut a, &mut b, cut, cut2).expect("crossover");
                assert_cells_keep_their_words(&l, &tagged, &a, "crossover, child a");
                assert_cells_keep_their_words(&l, &tagged, &b, "crossover, child b");
                express(&k, &l, &a, &toy_consts()).expect("child expresses");
                let (mut a, mut b) = (a0.clone(), b0.clone());
                l.crossover_rel(&l, &mut a, &mut b, FRAC_ONE / 3, cut2.map(|_| 2 * FRAC_ONE / 3)).expect("relative crossover");
                assert_cells_keep_their_words(&l, &tagged, &a, "relative crossover, child a");
                assert_cells_keep_their_words(&l, &tagged, &b, "relative crossover, child b");
            }
            // The position-level operators carry the lane too.
            let s = l.columns(Domain::Head)[0];
            let (mut a, mut b) = (a0.clone(), b0.clone());
            l.swap_range(&l, &mut a, &mut b, s.off, s.off + s.len).expect("swap");
            assert_cells_keep_their_words(&l, &tagged, &a, "swap_range");
            assert_cells_keep_their_words(&l, &tagged, &b, "swap_range");
        }
        assert!(moved > 300, "only {moved} columns changed");
        let dv = l.columns(Domain::Dv)[0];
        let mut g = toy_gene(&k, &l, 3, 0);
        assert_eq!(l.invert(&mut g, dv.off, dv.off + 2), Err(LayoutDefect::LaneMovedAlone { lo: dv.off, hi: dv.off + 2 }));
        assert!(matches!(l.invert_rows(&mut g, Domain::Dv, 0, 2), Err(LayoutDefect::LaneMovedAlone { .. })));
        assert!(matches!(l.transpose(&mut g, dv.off, 1, dv.off + 1), Err(LayoutDefect::LaneMovedAlone { .. })));
        let mut h = g.clone();
        assert!(matches!(l.swap_range(&l, &mut g, &mut h, dv.off, dv.off + 1), Err(LayoutDefect::LaneMovedAlone { .. })));
        assert!(LayoutDefect::LaneMovedAlone { lo: 1, hi: 2 }.to_string().starts_with("our software has a bug: "));
    }

    /// A Dv cell lists `DV_NEIGHBOURS` words, not all `DV_WORDS`: the bounded one-edit neighbours of an index word,
    /// every one legal; a word past them is still legal (G2 code review finding 4).
    #[test]
    fn a_dv_cell_lists_a_bounded_set_of_neighbour_words() {
        let k = toy();
        let l = GeneLayout::from_program_size(&k, Ty::S, 6).expect("layout");
        for s in l.columns(Domain::Dv) {
            for p in s.off..s.off + s.len {
                let a = l.alphabet(&k, p, &toy_consts());
                assert_eq!(a, (0..DV_NEIGHBOURS).collect::<Vec<_>>(), "Dv cell {p}");
                assert!(a.iter().all(|w| l.in_alphabet(&k, p, *w, &toy_consts())));
                assert!(l.in_alphabet(&k, p, DV_NEIGHBOURS, &toy_consts()) && l.in_alphabet(&k, p, DV_WORDS - 1, &toy_consts()));
            }
        }
    }

    /// A row inversion of the Dv table alone is `LaneMovedAlone` whatever the words hold: a range that reads the same
    /// both ways, and a range of one row, are the defect too, never `Ok(0)` (G2 code review finding 6).
    #[test]
    fn a_row_inversion_of_the_dv_table_is_always_a_defect() {
        let k = toy();
        let l = GeneLayout::from_program_size(&k, Ty::S, 6).expect("layout");
        let mut g = toy_gene(&k, &l, 5, 0);
        for s in l.columns(Domain::Dv) {
            g[s.off..s.off + s.len].fill(7);
        }
        let before = g.clone();
        assert!(matches!(l.invert_rows(&mut g, Domain::Dv, 0, 2), Err(LayoutDefect::LaneMovedAlone { .. })), "a palindrome");
        assert!(matches!(l.invert_rows(&mut g, Domain::Dv, 0, 1), Err(LayoutDefect::LaneMovedAlone { .. })), "one row");
        assert!(matches!(l.invert_rel(&mut g, Domain::Dv, 0, FRAC_ONE / 2), Err(LayoutDefect::LaneMovedAlone { .. })), "relative, a palindrome");
        assert!(matches!(l.invert_rel(&mut g, Domain::Dv, 0, 1), Err(LayoutDefect::LaneMovedAlone { .. })), "relative, no row");
        assert_eq!(g, before);
        // The other tables are unchanged by the check.
        assert!(l.invert_rows(&mut g, Domain::Head, 0, 2).is_ok() && l.invert_rel(&mut g, Domain::Tail, 0, FRAC_ONE).is_ok());
    }
}
