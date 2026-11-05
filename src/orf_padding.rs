//! ORF PADDING: champion material in a gene's non-coding cells, and symbol frequencies for the draw
//! (`docs/DESIGN_futhark_types_and_properties.md` section 11; Andrew Morgan's hypothesis of 2026-11-05).
//!
//! The cells of a gene after its expressed part are not dead: point mutation, transposition, inversion and
//! crossover expose them. A random draw fills them with uniformly random symbols, unlike the symbol frequencies of
//! real programs. This module is the kingdom-general core of the two remedies:
//!
//! * **overlay**: a champion's program written as Karva gives, per type column, its ORF (open reading frame): the
//!   head cells and tail cells Karva reads, each with its Dc and Dv words. [`overlay_chunks`] copies random
//!   contiguous runs of that ORF over a random gene, head cells into the head and tail cells into the tail of the
//!   same column, so every gene stays legal by construction (a head row takes any symbol of its column, a tail row a
//!   terminal of its column, and the lengths `h_T`, `t_T` do not move);
//! * **frequency**: [`symbol_counts`] counts the symbols of a pool's programs and [`repetitions`] turns counts into
//!   how many times a symbol is listed in its sampling list, so a uniform draw from the list is a frequency draw.
//!
//! Nothing here knows a kingdom: a program is an [`OrfTree`] of symbol ids with their type column, and a gene is
//! described by its [`ColumnFrame`]s. A kingdom maps its program text to the tree (phylu's `champion_pool` does it
//! for TSR).

use std::collections::VecDeque;

/// A program as Karva sees it: a symbol id, the dense type column it is drawn in, the Dc word a constant
/// placeholder reads (an index into `C_T`), the Dv word a variable reads, and the children in argument order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrfTree {
    pub symbol: u32,
    pub column: u32,
    pub dc: Option<u32>,
    pub dv: Option<u32>,
    pub kids: Vec<OrfTree>,
}

impl OrfTree {
    /// A leaf.
    pub fn leaf(symbol: u32, column: u32) -> OrfTree {
        OrfTree { symbol, column, dc: None, dv: None, kids: Vec::new() }
    }

    /// The number of nodes.
    pub fn size(&self) -> usize {
        1 + self.kids.iter().map(OrfTree::size).sum::<usize>()
    }

    /// Every subtree, root first (preorder).
    pub fn subtrees(&self) -> Vec<&OrfTree> {
        let mut out = vec![self];
        let mut i = 0;
        while i < out.len() {
            let t = out[i];
            out.extend(t.kids.iter());
            i += 1;
        }
        out
    }
}

/// One cell of an ORF: its symbol, and the Dc word (a constant placeholder) or Dv word (a variable) it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrfCell {
    pub symbol: u32,
    pub dc: Option<u32>,
    pub dv: Option<u32>,
}

/// The ORF of one type column: the head rows Karva reads (rows `0 .. min(j_T, h_T)`) and the tail rows
/// (rows `h_T .. j_T`), in row order, which is the order the walk reads them and the order the Dc words are read in.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OrfColumn {
    pub head: Vec<OrfCell>,
    pub tail: Vec<OrfCell>,
}

impl OrfColumn {
    /// `j_T`: the rows of the column the walk reads.
    pub fn len(&self) -> usize {
        self.head.len() + self.tail.len()
    }

    pub fn is_empty(&self) -> bool {
        self.head.is_empty() && self.tail.is_empty()
    }
}

/// One type column of a gene, as word offsets inside the gene: the head, the tail, the Dc words, the Dv words
/// (one a head or tail row; `dv_len` 0 for a column without variables) and the length of `C_T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnFrame {
    pub head_off: usize,
    pub head_len: usize,
    pub tail_off: usize,
    pub tail_len: usize,
    pub dc_off: usize,
    pub dc_len: usize,
    pub dv_off: usize,
    pub dv_len: usize,
    pub constant_len: u32,
}

/// THE ORF OF A PROGRAM WRITTEN AS KARVA on a gene of `frames`: the level-order walk from the root, each node
/// taking the next row of its own column. `None` when the program does not fit the gene: a function would land
/// past its column's head, a column would run past its tail, a column is not in `frames`, or a column's constant
/// placeholders outnumber its Dc words.
pub fn karva_orf(tree: &OrfTree, frames: &[ColumnFrame]) -> Option<Vec<OrfColumn>> {
    let mut cols = vec![OrfColumn::default(); frames.len()];
    let mut dc_used = vec![0usize; frames.len()];
    let mut queue = VecDeque::from([tree]);
    while let Some(n) = queue.pop_front() {
        let c = n.column as usize;
        let f = frames.get(c)?;
        let col = &mut cols[c];
        let row = col.len();
        let cell = OrfCell { symbol: n.symbol, dc: n.dc, dv: n.dv };
        if n.dc.is_some() {
            dc_used[c] += 1;
            if dc_used[c] > f.dc_len {
                return None;
            }
        }
        if row < f.head_len {
            col.head.push(cell);
        } else if row < f.head_len + f.tail_len && n.kids.is_empty() {
            col.tail.push(cell);
        } else {
            return None;
        }
        queue.extend(n.kids.iter());
    }
    Some(cols)
}

/// THE OVERLAY: copy `chunks` random contiguous runs of `donor`'s ORF over `gene`, per column, head cells into the
/// head and tail cells into the tail, each landing at a random row of its own region; a cell's Dv word goes to the
/// Dv word of the row it lands on, and a placeholder's Dc word to the Dc word of its order in the donor's ORF (the
/// index the donor's walk read it at). `below(n)` draws `0 .. n` (the caller's keyed generator). A column the
/// donor does not express keeps its draw. Legal by construction: no length moves and no tail row takes a function
/// (a donor tail cell is a terminal); a Dc word is copied only when it is below the target's `C_T` length.
pub fn overlay_chunks(gene: &mut [u32], frames: &[ColumnFrame], donor: &[OrfColumn], chunks: usize, below: &mut dyn FnMut(u32) -> u32) {
    for (c, f) in frames.iter().enumerate() {
        let Some(d) = donor.get(c) else { continue };
        if d.is_empty() {
            continue;
        }
        // The Dc order of each donor cell: placeholders are read in row order (head rows, then tail rows).
        let mut q = 0usize;
        let mut dc_q = |cells: &[OrfCell]| -> Vec<Option<usize>> {
            cells
                .iter()
                .map(|cell| {
                    cell.dc.map(|_| {
                        q += 1;
                        q - 1
                    })
                })
                .collect()
        };
        let head_q = dc_q(&d.head);
        let tail_q = dc_q(&d.tail);
        for k in 0..chunks {
            // Alternate head and tail chunks (head first), so both regions receive material.
            let (cells, qs, off, len, row0) = if k % 2 == 0 || d.tail.is_empty() || f.tail_len == 0 {
                (&d.head, &head_q, f.head_off, f.head_len, 0usize)
            } else {
                (&d.tail, &tail_q, f.tail_off, f.tail_len, f.head_len)
            };
            if cells.is_empty() || len == 0 {
                continue;
            }
            let start = below(cells.len() as u32) as usize;
            let room = (cells.len() - start).min(len);
            let run = 1 + below(room as u32) as usize;
            let at = below((len - run + 1) as u32) as usize;
            for i in 0..run {
                let cell = &cells[start + i];
                let row = at + i;
                gene[off + row] = cell.symbol;
                if let Some(w) = cell.dv {
                    if row0 + row < f.dv_len {
                        gene[f.dv_off + row0 + row] = w;
                    }
                }
                if let (Some(w), Some(q)) = (cell.dc, qs[start + i]) {
                    if q < f.dc_len && w < f.constant_len {
                        gene[f.dc_off + q] = w;
                    }
                }
            }
        }
    }
}

/// How often each symbol id occurs in `trees`, over `n_symbols` ids (an id past the end is not counted).
pub fn symbol_counts<'a>(trees: impl IntoIterator<Item = &'a OrfTree>, n_symbols: usize) -> Vec<u64> {
    let mut counts = vec![0u64; n_symbols];
    for t in trees {
        for s in t.subtrees() {
            if let Some(c) = counts.get_mut(s.symbol as usize) {
                *c += 1;
            }
        }
    }
    counts
}

/// THE FREQUENCY DRAW AS A LIST: each weight's share of `total` list entries, at least one each (so every symbol
/// stays drawable), by largest remainder so the shares sum exactly to `max(total, weights.len())` when every
/// weight is positive. A uniform draw from a list holding symbol `i` `repetitions[i]` times draws it in proportion
/// to its weight.
pub fn repetitions(weights: &[f64], total: u32) -> Vec<u32> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    let sum: f64 = weights.iter().map(|w| w.max(0.0)).sum();
    if sum <= 0.0 {
        return vec![1; n];
    }
    let spare = total.saturating_sub(n as u32) as f64;
    let exact: Vec<f64> = weights.iter().map(|w| w.max(0.0) / sum * spare).collect();
    let mut reps: Vec<u32> = exact.iter().map(|e| 1 + e.floor() as u32).collect();
    let mut left = (spare as u32).saturating_sub(exact.iter().map(|e| e.floor() as u32).sum());
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| (exact[b] - exact[b].floor()).total_cmp(&(exact[a] - exact[a].floor())).then(a.cmp(&b)));
    for &i in order.iter().cycle().take(n * 2) {
        if left == 0 {
            break;
        }
        reps[i] += 1;
        left -= 1;
    }
    reps
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two columns: 0 has functions 0 (arity 2) and 1 (arity 1), terminals 2 (a placeholder) and 3 (a variable);
    /// 1 has terminal 4.
    fn frames() -> Vec<ColumnFrame> {
        vec![
            ColumnFrame { head_off: 0, head_len: 5, tail_off: 5, tail_len: 6, dc_off: 11, dc_len: 6, dv_off: 17, dv_len: 11, constant_len: 9 },
            ColumnFrame { head_off: 28, head_len: 2, tail_off: 30, tail_len: 3, dc_off: 33, dc_len: 0, dv_off: 33, dv_len: 0, constant_len: 0 },
        ]
    }

    fn node(symbol: u32, kids: Vec<OrfTree>) -> OrfTree {
        OrfTree { symbol, column: 0, dc: None, dv: None, kids }
    }

    fn konst(w: u32) -> OrfTree {
        OrfTree { symbol: 2, column: 0, dc: Some(w), dv: None, kids: Vec::new() }
    }

    fn var(w: u32) -> OrfTree {
        OrfTree { symbol: 3, column: 0, dc: None, dv: Some(w), kids: Vec::new() }
    }

    /// The level order of `0(1(x), 0(?, x))` is 0, 1, 0, x, ?, x: three functions in the head, three terminals
    /// after; the placeholder's Dc word is the first one read.
    #[test]
    fn the_orf_is_the_level_order_split_at_the_head() {
        let t = node(0, vec![node(1, vec![var(7)]), node(0, vec![konst(4), var(8)])]);
        let orf = karva_orf(&t, &frames()).expect("fits");
        let syms = |cells: &[OrfCell]| cells.iter().map(|c| c.symbol).collect::<Vec<_>>();
        assert_eq!(syms(&orf[0].head), vec![0, 1, 0, 3, 2]);
        assert_eq!(syms(&orf[0].tail), vec![3]);
        assert_eq!(orf[0].head[4].dc, Some(4));
        assert_eq!(orf[0].tail[0].dv, Some(8));
        assert!(orf[1].is_empty());
    }

    /// A function past the head, or a column outside the frames, does not fit.
    #[test]
    fn a_program_too_deep_for_the_head_does_not_fit() {
        let mut t = var(0);
        for _ in 0..6 {
            t = node(1, vec![t]);
        }
        assert!(karva_orf(&t, &frames()).is_none());
        assert!(karva_orf(&OrfTree::leaf(4, 2), &frames()).is_none());
    }

    /// Every overlaid head row holds a donor head symbol or its draw, every tail row a donor tail terminal or its
    /// draw, and the same keys give the same gene.
    #[test]
    fn the_overlay_keeps_head_and_tail_kinds_and_is_deterministic() {
        let t = node(0, vec![node(0, vec![var(7), konst(4)]), node(0, vec![konst(5), var(1)])]);
        let orf = karva_orf(&t, &frames()).expect("fits");
        let fr = frames();
        let base: Vec<u32> = (0..36).map(|i| 100 + i).collect();
        let run = |seed: u64| {
            let mut g = base.clone();
            let mut s = seed;
            let mut below = |n: u32| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((u128::from(s >> 11) * u128::from(n)) >> 53) as u32
            };
            overlay_chunks(&mut g, &fr, &orf, 4, &mut below);
            g
        };
        for seed in 0..200 {
            let g = run(seed);
            assert_eq!(g, run(seed), "deterministic");
            assert_ne!(g, base, "seed {seed}: something was copied");
            for r in 0..5 {
                assert!(g[r] == base[r] || orf[0].head.iter().any(|c| c.symbol == g[r]), "head row {r}");
            }
            for r in 5..11 {
                assert!(g[r] == base[r] || orf[0].tail.iter().any(|c| c.symbol == g[r]), "tail row {r}");
                assert!(g[r] >= 100 || g[r] == 2 || g[r] == 3, "a tail row holds a terminal");
            }
            for d in 11..17 {
                assert!(g[d] == base[d] || g[d] == 4 || g[d] == 5, "Dc word {d}");
            }
            assert_eq!(&g[28..], &base[28..], "the column the donor does not express keeps its draw");
        }
    }

    #[test]
    fn counts_and_repetitions() {
        let t = node(0, vec![node(1, vec![var(7)]), node(0, vec![konst(4), var(8)])]);
        assert_eq!(symbol_counts([&t, &t], 5), vec![4, 2, 2, 4, 0]);
        let reps = repetitions(&[4.0, 2.0, 2.0, 4.0, 0.0], 1000);
        assert_eq!(reps.iter().sum::<u32>(), 1000);
        assert_eq!(reps[4], 1, "a symbol never seen stays drawable");
        assert!(reps[0].abs_diff(reps[3]) <= 1, "equal weights, equal shares to one entry");
        assert!(reps[0] > reps[1] * 2 - 3 && reps[0] < reps[1] * 2 + 3);
        assert_eq!(repetitions(&[0.0, 0.0], 10), vec![1, 1]);
    }
}
