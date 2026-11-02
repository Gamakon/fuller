//! THE DOMAIN OPERATORS of the Futhark kingdom's runner (`examples/futhark_evolve.rs`, `--ops` and `--measure`):
//! inversion, transposition and crossover, each in three forms — inside one type's span (`invert`, `transpose`,
//! `crossover`), on rows of the per-type table (`invert-rows`, `transpose-rows`, `cross-rows`) and on relative rows
//! (`invert-rel`, `transpose-rel`, `cross-rel`) — and the row forms AIMED AT CODING CELLS (`invert-rows-coding`,
//! `transpose-rows-coding`, `cross-rows-coding`; 2026-11-02, `docs/PLAN_futhark_kingdom.md` 3g). Every operator moves a
//! symbol only within its own type's column and domain, so every child is legal by construction.
//!
//! WHICH ARE AIMED (measured in this module's tests, `docs/PLAN_futhark_kingdom.md` 3g): the span operators already
//! draw a coding cell of the parent ([`invert_once`], [`transpose_once`]: a coding head cell; [`crossover_once`]: a
//! coding cell of the first parent) and act on the span that holds it, so they never pick a non-coding domain. The row
//! operators act on every column, and draw their rows uniformly over the whole table, whose rows are mostly past the
//! expressed cells; the `-coding` forms draw the rows around a coding cell (of either parent for the crossover)
//! instead, and are otherwise the same operators. The functions moved here from the runner unchanged, so each `--ops`
//! name means what it meant.

use std::collections::BTreeMap;

use crate::futhark_rings::coding_cells;
use crate::futhark_table::GeneRng;
use crate::geneframe::Ty;
use crate::kingdom_contract::KingdomContract;
use crate::typed_gep::{frac_of_row, Domain, GeneLayout, FRAC_ONE};

/// HOMOLOGOUS CROSSOVER (`crossover`): a range of the domain that holds one of `a`'s expressed cells, exchanged between
/// `a` and `b` (both children kept). A Dc span of one cell is left alone.
pub fn crossover_once(contract: &KingdomContract, layout: &GeneLayout, const_len: &BTreeMap<Ty, u32>, a: &mut [u32], b: &mut [u32], rng: &mut GeneRng) -> Result<(), String> {
    let cells = coding_cells(contract, layout, a, const_len)?;
    let pos = cells[rng.below(cells.len())];
    let span = layout.spans().into_iter().find(|s| s.off <= pos && pos < s.off + s.len).ok_or("a cell in no span")?;
    if span.domain == Domain::Dc && span.len < 2 {
        return Ok(());
    }
    let lo = span.off + rng.below(pos - span.off + 1);
    let hi = (pos + 1 + rng.below(span.off + span.len - pos)).min(span.off + span.len);
    layout.swap_range(layout, a, b, lo, hi).map_err(|e| format!("crossover: {e}"))
}

/// INVERSION (plan section I, step 3): a run of at least two cells, inside the one domain of one type that holds a
/// coding cell drawn uniformly, reversed (`GeneLayout::invert`). Every symbol stays in a cell of its own type and
/// domain, so the child is legal by construction. `None` when the drawn cell's span is a single cell.
pub fn invert_once(layout: &GeneLayout, gene: &mut [u32], coding: &[usize], rng: &mut GeneRng) -> Result<Option<String>, String> {
    let pos = coding[rng.below(coding.len())];
    let span = layout.spans().into_iter().find(|s| s.off <= pos && pos < s.off + s.len).ok_or("a coding cell in no span")?;
    if span.len < 2 {
        return Ok(None);
    }
    let lo = span.off + rng.below(pos - span.off + 1);
    let hi = (pos + 1 + rng.below(span.off + span.len - pos)).min(span.off + span.len);
    let (lo, hi) = if hi - lo < 2 { if hi < span.off + span.len { (lo, hi + 1) } else { (lo - 1, hi) } } else { (lo, hi) };
    if gene[lo..hi].iter().zip(gene[lo..hi].iter().rev()).all(|(a, b)| a == b) {
        return Ok(None);
    }
    layout.invert(gene, lo, hi).map_err(|e| format!("inversion: {e}"))?;
    Ok(Some(format!("{} cells [{lo}, {hi}) of a {} span reversed", hi - lo, span.domain.name())))
}

/// TRANSPOSITION (GEP's IS and RIS, plan section I, step 2): inside the head of one type holding a coding cell drawn
/// uniformly, a run of 1 to 3 cells is copied to another place in the same head; the cells after the insertion point
/// shift right and the last ones fall off the head's end. IS inserts anywhere; RIS takes a run that starts at a
/// function and inserts it at the head's start. Every symbol stays in its own type's head, so the child is legal by
/// construction. `None` when the drawn copy changes nothing.
pub fn transpose_once(contract: &KingdomContract, layout: &GeneLayout, gene: &mut [u32], coding: &[usize], rng: &mut GeneRng) -> Result<Option<String>, String> {
    let heads: Vec<usize> = coding.iter().copied().filter(|p| layout.key(*p).domain == Domain::Head).collect();
    if heads.is_empty() {
        return Ok(None);
    }
    let pos = heads[rng.below(heads.len())];
    let span = layout.spans().into_iter().find(|s| s.domain == Domain::Head && s.off <= pos && pos < s.off + s.len).ok_or("a coding head cell in no head span")?;
    if span.len < 2 {
        return Ok(None);
    }
    let ris = rng.below(2) == 1;
    let src = if ris {
        match (pos..span.off + span.len).find(|q| !contract.signature(gene[*q]).is_empty()) {
            Some(q) => q,
            None => return Ok(None),
        }
    } else {
        pos
    };
    let len = (1 + rng.below(3)).min(span.off + span.len - src);
    let dst = if ris { span.off } else { span.off + rng.below(span.len) };
    if !layout.transpose(gene, src, len, dst).map_err(|e| format!("transposition: {e}"))? {
        return Ok(None);
    }
    Ok(Some(format!("{} of {len} cells from {src} to {dst}, in a head of {} cells", if ris { "RIS" } else { "IS" }, span.len)))
}

// THE ROW OPERATORS (Andrew, 2026-11-01; `GeneLayout::invert_rows`, `transpose_rows`, `crossover_rows`): a domain of
// the gene is a table, row = position, column = type; the operator acts on rows across EVERY column, so one
// application touches every type's frame, not one of hundreds. Rows are drawn uniformly over the table's rows (its
// longest column); a column takes part only where it has the row. Every symbol stays in its own column: legal by
// construction.

/// ROW INVERSION: a domain drawn uniformly among those of at least two rows, rows `[lo, hi)` (at least two) drawn
/// uniformly, reversed in every column. `None` when no column changed.
pub fn invert_rows_once(layout: &GeneLayout, gene: &mut [u32], rng: &mut GeneRng) -> Result<Option<String>, String> {
    let domains: Vec<Domain> = [Domain::Head, Domain::Tail, Domain::Dc].into_iter().filter(|d| layout.domain_rows(*d) >= 2).collect();
    if domains.is_empty() {
        return Ok(None);
    }
    let d = domains[rng.below(domains.len())];
    let rows = layout.domain_rows(d);
    let lo = rng.below(rows - 1);
    let hi = lo + 2 + rng.below(rows - lo - 1);
    let changed = layout.invert_rows(gene, d, lo, hi).map_err(|e| format!("row inversion: {e}"))?;
    if changed == 0 {
        return Ok(None);
    }
    Ok(Some(format!("rows [{lo}, {hi}) of the {} table ({rows} rows) reversed in {changed} of {} columns", d.name(), layout.columns(d).len())))
}

/// ROW TRANSPOSITION (GEP's IS and RIS on the head table, one or the other with probability one half): a run of 1
/// to 3 rows copied to another row in every column, the rows after it shifting down and the overflow falling off
/// each column. IS: the run's first row and the destination uniform over the rows. RIS: from a row drawn uniformly,
/// the first row where the root's column holds a function, inserted at row 0. `None` when no column changed.
pub fn transpose_rows_once(contract: &KingdomContract, layout: &GeneLayout, gene: &mut [u32], rng: &mut GeneRng) -> Result<Option<String>, String> {
    let rows = layout.domain_rows(Domain::Head);
    if rows < 2 {
        return Ok(None);
    }
    let ris = rng.below(2) == 1;
    let src = if ris {
        match layout.ris_row(contract, gene, rng.below(rows)) {
            Some(r) => r,
            None => return Ok(None),
        }
    } else {
        rng.below(rows)
    };
    let len = 1 + rng.below(3);
    let dst = if ris { 0 } else { rng.below(rows) };
    let changed = layout.transpose_rows(gene, Domain::Head, src, len, dst).map_err(|e| format!("row transposition: {e}"))?;
    if changed == 0 {
        return Ok(None);
    }
    Ok(Some(format!("{} of rows [{src}, {}) to row {dst} of the head table ({rows} rows), {changed} of {} columns changed", if ris { "RIS" } else { "IS" }, src + len, layout.columns(Domain::Head).len())))
}

/// ROW CROSSOVER: GEP's one-point or two-point recombination (one or the other with probability one half) at rows of
/// the whole gene's table, a column being `head_T`, `tail_T`, `Dc_T` (`GeneLayout::crossover_rows`): one-point
/// exchanges rows `c..` for `c` uniform in `1 .. rows`, two-point rows `[c1, c2)` for `c1 < c2` uniform. Both
/// children kept. `None` when the table has one row.
pub fn cross_rows_once(layout: &GeneLayout, a: &mut [u32], b: &mut [u32], rng: &mut GeneRng) -> Result<Option<String>, String> {
    let rows = layout.gene_rows();
    if rows < 2 {
        return Ok(None);
    }
    let (c1, c2) = if rng.below(2) == 0 {
        (1 + rng.below(rows - 1), None)
    } else {
        let c1 = rng.below(rows - 1);
        (c1, Some(c1 + 1 + rng.below(rows - c1 - 1)))
    };
    let cells = layout.crossover_rows(layout, a, b, c1, c2).map_err(|e| format!("row crossover: {e}"))?;
    Ok(Some(match c2 {
        None => format!("one-point at row {c1} of {rows}: {cells} cells exchanged"),
        Some(c2) => format!("two-point, rows [{c1}, {c2}) of {rows}: {cells} cells exchanged"),
    }))
}

// THE RELATIVE-ROW OPERATORS (2026-11-01; `GeneLayout::invert_rel`, `transpose_rel`, `crossover_rel`): the cuts and
// runs are fractions in [0, 1), and each column applies them at its own row floor(f x len), so EVERY column takes
// part in every application (with absolute rows a short column has no row past its end: pathfinder's gene table is
// 2041 rows, its mean column 101). With one type they are the classic operators.

/// A fraction uniform in `[0, 1)`, in parts of `FRAC_ONE`.
pub fn frac(rng: &mut GeneRng) -> u64 {
    rng.next_u64() >> 32
}

/// Two fractions `lo < hi <= 1`, `lo` uniform in `[0, 1)`, `hi` uniform above it.
pub fn frac_pair(rng: &mut GeneRng) -> (u64, u64) {
    let lo = frac(rng);
    (lo, lo + 1 + rng.next_u64() % (FRAC_ONE - lo))
}

/// A fraction as a percentage, for the event's message.
pub fn pct(f: u64) -> f64 {
    100.0 * f as f64 / FRAC_ONE as f64
}

/// RELATIVE INVERSION: a domain drawn uniformly among those of at least two rows, fractions `lo < hi` drawn
/// uniformly, each column's rows `[floor(lo x len), floor(hi x len))` reversed. `None` when no column changed.
pub fn invert_rel_once(layout: &GeneLayout, gene: &mut [u32], rng: &mut GeneRng) -> Result<Option<String>, String> {
    let domains: Vec<Domain> = [Domain::Head, Domain::Tail, Domain::Dc].into_iter().filter(|d| layout.domain_rows(*d) >= 2).collect();
    if domains.is_empty() {
        return Ok(None);
    }
    let d = domains[rng.below(domains.len())];
    let (lo, hi) = frac_pair(rng);
    let changed = layout.invert_rel(gene, d, lo, hi).map_err(|e| format!("relative inversion: {e}"))?;
    if changed == 0 {
        return Ok(None);
    }
    Ok(Some(format!("fractions [{:.1} %, {:.1} %) of every {} column reversed, {changed} of {} columns changed", pct(lo), pct(hi), d.name(), layout.columns(d).len())))
}

/// RELATIVE TRANSPOSITION (IS or RIS, one half each, the head table): IS, the run's start and the destination uniform
/// fractions, the run 1 to 3 rows of the longest column long as a fraction (at least one row in every column); RIS,
/// the run starts where the root's column holds a function (from a row drawn uniformly) at that row's fraction, and is
/// inserted at row 0. `None` when no column changed.
pub fn transpose_rel_once(contract: &KingdomContract, layout: &GeneLayout, gene: &mut [u32], rng: &mut GeneRng) -> Result<Option<String>, String> {
    let rows = layout.domain_rows(Domain::Head);
    if rows < 2 {
        return Ok(None);
    }
    let ris = rng.below(2) == 1;
    let src = if ris {
        let root_len = layout.frame(layout.root).map_or(0, |f| f.head_len);
        if root_len == 0 {
            return Ok(None);
        }
        match layout.ris_row(contract, gene, rng.below(root_len)) {
            Some(r) => frac_of_row(r, root_len),
            None => return Ok(None),
        }
    } else {
        frac(rng)
    };
    let end = (src + frac_of_row(1 + rng.below(3), rows)).min(FRAC_ONE);
    let dst = if ris { 0 } else { frac(rng) };
    let changed = layout.transpose_rel(gene, Domain::Head, src, end, dst).map_err(|e| format!("relative transposition: {e}"))?;
    if changed == 0 {
        return Ok(None);
    }
    Ok(Some(format!("{} of fractions [{:.1} %, {:.1} %) to {:.1} % of every head column, {changed} of {} columns changed", if ris { "RIS" } else { "IS" }, pct(src), pct(end), pct(dst), layout.columns(Domain::Head).len())))
}

/// RELATIVE CROSSOVER: one-point at a fraction uniform in `[0, 1)` or two-point at fractions `lo < hi` (one half
/// each), each whole column (`head_T`, `tail_T`, `Dc_T`) cut at its own rows. Both children kept.
pub fn cross_rel_once(layout: &GeneLayout, a: &mut [u32], b: &mut [u32], rng: &mut GeneRng) -> Result<Option<String>, String> {
    let (c1, c2) = if rng.below(2) == 0 {
        (frac(rng), None)
    } else {
        let (x, y) = frac_pair(rng);
        (x, Some(y))
    };
    let cells = layout.crossover_rel(layout, a, b, c1, c2).map_err(|e| format!("relative crossover: {e}"))?;
    if cells == 0 {
        return Ok(None);
    }
    Ok(Some(match c2 {
        None => format!("one-point at {:.1} % of every column: {cells} cells exchanged", pct(c1)),
        Some(c2) => format!("two-point, [{:.1} %, {:.1} %) of every column: {cells} cells exchanged", pct(c1), pct(c2)),
    }))
}

// ---- THE ROW OPERATORS AIMED AT CODING CELLS (2026-11-02) -------------------------------------------------------
//
// The row operators above draw their rows uniformly over the table, whose longest column is far longer than the part
// any expressed program reads, so most applications move no coding cell (measured in the tests below). These draw a
// coding cell and choose the rows so the run or the cut holds that cell's row; the operation itself is the same
// (`GeneLayout::invert_rows`, `transpose_rows`, `crossover_rows`), every column taking part where it has the rows.

/// The row of the cell at `pos` in the WHOLE gene's table (a column is `head_T`, `tail_T`, `Dc_T`).
fn gene_row_of(layout: &GeneLayout, pos: usize) -> Option<usize> {
    layout.frames.iter().find_map(|f| {
        if (f.head_off..f.head_off + f.head_len).contains(&pos) {
            Some(pos - f.head_off)
        } else if (f.tail_off..f.tail_off + f.tail_len).contains(&pos) {
            Some(f.head_len + pos - f.tail_off)
        } else {
            f.dc_off.filter(|d| (*d..*d + f.dc_len).contains(&pos)).map(|d| f.head_len + f.tail_len + pos - d)
        }
    })
}

/// AIMED ROW INVERSION (`invert-rows-coding`): a coding cell drawn uniformly; in its domain's table, rows `[lo, hi)`
/// (at least two) holding the cell's row, `lo` uniform at or below it and `hi` uniform above it, reversed in every
/// column. `None` when no column changed or the domain has one row.
pub fn invert_rows_coding_once(layout: &GeneLayout, gene: &mut [u32], coding: &[usize], rng: &mut GeneRng) -> Result<Option<String>, String> {
    let key = layout.key(coding[rng.below(coding.len())]);
    let (d, r) = (key.domain, key.domain_seq_id);
    let rows = layout.domain_rows(d);
    if rows < 2 {
        return Ok(None);
    }
    let lo = rng.below(r + 1);
    let hi = r + 1 + rng.below(rows - r);
    let (lo, hi) = if hi - lo >= 2 {
        (lo, hi)
    } else if hi < rows {
        (lo, hi + 1)
    } else {
        (lo - 1, hi)
    };
    let changed = layout.invert_rows(gene, d, lo, hi).map_err(|e| format!("aimed row inversion: {e}"))?;
    if changed == 0 {
        return Ok(None);
    }
    Ok(Some(format!("rows [{lo}, {hi}) of the {} table ({rows} rows), around the coding row {r}, reversed in {changed} of {} columns", d.name(), layout.columns(d).len())))
}

/// AIMED ROW TRANSPOSITION (`transpose-rows-coding`, IS or RIS one half each, the head table): IS, the run of 1 to 3
/// rows starts at the row of a coding head cell drawn uniformly and is copied to a row drawn uniformly; RIS, the run
/// starts at the row of a coding cell of the root's head column that holds a function (drawn uniformly) and is
/// inserted at row 0. `None` when no column changed.
pub fn transpose_rows_coding_once(contract: &KingdomContract, layout: &GeneLayout, gene: &mut [u32], coding: &[usize], rng: &mut GeneRng) -> Result<Option<String>, String> {
    let rows = layout.domain_rows(Domain::Head);
    if rows < 2 {
        return Ok(None);
    }
    let ris = rng.below(2) == 1;
    let from: Vec<usize> = coding
        .iter()
        .map(|p| (*p, layout.key(*p)))
        .filter(|(p, k)| k.domain == Domain::Head && (!ris || (k.domain_type == layout.root && !contract.signature(gene[*p]).is_empty())))
        .map(|(_, k)| k.domain_seq_id)
        .collect();
    if from.is_empty() {
        return Ok(None);
    }
    let src = from[rng.below(from.len())];
    let len = 1 + rng.below(3);
    let dst = if ris { 0 } else { rng.below(rows) };
    let changed = layout.transpose_rows(gene, Domain::Head, src, len, dst).map_err(|e| format!("aimed row transposition: {e}"))?;
    if changed == 0 {
        return Ok(None);
    }
    Ok(Some(format!("{} of rows [{src}, {}) (a coding row) to row {dst} of the head table ({rows} rows), {changed} of {} columns changed", if ris { "RIS" } else { "IS" }, src + len, layout.columns(Domain::Head).len())))
}

/// AIMED ROW CROSSOVER (`cross-rows-coding`, one- or two-point, one half each): a coding cell of EITHER parent drawn
/// uniformly, `r` its row in the whole gene's table; one-point exchanges rows `c..` for `c` uniform in `1 ..= r`,
/// two-point rows `[c1, c2)` for `c1` uniform in `0 ..= r` and `c2` uniform above `r` (a cell of row 0, the root,
/// takes the two-point form). Both children kept. `None` when the table has one row.
pub fn cross_rows_coding_once(contract: &KingdomContract, layout: &GeneLayout, const_len: &BTreeMap<Ty, u32>, a: &mut [u32], b: &mut [u32], rng: &mut GeneRng) -> Result<Option<String>, String> {
    let rows = layout.gene_rows();
    if rows < 2 {
        return Ok(None);
    }
    let mut cells = coding_cells(contract, layout, a, const_len)?;
    cells.extend(coding_cells(contract, layout, b, const_len)?);
    let pos = cells[rng.below(cells.len())];
    let r = gene_row_of(layout, pos).ok_or("our software has a bug: a coding cell in no column")?;
    let (c1, c2) = if r >= 1 && rng.below(2) == 0 {
        (1 + rng.below(r), None)
    } else {
        let c1 = rng.below(r + 1);
        (c1, Some(r + 1 + rng.below(rows - r)))
    };
    let n = layout.crossover_rows(layout, a, b, c1, c2).map_err(|e| format!("aimed row crossover: {e}"))?;
    Ok(Some(match c2 {
        None => format!("one-point at row {c1} of {rows} (coding row {r}): {n} cells exchanged"),
        Some(c2) => format!("two-point, rows [{c1}, {c2}) of {rows} (coding row {r}): {n} cells exchanged"),
    }))
}

#[cfg(test)]
mod tests;
