//! HOMOLOGOUS SECTION CROSSOVER (stable section ids, the E3 option of
//! `docs/PLAN_experiments_2026-10-13.md`): a kernel's statements named by
//! what they do, the same name in every kernel of a population, and a child
//! that is one parent with one statement replaced by the other parent's
//! statement of the same name.
//!
//! A statement's id is its row (which carries its rung: where it may stand),
//! the sorted names it writes (the registers, private and workgroup arrays
//! it sets, the buffers it stores, its nested statements' included) and its
//! occurrence among the kernel's statements of that row and write set, in
//! tree order. The id comes from the statement's content, never from a gene
//! slot, so two kernels whose genes put the same section in different slots
//! still pair it. Karva crossover pairs slots: measured on nluca's 26 kernels
//! (phylu wgsl_cx_measure, 1,000 children an operator), 6, 27 and 15 of
//! 1,000 one-point, two-point and gene children were new, clean and legal.

use std::collections::{BTreeMap, BTreeSet};

use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::render::Tree;
use crate::wgsl::rules::{kernel_statements, replace_statement};

/// One statement of a kernel under its section id.
#[derive(Clone, Debug)]
pub struct Section {
    pub id: String,
    /// Its index as `rules::kernel_statements` numbers it.
    pub index: usize,
    pub tree: Tree,
}

/// The names a statement writes: the target of every register, private
/// array and workgroup array set in it, and every buffer it stores.
fn writes(t: &Tree, out: &mut BTreeSet<String>) {
    let n = t.name.as_str();
    if n.starts_with("reg.set_") || n.starts_with("parr.set_") || n.starts_with("wg.set_") {
        if let Some(target) = t.kids.first() {
            out.insert(target.name.clone());
        }
    }
    if let Some(rest) = n.strip_prefix("store.") {
        // The row's rung suffix is not part of the buffer's name.
        out.insert(format!("store.{}", rest.rsplit_once('_').map_or(rest, |(b, s)| if s.starts_with(['L', 'C', 'O']) { b } else { rest })));
    }
    for k in &t.kids {
        writes(k, out);
    }
}

/// The kernel's statements under their section ids, in tree order.
pub fn sections(root: &Tree) -> Vec<Section> {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    kernel_statements(root)
        .into_iter()
        .map(|s| {
            let mut w = BTreeSet::new();
            writes(&s.tree, &mut w);
            let key = format!("{}|{}", s.tree.name, w.into_iter().collect::<Vec<_>>().join(","));
            let n = seen.entry(key.clone()).or_default();
            *n += 1;
            Section { id: format!("{key}#{n}"), index: s.index, tree: s.tree }
        })
        .collect()
}

/// The sections both kernels hold under one id whose statements differ:
/// (index in `a`, `b`'s statement), in `a`'s tree order.
pub fn homologous(a: &Tree, b: &Tree) -> Vec<(usize, String, Tree)> {
    let theirs: BTreeMap<String, Tree> = sections(b).into_iter().map(|s| (s.id, s.tree)).collect();
    sections(a).into_iter().filter_map(|s| theirs.get(&s.id).filter(|t| **t != s.tree).map(|t| (s.index, s.id, t.clone()))).collect()
}

/// A child of `a` and `b`: `a` with one of its homologous sections (drawn
/// by `rng`) replaced by `b`'s; the section's id and the child. None when
/// the two kernels share no differing section.
pub fn section_crossover(a: &Tree, b: &Tree, kingdom: &WgslKingdom, rng: &mut u64) -> Result<Option<(String, Tree)>, String> {
    let pairs = homologous(a, b);
    if pairs.is_empty() {
        return Ok(None);
    }
    *rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    let (index, id, with) = &pairs[((*rng >> 33) as usize) % pairs.len()];
    Ok(Some((id.clone(), replace_statement(a, *index, vec![with.clone()], kingdom)?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::safe_reader::read_safe;

    fn nluca(name: &str) -> String {
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("problems/nluca_step").join(name)).expect("the kernel")
    }

    /// Two of nluca's kernels (the shipped kernel and the tuned f1) share
    /// sections by id; replacing one of the shipped kernel's by f1's gives a
    /// kernel that differs from both, and a kernel crossed with itself has
    /// no homologous section.
    #[test]
    fn two_nluca_kernels_pair_their_sections_by_what_they_write() {
        let kingdom = WgslKingdom::load_safe();
        let a = read_safe(&nluca("shipped.wgsl"), &kingdom).expect("shipped reads").tree;
        let b = read_safe(&nluca("tuned_f1.wgsl"), &kingdom).expect("f1 reads").tree;
        let ids_a: BTreeSet<String> = sections(&a).into_iter().map(|s| s.id).collect();
        assert_eq!(ids_a.len(), sections(&a).len(), "every section id is unique within a kernel");
        let pairs = homologous(&a, &b);
        println!("{} sections in the shipped kernel, {} homologous with f1 and different", ids_a.len(), pairs.len());
        assert!(!pairs.is_empty());
        assert!(homologous(&a, &a).is_empty());
        let mut rng = 7u64;
        let (id, child) = section_crossover(&a, &b, &kingdom, &mut rng).expect("replaces").expect("a pair");
        println!("swapped {id}");
        assert!(child != a && child != b);
    }
}
