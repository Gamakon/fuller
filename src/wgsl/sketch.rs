//! THE SKETCH'S OUTLINE (docs/PLAN_experiments_2026-10-13.md, E1b): the
//! part of a Gram kernel the problem fixes, drawn by
//! [`super::render::sample_sketch`] with every other subtree a random hole,
//! and recognised here in any tree, so a search can keep its variation to
//! the holes: an offspring whose tree has lost the outline is not a sketch
//! kernel.
//!
//! A tree holds the outline when, for one f32 register `R`:
//! - its root is `kernel.entry_safe`;
//! - among the body's top-level statements (plain blocks opened) is the item
//!   loop `loop.items_boundm` over `bound.m`, and among ITS body's top-level
//!   statements the update `R = R + <value>` (`reg.set_f_L1` of `R` and
//!   `arith.add` of `reg.get_f R` and any value);
//! - among the epilogue's top-level statements is the store of `R` into the
//!   thread's own cell, `store.h_O0` of `safe.cell_of(safe.thread_row(..),
//!   safe.thread_col(..))` and `reg.get_f R` (the kingdom's guarded
//!   read-add-write `h[cell] = h[cell] + R`).
//!
//! The register is not named by the outline: the reader allocates registers
//! afresh when a kernel text is read back (a rewrite child's), so the same
//! kernel may hold its accumulator in another register.

use super::render::Tree;

/// The statements of a block at its top level, plain blocks opened (a
/// `stmt.block_*` holding a `block.one_`/`seqN_` block).
fn statements<'t>(b: &'t Tree, out: &mut Vec<&'t Tree>) {
    for s in &b.kids {
        let plain = s.kids.len() == 1 && s.name.starts_with("stmt.block_") && ["block.one_", "block.seq2_", "block.seq3_", "block.seq4_"].iter().any(|x| s.kids[0].name.starts_with(x));
        if plain {
            statements(&s.kids[0], out);
        } else {
            out.push(s);
        }
    }
}

/// The register a `reg.get_f` reads, when `t` is one.
fn read_of(t: &Tree) -> Option<&str> {
    (t.name == "reg.get_f" && t.kids.len() == 1 && t.kids[0].kids.is_empty() && t.kids[0].name.starts_with("reg.f")).then(|| t.kids[0].name.as_str())
}

/// The accumulators the item loops of the body update (`R = R + v`).
fn accumulators(body: &Tree) -> Vec<&str> {
    let mut top = Vec::new();
    statements(body, &mut top);
    let mut out = Vec::new();
    for s in top.into_iter().filter(|s| s.name == "loop.items_boundm" && s.kids.len() == 2 && s.kids[0].name == "bound.m") {
        let mut inner = Vec::new();
        statements(&s.kids[1], &mut inner);
        for u in inner {
            if u.name != "reg.set_f_L1" || u.kids.len() != 2 || !u.kids[0].kids.is_empty() || !u.kids[0].name.starts_with("reg.f") {
                continue;
            }
            let add = &u.kids[1];
            if add.name == "arith.add" && add.kids.len() == 2 && read_of(&add.kids[0]) == Some(u.kids[0].name.as_str()) {
                out.push(u.kids[0].name.as_str());
            }
        }
    }
    out
}

/// The register the epilogue's own-cell stores write (`h[cell] = h[cell] + R`).
fn stored(epilogue: &Tree) -> Vec<&str> {
    let mut top = Vec::new();
    statements(epilogue, &mut top);
    top.into_iter()
        .filter(|e| e.name == "store.h_O0" && e.kids.len() == 2)
        .filter(|e| {
            let c = &e.kids[0];
            c.name == "safe.cell_of" && c.kids.len() == 2 && c.kids[0].name == "safe.thread_row" && c.kids[1].name == "safe.thread_col"
        })
        .filter_map(|e| read_of(&e.kids[1]))
        .collect()
}

/// The accumulator register of the outline `t` holds, or None when it does
/// not hold the outline.
pub fn outline_register(t: &Tree) -> Option<String> {
    if t.name != "kernel.entry_safe" || t.kids.len() != 3 {
        return None;
    }
    let updated = accumulators(&t.kids[1]);
    stored(&t.kids[2]).into_iter().find(|r| updated.contains(r)).map(str::to_string)
}

/// Whether `t` holds the sketch's outline.
pub fn holds_outline(t: &Tree) -> bool {
    outline_register(t).is_some()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::wgsl::check::Checker;
    use crate::wgsl::loader::WgslKingdom;
    use crate::wgsl::render::{cost, gram_role, gram_shipped_safe_tree, sample_sketch, sample_within, dual_ty};
    use crate::wgsl::safe_karva;
    use crate::wgsl::target::Target;

    /// The shipped kernel with its store at the epilogue's top level is a
    /// completion of the outline (as written its store is under its own
    /// range guard, `if (in range) { store }`, a second check of the one
    /// the kingdom's store carries, and a store under a condition is not
    /// the outline's); a random kernel of the free sampler almost never is.
    #[test]
    fn the_shipped_kernel_holds_the_outline() {
        let k = WgslKingdom::load_safe();
        let mut shipped = gram_shipped_safe_tree(&k).unwrap();
        assert!(!holds_outline(&shipped), "the store under the shipped kernel's guard");
        let store = shipped.kids[2].kids[0].kids[1].clone();
        assert_eq!(store.name, "block.one_O0");
        shipped.kids[2] = store;
        assert_eq!(outline_register(&shipped).as_deref(), Some("reg.f0"));
        let (role, root, bound) = (gram_role(), dual_ty(&k, "kernel.safe").unwrap(), safe_karva::default_bound(&k).unwrap());
        let free = (0..1000u64).filter(|&s| holds_outline(&sample_within(&k, &role, root, 7, s, &bound).unwrap())).count();
        println!("free sampler: {free} of 1000 hold the outline");
        assert!(free < 10, "{free} of 1000 free draws hold the outline");
    }

    /// 1,000 SKETCH DRAWS (seeds 0..1000, depth 7, the fitness's cost
    /// bound): every one passes the legality checker, holds the outline,
    /// is within the bound, renders, validates by naga (the recorded
    /// target, no device), and is distinct from every other by
    /// text; encoded into a chromosome (head 128, 128 tail slots, the dumps'
    /// layout) every gene decodes, unfolds and projects to the tree drawn;
    /// its rendered text read back by the safe reader still holds the
    /// outline (a rewrite child enters through the reader).
    #[test]
    fn a_thousand_sketch_draws_are_legal_distinct_and_hold_the_outline() {
        use crate::homeotic;
        use crate::karva::karva_to_terms_generic;
        let k = WgslKingdom::load_safe();
        let role = gram_role();
        let bound = safe_karva::default_bound(&k).unwrap();
        let checker = Checker::new(&k, &role, &Target::recorded());
        let (h, slots) = (128usize, 128usize);
        let mut texts = BTreeSet::new();
        let (mut nodes, mut value_nodes) = (0usize, 0usize);
        for seed in 0..1000u64 {
            let tree = sample_sketch(&k, &role, 7, seed, &bound).unwrap();
            checker.check(&tree).unwrap_or_else(|e| panic!("seed {seed}: the checker: {e}\n{}", tree.to_sexpr()));
            let acc = outline_register(&tree).unwrap_or_else(|| panic!("seed {seed}: no outline: {}", tree.to_sexpr()));
            let c = cost(&tree);
            assert!(c.per_item <= bound.per_item && c.base <= bound.base, "seed {seed}: over the bound: {c:?}");
            let text = safe_karva::render_tree(&tree, &k).unwrap_or_else(|e| panic!("seed {seed}: does not render: {e}")).text;
            crate::wgsl::render::validate_text(&text).unwrap_or_else(|e| panic!("seed {seed}: naga: {e}\n{text}"));
            if seed == 0 {
                println!("sketch seed 0:\n{text}");
            }
            assert!(texts.insert(text.clone()), "seed {seed}: a kernel drawn before:\n{text}");
            let back = safe_karva::encode(&tree, h, slots, seed).and_then(|c| {
                let pset = homeotic::pset_with_hrefs(&c.pset, slots);
                let decoded: Vec<String> = c.genes.iter().map(|(hd, tl)| karva_to_terms_generic(hd, tl, &pset)).collect::<Result<_, _>>()?;
                safe_karva::build(&safe_karva::unfold_decoded(&decoded, c.folded.filled)?, &k, &role)
            });
            match back {
                Ok(t) => assert!(t == tree, "seed {seed}: the genome decodes to another tree: {}", crate::wgsl::render::first_difference(&tree, &t).unwrap_or_default()),
                Err(e) => panic!("seed {seed}: the genome: {e}"),
            }
            let read = crate::wgsl::safe_reader::read_safe(&text, &k).unwrap_or_else(|e| panic!("seed {seed}: the reader: {e}\n{text}")).tree;
            assert!(holds_outline(&read), "seed {seed}: read back without the outline:\n{text}\n{}", read.to_sexpr());
            nodes += tree.size();
            // The update's value: the hole the search completes.
            fn value_size(t: &Tree) -> Option<usize> {
                if t.name == "reg.set_f_L1" && t.kids.len() == 2 && t.kids[1].name == "arith.add" {
                    return Some(t.kids[1].kids[1].size());
                }
                t.kids.iter().find_map(value_size)
            }
            value_nodes += value_size(&tree).unwrap_or(0);
            assert!(acc.starts_with("reg.f"));
        }
        println!("1000 sketch draws: legal, the outline held, distinct, through the genome; mean {:.1} nodes, the value hole {:.1}", nodes as f64 / 1000.0, value_nodes as f64 / 1000.0);
    }

    /// A tree that loses one part of the outline does not hold it: the
    /// store's value another register, the update's register another, the
    /// loop's bound another, the cell another thread's (a fallback cell).
    #[test]
    fn a_tree_without_a_part_of_the_outline_does_not_hold_it() {
        let k = WgslKingdom::load_safe();
        let bound = safe_karva::default_bound(&k).unwrap();
        let t = sample_sketch(&k, &gram_role(), 7, 3, &bound).unwrap();
        let acc = outline_register(&t).unwrap();
        fn rename(t: &Tree, pred: &dyn Fn(&Tree) -> bool, f: &dyn Fn(&Tree) -> Tree) -> Tree {
            if pred(t) {
                return f(t);
            }
            Tree { kids: t.kids.iter().map(|x| rename(x, pred, f)).collect(), ..t.clone() }
        }
        let other = if acc == "reg.f1" { "reg.f2" } else { "reg.f1" };
        let swap_reg = |kid: usize| {
            move |x: &Tree| {
                let mut y = x.clone();
                if kid == 1 {
                    y.kids[1] = Tree::new("reg.get_f", y.kids[1].ty, vec![Tree::new(other, y.kids[1].kids[0].ty, vec![])]);
                } else {
                    y.kids[0] = Tree::new(other, y.kids[0].ty, vec![]);
                }
                y
            }
        };
        let store = rename(&t, &|x| x.name == "store.h_O0", &swap_reg(1));
        assert!(!holds_outline(&store), "the store of another register");
        let update = rename(&t, &|x| x.name == "reg.set_f_L1" && x.kids[1].name == "arith.add" && read_of(&x.kids[1].kids[0]) == Some(acc.as_str()), &swap_reg(0));
        assert!(!holds_outline(&update), "the update of another register");
        let unbounded = rename(&t, &|x| x.name == "bound.m", &|x| Tree::fallback(x.ty));
        assert!(!holds_outline(&unbounded), "the loop's bound a fallback");
        let cell = rename(&t, &|x| x.name == "safe.cell_of", &|x| Tree::fallback(x.ty));
        assert!(!holds_outline(&cell), "the cell a fallback");
        assert!(holds_outline(&t));
    }
}
