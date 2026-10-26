//! The shape-classed Futhark kingdom: the interface, the shape-polymorphic
//! program, and the checks that the class table and the Futhark compiler
//! agree.
//!
//! The 1-D kingdom decides that shape is a *value*, not a type: `arr.f32` is
//! one type however long the array is. This kingdom decides that a shape
//! **class** is a type: `mat.n.m` is the array the interface calls `[n][m]`,
//! where `n` and `m` are dimension *roles*. The sizes stay symbolic -- the
//! generated source is shape-polymorphic -- while the *relations* between the
//! dimensions become part of the type system.
//!
//! The two kingdoms coexist, and they are separate kingdoms on purpose: the
//! 1-D one keeps the programs whose shapes are only equal at run time, which
//! static classes cannot express.
//!
//! The unification happens in the generator
//! (`kingdoms/futhark_shapes/gen_tables.py`), not here. This module only
//! reads the tables, renders a program, and checks that the result is legal
//! and that Futhark accepts it.

use std::collections::BTreeMap;

use crate::futhark_kingdom::{FutharkDual, FutharkKingdom, FutharkKind};
use crate::geneframe::Ty;
use crate::typed_gep::Tree;

/// The helper the `matmul` row needs.
///
/// Futhark 0.28.1 has no `matmul` in its prelude, and its `map` cannot map
/// over a matrix's rows with a *scalar* function -- so the kingdom's `matmul`
/// row needs a helper of its own. The helper is **shape-polymorphic**, which
/// is the point: Futhark's own type checker then enforces the
/// inner-dimension relation the class table states, so a table and a compiler
/// that disagree cannot both be right.
pub const MATMUL_HELPER: &str = "def matmul [n][m][p] (A: [n][m]f32) (B: [m][p]f32) : [n][p]f32 =\n  map (\\a -> map (\\b -> reduce (+) 0.0 (map2 (*) a b)) (transpose B)) A\n";

/// The dimension roles of the kingdom's interface, in a deterministic order.
///
/// A role is a dimension variable in the generated source, so `mat.n.m` and
/// `mat.m.p` share the `m`.
pub fn roles(k: &FutharkKingdom) -> Vec<String> {
    let mut set = std::collections::BTreeSet::new();
    for d in &k.duals {
        if d.slot == "mat" {
            for r in d.form.split('.') {
                set.insert(r.to_string());
            }
        }
    }
    set.into_iter().collect()
}

/// The dimensions of a matrix class, from its dual's `form` (`n.m`).
fn dims_of(d: &FutharkDual) -> Vec<String> {
    d.form.split('.').map(|s| s.to_string()).collect()
}

/// The interface's inputs: each matrix class's terminal, which is the Futhark
/// variable that class's input binds.
fn inputs(k: &FutharkKingdom) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    for d in &k.duals {
        if d.slot != "mat" {
            continue;
        }
        let ty = k.ty(&d.dual).expect("a dual's own type");
        let term = k
            .rows
            .iter()
            .find(|r| r.kind == FutharkKind::Terminal && r.output == ty)
            .expect("every matrix class has its input terminal");
        out.push((term.spelling.clone(), dims_of(d)));
    }
    out
}

/// The whole Futhark source for a program: the helper, the shape-polymorphic
/// kernel, and the entry point with concrete sizes.
///
/// The kernel is polymorphic in the interface's roles, so the same gene
/// describes a program for *any* sizes the interface allows; the entry point
/// picks one set.
pub fn program(
    k: &FutharkKingdom,
    root: Ty,
    tree: &Tree,
    sizes: &BTreeMap<String, usize>,
) -> Result<String, String> {
    let expr = k.render(tree)?;
    let root_dims = dims_of(k.dual(root).ok_or_else(|| "the root has no dual".to_string())?);
    let ins = inputs(k);
    let size_of = |r: &String| sizes.get(r).copied().unwrap_or(1).to_string();

    let params = ins
        .iter()
        .map(|(v, d)| format!("({v}: [{}]f32)", d.join("][")))
        .collect::<Vec<_>>()
        .join(" ");
    let entry_params = ins
        .iter()
        .map(|(v, d)| format!("({v}: [{}]f32)", d.iter().map(&size_of).collect::<Vec<_>>().join("][")))
        .collect::<Vec<_>>()
        .join(" ");
    let dims = format!("[{}]", roles(k).join("]["));
    let result = format!("[{}]f32", root_dims.join("]["));
    let entry_result = format!("[{}]f32", root_dims.iter().map(&size_of).collect::<Vec<_>>().join("]["));
    let args = ins.iter().map(|(v, _)| v.clone()).collect::<Vec<_>>().join(" ");

    Ok(format!(
        "{MATMUL_HELPER}\n-- the kernel: shape-polymorphic over the interface's roles\n\
         def kernel {dims} {params} : {result} =\n  {expr}\n\n\
         -- the entry point: concrete sizes\n\
         entry main {entry_params} : {entry_result} = kernel {args}\n"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::futhark_exec::{compile_program, Verdict};
    use crate::typed_gep::{express, GeneLayout};
    use std::process::Command;
    use std::time::Duration;

    fn kingdom() -> FutharkKingdom {
        FutharkKingdom::load_shapes().expect("the shape tables load")
    }

    /// The sizes the harness picks for the roles. The source is polymorphic in
    /// them; these are only the sizes one run uses.
    fn sizes() -> BTreeMap<String, usize> {
        [("m".to_string(), 3usize), ("n".to_string(), 2), ("p".to_string(), 4)].into_iter().collect()
    }

    #[test]
    fn the_shape_kingdom_passes_the_contract_check() {
        let k = kingdom();
        let contract = k.relations().check().expect("the kingdom's own check");
        assert_eq!(k.duals.len(), 11, "nine matrix classes and two function classes");
        assert_eq!(k.rows.len(), 70, "54 operators, 11 terminals, 5 elementary row functions");
        assert_eq!(k.roots().len(), 1, "the interface's result class");
        // Every demanded class has a terminal, which the contract check
        // enforces; and every class is in the SHAPE family, so the two
        // kingdoms' duals cannot be confused.
        for ty in contract.types() {
            assert!(k.family.owns(*ty), "{ty:?} is not in the kingdom's family");
        }
    }

    /// The design's central numeric claim, checked by the engine itself: the
    /// class set's frames and the gene's width.
    #[test]
    fn the_engine_derives_the_counts_the_design_predicted() {
        let k = kingdom();
        let contract = k.relations().check().expect("the kingdom's own check");
        let root = k.ty("mat.n.p").expect("the root class");
        let heads = contract.heads_from_program_size(root, 3).expect("heads");
        let bounds = contract.bounds(root, &heads).expect("bounds");
        let layout = GeneLayout::from_program_size(&contract, root, 3).expect("layout");
        assert_eq!(bounds.types.len(), 11, "eleven root-reachable frames");
        assert_eq!(layout.width, 322, "the gene's width at program size 3");
        assert_eq!(bounds.l, 75, "the leaf bound");
    }

    /// Legality by construction: every random gene expresses, with no repair
    /// and no rejection. The shape classes make an ill-shaped program
    /// UNREPRESENTABLE, so there is nothing to check after the fact.
    #[test]
    fn every_random_gene_expresses_a_legal_program() {
        let k = kingdom();
        let contract = k.relations().check().expect("the kingdom's own check");
        let root = k.ty("mat.n.p").expect("the root class");
        let layout = GeneLayout::from_program_size(&contract, root, 3).expect("layout");
        let const_len = k.const_len();
        let mut rng: u64 = 0x5EED_1234_ABCD_0002;
        let mut expressed = 0usize;
        for _ in 0..500 {
            let mut gene = vec![0u32; layout.width];
            for (pos, cell) in gene.iter_mut().enumerate() {
                let alphabet = layout.alphabet(&contract, pos, &const_len);
                assert!(!alphabet.is_empty(), "cell {pos} has an empty alphabet");
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *cell = alphabet[((rng >> 33) as usize) % alphabet.len()];
            }
            let e = express(&contract, &layout, &gene, &const_len).expect("a random gene is legal");
            let tree = e.tree();
            // The rendered program must mention every argument of every node:
            // the renderer checks that, so a silent drop is impossible.
            k.render(&tree).expect("the program renders");
            expressed += 1;
        }
        assert_eq!(expressed, 500);
    }

    /// The decisive check: Futhark accepts the rendered programs. The helper
    /// is shape-polymorphic, so a class table that disagreed with Futhark
    /// about the inner dimensions could not compile.
    #[test]
    fn futhark_accepts_the_rendered_programs() {
        let have = |p: &str| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have("futhark") {
            eprintln!("skipped: futhark is not installed");
            return;
        }
        let k = kingdom();
        let contract = k.relations().check().expect("the kingdom's own check");
        let root = k.ty("mat.n.p").expect("the root class");
        let layout = GeneLayout::from_program_size(&contract, root, 3).expect("layout");
        let const_len = k.const_len();
        let dir = std::env::temp_dir().join("futhark_shapes_test").join(format!("t{}", std::process::id()));
        let mut rng: u64 = 0x5EED_1234_ABCD_0003;
        let mut compiled = 0usize;
        for _ in 0..3 {
            let mut gene = vec![0u32; layout.width];
            for (pos, cell) in gene.iter_mut().enumerate() {
                let alphabet = layout.alphabet(&contract, pos, &const_len);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *cell = alphabet[((rng >> 33) as usize) % alphabet.len()];
            }
            let e = express(&contract, &layout, &gene, &const_len).expect("a random gene is legal");
            let source = program(&k, root, &e.tree(), &sizes()).expect("the program renders");
            let (verdict, _elapsed) = compile_program(&source, &dir, Duration::from_secs(120));
            match verdict {
                Verdict::Compiled => compiled += 1,
                Verdict::Refused(e) => panic!("futhark refused a legal program: {e}\n{source}"),
                Verdict::TimedOut => panic!("futhark did not finish on a legal program\n{source}"),
            }
        }
        assert_eq!(compiled, 3);
    }
}
