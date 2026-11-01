//! DEPRECATED: the 1-D Futhark kingdom. New work uses the Futhark kingdom (docs/PLAN_futhark_kingdom.md).
//!
//! The 1-D Futhark kingdom: its rows, loaded from the generated
//! tables (`kingdoms/futhark_1d/types.tsv` and `functions.tsv`, written by
//! `kingdoms/futhark_1d/gen_tables.py`).
//!
//! The kingdom evolves a typed, high-level *parallel program*: a tree of
//! combinators over arrays (`map`, `filter`, `zip`, `scan`, `gather`,
//! `reduce`). It exists to answer the search-space problem of
//! `kingdoms/wgsl_true`, where 305 types and 8,841 rows bought mostly ways to
//! spell a statement nest.
//!
//! Three decisions keep it small:
//!
//! * **Shape is a value, not a type.** `arr.f32` is one type however long the
//!   array is, so the type count does not follow the program size.
//! * **Nesting is free.** The combinators are value-level, so a program's
//!   depth costs no types -- the thing `wgsl_true` spent 112 types and 5,295
//!   rows on.
//! * **A function value is a type, not a lambda.** `fn.f32_f32` is a type
//!   with its own reading frame holding the elementary functions, so `map`
//!   composes without the GEP ever needing a bound variable: the elementary
//!   function's parameter is bound by its spelling, and its argument (a
//!   captured literal) is an ordinary child slot.
//!
//! The genetic representation is the Karva gene. The Futhark *source* is a
//! rendering of the expressed tree -- regenerated on every evaluation, never
//! read back into the genome.

use crate::futhark_table::{FutharkFamily, FutharkKingdom};
use crate::kingdom_contract::KingdomRelations;

/// The 1-D Futhark kingdom's name in the symbol table. It is a SEPARATE
/// kingdom from the Futhark kingdom ([`crate::futhark::FUTHARK`]): the two
/// live in separate [`crate::geneframe::Ty`] families so their dual codes
/// cannot collide.
pub const FUTHARK_1D: &str = "Futhark 1-D";

/// The tables, read at compile time: the kingdom's design is a file, and the
/// loader is the only thing that has to agree with it.
const TYPES_TSV: &str = include_str!("../kingdoms/futhark_1d/types.tsv");
const FUNCTIONS_TSV: &str = include_str!("../kingdoms/futhark_1d/functions.tsv");

impl FutharkKingdom {
    /// Load the 1-D Futhark kingdom's tables.
    pub fn load_1d() -> Result<FutharkKingdom, String> {
        Self::from_tables(FutharkFamily::Plain, FUTHARK_1D, TYPES_TSV, FUNCTIONS_TSV)
    }
}

/// The 1-D Futhark kingdom's relations, loaded.
pub fn futhark_1d_relations() -> Result<KingdomRelations, String> {
    FutharkKingdom::load_1d().map(|k| k.relations())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn the_tables_load_with_the_generators_counts() {
        let k = FutharkKingdom::load_1d().unwrap();
        assert_eq!(k.duals.len(), 4, "four types: a real, two array flavours, one function value");
        assert_eq!(k.rows.len(), 38, "34 combinators and elementary functions, 4 terminals");
        assert_eq!(k.roots().len(), 3, "an array root, a scan-result root and a reduction root");
    }

    #[test]
    fn the_kingdom_passes_the_contract_check() {
        let k = FutharkKingdom::load_1d().unwrap();
        let contract = k.relations().check().expect("the kingdom's own check");
        // Every type a symbol asks for has a terminal, so no leaf is missing.
        for dual in &k.duals {
            let ty = k.ty(&dual.dual).unwrap();
            if contract.reachable(ty).contains(&ty) {
                assert!(!contract.terminals_of(ty).is_empty(), "{} has no terminal", dual.dual);
            }
        }
    }

    /// The kingdom's central claim, checked against the real tools: a random
    /// gene renders to a Futhark program, that program compiles, runs, and
    /// agrees with the kingdom's own interpreter element for element.
    ///
    /// Skipped when the tools are not installed: this test needs `futhark`
    /// and a C compiler, and the rest of the suite must not depend on them.
    #[test]
    fn the_rendered_programs_agree_with_the_interpreter() {
        let have = |p: &str| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have("futhark") || !have("cc") {
            eprintln!("skipped: futhark or cc is not installed");
            return;
        }
        let k = FutharkKingdom::load_1d().unwrap();
        let contract = k.relations().check().unwrap();
        let const_len = k.const_len();
        let input: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.25).collect();
        let dir = std::env::temp_dir().join("futhark_1d_test").join(format!("t{}", std::process::id()));
        let mut checked = 0;
        for root_name in ["arr.f32", "scn.f32", "f32.real"] {
            let root = k.ty(root_name).unwrap();
            let layout = crate::typed_gep::GeneLayout::from_program_size(&contract, root, 3).unwrap();
            let mut rng: u64 = 0x5EED_1234_ABCD_0001 ^ root_name.len() as u64;
            for _ in 0..4 {
                let mut gene = vec![0u32; layout.width];
                for (pos, cell) in gene.iter_mut().enumerate() {
                    let alphabet = layout.alphabet(&contract, pos, &const_len);
                    assert!(!alphabet.is_empty(), "cell {pos} has an empty alphabet");
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    *cell = alphabet[((rng >> 33) as usize) % alphabet.len()];
                }
                let e = crate::typed_gep::express(&contract, &layout, &gene, &const_len)
                    .expect("a random gene is legal");
                let tree = e.tree();
                let scalar = root_name == "f32.real";
                let result = if scalar { "f32" } else { "[]f32" };
                let source = format!(
                    "entry main (xs: []f32) : {result} = {}\n",
                    k.render(&tree).unwrap()
                );
                let expected = k.eval(&tree, &input).unwrap();
                let got = crate::futhark_exec::compile_and_run(&source, scalar, &input, &dir).unwrap();
                let want = match &expected {
                    crate::futhark_1d_interp::Value::Arr(a) => a.clone(),
                    crate::futhark_1d_interp::Value::Real(r) => vec![*r],
                    crate::futhark_1d_interp::Value::Fun(_) => panic!("a program's root is not a function"),
                };
                assert_eq!(got.len(), want.len(), "the program and the interpreter disagree on the length\n{source}");
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    let same = if a == b {
                        // Equal, including the two infinities: `inf - inf` is
                        // NaN, so the tolerance test below cannot see it.
                        true
                    } else if a.is_nan() && b.is_nan() {
                        true
                    } else {
                        let scale = a.abs().max(b.abs()).max(1.0);
                        (a - b).abs() <= 1e-4 * scale
                    };
                    assert!(same, "value {i}: futhark says {a}, the interpreter says {b}\n{source}");
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 12, "every root and every program was checked");
    }

    #[test]
    fn the_symbol_table_is_two_orders_of_magnitude_smaller_than_wgsl_true() {
        let k = FutharkKingdom::load_1d().unwrap();
        assert!(k.rows.len() * 50 < 8841, "the kingdom is not compact enough to be worth a new one");
        assert!(k.duals.len() * 20 < 305, "the type set is not compact enough");
    }
}
