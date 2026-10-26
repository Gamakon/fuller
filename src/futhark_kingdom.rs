//! The Futhark-style parallel kingdom: its rows, loaded from the generated
//! tables (`kingdoms/futhark_true/types.tsv` and `functions.tsv`, written by
//! `kingdoms/futhark_true/gen_tables.py`).
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

use std::collections::BTreeMap;

use crate::geneframe::Ty;
use crate::kingdom_contract::SymbolId;
use crate::kingdom_contract::{relations_from_rows, KingdomRelations};
use crate::typed_gep::Tree;

/// The kingdom's name in the symbol table.
pub const FUTHARK: &str = "Futhark";

/// The tables, read at compile time: the kingdom's design is a file, and the
/// loader is the only thing that has to agree with it.
const TYPES_TSV: &str = include_str!("../kingdoms/futhark_true/types.tsv");
const FUNCTIONS_TSV: &str = include_str!("../kingdoms/futhark_true/functions.tsv");

/// One row of `types.tsv`: a dual of the kingdom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FutharkDual {
    /// The dual's name, `f32.real`.
    pub dual: String,
    /// The family, `f32` / `arr` / `scn` / `fn`.
    pub slot: String,
    /// The form within the family, `real` / `f32` / `f32_f32`.
    pub form: String,
    /// The values a `?_T` placeholder of this dual can take, in order.
    pub literals: Vec<String>,
    /// A program's root may be this dual.
    pub root: bool,
    pub meaning: String,
}

/// What a row of `functions.tsv` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FutharkKind {
    /// A combinator or an elementary function.
    Function,
    /// The constant placeholder `?_T` of its dual: its value is an index into
    /// the dual's literal list.
    Const,
    /// A leaf with a fixed meaning (the input array, the index array, the
    /// constant function).
    Terminal,
}

/// One row of `functions.tsv`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FutharkRow {
    /// The row's name: `map`, `fn.f32_f32.sqrt`, `?_f32`.
    pub name: String,
    /// The ordered argument duals.
    pub inputs: Vec<Ty>,
    pub output: Ty,
    /// The Futhark text, with `{0}`, `{1}`, ... the children.
    pub spelling: String,
    pub kind: FutharkKind,
}

/// The loaded kingdom.
#[derive(Debug, Clone)]
pub struct FutharkKingdom {
    pub duals: Vec<FutharkDual>,
    /// Functions, then terminals, in table order. The symbol id of a row is
    /// its index here.
    pub rows: Vec<FutharkRow>,
}

fn data_lines(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines().skip(1).filter(|l| !l.trim().is_empty()).map(|l| l.split('\t').collect())
}

impl FutharkKingdom {
    /// Load the tables.
    pub fn load() -> Result<FutharkKingdom, String> {
        let mut duals = Vec::new();
        for f in data_lines(TYPES_TSV) {
            if f.len() < 7 {
                return Err(format!("futhark_true: types.tsv row {f:?} has {} columns, wants 7", f.len()));
            }
            duals.push(FutharkDual {
                dual: f[1].to_string(),
                slot: f[2].to_string(),
                form: f[3].to_string(),
                literals: if f[4].is_empty() { Vec::new() } else { f[4].split(',').map(|s| s.to_string()).collect() },
                root: f[5] == "yes",
                meaning: f[6].to_string(),
            });
        }
        Ty::check_dual_count(FUTHARK, duals.len(), Ty::MAX_FUTHARK_DUALS)?;
        let ty_of = |name: &str| -> Result<Ty, String> {
            duals
                .iter()
                .position(|d| d.dual == name)
                .map(|i| Ty::Futhark(i as u16))
                .ok_or_else(|| format!("futhark_true: no dual {name}"))
        };

        let mut rows = Vec::new();
        for f in data_lines(FUNCTIONS_TSV) {
            if f.len() < 7 {
                return Err(format!("futhark_true: functions.tsv row {f:?} has {} columns, wants 7", f.len()));
            }
            let (name, kind, out, args, spelling) = (f[1], f[2], f[3], f[4], f[5]);
            let inputs = if args.is_empty() {
                Vec::new()
            } else {
                args.split(';').map(&ty_of).collect::<Result<Vec<_>, _>>()?
            };
            let kind = match kind {
                "function" => FutharkKind::Function,
                "const" => FutharkKind::Const,
                "terminal" => FutharkKind::Terminal,
                other => return Err(format!("futhark_true: unknown kind {other}")),
            };
            rows.push(FutharkRow {
                name: name.to_string(),
                inputs,
                output: ty_of(out)?,
                spelling: spelling.to_string(),
                kind,
            });
        }
        Ok(FutharkKingdom { duals, rows })
    }

    /// The dual of a type, `None` for a type outside this kingdom.
    pub fn dual(&self, ty: Ty) -> Option<&FutharkDual> {
        match ty {
            Ty::Futhark(i) => self.duals.get(i as usize),
            _ => None,
        }
    }

    /// The type of a dual by name.
    pub fn ty(&self, dual: &str) -> Option<Ty> {
        self.duals.iter().position(|d| d.dual == dual).map(|i| Ty::Futhark(i as u16))
    }

    /// The roots, the duals the table marks: an ordinary array program, a
    /// scan-result program, and a reduction.
    pub fn roots(&self) -> Vec<Ty> {
        self.duals
            .iter()
            .enumerate()
            .filter(|(_, d)| d.root)
            .map(|(i, _)| Ty::Futhark(i as u16))
            .collect()
    }

    /// The length of each dual's constant array `C_T`: the number of literals
    /// it has, zero for a dual with none.
    pub fn const_len(&self) -> BTreeMap<Ty, u32> {
        self.duals
            .iter()
            .enumerate()
            .map(|(i, d)| (Ty::Futhark(i as u16), d.literals.len() as u32))
            .collect()
    }

    /// The kingdom's relations: every row with its ordered input duals, the
    /// `?_T` placeholders as the constant placeholders, and the two roots.
    pub fn relations(&self) -> KingdomRelations {
        let rows: Vec<(String, Vec<Ty>, Ty)> =
            self.rows.iter().map(|r| (r.name.clone(), r.inputs.clone(), r.output)).collect();
        let constants: Vec<&str> = self
            .rows
            .iter()
            .filter(|r| r.kind == FutharkKind::Const)
            .map(|r| r.name.as_str())
            .collect();
        relations_from_rows(FUTHARK, self.roots(), &rows, &constants)
    }

    /// Render an expressed tree as a Futhark expression.
    ///
    /// A constant placeholder becomes the literal its constant-array index
    /// names; every other row's spelling has its `{0}`, `{1}`, ... replaced by
    /// the children's renderings, in order.
    pub fn render(&self, tree: &Tree) -> Result<String, String> {
        self.render_at(tree)
    }

    fn render_at(&self, tree: &Tree) -> Result<String, String> {
        let row = self
            .rows
            .get(tree.symbol as usize)
            .ok_or_else(|| format!("futhark_true: symbol {} is not a row of this kingdom", tree.symbol))?;
        match row.kind {
            FutharkKind::Const => {
                let dual = self
                    .dual(row.output)
                    .ok_or_else(|| "futhark_true: a constant placeholder with no dual".to_string())?;
                let idx = tree
                    .const_index
                    .ok_or_else(|| format!("futhark_true: {} has no constant index", row.name))? as usize;
                dual.literals
                    .get(idx)
                    .cloned()
                    .ok_or_else(|| format!("futhark_true: {} literal {idx} is past the list", row.name))
            }
            _ => {
                let kids: Vec<String> =
                    tree.kids.iter().map(|k| self.render_at(k)).collect::<Result<Vec<_>, _>>()?;
                let mut out = row.spelling.clone();
                for (i, kid) in kids.iter().enumerate() {
                    let placeholder = format!("{{{i}}}");
                    if !out.contains(&placeholder) {
                        return Err(format!(
                            "futhark_true: {}'s spelling has no {placeholder}, so its {i}th argument is dropped",
                            row.name
                        ));
                    }
                    // Every child is substituted as an ATOM: Futhark's
                    // application is greedy, so an unbracketed child would
                    // swallow its siblings.
                    out = out.replace(&placeholder, &format!("({kid})"));
                }
                // A row whose spelling still holds a placeholder has fewer
                // children than placeholders: that is a table bug, not a gene.
                if out.contains('{') && out.contains('}') {
                    return Err(format!("futhark_true: {}'s spelling still holds a placeholder", row.name));
                }
                Ok(out)
            }
        }
    }
}

/// The kingdom's relations, loaded.
pub fn futhark_relations() -> Result<KingdomRelations, String> {
    FutharkKingdom::load().map(|k| k.relations())
}

/// The symbol id of a row by name, for the loader's tests.
pub fn symbol_id(k: &FutharkKingdom, name: &str) -> Option<SymbolId> {
    k.rows.iter().position(|r| r.name == name).map(|i| i as SymbolId)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn the_tables_load_with_the_generators_counts() {
        let k = FutharkKingdom::load().unwrap();
        assert_eq!(k.duals.len(), 4, "four types: a real, two array flavours, one function value");
        assert_eq!(k.rows.len(), 38, "34 combinators and elementary functions, 4 terminals");
        assert_eq!(k.roots().len(), 3, "an array root, a scan-result root and a reduction root");
    }

    #[test]
    fn the_kingdom_passes_the_contract_check() {
        let k = FutharkKingdom::load().unwrap();
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
        let k = FutharkKingdom::load().unwrap();
        let contract = k.relations().check().unwrap();
        let const_len = k.const_len();
        let input: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.25).collect();
        let dir = std::env::temp_dir().join("futhark_true_test").join(format!("t{}", std::process::id()));
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
                    crate::futhark_interp::Value::Arr(a) => a.clone(),
                    crate::futhark_interp::Value::Real(r) => vec![*r],
                    crate::futhark_interp::Value::Fun(_) => panic!("a program's root is not a function"),
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
        let k = FutharkKingdom::load().unwrap();
        assert!(k.rows.len() * 50 < 8841, "the kingdom is not compact enough to be worth a new one");
        assert!(k.duals.len() * 20 < 305, "the type set is not compact enough");
    }
}
