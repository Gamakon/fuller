//! The nucleotable data model, adopted into fuller and owned here.
//!
//! Subsumed from `/Users/andrewmorgan/Dev/minkymorgan/nucleotable` (schema +
//! kingdom defs in `nucleotable/`). The design: one MASTER symbol table whose
//! rows carry a kingdom + a TYPED, many-hot arity signature; a "kingdom" is a
//! query over that table returning a stable pset; karva chromosomes are rows
//! (the `geneframe` layout). This replaces the flat hand-built `master_pset`.
//!
//! Implemented in pure Rust (no DuckDB dependency): the data model is the
//! valuable part; DuckDB/Parquet/Arrow are the store/exchange layer and can be
//! added later as an optional feature when SQL-evolution / exchange is needed.
//!
//! Types (the many-hot arity columns, base set — extensible per kingdom):
//! S(tring) I(nteger) F(loat) B(oolean) A(rray) L(ist).
//!
//! Two row kinds (design 2026-10-09, `docs/PLAN_population_dag.md` §1, not
//! yet built here): the population table holds ordinary chromosomes and
//! homeotic chromosomes. A homeotic chromosome is one population-wide shared
//! definition: a gene tree with a stable row id assigned when it is interned,
//! a root type (its dual, so a reference is type-checked by the typed
//! projection like a terminal), memoised row values and a referrer count.
//! Ordinary genes reference it by href, and an href names the stable id,
//! never a position in the homeotic tail. Fold is a variation operator that
//! adds homeotic rows, rewrites ordinary rows to reference them and re-sorts
//! the homeotic tail by use; execution follows a separate level schedule (a
//! definition after everything it references). The memoised top of the
//! homeotic tail is the expressed frame. The symbol table above is unchanged
//! by this: it types both row kinds.

use std::collections::BTreeMap;

pub mod formula;
pub mod master;


/// The value types a symbol's slots can carry — the `in_*`/`out_*` columns of
/// `docs/design/DataModel.md`, as enum variants so `Arity`'s many-hot maps are
/// keyed by them. THE FULL SET IS DECLARED NOW so the row structure is stable
/// over time: a new kingdom REUSES these columns, it does not add a variant and
/// migrate every row. Most types are zero on any given row. A new type is a
/// genuinely new value class (not a new kingdom), and only then is a variant
/// added — and because every match on `Ty` is non-exhaustive (a `_` arm), that
/// addition stays backward-compatible.
///
/// Ordering: the base types come FIRST and never move. Symbolic regression has ONE data type, `F`; a tree's
/// transcendental depth is a PROPERTY of it (`kingdoms/tsr/depth/tables.tsv`, the `depth` up flows and the
/// `in0.depth <= 1` guard on a raising unary), not a type. The old depth ladder `T1`/`T2` was deleted with the move
/// to the four kinds of row (plan 7.7, task K1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ty {
    // ---- Base types (Symbolic Regression / SQL / REGEX kingdoms) ----
    S, // String
    I, // Integer
    F, // Float
    B, // Boolean
    A, // Array
    L, // List

    // ---- NLP-English input types (NLP kingdom) — spaCy entity/phrase labels ----
    Person,
    Org,
    Gpe,
    Loc,
    Norp,
    Fac,
    Event,
    Product,
    Date,
    Time,
    Money,
    Quantity,
    Cardinal,
    Percent,
    Np,     // noun phrase
    Ap,     // adjective phrase
    Clause,
    Verb,

    // ---- NLP-English Phylo OUTPUT types (NLP kingdom) ----
    Entity,     // 📦
    Relation,   // 🔗
    Metric,     // 📊
    Procedure,  // ⚙️
    Narrative,  // 📜

    // ---- BotjiKingdom types ----
    Addr, // botji address string

    // ---- REGEX kingdom types ----
    // Appended last so the base types keep their positions (the ordering rule above).
    // Integer repeat counts reuse `Ty::I`; they are not needed until counted
    // repetition is added (Phase 1 omits it).
    Char,      // one literal byte
    CharClass, // a set of bytes ([a-z], \d, …)
    Pattern,   // a regex sub-tree — the kingdom's root/hub type

    // ---- Table-defined type families ----
    /// A dual `slot.form` of the WGSL kingdom: the index of its row in
    /// `kingdoms/wgsl/types.tsv` (ref `T001` → 0). One variant for the whole
    /// family, so a new dual is a table row, not a variant; its device code
    /// is [`Ty::TABLE_CODE_BASE`] + the index ([`Ty::code`]). Appended last.
    Wgsl(u16),
    /// A dual of the deprecated 1-D Futhark kingdom: the index of its row in
    /// `kingdoms/futhark_1d/types.tsv` (ref `T001` → 0). Its device code is
    /// [`Ty::FUTHARK_1D_CODE_BASE`] + the index. Appended last.
    Futhark1d(u16),
    /// A type of the Futhark kingdom: a data type of a view (its index in the view's table,
    /// `kingdoms/futhark/sig/<problem>/m6/tables.tsv`), or a dual of the pixel family (its row in
    /// `kingdoms/futhark/pixel/<shader>/types.tsv`, ref `T001` → 0). Its device code
    /// is [`Ty::FUTHARK_CODE_BASE`] + the index.
    ///
    /// A SEPARATE family from [`Ty::Futhark1d`], not another range of the same
    /// one: the two kingdoms coexist (the 1-D Futhark kingdom keeps the programs whose
    /// shapes are only equal at run time), and a type code must not mean one
    /// thing in one kingdom and another in the other. Two kingdoms that shared
    /// a family would make `Ty::Futhark1d(0)` ambiguous, and `homologous()`
    /// compares roots -- so the families are kept apart. Appended last.
    Futhark(u16),
}

impl Ty {
    /// Every variant, in declaration order. [`Ty::code`] is the position here.
    pub const ALL: [Ty; 33] = [
        Ty::S, Ty::I, Ty::F, Ty::B, Ty::A, Ty::L,
        Ty::Person, Ty::Org, Ty::Gpe, Ty::Loc, Ty::Norp, Ty::Fac, Ty::Event, Ty::Product,
        Ty::Date, Ty::Time, Ty::Money, Ty::Quantity, Ty::Cardinal, Ty::Percent, Ty::Np, Ty::Ap,
        Ty::Clause, Ty::Verb,
        Ty::Entity, Ty::Relation, Ty::Metric, Ty::Procedure, Ty::Narrative,
        Ty::Addr,
        Ty::Char, Ty::CharClass, Ty::Pattern,
    ];

    /// The type's code on the device: THE ONE SPELLING of a `Ty` as a `u32`
    /// (the node's `ty_code` word, the `out_ty`/`in_ty` tables). A base variant's code is its declaration index from 0
    /// (what `ty as u32` was before the payload variant made the cast
    /// uncompilable); a WGSL dual's code is [`Ty::TABLE_CODE_BASE`] + its row.
    pub fn code(self) -> u32 {
        match self {
            Ty::Wgsl(i) => {
                assert!(usize::from(i) < Ty::MAX_WGSL_DUALS, "our software has a bug: Wgsl({i}) is past its {} codes and would read back as Futhark1d", Ty::MAX_WGSL_DUALS);
                Ty::TABLE_CODE_BASE + u32::from(i)
            }
            Ty::Futhark1d(i) => {
                assert!(
                    usize::from(i) < Ty::MAX_FUTHARK_1D_DUALS,
                    "our software has a bug: Futhark1d({i}) is past its {} codes and would read back as itself",
                    Ty::MAX_FUTHARK_1D_DUALS
                );
                Ty::FUTHARK_1D_CODE_BASE + u32::from(i)
            }
            Ty::Futhark(i) => {
                assert!(
                    usize::from(i) < Ty::MAX_FUTHARK_DUALS,
                    "our software has a bug: Futhark({i}) is past its {} codes and would read back as itself",
                    Ty::MAX_FUTHARK_DUALS
                );
                Ty::FUTHARK_CODE_BASE + u32::from(i)
            }
            base => Ty::ALL
                .iter()
                .position(|t| *t == base)
                .map(|i| i as u32)
                .unwrap_or_else(|| unreachable!("every base variant is in Ty::ALL (test-pinned)")),
        }
    }

    /// The first code a table-defined type family uses; base variants stay
    /// below it.
    pub const TABLE_CODE_BASE: u32 = 256;

    /// The most duals a WGSL kingdom may number: its code range
    /// `TABLE_CODE_BASE..TABLE_CODE_BASE + MAX_WGSL_DUALS`.
    pub const MAX_WGSL_DUALS: usize = 4096;

    /// The 65,536 codes after the WGSL family were the deleted TSR dual family's (`Ty::Tsr`, plan 7.7 K1). They
    /// stay unassigned so that no device code of the families after them moved; `from_code` reads them as `None`.
    pub const RETIRED_TSR_CODES: u32 = u16::MAX as u32 + 1;

    /// The first code of the 1-D Futhark family: after the WGSL family (4,096 duals) and the retired TSR range.
    pub const FUTHARK_1D_CODE_BASE: u32 = Ty::TABLE_CODE_BASE + Ty::MAX_WGSL_DUALS as u32 + Ty::RETIRED_TSR_CODES;

    /// The most duals the 1-D Futhark kingdom may number: its index is a `u16`.
    pub const MAX_FUTHARK_1D_DUALS: usize = u16::MAX as usize + 1;

    /// The first code of the Futhark kingdom's family; the 1-D Futhark
    /// family owns `FUTHARK_1D_CODE_BASE..FUTHARK_CODE_BASE`.
    pub const FUTHARK_CODE_BASE: u32 = Ty::FUTHARK_1D_CODE_BASE + Ty::MAX_FUTHARK_1D_DUALS as u32;

    /// The most duals a Futhark kingdom may number: its index is
    /// a `u16`.
    pub const MAX_FUTHARK_DUALS: usize = u16::MAX as usize + 1;

    /// Load-time guard for a kingdom's dual numbering: `Err` when `count`
    /// duals exceed the `max` codes of its range, in every build.
    pub fn check_dual_count(kingdom: &str, count: usize, max: usize) -> Result<(), String> {
        if count > max {
            return Err(format!("our software has a bug: kingdom {kingdom} has {count} duals, more than its {max} codes"));
        }
        Ok(())
    }

    /// The inverse of [`Ty::code`]: a base variant by index, a WGSL dual by
    /// offset from [`Ty::TABLE_CODE_BASE`], a 1-D Futhark dual by offset from
    /// [`Ty::FUTHARK_1D_CODE_BASE`], a Futhark dual by offset from
    /// [`Ty::FUTHARK_CODE_BASE`].
    pub fn from_code(code: u32) -> Option<Ty> {
        if code >= Ty::FUTHARK_CODE_BASE {
            return u16::try_from(code - Ty::FUTHARK_CODE_BASE).ok().map(Ty::Futhark);
        }
        if code >= Ty::FUTHARK_1D_CODE_BASE {
            return u16::try_from(code - Ty::FUTHARK_1D_CODE_BASE).ok().map(Ty::Futhark1d);
        }
        if code >= Ty::TABLE_CODE_BASE {
            // A code in the retired TSR range is no type.
            return u16::try_from(code - Ty::TABLE_CODE_BASE).ok().filter(|i| usize::from(*i) < Ty::MAX_WGSL_DUALS).map(Ty::Wgsl);
        }
        Ty::ALL.get(code as usize).copied()
    }
}

/// A typed many-hot arity signature: how many inputs / outputs of each type.
/// `acquire: in={ORG:2, MONEY:1}` style, but for the base SR types here it is
/// usually a single `F` count (e.g. `Add: in F=2, out F=1`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Arity {
    pub inputs: BTreeMap<Ty, u32>,
    pub outputs: BTreeMap<Ty, u32>,
}

impl Arity {
    /// Total input arity = sum across all type slots (classical GEP arity).
    pub fn total_in(&self) -> u32 {
        self.inputs.values().sum()
    }
    /// Convenience for a uniform-typed function: `n` inputs of `t`, one `t` out.
    pub fn uniform(t: Ty, n: u32) -> Self {
        let mut a = Arity::default();
        if n > 0 {
            a.inputs.insert(t, n);
        }
        a.outputs.insert(t, 1);
        a
    }
}

/// One row of the master symbol table. `symbol > 0` = function; `symbol < 0`
/// = terminal (the nucleotable convention). `semantic_id` is what fuller
/// rewrites on (the `Math` op); `alias` is the target-language name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    pub kingdom: String,
    pub symbol: i64,
    pub symbol_name: String,
    pub alias: String,
    /// What this op COMPUTES — one of the `Math` semantic ids. The key fuller
    /// rewrites on (a kingdom may give the same semantic id several aliases).
    pub semantic_id: String,
    pub arity: Arity,
}

/// The master symbol table: all rows, all kingdoms. A "kingdom" is a filter.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    rows: Vec<Symbol>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable { rows: Vec::new() }
    }

    pub fn push(&mut self, s: Symbol) {
        self.rows.push(s);
    }

    /// The pset for a kingdom = all rows with that kingdom (the kingdom-query).
    pub fn kingdom(&self, kingdom: &str) -> Vec<&Symbol> {
        self.rows.iter().filter(|s| s.kingdom == kingdom).collect()
    }

    /// MaxArity for a kingdom = max total input arity across its functions.
    /// (Derived per kingdom, exactly as nucleotable computes it in SQL.)
    pub fn max_arity(&self, kingdom: &str) -> u32 {
        self.kingdom(kingdom)
            .iter()
            .filter(|s| s.symbol > 0)
            .map(|s| s.arity.total_in())
            .max()
            .unwrap_or(0)
    }

    /// All distinct kingdom names present.
    pub fn kingdoms(&self) -> Vec<String> {
        let mut ks: Vec<String> = self.rows.iter().map(|s| s.kingdom.clone()).collect();
        ks.sort();
        ks.dedup();
        ks
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Build the master symbol table with the kingdoms fuller ships.
///
/// The "Symbolic Regression" kingdom is the full `Math` op set (what the live
/// engine uses) — the typed-arity, kingdom-keyed replacement for the old flat
/// `master_pset()`. Other kingdoms (SQL, REGEX, NLP) come from the subsumed
/// nucleotable defs and are added as they are needed.
pub fn master_table() -> SymbolTable {
    let mut t = SymbolTable::new();
    let f = Ty::F;

    // (semantic_id, alias, arity_in_floats) for the Symbolic Regression kingdom.
    // All real-domain Math ops; arity is the float count (uniform F typing).
    let sr: &[(&str, &str, u32)] = &[
        ("add", "+", 2),
        ("sub", "-", 2),
        ("mul", "*", 2),
        ("div", "/", 2),
        ("neg", "neg", 1),
        ("sin", "sin", 1),
        ("cos", "cos", 1),
        ("tan", "tan", 1),
        ("log", "log", 1),
        ("exp", "exp", 1),
        ("sqrt", "sqrt", 1),
        ("abs", "abs", 1),
        ("tanh", "tanh", 1),
        ("pow2", "**2", 1),
        ("pow3", "**3", 1),
        ("pow", "**", 2),
        ("inv", "1/", 1),
        ("protected_sqrt", "protected_sqrt", 1),
        ("protected_log", "protected_log", 1),
        ("protected_exp", "protected_exp", 1),
        ("protected_inv", "protected_inv", 1),
        ("protected_div", "protected_div", 2),
        ("asin", "asin", 1),
        ("acos", "acos", 1),
        ("protected_asin", "protected_asin", 1),
        ("protected_acos", "protected_acos", 1),
    ];
    for (i, (sem, alias, n)) in sr.iter().enumerate() {
        t.push(Symbol {
            kingdom: "Symbolic Regression".to_string(),
            symbol: (i + 1) as i64,
            symbol_name: sem.to_string(),
            alias: alias.to_string(),
            semantic_id: sem.to_string(),
            arity: Arity::uniform(f, *n),
        });
    }

    // THE LAW KINGDOM — the flat subset with the transcendentals removed. A law
    // is flat: arithmetic, integer powers, a root, a reciprocal — never a stack
    // of nested sin/cos/exp/log. The subexpression regression (7j) draws from
    // this kingdom so it structurally cannot rebuild a tower while re-expressing
    // one. `protected_sqrt` is kept (real laws take roots); it raises the
    // sensor's t_depth but is not a transcendental that nests without bound. Its
    // semantic ids are a SUBSET of the SR kingdom's, so a gene found here decodes
    // in the SR table too (`law_kingdom_is_a_flat_subset` pins that).
    let law: &[(&str, &str, u32)] = &[
        ("add", "+", 2),
        ("sub", "-", 2),
        ("mul", "*", 2),
        ("protected_div", "protected_div", 2),
        ("neg", "neg", 1),
        ("pow2", "**2", 1),
        ("pow3", "**3", 1),
        ("protected_sqrt", "protected_sqrt", 1),
        ("protected_inv", "protected_inv", 1),
    ];
    for (i, (sem, alias, n)) in law.iter().enumerate() {
        t.push(Symbol {
            kingdom: LAW.to_string(),
            symbol: (i + 1) as i64,
            symbol_name: sem.to_string(),
            alias: alias.to_string(),
            semantic_id: sem.to_string(),
            arity: Arity::uniform(f, *n),
        });
    }
    t
}

/// The kingdom name the subexpression regression (7j) draws from: the flat law
/// alphabet, the transcendentals removed. A subset of the "Symbolic Regression"
/// kingdom's semantic ids.
pub const LAW: &str = "Law";

/// The REGEX kingdom — the regex AST operators, typed over `Pattern`, `CharClass`, `Char` and `I` (repeat counts).
/// Its rows are the kingdom `regex` of `kingdoms/master/`, read by [`crate::regex_typed::RegexRows`]; see
/// `docs/DESIGN_geneframe.md` 8.2 and `docs/PLAN_regex_kingdom_build.md`.
pub const REGEX: &str = "Regex";

/// The semantic ids that RAISE transcendental depth: the same list as the `depth := in0.depth + 1` Flow rows of
/// `kingdoms/tsr/depth/tables.tsv` within the `wide()` alphabet, and phylu's `t_depth`. `pow` is absorbing here:
/// `SymbolTable::wide` carries no raw `Pow`, and `pow2`/`pow3` are whole powers, which `t_depth` does not count.
pub const DEPTH_RAISING: &[&str] = &[
    "sin", "cos", "tan", "log", "exp", "sqrt", "abs", "tanh", "asin", "acos", "protected_sqrt",
    "protected_log", "protected_exp", "protected_asin", "protected_acos",
];

// ---------------------------------------------------------------------------
// The gene frame's configuration tables and formula language
// (`docs/DESIGN_futhark_types_and_properties.md` sections 3.3 and 3.4)
// ---------------------------------------------------------------------------

mod schema;

pub use formula::{Condition, Formula, PropertyExpr, Reference, Relation, SizeBounds, Truth};
pub use schema::{
    always, catalogue, escape, parse_catalogue, property, spell, unescape, Argument, Constant, DataKind, DataType, Direction, Entry, EntryParameter, Flow, FlowName, Guard, KingdomTables, Owner, Place,
    Property, PropertyKind, Repair, Role, Site, Source, Strength, SymbolRow, Transform, Way, CATALOGUE_TSV,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sr_kingdom_has_all_math_ops() {
        let t = master_table();
        let sr = t.kingdom("Symbolic Regression");
        // every semantic id the converter/generator can emit must be present
        let sems: Vec<&str> = sr.iter().map(|s| s.semantic_id.as_str()).collect();
        for needed in [
            "add", "sub", "mul", "div", "neg", "sin", "cos", "tan", "log", "exp",
            "sqrt", "abs", "tanh", "pow2", "pow3", "pow", "inv", "protected_sqrt",
            "protected_log", "protected_exp", "protected_inv", "protected_div", "asin", "acos",
            "protected_asin", "protected_acos",
        ] {
            assert!(sems.contains(&needed), "SR kingdom missing semantic id {needed}");
        }
        assert_eq!(sr.len(), 26, "SR kingdom should have the 26 Math ops");
    }

    /// LOCKSTEP with `karva::master_pset()`: the SR kingdom and the flat
    /// master pset are two renderings of the same op set. Without this test,
    /// adding an op to karva.rs silently drifts geneframe (each table was a
    /// hand-copy). When geneframe becomes the owner, master_pset() should be
    /// DERIVED from the kingdom query and this test becomes tautological.
    #[test]
    fn sr_kingdom_locksteps_with_master_pset() {
        let mut from_kingdom: Vec<(String, usize)> = master_table()
            .kingdom("Symbolic Regression")
            .iter()
            .map(|s| (s.semantic_id.clone(), s.arity.total_in() as usize))
            .collect();
        let mut from_pset: Vec<(String, usize)> = crate::karva::master_pset()
            .into_iter()
            .map(|(s, a)| (s.to_string(), a))
            .collect();
        from_kingdom.sort();
        from_pset.sort();
        assert_eq!(
            from_kingdom, from_pset,
            "geneframe SR kingdom and karva::master_pset() have drifted apart"
        );
    }

    #[test]
    fn max_arity_is_two_for_sr() {
        // binary ops (add/mul/div/pow/...) give max total input arity 2.
        assert_eq!(master_table().max_arity("Symbolic Regression"), 2);
    }

    #[test]
    fn typed_arity_round_trips() {
        let a = Arity::uniform(Ty::F, 2);
        assert_eq!(a.total_in(), 2);
        assert_eq!(a.outputs.get(&Ty::F), Some(&1));
    }

    #[test]
    fn kingdom_query_isolates() {
        let t = master_table();
        // The master table now holds two kingdoms: the full SR op set and the
        // flat Law subset. `kingdoms()` sorts, so Law comes first. REGEX is its own
        // rows (the kingdom `regex` of `kingdoms/master/`), so this assertion is unchanged.
        assert_eq!(t.kingdoms(), vec!["Law".to_string(), "Symbolic Regression".to_string()]);
        assert!(t.kingdom("SQL").is_empty()); // not loaded yet
        assert!(t.kingdom(REGEX).is_empty()); // the regex kingdom is its own rows
    }

    #[test]
    fn law_kingdom_is_a_flat_subset_of_symbolic_regression() {
        let t = master_table();
        let sr: std::collections::HashSet<&str> =
            t.kingdom("Symbolic Regression").iter().map(|s| s.semantic_id.as_str()).collect();
        let law = t.kingdom(LAW);
        assert!(!law.is_empty(), "the Law kingdom is populated");
        for s in &law {
            // Every Law op is an SR op, so a gene found in Law decodes in SR.
            assert!(sr.contains(s.semantic_id.as_str()), "Law op {} is not in SR", s.semantic_id);
            // No transcendental that nests without bound — the whole point. A
            // root (sqrt) is allowed: it is in DEPTH_RAISING but a law takes
            // roots, and it cannot build a tower on its own.
            let nesting_transcendental = DEPTH_RAISING.contains(&s.semantic_id.as_str())
                && !matches!(s.semantic_id.as_str(), "sqrt" | "protected_sqrt");
            assert!(!nesting_transcendental, "the Law kingdom must not contain {}", s.semantic_id);
        }
        // Max arity is still 2 (add/mul/div/protected_div), so a gene decodes the
        // same way as an SR gene.
        assert_eq!(t.max_arity(LAW), 2);
    }

    // ---- Ty::code(), the one spelling of a type on the device ----

    #[test]
    fn ty_code_is_the_declaration_index_and_every_variant_is_listed_once() {
        // Pins `code() == ty as u32` for every base variant, so phylu's moved
        // cast sites see the same numbers, and that `ALL` is complete and in
        // order: a variant added without a slot here fails this test.
        for (i, t) in Ty::ALL.iter().enumerate() {
            assert_eq!(t.code(), i as u32, "{t:?}");
            assert_eq!(Ty::from_code(i as u32), Some(*t));
        }
        assert_eq!(Ty::ALL.last(), Some(&Ty::Pattern), "ALL ends at the last base variant");
        assert_eq!(Ty::ALL.len(), 33);
        // The table-defined family sits past every base code.
        assert_eq!(Ty::Wgsl(0).code(), 256);
        assert_eq!(Ty::Wgsl(70).code(), 326);
        assert_eq!(Ty::from_code(300), Some(Ty::Wgsl(44)));
        assert!(Ty::Wgsl(0) > Ty::Pattern, "ordered after every base variant");
        let mut codes: Vec<u32> = Ty::ALL.iter().map(|t| t.code()).collect();
        codes.dedup();
        assert_eq!(codes.len(), Ty::ALL.len(), "codes are unique");
        assert!(Ty::Pattern.code() < Ty::TABLE_CODE_BASE, "base codes stay below the table-defined range");
        assert_eq!(Ty::from_code(Ty::ALL.len() as u32), None);
        // The table-defined ranges round-trip in full and do not overlap.
        // The retired TSR range (4352..69888) is no type, so the Futhark families' codes did not move.
        assert_eq!(Ty::FUTHARK_1D_CODE_BASE, 69888);
        for i in 0..Ty::MAX_WGSL_DUALS as u16 {
            assert_eq!(Ty::from_code(Ty::Wgsl(i).code()), Some(Ty::Wgsl(i)));
        }
        assert_eq!(Ty::from_code(4351), Some(Ty::Wgsl(4095)));
        assert_eq!((Ty::from_code(4352), Ty::from_code(69887)), (None, None));
        // The 1-D Futhark family sits after the WGSL family, and the two do
        // not overlap.
        for i in 0..=u16::MAX {
            assert_eq!(Ty::from_code(Ty::Futhark1d(i).code()), Some(Ty::Futhark1d(i)));
        }
        assert_eq!(Ty::from_code(Ty::FUTHARK_1D_CODE_BASE), Some(Ty::Futhark1d(0)));
        assert!(Ty::Futhark1d(0) > Ty::Wgsl(u16::MAX), "ordered after the WGSL family");
        // The Futhark family sits after the 1-D one, so the two
        // kingdoms' duals cannot be confused for one another.
        assert_eq!(Ty::FUTHARK_CODE_BASE, Ty::FUTHARK_1D_CODE_BASE + 65536);
        for i in 0..=u16::MAX {
            assert_eq!(Ty::from_code(Ty::Futhark(i).code()), Some(Ty::Futhark(i)));
        }
        assert_eq!(Ty::from_code(Ty::FUTHARK_CODE_BASE), Some(Ty::Futhark(0)));
        assert_eq!(Ty::from_code(Ty::FUTHARK_CODE_BASE + 65536), None);
        assert!(
            Ty::Futhark(0) > Ty::Futhark1d(u16::MAX),
            "the Futhark family is ordered after the 1-D Futhark family"
        );
        assert!(
            Ty::Futhark1d(u16::MAX) < Ty::Futhark(0),
            "the two Futhark families do not overlap"
        );
        assert!(Ty::Pattern.code() < Ty::TABLE_CODE_BASE, "base codes stay below the table-defined range");
        assert_eq!(Ty::F.code(), 2);
    }

    #[test]
    fn a_kingdom_with_too_many_duals_fails_loudly_in_every_build() {
        assert!(Ty::check_dual_count("k", Ty::MAX_WGSL_DUALS, Ty::MAX_WGSL_DUALS).is_ok());
        let e = Ty::check_dual_count("k", Ty::MAX_WGSL_DUALS + 1, Ty::MAX_WGSL_DUALS).unwrap_err();
        assert_eq!(e, "our software has a bug: kingdom k has 4097 duals, more than its 4096 codes");
        assert_eq!(Ty::FUTHARK_1D_CODE_BASE - Ty::TABLE_CODE_BASE, Ty::MAX_WGSL_DUALS as u32 + Ty::RETIRED_TSR_CODES);
    }

    #[test]
    #[should_panic(expected = "our software has a bug: Wgsl(4096)")]
    fn code_of_an_out_of_range_wgsl_dual_panics_in_every_build() {
        let _ = Ty::Wgsl(4096).code();
    }

    // ---- the data model of design section 3 (plan task G1) ----

    use crate::kingdom_contract::{relations_from_rows, report, KingdomRelations};

    /// Build the tables of `kingdom` from its ORDERED signatures (one per row of today's table, in order), derive the
    /// arity of every symbol from the Argument rows, and require today's table row for row.
    fn derives_todays_arity(today: &SymbolTable, kingdom: &str, ordered: Vec<(Vec<Ty>, Ty)>) {
        let rows = today.kingdom(kingdom);
        assert!(!rows.is_empty(), "{kingdom}: no rows");
        assert_eq!(rows.len(), ordered.len(), "{kingdom}: one ordered signature a row");
        let sigs: Vec<(SymbolRow, Vec<Ty>, Ty)> = rows.iter().zip(ordered).map(|(s, (ins, out))| (SymbolRow::of(s), ins, out)).collect();
        let tables = KingdomTables::from_signatures(kingdom, sigs.clone());
        tables.check().unwrap_or_else(|d| panic!("{kingdom}: {} defects, first {:?}", d.len(), d.first()));
        assert!(!tables.has_properties(), "{kingdom}: a kingdom with no property rows");
        let derived = tables.symbol_table();
        let derived: Vec<&Symbol> = derived.kingdom(kingdom);
        assert_eq!(derived, rows, "{kingdom}: the derived arity differs from today's");
        for (row, ins, out) in sigs.iter().take(64) {
            assert_eq!(&tables.inputs(row.symbol), ins, "{kingdom} {}: argument order", row.symbol_name);
            assert_eq!(tables.output(row.symbol), Some(*out));
            assert_eq!(&tables.arity(row.symbol), &today.kingdom(kingdom).iter().find(|s| s.symbol == row.symbol).expect("row").arity);
        }
        let mut seen = std::collections::BTreeSet::new();
        assert!(sigs.iter().all(|(r, _, _)| seen.insert(r.symbol)), "{kingdom}: symbol ids unique");
        let ordered_back: Vec<(Vec<Ty>, Ty)> = tables.signatures().into_iter().map(|(_, i, o)| (i, o)).collect();
        assert_eq!(ordered_back, sigs.into_iter().map(|(_, i, o)| (i, o)).collect::<Vec<_>>(), "{kingdom}: signatures in symbol order");
    }

    #[test]
    fn the_derived_arity_is_todays_for_every_symbol_of_every_symbol_table_kingdom() {
        let master = master_table();
        for k in ["Symbolic Regression", LAW] {
            let ordered = master.kingdom(k).iter().map(|s| (vec![Ty::F; s.arity.total_in() as usize], Ty::F)).collect();
            derives_todays_arity(&master, k, ordered);
        }
        let rx = crate::regex_typed::RegexRows::load(true).expect("the regex rows load");
        derives_todays_arity(&rx.tables.symbol_table(), REGEX, rx.tables.signatures().into_iter().map(|(_, i, o)| (i, o)).collect());
        #[cfg(feature = "wgsl")]
        for k in [crate::wgsl::loader::WgslKingdom::load(), crate::wgsl::loader::WgslKingdom::load_safe()] {
            derives_todays_arity(&k.symbol_table(), crate::wgsl::loader::WGSL, k.rows.iter().map(|r| (r.inputs.clone(), r.output)).collect());
        }
    }

    /// Carry a kingdom's typed-GEP relations through the tables: every symbol's ordered inputs become Argument rows,
    /// and the relations rebuilt from the tables' signatures are today's, so the contract, its bounds and every frame
    /// are sized exactly as now (design 3.2).
    fn the_contract_is_unchanged_through_the_tables(r: &KingdomRelations) {
        let mut ins: Vec<Vec<(u32, Ty)>> = vec![Vec::new(); r.names.len()];
        for &(s, pos, ty) in &r.argument_signature {
            ins[s as usize].push((pos, ty));
        }
        let mut out: Vec<Option<Ty>> = vec![None; r.names.len()];
        for &(s, ty) in &r.symbol_outputs {
            out[s as usize] = Some(ty);
        }
        let sigs: Vec<(SymbolRow, Vec<Ty>, Ty)> = r
            .names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let mut a = ins[i].clone();
                a.sort_unstable_by_key(|(p, _)| *p);
                let role = if r.constants.contains(&(i as u32)) { Role::Constant } else if r.variables.contains(&(i as u32)) { Role::Variable } else { Role::Function };
                let row = SymbolRow { role, ..SymbolRow::named(i as i64, name) };
                (row, a.into_iter().map(|(_, t)| t).collect(), out[i].expect("one output a symbol"))
            })
            .collect();
        let tables = KingdomTables::from_signatures(&r.kingdom, sigs);
        tables.check().unwrap_or_else(|d| panic!("{}: {} defects, first {:?}", r.kingdom, d.len(), d.first()));
        let rows: Vec<(String, Vec<Ty>, Ty)> = tables.signatures().into_iter().map(|(s, i, o)| (r.names[s as usize].clone(), i, o)).collect();
        let constants: Vec<&str> = r.constants.iter().map(|&c| r.names[c as usize].as_str()).collect();
        let mut back = relations_from_rows(&r.kingdom, r.roots.clone(), &rows, &constants);
        back.tail_free_roots = r.tail_free_roots;
        let mut want = r.argument_signature.clone();
        want.sort_unstable();
        let mut got = back.argument_signature.clone();
        got.sort_unstable();
        assert_eq!(got, want, "{}: argument signature", r.kingdom);
        assert_eq!(back.symbol_outputs, r.symbol_outputs, "{}: outputs", r.kingdom);
        assert_eq!(back.constants, r.constants, "{}: constants", r.kingdom);
        // The checked contracts agree on what sizes the frames: the types, `n_max`, `a_T` and `n_U`; and, for a
        // kingdom small enough to print, on the whole bound table (k-means's 14,359 types take two minutes to print).
        let (c_back, c_today) = (back.check().expect("the rebuilt relations check"), r.check().expect("today's relations check"));
        assert_eq!(c_back.types(), c_today.types(), "{}: types", r.kingdom);
        assert_eq!(c_back.n_max(), c_today.n_max(), "{}: n_max", r.kingdom);
        for &ty in c_today.types() {
            assert_eq!((c_back.a_t(ty), c_back.n_u(ty)), (c_today.a_t(ty), c_today.n_u(ty)), "{}: a_T and n_U of {ty:?}", r.kingdom);
        }
        if c_today.types().len() <= 1_000 {
            for n in [8usize, 32] {
                assert_eq!(report(&back, n), report(r, n), "{}: the bound table at N = {n}", r.kingdom);
            }
        }
        // The many-hot arity of each symbol is the count of its ordered inputs.
        let derived = tables.symbol_table();
        for (i, s) in derived.kingdom(&r.kingdom).iter().enumerate() {
            let mut a = Arity::default();
            for (_, ty) in &ins[i] {
                *a.inputs.entry(*ty).or_insert(0) += 1;
            }
            a.outputs.insert(out[i].expect("output"), 1);
            assert_eq!(s.arity, a, "{}: {}", r.kingdom, s.symbol_name);
        }
    }

    #[test]
    fn every_kingdoms_contract_is_unchanged_through_the_tables() {
        use crate::kingdom_contract::{regex_relations, sr_relations};
        the_contract_is_unchanged_through_the_tables(&sr_relations());
        the_contract_is_unchanged_through_the_tables(&regex_relations());
        #[cfg(feature = "wgsl")]
        {
            the_contract_is_unchanged_through_the_tables(&crate::kingdom_contract::wgsl_relations());
            the_contract_is_unchanged_through_the_tables(&crate::kingdom_contract::safewgsl_relations());
            let t = crate::wgsl::true_kingdom::TrueKingdom::load(&crate::wgsl::true_kingdom::default_dir()).expect("true WGSL table");
            the_contract_is_unchanged_through_the_tables(&t.relations());
        }
        // The deprecated 1-D Futhark kingdom.
        the_contract_is_unchanged_through_the_tables(&crate::futhark_table::FutharkKingdom::load_1d().expect("1-D kingdom").relations());
    }
}
