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

use std::collections::BTreeMap;

/// The base value types a symbol's slots can carry (the `in_*`/`out_*` columns).
/// Extensible: a new kingdom adds variants without changing existing rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ty {
    S,
    I,
    F,
    B,
    A,
    L,
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
    t
}

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
            "protected_log", "protected_exp", "protected_inv", "protected_div",
        ] {
            assert!(sems.contains(&needed), "SR kingdom missing semantic id {needed}");
        }
        assert_eq!(sr.len(), 22, "SR kingdom should have the 22 Math ops");
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
        assert_eq!(t.kingdoms(), vec!["Symbolic Regression".to_string()]);
        assert!(t.kingdom("SQL").is_empty()); // not loaded yet
    }
}
