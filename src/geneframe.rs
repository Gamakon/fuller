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
    /// TRANSCENDENTAL DEPTH 1: the output of one transcendental applied to
    /// plain floats. See [`typed_depth_table`].
    T1,
    /// TRANSCENDENTAL DEPTH 2: a transcendental applied to something already
    /// `T1`. **There is no `T3`, and that absence is the depth rule** — not a
    /// check and not a penalty, an arity signature that does not exist.
    T2,
}

impl Ty {
    /// The transcendental depth this type stands for, `None` for the types that
    /// are not part of the depth ladder.
    pub fn depth(self) -> Option<u32> {
        match self {
            Ty::F => Some(0),
            Ty::T1 => Some(1),
            Ty::T2 => Some(2),
            _ => None,
        }
    }

    /// The depth ladder in order: `F` (0), `T1` (1), `T2` (2). The ladder's
    /// LENGTH is the ceiling — a `T3` would be a fourth entry here, and there
    /// is none.
    pub const LADDER: [Ty; 3] = [Ty::F, Ty::T1, Ty::T2];

    /// The rung at `depth`, or `None` when the ladder has no such rung — which
    /// is what makes a third level unrepresentable.
    pub fn at_depth(depth: u32) -> Option<Ty> {
        Ty::LADDER.get(depth as usize).copied()
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
    t
}

/// The kingdom name [`typed_depth_table`] files its rows under.
pub const TYPED_SR: &str = "Symbolic Regression (typed depth)";

/// The semantic ids that RAISE transcendental depth. Lockstep with
/// `evolve::engine::t_depth`'s own list — the table and the engine's measured
/// depth must be the same predicate or the sampler and the checker disagree.
/// `pow` is absorbing here: `SymbolTable::wide` carries no raw `Pow`, and
/// `pow2`/`pow3` are whole powers, which `t_depth` does not count.
pub const DEPTH_RAISING: &[&str] = &[
    "sin", "cos", "tan", "log", "exp", "sqrt", "abs", "tanh", "asin", "acos", "protected_sqrt",
    "protected_log", "protected_exp", "protected_asin", "protected_acos",
];

/// THE TYPED-DEPTH KINGDOM — the same 26 `Math` ops, but with the typed,
/// many-hot arity signatures that make a tower of transcendentals
/// **unrepresentable** rather than merely penalised.
///
/// Three rules, and between them they are exactly `t_depth <= 2`:
///
/// * **Terminals** are `out {F}` — a variable or a constant is depth 0.
/// * **Arithmetic is ABSORBING**: it accepts any mix and yields the highest
///   depth it saw, so depth propagates through `+ - * /`. This is what makes
///   `exp(-(theta/sqrt(2))**2)` correctly depth 2 rather than depth 1, and the
///   three depth-2 SRBench laws reach depth 2 through arithmetic, not by
///   direct nesting.
/// * **Transcendentals are DEPTH-RAISING** and have **no row accepting `T2`**.
///   That missing row is the ceiling.
///
/// The measurement behind it: over all 133 SRBench true models there are ZERO
/// directly nested transcendental pairs; 78 laws sit at depth 0, 47 at 1, 3 at
/// 2, and none at 3 or beyond.
pub fn typed_depth_table() -> SymbolTable {
    let mut t = SymbolTable::new();
    let mut symbol = 1i64;
    for s in master_table().kingdom("Symbolic Regression") {
        let n = s.arity.total_in();
        let raising = DEPTH_RAISING.contains(&s.semantic_id.as_str());
        // Every combination of input depths this op accepts, as a signature.
        // A nullary terminal has one row, `out {F}`.
        let rows: Vec<(Vec<u32>, u32)> = match n {
            0 => vec![(Vec::new(), 0)],
            1 if raising => {
                // DEPTH-RAISING, and it stops at two: no row takes T2.
                (0..=1).map(|d| (vec![d], d + 1)).collect()
            }
            1 => (0..=2).map(|d| (vec![d], d)).collect(),
            _ => {
                // ABSORBING: any mix in, the highest depth seen out.
                let mut v = Vec::new();
                for a in 0..=2 {
                    for b in a..=2 {
                        v.push((vec![a, b], a.max(b)));
                    }
                }
                v
            }
        };
        for (ins, out) in rows {
            let mut arity = Arity::default();
            for d in ins {
                let ty = Ty::at_depth(d).expect("a ladder rung");
                *arity.inputs.entry(ty).or_insert(0) += 1;
            }
            arity.outputs.insert(Ty::at_depth(out).expect("a ladder rung"), 1);
            t.push(Symbol {
                kingdom: TYPED_SR.to_string(),
                symbol,
                symbol_name: s.semantic_id.clone(),
                alias: s.alias.clone(),
                semantic_id: s.semantic_id.clone(),
                arity,
            });
            symbol += 1;
        }
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
        assert_eq!(t.kingdoms(), vec!["Symbolic Regression".to_string()]);
        assert!(t.kingdom("SQL").is_empty()); // not loaded yet
    }

    /// The typed kingdom is a SEPARATE query, so the untyped SR rows are
    /// literally untouched — which is what the spec claims and what `Ty`'s own
    /// docstring promises about extension.
    #[test]
    fn the_typed_kingdom_leaves_the_untyped_rows_alone() {
        let plain = master_table();
        let typed = typed_depth_table();
        assert_eq!(plain.kingdom(TYPED_SR).len(), 0, "typed rows must not be in the plain table");
        assert_eq!(typed.kingdom("Symbolic Regression").len(), 0);
        // Same op set, several signatures apiece.
        let mut plain_sems: Vec<&str> =
            plain.kingdom("Symbolic Regression").iter().map(|s| s.semantic_id.as_str()).collect();
        let mut typed_sems: Vec<&str> =
            typed.kingdom(TYPED_SR).iter().map(|s| s.semantic_id.as_str()).collect();
        plain_sems.sort_unstable();
        plain_sems.dedup();
        typed_sems.sort_unstable();
        typed_sems.dedup();
        assert_eq!(plain_sems, typed_sems, "the typed kingdom must cover exactly the same ops");
    }

    /// The ceiling, stated as the absence the spec says it is: NO transcendental
    /// row accepts a `T2` input, so a third level has no signature to be built
    /// from.
    #[test]
    fn no_transcendental_row_accepts_t2() {
        for s in typed_depth_table().kingdom(TYPED_SR) {
            if DEPTH_RAISING.contains(&s.semantic_id.as_str()) {
                assert_eq!(
                    s.arity.inputs.get(&Ty::T2),
                    None,
                    "{} has a row accepting T2 — the ceiling is gone",
                    s.semantic_id
                );
                assert!(s.arity.outputs.contains_key(&Ty::T1) || s.arity.outputs.contains_key(&Ty::T2));
            }
        }
        // And nothing anywhere outputs a rung above T2, because there is none.
        assert_eq!(Ty::at_depth(3), None, "a T3 rung would be the ceiling removed");
    }

    /// Arithmetic ABSORBS: it yields the highest depth among its inputs. This is
    /// what carries depth through `+ - * /` so the three depth-2 SRBench laws,
    /// which reach depth 2 THROUGH arithmetic, are typed as depth 2.
    #[test]
    fn arithmetic_absorbs_the_highest_depth_it_saw() {
        let t = typed_depth_table();
        let add: Vec<&Symbol> =
            t.kingdom(TYPED_SR).into_iter().filter(|s| s.semantic_id == "add").collect();
        assert_eq!(add.len(), 6, "add wants the six binary depth mixes");
        for s in add {
            let highest = Ty::LADDER
                .iter()
                .rev()
                .find(|ty| s.arity.inputs.contains_key(ty))
                .copied()
                .expect("a binary row has inputs");
            assert_eq!(
                s.arity.outputs.get(&highest),
                Some(&1),
                "add {:?} must yield its highest input depth",
                s.arity.inputs
            );
        }
        // Unary arithmetic is depth-PRESERVING, at all three rungs.
        let neg: Vec<&Symbol> =
            t.kingdom(TYPED_SR).into_iter().filter(|s| s.semantic_id == "neg").collect();
        assert_eq!(neg.len(), 3);
        for s in neg {
            let (i, _) = s.arity.inputs.iter().next().expect("one input");
            assert_eq!(s.arity.outputs.get(i), Some(&1), "neg must preserve depth");
        }
    }

    /// A transcendental RAISES by one and has exactly two rows — F and T1 in,
    /// T1 and T2 out. Two rows, not three, is the whole design.
    #[test]
    fn a_transcendental_raises_by_one_and_has_exactly_two_rows() {
        let t = typed_depth_table();
        for sem in DEPTH_RAISING {
            let rows: Vec<&Symbol> =
                t.kingdom(TYPED_SR).into_iter().filter(|s| &s.semantic_id.as_str() == sem).collect();
            assert_eq!(rows.len(), 2, "{sem} wants exactly the F and T1 rows");
            for s in rows {
                let (i, _) = s.arity.inputs.iter().next().expect("one input");
                let (o, _) = s.arity.outputs.iter().next().expect("one output");
                assert_eq!(
                    o.depth(),
                    i.depth().map(|d| d + 1),
                    "{sem} must raise depth by one"
                );
            }
        }
    }
}
