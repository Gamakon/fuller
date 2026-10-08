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

use crate::gpu_eval::{GpuNode, Op};

/// The value types a symbol's slots can carry — the `in_*`/`out_*` columns of
/// `docs/design/DataModel.md`, as enum variants so `Arity`'s many-hot maps are
/// keyed by them. THE FULL SET IS DECLARED NOW so the row structure is stable
/// over time: a new kingdom REUSES these columns, it does not add a variant and
/// migrate every row. Most types are zero on any given row. A new type is a
/// genuinely new value class (not a new kingdom), and only then is a variant
/// added — and because every match on `Ty` is non-exhaustive (a `_` arm), that
/// addition stays backward-compatible.
///
/// Ordering: the base + depth-ladder types come FIRST and never move, so
/// [`Ty::depth`], [`Ty::LADDER`] and [`Ty::at_depth`] — the SR depth typing —
/// are unaffected by anything below `T2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ty {
    // ---- Base types (Symbolic Regression / SQL / REGEX kingdoms) ----
    S, // String
    I, // Integer
    F, // Float — also transcendental DEPTH 0
    B, // Boolean
    A, // Array
    L, // List
    /// TRANSCENDENTAL DEPTH 1: the output of one transcendental applied to
    /// plain floats. See [`typed_depth_table`].
    T1,
    /// TRANSCENDENTAL DEPTH 2: a transcendental applied to something already
    /// `T1`. **There is no `T3`, and that absence is the depth rule** — not a
    /// check and not a penalty, an arity signature that does not exist.
    T2,

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
    // Appended last so the base + depth-ladder types keep their positions and
    // `Ty::depth`/`LADDER`/`at_depth` are unaffected (the ordering rule above).
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

    /// Every variant, in declaration order. [`Ty::code`] is the position here.
    pub const ALL: [Ty; 35] = [
        Ty::S, Ty::I, Ty::F, Ty::B, Ty::A, Ty::L, Ty::T1, Ty::T2,
        Ty::Person, Ty::Org, Ty::Gpe, Ty::Loc, Ty::Norp, Ty::Fac, Ty::Event, Ty::Product,
        Ty::Date, Ty::Time, Ty::Money, Ty::Quantity, Ty::Cardinal, Ty::Percent, Ty::Np, Ty::Ap,
        Ty::Clause, Ty::Verb,
        Ty::Entity, Ty::Relation, Ty::Metric, Ty::Procedure, Ty::Narrative,
        Ty::Addr,
        Ty::Char, Ty::CharClass, Ty::Pattern,
    ];

    /// The type's code on the device: THE ONE SPELLING of a `Ty` as a `u32`
    /// (the node's `ty_code` word, the `out_ty`/`in_ty` tables, the fallback
    /// table's index). A base variant's code is its declaration index from 0
    /// (what `ty as u32` was before the payload variant made the cast
    /// uncompilable); a WGSL dual's code is [`Ty::TABLE_CODE_BASE`] + its row.
    pub fn code(self) -> u32 {
        match self {
            Ty::Wgsl(i) => Ty::TABLE_CODE_BASE + u32::from(i),
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

    /// The inverse of [`Ty::code`]: a base variant by index, a WGSL dual by
    /// offset from [`Ty::TABLE_CODE_BASE`].
    pub fn from_code(code: u32) -> Option<Ty> {
        if code >= Ty::TABLE_CODE_BASE {
            return u16::try_from(code - Ty::TABLE_CODE_BASE).ok().map(Ty::Wgsl);
        }
        Ty::ALL.get(code as usize).copied()
    }
}

/// THE PER-TYPE FALLBACK LEAF (Design C's projection): the zero-arity node a
/// decoder emits when a codon's type does not match the demanded type, so
/// every gene expresses. One per type; the device's table must equal this
/// (host↔device parity). The regex kingdom's four demanded types have their
/// own leaves; every other base type falls back to `Num 0`.
pub fn fallback_leaf(ty: Ty) -> GpuNode {
    let leaf = |op: Op, konst: f32| GpuNode { op: op as u32, arg0: 0, arg1: 0, konst };
    match ty {
        Ty::Pattern => leaf(Op::RegexEmpty, 0.0),
        Ty::CharClass => leaf(Op::RegexCcDigit, 0.0),
        // Char falls back to a printable byte (a `Num` the compiler reads as a
        // literal byte); Integer to the count 1 (a `{1}` is a no-op repeat).
        Ty::Char => leaf(Op::Num, f32::from(b'0')),
        Ty::I => leaf(Op::Num, 1.0),
        _ => leaf(Op::Num, 0.0),
    }
}

/// The fallback table indexed by [`Ty::code`]: `fallback_table()[ty.code()]`
/// is [`fallback_leaf`]`(ty)`. What a decoder uploads; sized by the base
/// variants today, extended by the table-defined duals when they land.
pub fn fallback_table() -> Vec<GpuNode> {
    Ty::ALL.iter().map(|t| fallback_leaf(*t)).collect()
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

/// The kingdom name [`typed_depth_table`] files its rows under.
pub const TYPED_SR: &str = "Symbolic Regression (typed depth)";

/// The REGEX kingdom — the regex AST operators, typed over `Char`, `CharClass`,
/// `Pattern` (and `I` for repeat counts, unused in the Phase-1 set). See
/// [`regex_table`] and `docs/PLAN_regex_kingdom_build.md`.
pub const REGEX: &str = "Regex";

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

/// THE REGEX KINGDOM — the regex AST operators as typed symbols. A gene decodes
/// to a regex AST (every operator arity ≤ 2, the decoder's limit), which the GPU
/// compile kernel lowers to the Thompson VM. The types make an ill-typed regex
/// unconstructable: `lit` takes a `Char` and yields a `Pattern`, `concat`/`alt`
/// combine `Pattern`s, `star` loops a `Pattern`, and the class/dot/anchor
/// terminals are `Pattern`s directly.
///
/// Phase-1 set: NO counted repetition (`{n}`/`{n,m}`), because it would expand
/// the instruction program multiplicatively; so no `Integer` operand appears and
/// `star` (an NFA loop, not a copy) is the only repetition. `semantic_id`s match
/// the `gpu_eval::Op` regex opcodes; `alias` is the regex surface spelling.
///
/// A `Char` terminal is the gene's `?` RNC byte (a printable-ASCII constant), so
/// it is not a row here — it enters via the RNC mechanism, like SR's constants.
pub fn regex_table() -> SymbolTable {
    let mut t = SymbolTable::new();
    let pat = Ty::Pattern;
    // (semantic_id, alias, inputs, output). Inputs are the typed child slots in
    // order; an empty input list is a terminal.
    //
    // THE GEP VALIDITY GUARANTEE: every head symbol takes only `Pattern` children
    // (or none), and the sole terminal is the `?` RNC byte, which decodes to a
    // one-byte `Pattern` (a literal-byte leaf) by coercion. So EVERY decoded gene
    // expresses — there is no symbol that demands a child the tail cannot supply.
    // There is deliberately NO `lit` operator and NO `Char` type: a byte literal
    // in a Pattern slot IS a one-byte pattern, so a separate `lit(Char)->Pattern`
    // would be the one symbol whose child slot the all-`?` tail cannot always
    // fill, breaking GEP's "all offspring are valid programs" guarantee.
    let cc = Ty::CharClass;
    let ch = Ty::Char;
    let int = Ty::I;
    let rows: &[(&str, &str, &[Ty], Ty)] = &[
        // ---- Pattern terminals (arity 0) ----
        ("regex_dot", ".", &[], pat),
        ("regex_anchor_start", "^", &[], pat),
        ("regex_anchor_end", "$", &[], pat),
        ("regex_empty", "", &[], pat),
        // ---- CharClass terminals (arity 0) — the built-in classes ----
        ("regex_cc_digit", r"\d", &[], cc),
        ("regex_cc_word", r"\w", &[], cc),
        ("regex_cc_space", r"\s", &[], cc),
        // ---- Pattern ops ----
        ("regex_star", "*", &[pat], pat),
        ("regex_plus", "+", &[pat], pat),
        ("regex_opt", "?", &[pat], pat),
        ("regex_concat", "concat", &[pat, pat], pat),
        ("regex_alt", "|", &[pat, pat], pat),
        // counted repetition — the Integer-typed slot that makes the kingdom
        // genuinely multi-typed (`docs/PLAN_multityped_gep.md`).
        ("regex_rep_n", "{n}", &[pat, int], pat),
        ("regex_rep_upto", "{0,k}", &[pat, int], pat),
        // ---- coercions / class algebra ----
        // a CharClass or a Char used as a Pattern (match one byte of it).
        ("regex_class_of", "class_of", &[cc], pat),
        ("regex_lit", "lit", &[ch], pat),
        ("regex_cc_range", "a-z", &[ch, ch], cc),
        ("regex_cc_union", "cc_union", &[cc, cc], cc),
        ("regex_cc_negate", "^cc", &[cc], cc),
    ];
    for (i, (sem, alias, ins, out)) in rows.iter().enumerate() {
        let mut arity = Arity::default();
        for &inp in *ins {
            *arity.inputs.entry(inp).or_insert(0) += 1;
        }
        arity.outputs.insert(*out, 1);
        t.push(Symbol {
            kingdom: REGEX.to_string(),
            symbol: (i + 1) as i64,
            symbol_name: sem.to_string(),
            alias: alias.to_string(),
            semantic_id: sem.to_string(),
            arity,
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
        // flat Law subset. `kingdoms()` sorts, so Law comes first. REGEX is a
        // SEPARATE table (`regex_table`), so this assertion is unchanged.
        assert_eq!(t.kingdoms(), vec!["Law".to_string(), "Symbolic Regression".to_string()]);
        assert!(t.kingdom("SQL").is_empty()); // not loaded yet
        assert!(t.kingdom(REGEX).is_empty()); // the regex kingdom is its own table
    }

    /// The REGEX kingdom is well-formed: every row is under `REGEX`, every arity
    /// is ≤ 2 (the decoder's limit), every output is the hub type `Pattern` or a
    /// `CharClass`, every input is one of the four value types, and — now that it
    /// is MULTI-TYPED — counted repetition carries an Integer slot. The total
    /// typed decoder (Design C) keeps every gene valid, so mixed types are safe.
    #[test]
    fn regex_kingdom_is_well_formed_and_arity_two() {
        let t = regex_table();
        let rows = t.kingdom(REGEX);
        assert_eq!(t.max_arity(REGEX), 2, "the decoder caps arity at 2");
        let value_types = [Ty::Pattern, Ty::CharClass, Ty::Char, Ty::I];
        for s in &rows {
            assert!(s.arity.total_in() <= 2, "{} has arity {}", s.semantic_id, s.arity.total_in());
            // Output is Pattern or CharClass (the two producible types).
            let out = *s.arity.outputs.keys().next().expect("an output type");
            assert!(out == Ty::Pattern || out == Ty::CharClass, "{} outputs {out:?}", s.semantic_id);
            // Every input slot is one of the four value types.
            for ty in s.arity.inputs.keys() {
                assert!(value_types.contains(ty), "{} takes an unexpected child {ty:?}", s.semantic_id);
            }
        }
        // The multi-typed POSIX ERE set is present, including counted repetition
        // (the Integer-typed slot that makes the kingdom genuinely multi-typed).
        let sems: std::collections::HashSet<&str> = rows.iter().map(|s| s.semantic_id.as_str()).collect();
        for needed in ["regex_concat", "regex_alt", "regex_star", "regex_plus", "regex_opt",
            "regex_rep_n", "regex_rep_upto", "regex_class_of", "regex_lit", "regex_cc_range",
            "regex_cc_union", "regex_cc_negate", "regex_dot", "regex_empty",
            "regex_cc_digit", "regex_cc_word", "regex_cc_space", "regex_anchor_start", "regex_anchor_end"] {
            assert!(sems.contains(needed), "regex kingdom missing {needed}");
        }
        // At least one operator demands an Integer child — the proof it is
        // multi-typed, not the single-Pattern Phase-1 set.
        assert!(rows.iter().any(|s| s.arity.inputs.get(&Ty::I) == Some(&1)), "no Integer-typed slot — not multi-typed");
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
        assert_eq!(Ty::ALL.len(), 35);
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
        // The depth ladder is unmoved by the numbering.
        assert_eq!((Ty::F.code(), Ty::T1.code(), Ty::T2.code()), (2, 6, 7));
    }

    #[test]
    fn fallback_table_is_total_over_the_base_codes_and_matches_the_decoders_leaves() {
        let table = fallback_table();
        assert_eq!(table.len(), Ty::ALL.len());
        for t in Ty::ALL {
            let leaf = table[t.code() as usize];
            let want = fallback_leaf(t);
            assert_eq!((leaf.op, leaf.arg0, leaf.arg1, leaf.konst.to_bits()), (want.op, want.arg0, want.arg1, want.konst.to_bits()), "{t:?}");
            assert_eq!(Op::from_u32(leaf.op).unwrap().arity(), 0, "{t:?}: a fallback is a leaf");
        }
        // The regex kingdom's demanded types, as the host and device decoders
        // have always emitted them (phylu engine.rs fallback_node, decode.wgsl).
        assert_eq!(table[Ty::Pattern.code() as usize].op, Op::RegexEmpty as u32);
        assert_eq!(table[Ty::CharClass.code() as usize].op, Op::RegexCcDigit as u32);
        assert_eq!((table[Ty::Char.code() as usize].op, table[Ty::Char.code() as usize].konst), (Op::Num as u32, 48.0));
        assert_eq!((table[Ty::I.code() as usize].op, table[Ty::I.code() as usize].konst), (Op::Num as u32, 1.0));
        assert_eq!((table[Ty::F.code() as usize].op, table[Ty::F.code() as usize].konst), (Op::Num as u32, 0.0));
    }
}
