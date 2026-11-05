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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    /// A dual `shape.level` (or the root dual `ROOT`) of the TSR kingdom: the
    /// index of its row in `kingdoms/tsr/<variant>/types.tsv` (ref `T001` →
    /// 0). Its device code is [`Ty::TSR_CODE_BASE`] + the index. Appended
    /// after [`Ty::Wgsl`].
    Tsr(u16),
    /// A dual of the deprecated 1-D Futhark kingdom: the index of its row in
    /// `kingdoms/futhark_1d/types.tsv` (ref `T001` → 0). Its device code is
    /// [`Ty::FUTHARK_1D_CODE_BASE`] + the index. Appended last.
    Futhark1d(u16),
    /// A dual of the Futhark kingdom: the index of its row in
    /// `kingdoms/futhark/types.tsv` (ref `T001` → 0). Its device code
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
    /// (the node's `ty_code` word, the `out_ty`/`in_ty` tables). A base variant's code is its declaration index from 0
    /// (what `ty as u32` was before the payload variant made the cast
    /// uncompilable); a WGSL dual's code is [`Ty::TABLE_CODE_BASE`] + its row,
    /// a TSR dual's [`Ty::TSR_CODE_BASE`] + its row.
    pub fn code(self) -> u32 {
        match self {
            Ty::Wgsl(i) => {
                assert!(usize::from(i) < Ty::MAX_WGSL_DUALS, "our software has a bug: Wgsl({i}) is past its {} codes and would read back as Tsr", Ty::MAX_WGSL_DUALS);
                Ty::TABLE_CODE_BASE + u32::from(i)
            }
            Ty::Tsr(i) => Ty::TSR_CODE_BASE + u32::from(i),
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

    /// The first code of the TSR family; the WGSL family owns
    /// `TABLE_CODE_BASE..TSR_CODE_BASE` (4,096 duals).
    pub const TSR_CODE_BASE: u32 = Ty::TABLE_CODE_BASE + Ty::MAX_WGSL_DUALS as u32;

    /// The most duals a WGSL kingdom may number: its code range
    /// `TABLE_CODE_BASE..TSR_CODE_BASE`.
    pub const MAX_WGSL_DUALS: usize = 4096;

    /// The most duals a TSR kingdom may number: its index is a `u16`.
    pub const MAX_TSR_DUALS: usize = u16::MAX as usize + 1;

    /// The first code of the 1-D Futhark family; the TSR family
    /// owns `TABLE_CODE_BASE..FUTHARK_1D_CODE_BASE`.
    pub const FUTHARK_1D_CODE_BASE: u32 = Ty::TSR_CODE_BASE + Ty::MAX_TSR_DUALS as u32;

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
    /// offset from [`Ty::TABLE_CODE_BASE`], a TSR dual by offset from
    /// [`Ty::TSR_CODE_BASE`], a 1-D Futhark dual by offset from
    /// [`Ty::FUTHARK_1D_CODE_BASE`], a Futhark dual by offset from
    /// [`Ty::FUTHARK_CODE_BASE`].
    pub fn from_code(code: u32) -> Option<Ty> {
        if code >= Ty::FUTHARK_CODE_BASE {
            return u16::try_from(code - Ty::FUTHARK_CODE_BASE).ok().map(Ty::Futhark);
        }
        if code >= Ty::FUTHARK_1D_CODE_BASE {
            return u16::try_from(code - Ty::FUTHARK_1D_CODE_BASE).ok().map(Ty::Futhark1d);
        }
        if code >= Ty::TSR_CODE_BASE {
            return u16::try_from(code - Ty::TSR_CODE_BASE).ok().map(Ty::Tsr);
        }
        if code >= Ty::TABLE_CODE_BASE {
            return u16::try_from(code - Ty::TABLE_CODE_BASE).ok().map(Ty::Wgsl);
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
    for (row, ins, out) in typed_depth_signatures() {
        let mut arity = Arity::default();
        for ty in ins {
            *arity.inputs.entry(ty).or_insert(0) += 1;
        }
        arity.outputs.insert(out, 1);
        t.push(Symbol {
            kingdom: TYPED_SR.to_string(),
            symbol: row.symbol,
            symbol_name: row.symbol_name,
            alias: row.alias,
            semantic_id: row.semantic_id,
            arity,
        });
    }
    t
}

/// The typed-depth kingdom's rows with their argument types IN ORDER, `(symbol, ordered inputs, output)`: what
/// [`typed_depth_table`] counts into its many-hot [`Arity`], and what its [`KingdomTables`] are built from.
pub fn typed_depth_signatures() -> Vec<(SymbolRow, Vec<Ty>, Ty)> {
    let mut t = Vec::new();
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
            let ins: Vec<Ty> = ins.into_iter().map(|d| Ty::at_depth(d).expect("a ladder rung")).collect();
            let row = SymbolRow {
                symbol,
                symbol_name: s.semantic_id.clone(),
                alias: s.alias.clone(),
                semantic_id: s.semantic_id.clone(),
                spelling: String::new(),
            };
            t.push((row, ins, Ty::at_depth(out).expect("a ladder rung")));
            symbol += 1;
        }
    }
    t
}

/// One row of [`regex_rows`]: `(semantic_id, alias, ordered inputs, output)`.
pub type RegexRow = (&'static str, &'static str, &'static [Ty], Ty);

/// The regex kingdom's rows with their argument types IN ORDER: `(semantic_id,
/// alias, ordered inputs, output)`. [`Arity`] is many-hot and loses the order
/// (`regex_rep_n` takes `(Pattern, I)`, which a type-keyed map reads back as
/// `(I, Pattern)`), so the ordered signature relation of the typed-GEP kingdom
/// contract ([`crate::kingdom_contract`]) is built from this, not from the
/// table.
pub fn regex_rows() -> &'static [RegexRow] {
    const PAT: Ty = Ty::Pattern;
    const CC: Ty = Ty::CharClass;
    const CH: Ty = Ty::Char;
    const INT: Ty = Ty::I;
    const ROWS: &[RegexRow] = &[
        // ---- Pattern terminals (arity 0) ----
        ("regex_dot", ".", &[], PAT),
        ("regex_anchor_start", "^", &[], PAT),
        ("regex_anchor_end", "$", &[], PAT),
        ("regex_empty", "", &[], PAT),
        // ---- CharClass terminals (arity 0) — the built-in classes ----
        ("regex_cc_digit", r"\d", &[], CC),
        ("regex_cc_word", r"\w", &[], CC),
        ("regex_cc_space", r"\s", &[], CC),
        // ---- Pattern ops ----
        ("regex_star", "*", &[PAT], PAT),
        ("regex_plus", "+", &[PAT], PAT),
        ("regex_opt", "?", &[PAT], PAT),
        ("regex_concat", "concat", &[PAT, PAT], PAT),
        ("regex_alt", "|", &[PAT, PAT], PAT),
        // counted repetition — the Integer-typed slot that makes the kingdom
        // genuinely multi-typed (`docs/PLAN_multityped_gep.md`).
        ("regex_rep_n", "{n}", &[PAT, INT], PAT),
        ("regex_rep_upto", "{0,k}", &[PAT, INT], PAT),
        // ---- coercions / class algebra ----
        // a CharClass or a Char used as a Pattern (match one byte of it).
        ("regex_class_of", "class_of", &[CC], PAT),
        ("regex_lit", "lit", &[CH], PAT),
        ("regex_cc_range", "a-z", &[CH, CH], CC),
        ("regex_cc_union", "cc_union", &[CC, CC], CC),
        ("regex_cc_negate", "^CC", &[CC], CC),
    ];
    ROWS
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
    let rows = regex_rows();
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

// ---------------------------------------------------------------------------
// The gene frame's data model: data types, properties and scope
// (`docs/DESIGN_futhark_types_and_properties.md` section 3; plan section 7, task G1)
// ---------------------------------------------------------------------------
//
// Six tables beside the symbols: DataType, PropertyType, Argument, PropertyRule, Requirement, Binding. A type is a
// data type and nothing else; a fact about a value that is not its data type (an array's length, an index's bound)
// is a property; what a node may use is scope, built by the decoder from the Binding rows. Arity is DERIVED from the
// ordered Argument rows (the count of inputs of each data type, the same many-hot arity as before), so Karva's
// frames are sized exactly as they were (design 3.2). Properties, rules, requirements and bindings never change
// arity: they are read only after Karva has expressed the tree. A kingdom with no property rows is today's kingdom.

/// What a data type is made of. The members are data types of the same table, by their [`Ty`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataKind {
    /// A value with no parts: `f32`, `i64`, a regex `Pattern`.
    Scalar,
    /// An array of `element` (`[]f32`; `[][]f32` is an array of `[]f32`). Its length is a property, not a type.
    Array { element: Ty },
    /// A tuple of its components, in order: `(f32, f32, f32)`.
    Tuple { components: Vec<Ty> },
    /// A function from its parameters, in order, to its result: `[]f32 -> []f32`.
    Function { parameters: Vec<Ty>, result: Ty },
}

impl DataKind {
    /// The data types this kind names.
    fn members(&self) -> Vec<Ty> {
        match self {
            DataKind::Scalar => Vec::new(),
            DataKind::Array { element } => vec![*element],
            DataKind::Tuple { components } => components.clone(),
            DataKind::Function { parameters, result } => parameters.iter().copied().chain(std::iter::once(*result)).collect(),
        }
    }
}

/// One row of the DataType table: a data type, as the target language spells it without sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataType {
    pub id: Ty,
    pub name: String,
    pub kind: DataKind,
    /// The data type's constant, by its properties (`[0f32]`: `len = 1`; `0i64`: `bound = 1`): what a `var.T` gives
    /// when no variable of the type is in scope, and what the work-bound substitution gives a node with no input of
    /// its own type (master 6 README sections 2 and 5.3). `None`: the type has no constant (`size`).
    pub constant: Option<Vec<(String, PropertyExpr)>>,
}

/// What kind of fact a property is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PropertyKind {
    /// A length or a size value: `[]f32.len`, an `i64` size argument's `value`.
    Size,
    /// The exclusive upper bound of an index: `i64.bound`.
    Bound,
    /// A range of values.
    Range,
    /// A yes-or-no fact.
    Flag,
}

impl PropertyKind {
    const ALL: [PropertyKind; 4] = [PropertyKind::Size, PropertyKind::Bound, PropertyKind::Range, PropertyKind::Flag];

    pub fn name(self) -> &'static str {
        match self {
            PropertyKind::Size => "size",
            PropertyKind::Bound => "bound",
            PropertyKind::Range => "range",
            PropertyKind::Flag => "flag",
        }
    }

    pub fn from_name(name: &str) -> Option<PropertyKind> {
        PropertyKind::ALL.into_iter().find(|k| k.name() == name)
    }
}

/// One row of the PropertyType table: a property that the values of one data type carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyType {
    pub id: u32,
    pub data_type: Ty,
    pub name: String,
    pub kind: PropertyKind,
}

/// Whether an argument is an input or the output of its symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Direction {
    In,
    Out,
}

/// One row of the Argument table: one ordered input, or the output, of a symbol. A symbol's inputs are at positions
/// `0..n` with no gap; its one output is at position 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Argument {
    pub symbol: i64,
    pub position: u32,
    pub direction: Direction,
    pub data_type: Ty,
}

/// A property's value as the decoder knows it before run time: a size name of the environment (`n`, `k`), a literal,
/// `?` (known only at run time), a property of one of the symbol's inputs (`in0.len`), or a sum or a minimum of those,
/// kept symbolic (design 8, weakness 3: a sum is equal only to the same sum; plan 7.1: the `min` of two sizes is
/// symbolic, not `?`).
///
/// **One spelling per value** (master 6 README 5.1): a sum's terms are flattened and sorted, so `a + b` is `b + a`; a
/// min's arguments are flattened, sorted and without repeats, so `min(n, k)` is `min(k, n)` and `min(n, n)` is `n`.
/// [`PropertyExpr::sum`], [`PropertyExpr::min`] and [`PropertyExpr::parse`] build only that form; the table check
/// reports a `Sum` or `Min` built by hand in another.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PropertyExpr {
    Size(String),
    Literal(i64),
    Unknown,
    Input { position: u32, property: String },
    /// At least two terms, sorted, none a sum (build one with [`PropertyExpr::sum`]).
    Sum(Vec<PropertyExpr>),
    /// The least of at least two expressions, sorted, no two equal, none a min (build one with [`PropertyExpr::min`]).
    Min(Vec<PropertyExpr>),
}

fn is_identifier(s: &str) -> bool {
    let mut cs = s.chars();
    cs.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') && cs.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl PropertyExpr {
    /// The sum of `terms` in its one spelling: a nested sum's terms are spliced in, the terms are sorted, a single
    /// term is itself, and the sum of no term is the literal 0.
    pub fn sum(terms: Vec<PropertyExpr>) -> PropertyExpr {
        let mut flat = Vec::new();
        for t in terms {
            match t {
                PropertyExpr::Sum(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        flat.sort();
        match flat.len() {
            0 => PropertyExpr::Literal(0),
            1 => flat.remove(0),
            _ => PropertyExpr::Sum(flat),
        }
    }

    /// The least of `args` in its one spelling: a nested min's arguments are spliced in, sorted, repeats removed, and
    /// a single argument is itself. The min of no argument is `?` (no table writes one).
    pub fn min(args: Vec<PropertyExpr>) -> PropertyExpr {
        let mut flat = Vec::new();
        for a in args {
            match a {
                PropertyExpr::Min(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        flat.sort();
        flat.dedup();
        match flat.len() {
            0 => PropertyExpr::Unknown,
            1 => flat.remove(0),
            _ => PropertyExpr::Min(flat),
        }
    }

    /// The expression in its one spelling, at every depth.
    pub fn normal(&self) -> PropertyExpr {
        match self {
            PropertyExpr::Sum(terms) => PropertyExpr::sum(terms.iter().map(PropertyExpr::normal).collect()),
            PropertyExpr::Min(args) => PropertyExpr::min(args.iter().map(PropertyExpr::normal).collect()),
            other => other.clone(),
        }
    }

    /// Every `in<k>.<property>` term replaced by `value(k, property)`, the result in its one spelling.
    pub fn substitute(&self, value: &dyn Fn(u32, &str) -> PropertyExpr) -> PropertyExpr {
        match self {
            PropertyExpr::Input { position, property } => value(*position, property).normal(),
            PropertyExpr::Sum(terms) => PropertyExpr::sum(terms.iter().map(|t| t.substitute(value)).collect()),
            PropertyExpr::Min(args) => PropertyExpr::min(args.iter().map(|a| a.substitute(value)).collect()),
            other => other.clone(),
        }
    }

    /// True when a size name occurs anywhere in the expression.
    fn has_size_name(&self) -> bool {
        match self {
            PropertyExpr::Size(_) => true,
            PropertyExpr::Sum(terms) | PropertyExpr::Min(terms) => terms.iter().any(PropertyExpr::has_size_name),
            PropertyExpr::Literal(_) | PropertyExpr::Unknown | PropertyExpr::Input { .. } => false,
        }
    }

    /// True when `?` occurs anywhere in the expression.
    pub fn has_unknown(&self) -> bool {
        match self {
            PropertyExpr::Unknown => true,
            PropertyExpr::Sum(terms) | PropertyExpr::Min(terms) => terms.iter().any(PropertyExpr::has_unknown),
            PropertyExpr::Size(_) | PropertyExpr::Literal(_) | PropertyExpr::Input { .. } => false,
        }
    }

    /// True when the value is at least 1 whatever the sizes are: a size name (every dimension is at least 1, as the
    /// entries' contracts state), a literal of at least 1, a min of such, a sum of such and of literals of at least 0
    /// with one such term.
    fn at_least_one(&self) -> bool {
        match self {
            PropertyExpr::Size(_) => true,
            PropertyExpr::Literal(v) => *v >= 1,
            PropertyExpr::Min(args) => args.iter().all(PropertyExpr::at_least_one),
            PropertyExpr::Sum(terms) => {
                terms.iter().all(|t| t.at_least_one() || matches!(t, PropertyExpr::Literal(v) if *v >= 0)) && terms.iter().any(PropertyExpr::at_least_one)
            }
            PropertyExpr::Unknown | PropertyExpr::Input { .. } => false,
        }
    }

    /// Parse the text [`std::fmt::Display`] writes: `n`, `3`, `-1`, `?`, `in0.len`, `in0.len + in1.len`,
    /// `min(n, k) + 1`.
    pub fn parse(text: &str) -> Result<PropertyExpr, String> {
        let mut rest = text;
        let e = PropertyExpr::parse_sum(&mut rest)?;
        if !rest.trim().is_empty() {
            return Err(format!("our software has a bug: the property expression {text:?} has {rest:?} left over"));
        }
        Ok(e)
    }

    /// `term ('+' term)*`, consuming from `rest`.
    fn parse_sum(rest: &mut &str) -> Result<PropertyExpr, String> {
        let mut terms = vec![PropertyExpr::parse_term(rest)?];
        while let Some(after) = rest.trim_start().strip_prefix('+') {
            *rest = after;
            terms.push(PropertyExpr::parse_term(rest)?);
        }
        Ok(PropertyExpr::sum(terms))
    }

    /// `min(sum, sum, ...)` or one token: `?`, a literal, `in<k>.<property>`, a size name.
    fn parse_term(rest: &mut &str) -> Result<PropertyExpr, String> {
        *rest = rest.trim_start();
        if let Some(after) = rest.strip_prefix("min(") {
            *rest = after;
            let mut args = vec![PropertyExpr::parse_sum(rest)?];
            loop {
                let r = rest.trim_start();
                if let Some(after) = r.strip_prefix(',') {
                    *rest = after;
                    args.push(PropertyExpr::parse_sum(rest)?);
                } else if let Some(after) = r.strip_prefix(')') {
                    *rest = after;
                    break;
                } else {
                    return Err(format!("our software has a bug: a min( in a property expression is not closed before {r:?}"));
                }
            }
            if args.len() < 2 {
                return Err("our software has a bug: a min in a property expression has one argument".to_string());
            }
            return Ok(PropertyExpr::min(args));
        }
        let end = rest.find(|c: char| c.is_whitespace() || "+,()".contains(c)).unwrap_or(rest.len());
        let (t, after) = rest.split_at(end);
        *rest = after;
        if t == "?" {
            return Ok(PropertyExpr::Unknown);
        }
        if let Ok(v) = t.parse::<i64>() {
            return Ok(PropertyExpr::Literal(v));
        }
        if let Some((arg, property)) = t.split_once('.') {
            let position = arg.strip_prefix("in").filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())).and_then(|d| d.parse::<u32>().ok());
            return match position {
                Some(position) if is_identifier(property) => Ok(PropertyExpr::Input { position, property: property.to_string() }),
                _ => Err(format!("our software has a bug: the property expression term {t:?} is not in<k>.<property>")),
            };
        }
        if is_identifier(t) {
            return Ok(PropertyExpr::Size(t.to_string()));
        }
        Err(format!("our software has a bug: {t:?} is not a property expression (a size name, a literal, ?, in<k>.<property>, a sum or a min)"))
    }

    /// What is malformed in this expression, at any depth: a sum of fewer than two terms, with a sum inside or its
    /// terms out of order; a min of fewer than two arguments, with a min inside, or its arguments out of order or
    /// repeated; a size name that is not an identifier.
    fn shape_defects(&self) -> Vec<String> {
        match self {
            PropertyExpr::Sum(terms) => {
                let mut v = Vec::new();
                if terms.len() < 2 || terms.iter().any(|t| matches!(t, PropertyExpr::Sum(_))) {
                    v.push(format!("has a sum of {} terms or a sum inside a sum", terms.len()));
                }
                if terms.windows(2).any(|w| w[0] > w[1]) {
                    v.push(format!("has the sum {self} with its terms out of order (one spelling: {})", self.normal()));
                }
                v.extend(terms.iter().flat_map(PropertyExpr::shape_defects));
                v
            }
            PropertyExpr::Min(args) => {
                let mut v = Vec::new();
                if args.len() < 2 {
                    v.push(format!("has a min of {} arguments", args.len()));
                }
                if args.iter().any(|a| matches!(a, PropertyExpr::Min(_))) || args.windows(2).any(|w| w[0] >= w[1]) {
                    v.push(format!("has the min {self} with a min inside or its arguments out of order or repeated (one spelling: {})", self.normal()));
                }
                v.extend(args.iter().flat_map(PropertyExpr::shape_defects));
                v
            }
            PropertyExpr::Size(n) if !is_identifier(n) => vec![format!("names size {n:?}, which is not an identifier")],
            PropertyExpr::Size(_) | PropertyExpr::Literal(_) | PropertyExpr::Unknown | PropertyExpr::Input { .. } => Vec::new(),
        }
    }

    /// Every `(input position, property)` this expression reads.
    fn inputs(&self) -> Vec<(u32, &str)> {
        match self {
            PropertyExpr::Input { position, property } => vec![(*position, property.as_str())],
            PropertyExpr::Sum(terms) | PropertyExpr::Min(terms) => terms.iter().flat_map(PropertyExpr::inputs).collect(),
            PropertyExpr::Size(_) | PropertyExpr::Literal(_) | PropertyExpr::Unknown => Vec::new(),
        }
    }
}

impl std::fmt::Display for PropertyExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PropertyExpr::Size(n) => write!(f, "{n}"),
            PropertyExpr::Literal(v) => write!(f, "{v}"),
            PropertyExpr::Unknown => write!(f, "?"),
            PropertyExpr::Input { position, property } => write!(f, "in{position}.{property}"),
            PropertyExpr::Sum(terms) => {
                for (i, t) in terms.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{t}")?;
                }
                Ok(())
            }
            PropertyExpr::Min(args) => {
                write!(f, "min(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// One row of the PropertyRule table: how a property of a symbol's output follows from its inputs
/// (`map2.add`: out.len = in0.len; `tabulate`: out.len = in0.value).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyRule {
    pub symbol: i64,
    /// A property of the output's data type.
    pub property: String,
    pub expr: PropertyExpr,
}

/// A requirement's relation (master 6 README 5.1): `=` for lengths, `<=` for an index's bound against a length,
/// `<` for the work bound `env.trips < depth` only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Relation {
    Eq,
    Le,
    Lt,
}

impl Relation {
    pub fn symbol(self) -> &'static str {
        match self {
            Relation::Eq => "=",
            Relation::Le => "<=",
            Relation::Lt => "<",
        }
    }

    pub fn from_symbol(s: &str) -> Option<Relation> {
        [Relation::Eq, Relation::Le, Relation::Lt].into_iter().find(|r| r.symbol() == s)
    }

    /// THE COMPARISON (master 6 README 5.1), on the one spelling of each side. `?` anywhere fails every relation.
    /// `a = b` when the spellings are identical. `a <= b` when identical, or both literals and `a <= b`, or `a` a
    /// literal of at most 1 and `b` at least 1 for every size, or `b` a min and `a <=` each argument, or `a` a min
    /// and one argument `<= b`. `a < b` when both are literals and `a < b`. A failure a smarter comparison would
    /// have avoided costs only a needless adapter.
    pub fn holds(self, a: &PropertyExpr, b: &PropertyExpr) -> bool {
        let (a, b) = (a.normal(), b.normal());
        if a.has_unknown() || b.has_unknown() {
            return false;
        }
        match self {
            Relation::Eq => a == b,
            Relation::Le => at_most(&a, &b),
            Relation::Lt => matches!((&a, &b), (PropertyExpr::Literal(x), PropertyExpr::Literal(y)) if x < y),
        }
    }
}

/// `a <= b` for two expressions in their one spelling, with no `?` ([`Relation::holds`]).
fn at_most(a: &PropertyExpr, b: &PropertyExpr) -> bool {
    if a == b {
        return true;
    }
    match (a, b) {
        (PropertyExpr::Literal(x), PropertyExpr::Literal(y)) => return x <= y,
        (PropertyExpr::Literal(x), _) if *x <= 1 && b.at_least_one() => return true,
        _ => {}
    }
    if let PropertyExpr::Min(args) = b {
        if args.iter().all(|y| at_most(a, y)) {
            return true;
        }
    }
    if let PropertyExpr::Min(args) = a {
        if args.iter().any(|x| at_most(x, b)) {
            return true;
        }
    }
    false
}

/// What a requirement constrains: a property of one input (`in1.len`), or the environment's trip count `env.trips`,
/// the number of trip binders above the node (master 6 README 5.3; a property of no data type).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Subject {
    Input { position: u32, property: String },
    Trips,
}

impl std::fmt::Display for Subject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Subject::Input { position, property } => write!(f, "in{position}.{property}"),
            Subject::Trips => write!(f, "env.trips"),
        }
    }
}

impl Subject {
    /// Parse the text [`std::fmt::Display`] writes: `in1.len` or `env.trips`.
    pub fn parse(text: &str) -> Result<Subject, String> {
        if text == "env.trips" {
            return Ok(Subject::Trips);
        }
        match PropertyExpr::parse(text) {
            Ok(PropertyExpr::Input { position, property }) => Ok(Subject::Input { position, property }),
            _ => Err(format!("our software has a bug: the requirement subject {text:?} is neither in<k>.<property> nor env.trips")),
        }
    }
}

/// One row of the Requirement table: `subject relation against -> adapter` (`map2.add`: `in1.len = in0.len ->
/// fit.pair`; `get`: `in1.bound <= in0.len -> clampix`; a heavy row: `env.trips < 3 -> subst`). Each requirement names
/// the adapter that makes it hold at run time where it fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Requirement {
    pub symbol: i64,
    pub subject: Subject,
    pub relation: Relation,
    pub against: PropertyExpr,
    pub adapter: String,
}

/// Where an adapter's work runs (master 6 README 5.3): at its node's trips, at its body's (the trip argument's, one
/// more), or at the entry's root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AdapterSite {
    Node,
    Body,
    Root,
}

impl AdapterSite {
    pub fn name(self) -> &'static str {
        match self {
            AdapterSite::Node => "node",
            AdapterSite::Body => "body",
            AdapterSite::Root => "root",
        }
    }

    pub fn from_name(s: &str) -> Option<AdapterSite> {
        [AdapterSite::Node, AdapterSite::Body, AdapterSite::Root].into_iter().find(|a| a.name() == s)
    }
}

/// The two substitution adapters (master 6 README 5.3), named by a work bound `env.trips < depth`: what a node the
/// bound substitutes renders as. The substitute is a table fact, never a coincidence of data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Substitute {
    /// `subst`: the node's first input of its own data type, else its data type's constant (`map2.*`, `scan.*`, a
    /// loop's state; `sum`, `tab`).
    OwnInput,
    /// `subst.const`: the node's data type's constant, whatever its inputs (`hist.*`, `histrows.*`: the constant
    /// fitted to the slot is the histogram with nothing counted, where an own-type input is its indices or values).
    Constant,
}

impl Substitute {
    pub const ALL: [Substitute; 2] = [Substitute::OwnInput, Substitute::Constant];

    /// The adapter's name in `adapters.tsv` and in a `requires` column.
    pub fn adapter(self) -> &'static str {
        match self {
            Substitute::OwnInput => "subst",
            Substitute::Constant => "subst.const",
        }
    }

    /// The substitution adapter named `name`, `None` for any other adapter.
    pub fn of_adapter(name: &str) -> Option<Substitute> {
        Substitute::ALL.into_iter().find(|s| s.adapter() == name)
    }
}

/// One row of the Adapter table (master 6 `adapters.tsv`): its name, whether it is heavy (does work of the data's
/// size, as every `fit`), and where its work runs. A requirement on `env.trips` names a substitution adapter
/// ([`Substitute`]), and no other requirement does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adapter {
    pub name: String,
    pub heavy: bool,
    pub runs_at: AdapterSite,
}

/// A symbol's trip argument (master 6 `trip` column): the input whose subtree runs once a trip of the symbol, so its
/// environment's `env.trips` is the node's plus one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TripArgument {
    pub symbol: i64,
    pub position: u32,
}

/// One row of the Binding table: a variable an input argument makes visible to its own subtree, with its data type
/// and its properties by rule (`tabulate`: argument 1 binds an `i64` whose bound = in0.value).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub symbol: i64,
    pub position: u32,
    pub data_type: Ty,
    pub properties: Vec<(String, PropertyExpr)>,
}

/// A symbol of the tables, as [`Symbol`] without its arity (which the Argument rows give) and with its spelling in
/// the target language (`map2 (+) {0} {1}`; empty where the kingdom renders otherwise).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolRow {
    pub symbol: i64,
    pub symbol_name: String,
    pub alias: String,
    pub semantic_id: String,
    pub spelling: String,
}

impl SymbolRow {
    /// The row of an existing [`Symbol`], its arity dropped, with `spelling`.
    pub fn of(s: &Symbol, spelling: &str) -> SymbolRow {
        SymbolRow {
            symbol: s.symbol,
            symbol_name: s.symbol_name.clone(),
            alias: s.alias.clone(),
            semantic_id: s.semantic_id.clone(),
            spelling: spelling.to_string(),
        }
    }
}

impl Arity {
    /// THE DERIVATION (design 3.2): a symbol's many-hot arity is the count of its Argument rows of each data type,
    /// inputs and output apart.
    pub fn from_arguments<'a>(arguments: impl IntoIterator<Item = &'a Argument>) -> Arity {
        let mut a = Arity::default();
        for arg in arguments {
            let side = match arg.direction {
                Direction::In => &mut a.inputs,
                Direction::Out => &mut a.outputs,
            };
            *side.entry(arg.data_type).or_insert(0) += 1;
        }
        a
    }
}

/// One kingdom's tables: its symbols and the six tables of the design, with the adapters its requirements name and
/// its trip arguments (master 6 README 5).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KingdomTables {
    pub kingdom: String,
    pub symbols: Vec<SymbolRow>,
    pub data_types: Vec<DataType>,
    pub property_types: Vec<PropertyType>,
    pub arguments: Vec<Argument>,
    pub property_rules: Vec<PropertyRule>,
    pub requirements: Vec<Requirement>,
    pub bindings: Vec<Binding>,
    pub adapters: Vec<Adapter>,
    pub trip_arguments: Vec<TripArgument>,
}

/// An entry parameter as [`KingdomTables::entry_defects`] reads it: its name, its data type and its properties.
pub type EntryParameter<'a> = (&'a str, Ty, &'a [(String, PropertyExpr)]);

/// A requirement on the entry's result as [`KingdomTables::entry_defects`] reads it: its property, its relation, the
/// expression it compares with and its adapter.
pub type EntryRequirement<'a> = (&'a str, Relation, &'a PropertyExpr, &'a str);

/// The comparability classes of property kinds (master 6 README 5.2): a size and a bound both count elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum KindClass {
    Count,
    Range,
    Flag,
}

impl KindClass {
    fn of(k: PropertyKind) -> KindClass {
        match k {
            PropertyKind::Size | PropertyKind::Bound => KindClass::Count,
            PropertyKind::Range => KindClass::Range,
            PropertyKind::Flag => KindClass::Flag,
        }
    }

    fn name(self) -> &'static str {
        match self {
            KindClass::Count => "size or bound",
            KindClass::Range => "range",
            KindClass::Flag => "flag",
        }
    }
}

/// The relations a property of `kind` takes (master 6 README 5.2): `=` and `<=` for a size, `<=` for a bound, `=`
/// for a range or a flag.
fn relations_of(kind: PropertyKind) -> &'static [Relation] {
    match kind {
        PropertyKind::Size => &[Relation::Eq, Relation::Le],
        PropertyKind::Bound => &[Relation::Le],
        PropertyKind::Range | PropertyKind::Flag => &[Relation::Eq],
    }
}

/// Whether a property of `kind` compares with an expression whose terms carry the classes `against` (README 5.2): the
/// same class, and a literal (no class) only for a size or a bound.
fn comparable(kind: PropertyKind, against: &std::collections::BTreeSet<KindClass>) -> bool {
    let mine = KindClass::of(kind);
    against.iter().all(|c| *c == mine) && (mine == KindClass::Count || !against.is_empty())
}

/// Text fields hold no tab and no line break; the text form is one row a line, its fields tab-separated.
fn field(kind: &str, s: &str) -> Result<String, String> {
    if s.contains(['\t', '\n', '\r']) {
        return Err(format!("our software has a bug: the {kind} {s:?} holds a tab or a line break, which the text form cannot carry"));
    }
    Ok(s.to_string())
}

fn codes(tys: &[Ty]) -> String {
    if tys.is_empty() {
        return "-".to_string();
    }
    tys.iter().map(|t| t.code().to_string()).collect::<Vec<_>>().join(",")
}

fn ty_of(text: &str) -> Result<Ty, String> {
    text.parse::<u32>().ok().and_then(Ty::from_code).ok_or_else(|| format!("our software has a bug: {text:?} is not a type code"))
}

fn tys_of(text: &str) -> Result<Vec<Ty>, String> {
    if text == "-" {
        return Ok(Vec::new());
    }
    text.split(',').map(ty_of).collect()
}

/// `p=e;q=f`, or `-` for none: a binding's or a constant's properties in the text form.
fn property_list(props: &[(String, PropertyExpr)]) -> String {
    if props.is_empty() {
        return "-".to_string();
    }
    props.iter().map(|(p, e)| format!("{p}={e}")).collect::<Vec<_>>().join(";")
}

fn parse_property_list(text: &str) -> Result<Vec<(String, PropertyExpr)>, String> {
    if text == "-" {
        return Ok(Vec::new());
    }
    text.split(';')
        .map(|pe| {
            let (p, e) = pe.split_once('=').ok_or_else(|| format!("our software has a bug: {pe:?} is not <property>=<expression>"))?;
            Ok((p.to_string(), PropertyExpr::parse(e)?))
        })
        .collect()
}

fn number<T: std::str::FromStr>(what: &str, text: &str) -> Result<T, String> {
    text.parse::<T>().map_err(|_| format!("our software has a bug: the {what} {text:?} is not a number"))
}

impl KingdomTables {
    /// The tables of a kingdom with no property rows, from its symbols with their ordered inputs and output: one
    /// Argument row a slot, and one scalar data type for each type that occurs, named by its `Debug` text.
    pub fn from_signatures(kingdom: &str, rows: Vec<(SymbolRow, Vec<Ty>, Ty)>) -> KingdomTables {
        let mut t = KingdomTables { kingdom: kingdom.to_string(), ..KingdomTables::default() };
        let mut seen: std::collections::BTreeSet<Ty> = std::collections::BTreeSet::new();
        for (row, ins, out) in rows {
            for (i, ty) in ins.iter().enumerate() {
                t.arguments.push(Argument { symbol: row.symbol, position: i as u32, direction: Direction::In, data_type: *ty });
            }
            t.arguments.push(Argument { symbol: row.symbol, position: 0, direction: Direction::Out, data_type: out });
            seen.extend(ins);
            seen.insert(out);
            t.symbols.push(row);
        }
        t.data_types = seen.into_iter().map(|ty| DataType { id: ty, name: format!("{ty:?}"), kind: DataKind::Scalar, constant: None }).collect();
        t
    }

    /// The order in which the decoder visits a symbol's inputs: each input after every input its own Binding rows
    /// read (`tabulate` binds `i` with bound = in0.value, so in0 before in1), otherwise in position order. An error
    /// when a binding reads its own argument or the bindings read each other's in a cycle (the check reports both).
    pub fn decode_order(&self, symbol: i64) -> Result<Vec<u32>, String> {
        let n = self.inputs(symbol).len() as u32;
        let mut after: BTreeMap<u32, std::collections::BTreeSet<u32>> = (0..n).map(|k| (k, std::collections::BTreeSet::new())).collect();
        for b in self.bindings.iter().filter(|b| b.symbol == symbol) {
            for (_, e) in &b.properties {
                for (j, p) in e.inputs() {
                    if j == b.position {
                        return Err(format!("our software has a bug: kingdom {}: symbol {symbol}'s binding of argument {j} reads in{j}.{p}, its own argument", self.kingdom));
                    }
                    if let Some(s) = after.get_mut(&b.position) {
                        s.insert(j);
                    }
                }
            }
        }
        let mut order: Vec<u32> = Vec::with_capacity(n as usize);
        while order.len() < n as usize {
            let next = (0..n).find(|k| !order.contains(k) && after[k].iter().all(|j| *j >= n || order.contains(j)));
            match next {
                Some(k) => order.push(k),
                None => return Err(format!("our software has a bug: kingdom {}: symbol {symbol}'s bindings read each other's arguments in a cycle", self.kingdom)),
            }
        }
        Ok(order)
    }

    /// True when the kingdom has any property, rule, requirement or binding row: false is today's kingdom.
    pub fn has_properties(&self) -> bool {
        !(self.property_types.is_empty() && self.property_rules.is_empty() && self.requirements.is_empty() && self.bindings.is_empty())
    }

    /// The Argument rows grouped by symbol, inputs in position order then the output.
    fn arguments_by_symbol(&self) -> BTreeMap<i64, Vec<&Argument>> {
        let mut by: BTreeMap<i64, Vec<&Argument>> = BTreeMap::new();
        for a in &self.arguments {
            by.entry(a.symbol).or_default().push(a);
        }
        for args in by.values_mut() {
            args.sort_by_key(|a| (a.direction, a.position));
        }
        by
    }

    /// A symbol's input data types in argument order.
    pub fn inputs(&self, symbol: i64) -> Vec<Ty> {
        let mut ins: Vec<&Argument> = self.arguments.iter().filter(|a| a.symbol == symbol && a.direction == Direction::In).collect();
        ins.sort_by_key(|a| a.position);
        ins.into_iter().map(|a| a.data_type).collect()
    }

    /// A symbol's output data type.
    pub fn output(&self, symbol: i64) -> Option<Ty> {
        self.arguments.iter().find(|a| a.symbol == symbol && a.direction == Direction::Out).map(|a| a.data_type)
    }

    /// A symbol's arity, derived from its Argument rows ([`Arity::from_arguments`]).
    pub fn arity(&self, symbol: i64) -> Arity {
        Arity::from_arguments(self.arguments.iter().filter(|a| a.symbol == symbol))
    }

    /// Every symbol's ordered signature, `(symbol, inputs in position order, output)`, in symbol order: what a typed
    /// kingdom contract is built from. A symbol with no output row is left out (the check reports it).
    pub fn signatures(&self) -> Vec<(i64, Vec<Ty>, Ty)> {
        let by = self.arguments_by_symbol();
        self.symbols
            .iter()
            .filter_map(|s| {
                let args = by.get(&s.symbol)?;
                let out = args.iter().find(|a| a.direction == Direction::Out)?.data_type;
                let ins = args.iter().filter(|a| a.direction == Direction::In).map(|a| a.data_type).collect();
                Some((s.symbol, ins, out))
            })
            .collect()
    }

    /// The kingdom as a [`SymbolTable`], each symbol's arity derived from its Argument rows, in symbol order.
    pub fn symbol_table(&self) -> SymbolTable {
        let by = self.arguments_by_symbol();
        let mut t = SymbolTable::new();
        for s in &self.symbols {
            t.push(Symbol {
                kingdom: self.kingdom.clone(),
                symbol: s.symbol,
                symbol_name: s.symbol_name.clone(),
                alias: s.alias.clone(),
                semantic_id: s.semantic_id.clone(),
                arity: Arity::from_arguments(by.get(&s.symbol).into_iter().flatten().copied()),
            });
        }
        t
    }

    /// THE TABLE CHECK. Every defect, each "our software has a bug: ...": symbols and data types unique; every type a
    /// data type names exists; every property type's data type exists and its name is unique on it; every symbol has
    /// one output at position 0 and inputs at dense positions `0..n`; every rule names a property of its symbol's
    /// output, every requirement a property of an existing input, every binding an existing input and properties of
    /// its own data type; every `in<k>.<property>` an expression reads names an existing input and one of its
    /// properties; every sum and min is in its one spelling. And (plan 7, G3; master 6 README 5): a constant sets
    /// properties its type carries, once each, reading no input; adapters are unique; a symbol has at most one trip
    /// argument and it exists; a requirement's relation is one its subject's kind takes (`=`, `<=` for a size, `<=`
    /// for a bound, `<` for `env.trips` against a literal only) and its other side is of a comparable kind (a size and
    /// a bound both count elements, a literal has no kind); it names an adapter that exists, not a root one, and a
    /// body one only on a symbol with a trip argument; no binding reads its own argument and the bindings of one
    /// symbol read each other's in no cycle ([`KingdomTables::decode_order`]). And the work bound (G3 review findings
    /// 2, 3, 6): a requirement on `env.trips` names a substitution adapter ([`Substitute`]) and no other requirement
    /// does, which is light and runs at the node; a symbol has at most one; when any symbol has one, every symbol that
    /// names a heavy adapter has one; and a substitution by a constant is of a data type that has one.
    pub fn check(&self) -> Result<(), Vec<String>> {
        let k = &self.kingdom;
        let mut d: Vec<String> = Vec::new();
        let mut symbols: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        for s in &self.symbols {
            if !symbols.insert(s.symbol) {
                d.push(format!("our software has a bug: kingdom {k}: symbol {} ({}) is listed twice", s.symbol, s.symbol_name));
            }
        }
        let mut types: BTreeMap<Ty, &DataType> = BTreeMap::new();
        let mut type_names: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for t in &self.data_types {
            if types.insert(t.id, t).is_some() {
                d.push(format!("our software has a bug: kingdom {k}: data type {:?} is listed twice", t.id));
            }
            if !type_names.insert(t.name.as_str()) {
                d.push(format!("our software has a bug: kingdom {k}: two data types are named {:?}", t.name));
            }
        }
        for t in &self.data_types {
            for m in t.kind.members() {
                if !types.contains_key(&m) {
                    d.push(format!("our software has a bug: kingdom {k}: data type {} names {m:?}, which is not a data type", t.name));
                }
            }
        }
        let mut props: BTreeMap<(Ty, &str), &PropertyType> = BTreeMap::new();
        let mut prop_ids: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for p in &self.property_types {
            if !prop_ids.insert(p.id) {
                d.push(format!("our software has a bug: kingdom {k}: property type {} is listed twice", p.id));
            }
            if !types.contains_key(&p.data_type) {
                d.push(format!("our software has a bug: kingdom {k}: property {} is on {:?}, which is not a data type", p.name, p.data_type));
            }
            if !is_identifier(&p.name) {
                d.push(format!("our software has a bug: kingdom {k}: property name {:?} is not an identifier", p.name));
            }
            if props.insert((p.data_type, p.name.as_str()), p).is_some() {
                d.push(format!("our software has a bug: kingdom {k}: data type {:?} has two properties named {}", p.data_type, p.name));
            }
        }
        let by = self.arguments_by_symbol();
        let mut ins_of: BTreeMap<i64, Vec<Ty>> = BTreeMap::new();
        let mut out_of: BTreeMap<i64, Ty> = BTreeMap::new();
        for (sym, args) in &by {
            if !symbols.contains(sym) {
                d.push(format!("our software has a bug: kingdom {k}: an argument names symbol {sym}, which is not a symbol"));
            }
            let outs: Vec<&&Argument> = args.iter().filter(|a| a.direction == Direction::Out).collect();
            match outs.as_slice() {
                [o] if o.position == 0 => {
                    out_of.insert(*sym, o.data_type);
                }
                [o] => d.push(format!("our software has a bug: kingdom {k}: symbol {sym}'s output is at position {}, not 0", o.position)),
                _ => d.push(format!("our software has a bug: kingdom {k}: symbol {sym} has {} outputs, not one", outs.len())),
            }
            let ins: Vec<&&Argument> = args.iter().filter(|a| a.direction == Direction::In).collect();
            for (i, a) in ins.iter().enumerate() {
                if a.position != i as u32 {
                    d.push(format!("our software has a bug: kingdom {k}: symbol {sym}'s input {i} is at position {} (positions must be dense from 0)", a.position));
                    break;
                }
            }
            ins_of.insert(*sym, ins.iter().map(|a| a.data_type).collect());
            for a in args {
                if !types.contains_key(&a.data_type) {
                    d.push(format!("our software has a bug: kingdom {k}: symbol {sym}'s argument takes {:?}, which is not a data type", a.data_type));
                }
            }
        }
        for s in &self.symbols {
            if !by.contains_key(&s.symbol) {
                d.push(format!("our software has a bug: kingdom {k}: symbol {} ({}) has no output", s.symbol, s.symbol_name));
            }
        }
        let has = |ty: Ty, name: &str| props.contains_key(&(ty, name));
        // An expression's `in<k>.<property>` terms checked against symbol `sym`'s inputs, and its shape.
        let expr_defects = |sym: i64, where_: &str, e: &PropertyExpr| -> Vec<String> {
            let mut v = Vec::new();
            let ins = ins_of.get(&sym).map_or(&[][..], Vec::as_slice);
            for (pos, p) in e.inputs() {
                match ins.get(pos as usize) {
                    None => v.push(format!("our software has a bug: kingdom {k}: symbol {sym}'s {where_} reads in{pos}.{p}, but it has {} inputs", ins.len())),
                    Some(ty) if !has(*ty, p) => {
                        v.push(format!("our software has a bug: kingdom {k}: symbol {sym}'s {where_} reads in{pos}.{p}, which {ty:?} does not carry"))
                    }
                    Some(_) => {}
                }
            }
            v.extend(e.shape_defects().into_iter().map(|what| format!("our software has a bug: kingdom {k}: symbol {sym}'s {where_} {what}")));
            v
        };
        let mut ruled: std::collections::BTreeSet<(i64, &str)> = std::collections::BTreeSet::new();
        for r in &self.property_rules {
            match out_of.get(&r.symbol) {
                None => d.push(format!("our software has a bug: kingdom {k}: a property rule names symbol {}, which has no output", r.symbol)),
                Some(ty) if !has(*ty, &r.property) => {
                    d.push(format!("our software has a bug: kingdom {k}: symbol {}'s rule sets out.{}, which {ty:?} does not carry", r.symbol, r.property))
                }
                Some(_) => {}
            }
            if !ruled.insert((r.symbol, r.property.as_str())) {
                d.push(format!("our software has a bug: kingdom {k}: symbol {} has two rules for out.{}", r.symbol, r.property));
            }
            d.extend(expr_defects(r.symbol, &format!("rule for out.{}", r.property), &r.expr));
        }
        for t in &self.data_types {
            let mut named: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
            for (p, e) in t.constant.iter().flatten() {
                if !has(t.id, p) {
                    d.push(format!("our software has a bug: kingdom {k}: data type {}'s constant sets {p}, which it does not carry", t.name));
                }
                if !named.insert(p.as_str()) {
                    d.push(format!("our software has a bug: kingdom {k}: data type {}'s constant sets {p} twice", t.name));
                }
                if !e.inputs().is_empty() {
                    d.push(format!("our software has a bug: kingdom {k}: data type {}'s constant sets {p} = {e}, which reads an input (a constant has none)", t.name));
                }
                d.extend(e.shape_defects().into_iter().map(|what| format!("our software has a bug: kingdom {k}: data type {}'s constant {what}", t.name)));
            }
        }
        let mut adapters: BTreeMap<&str, &Adapter> = BTreeMap::new();
        for a in &self.adapters {
            if a.name.is_empty() || a.name.contains(char::is_whitespace) {
                d.push(format!("our software has a bug: kingdom {k}: adapter name {:?} is empty or holds a space", a.name));
            }
            if adapters.insert(a.name.as_str(), a).is_some() {
                d.push(format!("our software has a bug: kingdom {k}: adapter {} is listed twice", a.name));
            }
            if Substitute::of_adapter(&a.name).is_some() && (a.heavy || a.runs_at != AdapterSite::Node) {
                d.push(format!("our software has a bug: kingdom {k}: adapter {} is a substitution adapter, so it is light and runs at the node", a.name));
            }
        }
        let mut trip_of: BTreeMap<i64, u32> = BTreeMap::new();
        for tr in &self.trip_arguments {
            let ins = ins_of.get(&tr.symbol).map_or(&[][..], Vec::as_slice);
            if (tr.position as usize) >= ins.len() {
                d.push(format!("our software has a bug: kingdom {k}: symbol {}'s trip argument {} does not exist", tr.symbol, tr.position));
            }
            if trip_of.insert(tr.symbol, tr.position).is_some() {
                d.push(format!("our software has a bug: kingdom {k}: symbol {} has two trip arguments", tr.symbol));
            }
        }
        // The kinds an expression's terms carry, as comparability classes (a literal and `?` carry none).
        let classes = |sym: i64, e: &PropertyExpr| -> std::collections::BTreeSet<KindClass> {
            let ins = ins_of.get(&sym).map_or(&[][..], Vec::as_slice);
            let mut c: std::collections::BTreeSet<KindClass> = e.inputs().into_iter().filter_map(|(pos, p)| ins.get(pos as usize).and_then(|ty| props.get(&(*ty, p))).map(|pt| KindClass::of(pt.kind))).collect();
            if e.has_size_name() {
                c.insert(KindClass::Count);
            }
            c
        };
        for q in &self.requirements {
            let s = q.symbol;
            let ins = ins_of.get(&s).map_or(&[][..], Vec::as_slice);
            let against = classes(s, &q.against);
            if against.len() > 1 {
                let names: Vec<&str> = against.iter().map(|c| c.name()).collect();
                d.push(format!("our software has a bug: kingdom {k}: symbol {s}'s requirement on {} compares against {}, which mixes kinds ({})", q.subject, q.against, names.join(", ")));
            }
            match &q.subject {
                Subject::Input { position, property } => match ins.get(*position as usize) {
                    None => d.push(format!("our software has a bug: kingdom {k}: symbol {s}'s requirement names input {position}, which does not exist")),
                    Some(ty) => match props.get(&(*ty, property.as_str())) {
                        None => d.push(format!("our software has a bug: kingdom {k}: symbol {s}'s requirement names in{position}.{property}, which {ty:?} does not carry")),
                        Some(pt) => {
                            if !relations_of(pt.kind).contains(&q.relation) {
                                d.push(format!(
                                    "our software has a bug: kingdom {k}: symbol {s}'s requirement {} {} {} uses {}, which a {} property does not take",
                                    q.subject,
                                    q.relation.symbol(),
                                    q.against,
                                    q.relation.symbol(),
                                    pt.kind.name()
                                ));
                            }
                            if !comparable(pt.kind, &against) {
                                d.push(format!("our software has a bug: kingdom {k}: symbol {s}'s requirement compares the {} property {} with {}, which is not of a comparable kind", pt.kind.name(), q.subject, q.against));
                            }
                        }
                    },
                },
                Subject::Trips => {
                    if q.relation != Relation::Lt || !matches!(q.against, PropertyExpr::Literal(_)) {
                        d.push(format!("our software has a bug: kingdom {k}: symbol {s}'s requirement on env.trips is {} {}, not < a literal", q.relation.symbol(), q.against));
                    }
                    if Substitute::of_adapter(&q.adapter).is_none() {
                        d.push(format!("our software has a bug: kingdom {k}: symbol {s}'s requirement on env.trips names adapter {}, which is not a substitution adapter (subst, subst.const)", q.adapter));
                    }
                }
            }
            if q.subject != Subject::Trips && Substitute::of_adapter(&q.adapter).is_some() {
                d.push(format!("our software has a bug: kingdom {k}: symbol {s}'s requirement on {} names the substitution adapter {}, which only env.trips takes", q.subject, q.adapter));
            }
            if q.relation == Relation::Lt && q.subject != Subject::Trips {
                d.push(format!("our software has a bug: kingdom {k}: symbol {s}'s requirement on {} uses <, which only env.trips takes", q.subject));
            }
            match adapters.get(q.adapter.as_str()) {
                None => d.push(format!("our software has a bug: kingdom {k}: symbol {s}'s requirement on {} names adapter {:?}, which is not an adapter", q.subject, q.adapter)),
                Some(a) if a.runs_at == AdapterSite::Root => {
                    d.push(format!("our software has a bug: kingdom {k}: symbol {s}'s requirement on {} names adapter {}, which runs at the root", q.subject, q.adapter))
                }
                Some(a) if a.runs_at == AdapterSite::Body && !trip_of.contains_key(&s) => d.push(format!(
                    "our software has a bug: kingdom {k}: symbol {s}'s requirement on {} names adapter {}, which runs at the body, but the symbol has no trip argument",
                    q.subject, q.adapter
                )),
                Some(_) => {}
            }
            d.extend(expr_defects(s, &format!("requirement on {}", q.subject), &q.against));
        }
        // The work bound (master 6 README 5.3): at most one a symbol; once any symbol has one, every symbol that names
        // a heavy adapter has one (else the adapter's work escapes `max(size) ** depth`); and a substitution by a
        // constant needs its data type to have one, found here and not when a gene reaches it.
        let mut work: BTreeMap<i64, &Requirement> = BTreeMap::new();
        for q in self.requirements.iter().filter(|q| q.subject == Subject::Trips) {
            if work.insert(q.symbol, q).is_some() {
                d.push(format!("our software has a bug: kingdom {k}: symbol {} has two requirements on env.trips", q.symbol));
            }
        }
        if !work.is_empty() {
            for q in self.requirements.iter().filter(|q| !work.contains_key(&q.symbol)) {
                if adapters.get(q.adapter.as_str()).is_some_and(|a| a.heavy && a.runs_at != AdapterSite::Root) {
                    d.push(format!(
                        "our software has a bug: kingdom {k}: symbol {}'s requirement on {} names the heavy adapter {}, but the symbol has no requirement on env.trips, so the adapter's work escapes the work bound",
                        q.symbol, q.subject, q.adapter
                    ));
                }
            }
        }
        for (s, q) in &work {
            let (Some(out), Some(by)) = (out_of.get(s), Substitute::of_adapter(&q.adapter)) else { continue };
            let own_input = ins_of.get(s).is_some_and(|ins| ins.contains(out));
            if (by == Substitute::Constant || !own_input) && types.get(out).is_some_and(|t| t.constant.is_none()) {
                d.push(format!("our software has a bug: kingdom {k}: symbol {s}'s work bound ({}) substitutes the constant of {}, which has none", q.adapter, types[out].name));
            }
        }
        for s in &self.symbols {
            if let Err(e) = self.decode_order(s.symbol) {
                d.push(e);
            }
        }
        for b in &self.bindings {
            let ins = ins_of.get(&b.symbol).map_or(&[][..], Vec::as_slice);
            if (b.position as usize) >= ins.len() {
                d.push(format!("our software has a bug: kingdom {k}: symbol {}'s binding names argument {}, which does not exist", b.symbol, b.position));
            }
            if !types.contains_key(&b.data_type) {
                d.push(format!("our software has a bug: kingdom {k}: symbol {}'s binding is of {:?}, which is not a data type", b.symbol, b.data_type));
            }
            let mut named: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
            for (p, e) in &b.properties {
                if !has(b.data_type, p) {
                    d.push(format!("our software has a bug: kingdom {k}: symbol {}'s binding sets {p}, which {:?} does not carry", b.symbol, b.data_type));
                }
                if !named.insert(p.as_str()) {
                    d.push(format!("our software has a bug: kingdom {k}: symbol {}'s binding sets {p} twice", b.symbol));
                }
                d.extend(expr_defects(b.symbol, &format!("binding of argument {}", b.position), e));
            }
        }
        if d.is_empty() {
            Ok(())
        } else {
            Err(d)
        }
    }

    /// THE ENTRY CHECK (G3 review finding 5), against these tables. Every defect, each "our software has a bug: ...":
    /// the result's data type is one of the tables'; each parameter's name is an identifier given once, its data type
    /// is the tables', and it sets properties its data type carries, once each, reading no input, in their one
    /// spelling; each requirement on the result names a property its data type carries, by a relation that
    /// property's kind takes, against an expression of a comparable kind (README 5.2) that reads no input and no `?`,
    /// and names an adapter that runs at the root. `variables` are `(name, data type, properties)`, `requirements`
    /// `(property, relation, against, adapter)`.
    pub fn entry_defects(&self, output: Ty, variables: &[EntryParameter], requirements: &[EntryRequirement]) -> Vec<String> {
        let k = &self.kingdom;
        let mut d: Vec<String> = Vec::new();
        let is_type = |ty: Ty| self.data_types.iter().any(|t| t.id == ty);
        let kind_of = |ty: Ty, p: &str| self.property_types.iter().find(|pt| pt.data_type == ty && pt.name == p).map(|pt| pt.kind);
        if !is_type(output) {
            d.push(format!("our software has a bug: kingdom {k}: the entry returns {output:?}, which is not a data type"));
        }
        let mut names: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for (name, ty, props) in variables {
            if !is_identifier(name) {
                d.push(format!("our software has a bug: kingdom {k}: the entry's parameter name {name:?} is not an identifier"));
            }
            if !names.insert(name) {
                d.push(format!("our software has a bug: kingdom {k}: the entry has two parameters named {name}"));
            }
            if !is_type(*ty) {
                d.push(format!("our software has a bug: kingdom {k}: the entry's parameter {name} is of {ty:?}, which is not a data type"));
            }
            let mut set: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
            for (p, e) in props.iter() {
                if kind_of(*ty, p).is_none() {
                    d.push(format!("our software has a bug: kingdom {k}: the entry's parameter {name} sets {p}, which {ty:?} does not carry"));
                }
                if !set.insert(p.as_str()) {
                    d.push(format!("our software has a bug: kingdom {k}: the entry's parameter {name} sets {p} twice"));
                }
                if !e.inputs().is_empty() {
                    d.push(format!("our software has a bug: kingdom {k}: the entry's parameter {name} sets {p} = {e}, which reads an input (the entry has none)"));
                }
                d.extend(e.shape_defects().into_iter().map(|what| format!("our software has a bug: kingdom {k}: the entry's parameter {name}'s {p} {what}")));
            }
        }
        for (p, relation, against, adapter) in requirements {
            let at = format!("the entry's requirement out.{p} {} {against}", relation.symbol());
            match kind_of(output, p) {
                None => d.push(format!("our software has a bug: kingdom {k}: {at} names out.{p}, which {output:?} does not carry")),
                Some(kind) => {
                    if !relations_of(kind).contains(relation) {
                        d.push(format!("our software has a bug: kingdom {k}: {at} uses {}, which a {} property does not take", relation.symbol(), kind.name()));
                    }
                    let classes: std::collections::BTreeSet<KindClass> = if against.has_size_name() { [KindClass::Count].into() } else { Default::default() };
                    if !comparable(kind, &classes) {
                        d.push(format!("our software has a bug: kingdom {k}: {at} compares the {} property with {against}, which is not of a comparable kind", kind.name()));
                    }
                }
            }
            if !against.inputs().is_empty() || against.has_unknown() {
                d.push(format!("our software has a bug: kingdom {k}: {at} reads an input or ?, which the entry's result never meets"));
            }
            d.extend(against.shape_defects().into_iter().map(|what| format!("our software has a bug: kingdom {k}: {at} {what}")));
            match self.adapters.iter().find(|a| a.name == *adapter) {
                None => d.push(format!("our software has a bug: kingdom {k}: {at} names adapter {adapter:?}, which is not an adapter")),
                Some(a) if a.runs_at != AdapterSite::Root => d.push(format!("our software has a bug: kingdom {k}: {at} names adapter {adapter}, which does not run at the root")),
                Some(_) => {}
            }
        }
        d
    }

    /// The tables as text: one row a line, its table's name first and its fields tab-separated; types by their
    /// [`Ty::code`]. [`KingdomTables::parse`] reads it back to the same tables. Tables that fail
    /// [`KingdomTables::check`] are an error, not text: they could read back as other tables.
    pub fn to_text(&self) -> Result<String, String> {
        if let Err(d) = self.check() {
            return Err(format!(
                "our software has a bug: kingdom {}: the tables fail their check ({} defects), so their text would not read back the same: {}",
                self.kingdom,
                d.len(),
                d.join("; ")
            ));
        }
        let mut lines = vec![format!("kingdom\t{}", field("kingdom name", &self.kingdom)?)];
        for s in &self.symbols {
            lines.push(format!(
                "symbol\t{}\t{}\t{}\t{}\t{}",
                s.symbol,
                field("symbol name", &s.symbol_name)?,
                field("alias", &s.alias)?,
                field("semantic id", &s.semantic_id)?,
                field("spelling", &s.spelling)?
            ));
        }
        for t in &self.data_types {
            let kind = match &t.kind {
                DataKind::Scalar => "scalar".to_string(),
                DataKind::Array { element } => format!("array\t{}", element.code()),
                DataKind::Tuple { components } => format!("tuple\t{}", codes(components)),
                DataKind::Function { parameters, result } => format!("function\t{}\t{}", codes(parameters), result.code()),
            };
            lines.push(format!("data_type\t{}\t{}\t{kind}", t.id.code(), field("data type name", &t.name)?));
            if let Some(c) = &t.constant {
                lines.push(format!("constant\t{}\t{}", t.id.code(), property_list(c)));
            }
        }
        for p in &self.property_types {
            lines.push(format!("property_type\t{}\t{}\t{}\t{}", p.id, p.data_type.code(), field("property name", &p.name)?, p.kind.name()));
        }
        for a in &self.arguments {
            let dir = match a.direction {
                Direction::In => "in",
                Direction::Out => "out",
            };
            lines.push(format!("argument\t{}\t{}\t{dir}\t{}", a.symbol, a.position, a.data_type.code()));
        }
        for r in &self.property_rules {
            lines.push(format!("rule\t{}\t{}\t{}", r.symbol, field("property name", &r.property)?, r.expr));
        }
        for q in &self.requirements {
            lines.push(format!("requirement\t{}\t{}\t{}\t{}\t{}", q.symbol, q.subject, q.relation.symbol(), q.against, field("adapter name", &q.adapter)?));
        }
        for b in &self.bindings {
            lines.push(format!("binding\t{}\t{}\t{}\t{}", b.symbol, b.position, b.data_type.code(), property_list(&b.properties)));
        }
        for a in &self.adapters {
            lines.push(format!("adapter\t{}\t{}\t{}", field("adapter name", &a.name)?, if a.heavy { "heavy" } else { "light" }, a.runs_at.name()));
        }
        for tr in &self.trip_arguments {
            lines.push(format!("trip\t{}\t{}", tr.symbol, tr.position));
        }
        let mut text = lines.join("\n");
        text.push('\n');
        Ok(text)
    }

    /// Read the text [`KingdomTables::to_text`] writes.
    pub fn parse(text: &str) -> Result<KingdomTables, String> {
        let mut t = KingdomTables::default();
        let mut named = false;
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            let bad = || format!("our software has a bug: line {} of the kingdom tables is not a row: {line:?}", n + 1);
            match (f[0], f.len()) {
                ("kingdom", 2) if named => return Err(format!("our software has a bug: line {} of the kingdom tables names a second kingdom: {line:?}", n + 1)),
                ("kingdom", 2) => {
                    t.kingdom = f[1].to_string();
                    named = true;
                }
                ("symbol", 6) => t.symbols.push(SymbolRow {
                    symbol: number("symbol id", f[1])?,
                    symbol_name: f[2].to_string(),
                    alias: f[3].to_string(),
                    semantic_id: f[4].to_string(),
                    spelling: f[5].to_string(),
                }),
                ("data_type", 4..=5) => {
                    let kind = match (f[3], f.len()) {
                        ("scalar", 4) => DataKind::Scalar,
                        ("array", 5) => DataKind::Array { element: ty_of(f[4])? },
                        ("tuple", 5) => DataKind::Tuple { components: tys_of(f[4])? },
                        _ => return Err(bad()),
                    };
                    t.data_types.push(DataType { id: ty_of(f[1])?, name: f[2].to_string(), kind, constant: None });
                }
                ("data_type", 6) if f[3] == "function" => {
                    let kind = DataKind::Function { parameters: tys_of(f[4])?, result: ty_of(f[5])? };
                    t.data_types.push(DataType { id: ty_of(f[1])?, name: f[2].to_string(), kind, constant: None });
                }
                ("constant", 3) => {
                    let id = ty_of(f[1])?;
                    let props = parse_property_list(f[2])?;
                    t.data_types.iter_mut().rev().find(|d| d.id == id && d.constant.is_none()).ok_or_else(bad)?.constant = Some(props);
                }
                ("property_type", 5) => t.property_types.push(PropertyType {
                    id: number("property type id", f[1])?,
                    data_type: ty_of(f[2])?,
                    name: f[3].to_string(),
                    kind: PropertyKind::from_name(f[4]).ok_or_else(bad)?,
                }),
                ("argument", 5) => t.arguments.push(Argument {
                    symbol: number("symbol id", f[1])?,
                    position: number("position", f[2])?,
                    direction: match f[3] {
                        "in" => Direction::In,
                        "out" => Direction::Out,
                        _ => return Err(bad()),
                    },
                    data_type: ty_of(f[4])?,
                }),
                ("rule", 4) => t.property_rules.push(PropertyRule { symbol: number("symbol id", f[1])?, property: f[2].to_string(), expr: PropertyExpr::parse(f[3])? }),
                ("requirement", 6) => t.requirements.push(Requirement {
                    symbol: number("symbol id", f[1])?,
                    subject: Subject::parse(f[2])?,
                    relation: Relation::from_symbol(f[3]).ok_or_else(bad)?,
                    against: PropertyExpr::parse(f[4])?,
                    adapter: f[5].to_string(),
                }),
                ("binding", 5) => {
                    let properties = parse_property_list(f[4])?;
                    t.bindings.push(Binding { symbol: number("symbol id", f[1])?, position: number("position", f[2])?, data_type: ty_of(f[3])?, properties });
                }
                ("adapter", 4) => t.adapters.push(Adapter {
                    name: f[1].to_string(),
                    heavy: match f[2] {
                        "heavy" => true,
                        "light" => false,
                        _ => return Err(bad()),
                    },
                    runs_at: AdapterSite::from_name(f[3]).ok_or_else(bad)?,
                }),
                ("trip", 3) => t.trip_arguments.push(TripArgument { symbol: number("symbol id", f[1])?, position: number("position", f[2])? }),
                _ => return Err(bad()),
            }
        }
        Ok(t)
    }
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
        // Both table-defined ranges round-trip in full and do not overlap.
        assert_eq!(Ty::TSR_CODE_BASE, 4352);
        for i in 0..(Ty::TSR_CODE_BASE - Ty::TABLE_CODE_BASE) as u16 {
            assert_eq!(Ty::from_code(Ty::Wgsl(i).code()), Some(Ty::Wgsl(i)));
        }
        for i in 0..=u16::MAX {
            assert_eq!(Ty::from_code(Ty::Tsr(i).code()), Some(Ty::Tsr(i)));
        }
        assert_eq!(Ty::Tsr(0).code(), 4352);
        assert_eq!(Ty::from_code(4351), Some(Ty::Wgsl(4095)));
        assert_eq!(Ty::from_code(4352), Some(Ty::Tsr(0)));
        // The 1-D Futhark family sits after the TSR family, and the two do
        // not overlap.
        assert_eq!(Ty::FUTHARK_1D_CODE_BASE, Ty::TSR_CODE_BASE + 65536);
        for i in 0..=u16::MAX {
            assert_eq!(Ty::from_code(Ty::Futhark1d(i).code()), Some(Ty::Futhark1d(i)));
        }
        assert_eq!(Ty::from_code(Ty::FUTHARK_1D_CODE_BASE), Some(Ty::Futhark1d(0)));
        assert!(Ty::Futhark1d(0) > Ty::Tsr(u16::MAX), "ordered after the TSR family");
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
        // The depth ladder is unmoved by the numbering.
        assert_eq!((Ty::F.code(), Ty::T1.code(), Ty::T2.code()), (2, 6, 7));
    }

    #[test]
    fn a_kingdom_with_too_many_duals_fails_loudly_in_every_build() {
        assert!(Ty::check_dual_count("k", Ty::MAX_WGSL_DUALS, Ty::MAX_WGSL_DUALS).is_ok());
        let e = Ty::check_dual_count("k", Ty::MAX_WGSL_DUALS + 1, Ty::MAX_WGSL_DUALS).unwrap_err();
        assert_eq!(e, "our software has a bug: kingdom k has 4097 duals, more than its 4096 codes");
        assert_eq!(Ty::TSR_CODE_BASE - Ty::TABLE_CODE_BASE, Ty::MAX_WGSL_DUALS as u32);
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
        let sigs: Vec<(SymbolRow, Vec<Ty>, Ty)> = rows.iter().zip(ordered).map(|(s, (ins, out))| (SymbolRow::of(s, ""), ins, out)).collect();
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
        derives_todays_arity(&typed_depth_table(), TYPED_SR, typed_depth_signatures().into_iter().map(|(_, i, o)| (i, o)).collect());
        derives_todays_arity(&regex_table(), REGEX, regex_rows().iter().map(|(_, _, ins, out)| (ins.to_vec(), *out)).collect());
        use crate::tsr_kingdom::{TsrKingdom, TsrVariant, TSR};
        for v in [TsrVariant::Single, TsrVariant::Order, TsrVariant::Depth, TsrVariant::OrderDepth] {
            let k = TsrKingdom::load(v, 3).expect("TSR tables load");
            derives_todays_arity(&k.symbol_table(), TSR, k.rows.iter().map(|r| (r.inputs.clone(), r.output)).collect());
        }
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
                let row = SymbolRow { symbol: i as i64, symbol_name: name.clone(), alias: name.clone(), semantic_id: name.clone(), spelling: String::new() };
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
        use crate::kingdom_contract::{regex_relations, sr_relations, tsr_relations};
        the_contract_is_unchanged_through_the_tables(&sr_relations());
        the_contract_is_unchanged_through_the_tables(&regex_relations());
        for v in [crate::tsr_kingdom::TsrVariant::Single, crate::tsr_kingdom::TsrVariant::Order, crate::tsr_kingdom::TsrVariant::Depth, crate::tsr_kingdom::TsrVariant::OrderDepth] {
            the_contract_is_unchanged_through_the_tables(&tsr_relations(v, 3).expect("TSR tables load"));
        }
        #[cfg(feature = "wgsl")]
        {
            the_contract_is_unchanged_through_the_tables(&crate::kingdom_contract::wgsl_relations());
            the_contract_is_unchanged_through_the_tables(&crate::kingdom_contract::safewgsl_relations());
            let t = crate::wgsl::true_kingdom::TrueKingdom::load(&crate::wgsl::true_kingdom::default_dir()).expect("true WGSL table");
            the_contract_is_unchanged_through_the_tables(&t.relations());
        }
        // The Futhark kingdoms on master 5: the matrix kingdom, the deprecated 1-D one, and every signature view.
        the_contract_is_unchanged_through_the_tables(&crate::futhark_table::FutharkKingdom::load().expect("Futhark kingdom").relations());
        the_contract_is_unchanged_through_the_tables(&crate::futhark_table::FutharkKingdom::load_1d().expect("1-D kingdom").relations());
        use crate::futhark_sig::{HISTOGRAM, PATHFINDER, REDUCE, SCAN};
        for inst in [REDUCE, SCAN, HISTOGRAM, PATHFINDER] {
            the_contract_is_unchanged_through_the_tables(&crate::futhark_sig::load(&inst).expect("view loads").relations());
        }
    }

    /// The k-means view (14,359 types on master 5), the largest kingdom we have, on its own so it runs beside the rest.
    #[test]
    fn the_kmeans_contract_is_unchanged_through_the_tables() {
        let t0 = std::time::Instant::now();
        let r = crate::futhark_sig::load(&crate::futhark_sig::KMEANS).expect("view loads").relations();
        eprintln!("kmeans view: {} symbols, loaded in {:.1} s", r.names.len(), t0.elapsed().as_secs_f64());
        the_contract_is_unchanged_through_the_tables(&r);
        eprintln!("kmeans through the tables: {:.1} s", t0.elapsed().as_secs_f64());
    }

    /// The histogram's tables (design section 6) with properties, a requirement, a binding, a tuple, a function type
    /// and a symbolic sum: the arguments keep their order across data types, the arity is their count, the check
    /// passes, and the text reads back to the same tables.
    fn histogram_tables() -> KingdomTables {
        let (i32_, arr, i64_, pair, fun) = (Ty::Futhark(0), Ty::Futhark(1), Ty::Futhark(2), Ty::Futhark(3), Ty::Futhark(4));
        let sym = |symbol: i64, name: &str, spelling: &str| SymbolRow {
            symbol,
            symbol_name: name.to_string(),
            alias: name.to_string(),
            semantic_id: name.to_string(),
            spelling: spelling.to_string(),
        };
        let mut t = KingdomTables {
            kingdom: "histogram test".to_string(),
            symbols: vec![
                sym(1, "hist.add", "hist (+) 0 {0} {1} {2}"),
                sym(2, "tabulate", "tabulate {0} (\\i -> {1})"),
                sym(3, "get", "{0}[{1}]"),
                sym(4, "concat", "{0} ++ {1}"),
                sym(-1, "var.[]i32", ""),
                sym(-2, "var.i64", ""),
            ],
            data_types: vec![
                DataType { id: i32_, name: "i32".into(), kind: DataKind::Scalar, constant: Some(vec![]) },
                DataType { id: arr, name: "[]i32".into(), kind: DataKind::Array { element: i32_ }, constant: Some(vec![("len".into(), PropertyExpr::Literal(1))]) },
                DataType { id: i64_, name: "i64".into(), kind: DataKind::Scalar, constant: Some(vec![("bound".into(), PropertyExpr::Literal(1))]) },
                DataType { id: pair, name: "(i32, i32)".into(), kind: DataKind::Tuple { components: vec![i32_, i32_] }, constant: Some(vec![]) },
                DataType { id: fun, name: "i32 -> i32".into(), kind: DataKind::Function { parameters: vec![i32_], result: i32_ }, constant: None },
            ],
            property_types: vec![
                PropertyType { id: 0, data_type: arr, name: "len".into(), kind: PropertyKind::Size },
                PropertyType { id: 1, data_type: i64_, name: "value".into(), kind: PropertyKind::Size },
                PropertyType { id: 2, data_type: i64_, name: "bound".into(), kind: PropertyKind::Bound },
            ],
            ..KingdomTables::default()
        };
        let mut arg = |symbol: i64, ins: &[Ty], out: Ty| {
            for (i, ty) in ins.iter().enumerate() {
                t.arguments.push(Argument { symbol, position: i as u32, direction: Direction::In, data_type: *ty });
            }
            t.arguments.push(Argument { symbol, position: 0, direction: Direction::Out, data_type: out });
        };
        arg(1, &[i64_, arr, arr], arr);
        arg(2, &[i64_, i32_], arr);
        arg(3, &[arr, i64_], i32_);
        arg(4, &[arr, arr], arr);
        arg(-1, &[], arr);
        arg(-2, &[], i64_);
        let input = |position: u32, property: &str| PropertyExpr::Input { position, property: property.to_string() };
        t.property_rules = vec![
            PropertyRule { symbol: 1, property: "len".into(), expr: input(0, "value") },
            PropertyRule { symbol: 2, property: "len".into(), expr: input(0, "value") },
            PropertyRule { symbol: 4, property: "len".into(), expr: PropertyExpr::sum(vec![input(0, "len"), input(1, "len")]) },
            PropertyRule { symbol: -1, property: "len".into(), expr: PropertyExpr::min(vec![PropertyExpr::Size("n".into()), PropertyExpr::Size("k".into())]) },
        ];
        let on = |position: u32, property: &str| Subject::Input { position, property: property.to_string() };
        t.requirements = vec![
            Requirement { symbol: 1, subject: on(1, "len"), relation: Relation::Eq, against: input(2, "len"), adapter: "fit.pair".into() },
            Requirement { symbol: 3, subject: on(1, "bound"), relation: Relation::Le, against: input(0, "len"), adapter: "clampix".into() },
            Requirement { symbol: 2, subject: Subject::Trips, relation: Relation::Lt, against: PropertyExpr::Literal(2), adapter: "subst".into() },
            Requirement { symbol: 1, subject: Subject::Trips, relation: Relation::Lt, against: PropertyExpr::Literal(2), adapter: "subst.const".into() },
            Requirement { symbol: 4, subject: Subject::Trips, relation: Relation::Lt, against: PropertyExpr::Literal(2), adapter: "subst".into() },
        ];
        t.bindings = vec![Binding { symbol: 2, position: 1, data_type: i64_, properties: vec![("bound".into(), input(0, "value"))] }];
        t.adapters = vec![
            Adapter { name: "fit.pair".into(), heavy: true, runs_at: AdapterSite::Node },
            Adapter { name: "clampix".into(), heavy: false, runs_at: AdapterSite::Node },
            Adapter { name: "subst".into(), heavy: false, runs_at: AdapterSite::Node },
            Adapter { name: "subst.const".into(), heavy: false, runs_at: AdapterSite::Node },
            Adapter { name: "fit.body".into(), heavy: true, runs_at: AdapterSite::Body },
            Adapter { name: "fit.root".into(), heavy: true, runs_at: AdapterSite::Root },
        ];
        t.trip_arguments = vec![TripArgument { symbol: 2, position: 1 }];
        t
    }

    #[test]
    fn a_symbol_with_ordered_arguments_of_mixed_data_types() {
        let t = histogram_tables();
        t.check().unwrap_or_else(|d| panic!("{d:#?}"));
        assert!(t.has_properties());
        let (i32_, arr, i64_) = (Ty::Futhark(0), Ty::Futhark(1), Ty::Futhark(2));
        assert_eq!(t.inputs(1), vec![i64_, arr, arr], "hist.add keeps its order: the size, the indices, the values");
        assert_eq!(t.inputs(2), vec![i64_, i32_]);
        assert_eq!(t.inputs(3), vec![arr, i64_], "get: the array, then the index");
        let a = t.arity(1);
        assert_eq!(a.inputs, BTreeMap::from([(arr, 2), (i64_, 1)]));
        assert_eq!(a.outputs, BTreeMap::from([(arr, 1)]));
        assert_eq!(a.total_in(), 3);
        assert_eq!(t.arity(-1), Arity { inputs: BTreeMap::new(), outputs: BTreeMap::from([(arr, 1)]) });
        // The regex kingdom's `{n}` takes (Pattern, I), which the many-hot map reads back as (I, Pattern); the
        // Argument rows keep the order.
        let rows: Vec<(SymbolRow, Vec<Ty>, Ty)> = regex_table().kingdom(REGEX).iter().zip(regex_rows()).map(|(s, (_, _, ins, out))| (SymbolRow::of(s, ""), ins.to_vec(), *out)).collect();
        let regex = KingdomTables::from_signatures(REGEX, rows);
        let rep = regex.symbols.iter().find(|s| s.semantic_id == "regex_rep_n").expect("{n}").symbol;
        assert_eq!(regex.inputs(rep), vec![Ty::Pattern, Ty::I]);
        assert_eq!(regex.arity(rep).inputs.keys().copied().collect::<Vec<_>>(), vec![Ty::I, Ty::Pattern], "the many-hot map loses the order");
    }

    #[test]
    fn the_tables_round_trip_through_text() {
        let hist = histogram_tables();
        let text = hist.to_text().expect("text");
        let back = KingdomTables::parse(&text).expect("parse");
        assert_eq!(back, hist);
        assert_eq!(back.to_text().expect("text"), text, "the text is a fixed point");
        let regex_rows_: Vec<(SymbolRow, Vec<Ty>, Ty)> =
            regex_table().kingdom(REGEX).iter().zip(regex_rows()).map(|(s, (_, alias, ins, out))| (SymbolRow::of(s, alias), ins.to_vec(), *out)).collect();
        for t in [KingdomTables::from_signatures(REGEX, regex_rows_), KingdomTables::from_signatures(TYPED_SR, typed_depth_signatures())] {
            let back = KingdomTables::parse(&t.to_text().expect("text")).expect("parse");
            assert_eq!(back, t, "{}", t.kingdom);
            assert_eq!(back.symbol_table().kingdom(&t.kingdom), t.symbol_table().kingdom(&t.kingdom));
        }
        let mut tab = hist.clone();
        tab.symbols[0].spelling = "a\tb".into();
        assert!(tab.to_text().expect_err("a tab").starts_with("our software has a bug:"));
        assert!(KingdomTables::parse("symbol\t1\tx").expect_err("short row").starts_with("our software has a bug:"));
        // Tables that fail the check are not written (a nested sum would be written flat and read back flat).
        let mut unchecked = hist.clone();
        unchecked.property_rules[2].expr = PropertyExpr::Sum(vec![PropertyExpr::Sum(vec![PropertyExpr::Size("k".into()), PropertyExpr::Size("n".into())]), PropertyExpr::Literal(1)]);
        let e = unchecked.to_text().expect_err("an unchecked table");
        assert!(e.starts_with("our software has a bug: kingdom histogram test: the tables fail their check"), "{e}");
        // A second kingdom line is an error, not a silent rename.
        let twice = format!("{text}kingdom\tother\n");
        assert!(KingdomTables::parse(&twice).expect_err("two kingdoms").contains("names a second kingdom"));
    }

    #[test]
    fn property_expressions_round_trip() {
        // Texts already in their one spelling read back to themselves.
        for text in [
            "n", "k_2", "3", "-1", "?", "in0.len", "in12.bound", "in0.len + in1.len", "n + 1 + ?", "min(k, n)", "1 + min(k, n)",
            "min(?, in1.len, 1 + in0.len)", "in0.value + min(d, k, n)", "min",
        ] {
            let e = PropertyExpr::parse(text).unwrap_or_else(|err| panic!("{text}: {err}"));
            assert_eq!(e.to_string(), text);
            assert_eq!(e, e.normal(), "{text}");
        }
        // Any other order reads as the one spelling (G1 review finding 3): a + b = b + a, min(n, k) = min(k, n).
        for (text, one) in [
            ("in0.len+n", "n + in0.len"),
            ("in1.len + in0.len", "in0.len + in1.len"),
            ("min(n, k)", "min(k, n)"),
            ("min(n, k) + 1", "1 + min(k, n)"),
            ("min(in0.len + 1, in1.len, ?)", "min(?, in1.len, 1 + in0.len)"),
            ("min(min(n, k), d) + in0.value", "in0.value + min(d, k, n)"),
            ("min(n, n)", "n"),
            ("min(n, min(k, n))", "min(k, n)"),
            ("min( n ,k )", "min(k, n)"),
        ] {
            assert_eq!(PropertyExpr::parse(text).expect(text).to_string(), one, "{text}");
        }
        assert_eq!(PropertyExpr::parse("a + b").expect("a + b"), PropertyExpr::parse("b + a").expect("b + a"));
        assert_eq!(PropertyExpr::sum(vec![PropertyExpr::Size("n".into())]), PropertyExpr::Size("n".into()));
        let nested = PropertyExpr::sum(vec![PropertyExpr::Sum(vec![PropertyExpr::Literal(1), PropertyExpr::Unknown]), PropertyExpr::Size("k".into())]);
        assert_eq!(nested, PropertyExpr::Sum(vec![PropertyExpr::Size("k".into()), PropertyExpr::Literal(1), PropertyExpr::Unknown]));
        for bad in ["", "in.len", "inx.len", "in0.", "a b", "n + ", "1x", "min(n)", "min(n, k", "min(n, k))", "(n)"] {
            assert!(PropertyExpr::parse(bad).expect_err(bad).starts_with("our software has a bug:"), "{bad}");
        }
    }

    /// The relations of master 6 README 5.1 on the one spelling of each side.
    #[test]
    fn the_relations_compare_one_spelling() {
        let p = |t: &str| PropertyExpr::parse(t).expect(t);
        let holds = |r: Relation, a: &str, b: &str| r.holds(&p(a), &p(b));
        assert!(holds(Relation::Eq, "n", "n") && !holds(Relation::Eq, "n", "k"));
        assert!(holds(Relation::Eq, "min(n, k)", "min(k, n)") && holds(Relation::Eq, "min(n, n)", "n"));
        assert!(holds(Relation::Eq, "n + k", "k + n") && !holds(Relation::Eq, "n + n", "n"));
        assert!(!holds(Relation::Eq, "?", "?") && !holds(Relation::Le, "?", "n") && !holds(Relation::Le, "1", "?") && !holds(Relation::Lt, "0", "min(1, ?)"));
        // An index of bound 1 (`0i64`) is valid for every length; a size's bound only for that size or a longer one.
        for len in ["n", "k", "1", "7", "min(k, n)", "min(d, k, n)", "n + k", "n + 0", "1 + min(k, n)"] {
            assert!(holds(Relation::Le, "1", len), "1 <= {len}");
            assert!(holds(Relation::Le, len, len), "{len} <= {len}");
        }
        assert!(!holds(Relation::Le, "1", "0") && !holds(Relation::Le, "1", "n + -1") && !holds(Relation::Le, "2", "n"));
        assert!(holds(Relation::Le, "min(k, n)", "n") && holds(Relation::Le, "min(k, n)", "k") && !holds(Relation::Le, "n", "min(k, n)"));
        assert!(holds(Relation::Le, "min(d, k, n)", "min(k, n)") && !holds(Relation::Le, "min(k, n)", "min(d, k, n)"));
        assert!(holds(Relation::Le, "3", "4") && !holds(Relation::Le, "4", "3") && !holds(Relation::Le, "k", "n"));
        assert!(holds(Relation::Lt, "1", "2") && !holds(Relation::Lt, "2", "2") && !holds(Relation::Lt, "n", "3"));
    }

    /// Finding 5 of the G1 review: the typed-depth kingdom's table as literals, not recounted from the signatures that
    /// build it. 26 operations (no terminal: the SR kingdom's leaves come from elsewhere): 6 binary absorbing ones of
    /// 6 rows, 15 raising unary ones of 2 rows (none takes `T2`), 5 absorbing unary ones of 3 rows: 81 rows.
    #[test]
    fn the_typed_depth_table_as_literals() {
        let t = typed_depth_table();
        let rows = t.kingdom(TYPED_SR);
        assert_eq!(rows.len(), 81);
        let arity = |ins: &[(Ty, u32)], out: Ty| Arity { inputs: ins.iter().copied().collect(), outputs: BTreeMap::from([(out, 1)]) };
        let (f, t1, t2) = (Ty::F, Ty::T1, Ty::T2);
        for (symbol, name, want) in [
            (1, "add", arity(&[(f, 2)], f)),
            (2, "add", arity(&[(f, 1), (t1, 1)], t1)),
            (3, "add", arity(&[(f, 1), (t2, 1)], t2)),
            (5, "add", arity(&[(t1, 1), (t2, 1)], t2)),
            (6, "add", arity(&[(t2, 2)], t2)),
            (25, "neg", arity(&[(f, 1)], f)),
            (26, "neg", arity(&[(t1, 1)], t1)),
            (27, "neg", arity(&[(t2, 1)], t2)),
            (28, "sin", arity(&[(f, 1)], t1)),
            (29, "sin", arity(&[(t1, 1)], t2)),
            (44, "pow2", arity(&[(f, 1)], f)),
            (46, "pow2", arity(&[(t2, 1)], t2)),
            (50, "pow", arity(&[(f, 2)], f)),
            (68, "protected_div", arity(&[(f, 2)], f)),
            (80, "protected_acos", arity(&[(f, 1)], t1)),
            (81, "protected_acos", arity(&[(t1, 1)], t2)),
        ] {
            let s = rows.iter().find(|s| s.symbol == symbol).unwrap_or_else(|| panic!("symbol {symbol}"));
            assert_eq!((s.semantic_id.as_str(), &s.arity), (name, &want), "symbol {symbol}");
        }
        let raising_rows = rows.iter().filter(|s| DEPTH_RAISING.contains(&s.semantic_id.as_str())).count();
        assert_eq!(raising_rows, 30);
        assert!(rows.iter().filter(|s| DEPTH_RAISING.contains(&s.semantic_id.as_str())).all(|s| !s.arity.inputs.contains_key(&t2)), "no raising row takes T2");
    }

    /// Finding 4 of the G1 review: every branch of the check, each provoked alone on the histogram tables and named by
    /// its message, with the number of defects it gives.
    #[test]
    fn every_branch_of_the_table_check() {
        let (i32_, arr, i64_) = (Ty::Futhark(0), Ty::Futhark(1), Ty::Futhark(2));
        let input = |position: u32, property: &str| PropertyExpr::Input { position, property: property.to_string() };
        let on = |position: u32, property: &str| Subject::Input { position, property: property.to_string() };
        let size = |n: &str| PropertyExpr::Size(n.to_string());
        let req = |symbol: i64, subject: Subject, relation: Relation, against: PropertyExpr, adapter: &str| Requirement { symbol, subject, relation, against, adapter: adapter.to_string() };
        let flag = PropertyType { id: 10, data_type: arr, name: "sorted".into(), kind: PropertyKind::Flag };
        type Edit = Box<dyn Fn(&mut KingdomTables)>;
        let cases: Vec<(&str, usize, Edit)> = vec![
            ("symbol 3 (get) is listed twice", 1, Box::new(|t| t.symbols.push(t.symbols[2].clone()))),
            ("data type Futhark(0) is listed twice", 1, Box::new(move |t| t.data_types.push(DataType { id: i32_, name: "int".into(), kind: DataKind::Scalar, constant: None }))),
            ("two data types are named \"i32\"", 1, Box::new(|t| t.data_types.push(DataType { id: Ty::Futhark(5), name: "i32".into(), kind: DataKind::Scalar, constant: None }))),
            (
                "data type []x names Futhark(9), which is not a data type",
                1,
                Box::new(|t| t.data_types.push(DataType { id: Ty::Futhark(6), name: "[]x".into(), kind: DataKind::Array { element: Ty::Futhark(9) }, constant: None })),
            ),
            ("property type 0 is listed twice", 1, Box::new(move |t| t.property_types.push(PropertyType { id: 0, data_type: i32_, name: "w".into(), kind: PropertyKind::Size }))),
            ("property len is on Futhark(9), which is not a data type", 1, Box::new(|t| t.property_types.push(PropertyType { id: 7, data_type: Ty::Futhark(9), name: "len".into(), kind: PropertyKind::Size }))),
            ("property name \"le n\" is not an identifier", 1, Box::new(move |t| t.property_types.push(PropertyType { id: 8, data_type: i32_, name: "le n".into(), kind: PropertyKind::Size }))),
            ("data type Futhark(1) has two properties named len", 1, Box::new(move |t| t.property_types.push(PropertyType { id: 9, data_type: arr, name: "len".into(), kind: PropertyKind::Size }))),
            ("an argument names symbol 99, which is not a symbol", 1, Box::new(move |t| t.arguments.push(Argument { symbol: 99, position: 0, direction: Direction::Out, data_type: arr }))),
            (
                "symbol 3's output is at position 1, not 0",
                1,
                Box::new(|t| t.arguments.iter_mut().find(|a| a.symbol == 3 && a.direction == Direction::Out).expect("out").position = 1),
            ),
            ("symbol 3 has 0 outputs, not one", 1, Box::new(|t| t.arguments.retain(|a| !(a.symbol == 3 && a.direction == Direction::Out)))),
            (
                "symbol 50 (lonely) has no output",
                1,
                Box::new(|t| t.symbols.push(SymbolRow { symbol: 50, symbol_name: "lonely".into(), alias: String::new(), semantic_id: String::new(), spelling: String::new() })),
            ),
            ("a property rule names symbol 77, which has no output", 1, Box::new(|t| t.property_rules.push(PropertyRule { symbol: 77, property: "len".into(), expr: PropertyExpr::Literal(1) }))),
            (
                "requirement on in1.len reads in0.len, which Futhark(2) does not carry",
                1,
                Box::new(move |t| t.requirements.push(req(1, on(1, "len"), Relation::Eq, input(0, "len"), "fit.pair"))),
            ),
            ("with its terms out of order (one spelling: in0.len + in1.len)", 1, Box::new(move |t| t.property_rules[2].expr = PropertyExpr::Sum(vec![input(1, "len"), input(0, "len")]))),
            ("has the min min(n, k) with a min inside or its arguments out of order", 1, Box::new(move |t| t.property_rules[3].expr = PropertyExpr::Min(vec![size("n"), size("k")]))),
            ("has the min min(k, k) with", 1, Box::new(move |t| t.property_rules[3].expr = PropertyExpr::Min(vec![size("k"), size("k")]))),
            ("data type []i32's constant sets bound, which it does not carry", 1, Box::new(|t| t.data_types[1].constant = Some(vec![("bound".into(), PropertyExpr::Literal(1))]))),
            ("data type []i32's constant sets len twice", 1, Box::new(|t| t.data_types[1].constant = Some(vec![("len".into(), PropertyExpr::Literal(1)), ("len".into(), PropertyExpr::Literal(2))]))),
            ("data type []i32's constant sets len = in0.len, which reads an input", 1, Box::new(move |t| t.data_types[1].constant = Some(vec![("len".into(), input(0, "len"))]))),
            ("adapter name \"fit pair\" is empty or holds a space", 1, Box::new(|t| t.adapters.push(Adapter { name: "fit pair".into(), heavy: true, runs_at: AdapterSite::Node }))),
            ("adapter clampix is listed twice", 1, Box::new(|t| t.adapters.push(Adapter { name: "clampix".into(), heavy: false, runs_at: AdapterSite::Node }))),
            ("symbol 1's trip argument 9 does not exist", 1, Box::new(|t| t.trip_arguments.push(TripArgument { symbol: 1, position: 9 }))),
            ("symbol 2 has two trip arguments", 1, Box::new(|t| t.trip_arguments.push(TripArgument { symbol: 2, position: 0 }))),
            (
                "compares against n + in0.sorted, which mixes kinds (size or bound, flag)",
                2,
                Box::new(move |t| {
                    t.property_types.push(flag.clone());
                    t.requirements.push(req(4, on(0, "len"), Relation::Eq, PropertyExpr::sum(vec![size("n"), input(0, "sorted")]), "fit.pair"));
                }),
            ),
            ("symbol 4's requirement names in0.bound, which Futhark(1) does not carry", 1, Box::new(move |t| t.requirements.push(req(4, on(0, "bound"), Relation::Le, size("n"), "clampix")))),
            ("symbol 3's requirement in1.bound = in0.len uses =, which a bound property does not take", 1, Box::new(move |t| t.requirements[1].relation = Relation::Eq)),
            (
                "compares the size property in0.len with in0.sorted, which is not of a comparable kind",
                1,
                Box::new(move |t| {
                    t.property_types.push(PropertyType { id: 10, data_type: arr, name: "sorted".into(), kind: PropertyKind::Flag });
                    t.requirements.push(req(4, on(0, "len"), Relation::Eq, input(0, "sorted"), "fit.pair"));
                }),
            ),
            ("symbol 2's requirement on env.trips is <= 2, not < a literal", 1, Box::new(|t| t.requirements[2].relation = Relation::Le)),
            ("symbol 2's requirement on env.trips is < n, not < a literal", 1, Box::new(move |t| t.requirements[2].against = size("n"))),
            ("symbol 4's requirement on in0.len uses <, which only env.trips takes", 2, Box::new(move |t| t.requirements.push(req(4, on(0, "len"), Relation::Lt, size("n"), "fit.pair")))),
            ("names adapter \"nope\", which is not an adapter", 1, Box::new(move |t| t.requirements.push(req(4, on(0, "len"), Relation::Eq, size("n"), "nope")))),
            ("names adapter fit.root, which runs at the root", 1, Box::new(move |t| t.requirements.push(req(4, on(0, "len"), Relation::Eq, size("n"), "fit.root")))),
            (
                "names adapter fit.body, which runs at the body, but the symbol has no trip argument",
                1,
                Box::new(move |t| t.requirements.push(req(4, on(0, "len"), Relation::Eq, size("n"), "fit.body"))),
            ),
            ("symbol 2's binding is of Futhark(9), which is not a data type", 1, Box::new(|t| t.bindings.push(Binding { symbol: 2, position: 1, data_type: Ty::Futhark(9), properties: vec![] }))),
            ("symbol 2's binding sets len, which Futhark(2) does not carry", 1, Box::new(move |t| t.bindings.push(Binding { symbol: 2, position: 1, data_type: i64_, properties: vec![("len".into(), size("n"))] }))),
            (
                "symbol 2's binding sets bound twice",
                1,
                Box::new(move |t| t.bindings.push(Binding { symbol: 2, position: 1, data_type: i64_, properties: vec![("bound".into(), size("n")), ("bound".into(), size("k"))] })),
            ),
            (
                "symbol 4's binding of argument 0 reads in0.len, its own argument",
                1,
                Box::new(move |t| t.bindings.push(Binding { symbol: 4, position: 0, data_type: i64_, properties: vec![("bound".into(), input(0, "len"))] })),
            ),
            (
                "symbol 4's bindings read each other's arguments in a cycle",
                1,
                Box::new(move |t| {
                    t.bindings.push(Binding { symbol: 4, position: 0, data_type: i64_, properties: vec![("bound".into(), input(1, "len"))] });
                    t.bindings.push(Binding { symbol: 4, position: 1, data_type: i64_, properties: vec![("bound".into(), input(0, "len"))] });
                }),
            ),
            // G3 review finding 7: a range and a flag compare with their own kind only, never with a literal.
            (
                "compares the range property in0.span with 1, which is not of a comparable kind",
                1,
                Box::new(move |t| {
                    t.property_types.push(PropertyType { id: 10, data_type: arr, name: "span".into(), kind: PropertyKind::Range });
                    t.requirements.push(req(4, on(0, "span"), Relation::Eq, PropertyExpr::Literal(1), "fit.pair"));
                }),
            ),
            (
                "compares the flag property in0.sorted with 1, which is not of a comparable kind",
                1,
                Box::new(move |t| {
                    t.property_types.push(PropertyType { id: 10, data_type: arr, name: "sorted".into(), kind: PropertyKind::Flag });
                    t.requirements.push(req(4, on(0, "sorted"), Relation::Eq, PropertyExpr::Literal(1), "fit.pair"));
                }),
            ),
            // G3 review finding 2: `env.trips` takes a substitution adapter, and only it does; one a symbol.
            ("symbol 2's requirement on env.trips names adapter fit.pair, which is not a substitution adapter", 1, Box::new(|t| t.requirements[2].adapter = "fit.pair".into())),
            (
                "symbol 4's requirement on in1.len names the substitution adapter subst, which only env.trips takes",
                1,
                Box::new(move |t| t.requirements.push(req(4, on(1, "len"), Relation::Eq, input(0, "len"), "subst"))),
            ),
            ("symbol 2 has two requirements on env.trips", 1, Box::new(move |t| t.requirements.push(req(2, Subject::Trips, Relation::Lt, PropertyExpr::Literal(3), "subst")))),
            ("adapter subst.const is a substitution adapter, so it is light and runs at the node", 1, Box::new(|t| t.adapters[3].heavy = true)),
            // G3 review finding 3: a heavy adapter on a symbol with no work bound escapes the bound.
            (
                "symbol 1's requirement on in1.len names the heavy adapter fit.pair, but the symbol has no requirement on env.trips",
                1,
                Box::new(|t| t.requirements.retain(|q| !(q.symbol == 1 && q.subject == Subject::Trips))),
            ),
            // G3 review finding 6: a substitution by the constant of a data type that has none, found by the tables.
            ("substitutes the constant of []i32, which has none", 2, Box::new(|t| t.data_types[1].constant = None)),
        ];
        histogram_tables().check().unwrap_or_else(|d| panic!("the base tables check: {d:#?}"));
        for (want, count, edit) in &cases {
            let mut t = histogram_tables();
            edit(&mut t);
            let d = t.check().expect_err(want);
            assert!(d.iter().all(|e| e.starts_with("our software has a bug: kingdom histogram test: ")), "{d:#?}");
            assert!(d.iter().any(|e| e.contains(want)), "missing {want:?} in {d:#?}");
            assert_eq!(d.len(), *count, "{want}: {d:#?}");
        }
        // The bound against the length (`get`) and the bound of a literal index pass: a size and a bound both count
        // elements (master 6 README 5.2, G1 review finding 1).
        let mut t = histogram_tables();
        t.requirements.push(req(3, on(1, "bound"), Relation::Le, PropertyExpr::Literal(1), "clampix"));
        t.requirements.push(req(4, on(0, "len"), Relation::Le, PropertyExpr::min(vec![size("n"), input(1, "len")]), "fit.pair"));
        t.check().unwrap_or_else(|d| panic!("{d:#?}"));
    }

    /// G3 review finding 5: the entry is checked against the tables, each branch provoked alone, by message and count.
    #[test]
    fn every_branch_of_the_entry_check() {
        let (arr, i64_) = (Ty::Futhark(1), Ty::Futhark(2));
        let t = histogram_tables();
        let size = |n: &str| PropertyExpr::Size(n.to_string());
        let len_n = vec![("len".to_string(), size("n"))];
        let value_k = vec![("value".to_string(), size("k"))];
        type Vars<'a> = Vec<EntryParameter<'a>>;
        let good_vars: Vars = vec![("is", arr, &len_n), ("k", i64_, &value_k)];
        let k_ = size("k");
        let good_reqs = vec![("len", Relation::Eq, &k_, "fit.root")];
        assert_eq!(t.entry_defects(arr, &good_vars, &good_reqs), Vec::<String>::new());
        let unknown = PropertyExpr::Unknown;
        let in0 = PropertyExpr::Input { position: 0, property: "len".into() };
        let one = PropertyExpr::Literal(1);
        let len_twice = vec![("len".to_string(), size("n")), ("len".to_string(), size("k"))];
        let len_in = vec![("len".to_string(), in0.clone())];
        let bound = vec![("bound".to_string(), size("n"))];
        let nested = vec![("len".to_string(), PropertyExpr::Min(vec![size("n"), size("k")]))];
        let cases: Vec<(&str, usize, Ty, Vars, Vec<EntryRequirement>)> = vec![
            ("the entry returns Futhark(9), which is not a data type", 2, Ty::Futhark(9), good_vars.clone(), good_reqs.clone()),
            ("the entry's parameter name \"a b\" is not an identifier", 1, arr, vec![("a b", arr, &len_n)], vec![]),
            ("the entry has two parameters named is", 1, arr, vec![("is", arr, &len_n), ("is", arr, &len_n)], vec![]),
            ("the entry's parameter x is of Futhark(9), which is not a data type", 2, arr, vec![("x", Ty::Futhark(9), &len_n)], vec![]),
            ("the entry's parameter is sets bound, which Futhark(1) does not carry", 1, arr, vec![("is", arr, &bound)], vec![]),
            ("the entry's parameter is sets len twice", 1, arr, vec![("is", arr, &len_twice)], vec![]),
            ("the entry's parameter is sets len = in0.len, which reads an input", 1, arr, vec![("is", arr, &len_in)], vec![]),
            ("the entry's parameter is's len has the min min(n, k)", 1, arr, vec![("is", arr, &nested)], vec![]),
            ("names out.lenn, which Futhark(1) does not carry", 1, arr, vec![], vec![("lenn", Relation::Eq, &k_, "fit.root")]),
            ("the entry's requirement out.len < k uses <", 1, arr, vec![], vec![("len", Relation::Lt, &k_, "fit.root")]),
            ("reads an input or ?, which the entry's result never meets", 1, arr, vec![], vec![("len", Relation::Eq, &unknown, "fit.root")]),
            ("reads an input or ?, which the entry's result never meets", 1, arr, vec![], vec![("len", Relation::Eq, &in0, "fit.root")]),
            ("names adapter \"nope\", which is not an adapter", 1, arr, vec![], vec![("len", Relation::Eq, &k_, "nope")]),
            ("names adapter fit.pair, which does not run at the root", 1, arr, vec![], vec![("len", Relation::Eq, &k_, "fit.pair")]),
        ];
        for (want, count, output, vars, reqs) in &cases {
            let d = t.entry_defects(*output, vars, reqs);
            assert!(d.iter().all(|e| e.starts_with("our software has a bug: kingdom histogram test: ")), "{d:#?}");
            assert!(d.iter().any(|e| e.contains(want)), "missing {want:?} in {d:#?}");
            assert_eq!(d.len(), *count, "{want}: {d:#?}");
        }
        // A literal length is comparable (a size counts elements), a range against a literal is not.
        assert_eq!(t.entry_defects(arr, &[], &[("len", Relation::Eq, &one, "fit.root")]), Vec::<String>::new());
        let mut ranged = histogram_tables();
        ranged.property_types.push(PropertyType { id: 10, data_type: arr, name: "span".into(), kind: PropertyKind::Range });
        let d = ranged.entry_defects(arr, &[], &[("span", Relation::Eq, &one, "fit.root")]);
        assert!(d.len() == 1 && d[0].contains("compares the range property with 1, which is not of a comparable kind"), "{d:#?}");
    }

    #[test]
    fn the_table_check_lists_every_defect() {
        let mut t = histogram_tables();
        let arr = Ty::Futhark(1);
        // A gap in hist.add's inputs (position 2 becomes 3), a second output on get, a rule on a property the
        // output does not carry, a requirement and a binding on inputs that do not exist, an expression reading an
        // input that does not exist, a data type that is not in the table, a sum of one term.
        t.arguments.iter_mut().find(|a| a.symbol == 1 && a.position == 2 && a.direction == Direction::In).expect("in2").position = 3;
        t.arguments.push(Argument { symbol: 3, position: 0, direction: Direction::Out, data_type: arr });
        t.property_rules.push(PropertyRule { symbol: -2, property: "len".into(), expr: PropertyExpr::Literal(1) });
        let req = |position: u32, against: PropertyExpr| Requirement {
            symbol: 4,
            subject: Subject::Input { position, property: "len".into() },
            relation: Relation::Eq,
            against,
            adapter: "fit.pair".into(),
        };
        t.requirements.push(req(7, PropertyExpr::Size("n".into())));
        t.bindings.push(Binding { symbol: 4, position: 2, data_type: Ty::Futhark(2), properties: vec![] });
        t.property_rules.push(PropertyRule { symbol: 2, property: "len".into(), expr: PropertyExpr::Input { position: 5, property: "len".into() } });
        t.arguments.push(Argument { symbol: -2, position: 0, direction: Direction::In, data_type: Ty::Futhark(9) });
        t.requirements.push(req(0, PropertyExpr::Sum(vec![PropertyExpr::Unknown])));
        t.requirements.push(req(1, PropertyExpr::sum(vec![PropertyExpr::Min(vec![PropertyExpr::Size("n k".into())]), PropertyExpr::Literal(1)])));
        let d = t.check().expect_err("defects");
        assert!(d.iter().all(|e| e.starts_with("our software has a bug: kingdom histogram test: ")), "{d:#?}");
        for want in [
            "symbol 1's input 2 is at position 3",
            "symbol 3 has 2 outputs",
            "symbol -2's rule sets out.len, which Futhark(2) does not carry",
            "names input 7, which does not exist",
            "binding names argument 2, which does not exist",
            "reads in5.len, but it has 2 inputs",
            "symbol 2 has two rules for out.len",
            "Futhark(9), which is not a data type",
            "a sum of 1 terms",
            "has a min of 1 arguments",
            "names size \"n k\", which is not an identifier",
        ] {
            assert!(d.iter().any(|e| e.contains(want)), "missing {want:?} in {d:#?}");
        }
        assert_eq!(d.len(), 11, "{d:#?}");
    }
}
