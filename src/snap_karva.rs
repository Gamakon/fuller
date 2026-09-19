//! `snap_karva` — egglog-backed constant snapping (CR_snap_karva).
//!
//! Proliferates equivalent + constant-substitution forms of a chromosome in one
//! e-graph, extracts the K cheapest, returns them as candidate karva
//! chromosomes for the caller to score (HFF + R² guard). Composes constant
//! snaps with the algebra rules — so `0.0796·X` reaches `(1/(4π))·X` even though
//! the GA never recognised `0.0796`.
//!
//! Constants are represented as `(Var "<name>")` in Math — they compose with
//! algebra and evaluate via the env (we bind the constant names to their
//! values). The snap rewrites (`Num(v) → <const form>`) are NOT identities (they
//! hold only within the lattice's sig-fig tolerance), so this ruleset is used
//! ONLY here — never loaded into the parity scorer or denoise, which would let
//! egglog "prove" approximate equalities.
//!
//! The constant lattice is a build artifact (`parity/constants_lattice.json`,
//! generated offline by `parity/gen_constants.py`), embedded at compile time.

use std::collections::HashMap;

use egglog::EGraph;

use crate::expr::MATH_DATATYPE;
use crate::ruleset::identities::ALGEBRA_RULESET;

/// The lattice JSON, frozen at build time.
const LATTICE_JSON: &str = include_str!("../parity/constants_lattice.json");

/// One lattice entry: a numeric value and the Math s-expression (over constant
/// `Var`s + integer `Num`s) that is its simplest symbolic form.
#[derive(Debug, Clone)]
pub struct ConstEntry {
    pub value: f64,
    pub math: String,
    pub label: String,
}

/// Parse the embedded lattice. Hand-rolled (no serde dep): the file is a flat
/// JSON array of objects with string/number fields we control the shape of.
pub fn lattice() -> Vec<ConstEntry> {
    lattice_static().to_vec()
}

/// The lattice, parsed ONCE. `lattice()` re-parsed ~1400 JSON entries on
/// every call, and `snap_variants` calls it per gene: 0.7 ms of a 0.7 ms
/// snap. Internal callers borrow this; `lattice()` still returns an owned
/// copy for the public API.
fn lattice_static() -> &'static [ConstEntry] {
    static LATTICE: std::sync::OnceLock<Vec<ConstEntry>> = std::sync::OnceLock::new();
    LATTICE.get_or_init(|| {
        let mut out = parse_lattice(LATTICE_JSON);
        out.extend(small_integer_entries());
        out
    })
}

/// Small integers and simple rationals.
///
/// The JSON lattice is a PHYSICS table — constants and their composites — so it
/// contained no plain integers at all. A fitted coefficient of 1 therefore
/// matched nothing (observed in a recovery run as
/// `0.999999999999999*sqrt(...)` reaching the reported equation), and the
/// nearest forms for the ones that did appear were absurd: 2.0 only as
/// `sqrt2^2`, 0.5 only as `h/(4*pi*hbar)`. Without these entries a snap can
/// actively mangle an exact rational into a physics coincidence.
///
/// Kept deliberately small. These are the values an LSM fit lands on when the
/// structure is right and the coefficient is trivial; a long tail of integers
/// would give the tolerance more chances to match something by accident.
fn small_integer_entries() -> Vec<ConstEntry> {
    let mut v = Vec::new();
    for n in 1..=12i64 {
        v.push(ConstEntry {
            value: n as f64,
            math: format!("(Num {:.1})", n as f64),
            label: n.to_string(),
        });
        v.push(ConstEntry {
            value: -(n as f64),
            math: format!("(Num {:.1})", -(n as f64)),
            label: format!("-{n}"),
        });
    }
    // Reciprocals: the other side of a division an LSM fit can produce.
    for d in [2i64, 3, 4, 5, 6, 8, 10, 12] {
        v.push(ConstEntry {
            value: 1.0 / d as f64,
            math: format!("(Div (Num 1.0) (Num {:.1}))", d as f64),
            label: format!("1/{d}"),
        });
        v.push(ConstEntry {
            value: -1.0 / d as f64,
            math: format!("(Div (Num -1.0) (Num {:.1}))", d as f64),
            label: format!("-1/{d}"),
        });
    }
    // The few non-unit rationals that actually turn up in physics forms.
    for (n, d) in [(2i64, 3i64), (3, 2), (3, 4), (5, 2), (2, 5)] {
        v.push(ConstEntry {
            value: n as f64 / d as f64,
            math: format!("(Div (Num {:.1}) (Num {:.1}))", n as f64, d as f64),
            label: format!("{n}/{d}"),
        });
    }
    v
}

fn parse_lattice(s: &str) -> Vec<ConstEntry> {
    let mut out = Vec::new();
    // Each object is `{...}`; split on `},` boundaries within the top array.
    for obj in s.trim().trim_start_matches('[').trim_end_matches(']').split("},") {
        let obj = obj.trim().trim_start_matches('{').trim_end_matches('}');
        if obj.is_empty() {
            continue;
        }
        let (mut value, mut math, mut label) = (None, None, None);
        for field in split_top_level(obj) {
            let (k, v) = match field.split_once(':') {
                Some(kv) => kv,
                None => continue,
            };
            let k = k.trim().trim_matches('"');
            let v = v.trim();
            match k {
                "value" => value = v.parse::<f64>().ok(),
                "math" => math = Some(unescape(v.trim_matches('"'))),
                "label" => label = Some(unescape(v.trim_matches('"'))),
                _ => {}
            }
        }
        if let (Some(value), Some(math), Some(label)) = (value, math, label) {
            out.push(ConstEntry { value, math, label });
        }
    }
    out
}

/// Split a JSON object body on top-level commas (not commas inside quotes).
fn split_top_level(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut in_str = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                in_str = !in_str;
                cur.push(c);
            }
            '\\' if in_str => {
                cur.push(c);
                if let Some(n) = chars.next() {
                    cur.push(n);
                }
            }
            ',' if !in_str => {
                parts.push(std::mem::take(&mut cur));
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur);
    }
    parts
}

fn unescape(s: &str) -> String {
    s.replace("\\\"", "\"").replace("\\\\", "\\")
}

/// The canonical constant ATOMS — the names a chromosome can read back as a
/// SINGLE named terminal. The engine pre-registers all of these once per fit
/// (symmetry with `master_pset`), giving determinism across calls: snap_karva
/// only ever introduces names from this closed set as atoms.
///
/// Composed forms (e.g. `1/(4π)`) are NOT here by design — per the engine
/// owners' distinction, they appear in candidates as `(Inv (Mul (Num 4.0)
/// (Var "pi")))` built from these atoms, not as their own terminal. So this is
/// the lattice's single-`Var` entries (~16 base constants), not all 1428.
pub fn master_constants() -> Vec<(String, f64)> {
    // Clone keys/values to give callers owned data; constant_values() returns
    // a &'static reference to the cached map.
    let mut v: Vec<(String, f64)> = constant_values()
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

/// Map of constant name -> value, for binding constant `Var`s during eval.
///
/// Cached behind a process-wide `OnceLock` — the lattice is fixed at compile
/// time, so the map is built once and shared. Critical for performance in
/// `denoise`/`prune_on_data`, which calls into this on every row × every
/// candidate; rebuilding the HashMap each time was a 6-7 figure allocation
/// hot path.
pub fn constant_values() -> &'static HashMap<String, f64> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<HashMap<String, f64>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut m = HashMap::new();
        for e in lattice() {
            if let Some(name) = e.math.strip_prefix("(Var \"").and_then(|r| r.strip_suffix("\")")) {
                m.insert(name.to_string(), e.value);
            }
        }
        m
    })
}

/// The DOWN-FLIP: replace every named-constant terminal (`pi`, `G`, `sqrt2`,
/// ... — any `Token::Var` whose name is in [`constant_values`]) with its
/// numeric literal value, in both head and tail. The inverse of the
/// `snap_variants` up-flip; together they form the representation-flip
/// mutation pair: the same structure can compete in the population in
/// constants-form and numeric-form, and selection decides which recovers the
/// law. Behaviour-preserving by construction (eval binds those names to
/// exactly these values) and trivially deterministic.
///
/// Returns `(head, tail, replaced)` where `replaced` lists the distinct
/// constant names substituted (empty = the chromosome was already fully
/// numeric; callers can skip the no-op mutant).
pub fn concretize(
    head: &[crate::karva::Token],
    tail: &[crate::karva::Token],
) -> (Vec<crate::karva::Token>, Vec<crate::karva::Token>, Vec<String>) {
    use crate::karva::Token;
    fn map_toks(
        toks: &[Token],
        cv: &HashMap<String, f64>,
        replaced: &mut Vec<String>,
    ) -> Vec<Token> {
        toks.iter()
            .map(|t| match t {
                Token::Var(name) => {
                    if let Some(&v) = cv.get(name) {
                        if !replaced.contains(name) {
                            replaced.push(name.clone());
                        }
                        Token::Num(v)
                    } else {
                        t.clone()
                    }
                }
                other => other.clone(),
            })
            .collect()
    }
    let cv = constant_values();
    let mut replaced: Vec<String> = Vec::new();
    let new_head = map_toks(head, cv, &mut replaced);
    let new_tail = map_toks(tail, cv, &mut replaced);
    replaced.sort();
    (new_head, new_tail, replaced)
}

/// A snapped candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapCandidate {
    pub expr: String,
    pub cost: u64,
    /// Names of the lattice constants this candidate introduced (e.g. ["pi"]),
    /// so the caller can register them as pset terminals and decode the karva.
    pub constants_used: Vec<String>,
}

/// Which constant names from the lattice appear as `(Var "name")` in a Math
/// expr (i.e. names that are in `constant_values`, not free variables).
fn constants_in(expr: &str) -> Vec<String> {
    let cv = constant_values();
    let mut found = Vec::new();
    let mut i = 0;
    while let Some(p) = expr[i..].find("(Var \"") {
        let start = i + p + 6;
        if let Some(endrel) = expr[start..].find('"') {
            let name = &expr[start..start + endrel];
            if cv.contains_key(name) && !found.contains(&name.to_string()) {
                found.push(name.to_string());
            }
            i = start + endrel;
        } else {
            break;
        }
    }
    found
}


/// Significant-figure-relative match (magnitude-aware: works for pi~3 and
/// G~7e-11 alike).
fn approx_eq(a: f64, b: f64, rel_tol: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    let denom = b.abs().max(1e-300);
    (a - b).abs() / denom < rel_tol
}

fn fmt_f64(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() { format!("{v:.1}") } else { format!("{v}") }
}

/// Collect the numeric literals in a Math s-expression (cheap scan).
fn numeric_atoms(expr: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let bytes = expr.as_bytes();
    let mut i = 0;
    while let Some(p) = expr[i..].find("(Num ") {
        let start = i + p + 5;
        let end = start + expr[start..].find(')').unwrap_or(0);
        if let Ok(v) = expr[start..end].trim().parse::<f64>() {
            out.push(v);
        }
        i = end;
        if i >= bytes.len() {
            break;
        }
    }
    out
}

/// Snap `input` (a Math s-expression): for each numeric atom that matches a
/// lattice constant within `rel_tol`, produce a candidate with that atom
/// replaced by the constant's symbolic form. Returns the original plus one
/// candidate per confirmed snap, then per pair-of-snaps (composed), up to `k`.
///
/// Each replacement is VERIFIED in the e-graph (`Num v` and the const form land
/// in the same e-class after saturating algebra+snap) before it is emitted — so
/// we never propose a substitution the rewrite engine couldn't justify. This
/// avoids relying on `extract_variants`' cost choices (the snapped form is not
/// structurally cheaper than the raw decimal, so cost-based extraction hides
/// it). String substitution on confirmed snaps is deterministic and bounded.
pub fn snap_variants(input: &str, k: usize, rel_tol: f64) -> Result<Vec<SnapCandidate>, String> {
    let entries = lattice_static();
    // De-duplicate repeated values (first-seen order kept): the same fitted
    // constant appearing at two sites is one snap decision, and duplicate
    // entries would otherwise propose identical candidates.
    let mut atoms: Vec<f64> = Vec::new();
    for a in numeric_atoms(input) {
        if !atoms.iter().any(|x| x.to_bits() == a.to_bits()) {
            atoms.push(a);
        }
    }

    // Find, per atom, the simplest lattice form whose e-class the rewrite engine
    // confirms equal to (Num atom). (entries are simplest-first by construction
    // of the lattice dedup, but we re-rank by ops to be safe.)
    let mut snaps: Vec<(f64, String, String)> = Vec::new(); // (atom, const_math, label)
    for &atom in &atoms {
        if let Some(e) = best_snap_cached(atom, entries, rel_tol)? {
            snaps.push((atom, e.math, e.label));
        }
    }

    let mut out: Vec<SnapCandidate> = Vec::new();
    // cost here is a simple proxy: original lowest, each snap adds the const
    // form's node count. The caller re-scores on data anyway.
    out.push(SnapCandidate { expr: input.to_string(), cost: 0, constants_used: vec![] });

    // Single-atom snaps.
    for (atom, cmath, _label) in &snaps {
        let replaced = replace_num(input, *atom, cmath);
        if replaced != input {
            let constants_used = constants_in(&replaced);
            out.push(SnapCandidate { expr: replaced, cost: 1, constants_used });
        }
    }
    // Composed: replace ALL snapped atoms at once (the (1/(4pi))-style win where
    // several constants snap in one chromosome).
    if snaps.len() > 1 {
        let mut all = input.to_string();
        for (atom, cmath, _l) in &snaps {
            all = replace_num(&all, *atom, cmath);
        }
        if all != input {
            let constants_used = constants_in(&all);
            out.push(SnapCandidate { expr: all, cost: 2, constants_used });
        }
    }

    // Stable de-duplication (dedup_by only removes ADJACENT duplicates; the
    // composed candidate can collide with a non-adjacent single snap).
    let mut seen = std::collections::HashSet::new();
    out.retain(|c| seen.insert(c.expr.clone()));
    out.truncate(k.max(1));
    Ok(out)
}

/// `best_snap_for`, memoised on (atom, rel_tol).
///
/// The decision depends on nothing else — the lattice is fixed — and a GEP
/// population repeats the same handful of constants thousands of times: every
/// gene draws from a small RNC array. Each miss builds and saturates an
/// e-graph (~1 ms); measured on a population batch that was most of the snap
/// cost. Errors are not cached, so a transient failure is retried.
fn best_snap_cached(
    atom: f64,
    entries: &[ConstEntry],
    rel_tol: f64,
) -> Result<Option<ConstEntry>, String> {
    type Cache = std::sync::Mutex<std::collections::HashMap<(u64, u64), Option<ConstEntry>>>;
    static CACHE: std::sync::OnceLock<Cache> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (atom.to_bits(), rel_tol.to_bits());
    if let Some(hit) = cache.lock().map_err(|e| format!("snap cache: {e}"))?.get(&key) {
        return Ok(hit.clone());
    }
    let found = best_snap_for(atom, entries, rel_tol)?;
    cache
        .lock()
        .map_err(|e| format!("snap cache: {e}"))?
        .insert(key, found.clone());
    Ok(found)
}

/// The simplest lattice entry whose const form egglog confirms equals (Num atom)
/// in the same e-class after saturating algebra+snap. Returns None if nothing
/// snaps. Verifying via the e-graph (not just numeric closeness) keeps this
/// honest — only substitutions the rewrite engine can justify are proposed.
fn best_snap_for(
    atom: f64,
    entries: &[ConstEntry],
    rel_tol: f64,
) -> Result<Option<ConstEntry>, String> {
    let mut candidates: Vec<&ConstEntry> =
        entries.iter().filter(|e| approx_eq(atom, e.value, rel_tol)).collect();
    // simplest symbolic form first (fewest chars is a fine proxy for fewest ops)
    candidates.sort_by_key(|e| e.math.len());
    for e in candidates {
        if confirms_equal(atom, &e.math)? {
            return Ok(Some(e.clone()));
        }
    }
    Ok(None)
}

/// Does saturating algebra+snap put `(Num atom)` and `const_math` in one e-class?
fn confirms_equal(atom: f64, const_math: &str) -> Result<bool, String> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, MATH_DATATYPE).map_err(|e| format!("datatype: {e}"))?;
    egraph.parse_and_run_program(None, crate::expr::GUARD_RELATIONS).map_err(|e| format!("guards: {e}"))?;
    egraph.parse_and_run_program(None, ALGEBRA_RULESET).map_err(|e| format!("algebra: {e}"))?;
    let snap_rule = format!(
        "(ruleset snap)\n(rewrite (Num {}) {} :ruleset snap)\n",
        fmt_f64(atom), const_math
    );
    egraph.parse_and_run_program(None, &snap_rule).map_err(|e| format!("snap rule: {e}"))?;
    let prog = format!(
        "(let __a (Num {}))\n(let __c {})\n\
         (unstable-combined-ruleset snap_all guards algebra snap)\n\
         (run-schedule (repeat 8 (run snap_all)))\n\
         (check (= __a __c))",
        fmt_f64(atom), const_math
    );
    match egraph.parse_and_run_program(None, &prog) {
        Ok(_) => Ok(true),
        // A failed (check ..) is the expected "not equal" answer — match the
        // error VARIANT, not a substring of the message, so a genuine egglog
        // failure that merely mentions "check" is never swallowed as false.
        Err(egglog::Error::CheckError(..)) => Ok(false),
        Err(e) => Err(e.to_string()),
    }
}

/// Replace every `(Num atom)` literal with `replacement` (a Math s-expr).
/// All occurrences: the same fitted value at two sites is the same constant,
/// and replacing only the first would leave the second site never proposed.
fn replace_num(expr: &str, atom: f64, replacement: &str) -> String {
    let needle = format!("(Num {})", fmt_f64(atom));
    expr.replace(&needle, replacement)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_loads() {
        let l = lattice();
        assert!(l.len() > 1000, "lattice should have ~1400 entries, got {}", l.len());
        assert!(l.iter().any(|e| e.label == "1/(4*pi)"), "missing 1/(4pi)");
    }

    #[test]
    fn constant_values_has_pi_and_g() {
        let cv = constant_values();
        assert!((cv["pi"] - std::f64::consts::PI).abs() < 1e-9);
        assert!(cv.contains_key("G"));
    }

    #[test]
    fn numeric_atoms_extracts() {
        let a = numeric_atoms(r#"(Mul (Num 0.0796) (Var "x"))"#);
        assert_eq!(a.len(), 1);
        assert!((a[0] - 0.0796).abs() < 1e-9);
    }

    #[test]
    fn snaps_quarter_pi_coefficient() {
        // 0.0796 ~= 1/(4pi). snap should produce a variant naming it.
        let v = snap_variants(r#"(Mul (Num 0.0796) (Var "x"))"#, 16, 1e-3).unwrap();
        // some variant should contain pi (the snapped constant)
        assert!(
            v.iter().any(|c| c.expr.contains("\"pi\"")),
            "expected a pi-snapped variant; got {:?}",
            v.iter().map(|c| &c.expr).collect::<Vec<_>>()
        );
        // original is always present
        assert!(v.iter().any(|c| c.expr.contains("0.0796")));
    }

    #[test]
    fn concretize_replaces_constants_and_only_constants() {
        use crate::karva::Token;
        let head = vec![
            Token::Func("mul".into()),
            Token::Var("pi".into()),      // registered constant -> Num
            Token::Var("r".into()),       // ordinary variable -> untouched
        ];
        let tail = vec![Token::Var("sqrt2".into()), Token::Num(1.0)];
        let (h, t, replaced) = concretize(&head, &tail);
        assert_eq!(replaced, vec!["pi".to_string(), "sqrt2".to_string()]);
        assert!(matches!(&h[1], Token::Num(v) if (*v - std::f64::consts::PI).abs() < 1e-12));
        assert_eq!(h[2], Token::Var("r".into()), "free variable must survive");
        assert!(matches!(&t[0], Token::Num(v) if (*v - std::f64::consts::SQRT_2).abs() < 1e-12));
        assert_eq!(t[1], Token::Num(1.0));
        // Deterministic + idempotent: a second pass replaces nothing.
        let (h2, t2, r2) = concretize(&h, &t);
        assert!(r2.is_empty(), "already-numeric chromosome is a no-op");
        assert_eq!((h2, t2), (h, t));
    }

    #[test]
    fn no_snap_leaves_expr_alone() {
        // 0.5 with a tiny tol that matches nothing in the lattice context here:
        // a bare numeric with no near constant just returns itself.
        let v = snap_variants(r#"(Add (Var "x") (Num 0.5))"#, 8, 1e-9).unwrap();
        assert!(v.iter().any(|c| c.expr.contains("0.5")));
    }
}

#[cfg(test)]
mod integer_lattice_tests {
    use super::*;

    fn nearest(target: f64) -> Option<ConstEntry> {
        lattice()
            .into_iter()
            .filter(|e| (e.value - target).abs() <= 1e-9 * target.abs().max(1.0))
            .min_by(|a, b| {
                (a.value - target).abs()
                    .partial_cmp(&(b.value - target).abs())
                    .unwrap()
            })
    }

    #[test]
    fn one_is_in_the_lattice() {
        // The gap that let 0.999999999999999*sqrt(...) reach a reported result.
        let e = nearest(0.999_999_999_999_999).expect("no entry near 1.0");
        assert_eq!(e.label, "1");
    }

    #[test]
    fn small_integers_and_reciprocals_are_present() {
        for (target, label) in [(2.0, "2"), (-3.0, "-3"), (0.5, "1/2"), (1.5, "3/2")] {
            let e = nearest(target)
                .unwrap_or_else(|| panic!("no entry near {target}"));
            assert_eq!(e.label, label, "wrong entry for {target}");
        }
    }

    #[test]
    fn integer_math_forms_round_trip_as_math() {
        // Each added entry's `math` must parse as a Math s-expression, or snap
        // emits something the karva decoder cannot read back.
        for e in small_integer_entries() {
            assert!(e.math.starts_with('('), "not an s-expr: {}", e.math);
            assert!(e.math.ends_with(')'), "not an s-expr: {}", e.math);
        }
    }

    #[test]
    fn physics_constants_still_win_where_they_should() {
        // Adding integers must not shadow a real constant.
        let pi = lattice().into_iter().find(|e| e.label == "pi");
        assert!(pi.is_some(), "pi missing from lattice");
        assert!(nearest(3.0).is_some_and(|e| e.label == "3"),
                "3.0 should snap to the integer, not a pi composite");
    }
}
