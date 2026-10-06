//! Sound, semantics-preserving, CONTRACTING regex rewrite rules.
//!
//! Every rule shrinks or keeps size and is individually sound (preserves the
//! matched language under the `regex` crate's full-match semantics). Soundness
//! is BY CONSTRUCTION — tree extraction can only pick a member of the input's
//! e-class, and every member is language-equal if the rules are sound. No DFA /
//! derivatives (the tests cross-check with the `regex` crate oracle).
//!
//! # Drawable-ops invariant
//!
//! No rule introduces `RepN`/`RepUpto` or a bare Integer leaf. Every right-hand
//! side is built only from drawable `Re` constructors, so a simplified rule
//! round-trips into a valid multi-typed phylu gene.
//!
//! # Boundedness
//!
//! All rules are CONTRACTING or size-neutral normalisers (associativity).
//! `regex_simplify` additionally caps iterations for defence-in-depth.
//!
//! # Rules NOT included (and why)
//!
//! - Distributivity (`x(y|z) = xy|xz`) — the EXPANDING direction blows up the
//!   e-graph; the factoring direction needs a pattern we do not have as a single
//!   contracting rewrite. SKIPPED (matches the Math layer's distribute caution).
//! - Full language equality (`(x|y)* = (x*y*)*` denesting etc.) — sound but
//!   EXPANDING; left out to keep the schedule bounded and the output smaller.

/// The regex rewrite ruleset (egglog surface syntax, over the flat `Re` sort).
pub const RE_RULESET: &str = r#"
(ruleset regex)

; ---- Concat identities ----
; Empty is the identity for Concat (eps . x = x, x . eps = x).
(rewrite (Concat (Empty) x) x :ruleset regex)
(rewrite (Concat x (Empty)) x :ruleset regex)
; Associativity (size-neutral normaliser; lets the above fire across nests).
(rewrite (Concat (Concat x y) z) (Concat x (Concat y z)) :ruleset regex)

; ---- Alt identities ----
; Idempotence: x | x = x.
(rewrite (Alt x x) x :ruleset regex)
; Associativity (size-neutral).
(rewrite (Alt (Alt x y) z) (Alt x (Alt y z)) :ruleset regex)
; Absorption with Star: x | x* = x*, eps | x* = x*.
(rewrite (Alt x (Star x)) (Star x) :ruleset regex)
(rewrite (Alt (Star x) x) (Star x) :ruleset regex)
(rewrite (Alt (Empty) (Star x)) (Star x) :ruleset regex)
(rewrite (Alt (Star x) (Empty)) (Star x) :ruleset regex)
; Opt is exactly (x | eps): x | eps  and  eps | x collapse to (Opt x).
(rewrite (Alt x (Empty)) (Opt x) :ruleset regex)
(rewrite (Alt (Empty) x) (Opt x) :ruleset regex)

; ---- Star identities ----
; Idempotence of Star: (x*)* = x*.
(rewrite (Star (Star x)) (Star x) :ruleset regex)
; Star of Empty is Empty: eps* = eps.
(rewrite (Star (Empty)) (Empty) :ruleset regex)
; Star absorbs Opt: (x?)* = x*.
(rewrite (Star (Opt x)) (Star x) :ruleset regex)
; Star . Star = Star.
(rewrite (Concat (Star x) (Star x)) (Star x) :ruleset regex)

; ---- Plus / Opt identities ----
; (x?)? = x?.
(rewrite (Opt (Opt x)) (Opt x) :ruleset regex)
; (x*)? = x*  and  (x+)? ... : Opt of Star is Star.
(rewrite (Opt (Star x)) (Star x) :ruleset regex)
; Plus of Star is Star: (x*)+ = x*.
(rewrite (Plus (Star x)) (Star x) :ruleset regex)
; Star of Plus is Star: (x+)* = x*.
(rewrite (Star (Plus x)) (Star x) :ruleset regex)
; Opt of Plus is Star: (x+)? = x*.
(rewrite (Opt (Plus x)) (Star x) :ruleset regex)
"#;

/// Build a fresh e-graph with RE_DATATYPE and RE_RULESET loaded.
pub fn re_egraph() -> Result<egglog::EGraph, String> {
    let mut egraph = egglog::EGraph::default();
    egraph
        .parse_and_run_program(None, crate::regex::expr::RE_DATATYPE)
        .map_err(|e| format!("RE_DATATYPE: {e}"))?;
    egraph
        .parse_and_run_program(None, RE_RULESET)
        .map_err(|e| format!("RE_RULESET: {e}"))?;
    Ok(egraph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn re_ruleset_loads() {
        re_egraph().expect("RE e-graph loads");
    }
}
