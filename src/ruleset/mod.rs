//! Ruleset registry. Each module is a standalone egglog ruleset; the registry
//! is intentionally data-first (rules are `&str` programs) so future rule
//! sources — including the sympy-free rule-extraction work in
//! `docs/BRIEF_rule_extraction.md` — can add modules without touching the
//! public API or `lib.rs`.

pub mod distribute;
pub mod identities;
pub mod powers;
pub mod trig;
