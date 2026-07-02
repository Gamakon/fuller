//! Ruleset registry. Each module is a standalone egglog ruleset; the registry
//! is intentionally data-first (rules are `&str` programs) so new rule sources
//! can add modules without touching the public API or `lib.rs`. NOTE: the
//! modules are NOT all mutually confluent — see `src/parity.rs` for the
//! per-family split (distribute/trig/rational explode if co-saturated).
//!
//! `sympy_mined` is defined and tested but deliberately wired into NO family
//! yet: it is staged material for the kingdom classifier (the learned router
//! that picks which family to load per input — see CLAUDE.md "In flight").
//! Adding it to an existing family without the router would re-open the
//! non-confluence trap the family split exists to avoid.

pub mod distribute;
pub mod identities;
pub mod powers;
pub mod rational;
pub mod sympy_mined;
pub mod trig;
pub mod trig_fu;
pub mod wide;
