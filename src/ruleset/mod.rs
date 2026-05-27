//! Ruleset registry. Each module is a standalone egglog ruleset; the registry
//! is intentionally data-first (rules are `&str` programs) so new rule sources
//! can add modules without touching the public API or `lib.rs`. NOTE: the
//! modules are NOT all mutually confluent — see `src/parity.rs` for the
//! per-family split (distribute/trig/rational explode if co-saturated).

pub mod distribute;
pub mod identities;
pub mod powers;
pub mod rational;
pub mod sympy_mined;
pub mod trig;
pub mod trig_fu;
pub mod wide;
