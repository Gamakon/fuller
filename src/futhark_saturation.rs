//! THE FUTHARK KINGDOM'S E-GRAPH: equality saturation over a view's programs, with the rules of
//! `kingdoms/futhark/master/6/rules.tsv` (each one an identity of Futhark's arithmetic, witnessed by `gen_tables.py`).
//! It is a local search on the elites: its rules are equalities by construction, so it moves the cost axis only.

/// The e-graph of every view, A7's included: one sort a data type, the variable constructor on the resolved variable
/// (plan 7, task F5).
pub mod master6;
