//! THE VIEW of the Futhark kingdom: a problem's kingdom is a view over the master table (`kingdoms/futhark/master/6/`).
//! Its specification (`kingdoms/futhark/sig/<problem>/m6/view.tsv`) is written by `kingdoms/futhark/sniff.py` from the
//! problem's Futhark source; the view filters the master's rows by it, instantiates them over the view's elements and
//! keeps what is reachable both ways ([`master6`]), and its e-graph is computed from the same table ([`egraph6`]).
//! `kingdoms/futhark/view.py` writes the same table for the generator's witnesses, and a test requires the two byte
//! for byte.

/// The master and the views: the master's format in `kingdoms/futhark/master/README.md`.
pub mod master6;

/// The views' e-graphs: one sort a data type, the variable constructor on the resolved variable (plan 7, task F5).
pub mod egraph6;
