//! lint — a table-driven rewriter for K-expressions (`docs/SPEC_fuller_gpu.md`).
//!
//! A graph linter with autofix: rule × expression × position, in bounded
//! steps, no e-graph. This module is the CPU reference: what the device kernel
//! is checked against, and a fast linter in its own right.
//!
//! The rules are not written here. They are READ from the egglog rule text the
//! rest of the crate runs, classified, and either admitted or refused with a
//! reason. `Tables::standard()` is that load.

pub mod classify;
pub mod engine;
pub mod node;
pub mod reader;
pub mod sexp;
pub mod tables;

use engine::{CallerFacts, Config, LitMode, Outcome, Search};
use node::Tree;
use tables::{Exactness, Refused, Tables};

/// The rule texts the linter is derived from, in load order. Rule ids follow
/// this order, so it is part of the linter's determinism.
pub fn standard_sources() -> [(&'static str, &'static str); 5] {
    [
        ("guards", crate::expr::GUARD_RELATIONS),
        ("algebra", crate::ruleset::identities::ALGEBRA_RULESET),
        ("powers", crate::ruleset::powers::POWERS_RULESET),
        ("sign", crate::ruleset::sign::SIGN_RULESET),
        ("rational", crate::ruleset::rational::RATIONAL_RULESET),
    ]
}

impl Tables {
    /// Read, classify and type-check `sources`. An unreadable form is an
    /// error; a form that cannot be a row is listed in `refused`.
    pub fn load(sources: &[(&str, &str)]) -> Result<Tables, String> {
        let read = reader::read(sources)?;
        let mut tables = Tables { rules: Vec::new(), guards: Vec::new(), refused: read.refused };
        for mut g in read.guards {
            match classify::classify_guard(&g) {
                Ok(range_only) => {
                    g.range_only = range_only;
                    g.id = tables.guards.len();
                    tables.guards.push(g);
                }
                Err(reason) => tables.refused.push(Refused {
                    ruleset: "guards".to_string(),
                    text: g.text.clone(),
                    reason,
                }),
            }
        }
        for d in &read.drafts {
            match classify::classify(d, tables.rules.len()) {
                Ok(rule) => tables.rules.push(rule),
                Err(reason) => tables.refused.push(Refused {
                    ruleset: d.ruleset.clone(),
                    text: d.text.clone(),
                    reason,
                }),
            }
        }
        Ok(tables)
    }

    /// The linter's tables for the crate's own rules.
    pub fn standard() -> Result<Tables, String> {
        Tables::load(&standard_sources())
    }
}

/// Options for [`lint`]. `inputs` is deliberately not here: it is a required
/// argument, never a default (spec §2a).
#[derive(Debug, Clone)]
pub struct Options {
    pub positive_vars: Vec<String>,
    pub nonzero_vars: Vec<String>,
    pub mode: LitMode,
    pub search: Search,
    pub max_steps: usize,
    pub admit: Exactness,
    pub computed_literals: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            positive_vars: Vec::new(),
            nonzero_vars: Vec::new(),
            mode: LitMode::F64,
            search: Search::Beam(8),
            max_steps: 64,
            admit: Exactness::Finite,
            computed_literals: true,
        }
    }
}

/// Lint one `Math` expression in the Symbolic Regression kingdom.
///
/// `inputs` names the data columns. A column is never folded to a constant,
/// whatever it is called.
pub fn lint(tables: &Tables, math: &str, inputs: &[String], opts: &Options) -> Result<Outcome, String> {
    let symbols = crate::geneframe::master_table();
    let kingdom = symbols.kingdom("Symbolic Regression");
    tables.type_check(&kingdom)?;
    let rules = tables.usable(&kingdom);
    let tree = Tree::parse(math)?;
    let caller = CallerFacts {
        positive: opts.positive_vars.clone(),
        nonzero: opts.nonzero_vars.clone(),
    };
    let cfg = Config {
        inputs,
        caller: &caller,
        mode: opts.mode,
        search: opts.search,
        max_steps: opts.max_steps,
        admit: opts.admit,
        computed_literals: opts.computed_literals,
    };
    Ok(engine::run(&tree, &rules, &tables.guards, &cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tidy(math: &str) -> String {
        let tables = Tables::standard().expect("tables load");
        let inputs: Vec<String> = ["a", "b", "c", "x", "y"].iter().map(|s| s.to_string()).collect();
        lint(&tables, math, &inputs, &Options::default()).expect("lints").best.to_math()
    }

    /// The load itself: what is admitted, what is refused, and why. A new
    /// egglog rule changes these numbers, so it can never vanish silently.
    #[test]
    fn census() {
        let tables = Tables::standard().expect("tables load");
        let mut by_set: std::collections::BTreeMap<&str, [usize; 5]> = Default::default();
        for r in &tables.rules {
            let e = by_set.entry(r.ruleset.as_str()).or_default();
            e[usize::from(r.order == tables::Order::B)] += 1;
            e[2 + r.exactness as usize] += 1;
        }
        let mut report = String::new();
        for (set, [a, b, bit, rounding, finite]) in &by_set {
            report.push_str(&format!("{set}: A={a} B={b} | bit={bit} rounding={rounding} finite={finite}\n"));
        }
        report.push_str(&format!(
            "guards={} range_only={}\n",
            tables.guards.len(),
            tables.guards.iter().filter(|g| g.range_only).count()
        ));
        for r in &tables.refused {
            report.push_str(&format!("REFUSED [{}] {} -- {}\n", r.ruleset, r.text, r.reason));
        }
        println!("{report}");
        assert!(!tables.rules.is_empty(), "no rule was admitted:\n{report}");
    }

    #[test]
    fn shrinks_the_shapes_the_rules_were_mined_for() {
        assert_eq!(tidy(r#"(Sub (Var "a") (Neg (Var "b")))"#), r#"(Add (Var "a") (Var "b"))"#);
        assert_eq!(tidy(r#"(Mul (Var "x") (Num 1.0))"#), r#"(Var "x")"#);
        assert_eq!(tidy(r#"(Abs (Pow2 (Var "x")))"#), r#"(Pow2 (Var "x"))"#);
        // A float-out (class B) reaching an absorber (class A): (-a - b)^2.
        assert_eq!(
            tidy(r#"(Pow2 (Sub (Neg (Var "a")) (Var "b")))"#),
            r#"(Pow2 (Add (Var "a") (Var "b")))"#
        );
    }

    /// K4: a data column is never folded, whatever it is called; a named
    /// constant that is NOT a column is.
    #[test]
    fn never_folds_an_input() {
        assert_eq!(tidy(r#"(Mul (Var "c") (Var "c"))"#), r#"(Pow2 (Var "c"))"#);
        let tables = Tables::standard().unwrap();
        let folded = lint(&tables, r#"(Mul (Var "c") (Var "c"))"#, &[], &Options::default()).unwrap();
        assert!(matches!(folded.best, Tree::Num(_)), "{}", folded.best.to_math());
    }

    #[test]
    fn is_deterministic() {
        let e = r#"(Sub (Neg (Mul (Var "a") (Num -1.0))) (Neg (Abs (Pow2 (Var "b")))))"#;
        let (one, two) = (tidy(e), tidy(e));
        assert_eq!(one, two);
    }
}
