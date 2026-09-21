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
pub mod device;
pub mod engine;
pub mod flat;
pub mod node;
pub mod pack;
pub mod reader;
pub mod sexp;
pub mod snap_graft;
pub mod snap_guard;
pub mod snap_table;
pub mod tables;

use engine::{CallerFacts, Config, LitMode, Outcome, Search};
use node::Tree;
use tables::{Exactness, Refused, Tables};

/// The rule texts the linter is derived from, in load order. Rule ids follow
/// this order, so it is part of the linter's determinism.
pub fn standard_sources() -> [(&'static str, &'static str); 6] {
    [
        ("guards", crate::expr::GUARD_RELATIONS),
        ("algebra", crate::ruleset::identities::ALGEBRA_RULESET),
        ("powers", crate::ruleset::powers::POWERS_RULESET),
        ("sign", crate::ruleset::sign::SIGN_RULESET),
        ("rational", crate::ruleset::rational::RATIONAL_RULESET),
        ("collect", crate::ruleset::collect::COLLECT_RULESET),
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
        fold_in_rounds: true,
    };
    Ok(engine::run(&tree, &rules, &tables.guards, &cfg))
}

/// What the caller's data says: rows to judge a prune on (none = no prune
/// candidates), and the inputs that are positive / non-zero on every row.
#[derive(Default)]
pub struct DataFacts<'a> {
    pub rows: &'a [Vec<(String, f64)>],
    pub positive_vars: Vec<String>,
    pub nonzero_vars: Vec<String>,
}

/// Candidate forms of `expr`, the input first: the linter's forms at `admit`
/// (up to `k`), the snap candidate (every literal within
/// [`engine::SNAP_CANDIDATE_TOL`] of a whole or half number taken for it), and
/// — when `rows` are given — data-judged prunes at five tolerances, each pruned
/// tree linted again. Labels: input / bit / rounding / finite / snap / prune.
/// A form is a CANDIDATE: only the linter's own are meaning-preserving; the
/// caller judges the rest on its data.
pub fn forms(
    tables: &Tables,
    expr: &str,
    inputs: &[String],
    admit: Exactness,
    k: usize,
    data: DataFacts<'_>,
) -> Result<Vec<(Tree, &'static str)>, String> {
    use engine::run;
    let DataFacts { rows, positive_vars, nonzero_vars } = data;
    let tree = Tree::parse(expr)?;
    let symbols = crate::geneframe::master_table();
    let kingdom = symbols.kingdom("Symbolic Regression");
    let rules = tables.usable(&kingdom);
    let caller = CallerFacts { positive: positive_vars, nonzero: nonzero_vars };
    let cfg = Config {
        inputs,
        caller: &caller,
        mode: LitMode::F64,
        search: Search::Beam(k.max(1)),
        max_steps: 64,
        admit,
        computed_literals: true,
        fold_in_rounds: true,
    };
    let outcome = run(&tree, &rules, &tables.guards, &cfg);
    let mut offered: Vec<(Tree, &'static str)> = vec![(tree.clone(), "input")];
    for (form, level) in outcome.forms.iter().zip(&outcome.levels).filter(|(f, _)| **f != tree).take(k.max(1)) {
        let label = match level {
            Exactness::Bit => "bit",
            Exactness::Rounding => "rounding",
            Exactness::Finite => "finite",
        };
        offered.push((form.clone(), label));
    }
    if let Some(snapped) = engine::snap_candidate(&tree, engine::SNAP_CANDIDATE_TOL) {
        let tidy = run(&snapped, &rules, &tables.guards, &cfg).best;
        if offered.iter().all(|(f, _)| *f != tidy) {
            offered.push((tidy, "snap"));
        }
    }
    if !rows.is_empty() {
        let tidy = outcome.best.to_math();
        if let Ok(reference) = crate::extract::eval_expr_rows(&tidy, rows) {
            for tol in [1e-10_f64, 1e-6, 1e-3, 1e-2, 1e-1] {
                // A prune leaves debris a rule can clear (`x*(-y)`, a bare
                // `Neg`), so the pruned tree goes through the linter again.
                let pruned = crate::extract::prune_on_data(&tidy, rows, &reference, tol)
                    .and_then(|p| Tree::parse(&p).ok())
                    .map(|p| run(&p, &rules, &tables.guards, &cfg).best);
                if let Some(pt) = pruned {
                    if offered.iter().all(|(f, _)| *f != pt) {
                        offered.push((pt, "prune"));
                    }
                }
            }
        }
    }
    Ok(offered)
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

    /// The least-squares intercept arrives as -7.000000000000002; snapped to
    /// -7 it cancels against the folded 3 + 4, and the model is what it is.
    #[test]
    fn a_literal_a_few_ulps_from_an_integer_is_that_integer() {
        let tables = Tables::standard().unwrap();
        let inputs = vec!["q".to_string()];
        let e = r#"(Add (Add (Add (Num 3.0) (Var "q")) (Num 4.0)) (Num -7.000000000000002))"#;
        let all = lint(&tables, e, &inputs, &Options::default()).unwrap();
        let snapped = all.forms.iter().zip(&all.levels).find(|(f, _)| f.to_math().contains("(Num -7.0)"));
        assert_eq!(snapped.map(|(_, l)| *l), Some(Exactness::Rounding), "the snapped form is rounding-exact");
        // Never at bit level: the value moved.
        let bit = lint(&tables, e, &inputs, &Options { admit: Exactness::Bit, ..Options::default() }).unwrap();
        assert!(bit.forms.iter().all(|f| !f.to_math().contains("(Num -7.0)")));
        // A real constant is not an integer: 7.0000001 stays.
        assert!(engine::snap_literals(&Tree::parse("(Num 7.0000001)").unwrap()).is_none());
        assert!(engine::snap_literals(&Tree::parse("(Num 0.0000000000000004)").unwrap()).is_none());
        assert_eq!(engine::snap_literals(&Tree::parse("(Num 2.4999999999999996)").unwrap()).unwrap().to_math(), "(Num 2.5)");
    }

    /// An evolved constant within 1e-4 of a whole number is OFFERED as that
    /// number — a candidate for the data to judge, not an equivalence.
    #[test]
    fn an_evolved_constant_near_a_whole_number_is_a_candidate() {
        let t = Tree::parse(r#"(Add (Mul (Num 2.99996) (Var "x")) (Num 0.00003))"#).unwrap();
        let snapped = engine::snap_candidate(&t, engine::SNAP_CANDIDATE_TOL).unwrap();
        assert_eq!(snapped.to_math(), r#"(Add (Mul (Num 3.0) (Var "x")) (Num 0.0))"#);
        // Outside the tolerance nothing is offered.
        assert!(engine::snap_candidate(&Tree::parse("(Num 2.9998)").unwrap(), engine::SNAP_CANDIDATE_TOL).is_none());
        assert!(engine::snap_candidate(&Tree::parse("(Num 3.0)").unwrap(), engine::SNAP_CANDIDATE_TOL).is_none());
    }

    /// Feynman I.47.23: three separate roots become the truth's one root.
    #[test]
    fn separate_roots_merge_into_one() {
        let tables = Tables::standard().unwrap();
        let inputs: Vec<String> = ["gamma", "pr", "rho"].iter().map(|s| s.to_string()).collect();
        let e = r#"(Div (Mul (ProtectedSqrt (Var "gamma")) (ProtectedSqrt (Var "pr"))) (ProtectedSqrt (Var "rho")))"#;
        let out = lint(&tables, e, &inputs, &Options::default()).unwrap();
        assert_eq!(
            out.best.to_math(),
            r#"(ProtectedSqrt (Div (Mul (Var "gamma") (Var "pr")) (Var "rho")))"#
        );
        // The protected divide of two protected roots must NOT merge: between
        // |b| = 1e-12 and 1e-6 the two sides are different finite numbers.
        let p = r#"(ProtectedDiv (ProtectedSqrt (Var "gamma")) (ProtectedSqrt (Var "rho")))"#;
        assert_eq!(lint(&tables, p, &inputs, &Options::default()).unwrap().best.to_math(), p);
    }

    /// Feynman III.7.38: with the data saying `mom` is positive,
    /// exp(ln|mom|) is `mom`. Without that fact it must stay.
    #[test]
    fn exp_of_protected_log_needs_a_caller_fact() {
        let tables = Tables::standard().unwrap();
        let inputs = vec!["mom".to_string()];
        let e = r#"(ProtectedExp (ProtectedLog (Var "mom")))"#;
        assert_eq!(lint(&tables, e, &inputs, &Options::default()).unwrap().best.to_math(), e);
        let told = Options { positive_vars: inputs.clone(), ..Options::default() };
        assert_eq!(lint(&tables, e, &inputs, &told).unwrap().best.to_math(), r#"(Var "mom")"#);
    }

    /// Feynman III.13.18 at seed 15795: every column positive, the only sign in
    /// the term is a literal's. |(1/x3)/(-19)| is (1/x3)/19 — and without the
    /// fact that x3 is positive the Abs must stay.
    #[test]
    fn abs_over_a_negative_literal_needs_a_caller_fact() {
        let tables = Tables::standard().unwrap();
        let inputs = vec!["x3".to_string()];
        let e = r#"(Abs (ProtectedDiv (Inv (Var "x3")) (Num -19.0)))"#;
        assert!(lint(&tables, e, &inputs, &Options::default()).unwrap().best.to_math().contains("Abs"));
        let told = Options { positive_vars: inputs.clone(), ..Options::default() };
        assert_eq!(
            lint(&tables, e, &inputs, &told).unwrap().best.to_math(),
            r#"(ProtectedDiv (Inv (Var "x3")) (Num 19.0))"#
        );
    }

    /// Feynman II.10.9, fitted to R2 = 1 as sqrt|x0^2/(x1^2*(x2+1)^2)| and
    /// scored unsolved. With every column positive it is x0/(x1*(x2+1)).
    #[test]
    fn a_root_of_collected_squares_sheds_root_and_abs() {
        let tables = Tables::standard().unwrap();
        let inputs: Vec<String> = ["x0", "x1", "x2"].iter().map(|s| s.to_string()).collect();
        let e = r#"(ProtectedSqrt (Abs (Div (Pow2 (Var "x0")) (Mul (Pow2 (Var "x1")) (Pow2 (Add (Var "x2") (Num 1.0)))))))"#;
        let told = Options { positive_vars: inputs.clone(), ..Options::default() };
        assert_eq!(
            lint(&tables, e, &inputs, &told).unwrap().best.to_math(),
            r#"(Div (Var "x0") (Mul (Var "x1") (Add (Var "x2") (Num 1.0))))"#
        );
    }

    /// Feynman I.47.23 as the Rust engine found it (generation 1): the square of
    /// a root inside the radical. One radical must come out, not two.
    #[test]
    fn the_square_of_a_root_is_the_absolute_value() {
        let tables = Tables::standard().unwrap();
        let inputs = vec!["x0".to_string()];
        let e = r#"(Sqrt (Abs (Mul (Pow2 (ProtectedSqrt (Var "x0"))) (Num 39.0))))"#;
        let told = Options { positive_vars: inputs.clone(), ..Options::default() };
        let tidy = lint(&tables, e, &inputs, &told).unwrap().best.to_math();
        assert!(!tidy.contains("ProtectedSqrt") && !tidy.contains("Pow2") && !tidy.contains("Abs"), "{tidy}");
        assert!(lint(&tables, e, &inputs, &Options::default()).unwrap().best.to_math().contains("Abs"));
    }

    #[test]
    fn is_deterministic() {
        let e = r#"(Sub (Neg (Mul (Var "a") (Num -1.0))) (Neg (Abs (Pow2 (Var "b")))))"#;
        let (one, two) = (tidy(e), tidy(e));
        assert_eq!(one, two);
    }
}
