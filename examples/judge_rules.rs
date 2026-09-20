//! Judge candidate rules mechanically.
//!
//!   cargo run --release --example judge_rules -- exprs.tsv proposal.md [more.md ..]
//!
//! Each proposal file holds one ```egglog fenced block of candidate rules. Every
//! top-level form is read and classified ON ITS OWN (one malformed rule cannot
//! take the rest down): admitted at a level, refused with the classifier's
//! counterexample, or unreadable. The admitted candidates are then run over a
//! corpus on top of the crate's own rules: which ever fire, and how many more
//! nodes come off.

use std::collections::BTreeMap;

use fuller::lint::classify::classify;
use fuller::lint::engine::{run, CallerFacts, Config, LitMode, Search};
use fuller::lint::node::Tree;
use fuller::lint::reader::{read, show};
use fuller::lint::sexp::parse_all;
use fuller::lint::tables::{Exactness, Rule, Tables};

fn egglog_block(md: &str) -> String {
    let mut out = String::new();
    let mut inside = false;
    for line in md.lines() {
        if line.trim_start().starts_with("```") {
            inside = !inside && line.contains("egglog");
            continue;
        }
        if inside {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn reduction(exprs: &[(Tree, Vec<String>)], rules: &[&Rule], tables: &Tables, hits: &mut BTreeMap<usize, usize>) -> usize {
    let caller = CallerFacts::default();
    let mut nodes = 0usize;
    for (tree, inputs) in exprs {
        let cfg = Config {
            inputs,
            caller: &caller,
            mode: LitMode::F64,
            search: Search::Beam(8),
            max_steps: 64,
            admit: Exactness::Finite,
            computed_literals: true,
            fold_in_rounds: true,
        };
        let out = run(tree, rules, &tables.guards, &cfg);
        nodes += out.best.node_count();
        for (r, k) in out.hits {
            *hits.entry(r).or_insert(0) += k;
        }
    }
    nodes
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (corpus, proposals) = args.split_first().expect("usage: judge_rules exprs.tsv proposal.md ..");
    let text = std::fs::read_to_string(corpus).expect("read corpus");
    let exprs: Vec<(Tree, Vec<String>)> = text
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .filter_map(|(m, v)| {
            let inputs = v.split(',').filter(|x| !x.is_empty()).map(str::to_string).collect();
            Tree::parse(m).ok().map(|t| (t, inputs))
        })
        .collect();

    let mut tables = Tables::standard().expect("standard tables");
    let n_standard = tables.rules.len();
    let existing: Vec<String> = tables.rules.iter().map(|r| r.text.clone()).collect();
    let mut source_of: BTreeMap<usize, String> = BTreeMap::new();

    for path in proposals {
        let md = std::fs::read_to_string(path).expect("read proposal");
        let forms = match parse_all(&egglog_block(&md)) {
            Ok(f) => f,
            Err(e) => {
                println!("{path}: the egglog block does not parse as s-expressions: {e}");
                continue;
            }
        };
        let (mut admitted, mut refused, mut unreadable, mut duplicate) = (0, 0, 0, 0);
        let mut by_level: BTreeMap<String, usize> = BTreeMap::new();
        let mut reasons: Vec<String> = Vec::new();
        for form in forms.iter().filter(|f| f.head() != Some("ruleset")) {
            let one = show(form);
            let parsed = match read(&[("candidate", &one)]) {
                Ok(p) => p,
                Err(e) => {
                    unreadable += 1;
                    reasons.push(format!("UNREADABLE {one}\n      {}", e.lines().next().unwrap_or("")));
                    continue;
                }
            };
            for r in parsed.refused {
                refused += 1;
                reasons.push(format!("REFUSED    {one}\n      {}", r.reason));
            }
            for d in parsed.drafts {
                // Same pattern and template as a rule we already run?
                if existing.iter().any(|t| t.split(":ruleset").next() == one.split(":ruleset").next()) {
                    duplicate += 1;
                    continue;
                }
                match classify(&d, tables.rules.len()) {
                    Ok(rule) => {
                        admitted += 1;
                        *by_level.entry(format!("{:?}/{:?}", rule.order, rule.exactness)).or_insert(0) += 1;
                        source_of.insert(rule.id, path.clone());
                        tables.rules.push(rule);
                    }
                    Err(reason) => {
                        refused += 1;
                        reasons.push(format!("REFUSED    {one}\n      {}", reason.chars().take(200).collect::<String>()));
                    }
                }
            }
        }
        println!(
            "\n=== {path}\n    forms {} | admitted {admitted} {by_level:?} | refused {refused} | unreadable {unreadable} | already present {duplicate}",
            forms.len()
        );
        for r in &reasons {
            println!("    {r}");
        }
    }

    let standard: Vec<&Rule> = tables.rules.iter().take(n_standard).collect();
    let all: Vec<&Rule> = tables.rules.iter().collect();
    let input_nodes: usize = exprs.iter().map(|(t, _)| t.node_count()).sum();
    let mut h0 = BTreeMap::new();
    let mut h1 = BTreeMap::new();
    let base = reduction(&exprs, &standard, &tables, &mut h0);
    let with = reduction(&exprs, &all, &tables, &mut h1);
    println!(
        "\n=== corpus: {} expressions, {input_nodes} nodes\n    crate's rules      -> {base} nodes (removes {})\n    + admitted candidates -> {with} nodes (removes {}; {} more)",
        exprs.len(),
        input_nodes - base,
        input_nodes - with,
        base - with
    );
    let mut fired: Vec<(usize, usize)> = h1.iter().filter(|(r, _)| **r >= n_standard).map(|(r, k)| (*k, *r)).collect();
    fired.sort_by(|a, b| b.cmp(a));
    println!("    candidates that fired: {} of {}", fired.len(), tables.rules.len() - n_standard);
    for (k, r) in &fired {
        let rule = &tables.rules[*r];
        println!("    {k:>7}  {:?}/{:?}  {}", rule.order, rule.exactness, rule.text);
    }
}
