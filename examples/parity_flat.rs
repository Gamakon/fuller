//! The kernel-shaped engine (`lint::flat`) against the tree engine
//! (`lint::engine`), over a corpus, at every exactness level. They must reach
//! the same form, and every form must be a canonical K-expression.
//!
//!   cargo run --release --example parity_flat -- exprs.tsv

use std::time::Instant;

use fuller::lint::engine::{fold, run, CallerFacts, Config, LitMode, Search};
use fuller::lint::flat::{run_greedy, Flat};
use fuller::lint::node::Tree;
use fuller::lint::pack::{arity, pack};
use fuller::lint::tables::{Exactness, Rule, Tables};

fn is_canonical(f: &Flat) -> bool {
    let mut next_free = 1u32;
    for n in &f.nodes {
        let a = arity(n.op) as u32;
        if (a >= 1 && n.arg0 != next_free) || (a >= 2 && n.arg1 != next_free + 1) {
            return false;
        }
        next_free += a;
    }
    next_free as usize == f.nodes.len()
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: parity_flat exprs.tsv");
    let text = std::fs::read_to_string(&path).expect("read exprs");
    let tables = Tables::standard().expect("tables load");
    let symbols = fuller::geneframe::master_table();
    let kingdom = symbols.kingdom("Symbolic Regression");
    let rules: Vec<&Rule> = tables.usable(&kingdom);
    let packed = pack(&rules);
    println!(
        "rules: {} usable, {} packed for the device, {} excluded",
        rules.len(),
        packed.rules.len(),
        packed.excluded.len()
    );
    for (id, why) in &packed.excluded {
        println!("  excluded: {} -- {why}", tables.rules[*id].text);
    }
    let caller = CallerFacts::default();
    for admit in [Exactness::Bit, Exactness::Rounding, Exactness::Finite] {
        let (mut n, mut differ, mut not_canonical, mut steps, mut max_steps) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let (mut t_tree, mut t_flat) = (0.0f64, 0.0f64);
        let mut shown = 0usize;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let (math, vars) = line.split_once('\t').expect("math<TAB>vars");
            let inputs: Vec<String> = vars.split(',').filter(|v| !v.is_empty()).map(str::to_string).collect();
            let Ok(tree) = Tree::parse(math) else { continue };
            n += 1;
            let cfg = Config {
                inputs: &inputs,
                caller: &caller,
                mode: LitMode::F64,
                search: Search::Greedy,
                max_steps: 64,
                admit,
                computed_literals: false,
                fold_in_rounds: false,
            };
            let t0 = Instant::now();
            let want = run(&tree, &rules, &tables.guards, &cfg).best;
            t_tree += t0.elapsed().as_secs_f64();
            let start = Flat::from_tree(&fold(tree.clone(), &inputs));
            let t1 = Instant::now();
            let got = run_greedy(&start, &packed.rules, &packed.codes, &tables.guards, &caller, admit, 64);
            t_flat += t1.elapsed().as_secs_f64();
            steps += got.steps;
            max_steps = max_steps.max(got.steps);
            if !is_canonical(&got.form) {
                not_canonical += 1;
            }
            if got.form.to_tree() != want {
                differ += 1;
                if shown < 5 {
                    shown += 1;
                    println!("  DIFFERS {math}\n     tree {}\n     flat {}", want.to_math(), got.form.to_tree().to_math());
                }
            }
        }
        println!(
            "{admit:?}: {n} expressions | differ {differ} | not canonical {not_canonical} | rewrites {steps} (max {max_steps} in one) | tree {:.3} ms/expr, flat {:.3} ms/expr",
            1e3 * t_tree / n as f64,
            1e3 * t_flat / n as f64
        );
    }
}
