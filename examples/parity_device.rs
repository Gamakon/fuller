//! The linter kernel on the GPU against the kernel-shaped CPU engine, over a
//! corpus, at every exactness level.
//!
//!   cargo run --release --features gpu --example parity_device -- exprs.tsv

use std::collections::BTreeMap;
use std::time::Instant;

use fuller::lint::device::{KernelTables, LintKernel, DEFAULT_ROUNDS};
use fuller::lint::engine::{fold, CallerFacts};
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
    let path = std::env::args().nth(1).expect("usage: parity_device exprs.tsv");
    let text = std::fs::read_to_string(&path).expect("read exprs");
    let tables = Tables::standard().expect("tables load");
    let symbols = fuller::geneframe::master_table();
    let kingdom = symbols.kingdom("Symbolic Regression");
    let rules: Vec<&Rule> = tables.usable(&kingdom);
    let packed = pack(&rules);
    let kt = KernelTables::new(&packed.rules, packed.codes.clone(), &tables.guards).expect("kernel tables");
    let t0 = Instant::now();
    let kernel = LintKernel::new(kt).expect("a GPU adapter");
    println!("kernel compiled and tables uploaded in {:.3} s", t0.elapsed().as_secs_f64());

    // One batch per column list: a batch shares its data columns.
    let mut batches: BTreeMap<String, Vec<Tree>> = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let (math, vars) = line.split_once('\t').expect("math<TAB>vars");
        if let Ok(tree) = Tree::parse(math) {
            batches.entry(vars.to_string()).or_default().push(tree);
        }
    }
    let caller = CallerFacts::default();
    for admit in [Exactness::Bit, Exactness::Rounding, Exactness::Finite] {
        let (mut n, mut differ, mut not_canonical, mut refused, mut steps) = (0usize, 0usize, 0usize, 0usize, 0u64);
        let (mut t_dev, mut t_cpu) = (0.0f64, 0.0f64);
        let mut shown = 0usize;
        for (vars, trees) in &batches {
            let vars: Vec<String> = vars.split(',').filter(|v| !v.is_empty()).map(str::to_string).collect();
            let starts: Vec<Flat> = trees
                .iter()
                .map(|t| Flat::from_tree_in(&fold(t.clone(), &vars), &vars).expect("columns"))
                .collect();
            let var_facts = vec![0u32; vars.len().max(1)];
            let t1 = Instant::now();
            let got = kernel.run(&starts, &vars, &var_facts, admit, DEFAULT_ROUNDS).expect("dispatch");
            t_dev += t1.elapsed().as_secs_f64();
            let t2 = Instant::now();
            let want: Vec<_> = starts
                .iter()
                .map(|s| run_greedy(s, &packed.rules, &packed.codes, &tables.guards, &caller, admit, DEFAULT_ROUNDS as usize))
                .collect();
            t_cpu += t2.elapsed().as_secs_f64();
            for ((start, dev), cpu) in starts.iter().zip(&got).zip(&want) {
                n += 1;
                let Some(form) = dev.form.as_ref() else {
                    refused += 1;
                    continue;
                };
                steps += u64::from(dev.steps);
                if !is_canonical(form) {
                    not_canonical += 1;
                }
                if form.to_tree() != cpu.form.to_tree() || dev.steps as usize != cpu.steps || dev.level != cpu.level {
                    differ += 1;
                    if shown < 5 {
                        shown += 1;
                        println!(
                            "  DIFFERS {}\n     cpu {} ({} steps, {:?})\n     gpu {} ({} steps, {:?})",
                            start.to_tree().to_math(),
                            cpu.form.to_tree().to_math(), cpu.steps, cpu.level,
                            form.to_tree().to_math(), dev.steps, dev.level
                        );
                    }
                }
            }
        }
        println!(
            "{admit:?}: {n} expressions in {} batches | differ {differ} | not canonical {not_canonical} | refused {refused} | rewrites {steps} | GPU {:.4} ms/expr ({:.2} s total incl. encode+readback) | CPU {:.4} ms/expr",
            batches.len(),
            1e3 * t_dev / n as f64,
            t_dev,
            1e3 * t_cpu / n as f64
        );
    }
}
