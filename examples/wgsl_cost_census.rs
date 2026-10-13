//! Cost census of the sampler's random kernels (2026-10-13): each kernel's
//! static cost against the fitness's bound, and for one over it, the
//! top-level statement whose removal lowers the cost most, with the largest
//! loop multiplier inside it and the loop row that gives it.
//!
//! The same for `sample_within` (the sampler under the bound): how many
//! are still over. `CENSUS_TREES=<file>` writes those trees, one a line.
//!
//! `cargo run --release --features wgsl --example wgsl_cost_census -- [n]`

use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::render::{cost, dual_ty, gram_role, sample, sample_within, worst_bound, Cost, Tree};
use fuller::wgsl::safe_karva::default_bound;
use std::collections::BTreeMap;

/// The largest loop trip count in `t` (its own, times the loops around it),
/// and the row of the loop that contributes the largest single factor.
fn largest_loop(t: &Tree, outer: f64, best: &mut (f64, String, f64)) {
    let lit = |k: Option<&Tree>, max: f64| k.map_or(1.0, |k| if k.name.starts_with("literal.") && k.konst.is_finite() { f64::from(k.konst).round().clamp(1.0, max) } else { 1.0 });
    let base = t.name.trim_end_matches(|c: char| c.is_ascii_digit()).trim_end_matches(['L', 'C', 'O', 'B']).trim_end_matches('_').to_string();
    let factor = if base.starts_with("loop.tilex") {
        t.kids.first().map_or(8192.0, worst_bound) / if base.contains("strided") { lit(t.kids.get(1), 64.0) } else { 1.0 }
    } else if base.starts_with("loop.tile") || base == "loop.unroll" {
        lit(t.kids.first(), 64.0) / if base.contains("strided") { lit(t.kids.get(1), 64.0) } else { 1.0 }
    } else {
        1.0
    };
    let here = outer * factor.max(1.0);
    if factor > 1.0 && here > best.0 {
        *best = (here, base.clone(), factor);
    }
    for k in &t.kids {
        largest_loop(k, here, best);
    }
}

/// The statements of a block (nested plain blocks opened).
fn statements(b: &Tree, out: &mut Vec<(Tree, Vec<usize>)>, path: Vec<usize>) {
    for (i, s) in b.kids.iter().enumerate() {
        let mut p = path.clone();
        p.push(i);
        if s.name.starts_with("stmt.block_") && s.kids.len() == 1 && s.kids[0].name.starts_with("block.") && !s.kids[0].name.starts_with("block.cont") {
            p.push(0);
            statements(&s.kids[0], out, p);
        } else {
            out.push((s.clone(), p));
        }
    }
}

fn replaced(t: &Tree, path: &[usize]) -> Tree {
    match path.split_first() {
        None => Tree::fallback(t.ty),
        Some((&i, rest)) => {
            let mut kids = t.kids.clone();
            kids[i] = replaced(&kids[i], rest);
            Tree { kids, ..t.clone() }
        }
    }
}

fn main() -> Result<(), String> {
    let n: u64 = std::env::args().nth(1).map_or(Ok(1000), |a| a.parse()).map_err(|e| format!("n: {e}"))?;
    let k = WgslKingdom::load_safe();
    let role = gram_role();
    let root = dual_ty(&k, "kernel.safe")?;
    let bound = default_bound(&k)?;
    let over = |c: &Cost| c.per_item > bound.per_item || c.base > bound.base;
    let (mut count_over, mut over_item, mut over_base) = (0, 0, 0);
    let mut drivers: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for seed in 0..n {
        let tree = sample(&k, &role, root, 7, seed)?;
        let c = cost(&tree);
        if !over(&c) {
            continue;
        }
        count_over += 1;
        over_item += usize::from(c.per_item > bound.per_item);
        over_base += usize::from(c.base > bound.base);
        // The statement whose removal lowers the cost most (body, kid 1; epilogue, kid 2).
        let mut stmts = Vec::new();
        for kid in [1usize, 2] {
            statements(&tree.kids[kid], &mut stmts, vec![kid]);
        }
        let worst = stmts
            .iter()
            .map(|(s, p)| {
                let without = cost(&replaced(&tree, p));
                ((c.per_item - without.per_item) / bound.per_item + (c.base - without.base) / bound.base, s.clone())
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
        let key = match worst {
            Some((_, s)) => {
                let mut best = (1.0, String::from("no loop"), 1.0);
                largest_loop(&s, 1.0, &mut best);
                let stem = s.name.trim_end_matches(|c: char| c.is_ascii_digit()).trim_end_matches(['L', 'C', 'O', 'B']).trim_end_matches('_').to_string();
                let size = if best.0 >= 1e6 { "1e6+" } else if best.0 >= 1e4 { "1e4..1e6" } else if best.0 >= 100.0 { "100..1e4" } else { "<100" };
                format!("{} | {stem} | loops x{size} (largest factor: {} x{})", if c.per_item > bound.per_item { "per item" } else { "base" }, best.1, best.2 as u64)
            }
            None => "no statement".into(),
        };
        let e = drivers.entry(key).or_insert((0, seed));
        e.0 += 1;
    }
    println!("{n} random kernels: {count_over} over the bound ({} per item, {} base); {over_item} over per item, {over_base} over base", bound.per_item, bound.base);
    let mut within_over = 0;
    let mut lines = Vec::new();
    for seed in 0..n {
        let tree = sample_within(&k, &role, root, 7, seed, &bound)?;
        within_over += usize::from(over(&cost(&tree)));
        lines.push(tree.to_sexpr());
    }
    println!("sample_within: {within_over} of {n} over the bound");
    if let Ok(path) = std::env::var("CENSUS_TREES") {
        std::fs::write(&path, lines.join("\n") + "\n").map_err(|e| format!("{path}: {e}"))?;
    }
    let mut v: Vec<_> = drivers.into_iter().collect();
    v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    for (k, (c, s)) in v.iter().take(30) {
        println!("{c:>4}  seed {s:>4}  {k}");
    }
    Ok(())
}
