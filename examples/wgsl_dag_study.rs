//! What a WGSL kernel looks like as a DAG, through naga's IR: per function,
//! how many expressions the arena holds, how many are structurally distinct
//! once hash-consed (operands by canonical id, effectful results kept unique),
//! so how many are repeats a value-numbering pass would remove; and how many
//! statements carry effects (stores) or control flow (if, loop, switch).
//!
//!   cargo run --release --features wgsl --example wgsl_dag_study -- <file.wgsl>...
use std::collections::HashMap;

fn canonical_count(func: &naga::Function) -> (usize, usize, usize) {
    // (expressions, distinct, effectful)
    let mut canon: Vec<usize> = Vec::with_capacity(func.expressions.len());
    let mut interned: HashMap<String, usize> = HashMap::new();
    let mut effectful = 0usize;
    for (handle, expr) in func.expressions.iter() {
        let unique = matches!(
            expr,
            naga::Expression::Load { .. }
                | naga::Expression::CallResult(_)
                | naga::Expression::AtomicResult { .. }
                | naga::Expression::WorkGroupUniformLoadResult { .. }
                | naga::Expression::RayQueryProceedResult
                | naga::Expression::SubgroupBallotResult
                | naga::Expression::SubgroupOperationResult { .. }
        );
        if unique {
            effectful += 1;
        }
        // Debug prints operand handles as `[N]`; rewrite each to the operand's
        // canonical id so two identical computations over identical operands
        // spell the same.
        let dbg = format!("{expr:?}");
        let mut key = String::with_capacity(dbg.len());
        let mut rest = dbg.as_str();
        while let Some(i) = rest.find('[') {
            key.push_str(&rest[..i]);
            let tail = &rest[i + 1..];
            if let Some(j) = tail.find(']') {
                if let Ok(n) = tail[..j].parse::<usize>() {
                    let c = canon.get(n).copied().unwrap_or(usize::MAX);
                    key.push_str(&format!("<{c}>"));
                    rest = &tail[j + 1..];
                    continue;
                }
            }
            key.push('[');
            rest = tail;
        }
        key.push_str(rest);
        if unique {
            key.push_str(&format!("@{}", handle.index()));
        }
        let next = interned.len();
        let id = *interned.entry(key).or_insert(next);
        canon.push(id);
    }
    (func.expressions.len(), interned.len(), effectful)
}

fn walk(block: &naga::Block, counts: &mut [usize; 4]) {
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Store { .. } => counts[0] += 1,
            naga::Statement::If { accept, reject, .. } => {
                counts[1] += 1;
                walk(accept, counts);
                walk(reject, counts);
            }
            naga::Statement::Loop { body, continuing, .. } => {
                counts[2] += 1;
                walk(body, counts);
                walk(continuing, counts);
            }
            naga::Statement::Switch { cases, .. } => {
                counts[3] += 1;
                for c in cases {
                    walk(&c.body, counts);
                }
            }
            naga::Statement::Block(b) => walk(b, counts),
            _ => {}
        }
    }
}

fn main() -> Result<(), String> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        return Err("usage: wgsl_dag_study <file.wgsl>...".into());
    }
    println!("{:<28} {:<22} {:>6} {:>8} {:>7} {:>7} {:>6} {:>5} {:>5} {:>6}", "file", "function", "exprs", "distinct", "repeat", "effect", "store", "if", "loop", "switch");
    for path in &paths {
        let src = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        let module = naga::front::wgsl::parse_str(&src).map_err(|e| format!("{path}: {}", e.emit_to_string(&src)))?;
        let file = std::path::Path::new(path).file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
        let report = |name: &str, f: &naga::Function| {
            let (n, distinct, effectful) = canonical_count(f);
            let mut counts = [0usize; 4];
            walk(&f.body, &mut counts);
            println!("{file:<28} {name:<22} {n:>6} {distinct:>8} {:>7} {effectful:>7} {:>6} {:>5} {:>5} {:>6}", n - distinct, counts[0], counts[1], counts[2], counts[3]);
        };
        for (_, f) in module.functions.iter() {
            report(f.name.as_deref().unwrap_or("?"), f);
        }
        for ep in &module.entry_points {
            report(&format!("@{}", ep.name), &ep.function);
        }
    }
    Ok(())
}
