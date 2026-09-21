//! Lint one Math expression and print the forms:  cargo run --example lint_probe -- '<math>' x_0,x_1 [positive]
use fuller::lint::tables::{Exactness, Tables};
use fuller::lint::{forms, DataFacts};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names: Vec<String> = args[1].split(',').map(str::to_string).collect();
    let positive = if args.get(2).is_some_and(|a| a == "positive") { names.clone() } else { Vec::new() };
    let tables = Tables::standard().expect("tables");
    let facts = DataFacts { rows: &[], positive_vars: positive.clone(), nonzero_vars: positive };
    for (tree, level) in forms(&tables, &args[0], &names, Exactness::Finite, 8, facts).expect("forms") {
        println!("{:>3} {:<8} {}", tree.node_count(), level, tree.to_infix());
    }
}
