//! Run `smallest_form` over a file of expressions (one per line:
//! `math<TAB>v1,v2,...`) and print `input_cost<TAB>cost<TAB>seconds` per line,
//! in order, so a ruleset change can be measured against a stored baseline.
//!
//!   cargo run --release --example measure_smallest_form -- exprs.tsv

use std::time::Instant;

use fuller::extract::smallest_form;

fn main() {
    let path = std::env::args().nth(1).expect("usage: measure_smallest_form exprs.tsv");
    let text = std::fs::read_to_string(&path).expect("read exprs file");
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let (math, vars) = line.split_once('\t').expect("math<TAB>vars");
        let inputs: Vec<String> =
            vars.split(',').filter(|v| !v.is_empty()).map(str::to_string).collect();
        let t0 = Instant::now();
        match smallest_form(math, &inputs, &[], &[]) {
            Ok(f) => println!("{}\t{}\t{:.6}", f.input_cost, f.cost, t0.elapsed().as_secs_f64()),
            Err(e) => println!("ERR\t{e}\t{:.6}", t0.elapsed().as_secs_f64()),
        }
    }
}
