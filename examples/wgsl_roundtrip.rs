//! One random tree of the kingdom's typed sampler through the round trip the
//! completeness test runs (`safe_reader` tests,
//! `random_sampler_trees_read_back_to_their_own_text`): its s-expression,
//! its text, the checker's verdict, the reader's error or the text read back
//! with the first differing lines.
//!
//! `cargo run --release --features wgsl --example wgsl_roundtrip -- <seed> [--tree] [--write <dir>]`
//! (`--write`: both texts, `<seed>.drawn.wgsl` and `<seed>.again.wgsl`, for a diff)

use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::render::{dual_ty, gram_role, sample};
use fuller::wgsl::safe_karva::render_tree;
use fuller::wgsl::safe_reader::read_safe;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seed: u64 = args.first().ok_or("usage: wgsl_roundtrip <seed> [--tree]")?.parse().map_err(|e| format!("seed: {e}"))?;
    let k = WgslKingdom::load_safe();
    let role = gram_role();
    let tree = sample(&k, &role, dual_ty(&k, "kernel.safe")?, 7, seed)?;
    if args.iter().any(|a| a == "--tree") {
        println!("{}\n", tree.to_sexpr());
    }
    let text = render_tree(&tree, &k)?.text;
    println!("{text}");
    let checker = fuller::wgsl::check::Checker::new(&k, &role, &fuller::wgsl::target::Target::recorded());
    println!("// check: {:?}", checker.check(&tree).map_err(|e| e.to_string()));
    match read_safe(&text, &k) {
        Err(e) => println!("// read: {e}"),
        Ok(back) => {
            let again = render_tree(&back.tree, &k)?.text;
            if let Some(dir) = args.iter().position(|a| a == "--write").and_then(|i| args.get(i + 1)) {
                std::fs::write(format!("{dir}/{seed}.drawn.wgsl"), &text).map_err(|e| e.to_string())?;
                std::fs::write(format!("{dir}/{seed}.again.wgsl"), &again).map_err(|e| e.to_string())?;
            }
            if again == text {
                println!("// read back to the same text");
            } else {
                for (i, (a, b)) in text.lines().zip(again.lines()).enumerate().filter(|(_, (a, b))| a != b).take(6) {
                    println!("// line {i}\n//   drawn: {a}\n//   again: {b}");
                }
                if text.lines().count() != again.lines().count() {
                    println!("// lines: drawn {}, again {}", text.lines().count(), again.lines().count());
                }
            }
        }
    }
    Ok(())
}
