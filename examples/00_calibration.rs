//! Phase 1.0 calibration example: drive egglog 2.0 from Rust end-to-end.
//!
//! Run with: `cargo run --example 00_calibration`
//!
//! Prints a handful of boolean-algebra simplifications. Each line is one
//! round-trip: Rust string -> egglog -> saturate -> extract -> Rust string.

fn main() -> Result<(), String> {
    let cases = [
        // identity: (And a T) -> a
        r#"(And (Var "a") (T))"#,
        // double negation: (Not (Not a)) -> a
        r#"(Not (Not (Var "a")))"#,
        // absorption: (Or a (And a b)) -> a
        r#"(Or (Var "a") (And (Var "a") (Var "b")))"#,
        // De Morgan then double-negation collapse:
        // (Not (And (Not a) (Not b))) -> (Or a b)
        r#"(Not (And (Not (Var "a")) (Not (Var "b"))))"#,
    ];

    for case in cases {
        let out = gamakast::calibration::simplify(case)?;
        println!("{case}\n  => {out}\n");
    }

    println!("calibration OK");
    Ok(())
}
