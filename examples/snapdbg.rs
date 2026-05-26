fn main() {
    // what Math does karva_to_terms produce for head=[mul] tail=[0.0796, x]?
    for c in gamakast::snap_karva::snap_variants(r#"(Mul (Num 0.0796) (Var "x"))"#, 16, 1e-3).unwrap() {
        println!("  rust: {} {}", c.cost, c.expr);
    }
}
