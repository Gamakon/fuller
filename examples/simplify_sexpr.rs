//! Print the smallest form fuller finds for one Math s-expression, as
//! s-expression and as infix.
//!
//!   cargo run --release --example simplify_sexpr -- <file.sexpr> x_0,x_1,...
use fuller::extract::smallest_form;
use fuller::lint::node::Tree;

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("usage: simplify_sexpr <file.sexpr> <vars>")?;
    let vars: Vec<String> = std::env::args().nth(2).unwrap_or_default().split(',').filter(|v| !v.is_empty()).map(str::to_string).collect();
    let math = std::fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;
    let f = smallest_form(math.trim(), &vars, &[], &[])?;
    println!("nodes {} -> {} in {} rounds", f.input_cost, f.cost, f.rounds);
    println!("SEXPR\t{}", f.expr);
    let tree = Tree::parse(&f.expr)?;
    println!("INFIX\t{}", tree.to_infix());
    Ok(())
}
