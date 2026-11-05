//! Check one Futhark-kingdom program against every stored case of its problem, on the host interpreter (no GPU).
//!
//!   cargo run --release --example futhark_check_program -- <problem> '<s-expression of the view, as build6 reads it>'
//!
//! Prints the rendered Futhark program, then one line a case: its name, its split and whether the interpreter's output
//! equals the stored output exactly. The last line is the count. Exits non-zero when any case differs.

use std::sync::Arc;

use fuller::futhark_sig::{build6, kingdom6, load_cases, problem, signature, Interp};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [pname, text] = args.as_slice() else {
        return Err("usage: futhark_check_program <problem> '<s-expression>'".into());
    };
    let p = problem(pname)?;
    let k = Arc::new(kingdom6(p.name)?);
    let sig = signature(p.name)?;
    let interp = Interp::new6(&k, &sig)?;
    let tree = build6(&k, text)?;
    println!("{}", k.program(&tree)?.text);
    let cases = load_cases(&p)?;
    let mut exact = 0;
    for c in &cases {
        let out = interp.eval(&tree, &c.inputs)?;
        let ok = out == *c.answer()?;
        exact += usize::from(ok);
        println!("{}\t{:?}\t{}", c.name, c.split, if ok { "exact" } else { "DIFFERS" });
    }
    println!("exact on {exact} of {} cases", cases.len());
    if exact == cases.len() {
        Ok(())
    } else {
        Err(format!("{} cases differ", cases.len() - exact))
    }
}
