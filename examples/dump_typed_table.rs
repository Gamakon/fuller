//! Print the TYPED symbol table (`geneframe::typed_depth_table`) as markdown —
//! the T1/T2 many-hot arity design — so it can sit beside the untyped `wide`
//! table the engine actually breeds from. Each op expands into one ROW PER
//! input-depth combination it accepts; a depth-raiser has no row that takes T2
//! in, which is the ceiling. Usage: dump_typed_table > typed.md
use fuller::geneframe::{typed_depth_table, Arity, Ty, TYPED_SR};
use std::fmt::Write as _;

fn ty(t: Ty) -> &'static str {
    match t {
        Ty::S => "S",
        Ty::I => "I",
        Ty::F => "F(d0)",
        Ty::B => "B",
        Ty::A => "A",
        Ty::L => "L",
        Ty::T1 => "T1(d1)",
        Ty::T2 => "T2(d2)",
        _ => "?", // NLP/Botji types — not used by the SR depth table
    }
}

fn sig(a: &Arity) -> (String, String) {
    let side = |m: &std::collections::BTreeMap<Ty, u32>| {
        if m.is_empty() {
            return "—".to_string();
        }
        m.iter().map(|(t, n)| if *n == 1 { ty(*t).to_string() } else { format!("{}×{}", n, ty(*t)) }).collect::<Vec<_>>().join(", ")
    };
    (side(&a.inputs), side(&a.outputs))
}

fn main() {
    let t = typed_depth_table();
    let rows = t.kingdom(TYPED_SR);
    let mut out = String::new();
    let _ = writeln!(out, "# The TYPED symbol table (`geneframe::typed_depth_table`, kingdom `{TYPED_SR}`)\n");
    let _ = writeln!(
        out,
        "This is the design in `fuller/src/geneframe.rs`. Each op becomes ONE ROW \
         PER input-depth combination it accepts, and its output type is the \
         resulting depth. A depth-raising function (sin, cos, sqrt, log, …) has \
         **no row that takes `T2` as input** — that missing row IS the depth-2 \
         ceiling, a *structural* fact of the arity signature, not a scalar check. \
         A gene built from THIS table cannot be depth-3, because no symbol \
         consumes a T2. **NOTE: the audit found this table is NOT wired into the \
         engine's population/vary path — `Engine::new` breeds from the untyped \
         `SymbolTable::wide` and applies `typed_depth` as a post-decode reject.**\n"
    );
    let _ = writeln!(out, "| symbol | op (semantic id) | inputs (typed) | output (typed) |");
    let _ = writeln!(out, "|-------:|------------------|----------------|----------------|");
    for s in &rows {
        let (i, o) = sig(&s.arity);
        let _ = writeln!(out, "| {} | `{}` | {} | {} |", s.symbol, s.semantic_id, i, o);
    }
    let _ = writeln!(out, "\nrows: {} (vs the untyped wide table's ~21: each op fans out into its typed depth-combinations)\n", rows.len());
    print!("{out}");
}
