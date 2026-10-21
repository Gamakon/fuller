//! THE CANONICAL AST (2026-10-13, Andrew's bar: random AST → emit → parse
//! → AST, equal to the starting AST). The grammar spells some kernels more
//! than one way, and the text the emitter writes cannot tell those spellings
//! apart; the parser reads each such text as one of them. `canonical` maps
//! any tree to that one spelling, structurally (it never emits or parses),
//! so the sampler draws only trees a text can express and the round trip
//! is tested against a tree, not a text. Each rule names what the text does
//! not record:
//!
//! 1. A block's grouping. The emitter writes a nested block's statements in
//!    place (no braces), nothing for a statement's own terminal and nothing
//!    after a `return`; the parser groups a
//!    statement list as one block of up to four, more than four as the
//!    first three and a `stmt.block` of the rest.
//! 2. The split between the body and the epilogue. The emitter writes the
//!    epilogue's statements after the body's, in place; the parser keeps
//!    only the last statement (the one that stores h) as the epilogue and
//!    reads the ones before it as the body's, at the body's rung. After a
//!    body that returns at the top level no epilogue is written: it is the
//!    own terminal. The first item loop that continues the item counter
//!    (`loop.items_cont_*`, written `for (; t < ..)`) continues it from 0:
//!    it is the item loop that starts it.
//! 3. The grid. The header writes the grid the driver dispatches (the
//!    workgroup cut to the device, the block factors snapped) and the
//!    declarations the workgroup plan (the arrays the statements use, at
//!    their declared or default lengths, halved to fit); the tile depth
//!    wrapper (`grid.with_ktile`) writes nothing. The parser reads one grid
//!    row and one list of the arrays the statements use, by row name (a
//!    kernel using only the original four arrays declares them at their
//!    original lengths: the plan is empty, the list is not).
//! 4. A literal's value. The emitter writes the value of the literal's form
//!    (a block factor snapped to {1, 2, 4, 8}, a count rounded and clamped,
//!    a flag 0 or 1); a knob reference (`literal.p_bcol`) as the value its
//!    kernel's knobs give it ([`params::to_literals`]), so it reads back as
//!    its dual's plain literal row; a vector literal as its scalar
//!    splatted (`vec2<f32>(x)` is shape.splat2 of x).
//! 5. An own terminal value. The emitter writes it as its zero (a literal 0, a
//!    vector splatted from 0, a matrix of such columns), which the parser
//!    reads as those rows, conformed to the demanded dual as it conforms
//!    any literal (a dual with no literal row keeps its own terminal).
//! 6. A workgroup reduction's count. The emitter rounds it down to a power
//!    of two, and one of 1 writes only its barrier: a barrier.
//! 7. A thread's origin. The emitter writes `gid.x * bx`, `wid.x * W bx +
//!    lid.x` from the grid: the builtins and block factor the row carries
//!    are not written, the grid's are. A literal offset within the block
//!    is written snapped into 1..=64 and below the block factor, none at 0.
//! 8. A strided loop stepping by 1 is written as the loop that steps by 1;
//!    an index bound that is a uniform value (literals of 1 to 4,096, the
//!    uniform's fields, the loop counters, under the uniform's operators) as
//!    that uniform value; a
//!    loop over an expression bound that is the bound's own terminal, as the
//!    literal tile loop over the own terminal's value.
//! 9. A negated matrix is written as the matrix times -1.0; degrees and
//!    radians as the products by their constants; a band (`x / max(y, 1u)`)
//!    as the division it reads back as; the role's input load (`load.phi`,
//!    `load.phi4`, `phi[i]`) as the kingdom's load it reads back as. A literal index
//!    into a private array is written masked to the array (`v & 15u`,
//!    below its length); a literal lane of a vector, masked (`v & 3u`). An
//!    `if` with an empty else is written as the if;
//!    a breaking loop whose body is empty, as the loop; an unrolled item
//!    loop whose body returns or continues, as the item loop.
//!    A fully unrolled loop (`loop.unroll`) is written as its copies, each
//!    with its counter a literal (the copy's), unless the copies would pass
//!    64 or a continue sits below the top of the body (then as its loop); a
//!    tile counter outside its loop is written 0. A counter written as a
//!    literal v is read by its parent's row: an offset by it as the offset
//!    by v, its index as v, its uniform as the uniform literal v (none for
//!    0). A lane of a vector built in place (a compose or a splat) is that
//!    lane's expression: naga flattens the constructors, and the parser
//!    folds the lane the same way. The item index is `t` in an
//!    item loop's body (the loop's counter) and 0 outside one.
//! 10. Then the parameters, as the parser assigns them ([`params::assign`]):
//!     every literal whose value a knob writes exactly, a reference to that
//!     knob, the tile depth declared on the grid when one is referenced.

use crate::wgsl::loader::WgslKingdom;
use crate::geneframe::Ty;
use crate::wgsl::params;
use crate::wgsl::render::{array_len, cont_below_top, dual_ty, holds_continue, grid_of, parr_var, snap, wgmem_var, workgroup_dim, workgroup_plan, workgroup_shape, Tree, BLK, UNROLL};

/// The rung suffix of a row name (`_L1`, `_C2`, `_O0`, ...), if it has one.
fn rung_of(name: &str) -> Option<&str> {
    let (_, r) = name.rsplit_once('_')?;
    let b = r.as_bytes();
    (b.len() == 2 && matches!(b[0], b'L' | b'C' | b'O' | b'B') && b[1].is_ascii_digit()).then_some(r)
}

/// Whether `name` is a plain block row (`block.one_L1`, `block.seq3_C0`).
fn is_plain_block(name: &str) -> bool {
    ["block.one_", "block.seq2_", "block.seq3_", "block.seq4_"].iter().any(|p| name.starts_with(p))
}

/// The `own.*` terminal of a dual (the test oracle: a dual without one is our software's bug).
fn own(kingdom: &WgslKingdom, ty: Ty) -> Tree {
    crate::wgsl::render::own_leaf(kingdom, ty).unwrap_or_else(|e| panic!("{e}"))
}

/// The statements a block writes, in order: nested plain blocks spliced in
/// place, an empty block (its own terminal) nothing, and nothing after a return.
fn statements(block: &Tree, out: &mut Vec<Tree>) -> bool {
    if block.name.starts_with("own.") {
        return false;
    }
    for s in &block.kids {
        // A statement's own terminal writes nothing.
        if s.name.starts_with("own.") {
            continue;
        }
        let nested = s.name.starts_with("stmt.block_") && s.kids.len() == 1 && (is_plain_block(&s.kids[0].name) || s.kids[0].name.starts_with("own."));
        if nested {
            if statements(&s.kids[0], out) {
                return true;
            }
            continue;
        }
        out.push(s.clone());
        if s.name.starts_with("stmt.return_") {
            return true;
        }
    }
    false
}

/// An epilogue block's statements as the parser groups them: at most
/// three conditional statements before its last, the leading ones nested
/// as one block statement of the conditional rung, the two before the last
/// kept.
fn epilogue_items(mut items: Vec<Tree>, rung: &str, kingdom: &WgslKingdom) -> Vec<Tree> {
    let Some(level) = rung.strip_prefix('O') else { return items };
    if items.len() <= 4 {
        return items;
    }
    let c = format!("C{level}");
    let (Ok(cst), Ok(cbt)) = (dual_ty(kingdom, &format!("stmt.{c}")), dual_ty(kingdom, &format!("block.{c}"))) else { return items };
    let last = items.pop().expect("more than four");
    let keep = items.split_off(items.len() - 2);
    let lead = grouped(items, &c, &own(kingdom, cbt), Some(cst));
    [vec![Tree::new(&format!("stmt.block_{c}"), cst, vec![lead])], keep, vec![last]].concat()
}

/// A statement list as the parser groups it at `rung`.
fn grouped(mut items: Vec<Tree>, rung: &str, like: &Tree, stmt_ty: Option<Ty>) -> Tree {
    if items.is_empty() {
        return like.clone();
    }
    // One statement that is a block of this rung (a continue's) is that block.
    if let [one] = items.as_slice() {
        if one.name == format!("stmt.block_{rung}") && one.kids.len() == 1 && one.kids[0].ty == like.ty {
            return one.kids[0].clone();
        }
    }
    if items.len() > 4 {
        let rest = items.split_off(3);
        let inner = grouped(rest, rung, like, stmt_ty);
        let st = stmt_ty.unwrap_or(items[0].ty);
        items.push(Tree::new(&format!("stmt.block_{rung}"), st, vec![inner]));
    }
    let name = match items.len() {
        1 => "one",
        2 => "seq2",
        3 => "seq3",
        _ => "seq4",
    };
    Tree::new(&format!("block.{name}_{rung}"), like.ty, items)
}

/// `t` as a node of the dual `demand`, row by row: the row of the same stem
/// at the demanded rung, each child converted to that row's input dual
/// (a child already of it kept as it is). `None` when a row has no such
/// counterpart.
fn convert(t: &Tree, demand: Ty, kingdom: &WgslKingdom) -> Option<Tree> {
    if t.ty == demand {
        return Some(t.clone());
    }
    if t.name.starts_with("own.") {
        return crate::wgsl::render::own_leaf(kingdom, demand).ok();
    }
    // A uniform bound under either bound dual (a conditional loop's, a
    // body loop's): the same uniform.
    if t.name == "bound.x_of_uniform" || t.name == "bound.of_uniform" {
        let name = if kingdom.dual(demand)?.dual == "u32.boundu" { "bound.of_uniform" } else { "bound.x_of_uniform" };
        return kingdom.rows.iter().any(|r| r.name == name && r.output == demand).then(|| Tree::new(name, demand, t.kids.clone()));
    }
    let rung = kingdom.dual(demand)?.dual.split_once('.')?.1.to_string();
    let old = rung_of(&t.name)?;
    let stem = t.name.strip_suffix(old)?.strip_suffix('_')?;
    let name = format!("{stem}_{rung}");
    let row = kingdom.rows.iter().find(|r| r.name == name && r.output == demand && r.inputs.len() == t.kids.len())?;
    let kids = t.kids.iter().zip(&row.inputs).map(|(k, &d)| convert(k, d, kingdom)).collect::<Option<Vec<_>>>()?;
    Some(Tree::new(&name, demand, kids))
}

/// A kernel's body and epilogue as the parser splits them (rule 2).
fn split_epilogue(grid: Tree, body: &Tree, epilogue: &Tree, kingdom: &WgslKingdom, like: &Tree) -> Tree {
    let (Ok(l0), Ok(b0), Ok(o0)) = (dual_ty(kingdom, "stmt.L0"), dual_ty(kingdom, "block.L0"), dual_ty(kingdom, "block.O0")) else { return like.clone() };
    let mut items = Vec::new();
    let returns = statements(body, &mut items);
    // The first item loop that continues the counter starts it.
    let mut seen = false;
    for s in items.iter_mut() {
        if !seen {
            if let Some(rest) = s.name.strip_prefix("loop.items_cont_") {
                let plain = format!("loop.items_{rest}");
                if kingdom.rows.iter().any(|r| r.name == plain && r.output == s.ty && r.inputs.len() == s.kids.len()) {
                    s.name = plain;
                }
            }
        }
        seen |= s.name.starts_with("loop.items");
    }
    let body_block = |items: Vec<Tree>| grouped(items, "L0", &own(kingdom, b0), Some(l0));
    if returns {
        return Tree::new(&like.name, like.ty, vec![grid, body_block(items), own(kingdom, o0)]);
    }
    let mut epi = Vec::new();
    statements(epilogue, &mut epi);
    let Some(last) = epi.pop() else { return like.clone() };
    // A conditional statement with no body-rung spelling (a loop over a
    // bound that is not uniform) is written in place, read as one block of
    // conditional statements at the body's rung (stmt.cblock), consecutive
    // ones together.
    let (Ok(c0), Ok(cb)) = (dual_ty(kingdom, "stmt.C0"), dual_ty(kingdom, "block.C0")) else { return like.clone() };
    let mut run: Vec<Tree> = Vec::new();
    let flush = |run: &mut Vec<Tree>, items: &mut Vec<Tree>| {
        if !run.is_empty() {
            items.push(Tree::new("stmt.cblock_L0", l0, vec![grouped(std::mem::take(run), "C0", &own(kingdom, cb), Some(c0))]));
        }
    };
    for s in epi {
        match convert(&s, l0, kingdom) {
            Some(x) => {
                flush(&mut run, &mut items);
                items.push(x);
            }
            None if s.ty == c0 => run.push(s),
            None => return like.clone(),
        }
    }
    flush(&mut run, &mut items);
    let epilogue = grouped(vec![last], "O0", &own(kingdom, o0), dual_ty(kingdom, "stmt.O0").ok());
    Tree::new(&like.name, like.ty, vec![grid, body_block(items), epilogue])
}

/// A literal's value as the emitter writes it for its form (rule 4).
fn literal_value(form: &str, v: f64) -> f64 {
    let whole = |lo: f64, hi: f64| if v.is_finite() { v.round().clamp(lo, hi) } else { lo };
    match form {
        "index" | "count" | "bits" | "hash32" | "code4" | "code8" | "code16" | "code1024" => {
            if v.is_finite() {
                v.clamp(0.0, 4_294_967_295.0) as u32 as f64
            } else {
                0.0
            }
        }
        "wgdim" => f64::from(workgroup_dim(v)),
        "lit64" => whole(1.0, 64.0),
        "lit4096" => whole(1.0, 4096.0),
        "blk" => f64::from(snap(&BLK, v)),
        "unroll" => f64::from(snap(&UNROLL, v)),
        "int" | "q15_16" => v.clamp(-2_147_483_648.0, 2_147_483_647.0) as i32 as f64,
        "flag" | "sign" => f64::from(u8::from(v != 0.0)),
        _ => v,
    }
}

/// A literal leaf canonical (rule 4): a knob reference as its dual's plain
/// literal row, the value as written.
fn canonical_literal(t: &Tree, kingdom: &WgslKingdom) -> Tree {
    let Some(d) = kingdom.dual(t.ty) else { return t.clone() };
    let form = d.form.as_str();
    let name = if t.name.starts_with("literal.p_") {
        let plain = format!("literal.{form}");
        if kingdom.rows.iter().any(|r| r.name == plain && r.output == t.ty) {
            plain
        } else {
            t.name.clone()
        }
    } else {
        t.name.clone()
    };
    let form = name.strip_prefix("literal.").filter(|f| !f.starts_with("p_")).unwrap_or(form);
    let v = literal_value(form, t.konst);
    // A vector literal is written as its scalar splatted (`vec2<f32>(x)`).
    if d.lanes > 1 && d.slot.starts_with("vec") {
        let splat = format!("shape.splat{}", d.lanes);
        if let Some(r) = kingdom.rows.iter().find(|r| r.name == splat && r.output == t.ty && r.inputs.len() == 1) {
            let s = r.inputs[0];
            if kingdom.rows.iter().any(|r| r.name == name && r.output == s) {
                return Tree::new(&splat, t.ty, vec![Tree::literal(&name, s, v)]);
            }
        }
    }
    Tree::literal(&name, t.ty, v)
}

/// The grid canonical (rule 3), for the kernel `root` (its statements
/// already canonical).
fn canonical_grid(grid: &Tree, root: &Tree, kingdom: &WgslKingdom) -> Option<Tree> {
    let g = grid_of(grid).ok()?;
    let lit = |n: &str, d: &str, v: u32| -> Option<Tree> { Some(Tree::literal(n, dual_ty(kingdom, d).ok()?, v as f32)) };
    let (x, y) = workgroup_shape(g.x, g.y);
    let gt = dual_ty(kingdom, "grid.grid").ok()?;
    let base = Tree::new(if g.rows_x { "grid.of_rows_x" } else { "grid.of" }, gt, vec![lit("literal.wgdim", "u32.wgdim", x)?, lit("literal.wgdim", "u32.wgdim", y)?, lit("literal.blk", "u32.blk", g.bx)?, lit("literal.blk", "u32.blk", g.by)?]);
    let mut plan = workgroup_plan(grid, root).ok()?;
    if plan.is_empty() {
        // The original four arrays, declared as they always were: the ones
        // the statements use, at their original lengths.
        fn uses(t: &Tree, kingdom: &WgslKingdom, out: &mut std::collections::BTreeMap<String, (String, u32)>) {
            let array = t.kids.is_empty() && (t.name.starts_with("wg.") || (t.name.starts_with("own.") && kingdom.dual(t.ty).is_some_and(|d| d.dual.starts_with("wgmem<"))));
            if let Some(v) = array.then(|| wgmem_var(t).ok()).flatten() {
                out.insert(v.to_string(), (String::new(), array_len(v)));
            }
            t.kids.iter().for_each(|k| uses(k, kingdom, out));
        }
        root.kids.iter().skip(1).for_each(|k| uses(k, kingdom, &mut plan));
        if plan.is_empty() {
            return Some(base);
        }
    }
    let (decl_ty, decls_ty) = (dual_ty(kingdom, "wg.decl").ok()?, dual_ty(kingdom, "wg.decls").ok()?);
    let mut decls: Vec<(String, Tree)> = Vec::new();
    for (var, (_, len)) in &plan {
        let which = format!("wg.{}", var.strip_prefix("wg_")?);
        let (row, arr) = if which.starts_with("wg.au") {
            ("wg.decl_au", "wgmem<atomic<u32>>.index")
        } else if which.starts_with("wg.ai") {
            ("wg.decl_ai", "wgmem<atomic<i32>>.int")
        } else if which.starts_with("wg.v2") {
            ("wg.decl_v2", "wgmem<vec2<f32>>.real")
        } else if which.starts_with("wg.u") {
            ("wg.decl_u", "wgmem<u32>.index")
        } else if which.starts_with("wg.n") {
            ("wg.decl_n", "wgmem<i32>.int")
        } else if which.starts_with("wg.v") {
            ("wg.decl_v4", "wgmem<vec4<f32>>.real")
        } else {
            ("wg.decl_f", "wgmem<f32>.real")
        };
        let decl = Tree::new(row, decl_ty, vec![Tree::new(&which, dual_ty(kingdom, arr).ok()?, vec![]), lit("literal.wglen", "u32.wglen", *len)?]);
        decls.push((which, decl));
    }
    decls.sort_by(|a, b| a.0.cmp(&b.0));
    let mut list: Option<Tree> = None;
    for (_, d) in decls.into_iter().rev() {
        list = Some(match list {
            None => Tree::new("wg.decls_one", decls_ty, vec![d]),
            Some(rest) => Tree::new("wg.decls_cons", decls_ty, vec![d, rest]),
        });
    }
    Some(Tree::new("grid.with_wg", gt, vec![base, list?]))
}

/// The tile counter row a loop at `depth` (enclosing tile loops) counts on.
fn counter_at(depth: usize) -> Option<&'static str> {
    ["safe.tile_counter", "safe.tile_counter2", "safe.tile_counter3", "safe.tile_counter4"].get(depth).copied()
}

/// Rule 9: unrolled loops written as their copies, counters as the text has
/// them. `depth` counts the enclosing tile loops, `copies` the copies the
/// enclosing unrolled loops make, `fixed` the counters a copy fixed.
fn loops(t: &Tree, depth: usize, copies: u32, fixed: &[(String, u32)], items: bool, kingdom: &WgslKingdom) -> Tree {
    // A counter: the copy's literal, 0 outside its loop, else itself.
    let value = |c: &Tree| -> Option<u32> {
        if c.name.starts_with("own.") {
            return Some(0);
        }
        if let Some((_, v)) = fixed.iter().find(|(n, _)| *n == c.name) {
            return Some(*v);
        }
        let need = ["safe.tile_counter", "safe.tile_counter2", "safe.tile_counter3", "safe.tile_counter4"].iter().position(|n| *n == c.name)?;
        (depth <= need).then_some(0)
    };
    let index = || dual_ty(kingdom, "u32.index").ok();
    let stem = t.name.rsplit_once('_').filter(|(_, r)| rung_of(&format!("x_{r}")).is_some()).map_or(t.name.as_str(), |(s, _)| s);
    // A parent of a counter written as a literal.
    if let [c] = t.kids.as_slice() {
        if let Some(v) = value(c) {
            if t.name.starts_with("safe.tile_index") {
                if let Some(ix) = index() {
                    return Tree::literal("literal.index", ix, v as f32);
                }
            }
            if t.name.starts_with("uniform.of_tcounter") {
                if v == 0 {
                    return own(kingdom, t.ty);
                }
                if let Ok(l) = dual_ty(kingdom, "u32.lit4096") {
                    return Tree::new("uniform.of_lit", t.ty, vec![Tree::literal("literal.lit4096", l, v as f32)]);
                }
            }
        }
    }
    if let [origin, c] = t.kids.as_slice() {
        let offset = if t.name.starts_with("safe.row_plus_tile") { Some("safe.row_plus_x") } else if t.name.starts_with("safe.col_plus_tile") { Some("safe.col_plus_x") } else { None };
        if let (Some(base), Some(v), Some(ix)) = (offset, value(c), index()) {
            let name = if t.name.ends_with("_s") { format!("{base}_s") } else { base.to_string() };
            if let Some(r) = kingdom.rows.iter().find(|r| r.name == name && r.output == t.ty && r.inputs.len() == 2) {
                // The origin under the offset row's own dual for it.
                let o = Tree { ty: r.inputs[0], ..loops(origin, depth, copies, fixed, items, kingdom) };
                return Tree::new(&name, t.ty, vec![o, Tree::literal("literal.index", ix, v as f32)]);
            }
        }
    }
    let is_loop = stem.starts_with("loop.tile") || stem == "loop.unroll";
    if stem == "loop.unroll" {
        if let ([n, body], Some(rung)) = (t.kids.as_slice(), rung_of(&t.name)) {
            let count = if n.name.starts_with("literal.") { n.konst.round().clamp(1.0, 64.0) as u32 } else { 1 };
            let total = copies.saturating_mul(count);
            let tile = format!("loop.tile_{rung}");
            if total > crate::wgsl::render::UNROLL_COPIES_MAX || cont_below_top(body, true) {
                if kingdom.rows.iter().any(|r| r.name == tile && r.output == t.ty) {
                    return Tree::new(&tile, t.ty, vec![n.clone(), loops(body, depth + 1, copies, fixed, items, kingdom)]);
                }
            } else if let (Some(counter), Ok(st), Ok(bt)) = (counter_at(depth), dual_ty(kingdom, &format!("stmt.{rung}")), dual_ty(kingdom, &format!("block.{rung}"))) {
                let mut raw = Vec::new();
                for i in 0..count {
                    let mut f = fixed.to_vec();
                    f.push((counter.to_string(), i));
                    let copy = loops(body, depth + 1, total, &f, items, kingdom);
                    // A continue at the top of a copy: the rest runs when its
                    // guard does not hold (`if (!(c)) { rest }`).
                    let copy = match (copy.name.strip_prefix("block.cont_"), copy.kids.as_slice()) {
                        (Some(r), [c, rest]) => match (dual_ty(kingdom, "bool.flag"), dual_ty(kingdom, &format!("stmt.{r}"))) {
                            (Ok(b), Ok(sr)) => {
                                let not = Tree::new("logic.not", b, vec![c.clone()]);
                                let iff = Tree::new(&format!("stmt.if_{r}"), sr, vec![not, rest.clone()]);
                                Tree::new(&format!("block.one_{r}"), copy.ty, vec![iff])
                            }
                            _ => copy,
                        },
                        _ => copy,
                    };
                    if statements(&copy, &mut raw) {
                        break;
                    }
                }
                // At an epilogue rung only the last statement is the
                // epilogue's; those before it are conditional statements.
                let cond = rung.strip_prefix('O').map(|l| format!("stmt.C{l}"));
                let last = raw.len().saturating_sub(1);
                let mut items = Vec::new();
                for (i, s) in raw.iter().enumerate() {
                    let target = match &cond {
                        Some(c) if i < last => dual_ty(kingdom, c).ok(),
                        _ => Some(st),
                    };
                    match target.and_then(|d| convert(s, d, kingdom)) {
                        Some(x) => items.push(x),
                        None => return t.clone(),
                    }
                }
                return Tree::new(&format!("stmt.block_{rung}"), st, vec![grouped(epilogue_items(items, rung, kingdom), rung, &own(kingdom, bt), Some(st))]);
            }
        }
    }
    // The item counter: `t` in an item loop's body, 0 outside it.
    if t.name == "builtin.item" || (t.name == "safe.counter_index" && t.kids.first().is_some_and(|c| c.name == "safe.counter")) {
        let counter = dual_ty(kingdom, "u32.counter").ok().map(|c| Tree::new("safe.counter", c, vec![]));
        return match (items, index(), counter) {
            (true, _, Some(c)) => Tree::new("safe.counter_index", t.ty, vec![c]),
            (false, Some(ix), _) => Tree::literal("literal.index", ix, 0.0),
            _ => t.clone(),
        };
    }
    if t.name == "uniform.of_counter" && !items {
        return own(kingdom, t.ty);
    }
    let in_items = items || stem.starts_with("loop.items");
    let last = t.kids.len().saturating_sub(1);
    let kids = t.kids.iter().enumerate().map(|(i, k)| loops(k, if is_loop && i == last { depth + 1 } else { depth }, copies, fixed, in_items && (i == last || items), kingdom)).collect();
    Tree { kids, ..t.clone() }
}

/// Rule 7: every thread origin in `t` as the grid writes it.
fn origins(t: &Tree, g: &crate::wgsl::render::Grid, kingdom: &WgslKingdom) -> Tree {
    let kids: Vec<Tree> = t.kids.iter().map(|k| origins(k, g, kingdom)).collect();
    let b = |name: &str, k: &Tree| Tree::new(name, k.ty, vec![]);
    let blk = |k: &Tree, v: u32| Tree::literal("literal.blk", k.ty, v as f32);
    let (row, by, bx) = (t.name.contains("_row"), g.by, g.bx);
    let f = if row { by } else { bx };
    let kids = match (t.name.as_str(), kids.as_slice()) {
        ("safe.thread_row" | "safe.thread_col", [gid, k]) => vec![b("builtin.global_invocation_id", gid), blk(k, f)],
        ("safe.thread_row_s" | "safe.thread_col_s", [wid, lid, k]) => vec![b("builtin.workgroup_id", wid), b("builtin.local_invocation_id", lid), blk(k, f)],
        ("safe.thread_row_g" | "safe.thread_col_g", [wid, lid, k, n]) => {
            // G clamped into 1..=B: one group is the plain origin
            // (`wid * W B + lid * B` is `gid * B`), B groups the strided one.
            let g = if n.name.starts_with("literal.") { (n.konst.round().max(1.0) as u32).min(f.max(1)) } else { f.max(1) };
            let (plain, strided) = if row { ("safe.thread_row", "safe.thread_row_s") } else { ("safe.thread_col", "safe.thread_col_s") };
            // (Under its own row's dual; the conform of the whole tree places it.)
            let own = |name: &str| kingdom.rows.iter().find(|r| r.name == name).map(|r| r.output);
            if let (1, Some(ty), Ok(gid)) = (g, own(plain), dual_ty(kingdom, "vec3<u32>.gid")) {
                return Tree::new(plain, ty, vec![Tree::new("builtin.global_invocation_id", gid, vec![]), blk(k, f)]);
            }
            if let (true, Some(ty)) = (g == f, own(strided)) {
                return Tree::new(strided, ty, vec![b("builtin.workgroup_id", wid), b("builtin.local_invocation_id", lid), blk(k, f)]);
            }
            vec![b("builtin.workgroup_id", wid), b("builtin.local_invocation_id", lid), blk(k, f), Tree::literal(&n.name, n.ty, g as f32)]
        }
        // A literal offset within the block: written snapped into 1..=64 and
        // below the block factor (`min(v, B - 1)`, the k-th row of a strided
        // block at k * W), so read back as that offset, none at 0
        // (2026-10-17: 45 of the 889 random trees that missed the AST drawn).
        ("safe.row_plus_s" | "safe.col_plus_s" | "safe.row_plus" | "safe.col_plus", [origin, off]) if off.kids.is_empty() && (off.name.starts_with("own.") || off.name.starts_with("literal.")) => {
            let v = if off.name.starts_with("own.") || !off.konst.is_finite() { 1 } else { (off.konst.round() as i64).clamp(1, 64) as u32 };
            let v = v.min(f.saturating_sub(1));
            if v == 0 {
                return origin.clone();
            }
            let lit = if off.name.starts_with("own.") { "literal.lit64" } else { off.name.as_str() };
            vec![origin.clone(), Tree::literal(lit, off.ty, v as f32)]
        }
        // A grouped offset: its groups are its origin's (the emitter reads
        // them there), the origin read in its grouped form (a strided origin
        // is B groups of one cell).
        ("safe.row_plus_g" | "safe.col_plus_g", [origin, n, grp, e]) => {
            let grouped = if row { "safe.thread_row_g" } else { "safe.thread_col_g" };
            let groups = match (origin.name.as_str(), origin.kids.as_slice()) {
                (x, [.., gk]) if x == grouped => gk.konst as u32,
                _ => f.max(1),
            };
            let origin = match (origin.name.as_str(), origin.kids.as_slice()) {
                (x, [wid, lid, k]) if x.ends_with("_s") => match kingdom.rows.iter().find(|r| r.name == grouped && r.output == origin.ty).or_else(|| kingdom.rows.iter().find(|r| r.name == grouped)) {
                    Some(r) => Tree::new(grouped, r.output, vec![wid.clone(), lid.clone(), k.clone(), Tree::literal(&n.name, n.ty, groups as f32)]),
                    None => origin.clone(),
                },
                _ => origin.clone(),
            };
            vec![origin, Tree::literal(&n.name, n.ty, groups as f32), grp.clone(), e.clone()]
        }
        _ => kids,
    };
    Tree { kids, ..t.clone() }
}

/// The canonical spelling of `t` under the interface `role` (see the module
/// header for the rules; the interface's own rows conform as the parser's).
pub fn canonical(t: &Tree, kingdom: &WgslKingdom, role: &crate::wgsl::render::Role) -> Tree {
    // A module of entry points: each entry point's own spelling (each is
    // written and read as a kernel of its own; the module adds none).
    if t.name.starts_with("kernel.module_") {
        return Tree { kids: t.kids.iter().map(|k| canonical(k, kingdom, role)).collect(), ..t.clone() };
    }
    let per_kernel = crate::wgsl::safe_karva::per_kernel_rows(kingdom, role);
    // Knob references first, at the drawn grid's knobs (rule 4).
    let t = params::to_literals(t, kingdom).unwrap_or_else(|_| t.clone());
    let c = degrees(&structural(&t, kingdom, &per_kernel), kingdom);
    // Every node under the dual its parent's row demands, the rows chosen as
    // the parser chooses them (it conforms the tree it reads, last).
    let c = crate::wgsl::safe_karva::conform(&c, c.ty, kingdom, &per_kernel).unwrap_or(c);
    params::assign(&c, kingdom).unwrap_or(c)
}

/// Rules 1 to 4.
fn structural(t: &Tree, kingdom: &WgslKingdom, per_kernel: &[crate::wgsl::safe_karva::PerKernelRow]) -> Tree {
    if t.kids.is_empty() && t.name.starts_with("literal.") {
        return canonical_literal(t, kingdom);
    }
    if t.name.starts_with("own.") || t.name.starts_with("const.") {
        if let Some(z) = canonical_own(t, kingdom, per_kernel) {
            return z;
        }
    }
    let kids: Vec<Tree> = t.kids.iter().map(|k| structural(k, kingdom, per_kernel)).collect();
    let t = Tree { kids, ..t.clone() };
    if t.name == "kernel.entry_safe" {
        if let [grid, body, epilogue] = t.kids.as_slice() {
            let split = split_epilogue(grid.clone(), body, epilogue, kingdom, &t);
            // The moved statements regrouped with the body's (rule 1).
            let kids: Vec<Tree> = split.kids.iter().map(|k| canonical_block(k, kingdom)).collect();
            let root = Tree { kids, ..split };
            let grid = canonical_grid(&root.kids[0], &root, kingdom).unwrap_or_else(|| root.kids[0].clone());
            let mut kids = root.kids.clone();
            for k in kids.iter_mut().skip(1) {
                // (Unrolled loops spliced, then the blocks regrouped; an
                // epilogue that is the own terminal after a body that returns
                // is not written, and stays so.)
                if !k.name.starts_with("own.") {
                    *k = structural(&loops(k, 0, 1, &[], false, kingdom), kingdom, per_kernel);
                }
            }
            if let Ok(g) = grid_of(&grid) {
                for k in kids.iter_mut().skip(1) {
                    *k = origins(k, &g, kingdom);
                }
            }
            kids[0] = grid;
            return Tree { kids, ..root };
        }
    }
    canonical_block(&t, kingdom)
}

/// Rule 5: an own terminal value leaf as the parser reads the zero written for it.
fn canonical_own(t: &Tree, kingdom: &WgslKingdom, per_kernel: &[crate::wgsl::safe_karva::PerKernelRow]) -> Option<Tree> {
    let d = kingdom.dual(t.ty)?;
    let ty = |n: &str| dual_ty(kingdom, n).ok();
    let gid = || -> Option<Tree> { Some(Tree::new("builtin.global_invocation_id", ty("vec3<u32>.gid")?, vec![])) };
    let blk = || -> Option<Tree> { Some(Tree::literal("literal.blk", ty("u32.blk")?, 1.0)) };
    let origin = |name: &str, dual: &str| -> Option<Tree> {
        let kids = if name.ends_with("_s") { vec![Tree::new("builtin.workgroup_id", ty("vec3<u32>.wid")?, vec![]), Tree::new("builtin.local_invocation_id", ty("vec3<u32>.lid")?, vec![]), blk()?] } else { vec![gid()?, blk()?] };
        Some(Tree::new(name, ty(dual)?, kids))
    };
    let cell = || -> Option<Tree> { Some(Tree::new("safe.cell_of", ty("u32.cell")?, vec![origin("safe.thread_row", "u32.threadrow")?, origin("safe.thread_col", "u32.threadcol")?])) };
    // An epilogue block: the store of f0 to the thread's first cell.
    if let Some(level) = d.dual.strip_prefix("block.O") {
        let f0 = Tree::new("reg.get_f", ty("f32.real")?, vec![Tree::new("reg.f0", ty("reg<f32>.real")?, vec![])]);
        let store = Tree::new(&format!("store.h_O{level}"), ty(&format!("stmt.O{level}"))?, vec![cell()?, f0]);
        return Some(Tree::new(&format!("block.one_O{level}"), t.ty, vec![store]));
    }
    // A loop bound's own terminal is the loop rule's (rule 8).
    if d.is_statement_level() || d.dual.starts_with("u32.bound") {
        return None;
    }
    // A thread's origin or cell: the thread's own (the emitter's own_own terminal).
    match d.dual.as_str() {
        "u32.threadrow" | "u32.rowbase" => return origin("safe.thread_row", "u32.threadrow"),
        "u32.threadcol" | "u32.colbase" => return origin("safe.thread_col", "u32.threadcol"),
        "u32.threadrows" | "u32.threadrowg" | "u32.rowbases" | "u32.rowbaseg" => return origin("safe.thread_row_s", "u32.threadrows"),
        "u32.threadcols" | "u32.threadcolg" | "u32.colbases" | "u32.colbaseg" => return origin("safe.thread_col_s", "u32.threadcols"),
        "u32.cell" => return cell(),
        // A uniform's zero is the uniform dual's own terminal, as the parser reads it.
        "u32.uniform" if t.name.starts_with("own.") => return Some(t.clone()),
        // A thread id's axis: the builtin's component the renderer writes for it.
        "u32.rowid" => return Some(Tree::new("axis.y_gid", t.ty, vec![gid()?])),
        "u32.colid" => return Some(Tree::new("axis.x_gid", t.ty, vec![gid()?])),
        _ => {}
    }
    // An array: the one the emitter names for the dual.
    if d.dual.starts_with("parr<") {
        let name = format!("parr.{}", parr_var(t).ok()?.strip_prefix('p')?);
        return Some(Tree::new(&name, t.ty, vec![]));
    }
    if d.dual.starts_with("wgmem<") {
        let name = format!("wg.{}", wgmem_var(t).ok()?.strip_prefix("wg_")?);
        return Some(Tree::new(&name, t.ty, vec![]));
    }
    // The value written: a u32 or bool dual's own value, else 0.
    let written = match d.scalar() {
        // (A `const` row writes the value it carries.)
        _ if t.name.starts_with("const.") => t.konst as f32,
        Some("u32") => d.own_bits() as f32,
        Some("bool") => f32::from(u8::from(d.own_bits() != 0)),
        _ => 0.0,
    };
    let scalar = |s: &str| -> Option<Tree> {
        let (lit, dual) = match s {
            "f32" => ("literal.real", "f32.real"),
            "u32" => ("literal.index", "u32.index"),
            "i32" => ("literal.int", "i32.int"),
            "bool" => ("literal.flag", "bool.flag"),
            _ => return None,
        };
        Some(Tree::literal(lit, dual_ty(kingdom, dual).ok()?, written))
    };
    let splat = |lanes: u32, s: &str| -> Option<Tree> {
        let z = scalar(s)?;
        let out = dual_ty(kingdom, &format!("vec{lanes}<{s}>.{}", kingdom.dual(z.ty)?.form)).ok()?;
        Some(Tree::new(&format!("shape.splat{lanes}"), out, vec![z]))
    };
    let zero = if let Some((c, r)) = d.matrix_shape() {
        let cols = (0..c).map(|_| splat(r, "f32")).collect::<Option<Vec<_>>>()?;
        let name = if c == 2 { "shape.compose_mat".to_string() } else { format!("shape.compose_mat{c}") };
        Tree::new(&name, t.ty, cols)
    } else if d.lanes > 1 && d.slot.starts_with("vec") {
        splat(d.lanes, d.scalar()?)?
    } else if d.lanes <= 1 {
        scalar(d.scalar()?)?
    } else {
        return None;
    };
    crate::wgsl::safe_karva::conform(&zero, t.ty, kingdom, per_kernel).ok().filter(|z| !z.name.starts_with("own.") && !z.name.starts_with("const.")).or_else(|| {
        // A safe index (a workgroup or private array's): the zero through
        // the safe row the parser reads it with (`safe.wga_of(0u)`).
        match kingdom.rows.iter().find(|r| r.name.starts_with("safe.") && r.output == t.ty && r.inputs.as_slice() == [zero.ty]) {
            Some(r) => Some(Tree::new(&r.name, t.ty, vec![zero])),
            // Otherwise the zero as written: the conform of the whole tree
            // reads it under the row its parent takes, as the parser does.
            None => Some(zero),
        }
    })
}

/// Rule 6 at a reduction (its children already canonical).
fn canonical_reduction(t: &Tree, kingdom: &WgslKingdom) -> Option<Tree> {
    let stem = t.name.strip_prefix("wg.reduce_")?;
    let rung = rung_of(stem)?;
    let (n, arrays) = t.kids.split_last()?;
    if !(1..=2).contains(&arrays.len()) {
        return None;
    }
    let v = if n.name.starts_with("own.") { 1 } else { n.konst.max(1.0) as u32 };
    let p = 1u32 << (31 - v.leading_zeros());
    if p == 1 {
        return Some(Tree::new(&format!("stmt.barrier_{rung}"), t.ty, vec![]));
    }
    let lit = Tree::literal("literal.lit4096", dual_ty(kingdom, "u32.lit4096").ok()?, p as f32);
    Some(Tree::new(&t.name, t.ty, arrays.iter().cloned().chain(std::iter::once(lit)).collect()))
}

/// Rule 8 at a strided tile or item loop (its children already canonical).
fn canonical_stride(t: &Tree, kingdom: &WgslKingdom) -> Option<Tree> {
    // An item loop stepping by 1 (`loop.items_strided_boundd`, `_cont_`).
    if let (true, [bound, step, body]) = (t.name.starts_with("loop.items_") && t.name.contains("strided_"), t.kids.as_slice()) {
        let name = t.name.replacen("strided_", "", 1);
        if step.name.starts_with("literal.") && step.konst == 1.0 && kingdom.rows.iter().any(|r| r.name == name && r.output == t.ty) {
            return Some(Tree::new(&name, t.ty, vec![bound.clone(), body.clone()]));
        }
        return None;
    }
    let (stem, rung) = t.name.rsplit_once('_').filter(|(_, r)| rung_of(&format!("x_{r}")).is_some())?;
    // (And a stepped loop that breaks, 2026-10-17: stepping by 1 it is the
    // loop that breaks.)
    let plain = if stem == "loop.tilex_strided_brk" { "loop.tilex_brk" } else { stem.strip_suffix("_strided").filter(|s| matches!(*s, "loop.tile" | "loop.tilex"))? };
    let [bound, step, body] = t.kids.as_slice() else { return None };
    // (The step is a lit64 on a tile loop, a lit4096 on a tilex loop since 2026-10-16.)
    if !matches!(step.name.as_str(), "literal.lit64" | "literal.lit4096") || step.konst != 1.0 {
        return None;
    }
    let name = format!("{plain}_{rung}");
    kingdom.rows.iter().any(|r| r.name == name && r.output == t.ty).then(|| Tree::new(&name, t.ty, vec![bound.clone(), body.clone()]))
}

/// Rule 9 at degrees, radians and a band: the rows their text reads as.
fn canonical_written_as(t: &Tree, kingdom: &WgslKingdom) -> Option<Tree> {
    match (t.name.as_str(), t.kids.as_slice()) {
        ("index.band_of", [x, y]) => kingdom.rows.iter().any(|r| r.name == "index.div" && r.output == t.ty).then(|| Tree::new("index.div", t.ty, vec![x.clone(), y.clone()])),
        _ => None,
    }
}

/// Rule 9 at degrees and radians, after the structural rules (their
/// constant is a scalar in the text, `x * 57.29..`, under the value's dual,
/// which a vector literal's rule would splat).
fn degrees(t: &Tree, kingdom: &WgslKingdom) -> Tree {
    let kids: Vec<Tree> = t.kids.iter().map(|k| degrees(k, kingdom)).collect();
    if let ("trig.degrees" | "trig.radians", [x]) = (t.name.as_str(), kids.as_slice()) {
        let k = if t.name == "trig.degrees" { 57.295_78_f32 } else { 0.017_453_292_f32 };
        if kingdom.rows.iter().any(|r| r.name == "arith.mul" && r.output == t.ty && r.inputs.as_slice() == [t.ty, t.ty]) {
            return Tree::new("arith.mul", t.ty, vec![x.clone(), Tree::literal("literal.real", t.ty, k)]);
        }
    }
    Tree { kids, ..t.clone() }
}

/// Rule 9 at a negated matrix.
fn canonical_neg_mat(t: &Tree, kingdom: &WgslKingdom) -> Option<Tree> {
    let [m] = t.kids.as_slice() else { return None };
    if t.name != "arith.neg_mat" {
        return None;
    }
    let real = dual_ty(kingdom, "f32.real").ok()?;
    kingdom.rows.iter().any(|r| r.name == "arith.mul_mat_scalar" && r.output == t.ty).then(|| Tree::new("arith.mul_mat_scalar", t.ty, vec![m.clone(), Tree::literal("literal.real", real, -1.0)]))
}

/// Rule 8 at a loop over an own terminal expression bound.
fn canonical_own_bound(t: &Tree, kingdom: &WgslKingdom) -> Option<Tree> {
    let (stem, rung) = t.name.rsplit_once('_').filter(|(_, r)| rung_of(&format!("x_{r}")).is_some())?;
    let tile = stem.strip_prefix("loop.tilex").map(|rest| format!("loop.tile{rest}_{rung}"))?;
    let bound = t.kids.first().filter(|b| b.name.starts_with("own."))?;
    // A step wider than the short loop's 64 stays expression-bounded.
    if t.kids.len() == 3 && t.kids[1].konst > 64.0 {
        return None;
    }
    let v = kingdom.dual(bound.ty)?.own_bits() as f32;
    if !(1.0..=64.0).contains(&v) {
        return None;
    }
    let row = kingdom.rows.iter().find(|r| r.name == tile && r.output == t.ty && r.inputs.len() == t.kids.len())?;
    let mut kids = t.kids.clone();
    kids[0] = Tree::literal("literal.lit64", row.inputs[0], v);
    Some(Tree::new(&tile, t.ty, kids))
}

/// Rule 8 at an index bound that is a uniform value: a literal in 1..=4,096,
/// or literals, the uniform's fields, the workgroup id and the loop counters
/// under the operators the uniform rows have (`23u + 52u`, 2026-10-17: 32 of
/// the 894 random trees that missed the AST drawn), as the parser reads it
/// ([`uniform_value`]).
fn canonical_bound(t: &Tree, kingdom: &WgslKingdom) -> Option<Tree> {
    let [x] = t.kids.as_slice() else { return None };
    if t.name != "bound.of_index" {
        return None;
    }
    let u = uniform_value(x, kingdom).filter(|u| dual_ty(kingdom, "u32.uniform").is_ok_and(|d| u.ty == d))?;
    kingdom.rows.iter().any(|r| r.name == "bound.x_of_uniform" && r.output == t.ty).then(|| Tree::new("bound.x_of_uniform", t.ty, vec![u]))
}

/// An index as the uniform value the parser reads it as
/// (`safe_reader`'s `uniform_of`, over the drawn tree's rows): a product or
/// a shift under u32.uniformw, which no bound takes; None when any part is
/// not uniform.
fn uniform_value(t: &Tree, kingdom: &WgslKingdom) -> Option<Tree> {
    let u = dual_ty(kingdom, "u32.uniform").ok()?;
    let uw = dual_ty(kingdom, "u32.uniformw").ok()?;
    let leaf = |name: &str| Some(Tree::new(name, u, vec![]));
    match (t.name.as_str(), t.kids.as_slice()) {
        ("load.cfg_m", []) => leaf("uniform.cfg_m"),
        ("load.cfg_d", []) => leaf("uniform.cfg_d"),
        ("load.cfg_r_rows", []) => leaf("uniform.cfg_r_rows"),
        ("load.cfg_r_off", []) => leaf("uniform.cfg_r_off"),
        ("builtin.subgroup_size", []) => leaf("uniform.subgroup_size"),
        ("builtin.num_subgroups", []) => leaf("uniform.num_subgroups"),
        ("index.of_count" | "index.of_uniform", [k]) => uniform_value(k, kingdom),
        ("safe.counter_index" | "safe.tile_index" | "safe.tile_index2" | "safe.tile_index3" | "safe.tile_index4", [c]) => {
            let row = match c.name.as_str() {
                "safe.counter" => "uniform.of_counter",
                "safe.tile_counter" => "uniform.of_tcounter",
                "safe.tile_counter2" => "uniform.of_tcounter2",
                "safe.tile_counter3" => "uniform.of_tcounter3",
                "safe.tile_counter4" => "uniform.of_tcounter4",
                _ => return None,
            };
            Some(Tree::new(row, u, vec![c.clone()]))
        }
        (n, []) if n.starts_with("literal.") && crate::wgsl::params::parse_ref(n).is_none() => {
            let v = t.konst;
            if v == 0.0 {
                return Some(own(kingdom, u));
            }
            let l = dual_ty(kingdom, "u32.lit4096").ok()?;
            (v.fract() == 0.0 && (1.0..=4096.0).contains(&v)).then(|| Tree::new("uniform.of_lit", u, vec![Tree::literal("literal.lit4096", l, v)]))
        }
        ("index.clamp", [a, b, c]) => {
            let (a, b, c) = (uniform_value(a, kingdom)?, uniform_value(b, kingdom)?, uniform_value(c, kingdom)?);
            (a.ty == u && b.ty == u && c.ty == u).then(|| Tree::new("uniform.clamp", u, vec![a, b, c]))
        }
        (n, [a, b]) => {
            let op = match n {
                "index.mod" => "mod",
                "index.round_down" => "round_down",
                "index.band_of" => "band_of",
                "index.clamp_to" => "clamp_to",
                "index.add" | "count.add" => "add",
                "index.sub_floor" | "count.sub_floor" => "sub_floor",
                "index.mul" => "mul",
                "index.div" => "div",
                "index.min" | "count.min" => "min",
                "index.max" | "count.max" => "max",
                "index.shl" | "bits.shl" => "shl",
                "index.shr" | "bits.shr" => "shr",
                "index.and" | "bits.and" => "and",
                "index.or" | "bits.or" => "or",
                _ => return None,
            };
            let (a, b) = (uniform_value(a, kingdom)?, uniform_value(b, kingdom)?);
            if a.ty == uw || b.ty == uw {
                return None;
            }
            Some(Tree::new(&format!("uniform.{op}"), if matches!(op, "mul" | "shl") { uw } else { u }, vec![a, b]))
        }
        _ => None,
    }
}

/// Rule 1 at `t` itself (its children already canonical).
/// Rule 9 at a uniform test whose body needs its rung (a barrier or an item
/// loop in it): the test as the parser spells its comparison, simple_test's
/// rows (`x == 0` is `x < 1`, `x != 0` and `x > 0` are `1 <= x`, `>` and `>=`
/// their operands swapped), a workgroup-uniform predicate of one comparison
/// as that test's row (2026-10-17: the uniform tests drawn).
fn canonical_uniform_test(t: &Tree, kingdom: &WgslKingdom) -> Option<Tree> {
    let rung = rung_of(&t.name)?;
    let stem = &t.name[..t.name.len() - rung.len() - 1];
    let (has_else, rest) = match stem.strip_prefix("stmt.if_else_") {
        Some(r) => (true, r),
        None => (false, stem.strip_prefix("stmt.if_")?),
    };
    let (op, x, y, bodies): (&str, Tree, Tree, &[Tree]) = match (rest, t.kids.as_slice()) {
        ("uflag", [f, bodies @ ..]) => {
            let op = f.name.strip_prefix("uflag.").filter(|o| matches!(*o, "eq" | "ne" | "lt" | "le" | "gt" | "ge"))?;
            let [x, y] = f.kids.as_slice() else { return None };
            (op, x.clone(), y.clone(), bodies)
        }
        ("ueq" | "une" | "ult" | "ule", [x, y, bodies @ ..]) => (&rest[1..], x.clone(), y.clone(), bodies),
        _ => return None,
    };
    if bodies.len() != usize::from(has_else) + 1 {
        return None;
    }
    // (The parser keeps the rung only when a body needs it; else the test is
    // a plain if at the conditional rung, which this rule does not spell.)
    fn needs_rung(t: &Tree) -> bool {
        let n = t.name.as_str();
        ["stmt.barrier", "stmt.storage_barrier", "stmt.subgroup_barrier", "wg.reduce_", "wg.put", "wg.uload_", "loop.items"].iter().any(|p| n.starts_with(p)) || t.kids.iter().any(needs_rung)
    }
    if !bodies.iter().any(needs_rung) {
        // A body that does not need its rung: the plain if the parser reads,
        // `if (a op b)` over the uniform values as indices, its bodies at the
        // conditional rung (L0 is C0, L(d) is C(d - 1) below it). Except
        // `<=` with no else, which the parser keeps as the uniform test.
        if rest == "ule" && !has_else {
            return None;
        }
        return plain_if(t, op, &x, &y, bodies, kingdom);
    }
    let u = dual_ty(kingdom, "u32.uniform").ok()?;
    let zero = |t: &Tree| t.name.starts_with("own.") || (t.name == "uniform.of_lit" && t.kids.first().is_some_and(|l| l.konst == 0.0));
    let one = || Tree::new("uniform.of_lit", u, vec![Tree::literal("literal.lit4096", dual_ty(kingdom, "u32.lit4096").ok().unwrap_or(u), 1.0)]);
    let (row, a, b) = match op {
        "eq" if zero(&y) => ("ult", x, one()),
        "eq" if zero(&x) => ("ult", y, one()),
        "ne" | "gt" if zero(&y) => ("ule", one(), x),
        "ne" | "lt" if zero(&x) => ("ule", one(), y),
        "le" => ("ule", x, y),
        "ge" => ("ule", y, x),
        "eq" => ("ueq", x, y),
        "ne" => ("une", x, y),
        "lt" => ("ult", x, y),
        _ => ("ult", y, x),
    };
    let name = format!("stmt.if{}_{row}_{rung}", if has_else { "_else" } else { "" });
    kingdom.rows.iter().any(|r| r.name == name).then(|| Tree::new(&name, t.ty, [vec![a, b], bodies.to_vec()].concat()))
}

/// [`canonical_uniform_test`]'s plain if.
fn plain_if(t: &Tree, op: &str, x: &Tree, y: &Tree, bodies: &[Tree], kingdom: &WgslKingdom) -> Option<Tree> {
    let rung = rung_of(&t.name)?;
    let (has_else, ty) = (bodies.len() == 2, t.ty);
    if !rung.starts_with('L') {
        return None;
    }
    let flag = dual_ty(kingdom, "bool.flag").ok()?;
    let index = dual_ty(kingdom, "u32.index").ok()?;
    let ix = |t: &Tree| Tree::new("index.of_uniform", index, vec![t.clone()]);
    let cond = Tree::new(&format!("compare.{op}"), flag, vec![ix(x), ix(y)]);
    // A row's rung moved down: L0 to C0, L(d) to C(d - 1); None where the
    // kingdom has no such row.
    fn down(t: &Tree, kingdom: &WgslKingdom) -> Option<Tree> {
        let kids = t.kids.iter().map(|k| down(k, kingdom)).collect::<Option<Vec<_>>>()?;
        let Some(r) = rung_of(&t.name) else { return Some(Tree { kids, ..t.clone() }) };
        let d: u32 = r[1..].parse().ok()?;
        let to = match (&r[..1], d) {
            ("L", 0) => "C0".to_string(),
            ("L", d) if d >= 2 => format!("C{}", d - 1),
            ("C", d) => format!("C{d}"),
            _ => return None,
        };
        let name = format!("{}_{to}", &t.name[..t.name.len() - r.len() - 1]);
        let ty = if t.name.starts_with("own.") { t.ty } else { kingdom.rows.iter().find(|row| row.name == name).map(|row| row.output)? };
        Some(Tree { name, ty, konst: t.konst, kids })
    }
    // (An own terminal body, the empty block, keeps its leaf under its new dual.)
    let body = |b: &Tree| -> Option<Tree> {
        if b.name.starts_with("own.") {
            let bt = dual_ty(kingdom, &format!("block.C{}", &rung[1..])).ok()?;
            return Some(own(kingdom, bt));
        }
        down(b, kingdom)
    };
    let bodies: Vec<Tree> = bodies.iter().map(body).collect::<Option<_>>()?;
    let name = format!("stmt.if{}_{rung}", if has_else { "_else" } else { "" });
    kingdom.rows.iter().any(|r| r.name == name).then(|| Tree::new(&name, ty, [vec![cond], bodies].concat()))
}

fn canonical_block(t: &Tree, kingdom: &WgslKingdom) -> Tree {
    if let Some(u) = canonical_uniform_test(t, kingdom).filter(|u| u != t) {
        return canonical_block(&u, kingdom);
    }
    // A literal lane of a vector, masked as written (`(v) & 3u`).
    if t.name.contains("_v_lane") {
        let at = if t.name.starts_with("reg.") { 1 } else { 2 };
        if let Some(k) = t.kids.get(at).filter(|k| k.name.starts_with("literal.") && k.kids.is_empty() && k.konst.is_finite()) {
            let mut kids = t.kids.clone();
            kids[at] = Tree::literal(&k.name, k.ty, (k.konst.max(0.0) as u32 & 3) as f32);
            return Tree { kids, ..t.clone() };
        }
    }
    // A literal private index, masked as written.
    if let (true, [arr, idx, rest @ ..]) = (t.name.starts_with("parr.get_") || t.name.starts_with("parr.set_"), t.kids.as_slice()) {
        if let ([k], Ok(var)) = (idx.kids.as_slice(), parr_var(arr)) {
            let mask = match idx.name.as_str() {
                "safe.pidx64" => Some(63u32),
                "safe.pidx128" => Some(127u32),
                "safe.pidx16" => Some(15u32),
                _ => None,
            };
            if let (Some(mask), true) = (mask, k.name.starts_with("literal.") && k.konst.is_finite()) {
                let v = (k.konst.max(0.0) as u32 & mask).min(array_len(var).saturating_sub(1));
                let idx = Tree::new(&idx.name, idx.ty, vec![Tree::literal(&k.name, k.ty, v as f32)]);
                let kids = [vec![arr.clone(), idx], rest.to_vec()].concat();
                return Tree { kids, ..t.clone() };
            }
        }
    }
    // A breaking loop with an empty body (no break written) is the loop.
    if let (Some(rest), Some(body)) = (t.name.strip_prefix("loop.tile_brk_").map(|r| format!("loop.tile_{r}")).or_else(|| t.name.strip_prefix("loop.tilex_brk_").map(|r| format!("loop.tilex_{r}"))), t.kids.last()) {
        if body.name.starts_with("own.") {
            if let Some(r) = kingdom.rows.iter().find(|r| r.name == rest && r.output == t.ty && r.inputs.len() == t.kids.len()) {
                let mut kids = t.kids.clone();
                let last = kids.len() - 1;
                kids[last] = own(kingdom, r.inputs[last]);
                return Tree::new(&rest, t.ty, kids);
            }
        }
    }
    // An unrolled item loop whose body returns or continues: the item loop.
    if let (Some(kind), [_, bound, body]) = (t.name.strip_prefix("loop.items_unrolled_"), t.kids.as_slice()) {
        let mut items = Vec::new();
        let plain = format!("loop.items_{kind}");
        if (statements(body, &mut items) || holds_continue(body)) && kingdom.rows.iter().any(|r| r.name == plain && r.output == t.ty) {
            return Tree::new(&plain, t.ty, vec![bound.clone(), body.clone()]);
        }
    }
    // The role's input load: the kingdom's safe load the parser reads.
    if let Some(safe) = match t.name.as_str() {
        "load.phi" => Some("load.phi_safe"),
        "load.phi4" => Some("load.phi4_safe"),
        _ => None,
    } {
        if kingdom.rows.iter().any(|r| r.name == safe && r.output == t.ty) {
            return Tree::new(safe, t.ty, t.kids.clone());
        }
    }
    // An if with an empty else is the if.
    if let (Some(rung), [c, then, other]) = (t.name.strip_prefix("stmt.if_else_"), t.kids.as_slice()) {
        let name = format!("stmt.if_{rung}");
        if other.name.starts_with("own.") && kingdom.rows.iter().any(|r| r.name == name && r.output == t.ty) {
            return Tree::new(&name, t.ty, vec![c.clone(), then.clone()]);
        }
    }
    if let (Some(lane), [v]) = (t.name.strip_prefix("shape.access_").and_then(|i| i.parse::<usize>().ok()), t.kids.as_slice()) {
        if let Some(c) = crate::wgsl::safe_reader::component(v, lane, kingdom) {
            return c;
        }
    }
    if let Some(r) = canonical_stride(t, kingdom).or_else(|| canonical_bound(t, kingdom)).or_else(|| canonical_neg_mat(t, kingdom)).or_else(|| canonical_own_bound(t, kingdom)).or_else(|| canonical_written_as(t, kingdom)) {
        return r;
    }
    if let Some(r) = canonical_reduction(t, kingdom) {
        return r;
    }
    if is_plain_block(&t.name) {
        if let Some(rung) = rung_of(&t.name) {
            let mut items = Vec::new();
            statements(t, &mut items);
            let stmt_ty = dual_ty(kingdom, &format!("stmt.{rung}")).ok();
            return grouped(epilogue_items(items, rung, kingdom), rung, &own(kingdom, t.ty), stmt_ty);
        }
    }
    t.clone()
}
