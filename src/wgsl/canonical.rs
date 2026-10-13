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
//!    place (no braces) and nothing after a `return`; the parser groups a
//!    statement list as one block of up to four, more than four as the
//!    first three and a `stmt.block` of the rest.
//! 2. The split between the body and the epilogue. The emitter writes the
//!    epilogue's statements after the body's, in place; the parser keeps
//!    only the last statement (the one that stores h) as the epilogue and
//!    reads the ones before it as the body's, at the body's rung. After a
//!    body that returns at the top level no epilogue is written: it is the
//!    fallback. The first item loop that continues the item counter
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
//! 5. A fallback value. The emitter writes it as its zero (a literal 0, a
//!    vector splatted from 0, a matrix of such columns), which the parser
//!    reads as those rows, conformed to the demanded dual as it conforms
//!    any literal (a dual with no literal row keeps its fallback).
//! 6. A workgroup reduction's count. The emitter rounds it down to a power
//!    of two, and one of 1 writes only its barrier: a barrier.
//! 7. A thread's origin. The emitter writes `gid.x * bx`, `wid.x * W bx +
//!    lid.x` from the grid: the builtins and block factor the row carries
//!    are not written, the grid's are.
//! 8. Then the parameters, as the parser assigns them ([`params::assign`]):
//!    every literal whose value a knob writes exactly, a reference to that
//!    knob, the tile depth declared on the grid when one is referenced.

use crate::wgsl::loader::WgslKingdom;
use crate::geneframe::Ty;
use crate::wgsl::params;
use crate::wgsl::render::{array_len, dual_ty, grid_of, parr_var, snap, wgmem_var, workgroup_dim, workgroup_plan, workgroup_shape, Tree, BLK, UNROLL};

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

/// The statements a block writes, in order: nested plain blocks spliced in
/// place, an empty block (its fallback) nothing, and nothing after a return.
fn statements(block: &Tree, out: &mut Vec<Tree>) -> bool {
    if block.name == "fallback" {
        return false;
    }
    for s in &block.kids {
        let nested = s.name.starts_with("stmt.block_") && s.kids.len() == 1 && (is_plain_block(&s.kids[0].name) || s.kids[0].name == "fallback");
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

/// A statement list as the parser groups it at `rung`.
fn grouped(mut items: Vec<Tree>, rung: &str, like: &Tree, stmt_ty: Option<Ty>) -> Tree {
    if items.is_empty() {
        return Tree::fallback(like.ty);
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
    if t.name == "fallback" {
        return Some(Tree::fallback(demand));
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
    let body_block = |items: Vec<Tree>| grouped(items, "L0", &Tree::fallback(b0), Some(l0));
    if returns {
        return Tree::new(&like.name, like.ty, vec![grid, body_block(items), Tree::fallback(o0)]);
    }
    let mut epi = Vec::new();
    statements(epilogue, &mut epi);
    let Some(last) = epi.pop() else { return like.clone() };
    for s in epi {
        match convert(&s, l0, kingdom) {
            Some(x) => items.push(x),
            None => return like.clone(),
        }
    }
    let epilogue = grouped(vec![last], "O0", &Tree::fallback(o0), dual_ty(kingdom, "stmt.O0").ok());
    Tree::new(&like.name, like.ty, vec![grid, body_block(items), epilogue])
}

/// A literal's value as the emitter writes it for its form (rule 4).
fn literal_value(form: &str, v: f32) -> f32 {
    let whole = |lo: f64, hi: f64| if v.is_finite() { (f64::from(v).round()).clamp(lo, hi) as f32 } else { lo as f32 };
    match form {
        "index" | "count" | "bits" | "hash32" | "code4" | "code8" | "code16" | "code1024" => {
            if v.is_finite() {
                v.clamp(0.0, 4_294_967_040.0) as u32 as f32
            } else {
                0.0
            }
        }
        "wgdim" => workgroup_dim(v) as f32,
        "lit64" => whole(1.0, 64.0),
        "lit4096" => whole(1.0, 4096.0),
        "blk" => snap(&BLK, v) as f32,
        "unroll" => snap(&UNROLL, v) as f32,
        "int" | "q15_16" => v.clamp(-2_147_483_648.0, 2_147_483_520.0) as i32 as f32,
        "flag" | "sign" => f32::from(u8::from(v != 0.0)),
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
    let mut plan = workgroup_plan(kingdom, grid, root).ok()?;
    if plan.is_empty() {
        // The original four arrays, declared as they always were: the ones
        // the statements use, at their original lengths.
        fn uses(t: &Tree, kingdom: &WgslKingdom, out: &mut std::collections::BTreeMap<String, (String, u32)>) {
            let array = t.kids.is_empty() && (t.name.starts_with("wg.") || (t.name == "fallback" && kingdom.dual(t.ty).is_some_and(|d| d.dual.starts_with("wgmem<"))));
            if let Some(v) = array.then(|| wgmem_var(kingdom, t).ok()).flatten() {
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

/// Rule 7: every thread origin in `t` as the grid writes it.
fn origins(t: &Tree, g: &crate::wgsl::render::Grid) -> Tree {
    let kids: Vec<Tree> = t.kids.iter().map(|k| origins(k, g)).collect();
    let b = |name: &str, k: &Tree| Tree::new(name, k.ty, vec![]);
    let blk = |k: &Tree, v: u32| Tree::literal("literal.blk", k.ty, v as f32);
    let (row, by, bx) = (t.name.contains("_row"), g.by, g.bx);
    let f = if row { by } else { bx };
    let kids = match (t.name.as_str(), kids.as_slice()) {
        ("safe.thread_row" | "safe.thread_col", [gid, k]) => vec![b("builtin.global_invocation_id", gid), blk(k, f)],
        ("safe.thread_row_s" | "safe.thread_col_s", [wid, lid, k]) => vec![b("builtin.workgroup_id", wid), b("builtin.local_invocation_id", lid), blk(k, f)],
        ("safe.thread_row_g" | "safe.thread_col_g", [wid, lid, k, n]) => vec![b("builtin.workgroup_id", wid), b("builtin.local_invocation_id", lid), blk(k, f), n.clone()],
        _ => kids,
    };
    Tree { kids, ..t.clone() }
}

/// The canonical spelling of `t` (see the module header for the rules).
pub fn canonical(t: &Tree, kingdom: &WgslKingdom) -> Tree {
    // Knob references first, at the drawn grid's knobs (rule 4).
    let t = params::to_literals(t, kingdom).unwrap_or_else(|_| t.clone());
    let c = structural(&t, kingdom);
    params::assign(&c, kingdom).unwrap_or(c)
}

/// Rules 1 to 4.
fn structural(t: &Tree, kingdom: &WgslKingdom) -> Tree {
    if t.kids.is_empty() && t.name.starts_with("literal.") {
        return canonical_literal(t, kingdom);
    }
    if t.name == "fallback" {
        if let Some(z) = canonical_fallback(t, kingdom) {
            return z;
        }
    }
    let kids: Vec<Tree> = t.kids.iter().map(|k| structural(k, kingdom)).collect();
    let t = Tree { kids, ..t.clone() };
    if t.name == "kernel.entry_safe" {
        if let [grid, body, epilogue] = t.kids.as_slice() {
            let split = split_epilogue(grid.clone(), body, epilogue, kingdom, &t);
            // The moved statements regrouped with the body's (rule 1).
            let kids: Vec<Tree> = split.kids.iter().map(|k| canonical_block(k, kingdom)).collect();
            let root = Tree { kids, ..split };
            let grid = canonical_grid(&root.kids[0], &root, kingdom).unwrap_or_else(|| root.kids[0].clone());
            let mut kids = root.kids.clone();
            if let Ok(g) = grid_of(&grid) {
                for k in kids.iter_mut().skip(1) {
                    *k = origins(k, &g);
                }
            }
            kids[0] = grid;
            return Tree { kids, ..root };
        }
    }
    canonical_block(&t, kingdom)
}

/// Rule 5: a fallback value leaf as the parser reads the zero written for it.
fn canonical_fallback(t: &Tree, kingdom: &WgslKingdom) -> Option<Tree> {
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
    if d.is_statement_level() {
        return None;
    }
    // A thread's origin or cell: the thread's own (the emitter's own_fallback).
    match d.dual.as_str() {
        "u32.threadrow" | "u32.rowbase" => return origin("safe.thread_row", "u32.threadrow"),
        "u32.threadcol" | "u32.colbase" => return origin("safe.thread_col", "u32.threadcol"),
        "u32.threadrows" | "u32.threadrowg" | "u32.rowbases" | "u32.rowbaseg" => return origin("safe.thread_row_s", "u32.threadrows"),
        "u32.threadcols" | "u32.threadcolg" | "u32.colbases" | "u32.colbaseg" => return origin("safe.thread_col_s", "u32.threadcols"),
        "u32.cell" => return cell(),
        _ => {}
    }
    // An array: the one the emitter names for the dual.
    if d.dual.starts_with("parr<") {
        let name = format!("parr.{}", parr_var(kingdom, t).ok()?.strip_prefix('p')?);
        return Some(Tree::new(&name, t.ty, vec![]));
    }
    if d.dual.starts_with("wgmem<") {
        let name = format!("wg.{}", wgmem_var(kingdom, t).ok()?.strip_prefix("wg_")?);
        return Some(Tree::new(&name, t.ty, vec![]));
    }
    let scalar = |s: &str| -> Option<Tree> {
        let (lit, dual) = match s {
            "f32" => ("literal.real", "f32.real"),
            "u32" => ("literal.index", "u32.index"),
            "i32" => ("literal.int", "i32.int"),
            "bool" => ("literal.flag", "bool.flag"),
            _ => return None,
        };
        Some(Tree::literal(lit, dual_ty(kingdom, dual).ok()?, 0.0))
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
    let per_kernel = crate::wgsl::safe_karva::per_kernel_rows(kingdom, &crate::wgsl::render::gram_role());
    crate::wgsl::safe_karva::conform(&zero, t.ty, kingdom, &per_kernel).ok().filter(|z| z.name != "fallback").or_else(|| {
        // A safe index (a workgroup or private array's): the zero through
        // the safe row the parser reads it with (`safe.wga_of(0u)`).
        let r = kingdom.rows.iter().find(|r| r.name.starts_with("safe.") && r.output == t.ty && r.inputs.as_slice() == [zero.ty])?;
        Some(Tree::new(&r.name, t.ty, vec![zero]))
    })
}

/// Rule 6 at a reduction (its children already canonical).
fn canonical_reduction(t: &Tree, kingdom: &WgslKingdom) -> Option<Tree> {
    let stem = t.name.strip_prefix("wg.reduce_")?;
    let rung = rung_of(stem)?;
    let [a, n] = t.kids.as_slice() else { return None };
    let v = if n.name == "fallback" { 1 } else { n.konst.max(1.0) as u32 };
    let p = 1u32 << (31 - v.leading_zeros());
    if p == 1 {
        return Some(Tree::new(&format!("stmt.barrier_{rung}"), t.ty, vec![]));
    }
    let lit = Tree::literal("literal.lit4096", dual_ty(kingdom, "u32.lit4096").ok()?, p as f32);
    Some(Tree::new(&t.name, t.ty, vec![a.clone(), lit]))
}

/// Rule 1 at `t` itself (its children already canonical).
fn canonical_block(t: &Tree, kingdom: &WgslKingdom) -> Tree {
    if let Some(r) = canonical_reduction(t, kingdom) {
        return r;
    }
    if is_plain_block(&t.name) {
        if let Some(rung) = rung_of(&t.name) {
            let mut items = Vec::new();
            statements(t, &mut items);
            let stmt_ty = dual_ty(kingdom, &format!("stmt.{rung}")).ok();
            return grouped(items, rung, t, stmt_ty);
        }
    }
    t.clone()
}
