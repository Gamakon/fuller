//! RETURNS INSIDE LOOPS, LOWERED (2026-10-19): the kingdom spells a `return`
//! under a condition at the end of a block (the rest of the block is its else)
//! but not inside a loop, where the rest of the kernel is skipped from within.
//! A helper that searches (`for (k < 8) { if (hit) { return true; } }
//! return false;`) and a kernel that culls (`for (j < 6) { if (outside) {
//! return; } }`) are common. This pass, run on the parsed module before it is
//! validated, rewrites them as WGSL's usual loop-exit idiom with a flag: each
//! such `return v;` becomes `ret_val = v; ret_flag = true; break;`, each loop
//! that held one is followed by `if (ret_flag) { break; }` inside another loop
//! or `if (ret_flag) { return ret_val; }` outside, and the flag starts false.
//! The threads compute what they did.
//!
//! A return inside a `switch` inside a loop (a `break` there leaves the switch)
//! or inside a loop's `continuing` block is left as it is, for the reader to name.

use naga::{Block, Expression, Function, Handle, Literal, Module, Span, Statement};

/// What the lowering of one function needs.
struct Lowering {
    flag: Handle<Expression>,
    value: Option<Handle<Expression>>,
    on: Handle<Expression>,
}

/// Whether `block` holds a return inside a loop that can be lowered, and none that cannot.
fn plan(block: &Block, in_loop: bool, in_switch: bool, found: &mut bool) -> bool {
    for s in block.iter() {
        match s {
            Statement::Return { .. } if in_loop => {
                if in_switch {
                    return false;
                }
                *found = true;
            }
            Statement::Block(b) => {
                if !plan(b, in_loop, in_switch, found) {
                    return false;
                }
            }
            Statement::If { accept, reject, .. } => {
                if !plan(accept, in_loop, in_switch, found) || !plan(reject, in_loop, in_switch, found) {
                    return false;
                }
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    if !plan(&c.body, in_loop, in_loop || in_switch, found) {
                        return false;
                    }
                }
            }
            Statement::Loop { body, continuing, .. } => {
                let mut in_continuing = false;
                if !plan(body, true, false, found) || !plan(continuing, true, false, &mut in_continuing) || in_continuing {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
}

fn load(f: &mut Function, pointer: Handle<Expression>) -> (Handle<Expression>, Statement) {
    let h = f.expressions.append(Expression::Load { pointer }, Span::UNDEFINED);
    (h, Statement::Emit(naga::Range::new_from_bounds(h, h)))
}

/// What lowering a block found: whether it lowered a return, and whether a
/// lowered loop outside any loop sits in it (or under it) so that the statements
/// after the block's own statement must not run once the flag is set.
struct Lowered {
    block: Block,
    any: bool,
    pending: bool,
}

/// `block` with its returns inside loops lowered. `top` is the function's own
/// block: a loop that returned there is followed by the return; in a nested
/// block the rest of the block (and, by the caller, of the blocks around it) is
/// guarded by the flag instead.
fn rewrite(f: &mut Function, block: &Block, depth: usize, top: bool, l: &Lowering) -> Lowered {
    let stmts: Vec<Statement> = block.iter().cloned().collect();
    let mut out = Vec::new();
    let mut any = false;
    for (at, s) in stmts.iter().enumerate() {
        let mut pending = false;
        match s {
            Statement::Return { value } if depth > 0 => {
                if let (Some(v), Some(p)) = (value, l.value) {
                    out.push(Statement::Store { pointer: p, value: *v });
                }
                out.push(Statement::Store { pointer: l.flag, value: l.on });
                out.push(Statement::Break);
                any = true;
            }
            Statement::Block(b) => {
                let r = rewrite(f, b, depth, false, l);
                any |= r.any;
                pending = r.pending;
                out.push(Statement::Block(r.block));
            }
            Statement::If { condition, accept, reject } => {
                let (a, r) = (rewrite(f, accept, depth, false, l), rewrite(f, reject, depth, false, l));
                any |= a.any | r.any;
                pending = a.pending | r.pending;
                out.push(Statement::If { condition: *condition, accept: a.block, reject: r.block });
            }
            Statement::Switch { selector, cases } => {
                // (Only at depth 0: a switch in a loop with a return was not planned.)
                let cases = cases
                    .iter()
                    .map(|c| {
                        let r = rewrite(f, &c.body, depth, false, l);
                        any |= r.any;
                        pending |= r.pending;
                        naga::SwitchCase { value: c.value, body: r.block, fall_through: c.fall_through }
                    })
                    .collect();
                out.push(Statement::Switch { selector: *selector, cases });
            }
            Statement::Loop { body, continuing, break_if } => {
                let r = rewrite(f, body, depth + 1, false, l);
                out.push(Statement::Loop { body: r.block, continuing: continuing.clone(), break_if: *break_if });
                if r.any {
                    any = true;
                    if depth > 0 {
                        let (flag, emit) = load(f, l.flag);
                        out.push(emit);
                        out.push(Statement::If { condition: flag, accept: Block::from_vec(vec![Statement::Break]), reject: Block::new() });
                    } else {
                        pending = true;
                    }
                }
            }
            other => out.push(other.clone()),
        }
        if pending {
            // A loop outside any loop returned: at the function's own block the return
            // follows; below it the rest of this block runs only when the flag is clear
            // and the caller is told to do the same for its own rest.
            if top {
                let exit = match l.value {
                    Some(p) => {
                        let (v, emit_v) = load(f, p);
                        Block::from_vec(vec![emit_v, Statement::Return { value: Some(v) }])
                    }
                    None => Block::from_vec(vec![Statement::Return { value: None }]),
                };
                let (flag, emit) = load(f, l.flag);
                out.push(emit);
                out.push(Statement::If { condition: flag, accept: exit, reject: Block::new() });
            } else if at + 1 < stmts.len() {
                let rest = rewrite(f, &Block::from_vec(stmts[at + 1..].to_vec()), depth, false, l);
                any |= rest.any;
                let (flag, emit) = load(f, l.flag);
                let clear = f.expressions.append(Expression::Unary { op: naga::UnaryOperator::LogicalNot, expr: flag }, Span::UNDEFINED);
                out.push(emit);
                out.push(Statement::Emit(naga::Range::new_from_bounds(clear, clear)));
                out.push(Statement::If { condition: clear, accept: rest.block, reject: Block::new() });
                return Lowered { block: Block::from_vec(out), any, pending: true };
            } else {
                return Lowered { block: Block::from_vec(out), any, pending: true };
            }
        }
    }
    Lowered { block: Block::from_vec(out), any, pending: false }
}

/// One function's returns inside loops lowered; whether any was.
fn lower_function(module_types: &mut naga::UniqueArena<naga::Type>, f: &mut Function) -> bool {
    let mut found = false;
    if !plan(&f.body, false, false, &mut found) || !found {
        return false;
    }
    let bool_ty = module_types.insert(naga::Type { name: None, inner: naga::TypeInner::Scalar(naga::Scalar::BOOL) }, Span::UNDEFINED);
    let flag_var = f.local_variables.append(naga::LocalVariable { name: Some("ret_flag".into()), ty: bool_ty, init: None }, Span::UNDEFINED);
    let flag = f.expressions.append(Expression::LocalVariable(flag_var), Span::UNDEFINED);
    let value = f.result.as_ref().map(|r| {
        let v = f.local_variables.append(naga::LocalVariable { name: Some("ret_val".into()), ty: r.ty, init: None }, Span::UNDEFINED);
        f.expressions.append(Expression::LocalVariable(v), Span::UNDEFINED)
    });
    let on = f.expressions.append(Expression::Literal(Literal::Bool(true)), Span::UNDEFINED);
    let off = f.expressions.append(Expression::Literal(Literal::Bool(false)), Span::UNDEFINED);
    let l = Lowering { flag, value, on };
    let body = std::mem::take(&mut f.body);
    let r = rewrite(f, &body, 0, true, &l);
    let mut stmts = vec![Statement::Store { pointer: flag, value: off }];
    stmts.extend(r.block.iter().cloned());
    f.body = Block::from_vec(stmts);
    true
}

/// The module with every function's returns inside loops lowered; how many
/// functions changed.
pub fn lower_returns_in_loops(module: &mut Module) -> usize {
    let mut n = 0;
    let types = &mut module.types;
    for (_, f) in module.functions.iter_mut() {
        n += usize::from(lower_function(types, f));
    }
    for ep in module.entry_points.iter_mut() {
        n += usize::from(lower_function(types, &mut ep.function));
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::cpu::{run, Val};
    use std::collections::HashMap;

    /// The lowered module computes what the original does: a return inside a
    /// loop, inside two loops, in a helper that searches, and one in a loop
    /// nested in an `if` that the rest of the kernel follows. Both run on the
    /// CPU executor over the same inputs; the lowered one is valid for naga.
    #[test]
    fn a_return_inside_a_loop_lowered_computes_what_it_did() {
        let src = "@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
fn find(x: u32) -> bool {
    for (var k = 0u; k < 8u; k++) {
        if (keys[k] == x) { return true; }
        if (keys[k] > x) { return false; }
    }
    return true;
}
@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var r = 0u;
    if gid.x < 12u {
        for (var j = 0u; j < 4u; j++) {
            for (var i = 0u; i < 4u; i++) {
                if (gid.x + i * j == 7u) { out[gid.x] = 100u; return; }
            }
            r += j;
        }
        if find(gid.x) { r += 50u; }
    }
    out[gid.x] = r + 1u;
}
";
        let original = crate::wgsl::naga_shape::parse(src).unwrap();
        let mut lowered = original.clone();
        assert_eq!(lower_returns_in_loops(&mut lowered), 2, "main and find");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&lowered).unwrap_or_else(|e| panic!("{e:?}"));
        let buffers = || -> HashMap<String, Val> {
            let keys = Val::V([0u32, 2, 3, 5, 7, 9, 11, 13].iter().map(|v| Val::U(*v)).collect());
            let out = Val::V((0..16).map(|_| Val::U(0)).collect());
            HashMap::from([("keys".to_string(), keys), ("out".to_string(), out)])
        };
        let (a, b) = (run(&original, buffers(), [1, 1, 1], 1_000_000).unwrap(), run(&lowered, buffers(), [1, 1, 1], 1_000_000).unwrap());
        assert_eq!(a.buffers["out"], b.buffers["out"]);
        // (Some thread must have returned from the loops, and some from neither.)
        let Val::V(cells) = &b.buffers["out"] else { panic!("out is an array") };
        assert!(cells.contains(&Val::U(100)) && cells.iter().any(|c| matches!(c, Val::U(v) if *v != 100 && *v != 0)), "{cells:?}");
    }
}
