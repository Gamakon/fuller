//! ONE PARSE AND THE FEATURE SCAN (2026-10-19, the naga/wgpu 30 upgrade,
//! docs/PLAN_naga_wgpu_30.md section 2.1). Every WGSL text the kingdom reads
//! is parsed here, by naga 30's WGSL front end, and scanned at once for
//! what naga 30 can express and naga 0.20 could not: a cooperative matrix, a
//! 16-bit literal, an override-sized array, the immediate, task and ray
//! address spaces, the mesh and ray builtins, a memory barrier without
//! control, an image atomic, a ray pipeline call, a cooperative store, the
//! quad gathers. The kingdom has no row for any of them: each is refused by
//! name before a reader, the safety checker or the CPU executor sees the
//! module, so none can fall into a wildcard downstream (Codex's review,
//! finding 4).
//!
//! And THE READER'S VIEW OF A SHORT CIRCUIT ([`fold_short_circuits`]): naga
//! 30 lowers a run-time `a && b` into an unnamed bool local set inside an
//! `if` (naga 0.20 wrote one Binary). The tree readers (the scaffold reader
//! and through it the safe reader) read the kernel's expressions as the
//! kingdom's pure rows, `logic.and(a, b)`, so for them, and only them, each
//! such `if` whose right side calls only pure functions, and whose local is
//! read, is the Binary it was written as. The module the CPU executor, the
//! safety checker and the device run is never folded: there the short
//! circuit is what the kernel does (Codex's review, findings 1 to 3).

use std::collections::HashMap;

use naga::{AddressSpace, ArraySize, BinaryOperator, Binding, BuiltIn, Expression, Function, GatherMode, Handle, Literal, LocalVariable, Module, Statement, TypeInner, UnaryOperator};

/// Parse `source` and scan it ([`supported`]).
pub fn parse(source: &str) -> Result<Module, String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    supported(&module)?;
    Ok(module)
}

/// Parse `source` without the scan: to validate text the RENDERER wrote that the readers have no row for (a module that
/// uses f16, whose constants the typed-GEP terminals write, docs/PLAN_wgsl_terminals.md). Never a way to read a kernel.
pub fn parse_unscanned(source: &str) -> Result<Module, String> {
    naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))
}

fn refuse(what: &str) -> Result<(), String> {
    Err(format!("our software has a bug: {what} has no row in the kingdom"))
}

/// Every type, global, function, argument, result, expression and statement
/// of `module` once: Err naming the first construct the kingdom has no row
/// for (the module's header lists them).
pub fn supported(module: &Module) -> Result<(), String> {
    for (_, t) in module.types.iter() {
        match &t.inner {
            TypeInner::CooperativeMatrix { .. } => refuse("a cooperative matrix type")?,
            TypeInner::Array { size: ArraySize::Pending(_), .. } => refuse("an override-sized array")?,
            TypeInner::Scalar(sc) | TypeInner::Vector { scalar: sc, .. } | TypeInner::Matrix { scalar: sc, .. } if sc.width == 2 => refuse("a 16-bit type")?,
            _ => {}
        }
        if let TypeInner::Struct { members, .. } = &t.inner {
            for m in members {
                binding(m.binding.as_ref())?;
            }
        }
    }
    for (_, g) in module.global_variables.iter() {
        match g.space {
            AddressSpace::Immediate => refuse("the immediate address space")?,
            AddressSpace::TaskPayload => refuse("the task payload address space")?,
            AddressSpace::RayPayload | AddressSpace::IncomingRayPayload => refuse("a ray payload address space")?,
            _ => {}
        }
    }
    for (_, f) in module.functions.iter() {
        function(f)?;
    }
    for ep in &module.entry_points {
        function(&ep.function)?;
    }
    Ok(())
}

fn binding(b: Option<&Binding>) -> Result<(), String> {
    let Some(Binding::BuiltIn(b)) = b else { return Ok(()) };
    match b {
        BuiltIn::DrawIndex | BuiltIn::Barycentric { .. } => refuse(&format!("the builtin {b:?}")),
        BuiltIn::MeshTaskSize | BuiltIn::CullPrimitive | BuiltIn::PointIndex | BuiltIn::LineIndices | BuiltIn::TriangleIndices | BuiltIn::VertexCount | BuiltIn::Vertices | BuiltIn::PrimitiveCount | BuiltIn::Primitives => refuse(&format!("the mesh builtin {b:?}")),
        BuiltIn::RayInvocationId | BuiltIn::NumRayInvocations | BuiltIn::InstanceCustomData | BuiltIn::GeometryIndex | BuiltIn::WorldRayOrigin | BuiltIn::WorldRayDirection | BuiltIn::ObjectRayOrigin | BuiltIn::ObjectRayDirection | BuiltIn::RayTmin | BuiltIn::RayTCurrentMax | BuiltIn::ObjectToWorld | BuiltIn::WorldToObject | BuiltIn::HitKind => refuse(&format!("the ray builtin {b:?}")),
        _ => Ok(()),
    }
}

fn function(f: &naga::Function) -> Result<(), String> {
    for a in &f.arguments {
        binding(a.binding.as_ref())?;
    }
    if let Some(r) = &f.result {
        binding(r.binding.as_ref())?;
    }
    for (_, e) in f.expressions.iter() {
        match e {
            Expression::Literal(Literal::F16(_) | Literal::U16(_) | Literal::I16(_)) => refuse("a 16-bit literal")?,
            Expression::CooperativeLoad { .. } | Expression::CooperativeMultiplyAdd { .. } => refuse("a cooperative matrix operation")?,
            Expression::RayQueryVertexPositions { .. } => refuse("a ray query's vertex positions")?,
            _ => {}
        }
    }
    block(&f.body)
}

fn block(b: &naga::Block) -> Result<(), String> {
    for s in b.iter() {
        match s {
            Statement::MemoryBarrier(_) => refuse("a memory barrier without control")?,
            Statement::ImageAtomic { .. } => refuse("an image atomic")?,
            Statement::RayPipelineFunction(_) => refuse("a ray pipeline function")?,
            Statement::CooperativeStore { .. } => refuse("a cooperative matrix store")?,
            Statement::SubgroupGather { mode: GatherMode::QuadBroadcast(_) | GatherMode::QuadSwap(_), .. } => refuse("a quad gather")?,
            Statement::Block(x) => block(x)?,
            Statement::If { accept, reject, .. } => {
                block(accept)?;
                block(reject)?;
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    block(&c.body)?;
                }
            }
            Statement::Loop { body, continuing, .. } => {
                block(body)?;
                block(continuing)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Whether `f` (a module function) has no effect a caller could see: no
/// store through a global or a pointer argument, no atomic, no barrier, no
/// image store, and every call pure. A loop is allowed: the readers' tree
/// is a value, not a run, and the device runs the short circuit itself.
fn pure_function(module: &Module, f: Handle<Function>, depth: usize) -> bool {
    if depth > 16 {
        return false;
    }
    let func = &module.functions[f];
    fn local_place(func: &Function, mut p: Handle<Expression>) -> bool {
        loop {
            match func.expressions[p] {
                Expression::LocalVariable(_) => return true,
                Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => p = base,
                _ => return false,
            }
        }
    }
    fn block(module: &Module, func: &Function, b: &naga::Block, depth: usize) -> bool {
        b.iter().all(|s| match s {
            Statement::Emit(_) | Statement::Break | Statement::Continue | Statement::Return { .. } => true,
            Statement::Store { pointer, .. } => local_place(func, *pointer),
            Statement::Block(x) => block(module, func, x, depth),
            Statement::If { accept, reject, .. } => block(module, func, accept, depth) && block(module, func, reject, depth),
            Statement::Switch { cases, .. } => cases.iter().all(|c| block(module, func, &c.body, depth)),
            Statement::Loop { body, continuing, .. } => block(module, func, body, depth) && block(module, func, continuing, depth),
            Statement::Call { function, .. } => pure_function(module, *function, depth + 1),
            _ => false,
        })
    }
    block(module, func, &func.body, depth)
}

/// A folded short circuit: its operator and its two sides.
type Fold = (BinaryOperator, Handle<Expression>, Handle<Expression>);

/// The short-circuit `if` at `s`, when the reader may read it as the Binary
/// it was written as: its local, the operator and the two sides.
fn short_circuit(module: &Module, func: &Function, s: &Statement, loaded: &HashMap<Handle<LocalVariable>, usize>, stores: &HashMap<Handle<LocalVariable>, usize>) -> Option<(Handle<LocalVariable>, Fold)> {
    let Statement::If { condition, accept, reject } = s else { return None };
    let store_of = |st: &Statement| match st {
        Statement::Store { pointer, value } => match func.expressions[*pointer] {
            Expression::LocalVariable(v) => Some((v, *value)),
            _ => None,
        },
        _ => None,
    };
    let mut rejects = reject.iter().filter(|st| !matches!(st, Statement::Emit(_)));
    let (v, else_value) = store_of(rejects.next()?)?;
    if rejects.next().is_some() {
        return None;
    }
    let local = &func.local_variables[v];
    let is_bool = matches!(module.types[local.ty].inner, TypeInner::Scalar(naga::Scalar::BOOL));
    if local.name.is_some() || local.init.is_some() || !is_bool || stores.get(&v) != Some(&2) || !loaded.contains_key(&v) {
        return None;
    }
    let (op, lhs) = match (&func.expressions[else_value], &func.expressions[*condition]) {
        (Expression::Literal(Literal::Bool(false)), _) => (BinaryOperator::LogicalAnd, *condition),
        (Expression::Literal(Literal::Bool(true)), Expression::Unary { op: UnaryOperator::LogicalNot, expr }) => (BinaryOperator::LogicalOr, *expr),
        _ => return None,
    };
    let (last, before) = accept.split_last()?;
    let (w, rhs) = store_of(last)?;
    let pure = before.iter().all(|st| match st {
        Statement::Emit(_) => true,
        Statement::Call { function, .. } => pure_function(module, *function, 0),
        _ => false,
    });
    (w == v && pure).then_some((v, (op, lhs, rhs)))
}

/// THE READER'S VIEW of every short circuit in `module` (the module header):
/// post-order through every block (an inner `&&` of a right side first, so
/// its `if` is gone when the outer one is matched), each matched `if`
/// replaced by its accept's Emits and Calls, in place, and each load of its
/// local by the Binary. An `if` whose right side does anything else, or
/// whose local is never read, stays the `if` it is.
pub fn fold_short_circuits(module: &mut Module) {
    let fold = |module: &Module, func: &mut Function| {
        let mut loaded: HashMap<Handle<LocalVariable>, usize> = HashMap::new();
        let mut stores: HashMap<Handle<LocalVariable>, usize> = HashMap::new();
        for (_, e) in func.expressions.iter() {
            if let Expression::Load { pointer } = e {
                if let Expression::LocalVariable(v) = func.expressions[*pointer] {
                    *loaded.entry(v).or_default() += 1;
                }
            }
        }
        fn count_stores(func: &Function, b: &naga::Block, stores: &mut HashMap<Handle<LocalVariable>, usize>) {
            for s in b.iter() {
                match s {
                    Statement::Store { pointer, .. } => {
                        if let Expression::LocalVariable(v) = func.expressions[*pointer] {
                            *stores.entry(v).or_default() += 1;
                        }
                    }
                    Statement::Block(x) => count_stores(func, x, stores),
                    Statement::If { accept, reject, .. } => {
                        count_stores(func, accept, stores);
                        count_stores(func, reject, stores);
                    }
                    Statement::Switch { cases, .. } => cases.iter().for_each(|c| count_stores(func, &c.body, stores)),
                    Statement::Loop { body, continuing, .. } => {
                        count_stores(func, body, stores);
                        count_stores(func, continuing, stores);
                    }
                    _ => {}
                }
            }
        }
        count_stores(func, &func.body, &mut stores);
        let mut folded: HashMap<Handle<LocalVariable>, Fold> = HashMap::new();
        fn walk(module: &Module, func: &Function, b: naga::Block, loaded: &HashMap<Handle<LocalVariable>, usize>, stores: &HashMap<Handle<LocalVariable>, usize>, folded: &mut HashMap<Handle<LocalVariable>, Fold>) -> naga::Block {
            let mut out = naga::Block::with_capacity(b.len());
            for (s, span) in b.span_into_iter() {
                let s = match s {
                    Statement::Block(x) => Statement::Block(walk(module, func, x, loaded, stores, folded)),
                    Statement::If { condition, accept, reject } => Statement::If { condition, accept: walk(module, func, accept, loaded, stores, folded), reject: walk(module, func, reject, loaded, stores, folded) },
                    Statement::Switch { selector, cases } => Statement::Switch { selector, cases: cases.into_iter().map(|c| naga::SwitchCase { body: walk(module, func, c.body, loaded, stores, folded), ..c }).collect() },
                    Statement::Loop { body, continuing, break_if } => Statement::Loop { body: walk(module, func, body, loaded, stores, folded), continuing: walk(module, func, continuing, loaded, stores, folded), break_if },
                    other => other,
                };
                match short_circuit(module, func, &s, loaded, stores) {
                    Some((v, fold)) => {
                        let Statement::If { accept, .. } = s else { unreachable!("short_circuit matched an if") };
                        let n = accept.len();
                        for (st, sp) in accept.span_into_iter().take(n - 1) {
                            out.push(st, sp);
                        }
                        folded.insert(v, fold);
                    }
                    None => out.push(s, span),
                }
            }
            out
        }
        let body = std::mem::take(&mut func.body);
        func.body = walk(module, func, body, &loaded, &stores, &mut folded);
        let loads: Vec<Handle<Expression>> = func.expressions.iter().filter_map(|(h, e)| match e {
            Expression::Load { pointer } => match func.expressions[*pointer] {
                Expression::LocalVariable(v) if folded.contains_key(&v) => Some(h),
                _ => None,
            },
            _ => None,
        }).collect();
        for h in loads {
            let Expression::Load { pointer } = func.expressions[h] else { continue };
            let Expression::LocalVariable(v) = func.expressions[pointer] else { continue };
            let (op, left, right) = folded[&v];
            *func.expressions.get_mut(h) = Expression::Binary { op, left, right };
        }
    };
    let handles: Vec<Handle<Function>> = module.functions.iter().map(|(h, _)| h).collect();
    for h in handles {
        let mut f = std::mem::take(&mut module.functions[h]);
        fold(module, &mut f);
        module.functions[h] = f;
    }
    for i in 0..module.entry_points.len() {
        let mut f = std::mem::take(&mut module.entry_points[i].function);
        fold(module, &mut f);
        module.entry_points[i].function = f;
    }
}

/// Whether no load, store or atomic of `f` reaches the local `v` (a folded
/// short circuit's local, left in the arena so no handle moves): the
/// readers allocate no register for it and zero none.
pub fn is_dead_local(f: &Function, v: Handle<LocalVariable>) -> bool {
    let roots_at_v = |mut p: Handle<Expression>| loop {
        match f.expressions[p] {
            Expression::LocalVariable(l) => return l == v,
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => p = base,
            _ => return false,
        }
    };
    let loads = f.expressions.iter().any(|(_, e)| matches!(e, Expression::Load { pointer } if roots_at_v(*pointer)));
    fn touches(b: &naga::Block, at: &dyn Fn(Handle<Expression>) -> bool) -> bool {
        b.iter().any(|s| match s {
            Statement::Store { pointer, .. } | Statement::Atomic { pointer, .. } => at(*pointer),
            Statement::Call { arguments, .. } => arguments.iter().any(|a| at(*a)),
            Statement::Block(x) => touches(x, at),
            Statement::If { accept, reject, .. } => touches(accept, at) || touches(reject, at),
            Statement::Switch { cases, .. } => cases.iter().any(|c| touches(&c.body, at)),
            Statement::Loop { body, continuing, .. } => touches(body, at) || touches(continuing, at),
            _ => false,
        })
    }
    !loads && !touches(&f.body, &roots_at_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHAT NAGA 30 ADDED IS REFUSED BY NAME: a 16-bit type (behind `enable
    /// f16`), a quad gather; an ordinary kernel parses.
    #[test]
    fn what_naga_30_added_is_refused_by_name() {
        let ok = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) { out[gid.x] = gid.x; }\n";
        parse(ok).expect("an ordinary kernel");
        let f16 = "enable f16;\n@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) { var h: f16 = f16(gid.x); out[gid.x] = f32(h); }\n";
        let e = parse(f16).expect_err("a 16-bit type");
        assert!(e.contains("a 16-bit type has no row"), "{e}");
        let quad = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) { out[gid.x] = quadSwapX(gid.x); }\n";
        let e = parse(quad).expect_err("a quad gather");
        assert!(e.contains("a quad gather has no row"), "{e}");
    }

    /// The statements of `main` that are an `if` storing to an unnamed local
    /// (a short circuit left as naga 30 lowered it), anywhere in the body.
    fn short_circuit_ifs(m: &Module) -> usize {
        fn count(f: &Function, b: &naga::Block) -> usize {
            b.iter().map(|s| match s {
                Statement::If { accept, reject, .. } => {
                    let unnamed_store = |x: &naga::Block| x.iter().any(|t| matches!(t, Statement::Store { pointer, .. } if matches!(f.expressions[*pointer], Expression::LocalVariable(v) if f.local_variables[v].name.is_none())));
                    usize::from(unnamed_store(accept) || unnamed_store(reject)) + count(f, accept) + count(f, reject)
                }
                Statement::Block(x) => count(f, x),
                Statement::Loop { body, continuing, .. } => count(f, body) + count(f, continuing),
                Statement::Switch { cases, .. } => cases.iter().map(|c| count(f, &c.body)).sum(),
                _ => 0,
            }).sum()
        }
        m.entry_points.iter().map(|ep| count(&ep.function, &ep.function.body)).sum()
    }

    /// THE READER'S VIEW OF A SHORT CIRCUIT (Codex's review, findings 1 to
    /// 3): each `&&` / `||` whose right side calls only pure helpers is folded
    /// back to its Binary, wherever it stands (an `if` condition, nested in a
    /// right side, a loop's test, `break if` in continuing), and the folded
    /// module validates; a short circuit whose value is never read, or whose
    /// right side calls a function that stores, stays the `if` it is. A
    /// folded short circuit's local is dead ([`is_dead_local`]).
    #[test]
    fn a_short_circuit_folds_back_only_where_it_is_a_pure_value() {
        let head = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\nfn id(x: u32) -> u32 { return x; }\nfn bump(x: u32) -> u32 { out[0] = x; return x; }\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n";
        let folds = [
            "if (gid.x < 4u && gid.y < 2u && gid.z < 3u) { out[gid.x] = 1u; }",
            "if (gid.x < 4u && (gid.y < 2u || id(gid.z) < 3u)) { out[gid.x] = 1u; }",
            "var i = 0u; loop { if !(i < 8u && gid.x < 4u) { break; } out[i] = i; continuing { i = i + 1u; break if i > 6u || gid.y > 1u; } }",
            "for (var i = 0u; i < 8u && id(gid.x) < 4u; i++) { out[i] = i; }",
        ];
        for body in folds {
            let text = format!("{head}{body}\n}}\n");
            let mut m = parse(&text).unwrap_or_else(|e| panic!("{e}"));
            assert!(short_circuit_ifs(&m) > 0, "naga 30 lowers {body} to an if");
            fold_short_circuits(&mut m);
            assert_eq!(short_circuit_ifs(&m), 0, "folded: {body}");
            let f = &m.entry_points[0].function;
            assert!(f.local_variables.iter().filter(|(_, l)| l.name.is_none()).all(|(h, _)| is_dead_local(f, h)), "{body}: a folded local is dead");
            crate::wgsl::target::Target::recorded().validate(&m, &text).unwrap_or_else(|e| panic!("{body}: the folded module: {e}"));
        }
        // (A short circuit is always read where WGSL lets one stand, so the
        // view's other guard, a local never loaded, has no WGSL to test it.)
        let stays = ["if (gid.x < 4u && bump(gid.y) < 2u) { out[gid.x] = 1u; }"];
        for body in stays {
            let text = format!("{head}{body}\n}}\n");
            let mut m = parse(&text).unwrap_or_else(|e| panic!("{e}"));
            let before = short_circuit_ifs(&m);
            fold_short_circuits(&mut m);
            assert_eq!(short_circuit_ifs(&m), before, "stays an if: {body}");
        }
    }

    /// EVERY PARSE GOES THROUGH [`parse`]: no other call of naga's WGSL front
    /// end in the crate (so the scan runs on every module read).
    #[test]
    fn every_parse_goes_through_the_scan() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut others = Vec::new();
        for dir in ["src", "examples"] {
            let mut stack = vec![root.join(dir)];
            while let Some(d) = stack.pop() {
                for e in std::fs::read_dir(&d).expect("a source directory").flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        stack.push(p);
                    } else if p.extension().is_some_and(|x| x == "rs") && !p.ends_with("naga_shape.rs") {
                        let text = std::fs::read_to_string(&p).expect("a source file");
                        let needle = ["front::wgsl::", "parse_str("].concat();
                        if text.contains(&needle) {
                            others.push(p.display().to_string());
                        }
                    }
                }
            }
        }
        assert!(others.is_empty(), "naga's WGSL front end called outside naga_shape: {others:?}");
    }
}
