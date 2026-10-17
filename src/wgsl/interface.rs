//! A kernel's interface INFERRED FROM ITS MODULE (`docs/PLAN_general_safety_layer.md`,
//! Revision 1, order (a)): no `// interface` line, no declaration written for
//! the problem. A storage binding the entry point writes (a store or an
//! atomic, in the entry point or any function it calls) is an OUTPUT, one it
//! only reads (or never touches) an INPUT, whatever access it declares
//! (CubeCL declares every binding `read_write`); a `var<uniform>` is a
//! PARAMETER block. The workgroup size is the entry point's own.
//!
//! The kingdom's rows that carry a meaning (the input `phi` the item loop
//! reads, the output `h` whose cells the threads own, the uniform fields `d`,
//! `m`, `r_rows`, `r_off`) take the bindings the module gives them: the first
//! read-only `array<f32>` is the input, the first written `array<f32>` the
//! output, a field named for a kingdom role takes that role. Every other
//! binding and field gets the generic rows of its use (`render::Role::inferred`):
//! a load of an input, a load and a store of an output, a field's value.

use std::collections::BTreeSet;

use naga::{AddressSpace, Expression, Function, Handle, Module, Statement, StorageAccess, TypeInner};

/// How the entry point uses a binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Use {
    /// A storage binding the entry point stores to or updates atomically.
    Output,
    /// A storage binding the entry point only reads (or never touches).
    Input,
    /// A `var<uniform>` block.
    Parameter,
}

/// One binding of the module, as declared and as used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingUse {
    pub name: String,
    pub group: u32,
    pub binding: u32,
    /// The declared space as WGSL writes it: `storage, read`,
    /// `storage, read_write` or `uniform`.
    pub space: String,
    /// The declared type as WGSL writes it (`array<f32>`, `atomic<u32>`, the
    /// uniform struct's name).
    pub ty: String,
    /// The element type of a runtime-sized or fixed array (`f32`,
    /// `vec4<f32>`, `atomic<u32>`); None for a binding that is not an array.
    pub elem: Option<String>,
    pub written: bool,
    pub read: bool,
    /// Updated by an atomic (a read-modify-write).
    pub atomic: bool,
    pub role: Use,
}

/// A uniform block: its variable, its struct's name and its fields with
/// their scalar types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniformBlock {
    pub var: String,
    pub struct_name: String,
    pub fields: Vec<(String, String)>,
}

/// The module's interface: the entry point, its workgroup size, every
/// binding in (group, binding) order, and the first uniform block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleInterface {
    pub entry: String,
    pub workgroup_size: [u32; 3],
    pub bindings: Vec<BindingUse>,
    pub uniform: Option<UniformBlock>,
}

impl ModuleInterface {
    pub fn binding(&self, name: &str) -> Option<&BindingUse> {
        self.bindings.iter().find(|b| b.name == name)
    }

    pub fn outputs(&self) -> impl Iterator<Item = &BindingUse> {
        self.bindings.iter().filter(|b| b.role == Use::Output)
    }

    pub fn inputs(&self) -> impl Iterator<Item = &BindingUse> {
        self.bindings.iter().filter(|b| b.role == Use::Input)
    }
}

/// A scalar type as WGSL writes it (`u32`, `f32`, `bool`).
pub fn scalar_text(s: &naga::Scalar) -> String {
    let k = match s.kind {
        naga::ScalarKind::Float => "f",
        naga::ScalarKind::Sint => "i",
        naga::ScalarKind::Uint => "u",
        naga::ScalarKind::Bool => return "bool".into(),
        naga::ScalarKind::AbstractInt => return "abstract-int".into(),
        naga::ScalarKind::AbstractFloat => return "abstract-float".into(),
    };
    format!("{k}{}", u32::from(s.width) * 8)
}

/// A type as WGSL writes it.
pub fn type_text(module: &Module, ty: Handle<naga::Type>) -> String {
    let t = &module.types[ty];
    let scalar = scalar_text;
    match &t.inner {
        TypeInner::Scalar(s) => scalar(s),
        TypeInner::Vector { size, scalar: s } => format!("vec{}<{}>", *size as u32, scalar(s)),
        TypeInner::Matrix { columns, rows, scalar: s } => format!("mat{}x{}<{}>", *columns as u32, *rows as u32, scalar(s)),
        TypeInner::Atomic(s) => format!("atomic<{}>", scalar(s)),
        TypeInner::Array { base, size, .. } => match size {
            naga::ArraySize::Constant(n) => format!("array<{}, {n}>", type_text(module, *base)),
            naga::ArraySize::Dynamic => format!("array<{}>", type_text(module, *base)),
        },
        _ => t.name.clone().unwrap_or_else(|| format!("{:?}", t.inner)),
    }
}

/// The global variable a pointer expression is rooted at (through accesses).
pub fn pointer_root(f: &Function, mut e: Handle<Expression>) -> Option<Handle<naga::GlobalVariable>> {
    loop {
        match &f.expressions[e] {
            Expression::GlobalVariable(g) => return Some(*g),
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => e = *base,
            _ => return None,
        }
    }
}

/// What a function (and every function it calls) does to the globals.
#[derive(Default)]
struct Uses {
    written: BTreeSet<Handle<naga::GlobalVariable>>,
    read: BTreeSet<Handle<naga::GlobalVariable>>,
    atomic: BTreeSet<Handle<naga::GlobalVariable>>,
}

fn uses_of(module: &Module, f: &Function, out: &mut Uses, seen: &mut BTreeSet<Handle<Function>>) {
    for (_, e) in f.expressions.iter() {
        match e {
            Expression::Load { pointer } => {
                if let Some(g) = pointer_root(f, *pointer) {
                    out.read.insert(g);
                }
            }
            Expression::ImageLoad { image, .. } => {
                if let Some(g) = pointer_root(f, *image) {
                    out.read.insert(g);
                }
            }
            _ => {}
        }
    }
    fn walk(module: &Module, f: &Function, block: &naga::Block, out: &mut Uses, seen: &mut BTreeSet<Handle<Function>>) {
        for s in block.iter() {
            match s {
                Statement::Store { pointer, .. } => {
                    if let Some(g) = pointer_root(f, *pointer) {
                        out.written.insert(g);
                    }
                }
                Statement::Atomic { pointer, .. } => {
                    if let Some(g) = pointer_root(f, *pointer) {
                        out.written.insert(g);
                        out.read.insert(g);
                        out.atomic.insert(g);
                    }
                }
                Statement::ImageStore { image, .. } => {
                    if let Some(g) = pointer_root(f, *image) {
                        out.written.insert(g);
                    }
                }
                Statement::WorkGroupUniformLoad { pointer, .. } => {
                    if let Some(g) = pointer_root(f, *pointer) {
                        out.read.insert(g);
                    }
                }
                Statement::Call { function, .. } => {
                    if seen.insert(*function) {
                        uses_of(module, &module.functions[*function], out, seen);
                    }
                }
                Statement::Block(b) => walk(module, f, b, out, seen),
                Statement::If { accept, reject, .. } => {
                    walk(module, f, accept, out, seen);
                    walk(module, f, reject, out, seen);
                }
                Statement::Switch { cases, .. } => cases.iter().for_each(|c| walk(module, f, &c.body, out, seen)),
                Statement::Loop { body, continuing, .. } => {
                    walk(module, f, body, out, seen);
                    walk(module, f, continuing, out, seen);
                }
                _ => {}
            }
        }
    }
    walk(module, f, &f.body, out, seen);
}

/// The names of `module`'s compute entry points, in its order.
pub fn compute_entries(module: &Module) -> Vec<String> {
    module.entry_points.iter().filter(|e| e.stage == naga::ShaderStage::Compute).map(|e| e.name.clone()).collect()
}

/// The interface of `module`'s compute entry points TOGETHER (2026-10-17,
/// v24's `ingest` and `search` in one module): each entry point's inferred
/// alone ([`infer_entry`]), a binding any of them writes an output, read
/// or updated atomically by any of them so; the first entry point's name
/// and workgroup size.
pub fn infer(module: &Module) -> Result<ModuleInterface, String> {
    let names = compute_entries(module);
    let (first, rest) = names.split_first().ok_or("no compute entry point")?;
    let mut all = infer_entry(module, first)?;
    for n in rest {
        let i = infer_entry(module, n)?;
        for (a, b) in all.bindings.iter_mut().zip(&i.bindings) {
            a.written |= b.written;
            a.read |= b.read;
            a.atomic |= b.atomic;
            if b.role == Use::Output {
                a.role = Use::Output;
            }
        }
    }
    Ok(all)
}

/// The interface of the compute entry point `entry` alone: the bindings as
/// it uses them (the safety layer proves each entry point, one dispatch,
/// under its own).
pub fn infer_entry(module: &Module, entry: &str) -> Result<ModuleInterface, String> {
    let ep = module.entry_points.iter().find(|e| e.stage == naga::ShaderStage::Compute && e.name == entry).ok_or_else(|| format!("no compute entry point {entry}"))?;
    let mut uses = Uses::default();
    uses_of(module, &ep.function, &mut uses, &mut BTreeSet::new());
    let mut bindings = Vec::new();
    let mut uniform = None;
    for (h, g) in module.global_variables.iter() {
        let Some(rb) = &g.binding else { continue };
        let name = g.name.clone().unwrap_or_else(|| format!("global{}", h.index()));
        let ty = type_text(module, g.ty);
        let elem = match &module.types[g.ty].inner {
            TypeInner::Array { base, .. } => Some(type_text(module, *base)),
            _ => None,
        };
        let (space, role) = match g.space {
            AddressSpace::Storage { access } => {
                let space = if access.contains(StorageAccess::STORE) { "storage, read_write" } else { "storage, read" };
                (space.to_string(), if uses.written.contains(&h) { Use::Output } else { Use::Input })
            }
            AddressSpace::Uniform => {
                if uniform.is_none() {
                    match &module.types[g.ty].inner {
                        TypeInner::Struct { members, .. } => {
                            let fields = members.iter().enumerate().map(|(i, m)| (m.name.clone().unwrap_or_else(|| format!("f{i}")), type_text(module, m.ty))).collect();
                            uniform = Some(UniformBlock { var: name.clone(), struct_name: ty.clone(), fields });
                        }
                        // A vector block (`var<uniform> params: vec4<u32>`,
                        // UltraDim's kernels): its lanes are its fields, read
                        // `params.x` .. `params.w`; no struct is declared.
                        TypeInner::Vector { size, scalar } => {
                            let lane = scalar_text(scalar);
                            let fields = ["x", "y", "z", "w"].iter().take(*size as usize).map(|l| (l.to_string(), lane.clone())).collect();
                            uniform = Some(UniformBlock { var: name.clone(), struct_name: String::new(), fields });
                        }
                        _ => {}
                    }
                }
                ("uniform".to_string(), Use::Parameter)
            }
            AddressSpace::Handle => return Err(format!("binding {name}: images and samplers are not in the kingdom")),
            other => return Err(format!("binding {name} in {other:?}: not a storage or uniform buffer")),
        };
        bindings.push(BindingUse { name, group: rb.group, binding: rb.binding, space, ty, elem, written: uses.written.contains(&h), read: uses.read.contains(&h), atomic: uses.atomic.contains(&h), role });
    }
    bindings.sort_by_key(|b| (b.group, b.binding));
    Ok(ModuleInterface { entry: ep.name.clone(), workgroup_size: ep.workgroup_size, bindings, uniform })
}

/// [`infer`] over WGSL text.
pub fn infer_text(source: &str) -> Result<ModuleInterface, String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    infer(&module)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHIPPED: &str = include_str!("../../kingdoms/wgsl/gram/shipped.wgsl");
    const NLUCA_STEP: &str = "
@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;
@compute @workgroup_size(256, 1, 1)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x + buffer_0_global[0u];
    if row >= buffer_0_global[1u] { return; }
    buffer_2_global[row] = buffer_1_global[row] * 2.0;
}";

    /// The Gram kernel's interface from its module alone: phi read, h
    /// written, cfg the parameter block of four u32 fields.
    #[test]
    fn the_gram_interface_is_inferred_from_its_module() {
        let i = infer_text(SHIPPED).unwrap();
        assert_eq!(i.workgroup_size, [16, 16, 1]);
        let roles: Vec<(&str, Use, &str)> = i.bindings.iter().map(|b| (b.name.as_str(), b.role, b.space.as_str())).collect();
        assert_eq!(roles, vec![("phi", Use::Input, "storage, read"), ("h", Use::Output, "storage, read_write"), ("cfg", Use::Parameter, "uniform")]);
        let u = i.uniform.unwrap();
        assert_eq!((u.var.as_str(), u.struct_name.as_str()), ("cfg", "Cfg"));
        assert_eq!(u.fields.iter().map(|(f, _)| f.as_str()).collect::<Vec<_>>(), ["d", "m", "r_rows", "r_off"]);
    }

    /// Every binding declared read_write (CubeCL's way): the one the entry
    /// point stores to is the output, the rest are inputs, an unused one too.
    #[test]
    fn read_write_bindings_are_roles_by_use() {
        let i = infer_text(NLUCA_STEP).unwrap();
        let roles: Vec<(&str, Use, bool, bool)> = i.bindings.iter().map(|b| (b.name.as_str(), b.role, b.read, b.written)).collect();
        assert_eq!(roles, vec![("buffer_0_global", Use::Input, true, false), ("buffer_1_global", Use::Input, true, false), ("buffer_2_global", Use::Output, false, true), ("info", Use::Input, false, false)]);
        assert_eq!(i.entry, "step");
        assert_eq!(i.workgroup_size, [256, 1, 1]);
        assert!(i.uniform.is_none());
        assert_eq!(i.binding("buffer_2_global").and_then(|b| b.elem.clone()).as_deref(), Some("f32"));
    }

    /// A vector uniform block (UltraDim's `params: vec4<u32>`) is a parameter
    /// block whose fields are its lanes, with no struct.
    #[test]
    fn a_vector_uniform_is_a_block_of_its_lanes() {
        let src = "
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { if (gid.x < params.x) { out[params.w + gid.x] = params.y; } }";
        let u = infer_text(src).unwrap().uniform.expect("the vector block");
        assert_eq!((u.var.as_str(), u.struct_name.as_str()), ("params", ""));
        assert_eq!(u.fields, ["x", "y", "z", "w"].map(|l| (l.to_string(), "u32".to_string())).to_vec());
    }

    /// TWO ENTRY POINTS (2026-10-17): `ingest` writes `idx` atomically,
    /// `search` writes `out`; together both are outputs, each alone only its
    /// own, and `keys`, which neither writes, is an input.
    #[test]
    fn a_two_entry_module_has_the_outputs_of_both() {
        let src = "
@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read_write> idx: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>) { _ = atomicAdd(&idx[u32(keys[gid.x]) & 63u], 1u); }
@compute @workgroup_size(32)
fn search(@builtin(global_invocation_id) gid: vec3<u32>) { out[gid.x] = atomicLoad(&idx[gid.x & 63u]); }";
        let module = naga::front::wgsl::parse_str(src).unwrap();
        assert_eq!(compute_entries(&module), ["ingest", "search"]);
        let roles = |i: &ModuleInterface| i.bindings.iter().map(|b| (b.name.clone(), b.role)).collect::<Vec<_>>();
        let both = infer(&module).unwrap();
        assert_eq!((both.entry.as_str(), both.workgroup_size), ("ingest", [64, 1, 1]));
        assert_eq!(roles(&both), [("keys".to_string(), Use::Input), ("idx".to_string(), Use::Output), ("out".to_string(), Use::Output)]);
        assert!(both.binding("idx").is_some_and(|b| b.atomic && b.read));
        let search = infer_entry(&module, "search").unwrap();
        // (atomicLoad is a load: alone, `search` only reads idx.)
        assert_eq!(roles(&search), [("keys".to_string(), Use::Input), ("idx".to_string(), Use::Input), ("out".to_string(), Use::Output)]);
        let ingest = infer_entry(&module, "ingest").unwrap();
        assert_eq!(roles(&ingest), [("keys".to_string(), Use::Input), ("idx".to_string(), Use::Output), ("out".to_string(), Use::Input)]);
    }

    /// A store made in a function the entry point calls makes its binding an output.
    #[test]
    fn a_store_in_a_called_function_is_an_output() {
        let src = "
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
fn put(i: u32, v: f32) { b[i] = v; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { put(gid.x, a[gid.x]); }";
        let i = infer_text(src).unwrap();
        assert_eq!(i.binding("a").map(|b| b.role), Some(Use::Input));
        assert_eq!(i.binding("b").map(|b| b.role), Some(Use::Output));
    }
}
