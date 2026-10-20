//! The renderer of the multityped kingdom (`docs/PLAN_wgsl_gram_superopt.md`
//! §2): a typed tree whose root is `kernel.kernel` becomes WGSL text. Every
//! node is a row of the kingdom's table (`class.instance`), a per-kernel row
//! of the [`Role`] (the kernel's own bindings), a `literal.*` leaf carrying
//! its constant, or a `fallback` leaf (the typed projection's substitute,
//! given a meaning here per dual: an empty statement, the first register,
//! the form's zero). Value rows render from their naga column (an operator,
//! a builtin function, a conversion) or from the composite library below;
//! statement rows render structurally. Nothing is checked twice: the text
//! goes to naga, which validates it, and a refusal names its reason.

use std::collections::{BTreeMap, BTreeSet};

use crate::geneframe::Ty;
use crate::wgsl::loader::{WgslDual, WgslKingdom, WgslRow};
use crate::wgsl::table::FunctionTable;

/// A typed tree. `name` is a row name (`arith.add`), a per-kernel row
/// (`load.phi`), a `literal.*` row with `konst`, or `fallback`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    pub name: String,
    pub ty: Ty,
    pub konst: f64,
    pub kids: Vec<Tree>,
}

impl Tree {
    pub fn new(name: &str, ty: Ty, kids: Vec<Tree>) -> Tree {
        Tree { name: name.to_string(), ty, konst: 0.0, kids }
    }

    pub fn literal(name: &str, ty: Ty, konst: impl Into<f64>) -> Tree {
        Tree { name: name.to_string(), ty, konst: konst.into(), kids: Vec::new() }
    }

    pub fn fallback(ty: Ty) -> Tree {
        Tree { name: "fallback".into(), ty, konst: 0.0, kids: Vec::new() }
    }

    pub fn size(&self) -> usize {
        1 + self.kids.iter().map(Tree::size).sum::<usize>()
    }

    pub fn depth(&self) -> usize {
        1 + self.kids.iter().map(Tree::depth).max().unwrap_or(0)
    }

    /// The constant as text: an f32's shortest form where the value is one (`0.1`, not
    /// `0.10000000149011612`), else the f64's (an integer literal of any u32 or i32).
    pub fn konst_text(&self) -> String {
        let f = self.konst as f32;
        if f64::from(f) == self.konst { format!("{f}") } else { format!("{}", self.konst) }
    }

    /// `(name kids…)` with literals as `(literal.real 1.5)`.
    pub fn to_sexpr(&self) -> String {
        if self.kids.is_empty() {
            if self.name.starts_with("literal.") {
                return format!("({} {})", self.name, self.konst_text());
            }
            return self.name.clone();
        }
        format!("({} {})", self.name, self.kids.iter().map(Tree::to_sexpr).collect::<Vec<_>>().join(" "))
    }
}

/// A per-kernel row: the kernel's own binding read or written, rendered
/// from a text template with `{0}`, `{1}`… for the children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerKernel {
    pub name: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub template: String,
}

/// One binding of a kernel's interface: `@group(g) @binding(b) var<space> name: ty;`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub group: u32,
    pub binding: u32,
    /// `storage, read`, `storage, read_write` or `uniform`.
    pub space: String,
    /// `array<f32>`, `atomic<u32>`, the uniform struct's name.
    pub ty: String,
}

/// A field of the interface's uniform struct, and the role the kingdom gives
/// it: `d` the output's columns (the row stride of its cells), `m` the items
/// (the item loop's bound), `r_rows` the band's rows, `r_off` the band's first
/// row; empty for a field the kingdom does not read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub role: String,
    /// The field's type as WGSL writes it (`u32` for the Gram's).
    pub ty: String,
}

/// A struct the interface declares beyond its first uniform block's (an
/// element type of a storage array, another uniform block's type): its
/// members as (name, type).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructDecl {
    pub name: String,
    pub members: Vec<(String, String)>,
}

/// A uniform block beyond the first: its variable and fields as (name, type).
/// The fields have no role in the kingdom; each is read by its row
/// `load.<var>_<field>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniformDecl {
    pub var: String,
    pub fields: Vec<(String, String)>,
}

/// The fixed part of a kernel, its declared INTERFACE (docs/PLAN_constrained_mutation.md
/// section 2.4, 2026-10-12): its bindings, the uniform struct's fields and
/// their roles, the read-only input the kingdom's `phi` rows read, the output
/// whose cells the threads own and the guard that bounds a thread's cell,
/// plus the per-kernel rows over them. The prologue and every name the
/// renderer writes come from it; the safe reader renames the interface's
/// names to the kingdom's canonical roles (`phi`, `h`, `cfg.d` ...), so the
/// ownership proof, the guarded store and the checker read the interface,
/// not fixed names. The Gram kernel's h / phi / cfg are one interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Role {
    pub name: String,
    pub prologue: String,
    /// The item loop's bound, a `u32` expression over the bindings (`cfg.m`).
    pub items: String,
    pub per_kernel: Vec<PerKernel>,
    pub bindings: Vec<Binding>,
    /// The uniform struct: its variable, its type's name and its fields.
    pub uniform_var: String,
    pub uniform_struct: String,
    pub fields: Vec<Field>,
    /// The read-only input the kingdom's `phi` rows read.
    pub input: String,
    /// The output whose cells the threads own (the kingdom's `h`).
    pub output: String,
    /// The bound check of a thread's own cell, over `{row}` and `{col}`.
    pub guard: String,
    /// The structs declared beyond the first uniform block's, in order.
    pub structs: Vec<StructDecl>,
    /// The uniform blocks beyond the first.
    pub more_uniforms: Vec<UniformDecl>,
    /// The entry point's name the renderer writes (`main`; an inferred
    /// interface keeps its module's, nluca's `generated_step`).
    pub entry: String,
    /// Every compute entry point of the module, in its order (`entry` the
    /// first): one for a single kernel; two to four for a module whose
    /// entry points share the bindings (v24's `ingest` and `search`, a gene
    /// rooted at `kernel.module_N`).
    pub entries: Vec<String>,
}

/// A kernel interface as declared: what [`Role::interface`] builds a role from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interface {
    pub name: String,
    pub bindings: Vec<Binding>,
    pub uniform_var: String,
    pub uniform_struct: String,
    pub fields: Vec<Field>,
    pub input: String,
    pub output: String,
    pub guard: String,
    /// Per-kernel rows beyond the standard ones (a second input's load).
    pub extra: Vec<PerKernel>,
    pub structs: Vec<StructDecl>,
    pub more_uniforms: Vec<UniformDecl>,
}

impl Role {
    /// An interface from its declarations: the prologue (the uniform struct,
    /// then the bindings), the item bound and the per-kernel rows over the
    /// input, the output and the fields, plus the declaration's extra rows.
    pub fn interface(decl: Interface) -> Role {
        let Interface { name, bindings, uniform_var, uniform_struct, fields, input, output, guard, extra, structs, more_uniforms } = decl;
        let (name, uniform_var, uniform_struct, input, output, guard) = (name.as_str(), uniform_var.as_str(), uniform_struct.as_str(), input.as_str(), output.as_str(), guard.as_str());
        // (A module with no uniform block has no struct line.)
        let mut prologue = if uniform_struct.is_empty() { String::new() } else { format!("struct {uniform_struct} {{ {} }};", fields.iter().map(|f| format!("{}: {},", f.name, f.ty)).collect::<Vec<_>>().join(" ")) };
        for d in &structs {
            if !prologue.is_empty() {
                prologue.push('\n');
            }
            prologue.push_str(&format!("struct {} {{ {} }};", d.name, d.members.iter().map(|(n, t)| format!("{n}: {t},")).collect::<Vec<_>>().join(" ")));
        }
        for b in &bindings {
            if !prologue.is_empty() {
                prologue.push('\n');
            }
            if b.space.is_empty() {
                prologue.push_str(&format!("@group({}) @binding({}) var {}: {};", b.group, b.binding, b.name, b.ty));
            } else {
                prologue.push_str(&format!("@group({}) @binding({}) var<{}> {}: {};", b.group, b.binding, b.space, b.name, b.ty));
            }
        }
        let mut role = Role { name: name.into(), prologue, items: String::new(), per_kernel: Vec::new(), bindings, uniform_var: uniform_var.into(), uniform_struct: uniform_struct.into(), fields, input: input.into(), output: output.into(), guard: guard.into(), structs, more_uniforms, entry: "main".into(), entries: vec!["main".into()] };
        role.items = role.field("m");
        let pk = |name: &str, inputs: &[&str], output: &str, template: String| PerKernel { name: name.into(), inputs: inputs.iter().map(|s| s.to_string()).collect(), output: output.into(), template };
        let (i, o) = (input, output);
        // The kingdom's Gram rows (phi, h, the fields' roles) for an interface
        // that names an input and an output; an inferred interface without the
        // Gram's field d has its generic rows only (`Role::inferred`).
        role.per_kernel = if i.is_empty() && o.is_empty() { Vec::new() } else { vec![
            pk("load.phi", &["u32.index"], "f32.real", format!("{i}[{{0}}]")),
            pk("load.phi4", &["u32.index"], "vec4<f32>.real", format!("vec4<f32>({i}[{{0}}], {i}[{{0}} + 1u], {i}[{{0}} + 2u], {i}[{{0}} + 3u])")),
            pk("load.h", &["u32.index"], "f32.real", format!("{o}[{{0}}]")),
            pk("store.h", &["u32.index", "f32.real"], "stmt.stmt", format!("{o}[{{0}}] = {{1}};")),
            pk("load.cfg_d", &[], "u32.count", role.field("d")),
            pk("load.cfg_m", &[], "u32.count", role.field("m")),
            pk("load.cfg_r_rows", &[], "u32.count", role.field("r_rows")),
            pk("load.cfg_r_off", &[], "u32.index", role.field("r_off")),
        ] };
        // A storage atomic the interface declares (`atomic<u32>` or
        // `atomic<i32>` in a read_write buffer): its read-modify-writes, each
        // returning the old value, as per-kernel rows `atomic.<name>_<fun>`.
        for b in &role.bindings {
            let value = match b.ty.as_str() {
                "atomic<u32>" => "u32.index",
                "atomic<i32>" => "i32.int",
                _ => continue,
            };
            for (fun, wgsl) in ATOMIC_FUNS {
                role.per_kernel.push(pk(&format!("atomic.{}_{fun}", b.name), &[value], value, format!("{wgsl}(&{}, {{0}})", b.name)));
            }
        }
        // A storage ARRAY of atomics (`array<atomic<u32>>`, v24's index arena,
        // 2026-10-17): its element at the index as written (the bounds policy
        // decides an access past the binding, as for any storage access): atomicLoad
        // (`load.<name>`), atomicStore at every statement rung
        // (`store.<name>_<rung>`), and the read-modify-writes returning the
        // old value (`atomic.<name>_<fun>`). Every access to such a binding
        // is atomic, so no two of them race (layer 7).
        for b in &role.bindings {
            let Some((value, at)) = atomic_array_access(b) else { continue };
            let name = &b.name;
            role.per_kernel.push(pk(&format!("load.{name}"), &["u32.index"], value, format!("atomicLoad(&{name}[{at}])")));
            for r in STORE_RUNGS {
                role.per_kernel.push(pk(&format!("store.{name}_{r}"), &["u32.index", value], &format!("stmt.{r}"), format!("atomicStore(&{name}[{at}], {{1}});")));
            }
            for (fun, wgsl) in ATOMIC_FUNS {
                role.per_kernel.push(pk(&format!("atomic.{name}_{fun}"), &["u32.index", value], value, format!("{wgsl}(&{name}[{at}], {{1}})")));
            }
        }
        role.per_kernel.extend(extra);
        role
    }
}

/// The dual of an `i` (i32) or `u` (u32) signature letter's scalar.
fn scalar_dual(sign: char) -> &'static str {
    if sign == 'i' { "i32.int" } else { "u32.index" }
}

/// A vector of `n` lanes of scalar `t` over the form `f` (n of 1 the scalar).
fn lanes_dual(n: usize, t: &str, f: &str) -> String {
    if n == 1 { format!("{t}.{f}") } else { format!("vec{n}<{t}>.{f}") }
}

/// The texel's dual: `f32` real, `i32` int, `u32` index, as a 4-vector.
fn texel_dual(texel: &str) -> String {
    let f = match texel {
        "i32" => "int",
        "u32" => "index",
        _ => "real",
    };
    format!("vec4<{texel}>.{f}")
}

/// Every combination of signs for `n` operands: `["i", "u"]` for one,
/// `["ii", "iu", "ui", "uu"]` for two.
fn sign_signatures(n: usize) -> Vec<String> {
    (0..1usize << n).map(|m| (0..n).map(|k| if (m >> (n - 1 - k)) & 1 == 0 { 'i' } else { 'u' }).collect()).collect()
}

/// The rows of the texture and sampler bindings of an interface (the safe
/// reader names them from the operands' types): `tex.load.<b>.<sig>`,
/// `store.image.<b>.<sig>_<rung>`, `tex.dim.<b>`, `tex.layers.<b>`,
/// `tex.levels.<b>`, `tex.samples.<b>` and `tex.sample.<b>.<s>.<mode>[.cmp].<sig>`.
/// A signature is a letter for each integer operand in order (the coordinate,
/// the array index, the level or sample index): `i` for an i32, `u` for a
/// u32; a float level is `f`.
fn image_rows(i: &crate::wgsl::interface::ModuleInterface) -> Vec<PerKernel> {
    let pk = |name: String, inputs: Vec<String>, output: String, template: String| PerKernel { name, inputs, output, template };
    let mut out = Vec::new();
    let samplers: Vec<&crate::wgsl::interface::BindingUse> = i.bindings.iter().filter(|b| b.image.as_ref().is_some_and(|m| m.dim == "sampler")).collect();
    for b in &i.bindings {
        let Some(m) = b.image.as_ref().filter(|m| m.dim != "sampler") else { continue };
        let name = &b.name;
        let comps = match m.dim.as_str() {
            "1d" => 1,
            "2d" => 2,
            _ => 3,
        };
        let texel = texel_dual(&m.texel);
        let value = if m.depth { "f32.real".to_string() } else { texel.clone() };
        let has_extra = m.storage.is_none();
        let can_load = m.storage.as_deref().is_none_or(|a| a != "write") && m.dim != "cube";
        // textureLoad(t, coords, [array_index], [level | sample_index])
        if can_load {
            let n_int = 1 + usize::from(m.arrayed) + usize::from(has_extra);
            for sig in sign_signatures(n_int) {
                let c: Vec<char> = sig.chars().collect();
                let mut inputs = vec![lanes_dual(comps, if c[0] == 'i' { "i32" } else { "u32" }, if c[0] == 'i' { "int" } else { "index" })];
                inputs.extend(c[1..].iter().map(|k| scalar_dual(*k).to_string()));
                let args: Vec<String> = (0..inputs.len()).map(|k| format!("{{{k}}}")).collect();
                out.push(pk(format!("tex.load.{name}.{sig}"), inputs, value.clone(), format!("textureLoad({name}, {})", args.join(", "))));
            }
        }
        // textureStore(t, coords, [array_index], value)
        if m.storage.as_deref().is_some_and(|a| a != "read") {
            let n_int = 1 + usize::from(m.arrayed);
            for sig in sign_signatures(n_int) {
                let c: Vec<char> = sig.chars().collect();
                let mut inputs = vec![lanes_dual(comps, if c[0] == 'i' { "i32" } else { "u32" }, if c[0] == 'i' { "int" } else { "index" })];
                inputs.extend(c[1..].iter().map(|k| scalar_dual(*k).to_string()));
                inputs.push(texel.clone());
                let args: Vec<String> = (0..inputs.len()).map(|k| format!("{{{k}}}")).collect();
                for r in STORE_RUNGS {
                    out.push(pk(format!("store.image.{name}.{sig}_{r}"), inputs.clone(), format!("stmt.{r}"), format!("textureStore({name}, {});", args.join(", "))));
                }
            }
        }
        // Queries.
        let dims = lanes_dual(match m.dim.as_str() { "1d" => 1, "3d" => 3, "cube" => 2, _ => 2 }, "u32", "index");
        out.push(pk(format!("tex.dim.{name}"), Vec::new(), dims.clone(), format!("textureDimensions({name})")));
        if m.storage.is_none() && !m.multisampled {
            for sign in ['i', 'u'] {
                out.push(pk(format!("tex.dim.{name}.l{sign}"), vec![scalar_dual(sign).to_string()], dims.clone(), format!("textureDimensions({name}, {{0}})")));
            }
            out.push(pk(format!("tex.levels.{name}"), Vec::new(), "u32.index".into(), format!("textureNumLevels({name})")));
        }
        if m.arrayed {
            out.push(pk(format!("tex.layers.{name}"), Vec::new(), "u32.index".into(), format!("textureNumLayers({name})")));
        }
        if m.multisampled {
            out.push(pk(format!("tex.samples.{name}"), Vec::new(), "u32.index".into(), format!("textureNumSamples({name})")));
        }
        // textureSampleLevel / textureSampleGrad / textureSampleCompareLevel
        if m.storage.is_none() && !m.multisampled {
            for s in &samplers {
                let compare = s.image.as_ref().is_some_and(|x| x.depth);
                if compare && !m.depth {
                    continue;
                }
                let sname = &s.name;
                let coord = lanes_dual(if m.dim == "cube" { 3 } else { comps }, "f32", "real");
                let res = if m.depth { "f32.real".to_string() } else { "vec4<f32>.real".to_string() };
                let arr_sigs: Vec<String> = if m.arrayed { vec!["i".into(), "u".into()] } else { vec![String::new()] };
                for asig in &arr_sigs {
                    let mut head = vec![coord.clone()];
                    if let Some(a) = asig.chars().next() {
                        head.push(scalar_dual(a).to_string());
                    }
                    let call = |tail: &str, ins: &[String], cmp: bool| {
                        let mut inputs = head.clone();
                        let mut args: Vec<String> = Vec::new();
                        if cmp {
                            inputs.push("f32.real".into());
                        }
                        inputs.extend(ins.iter().cloned());
                        for k in 0..inputs.len() {
                            args.push(format!("{{{k}}}"));
                        }
                        // The call's order: coords, [array_index], [depth_ref], level / gradients.
                        let fun = if cmp { "textureSampleCompareLevel" } else if tail == "grad" { "textureSampleGrad" } else { "textureSampleLevel" };
                        (inputs, format!("{fun}({name}, {sname}, {})", args.join(", ")))
                    };
                    // lvl: a float level, or for a depth texture an i32 or u32 one.
                    let levels: Vec<(char, String)> = if m.depth && !compare { vec![('i', "i32.int".into()), ('u', "u32.index".into())] } else { vec![('f', "f32.real".into())] };
                    if !compare {
                        for (l, ld) in &levels {
                            let (inputs, template) = call("lvl", std::slice::from_ref(ld), false);
                            out.push(pk(format!("tex.sample.{name}.{sname}.lvl.{asig}{l}"), inputs, res.clone(), template));
                        }
                        let gradient = vec![coord.clone(), coord.clone()];
                        let (inputs, template) = call("grad", &gradient, false);
                        out.push(pk(format!("tex.sample.{name}.{sname}.grad.{asig}"), inputs, res.clone(), template));
                    } else {
                        let (inputs, template) = call("zero", &[], true);
                        out.push(pk(format!("tex.sample.{name}.{sname}.zero.cmp.{asig}"), inputs, res.clone(), template));
                    }
                }
            }
        }
    }
    out
}

/// The atomic read-modify-writes the kingdom spells, by row suffix and
/// WGSL built-in. atomicCompareExchangeWeak is not among them: naga 0.20's
/// Metal backend does not implement it (`back::msl`, `AtomicFunction::to_msl`:
/// `FeatureNotImplemented("atomic CompareExchange")`).
pub const ATOMIC_FUNS: [(&str, &str); 8] = [("add", "atomicAdd"), ("sub", "atomicSub"), ("max", "atomicMax"), ("min", "atomicMin"), ("and", "atomicAnd"), ("or", "atomicOr"), ("xor", "atomicXor"), ("exchange", "atomicExchange")];

/// A storage binding that is an array of atomics (`array<atomic<u32>>`,
/// `array<atomic<i32>, 1024>` in a `read_write` buffer): its element's value
/// dual and the text of an element index, `{0}` the index as written. None
/// for any other binding.
pub fn atomic_array_access(b: &Binding) -> Option<(&'static str, String)> {
    if b.space != "storage, read_write" {
        return None;
    }
    let inner = b.ty.strip_prefix("array<")?.strip_suffix('>')?;
    let (elem, len) = match inner.split_once(", ") {
        Some((e, n)) => (e, Some(n.parse::<u32>().ok()?)),
        None => (inner, None),
    };
    let value = match elem {
        "atomic<u32>" => "u32.index",
        "atomic<i32>" => "i32.int",
        _ => return None,
    };
    // The index as written: an access past the binding is the bounds policy's
    // (wgpu drops the write, reads zero), as for every other storage access.
    // Clamped to the last element (2026-10-17, until the CPU round trip by
    // value showed it moved 20 of run 1's seeds' dropped writes onto that
    // element) it computed something else.
    len.map_or(Some(()), |n| (n >= 1).then_some(()))?;
    Some((value, "{0}".to_string()))
}

impl Role {
    /// The text of the uniform field with the kingdom's role `role` (`d`,
    /// `m`, `r_rows`, `r_off`): `cfg.d`, `p.nc`.
    pub fn field(&self, role: &str) -> String {
        let name = self.fields.iter().find(|f| f.role == role).map_or(role, |f| f.name.as_str());
        format!("{}.{name}", self.uniform_var)
    }

    /// The bound check of a thread's own cell (row, column).
    pub fn guard_text(&self, row: &str, col: &str) -> String {
        self.guard.replace("{row}", row).replace("{col}", col)
    }
    /// The per-kernel rows typed over the kingdom's duals, for the symbol
    /// table and the sampler.
    pub fn rows(&self, kingdom: &WgslKingdom) -> Result<Vec<WgslRow>, String> {
        let dual = |name: &str| -> Result<Ty, String> {
            kingdom.duals.iter().position(|d| d.dual == name).map(|i| Ty::Wgsl(i as u16)).ok_or_else(|| format!("{}: no dual {name}", self.name))
        };
        self.per_kernel
            .iter()
            .map(|p| {
                Ok(WgslRow { name: p.name.clone(), template: format!("role:{}", self.name), inputs: p.inputs.iter().map(|s| dual(s)).collect::<Result<_, _>>()?, output: dual(&p.output)?, terminal: p.inputs.is_empty() })
            })
            .collect()
    }
}

/// The dual named `name` as a `Ty`, or why not.
pub fn dual_ty(kingdom: &WgslKingdom, name: &str) -> Result<Ty, String> {
    kingdom.duals.iter().position(|d| d.dual == name).map(|i| Ty::Wgsl(i as u16)).ok_or_else(|| format!("no dual {name}"))
}

/// The Gram's role (`docs/PLAN_wgsl_gram_superopt.md`): the three bindings
/// of `H_GRAM_ACCUM_WGSL`, the item loop bounded by `cfg.m`, and the
/// per-kernel rows that read and write them.
pub fn gram_role() -> Role {
    Role::interface(gram_interface())
}

/// The Gram's interface as declared.
fn gram_interface() -> Interface {
    let b = |name: &str, binding: u32, space: &str, ty: &str| Binding { name: name.into(), group: 0, binding, space: space.into(), ty: ty.into() };
    let f = |name: &str, role: &str| Field { name: name.into(), role: role.into(), ty: "u32".into() };
    Interface {
        name: "gram".into(),
        bindings: vec![b("phi", 0, "storage, read", "array<f32>"), b("h", 1, "storage, read_write", "array<f32>"), b("cfg", 2, "uniform", "Cfg")],
        uniform_var: "cfg".into(),
        uniform_struct: "Cfg".into(),
        fields: vec![f("d", "d"), f("m", "m"), f("r_rows", "r_rows"), f("r_off", "r_off")],
        input: "phi".into(),
        output: "h".into(),
        // The band's row, the column, and the global row (square output: d x d).
        guard: "(({row}) < cfg.r_rows) & (({col}) < cfg.d) & ((cfg.r_off + ({row})) < cfg.d)".into(),
        extra: Vec::new(),
        structs: Vec::new(),
        more_uniforms: Vec::new(),
    }
}

/// The Gram's interface with a storage atomic (2026-10-12, the coverage
/// probe `atomic_storage`): `counter: atomic<u32>` at binding 3, its
/// read-modify-writes as per-kernel rows.
pub fn gram_counter_role() -> Role {
    Role::interface(gram_counter_interface())
}

fn gram_counter_interface() -> Interface {
    let mut decl = gram_interface();
    decl.name = "gram_counter".into();
    decl.bindings.push(Binding { name: "counter".into(), group: 0, binding: 3, space: "storage, read_write".into(), ty: "atomic<u32>".into() });
    decl
}

/// A second interface (2026-10-12), UltraDim's exact re-score: a dense
/// batched cosine score of `nq` queries against `nc` candidates over `dim`
/// features. `cand` (nc x dim, row-major) is the input the kingdom's `phi`
/// rows read, `qry` (nq x dim) a second input read through its own row
/// `load.qry`, and `score` (nq x nc) the output; a thread owns one
/// (query, candidate) cell. The uniform's fields take the kingdom's roles:
/// `nc` the output's columns (d), `dim` the items the loop runs over (m),
/// `nq` the band's rows (r_rows), `q_off` the band's first query (r_off).
pub fn cosine_role() -> Role {
    let b = |name: &str, binding: u32, space: &str, ty: &str| Binding { name: name.into(), group: 0, binding, space: space.into(), ty: ty.into() };
    let f = |name: &str, role: &str| Field { name: name.into(), role: role.into(), ty: "u32".into() };
    Role::interface(Interface {
        name: "cosine".into(),
        bindings: vec![b("cand", 0, "storage, read", "array<f32>"), b("score", 1, "storage, read_write", "array<f32>"), b("p", 2, "uniform", "Params"), b("qry", 3, "storage, read", "array<f32>")],
        uniform_var: "p".into(),
        uniform_struct: "Params".into(),
        fields: vec![f("nc", "d"), f("dim", "m"), f("nq", "r_rows"), f("q_off", "r_off")],
        input: "cand".into(),
        output: "score".into(),
        // The band's query and the candidate (the output is nq x nc).
        guard: "(({row}) < p.nq) & (({col}) < p.nc)".into(),
        extra: vec![PerKernel { name: "load.qry".into(), inputs: vec!["u32.index".into()], output: "f32.real".into(), template: "qry[{0}]".into() }],
        structs: Vec::new(),
        more_uniforms: Vec::new(),
    })
}

/// The name of a role inferred from its module ([`Role::inferred`]).
pub const INFERRED: &str = "inferred";

/// The uniform fields' kingdom roles: a field of one of these names takes it.
pub const FIELD_ROLES: [&str; 4] = ["d", "m", "r_rows", "r_off"];

/// The statement rungs a store row is typed at.
pub const STORE_RUNGS: [&str; 17] = ["L0", "L1", "L2", "L3", "L4", "L5", "C0", "C1", "C2", "C3", "C4", "C5", "O0", "O1", "O2", "O3", "O4"];

/// The kingdom's value dual of an element type (`f32` → `f32.real`).
pub fn elem_dual(elem: &str) -> Option<String> {
    let form = |s: &str| match s {
        "f32" => Some("real"),
        "u32" => Some("index"),
        "i32" => Some("int"),
        _ => None,
    };
    if let Some(f) = form(elem) {
        return Some(format!("{elem}.{f}"));
    }
    // A float matrix (`mat4x4<f32>`).
    if let Some(m) = elem.strip_prefix("mat").filter(|m| m.ends_with("<f32>") && m.len() == 8 && m.as_bytes()[1] == b'x') {
        return (matches!(m.as_bytes()[0], b'2'..=b'4') && matches!(m.as_bytes()[2], b'2'..=b'4')).then(|| format!("{elem}.real"));
    }
    let inner = elem.strip_prefix("vec")?;
    let (n, rest) = inner.split_at(1);
    let s = rest.strip_prefix('<')?.strip_suffix('>')?;
    (matches!(n, "2" | "3" | "4")).then(|| form(s).map(|f| format!("vec{n}<{s}>.{f}"))).flatten()
}

impl Role {
    /// THE INTERFACE INFERRED FROM THE MODULE (docs/PLAN_general_safety_layer.md,
    /// Revision 1, order (a)): the bindings as the module declares them, the
    /// uniform block's fields (one named for a kingdom role takes it), and
    /// the rows of each binding's use: a load of every storage binding at any
    /// index, a store of every output at any index at every statement rung
    /// (layer 7 of the checker proves its write set), a field's value. A
    /// module with the Gram's field `d` gives its first read-only and first
    /// written `array<f32>` the kingdom's phi and h rows. A module that
    /// declares exactly a declared interface is that interface (the Gram's
    /// kernels read as they did when the Gram's was assumed).
    pub fn inferred(i: &crate::wgsl::interface::ModuleInterface) -> Role {
        use crate::wgsl::interface::Use;
        let bindings: Vec<Binding> = i.bindings.iter().map(|b| Binding { name: b.name.clone(), group: b.group, binding: b.binding, space: b.space.clone(), ty: b.ty.clone() }).collect();
        let (uniform_var, uniform_struct, fields) = match &i.uniform {
            Some(u) => (u.var.clone(), u.struct_name.clone(), u.fields.iter().map(|(f, ty)| Field { name: f.clone(), role: if FIELD_ROLES.contains(&f.as_str()) { f.clone() } else { String::new() }, ty: ty.clone() }).collect::<Vec<_>>()),
            None => (String::new(), String::new(), Vec::new()),
        };
        let field = |r: &str| fields.iter().find(|f| f.role == r).map(|f| format!("{uniform_var}.{}", f.name));
        let gram_rows = field("d").is_some();
        let f32_array = |b: &&crate::wgsl::interface::BindingUse| b.elem.as_deref() == Some("f32") && b.ty.starts_with("array<");
        let input = if gram_rows { i.inputs().find(f32_array).map(|b| b.name.clone()).unwrap_or_default() } else { String::new() };
        let output = if gram_rows { i.outputs().find(f32_array).map(|b| b.name.clone()).unwrap_or_default() } else { String::new() };
        let guard = match (field("r_rows"), field("d"), field("r_off")) {
            (Some(r), Some(d), Some(o)) => format!("(({{row}}) < {r}) & (({{col}}) < {d}) & (({o} + ({{row}})) < {d})"),
            (Some(r), Some(d), None) => format!("(({{row}}) < {r}) & (({{col}}) < {d})"),
            (None, Some(d), _) => format!("({{col}}) < {d}"),
            _ => String::new(),
        };
        let pk = |name: String, inputs: Vec<String>, output: String, template: String| PerKernel { name, inputs, output, template };
        let mut extra = Vec::new();
        for b in &i.bindings {
            // A uniform that is no struct and no vector of lanes (a matrix, a
            // scalar, an array): its value `load.<var>`, an array's element
            // over an index.
            if b.role == Use::Parameter && !i.uniforms.iter().any(|u| u.var == b.name) {
                if let Some(inner) = b.ty.strip_prefix("array<").and_then(|r| r.strip_suffix('>')) {
                    if let Some(dual) = elem_dual(inner.split(", ").next().unwrap_or(inner)) {
                        extra.push(pk(format!("load.{}", b.name), vec!["u32.index".into()], dual, format!("{}[{{0}}]", b.name)));
                    }
                } else if let Some(dual) = elem_dual(&b.ty) {
                    let dual = if dual == "u32.index" { "u32.count".to_string() } else { dual };
                    if dual == "u32.count" {
                        extra.push(pk(format!("uniform.{}", b.name), Vec::new(), "u32.uniform".into(), b.name.clone()));
                    }
                    extra.push(pk(format!("load.{}", b.name), Vec::new(), dual, b.name.clone()));
                }
            }
            if b.role == Use::Parameter || b.name == input || b.name == output {
                continue;
            }
            // The length of a runtime-sized array (`arrayLength(&name)`), of any
            // element type: the row `len.<name>`.
            if b.ty.starts_with("array<") && !b.ty.contains(", ") {
                extra.push(pk(format!("len.{}", b.name), Vec::new(), "u32.count".into(), format!("arrayLength(&{})", b.name)));
            }
            // The leaves of a struct binding or of a struct-element array's
            // element: a load row `load.<binding>.<path>`, a store row at each
            // statement rung when the kernel writes the binding, and the
            // atomic rows of an atomic leaf.
            let array = b.ty.starts_with("array<");
            for (path, lty) in &b.leaves {
                // The place as WGSL writes it: the binding, its element index
                // when it is an array, then the path (`#` an index, else a member).
                let mut k = usize::from(array);
                let mut place = if array { format!("{}[{{0}}]", b.name) } else { b.name.clone() };
                for seg in path.split('.') {
                    if seg == "#" {
                        place.push_str(&format!("[{{{k}}}]"));
                        k += 1;
                    } else {
                        place.push_str(&format!(".{seg}"));
                    }
                }
                let idx: Vec<String> = vec!["u32.index".to_string(); k];
                let atomic = match lty.as_str() {
                    "atomic<u32>" => Some("u32.index"),
                    "atomic<i32>" => Some("i32.int"),
                    _ => None,
                };
                if let Some(value) = atomic {
                    let with = |v: &[&str]| idx.iter().cloned().chain(v.iter().map(|s| s.to_string())).collect::<Vec<_>>();
                    extra.push(pk(format!("load.{}.{path}", b.name), idx.clone(), value.into(), format!("atomicLoad(&{place})")));
                    for r in STORE_RUNGS {
                        extra.push(pk(format!("store.{}.{path}_{r}", b.name), with(&[value]), format!("stmt.{r}"), format!("atomicStore(&{place}, {{{k}}});")));
                    }
                    for (fun, wgsl) in ATOMIC_FUNS {
                        extra.push(pk(format!("atomic.{}.{path}_{fun}", b.name), with(&[value]), value.into(), format!("{wgsl}(&{place}, {{{k}}})")));
                    }
                    continue;
                }
                let Some(dual) = elem_dual(lty) else { continue };
                extra.push(pk(format!("load.{}.{path}", b.name), idx.clone(), dual.clone(), place.clone()));
                // A u32 read from memory (or a lane of a u32 vector) is also a
                // loop bound's value: a load cannot wrap, `for (k < s.count)`.
                if lty == "u32" && b.role != Use::Output {
                    extra.push(pk(format!("load.{}.{path}", b.name), idx.clone(), "u32.xbound".into(), place.clone()));
                }
                if let Some(n) = lty.strip_prefix("vec").and_then(|r| r[..1].parse::<usize>().ok()).filter(|_| lty.ends_with("<u32>") && b.role != Use::Output) {
                    for (l, lane) in ["x", "y", "z", "w"].iter().enumerate().take(n) {
                        extra.push(pk(format!("load.{}.{path}.lane{l}", b.name), idx.clone(), "u32.xbound".into(), format!("{place}.{lane}")));
                    }
                }
                if b.role == Use::Output {
                    for r in STORE_RUNGS {
                        extra.push(pk(format!("store.{}.{path}_{r}", b.name), idx.iter().cloned().chain([dual.clone()]).collect(), format!("stmt.{r}"), format!("{place} = {{{k}}};")));
                    }
                }
            }
            // A binding that is one scalar, vector or matrix: its value, read
            // and (when the kernel writes it) written whole.
            if !array && b.leaves.is_empty() {
                if let Some(dual) = elem_dual(&b.ty) {
                    extra.push(pk(format!("load.{}", b.name), Vec::new(), dual.clone(), b.name.clone()));
                    if b.role == Use::Output {
                        for r in STORE_RUNGS {
                            extra.push(pk(format!("store.{}_{r}", b.name), vec![dual.clone()], format!("stmt.{r}"), format!("{} = {{0}};", b.name)));
                        }
                    }
                }
                continue;
            }
            let Some(dual) = b.elem.as_deref().and_then(elem_dual) else { continue };
            extra.push(pk(format!("load.{}", b.name), vec!["u32.index".into()], dual.clone(), format!("{}[{{0}}]", b.name)));
            // A u32 read from an input is also a loop bound's value (a load
            // cannot wrap; the bound renders capped): `for (j < n[0u])`.
            if dual == "u32.index" && b.role != Use::Output {
                extra.push(pk(format!("load.{}", b.name), vec!["u32.index".into()], "u32.xbound".into(), format!("{}[{{0}}]", b.name)));
            }
            if b.role == Use::Output {
                for r in STORE_RUNGS {
                    extra.push(pk(format!("store.{}_{r}", b.name), vec!["u32.index".into(), dual.clone()], format!("stmt.{r}"), format!("{}[{{0}}] = {{1}};", b.name)));
                }
            }
        }
        // Every uniform block's fields: a value row each (an array field read at
        // an index). The first block's fields that take a kingdom role are the
        // kingdom's own rows when the interface is the Gram's shape.
        for (n, u) in i.uniforms.iter().enumerate() {
            for (f, ty) in &u.fields {
                if n == 0 && FIELD_ROLES.contains(&f.as_str()) && gram_rows {
                    continue;
                }
                if let Some(inner) = ty.strip_prefix("array<").and_then(|r| r.strip_suffix('>')) {
                    // (`array<vec4<f32>, 6>`: the element type, then the length.)
                    let elem = inner.split(", ").next().unwrap_or(inner);
                    if let Some(dual) = elem_dual(elem) {
                        extra.push(pk(format!("load.{}_{f}", u.var), vec!["u32.index".into()], dual, format!("{}.{f}[{{0}}]", u.var)));
                    }
                    continue;
                }
                let Some(dual) = elem_dual(ty).map(|d| if d == "u32.index" { "u32.count".to_string() } else { d }) else { continue };
                // A u32 field is also a uniform value (a loop bound, a
                // uniform branch): its row of the uniform dual.
                if dual == "u32.count" {
                    extra.push(pk(format!("uniform.{}_{f}", u.var), Vec::new(), "u32.uniform".into(), format!("{}.{f}", u.var)));
                }
                extra.push(pk(format!("load.{}_{f}", u.var), Vec::new(), dual, format!("{}.{f}", u.var)));
            }
        }
        extra.extend(image_rows(i));
        let primary_struct = uniform_struct.clone();
        let structs: Vec<StructDecl> = i.structs.iter().filter(|(n, _)| *n != primary_struct).map(|(n, m)| StructDecl { name: n.clone(), members: m.clone() }).collect();
        let more_uniforms: Vec<UniformDecl> = i.uniforms.iter().skip(1).map(|u| UniformDecl { var: u.var.clone(), fields: u.fields.clone() }).collect();
        let decl = Interface { name: INFERRED.into(), bindings, uniform_var, uniform_struct, fields, input, output, guard, extra, structs, more_uniforms };
        let mut role = Role::interface(decl.clone());
        for known in [gram_interface(), gram_counter_interface()] {
            if (Interface { name: INFERRED.into(), ..known.clone() }) == decl {
                role = Role::interface(known);
            }
        }
        role.entry = i.entry.clone();
        role.entries = vec![i.entry.clone()];
        role
    }

    /// The root dual of a gene under this interface: `kernel.module` for a
    /// module of several entry points, else `kernel.safe`.
    pub fn root_dual(&self) -> &'static str {
        if self.entries.len() > 1 { "kernel.module" } else { "kernel.safe" }
    }

    /// This interface at one of its entry points alone (the checker proves a
    /// module entry by entry: each is its own dispatch).
    pub fn at_entry(&self, entry: &str) -> Role {
        Role { entry: entry.to_string(), entries: vec![entry.to_string()], ..self.clone() }
    }
}

/// The interface of a module's compute entry points together
/// ([`crate::wgsl::interface::infer`]: a binding any of them writes an
/// output) and their names in the module's order.
pub fn infer_entries(module: &naga::Module) -> Result<(crate::wgsl::interface::ModuleInterface, Vec<String>), String> {
    Ok((crate::wgsl::interface::infer(module)?, crate::wgsl::interface::compute_entries(module)))
}

/// The entry points of a module tree in order: the kernels of a `kernel.module_N`,
/// or of its first kid (a module) followed by the kernels a `kernel.module_ext<k>`
/// appends. A node holds at most four children, so a module of more than four
/// entry points is written as a chain of these.
pub fn module_entries(t: &Tree) -> Vec<Tree> {
    if t.name.starts_with("kernel.module_ext") {
        let mut out = t.kids.first().map(module_entries).unwrap_or_default();
        out.extend(t.kids.iter().skip(1).cloned());
        out
    } else {
        t.kids.clone()
    }
}

/// A module of the kernels `entries` (two or more): `kernel.module_N` up to
/// four, then `kernel.module_ext<k>` for each further three or fewer.
pub fn module_tree(kingdom: &WgslKingdom, entries: Vec<Tree>) -> Result<Tree, String> {
    if entries.len() < 2 {
        return Err(format!("a module of {} entry points: two or more", entries.len()));
    }
    let ty = dual_ty(kingdom, "kernel.module")?;
    let mut rest = entries.into_iter();
    let first: Vec<Tree> = rest.by_ref().take(4).collect();
    let mut tree = Tree::new(&format!("kernel.module_{}", first.len()), ty, first);
    loop {
        let more: Vec<Tree> = rest.by_ref().take(3).collect();
        if more.is_empty() {
            return Ok(tree);
        }
        let name = format!("kernel.module_ext{}", more.len());
        tree = Tree::new(&name, ty, std::iter::once(tree).chain(more).collect());
    }
}

/// The declared interfaces by name (a kernel names its own in a
/// `// interface <name>` header line; none is the interface its module gives).
pub fn role_named(name: &str) -> Result<Role, String> {
    match name {
        "gram" => Ok(gram_role()),
        "cosine" => Ok(cosine_role()),
        "gram_counter" => Ok(gram_counter_role()),
        other => Err(format!("no interface {other} (declared: gram, cosine, gram_counter)")),
    }
}

/// The interface a kernel text declares (`// interface <name>`, kept for
/// the texts written with one), else the one its module gives
/// ([`Role::inferred`]).
pub fn interface_of(source: &str) -> Result<Role, String> {
    match source.lines().map(str::trim).find_map(|l| l.strip_prefix("// interface ")) {
        Some(name) => role_named(name.trim()),
        None => {
            let module = crate::wgsl::naga_shape::parse(source)?;
            let (i, entries) = infer_entries(&module)?;
            let mut role = Role::inferred(&i);
            role.entries = entries;
            Ok(role)
        }
    }
}

/// The header line naming a kernel's interface (none for the Gram's, so a
/// Gram kernel's text is what it was before interfaces were declared).
pub fn interface_header(role: &Role) -> String {
    if role.name == "gram" || role.name == INFERRED { String::new() } else { format!("// interface {}\n", role.name) }
}

/// The shipped Gram kernel as a typed tree: the seed of the population and
/// the proof that the representation holds a real kernel.
pub fn gram_shipped_tree(kingdom: &WgslKingdom) -> Result<Tree, String> {
    let t = |n: &str| dual_ty(kingdom, n);
    let (index, count, real, flag, stmt, block) = (t("u32.index")?, t("u32.count")?, t("f32.real")?, t("bool.flag")?, t("stmt.stmt")?, t("block.block")?);
    let (ri, rf) = (t("reg<u32>.index")?, t("reg<f32>.real")?);
    let gid = Tree::new("builtin.global_invocation_id", t("vec3<u32>.index")?, vec![]);
    let reg_i = |n: &str| Tree::new(n, ri, vec![]);
    let get_i = |n: &str| Tree::new("reg.get_i", index, vec![reg_i(n)]);
    let set_i = |n: &str, v: Tree| Tree::new("reg.set_i", stmt, vec![reg_i(n), v]);
    let cnt = |n: &str| Tree::new("index.of_count", index, vec![Tree::new(n, count, vec![])]);
    let one = |s: Tree| Tree::new("block.one", block, vec![s]);
    let d_times = |x: Tree| Tree::new("index.mul", index, vec![x, cnt("load.cfg_d")]);
    let add = |a: Tree, b: Tree| Tree::new("index.add", index, vec![a, b]);
    let ge = |a: Tree, b: Tree| Tree::new("compare.ge", flag, vec![a, b]);
    // lr = gid.y; c = gid.x; if (lr >= r_rows || c >= d) return; gr = r_off + lr
    let guard = Tree::new("stmt.if", stmt, vec![Tree::new("logic.or", flag, vec![ge(get_i("reg.i1"), cnt("load.cfg_r_rows")), ge(get_i("reg.i2"), cnt("load.cfg_d"))]), one(Tree::new("stmt.return", stmt, vec![]))]);
    let setup = Tree::new("stmt.block", stmt, vec![Tree::new("block.seq4", block, vec![set_i("reg.i1", Tree::new("shape.access_1", index, vec![gid.clone()])), set_i("reg.i2", Tree::new("shape.access_0", index, vec![gid])), guard, set_i("reg.i3", add(Tree::new("load.cfg_r_off", index, vec![]), get_i("reg.i1")))])]);
    // for t: f0 = f0 + phi[t*d + gr] * phi[t*d + c]
    let base = || d_times(Tree::new("builtin.item", index, vec![]));
    let phi = |i: Tree| Tree::new("load.phi", real, vec![i]);
    let f0 = || Tree::new("reg.get_f", real, vec![Tree::new("reg.f0", rf, vec![])]);
    let body = Tree::new("reg.set_f", stmt, vec![Tree::new("reg.f0", rf, vec![]), Tree::new("arith.add", real, vec![f0(), Tree::new("arith.mul", real, vec![phi(add(base(), get_i("reg.i3"))), phi(add(base(), get_i("reg.i2")))])])]);
    let items = Tree::new("loop.items", stmt, vec![one(body)]);
    // idx = lr*d + c; h[idx] = h[idx] + f0
    let idx = || add(d_times(get_i("reg.i1")), get_i("reg.i2"));
    let store = Tree::new("store.h", stmt, vec![idx(), Tree::new("arith.add", real, vec![Tree::new("load.h", real, vec![idx()]), f0()])]);
    let grid = Tree::new("grid.of", t("grid.grid")?, vec![Tree::literal("literal.wgdim", t("u32.wgdim")?, 16.0), Tree::literal("literal.wgdim", t("u32.wgdim")?, 16.0), Tree::literal("literal.blk", t("u32.blk")?, 1.0), Tree::literal("literal.blk", t("u32.blk")?, 1.0)]);
    Ok(Tree::new("kernel.entry", t("kernel.kernel")?, vec![grid, Tree::new("block.seq3", block, vec![setup, items, store])]))
}

/// The shipped Gram kernel as a safeWGSL tree: the item loop bounded by
/// cfg.m, the store at the thread's own cell, the loads at phi indices.
/// The guard `if (lr >= r_rows || c >= d) return;` has no spelling in the
/// safe grammar (a return under a condition is `stmt.return_C0`, allowed),
/// so it is kept as such.
pub fn gram_shipped_safe_tree(kingdom: &WgslKingdom) -> Result<Tree, String> {
    let t = |n: &str| dual_ty(kingdom, n);
    let (index, count, real, flag) = (t("u32.index")?, t("u32.count")?, t("f32.real")?, t("bool.flag")?);
    let (l0, b0, l1, b1) = (t("stmt.L0")?, t("block.L0")?, t("stmt.L1")?, t("block.L1")?);
    let (rf, ri) = (t("reg<f32>.real")?, t("reg<u32>.index")?);
    let gid = || Tree::new("builtin.global_invocation_id", t("vec3<u32>.gid").expect("dual"), vec![]);
    let blk = || Tree::literal("literal.blk", t("u32.blk").expect("dual"), 1.0);
    let row = || Tree::new("safe.thread_row", t("u32.threadrow").expect("dual"), vec![gid(), blk()]);
    let col = || Tree::new("safe.thread_col", t("u32.threadcol").expect("dual"), vec![gid(), blk()]);
    let cnt = |n: &str| Tree::new("index.of_count", index, vec![Tree::new(n, count, vec![])]);
    let f0 = || Tree::new("reg.get_f", real, vec![Tree::new("reg.f0", rf, vec![])]);
    // gr = r_off + row (kept in i3 as a plain index)
    let gr = Tree::new("reg.set_i_L0", l0, vec![Tree::new("reg.i3", ri, vec![]), Tree::new("index.add", index, vec![Tree::new("load.cfg_r_off", index, vec![]), Tree::new("safe.row_as_index", index, vec![row()])])]);
    // The guard: the safe grammar has no return under a condition (a later
    // barrier would be non-uniform), so an out-of-range thread computes
    // (its loads read zero past the buffer) and the STORE is guarded.
    let lt = |a: Tree, b: Tree| Tree::new("compare.lt", flag, vec![a, b]);
    let in_range = Tree::new("logic.and", flag, vec![lt(Tree::new("safe.row_as_index", index, vec![row()]), cnt("load.cfg_r_rows")), lt(Tree::new("safe.col_as_index", index, vec![col()]), cnt("load.cfg_d"))]);
    // for t < m: f0 = f0 + phi[t*d + gr] * phi[t*d + col]
    let counter = || Tree::new("safe.counter", t("u32.counter").expect("dual"), vec![]);
    let phi = |c: Tree| Tree::new("load.phi_safe", real, vec![Tree::new("index.add", index, vec![Tree::new("index.mul", index, vec![Tree::new("safe.counter_index", index, vec![counter()]), cnt("load.cfg_d")]), c])]);
    let get_i3 = || Tree::new("reg.get_i", index, vec![Tree::new("reg.i3", ri, vec![])]);
    let body = Tree::new("reg.set_f_L1", l1, vec![Tree::new("reg.f0", rf, vec![]), Tree::new("arith.add", real, vec![f0(), Tree::new("arith.mul", real, vec![phi(get_i3()), phi(Tree::new("safe.col_as_index", index, vec![col()]))])])]);
    let items = Tree::new("loop.items_boundm", l0, vec![Tree::new("bound.m", t("u32.boundm")?, vec![]), Tree::new("block.one_L1", b1, vec![body])]);
    // The epilogue: if (in range) { h[cell] = h[cell] + f0 }
    let (o0, bo0) = (t("stmt.O0")?, t("block.O0")?);
    let store = Tree::new("stmt.if_O0", o0, vec![in_range, Tree::new("block.one_O0", bo0, vec![Tree::new("store.h_O0", o0, vec![Tree::new("safe.cell_of", t("u32.cell")?, vec![row(), col()]), f0()])])]);
    let grid = Tree::new("grid.of", t("grid.grid")?, vec![Tree::literal("literal.wgdim", t("u32.wgdim")?, 16.0), Tree::literal("literal.wgdim", t("u32.wgdim")?, 16.0), blk(), blk()]);
    Ok(Tree::new("kernel.entry_safe", t("kernel.safe")?, vec![grid, Tree::new("block.seq2_L0", b0, vec![gr, items]), Tree::new("block.one_O0", bo0, vec![store])]))
}

/// A uniform test's operator by its row's stem (`stmt.if_ule` is `<=`), and
/// the comparison row it is (`compare.le`): the tests whose body keeps its
/// rung, over two workgroup-uniform values.
pub fn uniform_test_op(stem: &str) -> Option<&'static str> {
    Some(match stem {
        "stmt.if_ule" | "stmt.if_else_ule" => "<=",
        "stmt.if_ueq" | "stmt.if_else_ueq" => "==",
        "stmt.if_une" | "stmt.if_else_une" => "!=",
        "stmt.if_ult" | "stmt.if_else_ult" => "<",
        _ => return None,
    })
}

/// [`uniform_test_op`]'s comparison row.
pub fn uniform_test_compare(stem: &str) -> Option<&'static str> {
    Some(match stem {
        "stmt.if_ule" | "stmt.if_else_ule" => "compare.le",
        "stmt.if_ueq" | "stmt.if_else_ueq" => "compare.eq",
        "stmt.if_une" | "stmt.if_else_une" => "compare.ne",
        "stmt.if_ult" | "stmt.if_else_ult" => "compare.lt",
        _ => return None,
    })
}

/// The grid a rendered kernel asks the driver for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub x: u32,
    pub y: u32,
    pub bx: u32,
    pub by: u32,
    /// The layout knob (2026-10-12): the output's rows on thread x, the fast
    /// index, and its columns on y (`grid.of_rows_x`); false is `grid.of`,
    /// columns on x. `x`, `y` stay the `@workgroup_size`; `bx` is always the
    /// column block and `by` the row block.
    pub rows_x: bool,
}

impl Grid {
    /// The thread-id component that carries the output's rows and the
    /// workgroup's extent along it.
    pub fn row_axis(&self) -> (&'static str, u32) {
        if self.rows_x { ("x", self.x) } else { ("y", self.y) }
    }

    /// The thread-id component that carries the output's columns and the
    /// workgroup's extent along it.
    pub fn col_axis(&self) -> (&'static str, u32) {
        if self.rows_x { ("y", self.y) } else { ("x", self.x) }
    }

    /// The workgroups to dispatch for `rows` output rows of `d` columns:
    /// `(along x, along y)`.
    pub fn dispatch(&self, rows: u32, d: u32) -> (u32, u32) {
        let (r, c) = (rows.div_ceil(self.row_axis().1 * self.by), d.div_ceil(self.col_axis().1 * self.bx));
        if self.rows_x { (r, c) } else { (c, r) }
    }

    /// The `// grid x y bx by` header line (`rows-x` appended for the knob).
    pub fn header(&self) -> String {
        format!("// grid {} {} {} {}{}", self.x, self.y, self.bx, self.by, if self.rows_x { " rows-x" } else { "" })
    }

    /// The grid line of one entry point of a module (`// entry <name> grid
    /// x y bx by [rows-x]`): the reader reads each entry point's from its own.
    pub fn entry_header(&self, entry: &str) -> String {
        format!("// entry {entry} grid {} {} {} {}{}", self.x, self.y, self.bx, self.by, if self.rows_x { " rows-x" } else { "" })
    }
}

/// One entry point as the renderer writes it, before the module is
/// assembled ([`Renderer::kernel`]).
struct EntryText {
    name: String,
    grid: Grid,
    /// The workgroup arrays it declares: variable, element type, length.
    wg: BTreeMap<String, (String, u32)>,
    helpers: Vec<String>,
    masked: Vec<String>,
    /// Its body's statements (what the pass-through functions are looked for in).
    body: String,
    /// Its whole function, `@compute ... fn <name>(...) { ... }`.
    function: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    pub text: String,
    /// The grid of the first entry point (of the only one, for a kernel).
    pub grid: Grid,
    /// Composite helpers the text uses, by name.
    pub helpers: Vec<String>,
    /// Statements the renderer masked to keep the text valid (a barrier
    /// in non-uniform control flow), each named; the typed projection's
    /// masking, at the statement level.
    pub masked: Vec<String>,
    /// Every entry point written, in order, with its grid.
    pub entries: Vec<(String, Grid)>,
}

pub const WGDIM: [u32; 4] = [4, 8, 16, 32];
pub const BLK: [u32; 4] = [1, 2, 4, 8];
pub const UNROLL: [u32; 3] = [2, 4, 8];
pub const WGMEM_LEN: u32 = 2048;
/// The length of each vec4 workgroup array (`wg_va`, `wg_vb`); the four
/// arrays together take 24 KB, under the 32 KB a Metal threadgroup has.
pub const WGVEC_LEN: u32 = 512;
pub const MAX_THREADS: u32 = 1024;

/// A workgroup dimension as a grid row writes it (2026-10-12: the full legal
/// range, not {4, 8, 16, 32}): any integer from 1 to [`MAX_THREADS`] (the
/// M3 Max's per-axis and per-workgroup limit is 1,024), rounded.
pub fn workgroup_dim(v: f64) -> u32 {
    if v.is_finite() { (v.round() as i64).clamp(1, i64::from(MAX_THREADS)) as u32 } else { 1 }
}

/// A workgroup shape within [`MAX_THREADS`] invocations: x as written, y cut
/// to the largest that fits beside it (64 x 32 is 64 x 16), so every grid a
/// gene spells is a shape the device runs.
pub fn workgroup_shape(x: u32, y: u32) -> (u32, u32) {
    (x, y.min(MAX_THREADS / x.max(1)).max(1))
}

/// The grid a grid row asks the driver for, as the renderer writes it: the
/// workgroup dimensions rounded and cut to the device's invocations, the
/// block factors snapped to {1, 2, 4, 8}; a fallback is 16 x 16 at 1 x 1.
/// The workgroup arrays (`grid.with_wg`) and the tile depth
/// (`grid.with_ktile`) wrap a grid without changing it.
pub fn grid_of(g: &Tree) -> Result<Grid, String> {
    let code = |t: &Tree| -> Result<u32, String> {
        match t.name.as_str() {
            "fallback" => Ok(BLK[0]),
            n if n.starts_with("literal.") => Ok(snap(&BLK, t.konst)),
            other => Err(format!("{other} is not a literal code")),
        }
    };
    match g.name.as_str() {
        "grid.of" | "grid.of_rows_x" => {
            let [x, y, bx, by] = g.kids.as_slice() else { return Err(format!("{} with {} children", g.name, g.kids.len())) };
            let dim = |t: &Tree| -> Result<u32, String> {
                match t.name.as_str() {
                    "fallback" => Ok(16),
                    n if n.starts_with("literal.") => Ok(workgroup_dim(t.konst)),
                    other => Err(format!("{other} is not a workgroup dimension")),
                }
            };
            let (gx, gy) = workgroup_shape(dim(x)?, dim(y)?);
            Ok(Grid { x: gx, y: gy, bx: code(bx)?, by: code(by)?, rows_x: g.name == "grid.of_rows_x" })
        }
        "fallback" => Ok(Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false }),
        // A grid and the kernel's workgroup arrays: the arrays are the plan's;
        // a grid and its tile depth: the depth is the references' (params.rs).
        "grid.with_wg" | "grid.with_ktile" if g.kids.len() == 2 => grid_of(&g.kids[0]),
        other => Err(format!("{other} is not a grid row")),
    }
}

/// Bytes of one element of a workgroup array's element type.
fn elem_bytes(e: &str) -> u64 {
    if e == "vec2<f32>" {
        8
    } else if e == "vec4<f32>" {
        16
    } else {
        4
    }
}

/// Bytes a workgroup plan (array -> (element type, length)) takes.
fn plan_bytes(plan: &BTreeMap<String, (String, u32)>) -> u64 {
    plan.values().map(|(e, l)| elem_bytes(e) * u64::from(*l)).sum()
}

/// The member of `set` nearest `v` (the first on a tie; a non-finite `v` is taken as 0).
pub fn snap(set: &[u32], v: f64) -> u32 {
    let v = if v.is_finite() { v } else { 0.0 };
    *set.iter().min_by(|a, b| (f64::from(**a) - v).abs().partial_cmp(&(f64::from(**b) - v).abs()).expect("finite")).expect("a non-empty set")
}

pub struct Renderer<'a> {
    pub kingdom: &'a WgslKingdom,
    pub table: &'a FunctionTable,
    pub role: &'a Role,
    per_kernel: BTreeMap<&'a str, &'a PerKernel>,
    /// A module's workgroup arrays at one length for every entry point (each
    /// entry point's arrays within the device's 32 KB at those lengths),
    /// while its entry points render; None for a single kernel.
    joint_wg: std::cell::RefCell<Option<BTreeMap<String, (String, u32)>>>,
}

/// What the renderer is inside while walking statements.
/// The tile loops' counters by depth (a fourth since run A, 2026-10-11).
const TILE_COUNTERS: [&str; 4] = ["k", "k2", "k3", "k4"];

/// Whether a block holds a `block.cont` other than one that runs to the end
/// of the block (`top`: the block itself, or the last statement's block of a
/// sequence); a continue inside a loop of the body belongs to that loop.
/// Whether a loop body holds a continue of its own loop (one inside a
/// nested loop is that loop's).
pub fn holds_continue(t: &Tree) -> bool {
    let base = safe_base(&t.name);
    if base.starts_with("loop.") {
        return false;
    }
    base == "block.cont" || base == "stmt.continue" || t.kids.iter().any(holds_continue)
}

pub fn cont_below_top(t: &Tree, top: bool) -> bool {
    let base = safe_base(&t.name);
    if base.starts_with("loop.") {
        return false;
    }
    match base {
        "block.cont" => !top || t.kids.get(1).is_some_and(|rest| cont_below_top(rest, true)),
        "block.one" | "block.seq2" | "block.seq3" | "block.seq4" => {
            let last = t.kids.len().saturating_sub(1);
            t.kids.iter().enumerate().any(|(i, k)| {
                if i == last && safe_base(&k.name) == "stmt.block" && k.kids.len() == 1 {
                    cont_below_top(&k.kids[0], top)
                } else {
                    cont_below_top(k, false)
                }
            })
        }
        _ => t.kids.iter().any(|k| cont_below_top(k, false)),
    }
}

/// A uniform row as the index row (or the role's load) it computes alike:
/// `uniform.add` is `index.add`, `uniform.cfg_d` is `load.cfg_d`.
/// The rows written as a call of a kingdom helper (`COMPOSITES`), by row and
/// helper; the reader reads a call of one back as its row.
/// (The fixed-point, hash and sign conversions were written inline until
/// 2026-10-12: their forms have no literal and their text read back as other
/// forms' rows, 25 of 1,000 random trees.)
pub const COMPOSITE_ROWS: [(&str, &str); 34] = [
    ("bits.insert_lo", "insert_lo"),
    ("bits.insert_hi", "insert_hi"),
    ("bits.pack_sign", "pack_sign"),
    ("bits.unpack_sign", "unpack_sign"),
    ("convert.q15_16_to_f32", "fx_to_f32"),
    ("convert.f32_to_q15_16", "fx_from_f32"),
    ("convert.sign_to_real", "sign_to_real"),
    ("convert.hash32_to_bits", "hash_to_bits"),
    ("convert.hash32_to_unit", "hash_to_unit"),
    ("hash.combine", "hash_combine"),
    ("hash.split_lo", "hash_lo"),
    ("hash.split_hi", "hash_hi"),
    ("fixed.add_sat", "fx_add_sat"),
    ("fixed.sub_sat", "fx_sub_sat"),
    ("fixed.mul", "fx_mul"),
    ("fixed.shl", "fx_shl"),
    ("fixed.shr", "fx_shr"),
    ("fixed.wide_mul", "fx_wide_mul"),
    ("fixed.wide_add_sat", "fx_wide_add_sat"),
    ("fixed.narrow", "fx_narrow"),
    ("hash.mix64", "hash_mix64"),
    ("hash.fmix32", "hash_fmix32"),
    ("convert.hash32_to_index", "hash32_to_index"),
    // The quantise conversions (2026-10-13, AST to AST: written inline,
    // their arithmetic read back as other rows).
    ("quantise.real_to_code4", "q_real_to_code4"),
    ("quantise.code4_to_real", "q_code4_to_real"),
    ("quantise.real_to_code8", "q_real_to_code8"),
    ("quantise.code8_to_real", "q_code8_to_real"),
    ("quantise.real_to_code16", "q_real_to_code16"),
    ("quantise.code16_to_real", "q_code16_to_real"),
    ("quantise.real_to_code1024", "q_real_to_code1024"),
    ("quantise.code1024_to_real", "q_code1024_to_real"),
    ("quantise.pack4_code8", "q_pack4_code8"),
    ("quantise.pack4_code8_hi", "q_pack4_code8_hi"),
    ("quantise.unpack_code8", "q_unpack_code8"),
];

/// The row a kingdom helper's call is (`COMPOSITE_ROWS`, and the matrix
/// helpers per shape: `inverse_N` is geom.inverse, `outer_R_C` geom.outer).
pub fn helper_row(helper: &str) -> Option<&'static str> {
    if let Some((row, _)) = COMPOSITE_ROWS.iter().find(|(_, h)| *h == helper) {
        return Some(row);
    }
    if helper.starts_with("inverse_") {
        return Some("geom.inverse");
    }
    if helper.starts_with("outer_") {
        return Some("geom.outer");
    }
    if helper.starts_with("int_div") {
        return Some("int.div");
    }
    if helper.starts_with("int_mod") {
        return Some("int.mod");
    }
    None
}

pub fn uniform_as_index(name: &str) -> Option<&'static str> {
    Some(match name {
        "uniform.cfg_m" => "load.cfg_m",
        "uniform.cfg_d" => "load.cfg_d",
        "uniform.cfg_r_rows" => "load.cfg_r_rows",
        "uniform.cfg_r_off" => "load.cfg_r_off",
        "uniform.add" => "index.add",
        "uniform.sub_floor" => "index.sub_floor",
        "uniform.mul" => "index.mul",
        "uniform.div" => "index.div",
        "uniform.min" => "index.min",
        "uniform.max" => "index.max",
        "uniform.shl" => "index.shl",
        "uniform.shr" => "index.shr",
        "uniform.and" => "index.and",
        "uniform.or" => "index.or",
        "uniform.sub" => "index.sub",
        "uniform.mod" => "index.mod",
        "uniform.round_down" => "index.round_down",
        "uniform.band_of" => "index.band_of",
        "uniform.clamp_to" => "index.clamp_to",
        "uniform.clamp" => "index.clamp",
        _ => return None,
    })
}

/// An index masked into an array of `len` elements: `&` for a power of two,
/// `min` otherwise.
fn masked_to(x: &str, len: u32) -> String {
    let last = len.max(1) - 1;
    if len.is_power_of_two() { format!("(({x}) & {last}u)") } else { format!("min(({x}), {last}u)") }
}

/// The kingdom's declared maximum trip count of a loop, MAX_LOOP_TRIP =
/// 2^24 = 16,777,216: no real workload exceeds it (production builds run
/// m ~ 420,000 items, d up to 8,192; the test shapes' 65,536 truncated them,
/// 2026-10-12). An expression bound (`bound.of_index`) renders capped at it,
/// so a bound in range computes the same; the checker ranges such a bound up
/// to it. Exact as an f32 literal (2^24), so a gene carries it whole.
pub const MAX_LOOP_TRIP: u32 = 1 << 24;

/// A loop counter read outside its loop: 0, through `lit_u32` so naga folds
/// no constant expression it would refuse (`0u - 18u`).
const OUT_OF_LOOP: &str = "lit_u32(0u)";

/// The most copies of a body nested full unrolls write out (one unroll of up
/// to 64 always is). Past it an unroll is written as its loop, the same
/// computation: a random gene of three nested unrolls (57 x 38 x 38 copies of
/// its epilogue store, 2026-10-12) rendered 17 MB of WGSL that naga took
/// 336 s to validate.
pub const UNROLL_COPIES_MAX: u32 = 64;

#[derive(Clone)]
struct Scope {
    /// The tile counters a fully unrolled loop has made literals, by name
    /// (`k` is `lit_u32(2u)` in the third copy of an unrolled body).
    unrolled: BTreeMap<&'static str, u32>,
    /// The copies the enclosing fully unrolled loops write (their counts
    /// multiplied; 1 outside any): an unroll whose copies would take the
    /// product past [`UNROLL_COPIES_MAX`] is written as its loop.
    unroll_copies: u32,
    /// Whether the innermost loop around the statement being written is a
    /// fully unrolled one (no loop in the text): a `continue` there is the
    /// rest of the copy guarded instead.
    in_unrolled_copy: bool,
    /// The kernel's workgroup arrays by variable name: element type and
    /// length (2026-10-12: the arrays and their lengths are the gene's);
    /// empty for a kernel that declares none and uses only the original
    /// four (`wg_a`, `wg_b` of 2,048 f32, `wg_va`, `wg_vb` of 512 vec4).
    wg: BTreeMap<String, (String, u32)>,
    /// The arrays the first writing of the kernel did not write (a
    /// reduction over one element writes none), so the second does not
    /// declare them: a reduction over one writes no step either.
    unwritten: BTreeSet<String>,
    in_items: bool,
    in_tile: bool,
    tile_depth: usize,
    /// The kernel's grid, for the safe rows' block offsets.
    grid: Grid,
    /// Inside an `if`: control flow naga's uniformity analysis treats as
    /// non-uniform, where a barrier is refused.
    in_cond: usize,
    /// A `return` has been emitted: everything after it is non-uniform.
    returned: bool,
    helpers: Vec<String>,
    masked: Vec<String>,
    /// Inside an arithmetic subtree whose leaves are all literals: every
    /// literal passes through `lit_*` so naga's constant evaluator never
    /// sees a literal-only operation (`1u / 0u`, `1u - 2u`, `1u << 40u`,
    /// `asin(1.4)` are refused at validation; on the device they are values
    /// the criterion judges; a nested literal-only operation is evaluated on
    /// its own, so one wrapped leaf is not enough). Everywhere else a
    /// literal is written plainly, so the text a model is shown reads as
    /// ordinary WGSL (run A: two replies copied the wrapped style and lost
    /// a brace).
    const_depth: usize,
}

impl<'a> Renderer<'a> {
    pub fn new(kingdom: &'a WgslKingdom, table: &'a FunctionTable, role: &'a Role) -> Renderer<'a> {
        Renderer { kingdom, table, role, per_kernel: role.per_kernel.iter().map(|p| (p.name.as_str(), p)).collect(), joint_wg: std::cell::RefCell::new(None) }
    }

    fn dual(&self, ty: Ty) -> Result<&WgslDual, String> {
        self.kingdom.dual(ty).ok_or_else(|| format!("{ty:?} is not a dual of the kingdom"))
    }

    /// Render a whole kernel: the root must be `kernel.entry` or a
    /// `fallback` of the kernel dual (an empty kernel), or a module of
    /// several entry points (`kernel.module_N`, or its fallback: an empty
    /// kernel at each), one per entry point the interface names, in its
    /// order; the bindings, the workgroup arrays and the helpers written
    /// once for the module.
    pub fn kernel(&self, root: &Tree) -> Result<Rendered, String> {
        if self.dual(root.ty)?.dual == "kernel.module" {
            let kids: Vec<Tree> = match root.name.as_str() {
                "fallback" => vec![Tree::fallback(dual_ty(self.kingdom, "kernel.safe")?); self.role.entries.len()],
                n if n.starts_with("kernel.module_") => module_entries(root),
                other => return Err(format!("{other} is not a module row")),
            };
            if kids.len() != self.role.entries.len() || kids.len() < 2 {
                return Err(format!("a module of {} entry points under an interface of {} ({:?})", kids.len(), self.role.entries.len(), self.role.entries));
            }
            // One length per workgroup array for the whole module (a module
            // declares each once; an entry point that halved an array another
            // did not went past 32 KB at the larger, 2026-10-17, random modules).
            let joint = self.joint_plan(&kids)?;
            *self.joint_wg.borrow_mut() = Some(joint);
            let parts = kids.iter().zip(&self.role.entries).map(|(k, e)| self.entry(k, e).map_err(|err| format!("entry point {e}: {err}"))).collect::<Result<Vec<_>, _>>();
            *self.joint_wg.borrow_mut() = None;
            return Ok(self.assemble(parts?));
        }
        let part = self.entry(root, &self.role.entry)?;
        Ok(self.assemble(vec![part]))
    }

    /// The module text of rendered entry points: the interface's prologue,
    /// the workgroup arrays any of them declares (an array two declare at
    /// different lengths at the larger; each indexes it within its own),
    /// the pass-through functions and kingdom helpers any of them calls,
    /// then each entry point's function.
    fn assemble(&self, parts: Vec<EntryText>) -> Rendered {
        let mut text = String::new();
        text.push_str(&self.role.prologue);
        text.push('\n');
        let mut wg: BTreeMap<String, (String, u32)> = BTreeMap::new();
        for p in &parts {
            for (var, (elem, len)) in &p.wg {
                let e = wg.entry(var.clone()).or_insert_with(|| (elem.clone(), *len));
                e.1 = e.1.max(*len);
            }
        }
        for (var, (elem, len)) in &wg {
            text.push_str(&format!("var<workgroup> {var}: array<{elem}, {len}>;\n"));
        }
        // The pass-through functions a literal-only expression needs (see
        // `Scope::const_depth`), declared only when a body uses them.
        for (name, ty) in [("lit_f32", "f32"), ("lit_u32", "u32"), ("lit_i32", "i32"), ("lit_bool", "bool")] {
            if parts.iter().any(|p| p.body.contains(&format!("{name}("))) {
                text.push_str(&format!("fn {name}(x: {ty}) -> {ty} {{ return x; }}\n"));
            }
        }
        let mut helpers: Vec<String> = parts.iter().flat_map(|p| p.helpers.iter().cloned()).collect();
        helpers.sort();
        helpers.dedup();
        for h in &helpers {
            // (Every helper an entry names has a body: `entry` checked it.)
            if let Some((_, body)) = COMPOSITES.iter().find(|(n, _)| n == h) {
                text.push_str(body);
                text.push('\n');
            }
        }
        for p in &parts {
            text.push_str(&p.function);
        }
        let entries: Vec<(String, Grid)> = parts.iter().map(|p| (p.name.clone(), p.grid)).collect();
        let grid = parts.first().map_or(Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false }, |p| p.grid);
        Rendered { text, grid, helpers, masked: parts.into_iter().flat_map(|p| p.masked).collect(), entries }
    }

    /// One entry point named `name` from a safe kernel root (`kernel.entry`,
    /// `kernel.entry_safe` or a fallback of either): its function's text,
    /// the workgroup arrays it declares, the helpers it calls.
    fn entry(&self, root: &Tree, name: &str) -> Result<EntryText, String> {
        // Every knob reference written as the value its kernel's knobs give.
        let resolved = crate::wgsl::params::resolve(root)?;
        let root = &resolved;
        let d = self.dual(root.ty)?;
        if d.dual != "kernel.kernel" && d.dual != "kernel.safe" {
            return Err(format!("the root is {} ({}), not kernel.kernel or kernel.safe", root.name, d.dual));
        }
        let mut scope = Scope { unrolled: BTreeMap::new(), unroll_copies: 1, in_unrolled_copy: false, wg: BTreeMap::new(), unwritten: BTreeSet::new(), in_items: false, in_tile: false, tile_depth: 0, grid: Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false }, in_cond: 0, returned: false, helpers: Vec::new(), masked: Vec::new(), const_depth: 0 };
        let (grid, body) = match root.name.as_str() {
            "kernel.entry" => {
                let [g, b] = root.kids.as_slice() else { return Err(format!("{} with {} children", root.name, root.kids.len())) };
                let grid = self.grid(g)?;
                scope.grid = grid;
                (grid, self.block(b, &mut scope)?)
            }
            "kernel.entry_safe" => {
                // A body, then the epilogue that ends in the store of the
                // thread's cell (unless the body returned on every path).
                let [g, b, e] = root.kids.as_slice() else { return Err(format!("{} with {} children, 3 expected (grid, body, epilogue)", root.name, root.kids.len())) };
                let grid = self.grid(g)?;
                scope.grid = grid;
                let fresh = scope.clone();
                scope.wg = self.workgroup_plan(g, root)?;
                let (mut text, terminated) = self.block_t(b, &mut scope)?;
                if !terminated {
                    text.push_str(&self.block(e, &mut scope)?);
                }
                // The arrays the text uses: a statement after a body that
                // returns is not written, nor its arrays' declarations (the
                // reader reads the arrays a kernel uses; 2026-10-12). Written
                // again over those, so a halving the unwritten ones forced is undone.
                let written: BTreeSet<String> = scope.wg.keys().filter(|v| text.contains(&format!("{v}["))).cloned().collect();
                if written.len() != scope.wg.len() {
                    let unwritten = scope.wg.keys().filter(|v| !written.contains(*v)).cloned().collect();
                    scope = fresh;
                    scope.grid = grid;
                    scope.unwritten = unwritten;
                    scope.wg = self.at_joint(workgroup_plan_of(self.kingdom, g, root, Some(&written))?)?;
                    let (again, terminated) = self.block_t(b, &mut scope)?;
                    text = again;
                    if !terminated {
                        text.push_str(&self.block(e, &mut scope)?);
                    }
                }
                (grid, text)
            }
            // An empty kernel of the safe root still ends in its thread's store.
            "fallback" if d.dual == "kernel.safe" => (Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false }, self.epilogue_fallback(&mut scope)?),
            "fallback" => (Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false }, String::new()),
            other => return Err(format!("{other} is not a kernel row")),
        };
        if grid.x * grid.y > MAX_THREADS {
            return Err(format!("workgroup {} x {} exceeds {MAX_THREADS} threads", grid.x, grid.y));
        }
        let wg: BTreeMap<String, (String, u32)> = if scope.wg.is_empty() {
            // The default workgroup arrays, each only when the body names it.
            [("wg_a", "f32", WGMEM_LEN), ("wg_b", "f32", WGMEM_LEN), ("wg_va", "vec4<f32>", WGVEC_LEN), ("wg_vb", "vec4<f32>", WGVEC_LEN)].iter().filter(|(var, _, _)| names_word(&body, var)).map(|(var, elem, len)| (var.to_string(), (elem.to_string(), *len))).collect()
        } else {
            scope.wg.clone()
        };
        scope.helpers.sort();
        scope.helpers.dedup();
        if let Some(h) = scope.helpers.iter().find(|h| !COMPOSITES.iter().any(|(n, _)| n == h)) {
            return Err(format!("composite {h} has no body"));
        }
        let mut text = String::new();
        // The subgroup values a kernel reads (they need the device's subgroup
        // feature, so a kernel that reads none does not declare them). naga
        // refuses @builtin(subgroup_invocation_id) and @builtin(subgroup_id)
        // on a 2-D workgroup (InvalidMultiDimensionalSubgroupBuiltIn), so
        // there the lane is the count of active lanes below it at entry (every
        // lane is active there, so it is the lane's number) and the subgroup's
        // number is the flat local index over the subgroup size (Metal's
        // numbering; a device numbering otherwise gives other values, which
        // selection judges).
        let flat = grid.y == 1;
        let lets: String = [("sg_lane", "subgroupExclusiveAdd(1u)"), ("sg_index", "lidx / sg_size")]
            .iter()
            .filter(|(var, _)| !flat && body.contains(var))
            .map(|(var, e)| format!("    let {var}: u32 = {e};\n"))
            .collect();
        let subgroup_args: String = [("sg_lane", "subgroup_invocation_id"), ("sg_size", "subgroup_size"), ("sg_count", "num_subgroups"), ("sg_index", "subgroup_id")]
            .iter()
            .filter(|(var, _)| if *var == "sg_lane" || *var == "sg_index" { flat && body.contains(var) } else { body.contains(var) || (*var == "sg_size" && lets.contains(var)) })
            .map(|(var, b)| format!(", @builtin({b}) {var}: u32"))
            .collect();
        text.push_str(&format!("@compute @workgroup_size({}, {}, 1)\nfn {name}(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>{subgroup_args}) {{\n", grid.x, grid.y));
        // The registers, counters and private arrays of the kingdom's rows:
        // 32 f32, 16 vec4, 32 u32, 8 bool registers; six f32, two vec4, three
        // u32 and two bool private arrays.
        text.push_str(&lets);
        text.push_str(&register_declarations_for(&body));
        text.push_str(&body);
        text.push_str("}\n");
        Ok(EntryText { name: name.to_string(), grid, wg, helpers: scope.helpers, masked: scope.masked, body, function: text })
    }

    fn grid(&self, g: &Tree) -> Result<Grid, String> {
        grid_of(g)
    }

    /// A bounded literal (`literal.lit64`, `literal.lit4096`) or its fallback, clamped into 1..=max.
    fn lit(&self, t: &Tree, max: u32) -> Result<u32, String> {
        match t.name.as_str() {
            "fallback" => Ok(1),
            n if n.starts_with("literal.") => Ok(if t.konst.is_finite() { (t.konst.round() as i64).clamp(1, i64::from(max)) as u32 } else { 1 }),
            other => Err(format!("{other} is not a bounded literal")),
        }
    }

    /// A literal code (`literal.wgdim`…) or its fallback, snapped to the set.
    fn code(&self, t: &Tree, set: &[u32]) -> Result<u32, String> {
        match t.name.as_str() {
            "fallback" => Ok(set[0]),
            n if n.starts_with("literal.") => Ok(snap(set, t.konst)),
            other => Err(format!("{other} is not a literal code")),
        }
    }

    fn block(&self, b: &Tree, scope: &mut Scope) -> Result<String, String> {
        Ok(self.block_t(b, scope)?.0)
    }

    /// Whether a dual is one of the epilogue's (`stmt.O<d>`, `block.O<d>`).
    fn is_epilogue(&self, ty: Ty) -> bool {
        self.kingdom.dual(ty).is_some_and(|d| (d.slot == "stmt" || d.slot == "block") && d.form.starts_with('O'))
    }

    /// THE EPILOGUE'S FALLBACK: a store, never nothing (a gene whose epilogue
    /// token is of the wrong dual still writes h): `h[cell] = h[cell] + f0`
    /// at the thread's first cell, `(gid.y * by, gid.x * bx)`, under the
    /// store's own bounds check.
    fn epilogue_fallback(&self, scope: &mut Scope) -> Result<String, String> {
        // An interface with no output cell (no h: v24's) has no store of its
        // own to end in: its epilogue's fallback is empty, as its body's is.
        if !self.per_kernel.contains_key("store.h") {
            return Ok(String::new());
        }
        let t = |n: &str| dual_ty(self.kingdom, n);
        let gid = || -> Result<Tree, String> { Ok(Tree::new("builtin.global_invocation_id", t("vec3<u32>.gid")?, vec![])) };
        let blk = |v: u32| -> Result<Tree, String> { Ok(Tree::literal("literal.blk", t("u32.blk")?, v as f32)) };
        let row = Tree::new("safe.thread_row", t("u32.threadrow")?, vec![gid()?, blk(scope.grid.by)?]);
        let col = Tree::new("safe.thread_col", t("u32.threadcol")?, vec![gid()?, blk(scope.grid.bx)?]);
        let cell = Tree::new("safe.cell_of", t("u32.cell")?, vec![row, col]);
        let f0 = Tree::new("reg.get_f", t("f32.real")?, vec![Tree::new("reg.f0", t("reg<f32>.real")?, vec![])]);
        self.stmt(&Tree::new("store.h_O0", t("stmt.O0")?, vec![cell, f0]), scope)
    }

    /// A block's text and whether it ends in a return on every path: naga
    /// refuses instructions after a `return`, so a block stops at one and
    /// what follows in the gene is unreachable and dropped.
    fn block_t(&self, b: &Tree, scope: &mut Scope) -> Result<(String, bool), String> {
        match safe_base(&b.name) {
            "fallback" if self.is_epilogue(b.ty) => Ok((self.epilogue_fallback(scope)?, false)),
            "fallback" => Ok((String::new(), false)),
            // A loop body that breaks: before, `if (c) { break; }`, after.
            "block.brk" if b.kids.len() == 3 => {
                let (before, terminated) = self.block_t(&b.kids[0], scope)?;
                if terminated {
                    return Ok((before, true));
                }
                let c = self.value(&b.kids[1], scope)?;
                let (after, terminated) = self.block_t(&b.kids[2], scope)?;
                Ok((format!("{before}    if ({c}) {{ break; }}\n{after}"), terminated))
            }
            // `if (c) { continue; }`, then the rest of its block (the epilogue's,
            // and a loop body's at an L or C rung, the rest conditional).
            "block.cont" if b.kids.len() == 2 => {
                let c = self.value(&b.kids[0], scope)?;
                let (rest, terminated) = self.block_t(&b.kids[1], scope)?;
                if scope.in_unrolled_copy {
                    // A copy of an unrolled body has no loop to continue: the
                    // rest of the copy runs when the guard does not hold.
                    return Ok((format!("    if (!({c})) {{\n{rest}    }}\n"), false));
                }
                Ok((format!("    if ({c}) {{ continue; }}\n{rest}"), terminated))
            }
            "block.one" | "block.seq2" | "block.seq3" | "block.seq4" => {
                let mut out = String::new();
                for k in &b.kids {
                    // A nested block's statements are written in place: every
                    // declaration is at the top of the kernel, so `{ }` would
                    // scope nothing, and the braces misled a model (run A).
                    let (text, terminated) = if safe_base(&k.name) == "stmt.block" && k.kids.len() == 1 { self.block_t(&k.kids[0], scope)? } else { self.stmt_t(k, scope)? };
                    out.push_str(&text);
                    if terminated {
                        return Ok((out, true));
                    }
                }
                Ok((out, false))
            }
            other => Err(format!("{other} is not a block row")),
        }
    }

    fn stmt_t(&self, s: &Tree, scope: &mut Scope) -> Result<(String, bool), String> {
        let kids = s.kids.as_slice();
        match safe_base(&s.name) {
            "loop.unroll" if kids.len() == 2 => self.unrolled(s, scope),
            "stmt.return" => Ok((self.stmt(s, scope)?, true)),
            "stmt.block" if kids.len() == 1 => self.block_t(&kids[0], scope),
            "stmt.if_else" if kids.len() == 3 => {
                let c = self.value(&kids[0], scope)?;
                scope.in_cond += 1;
                let b = self.block_t(&kids[1], scope).and_then(|b1| self.block_t(&kids[2], scope).map(|b2| (b1, b2)));
                scope.in_cond -= 1;
                let ((b1, t1), (b2, t2)) = b?;
                Ok((format!("    if ({c}) {{\n{b1}    }} else {{\n{b2}    }}\n"), t1 && t2))
            }
            _ => Ok((self.stmt(s, scope)?, false)),
        }
    }

    /// A fully unrolled loop (lit64, block): the body n times, straight-line,
    /// the counter of the depth a literal in each copy (lit_u32(i)); a copy
    /// that returns on every path ends the text (and the statement).
    fn unrolled(&self, s: &Tree, scope: &mut Scope) -> Result<(String, bool), String> {
        let count = self.lit(&s.kids[0], 64)?;
        let Some(&var) = TILE_COUNTERS.get(scope.tile_depth) else {
            return Err("a fifth nested tile loop (the safe grammar does not produce this)".into());
        };
        let copies = scope.unroll_copies.saturating_mul(count);
        // A continue below the top of the body (inside an if or a block of a
        // copy) would skip the rest of the copy, which straight-line copies
        // cannot say: such a body is written as its loop too.
        if copies > UNROLL_COPIES_MAX || cont_below_top(&s.kids[1], true) {
            // Written as its loop: the counter a variable again, a continue
            // in the body continues the loop.
            let (was, copy_was) = (scope.in_tile, scope.in_unrolled_copy);
            scope.in_tile = true;
            scope.in_unrolled_copy = false;
            scope.tile_depth += 1;
            let body = self.block_t(&s.kids[1], scope);
            scope.tile_depth -= 1;
            scope.in_tile = was;
            scope.in_unrolled_copy = copy_was;
            let (body, terminated) = body?;
            return Ok((format!("    for ({var} = 0u; {var} < {count}u; {var} = {var} + 1u) {{\n{body}    }}\n"), terminated));
        }
        let outer_copies = scope.unroll_copies;
        scope.unroll_copies = copies;
        let (was_tile, was_copy) = (scope.in_tile, scope.in_unrolled_copy);
        scope.in_tile = true;
        scope.in_unrolled_copy = true;
        scope.tile_depth += 1;
        let mut out = String::new();
        let mut result = Ok(false);
        for i in 0..count {
            scope.unrolled.insert(var, i);
            match self.block_t(&s.kids[1], scope) {
                Ok((b, terminated)) => {
                    out.push_str(&b);
                    if terminated {
                        result = Ok(true);
                        break;
                    }
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        scope.unrolled.remove(var);
        scope.tile_depth -= 1;
        scope.in_tile = was_tile;
        scope.in_unrolled_copy = was_copy;
        scope.unroll_copies = outer_copies;
        result.map(|terminated| (out, terminated))
    }

    fn stmt(&self, s: &Tree, scope: &mut Scope) -> Result<String, String> {
        let kids = s.kids.as_slice();
        let n = |want: usize| -> Result<(), String> { if kids.len() == want { Ok(()) } else { Err(format!("{} with {} children, {want} expected", s.name, kids.len())) } };
        match safe_base(&s.name) {
            "fallback" if self.is_epilogue(s.ty) => self.epilogue_fallback(scope),
            "fallback" => Ok(String::new()),
            // ---- safeWGSL statements ----
            "loop.items_bounded" => {
                // (bound, block): for (t = 0u; t < bound; t = t + 1u), the counter the loop's own.
                n(2)?;
                if scope.in_items {
                    return Err("a long loop inside a long loop (the safe grammar does not produce this)".into());
                }
                let bound = self.value(&kids[0], scope)?;
                scope.in_items = true;
                let body = self.block(&kids[1], scope);
                scope.in_items = false;
                Ok(format!("    for (t = 0u; t < {bound}; t = t + 1u) {{\n{}    }}\n", body?))
            }
            "loop.items_unrolled_bounded" => {
                n(3)?;
                if scope.in_items {
                    return Err("a long loop inside a long loop (the safe grammar does not produce this)".into());
                }
                let u = self.code(&kids[0], &UNROLL)?;
                let bound = self.value(&kids[1], scope)?;
                scope.in_items = true;
                let body = self.block_t(&kids[2], scope);
                scope.in_items = false;
                let (body, terminated) = body?;
                // A body that returns runs once; one that continues would
                // skip the later copies and the counter's reset (the
                // continue goes to the step by u): each is written as its
                // loop (2026-10-13, AST to AST: the copies stepped t wrongly).
                if terminated || holds_continue(&kids[2]) {
                    return Ok(format!("    for (t = 0u; t < {bound}; t = t + 1u) {{\n{body}    }}\n"));
                }
                let mut out = format!("    for (t = 0u; t + {u}u <= {bound}; t = t + {u}u) {{\n");
                for j in 0..u {
                    if j > 0 {
                        out.push_str("        t = t + 1u;\n");
                    }
                    out.push_str(&body);
                }
                out.push_str(&format!("        t = t - {}u;\n    }}\n    for (; t < {bound}; t = t + 1u) {{\n{body}    }}\n", u - 1));
                Ok(out)
            }
            "loop.items_strided_bounded" => {
                // (bound, stride, block): for (t = 0u; t < bound; t = t + stride)
                n(3)?;
                if scope.in_items {
                    return Err("a long loop inside a long loop (the safe grammar does not produce this)".into());
                }
                let bound = self.value(&kids[0], scope)?;
                let stride = self.lit(&kids[1], 4096)?;
                scope.in_items = true;
                let body = self.block(&kids[2], scope);
                scope.in_items = false;
                Ok(format!("    for (t = 0u; t < {bound}; t = t + {stride}u) {{\n{}    }}\n", body?))
            }
            "loop.items_cont_bounded" | "loop.items_cont_strided_bounded" => {
                // (bound, [stride,] block): for (; t < bound; t = t + stride): the
                // counter continues from where the earlier item loop left it.
                let strided = safe_base(&s.name) == "loop.items_cont_strided_bounded";
                n(if strided { 3 } else { 2 })?;
                if scope.in_items {
                    return Err("a long loop inside a long loop (the safe grammar does not produce this)".into());
                }
                let bound = self.value(&kids[0], scope)?;
                let stride = if strided { self.lit(&kids[1], 4096)? } else { 1 };
                scope.in_items = true;
                let body = self.block(&kids[if strided { 2 } else { 1 }], scope);
                scope.in_items = false;
                Ok(format!("    for (; t < {bound}; t = t + {stride}u) {{\n{}    }}\n", body?))
            }
            "loop.tilex_bounded" | "loop.tilex_strided_bounded" => {
                // (bound expression, [step,] block): for (k = 0u; k < (expr); k = k + step),
                // the counter of the depth; the bound as written, never capped.
                let strided = safe_base(&s.name) == "loop.tilex_strided_bounded";
                n(if strided { 3 } else { 2 })?;
                let bound = self.value(&kids[0], scope)?;
                let step = if strided { self.lit(&kids[1], 4096)? } else { 1 };
                let Some(var) = TILE_COUNTERS.get(scope.tile_depth) else {
                    return Err("a fifth nested tile loop (the safe grammar does not produce this)".into());
                };
                let (was, copy_was) = (scope.in_tile, scope.in_unrolled_copy);
                scope.in_tile = true;
                // A loop in the text: a continue in its body continues it.
                scope.in_unrolled_copy = false;
                scope.tile_depth += 1;
                let body = self.block(&kids[if strided { 2 } else { 1 }], scope);
                scope.tile_depth -= 1;
                scope.in_tile = was;
                scope.in_unrolled_copy = copy_was;
                Ok(format!("    for ({var} = 0u; {var} < {bound}; {var} = {var} + {step}u) {{\n{}    }}\n", body?))
            }
            "loop.tile_bounded" | "loop.tile_strided_bounded" => {
                // (lit64, [step,] block): for (k = 0u; k < lit; k = k + step); nested
                // tile loops count on k2, then k3.
                let strided = safe_base(&s.name) == "loop.tile_strided_bounded";
                n(if strided { 3 } else { 2 })?;
                let count = self.lit(&kids[0], 64)?;
                let step = if strided { self.lit(&kids[1], 64)? } else { 1 };
                let Some(var) = TILE_COUNTERS.get(scope.tile_depth) else {
                    return Err("a fifth nested tile loop (the safe grammar does not produce this)".into());
                };
                let (was, copy_was) = (scope.in_tile, scope.in_unrolled_copy);
                scope.in_tile = true;
                // A loop in the text: a continue in its body continues it.
                scope.in_unrolled_copy = false;
                scope.tile_depth += 1;
                let body = self.block(&kids[if strided { 2 } else { 1 }], scope);
                scope.tile_depth -= 1;
                scope.in_tile = was;
                scope.in_unrolled_copy = copy_was;
                Ok(format!("    for ({var} = 0u; {var} < {count}u; {var} = {var} + {step}u) {{\n{}    }}\n", body?))
            }
            "store.h_safe" if self.role.output.is_empty() => Err("our software has a bug: store.h_safe under an interface with no output cell (render::role_spells withholds it)".into()),
            "store.h_safe" => {
                // THE GUARDED STORE (the proposals of 2026-10-10/11, class 1: a
                // guard and its store built apart could disagree). Every store
                // to h carries the bounds check of ITS OWN cell: the row below
                // the band, the column below d, the global row below d. A
                // kernel's own guard, when it has one, is a second check.
                n(2)?;
                let cell = self.value(&kids[0], scope)?;
                let (row, col) = if safe_base(&kids[0].name) == "safe.cell_of" && kids[0].kids.len() == 2 {
                    (self.value(&kids[0].kids[0], scope)?, self.value(&kids[0].kids[1], scope)?)
                } else {
                    // A fallback cell is the thread's own first cell.
                    (self.own_fallback("u32.threadrow", scope)?.unwrap_or_default(), self.own_fallback("u32.threadcol", scope)?.unwrap_or_default())
                };
                let (guard, h) = (self.role.guard_text(&row, &col), &self.role.output);
                Ok(format!("    if ({guard}) {{ {h}[{cell}] = {h}[{cell}] + {}; }}\n", self.value(&kids[1], scope)?))
            }
            // THE MIRROR STORE (the symmetric half, 2026-10-14): h at the
            // mirror of the thread's own cell, under that cell's own bounds
            // check (in a full-H kernel, `// mode full`, the mirror of an
            // in-range cell is in range).
            "store.h_mirror" => {
                n(2)?;
                let cell = self.value(&kids[0], scope)?;
                let (row, col) = if safe_base(&kids[0].name) == "safe.mirror_of" && kids[0].kids.len() == 2 {
                    (self.value(&kids[0].kids[0], scope)?, self.value(&kids[0].kids[1], scope)?)
                } else {
                    (self.own_fallback("u32.threadrow", scope)?.unwrap_or_default(), self.own_fallback("u32.threadcol", scope)?.unwrap_or_default())
                };
                let (guard, h) = (self.role.guard_text(&row, &col), &self.role.output);
                Ok(format!("    if ({guard}) {{ {h}[{cell}] = {h}[{cell}] + {}; }}\n", self.value(&kids[1], scope)?))
            }
            "reg.set_v_lane" => {
                n(3)?;
                Ok(format!("    {}[({}) & 3u] = {};\n", self.register(&kids[0])?, self.value(&kids[1], scope)?, self.value(&kids[2], scope)?))
            }
            "wg.set_safe" => {
                n(3)?;
                let a = self.wgmem(&kids[0])?;
                let len = self.wg_len(a, scope);
                Ok(format!("    {a}[{}] = {};\n", self.index_text(&kids[1], len, scope)?, self.value(&kids[2], scope)?))
            }
            // THE WORKGROUP PUT: this thread's own element, below the array's
            // length, between two workgroup barriers (masked, as a barrier is,
            // in non-uniform control flow, which the kingdom does not draw).
            "wg.put" => {
                n(3)?;
                let a = self.wgmem(&kids[0])?;
                let len = self.wg_len(a, scope);
                let (i, v) = (self.wg_own(&kids[1], len, scope)?, self.value(&kids[2], scope)?);
                if scope.in_cond > 0 || scope.returned {
                    scope.masked.push("wg.put in non-uniform control flow".into());
                    return Ok(format!("    if (({i}) < {len}u) {{ {a}[{i}] = {v}; }}\n"));
                }
                Ok(format!("    workgroupBarrier();\n    if (({i}) < {len}u) {{ {a}[{i}] = {v}; }}\n    workgroupBarrier();\n"))
            }
            "wg.set_v_lane" => {
                n(4)?;
                let a = self.wgmem(&kids[0])?;
                let len = self.wg_len(a, scope);
                Ok(format!("    {a}[{}][({}) & 3u] = {};\n", self.index_text(&kids[1], len, scope)?, self.value(&kids[2], scope)?, self.value(&kids[3], scope)?))
            }
            "parr.set_v_lane" => {
                n(4)?;
                let a = self.parr_name(&kids[0])?;
                Ok(format!("    {a}[{}][({}) & 3u] = {};\n", self.index_text(&kids[1], array_len(a), scope)?, self.value(&kids[2], scope)?, self.value(&kids[3], scope)?))
            }
            "parr.set_safe" => {
                n(3)?;
                let a = self.parr_name(&kids[0])?;
                Ok(format!("    {a}[{}] = {};\n", self.index_text(&kids[1], array_len(a), scope)?, self.value(&kids[2], scope)?))
            }
            // Conditional statements (no barrier) written in place: a loop whose
            // bound is not uniform, read at its conditional rung.
            "stmt.cblock" => {
                n(1)?;
                self.block(&kids[0], scope)
            }
            "stmt.block" => {
                n(1)?;
                Ok(format!("    {{\n{}    }}\n", self.block(&kids[0], scope)?))
            }
            // A test of two workgroup-uniform values: its body keeps its rung
            // (a barrier there is in uniform control), so it is no condition
            // the renderer masks a barrier under.
            "stmt.if_ule" | "stmt.if_ueq" | "stmt.if_une" | "stmt.if_ult" => {
                n(3)?;
                let (a, b) = (self.value(&kids[0], scope)?, self.value(&kids[1], scope)?);
                let op = uniform_test_op(safe_base(&s.name)).unwrap_or("<=");
                Ok(format!("    if (({a}) {op} ({b})) {{\n{}    }}\n", self.block(&kids[2], scope)?))
            }
            // A test of a workgroup-uniform predicate: its bodies keep the rung.
            "stmt.if_uflag" => {
                n(2)?;
                let c = self.value(&kids[0], scope)?;
                Ok(format!("    if ({c}) {{\n{}    }}\n", self.block(&kids[1], scope)?))
            }
            "stmt.if_else_uflag" => {
                n(3)?;
                let c = self.value(&kids[0], scope)?;
                let then = self.block(&kids[1], scope)?;
                Ok(format!("    if ({c}) {{\n{then}    }} else {{\n{}    }}\n", self.block(&kids[2], scope)?))
            }
            "stmt.if_else_ule" | "stmt.if_else_ueq" | "stmt.if_else_une" | "stmt.if_else_ult" => {
                n(4)?;
                let (a, b) = (self.value(&kids[0], scope)?, self.value(&kids[1], scope)?);
                let op = uniform_test_op(safe_base(&s.name)).unwrap_or("<=");
                let then = self.block(&kids[2], scope)?;
                Ok(format!("    if (({a}) {op} ({b})) {{\n{then}    }} else {{\n{}    }}\n", self.block(&kids[3], scope)?))
            }
            "stmt.if" => {
                n(2)?;
                let c = self.value(&kids[0], scope)?;
                scope.in_cond += 1;
                let b = self.block(&kids[1], scope);
                scope.in_cond -= 1;
                Ok(format!("    if ({c}) {{\n{}    }}\n", b?))
            }
            "stmt.if_else" => {
                n(3)?;
                let c = self.value(&kids[0], scope)?;
                scope.in_cond += 1;
                let b = self.block(&kids[1], scope).and_then(|b1| self.block(&kids[2], scope).map(|b2| (b1, b2)));
                scope.in_cond -= 1;
                let (b1, b2) = b?;
                Ok(format!("    if ({c}) {{\n{b1}    }} else {{\n{b2}    }}\n"))
            }
            "stmt.return" => {
                n(0)?;
                if scope.in_cond > 0 {
                    scope.returned = true;
                }
                Ok("    return;\n".into())
            }
            "atomic.store_u" | "atomic.store_i" => {
                n(3)?;
                let a = self.wgmem(&kids[0])?;
                let len = self.wg_len(a, scope);
                Ok(format!("    atomicStore(&{a}[{}], {});\n", self.index_text(&kids[1], len, scope)?, self.value(&kids[2], scope)?))
            }
            "stmt.discard_u" | "stmt.discard_i" => {
                n(1)?;
                Ok(format!("    _ = {};\n", self.value(&kids[0], scope)?))
            }
            "wg.reduce_argmax" | "wg.reduce_argmin" => {
                // v[0..n) and i[0..n) into element 0, i moving with v: a
                // barrier, then for s = n/2, .., 1 `if (lidx < s) { if (v[lidx +
                // s] > v[lidx]) { v[lidx] = v[lidx + s]; i[lidx] = i[lidx + s];
                // } }` and a barrier (< for argmin); n at most both arrays'
                // elements, rounded down to a power of two.
                n(3)?;
                if scope.in_cond > 0 || scope.returned {
                    scope.masked.push(format!("{} in non-uniform control flow", safe_base(&s.name)));
                    return Ok("    // reduction masked: non-uniform control flow\n".into());
                }
                let (a, b) = (self.wgmem(&kids[0])?, self.wgmem(&kids[1])?);
                let len_of = |x: &str, scope: &Scope| if scope.unwritten.contains(x) { 1 } else { self.wg_len(x, scope).max(1) };
                let (la, lb) = (len_of(a, scope), len_of(b, scope));
                let len = la.min(lb);
                let count = self.lit(&kids[2], 4096)?;
                let count = (1u32 << (31 - count.leading_zeros())).min(1u32 << (31 - len.leading_zeros()));
                let mask = |x: String, l: u32| if l.is_power_of_two() { format!("(({x}) & {}u)", l - 1) } else { format!("min(({x}), {}u)", l - 1) };
                let cmp = if safe_base(&s.name) == "wg.reduce_argmax" { ">" } else { "<" };
                let mut out = String::from("    workgroupBarrier();\n");
                let mut stride = count / 2;
                while stride > 0 {
                    let (ai, aj) = (mask("lidx".into(), la), mask(format!("lidx + {stride}u"), la));
                    let (bi, bj) = (mask("lidx".into(), lb), mask(format!("lidx + {stride}u"), lb));
                    out.push_str(&format!("    if (lidx < {stride}u) {{ if ({a}[{aj}] {cmp} {a}[{ai}]) {{ {a}[{ai}] = {a}[{aj}]; {b}[{bi}] = {b}[{bj}]; }} }}\n    workgroupBarrier();\n"));
                    stride /= 2;
                }
                Ok(out)
            }
            "wg.reduce_add" | "wg.reduce_max" | "wg.reduce_min" => {
                // a[0..n) into a[0]: a barrier, then for s = n/2, n/4, .., 1
                // `if (lidx < s) { a[lidx] = a[lidx] op a[lidx + s]; }` and a
                // barrier; n rounded down to a power of two and to at most the
                // array's elements (a reduction wider than its array had two
                // threads write one element through the mask: 2026-10-14, the
                // checker's layer 7). Masked, as a barrier is, in non-uniform
                // control flow.
                n(2)?;
                if scope.in_cond > 0 || scope.returned {
                    scope.masked.push(format!("{} in non-uniform control flow", safe_base(&s.name)));
                    return Ok("    // reduction masked: non-uniform control flow\n".into());
                }
                let a = self.wgmem(&kids[0])?;
                let len = if scope.unwritten.contains(a) { 1 } else { self.wg_len(a, scope).max(1) };
                let count = self.lit(&kids[1], 4096)?;
                let count = (1u32 << (31 - count.leading_zeros())).min(1u32 << (31 - len.leading_zeros()));
                let mask = |x: String| if len.is_power_of_two() { format!("(({x}) & {}u)", len - 1) } else { format!("min(({x}), {}u)", len - 1) };
                let combine = |x: String, y: String| match safe_base(&s.name) {
                    "wg.reduce_max" => format!("max({x}, {y})"),
                    "wg.reduce_min" => format!("min({x}, {y})"),
                    _ => format!("({x} + {y})"),
                };
                let mut out = String::from("    workgroupBarrier();\n");
                let mut stride = count / 2;
                while stride > 0 {
                    let (i, j) = (mask("lidx".into()), mask(format!("lidx + {stride}u")));
                    out.push_str(&format!("    if (lidx < {stride}u) {{ {a}[{i}] = {}; }}\n    workgroupBarrier();\n", combine(format!("{a}[{i}]"), format!("{a}[{j}]"))));
                    stride /= 2;
                }
                Ok(out)
            }
            "stmt.subgroup_barrier" => {
                n(0)?;
                if scope.in_cond > 0 || scope.returned {
                    scope.masked.push("stmt.subgroup_barrier in non-uniform control flow".into());
                    return Ok("    // subgroup barrier masked: non-uniform control flow\n".into());
                }
                Ok("    subgroupBarrier();\n".into())
            }
            "stmt.storage_barrier" => {
                n(0)?;
                if scope.in_cond > 0 || scope.returned {
                    scope.masked.push("stmt.storage_barrier in non-uniform control flow".into());
                    return Ok("    // storage barrier masked: non-uniform control flow\n".into());
                }
                Ok("    storageBarrier();\n".into())
            }
            "stmt.barrier" => {
                n(0)?;
                // WGSL's uniformity rule: a barrier only in uniform control
                // flow. A conditional on anything a thread computes is not
                // uniform, nor is anything after a conditional return.
                if scope.in_cond > 0 || scope.returned {
                    scope.masked.push("stmt.barrier in non-uniform control flow".into());
                    return Ok("    // barrier masked: non-uniform control flow\n".into());
                }
                Ok("    workgroupBarrier();\n".into())
            }
            "loop.items" => {
                n(1)?;
                if scope.in_items {
                    return Err("loop.items nested in loop.items".into());
                }
                scope.in_items = true;
                let body = self.block(&kids[0], scope);
                scope.in_items = false;
                Ok(format!("    for (t = 0u; t < {}; t = t + 1u) {{\n{}    }}\n", self.role.items, body?))
            }
            "loop.items_unrolled" => {
                n(2)?;
                if scope.in_items {
                    return Err("loop.items_unrolled nested in loop.items".into());
                }
                let u = self.code(&kids[0], &UNROLL)?;
                scope.in_items = true;
                let body = self.block_t(&kids[1], scope);
                scope.in_items = false;
                let (body, terminated) = body?;
                if terminated || holds_continue(&kids[1]) {
                    // A body that returns runs once; unrolling it would put
                    // the step after the return. One that continues would
                    // skip the later copies and the counter's reset.
                    return Ok(format!("    for (t = 0u; t < {}; t = t + 1u) {{\n{body}    }}\n", self.role.items));
                }
                // The body repeated with `t` stepped inside the iteration, then the remainder.
                let mut out = format!("    for (t = 0u; t + {u}u <= {m}; t = t + {u}u) {{\n", m = self.role.items);
                for j in 0..u {
                    if j > 0 {
                        out.push_str("        t = t + 1u;\n");
                    }
                    out.push_str(&body);
                }
                out.push_str(&format!("        t = t - {}u;\n    }}\n    for (; t < {m}; t = t + 1u) {{\n{body}    }}\n", u - 1, m = self.role.items));
                Ok(out)
            }
            "loop.tile" => {
                n(2)?;
                if scope.in_tile {
                    return Err("loop.tile nested in loop.tile".into());
                }
                let count = self.code(&kids[0], &WGDIM)?;
                scope.in_tile = true;
                let body = self.block(&kids[1], scope);
                scope.in_tile = false;
                Ok(format!("    for (k = 0u; k < {count}u; k = k + 1u) {{\n{}    }}\n", body?))
            }
            "wg.uload_i" | "wg.uload_f" => {
                n(3)?;
                let a = self.wgmem(&kids[1])?;
                let len = self.wg_len(a, scope);
                Ok(format!("    {} = workgroupUniformLoad(&{a}[{}]);\n", self.register(&kids[0])?, self.index_text(&kids[2], len, scope)?))
            }
            "reg.set_f" | "reg.set_v" | "reg.set_i" | "reg.set_b" | "reg.set_n" | "reg.set_u" | "reg.set_w" | "reg.set_x" | "reg.set_y" | "reg.set_z" => {
                n(2)?;
                Ok(format!("    {} = {};\n", self.register(&kids[0])?, self.value(&kids[1], scope)?))
            }
            base if is_extra_register_row(base) => {
                n(2)?;
                Ok(format!("    {} = {};\n", self.register(&kids[0])?, self.value(&kids[1], scope)?))
            }
            "wg.set" => {
                n(3)?;
                let a = self.wgmem(&kids[0])?;
                let len = self.wg_len(a, scope);
                Ok(format!("    {a}[{}] = {};\n", masked_to(&self.value(&kids[1], scope)?, len), self.value(&kids[2], scope)?))
            }
            other => {
                // (A role's store at a statement rung is named with its rung.)
                if let Some(p) = self.per_kernel.get(s.name.as_str()).or_else(|| self.per_kernel.get(other)) {
                    let d = self.dual(s.ty)?;
                    if d.slot != "stmt" {
                        return Err(format!("{other} is a value row used as a statement"));
                    }
                    return Ok(format!("    {}\n", self.fill(p, s, scope)?));
                }
                Err(format!("{other} is not a statement row"))
            }
        }
    }

    fn register(&self, r: &Tree) -> Result<String, String> {
        register_name(self.kingdom, r)
    }

    /// A workgroup array leaf's variable name ([`wgmem_var`]).
    fn wgmem(&self, w: &Tree) -> Result<&'static str, String> {
        wgmem_var(self.kingdom, w)
    }

    /// The workgroup memory a kernel asks for, in bytes, before the renderer
    /// halves any array to fit 32 KB (knob references resolved): the knob
    /// mutation's join keeps it within the device's limit, so a child is
    /// never silently given shorter arrays than its references assume.
    pub fn workgroup_request(&self, root: &Tree) -> Result<u64, String> {
        workgroup_request(self.kingdom, root)
    }

    /// The kernel's workgroup arrays ([`workgroup_plan`]).
    fn workgroup_plan(&self, grid: &Tree, root: &Tree) -> Result<BTreeMap<String, (String, u32)>, String> {
        self.at_joint(workgroup_plan(self.kingdom, grid, root)?)
    }

    /// A plan's arrays at the module's joint lengths, when a module renders.
    fn at_joint(&self, plan: BTreeMap<String, (String, u32)>) -> Result<BTreeMap<String, (String, u32)>, String> {
        let joint = self.joint_wg.borrow();
        let Some(j) = joint.as_ref() else { return Ok(plan) };
        plan.into_keys().map(|v| j.get(&v).cloned().map(|e| (v.clone(), e)).ok_or_else(|| format!("our software has a bug: the workgroup array {v} is not in the module's joint plan"))).collect()
    }

    /// The module's joint plan: every entry point's requested arrays, each at
    /// the largest length any requests, the largest array an entry point
    /// over 32 KB uses halved until every entry point fits.
    fn joint_plan(&self, kids: &[Tree]) -> Result<BTreeMap<String, (String, u32)>, String> {
        let mut uses: Vec<BTreeSet<String>> = Vec::new();
        let mut joint: BTreeMap<String, (String, u32)> = BTreeMap::new();
        for k in kids {
            let r = crate::wgsl::params::resolve(k)?;
            let Some(grid) = r.kids.first().filter(|_| r.name == "kernel.entry_safe") else { continue };
            let req = workgroup_plan_requested_of(self.kingdom, grid, &r, None)?;
            uses.push(req.keys().cloned().collect());
            for (v, (e, l)) in req {
                let at = joint.entry(v).or_insert((e, l));
                at.1 = at.1.max(l);
            }
        }
        let bytes = |j: &BTreeMap<String, (String, u32)>, u: &BTreeSet<String>| -> u64 { u.iter().filter_map(|v| j.get(v)).map(|(e, l)| elem_bytes(e) * u64::from(*l)).sum() };
        while let Some(over) = uses.iter().find(|u| bytes(&joint, u) > 32768) {
            let largest = over.iter().filter_map(|v| joint.get(v).map(|(e, l)| (elem_bytes(e) * u64::from(*l), std::cmp::Reverse(v.clone())))).max().map(|(_, std::cmp::Reverse(v))| v).expect("an entry point over 32 KB uses an array");
            let at = joint.get_mut(&largest).expect("present");
            if at.1 == 1 {
                return Err(format!("our software has a bug: an entry point's workgroup arrays exceed 32 KB at one element each ({largest})"));
            }
            at.1 = (at.1 / 2).max(1);
        }
        Ok(joint)
    }
}

/// Whether an interface can spell a kingdom row: under an interface without
/// the Gram's input, output cell and fields (v24's), the rows that spell
/// them (phi, h, a thread's cell, the bounds and uniform values of cfg.m,
/// cfg.d, cfg.r_rows, cfg.r_off) have nothing to render. A mutation or a
/// sampler draws only the rows this admits (a point mutation drew
/// load.phi4_safe under v24's role and rendered `vec4<f32>([i], ...)`,
/// 2026-10-17).
pub fn role_spells(kingdom: &WgslKingdom, role: &Role, r: &WgslRow) -> bool {
    if role.per_kernel.iter().any(|p| p.name == "store.h") {
        return true;
    }
    let gram_dual = |t: &Ty| kingdom.dual(*t).is_some_and(|d| matches!(d.dual.as_str(), "u32.boundm" | "u32.boundd" | "u32.boundrows" | "u32.cell" | "u32.mirror"));
    let gram_name = |n: &str| ["load.phi", "load.h", "store.h", "uniform.cfg_", "bound.m", "bound.d", "bound.rows"].iter().any(|p| n.starts_with(p));
    !gram_name(&r.name) && !gram_dual(&r.output) && !r.inputs.iter().any(gram_dual)
}

/// A register kind beyond the first ten (f v i b n u w x y z): a vector of
/// i32, u32, f32 or bool. `kingdoms/safewgsl/gen_tables.py` writes the same
/// list as REG_KINDS (a test checks each entry against the rows).
pub struct ExtraRegister {
    /// The register's name and its rows' suffix: `C0`, `reg.get_C`, `reg.set_C_L0`.
    pub letter: char,
    /// The vector type the register holds.
    pub slot: &'static str,
    /// The register's dual.
    pub dual: &'static str,
    /// How many the kernel may name.
    pub count: usize,
}

pub const EXTRA_REGISTERS: [ExtraRegister; 9] = [
    ExtraRegister { letter: 'C', slot: "vec3<f32>", dual: "reg<vec3<f32>>.real", count: 16 },
    ExtraRegister { letter: 'D', slot: "vec2<i32>", dual: "reg<vec2<i32>>.int", count: 16 },
    ExtraRegister { letter: 'E', slot: "vec3<i32>", dual: "reg<vec3<i32>>.int", count: 16 },
    ExtraRegister { letter: 'F', slot: "vec4<i32>", dual: "reg<vec4<i32>>.int", count: 16 },
    ExtraRegister { letter: 'G', slot: "vec2<u32>", dual: "reg<vec2<u32>>.index", count: 16 },
    ExtraRegister { letter: 'H', slot: "vec3<u32>", dual: "reg<vec3<u32>>.index", count: 16 },
    ExtraRegister { letter: 'J', slot: "vec2<bool>", dual: "reg<vec2<bool>>.flag", count: 16 },
    ExtraRegister { letter: 'K', slot: "vec3<bool>", dual: "reg<vec3<bool>>.flag", count: 16 },
    ExtraRegister { letter: 'L', slot: "vec4<bool>", dual: "reg<vec4<bool>>.flag", count: 16 },
];

/// The extra register kind that holds a value dual (any form of its vector type).
pub fn extra_register_of_value(dual: &str) -> Option<&'static ExtraRegister> {
    let slot = dual.split('.').next()?;
    EXTRA_REGISTERS.iter().find(|r| r.slot == slot)
}

/// The per-lane conversions between floats and unsigned integers that the
/// reader writes for a `vec2<f32>(v)` (convert.vindex_to_f32, vcount_to_f32,
/// vf32_to_count), added 2026-10-19. Conforming a tree never inserts one: a
/// vector conversion where a demand differs by form would move the splat of
/// a scalar's conversion (`splat4(convert(x))`) to the conversion of a splat,
/// and the saved genomes read as they did.
pub fn is_vector_conversion_read_only(r: &WgslRow) -> bool {
    matches!(r.name.as_str(), "convert.vindex_to_f32" | "convert.vcount_to_f32" | "convert.vf32_to_count")
}

/// Whether a row's base name sets or reads an extra register (`reg.set_C`, `reg.get_C`).
pub fn is_extra_register_row(base: &str) -> bool {
    base.strip_prefix("reg.set_").or_else(|| base.strip_prefix("reg.get_")).is_some_and(|k| k.len() == 1 && EXTRA_REGISTERS.iter().any(|r| k.starts_with(r.letter)))
}

/// A register leaf's WGSL variable name (`f3`, `i32r`); a fallback leaf
/// names the first register of ITS DUAL.
pub fn register_name(kingdom: &WgslKingdom, r: &Tree) -> Result<String, String> {
    let d = kingdom.dual(r.ty).ok_or_else(|| format!("{:?} is not a dual of the kingdom", r.ty))?;
    let ok = |n: &str, kind: char, max: usize| n.strip_prefix(kind).and_then(|k| k.parse::<usize>().ok()).is_some_and(|k| k < max);
    Ok(match (r.name.as_str(), d.dual.as_str()) {
        ("fallback", "reg<f32>.real") => "f0".to_string(),
        ("fallback", "reg<vec4<f32>>.real") => "v0".to_string(),
        ("fallback", "reg<u32>.index") => "i0".to_string(),
        ("fallback", "reg<bool>.flag") => "b0".to_string(),
        ("fallback", "reg<i32>.int") => "n0".to_string(),
        ("fallback", "reg<vec4<u32>>.bits") => "u0".to_string(),
        ("fallback", "reg<vec2<f32>>.real") => "w0".to_string(),
        ("fallback", "reg<mat2x2<f32>>.real") => "x0".to_string(),
        ("fallback", "reg<mat3x3<f32>>.real") => "y0".to_string(),
        ("fallback", "reg<mat4x4<f32>>.real") => "z0".to_string(),
        ("fallback", dual) if EXTRA_REGISTERS.iter().any(|x| x.dual == dual) => format!("{}0", EXTRA_REGISTERS.iter().find(|x| x.dual == dual).map_or('C', |x| x.letter)),
        (name, _) => {
            let n = name.strip_prefix("reg.").unwrap_or("");
            if ok(n, 'f', 64) || ok(n, 'v', 32) || ok(n, 'i', 64) || ok(n, 'b', 16) || ok(n, 'n', 16) || ok(n, 'u', 16) || ok(n, 'w', 16) || ok(n, 'x', 4) || ok(n, 'y', 4) || ok(n, 'z', 4) || EXTRA_REGISTERS.iter().any(|x| ok(n, x.letter, x.count)) {
                let k: usize = n[1..].parse().unwrap_or(0);
                register_var(&n[..1], k)
            } else {
                return Err(format!("{name} ({}) is not a register", d.dual));
            }
        }
    })
}


/// A family of arrays of one dual: the terminal rows `wg.<name>` or
/// `parr.<name>` that name them. The first are the names the kingdom grew
/// with (`legacy`, in the order a reader takes them), the rest are generated
/// from `stem` and their number up to `total`. `kingdoms/safewgsl/gen_tables.py`
/// writes the same list (ARRAY_FAMILIES, the parameter of the slot counts); a
/// test checks every name against the rows. Raising `total` here and there is
/// the whole change a kernel with more arrays of a kind needs.
pub struct ArrayFamily {
    /// The array dual the rows output.
    pub dual: &'static str,
    pub legacy: &'static [&'static str],
    pub stem: &'static str,
    pub total: usize,
    /// A private array's element type and length (a workgroup array's are the
    /// plan's: [`wg_elem`], [`wg_default_len`]).
    pub elem: &'static str,
    pub len: u32,
}

impl ArrayFamily {
    /// Every name of the family in the order a reader takes them.
    pub fn names(&self) -> impl Iterator<Item = String> + '_ {
        self.legacy.iter().map(|n| n.to_string()).chain((self.legacy.len()..self.total).map(|i| format!("{}{i}", self.stem)))
    }
}

pub const WORKGROUP_ARRAYS: [ArrayFamily; 7] = [
    ArrayFamily { dual: "wgmem<f32>.real", legacy: &["a", "b", "c", "d", "e", "f", "g", "h"], stem: "f", total: 12, elem: "", len: 0 },
    ArrayFamily { dual: "wgmem<vec4<f32>>.real", legacy: &["va", "vb", "vc", "vd"], stem: "vq", total: 8, elem: "", len: 0 },
    ArrayFamily { dual: "wgmem<vec2<f32>>.real", legacy: &["v2a", "v2b", "v2c", "v2d"], stem: "v2_", total: 8, elem: "", len: 0 },
    ArrayFamily { dual: "wgmem<u32>.index", legacy: &["ua", "ub", "uc", "ud", "ue", "uf", "ug", "uh", "ui", "uj", "uk", "ul"], stem: "u_", total: 16, elem: "", len: 0 },
    ArrayFamily { dual: "wgmem<atomic<u32>>.index", legacy: &["au_a", "au_b", "au_c", "au_d"], stem: "au_", total: 8, elem: "", len: 0 },
    ArrayFamily { dual: "wgmem<atomic<i32>>.int", legacy: &["ai_a", "ai_b"], stem: "ai_", total: 4, elem: "", len: 0 },
    ArrayFamily { dual: "wgmem<i32>.int", legacy: &["na", "nb"], stem: "n_", total: 6, elem: "", len: 0 },
];

pub const PRIVATE_ARRAYS: [ArrayFamily; 8] = [
    ArrayFamily { dual: "parr<f32>.real", legacy: &["a", "b", "c", "d", "e", "f"], stem: "f", total: 8, elem: "f32", len: 64 },
    ArrayFamily { dual: "parr<vec4<f32>>.real", legacy: &["v", "w", "x", "y"], stem: "w", total: 6, elem: "vec4<f32>", len: 16 },
    ArrayFamily { dual: "parr<u32>.index", legacy: &["i", "j", "k"], stem: "i", total: 5, elem: "u32", len: 16 },
    ArrayFamily { dual: "parr<bool>.flag", legacy: &["m", "n"], stem: "m", total: 4, elem: "bool", len: 16 },
    ArrayFamily { dual: "parr<vec2<f32>>.real", legacy: &["q", "r"], stem: "q", total: 4, elem: "vec2<f32>", len: 16 },
    ArrayFamily { dual: "parr64<u32>.index", legacy: &["ua", "uc"], stem: "ua", total: 4, elem: "u32", len: 64 },
    ArrayFamily { dual: "parr64<vec4<f32>>.real", legacy: &["vq", "vr"], stem: "vq", total: 4, elem: "vec4<f32>", len: 64 },
    ArrayFamily { dual: "parr128<u32>.index", legacy: &["ux", "uy"], stem: "ux", total: 4, elem: "u32", len: 128 },
];

/// The rows of a family as `wg.<name>` / `parr.<name>` and their variables
/// (`wg_<name>`, `p<name>`), built once: the names are `'static` for the
/// callers that hold them.
struct ArrayNames {
    /// Row name to variable.
    var: BTreeMap<String, &'static str>,
    /// Array dual to its row names in a reader's order.
    rows: BTreeMap<&'static str, Vec<&'static str>>,
}

fn array_names() -> &'static ArrayNames {
    static NAMES: std::sync::OnceLock<ArrayNames> = std::sync::OnceLock::new();
    NAMES.get_or_init(|| {
        let mut n = ArrayNames { var: BTreeMap::new(), rows: BTreeMap::new() };
        for (class, var, families) in [("wg", "wg_", &WORKGROUP_ARRAYS[..]), ("parr", "p", &PRIVATE_ARRAYS[..])] {
            for f in families {
                for name in f.names() {
                    let row: &'static str = Box::leak(format!("{class}.{name}").into_boxed_str());
                    let v: &'static str = Box::leak(format!("{var}{name}").into_boxed_str());
                    n.var.insert(row.to_string(), v);
                    n.rows.entry(f.dual).or_default().push(row);
                }
            }
        }
        n
    })
}

/// The rows of the arrays of one dual, in the order a reader takes them
/// (`wgmem<u32>.index`: `wg.ua`, `wg.ub`, ..).
pub fn array_rows(dual: &str) -> &'static [&'static str] {
    array_names().rows.get(dual).map_or(&[], |v| v.as_slice())
}

/// The dual of the arrays a row (`wg.ua`, `parr.ux`) names.
pub fn array_dual_of_row(row: &str) -> Option<&'static str> {
    array_names().rows.iter().find(|(_, rows)| rows.contains(&row)).map(|(d, _)| *d)
}

/// A workgroup array leaf's variable name; a fallback leaf names the
/// array of ITS DUAL (the vec4 array for a vec4 demand), never `wg_a`
/// regardless — a vec4 value stored into the f32 array was a compile
/// error the kingdom's types had not made (fit_safe18_v2, 2026-10-10).
pub fn wgmem_var(kingdom: &WgslKingdom, w: &Tree) -> Result<&'static str, String> {
    if w.name == "fallback" {
        let d = kingdom.dual(w.ty).map(|d| d.dual.clone()).unwrap_or_default();
        return Ok(array_names().rows.get(d.as_str()).and_then(|r| r.first()).and_then(|r| array_names().var.get(*r)).copied().unwrap_or("wg_a"));
    }
    array_names().var.get(w.name.as_str()).copied().filter(|_| w.name.starts_with("wg.")).ok_or_else(|| format!("{} is not a workgroup array", w.name))
}

/// The kernel's workgroup arrays (2026-10-12): every array it declares on
/// its grid (`grid.with_wg`) at its length, every array it uses but does
/// not declare at its default length (f32 2,048, vec2 1,024, vec4 512),
/// then the largest halved until the total fits the device's 32 KB. Empty
/// (the original four declarations) for a kernel that declares none and
/// uses only `wg_a`, `wg_b`, `wg_va`, `wg_vb`, whose text is unchanged.
pub fn workgroup_plan(kingdom: &WgslKingdom, grid: &Tree, root: &Tree) -> Result<BTreeMap<String, (String, u32)>, String> {
    workgroup_plan_of(kingdom, grid, root, None)
}

/// [`workgroup_plan`] over the arrays `only` names when given (the arrays a
/// first rendering's text uses: a statement after a return is not written).
pub fn workgroup_plan_of(kingdom: &WgslKingdom, grid: &Tree, root: &Tree, only: Option<&BTreeSet<String>>) -> Result<BTreeMap<String, (String, u32)>, String> {
    let mut plan = workgroup_plan_requested_of(kingdom, grid, root, only)?;
    while plan_bytes(&plan) > 32768 {
        let largest = plan.iter().max_by_key(|(v, (e, l))| (elem_bytes(e) * u64::from(*l), std::cmp::Reverse((*v).clone()))).map(|(v, _)| v.clone()).expect("a non-empty plan");
        let entry = plan.get_mut(&largest).expect("present");
        entry.1 = (entry.1 / 2).max(1);
    }
    Ok(plan)
}

/// The bytes of workgroup memory a kernel requests (knob references resolved;
/// the interface plays no part: the arrays are the kernel's own). A module:
/// its most demanding entry point (each runs as its own pipeline).
pub fn workgroup_request(kingdom: &WgslKingdom, root: &Tree) -> Result<u64, String> {
    if root.name.starts_with("kernel.module_") {
        return root.kids.iter().map(|k| workgroup_request(kingdom, k)).try_fold(0, |m, r| r.map(|r| m.max(r)));
    }
    let root = crate::wgsl::params::resolve(root)?;
    let Some(grid) = root.kids.first().filter(|_| root.name == "kernel.entry_safe") else { return Ok(0) };
    Ok(plan_bytes(&workgroup_plan_requested(kingdom, grid, &root)?))
}

/// [`workgroup_plan`] before any array is halved.
pub fn workgroup_plan_requested(kingdom: &WgslKingdom, grid: &Tree, root: &Tree) -> Result<BTreeMap<String, (String, u32)>, String> {
    workgroup_plan_requested_of(kingdom, grid, root, None)
}

/// [`workgroup_plan_requested`] over the arrays `only` names when given.
fn workgroup_plan_requested_of(kingdom: &WgslKingdom, grid: &Tree, root: &Tree, only: Option<&BTreeSet<String>>) -> Result<BTreeMap<String, (String, u32)>, String> {
    let elem_of = wg_elem;
    let default_len = wg_default_len;
    let mut declared: Vec<(String, u32)> = Vec::new();
    if grid.name == "grid.with_wg" && grid.kids.len() == 2 {
        let mut at = Some(&grid.kids[1]);
        while let Some(d) = at {
            let (decl, rest) = match (d.name.as_str(), d.kids.as_slice()) {
                ("wg.decls_one", [x]) => (Some(x), None),
                ("wg.decls_cons", [x, r]) => (Some(x), Some(r)),
                _ => (None, None),
            };
            if let Some(x) = decl.filter(|x| x.kids.len() == 2 && x.name.starts_with("wg.decl_")) {
                let var = wgmem_var(kingdom, &x.kids[0])?.to_string();
                let len = match x.kids[1].name.as_str() {
                    n if n.starts_with("literal.") && x.kids[1].konst.is_finite() => (x.kids[1].konst.round() as i64).clamp(1, 8192) as u32,
                    _ => default_len(&var),
                };
                if !declared.iter().any(|(v, _)| *v == var) {
                    declared.push((var, len));
                }
            }
            at = rest;
        }
    }
    let mut used: BTreeSet<String> = BTreeSet::new();
    fn uses(kingdom: &WgslKingdom, t: &Tree, out: &mut BTreeSet<String>) {
        let array = t.kids.is_empty() && (t.name.starts_with("wg.") || (t.name == "fallback" && kingdom.dual(t.ty).is_some_and(|d| d.dual.starts_with("wgmem<"))));
        if array {
            if let Ok(v) = wgmem_var(kingdom, t) {
                out.insert(v.to_string());
            }
        }
        t.kids.iter().for_each(|k| uses(kingdom, k, out));
    }
    // The statements' uses, not the grid's own declarations.
    if root.name == "kernel.entry_safe" {
        root.kids.iter().skip(1).for_each(|k| uses(kingdom, k, &mut used));
    } else {
        uses(kingdom, root, &mut used);
    }
    if let Some(only) = only {
        used.retain(|v| only.contains(v));
    }
    let legacy = ["wg_a", "wg_b", "wg_va", "wg_vb"];
    if declared.is_empty() && used.iter().all(|v| legacy.contains(&v.as_str())) {
        return Ok(BTreeMap::new());
    }
    // A declared array no statement uses is not written (2026-10-12): the
    // reader reads the arrays a kernel uses, so an unused declaration did not
    // read back (163 of 1,000 random trees).
    let mut plan: BTreeMap<String, (String, u32)> = declared.into_iter().filter(|(v, _)| used.contains(v)).map(|(v, l)| { let e = elem_of(&v).to_string(); (v, (e, l)) }).collect();
    for v in used {
        plan.entry(v.clone()).or_insert_with(|| (elem_of(&v).to_string(), default_len(&v)));
    }
    Ok(plan)
}

impl Renderer<'_> {
    /// A workgroup array's length in this kernel (its plan's, or the
    /// original four's).
    fn wg_len(&self, var: &str, scope: &Scope) -> u32 {
        scope.wg.get(var).map_or_else(|| array_len(var), |(_, l)| *l)
    }

    /// A put's element of an array of `len`: a literal k taken modulo the
    /// elements each thread owns there, ceil(len / threads), so the put is
    /// never one no thread executes (seed 18, 2026-10-14: lidx + 2160 into
    /// 1,024 elements, a dead store layer 7 could not place); a tile
    /// counter's k as min(k, ceil(len / threads)): at that k and past it the
    /// put's guard `i < len` fails either way, and the product cannot wrap
    /// (2026-10-17: a put in a loop bounded near 2^24 wrapped lidx + k * 780
    /// back under the length, two writes of one element, layer 7 found it).
    fn wg_own(&self, i: &Tree, len: u32, scope: &mut Scope) -> Result<String, String> {
        let threads = scope.grid.x * scope.grid.y;
        match (safe_base(&i.name), i.kids.first()) {
            ("safe.wg_own", Some(t)) if t.name.starts_with("literal.") && t.konst.is_finite() => {
                let k = t.konst.round().clamp(0.0, 63.0) as u32 % len.div_ceil(threads.max(1)).max(1);
                Ok(format!("(lidx + {}u)", k * threads))
            }
            ("safe.wg_own_tile" | "safe.wg_own_tile2", Some(k)) => Ok(format!("(lidx + min({}, {}u) * {threads}u)", self.value(k, scope)?, len.div_ceil(threads.max(1)))),
            _ => self.value(i, scope),
        }
    }

    fn fill(&self, p: &PerKernel, t: &Tree, scope: &mut Scope) -> Result<String, String> {
        if t.kids.len() != p.inputs.len() {
            return Err(format!("{} with {} children, {} expected", p.name, t.kids.len(), p.inputs.len()));
        }
        let mut out = p.template.clone();
        for (i, k) in t.kids.iter().enumerate() {
            out = out.replace(&format!("{{{i}}}"), &format!("({})", self.value(k, scope)?));
        }
        Ok(out)
    }

    /// A value whose literals are written through lit_*, so naga folds none
    /// of them: a cell's offset clamped into its block (`min(off, B - 1)`
    /// over a literal offset was folded, and an offset of 0 then read back
    /// as no offset; 2026-10-13, the commonest text that was not a fixpoint).
    fn unfolded(&self, v: &Tree, scope: &mut Scope) -> Result<String, String> {
        let was = scope.const_depth;
        scope.const_depth = 1;
        let out = self.value_inner(v, scope);
        scope.const_depth = was;
        out
    }

    /// A value expression, parenthesised where an operator could bind.
    fn value(&self, v: &Tree, scope: &mut Scope) -> Result<String, String> {
        // Any expression of literals only writes them through lit_*, so naga
        // folds none of it (2026-10-13, AST to AST: a loop bound, a
        // comparison, a select or an access of literals was folded to a
        // literal, and read back as another row; 135 of 1,000 random trees).
        // The exceptions: a constructor of literals (naga folds it to a
        // constant the reader reads back as the same splat or compose), and
        // a row written as its one child (`uniform.of_lit` is its literal).
        let constructor = v.name.starts_with("shape.splat") || v.name.starts_with("shape.compose");
        let as_child = matches!(v.name.as_str(), "uniform.of_lit" | "index.of_uniform" | "uniform.of_checked" | "uniform.of_counter" | "uniform.of_tcounter" | "uniform.of_tcounter2" | "uniform.of_tcounter3" | "uniform.of_tcounter4");
        if scope.const_depth == 0 && !v.kids.is_empty() && !constructor && !as_child && literal_only(v) {
            scope.const_depth = 1;
            let out = self.value_inner(v, scope);
            scope.const_depth = 0;
            return out;
        }
        self.value_inner(v, scope)
    }

    fn value_inner(&self, v: &Tree, scope: &mut Scope) -> Result<String, String> {
        let d = self.dual(v.ty)?;
        let name = v.name.as_str();
        // The leaves of a literal-only operation pass through `lit_*`.
        let wrap = scope.const_depth > 0;
        // A fallback where a thread's row, column or cell is demanded is the
        // thread's own (its block's origin): a constant would make every
        // thread store to one cell, a data race (the legality checker: 415
        // of 1,000 random genes, 2026-10-12).
        if name == "fallback" {
            if let Some(own) = self.own_fallback(&d.dual, scope)? {
                return Ok(own);
            }
        }
        if name == "fallback" {
            return self.fallback_value(d, wrap);
        }
        if let Some(form) = name.strip_prefix("literal.") {
            // A knob reference (resolved before rendering) is written as a
            // literal of its dual's form.
            let form = if form.starts_with("p_") { d.form.as_str() } else { form };
            return self.literal(form, v.konst, d, wrap);
        }
        if let Some(p) = self.per_kernel.get(name) {
            return self.fill(p, v, scope);
        }
        // The uniform values: as the index rows and loads they compute alike.
        if let Some(ix) = uniform_as_index(name) {
            let index = dual_ty(self.kingdom, if ix.starts_with("load.cfg_r_off") { "u32.index" } else if ix.starts_with("load.") { "u32.count" } else { "u32.index" })?;
            return self.value(&Tree { name: ix.into(), ty: index, konst: v.konst, kids: v.kids.clone() }, scope);
        }
        match name {
            "uniform.of_lit" | "index.of_uniform" | "uniform.of_checked" | "uniform.of_counter" | "uniform.of_tcounter" | "uniform.of_tcounter2" | "uniform.of_tcounter3" | "uniform.of_tcounter4" => return self.value(&v.kids[0], scope),
            "uniform.wid_x" | "uniform.wid_y" | "uniform.wid_z" => return Ok(format!("wid.{}", &name[name.len() - 1..])),
            // A workgroup-uniform predicate: its comparison or logic.
            "uflag.eq" | "uflag.ne" | "uflag.lt" | "uflag.le" | "uflag.gt" | "uflag.ge" | "uflag.and" | "uflag.or" if v.kids.len() == 2 => {
                let op = match &name[6..] {
                    "eq" => "==",
                    "ne" => "!=",
                    "lt" => "<",
                    "le" => "<=",
                    "gt" => ">",
                    "ge" => ">=",
                    "and" => "&",
                    _ => "|",
                };
                return Ok(format!("(({}) {op} ({}))", self.value(&v.kids[0], scope)?, self.value(&v.kids[1], scope)?));
            }
            "uflag.not" if v.kids.len() == 1 => return Ok(format!("(!({}))", self.value(&v.kids[0], scope)?)),
            "uflag.of_flag" if v.kids.len() == 1 => return self.value(&v.kids[0], scope),
            "uniform.nwg_x" | "uniform.nwg_y" | "uniform.nwg_z" => return Ok(format!("nwg.{}", &name[name.len() - 1..])),
            "uniform.subgroup_size" => return Ok("sg_size".into()),
            "uniform.num_subgroups" => return Ok("sg_count".into()),
            _ => {}
        }
        match name {
            "reg.get_f" | "reg.get_v" | "reg.get_i" | "reg.get_b" | "reg.get_n" | "reg.get_u" | "reg.get_w" | "reg.get_x" | "reg.get_y" | "reg.get_z" => return self.register(&v.kids[0]),
            base if is_extra_register_row(base) => return self.register(&v.kids[0]),
            "reg.get_v_lane" => return Ok(format!("{}[({}) & 3u]", self.register(&v.kids[0])?, self.value(&v.kids[1], scope)?)),
            // A lane of a vec4 element of a workgroup or private array (seed08 of run A): the
            // element is a reference, so the lane index may be a run-time value.
            "wg.get_v_lane" => {
                let a = self.wgmem(&v.kids[0])?;
                return Ok(format!("{a}[{}][({}) & 3u]", self.index_text(&v.kids[1], self.wg_len(a, scope), scope)?, self.value(&v.kids[2], scope)?));
            }
            "parr.get_v_lane" => {
                let a = self.parr_name(&v.kids[0])?;
                return Ok(format!("{a}[{}][({}) & 3u]", self.index_text(&v.kids[1], array_len(a), scope)?, self.value(&v.kids[2], scope)?));
            }
            // Masked into the array's declared length (its knob), not 2,048:
            // an array of 256 indexed `& 2047u` read past its end.
            "wg.get" => {
                let a = self.wgmem(&v.kids[0])?;
                let len = self.wg_len(a, scope);
                return Ok(format!("{a}[{}]", masked_to(&self.value(&v.kids[1], scope)?, len)));
            }
            // The safe read: the index is in range by its type (safe.wg_of masks it).
            // The subgroup operations.
            n if n.starts_with("subgroup.") => {
                let f = match n.trim_start_matches("subgroup.") {
                    "add" => "subgroupAdd",
                    "mul" => "subgroupMul",
                    "min" => "subgroupMin",
                    "max" => "subgroupMax",
                    "and" => "subgroupAnd",
                    "or" => "subgroupOr",
                    "xor" => "subgroupXor",
                    "all" => "subgroupAll",
                    "any" => "subgroupAny",
                    "inclusive_add" => "subgroupInclusiveAdd",
                    "inclusive_mul" => "subgroupInclusiveMul",
                    "exclusive_add" => "subgroupExclusiveAdd",
                    "exclusive_mul" => "subgroupExclusiveMul",
                    "broadcast_first" => "subgroupBroadcastFirst",
                    "broadcast" => "subgroupBroadcast",
                    "shuffle" => "subgroupShuffle",
                    "shuffle_down" => "subgroupShuffleDown",
                    "shuffle_up" => "subgroupShuffleUp",
                    "shuffle_xor" => "subgroupShuffleXor",
                    "ballot" => "subgroupBallot",
                    other => return Err(format!("subgroup.{other} is not a subgroup built-in")),
                };
                let mut args: Vec<String> = v.kids.iter().map(|k| self.value(k, scope)).collect::<Result<_, _>>()?;
                // A broadcast's lane is a const-expression for naga 30's
                // validator: the literal itself, never through `lit_u32`
                // (2026-10-19, the naga/wgpu 30 upgrade).
                if f == "subgroupBroadcast" && v.kids.len() == 2 {
                    args[1] = format!("{}u", self.lit(&v.kids[1], 64)?);
                }
                return Ok(format!("{f}({})", args.join(", ")));
            }
            // The atomic built-ins on a workgroup array (their old value).
            n if n.starts_with("atomic.") => {
                let a = self.wgmem(&v.kids[0])?;
                let len = self.wg_len(a, scope);
                let i = self.index_text(&v.kids[1], len, scope)?;
                let op = n.trim_start_matches("atomic.").trim_end_matches("_u").trim_end_matches("_i");
                return Ok(match (op, v.kids.len()) {
                    ("load", 2) => format!("atomicLoad(&{a}[{i}])"),
                    (op, 3) => {
                        let f = match op {
                            "add" => "atomicAdd",
                            "sub" => "atomicSub",
                            "max" => "atomicMax",
                            "min" => "atomicMin",
                            "and" => "atomicAnd",
                            "or" => "atomicOr",
                            "xor" => "atomicXor",
                            "exchange" => "atomicExchange",
                            other => return Err(format!("atomic.{other} is not an atomic built-in")),
                        };
                        format!("{f}(&{a}[{i}], {})", self.value(&v.kids[2], scope)?)
                    }
                    (other, k) => return Err(format!("atomic.{other} with {k} children")),
                });
            }
            "wg.get_safe" | "wg.get_v_safe" | "wg.get_v2_safe" | "wg.get_u_safe" | "wg.get_n_safe" => {
                let a = self.wgmem(&v.kids[0])?;
                let len = self.wg_len(a, scope);
                return Ok(format!("{a}[{}]", self.index_text(&v.kids[1], len, scope)?));
            }
            "parr.get_f" | "parr.get_v" | "parr.get_i" | "parr.get_b" | "parr.get_t" => {
                let a = self.parr_name(&v.kids[0])?;
                return Ok(format!("{a}[{}]", self.index_text(&v.kids[1], array_len(a), scope)?));
            }
            _ => {}
        }
        let kids: Result<Vec<String>, String> = v.kids.iter().map(|k| self.value(k, scope)).collect();
        let a = kids?;
        match safe_base(name) {
            // ---- safeWGSL values ----
            // Any expression bound carries its cap: a loop bounded by a register,
            // an atomic or a load cannot run past the largest m the fitness runs
            // (the legality checker's bounded loops, 177 of 1,000 random genes).
            "bound.of_index" => return Ok(format!("min(({}), {MAX_LOOP_TRIP}u)", self.value(&v.kids[0], scope)?)),
            // Every loop bound a gene computes is capped (2026-10-12, Codex's
            // review): a uniform one too, so no loop runs past MAX_LOOP_TRIP.
            "bound.of_uniform" | "bound.x_of_uniform" => return Ok(format!("min(({}), {MAX_LOOP_TRIP}u)", self.value(&v.kids[0], scope)?)),
            "bound.m" => return Ok(self.role.field("m")),
            "bound.d" => return Ok(self.role.field("d")),
            "bound.rows" => return Ok(self.role.field("r_rows")),
            // A loop's counter is its loop's: outside the loop it is 0u, as
            // the rows say (written `t` it read the last loop's exit value,
            // 852 of 1,000 random genes, 2026-10-12).
            "safe.counter" => return Ok(if scope.in_items { "t".into() } else { OUT_OF_LOOP.into() }),
            "safe.tile_counter" | "safe.tile_counter2" | "safe.tile_counter3" | "safe.tile_counter4" => {
                let (var, depth) = match name {
                    "safe.tile_counter2" => ("k2", 2),
                    "safe.tile_counter3" => ("k3", 3),
                    "safe.tile_counter4" => ("k4", 4),
                    _ => ("k", 1),
                };
                if scope.tile_depth < depth {
                    return Ok(OUT_OF_LOOP.into());
                }
                // In a copy of a fully unrolled body the counter is that copy's
                // literal, through lit_u32 so naga folds no constant expression.
                return Ok(scope.unrolled.get(var).map_or_else(|| var.to_string(), |i| format!("lit_u32({i}u)")));
            }
            "safe.counter_index" | "safe.tile_index" | "safe.tile_index2" | "safe.tile_index3" | "safe.tile_index4" => return self.value(&v.kids[0], scope),
            "safe.row_plus_tile" | "safe.row_plus_tile2" | "safe.row_plus_tile3" | "safe.row_plus_tile4" | "safe.col_plus_tile" | "safe.col_plus_tile2" | "safe.col_plus_tile3" | "safe.col_plus_tile4" => {
                let blk = if safe_base(name).starts_with("safe.row") { scope.grid.by } else { scope.grid.bx };
                return Ok(format!("({} + min({}, {}u))", self.value(&v.kids[0], scope)?, self.unfolded(&v.kids[1], scope)?, blk.saturating_sub(1)));
            }
            "safe.row_plus_tile_s" | "safe.row_plus_tile2_s" | "safe.row_plus_tile3_s" | "safe.row_plus_tile4_s" | "safe.col_plus_tile_s" | "safe.col_plus_tile2_s" | "safe.col_plus_tile3_s" | "safe.col_plus_tile4_s" => {
                // The k-th row of a strided block: rows W apart, k snapped below the block factor.
                let (blk, w) = if safe_base(name).starts_with("safe.row") { (scope.grid.by, scope.grid.row_axis().1) } else { (scope.grid.bx, scope.grid.col_axis().1) };
                return Ok(format!("({} + min({}, {}u) * {w}u)", self.value(&v.kids[0], scope)?, self.unfolded(&v.kids[1], scope)?, blk.saturating_sub(1)));
            }
            "safe.row_plus_x" | "safe.col_plus_x" => {
                // Any offset expression, clamped into the block.
                let blk = if safe_base(name) == "safe.row_plus_x" { scope.grid.by } else { scope.grid.bx };
                return Ok(format!("({} + min({}, {}u))", self.value(&v.kids[0], scope)?, self.unfolded(&v.kids[1], scope)?, blk.saturating_sub(1)));
            }
            "safe.row_plus_x_s" | "safe.col_plus_x_s" => {
                let (blk, w) = if safe_base(name) == "safe.row_plus_x_s" { (scope.grid.by, scope.grid.row_axis().1) } else { (scope.grid.bx, scope.grid.col_axis().1) };
                return Ok(format!("({} + min({}, {}u) * {w}u)", self.value(&v.kids[0], scope)?, self.unfolded(&v.kids[1], scope)?, blk.saturating_sub(1)));
            }
            "safe.thread_row_g" | "safe.thread_col_g" => {
                // The first cell of a thread's block of G groups: wid * (W * B) + lid * (B / G).
                let ((axis, w), blk) = if safe_base(name) == "safe.thread_row_g" { (scope.grid.row_axis(), scope.grid.by) } else { (scope.grid.col_axis(), scope.grid.bx) };
                let g = self.lit(&v.kids[3], 64)?.clamp(1, blk.max(1));
                return Ok(format!("(wid.{axis} * {}u + lid.{axis} * {}u)", w * blk, blk / g));
            }
            "safe.row_plus_g" | "safe.col_plus_g" => {
                // A cell of the grouped block by group and lane, both clamped.
                let (blk, w) = if safe_base(name) == "safe.row_plus_g" { (scope.grid.by, scope.grid.row_axis().1) } else { (scope.grid.bx, scope.grid.col_axis().1) };
                // The groups are the ORIGIN's (2026-10-12): an offset over
                // another count of groups than its origin's left the thread's
                // block (lid scaled by B / G for one G, offsets up to B / G' - 1
                // for another: two threads' cells). A fallback origin is the
                // strided one, B groups of one cell.
                let origin = &v.kids[0];
                let g = if matches!(safe_base(&origin.name), "safe.thread_row_g" | "safe.thread_col_g") && origin.kids.len() == 4 { self.lit(&origin.kids[3], 64)? } else { blk.max(1) };
                let g = g.clamp(1, blk.max(1));
                let s = blk / g;
                return Ok(format!("({} + min({}, {}u) * {}u + min({}, {}u))", self.value(&v.kids[0], scope)?, self.unfolded(&v.kids[2], scope)?, g - 1, w * s, self.unfolded(&v.kids[3], scope)?, s.saturating_sub(1)));
            }
            "safe.thread_row" => return Ok(format!("(gid.{} * {}u)", scope.grid.row_axis().0, scope.grid.by)),
            "safe.thread_col" => return Ok(format!("(gid.{} * {}u)", scope.grid.col_axis().0, scope.grid.bx)),
            "safe.thread_row_s" | "safe.thread_col_s" => {
                let ((a, w), blk) = if safe_base(name) == "safe.thread_row_s" { (scope.grid.row_axis(), scope.grid.by) } else { (scope.grid.col_axis(), scope.grid.bx) };
                return Ok(format!("(wid.{a} * {}u + lid.{a})", w * blk));
            }
            "safe.row_plus_s" | "safe.col_plus_s" => {
                let (blk, w) = if safe_base(name) == "safe.row_plus_s" { (scope.grid.by, scope.grid.row_axis().1) } else { (scope.grid.bx, scope.grid.col_axis().1) };
                let off = self.lit(&v.kids[1], 64)?.min(blk.saturating_sub(1));
                return Ok(format!("({} + {}u)", self.value(&v.kids[0], scope)?, off * w));
            }
            "safe.row_plus" | "safe.col_plus" => {
                // An offset within the thread's block, snapped below the block factor.
                let blk = if safe_base(name) == "safe.row_plus" { scope.grid.by } else { scope.grid.bx };
                let off = self.lit(&v.kids[1], 64)?.min(blk.saturating_sub(1));
                return Ok(format!("({} + {off}u)", self.value(&v.kids[0], scope)?));
            }
            "safe.cell_of" => {
                let r = self.value(&v.kids[0], scope)?;
                let c = self.value(&v.kids[1], scope)?;
                return Ok(format!("(({r}) * {} + ({c}))", self.role.field("d")));
            }
            // The mirror of the thread's cell: column * d + row (the symmetric half).
            "safe.mirror_of" => {
                let r = self.value(&v.kids[0], scope)?;
                let c = self.value(&v.kids[1], scope)?;
                return Ok(format!("(({c}) * {} + ({r}))", self.role.field("d")));
            }
            "safe.phi_of" => {
                let t = self.value(&v.kids[0], scope)?;
                let c = self.value(&v.kids[1], scope)?;
                return Ok(format!("(({t}) * {} + ({c}))", self.role.field("d")));
            }
            "safe.wg_of" => return Ok(format!("(({}) & {}u)", self.value(&v.kids[0], scope)?, WGMEM_LEN - 1)),
            // This thread's k-th element of a workgroup array: lidx + k * threads
            // (a put writes k modulo the array's elements a thread owns: `wg_own`).
            "safe.wg_own" => {
                let k = v.kids.first().filter(|t| t.name.starts_with("literal.") && t.konst.is_finite()).map_or(0, |t| t.konst.round().clamp(0.0, 63.0) as u32);
                return Ok(format!("(lidx + {}u)", k * scope.grid.x * scope.grid.y));
            }
            "safe.wg_own_tile" | "safe.wg_own_tile2" => return Ok(format!("(lidx + {} * {}u)", self.value(&v.kids[0], scope)?, scope.grid.x * scope.grid.y)),
            "safe.wgv_of" => return Ok(format!("(({}) & {}u)", self.value(&v.kids[0], scope)?, WGVEC_LEN - 1)),
            "safe.wgv2_of" => return Ok(format!("(({}) & 1023u)", self.value(&v.kids[0], scope)?)),
            "safe.pidx64" => return Ok(format!("(({}) & 63u)", self.value(&v.kids[0], scope)?)),
            "safe.pidx128" => return Ok(format!("(({}) & 127u)", self.value(&v.kids[0], scope)?)),
            "safe.pidx16" => return Ok(format!("(({}) & 15u)", self.value(&v.kids[0], scope)?)),
            "safe.row_as_index" | "safe.col_as_index" => return self.value(&v.kids[0], scope),
            // (The Gram's input and output: an interface naming neither has no
            // text for them, a bug in our software where one is asked for.)
            "load.phi_safe" | "load.phi4_safe" | "load.phi_len" if self.role.input.is_empty() => return Err(format!("our software has a bug: {name} under an interface with no input (render::role_spells withholds it)")),
            "load.h_safe" | "load.h_len" if self.role.output.is_empty() => return Err(format!("our software has a bug: {name} under an interface with no output cell (render::role_spells withholds it)")),
            "load.phi_safe" => return Ok(format!("{}[{}]", self.role.input, self.value(&v.kids[0], scope)?)),
            "load.phi4_safe" => {
                let i = self.value(&v.kids[0], scope)?;
                let p = &self.role.input;
                return Ok(format!("vec4<f32>({p}[{i}], {p}[{i} + 1u], {p}[{i} + 2u], {p}[{i} + 3u])"));
            }
            "load.h_safe" => return Ok(format!("{}[{}]", self.role.output, self.value(&v.kids[0], scope)?)),
            "load.phi_len" => return Ok(format!("arrayLength(&{})", self.role.input)),
            "load.h_len" => return Ok(format!("arrayLength(&{})", self.role.output)),
            // A matrix's column chosen at run time, clamped to its columns,
            // through a helper's local copy (naga 0.20 indexes a matrix value
            // by a constant only).
            "shape.col_at" => {
                let (cols, rows) = self.dual(v.kids[0].ty)?.matrix_shape().ok_or_else(|| format!("shape.col_at of {}, not a matrix", v.kids[0].name))?;
                let helper = format!("col_mat{cols}x{rows}f");
                scope.helpers.push(helper.clone());
                return Ok(format!("{helper}({}, {})", self.value(&v.kids[0], scope)?, self.value(&v.kids[1], scope)?));
            }
            _ => {}
        }
        if name.starts_with("safe.lt_") {
            let a: Vec<String> = v.kids.iter().map(|k| self.value(k, scope)).collect::<Result<_, _>>()?;
            return Ok(format!("({} < {})", a[0], a[1]));
        }
        // A thread id's component by axis (`axis.y_lid` is `lid.y`), and an
        // axis index used as a plain index (its own text).
        // The axis is the output's (y rows, x columns); the component is the
        // grid's layout (rows on x under grid.of_rows_x).
        if let Some((component, id)) = name.strip_prefix("axis.").and_then(|r| r.split_once('_')) {
            if v.kids.len() == 1 {
                let component = match component {
                    "y" => scope.grid.row_axis().0,
                    "x" => scope.grid.col_axis().0,
                    other => other,
                };
                return Ok(format!("{id}.{component}"));
            }
        }
        if matches!(name, "index.of_row" | "index.of_col" | "index.of_lidx" | "index.of_gid" | "index.of_lid" | "index.of_wid") && v.kids.len() == 1 {
            return self.value(&v.kids[0], scope);
        }
        match name {
            "builtin.global_invocation_id" => return Ok("gid".into()),
            "builtin.local_invocation_id" => return Ok("lid".into()),
            "builtin.workgroup_id" => return Ok("wid".into()),
            "builtin.local_invocation_index" => return Ok("lidx".into()),
            "builtin.num_workgroups" => return Ok("nwg".into()),
            // The subgroup values, declared on main only when a kernel reads them.
            "builtin.subgroup_invocation_id" => return Ok("sg_lane".into()),
            "builtin.subgroup_size" => return Ok("sg_size".into()),
            "builtin.num_subgroups" => return Ok("sg_count".into()),
            "builtin.subgroup_id" => return Ok("sg_index".into()),
            "builtin.item" => return Ok(if scope.in_items { "t".into() } else { OUT_OF_LOOP.into() }),
            "builtin.tile" if !scope.in_tile => return Ok(OUT_OF_LOOP.into()),
            "builtin.tile" => return Ok(scope.unrolled.get("k").map_or_else(|| "k".to_string(), |i| format!("lit_u32({i}u)"))),
            _ => {}
        }
        if name == "wg.a" || name == "wg.b" {
            return Err(format!("{name} is a workgroup array, not a value (a row took wgmem as a value input)"));
        }
        // `m & ~(k - 1)` as a rounding down to a multiple of a literal k: no
        // zero guard, so the text stays plain (`cfg.m - cfg.m % 4u`).
        if name == "index.round_down" && a.len() == 2 {
            if let Some(k) = v.kids.get(1).filter(|k| k.name.starts_with("literal.") && k.konst >= 1.0) {
                return Ok(format!("(({0}) - (({0}) % {1}u))", a[0], k.konst as u32));
            }
        }
        // No zero divisor (2026-10-12, Codex's review): a divisor that is not
        // a literal of at least 1 is written max(b, 1u). WGSL's x / 0u is x,
        // and x / max(0u, 1u) is x too, so the value is the same everywhere.
        // (The same for a u32 remainder, 2026-10-13: naga writes `%` to Metal
        // raw, and Metal leaves x % 0 undefined; a random kernel's left the
        // GPU running. WGSL's x % 0u is 0, and x % max(0u, 1u) is x % 1u, 0.)
        // (A literal seen through the rows written as their one child, a
        // uniform's `uniform.of_lit` and `index.of_uniform` (no other row: a
        // conversion changes the value, `u32(max(-3, 0))` is 0), and the zero
        // a fallback leaf writes:
        // `wid.y / max(1u, 1u)` read back from naga's folded `max(0u, 1u)`
        // was not a fixpoint, 5 of 1,000 random modules, 2026-10-17.)
        fn literal_through(t: &Tree) -> Option<f64> {
            match (t.name.as_str(), t.kids.as_slice()) {
                (n, []) if n.starts_with("literal.") => Some(t.konst),
                ("fallback", []) => Some(0.0),
                ("uniform.of_lit" | "index.of_uniform", [k]) => literal_through(k),
                _ => None,
            }
        }
        if (name == "index.div" || name == "index.mod") && a.len() == 2 && !v.kids.get(1).and_then(literal_through).is_some_and(|k| k >= 1.0) {
            let one = if d.lanes > 1 { format!("vec{}<u32>(1u)", d.lanes) } else { "1u".to_string() };
            // (A divisor of literal 0 through lit_*: naga folds max(0u, 1u)
            // to 1u, read back as a division by 1, 2026-10-13. Any other
            // literal folds to itself, read back the same.)
            // (A zero under any one-child rows, or a fallback's zero, written
            // through lit_* so naga folds none of it.)
            fn zero(t: &Tree) -> bool {
                (t.kids.is_empty() && ((t.name.starts_with("literal.") && t.konst == 0.0) || t.name == "fallback")) || (t.kids.len() == 1 && !t.name.starts_with("literal.") && zero(&t.kids[0]))
            }
            let b = if zero(&v.kids[1]) { self.unfolded(&v.kids[1], scope)? } else { a[1].clone() };
            let op = if name == "index.div" { "/" } else { "%" };
            return Ok(format!("({} {op} max({b}, {one}))", a[0]));
        }
        // An i32 division or remainder whose divisor is not a literal other
        // than 0 and -1: a kingdom helper with WGSL's value where Metal has
        // none (x / 0 is x, x % 0 is 0, i32::MIN / -1 is i32::MIN, i32::MIN
        // % -1 is 0), each operand evaluated once.
        if (name == "int.div" || name == "int.mod") && a.len() == 2 && !v.kids.get(1).is_some_and(|k| k.name.starts_with("literal.") && k.konst != 0.0 && k.konst != -1.0) {
            let helper = format!("{}{}", if name == "int.div" { "int_div" } else { "int_mod" }, if d.lanes > 1 { d.lanes.to_string() } else { String::new() });
            scope.helpers.push(helper.clone());
            return Ok(format!("{helper}({}, {})", a[0], a[1]));
        }
        // A run-time vector component: a helper copies the value into a `var`
        // and indexes it, which naga allows; the index is masked to the lanes.
        if name == "shape.at" && a.len() == 2 {
            let vd = self.dual(v.kids[0].ty)?;
            let letter = match vd.scalar() {
                Some("f32") => "f",
                Some("u32") => "u",
                Some("i32") => "i",
                Some("bool") => "b",
                other => return Err(format!("shape.at over {}: no helper for scalar {other:?}", vd.dual)),
            };
            if !(2..=4).contains(&vd.lanes) {
                return Err(format!("shape.at over {}: not a vector", vd.dual));
            }
            let helper = format!("at_vec{}{letter}", vd.lanes);
            scope.helpers.push(helper.clone());
            return Ok(format!("{helper}({}, {})", a[0], a[1]));
        }
        // A per-kernel swizzle (`shape.swizzle.xy`): the pattern is the instance.
        if let Some(pat) = name.strip_prefix("shape.swizzle.") {
            if a.len() == 1 && !pat.is_empty() && pat.chars().all(|c| "xyzw".contains(c)) {
                return Ok(format!("({}).{pat}", a[0]));
            }
        }
        let template = self.table.named(name).ok_or_else(|| format!("{name} is not a row of the function table"))?;
        let col = template.naga.as_str();
        let arity = |n: usize| -> Result<(), String> { if a.len() == n { Ok(()) } else { Err(format!("{name} with {} children, {n} expected", a.len())) } };
        if let Some(op) = col.strip_prefix("Binary::") {
            arity(2)?;
            let sym = match op {
                "Add" => "+",
                "Subtract" => "-",
                "Multiply" => "*",
                "Divide" => "/",
                "Modulo" => "%",
                "Equal" => "==",
                "NotEqual" => "!=",
                "Less" => "<",
                "LessEqual" => "<=",
                "Greater" => ">",
                "GreaterEqual" => ">=",
                "And" => "&",
                "ExclusiveOr" => "^",
                "InclusiveOr" => "|",
                // The kingdom's logic rows are eager (WGSL's `&` and `|` on bool):
                // naga 30 runs a `&&`'s right side inside an if, where a subgroup
                // operation would leave uniform control (2026-10-19).
                "LogicalAnd" => "&",
                "LogicalOr" => "|",
                "ShiftLeft" => "<<",
                "ShiftRight" => ">>",
                other => return Err(format!("{name}: binary operator {other} has no text")),
            };
            // A shift count is masked, as WGSL requires it below the width.
            if op == "ShiftLeft" || op == "ShiftRight" {
                // The shift count is masked below the width and, for a vector
                // operand, splatted to the operand's lanes (WGSL's rule).
                let count = if d.lanes > 1 && d.slot.starts_with("vec") { format!("vec{}<u32>(({}) & 31u)", d.lanes, a[1]) } else { format!("(({}) & 31u)", a[1]) };
                return Ok(format!("({} {sym} {count})", a[0], count = count));
            }
            return Ok(format!("({} {sym} {})", a[0], a[1]));
        }
        if let Some(op) = col.strip_prefix("Unary::") {
            arity(1)?;
            let sym = match op {
                "Negate" => "-",
                "LogicalNot" => "!",
                "BitwiseNot" => "~",
                other => return Err(format!("{name}: unary operator {other} has no text")),
            };
            // A matrix negated is the matrix times -1.0, the same value:
            // naga 0.20 writes `-m` to Metal as a unary minus on a
            // metal::floatCxR, which Metal does not define (six random
            // kernels of fit_SF_random failed to compile, 2026-10-12).
            if op == "Negate" && d.slot.starts_with("mat") {
                return Ok(format!("({} * -1.0)", a[0]));
            }
            return Ok(format!("({sym}{})", a[0]));
        }
        if let Some(f) = col.strip_prefix("Relational::") {
            arity(1)?;
            return Ok(match f {
                "All" => format!("all({})", a[0]),
                "Any" => format!("any({})", a[0]),
                // WGSL has no isNan/isInf: the IEEE tests, written out.
                "IsNan" => format!("({0} != {0})", a[0]),
                "IsInf" => {
                    let big = if d.lanes > 1 { format!("vec{}<f32>(3.4028235e38)", d.lanes) } else { "3.4028235e38".to_string() };
                    format!("(abs({}) > {big})", a[0])
                }
                other => return Err(format!("{name}: relational {other} has no text")),
            });
        }
        if col.starts_with("Math::") {
            let f = match template.instance.as_str() {
                "inverse_sqrt" => "inverseSqrt",
                "face_forward" => "faceForward",
                "count_ones" => "countOneBits",
                "leading_zeros" => "countLeadingZeros",
                "trailing_zeros" => "countTrailingZeros",
                "find_lsb" => "firstTrailingBit",
                "find_msb" => "firstLeadingBit",
                "reverse" => "reverseBits",
                "extract" => "extractBits",
                "insert" => "insertBits",
                "outer" => {
                    // WGSL has no outer product builtin: a helper per shape.
                    let (c, r) = d.matrix_shape().ok_or_else(|| format!("{name}: output {} is not a matrix", d.dual))?;
                    let helper = format!("outer_{r}_{c}");
                    scope.helpers.push(helper.clone());
                    return Ok(format!("{helper}({})", a.join(", ")));
                }
                "inverse" => {
                    // WGSL has no inverse builtin: the adjugate over the determinant, per size.
                    let (c, _) = d.matrix_shape().ok_or_else(|| format!("{name}: output {} is not a matrix", d.dual))?;
                    let helper = format!("inverse_{c}");
                    scope.helpers.push(helper.clone());
                    return Ok(format!("{helper}({})", a.join(", ")));
                }
                // degrees and radians as the products naga writes for them,
                // without naga's bare parentheses: naga 0.20 writes
                // `((x) * 57.29..)` to Metal, and when x is a column of a
                // matrix composed in place, `(metal::float2x3(metal::float3(_e1),
                // ..)[0])` parses as a cast and the `*` as an indirection (Metal
                // refused random seed 673 to compile, 2026-10-13). The same value.
                "degrees" => return Ok(format!("({} * 57.29577951308232)", a[0])),
                "radians" => return Ok(format!("({} * 0.017453292519943295)", a[0])),
                "mix_scalar_weight" => "mix",
                other => other,
            };
            if template.instance == "ldexp" {
                // The exponent is integer in WGSL, per lane.
                let int_slot = if d.lanes > 1 { format!("vec{}<i32>", d.lanes) } else { "i32".to_string() };
                return Ok(format!("ldexp({}, {int_slot}({}))", a[0], a[1]));
            }
            return Ok(format!("{f}({})", a.join(", ")));
        }
        match col {
            "Select" => {
                arity(3)?;
                Ok(format!("select({}, {}, {})", a[1], a[0], a[2]))
            }
            "As" => {
                arity(1)?;
                Ok(format!("{}({})", slot_ctor(&d.slot)?, a[0]))
            }
            "As(bitcast)" => {
                arity(1)?;
                Ok(format!("bitcast<{}>({})", d.slot, a[0]))
            }
            "Compose" => Ok(format!("{}({})", slot_ctor(&d.slot)?, a.join(", "))),
            "Splat" => {
                arity(1)?;
                Ok(format!("{}({})", slot_ctor(&d.slot)?, a[0]))
            }
            "AccessIndex" => {
                arity(1)?;
                // A matrix's column: `m[i]`.
                if let Some(c) = template.instance.strip_prefix("col_") {
                    return Ok(format!("{}[{c}]", a[0]));
                }
                let i: usize = template.instance.strip_prefix("access_").and_then(|n| n.parse().ok()).ok_or_else(|| format!("{name}: not an access"))?;
                Ok(format!("{}.{}", a[0], ["x", "y", "z", "w"][i]))
            }
            "composite" => self.composite(name, &a, d, scope),
            other => Err(format!("{name}: naga column {other} has no text")),
        }
    }

    fn composite(&self, name: &str, a: &[String], out: &WgslDual, scope: &mut Scope) -> Result<String, String> {
        // Inline composites first; the rest call a helper from the library.
        let inline = match name {
            "index.of_int" => Some(format!("u32(max({}, 0))", a[0])),
            "index.of_count" => Some(a[0].clone()),
            "arith.modf_fract" => Some(format!("modf({}).fract", a[0])),
            "arith.modf_whole" => Some(format!("modf({}).whole", a[0])),
            "arith.frexp_fract" => Some(format!("frexp({}).fract", a[0])),
            "arith.frexp_exp" => Some(format!("frexp({}).exp", a[0])),
            // (A literal count through lit_u32: naga folds `max(0u, 1u) - 1u`
            // to 0u, read back as a min, 2026-10-17.)
            "index.clamp_to" => {
                let plain = a[1].strip_suffix('u').is_some_and(|d| !d.is_empty() && d.bytes().all(|c| c.is_ascii_digit()));
                Some(format!("min({}, max({}, 1u) - 1u)", a[0], if plain { format!("lit_u32({})", a[1]) } else { a[1].clone() }))
            }
            "index.band_of" => Some(format!("({} / max({}, 1u))", a[0], a[1])),
            "count.sub_floor" => Some(format!("select({0} - {1}, 0u, {1} > {0})", a[0], a[1])),
            "select.negate_if" => Some(format!("select(-({0}), {0}, {1})", a[0], a[1])),
            "index.shl" => Some(format!("(({}) << (({}) & 31u))", a[0], a[1])),
            "index.shr" => Some(format!("(({}) >> (({}) & 31u))", a[0], a[1])),
            "index.and" => Some(format!("(({}) & ({}))", a[0], a[1])),
            "index.sub_floor" => Some(format!("select(({0}) - ({1}), 0u, ({1}) > ({0}))", a[0], a[1])),
            "index.round_down" => Some(format!("(({0}) - (({0}) % max({1}, 1u)))", a[0], a[1])),
            "index.or" => Some(format!("(({}) | ({}))", a[0], a[1])),
            _ => None,
        };
        if let Some(text) = inline {
            return Ok(text);
        }
        let Some(&(_, helper)) = COMPOSITE_ROWS.iter().find(|(row, _)| *row == name) else {
            return Err(format!("{name}: composite with no WGSL body (out {})", out.dual));
        };
        // Helpers call helpers: declare their dependencies too.
        for dep in COMPOSITES.iter().filter(|(n, _)| *n == helper).flat_map(|(_, body)| COMPOSITES.iter().map(|(n, _)| *n).filter(|n| *n != helper && body.contains(&format!("{n}(")))) {
            scope.helpers.push(dep.to_string());
        }
        scope.helpers.push(helper.to_string());
        Ok(format!("{helper}({})", a.join(", ")))
    }

    fn literal(&self, form: &str, v: f64, d: &WgslDual, wrap: bool) -> Result<String, String> {
        let u = |x: u32| if wrap { format!("lit_u32({x}u)") } else { format!("{x}u") };
        let scalar = match form {
            "real" => {
                if !v.is_finite() {
                    return Err(format!("literal.real {v} is not finite"));
                }
                if d.scalar() == Some("f16") {
                    return Err("literal.real over f16: the prologue does not enable f16".into());
                }
                if wrap { format!("lit_f32({v:?}f)") } else { format!("{v:?}f") }
            }
            "index" | "count" | "bits" | "hash32" | "code4" | "code8" | "code16" | "code1024" => u(if v.is_finite() { v.clamp(0.0, 4_294_967_295.0) as u32 } else { 0 }),
            "wgdim" => u(workgroup_dim(v)),
            "lit64" => u(if v.is_finite() { (v.round() as i64).clamp(1, 64) as u32 } else { 1 }),
            "lit4096" => u(if v.is_finite() { (v.round() as i64).clamp(1, 4096) as u32 } else { 1 }),
            "blk" => u(snap(&BLK, v)),
            "unroll" => u(snap(&UNROLL, v)),
            "int" | "q15_16" => {
                // Concrete i32 (`-49i`): an abstract integer literal shifted or
                // multiplied by a runtime u32 or i32 is refused by naga 0.20
                // ("Cannot cast scalar components ... to type i32", random
                // kernels with the i32 rows, 2026-10-12). i32::MIN has no
                // literal (2147483648i is out of range).
                let x = v.clamp(-2_147_483_648.0, 2_147_483_647.0) as i32;
                let text = if x == i32::MIN { "(-2147483647i - 1i)".to_string() } else { format!("{x}i") };
                if wrap { format!("lit_i32({text})") } else { text }
            }
            // A bool in a literal-only expression passes through `lit_bool`
            // too: `log(f32(false))` was folded by naga to log(0) = -inf and
            // refused (a random kernel, 2026-10-11).
            "flag" | "sign" => if wrap { format!("lit_bool({})", v != 0.0) } else { (v != 0.0).to_string() },
            other => return Err(format!("literal.{other} has no text")),
        };
        if d.lanes > 1 && d.slot.starts_with("vec") {
            return Ok(format!("{}({scalar})", slot_ctor(&d.slot)?));
        }
        Ok(scalar)
    }

    fn fallback_value(&self, d: &WgslDual, wrap: bool) -> Result<String, String> {
        if d.is_statement_level() {
            return Err(format!("a {} fallback where a value is needed", d.dual));
        }
        // The zero-value constructor for a vector or matrix; a scalar through
        // lit_* only inside a literal-only expression. An i32 zero is the
        // concrete `0i`, as a literal is written (an abstract `0` read back
        // as another value than the literal 0i wrote; 2026-10-13).
        let f = |x: String, lit: &str| if wrap { format!("{lit}({x})") } else { x };
        if d.slot.starts_with("vec") || d.slot.starts_with("mat") {
            if d.scalar() == Some("f16") {
                return Err("f16 fallback: the prologue does not enable f16".into());
            }
            let lane = match d.scalar() {
                Some("f32") => f("0.0f".to_string(), "lit_f32"),
                Some("i32") => f("0i".to_string(), "lit_i32"),
                Some("u32") => f(format!("{}u", d.fallback_bits), "lit_u32"),
                Some("bool") => f((d.fallback_bits != 0).to_string(), "lit_bool"),
                _ => return Err(format!("{}: no value fallback", d.dual)),
            };
            if let Some((c, r)) = d.matrix_shape() {
                let col = format!("vec{r}<f32>({lane})");
                return Ok(format!("{}({})", slot_ctor(&d.slot)?, vec![col; c as usize].join(", ")));
            }
            return Ok(format!("{}({lane})", slot_ctor(&d.slot)?));
        }
        Ok(match d.scalar() {
            Some("f32") => f("0.0f".to_string(), "lit_f32"),
            Some("f16") => return Err("f16 fallback: the prologue does not enable f16".into()),
            Some("i32") => f("0i".to_string(), "lit_i32"),
            Some("u32") => f(format!("{}u", d.fallback_bits), "lit_u32"),
            Some("bool") => f((d.fallback_bits != 0).to_string(), "lit_bool"),
            _ => return Err(format!("{}: no value fallback", d.dual)),
        })
    }
}

/// A static estimate of a kernel's work per thread: `base` operations
/// outside the item loop, `per_item` inside it (multiplied by `cfg.m` at
/// run time), from node counts with loop multipliers. A search bounds the
/// per-item cost against the shipped kernel's before dispatching: a tile
/// loop inside the item loop multiplies the work by up to 32, and a
/// kernel a few hundred times the shipped cost wedges the device (the
/// 2026-10-09 run hung 22 minutes in generation 23).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost {
    pub base: f64,
    pub per_item: f64,
}

pub fn cost(t: &Tree) -> Cost {
    fn node_cost(name: &str) -> f64 {
        match name {
            n if n.starts_with("load.") || n.starts_with("store.") || n.starts_with("wg.") => 4.0,
            // An atomic read-modify-write on workgroup memory.
            n if n.starts_with("atomic.") => 8.0,
            // A subgroup operation (a few SIMD lane exchanges) and its barrier.
            n if n.starts_with("subgroup.") => 4.0,
            n if n.starts_with("stmt.subgroup_barrier") => 10.0,
            "geom.inverse" | "hash.mix64" | "fixed.wide_mul" | "fixed.mul" => 40.0,
            n if n.starts_with("trig.") || n.starts_with("arith.exp") || n.starts_with("arith.log") || n.starts_with("arith.pow") => 8.0,
            "stmt.barrier" => 50.0,
            n if n.starts_with("stmt.storage_barrier") => 50.0,
            "fallback" => 0.0,
            _ => 1.0,
        }
    }
    // (base, per_item) of a subtree; `tile` is the enclosing tile multiplier,
    // `in_items` whether an item loop encloses it.
    fn walk(t: &Tree, tile: f64, in_items: bool) -> (f64, f64) {
        let own = node_cost(&t.name) * tile;
        let (mut base, mut per_item) = if in_items { (0.0, own) } else { (own, 0.0) };
        // A bounded literal as the renderer snaps it (1..=max), 1 for a fallback.
        let lit = |k: Option<&Tree>, max: f64| k.map_or(1.0, |k| if k.name.starts_with("literal.") && k.konst.is_finite() { f64::from(k.konst).round().clamp(1.0, max) } else { 1.0 });
        match safe_base(&t.name) {
            "loop.items" | "loop.items_unrolled" | "loop.items_bounded" | "loop.items_unrolled_bounded" | "loop.items_cont_bounded" => {
                // The body runs once per item, times the passes an expression
                // bound makes over the items; a nested long loop is refused by the renderer.
                let bound_i = if safe_base(&t.name) == "loop.items_unrolled_bounded" { 1 } else { 0 };
                let passes = items_passes(t.kids.get(bound_i));
                for k in &t.kids {
                    let (b, p) = walk(k, tile, true);
                    per_item += (b + p) * passes;
                }
            }
            "loop.items_strided_bounded" | "loop.items_cont_strided_bounded" => {
                // (bound, stride, body): the body runs once per `stride` items.
                let stride = lit(t.kids.get(1), 4096.0);
                let passes = items_passes(t.kids.first());
                for k in &t.kids {
                    let (b, p) = walk(k, tile, true);
                    per_item += (b + p) * passes / stride;
                }
            }
            "loop.tile" => {
                let n = t.kids.first().map_or(4.0, |k| if k.name.starts_with("literal.") { f64::from(snap(&WGDIM, k.konst)) } else { f64::from(WGDIM[0]) });
                for k in &t.kids {
                    let (b, p) = walk(k, tile * n, in_items);
                    base += b;
                    per_item += p;
                }
            }
            "loop.tilex_bounded" | "loop.tilex_strided_bounded" => {
                // (bound expression, [step,] body): the bound evaluated at the worst
                // shape (d 8192, m 65536, r_rows 8192; a register unknown: 8192).
                let count = t.kids.first().map_or(8192.0, worst_bound);
                let step = if safe_base(&t.name) == "loop.tilex_strided_bounded" { lit(t.kids.get(1), 4096.0) } else { 1.0 };
                let n = (count / step).ceil().max(1.0);
                // A loop OUTSIDE the item loop whose trip count grows with m is
                // item work (fit_A2 gen 6's r12p1: `for (tt < cfg.m)` under the
                // thread's guard): its body is priced per item, n / m passes over
                // the items, not n times into the base.
                if !in_items && t.kids.first().is_some_and(grows_with_m) {
                    let passes = n / 65536.0;
                    for k in &t.kids {
                        let (b, p) = walk(k, tile, true);
                        per_item += (b + p) * passes;
                    }
                    return (base, per_item);
                }
                for k in &t.kids {
                    let (b, p) = walk(k, tile * n, in_items);
                    base += b;
                    per_item += p;
                }
            }
            "loop.tile_bounded" | "loop.tile_strided_bounded" | "loop.unroll" => {
                // (count, [step,] body): the body runs ceil(count / step) times, nested tile loops multiply
                // (an unrolled loop as many times, written out).
                let count = lit(t.kids.first(), 64.0);
                let step = if safe_base(&t.name) == "loop.tile_strided_bounded" { lit(t.kids.get(1), 64.0) } else { 1.0 };
                let n = (count / step).ceil();
                for k in &t.kids {
                    let (b, p) = walk(k, tile * n, in_items);
                    base += b;
                    per_item += p;
                }
            }
            "wg.reduce_add" | "wg.reduce_max" | "wg.reduce_min" | "wg.reduce_argmax" | "wg.reduce_argmin" => {
                // A tree reduction over n (rounded down to a power of two): the
                // leading barrier, then log2(n) steps of a guard, two loads, the
                // operation, a store and a barrier (an argmax's two more loads
                // and a store: counted alike, the steps dominate).
                let n = lit(t.kids.last(), 4096.0);
                let steps = n.log2().floor();
                let extra = (50.0 + steps * (1.0 + 4.0 + 4.0 + 1.0 + 4.0 + 50.0)) * tile;
                if in_items {
                    per_item += extra;
                } else {
                    base += extra;
                }
                for k in &t.kids {
                    let (b, p) = walk(k, tile, in_items);
                    base += b;
                    per_item += p;
                }
            }
            _ => {
                for k in &t.kids {
                    let (b, p) = walk(k, tile, in_items);
                    base += b;
                    per_item += p;
                }
            }
        }
        (base, per_item)
    }
    // Knob references priced at their values; a grid the renderer cannot
    // read prices them as their payloads (the render names that grid).
    let resolved = crate::wgsl::params::resolve(t).unwrap_or_else(|_| t.clone());
    let (base, per_item) = walk(&resolved, 1.0, false);
    Cost { base, per_item }
}

/// The worst shape the fitness times: the width d, the item count m and the
/// band's rows (r_off is 0). The legality checker (`check.rs`) proves its
/// ranges at the same shape.
pub const WORST_D: f64 = 8192.0;
pub const WORST_M: f64 = 65536.0;
pub const WORST_ROWS: f64 = 8192.0;

/// An index expression evaluated at the worst shape the fitness times
/// (d 8192, m 65536, r_rows 8192, r_off 0), for the cost of a loop it
/// bounds; a value the estimator cannot see (a register, a load) is 8192.
/// The result is clamped to [1, 65536].
pub fn worst_bound(t: &Tree) -> f64 {
    bound_at_worst(t).0
}

/// Whether a loop bound grows with the item count m: valued fully (every
/// leaf seen) at the worst shape and again with m doubled, it is larger.
fn grows_with_m(t: &Tree) -> bool {
    let (at, seen) = bound_at(t, WORST_M);
    seen && bound_at(t, 2.0 * WORST_M).0 > at
}

/// [`worst_bound`] and whether every value in the expression was seen: a
/// register, a load or a row the estimator has no case for is taken as
/// 8192 and makes the value unseen.
fn bound_at_worst(t: &Tree) -> (f64, bool) {
    bound_at(t, WORST_M)
}

/// [`bound_at_worst`] with the item count m given.
fn bound_at(t: &Tree, m: f64) -> (f64, bool) {
    fn eval(t: &Tree, seen: &mut bool, m: f64) -> f64 {
        if t.name.starts_with("literal.") {
            return if t.konst.is_finite() { f64::from(t.konst).max(0.0) } else { 0.0 };
        }
        let mut k = |i: usize| match t.kids.get(i) {
            Some(x) => eval(x, seen, m),
            None => {
                *seen = false;
                8192.0
            }
        };
        match uniform_as_index(&t.name).unwrap_or(safe_base(&t.name)) {
            "index.of_uniform" | "uniform.of_lit" | "uniform.of_checked" | "uniform.of_counter" | "uniform.of_tcounter" | "uniform.of_tcounter2" | "uniform.of_tcounter3" | "uniform.of_tcounter4" => k(0),
            "bound.of_index" | "bound.of_uniform" | "bound.x_of_uniform" => k(0).min(f64::from(MAX_LOOP_TRIP)),
            "load.cfg_d" | "bound.d" => WORST_D,
            "load.cfg_r_rows" | "bound.rows" => WORST_ROWS,
            "load.cfg_m" | "bound.m" => m,
            "load.cfg_r_off" => 0.0,
            "index.of_count" | "safe.counter_index" | "safe.tile_index" | "safe.tile_index2" | "safe.tile_index3" => k(0),
            "index.add" | "count.add" => k(0) + k(1),
            "index.sub" => (k(0) - k(1)).max(0.0),
            "index.sub_floor" | "count.sub_floor" => (k(0) - k(1)).max(0.0),
            "index.mul" => k(0) * k(1),
            "index.div" | "index.band_of" => k(0) / k(1).max(1.0),
            "index.mod" => k(1).max(1.0) - 1.0,
            "index.min" | "count.min" | "index.clamp_to" => k(0).min(k(1)),
            "index.max" | "count.max" => k(0).max(k(1)),
            "index.and" => k(0).min(k(1)),
            "index.or" => k(0).max(k(1)) * 2.0,
            "index.shl" => k(0) * 2f64.powf(k(1).min(31.0)),
            "index.shr" => k(0) / 2f64.powf(k(1).min(31.0)),
            _ => {
                *seen = false;
                8192.0
            }
        }
    }
    let mut seen = true;
    let v = eval(t, &mut seen, m).clamp(1.0, 1.0e12);
    (v, seen)
}

/// The item loop's passes over the items relative to the worst shape's m:
/// 1 for the bindings' counts; for an expression bound, its value at the
/// worst shape over m, so a quadratic bound (`cfg.m * cfg.d`, 8,192 passes)
/// is priced by its true size and a loop over item groups
/// (`(cfg.m + 15u) / 16u`, fit_A2's fresh43_g12: a sixteenth of a pass) by
/// its true size too. An expression with a value the estimator cannot see
/// is at least one pass.
fn items_passes(bound: Option<&Tree>) -> f64 {
    match bound {
        Some(b) if matches!(safe_base(&b.name), "bound.of_index" | "bound.of_uniform") => {
            let (v, seen) = bound_at_worst(b);
            if seen {
                v / 65536.0
            } else {
                (v / 65536.0).max(1.0)
            }
        }
        _ => 1.0,
    }
}

/// A safeWGSL row name reduced to its renderer case: the rung suffix
/// (`_L0`, `_C1`, …) stripped, a bound-specific loop (`loop.items_boundm`)
/// mapped to `loop.items_bounded`, a rung-specific store (`store.h_L1`) to
/// `store.h_safe`, a rung-specific wg set to `wg.set_safe`. A WGSL kingdom
/// name passes through unchanged.
/// Whether a tree writes a mirror cell (the symmetric half, h[col * d + row]):
/// its kernel runs with the whole H in one band (`// mode full`).
pub fn writes_mirror(t: &Tree) -> bool {
    t.name.starts_with("safe.mirror_of") || t.name.starts_with("store.h_mirror") || t.kids.iter().any(writes_mirror)
}

pub fn safe_base(name: &str) -> &str {
    for rung in ["_L0", "_L1", "_L2", "_L3", "_L4", "_L5", "_C0", "_C1", "_C2", "_C3", "_C4", "_C5", "_O0", "_O1", "_O2", "_O3", "_O4", "_B1", "_B2", "_B3", "_B4", "_B5"] {
        if let Some(stem) = name.strip_suffix(rung) {
            return match stem {
                "store.h" => "store.h_safe",
                "wg.set" | "wg.set_v" | "wg.set_v2" | "wg.set_u" | "wg.set_n" => "wg.set_safe",
                "wg.put" | "wg.put_v" | "wg.put_v2" | "wg.put_u" | "wg.put_n" => "wg.put",
                "parr.set_f" | "parr.set_v" | "parr.set_i" | "parr.set_b" | "parr.set_t" => "parr.set_safe",
                "reg.set_v_lane" => "reg.set_v_lane",
                "wg.set_v_lane" => "wg.set_v_lane",
                "parr.set_v_lane" => "parr.set_v_lane",
                "loop.tile" => "loop.tile_bounded",
                "loop.tile_strided" => "loop.tile_strided_bounded",
                "loop.tilex" => "loop.tilex_bounded",
                "loop.tilex_strided" => "loop.tilex_strided_bounded",
                // A loop that breaks is its tile loop; its body (block.brk) holds the break.
                "loop.tile_brk" => "loop.tile_bounded",
                "loop.tilex_brk" => "loop.tilex_bounded",
                "loop.tilex_strided_brk" => "loop.tilex_strided_bounded",
                other => other,
            };
        }
    }
    if name.starts_with("loop.items_unrolled_bound") {
        return "loop.items_unrolled_bounded";
    }
    if name.starts_with("loop.items_cont_strided_bound") {
        return "loop.items_cont_strided_bounded";
    }
    if name.starts_with("loop.items_cont_bound") {
        return "loop.items_cont_bounded";
    }
    if name.starts_with("loop.items_strided_bound") {
        return "loop.items_strided_bounded";
    }
    if name.starts_with("loop.items_bound") {
        return "loop.items_bounded";
    }
    name
}

impl Renderer<'_> {
    /// A private array leaf's variable name; a fallback leaf names the array
    /// of ITS DUAL (`pv` for a vec4 demand, `pi` for u32, `pm` for bool).
    /// An array's index as text: a masked literal (`pv[((lit_u32(0u)) & 15u)]`
    /// before 2026-10-11) is folded on the host into a plain literal inside
    /// the array, `pv[0u]`, so the text a model copies is plain; any other
    /// index is written as its row renders it.
    fn index_text(&self, idx: &Tree, len: u32, scope: &mut Scope) -> Result<String, String> {
        // A workgroup array's index masked into ITS length (the plan's): `&`
        // for a power of two, `min` otherwise (an array padded to 1,056).
        if let ("safe.wg_of" | "safe.wgv_of" | "safe.wgv2_of" | "safe.wga_of", [k]) = (safe_base(&idx.name), idx.kids.as_slice()) {
            let last = len.max(1) - 1;
            if k.name.starts_with("literal.") && k.konst.is_finite() {
                let v = k.konst.max(0.0) as u32;
                return Ok(format!("{}u", if len.is_power_of_two() { v & last } else { v.min(last) }));
            }
            let x = self.value(k, scope)?;
            return Ok(if len.is_power_of_two() { format!("(({x}) & {last}u)") } else { format!("min(({x}), {last}u)") });
        }
        if let ["safe.pidx64" | "safe.pidx16" | "safe.pidx128"] = [safe_base(&idx.name)] {
            if let [k] = idx.kids.as_slice() {
                if k.name.starts_with("literal.") && k.konst.is_finite() {
                    let mask = match safe_base(&idx.name) {
                        "safe.pidx64" => 63,
                        "safe.pidx128" => 127,
                        _ => 15,
                    };
                    let v = (k.konst.max(0.0) as u32 & mask).min(len.saturating_sub(1));
                    return Ok(format!("{v}u"));
                }
            }
        }
        self.value(idx, scope)
    }

    /// The thread's own row, column or cell for a fallback of those duals
    /// (the origin of its block: `gid * blk`, or for a strided or grouped
    /// block `wid * (W * blk) + lid`); None for any other dual.
    fn own_fallback(&self, dual: &str, scope: &mut Scope) -> Result<Option<String>, String> {
        let ty = |n: &str| dual_ty(self.kingdom, n);
        let row = |name: &str, d: &str| -> Result<Tree, String> { Ok(Tree::new(name, ty(d)?, vec![])) };
        let tree = match dual {
            "u32.threadrow" | "u32.rowbase" => row("safe.thread_row", "u32.threadrow")?,
            "u32.threadcol" | "u32.colbase" => row("safe.thread_col", "u32.threadcol")?,
            "u32.threadrows" | "u32.threadrowg" | "u32.rowbases" | "u32.rowbaseg" => row("safe.thread_row_s", "u32.threadrows")?,
            "u32.threadcols" | "u32.threadcolg" | "u32.colbases" | "u32.colbaseg" => row("safe.thread_col_s", "u32.threadcols")?,
            "u32.cell" => Tree::new("safe.cell_of", ty("u32.cell")?, vec![row("safe.thread_row", "u32.threadrow")?, row("safe.thread_col", "u32.threadcol")?]),
            "u32.mirror" => Tree::new("safe.mirror_of", ty("u32.mirror")?, vec![row("safe.thread_row", "u32.threadrow")?, row("safe.thread_col", "u32.threadcol")?]),
            "u32.wgown" => return Ok(Some("lidx".into())),
            _ => return Ok(None),
        };
        self.value(&tree, scope).map(Some)
    }

    fn parr_name(&self, t: &Tree) -> Result<&'static str, String> {
        parr_var(self.kingdom, t)
    }
}

/// A private array leaf's variable name; a fallback leaf names the array
/// of ITS DUAL (`pv` for a vec4 demand, `pi` for u32, `pm` for bool).
pub fn parr_var(kingdom: &WgslKingdom, t: &Tree) -> Result<&'static str, String> {
    if t.name == "fallback" {
        let d = kingdom.dual(t.ty).map(|d| d.dual.clone()).unwrap_or_default();
        return Ok(array_names().rows.get(d.as_str()).and_then(|r| r.first()).and_then(|r| array_names().var.get(*r)).copied().unwrap_or("pa"));
    }
    parr_name(t)
}

/// Whether every leaf of `t` is a literal or a fallback (a constant expression to naga).
fn literal_only(t: &Tree) -> bool {
    if t.kids.is_empty() {
        return t.name.starts_with("literal.") || t.name == "fallback";
    }
    t.kids.iter().all(literal_only)
}

/// A register's WGSL variable name: `f3`, `v0`, `i12`, `b1`; `f16`, `f32`
/// and `i32` are spelled `f16r`, `f32r`, `i32r`, because they are reserved
/// WGSL words (the types).
pub fn register_var(prefix: &str, k: usize) -> String {
    let name = format!("{prefix}{k}");
    if ["f16", "f32", "i32", "u32"].contains(&name.as_str()) {
        format!("{name}r")
    } else {
        name
    }
}

/// The declarations every rendered kernel opens with (registers initialised
/// to their form's fallback, the loop counters, the private arrays).
pub fn register_declarations() -> String {
    let mut out = String::new();
    for (prefix, n, ty, init) in [("f", 64, "f32", "0.0"), ("v", 32, "vec4<f32>", "vec4<f32>(0.0)"), ("i", 64, "u32", "0u"), ("b", 16, "bool", "false"), ("n", 16, "i32", "0"), ("u", 16, "vec4<u32>", "vec4<u32>(0u)"), ("w", 16, "vec2<f32>", "vec2<f32>(0.0)"), ("x", 4, "mat2x2<f32>", "mat2x2<f32>()"), ("y", 4, "mat3x3<f32>", "mat3x3<f32>()"), ("z", 4, "mat4x4<f32>", "mat4x4<f32>()")].into_iter().map(|(p, n, t, i)| (p.to_string(), n, t.to_string(), i.to_string())).chain(EXTRA_REGISTERS.iter().map(|x| (x.letter.to_string(), x.count, x.slot.to_string(), format!("{}()", x.slot)))) {
        for k in 0..n {
            if k % 8 == 0 {
                out.push_str("    ");
            }
            out.push_str(&format!("var {}: {ty} = {init}; ", register_var(&prefix, k)));
            if k % 8 == 7 || k == n - 1 {
                out.push('\n');
            }
        }
    }
    out.push_str("    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u; var k4: u32 = 0u;\n");
    out.push_str("    var pa: array<f32, 64>; var pb: array<f32, 64>; var pc: array<f32, 64>; var pd: array<f32, 64>; var pe: array<f32, 64>; var pf: array<f32, 64>;\n");
    out.push_str("    var pv: array<vec4<f32>, 16>; var pw: array<vec4<f32>, 16>; var px: array<vec4<f32>, 16>; var py: array<vec4<f32>, 16>; var pi: array<u32, 16>; var pj: array<u32, 16>; var pk: array<u32, 16>; var pm: array<bool, 16>; var pn: array<bool, 16>; var pq: array<vec2<f32>, 16>; var pr: array<vec2<f32>, 16>; var pua: array<u32, 64>; var puc: array<u32, 64>; var pvq: array<vec4<f32>, 64>; var pvr: array<vec4<f32>, 64>; var pux: array<u32, 128>; var puy: array<u32, 128>;\n");
    // The private arrays past the ones above, from the families.
    for f in &PRIVATE_ARRAYS {
        for name in f.names().skip(f.legacy.len()) {
            out.push_str(&format!("    var p{name}: array<{}, {}>;\n", f.elem, f.len));
        }
    }
    out
}

/// Whether `text` names the identifier `name` (a whole word: no identifier
/// character on either side).
pub fn names_word(text: &str, name: &str) -> bool {
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut from = 0;
    while let Some(i) = text[from..].find(name) {
        let at = from + i;
        let before = text[..at].chars().next_back();
        let after = text[at + name.len()..].chars().next();
        if !before.is_some_and(ident) && !after.is_some_and(ident) {
            return true;
        }
        from = at + name.len();
    }
    false
}

/// THE DECLARATIONS A KERNEL USES (2026-10-14: every kernel declared all 64
/// f32, 32 vec4, 64 u32 registers, 16 bools, 16 i32s, 8 vec4<u32>s, six
/// matrices, the counters and 17 private arrays, zero-initialised per
/// thread): of [`register_declarations`], each variable `body` names. One
/// the body never names is never read or written, so leaving it out
/// changes no value.
pub fn register_declarations_for(body: &str) -> String {
    let mut out = String::new();
    for decl in register_declarations().split(';').map(str::trim).filter(|d| !d.is_empty()) {
        let Some(name) = decl.strip_prefix("var ").and_then(|r| r.split(':').next()).map(str::trim) else { continue };
        if names_word(body, name) {
            out.push_str(&format!("    {decl};\n"));
        }
    }
    out
}

/// The length of a workgroup or private array by its variable name.
pub fn array_len(name: &str) -> u32 {
    if name.starts_with("wg_") {
        return wg_default_len(name);
    }
    // A private array: the length its dual says (`parr64<u32>`, `parr128<u32>`; 16 for the rest).
    let dual = PRIVATE_ARRAYS.iter().find(|f| f.names().any(|n| format!("p{n}") == name)).map(|f| f.dual);
    dual.and_then(|d| d.strip_prefix("parr")).and_then(|r| r.split('<').next()).and_then(|n| n.parse().ok()).unwrap_or(if matches!(name, "pa" | "pb" | "pc" | "pd" | "pe" | "pf") || (name.starts_with("pf") && name.len() > 2) { 64 } else { 16 })
}

/// The element type of a workgroup array by its variable name.
pub fn wg_elem(var: &str) -> &'static str {
    if var.starts_with("wg_au") {
        "atomic<u32>"
    } else if var.starts_with("wg_ai") {
        "atomic<i32>"
    } else if var.starts_with("wg_u") {
        "u32"
    } else if var.starts_with("wg_n") {
        "i32"
    } else if var.starts_with("wg_v2") {
        "vec2<f32>"
    } else if var.starts_with("wg_v") {
        "vec4<f32>"
    } else {
        "f32"
    }
}

/// A workgroup array's length when the kernel does not declare one.
pub fn wg_default_len(var: &str) -> u32 {
    if var.starts_with("wg_au") || var.starts_with("wg_ai") || var.starts_with("wg_v2") {
        1024
    } else if var.starts_with("wg_v") {
        WGVEC_LEN
    } else {
        WGMEM_LEN
    }
}

/// A private array leaf's variable name.
fn parr_name(t: &Tree) -> Result<&'static str, String> {
    array_names().var.get(t.name.as_str()).copied().filter(|_| t.name.starts_with("parr.")).ok_or_else(|| format!("{} is not a private array", t.name))
}

/// The constructor spelling of a slot: `vec3<f32>`, `f32`, `mat2x3<f32>`.
fn slot_ctor(slot: &str) -> Result<String, String> {
    if slot.starts_with("vec") || slot.starts_with("mat") || ["f32", "i32", "u32", "bool"].contains(&slot) {
        return Ok(slot.to_string());
    }
    Err(format!("slot {slot} has no constructor"))
}

/// The composite library: a helper function per composite row that is not
/// a one-line expression. Q15.16 arithmetic is on 32-bit two's complement
/// with explicit saturation; the 64-bit products use 16-bit limbs.
pub const COMPOSITES: &[(&str, &str)] = &[
    // The i32 division and remainder with WGSL's values where Metal has
    // none (a divisor of 0 or -1), 2026-10-13.
    ("int_div", "fn int_div(x: i32, b: i32) -> i32 { return select(x / select(select(b, 1i, b == 0i), 1i, b == -1i), bitcast<i32>(0u - bitcast<u32>(x)), b == -1i); }"),
    ("int_mod", "fn int_mod(x: i32, b: i32) -> i32 { return x % select(select(b, 1i, b == 0i), 1i, b == -1i); }"),
    ("int_div2", "fn int_div2(x: vec2<i32>, b: vec2<i32>) -> vec2<i32> { return select(x / select(select(b, vec2<i32>(1i), b == vec2<i32>(0i)), vec2<i32>(1i), b == vec2<i32>(-1i)), bitcast<vec2<i32>>(vec2<u32>(0u) - bitcast<vec2<u32>>(x)), b == vec2<i32>(-1i)); }"),
    ("int_mod2", "fn int_mod2(x: vec2<i32>, b: vec2<i32>) -> vec2<i32> { return x % select(select(b, vec2<i32>(1i), b == vec2<i32>(0i)), vec2<i32>(1i), b == vec2<i32>(-1i)); }"),
    ("int_div3", "fn int_div3(x: vec3<i32>, b: vec3<i32>) -> vec3<i32> { return select(x / select(select(b, vec3<i32>(1i), b == vec3<i32>(0i)), vec3<i32>(1i), b == vec3<i32>(-1i)), bitcast<vec3<i32>>(vec3<u32>(0u) - bitcast<vec3<u32>>(x)), b == vec3<i32>(-1i)); }"),
    ("int_mod3", "fn int_mod3(x: vec3<i32>, b: vec3<i32>) -> vec3<i32> { return x % select(select(b, vec3<i32>(1i), b == vec3<i32>(0i)), vec3<i32>(1i), b == vec3<i32>(-1i)); }"),
    ("int_div4", "fn int_div4(x: vec4<i32>, b: vec4<i32>) -> vec4<i32> { return select(x / select(select(b, vec4<i32>(1i), b == vec4<i32>(0i)), vec4<i32>(1i), b == vec4<i32>(-1i)), bitcast<vec4<i32>>(vec4<u32>(0u) - bitcast<vec4<u32>>(x)), b == vec4<i32>(-1i)); }"),
    ("int_mod4", "fn int_mod4(x: vec4<i32>, b: vec4<i32>) -> vec4<i32> { return x % select(select(b, vec4<i32>(1i), b == vec4<i32>(0i)), vec4<i32>(1i), b == vec4<i32>(-1i)); }"),
    ("q_real_to_code4", "fn q_real_to_code4(x: f32) -> u32 { return u32(clamp(round((clamp(x, -1.0, 1.0) + 1.0) * 1.5), 0.0, 3.0)); }"),
    ("q_code4_to_real", "fn q_code4_to_real(c: u32) -> f32 { return f32(min(c, 3u)) * 0.6666667 - 1.0; }"),
    ("q_real_to_code8", "fn q_real_to_code8(x: f32) -> u32 { return u32(clamp(round((clamp(x, -1.0, 1.0) + 1.0) * 3.5), 0.0, 7.0)); }"),
    ("q_code8_to_real", "fn q_code8_to_real(c: u32) -> f32 { return f32(min(c, 7u)) * 0.2857143 - 1.0; }"),
    ("q_real_to_code16", "fn q_real_to_code16(x: f32) -> u32 { return u32(clamp(round((clamp(x, -1.0, 1.0) + 1.0) * 7.5), 0.0, 15.0)); }"),
    ("q_code16_to_real", "fn q_code16_to_real(c: u32) -> f32 { return f32(min(c, 15u)) * 0.13333334 - 1.0; }"),
    ("q_real_to_code1024", "fn q_real_to_code1024(x: f32) -> u32 { return u32(clamp(round((clamp(x, -1.0, 1.0) + 1.0) * 511.5), 0.0, 1023.0)); }"),
    ("q_code1024_to_real", "fn q_code1024_to_real(c: u32) -> f32 { return f32(min(c, 1023u)) * 0.0019550342 - 1.0; }"),
    ("q_pack4_code8", "fn q_pack4_code8(a: u32, b: u32, c: u32) -> u32 { return (a & 255u) | ((b & 255u) << 8u) | ((c & 255u) << 16u); }"),
    ("q_pack4_code8_hi", "fn q_pack4_code8_hi(p: u32, d: u32) -> u32 { return (p & 16777215u) | ((d & 255u) << 24u); }"),
    ("q_unpack_code8", "fn q_unpack_code8(p: u32, i: u32) -> u32 { return (p >> ((i & 3u) * 8u)) & 255u; }"),
    ("fx_to_f32", "fn fx_to_f32(a: i32) -> f32 { return f32(a) * (1.0 / 65536.0); }"),
    ("fx_from_f32", "fn fx_from_f32(x: f32) -> i32 { return i32(clamp(round(x * 65536.0), -2147483648.0, 2147483520.0)); }"),
    ("sign_to_real", "fn sign_to_real(s: bool) -> f32 { return select(-1.0, 1.0, s); }"),
    ("pack_sign", "fn pack_sign(s: bool) -> u32 { return select(0u, 1u, s); }"),
    ("insert_lo", "fn insert_lo(e: u32, n: u32, o: u32) -> u32 { return insertBits(e, n, o & 31u, 32u - (o & 31u)); }"),
    ("insert_hi", "fn insert_hi(e: u32, o: u32) -> u32 { return select(e & ((1u << (o & 31u)) - 1u), e, o >= 32u); }"),
    ("unpack_sign", "fn unpack_sign(b: u32, i: u32) -> bool { return ((b >> (i & 31u)) & 1u) == 1u; }"),
    ("hash_to_bits", "fn hash_to_bits(h: u32) -> u32 { return h; }"),
    ("hash_to_unit", "fn hash_to_unit(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }"),
    ("hash_combine", "fn hash_combine(h: u32, b: u32) -> u32 { return h ^ b; }"),
    ("hash_lo", "fn hash_lo(h: vec2<u32>) -> u32 { return h.x; }"),
    ("hash_hi", "fn hash_hi(h: vec2<u32>) -> u32 { return h.y; }"),
    ("at_vec2f", "fn at_vec2f(v: vec2<f32>, i: u32) -> f32 { var t = v; return t[i & 1u]; }"),
    ("at_vec3f", "fn at_vec3f(v: vec3<f32>, i: u32) -> f32 { var t = v; return t[min(i, 2u)]; }"),
    ("at_vec4f", "fn at_vec4f(v: vec4<f32>, i: u32) -> f32 { var t = v; return t[i & 3u]; }"),
    ("at_vec2u", "fn at_vec2u(v: vec2<u32>, i: u32) -> u32 { var t = v; return t[i & 1u]; }"),
    ("at_vec3u", "fn at_vec3u(v: vec3<u32>, i: u32) -> u32 { var t = v; return t[min(i, 2u)]; }"),
    ("at_vec4u", "fn at_vec4u(v: vec4<u32>, i: u32) -> u32 { var t = v; return t[i & 3u]; }"),
    ("at_vec2i", "fn at_vec2i(v: vec2<i32>, i: u32) -> i32 { var t = v; return t[i & 1u]; }"),
    ("at_vec3i", "fn at_vec3i(v: vec3<i32>, i: u32) -> i32 { var t = v; return t[min(i, 2u)]; }"),
    ("at_vec4i", "fn at_vec4i(v: vec4<i32>, i: u32) -> i32 { var t = v; return t[i & 3u]; }"),
    ("at_vec2b", "fn at_vec2b(v: vec2<bool>, i: u32) -> bool { var t = v; return t[i & 1u]; }"),
    ("at_vec3b", "fn at_vec3b(v: vec3<bool>, i: u32) -> bool { var t = v; return t[min(i, 2u)]; }"),
    ("at_vec4b", "fn at_vec4b(v: vec4<bool>, i: u32) -> bool { var t = v; return t[i & 3u]; }"),
    ("inverse_2", "fn inverse_2(m: mat2x2<f32>) -> mat2x2<f32> {\n    let inv_det = 1.0 / determinant(m);\n    return mat2x2<f32>(vec2<f32>(m[1][1], -m[0][1]) * inv_det, vec2<f32>(-m[1][0], m[0][0]) * inv_det);\n}"),
    ("inverse_3", "fn inverse_3(m: mat3x3<f32>) -> mat3x3<f32> {\n    // Columns of the adjugate: cross products of the columns, transposed.\n    let c0 = cross(m[1], m[2]);\n    let c1 = cross(m[2], m[0]);\n    let c2 = cross(m[0], m[1]);\n    let inv_det = 1.0 / dot(m[0], c0);\n    return transpose(mat3x3<f32>(c0, c1, c2)) * inv_det;\n}"),
    ("inverse_4", "fn inverse_4(m: mat4x4<f32>) -> mat4x4<f32> {\n    // The adjugate from 2x2 sub-determinants (column-major m[col][row]).\n    let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];\n    let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];\n    let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];\n    let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];\n    let b00 = a00 * a11 - a01 * a10; let b01 = a00 * a12 - a02 * a10; let b02 = a00 * a13 - a03 * a10;\n    let b03 = a01 * a12 - a02 * a11; let b04 = a01 * a13 - a03 * a11; let b05 = a02 * a13 - a03 * a12;\n    let b06 = a20 * a31 - a21 * a30; let b07 = a20 * a32 - a22 * a30; let b08 = a20 * a33 - a23 * a30;\n    let b09 = a21 * a32 - a22 * a31; let b10 = a21 * a33 - a23 * a31; let b11 = a22 * a33 - a23 * a32;\n    let inv_det = 1.0 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);\n    return mat4x4<f32>(\n        vec4<f32>(a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09, a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03),\n        vec4<f32>(a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07, a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01),\n        vec4<f32>(a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06, a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00),\n        vec4<f32>(a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06, a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00)) * inv_det;\n}"),
    ("col_mat2x2f", "fn col_mat2x2f(m: mat2x2<f32>, i: u32) -> vec2<f32> { var t = m; return t[min(i, 1u)]; }"),
    ("col_mat2x3f", "fn col_mat2x3f(m: mat2x3<f32>, i: u32) -> vec3<f32> { var t = m; return t[min(i, 1u)]; }"),
    ("col_mat2x4f", "fn col_mat2x4f(m: mat2x4<f32>, i: u32) -> vec4<f32> { var t = m; return t[min(i, 1u)]; }"),
    ("col_mat3x2f", "fn col_mat3x2f(m: mat3x2<f32>, i: u32) -> vec2<f32> { var t = m; return t[min(i, 2u)]; }"),
    ("col_mat3x3f", "fn col_mat3x3f(m: mat3x3<f32>, i: u32) -> vec3<f32> { var t = m; return t[min(i, 2u)]; }"),
    ("col_mat3x4f", "fn col_mat3x4f(m: mat3x4<f32>, i: u32) -> vec4<f32> { var t = m; return t[min(i, 2u)]; }"),
    ("col_mat4x2f", "fn col_mat4x2f(m: mat4x2<f32>, i: u32) -> vec2<f32> { var t = m; return t[min(i, 3u)]; }"),
    ("col_mat4x3f", "fn col_mat4x3f(m: mat4x3<f32>, i: u32) -> vec3<f32> { var t = m; return t[min(i, 3u)]; }"),
    ("col_mat4x4f", "fn col_mat4x4f(m: mat4x4<f32>, i: u32) -> vec4<f32> { var t = m; return t[min(i, 3u)]; }"),
    ("outer_2_2", "fn outer_2_2(a: vec2<f32>, b: vec2<f32>) -> mat2x2<f32> { return mat2x2<f32>(a * b.x, a * b.y); }"),
    ("outer_2_3", "fn outer_2_3(a: vec2<f32>, b: vec3<f32>) -> mat3x2<f32> { return mat3x2<f32>(a * b.x, a * b.y, a * b.z); }"),
    ("outer_2_4", "fn outer_2_4(a: vec2<f32>, b: vec4<f32>) -> mat4x2<f32> { return mat4x2<f32>(a * b.x, a * b.y, a * b.z, a * b.w); }"),
    ("outer_3_2", "fn outer_3_2(a: vec3<f32>, b: vec2<f32>) -> mat2x3<f32> { return mat2x3<f32>(a * b.x, a * b.y); }"),
    ("outer_3_3", "fn outer_3_3(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> { return mat3x3<f32>(a * b.x, a * b.y, a * b.z); }"),
    ("outer_3_4", "fn outer_3_4(a: vec3<f32>, b: vec4<f32>) -> mat4x3<f32> { return mat4x3<f32>(a * b.x, a * b.y, a * b.z, a * b.w); }"),
    ("outer_4_2", "fn outer_4_2(a: vec4<f32>, b: vec2<f32>) -> mat2x4<f32> { return mat2x4<f32>(a * b.x, a * b.y); }"),
    ("outer_4_3", "fn outer_4_3(a: vec4<f32>, b: vec3<f32>) -> mat3x4<f32> { return mat3x4<f32>(a * b.x, a * b.y, a * b.z); }"),
    ("outer_4_4", "fn outer_4_4(a: vec4<f32>, b: vec4<f32>) -> mat4x4<f32> { return mat4x4<f32>(a * b.x, a * b.y, a * b.z, a * b.w); }"),
    ("fx_add_sat", "fn fx_add_sat(a: i32, b: i32) -> i32 {\n    let s = a + b;\n    if (a >= 0 && b >= 0 && s < 0) { return 2147483647; }\n    if (a < 0 && b < 0 && s >= 0) { return -2147483648; }\n    return s;\n}"),
    ("fx_sub_sat", "fn fx_sub_sat(a: i32, b: i32) -> i32 {\n    let s = a - b;\n    if (a >= 0 && b < 0 && s < 0) { return 2147483647; }\n    if (a < 0 && b >= 0 && s >= 0) { return -2147483648; }\n    return s;\n}"),
    ("fx_wide_mul", "fn fx_wide_mul(a: i32, b: i32) -> vec2<i32> {\n    // The 64-bit product as (lo, hi) limbs: unsigned 16-bit limb products, then the sign corrections.\n    let ua = u32(a); let ub = u32(b);\n    let al = ua & 65535u; let ah = ua >> 16u; let bl = ub & 65535u; let bh = ub >> 16u;\n    let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;\n    let mid = (ll >> 16u) + (lh & 65535u) + (hl & 65535u);\n    let lo = (ll & 65535u) | ((mid & 65535u) << 16u);\n    var hi = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);\n    if (a < 0) { hi = hi - ub; }\n    if (b < 0) { hi = hi - ua; }\n    return vec2<i32>(i32(lo), i32(hi));\n}"),
    ("fx_mul", "fn fx_mul(a: i32, b: i32) -> i32 {\n    // (a * b) >> 16 with rounding, from the 64-bit product's limbs.\n    let w = fx_wide_mul(a, b);\n    let lo = u32(w.x);\n    let hi = u32(w.y);\n    let r = (lo >> 16u) | (hi << 16u);\n    let round = (lo >> 15u) & 1u;\n    return i32(r + round);\n}"),
    ("fx_shl", "fn fx_shl(a: i32, n: u32) -> i32 {\n    return a << (n & 31u);\n}"),
    ("fx_shr", "fn fx_shr(a: i32, n: u32) -> i32 {\n    return a >> (n & 31u);\n}"),
    ("fx_wide_add_sat", "fn fx_wide_add_sat(a: vec2<i32>, b: vec2<i32>) -> vec2<i32> {\n    let lo = u32(a.x) + u32(b.x);\n    let carry = select(0u, 1u, lo < u32(a.x));\n    let hi = a.y + b.y + i32(carry);\n    if (a.y >= 0 && b.y >= 0 && hi < 0) { return vec2<i32>(-1, 2147483647); }\n    if (a.y < 0 && b.y < 0 && hi >= 0) { return vec2<i32>(0, -2147483648); }\n    return vec2<i32>(i32(lo), hi);\n}"),
    ("fx_narrow", "fn fx_narrow(w: vec2<i32>) -> i32 {\n    return i32((u32(w.x) >> 16u) | (u32(w.y) << 16u));\n}"),
    ("hash_fmix32", "fn hash_fmix32(x: u32) -> u32 {\n    var h = x;\n    h = h ^ (h >> 16u);\n    h = h * 2246822507u;\n    h = h ^ (h >> 13u);\n    h = h * 3266489909u;\n    h = h ^ (h >> 16u);\n    return h;\n}"),
    ("hash_mul64", "fn hash_mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {\n    // (lo, hi) limbs of the low 64 bits of a 64 x 64 product.\n    let al = a.x & 65535u; let ah = a.x >> 16u; let bl = b.x & 65535u; let bh = b.x >> 16u;\n    let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;\n    let mid = (ll >> 16u) + (lh & 65535u) + (hl & 65535u);\n    let lo = (ll & 65535u) | ((mid & 65535u) << 16u);\n    let hi = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u) + a.x * b.y + a.y * b.x;\n    return vec2<u32>(lo, hi);\n}"),
    ("hash_mix64", "fn hash_mix64(x: vec2<u32>) -> vec2<u32> {\n    // SplitMix64's finaliser on (lo, hi) limbs.\n    var z = x;\n    z = vec2<u32>(z.x ^ ((z.y << 2u) | (z.x >> 30u)), z.y ^ (z.y >> 30u));\n    z = hash_mul64(z, vec2<u32>(0x1ce4e5b9u, 0xbf58476du));\n    z = vec2<u32>(z.x ^ ((z.y << 5u) | (z.x >> 27u)), z.y ^ (z.y >> 27u));\n    z = hash_mul64(z, vec2<u32>(0x133111ebu, 0x94d049bbu));\n    z = vec2<u32>(z.x ^ ((z.y << 1u) | (z.x >> 31u)), z.y ^ (z.y >> 31u));\n    return z;\n}"),
    ("hash32_to_index", "fn hash32_to_index(h: u32, n: u32) -> u32 {\n    // Two masked draws under the next power of two, then a reflection; never a modulo.\n    if (n <= 1u) { return 0u; }\n    let mask = (1u << (32u - countLeadingZeros(n - 1u))) - 1u;\n    var v = h & mask;\n    if (v >= n) { v = (h >> 16u) & mask; }\n    if (v >= n) { v = n - 1u - (v - n); }\n    return v;\n}"),
];

/// A random typed tree over the kingdom's rows plus the role's: the demand
/// is met by a row whose output is the dual, functions while depth allows,
/// terminals at the limit; a dual with no terminal row gets a `fallback`
/// leaf, as the typed projection would substitute. Deterministic in `seed`.
/// What a subtree the sampler draws must read.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Need {
    Nothing,
    /// A declared input (an output store's value).
    Input,
    /// Anything not a literal (a control row's condition).
    Variable,
    /// A value that varies by thread or data (a condition on the output
    /// store's path).
    Varying,
}

pub fn sample(kingdom: &WgslKingdom, role: &Role, root: Ty, max_depth: usize, seed: u64) -> Result<Tree, String> {
    sample_with(kingdom, role, root, max_depth, seed, None, Shape::Free)
}

/// [`sample`] within a cost bound (2026-10-13: 522 of 1,000 random kernels
/// were over the fitness's static bound, a loop over an expression bound
/// priced at its worst shape): while the kernel's [`cost`] is over `bound`,
/// the top-level statement whose removal lowers the cost most is drawn
/// again in place (its dual and depth), up to 16 times, then left as its
/// fallback; the epilogue's last statement is drawn again at the depth cap,
/// so the kernel keeps its store. Nothing the kingdom allows is forbidden.
pub fn sample_within(kingdom: &WgslKingdom, role: &Role, root: Ty, max_depth: usize, seed: u64, bound: &Cost) -> Result<Tree, String> {
    sample_with(kingdom, role, root, max_depth, seed, Some(bound), Shape::Free)
}

/// THE SKETCH START (docs/PLAN_experiments_2026-10-13.md, E1b): a random
/// safe kernel whose OUTLINE is the one the Gram problem fixes, every other
/// subtree a hole drawn by this sampler under its own rules and `bound`.
/// The outline ([`super::sketch`] recognises it): each thread owns its cell
/// (`safe.cell_of` of `safe.thread_row` and `safe.thread_col`); the item
/// loop `loop.items_boundm` over `bound.m`; in its body an f32 register `R`
/// (one of the kingdom's, drawn) updated as `R = R + <hole>`; the
/// epilogue's last statement the kingdom's guarded read-add-write store
/// `h[cell] = h[cell] + R` (`store.h_O0`). The holes: the grid; zero to two
/// statements before the item loop and zero or one after it; zero or one
/// statement before and after the update in the loop body; zero or one
/// statement before the store; the update's value; the two block factors.
/// A statement hole is drawn at the depth a free draw reaches it; the value
/// hole with the sampler's whole depth (a free draw reaches a value in the
/// item loop at depth 5 of 7, too shallow for a product of two indexed
/// loads) and under the rule a stored value has, that it reads an input.
/// Over `bound`, the hole whose removal lowers the cost most is drawn
/// again, as [`sample_within`] does with statements; the outline is never
/// drawn again. Deterministic in `seed`.
pub fn sample_sketch(kingdom: &WgslKingdom, role: &Role, max_depth: usize, seed: u64, bound: &Cost) -> Result<Tree, String> {
    let root = dual_ty(kingdom, "kernel.safe")?;
    sample_with(kingdom, role, root, max_depth, seed, Some(bound), Shape::Sketch)
}

/// What [`sample_with`] draws: a free tree of the root dual, or the sketch.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Free,
    Sketch,
}

fn sample_with(kingdom: &WgslKingdom, role: &Role, root: Ty, max_depth: usize, seed: u64, bound: Option<&Cost>, shape: Shape) -> Result<Tree, String> {
    // A MODULE (2026-10-17, v24's `ingest` and `search`): one safe kernel per
    // entry point the interface names, each drawn as a kernel of its own (its
    // own grid, body and epilogue) from a seed of its own, in the
    // interface's order; the module row of that many.
    if kingdom.dual(root).is_some_and(|d| d.dual == "kernel.module") {
        let n = role.entries.len();
        let safe = dual_ty(kingdom, "kernel.safe")?;
        let kids = (0..n).map(|i| sample_with(kingdom, &role.at_entry(&role.entries[i]), safe, max_depth, seed ^ ((i as u64) << 48), bound, shape)).collect::<Result<Vec<_>, _>>()?;
        return module_tree(kingdom, kids);
    }
    let f16 =|t: &Ty| kingdom.dual(*t).is_some_and(|d| d.scalar() == Some("f16"));
    // (No return: a return before the epilogue ends the thread before it
    // stores its output, so the kernel writes nothing: 187 of 1,000 random
    // kernels, 2026-10-13. The kingdom keeps the rows for text it reads.)
    // (Nor a mirror: the symmetric-half rule writes it, under its triangle
    // guard, in a kernel that runs with the whole H; 2026-10-14.)
    // (Nor the two rows the reader writes for a test or a bound over a
    // barrier that it cannot see is uniform, uflag.of_flag and
    // uniform.of_checked: the checker proves them, and drawn over a value
    // with uniform rows of its own they read back as those, over a product
    // as no bound at all; the sampler draws the uniform rows themselves.)
    // (Nor, for now, the uniform tests whose bodies keep their rung,
    // stmt.if_ule and its kin and stmt.if_uflag. Drawn, they read back 1,000
    // of 1,000 since 2026-10-17 16:20Z (an item loop in such a body read at
    // the conditional rung, and a continued item loop at a literal bound as
    // a tile loop: both fixed), but a body with no barrier and no item loop
    // is written as a plain if and read back as one, at the conditional
    // rung, and the canonical spelling has no rule for that yet: drawn they
    // cost the AST floor 43 -> 34 and the modules' legal 134 -> 125, as
    // differences of spelling, not of reading. With canonical's rule 9 for
    // them (16:50Z), drawn: random trees AST 35, fixpoint 920; modules AST
    // 2, fixpoint 967, legal 125: below the current floors, so not drawn yet.)
    // (Nor the i32 workgroup arrays and their rows, 2026-10-19, for the same
    // reason as wg.au_c and wg.au_d below.)
    let wg_i32 = |r: &WgslRow| kingdom.dual(r.output).is_some_and(|d| d.dual == "wgmem<i32>.int") || r.inputs.iter().any(|&t| kingdom.dual(t).is_some_and(|d| d.dual == "wgmem<i32>.int"));
    // (Nor the 128-element u32 private arrays, the same reason.)
    let parr128 = |r: &WgslRow| [r.output].iter().chain(r.inputs.iter()).any(|&t| kingdom.dual(t).is_some_and(|d| d.dual == "parr128<u32>.index" || d.dual == "u32.pindex128"));
    let extra_reg = |r: &WgslRow| [r.output].iter().chain(r.inputs.iter()).any(|&t| kingdom.dual(t).is_some_and(|d| EXTRA_REGISTERS.iter().any(|x| x.dual == d.dual)));
    let not_drawn = |r: &WgslRow| extra_reg(r) || is_vector_conversion_read_only(r) || uniform_test_op(safe_base(&r.name)).is_some() || matches!(safe_base(&r.name), "stmt.if_uflag" | "stmt.if_else_uflag") || matches!(r.name.as_str(), "uflag.of_flag" | "uniform.of_checked" | "wg.au_c" | "wg.au_d" | "safe.pidx128") || wg_i32(r) || parr128(r);
    // (Nor the third and fourth atomic<u32> workgroup arrays, wg.au_c and
    // wg.au_d, 2026-10-19: rows for the seeds that keep more counters; drawn,
    // they move the random floors by draws alone, so they wait for a floor
    // run of their own.)
    let mut rows: Vec<WgslRow> = kingdom.rows.iter().filter(|r| !f16(&r.output) && !r.inputs.iter().any(f16) && !r.name.starts_with("stmt.return") && !r.name.starts_with("safe.mirror_of") && !r.name.starts_with("store.h_mirror") && !not_drawn(r) && !r.name.starts_with("wg.set")).cloned().collect();
    // The safe kingdom reads h only at the thread's own cell (load.h_safe):
    // the role's load.h, h at any index, is the plain kingdom's (the
    // legality checker: 25 of 1,000 random safe genes read another thread's
    // cell through it). The reader's per-kernel rows leave it out the same way.
    let safe = kingdom.dual(root).is_some_and(|d| d.dual == "kernel.safe");
    rows.extend(role.rows(kingdom)?.into_iter().filter(|r| !(safe && r.name == "load.h")));
    // An interface without the Gram's input, output cell and fields (v24's,
    // 2026-10-17): the rows that spell them (phi, h, a thread's cell, the
    // bounds and uniform values of cfg.m, cfg.d, cfg.r_rows, cfg.r_off) have
    // nothing to render and are not drawn; its own bindings' rows are.
    rows.retain(|r| role_spells(kingdom, role, r));
    let mut by_out: BTreeMap<Ty, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        let e = by_out.entry(r.output).or_default();
        if r.terminal {
            e.1.push(i);
        } else {
            e.0.push(i);
        }
    }
    // THE INPUT AND THE OUTPUT (2026-10-13: 5 of 1,000 random kernels stored
    // a value that read an input; 524 ended in the fallback store of f0,
    // never written). An input read is a row that indexes a read-only
    // storage binding of the role (its per-kernel load, or the kingdom's
    // `load.<binding>` rows); an output store a row that stores to the role's
    // output. For each dual, the least depth to each.
    let inputs: Vec<String> = role.bindings.iter().filter(|b| b.space.starts_with("storage") && !b.space.contains("write")).map(|b| b.name.clone()).collect();
    let reads_input = |r: &WgslRow| -> bool {
        !r.inputs.is_empty()
            && (inputs.iter().any(|b| r.name.starts_with(&format!("load.{b}")) && !r.name.ends_with("_len"))
                || role.per_kernel.iter().any(|p| p.name == r.name && inputs.iter().any(|b| p.template.contains(&format!("{b}[")))))
    };
    // (The safe kingdom's own stores, store.h_*, write the role's output
    // whatever its name: under the cosine interface no row reached a store,
    // and an epilogue at the depth cap fell back to the plain cell's.)
    let stores_output = |r: &WgslRow| (r.name.starts_with(&format!("store.{}", role.output)) || safe_base(&r.name) == "store.h_safe") && !r.inputs.is_empty();
    let reach = |base: &dyn Fn(&WgslRow) -> bool| -> BTreeMap<Ty, usize> {
        let mut d: BTreeMap<Ty, usize> = BTreeMap::new();
        loop {
            let mut changed = false;
            for r in &rows {
                let via = if base(r) { Some(1) } else { r.inputs.iter().filter_map(|t| d.get(t)).min().map(|x| x + 1) };
                if let Some(v) = via {
                    if d.get(&r.output).is_none_or(|&old| v < old) {
                        d.insert(r.output, v);
                        changed = true;
                    }
                }
            }
            if !changed {
                return d;
            }
        }
    };
    // A variable: a terminal that is not a literal (a builtin, a load, a
    // register or array read, a counter): what a control condition reads.
    // (Not a counter: outside its loop the emitter writes it 0, and a
    // condition over counters alone was a constant in the text; 392 of
    // 2,398 conditions, 2026-10-13.)
    let reads_variable = |r: &WgslRow| r.terminal && !r.name.starts_with("literal.") && !matches!(r.name.as_str(), "safe.counter" | "safe.tile_counter" | "safe.tile_counter2" | "safe.tile_counter3" | "safe.tile_counter4" | "builtin.item" | "builtin.tile");
    // A value that varies by thread or data: a variable that is not the
    // same in every thread (a uniform field, a buffer's length, the
    // workgroup count, the subgroup size): what a condition on the output
    // store's path reads (2026-10-13: 144 of 1,000 kernels had every store
    // under a condition of such values alone, fixed for a whole dispatch).
    let reads_varying = |r: &WgslRow| reads_variable(r) && !(r.name.starts_with("load.cfg") || r.name.ends_with("_len") || r.name.starts_with("uniform.") || r.name.starts_with("bound.") || matches!(r.name.as_str(), "builtin.num_workgroups" | "builtin.subgroup_size" | "builtin.num_subgroups"));
    let input_reach = reach(&reads_input);
    let store_reach = reach(&stores_output);
    let var_reach = reach(&reads_variable);
    let varying_reach = reach(&reads_varying);
    let flags: Vec<(bool, bool, bool, bool)> = rows.iter().map(|r| (reads_input(r), stores_output(r), reads_variable(r), reads_varying(r))).collect();
    let mut state = seed ^ 0x5DEE_CE66_D1CE_4E5B;
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    struct Draw<'a> {
        rows: &'a [WgslRow],
        by_out: &'a BTreeMap<Ty, (Vec<usize>, Vec<usize>)>,
        kingdom: &'a WgslKingdom,
        max_depth: usize,
        input_reach: &'a BTreeMap<Ty, usize>,
        store_reach: &'a BTreeMap<Ty, usize>,
        var_reach: &'a BTreeMap<Ty, usize>,
        varying_reach: &'a BTreeMap<Ty, usize>,
        flags: &'a [(bool, bool, bool, bool)],
        /// THE KERNEL'S CELL (2026-10-14): u32.cell, and the cell row the
        /// kernel's first cell was drawn with (its row kind and column kind:
        /// plain, strided or grouped), which every later cell of the kernel
        /// takes, and the grouped origins' group counts (row, column); the
        /// plain cell's row, which a fallback cell renders. Two store sites
        /// through a block of one kind and of another write one cell from
        /// two threads (126 of 148 write_sets_overlap of 1,000 random
        /// kernels, the checker's layer 7).
        cell_ty: Option<Ty>,
        cell_row: std::cell::Cell<Option<usize>>,
        groups: std::cell::RefCell<[Option<Tree>; 2]>,
        plain_cell: Option<usize>,
    }
    impl Draw<'_> {
        /// Whether row `i` would nest a long loop in one of its kind.
        fn nests(&self, i: usize, inside: u8) -> bool {
            let name = self.rows[i].name.as_str();
            (inside & 1 != 0 && matches!(name, "loop.items" | "loop.items_unrolled")) || (inside & 2 != 0 && name == "loop.tile")
        }

        /// The depth a comparison's second operand is drawn again at: its
        /// own, then (after four draws equal to the first, a leaf at the cap
        /// with no other: a fallback, the workgroup count) two levels up, so
        /// it can be an expression.
        fn redraw_depth(&self, depth: usize, attempt: usize) -> usize {
            if attempt < 4 {
                depth + 1
            } else {
                depth.min(self.max_depth).saturating_sub(2)
            }
        }

        /// What kid `k` of row `i` must read on a required path: the path's
        /// own need where it goes on through the kid, else the kid's own.
        fn path_need(&self, which: usize, on_path: bool, i: usize, k: usize) -> Need {
            match (which, on_path) {
                (0, true) => Need::Input,
                (2, true) => Need::Variable,
                (3, true) => Need::Varying,
                _ => self.kid_need(i, k),
            }
        }

        /// What kid `k` of row `i` must read: an output store's value an
        /// input; a control row's condition (an if's, a continue's, a
        /// break's) a variable, so no literal decides it (2026-10-13: 93 of
        /// 1,000 random kernels had every store under a literal condition
        /// that never held; 1,931 of 2,381 conditions were constants).
        fn kid_need(&self, i: usize, k: usize) -> Need {
            let r = &self.rows[i];
            if self.flags[i].1 && k + 1 == r.inputs.len() {
                return Need::Input;
            }
            let condition = if r.name.starts_with("stmt.if") || r.name.starts_with("block.cont") { 0 } else if r.name.starts_with("block.brk") { 1 } else { usize::MAX };
            if k == condition {
                // On the output store's path (a row of the epilogue's): a
                // value that varies by thread or data.
                let epilogue = self.kingdom.dual(r.output).is_some_and(|d| (d.slot == "stmt" || d.slot == "block") && d.form.starts_with('O'));
                return if epilogue { Need::Varying } else { Need::Variable };
            }
            Need::Nothing
        }

        /// The long loops kid `k` of row `i` is inside: bit 1 an item loop,
        /// bit 2 a tile loop (the plain kingdom's rows, which the renderer
        /// does not nest); bit 4 control that may not be uniform (2026-10-14:
        /// a statement or block at a conditional, break or epilogue rung, a
        /// store's value under its own bounds check), where no subgroup
        /// operation is drawn (WGSL's subgroup uniformity; 320 of 1,000
        /// random kernels drew one there, the checker's layer 7).
        fn kid_inside(&self, i: usize, k: usize, inside: u8) -> u8 {
            let r = &self.rows[i];
            let loops = match r.name.as_str() {
                "loop.items" | "loop.items_unrolled" => inside | 1,
                "loop.tile" => inside | 2,
                _ => inside,
            };
            let conditional = |t: Ty| self.kingdom.dual(t).is_some_and(|d| (d.slot == "stmt" || d.slot == "block") && (d.form.starts_with('C') || d.form.starts_with('B') || d.form.starts_with('O')));
            // (A continue's or a break's guard: its condition decides an exit
            // some threads may take, and is read again on the next pass.)
            let store = matches!(safe_base(&r.name), "store.h_safe" | "store.h_mirror" | "wg.put") || r.name.starts_with("block.cont") || r.name.starts_with("block.brk");
            // (A loop's bound or step: evaluated again on every pass, its value
            // deciding when the loop ends; a subgroup result there is the same
            // in a subgroup, not in the workgroup.)
            let header = r.name.starts_with("loop.") && r.inputs.get(k).copied().is_some_and(|t| self.kingdom.dual(t).is_some_and(|d| d.slot != "stmt" && d.slot != "block"));
            let loops = if conditional(r.output) || store || header || r.inputs.get(k).copied().is_some_and(conditional) { loops | 4 } else { loops };
            // Bit 8: a cell's offset within the thread's block, drawn a
            // literal (an offset any expression gives, clamped into the
            // block, may equal another store's in the same thread, which
            // then writes one cell twice; it reads memory another thread may
            // change between the cell's guard and its store: 22 of 27
            // write_sets_overlap of 1,000 random kernels, 2026-10-14).
            let offset = match safe_base(&r.name) {
                "safe.row_plus_x" | "safe.col_plus_x" | "safe.row_plus_x_s" | "safe.col_plus_x_s" => k == 1,
                "safe.row_plus_g" | "safe.col_plus_g" => k >= 2,
                _ => false,
            };
            let loops = if offset { loops | 8 } else { loops };
            // Bit 16: inside a cell, where no offset is a tile counter (a
            // store through r + k in a tile loop and one through r elsewhere
            // write one cell at k = 0: 4 of 1,000 random kernels). Bit 32: a
            // store's value, the one place a read of h is drawn (under the
            // store's own bounds check, at the store's own cell: unguarded,
            // a read at a column past d is another row's, another thread's
            // cell: 17 of 1,000 random kernels).
            let loops = if Some(r.output) == self.cell_ty { loops | 16 } else { loops };
            if safe_base(&r.name) == "store.h_safe" && k == 1 { loops | 32 } else { loops }
        }

        /// Whether row `i` may be drawn where control may not be uniform
        /// (`inside` bit 4): not a subgroup operation; at a cell's offset
        /// (bit 8): a literal only; in a cell (bit 16): no tile counter's
        /// offset; outside a store's value (bit 32): no read of h.
        fn allowed(&self, i: usize, inside: u8) -> bool {
            let name = self.rows[i].name.as_str();
            (inside & 4 == 0 || !name.starts_with("subgroup.")) && (inside & 8 == 0 || name.starts_with("literal.")) && (inside & 16 == 0 || !name.contains("_plus_tile")) && (inside & 32 != 0 || !name.starts_with("load.h_safe"))
        }

        /// A store's value with every read of h at the store's own cell.
        fn reads_at(t: &mut Tree, cell: &Tree) {
            if t.name.starts_with("load.h_safe") && t.kids.len() == 1 {
                t.kids[0] = cell.clone();
            }
            t.kids.iter_mut().for_each(|k| Self::reads_at(k, cell));
        }

        /// Whether `kids` of row `i` are a put whose value reads the array
        /// it writes: between the put's barriers a thread would read an
        /// element another thread writes (2 of 1,000 random kernels, seeds
        /// 79 and 545, 2026-10-14).
        fn put_reads_its_array(&self, i: usize, kids: &[Tree]) -> bool {
            fn holds(t: &Tree, a: &Tree) -> bool {
                t == a || t.kids.iter().any(|k| holds(k, a))
            }
            safe_base(&self.rows[i].name) == "wg.put" && kids.len() == 3 && holds(&kids[2], &kids[0])
        }

        /// Whether kid `k` of row `i` is drawn under the no-continue rule: a
        /// conditional block of an L-rung statement, and anything under one
        /// up to a nested loop (whose continues are its own).
        fn kid_no_cont(&self, i: usize, k: usize, no_cont: bool) -> bool {
            let r = &self.rows[i];
            if r.name.starts_with("loop.") {
                return false;
            }
            let form = |t: Ty| self.kingdom.dual(t).map(|d| (d.slot.clone(), d.form.clone()));
            // (Any block a statement holds at an L rung: an if's conditional
            // block, and a block statement's own, whose continue would skip
            // the statements after it, a barrier among them: the legality
            // checker's layer 4. A continue is then a loop body's own.)
            let opens = matches!(form(r.output), Some((slot, f)) if slot == "stmt" && f.starts_with('L')) && matches!(r.inputs.get(k).copied().and_then(form), Some((slot, _)) if slot == "block");
            no_cont || opens
        }

        /// The least depth from row `i` to what `reach` measures (`own`: the
        /// row is it itself).
        fn row_reach(&self, i: usize, reach: &BTreeMap<Ty, usize>, own: bool) -> Option<usize> {
            if own {
                return Some(1);
            }
            self.rows[i].inputs.iter().filter_map(|t| reach.get(t)).min().map(|x| x + 1)
        }
        /// A tree of `demand` under the kernel's cell ([`Draw::cell_row`]):
        /// a cell is drawn with the kernel's cell row once there is one (at
        /// the depth cap too, its kids then leaves), the first fixes it; a
        /// grouped origin takes the kernel's group count on its axis, the
        /// first fixes it (a fallback origin is the strided block, every
        /// cell a group: the most groups).
        fn grow(&self, demand: Ty, depth: usize, need: Need, no_cont: bool, inside: u8, state: &mut u64) -> Tree {
            if self.cell_ty == Some(demand) {
                if let Some(i) = self.cell_row.get() {
                    let r = &self.rows[i];
                    let kids = r.inputs.iter().enumerate().map(|(k, &t)| self.grow(t, depth + 1, self.kid_need(i, k), self.kid_no_cont(i, k, no_cont), self.kid_inside(i, k, inside), state)).collect();
                    return Tree::new(&r.name, demand, kids);
                }
                let t = self.grow_free(demand, depth, need, no_cont, inside, state);
                let row = self.rows.iter().position(|r| r.name == t.name && r.output == demand && r.inputs.len() == t.kids.len() && r.inputs.iter().zip(&t.kids).all(|(a, k)| *a == k.ty));
                self.cell_row.set(row.or(self.plain_cell));
                return t;
            }
            let mut t = self.grow_free(demand, depth, need, no_cont, inside, state);
            // An epilogue statement's or block's fallback is the store of the
            // plain cell: the kernel's cell, or the store's path drawn instead.
            let epilogue = self.kingdom.dual(demand).is_some_and(|d| (d.slot == "stmt" || d.slot == "block") && d.form.starts_with('O'));
            if t.name == "fallback" && epilogue && self.cell_ty.is_some() {
                match self.cell_row.get() {
                    None => self.cell_row.set(self.plain_cell),
                    Some(c) if Some(c) != self.plain_cell => t = self.grow_free(demand, depth.max(self.max_depth), need, no_cont, inside, state),
                    _ => {}
                }
            }
            let axis = match self.kingdom.dual(demand).map(|d| d.dual.as_str()) {
                Some("u32.threadrowg" | "u32.rowbaseg") => 0,
                Some("u32.threadcolg" | "u32.colbaseg") => 1,
                _ => return t,
            };
            // (A fallback origin under a group count already drawn: the
            // grouped origin itself, its kids leaves at the cap.)
            let origin = if axis == 0 { "safe.thread_row_g" } else { "safe.thread_col_g" };
            if t.name == "fallback" && self.groups.borrow()[axis].is_some() {
                if let Some(i) = self.rows.iter().position(|r| r.name == origin && r.output == demand) {
                    let kids = self.rows[i].inputs.iter().map(|&k| self.grow_free(k, depth + 1, Need::Nothing, no_cont, inside, state)).collect();
                    t = Tree::new(origin, demand, kids);
                }
            }
            let mut groups = self.groups.borrow_mut();
            if safe_base(&t.name) == origin && t.kids.len() == 4 {
                match &groups[axis] {
                    Some(g) => t.kids[3] = g.clone(),
                    None => groups[axis] = Some(t.kids[3].clone()),
                }
            } else if t.name == "fallback" && groups[axis].is_none() {
                // (The strided block a fallback origin renders: every cell a group.)
                let lit = self.rows.iter().find(|r| r.name == origin).and_then(|r| r.inputs.get(3).copied());
                groups[axis] = lit.map(|ty| Tree::literal("literal.lit64", ty, 64.0));
            }
            drop(groups);
            t
        }

        /// [`Draw::grow`] without the kernel's cell. `need_input`: it must
        /// read an input (the value an output store writes). An epilogue
        /// block or statement at the depth cap draws its way to an output
        /// store, not its fallback.
        fn grow_free(&self, demand: Ty, depth: usize, need: Need, no_cont: bool, inside: u8, state: &mut u64) -> Tree {
            let Some((functions_all, terminals)) = self.by_out.get(&demand) else { return Tree::fallback(demand) };
            // A continue inside a conditional block of a loop body's statement
            // (an if at an L rung) would skip the rest of the body in some
            // threads, a barrier after it among them: not drawn there
            // (2026-10-13, the legality checker's layer 4).
            // No long loop inside a long loop of its kind (the plain
            // kingdom's grammar admits an item loop in an item loop and a
            // tile loop in a tile loop; the renderer refuses them, m² or
            // more iterations: 60 of 500 random plain kernels, 2026-10-13).
            let functions_kept: Vec<usize> = functions_all.iter().copied().filter(|&i| !((no_cont && self.rows[i].name.starts_with("block.cont")) || self.nests(i, inside)) && self.allowed(i, inside)).collect();
            let functions = &functions_kept;
            let terminals_kept: Vec<usize> = terminals.iter().copied().filter(|&i| self.allowed(i, inside)).collect();
            let terminals = &terminals_kept;
            let epilogue = self.kingdom.dual(demand).is_some_and(|d| (d.slot == "stmt" || d.slot == "block") && d.form.starts_with('O'));
            // A required path (to an input, or at the cap to an output store):
            // the rows on it, the nearest at or past the cap.
            let required = if need == Need::Input && self.input_reach.contains_key(&demand) {
                Some((self.input_reach, 0usize))
            } else if need == Need::Variable && self.var_reach.contains_key(&demand) {
                Some((self.var_reach, 2usize))
            } else if need == Need::Varying && self.varying_reach.contains_key(&demand) {
                Some((self.varying_reach, 3usize))
            } else if epilogue && depth >= self.max_depth && self.store_reach.contains_key(&demand) {
                Some((self.store_reach, 1usize))
            } else {
                None
            };
            let own_of = |i: usize, which: usize| match which {
                0 => self.flags[i].0,
                1 => self.flags[i].1,
                2 => self.flags[i].2,
                _ => self.flags[i].3,
            };
            if let Some((reach, which)) = required {
                let all: Vec<usize> = functions.iter().chain(terminals.iter()).copied().collect();
                let scored: Vec<(usize, usize)> = all.iter().filter(|&&i| !((no_cont && self.rows[i].name.starts_with("block.cont")) || self.nests(i, inside)) && self.allowed(i, inside)).filter_map(|&i| self.row_reach(i, reach, own_of(i, which)).map(|d| (i, d))).collect();
                // The shortest way there (a load of the input, a block holding
                // the store): the path adds no structure of its own; every
                // other child is drawn as any is.
                let best = scored.iter().map(|(_, d)| *d).min().unwrap_or(1);
                let pool: Vec<usize> = scored.iter().filter(|(_, d)| *d == best).map(|(i, _)| *i).collect();
                let i = pool[(next(state) % pool.len() as u64) as usize];
                let r = &self.rows[i];
                let own = own_of(i, which);
                // The path goes on through one child that reaches it (none
                // when the row is it: its children are drawn freely, an
                // output store's value then reading an input).
                let carriers: Vec<usize> = if own { Vec::new() } else { r.inputs.iter().enumerate().filter(|(_, t)| reach.contains_key(t)).map(|(k, _)| k).collect() };
                let carry = (!carriers.is_empty()).then(|| carriers[(next(state) % carriers.len() as u64) as usize]);
                let kids = r
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(k, &t)| {
                        let (nc, ins) = (self.kid_no_cont(i, k, no_cont), self.kid_inside(i, k, inside));
                        self.grow(t, depth + 1, self.path_need(which, carry == Some(k), i, k), nc, ins, state)
                    })
                    .collect::<Vec<Tree>>();
                let mut kids = kids;
                if r.name.starts_with("compare.") && kids.len() == 2 {
                    for attempt in 0..8 {
                        if kids[0] != kids[1] {
                            break;
                        }
                        kids[1] = self.grow(r.inputs[1], self.redraw_depth(depth, attempt), self.path_need(which, carry == Some(1), i, 1), self.kid_no_cont(i, 1, no_cont), self.kid_inside(i, 1, inside), state);
                    }
                    // Operands of a dual with one value only: the node drawn again.
                    if kids[0] == kids[1] {
                        return self.grow(demand, depth, need, no_cont, inside, state);
                    }
                }
                // A put's value drawn again until it reads another array (or
                // none), then the node.
                for attempt in 0..8 {
                    if !self.put_reads_its_array(i, &kids) {
                        break;
                    }
                    kids[2] = self.grow(r.inputs[2], self.redraw_depth(depth, attempt), self.path_need(which, carry == Some(2), i, 2), self.kid_no_cont(i, 2, no_cont), self.kid_inside(i, 2, inside), state);
                }
                if self.put_reads_its_array(i, &kids) {
                    return self.grow(demand, depth, need, no_cont, inside, state);
                }
                if safe_base(&r.name) == "store.h_safe" && kids.len() == 2 {
                    let cell = kids[0].clone();
                    Self::reads_at(&mut kids[1], &cell);
                }
                return Tree::new(&r.name, demand, kids);
            }
            let pick_terminal = depth >= self.max_depth || functions.is_empty() || (!terminals.is_empty() && next(state).is_multiple_of(3));
            if pick_terminal {
                if terminals.is_empty() {
                    return Tree::fallback(demand);
                }
                let r = &self.rows[terminals[(next(state) % terminals.len() as u64) as usize]];
                if r.name.starts_with("literal.") {
                    let form = self.kingdom.dual(demand).map(|d| d.form.as_str()).unwrap_or("");
                    let u = (next(state) >> 11) as f64 / (1u64 << 53) as f64;
                    let konst = match form {
                        "real" => ((u * 8.0) - 4.0) as f32,
                        "flag" | "sign" => f32::from(u8::from(u >= 0.5)),
                        "int" | "q15_16" => ((u * 200.0) - 100.0).round() as f32,
                        _ => (u * 64.0).floor() as f32,
                    };
                    return Tree::literal(&r.name, demand, konst);
                }
                return Tree::new(&r.name, demand, Vec::new());
            }
            let i = functions[(next(state) % functions.len() as u64) as usize];
            let r = &self.rows[i];
            // An output store's value reads an input, a condition a
            // variable, wherever they are drawn.
            let mut kids: Vec<Tree> = r.inputs.iter().enumerate().map(|(k, &t)| self.grow(t, depth + 1, self.kid_need(i, k), self.kid_no_cont(i, k, no_cont), self.kid_inside(i, k, inside), state)).collect();
            // A comparison of a value with itself decides nothing (`nwg <=
            // nwg`: 111 of 1,560 conditions, 2026-10-13): its second operand
            // drawn again, a few times.
            if r.name.starts_with("compare.") && kids.len() == 2 {
                for attempt in 0..8 {
                    if kids[0] != kids[1] {
                        break;
                    }
                    kids[1] = self.grow(r.inputs[1], self.redraw_depth(depth, attempt), self.kid_need(i, 1), self.kid_no_cont(i, 1, no_cont), self.kid_inside(i, 1, inside), state);
                }
                // Operands of a dual with one value only (the workgroup
                // count): the node drawn again, another row.
                if kids[0] == kids[1] {
                    return self.grow(demand, depth, need, no_cont, inside, state);
                }
            }
            for attempt in 0..8 {
                if !self.put_reads_its_array(i, &kids) {
                    break;
                }
                kids[2] = self.grow(r.inputs[2], self.redraw_depth(depth, attempt), self.kid_need(i, 2), self.kid_no_cont(i, 2, no_cont), self.kid_inside(i, 2, inside), state);
            }
            if self.put_reads_its_array(i, &kids) {
                return self.grow(demand, depth, need, no_cont, inside, state);
            }
            if safe_base(&r.name) == "store.h_safe" && kids.len() == 2 {
                let cell = kids[0].clone();
                Self::reads_at(&mut kids[1], &cell);
            }
            Tree::new(&r.name, demand, kids)
        }
    }
    let cell_ty = dual_ty(kingdom, "u32.cell").ok();
    let plain_cell = cell_ty.and_then(|c| {
        let (row, col) = (dual_ty(kingdom, "u32.threadrow").ok()?, dual_ty(kingdom, "u32.threadcol").ok()?);
        rows.iter().position(|r| r.name == "safe.cell_of" && r.output == c && r.inputs == [row, col])
    });
    let draw = Draw { rows: &rows, by_out: &by_out, kingdom, max_depth, input_reach: &input_reach, store_reach: &store_reach, var_reach: &var_reach, varying_reach: &varying_reach, flags: &flags, cell_ty, cell_row: std::cell::Cell::new(None), groups: std::cell::RefCell::new([None, None]), plain_cell };
    /// A hole of the sketch: where it is, the depth it is drawn at and what it must read.
    struct Hole {
        path: Vec<usize>,
        depth: usize,
        need: Need,
    }
    /// The sketch's outline with its holes drawn ([`sample_sketch`]).
    fn sketch(draw: &Draw<'_>, kingdom: &WgslKingdom, state: &mut u64) -> Result<(Tree, Vec<Hole>), String> {
        let t = |n: &str| dual_ty(kingdom, n);
        let (l0, l1, c0, real, rf) = (t("stmt.L0")?, t("stmt.L1")?, t("stmt.C0")?, t("f32.real")?, t("reg<f32>.real")?);
        const SEQ: [&str; 4] = ["one", "seq2", "seq3", "seq4"];
        let block = |rung: &str, kids: Vec<Tree>| -> Result<Tree, String> { Ok(Tree::new(&format!("block.{}_{rung}", SEQ[kids.len() - 1]), t(&format!("block.{rung}"))?, kids)) };
        // The outline's store is through the plain block: every cell a hole
        // draws is too (the kernel's cell).
        draw.cell_row.set(draw.plain_cell);
        let mut holes: Vec<Hole> = Vec::new();
        let mut hole = |demand: Ty, path: Vec<usize>, depth: usize, need: Need, state: &mut u64| -> Tree {
            holes.push(Hole { path, depth, need });
            draw.grow(demand, depth, need, false, 0, state)
        };
        let grid = hole(t("grid.grid")?, vec![0], 1, Need::Nothing, state);
        // The accumulator: one of the kingdom's f32 registers.
        let registers: Vec<&str> = kingdom.rows.iter().filter(|r| r.terminal && r.output == rf).map(|r| r.name.as_str()).collect();
        if registers.is_empty() {
            return Err("the kingdom has no f32 register for the sketch's accumulator".into());
        }
        let acc = registers[(next(state) % registers.len() as u64) as usize].to_string();
        let get = || Tree::new("reg.get_f", real, vec![Tree::new(&acc, rf, vec![])]);
        let (pre, post) = ((next(state) % 3) as usize, (next(state) % 2) as usize);
        let (lpre, lpost) = ((next(state) % 2) as usize, (next(state) % 2) as usize);
        let mut body: Vec<Tree> = (0..pre).map(|k| hole(l0, vec![1, k], 2, Need::Nothing, state)).collect();
        let mut inner: Vec<Tree> = (0..lpre).map(|k| hole(l1, vec![1, pre, 1, k], 4, Need::Nothing, state)).collect();
        let value = hole(real, vec![1, pre, 1, lpre, 1, 1], 0, Need::Input, state);
        inner.push(Tree::new("reg.set_f_L1", l1, vec![Tree::new(&acc, rf, vec![]), Tree::new("arith.add", real, vec![get(), value])]));
        for k in 0..lpost {
            inner.push(hole(l1, vec![1, pre, 1, lpre + 1 + k], 4, Need::Nothing, state));
        }
        body.push(Tree::new("loop.items_boundm", l0, vec![Tree::new("bound.m", t("u32.boundm")?, vec![]), block("L1", inner)?]));
        for k in 0..post {
            body.push(hole(l0, vec![1, pre + 1 + k], 2, Need::Nothing, state));
        }
        // The epilogue: a C0 statement before the store or none (block.seq2_O0 is (C0, O0)).
        let epre = (next(state) % 2) as usize;
        let mut epilogue: Vec<Tree> = (0..epre).map(|k| hole(c0, vec![2, k], 2, Need::Nothing, state)).collect();
        let gid = || -> Result<Tree, String> { Ok(Tree::new("builtin.global_invocation_id", t("vec3<u32>.gid")?, vec![])) };
        let blk_row = hole(t("u32.blk")?, vec![2, epre, 0, 0, 1], 5, Need::Nothing, state);
        let blk_col = hole(t("u32.blk")?, vec![2, epre, 0, 1, 1], 5, Need::Nothing, state);
        let cell = Tree::new("safe.cell_of", t("u32.cell")?, vec![Tree::new("safe.thread_row", t("u32.threadrow")?, vec![gid()?, blk_row]), Tree::new("safe.thread_col", t("u32.threadcol")?, vec![gid()?, blk_col])]);
        epilogue.push(Tree::new("store.h_O0", t("stmt.O0")?, vec![cell, get()]));
        let epilogue = block("O0", epilogue)?;
        let body = block("L0", body)?;
        Ok((Tree::new("kernel.entry_safe", t("kernel.safe")?, vec![grid, body, epilogue]), holes))
    }
    let (mut tree, holes) = match shape {
        Shape::Free => (draw.grow(root, 0, Need::Nothing, false, 0, &mut state), None),
        Shape::Sketch => {
            let (t, h) = sketch(&draw, kingdom, &mut state)?;
            (t, Some(h))
        }
    };
    let Some(bound) = bound else { return Ok(tree) };
    let over = |c: &Cost| c.per_item > bound.per_item || c.base > bound.base;
    // The top-level statements of the body (kid 1) and the epilogue (kid 2),
    // nested plain blocks opened, by path; whether each is the epilogue's last.
    fn statements(b: &Tree, path: Vec<usize>, out: &mut Vec<Vec<usize>>) {
        for (i, s) in b.kids.iter().enumerate() {
            let mut p = path.clone();
            p.push(i);
            let plain = s.kids.len() == 1 && s.name.starts_with("stmt.block_") && ["block.one_", "block.seq2_", "block.seq3_", "block.seq4_"].iter().any(|x| s.kids[0].name.starts_with(x));
            if plain {
                p.push(0);
                statements(&s.kids[0], p, out);
            } else {
                out.push(p);
            }
        }
    }
    fn at<'t>(t: &'t Tree, path: &[usize]) -> &'t Tree {
        path.iter().fold(t, |n, &i| &n.kids[i])
    }
    fn put(t: &Tree, path: &[usize], with: Tree) -> Tree {
        match path.split_first() {
            None => with,
            Some((&i, rest)) => {
                let mut kids = t.kids.clone();
                kids[i] = put(&kids[i], rest, with);
                Tree { kids, ..t.clone() }
            }
        }
    }
    let mut tries: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
    for _ in 0..256 {
        let c = cost(&tree);
        if !over(&c) || tree.kids.len() != 3 {
            return Ok(tree);
        }
        // The sites drawn again: the sketch's holes (each at its own depth,
        // under its own need), or the top-level statements (at their depth).
        // A site: its path, the depth it is drawn at, what it must read.
        type Site = (Vec<usize>, usize, Need);
        let (paths, last): (Vec<Site>, Option<Vec<usize>>) = match &holes {
            Some(h) => (h.iter().map(|h| (h.path.clone(), h.depth, h.need)).collect(), None),
            None => {
                let mut paths = Vec::new();
                statements(&tree.kids[1], vec![1], &mut paths);
                let mut epi = Vec::new();
                statements(&tree.kids[2], vec![2], &mut epi);
                let last = epi.last().cloned();
                paths.extend(epi);
                (paths.into_iter().map(|p| { let d = p.len(); (p, d, Need::Nothing) }).collect(), last)
            }
        };
        // The statement (the hole) whose removal lowers the cost most.
        let gain = |p: &Vec<usize>| {
            let without = cost(&put(&tree, p, Tree::fallback(at(&tree, p).ty)));
            (c.per_item - without.per_item) / bound.per_item.max(1.0) + (c.base - without.base) / bound.base.max(1.0)
        };
        let Some((worst, depth, need)) = paths.iter().max_by(|a, b| gain(&a.0).partial_cmp(&gain(&b.0)).unwrap_or(std::cmp::Ordering::Equal)).cloned() else { return Ok(tree) };
        let node = at(&tree, &worst).clone();
        let is_last = last.as_ref() == Some(&worst);
        let n = tries.entry(worst.clone()).or_insert(0);
        *n += 1;
        let again = if is_last {
            // The epilogue's store, its shortest way.
            draw.grow(node.ty, max_depth, Need::Nothing, false, 0, &mut state)
        } else if *n > 16 {
            Tree::fallback(node.ty)
        } else {
            draw.grow(node.ty, depth, need, false, 0, &mut state)
        };
        tree = put(&tree, &worst, again);
    }
    Ok(tree)
}

/// Parse text with naga and validate it for the device in use
/// ([`super::target::Target::current`]); the error names the line.
pub fn validate_text(text: &str) -> Result<(), String> {
    let module = crate::wgsl::naga_shape::parse(text)?;
    super::target::Target::current().validate(&module, text).map(|_| ())
}

impl Tree {
    /// A short description for a refusal ledger.
    pub fn head(&self) -> String {
        format!("{} ({} nodes, depth {})", self.name, self.size(), self.depth())
    }
}

/// The first node where two trees differ, as `path: drawn / parsed`
/// (name, type and literal value; a different count of children is a
/// difference at the parent). `None` when they are the same tree.
/// Every node where two trees differ (not below a differing node), as
/// `parent: a / b` keys with the digits of literals kept out; a block of a
/// different length compared statement by statement first.
#[cfg(test)]
pub(crate) fn all_differences(a: &Tree, b: &Tree) -> Vec<String> {
    fn at(a: &Tree, b: &Tree, parent: &str, out: &mut Vec<String>) {
        let same = a.name == b.name && a.ty == b.ty && a.konst.to_bits() == b.konst.to_bits() && a.kids.len() == b.kids.len();
        if same || (a.ty == b.ty && a.name.starts_with("block.") && b.name.starts_with("block.")) {
            let before = out.len();
            for (x, y) in a.kids.iter().zip(&b.kids) {
                at(x, y, &a.name, out);
            }
            if same || out.len() > before {
                return;
            }
        }
        out.push(format!("{parent}: {} / {}", a.name, b.name));
    }
    let mut out = Vec::new();
    at(a, b, "root", &mut out);
    out
}

#[cfg(test)]
pub(crate) fn first_difference(a: &Tree, b: &Tree) -> Option<String> {
    fn at(a: &Tree, b: &Tree, path: &mut Vec<String>) -> Option<String> {
        let same = a.name == b.name && a.ty == b.ty && a.konst.to_bits() == b.konst.to_bits() && a.kids.len() == b.kids.len();
        // Two blocks of a different length: their first different statement,
        // if one is, before the lengths.
        let blocks = !same && a.ty == b.ty && a.name.starts_with("block.") && b.name.starts_with("block.");
        if blocks {
            for (i, (x, y)) in a.kids.iter().zip(&b.kids).enumerate() {
                path.push(format!("{}[{i}]", a.name));
                let d = at(x, y, path);
                path.pop();
                if d.is_some() {
                    return d;
                }
            }
        }
        if !same {
            let show = |t: &Tree| if t.name.starts_with("literal.") { format!("{} {}", t.name, t.konst) } else { format!("{}{}", t.name, if t.kids.is_empty() { String::new() } else { format!("/{}", t.kids.len()) }) };
            let tail: Vec<&String> = path.iter().rev().take(2).collect();
            return Some(format!("{}: {} / {}", tail.into_iter().rev().map(String::as_str).collect::<Vec<_>>().join(" > "), show(a), show(b)));
        }
        for (i, (x, y)) in a.kids.iter().zip(&b.kids).enumerate() {
            path.push(format!("{}[{i}]", a.name));
            let d = at(x, y, path);
            path.pop();
            if d.is_some() {
                return d;
            }
        }
        None
    }
    at(a, b, &mut Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every array name of the families is a terminal row of the family's dual,
    /// and no row past `total` exists: the lists here and `ARRAY_FAMILIES` in
    /// gen_tables.py are one list.
    #[test]
    fn the_array_families_are_the_rows_the_generator_wrote() {
        let k = WgslKingdom::load_safe();
        let dual = |t: Ty| k.dual(t).map(|d| d.dual.clone()).unwrap_or_default();
        for (class, families) in [("wg", &WORKGROUP_ARRAYS[..]), ("parr", &PRIVATE_ARRAYS[..])] {
            for f in families {
                let names: Vec<String> = f.names().collect();
                assert_eq!(names.len(), f.total, "{}", f.dual);
                for n in &names {
                    let row = format!("{class}.{n}");
                    let r = k.rows.iter().find(|r| r.name == row).unwrap_or_else(|| panic!("{row}: no such row"));
                    assert!(r.inputs.is_empty() && dual(r.output) == f.dual, "{row}");
                }
                let rows = k.rows.iter().filter(|r| r.inputs.is_empty() && dual(r.output) == f.dual).count();
                assert_eq!(rows, f.total, "rows of {} beyond the family's {}", f.dual, f.total);
            }
        }
    }

    /// Every extra register kind has its registers, its read and its sets in
    /// the rows `gen_tables.py` wrote, at the dual and the count the table says.
    #[test]
    fn the_extra_registers_are_the_rows_the_generator_wrote() {
        let k = WgslKingdom::load_safe();
        let dual = |t: Ty| k.dual(t).map(|d| d.dual.clone()).unwrap_or_default();
        for x in &EXTRA_REGISTERS {
            let last = format!("reg.{}{}", x.letter, x.count - 1);
            let reg = k.rows.iter().find(|r| r.name == last).unwrap_or_else(|| panic!("{last}: no such row"));
            assert_eq!(dual(reg.output), x.dual, "{last}");
            assert!(!k.rows.iter().any(|r| r.name == format!("reg.{}{}", x.letter, x.count)), "more registers than the table's count of {}", x.count);
            let get = k.rows.iter().find(|r| r.name == format!("reg.get_{}", x.letter)).unwrap_or_else(|| panic!("reg.get_{}", x.letter));
            assert_eq!((dual(get.inputs[0]), dual(get.output).split('.').next().unwrap_or("").to_string()), (x.dual.to_string(), x.slot.to_string()));
            assert!(k.rows.iter().any(|r| r.name == format!("reg.set_{}_L0", x.letter) && dual(r.inputs[0]) == x.dual), "reg.set_{}_L0", x.letter);
            assert_eq!(extra_register_of_value(&format!("{}.opaque", x.slot)).map(|r| r.letter), Some(x.letter));
        }
    }

    /// THE GUARDED INTEGER DIVISION AND REMAINDER (2026-10-13: naga 0.20
    /// writes `/` and `%` to Metal raw, where x / 0, x % 0 and i32::MIN / -1
    /// are undefined; a random kernel's `x % 0` left the GPU running): run
    /// by the interpreter, the kingdom's helpers and the u32 remainder's
    /// guard give WGSL's values at b = 0, b = -1 and i32::MIN.
    #[test]
    fn the_guarded_division_and_remainder_give_wgsl_values() {
        use crate::wgsl::chromosome::{chromosome, ChromosomeOptions};
        use crate::wgsl::interp::{Interp, Invocation, Memory, Value};
        let body = |name: &str| COMPOSITES.iter().find(|(n, _)| *n == name).map(|(_, b)| b.to_string()).unwrap();
        let cases: [(i32, i32); 7] = [(7, 0), (-7, 0), (i32::MIN, -1), (7, -1), (7, 2), (-7, 2), (i32::MIN, 0)];
        let n = cases.len();
        let source = format!(
            "@group(0) @binding(0) var<storage, read> xs: array<i32, {n}>;\n@group(0) @binding(1) var<storage, read> bs: array<i32, {n}>;\n@group(0) @binding(2) var<storage, read> us: array<u32, {n}>;\n@group(0) @binding(3) var<storage, read_write> out: array<i32, {m}>;\n@group(0) @binding(4) var<storage, read_write> uout: array<u32, {n}>;\n{}\n{}\n@compute @workgroup_size(1)\nfn main() {{\n    for (var i = 0u; i < {n}u; i = i + 1u) {{\n        out[2u * i] = int_div(xs[i], bs[i]);\n        out[2u * i + 1u] = int_mod(xs[i], bs[i]);\n        uout[i] = us[i] % max(bitcast<u32>(bs[i]) * select(0u, 1u, bs[i] > 0i), 1u);\n    }}\n}}\n",
            body("int_div"),
            body("int_mod"),
            m = 2 * n
        );
        let kernel = crate::wgsl::reader::read(&source).unwrap_or_else(|e| panic!("{e}\n{source}"));
        let chromosomes = kernel.functions.iter().map(|f| chromosome(f, &ChromosomeOptions::default())).collect::<Result<Vec<_>, _>>().unwrap();
        let mut memory = Memory::default();
        memory.set("buffer.xs", Value::Vec(cases.iter().map(|c| Value::I32(c.0)).collect()));
        memory.set("buffer.bs", Value::Vec(cases.iter().map(|c| Value::I32(c.1)).collect()));
        memory.set("buffer.us", Value::Vec(cases.iter().map(|c| Value::U32(c.0.unsigned_abs())).collect()));
        memory.set("buffer.out", Value::Vec(vec![Value::I32(0); 2 * n]));
        memory.set("buffer.uout", Value::Vec(vec![Value::U32(0); n]));
        let mut it = Interp::new(&kernel, &chromosomes, memory, Invocation::default()).unwrap();
        it.run("main").unwrap();
        let got = |loc: &str| match it.memory.get(loc) {
            Some(Value::Vec(v)) => v.clone(),
            other => panic!("{loc}: {other:?}"),
        };
        let out = got("buffer.out");
        let uout = got("buffer.uout");
        for (i, &(x, b)) in cases.iter().enumerate() {
            // WGSL: x / 0 is x, x % 0 is 0, i32::MIN / -1 is i32::MIN, i32::MIN % -1 is 0.
            let want_div = if b == 0 || (x == i32::MIN && b == -1) { x } else { x / b };
            let want_mod = if b == 0 || (x == i32::MIN && b == -1) { 0 } else { x % b };
            assert_eq!(out[2 * i], Value::I32(want_div), "{x} / {b}");
            assert_eq!(out[2 * i + 1], Value::I32(want_mod), "{x} % {b}");
            // The u32 remainder's guard: x % max(b, 1u), b = 0 gives 0.
            let ub = if b > 0 { b as u32 } else { 0 };
            let want_u = if ub == 0 { 0 } else { x.unsigned_abs() % ub };
            assert_eq!(uout[i], Value::U32(want_u), "{} % {ub}", x.unsigned_abs());
        }
    }

    /// NO RAW INTEGER DIVISION OR REMAINDER REACHES METAL (2026-10-13): in
    /// the module naga translates for 1,000 random safe kernels, every
    /// integer `/` and `%` has a divisor that is a nonzero literal (not -1
    /// for i32), a `max(b, 1)`, a subgroup size, or sits in the kingdom's
    /// guarded helpers (int_div, int_mod), whose divisor is never 0 or -1.
    /// naga writes these to Metal as they are (examples/wgsl_msl).
    #[test]
    fn no_random_kernel_divides_by_a_value_that_may_be_zero() {
        use naga::{BinaryOperator, Expression, Literal, MathFunction, ScalarKind};
        let k = crate::wgsl::loader::WgslKingdom::load_safe();
        let role = gram_role();
        let root = dual_ty(&k, "kernel.safe").unwrap();
        let mut raw: Vec<String> = Vec::new();
        let (mut divisions, mut kernels) = (0usize, 0usize);
        for seed in 0..1000 {
            let tree = sample(&k, &role, root, 7, seed).unwrap();
            let text = crate::wgsl::safe_karva::render_tree_with(&tree, &k, &crate::wgsl::render::gram_role()).unwrap().text;
            let module = crate::wgsl::naga_shape::parse(&text).unwrap_or_else(|e| panic!("{e}"));
            let info = crate::wgsl::target::Target::recorded().validate(&module, &text).unwrap_or_else(|e| panic!("{e}"));
            kernels += 1;
            let mut check = |name: &str, f: &naga::Function, fi: &naga::valid::FunctionInfo, entry: bool| {
                let guarded_helper = name.starts_with("int_div") || name.starts_with("int_mod");
                let literal_ok = |e: &Expression| match e {
                    Expression::Literal(Literal::U32(v)) => *v != 0,
                    Expression::Literal(Literal::I32(v)) => *v != 0 && *v != -1,
                    _ => false,
                };
                let one = |h: naga::Handle<Expression>| match &f.expressions[h] {
                    Expression::Literal(Literal::U32(1) | Literal::I32(1)) => true,
                    Expression::Splat { value, .. } => matches!(f.expressions[*value], Expression::Literal(Literal::U32(1) | Literal::I32(1))),
                    _ => false,
                };
                // A call's result: the function called and its arguments
                // (a literal written through lit_u32 or lit_i32 is a call).
                type Call = (naga::Handle<Expression>, naga::Handle<naga::Function>, Vec<naga::Handle<Expression>>);
                fn calls(b: &naga::Block, out: &mut Vec<Call>) {
                    for s in b.iter() {
                        match s {
                            naga::Statement::Call { function, arguments, result: Some(r) } => out.push((*r, *function, arguments.clone())),
                            naga::Statement::Block(b) => calls(b, out),
                            naga::Statement::If { accept, reject, .. } => {
                                calls(accept, out);
                                calls(reject, out);
                            }
                            naga::Statement::Loop { body, continuing, .. } => {
                                calls(body, out);
                                calls(continuing, out);
                            }
                            naga::Statement::Switch { cases, .. } => cases.iter().for_each(|c| calls(&c.body, out)),
                            _ => {}
                        }
                    }
                }
                let mut results = Vec::new();
                calls(&f.body, &mut results);
                let lit_call = |h: naga::Handle<Expression>| {
                    results.iter().any(|(r, func, args)| *r == h && module.functions[*func].name.as_deref().is_some_and(|n| n.starts_with("lit_")) && args.first().is_some_and(|a| literal_ok(&f.expressions[*a])))
                };
                for (_, e) in f.expressions.iter() {
                    let Expression::Binary { op: BinaryOperator::Divide | BinaryOperator::Modulo, left, right } = e else { continue };
                    let integer = fi[*left].ty.inner_with(&module.types).scalar_kind().is_some_and(|s| matches!(s, ScalarKind::Sint | ScalarKind::Uint));
                    if !integer {
                        continue;
                    }
                    divisions += 1;
                    let d = &f.expressions[*right];
                    // (A constant vector of such literals: naga's constant
                    // evaluator folds max(vec3<u32>(vec2<u32>(0u), 3u),
                    // vec3<u32>(1u)) into one; seeds 671 and 763, 2026-10-14.)
                    fn nonzero(f: &naga::Function, e: &Expression, ok: &dyn Fn(&Expression) -> bool) -> bool {
                        match e {
                            Expression::Compose { components, .. } => !components.is_empty() && components.iter().all(|c| nonzero(f, &f.expressions[*c], ok)),
                            Expression::Splat { value, .. } => nonzero(f, &f.expressions[*value], ok),
                            other => ok(other),
                        }
                    }
                    let ok = guarded_helper
                        || nonzero(f, d, &literal_ok)
                        || lit_call(*right)
                        || matches!(d, Expression::Math { fun: MathFunction::Max, arg1: Some(a), .. } if one(*a))
                        // (The subgroup size, a builtin argument of the entry point: at least 1.)
                        || (entry && matches!(d, Expression::FunctionArgument(_)));
                    if !ok && raw.len() < 8 {
                        raw.push(format!("seed {seed}, fn {name}: {d:?}"));
                    }
                }
            };
            for (h, f) in module.functions.iter() {
                check(f.name.as_deref().unwrap_or(""), f, &info[h], false);
            }
            for (i, ep) in module.entry_points.iter().enumerate() {
                check(&ep.name, &ep.function, info.get_entry_point(i), true);
            }
        }
        println!("{kernels} random kernels, {divisions} integer divisions and remainders, {} raw", raw.len());
        assert!(raw.is_empty(), "{}", raw.join("\n"));
    }

    /// THE SAMPLER WITHIN THE BOUND (2026-10-13): 1,000 random safe kernels
    /// drawn by `sample_within` under the fitness's static cost bound are
    /// all within it (524 of 1,000 drawn by `sample` were over), and each
    /// still ends in a store.
    #[test]
    fn the_sampler_within_the_bound_draws_no_kernel_over_it() {
        let k = crate::wgsl::loader::WgslKingdom::load_safe();
        let role = gram_role();
        let root = dual_ty(&k, "kernel.safe").unwrap();
        let bound = crate::wgsl::safe_karva::default_bound(&k).unwrap();
        let mut over = 0;
        for seed in 0..1000 {
            let t = sample_within(&k, &role, root, 7, seed, &bound).unwrap();
            let c = cost(&t);
            over += usize::from(c.per_item > bound.per_item || c.base > bound.base);
            fn stores(t: &Tree) -> bool {
                t.name.starts_with("store.h") || t.kids.iter().any(stores)
            }
            assert!(stores(&t.kids[2]), "seed {seed}: an epilogue with no store: {}", t.kids[2].to_sexpr());
        }
        assert_eq!(over, 0, "random kernels over the cost bound");
    }

    /// An expression-bounded item loop is costed by its bound at the worst
    /// shape: `cfg.m * cfg.d` is 8,192 passes over the items and is refused
    /// by the cost bound, `(cfg.d + 63u) / 64u` is less than one pass; a
    /// tilex loop outside the item loop bounded by cfg.m is item work, one
    /// body per item; bounded by cfg.d, 8,192 bodies of base.
    #[test]
    fn an_expression_bound_is_costed_at_the_worst_shape() {
        let k = crate::wgsl::loader::WgslKingdom::load_safe();
        let t = |n: &str| dual_ty(&k, n).unwrap();
        // (Uniform bounds since legality item 5: the loops may hold barriers.)
        let u = t("u32.uniform");
        let m = || Tree::new("uniform.cfg_m", u, vec![]);
        let d = || Tree::new("uniform.cfg_d", u, vec![]);
        let lit = |v: f32| Tree::new("uniform.of_lit", u, vec![Tree::literal("literal.lit4096", t("u32.lit4096"), v)]);
        let body = || Tree::new("block.one_L1", t("block.L1"), vec![Tree::new("stmt.barrier_L1", t("stmt.L1"), vec![])]);
        let bound = |e: Tree| Tree::new("bound.of_uniform", t("u32.boundu"), vec![e]);
        let quadratic = Tree::new("loop.items_boundx", t("stmt.L0"), vec![bound(Tree::new("uniform.mul", u, vec![m(), d()])), body()]);
        let linear = Tree::new("loop.items_boundx", t("stmt.L0"), vec![bound(Tree::new("uniform.div", u, vec![Tree::new("uniform.add", u, vec![d(), lit(63.0)]), lit(64.0)])), body()]);
        let plain = Tree::new("loop.items_boundm", t("stmt.L0"), vec![Tree::new("bound.m", t("u32.boundm"), vec![]), body()]);
        let (q, l, p) = (cost(&quadratic).per_item, cost(&linear).per_item, cost(&plain).per_item);
        // Capped at MAX_LOOP_TRIP (every computed bound, 2026-10-12): 2^24
        // trips, 256 passes of m; the bound's own nodes are costed with the
        // body, so the ratio is 256 times (bound nodes + body) / (1 + body).
        assert!(q / p > 256.0 && q / p < 2048.0, "quadratic {q} vs plain {p}");
        assert!(l < 8.0 * p, "a bound below m is one pass (its nodes aside): {l} vs {p}");
        // A tilex loop outside the item loop bounded by cfg.m is item work
        // (fit_A2 gen 6's r12p1, 2026-10-11): one body per item, as the item
        // loop costs it, not 65,536 bodies into the base. Bounded by cfg.d it
        // does not grow with m and is base work, 8,192 bodies.
        let tile_body = || Tree::new("block.one_L2", t("block.L2"), vec![Tree::new("stmt.barrier_L2", t("stmt.L2"), vec![])]);
        let tilex = Tree::new("loop.tilex_L0", t("stmt.L0"), vec![bound(m()), tile_body()]);
        let (c, p) = (cost(&tilex), cost(&plain));
        // (its bound's nodes are read every pass too, so a few more per item)
        assert!(c.per_item >= p.per_item && c.per_item <= p.per_item + 8.0 && c.base < 10.0, "{c:?} vs the item loop {p:?}");
        let over_d = cost(&Tree::new("loop.tilex_L0", t("stmt.L0"), vec![bound(d()), tile_body()]));
        assert!(over_d.base >= 8192.0 * 2.0 && over_d.per_item == 0.0, "{over_d:?}");
    }

    /// THE OLD wg.get / wg.set MASK TO THE ARRAY'S DECLARED LENGTH
    /// (2026-10-12; they masked to 2,047 whatever the array's length, the
    /// legality checker's item 6): an array declared with 256 elements read
    /// and written at a run-time index masks it `& 255u`, one of 1,056 by
    /// `min(i, 1055u)`.
    #[test]
    fn the_old_workgroup_rows_mask_to_the_declared_length() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        let ty = |n: &str| dual_ty(&k, n).unwrap();
        for (len, want) in [(256.0, "& 255u)"), (1056.0, ", 1055u)")] {
            let decl = Tree::new("wg.decls_one", ty("wg.decls"), vec![Tree::new("wg.decl_f", ty("wg.decl"), vec![Tree::new("wg.a", ty("wgmem<f32>.real"), vec![]), Tree::literal("literal.wglen", ty("u32.wglen"), len)])]);
            let grid = Tree::new("grid.with_wg", ty("grid.grid"), vec![Tree::fallback(ty("grid.grid")), decl]);
            let i = || Tree::new("safe.counter_index", ty("u32.index"), vec![Tree::new("safe.counter", ty("u32.counter"), vec![])]);
            let get = Tree::new("wg.get", ty("f32.real"), vec![Tree::new("wg.a", ty("wgmem<f32>.real"), vec![]), i()]);
            let set = Tree::new("reg.set_f_L1", ty("stmt.L1"), vec![Tree::new("reg.f1", ty("reg<f32>.real"), vec![]), get]);
            let body = Tree::new("block.one_L0", ty("block.L0"), vec![Tree::new("loop.items_boundm", ty("stmt.L0"), vec![Tree::new("bound.m", ty("u32.boundm"), vec![]), Tree::new("block.one_L1", ty("block.L1"), vec![set])])]);
            let kernel = Tree::new("kernel.entry_safe", ty("kernel.safe"), vec![grid, body, Tree::fallback(ty("block.O0"))]);
            let text = rr.kernel(&kernel).unwrap().text;
            assert!(text.contains(want) && !text.contains("& 2047u"), "{text}");
            validate_text(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        }
    }

    /// A 16-ELEMENT PRIVATE ARRAY TAKES ITS OWN INDEX (2026-10-12; 369 of
    /// 1,000 random genes indexed one through safe.pidx64, i & 63u): every
    /// row on a vec4, u32, bool or vec2 private array takes u32.pindex16,
    /// which only safe.pidx16 makes, and 2,000 random safe kernels never
    /// index such an array through safe.pidx64.
    #[test]
    fn a_sixteen_element_private_array_takes_its_own_index() {
        let k = WgslKingdom::load_safe();
        let dual = |t: Ty| k.dual(t).map(|d| d.dual.clone()).unwrap_or_default();
        let sixteen = ["parr<vec4<f32>>.real", "parr<u32>.index", "parr<bool>.flag", "parr<vec2<f32>>.real"];
        for r in k.rows.iter().filter(|r| r.inputs.first().is_some_and(|&t| sixteen.contains(&dual(t).as_str()))) {
            assert_eq!(r.inputs.get(1).map(|&t| dual(t)).as_deref(), Some("u32.pindex16"), "{}", r.name);
        }
        let makers: Vec<&str> = k.rows.iter().filter(|r| dual(r.output) == "u32.pindex16").map(|r| r.name.as_str()).collect();
        assert_eq!(makers, vec!["safe.pidx16"]);
        fn wide(t: &Tree, k: &WgslKingdom, sixteen: &[&str]) -> bool {
            let on_sixteen = t.kids.first().and_then(|a| k.dual(a.ty)).is_some_and(|d| sixteen.contains(&d.dual.as_str()));
            (on_sixteen && t.kids.get(1).is_some_and(|i| i.name == "safe.pidx64")) || t.kids.iter().any(|c| wide(c, k, sixteen))
        }
        let (role, root) = (gram_role(), dual_ty(&k, "kernel.safe").unwrap());
        let bad = (0..2000).filter(|&seed| wide(&sample(&k, &role, root, 7, seed).unwrap(), &k, &sixteen)).count();
        assert_eq!(bad, 0);
    }

    /// THE SAFE SAMPLER NEVER READS h AT ANY INDEX (2026-10-12; 25 of 1,000
    /// random safe genes reached the role's load.h): 2,000 random safe
    /// kernels hold no load.h (h is read through load.h_safe only).
    #[test]
    fn the_safe_sampler_never_reads_h_at_any_index() {
        let k = WgslKingdom::load_safe();
        let role = gram_role();
        let root = dual_ty(&k, "kernel.safe").unwrap();
        fn has(t: &Tree, name: &str) -> bool {
            t.name == name || t.kids.iter().any(|c| has(c, name))
        }
        let reads = (0..2000).filter(|&seed| has(&sample(&k, &role, root, 7, seed).unwrap(), "load.h")).count();
        assert_eq!(reads, 0, "random safe kernels reading h at any index");
    }

    /// A FALLBACK CELL IS THE THREAD'S OWN (2026-10-12; the legality checker
    /// found 415 of 1,000 random genes storing every thread to cell 0, a
    /// data race): a fallback of u32.cell, u32.threadrow or u32.threadcol
    /// renders the thread's own first row, column and cell, the store's
    /// guard included.
    #[test]
    fn a_fallback_cell_is_the_threads_own() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        let ty = |n: &str| dual_ty(&k, n).unwrap();
        let store = |cell: Tree| Tree::new("block.one_O0", ty("block.O0"), vec![Tree::new("store.h_O0", ty("stmt.O0"), vec![cell, Tree::literal("literal.real", ty("f32.real"), 1.0)])]);
        let kernel = |e: Tree| Tree::new("kernel.entry_safe", ty("kernel.safe"), vec![Tree::fallback(ty("grid.grid")), Tree::fallback(ty("block.L0")), e]);
        let own = "h[(((gid.y * 1u)) * cfg.d + ((gid.x * 1u)))]";
        let whole = rr.kernel(&kernel(store(Tree::fallback(ty("u32.cell"))))).unwrap().text;
        assert!(whole.contains(own) && whole.contains("if ((((gid.y * 1u)) < cfg.r_rows)"), "{whole}");
        let parts = rr.kernel(&kernel(store(Tree::new("safe.cell_of", ty("u32.cell"), vec![Tree::fallback(ty("u32.threadrow")), Tree::fallback(ty("u32.threadcol"))])))).unwrap().text;
        assert!(parts.contains(own), "{parts}");
        for text in [&whole, &parts] {
            validate_text(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
            assert!(!text.contains("h[(0u"), "{text}");
        }
    }

    /// A LOOP COUNTER OUTSIDE ITS LOOP IS 0 (2026-10-12; the legality
    /// checker found 852 of 1,000 random genes reading `t` or `k` outside
    /// their loop, which read the last loop's exit value): the item counter
    /// and a tile counter read at the top level render `lit_u32(0u)`; inside
    /// their loops, `t` and `k`.
    #[test]
    fn a_loop_counter_outside_its_loop_renders_zero() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        let ty = |n: &str| dual_ty(&k, n).unwrap();
        let index = ty("u32.index");
        let set_i = |rung: &str, v: Tree| Tree::new(&format!("reg.set_i_{rung}"), ty(&format!("stmt.{rung}")), vec![Tree::new("reg.i0", ty("reg<u32>.index"), vec![]), v]);
        let item = || Tree::new("safe.counter_index", index, vec![Tree::new("safe.counter", ty("u32.counter"), vec![])]);
        let tile = || Tree::new("safe.tile_index", index, vec![Tree::new("safe.tile_counter", ty("u32.tcounter"), vec![])]);
        let outside = Tree::new("block.seq2_L0", ty("block.L0"), vec![set_i("L0", item()), set_i("L0", tile())]);
        let items = Tree::new("loop.items_boundm", ty("stmt.L0"), vec![Tree::new("bound.m", ty("u32.boundm"), vec![]), Tree::new("block.one_L1", ty("block.L1"), vec![set_i("L1", item())])]);
        let tiled = Tree::new("loop.tile_L0", ty("stmt.L0"), vec![Tree::literal("literal.lit64", ty("u32.lit64"), 4.0), Tree::new("block.one_L2", ty("block.L2"), vec![set_i("L2", tile())])]);
        let inside = Tree::new("block.seq2_L0", ty("block.L0"), vec![items, tiled]);
        let kernel = |body: Tree| Tree::new("kernel.entry_safe", ty("kernel.safe"), vec![Tree::fallback(ty("grid.grid")), body, Tree::fallback(ty("block.O0"))]);
        let out = rr.kernel(&kernel(outside)).unwrap().text;
        assert_eq!(out.matches("i0 = lit_u32(0u);").count(), 2, "{out}");
        validate_text(&out).unwrap_or_else(|e| panic!("{e}\n{out}"));
        let inn = rr.kernel(&kernel(inside)).unwrap().text;
        assert!(inn.contains("i0 = t;") && inn.contains("i0 = k;"), "{inn}");
    }

    /// NESTED FULL UNROLLS ARE BOUNDED (2026-10-12): an unroll of 8 inside
    /// an unroll of 8 is written out (64 copies); an unroll of 38 inside an
    /// unroll of 57 is written as its loop inside the 57 copies, the same
    /// computation (the 57 x 38 x 38 gene of the random kernels was 17 MB).
    #[test]
    fn nested_full_unrolls_write_at_most_the_bound_in_copies() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        let ty = |n: &str| dual_ty(&k, n).unwrap();
        let lit = |v: f32| Tree::literal("literal.lit64", ty("u32.lit64"), v);
        let set = Tree::new("reg.set_f_L2", ty("stmt.L2"), vec![Tree::new("reg.f1", ty("reg<f32>.real"), vec![]), Tree::literal("literal.real", ty("f32.real"), 0.5)]);
        let nested = |outer: f32, inner: f32| {
            let inner = Tree::new("loop.unroll_L1", ty("stmt.L1"), vec![lit(inner), Tree::new("block.one_L2", ty("block.L2"), vec![set.clone()])]);
            let outer = Tree::new("loop.unroll_L0", ty("stmt.L0"), vec![lit(outer), Tree::new("block.one_L1", ty("block.L1"), vec![inner])]);
            Tree::new("kernel.entry_safe", ty("kernel.safe"), vec![Tree::fallback(ty("grid.grid")), Tree::new("block.one_L0", ty("block.L0"), vec![outer]), Tree::fallback(ty("block.O0"))])
        };
        let small = rr.kernel(&nested(8.0, 8.0)).unwrap().text;
        assert_eq!(small.matches("f1 = 0.5f;").count(), 64, "{small}");
        assert!(!small.contains("for (k2 = 0u"), "{small}");
        let big = rr.kernel(&nested(57.0, 38.0)).unwrap().text;
        assert_eq!(big.matches("f1 = 0.5f;").count(), 57, "the inner unroll is its loop, once per outer copy");
        assert_eq!(big.matches("for (k2 = 0u; k2 < 38u; k2 = k2 + 1u)").count(), 57);
        validate_text(&big).unwrap_or_else(|e| panic!("{e}"));
    }

    fn setup() -> (WgslKingdom, FunctionTable, Role) {
        (WgslKingdom::load(), FunctionTable::shipped(), gram_role())
    }

    #[test]
    fn the_shipped_kernel_as_a_tree_renders_and_validates() {
        let (k, t, r) = setup();
        let tree = gram_shipped_tree(&k).unwrap();
        assert_eq!(tree.size(), 82, "{}", tree.to_sexpr());
        let out = Renderer::new(&k, &t, &r).kernel(&tree).unwrap();
        assert_eq!(out.grid, Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false });
        assert!(out.text.contains("for (t = 0u; t < cfg.m; t = t + 1u)"), "{}", out.text);
        assert!(out.text.contains("h[(((i1 * cfg.d) + i2))] = ((h[(((i1 * cfg.d) + i2))] + f0));"), "{}", out.text);
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
    }

    #[test]
    fn the_cost_estimate_counts_loop_multipliers() {
        let (k, _, _) = setup();
        let ty = |n: &str| dual_ty(&k, n).unwrap();
        let shipped = cost(&gram_shipped_tree(&k).unwrap());
        assert!(shipped.per_item > 10.0 && shipped.per_item < 40.0, "{shipped:?}");
        assert!(shipped.base > 10.0 && shipped.base < 200.0, "{shipped:?}");
        // A tile of 32 around the item loop multiplies the per-item cost by 32.
        let (stmt, block) = (ty("stmt.stmt"), ty("block.block"));
        let body = Tree::new("loop.items", stmt, vec![Tree::new("block.one", block, vec![Tree::new("reg.set_f", stmt, vec![Tree::new("reg.f0", ty("reg<f32>.real"), vec![]), Tree::new("load.phi", ty("f32.real"), vec![Tree::new("builtin.item", ty("u32.index"), vec![])])])])]);
        let inner = cost(&Tree::new("kernel.entry", ty("kernel.kernel"), vec![Tree::fallback(ty("grid.grid")), Tree::new("block.one", block, vec![body.clone()])]));
        let tiled = cost(&Tree::new("kernel.entry", ty("kernel.kernel"), vec![Tree::fallback(ty("grid.grid")), Tree::new("block.one", block, vec![Tree::new("loop.tile", stmt, vec![Tree::literal("literal.wgdim", ty("u32.wgdim"), 32.0), Tree::new("block.one", block, vec![body])])])]));
        assert_eq!(tiled.per_item, 32.0 * inner.per_item, "{inner:?} {tiled:?}");
    }

    /// Every statement row inside a minimal kernel, validated by naga.
    #[test]
    fn every_statement_row_renders_and_validates() {
        let (k, t, r) = setup();
        let ty = |n: &str| dual_ty(&k, n).unwrap();
        let (stmt, block, index, real, flag) = (ty("stmt.stmt"), ty("block.block"), ty("u32.index"), ty("f32.real"), ty("bool.flag"));
        let one = |s: Tree| Tree::new("block.one", block, vec![s]);
        let lit_i = |v: f32| Tree::literal("literal.index", index, v);
        let lit_f = |v: f32| Tree::literal("literal.real", real, v);
        let set_f = || Tree::new("reg.set_f", stmt, vec![Tree::new("reg.f1", ty("reg<f32>.real"), vec![]), Tree::new("arith.add", real, vec![Tree::new("reg.get_f", real, vec![Tree::new("reg.f1", ty("reg<f32>.real"), vec![])]), lit_f(1.5)])]);
        let cases: Vec<(&str, Tree)> = vec![
            ("return", Tree::new("stmt.return", stmt, vec![])),
            ("barrier", Tree::new("stmt.barrier", stmt, vec![])),
            ("block", Tree::new("stmt.block", stmt, vec![one(set_f())])),
            ("if", Tree::new("stmt.if", stmt, vec![Tree::new("reg.get_b", flag, vec![Tree::new("reg.b0", ty("reg<bool>.flag"), vec![])]), one(set_f())])),
            ("barrier masked under if", Tree::new("stmt.if", stmt, vec![Tree::literal("literal.flag", flag, 1.0), one(Tree::new("stmt.barrier", stmt, vec![]))])),
            ("outer", Tree::new("reg.set_f", stmt, vec![Tree::new("reg.f2", ty("reg<f32>.real"), vec![]), Tree::new("geom.determinant", real, vec![Tree::new("geom.outer", ty("mat3x3<f32>.real"), vec![Tree::fallback(ty("vec3<f32>.real")), Tree::fallback(ty("vec3<f32>.real"))])])])),
            ("inverse", Tree::new("reg.set_f", stmt, vec![Tree::new("reg.f2", ty("reg<f32>.real"), vec![]), Tree::new("geom.determinant", real, vec![Tree::new("geom.inverse", ty("mat4x4<f32>.real"), vec![Tree::new("geom.inverse", ty("mat4x4<f32>.real"), vec![Tree::fallback(ty("mat4x4<f32>.real"))])])])])),
            ("inverse2", Tree::new("reg.set_f", stmt, vec![Tree::new("reg.f2", ty("reg<f32>.real"), vec![]), Tree::new("geom.determinant", real, vec![Tree::new("geom.inverse", ty("mat2x2<f32>.real"), vec![Tree::fallback(ty("mat2x2<f32>.real"))]), ])])),
            ("inverse3", Tree::new("reg.set_f", stmt, vec![Tree::new("reg.f2", ty("reg<f32>.real"), vec![]), Tree::new("geom.determinant", real, vec![Tree::new("geom.inverse", ty("mat3x3<f32>.real"), vec![Tree::fallback(ty("mat3x3<f32>.real"))]), ])])),
            ("if_else", Tree::new("stmt.if_else", stmt, vec![Tree::literal("literal.flag", flag, 1.0), one(set_f()), one(Tree::new("stmt.return", stmt, vec![]))])),
            ("items", Tree::new("loop.items", stmt, vec![one(set_f())])),
            ("items_unrolled", Tree::new("loop.items_unrolled", stmt, vec![Tree::literal("literal.unroll", ty("u32.unroll"), 4.0), one(Tree::new("store.h", stmt, vec![Tree::new("builtin.item", index, vec![]), Tree::new("load.phi", real, vec![Tree::new("builtin.item", index, vec![])])]))])),
            ("tile", Tree::new("loop.tile", stmt, vec![Tree::literal("literal.wgdim", ty("u32.wgdim"), 32.0), one(Tree::new("wg.set", stmt, vec![Tree::new("wg.b", ty("wgmem<f32>.real"), vec![]), Tree::new("builtin.tile", index, vec![]), Tree::new("wg.get", real, vec![Tree::new("wg.a", ty("wgmem<f32>.real"), vec![]), lit_i(7.0)])]))])),
            ("set_v", Tree::new("reg.set_v", stmt, vec![Tree::new("reg.v1", ty("reg<vec4<f32>>.real"), vec![]), Tree::new("load.phi4", ty("vec4<f32>.real"), vec![lit_i(4.0)])])),
            ("set_i", Tree::new("reg.set_i", stmt, vec![Tree::new("reg.i3", ty("reg<u32>.index"), vec![]), Tree::new("shape.access_2", index, vec![Tree::new("builtin.num_workgroups", ty("vec3<u32>.count"), vec![])])])),
            ("set_b", Tree::new("reg.set_b", stmt, vec![Tree::new("reg.b0", ty("reg<bool>.flag"), vec![]), Tree::new("logic.is_nan", flag, vec![lit_f(0.0)])])),
            ("fallback stmt", Tree::fallback(stmt)),
        ];
        let rr = Renderer::new(&k, &t, &r);
        let grid = Tree::fallback(ty("grid.grid"));
        for (name, s) in cases {
            let tree = Tree::new("kernel.entry", ty("kernel.kernel"), vec![grid.clone(), one(s)]);
            let out = rr.kernel(&tree).unwrap_or_else(|e| panic!("{name}: {e}"));
            validate_text(&out.text).unwrap_or_else(|e| panic!("{name}: {e}\n{}", out.text));
            if name == "barrier masked under if" {
                assert_eq!(out.masked, vec!["stmt.barrier in non-uniform control flow".to_string()]);
            }
            if name == "outer" {
                assert_eq!(out.helpers, vec!["outer_3_3".to_string()]);
            }
        }
        // Refusals by construction.
        let nested = Tree::new("kernel.entry", ty("kernel.kernel"), vec![grid.clone(), one(Tree::new("loop.items", stmt, vec![one(Tree::new("loop.items", stmt, vec![one(set_f())]))]))]);
        assert_eq!(rr.kernel(&nested).unwrap_err(), "loop.items nested in loop.items");
        // A grid over 1,024 threads is cut to the device's limit: x as written, y the largest that fits
        // (64 x 64 runs as 64 x 16; until 2026-10-12 each axis snapped to {4, 8, 16, 32}).
        let big = Tree::new("kernel.entry", ty("kernel.kernel"), vec![Tree::new("grid.of", ty("grid.grid"), vec![Tree::literal("literal.wgdim", ty("u32.wgdim"), 64.0), Tree::literal("literal.wgdim", ty("u32.wgdim"), 64.0), Tree::fallback(ty("u32.blk")), Tree::fallback(ty("u32.blk"))]), one(set_f())]);
        assert_eq!(rr.kernel(&big).unwrap().grid, Grid { x: 64, y: 16, bx: 1, by: 1, rows_x: false });
        // An empty kernel (the root's fallback) is a kernel.
        let empty = rr.kernel(&Tree::fallback(ty("kernel.kernel"))).unwrap();
        validate_text(&empty.text).unwrap();
    }

    /// The safe kingdom: the shipped kernel as a safe tree renders and
    /// validates; 500 random typed kernels from the safe root all render
    /// (no refusal is possible by construction: no nested long loop, no
    /// barrier under a condition) and all validate.
    #[test]
    fn the_safe_kingdom_renders_the_shipped_kernel_and_every_random_kernel() {
        let (k, t, r) = (WgslKingdom::load_safe(), FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &r);
        let tree = gram_shipped_safe_tree(&k).unwrap();
        let out = rr.kernel(&tree).unwrap();
        assert!(out.text.contains("for (t = 0u; t < cfg.m; t = t + 1u)"), "{}", out.text);
        assert!(out.text.contains("if (") && out.text.contains("h[(((gid.y * 1u)) * cfg.d + ((gid.x * 1u)))] = h["), "{}", out.text);
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let root = dual_ty(&k, "kernel.safe").unwrap();
        let n = 500;
        let (mut refused, mut invalid, mut masked) = (Vec::new(), Vec::new(), 0usize);
        for seed in 0..n {
            let tree = sample(&k, &r, root, 7, seed).unwrap();
            match rr.kernel(&tree) {
                Err(e) => refused.push(format!("seed {seed}: {e}")),
                Ok(out) => {
                    masked += out.masked.len();
                    if let Err(e) = validate_text(&out.text) {
                        let brief: String = e.lines().filter(|l| !l.trim().is_empty()).take(4).collect::<Vec<_>>().join(" | ");
                        invalid.push(format!("seed {seed} {}: {}", tree.head(), &brief[..brief.len().min(300)]));
                    }
                }
            }
        }
        println!("safe kingdom: {n} random kernels, {} refused, {} invalid, {masked} barriers masked", refused.len(), invalid.len());
        assert!(refused.is_empty(), "{} refused by construction:\n{}", refused.len(), refused.join("\n"));
        assert_eq!(masked, 0, "a barrier under a condition has no spelling in the safe grammar");
        assert!(invalid.is_empty(), "{} invalid:\n{}", invalid.len(), invalid.join("\n"));
    }


    /// The rows run A's declines added (2026-10-11): the registers up to
    /// f63, v31, i63 and b15 (`f32` and `i32` are reserved words, spelled
    /// `f32r` and `i32r`), and a lane of a vec4 element of a workgroup or
    /// private array; each renders in a safe kernel and validates.
    #[test]
    fn the_rows_of_run_a_render_and_validate() {
        let (k, t, r) = (WgslKingdom::load_safe(), FunctionTable::safe(), gram_role());
        let ty = |n: &str| dual_ty(&k, n).unwrap();
        let (stmt, index, real, flag) = (ty("stmt.L0"), ty("u32.index"), ty("f32.real"), ty("bool.flag"));
        let lit_i = |v: f32| Tree::literal("literal.index", index, v);
        let cases: Vec<(&str, Tree)> = vec![
            ("set_f at the last register, a reserved word spelled with r", Tree::new("reg.set_f", stmt, vec![Tree::new("reg.f32", ty("reg<f32>.real"), vec![]), Tree::new("reg.get_f", real, vec![Tree::new("reg.f63", ty("reg<f32>.real"), vec![])])])),
            ("set_i at a reserved word", Tree::new("reg.set_i", stmt, vec![Tree::new("reg.i32", ty("reg<u32>.index"), vec![]), Tree::new("reg.get_i", index, vec![Tree::new("reg.i63", ty("reg<u32>.index"), vec![])])])),
            ("set_v at the last register", Tree::new("reg.set_v", stmt, vec![Tree::new("reg.v31", ty("reg<vec4<f32>>.real"), vec![]), Tree::new("reg.get_v", ty("vec4<f32>.real"), vec![Tree::new("reg.v16", ty("reg<vec4<f32>>.real"), vec![])])])),
            ("set_b at the last register", Tree::new("reg.set_b", stmt, vec![Tree::new("reg.b15", ty("reg<bool>.flag"), vec![]), Tree::new("reg.get_b", flag, vec![Tree::new("reg.b8", ty("reg<bool>.flag"), vec![])])])),
            ("wg.set_v_lane", Tree::new("wg.set_v_lane_L0", ty("stmt.L0"), vec![Tree::new("wg.va", ty("wgmem<vec4<f32>>.real"), vec![]), Tree::new("safe.wgv_of", ty("u32.wgvindex"), vec![lit_i(3.0)]), lit_i(2.0), Tree::new("wg.get_v_lane", real, vec![Tree::new("wg.vb", ty("wgmem<vec4<f32>>.real"), vec![]), Tree::new("safe.wgv_of", ty("u32.wgvindex"), vec![lit_i(5.0)]), lit_i(1.0)])])),
            ("parr.set_v_lane", Tree::new("parr.set_v_lane_L0", ty("stmt.L0"), vec![Tree::new("parr.v", ty("parr<vec4<f32>>.real"), vec![]), Tree::new("safe.pidx16", ty("u32.pindex16"), vec![lit_i(3.0)]), lit_i(2.0), Tree::new("parr.get_v_lane", real, vec![Tree::new("parr.w", ty("parr<vec4<f32>>.real"), vec![]), Tree::new("safe.pidx16", ty("u32.pindex"), vec![lit_i(5.0)]), lit_i(1.0)])])),
        ];
        let rr = Renderer::new(&k, &t, &r);
        let grid = Tree::fallback(ty("grid.grid"));
        for (name, s) in cases {
            let tree = Tree::new("kernel.entry_safe", ty("kernel.safe"), vec![grid.clone(), Tree::new("block.one_L0", ty("block.L0"), vec![s]), Tree::fallback(ty("block.O0"))]);
            let out = rr.kernel(&tree).unwrap_or_else(|e| panic!("{name}: {e}"));
            validate_text(&out.text).unwrap_or_else(|e| panic!("{name}: {e}\n{}", out.text));
            if name.starts_with("set_f at") {
                assert!(out.text.contains("f32r = f63;"), "{}", out.text);
            }
            if name.starts_with("set_i at") {
                assert!(out.text.contains("i32r = i63;"), "{}", out.text);
            }
            if name == "wg.set_v_lane" {
                assert!(out.text.contains("wg_va[3u][(2u) & 3u] = wg_vb[5u][(1u) & 3u];"), "{}", out.text);
            }
        }
        assert_eq!(register_var("i", 32), "i32r");
        assert_eq!(register_var("f", 16), "f16r");
        assert_eq!(register_var("v", 31), "v31");
    }
    /// Every composite helper, called from a kernel, validates.
    #[test]
    fn every_composite_row_renders_and_validates() {
        let (k, t, r) = setup();
        let ty = |n: &str| dual_ty(&k, n).unwrap();
        let rr = Renderer::new(&k, &t, &r);
        let composites: Vec<&WgslRow> = k.rows.iter().filter(|row| t.named(&row.name).is_some_and(|tp| tp.naga == "composite") && !row.name.starts_with("kernel.") && !row.name.starts_with("grid.") && !row.name.starts_with("block.") && !row.name.starts_with("stmt.") && !row.name.starts_with("loop.") && !row.name.starts_with("reg.") && !row.name.starts_with("wg.") && !row.name.starts_with("builtin.")).collect();
        assert!(composites.len() >= 40, "{}", composites.len());
        let mut failed = Vec::new();
        for row in composites {
            if k.dual(row.output).unwrap().scalar() == Some("f16") || row.inputs.iter().any(|i| k.dual(*i).unwrap().scalar() == Some("f16")) {
                continue; // the prologue does not enable f16
            }
            // The value, its children literal fallbacks of their duals, assigned
            // to a register through a conversion that the value's own dual permits.
            let kids: Vec<Tree> = row.inputs.iter().map(|&i| Tree::fallback(i)).collect();
            let value = Tree::new(&row.name, row.output, kids);
            let out_dual = k.dual(row.output).unwrap();
            let value = if out_dual.lanes > 1 { Tree::new("shape.access_0", row.output, vec![value]) } else { value };
            let lanes = if out_dual.lanes > 1 { 1 } else { out_dual.lanes };
            let stmt = match (out_dual.scalar(), lanes) {
                (Some("f32"), 1) => Tree::new("reg.set_f", ty("stmt.stmt"), vec![Tree::new("reg.f0", ty("reg<f32>.real"), vec![]), value]),
                (Some("u32"), 1) => Tree::new("reg.set_i", ty("stmt.stmt"), vec![Tree::new("reg.i0", ty("reg<u32>.index"), vec![]), value]),
                (Some("bool"), 1) => Tree::new("reg.set_b", ty("stmt.stmt"), vec![Tree::new("reg.b0", ty("reg<bool>.flag"), vec![]), value]),
                (Some("i32"), 1) => Tree::new("reg.set_i", ty("stmt.stmt"), vec![Tree::new("reg.i0", ty("reg<u32>.index"), vec![]), Tree::new("index.of_int", ty("u32.index"), vec![value])]),
                (s, l) => panic!("{}: out {} ({s:?}, {l} lanes) has no register", row.name, out_dual.dual),
            };
            let tree = Tree::new("kernel.entry", ty("kernel.kernel"), vec![Tree::fallback(ty("grid.grid")), Tree::new("block.one", ty("block.block"), vec![stmt])]);
            match rr.kernel(&tree).and_then(|o| validate_text(&o.text).map(|_| o)) {
                Ok(_) => {}
                Err(e) => failed.push(format!("{}: {}", row.name, e.lines().next().unwrap_or(""))),
            }
        }
        assert!(failed.is_empty(), "{} composites fail:\n{}", failed.len(), failed.join("\n"));
    }

    /// The kingdom gate: random typed trees from the kernel root, rendered
    /// and validated, every failure named. N of N is the bar.
    #[test]
    fn random_typed_kernels_render_and_validate_n_of_n() {
        let (k, t, r) = setup();
        let rr = Renderer::new(&k, &t, &r);
        let root = dual_ty(&k, "kernel.kernel").unwrap();
        let n = 500;
        let mut refused: BTreeMap<String, usize> = BTreeMap::new();
        let mut invalid: Vec<String> = Vec::new();
        let (mut rendered, mut sizes, mut masked) = (0, 0usize, 0usize);
        for seed in 0..n {
            let tree = sample(&k, &r, root, 7, seed).unwrap();
            sizes += tree.size();
            match rr.kernel(&tree) {
                Err(e) => *refused.entry(e).or_default() += 1,
                Ok(out) => {
                    rendered += 1;
                    masked += out.masked.len();
                    if let Err(e) = validate_text(&out.text) {
                        if invalid.len() < 8 {
                            let dir = std::env::var("CARGO_TARGET_TMPDIR").unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().to_string());
                            std::fs::write(format!("{dir}/wgsl_random_invalid_{seed}.wgsl"), format!("// {e}\n{}", out.text)).unwrap_or_else(|err| panic!("writing the invalid kernel of seed {seed}: {err}"));
                        }
                        let brief: String = e.lines().filter(|l| !l.trim().is_empty()).take(6).collect::<Vec<_>>().join(" | ");
                        invalid.push(format!("seed {seed} {}: {}", tree.head(), &brief[..brief.len().min(400)]));
                    }
                }
            }
        }
        println!("{n} random kernels, mean {} nodes: {rendered} rendered ({masked} barriers masked), {} refused by construction {refused:?}, {} invalid", sizes / n as usize, n as usize - rendered, invalid.len());
        assert!(invalid.is_empty(), "{} of {rendered} rendered kernels fail naga:\n{}", invalid.len(), invalid.join("\n"));
        // N of N (2026-10-13: the sampler nests no long loop in one of its
        // kind, the renderer's only refusal; 440 of 500 before).
        assert!(refused.is_empty() && rendered == n as usize, "only {rendered} of {n} rendered: {refused:?}");
    }

    /// THE GRAM'S INPUT AND OUTPUT ROWS UNDER AN INTERFACE WITHOUT THEM
    /// (2026-10-17, a point mutation under v24's role drew load.phi4_safe and
    /// rendered `vec4<f32>([i], [i + 1u], ...)`): role_spells withholds every
    /// such row under v24's interface (and none under the Gram's), and the
    /// renderer names any it is asked for as our software's bug, never text.
    #[test]
    fn the_grams_input_and_output_rows_are_withheld_where_an_interface_has_none() {
        let k = WgslKingdom::load_safe();
        let src = "struct Cfg { n_queries: u32, };\n@group(0) @binding(0) var<storage, read> keys: array<f32>;\n@group(0) @binding(1) var<storage, read_write> cands: array<u32>;\n@group(0) @binding(2) var<uniform> cfg: Cfg;\n@compute @workgroup_size(32, 1, 1)\nfn search(@builtin(global_invocation_id) gid: vec3<u32>) { cands[gid.x] = u32(keys[gid.x]); }\n";
        let v24 = interface_of(src).unwrap();
        let gram = gram_role();
        for name in ["load.phi_safe", "load.phi4_safe", "load.phi_len", "load.h_safe", "load.h_len", "store.h_safe"] {
            let rows: Vec<&WgslRow> = k.rows.iter().filter(|r| safe_base(&r.name) == name).collect();
            assert!(!rows.is_empty(), "{name}");
            assert!(rows.iter().all(|r| !role_spells(&k, &v24, r)), "{name} offered under v24's role");
            assert!(rows.iter().all(|r| role_spells(&k, &gram, r)), "{name} withheld under the Gram's role");
        }
        let t = |n: &str, d: &str| Tree::new(n, dual_ty(&k, d).unwrap(), vec![]);
        let phi4 = Tree::new("load.phi4_safe", dual_ty(&k, "vec4<f32>.real").unwrap(), vec![Tree::literal("literal.index", dual_ty(&k, "u32.index").unwrap(), 0.0)]);
        let table = FunctionTable::safe();
        let r = Renderer::new(&k, &table, &v24);
        let mut scope = Scope { unrolled: BTreeMap::new(), unroll_copies: 1, in_unrolled_copy: false, wg: BTreeMap::new(), unwritten: BTreeSet::new(), in_items: false, in_tile: false, tile_depth: 0, grid: Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false }, in_cond: 0, returned: false, helpers: Vec::new(), masked: Vec::new(), const_depth: 0 };
        let e = r.value(&phi4, &mut scope).expect_err("load.phi4_safe rendered under v24's role");
        assert!(e.starts_with("our software has a bug: load.phi4_safe"), "{e}");
        let len = t("load.h_len", "u32.count");
        let e = r.value(&len, &mut scope).expect_err("load.h_len rendered under v24's role");
        assert!(e.starts_with("our software has a bug: load.h_len"), "{e}");
    }

    /// AN INTEGER LITERAL OF ANY u32 OR i32 ROUND-TRIPS EXACTLY: u32::MAX and i32::MIN are written and read as the values
    /// they are, not as the nearest f32 or as halves of dual bits.
    #[test]
    fn an_integer_literal_of_any_u32_or_i32_round_trips() {
        let (k, t, r) = (WgslKingdom::load_safe(), FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &r);
        let ty = |n: &str| dual_ty(&k, n).unwrap();
        for (set, reg, lit, dual, want) in [
            ("reg.set_i_L0", ("reg.i0", "reg<u32>.index"), "literal.index", "u32.index", f64::from(u32::MAX)),
            ("reg.set_i_L0", ("reg.i0", "reg<u32>.index"), "literal.index", "u32.index", 2_246_822_507.0),
            ("reg.set_i_L0", ("reg.i0", "reg<u32>.index"), "literal.index", "u32.index", 16_777_217.0),
            ("reg.set_n_L0", ("reg.n0", "reg<i32>.int"), "literal.int", "i32.int", f64::from(i32::MIN)),
            ("reg.set_n_L0", ("reg.n0", "reg<i32>.int"), "literal.int", "i32.int", f64::from(i32::MAX)),
        ] {
            let stmt = Tree::new(set, ty("stmt.L0"), vec![Tree::new(reg.0, ty(reg.1), vec![]), Tree::literal(lit, ty(dual), want)]);
            let kernel = Tree::new("kernel.entry_safe", ty("kernel.safe"), vec![Tree::fallback(ty("grid.grid")), Tree::new("block.one_L0", ty("block.L0"), vec![stmt]), Tree::fallback(ty("block.O0"))]);
            let text = rr.kernel(&kernel).unwrap_or_else(|e| panic!("{e}")).text;
            validate_text(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
            let back = crate::wgsl::safe_reader::read_safe(&text, &k).unwrap_or_else(|e| panic!("{want}: does not read back: {e}\n{text}"));
            fn values(t: &Tree, name: &str, out: &mut Vec<f64>) {
                if t.name == name && t.kids.is_empty() {
                    out.push(t.konst);
                }
                t.kids.iter().for_each(|c| values(c, name, out));
            }
            let mut got = Vec::new();
            values(&back.tree, lit, &mut got);
            assert!(got.contains(&want), "{want} read back as {got:?}\n{text}");
        }
    }
}
