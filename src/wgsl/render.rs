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

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::geneframe::Ty;
use crate::wgsl::loader::{WgslDual, WgslKingdom, WgslRow};
use crate::wgsl::table::FunctionTable;

/// A typed tree. `name` is a row name (`arith.add`), a per-kernel row
/// (`load.phi`), a `literal.*` row with `konst`, or `fallback`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    pub name: String,
    pub ty: Ty,
    pub konst: f32,
    pub kids: Vec<Tree>,
}

impl Tree {
    pub fn new(name: &str, ty: Ty, kids: Vec<Tree>) -> Tree {
        Tree { name: name.to_string(), ty, konst: 0.0, kids }
    }

    pub fn literal(name: &str, ty: Ty, konst: f32) -> Tree {
        Tree { name: name.to_string(), ty, konst, kids: Vec::new() }
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

    /// `(name kids…)` with literals as `(literal.real 1.5)`.
    pub fn to_sexpr(&self) -> String {
        if self.kids.is_empty() {
            if self.name.starts_with("literal.") {
                return format!("({} {})", self.name, self.konst);
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

/// The fixed part of a kernel: its bindings (the prologue text), the
/// expression bounding the item loop, and its per-kernel rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Role {
    pub name: String,
    pub prologue: String,
    /// The item loop's bound, a `u32` expression over the bindings (`cfg.m`).
    pub items: String,
    pub per_kernel: Vec<PerKernel>,
}

impl Role {
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
    let pk = |name: &str, inputs: &[&str], output: &str, template: &str| PerKernel { name: name.into(), inputs: inputs.iter().map(|s| s.to_string()).collect(), output: output.into(), template: template.into() };
    Role {
        name: "gram".into(),
        prologue: "struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };\n@group(0) @binding(0) var<storage, read> phi: array<f32>;\n@group(0) @binding(1) var<storage, read_write> h: array<f32>;\n@group(0) @binding(2) var<uniform> cfg: Cfg;".into(),
        items: "cfg.m".into(),
        per_kernel: vec![
            pk("load.phi", &["u32.index"], "f32.real", "phi[{0}]"),
            pk("load.phi4", &["u32.index"], "vec4<f32>.real", "vec4<f32>(phi[{0}], phi[{0} + 1u], phi[{0} + 2u], phi[{0} + 3u])"),
            pk("load.h", &["u32.index"], "f32.real", "h[{0}]"),
            pk("store.h", &["u32.index", "f32.real"], "stmt.stmt", "h[{0}] = {1};"),
            pk("load.cfg_d", &[], "u32.count", "cfg.d"),
            pk("load.cfg_m", &[], "u32.count", "cfg.m"),
            pk("load.cfg_r_rows", &[], "u32.count", "cfg.r_rows"),
            pk("load.cfg_r_off", &[], "u32.index", "cfg.r_off"),
        ],
    }
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

/// The grid a rendered kernel asks the driver for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub x: u32,
    pub y: u32,
    pub bx: u32,
    pub by: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    pub text: String,
    pub grid: Grid,
    /// Composite helpers the text uses, by name.
    pub helpers: Vec<String>,
    /// Statements the renderer masked to keep the text valid (a barrier
    /// in non-uniform control flow), each named; the typed projection's
    /// masking, at the statement level.
    pub masked: Vec<String>,
}

pub const WGDIM: [u32; 4] = [4, 8, 16, 32];
pub const BLK: [u32; 3] = [1, 2, 4];
pub const UNROLL: [u32; 3] = [2, 4, 8];
pub const WGMEM_LEN: u32 = 4096;
pub const MAX_THREADS: u32 = 1024;

fn snap(set: &[u32], v: f32) -> u32 {
    let v = if v.is_finite() { v } else { 0.0 };
    *set.iter().min_by(|a, b| (f64::from(**a) - f64::from(v)).abs().partial_cmp(&(f64::from(**b) - f64::from(v)).abs()).expect("finite")).expect("a non-empty set")
}

pub struct Renderer<'a> {
    pub kingdom: &'a WgslKingdom,
    pub table: &'a FunctionTable,
    pub role: &'a Role,
    per_kernel: BTreeMap<&'a str, &'a PerKernel>,
}

/// What the renderer is inside while walking statements.
#[derive(Default)]
struct Scope {
    in_items: bool,
    in_tile: bool,
    /// Inside an `if`: control flow naga's uniformity analysis treats as
    /// non-uniform, where a barrier is refused.
    in_cond: usize,
    /// A `return` has been emitted: everything after it is non-uniform.
    returned: bool,
    helpers: Vec<String>,
    masked: Vec<String>,
}

impl<'a> Renderer<'a> {
    pub fn new(kingdom: &'a WgslKingdom, table: &'a FunctionTable, role: &'a Role) -> Renderer<'a> {
        Renderer { kingdom, table, role, per_kernel: role.per_kernel.iter().map(|p| (p.name.as_str(), p)).collect() }
    }

    fn dual(&self, ty: Ty) -> Result<&WgslDual, String> {
        self.kingdom.dual(ty).ok_or_else(|| format!("{ty:?} is not a dual of the kingdom"))
    }

    /// Render a whole kernel: the root must be `kernel.entry` or a
    /// `fallback` of the kernel dual (an empty kernel).
    pub fn kernel(&self, root: &Tree) -> Result<Rendered, String> {
        let d = self.dual(root.ty)?;
        if d.dual != "kernel.kernel" {
            return Err(format!("the root is {} ({}), not kernel.kernel", root.name, d.dual));
        }
        let mut scope = Scope::default();
        let (grid, body) = match root.name.as_str() {
            "kernel.entry" => {
                let [g, b] = root.kids.as_slice() else { return Err(format!("kernel.entry with {} children", root.kids.len())) };
                (self.grid(g)?, self.block(b, &mut scope)?)
            }
            "fallback" => (Grid { x: 16, y: 16, bx: 1, by: 1 }, String::new()),
            other => return Err(format!("{other} is not a kernel row")),
        };
        if grid.x * grid.y > MAX_THREADS {
            return Err(format!("workgroup {} x {} exceeds {MAX_THREADS} threads", grid.x, grid.y));
        }
        let mut text = String::new();
        text.push_str(&self.role.prologue);
        text.push('\n');
        text.push_str(&format!("var<workgroup> wg_a: array<f32, {WGMEM_LEN}>;\nvar<workgroup> wg_b: array<f32, {WGMEM_LEN}>;\n"));
        // Literals pass through these so naga's constant evaluator never sees
        // a literal-only expression: a NaN, a division or a remainder by a
        // literal zero would otherwise be refused at validation, while on the
        // device they are values the criterion judges.
        text.push_str("fn lit_f32(x: f32) -> f32 { return x; }\nfn lit_u32(x: u32) -> u32 { return x; }\nfn lit_i32(x: i32) -> i32 { return x; }\n");
        scope.helpers.sort();
        scope.helpers.dedup();
        for h in &scope.helpers {
            let (_, body) = COMPOSITES.iter().find(|(n, _)| n == h).ok_or_else(|| format!("composite {h} has no body"))?;
            text.push_str(body);
            text.push('\n');
        }
        text.push_str(&format!("@compute @workgroup_size({}, {}, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {{\n", grid.x, grid.y));
        text.push_str("    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;\n    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0);\n    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u;\n    var b0: bool = false;\n    var t: u32 = 0u; var k: u32 = 0u;\n");
        text.push_str(&body);
        text.push_str("}\n");
        Ok(Rendered { text, grid, helpers: scope.helpers, masked: scope.masked })
    }

    fn grid(&self, g: &Tree) -> Result<Grid, String> {
        match g.name.as_str() {
            "grid.of" => {
                let [x, y, bx, by] = g.kids.as_slice() else { return Err(format!("grid.of with {} children", g.kids.len())) };
                Ok(Grid { x: self.code(x, &WGDIM)?, y: self.code(y, &WGDIM)?, bx: self.code(bx, &BLK)?, by: self.code(by, &BLK)? })
            }
            "fallback" => Ok(Grid { x: 16, y: 16, bx: 1, by: 1 }),
            other => Err(format!("{other} is not a grid row")),
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

    /// A block's text and whether it ends in a return on every path: naga
    /// refuses instructions after a `return`, so a block stops at one and
    /// what follows in the gene is unreachable and dropped.
    fn block_t(&self, b: &Tree, scope: &mut Scope) -> Result<(String, bool), String> {
        match b.name.as_str() {
            "fallback" => Ok((String::new(), false)),
            "block.one" | "block.seq2" | "block.seq3" | "block.seq4" => {
                let mut out = String::new();
                for k in &b.kids {
                    let (text, terminated) = self.stmt_t(k, scope)?;
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
        match s.name.as_str() {
            "stmt.return" => Ok((self.stmt(s, scope)?, true)),
            "stmt.block" if kids.len() == 1 => {
                let (b, t) = self.block_t(&kids[0], scope)?;
                Ok((format!("    {{\n{b}    }}\n"), t))
            }
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

    fn stmt(&self, s: &Tree, scope: &mut Scope) -> Result<String, String> {
        let kids = s.kids.as_slice();
        let n = |want: usize| -> Result<(), String> { if kids.len() == want { Ok(()) } else { Err(format!("{} with {} children, {want} expected", s.name, kids.len())) } };
        match s.name.as_str() {
            "fallback" => Ok(String::new()),
            "stmt.block" => {
                n(1)?;
                Ok(format!("    {{\n{}    }}\n", self.block(&kids[0], scope)?))
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
                if terminated {
                    // A body that returns runs once; unrolling it would put
                    // the step after the return.
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
            "reg.set_f" | "reg.set_v" | "reg.set_i" | "reg.set_b" => {
                n(2)?;
                Ok(format!("    {} = {};\n", self.register(&kids[0])?, self.value(&kids[1], scope)?))
            }
            "wg.set" => {
                n(3)?;
                Ok(format!("    {}[({}) & {}u] = {};\n", self.wgmem(&kids[0])?, self.value(&kids[1], scope)?, WGMEM_LEN - 1, self.value(&kids[2], scope)?))
            }
            other => {
                if let Some(p) = self.per_kernel.get(other) {
                    let d = self.dual(s.ty)?;
                    if d.dual != "stmt.stmt" {
                        return Err(format!("{other} is a value row used as a statement"));
                    }
                    return Ok(format!("    {}\n", self.fill(p, s, scope)?));
                }
                Err(format!("{other} is not a statement row"))
            }
        }
    }

    fn register(&self, r: &Tree) -> Result<&'static str, String> {
        let d = self.dual(r.ty)?;
        Ok(match (r.name.as_str(), d.dual.as_str()) {
            ("reg.f0", _) => "f0",
            ("reg.f1", _) => "f1",
            ("reg.f2", _) => "f2",
            ("reg.f3", _) => "f3",
            ("reg.v0", _) => "v0",
            ("reg.v1", _) => "v1",
            ("reg.i0", _) => "i0",
            ("reg.i1", _) => "i1",
            ("reg.i2", _) => "i2",
            ("reg.i3", _) => "i3",
            ("reg.b0", _) => "b0",
            ("fallback", "reg<f32>.real") => "f0",
            ("fallback", "reg<vec4<f32>>.real") => "v0",
            ("fallback", "reg<u32>.index") => "i0",
            ("fallback", "reg<bool>.flag") => "b0",
            (other, dual) => return Err(format!("{other} ({dual}) is not a register")),
        })
    }

    fn wgmem(&self, w: &Tree) -> Result<&'static str, String> {
        Ok(match w.name.as_str() {
            "wg.a" | "fallback" => "wg_a",
            "wg.b" => "wg_b",
            other => return Err(format!("{other} is not a workgroup array")),
        })
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

    /// A value expression, parenthesised where an operator could bind.
    fn value(&self, v: &Tree, scope: &mut Scope) -> Result<String, String> {
        let d = self.dual(v.ty)?;
        let name = v.name.as_str();
        if name == "fallback" {
            return self.fallback_value(d);
        }
        if let Some(form) = name.strip_prefix("literal.") {
            return self.literal(form, v.konst, d);
        }
        if let Some(p) = self.per_kernel.get(name) {
            return self.fill(p, v, scope);
        }
        match name {
            "reg.get_f" | "reg.get_v" | "reg.get_i" | "reg.get_b" => return Ok(self.register(&v.kids[0])?.into()),
            "wg.get" => return Ok(format!("{}[({}) & {}u]", self.wgmem(&v.kids[0])?, self.value(&v.kids[1], scope)?, WGMEM_LEN - 1)),
            _ => {}
        }
        let kids: Result<Vec<String>, String> = v.kids.iter().map(|k| self.value(k, scope)).collect();
        let a = kids?;
        match name {
            "builtin.global_invocation_id" => return Ok("gid".into()),
            "builtin.local_invocation_id" => return Ok("lid".into()),
            "builtin.workgroup_id" => return Ok("wid".into()),
            "builtin.local_invocation_index" => return Ok("lidx".into()),
            "builtin.num_workgroups" => return Ok("nwg".into()),
            "builtin.item" => return Ok("t".into()),
            "builtin.tile" => return Ok("k".into()),
            _ => {}
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
                "LogicalAnd" => "&&",
                "LogicalOr" => "||",
                "ShiftLeft" => "<<",
                "ShiftRight" => ">>",
                other => return Err(format!("{name}: binary operator {other} has no text")),
            };
            // A shift count is masked, as WGSL requires it below the width.
            if op == "ShiftLeft" || op == "ShiftRight" {
                return Ok(format!("({} {sym} (({}) & 31u))", a[0], a[1]));
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
                let i: usize = template.instance.strip_prefix("access_").and_then(|n| n.parse().ok()).ok_or_else(|| format!("{name}: not an access"))?;
                Ok(format!("{}.{}", a[0], ["x", "y", "z", "w"][i]))
            }
            "composite" => self.composite(name, template.instance.as_str(), &a, d, scope),
            other => Err(format!("{name}: naga column {other} has no text")),
        }
    }

    fn composite(&self, name: &str, instance: &str, a: &[String], out: &WgslDual, scope: &mut Scope) -> Result<String, String> {
        // Inline composites first; the rest call a helper from the library.
        let inline = match name {
            "index.of_int" => Some(format!("u32(max({}, 0))", a[0])),
            "index.of_count" => Some(a[0].clone()),
            "index.clamp_to" => Some(format!("min({}, max({}, 1u) - 1u)", a[0], a[1])),
            "index.band_of" => Some(format!("({} / max({}, 1u))", a[0], a[1])),
            "count.sub_floor" => Some(format!("select({0} - {1}, 0u, {1} > {0})", a[0], a[1])),
            "convert.q15_16_to_f32" => Some(format!("(f32({}) * (1.0 / 65536.0))", a[0])),
            "convert.f32_to_q15_16" => Some(format!("i32(clamp(round({} * 65536.0), -2147483648.0, 2147483520.0))", a[0])),
            "convert.sign_to_real" => Some(format!("select(-1.0, 1.0, {})", a[0])),
            "convert.hash32_to_bits" => Some(a[0].clone()),
            "convert.hash32_to_unit" => Some(format!("(f32({} >> 8u) * (1.0 / 16777216.0))", a[0])),
            "select.negate_if" => Some(format!("select(-({0}), {0}, {1})", a[0], a[1])),
            "quantise.pack4_code8" => Some(format!("(({} & 255u) | (({} & 255u) << 8u) | (({} & 255u) << 16u))", a[0], a[1], a[2])),
            "quantise.pack4_code8_hi" => Some(format!("(({} & 16777215u) | (({} & 255u) << 24u))", a[0], a[1])),
            "quantise.unpack_code8" => Some(format!("(({} >> (({} & 3u) * 8u)) & 255u)", a[0], a[1])),
            "bits.insert_lo" => Some(format!("insertBits({}, {}, ({2}) & 31u, 32u - (({2}) & 31u))", a[0], a[1], a[2])),
            "bits.insert_hi" => Some(format!("select({0} & ((1u << ({1} & 31u)) - 1u), {0}, {1} >= 32u)", a[0], a[1])),
            "bits.pack_sign" => Some(format!("select(0u, 1u, {})", a[0])),
            "bits.unpack_sign" => Some(format!("((({} >> ({} & 31u)) & 1u) == 1u)", a[0], a[1])),
            "hash.combine" => Some(format!("({} ^ {})", a[0], a[1])),
            "hash.split_lo" => Some(format!("{}.x", a[0])),
            "hash.split_hi" => Some(format!("{}.y", a[0])),
            _ => None,
        };
        if let Some(text) = inline {
            return Ok(text);
        }
        if let Some(k) = instance.strip_prefix("real_to_code") {
            let levels: f32 = k.parse::<f32>().map_err(|_| format!("{name}: codebook size"))?;
            return Ok(format!("u32(clamp(round((clamp({}, -1.0, 1.0) + 1.0) * {:?}), 0.0, {:?}))", a[0], (levels - 1.0) / 2.0, levels - 1.0));
        }
        if let Some(k) = instance.strip_suffix("_to_real") {
            let levels: f32 = k.strip_prefix("code").and_then(|n| n.parse().ok()).ok_or_else(|| format!("{name}: codebook size"))?;
            return Ok(format!("(f32(min({}, {}u)) * {:?} - 1.0)", a[0], levels as u32 - 1, 2.0 / (levels - 1.0)));
        }
        let helper = match name {
            "fixed.add_sat" => "fx_add_sat",
            "fixed.sub_sat" => "fx_sub_sat",
            "fixed.mul" => "fx_mul",
            "fixed.shl" => "fx_shl",
            "fixed.shr" => "fx_shr",
            "fixed.wide_mul" => "fx_wide_mul",
            "fixed.wide_add_sat" => "fx_wide_add_sat",
            "fixed.narrow" => "fx_narrow",
            "hash.mix64" => "hash_mix64",
            "hash.fmix32" => "hash_fmix32",
            "convert.hash32_to_index" => "hash32_to_index",
            other => return Err(format!("{other}: composite with no WGSL body (out {})", out.dual)),
        };
        // Helpers call helpers: declare their dependencies too.
        for dep in COMPOSITES.iter().filter(|(n, _)| *n == helper).flat_map(|(_, body)| COMPOSITES.iter().map(|(n, _)| *n).filter(|n| *n != helper && body.contains(&format!("{n}(")))) {
            scope.helpers.push(dep.to_string());
        }
        scope.helpers.push(helper.to_string());
        Ok(format!("{helper}({})", a.join(", ")))
    }

    fn literal(&self, form: &str, v: f32, d: &WgslDual) -> Result<String, String> {
        let scalar = match form {
            "real" => {
                if !v.is_finite() {
                    return Err(format!("literal.real {v} is not finite"));
                }
                if d.scalar() == Some("f16") {
                    return Err("literal.real over f16: the prologue does not enable f16".into());
                }
                format!("lit_f32({v:?}f)")
            }
            "index" | "count" | "bits" | "hash32" | "code4" | "code8" | "code16" | "code1024" => format!("lit_u32({}u)", if v.is_finite() { v.clamp(0.0, 4_294_967_040.0) as u32 } else { 0 }),
            "wgdim" => format!("lit_u32({}u)", snap(&WGDIM, v)),
            "blk" => format!("lit_u32({}u)", snap(&BLK, v)),
            "unroll" => format!("lit_u32({}u)", snap(&UNROLL, v)),
            "int" | "q15_16" => format!("lit_i32({})", v.clamp(-2_147_483_648.0, 2_147_483_520.0) as i32),
            "flag" | "sign" => (v != 0.0).to_string(),
            other => return Err(format!("literal.{other} has no text")),
        };
        if d.lanes > 1 && d.slot.starts_with("vec") {
            return Ok(format!("{}({scalar})", slot_ctor(&d.slot)?));
        }
        Ok(scalar)
    }

    fn fallback_value(&self, d: &WgslDual) -> Result<String, String> {
        if d.is_statement_level() {
            return Err(format!("a {} fallback where a value is needed", d.dual));
        }
        // The zero-value constructor for a vector or matrix; a scalar through
        // lit_* so naga folds nothing.
        if d.slot.starts_with("vec") || d.slot.starts_with("mat") {
            if d.scalar() == Some("f16") {
                return Err("f16 fallback: the prologue does not enable f16".into());
            }
            let lane = match d.scalar() {
                Some("f32") => "lit_f32(0.0f)".to_string(),
                Some("i32") => "lit_i32(0)".to_string(),
                Some("u32") => format!("lit_u32({}u)", d.fallback_bits),
                Some("bool") => (d.fallback_bits != 0).to_string(),
                _ => return Err(format!("{}: no value fallback", d.dual)),
            };
            if let Some((c, r)) = d.matrix_shape() {
                let col = format!("vec{r}<f32>({lane})");
                return Ok(format!("{}({})", slot_ctor(&d.slot)?, vec![col; c as usize].join(", ")));
            }
            return Ok(format!("{}({lane})", slot_ctor(&d.slot)?));
        }
        Ok(match d.scalar() {
            Some("f32") => "lit_f32(0.0f)".to_string(),
            Some("f16") => return Err("f16 fallback: the prologue does not enable f16".into()),
            Some("i32") => "lit_i32(0)".to_string(),
            Some("u32") => format!("lit_u32({}u)", d.fallback_bits),
            Some("bool") => (d.fallback_bits != 0).to_string(),
            _ => return Err(format!("{}: no value fallback", d.dual)),
        })
    }
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
    ("inverse_2", "fn inverse_2(m: mat2x2<f32>) -> mat2x2<f32> {\n    let inv_det = 1.0 / determinant(m);\n    return mat2x2<f32>(vec2<f32>(m[1][1], -m[0][1]) * inv_det, vec2<f32>(-m[1][0], m[0][0]) * inv_det);\n}"),
    ("inverse_3", "fn inverse_3(m: mat3x3<f32>) -> mat3x3<f32> {\n    // Columns of the adjugate: cross products of the columns, transposed.\n    let c0 = cross(m[1], m[2]);\n    let c1 = cross(m[2], m[0]);\n    let c2 = cross(m[0], m[1]);\n    let inv_det = 1.0 / dot(m[0], c0);\n    return transpose(mat3x3<f32>(c0, c1, c2)) * inv_det;\n}"),
    ("inverse_4", "fn inverse_4(m: mat4x4<f32>) -> mat4x4<f32> {\n    // The adjugate from 2x2 sub-determinants (column-major m[col][row]).\n    let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];\n    let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];\n    let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];\n    let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];\n    let b00 = a00 * a11 - a01 * a10; let b01 = a00 * a12 - a02 * a10; let b02 = a00 * a13 - a03 * a10;\n    let b03 = a01 * a12 - a02 * a11; let b04 = a01 * a13 - a03 * a11; let b05 = a02 * a13 - a03 * a12;\n    let b06 = a20 * a31 - a21 * a30; let b07 = a20 * a32 - a22 * a30; let b08 = a20 * a33 - a23 * a30;\n    let b09 = a21 * a32 - a22 * a31; let b10 = a21 * a33 - a23 * a31; let b11 = a22 * a33 - a23 * a32;\n    let inv_det = 1.0 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);\n    return mat4x4<f32>(\n        vec4<f32>(a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09, a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03),\n        vec4<f32>(a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07, a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01),\n        vec4<f32>(a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06, a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00),\n        vec4<f32>(a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06, a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00)) * inv_det;\n}"),
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
pub fn sample(kingdom: &WgslKingdom, role: &Role, root: Ty, max_depth: usize, seed: u64) -> Result<Tree, String> {
    let f16 = |t: &Ty| kingdom.dual(*t).is_some_and(|d| d.scalar() == Some("f16"));
    let mut rows: Vec<WgslRow> = kingdom.rows.iter().filter(|r| !f16(&r.output) && !r.inputs.iter().any(f16)).cloned().collect();
    rows.extend(role.rows(kingdom)?);
    let mut by_out: BTreeMap<Ty, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        let e = by_out.entry(r.output).or_default();
        if r.terminal {
            e.1.push(i);
        } else {
            e.0.push(i);
        }
    }
    let mut state = seed ^ 0x5DEE_CE66_D1CE_4E5B;
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn grow(rows: &[WgslRow], by_out: &BTreeMap<Ty, (Vec<usize>, Vec<usize>)>, kingdom: &WgslKingdom, demand: Ty, depth: usize, max_depth: usize, state: &mut u64) -> Tree {
        let Some((functions, terminals)) = by_out.get(&demand) else { return Tree::fallback(demand) };
        let pick_terminal = depth >= max_depth || functions.is_empty() || (!terminals.is_empty() && next(state).is_multiple_of(3));
        if pick_terminal {
            if terminals.is_empty() {
                return Tree::fallback(demand);
            }
            let r = &rows[terminals[(next(state) % terminals.len() as u64) as usize]];
            if r.name.starts_with("literal.") {
                let form = kingdom.dual(demand).map(|d| d.form.as_str()).unwrap_or("");
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
        let r = &rows[functions[(next(state) % functions.len() as u64) as usize]];
        let kids = r.inputs.iter().map(|&t| grow(rows, by_out, kingdom, t, depth + 1, max_depth, state)).collect();
        Tree::new(&r.name, demand, kids)
    }
    Ok(grow(&rows, &by_out, kingdom, root, 0, max_depth, &mut state))
}

/// Parse text with naga and validate it; the error names the line.
pub fn validate_text(text: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(text).map_err(|e| e.emit_to_string(text))?;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).map(|_| ()).map_err(|e| e.emit_to_string(text))
}

impl Tree {
    /// A short description for a refusal ledger.
    pub fn head(&self) -> String {
        let mut s = String::new();
        let _ = write!(s, "{} ({} nodes, depth {})", self.name, self.size(), self.depth());
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (WgslKingdom, FunctionTable, Role) {
        (WgslKingdom::load(), FunctionTable::shipped(), gram_role())
    }

    #[test]
    fn the_shipped_kernel_as_a_tree_renders_and_validates() {
        let (k, t, r) = setup();
        let tree = gram_shipped_tree(&k).unwrap();
        assert_eq!(tree.size(), 82, "{}", tree.to_sexpr());
        let out = Renderer::new(&k, &t, &r).kernel(&tree).unwrap();
        assert_eq!(out.grid, Grid { x: 16, y: 16, bx: 1, by: 1 });
        assert!(out.text.contains("for (t = 0u; t < cfg.m; t = t + 1u)"), "{}", out.text);
        assert!(out.text.contains("h[(((i1 * cfg.d) + i2))] = ((h[(((i1 * cfg.d) + i2))] + f0));"), "{}", out.text);
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
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
        // The largest grid the codes allow is 32 x 32 = 1,024 threads, the limit; a literal above the set snaps down.
        let big = Tree::new("kernel.entry", ty("kernel.kernel"), vec![Tree::new("grid.of", ty("grid.grid"), vec![Tree::literal("literal.wgdim", ty("u32.wgdim"), 64.0), Tree::literal("literal.wgdim", ty("u32.wgdim"), 64.0), Tree::fallback(ty("u32.blk")), Tree::fallback(ty("u32.blk"))]), one(set_f())]);
        assert_eq!(rr.kernel(&big).unwrap().grid, Grid { x: 32, y: 32, bx: 1, by: 1 });
        // An empty kernel (the root's fallback) is a kernel.
        let empty = rr.kernel(&Tree::fallback(ty("kernel.kernel"))).unwrap();
        validate_text(&empty.text).unwrap();
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
                            let _ = std::fs::write(format!("{dir}/wgsl_random_invalid_{seed}.wgsl"), format!("// {e}\n{}", out.text));
                        }
                        let brief: String = e.lines().filter(|l| !l.trim().is_empty()).take(6).collect::<Vec<_>>().join(" | ");
                        invalid.push(format!("seed {seed} {}: {}", tree.head(), &brief[..brief.len().min(400)]));
                    }
                }
            }
        }
        println!("{n} random kernels, mean {} nodes: {rendered} rendered ({masked} barriers masked), {} refused by construction {refused:?}, {} invalid", sizes / n as usize, n as usize - rendered, invalid.len());
        assert!(invalid.is_empty(), "{} of {rendered} rendered kernels fail naga:\n{}", invalid.len(), invalid.join("\n"));
        // The only refusal by construction is a nested long loop (m² iterations).
        assert!(refused.keys().all(|k| k.contains("nested")), "{refused:?}");
        assert!(rendered >= n as usize * 85 / 100, "only {rendered} of {n} rendered: {refused:?}");
    }
}
