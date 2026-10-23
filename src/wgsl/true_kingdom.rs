//! The true multi-typed WGSL kingdom: the generated symbol table of
//! `kingdoms/wgsl_true` (`gen_tables.py`, from `nucleotable/docs/
//! MultiTypedGEP_s9_WGSL_draft.md`) loaded through the kingdom contract of
//! `MultiTypedGEP.md` section 1, a typed initialiser over its layout, and the
//! renderer: a pure substitution of argument texts into each row's spelling.
//!
//! The table is data read at run time (`TrueKingdom::load`), not `include_str!`,
//! so a regenerated table needs no rebuild. A row is `(name, kind, output type,
//! ordered argument types, spelling)`; a constant row is the placeholder `?_T`
//! of its output type and `constants.tsv` is that type's append-only `C_T`.
//!
//! Spelling templates: `{N}` the text of argument N, `{N:F}` field F of that
//! text split on whitespace (a literal such as the grid shape `16 16`), `{v}`
//! the value text a constant reads from its `C_T`. Every other brace is WGSL.
//! The renderer does nothing else: no mask, no clamp, no fallback, no rewrite.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::geneframe::Ty;
use crate::kingdom_contract::{relations_from_rows, KingdomContract, KingdomRelations};
use crate::typed_gep::{GeneLayout, Tree};
use crate::wgsl::loader::{parse_duals, WgslDual};
use crate::wgsl::work_guard::Fuel;

pub mod egraph;

/// The kingdom's name in its relations.
pub const TRUE_WGSL: &str = "TRUE_WGSL";

/// The directory of the generated table.
pub fn default_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("kingdoms/wgsl_true")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    Function,
    Terminal,
    /// The placeholder `?_T` of its output type.
    Constant,
}

#[derive(Debug, Clone)]
pub struct TrueRow {
    pub name: String,
    pub kind: RowKind,
    pub out: Ty,
    pub args: Vec<Ty>,
    pub spelling: String,
    pub note: String,
    /// The declarations of the kernel-scope variables the row's spelling names (a register, a private array, a loop's
    /// counter and bound): its prologue fragment, emitted when the row is in the expressed tree ([`TrueKingdom::declarations_of`]).
    pub declares: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TrueKingdom {
    pub dir: PathBuf,
    pub types: Vec<WgslDual>,
    pub rows: Vec<TrueRow>,
    /// `C_T`: the value texts of each constant-capable type.
    pub constants: BTreeMap<Ty, Vec<String>>,
    pub prologue: String,
    pub declarations: String,
    pub root: Ty,
    /// The interface's fuel budget `K`.
    pub fuel_budget: u32,
    /// For every row, how it repeats the nodes below it ([`LoopRow`]); `None` for a row whose spelling has no loop.
    pub loops: Vec<Option<LoopRow>>,
    /// A module interface (s9 batch 8): the roots of its normal genes, one entry point each, in gene order; empty for
    /// a one-entry interface, whose genes are rooted at [`TrueKingdom::root`].
    pub gene_roots: Vec<Ty>,
    /// A module interface's fixed homeotic linker (design section 6): the link function whose signature is the gene
    /// roots in order and whose output is [`TrueKingdom::root`].
    pub linker: Option<String>,
}

/// How a loop row repeats its arguments, read from its own spelling and types.
///
/// `inside[i]`: argument `i` is spelled inside the loop (its placeholder comes after the row's first `loop {` or
/// `for (`), so it runs once per trip; an argument spelled before the loop (a bound, a stride bound by a `let`) runs
/// once. `cap` is the trip cap the types give: `2^(6c)` for the stratum class `c` the row raises (the output's fuel
/// stratum less the body's, or less the row's own `_f<k>` for a row with no body, the workgroup reductions); a tile
/// row whose bound argument (`u32.lit64`) is a `?_T` constant uses that literal `n` when it is below the cap. A row whose
/// loop holds no argument (a reduction) executes `cap` trips of its own text: its own work is `cap`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopRow {
    pub inside: Vec<bool>,
    pub cap: u128,
    /// The argument that is the literal trip bound (`u32.lit64`), if the row has one.
    pub literal_arg: Option<usize>,
    /// No argument is inside the loop: the row's own text is what repeats.
    pub own_work: bool,
}

/// The fuel stratum `k` of a type or row name ending in `_f<k>`.
fn stratum(name: &str) -> Option<u32> {
    let at = name.rfind("_f")?;
    name[at + 2..].parse().ok()
}

/// The [`LoopRow`] of a row, or `None` when its spelling has no loop.
fn loop_row(row: &TrueRow, type_name: &dyn Fn(Ty) -> String) -> Result<Option<LoopRow>, String> {
    let at = [row.spelling.find("loop {"), row.spelling.find("for (")].into_iter().flatten().min();
    let Some(at) = at else { return Ok(None) };
    let inside: Vec<bool> = (0..row.args.len())
        .map(|i| {
            let (plain, field) = (format!("{{{i}}}"), format!("{{{i}:"));
            let first = [row.spelling.find(&plain), row.spelling.find(&field)].into_iter().flatten().min();
            first.is_some_and(|p| p > at)
        })
        .collect();
    let out_k = stratum(&type_name(row.out)).ok_or_else(|| format!("our software has a bug: loop row {} outputs {}, which has no fuel stratum", row.name, type_name(row.out)))?;
    let body_k = row.args.iter().zip(&inside).filter(|(_, ins)| **ins).filter_map(|(t, _)| stratum(&type_name(*t))).max();
    let in_k = match body_k {
        Some(k) => k,
        None => stratum(&row.name).ok_or_else(|| format!("our software has a bug: loop row {} has no body and no _f<k> in its name", row.name))?,
    };
    let class = out_k.checked_sub(in_k).ok_or_else(|| format!("our software has a bug: loop row {} lowers the stratum ({in_k} to {out_k})", row.name))?;
    let shift = 6 * class;
    let cap = if shift >= 127 { u128::MAX } else { 1u128 << shift };
    let literal_arg = row.args.iter().position(|t| type_name(*t) == "u32.lit64").filter(|i| !inside[*i]);
    let own_work = !inside.iter().any(|b| *b);
    Ok(Some(LoopRow { inside, cap, literal_arg, own_work }))
}

/// `H`: the sum of the head lengths `h_T` over the types `root` reaches, at program size `n`.
pub fn head_total_of(contract: &KingdomContract, root: Ty, n: usize) -> Result<u64, String> {
    let heads = contract.heads_from_program_size(root, n)?;
    let b = contract.bounds(root, &heads)?;
    Ok(b.types.iter().map(|t| t.h as u64).sum())
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('\\') => out.push('\\'),
                Some(o) => {
                    out.push('\\');
                    out.push(o);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn read(dir: &Path, file: &str) -> Result<String, String> {
    std::fs::read_to_string(dir.join(file)).map_err(|e| format!("our software has a bug: {}: {e}", dir.join(file).display()))
}

impl TrueKingdom {
    pub fn load(dir: &Path) -> Result<TrueKingdom, String> {
        let types = parse_duals(&read(dir, "types.tsv")?)?;
        let by_name: BTreeMap<&str, Ty> = types.iter().map(|d| (d.dual.as_str(), d.ty())).collect();
        let ty_of = |n: &str, what: &str| by_name.get(n).copied().ok_or_else(|| format!("our software has a bug: {what}: type {n} is not in types.tsv"));
        let text = read(dir, "functions.tsv")?;
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().ok_or("empty functions.tsv")?.split('\t').collect();
        let want = ["ref", "name", "kind", "out", "args", "spelling", "note", "declares"];
        if header != want {
            return Err(format!("functions.tsv header is {header:?}, want {want:?}"));
        }
        let mut rows = Vec::new();
        for (n, line) in lines.enumerate() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 8 {
                return Err(format!("functions.tsv line {}: {} fields, want 8", n + 2, f.len()));
            }
            let kind = match f[2] {
                "function" => RowKind::Function,
                "terminal" => RowKind::Terminal,
                "constant" => RowKind::Constant,
                o => return Err(format!("functions.tsv line {}: kind {o:?}", n + 2)),
            };
            let args = if f[4].is_empty() { Vec::new() } else { f[4].split(';').map(|a| ty_of(a, f[1])).collect::<Result<Vec<_>, _>>()? };
            let declares = if f[7].is_empty() { Vec::new() } else { unescape(f[7]).lines().map(str::to_string).collect() };
            rows.push(TrueRow { name: f[1].to_string(), kind, out: ty_of(f[3], f[1])?, args, spelling: unescape(f[5]), note: unescape(f[6]), declares });
        }
        let text = read(dir, "constants.tsv")?;
        let mut constants: BTreeMap<Ty, Vec<String>> = BTreeMap::new();
        for (n, line) in text.lines().skip(1).enumerate() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 3 {
                return Err(format!("constants.tsv line {}: {} fields, want 3", n + 2, f.len()));
            }
            let idx: usize = f[1].parse().map_err(|e| format!("constants.tsv line {}: {e}", n + 2))?;
            let v = constants.entry(ty_of(f[0], "constants.tsv")?).or_default();
            if idx != v.len() {
                return Err(format!("our software has a bug: constants.tsv line {}: index {idx} is not the next of {}", n + 2, f[0]));
            }
            v.push(unescape(f[2]));
        }
        let names: Vec<String> = types.iter().map(|d| d.dual.clone()).collect();
        let type_name = |t: Ty| match t {
            Ty::Wgsl(i) => names.get(i as usize).cloned().unwrap_or_default(),
            o => format!("{o:?}"),
        };
        let loops = rows.iter().map(|r| loop_row(r, &type_name)).collect::<Result<Vec<_>, _>>()?;
        let meta = read(dir, "interface.tsv")?;
        let get = |key: &str| meta.lines().filter_map(|l| l.split_once('\t')).find(|(k, _)| *k == key).map(|(_, v)| v.to_string()).ok_or_else(|| format!("interface.tsv has no {key}"));
        let gene_roots = match get("gene_roots") {
            Ok(v) => v.split(';').map(|t| ty_of(t, "interface.tsv gene_roots")).collect::<Result<Vec<_>, _>>()?,
            Err(_) => Vec::new(),
        };
        let linker = get("linker").ok();
        if let Some(l) = &linker {
            let row = rows.iter().find(|r| &r.name == l).ok_or_else(|| format!("our software has a bug: interface.tsv names the linker {l}, which is not a row"))?;
            if row.args != gene_roots {
                return Err(format!("our software has a bug: the linker {l} takes {:?}, the gene roots are {gene_roots:?}", row.args));
            }
        }
        Ok(TrueKingdom {
            gene_roots,
            linker,
            dir: dir.to_path_buf(),
            root: ty_of(&get("root")?, "interface.tsv root")?,
            fuel_budget: get("fuel_budget")?.parse().map_err(|e| format!("interface.tsv fuel_budget: {e}"))?,
            types,
            rows,
            constants,
            prologue: read(dir, "prologue.wgsl")?,
            declarations: read(dir, "declarations.wgsl")?,
            loops,
        })
    }

    /// The exact work bound of one invocation of the kernel `tree` spells, from its types: the sum over the expressed
    /// nodes of the product of the trip caps of the loops enclosing each ([`LoopRow`]); a loop whose text holds no
    /// argument counts its own `cap` trips. Every legal tree obeys it, whatever its uniforms; it is never more than `H * 2^(6K)`. It is a
    /// property of the expressed tree (the work guard's split of a dispatch into submits), never a verdict on the gene.
    pub fn tree_work_bound(&self, tree: &Tree) -> Result<u128, String> {
        let mut total = 0u128;
        let mut stack: Vec<(&Tree, u128)> = vec![(tree, 1)];
        while let Some((node, mult)) = stack.pop() {
            let row = self.rows.get(node.symbol as usize).ok_or_else(|| format!("our software has a bug: symbol {} is not a row", node.symbol))?;
            match self.loops.get(node.symbol as usize).and_then(|l| l.as_ref()) {
                None => {
                    total = total.saturating_add(mult);
                    stack.extend(node.kids.iter().map(|k| (k, mult)));
                }
                Some(l) => {
                    let mut cap = l.cap;
                    if let Some(i) = l.literal_arg {
                        let kid = node.kids.get(i).ok_or_else(|| format!("our software has a bug: {} has no argument {i}", row.name))?;
                        let kid_row = &self.rows[kid.symbol as usize];
                        if let (RowKind::Constant, Some(c)) = (kid_row.kind, kid.const_index) {
                            let text = self.constants.get(&kid_row.out).and_then(|v| v.get(c as usize)).ok_or_else(|| format!("our software has a bug: {}: constant {c} is past C_T", kid_row.name))?;
                            let n: u128 = text.trim_end_matches('u').parse().map_err(|e| format!("our software has a bug: {}: the literal {text:?} is not a count: {e}", kid_row.name))?;
                            cap = cap.min(n);
                        }
                    }
                    total = total.saturating_add(mult.saturating_mul(if l.own_work { cap.max(1) } else { 1 }));
                    for (i, k) in node.kids.iter().enumerate() {
                        let inside = l.inside.get(i).copied().unwrap_or(false);
                        stack.push((k, if inside { mult.saturating_mul(cap) } else { mult }));
                    }
                }
            }
        }
        Ok(total)
    }

    /// [`TrueKingdom::tree_work_bound`] for one run whose uniform fields are known (`fields`: each field's value, by
    /// name): a loop's trips are its bound's value in this run (interval arithmetic over the bound's own rows: the
    /// fields, literals, `+ * min max >> / & -`, the cap `min(.., 2^c)`), divided by its literal stride, never more than
    /// the cap its types give. A bound whose value the rows do not give (a register, a load) keeps the cap. It is an
    /// upper bound of the node executions of one invocation of that run, never more than the type bound.
    pub fn tree_work_bound_in(&self, tree: &Tree, fields: &BTreeMap<String, u64>) -> Result<u128, String> {
        self.tree_work_bound_workgroup_in(tree, fields, 1)
    }

    /// The bound of ONE WORKGROUP of `invocations` invocations of the run [`TrueKingdom::tree_work_bound_in`] bounds:
    /// every node counts once an invocation, except inside the body of a solo phase (`wg.solo_*`, spelled
    /// `if (lidx == 0u) { .. }`), which one invocation of the workgroup runs, so it counts once. With one invocation it
    /// is [`TrueKingdom::tree_work_bound_in`]; it is never more than `invocations` times it.
    pub fn tree_work_bound_workgroup_in(&self, tree: &Tree, fields: &BTreeMap<String, u64>, invocations: u64) -> Result<u128, String> {
        let mut total = 0u128;
        let all = u128::from(invocations.max(1));
        let mut stack: Vec<(&Tree, u128)> = vec![(tree, all)];
        while let Some((node, mult)) = stack.pop() {
            let row = self.rows.get(node.symbol as usize).ok_or_else(|| format!("our software has a bug: symbol {} is not a row", node.symbol))?;
            match self.loops.get(node.symbol as usize).and_then(|l| l.as_ref()) {
                None => {
                    total = total.saturating_add(mult);
                    // a solo phase's first argument is its body: one invocation of the workgroup runs it
                    let solo = row.name.starts_with("wg.solo_");
                    stack.extend(node.kids.iter().enumerate().map(|(i, k)| (k, if solo && i == 0 { (mult / all).max(1) } else { mult })));
                }
                Some(l) => {
                    let mut cap = l.cap;
                    // the bound: the first argument spelled before the loop; the stride: a literal second argument before it
                    let outside: Vec<usize> = (0..node.kids.len()).filter(|i| !l.inside.get(*i).copied().unwrap_or(false)).collect();
                    if let Some(b) = outside.first().and_then(|i| node.kids.get(*i)) {
                        if let Some(n) = self.value_max(b, fields) {
                            let stride = outside.get(1).and_then(|i| node.kids.get(*i)).and_then(|s| self.value_min(s)).filter(|s| *s > 0).unwrap_or(1);
                            cap = cap.min(u128::from(n.div_ceil(stride)));
                        }
                    }
                    total = total.saturating_add(mult.saturating_mul(if l.own_work { cap.max(1) } else { 1 }));
                    for (i, k) in node.kids.iter().enumerate() {
                        let inside = l.inside.get(i).copied().unwrap_or(false);
                        stack.push((k, if inside { mult.saturating_mul(cap) } else { mult }));
                    }
                }
            }
        }
        Ok(total)
    }

    /// A constant node's value, when its `C_T` text is an unsigned integer.
    fn const_value(&self, node: &Tree) -> Option<u64> {
        let row = self.rows.get(node.symbol as usize)?;
        let text = self.constants.get(&row.out)?.get(node.const_index? as usize)?;
        text.trim().trim_end_matches('u').parse().ok()
    }

    fn value_min(&self, node: &Tree) -> Option<u64> {
        let row = self.rows.get(node.symbol as usize)?;
        match row.kind {
            RowKind::Constant => self.const_value(node),
            RowKind::Terminal => row.spelling.trim().trim_end_matches('u').parse().ok(),
            RowKind::Function if row.spelling == "{0}" || row.spelling == "lit_u32({0})" => self.value_min(node.kids.first()?),
            _ => None,
        }
    }

    /// The largest value a u32 node can take in a run whose fields are `fields`, from its rows' spellings; `None` when
    /// the rows do not say.
    fn value_max(&self, node: &Tree, fields: &BTreeMap<String, u64>) -> Option<u64> {
        let row = self.rows.get(node.symbol as usize)?;
        let sp = row.spelling.as_str();
        let kid = |i: usize| node.kids.get(i).and_then(|k| self.value_max(k, fields));
        let r = match row.kind {
            RowKind::Constant => self.const_value(node)?,
            RowKind::Terminal => match sp.strip_prefix("cfg.") {
                Some(f) => *fields.get(f)?,
                None => sp.trim().trim_start_matches("lit_u32(").trim_end_matches(')').trim_end_matches('u').parse().ok()?,
            },
            RowKind::Function => match sp {
                "{0}" | "lit_u32({0})" => kid(0)?,
                "({0} + {1})" => kid(0)?.checked_add(kid(1)?)?,
                "({0} * {1})" => kid(0)?.checked_mul(kid(1)?)?,
                "({0} - {1})" | "({0} / {1})" | "({0} >> {1})" | "sub_floor({0}, {1})" => kid(0)?,
                "({0} & {1})" | "min({0}, {1})" => match (kid(0), kid(1)) {
                    (Some(a), Some(b)) => a.min(b),
                    (Some(a), None) | (None, Some(a)) => a,
                    _ => return None,
                },
                "max({0}, {1})" => kid(0)?.max(kid(1)?),
                o => {
                    // the bound rows' cap: `min({0}, Nu)`
                    let n = o.strip_prefix("min({0}, ").and_then(|t| t.strip_suffix("u)")).and_then(|t| t.parse::<u64>().ok())?;
                    kid(0).map_or(n, |a| a.min(n))
                }
            },
        };
        (r <= u64::from(u32::MAX)).then_some(r)
    }

    /// The work guard's fuel for one expressed tree ([`TrueKingdom::tree_work_bound`]).
    pub fn tree_fuel(&self, tree: &Tree) -> Result<Fuel, String> {
        Ok(Fuel::Tree { per_invocation: self.tree_work_bound(tree)? })
    }

    pub fn type_name(&self, t: Ty) -> String {
        match t {
            Ty::Wgsl(i) => self.types[i as usize].dual.clone(),
            o => format!("{o:?}"),
        }
    }

    /// The two relations of the kingdom contract; the symbol id is the row's position.
    pub fn relations(&self) -> KingdomRelations {
        let rows: Vec<(String, Vec<Ty>, Ty)> = self.rows.iter().map(|r| (r.name.clone(), r.args.clone(), r.out)).collect();
        let constants: Vec<&str> = self.rows.iter().filter(|r| r.kind == RowKind::Constant).map(|r| r.name.as_str()).collect();
        let mut roots = vec![self.root];
        roots.extend(self.gene_roots.iter().copied());
        let mut r = relations_from_rows(TRUE_WGSL, roots, &rows, &constants);
        r.tail_free_roots = true;
        r
    }

    /// `|C_T|` for every constant-capable type.
    pub fn const_lens(&self) -> BTreeMap<Ty, u32> {
        self.constants.iter().map(|(t, v)| (*t, v.len() as u32)).collect()
    }

    /// `H`, the sum of the head lengths over the types the root reaches, at program size `n`.
    pub fn head_total(&self, contract: &KingdomContract, n: usize) -> Result<u64, String> {
        head_total_of(contract, self.root, n)
    }

    /// The work guard's fuel for kernels of this kingdom at program size `n`: `H` and the interface's `K`. Every
    /// device run of a kernel this kingdom produced passes it (`work_guard`).
    pub fn fuel(&self, contract: &KingdomContract, n: usize) -> Result<Fuel, String> {
        Ok(Fuel::Kingdom { h: self.head_total(contract, n)?, k: self.fuel_budget })
    }

    /// The bound table at program size `n` (design section 2): for every type the root reaches its head `h_T`, the
    /// demand `P_T`, the tail `t_T` and the Dc length `d_T`, the leaf bound `L` and the gene width; then the types
    /// the root cannot reach.
    pub fn bound_table(&self, contract: &KingdomContract, n: usize) -> Result<String, String> {
        let heads = contract.heads_from_program_size(self.root, n)?;
        let b = contract.bounds(self.root, &heads)?;
        let mut s = format!(
            "bound table at program size N = {n}: root {}, {} types reachable of {}, L = {}, gene width {} words (heads guarantee N = {})\n  {:<34} {:>5} {:>7} {:>7} {:>7}\n",
            self.type_name(self.root),
            b.types.len(),
            self.types.len(),
            b.l,
            b.width(),
            contract.guaranteed_program_size(self.root, &heads),
            "type",
            "h_T",
            "P_T",
            "t_T",
            "d_T"
        );
        let mut tys: Vec<_> = b.types.iter().collect();
        tys.sort_by_key(|t| (std::cmp::Reverse(t.h + t.t + if t.constant { t.d } else { 0 }), t.ty.code()));
        for t in &tys {
            s.push_str(&format!("  {:<34} {:>5} {:>7} {:>7} {:>7}{}\n", self.type_name(t.ty), t.h, t.p, t.t, t.d, if t.constant { "" } else { "  (no Dc)" }));
        }
        let reach: std::collections::BTreeSet<Ty> = b.types.iter().map(|t| t.ty).collect();
        let dead: Vec<String> = self.types.iter().map(|d| d.ty()).filter(|t| !reach.contains(t)).map(|t| self.type_name(t)).collect();
        s.push_str(&format!("  unreachable from the root ({}): {}\n", dead.len(), dead.join(" ")));
        Ok(s)
    }

    /// Spell an expressed tree: substitute each node's argument texts into its
    /// row's spelling, then put the declarations of the rows the tree holds in the
    /// entry point behind the prologue. Nothing else is done to the text.
    pub fn render(&self, tree: &Tree) -> Result<String, String> {
        Ok(format!("{}{}", self.prologue, self.spell(tree)?))
    }

    /// The declarations a tree's kernel needs: the `declares` fragments of the rows of its nodes, deduplicated, in the
    /// table's order (`declarations.wgsl`). Pure row data: a variable is declared exactly when a row in the tree names
    /// it, zero-initialised as the table writes it. A node that is not a base row (a share row) declares nothing.
    pub fn declarations_of(&self, tree: &Tree) -> Result<String, String> {
        let mut named: BTreeSet<&str> = BTreeSet::new();
        let mut stack = vec![tree];
        while let Some(n) = stack.pop() {
            if let Some(r) = self.rows.get(n.symbol as usize) {
                named.extend(r.declares.iter().map(String::as_str));
            }
            stack.extend(n.kids.iter());
        }
        let out: Vec<&str> = self.declarations.lines().filter(|l| named.contains(l)).collect();
        if out.len() != named.len() {
            let missing: Vec<&&str> = named.iter().filter(|d| !out.contains(d)).collect();
            return Err(format!("our software has a bug: rows declare {missing:?}, which declarations.wgsl does not hold"));
        }
        Ok(out.join("\n"))
    }

    fn spell(&self, node: &Tree) -> Result<String, String> {
        let row = self.rows.get(node.symbol as usize).ok_or_else(|| format!("our software has a bug: symbol {} is not a row", node.symbol))?;
        if node.kids.len() != row.args.len() {
            return Err(format!("our software has a bug: {} has {} children, its signature {}", row.name, node.kids.len(), row.args.len()));
        }
        let kids: Vec<String> = node.kids.iter().map(|k| self.spell(k)).collect::<Result<_, _>>()?;
        let value = match (row.kind, node.const_index) {
            (RowKind::Constant, Some(i)) => Some(
                self.constants
                    .get(&row.out)
                    .and_then(|v| v.get(i as usize))
                    .ok_or_else(|| format!("our software has a bug: {}: constant index {i} is past C_T of {}", row.name, self.type_name(row.out)))?
                    .as_str(),
            ),
            (RowKind::Constant, None) => return Err(format!("our software has a bug: {}: a placeholder without a Dc index", row.name)),
            _ => None,
        };
        let text = substitute(&row.spelling, &kids, value).map_err(|e| format!("our software has a bug: {}: {e}", row.name))?;
        // An entry point's declarations are those of the rows of its own subtree: a one-entry kernel's root holds the
        // whole tree, and each entry of a module (s9 batch 8) declares only what its own gene names.
        if row.spelling.contains("<declarations>") {
            return Ok(text.replace("<declarations>", &self.declarations_of(node)?));
        }
        Ok(text)
    }
}

/// `{N}`, `{N:F}` and `{v}` of a spelling; every other brace is copied.
pub fn substitute(spelling: &str, kids: &[String], value: Option<&str>) -> Result<String, String> {
    let b = spelling.as_bytes();
    let mut out = String::with_capacity(spelling.len() + kids.iter().map(String::len).sum::<usize>());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'{' {
            if let Some(close) = spelling[i..].find('}') {
                let inner = &spelling[i + 1..i + close];
                if inner == "v" {
                    out.push_str(value.ok_or("{v} in a row with no constant")?);
                    i += close + 1;
                    continue;
                }
                let (a, f) = match inner.split_once(':') {
                    Some((a, f)) => (a, Some(f)),
                    None => (inner, None),
                };
                if !a.is_empty() && a.bytes().all(|c| c.is_ascii_digit()) && f.is_none_or(|f| !f.is_empty() && f.bytes().all(|c| c.is_ascii_digit())) {
                    let n: usize = a.parse().map_err(|_| "argument index")?;
                    let text = kids.get(n).ok_or_else(|| format!("{{{inner}}} but only {} arguments", kids.len()))?;
                    match f {
                        None => out.push_str(text),
                        Some(f) => {
                            let f: usize = f.parse().map_err(|_| "field index")?;
                            out.push_str(text.split_whitespace().nth(f).ok_or_else(|| format!("{{{inner}}}: the argument has no field {f}: {text:?}"))?);
                        }
                    }
                    i += close + 1;
                    continue;
                }
            }
        }
        let ch = spelling[i..].chars().next().ok_or("utf8")?;
        out.push(ch);
        i += ch.len_utf8();
    }
    Ok(out)
}

/// splitmix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// How often a head cell draws a function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Share {
    /// A fair coin per cell.
    Half,
    /// In proportion to the functions against the terminals of the cell's type.
    Proportional,
}

/// The host typed initialiser (phylu `typed_init`, the same draws): a head cell
/// is a function or a terminal of its type, a tail cell a terminal of its type
/// drawn uniformly, a Dc cell an index below `|C_T|`; `root_function` makes the
/// root type's first head cell a function. Nothing is masked or redrawn.
pub fn random_gene(k: &KingdomContract, layout: &GeneLayout, const_len: &BTreeMap<Ty, u32>, rng: &mut Rng, share: Share, root_function: bool) -> Vec<u32> {
    use crate::typed_gep::Domain;
    let mut gene = vec![0u32; layout.width];
    for s in layout.spans() {
        let (functions, terminals) = (k.functions_of(s.ty), k.terminals_of(s.ty));
        for (i, cell) in gene[s.off..s.off + s.len].iter_mut().enumerate() {
            *cell = match s.domain {
                Domain::Head => {
                    let force = root_function && s.ty == layout.root && i == 0 && !functions.is_empty();
                    let p = match share {
                        Share::Half => 0.5,
                        Share::Proportional => functions.len() as f64 / (functions.len() + terminals.len()).max(1) as f64,
                    };
                    let function = !functions.is_empty() && (terminals.is_empty() || force || rng.unit() < p);
                    if function {
                        functions[rng.below(functions.len())]
                    } else {
                        terminals[rng.below(terminals.len())]
                    }
                }
                Domain::Tail => terminals[rng.below(terminals.len())],
                Domain::Dc => rng.below(const_len.get(&s.ty).copied().unwrap_or(1).max(1) as usize) as u32,
            };
        }
    }
    gene
}

/// The depth-biased typed initialiser, for the compiler test only: the same draws as [`random_gene`], but a row is
/// chosen in proportion to `weight[row]` (a function among the functions of the cell's type, a terminal among its
/// terminals) and a head cell of a type in `function_p` draws a function with that probability. Nothing is masked or
/// redrawn: every cell still holds a symbol of its domain's type.
pub fn random_gene_weighted(k: &KingdomContract, layout: &GeneLayout, const_len: &BTreeMap<Ty, u32>, rng: &mut Rng, weight: &[f64], function_p: &BTreeMap<Ty, f64>, root_function: bool) -> Vec<u32> {
    use crate::typed_gep::Domain;
    fn cumulative(ids: &[u32], weight: &[f64]) -> Vec<f64> {
        let mut acc = 0.0;
        ids.iter()
            .map(|i| {
                acc += weight[*i as usize].max(1e-9);
                acc
            })
            .collect()
    }
    fn pick(ids: &[u32], cum: &[f64], rng: &mut Rng) -> u32 {
        let x = rng.unit() * cum.last().copied().unwrap_or(1.0);
        ids[cum.partition_point(|c| *c <= x).min(ids.len() - 1)]
    }
    let mut tables: BTreeMap<Ty, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for f in &layout.frames {
        tables.insert(f.ty, (cumulative(k.functions_of(f.ty), weight), cumulative(k.terminals_of(f.ty), weight)));
    }
    let mut gene = vec![0u32; layout.width];
    for s in layout.spans() {
        let (functions, terminals) = (k.functions_of(s.ty), k.terminals_of(s.ty));
        let (fcum, tcum) = &tables[&s.ty];
        for (i, cell) in gene[s.off..s.off + s.len].iter_mut().enumerate() {
            *cell = match s.domain {
                Domain::Head => {
                    let force = root_function && s.ty == layout.root && i == 0 && !functions.is_empty();
                    let p = function_p.get(&s.ty).copied().unwrap_or(0.5);
                    let function = !functions.is_empty() && (terminals.is_empty() || force || rng.unit() < p);
                    if function {
                        pick(functions, fcum, rng)
                    } else {
                        pick(terminals, tcum, rng)
                    }
                }
                Domain::Tail => pick(terminals, tcum, rng),
                Domain::Dc => rng.below(const_len.get(&s.ty).copied().unwrap_or(1).max(1) as usize) as u32,
            };
        }
    }
    gene
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kingdom() -> TrueKingdom {
        TrueKingdom::load(&default_dir()).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn substitution_copies_wgsl_braces_and_fills_arguments() {
        let kids = vec!["16 16".to_string(), "x".to_string()];
        assert_eq!(substitute("{ let r = {1}; w({0:1}) }", &kids, None).unwrap(), "{ let r = x; w(16) }");
        assert_eq!(substitute("lit_u32({v}u)", &[], Some("7")).unwrap(), "lit_u32(7u)");
        assert!(substitute("{2}", &kids, None).is_err());
        assert!(substitute("{v}", &[], None).is_err());
    }

    /// The kingdom check of rev 4 section 1 passes on the generated table: one output per symbol, ordered signatures,
    /// a terminal for every demanded type; the bound table at N = 8 is printed.
    #[test]
    fn the_generated_table_passes_the_kingdom_check_and_prints_its_bound_table() {
        let k = kingdom();
        let r = k.relations();
        let c = r.check().unwrap_or_else(|d| panic!("{} defects, first: {:?}", d.len(), &d[..d.len().min(5)]));
        assert_eq!(c.roots(), &[k.root]);
        eprintln!("{}", crate::kingdom_contract::report(&r, 8));
        eprintln!("{}", k.bound_table(&c, 8).unwrap());
        let heads = c.heads_from_program_size(k.root, 8).unwrap();
        let b = c.bounds(k.root, &heads).unwrap();
        assert!(b.width() > 0);
    }

    /// A small table by hand: one type R, a binary function and a terminal; `h = 1 + (N - 1) * 2`, so at N = 4,
    /// H = 7; with a second type T (R -> T -> R, a unary each) the sum is over both heads.
    #[test]
    fn the_head_total_is_the_sum_of_the_heads_of_a_small_table() {
        let (r, t) = (Ty::Wgsl(0), Ty::Wgsl(1));
        let one = relations_from_rows("one", vec![r], &[("add".into(), vec![r, r], r), ("x".into(), vec![], r)], &[]).check().expect("one type");
        assert_eq!(head_total_of(&one, r, 4).expect("H"), 7);
        let two = relations_from_rows("two", vec![r], &[("f".into(), vec![t], r), ("g".into(), vec![r], t), ("x".into(), vec![], r), ("y".into(), vec![], t)], &[]).check().expect("two types");
        let heads = two.heads_from_program_size(r, 4).expect("heads");
        assert_eq!(head_total_of(&two, r, 4).expect("H"), heads.values().sum::<usize>() as u64);
        assert!(head_total_of(&two, r, 4).expect("H") > 4, "more than N");
    }

    /// The crash configuration: K = 2 (2026-10-21 17:33), the Gram dispatch (the 128 x 64 rung, a 16 x 16
    /// workgroup) is over the cap, and so is even the unit-test dispatch (one workgroup); K = 0 at the unit-test
    /// dispatch is under it. The tables here are the K = 0 tables (a lower H than the K = 2 ones).
    #[test]
    fn the_crash_configuration_k2_with_the_gram_dispatch_is_refused_and_the_unit_test_at_k0_runs() {
        use crate::wgsl::gram::{band_layout, submit_invocations, RUNGS};
        use crate::wgsl::render::Grid;
        use crate::wgsl::work_guard::work_bound;
        let k = kingdom();
        assert!(k.fuel_budget <= 1, "interface.tsv is at fuel_budget 0 or 1 for the compiler and validity test");
        let c = k.relations().check().expect("check");
        let h = k.head_total(&c, 8).expect("H");
        let grid = Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false };
        let (d, m) = RUNGS[0];
        let bands = band_layout(d, m, 1 << 30);
        let rung = submit_invocations(grid, d, bands.band_rows.min(d), bands.item_bands);
        assert_eq!(rung, 64 * 256, "8 x 8 workgroups of 256");
        let crash = Fuel::Kingdom { h, k: 2 };
        let e = crash.check("the Gram rung", rung).expect_err("K = 2 at the Gram dispatch");
        eprintln!("H = {h}: K = 2 at the 128 x 64 rung: {}; K = 0 unit test: {}; {e}", work_bound(h, 2, rung), work_bound(h, 0, 256));
        assert!(k.fuel(&c, 8).expect("fuel").check("the unit test", 256).is_ok(), "the table's K at the unit-test dispatch");
        let at_rung = k.fuel(&c, 8).expect("fuel").check("the Gram rung", rung);
        assert_eq!(at_rung.is_ok(), k.fuel_budget == 0, "K = 0 is under the cap even at the rung, K = 1 is not: {at_rung:?}");
    }

    /// The image instance at K = 2: a literal loop of 5 trips holding a literal loop of 7 trips holding one register
    /// store. Each node counts once per trip of every loop enclosing it: the bounds and the loop nodes themselves are
    /// outside their own loops; the inner body (4 nodes) runs 35 times, the outer body (3 nodes) 5 times.
    #[test]
    fn a_tree_with_nested_literal_loops_is_bounded_by_the_product_of_their_trips() {
        let k = TrueKingdom::load(&default_dir().join("img").join("volumepassage")).expect("the image table");
        assert_eq!(k.fuel_budget, 2);
        let set = node(&k, "reg.set_f_L2_f0", None, vec![node(&k, "reg.rf0", None, vec![]), node(&k, "literal.real", Some("1.0"), vec![])]);
        let inner = node(&k, "loop.tile_L1_f0", None, vec![node(&k, "literal.lit64", Some("7"), vec![]), node(&k, "block.one_L2_f0", None, vec![set])]);
        let outer = node(&k, "loop.tile_L0_f1", None, vec![node(&k, "literal.lit64", Some("5"), vec![]), node(&k, "block.one_L1_f1", None, vec![inner])]);
        let grid = node(&k, "grid.of", None, vec![node(&k, "literal.shape", Some("8 8"), vec![]), node(&k, "literal.blk_col", Some("1"), vec![]), node(&k, "literal.blk_row", Some("1"), vec![]), node(&k, "own.u32_ktile", None, vec![])]);
        let entry = node(&k, "kernel.entry_safe", None, vec![grid, node(&k, "block.one_L0_f2", None, vec![outer]), node(&k, "own.block_O0_f2", None, vec![])]);
        assert_typed(&k, &entry);
        // outside: entry, grid (5), block.one_L0_f2, the outer loop and its bound, the epilogue = 10; outer body: block,
        // inner loop, its bound = 3, x 5; inner body: block, set, register, value = 4, x 35
        assert_eq!(k.tree_work_bound(&entry).expect("bound"), 10 + 3 * 5 + 4 * 35);
        // the same loops with a bound that is not a literal (the domain's terminal) take the type's cap, 64 trips each
        let outer_own = node(&k, "loop.tile_L0_f1", None, vec![node(&k, "own.u32_lit64", None, vec![]), node(&k, "block.one_L1_f1", None, vec![node(&k, "loop.tile_L1_f0", None, vec![node(&k, "own.u32_lit64", None, vec![]), node(&k, "block.one_L2_f0", None, vec![node(&k, "reg.set_f_L2_f0", None, vec![node(&k, "reg.rf0", None, vec![]), node(&k, "literal.real", Some("1.0"), vec![])])])])])]);
        let entry_own = node(&k, "kernel.entry_safe", None, vec![entry.kids[0].clone(), node(&k, "block.one_L0_f2", None, vec![outer_own]), node(&k, "own.block_O0_f2", None, vec![])]);
        assert_typed(&k, &entry_own);
        assert_eq!(k.tree_work_bound(&entry_own).expect("bound"), 10 + 3 * 64 + 4 * 64 * 64);
        // a workgroup reduction of class 2: its array is spelled inside its loop, so the array's node counts 4,096
        // times (the reduction's trips), its element count once
        let red = k.rows.iter().position(|r| r.name == "wg.reduce_add_f_11_L0_f0").expect("a reduction row");
        assert_eq!(k.loops[red].as_ref().map(|l| (l.cap, l.inside.clone(), l.own_work)), Some((4096, vec![true, false], false)));
        let tile = k.rows.iter().position(|r| r.name == "loop.tile_strided_L0_f0").expect("a strided row");
        assert_eq!(k.loops[tile].as_ref().map(|l| (l.inside.clone(), l.literal_arg)), Some((vec![false, false, true], Some(0))), "the stride is bound by a let before the loop");
        let c = k.relations().check().expect("check");
        let h = k.head_total(&c, 8).expect("H");
        assert!(k.tree_work_bound(&entry_own).expect("bound") <= u128::from(h) << 12, "never more than H * 2^(6K)");
    }

    /// Declarations are row data: a tree with no register, array, counter or loop row declares no variable; adding a
    /// store of `rf3` declares `rf3` alone; a literal loop declares its bound and counter, and the encoded volumepassage
    /// declares exactly the variables its text names, each once.
    #[test]
    fn a_tree_declares_only_the_variables_its_rows_name() {
        let k = TrueKingdom::load(&default_dir().join("img").join("volumepassage")).expect("the image table");
        let kf = k.fuel_budget;
        let grid = || node(&k, "own.grid_grid", None, vec![]);
        let vars = |src: &str| src.lines().filter(|l| l.starts_with("var ")).map(str::to_string).collect::<Vec<_>>();
        let bare = node(&k, "kernel.entry_safe", None, vec![grid(), node(&k, &format!("own.block_L0_f{kf}"), None, vec![]), node(&k, &format!("own.block_O0_f{kf}"), None, vec![])]);
        assert_typed(&k, &bare);
        let src = k.render(&bare).expect("render");
        assert_eq!(vars(&src), Vec::<String>::new(), "a tree with no register row declares nothing:\n{src}");
        let set = node(&k, &format!("reg.set_f_L0_f{kf}"), None, vec![node(&k, "reg.rf3", None, vec![]), node(&k, "literal.real", Some("1.0"), vec![])]);
        let one = node(&k, "kernel.entry_safe", None, vec![grid(), node(&k, &format!("block.one_L0_f{kf}"), None, vec![set]), node(&k, &format!("own.block_O0_f{kf}"), None, vec![])]);
        assert_typed(&k, &one);
        assert_eq!(vars(&k.render(&one).expect("render")), vec!["var rf3: f32 = 0.0;".to_string()]);
        let tree = crate::wgsl::cuneus_originals::volumepassage(&k).unwrap_or_else(|g| panic!("table gaps: {g:?}"));
        let src = k.render(&tree).expect("render");
        let declared = vars(&src);
        let mut names: Vec<String> = declared.iter().map(|l| l["var ".len()..l.find(':').expect("a typed var")].to_string()).collect();
        let n = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), n, "each declared once: {declared:?}");
        let body: String = src.lines().filter(|l| !l.starts_with("var ")).collect::<Vec<_>>().join("\n");
        for name in &names {
            assert!(body.split(|c: char| !c.is_ascii_alphanumeric() && c != '_').any(|w| w == name), "{name} is declared and never named");
        }
        assert!(names.contains(&"n1".to_string()) && names.contains(&"k1".to_string()) && !names.contains(&"rf63".to_string()), "{names:?}");
    }

    /// Every child's row outputs the type its parent's signature asks for at that slot.
    fn assert_typed(k: &TrueKingdom, t: &Tree) {
        let row = &k.rows[t.symbol as usize];
        assert_eq!(row.args.len(), t.kids.len(), "{}", row.name);
        for (i, kid) in t.kids.iter().enumerate() {
            let kr = &k.rows[kid.symbol as usize];
            assert_eq!(kr.out, row.args[i], "{} argument {i} is {} of {}, the signature asks for {}", row.name, kr.name, k.type_name(kr.out), k.type_name(row.args[i]));
            assert_typed(k, kid);
        }
    }

    /// A tree by row names; a constant placeholder names its `C_T` text.
    fn node(k: &TrueKingdom, name: &str, value: Option<&str>, kids: Vec<Tree>) -> Tree {
        let i = k.rows.iter().position(|r| r.name == name).unwrap_or_else(|| panic!("no row {name}"));
        let const_index = value.map(|v| k.constants[&k.rows[i].out].iter().position(|t| t == v).unwrap_or_else(|| panic!("{name}: no constant {v}")) as u32);
        Tree { symbol: i as u32, const_index, kids }
    }

    /// `grid.of_rows_x` binds the output's rows to thread x: the ownership rows take the row axis the grid binds, so
    /// a kernel storing 1.0 to its own cell covers every cell of the output, for both grids and every workgroup shape of
    /// the table under test (its `C_T` of `grid.shape`), the non-square ones included.
    #[cfg(feature = "gpu")]
    #[test]
    fn both_grids_cover_every_cell_of_the_output_on_the_device() {
        use crate::wgsl::gram::{phi, Gram};
        use crate::wgsl::oracle::device::Device;
        let k = kingdom();
        let shape_ty = k.rows.iter().find(|r| r.name == "literal.shape").expect("the shape placeholder").out;
        let shapes = k.constants.get(&shape_ty).expect("C_T of grid.shape").clone();
        let square = |s: &str| s.split_whitespace().collect::<Vec<_>>().windows(2).all(|w| w[0] == w[1]);
        assert!(shapes.iter().filter(|s| !square(s)).count() >= 2, "the table has non-square shapes, so a swapped axis would miss cells: {shapes:?}");
        let dev = Device::new().expect("device");
        let (d, m) = (32usize, 4usize);
        // four row bands of 8 rows: the output is not square per dispatch, so a swapped axis misses cells
        let band_limit = (d * 4 * 8) as u64;
        let mut cases = Vec::new();
        for shape in &shapes {
            cases.push(("grid.of", shape.as_str(), "literal.blk_col", "literal.blk_row"));
            cases.push(("grid.of_rows_x", shape.as_str(), "literal.blk_row", "literal.blk_col"));
        }
        let kf = k.fuel_budget;
        for (grid, shape, first, second) in cases {
            let g = node(&k, grid, None, vec![node(&k, "literal.shape", Some(shape), vec![]), node(&k, first, Some("1"), vec![]), node(&k, second, Some("1"), vec![]), node(&k, "own.u32_ktile", None, vec![])]);
            let store = node(
                &k,
                &format!("store.h_O0_f{kf}"),
                None,
                vec![
                    node(&k, "safe.rowbase_as_row_h", None, vec![node(&k, "own.u32_rowbase_h", None, vec![])]),
                    node(&k, "safe.colbase_as_col_h", None, vec![node(&k, "own.u32_colbase_h", None, vec![])]),
                    node(&k, "literal.real", Some("1.0"), vec![]),
                ],
            );
            let entry = node(&k, "kernel.entry_safe", None, vec![g, node(&k, &format!("own.block_L0_f{kf}"), None, vec![]), node(&k, &format!("block.one_O0_f{kf}"), None, vec![store])]);
            assert_typed(&k, &entry);
            let src = k.render(&entry).expect("render");
            let gram = Gram::compile(&dev, &src, k.tree_fuel(&entry).expect("the tree's bound")).unwrap_or_else(|e| panic!("{grid} {shape}: {e}"));
            let built = gram.build(&phi(d, m, 3), d, m, band_limit).expect("run");
            let missed = built.h.iter().filter(|v| **v != 1.0).count();
            assert_eq!(missed, 0, "{grid} {shape}: {missed} of {} cells not written", d * d);
        }
    }
}
