//! ST7, share-equivalence on the true multi-typed WGSL kingdom
//! (`docs/PLAN_true_gep_wgsl.md`): the expressed tree of a gene becomes a
//! DAG-shaped one, in two places.
//!
//! **Within one kernel: a `let`.** A subtree the kernel text spells more than
//! once is spelled once, as a WGSL `let` at function scope, and referenced by
//! its name everywhere else. The binding and the reference are typed rows of
//! the table (`kingdoms/wgsl_true/share.tsv`, written by `gen_share.py`):
//! `share.let.<dual>.<body>(share.name, <dual>, <body>) -> <body>` spelled
//! `let {0}: <slot> = {1};\n{2}`, `share.ref.<dual>(share.name) -> <dual>`
//! spelled `{0}`, and the constant `share.name` whose `C_T` is the names. The
//! shared tree is therefore a tree of rows whose every node fills a slot of its
//! own type ([`Share::check_types`]), and the renderer only substitutes
//! argument texts. The rows are not in the evolution alphabet: the share pass
//! writes them into an expressed tree and [`Share::unshare`] removes them, and
//! the round trip is the identity (tested on random genes).
//!
//! Which subtrees may be bound is a property of the rows, derived here from the
//! table the way the kingdom check derives `a_{T,U}`: a row is **context-free**
//! when its value is the same wherever the kernel body spells it. Its output and
//! every argument are value duals (a WGSL scalar, vector or matrix slot), its
//! spelling holds no statement (no `;`, no WGSL brace), and every identifier in
//! it is a WGSL builtin function or type, a read-only binding (`var<storage,
//! read>`, `var<uniform>`), a module `const`, a builtin parameter or `let` of
//! the entry point, or a function (prologue or row) whose body holds only such
//! names and its own locals. A register, an output binding, workgroup memory, a
//! subgroup operation or a name a row binds locally (`r`, `c`, a loop counter)
//! makes a row context-dependent, so it is never moved. A subtree is bound only
//! if every node in it is context-free, it holds a function, and the kernel
//! body spells it at least twice (counted over the DAG's edges, so a subtree
//! inside a bound one is bound only if it repeats inside that definition too).
//! The `let`s go ahead of the root row's first argument spelled after
//! `<declarations>`, in definition order (a definition after everything it
//! references); every in-body use is after them. Context-free subtrees have no
//! effects, and WGSL integer and float operations never trap, so evaluating one
//! once, earlier, gives the value every use read before: the tests compare the
//! outputs of both texts bit for bit on the CPU executor.
//!
//! **Across the population: an ADF** (`nucleotable/docs/MultiTypedGEP.md` rev 4
//! section 6). The population's expressed trees are hash-consed into one DAG;
//! a node with at least two DAG edges into it, occurring in at least two
//! individuals and holding a function, is an ADF (the population DAG's rule,
//! [`crate::population_dag::is_adf`], on edges): an immutable expressed tree
//! with a stable id ([`AdfTable`], ids never reused), a root type and a level,
//! stored as `adf_node` rows ([`crate::typed_chromosome::Adf`]). Every gene
//! holding one is rewritten through the per-type write-back to hold the ADF's
//! reference terminal `A_d` (a removal-only rewrite, so it always fits), and the
//! folded gene expressed and inlined is the original tree: the kernel text, and
//! so every value, is unchanged.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use crate::geneframe::Ty;
use crate::kingdom_contract::{KingdomContract, SymbolId};
use crate::typed_chromosome::{express_chromosome, Adf, AdfDecl, AdfId, AdfSet, Chromosome, ChromosomeLayout, Epoch};
use crate::typed_gep::{write_back, GeneLayout, Layout, Tree};
use crate::wgsl::true_kingdom::{substitute, RowKind, TrueKingdom, TrueRow};

/// The name type's dual.
pub const NAME_DUAL: &str = "share.name";

/// WGSL builtin functions and type names whose value depends only on their arguments.
const BUILTINS: &[&str] = &[
    "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "atan2", "ceil", "clamp", "cos", "cosh", "countLeadingZeros", "countOneBits", "countTrailingZeros", "cross", "degrees", "determinant", "distance", "dot", "dot4U8Packed", "dot4I8Packed", "exp", "exp2", "extractBits", "faceForward", "firstLeadingBit", "firstTrailingBit", "floor", "fma", "fract", "frexp", "insertBits", "inverseSqrt", "ldexp", "length", "log", "log2", "max", "min", "mix", "modf", "normalize", "pow", "quantizeToF16", "radians", "reflect", "refract", "reverseBits", "round", "saturate", "sign", "sin", "sinh", "smoothstep", "sqrt", "step", "tan", "tanh", "transpose", "trunc", "select", "all", "any", "bitcast", "pack4x8snorm", "pack4x8unorm", "pack2x16snorm", "pack2x16unorm", "pack2x16float", "unpack4x8snorm", "unpack4x8unorm", "unpack2x16snorm", "unpack2x16unorm", "unpack2x16float", "arrayLength", "bool", "i32", "u32", "f32", "f16", "vec2", "vec3", "vec4", "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3", "mat3x4", "mat4x2", "mat4x3", "mat4x4", "array", "true", "false",
];

fn bug(what: impl std::fmt::Display) -> String {
    format!("our software has a bug: {what}")
}

/// A WGSL scalar, vector or matrix slot: what a `let` can hold.
pub fn value_slot(slot: &str) -> bool {
    const SCALARS: [&str; 5] = ["bool", "i32", "u32", "f32", "f16"];
    if SCALARS.contains(&slot) {
        return true;
    }
    if let Some(rest) = slot.strip_prefix("vec").filter(|r| r.len() > 3 && matches!(&r[..1], "2" | "3" | "4")) {
        return rest[1..].strip_prefix('<').and_then(|r| r.strip_suffix('>')).is_some_and(|s| SCALARS.contains(&s));
    }
    if let Some(rest) = slot.strip_prefix("mat") {
        let b = rest.as_bytes();
        return b.len() >= 5 && matches!(b[0], b'2'..=b'4') && b[1] == b'x' && matches!(b[2], b'2'..=b'4') && matches!(&rest[3..], "<f32>" | "<f16>");
    }
    false
}

/// A spelling's placeholders: how often each argument is spelled, whether one is split into fields, and the text with
/// every placeholder removed (what is left is WGSL). The scan is [`substitute`]'s.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Placeholders {
    pub count: Vec<u32>,
    pub field: Vec<bool>,
    /// Byte offset of each placeholder in the spelling, with its argument.
    pub at: Vec<(usize, usize)>,
    pub wgsl: String,
}

pub fn placeholders(spelling: &str, arity: usize) -> Placeholders {
    let mut p = Placeholders { count: vec![0; arity], field: vec![false; arity], at: Vec::new(), wgsl: String::with_capacity(spelling.len()) };
    let b = spelling.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'{' {
            if let Some(close) = spelling[i..].find('}') {
                let inner = &spelling[i + 1..i + close];
                if inner == "v" {
                    p.wgsl.push_str(" 0");
                    i += close + 1;
                    continue;
                }
                let (a, f) = match inner.split_once(':') {
                    Some((a, f)) => (a, Some(f)),
                    None => (inner, None),
                };
                if !a.is_empty() && a.bytes().all(|c| c.is_ascii_digit()) && f.is_none_or(|f| !f.is_empty() && f.bytes().all(|c| c.is_ascii_digit())) {
                    if let Ok(n) = a.parse::<usize>() {
                        if n < arity {
                            p.count[n] += 1;
                            p.field[n] |= f.is_some();
                            p.at.push((i, n));
                        }
                    }
                    p.wgsl.push_str(" 0");
                    i += close + 1;
                    continue;
                }
            }
        }
        let ch = spelling[i..].chars().next().unwrap_or(' ');
        p.wgsl.push(ch);
        i += ch.len_utf8();
    }
    p
}

/// The identifiers of WGSL text, skipping member names after `.`, numbers (with their suffixes) and attributes.
fn identifiers(text: &str) -> Vec<&str> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_digit() {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.') {
                i += 1;
            }
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let prev = text[..start].trim_end().as_bytes().last().copied();
            if prev != Some(b'.') && prev != Some(b'@') {
                out.push(&text[start..i]);
            }
            continue;
        }
        i += 1;
    }
    out
}

/// The names `let NAME` / `var NAME` / `var<...> NAME` bind in `text`.
fn bound_names(text: &str) -> Vec<&str> {
    let ids = identifiers(text);
    let mut out = Vec::new();
    for w in ids.windows(2) {
        if w[0] == "let" || w[0] == "const" {
            out.push(w[1]);
        }
    }
    // `var NAME` and `var<space, mode> NAME`: the first identifier after `var` that is not an address space or mode.
    let mut it = ids.iter().peekable();
    while let Some(id) = it.next() {
        if *id == "var" {
            for n in it.by_ref() {
                if !matches!(*n, "function" | "private" | "workgroup" | "uniform" | "storage" | "read" | "write" | "read_write") {
                    out.push(*n);
                    break;
                }
            }
        }
    }
    out
}

/// One function definition: its name, its parameters, and its body.
struct FnDef<'a> {
    name: &'a str,
    params: Vec<&'a str>,
    body: &'a str,
}

/// Every `fn NAME(params) [-> T] { body }` in `text`.
fn fn_defs(text: &str) -> Vec<FnDef<'_>> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(off) = text[from..].find("fn ") {
        let at = from + off;
        from = at + 3;
        if at > 0 && (text.as_bytes()[at - 1].is_ascii_alphanumeric() || text.as_bytes()[at - 1] == b'_') {
            continue;
        }
        let rest = &text[at + 3..];
        let Some(open) = rest.find('(') else { break };
        let name = rest[..open].trim();
        if name.is_empty() || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
            continue;
        }
        let mut depth = 0i32;
        let mut close = None;
        for (j, c) in rest[open..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + j);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else { break };
        let params: Vec<&str> = rest[open + 1..close].split(',').filter_map(|p| p.split(':').next()).map(|p| p.trim()).filter(|p| !p.is_empty()).map(|p| p.rsplit(' ').next().unwrap_or(p)).collect();
        let Some(bo) = rest[close..].find('{') else { break };
        let bstart = close + bo;
        let mut depth = 0i32;
        let mut bend = None;
        for (j, c) in rest[bstart..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        bend = Some(bstart + j);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(bend) = bend else { break };
        out.push(FnDef { name, params, body: &rest[bstart + 1..bend] });
        from = at + 3 + bend;
    }
    out
}

/// How a root row places its arguments: which are spelled inside the entry point's body (after `<declarations>`), and
/// the first of them, where the `let`s go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootPlacement {
    pub in_body: Vec<bool>,
    pub first: usize,
}

/// The share rows and the context-free classification of the base rows.
#[derive(Debug, Clone)]
pub struct Share {
    /// The number of base rows; share row `j` is symbol `n_base + j`.
    pub n_base: usize,
    pub rows: Vec<TrueRow>,
    /// `C_T` of `share.name`.
    pub names: Vec<String>,
    pub name_ty: Ty,
    pub name_row: SymbolId,
    /// `(dual, body type)` -> the let row.
    pub let_row: BTreeMap<(Ty, Ty), SymbolId>,
    /// dual -> the reference row.
    pub ref_row: BTreeMap<Ty, SymbolId>,
    /// Per base row: context-free.
    pub context_free: Vec<bool>,
    /// Per base row: its placeholders.
    pub spelled: Vec<Placeholders>,
    /// Per root-producing base row.
    pub roots: BTreeMap<SymbolId, RootPlacement>,
}

/// What [`Share::share`] did to one kernel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KernelShare {
    /// Nodes of the expressed tree.
    pub nodes: usize,
    /// Distinct subtrees of the expressed tree (its hash-consed DAG).
    pub distinct: usize,
    pub lets: usize,
    /// Function nodes the kernel text spells (each spelled argument counted), without and with the `let`s.
    pub spelled_ops: u64,
    pub spelled_ops_shared: u64,
}

/// One kernel's tree with its `let`s, and the counts.
#[derive(Debug, Clone)]
pub struct Shared {
    pub tree: Tree,
    pub stats: KernelShare,
}

fn tsv_rows(text: &str, file: &str) -> Result<Vec<Vec<String>>, String> {
    let mut lines = text.lines();
    lines.next().ok_or_else(|| bug(format!("{file} is empty")))?;
    Ok(lines.filter(|l| !l.is_empty()).map(|l| l.split('\t').map(str::to_string).collect()).collect())
}

/// Hash-consing of one tree: per occurrence its DAG id.
struct Occ {
    id: usize,
    kids: Vec<Occ>,
}

#[derive(Default)]
struct Cons {
    index: HashMap<(SymbolId, Option<u32>, Vec<usize>), usize>,
    /// Per id: symbol, the function count of the subtree, all nodes context-free.
    info: Vec<(SymbolId, u64, bool)>,
}

impl Cons {
    fn intern(&mut self, t: &Tree, functions: &dyn Fn(SymbolId) -> bool, context_free: &dyn Fn(SymbolId) -> bool) -> Occ {
        let kids: Vec<Occ> = t.kids.iter().map(|k| self.intern(k, functions, context_free)).collect();
        let key = (t.symbol, t.const_index, kids.iter().map(|k| k.id).collect::<Vec<_>>());
        if let Some(&id) = self.index.get(&key) {
            return Occ { id, kids };
        }
        let ops = u64::from(functions(t.symbol)) + kids.iter().map(|k| self.info[k.id].1).sum::<u64>();
        let free = context_free(t.symbol) && kids.iter().all(|k| self.info[k.id].2);
        let id = self.info.len();
        self.info.push((t.symbol, ops, free));
        self.index.insert(key, id);
        Occ { id, kids }
    }
}

impl Share {
    /// The share rows of `k`'s table (`share.tsv`, `share_constants.tsv` in its directory) and the classification of
    /// its rows. Every value dual of `types.tsv` must have its rows: a table regenerated without them is a bug to fix
    /// by running `gen_share.py`, never a dual left unshared.
    pub fn load(k: &TrueKingdom) -> Result<Share, String> {
        Self::load_from(k, &k.dir)
    }

    pub fn load_from(k: &TrueKingdom, dir: &Path) -> Result<Share, String> {
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).map_err(|e| bug(format!("{}: {e} (run python3 kingdoms/wgsl_true/gen_share.py)", dir.join(f).display())));
        let n_base = k.rows.len();
        let name_ty = Ty::Wgsl(u16::try_from(k.types.len()).map_err(|_| bug("too many duals for share.name"))?);
        Ty::check_dual_count("wgsl", k.types.len() + 1, Ty::MAX_WGSL_DUALS)?;
        let by_name: BTreeMap<&str, Ty> = k.types.iter().map(|d| (d.dual.as_str(), d.ty())).chain([(NAME_DUAL, name_ty)]).collect();
        let ty_of = |n: &str, row: &str| by_name.get(n).copied().ok_or_else(|| bug(format!("share.tsv row {row}: type {n} is not in types.tsv (run gen_share.py)")));
        let mut rows = Vec::new();
        for f in tsv_rows(&read("share.tsv")?, "share.tsv")? {
            if f.len() != 7 {
                return Err(bug(format!("share.tsv: a row of {} fields, want 7", f.len())));
            }
            let kind = match f[2].as_str() {
                "function" => RowKind::Function,
                "constant" => RowKind::Constant,
                o => return Err(bug(format!("share.tsv row {}: kind {o}", f[1]))),
            };
            let args = if f[4].is_empty() { Vec::new() } else { f[4].split(';').map(|a| ty_of(a, &f[1])).collect::<Result<Vec<_>, _>>()? };
            rows.push(TrueRow { name: f[1].clone(), kind, out: ty_of(&f[3], &f[1])?, args, spelling: f[5].replace("\\n", "\n"), note: f[6].clone() });
        }
        let mut names = Vec::new();
        for f in tsv_rows(&read("share_constants.tsv")?, "share_constants.tsv")? {
            if f.len() != 3 || f[0] != NAME_DUAL || f[1].parse::<usize>().ok() != Some(names.len()) {
                return Err(bug(format!("share_constants.tsv: row {f:?} is not the next name")));
            }
            names.push(f[2].clone());
        }
        let mut let_row = BTreeMap::new();
        let mut ref_row = BTreeMap::new();
        let mut name_row = None;
        for (j, r) in rows.iter().enumerate() {
            let s = (n_base + j) as SymbolId;
            match (r.kind, r.args.as_slice()) {
                (RowKind::Constant, []) if r.out == name_ty => name_row = Some(s),
                (RowKind::Function, [n, v, b]) if *n == name_ty && r.out == *b => {
                    let_row.insert((*v, *b), s);
                }
                (RowKind::Function, [n]) if *n == name_ty => {
                    ref_row.insert(r.out, s);
                }
                _ => return Err(bug(format!("share.tsv row {} is not a name, let or reference row", r.name))),
            }
        }
        let name_row = name_row.ok_or_else(|| bug("share.tsv has no share.name row"))?;
        let used: BTreeSet<&str> = names.iter().map(String::as_str).collect();
        for r in &k.rows {
            if identifiers(&placeholders(&r.spelling, r.args.len()).wgsl).iter().any(|i| used.contains(i)) {
                return Err(bug(format!("row {} spells a share name; the names must be fresh", r.name)));
            }
        }
        let spelled: Vec<Placeholders> = k.rows.iter().map(|r| placeholders(&r.spelling, r.args.len())).collect();
        let mut roots = BTreeMap::new();
        for (s, r) in k.rows.iter().enumerate() {
            if r.out != k.root {
                continue;
            }
            let at = r.spelling.find("<declarations>").ok_or_else(|| bug(format!("root row {} has no <declarations>", r.name)))?;
            let mut in_body = vec![false; r.args.len()];
            let mut first: Option<(usize, usize)> = None;
            for (pos, a) in &spelled[s].at {
                if *pos > at {
                    in_body[*a] = true;
                    if first.is_none_or(|(p, _)| *pos < p) {
                        first = Some((*pos, *a));
                    }
                }
            }
            for (pos, a) in &spelled[s].at {
                if in_body[*a] && *pos < at {
                    return Err(bug(format!("root row {} spells argument {a} both before and after <declarations>", r.name)));
                }
            }
            let first = first.ok_or_else(|| bug(format!("root row {} spells no argument in its body", r.name)))?.1;
            roots.insert(s as SymbolId, RootPlacement { in_body, first });
        }
        let context_free = context_free_rows(k);
        let share = Share { n_base, rows, names, name_ty, name_row, let_row, ref_row, context_free, spelled, roots };
        for d in &k.types {
            if value_slot(&d.slot) && !share.ref_row.contains_key(&d.ty()) {
                return Err(bug(format!("value dual {} has no share rows: run python3 kingdoms/wgsl_true/gen_share.py", d.dual)));
            }
        }
        for (s, p) in &share.roots {
            let body = k.rows[*s as usize].args[p.first];
            if share.ref_row.keys().any(|v| !share.let_row.contains_key(&(*v, body))) {
                return Err(bug(format!("root row {}: a value dual has no let row over {} (run gen_share.py)", k.rows[*s as usize].name, k.type_name(body))));
            }
        }
        Ok(share)
    }

    fn row<'a>(&'a self, k: &'a TrueKingdom, s: SymbolId) -> Result<&'a TrueRow, String> {
        let i = s as usize;
        if i < self.n_base {
            Ok(&k.rows[i])
        } else {
            self.rows.get(i - self.n_base).ok_or_else(|| bug(format!("symbol {s} is not a row")))
        }
    }

    fn is_share_row(&self, s: SymbolId) -> bool {
        s as usize >= self.n_base
    }

    /// The kernel with every subtree its body spells more than once bound by a `let` (module doc).
    pub fn share(&self, k: &TrueKingdom, tree: &Tree) -> Result<Shared, String> {
        let root_row = self.roots.get(&tree.symbol).ok_or_else(|| bug(format!("the tree's root {} is not a root row", tree.symbol)))?;
        let n_base = self.n_base;
        let functions = |s: SymbolId| (s as usize) < n_base && k.rows[s as usize].kind == RowKind::Function;
        let free = |s: SymbolId| (s as usize) < n_base && self.context_free[s as usize];
        let mut cons = Cons::default();
        let occ = cons.intern(tree, &functions, &free);
        let body_ty = k.rows[tree.symbol as usize].args[root_row.first];
        let candidate = |id: usize| -> bool {
            let (s, ops, free) = cons.info[id];
            free && ops >= 1 && self.let_row.contains_key(&(k.rows[s as usize].out, body_ty))
        };
        // DAG edges into each node, counted where the kernel body spells them.
        let mut refs = vec![0u64; cons.info.len()];
        let mut visited = vec![false; cons.info.len()];
        fn count(o: &Occ, t: &Tree, mult: u64, sh: &Share, cand: &dyn Fn(usize) -> bool, refs: &mut [u64], visited: &mut [bool]) {
            let p = &sh.spelled[t.symbol as usize];
            let m = if cand(o.id) {
                refs[o.id] += mult;
                if visited[o.id] {
                    return;
                }
                visited[o.id] = true;
                1
            } else {
                mult
            };
            for (i, (ko, kt)) in o.kids.iter().zip(&t.kids).enumerate() {
                if !p.field[i] && p.count[i] > 0 {
                    count(ko, kt, m * u64::from(p.count[i]), sh, cand, refs, visited);
                }
            }
        }
        let rp = &self.spelled[tree.symbol as usize];
        for (i, (ko, kt)) in occ.kids.iter().zip(&tree.kids).enumerate() {
            if root_row.in_body[i] && !rp.field[i] {
                count(ko, kt, u64::from(rp.count[i]), self, &candidate, &mut refs, &mut visited);
            }
        }
        let shared: Vec<bool> = (0..cons.info.len()).map(|id| candidate(id) && refs[id] >= 2).collect();
        // Definition order: post-order, so a definition follows every binding it references.
        let mut order: Vec<usize> = Vec::new();
        let mut placed = vec![false; cons.info.len()];
        fn emit(o: &Occ, t: &Tree, sh: &Share, shared: &[bool], placed: &mut [bool], order: &mut Vec<usize>) {
            if shared[o.id] && placed[o.id] {
                return;
            }
            let p = &sh.spelled[t.symbol as usize];
            for (i, (ko, kt)) in o.kids.iter().zip(&t.kids).enumerate() {
                if !p.field[i] && p.count[i] > 0 {
                    emit(ko, kt, sh, shared, placed, order);
                }
            }
            if shared[o.id] {
                placed[o.id] = true;
                order.push(o.id);
            }
        }
        for (i, (ko, kt)) in occ.kids.iter().zip(&tree.kids).enumerate() {
            if root_row.in_body[i] && !rp.field[i] {
                emit(ko, kt, self, &shared, &mut placed, &mut order);
            }
        }
        if order.len() > self.names.len() {
            return Err(bug(format!("a kernel binds {} lets, share.name has {} names (raise NAMES in gen_share.py)", order.len(), self.names.len())));
        }
        let slot_of: HashMap<usize, u32> = order.iter().enumerate().map(|(j, id)| (*id, j as u32)).collect();
        let name = |j: u32| Tree { symbol: self.name_row, const_index: Some(j), kids: Vec::new() };
        fn rebuild(o: &Occ, t: &Tree, sh: &Share, k: &TrueKingdom, slot_of: &HashMap<usize, u32>, top: bool, name: &dyn Fn(u32) -> Tree) -> Tree {
            if !top {
                if let Some(j) = slot_of.get(&o.id) {
                    return Tree { symbol: sh.ref_row[&k.rows[t.symbol as usize].out], const_index: None, kids: vec![name(*j)] };
                }
            }
            let p = &sh.spelled[t.symbol as usize];
            let kids = o.kids.iter().zip(&t.kids).enumerate().map(|(i, (ko, kt))| if p.field[i] || p.count[i] == 0 { kt.clone() } else { rebuild(ko, kt, sh, k, slot_of, false, name) }).collect();
            Tree { symbol: t.symbol, const_index: t.const_index, kids }
        }
        // One occurrence of each binding, for its definition.
        let mut first_occ: HashMap<usize, (&Occ, &Tree)> = HashMap::new();
        fn find<'a>(o: &'a Occ, t: &'a Tree, want: &HashMap<usize, u32>, out: &mut HashMap<usize, (&'a Occ, &'a Tree)>) {
            if want.contains_key(&o.id) {
                out.entry(o.id).or_insert((o, t));
            }
            for (ko, kt) in o.kids.iter().zip(&t.kids) {
                find(ko, kt, want, out);
            }
        }
        for (i, (ko, kt)) in occ.kids.iter().zip(&tree.kids).enumerate() {
            if root_row.in_body[i] {
                find(ko, kt, &slot_of, &mut first_occ);
            }
        }
        let mut kids: Vec<Tree> = occ.kids.iter().zip(&tree.kids).enumerate().map(|(i, (ko, kt))| if root_row.in_body[i] && !rp.field[i] { rebuild(ko, kt, self, k, &slot_of, false, &name) } else { kt.clone() }).collect();
        let mut body = kids[root_row.first].clone();
        for (j, id) in order.iter().enumerate().rev() {
            let (o, t) = first_occ.get(id).ok_or_else(|| bug("a binding with no occurrence"))?;
            let def = rebuild(o, t, self, k, &slot_of, true, &name);
            let out = k.rows[t.symbol as usize].out;
            body = Tree { symbol: self.let_row[&(out, body_ty)], const_index: None, kids: vec![name(j as u32), def, body] };
        }
        kids[root_row.first] = body;
        let shared_tree = Tree { symbol: tree.symbol, const_index: tree.const_index, kids };
        let stats = KernelShare { nodes: count_nodes(tree), distinct: cons.info.len(), lets: order.len(), spelled_ops: self.spelled_ops(k, tree)?, spelled_ops_shared: self.spelled_ops(k, &shared_tree)? };
        Ok(Shared { tree: shared_tree, stats })
    }

    /// The kingdom functions the text of `tree` spells: each node once per time its parent spells it.
    pub fn spelled_ops(&self, k: &TrueKingdom, tree: &Tree) -> Result<u64, String> {
        let row = self.row(k, tree.symbol)?;
        let p = if self.is_share_row(tree.symbol) { placeholders(&row.spelling, row.args.len()) } else { self.spelled[tree.symbol as usize].clone() };
        let mut n = u64::from(!self.is_share_row(tree.symbol) && row.kind == RowKind::Function);
        for (i, kid) in tree.kids.iter().enumerate() {
            n += u64::from(*p.count.get(i).unwrap_or(&0)) * self.spelled_ops(k, kid)?;
        }
        Ok(n)
    }

    /// The tree with every `let` removed and every reference replaced by its definition.
    pub fn unshare(&self, k: &TrueKingdom, tree: &Tree) -> Result<Tree, String> {
        let Some(rp) = self.roots.get(&tree.symbol) else { return Err(bug("the tree's root is not a root row")) };
        let mut defs: BTreeMap<u32, Tree> = BTreeMap::new();
        let mut body = tree.kids.get(rp.first).ok_or_else(|| bug("the root has too few arguments"))?;
        while self.let_row.values().any(|s| *s == body.symbol) {
            let j = body.kids[0].const_index.ok_or_else(|| bug("a let with no name"))?;
            let def = self.inline(k, &body.kids[1], &defs)?;
            if defs.insert(j, def).is_some() {
                return Err(bug(format!("name {j} bound twice")));
            }
            body = &body.kids[2];
        }
        let mut kids = Vec::with_capacity(tree.kids.len());
        for (i, kid) in tree.kids.iter().enumerate() {
            kids.push(if i == rp.first { self.inline(k, body, &defs)? } else { self.inline(k, kid, &defs)? });
        }
        Ok(Tree { symbol: tree.symbol, const_index: tree.const_index, kids })
    }

    fn inline(&self, k: &TrueKingdom, t: &Tree, defs: &BTreeMap<u32, Tree>) -> Result<Tree, String> {
        if self.is_share_row(t.symbol) {
            if self.ref_row.values().any(|s| *s == t.symbol) {
                let j = t.kids.first().and_then(|n| n.const_index).ok_or_else(|| bug("a reference with no name"))?;
                return defs.get(&j).cloned().ok_or_else(|| bug(format!("a reference to name {j} before its let")));
            }
            return Err(bug(format!("share row {} below the let chain", self.row(k, t.symbol)?.name)));
        }
        Ok(Tree { symbol: t.symbol, const_index: t.const_index, kids: t.kids.iter().map(|c| self.inline(k, c, defs)).collect::<Result<_, _>>()? })
    }

    /// Every node of a shared tree fills a slot of its own type, with its signature's number of arguments and, for a
    /// constant, an index inside its `C_T`.
    pub fn check_types(&self, k: &TrueKingdom, tree: &Tree, want: Ty) -> Result<(), String> {
        let row = self.row(k, tree.symbol)?;
        if row.out != want {
            return Err(bug(format!("{} outputs {} in a slot of {}", row.name, self.type_name(k, row.out), self.type_name(k, want))));
        }
        if tree.kids.len() != row.args.len() {
            return Err(bug(format!("{} has {} arguments, its signature {}", row.name, tree.kids.len(), row.args.len())));
        }
        if row.kind == RowKind::Constant {
            let i = tree.const_index.ok_or_else(|| bug(format!("{} has no C_T index", row.name)))? as usize;
            let len = if row.out == self.name_ty { self.names.len() } else { k.constants.get(&row.out).map_or(0, Vec::len) };
            if i >= len {
                return Err(bug(format!("{}: index {i} past C_T of {len}", row.name)));
            }
        }
        for (kid, a) in tree.kids.iter().zip(&row.args) {
            self.check_types(k, kid, *a)?;
        }
        Ok(())
    }

    pub fn type_name(&self, k: &TrueKingdom, t: Ty) -> String {
        if t == self.name_ty {
            NAME_DUAL.to_string()
        } else {
            k.type_name(t)
        }
    }

    /// [`TrueKingdom::render`] over base and share rows: substitution of argument texts into each row's spelling, then
    /// the declarations and the prologue. A tree with no share row renders to `k.render`'s text exactly.
    pub fn render(&self, k: &TrueKingdom, tree: &Tree) -> Result<String, String> {
        let body = self.spell(k, tree)?;
        Ok(format!("{}{}", k.prologue, body.replace("<declarations>", k.declarations.trim_end_matches('\n'))))
    }

    fn spell(&self, k: &TrueKingdom, node: &Tree) -> Result<String, String> {
        let row = self.row(k, node.symbol)?;
        if node.kids.len() != row.args.len() {
            return Err(bug(format!("{} has {} children, its signature {}", row.name, node.kids.len(), row.args.len())));
        }
        let kids: Vec<String> = node.kids.iter().map(|c| self.spell(k, c)).collect::<Result<_, _>>()?;
        let value = match (row.kind, node.const_index) {
            (RowKind::Constant, Some(i)) => Some(if row.out == self.name_ty {
                self.names.get(i as usize).ok_or_else(|| bug(format!("name {i} past share.name's C_T")))?.as_str()
            } else {
                k.constants.get(&row.out).and_then(|v| v.get(i as usize)).ok_or_else(|| bug(format!("{}: constant index {i} past C_T", row.name)))?.as_str()
            }),
            (RowKind::Constant, None) => return Err(bug(format!("{}: a placeholder without a Dc index", row.name))),
            _ => None,
        };
        substitute(&row.spelling, &kids, value).map_err(|e| bug(format!("{}: {e}", row.name)))
    }
}

fn count_nodes(t: &Tree) -> usize {
    1 + t.kids.iter().map(count_nodes).sum::<usize>()
}

/// Per base row: whether its value is the same wherever the kernel body spells it (module doc).
pub fn context_free_rows(k: &TrueKingdom) -> Vec<bool> {
    let module_text: String = std::iter::once(k.prologue.as_str()).chain(k.rows.iter().map(|r| r.spelling.as_str())).collect::<Vec<_>>().join("\n");
    // Names any row binds locally (`let r`, a loop counter): such a name means something else inside that row.
    let mut local: BTreeSet<String> = BTreeSet::new();
    for r in &k.rows {
        if r.out == k.root {
            continue;
        }
        let p = placeholders(&r.spelling, r.args.len());
        let mut outside = p.wgsl.clone();
        for f in fn_defs(&p.wgsl) {
            outside = outside.replace(f.body, " ");
        }
        local.extend(bound_names(&outside).into_iter().filter(|n| !n.starts_with("KNOB_") && !n.starts_with("WG_")).map(str::to_string));
    }
    let mut names: BTreeSet<String> = BUILTINS.iter().map(|s| s.to_string()).collect();
    // Read-only bindings and the uniform.
    for line in k.prologue.lines() {
        let l = line.trim();
        if let Some(rest) = l.split_once("var<storage, read>").map(|(_, r)| r).or_else(|| l.split_once("var<uniform>").map(|(_, r)| r)) {
            if let Some(n) = identifiers(rest).first() {
                names.insert(n.to_string());
            }
        }
    }
    // Module constants, and the entry point's builtin parameters and lets.
    for r in &k.rows {
        let p = placeholders(&r.spelling, r.args.len());
        let ids = identifiers(&p.wgsl);
        for w in ids.windows(2) {
            if w[0] == "const" {
                names.insert(w[1].to_string());
            }
        }
        if r.out == k.root {
            let mut entry = p.wgsl.clone();
            for f in fn_defs(&p.wgsl).iter().filter(|f| f.name != "main") {
                entry = entry.replace(f.body, " ");
            }
            for seg in entry.split("@builtin(").skip(1) {
                if let Some(n) = seg.split_once(')').map(|(_, r)| r).and_then(|r| identifiers(r).first().copied()) {
                    names.insert(n.to_string());
                }
            }
            for w in identifiers(&entry).windows(2) {
                if w[0] == "let" {
                    names.insert(w[1].to_string());
                }
            }
        }
    }
    for l in &local {
        names.remove(l);
    }
    // Functions: context-free when every name in every definition is one, or its own parameter or local.
    let defs = fn_defs(&module_text);
    let mut fns: BTreeSet<String> = BTreeSet::new();
    let mut impure: BTreeSet<&str> = BTreeSet::new();
    loop {
        let mut changed = false;
        for f in &defs {
            if fns.contains(f.name) || impure.contains(f.name) {
                continue;
            }
            let own: BTreeSet<&str> = f.params.iter().copied().chain(bound_names(f.body)).collect();
            let mut pending = false;
            let mut ok = true;
            for id in identifiers(f.body) {
                if matches!(id, "let" | "var" | "return" | "if" | "else" | "switch" | "case" | "default" | "for" | "loop" | "break" | "continue" | "while") || own.contains(id) || names.contains(id) || fns.contains(id) {
                    continue;
                }
                if defs.iter().any(|g| g.name == id) && !impure.contains(id) {
                    pending = true;
                } else {
                    ok = false;
                }
            }
            if !ok {
                impure.insert(f.name);
                changed = true;
            } else if !pending && defs.iter().filter(|g| g.name == f.name).all(|g| std::ptr::eq(g, f) || fns.contains(g.name)) {
                // Every definition of the name must be context-free: mark only once all are checked.
                let all_ok = defs.iter().filter(|g| g.name == f.name).all(|g| {
                    let own: BTreeSet<&str> = g.params.iter().copied().chain(bound_names(g.body)).collect();
                    identifiers(g.body).iter().all(|id| matches!(*id, "let" | "var" | "return" | "if" | "else" | "switch" | "case" | "default" | "for" | "loop" | "break" | "continue" | "while") || own.contains(id) || names.contains(*id) || fns.contains(*id))
                });
                if all_ok {
                    fns.insert(f.name.to_string());
                } else {
                    impure.insert(f.name);
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let value = |t: Ty| match t {
        Ty::Wgsl(i) => k.types.get(i as usize).is_some_and(|d| value_slot(&d.slot)),
        _ => false,
    };
    k.rows
        .iter()
        .map(|r| {
            if r.out == k.root || !value(r.out) || !r.args.iter().all(|a| value(*a)) {
                return false;
            }
            let p = placeholders(&r.spelling, r.args.len());
            if p.wgsl.contains(';') || p.wgsl.contains('{') || p.wgsl.contains('}') {
                return false;
            }
            identifiers(&p.wgsl).iter().all(|id| names.contains(*id) || fns.contains(*id))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Across the population: ADFs
// ---------------------------------------------------------------------------

/// The population's ADFs, by body: ids are stable and never reused (an epoch's append-only ADF set).
#[derive(Debug, Clone, Default)]
pub struct AdfTable {
    /// The expanded body (kingdom symbols only) -> id.
    pub index: HashMap<Tree, AdfId>,
    pub next: AdfId,
}

/// One population folded into ADFs.
#[derive(Debug, Clone)]
pub struct PopulationFold {
    pub epoch: Epoch,
    pub adfs: AdfSet,
    /// The ADFs this population uses: id, root type, level, body nodes (expanded).
    pub used: Vec<(AdfId, Ty, u32, usize)>,
    /// Each individual's gene after the write-back.
    pub genes: Vec<Vec<u32>>,
    /// References written into the genes.
    pub references: usize,
    pub stats: PopulationShare,
}

/// The population DAG's counts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PopulationShare {
    pub individuals: usize,
    /// Nodes of every expressed tree.
    pub nodes: usize,
    /// Distinct subtrees over the population (the hash-consed DAG's nodes).
    pub distinct: usize,
    /// Occurrences of subtrees holding a function, and those whose subtree occurs more than once in the population.
    pub occurrences: usize,
    pub repeated: usize,
    pub adfs: usize,
    /// Nodes stored with the ADFs: every folded tree plus every ADF body once.
    pub stored_nodes: usize,
}

impl AdfTable {
    fn id_of(&mut self, body: &Tree) -> (AdfId, bool) {
        if let Some(id) = self.index.get(body) {
            return (*id, false);
        }
        let id = self.next;
        self.next += 1;
        self.index.insert(body.clone(), id);
        (id, true)
    }

    /// Find the population's shared subtrees, make each an ADF (a stable id per body, reused when an earlier fold saw
    /// it), and write every gene holding one back with its reference. `genes[i]` expresses to `trees[i]` under
    /// `layout`. Every folded gene is expressed again and inlined, and must give its tree back: a gene that does not
    /// is a bug, returned as one.
    pub fn fold(&mut self, base: &KingdomContract, layout: &GeneLayout, const_len: &BTreeMap<Ty, u32>, genes: &[Vec<u32>], trees: &[Tree]) -> Result<PopulationFold, String> {
        if genes.len() != trees.len() {
            return Err(bug("genes and trees differ in number"));
        }
        let functions = |s: SymbolId| !base.is_terminal(s);
        let all = |_: SymbolId| true;
        let mut cons = Cons::default();
        let occs: Vec<Occ> = trees.iter().map(|t| cons.intern(t, &functions, &all)).collect();
        let n = cons.info.len();
        // DAG edges, and in how many individuals each node occurs.
        let mut refs = vec![0u64; n];
        let mut visited = vec![false; n];
        fn edges(o: &Occ, refs: &mut [u64], visited: &mut [bool]) {
            refs[o.id] += 1;
            if visited[o.id] {
                return;
            }
            visited[o.id] = true;
            for k in &o.kids {
                edges(k, refs, visited);
            }
        }
        let mut individuals = vec![0usize; n];
        let mut occurrences_of = vec![0usize; n];
        for o in &occs {
            edges(o, &mut refs, &mut visited);
            let mut seen = BTreeSet::new();
            fn mark(o: &Occ, seen: &mut BTreeSet<usize>, occ: &mut [usize]) {
                seen.insert(o.id);
                occ[o.id] += 1;
                for k in &o.kids {
                    mark(k, seen, occ);
                }
            }
            mark(o, &mut seen, &mut occurrences_of);
            for id in seen {
                individuals[id] += 1;
            }
        }
        let is_adf = |id: usize| cons.info[id].1 >= 1 && refs[id] >= 2 && individuals[id] >= 2;
        // ADF bodies with lower ADFs as references, in post-order (a definition after everything it references).
        let mut order: Vec<usize> = Vec::new();
        let mut placed = vec![false; n];
        fn post(o: &Occ, is_adf: &dyn Fn(usize) -> bool, placed: &mut [bool], order: &mut Vec<usize>) {
            if placed[o.id] {
                return;
            }
            placed[o.id] = true;
            for k in &o.kids {
                post(k, is_adf, placed, order);
            }
            if is_adf(o.id) {
                order.push(o.id);
            }
        }
        for o in &occs {
            post(o, &is_adf, &mut placed, &mut order);
        }
        // The first occurrence of every node, with its tree.
        let mut first: HashMap<usize, (&Occ, &Tree)> = HashMap::new();
        fn firsts<'a>(o: &'a Occ, t: &'a Tree, out: &mut HashMap<usize, (&'a Occ, &'a Tree)>) {
            if out.contains_key(&o.id) {
                return;
            }
            out.insert(o.id, (o, t));
            for (ko, kt) in o.kids.iter().zip(&t.kids) {
                firsts(ko, kt, out);
            }
        }
        for (o, t) in occs.iter().zip(trees) {
            firsts(o, t, &mut first);
        }
        let mut adf_id: HashMap<usize, AdfId> = HashMap::new();
        let mut level: HashMap<usize, u32> = HashMap::new();
        let mut decls = Vec::new();
        for id in &order {
            let (occ, t) = first[id];
            let (a, _) = self.id_of(t);
            adf_id.insert(*id, a);
            fn deepest(o: &Occ, top: bool, adf: &HashMap<usize, u32>) -> Option<u32> {
                if !top {
                    if let Some(l) = adf.get(&o.id) {
                        return Some(*l);
                    }
                }
                o.kids.iter().filter_map(|k| deepest(k, false, adf)).max()
            }
            let l = deepest(occ, true, &level).map_or(0, |d| d + 1);
            level.insert(*id, l);
            decls.push(AdfDecl { id: a, root: base.output(t.symbol), level: l });
        }
        let epoch = Epoch::new(base, &[layout.root], &decls).map_err(|e| e.to_string())?;
        // `t` with every ADF subtree (below the top, or at it too) replaced by its reference.
        fn fold_tree(o: &Occ, t: &Tree, top_too: bool, adf_id: &HashMap<usize, AdfId>, epoch: &Epoch) -> Result<(Tree, usize), String> {
            if top_too {
                if let Some(a) = adf_id.get(&o.id) {
                    let s = epoch.adf_terminal(*a).ok_or_else(|| bug(format!("ADF {a} has no reference terminal")))?;
                    return Ok((Tree { symbol: s, const_index: None, kids: Vec::new() }, 1));
                }
            }
            let mut n = 0;
            let mut kids = Vec::with_capacity(t.kids.len());
            for (ko, kt) in o.kids.iter().zip(&t.kids) {
                let (f, m) = fold_tree(ko, kt, true, adf_id, epoch)?;
                n += m;
                kids.push(f);
            }
            Ok((Tree { symbol: t.symbol, const_index: t.const_index, kids }, n))
        }
        let mut bodies = Vec::new();
        let mut used = Vec::new();
        for (id, d) in order.iter().zip(&decls) {
            let (occ, t) = first[id];
            let (body, _) = fold_tree(occ, t, false, &adf_id, &epoch)?;
            bodies.push(Adf::from_tree(&epoch, d.id, d.level, &body));
            used.push((d.id, d.root, d.level, count_nodes(&body)));
        }
        let adfs = AdfSet::new(&epoch, bodies).map_err(|e| e.to_string())?;
        let cl = ChromosomeLayout { normal: Layout::new(0, vec![layout.clone()]), mains: Vec::new() };
        let mut out_genes = Vec::with_capacity(genes.len());
        let mut references = 0;
        let mut stored = used.iter().map(|u| u.3).sum::<usize>();
        for (i, ((g, o), t)) in genes.iter().zip(&occs).zip(trees).enumerate() {
            let (folded, n) = fold_tree(o, t, true, &adf_id, &epoch)?;
            references += n;
            stored += count_nodes(&folded);
            let words = if n == 0 { g.clone() } else { write_back(epoch.kingdom(), layout, g, &folded).map_err(|e| bug(format!("individual {i}: the removal-only write-back did not fit: {e}")))? };
            let c = Chromosome { chromosome_id: i as u64, genes: vec![words.clone()], mains: Vec::new() };
            let p = express_chromosome(&epoch, &adfs, &cl, &c, const_len).map_err(|e| format!("individual {i}: {e}"))?;
            let back = p.inlined(&epoch, &adfs);
            if back.len() != 1 || back[0].1 != *t {
                return Err(bug(format!("individual {i}: the folded gene does not express its tree")));
            }
            out_genes.push(words);
        }
        let occurrences = occurrences_of.iter().enumerate().filter(|(id, _)| cons.info[*id].1 >= 1).map(|(_, c)| *c).sum();
        let repeated = occurrences_of.iter().enumerate().filter(|(id, c)| cons.info[*id].1 >= 1 && **c >= 2).map(|(_, c)| *c).sum();
        let stats = PopulationShare { individuals: trees.len(), nodes: trees.iter().map(count_nodes).sum(), distinct: n, occurrences, repeated, adfs: used.len(), stored_nodes: stored };
        Ok(PopulationFold { epoch, adfs, used, genes: out_genes, references, stats })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_gep::express;
    use crate::wgsl::true_kingdom::{default_dir, random_gene, Rng, Share as Draw};

    fn setup() -> (TrueKingdom, Share, KingdomContract, GeneLayout, BTreeMap<Ty, u32>) {
        let k = TrueKingdom::load(&default_dir()).unwrap_or_else(|e| panic!("{e}"));
        let s = Share::load(&k).unwrap_or_else(|e| panic!("{e}"));
        let c = k.relations().check().unwrap_or_else(|d| panic!("{d:?}"));
        let layout = GeneLayout::from_program_size(&c, k.root, 8).unwrap_or_else(|e| panic!("{e}"));
        let cl = k.const_lens();
        (k, s, c, layout, cl)
    }

    fn genes(c: &KingdomContract, layout: &GeneLayout, cl: &BTreeMap<Ty, u32>, n: usize, seed: u64) -> Vec<Vec<u32>> {
        let mut rng = Rng(seed);
        (0..n).map(|i| random_gene(c, layout, cl, &mut rng, if i % 2 == 0 { Draw::Half } else { Draw::Proportional }, (i / 2) % 2 == 1)).collect()
    }

    fn row(k: &TrueKingdom, name: &str) -> usize {
        k.rows.iter().position(|r| r.name == name).unwrap_or_else(|| panic!("no row {name}"))
    }

    #[test]
    fn identifiers_skip_members_numbers_and_placeholders() {
        let p = placeholders("min({0}, arrayLength(&phi) - 1u) + v.x + {1:0} + 0x9e37u", 2);
        assert_eq!(p.count, vec![1, 1]);
        assert_eq!(p.field, vec![false, true]);
        assert_eq!(identifiers(&p.wgsl), vec!["min", "arrayLength", "phi", "v"]);
        assert!(value_slot("vec3<u32>") && value_slot("mat4x4<f32>") && value_slot("f16") && !value_slot("reg<f32>") && !value_slot("block") && !value_slot("vec5<f32>"));
    }

    /// Arithmetic, read-only loads and the entry point's builtins are context-free; a load of the output, of
    /// workgroup memory, a register and a statement are not.
    #[test]
    fn the_context_free_rows_are_the_pure_value_rows() {
        let (k, s, ..) = setup();
        for name in ["arith.add", "arith.fma", "load.phi", "index.add", "literal.real", "literal.index", "convert.index_to_f32"] {
            assert!(s.context_free[row(&k, name)], "{name} is context-free");
        }
        for name in ["load.h_own", "store.h_L0_f0", "kernel.entry_safe"] {
            assert!(!s.context_free[row(&k, name)], "{name} is not context-free");
        }
        let impure_spelled = |needle: &str| k.rows.iter().enumerate().filter(|(_, r)| r.spelling.contains(needle)).map(|(i, _)| i).collect::<Vec<_>>();
        for needle in ["own_get_wg_", "shuffle_down_f(", "rf0", "own_load_h("] {
            let rows = impure_spelled(needle);
            assert!(rows.iter().all(|i| !s.context_free[*i]), "a row spelling {needle} is context-free");
        }
        let free = s.context_free.iter().filter(|b| **b).count();
        eprintln!("{free} of {} rows context-free", k.rows.len());
        assert!(free > 100);
    }

    /// On random genes: the shared tree is well typed, unshares to the expressed tree, renders without lets to the
    /// kingdom's text, and naga accepts the shared text exactly when it accepts the unshared one.
    #[test]
    fn sharing_round_trips_is_well_typed_and_naga_accepts_both_texts() {
        let (k, s, c, layout, cl) = setup();
        let target = crate::wgsl::target::Target::current();
        let mut lets = 0;
        let mut with_lets = 0;
        for (i, g) in genes(&c, &layout, &cl, 200, 7).iter().enumerate() {
            let t = express(&c, &layout, g, &cl).unwrap_or_else(|e| panic!("{i}: {e}")).tree();
            let sh = s.share(&k, &t).unwrap_or_else(|e| panic!("{i}: {e}"));
            s.check_types(&k, &sh.tree, k.root).unwrap_or_else(|e| panic!("{i}: {e}"));
            assert_eq!(s.unshare(&k, &sh.tree).unwrap_or_else(|e| panic!("{i}: {e}")), t, "{i}: unshare");
            let off = k.render(&t).expect("render");
            assert_eq!(s.render(&k, &t).expect("render"), off, "{i}: no share rows renders the kingdom's text");
            let on = s.render(&k, &sh.tree).expect("render");
            let ok = |src: &str| crate::wgsl::naga_shape::parse_unscanned(src).and_then(|m| target.validate(&m, src).map(|_| ()));
            let (a, b) = (ok(&off), ok(&on));
            assert_eq!(a.is_ok(), b.is_ok(), "{i}: naga off {a:?} on {b:?}\n{on}");
            assert!(sh.stats.spelled_ops_shared <= sh.stats.spelled_ops, "{i}: sharing spells more operations");
            if sh.stats.lets > 0 {
                with_lets += 1;
                assert!(on.contains("let st7_s0: "), "{i}: the first let");
                assert!(sh.stats.spelled_ops_shared < sh.stats.spelled_ops, "{i}: a let saves operations");
            } else {
                assert_eq!(on, off);
            }
            lets += sh.stats.lets;
        }
        eprintln!("{with_lets} of 200 kernels bind {lets} lets");
        assert!(with_lets > 0, "random genes repeat no context-free subtree: the test exercises nothing");
    }

    /// Values are bit-identical with sharing on and off: both texts run on the CPU executor over the Gram unit test's
    /// inputs (one workgroup) and every output word and every recorded write agree.
    #[test]
    fn values_are_bit_identical_with_sharing_on_and_off() {
        use crate::wgsl::cpu::{run_text, Val};
        let (k, s, c, layout, cl) = setup();
        let (d, m) = (16usize, 4usize);
        let phi = crate::wgsl::gram::phi(d, m, 3);
        let inputs = || {
            HashMap::from([
                ("phi".to_string(), Val::V(phi.iter().map(|v| Val::F(*v)).collect())),
                ("h".to_string(), Val::V(vec![Val::F(0.0); d * d])),
                ("cfg".to_string(), Val::V(vec![Val::U(d as u32), Val::U(m as u32), Val::U(d as u32), Val::U(0)])),
            ])
        };
        let (mut compared, mut ran) = (0, 0);
        let mut failed: BTreeMap<String, usize> = BTreeMap::new();
        for (i, g) in genes(&c, &layout, &cl, 400, 11).iter().enumerate() {
            let t = express(&c, &layout, g, &cl).expect("express").tree();
            let sh = s.share(&k, &t).expect("share");
            if sh.stats.lets == 0 {
                continue;
            }
            compared += 1;
            let off = run_text(&k.render(&t).expect("render"), inputs(), [1, 1, 1], 20_000_000);
            let on = run_text(&s.render(&k, &sh.tree).expect("render"), inputs(), [1, 1, 1], 20_000_000);
            match (off, on) {
                (Ok(a), Ok(b)) => {
                    let (mut wa, mut wb) = (Vec::new(), Vec::new());
                    a.buffers["h"].words(&mut wa);
                    b.buffers["h"].words(&mut wb);
                    assert_eq!(wa, wb, "{i}: h differs");
                    let key = |w: &crate::wgsl::cpu::Write| (w.binding.clone(), w.path.clone(), w.gid);
                    assert_eq!(a.writes.iter().map(key).collect::<Vec<_>>(), b.writes.iter().map(key).collect::<Vec<_>>(), "{i}: writes differ");
                    ran += 1;
                }
                (Err(a), Err(b)) => {
                    assert_eq!(a, b, "{i}: both fail, differently");
                    *failed.entry(a.lines().next().unwrap_or_default().chars().take(80).collect::<String>()).or_insert(0usize) += 1;
                }
                (a, b) => panic!("{i}: one text runs and the other does not: off {:?} on {:?}", a.err(), b.err()),
            }
        }
        eprintln!("{ran} of {compared} shared kernels ran on the CPU executor, all bit-identical; both texts failed alike: {failed:?}");
        assert!(ran > 0, "no shared kernel ran: the comparison checked nothing");
    }

    /// Fold -> write-back -> unfold on random genes: every gene holding a population-shared subtree is rewritten to
    /// reference its ADF and expresses its tree again; a second fold of the same population reuses every id and a
    /// new population only appends.
    #[test]
    fn population_adfs_fold_into_genes_and_keep_their_ids() {
        let (k, _, c, layout, cl) = setup();
        let mut g = genes(&c, &layout, &cl, 60, 5);
        // Two copies of one individual, so the whole kernel is shared too.
        g.push(g[0].clone());
        let trees: Vec<Tree> = g.iter().map(|w| express(&c, &layout, w, &cl).expect("express").tree()).collect();
        let mut table = AdfTable::default();
        let f = table.fold(&c, &layout, &cl, &g, &trees).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("{:?}, {} references", f.stats, f.references);
        assert!(f.stats.adfs > 0 && f.references > 0);
        assert!(f.stats.stored_nodes < f.stats.nodes, "the ADFs store fewer nodes");
        assert!(f.used.iter().any(|u| u.1 == k.root), "the duplicated kernel is an ADF");
        for (i, w) in f.genes.iter().enumerate() {
            crate::typed_gep::validate(f.epoch.kingdom(), &layout, w, &cl).unwrap_or_else(|e| panic!("{i}: {e}"));
        }
        let ids: Vec<AdfId> = f.used.iter().map(|u| u.0).collect();
        let again = table.fold(&c, &layout, &cl, &g, &trees).expect("again");
        assert_eq!(again.used.iter().map(|u| u.0).collect::<Vec<_>>(), ids, "a second fold reuses every id");
        let before = table.next;
        let g2 = genes(&c, &layout, &cl, 30, 99);
        let t2: Vec<Tree> = g2.iter().map(|w| express(&c, &layout, w, &cl).expect("express").tree()).collect();
        table.fold(&c, &layout, &cl, &g2, &t2).expect("new population");
        assert!(table.next >= before && table.index.values().all(|id| *id < table.next), "ids only append");
    }
}
