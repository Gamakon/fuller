//! The WGSL kingdom's rows, loaded from the generated tables: the 71 duals of
//! `types.tsv` as [`WgslDual`]s (each a `Ty::Wgsl(index)`), and the 197
//! function templates of `functions.tsv` instantiated over the duals they
//! name into concrete typed rows ([`WgslRow`]), one per legal dual, lanes
//! equal across inputs and output unless the template says otherwise.
//!
//! What is instantiated here is the kingdom's FIXED vocabulary. The rows a
//! kernel brings with it (`load.<buffer>`, `arg.<name>`, `store.<target>`,
//! `shape.swizzle.<pattern>`, `shape.access_dyn` over a buffer) are per
//! kernel and are added by the reader, not here; the templates that stand
//! for them are listed by [`per_kernel_templates`] so nothing is silently
//! dropped.

use std::collections::BTreeMap;

use crate::geneframe::{Arity, Symbol, SymbolTable, Ty};
use crate::gpu_eval::{GpuNode, FN_ID_NUM};

use super::table::{FunctionTable, Template};

/// The kingdom's name in the symbol table.
pub const WGSL: &str = "WGSL";

/// One dual `slot.form` of `types.tsv`.
#[derive(Debug, Clone, PartialEq)]
pub struct WgslDual {
    /// The row index, `T001` → 0; `Ty::Wgsl(index)`.
    pub index: u16,
    pub reference: String,
    /// `f32.real`, `vec3<u32>.index`, …
    pub dual: String,
    pub slot: String,
    pub form: String,
    /// 1 for a scalar, 2–4 for a vector, columns × rows for a matrix, 0 for
    /// the array and store rows.
    pub lanes: u32,
    /// The fallback's lane bit pattern, from the generated column.
    pub fallback_bits: u32,
    /// 1: the fallback substitutes nothing (`store.store`).
    pub fallback_noop: bool,
    /// 1: the fallback is the element's (`array<T>`).
    pub fallback_elem: bool,
    /// The values a literal of this dual takes (the `literal` column); `None`
    /// for a dual no literal takes.
    pub literal: Option<LiteralDomain>,
}

/// THE VALUES A LITERAL TAKES (the `literal` column of `types.tsv`; phylu
/// `docs/PLAN_wgsl_literals_as_constants.md`, 2.3): an integer range, a real
/// range, or a set. A gene holds a literal's value, never a symbol per value;
/// mutation draws inside the domain and every value is read into it
/// ([`LiteralDomain::canonical`]) before it becomes a tree.
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralDomain {
    Int { lo: i64, hi: i64 },
    Real { lo: f64, hi: f64 },
    Set(Vec<i64>),
}

impl LiteralDomain {
    /// A `literal` cell: `-` (none), `lo..=hi` (real for an f32 slot or a
    /// vector or matrix of f32), or `a,b,c`.
    pub fn parse(cell: &str, slot: &str) -> Result<Option<LiteralDomain>, String> {
        if cell == "-" {
            return Ok(None);
        }
        let int = |t: &str| t.trim().parse::<i64>().map_err(|e| format!("literal domain {cell:?}: {t:?}: {e}"));
        if let Some((lo, hi)) = cell.split_once("..=") {
            if slot == "f32" || slot.ends_with("<f32>") {
                let real = |t: &str| t.trim().parse::<f64>().map_err(|e| format!("literal domain {cell:?}: {t:?}: {e}"));
                let (lo, hi) = (real(lo)?, real(hi)?);
                if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
                    return Err(format!("literal domain {cell:?}: not a finite range"));
                }
                return Ok(Some(LiteralDomain::Real { lo, hi }));
            }
            let (lo, hi) = (int(lo)?, int(hi)?);
            if lo > hi {
                return Err(format!("literal domain {cell:?}: empty"));
            }
            return Ok(Some(LiteralDomain::Int { lo, hi }));
        }
        let mut set = cell.split(',').map(int).collect::<Result<Vec<i64>, String>>()?;
        set.sort_unstable();
        set.dedup();
        Ok(Some(LiteralDomain::Set(set)))
    }

    /// The value nearest `v` inside the domain: an integer range rounds and
    /// clamps, a set takes its nearest member (the smaller on a tie), a real
    /// range clamps; a value that is not finite is `fallback`, read into the
    /// domain likewise.
    pub fn canonical(&self, v: f64, fallback: f64) -> f64 {
        let v = if v.is_finite() { v } else { fallback };
        let v = if v.is_finite() { v } else { 0.0 };
        match self {
            LiteralDomain::Int { lo, hi } => v.round().clamp(*lo as f64, *hi as f64),
            LiteralDomain::Real { lo, hi } => v.clamp(*lo, *hi),
            LiteralDomain::Set(set) => {
                let mut best = set[0] as f64;
                for &m in set {
                    if ((m as f64) - v).abs() < (best - v).abs() {
                        best = m as f64;
                    }
                }
                best
            }
        }
    }
}

/// THE KNOB PAYLOAD'S DUAL (2026-10-17): the constant of a knob reference
/// (`literal.p_*`, `params.rs`: a payload in units of `Q`, 1 to 2^20, the
/// dual's `literal` column) is typed by its own dual, not by the row's
/// output, so a reference and an index literal (both u32.index -> u32.index
/// until then) have different signatures and a point mutation cannot swap
/// one for the other (phylu 931ac044: seed01's mutant 616 turned
/// literal.index 31153 into literal.p_wcol_m1, rendered 7787u).
pub const PAYLOAD_DUAL: &str = "u32.payload";

/// The payload dual of the safe kingdom, the only one with knob references
/// (the encoder types a payload token by it without a kingdom at hand).
pub fn payload_ty() -> Option<Ty> {
    static TY: std::sync::OnceLock<Option<Ty>> = std::sync::OnceLock::new();
    *TY.get_or_init(|| safe_duals().iter().find(|d| d.dual == PAYLOAD_DUAL).map(WgslDual::ty))
}

impl WgslDual {
    pub fn ty(&self) -> Ty {
        Ty::Wgsl(self.index)
    }

    /// The scalar inside the slot: `f32`, `f16`, `i32`, `u32`, `bool`;
    /// `None` for the array and store rows.
    pub fn scalar(&self) -> Option<&str> {
        let s = self.slot.as_str();
        let base = match s.find('<') {
            Some(i) => {
                // Only a vector or matrix slot wraps a scalar; `reg<f32>`,
                // `wgmem<f32>` and `array<T>` are not value slots.
                if !(s.starts_with("vec") || s.starts_with("mat")) {
                    return None;
                }
                &s[i + 1..s.len() - 1]
            }
            None => s,
        };
        ["f32", "f16", "i32", "u32", "bool"].contains(&base).then_some(base)
    }

    /// A statement-level dual (`kernel`, `grid`, `block`, `stmt`, a register
    /// or a workgroup array): no value slot, rendered by `render.rs`.
    pub fn is_statement_level(&self) -> bool {
        let s = self.slot.as_str();
        matches!(s, "kernel" | "grid" | "block" | "stmt" | "wg") || s.starts_with("reg<") || s.starts_with("wgmem<") || s.starts_with("parr<")
    }

    /// `Some(n)` for a vector of n lanes, `None` otherwise.
    pub fn vector_lanes(&self) -> Option<u32> {
        self.slot.starts_with("vec").then_some(self.lanes)
    }

    pub fn is_scalar(&self) -> bool {
        !self.slot.contains('<') && self.scalar().is_some()
    }

    pub fn is_matrix(&self) -> bool {
        self.slot.starts_with("mat")
    }

    /// `(columns, rows)` of a matrix slot.
    pub fn matrix_shape(&self) -> Option<(u32, u32)> {
        let s = self.slot.strip_prefix("mat")?;
        let dims = &s[..s.find('<')?];
        let (c, r) = dims.split_once('x')?;
        Some((c.parse().ok()?, r.parse().ok()?))
    }
}

/// The 83 duals, in table order.
pub fn wgsl_duals() -> Vec<WgslDual> {
    parse_duals(include_str!("../../kingdoms/wgsl/types.tsv")).unwrap_or_else(|e| panic!("kingdoms/wgsl/types.tsv does not parse: {e}"))
}

/// The safeWGSL kingdom's duals (`kingdoms/safewgsl/types.tsv`): the WGSL
/// duals, every one at its index, then the safety duals (T084 on).
pub fn safe_duals() -> Vec<WgslDual> {
    parse_duals(include_str!("../../kingdoms/safewgsl/types.tsv")).unwrap_or_else(|e| panic!("kingdoms/safewgsl/types.tsv does not parse: {e}"))
}

pub const SAFEWGSL: &str = "SAFEWGSL";

pub fn parse_duals(tsv: &str) -> Result<Vec<WgslDual>, String> {
    let mut lines = tsv.lines();
    let header: Vec<&str> = lines.next().ok_or("empty types table")?.split('\t').collect();
    let want = ["ref", "dual", "slot", "form", "lanes", "fallback", "fallback_bits", "fallback_noop", "fallback_elem", "meaning", "literal"];
    if header != want {
        return Err(format!("types.tsv header is {header:?}, want {want:?}"));
    }
    let mut out = Vec::new();
    for (n, line) in lines.enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 11 {
            return Err(format!("line {}: {} fields, want 11", n + 2, f.len()));
        }
        let num = |i: usize| f[i].parse::<u32>().map_err(|e| format!("line {}: {:?}: {e}", n + 2, f[i]));
        out.push(WgslDual {
            index: u16::try_from(n).map_err(|_| "more than 65,536 duals")?,
            reference: f[0].to_string(),
            dual: f[1].to_string(),
            slot: f[2].to_string(),
            form: f[3].to_string(),
            lanes: num(4)?,
            fallback_bits: num(6)?,
            fallback_noop: num(7)? == 1,
            fallback_elem: num(8)? == 1,
            literal: LiteralDomain::parse(f[10], f[2]).map_err(|e| format!("line {}: {e}", n + 2))?,
        });
    }
    Ok(out)
}

/// The fallback leaf of a dual, as a device node, or `None` where the table
/// says no value is substituted (`store`) or the element's is (`array<T>`).
/// The bit pattern rides in `konst` as raw bits (`f32::from_bits`), the way
/// a literal of a non-float dual is carried until phylu's `ty_code` word
/// says how to read it; an `f16` dual with a non-zero pattern is refused
/// rather than widened silently, since a 16-bit pattern in a 32-bit word has
/// no agreed placement yet.
pub fn wgsl_fallback_leaf(d: &WgslDual) -> Result<Option<GpuNode>, String> {
    if d.fallback_noop || d.fallback_elem {
        return Ok(None);
    }
    if d.scalar() == Some("f16") && d.fallback_bits != 0 {
        return Err(format!("{}: a non-zero f16 fallback ({:#x}) has no 32-bit placement", d.dual, d.fallback_bits));
    }
    Ok(Some(GpuNode { op: FN_ID_NUM, arg0: 0, ty_code: d.ty().code(), konst: f32::from_bits(d.fallback_bits) }))
}

/// The fallback table over the duals, indexed by `Ty::Wgsl(i).code() -
/// Ty::TABLE_CODE_BASE`, i.e. by dual index.
pub fn wgsl_fallback_table() -> Result<Vec<Option<GpuNode>>, String> {
    wgsl_duals().iter().map(wgsl_fallback_leaf).collect()
}

/// The fallback table over the safeWGSL duals: the WGSL table extended
/// (the first 83 duals are the WGSL kingdom's, unchanged, so this table
/// serves both kingdoms). A typed decoder without the safe entries refused
/// every gene a mutation touched at a block, statement or bounded-count
/// dual (108 of 108 offspring, fit_safe18, 2026-10-10).
pub fn safe_fallback_table() -> Result<Vec<Option<GpuNode>>, String> {
    safe_duals().iter().map(wgsl_fallback_leaf).collect()
}

/// One concrete typed row: a template instantiated at one choice of duals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgslRow {
    /// `class.instance`, the token name the kernel reader emits.
    pub name: String,
    /// The template's `F…` reference.
    pub template: String,
    /// Ordered input duals, one per child slot (what a decoder's `in_ty`
    /// needs; the many-hot `Arity` on the `Symbol` loses the order).
    pub inputs: Vec<Ty>,
    pub output: Ty,
    pub terminal: bool,
}

impl WgslRow {
    /// The unique semantic id: `class.instance` when the template has one
    /// instance, `class.instance@<out dual>` (and `@in…` when the out dual
    /// does not distinguish) otherwise.
    pub fn semantic_id(&self, duals: &[WgslDual], n_instances: usize) -> String {
        if n_instances == 1 {
            return self.name.clone();
        }
        let show = |t: &Ty| match t {
            Ty::Wgsl(i) => duals[*i as usize].dual.clone(),
            other => format!("{other:?}"),
        };
        let ins: Vec<String> = self.inputs.iter().map(show).collect();
        format!("{}@{}<-{}", self.name, show(&self.output), ins.join(","))
    }
}

/// The templates the loader does not instantiate because their rows are per
/// kernel: the reader names them from the kernel's own buffers, arguments,
/// locals and swizzle patterns.
pub fn per_kernel_templates() -> &'static [&'static str] {
    &[
        "load.buffer",
        "load.uniform",
        "load.local",
        "arg.<name>",
        "store.buffer",
        "store.local",
        "shape.swizzle",
        "shape.access_dyn",
    ]
}

/// The instantiated kingdom: its rows and the duals they are typed over.
#[derive(Debug, Clone)]
pub struct WgslKingdom {
    pub duals: Vec<WgslDual>,
    pub rows: Vec<WgslRow>,
    /// Templates that produced no row, with the reason (a per-kernel row, or
    /// a pattern the expander has no rule for), so a silent drop is
    /// impossible.
    pub uninstantiated: BTreeMap<String, String>,
}

impl WgslKingdom {
    pub fn load() -> WgslKingdom {
        instantiate(&FunctionTable::shipped(), wgsl_duals())
    }

    /// The safeWGSL kingdom: the WGSL rows plus the rows that make a
    /// crashing or hanging kernel unrepresentable (`kingdoms/safewgsl`).
    pub fn load_safe() -> WgslKingdom {
        instantiate(&FunctionTable::safe(), safe_duals())
    }

    /// The rows as a `SymbolTable` kingdom: `symbol = i + 1` for a function,
    /// `-(i + 1)` for a terminal, many-hot arity keyed by `Ty::Wgsl`.
    pub fn symbol_table(&self) -> SymbolTable {
        let mut t = SymbolTable::new();
        let mut per_name: BTreeMap<&str, usize> = BTreeMap::new();
        for r in &self.rows {
            *per_name.entry(r.name.as_str()).or_default() += 1;
        }
        let (mut nf, mut nt) = (0i64, 0i64);
        for r in &self.rows {
            let mut arity = Arity::default();
            for i in &r.inputs {
                *arity.inputs.entry(*i).or_default() += 1;
            }
            arity.outputs.insert(r.output, 1);
            let symbol = if r.terminal {
                nt += 1;
                -nt
            } else {
                nf += 1;
                nf
            };
            let instance = r.name.split_once('.').map(|(_, i)| i.to_string()).unwrap_or_else(|| r.name.clone());
            t.push(Symbol {
                kingdom: WGSL.to_string(),
                symbol,
                symbol_name: r.name.clone(),
                alias: instance,
                semantic_id: r.semantic_id(&self.duals, per_name[r.name.as_str()]),
                arity,
            });
        }
        t
    }

    pub fn dual(&self, ty: Ty) -> Option<&WgslDual> {
        match ty {
            Ty::Wgsl(i) => self.duals.get(i as usize),
            _ => None,
        }
    }

    /// THE DUAL OF A LITERAL ROW's constant: a knob reference's
    /// (`literal.p_*`, `params::parse_ref`) the payload dual
    /// ([`PAYLOAD_DUAL`]), any other `literal.*` row its output dual. `None`
    /// for a row that is not a literal, or a knob reference in a kingdom
    /// without the payload dual.
    pub fn payload_dual(&self, row: &str, out: Ty) -> Option<Ty> {
        if !row.starts_with("literal.") {
            return None;
        }
        if super::params::parse_ref(row).is_some() {
            return self.duals.iter().find(|d| d.dual == PAYLOAD_DUAL).map(WgslDual::ty);
        }
        Some(out)
    }

    /// THE DOMAIN OF A LITERAL ROW's value: its constant's dual's
    /// ([`WgslKingdom::payload_dual`]) `literal` column. `None` for a row that
    /// is not a literal or a dual no literal takes.
    pub fn literal_domain(&self, row: &str, out: Ty) -> Option<LiteralDomain> {
        self.dual(self.payload_dual(row, out)?)?.literal.clone()
    }
}

/// `wgsl_table()`: the kingdom as rows of the master symbol table.
pub fn wgsl_table() -> SymbolTable {
    WgslKingdom::load().symbol_table()
}

// ---------------------------------------------------------------------------
// Instantiation
// ---------------------------------------------------------------------------

/// A pattern of the `in`/`out` columns, parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Pat {
    /// A dual written out: `f32.real`, `vec2<u32>.hash64`.
    Exact(String),
    /// `<S.real>`, `<S.c>`, `<u32.index>`, `<i32.int>`, `<u32.bits>`:
    /// a scalar family (`S` = any scalar; or one named scalar) over scalar
    /// and vector lanes, of a form (`c` = any).
    Family { scalar: Option<String>, form: String },
    /// `<vecL.c>`, `<vecL<f32>.real>`, `<vec2.c>`, `<vecR<f32>.real>` …:
    /// a vector of L lanes (`None` = any 2–4), of a scalar (`None` = any),
    /// of a form.
    Vector { lanes: Option<u32>, scalar: Option<String>, form: String, lane_var: Option<char> },
    /// `<matCxR<f32>.real>`, `<matNxN<f32>.real>`, `<mat2xR<f32>.real>`.
    Matrix { cols: Dim, rows: Dim },
    /// `<bool.flag per lane>`: bool.flag with the lanes of the first input.
    FlagPerLane,
    /// `<T.form of the buffer>` and friends: per kernel.
    PerKernel(String),
    /// `<array<T>.c>`: per kernel.
    Array,
    /// `store`.
    Store,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Dim {
    Fixed(u32),
    Var(char),
}

fn parse_pat(p: &str) -> Result<Pat, String> {
    if p == "store" {
        return Ok(Pat::Store);
    }
    if !p.starts_with('<') {
        return Ok(Pat::Exact(p.to_string()));
    }
    let inner = &p[1..p.len() - 1];
    if inner == "bool.flag per lane" {
        return Ok(Pat::FlagPerLane);
    }
    if inner.contains(" of the ") || inner.starts_with("T.") {
        return Ok(Pat::PerKernel(inner.to_string()));
    }
    if inner.starts_with("array<") {
        return Ok(Pat::Array);
    }
    let (slot, form) = inner.rsplit_once('.').ok_or_else(|| format!("pattern {p:?} has no form"))?;
    let form = form.to_string();
    if slot == "S" {
        return Ok(Pat::Family { scalar: None, form });
    }
    if let Some(rest) = slot.strip_prefix("vec") {
        // `L`, `K`, `R`, `C` (a lane variable), `2`..`4`, with an optional `<f32>`.
        let (lanes_part, scalar) = match rest.split_once('<') {
            Some((l, s)) => (l, Some(s.trim_end_matches('>').to_string())),
            None => (rest, None),
        };
        let (lanes, lane_var) = match lanes_part.parse::<u32>() {
            Ok(n) => (Some(n), None),
            Err(_) => (None, lanes_part.chars().next()),
        };
        return Ok(Pat::Vector { lanes, scalar, form, lane_var });
    }
    if let Some(rest) = slot.strip_prefix("mat") {
        let dims = rest.split_once('<').map(|(d, _)| d).unwrap_or(rest);
        let (c, r) = dims.split_once('x').ok_or_else(|| format!("matrix pattern {p:?}"))?;
        let dim = |s: &str| match s.parse::<u32>() {
            Ok(n) => Dim::Fixed(n),
            Err(_) => Dim::Var(s.chars().next().unwrap_or('?')),
        };
        return Ok(Pat::Matrix { cols: dim(c), rows: dim(r) });
    }
    // `<u32.index>`, `<i32.int>`, `<u32.bits>`: one scalar, over lanes.
    Ok(Pat::Family { scalar: Some(slot.to_string()), form })
}

/// Which forms a `c` (any-form) pattern ranges over, per class: comparison
/// excludes bits, hash and opaque and allows the codes only for equality;
/// everything else ranges over every scalar-or-vector form.
fn forms_for_c(template: &Template) -> Vec<&'static str> {
    const ALL: &[&str] = &["real", "int", "q15_16", "index", "count", "code4", "code8", "code16", "code1024", "bits", "hash32", "flag", "sign", "opaque"];
    const ORDERED: &[&str] = &["real", "int", "q15_16", "index", "count"];
    const EQUAL: &[&str] = &["real", "int", "q15_16", "index", "count", "code4", "code8", "code16", "code1024", "flag", "sign"];
    match (template.class.as_str(), template.instance.as_str()) {
        ("compare", "eq" | "ne") => EQUAL.to_vec(),
        ("compare", _) => ORDERED.to_vec(),
        _ => ALL.to_vec(),
    }
}

/// Every instantiation of one template: a binding of each pattern to a dual.
fn instantiate_template(t: &Template, duals: &[WgslDual], by_name: &BTreeMap<&str, usize>) -> Result<Vec<WgslRow>, String> {
    let ins: Vec<Pat> = t.inputs.iter().map(|p| parse_pat(p)).collect::<Result<_, _>>()?;
    let out = parse_pat(&t.output)?;
    let all: Vec<&Pat> = ins.iter().chain(std::iter::once(&out)).collect();
    if all.iter().any(|p| matches!(p, Pat::PerKernel(_) | Pat::Array)) {
        return Err("per-kernel".into());
    }
    if t.class == "shape" && t.instance == "swizzle" {
        return Err("per-kernel".into());
    }
    let exact = |name: &str| -> Result<Ty, String> {
        by_name.get(name).map(|&i| Ty::Wgsl(i as u16)).ok_or_else(|| format!("no dual {name:?}"))
    };
    // The candidate duals a pattern admits, as (dual index, lanes).
    let candidates = |p: &Pat, form_c: &str| -> Vec<usize> {
        duals
            .iter()
            .enumerate()
            .filter(|(_, d)| d.scalar().is_some() && !d.is_matrix())
            .filter(|(_, d)| match p {
                Pat::Family { scalar, form } => {
                    (form == "c" || *form == d.form) && (form != "c" || d.form == form_c) && scalar.as_ref().is_none_or(|s| d.scalar() == Some(s.as_str()))
                }
                Pat::Vector { lanes, scalar, form, .. } => {
                    d.vector_lanes().is_some()
                        && lanes.is_none_or(|l| d.lanes == l)
                        && (form == "c" || *form == d.form)
                        && (form != "c" || d.form == form_c)
                        && scalar.as_ref().is_none_or(|s| d.scalar() == Some(s.as_str()))
                }
                _ => false,
            })
            .map(|(i, _)| i)
            .collect()
    };
    let uses_c = all.iter().any(|p| matches!(p, Pat::Family { form, .. } | Pat::Vector { form, .. } if form == "c"));
    let forms: Vec<&str> = if uses_c { forms_for_c(t) } else { vec![""] };
    let mut rows = Vec::new();
    for form_c in forms {
        // The first family/vector input (or the output, for a terminal) fixes
        // the lanes and the scalar; the rest follow with lanes equal.
        let anchor = ins.iter().find(|p| matches!(p, Pat::Family { .. } | Pat::Vector { .. })).or(match &out {
            Pat::Family { .. } | Pat::Vector { .. } => Some(&out),
            _ => None,
        });
        // `shape.access_N`: the note says the row exists only for L > N.
        let min_lanes = t.instance.strip_prefix("access_").and_then(|n| n.parse::<u32>().ok()).map(|n| n + 1);
        let scalar_anchor = t.class == "shape" && matches!(anchor, Some(Pat::Family { .. }));
        let anchors: Vec<Option<usize>> = match anchor {
            Some(p) => candidates(p, form_c)
                .into_iter()
                .filter(|&i| min_lanes.is_none_or(|m| duals[i].lanes >= m))
                .filter(|&i| !scalar_anchor || duals[i].lanes == 1)
                .map(Some)
                .collect(),
            None => vec![None],
        };
        for a in anchors {
            let bind = |p: &Pat, a: Option<usize>| -> Result<Ty, String> {
                match p {
                    Pat::Exact(name) => exact(name),
                    Pat::Store => exact("store.store"),
                    Pat::Family { scalar, form } | Pat::Vector { scalar, form, .. } => {
                        let ad = &duals[a.ok_or("a family pattern with no anchor")?];
                        let scalar = scalar.clone().unwrap_or_else(|| ad.scalar().unwrap_or("?").to_string());
                        let form = if form == "c" { form_c.to_string() } else { form.clone() };
                        // `<vec2.c>` is fixed; `<vecL.c>` follows the anchor's lanes; a
                        // family `<S.c>` follows them too, EXCEPT in the `shape` class,
                        // where `S` is the scalar a vector is built from or taken apart
                        // into (`access_N: <vecL.c> → <S.c>`, `splatN: <S.c> → <vecN.c>`).
                        let lanes = match p {
                            Pat::Vector { lanes: Some(l), .. } => *l,
                            Pat::Family { .. } if t.class == "shape" => 1,
                            _ => ad.lanes,
                        };
                        let slot = if lanes == 1 { scalar } else { format!("vec{lanes}<{scalar}>") };
                        exact(&format!("{slot}.{form}"))
                    }
                    Pat::FlagPerLane => {
                        let ad = &duals[a.ok_or("per-lane flag with no anchor")?];
                        let slot = if ad.lanes == 1 { "bool".to_string() } else { format!("vec{}<bool>", ad.lanes) };
                        exact(&format!("{slot}.flag"))
                    }
                    Pat::Matrix { .. } => Err("matrix in a non-matrix template".into()),
                    Pat::PerKernel(_) | Pat::Array => Err("per-kernel".into()),
                }
            };
            // Matrix templates are instantiated over every matrix dual
            // consistent with their dimension variables.
            if all.iter().any(|p| matches!(p, Pat::Matrix { .. })) {
                rows.extend(instantiate_matrix(t, &ins, &out, duals, by_name)?);
                break;
            }
            let inputs: Result<Vec<Ty>, String> = ins.iter().map(|p| bind(p, a)).collect();
            let output = bind(&out, a);
            match (inputs, output) {
                (Ok(inputs), Ok(output)) => rows.push(WgslRow {
                    name: t.name(),
                    template: t.reference.clone(),
                    inputs,
                    output,
                    terminal: t.arity == 0,
                }),
                // A binding the table has no dual for (e.g. `vec3<bool>.sign`)
                // is not a row; the pattern simply does not reach it.
                (Err(e), _) | (_, Err(e)) if e.starts_with("no dual") => {}
                (Err(e), _) | (_, Err(e)) => return Err(e),
            }
        }
    }
    rows.sort_by(|a, b| (a.output, &a.inputs).cmp(&(b.output, &b.inputs)));
    rows.dedup();
    if rows.is_empty() {
        return Err("no dual satisfies the pattern".into());
    }
    Ok(rows)
}

fn instantiate_matrix(t: &Template, ins: &[Pat], out: &Pat, duals: &[WgslDual], by_name: &BTreeMap<&str, usize>) -> Result<Vec<WgslRow>, String> {
    let mats: Vec<&WgslDual> = duals.iter().filter(|d| d.is_matrix()).collect();
    let exact = |name: &str| by_name.get(name).map(|&i| Ty::Wgsl(i as u16));
    let mut rows = Vec::new();
    // Enumerate bindings of dimension variables from the matrix duals (2..4).
    let dims = [2u32, 3, 4];
    for &a in &dims {
        for &b in &dims {
            for &c in &dims {
                // WGSL spells a matrix matCxR (C columns, R rows). K is the contracted dimension of a product.
                let env: BTreeMap<char, u32> = [('A', a), ('B', b), ('C', c), ('R', b), ('N', a), ('L', a), ('K', a)].into_iter().collect();
                let dim = |d: &Dim| match d {
                    Dim::Fixed(n) => Some(*n),
                    Dim::Var(v) => env.get(v).copied(),
                };
                let bind = |p: &Pat| -> Option<Ty> {
                    match p {
                        Pat::Matrix { cols, rows } => {
                            let (cn, rn) = (dim(cols)?, dim(rows)?);
                            let name = format!("mat{cn}x{rn}<f32>.real");
                            mats.iter().find(|m| m.dual == name).and_then(|m| exact(&m.dual))
                        }
                        Pat::Vector { lanes, lane_var, .. } => {
                            let n = match (lanes, lane_var) {
                                (Some(n), _) => *n,
                                (None, Some(v)) => *env.get(v)?,
                                _ => return None,
                            };
                            exact(&format!("vec{n}<f32>.real"))
                        }
                        Pat::Exact(name) => exact(name),
                        _ => None,
                    }
                };
                let inputs: Option<Vec<Ty>> = ins.iter().map(bind).collect();
                let output = bind(out);
                // `NxN` is square by construction: both dims read the one
                // variable `N` from the same binding.
                if let (Some(inputs), Some(output)) = (inputs, output) {
                    rows.push(WgslRow { name: t.name(), template: t.reference.clone(), inputs, output, terminal: false });
                }
            }
        }
    }
    rows.sort_by(|a, b| (a.output, &a.inputs).cmp(&(b.output, &b.inputs)));
    rows.dedup();
    Ok(rows)
}

/// Instantiate every template of `table` over `duals`.
pub fn instantiate(table: &FunctionTable, duals: Vec<WgslDual>) -> WgslKingdom {
    let by_name: BTreeMap<&str, usize> = duals.iter().enumerate().map(|(i, d)| (d.dual.as_str(), i)).collect();
    let mut rows = Vec::new();
    let mut uninstantiated = BTreeMap::new();
    for t in &table.rows {
        match instantiate_template(t, &duals, &by_name) {
            Ok(r) => rows.extend(r),
            Err(why) => {
                uninstantiated.insert(t.name(), why);
            }
        }
    }
    WgslKingdom { duals, rows, uninstantiated }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::render::Tree;

    /// THE LITERAL COLUMN PARSES AND CANONICALISES: `-` is none, a range is an
    /// integer range (a real range on f32), a set is sorted; canonical rounds
    /// and clamps an integer, clamps a real, takes a set's nearest member (the
    /// smaller on a tie), and reads a value that is not finite as the fallback.
    #[test]
    fn a_literal_domain_parses_and_reads_any_value_into_itself() {
        assert_eq!(LiteralDomain::parse("-", "u32"), Ok(None));
        let int = LiteralDomain::parse("0..=16777216", "u32").unwrap().unwrap();
        assert_eq!(int, LiteralDomain::Int { lo: 0, hi: 16_777_216 });
        assert_eq!(LiteralDomain::parse("-1024..=1024", "f32").unwrap(), Some(LiteralDomain::Real { lo: -1024.0, hi: 1024.0 }));
        assert_eq!(LiteralDomain::parse("8,1,4,2", "u32").unwrap(), Some(LiteralDomain::Set(vec![1, 2, 4, 8])));
        assert!(LiteralDomain::parse("5..=1", "u32").is_err() && LiteralDomain::parse("a,b", "u32").is_err());
        assert_eq!((int.canonical(-3.0, 0.0), int.canonical(2.6, 0.0), int.canonical(1e12, 0.0), int.canonical(f64::NAN, 7.0)), (0.0, 3.0, 16_777_216.0, 7.0));
        let set = LiteralDomain::Set(vec![1, 2, 4, 8]);
        assert_eq!((set.canonical(3.0, 1.0), set.canonical(6.0, 1.0), set.canonical(100.0, 1.0), set.canonical(f64::INFINITY, 1.0)), (2.0, 4.0, 8.0, 1.0));
        let real = LiteralDomain::Real { lo: -1024.0, hi: 1024.0 };
        assert_eq!((real.canonical(0.25, 0.0), real.canonical(-5e9, 0.0), real.canonical(f64::NEG_INFINITY, f64::NAN)), (0.25, -1024.0, 0.0));
    }

    /// EVERY LITERAL ROW HAS A DOMAIN, AND ITS VALUES READ BACK: in the safe
    /// kingdom every `literal.*` row's value has a domain (a knob reference's
    /// payload, else its output dual's column) but a literal over an f16 dual
    /// (the renderer writes no f16 literal). For every literal row and dual met
    /// in the tracked safe kernels (logs/HGP/001/safe, v24's reference), knob
    /// references aside (their payload renders the value it scales to), the
    /// leaf is given its domain's bounds, the bounds one past (canonicalised),
    /// every set member, 200 values drawn inside, and the raw words NaN, plus
    /// and minus infinity, 0xFFFFFFFF and 0.5, each canonicalised; the kernel
    /// is rendered and must read again; the value is among that row's literals
    /// read back (a workgroup array's length at most it: the renderer halves
    /// arrays to fit 32 KB), but for a loop's trip count (lit64, lit4096),
    /// which the renderer unrolls or, at one trip, drops.
    /// A KNOB REFERENCE CANNOT BECOME AN INDEX LITERAL (phylu 931ac044,
    /// seed01's point mutant 616): phylu mints a literal row over its
    /// constant's dual ([`WgslKingdom::payload_dual`]) and a point mutation
    /// swaps a row only for one of the same signature, (constant's dual,
    /// output); no knob reference shares that signature with a literal that
    /// is not one. Every reference's constant is u32.payload, 1..=2^20.
    #[test]
    fn a_knob_reference_and_a_plain_literal_never_share_a_signature() {
        let k = WgslKingdom::load_safe();
        let payload = k.duals.iter().find(|d| d.dual == PAYLOAD_DUAL).expect("the payload dual").ty();
        let sig = |r: &WgslRow| (k.payload_dual(&r.name, r.output).expect("a literal's constant dual"), r.output);
        let literals: Vec<&WgslRow> = k.rows.iter().filter(|r| r.name.starts_with("literal.")).collect();
        let (knobs, plain): (Vec<&WgslRow>, Vec<&WgslRow>) = literals.iter().partition(|r| crate::wgsl::params::parse_ref(&r.name).is_some());
        assert!(knobs.len() >= 8 && !plain.is_empty(), "{} references, {} plain literals", knobs.len(), plain.len());
        for r in &knobs {
            assert_eq!(sig(r).0, payload, "{}", r.name);
            assert_eq!(k.literal_domain(&r.name, r.output), Some(LiteralDomain::Int { lo: 1, hi: 1 << 20 }), "{}", r.name);
            for o in &plain {
                assert_ne!(sig(r), sig(o), "{} and {} share a signature", r.name, o.name);
            }
        }
        for o in &plain {
            assert_eq!(sig(o).0, o.output, "{}", o.name);
        }
        // No row outputs or takes the payload dual: it types a constant only.
        assert!(k.rows.iter().all(|r| r.output != payload && !r.inputs.contains(&payload)));
    }

    #[test]
    fn every_literal_domain_round_trips() {
        let kingdom = WgslKingdom::load_safe();
        for r in kingdom.rows.iter().filter(|r| r.name.starts_with("literal.")) {
            let d = kingdom.dual(r.output).expect("a dual");
            let dom = kingdom.literal_domain(&r.name, r.output);
            assert!(dom.is_some() || d.scalar() == Some("f16"), "{} over {}: no literal domain", r.name, d.dual);
        }
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/logs/HGP/001/safe");
        let mut texts: Vec<(String, String)> = std::fs::read_dir(root).expect("the safe kernels").flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with(".safe.wgsl")).map(|p| (p.display().to_string(), std::fs::read_to_string(&p).expect("a kernel"))).collect();
        texts.sort();
        texts.push(("v24 reference".into(), include_str!("../../problems/v24_ingest_small/reference.wgsl").to_string()));
        fn sites(t: &Tree, at: &mut Vec<usize>, out: &mut Vec<(Vec<usize>, String, Ty)>) {
            if t.kids.is_empty() && t.name.starts_with("literal.") {
                out.push((at.clone(), t.name.clone(), t.ty));
            }
            for (i, k) in t.kids.iter().enumerate() {
                at.push(i);
                sites(k, at, out);
                at.pop();
            }
        }
        fn set_at(t: &Tree, at: &[usize], v: f32) -> Tree {
            match at.split_first() {
                None => Tree { konst: v, ..t.clone() },
                Some((&i, rest)) => {
                    let mut kids = t.kids.clone();
                    kids[i] = set_at(&t.kids[i], rest, v);
                    Tree { kids, ..t.clone() }
                }
            }
        }
        fn values_of(t: &Tree, name: &str, ty: Ty, out: &mut Vec<f32>) {
            if t.kids.is_empty() && t.name == name && t.ty == ty {
                out.push(t.konst);
            }
            for k in &t.kids {
                values_of(k, name, ty, out);
            }
        }
        let mut done: std::collections::BTreeSet<(String, Ty)> = std::collections::BTreeSet::new();
        let mut failures: Vec<String> = Vec::new();
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for (file, text) in &texts {
            let Ok(read) = crate::wgsl::safe_reader::read_safe(text, &kingdom) else { continue };
            let mut found = Vec::new();
            sites(&read.tree, &mut Vec::new(), &mut found);
            for (at, name, ty) in found {
                if crate::wgsl::params::parse_ref(&name).is_some() || !done.insert((name.clone(), ty)) {
                    continue;
                }
                let dual = kingdom.dual(ty).expect("a dual");
                let dom = kingdom.literal_domain(&name, ty).unwrap_or_else(|| panic!("{name} over {}: no domain", dual.dual));
                let fallback = match dual.scalar() {
                    Some("f32") => f64::from(f32::from_bits(dual.fallback_bits)),
                    Some("i32") => f64::from(dual.fallback_bits as i32),
                    _ => f64::from(dual.fallback_bits),
                };
                let mut raw: Vec<f64> = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::from(f32::from_bits(0xFFFF_FFFF)), 0.5];
                match &dom {
                    LiteralDomain::Int { lo, hi } => {
                        let (lo, hi) = (*lo as f64, *hi as f64);
                        raw.extend([lo, hi, lo - 1.0, hi + 1.0]);
                        raw.extend((0..200).map(|_| lo + (next() % 1_000_000) as f64 / 1_000_000.0 * (hi - lo)));
                    }
                    // Every finite f32: its bounds, past them, and values of every scale.
                    LiteralDomain::Real { lo, hi } => {
                        raw.extend([*lo, *hi, lo * 2.0, hi * 2.0]);
                        raw.extend((0..200).map(|_| {
                            let mantissa = (next() % 2_000_000) as f64 / 1_000_000.0 - 1.0;
                            mantissa * 10f64.powi((next() % 76) as i32 - 38)
                        }));
                    }
                    LiteralDomain::Set(set) => raw.extend(set.iter().map(|&m| m as f64)),
                }
                for v in raw {
                    let c = dom.canonical(v, fallback) as f32;
                    let child = set_at(&read.tree, &at, c);
                    let back = crate::wgsl::safe_karva::render_tree_with(&child, &kingdom, &read.role).and_then(|r| crate::wgsl::safe_reader::read_safe(&r.text, &kingdom).map_err(|e| format!("does not read back: {e}")));
                    match back {
                        Err(e) => failures.push(format!("{file}: {name} over {} = {c}: {}", dual.dual, e.lines().next().unwrap_or(""))),
                        Ok(b) => {
                            let mut vals = Vec::new();
                            values_of(&b.tree, &name, ty, &mut vals);
                            // A loop's trip count (lit64, lit4096) is structure: the
                            // renderer unrolls a counted loop and drops one of one trip,
                            // so the kernel reads back valid with other literals; an
                            // array's length may be halved to fit 32 KB.
                            let ok = match dual.form.as_str() {
                                "lit64" | "lit4096" => true,
                                "wglen" => vals.iter().any(|&x| x <= c && x >= 1.0),
                                _ => vals.contains(&c),
                            };
                            if !ok {
                                failures.push(format!("{file}: {name} over {} = {c} (from {v}): read back as {vals:?}", dual.dual));
                            }
                        }
                    }
                }
            }
        }
        println!("literal rows and duals tried: {}", done.len());
        assert!(done.len() >= 4, "only {} literal kinds met in the tracked kernels", done.len());
        // By literal row and dual: how many values, and the first.
        let mut by_kind: BTreeMap<String, (usize, String)> = BTreeMap::new();
        for f in &failures {
            let kind = f.split(": ").nth(1).and_then(|r| r.split(" = ").next()).unwrap_or("?").to_string();
            by_kind.entry(kind).or_insert((0, f.clone())).0 += 1;
        }
        assert!(failures.is_empty(), "{} literal values do not round-trip:\n{}", failures.len(), by_kind.values().map(|(n, first)| format!("{n} like {first}")).collect::<Vec<_>>().join("\n"));
    }

    #[test]
    fn the_83_duals_load_with_their_machine_fallbacks() {
        let d = wgsl_duals();
        assert_eq!(d.len(), 83);
        assert_eq!((d[0].reference.as_str(), d[0].dual.as_str(), d[0].ty()), ("T001", "f32.real", Ty::Wgsl(0)));
        assert_eq!(d[70].dual, "store.store");
        assert!(d[70].fallback_noop && !d[69].fallback_noop);
        assert!(d[69].fallback_elem, "T070 array<T> defers to its element");
        let sign = d.iter().find(|x| x.dual == "bool.sign").unwrap();
        assert_eq!(sign.fallback_bits, 1);
        assert_eq!(d.iter().filter(|x| x.fallback_bits != 0).count(), 1, "only sign is non-zero");
        assert_eq!(d.iter().find(|x| x.dual == "mat3x4<f32>.real").unwrap().matrix_shape(), Some((3, 4)));
        assert_eq!(d.iter().find(|x| x.dual == "vec3<u32>.index").unwrap().scalar(), Some("u32"));
        // The statement-level duals come after the value duals, so every
        // earlier ref is unchanged; their fallback is a plain leaf the
        // renderer gives a meaning (an empty statement, the first register).
        assert_eq!(d[71].dual, "kernel.kernel");
        assert_eq!(d[82].dual, "u32.unroll");
        assert!(d[71..].iter().all(|x| x.fallback_bits == 0 && !x.fallback_noop && !x.fallback_elem));
        assert_eq!(d.iter().filter(|x| x.is_statement_level()).count(), 9);
        assert_eq!(d.iter().find(|x| x.dual == "reg<f32>.real").unwrap().scalar(), None, "a register is not a value slot");
        assert_eq!(d.iter().find(|x| x.dual == "u32.wgdim").unwrap().scalar(), Some("u32"));
        let table = wgsl_fallback_table().unwrap();
        assert_eq!(table.len(), 83);
        assert!(table[70].is_none() && table[69].is_none());
        assert_eq!(table[71].unwrap().ty_code, Ty::Wgsl(71).code());
        assert_eq!(table[0].unwrap().op, FN_ID_NUM);
        assert_eq!(table[sign.index as usize].unwrap().konst.to_bits(), 1);
        // A non-zero f16 pattern is refused, not widened.
        let mut bad = d.iter().find(|x| x.dual == "f16.real").unwrap().clone();
        bad.fallback_bits = 0x3c00;
        assert!(wgsl_fallback_leaf(&bad).is_err());
    }

    /// The safeWGSL tables: the WGSL duals and rows unchanged at their refs,
    /// and the safety rules pinned: one way to make a loop condition, no
    /// barrier under a condition, no loop nested past its rung, a store to
    /// h only at an owned cell, a load of phi only at a phi index.
    /// The safe fallback table extends the WGSL one entry for entry, and has
    /// a leaf for every safe dual but the no-op and array ones.
    #[test]
    fn the_safe_fallback_table_extends_the_wgsl_one() {
        let w = wgsl_fallback_table().unwrap();
        let s = safe_fallback_table().unwrap();
        assert_eq!(s.len(), safe_duals().len());
        assert!(s.len() > w.len());
        let key = |l: &Option<GpuNode>| l.as_ref().map(|n| (n.op, n.ty_code, n.konst.to_bits()));
        assert!(s.iter().zip(&w).all(|(a, b)| key(a) == key(b)), "the WGSL prefix differs");
        let duals = safe_duals();
        for (d, leaf) in duals.iter().zip(&s).skip(w.len()) {
            assert_eq!(leaf.is_none(), d.fallback_noop || d.fallback_elem, "{}", d.dual);
        }
        assert!(s.iter().skip(w.len()).filter(|l| l.is_some()).count() >= 40);
    }

    #[test]
    fn the_safe_kingdom_keeps_wgsl_and_pins_its_rules() {
        let (w, s) = (wgsl_duals(), safe_duals());
        // 126 duals until 2026-10-11; 127 with u32.boundx (the complete kingdom, step 6);
        // 132 with the fourth nested tile loop (tcounter4, L5, C5; run A's declines);
        // 142 with the epilogue (stmt and block at O0..O4: every kernel ends in its thread's store);
        // 152 with the thread's axes (gid, lid, wid, rowid, colid, lidx; strided and grouped rows and columns);
        // 157 with the workgroup arrays as knobs (vec2 arrays, their index, a length, a declaration and the list);
        // 160 with the workgroup atomics (atomic<u32> and atomic<i32> arrays, their index);
        // 161 with the i32 register (WGSL_COVERAGE.md gap 6); 166 with the loop bodies that break (gap 7);
        // 168 with the vec4<u32> register and the vec2 private array (gap 8); 171 with the matrix
        // registers (gap 9); 172 with the 16-element private array's index (legality item 4).
        // 174 with the uniform values and the uniform loop bound (legality item 5); 176 with the
        // uniform values that may wrap (u32.uniformw) and an expression bound's value (u32.xbound).
        // 177 with u32.indexw (a raw difference, which no row takes: no wrap anywhere).
        // 183 with the origins an offset row takes (rowbase, colbase, strided, grouped).
        // 184 with the mirror of a thread's cell (u32.mirror, the symmetric half, 2026-10-14).
        // 185 with a workgroup element a thread owns (u32.wgown, the workgroup put, 2026-10-14).
        // 186 with the workgroup arrays of indices (wgmem<u32>.index, 2026-10-16).
        // 187 with the module of several entry points (kernel.module, 2026-10-17); 188 with bool.uflag;
        // 189 with the 64-element u32 private array (parr64<u32>.index); 190 with the vec2<f32>
        // register (reg<vec2<f32>>.real); 191 with the 64-element vec4 private array; 192
        // with the knob payload (u32.payload, a literal.p_* row's constant); 193 with the i32
        // workgroup arrays (wgmem<i32>.int, run 4's seeds, 2026-10-19); 195 with the 128-element u32
        // private array and its index (parr128<u32>.index, u32.pindex128, run 4's seed10).
        // 204 with the registers of the other vector types (REG_KINDS, 2026-10-19).
        assert_eq!(s.len(), 204);
        for (a, b) in w.iter().zip(&s) {
            assert_eq!((a.reference.as_str(), a.dual.as_str()), (b.reference.as_str(), b.dual.as_str()));
        }
        let k = WgslKingdom::load_safe();
        let (fw, fs) = (FunctionTable::shipped(), FunctionTable::safe());
        for (a, b) in fw.rows.iter().zip(&fs.rows) {
            assert_eq!((a.reference.as_str(), a.name()), (b.reference.as_str(), b.name()));
        }
        // 561 rows until 2026-10-11; 606 with the complete kingdom's registers and arrays (step 6);
        // 758 after run A's declines (64 f32, 32 vec4, 64 u32, 16 bool registers; the array lane rows).
        // 758 after run A's declines (64 f32, 32 vec4, 64 u32, 16 bool registers; the array lane rows);
        // 816 with the fourth nested tile loop; 822 with the clamped offsets, u32(flag) and f32(flag) (fit_A2).
        // 826 with the grouped block (fit_A2's r6x_reinvent).
        // 828 with the third and fourth vec4 private arrays (fit_A2 gen 2's r2x_reinvent);
        // 884 with the epilogue's rows (O0..O4); 908 with the thread's axes; 912 with the epilogue's continue;
        // 941 with the workgroup arrays as knobs; 955 with full unrolling; 1,009 with the tree reductions;
        // 1,082 with the workgroup atomics; 1,125 with the subgroup values, operations and barrier;
        // 1,126 with the layout knob (grid.of_rows_x); 1,157 with the u32 form crossings and the bit,
        // dot and modulo overloads (WGSL_COVERAGE.md gaps 2 and 3); 1,163 with storageBarrier;
        // 1,237 with i32 (registers, crossings, integer rows, subgroup operations; gap 6);
        // 1,272 with break and continue in loops (gap 7); 1,281 with the thread ids as vector operands
        // and the vector bitcasts (gap 8); 1,318 with the vec4<u32> registers, subgroupBallot and the
        // vec2 private arrays; 1,399 with the matrices (gap 9); 1,405 with modf / frexp's fields and
        // the bindings' lengths (gap 10); 1,454 with the kernel parameters (the knob references
        // literal.p_* and grid.with_ktile; PLAN_constrained_mutation.md task 0).
        // 1,497 with bound.x_of_uniform; 1,496 with the uniform values, the uniform bound and the in-place conditional block (legality item 5).
        // 1,522 with no wrap in a loop bound (the wrap-free rows of u32.xbound, the clamped min of a
        // uniform value that may wrap; 2026-10-12).
        // 1,523 with a literal bound value (literal.index at u32.xbound; a random kernel's bound of 0).
        // 1,529 with the thread's origin rows at the origins' duals.
        // 1,556 with the mirror of a thread's cell (9 safe.mirror_of, 12 store.h_mirror) and the
        // uniform test whose body keeps its rung (6 stmt.if_ule; the symmetric half, 2026-10-14).
        // 1,577 with the workgroup put (3 safe.wg_own*, 18 wg.put* at the barrier rungs; 2026-10-14).
        // 1,601 with the u32 workgroup arrays (wg.ua-ud, wg.get_u_safe, 12 wg.set_u, wg.decl_u,
        // 6 wg.put_u; 2026-10-16).
        // 1,604 with the modules of two, three and four entry points (kernel.module_2..4, 2026-10-17).
        // 1,622 with the uniform tests ==, != and < (18 stmt.if_ueq/une/ult, v24 run 1's seeds, 2026-10-17);
        // 1,646 with their else forms (24 stmt.if_else_ule/ueq/une/ult); 1,647 with an index read as a count;
        // 1,651 with four more u32 workgroup arrays (wg.ue..uh); 1,672 with the workgroup-uniform
        // predicate (9 uflag rows, 12 stmt.if_uflag / if_else_uflag); 1,677 with wg.e..h and a
        // workgroup u32 as a loop bound's value; 1,687 with the stepped loops that break; 1,702 with
        // the 64-element u32 private arrays (parr.ua, parr.uc, their get and 12 sets); 1,706 with
        // four more u32 workgroup arrays (wg.ui..ul, seed20 of v24 run 1); 1,727 with the vec2<f32>
        // registers (reg.w0..w7, reg.get_w and 12 reg.set_w, seed11); 1,728 with uflag.of_flag;
        // 1,729 with uniform.of_checked; 1,741 with the uniform loads (wg.uload_i, wg.uload_f at L0..L5);
        // 1,756 with the 64-element vec4 private arrays (parr.vq, parr.vr, their get and 12 sets);
        // 1,768 with the argmax and argmin reductions (wg.reduce_argmax/argmin at L0..L5); 1,770 with
        // two more atomic<u32> workgroup arrays (wg.au_c, wg.au_d, run 4's seeds, 2026-10-19); 1,792
        // with the i32 workgroup arrays (wg.na, wg.nb, wg.get_n_safe, 12 wg.set_n, wg.decl_n, 6 wg.put_n);
        // 1,808 with the 128-element u32 private arrays (safe.pidx128, parr.ux, parr.uy, get_i, 12 set_i).
        // 2,006 with the vector conversions of floats and unsigned integers (9) and the nine
        // register kinds of REG_KINDS (8 registers, the read and a set at each of the 22 rungs each);
        // 2,009 with kernel.module_ext1..3 (a module of more than four entry points).
        assert_eq!(fs.rows.len(), 2009);
        let dual = |t: Ty| k.dual(t).unwrap().dual.clone();
        let producers = |d: &str| -> Vec<String> { k.rows.iter().filter(|r| dual(r.output) == d).map(|r| r.name.clone()).collect() };
        assert_eq!(producers("kernel.safe"), vec!["kernel.entry_safe"]);
        // A module of entry points is a root only: its rows take safe kernels,
        // and no row takes a module (none nests, none is a single kernel's mutation).
        assert_eq!(producers("kernel.module"), vec!["kernel.module_2", "kernel.module_3", "kernel.module_4", "kernel.module_ext1", "kernel.module_ext2", "kernel.module_ext3"]);
        assert!(k.rows.iter().filter(|r| r.name.starts_with("kernel.module_") && !r.name.starts_with("kernel.module_ext")).all(|r| r.inputs.iter().all(|&t| dual(t) == "kernel.safe")));
        // (Only the extension rows take a module, as their first input.)
        assert!(k.rows.iter().filter(|r| !r.name.starts_with("kernel.module_ext")).all(|r| r.inputs.iter().all(|&t| dual(t) != "kernel.module")));
        // THE EPILOGUE: the root takes a body and an epilogue; every epilogue
        // block ends in an epilogue statement, and every epilogue statement is
        // the store of the thread's cell or holds an epilogue block (an if, an
        // if-else, a block, a tile loop one deeper), so every gene writes h.
        // THE THREAD'S AXES: each thread id is its own dual (one is never
        // another's mutation), its x component is a column-axis index and its
        // y a row-axis index, and only the index.of_* rows carry them into
        // plain index arithmetic; a strided or grouped block's offsets apply to
        // that block's own rows and columns only.
        for (b, d) in [("builtin.global_invocation_id", "vec3<u32>.gid"), ("builtin.local_invocation_id", "vec3<u32>.lid"), ("builtin.workgroup_id", "vec3<u32>.wid"), ("builtin.local_invocation_index", "u32.lidx")] {
            assert_eq!(producers(d), vec![b.to_string()], "{d}");
        }
        assert!(producers("u32.rowid").iter().all(|n| n.starts_with("axis.y_")) && producers("u32.colid").iter().all(|n| n.starts_with("axis.x_")));
        let consumers = |d: &str| -> Vec<String> { k.rows.iter().filter(|r| r.inputs.iter().any(|&t| dual(t) == d)).map(|r| r.name.clone()).collect() };
        assert_eq!(consumers("u32.rowid"), vec!["index.of_row"]);
        assert_eq!(consumers("u32.colid"), vec!["index.of_col"]);
        for (kind, plus) in [("u32.threadrows", "_s"), ("u32.threadcols", "_s"), ("u32.threadrowg", "_g"), ("u32.threadcolg", "_g")] {
            assert!(consumers(kind).iter().all(|n| n.ends_with(plus) || n == "safe.cell_of" || n == "safe.mirror_of" || n.ends_with("_as_index")), "{kind}: {:?}", consumers(kind));
        }
        let entry = k.rows.iter().find(|r| r.name == "kernel.entry_safe").unwrap();
        assert_eq!(entry.inputs.iter().map(|&t| dual(t)).collect::<Vec<_>>(), vec!["grid.grid", "block.L0", "block.O0"]);
        for d in 0..5 {
            let (st, bl) = (format!("stmt.O{d}"), format!("block.O{d}"));
            for r in k.rows.iter().filter(|r| dual(r.output) == bl) {
                if r.name.starts_with("block.cont_") {
                    // `if (c) { continue; }` then an epilogue block: inside an epilogue tile loop only.
                    assert!(d >= 1, "{}", r.name);
                    assert_eq!(r.inputs.iter().map(|&t| dual(t)).collect::<Vec<_>>(), vec!["bool.flag".to_string(), bl.clone()], "{}", r.name);
                    continue;
                }
                assert_eq!(dual(*r.inputs.last().unwrap()), st, "{}", r.name);
                assert!(r.inputs[..r.inputs.len() - 1].iter().all(|&t| dual(t) == format!("stmt.C{d}")), "{}", r.name);
            }
            for r in k.rows.iter().filter(|r| dual(r.output) == st) {
                let ins: Vec<String> = r.inputs.iter().map(|&t| dual(t)).collect();
                let stores = r.name.starts_with("store.h_") && ins == ["u32.cell", "f32.real"];
                let holds = ins.iter().any(|i| i == &bl || i == &format!("block.O{}", d + 1));
                assert!(stores || holds, "{} ({ins:?}) neither stores nor holds an epilogue block", r.name);
                assert!(!holds || ins.iter().filter(|i| i.starts_with("block.")).all(|i| i.starts_with("block.O")), "{}: every block it holds is an epilogue", r.name);
            }
            assert!(k.rows.iter().any(|r| dual(r.output) == st && r.name.starts_with("store.h_")));
        }
        // One kind of loop condition: a loop's own counter below one of four
        // bounds (the item loop), or a tile counter below a literal (three depths).
        let bound = producers("bool.bound");
        assert_eq!(bound.len(), 8, "{bound:?}");
        assert!(bound.iter().all(|n| n.starts_with("safe.lt_")));
        assert!(k.rows.iter().filter(|r| r.name.starts_with("safe.lt_")).all(|r| dual(r.inputs[0]) == "u32.counter" || dual(r.inputs[0]).starts_with("u32.tcounter")));
        // No barrier in a conditional context; barriers at L0..L5 only.
        // (The WGSL kingdom's own `stmt.barrier → stmt.stmt` stays; `stmt.stmt`
        // is never demanded under the safe root `kernel.entry_safe`.)
        let barriers: Vec<String> = k.rows.iter().filter(|r| r.name.starts_with("stmt.barrier")).map(|r| dual(r.output)).filter(|d| d != "stmt.stmt").collect();
        assert_eq!(barriers, vec!["stmt.L0", "stmt.L1", "stmt.L2", "stmt.L3", "stmt.L4", "stmt.L5"]);
        assert!(k.rows.iter().all(|r| !(r.name.starts_with("stmt.barrier") && dual(r.output).starts_with("stmt.C"))));
        // The item loop only produces an L0 statement; no loop at L5 or C5 (tile loops nest four deep).
        assert!(k.rows.iter().filter(|r| r.name.starts_with("loop.items_bound") || r.name.starts_with("loop.items_unrolled_bound")).all(|r| dual(r.output) == "stmt.L0"));
        assert!(k.rows.iter().filter(|r| r.name.starts_with("loop.")).all(|r| dual(r.output) != "stmt.L5" && dual(r.output) != "stmt.C5"));
        assert!(k.rows.iter().filter(|r| r.name.starts_with("loop.tile_")).all(|r| dual(r.inputs[0]) == "u32.lit64"));
        // No return in a conditional context (a later barrier would be non-uniform).
        assert!(k.rows.iter().filter(|r| r.name.starts_with("stmt.return_")).all(|r| !dual(r.output).starts_with("stmt.C")));
        // A conditional block's statements are C statements, which no loop or barrier produces.
        // (stmt.if_ule tests two uniform values: its body keeps the rung, 2026-10-14; ==, != and < too, 2026-10-17.)
        let uniform_test = |n: &str| ["stmt.if_ule_", "stmt.if_ueq_", "stmt.if_une_", "stmt.if_ult_", "stmt.if_else_ule_", "stmt.if_else_ueq_", "stmt.if_else_une_", "stmt.if_else_ult_", "stmt.if_uflag_", "stmt.if_else_uflag_"].iter().any(|p| n.starts_with(p));
        let uflag_test = |n: &str| n.starts_with("stmt.if_uflag_") || n.starts_with("stmt.if_else_uflag_");
        assert!(k.rows.iter().filter(|r| uflag_test(&r.name)).all(|r| dual(r.inputs[0]) == "bool.uflag" && r.inputs[1..].iter().all(|&b| dual(b).replace("block.", "stmt.") == dual(r.output))));
        assert!(k.rows.iter().filter(|r| uniform_test(&r.name) && !uflag_test(&r.name)).all(|r| dual(r.inputs[0]) == "u32.uniform" && dual(r.inputs[1]) == "u32.uniform" && dual(r.inputs[2]).replace("block.", "stmt.") == dual(r.output)));
        assert!(k.rows.iter().filter(|r| r.name.starts_with("stmt.if_") && !uniform_test(&r.name) && (dual(r.output).starts_with("stmt.L") || dual(r.output).starts_with("stmt.C"))).all(|r| dual(r.inputs[1]).starts_with("block.C")));
        // store.h only at an owned cell; load.phi only at a phi index; both produced by one row each.
        assert!(k.rows.iter().filter(|r| r.name.starts_with("store.h_") && !r.name.starts_with("store.h_mirror_")).all(|r| dual(r.inputs[0]) == "u32.cell"));
        // (and the symmetric half's mirror store at the mirror of one, 2026-10-14)
        assert!(k.rows.iter().filter(|r| r.name.starts_with("store.h_mirror_")).all(|r| dual(r.inputs[0]) == "u32.mirror"));
        assert!(producers("u32.mirror").iter().all(|n| n == "safe.mirror_of") && producers("u32.mirror").len() == 9);
        // (one cell_of per pair of block kinds, nine since the thread's axes)
        assert!(producers("u32.cell").iter().all(|n| n == "safe.cell_of") && producers("u32.cell").len() == 9);
        assert_eq!(producers("u32.wgindex"), vec!["safe.wg_of"]);
        assert_eq!(producers("u32.wgvindex"), vec!["safe.wgv_of"]);
        assert!(k.rows.iter().filter(|r| r.name.starts_with("loop.items_strided")).all(|r| dual(r.inputs[1]) == "u32.lit4096"));
        // An expression-bounded loop takes the bound dual only `bound.of_index` produces.
        assert_eq!(producers("u32.boundx"), vec!["bound.of_index", "bound.x_of_uniform"]);
        // A loop that may hold a barrier (an expression-bounded loop at an L rung that does not break)
        // takes the uniform bound only `bound.of_uniform` produces; every other expression-bounded
        // loop (C and O rungs, loops that break) takes `u32.boundx` (legality item 5).
        assert_eq!(producers("u32.boundu"), vec!["bound.of_uniform"]);
        for r in k.rows.iter().filter(|r| r.name.starts_with("loop.tilex_") || r.name.ends_with("_boundx")) {
            let barrier_capable = dual(r.output).starts_with("stmt.L") && !r.name.contains("_brk_");
            let want = if barrier_capable { "u32.boundu" } else { "u32.boundx" };
            assert!(r.inputs.iter().any(|i| dual(*i) == want), "{} takes {want}", r.name);
        }
    }

    #[test]
    fn every_template_is_instantiated_or_named_as_per_kernel() {
        let k = WgslKingdom::load();
        let per_kernel: Vec<&String> = k.uninstantiated.iter().filter(|(_, why)| why.as_str() == "per-kernel").map(|(n, _)| n).collect();
        let other: Vec<(&str, &str)> = k.uninstantiated.iter().filter(|(_, why)| why.as_str() != "per-kernel").map(|(n, w)| (n.as_str(), w.as_str())).collect();
        // No template is left without a rule: the vector count duals (T022–T024)
        // were added for `builtin.num_workgroups`, which was the one gap.
        assert!(other.is_empty(), "templates with no instantiation rule: {other:?}");
        let expected: std::collections::BTreeSet<&str> = per_kernel_templates().iter().copied().collect();
        let got: std::collections::BTreeSet<&str> = per_kernel.iter().map(|s| s.as_str()).collect();
        assert_eq!(got, expected, "the per-kernel set is exactly the documented one");
        println!("WGSL kingdom: {} rows from {} templates ({} per-kernel)", k.rows.len(), 239 - k.uninstantiated.len(), per_kernel.len());
        assert!(k.rows.len() > 239);
    }

    #[test]
    fn rows_are_typed_over_duals_with_lanes_equal_and_arity_at_most_k_max() {
        let k = WgslKingdom::load();
        for r in &k.rows {
            assert!(r.inputs.len() <= crate::gpu_eval::K_MAX, "{}: {} inputs", r.name, r.inputs.len());
            let out = k.dual(r.output).unwrap_or_else(|| panic!("{}: output is not a dual", r.name));
            for i in &r.inputs {
                let d = k.dual(*i).unwrap_or_else(|| panic!("{}: input is not a dual", r.name));
                // Lanes equal across inputs and output, except the rows that say otherwise.
                let shape_changing = r.name.starts_with("shape.") || r.name.starts_with("geom.") || r.name.starts_with("bits.") || r.name.starts_with("arith.mul_mat") || r.name == "arith.mul_scalar_vec" || r.name == "arith.mix_scalar_weight" || r.name.starts_with("hash.") || r.name.starts_with("fixed.") || r.name.starts_with("quantise.") || r.name.starts_with("convert.") || r.name.starts_with("select.") || r.name.starts_with("logic.") || r.name.starts_with("store.") || r.name.starts_with("index.") || r.name.starts_with("compare.") || r.name.starts_with("wg.") || r.name.starts_with("reg.");
                if !shape_changing && out.lanes > 0 && d.lanes > 0 {
                    assert_eq!(d.lanes, out.lanes, "{}: {} vs {}", r.name, d.dual, out.dual);
                }
            }
        }
        assert_eq!(k.rows.iter().map(|r| r.inputs.len()).max(), Some(crate::gpu_eval::K_MAX), "grid.of takes four");
    }

    #[test]
    fn the_forms_only_cross_in_the_classes_the_spec_names() {
        let k = WgslKingdom::load();
        let mut crossing: std::collections::BTreeSet<String> = Default::default();
        for r in &k.rows {
            let out = &k.dual(r.output).unwrap().form;
            if r.inputs.iter().any(|i| &k.dual(*i).unwrap().form != out) {
                crossing.insert(r.name.split('.').next().unwrap().to_string());
            }
        }
        // convert is the designed crossing; compare/logic produce flags;
        // bits counts produce counts; quantise and hash land in codes,
        // indices and reals; index/count/fixed/select/store take a second
        // form as a count, an int, a sign or a condition.
        // The statement classes cross by nature: a condition and a block
        // make a statement, statements make a block, a grid and a block a
        // kernel, a register and a value a statement, an array and an
        // index a value.
        let allowed: std::collections::BTreeSet<String> = ["convert", "compare", "logic", "bits", "quantise", "hash", "index", "count", "fixed", "select", "store", "kernel", "grid", "block", "stmt", "loop", "reg", "wg"].iter().map(|s| s.to_string()).collect();
        assert!(crossing.is_subset(&allowed), "forms cross in {crossing:?}, allowed {allowed:?}");
        assert!(crossing.contains("convert"));
        assert!(!crossing.contains("arith") && !crossing.contains("trig") && !crossing.contains("int"), "{crossing:?}");
    }

    #[test]
    fn the_named_instances_come_out_as_the_spec_reads() {
        let k = WgslKingdom::load();
        let by = |name: &str| -> Vec<&WgslRow> { k.rows.iter().filter(|r| r.name == name).collect() };
        let dual = |t: Ty| k.dual(t).unwrap().dual.clone();
        // arith.add: the eight real duals, lanes equal.
        let add = by("arith.add");
        assert_eq!(add.len(), 8);
        assert!(add.iter().all(|r| r.inputs.len() == 2 && r.inputs[0] == r.inputs[1] && r.inputs[0] == r.output));
        // index.add: scalar and vector u32.index.
        assert_eq!(by("index.add").len(), 4);
        // count.add: one row (its template names u32.count exactly; the vector count duals serve the builtin).
        assert_eq!(by("count.add").len(), 1);
        // compare.lt over the ordered forms, scalar and vector, out flag per lane.
        let lt = by("compare.lt");
        assert!(lt.iter().all(|r| dual(r.output).ends_with(".flag")));
        assert!(lt.iter().any(|r| dual(r.inputs[0]) == "vec3<f32>.real" && dual(r.output) == "vec3<bool>.flag"));
        assert!(!lt.iter().any(|r| dual(r.inputs[0]).ends_with(".bits")), "no ordering on bits");
        // select.scalar_cond: (value, value, flag) over every form.
        let sel = by("select.scalar_cond");
        assert!(sel.iter().all(|r| r.inputs.len() == 3 && r.inputs[0] == r.inputs[1] && r.inputs[0] == r.output && dual(r.inputs[2]) == "bool.flag"));
        assert!(sel.iter().any(|r| dual(r.output) == "u32.index"));
        // geom.dot: vec2/3/4<f32> → f32.
        assert_eq!(by("geom.dot").len(), 3);
        // arith.mul_mat_vec: matCxR × vecC → vecR for every matrix.
        let mv = by("arith.mul_mat_vec");
        assert_eq!(mv.len(), 9);
        assert!(mv.iter().any(|r| dual(r.inputs[0]) == "mat2x3<f32>.real" && dual(r.inputs[1]) == "vec2<f32>.real" && dual(r.output) == "vec3<f32>.real"));
        // arith.mul_mat_mat: the left's columns equal the right's rows (WGSL matCxR), 27 shapes.
        let mm = by("arith.mul_mat_mat");
        assert_eq!(mm.len(), 27);
        assert!(mm.iter().all(|r| {
            let (l, rt, o) = (k.dual(r.inputs[0]).unwrap().matrix_shape().unwrap(), k.dual(r.inputs[1]).unwrap().matrix_shape().unwrap(), k.dual(r.output).unwrap().matrix_shape().unwrap());
            l.0 == rt.1 && o == (rt.0, l.1)
        }), "a product's shapes");
        // geom.determinant: square matrices only.
        assert_eq!(by("geom.determinant").len(), 3);
        // literal.real is a terminal over the eight real duals; builtins are terminals.
        assert!(by("literal.real").iter().all(|r| r.terminal) && by("literal.real").len() == 8);
        assert_eq!(by("builtin.global_invocation_id").len(), 1);
        // shape.access_2 exists only for 3 and 4 lanes, and takes a vector apart into its scalar.
        assert!(by("shape.access_2").iter().all(|r| k.dual(r.inputs[0]).unwrap().lanes >= 3 && k.dual(r.output).unwrap().lanes == 1));
        assert!(by("shape.access_0").iter().any(|r| dual(r.inputs[0]) == "vec3<u32>.index" && dual(r.output) == "u32.index"));
        // splat3 builds a vec3 from a scalar, never from a vector.
        assert!(by("shape.splat3").iter().all(|r| k.dual(r.inputs[0]).unwrap().lanes == 1 && k.dual(r.output).unwrap().lanes == 3));
        assert!(by("shape.compose2").iter().all(|r| k.dual(r.inputs[0]).unwrap().lanes == 1 && k.dual(r.output).unwrap().lanes == 2));
        // convert.f32_to_q15_16 is one row.
        assert_eq!(by("convert.f32_to_q15_16").len(), 1);
        // The statement rows: one each, typed over the statement duals; the
        // value rows did not pick up a register or an array as a value.
        assert_eq!(by("kernel.entry").len(), 1);
        assert_eq!(by("grid.of")[0].inputs.len(), 4);
        assert_eq!(by("block.seq4")[0].inputs.len(), 4);
        assert_eq!(by("loop.items").len(), 1);
        assert!(by("stmt.return")[0].terminal && by("stmt.barrier")[0].terminal);
        assert!(by("reg.f0")[0].terminal && dual(by("reg.f0")[0].output) == "reg<f32>.real");
        assert_eq!(dual(by("reg.get_f")[0].output), "f32.real");
        assert_eq!(dual(by("reg.set_v")[0].inputs[1]), "vec4<f32>.real");
        assert!(by("arith.add").iter().all(|r| !dual(r.output).starts_with("reg")), "no arithmetic on registers");
        assert!(!by("select.scalar_cond").iter().any(|r| dual(r.output).ends_with(".wgdim")), "no select over the literal codes");
        assert_eq!(dual(by("wg.get")[0].inputs[0]), "wgmem<f32>.real");
        assert!(dual(by("literal.wgdim")[0].output) == "u32.wgdim" && by("literal.wgdim")[0].terminal);
        // The symbol table view: kingdom name, unique semantic ids, k-hot keys are duals.
        let t = k.symbol_table();
        assert_eq!(t.kingdoms(), vec![WGSL.to_string()]);
        assert_eq!(t.len(), k.rows.len());
        let mut ids: Vec<&str> = t.kingdom(WGSL).iter().map(|s| s.semantic_id.as_str()).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "semantic ids are unique");
        assert!(t.kingdom(WGSL).iter().all(|s| s.arity.inputs.keys().chain(s.arity.outputs.keys()).all(|ty| matches!(ty, Ty::Wgsl(_)))));
        assert_eq!(t.max_arity(WGSL), 4);
    }
}
