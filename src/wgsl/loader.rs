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
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

impl WgslDual {
    pub fn ty(&self) -> Ty {
        Ty::Wgsl(self.index)
    }

    /// The scalar inside the slot: `f32`, `f16`, `i32`, `u32`, `bool`;
    /// `None` for the array and store rows.
    pub fn scalar(&self) -> Option<&str> {
        let s = self.slot.as_str();
        if s == "array<T>" || s == "store" {
            return None;
        }
        Some(match s.find('<') {
            Some(i) => &s[i + 1..s.len() - 1],
            None => s,
        })
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

/// The 71 duals, in table order.
pub fn wgsl_duals() -> Vec<WgslDual> {
    parse_duals(include_str!("../../kingdoms/wgsl/types.tsv")).unwrap_or_else(|e| panic!("kingdoms/wgsl/types.tsv does not parse: {e}"))
}

pub fn parse_duals(tsv: &str) -> Result<Vec<WgslDual>, String> {
    let mut lines = tsv.lines();
    let header: Vec<&str> = lines.next().ok_or("empty types table")?.split('\t').collect();
    let want = ["ref", "dual", "slot", "form", "lanes", "fallback", "fallback_bits", "fallback_noop", "fallback_elem", "meaning"];
    if header != want {
        return Err(format!("types.tsv header is {header:?}, want {want:?}"));
    }
    let mut out = Vec::new();
    for (n, line) in lines.enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 10 {
            return Err(format!("line {}: {} fields, want 10", n + 2, f.len()));
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
    Ok(Some(GpuNode { op: FN_ID_NUM, arg0: 0, arg1: 0, konst: f32::from_bits(d.fallback_bits) }))
}

/// The fallback table over the duals, indexed by `Ty::Wgsl(i).code() -
/// Ty::TABLE_CODE_BASE`, i.e. by dual index.
pub fn wgsl_fallback_table() -> Result<Vec<Option<GpuNode>>, String> {
    wgsl_duals().iter().map(wgsl_fallback_leaf).collect()
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
                let env: BTreeMap<char, u32> = [('A', a), ('B', b), ('C', c), ('R', b), ('N', a), ('L', a)].into_iter().collect();
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

    #[test]
    fn the_68_duals_load_with_their_machine_fallbacks() {
        let d = wgsl_duals();
        assert_eq!(d.len(), 71);
        assert_eq!((d[0].reference.as_str(), d[0].dual.as_str(), d[0].ty()), ("T001", "f32.real", Ty::Wgsl(0)));
        assert_eq!(d[70].dual, "store.store");
        assert!(d[70].fallback_noop && !d[69].fallback_noop);
        assert!(d[69].fallback_elem, "T070 array<T> defers to its element");
        let sign = d.iter().find(|x| x.dual == "bool.sign").unwrap();
        assert_eq!(sign.fallback_bits, 1);
        assert_eq!(d.iter().filter(|x| x.fallback_bits != 0).count(), 1, "only sign is non-zero");
        assert_eq!(d.iter().find(|x| x.dual == "mat3x4<f32>.real").unwrap().matrix_shape(), Some((3, 4)));
        assert_eq!(d.iter().find(|x| x.dual == "vec3<u32>.index").unwrap().scalar(), Some("u32"));
        let table = wgsl_fallback_table().unwrap();
        assert_eq!(table.len(), 71);
        assert!(table[70].is_none() && table[69].is_none());
        assert_eq!(table[0].unwrap().op, FN_ID_NUM);
        assert_eq!(table[sign.index as usize].unwrap().konst.to_bits(), 1);
        // A non-zero f16 pattern is refused, not widened.
        let mut bad = d.iter().find(|x| x.dual == "f16.real").unwrap().clone();
        bad.fallback_bits = 0x3c00;
        assert!(wgsl_fallback_leaf(&bad).is_err());
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
        println!("WGSL kingdom: {} rows from {} templates ({} per-kernel)", k.rows.len(), 197 - k.uninstantiated.len(), per_kernel.len());
        assert!(k.rows.len() > 197);
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
                let shape_changing = r.name.starts_with("shape.") || r.name.starts_with("geom.") || r.name.starts_with("bits.") || r.name.starts_with("arith.mul_mat") || r.name == "arith.mul_scalar_vec" || r.name == "arith.mix_scalar_weight" || r.name.starts_with("hash.") || r.name.starts_with("fixed.") || r.name.starts_with("quantise.") || r.name.starts_with("convert.") || r.name.starts_with("select.") || r.name.starts_with("logic.") || r.name.starts_with("store.") || r.name.starts_with("index.") || r.name.starts_with("compare.");
                if !shape_changing && out.lanes > 0 && d.lanes > 0 {
                    assert_eq!(d.lanes, out.lanes, "{}: {} vs {}", r.name, d.dual, out.dual);
                }
            }
        }
        assert_eq!(k.rows.iter().map(|r| r.inputs.len()).max(), Some(3));
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
        let allowed: std::collections::BTreeSet<String> = ["convert", "compare", "logic", "bits", "quantise", "hash", "index", "count", "fixed", "select", "store"].iter().map(|s| s.to_string()).collect();
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
        assert_eq!(t.max_arity(WGSL), 3);
    }
}
