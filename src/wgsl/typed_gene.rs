//! The WGSL kingdom on the typed core (rev 4), fuller side: a read safe-kingdom
//! tree encoded into a typed gene (`crate::typed_gep`) and expressed back.
//!
//! A kernel is one gene rooted at `kernel.safe`, or at `kernel.module` for
//! several entry points (the module row takes the safe kernels, so no linker is
//! needed). The alphabet is the kingdom's rows plus the kernel interface's
//! per-kernel rows (its buffer loads and stores), one symbol each, a polymorphic
//! row one symbol per output dual. Heads are the exact `h_T` the tree needs
//! ([`crate::typed_gep::required_heads`]); the cells are the write-back of
//! `typed_gep::write_back`.
//!
//! Constants are indices into `C_T`, one append-only pool per dual holding each
//! constant's EXACT BIT PATTERN as a `u32` (an f32's bits, a u32, an i32's two's
//! complement, a flag's 0 or 1, an f16's 16 bits): never an f32 `konst`.
//!
//! Anything that does not encode, express or render back is a [`Gap`], named.

use std::collections::BTreeMap;

use crate::geneframe::Ty;
use crate::kingdom_contract::{relations_from_rows, KingdomContract, SymbolId};
use crate::typed_gep::{self, Domain, GeneLayout, Tree as Typed};
use crate::wgsl::loader::{WgslDual, WgslKingdom};
use crate::wgsl::render::{Role, Tree};

/// A named kingdom gap: the grouping `reason` (numbers stripped) and the `detail`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gap {
    pub reason: String,
    pub detail: String,
}

fn gap(reason: &str, detail: String) -> Gap {
    Gap { reason: reason.to_string(), detail }
}

/// The kingdom and one interface's per-kernel rows as a checked typed contract.
pub struct Alphabet {
    pub contract: KingdomContract,
    /// Per symbol: the WGSL row name, its inputs and output.
    rows: Vec<(String, Vec<Ty>, Ty)>,
    by_key: BTreeMap<(String, Ty, Vec<Ty>), SymbolId>,
}

impl Alphabet {
    pub fn new(kingdom: &WgslKingdom, role: &Role) -> Result<Alphabet, Gap> {
        // A module's size is the interface's: only the module row of that many entry points is in the alphabet (a
        // module of 4 under an interface of 2 is not a kernel of this interface, and a gene must not name it).
        let wanted = role.entries.len();
        let own_size = |name: &str| -> bool {
            match name.strip_prefix("kernel.module_") {
                Some(n) if n.starts_with("ext") => wanted > 4,
                Some(n) => n.parse::<usize>().is_ok_and(|n| n == wanted || wanted > 4),
                None => true,
            }
        };
        // (And only the rows the interface can spell: a row that reads a field the interface's uniform does not have, or
        // the Gram's input, output cell or fields, is no symbol of this interface: it rendered to a name the module
        // does not declare, `render::role_spells`. Run 6i: 292 of its 558 illegal children, `uniform.cfg_m` through
        // `load.cfg_m`, `bound.d`, `cell_of`, under v24's interface.)
        let kept: Vec<&crate::wgsl::loader::WgslRow> = kingdom.rows.iter().filter(|r| own_size(&r.name) && crate::wgsl::render::role_spells(kingdom, role, r)).collect();
        let mut rows: Vec<(String, Vec<Ty>, Ty)> = kept.iter().map(|r| (r.name.clone(), r.inputs.clone(), r.output)).collect();
        let mut per_name: BTreeMap<&str, usize> = BTreeMap::new();
        for r in &kingdom.rows {
            *per_name.entry(r.name.as_str()).or_default() += 1;
        }
        let mut ids: Vec<(String, Vec<Ty>, Ty)> = kept.iter().map(|r| (r.semantic_id(&kingdom.duals, per_name[r.name.as_str()]), r.inputs.clone(), r.output)).collect();
        // The reference's own uses of its bindings first, then every access the interface ALLOWS (a load of every storage
        // binding, a store of every writable one at every statement rung, the atomic rows): any kernel over the same
        // bindings is nameable, not just the one the interface was read from.
        let mut seen: std::collections::BTreeSet<(String, Ty, Vec<Ty>)> = std::collections::BTreeSet::new();
        for (name, ins, out) in crate::wgsl::safe_karva::per_kernel_rows(kingdom, role).into_iter().chain(capability_rows(kingdom, role)) {
            if !seen.insert((name.clone(), out, ins.clone())) {
                continue;
            }
            rows.push((name.clone(), ins.clone(), out));
            ids.push((format!("kernel:{name}"), ins, out));
        }
        // The rows the interface's symbols can all be built from: a row with an input type no kept row produces (a loop over
        // `cfg.m` where the interface has no `m`) is no symbol of this interface either, to a fixed point. The instantiated
        // kingdom, once, before any gene exists.
        let mut producible: std::collections::BTreeSet<Ty> = std::collections::BTreeSet::new();
        loop {
            let before = producible.len();
            for (_, ins, out) in &rows {
                if ins.iter().all(|t| producible.contains(t)) {
                    producible.insert(*out);
                }
            }
            if producible.len() == before {
                break;
            }
        }
        let usable: Vec<bool> = rows.iter().map(|(_, ins, _)| ins.iter().all(|t| producible.contains(t))).collect();
        let mut at = 0;
        rows.retain(|_| {
            at += 1;
            usable[at - 1]
        });
        let mut at = 0;
        ids.retain(|_| {
            at += 1;
            usable[at - 1]
        });
        let constants: Vec<&str> = ids.iter().zip(&rows).filter(|((_, ins, _), (n, _, _))| ins.is_empty() && (n.starts_with("literal.") || n.starts_with("const."))).map(|((id, _, _), _)| id.as_str()).collect();
        let roots: Vec<Ty> = kingdom.duals.iter().filter(|d| (d.dual == "kernel.safe" || d.dual == "kernel.module") && rows.iter().any(|(_, _, o)| *o == d.ty())).map(WgslDual::ty).collect();
        let contract = relations_from_rows("safeWGSL", roots, &ids, &constants).check().map_err(|d| gap("kingdom check", d.join("; ")))?;
        let mut by_key = BTreeMap::new();
        for (i, (name, ins, out)) in rows.iter().enumerate() {
            by_key.entry((name.clone(), *out, ins.clone())).or_insert(i as SymbolId);
        }
        Ok(Alphabet { contract, rows, by_key })
    }

    pub fn row(&self, s: SymbolId) -> &(String, Vec<Ty>, Ty) {
        &self.rows[s as usize]
    }
}

/// The per-kernel rows of every access an interface's bindings allow: a load of each storage array at an index, a store
/// of each `read_write` one at every statement rung, the length of a runtime array, and the rows of an array of atomics.
fn capability_rows(kingdom: &WgslKingdom, role: &Role) -> Vec<crate::wgsl::safe_karva::PerKernelRow> {
    use crate::wgsl::render::{atomic_array_access, dual_ty, elem_dual, ATOMIC_FUNS, STORE_RUNGS};
    let ty = |n: &str| dual_ty(kingdom, n).ok();
    let mut out = Vec::new();
    for b in &role.bindings {
        if !b.space.starts_with("storage") {
            continue;
        }
        let writable = b.space == "storage, read_write";
        let Some(inner) = b.ty.strip_prefix("array<").and_then(|r| r.strip_suffix('>')) else { continue };
        let index = ty("u32.index");
        if let Some((value, _)) = atomic_array_access(b) {
            let (Some(index), Some(value)) = (index, ty(value)) else { continue };
            out.push((format!("load.{}", b.name), vec![index], value));
            for r in STORE_RUNGS {
                if let Some(stmt) = ty(&format!("stmt.{r}")) {
                    out.push((format!("store.{}_{r}", b.name), vec![index, value], stmt));
                }
            }
            for (fun, _) in ATOMIC_FUNS {
                out.push((format!("atomic.{}_{fun}", b.name), vec![index, value], value));
            }
            continue;
        }
        let elem = inner.split(", ").next().unwrap_or(inner);
        let (Some(index), Some(dual)) = (index, elem_dual(elem).and_then(|d| ty(&d))) else { continue };
        out.push((format!("load.{}", b.name), vec![index], dual));
        // (The length of the interface's own input and output is the kingdom's row, `load.phi_len` / `load.h_len`.)
        if !inner.contains(", ") && b.name != role.input && b.name != role.output {
            if let Some(count) = ty("u32.count") {
                out.push((format!("len.{}", b.name), Vec::new(), count));
            }
        }
        if !writable && elem == "u32" {
            if let Some(xbound) = ty("u32.xbound") {
                out.push((format!("load.{}", b.name), vec![index], xbound));
            }
        }
        if writable {
            for r in STORE_RUNGS {
                if let Some(stmt) = ty(&format!("stmt.{r}")) {
                    out.push((format!("store.{}_{r}", b.name), vec![index, dual], stmt));
                }
            }
        }
    }
    out
}

/// The exact bit pattern of a constant of `dual` (see the module header).
pub fn const_bits(dual: &WgslDual, konst: f64) -> Result<u32, Gap> {
    let whole = |lo: f64, hi: f64| {
        if konst.is_finite() && konst.fract() == 0.0 && konst >= lo && konst <= hi {
            Ok(konst)
        } else {
            Err(gap("constant out of range", format!("{konst} for {}", dual.dual)))
        }
    };
    match dual.scalar() {
        Some("f32") => Ok((konst as f32).to_bits()),
        Some("f16") => Ok(u32::from(f32_to_f16_bits(konst as f32))),
        Some("u32") => whole(0.0, f64::from(u32::MAX)).map(|v| v as u32),
        Some("i32") => whole(f64::from(i32::MIN), f64::from(i32::MAX)).map(|v| (v as i32) as u32),
        Some("bool") => Ok(u32::from(konst != 0.0)),
        other => Err(gap("constant of a dual with no scalar", format!("{other:?} for {}", dual.dual))),
    }
}

/// The constant a pool entry holds.
pub fn const_value(dual: &WgslDual, bits: u32) -> Result<f64, Gap> {
    match dual.scalar() {
        Some("f32") => Ok(f64::from(f32::from_bits(bits))),
        Some("f16") => Ok(f64::from(f16_bits_to_f32(bits as u16))),
        Some("u32") => Ok(f64::from(bits)),
        Some("i32") => Ok(f64::from(bits as i32)),
        Some("bool") => Ok(f64::from(bits != 0)),
        other => Err(gap("constant of a dual with no scalar", format!("{other:?} for {}", dual.dual))),
    }
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = (m >> shift) as u16;
        let round = (m >> (shift - 1)) & 1;
        let sticky = m & ((1 << (shift - 1)) - 1);
        return sign | (half + u16::from(round == 1 && (sticky != 0 || half & 1 == 1)));
    }
    let half = sign | ((e as u16) << 10) | (man >> 13) as u16;
    let round = (man >> 12) & 1;
    let sticky = man & 0x0fff;
    half + u16::from(round == 1 && (sticky != 0 || half & 1 == 1))
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x03ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            let shift = m.leading_zeros() - 21;
            sign | ((127 - 15 + 1 - shift) << 23) | (((m << shift) & 0x03ff) << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// A kernel as a typed gene: its layout, words and the constant pools.
#[derive(Debug, Clone)]
pub struct Encoded {
    pub layout: GeneLayout,
    pub gene: Vec<u32>,
    /// `C_T`: the bit patterns, append-only, per dual.
    pub pools: BTreeMap<Ty, Vec<u32>>,
}

impl Encoded {
    pub fn const_len(&self) -> BTreeMap<Ty, u32> {
        self.pools.iter().map(|(t, v)| (*t, v.len() as u32)).collect()
    }
}

/// A read tree as typed nodes of the alphabet.
fn typed_of(a: &Alphabet, kingdom: &WgslKingdom, t: &Tree, pools: &mut BTreeMap<Ty, Vec<u32>>) -> Result<Typed, Gap> {
    let dual_name = |ty: Ty| kingdom.dual(ty).map_or_else(|| format!("{ty:?}"), |d| d.dual.clone());
    let kids = t.kids.iter().map(|k| typed_of(a, kingdom, k, pools)).collect::<Result<Vec<_>, _>>()?;
    let ins: Vec<Ty> = t.kids.iter().map(|k| k.ty).collect();
    let symbol = *a.by_key.get(&(t.name.clone(), t.ty, ins.clone())).ok_or_else(|| {
        let ins: Vec<String> = ins.iter().map(|i| dual_name(*i)).collect();
        let known: Vec<String> = a.by_key.keys().filter(|(n, _, _)| *n == t.name).map(|(_, o, i)| format!("{} <- {}", dual_name(*o), i.iter().map(|x| dual_name(*x)).collect::<Vec<_>>().join(","))).collect();
        gap("no row", format!("{} {} <- {} (the alphabet has: {})", t.name, dual_name(t.ty), ins.join(","), known.join(" | ")))
    })?;
    let const_index = if a.contract.is_constant(symbol) {
        let d = kingdom.dual(t.ty).ok_or_else(|| gap("constant of an unknown dual", t.name.clone()))?;
        let bits = const_bits(d, t.konst)?;
        let pool = pools.entry(t.ty).or_insert_with(|| vec![0]);
        let i = pool.iter().position(|b| *b == bits).unwrap_or_else(|| {
            pool.push(bits);
            pool.len() - 1
        });
        Some(i as u32)
    } else {
        None
    };
    Ok(Typed { symbol, const_index, kids })
}

/// Encode a read kernel tree into a GIVEN layout (a population's gene layout), appending its constants to the
/// population's pools. A tree the layout cannot hold (`j_T > h_T`, a tail or Dc too short) is a named
/// [`Gap`] (`reason` "write-back", the defect in `detail`), never a repair; `pools` is left as it was.
pub fn encode_into(a: &Alphabet, kingdom: &WgslKingdom, tree: &Tree, layout: &GeneLayout, pools: &mut BTreeMap<Ty, Vec<u32>>) -> Result<Vec<u32>, Gap> {
    let k = &a.contract;
    let mut trial = pools.clone();
    let typed = typed_of(a, kingdom, tree, &mut trial)?;
    let mut filler = vec![0u32; layout.width];
    for s in layout.spans() {
        for w in &mut filler[s.off..s.off + s.len] {
            *w = match s.domain {
                Domain::Head | Domain::Tail => k.terminals_of(s.ty).iter().copied().find(|t| !k.is_constant(*t)).or_else(|| k.terminals_of(s.ty).first().copied()).ok_or_else(|| gap("layout", format!("no terminal of {:?}", s.ty)))?,
                Domain::Dc => 0,
            };
        }
    }
    for f in &layout.frames {
        if f.dc_off.is_some() {
            trial.entry(f.ty).or_insert_with(|| vec![0]);
        }
    }
    let gene = typed_gep::write_back(k, layout, &filler, &typed).map_err(|e| gap("write-back", e.to_string()))?;
    *pools = trial;
    Ok(gene)
}

/// Encode a read kernel tree into a typed gene.
pub fn encode(a: &Alphabet, kingdom: &WgslKingdom, tree: &Tree) -> Result<Encoded, Gap> {
    let mut pools: BTreeMap<Ty, Vec<u32>> = BTreeMap::new();
    let typed = typed_of(a, kingdom, tree, &mut pools)?;
    let k = &a.contract;
    let root = k.output(typed.symbol);
    let heads = typed_gep::required_heads(k, &typed);
    let layout = GeneLayout::from_heads(k, root, &heads).map_err(|e| gap("layout", e.to_string()))?;
    // Every constant-capable type with a Dc holds at least the constant 0, the filler's index.
    for f in &layout.frames {
        if f.dc_off.is_some() {
            pools.entry(f.ty).or_insert_with(|| vec![0]);
        }
    }
    let mut filler = vec![0u32; layout.width];
    for s in layout.spans() {
        for w in &mut filler[s.off..s.off + s.len] {
            *w = match s.domain {
                Domain::Head | Domain::Tail => k.terminals_of(s.ty).iter().copied().find(|t| !k.is_constant(*t)).or_else(|| k.terminals_of(s.ty).first().copied()).ok_or_else(|| gap("layout", format!("no terminal of {:?}", s.ty)))?,
                Domain::Dc => 0,
            };
        }
    }
    let gene = typed_gep::write_back(k, &layout, &filler, &typed).map_err(|e| gap("write-back", e.to_string()))?;
    Ok(Encoded { layout, gene, pools })
}

/// THE HOST TYPED INITIALISER: a gene of `layout` drawn cell by cell, every draw taken from the alphabet of the cell
/// it writes, from the run's seeded generator `state`. A head cell is a function of the cell's type or a terminal of it
/// by a fair coin (the design's `FunctionShare::half`; the kingdom's types hold hundreds of functions against tens of
/// terminals, so a draw in proportion to their numbers expresses over a thousand nodes), then uniform within the chosen
/// kind; a tail cell: any terminal of the type; a Dc cell: any index of the type's pool `C_T`. No cell is drawn and then
/// changed: the gene is a program of the kingdom whatever the draws are. `C_T` holds, for each constant-capable type,
/// the constant 0 and seven values drawn from the dual's literal domain.
pub fn random_gene(a: &Alphabet, kingdom: &WgslKingdom, layout: &GeneLayout, state: &mut u64) -> Result<Encoded, Gap> {
    use crate::wgsl::render::{drawn_constant, next_draw};
    let k = &a.contract;
    let mut pools: BTreeMap<Ty, Vec<u32>> = BTreeMap::new();
    for f in layout.frames.iter().filter(|f| f.dc_off.is_some()) {
        let d = kingdom.dual(f.ty).ok_or_else(|| gap("constant of an unknown dual", format!("{:?}", f.ty)))?;
        let mut pool = vec![0u32];
        for _ in 0..7 {
            let u = (next_draw(state) >> 11) as f64 / (1u64 << 53) as f64;
            pool.push(const_bits(d, drawn_constant(d, u))?);
        }
        pools.insert(f.ty, pool);
    }
    let mut gene = vec![0u32; layout.width];
    for s in layout.spans() {
        let (functions, terminals): (&[u32], Vec<u32>) = match s.domain {
            Domain::Head => (k.functions_of(s.ty), k.terminals_of(s.ty).to_vec()),
            Domain::Tail => (&[], k.terminals_of(s.ty).to_vec()),
            Domain::Dc => (&[], (0..pools.get(&s.ty).map_or(0, Vec::len) as u32).collect()),
        };
        if terminals.is_empty() && functions.is_empty() {
            return Err(gap("layout", format!("no symbol for the {:?} cells of {:?}", s.domain, s.ty)));
        }
        for w in &mut gene[s.off..s.off + s.len] {
            let pick_function = !functions.is_empty() && (terminals.is_empty() || next_draw(state) & 1 == 0);
            let kind: &[u32] = if pick_function { functions } else { &terminals };
            *w = kind[(next_draw(state) % kind.len() as u64) as usize];
        }
    }
    Ok(Encoded { layout: layout.clone(), gene, pools })
}

/// Express an encoded kernel back into a render tree.
pub fn express(a: &Alphabet, kingdom: &WgslKingdom, e: &Encoded) -> Result<Tree, Gap> {
    let x = typed_gep::express(&a.contract, &e.layout, &e.gene, &e.const_len()).map_err(|d| gap("express", d.to_string()))?;
    fn back(a: &Alphabet, kingdom: &WgslKingdom, e: &Encoded, t: &Typed) -> Result<Tree, Gap> {
        let (name, _, ty) = a.row(t.symbol);
        let mut out = Tree::new(name, *ty, t.kids.iter().map(|k| back(a, kingdom, e, k)).collect::<Result<Vec<_>, _>>()?);
        if let Some(i) = t.const_index {
            let d = kingdom.dual(*ty).ok_or_else(|| gap("constant of an unknown dual", name.clone()))?;
            let bits = *e.pools.get(ty).and_then(|p| p.get(i as usize)).ok_or_else(|| gap("express", format!("constant {i} past C_T")))?;
            out.konst = const_value(d, bits)?;
        }
        Ok(out)
    }
    back(a, kingdom, e, &x.tree())
}

/// The alphabets of the interfaces met, by their per-kernel rows.
#[derive(Default)]
pub struct Alphabets {
    by_interface: BTreeMap<String, Result<Alphabet, Gap>>,
}

impl Alphabets {
    pub fn of(&mut self, kingdom: &WgslKingdom, role: &Role) -> &Result<Alphabet, Gap> {
        // (The aggregate duals a module declared are part of the alphabet.)
        // (The rows over them too: two modules may declare a struct of one name with other members.)
        let added = kingdom.duals.iter().filter(|d| d.reference.starts_with(['A', 'R'])).map(|d| d.dual.as_str()).collect::<Vec<_>>().join(",");
        let rows = kingdom.rows.iter().filter(|r| r.name.starts_with("agg.") || r.name.starts_with("reg.") && r.name[4..].starts_with(|c: char| c.is_ascii_uppercase()) || r.name.starts_with("shape.col_")).map(|r| format!("{}:{:?}:{:?}", r.name, r.inputs, r.output)).collect::<Vec<_>>().join(",");
        let key = format!("{}|{}|{added}|{rows}|{}", role.name, role.per_kernel.iter().map(|p| format!("{}:{}:{}", p.name, p.inputs.join("+"), p.output)).collect::<Vec<_>>().join(","), role.entries.len());
        self.by_interface.entry(key).or_insert_with(|| Alphabet::new(kingdom, role))
    }
}

/// What the round trip of one kernel's text came to.
#[derive(Debug)]
pub enum Outcome {
    /// The safe reader refused the text (a reader gap, not the encoder's).
    NotRead(String),
    /// The read tree does not render (the original text cannot be compared).
    Unrenderable(String),
    /// A named gap in the typed core.
    Gap(Gap),
    /// Encoded and expressed: the tree equal to the read one, the text equal to
    /// the read tree's, and naga's verdict on that text.
    Expressed { tree_equal: bool, text_equal: bool, valid: Result<(), String>, difference: String },
}

/// A tree encoded into a typed gene and expressed again: the tree back, as the alphabet normalises it, or the named
/// gap.
pub fn through_the_gene(alphabets: &mut Alphabets, kingdom: &WgslKingdom, role: &Role, tree: &Tree) -> Result<(), String> {
    let a = alphabets.of(kingdom, role).as_ref().map_err(|g| format!("{}: {}", g.reason, g.detail))?;
    let enc = encode(a, kingdom, tree).map_err(|g| format!("{}: {}", g.reason, g.detail))?;
    let back = express(a, kingdom, &enc).map_err(|g| format!("{}: {}", g.reason, g.detail))?;
    if back == *tree {
        Ok(())
    } else {
        Err(format!("the gene expresses another tree: {}", first_difference(tree, &back)))
    }
}

/// WGSL text, read, encoded, expressed, rendered (design test 5).
pub fn round_trip(alphabets: &mut Alphabets, kingdom: &WgslKingdom, text: &str) -> Outcome {
    use crate::wgsl::render::validate_text;
    use crate::wgsl::safe_karva::render_tree_with;
    let read = match crate::wgsl::safe_reader::read_safe(text, kingdom) {
        Ok(r) => r,
        Err(e) => return Outcome::NotRead(e),
    };
    // (A module that declares aggregate types reads in the kingdom extended by them.)
    let kingdom = read.kingdom.as_ref().unwrap_or(kingdom);
    let original = match render_tree_with(&read.tree, kingdom, &read.role) {
        Ok(r) => r.text,
        Err(e) => return Outcome::Unrenderable(e),
    };
    let a = match alphabets.of(kingdom, &read.role) {
        Ok(a) => a,
        Err(g) => return Outcome::Gap(g.clone()),
    };
    let enc = match encode(a, kingdom, &read.tree) {
        Ok(e) => e,
        Err(g) => return Outcome::Gap(g),
    };
    let back = match express(a, kingdom, &enc) {
        Ok(t) => t,
        Err(g) => return Outcome::Gap(g),
    };
    let difference = first_difference(&read.tree, &back);
    match render_tree_with(&back, kingdom, &read.role) {
        Err(e) => Outcome::Gap(gap("render of the expressed tree", e)),
        Ok(r) => Outcome::Expressed { tree_equal: back == read.tree, text_equal: r.text == original, valid: validate_text(&r.text), difference },
    }
}

fn first_difference(a: &Tree, b: &Tree) -> String {
    if a.name != b.name || a.ty != b.ty || a.konst != b.konst || a.kids.len() != b.kids.len() {
        return format!("{} {:?} {} / {} {:?} {}", a.name, a.ty, a.konst, b.name, b.ty, b.konst);
    }
    a.kids.iter().zip(&b.kids).map(|(x, y)| first_difference(x, y)).find(|d| !d.is_empty()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_constant_is_its_exact_bit_pattern() {
        let k = WgslKingdom::load_safe();
        let dual = |n: &str| k.duals.iter().find(|d| d.dual == n).expect("dual");
        for (d, v) in [("u32.index", f64::from(u32::MAX)), ("u32.index", 2_246_822_507.0), ("i32.int", f64::from(i32::MIN)), ("i32.int", -1.0), ("f32.real", 0.1f32 as f64), ("f32.real", -0.0), ("bool.flag", 1.0)] {
            let bits = const_bits(dual(d), v).expect("bits");
            assert_eq!(const_value(dual(d), bits).expect("value").to_bits(), v.to_bits(), "{d} {v}");
        }
        assert!(const_bits(dual("u32.index"), -1.0).is_err() && const_bits(dual("u32.index"), 0.5).is_err());
        for h in [0u16, 0x3c00, 0xbc00, 0x7bff, 0x0001, 0x03ff, 0x3555] {
            assert_eq!(f32_to_f16_bits(f16_bits_to_f32(h)), h, "{h:#x}");
        }
    }

    /// Design test 5 on the committed corpora: every kernel the safe reader reads is encoded into a typed gene,
    /// expressed, and rendered to the same text, naga-valid. (The model replies: the first 300 of the 1,798.)
    #[test]
    fn every_kernel_the_reader_reads_round_trips_through_a_typed_gene() {
        let kingdom = WgslKingdom::load_safe();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut alphabets = Alphabets::default();
        let (mut read, mut total) = (0usize, 0usize);
        let mut failures: Vec<String> = Vec::new();
        let mut outside: Vec<String> = Vec::new();
        for (dir, limit) in [("tests/corpus/safewgsl", usize::MAX), ("tests/corpus/nluca", usize::MAX), ("tests/corpus/interfaces", usize::MAX), ("tests/corpus/model_replies", 300)] {
            let mut files: Vec<_> = std::fs::read_dir(root.join(dir)).expect("the corpus").flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
            files.sort();
            for f in files.into_iter().take(limit) {
                total += 1;
                match round_trip(&mut alphabets, &kingdom, &std::fs::read_to_string(&f).expect("a kernel")) {
                    Outcome::NotRead(_) | Outcome::Unrenderable(_) => {}
                    // A kernel that returns before the output store its interface demands has no spelling: the Gram's
                    // alphabet has no `return` symbol. Named, and counted below.
                    Outcome::Gap(g) if g.reason == "no row" && g.detail.starts_with("stmt.return_") => {
                        read += 1;
                        outside.push(f.file_name().map_or_else(String::new, |n| n.to_string_lossy().to_string()));
                    }
                    Outcome::Gap(g) => {
                        read += 1;
                        failures.push(format!("{}: {}: {}", f.display(), g.reason, g.detail));
                    }
                    Outcome::Expressed { tree_equal, text_equal, valid, difference } => {
                        read += 1;
                        if !(tree_equal && text_equal && valid.is_ok()) {
                            failures.push(format!("{}: tree {tree_equal} text {text_equal} valid {valid:?} {difference}", f.display()));
                        }
                    }
                }
            }
        }
        println!("{read} of {total} kernels read; {} do not round trip; outside the typed kingdom (a return before the store): {outside:?}", failures.len());
        assert_eq!(outside, vec!["draft_no_store.wgsl".to_string()], "the kernels that return before the output store");
        assert!(read > 400, "only {read} kernels read");
        assert!(failures.is_empty(), "{} kernels do not round trip through a typed gene:\n{}", failures.len(), failures.join("\n"));
    }

    /// A given layout holds the tree it was made for, and names the head it is short of for a layout one row smaller.
    #[test]
    fn a_given_layout_holds_what_fits_and_names_what_does_not() {
        let kingdom = WgslKingdom::load_safe();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(root.join("tests/corpus/nluca/").join(std::fs::read_dir(root.join("tests/corpus/nluca")).expect("dir").flatten().map(|e| e.file_name()).filter(|n| n.to_string_lossy().ends_with(".wgsl")).min().expect("a kernel"))).expect("text");
        let read = crate::wgsl::safe_reader::read_safe(&text, &kingdom).expect("reads");
        let mut alphabets = Alphabets::default();
        let a = alphabets.of(&kingdom, &read.role).as_ref().expect("alphabet");
        let enc = encode(a, &kingdom, &read.tree).expect("encodes");
        let mut pools = BTreeMap::new();
        let gene = encode_into(a, &kingdom, &read.tree, &enc.layout, &mut pools).expect("fits its own layout");
        assert_eq!(gene, enc.gene);
        assert_eq!(pools, enc.pools);
        // One head row fewer for a type that has functions.
        let mut heads = enc.layout.head_lens();
        let (ty, h) = heads.iter().map(|(t, h)| (*t, *h)).find(|(_, h)| *h > 0).expect("a head");
        heads.insert(ty, h - 1);
        let short = GeneLayout::from_heads(&a.contract, enc.layout.root, &heads).expect("layout");
        let mut kept = pools.clone();
        let g = encode_into(a, &kingdom, &read.tree, &short, &mut kept).expect_err("one head row short");
        assert_eq!(g.reason, "write-back");
        assert!(g.detail.contains("head rows"), "{}", g.detail);
        assert_eq!(kept, pools, "pools untouched on a refusal");
    }

    /// THE SAVED GENOMES, RE-ENCODED (the 481 dumps of `popA2_g006j/karva`, `library_ep` and `safe96/karva`): each is
    /// today's reading of its text (safe_karva's test), and each encodes into a rev-4 typed gene, expresses and renders
    /// to the same text, naga-valid. From the tracked fixtures tests/fixtures/wgsl_genomes (T63).
    #[test]
    fn the_saved_genomes_round_trip_through_typed_genes() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wgsl_genomes");
        let kingdom = WgslKingdom::load_safe();
        let mut alphabets = Alphabets::default();
        let (mut total, mut failures) = (0usize, Vec::new());
        // (set, dumps expected): a dump is `<id>.json.gz` beside its text `texts/<id>.wgsl`.
        for (set, expected) in [("popA2_g006j", 20usize), ("library_ep", 430), ("safe96", 31)] {
            let dumps = root.join(set).join("dumps");
            let entries = std::fs::read_dir(&dumps).unwrap_or_else(|e| panic!("saved-genome fixture {} is missing: {e}", dumps.display()));
            let mut files: Vec<_> = entries.flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with(".json.gz")).collect();
            files.sort();
            assert_eq!(files.len(), expected, "{set}: {} dumps in the fixture, expected {expected}", files.len());
            for f in files {
                let id = f.file_name().map(|s| s.to_string_lossy().trim_end_matches(".json.gz").to_string()).unwrap_or_default();
                let text = std::fs::read_to_string(root.join(set).join("texts").join(format!("{id}.wgsl"))).unwrap_or_else(|e| panic!("{set}/{id}: its text: {e}"));
                total += 1;
                match round_trip(&mut alphabets, &kingdom, &text) {
                    Outcome::Expressed { tree_equal: true, text_equal: true, valid: Ok(()), .. } => {}
                    other => failures.push(format!("{set}/{id}: {other:?}")),
                }
            }
        }
        println!("saved genomes: {} of {total} round trip through a typed gene", total - failures.len());
        assert!(failures.is_empty(), "{} of {total}:\n{}", failures.len(), failures.iter().take(10).cloned().collect::<Vec<_>>().join("\n"));
    }

    /// A kernel that computes in f16 reads, encodes into a typed gene (f16 duals, constants as f16 bit patterns), expresses and
    /// renders to the same text, which enables f16 and validates.
    #[test]
    fn an_f16_kernel_round_trips_through_a_typed_gene() {
        let kingdom = WgslKingdom::load_safe();
        let text = "enable f16;\n@group(0) @binding(0) var<storage, read> src: array<f32>;\n@group(0) @binding(1) var<storage, read_write> dst: array<f32>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let h = f16(src[gid.x]) * f16(2.5) + f16(1.0);\n    dst[gid.x] = f32(h);\n}\n";
        match round_trip(&mut Alphabets::default(), &kingdom, text) {
            Outcome::Expressed { tree_equal: true, text_equal: true, valid: Ok(()), .. } => {}
            other => panic!("{other:?}"),
        }
    }

    /// The alphabet of an interface holds every access its bindings allow, not only the ones the kernel it was read from
    /// made: a kernel that only reads `dst` still has `store.dst_<rung>`, and another kernel that writes it encodes.
    #[test]
    fn the_alphabet_of_an_interface_names_every_access_its_bindings_allow() {
        let kingdom = WgslKingdom::load_safe();
        let head = "@group(0) @binding(0) var<storage, read> src: array<f32>;\n@group(0) @binding(1) var<storage, read_write> dst: array<f32>;\n@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n";
        let reads = format!("{head}    let x = dst[gid.x] + src[gid.x];\n    _ = x;\n}}\n");
        let writes = format!("{head}    dst[gid.x] = src[gid.x] * 2.0;\n    atomicAdd(&hist[gid.x & 7u], 1u);\n}}\n");
        let role = crate::wgsl::safe_reader::read_safe(&reads, &kingdom).expect("reads").role;
        let mut alphabets = Alphabets::default();
        let a = alphabets.of(&kingdom, &role).as_ref().expect("alphabet");
        let named = |n: &str| (0..a.contract.n_symbols() as SymbolId).any(|s| a.row(s).0 == n);
        for n in ["load.dst", "store.dst_O0", "store.dst_C3", "atomic.hist_add", "store.hist_L0", "len.dst"] {
            assert!(named(n), "{n}");
        }
        assert!(!named("store.src_O0"), "a read-only binding is never stored");
        match round_trip(&mut alphabets, &kingdom, &writes) {
            Outcome::Expressed { tree_equal: true, text_equal: true, valid: Ok(()), .. } => {}
            Outcome::NotRead(e) => panic!("does not read: {e}"),
            other => panic!("{other:?}"),
        }
    }

    /// An override with no default and no header line reads as a constant of its type (a size at 64): the kernel round trips
    /// through a typed gene with the constant in a `C_T` pool, which evolution may change.
    #[test]
    fn an_override_with_no_default_is_a_constant_a_gene_holds() {
        let kingdom = WgslKingdom::load_safe();
        let text = "override workgroup_width: u32;\noverride scale: f32;\n@group(0) @binding(0) var<storage, read_write> dst: array<f32>;\n@compute @workgroup_size(workgroup_width)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    dst[gid.x] = f32(gid.x) * scale;\n}\n";
        let read = crate::wgsl::safe_reader::read_safe(text, &kingdom).unwrap_or_else(|e| panic!("{e}"));
        let rendered = crate::wgsl::safe_karva::render_tree_with(&read.tree, &kingdom, &read.role).expect("renders").text;
        assert!(rendered.contains("@workgroup_size(64"), "{rendered}");
        match round_trip(&mut Alphabets::default(), &kingdom, text) {
            Outcome::Expressed { tree_equal: true, text_equal: true, valid: Ok(()), .. } => {}
            other => panic!("{other:?}"),
        }
    }

    /// The quad gathers (a quad is four consecutive lanes) read, encode into a typed gene, express and render to the same text.
    #[test]
    fn the_quad_gathers_round_trip_through_a_typed_gene() {
        let kingdom = WgslKingdom::load_safe();
        let text = "enable subgroups;\n@group(0) @binding(0) var<storage, read> src: array<f32>;\n@group(0) @binding(1) var<storage, read_write> dst: array<f32>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let v = src[gid.x];\n    dst[gid.x] = quadBroadcast(v, 2u) + quadSwapX(v) + quadSwapY(v) + quadSwapDiagonal(v);\n}\n";
        match round_trip(&mut Alphabets::default(), &kingdom, text) {
            Outcome::Expressed { tree_equal: true, text_equal: true, valid: Ok(()), .. } => {}
            other => panic!("{other:?}"),
        }
    }

    /// An array declared by the module is a value: a uniform `array<mat2x2<f32>, 4>` passed whole to a helper, indexed and read
    /// by member reads, round trips through a typed gene over the module's extended kingdom (the dual `array<..., 4>.value`,
    /// its rows `agg.at`, `agg.zero`).
    #[test]
    fn an_array_value_passed_to_a_helper_round_trips_through_a_typed_gene() {
        let kingdom = WgslKingdom::load_safe();
        let text = "@group(0) @binding(0) var<uniform> u : array<mat2x2<f32>, 4>;\n@group(0) @binding(1) var<storage, read_write> s: f32;\nfn a(a : array<mat2x2<f32>, 4>) -> f32 { return a[0][0].x; }\nfn b(m : mat2x2<f32>) -> f32 { return m[0].x; }\n@compute @workgroup_size(1)\nfn f() { s = a(u) + b(u[1]); }\n";
        let read = crate::wgsl::safe_reader::read_safe(text, &kingdom).unwrap_or_else(|e| panic!("{e}"));
        let ext = read.kingdom.as_ref().expect("the module declares an aggregate type");
        assert!(ext.duals.iter().any(|d| d.dual == "array<mat2x2<f32>, 4>.value"), "{:?}", ext.duals.iter().skip(204).map(|d| &d.dual).collect::<Vec<_>>());
        assert!(ext.rows.iter().any(|r| r.name == "agg.at") && ext.rows.iter().any(|r| r.name == "agg.zero"));
        match round_trip(&mut Alphabets::default(), &kingdom, text) {
            Outcome::Expressed { tree_equal: true, text_equal: true, valid: Ok(()), .. } => {}
            other => panic!("{other:?}"),
        }
    }

    /// A matrix of f16 is a value the shipped duals have no matrix of: the module's extended kingdom holds `mat2x2<f16>.real`
    /// with its columns as rows (`shape.col_0`, `shape.col_1`), and a uniform of it passed to a helper round trips.
    #[test]
    fn an_f16_matrix_is_a_dual_of_the_modules_extended_kingdom() {
        let kingdom = WgslKingdom::load_safe();
        let text = "enable f16;\n@group(0) @binding(0) var<uniform> u : mat2x2<f16>;\nfn a(m : mat2x2<f16>) {}\nfn b(v : vec2<f16>) {}\nfn c(f : f16) {}\n@compute @workgroup_size(1)\nfn f() { a(u); b(u[1]); b(u[1].yx); c(u[1].x); c(u[1].yx.x); }\n";
        let read = crate::wgsl::safe_reader::read_safe(text, &kingdom).unwrap_or_else(|e| panic!("{e}"));
        let ext = read.kingdom.as_ref().expect("the module declares a matrix of f16");
        assert!(ext.duals.iter().any(|d| d.dual == "mat2x2<f16>.real" && d.lanes == 4), "{:?}", ext.duals.iter().skip(204).map(|d| &d.dual).collect::<Vec<_>>());
        assert!(ext.rows.iter().any(|r| r.name == "shape.col_1" && r.template == "A-col"));
        match round_trip(&mut Alphabets::default(), &kingdom, text) {
            Outcome::Expressed { tree_equal: true, text_equal: true, valid: Ok(()), .. } => {}
            other => panic!("{other:?}"),
        }
    }

    /// A private array of structs is an aggregate register: written whole (`p = u`), at an element (`p[1] = u[2]`), at a member
    /// (`p[3].m = ..`) and at a column inside both (`p[1].m[0] = ..`), each a chain of `agg.with_*` rows over the register; read
    /// back at any depth. The kernel round trips through a typed gene and the text it renders is the kernel naga accepts.
    #[test]
    fn a_private_array_of_structs_is_an_aggregate_register() {
        let kingdom = WgslKingdom::load_safe();
        let text = "struct S { before : i32, m : mat2x2<f32>, after : i32, }\n@group(0) @binding(0) var<uniform> u : array<S, 4>;\n@group(0) @binding(1) var<storage, read_write> s: f32;\nvar<private> p : array<S, 4>;\n@compute @workgroup_size(1)\nfn f() { p = u; p[1] = u[2]; p[3].m = u[2].m; p[1].m[0] = u[0].m[1].yx; s = p[1].m[0].x + f32(p[2].before); }\n";
        let read = crate::wgsl::safe_reader::read_safe(text, &kingdom).unwrap_or_else(|e| panic!("{e}"));
        let ext = read.kingdom.as_ref().expect("the module declares aggregate types");
        assert!(ext.rows.iter().any(|r| r.name == "agg.with_at") && ext.rows.iter().any(|r| r.name == "agg.with_m" && r.inputs.len() == 2), "the with rows");
        assert!(ext.rows.iter().any(|r| r.name.starts_with("reg.set_") && r.name.len() > 8 && r.name[8..].starts_with(|c: char| c.is_ascii_uppercase())) || ext.rows.iter().any(|r| r.name.starts_with("reg.set_A_") || r.name.starts_with("reg.set_B_")), "a register kind of the array");
        let rendered = crate::wgsl::safe_karva::render_tree_with(&read.tree, ext, &read.role).expect("renders").text;
        assert!(rendered.contains("with_array_S__4__at(") && rendered.contains("with_S_m("), "{rendered}");
        match round_trip(&mut Alphabets::default(), &kingdom, text) {
            Outcome::Expressed { tree_equal: true, text_equal: true, valid: Ok(()), .. } => {}
            other => panic!("{other:?}"),
        }
    }

    /// A `var<private>` with a constant initializer of a vector, a matrix of f16, an array or a struct: its value built from
    /// literals (a vector composed, a matrix of f16 and an aggregate a zero with its nonzero parts written) and stored at the
    /// start; the kernel round trips through a typed gene.
    #[test]
    fn a_private_variable_with_a_constant_initializer_round_trips() {
        let kingdom = WgslKingdom::load_safe();
        let text = "enable f16;\nstruct P { a : f32, b : vec2<f32>, }\nvar<private> v : vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);\nvar<private> h : mat2x2<f16> = mat2x2<f16>(1.0h, 2.0h, 3.0h, 4.0h);\nvar<private> a : array<i32, 3> = array<i32, 3>(1, 0, 2);\nvar<private> p : P = P(5.0, vec2<f32>(6.0, 7.0));\n@group(0) @binding(0) var<storage, read_write> s: f32;\n@compute @workgroup_size(1)\nfn f() { s = v.y + f32(h[1][0]) + f32(a[2]) + p.b.y + p.a; }\n";
        let read = crate::wgsl::safe_reader::read_safe(text, &kingdom).unwrap_or_else(|e| panic!("{e}"));
        let ext = read.kingdom.as_ref().expect("the module declares aggregate types");
        let rendered = crate::wgsl::safe_karva::render_tree_with(&read.tree, ext, &read.role).expect("renders").text;
        assert!(rendered.contains("vec3<f32>(") && rendered.contains("with_P_a(") && rendered.contains("with_mat2x2_f16__at("), "{rendered}");
        match round_trip(&mut Alphabets::default(), &kingdom, text) {
            Outcome::Expressed { tree_equal: true, text_equal: true, valid: Ok(()), .. } => {}
            other => panic!("{other:?}"),
        }
    }
    /// EVERY `load.cfg_*` ROW OF AN INTERFACE RENDERS IN A KERNEL AND NAGA VALIDATES IT (run 6i: 292 of the 558 illegal
    /// children of the v24 run were `kernel_does_not_render` / `kernel_does_not_parse` naming `load.cfg_m`, `load.cfg_d`,
    /// `load.cfg_r_rows` and `cfg.d`). The rows read a field of the interface's uniform: the Gram's and the cosine's
    /// interfaces have the four fields (`cfg.m`, `p.nc`, ...) and each row renders to its own field's text; v24's has
    /// none, and no row that reads one is a symbol of its alphabet.
    #[test]
    fn every_cfg_row_of_an_interface_renders_in_a_kernel_and_validates() {
        use crate::wgsl::render::{cosine_role, gram_role, validate_text};
        use crate::wgsl::safe_karva::render_tree_with;
        let kingdom = WgslKingdom::load_safe();
        let ty = |n: &str| crate::wgsl::render::dual_ty(&kingdom, n).unwrap_or_else(|e| panic!("{e}"));
        for (role, fields) in [(gram_role(), ["cfg.d", "cfg.m", "cfg.r_rows", "cfg.r_off"]), (cosine_role(), ["p.nc", "p.dim", "p.nq", "p.q_off"])] {
            let alphabet = Alphabet::new(&kingdom, &role).unwrap_or_else(|g| panic!("{}: {}: {}", role.name, g.reason, g.detail));
            let mut checked = 0;
            for (row, field) in ["load.cfg_d", "load.cfg_m", "load.cfg_r_rows", "load.cfg_r_off"].into_iter().zip(fields) {
                let p = role.per_kernel.iter().find(|p| p.name == row).unwrap_or_else(|| panic!("{}: no per-kernel row {row}", role.name));
                assert_eq!(p.template, field, "{} {row}", role.name);
                assert!((0..alphabet.contract.n_symbols()).any(|s| alphabet.contract.name(s as SymbolId) == format!("kernel:{row}")), "{} {row} is not a symbol of its alphabet", role.name);
                let out = ty(&p.output);
                let to_real = if p.output == "u32.count" { "convert.count_to_f32" } else { "convert.index_to_f32" };
                let value = Tree::new(to_real, ty("f32.real"), vec![Tree::new(row, out, vec![])]);
                let cell = crate::wgsl::render::own_leaf(&kingdom, ty("u32.cell")).unwrap_or_else(|e| panic!("{e}"));
                let store = Tree::new("block.one_O0", ty("block.O0"), vec![Tree::new("store.h_O0", ty("stmt.O0"), vec![cell, value])]);
                let kernel = Tree::new("kernel.entry_safe", ty("kernel.safe"), vec![crate::wgsl::render::own_leaf(&kingdom, ty("grid.grid")).unwrap_or_else(|e| panic!("{e}")), crate::wgsl::render::own_leaf(&kingdom, ty("block.L0")).unwrap_or_else(|e| panic!("{e}")), store]);
                let text = render_tree_with(&kernel, &kingdom, &role).unwrap_or_else(|e| panic!("{} {row} does not render: {e}", role.name)).text;
                assert!(text.contains(field), "{} {row}: the kernel does not read {field}:\n{text}", role.name);
                validate_text(&text).unwrap_or_else(|e| panic!("{} {row}: naga refuses the kernel: {e}\n{text}", role.name));
                checked += 1;
            }
            assert_eq!(checked, 4, "{}", role.name);
        }
    }

    /// v24's INTERFACE HAS NONE OF THE GRAM'S FIELDS, INPUT OR OUTPUT CELL: the rows that spell them are no symbols of its
    /// alphabet, so no gene drawn from it names `cfg.m`, `cfg.d` or the Gram's cell.
    #[test]
    fn an_interface_without_the_grams_fields_has_none_of_the_rows_that_read_them() {
        let kingdom = WgslKingdom::load_safe();
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/problems/v24_ingest_small/reference.wgsl")).expect("v24's reference kernel");
        let read = crate::wgsl::safe_reader::read_safe(&text, &kingdom).unwrap_or_else(|e| panic!("{e}"));
        let kingdom = read.kingdom.as_ref().unwrap_or(&kingdom);
        assert!(read.role.field("m").is_err() && read.role.field("d").is_err(), "v24's uniform has no m or d field");
        let alphabet = Alphabet::new(kingdom, &read.role).unwrap_or_else(|g| panic!("{}: {}", g.reason, g.detail));
        let names: Vec<String> = (0..alphabet.contract.n_symbols()).map(|s| alphabet.contract.name(s as SymbolId).to_string()).collect();
        let gram: Vec<&String> = names.iter().filter(|n| ["uniform.cfg_", "kernel:load.cfg_m", "kernel:load.cfg_d", "kernel:load.cfg_r_rows", "kernel:load.cfg_r_off", "bound.m", "bound.d", "bound.rows", "safe.cell_of", "safe.mirror_of", "kernel:load.phi", "kernel:load.h"].iter().any(|p| n.starts_with(p))).collect();
        assert!(gram.is_empty(), "rows that read the Gram's fields are symbols of v24's alphabet: {gram:?}");
    }

    /// N GENES FROM THE TYPED INITIALISER OF v24's INTERFACE, rendered and validated by naga: none fails on a `cfg` row
    /// (run 6i: `load.cfg_m` is not a row of the function table, `invalid field accessor 'd'`). The other render
    /// failures the genes show (printed: a vector of f16 indexed by `shape.at`, `store.index_*` named as a row, naga's
    /// folding of `acos`, `asin` and `pow` of constants, a fifth nested tile loop) are kingdom bugs of their own, listed in
    /// the A0 report for the rows' owners; the legality layers are not judged here.
    #[test]
    fn random_typed_genes_of_v24s_interface_name_no_cfg_row() {
        use crate::wgsl::render::validate_text;
        use crate::wgsl::safe_karva::render_tree_with;
        const N: usize = 200;
        let kingdom = WgslKingdom::load_safe();
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/problems/v24_ingest_small/reference.wgsl")).expect("v24's reference kernel");
        let read = crate::wgsl::safe_reader::read_safe(&text, &kingdom).unwrap_or_else(|e| panic!("{e}"));
        let kingdom = read.kingdom.as_ref().unwrap_or(&kingdom);
        let a = Alphabet::new(kingdom, &read.role).unwrap_or_else(|g| panic!("{}: {}", g.reason, g.detail));
        let root = crate::wgsl::render::dual_ty(kingdom, read.role.root_dual()).unwrap_or_else(|e| panic!("{e}"));
        let layout = GeneLayout::from_program_size(&a.contract, root, 16).unwrap_or_else(|e| panic!("{e}"));
        let mut state = 0xA11CE_u64;
        let mut failures: BTreeMap<String, usize> = BTreeMap::new();
        for i in 0..N {
            let gene = random_gene(&a, kingdom, &layout, &mut state).unwrap_or_else(|g| panic!("gene {i}: {}: {}", g.reason, g.detail));
            let tree = express(&a, kingdom, &gene).unwrap_or_else(|g| panic!("gene {i}: {}: {}", g.reason, g.detail));
            match render_tree_with(&tree, kingdom, &read.role).and_then(|r| validate_text(&r.text)) {
                Ok(()) => {}
                Err(e) => *failures.entry(e.lines().next().unwrap_or("").chars().take(140).collect()).or_default() += 1,
            }
        }
        println!("{N} random typed genes of v24's interface: {} failed to render or validate: {failures:?}", failures.values().sum::<usize>());
        let cfg: Vec<&String> = failures.keys().filter(|e| e.contains("cfg") || e.contains("field accessor") || e.contains("is not a row of the function table")).collect();
        assert!(cfg.is_empty(), "genes of v24's interface fail on the Gram's fields: {cfg:?}");
    }

    /// A kernel whose body sets register b0 to `flag`, rendered and validated by naga.
    fn flag_kernel_validates(flag: &Tree, what: &str) {
        use crate::wgsl::render::{dual_ty, gram_role, own_leaf, validate_text};
        let k = WgslKingdom::load_safe();
        let role = gram_role();
        let ty = |n: &str| dual_ty(&k, n).unwrap_or_else(|e| panic!("{e}"));
        let set = Tree::new("reg.set_b_L0", ty("stmt.L0"), vec![Tree::new("reg.b0", ty("reg<bool>.flag"), vec![]), flag.clone()]);
        let kernel = Tree::new("kernel.entry_safe", ty("kernel.safe"), vec![own_leaf(&k, ty("grid.grid")).unwrap(), Tree::new("block.one_L0", ty("block.L0"), vec![set]), own_leaf(&k, ty("block.O0")).unwrap()]);
        let text = crate::wgsl::safe_karva::render_tree_with(&kernel, &k, &role).unwrap_or_else(|e| panic!("{what}: {e}")).text;
        validate_text(&text).unwrap_or_else(|e| panic!("{what}: naga refuses the kernel: {e}\n{text}"));
    }

    /// CAUSE 1 OF THE NAGA ERRORS OF THE TYPED GENES: `logic.is_inf` compared an f16 operand with the f32 maximum
    /// (`abs(x) > 3.4028235e38`: "the concrete type f16 cannot represent the abstract value", and a vec3<f16> against a
    /// vec3<f32>). It is written with the operand's own width.
    #[test]
    fn is_inf_of_an_f16_operand_is_spelled_in_f16() {
        let k = WgslKingdom::load_safe();
        let ty = |n: &str| crate::wgsl::render::dual_ty(&k, n).unwrap_or_else(|e| panic!("{e}"));
        let scalar = Tree::new("logic.is_inf", ty("bool.flag"), vec![Tree::literal("literal.real", ty("f16.real"), 1.5)]);
        flag_kernel_validates(&scalar, "is_inf of an f16 scalar");
        let lanes = Tree::new("logic.is_inf", ty("vec3<bool>.flag"), vec![Tree::literal("literal.real", ty("vec3<f16>.real"), 1.5)]);
        flag_kernel_validates(&Tree::new("logic.any", ty("bool.flag"), vec![lanes]), "is_inf of a vec3<f16>");
    }

    /// CAUSE 2: an f16 literal in a literal-only expression was written bare (`asin(vec4<f16>(f16(-3.88f)))`), which naga
    /// folds and refuses ("asin built-in function argument is out of valid range"); an f32 one passes through `lit_f32`.
    /// An f16 literal passes through `lit_f16` the same way.
    #[test]
    fn a_literal_only_expression_over_f16_is_not_folded_by_naga() {
        let k = WgslKingdom::load_safe();
        let ty = |n: &str| crate::wgsl::render::dual_ty(&k, n).unwrap_or_else(|e| panic!("{e}"));
        let asin = Tree::new("trig.asin", ty("f16.real"), vec![Tree::literal("literal.real", ty("f16.real"), -3.5)]);
        flag_kernel_validates(&Tree::new("logic.is_nan", ty("bool.flag"), vec![asin]), "asin of an f16 constant");
        let lanes = Tree::new("trig.asin", ty("vec4<f16>.real"), vec![Tree::literal("literal.real", ty("vec4<f16>.real"), -3.5)]);
        let nan = Tree::new("logic.is_nan", ty("vec4<bool>.flag"), vec![lanes]);
        flag_kernel_validates(&Tree::new("logic.any", ty("bool.flag"), vec![nan]), "asin of a vec4<f16> constant");
    }

    /// AN INTERFACE THAT DEMANDS THE OUTPUT STORE HAS NO `return` SYMBOL (T66: 76 of 1,000 random typed genes under the
    /// Gram's interface returned before the epilogue and never wrote h); v24's, with no output cell, keeps it.
    #[test]
    fn no_return_can_skip_the_epilogue_store_the_problem_demands() {
        let kingdom = WgslKingdom::load_safe();
        let returns = |role: &Role| -> usize {
            let a = Alphabet::new(&kingdom, role).unwrap_or_else(|g| panic!("{}: {}", g.reason, g.detail));
            (0..a.contract.n_symbols()).filter(|s| a.contract.name(*s as SymbolId).starts_with("stmt.return_")).count()
        };
        assert_eq!(returns(&crate::wgsl::render::gram_role()), 0);
        assert_eq!(returns(&crate::wgsl::render::cosine_role()), 0);
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/problems/v24_ingest_small/reference.wgsl")).expect("v24's reference kernel");
        let read = crate::wgsl::safe_reader::read_safe(&text, &kingdom).unwrap_or_else(|e| panic!("{e}"));
        let ext = read.kingdom.as_ref().unwrap_or(&kingdom);
        let a = Alphabet::new(ext, &read.role).unwrap_or_else(|g| panic!("{}: {}", g.reason, g.detail));
        assert!((0..a.contract.n_symbols()).any(|s| a.contract.name(s as SymbolId).starts_with("stmt.return_")), "v24's interface keeps its return rows");
    }
}
