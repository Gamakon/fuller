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
//! A `fallback` leaf of a read tree is the `own.*` terminal of its dual.
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
    /// The `own.*` terminal of each dual.
    own: BTreeMap<Ty, SymbolId>,
}

impl Alphabet {
    pub fn new(kingdom: &WgslKingdom, role: &Role) -> Result<Alphabet, Gap> {
        let mut rows: Vec<(String, Vec<Ty>, Ty)> = kingdom.rows.iter().map(|r| (r.name.clone(), r.inputs.clone(), r.output)).collect();
        let mut per_name: BTreeMap<&str, usize> = BTreeMap::new();
        for r in &kingdom.rows {
            *per_name.entry(r.name.as_str()).or_default() += 1;
        }
        let mut ids: Vec<(String, Vec<Ty>, Ty)> = kingdom.rows.iter().map(|r| (r.semantic_id(&kingdom.duals, per_name[r.name.as_str()]), r.inputs.clone(), r.output)).collect();
        for (name, ins, out) in crate::wgsl::safe_karva::per_kernel_rows(kingdom, role) {
            rows.push((name.clone(), ins.clone(), out));
            ids.push((format!("kernel:{name}"), ins, out));
        }
        let constants: Vec<&str> = ids.iter().zip(&rows).filter(|((_, ins, _), (n, _, _))| ins.is_empty() && (n.starts_with("literal.") || n.starts_with("const."))).map(|((id, _, _), _)| id.as_str()).collect();
        let roots: Vec<Ty> = kingdom.duals.iter().filter(|d| d.dual == "kernel.safe" || d.dual == "kernel.module").map(WgslDual::ty).collect();
        let contract = relations_from_rows("safeWGSL", roots, &ids, &constants).check().map_err(|d| gap("kingdom check", d.join("; ")))?;
        let mut by_key = BTreeMap::new();
        let mut own = BTreeMap::new();
        for (i, (name, ins, out)) in rows.iter().enumerate() {
            by_key.entry((name.clone(), *out, ins.clone())).or_insert(i as SymbolId);
            if name.starts_with("own.") {
                own.entry(*out).or_insert(i as SymbolId);
            }
        }
        Ok(Alphabet { contract, rows, by_key, own })
    }

    pub fn row(&self, s: SymbolId) -> &(String, Vec<Ty>, Ty) {
        &self.rows[s as usize]
    }
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

/// A read tree with its `fallback` leaves as the `own.*` terminal of their dual.
fn typed_of(a: &Alphabet, kingdom: &WgslKingdom, t: &Tree, pools: &mut BTreeMap<Ty, Vec<u32>>) -> Result<Typed, Gap> {
    let dual_name = |ty: Ty| kingdom.dual(ty).map_or_else(|| format!("{ty:?}"), |d| d.dual.clone());
    if t.name == "fallback" {
        let s = *a.own.get(&t.ty).ok_or_else(|| gap("fallback leaf of a dual with no own terminal", dual_name(t.ty)))?;
        return Ok(Typed { symbol: s, const_index: None, kids: Vec::new() });
    }
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

/// A read tree with its `fallback` leaves as `own.*` (what [`express`] returns for them).
pub fn normalised(a: &Alphabet, t: &Tree) -> Tree {
    if t.name == "fallback" {
        if let Some(s) = a.own.get(&t.ty) {
            return Tree::new(&a.row(*s).0, t.ty, Vec::new());
        }
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| normalised(a, k)).collect() }
}

/// The alphabets of the interfaces met, by their per-kernel rows.
#[derive(Default)]
pub struct Alphabets {
    by_interface: BTreeMap<String, Result<Alphabet, Gap>>,
}

impl Alphabets {
    pub fn of(&mut self, kingdom: &WgslKingdom, role: &Role) -> &Result<Alphabet, Gap> {
        let key = format!("{}|{}", role.name, role.per_kernel.iter().map(|p| format!("{}:{}:{}", p.name, p.inputs.join("+"), p.output)).collect::<Vec<_>>().join(","));
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
    /// Encoded and expressed: the tree equal to the read one (fallbacks as their own terminals), the text equal to
    /// the read tree's, and naga's verdict on that text.
    Expressed { tree_equal: bool, text_equal: bool, valid: Result<(), String>, difference: String },
}

/// WGSL text, read, encoded, expressed, rendered (design test 5).
pub fn round_trip(alphabets: &mut Alphabets, kingdom: &WgslKingdom, text: &str) -> Outcome {
    use crate::wgsl::render::validate_text;
    use crate::wgsl::safe_karva::render_tree_with;
    let read = match crate::wgsl::safe_reader::read_safe(text, kingdom) {
        Ok(r) => r,
        Err(e) => return Outcome::NotRead(e),
    };
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
    let want = normalised(a, &read.tree);
    let difference = first_difference(&want, &back);
    match render_tree_with(&back, kingdom, &read.role) {
        Err(e) => Outcome::Gap(gap("render of the expressed tree", e)),
        Ok(r) => Outcome::Expressed { tree_equal: back == want, text_equal: r.text == original, valid: validate_text(&r.text), difference },
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
        for (dir, limit) in [("tests/corpus/safewgsl", usize::MAX), ("tests/corpus/nluca", usize::MAX), ("tests/corpus/interfaces", usize::MAX), ("tests/corpus/model_replies", 300)] {
            let mut files: Vec<_> = std::fs::read_dir(root.join(dir)).expect("the corpus").flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
            files.sort();
            for f in files.into_iter().take(limit) {
                total += 1;
                match round_trip(&mut alphabets, &kingdom, &std::fs::read_to_string(&f).expect("a kernel")) {
                    Outcome::NotRead(_) | Outcome::Unrenderable(_) => {}
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
        println!("{read} of {total} kernels read; {} do not round trip", failures.len());
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
    /// to the same text, naga-valid. Skipped when the (untracked) logs are absent.
    #[test]
    fn the_saved_genomes_round_trip_through_typed_genes() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/HGP/001");
        let kingdom = WgslKingdom::load_safe();
        let mut alphabets = Alphabets::default();
        let (mut total, mut failures) = (0usize, Vec::new());
        for (dumps, texts) in [("popA2_g006j/karva", "popA2_g006j/texts"), ("library_ep", "library_ep_texts"), ("safe96/karva", "gen0")] {
            let Ok(entries) = std::fs::read_dir(root.join(dumps)) else { return };
            let mut files: Vec<_> = entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
            files.sort();
            for f in files {
                let id = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                let text = std::fs::read_to_string(root.join(texts).join(format!("{id}.wgsl"))).unwrap_or_else(|e| panic!("{dumps}/{id}: its text: {e}"));
                total += 1;
                match round_trip(&mut alphabets, &kingdom, &text) {
                    Outcome::Expressed { tree_equal: true, text_equal: true, valid: Ok(()), .. } => {}
                    other => failures.push(format!("{dumps}/{id}: {other:?}")),
                }
            }
        }
        println!("saved genomes: {} of {total} round trip through a typed gene", total - failures.len());
        assert!(failures.is_empty(), "{} of {total}:\n{}", failures.len(), failures.iter().take(10).cloned().collect::<Vec<_>>().join("\n"));
    }
}
