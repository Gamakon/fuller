//! THE RINGS AND THE CANVAS (`docs/PLAN_futhark_fitness_fixes.md` section H, Andrew's method of 2026-10-28): a gene of
//! the signature kingdom padded with real sequences instead of random symbols.
//!
//! * **The pool**: programs read into the kingdom by [`crate::futhark_reader`], each tagged with its SOURCE (a
//!   published benchmark's function, or a program a language model wrote) and its ORIGIN (the file and function, or
//!   the reply). Stored as `pool.tsv` lines: the tree as the builder's s-expression at its type.
//! * **The rings**: each program's coding symbols, in the order Karva lays them out (the nodes of a type in level
//!   order are that type's frame rows 0, 1, 2, ...), split by type: every symbol to its type's HEAD ring, every
//!   terminal also to its type's TERMINAL ring, every constant's pool index to its type's CONSTANT ring. Programs are
//!   added one by one, in the pool's order.
//! * **The canvas**: a new gene is, for every frame of the layout, a window of its type's rings, each from its own
//!   random start: the head from the head ring, the tail from the terminal ring, the Dc from the constant ring. The
//!   root's window starts at a function (as a random gene's root cell is a function: a terminal root is a constant
//!   program).
//! * **Types the programs never reach**: a ring left empty is filled from the rings of the same CLASS (the basic
//!   shape, `a.n`, `i32`, `ix.k`) in the other scopes, each symbol written as the row of the same operation in the
//!   empty type when the kingdom has that row (a symbol with no such row is left out). A class no program reached at
//!   all is a gap of the kingdom; [`Rings::complete`] reports it, and fills it from the same slot family (`arr`,
//!   `ix`, `i32`) by the same rule, named as such in the report. A ring with nothing at all to fill it from is an
//!   error: it is never filled at random.
//!
//! Leave one out: a benchmark's discovery runs must not see its own published program, so [`published_source`]
//! names the file every canvas of that benchmark's R and C runs excludes.

use std::collections::BTreeMap;
use std::path::Path;

use crate::futhark_kingdom::{FutharkKind, FutharkKingdom, GeneRng};
use crate::geneframe::Ty;
use crate::kingdom_contract::KingdomContract;
use crate::typed_gep::{GeneLayout, Tree};

/// Where a pool program came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Source {
    /// A function of a published benchmark (diku-dk/futhark-benchmarks).
    Bench,
    /// A program a language model wrote for the benchmark's signature.
    Llm,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Bench => "bench",
            Source::Llm => "llm",
        }
    }
    pub fn of(s: &str) -> Result<Source, String> {
        match s {
            "bench" => Ok(Source::Bench),
            "llm" => Ok(Source::Llm),
            o => Err(format!("a pool source {o}: bench or llm")),
        }
    }
    /// `bench`, `llm` or `both`.
    pub fn set(s: &str) -> Result<Vec<Source>, String> {
        match s {
            "both" => Ok(vec![Source::Bench, Source::Llm]),
            o => Ok(vec![Source::of(o)?]),
        }
    }
}

/// The published file of a problem's program (relative to the benchmarks' root): excluded from the rings of its
/// discovery runs (leave one out), or the answer is planted in the canvas.
pub fn published_source(problem: &str) -> Option<&'static str> {
    match problem {
        "histogram" => Some("micro/reduce_by_index.fut"),
        "pathfinder" => Some("rodinia/pathfinder/pathfinder.fut"),
        "reduce" => Some("micro/reduce.fut"),
        "scan" => Some("micro/scan.fut"),
        _ => None,
    }
}

/// One program of the pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub source: Source,
    /// The benchmark file (relative to the benchmarks' root) or the reply's name.
    pub origin: String,
    /// The function's name (`main` for a model's program).
    pub function: String,
    /// The type the program was read at.
    pub dual: String,
    pub sexp: String,
}

pub const POOL_HEADER: &str = "seq\tsource\torigin\tfunction\tdual\tsexp";

impl Entry {
    pub fn line(&self, seq: usize) -> String {
        format!("{seq}\t{}\t{}\t{}\t{}\t{}", self.source.name(), self.origin, self.function, self.dual, self.sexp)
    }
}

/// The pool's programs (`pool.tsv`), in order.
pub fn load_pool(path: &Path) -> Result<Vec<Entry>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 6 {
            return Err(format!("{}: the line {line:?} has {} fields, wants 6", path.display(), f.len()));
        }
        out.push(Entry { source: Source::of(f[1])?, origin: f[2].into(), function: f[3].into(), dual: f[4].into(), sexp: f[5].into() });
    }
    Ok(out)
}

/// A program of the pool as a tree at its type: the builder's s-expression ([`crate::futhark_sig::sexp`]) at any type
/// of the kingdom (`futhark_sig::build` builds at the root only).
pub fn build_at(k: &FutharkKingdom, text: &str, dual: &str) -> Result<Tree, String> {
    let ty = k.ty(dual).ok_or_else(|| format!("no type {dual}"))?;
    let spaced = text.replace('(', " ( ").replace(')', " ) ");
    let toks: Vec<&str> = spaced.split_whitespace().collect();
    let mut pos = 0;
    let t = build_tok(k, &toks, &mut pos, ty)?;
    if pos != toks.len() {
        return Err(format!("text after the program: {:?}", &toks[pos..]));
    }
    Ok(t)
}

fn build_tok(k: &FutharkKingdom, toks: &[&str], pos: &mut usize, ty: Ty) -> Result<Tree, String> {
    let dual = k.dual(ty).ok_or("a slot of no dual")?;
    let row = |op: &str| -> Result<u32, String> { k.rows.iter().position(|r| r.name == format!("{op}@{}", dual.dual)).map(|i| i as u32).ok_or_else(|| format!("the kingdom has no row {op}@{}", dual.dual)) };
    let t = *toks.get(*pos).ok_or("the program ends early")?;
    *pos += 1;
    if t == "(" {
        let head = *toks.get(*pos).ok_or("a list with no head")?;
        *pos += 1;
        let sym = row(head)?;
        let mut kids = Vec::new();
        for a in k.rows[sym as usize].inputs.clone() {
            kids.push(build_tok(k, toks, pos, a)?);
        }
        if toks.get(*pos) != Some(&")") {
            return Err(format!("{head}@{} given too many arguments", dual.dual));
        }
        *pos += 1;
        return Ok(Tree { symbol: sym, const_index: None, kids });
    }
    if let Ok(v) = t.parse::<i64>() {
        let i = dual.literals.iter().position(|l| l.trim_end_matches(dual.slot.as_str()).parse::<i64>() == Ok(v)).ok_or_else(|| format!("the literal {t} is not in the pool of {}", dual.dual))?;
        let sym = k.rows.iter().position(|r| r.kind == FutharkKind::Const && r.output == ty).ok_or_else(|| format!("{} has no constant", dual.dual))? as u32;
        return Ok(Tree { symbol: sym, const_index: Some(i as u32), kids: vec![] });
    }
    Ok(Tree { symbol: row(t)?, const_index: None, kids: vec![] })
}

/// Which ring of a type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ring {
    Head,
    Terminal,
    Constant,
}

impl Ring {
    pub fn name(self) -> &'static str {
        match self {
            Ring::Head => "head",
            Ring::Terminal => "terminal",
            Ring::Constant => "constant",
        }
    }
}

/// How an empty ring was filled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fill {
    pub dual: String,
    pub ring: Ring,
    /// `class` (the same basic shape in other scopes) or `slot family (class gap)` (no program reached the class).
    pub from: &'static str,
    pub len: usize,
}

/// The three rings of every type.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Rings {
    pub rings: BTreeMap<(Ty, Ring), Vec<u32>>,
    /// Programs added.
    pub programs: usize,
}

impl Rings {
    /// Add a program: its symbols by type in level order (a type's frame rows in Karva order).
    pub fn add(&mut self, contract: &KingdomContract, tree: &Tree) {
        for (t, _) in tree.level_order() {
            let ty = contract.output(t.symbol);
            self.rings.entry((ty, Ring::Head)).or_default().push(t.symbol);
            if contract.is_terminal(t.symbol) {
                self.rings.entry((ty, Ring::Terminal)).or_default().push(t.symbol);
            }
            if let Some(c) = t.const_index {
                self.rings.entry((ty, Ring::Constant)).or_default().push(c);
            }
        }
        self.programs += 1;
    }

    pub fn get(&self, ty: Ty, ring: Ring) -> &[u32] {
        self.rings.get(&(ty, ring)).map_or(&[], Vec::as_slice)
    }

    /// The rings of every frame of `layout` filled (section H.6), with how each empty one was filled. A ring a frame
    /// needs that nothing fills is an error naming every such ring.
    pub fn complete(&self, k: &FutharkKingdom, contract: &KingdomContract, layout: &GeneLayout) -> Result<(Rings, Vec<Fill>), String> {
        let mut out = self.clone();
        let mut fills = Vec::new();
        let mut missing = Vec::new();
        let dual_of = |ty: Ty| k.dual(ty).map(|d| d.dual.clone()).unwrap_or_default();
        let class_of = |ty: Ty| dual_of(ty).split(':').next().unwrap_or("").to_string();
        let slot_of = |ty: Ty| k.dual(ty).map(|d| d.slot.clone()).unwrap_or_default();
        let by_name: BTreeMap<&str, u32> = k.rows.iter().enumerate().map(|(i, r)| (r.name.as_str(), i as u32)).collect();
        for f in &layout.frames {
            let need = [(Ring::Head, f.head_len), (Ring::Terminal, f.tail_len), (Ring::Constant, f.dc_len)];
            for (ring, len) in need {
                if len == 0 || !self.get(f.ty, ring).is_empty() {
                    continue;
                }
                let target = dual_of(f.ty);
                let tclass = target.split(':').next().unwrap_or("").to_string();
                let translate = |src_ty: Ty, sym: u32| -> Option<u32> {
                    if ring == Ring::Constant {
                        let (sd, td) = (k.dual(src_ty)?, k.dual(f.ty)?);
                        let lit = sd.literals.get(sym as usize)?;
                        return td.literals.iter().position(|l| l == lit).map(|i| i as u32);
                    }
                    let name = &k.rows[sym as usize].name;
                    let op = name.split_once('@').map_or(name.as_str(), |(o, _)| o);
                    let s = *by_name.get(format!("{op}@{target}").as_str())?;
                    (contract.output(s) == f.ty && (ring == Ring::Head || contract.is_terminal(s))).then_some(s)
                };
                let gather = |same: &dyn Fn(Ty) -> bool| -> Vec<u32> {
                    let mut v = Vec::new();
                    for ((ty, r), syms) in &self.rings {
                        if *r == ring && *ty != f.ty && same(*ty) {
                            v.extend(syms.iter().filter_map(|s| translate(*ty, *s)));
                        }
                    }
                    v
                };
                let by_class = gather(&|t| class_of(t) == tclass);
                let (v, from) = if !by_class.is_empty() { (by_class, "class") } else { (gather(&|t| slot_of(t) == slot_of(f.ty)), "slot family (class gap)") };
                if v.is_empty() {
                    missing.push(format!("{target} {}", ring.name()));
                    continue;
                }
                fills.push(Fill { dual: target, ring, from, len: v.len() });
                out.rings.insert((f.ty, ring), v);
            }
        }
        if !missing.is_empty() {
            return Err(format!("a kingdom gap: {} ring(s) a gene needs have no recorded operation at all, even in their slot family: {}", missing.len(), missing.join(", ")));
        }
        Ok((out, fills))
    }

    /// A canvas gene (section H.4): every frame's head, tail and Dc a window of its type's rings, each from its own
    /// random start; the root's head window starts at a function. The rings must be [`Rings::complete`]d for the
    /// layout.
    pub fn canvas(&self, contract: &KingdomContract, layout: &GeneLayout, rng: &mut GeneRng) -> Result<Vec<u32>, String> {
        let mut gene = vec![0u32; layout.width];
        for f in &layout.frames {
            let windows = [(Ring::Head, f.head_off, f.head_len), (Ring::Terminal, f.tail_off, f.tail_len), (Ring::Constant, f.dc_off.unwrap_or(0), f.dc_len)];
            for (ring, off, len) in windows {
                if len == 0 {
                    continue;
                }
                let r = self.get(f.ty, ring);
                if r.is_empty() {
                    return Err(format!("our software has a bug: the {} ring of a frame is empty (complete the rings first)", ring.name()));
                }
                let start = if ring == Ring::Head && f.ty == layout.root {
                    let fs: Vec<usize> = (0..r.len()).filter(|&i| !contract.is_terminal(r[i])).collect();
                    if fs.is_empty() {
                        return Err("the root's head ring holds no function".into());
                    }
                    fs[rng.below(fs.len())]
                } else {
                    rng.below(r.len())
                };
                for i in 0..len {
                    gene[off + i] = r[(start + i) % r.len()];
                }
            }
        }
        Ok(gene)
    }
}

/// What a [`Rings::walk`] did: the type and domain it moved, and by how many positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Walk {
    pub ty: Ty,
    pub domain: crate::typed_gep::Domain,
    pub k: usize,
}

impl Rings {
    /// RING WALK (section I.1): one type's head or tail window advanced by `k` positions along its ring. The domain
    /// is drawn uniformly among the heads and tails that hold a coding cell (`coding`, the cells the walk of the gene
    /// reads), `k` uniformly in `1 ..= min(k_max, length)`. The window's cells shift `k` towards its start, and the
    /// `k` cells freed at its end take the ring's symbols that follow its last cell (after an occurrence of that symbol
    /// in the ring, drawn uniformly; a symbol the ring does not hold continues from a uniform start). A head takes
    /// from its type's head ring, a tail from its terminal ring, so every cell stays a legal row of its own type and a
    /// tail cell a terminal. The rings must be [`Rings::complete`]d for the layout.
    pub fn walk(&self, layout: &GeneLayout, gene: &mut [u32], coding: &[usize], k_max: usize, rng: &mut GeneRng) -> Result<Option<Walk>, String> {
        use crate::typed_gep::Domain;
        let spans: Vec<crate::typed_gep::Span> = layout.spans().into_iter().filter(|s| s.domain != Domain::Dc && coding.iter().any(|&p| p >= s.off && p < s.off + s.len)).collect();
        if spans.is_empty() {
            return Ok(None);
        }
        let s = spans[rng.below(spans.len())];
        let ring = self.get(s.ty, if s.domain == Domain::Head { Ring::Head } else { Ring::Terminal });
        if ring.is_empty() {
            return Err("our software has a bug: a ring walk on an empty ring (complete the rings first)".into());
        }
        let k = 1 + rng.below(k_max.max(1).min(s.len));
        let cells = &mut gene[s.off..s.off + s.len];
        let last = cells[s.len - 1];
        let at: Vec<usize> = (0..ring.len()).filter(|&i| ring[i] == last).collect();
        let from = if at.is_empty() { rng.below(ring.len()) } else { at[rng.below(at.len())] + 1 };
        cells.rotate_left(k);
        for j in 0..k {
            cells[s.len - k + j] = ring[(from + j) % ring.len()];
        }
        Ok(Some(Walk { ty: s.ty, domain: s.domain, k }))
    }
}

/// A GROWTH MUTATION (section H's measurement): one coding head cell set to a symbol of its alphabet with more
/// arguments than it has (a terminal turned function, or a function with more arguments), the cell and the symbol
/// each drawn uniformly. It depends on the expressed program only, never on the padding, so the same draws on two
/// paddings of one program make the same edit. `None` when no coding head cell can grow.
pub fn grow(contract: &KingdomContract, layout: &GeneLayout, gene: &mut [u32], coding: &[usize], const_len: &BTreeMap<Ty, u32>, rng: &mut GeneRng) -> Option<(usize, u32, u32)> {
    let mut options: Vec<(usize, Vec<u32>)> = Vec::new();
    let mut heads: Vec<usize> = coding.iter().copied().filter(|&p| layout.key(p).domain == crate::typed_gep::Domain::Head).collect();
    heads.sort_unstable();
    heads.dedup();
    for p in heads {
        let n = contract.signature(gene[p]).len();
        let bigger: Vec<u32> = layout.alphabet(contract, p, const_len).into_iter().filter(|s| contract.signature(*s).len() > n).collect();
        if !bigger.is_empty() {
            options.push((p, bigger));
        }
    }
    if options.is_empty() {
        return None;
    }
    let (p, bigger) = &options[rng.below(options.len())];
    let new = bigger[rng.below(bigger.len())];
    let old = std::mem::replace(&mut gene[*p], new);
    Some((*p, old, new))
}

/// The cells the walk of a gene reads: every expressed node's symbol cell and, for an expressed constant, its Dc
/// cell (as `futhark_evolve`'s mutation picks them).
pub fn coding_cells(contract: &KingdomContract, layout: &GeneLayout, gene: &[u32], const_len: &BTreeMap<Ty, u32>) -> Result<Vec<usize>, String> {
    let e = crate::typed_gep::express(contract, layout, gene, const_len).map_err(|e| format!("a gene does not express: {e}"))?;
    let mut q: BTreeMap<Ty, usize> = BTreeMap::new();
    let mut out = Vec::with_capacity(e.nodes.len() * 2);
    for n in &e.nodes {
        out.push(n.pos);
        if n.const_index.is_some() {
            let qq = q.entry(n.ty).or_default();
            let frame = layout.frame(n.ty).ok_or("a node of a type with no frame")?;
            out.push(frame.dc_off.ok_or("a constant with no Dc")? + *qq);
            *qq += 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
