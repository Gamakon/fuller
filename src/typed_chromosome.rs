//! Multigenic chromosomes, typed homeotic genes, ADFs and multicellular systems,
//! host side (`nucleotable/docs/MultiTypedGEP.md`, revision 4, section 6).
//!
//! The **epoch alphabet** is the kingdom's symbols plus two kinds of typed
//! reference terminal, each a terminal with no signature: a **genic terminal**
//! `G_g` for every normal gene id `g` of the layout, outputting that gene's root
//! type (the root belongs to the layout, never to a record), and an **ADF
//! reference** `A_d` for every ADF `d` of the epoch, outputting its declared
//! root type. [`Epoch`] is that alphabet, a checked [`KingdomContract`].
//!
//! A chromosome's normal genes are combined by one or more **homeotic genes**
//! (Ferreira's evolved linkers): typed genes with the layout of
//! [`crate::typed_gep`] whose root is the main program's phenotype type and whose
//! alphabet is the whole epoch alphabet; only the genic terminals of the genes it
//! links are valid in it, and none is valid in a normal gene. A **fixed linker**
//! is a homeotic gene marked immutable. Several homeotic genes are a
//! multicellular system: each is evaluated independently and their results keep
//! the order of the mains.
//!
//! An **ADF** is a population-wide shared definition: immutable, stored as its
//! expressed tree in level order (`adf_node` rows), with a stable id, a declared
//! root type and a level. A body holds kingdom symbols and ADF references of
//! strictly lower level only; a genic terminal in one is a defect. Every
//! violation is a [`ChromosomeDefect`], "our software has a bug: ...", never a
//! fallback.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::geneframe::Ty;
use crate::kingdom_contract::{relations_from_rows, KingdomContract, SymbolId};
use crate::typed_gep::{express, write_back, Domain, Expressed, GeneLayout, Layout, LayoutDefect};
pub use crate::typed_gep::Tree;

/// An ADF's stable id; never a position, never reused.
pub type AdfId = u64;

/// What a reference terminal names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefKind {
    /// The expressed tree of normal gene `gene_id` of the chromosome.
    Genic { gene_id: u64 },
    Adf { adf_id: AdfId },
}

/// An ADF as the epoch declares it: the reference terminal's type and the level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdfDecl {
    pub id: AdfId,
    pub root: Ty,
    pub level: u32,
}

/// Where a reference was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Gene(u64),
    Main(u64),
    Adf(AdfId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChromosomeDefect {
    /// The epoch alphabet failed the kingdom check.
    Kingdom(String),
    /// Two declarations of one ADF id.
    AdfDuplicate { id: AdfId },
    /// A body for an ADF the epoch did not declare, or two bodies for one.
    AdfUndeclared { id: AdfId },
    /// A declared ADF with no body.
    AdfMissingBody { id: AdfId },
    /// The body's level or root type is not the declaration's.
    AdfDeclaration { id: AdfId },
    /// A body with no nodes, or whose first node is not the root.
    AdfRoot { id: AdfId },
    /// Node `seq` has a parent that is not an earlier node, or an argument index its parent does not take.
    AdfNodeParent { id: AdfId, seq: usize },
    /// Node `seq` fills a slot of type `want` and outputs `got`.
    AdfNodeType { id: AdfId, seq: usize, want: Ty, got: Ty },
    /// A function node `seq` with `got` of its `want` arguments present.
    AdfArity { id: AdfId, seq: usize, want: usize, got: usize },
    /// A `?_T` node with no `C_T` index.
    AdfConstIndex { id: AdfId, seq: usize },
    /// A genic terminal in a body.
    AdfGenicTerminal { id: AdfId, seq: usize, symbol: SymbolId },
    /// A reference from a body to an ADF whose level is not strictly lower.
    AdfRefNotLower { id: AdfId, seq: usize, target: AdfId, level: u32, target_level: u32 },
    /// The number of gene (or main) word vectors is not the layout's.
    ShapeGenes { want: usize, got: usize },
    ShapeMains { want: usize, got: usize },
    Gene { gene_id: u64, defect: LayoutDefect },
    Main { main_id: u64, defect: LayoutDefect },
    /// A genic terminal in a normal gene.
    GenicInNormalGene { gene_id: u64, pos: usize },
    /// A genic terminal in a main for a gene the main does not link.
    GenicNotLinked { main_id: u64, gene_id: u64, pos: usize },
    /// A main links a gene id the layout does not have.
    LinkedUnknown { main_id: u64, gene_id: u64 },
    /// More than one normal gene and no homeotic gene.
    NoLinker { genes: usize },
    /// A fixed linker's symbol is a terminal, or its signature is not the linked gene roots.
    FixedLinkerSignature { main_id: u64, want: Vec<Ty>, got: Vec<Ty> },
    /// A fixed linker whose tails cannot hold its genic terminals.
    FixedLinkerTail { main_id: u64, ty: Ty, need: usize, rows: usize },
    /// A fixed linker's words are not the canonical ones.
    FixedLinkerAltered { main_id: u64 },
}

impl fmt::Display for ChromosomeDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "our software has a bug: ")?;
        match self {
            ChromosomeDefect::Kingdom(m) => write!(f, "the epoch alphabet fails the kingdom check: {m}"),
            ChromosomeDefect::AdfDuplicate { id } => write!(f, "ADF {id} is declared twice"),
            ChromosomeDefect::AdfUndeclared { id } => write!(f, "ADF {id} has a body the epoch did not declare, or two"),
            ChromosomeDefect::AdfMissingBody { id } => write!(f, "ADF {id} is declared and has no body"),
            ChromosomeDefect::AdfDeclaration { id } => write!(f, "ADF {id} has a level or root type other than its declaration"),
            ChromosomeDefect::AdfRoot { id } => write!(f, "ADF {id} has no root node"),
            ChromosomeDefect::AdfNodeParent { id, seq } => write!(f, "ADF {id} node {seq} has a parent that is not an earlier node or an argument index its parent does not take"),
            ChromosomeDefect::AdfNodeType { id, seq, want, got } => write!(f, "ADF {id} node {seq} fills a slot of type {want:?} and outputs {got:?}"),
            ChromosomeDefect::AdfArity { id, seq, want, got } => write!(f, "ADF {id} node {seq} has {got} of its {want} arguments"),
            ChromosomeDefect::AdfConstIndex { id, seq } => write!(f, "ADF {id} node {seq} is a constant placeholder with no C_T index"),
            ChromosomeDefect::AdfGenicTerminal { id, seq, symbol } => write!(f, "ADF {id} node {seq} is genic terminal {symbol}; an ADF is one tree for the population"),
            ChromosomeDefect::AdfRefNotLower { id, seq, target, level, target_level } => {
                write!(f, "ADF {id} (level {level}) node {seq} references ADF {target} at level {target_level}, which is not strictly lower")
            }
            ChromosomeDefect::ShapeGenes { want, got } => write!(f, "a chromosome has {got} normal genes, its layout has {want}"),
            ChromosomeDefect::ShapeMains { want, got } => write!(f, "a chromosome has {got} homeotic genes, its layout has {want}"),
            ChromosomeDefect::Gene { gene_id, defect } => write!(f, "gene {gene_id}: {defect}"),
            ChromosomeDefect::Main { main_id, defect } => write!(f, "homeotic gene {main_id}: {defect}"),
            ChromosomeDefect::GenicInNormalGene { gene_id, pos } => write!(f, "normal gene {gene_id} holds a genic terminal at position {pos}"),
            ChromosomeDefect::GenicNotLinked { main_id, gene_id, pos } => write!(f, "homeotic gene {main_id} holds the genic terminal of gene {gene_id} at position {pos}, which it does not link"),
            ChromosomeDefect::LinkedUnknown { main_id, gene_id } => write!(f, "homeotic gene {main_id} links gene {gene_id}, which the layout does not have"),
            ChromosomeDefect::NoLinker { genes } => write!(f, "a chromosome of {genes} genes has no homeotic gene"),
            ChromosomeDefect::FixedLinkerSignature { main_id, want, got } => write!(f, "fixed linker {main_id} takes {got:?}, the linked gene roots are {want:?}"),
            ChromosomeDefect::FixedLinkerTail { main_id, ty, need, rows } => write!(f, "fixed linker {main_id} needs {need} {ty:?} rows for its genic terminals, its layout has {rows}"),
            ChromosomeDefect::FixedLinkerAltered { main_id } => write!(f, "fixed linker {main_id} is not its canonical words"),
        }
    }
}

// ---------------------------------------------------------------------------
// The epoch alphabet
// ---------------------------------------------------------------------------

/// The epoch alphabet: a kingdom plus one genic terminal per normal gene id and
/// one ADF reference per declared ADF, as a checked contract. The reference
/// terminals have no signature, so `a_{T,U}`, `n_U` and every bound are those of
/// the kingdom.
#[derive(Debug, Clone)]
pub struct Epoch {
    k: KingdomContract,
    n_base: usize,
    refs: BTreeMap<SymbolId, RefKind>,
    genic: Vec<SymbolId>,
    adf_sym: BTreeMap<AdfId, SymbolId>,
    decls: BTreeMap<AdfId, AdfDecl>,
}

impl Epoch {
    /// `gene_roots[g]` is the layout's root of normal gene `g`.
    pub fn new(base: &KingdomContract, gene_roots: &[Ty], adfs: &[AdfDecl]) -> Result<Epoch, ChromosomeDefect> {
        let mut rows: Vec<(String, Vec<Ty>, Ty)> = (0..base.n_symbols() as SymbolId).map(|s| (base.name(s).to_string(), base.signature(s).to_vec(), base.output(s))).collect();
        let constants: Vec<String> = (0..base.n_symbols() as SymbolId).filter(|s| base.is_constant(*s)).map(|s| base.name(s).to_string()).collect();
        let n_base = rows.len();
        let mut refs = BTreeMap::new();
        let mut genic = Vec::new();
        for (g, root) in gene_roots.iter().enumerate() {
            let id = rows.len() as SymbolId;
            rows.push((format!("G{g}"), Vec::new(), *root));
            refs.insert(id, RefKind::Genic { gene_id: g as u64 });
            genic.push(id);
        }
        let mut decls = BTreeMap::new();
        let mut sorted: Vec<AdfDecl> = adfs.to_vec();
        sorted.sort_by_key(|d| d.id);
        let mut adf_sym = BTreeMap::new();
        for d in sorted {
            if decls.insert(d.id, d).is_some() {
                return Err(ChromosomeDefect::AdfDuplicate { id: d.id });
            }
            let id = rows.len() as SymbolId;
            rows.push((format!("A{}", d.id), Vec::new(), d.root));
            refs.insert(id, RefKind::Adf { adf_id: d.id });
            adf_sym.insert(d.id, id);
        }
        let c: Vec<&str> = constants.iter().map(String::as_str).collect();
        let k = relations_from_rows(base.kingdom(), base.roots().to_vec(), &rows, &c).check().map_err(|d| ChromosomeDefect::Kingdom(d.join("; ")))?;
        Ok(Epoch { k, n_base, refs, genic, adf_sym, decls })
    }

    /// The epoch alphabet as a contract: decode and lay genes out against this.
    pub fn kingdom(&self) -> &KingdomContract {
        &self.k
    }
    /// The number of the kingdom's own symbols; the references follow.
    pub fn n_base(&self) -> usize {
        self.n_base
    }
    pub fn reference(&self, symbol: SymbolId) -> Option<RefKind> {
        self.refs.get(&symbol).copied()
    }
    /// The genic terminal of normal gene `gene_id`.
    pub fn genic_terminal(&self, gene_id: u64) -> Option<SymbolId> {
        self.genic.get(gene_id as usize).copied()
    }
    /// `g_T`: how many genic terminals output `ty`.
    pub fn genic_count(&self, ty: Ty) -> usize {
        self.genic.iter().filter(|s| self.k.output(**s) == ty).count()
    }
    pub fn adf_terminal(&self, id: AdfId) -> Option<SymbolId> {
        self.adf_sym.get(&id).copied()
    }
    pub fn decl(&self, id: AdfId) -> Option<AdfDecl> {
        self.decls.get(&id).copied()
    }
    fn is_genic(&self, symbol: SymbolId) -> Option<u64> {
        match self.refs.get(&symbol) {
            Some(RefKind::Genic { gene_id }) => Some(*gene_id),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ADFs: immutable expressed trees
// ---------------------------------------------------------------------------

/// One row of `adf_node(adf_id, node_seq, symbol_id, parent_seq, arg_index, dc_index)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdfNode {
    pub symbol: SymbolId,
    pub parent_seq: Option<usize>,
    pub arg_index: Option<u32>,
    /// The `C_T` index of a `?_T` node.
    pub dc_index: Option<u32>,
}

/// A shared definition: an expressed tree in level order, never a gene.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adf {
    pub id: AdfId,
    pub root_type: Ty,
    pub level: u32,
    pub nodes: Vec<AdfNode>,
}

impl Adf {
    /// The rows of `tree`, in level order. Validated by [`AdfSet::new`].
    pub fn from_tree(epoch: &Epoch, id: AdfId, level: u32, tree: &Tree) -> Adf {
        let order = tree.level_order();
        let seq_of: BTreeMap<Vec<u32>, usize> = order.iter().enumerate().map(|(i, (_, p))| (p.clone(), i)).collect();
        let nodes = order
            .iter()
            .map(|(t, path)| AdfNode {
                symbol: t.symbol,
                parent_seq: path.split_last().map(|(_, up)| seq_of[up]),
                arg_index: path.last().copied(),
                dc_index: t.const_index,
            })
            .collect();
        Adf { id, root_type: epoch.k.output(tree.symbol), level, nodes }
    }

    /// The tree the rows describe. Only meaningful for rows that passed [`AdfSet::new`].
    pub fn tree(&self) -> Tree {
        fn build(nodes: &[AdfNode], kids: &[Vec<(u32, usize)>], i: usize) -> Tree {
            let mut ks = kids[i].clone();
            ks.sort_unstable();
            Tree { symbol: nodes[i].symbol, const_index: nodes[i].dc_index, kids: ks.into_iter().map(|(_, c)| build(nodes, kids, c)).collect() }
        }
        let mut kids: Vec<Vec<(u32, usize)>> = vec![Vec::new(); self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            if let (Some(p), Some(a)) = (n.parent_seq, n.arg_index) {
                if p < i {
                    kids[p].push((a, i));
                }
            }
        }
        build(&self.nodes, &kids, 0)
    }

    fn check(&self, epoch: &Epoch, levels: &BTreeMap<AdfId, u32>) -> Result<(), ChromosomeDefect> {
        let id = self.id;
        let k = &epoch.k;
        let decl = epoch.decl(id).ok_or(ChromosomeDefect::AdfUndeclared { id })?;
        if decl.root != self.root_type || decl.level != self.level {
            return Err(ChromosomeDefect::AdfDeclaration { id });
        }
        let Some(root) = self.nodes.first() else { return Err(ChromosomeDefect::AdfRoot { id }) };
        if root.parent_seq.is_some() || root.arg_index.is_some() || k.output(root.symbol) != self.root_type {
            return Err(ChromosomeDefect::AdfRoot { id });
        }
        let mut have: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); self.nodes.len()];
        for (seq, n) in self.nodes.iter().enumerate() {
            if n.symbol as usize >= k.n_symbols() {
                return Err(ChromosomeDefect::AdfNodeParent { id, seq });
            }
            if epoch.is_genic(n.symbol).is_some() {
                return Err(ChromosomeDefect::AdfGenicTerminal { id, seq, symbol: n.symbol });
            }
            if let Some(RefKind::Adf { adf_id }) = epoch.reference(n.symbol) {
                let target_level = levels.get(&adf_id).copied().unwrap_or(u32::MAX);
                if target_level >= self.level {
                    return Err(ChromosomeDefect::AdfRefNotLower { id, seq, target: adf_id, level: self.level, target_level });
                }
            }
            if k.is_constant(n.symbol) && n.dc_index.is_none() {
                return Err(ChromosomeDefect::AdfConstIndex { id, seq });
            }
            if seq == 0 {
                continue;
            }
            let (Some(p), Some(a)) = (n.parent_seq, n.arg_index) else { return Err(ChromosomeDefect::AdfNodeParent { id, seq }) };
            if p >= seq {
                return Err(ChromosomeDefect::AdfNodeParent { id, seq });
            }
            let sig = k.signature(self.nodes[p].symbol);
            let want = *sig.get(a as usize).ok_or(ChromosomeDefect::AdfNodeParent { id, seq })?;
            if !have[p].insert(a) {
                return Err(ChromosomeDefect::AdfNodeParent { id, seq });
            }
            if k.output(n.symbol) != want {
                return Err(ChromosomeDefect::AdfNodeType { id, seq, want, got: k.output(n.symbol) });
            }
        }
        for (seq, n) in self.nodes.iter().enumerate() {
            let want = k.signature(n.symbol).len();
            if have[seq].len() != want {
                return Err(ChromosomeDefect::AdfArity { id, seq, want, got: have[seq].len() });
            }
        }
        Ok(())
    }
}

/// The epoch's ADFs, checked: one body per declaration, every body well typed,
/// holding no genic terminal and referencing only strictly lower levels.
#[derive(Debug, Clone)]
pub struct AdfSet {
    adfs: BTreeMap<AdfId, Adf>,
}

impl AdfSet {
    pub fn new(epoch: &Epoch, bodies: Vec<Adf>) -> Result<AdfSet, ChromosomeDefect> {
        let mut adfs: BTreeMap<AdfId, Adf> = BTreeMap::new();
        for b in bodies {
            let id = b.id;
            if epoch.decl(id).is_none() || adfs.insert(id, b).is_some() {
                return Err(ChromosomeDefect::AdfUndeclared { id });
            }
        }
        for id in epoch.decls.keys() {
            if !adfs.contains_key(id) {
                return Err(ChromosomeDefect::AdfMissingBody { id: *id });
            }
        }
        let levels: BTreeMap<AdfId, u32> = adfs.values().map(|a| (a.id, a.level)).collect();
        for a in adfs.values() {
            a.check(epoch, &levels)?;
        }
        Ok(AdfSet { adfs })
    }

    pub fn get(&self, id: AdfId) -> Option<&Adf> {
        self.adfs.get(&id)
    }

    /// Execution order: a definition after everything it references.
    pub fn schedule(&self) -> Vec<(AdfId, u32)> {
        let mut v: Vec<(AdfId, u32)> = self.adfs.values().map(|a| (a.id, a.level)).collect();
        v.sort_by_key(|(id, level)| (*level, *id));
        v
    }

    /// Each ADF's tree with every reference to a lower ADF replaced by its tree.
    fn expanded(&self, epoch: &Epoch) -> BTreeMap<AdfId, Tree> {
        let mut out: BTreeMap<AdfId, Tree> = BTreeMap::new();
        for (id, _) in self.schedule() {
            let t = expand(&self.adfs[&id].tree(), epoch, &out, &BTreeMap::new());
            out.insert(id, t);
        }
        out
    }

    /// How many references to `id` the ADF bodies and the chromosomes make; an
    /// ADF with none may be retired at an epoch boundary.
    pub fn referrers(&self, epoch: &Epoch, id: AdfId, chromosomes: &[(&ChromosomeLayout, &Chromosome)], const_len: &BTreeMap<Ty, u32>) -> Result<usize, ChromosomeDefect> {
        let Some(sym) = epoch.adf_terminal(id) else { return Ok(0) };
        let mut n = self.adfs.values().map(|a| a.nodes.iter().filter(|x| x.symbol == sym).count()).sum::<usize>();
        for (cl, c) in chromosomes {
            let p = express_chromosome(epoch, self, cl, c, const_len)?;
            n += p.refs.iter().filter(|r| r.symbol == sym && !matches!(r.source, Source::Adf(_))).count();
        }
        Ok(n)
    }
}

/// `t` with every ADF reference replaced by that ADF's tree and every genic
/// terminal by its gene's tree.
fn expand(t: &Tree, epoch: &Epoch, adfs: &BTreeMap<AdfId, Tree>, genes: &BTreeMap<u64, Tree>) -> Tree {
    match epoch.reference(t.symbol) {
        Some(RefKind::Adf { adf_id }) => {
            if let Some(body) = adfs.get(&adf_id) {
                return body.clone();
            }
        }
        Some(RefKind::Genic { gene_id }) => {
            if let Some(body) = genes.get(&gene_id) {
                return body.clone();
            }
        }
        None => {}
    }
    Tree { symbol: t.symbol, const_index: t.const_index, kids: t.kids.iter().map(|c| expand(c, epoch, adfs, genes)).collect() }
}

// ---------------------------------------------------------------------------
// Chromosomes
// ---------------------------------------------------------------------------

/// One homeotic gene of a layout: its own gene layout (the root is the main
/// program's phenotype type), the normal genes it links, and, for a fixed
/// linker, the linking function that makes it immutable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MainLayout {
    pub main_id: u64,
    pub gene: GeneLayout,
    pub linked: Vec<u64>,
    /// `Some(link)`: a fixed linker `head_R = (link)`, tails the genic
    /// terminals of the linked genes in gene order; every operator's domain on
    /// it is empty.
    pub fixed: Option<SymbolId>,
}

/// The layout of a chromosome: its normal genes and its homeotic genes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChromosomeLayout {
    pub normal: Layout,
    pub mains: Vec<MainLayout>,
}

impl ChromosomeLayout {
    /// The gene roots in gene-id order: what [`Epoch::new`] needs.
    pub fn gene_roots(&self) -> Vec<Ty> {
        self.normal.genes().iter().map(|g| g.root).collect()
    }

    /// The homologous ordered pairs of normal gene ids.
    pub fn homologous_pairs(&self) -> Vec<(u64, u64)> {
        self.normal.homologous_pairs()
    }

    /// A fixed linker over `linked` genes: the function `link` takes the linked
    /// roots in order, its output is the phenotype type, `head_R = (link)`, and
    /// the tails hold the genic terminals. The words are
    /// [`ChromosomeLayout::fixed_words`].
    pub fn fixed_main(&self, epoch: &Epoch, main_id: u64, link: SymbolId, linked: Vec<u64>) -> Result<MainLayout, ChromosomeDefect> {
        let k = epoch.kingdom();
        let roots: Vec<Ty> = linked
            .iter()
            .map(|g| self.normal.gene(*g).map(|l| l.root).ok_or(ChromosomeDefect::LinkedUnknown { main_id, gene_id: *g }))
            .collect::<Result<_, _>>()?;
        if k.is_terminal(link) || k.signature(link) != roots.as_slice() {
            return Err(ChromosomeDefect::FixedLinkerSignature { main_id, want: roots, got: k.signature(link).to_vec() });
        }
        let phenotype = k.output(link);
        let gene = GeneLayout::from_heads(k, phenotype, &BTreeMap::from([(phenotype, 1)])).map_err(|defect| ChromosomeDefect::Main { main_id, defect })?;
        let main = MainLayout { main_id, gene, linked, fixed: Some(link) };
        self.fixed_words(epoch, &main)?;
        Ok(main)
    }

    /// The canonical words of a fixed linker.
    pub fn fixed_words(&self, epoch: &Epoch, main: &MainLayout) -> Result<Vec<u32>, ChromosomeDefect> {
        let k = epoch.kingdom();
        let Some(link) = main.fixed else { return Err(ChromosomeDefect::FixedLinkerAltered { main_id: main.main_id }) };
        let root = main.gene.root;
        // Filler for rows the walk never reads: the lowest terminal of the type, or index zero.
        let mut words = vec![0u32; main.gene.width];
        for s in main.gene.spans() {
            for w in &mut words[s.off..s.off + s.len] {
                *w = match s.domain {
                    Domain::Head => link,
                    Domain::Tail => k.terminals_of(s.ty)[0],
                    Domain::Dc => 0,
                };
            }
        }
        let mut next: BTreeMap<Ty, usize> = BTreeMap::new();
        next.insert(root, 1);
        for g in &main.linked {
            let sym = epoch.genic_terminal(*g).ok_or(ChromosomeDefect::LinkedUnknown { main_id: main.main_id, gene_id: *g })?;
            let ty = k.output(sym);
            let frame = main.gene.frame(ty).ok_or(ChromosomeDefect::FixedLinkerTail { main_id: main.main_id, ty, need: 1, rows: 0 })?;
            let r = next.entry(ty).or_default();
            let pos = frame.row_pos(*r).ok_or(ChromosomeDefect::FixedLinkerTail { main_id: main.main_id, ty, need: *r + 1, rows: frame.head_len + frame.tail_len })?;
            words[pos] = sym;
            *r += 1;
        }
        Ok(words)
    }
}

/// A chromosome: the words of each normal gene (gene id = index) and of each
/// homeotic gene (in the layout's order). The records carry no root and no
/// length; both belong to the layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chromosome {
    pub chromosome_id: u64,
    pub genes: Vec<Vec<u32>>,
    pub mains: Vec<Vec<u32>>,
}

/// One reference, as expressed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefUse {
    pub source: Source,
    pub symbol: SymbolId,
    pub step: usize,
    pub kind: RefKind,
}

/// One main program's expression.
#[derive(Debug, Clone)]
pub struct MainPhenotype {
    pub main_id: u64,
    /// The declared phenotype type: the root of the homeotic gene.
    pub ty: Ty,
    pub expressed: Expressed,
}

/// The expressed chromosome.
#[derive(Debug, Clone)]
pub struct Phenotype {
    /// Each normal gene's expression, in gene order.
    pub genes: Vec<Expressed>,
    /// Each homeotic gene's expression, in main order.
    pub mains: Vec<MainPhenotype>,
    /// The ADFs in execution order: a definition after everything it references.
    pub schedule: Vec<(AdfId, u32)>,
    /// Every reference expressed in a gene or a main.
    pub refs: Vec<RefUse>,
}

fn refs_of(epoch: &Epoch, e: &Expressed, source: Source) -> Vec<RefUse> {
    e.nodes.iter().filter_map(|n| epoch.reference(n.symbol).map(|kind| RefUse { source, symbol: n.symbol, step: n.step, kind })).collect()
}

/// The cells of `gene` that hold a genic terminal, by position: only head and
/// tail cells hold symbols.
fn genic_cells(epoch: &Epoch, layout: &GeneLayout, gene: &[u32]) -> Vec<(usize, u64)> {
    layout
        .spans()
        .iter()
        .filter(|s| s.domain != Domain::Dc)
        .flat_map(|s| (s.off..s.off + s.len).filter_map(|p| gene.get(p).and_then(|w| epoch.is_genic(*w)).map(|g| (p, g))))
        .collect()
}

/// Express a chromosome against the epoch: every normal gene (no genic
/// terminal in any), every homeotic gene (only the genic terminals of the genes
/// it links; a fixed linker must be its canonical words), and the ADF schedule.
pub fn express_chromosome(epoch: &Epoch, adfs: &AdfSet, cl: &ChromosomeLayout, c: &Chromosome, const_len: &BTreeMap<Ty, u32>) -> Result<Phenotype, ChromosomeDefect> {
    let k = epoch.kingdom();
    if c.genes.len() != cl.normal.n_genes() {
        return Err(ChromosomeDefect::ShapeGenes { want: cl.normal.n_genes(), got: c.genes.len() });
    }
    if c.mains.len() != cl.mains.len() {
        return Err(ChromosomeDefect::ShapeMains { want: cl.mains.len(), got: c.mains.len() });
    }
    let mut refs = Vec::new();
    let mut genes = Vec::new();
    for (g, words) in c.genes.iter().enumerate() {
        let gene_id = g as u64;
        let layout = cl.normal.gene(gene_id).ok_or(ChromosomeDefect::ShapeGenes { want: cl.normal.n_genes(), got: c.genes.len() })?;
        if let Some((pos, _)) = genic_cells(epoch, layout, words).first() {
            return Err(ChromosomeDefect::GenicInNormalGene { gene_id, pos: *pos });
        }
        let e = express(k, layout, words, const_len).map_err(|defect| ChromosomeDefect::Gene { gene_id, defect })?;
        refs.extend(refs_of(epoch, &e, Source::Gene(gene_id)));
        genes.push(e);
    }
    let mut mains = Vec::new();
    for (m, words) in cl.mains.iter().zip(&c.mains) {
        for g in &m.linked {
            if cl.normal.gene(*g).is_none() {
                return Err(ChromosomeDefect::LinkedUnknown { main_id: m.main_id, gene_id: *g });
            }
        }
        if let Some((pos, g)) = genic_cells(epoch, &m.gene, words).into_iter().find(|(_, g)| !m.linked.contains(g)) {
            return Err(ChromosomeDefect::GenicNotLinked { main_id: m.main_id, gene_id: g, pos });
        }
        if m.fixed.is_some() && *words != cl.fixed_words(epoch, m)? {
            return Err(ChromosomeDefect::FixedLinkerAltered { main_id: m.main_id });
        }
        let e = express(k, &m.gene, words, const_len).map_err(|defect| ChromosomeDefect::Main { main_id: m.main_id, defect })?;
        refs.extend(refs_of(epoch, &e, Source::Main(m.main_id)));
        mains.push(MainPhenotype { main_id: m.main_id, ty: m.gene.root, expressed: e });
    }
    if mains.is_empty() && genes.len() > 1 {
        return Err(ChromosomeDefect::NoLinker { genes: genes.len() });
    }
    for (id, _) in adfs.schedule() {
        if let Some(a) = adfs.get(id) {
            for (seq, n) in a.nodes.iter().enumerate() {
                if let Some(kind) = epoch.reference(n.symbol) {
                    refs.push(RefUse { source: Source::Adf(id), symbol: n.symbol, step: seq + 1, kind });
                }
            }
        }
    }
    Ok(Phenotype { genes, mains, schedule: adfs.schedule(), refs })
}

impl Phenotype {
    /// The result tuple: one tree per main program, in main order, with every
    /// ADF reference replaced by its tree and every genic terminal by its gene's
    /// tree. A lone normal gene with no homeotic gene is its own program.
    pub fn inlined(&self, epoch: &Epoch, adfs: &AdfSet) -> Vec<(u64, Tree)> {
        let adf_trees = adfs.expanded(epoch);
        let gene_trees: BTreeMap<u64, Tree> = self.genes.iter().enumerate().map(|(g, e)| (g as u64, expand(&e.tree(), epoch, &adf_trees, &BTreeMap::new()))).collect();
        if self.mains.is_empty() {
            return self.genes.first().map(|_| vec![(0, gene_trees[&0].clone())]).unwrap_or_default();
        }
        self.mains.iter().map(|m| (m.main_id, expand(&m.expressed.tree(), epoch, &adf_trees, &gene_trees))).collect()
    }
}

// ---------------------------------------------------------------------------
// Fold: a shared subtree becomes an ADF reference
// ---------------------------------------------------------------------------

/// `tree` with every subtree equal to `body` replaced by the leaf `reference`;
/// the number of replacements.
fn replace_subtrees(tree: &Tree, body: &Tree, reference: SymbolId) -> (Tree, usize) {
    if tree == body {
        return (Tree { symbol: reference, const_index: None, kids: Vec::new() }, 1);
    }
    let mut n = 0;
    let kids = tree
        .kids
        .iter()
        .map(|c| {
            let (t, m) = replace_subtrees(c, body, reference);
            n += m;
            t
        })
        .collect();
    (Tree { symbol: tree.symbol, const_index: tree.const_index, kids }, n)
}

/// Fold ADF `adf_id` into a chromosome: every subtree of a normal gene or a
/// non-fixed homeotic gene equal to the ADF's body becomes the reference `A_d`,
/// by the per-type write-back. The rewrite only deletes nodes, so no rank grows
/// and it always fits; a write-back that fails anyway is returned as a defect.
/// Returns the chromosome and the number of subtrees replaced.
pub fn fold_adf(epoch: &Epoch, adfs: &AdfSet, cl: &ChromosomeLayout, c: &Chromosome, adf_id: AdfId, const_len: &BTreeMap<Ty, u32>) -> Result<(Chromosome, usize), ChromosomeDefect> {
    let body = adfs.get(adf_id).ok_or(ChromosomeDefect::AdfUndeclared { id: adf_id })?.tree();
    let reference = epoch.adf_terminal(adf_id).ok_or(ChromosomeDefect::AdfUndeclared { id: adf_id })?;
    let p = express_chromosome(epoch, adfs, cl, c, const_len)?;
    let k = epoch.kingdom();
    let mut out = c.clone();
    let mut total = 0;
    for (g, e) in p.genes.iter().enumerate() {
        let (t, n) = replace_subtrees(&e.tree(), &body, reference);
        if n > 0 {
            let layout = cl.normal.gene(g as u64).ok_or(ChromosomeDefect::ShapeGenes { want: cl.normal.n_genes(), got: c.genes.len() })?;
            out.genes[g] = write_back(k, layout, &c.genes[g], &t).map_err(|defect| ChromosomeDefect::Gene { gene_id: g as u64, defect })?;
            total += n;
        }
    }
    for (i, (m, mp)) in cl.mains.iter().zip(&p.mains).enumerate() {
        if m.fixed.is_some() {
            continue;
        }
        let (t, n) = replace_subtrees(&mp.expressed.tree(), &body, reference);
        if n > 0 {
            out.mains[i] = write_back(k, &m.gene, &c.mains[i], &t).map_err(|defect| ChromosomeDefect::Main { main_id: m.main_id, defect })?;
            total += n;
        }
    }
    Ok((out, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAIR: SymbolId = 0;
    const LINK: SymbolId = 1;
    const S: SymbolId = 2;
    const I: SymbolId = 3;
    const Q: SymbolId = 4; // constant placeholder, S

    /// `pair:S(S,I)`, `link:S(S,S)`, `s:S`, `i:I`, and `?:S`.
    fn base() -> KingdomContract {
        relations_from_rows(
            "h",
            vec![Ty::S],
            &[
                ("pair".to_string(), vec![Ty::S, Ty::I], Ty::S),
                ("link".to_string(), vec![Ty::S, Ty::S], Ty::S),
                ("s".to_string(), vec![], Ty::S),
                ("i".to_string(), vec![], Ty::I),
                ("?".to_string(), vec![], Ty::S),
            ],
            &["?"],
        )
        .check()
        .expect("valid")
    }

    fn consts() -> BTreeMap<Ty, u32> {
        BTreeMap::from([(Ty::S, 4)])
    }

    fn leaf(symbol: SymbolId) -> Tree {
        Tree { symbol, const_index: None, kids: vec![] }
    }
    fn node(symbol: SymbolId, kids: Vec<Tree>) -> Tree {
        Tree { symbol, const_index: None, kids }
    }

    /// Normal genes: 0 and 1 rooted at S with N = 2 (homologous), gene 2 rooted at S with N = 1.
    fn layout(k: &KingdomContract) -> Layout {
        let s2 = || GeneLayout::from_program_size(k, Ty::S, 2).expect("layout");
        Layout::new(5, vec![s2(), s2(), GeneLayout::from_program_size(k, Ty::S, 1).expect("layout")])
    }

    /// Words of a normal gene of layout `l` holding exactly `tree`, the rest `s`/`i`/0.
    fn gene_for(k: &KingdomContract, l: &GeneLayout, tree: &Tree) -> Vec<u32> {
        let mut filler = vec![0u32; l.width];
        for s in l.spans() {
            for w in &mut filler[s.off..s.off + s.len] {
                *w = match s.domain {
                    Domain::Head | Domain::Tail => k.terminals_of(s.ty).iter().copied().find(|t| !k.is_constant(*t)).unwrap_or(k.terminals_of(s.ty)[0]),
                    Domain::Dc => 0,
                };
            }
        }
        write_back(k, l, &filler, tree).expect("fits")
    }

    struct Fixture {
        epoch: Epoch,
        adfs: AdfSet,
        cl: ChromosomeLayout,
    }

    /// An epoch over the layout above with the declared ADFs and their bodies.
    fn fixture(adf_defs: &[(AdfId, u32, Tree)], mains: impl Fn(&Epoch, &Layout) -> Vec<MainLayout>) -> Fixture {
        let k = base();
        let normal = layout(&k);
        let roots: Vec<Ty> = normal.genes().iter().map(|g| g.root).collect();
        let decls: Vec<AdfDecl> = adf_defs.iter().map(|(id, level, t)| AdfDecl { id: *id, root: k.output(t.symbol), level: *level }).collect();
        let epoch = Epoch::new(&k, &roots, &decls).expect("epoch");
        let bodies = adf_defs.iter().map(|(id, level, t)| Adf::from_tree(&epoch, *id, *level, t)).collect();
        let adfs = AdfSet::new(&epoch, bodies).expect("adfs");
        let cl = ChromosomeLayout { mains: mains(&epoch, &normal), normal };
        Fixture { epoch, adfs, cl }
    }

    /// The evolved homeotic gene: root S, N = 1, so `[head_S0, tail_S0, tail_S1, tail_I0, dc..]`.
    fn evolved_main(epoch: &Epoch, main_id: u64, linked: Vec<u64>) -> MainLayout {
        MainLayout { main_id, gene: GeneLayout::from_program_size(epoch.kingdom(), Ty::S, 1).expect("layout"), linked, fixed: None }
    }

    fn main_words(epoch: &Epoch, m: &MainLayout, tree: &Tree) -> Vec<u32> {
        gene_for(epoch.kingdom(), &m.gene, tree)
    }

    #[test]
    fn genic_terminals_come_one_per_gene_with_the_genes_own_root() {
        let f = fixture(&[], |e, _| vec![evolved_main(e, 0, vec![0, 1])]);
        let k = f.epoch.kingdom();
        assert_eq!(f.epoch.n_base(), 5);
        for g in 0..3u64 {
            let sym = f.epoch.genic_terminal(g).expect("genic terminal");
            assert!(k.is_terminal(sym) && f.epoch.reference(sym) == Some(RefKind::Genic { gene_id: g }));
            assert_eq!(k.output(sym), f.cl.normal.gene(g).expect("gene").root, "the genic terminal's type is the layout's root");
        }
        assert_eq!(f.epoch.genic_count(Ty::S), f.cl.normal.genes_rooted_at(Ty::S));
        assert_eq!(f.epoch.genic_count(Ty::I), 0);
        assert!(f.epoch.genic_terminal(3).is_none());
        assert_eq!(f.cl.homologous_pairs(), vec![(0, 1), (1, 0)], "genes 0 and 1 are homologous, gene 2 has another head set");
    }

    #[test]
    fn two_genes_under_an_evolved_homeotic_gene_over_the_whole_alphabet() {
        let f = fixture(&[], |e, _| vec![evolved_main(e, 0, vec![0, 1])]);
        let k = f.epoch.kingdom();
        let (g0, g1) = (f.epoch.genic_terminal(0).expect("G0"), f.epoch.genic_terminal(1).expect("G1"));
        let t0 = node(PAIR, vec![leaf(S), leaf(I)]);
        let t1 = leaf(S);
        // The homeotic head holds a linking function and the tail the genic terminals.
        let main_tree = node(LINK, vec![leaf(g0), leaf(g1)]);
        let c = Chromosome {
            chromosome_id: 3,
            genes: vec![gene_for(k, f.cl.normal.gene(0).expect("g"), &t0), gene_for(k, f.cl.normal.gene(1).expect("g"), &t1), gene_for(k, f.cl.normal.gene(2).expect("g"), &leaf(S))],
            mains: vec![main_words(&f.epoch, &f.cl.mains[0], &main_tree)],
        };
        let p = express_chromosome(&f.epoch, &f.adfs, &f.cl, &c, &consts()).expect("expresses");
        assert_eq!(p.mains[0].ty, Ty::S);
        assert_eq!(p.inlined(&f.epoch, &f.adfs), vec![(0, node(LINK, vec![t0.clone(), t1.clone()]))]);
        // The homeotic head may also hold ordinary functions: pair(link(G0, G1), i) is a program too.
        let wide = node(PAIR, vec![node(LINK, vec![leaf(g0), leaf(g1)]), leaf(I)]);
        let mut c2 = c.clone();
        c2.mains = vec![gene_for(k, &GeneLayout::from_program_size(k, Ty::S, 2).expect("layout"), &wide)];
        let cl2 = ChromosomeLayout { normal: f.cl.normal.clone(), mains: vec![MainLayout { gene: GeneLayout::from_program_size(k, Ty::S, 2).expect("layout"), ..f.cl.mains[0].clone() }] };
        let p2 = express_chromosome(&f.epoch, &f.adfs, &cl2, &c2, &consts()).expect("expresses");
        assert_eq!(p2.inlined(&f.epoch, &f.adfs), vec![(0, node(PAIR, vec![node(LINK, vec![t0, t1]), leaf(I)]))]);
        assert_eq!(p.refs.iter().filter(|r| matches!(r.kind, RefKind::Genic { .. })).count(), 2);
    }

    #[test]
    fn a_fixed_linker_is_the_canonical_words_and_equals_the_evolved_one() {
        let f = fixture(&[], |e, l| {
            let cl = ChromosomeLayout { normal: l.clone(), mains: vec![] };
            vec![cl.fixed_main(e, 0, LINK, vec![0, 1]).expect("fixed")]
        });
        let words = f.cl.fixed_words(&f.epoch, &f.cl.mains[0]).expect("words");
        let (g0, g1) = (f.epoch.genic_terminal(0).expect("G0"), f.epoch.genic_terminal(1).expect("G1"));
        assert_eq!(&words[..3], &[LINK, g0, g1], "head_S = (link), tail_S = the genic terminals in gene order");
        let k = f.epoch.kingdom();
        let genes = vec![gene_for(k, f.cl.normal.gene(0).expect("g"), &leaf(S)), gene_for(k, f.cl.normal.gene(1).expect("g"), &node(PAIR, vec![leaf(S), leaf(I)])), gene_for(k, f.cl.normal.gene(2).expect("g"), &leaf(S))];
        let fixed = Chromosome { chromosome_id: 1, genes: genes.clone(), mains: vec![words.clone()] };
        let evolved_layout = ChromosomeLayout { normal: f.cl.normal.clone(), mains: vec![MainLayout { fixed: None, ..f.cl.mains[0].clone() }] };
        let a = express_chromosome(&f.epoch, &f.adfs, &f.cl, &fixed, &consts()).expect("fixed expresses");
        let b = express_chromosome(&f.epoch, &f.adfs, &evolved_layout, &fixed, &consts()).expect("same words, evolved");
        assert_eq!(a.inlined(&f.epoch, &f.adfs), b.inlined(&f.epoch, &f.adfs));
        // Any change to a fixed linker is a defect; the same change to an evolved one is a variation.
        let mut altered = fixed.clone();
        altered.mains[0][1] = g1;
        assert_eq!(express_chromosome(&f.epoch, &f.adfs, &f.cl, &altered, &consts()).expect_err("altered"), ChromosomeDefect::FixedLinkerAltered { main_id: 0 });
        assert!(express_chromosome(&f.epoch, &f.adfs, &evolved_layout, &altered, &consts()).is_ok());
        // A link whose signature is not the linked roots is a defect at construction.
        let cl = ChromosomeLayout { normal: f.cl.normal.clone(), mains: vec![] };
        assert!(matches!(cl.fixed_main(&f.epoch, 1, LINK, vec![0]), Err(ChromosomeDefect::FixedLinkerSignature { .. })));
        assert!(matches!(cl.fixed_main(&f.epoch, 1, S, vec![0, 1]), Err(ChromosomeDefect::FixedLinkerSignature { .. })));
        assert!(matches!(cl.fixed_main(&f.epoch, 1, LINK, vec![0, 9]), Err(ChromosomeDefect::LinkedUnknown { main_id: 1, gene_id: 9 })));
    }

    #[test]
    fn a_genic_terminal_is_valid_only_in_a_homeotic_gene_that_links_its_gene() {
        let f = fixture(&[], |e, _| vec![evolved_main(e, 0, vec![0])]);
        let k = f.epoch.kingdom();
        let (g0, g1) = (f.epoch.genic_terminal(0).expect("G0"), f.epoch.genic_terminal(1).expect("G1"));
        let plain = |g: u64| gene_for(k, f.cl.normal.gene(g).expect("g"), &leaf(S));
        let ok = Chromosome { chromosome_id: 0, genes: vec![plain(0), plain(1), plain(2)], mains: vec![main_words(&f.epoch, &f.cl.mains[0], &leaf(g0))] };
        assert!(express_chromosome(&f.epoch, &f.adfs, &f.cl, &ok, &consts()).is_ok());
        // G1 in a main that links only gene 0.
        let mut unlinked = ok.clone();
        unlinked.mains = vec![main_words(&f.epoch, &f.cl.mains[0], &leaf(g1))];
        assert_eq!(express_chromosome(&f.epoch, &f.adfs, &f.cl, &unlinked, &consts()).expect_err("unlinked"), ChromosomeDefect::GenicNotLinked { main_id: 0, gene_id: 1, pos: 0 });
        // Even in a cell the walk never reads.
        let mut unread = ok.clone();
        unread.mains[0][2] = g1;
        assert!(matches!(express_chromosome(&f.epoch, &f.adfs, &f.cl, &unread, &consts()), Err(ChromosomeDefect::GenicNotLinked { gene_id: 1, pos: 2, .. })));
        // A genic terminal in a normal gene.
        let mut in_normal = ok.clone();
        in_normal.genes[1] = gene_for(k, f.cl.normal.gene(1).expect("g"), &leaf(g0));
        let err = express_chromosome(&f.epoch, &f.adfs, &f.cl, &in_normal, &consts()).expect_err("genic in a normal gene");
        assert!(matches!(err, ChromosomeDefect::GenicInNormalGene { gene_id: 1, .. }), "{err}");
        assert!(err.to_string().starts_with("our software has a bug: "));
    }

    #[test]
    fn several_homeotic_genes_are_a_multicellular_system_with_ordered_results() {
        let f = fixture(&[], |e, _| vec![evolved_main(e, 10, vec![0, 1]), evolved_main(e, 20, vec![1])]);
        let k = f.epoch.kingdom();
        let (g0, g1) = (f.epoch.genic_terminal(0).expect("G0"), f.epoch.genic_terminal(1).expect("G1"));
        let a = node(PAIR, vec![leaf(S), leaf(I)]);
        let c = Chromosome {
            chromosome_id: 0,
            genes: vec![gene_for(k, f.cl.normal.gene(0).expect("g"), &a), gene_for(k, f.cl.normal.gene(1).expect("g"), &leaf(S)), gene_for(k, f.cl.normal.gene(2).expect("g"), &leaf(S))],
            mains: vec![main_words(&f.epoch, &f.cl.mains[0], &node(LINK, vec![leaf(g1), leaf(g0)])), main_words(&f.epoch, &f.cl.mains[1], &leaf(g1))],
        };
        let p = express_chromosome(&f.epoch, &f.adfs, &f.cl, &c, &consts()).expect("expresses");
        assert_eq!(p.inlined(&f.epoch, &f.adfs), vec![(10, node(LINK, vec![leaf(S), a])), (20, leaf(S))]);
        assert_eq!(p.mains.iter().map(|m| m.main_id).collect::<Vec<_>>(), vec![10, 20]);
        let mut short = c.clone();
        short.mains.pop();
        assert_eq!(express_chromosome(&f.epoch, &f.adfs, &f.cl, &short, &consts()).expect_err("shape"), ChromosomeDefect::ShapeMains { want: 2, got: 1 });
    }

    #[test]
    fn several_normal_genes_with_no_homeotic_gene_are_a_defect_and_one_is_its_own_program() {
        let f = fixture(&[], |_, _| vec![]);
        let k = f.epoch.kingdom();
        let genes = |t: &Tree| vec![gene_for(k, f.cl.normal.gene(0).expect("g"), t), gene_for(k, f.cl.normal.gene(1).expect("g"), t), gene_for(k, f.cl.normal.gene(2).expect("g"), &leaf(S))];
        let c = Chromosome { chromosome_id: 0, genes: genes(&leaf(S)), mains: vec![] };
        assert_eq!(express_chromosome(&f.epoch, &f.adfs, &f.cl, &c, &consts()).expect_err("no linker"), ChromosomeDefect::NoLinker { genes: 3 });
        let lone = ChromosomeLayout { normal: Layout::new(9, vec![f.cl.normal.gene(0).expect("g").clone()]), mains: vec![] };
        let e = Epoch::new(k, &lone.gene_roots(), &[]).expect("epoch");
        let c1 = Chromosome { chromosome_id: 0, genes: vec![gene_for(k, lone.normal.gene(0).expect("g"), &leaf(S))], mains: vec![] };
        let p = express_chromosome(&e, &f.adfs, &lone, &c1, &consts()).expect("one gene");
        assert_eq!(p.inlined(&e, &f.adfs), vec![(0, leaf(S))]);
    }

    /// A tree with more `S` functions than any `h_S` of the normal genes.
    fn big_body() -> Tree {
        let mut t = node(PAIR, vec![leaf(S), leaf(I)]);
        for _ in 0..9 {
            t = node(PAIR, vec![t, leaf(I)]);
        }
        t
    }

    #[test]
    fn an_adf_larger_than_every_head_is_stored_and_referenced_from_normal_and_homeotic_genes() {
        let body = big_body();
        let f = fixture(&[(7, 0, body.clone())], |e, _| vec![evolved_main(e, 0, vec![0, 1])]);
        let k = f.epoch.kingdom();
        let a7 = f.epoch.adf_terminal(7).expect("A7");
        assert_eq!(k.output(a7), Ty::S);
        assert!(crate::typed_gep::fits(k, f.cl.normal.gene(0).expect("g"), &body).is_err(), "a Karva gene of this layout would not hold the body");
        assert_eq!(f.adfs.get(7).expect("ADF").tree(), body);
        assert_eq!(f.adfs.get(7).expect("ADF").nodes.len(), 21);
        // A7 in the tail of a normal gene: link(A7, s) is not a pair, so use pair(A7, i).
        let in_gene = node(PAIR, vec![leaf(a7), leaf(I)]);
        let g0 = f.epoch.genic_terminal(0).expect("G0");
        let c = Chromosome {
            chromosome_id: 0,
            genes: vec![gene_for(k, f.cl.normal.gene(0).expect("g"), &in_gene), gene_for(k, f.cl.normal.gene(1).expect("g"), &leaf(S)), gene_for(k, f.cl.normal.gene(2).expect("g"), &leaf(S))],
            // The homeotic gene refers to the ADF as well: link(G0, A7).
            mains: vec![main_words(&f.epoch, &f.cl.mains[0], &node(LINK, vec![leaf(g0), leaf(a7)]))],
        };
        let p = express_chromosome(&f.epoch, &f.adfs, &f.cl, &c, &consts()).expect("expresses");
        let direct = node(LINK, vec![node(PAIR, vec![body.clone(), leaf(I)]), body]);
        assert_eq!(p.inlined(&f.epoch, &f.adfs), vec![(0, direct)]);
        assert_eq!(p.refs.iter().filter(|r| r.kind == RefKind::Adf { adf_id: 7 }).count(), 2, "one from the gene, one from the homeotic gene");
        assert_eq!(f.adfs.referrers(&f.epoch, 7, &[(&f.cl, &c)], &consts()).expect("count"), 2);
    }

    #[test]
    fn an_adf_with_zero_referrers_is_retirable() {
        let f = fixture(&[(1, 0, node(PAIR, vec![leaf(S), leaf(I)]))], |e, _| vec![evolved_main(e, 0, vec![0])]);
        let k = f.epoch.kingdom();
        let g0 = f.epoch.genic_terminal(0).expect("G0");
        let c = Chromosome {
            chromosome_id: 0,
            genes: (0..3).map(|g| gene_for(k, f.cl.normal.gene(g).expect("g"), &leaf(S))).collect(),
            mains: vec![main_words(&f.epoch, &f.cl.mains[0], &leaf(g0))],
        };
        assert_eq!(f.adfs.referrers(&f.epoch, 1, &[(&f.cl, &c)], &consts()).expect("count"), 0);
        assert_eq!(f.adfs.referrers(&f.epoch, 99, &[(&f.cl, &c)], &consts()).expect("count"), 0, "an unknown id has none");
    }

    #[test]
    fn adf_levels_schedule_and_defects() {
        let k = base();
        let roots = [Ty::S];
        let decls = |l0: u32, l1: u32| [AdfDecl { id: 1, root: Ty::S, level: l0 }, AdfDecl { id: 2, root: Ty::S, level: l1 }];
        // ADF 2 (level 1) = pair(A1, i), where ADF 1 (level 0) = s.
        let e = Epoch::new(&k, &roots, &decls(0, 1)).expect("epoch");
        let a1 = e.adf_terminal(1).expect("A1");
        let b1 = Adf::from_tree(&e, 1, 0, &leaf(S));
        let b2 = Adf::from_tree(&e, 2, 1, &node(PAIR, vec![leaf(a1), leaf(I)]));
        let set = AdfSet::new(&e, vec![b2.clone(), b1.clone()]).expect("lower references pass");
        assert_eq!(set.schedule(), vec![(1, 0), (2, 1)], "a definition after everything it references");
        assert_eq!(set.expanded(&e)[&2], node(PAIR, vec![leaf(S), leaf(I)]));
        // Not strictly lower: the same level, and upward.
        let same = Epoch::new(&k, &roots, &decls(1, 1)).expect("epoch");
        let err = AdfSet::new(&same, vec![Adf::from_tree(&same, 1, 1, &leaf(S)), Adf::from_tree(&same, 2, 1, &node(PAIR, vec![leaf(same.adf_terminal(1).expect("A1")), leaf(I)]))]).expect_err("same level");
        assert_eq!(err, ChromosomeDefect::AdfRefNotLower { id: 2, seq: 1, target: 1, level: 1, target_level: 1 });
        let up = Epoch::new(&k, &roots, &decls(0, 1)).expect("epoch");
        let err = AdfSet::new(&up, vec![Adf::from_tree(&up, 1, 0, &node(PAIR, vec![leaf(up.adf_terminal(2).expect("A2")), leaf(I)])), Adf::from_tree(&up, 2, 1, &leaf(S))]).expect_err("upward");
        assert!(matches!(err, ChromosomeDefect::AdfRefNotLower { id: 1, target: 2, level: 0, target_level: 1, .. }));
        let selfref = AdfSet::new(&e, vec![Adf::from_tree(&e, 1, 0, &leaf(a1)), b2.clone()]).expect_err("self");
        assert!(matches!(selfref, ChromosomeDefect::AdfRefNotLower { id: 1, target: 1, .. }));
        // A genic terminal in a body.
        let with_gene = Epoch::new(&k, &[Ty::S], &[AdfDecl { id: 1, root: Ty::S, level: 0 }]).expect("epoch");
        let g0 = with_gene.genic_terminal(0).expect("G0");
        let err = AdfSet::new(&with_gene, vec![Adf::from_tree(&with_gene, 1, 0, &node(PAIR, vec![leaf(g0), leaf(I)]))]).expect_err("genic");
        assert_eq!(err, ChromosomeDefect::AdfGenicTerminal { id: 1, seq: 1, symbol: g0 });
        // A reference of the wrong type: A1 outputs S in an I slot.
        let wrong = AdfSet::new(&e, vec![b1.clone(), Adf::from_tree(&e, 2, 1, &node(PAIR, vec![leaf(S), leaf(a1)]))]).expect_err("wrong type");
        assert_eq!(wrong, ChromosomeDefect::AdfNodeType { id: 2, seq: 2, want: Ty::I, got: Ty::S });
        // Declarations and bodies must agree.
        assert_eq!(AdfSet::new(&e, vec![b1.clone()]).expect_err("missing"), ChromosomeDefect::AdfMissingBody { id: 2 });
        assert_eq!(AdfSet::new(&e, vec![b1.clone(), b1.clone(), b2.clone()]).expect_err("twice"), ChromosomeDefect::AdfUndeclared { id: 1 });
        assert_eq!(AdfSet::new(&e, vec![Adf { level: 3, ..b1.clone() }, b2.clone()]).expect_err("level"), ChromosomeDefect::AdfDeclaration { id: 1 });
        assert_eq!(
            Epoch::new(&k, &roots, &[AdfDecl { id: 1, root: Ty::S, level: 0 }, AdfDecl { id: 1, root: Ty::I, level: 0 }]).expect_err("duplicate"),
            ChromosomeDefect::AdfDuplicate { id: 1 }
        );
        // Rows that are not a tree: an argument given twice, a missing argument, a bad parent.
        let mut twice = b2.clone();
        twice.nodes[2].arg_index = Some(0);
        assert!(matches!(AdfSet::new(&e, vec![b1.clone(), twice]), Err(ChromosomeDefect::AdfNodeParent { id: 2, seq: 2 })));
        let mut missing = b2.clone();
        missing.nodes.pop();
        assert_eq!(AdfSet::new(&e, vec![b1.clone(), missing]).expect_err("arity"), ChromosomeDefect::AdfArity { id: 2, seq: 0, want: 2, got: 1 });
        let mut empty = b1.clone();
        empty.nodes.clear();
        assert_eq!(AdfSet::new(&e, vec![empty, b2.clone()]).expect_err("empty"), ChromosomeDefect::AdfRoot { id: 1 });
        let no_index = Adf::from_tree(&e, 1, 0, &leaf(Q));
        assert_eq!(AdfSet::new(&e, vec![no_index, b2]).expect_err("constant"), ChromosomeDefect::AdfConstIndex { id: 1, seq: 0 });
    }

    #[test]
    fn a_constant_inside_an_adf_keeps_its_index_through_expansion() {
        let k = base();
        let body = node(PAIR, vec![Tree { symbol: Q, const_index: Some(3), kids: vec![] }, leaf(I)]);
        let lone = ChromosomeLayout { normal: Layout::new(9, vec![GeneLayout::from_program_size(&k, Ty::S, 2).expect("layout")]), mains: vec![] };
        let e = Epoch::new(&k, &lone.gene_roots(), &[AdfDecl { id: 4, root: Ty::S, level: 0 }]).expect("epoch");
        let set = AdfSet::new(&e, vec![Adf::from_tree(&e, 4, 0, &body)]).expect("set");
        let a4 = e.adf_terminal(4).expect("A4");
        let c = Chromosome { chromosome_id: 0, genes: vec![gene_for(e.kingdom(), lone.normal.gene(0).expect("g"), &node(PAIR, vec![leaf(a4), leaf(I)]))], mains: vec![] };
        let p = express_chromosome(&e, &set, &lone, &c, &consts()).expect("expresses");
        assert_eq!(p.inlined(&e, &set), vec![(0, node(PAIR, vec![body, leaf(I)]))]);
    }

    // -- fold, write-back, unfold ---------------------------------------------

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// A gene valid by layout over the epoch alphabet minus the genic terminals
    /// (`allow` filters them): heads from the symbols of the type.
    fn random_words(epoch: &Epoch, l: &GeneLayout, rng: &mut Rng, allow: &dyn Fn(SymbolId) -> bool) -> Vec<u32> {
        let k = epoch.kingdom();
        let mut w = vec![0u32; l.width];
        for s in l.spans() {
            let alpha: Vec<u32> = (0..k.n_symbols() as SymbolId).filter(|x| k.output(*x) == s.ty && allow(*x) && (s.domain != Domain::Tail || k.is_terminal(*x))).collect();
            for cell in &mut w[s.off..s.off + s.len] {
                *cell = if s.domain == Domain::Dc { rng.below(4) as u32 } else { alpha[rng.below(alpha.len())] };
            }
        }
        w
    }

    fn subtrees<'a>(t: &'a Tree, out: &mut Vec<&'a Tree>) {
        out.push(t);
        for c in &t.kids {
            subtrees(c, out);
        }
    }

    /// Fold, write-back, unfold is value-preserving: for random chromosomes, an
    /// ADF taken from a random subtree of one gene is folded into every gene
    /// and homeotic gene; the result expresses, holds the references, and its
    /// inlined phenotype equals the phenotype before.
    #[test]
    fn fold_write_back_unfold_preserves_the_value_on_random_chromosomes() {
        let mut rng = Rng(0xf01d);
        let (mut folded, mut refs_made) = (0usize, 0usize);
        let plain = fixture(&[], |e, _| vec![evolved_main(e, 0, vec![0, 1])]);
        for round in 0..100_000usize {
            let g_ok = |s: SymbolId| plain.epoch.reference(s).is_none();
            let genes: Vec<Vec<u32>> = (0..3).map(|g| random_words(&plain.epoch, plain.cl.normal.gene(g).expect("g"), &mut rng, &g_ok)).collect();
            let linked_ok = |s: SymbolId| plain.epoch.reference(s).is_none() || matches!(plain.epoch.reference(s), Some(RefKind::Genic { gene_id }) if gene_id < 2);
            let main = random_words(&plain.epoch, &plain.cl.mains[0].gene, &mut rng, &linked_ok);
            let c0 = Chromosome { chromosome_id: round as u64, genes, mains: vec![main] };
            let p0 = express_chromosome(&plain.epoch, &plain.adfs, &plain.cl, &c0, &consts()).unwrap_or_else(|d| panic!("round {round}: {d}"));
            // The ADF: a random non-leaf subtree of a random normal gene, taken from the expressed tree.
            let src = p0.genes[rng.below(3)].tree();
            let mut subs = Vec::new();
            subtrees(&src, &mut subs);
            let body = subs[rng.below(subs.len())].clone();
            let f = fixture(&[(1, 0, body.clone())], |e, _| vec![evolved_main(e, 0, vec![0, 1])]);
            // Same words, read in the epoch that has A1 (the alphabet only grew at the end).
            let before = express_chromosome(&f.epoch, &f.adfs, &f.cl, &c0, &consts()).unwrap_or_else(|d| panic!("round {round}: {d}")).inlined(&f.epoch, &f.adfs);
            let (c1, n) = fold_adf(&f.epoch, &f.adfs, &f.cl, &c0, 1, &consts()).unwrap_or_else(|d| panic!("round {round}: fold {d}"));
            let p1 = express_chromosome(&f.epoch, &f.adfs, &f.cl, &c1, &consts()).unwrap_or_else(|d| panic!("round {round}: folded {d}"));
            assert_eq!(p1.inlined(&f.epoch, &f.adfs), before, "round {round}: fold changed the value");
            let a1 = f.epoch.adf_terminal(1).expect("A1");
            let made = p1.refs.iter().filter(|r| r.symbol == a1 && !matches!(r.source, Source::Adf(_))).count();
            assert_eq!(made, n, "round {round}: every folded subtree is one expressed reference");
            folded += n;
            refs_made += made;
        }
        println!("fold: {folded} subtrees folded, {refs_made} references expressed, 100000 chromosomes value-preserving");
        assert!(folded > 0);
    }

    #[test]
    fn expression_with_adfs_and_homeotic_genes_is_deterministic() {
        let f = fixture(&[(2, 0, node(PAIR, vec![leaf(S), leaf(I)]))], |e, _| vec![evolved_main(e, 0, vec![0, 1])]);
        let mut rng = Rng(7);
        let ok = |s: SymbolId| f.epoch.reference(s).is_none();
        let mk = |rng: &mut Rng| Chromosome {
            chromosome_id: 0,
            genes: (0..3).map(|g| random_words(&f.epoch, f.cl.normal.gene(g).expect("g"), rng, &ok)).collect(),
            mains: vec![random_words(&f.epoch, &f.cl.mains[0].gene, rng, &|s| f.epoch.reference(s).is_none())],
        };
        for _ in 0..100 {
            let c = mk(&mut rng);
            let a = express_chromosome(&f.epoch, &f.adfs, &f.cl, &c, &consts()).expect("a");
            let b = express_chromosome(&f.epoch, &f.adfs, &f.cl, &c, &consts()).expect("b");
            assert_eq!(a.inlined(&f.epoch, &f.adfs), b.inlined(&f.epoch, &f.adfs));
            assert_eq!(a.refs, b.refs);
        }
    }
}
