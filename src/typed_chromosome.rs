//! Multigenic chromosomes and homeotic definitions, host side
//! (`nucleotable/docs/MultiTypedGEP.md` section 6).
//!
//! A chromosome is an ordered list of genes, each with an explicit `gene_id` and
//! root type, joined by a typed linker symbol whose ordered signature is the
//! gene roots (for two `R` genes, `link:R(R,R)`). It is not a bag of evaluated
//! genes.
//!
//! A homeotic definition has a stable immutable id, an assigned level and a
//! declared root type. A reference to it is an ordinary terminal of that root
//! type, one per definition, which names the id and never a position. A
//! definition at level `L` may reference only definitions at lower levels; the
//! strict decrease proves there is no cycle. A reference that is missing, not
//! lower, or of a type other than its definition's root is a
//! [`ChromosomeDefect`], not a fallback opportunity.

use std::collections::BTreeMap;
use std::fmt;

use crate::geneframe::Ty;
use crate::kingdom_contract::{KingdomContract, SymbolId};
use crate::typed_gep::{express, Expressed, GeneLayout, LayoutDefect};

/// A homeotic definition's stable id; never a position.
pub type DefId = u64;

/// One gene of a chromosome or one definition's body: its boundary, root and words.
#[derive(Debug, Clone)]
pub struct GeneRecord {
    pub gene_id: u64,
    pub root: Ty,
    pub layout: GeneLayout,
    pub words: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct Definition {
    pub id: DefId,
    pub level: u32,
    pub body: GeneRecord,
}

/// The definitions visible to chromosomes. Shared and immutable while a
/// chromosome is varied: a chromosome only borrows it.
#[derive(Debug, Clone)]
pub struct Definitions {
    defs: BTreeMap<DefId, Definition>,
    /// The reference terminal of each definition.
    refs: BTreeMap<SymbolId, DefId>,
}

/// Where a reference was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Gene(u64),
    Definition(DefId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChromosomeDefect {
    Gene { gene_id: u64, defect: LayoutDefect },
    Definition { id: DefId, defect: LayoutDefect },
    /// A gene record whose root differs from its layout's.
    RootMismatch { gene_id: u64, record: Ty, layout: Ty },
    /// More than one gene and no linker.
    NoLinker { genes: usize },
    LinkerNotAFunction { symbol: SymbolId },
    /// The linker's ordered signature is not the gene roots.
    LinkerSignature { want: Vec<Ty>, got: Vec<Ty> },
    /// A reference terminal names a definition that does not exist.
    RefMissing { symbol: SymbolId, def: DefId },
    RefNotATerminal { symbol: SymbolId },
    /// Two reference terminals name one definition.
    RefDuplicate { def: DefId },
    /// The terminal's output type is not the definition's declared root.
    RefType { symbol: SymbolId, def: DefId, symbol_ty: Ty, root: Ty },
    /// A reference from level `from_level` (or a gene) to a definition not strictly lower.
    RefNotLower { source: Source, def: DefId, from_level: Option<u32>, to_level: u32 },
}

impl fmt::Display for ChromosomeDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "our software has a bug: ")?;
        match self {
            ChromosomeDefect::Gene { gene_id, defect } => write!(f, "gene {gene_id}: {defect}"),
            ChromosomeDefect::Definition { id, defect } => write!(f, "definition {id}: {defect}"),
            ChromosomeDefect::RootMismatch { gene_id, record, layout } => write!(f, "gene {gene_id} has root {record:?} and a layout rooted at {layout:?}"),
            ChromosomeDefect::NoLinker { genes } => write!(f, "a chromosome of {genes} genes has no linker symbol"),
            ChromosomeDefect::LinkerNotAFunction { symbol } => write!(f, "linker symbol {symbol} is a terminal"),
            ChromosomeDefect::LinkerSignature { want, got } => write!(f, "the linker takes {got:?}, the gene roots are {want:?}"),
            ChromosomeDefect::RefMissing { symbol, def } => write!(f, "reference terminal {symbol} names definition {def}, which does not exist"),
            ChromosomeDefect::RefNotATerminal { symbol } => write!(f, "reference symbol {symbol} takes arguments"),
            ChromosomeDefect::RefDuplicate { def } => write!(f, "two reference terminals name definition {def}"),
            ChromosomeDefect::RefType { symbol, def, symbol_ty, root } => write!(f, "reference terminal {symbol} outputs {symbol_ty:?}, definition {def} is rooted at {root:?}"),
            ChromosomeDefect::RefNotLower { source, def, from_level, to_level } => {
                write!(f, "{source:?} (level {from_level:?}) references definition {def} at level {to_level}, which is not lower")
            }
        }
    }
}

/// A chromosome: ordered genes and the linker over their roots.
#[derive(Debug, Clone)]
pub struct Chromosome {
    pub chromosome_id: u64,
    pub genes: Vec<GeneRecord>,
    pub linker: Option<SymbolId>,
}

/// One reference, as expressed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefUse {
    pub source: Source,
    pub symbol: SymbolId,
    pub step: usize,
    pub def: DefId,
}

/// The expressed chromosome.
#[derive(Debug, Clone)]
pub struct Phenotype {
    /// Each gene's expression, in gene order.
    pub genes: Vec<Expressed>,
    /// Every definition's expression in level order (ties by id), the schedule
    /// of execution: a definition after everything it references.
    pub definitions: Vec<(DefId, u32, Expressed)>,
    pub refs: Vec<RefUse>,
    pub linker: Option<SymbolId>,
    /// The type of the phenotype: the linker's output, or the one gene's root.
    pub ty: Ty,
}

/// An expression as a tree of symbols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tree {
    pub symbol: SymbolId,
    pub const_index: Option<u32>,
    pub kids: Vec<Tree>,
}

impl Expressed {
    pub fn tree(&self) -> Tree {
        let mut kids: Vec<Vec<(u32, usize)>> = vec![Vec::new(); self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            if let (Some(p), Some(a)) = (n.parent_step, n.arg_index) {
                kids[p - 1].push((a, i));
            }
        }
        fn build(e: &Expressed, kids: &[Vec<(u32, usize)>], i: usize) -> Tree {
            let mut ks = kids[i].clone();
            ks.sort_unstable();
            Tree { symbol: e.nodes[i].symbol, const_index: e.nodes[i].const_index, kids: ks.into_iter().map(|(_, c)| build(e, kids, c)).collect() }
        }
        build(self, &kids, 0)
    }
}

impl Definitions {
    /// Check the set: each reference terminal is a terminal of the declared root
    /// type naming a definition that exists, one terminal per definition.
    pub fn new(k: &KingdomContract, defs: Vec<Definition>, refs: BTreeMap<SymbolId, DefId>) -> Result<Definitions, ChromosomeDefect> {
        let defs: BTreeMap<DefId, Definition> = defs.into_iter().map(|d| (d.id, d)).collect();
        let mut named: BTreeMap<DefId, SymbolId> = BTreeMap::new();
        for (sym, id) in &refs {
            if !k.is_terminal(*sym) {
                return Err(ChromosomeDefect::RefNotATerminal { symbol: *sym });
            }
            let def = defs.get(id).ok_or(ChromosomeDefect::RefMissing { symbol: *sym, def: *id })?;
            if k.output(*sym) != def.body.root {
                return Err(ChromosomeDefect::RefType { symbol: *sym, def: *id, symbol_ty: k.output(*sym), root: def.body.root });
            }
            if named.insert(*id, *sym).is_some() {
                return Err(ChromosomeDefect::RefDuplicate { def: *id });
            }
        }
        Ok(Definitions { defs, refs })
    }

    pub fn get(&self, id: DefId) -> Option<&Definition> {
        self.defs.get(&id)
    }

    /// The definition a symbol references, if it is a reference terminal.
    pub fn reference(&self, symbol: SymbolId) -> Option<DefId> {
        self.refs.get(&symbol).copied()
    }
}

fn record_check(r: &GeneRecord) -> Result<(), ChromosomeDefect> {
    if r.root != r.layout.root {
        return Err(ChromosomeDefect::RootMismatch { gene_id: r.gene_id, record: r.root, layout: r.layout.root });
    }
    Ok(())
}

/// The references an expression makes, in level order.
fn refs_of(defs: &Definitions, e: &Expressed, source: Source) -> Vec<RefUse> {
    e.nodes.iter().filter_map(|n| defs.reference(n.symbol).map(|def| RefUse { source, symbol: n.symbol, step: n.step, def })).collect()
}

/// Express a chromosome against the definitions: every definition (level
/// order, the execution schedule), every gene, the linker's signature, and
/// every reference checked against the level rule.
pub fn express_chromosome(k: &KingdomContract, defs: &Definitions, c: &Chromosome, const_len: &BTreeMap<Ty, u32>) -> Result<Phenotype, ChromosomeDefect> {
    let mut order: Vec<&Definition> = defs.defs.values().collect();
    order.sort_by_key(|d| (d.level, d.id));
    let mut refs = Vec::new();
    let mut definitions = Vec::new();
    for d in order {
        record_check(&d.body)?;
        let e = express(k, &d.body.layout, &d.body.words, const_len).map_err(|defect| ChromosomeDefect::Definition { id: d.id, defect })?;
        for r in refs_of(defs, &e, Source::Definition(d.id)) {
            let to = defs.get(r.def).ok_or(ChromosomeDefect::RefMissing { symbol: r.symbol, def: r.def })?;
            if to.level >= d.level {
                return Err(ChromosomeDefect::RefNotLower { source: r.source, def: r.def, from_level: Some(d.level), to_level: to.level });
            }
            refs.push(r);
        }
        definitions.push((d.id, d.level, e));
    }
    let mut genes = Vec::new();
    for g in &c.genes {
        record_check(g)?;
        let e = express(k, &g.layout, &g.words, const_len).map_err(|defect| ChromosomeDefect::Gene { gene_id: g.gene_id, defect })?;
        refs.extend(refs_of(defs, &e, Source::Gene(g.gene_id)));
        genes.push(e);
    }
    let roots: Vec<Ty> = c.genes.iter().map(|g| g.root).collect();
    let ty = match (c.linker, roots.as_slice()) {
        (Some(l), _) => {
            if k.is_terminal(l) {
                return Err(ChromosomeDefect::LinkerNotAFunction { symbol: l });
            }
            if k.signature(l) != roots.as_slice() {
                return Err(ChromosomeDefect::LinkerSignature { want: roots, got: k.signature(l).to_vec() });
            }
            k.output(l)
        }
        (None, [one]) => *one,
        (None, _) => return Err(ChromosomeDefect::NoLinker { genes: roots.len() }),
    };
    Ok(Phenotype { genes, definitions, refs, linker: c.linker, ty })
}

impl Phenotype {
    /// The phenotype as one tree with every reference replaced by its
    /// definition's body: what the chromosome computes with the shared
    /// definitions expanded.
    pub fn inlined(&self, defs: &Definitions) -> Tree {
        let bodies: BTreeMap<DefId, Tree> = {
            let mut m: BTreeMap<DefId, Tree> = BTreeMap::new();
            for (id, _, e) in &self.definitions {
                let t = expand(&e.tree(), defs, &m);
                m.insert(*id, t);
            }
            m
        };
        let genes: Vec<Tree> = self.genes.iter().map(|g| expand(&g.tree(), defs, &bodies)).collect();
        match self.linker {
            Some(l) => Tree { symbol: l, const_index: None, kids: genes },
            None => genes.into_iter().next().expect("one gene without a linker"),
        }
    }
}

fn expand(t: &Tree, defs: &Definitions, bodies: &BTreeMap<DefId, Tree>) -> Tree {
    if let Some(body) = defs.reference(t.symbol).and_then(|id| bodies.get(&id)) {
        return body.clone();
    }
    Tree { symbol: t.symbol, const_index: t.const_index, kids: t.kids.iter().map(|c| expand(c, defs, bodies)).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kingdom_contract::relations_from_rows;

    const PAIR: SymbolId = 0;
    const LINK: SymbolId = 1;
    const S: SymbolId = 2;
    const I: SymbolId = 3;
    const HREF_D0: SymbolId = 4; // outputs I
    const HREF_D1: SymbolId = 5; // outputs S
    const HREF_BAD: SymbolId = 6; // outputs S, no definition

    fn kingdom() -> KingdomContract {
        relations_from_rows(
            "h",
            vec![Ty::S, Ty::I],
            &[
                ("pair".to_string(), vec![Ty::S, Ty::I], Ty::S),
                ("link".to_string(), vec![Ty::S, Ty::S], Ty::S),
                ("s".to_string(), vec![], Ty::S),
                ("i".to_string(), vec![], Ty::I),
                ("href_d0".to_string(), vec![], Ty::I),
                ("href_d1".to_string(), vec![], Ty::S),
                ("href_bad".to_string(), vec![], Ty::S),
            ],
            &[],
        )
        .check()
        .expect("valid")
    }

    fn record(k: &KingdomContract, gene_id: u64, root: Ty, h: usize, cells: &[u32]) -> GeneRecord {
        let layout = GeneLayout::new(k, h, root).expect("layout");
        assert_eq!(cells.len(), layout.width, "fixture width");
        GeneRecord { gene_id, root, layout, words: cells.to_vec() }
    }

    /// d0 (level 0, root I) = i; d1 (level 1, root S) = pair(s, href_d0).
    fn defs(k: &KingdomContract) -> Definitions {
        let d0 = Definition { id: 10, level: 0, body: record(k, 0, Ty::I, 1, &[I, I, I, S, S]) };
        let d1 = Definition { id: 11, level: 1, body: record(k, 0, Ty::S, 1, &[PAIR, S, S, HREF_D0]) };
        Definitions::new(k, vec![d1, d0], BTreeMap::from([(HREF_D0, 10), (HREF_D1, 11)])).expect("definitions")
    }

    fn gene_with_href(k: &KingdomContract, gene_id: u64) -> GeneRecord {
        // pair(href_d1, i)
        record(k, gene_id, Ty::S, 1, &[PAIR, HREF_D1, S, I])
    }

    fn leaf(symbol: SymbolId) -> Tree {
        Tree { symbol, const_index: None, kids: vec![] }
    }
    fn node(symbol: SymbolId, kids: Vec<Tree>) -> Tree {
        Tree { symbol, const_index: None, kids }
    }

    #[test]
    fn two_genes_under_a_typed_linker() {
        let k = kingdom();
        let d = Definitions::new(&k, vec![], BTreeMap::new()).expect("no definitions");
        let g0 = record(&k, 0, Ty::S, 1, &[PAIR, S, S, I]);
        let g1 = record(&k, 1, Ty::S, 1, &[S, S, S, I]);
        let c = Chromosome { chromosome_id: 3, genes: vec![g0, g1], linker: Some(LINK) };
        let p = express_chromosome(&k, &d, &c, &BTreeMap::new()).expect("expresses");
        assert_eq!(p.ty, Ty::S);
        assert_eq!(p.inlined(&d), node(LINK, vec![node(PAIR, vec![leaf(S), leaf(I)]), leaf(S)]));
        for (g, e) in c.genes.iter().zip(&p.genes) {
            assert_eq!(e, &express(&k, &g.layout, &g.words, &BTreeMap::new()).expect("alone"), "a gene expresses the same alone");
        }
    }

    #[test]
    fn linker_defects_are_reported() {
        let k = kingdom();
        let d = Definitions::new(&k, vec![], BTreeMap::new()).expect("none");
        let g = |id| record(&k, id, Ty::S, 1, &[S, S, S, I]);
        let none = Chromosome { chromosome_id: 0, genes: vec![g(0), g(1)], linker: None };
        assert_eq!(express_chromosome(&k, &d, &none, &BTreeMap::new()).expect_err("no linker"), ChromosomeDefect::NoLinker { genes: 2 });
        let three = Chromosome { chromosome_id: 0, genes: vec![g(0), g(1), g(2)], linker: Some(LINK) };
        assert!(matches!(express_chromosome(&k, &d, &three, &BTreeMap::new()), Err(ChromosomeDefect::LinkerSignature { .. })));
        let mixed = Chromosome { chromosome_id: 0, genes: vec![g(0), record(&k, 1, Ty::I, 1, &[I, I, I, S, S])], linker: Some(LINK) };
        assert_eq!(
            express_chromosome(&k, &d, &mixed, &BTreeMap::new()).expect_err("wrong root"),
            ChromosomeDefect::LinkerSignature { want: vec![Ty::S, Ty::I], got: vec![Ty::S, Ty::S] }
        );
        let term = Chromosome { chromosome_id: 0, genes: vec![g(0)], linker: Some(S) };
        assert_eq!(express_chromosome(&k, &d, &term, &BTreeMap::new()).expect_err("terminal"), ChromosomeDefect::LinkerNotAFunction { symbol: S });
        let one = Chromosome { chromosome_id: 0, genes: vec![g(0)], linker: None };
        assert_eq!(express_chromosome(&k, &d, &one, &BTreeMap::new()).expect("one gene").ty, Ty::S);
    }

    #[test]
    fn lower_level_references_express_in_level_order_and_inline_to_the_direct_gene() {
        let k = kingdom();
        let d = defs(&k);
        let c = Chromosome { chromosome_id: 1, genes: vec![gene_with_href(&k, 0)], linker: None };
        let p = express_chromosome(&k, &d, &c, &BTreeMap::new()).expect("expresses");
        assert_eq!(p.definitions.iter().map(|(id, l, _)| (*id, *l)).collect::<Vec<_>>(), vec![(10, 0), (11, 1)], "a definition after everything it references");
        assert_eq!(
            p.refs,
            vec![
                RefUse { source: Source::Definition(11), symbol: HREF_D0, step: 3, def: 10 },
                RefUse { source: Source::Gene(0), symbol: HREF_D1, step: 2, def: 11 },
            ]
        );
        // The same expression written directly: pair(pair(s, i), i).
        let direct = record(&k, 0, Ty::S, 2, &[PAIR, PAIR, S, S, S, I, I]);
        let direct = Chromosome { chromosome_id: 1, genes: vec![direct], linker: None };
        let dp = express_chromosome(&k, &Definitions::new(&k, vec![], BTreeMap::new()).expect("none"), &direct, &BTreeMap::new()).expect("direct");
        let want = node(PAIR, vec![node(PAIR, vec![leaf(S), leaf(I)]), leaf(I)]);
        assert_eq!(dp.inlined(&d), want);
        assert_eq!(p.inlined(&d), want, "the chromosome with shared definitions computes what the direct gene does");
    }

    #[test]
    fn references_that_are_not_strictly_lower_are_defects() {
        let k = kingdom();
        // d1 at level 1 referencing itself.
        let me = Definition { id: 11, level: 1, body: record(&k, 0, Ty::S, 1, &[PAIR, HREF_D1, S, I]) };
        let d = Definitions::new(&k, vec![me], BTreeMap::from([(HREF_D1, 11)])).expect("set");
        let c = Chromosome { chromosome_id: 0, genes: vec![record(&k, 0, Ty::S, 1, &[S, S, S, I])], linker: None };
        assert!(matches!(express_chromosome(&k, &d, &c, &BTreeMap::new()), Err(ChromosomeDefect::RefNotLower { from_level: Some(1), to_level: 1, .. })));
        // d0 at level 0 referencing d1 at level 1: a reference upward.
        let d0 = Definition { id: 10, level: 0, body: record(&k, 0, Ty::I, 1, &[I, I, I, S, S]) };
        let up = Definition { id: 11, level: 1, body: record(&k, 0, Ty::S, 1, &[S, S, S, I]) };
        let low = Definition { id: 12, level: 0, body: record(&k, 0, Ty::S, 1, &[PAIR, HREF_D1, S, I]) };
        let d = Definitions::new(&k, vec![d0, up, low], BTreeMap::from([(HREF_D1, 11)])).expect("set");
        let err = express_chromosome(&k, &d, &c, &BTreeMap::new()).expect_err("upward");
        assert_eq!(err, ChromosomeDefect::RefNotLower { source: Source::Definition(12), def: 11, from_level: Some(0), to_level: 1 });
        assert!(err.to_string().starts_with("our software has a bug: "));
    }

    #[test]
    fn missing_mistyped_and_duplicate_references_are_defects() {
        let k = kingdom();
        let d0 = || Definition { id: 10, level: 0, body: record(&k, 0, Ty::I, 1, &[I, I, I, S, S]) };
        assert_eq!(
            Definitions::new(&k, vec![d0()], BTreeMap::from([(HREF_BAD, 99)])).expect_err("missing"),
            ChromosomeDefect::RefMissing { symbol: HREF_BAD, def: 99 }
        );
        assert_eq!(
            Definitions::new(&k, vec![d0()], BTreeMap::from([(HREF_D1, 10)])).expect_err("an S terminal for an I definition"),
            ChromosomeDefect::RefType { symbol: HREF_D1, def: 10, symbol_ty: Ty::S, root: Ty::I }
        );
        assert_eq!(
            Definitions::new(&k, vec![d0()], BTreeMap::from([(PAIR, 10)])).expect_err("a function"),
            ChromosomeDefect::RefNotATerminal { symbol: PAIR }
        );
        let d0s = Definition { id: 10, level: 0, body: record(&k, 0, Ty::S, 1, &[S, S, S, I]) };
        assert_eq!(
            Definitions::new(&k, vec![d0s], BTreeMap::from([(HREF_D1, 10), (HREF_BAD, 10)])).expect_err("duplicate"),
            ChromosomeDefect::RefDuplicate { def: 10 }
        );
    }

    #[test]
    fn a_gene_defect_names_its_gene_and_a_root_mismatch_is_reported() {
        let k = kingdom();
        let d = Definitions::new(&k, vec![], BTreeMap::new()).expect("none");
        let mut bad = record(&k, 7, Ty::S, 1, &[PAIR, S, S, I]);
        bad.words[1] = I;
        let c = Chromosome { chromosome_id: 0, genes: vec![bad], linker: None };
        assert!(matches!(express_chromosome(&k, &d, &c, &BTreeMap::new()), Err(ChromosomeDefect::Gene { gene_id: 7, .. })));
        let mut lie = record(&k, 8, Ty::S, 1, &[S, S, S, I]);
        lie.root = Ty::I;
        let c = Chromosome { chromosome_id: 0, genes: vec![lie], linker: None };
        assert_eq!(
            express_chromosome(&k, &d, &c, &BTreeMap::new()).expect_err("root"),
            ChromosomeDefect::RootMismatch { gene_id: 8, record: Ty::I, layout: Ty::S }
        );
    }

    #[test]
    fn expression_with_definitions_is_deterministic() {
        let k = kingdom();
        let d = defs(&k);
        let c = Chromosome { chromosome_id: 1, genes: vec![gene_with_href(&k, 0), gene_with_href(&k, 1)], linker: Some(LINK) };
        let a = express_chromosome(&k, &d, &c, &BTreeMap::new()).expect("a");
        let b = express_chromosome(&k, &d, &c, &BTreeMap::new()).expect("b");
        assert_eq!(a.inlined(&d), b.inlined(&d));
        assert_eq!(a.refs, b.refs);
    }
}
