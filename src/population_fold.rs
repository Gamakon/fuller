//! Population-wide shared definitions (ADFs) by structural hashing, for any
//! kingdom of the shared typed core (`nucleotable/docs/MultiTypedGEP.md` rev 4
//! section 6). Moved here from `wgsl/true_share.rs` unchanged, so the WGSL kingdom
//! and the Futhark kingdoms fold through one mechanism; `true_share` re-exports
//! it.
//!
//! The population's expressed trees are hash-consed into one DAG; a node with at
//! least two DAG edges into it, occurring in at least two individuals and holding
//! a function, is an ADF: an immutable expressed tree with a stable id
//! ([`AdfTable`], ids never reused), a root type and a level, stored as
//! `adf_node` rows ([`crate::typed_chromosome::Adf`]). Every gene holding one is
//! rewritten through the per-type write-back to hold the ADF's reference
//! terminal `A_d` (a removal-only rewrite, so it always fits), and the folded
//! gene expressed and inlined is the original tree.
//!
//! This is structural hashing only: two subtrees are shared when they are the
//! same tree. The saturated share-equivalence (equal after the kingdom's
//! e-graph rewrites) is [`crate::saturated_fold`], which uses this as its fast
//! first step and as the measured baseline.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::geneframe::Ty;
use crate::kingdom_contract::{KingdomContract, SymbolId};
use crate::typed_chromosome::{express_chromosome, Adf, AdfDecl, AdfId, AdfSet, Chromosome, ChromosomeLayout, Epoch};
use crate::typed_gep::{write_back, GeneLayout, Layout, Tree};

fn bug(what: impl std::fmt::Display) -> String {
    format!("our software has a bug: {what}")
}

/// Hash-consing of one tree: per occurrence its DAG id.
pub(crate) struct Occ {
    pub(crate) id: usize,
    pub(crate) kids: Vec<Occ>,
}

#[derive(Default)]
pub(crate) struct Cons {
    pub(crate) index: HashMap<(SymbolId, Option<u32>, Vec<usize>), usize>,
    /// Per id: symbol, the function count of the subtree, all nodes context-free.
    pub(crate) info: Vec<(SymbolId, u64, bool)>,
}

impl Cons {
    pub(crate) fn intern(&mut self, t: &Tree, functions: &dyn Fn(SymbolId) -> bool, context_free: &dyn Fn(SymbolId) -> bool) -> Occ {
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

pub(crate) fn count_nodes(t: &Tree) -> usize {
    1 + t.kids.iter().map(count_nodes).sum::<usize>()
}

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
    pub(crate) fn id_of(&mut self, body: &Tree) -> (AdfId, bool) {
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
