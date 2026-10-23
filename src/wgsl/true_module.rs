//! s9 batch 8: a module of several entry points in the true WGSL kingdom (the v24 interface, `kingdoms/wgsl_true/mod/v24`).
//!
//! Each entry point is one normal gene, rooted at its entry type (`kernel.ingest`, `kernel.search`); the module is the
//! fixed homeotic linker of `nucleotable/docs/MultiTypedGEP.md` section 6 over them: head `(kernel.module_<iface>)`, its
//! tails the genic terminals of the genes in gene order, immutable (every operator's domain on it is empty). The module's
//! text is the expressed linker with each genic terminal inlined by its gene's tree, rendered by the kingdom's
//! substitution ([`TrueKingdom::render`]); each entry declares the kernel-scope variables of its own subtree.

use std::collections::BTreeMap;

use crate::geneframe::Ty;
use crate::kingdom_contract::KingdomContract;
use crate::typed_chromosome::{express_chromosome, AdfSet, Chromosome, ChromosomeLayout, Epoch, MainLayout};
use crate::typed_gep::{write_back, GeneLayout, Layout, Tree};
use crate::wgsl::true_kingdom::{random_gene, Rng, Share, TrueKingdom};

fn bug(what: String) -> String {
    format!("our software has a bug: {what}")
}

/// A module interface's chromosome layout: one normal gene per entry point and the fixed linker.
pub struct TrueModule<'k> {
    pub kingdom: &'k TrueKingdom,
    /// The kingdom's own contract (its rows only): what a normal gene's cells are drawn from.
    pub base: KingdomContract,
    /// The epoch alphabet: the kingdom's rows and one genic terminal per entry gene.
    pub epoch: Epoch,
    pub layout: ChromosomeLayout,
    pub const_len: BTreeMap<Ty, u32>,
}

impl<'k> TrueModule<'k> {
    /// Gene `g` holds every program of at most `program_sizes[g]` functions rooted at entry `g` (design section 2.1).
    pub fn new(kingdom: &'k TrueKingdom, program_sizes: &[usize]) -> Result<TrueModule<'k>, String> {
        let base = kingdom.relations().check().map_err(|d| bug(format!("the kingdom check of {}: {}", kingdom.dir.display(), d.join("; "))))?;
        let genes = kingdom
            .gene_roots
            .iter()
            .zip(program_sizes)
            .map(|(root, n)| GeneLayout::from_program_size(&base, *root, *n).map_err(|e| bug(format!("the layout of entry {}: {e}", kingdom.type_name(*root)))))
            .collect::<Result<Vec<_>, _>>()?;
        Self::with_genes(kingdom, base, genes)
    }

    /// Gene `g` with its own head lengths (design section 2.1, explicit `h_T`).
    pub fn with_heads(kingdom: &'k TrueKingdom, heads: &[BTreeMap<Ty, usize>]) -> Result<TrueModule<'k>, String> {
        let base = kingdom.relations().check().map_err(|d| bug(format!("the kingdom check of {}: {}", kingdom.dir.display(), d.join("; "))))?;
        let genes = kingdom
            .gene_roots
            .iter()
            .zip(heads)
            .map(|(root, h)| GeneLayout::from_heads(&base, *root, h).map_err(|e| bug(format!("the layout of entry {}: {e}", kingdom.type_name(*root)))))
            .collect::<Result<Vec<_>, _>>()?;
        Self::with_genes(kingdom, base, genes)
    }

    fn with_genes(kingdom: &'k TrueKingdom, base: KingdomContract, genes: Vec<GeneLayout>) -> Result<TrueModule<'k>, String> {
        if kingdom.gene_roots.is_empty() || genes.len() != kingdom.gene_roots.len() {
            return Err(bug(format!("{} is not a module interface with {} entries", kingdom.dir.display(), genes.len())));
        }
        let linker = kingdom.linker.as_ref().ok_or_else(|| bug(format!("{} names no linker", kingdom.dir.display())))?;
        let link = kingdom.rows.iter().position(|r| &r.name == linker).ok_or_else(|| bug(format!("the linker {linker} is not a row")))? as u32;
        let normal = Layout::new(0, genes);
        let mut layout = ChromosomeLayout { normal, mains: Vec::new() };
        let epoch = Epoch::new(&base, &layout.gene_roots(), &[]).map_err(|e| bug(format!("the epoch alphabet: {e}")))?;
        let linked: Vec<u64> = (0..layout.normal.n_genes() as u64).collect();
        let main: MainLayout = layout.fixed_main(&epoch, 0, link, linked).map_err(|e| bug(format!("the fixed linker: {e}")))?;
        layout.mains.push(main);
        Ok(TrueModule { kingdom, base, epoch, layout, const_len: kingdom.const_lens() })
    }

    /// The fixed linker's canonical words.
    pub fn linker_words(&self) -> Result<Vec<u32>, String> {
        self.layout.fixed_words(&self.epoch, &self.layout.mains[0]).map_err(|e| bug(format!("the fixed linker: {e}")))
    }

    /// A random chromosome: each normal gene drawn by the host typed initialiser over the kingdom's own rows (a normal
    /// gene never holds a genic terminal), the linker its canonical words.
    pub fn random(&self, chromosome_id: u64, rng: &mut Rng, share: Share, root_function: bool) -> Result<Chromosome, String> {
        let genes = self.layout.normal.genes().iter().map(|g| random_gene(&self.base, g, &self.const_len, rng, share, root_function)).collect();
        Ok(Chromosome { chromosome_id, genes, mains: vec![self.linker_words()?] })
    }

    /// A random module whose cells are drawn by row weight (`weight`, one per row of the kingdom) and per-type function
    /// share: a test that asks for a family's rows often (`true_kingdom::random_gene_weighted`).
    pub fn random_weighted(&self, chromosome_id: u64, rng: &mut Rng, weight: &[f64], function_p: &BTreeMap<Ty, f64>, root_function: bool) -> Result<Chromosome, String> {
        let genes = self.layout.normal.genes().iter().map(|g| crate::wgsl::true_kingdom::random_gene_weighted(&self.base, g, &self.const_len, rng, weight, function_p, root_function)).collect();
        Ok(Chromosome { chromosome_id, genes, mains: vec![self.linker_words()?] })
    }

    /// The expressed module: the linker's tree with every genic terminal inlined by its gene's tree.
    pub fn tree(&self, c: &Chromosome) -> Result<Tree, String> {
        let adfs = AdfSet::new(&self.epoch, Vec::new()).map_err(|e| bug(format!("the empty ADF set: {e}")))?;
        let p = express_chromosome(&self.epoch, &adfs, &self.layout, c, &self.const_len).map_err(|e| bug(format!("express: {e}")))?;
        let mut mains = p.inlined(&self.epoch, &adfs);
        if mains.len() != 1 {
            return Err(bug(format!("the module expressed {} main programs, want one", mains.len())));
        }
        Ok(mains.remove(0).1)
    }

    /// Each entry's expressed tree, in gene order.
    pub fn entry_trees(&self, c: &Chromosome) -> Result<Vec<Tree>, String> {
        let t = self.tree(c)?;
        if t.kids.len() != self.layout.normal.n_genes() {
            return Err(bug(format!("the linker has {} arguments, the module {} genes", t.kids.len(), self.layout.normal.n_genes())));
        }
        Ok(t.kids)
    }

    /// The module's WGSL.
    pub fn render(&self, c: &Chromosome) -> Result<String, String> {
        self.kingdom.render(&self.tree(c)?)
    }

    /// The linker's tree over the given entry trees (what [`TrueModule::tree`] expresses for a chromosome holding them).
    pub fn module_tree(&self, entries: Vec<Tree>) -> Result<Tree, String> {
        let linker = self.kingdom.linker.as_ref().ok_or_else(|| bug("no linker".into()))?;
        let link = self.kingdom.rows.iter().position(|r| &r.name == linker).ok_or_else(|| bug(format!("the linker {linker} is not a row")))? as u32;
        Ok(Tree { symbol: link, const_index: None, kids: entries })
    }

    /// The per-type write-back (design section 5) of each entry's tree into its gene: rows the walk does not read keep
    /// `previous`'s words. A tree that does not fit its gene's layout is a named defect.
    pub fn encode(&self, entries: &[Tree], previous: &Chromosome) -> Result<Chromosome, String> {
        if entries.len() != self.layout.normal.n_genes() {
            return Err(bug(format!("{} entry trees for {} genes", entries.len(), self.layout.normal.n_genes())));
        }
        let genes = entries
            .iter()
            .enumerate()
            .map(|(g, t)| {
                let layout = &self.layout.normal.genes()[g];
                write_back(&self.base, layout, &previous.genes[g], t).map_err(|e| format!("entry {} does not fit its gene: {e}", self.kingdom.type_name(layout.root)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Chromosome { chromosome_id: previous.chromosome_id, genes, mains: vec![self.linker_words()?] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kingdom(name: &str) -> TrueKingdom {
        TrueKingdom::load(&crate::wgsl::true_kingdom::default_dir().join("mod").join(name)).expect("the v24 table")
    }

    /// The v24 table passes the kingdom check with its three roots; the module layout is two genes under the fixed
    /// linker, whose words express `kernel.module_v24(G0, G1)`.
    #[test]
    fn the_v24_module_is_two_entry_genes_under_the_fixed_linker() {
        let k = kingdom("v24_k1");
        let m = TrueModule::new(&k, &[8, 8]).expect("module");
        assert_eq!(m.layout.normal.n_genes(), 2);
        assert_eq!(k.type_name(m.layout.normal.genes()[0].root), "kernel.ingest");
        assert_eq!(k.type_name(m.layout.normal.genes()[1].root), "kernel.search");
        let c = m.random(1, &mut Rng(7), Share::Half, true).expect("random");
        let t = m.tree(&c).expect("tree");
        assert_eq!(k.rows[t.symbol as usize].name, "kernel.module_v24_k1");
        assert_eq!(k.rows[t.kids[0].symbol as usize].name, "kernel.entry_ingest");
        assert_eq!(k.rows[t.kids[1].symbol as usize].name, "kernel.entry_search");
        let src = m.render(&c).expect("render");
        assert!(src.contains("fn ingest(") && src.contains("fn search(") && src.contains("// passes "), "{src}");
    }

    /// The UI1 batch 10 phases, drawn often: random K = 1 modules whose swap-pair, block-write and slot rows weigh 400 times the
    /// rest each render to WGSL naga validates and round-trip through the write-back; the families are expressed.
    #[test]
    fn random_modules_rich_in_the_batch_10_phases_validate_in_naga() {
        let k = kingdom("v24_k1");
        let m = TrueModule::new(&k, &[8, 8]).expect("module");
        let fam = |n: &str| n.starts_with("wg.xchg_") || n.starts_with("wg.xset_") || n.starts_with("wg.xload_") || n.starts_with("wg.bphase_") || n.starts_with("wg.bsplit_") || n.starts_with("wg.yload_") || n.starts_with("wg.bset_") || n.starts_with("wg.bload_") || n.starts_with("wg.slot_") || n.starts_with("wg.qsplit_") || n.starts_with("wg.split2_") || n.starts_with("wg.gsplit_") || n.starts_with("wg.gset_") || n.starts_with("wg.qset_") || n.starts_with("wg.qpair_") || n.starts_with("wg.qload_") || n.starts_with("wg.pload_q_") || n.starts_with("wg.sset_") || n.starts_with("wg.aload_") || n.starts_with("slot.");
        let weight: Vec<f64> = k.rows.iter().map(|r| if fam(&r.name) { 400.0 } else { 1.0 }).collect();
        let mut rng = Rng(23);
        let mut seen: BTreeMap<String, usize> = BTreeMap::new();
        for i in 0..300u64 {
            let c = m.random_weighted(i, &mut rng, &weight, &BTreeMap::new(), true).expect("random");
            let src = m.render(&c).expect("render");
            let module = crate::wgsl::naga_shape::parse_unscanned(&src).unwrap_or_else(|e| panic!("gene {i}: naga parse: {e}\n{src}"));
            crate::wgsl::target::Target::current().validate(&module, &src).unwrap_or_else(|e| panic!("gene {i}: naga validate: {e}\n{src}"));
            let t = m.tree(&c).expect("tree");
            for (n, _) in t.level_order() {
                let name = &k.rows[n.symbol as usize].name;
                if fam(name) {
                    *seen.entry(name.split('_').next().unwrap_or(name).to_string()).or_default() += 1;
                }
            }
            let trees = m.entry_trees(&c).expect("trees");
            let back = m.encode(&trees, &m.random(1000 + i, &mut rng, Share::Half, true).expect("random")).expect("encode");
            assert_eq!(m.tree(&back).expect("tree"), t, "gene {i}");
        }
        println!("batch 10 rows expressed: {seen:?}");
        for f in ["wg.xchg", "wg.xset", "wg.xload", "wg.bphase", "wg.bset", "wg.bload", "wg.slot", "wg.sset", "wg.aload", "slot.take", "wg.qsplit", "wg.qset", "wg.qpair", "wg.qload", "wg.gsplit", "wg.gset"] {
            assert!(seen.get(f).copied().unwrap_or(0) > 0, "{f} never expressed in 300 modules: {seen:?}");
        }
    }

    /// Every random module at K = 1 renders to WGSL naga reads and validates for the installed target, through the
    /// write-back: the entry trees encoded into a fresh chromosome express the same module.
    #[test]
    fn random_v24_modules_validate_in_naga_and_round_trip_through_the_write_back() {
        let k = kingdom("v24_k1");
        let m = TrueModule::new(&k, &[8, 8]).expect("module");
        let mut rng = Rng(11);
        // `UI1_RANDOM_N=1000`: the host half of UI1's 1,000-module loop (naga and the write-back; the device half is
        // the example `v24_true_read --random`)
        let n: u64 = std::env::var("UI1_RANDOM_N").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
        for i in 0..n {
            let c = m.random(i, &mut rng, Share::Half, true).expect("random");
            let src = m.render(&c).expect("render");
            let module = crate::wgsl::naga_shape::parse_unscanned(&src).unwrap_or_else(|e| panic!("gene {i}: naga parse: {e}\n{src}"));
            crate::wgsl::target::Target::current().validate(&module, &src).unwrap_or_else(|e| panic!("gene {i}: naga validate: {e}\n{src}"));
            let trees = m.entry_trees(&c).expect("trees");
            let fresh = m.random(1000 + i, &mut rng, Share::Half, true).expect("random");
            let back = m.encode(&trees, &fresh).expect("encode");
            assert_eq!(m.tree(&back).expect("tree"), m.tree(&c).expect("tree"), "gene {i}");
        }
    }
}
