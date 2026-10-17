//! An individual of the WGSL kingdom's population: its scaffold (the
//! source kernel read by the reader) and its chromosomes (one per
//! function: the folded head and tail genes over the function's pset).
//! phylu varies the genes as Karva tokens; this module turns tokens back
//! into a validated kernel through the existing chain: tokens →
//! `karva_to_terms_generic` → the folded chromosome's genes →
//! `rebuild_folded` → a naga module and its WGSL text, which `gram::Gram`
//! compiles. The 32 kernels of `logs/HGP/001/gen0` all round-trip this way.

use crate::karva::{karva_to_terms_generic, Token};
use crate::wgsl::chromosome::{chromosome_typed, ChromosomeOptions, WgslChromosome};
use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::reader::{read, Kernel};
use crate::wgsl::scaffold::{rebuild_folded, Rebuilt};

pub struct Individual {
    pub kernel: Kernel,
    pub chromosomes: Vec<WgslChromosome>,
    /// The source's `// grid x y bx by` line, if any: a comment, which the
    /// rebuild drops, and which the driver needs (without it a 2 x 2
    /// register-blocked kernel was dispatched with four times the
    /// workgroups and read 0.63x where its draft read 2.45x, 2026-10-10).
    pub header: Option<String>,
}

/// A gene as phylu holds it: head tokens and tail tokens.
pub type Gene = (Vec<Token>, Vec<Token>);

impl Individual {
    /// Read `source` and make its chromosomes, the same way the dump
    /// (`wgsl_read_kernel --dump`) did: typed over the kingdom, default
    /// options.
    pub fn from_source(source: &str, kingdom: &WgslKingdom) -> Result<Individual, String> {
        let kernel = read(source)?;
        let opts = ChromosomeOptions::default();
        let chromosomes = kernel.functions.iter().map(|f| chromosome_typed(f, kingdom, &opts)).collect::<Result<Vec<_>, _>>()?;
        let header = source.lines().map(str::trim).find(|l| l.starts_with("// grid ")).map(str::to_string);
        Ok(Individual { kernel, chromosomes, header })
    }

    /// The genes of function `i` as Karva tokens, head genes then tail
    /// slots, with the head length they were encoded at.
    pub fn genes(&self, i: usize) -> Result<(usize, Vec<Gene>), String> {
        let c = self.chromosomes.get(i).ok_or_else(|| format!("no function {i}"))?;
        let (head_len, genes) = c.genes.as_ref().ok_or_else(|| format!("{}: no Karva genes (a gene exceeded every head length)", c.function))?;
        Ok((*head_len, genes.clone()))
    }

    /// Rebuild the kernel with function `i`'s genes replaced by `genes`
    /// (head genes first, then the tail slots, as [`Individual::genes`]
    /// gives them): each gene decoded from its tokens over the function's
    /// pset, the folded chromosome's head and tail replaced, and the
    /// kernel rebuilt and validated. An s-expression that decodes but
    /// does not rebuild names the reason.
    pub fn rebuild_with(&self, i: usize, genes: &[Gene]) -> Result<Rebuilt, String> {
        let c = self.chromosomes.get(i).ok_or_else(|| format!("no function {i}"))?;
        let n_head = c.folded.head.len();
        let n_tail = c.folded.tail.len();
        if genes.len() != n_head + n_tail {
            return Err(format!("{}: {} genes given, the chromosome has {n_head} head and {n_tail} tail", c.function, genes.len()));
        }
        let decoded: Vec<String> = genes.iter().map(|(h, t)| karva_to_terms_generic(h, t, &c.pset)).collect::<Result<_, _>>()?;
        let mut chromosomes = self.chromosomes.clone();
        chromosomes[i].folded.head = decoded[..n_head].to_vec();
        chromosomes[i].folded.tail = decoded[n_head..].to_vec();
        self.with_header(rebuild_folded(&self.kernel, &chromosomes)?)
    }

    /// The grid header back on the rebuilt text.
    fn with_header(&self, mut r: Rebuilt) -> Result<Rebuilt, String> {
        if let Some(h) = &self.header {
            r.wgsl = format!("{h}\n{}", r.wgsl);
        }
        Ok(r)
    }

    /// Rebuild with function `i`'s genes given as decoded s-expressions in
    /// the reader's form (head genes first, then the tail slots): the path
    /// for a decoder that applies the typed projection itself (phylu's host
    /// decoder renders this form).
    pub fn rebuild_with_sexprs(&self, i: usize, decoded: &[String]) -> Result<Rebuilt, String> {
        let c = self.chromosomes.get(i).ok_or_else(|| format!("no function {i}"))?;
        let n_head = c.folded.head.len();
        let n_tail = c.folded.tail.len();
        if decoded.len() != n_head + n_tail {
            return Err(format!("{}: {} genes given, the chromosome has {n_head} head and {n_tail} tail", c.function, decoded.len()));
        }
        // A loop's break condition is a root gene. A changed condition can
        // make a loop unbounded, and an unbounded kernel occupies the GPU
        // until the machine restarts (2026-10-10, twice). Conditions belong
        // to the scaffold: a decoded condition gene must equal the draft's.
        for (g, root) in self.kernel.functions[i].roots.iter().enumerate() {
            if matches!(root.kind, crate::wgsl::reader::RootKind::Condition { .. }) && decoded.get(g) != c.folded.head.get(g) {
                return Err(format!("{}: gene {g} is a loop or branch condition and may not change (got {}, the draft has {})", c.function, decoded.get(g).map_or("", |s| s.as_str()), c.folded.head.get(g).map_or("", |s| s.as_str())));
            }
        }
        let mut chromosomes = self.chromosomes.clone();
        chromosomes[i].folded.head = decoded[..n_head].to_vec();
        chromosomes[i].folded.tail = decoded[n_head..].to_vec();
        self.with_header(rebuild_folded(&self.kernel, &chromosomes)?)
    }

    /// The kernel rebuilt from its own genes, unchanged: the round trip.
    pub fn rebuild_self(&self) -> Result<Rebuilt, String> {
        let mut chromosomes = self.chromosomes.clone();
        for (i, c) in self.chromosomes.iter().enumerate() {
            let (_, genes) = self.genes(i)?;
            let decoded: Vec<String> = genes.iter().map(|(h, t)| karva_to_terms_generic(h, t, &c.pset)).collect::<Result<_, _>>()?;
            let n_head = c.folded.head.len();
            chromosomes[i].folded.head = decoded[..n_head].to_vec();
            chromosomes[i].folded.tail = decoded[n_head..].to_vec();
        }
        self.with_header(rebuild_folded(&self.kernel, &chromosomes)?)
    }
}

/// Tokens from the dump's JSON form: `{"f": name}`, `{"v": name}`, `{"n": x}`.
pub fn tokens_from_json(v: &serde_json::Value) -> Result<Vec<Token>, String> {
    v.as_array()
        .ok_or("tokens: not an array")?
        .iter()
        .map(|t| {
            if let Some(f) = t.get("f").and_then(|x| x.as_str()) {
                Ok(Token::Func(f.to_string()))
            } else if let Some(v) = t.get("v").and_then(|x| x.as_str()) {
                Ok(Token::Var(v.to_string()))
            } else if let Some(n) = t.get("n").and_then(|x| x.as_f64()) {
                Ok(Token::Num(n))
            } else {
                Err(format!("token {t} is not f, v or n"))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    // The hand-written kernels' texts (the GPU-only gram module carries the same files).
    const SHIPPED: &str = include_str!("../../kingdoms/wgsl/gram/shipped.wgsl");
    const BLOCKED: &str = include_str!("../../kingdoms/wgsl/gram/blocked.wgsl");
    const TILED: &str = include_str!("../../kingdoms/wgsl/gram/tiled.wgsl");

    #[test]
    fn the_hand_written_kernels_round_trip_through_their_own_genes() {
        let kingdom = WgslKingdom::load();
        for (name, src) in [("shipped", SHIPPED), ("blocked", BLOCKED), ("tiled", TILED)] {
            let ind = Individual::from_source(src, &kingdom).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (head_len, genes) = ind.genes(0).unwrap();
            assert!(head_len >= 8 && !genes.is_empty(), "{name}");
            let r = ind.rebuild_self().unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(r.wgsl.contains("fn main"), "{name}");
            // The grid header survives the rebuild (the source's, or none).
            assert_eq!(r.wgsl.starts_with("// grid "), src.starts_with("// grid "), "{name}");
            if let Some(h) = &ind.header {
                assert!(r.wgsl.starts_with(h), "{name}: {}", r.wgsl.lines().next().unwrap_or(""));
            }
            // The same text from rebuild_with as from rebuild_self, and from
            // the s-expression path given the chromosome's own folded genes.
            let r2 = ind.rebuild_with(0, &genes).unwrap();
            assert_eq!(r.wgsl, r2.wgsl, "{name}");
            let own: Vec<String> = ind.chromosomes[0].folded.head.iter().chain(ind.chromosomes[0].folded.tail.iter()).cloned().collect();
            let r3 = ind.rebuild_with_sexprs(0, &own).unwrap();
            assert_eq!(r.wgsl, r3.wgsl, "{name}");
        }
    }

    /// The whole chain on a drafted individual: text → chromosomes →
    /// Karva tokens → rebuilt kernel → compiled and evaluated: feasible,
    /// with the same speed as the draft's own text.
    #[cfg(feature = "gpu")]
    #[test]
    fn a_drafted_individual_rebuilds_from_its_genes_and_evaluates_feasible_on_the_device() {
        use crate::wgsl::gram::FitnessContext;
        use crate::wgsl::oracle::device::Device;
        let kingdom = WgslKingdom::load();
        let ind = Individual::from_source(BLOCKED, &kingdom).unwrap();
        let rebuilt = ind.rebuild_self().unwrap();
        let dev = Device::new().expect("device");
        let ctx = FitnessContext::new(&dev, std::path::Path::new("logs/WGSL_GRAM/pairs")).expect("context");
        let original = ctx.fitness(&dev, BLOCKED);
        let again = ctx.fitness(&dev, &rebuilt.wgsl);
        assert!(original.feasible && again.feasible, "{original:?} {again:?}");
        assert!((original.hff - again.hff).abs() < 0.1, "{original:?} {again:?}");
    }

    /// A condition gene changed is refused by name, whatever it decodes to.
    #[test]
    fn a_changed_condition_gene_is_refused() {
        let kingdom = WgslKingdom::load();
        let ind = Individual::from_source(SHIPPED, &kingdom).unwrap();
        let roots = &ind.kernel.functions[0].roots;
        let cond = roots.iter().position(|r| matches!(r.kind, crate::wgsl::reader::RootKind::Condition { .. })).expect("the shipped kernel has a loop condition and a guard");
        let mut own: Vec<String> = ind.chromosomes[0].folded.head.iter().chain(ind.chromosomes[0].folded.tail.iter()).cloned().collect();
        own[cond] = "(literal.flag (Num 1.0))".into();
        let err = match ind.rebuild_with_sexprs(0, &own) { Ok(_) => panic!("a changed condition rebuilt"), Err(e) => e };
        assert!(err.contains("condition and may not change"), "{err}");
    }

    #[test]
    fn a_changed_gene_rebuilds_or_names_why_not() {
        let kingdom = WgslKingdom::load();
        let ind = Individual::from_source(SHIPPED, &kingdom).unwrap();
        let (_, mut genes) = ind.genes(0).unwrap();
        // Swap the first two head tokens of the first gene: a different
        // tree (or a refusal with a reason), never a panic.
        if genes[0].0.len() > 1 {
            genes[0].0.swap(0, 1);
        }
        match ind.rebuild_with(0, &genes) {
            Ok(r) => assert!(r.wgsl.contains("fn main")),
            Err(e) => assert!(!e.is_empty()),
        }
        let wrong: Vec<Gene> = genes[..1].to_vec();
        let err = match ind.rebuild_with(0, &wrong) { Ok(_) => panic!("a short gene list rebuilt"), Err(e) => e };
        assert!(err.contains("genes given"), "{err}");
    }
}
