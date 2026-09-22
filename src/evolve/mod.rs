//! The evolution engine on the device — plan: `docs/PLAN_engine_on_gpu.md`.
//!
//! A GEP population is fixed-length rows of integer tokens, so every operator
//! is indexing. This module holds the layout every kernel reads, the
//! counter-based random generator, and — for each kernel — a CPU reference in
//! plain Rust that the device is checked against bit for bit.
//!
//! Step 1 (here): the resident population, the generator, initialisation.

#[cfg(feature = "gpu")]
pub mod device;
#[cfg(feature = "gpu")]
pub mod engine;
pub mod genealogy;
#[cfg(feature = "gpu")]
pub mod score;
pub mod smogd;
pub mod smote;
pub mod umap2d;
pub mod vary;
#[cfg(feature = "gpu")]
pub mod write_back;

/// mix64.wgsl (the generator, copied verbatim from the qdrant workspace) in
/// front of the kernels that draw from it.
pub const EVOLVE_WGSL: &str = concat!(include_str!("mix64.wgsl"), include_str!("evolve.wgsl"));
/// The generator in front of the selection and variation kernels (step 2).
pub const VARY_WGSL: &str = concat!(include_str!("mix64.wgsl"), include_str!("vary.wgsl"));

/// The shape of a population. A gene is `head + tail + Dc`, geppy's own layout,
/// and the Dc domain is as long as the tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub pop: u32,
    pub n_genes: u32,
    pub head: u32,
    pub tail: u32,
    pub n_rnc: u32,
}

impl Layout {
    /// `tail = head * (max_arity - 1) + 1`: long enough to close any head.
    pub fn for_arity(pop: u32, n_genes: u32, head: u32, max_arity: u32, n_rnc: u32) -> Layout {
        Layout { pop, n_genes, head, tail: head * (max_arity.max(1) - 1) + 1, n_rnc }
    }

    pub fn gene_width(&self) -> u32 {
        self.head + 2 * self.tail
    }

    pub fn genome_len(&self) -> usize {
        (self.pop * self.n_genes * self.gene_width()) as usize
    }

    pub fn rnc_len(&self) -> usize {
        (self.pop * self.n_genes * self.n_rnc) as usize
    }
}

/// What a kernel needs to know about the symbols: each id's arity, and the two
/// lists initialisation and uniform mutation may DRAW from. A terminal withheld
/// from sampling (a named constant, graftable by snap) has an id and an arity
/// but is in neither list.
#[derive(Clone, Debug)]
pub struct SymbolCodes {
    pub arity: Vec<u32>,
    pub sample_functions: Vec<u32>,
    pub sample_terminals: Vec<u32>,
    /// The id of "?", the random-constant placeholder, if the set has one: the
    /// n-th "?" of a gene's expression reads `rnc[dc[n]]`. The cleansing
    /// mutation needs it to keep each surviving "?" on its own constant, and to
    /// collapse a subtree into a constant.
    pub rnc_id: Option<u32>,
}

impl SymbolCodes {
    pub fn validate(&self) -> Result<(), String> {
        if self.sample_functions.is_empty() || self.sample_terminals.is_empty() {
            return Err("a population needs at least one function and one terminal to draw".into());
        }
        let n = self.arity.len() as u32;
        for &f in &self.sample_functions {
            if f >= n || self.arity[f as usize] == 0 {
                return Err(format!("sample_functions holds {f}, which is not a function id"));
            }
        }
        for &t in &self.sample_terminals {
            if t >= n || self.arity[t as usize] != 0 {
                return Err(format!("sample_terminals holds {t}, which is not a terminal id"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InitParams {
    pub seed: u32,
    pub generation: u32,
    pub rnc_lo: i32,
    pub rnc_hi: i32,
    pub n_wrappers: u32,
    /// The VIRTUAL HEAD: only the first `vhead` positions of a head may hold a
    /// function; the rest of the head holds terminals, as the tail does. 0 = the
    /// whole head, the ordinary GEP gene. See [`virtual_head`].
    pub vhead: u32,
}

/// The head length the operators work with: `vhead`, or the whole head when it is
/// 0. The genome keeps its physical width whatever this is — a gene with a short
/// virtual head is an ordinary gene whose later head positions happen to hold
/// terminals, so it decodes as it always did, and RAISING the virtual head changes
/// no expression: it only opens the next position to a function.
pub fn virtual_head(vhead: u32, layout: Layout) -> Result<u32, String> {
    match vhead {
        0 => Ok(layout.head),
        v if (2..=layout.head).contains(&v) => Ok(v),
        v => Err(format!("virtual head {v} is not in 2..={}", layout.head)),
    }
}

/// A population on the host: what `device::EvolveDevice::read` returns and what
/// the CPU reference produces.
#[derive(Clone, Debug, PartialEq)]
pub struct Population {
    pub layout: Layout,
    pub genome: Vec<u32>,
    pub rnc: Vec<f32>,
    pub wrapper_id: Vec<u32>,
}

// ---------------------------------------------------------------------------
// The generator. Counter-based: a draw is a pure function of where it is used,
// so there is no state to advance and no dependence on thread order.
// ---------------------------------------------------------------------------

pub const STREAM_KIND: u32 = 1;
pub const STREAM_SYMBOL: u32 = 2;
pub const STREAM_DC: u32 = 3;
pub const STREAM_RNC: u32 = 4;
pub const STREAM_WRAPPER: u32 = 5;

/// SHA-256("fuller_evolve_v1")[:8], big-endian — this engine's draw domain, by
/// the workspace's convention (rp-formula, udv23-umap).
pub const DOMAIN_EVOLVE: u64 = 0x6B86_E41A_79C1_9B14;

/// The SplitMix64 finaliser (Stafford mix13): the workspace's pinned recipe
/// (qdrant `lib/rp-formula`, `lib/udv23-umap/.../umap/rng.rs`). Do not change a
/// constant or the order: `mix64_matches_the_workspace_pinned_vectors` holds it.
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

pub fn mix64_3(domain: u64, a: u64, b: u64, c: u64) -> u64 {
    mix64(mix64(mix64(domain ^ a) ^ b) ^ c)
}

/// One draw: a pure function of where it is used. The domain carries the fit's
/// seed (low word) and the stream (high word).
pub fn draw(seed: u32, generation: u32, row: u32, slot: u32, stream: u32) -> u64 {
    let domain = DOMAIN_EVOLVE ^ u64::from(seed) ^ (u64::from(stream) << 32);
    mix64_3(domain, u64::from(generation), u64::from(row), u64::from(slot))
}

/// A draw mapped onto `0 .. n-1` by the Lemire multiply-shift (no low-bit bias).
pub fn below(h: u64, n: u32) -> u32 {
    ((u128::from(h) * u128::from(n)) >> 64) as u32
}

/// A fair coin: the hash's top bit.
pub fn coin(h: u64) -> bool {
    h >> 63 == 1
}

// ---------------------------------------------------------------------------
// CPU reference: initialisation. geppy's rule — a head slot is a function or a
// terminal with equal odds, a tail slot a terminal, a Dc slot an index into the
// gene's constants, a constant an integer in [rnc_lo, rnc_hi].
// ---------------------------------------------------------------------------

pub fn init(layout: Layout, codes: &SymbolCodes, p: &InitParams) -> Result<Population, String> {
    codes.validate()?;
    if p.rnc_hi < p.rnc_lo || p.n_wrappers == 0 || layout.n_rnc == 0 {
        return Err("init: need rnc_lo <= rnc_hi, n_wrappers > 0 and n_rnc > 0".into());
    }
    let (nf, nt) = (codes.sample_functions.len() as u32, codes.sample_terminals.len() as u32);
    let span = (p.rnc_hi - p.rnc_lo + 1) as u32;
    let width = layout.gene_width();
    let mut genome = vec![0u32; layout.genome_len()];
    let mut rnc = vec![0f32; layout.rnc_len()];
    let mut wrapper_id = vec![0u32; layout.pop as usize];
    let vhead = virtual_head(p.vhead, layout)?;
    let d = |row, slot, stream| draw(p.seed, p.generation, row, slot, stream);
    for row in 0..layout.pop {
        for g in 0..layout.n_genes {
            let base = ((row * layout.n_genes + g) * width) as usize;
            for pos in 0..width {
                let slot = g * width + pos;
                let terminal = codes.sample_terminals[below(d(row, slot, STREAM_SYMBOL), nt) as usize];
                genome[base + pos as usize] = if pos < vhead {
                    if coin(d(row, slot, STREAM_KIND)) {
                        codes.sample_functions[below(d(row, slot, STREAM_SYMBOL), nf) as usize]
                    } else {
                        terminal
                    }
                } else if pos < layout.head + layout.tail {
                    terminal
                } else {
                    below(d(row, slot, STREAM_DC), layout.n_rnc)
                };
            }
            let rbase = ((row * layout.n_genes + g) * layout.n_rnc) as usize;
            for k in 0..layout.n_rnc {
                let v = p.rnc_lo + below(d(row, g * layout.n_rnc + k, STREAM_RNC), span) as i32;
                rnc[rbase + k as usize] = v as f32;
            }
        }
        wrapper_id[row as usize] = below(d(row, 0, STREAM_WRAPPER), p.n_wrappers);
    }
    Ok(Population { layout, genome, rnc, wrapper_id })
}

impl Population {
    /// The structural rules every GEP population keeps. `Err` names the first
    /// one broken.
    pub fn check(&self, codes: &SymbolCodes) -> Result<(), String> {
        let l = self.layout;
        let width = l.gene_width() as usize;
        for (i, gene) in self.genome.chunks(width).enumerate() {
            let (head_tail, dc) = gene.split_at((l.head + l.tail) as usize);
            if let Some(&id) = head_tail.iter().find(|&&id| id as usize >= codes.arity.len()) {
                return Err(format!("gene {i}: id {id} is outside the symbol table"));
            }
            if let Some(&id) = head_tail[l.head as usize..].iter().find(|&&id| codes.arity[id as usize] != 0) {
                return Err(format!("gene {i}: the tail holds function {id}"));
            }
            if let Some(&k) = dc.iter().find(|&&k| k >= l.n_rnc) {
                return Err(format!("gene {i}: Dc index {k} is outside the gene's {} constants", l.n_rnc));
            }
        }
        if self.rnc.iter().any(|v| !v.is_finite()) {
            return Err("a constant is not finite".into());
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// ids 0..4 functions (arity 2,2,2,1), 4..8 terminals; 7 is withheld.
    pub(crate) fn codes() -> SymbolCodes {
        SymbolCodes {
            arity: vec![2, 2, 2, 1, 0, 0, 0, 0],
            sample_functions: vec![0, 1, 2, 3],
            sample_terminals: vec![4, 5, 6],
            rnc_id: Some(6),
        }
    }

    pub(crate) fn params(seed: u32) -> InitParams {
        InitParams { seed, generation: 0, rnc_lo: -100, rnc_hi: 100, n_wrappers: 3, vhead: 0 }
    }

    #[test]
    fn a_new_population_keeps_the_rules_and_never_draws_a_withheld_terminal() {
        let layout = Layout::for_arity(500, 3, 48, 2, 10);
        let pop = init(layout, &codes(), &params(7)).unwrap();
        pop.check(&codes()).unwrap();
        let width = layout.gene_width() as usize;
        let symbols = (layout.head + layout.tail) as usize;
        assert!(pop.genome.chunks(width).all(|g| g[..symbols].iter().all(|&id| id != 7)));
        assert!(pop.rnc.iter().all(|v| (-100.0..=100.0).contains(v) && v.fract() == 0.0));
        assert!(pop.wrapper_id.iter().all(|&w| w < 3));
    }

    #[test]
    fn the_same_seed_is_the_same_population_and_another_seed_is_not() {
        let layout = Layout::for_arity(64, 3, 16, 2, 10);
        let a = init(layout, &codes(), &params(1)).unwrap();
        assert_eq!(a, init(layout, &codes(), &params(1)).unwrap());
        assert_ne!(a.genome, init(layout, &codes(), &params(2)).unwrap().genome);
    }

    /// rp-formula's known-answer table (seed, p, d, h), derived by an
    /// independent longhand implementation in the qdrant workspace:
    /// mix64_3(RADEMACHER_DOMAIN_V1, seed, p, d) must reproduce every h.
    #[test]
    fn mix64_matches_the_workspace_pinned_vectors() {
        const RADEMACHER_DOMAIN_V1: u64 = 0x6C1C_8DC3_E04C_9F0A;
        let pinned: [(u64, u64, u64, u64); 9] = [
            (0x0000_0000_0000_0000, 0, 0, 0xE7AD_6C0B_BC6D_D8CB),
            (0x0000_0000_0000_0000, 0, 999_999_999, 0x6A94_75DD_400B_4F65),
            (0x0000_0000_0000_0000, 63, 0, 0xA725_BE0E_EFDE_F80F),
            (0x0000_0000_0000_0000, 1023, 123_456_789, 0x3BD9_252B_5AC4_7D51),
            (0xFFFF_FFFF_FFFF_FFFF, 0, 0, 0xBFC1_D3D0_BD60_7648),
            (0xFFFF_FFFF_FFFF_FFFF, 511, 999_999_999, 0xE3F0_6326_640C_A734),
            (0x0000_0000_0000_002A, 7, 1, 0x71C8_3BCE_AF92_83EC),
            (0x0000_0000_DEAD_BEEF, 100_000, 649_999_999, 0xED0E_F039_0AB0_C0C4),
            (0x0000_0000_0000_0001, 0, 1, 0xB2A5_0EAF_07E7_2EC1),
        ];
        for (seed, p, d, want) in pinned {
            assert_eq!(mix64_3(RADEMACHER_DOMAIN_V1, seed, p, d), want, "seed {seed:#X} p {p} d {d}");
        }
    }

    #[test]
    fn draws_cover_their_range_evenly() {
        let mut seen = [0u32; 7];
        for slot in 0..70_000 {
            seen[below(draw(3, 0, 0, slot, STREAM_SYMBOL), 7) as usize] += 1;
        }
        assert!(seen.iter().all(|&n| (9_500..10_500).contains(&n)), "{seen:?}");
        let heads = (0..10_000).filter(|&s| coin(draw(3, 0, 0, s, STREAM_KIND))).count();
        assert!((4_700..5_300).contains(&heads), "{heads}");
    }

    #[test]
    fn bad_symbol_lists_are_refused() {
        let mut c = codes();
        c.sample_terminals.push(0); // a function id among the terminals
        assert!(c.validate().is_err());
        let mut c = codes();
        c.sample_functions.clear();
        assert!(c.validate().is_err());
    }
}
