//! The Futhark kingdom's tables: the Futhark kingdom's views of four kinds of row ([`Kingdom6`], `kingdom6`), and the
//! gene RNG. (The kingdoms whose types were duals of a `types.tsv`, the 1-D Futhark kingdom and the pixel family's
//! old tables, are deleted: plan 7.7 K4, K5. The pixel family is rows of the master tables, [`crate::futhark_pixel::rows`].)
//!
//! The genetic representation is the Karva gene. The Futhark *source* is a
//! rendering of the expressed tree -- regenerated on every evaluation, never
//! read back into the genome.

/// A splitmix64: deterministic, no dependency, and the same sequence every
/// run (so a failure is reproducible).
///
/// It lives here, not in an example, because a *reported* failure has to be
/// reproducible: the gene that made the compiler hang must be regenerable
/// from its seed alone, by the rescue experiment as well as by the sweep that
/// found it.
#[derive(Debug, Clone)]
pub struct GeneRng(u64);

impl GeneRng {
    /// A generator with the given seed.
    pub fn new(seed: u64) -> GeneRng {
        GeneRng(seed)
    }

    /// The generator's state: `GeneRng::new(rng.state())` continues the sequence exactly (a checkpoint keeps it).
    pub fn state(&self) -> u64 {
        self.0
    }

    /// The next number in the sequence.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

// ---- the Futhark kingdom's views: a kingdom of four kinds of row (`kingdom6`) ----------------------------------------

mod kingdom6;
pub use kingdom6::{Kingdom6, Program6};

#[cfg(test)]
mod tests6;

