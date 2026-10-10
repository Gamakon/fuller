//! The input–output pairs of `docs/PLAN_wgsl_gram_superopt.md` §1b: 47
//! generated training pairs of eleven kinds (the three Blue World exports
//! are not available and are not simulated), 20 validation pairs on unseen
//! seeds, 20 extrapolation pairs at d = 8,192; every pair's f64 reference
//! cached by `(kind, d, m, seed)`.

use std::path::Path;
use std::time::Instant;

use crate::wgsl::gram::{phi_of, reference, Kind};

pub struct Pair {
    pub kind: Kind,
    pub d: usize,
    pub m: usize,
    pub seed: u64,
    pub phi: Vec<f32>,
    pub h_ref: Vec<f64>,
}

impl Pair {
    pub fn name(&self) -> String {
        format!("{}_d{}_m{}_s{}", self.kind.name(), self.d, self.m, self.seed)
    }

    pub fn work(&self) -> f64 {
        (self.d * self.d * self.m) as f64
    }

    pub fn load(kind: Kind, d: usize, m: usize, seed: u64, cache: &Path) -> Result<Pair, String> {
        let phi = phi_of(kind, d, m, seed);
        let path = cache.join(format!("ref_{}_d{d}_m{m}_s{seed}.f64", kind.name()));
        let h_ref = if path.exists() {
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            if bytes.len() != d * d * 8 {
                return Err(format!("{}: {} bytes", path.display(), bytes.len()));
            }
            bytes.chunks(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect()
        } else {
            let h = reference(&phi, d, m);
            std::fs::write(&path, h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| format!("{}: {e}", path.display()))?;
            h
        };
        Ok(Pair { kind, d, m, seed, phi, h_ref })
    }
}

pub struct Splits {
    pub train: Vec<Pair>,
    pub val: Vec<Pair>,
    pub extrap: Vec<Pair>,
    pub seconds: f64,
}

pub const COUNTS: [(Kind, usize); 11] = [(Kind::Gaussian, 6), (Kind::Weighted, 6), (Kind::LowRank, 4), (Kind::Sparse, 4), (Kind::HeavyTailed, 4), (Kind::Cancelling, 4), (Kind::Integer, 4), (Kind::Extreme, 4), (Kind::NonFinite, 4), (Kind::Duplicate, 3), (Kind::Ragged, 4)];

/// The three splits, references built on first use and read thereafter.
pub fn splits(cache: &Path) -> Result<Splits, String> {
    std::fs::create_dir_all(cache).map_err(|e| format!("{}: {e}", cache.display()))?;
    let t = Instant::now();
    let shapes = [(512, 2048), (1024, 4096), (2048, 8192), (1024, 16384)];
    let mut train = Vec::new();
    let mut seed = 100u64;
    for (kind, n) in COUNTS {
        for i in 0..n {
            let (d, m) = if kind == Kind::Ragged { [(520, 4097), (1000, 3001), (2040, 5000), (512, 6151)][i % 4] } else { shapes[i % shapes.len()] };
            train.push(Pair::load(kind, d, m, seed, cache)?);
            seed += 1;
        }
    }
    let mut val = Vec::new();
    for (i, (kind, _)) in COUNTS.iter().cycle().take(20).enumerate() {
        let (d, m) = if *kind == Kind::Ragged { (1032, 2049) } else { shapes[(i + 1) % shapes.len()] };
        val.push(Pair::load(*kind, d, m, 1000 + i as u64, cache)?);
    }
    let mut extrap = Vec::new();
    for (i, (kind, _)) in COUNTS.iter().cycle().take(20).enumerate() {
        let m = [1024, 2048, 3000, 4096][i % 4];
        extrap.push(Pair::load(*kind, 8192, m, 2000 + i as u64, cache)?);
    }
    Ok(Splits { train, val, extrap, seconds: t.elapsed().as_secs_f64() })
}
