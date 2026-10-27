//! What a shape-kingdom program costs **on the GPU**, measured from the
//! compiled GPU program rather than modelled from the source.
//!
//! # Why this exists
//!
//! The static cost in [`crate::futhark_shapes_cost`] is a *pre-optimisation*
//! model: it prices the program as the search writes it, before Futhark's
//! own passes. That is the right thing for a cheap filter, but it is not
//! what the machine sees, and the two can disagree — measured: scaling an
//! operand before a product is *cheaper* in the model (it touches n·m
//! elements instead of n·p) and *more expensive* on the GPU (the extra
//! array is materialised, so the GPU allocates 4648 bytes where the plain
//! product allocates 4560).
//!
//! So the score that matters is measured from the **GPU program**: the
//! ImpGPU that `futhark dev --gpu-mem --compile-imp-gpu` prints after the
//! whole GPU pipeline has run. From it this module counts:
//!
//! * the **device memory** the program allocates, and how many
//!   allocations;
//! * the **shared memory** the tiler asks for (the block+register tiler's
//!   footprint);
//! * the number of **kernels**;
//! * and whether the **tiler fired** at all.
//!
//! That last one settles the question the plan raised — "may the cost be
//! credited to a form the tiler still fires on?" — by *measuring* it
//! rather than guessing from the tree. The tiler is Futhark's own
//! `tileLoops` and the block+register tiler, and nothing in the kingdom's
//! vocabulary can un-nest the helper's reduction, so a candidate that
//! keeps the helper keeps the tiling.
//!
//! # The cost
//!
//! This is expensive — the GPU pipeline is seconds per program — so it is
//! for the **elites**, not for every child. The cheap polynomial filters
//! the population; this decides.
//!
//! The comparison is lexicographic: fewer device bytes, then fewer device
//! allocations, then fewer shared allocations, then fewer kernels.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use crate::futhark_exec::output_with_timeout;

/// How long the GPU pipeline may take before it is killed.
const GPU_TIMEOUT: Duration = Duration::from_secs(120);

/// What a program costs on the GPU, measured from its compiled GPU program.
///
/// The comparison is lexicographic, and the **order matters**: the number
/// of device allocations comes first, because it is the one figure that is
/// always exact. The byte total comes after it, because the ImpGPU holds
/// several alternative versions of the same work, chosen at run time by
/// thresholds, and a malloc whose size is computed cannot be added up
/// statically — so the byte total is a lower bound, and the count of the
/// unknown ones is reported beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GpuCost {
    /// The number of device allocation sites.
    pub device_allocs: usize,
    /// The number of device allocations whose size is computed at run time,
    /// and so cannot be added up statically.
    pub computed_allocs: usize,
    /// The total of the allocation sizes that are literal, in bytes. A lower
    /// bound when `computed_allocs` is not zero.
    pub device_bytes: u64,
    /// The number of shared-memory allocations — the tiler's footprint.
    pub shared_allocs: usize,
    /// The number of kernels in the compiled GPU program.
    pub kernels: usize,
}

/// The cost, and whether the tiler fired.
///
/// The tiler's answer is not part of the ordering — it is a *gate*: a
/// candidate is only credited with a cost when the compiler still tiles it
/// the way it tiles the reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMeasurement {
    /// The measured cost.
    pub cost: GpuCost,
    /// Whether the block+register tiler fired.
    pub tiled: bool,
}

impl GpuMeasurement {
    /// Whether the block+register tiler fired: the compiled program asks
    /// for a `reg_tile_size`, which only the tiler sets.
    pub fn is_tiled(&self) -> bool {
        self.tiled
    }
}

/// The cache: the same program always costs the same, and the pipeline is
/// slow, so the measurement is remembered.
fn cache() -> &'static Mutex<BTreeMap<String, GpuMeasurement>> {
    static CACHE: OnceLock<Mutex<BTreeMap<String, GpuMeasurement>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// The cost of a program on the GPU, measured from its compiled GPU
/// program.
///
/// The source is the whole Futhark program — the helper, the kernel and
/// the entry point — exactly as the engine renders it.
pub fn measure(source: &str, dir: &Path) -> Result<GpuMeasurement, String> {
    if let Ok(cache) = cache().lock() {
        if let Some(hit) = cache.get(source) {
            return Ok(*hit);
        }
    }
    let measurement = measure_uncached(source, dir)?;
    if let Ok(mut cache) = cache().lock() {
        cache.insert(source.to_string(), measurement);
    }
    Ok(measurement)
}

/// The measurement, without the cache.
fn measure_uncached(source: &str, dir: &Path) -> Result<GpuMeasurement, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let fut = dir.join("cost.fut");
    std::fs::write(&fut, source).map_err(|e| format!("write: {e}"))?;
    let mut cmd = std::process::Command::new("futhark");
    cmd.arg("dev")
        .arg("--gpu-mem")
        .arg("--compile-imp-gpu")
        .arg("cost.fut")
        .current_dir(dir);
    let out = output_with_timeout(&mut cmd, GPU_TIMEOUT, None)
        .map_err(|e| format!("the GPU pipeline did not finish: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "the GPU pipeline could not compile the program:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    parse_impgpu(&String::from_utf8_lossy(&out.stdout))
}

/// Read the cost out of an ImpGPU program.
///
/// The ImpGPU is the compiled GPU program: the kernels, the device
/// allocations and the shared-memory allocations.
///
/// Three things are counted carefully, because the obvious reading of each
/// is wrong:
///
/// * **The kernels** are the `kernel {` blocks. The `entry {` blocks are
///   the *functions*, of which there are two whatever the program does.
/// * **The tiler** is detected by a kernel *using* `reg_tile_size`, which
///   only the block+register tiler sets. The file's header declares
///   `reg_tile_size` as a tuning parameter whatever the program is, so
///   matching the name alone would say "tiled" for everything.
/// * **The allocations** are every `malloc(...)@device` site. A site whose
///   size is computed at run time still counts as an allocation, but its
///   bytes are unknown, and the count of those is reported.
fn parse_impgpu(text: &str) -> Result<GpuMeasurement, String> {
    let mut device_allocs = 0usize;
    let mut computed_allocs = 0usize;
    let mut device_bytes = 0u64;
    let mut shared_allocs = 0usize;
    let mut kernels = 0usize;
    let mut tiled = false;

    // The brace depth, so that a kernel's body can be recognised.
    let mut depth = 0usize;
    // The depth at which the current kernel started, if one has.
    let mut kernel_depth: Option<usize> = None;

    for raw in text.lines() {
        let line = raw.trim();
        if kernel_depth.is_none() && line.starts_with("kernel") && line.ends_with('{') {
            kernels += 1;
            kernel_depth = Some(depth);
        } else if kernel_depth.is_some() && line.contains("reg_tile_size") {
            tiled = true;
        }
        if line.contains("shared_alloc") {
            shared_allocs += 1;
        }
        // A device allocation: `name <- malloc(4096i64)@device`.
        if let Some(rest) = line.split("malloc(").nth(1) {
            if rest.contains("@device") {
                device_allocs += 1;
                let size: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                match size.parse::<u64>() {
                    Ok(bytes) => device_bytes += bytes,
                    Err(_) => computed_allocs += 1,
                }
            }
        }
        let opens = line.chars().filter(|c| *c == '{').count();
        let closes = line.chars().filter(|c| *c == '}').count();
        depth += opens;
        depth = depth.saturating_sub(closes);
        if let Some(k) = kernel_depth {
            if depth <= k {
                kernel_depth = None;
            }
        }
    }

    Ok(GpuMeasurement {
        cost: GpuCost {
            device_allocs,
            computed_allocs,
            device_bytes,
            shared_allocs,
            kernels,
        },
        tiled,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_impgpu_is_read_for_its_cost() {
        // A kernel, a device allocation and a shared allocation.
        let text = "\
function entry_main {
  entry {
    kernel  {
      counters_mem_9100 <- malloc(4096i64)@device
      color_9011 = shared_alloc(bytes_8943)
      Rx_8366 <- get_size(main.Rx_8364, reg_tile_size)
    }
  }
  in {counters_mem_9100}
}
";
        let m = parse_impgpu(text).expect("the ImpGPU parses");
        assert_eq!(m.cost.device_bytes, 4096);
        assert_eq!(m.cost.device_allocs, 1);
        assert_eq!(m.cost.computed_allocs, 0);
        assert_eq!(m.cost.shared_allocs, 1);
        assert_eq!(m.cost.kernels, 1);
        assert!(m.is_tiled(), "the tiler fired");
    }

    /// A kernel that only uses the ordinary `tile_size` is not the
    /// block+register tiler, and a header that merely *declares*
    /// `reg_tile_size` is not a tiled kernel either.
    #[test]
    fn only_a_kernel_that_uses_the_register_tile_counts_as_tiled() {
        let text = "\
tuning params {
  reg_tile_size main.Rx_8364 (main)
  tile_size main.Tx_8363 (main)
}
function entry_main {
  entry {
    kernel  {
      Tx_8365 <- get_size(main.Tx_8363, tile_size)
    }
  }
  in {Tx_8365}
}
";
        let m = parse_impgpu(text).expect("the ImpGPU parses");
        assert!(!m.is_tiled(), "the block+register tiler did not fire");
        assert_eq!(m.cost.kernels, 1, "the kernel is counted");
    }

    /// A malloc whose size is computed at run time still counts as an
    /// allocation, but its bytes are unknown.
    #[test]
    fn a_computed_allocation_is_counted_but_not_summed() {
        let text = "\
function entry_main {
  entry {
    kernel  {
      segred_tmp_mem <- malloc(mul_nw64 (4i64) (num_virtblocks))@device
    }
  }
  in {segred_tmp_mem}
}
";
        let m = parse_impgpu(text).expect("the ImpGPU parses");
        assert_eq!(m.cost.device_allocs, 1, "the allocation is counted");
        assert_eq!(m.cost.computed_allocs, 1, "its size is unknown");
        assert_eq!(m.cost.device_bytes, 0, "and so no bytes are claimed");
    }

    #[test]
    fn an_untiled_program_is_reported_as_untiled() {
        let text = "\
function entry_main {
  entry {
    counters_mem_9100 <- malloc(16i64)@device
  }
  in {counters_mem_9100}
}
";
        let m = parse_impgpu(text).expect("the ImpGPU parses");
        assert!(!m.is_tiled(), "the tiler did not fire");
        assert_eq!(m.cost.device_bytes, 16);
    }
}
