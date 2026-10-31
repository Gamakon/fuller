//! The work guard: no submit of a kernel that is not proven may exceed a bound on its node executions.
//!
//! 2026-10-21 17:33 the machine crashed (WindowServer watchdog, kernel panic) while `examples/wgsl_true_loop.rs`
//! ran random genes of the true WGSL kingdom at fuel budget K = 2. A random kernel at budget K is bounded only by
//! `H * 2^(6K)` node executions per invocation (H the sum of the head lengths over the reachable types), and the
//! dispatch runs it in every invocation. On this laptop the GPU also drives the screen, and a process killed after
//! its submit does not stop the work already submitted.
//!
//! The bound of one submit is `H * 2^(6K) * invocations`, `invocations` the dispatch's workgroup count times the
//! workgroup size, summed over every dispatch the submit holds. [`work_bound`] computes it; [`Fuel::check`] requires
//! it to be at most [`CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT`] BEFORE the submit. A bound over the cap is a
//! misconfigured experiment (the kingdom's budget or the dispatch is too large for one submit), not a verdict on
//! the kernel: the message names the bound, the cap and what to change.
//!
//! An expressed tree of the true kingdom has its own exact bound ([`Fuel::Tree`], `TrueKingdom::tree_work_bound`: the
//! sum over its nodes of the product of the trip caps of the loops enclosing each), far below `H * 2^(6K)`; the image
//! harness splits a dispatch into submits of whole workgroups ([`Fuel::workgroups_per_submit`]) so each submit's bound
//! is at most the cap.

/// Node executions one submit may be bounded by, on this machine (Apple M3 Max, the laptop that also drives the
/// display). Basis: the evaluator throughput measured earlier, about 17 G node evaluations per second (phylu tsr
/// speed report), times 0.1 s (a submit stays well under the second the display can wait), divided by a safety
/// factor of 10 for a kernel slower than the evaluator: 17e9 * 0.1 / 10 = 1.7e8.
pub const CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT: u128 = 170_000_000;

/// A7's bound of ONE WORKGROUP at the k-means timing input: UltraDim's `update_centroids` read into the true kingdom
/// (`problems/kmeans_update_segmented/gather.wgsl`, its `mean` entry), 256 invocations at n = 8,192
/// (`TrueKingdom::tree_work_bound_workgroup_in`; `true_reader::tests::kmeans_timing_bounds_fit_the_measured_cap`
/// recomputes it). The basis of [`CAP_M3_MAX_MEASURED_NODE_EXECUTIONS_PER_SUBMIT`].
pub const A7_TIMING_WORKGROUP_BOUND: u128 = 125_841_920;

/// THE MEASURED CAP, for the kernels of a harness problem whose yardstick was timed on this machine (Apple M3 Max,
/// Metal). The machine's cap above assumes 17 G node executions a second; A7 itself, timed 2026-10-28 10:18:55Z
/// (`wgsl_true_case_fit`, timing case n8192_d256_k1024: 1,024 workgroups of [`A7_TIMING_WORKGROUP_BOUND`] each) ran
/// in 1,054.9 GPU ms, so it was bounded by 1,024 x A7_TIMING_WORKGROUP_BOUND / 1.0549 s = 1.22e11 node executions a
/// second, 7.2 times the machine cap's assumed 17e9 (each of its 1,024 submits held one workgroup and took about 1 ms
/// of the 100 ms a submit may take). The cap keeps the machine cap's own rule, a submit bounded by a tenth of a second
/// at the measured rate over a safety factor of 10: `rate * 0.1 / 10` = 1.22e9. The scatter form's `mean` entry
/// (171,993,344 a workgroup at the timing input, 1 % over the machine's cap, so it was never timed) fits it 7 times. A runner that uses it checks, before its run, that its own yardstick's
/// measured rate still gives at least this cap (`measured_cap_holds`). The per-submit GPU wait bound
/// (`Budget::gpu_wait_ms`) stays the hard stop.
pub const CAP_M3_MAX_MEASURED_NODE_EXECUTIONS_PER_SUBMIT: u128 = A7_TIMING_WORKGROUP_BOUND * 1024 * 10_000 / 10_549 / 100;

/// Whether a kernel bounded by `bound` node executions in all and timed at `gpu_ms` GPU milliseconds justifies `cap`:
/// at its measured rate a submit of `cap` takes at most a hundredth of a second (the tenth of a second over the safety
/// factor of 10). Returns the measured rate (node executions a second) and the verdict.
pub fn measured_cap_holds(bound: u128, gpu_ms: f64, cap: u128) -> (f64, bool) {
    let rate = bound as f64 / (gpu_ms / 1e3);
    (rate, gpu_ms > 0.0 && cap as f64 <= rate * 0.1 / 10.0)
}

/// `H * 2^(6K) * invocations`, saturating at `u128::MAX`.
pub fn work_bound(h: u64, k: u32, invocations: u64) -> u128 {
    let shift = 6u32.saturating_mul(k);
    let fuel = if shift >= 127 { u128::MAX } else { 1u128 << shift };
    u128::from(h).saturating_mul(fuel).saturating_mul(u128::from(invocations))
}

/// What a device run knows about the kernel it submits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fuel {
    /// A kernel proven by hand or by measurement (the shipped kernel, a hand kernel of the oracle, a machinery
    /// kernel): no bound is computed. A random gene, an evolved child or a seed is never proven.
    /// The name says which kernel and why it is proven.
    Proven(&'static str),
    /// A caller of the legacy kingdom, whose candidates have no fuel budget: nothing runs until it is ported to
    /// `kingdoms/wgsl_true`.
    Unported,
    /// A kernel of the true WGSL kingdom: `h` the sum of the head lengths over the reachable types, `k` the
    /// interface's fuel budget.
    Kingdom { h: u64, k: u32 },
    /// One expressed tree of the true kingdom: `per_invocation` is the exact bound its types give
    /// (`TrueKingdom::tree_work_bound`: the sum over the expressed nodes of the product of the trip caps of the loops
    /// enclosing each), never more than the kingdom's `H * 2^(6K)`.
    Tree { per_invocation: u128 },
}

/// The message of every legacy caller.
pub const LEGACY_NO_FUEL: &str = "legacy kingdom: no fuel budget; port to kingdoms/wgsl_true";

impl Fuel {
    /// An error, before any device work, for a caller that has not been ported.
    pub fn refuse_unported(&self) -> Result<(), String> {
        if *self == Fuel::Unported {
            Err(LEGACY_NO_FUEL.to_string())
        } else {
            Ok(())
        }
    }

    /// The bound of one invocation, or `None` for a proven kernel.
    pub fn per_invocation(&self) -> Option<u128> {
        match self {
            Fuel::Proven(_) => None,
            Fuel::Unported => Some(u128::MAX),
            Fuel::Kingdom { h, k } => Some(work_bound(*h, *k, 1)),
            Fuel::Tree { per_invocation } => Some(*per_invocation),
        }
    }

    /// The bound of a submit of `invocations`, or `None` for a proven kernel.
    pub fn bound(&self, invocations: u64) -> Option<u128> {
        self.per_invocation().map(|p| p.saturating_mul(u128::from(invocations)))
    }

    /// How many whole workgroups of `workgroup_invocations` one submit may hold under `cap`: `None` for a proven kernel
    /// (one submit holds the dispatch), else at least one. A tree whose single workgroup is bounded above the cap cannot
    /// be split into whole workgroups: the error names the bound and the cap, and nothing is submitted.
    pub fn workgroups_per_submit(&self, workgroup_invocations: u64, cap: u128) -> Result<Option<u64>, String> {
        self.refuse_unported()?;
        let Some(per) = self.per_invocation() else { return Ok(None) };
        let one = per.saturating_mul(u128::from(workgroup_invocations.max(1)));
        if one > cap {
            return Err(format!(
                "one workgroup ({workgroup_invocations} invocations) of this kernel is bounded by {one} node executions ({per} an invocation), over the per-submit cap {cap}: whole workgroups cannot split it; nothing was submitted"
            ));
        }
        Ok(Some(u64::try_from((cap / one.max(1)).min(u128::from(u64::MAX))).unwrap_or(u64::MAX)))
    }

    /// The guard: the submit of `what` (`invocations` in all its dispatches) may start only if its bound is at
    /// most the cap. Returns the bound (0 for a proven kernel).
    pub fn check(&self, what: &str, invocations: u64) -> Result<u128, String> {
        self.check_under(what, invocations, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)
    }

    /// [`Fuel::check`] against an explicit cap (at most the machine's; a test lowers it to force a split).
    pub fn check_under(&self, what: &str, invocations: u64, cap: u128) -> Result<u128, String> {
        self.refuse_unported()?;
        let cap = cap.min(CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT);
        if let Fuel::Tree { per_invocation } = self {
            let bound = per_invocation.saturating_mul(u128::from(invocations));
            if bound <= cap {
                return Ok(bound);
            }
            return Err(format!("{what}: the expressed tree is bounded by {per_invocation} node executions an invocation, {bound} for {invocations} invocations in one submit, over the cap {cap}. Nothing was submitted; the dispatch must be split into smaller submits."));
        }
        let Fuel::Kingdom { h, k } = self else { return Ok(0) };
        let bound = work_bound(*h, *k, invocations);
        if bound <= cap {
            return Ok(bound);
        }
        Err(format!(
            "the experiment is misconfigured, not the kernel: {what} is bounded by H * 2^(6K) * invocations = {h} * 2^{} * {invocations} = {bound} node executions in one submit, over the cap {cap} (M3 Max, about 0.1 s at the measured 17 G/s with a safety factor of 10). Nothing was submitted. Lower the fuel budget K (now {k}), use a smaller dispatch, or split the work into several submits.",
            6 * k
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bound_is_h_times_two_to_the_6k_times_invocations() {
        assert_eq!(work_bound(10, 0, 100), 1000);
        assert_eq!(work_bound(10, 1, 100), 10 * 64 * 100);
        assert_eq!(work_bound(3, 2, 5), 3 * 4096 * 5);
        assert_eq!(work_bound(0, 2, 5), 0);
        assert_eq!(work_bound(7, 11, 1), 7 * (1u128 << 66));
        assert_eq!(work_bound(7, 100, 1000), u128::MAX, "saturates, never wraps");
    }

    #[test]
    fn a_bound_over_the_cap_stops_the_run_and_names_the_bound_the_cap_and_the_remedy() {
        let ok = Fuel::Kingdom { h: 100, k: 0 };
        assert_eq!(ok.check("a run", 1_000_000).expect("under the cap"), 100_000_000);
        let over = Fuel::Kingdom { h: 100, k: 0 };
        let e = over.check("a run", 2_000_000).expect_err("200 M is over 170 M");
        assert!(e.contains("200000000") && e.contains("170000000") && e.contains("Lower the fuel budget K") && e.contains("smaller dispatch") && e.contains("several submits"), "{e}");
        assert!(Fuel::Kingdom { h: 1, k: 3 }.check("a run", 1_000_000_000).is_err());
        assert_eq!(Fuel::Proven("a test").check("a run", u64::MAX).expect("proven"), 0);
        assert_eq!(Fuel::Proven("a test").bound(5), None);
    }

    /// A tree's own bound: the submit of `invocations` is bounded by `per_invocation * invocations`; whole workgroups per
    /// submit are the largest count under the cap, at least one; a single workgroup over the cap is an error, never a
    /// submit; a proven kernel is one submit.
    #[test]
    fn a_tree_bound_splits_into_whole_workgroups_under_the_cap() {
        let t = Fuel::Tree { per_invocation: 35_000 };
        assert_eq!(t.bound(64), Some(35_000 * 64));
        assert_eq!(t.check("a run", 64).expect("one workgroup"), 2_240_000);
        assert!(t.check("a run", 65_536).is_err(), "a 256 x 256 picture in one submit is over the cap");
        assert_eq!(t.workgroups_per_submit(64, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT).expect("split"), Some(170_000_000 / 2_240_000));
        assert_eq!(t.workgroups_per_submit(64, 2_240_000).expect("exactly one"), Some(1));
        let e = t.workgroups_per_submit(64, 2_239_999).expect_err("one workgroup over the cap");
        assert!(e.contains("2240000") && e.contains("whole workgroups cannot split it"), "{e}");
        assert_eq!(Fuel::Proven("a test").workgroups_per_submit(64, 1).expect("proven"), None);
        assert!(Fuel::Unported.workgroups_per_submit(64, 1).is_err());
        assert!(t.check_under("a run", 64, 1_000_000).is_err(), "a lowered cap is obeyed");
        assert!(t.check_under("a run", 64, u128::MAX).is_ok(), "a cap above the machine's is the machine's");
        assert!(Fuel::Tree { per_invocation: 1 }.check_under("a run", 200_000_000, u128::MAX).is_err(), "never above the machine's cap");
    }

    /// The text of the function starting at `marker` up to the next function at the same indentation.
    fn body<'a>(src: &'a str, marker: &str) -> &'a str {
        let at = src.find(marker).unwrap_or_else(|| panic!("{marker} is not in the source"));
        let rest = &src[at + marker.len()..];
        let end = rest.find("\n    pub fn ").into_iter().chain(rest.find("\n    fn ")).min().unwrap_or(rest.len());
        &rest[..end]
    }

    /// Every device-run path that can take an unproven kernel calls the guard before it submits. The device
    /// cannot be opened in a host-only test, so the source of each path is held to contain the guard call ahead
    /// of its first submit.
    #[test]
    fn every_device_run_path_calls_the_guard_before_its_first_submit() {
        let gram = include_str!("gram.rs");
        for marker in ["pub fn run(&self, r: &Resident, read_h: bool)", "pub fn run_batch(&self, rs: &[&Resident])"] {
            let b = body(gram, marker);
            let (g, s) = (b.find("self.fuel.check(").unwrap_or_else(|| panic!("{marker}: no guard")), b.find("queue.submit(").expect("a submit"));
            assert!(g < s, "{marker}: the guard comes after the submit");
        }
        let harness = include_str!("harness.rs");
        let b = body(harness, "fn exec<'p>(&self, k: &DeviceKernel, live: &mut [Live<'p>]");
        let (g, s) = (b.find("self.fuel.check(").expect("harness exec: no guard"), b.find("pass.dispatch_workgroups(").expect("a dispatch"));
        assert!(g < s, "harness exec: the guard comes after the dispatch is encoded");
        // The image harness (ST9): the guard is checked, with the dispatch's invocation count, before anything is encoded.
        let image = include_str!("image.rs");
        let b = body(image, "pub fn passes(&self, warmup: usize, repeats: usize)");
        let (g, s) = (b.find("self.fuel.check_under(").expect("image run: no guard"), b.find("queue.submit(").expect("a submit"));
        assert!(g < s, "image run: the guard comes after the submit");
        // every submit of the tiled pass is preceded by the guard on its own tile, inside the same loop
        let tile_loop = &b[b.find("for (c, tile) in self.tiles.iter().enumerate()").expect("the tile loop")..];
        let (g, s) = (tile_loop.find("self.fuel.check_under(").expect("no guard in the tile loop"), tile_loop.find("queue.submit(").expect("a submit"));
        assert!(g < s, "image run: a tile is submitted before its guard");
        let example = include_str!("../../examples/wgsl_true_loop.rs");
        assert!(example.contains("Gram::compile(dev, src, self.fuel.clone())") && example.contains("fuel.check(\"stage (c) at the unit-test dispatch\"") && example.contains("self.fuel.check(\"stage (c)\""), "wgsl_true_loop stage (c)");
        assert!(example.find("fuel.check(\"stage (c) at the unit-test dispatch\"") < example.find("Device::new()"), "the guard comes before the device is opened");
    }

    #[test]
    fn the_cap_is_the_measured_rate_over_a_tenth_of_a_second_over_ten() {
        assert_eq!(CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT, 17_000_000_000 / 10 / 10);
    }

    /// The measured cap is A7's measured rate over a tenth of a second over ten; A7's own timing justifies it, a
    /// timing ten times slower does not; it is above the machine's cap (the reason it exists) and below the rate
    /// itself.
    #[test]
    fn the_measured_cap_is_a7s_measured_rate_over_a_tenth_of_a_second_over_ten() {
        let a7 = A7_TIMING_WORKGROUP_BOUND * 1024;
        let (rate, ok) = measured_cap_holds(a7, 1054.9, CAP_M3_MAX_MEASURED_NODE_EXECUTIONS_PER_SUBMIT);
        assert!(ok, "A7's own timing justifies the cap (rate {rate:.3e})");
        assert!(!measured_cap_holds(a7, 10.0 * 1054.9, CAP_M3_MAX_MEASURED_NODE_EXECUTIONS_PER_SUBMIT).1, "a ten times slower yardstick does not");
        assert!(!measured_cap_holds(a7, 0.0, 1).1, "no time measured, no cap");
        const { assert!(CAP_M3_MAX_MEASURED_NODE_EXECUTIONS_PER_SUBMIT > CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT) };
        assert!((CAP_M3_MAX_MEASURED_NODE_EXECUTIONS_PER_SUBMIT as f64 - rate / 100.0).abs() <= 1.0 + rate * 1e-9);
    }
}
