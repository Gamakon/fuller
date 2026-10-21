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

/// Node executions one submit may be bounded by, on this machine (Apple M3 Max, the laptop that also drives the
/// display). Basis: the evaluator throughput measured earlier, about 17 G node evaluations per second (phylu tsr
/// speed report), times 0.1 s (a submit stays well under the second the display can wait), divided by a safety
/// factor of 10 for a kernel slower than the evaluator: 17e9 * 0.1 / 10 = 1.7e8.
pub const CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT: u128 = 170_000_000;

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

    /// The bound of a submit of `invocations`, or `None` for a proven kernel.
    pub fn bound(&self, invocations: u64) -> Option<u128> {
        match self {
            Fuel::Proven(_) => None,
            Fuel::Unported => Some(u128::MAX),
            Fuel::Kingdom { h, k } => Some(work_bound(*h, *k, invocations)),
        }
    }

    /// The guard: the submit of `what` (`invocations` in all its dispatches) may start only if its bound is at
    /// most the cap. Returns the bound (0 for a proven kernel).
    pub fn check(&self, what: &str, invocations: u64) -> Result<u128, String> {
        self.refuse_unported()?;
        let Fuel::Kingdom { h, k } = self else { return Ok(0) };
        let bound = work_bound(*h, *k, invocations);
        if bound <= CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT {
            return Ok(bound);
        }
        Err(format!(
            "the experiment is misconfigured, not the kernel: {what} is bounded by H * 2^(6K) * invocations = {h} * 2^{} * {invocations} = {bound} node executions in one submit, over the cap {CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT} (M3 Max, about 0.1 s at the measured 17 G/s with a safety factor of 10). Nothing was submitted. Lower the fuel budget K (now {k}), use a smaller dispatch, or split the work into several submits.",
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
        let example = include_str!("../../examples/wgsl_true_loop.rs");
        assert!(example.contains("Gram::compile(dev, src, self.fuel.clone())") && example.contains("fuel.check(\"stage (c) at the unit-test dispatch\"") && example.contains("self.fuel.check(\"stage (c)\""), "wgsl_true_loop stage (c)");
        assert!(example.find("fuel.check(\"stage (c) at the unit-test dispatch\"") < example.find("Device::new()"), "the guard comes before the device is opened");
    }

    #[test]
    fn the_cap_is_the_measured_rate_over_a_tenth_of_a_second_over_ten() {
        assert_eq!(CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT, 17_000_000_000 / 10 / 10);
    }
}
