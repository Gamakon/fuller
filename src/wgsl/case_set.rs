//! THE CASES OF A HARNESS PROBLEM, ONE BY ONE (`docs/kmeans_update_dropin.md`): a problem directory's constraint
//! cases staged once on the device with their expected outputs (the shipped kernel is the oracle, `harness`), and its
//! timing cases staged once, so a runner can run any kernel of the problem's interface case by case and judge each
//! case on its own: its norm against the expected output, the case's split (a name starting `held_` is held out, every
//! other a training case) and its GPU milliseconds.
//!
//! [`crate::wgsl::harness::Harness::evaluate`] folds every case into one gated [`crate::wgsl::gram::Fitness`]; the
//! staged two-level fitness of the runner standard (`docs/STANDARD_runners.md`) needs each case as its own input
//! instead, and timed pairs on a timing case that is never a quality case. This is that, over the same plans, the
//! same executor and the same oracle.

use std::path::Path;

use crate::wgsl::harness::{norm_value, Case, CaseData, DeviceExecutor, Executor, KernelInfo, Norm, OutputSpec, Plan, Problem, Role, RunOut};

/// The bound of ONE WORKGROUP of a dispatch: (entry, the dispatch's uniform words, invocations a workgroup), as the
/// executor's `harness::DispatchBound` reads it.
pub type WorkgroupBound<'a> = dyn Fn(usize, &[u8], u64) -> Result<u128, String> + 'a;

/// The two splits of a constraint case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Held,
}

/// One staged case.
pub struct Slot<S> {
    pub case: Case,
    pub split: Split,
    pub data: CaseData,
    /// The constraint output's expected values (empty for a timing case).
    pub expected: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    plan: Plan,
    staged: S,
}

/// What one kernel did on one constraint case.
#[derive(Debug, Clone)]
pub struct CaseOut {
    pub name: String,
    pub split: Split,
    /// The suite's norm of the output against the expected output (0 exact; infinite with a non-finite mismatch).
    pub norm: f64,
    /// The suite's pass value on this case.
    pub pass: f64,
    /// Cells not bit for bit the expected value.
    pub differ: usize,
    pub cells: usize,
    /// The output.
    pub out: Vec<f32>,
    pub gpu_ms: f64,
}

/// A problem's cases on one device executor.
pub struct CaseSet<'d> {
    pub problem: Problem,
    pub exec: DeviceExecutor<'d>,
    pub out: OutputSpec,
    out_index: usize,
    pub norm: Norm,
    pass: Vec<f64>,
    pub cases: Vec<Slot<<DeviceExecutor<'d> as Executor>::Staged>>,
    pub timing: Vec<Slot<<DeviceExecutor<'d> as Executor>::Staged>>,
}

impl<'d> CaseSet<'d> {
    /// Load `dir` and stage every constraint and timing case on `exec`; the constraint cases' expected outputs by the
    /// oracle (the shipped kernel's own outputs, cached under the problem's `oracle/`).
    pub fn new(dir: &Path, exec: DeviceExecutor<'d>) -> Result<CaseSet<'d>, String> {
        let problem = Problem::load(dir)?;
        let limit = exec.binding_limit();
        let su = problem.suites.iter().find(|s| s.spec.role == Role::Constraint).ok_or("the problem has no constraint suite")?;
        let out = problem.output_of(su)?.clone();
        let out_index = problem.spec.output.iter().position(|o| o.name == out.name).ok_or("our software has a bug: the constraint output is not an output")?;
        let norm = su.spec.norm.unwrap_or(Norm::MeanAbsOverMax);
        let (shipped, info) = exec.compile(&problem, &problem.reference).map_err(|e| format!("the shipped kernel: {e}"))?;
        let mut cases = Vec::new();
        let mut pass = Vec::new();
        let mut timing = Vec::new();
        for s in &problem.suites {
            let judged = s.spec.role == Role::Constraint;
            if !judged && s.spec.role != Role::Timing {
                continue;
            }
            for c in &s.cases {
                let env = problem.env(c, limit)?;
                let mut data = problem.case_data(c, limit)?;
                let plan = problem.plan(c, limit, None)?;
                let (staged, _) = exec.stage(&plan, &data)?;
                let (rows, cols) = problem.output_dims(&out, &env)?;
                let mut expected = Vec::new();
                if judged {
                    problem.oracle_expected(&exec, &shipped, &info, c, &mut data, &staged).map_err(|e| format!("the oracle, case {}: {e}", c.name))?;
                    expected = data.expected.get(&out.name).cloned().ok_or_else(|| format!("case {}: no expected {}", c.name, out.name))?;
                    pass.push(match &s.spec.pass {
                        Some(p) => crate::wgsl::harness::eval(p, &env)?,
                        None => 0.0,
                    });
                }
                let split = if c.name.starts_with("held_") { Split::Held } else { Split::Train };
                let slot = Slot { case: c.clone(), split, data, expected, rows, cols, plan, staged };
                if judged {
                    cases.push(slot);
                } else {
                    timing.push(slot);
                }
            }
        }
        Ok(CaseSet { problem, exec, out, out_index, norm, pass, cases, timing })
    }

    /// Compile a kernel of the problem's interface.
    pub fn compile(&self, src: &str) -> Result<(<DeviceExecutor<'d> as Executor>::Kernel, KernelInfo), String> {
        self.exec.compile(&self.problem, src)
    }

    fn run_slot(&self, k: &<DeviceExecutor<'d> as Executor>::Kernel, info: &KernelInfo, s: &Slot<<DeviceExecutor<'d> as Executor>::Staged>, read: bool) -> Result<RunOut, String> {
        // the plan's run steps read the kernel's workgroup size: planned again with it, the setup as staged
        let plan = self.problem.plan(&s.case, self.exec.binding_limit(), Some(info))?;
        if plan.setup != s.plan.setup {
            return Err(format!("our software has a bug: case {}'s setup depends on the kernel", s.case.name));
        }
        self.exec.run(k, &plan, &s.data, &s.staged, read)
    }

    /// Run a compiled kernel once on every constraint case and judge each.
    pub fn judge(&self, k: &<DeviceExecutor<'d> as Executor>::Kernel, info: &KernelInfo) -> Result<Vec<CaseOut>, String> {
        let mut v = Vec::with_capacity(self.cases.len());
        for (s, pass) in self.cases.iter().zip(&self.pass) {
            let r = self.run_slot(k, info, s, true)?;
            let bytes = r.outputs.get(self.out_index).ok_or_else(|| format!("case {}: the run read no {}", s.case.name, self.out.name))?;
            let out = crate::wgsl::harness::bytemuck_f32(bytes);
            if out.len() != s.expected.len() {
                return Err(format!("case {}: {} values read, {} expected", s.case.name, out.len(), s.expected.len()));
            }
            let (norm, _) = norm_value(self.norm, &out, &s.expected, s.rows, s.cols, 0.0);
            let differ = out.iter().zip(&s.expected).filter(|(g, w)| f64::from(**g).to_bits() != w.to_bits()).count();
            v.push(CaseOut { name: s.case.name.clone(), split: s.split, norm, pass: *pass, differ, cells: out.len(), out, gpu_ms: r.gpu_ms });
        }
        Ok(v)
    }

    /// THE WORK BOUND of one run of a compiled kernel on timing case `i`: `bound` is the per-workgroup bound of a
    /// dispatch as the executor's dispatch bound reads it (entry, the dispatch's uniform words, its invocations a
    /// workgroup). Returns the bound of the whole run (every dispatch's workgroups) and the largest bound of ONE
    /// workgroup of any dispatch (a run whose largest is over the executor's cap cannot be split into submits).
    pub fn timing_bound(&self, info: &KernelInfo, i: usize, bound: &WorkgroupBound<'_>) -> Result<(u128, u128), String> {
        let s = self.timing.get(i).ok_or_else(|| format!("no timing case {i}"))?;
        let plan = self.problem.plan(&s.case, self.exec.binding_limit(), Some(info))?;
        let (mut total, mut most) = (0u128, 0u128);
        for c in &plan.run {
            let crate::wgsl::harness::Cmd::Dispatch { bind, workgroups, entry, .. } = c else { continue };
            let words: &[u8] = bind
                .iter()
                .find_map(|(_, id)| match &plan.buffers[*id] {
                    crate::wgsl::harness::BufferDecl { uniform: true, init: crate::wgsl::harness::Init::Words(w), .. } => Some(w.as_slice()),
                    _ => None,
                })
                .unwrap_or(&[]);
            let inv = info.entry_workgroups.get(*entry).map_or(1, |w| u64::from(w[0]) * u64::from(w[1]) * u64::from(w[2]));
            let per = bound(*entry, words, inv)?;
            let groups = workgroups.iter().fold(1u128, |a, w| a.saturating_mul(u128::from(*w)));
            total = total.saturating_add(per.saturating_mul(groups));
            most = most.max(per);
        }
        Ok((total, most))
    }

    /// Each dispatch's GPU milliseconds of one unsplit run on timing case `i` (in dispatch order: which entry costs what).
    pub fn dispatch_ms_unsplit(&self, k: &<DeviceExecutor<'d> as Executor>::Kernel, info: &KernelInfo, i: usize, whole: u128) -> Result<Vec<f64>, String> {
        let s = self.timing.get(i).ok_or_else(|| format!("no timing case {i}"))?;
        Ok(self.exec.unsplit(whole, || self.run_slot(k, info, s, false))?.dispatch_ms)
    }

    /// THE PLACEMENT PROBE: timing case `i` staged again, into a fresh set of resident buffers (its own allocations).
    pub fn timing_restage(&self, i: usize) -> Result<<DeviceExecutor<'d> as Executor>::Staged, String> {
        let s = self.timing.get(i).ok_or_else(|| format!("no timing case {i}"))?;
        Ok(self.exec.stage(&s.plan, &s.data)?.0)
    }

    /// [`CaseSet::dispatch_ms_unsplit`] on the resident buffers `staged` (the case's own, or a [`CaseSet::timing_restage`]).
    pub fn dispatch_ms_unsplit_on(&self, k: &<DeviceExecutor<'d> as Executor>::Kernel, info: &KernelInfo, i: usize, staged: &<DeviceExecutor<'d> as Executor>::Staged, whole: u128) -> Result<Vec<f64>, String> {
        let s = self.timing.get(i).ok_or_else(|| format!("no timing case {i}"))?;
        let plan = self.problem.plan(&s.case, self.exec.binding_limit(), Some(info))?;
        Ok(self.exec.unsplit(whole, || self.exec.run(k, &plan, &s.data, staged, false))?.dispatch_ms)
    }

    /// [`CaseSet::time`] UNSPLIT: every dispatch whole, as UltraDim dispatches it (`DeviceExecutor::unsplit`), the
    /// whole run bounded by `whole`.
    pub fn time_unsplit(&self, k: &<DeviceExecutor<'d> as Executor>::Kernel, info: &KernelInfo, i: usize, whole: u128) -> Result<f64, String> {
        self.exec.unsplit(whole, || self.time(k, info, i))
    }

    /// One run of a compiled kernel on timing case `i` UNSPLIT with its output read back (the timing case's output is
    /// checked against the yardstick's: a kernel exact on the quality cases need not be on the timing case's sizes).
    pub fn timing_output_unsplit(&self, k: &<DeviceExecutor<'d> as Executor>::Kernel, info: &KernelInfo, i: usize, whole: u128) -> Result<Vec<f32>, String> {
        let s = self.timing.get(i).ok_or_else(|| format!("no timing case {i}"))?;
        let r = self.exec.unsplit(whole, || self.run_slot(k, info, s, true))?;
        let bytes = r.outputs.get(self.out_index).ok_or_else(|| format!("timing case {}: the run read no {}", s.case.name, self.out.name))?;
        Ok(crate::wgsl::harness::bytemuck_f32(bytes))
    }

    /// The constraint suite's pass value (its tolerance) at timing case `i`'s sizes.
    pub fn timing_pass(&self, i: usize) -> Result<f64, String> {
        let s = self.timing.get(i).ok_or_else(|| format!("no timing case {i}"))?;
        let su = self.problem.suites.iter().find(|x| x.spec.role == crate::wgsl::harness::Role::Constraint).ok_or("the problem has no constraint suite")?;
        let env = self.problem.env(&s.case, self.exec.binding_limit())?;
        match &su.spec.pass {
            Some(p) => crate::wgsl::harness::eval(p, &env),
            None => Ok(0.0),
        }
    }

    /// One timed run of a compiled kernel on timing case `i`: its GPU milliseconds (the sum of its dispatches'
    /// timestamps), nothing read back.
    pub fn time(&self, k: &<DeviceExecutor<'d> as Executor>::Kernel, info: &KernelInfo, i: usize) -> Result<f64, String> {
        let s = self.timing.get(i).ok_or_else(|| format!("no timing case {i}"))?;
        Ok(self.run_slot(k, info, s, false)?.gpu_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::oracle::device::{Device, Fuel};

    /// THE KEPT RUN BUFFERS OVER MANY RUNS AND SEVERAL KERNELS (GPU): problems/kmeans_update_segmented, three kernels
    /// of its interface (the scatter form, the gather, the evolved champion), each judged on every constraint case and
    /// run on the timing case, five rounds, interleaved, with a re-staged timing case made and dropped between rounds.
    /// Every round's outputs are bit for bit those of an executor that keeps no buffer, and the kept set stops
    /// growing after the first round. (2026-10-31: a kept buffer was destroyed at the end of its first run, and the
    /// second run's write to it was a validation error, "Buffer with 'cen' label has been destroyed".)
    #[test]
    fn kept_run_buffers_survive_many_runs_of_several_kernels_on_the_gpu() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dir = root.join("problems/kmeans_update_segmented");
        let texts = [
            std::fs::read_to_string(dir.join("segmented.wgsl")).unwrap(),
            std::fs::read_to_string(dir.join("gather.wgsl")).unwrap(),
            std::fs::read_to_string(root.join("docs/kmeans_champion_s1/champion.wgsl")).unwrap(),
        ];
        let dev = Device::new().expect("device");
        let fresh = CaseSet::new(&dir, DeviceExecutor::new(&dev, Fuel::Proven("the problem's and the champion's kernels"))).unwrap();
        let want: Vec<Vec<Vec<u32>>> = texts
            .iter()
            .map(|t| {
                let (k, info) = fresh.compile(t).unwrap();
                fresh.judge(&k, &info).unwrap().iter().map(|c| c.out.iter().map(|x| x.to_bits()).collect()).collect()
            })
            .collect();
        assert_eq!(fresh.exec.kept_run_buffers(), 0, "an executor that keeps no buffer kept one");
        drop(fresh);
        let set = CaseSet::new(&dir, DeviceExecutor::new(&dev, Fuel::Proven("the problem's and the champion's kernels"))).unwrap();
        set.exec.reuse_run_buffers();
        let kernels: Vec<_> = texts.iter().map(|t| set.compile(t).unwrap()).collect();
        let mut kept_after_first = 0;
        for round in 0..5 {
            for (j, (k, info)) in kernels.iter().enumerate() {
                let got: Vec<Vec<u32>> = set.judge(k, info).unwrap().iter().map(|c| c.out.iter().map(|x| x.to_bits()).collect()).collect();
                assert_eq!(got, want[j], "round {round}, kernel {j}: the outputs on kept buffers are not those on fresh ones");
                let ms = set.time(k, info, 0).unwrap();
                assert!(ms > 0.0, "round {round}, kernel {j}: the timing run took {ms} ms");
            }
            let restaged = set.timing_restage(0).unwrap();
            drop(restaged);
            if round == 0 {
                kept_after_first = set.exec.kept_run_buffers();
                assert!(kept_after_first > 0, "the executor kept no run buffer");
            } else {
                assert_eq!(set.exec.kept_run_buffers(), kept_after_first, "round {round}: the kept set grew");
            }
        }
    }
}
