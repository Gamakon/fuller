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

    /// One timed run of a compiled kernel on timing case `i`: its GPU milliseconds (the sum of its dispatches'
    /// timestamps), nothing read back.
    pub fn time(&self, k: &<DeviceExecutor<'d> as Executor>::Kernel, info: &KernelInfo, i: usize) -> Result<f64, String> {
        let s = self.timing.get(i).ok_or_else(|| format!("no timing case {i}"))?;
        Ok(self.run_slot(k, info, s, false)?.gpu_ms)
    }
}
