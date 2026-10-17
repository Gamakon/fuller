//! THE SHIPPED KERNEL THROUGH THE FLOW (`docs/USER_FLOW_wgsl_tuning.md`,
//! steps 1-3, on the CPU): one function, [`shipped_gates`], takes any problem
//! directory and runs its shipped kernel (`problem.toml`'s `[kernel]
//! reference`) through every gate the flow puts before evolution, with no
//! code written for the problem:
//!
//! 1. parsed into the kingdom by the safe reader, under the interface the
//!    module itself gives (no `// interface` line: `render::INFERRED`);
//! 2. legal under the checker's seven layers (`check.rs`), layer 7 under the
//!    problem's dispatch contract, `<problem>/safety.contract` (the format of
//!    `safety::Contract::parse`; absent, the contract that names nothing);
//! 3. the Karva round trip: the tree encoded into a chromosome, every gene
//!    decoded and unfolded, the tree built back node for node;
//! 4. rendered under its interface, validated by naga for the target;
//! 5. the round trip by value (`roundtrip::same_values`): the source and its
//!    rendering compute the same bits on the CPU executor.
//!
//! The harness's own gate (the shipped kernel scored feasible against its
//! own outputs) needs an executor; `harness::CpuExecutor` runs it on the CPU.

use std::path::Path;

use crate::homeotic;
use crate::karva::karva_to_terms_generic;
use crate::wgsl::harness::Problem;
use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::roundtrip::{same_values, RoundTrip, Trial};
use crate::wgsl::safe_karva::{build, encode, render_tree_with, split, unfold_decoded};
use crate::wgsl::safety::Contract;

/// The head length and the chromosome's layout the dumps use.
pub const HEAD: usize = 128;

/// What the gates measured.
#[derive(Debug, Clone)]
pub struct Gates {
    /// The interface the kernel was read under, and its entry point.
    pub role: String,
    pub entry: String,
    pub nodes: usize,
    pub genes: usize,
    /// The rendered text (validated).
    pub rendered: String,
    pub round_trip: RoundTrip,
}

/// The problem's dispatch contract: `<dir>/safety.contract`, or the
/// contract that names nothing when the problem gives none.
pub fn contract_of(dir: &Path) -> Result<Contract, String> {
    let path = dir.join("safety.contract");
    if !path.exists() {
        return Ok(Contract::default());
    }
    Contract::parse(&std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?)
}

/// Gates 1-5 for the problem's shipped kernel; an error names the gate
/// that failed and why.
pub fn shipped_gates(dir: &Path, kingdom: &WgslKingdom, trial: &Trial) -> Result<Gates, String> {
    let problem = Problem::load(dir)?;
    source_gates(&problem.reference, &problem.spec.kernel.entry, contract_of(dir)?, kingdom, trial)
}

/// Gates 1-5 for a kernel's text with the problem's entry point and contract.
pub fn source_gates(source: &str, entry: &str, contract: Contract, kingdom: &WgslKingdom, trial: &Trial) -> Result<Gates, String> {
    let r = crate::wgsl::safe_reader::read_safe(source, kingdom).map_err(|e| format!("parse: {e}"))?;
    if r.role.name != crate::wgsl::render::INFERRED {
        return Err(format!("parse: read under the interface {}, not the one the module gives", r.role.name));
    }
    if r.role.entry != entry {
        return Err(format!("parse: the entry point read is {}, the problem's {entry}", r.role.entry));
    }
    let checker = crate::wgsl::check::Checker::with_contract(kingdom, &r.role, &crate::wgsl::target::Target::current(), contract);
    checker.check(&r.tree).map_err(|e| format!("legality: {e}"))?;
    let tree = r.tree;
    let slots = split(&tree, HEAD)?.tail.len().div_ceil(8) * 8;
    let c = encode(&tree, HEAD, slots, 7013).map_err(|e| format!("karva: does not encode: {e}"))?;
    let pset = homeotic::pset_with_hrefs(&c.pset, slots);
    let decoded: Vec<String> = c.genes.iter().map(|(hd, tl)| karva_to_terms_generic(hd, tl, &pset)).collect::<Result<_, _>>().map_err(|e| format!("karva: a gene does not decode: {e}"))?;
    let back = build(&unfold_decoded(&decoded, c.folded.filled)?, kingdom, &r.role).map_err(|e| format!("karva: the decoded genes do not build: {e}"))?;
    if back != tree {
        return Err(format!("karva: the genome decodes to another tree ({} nodes, decoded {})", tree.size(), back.size()));
    }
    let rendered = render_tree_with(&tree, kingdom, &r.role).map_err(|e| format!("render: {e}"))?.text;
    crate::wgsl::render::validate_text(&rendered).map_err(|e| format!("validate: {e}"))?;
    if !rendered.contains(&format!("fn {}(", r.role.entry)) {
        return Err(format!("render: the rendering does not keep the entry point {}", r.role.entry));
    }
    let round_trip = same_values(source, kingdom, trial).map_err(|e| format!("same values: {e}"))?;
    Ok(Gates { role: r.role.name.clone(), entry: r.role.entry.clone(), nodes: tree.size(), genes: c.genes.len(), rendered, round_trip })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::harness::{CpuExecutor, Harness, Role};

    /// EVERY PROBLEM DIRECTORY'S SHIPPED KERNEL THROUGH GATES 1-5, no code
    /// for any problem: the directories are listed, not named.
    #[test]
    fn every_problems_shipped_kernel_passes_the_flows_gates() {
        let k = WgslKingdom::load_safe();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("problems");
        let mut dirs: Vec<_> = std::fs::read_dir(&root).expect("problems/").flatten().map(|e| e.path()).filter(|p| p.join("problem.toml").exists()).collect();
        dirs.sort();
        let mut failed = Vec::new();
        for d in &dirs {
            let p = Problem::load(d).expect("the problem loads");
            // The Gram's reference is the grid-headed Gram kernel of its own
            // interface; the flow's inferred interface is for a captured kernel.
            if p.spec.kernel.grid_header {
                continue;
            }
            match shipped_gates(d, &k, &Trial::default()) {
                Ok(g) => {
                    println!("{}: entry {}, {} nodes, {} genes, every layer passed, round trip {:?}", d.display(), g.entry, g.nodes, g.genes, g.round_trip);
                    if g.round_trip != RoundTrip::Same {
                        failed.push(format!("{}: round trip {:?}", d.display(), g.round_trip));
                    }
                }
                Err(e) => failed.push(format!("{}: {e}", d.display())),
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    /// The problems whose judged cases take their expected outputs from the
    /// shipped kernel (no expected file), with their timing shapes made the
    /// constraint's cheapest case (the CPU executor cannot time production
    /// shapes) and the oracle's cache in a fresh directory.
    const CPU_WORK: f64 = 3e6;

    fn oracle_problems(tag: &str) -> Vec<Problem> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("problems");
        let mut dirs: Vec<_> = std::fs::read_dir(&root).expect("problems/").flatten().map(|e| e.path()).filter(|p| p.join("problem.toml").exists()).collect();
        dirs.sort();
        let mut out = Vec::new();
        for d in dirs {
            let mut p = Problem::load(&d).expect("the problem loads");
            let judged = |r: Role| matches!(r, Role::Constraint | Role::Accuracy);
            // A pooled_recall problem's expected values are its accept sets
            // and today's counts, not the shipped kernel's outputs.
            if p.pooled_output().is_some() {
                continue;
            }
            if !p.suites.iter().any(|s| judged(s.spec.role) && s.cases.iter().any(|c| c.expected.is_empty())) {
                continue;
            }
            // The cases the CPU executor affords (it runs about a million
            // multiply-adds a second): the judged suites' cases of at most
            // CPU_WORK work, no deadline and no budget gate (its times are a
            // loaded host's wall clock, not a device's).
            let probe = p.clone();
            for s in p.suites.iter_mut().filter(|s| judged(s.spec.role)) {
                s.cases.retain(|c| probe.work(c, 1 << 27).is_ok_and(|w| w <= CPU_WORK));
            }
            p.spec.budget.base_ms = 1e9;
            p.spec.budget.deadline_factor = None;
            p.spec.budget.deadline_floor_ms = None;
            let cheapest = p.suites.iter().find(|s| s.spec.role == Role::Constraint).and_then(|s| s.cases.iter().min_by(|a, b| p.work(a, 1 << 27).unwrap_or(f64::MAX).total_cmp(&p.work(b, 1 << 27).unwrap_or(f64::MAX)))).cloned().expect("a constraint case");
            for s in p.suites.iter_mut().filter(|s| s.spec.role == Role::Timing) {
                s.cases = vec![cheapest.clone(); s.cases.len()];
            }
            let cache = std::env::temp_dir().join(format!("fuller_oracle_{tag}_{}_{}", p.spec.name, std::process::id()));
            if cache.exists() {
                std::fs::remove_dir_all(&cache).expect("a stale cache");
            }
            p.spec.oracle.cache = cache.display().to_string();
            out.push(p);
        }
        out
    }

    /// GATE 6, THE SHIPPED KERNEL SCORED AGAINST ITS OWN OUTPUTS: every
    /// problem whose oracle is the shipped kernel, its inputs generated, its
    /// expected outputs the shipped kernel's on the CPU executor (cached),
    /// then the shipped kernel evaluated by the harness: feasible, every
    /// constraint case passed. A second harness reads the oracle back from
    /// the cache and scores the same; a kernel that takes the argmin instead
    /// of the argmax (one comparison flipped in the text) is infeasible.
    #[test]
    fn every_shipped_oracle_problem_scores_its_shipped_kernel_feasible_on_the_cpu() {
        let problems = oracle_problems("gate6");
        assert!(problems.iter().any(|p| p.spec.name == "kmeans_assign"), "problems/kmeans_assign is an oracle problem");
        let exec = CpuExecutor::new(1 << 27);
        for p in problems {
            let (name, reference, cache) = (p.spec.name.clone(), p.reference.clone(), p.dir.join(&p.spec.oracle.cache));
            let t = std::time::Instant::now();
            let h = Harness::new(p.clone(), &exec).unwrap_or_else(|e| panic!("{name}: {e}"));
            let f = h.evaluate(&reference);
            println!("{name}: harness {:.1} s; shipped kernel feasible {} ({} of {} constraint cases), detail {:?}", t.elapsed().as_secs_f64(), f.feasible, f.passed, h.constraint.len(), f.detail);
            assert!(f.feasible && f.passed == h.constraint.len(), "{name}: the shipped kernel against its own outputs: {}", f.detail);
            let files = std::fs::read_dir(&cache).map(|r| r.count()).unwrap_or(0);
            assert!(files >= h.constraint.len(), "{name}: {files} cached oracle files for {} cases", h.constraint.len());
            let again = Harness::new(p.clone(), &exec).unwrap_or_else(|e| panic!("{name}: {e}"));
            for (a, b) in h.constraint.iter().zip(&again.constraint) {
                assert_eq!(a.data.expected, b.data.expected, "{name}: case {}: the cached oracle differs", a.case.name);
            }
            if name == "kmeans_assign" {
                let argmin = reference.replace("sim > best_sim", "sim < best_sim");
                assert_ne!(argmin, reference);
                let g = again.evaluate(&argmin);
                println!("{name}: the argmin kernel: feasible {}, {}", g.feasible, g.detail);
                assert!(!g.feasible, "{name}: the argmin kernel scored feasible");
            }
            std::fs::remove_dir_all(&cache).expect("the test's cache");
        }
    }
}
