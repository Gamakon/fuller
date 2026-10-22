//! UI1: a v24 kernel read exactly into the true kingdom (`fuller::wgsl::true_reader`), put through its genes (the
//! per-type write-back into the module's two entry genes under the fixed linker, `fuller::wgsl::true_module`), and its
//! expressed text run on the device against its own text on the problem's admission case.
//!
//! Each text runs `--runs` fresh runs (zeroed arenas each, the harness's run program). The candidates are compared row
//! by row (one stage, seed and query): as sets with the pads dropped (a candidate list is a set, contract rule 4), and
//! byte for byte. A text whose own runs differ is order-dependent (the relaxed atomics of the contract) and is reported
//! so. Every dispatch is bounded before it is submitted by the work guard on the read tree
//! (`TrueKingdom::tree_work_bound_in` with the dispatch's own `cfg`, `DeviceExecutor::with_dispatch_bound`): the seed's
//! own text has the loops of its exact reading, so the same bound holds for it. A dispatch over the per-submit cap is
//! not submitted, and the kernel is reported as not checked on the device.
//!
//!   cargo run --release --features gpu,wgsl --example v24_true_read -- --problem problems/v24_ingest_small \
//!       [--runs 2] <kernel.wgsl>...
//!
//! `--random N [--seed S] [--size P]`: instead, N random modules of the K = 1 table (`mod/v24_k1`, program size P a
//! gene, default 8): each through naga, the device pipeline, and one guarded run on the admission case (the
//! 1,000-gene loop of UI1: no gene may fail any stage).
//!
//! One GPU job at a time, under the GPU lock.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use fuller::typed_gep::Tree;
use fuller::wgsl::harness::{DeviceExecutor, Executor, Problem};
use fuller::wgsl::oracle::device::{Device, Fuel};
use fuller::wgsl::true_kingdom::{default_dir, Rng, Share, TrueKingdom};
use fuller::wgsl::true_module::TrueModule;
use fuller::wgsl::true_reader::{read_module, Contract};

fn stamp() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{:02}:{:02}:{:02}Z", (s / 3600) % 24, (s / 60) % 60, s % 60)
}

const FIELDS: [&str; 8] = ["seed_slot", "stage", "arrived_begin", "arrived_end", "n_queries", "pass_index", "n_passes", "c_max"];

fn functions(t: &Tree, k: &TrueKingdom) -> usize {
    usize::from(!k.rows[t.symbol as usize].args.is_empty()) + t.kids.iter().map(|c| functions(c, k)).sum::<usize>()
}

/// The candidates of every row (stage, seed, query) as sets of stream rows, pads dropped.
fn rows_of(bytes: &[u8]) -> Vec<BTreeSet<u32>> {
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    words.chunks(64).map(|r| r.iter().copied().filter(|w| *w != u32::MAX).collect()).collect()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| -> Option<String> { args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned() };
    let problem_dir = PathBuf::from(value("--problem").ok_or("--problem <dir> is required")?);
    let runs: usize = value("--runs").unwrap_or_else(|| "2".into()).parse().map_err(|e| format!("--runs: {e}"))?;
    let mut kernels = Vec::new();
    let mut skip = false;
    for a in &args {
        if skip {
            skip = false;
            continue;
        }
        if a.starts_with("--") {
            skip = true;
            continue;
        }
        kernels.push(a.clone());
    }
    let problem = Problem::load(&problem_dir)?;
    if let Some(n) = value("--random") {
        let n: u64 = n.parse().map_err(|e| format!("--random: {e}"))?;
        let seed: u64 = value("--seed").unwrap_or_else(|| "1".into()).parse().map_err(|e| format!("--seed: {e}"))?;
        let size: usize = value("--size").unwrap_or_else(|| "8".into()).parse().map_err(|e| format!("--size: {e}"))?;
        return random_loop(&problem, n, seed, size);
    }
    let k = Arc::new(TrueKingdom::load(&default_dir().join("mod").join("v24"))?);
    let contract = Contract::parse(&std::fs::read_to_string(problem_dir.join("safety.contract")).map_err(|e| format!("safety.contract: {e}"))?, &FIELDS)?;
    let case = problem.cases_of("cases/admission.toml")?.into_iter().next().ok_or("no admission case")?;
    let out = problem.pooled_output().cloned().ok_or("the problem has no pooled_recall output")?;
    let oi = problem.spec.output.iter().position(|o| o.name == out.name).ok_or("the pooled output is not the problem's")?;
    let dev = Device::new()?;
    println!("[{}] {} case {}; {} kernels; {runs} fresh runs a text", stamp(), problem.spec.name, case.label, kernels.len());
    println!("kernel\tstatus\tdetail");
    for path in &kernels {
        let name = std::path::Path::new(path).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let trees = match read_module(&k, &text, &contract) {
            Ok(t) => t,
            Err(e) => {
                println!("{name}\tnot read\t{e}");
                continue;
            }
        };
        let sizes: Vec<usize> = trees.iter().map(|t| functions(t, &k)).collect();
        let m = TrueModule::new(&k, &sizes)?;
        let prev = m.random(1, &mut Rng(5), Share::Half, true)?;
        let genes = m.encode(&trees, &prev)?;
        let expressed = m.entry_trees(&genes)?;
        if expressed != trees {
            println!("{name}\tbug\tour software has a bug: the genes express another tree");
            continue;
        }
        let rendered = m.render(&genes)?;
        let (kb, tb) = (k.clone(), Arc::new(trees.clone()));
        let bound = move |entry: usize, words: &[u8]| -> Result<u128, String> {
            let w: Vec<u64> = words.chunks_exact(4).map(|c| u64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
            let fields: BTreeMap<String, u64> = FIELDS.iter().zip(&w).map(|(f, v)| (f.to_string(), *v)).collect();
            let t = tb.get(entry).ok_or_else(|| format!("no tree for entry {entry}"))?;
            kb.tree_work_bound_in(t, &fields)
        };
        let exec = DeviceExecutor::for_problem(&dev, &problem, Fuel::Proven("unused: every dispatch is bounded by with_dispatch_bound")).with_dispatch_bound(Box::new(bound));
        let limit = exec.binding_limit();
        let data = problem.case_data(&case, limit)?;
        let mut results: Vec<Result<Vec<Vec<u8>>, String>> = Vec::new();
        for src in [&text, &rendered] {
            let r = (|| -> Result<Vec<Vec<u8>>, String> {
                let (kern, info) = exec.compile(&problem, src)?;
                let plan = problem.plan(&case, limit, Some(&info))?;
                let (staged, _) = exec.stage(&plan, &data)?;
                let mut outs = Vec::new();
                for _ in 0..runs {
                    let r = exec.run(&kern, &plan, &data, &staged, true)?;
                    outs.push(r.outputs.get(oi).cloned().ok_or("the run read no candidates")?);
                }
                Ok(outs)
            })();
            results.push(r);
        }
        match (&results[0], &results[1]) {
            (Err(e), _) | (_, Err(e)) => println!("{name}\tnot run\t{e}"),
            (Ok(a), Ok(b)) => {
                let sets_a: Vec<Vec<BTreeSet<u32>>> = a.iter().map(|x| rows_of(x)).collect();
                let sets_b: Vec<Vec<BTreeSet<u32>>> = b.iter().map(|x| rows_of(x)).collect();
                let own_a = sets_a.windows(2).all(|w| w[0] == w[1]);
                let own_b = sets_b.windows(2).all(|w| w[0] == w[1]);
                let same = sets_a[0] == sets_b[0];
                let bytes_same = a[0] == b[0];
                let differ = sets_a[0].iter().zip(&sets_b[0]).filter(|(x, y)| x != y).count();
                let status = if same && own_a && own_b { "same values" } else if !own_a { "text order-dependent" } else { "values differ" };
                println!("{name}\t{status}\trows {} differing {differ}; bytes {}; text runs agree {own_a}; expressed runs agree {own_b}; nodes {}", sets_a[0].len(), if bytes_same { "equal" } else { "differ (order)" }, trees.iter().map(|t| t.level_order().len()).sum::<usize>());
            }
        }
    }
    println!("[{}] done", stamp());
    Ok(())
}

/// The random-gene loop: every module through naga, the pipeline and one guarded run; any failure is printed with its
/// gene and kernel and counted, never skipped.
fn random_loop(problem: &Problem, n: u64, seed: u64, size: usize) -> Result<(), String> {
    let k = Arc::new(TrueKingdom::load(&default_dir().join("mod").join("v24_k1"))?);
    let m = TrueModule::new(&k, &[size, size])?;
    let case = problem.cases_of("cases/admission.toml")?.into_iter().next().ok_or("no admission case")?;
    let dev = Device::new()?;
    let mut rng = Rng(seed);
    let (mut naga_fail, mut compile_fail, mut run_fail, mut ok) = (0, 0, 0, 0);
    let t0 = std::time::Instant::now();
    println!("[{}] {n} random v24_k1 modules, program size {size} a gene, seed {seed}, on {} {}", stamp(), problem.spec.name, case.label);
    for i in 0..n {
        let c = m.random(i, &mut rng, Share::Half, true)?;
        let trees = m.entry_trees(&c)?;
        let src = m.render(&c)?;
        let parsed = fuller::wgsl::naga_shape::parse_unscanned(&src).and_then(|md| fuller::wgsl::target::Target::current().validate(&md, &src).map(|_| ()));
        if let Err(e) = parsed {
            naga_fail += 1;
            println!("gene {i}\tnaga\t{e}\n{src}");
            continue;
        }
        let (kb, tb) = (k.clone(), Arc::new(trees));
        let bound = move |entry: usize, words: &[u8]| -> Result<u128, String> {
            let w: Vec<u64> = words.chunks_exact(4).map(|c| u64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
            let fields: BTreeMap<String, u64> = FIELDS.iter().zip(&w).map(|(f, v)| (f.to_string(), *v)).collect();
            kb.tree_work_bound_in(tb.get(entry).ok_or("no tree")?, &fields)
        };
        let exec = DeviceExecutor::for_problem(&dev, problem, Fuel::Proven("unused: every dispatch is bounded by with_dispatch_bound")).with_dispatch_bound(Box::new(bound));
        let limit = exec.binding_limit();
        let compiled = exec.compile(problem, &src);
        let (kern, info) = match compiled {
            Ok(x) => x,
            Err(e) => {
                compile_fail += 1;
                println!("gene {i}\tpipeline\t{e}\n{src}");
                continue;
            }
        };
        let run = (|| -> Result<(), String> {
            let data = problem.case_data(&case, limit)?;
            let plan = problem.plan(&case, limit, Some(&info))?;
            let (staged, _) = exec.stage(&plan, &data)?;
            exec.run(&kern, &plan, &data, &staged, true).map(|_| ())
        })();
        match run {
            Ok(()) => ok += 1,
            Err(e) => {
                run_fail += 1;
                println!("gene {i}\trun\t{e}\n{src}");
            }
        }
        if (i + 1) % 100 == 0 {
            println!("[{}] {} genes: {ok} ran, naga {naga_fail}, pipeline {compile_fail}, run {run_fail} ({:.0} s)", stamp(), i + 1, t0.elapsed().as_secs_f64());
        }
    }
    println!("[{}] done: {n} genes, {ok} through naga, the pipeline and a guarded run; naga {naga_fail}, pipeline {compile_fail}, run {run_fail}", stamp());
    Ok(())
}
