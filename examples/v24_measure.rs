//! MEASURE ONE v24 KERNEL ON THE DEVICE, window by window and stage by stage
//! (fuller docs/PLAN_true_gep_wgsl.md task UI2, the anti-scan test): a
//! pooled_recall problem's stream case, staged once, then `--runs` fresh runs
//! of the kernel (each zeroes its arenas, as every timing run of the fitness
//! does); every run's candidates judged on the device (`judge_pooled`), its
//! dispatches' GPU timestamps summed by label (`samples_by_label`) and set
//! beside today's (`pooled_timing`: the ingest through a stage against
//! today's settle of that stage, a stage's search against today's search of
//! the same queries).
//!
//! Printed, every measurement its own line (nothing averaged): per run and
//! window (stage x oracle half) the queries, recall@10 = found / (10 x
//! queries), misses M and today's, pool P and today's, broken entries B; per
//! run and stage the kernel's ingest-through-stage and search GPU ms with
//! today's, and the HFF time inputs (`time_input`).
//!
//! The kernel is measured as written, with `Fuel::Proven`: run it only on a
//! kernel whose loops were read and bounded by hand (the run-4 champion: its
//! search reads every arrived row's 128 words once a query, its other loops
//! are constants), one GPU job at a time, under the GPU lock.
//!
//! Run: cargo run --release --features gpu,wgsl --example v24_measure --
//!      --problem problems/v24_ingest250k --kernel <file.wgsl | reference>
//!      [--runs 3] [--set name=value ...]   (case parameters: stages=1, arena_u32=33554432)

use std::path::PathBuf;
use std::time::Instant;

use fuller::wgsl::harness::{samples_by_label, pooled_timing, time_input, DeviceExecutor, Executor, Problem};
use fuller::wgsl::oracle::device::{Device, Fuel};

fn stamp() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{:02}:{:02}:{:02}Z", (s / 3600) % 24, (s / 60) % 60, s % 60)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| -> Option<String> { args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned() };
    let problem_dir = PathBuf::from(value("--problem").ok_or("--problem <dir> is required")?);
    let kernel = value("--kernel").ok_or("--kernel <file.wgsl | reference> is required")?;
    let runs: usize = value("--runs").unwrap_or_else(|| "3".into()).parse().map_err(|e| format!("--runs: {e}"))?;
    if runs == 0 {
        return Err("--runs must be at least 1".into());
    }
    let mut sets = Vec::new();
    for (i, a) in args.iter().enumerate() {
        if a == "--set" {
            let kv = args.get(i + 1).ok_or("--set name=value")?;
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("--set {kv}: not name=value"))?;
            sets.push((k.to_string(), v.parse::<f64>().map_err(|e| format!("--set {kv}: {e}"))?));
        }
    }

    let t_all = Instant::now();
    let problem = Problem::load(&problem_dir)?;
    let source = if kernel == "reference" { problem.reference.clone() } else { std::fs::read_to_string(&kernel).map_err(|e| format!("{kernel}: {e}"))? };
    let out = problem.pooled_output().cloned().ok_or("the problem has no pooled_recall output")?;
    let oi = problem.spec.output.iter().position(|o| o.name == out.name).ok_or("the pooled output is not the problem's")?;
    let mut case = problem.suites.first().and_then(|s| s.cases.first()).cloned().ok_or("the problem has no stream case")?;
    for (k, v) in &sets {
        case.params.insert(k.clone(), *v);
    }
    let dev = Device::new()?;
    let exec = DeviceExecutor::for_problem(&dev, &problem, Fuel::Proven("a kernel measured by v24_measure: its loops read and bounded by hand before the run"));
    let limit = exec.binding_limit();
    println!("[{}] {} case {} {:?}; kernel {kernel}; binding limit {limit} bytes; {runs} fresh runs", stamp(), problem.spec.name, case.label, sets);
    let (k, info) = exec.compile(&problem, &source)?;
    let data = problem.case_data(&case, limit)?;
    let env = problem.env(&case, limit)?;
    let plan = problem.plan(&case, limit, Some(&info))?;
    let (staged, stage_ms) = exec.stage(&plan, &data)?;
    println!("[{}] staged (keys projected on the device) in {stage_ms:.0} ms", stamp());

    println!("run\tstage\thalf\tqueries\trecall@10\tM\tM_today\tP\tP_today\tB");
    let mut timing_rows = Vec::new();
    for run in 1..=runs {
        let t = Instant::now();
        let r = exec.run(&k, &plan, &data, &staged, true)?;
        let bytes = r.outputs.get(oi).ok_or("the run read no candidates")?;
        let windows = exec.judge_pooled(&problem, &out, &env, &data, bytes)?;
        for w in &windows {
            let recall = w.found as f64 / (10.0 * w.queries as f64);
            println!("{run}\t{}\t{}\t{}\t{recall:.4}\t{}\t{}\t{}\t{}\t{}", w.stage, w.half, w.queries, w.misses, w.misses_today, w.pooled, w.pooled_today, w.broken);
        }
        let samples = pooled_timing(&problem, &out, &env, &data, &samples_by_label(&plan, &r)?)?;
        for (label, ms, today) in samples {
            timing_rows.push(format!("{run}\t{label}\t{ms:.2}\t{today:.1}\t{:.4}", time_input(today, ms)));
        }
        println!("[{}] run {run}: {:.0} ms GPU, {:.1} s wall", stamp(), r.gpu_ms, t.elapsed().as_secs_f64());
    }
    println!("run\tsample\tkernel_gpu_ms\ttoday_ms\ttime_input");
    for row in timing_rows {
        println!("{row}");
    }
    println!("[{}] done in {:.1} s", stamp(), t_all.elapsed().as_secs_f64());
    Ok(())
}
