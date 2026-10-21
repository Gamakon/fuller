//! THE DEVICE TABLE OF THE SYMMETRIC HALF: parents from a population, each
//! against its symmetric-half child (`rules_more::symmetric_half`), scored
//! by fuller's harness (`problems/gram`, so the child's `// mode full` runs
//! one band over all of H).
//!
//! - Correctness on the held-out shapes: every case of the constraint,
//!   train and validation suites, and the extrapolation suite where the
//!   full-H mode fits the device's binding (d * d * 4 <= the binding
//!   limit), each kernel's norm against the harness's pass bound.
//! - Speed: `--rounds` randomized ABBA rounds at the timing shapes the full-H
//!   mode fits (the harness's timing cases whose H fits one binding): per
//!   round the order P C C P or C P P C drawn from a fixed seed, the
//!   round's ratio the parents' mean GPU ms over the children's, per shape.
//! - Per pair: the median ratio per shape, the rounds the child is faster,
//!   the geometric mean over the shapes of the median ratios.
//! - Over the pairs: a two-stage cluster bootstrap (pairs resampled, then
//!   each pair's rounds) of the geometric mean of the pairs' speed-ups, its
//!   95 % percentile interval, and the decision: keep the rule if the lower
//!   bound is at least `--keep` (1.3).
//!
//! The parents are the first `--parents` kernels of the directories, in
//! name order, that the checker passes, on which the rule fires, and whose
//! rendered texts are distinct. Before every round the run file
//! `<out>/killme.trigger` must exist (delete it to stop). A watchdog ends the
//! process if one run passes `--deadline` seconds.
//!
//! `--dry` writes the pairs and stops before the device.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_symmetric_table -- --problem problems/gram --out <dir> [--parents 15] [--rounds 20] [--seed 1] [--keep 1.3] [--deadline 120] [--dry] <dir|file>...

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use fuller::wgsl::harness::{bytemuck_f32, points_value, DeviceExecutor, Executor, Harness, Problem, Staged};
use fuller::wgsl::knob;
use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::oracle::device::Device;
use fuller::wgsl::rules_more::{symmetric_half, symmetric_half_sites};
use fuller::wgsl::safe_karva::render_tree_with;
use fuller::wgsl::safe_reader::read_safe;

static DEADLINE: AtomicU64 = AtomicU64::new(0);

/// A pair: its name, the rounds' ratios per shape, the parent and the child
/// correct on the held-out cases, a note.
type PairResult = (String, Vec<Vec<f64>>, bool, bool, String);

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn gmean(v: &[f64]) -> f64 {
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len().max(1) as f64).exp()
}

/// A kernel's verdict on one held-out case: its norm value, the pass bound.
fn held_out<E: Executor>(h: &Harness<E>, k: &E::Kernel, info: &fuller::wgsl::harness::KernelInfo, s: &Staged<E::Staged>, norm: fuller::wgsl::harness::Norm, bound: f64) -> Result<(f64, bool), String> {
    let limit = h.exec.binding_limit();
    let plan = h.problem.plan(&s.case, limit, Some(info))?;
    let r = h.exec.run(k, &plan, &s.data, &s.staged, true)?;
    let out = h.problem.spec.output.first().ok_or("no output")?;
    let hv = bytemuck_f32(r.outputs.first().ok_or("no output read")?);
    let want = s.data.expected.get(&out.name).ok_or("no expected output")?;
    let (v, nonfinite) = points_value(norm, &hv, want, s.dims, s.points, s.tau);
    Ok((v, nonfinite == 0 && v <= bound))
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |flag: &str| -> Result<Option<String>, String> {
        match args.iter().position(|a| a == flag) {
            Some(i) => {
                let v = args.get(i + 1).cloned().ok_or(format!("{flag} needs a value"))?;
                args.drain(i..i + 2);
                Ok(Some(v))
            }
            None => Ok(None),
        }
    };
    let problem_dir = PathBuf::from(take("--problem")?.ok_or("--problem <dir>")?);
    let out = PathBuf::from(take("--out")?.ok_or("--out <dir>")?);
    let parents_n: usize = take("--parents")?.map_or(Ok(15), |v| v.parse().map_err(|e| format!("--parents: {e}")))?;
    let rounds: usize = take("--rounds")?.map_or(Ok(20), |v| v.parse().map_err(|e| format!("--rounds: {e}")))?;
    let seed: u64 = take("--seed")?.map_or(Ok(1), |v| v.parse().map_err(|e| format!("--seed: {e}")))?;
    let keep: f64 = take("--keep")?.map_or(Ok(1.3), |v| v.parse().map_err(|e| format!("--keep: {e}")))?;
    let deadline_s: u64 = take("--deadline")?.map_or(Ok(120), |v| v.parse().map_err(|e| format!("--deadline: {e}")))?;
    let dry = args.iter().position(|a| a == "--dry").map(|i| args.remove(i)).is_some();
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let trigger = out.join("killme.trigger");
    let start = Instant::now();
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let d = DEADLINE.load(Ordering::SeqCst);
        if d != 0 && start.elapsed().as_millis() as u64 > d {
            eprintln!("watchdog: a run passed its deadline; exiting");
            std::process::exit(3);
        }
    });
    let arm = || DEADLINE.store(start.elapsed().as_millis() as u64 + deadline_s * 1000, Ordering::SeqCst);
    let disarm = || DEADLINE.store(0, Ordering::SeqCst);

    // The parents and their children, CPU only.
    let kingdom = WgslKingdom::load_safe();
    let mut files: Vec<PathBuf> = Vec::new();
    for a in &args {
        let p = Path::new(a);
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p).map_err(|e| format!("{a}: {e}"))?.flatten().map(|e| e.path()).filter(|f| f.extension().is_some_and(|x| x == "wgsl")).collect();
            v.sort();
            files.extend(v);
        } else {
            files.push(p.to_path_buf());
        }
    }
    let mut pairs: Vec<(String, String, String)> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for f in &files {
        if pairs.len() >= parents_n {
            break;
        }
        let name = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let Ok(src) = std::fs::read_to_string(f) else { continue };
        let Ok(r) = read_safe(&src, &kingdom) else { continue };
        if symmetric_half_sites(&r.tree, &kingdom).is_empty() || knob::check_legal_under(&r.tree, &kingdom, &knob::Under::of(&r.role)).is_err() {
            continue;
        }
        let parent = render_tree_with(&r.tree, &kingdom, &r.role)?.text;
        if !seen.insert(parent.clone()) {
            continue;
        }
        let child_tree = symmetric_half(&r.tree, &kingdom)?;
        let child = knob::check_legal_under(&child_tree, &kingdom, &knob::Under::of(&r.role)).map_err(|(s, e)| format!("{name}: the child: {}: {e}", s.name()))?;
        std::fs::write(out.join(format!("{name}.parent.wgsl")), &parent).map_err(|e| e.to_string())?;
        std::fs::write(out.join(format!("{name}.child.wgsl")), &child).map_err(|e| e.to_string())?;
        pairs.push((name, parent, child));
    }
    eprintln!("{} pairs; the device next", pairs.len());
    // --dry: the pairs only, no device.
    if dry {
        return Ok(());
    }

    let dev = Device::new()?;
    let exec = DeviceExecutor::new(&dev, fuller::wgsl::oracle::device::Fuel::Proven("legacy kingdom candidate: no fuel budget exists for it, bounded only by the harness runaway guard"));
    let harness = Harness::new(Problem::load(&problem_dir)?, &exec)?;
    let limit = exec.binding_limit();
    // The timing shapes the full-H mode fits.
    let shapes: Vec<usize> = (0..harness.timing.len())
        .filter(|&i| {
            let env = harness.problem.env(&harness.timing[i].case, limit);
            env.ok().and_then(|e| e.get("d").copied()).is_some_and(|d| d * d * 4.0 <= limit as f64)
        })
        .collect();
    let shape_names: Vec<String> = shapes.iter().map(|&i| harness.timing[i].case.label.clone()).collect();
    eprintln!("timing shapes the full-H mode fits (binding limit {limit} B): {shape_names:?}");
    let mut held = std::fs::File::create(out.join("held_out.tsv")).map_err(|e| e.to_string())?;
    writeln!(held, "pair\tkernel\tsuite\tcase\tvalue\tbound\tpass").map_err(|e| e.to_string())?;
    let mut rounds_tsv = std::fs::File::create(out.join("rounds.tsv")).map_err(|e| e.to_string())?;
    writeln!(rounds_tsv, "pair\tround\torder\tshape\tparent_ms_1\tchild_ms_1\tchild_ms_2\tparent_ms_2\tratio").map_err(|e| e.to_string())?;
    let mut rng = seed;
    // per pair: per shape, the rounds' ratios; and correctness.
    let mut results: Vec<PairResult> = Vec::new();
    for (name, parent, child) in &pairs {
        if !trigger.exists() {
            return Err("the run file is gone: stopped".into());
        }
        let (kp, ip) = exec.compile(&harness.problem, parent)?;
        let (kc, ic) = match exec.compile(&harness.problem, child) {
            Ok(x) => x,
            Err(e) => {
                results.push((name.clone(), Vec::new(), false, false, format!("the child does not compile: {e}")));
                continue;
            }
        };
        // Held-out correctness: the constraint, train, validation and (where
        // the full H fits) extrapolation cases.
        let (mut ok_p, mut ok_c, mut notes) = (true, true, String::new());
        let mut cases: Vec<(&str, &Staged<_>, fuller::wgsl::harness::Norm, f64)> = harness.constraint.iter().map(|s| ("constraint", s, harness.constraint_norm, s.pass)).collect();
        for (split, label) in [(0usize, "train"), (1, "val"), (2, "extrap")] {
            for (s, norm, slack) in &harness.accuracy[split] {
                cases.push((label, s, *norm, (slack * s.reference).max(1.0)));
            }
        }
        for (suite, s, norm, bound) in cases {
            let d = harness.problem.env(&s.case, limit)?.get("d").copied().unwrap_or(0.0);
            arm();
            let p = held_out(&harness, &kp, &ip, s, norm, bound);
            let c = if d * d * 4.0 <= limit as f64 { Some(held_out(&harness, &kc, &ic, s, norm, bound)) } else { None };
            disarm();
            let (pv, pp) = p?;
            ok_p &= pp;
            writeln!(held, "{name}\tparent\t{suite}\t{}\t{pv:.4e}\t{bound:.4e}\t{pp}", s.case.name).map_err(|e| e.to_string())?;
            match c {
                Some(c) => {
                    let (cv, cp) = c?;
                    ok_c &= cp;
                    writeln!(held, "{name}\tchild\t{suite}\t{}\t{cv:.4e}\t{bound:.4e}\t{cp}", s.case.name).map_err(|e| e.to_string())?;
                }
                None => {
                    notes = format!("{} not held by the full-H mode (d {d}: {} B over the {limit} B binding)", s.case.name, d * d * 4.0);
                    writeln!(held, "{name}\tchild\t{suite}\t{}\t-\t{bound:.4e}\tnot run: full H over the binding", s.case.name).map_err(|e| e.to_string())?;
                }
            }
        }
        // The ABBA rounds.
        let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); shapes.len()];
        for round in 0..rounds {
            if !trigger.exists() {
                return Err("the run file is gone: stopped".into());
            }
            let pcp = next(&mut rng).is_multiple_of(2);
            for (j, &i) in shapes.iter().enumerate() {
                let s = &harness.timing[i];
                let (plan_p, plan_c) = (harness.problem.plan(&s.case, limit, Some(&ip))?, harness.problem.plan(&s.case, limit, Some(&ic))?);
                let t = |k: &_, plan: &_| -> Result<f64, String> {
                    arm();
                    let r = exec.run(k, plan, &s.data, &s.staged, false);
                    disarm();
                    Ok(r?.gpu_ms)
                };
                let (p1, c1, c2, p2) = if pcp {
                    let p1 = t(&kp, &plan_p)?;
                    let c1 = t(&kc, &plan_c)?;
                    let c2 = t(&kc, &plan_c)?;
                    (p1, c1, c2, t(&kp, &plan_p)?)
                } else {
                    let c1 = t(&kc, &plan_c)?;
                    let p1 = t(&kp, &plan_p)?;
                    let p2 = t(&kp, &plan_p)?;
                    (p1, c1, t(&kc, &plan_c)?, p2)
                };
                let ratio = (p1 + p2) / (c1 + c2);
                ratios[j].push(ratio);
                writeln!(rounds_tsv, "{name}\t{round}\t{}\t{}\t{p1:.4}\t{c1:.4}\t{c2:.4}\t{p2:.4}\t{ratio:.4}", if pcp { "PCCP" } else { "CPPC" }, shape_names[j]).map_err(|e| e.to_string())?;
            }
        }
        eprintln!("{name}: median ratios {:?}", ratios.iter().map(|r| median(r)).collect::<Vec<_>>());
        results.push((name.clone(), ratios, ok_p, ok_c, notes));
    }

    // Per pair, and the bootstrap over pairs.
    let mut table = std::fs::File::create(out.join("pairs.tsv")).map_err(|e| e.to_string())?;
    writeln!(table, "pair\t{}\trounds_child_faster\tgeomean_speedup\tparent_correct\tchild_correct\tnote", shape_names.iter().map(|s| format!("median_ratio[{s}]")).collect::<Vec<_>>().join("\t")).map_err(|e| e.to_string())?;
    let speed = |r: &[Vec<f64>]| gmean(&r.iter().map(|x| median(x)).collect::<Vec<_>>());
    let timed: Vec<&PairResult> = results.iter().filter(|r| !r.1.is_empty() && r.1.iter().all(|x| !x.is_empty())).collect();
    for (name, r, okp, okc, note) in &results {
        if r.is_empty() || r.iter().any(|x| x.is_empty()) {
            writeln!(table, "{name}\t{}\t-\t-\t{okp}\t{okc}\t{note}", vec!["-"; shape_names.len()].join("\t")).map_err(|e| e.to_string())?;
            continue;
        }
        let faster = (0..rounds).filter(|&k| gmean(&r.iter().map(|x| x[k]).collect::<Vec<_>>()) > 1.0).count();
        writeln!(table, "{name}\t{}\t{faster}/{rounds}\t{:.4}\t{okp}\t{okc}\t{note}", r.iter().map(|x| format!("{:.4}", median(x))).collect::<Vec<_>>().join("\t"), speed(r)).map_err(|e| e.to_string())?;
    }
    let point = gmean(&timed.iter().map(|r| speed(&r.1)).collect::<Vec<_>>());
    let mut boot = Vec::with_capacity(10_000);
    let mut brng = seed ^ 0xB007;
    for _ in 0..10_000 {
        let mut s = Vec::with_capacity(timed.len());
        for _ in 0..timed.len() {
            let p = &timed[(next(&mut brng) % timed.len() as u64) as usize].1;
            let n = p[0].len();
            let picks: Vec<usize> = (0..n).map(|_| (next(&mut brng) % n as u64) as usize).collect();
            let resampled: Vec<Vec<f64>> = p.iter().map(|x| picks.iter().map(|&k| x[k]).collect()).collect();
            s.push(speed(&resampled));
        }
        boot.push(gmean(&s));
    }
    boot.sort_by(|a, b| a.total_cmp(b));
    let (lo, hi) = (boot[249], boot[9_749]);
    let all_correct = results.iter().all(|r| r.2 && r.3);
    let decision = if lo >= keep && all_correct { "KEEP" } else { "DROP" };
    let summary = format!(
        "symmetric half: {} pairs ({} timed), {rounds} ABBA rounds at {shape_names:?}\ngeometric-mean speed-up {point:.4}, cluster-bootstrap 95 % interval [{lo:.4}, {hi:.4}]\nchildren correct on the held-out cases: {} of {}; parents: {} of {}\ndecision (keep if the lower bound >= {keep} and every child correct): {decision}\n",
        results.len(),
        timed.len(),
        results.iter().filter(|r| r.3).count(),
        results.len(),
        results.iter().filter(|r| r.2).count(),
        results.len()
    );
    std::fs::write(out.join("summary.txt"), &summary).map_err(|e| e.to_string())?;
    print!("{summary}");
    Ok(())
}
