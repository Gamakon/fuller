//! The search of `docs/PLAN_wgsl_gram_superopt.md` §3 over the multityped
//! kingdom: a population of typed trees (root `kernel.kernel`), varied by
//! typed subtree crossover and typed subtree mutation, ranked by the HFF
//! angle over the PROBES (`src/wgsl/probes.rs`): 1,024 independent
//! behavioural measures of distance to the correct kernel, each a tiny Φ
//! with a known H; the training probes (seed 1) are selected on, the
//! validation probes (seed 2) are reported. No performance measure is in
//! the angle (Andrew, 2026-10-09: quality first, then performance). The
//! large pairs of §1b judge the winner afterwards. Every evaluation is a line of
//! the ledger. `--seed-shipped` starts from the shipped kernel's tree,
//! `--damaged k` from it mutated k times, the default from random trees,
//! which is the learning proof: the distance must fall to zero.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_gram_evolve -- \
//!       [--pop 32] [--gens 30] [--seed 1] [--sample 4 (training pairs, fixed)] [--seed-shipped | --damaged k] [--out logs/WGSL_GRAM/evolve]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use fuller::geneframe::Ty;
use fuller::wgsl::gram::{check, effective_buffer_limit, gram_shipped_tree_check, Gram, SHIPPED};
use fuller::wgsl::pairs::{splits, Pair};
use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::probes::{probes, run_probes, Probe, PASS};
use fuller::wgsl::oracle::device::Device;
use fuller::wgsl::render::{cost, dual_ty, gram_role, gram_shipped_tree, sample, Renderer, Tree};
use fuller::wgsl::table::FunctionTable;

fn stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    format!("{:02}:{:02}:{:02}Z", (secs / 3600) % 24, (secs / 60) % 60, secs % 60)
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(state: &mut u64) -> f64 {
    (next(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ---- trees: paths, typed crossover and mutation ----

fn paths(t: &Tree, prefix: &mut Vec<usize>, out: &mut Vec<(Vec<usize>, Ty)>) {
    out.push((prefix.clone(), t.ty));
    for (i, k) in t.kids.iter().enumerate() {
        prefix.push(i);
        paths(k, prefix, out);
        prefix.pop();
    }
}

fn at<'a>(t: &'a Tree, path: &[usize]) -> &'a Tree {
    path.iter().fold(t, |n, &i| &n.kids[i])
}

fn replace(t: &mut Tree, path: &[usize], sub: Tree) {
    if path.is_empty() {
        *t = sub;
    } else {
        replace(&mut t.kids[path[0]], &path[1..], sub);
    }
}

fn crossover(a: &Tree, b: &Tree, state: &mut u64) -> Option<Tree> {
    let mut pa = Vec::new();
    paths(a, &mut Vec::new(), &mut pa);
    let mut pb = Vec::new();
    paths(b, &mut Vec::new(), &mut pb);
    // A point in A below the root, and a point in B of the same dual.
    let (path_a, ty) = pa[1 + (next(state) % (pa.len() as u64 - 1).max(1)) as usize].clone();
    let same: Vec<&(Vec<usize>, Ty)> = pb.iter().filter(|(_, t)| *t == ty).collect();
    if same.is_empty() {
        return None;
    }
    let (path_b, _) = same[(next(state) % same.len() as u64) as usize];
    let mut child = a.clone();
    replace(&mut child, &path_a, at(b, path_b).clone());
    Some(child)
}

fn mutate(a: &Tree, kingdom: &WgslKingdom, role: &fuller::wgsl::render::Role, state: &mut u64) -> Result<Tree, String> {
    let mut pa = Vec::new();
    paths(a, &mut Vec::new(), &mut pa);
    let (path, ty) = pa[(next(state) % pa.len() as u64) as usize].clone();
    let mut child = a.clone();
    let node = at(a, &path);
    if node.name.starts_with("literal.") && next(state).is_multiple_of(2) {
        // A constant nudged: real by a factor, a code or index by a step.
        let mut n = node.clone();
        let form = kingdom.dual(ty).map(|d| d.form.clone()).unwrap_or_default();
        n.konst = match form.as_str() {
            "real" => n.konst * (0.5 + unit(state)),
            "flag" | "sign" => 1.0 - n.konst,
            _ => (n.konst + if next(state).is_multiple_of(2) { 1.0 } else { -1.0 }).max(0.0),
        };
        replace(&mut child, &path, n);
    } else {
        let sub = sample(kingdom, role, ty, 1 + (next(state) % 3) as usize, next(state))?;
        replace(&mut child, &path, sub);
    }
    Ok(child)
}

// ---- fitness ----

/// A kernel's probe distances, memoised per tree (deterministic).
#[derive(Clone, Debug, Default)]
struct Score {
    /// Every training and validation probe passed.
    correct: bool,
    /// The training probes' distances, the angle's columns.
    train: Vec<f64>,
    /// Passed counts: training, validation.
    passed_train: usize,
    passed_val: usize,
    /// Summaries for the log: mean distance on train and validation.
    train_mean: f64,
    val_mean: f64,
    angle: f64,
    refused: Option<String>,
}

/// The TrueNorth angle as phylu's `hff.wgsl` computes it (lines 212–213):
/// over scaled columns `x` in [0, 1], `cos θ = 1 − mean(x²)`, `θ = acos`.
/// Zero only when every column is 0; a constant vector c gives
/// `acos(1 − c²)`, so the angle sees magnitude. The earlier formula here
/// (the angle to the diagonal) gave 0 for ANY constant vector and was
/// blind to magnitude; both reviews of `docs/PLAN_hgp_gram.md` found it.
fn angle(x: &[f64]) -> f64 {
    if x.is_empty() {
        return std::f64::consts::FRAC_PI_2;
    }
    let energy = x.iter().map(|v| { let c = v.clamp(0.0, 1.0); c * c }).sum::<f64>();
    let cos = (1.0 - (energy / x.len() as f64).min(1.0)).clamp(-1.0, 1.0);
    if cos > 1.0 - 1.1920929e-7 { 0.0 } else { cos.acos() }
}

/// The work a candidate may do, relative to the shipped kernel: eight
/// times its per-item cost and a base of 20,000 operations; the deadline
/// for one build is 50 times the shipped build on that pair, at least ten
/// seconds (the bound makes a longer build a wedged device, not work).
struct CostBound {
    per_item: f64,
    base: f64,
}

impl CostBound {
    fn deadline_ms(&self, shipped_ms: f64) -> f64 {
        (50.0 * shipped_ms).max(10_000.0)
    }
}

/// The watchdog's deadline as milliseconds since the process started; 0 = disarmed.
static WATCHDOG: std::sync::Mutex<std::sync::atomic::AtomicU64> = std::sync::Mutex::new(std::sync::atomic::AtomicU64::new(0));
static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

fn deadline_at(ms_from_now: f64) -> u64 {
    START.get_or_init(Instant::now).elapsed().as_millis() as u64 + ms_from_now as u64
}

fn start_watchdog() {
    START.get_or_init(Instant::now);
    std::thread::spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let deadline = WATCHDOG.lock().expect("watchdog").load(std::sync::atomic::Ordering::SeqCst);
        if deadline != 0 && START.get().expect("started").elapsed().as_millis() as u64 > deadline {
            eprintln!("[{}] watchdog: a build exceeded its deadline; the device is wedged; exiting (the log's last 'will run' line and the ledger's last pending line name the kernel)", stamp());
            std::process::exit(3);
        }
    });
}

fn arm(ms: f64) {
    WATCHDOG.lock().expect("watchdog").store(deadline_at(ms), std::sync::atomic::Ordering::SeqCst);
}

fn disarm() {
    WATCHDOG.lock().expect("watchdog").store(0, std::sync::atomic::Ordering::SeqCst);
}

/// Each probe distance normalised into [0, 1]: `log10(1 + δ / PASS)` over
/// a fixed maximum of six decades, so a probe exactly right is 0, one
/// within rounding is near 0, and one wrong by a million PASS is 1.
fn columns(s: &Score) -> Vec<f64> {
    s.train.iter().map(|d| ((1.0 + d.min(1e30) / PASS).log10() / 6.0).clamp(0.0, 1.0)).collect()
}

/// The gate on one pair: every cell finite in class, and the largest ratio
/// within τ or within twice the shipped kernel's own largest ratio on that
/// pair, whichever is larger. The second clause was added after the smoke
/// run (2026-10-09): on the duplicate kind the shipped kernel's error is
/// systematic (identical rows), 1.45 τ at d = 2,048, m = 8,192, and the
/// random-walk model behind τ does not hold there.
fn gate(v: &fuller::wgsl::gram::Verdict, shipped_max: f64) -> bool {
    v.non_finite == 0 && v.max_ratio <= (2.0 * shipped_max).max(1.0)
}

struct EvalSet {
    train: Vec<Probe>,
    val: Vec<Probe>,
}

fn score(dev: &Device, rr: &Renderer, tree: &Tree, set: &EvalSet, bound: &CostBound) -> Score {
    let refused = |why: String| Score { refused: Some(why), angle: f64::INFINITY, train: vec![1e30; set.train.len()], train_mean: f64::INFINITY, val_mean: f64::INFINITY, ..Default::default() };
    let c = cost(tree);
    if c.per_item > bound.per_item || c.base > bound.base {
        return refused(format!("cost: {:.0} per item, {:.0} base (bound {:.0}, {:.0})", c.per_item, c.base, bound.per_item, bound.base));
    }
    let out = match rr.kernel(tree) {
        Ok(o) => o,
        Err(e) => return refused(e),
    };
    let gram = match Gram::compile_rendered(dev, &out) {
        Ok(g) => g,
        Err(e) => return refused(format!("compile: {}", e.lines().next().unwrap_or(""))),
    };
    arm(bound.deadline_ms(200.0) * 10.0);
    let train = run_probes(&gram, &set.train);
    let val = train.as_ref().ok().map(|_| run_probes(&gram, &set.val));
    disarm();
    let (train, val) = match (train, val) {
        (Ok(t), Some(Ok(v))) => (t, v),
        (Err(e), _) | (_, Some(Err(e))) => return refused(format!("run: {e}")),
        (Ok(_), None) => unreachable!("val runs when train ran"),
    };
    let passed_train = train.iter().filter(|d| **d <= PASS).count();
    let passed_val = val.iter().filter(|d| **d <= PASS).count();
    let mean = |v: &[f64]| v.iter().map(|d| d.min(1e30)).sum::<f64>() / v.len() as f64;
    Score { correct: passed_train == train.len() && passed_val == val.len(), train_mean: mean(&train), val_mean: mean(&val), train, passed_train, passed_val, angle: 0.0, refused: None }
}

#[allow(dead_code)]
fn timing(dev: &Device, rr: &Renderer, tree: &Tree, shipped: &Gram, train: &[Pair], pairs: &[usize], limit: u64, bound: &CostBound, state: &mut u64) -> Result<(f64, f64), String> {
    let out = rr.kernel(tree)?;
    let gram = Gram::compile_rendered(dev, &out)?;
    let mut ratios = Vec::with_capacity(pairs.len());
    for &i in pairs {
        let p = &train[i];
        let candidate_first = next(state).is_multiple_of(2);
        let build = |g: &Gram| -> Result<f64, String> {
            arm(bound.deadline_ms(2000.0));
            let b = g.build(&p.phi, p.d, p.m, limit);
            disarm();
            Ok(b?.gpu_ms)
        };
        let (tc, ts) = if candidate_first { let tc = build(&gram)?; (tc, build(shipped)?) } else { let ts = build(shipped)?; (build(&gram)?, ts) };
        if ts <= 0.0 || !tc.is_finite() || !ts.is_finite() {
            return Err(format!("times {tc} / {ts} ms on {}", p.name()));
        }
        ratios.push(tc / ts);
    }
    let mut r = ratios.clone();
    r.sort_by(f64::total_cmp);
    let median = r[r.len() / 2];
    Ok((median, (r[r.len() - 1] - r[0]) / median))
}

fn main() -> Result<(), String> {
    let mut pop = 32usize;
    let mut gens = 30usize;
    let mut seed = 1u64;
    let mut sample_n = 4usize;
    let mut seed_shipped = false;
    let mut damaged = 0usize;
    let mut out_dir = PathBuf::from("logs/WGSL_GRAM/evolve");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "--pop" => pop = value("--pop")?.parse().map_err(|e| format!("--pop: {e}"))?,
            "--gens" => gens = value("--gens")?.parse().map_err(|e| format!("--gens: {e}"))?,
            "--seed" => seed = value("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--sample" => sample_n = value("--sample")?.parse().map_err(|e| format!("--sample: {e}"))?,
            "--out" => out_dir = PathBuf::from(value("--out")?),
            "--seed-shipped" => seed_shipped = true,
            "--damaged" => damaged = value("--damaged")?.parse().map_err(|e| format!("--damaged: {e}"))?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let cache = fuller::wgsl::pairs::cache_dir();
    std::fs::create_dir_all(&cache).map_err(|e| format!("{}: {e}", cache.display()))?;
    let t_all = Instant::now();
    println!("[{}] wgsl_gram_evolve pop {pop} gens {gens} seed {seed} sample {sample_n} start {} out {}", stamp(), if seed_shipped { "shipped tree".to_string() } else if damaged > 0 { format!("shipped tree damaged by {damaged} mutations") } else { "random trees".to_string() }, out_dir.display());

    let sp = splits(&cache)?;
    println!("[{}] pairs: {} train, {} validation, {} extrapolation; references built or read in {:.1} s", stamp(), sp.train.len(), sp.val.len(), sp.extrap.len(), sp.seconds);
    let (train, val, extrap) = (sp.train, sp.val, sp.extrap);
    let kingdom = WgslKingdom::load();
    let table = FunctionTable::shipped();
    let role = gram_role();
    let rr = Renderer::new(&kingdom, &table, &role);
    let root = dual_ty(&kingdom, "kernel.kernel")?;
    let dev = Device::new()?;
    let limit = effective_buffer_limit(&dev.device);
    println!("[{}] device {} limit {} MiB", stamp(), dev.name, limit >> 20);
    let shipped_tree = gram_shipped_tree(&kingdom)?;
    gram_shipped_tree_check(&shipped_tree)?;
    let shipped_cost = cost(&shipped_tree);
    let bound = CostBound { per_item: 8.0 * shipped_cost.per_item, base: 20_000.0 };
    println!("[{}] shipped cost {:.0} per item, {:.0} base; bound {:.0} per item, {:.0} base; deadline 50x the shipped build, at least 10 s", stamp(), shipped_cost.per_item, shipped_cost.base, bound.per_item, bound.base);
    start_watchdog();
    let shipped_gram = Gram::compile(&dev, SHIPPED)?;

    let mut ledger = std::fs::File::create(out_dir.join("ledger.tsv")).map_err(|e| format!("ledger: {e}"))?;
    writeln!(ledger, "gen\tid\torigin\tnodes\tcorrect\tpassed_train\tpassed_val\ttrain_mean\tval_mean\tangle\trefused\tsexpr").map_err(|e| e.to_string())?;

    // The population: random trees (the learning proof), or the shipped
    // tree and its mutants, or a damaged shipped tree and its mutants.
    let mut state = seed ^ 0x2545_F491_4F6C_DD1D;
    let mut population: Vec<(Tree, String)> = Vec::new();
    if seed_shipped {
        population.push((shipped_tree.clone(), "shipped".into()));
    }
    let start = if damaged > 0 {
        let mut t = shipped_tree.clone();
        for _ in 0..damaged {
            t = mutate(&t, &kingdom, &role, &mut state)?;
        }
        println!("[{}] damaged start: {}", stamp(), t.to_sexpr());
        Some(t)
    } else if seed_shipped {
        Some(shipped_tree.clone())
    } else {
        None
    };
    while population.len() < pop {
        let (t, origin) = match &start {
            Some(base) if population.len() < pop / 2 => (mutate(base, &kingdom, &role, &mut state)?, "mutant-of-start".to_string()),
            _ => (sample(&kingdom, &role, root, 6, next(&mut state))?, "random".to_string()),
        };
        if t.size() <= 128 {
            population.push((t, origin));
        }
    }

    // The probes: training (selected on) and validation (reported).
    let set = EvalSet { train: probes(1), val: probes(2) };
    println!("[{}] probes: {} training, {} validation", stamp(), set.train.len(), set.val.len());
    let t = Instant::now();
    let shipped_score = score(&dev, &rr, &shipped_tree, &set, &bound);
    println!("[{}] the shipped tree: passes {}/{} training and {}/{} validation probes in {:.1} s (correct {})", stamp(), shipped_score.passed_train, set.train.len(), shipped_score.passed_val, set.val.len(), t.elapsed().as_secs_f64(), shipped_score.correct);
    let mut memo: BTreeMap<String, Score> = BTreeMap::new();
    let mut best: Option<(f64, Tree, Score)> = None;
    for gen in 0..gens {
        let t_gen = Instant::now();
        let mut scores: Vec<Score> = vec![Score::default(); population.len()];
        let mut order: Vec<usize> = (0..population.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, (next(&mut state) % (i as u64 + 1)) as usize);
        }
        let (mut memo_hits, mut built) = (0, 0);
        for &i in &order {
            let (tree, origin) = &population[i];
            let key = tree.to_sexpr();
            writeln!(ledger, "{gen}\t{i}\t{origin}\t{}\tpending\t-\t-\t-\t-\t-\t-\t{key}", tree.size()).map_err(|e| e.to_string())?;
            ledger.flush().map_err(|e| e.to_string())?;
            let t = Instant::now();
            let s = match memo.get(&key) {
                Some(a) => {
                    memo_hits += 1;
                    a.clone()
                }
                None => {
                    let c = cost(tree);
                    println!("[{}] gen {gen:>2} id {i:>2} will run: {origin}, {} nodes, cost {:.0}/item {:.0} base; {}", stamp(), tree.size(), c.per_item, c.base, &key[..key.len().min(140)]);
                    built += 1;
                    let a = score(&dev, &rr, tree, &set, &bound);
                    memo.insert(key.clone(), a.clone());
                    println!("[{}] gen {gen:>2} id {i:>2} ran in {:.2} s: {}", stamp(), t.elapsed().as_secs_f64(), match &a.refused { Some(r) => format!("refused ({r})"), None => format!("passes {}/{} train, {}/{} val; mean distance train {:.2e} val {:.2e}", a.passed_train, set.train.len(), a.passed_val, set.val.len(), a.train_mean, a.val_mean) });
                    a
                }
            };
            scores[i] = s;
        }
        for s in scores.iter_mut() {
            s.angle = if s.refused.is_some() { f64::INFINITY } else { angle(&columns(s)) };
        }
        for (i, (tree, origin)) in population.iter().enumerate() {
            let s = &scores[i];
            writeln!(ledger, "{gen}\t{i}\t{origin}\t{}\t{}\t{}\t{}\t{:.4e}\t{:.4e}\t{:.5}\t{}\t{}", tree.size(), s.correct, s.passed_train, s.passed_val, s.train_mean, s.val_mean, s.angle, s.refused.as_deref().unwrap_or("-"), tree.to_sexpr()).map_err(|e| e.to_string())?;
        }
        ledger.flush().map_err(|e| e.to_string())?;
        let mut ranked: Vec<usize> = (0..population.len()).collect();
        ranked.sort_by(|&a, &b| scores[a].angle.total_cmp(&scores[b].angle).then(a.cmp(&b)));
        let correct = scores.iter().filter(|s| s.correct).count();
        let refused = scores.iter().filter(|s| s.refused.is_some()).count();
        let top = ranked[0];
        let finite: Vec<f64> = ranked.iter().map(|&i| scores[i].angle).filter(|a| a.is_finite()).collect();
        let median = finite.get(finite.len() / 2).copied().unwrap_or(f64::INFINITY);
        let b = &scores[top];
        let median_passed = { let mut v: Vec<usize> = scores.iter().filter(|s| s.refused.is_none()).map(|s| s.passed_train).collect(); v.sort_unstable(); v.get(v.len() / 2).copied().unwrap_or(0) };
        println!("[{}] gen {gen:>2}: correct {correct}/{}, refused {refused}, memoised {memo_hits}, built {built}; best fitness {:.4} (passes {}/{} train, {}/{} val), median fitness {:.4}, median passes {}; best {} nodes, {}; {:.1} s", stamp(), population.len(), b.angle, b.passed_train, set.train.len(), b.passed_val, set.val.len(), median, median_passed, population[top].0.size(), population[top].1, t_gen.elapsed().as_secs_f64());
        if b.refused.is_none() && best.as_ref().is_none_or(|x| b.angle < x.0) {
            best = Some((b.angle, population[top].0.clone(), b.clone()));
            let out = rr.kernel(&population[top].0)?;
            std::fs::write(out_dir.join(format!("best_g{gen}.wgsl")), &out.text).map_err(|e| e.to_string())?;
        }
        if gen + 1 == gens {
            break;
        }
        // The next generation: 4 elites, the rest by tournament of 3 with crossover then mutation.
        let mut next_pop: Vec<(Tree, String)> = ranked[..4.min(ranked.len())].iter().map(|&i| (population[i].0.clone(), format!("elite({i})"))).collect();
        let tournament = |state: &mut u64| -> usize {
            let mut b = (next(state) % population.len() as u64) as usize;
            for _ in 0..2 {
                let c = (next(state) % population.len() as u64) as usize;
                if scores[c].angle < scores[b].angle {
                    b = c;
                }
            }
            b
        };
        let mut tries = 0;
        while next_pop.len() < pop && tries < pop * 20 {
            tries += 1;
            let a = tournament(&mut state);
            let child = if next(&mut state).is_multiple_of(2) {
                let b = tournament(&mut state);
                match crossover(&population[a].0, &population[b].0, &mut state) {
                    Some(c) => (c, format!("x({a},{b})")),
                    None => continue,
                }
            } else {
                (mutate(&population[a].0, &kingdom, &role, &mut state)?, format!("m({a})"))
            };
            let (mut tree, mut origin) = child;
            if next(&mut state).is_multiple_of(4) {
                tree = mutate(&tree, &kingdom, &role, &mut state)?;
                origin.push_str("+m");
            }
            if tree.size() <= 128 {
                next_pop.push((tree, origin));
            }
        }
        population = next_pop;
    }

    // The elites on every pair of every split, five paired rounds for time on three train pairs.
    let Some((best_angle, best_tree, best_score)) = best else {
        println!("[{}] no scored kernel in any generation", stamp());
        return Ok(());
    };
    println!("[{}] search done in {:.0} s; best fitness {best_angle:.4}, correct {}: passes {}/{} train, {}/{} val", stamp(), t_all.elapsed().as_secs_f64(), best_score.correct, best_score.passed_train, set.train.len(), best_score.passed_val, set.val.len());
    if !best_score.correct {
        println!("[{}] the best kernel does not pass every probe: no split scoring and no timing", stamp());
        let out = rr.kernel(&best_tree)?;
        std::fs::write(out_dir.join("best.wgsl"), &out.text).map_err(|e| e.to_string())?;
        std::fs::write(out_dir.join("best.sexpr"), best_tree.to_sexpr()).map_err(|e| e.to_string())?;
        return Ok(());
    }
    let out = rr.kernel(&best_tree)?;
    std::fs::write(out_dir.join("best.wgsl"), &out.text).map_err(|e| e.to_string())?;
    std::fs::write(out_dir.join("best.sexpr"), best_tree.to_sexpr()).map_err(|e| e.to_string())?;
    let best_gram = Gram::compile_rendered(&dev, &out)?;
    let mut report = String::new();
    writeln!(report, "split\tpairs\tgated\tmean_ratio\tmax_ratio\tshipped_mean\tshipped_max").map_err(|e| e.to_string())?;
    for (name, pairs) in [("train", &train), ("validation", &val), ("extrapolation", &extrap)] {
        let t = Instant::now();
        let (mut gated, mut mean, mut max, mut smean, mut smax) = (0, 0.0, 0.0f64, 0.0, 0.0f64);
        for p in pairs.iter() {
            let b = best_gram.build(&p.phi, p.d, p.m, limit)?;
            let v = check(&b.h, &p.h_ref, p.d, p.m);
            let sb = shipped_gram.build(&p.phi, p.d, p.m, limit)?;
            let sv = check(&sb.h, &p.h_ref, p.d, p.m);
            gated += usize::from(gate(&v, sv.max_ratio));
            mean += v.mean_ratio / pairs.len() as f64;
            max = max.max(v.max_ratio);
            smean += sv.mean_ratio / pairs.len() as f64;
            smax = smax.max(sv.max_ratio);
            if !gate(&v, sv.max_ratio) {
                println!("  {name} {}: FAIL max ratio {:.3} (shipped {:.3}) non-finite {}", p.name(), v.max_ratio, sv.max_ratio, v.non_finite);
            }
        }
        writeln!(report, "{name}\t{}\t{gated}\t{mean:.3e}\t{max:.3e}\t{smean:.3e}\t{smax:.3e}", pairs.len()).map_err(|e| e.to_string())?;
        println!("[{}] {name}: {gated}/{} gated, mean {mean:.2e} max {max:.2e} (shipped {smean:.2e} / {smax:.2e}); {:.0} s", stamp(), pairs.len(), t.elapsed().as_secs_f64());
    }
    // Time: five rounds, randomised, on the three largest training pairs.
    let mut big: Vec<&Pair> = train.iter().collect();
    big.sort_by_key(|p| std::cmp::Reverse(p.d * p.d * p.m));
    let mut lines = BTreeMap::new();
    for p in big.iter().take(3) {
        let mut ratios = Vec::new();
        for _ in 0..5 {
            let first_best = next(&mut state).is_multiple_of(2);
            let (tb, ts) = if first_best {
                let tb = best_gram.build(&p.phi, p.d, p.m, limit)?.gpu_ms;
                (tb, shipped_gram.build(&p.phi, p.d, p.m, limit)?.gpu_ms)
            } else {
                let ts = shipped_gram.build(&p.phi, p.d, p.m, limit)?.gpu_ms;
                (best_gram.build(&p.phi, p.d, p.m, limit)?.gpu_ms, ts)
            };
            ratios.push((tb, ts));
        }
        let mut r: Vec<f64> = ratios.iter().map(|(b, s)| b / s).collect();
        r.sort_by(f64::total_cmp);
        lines.insert(p.name(), format!("{} best {:.2} ms shipped {:.2} ms ratio median {:.3} (min {:.3} max {:.3})", p.name(), ratios.iter().map(|x| x.0).sum::<f64>() / 5.0, ratios.iter().map(|x| x.1).sum::<f64>() / 5.0, r[2], r[0], r[4]));
    }
    for l in lines.values() {
        println!("  time: {l}");
        writeln!(report, "time\t{l}").map_err(|e| e.to_string())?;
    }
    std::fs::write(out_dir.join("report.tsv"), &report).map_err(|e| e.to_string())?;
    println!("[{}] done in {:.0} s; {}", stamp(), t_all.elapsed().as_secs_f64(), out_dir.join("report.tsv").display());
    Ok(())
}
