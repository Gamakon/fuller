//! The search of `docs/PLAN_wgsl_gram_superopt.md` §3 over the multityped
//! kingdom: a population of typed trees (root `kernel.kernel`), seeded with
//! the shipped Gram kernel and random trees, varied by typed subtree
//! crossover and typed subtree mutation, gated by the criterion on a sample
//! of training pairs each generation, ranked by the HFF angle over the
//! §1b columns (ShapeDistance mean and max on the sample, the paired GPU
//! time ratio against the shipped kernel, and its spread). Every evaluation
//! is a line of the ledger. At the end the elites are scored on every
//! training, validation and extrapolation pair.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_gram_evolve -- \
//!       [--pop 32] [--gens 30] [--seed 1] [--sample 5 (timing pairs, fixed for the run)] [--out logs/WGSL_GRAM/evolve]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use fuller::geneframe::Ty;
use fuller::wgsl::gram::{check, effective_buffer_limit, gram_shipped_tree_check, phi_of, reference, Gram, Kind, SHIPPED};
use fuller::wgsl::loader::WgslKingdom;
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

/// One input–output pair: a kind of Φ at a shape, with its f64 reference.
struct Pair {
    kind: Kind,
    d: usize,
    m: usize,
    seed: u64,
    phi: Vec<f32>,
    h_ref: Vec<f64>,
}

impl Pair {
    fn name(&self) -> String {
        format!("{}_d{}_m{}_s{}", self.kind.name(), self.d, self.m, self.seed)
    }

    fn load(kind: Kind, d: usize, m: usize, seed: u64, cache: &Path) -> Result<Pair, String> {
        let phi = phi_of(kind, d, m, seed);
        let path = cache.join(format!("ref_{}_d{d}_m{m}_s{seed}.f64", kind.name()));
        let h_ref = if path.exists() {
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            if bytes.len() != d * d * 8 {
                return Err(format!("{}: {} bytes", path.display(), bytes.len()));
            }
            bytes.chunks(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect()
        } else {
            let h = reference(&phi, d, m);
            std::fs::write(&path, h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| format!("{}: {e}", path.display()))?;
            h
        };
        Ok(Pair { kind, d, m, seed, phi, h_ref })
    }
}

/// The splits of §1b: 47 generated training pairs (the three Blue World
/// exports are not available and are not simulated), 20 validation, 20
/// extrapolation at d = 8,192.
type Splits = (Vec<Pair>, Vec<Pair>, Vec<Pair>);

fn splits(cache: &Path) -> Result<Splits, String> {
    let t = Instant::now();
    let mut train = Vec::new();
    let shapes = [(512, 2048), (1024, 4096), (2048, 8192), (1024, 16384)];
    let counts: [(Kind, usize); 11] = [(Kind::Gaussian, 6), (Kind::Weighted, 6), (Kind::LowRank, 4), (Kind::Sparse, 4), (Kind::HeavyTailed, 4), (Kind::Cancelling, 4), (Kind::Integer, 4), (Kind::Extreme, 4), (Kind::NonFinite, 4), (Kind::Duplicate, 3), (Kind::Ragged, 4)];
    let mut seed = 100u64;
    for (kind, n) in counts {
        for i in 0..n {
            let (d, m) = if kind == Kind::Ragged { [(520, 4097), (1000, 3001), (2040, 5000), (512, 6151)][i % 4] } else { shapes[i % shapes.len()] };
            train.push(Pair::load(kind, d, m, seed, cache)?);
            seed += 1;
        }
    }
    let mut val = Vec::new();
    for (i, (kind, _)) in counts.iter().cycle().take(20).enumerate() {
        let (d, m) = if *kind == Kind::Ragged { (1032, 2049) } else { shapes[(i + 1) % shapes.len()] };
        val.push(Pair::load(*kind, d, m, 1000 + i as u64, cache)?);
    }
    let mut extrap = Vec::new();
    for (i, (kind, _)) in counts.iter().cycle().take(20).enumerate() {
        let m = [1024, 2048, 3000, 4096][i % 4];
        extrap.push(Pair::load(*kind, 8192, m, 2000 + i as u64, cache)?);
    }
    println!("[{}] pairs: {} train, {} validation, {} extrapolation; references built or read in {:.1} s", stamp(), train.len(), val.len(), extrap.len(), t.elapsed().as_secs_f64());
    Ok((train, val, extrap))
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
            "real" => n.konst * (0.5 + unit(state) as f32),
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

#[derive(Clone, Debug, Default)]
struct Score {
    gated: bool,
    mean_ratio: f64,
    max_ratio: f64,
    time_ratio: f64,
    spread: f64,
    angle: f64,
    refused: Option<String>,
}

/// Accuracy on the fixed evaluation set, memoised per tree (the set and
/// the device are deterministic, so one measurement is the measurement).
#[derive(Clone, Debug)]
struct Accuracy {
    gated: bool,
    mean_ratio: f64,
    max_ratio: f64,
    refused: Option<String>,
}

/// The TrueNorth angle over normalised columns `x` (each in [0, 1], 0 best):
/// the angle between the complements and the diagonal.
fn angle(x: &[f64]) -> f64 {
    let comp: Vec<f64> = x.iter().map(|v| 1.0 - v.clamp(0.0, 1.0)).collect();
    let norm = comp.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 {
        return std::f64::consts::FRAC_PI_2;
    }
    (comp.iter().sum::<f64>() / ((x.len() as f64).sqrt() * norm)).clamp(-1.0, 1.0).acos()
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

struct Caps {
    mean: f64,
    max: f64,
    spread: f64,
    /// The frozen column maxima of `log10(1 + v / c)`.
    col_max: [f64; 4],
}

fn columns(s: &Score, caps: &Caps) -> [f64; 4] {
    let v = [s.mean_ratio / caps.mean, s.max_ratio / caps.max, s.time_ratio, s.spread / caps.spread];
    let mut x = [0.0; 4];
    for i in 0..4 {
        x[i] = ((1.0 + v[i]).log10() / caps.col_max[i]).clamp(0.0, 1.0);
    }
    x
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

/// The fixed evaluation set: the gate pair (the cheapest training pair,
/// where a broken kernel fails in milliseconds) and `n` timing pairs of at
/// least 1.5e10 multiply-adds, chosen once per run; the shipped kernel's
/// own accuracy and time on each.
struct EvalSet {
    gate_pair: usize,
    pairs: Vec<usize>,
    /// Per training pair index: the shipped kernel's largest ratio (the gate's second clause).
    shipped_max: Vec<f64>,
    shipped_mean: Vec<f64>,
}

impl EvalSet {
    fn choose(train: &[Pair], n: usize, state: &mut u64, shipped: &Gram, limit: u64) -> Result<EvalSet, String> {
        let gate_pair = (0..train.len()).min_by_key(|&i| train[i].d * train[i].d * train[i].m).expect("pairs");
        let mut idx: Vec<usize> = (0..train.len()).filter(|&i| (train[i].d * train[i].d * train[i].m) as f64 >= 1.5e10).collect();
        for i in (1..idx.len()).rev() {
            idx.swap(i, (next(state) % (i as u64 + 1)) as usize);
        }
        idx.truncate(n);
        let mut shipped_max = vec![0.0; train.len()];
        let mut shipped_mean = vec![0.0; train.len()];
        for &i in idx.iter().chain(std::iter::once(&gate_pair)) {
            let p = &train[i];
            let b = shipped.build(&p.phi, p.d, p.m, limit)?;
            let v = check(&b.h, &p.h_ref, p.d, p.m);
            shipped_max[i] = v.max_ratio;
            shipped_mean[i] = v.mean_ratio;
        }
        Ok(EvalSet { gate_pair, pairs: idx, shipped_max, shipped_mean })
    }
}

/// Accuracy of one kernel: the gate pair first, then every timing pair;
/// the mean ratio over the set, the largest, and whether every pair gates.
fn accuracy(dev: &Device, rr: &Renderer, tree: &Tree, train: &[Pair], set: &EvalSet, limit: u64, bound: &CostBound, shipped_ms: f64) -> Accuracy {
    let c = cost(tree);
    if c.per_item > bound.per_item || c.base > bound.base {
        return Accuracy { gated: false, mean_ratio: f64::INFINITY, max_ratio: f64::INFINITY, refused: Some(format!("cost: {:.0} per item, {:.0} base (bound {:.0}, {:.0})", c.per_item, c.base, bound.per_item, bound.base)) };
    }
    let out = match rr.kernel(tree) {
        Ok(o) => o,
        Err(e) => return Accuracy { gated: false, mean_ratio: f64::INFINITY, max_ratio: f64::INFINITY, refused: Some(e) },
    };
    let gram = match Gram::compile_rendered(dev, &out) {
        Ok(g) => g,
        Err(e) => return Accuracy { gated: false, mean_ratio: f64::INFINITY, max_ratio: f64::INFINITY, refused: Some(format!("compile: {}", e.lines().next().unwrap_or(""))) },
    };
    let (mut mean, mut max, mut gated) = (0.0, 0.0f64, true);
    let order: Vec<usize> = std::iter::once(set.gate_pair).chain(set.pairs.iter().copied()).collect();
    for (k, &i) in order.iter().enumerate() {
        let p = &train[i];
        arm(bound.deadline_ms(shipped_ms));
        let b = gram.build(&p.phi, p.d, p.m, limit);
        disarm();
        let b = match b {
            Ok(b) => b,
            Err(e) => return Accuracy { gated: false, mean_ratio: f64::INFINITY, max_ratio: f64::INFINITY, refused: Some(format!("run: {e}")) },
        };
        let v = check(&b.h, &p.h_ref, p.d, p.m);
        let ok = gate(&v, set.shipped_max[i]);
        gated &= ok;
        max = max.max(v.max_ratio);
        mean += v.mean_ratio / order.len() as f64;
        // A kernel that fails the gate pair is not built on the rest: its
        // mean is the gate pair's, which still ranks the failures.
        if k == 0 && !ok {
            return Accuracy { gated: false, mean_ratio: v.mean_ratio, max_ratio: v.max_ratio, refused: None };
        }
    }
    Accuracy { gated, mean_ratio: mean, max_ratio: max, refused: None }
}

/// Paired timing of one gated kernel: on every timing pair, the shipped
/// kernel and the candidate built back to back in a random order; the
/// median ratio over the pairs and the spread.
fn timing(dev: &Device, rr: &Renderer, tree: &Tree, shipped: &Gram, train: &[Pair], set: &EvalSet, limit: u64, bound: &CostBound, state: &mut u64) -> Result<(f64, f64), String> {
    let out = rr.kernel(tree)?;
    let gram = Gram::compile_rendered(dev, &out)?;
    let mut ratios = Vec::with_capacity(set.pairs.len());
    for &i in &set.pairs {
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
    let mut sample_n = 5usize;
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
            other => return Err(format!("unknown argument {other}")),
        }
    }
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let cache = PathBuf::from("logs/WGSL_GRAM/pairs");
    std::fs::create_dir_all(&cache).map_err(|e| format!("{}: {e}", cache.display()))?;
    let t_all = Instant::now();
    println!("[{}] wgsl_gram_evolve pop {pop} gens {gens} seed {seed} sample {sample_n} out {}", stamp(), out_dir.display());

    let (train, val, extrap) = splits(&cache)?;
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
    writeln!(ledger, "gen\tid\torigin\tnodes\tgated\tmean_ratio\tmax_ratio\ttime_ratio\tspread\tangle\trefused\tsexpr").map_err(|e| e.to_string())?;

    // The population: the shipped tree, its mutants, and random trees.
    let mut state = seed ^ 0x2545_F491_4F6C_DD1D;
    let mut population: Vec<(Tree, String)> = vec![(shipped_tree.clone(), "shipped".into())];
    while population.len() < pop {
        let origin;
        let t = if population.len() < pop / 2 {
            origin = "mutant-of-shipped".to_string();
            mutate(&shipped_tree, &kingdom, &role, &mut state)?
        } else {
            origin = "random".to_string();
            sample(&kingdom, &role, root, 6, next(&mut state))?
        };
        if t.size() <= 128 {
            population.push((t, origin));
        }
    }

    // The fixed evaluation set for the whole run.
    let set = EvalSet::choose(&train, sample_n, &mut state, &shipped_gram, limit)?;
    println!("[{}] evaluation set: gate pair {}; timing pairs {}", stamp(), train[set.gate_pair].name(), set.pairs.iter().map(|&i| format!("{} (shipped max ratio {:.3})", train[i].name(), set.shipped_max[i])).collect::<Vec<_>>().join(", "));
    let ship_mean = set.pairs.iter().chain(std::iter::once(&set.gate_pair)).map(|&i| set.shipped_mean[i]).sum::<f64>() / (set.pairs.len() + 1) as f64;
    let ship_max = set.pairs.iter().chain(std::iter::once(&set.gate_pair)).map(|&i| set.shipped_max[i]).fold(0.0, f64::max);
    let gate_ms = { let p = &train[set.gate_pair]; shipped_gram.build(&p.phi, p.d, p.m, limit)?.gpu_ms };
    let mut memo: BTreeMap<String, Accuracy> = BTreeMap::new();
    let mut caps: Option<Caps> = None;
    let mut best: Option<(f64, Tree, Score)> = None;
    for gen in 0..gens {
        let t_gen = Instant::now();
        // The control: the shipped tree against the shipped text, the noise floor of this generation.
        let (control_ratio, control_spread) = timing(&dev, &rr, &shipped_tree, &shipped_gram, &train, &set, limit, &bound, &mut state)?;
        let mut scores: Vec<Score> = vec![Score::default(); population.len()];
        let mut order: Vec<usize> = (0..population.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, (next(&mut state) % (i as u64 + 1)) as usize);
        }
        let (mut memo_hits, mut gate_builds) = (0, 0);
        for &i in &order {
            let (tree, origin) = &population[i];
            let key = tree.to_sexpr();
            writeln!(ledger, "{gen}\t{i}\t{origin}\t{}\tpending\t-\t-\t-\t-\t-\t-\t{key}", tree.size()).map_err(|e| e.to_string())?;
            ledger.flush().map_err(|e| e.to_string())?;
            let c = cost(tree);
            let t = Instant::now();
            let acc = match memo.get(&key) {
                Some(a) => {
                    memo_hits += 1;
                    a.clone()
                }
                None => {
                    println!("[{}] gen {gen:>2} id {i:>2} will run: {origin}, {} nodes, cost {:.0}/item {:.0} base; {}", stamp(), tree.size(), c.per_item, c.base, &key[..key.len().min(140)]);
                    gate_builds += 1;
                    let a = accuracy(&dev, &rr, tree, &train, &set, limit, &bound, gate_ms);
                    memo.insert(key.clone(), a.clone());
                    a
                }
            };
            let mut s = Score { gated: acc.gated, mean_ratio: acc.mean_ratio, max_ratio: acc.max_ratio, refused: acc.refused.clone(), ..Default::default() };
            if s.gated {
                let (ratio, spread) = timing(&dev, &rr, tree, &shipped_gram, &train, &set, limit, &bound, &mut state)?;
                s.time_ratio = ratio;
                s.spread = spread;
            }
            println!("[{}] gen {gen:>2} id {i:>2} ran in {:.2} s: {}", stamp(), t.elapsed().as_secs_f64(), match &s.refused { Some(r) => format!("refused ({r})"), None => format!("gated {}, mean {:.2e}, max {:.2e}, time {:.3}x, spread {:.3}", s.gated, s.mean_ratio, s.max_ratio, s.time_ratio, s.spread) });
            scores[i] = s;
        }
        if caps.is_none() {
            let mut col_max = [2f64.log10(); 4];
            for s in scores.iter().filter(|s| s.gated) {
                let v = [s.mean_ratio / ship_mean.max(1e-12), s.max_ratio / ship_max.max(1e-12), s.time_ratio, s.spread / 0.12];
                for i in 0..4 {
                    col_max[i] = col_max[i].max((1.0 + v[i]).log10());
                }
            }
            caps = Some(Caps { mean: ship_mean.max(1e-12), max: ship_max.max(1e-12), spread: 0.12, col_max });
        }
        let c = caps.as_ref().expect("set");
        for s in scores.iter_mut() {
            if s.refused.is_some() {
                s.angle = f64::INFINITY;
                continue;
            }
            s.angle = if s.gated { angle(&columns(s, c)) } else { std::f64::consts::FRAC_PI_2 + (1.0 + s.mean_ratio).log10().min(100.0) };
        }
        for (i, (tree, origin)) in population.iter().enumerate() {
            let s = &scores[i];
            writeln!(ledger, "{gen}\t{i}\t{origin}\t{}\t{}\t{:.4e}\t{:.4e}\t{:.4}\t{:.4}\t{:.5}\t{}\t{}", tree.size(), s.gated, s.mean_ratio, s.max_ratio, s.time_ratio, s.spread, s.angle, s.refused.as_deref().unwrap_or("-"), tree.to_sexpr()).map_err(|e| e.to_string())?;
        }
        ledger.flush().map_err(|e| e.to_string())?;
        let mut ranked: Vec<usize> = (0..population.len()).collect();
        ranked.sort_by(|&a, &b| scores[a].angle.total_cmp(&scores[b].angle).then(a.cmp(&b)));
        let gated = scores.iter().filter(|s| s.gated).count();
        let refused = scores.iter().filter(|s| s.refused.is_some()).count();
        let top = ranked[0];
        let faster = scores.iter().filter(|s| s.gated && s.time_ratio < control_ratio - 2.0 * control_spread.max(0.02)).count();
        println!("[{}] gen {gen:>2}: {gated}/{} gated, {refused} refused, {memo_hits} memoised, {gate_builds} built; control (shipped tree vs text) {control_ratio:.3}x spread {control_spread:.3}; {faster} gated faster than the control by 2 spreads; best angle {:.4} time {:.3}x mean {:.2e} max {:.2e} ({} nodes, {}); {:.1} s", stamp(), population.len(), scores[top].angle, scores[top].time_ratio, scores[top].mean_ratio, scores[top].max_ratio, population[top].0.size(), population[top].1, t_gen.elapsed().as_secs_f64());
        if scores[top].gated && best.as_ref().is_none_or(|b| scores[top].angle < b.0) {
            best = Some((scores[top].angle, population[top].0.clone(), scores[top].clone()));
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
        println!("[{}] no gated kernel in any generation", stamp());
        return Ok(());
    };
    println!("[{}] search done in {:.0} s; best angle {best_angle:.4}: {:?}", stamp(), t_all.elapsed().as_secs_f64(), best_score);
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
