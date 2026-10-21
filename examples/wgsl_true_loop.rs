//! ST3: the compiler loop of the true multi-typed WGSL kingdom
//! (`docs/PLAN_true_gep_wgsl.md`). Random genes from the host typed initialiser
//! are expressed by the typed walk, rendered by substitution into the rows'
//! spellings, and put through, in order:
//!   (a) naga 30 parse and validate for the installed target;
//!   (b) the Metal compute pipeline, in the Internal and Validation error scopes;
//!   (c) one run on the Gram unit test's inputs through the harness binding.
//! A failing gene is never filtered, repaired or skipped: it is recorded with its
//! kernel, its error, and the smallest subtree that still fails (every other
//! subtree replaced by a terminal of its type) so the rows responsible are named.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_true_loop -- \
//!       --genes 100 --seed 1 --size 8 --stage all --out /private/tmp/claude-501/st_runs/x
//! `--stage naga` runs (a) only (no device); `--dump N` prints gene N's kernel.
//!
//! Stage (c) is the smallest dispatch that runs the Gram unit test: d = 16, m = 4 (UNIT_TEST_D, UNIT_TEST_M), one
//! 16 x 16 workgroup of 256 invocations in one pass. Every submit is bounded before it is made by the work guard
//! (`fuller::wgsl::work_guard`): H * 2^(6K) * invocations must be at most the cap, else the tool stops with the
//! bound, the cap and what to change (2026-10-21 17:33: K = 2 at the 128 x 64 rung crashed the machine).

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use fuller::kingdom_contract::KingdomContract;
use fuller::typed_gep::{express, GeneLayout, Tree};
use fuller::wgsl::gram::{band_layout, phi, submit_invocations, Gram, RUNG_SEED};
use fuller::wgsl::oracle::device::{Device, Fuel};
use fuller::wgsl::true_kingdom::{default_dir, random_gene, Rng, Share, TrueKingdom};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Naga,
    Pipeline,
    Run,
}

type Verdict = Result<(), (Stage, String)>;

#[derive(Default)]
struct Timing {
    naga_ms: Cell<f64>,
    pipeline_ms: Cell<f64>,
    run_ms: Cell<f64>,
    slowest_run_ms: Cell<f64>,
}

/// Stage (c)'s dispatch: the smallest that runs the Gram unit test, d = 16 columns and m = 4 items, one 16 x 16
/// workgroup (256 invocations) in one pass.
const UNIT_TEST_D: usize = 16;
const UNIT_TEST_M: usize = 4;
const UNIT_TEST_INVOCATIONS: u64 = 16 * 16;

struct Ctx<'a> {
    timing: Timing,
    kingdom: &'a TrueKingdom,
    contract: &'a KingdomContract,
    dev: Option<&'a Device>,
    phi: Vec<f32>,
    d: usize,
    m: usize,
    limit: u64,
    fuel: Fuel,
}

impl Ctx<'_> {
    fn check_source(&self, src: &str) -> Verdict {
        let t = Instant::now();
        let module = naga::front::wgsl::parse_str(src).map_err(|e| (Stage::Naga, e.emit_to_string(src)))?;
        fuller::wgsl::target::Target::current().validate(&module, src).map_err(|e| (Stage::Naga, e))?;
        self.timing.naga_ms.set(self.timing.naga_ms.get() + t.elapsed().as_secs_f64() * 1e3);
        let Some(dev) = self.dev else { return Ok(()) };
        let t = Instant::now();
        let gram = Gram::compile(dev, src, self.fuel.clone()).map_err(|e| (Stage::Pipeline, e))?;
        // The work guard, before the run: a bound over the cap is a misconfigured experiment, not a verdict on
        // this kernel, so the tool stops (nothing was submitted).
        let bands = band_layout(self.d, self.m, self.limit);
        if let Err(e) = self.fuel.check("stage (c)", submit_invocations(gram.grid, self.d, bands.band_rows.min(self.d), bands.item_bands)) {
            eprintln!("{e}");
            std::process::exit(4);
        }
        self.timing.pipeline_ms.set(self.timing.pipeline_ms.get() + t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        let built = gram.build(&self.phi, self.d, self.m, self.limit).map_err(|e| (Stage::Run, e))?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        self.timing.run_ms.set(self.timing.run_ms.get() + ms);
        self.timing.slowest_run_ms.set(self.timing.slowest_run_ms.get().max(ms));
        if built.h.len() != self.d * self.d {
            return Err((Stage::Run, format!("the run returned {} values, not d x d = {}", built.h.len(), self.d * self.d)));
        }
        Ok(())
    }

    fn check_tree(&self, t: &Tree) -> Verdict {
        let src = self.kingdom.render(t).map_err(|e| (Stage::Naga, e))?;
        self.check_source(&src)
    }
}

/// The error with its numbers, identifiers and source excerpt taken out, so equal faults group.
fn signature(stage: Stage, msg: &str) -> String {
    let mut keep: Vec<String> = Vec::new();
    for line in msg.lines() {
        let t = line.trim();
        let take = t.starts_with("error:") || t.starts_with("our software") || t.starts_with("pipeline") || t.starts_with("validation") || t.contains("^^^") || t.contains("Error") || t.contains("error");
        if take {
            let label = t.rsplit("^ ").next().unwrap_or(t);
            let mut s = String::new();
            let mut prev_digit = false;
            for c in label.chars() {
                if c.is_ascii_digit() {
                    if !prev_digit {
                        s.push('N');
                    }
                    prev_digit = true;
                } else {
                    s.push(c);
                    prev_digit = false;
                }
            }
            keep.push(s.chars().take(160).collect());
        }
        if keep.len() >= 2 {
            break;
        }
    }
    if keep.is_empty() {
        keep.push(msg.lines().next().unwrap_or("").chars().take(160).collect());
    }
    format!("{stage:?}: {}", keep.join(" | "))
}

fn paths(t: &Tree, prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    out.push(prefix.clone());
    for (i, c) in t.kids.iter().enumerate() {
        prefix.push(i);
        paths(c, prefix, out);
        prefix.pop();
    }
}

fn at<'a>(t: &'a Tree, p: &[usize]) -> &'a Tree {
    p.iter().fold(t, |n, i| &n.kids[*i])
}

fn replaced(t: &Tree, p: &[usize], with: &Tree) -> Tree {
    match p.split_first() {
        None => with.clone(),
        Some((i, rest)) => {
            let mut kids = t.kids.clone();
            kids[*i] = replaced(&kids[*i], rest, with);
            Tree { symbol: t.symbol, const_index: t.const_index, kids }
        }
    }
}

fn size(t: &Tree) -> usize {
    1 + t.kids.iter().map(size).sum::<usize>()
}

/// The smallest tree that still fails with `sig`: each subtree in turn replaced by a terminal of its type.
fn minimise(ctx: &Ctx, tree: &Tree, sig: &str) -> Tree {
    let k = ctx.contract;
    let mut cur = tree.clone();
    let mut i = 1;
    loop {
        let mut all = Vec::new();
        paths(&cur, &mut Vec::new(), &mut all);
        if i >= all.len() {
            return cur;
        }
        let p = &all[i];
        let node = at(&cur, p);
        if node.kids.is_empty() {
            i += 1;
            continue;
        }
        let ty = k.output(node.symbol);
        let terms = k.terminals_of(ty);
        let plain = terms.iter().copied().find(|s| !k.is_constant(*s)).or_else(|| terms.first().copied());
        let Some(sym) = plain else {
            i += 1;
            continue;
        };
        let leaf = Tree { symbol: sym, const_index: k.is_constant(sym).then_some(0), kids: Vec::new() };
        let cand = replaced(&cur, p, &leaf);
        match ctx.check_tree(&cand) {
            Err((st, msg)) if signature(st, &msg) == sig => cur = cand,
            _ => i += 1,
        }
    }
}

/// Run one kernel text in a child process of this binary (`--file`), killed after `secs`: a kernel that wedges
/// the device cannot take this process with it. A kill is the verdict `Run: hang`.
fn isolated(src: &str, secs: u64) -> Verdict {
    let path = std::env::temp_dir().join(format!("wgsl_true_isolated_{}.wgsl", std::process::id()));
    std::fs::write(&path, src).map_err(|e| (Stage::Naga, e.to_string()))?;
    let exe = std::env::current_exe().map_err(|e| (Stage::Naga, e.to_string()))?;
    let mut child = std::process::Command::new(exe).arg("--file").arg(&path).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null()).spawn().map_err(|e| (Stage::Naga, e.to_string()))?;
    let t = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if t.elapsed().as_secs() >= secs => {
                let _ = child.kill();
                let _ = child.wait();
                drain();
                return Err((Stage::Run, "hang: the kernel ran past the guard".into()));
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(e) => return Err((Stage::Run, e.to_string())),
        }
    }
    let mut text = String::new();
    if let Some(mut o) = child.stdout.take() {
        std::io::Read::read_to_string(&mut o, &mut text).map_err(|e| (Stage::Run, e.to_string()))?;
    }
    let line = text.lines().find(|l| l.starts_with("VERDICT")).unwrap_or("VERDICT err Run no verdict").to_string();
    if line.starts_with("VERDICT ok") {
        Ok(())
    } else {
        let stage = if line.contains(" Naga ") { Stage::Naga } else if line.contains(" Pipeline ") { Stage::Pipeline } else { Stage::Run };
        Err((stage, line.splitn(4, ' ').nth(3).unwrap_or("").to_string()))
    }
}

/// Wait until the device runs a trivial kernel promptly again (a killed runaway keeps it busy for a while).
fn drain() {
    let trivial = "@group(0) @binding(0) var<storage, read> phi: array<f32>;\n@group(0) @binding(1) var<storage, read_write> h: array<f32>;\nstruct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };\n@group(0) @binding(2) var<uniform> cfg: Cfg;\n// grid 16 16 1 1\n@compute @workgroup_size(16, 16, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) { if (gid.x < cfg.d && gid.y < cfg.r_rows) { h[gid.y * cfg.d + gid.x] = 1.0; } }\n";
    let path = std::env::temp_dir().join(format!("wgsl_true_drain_{}.wgsl", std::process::id()));
    if std::fs::write(&path, trivial).is_err() {
        return;
    }
    let Ok(exe) = std::env::current_exe() else { return };
    for _ in 0..120 {
        let t = Instant::now();
        let Ok(mut child) = std::process::Command::new(&exe).arg("--file").arg(&path).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn() else { return };
        while child.try_wait().ok().flatten().is_none() && t.elapsed().as_secs() < 15 {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let done = child.try_wait().ok().flatten().is_some();
        if !done {
            let _ = child.kill();
            let _ = child.wait();
        }
        if done && t.elapsed().as_secs_f64() < 4.0 {
            return;
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}

fn rows_of(t: &Tree, kingdom: &TrueKingdom, out: &mut Vec<String>) {
    out.push(kingdom.rows[t.symbol as usize].name.clone());
    for c in &t.kids {
        rows_of(c, kingdom, out);
    }
}

fn count_rows(t: &Tree, hit: &mut BTreeSet<u32>) {
    hit.insert(t.symbol);
    for c in &t.kids {
        count_rows(c, hit);
    }
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = std::env::args().collect();
    let genes: usize = arg(&args, "--genes").map_or(Ok(100), |v| v.parse().map_err(|e| format!("--genes: {e}")))?;
    let seed: u64 = arg(&args, "--seed").map_or(Ok(1), |v| v.parse().map_err(|e| format!("--seed: {e}")))?;
    let start: usize = arg(&args, "--start").map_or(Ok(0), |v| v.parse().map_err(|e| format!("--start: {e}")))?;
    let n: usize = arg(&args, "--size").map_or(Ok(8), |v| v.parse().map_err(|e| format!("--size: {e}")))?;
    let stage = arg(&args, "--stage").unwrap_or_else(|| "all".into());
    let dump: Option<usize> = arg(&args, "--dump").map(|v| v.parse().map_err(|e| format!("--dump: {e}"))).transpose()?;
    let dir = arg(&args, "--dir").map_or_else(default_dir, PathBuf::from);
    let out = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "/private/tmp/claude-501/st_runs/latest".into()));
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;

    let t0 = Instant::now();
    let kingdom = TrueKingdom::load(&dir)?;
    let relations = kingdom.relations();
    let contract = relations.check().map_err(|d| format!("the kingdom check failed, {} defects, first: {:?}", d.len(), &d[..d.len().min(3)]))?;
    println!("{}", fuller::kingdom_contract::report(&relations, n));
    let layout = GeneLayout::from_program_size(&contract, kingdom.root, n).map_err(|e| e.to_string())?;
    let const_len = kingdom.const_lens();
    let fuel = kingdom.fuel(&contract, n)?;
    println!("gene width {} words; {} types, {} rows; fuel budget K = {}", layout.width, kingdom.types.len(), kingdom.rows.len(), kingdom.fuel_budget);
    println!(
        "work guard: H = {}, K = {}; bound at the unit-test dispatch ({UNIT_TEST_INVOCATIONS} invocations) {}, at the 128 x 64 rung (16384) {}; cap {}",
        kingdom.head_total(&contract, n)?,
        kingdom.fuel_budget,
        fuel.bound(UNIT_TEST_INVOCATIONS).unwrap_or(0),
        fuel.bound(16384).unwrap_or(0),
        fuller::wgsl::oracle::device::CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT
    );

    // The work guard before the device is opened: the kingdom's H and K against the unit test's dispatch.
    if stage != "naga" {
        fuel.check("stage (c) at the unit-test dispatch", UNIT_TEST_INVOCATIONS)?;
    }
    let dev = if stage == "naga" { None } else { Some(Device::new()?) };
    let (d, m) = (UNIT_TEST_D, UNIT_TEST_M);
    let ctx = Ctx { timing: Timing::default(), kingdom: &kingdom, contract: &contract, dev: dev.as_ref(), phi: phi(d, m, RUNG_SEED), d, m, limit: dev.as_ref().map_or(0, |v| fuller::wgsl::gram::effective_buffer_limit(&v.device)), fuel };

    if let Some(i) = arg(&args, "--minimise").map(|v| v.parse::<usize>().map_err(|e| format!("--minimise: {e}"))).transpose()? {
        let mut rng = Rng(seed.wrapping_mul(0x1000_0000_01B3).wrapping_add(i as u64));
        let share = if i % 2 == 0 { Share::Half } else { Share::Proportional };
        let gene = random_gene(&contract, &layout, &const_len, &mut rng, share, (i / 2) % 2 == 1);
        let tree = express(&contract, &layout, &gene, &const_len).map_err(|e| e.to_string())?.tree();
        let guard: u64 = arg(&args, "--guard").map_or(Ok(15), |v| v.parse().map_err(|e| format!("--guard: {e}")))?;
        let first = isolated(&kingdom.render(&tree)?, guard);
        let Err((st, msg)) = first else { return Err(format!("gene {i} passes in isolation")) };
        let sig = signature(st, &msg);
        println!("gene {i} fails in isolation: {sig}");
        let mut cur = tree;
        let mut idx = 1;
        loop {
            let mut all = Vec::new();
            paths(&cur, &mut Vec::new(), &mut all);
            if idx >= all.len() {
                break;
            }
            let p = &all[idx];
            let node = at(&cur, p);
            let ty = contract.output(node.symbol);
            let terms = contract.terminals_of(ty);
            let leaf_sym = terms.iter().copied().find(|s| !contract.is_constant(*s)).or_else(|| terms.first().copied());
            if node.kids.is_empty() || leaf_sym.is_none() {
                idx += 1;
                continue;
            }
            let sym = leaf_sym.unwrap_or_default();
            let leaf = Tree { symbol: sym, const_index: contract.is_constant(sym).then_some(0), kids: Vec::new() };
            let cand = replaced(&cur, p, &leaf);
            match isolated(&kingdom.render(&cand)?, guard) {
                Err((st2, msg2)) if signature(st2, &msg2) == sig => {
                    cur = cand;
                    println!("  reduced to {} nodes", size(&cur));
                }
                _ => idx += 1,
            }
        }
        let mut rows = Vec::new();
        rows_of(&cur, &kingdom, &mut rows);
        let min_src = kingdom.render(&cur)?;
        std::fs::write(out.join(format!("gene_{i}.min.wgsl")), &min_src).map_err(|e| e.to_string())?;
        println!("minimal tree {} nodes, rows: {}", size(&cur), rows.join(" "));
        return Ok(0);
    }
    if let Some(file) = arg(&args, "--file") {
        let src = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
        let t = Instant::now();
        let v = ctx.check_source(&src);
        match &v {
            Ok(()) => println!("VERDICT ok {:.0} ms", t.elapsed().as_secs_f64() * 1e3),
            Err((st, m)) => println!("VERDICT err {st:?} {}", signature(*st, m)),
        }
        return Ok(i32::from(v.is_err()));
    }
    let mut hit: BTreeSet<u32> = BTreeSet::new();
    let mut by_sig: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut passed = 0usize;
    let mut sizes = Vec::new();
    for i in start..start + genes {
        let mut rng = Rng(seed.wrapping_mul(0x1000_0000_01B3).wrapping_add(i as u64));
        let share = if i % 2 == 0 { Share::Half } else { Share::Proportional };
        let root_function = (i / 2) % 2 == 1;
        let gene = random_gene(&contract, &layout, &const_len, &mut rng, share, root_function);
        let expressed = express(&contract, &layout, &gene, &const_len).map_err(|e| format!("gene {i}: {e}"))?;
        let tree = expressed.tree();
        count_rows(&tree, &mut hit);
        sizes.push(expressed.nodes.len());
        let src = kingdom.render(&tree)?;
        if dump == Some(i) {
            println!("{src}");
        }
        let t = Instant::now();
        match ctx.check_source(&src) {
            Ok(()) => passed += 1,
            Err((st, msg)) => {
                let sig = signature(st, &msg);
                std::fs::write(out.join(format!("gene_{i}.wgsl")), &src).map_err(|e| e.to_string())?;
                std::fs::write(out.join(format!("gene_{i}.error.txt")), &msg).map_err(|e| e.to_string())?;
                if st == Stage::Run && msg.starts_with("runaway") {
                    // The device is wedged by this kernel: nothing after it in this process can be trusted. The
                    // wrapper restarts at the next gene; `--minimise {i}` names the rows in isolation.
                    println!("RUNAWAY gene {i} ({} nodes): {sig}", expressed.nodes.len());
                    println!("{passed} of {} genes before it pass", i - start);
                    std::process::exit(3);
                }
                let small = minimise(&ctx, &tree, &sig);
                let mut rows = Vec::new();
                rows_of(&small, &kingdom, &mut rows);
                let small_src = kingdom.render(&small)?;
                std::fs::write(out.join(format!("gene_{i}.min.wgsl")), &small_src).map_err(|e| e.to_string())?;
                println!("FAIL gene {i} ({} nodes, {:.0} ms) {sig}\n     minimal tree {} nodes, rows: {}", expressed.nodes.len(), t.elapsed().as_secs_f64() * 1e3, size(&small), rows.join(" "));
                by_sig.entry(sig).or_default().push(i);
            }
        }
    }
    sizes.sort_unstable();
    println!(
        "{} of {} genes pass ({:?}); nodes median {} max {}; rows used {} of {}; {:.0} s",
        passed,
        genes,
        match stage.as_str() {
            "naga" => "naga",
            _ => "naga, pipeline, run",
        },
        sizes.get(sizes.len() / 2).copied().unwrap_or(0),
        sizes.last().copied().unwrap_or(0),
        hit.len(),
        kingdom.rows.len(),
        t0.elapsed().as_secs_f64()
    );
    println!(
        "stage totals: naga {:.0} ms, pipeline {:.0} ms, run {:.0} ms (slowest run {:.0} ms)",
        ctx.timing.naga_ms.get(),
        ctx.timing.pipeline_ms.get(),
        ctx.timing.run_ms.get(),
        ctx.timing.slowest_run_ms.get()
    );
    for (sig, genes) in &by_sig {
        println!("  {} gene(s) {:?}: {sig}", genes.len(), &genes[..genes.len().min(8)]);
    }
    let unused: Vec<&str> = kingdom.rows.iter().enumerate().filter(|(i, _)| !hit.contains(&(*i as u32))).map(|(_, r)| r.name.as_str()).collect();
    std::fs::write(out.join("rows_not_used.txt"), unused.join("\n")).map_err(|e| e.to_string())?;
    Ok(if passed == genes { 0 } else { 1 })
}

fn main() {
    let child = std::thread::Builder::new().stack_size(1 << 30).spawn(run).expect("thread");
    match child.join().expect("the loop thread") {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    }
}
