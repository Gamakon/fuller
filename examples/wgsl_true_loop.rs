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

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use fuller::kingdom_contract::KingdomContract;
use fuller::typed_gep::{express, GeneLayout, Tree};
use fuller::wgsl::gram::{phi, Gram, RUNGS, RUNG_SEED};
use fuller::wgsl::oracle::device::Device;
use fuller::wgsl::true_kingdom::{default_dir, random_gene, Rng, Share, TrueKingdom};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Naga,
    Pipeline,
    Run,
}

type Verdict = Result<(), (Stage, String)>;

struct Ctx<'a> {
    kingdom: &'a TrueKingdom,
    contract: &'a KingdomContract,
    dev: Option<&'a Device>,
    phi: Vec<f32>,
    d: usize,
    m: usize,
    limit: u64,
}

impl Ctx<'_> {
    fn check_source(&self, src: &str) -> Verdict {
        let module = naga::front::wgsl::parse_str(src).map_err(|e| (Stage::Naga, e.emit_to_string(src)))?;
        fuller::wgsl::target::Target::current().validate(&module, src).map_err(|e| (Stage::Naga, e))?;
        let Some(dev) = self.dev else { return Ok(()) };
        let gram = Gram::compile(dev, src).map_err(|e| (Stage::Pipeline, e))?;
        gram.build(&self.phi, self.d, self.m, self.limit).map_err(|e| (Stage::Run, e))?;
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
    println!("gene width {} words; {} types, {} rows; fuel budget K = {}", layout.width, kingdom.types.len(), kingdom.rows.len(), kingdom.fuel_budget);

    let dev = if stage == "naga" { None } else { Some(Device::new()?) };
    let (d, m) = RUNGS[0];
    let ctx = Ctx { kingdom: &kingdom, contract: &contract, dev: dev.as_ref(), phi: phi(d, m, RUNG_SEED), d, m, limit: dev.as_ref().map_or(0, |v| fuller::wgsl::gram::effective_buffer_limit(&v.device)) };

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
