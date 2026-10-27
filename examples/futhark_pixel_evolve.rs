//! The Futhark route to a cuneus shader, step 4: evolve a Futhark program that draws volumepassage, in the pixel
//! family of the shape kingdom (`fuller::futhark_pixel`). Every gene expresses a legal Futhark program; each candidate
//! is compiled by the branch compiler's webgpu backend, linked against the shim, and run on our wgpu/Metal path.
//!
//!   cargo run --release --features gpu,wgsl --example futhark_pixel_evolve -- --regime petri --minutes 30 \
//!       --pop 24 --seed 1 --out /private/tmp/claude-501/fpix/petri
//!
//! Two regimes:
//! * `petri`: a random population (the root cell is the image row) evolves toward the original's picture;
//! * `champion`: the reference written as a gene, its e-graph variants and the saturated fold's jointly extracted
//!   forms make the population, and the search looks for an equivalent that is faster; every `--pump-every`
//!   generations the e-graph variants of the best replace the worst.
//!
//! Fitness, the two-level HFF (hff library TrueNorth, every measurement its own input, 0 best, pi/2 worst):
//! * quality group: for each training case (t = 5 and 8, `--size` square), six statistics of the picture against the
//!   WGSL original's: the differing channel fraction; max |d|, mean |d| and rmse, each divided by the original's own
//!   max, mean and rms (so drawing nothing is an error of 1) and mapped e -> e / (1 + e), as the speed ratios are (so
//!   drawing nothing scores 0.5, a picture brighter than the original's scale approaches 1, and there is a gradient
//!   at every magnitude: clamped at 1, a random population scored 1 on all three and the petri dish of 2026-10-27
//!   14:38Z did not move in 9 generations); the fraction of pixels over 1/255; the fraction of non-finite mismatches;
//! * speed group: `--pairs` interleaved timed pairs at t = 5 against the reference program, each `r / (1 + r)` with
//!   `r` the candidate's GPU time over the reference's (0.5 is equal speed); each side of a pair is one process
//!   running the program `--runs` times, its time the fastest run's (the sum of that run's GPU-timed passes; the
//!   GPU's clock ramps up within a process, see `futhark_pixel::draw`);
//! * fitness = TrueNorth(theta_quality / (pi/2), theta_speed / (pi/2)).
//!
//! The work bound: every candidate is drawn in tiles whose bound (node executions a pixel, exact for the expressed
//! tree, times the tile's pixels) is at most the per-submit cap. The device of this process draws the originals at the
//! start and is dropped before any Futhark program runs; then one program runs at a time.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::FRAC_PI_2;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use fuller::futhark_kingdom::{FutharkKingdom, GeneRng};
use fuller::futhark_pixel::{build, draw, inputs, interface, load, program, tiles, variants, work_per_pixel, Bridge, FoldEgraph, Input, Tile, VOLUMEPASSAGE, VOLUMEPASSAGE_REFERENCE};
use fuller::futhark_wgpu_host::compile_program;
use fuller::geneframe::Ty;
use fuller::kingdom_contract::KingdomContract;
use fuller::population_fold::AdfTable;
use fuller::saturated_fold::{fold_saturated, Budget};
use fuller::typed_gep::{express, write_back, Domain, GeneLayout, Tree};
use fuller::wgsl::cuneus_originals::case_bindings;
use fuller::wgsl::gram::truenorth;
use fuller::wgsl::image::{compare, side_by_side, srgb_display, write_png, BindingData, Difference, ImageKernel, Picture};
use fuller::wgsl::oracle::device::{Device, Fuel, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    arg(args, name).map_or(Ok(default), |v| v.parse::<T>().map_err(|_| format!("{name}: {v} is not a number")))
}

/// UTC wall time `HH:MM:SSZ`.
fn utc() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) % 86_400;
    format!("{:02}:{:02}:{:02}Z", s / 3600, (s / 60) % 60, s % 60)
}

fn blocks_of(b: &[BindingData]) -> Vec<(u32, u32, Vec<u8>)> {
    b.iter()
        .filter_map(|d| match d {
            BindingData::Uniform { group, binding, bytes } => Some((*group, *binding, bytes.clone())),
            _ => None,
        })
        .collect()
}

/// One case: a time, its uniforms, the original's picture.
struct Case {
    time: f32,
    size: u32,
    blocks: Vec<(u32, u32, Vec<u8>)>,
    original: Picture,
}

/// What a candidate measured.
#[derive(Debug, Clone)]
struct Eval {
    theta: f64,
    theta_q: f64,
    theta_s: f64,
    quality: Vec<f64>,
    speed: Vec<f64>,
    /// Median of the pairs' GPU-time ratios (candidate over reference).
    ratio: f64,
    /// The t = 5 training case's difference.
    diff5: Option<Difference>,
    work: u128,
    tiles: usize,
}

impl Eval {
    /// A legal gene's program that did not compile or run: our software has a bug (logged in full), and the
    /// individual takes the worst place.
    fn worst() -> Eval {
        Eval { theta: FRAC_PI_2, theta_q: FRAC_PI_2, theta_s: FRAC_PI_2, quality: vec![], speed: vec![], ratio: f64::INFINITY, diff5: None, work: 0, tiles: 0 }
    }
}

#[derive(Clone)]
struct Individual {
    gene: Vec<u32>,
    tree: Tree,
    source: String,
    eval: Eval,
}

/// The fitness machinery: the kingdom, the cases, the reference's executable, and the cache by program text.
struct Judge<'a> {
    k: &'a FutharkKingdom,
    iface: &'a [Input],
    cases: Vec<Case>,
    size: u32,
    pairs: usize,
    /// Runs a timed process makes (Futhark's `-r`); its time is the fastest run.
    runs: usize,
    ref_exe: PathBuf,
    ref_plan: Vec<Tile>,
    work_dir: PathBuf,
    cache: HashMap<String, Eval>,
    evaluated: usize,
    hits: usize,
    bugs: usize,
    bug_log: PathBuf,
}

impl Judge<'_> {
    fn ps<'b>(&'b self, case: &'b Case) -> impl Fn(Tile) -> Result<Vec<f32>, String> + 'b {
        move |t: Tile| {
            let bl: Vec<(u32, u32, &[u8])> = case.blocks.iter().map(|(g, b, v)| (*g, *b, v.as_slice())).collect();
            inputs(self.iface, &bl, case.size, case.size, t)
        }
    }

    fn picture(&self, exe: &Path, plan: &[Tile], case: &Case, runs: usize) -> Result<(Picture, f64), String> {
        let (rgba, ns) = draw(exe, true, &self.ps(case), plan, case.size, case.size, runs)?;
        Ok((Picture { width: case.size, height: case.size, rgba }, ns))
    }

    /// The paired GPU-time ratios at the first case: candidate and reference interleaved, the order alternating.
    fn time_pairs(&self, exe: &Path, plan: &[Tile], pairs: usize) -> Result<Vec<f64>, String> {
        let case = &self.cases[0];
        let mut out = Vec::with_capacity(pairs);
        for i in 0..pairs {
            let (c, r) = if i % 2 == 0 {
                let c = self.picture(exe, plan, case, self.runs)?.1;
                (c, self.picture(&self.ref_exe, &self.ref_plan, case, self.runs)?.1)
            } else {
                let r = self.picture(&self.ref_exe, &self.ref_plan, case, self.runs)?.1;
                (self.picture(exe, plan, case, self.runs)?.1, r)
            };
            out.push(c / r);
        }
        Ok(out)
    }

    fn evaluate(&mut self, tree: &Tree, source: &str) -> Eval {
        if let Some(e) = self.cache.get(source) {
            self.hits += 1;
            return e.clone();
        }
        self.evaluated += 1;
        let e = match self.measure(tree, source) {
            Ok(e) => e,
            Err(msg) => {
                self.bugs += 1;
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.bug_log).expect("the bug log opens");
                let _ = writeln!(f, "==== {} our software has a bug: a legal gene's program failed\n{msg}\n---- program\n{source}", utc());
                Eval::worst()
            }
        };
        self.cache.insert(source.to_string(), e.clone());
        e
    }

    fn measure(&self, tree: &Tree, source: &str) -> Result<Eval, String> {
        let work = work_per_pixel(self.k, tree)?;
        let plan = tiles(work, self.size, self.size, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
        let exe = compile_program(source, &self.work_dir)?;
        let mut quality = Vec::new();
        let mut diff5 = None;
        for case in &self.cases {
            let (pic, _) = self.picture(&exe, &plan, case, 1)?;
            let d = compare(&pic, &case.original)?;
            let channels = (d.pixels * 4) as f64;
            let (o_max, o_mean, o_rms) = scale(&case.original);
            let rel = |e: f64| if e.is_finite() { e / (1.0 + e) } else { 1.0 };
            quality.extend([d.differing_channels as f64 / channels, rel(d.max_abs / o_max), rel(d.mean_abs / o_mean), rel(d.rmse / o_rms), d.pixels_over_one_step as f64 / d.pixels as f64, d.non_finite_mismatch as f64 / channels]);
            if diff5.is_none() {
                diff5 = Some(d);
            }
        }
        let ratios = self.time_pairs(&exe, &plan, self.pairs)?;
        let speed: Vec<f64> = ratios.iter().map(|r| r / (1.0 + r)).collect();
        let theta_q = truenorth(&quality);
        let theta_s = truenorth(&speed);
        let theta = truenorth(&[theta_q / FRAC_PI_2, theta_s / FRAC_PI_2]);
        Ok(Eval { theta, theta_q, theta_s, quality, speed, ratio: median(&ratios), diff5, work, tiles: plan.len() })
    }
}

/// The original's own magnitude over its colour channels, in display space (as `compare` measures): the largest, the mean and the root mean square. An error
/// statistic divided by it is the error relative to drawing nothing (the black picture's is 1), on the original's own
/// scale: volumepassage is a dark picture (mean about 0.03), and an absolute error would rank
/// a black picture as nearly right.
fn scale(p: &Picture) -> (f64, f64, f64) {
    let c: Vec<f64> = p.rgba.chunks_exact(4).flat_map(|px| px[..3].iter().map(|v| f64::from(srgb_display(*v)))).filter(|v| v.is_finite()).collect();
    let n = c.len().max(1) as f64;
    let max = c.iter().cloned().fold(0.0, f64::max);
    let mean = c.iter().map(|v| v.abs()).sum::<f64>() / n;
    let rms = (c.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    (max.max(f64::MIN_POSITIVE), mean.max(f64::MIN_POSITIVE), rms.max(f64::MIN_POSITIVE))
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    match s.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => s[n / 2],
        n => 0.5 * (s[n / 2 - 1] + s[n / 2]),
    }
}

/// The genetic side: the contract, the layout and the variation operators.
struct Genetics<'a> {
    k: &'a FutharkKingdom,
    contract: &'a KingdomContract,
    layout: &'a GeneLayout,
    const_len: &'a BTreeMap<Ty, u32>,
}

impl Genetics<'_> {
    fn random_gene(&self, rng: &mut GeneRng) -> Vec<u32> {
        let mut gene: Vec<u32> = (0..self.layout.width)
            .map(|p| {
                let a = self.layout.alphabet(self.contract, p, self.const_len);
                a[rng.below(a.len())]
            })
            .collect();
        // The root cell is the image row (phylu's root_function): the root type has one function, and a gene
        // whose root is the black-picture terminal draws nothing to evolve from.
        let image = self.k.rows.iter().position(|r| r.name == "image").expect("the image row") as u32;
        let root_frame = self.layout.frame(self.layout.root).expect("the root's frame");
        let pos = root_frame.row_pos(0).expect("the root's first row");
        gene[pos] = image;
        gene
    }

    fn tree(&self, gene: &[u32]) -> Result<Tree, String> {
        Ok(express(self.contract, self.layout, gene, self.const_len).map_err(|e| format!("a gene does not express: {e}"))?.tree())
    }

    /// The cells the walk reads: every expressed node's symbol cell and, for an expressed constant, its Dc cell.
    fn coding_cells(&self, gene: &[u32]) -> Result<Vec<usize>, String> {
        let e = express(self.contract, self.layout, gene, self.const_len).map_err(|e| format!("a gene does not express: {e}"))?;
        let mut q: BTreeMap<Ty, usize> = BTreeMap::new();
        let mut out = Vec::with_capacity(e.nodes.len() * 2);
        for n in &e.nodes {
            out.push(n.pos);
            if n.const_index.is_some() {
                let qq = q.entry(n.ty).or_default();
                let frame = self.layout.frame(n.ty).ok_or("a node of a type with no frame")?;
                out.push(frame.dc_off.ok_or("a constant with no Dc")? + *qq);
                *qq += 1;
            }
        }
        Ok(out)
    }

    /// LOW-CHANGE mutation: `m` expressed cells, each set to another member of its own alphabet.
    fn mutate(&self, gene: &mut [u32], m: usize, rng: &mut GeneRng) -> Result<(), String> {
        for _ in 0..m {
            let cells = self.coding_cells(gene)?;
            for _ in 0..16 {
                let pos = cells[rng.below(cells.len())];
                let a = self.layout.alphabet(self.contract, pos, self.const_len);
                if a.len() < 2 {
                    continue;
                }
                let v = a[rng.below(a.len())];
                if v != gene[pos] {
                    self.layout.mutate(self.contract, gene, pos, v, self.const_len).map_err(|e| format!("mutation: {e}"))?;
                    break;
                }
            }
        }
        Ok(())
    }

    /// Homologous crossover: a range of the domain that holds one of `a`'s expressed cells, exchanged with `b`.
    fn crossover(&self, a: &mut [u32], b: &[u32], rng: &mut GeneRng) -> Result<(), String> {
        let cells = self.coding_cells(a)?;
        let pos = cells[rng.below(cells.len())];
        let span = self.layout.spans().into_iter().find(|s| s.off <= pos && pos < s.off + s.len).ok_or("a cell in no span")?;
        if span.domain == Domain::Dc && span.len < 2 {
            return Ok(());
        }
        let lo = span.off + rng.below(pos - span.off + 1);
        let hi = (pos + 1 + rng.below(span.off + span.len - pos)).min(span.off + span.len);
        let mut other = b.to_vec();
        self.layout.swap_range(self.layout, a, &mut other, lo, hi).map_err(|e| format!("crossover: {e}"))
    }
}

/// The original's pictures, drawn by the image harness; the device is dropped on return.
fn originals(size: u32, times: &[f32], report_size: u32) -> Result<(Vec<Case>, Vec<Case>), String> {
    let src = std::fs::read_to_string(format!("{}/tests/corpus/cuneus_visual/volumepassage.wgsl", env!("CARGO_MANIFEST_DIR"))).map_err(|e| format!("original: {e}"))?;
    let dev = Device::new()?;
    let draw_case = |s: u32, t: f32| -> Result<Case, String> {
        let b = case_bindings("volumepassage", t, s, s)?;
        let k = ImageKernel::compile(&dev, "volumepassage", &src, "main", &b, s, s, Fuel::Proven("cuneus original volumepassage"))?;
        let run = k.run(1, 1)?;
        Ok(Case { time: t, size: s, blocks: blocks_of(&b), original: run.picture })
    };
    let train = [5.0f32, 8.0].iter().map(|t| draw_case(size, *t)).collect::<Result<Vec<_>, _>>()?;
    let report = times.iter().map(|t| draw_case(report_size, *t)).collect::<Result<Vec<_>, _>>()?;
    Ok((train, report))
}

const CHART: &str = r##"
import sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
g = [int(r["generation"]) for r in rows]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(g, [float(r["best_theta"]) for r in rows], label="best HFF", color="#1f6feb", linewidth=2)
ax.plot(g, [float(r["median_theta"]) for r in rows], label="median HFF", color="#8b949e")
ax.plot(g, [float(r["best_theta_q"]) for r in rows], label="best's quality angle", color="#2da44e", linestyle="--")
ax.plot(g, [float(r["best_theta_s"]) for r in rows], label="best's speed angle", color="#bf8700", linestyle=":")
ax.axhline(float(sys.argv[3]), color="#cf222e", linewidth=1, label="the reference's HFF")
ax.set_ylim(0, 1.6)
ax.set_xlabel("generation")
ax.set_ylabel("HFF (0 best, pi/2 worst)")
ax.set_title(sys.argv[4])
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=120)
"##;

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let regime = arg(&args, "--regime").ok_or("--regime petri|champion")?;
    if regime != "petri" && regime != "champion" {
        return Err(format!("--regime {regime}: petri or champion"));
    }
    let minutes: f64 = num(&args, "--minutes", 30.0)?;
    let pop: usize = num(&args, "--pop", 24)?;
    let seed: u64 = num(&args, "--seed", 1)?;
    let size: u32 = num(&args, "--size", 64)?;
    let pairs: usize = num(&args, "--pairs", 3)?;
    let runs: usize = num(&args, "--runs", 40)?;
    let elites: usize = num(&args, "--elites", 2)?;
    let pump_every: usize = num(&args, "--pump-every", 5)?;
    let max_generations: usize = num(&args, "--generations", 100_000)?;
    let out = PathBuf::from(arg(&args, "--out").ok_or("--out DIR")?);
    std::fs::create_dir_all(out.join("hall_of_fame")).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let limit = std::time::Duration::from_secs_f64(minutes * 60.0);
    let low_change = regime == "champion";
    println!("# {} futhark_pixel_evolve: regime {regime}, {minutes} min, population {pop}, seed {seed}, {size}x{size}, {pairs} timed pairs, elites {elites}", utc());

    let (train, report_cases) = originals(size, &[2.0, 5.0, 8.0], 128)?;
    println!("# {} the original drawn: {} training cases at {size}x{size}, {} report cases at 128x128; the device is dropped", utc(), train.len(), report_cases.len());

    let k = load(&VOLUMEPASSAGE)?;
    let iface = interface(&VOLUMEPASSAGE)?;
    let contract = k.relations().check().map_err(|d| format!("the kingdom check: {:?}", &d[..d.len().min(3)]))?;
    let reference = build(&k, VOLUMEPASSAGE_REFERENCE)?;
    let n_funcs = reference.level_order().iter().filter(|(t, _)| !k.rows[t.symbol as usize].inputs.is_empty()).count();
    let root = k.ty("image").ok_or("no image dual")?;
    let layout = GeneLayout::from_program_size(&contract, root, n_funcs).map_err(|e| e.to_string())?;
    let const_len = k.const_len();
    let gen = Genetics { k: &k, contract: &contract, layout: &layout, const_len: &const_len };
    println!("# layout: program size N = {n_funcs} (the reference's functions), {} words a gene", layout.width);

    let ref_source = program(&k, &iface, &reference)?;
    let ref_exe = compile_program(&ref_source, &out.join("reference_build"))?;
    let ref_plan = tiles(work_per_pixel(&k, &reference)?, size, size, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
    let mut judge = Judge {
        k: &k,
        iface: &iface,
        cases: train,
        size,
        pairs,
        runs,
        ref_exe,
        ref_plan,
        work_dir: out.join("candidate_build"),
        cache: HashMap::new(),
        evaluated: 0,
        hits: 0,
        bugs: 0,
        bug_log: out.join("bugs.log"),
    };
    // One pair to warm the GPU after the image harness, discarded: the first pair of a trial measured 2.36.
    judge.time_pairs(&judge.ref_exe.clone(), &judge.ref_plan.clone(), 1)?;
    let noise = judge.time_pairs(&judge.ref_exe.clone(), &judge.ref_plan.clone(), 11)?;
    println!(
        "# {} the timing's noise floor, the reference against itself, 11 pairs of {runs} runs: ratio median {:.4}, min {:.4}, max {:.4}",
        utc(),
        median(&noise),
        noise.iter().cloned().fold(f64::INFINITY, f64::min),
        noise.iter().cloned().fold(0.0, f64::max)
    );
    let ref_eval = judge.evaluate(&reference, &ref_source);
    println!(
        "# {} the reference: HFF {:.5} (quality {:.5}, speed {:.5}), ratio {:.4}, {} node executions a pixel, {} tile(s); t=5 differing channels {}",
        utc(),
        ref_eval.theta,
        ref_eval.theta_q,
        ref_eval.theta_s,
        ref_eval.ratio,
        ref_eval.work,
        ref_eval.tiles,
        ref_eval.diff5.as_ref().map_or(0, |d| d.differing_channels)
    );

    let mut rng = GeneRng::new(seed);
    let bridge = Bridge::load(&VOLUMEPASSAGE)?;
    let ref_gene = write_back(&contract, &layout, &gen.random_gene(&mut rng), &reference).map_err(|e| format!("the reference's write-back: {e}"))?;
    let mut genes: Vec<Vec<u32>> = Vec::new();
    if regime == "petri" {
        while genes.len() < pop {
            genes.push(gen.random_gene(&mut rng));
        }
    } else {
        genes.push(ref_gene.clone());
        let vs = variants(&VOLUMEPASSAGE, &k, &bridge, &reference, 2, 16, pop * 2)?;
        println!("# {} e-graph: {} variants of the reference", utc(), vs.len());
        for v in &vs {
            genes.push(write_back(&contract, &layout, &ref_gene, v).map_err(|e| format!("a variant's write-back: {e}"))?);
        }
        let trees: Vec<Tree> = genes.iter().map(|g| gen.tree(g)).collect::<Result<_, _>>()?;
        let mut table = AdfTable::default();
        let fe = FoldEgraph { inst: &VOLUMEPASSAGE, kingdom: &k, bridge: &bridge };
        let fold = fold_saturated(&mut table, &fe, &contract, &layout, &const_len, &genes, &trees, &Budget::default())?;
        println!(
            "# {} saturated fold over {} genes: {} distinct trees, {} iterations, {} shared parts (structural hashing {}), {} forms differ from their submitted tree",
            utc(),
            genes.len(),
            fold.distinct,
            fold.iterations,
            fold.used.len(),
            fold.structural.adfs,
            fold.reformed
        );
        let mut forms = Vec::new();
        for f in &fold.forms {
            if !trees.contains(f) && !forms.contains(f) {
                forms.push(f.clone());
            }
        }
        for f in &forms {
            genes.push(write_back(&contract, &layout, &ref_gene, f).map_err(|e| format!("a fold form's write-back: {e}"))?);
        }
        println!("# the fold's jointly extracted forms add {} new trees", forms.len());
        genes.truncate(pop);
    }

    let mut series = String::from("generation\tutc\tseconds\tevaluated\tcache_hits\tbugs\tbest_theta\tbest_theta_q\tbest_theta_s\tmedian_theta\tbest_ratio\tbest_differing_t5\tbest_mean_abs_t5\tbest_nodes\tbest_work\tdistinct\n");
    let mut hof: Vec<Individual> = Vec::new();
    let mut population: Vec<Individual> = Vec::new();
    let mut generation = 0usize;
    'run: while generation < max_generations {
        // ---- evaluate ----
        let mut next = Vec::with_capacity(genes.len());
        for g in &genes {
            if started.elapsed() > limit {
                break 'run;
            }
            let tree = gen.tree(g)?;
            let source = program(&k, &iface, &tree)?;
            let eval = judge.evaluate(&tree, &source);
            next.push(Individual { gene: g.clone(), tree, source, eval });
        }
        population = next;
        population.sort_by(|a, b| a.eval.theta.total_cmp(&b.eval.theta));
        for ind in &population {
            if !hof.iter().any(|h| h.source == ind.source) {
                hof.push(ind.clone());
            }
        }
        hof.sort_by(|a, b| a.eval.theta.total_cmp(&b.eval.theta));
        hof.truncate(5);
        let best = &population[0];
        let thetas: Vec<f64> = population.iter().map(|i| i.eval.theta).collect();
        let distinct = {
            let mut s: Vec<&str> = population.iter().map(|i| i.source.as_str()).collect();
            s.sort_unstable();
            s.dedup();
            s.len()
        };
        let d5 = best.eval.diff5.as_ref();
        series.push_str(&format!(
            "{generation}\t{}\t{:.1}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}\t{}\t{:.3e}\t{}\t{}\t{distinct}\n",
            utc(),
            started.elapsed().as_secs_f64(),
            judge.evaluated,
            judge.hits,
            judge.bugs,
            best.eval.theta,
            best.eval.theta_q,
            best.eval.theta_s,
            median(&thetas),
            best.eval.ratio,
            d5.map_or(0, |d| d.differing_channels),
            d5.map_or(f64::NAN, |d| d.mean_abs),
            best.tree.level_order().len(),
            best.eval.work,
        ));
        println!(
            "{} gen {generation}: best HFF {:.5} (q {:.5}, s {:.5}, ratio {:.3}), median {:.5}, {distinct} distinct; {} evaluated, {} cache hits, {} bugs",
            utc(),
            best.eval.theta,
            best.eval.theta_q,
            best.eval.theta_s,
            best.eval.ratio,
            median(&thetas),
            judge.evaluated,
            judge.hits,
            judge.bugs
        );
        std::fs::write(out.join("series.tsv"), &series).map_err(|e| e.to_string())?;

        // ---- the next generation ----
        let mut children: Vec<Vec<u32>> = Vec::with_capacity(pop);
        let mut taken: Vec<&str> = Vec::new();
        for ind in &population {
            if children.len() >= elites {
                break;
            }
            if !taken.contains(&ind.source.as_str()) {
                taken.push(&ind.source);
                children.push(ind.gene.clone());
            }
        }
        if low_change && pump_every > 0 && generation > 0 && generation.is_multiple_of(pump_every) {
            // The pump: the e-graph variants of the best enter the population.
            for v in variants(&VOLUMEPASSAGE, &k, &bridge, &population[0].tree, 2, 16, 4)? {
                if children.len() < pop {
                    if let Ok(g) = write_back(&contract, &layout, &population[0].gene, &v) {
                        children.push(g);
                    }
                }
            }
        }
        let tournament = |rng: &mut GeneRng| -> &Individual {
            let mut best = &population[rng.below(population.len())];
            for _ in 0..2 {
                let c = &population[rng.below(population.len())];
                if c.eval.theta < best.eval.theta {
                    best = c;
                }
            }
            best
        };
        while children.len() < pop {
            let a = tournament(&mut rng);
            let mut child = a.gene.clone();
            if rng.below(2) == 0 {
                // Crossover parents are different expressed trees; with no such partner, no crossover.
                let mut b = tournament(&mut rng);
                for _ in 0..8 {
                    if b.source != a.source {
                        break;
                    }
                    b = tournament(&mut rng);
                }
                if b.source != a.source {
                    gen.crossover(&mut child, &b.gene, &mut rng)?;
                }
            }
            let m = if low_change { 1 + rng.below(2) } else { 1 + rng.below(4) };
            gen.mutate(&mut child, m, &mut rng)?;
            children.push(child);
        }
        genes = children;
        generation += 1;
    }
    if population.is_empty() {
        return Err("no generation completed within the time limit".into());
    }
    println!("# {} stopped after {generation} generation(s), {:.1} min; {} evaluated, {} cache hits, {} bugs", utc(), started.elapsed().as_secs_f64() / 60.0, judge.evaluated, judge.hits, judge.bugs);

    // ---- the report ----
    let best = hof[0].clone();
    for (i, h) in hof.iter().enumerate() {
        std::fs::write(out.join("hall_of_fame").join(format!("rank{}.fut", i + 1)), &h.source).map_err(|e| e.to_string())?;
    }
    std::fs::write(out.join("reference.fut"), &ref_source).map_err(|e| e.to_string())?;
    let best_exe = compile_program(&best.source, &out.join("best_build"))?;
    let best_plan_r = tiles(best.eval.work, 128, 128, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
    let ref_plan_r = tiles(work_per_pixel(&k, &reference)?, 128, 128, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
    let mut report = format!(
        "regime\t{regime}\nseed\t{seed}\ngenerations\t{generation}\nminutes\t{:.1}\nevaluated\t{}\ncache_hits\t{}\nbugs\t{}\nreference_hff\t{:.6}\nbest_hff\t{:.6}\nbest_theta_quality\t{:.6}\nbest_theta_speed\t{:.6}\nbest_nodes\t{}\nreference_nodes\t{}\nbest_work_per_pixel\t{}\nreference_work_per_pixel\t{}\n",
        started.elapsed().as_secs_f64() / 60.0,
        judge.evaluated,
        judge.hits,
        judge.bugs,
        ref_eval.theta,
        best.eval.theta,
        best.eval.theta_q,
        best.eval.theta_s,
        best.tree.level_order().len(),
        reference.level_order().len(),
        best.eval.work,
        ref_eval.work
    );
    let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.5}")).collect::<Vec<_>>().join(",");
    report.push_str(&format!("best_quality_inputs\t{}\nbest_speed_inputs\t{}\nbest_tiles_at_training_size\t{}\nreference_quality_inputs\t{}\nreference_speed_inputs\t{}\n", fmt(&best.eval.quality), fmt(&best.eval.speed), best.eval.tiles, fmt(&ref_eval.quality), fmt(&ref_eval.speed)));
    report.push_str("case\tpicture\tdiffering_channels\tchannels\tmax_abs\tmean_abs\trmse\tpixels_over_1_255\tnon_finite\n");
    for case in &report_cases {
        let (bp, _) = judge.picture(&best_exe, &best_plan_r, case, 1)?;
        let (rp, _) = judge.picture(&judge.ref_exe, &ref_plan_r, case, 1)?;
        for (name, p) in [("best vs original", &bp), ("reference vs original", &rp)] {
            let d = compare(p, &case.original)?;
            report.push_str(&format!(
                "128x128 t={}\t{name}\t{}\t{}\t{:e}\t{:e}\t{:e}\t{}\t{}\n",
                case.time,
                d.differing_channels,
                d.pixels * 4,
                d.max_abs,
                d.mean_abs,
                d.rmse,
                d.pixels_over_one_step,
                d.non_finite_mismatch
            ));
        }
        write_png(&out.join(format!("original_best_reference_128_t{}.png", case.time)), &side_by_side(&[case.original.clone(), bp, rp])?)?;
    }
    // The speed: 11 interleaved pairs at the training size, t = 5.
    let best_plan = tiles(best.eval.work, size, size, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
    let ratios = judge.time_pairs(&best_exe, &best_plan, 11)?;
    let faster = ratios.iter().filter(|r| **r < 1.0).count();
    report.push_str(&format!("noise_floor_ratios_reference_vs_itself\t{}\nspeed_pairs\t11\nspeed_ratio_median\t{:.4}\nspeed_ratio_min\t{:.4}\nspeed_ratio_max\t{:.4}\nfaster_rounds\t{faster}\n", fmt(&noise), median(&ratios), ratios.iter().cloned().fold(f64::INFINITY, f64::min), ratios.iter().cloned().fold(0.0, f64::max)));
    std::fs::write(out.join("report.tsv"), &report).map_err(|e| e.to_string())?;
    print!("{report}");

    let chart = std::process::Command::new("python3")
        .arg("-I")
        .arg("-c")
        .arg(CHART)
        .arg(out.join("series.tsv"))
        .arg(out.join("chart_hff_vs_generation.png"))
        .arg(format!("{:.6}", ref_eval.theta))
        .arg(format!("volumepassage in Futhark, regime {regime}: HFF against generation"))
        .output()
        .map_err(|e| format!("python3: {e}"))?;
    if !chart.status.success() {
        return Err(format!("the chart failed: {}", String::from_utf8_lossy(&chart.stderr)));
    }
    println!("# {} done: {}", utc(), out.display());
    Ok(())
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
