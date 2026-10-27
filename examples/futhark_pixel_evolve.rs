//! The Futhark route to a cuneus shader, step 4: evolve a Futhark program that draws the shader's picture, in the
//! pixel family of the shape kingdom (`fuller::futhark_pixel`). Every gene expresses a legal Futhark program; each
//! candidate is compiled by the branch compiler's webgpu backend, linked against the shim, and run on our wgpu/Metal
//! path. The runner meets `docs/STANDARD_runners.md`: the stream (`OUT/stream.jsonl`, `fuller::telemetry`), the
//! two-level HFF, checkpoints and resume, the kill trigger (`OUT/killme.trigger`), guards, the hall of fame with
//! pictures and OCR text, and the ledger with lineage.
//!
//!   cargo run --release --features gpu,wgsl --example futhark_pixel_evolve -- --shader cuneus --regime petri \
//!       --minutes 30 --pop 16 --seed 1 --out /private/tmp/claude-501/fpix/cuneus_petri
//!   phylu-sr-watch --follow /private/tmp/claude-501/fpix/cuneus_petri
//!
//! Two regimes:
//! * `petri`: a random population (the root cell is the image row) evolves toward the original's picture;
//! * `champion`: the reference written as a gene, its e-graph variants and the saturated fold's jointly extracted
//!   forms make the population, and the search looks for an equivalent that is faster; every `--pump-every`
//!   generations the e-graph variants of the best replace the worst.
//!
//! Fitness, the two-level HFF (hff library TrueNorth, every measurement its own input, 0 best, pi/2 worst):
//! * quality group, for each training case (t = 5 and 8, `--train` size): the differing channel fraction; max |d|,
//!   mean |d| and rmse, each divided by the original's own max, mean and rms (drawing nothing is an error of 1) and
//!   mapped e -> e / (1 + e); the fraction of pixels over 1/255; the fraction of non-finite mismatches. All in sRGB
//!   display space (`image::compare`), as the cuneus host shows the picture;
//! * speed group: `--pairs` interleaved timed pairs at t = 5 against the reference program, each `r / (1 + r)` with
//!   `r` the candidate's GPU time over the reference's; each side is one process running the program `--runs`
//!   times, its time the fastest run's (the GPU's clock ramps up within a process, see `futhark_pixel::draw`);
//! * fitness = TrueNorth(theta_quality / (pi/2), theta_speed / (pi/2)).
//!
//! OCR (cuneus): the hall of fame's pictures are read by tesseract at every checkpoint and at the end
//! (`image::ocr_word`), a report measurement, not a fitness input.
//!
//! Guards: a wall-clock cap (`--minutes`) and the work bound (every candidate is drawn in tiles whose bound, node
//! executions a pixel times the tile's pixels, is at most the per-submit cap). This process's device draws the
//! originals at the start and is dropped before any Futhark program runs; then one program runs at a time.
//!
//! Checkpoints (`OUT/checkpoint.json`, every `--checkpoint-every` generations) hold the generation, the population's
//! expressed trees with their ids, the hall of fame and the counters. `--resume` re-encodes each tree into a gene of
//! the layout by the section 5 write-back (the non-coding cells are drawn afresh, so the phenotypes are the
//! checkpoint's and the non-coding region is not), and each generation's random numbers come from (seed,
//! generation), so the run continues as it would have.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::FRAC_PI_2;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use fuller::futhark_kingdom::{FutharkKingdom, GeneRng};
use fuller::futhark_pixel::{build, draw, inputs, instance, interface, load, program, random_gene, tiles, variants, work_per_pixel, Bridge, FoldEgraph, HeadDraw, Input, Instance, Tile};
use fuller::futhark_wgpu_host::compile_program;
use fuller::geneframe::Ty;
use fuller::kingdom_contract::KingdomContract;
use fuller::population_fold::AdfTable;
use fuller::saturated_fold::{fold_saturated, Budget};
use fuller::telemetry::{EventKind, GenerationStats, HofRow, Mutation, RunInfo, RunTelemetry};
use fuller::typed_gep::{express, write_back, Domain, GeneLayout, Tree};
use fuller::wgsl::cuneus_originals::case_bindings;
use fuller::wgsl::gram::truenorth;
use fuller::wgsl::image::{compare, largest_colour_share, ocr_word, quality_inputs, read_ppm, side_by_side, write_png, BindingData, Difference, ImageKernel, Ocr, Picture};
use fuller::wgsl::oracle::device::{Device, Fuel, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    arg(args, name).map_or(Ok(default), |v| v.parse::<T>().map_err(|_| format!("{name}: {v} is not a number")))
}

fn size_arg(args: &[String], name: &str, default: (u32, u32)) -> Result<(u32, u32), String> {
    match arg(args, name) {
        None => Ok(default),
        Some(s) => {
            let (w, h) = s.split_once('x').ok_or_else(|| format!("{name}: {s} is not WxH"))?;
            Ok((w.parse().map_err(|_| format!("{name}: {s}"))?, h.parse().map_err(|_| format!("{name}: {s}"))?))
        }
    }
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

/// One case: a time, its size and uniforms, the original's picture.
struct Case {
    time: f32,
    width: u32,
    height: u32,
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
    /// The first training case's difference.
    diff: Option<Difference>,
    work: u128,
    tiles: usize,
}

impl Eval {
    /// A legal gene's program that did not compile or run: our software has a bug (logged in full in `bugs.log`),
    /// and the individual takes the worst place.
    fn worst() -> Eval {
        Eval { theta: FRAC_PI_2, theta_q: FRAC_PI_2, theta_s: FRAC_PI_2, quality: vec![], speed: vec![], ratio: f64::INFINITY, diff: None, work: 0, tiles: 0 }
    }
}

#[derive(Clone)]
struct Individual {
    id: u64,
    parents: Vec<u64>,
    born: u32,
    operator: String,
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
    pairs: usize,
    /// Runs a timed process makes (Futhark's `-r`); its time is the fastest run.
    runs: usize,
    ref_exe: PathBuf,
    ref_plan: Vec<Tile>,
    work_dir: PathBuf,
    cache: HashMap<String, Eval>,
    evaluated: u64,
    hits: u64,
    bugs: u64,
    bug_log: PathBuf,
}

impl Judge<'_> {
    fn ps<'b>(&'b self, case: &'b Case) -> impl Fn(Tile) -> Result<Vec<f32>, String> + 'b {
        move |t: Tile| {
            let bl: Vec<(u32, u32, &[u8])> = case.blocks.iter().map(|(g, b, v)| (*g, *b, v.as_slice())).collect();
            inputs(self.iface, &bl, case.width, case.height, t)
        }
    }

    fn picture(&self, exe: &Path, plan: &[Tile], case: &Case, runs: usize) -> Result<(Picture, f64), String> {
        let (rgba, ns) = draw(exe, true, &self.ps(case), plan, case.width, case.height, runs)?;
        Ok((Picture { width: case.width, height: case.height, rgba }, ns))
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
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.bug_log) {
                    let _ = writeln!(f, "==== {} our software has a bug: a legal gene's program failed\n{msg}\n---- program\n{source}", utc());
                }
                Eval::worst()
            }
        };
        self.cache.insert(source.to_string(), e.clone());
        e
    }

    fn measure(&self, tree: &Tree, source: &str) -> Result<Eval, String> {
        let work = work_per_pixel(self.k, tree)?;
        let first = &self.cases[0];
        let plan = tiles(work, first.width, first.height, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
        let exe = compile_program(source, &self.work_dir)?;
        let mut quality = Vec::new();
        let mut diff = None;
        for case in &self.cases {
            let case_plan = tiles(work, case.width, case.height, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
            let (pic, _) = self.picture(&exe, &case_plan, case, 1)?;
            let (q, d) = quality_inputs(&pic, &case.original)?;
            quality.extend(q);
            if diff.is_none() {
                diff = Some(d);
            }
        }
        let ratios = self.time_pairs(&exe, &plan, self.pairs)?;
        let speed: Vec<f64> = ratios.iter().map(|r| r / (1.0 + r)).collect();
        let theta_q = truenorth(&quality);
        let theta_s = truenorth(&speed);
        let theta = truenorth(&[theta_q / FRAC_PI_2, theta_s / FRAC_PI_2]);
        Ok(Eval { theta, theta_q, theta_s, quality, speed, ratio: median(&ratios), diff, work, tiles: plan.len() })
    }
}

fn median(v: &[f64]) -> f64 {
    let mut s: Vec<f64> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
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
    /// A random gene: head cells by the half draw (`futhark_pixel::HeadDraw::Half`); drawn uniformly over the whole
    /// alphabet a block's head is 8 functions in 9 and a random tree fills the layout (about 8,000 nodes at program
    /// size 60, 2026-10-27), which no compile keeps up with.
    fn random_gene(&self, rng: &mut GeneRng) -> Vec<u32> {
        random_gene(self.k, self.contract, self.layout, self.const_len, HeadDraw::Half, rng).expect("a random gene of the layout")
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

/// An image target's pictures (`tests/corpus/targets/<name>_<w>x<h>.ppm`): the training case, the held-out case
/// (both fitness cases, the held-out one the validation inputs) and the report case; the largest colour's share of
/// each is logged (a target whose one colour covers most of the picture lets a flat program score well).
fn target_cases(name: &str, train: (u32, u32), held: (u32, u32), report: (u32, u32)) -> Result<(Vec<Case>, Vec<Case>, Case), String> {
    let load = |(w, h): (u32, u32)| -> Result<Case, String> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/corpus/targets/{name}_{w}x{h}.ppm"));
        let original = read_ppm(&path)?;
        if (original.width, original.height) != (w, h) {
            return Err(format!("{} is {}x{}, not {w}x{h}", path.display(), original.width, original.height));
        }
        let (share, bins) = largest_colour_share(&original, 4);
        println!("# target {}: largest colour {:.1} % of the pixels, {bins} of 64 colour bins", path.display(), share * 100.0);
        Ok(Case { time: 0.0, width: w, height: h, blocks: Vec::new(), original })
    };
    let check = load(held)?;
    Ok((vec![load(train)?, load(held)?], vec![load(report)?], check))
}

/// The original's pictures, drawn by the image harness; the device is dropped on return.
fn originals(shader: &str, train: (u32, u32), report: (u32, u32), check: (u32, u32)) -> Result<(Vec<Case>, Vec<Case>, Case), String> {
    let src = std::fs::read_to_string(format!("{}/tests/corpus/cuneus_visual/{shader}.wgsl", env!("CARGO_MANIFEST_DIR"))).map_err(|e| format!("original: {e}"))?;
    let dev = Device::new()?;
    let draw_case = |(w, h): (u32, u32), t: f32| -> Result<Case, String> {
        let b = case_bindings(shader, t, w, h)?;
        let k = ImageKernel::compile(&dev, shader, &src, "main", &b, w, h, Fuel::Proven("a cuneus original"))?;
        let run = k.run(1, 1)?;
        Ok(Case { time: t, width: w, height: h, blocks: blocks_of(&b), original: run.picture })
    };
    let train_cases = [5.0f32, 8.0].iter().map(|t| draw_case(train, *t)).collect::<Result<Vec<_>, _>>()?;
    let report_cases = [2.0f32, 5.0, 8.0].iter().map(|t| draw_case(report, *t)).collect::<Result<Vec<_>, _>>()?;
    Ok((train_cases, report_cases, draw_case(check, 5.0)?))
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

fn tree_json(t: &Tree) -> serde_json::Value {
    serde_json::json!([t.symbol, t.const_index, t.kids.iter().map(tree_json).collect::<Vec<_>>()])
}

fn tree_of_json(v: &serde_json::Value) -> Result<Tree, String> {
    let a = v.as_array().filter(|a| a.len() == 3).ok_or("a checkpoint tree is not [symbol, const, kids]")?;
    Ok(Tree {
        symbol: a[0].as_u64().ok_or("a checkpoint symbol")? as u32,
        const_index: a[1].as_u64().map(|c| c as u32),
        kids: a[2].as_array().ok_or("checkpoint kids")?.iter().map(tree_of_json).collect::<Result<_, _>>()?,
    })
}

/// The hash of a program's text, as the stream's tree identity.
fn text_hash(s: &str) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// Everything the run carries between generations.
struct Run<'a> {
    inst: Instance,
    shader: String,
    regime: String,
    out: PathBuf,
    gen: Genetics<'a>,
    judge: Judge<'a>,
    tel: RunTelemetry,
    ledger: std::fs::File,
    next_id: u64,
    hof: Vec<Individual>,
    check_case: Case,
}

impl Run<'_> {
    fn ledger_line(&mut self, generation: u32, ind: &Individual) -> Result<(), String> {
        let d = ind.eval.diff.as_ref();
        writeln!(
            self.ledger,
            "{}\t{generation}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.4}\t{}\t{}\t{}\t{}",
            ind.id,
            ind.parents.iter().map(u64::to_string).collect::<Vec<_>>().join(","),
            ind.operator,
            ind.eval.theta,
            ind.eval.theta_q,
            ind.eval.theta_s,
            ind.eval.ratio,
            d.map_or(0, |d| d.differing_channels),
            ind.tree.level_order().len(),
            ind.eval.work,
            text_hash(&ind.source)
        )
        .map_err(|e| format!("the ledger: {e}"))
    }

    /// The hall of fame's pictures at the checkpoint size (original beside each), OCR of each (cuneus), the stream's
    /// hall-of-fame record and a note with the OCR texts. Returns the best's OCR.
    fn hof_checkpoint(&mut self, generation: u32) -> Result<Option<Ocr>, String> {
        let mut rows = Vec::new();
        let mut best_ocr = None;
        let mut notes = Vec::new();
        for (r, h) in self.hof.clone().iter().enumerate() {
            let exe = compile_program(&h.source, &self.out.join("hof_build"))?;
            let case = &self.check_case;
            let plan = tiles(h.eval.work.max(1), case.width, case.height, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
            let (pic, _) = self.judge.picture(&exe, &plan, case, 1)?;
            let png = self.out.join("hall_of_fame").join(format!("g{generation:04}_rank{}.png", r + 1));
            write_png(&png, &side_by_side(&[case.original.clone(), pic.clone()])?)?;
            let mut extras = BTreeMap::new();
            if let Some(d) = &h.eval.diff {
                extras.insert("differing_channels".to_string(), d.differing_channels as f64);
                extras.insert("mean_abs".to_string(), d.mean_abs);
            }
            extras.insert("theta_quality".to_string(), h.eval.theta_q);
            extras.insert("theta_speed".to_string(), h.eval.theta_s);
            extras.insert("speed_ratio".to_string(), h.eval.ratio);
            if self.shader == "cuneus" {
                let o = ocr_word(&pic, &self.out.join("hall_of_fame").join(format!("g{generation:04}_rank{}_ocr.png", r + 1)), "CUNEUS")?;
                extras.insert("ocr_match".to_string(), f64::from(u8::from(o.matches)));
                extras.insert("ocr_distance".to_string(), o.distance as f64);
                extras.insert("ocr_confidence".to_string(), o.confidence);
                notes.push(format!("rank {} OCR {:?} (conf {:.1}, distance {})", r + 1, o.text, o.confidence, o.distance));
                if r == 0 {
                    best_ocr = Some(o);
                }
            }
            rows.push(HofRow { rank: r as u32 + 1, hff: Some(h.eval.theta), born: h.born, tree: text_hash(&h.source), nodes: Some(h.tree.level_order().len() as u32), gpu_ms: None, picture: Some(png.display().to_string()), extras });
        }
        self.tel.hall_of_fame(generation, rows)?;
        if !notes.is_empty() {
            self.tel.note(generation, notes.join("; "))?;
        }
        Ok(best_ocr)
    }

    fn checkpoint(&self, generation: u32, population: &[Individual], seconds: f64) -> Result<(), String> {
        let ind = |i: &Individual| serde_json::json!({"id": i.id, "parents": i.parents, "born": i.born, "operator": i.operator, "tree": tree_json(&i.tree)});
        let v = serde_json::json!({
            "shader": self.shader, "regime": self.regime, "generation": generation, "seconds": seconds, "next_id": self.next_id,
            "evaluated": self.judge.evaluated, "hits": self.judge.hits, "bugs": self.judge.bugs, "best_ever": self.tel.best_ever(),
            "population": population.iter().map(ind).collect::<Vec<_>>(),
            "hof": self.hof.iter().map(ind).collect::<Vec<_>>(),
        });
        let tmp = self.out.join("checkpoint.json.new");
        std::fs::write(&tmp, v.to_string()).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, self.out.join("checkpoint.json")).map_err(|e| e.to_string())
    }
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let shader = arg(&args, "--shader").unwrap_or_else(|| "volumepassage".into());
    let regime = arg(&args, "--regime").ok_or("--regime petri|champion")?;
    if regime != "petri" && regime != "champion" {
        return Err(format!("--regime {regime}: petri or champion"));
    }
    let wide = shader == "cuneus";
    let image_target = shader == "image";
    let target = arg(&args, "--target").unwrap_or_else(|| "bugberman".into());
    if image_target && regime == "champion" {
        return Err("the image target has no reference: the champion regime needs one".into());
    }
    let minutes: f64 = num(&args, "--minutes", 30.0)?;
    let pop: usize = num(&args, "--pop", 16)?;
    let seed: u64 = num(&args, "--seed", 1)?;
    let train = size_arg(&args, "--train", if image_target { (96, 88) } else if wide { (96, 48) } else { (64, 64) })?;
    let report = size_arg(&args, "--report", if image_target { (512, 469) } else if wide { (512, 256) } else { (128, 128) })?;
    let check = size_arg(&args, "--check", if image_target { (128, 117) } else if wide { (256, 128) } else { (128, 128) })?;
    let pairs: usize = num(&args, "--pairs", 3)?;
    let runs: usize = num(&args, "--runs", 40)?;
    let elites: usize = num(&args, "--elites", 2)?;
    let pump_every: usize = num(&args, "--pump-every", 5)?;
    let checkpoint_every: u32 = num(&args, "--checkpoint-every", 10)?;
    let max_generations: u32 = num(&args, "--generations", 100_000)?;
    let resume = args.iter().any(|a| a == "--resume");
    let out = PathBuf::from(arg(&args, "--out").ok_or("--out DIR")?);
    std::fs::create_dir_all(out.join("hall_of_fame")).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let low_change = regime == "champion";
    let checkpoint: Option<serde_json::Value> = if resume {
        let text = std::fs::read_to_string(out.join("checkpoint.json")).map_err(|e| format!("--resume: {e}"))?;
        Some(serde_json::from_str(&text).map_err(|e| format!("the checkpoint: {e}"))?)
    } else {
        None
    };
    let spent = checkpoint.as_ref().and_then(|c| c["seconds"].as_f64()).unwrap_or(0.0);
    let limit = std::time::Duration::from_secs_f64((minutes * 60.0 - spent).max(0.0));
    println!("# {} futhark_pixel_evolve: {shader}, regime {regime}, {minutes} min, population {pop}, seed {seed}, train {}x{}, {pairs} timed pairs of {runs} runs, elites {elites}{}", utc(), train.0, train.1, if resume { ", RESUMED" } else { "" });

    let (train_cases, report_cases, check_case) = if image_target { target_cases(&target, train, check, report)? } else { originals(&shader, train, report, check)? };
    println!("# {} the original drawn: training {}x{} at t = 5, 8; report {}x{}; checkpoint pictures {}x{}; the device is dropped", utc(), train.0, train.1, report.0, report.1, check.0, check.1);

    let (inst, ref_text) = instance(&shader)?;
    let k = load(&inst)?;
    let iface = interface(&inst)?;
    let contract = k.relations().check().map_err(|d| format!("the kingdom check: {:?}", &d[..d.len().min(3)]))?;
    // The reference: the hand-written program for a shader; for an image target, which has none, the speed group's
    // yardstick is the simplest picture, the position's colour (`coord`), and the program size defaults to 60.
    let reference = build(&k, ref_text.unwrap_or("(image coord)"))?;
    let n_funcs = reference.level_order().iter().filter(|(t, _)| !k.rows[t.symbol as usize].inputs.is_empty()).count();
    let root = k.ty("image").ok_or("no image dual")?;
    let program_size: usize = num(&args, "--program-size", if image_target { 60 } else { n_funcs })?;
    if regime == "champion" && program_size < n_funcs {
        return Err(format!("--program-size {program_size}: the champion regime needs the reference's {n_funcs} functions"));
    }
    let layout = GeneLayout::from_program_size(&contract, root, program_size).map_err(|e| e.to_string())?;
    let const_len = k.const_len();
    println!("# layout: program size N = {program_size} (the reference has {n_funcs} functions), {} words a gene", layout.width);

    let ref_source = program(&k, &iface, &reference)?;
    let ref_exe = compile_program(&ref_source, &out.join("reference_build"))?;
    let ref_plan = tiles(work_per_pixel(&k, &reference)?, train.0, train.1, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
    let mut objectives = Vec::new();
    let case_names: Vec<String> = if image_target { vec![format!("train {}x{}", train.0, train.1), format!("held-out {}x{}", check.0, check.1)] } else { vec!["t=5".into(), "t=8".into()] };
    for c in &case_names {
        for m in ["differing", "max_abs_rel", "mean_abs_rel", "rmse_rel", "over_1_255", "non_finite"] {
            objectives.push(format!("quality {c} {m}"));
        }
    }
    for p in 0..pairs {
        objectives.push(format!("speed pair {p} r/(1+r)"));
    }
    let card = serde_json::json!({"shader": shader, "regime": regime, "minutes": minutes, "pop": pop, "seed": seed, "train": train, "pairs": pairs, "runs": runs, "elites": elites, "pump_every": pump_every, "checkpoint_every": checkpoint_every, "program_size": program_size, "gene_words": layout.width, "fitness": "TrueNorth(theta_quality/(pi/2), theta_speed/(pi/2))"});
    let tel = RunTelemetry::start(
        &out,
        format!("futhark-pixel-{shader}-{regime}-seed{seed}"),
        RunInfo {
            kingdom: "futhark_pixel".into(),
            dataset: if image_target { format!("image target {target} {}x{}", train.0, train.1) } else { format!("{shader} {}x{}", train.0, train.1) },
            seed: seed as u32,
            population: pop as u32,
            max_generations,
            budget_ms: (minutes * 60_000.0) as u64,
            hff_objectives: objectives,
            config_json: Some(card.to_string()),
            resume,
            already_spent_seconds: spent,
        },
    )?;
    let ledger = std::fs::OpenOptions::new().create(true).append(true).open(out.join("ledger.tsv")).map_err(|e| format!("the ledger: {e}"))?;
    let mut run = Run {
        inst,
        shader: shader.clone(),
        regime: regime.clone(),
        out: out.clone(),
        gen: Genetics { k: &k, contract: &contract, layout: &layout, const_len: &const_len },
        judge: Judge { k: &k, iface: &iface, cases: train_cases, pairs, runs, ref_exe, ref_plan, work_dir: out.join("candidate_build"), cache: HashMap::new(), evaluated: 0, hits: 0, bugs: 0, bug_log: out.join("bugs.log") },
        tel,
        ledger,
        next_id: 0,
        hof: Vec::new(),
        check_case,
    };
    if !resume {
        writeln!(run.ledger, "id\tgeneration\tparents\toperator\ttheta\ttheta_q\ttheta_s\tratio\tdiffering_t5\tnodes\twork\tprogram_hash").map_err(|e| e.to_string())?;
    }

    // One pair to warm the GPU after the image harness, discarded (the first pair of a trial measured 2.36).
    run.judge.time_pairs(&run.judge.ref_exe.clone(), &run.judge.ref_plan.clone(), 1)?;
    let noise = run.judge.time_pairs(&run.judge.ref_exe.clone(), &run.judge.ref_plan.clone(), 11)?;
    println!("# {} the timing's noise floor, the reference against itself, 11 pairs of {runs} runs: ratio median {:.4}, min {:.4}, max {:.4}", utc(), median(&noise), noise.iter().cloned().fold(f64::INFINITY, f64::min), noise.iter().cloned().fold(0.0, f64::max));
    let ref_eval = run.judge.evaluate(&reference, &ref_source);
    println!("# {} the reference: HFF {:.5} (quality {:.5}, speed {:.5}), ratio {:.4}, {} node executions a pixel, {} tile(s); differing channels {}", utc(), ref_eval.theta, ref_eval.theta_q, ref_eval.theta_s, ref_eval.ratio, ref_eval.work, ref_eval.tiles, ref_eval.diff.as_ref().map_or(0, |d| d.differing_channels));

    let bridge = Bridge::load(&run.inst)?;
    let mut rng0 = GeneRng::new(seed);
    let ref_gene = write_back(&contract, &layout, &run.gen.random_gene(&mut rng0), &reference).map_err(|e| format!("the reference's write-back: {e}"))?;
    let fresh = |run: &mut Run<'_>, parents: Vec<u64>, operator: &str, gene: Vec<u32>, born: u32| -> Result<Individual, String> {
        let tree = run.gen.tree(&gene)?;
        let source = program(&k, &iface, &tree)?;
        run.next_id += 1;
        Ok(Individual { id: run.next_id, parents, born, operator: operator.to_string(), gene, tree, source, eval: Eval::worst() })
    };
    let mut generation: u32;
    let mut cands: Vec<Individual> = Vec::new();
    if let Some(c) = &checkpoint {
        generation = c["generation"].as_u64().ok_or("checkpoint generation")? as u32 + 1;
        run.next_id = c["next_id"].as_u64().ok_or("checkpoint next_id")?;
        run.judge.evaluated = c["evaluated"].as_u64().unwrap_or(0);
        run.judge.hits = c["hits"].as_u64().unwrap_or(0);
        run.judge.bugs = c["bugs"].as_u64().unwrap_or(0);
        run.tel.restore(c["best_ever"].as_f64(), run.judge.evaluated);
        let load_inds = |v: &serde_json::Value, rng: &mut GeneRng, run: &Run<'_>| -> Result<Vec<Individual>, String> {
            v.as_array()
                .ok_or("checkpoint individuals")?
                .iter()
                .map(|i| {
                    let tree = tree_of_json(&i["tree"])?;
                    let gene = write_back(&contract, &layout, &run.gen.random_gene(rng), &tree).map_err(|e| format!("re-encoding a checkpoint tree: {e}"))?;
                    Ok(Individual {
                        id: i["id"].as_u64().unwrap_or(0),
                        parents: i["parents"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default(),
                        born: i["born"].as_u64().unwrap_or(0) as u32,
                        operator: i["operator"].as_str().unwrap_or("").to_string(),
                        source: program(&k, &iface, &tree)?,
                        gene,
                        tree,
                        eval: Eval::worst(),
                    })
                })
                .collect()
        };
        let mut rng = GeneRng::new(seed ^ 0x5245_5355_4d45);
        cands = load_inds(&c["population"], &mut rng, &run)?;
        run.hof = load_inds(&c["hof"], &mut rng, &run)?;
        let mut hof = std::mem::take(&mut run.hof);
        for h in hof.iter_mut() {
            h.eval = run.judge.evaluate(&h.tree, &h.source);
        }
        run.hof = hof;
        println!("# {} resumed at generation {generation}: {} individuals, {} in the hall of fame, re-encoded from their trees", utc(), cands.len(), run.hof.len());
    } else {
        generation = 0;
        if regime == "petri" {
            while cands.len() < pop {
                let g = run.gen.random_gene(&mut rng0);
                let ind = fresh(&mut run, vec![], "random", g, 0)?;
                cands.push(ind);
            }
        } else {
            let r = fresh(&mut run, vec![], "reference", ref_gene.clone(), 0)?;
            let ref_id = r.id;
            cands.push(r);
            let vs = variants(&run.inst, &k, &bridge, &reference, 2, 16, pop * 2)?;
            println!("# {} e-graph: {} variants of the reference", utc(), vs.len());
            for v in &vs {
                let g = write_back(&contract, &layout, &ref_gene, v).map_err(|e| format!("a variant's write-back: {e}"))?;
                let ind = fresh(&mut run, vec![ref_id], "egraph_exact", g, 0)?;
                let mut m = Mutation::new(EventKind::EgraphExact, "egraph_variant", "an e-graph variant of the reference (bit-identical IEEE rules)".into());
                m.id = Some(ind.id);
                m.parents = vec![ref_id];
                m.text_before = Some(ref_source.clone());
                m.text_after = Some(ind.source.clone());
                run.tel.mutation(0, m)?;
                cands.push(ind);
            }
            let trees: Vec<Tree> = cands.iter().map(|c| c.tree.clone()).collect();
            let genes: Vec<Vec<u32>> = cands.iter().map(|c| c.gene.clone()).collect();
            let mut table = AdfTable::default();
            let fe = FoldEgraph { inst: &run.inst, kingdom: &k, bridge: &bridge };
            let fold = fold_saturated(&mut table, &fe, &contract, &layout, &const_len, &genes, &trees, &Budget::default())?;
            println!("# {} saturated fold over {} genes: {} distinct trees, {} iterations, {} shared parts (structural hashing {}), {} forms differ from their submitted tree", utc(), genes.len(), fold.distinct, fold.iterations, fold.used.len(), fold.structural.adfs, fold.reformed);
            let mut m = Mutation::new(EventKind::SaturatedFold, "saturated_fold", format!("{} shared parts over {} genes ({} by structural hashing); {} forms differ", fold.used.len(), genes.len(), fold.structural.adfs, fold.reformed));
            m.count = Some(fold.used.len() as u32);
            m.nodes = Some(fold.reformed as u32);
            run.tel.mutation(0, m)?;
            let mut added = 0;
            for (i, f) in fold.forms.iter().enumerate() {
                if !trees.contains(f) && !cands.iter().any(|c| &c.tree == f) {
                    let g = write_back(&contract, &layout, &ref_gene, f).map_err(|e| format!("a fold form's write-back: {e}"))?;
                    let parent = cands[i].id;
                    let ind = fresh(&mut run, vec![parent], "saturated_fold", g, 0)?;
                    cands.push(ind);
                    added += 1;
                }
            }
            println!("# the fold's jointly extracted forms add {added} new trees");
            cands.truncate(pop);
        }
    }

    let mut series = if resume { std::fs::read_to_string(out.join("series.tsv")).unwrap_or_default() } else { String::new() };
    if series.is_empty() {
        series = String::from("generation\tutc\tseconds\tevaluated\tcache_hits\tbugs\tbest_theta\tbest_theta_q\tbest_theta_s\tmedian_theta\tbest_ratio\tbest_differing\tbest_mean_abs\tbest_nodes\tbest_work\tdistinct\tocr_text\tocr_match\tocr_confidence\tocr_distance\n");
    }
    let mut population: Vec<Individual> = Vec::new();
    let mut stopped_by = "n_gen";
    let mut last_generation = generation;
    while generation < max_generations {
        if run.tel.stop_requested() {
            stopped_by = "killed";
            break;
        }
        // ---- evaluate ----
        let mut next = Vec::with_capacity(cands.len());
        let mut timed_out = false;
        for mut c in std::mem::take(&mut cands) {
            if started.elapsed() > limit {
                timed_out = true;
                break;
            }
            c.eval = run.judge.evaluate(&c.tree, &c.source);
            run.ledger_line(generation, &c)?;
            next.push(c);
        }
        if timed_out {
            stopped_by = "time";
            break;
        }
        population = next;
        population.sort_by(|a, b| a.eval.theta.total_cmp(&b.eval.theta));
        for ind in &population {
            if !run.hof.iter().any(|h| h.source == ind.source) {
                run.hof.push(ind.clone());
            }
        }
        run.hof.sort_by(|a, b| a.eval.theta.total_cmp(&b.eval.theta));
        run.hof.truncate(5);
        let thetas: Vec<f64> = population.iter().map(|i| i.eval.theta).collect();
        let distinct = {
            let mut s: Vec<&str> = population.iter().map(|i| i.source.as_str()).collect();
            s.sort_unstable();
            s.dedup();
            s.len()
        };
        let checkpoint_now = generation.is_multiple_of(checkpoint_every.max(1));
        let ocr = if checkpoint_now { run.hof_checkpoint(generation)? } else { None };
        let best = population[0].clone();
        let d = best.eval.diff.as_ref();
        let mut groups = BTreeMap::new();
        groups.insert("quality".to_string(), best.eval.theta_q);
        groups.insert("speed".to_string(), best.eval.theta_s);
        let mut extras = BTreeMap::new();
        extras.insert("median_hff".to_string(), median(&thetas));
        extras.insert("speed_ratio".to_string(), best.eval.ratio);
        extras.insert("differing_channels".to_string(), d.map_or(f64::NAN, |d| d.differing_channels as f64));
        extras.insert("mean_abs".to_string(), d.map_or(f64::NAN, |d| d.mean_abs));
        extras.insert("cache_hits".to_string(), run.judge.hits as f64);
        extras.insert("bugs".to_string(), run.judge.bugs as f64);
        if let Some(o) = &ocr {
            extras.insert("ocr_match".to_string(), f64::from(u8::from(o.matches)));
            extras.insert("ocr_distance".to_string(), o.distance as f64);
            extras.insert("ocr_confidence".to_string(), o.confidence);
        }
        let evaluated_before = run.tel.individuals();
        run.tel.generation(GenerationStats { generation, fitness: &thetas, distinct: Some(distinct as u32), angle_groups: groups, extras, evaluated: run.judge.evaluated.saturating_sub(evaluated_before) })?;
        series.push_str(&format!(
            "{generation}\t{}\t{:.1}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}\t{}\t{:.3e}\t{}\t{}\t{distinct}\t{}\t{}\t{}\t{}\n",
            utc(),
            spent + started.elapsed().as_secs_f64(),
            run.judge.evaluated,
            run.judge.hits,
            run.judge.bugs,
            best.eval.theta,
            best.eval.theta_q,
            best.eval.theta_s,
            median(&thetas),
            best.eval.ratio,
            d.map_or(0, |d| d.differing_channels),
            d.map_or(f64::NAN, |d| d.mean_abs),
            best.tree.level_order().len(),
            best.eval.work,
            ocr.as_ref().map_or(String::new(), |o| format!("{:?}", o.text)),
            ocr.as_ref().map_or(String::new(), |o| o.matches.to_string()),
            ocr.as_ref().map_or(String::new(), |o| format!("{:.1}", o.confidence)),
            ocr.as_ref().map_or(String::new(), |o| o.distance.to_string()),
        ));
        std::fs::write(out.join("series.tsv"), &series).map_err(|e| e.to_string())?;
        println!(
            "{} gen {generation}: best HFF {:.5} (q {:.5}, s {:.5}, ratio {:.3}), median {:.5}, {distinct} distinct; {} evaluated, {} cache hits, {} bugs{}",
            utc(),
            best.eval.theta,
            best.eval.theta_q,
            best.eval.theta_s,
            best.eval.ratio,
            median(&thetas),
            run.judge.evaluated,
            run.judge.hits,
            run.judge.bugs,
            ocr.as_ref().map_or(String::new(), |o| format!("; OCR of the best {:?} (distance {})", o.text, o.distance))
        );
        if checkpoint_now {
            run.checkpoint(generation, &population, spent + started.elapsed().as_secs_f64())?;
        }
        last_generation = generation;

        // ---- the next generation; its random numbers come from (seed, generation), so a resume continues it ----
        let mut rng = GeneRng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ u64::from(generation + 1));
        let mut children: Vec<Individual> = Vec::with_capacity(pop);
        let mut taken: Vec<String> = Vec::new();
        for ind in &population {
            if children.len() >= elites {
                break;
            }
            if !taken.contains(&ind.source) {
                taken.push(ind.source.clone());
                let mut e = ind.clone();
                e.operator = "elite".into();
                children.push(e);
            }
        }
        if low_change && pump_every > 0 && generation > 0 && (generation as usize).is_multiple_of(pump_every) {
            // The pump: the e-graph variants of the best enter the population.
            let b = population[0].clone();
            for v in variants(&run.inst, &k, &bridge, &b.tree, 2, 16, 4)? {
                if children.len() < pop {
                    if let Ok(g) = write_back(&contract, &layout, &b.gene, &v) {
                        let ind = fresh(&mut run, vec![b.id], "pump_egraph", g, generation + 1)?;
                        let mut m = Mutation::new(EventKind::Pump, "egraph_variant", "the pump: an e-graph variant of the best".into());
                        m.id = Some(ind.id);
                        m.parents = vec![b.id];
                        m.text_before = Some(b.source.clone());
                        m.text_after = Some(ind.source.clone());
                        m.fitness_before = Some(b.eval.theta);
                        m.entered = Some(true);
                        run.tel.mutation(generation, m)?;
                        children.push(ind);
                    }
                }
            }
        }
        let tournament = |rng: &mut GeneRng| -> usize {
            let mut best = rng.below(population.len());
            for _ in 0..2 {
                let c = rng.below(population.len());
                if population[c].eval.theta < population[best].eval.theta {
                    best = c;
                }
            }
            best
        };
        while children.len() < pop {
            let a = tournament(&mut rng);
            let mut gene = population[a].gene.clone();
            let mut parents = vec![population[a].id];
            let mut op = String::from("mutation");
            if rng.below(2) == 0 {
                // Crossover parents are different expressed trees; with no such partner, no crossover.
                let mut b = tournament(&mut rng);
                for _ in 0..8 {
                    if population[b].source != population[a].source {
                        break;
                    }
                    b = tournament(&mut rng);
                }
                if population[b].source != population[a].source {
                    run.gen.crossover(&mut gene, &population[b].gene, &mut rng)?;
                    parents.push(population[b].id);
                    op = "crossover+mutation".into();
                }
            }
            let m = if low_change { 1 + rng.below(2) } else { 1 + rng.below(4) };
            run.gen.mutate(&mut gene, m, &mut rng)?;
            let ind = fresh(&mut run, parents, &op, gene, generation + 1)?;
            children.push(ind);
        }
        cands = children;
        generation += 1;
    }
    if population.is_empty() {
        return Err("no generation completed within the time limit".into());
    }
    let seconds = spent + started.elapsed().as_secs_f64();
    run.checkpoint(last_generation, &population, seconds)?;
    println!("# {} stopped ({stopped_by}) after generation {last_generation}, {:.1} min; {} evaluated, {} cache hits, {} bugs", utc(), seconds / 60.0, run.judge.evaluated, run.judge.hits, run.judge.bugs);

    // ---- the report ----
    let final_ocr = run.hof_checkpoint(last_generation)?;
    let best = run.hof[0].clone();
    for (i, h) in run.hof.iter().enumerate() {
        std::fs::write(out.join("hall_of_fame").join(format!("rank{}.fut", i + 1)), &h.source).map_err(|e| e.to_string())?;
    }
    std::fs::write(out.join("reference.fut"), &ref_source).map_err(|e| e.to_string())?;
    let best_exe = compile_program(&best.source, &out.join("best_build"))?;
    let best_plan_r = tiles(best.eval.work.max(1), report.0, report.1, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
    let ref_plan_r = tiles(work_per_pixel(&k, &reference)?, report.0, report.1, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
    let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.5}")).collect::<Vec<_>>().join(",");
    let mut rep = format!(
        "shader\t{shader}\nregime\t{regime}\nseed\t{seed}\nstopped_by\t{stopped_by}\nlast_generation\t{last_generation}\nminutes\t{:.1}\nevaluated\t{}\ncache_hits\t{}\nbugs\t{}\nreference_hff\t{:.6}\nreference_theta_quality\t{:.6}\nreference_theta_speed\t{:.6}\nbest_hff\t{:.6}\nbest_theta_quality\t{:.6}\nbest_theta_speed\t{:.6}\nbest_born\t{}\nbest_operator\t{}\nbest_nodes\t{}\nreference_nodes\t{}\nbest_work_per_pixel\t{}\nreference_work_per_pixel\t{}\nbest_quality_inputs\t{}\nbest_speed_inputs\t{}\nreference_quality_inputs\t{}\nreference_speed_inputs\t{}\n",
        seconds / 60.0,
        run.judge.evaluated,
        run.judge.hits,
        run.judge.bugs,
        ref_eval.theta,
        ref_eval.theta_q,
        ref_eval.theta_s,
        best.eval.theta,
        best.eval.theta_q,
        best.eval.theta_s,
        best.born,
        best.operator,
        best.tree.level_order().len(),
        reference.level_order().len(),
        best.eval.work,
        ref_eval.work,
        fmt(&best.eval.quality),
        fmt(&best.eval.speed),
        fmt(&ref_eval.quality),
        fmt(&ref_eval.speed)
    );
    if let Some(o) = &final_ocr {
        rep.push_str(&format!("best_ocr_text_{}x{}\t{:?}\nbest_ocr_match\t{}\nbest_ocr_confidence\t{:.1}\nbest_ocr_distance\t{}\n", check.0, check.1, o.text, o.matches, o.confidence, o.distance));
    }
    rep.push_str("case\tpicture\tdiffering_channels\tchannels\tmax_abs\tmean_abs\trmse\tpixels_over_1_255\tnon_finite\tocr_text\tocr_match\tocr_confidence\tocr_distance\n");
    for case in &report_cases {
        let (bp, _) = run.judge.picture(&best_exe, &best_plan_r, case, 1)?;
        let (rp, _) = run.judge.picture(&run.judge.ref_exe, &ref_plan_r, case, 1)?;
        for (name, p) in [("original", &case.original), ("best", &bp), ("reference", &rp)] {
            let d = compare(p, &case.original)?;
            let o = if shader == "cuneus" { Some(ocr_word(p, &out.join(format!("ocr_{name}_{}x{}_t{}.png", report.0, report.1, case.time)), "CUNEUS")?) } else { None };
            rep.push_str(&format!(
                "{}x{} t={}\t{name} vs original\t{}\t{}\t{:e}\t{:e}\t{:e}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                report.0,
                report.1,
                case.time,
                d.differing_channels,
                d.pixels * 4,
                d.max_abs,
                d.mean_abs,
                d.rmse,
                d.pixels_over_one_step,
                d.non_finite_mismatch,
                o.as_ref().map_or(String::new(), |o| format!("{:?}", o.text)),
                o.as_ref().map_or(String::new(), |o| o.matches.to_string()),
                o.as_ref().map_or(String::new(), |o| format!("{:.1}", o.confidence)),
                o.as_ref().map_or(String::new(), |o| o.distance.to_string()),
            ));
        }
        write_png(&out.join(format!("original_best_reference_{}x{}_t{}.png", report.0, report.1, case.time)), &side_by_side(&[case.original.clone(), bp, rp])?)?;
    }
    // The speed: 11 interleaved pairs at the training size, t = 5.
    let best_plan = tiles(best.eval.work.max(1), train.0, train.1, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
    let ratios = run.judge.time_pairs(&best_exe, &best_plan, 11)?;
    let faster = ratios.iter().filter(|r| **r < 1.0).count();
    rep.push_str(&format!("noise_floor_ratios_reference_vs_itself\t{}\nspeed_pairs\t11\nspeed_ratio_median\t{:.4}\nspeed_ratio_min\t{:.4}\nspeed_ratio_max\t{:.4}\nfaster_rounds\t{faster}\n", fmt(&noise), median(&ratios), ratios.iter().cloned().fold(f64::INFINITY, f64::min), ratios.iter().cloned().fold(0.0, f64::max)));
    std::fs::write(out.join("report.tsv"), &rep).map_err(|e| e.to_string())?;
    print!("{rep}");

    let chart = std::process::Command::new("python3")
        .arg("-I")
        .arg("-c")
        .arg(CHART)
        .arg(out.join("series.tsv"))
        .arg(out.join("chart_hff_vs_generation.png"))
        .arg(format!("{:.6}", ref_eval.theta))
        .arg(format!("{shader} in Futhark, regime {regime}: HFF against generation"))
        .output()
        .map_err(|e| format!("python3: {e}"))?;
    if !chart.status.success() {
        return Err(format!("the chart failed: {}", String::from_utf8_lossy(&chart.stderr)));
    }
    run.tel.end(stopped_by, last_generation, Some(best.eval.theta))?;
    println!("# {} done: {}", utc(), out.display());
    Ok(())
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
