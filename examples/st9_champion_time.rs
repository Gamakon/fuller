//! ST9: does the evolved champion draw volumepassage faster than the original? (fuller `docs/PLAN_true_gep_wgsl.md`)
//!
//!   cargo run --release --features gpu,wgsl --example st9_champion_time -- --hof RUN/hof.bin --rounds 61 \
//!       --sizes 256,512 --out DIR
//!
//! Reads the hall of fame a phylu `wgsl_true_image_fit` run saved beside its checkpoint (`hof.bin`, phylu's checkpoint
//! format: magic, generation, count, row words, two f64, the genes, the fitness of each), expresses each gene in the
//! volumepassage image instance and renders its kernel. For every size, every kernel is timed in two dispatch modes:
//!
//! - tiled: the submits of whole workgroups the run used (`Fuel::Tree` with the tree's own work bound), one submit
//!   at a time with a wait between, the encoded original as the reference, as in the run's fitness;
//! - one submit: the whole dispatch at once, as the shipped original runs (`Fuel::Proven`). A candidate runs this way
//!   only when its work bound is at most the encoded original's, i.e. it does no more work per invocation than the
//!   shipped original; otherwise it is not timed in this mode and the report says so.
//!
//! Each round runs every kernel once, in a random order drawn per round (interleaved, so the bimodal GPU clock affects
//! all of them alike); GPU ms is the sum of the pass's timestamp intervals. Pictures of every kernel in both modes are
//! compared with the shipped original's at every size and time value of the run (t 2, 5, 8 at each size, and the run's
//! held-out 200 x 136 at t 5). Outputs: `times.tsv` (every pass), `summary.tsv`, `ratios.tsv`, `pictures.tsv`, each
//! kernel's WGSL, and side-by-side frames (shipped original | champion) at each time.

use std::io::Write;
use std::path::{Path, PathBuf};

use fuller::typed_gep::{express, GeneLayout, Tree};
use fuller::wgsl::cuneus_originals::{case_bindings, image_dir, volumepassage};
use fuller::wgsl::image::{compare, median, side_by_side, stamp, write_png, ImageKernel, Picture};
use fuller::wgsl::oracle::device::{Device, Fuel};
use fuller::wgsl::true_kingdom::{Rng, TrueKingdom};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn now() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    format!("{:02}:{:02}:{:02}Z", (t / 3600) % 24, (t / 60) % 60, t % 60)
}

fn functions(t: &Tree, k: &TrueKingdom) -> usize {
    usize::from(!k.rows[t.symbol as usize].args.is_empty()) + t.kids.iter().map(|c| functions(c, k)).sum::<usize>()
}

/// The time block of the cuneus hosts (`time, delta, frame, pad`), as the run wrote it.
fn time_bytes(time: f32) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&time.to_le_bytes());
    b.extend_from_slice(&0.016f32.to_le_bytes());
    b.extend_from_slice(&300u32.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b
}

/// A hall of fame saved by phylu `true_wgsl_fit` (`Checkpoint::write`): generation, genes, fitness.
struct Hof {
    generation: u32,
    genes: Vec<Vec<u32>>,
    fitness: Vec<f32>,
}

const CHECKPOINT_MAGIC: u64 = 0x5354_385f_434b_5031;

fn read_hof(path: &Path) -> Result<Hof, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() < 36 {
        return Err(format!("{}: {} bytes, not a checkpoint", path.display(), b.len()));
    }
    let word = |at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
    let magic = u64::from(word(0)) | (u64::from(word(4)) << 32);
    if magic != CHECKPOINT_MAGIC {
        return Err(format!("{}: not a checkpoint (magic {magic:#x})", path.display()));
    }
    let (generation, count, row) = (word(8), word(12) as usize, word(16) as usize);
    if b.len() != 36 + 4 * (count * row + count) {
        return Err(format!("{}: {} bytes for {count} genes of {row} words", path.display(), b.len()));
    }
    let genes = (0..count).map(|g| (0..row).map(|i| word(36 + 4 * (g * row + i))).collect()).collect();
    let fitness = (0..count).map(|i| f32::from_bits(word(36 + 4 * (count * row + i)))).collect();
    Ok(Hof { generation, genes, fitness })
}

/// The `q` quantile by nearest rank of a sorted copy.
fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() {
        return f64::NAN;
    }
    let i = ((q * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1;
    s[i]
}

/// Two-sided exact binomial sign test: `k` of `n` below one, p = 0.5.
fn sign_test(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut pmf = vec![0f64; n + 1];
    let ln_half = 0.5f64.ln() * n as f64;
    let mut ln_c = 0f64;
    for (i, p) in pmf.iter_mut().enumerate() {
        if i > 0 {
            ln_c += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        *p = (ln_c + ln_half).exp();
    }
    let tail = k.min(n - k);
    let one: f64 = pmf[..=tail].iter().sum();
    (2.0 * one).min(1.0)
}

/// Median of the per-round ratios and its 95 % bootstrap interval over rounds (deterministic draw).
fn ratio_stats(r: &[f64], seed: u64) -> (f64, f64, f64) {
    let mut rng = Rng(seed);
    let mut meds: Vec<f64> = (0..4000).map(|_| median(&(0..r.len()).map(|_| r[rng.below(r.len())]).collect::<Vec<_>>())).collect();
    meds.sort_by(f64::total_cmp);
    (median(r), meds[(0.025 * meds.len() as f64) as usize], meds[(0.975 * meds.len() as f64) as usize - 1])
}

/// One kernel in one mode at one size.
struct Timed<'a> {
    label: String,
    kernel: ImageKernel<'a>,
    gpu: Vec<f64>,
    wall: Vec<f64>,
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let hof_path = PathBuf::from(arg(&args, "--hof").ok_or("--hof PATH is required")?);
    let out = PathBuf::from(arg(&args, "--out").ok_or("--out DIR is required")?);
    let rounds: usize = arg(&args, "--rounds").map_or(Ok(61), |v| v.parse().map_err(|e| format!("--rounds: {e}")))?;
    let sizes: Vec<u32> = arg(&args, "--sizes").unwrap_or_else(|| "256,512".into()).split(',').map(|s| s.parse().map_err(|e| format!("--sizes: {e}"))).collect::<Result<_, _>>()?;
    let seed: u64 = arg(&args, "--seed").map_or(Ok(1), |v| v.parse().map_err(|e| format!("--seed: {e}")))?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let shader = "volumepassage";
    let time_t = 5.0f32;
    let check_times = [2.0f32, 5.0, 8.0];

    let kingdom = TrueKingdom::load(&image_dir(shader))?;
    let contract = kingdom.relations().check().map_err(|d| format!("the kingdom check failed: {:?}", &d[..d.len().min(3)]))?;
    let tree = volumepassage(&kingdom).map_err(|g| format!("table gaps: {g:?}"))?;
    let layout = GeneLayout::from_program_size(&contract, kingdom.root, functions(&tree, &kingdom)).map_err(|e| e.to_string())?;
    let const_len = kingdom.const_lens();
    let hof = read_hof(&hof_path)?;
    let original_bound = kingdom.tree_work_bound(&tree)?;
    println!("# {} ST9 champion timing\thall of fame generation {}, {} entries\t{rounds} rounds\tsizes {sizes:?}\tseed {seed}", now(), hof.generation, hof.genes.len());

    // the kernels: shipped original, encoded original, each hall-of-fame tree
    let shipped = std::fs::read_to_string(format!("{}/tests/corpus/cuneus_visual/{shader}.wgsl", env!("CARGO_MANIFEST_DIR"))).map_err(|e| format!("original: {e}"))?;
    let encoded = kingdom.render(&tree)?;
    std::fs::write(out.join("encoded_original.wgsl"), &encoded).map_err(|e| e.to_string())?;
    let mut cands: Vec<(String, String, Tree, bool)> = Vec::new();
    for (i, g) in hof.genes.iter().enumerate() {
        if g.len() != layout.width {
            return Err(format!("hall-of-fame gene {} has {} words, the layout {}", i + 1, g.len(), layout.width));
        }
        let t = express(&contract, &layout, g, &const_len).map_err(|e| format!("our software has a bug: hall-of-fame gene {} does not express: {e}", i + 1))?.tree();
        let src = kingdom.render(&t)?;
        let b = kingdom.tree_work_bound(&t)?;
        std::fs::write(out.join(format!("rank{}.wgsl", i + 1)), &src).map_err(|e| e.to_string())?;
        println!("# rank {}\tHFF {:.6}\t{} nodes\twork bound {b} an invocation (original {original_bound})\t{}", i + 1, hof.fitness[i], t.level_order().len(), if src == encoded { "text identical to the encoded original" } else { "text differs from the encoded original" });
        cands.push((format!("rank{}", i + 1), src, t, b <= original_bound));
    }

    let dev = Device::new()?;
    println!("# device {}", dev.name);
    let mut times = std::fs::File::create(out.join("times.tsv")).map_err(|e| e.to_string())?;
    writeln!(times, "size\tround\torder\tkernel\tsubmits\tgpu_ms\twall_ms").map_err(|e| e.to_string())?;
    let mut summary = std::fs::File::create(out.join("summary.tsv")).map_err(|e| e.to_string())?;
    writeln!(summary, "size\tkernel\tsubmits\trounds\tgpu_median_ms\tgpu_p10_ms\tgpu_p90_ms\twall_median_ms").map_err(|e| e.to_string())?;
    let mut ratios = std::fs::File::create(out.join("ratios.tsv")).map_err(|e| e.to_string())?;
    writeln!(ratios, "size\tcandidate\treference\trounds\tratio_median\tci95_lo\tci95_hi\tfaster_rounds\tslower_rounds\tsign_p").map_err(|e| e.to_string())?;
    let mut pictures = std::fs::File::create(out.join("pictures.tsv")).map_err(|e| e.to_string())?;
    writeln!(pictures, "width\theight\ttime\tkernel\tsubmits\tdiffering_channels\tchannels\tmax_abs_error").map_err(|e| e.to_string())?;

    let mut shapes: Vec<(u32, u32)> = sizes.iter().map(|s| (*s, *s)).collect();
    shapes.push((200, 136));
    for (w, h) in shapes {
        let timed_size = sizes.contains(&w) && w == h;
        println!("# {} size {w}x{h}: compile", now());
        let bind = case_bindings(shader, time_t, w, h)?;
        let mut ks: Vec<Timed<'_>> = Vec::new();
        ks.push(Timed { label: "shipped.one".into(), kernel: ImageKernel::compile(&dev, shader, &shipped, "main", &bind, w, h, Fuel::Proven("the cuneus original shader as shipped"))?, gpu: vec![], wall: vec![] });
        ks.push(Timed { label: "encoded.tiled".into(), kernel: ImageKernel::compile(&dev, shader, &encoded, "main", &bind, w, h, kingdom.tree_fuel(&tree)?)?, gpu: vec![], wall: vec![] });
        ks.push(Timed { label: "encoded.one".into(), kernel: ImageKernel::compile(&dev, shader, &encoded, "main", &bind, w, h, Fuel::Proven("the encoded original: the shipped original's loops, 50 x 7, as one submit"))?, gpu: vec![], wall: vec![] });
        for (name, src, t, one_ok) in &cands {
            ks.push(Timed { label: format!("{name}.tiled"), kernel: ImageKernel::compile(&dev, shader, src, "main", &bind, w, h, kingdom.tree_fuel(t)?)?, gpu: vec![], wall: vec![] });
            if *one_ok {
                ks.push(Timed { label: format!("{name}.one"), kernel: ImageKernel::compile(&dev, shader, src, "main", &bind, w, h, Fuel::Proven("an ST9 hall-of-fame tree whose work bound is at most the encoded original's, as one submit like the shipped original"))?, gpu: vec![], wall: vec![] });
            } else {
                println!("# {name}: work bound above the original's, not timed as one submit");
            }
        }
        // the pictures at every time value, every kernel and mode, against the shipped original
        let mut frames: Vec<Picture> = Vec::new();
        for &tv in &check_times {
            if (w, h) == (200, 136) && tv != 5.0 {
                continue;
            }
            let mut want: Option<Picture> = None;
            for k in &ks {
                k.kernel.set_uniform(0, 0, &time_bytes(tv))?;
                let pic = k.kernel.run(0, 1)?.picture;
                let r = want.get_or_insert_with(|| pic.clone());
                let d = compare(&pic, r)?;
                writeln!(pictures, "{w}\t{h}\t{tv}\t{}\t{}\t{}\t{}\t{}", k.label, k.kernel.submits(), d.differing_channels, d.pixels * 4, d.max_abs).map_err(|e| e.to_string())?;
                if d.differing_channels != 0 {
                    println!("# PICTURE DIFFERS: {} at {w}x{h} t {tv}: {} channels", k.label, d.differing_channels);
                }
                if w == 256 && (k.label == "shipped.one" || k.label == "rank1.tiled") {
                    let mut p = pic.clone();
                    // the stamp font has digits, '.', '#', 'm', 's', 'x': #0 is the original, #1 the champion, then the time
                    stamp(&mut p, &format!("#{} {tv}s", u8::from(k.label != "shipped.one")), 2, 2, 2)?;
                    frames.push(p);
                }
            }
        }
        if frames.len() == 2 * check_times.len() {
            for (i, pair) in frames.chunks(2).enumerate() {
                write_png(&out.join(format!("frame_t{}_original_vs_champion.png", check_times[i])), &side_by_side(pair)?)?;
            }
        }
        if !timed_size {
            continue;
        }
        for k in &ks {
            k.kernel.set_uniform(0, 0, &time_bytes(time_t))?;
            k.kernel.passes(3, 0)?;
        }
        println!("# {} size {w}x{h}: {} kernels, {rounds} rounds", now(), ks.len());
        let mut rng = Rng(seed ^ (u64::from(w) << 20));
        for round in 0..rounds {
            let mut order: Vec<usize> = (0..ks.len()).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, rng.below(i + 1));
            }
            for (pos, &i) in order.iter().enumerate() {
                let (wall, gpu) = ks[i].kernel.passes(0, 1)?;
                writeln!(times, "{w}\t{round}\t{pos}\t{}\t{}\t{:.5}\t{:.5}", ks[i].label, ks[i].kernel.submits(), gpu[0], wall[0]).map_err(|e| e.to_string())?;
                ks[i].gpu.push(gpu[0]);
                ks[i].wall.push(wall[0]);
            }
        }
        for k in &ks {
            writeln!(summary, "{w}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}", k.label, k.kernel.submits(), k.gpu.len(), median(&k.gpu), quantile(&k.gpu, 0.1), quantile(&k.gpu, 0.9), median(&k.wall)).map_err(|e| e.to_string())?;
            println!("{w}\t{:<14}\t{:>2} submits\tGPU ms median {:.4} p10 {:.4} p90 {:.4}\twall median {:.4}", k.label, k.kernel.submits(), median(&k.gpu), quantile(&k.gpu, 0.1), quantile(&k.gpu, 0.9), median(&k.wall));
        }
        let find = |l: &str| ks.iter().find(|k| k.label == l);
        let mut pairs: Vec<(String, String)> = vec![("encoded.tiled".into(), "shipped.one".into()), ("encoded.one".into(), "shipped.one".into())];
        for (name, _, _, one_ok) in &cands {
            pairs.push((format!("{name}.tiled"), "encoded.tiled".into()));
            if *one_ok {
                pairs.push((format!("{name}.one"), "shipped.one".into()));
                pairs.push((format!("{name}.one"), "encoded.one".into()));
            }
        }
        for (c, r) in pairs {
            let (Some(kc), Some(kr)) = (find(&c), find(&r)) else { continue };
            let rs: Vec<f64> = kc.gpu.iter().zip(&kr.gpu).map(|(a, b)| a / b).collect();
            let (m, lo, hi) = ratio_stats(&rs, seed ^ 0xB007);
            let faster = rs.iter().filter(|x| **x < 1.0).count();
            let slower = rs.iter().filter(|x| **x > 1.0).count();
            let p = sign_test(faster, faster + slower);
            writeln!(ratios, "{w}\t{c}\t{r}\t{}\t{m:.4}\t{lo:.4}\t{hi:.4}\t{faster}\t{slower}\t{p:.3e}", rs.len()).map_err(|e| e.to_string())?;
            println!("{w}\t{c} / {r}\tratio median {m:.4} [95 % {lo:.4}, {hi:.4}]\tfaster in {faster}, slower in {slower} of {}\tsign p {p:.3e}", rs.len());
        }
    }
    println!("# {} done: {}", now(), out.display());
    Ok(())
}

fn main() {
    let child = std::thread::Builder::new().stack_size(1 << 29).spawn(real_main).expect("thread");
    match child.join().expect("the timing thread") {
        Ok(()) => {}
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    }
}
