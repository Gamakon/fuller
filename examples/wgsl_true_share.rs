//! ST7 measurements (docs/PLAN_true_gep_wgsl.md): share-equivalence on the true WGSL kingdom, host side, naga only.
//! No device is opened.
//!
//!   cargo run --release --features wgsl --example wgsl_true_share -- \
//!       [--dir kingdoms/wgsl_true] [--genes 1000] [--seed 1] [--size 8] [--seeds FILE] [--fold 100000] [--batch 1000]
//!
//! For each gene (random K = 1 genes from the host typed initialiser, or the rows of `--seeds FILE`, one gene of
//! space-separated words a line): the expressed tree, its kernel text with sharing off and on (`fuller::wgsl::
//! true_share`), the bytes of each, naga parse + validate time of each, the lets, the operations each text spells,
//! the tree's nodes and distinct subtrees; both texts run on the CPU executor over the Gram unit test's inputs (d 16,
//! m 4, one workgroup) and their outputs compared bit for bit. Then the population is folded into ADFs (stable ids,
//! per-type write-back, unfold). `--fold N` runs fold -> write-back -> unfold on N further random genes, in
//! populations of `--batch`, asserting every folded gene renders its original text.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::Instant;

use fuller::typed_gep::{express, GeneLayout, Tree};
use fuller::wgsl::cpu::{run_text, Val};
use fuller::wgsl::true_kingdom::{default_dir, random_gene, Rng, Share as Draw, TrueKingdom};
use fuller::wgsl::true_share::{AdfTable, Share};

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn num<T: std::str::FromStr>(args: &[String], k: &str, d: T) -> Result<T, String> {
    match arg(args, k) {
        Some(v) => v.parse().map_err(|_| format!("{k} {v}: not a number")),
        None => Ok(d),
    }
}

fn naga_ms(src: &str) -> (f64, Result<(), String>) {
    let t = Instant::now();
    let r = fuller::wgsl::naga_shape::parse_unscanned(src).and_then(|m| fuller::wgsl::target::Target::current().validate(&m, src).map(|_| ()));
    (t.elapsed().as_secs_f64() * 1e3, r)
}

#[derive(Default)]
struct Sum {
    n: usize,
    bytes_off: usize,
    bytes_on: usize,
    naga_off: f64,
    naga_on: f64,
    naga_fail_off: usize,
    naga_fail_on: usize,
    with_lets: usize,
    lets: usize,
    ops_off: u64,
    ops_on: u64,
    nodes: usize,
    distinct: usize,
    share_ms: f64,
    cpu_ran: usize,
    cpu_failed: BTreeMap<String, usize>,
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = arg(&args, "--dir").map(PathBuf::from).unwrap_or_else(default_dir);
    let n_genes: usize = num(&args, "--genes", 1000)?;
    let seed: u64 = num(&args, "--seed", 1)?;
    let size: usize = num(&args, "--size", 8)?;
    let fold_n: usize = num(&args, "--fold", 0)?;
    let batch: usize = num(&args, "--batch", 1000)?;
    let t0 = Instant::now();
    let k = TrueKingdom::load(&dir)?;
    let share = Share::load(&k)?;
    let c = k.relations().check().map_err(|d| d.join("; "))?;
    let layout = GeneLayout::from_program_size(&c, k.root, size).map_err(|e| e.to_string())?;
    let cl = k.const_lens();
    println!("# table {} ({} rows, {} context-free), N = {size}, gene width {} words, K = {}", dir.display(), k.rows.len(), share.context_free.iter().filter(|b| **b).count(), layout.width, k.fuel_budget);
    let mut rng = Rng(seed);
    let genes: Vec<Vec<u32>> = match arg(&args, "--seeds") {
        Some(f) => {
            let text = std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?;
            text.lines().filter(|l| !l.trim().is_empty()).map(|l| l.split_whitespace().map(|w| w.parse::<u32>().map_err(|e| format!("{f}: {e}"))).collect()).collect::<Result<_, _>>()?
        }
        None => (0..n_genes).map(|i| random_gene(&c, &layout, &cl, &mut rng, if i % 2 == 0 { Draw::Half } else { Draw::Proportional }, (i / 2) % 2 == 1)).collect(),
    };
    let mut trees: Vec<Tree> = Vec::with_capacity(genes.len());
    for (i, g) in genes.iter().enumerate() {
        trees.push(express(&c, &layout, g, &cl).map_err(|e| format!("gene {i} does not express on this table: {e}"))?.tree());
    }
    let (d, m) = (16usize, 4usize);
    let phi = fuller::wgsl::gram::phi(d, m, 3);
    let inputs = || {
        HashMap::from([
            ("phi".to_string(), Val::V(phi.iter().map(|v| Val::F(*v)).collect())),
            ("h".to_string(), Val::V(vec![Val::F(0.0); d * d])),
            ("cfg".to_string(), Val::V(vec![Val::U(d as u32), Val::U(m as u32), Val::U(d as u32), Val::U(0)])),
        ])
    };
    let mut s = Sum::default();
    for (i, t) in trees.iter().enumerate() {
        let ts = Instant::now();
        let sh = share.share(&k, t)?;
        s.share_ms += ts.elapsed().as_secs_f64() * 1e3;
        share.check_types(&k, &sh.tree, k.root)?;
        if share.unshare(&k, &sh.tree)? != *t {
            return Err(format!("our software has a bug: gene {i}: unshare is not the expressed tree"));
        }
        let off = k.render(t)?;
        let on = share.render(&k, &sh.tree)?;
        // Alternate which text naga sees first, so a warm cache favours neither.
        let ((a, ra), (b, rb)) = if i % 2 == 0 {
            let x = naga_ms(&off);
            (x, naga_ms(&on))
        } else {
            let y = naga_ms(&on);
            (naga_ms(&off), y)
        };
        if ra.is_ok() != rb.is_ok() {
            return Err(format!("our software has a bug: gene {i}: naga off {ra:?}, on {rb:?}"));
        }
        s.n += 1;
        s.bytes_off += off.len();
        s.bytes_on += on.len();
        s.naga_off += a;
        s.naga_on += b;
        s.naga_fail_off += usize::from(ra.is_err());
        s.naga_fail_on += usize::from(rb.is_err());
        s.lets += sh.stats.lets;
        s.with_lets += usize::from(sh.stats.lets > 0);
        s.ops_off += sh.stats.spelled_ops;
        s.ops_on += sh.stats.spelled_ops_shared;
        s.nodes += sh.stats.nodes;
        s.distinct += sh.stats.distinct;
        match (run_text(&off, inputs(), [1, 1, 1], 20_000_000), run_text(&on, inputs(), [1, 1, 1], 20_000_000)) {
            (Ok(x), Ok(y)) => {
                let (mut wx, mut wy) = (Vec::new(), Vec::new());
                x.buffers["h"].words(&mut wx);
                y.buffers["h"].words(&mut wy);
                let key = |w: &fuller::wgsl::cpu::Write| (w.binding.clone(), w.path.clone(), w.gid);
                if wx != wy || x.writes.iter().map(key).collect::<Vec<_>>() != y.writes.iter().map(key).collect::<Vec<_>>() {
                    return Err(format!("our software has a bug: gene {i}: the outputs differ with sharing on"));
                }
                s.cpu_ran += 1;
            }
            (Err(x), Err(y)) if sh.stats.lets == 0 || x == y => *s.cpu_failed.entry(x.lines().next().unwrap_or_default().chars().take(70).collect()).or_default() += 1,
            (x, y) => return Err(format!("our software has a bug: gene {i}: off {:?}, on {:?}", x.err(), y.err())),
        }
    }
    let nf = s.n as f64;
    println!("kernels\t{}", s.n);
    println!("kernel text bytes, mean\toff {:.0}\ton {:.0}\t{:+.2} %", s.bytes_off as f64 / nf, s.bytes_on as f64 / nf, 100.0 * (s.bytes_on as f64 / s.bytes_off as f64 - 1.0));
    println!("naga parse+validate ms, mean\toff {:.3}\ton {:.3}\t{:+.2} %", s.naga_off / nf, s.naga_on / nf, 100.0 * (s.naga_on / s.naga_off - 1.0));
    println!("naga failures\toff {}\ton {}", s.naga_fail_off, s.naga_fail_on);
    println!("kernels with a let\t{} ({:.1} %)\tlets {}\tlets per kernel {:.2}", s.with_lets, 100.0 * s.with_lets as f64 / nf, s.lets, s.lets as f64 / nf);
    println!("operations spelled, mean\toff {:.1}\ton {:.1}\tshare rate {:.2} %", s.ops_off as f64 / nf, s.ops_on as f64 / nf, 100.0 * (1.0 - s.ops_on as f64 / s.ops_off as f64));
    println!("tree nodes, mean\t{:.1}\tdistinct subtrees, mean {:.1}\t({:.2} % of nodes repeat within the kernel)", s.nodes as f64 / nf, s.distinct as f64 / nf, 100.0 * (1.0 - s.distinct as f64 / s.nodes as f64));
    println!("share pass ms, mean\t{:.3}", s.share_ms / nf);
    println!("CPU executor\t{} of {} ran, outputs bit-identical with sharing on and off; failed alike: {:?}", s.cpu_ran, s.n, s.cpu_failed);
    let tf = Instant::now();
    let mut table = AdfTable::default();
    let f = table.fold(&c, &layout, &cl, &genes, &trees)?;
    for (i, (g, t)) in f.genes.iter().zip(&trees).enumerate() {
        // The folded gene's text is the original's (fold already checked the tree; this checks the text the device gets).
        let back = fuller::typed_chromosome::express_chromosome(&f.epoch, &f.adfs, &fuller::typed_chromosome::ChromosomeLayout { normal: fuller::typed_gep::Layout::new(0, vec![layout.clone()]), mains: Vec::new() }, &fuller::typed_chromosome::Chromosome { chromosome_id: i as u64, genes: vec![g.clone()], mains: Vec::new() }, &cl).map_err(|e| e.to_string())?.inlined(&f.epoch, &f.adfs);
        if k.render(&back[0].1)? != k.render(t)? {
            return Err(format!("our software has a bug: individual {i}: the folded gene renders other text"));
        }
    }
    let p = &f.stats;
    println!("population\t{} individuals, {} nodes, {} distinct subtrees ({:.2} % of nodes)", p.individuals, p.nodes, p.distinct, 100.0 * p.distinct as f64 / p.nodes as f64);
    println!("population share rate\t{:.2} % of {} function-holding subtree occurrences repeat in the population", 100.0 * p.repeated as f64 / p.occurrences as f64, p.occurrences);
    let levels = f.used.iter().map(|u| u.2).max().unwrap_or(0);
    println!("ADFs\t{} (levels 0..={levels}), {} references written back, stored nodes {} vs {} ({:+.2} %), fold {:.0} ms", p.adfs, f.references, p.stored_nodes, p.nodes, 100.0 * (p.stored_nodes as f64 / p.nodes as f64 - 1.0), tf.elapsed().as_secs_f64() * 1e3);
    if fold_n > 0 {
        let tb = Instant::now();
        let (mut done, mut refs, mut adfs) = (0usize, 0usize, 0usize);
        let mut big = AdfTable::default();
        let mut rng = Rng(seed ^ 0x5354_375f_464f_4c44);
        while done < fold_n {
            let n = batch.min(fold_n - done);
            let g: Vec<Vec<u32>> = (0..n).map(|i| random_gene(&c, &layout, &cl, &mut rng, if i % 2 == 0 { Draw::Half } else { Draw::Proportional }, (i / 2) % 2 == 1)).collect();
            let t: Vec<Tree> = g.iter().map(|w| express(&c, &layout, w, &cl).map(|e| e.tree()).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
            let f = big.fold(&c, &layout, &cl, &g, &t)?;
            refs += f.references;
            adfs += f.stats.adfs;
            done += n;
        }
        println!("fold -> write-back -> unfold\t{done} random genes in populations of {batch}: every folded gene expresses its tree; {adfs} ADFs used, {} ids issued, {refs} references, {:.1} s", big.next, tb.elapsed().as_secs_f64());
    }
    println!("# wall {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}
