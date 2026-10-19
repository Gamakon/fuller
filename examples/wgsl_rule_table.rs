//! THE DEVICE TABLE OF THE STRUCTURAL RULES: for each rule and each kernel
//! of a population it fires on, its child (one site a kernel, drawn with a
//! fixed seed) checked (`knob::check_text_under`), scored by the fitness the fit
//! runs (`FitnessContext::fitness`: correct on its 128 cases and the pair
//! sets), compared with the parent's output on the same Φ (the largest cell
//! difference over the parent's largest cell), and timed against the parent
//! in interleaved rounds at the fit's three timing shapes (parent, child,
//! parent, child, ...: the device drifts about 10 % over minutes, so only
//! interleaved pairs compare). One line a child in `<out>/children.tsv`,
//! one row a rule in the table printed at the end.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_rule_table -- --out <dir> [--rules a,b,..] [--rounds 6] <dir|file>...

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use fuller::wgsl::gram::{FitnessContext, Gram};
use fuller::wgsl::knob;
use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::oracle::device::Device;
use fuller::wgsl::rules::Rule;
use fuller::wgsl::safe_karva::render_tree_with;
use fuller::wgsl::safe_reader::read_safe;

/// The kernels of the arguments: `.wgsl` files, or every one in a directory.
fn kernels(args: &[String]) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for a in args {
        let p = PathBuf::from(a);
        if p.is_dir() {
            let mut fs: Vec<PathBuf> = std::fs::read_dir(&p).map_err(|e| format!("{a}: {e}"))?.flatten().map(|e| e.path()).filter(|f| f.extension().is_some_and(|x| x == "wgsl")).collect();
            fs.sort();
            out.extend(fs);
        } else {
            out.push(p);
        }
    }
    Ok(out)
}

/// The geometric mean of the three shapes' speed-ups.
fn gmean(s: &[f64; 3]) -> f64 {
    (s.iter().map(|x| x.ln()).sum::<f64>() / 3.0).exp()
}

/// The child's output against the parent's on one Φ (d 192, m 640, a fixed
/// seed): the largest cell difference over the parent's largest cell.
fn max_error(dev: &Device, parent: &str, child: &str) -> Result<f64, String> {
    let (d, m) = (192usize, 640usize);
    let mut state = 0x9e37_79b9u64;
    let phi: Vec<f32> = (0..d * m)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    let limit = 1u64 << 30;
    let p = Gram::compile(dev, parent)?.build(&phi, d, m, limit)?.h;
    let c = Gram::compile(dev, child)?.build(&phi, d, m, limit)?.h;
    let scale = p.iter().fold(0.0f64, |a, v| a.max(f64::from(v.abs())));
    let diff = p.iter().zip(&c).fold(0.0f64, |a, (x, y)| a.max(f64::from((x - y).abs())));
    Ok(if scale > 0.0 { diff / scale } else { diff })
}

/// One rule's row of the table.
#[derive(Default)]
struct Row {
    fires: usize,
    checked: usize,
    correct: usize,
    /// Children faster than the parent in all rounds but at most one.
    faster: usize,
    /// Each correct child's median speed ratio, child over parent.
    ratios: Vec<f64>,
    /// The largest error of a child's output against its parent's.
    worst: f64,
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |flag: &str| -> Result<Option<String>, String> {
        match args.iter().position(|a| a == flag) {
            Some(i) => {
                let v = args.get(i + 1).ok_or(format!("{flag} needs a value"))?.clone();
                args.drain(i..i + 2);
                Ok(Some(v))
            }
            None => Ok(None),
        }
    };
    let out = PathBuf::from(take("--out")?.ok_or("--out <dir> is required")?);
    let rounds: usize = take("--rounds")?.unwrap_or_else(|| "6".into()).parse().map_err(|e| format!("--rounds: {e}"))?;
    let rules: Vec<Rule> = match take("--rules")? {
        None => Rule::ALL.to_vec(),
        Some(list) => list.split(',').map(|n| Rule::by_name(n.trim()).ok_or_else(|| format!("no rule {n}"))).collect::<Result<_, _>>()?,
    };
    let files = kernels(&args)?;
    std::fs::create_dir_all(out.join("children")).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut tsv = std::fs::File::create(out.join("children.tsv")).map_err(|e| e.to_string())?;
    writeln!(tsv, "rule\tparent\tsite\tcheck\tcorrect\tparent_hff\tchild_hff\tmax_error\tfaster_rounds\tmedian_ratio\tratios").map_err(|e| e.to_string())?;
    let kingdom = WgslKingdom::load_safe();
    let dev = Device::new()?;
    let ctx = FitnessContext::new(&dev, Path::new("logs/WGSL_GRAM/pairs"))?;
    let started = std::time::Instant::now();
    let mut rows: BTreeMap<&str, Row> = BTreeMap::new();
    for (pi, f) in files.iter().enumerate() {
        let text = std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        let Ok(r) = read_safe(&text, &kingdom) else {
            println!("{}\tnot read", f.display());
            continue;
        };
        let parent_text = render_tree_with(&r.tree, &kingdom, &r.role)?.text;
        let parent = ctx.fitness(&dev, &parent_text);
        if !parent.feasible {
            println!("{}\tthe parent is not correct ({}); left out", f.display(), parent.detail);
            continue;
        }
        let name = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        for &rule in &rules {
            let sites = rule.sites(&r.tree, &kingdom);
            let row = rows.entry(rule.name()).or_default();
            if sites.is_empty() {
                continue;
            }
            row.fires += 1;
            let site = &sites[(pi * 7 + rule.name().len()) % sites.len()];
            let checked = rule.apply(&r.tree, site, &kingdom).map_err(|e| (knob::Stage::Render, e)).and_then(|t| knob::check_text_under(&t, &kingdom, &knob::Under::of(&r.role)));
            let (child_text, _) = match checked {
                Ok(x) => x,
                Err((stage, e)) => {
                    writeln!(tsv, "{}\t{name}\t{}\t{}: {}\t-\t{:.4}\t-\t-\t-\t-\t-", rule.name(), site.what, stage.name(), e.replace(['\t', '\n'], " "), parent.hff).map_err(|e| e.to_string())?;
                    continue;
                }
            };
            row.checked += 1;
            std::fs::write(out.join("children").join(format!("{name}__{}.wgsl", rule.name())), &child_text).map_err(|e| e.to_string())?;
            let child = ctx.fitness(&dev, &child_text);
            let err = max_error(&dev, &parent_text, &child_text).unwrap_or(f64::INFINITY);
            row.worst = row.worst.max(err);
            if !child.feasible {
                writeln!(tsv, "{}\t{name}\t{}\tpass\tno: {}\t{:.4}\t-\t{err:.3e}\t-\t-\t-", rule.name(), site.what, child.detail.replace(['\t', '\n'], " "), parent.hff).map_err(|e| e.to_string())?;
                continue;
            }
            row.correct += 1;
            // Interleaved rounds: parent, child, parent, child, ...
            let mut ratios = Vec::with_capacity(rounds);
            for _ in 0..rounds {
                let (p, _, _) = ctx.retime(&dev, &parent_text, &parent.columns)?;
                let (c, _, _) = ctx.retime(&dev, &child_text, &child.columns)?;
                ratios.push(gmean(&c) / gmean(&p));
            }
            let faster = ratios.iter().filter(|x| **x > 1.0).count();
            let mut sorted = ratios.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let median = (sorted[(rounds - 1) / 2] + sorted[rounds / 2]) / 2.0;
            if faster + 1 >= rounds {
                row.faster += 1;
            }
            row.ratios.push(median);
            writeln!(tsv, "{}\t{name}\t{}\tpass\tyes\t{:.4}\t{:.4}\t{err:.3e}\t{faster}\t{median:.3}\t{}", rule.name(), site.what, parent.hff, child.hff, ratios.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(",")).map_err(|e| e.to_string())?;
            tsv.flush().map_err(|e| e.to_string())?;
            println!("{:.0} s\t{}\t{name}\tmedian {median:.3}, faster in {faster} of {rounds}, max error {err:.2e}", started.elapsed().as_secs_f64(), rule.name());
        }
    }
    println!("\n| rule | fires on | children checked | correct | faster in {} or {rounds} of {rounds} | median ratio (child / parent speed) | max error vs parent |", rounds - 1);
    println!("|---|---|---|---|---|---|---|");
    for (rule, row) in &rows {
        let mut r = row.ratios.clone();
        r.sort_by(|a, b| a.total_cmp(b));
        let median = if r.is_empty() { "-".to_string() } else { format!("{:.3}", (r[(r.len() - 1) / 2] + r[r.len() / 2]) / 2.0) };
        let err = if row.checked == 0 { "-".to_string() } else { format!("{:.2e}", row.worst) };
        println!("| {rule} | {} | {} | {} | {} | {median} | {err} |", row.fires, row.checked, row.correct, row.faster);
    }
    println!("{:.0} s", started.elapsed().as_secs_f64());
    Ok(())
}
