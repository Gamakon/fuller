//! ST9: what each extra of the row templates costs. The encoded volumepassage draws the shipped shader's picture; this
//! times kernel texts against the shipped original at 64 x 64, t 5, one submit, with the run's timing path
//! (`image::timed_pairs`: one command buffer a pair, a warm-up of the reference, the order alternating).
//!
//! Two steps, because the texts of "before" come from the tables as they were:
//!
//!   cargo run --release --features gpu,wgsl --example st9_template_cost -- dump DIR
//!       renders the encoded volumepassage with the current table into DIR/encoded.wgsl, and beside it the probe texts
//!       of the encoded one with ONE extra taken out by a text edit (a measurement probe, never a kernel of the kingdom):
//!       nodecl (only the declarations the body names), bare (every lit_* call replaced by its literal), notile (the
//!       builtins of the dispatch, no DispatchTile uniform), noclamp (the clamps of the ownership offsets removed), and
//!       all four together.
//!   cargo run --release --features gpu,wgsl --example st9_template_cost -- time PAIRS ROUNDS LABEL=FILE ...
//!       times shipped against shipped and the first FILE against itself (the two A/As), every FILE against the shipped
//!       original, and every later FILE against the first one directly (one command buffer a pair, so the clock state
//!       cancels between them), PAIRS pairs a comparison in each of ROUNDS rounds, the order of the comparisons
//!       reversed every other round; prints one line a comparison: the median ratio over all pairs, p10, p90, and the
//!       range of the round medians.
use fuller::wgsl::cuneus_originals::{case_bindings, image_dir, volumepassage};
use fuller::wgsl::image::{timed_pairs, ImageKernel};
use fuller::wgsl::oracle::device::{Device, Fuel};
use fuller::wgsl::true_kingdom::TrueKingdom;
use regex::Regex;

fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * q).round() as usize]
}

fn utc() -> String {
    std::process::Command::new("date").args(["-u", "+%H:%M:%SZ"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

/// Only the function-scope `var` declarations whose name the rest of the text uses.
fn nodecl(src: &str) -> String {
    let decl = Regex::new(r"^var (\w+):").expect("regex");
    src.lines()
        .filter(|l| match decl.captures(l) {
            None => true,
            Some(c) => {
                let word = Regex::new(&format!(r"\b{}\b", regex::escape(&c[1]))).expect("regex");
                src.lines().filter(|o| !decl.is_match(o)).any(|o| word.is_match(o))
            }
        })
        .map(|l| format!("{l}\n"))
        .collect()
}

/// Every `lit_*(x)` call replaced by `x` (the helpers' own definitions are kept).
fn bare(src: &str) -> String {
    Regex::new(r"(fn )?lit_(?:u32|f32|i32|bool|f16)\(([^()]*)\)")
        .expect("regex")
        .replace_all(src, |c: &regex::Captures| if c.get(1).is_some() { c[0].to_string() } else { c[2].to_string() })
        .into_owned()
}

/// The dispatch's own builtins: no DispatchTile uniform in the entry point.
fn notile(src: &str) -> Result<String, String> {
    let mut s = src.replace("@builtin(global_invocation_id) gid_tile: vec3<u32>", "@builtin(global_invocation_id) gid: vec3<u32>");
    s = s.replace("@builtin(workgroup_id) wid_tile: vec3<u32>", "@builtin(workgroup_id) wid: vec3<u32>");
    s = s.replace("@builtin(num_workgroups) nwg_tile: vec3<u32>", "@builtin(num_workgroups) nwg: vec3<u32>");
    let lets = [
        "let wid: vec3<u32> = select(wid_tile, wid_tile + dispatch_tile.origin.xyz, DISPATCH_TILED);\n",
        "let gid: vec3<u32> = select(gid_tile, gid_tile + dispatch_tile.origin.xyz * vec3<u32>(WG_X, WG_Y, 1u), DISPATCH_TILED);\n",
        "let nwg: vec3<u32> = select(nwg_tile, dispatch_tile.grid.xyz, DISPATCH_TILED);\n",
    ];
    for l in lets {
        if !s.contains(l) {
            return Err(format!("our software has a bug: the tiled entry has no line {l:?}"));
        }
        s = s.replace(l, "");
    }
    Ok(s)
}

/// The ownership offsets' clamps removed: `min(X, KNOB_BROW - 1u)` and `min(X, KNOB_BCOL - 1u)` become `X`.
fn noclamp(src: &str) -> String {
    Regex::new(r"min\(([^(),]*), KNOB_B(?:ROW|COL) - 1u\)").expect("regex").replace_all(src, "$1").into_owned()
}

fn dump(dir: &str) -> Result<(), String> {
    let k = TrueKingdom::load(&image_dir("volumepassage"))?;
    let tree = volumepassage(&k).map_err(|g| format!("table gaps: {g:?}"))?;
    let src = k.render(&tree)?;
    let out = std::path::Path::new(dir);
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let all = noclamp(&notile(&bare(&nodecl(&src)))?);
    for (name, text) in [("encoded", src.clone()), ("nodecl", nodecl(&src)), ("bare", bare(&src)), ("notile", notile(&src)?), ("noclamp", noclamp(&src)), ("all", all)] {
        std::fs::write(out.join(format!("{name}.wgsl")), &text).map_err(|e| e.to_string())?;
        println!("{name}\t{} bytes\t{} lines", text.len(), text.lines().count());
    }
    Ok(())
}

fn time(pairs: usize, rounds: usize, files: &[(String, String)]) -> Result<(), String> {
    let original = std::fs::read_to_string(format!("{}/tests/corpus/cuneus_visual/volumepassage.wgsl", env!("CARGO_MANIFEST_DIR"))).map_err(|e| e.to_string())?;
    let dev = Device::new()?;
    let size = 64u32;
    let bindings = case_bindings("volumepassage", 5.0, size, size)?;
    let proven = || Fuel::Proven("the shipped volumepassage's loops (50 x 7): the shipped original, the encoded original or a text probe of it");
    let compile = |name: &str, text: &str| ImageKernel::compile(&dev, name, text, "main", &bindings, size, size, proven());
    let mut kernels: Vec<(String, ImageKernel<'_>)> = vec![("shipped".into(), compile("shipped", &original)?), ("shipped(2)".into(), compile("shipped again", &original)?)];
    for (l, p) in files {
        let t = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
        kernels.push((l.clone(), compile(l, &t)?));
        if kernels.len() == 3 {
            kernels.push((format!("{l}(2)"), compile(l, &t)?));
        }
    }
    let reference = kernels[0].1.run(0, 1)?.picture;
    for (l, kk) in &kernels[2..] {
        let d = fuller::wgsl::image::compare(&kk.run(0, 1)?.picture, &reference)?;
        println!("# {l}: {} differing channels against the shipped original", d.differing_channels);
    }
    // the comparisons, as (candidate, reference) indices: the two A/As, every file against the shipped original, and
    // every later file against the first one (a direct A/B in one command buffer a pair, so no clock drift between them)
    let mut comps: Vec<(usize, usize)> = vec![(1, 0), (3, 2)];
    let files_at: Vec<usize> = std::iter::once(2).chain(4..kernels.len()).collect();
    comps.extend(files_at.iter().map(|i| (*i, 0)));
    comps.extend(files_at[1..].iter().map(|i| (*i, 2)));
    let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); comps.len()];
    let mut round_medians: Vec<Vec<f64>> = vec![Vec::new(); comps.len()];
    println!("# {} {size}x{size} t 5, one submit, {rounds} rounds of {pairs} pairs a comparison, the order of the comparisons reversed every other round", utc());
    for r in 0..rounds {
        let order: Vec<usize> = if r % 2 == 0 { (0..comps.len()).collect() } else { (0..comps.len()).rev().collect() };
        for c in order {
            let (ci, ri) = comps[c];
            let pt = timed_pairs(&kernels[ci].1, &kernels[ri].1, 1, pairs)?;
            let v: Vec<f64> = pt.times.iter().map(|t| t.0 / t.1).collect();
            round_medians[c].push(quantile(&v, 0.5));
            ratios[c].extend(v);
        }
    }
    println!("comparison\tmedian ratio (all pairs)\tp10\tp90\tround medians min\tmax\tpairs");
    for (c, (ci, ri)) in comps.iter().enumerate() {
        let v = &ratios[c];
        let m = &round_medians[c];
        println!("{} / {}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}", kernels[*ci].0, kernels[*ri].0, quantile(v, 0.5), quantile(v, 0.1), quantile(v, 0.9), quantile(m, 0.0), quantile(m, 1.0), v.len());
    }
    println!("# {}", utc());
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("dump") => dump(args.get(1).ok_or("dump DIR")?),
        Some("time") => {
            let pairs: usize = args.get(1).ok_or("time PAIRS ROUNDS LABEL=FILE ...")?.parse().map_err(|e| format!("PAIRS: {e}"))?;
            let rounds: usize = args.get(2).ok_or("time PAIRS ROUNDS LABEL=FILE ...")?.parse().map_err(|e| format!("ROUNDS: {e}"))?;
            let files: Vec<(String, String)> = args[3..].iter().map(|a| a.split_once('=').map(|(l, p)| (l.to_string(), p.to_string())).ok_or_else(|| format!("{a}: want LABEL=FILE"))).collect::<Result<_, _>>()?;
            if files.is_empty() {
                return Err("time PAIRS LABEL=FILE ...: no file".into());
            }
            time(pairs, rounds, &files)
        }
        _ => Err("st9_template_cost dump DIR | time PAIRS ROUNDS LABEL=FILE ...".into()),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
