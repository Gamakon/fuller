//! THE KNOB MUTATION, MEASURED (`docs/PLAN_constrained_mutation.md` task 0).
//!
//! `--survey`: for every kernel text given (files or directories of
//! `.wgsl`), the reader's parameters: whether the parameterised tree renders
//! the same text as the literal tree (it must), and the references it holds
//! by knob and form.
//!
//! `--mutate N --out <dir>`: N knob mutations of every kernel
//! (`knob::mutate`), each child through the layered check (render, validate
//! for the adapter, read back by the reader's ownership proof); every
//! child that passes is written as `<out>/children/<parent>__<i>.wgsl`, every
//! parent as `<out>/parents/<parent>.wgsl`, one line a child in
//! `<out>/children.tsv` (parent, child, knob, old, new, check result), one
//! line a failed check in `<out>/errors.tsv` (origin "knob mutation"); the
//! table of shares is printed.
//!
//!   cargo run --release --features wgsl --example wgsl_knob -- --survey <dir|file>...
//!   cargo run --release --features wgsl --example wgsl_knob -- --mutate 10 --seed 1 --out <dir> <dir|file>...

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fuller::wgsl::{knob, rules};
use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::params::parse_ref;
use fuller::wgsl::render::Tree;
use fuller::wgsl::safe_karva::render_tree;
use fuller::wgsl::safe_reader::{read_safe, read_safe_literal};
use fuller::wgsl::target::Target;

fn kernels(paths: &[String]) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for p in paths {
        let p = Path::new(p);
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p).map_err(|e| format!("{}: {e}", p.display()))?.flatten().map(|e| e.path()).filter(|f| f.extension().is_some_and(|x| x == "wgsl")).collect();
            v.sort();
            out.extend(v);
        } else {
            out.push(p.to_path_buf());
        }
    }
    Ok(out)
}

fn refs(t: &Tree, out: &mut BTreeMap<String, usize>) {
    if parse_ref(&t.name).is_some() {
        *out.entry(format!("{}:{}", t.name.trim_start_matches("literal.p_"), t.konst)).or_default() += 1;
    }
    t.kids.iter().for_each(|k| refs(k, out));
}

fn survey(files: &[PathBuf], kingdom: &WgslKingdom) -> Result<(), String> {
    let (mut read, mut same, mut with_t, mut refused) = (0usize, 0usize, 0usize, 0usize);
    let mut all: BTreeMap<String, usize> = BTreeMap::new();
    for f in files {
        let text = std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        let (lit, par) = match (read_safe_literal(&text, kingdom), read_safe(&text, kingdom)) {
            (Ok(l), Ok(p)) => (l, p),
            (Err(_), Err(_)) => {
                refused += 1;
                continue;
            }
            (l, p) => return Err(format!("{}: our software has a bug: the literal read {} and the parameterised read {}", f.display(), if l.is_ok() { "succeeded" } else { "failed" }, if p.is_ok() { "succeeded" } else { "failed" })),
        };
        read += 1;
        let (a, b) = (render_tree(&lit.tree, kingdom)?.text, render_tree(&par.tree, kingdom)?.text);
        if a == b {
            same += 1;
        } else {
            println!("DIFFERENT {}", f.display());
        }
        let mut r = BTreeMap::new();
        refs(&par.tree, &mut r);
        if r.keys().any(|k| k.starts_with("ktile")) {
            with_t += 1;
        }
        let line: Vec<String> = r.iter().map(|(k, n)| format!("{k}x{n}")).collect();
        println!("{}\t{}", f.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), line.join(" "));
        for (k, n) in r {
            *all.entry(k).or_default() += n;
        }
    }
    println!("\nkernels {}: read {read}, not read {refused}; parameterised text identical {same} of {read}; declaring a tile depth {with_t}", files.len());
    for (k, n) in &all {
        println!("  {k}\t{n}");
    }
    Ok(())
}

/// A 95 % Wilson interval of k of n, as percentages.
fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let (p, n, z) = (k as f64 / n as f64, n as f64, 1.96f64);
    let c = (p + z * z / (2.0 * n)) / (1.0 + z * z / n);
    let h = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt() / (1.0 + z * z / n);
    (100.0 * (c - h).max(0.0), 100.0 * (c + h).min(1.0))
}

fn mutate_all(files: &[PathBuf], n: usize, seed: u64, out: &Path, kingdom: &WgslKingdom) -> Result<(), String> {
    use std::io::Write as _;
    let target = Target::current();
    for d in ["parents", "children"] {
        std::fs::create_dir_all(out.join(d)).map_err(|e| format!("{}: {e}", out.join(d).display()))?;
    }
    let mut tsv = std::fs::File::create(out.join("children.tsv")).map_err(|e| e.to_string())?;
    writeln!(tsv, "parent\tchild\tknob\told\tnew\tcheck").map_err(|e| e.to_string())?;
    let mut errors = std::fs::File::create(out.join("errors.tsv")).map_err(|e| e.to_string())?;
    writeln!(errors, "origin\tparent\tknob\told\tnew\tstage\terror").map_err(|e| e.to_string())?;
    // Per knob: attempted, passed; and per stage, failures.
    let mut by_knob: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut by_stage: BTreeMap<&str, usize> = BTreeMap::new();
    let (mut parents, mut no_knob, mut unread) = (0usize, 0usize, 0usize);
    let mut seen: std::collections::BTreeSet<String> = Default::default();
    let mut distinct = 0usize;
    let started = std::time::Instant::now();
    for (pi, f) in files.iter().enumerate() {
        let text = std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        let Ok(r) = read_safe(&text, kingdom) else {
            unread += 1;
            continue;
        };
        parents += 1;
        let id = format!("p{pi:03}");
        let parent_text = render_tree(&r.tree, kingdom)?.text;
        std::fs::write(out.join("parents").join(format!("{id}.wgsl")), &parent_text).map_err(|e| e.to_string())?;
        let mut rng = seed ^ (pi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for i in 0..n {
            let c = match knob::mutate(&r.tree, kingdom, &target, &mut rng) {
                Ok(c) => c,
                Err(e) => {
                    no_knob += 1;
                    writeln!(errors, "knob mutation\t{}\t-\t-\t-\tchoose\t{e}", f.display()).map_err(|e| e.to_string())?;
                    break;
                }
            };
            let entry = by_knob.entry(knob_class(&c.site)).or_default();
            entry.0 += 1;
            let child = format!("{id}__{i:02}");
            match knob::check_read_back(&c.tree, kingdom) {
                Ok(t) => {
                    entry.1 += 1;
                    if seen.insert(t.clone()) {
                        distinct += 1;
                    }
                    std::fs::write(out.join("children").join(format!("{child}.wgsl")), &t).map_err(|e| e.to_string())?;
                    writeln!(tsv, "{id}\t{child}\t{}\t{}\t{}\tpass", c.site.name(), c.old, c.new).map_err(|e| e.to_string())?;
                }
                Err((stage, e)) => {
                    *by_stage.entry(stage.name()).or_default() += 1;
                    writeln!(tsv, "{id}\t{child}\t{}\t{}\t{}\t{}", c.site.name(), c.old, c.new, stage.name()).map_err(|e| e.to_string())?;
                    writeln!(errors, "knob mutation\t{}\t{}\t{}\t{}\t{}\t{}", f.display(), c.site.name(), c.old, c.new, stage.name(), e.replace(['\t', '\n'], " ")).map_err(|e| e.to_string())?;
                }
            }
        }
    }
    let (att, pass): (usize, usize) = by_knob.values().fold((0, 0), |(a, p), (x, y)| (a + x, p + y));
    println!("kernels {}: read {parents}, not read {unread}; {n} knob mutations a kernel, seed {seed}; {:.1} s", files.len(), started.elapsed().as_secs_f64());
    println!("parents with no knob to change: {no_knob}");
    println!("| knob | children | legal, render, validate, read back | share | 95 % interval |");
    println!("|---|---|---|---|---|");
    for (k, (a, p)) in by_knob.iter().chain(std::iter::once((&"all".to_string(), &(att, pass)))) {
        let (lo, hi) = wilson(*p, *a);
        println!("| {k} | {a} | {p} | {:.1} % | {lo:.1} to {hi:.1} % |", 100.0 * *p as f64 / (*a).max(1) as f64);
    }
    println!("distinct child texts: {distinct}; failures by checker stage: {by_stage:?}");
    Ok(())
}

/// RULE CHILDREN (task 2): for every kernel, up to `n` of the sites where
/// `rule` fires (all of them when fewer, else a seeded sample), each child
/// through the layered check; files and table as `mutate_all` writes them.
fn rule_all(files: &[PathBuf], rule: rules::Rule, n: usize, seed: u64, out: &Path, kingdom: &WgslKingdom) -> Result<(), String> {
    use std::io::Write as _;
    for d in ["parents", "children"] {
        std::fs::create_dir_all(out.join(d)).map_err(|e| format!("{}: {e}", out.join(d).display()))?;
    }
    let mut tsv = std::fs::File::create(out.join("children.tsv")).map_err(|e| e.to_string())?;
    writeln!(tsv, "parent\tchild\tknob\told\tnew\tcheck").map_err(|e| e.to_string())?;
    let mut errors = std::fs::File::create(out.join("errors.tsv")).map_err(|e| e.to_string())?;
    writeln!(errors, "origin\tparent\trule\tsite\tstage\terror").map_err(|e| e.to_string())?;
    let (mut parents, mut unread, mut lit, mut made, mut passed) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut by_stage: BTreeMap<&str, usize> = BTreeMap::new();
    let mut not_fired: BTreeMap<String, usize> = BTreeMap::new();
    let started = std::time::Instant::now();
    for (pi, f) in files.iter().enumerate() {
        let text = std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        let Ok(r) = read_safe(&text, kingdom) else {
            unread += 1;
            continue;
        };
        parents += 1;
        let id = format!("p{pi:03}");
        std::fs::write(out.join("parents").join(format!("{id}.wgsl")), render_tree(&r.tree, kingdom)?.text).map_err(|e| e.to_string())?;
        let mut sites = rule.sites(&r.tree, kingdom);
        if sites.is_empty() {
            let why = rule.why_not(&r.tree, kingdom).unwrap_or_default();
            *not_fired.entry(why.chars().take(300).collect()).or_default() += 1;
            continue;
        }
        lit += 1;
        let mut rng = seed ^ (pi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        while sites.len() > n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            sites.remove(((rng >> 33) % sites.len() as u64) as usize);
        }
        for (i, site) in sites.iter().enumerate() {
            made += 1;
            let child = format!("{id}__{i:02}");
            let outcome = rule.apply(&r.tree, site, kingdom).map_err(|e| (knob::Stage::Render, e)).and_then(|t| knob::check_text(&t, kingdom));
            match outcome {
                Ok((t, _)) => {
                    passed += 1;
                    std::fs::write(out.join("children").join(format!("{child}.wgsl")), &t).map_err(|e| e.to_string())?;
                    writeln!(tsv, "{id}\t{child}\t{}\t-\t{}\tpass", rule.name(), site.what).map_err(|e| e.to_string())?;
                }
                Err((stage, e)) => {
                    *by_stage.entry(stage.name()).or_default() += 1;
                    writeln!(tsv, "{id}\t{child}\t{}\t-\t{}\t{}", rule.name(), site.what, stage.name()).map_err(|e| e.to_string())?;
                    writeln!(errors, "rule mutation\t{}\t{}\t{}\t{}\t{}", f.display(), rule.name(), site.what, stage.name(), e.replace(['\t', '\n'], " ")).map_err(|e| e.to_string())?;
                }
            }
        }
    }
    let (lo, hi) = wilson(passed, made);
    println!("kernels {}: read {parents}, not read {unread}; rule {} fires in {lit}; up to {n} sites a kernel, seed {seed}; {:.1} s", files.len(), rule.name(), started.elapsed().as_secs_f64());
    println!("| rule | children | legal, render, validate, read back | share | 95 % interval |\n|---|---|---|---|---|\n| {} | {made} | {passed} | {:.1} % | {lo:.1} to {hi:.1} % |", rule.name(), 100.0 * passed as f64 / made.max(1) as f64);
    println!("failures by checker stage: {by_stage:?}");
    println!("kernels where the rule does not fire, by reason:");
    let mut why: Vec<(&String, &usize)> = not_fired.iter().collect();
    why.sort_by(|a, b| b.1.cmp(a.1));
    for (w, n) in why {
        println!("  {n}\t{w}");
    }
    Ok(())
}

/// THE RULES THAT FIRE on every kernel given: whether it reads, renders
/// the same text with its parameters, and passes the layered check
/// (`knob::check_text`, the legality checker first); the sites of every rule;
/// and, after one step of each climbing rule (vec4-tile, grow-rows; its
/// first site), the rules that fire on that child.
fn fires(files: &[PathBuf], kingdom: &WgslKingdom) -> Result<(), String> {
    let climbing = [rules::Rule::VecTile, rules::Rule::GrowRows];
    print!("member\treads\tchecks");
    for r in rules::Rule::ALL {
        print!("\t{}", r.name());
    }
    for c in climbing {
        print!("\tafter {}", c.name());
    }
    println!();
    let (mut read, mut checked, mut climbs, mut climbs_again) = (0usize, 0usize, 0usize, 0usize);
    let mut fired: BTreeMap<&str, usize> = BTreeMap::new();
    let mut depths: BTreeMap<usize, usize> = BTreeMap::new();
    for f in files {
        let name = f.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let text = std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        let (Ok(lit), Ok(r)) = (read_safe_literal(&text, kingdom), read_safe(&text, kingdom)) else {
            println!("{name}\tno");
            continue;
        };
        read += 1;
        let same = render_tree(&lit.tree, kingdom)?.text == render_tree(&r.tree, kingdom)?.text;
        let check = knob::check_text(&r.tree, kingdom);
        if same && check.is_ok() {
            checked += 1;
        }
        print!("{name}\tyes\t{}", match (&check, same) {
            (Ok(_), true) => "pass".to_string(),
            (Ok(_), false) => "our software has a bug: the parameters change the text".to_string(),
            (Err((s, e)), _) => format!("{}: {}", s.name(), e.chars().take(80).collect::<String>()),
        });
        for rule in rules::Rule::ALL {
            let n = rule.sites(&r.tree, kingdom).len();
            if n > 0 {
                *fired.entry(rule.name()).or_default() += 1;
            }
            print!("\t{n}");
        }
        let mut climbed = false;
        let mut again = false;
        for c in climbing {
            let sites = c.sites(&r.tree, kingdom);
            let after: Vec<String> = match sites.first().map(|site| c.apply(&r.tree, site, kingdom)) {
                Some(Ok(child)) => {
                    climbed = true;
                    let back = knob::check_text(&child, kingdom).map(|(_, b)| b.tree);
                    match back {
                        Ok(t) => {
                            let fire: Vec<String> = rules::Rule::ALL.iter().filter(|r| !r.sites(&t, kingdom).is_empty()).map(|r| r.name().to_string()).collect();
                            if fire.iter().any(|n| n == "vec4-tile" || n == "grow-rows") {
                                again = true;
                            }
                            fire
                        }
                        Err((s, e)) => vec![format!("our software has a bug: the child fails {}: {}", s.name(), e.chars().take(60).collect::<String>())],
                    }
                }
                Some(Err(e)) => vec![format!("our software has a bug: {e}")],
                None => vec!["-".to_string()],
            };
            print!("\t{}", after.join(","));
        }
        climbs += usize::from(climbed);
        climbs_again += usize::from(again);
        let depth = chain_depth(&r.tree, kingdom, 3);
        *depths.entry(depth).or_default() += 1;
        println!("\tchain {depth}");
    }
    println!("\nmembers {}: read {read}, read with the same text and passing the check {checked}", files.len());
    println!("members a rule fires on, by rule: {fired:?}");
    println!("members with a climbing step: {climbs}; whose child a climbing rule (vec4-tile or grow-rows) fires on again: {climbs_again}");
    let at_least = |n: usize| depths.iter().filter(|(d, _)| **d >= n).map(|(_, c)| c).sum::<usize>();
    println!("members by their longest chain of climbing steps (up to 3): {depths:?}; 1 or more {}, 2 or more {}, 3 or more {}", at_least(1), at_least(2), at_least(3));
    Ok(())
}

/// The longest chain of climbing steps (vec4-tile, grow-rows; every site
/// tried) from a kernel, up to `limit`, each child passing the layered
/// check and read back before the next step.
fn chain_depth(t: &fuller::wgsl::render::Tree, kingdom: &WgslKingdom, limit: usize) -> usize {
    if limit == 0 {
        return 0;
    }
    let mut best = 0;
    for rule in [rules::Rule::VecTile, rules::Rule::GrowRows] {
        for site in rule.sites(t, kingdom) {
            let Ok(child) = rule.apply(t, &site, kingdom) else { continue };
            let Ok((_, back)) = knob::check_text(&child, kingdom) else { continue };
            best = best.max(1 + chain_depth(&back.tree, kingdom, limit - 1));
            if best == limit {
                return best;
            }
        }
    }
    best
}

/// A knob's class for the table (a workgroup array's length as one class).
fn knob_class(s: &knob::Site) -> String {
    match s {
        knob::Site::Len(_) => "len".into(),
        other => other.name(),
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let kingdom = WgslKingdom::load_safe();
    match args.first().map(String::as_str) {
        Some("--survey") => survey(&kernels(&args[1..])?, &kingdom),
        Some("--fires") => fires(&kernels(&args[1..])?, &kingdom),
        Some("--mutate") => {
            let n: usize = args.get(1).ok_or("--mutate needs N")?.parse().map_err(|e| format!("--mutate: {e}"))?;
            let mut rest: Vec<String> = args[2..].to_vec();
            let mut take = |flag: &str| -> Result<Option<String>, String> {
                match rest.iter().position(|a| a == flag) {
                    Some(i) => {
                        let v = rest.get(i + 1).ok_or(format!("{flag} needs a value"))?.clone();
                        rest.drain(i..i + 2);
                        Ok(Some(v))
                    }
                    None => Ok(None),
                }
            };
            let seed: u64 = take("--seed")?.unwrap_or_else(|| "1".into()).parse().map_err(|e| format!("--seed: {e}"))?;
            let out = PathBuf::from(take("--out")?.ok_or("--out <dir> is required")?);
            mutate_all(&kernels(&rest)?, n, seed, &out, &kingdom)
        }
        Some("--rule") => {
            let name = args.get(1).ok_or("--rule needs a rule name")?;
            let rule = rules::Rule::ALL.into_iter().find(|r| r.name() == name).ok_or_else(|| format!("no rule {name}; rules: {}", rules::Rule::ALL.map(|r| r.name()).join(", ")))?;
            let n: usize = args.get(2).ok_or("--rule needs N")?.parse().map_err(|e| format!("N: {e}"))?;
            let mut rest: Vec<String> = args[3..].to_vec();
            let mut take = |flag: &str| -> Result<Option<String>, String> {
                match rest.iter().position(|a| a == flag) {
                    Some(i) => {
                        let v = rest.get(i + 1).ok_or(format!("{flag} needs a value"))?.clone();
                        rest.drain(i..i + 2);
                        Ok(Some(v))
                    }
                    None => Ok(None),
                }
            };
            let seed: u64 = take("--seed")?.unwrap_or_else(|| "1".into()).parse().map_err(|e| format!("--seed: {e}"))?;
            let out = PathBuf::from(take("--out")?.ok_or("--out <dir> is required")?);
            rule_all(&kernels(&rest)?, rule, n, seed, &out, &kingdom)
        }
        _ => Err("usage: wgsl_knob --survey <dir|file>... | --mutate N [--seed S] --out <dir> <dir|file>... | --rule NAME N [--seed S] --out <dir> <dir|file>...".into()),
    }
}
