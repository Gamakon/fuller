//! T43 groundwork: which tracked kernels the safe reader (the any-WGSL gate) does not read.
//!
//!   cargo run --release --features wgsl --example wgsl_reader_census
//!
//! Reads every tracked `.wgsl` kernel (the corpora, gen0, the safe kernels, the seeds, the saved-genome texts, the problems'
//! references) and every gzipped one under tests/fixtures/wgsl_texts, with `read_safe`, and prints one line per directory (kernels,
//! read, refused) and then each refused kernel with the first line of the reader's error and its class.

use std::io::Read;
use std::path::{Path, PathBuf};

use fuller::wgsl::loader::WgslKingdom;

fn kernels(dir: &Path) -> Vec<(String, String)> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).map(|r| r.flatten().map(|e| e.path()).collect()).unwrap_or_default();
    files.sort();
    files
        .iter()
        .filter_map(|f| {
            let name = f.file_name()?.to_string_lossy().to_string();
            if name.ends_with(".wgsl") {
                Some((name, std::fs::read_to_string(f).ok()?))
            } else if name.ends_with(".wgsl.gz") {
                let mut text = String::new();
                flate2::read::GzDecoder::new(std::fs::File::open(f).ok()?).read_to_string(&mut text).ok()?;
                Some((name.trim_end_matches(".gz").to_string(), text))
            } else {
                None
            }
        })
        .collect()
}

/// The refusal's class: the first words of its first line, numbers and quoted names removed.
fn class(error: &str) -> String {
    let line = error.lines().next().unwrap_or("");
    let line = line.rsplit(": ").next().unwrap_or(line);
    let mut out = String::new();
    let mut quoted = false;
    for c in line.chars() {
        match c {
            '`' | '"' => quoted = !quoted,
            _ if quoted || c.is_ascii_digit() => {}
            _ => out.push(c),
        }
    }
    out.split_whitespace().take(9).collect::<Vec<_>>().join(" ")
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let kingdom = WgslKingdom::load_safe();
    let mut dirs: Vec<PathBuf> = ["tests/corpus/safewgsl", "tests/corpus/interfaces", "tests/corpus/nluca", "tests/corpus/model_replies", "tests/corpus/cuneus", "tests/corpus/harvest", "logs/HGP/001/gen0", "logs/HGP/001/safe", "tests/fixtures/v24_seeds5", "tests/fixtures"].iter().map(|d| root.join(d)).collect();
    for sub in ["tests/fixtures/wgsl_genomes", "tests/fixtures/wgsl_texts", "problems"] {
        let mut v: Vec<PathBuf> = std::fs::read_dir(root.join(sub)).map(|r| r.flatten().map(|e| e.path()).collect()).unwrap_or_default();
        v.sort();
        for d in v {
            if sub == "tests/fixtures/wgsl_genomes" {
                dirs.push(d.join("texts"));
            } else if sub == "problems" {
                dirs.push(d);
            } else {
                dirs.push(d);
            }
        }
    }
    println!("directory\tkernels\tread\trefused");
    let mut refused: Vec<(String, String, String)> = Vec::new();
    for d in &dirs {
        let ks = kernels(d);
        if ks.is_empty() {
            continue;
        }
        let mut bad = 0;
        for (name, text) in &ks {
            if let Err(e) = fuller::wgsl::safe_reader::read_safe(text, &kingdom) {
                bad += 1;
                refused.push((d.strip_prefix(root).unwrap_or(d).display().to_string(), name.clone(), e));
            }
        }
        println!("{}\t{}\t{}\t{}", d.strip_prefix(root).unwrap_or(d).display(), ks.len(), ks.len() - bad, bad);
    }
    println!("\nclass\tkernels");
    let mut by_class: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for (_, _, e) in &refused {
        *by_class.entry(class(e)).or_default() += 1;
    }
    let mut classes: Vec<_> = by_class.into_iter().collect();
    classes.sort_by(|a, b| b.1.cmp(&a.1));
    for (c, n) in classes {
        println!("{c}\t{n}");
    }
    println!("\ndirectory\tkernel\tfirst error line");
    for (d, n, e) in refused {
        println!("{d}\t{n}\t{}", e.lines().next().unwrap_or("").chars().take(3000).collect::<String>());
    }
}
