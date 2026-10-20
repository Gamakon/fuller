//! The design's round trip on the typed core (docs/MultiTypedGEP, test 5):
//! WGSL text, read, encode into a typed gene, express, render; the tree and the
//! text equal the read tree's and the text naga-valid. Counts per corpus and the
//! named kingdom gaps by count.
//!
//!   cargo run --release --features wgsl --example wgsl_typed_roundtrip -- <label>=<dir> ...
//!
//! A directory is searched for `*.wgsl`, recursively.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fuller::wgsl::loader::WgslKingdom;
use fuller::wgsl::typed_gene::{round_trip, Alphabets, Outcome};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            files(&p, out);
        } else if p.extension().is_some_and(|x| x == "wgsl") {
            out.push(p);
        }
    }
}

fn key(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_digit() { 'N' } else { c }).take(110).collect()
}

fn main() {
    let kingdom = WgslKingdom::load_safe();
    let mut alphabets = Alphabets::default();
    for arg in std::env::args().skip(1) {
        let (label, dir) = arg.split_once('=').unwrap_or((&arg, &arg));
        let mut list = Vec::new();
        files(Path::new(dir), &mut list);
        let started = std::time::Instant::now();
        let (mut read, mut unrenderable, mut encoded, mut tree_equal, mut text_equal, mut valid) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut gaps: BTreeMap<String, (usize, String)> = BTreeMap::new();
        let mut note = |reason: String, detail: String| {
            gaps.entry(key(&reason)).or_insert((0, detail)).0 += 1;
        };
        for f in &list {
            let Ok(text) = std::fs::read_to_string(f) else { continue };
            let here = f.display().to_string();
            match round_trip(&mut alphabets, &kingdom, &text) {
                Outcome::NotRead(e) => note(format!("not read: {}", e.lines().next().unwrap_or("")), here),
                Outcome::Unrenderable(e) => {
                    read += 1;
                    unrenderable += 1;
                    note(format!("read tree does not render: {e}"), here);
                }
                Outcome::Gap(g) => {
                    read += 1;
                    note(format!("{}: {}", g.reason, if g.reason == "no row" { g.detail.split(' ').next().unwrap_or("").to_string() } else { g.detail.clone() }), format!("{here}: {}", g.detail));
                }
                Outcome::Expressed { tree_equal: t, text_equal: x, valid: v, difference } => {
                    read += 1;
                    encoded += 1;
                    tree_equal += usize::from(t);
                    text_equal += usize::from(x);
                    if !t {
                        note(format!("tree differs: {difference}"), here.clone());
                    }
                    if !x {
                        note("text differs".into(), here.clone());
                    }
                    match v {
                        Ok(()) => valid += 1,
                        Err(e) => note(format!("naga: {}", e.lines().next().unwrap_or("")), here),
                    }
                }
            }
        }
        println!("== {label}: {} files in {:.0} s", list.len(), started.elapsed().as_secs_f64());
        println!("   read {read} (not read {}); read tree renders {} ; encoded, expressed and rendered {encoded}; tree equal {tree_equal}; text equal {text_equal}; naga-valid {valid}", list.len() - read, read - unrenderable);
        let mut top: Vec<(&String, &(usize, String))> = gaps.iter().collect();
        top.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        let show: usize = std::env::var("TOP").ok().and_then(|v| v.parse().ok()).unwrap_or(14);
        for (reason, (n, first)) in top.iter().take(show) {
            println!("   {n:>6}  {reason}    e.g. {}", first.chars().take(160).collect::<String>());
        }
    }
}
