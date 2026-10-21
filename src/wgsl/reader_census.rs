//! Which tracked kernels the safe reader does not read (T43), asserted against the list in
//! `tests/fixtures/wgsl_genomes/REFUSED.md`. The directories are those of `examples/wgsl_reader_census`.

use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::loader::WgslKingdom;

/// The `.wgsl` kernels of a directory and the `.wgsl.gz` ones, by file name (without `.gz`), sorted.
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

/// The directories the census reads, relative to the crate root.
fn directories(root: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = ["tests/corpus/safewgsl", "tests/corpus/interfaces", "tests/corpus/nluca", "tests/corpus/model_replies", "tests/corpus/cuneus", "tests/corpus/harvest", "logs/HGP/001/gen0", "logs/HGP/001/safe", "tests/fixtures/v24_seeds5", "tests/fixtures"].iter().map(|d| root.join(d)).collect();
    for sub in ["tests/fixtures/wgsl_genomes", "tests/fixtures/wgsl_texts", "problems"] {
        let mut v: Vec<PathBuf> = std::fs::read_dir(root.join(sub)).map(|r| r.flatten().map(|e| e.path()).collect()).unwrap_or_default();
        v.sort();
        dirs.extend(v.into_iter().map(|d| if sub == "tests/fixtures/wgsl_genomes" { d.join("texts") } else { d }));
    }
    dirs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EXACTLY THE LISTED KERNELS ARE REFUSED. A kernel the reader newly reads (a closed gap) or newly refuses fails here until
    /// REFUSED.md says so; and the census must have looked at the kernels it has always looked at (a floor on how many).
    #[test]
    fn every_tracked_kernel_the_reader_refuses_is_listed() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let kingdom = WgslKingdom::load_safe();
        let (mut seen, mut refused) = (0usize, BTreeSet::new());
        for d in directories(root) {
            let rel = d.strip_prefix(root).unwrap_or(&d).display().to_string();
            for (name, text) in kernels(&d) {
                seen += 1;
                if crate::wgsl::safe_reader::read_safe(&text, &kingdom).is_err() {
                    refused.insert(format!("{rel} | {name}"));
                }
            }
        }
        assert!(seen >= 3071, "the census looked at {seen} kernels; it looked at 3,071 when this list was made");
        let md = std::fs::read_to_string(root.join("tests/fixtures/wgsl_genomes/REFUSED.md")).expect("REFUSED.md");
        let listed: BTreeSet<String> = md
            .lines()
            .filter(|l| l.starts_with("| ") && !l.starts_with("| directory") && !l.starts_with("|---"))
            .filter_map(|l| {
                let cells: Vec<&str> = l.trim_matches('|').split('|').map(str::trim).collect();
                (cells.len() >= 4).then(|| format!("{} | {}", cells[0], cells[1]))
            })
            .collect();
        let (unlisted, unrefused): (Vec<_>, Vec<_>) = (refused.difference(&listed).collect(), listed.difference(&refused).collect());
        assert!(unlisted.is_empty() && unrefused.is_empty(), "refused but not listed in REFUSED.md: {unlisted:?}\nlisted but read now (or gone): {unrefused:?}");
    }
}
