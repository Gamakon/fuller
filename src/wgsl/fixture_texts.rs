//! The kernel texts the tests read, tracked under `tests/fixtures/wgsl_texts/<set>/<name>.wgsl.gz` (T63, 2026-10-21). They were
//! read from `logs/`, which is not tracked, so a clean checkout failed eight tests that pass on the machine that has the logs.
//! Each set holds exactly the `.wgsl` files its source directory held; a set that is missing or has another number of files
//! than recorded here fails the test that asks for it, loudly.

use std::io::Read;

/// (set, source directory when it was copied, files). A set's files end `.wgsl.gz`, except `nluca_s3_replies`, whose
/// model replies end `.reply.txt.gz`.
pub(crate) const SETS: [(&str, &str, usize); 12] = [
    ("popA_g001", "logs/HGP/001/popA_g001/texts", 20),
    ("popA2_g001b", "logs/HGP/001/popA2_g001b/texts", 20),
    ("library_texts", "logs/HGP/001/library_texts", 22),
    ("popB_g080", "logs/HGP/001/popB_g080/texts", 20),
    ("reserve2", "logs/HGP/001/reserve2/texts", 283),
    ("popC_rand20", "logs/HGP/001/popC_rand20/texts", 20),
    ("popR_rules", "logs/HGP/001/popR_rules/texts", 49),
    ("v24_run1_seeds", "logs/V24/run1/seeds", 32),
    ("v24_run1_raw", "logs/V24/run1/raw", 73),
    ("nluca_s3_seeds", "logs/HGP/001/nluca_s3/seeds", 24),
    ("opus_cross", "logs/HGP/001/opus_cross", 1),
    ("nluca_s3_replies", "logs/HGP/001/nluca_s3/calls", 63),
];

/// The texts of a set, `(file name, text)`, sorted by file name.
///
/// # Panics
/// When the set is not one of [`SETS`], its fixture directory is missing, or it holds another number of files.
pub(crate) fn set(name: &str) -> Vec<(String, String)> {
    let (_, source, expected) = SETS.iter().find(|(n, _, _)| *n == name).unwrap_or_else(|| panic!("fixture set {name}: not one of the tracked sets"));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wgsl_texts").join(name);
    let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("fixture set {name} ({source}) is missing: {}: {e}", dir.display()));
    let mut files: Vec<_> = entries.flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with(".gz")).collect();
    files.sort();
    assert_eq!(files.len(), *expected, "fixture set {name} ({source}): {} files, expected {expected}", files.len());
    files
        .iter()
        .map(|f| {
            let file = f.file_name().map(|s| s.to_string_lossy().trim_end_matches(".gz").to_string()).unwrap_or_default();
            let mut text = String::new();
            flate2::read::GzDecoder::new(std::fs::File::open(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()))).read_to_string(&mut text).unwrap_or_else(|e| panic!("{name}/{file}: {e}"));
            (file, text)
        })
        .collect()
}

/// One text of a set by its file name (`name.wgsl`).
///
/// # Panics
/// When the set is not there as [`set`] requires, or has no file of that name.
pub(crate) fn text(set_name: &str, file: &str) -> String {
    set(set_name).into_iter().find(|(n, _)| n == file).map(|(_, t)| t).unwrap_or_else(|| panic!("fixture set {set_name} has no {file}"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_set_is_there_with_its_count() {
        for (name, _, expected) in super::SETS {
            assert_eq!(super::set(name).len(), expected);
        }
    }
}
