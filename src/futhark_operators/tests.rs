//! The domain operators, measured on the CPU (no fitness, no device): per operator, on pathfinder and histogram at
//! the runner's default program size, how often one application changes the expressed program, and what share of the
//! cells it changes are coding cells (of the parent or, for a crossover, of either parent). The review of 2026-11-01
//! (F2) said the domain operators land on a non-coding cell 99.97 % of the time; this is the per-operator count. Every
//! child must also validate and express (legal by construction).

use super::*;
use crate::futhark_sig::{build, load, problem, random_gene, DEFAULT_PROGRAM_SIZE};
use crate::typed_gep::{express, validate, write_back, Tree};

/// The `--ops` names measured, the span operators, the row operators, the relative rows and the aimed rows.
const KINDS: [&str; 12] = ["invert", "transpose", "crossover", "invert-rows", "transpose-rows", "cross-rows", "invert-rel", "transpose-rel", "cross-rel", "invert-rows-coding", "transpose-rows-coding", "cross-rows-coding"];

/// One application of the operator `kind` to `a` (and its partner `b`): whether it applied.
fn apply(kind: &str, contract: &KingdomContract, layout: &GeneLayout, const_len: &BTreeMap<Ty, u32>, a: &mut [u32], b: &mut [u32], rng: &mut GeneRng) -> bool {
    let coding = coding_cells(contract, layout, a, const_len).unwrap();
    match kind {
        "invert" => invert_once(layout, a, &coding, rng).unwrap().is_some(),
        "transpose" => transpose_once(contract, layout, a, &coding, rng).unwrap().is_some(),
        "crossover" => {
            crossover_once(contract, layout, const_len, a, b, rng).unwrap();
            true
        }
        "invert-rows" => invert_rows_once(layout, a, rng).unwrap().is_some(),
        "transpose-rows" => transpose_rows_once(contract, layout, a, rng).unwrap().is_some(),
        "cross-rows" => cross_rows_once(layout, a, b, rng).unwrap().is_some(),
        "invert-rel" => invert_rel_once(layout, a, rng).unwrap().is_some(),
        "transpose-rel" => transpose_rel_once(contract, layout, a, rng).unwrap().is_some(),
        "cross-rel" => cross_rel_once(layout, a, b, rng).unwrap().is_some(),
        "invert-rows-coding" => invert_rows_coding_once(layout, a, &coding, rng).unwrap().is_some(),
        "transpose-rows-coding" => transpose_rows_coding_once(contract, layout, a, &coding, rng).unwrap().is_some(),
        "cross-rows-coding" => cross_rows_coding_once(contract, layout, const_len, a, b, rng).unwrap().is_some(),
        other => panic!("no operator {other}"),
    }
}

/// Per operator: (applications, children whose expressed program differs from the parent's, cells changed, of which
/// coding in the parent or partner).
fn measure(name: &str, trials: usize) -> Vec<(&'static str, usize, usize, usize, usize)> {
    let prob = problem(name).unwrap();
    let k = load(&prob.instance).unwrap();
    let contract = k.relations().check().unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], DEFAULT_PROGRAM_SIZE).unwrap();
    let const_len = k.const_len();
    let parents: Vec<Tree> = [prob.reference, prob.planted].iter().map(|t| build(&k, t).unwrap()).collect();
    let mut out = Vec::new();
    for kind in KINDS {
        let (mut pad, mut op) = (GeneRng::new(0x0FE7_0001), GeneRng::new(0x0FE7_0002));
        let (mut applied, mut changed, mut cells, mut coding_cells_changed) = (0, 0, 0, 0);
        for i in 0..trials {
            let pa = &parents[i % 2];
            let pb = &parents[(i + 1) % 2];
            let mut a = write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut pad).unwrap(), pa).unwrap();
            let mut b = write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut pad).unwrap(), pb).unwrap();
            let (a0, b0) = (a.clone(), b.clone());
            let coding: std::collections::BTreeSet<usize> = coding_cells(&contract, &layout, &a0, &const_len).unwrap().into_iter().chain(coding_cells(&contract, &layout, &b0, &const_len).unwrap()).collect();
            if !apply(kind, &contract, &layout, &const_len, &mut a, &mut b, &mut op) {
                continue;
            }
            applied += 1;
            for g in [&a, &b] {
                validate(&contract, &layout, g, &const_len).unwrap_or_else(|e| panic!("{name} {kind}: a child does not validate: {e}"));
            }
            let before = express(&contract, &layout, &a0, &const_len).unwrap().tree();
            let after = express(&contract, &layout, &a, &const_len).unwrap().tree();
            changed += usize::from(before != after);
            let diff: Vec<usize> = (0..a.len()).filter(|p| a[*p] != a0[*p]).collect();
            cells += diff.len();
            coding_cells_changed += diff.iter().filter(|p| coding.contains(p)).count();
        }
        out.push((kind, applied, changed, cells, coding_cells_changed));
    }
    out
}

/// The share of the review's per-type pick that is coding: (type, domain) spans holding a coding cell of the reference
/// on its padding, of all spans; coding cells of all cells.
fn coding_share(name: &str) -> (usize, usize, usize, usize) {
    let prob = problem(name).unwrap();
    let k = load(&prob.instance).unwrap();
    let contract = k.relations().check().unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], DEFAULT_PROGRAM_SIZE).unwrap();
    let const_len = k.const_len();
    let g = write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut GeneRng::new(0x0FE7_0003)).unwrap(), &build(&k, prob.reference).unwrap()).unwrap();
    let coding = coding_cells(&contract, &layout, &g, &const_len).unwrap();
    let spans = layout.spans();
    let coding_spans = spans.iter().filter(|s| coding.iter().any(|p| (s.off..s.off + s.len).contains(p))).count();
    (coding_spans, spans.len(), coding.len(), layout.width)
}

#[test]
fn the_operators_measured_per_kind_on_pathfinder_and_histogram() {
    let mut results = Vec::new();
    for name in ["pathfinder", "histogram"] {
        let (cs, s, cc, w) = coding_share(name);
        eprintln!("{name}: N = {DEFAULT_PROGRAM_SIZE}; spans coding {cs} of {s}; cells coding {cc} of {w}");
        let got = measure(name, 100);
        for (kind, applied, changed, cells, coding) in &got {
            eprintln!("{name}\t{kind}\tapplied {applied}\tchanged the program {changed}\tcells changed {cells}\tof them coding {coding}");
        }
        // (operator, applications of 100, children whose program changed, cells changed, of them coding), measured
        // 2026-11-02 on the count-context layout (docs/PLAN_futhark_kingdom.md 3g; on the stack-context layout the
        // programs changed were within 4 of these, cross-rows 0 and 0). The unaimed row operators change the program
        // in 0 to 67 of 100 applications (cross-rows in 1 and 0), the aimed ones in 36 to 100.
        let want: &[(&str, usize, usize, usize, usize)] = match name {
            "pathfinder" => &[
                ("invert", 88, 88, 2286, 562),
                ("transpose", 92, 57, 3518, 367),
                ("crossover", 100, 88, 2239, 532),
                ("invert-rows", 82, 6, 221674, 91),
                ("transpose-rows", 83, 44, 293624, 1011),
                ("cross-rows", 100, 1, 643936, 5),
                ("invert-rel", 95, 11, 279592, 78),
                ("transpose-rel", 100, 67, 493564, 1789),
                ("cross-rel", 100, 6, 1371604, 62),
                ("invert-rows-coding", 100, 100, 397628, 2785),
                ("transpose-rows-coding", 100, 59, 430034, 1566),
                ("cross-rows-coding", 100, 97, 3058715, 2584),
            ],
            _ => &[
                ("invert", 100, 92, 2404, 157),
                ("transpose", 99, 36, 4522, 81),
                ("crossover", 100, 29, 2398, 65),
                ("invert-rows", 100, 1, 249930, 4),
                ("transpose-rows", 84, 41, 261914, 289),
                ("cross-rows", 100, 0, 808366, 0),
                ("invert-rel", 95, 0, 257614, 0),
                ("transpose-rel", 97, 53, 423655, 432),
                ("cross-rel", 100, 0, 1273602, 0),
                ("invert-rows-coding", 100, 98, 358288, 679),
                ("transpose-rows-coding", 100, 36, 378587, 119),
                ("cross-rows-coding", 100, 94, 2622237, 256),
            ],
        };
        results.push((name, got == want, (cs, s, cc, w)));
    }
    for (name, same, shares) in results {
        assert!(same, "{name}: the operator measurement differs from the pinned one (printed above)");
        assert_eq!(shares, if name == "pathfinder" { (8, 404, 24, 57487) } else { (5, 354, 7, 47999) }, "{name}: the coding spans and cells");
    }
}

#[test]
fn the_aimed_row_crossover_is_an_exchange_and_both_children_are_legal() {
    let prob = problem("histogram").unwrap();
    let k = load(&prob.instance).unwrap();
    let contract = k.relations().check().unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], DEFAULT_PROGRAM_SIZE).unwrap();
    let const_len = k.const_len();
    let mut rng = GeneRng::new(7);
    for _ in 0..50 {
        let mut a = random_gene(&contract, &layout, &const_len, &mut rng).unwrap();
        let mut b = random_gene(&contract, &layout, &const_len, &mut rng).unwrap();
        let (a0, b0) = (a.clone(), b.clone());
        cross_rows_coding_once(&contract, &layout, &const_len, &mut a, &mut b, &mut rng).unwrap().unwrap();
        // Legal both ways, and the cells moved only between the two parents at the same position.
        for p in 0..a.len() {
            assert!((a[p] == a0[p] && b[p] == b0[p]) || (a[p] == b0[p] && b[p] == a0[p]), "cell {p} is not an exchange");
        }
        validate(&contract, &layout, &a, &const_len).unwrap();
        validate(&contract, &layout, &b, &const_len).unwrap();
    }
}
