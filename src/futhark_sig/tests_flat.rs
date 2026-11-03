//! THE ANSWER-SCORED FLAT SCHEME'S WEIGHT, MEASURED (`futhark-slope4-qqs`, `fitness::ANSWER_QUALITY_WEIGHT`; Andrew's
//! ruling of 2026-11-03: no gate, quality weighs more than speed by repetition only). On histogram's and pathfinder's
//! stored cases, the inexact one-edit neighbours of the published program (point mutations of its gene, through the
//! host interpreter) are each given the best speed there is, every one of the 11 timing slots 0, and the exact
//! published program its own speed, every slot 1/2: the least repetition count at which the exact program ranks above
//! every one of them is printed and the recorded constant is at least it.

use super::*;
use crate::futhark_rings::point_mutate;
use crate::futhark_table::GeneRng;
use crate::typed_gep::{express, write_back, GeneLayout};

/// The sum of squares of an inexact neighbour's quality inputs decides its rank against the exact program at the
/// reference's speed: with S speed inputs the exact program at 1/2 has sum S/4, the neighbour at speed 0 has
/// m * sum q^2, so the least m that ranks the exact one above is the least integer with m * sum q^2 > S / 4.
#[test]
fn an_inexact_program_ranks_below_the_exact_one_at_the_recorded_weight() {
    const SLOTS: usize = 11;
    const MUTANTS: usize = 80;
    let mut least_overall = 1usize;
    let mut report = Vec::new();
    for name in ["histogram", "pathfinder"] {
        let p = problem(name).unwrap();
        let k = load(&p.instance).unwrap();
        let sig = signature(&p.instance).unwrap();
        let interp = Interp::new(&k, &sig).unwrap();
        let contract = k.relations().check().map_err(|d| format!("{d:?}")).unwrap();
        let layout = GeneLayout::from_program_size(&contract, k.roots()[0], p.program_size).unwrap();
        let const_len = k.const_len();
        let reference = build(&k, p.reference().unwrap()).unwrap();
        let cases = load_cases(&p).unwrap();
        let fit: Vec<&Case> = cases.iter().filter(|c| c.split.in_fitness()).collect();
        let mut rng = GeneRng::new(0xF1A7_0000 + name.len() as u64);
        let base = write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut rng).unwrap(), &reference).unwrap();
        // The exact program's inputs are all 0 on every case: its flat vector at the reference's speed.
        let exact_q: Vec<f64> = fit.iter().flat_map(|c| vec![0.0; quality_input_names_of(QUALITY_PARTS, c.answer().unwrap()).len()]).collect();
        let (mut inexact, mut exact, mut failed) = (Vec::new(), 0, 0);
        for _ in 0..MUTANTS {
            let mut gene = base.clone();
            point_mutate(&contract, &layout, &const_len, &mut gene, 1, &mut rng).unwrap();
            let tree = express(&contract, &layout, &gene, &const_len).unwrap().tree();
            if tree == reference {
                continue;
            }
            let mut q = Vec::new();
            let mut all_exact = true;
            let mut ran = true;
            for c in &fit {
                match interp.eval(&tree, &c.inputs) {
                    Ok(got) => {
                        let want = c.answer().unwrap();
                        all_exact &= compare(&got, want) == [0.0; 3];
                        q.extend(quality_inputs_of(QUALITY_PARTS, &got, want));
                    }
                    Err(_) => {
                        ran = false;
                        break;
                    }
                }
            }
            if !ran {
                failed += 1;
            } else if all_exact {
                exact += 1;
            } else {
                inexact.push(q);
            }
        }
        assert!(inexact.len() >= 10, "{name}: only {} inexact neighbours of {MUTANTS} mutants", inexact.len());
        let sums: Vec<f64> = inexact.iter().map(|q| q.iter().map(|x| x * x).sum::<f64>()).collect();
        let min_sum = sums.iter().copied().fold(f64::INFINITY, f64::min);
        let least = (1..=10_000).find(|m| *m as f64 * min_sum > SLOTS as f64 / 4.0).expect("a weight under 10,000");
        least_overall = least_overall.max(least);
        // At the recorded weight: the exact program at its own speed against every inexact neighbour at the best
        // speed there is, through the scheme's own function.
        let exact_f = crate::fitness::qqs(&exact_q, &[0.5; SLOTS], crate::fitness::ANSWER_QUALITY_WEIGHT).unwrap().fitness;
        let worst_neighbour = inexact.iter().map(|q| crate::fitness::qqs(q, &[0.0; SLOTS], crate::fitness::ANSWER_QUALITY_WEIGHT).unwrap().fitness).fold(f64::INFINITY, f64::min);
        eprintln!("{name}: {} inexact, {exact} exact, {failed} did not run of {MUTANTS} one-edit neighbours over {} fitness cases ({} quality inputs); the least sum of squares of an inexact neighbour's quality inputs {min_sum:.4} (the largest {:.4}); the least m that ranks the exact program (every slot 1/2) above every inexact one (every slot 0) is {least}; at m = {} the exact program scores {exact_f:.4}, the best inexact neighbour {worst_neighbour:.4}", inexact.len(), fit.len(), exact_q.len(), sums.iter().copied().fold(0.0, f64::max), crate::fitness::ANSWER_QUALITY_WEIGHT);
        report.push((name, exact_f, worst_neighbour, least));
    }
    for (name, exact_f, worst_neighbour, least) in &report {
        assert!(exact_f < worst_neighbour, "{name}: at m = {} an inexact neighbour at zero cost ranks above the exact program at the reference's cost: {exact_f} against {worst_neighbour}; the least m is {least}", crate::fitness::ANSWER_QUALITY_WEIGHT);
    }
    eprintln!("the least m over both problems is {least_overall}; fitness::ANSWER_QUALITY_WEIGHT is {}", crate::fitness::ANSWER_QUALITY_WEIGHT);
    assert!(crate::fitness::ANSWER_QUALITY_WEIGHT >= least_overall, "the recorded weight {} is below the measured least {least_overall}", crate::fitness::ANSWER_QUALITY_WEIGHT);
}
