//! How long does one generation's variation phase take on the device?
//!
//!   cargo run --release --features gpu --example evolve_speed -- [population] [generations]
//!
//! Selection (two tournaments per island + elites), cloning, the nine mutation
//! operators and the three crossovers, for a population shaped like the SRBench
//! entry's (3 genes, head 48, 10 constants; islands 3:1, tournament 7%).
//! Fitness is a stand-in uploaded each generation: scoring is step 3.

use std::time::Instant;

use fuller::evolve::device::EvolveDevice;
use fuller::evolve::vary::{GenParams, Island, Rates};
use fuller::evolve::{InitParams, Layout, SymbolCodes};

fn main() {
    let args: Vec<u32> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let pop = args.first().copied().unwrap_or(800);
    let generations = args.get(1).copied().unwrap_or(1000);
    let intake = pop / 4 * 3;
    let layout = Layout::for_arity(pop, 3, 48, 2, 10);
    let rates = Rates::engine_defaults(layout);
    let islands = [
        Island { lo: 0, hi: intake, elites: 2, tournsize: (intake * 7 / 100).max(2), rates },
        Island { lo: intake, hi: pop, elites: 2, tournsize: ((pop - intake) * 7 / 100).max(2), rates },
    ];
    // 12 functions (8 binary, 4 unary) and 8 terminals: the size of a real primitive set.
    let codes = SymbolCodes {
        arity: [vec![2; 8], vec![1; 4], vec![0; 8]].concat(),
        sample_functions: (0..12).collect(),
        sample_terminals: (12..20).collect(),
        rnc_id: Some(19),
    };
    let mut dev = EvolveDevice::new(layout, &codes).expect("device");
    dev.init(&InitParams { seed: 1, generation: 0, rnc_lo: -100, rnc_hi: 100, n_wrappers: 3, vhead: 0 }).expect("init");
    let fitness: Vec<f32> = (0..pop).map(|r| ((r * 2_654_435_761u32.wrapping_mul(r + 1)) % 100_000) as f32).collect();
    dev.write_fitness(&fitness).expect("fitness");
    dev.finish();

    let t = Instant::now();
    for generation in 1..=generations {
        dev.vary(&islands, &GenParams { seed: 1, generation, rnc_lo: -100, rnc_hi: 100, cohort_merge: 0, vhead: 0 }).expect("vary");
        dev.write_fitness(&fitness).expect("fitness");
    }
    dev.finish();
    let per = t.elapsed().as_secs_f64() / f64::from(generations);
    println!(
        "population {pop} (islands {intake}+{}, tournaments {}+{}), {generations} generations: {:.3} ms per generation, {:.0} individuals per second",
        pop - intake,
        islands[0].tournsize,
        islands[1].tournsize,
        per * 1e3,
        f64::from(pop) / per
    );
    dev.read().expect("read").check(&codes).expect("the population still keeps the rules");
}
