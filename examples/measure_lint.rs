//! The Phase 2 gate for the K-expression linter (docs/PLAN_fuller_gpu.md).
//!
//! Runs the CPU reference linter over a corpus and reports, against egglog's
//! `smallest_form` on the same expressions: coverage (weighted nodes) for each
//! search and rule subset, soundness on probe rows, variety, steps per
//! expression, device eligibility, f32/f64 agreement, per-rule hits and time.
//!
//!   cargo run --release --example measure_lint -- exprs.tsv counts.txt egglog.tsv [psets.txt head_length]
//!
//! With `psets.txt` (one line per expression: `semantic_id/arity,..`, the
//! functions that problem's primitive set has) and a head length, it also
//! reports the write-back status of every expression under the spec's §6a
//! contract, for the form the DEVICE would produce (greedy, device subset,
//! f32) replayed in f64 on the host.
//!
//! `exprs.tsv`: `math<TAB>v1,v2,..` per line. `counts.txt`: one weight per
//! line. `egglog.tsv`: `input_cost<TAB>cost<TAB>seconds` per line, as written
//! by `measure_smallest_form`.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use fuller::gpu_eval::MAX_NODES;
use fuller::karva::{karva_to_terms, terms_to_karva_sized, FunctionSpec, PsetSpec};
use fuller::lint::engine::{run, CallerFacts, Config, LitMode, Search};
use fuller::lint::node::{Compiled, Tree};
use fuller::lint::tables::Tables;

/// Probe values for input columns: zero, negatives, the protected-divide band,
/// ordinary and large magnitudes.
const PROBE_VALUES: [f64; 8] = [-3.7, -1.0, 0.0, 1e-7, 0.4, 1.0, 2.9, 1e6];
const N_PROBE_ROWS: usize = 24;
const AGREE_REL_TOL: f64 = 1e-9;

struct Arm {
    name: &'static str,
    search: Search,
    mode: LitMode,
    finite_exact: bool,
    computed_literals: bool,
}

fn probe_rows(vars: &[String]) -> Vec<Vec<(String, f64)>> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..N_PROBE_ROWS)
        .map(|_| {
            vars.iter()
                .map(|v| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (v.clone(), PROBE_VALUES[(state >> 33) as usize % PROBE_VALUES.len()])
                })
                .collect()
        })
        .collect()
}

fn agree(a: f64, b: f64, scale: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= AGREE_REL_TOL * scale.max(a.abs()).max(b.abs())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (exprs, counts, egglog) = match args.as_slice() {
        [e, c, g] | [e, c, g, _, _] => (e, c, g),
        _ => panic!("usage: measure_lint exprs.tsv counts.txt egglog.tsv [psets.txt head_length]"),
    };
    let write_back: Option<(Vec<String>, usize)> = args.get(3).map(|p| {
        let psets = std::fs::read_to_string(p).expect("read psets").lines().map(str::to_string).collect();
        (psets, args[4].parse().expect("head_length"))
    });
    let exprs = std::fs::read_to_string(exprs).expect("read exprs");
    let counts: Vec<u64> = std::fs::read_to_string(counts)
        .expect("read counts")
        .lines()
        .map(|l| l.trim().parse().expect("count"))
        .collect();
    let egglog: Vec<u64> = std::fs::read_to_string(egglog)
        .expect("read egglog")
        .lines()
        .map(|l| l.split('\t').nth(1).and_then(|c| c.parse().ok()).expect("egglog cost"))
        .collect();
    let lines: Vec<&str> = exprs.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), counts.len(), "counts do not line up with expressions");
    assert_eq!(lines.len(), egglog.len(), "egglog costs do not line up with expressions");

    let t0 = Instant::now();
    let tables = Tables::standard().expect("tables load");
    let symbols = fuller::geneframe::master_table();
    let kingdom = symbols.kingdom("Symbolic Regression");
    tables.type_check(&kingdom).expect("type check");
    let rules = tables.usable(&kingdom);
    println!(
        "tables: {} rules usable of {} admitted, {} guard rows, {} refused; loaded in {:.3} s",
        rules.len(),
        tables.rules.len(),
        tables.guards.len(),
        tables.refused.len(),
        t0.elapsed().as_secs_f64()
    );

    let arms = [
        Arm { name: "beam8", search: Search::Beam(8), mode: LitMode::F64, finite_exact: true, computed_literals: true },
        Arm { name: "greedy", search: Search::Greedy, mode: LitMode::F64, finite_exact: true, computed_literals: true },
        Arm { name: "beam8 exact-only (no class F)", search: Search::Beam(8), mode: LitMode::F64, finite_exact: false, computed_literals: true },
        Arm { name: "greedy device-subset (no computed literals)", search: Search::Greedy, mode: LitMode::F64, finite_exact: true, computed_literals: false },
        Arm { name: "greedy device-subset f32", search: Search::Greedy, mode: LitMode::F32, finite_exact: true, computed_literals: false },
    ];

    let caller = CallerFacts::default();
    let mut w_in = 0u64;
    let mut w_egg = 0u64;
    let mut w_arm = vec![0u64; arms.len()];
    let mut secs = vec![0.0f64; arms.len()];
    let mut smaller_than_egg = vec![0usize; arms.len()];
    let mut larger_than_egg = vec![0usize; arms.len()];
    let mut parse_errors = 0usize;
    let mut over_device = 0usize;
    let mut forms_total = 0usize;
    let mut steps_hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut hits: BTreeMap<usize, u64> = BTreeMap::new();
    let mut finite_divergent = [0usize; 2];
    let mut nonfinite_changed = [0usize; 2];
    let mut examples: Vec<String> = Vec::new();
    let mut shown = 0usize;
    let mut f32_disagrees = 0usize;
    let mut leaks: Vec<(u64, String, String)> = Vec::new();
    let mut status: BTreeMap<&'static str, (usize, u64)> = BTreeMap::new();

    for (i, line) in lines.iter().enumerate() {
        let (math, vars) = line.split_once('\t').expect("math<TAB>vars");
        let inputs: Vec<String> = vars.split(',').filter(|v| !v.is_empty()).map(str::to_string).collect();
        let Ok(tree) = Tree::parse(math) else {
            parse_errors += 1;
            continue;
        };
        let n_in = tree.node_count() as u64;
        w_in += counts[i] * n_in;
        w_egg += counts[i] * egglog[i];
        if tree.node_count() > MAX_NODES {
            over_device += 1;
        }

        let mut bests: Vec<Tree> = Vec::new();
        for (a, arm) in arms.iter().enumerate() {
            let cfg = Config {
                inputs: &inputs,
                caller: &caller,
                mode: arm.mode,
                search: arm.search,
                max_steps: 64,
                finite_exact: arm.finite_exact,
                computed_literals: arm.computed_literals,
            };
            let t = Instant::now();
            let out = run(&tree, &rules, &tables.guards, &cfg);
            secs[a] += t.elapsed().as_secs_f64();
            let n = out.best.node_count() as u64;
            w_arm[a] += counts[i] * n;
            smaller_than_egg[a] += usize::from(n < egglog[i]);
            larger_than_egg[a] += usize::from(n > egglog[i]);
            if a == 0 {
                forms_total += out.forms.len().min(8);
                for (rule, k) in &out.hits {
                    *hits.entry(*rule).or_insert(0) += *k as u64 * counts[i];
                }
                if n > egglog[i] {
                    leaks.push((counts[i] * (n - egglog[i]), math.to_string(), out.best.to_math()));
                }
            }
            if a == 1 {
                *steps_hist.entry(out.steps).or_insert(0) += 1;
            }
            bests.push(out.best);
        }
        if bests[3] != bests[4] {
            f32_disagrees += 1;
        }

        if let Some((psets, head_length)) = &write_back {
            let functions: HashMap<String, FunctionSpec> = psets[i]
                .split(',')
                .filter_map(|f| f.split_once('/'))
                .map(|(id, arity)| {
                    let spec = FunctionSpec { semantic_id: id.to_string(), arity: arity.parse().expect("arity") };
                    (id.to_string(), spec)
                })
                .collect();
            let pset = PsetSpec { variables: inputs.clone(), functions, rnc_values: Vec::new() };
            let tidy = bests[3].to_math();
            let s = if tree.node_count() > MAX_NODES {
                "node_oversize"
            } else if bests[3] != bests[4] {
                "f64_replay_mismatch"
            } else if bests[3] == tree {
                "unchanged"
            } else {
                match terms_to_karva_sized(&tidy, &pset, 0, Some(*head_length)) {
                    Err(_) => "encode_error (function not in the primitive set)",
                    Ok((_, _, true)) => "head_oversize",
                    Ok((h, t, false)) => match karva_to_terms(&h, &t, &pset).ok().and_then(|m| Tree::parse(&m).ok()) {
                        Some(back) if back == bests[3] => "graftable",
                        _ => "encode_error (round trip changed the expression)",
                    },
                }
            };
            let e = status.entry(s).or_insert((0, 0));
            e.0 += 1;
            e.1 += counts[i];
        }

        // Soundness against the INPUT on probe rows: the full rule set (arm 0)
        // and the exact-only set (arm 2), which must never diverge.
        let rows = probe_rows(&inputs);
        let before = Compiled::new(&tree);
        for (arm, slot) in [(0usize, 0usize), (2, 1)] {
            let after = Compiled::new(&bests[arm]);
            for row in &rows {
                let (Ok(x), Ok(y)) = (before.eval(row), after.eval(row)) else {
                    continue;
                };
                let scale = row.iter().fold(0.0_f64, |m, (_, v)| m.max(v.abs()));
                if agree(x, y, scale) {
                    continue;
                }
                if x.is_finite() {
                    finite_divergent[slot] += 1;
                    if slot == 1 || shown < 6 {
                        shown += usize::from(slot == 0);
                        examples.push(format!(
                            "[{}] {x:e} vs {y:e}\n     in  {math}\n     got {}",
                            arms[arm].name,
                            bests[arm].to_math()
                        ));
                    }
                } else {
                    nonfinite_changed[slot] += 1;
                }
                break;
            }
        }
    }

    let n = lines.len() - parse_errors;
    println!("expressions {n} (parse errors {parse_errors}); over {MAX_NODES} nodes: {over_device}");
    println!("weighted nodes  input {w_in}   egglog smallest_form {w_egg}  (removes {})", w_in - w_egg);
    println!();
    println!("{:<46} {:>10} {:>9} {:>8} {:>8} {:>9}", "arm", "weighted", "removes", "<egglog", ">egglog", "ms/expr");
    for (a, arm) in arms.iter().enumerate() {
        println!(
            "{:<46} {:>10} {:>9} {:>8} {:>8} {:>9.3}",
            arm.name,
            w_arm[a],
            w_in - w_arm[a],
            smaller_than_egg[a],
            larger_than_egg[a],
            1e3 * secs[a] / n as f64
        );
    }
    println!();
    println!("coverage of egglog's reduction: beam8 {:.1}%  greedy {:.1}%  device-subset greedy {:.1}%",
        100.0 * (w_in - w_arm[0]) as f64 / (w_in - w_egg) as f64,
        100.0 * (w_in - w_arm[1]) as f64 / (w_in - w_egg) as f64,
        100.0 * (w_in - w_arm[3]) as f64 / (w_in - w_egg) as f64);
    println!("variety: {:.2} distinct forms per expression (beam8, capped at 8)", forms_total as f64 / n as f64);
    println!("f32 vs f64 device-subset greedy results differ on {f32_disagrees} expressions");
    println!("soundness vs input on {N_PROBE_ROWS} probe rows (expressions whose value changes):");
    println!("  all rules        : input finite {}, input non-finite {}", finite_divergent[0], nonfinite_changed[0]);
    println!("  exact-only rules : input finite {}, input non-finite {}", finite_divergent[1], nonfinite_changed[1]);
    for e in &examples {
        println!("  DIVERGES {e}");
    }
    if write_back.is_some() {
        println!("write-back status of the device form (greedy, device subset, f32; replayed in f64):");
        for (name, (n_expr, weight)) in &status {
            println!("  {name:<58} {n_expr:>8} expressions {weight:>9} weighted");
        }
    }
    println!("greedy steps per expression: {steps_hist:?}");
    println!("rule hits (beam8, weighted), top 25:");
    let mut ranked: Vec<(u64, usize)> = hits.iter().map(|(r, k)| (*k, *r)).collect();
    ranked.sort_by(|a, b| b.cmp(a));
    for (k, r) in ranked.iter().take(25) {
        println!("  {k:>8}  {}", tables.rules[*r].text);
    }
    println!("rules that never fired: {}", rules.len() - hits.len());
    leaks.sort_by(|a, b| b.0.cmp(&a.0));
    println!("largest leaks against egglog (weighted extra nodes), top 12:");
    for (w, input, got) in leaks.iter().take(12) {
        println!("  +{w}\n     in  {input}\n     got {got}");
    }
}
