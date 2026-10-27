//! THE QUALITY INPUTS, MEASURED (host only, no GPU): do the signature family's quality inputs give partial
//! correctness a slope? N random genes of a problem (the runner's layout and head draw) are evaluated by the host
//! interpreter on every stored case, and scored by the current inputs and by the proposed ones
//! (`fuller::futhark_sig::quality_inputs`); probe outputs (all zero, right on half the cells, the answer) are scored
//! beside them. Reports the quality angle's p10, median and p90, the best, and the probes.
//!
//!   cargo run --release --example futhark_sig_quality_study -- --problem pathfinder --n 1000

use fuller::futhark_kingdom::GeneRng;
use fuller::futhark_sig::{build, compare, load, load_cases, problem, quality_inputs, random_gene, signature, Array, Interp};
use fuller::runner::{log_scaled, staged_hff, FitnessStage};
use fuller::typed_gep::{express, GeneLayout};

fn angle(q: &[f64]) -> f64 {
    staged_hff(FitnessStage::Quality, q, &[]).map(|t| t.quality).unwrap_or(f64::NAN)
}

fn current(got: &Array, want: &Array) -> Vec<f64> {
    compare(got, want).iter().map(|x| log_scaled(*x)).collect()
}

fn pct(v: &[f64], p: f64) -> f64 {
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let get = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let pname = get("--problem").unwrap_or_else(|| "pathfinder".into());
    let n: usize = get("--n").map_or(Ok(1000), |v| v.parse().map_err(|_| "--n".to_string()))?;
    let p = problem(&pname)?;
    let k = load(&p.instance)?;
    let sig = signature(&p.instance)?;
    let interp = Interp::new(&k, &sig)?;
    let contract = k.relations().check().map_err(|d| format!("{:?}", &d[..d.len().min(3)]))?;
    let reference = build(&k, p.reference)?;
    let n_funcs = reference.level_order().iter().filter(|(t, _)| !k.rows[t.symbol as usize].inputs.is_empty()).count();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], n_funcs.max(8)).map_err(|e| e.to_string())?;
    let const_len = k.const_len();
    let cases = load_cases(&p)?;
    let score = |outs: &[Array]| -> (f64, f64) {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for (o, c) in outs.iter().zip(&cases) {
            a.extend(current(o, &c.output));
            b.extend(quality_inputs(o, &c.output));
        }
        (angle(&a), angle(&b))
    };
    // The probes.
    let probe = |f: &dyn Fn(&Array) -> Array| -> (f64, f64) { score(&cases.iter().map(|c| f(&c.output)).collect::<Vec<_>>()) };
    let zero = probe(&|w| Array { dims: w.dims.clone(), data: vec![0; w.data.len()] });
    let half = probe(&|w| Array { dims: w.dims.clone(), data: w.data.iter().enumerate().map(|(i, x)| if i % 2 == 0 { *x } else { 0 }).collect() });
    let near = probe(&|w| Array { dims: w.dims.clone(), data: w.data.iter().map(|x| x.wrapping_add(1)).collect() });
    let exact = probe(&|w| w.clone());
    println!("{pname}: {n} random genes, program size {}, {} cases", n_funcs.max(8), cases.len());
    println!("probe                         current   proposed");
    for (name, (a, b)) in [("all zero", zero), ("right on half the cells", half), ("every cell off by 1", near), ("the answer", exact)] {
        println!("{name:<28}  {a:.5}   {b:.5}");
    }
    let mut rng = GeneRng::new(0x0051_7DE5);
    let (mut cur, mut pro) = (Vec::new(), Vec::new());
    let mut best = (f64::INFINITY, String::new());
    for _ in 0..n {
        let gene = random_gene(&contract, &layout, &const_len, &mut rng)?;
        let tree = express(&contract, &layout, &gene, &const_len).map_err(|e| e.to_string())?.tree();
        let outs = cases.iter().map(|c| interp.eval(&tree, &c.inputs)).collect::<Result<Vec<_>, _>>()?;
        let (a, b) = score(&outs);
        if b < best.0 {
            best = (b, fuller::futhark_sig::sexp(&k, &tree));
        }
        cur.push(a);
        pro.push(b);
    }
    for v in [&mut cur, &mut pro] {
        v.sort_by(|x, y| x.total_cmp(y));
    }
    let distinct = |v: &[f64]| {
        let mut d: Vec<u64> = v.iter().map(|x| x.to_bits()).collect();
        d.dedup();
        d.len()
    };
    println!("random programs              p10       median    p90       best      distinct angles");
    println!("current                      {:.5}   {:.5}   {:.5}   {:.5}   {}", pct(&cur, 0.1), pct(&cur, 0.5), pct(&cur, 0.9), cur[0], distinct(&cur));
    println!("proposed                     {:.5}   {:.5}   {:.5}   {:.5}   {}", pct(&pro, 0.1), pct(&pro, 0.5), pct(&pro, 0.9), pro[0], distinct(&pro));
    println!("best random program (proposed): {}", best.1);
    Ok(())
}
