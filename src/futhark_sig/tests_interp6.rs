//! Master 6's interpreter, work and static cost (plan section 7, task F3; proof 1's second half, plan 7.3 item 1):
//! the reference programs keep master 5's values, work and static cost; every random gene of every view runs on the
//! interpreter with no run-time error and its per-submit cap is computed, adapters counted; 100 random genes a view
//! compile with Futhark's `c` backend and equal the interpreter, N of N; an adapter costs more than its clean form.

use super::*;
use crate::fitness::Exactness;
use crate::futhark_table::Kingdom6;
use crate::typed_gep::express;
use std::collections::BTreeSet;
use std::process::Command;
use std::time::{Duration, Instant};

/// Every master 6 view (plan 7.3 item 1): the problems' views and the `i32` + `f32` k-means view.
const VIEWS6: [&str; 7] = ["reduce", "scan", "histogram", "pathfinder", "a7", "kmeans", "kmeans_i32"];

/// The problem whose cases, signature and program size a view takes (the `i32` + `f32` k-means view: k-means's).
fn problem_of(view6: &str) -> Problem {
    problem(if view6 == "kmeans_i32" { "kmeans" } else { view6 }).unwrap()
}

/// A view's kingdom and its master 6 interpreter.
fn interp6(view6: &str) -> (Arc<Kingdom6>, Interp) {
    let k = Arc::new(kingdom6(view6).unwrap_or_else(|e| panic!("{view6}: {e}")));
    let sig = signature(&problem_of(view6).instance).unwrap();
    let i = Interp::new6(&k, &sig).unwrap_or_else(|e| panic!("{view6}: {e}"));
    (k, i)
}

/// `n` random genes of a view, expressed, from the random seed `seed` (F2's draw).
fn random_trees(k: &Kingdom6, view6: &str, n: usize, seed: u64) -> Vec<Tree> {
    let contract = k.contract().unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.root.data_type, problem_of(view6).program_size).unwrap();
    let consts = const_len6(k);
    let mut rng = GeneRng::new(seed);
    (0..n)
        .map(|i| {
            let gene = random_gene(&contract, &layout, &consts, &mut rng).unwrap();
            express(&contract, &layout, &gene, &consts).unwrap_or_else(|e| panic!("{view6} gene {i}: {e:?}")).tree()
        })
        .collect()
}

/// The sizes of a case's inputs: every dimension and every integer scalar (the histogram's `k`).
fn case_sizes(case: &Case) -> BTreeSet<i64> {
    let mut sizes = BTreeSet::new();
    for a in &case.inputs {
        if a.dims.is_empty() {
            if a.elem == Elem::I32 {
                sizes.insert(i64::from(a.data[0] as i32));
            }
        } else {
            sizes.extend(a.dims.iter().map(|d| *d as i64));
        }
    }
    sizes
}

/// The cases a view's random programs run on (F2's choice, `futhark_table::tests6::run_cases`): never a test case;
/// the smallest, the smallest whose sizes are as many different values as the view has size names (each at least
/// 2: a `fit` or `clampix` to the wrong size moves data or reads out of bounds there), the smallest training case.
fn run_cases(k: &Kingdom6, view6: &str) -> Vec<Case> {
    run_cases_and_distinct(k, view6).0
}

/// [`run_cases`] and the one of them whose sizes are distinct.
fn run_cases_and_distinct(k: &Kingdom6, view6: &str) -> (Vec<Case>, Case) {
    let mut names: BTreeSet<String> = k.variables.iter().filter(|v| v.kind == "size").map(|v| v.name.clone()).collect();
    for v in k.variables.iter().filter(|v| v.kind == "input") {
        names.extend(v.properties.iter().filter_map(|(_, e)| if let crate::geneframe::PropertyExpr::Size(n) = e { Some(n.clone()) } else { None }));
    }
    let mut cases: Vec<Case> = load_cases(&problem_of(view6)).unwrap().into_iter().filter(|c| c.split != Split::Test).collect();
    cases.sort_by_key(|c| (c.stdin.len(), c.name.clone()));
    let distinct = |c: &Case| {
        let s = case_sizes(c);
        s.len() == names.len() && s.iter().all(|x| *x >= 2)
    };
    let picks = [cases.first(), cases.iter().find(|c| distinct(c)), cases.iter().find(|c| c.split == Split::Train)];
    let distinct_case = picks[1].expect("a view has a case of distinct sizes").clone();
    let mut out: Vec<Case> = Vec::new();
    for c in picks.into_iter().map(|c| c.expect("a view has a case of each kind")) {
        if !out.iter().any(|o| o.name == c.name) {
            out.push(c.clone());
        }
    }
    (out, distinct_case)
}

/// A stored case CROPPED to small sizes, for a gene whose work on the full case is over the per-submit cap
/// ([`CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT`], which phylu never runs past): each size of the signature a distinct
/// small length (7, 5, 3, 2 in the signature's order, none above its own), each input its leading rows and columns, a
/// size input its cropped value. The case's own values, at sizes the interpreter runs every gene at in seconds.
fn cropped(view6: &str, case: &Case) -> (Vec<Array>, Vec<u8>) {
    let sig = signature(&problem_of(view6).instance).unwrap();
    let full = sig.sizes(&case.inputs).unwrap();
    let small: BTreeMap<&String, usize> = sig.dims.iter().zip([7usize, 5, 3, 2, 11, 13]).map(|(d, s)| (d, s.min(full[d]))).collect();
    let inputs: Vec<Array> = sig
        .inputs
        .iter()
        .zip(&case.inputs)
        .map(|(p, a)| {
            if sig.size_inputs.contains(&p.name) {
                return Array { dims: a.dims.clone(), elem: a.elem, data: vec![small[&p.name] as u32] };
            }
            let dims: Vec<usize> = p.dims.iter().map(|d| small[d]).collect();
            let data = match (&a.dims[..], &dims[..]) {
                ([_], [l]) => a.data[..*l].to_vec(),
                ([_, c], [r2, c2]) => (0..*r2).flat_map(|i| a.data[i * c..i * c + c2].to_vec()).collect(),
                _ => panic!("{view6}: an input of rank {}", a.dims.len()),
            };
            Array { dims, elem: a.elem, data }
        })
        .collect();
    let stdin = write_binary(&inputs.iter().collect::<Vec<_>>());
    (inputs, stdin)
}

/// THE REFERENCE PROGRAMS ARE UNCHANGED (plan 7.3 item 2, F3's half): every reference, read as its master 6 tree, has
/// on every stored case the value master 5's interpreter gives its master 5 tree (bit for bit, two NaNs equal), the
/// same work (both parts; k-means on its two smallest cases), and, where master 5 has one (A7, the matrix kingdom), the same static cost; every master 6
/// reference has a static cost, adapter-free.
#[test]
fn every_reference_has_its_master_5_value_work_and_static_cost() {
    let mut report = Vec::new();
    for r in &REFERENCES6 {
        let key = r.key();
        let p = problem(r.problem).unwrap();
        let (k5, sig) = (load(&p.instance).unwrap(), signature(&p.instance).unwrap());
        let i5 = Interp::new(&k5, &sig).unwrap();
        let t5 = build(&k5, &(r.master5)()).unwrap();
        let (k6, i6) = interp6(r.problem);
        let t6 = build6(&k6, &(r.master6)()).unwrap_or_else(|e| panic!("{key}: {e}"));
        // Every stored case; k-means's (n up to 1000 points of d = 1536) its two smallest, the interpreters' time.
        let mut cases = load_cases(&p).unwrap();
        if r.problem == "kmeans" {
            cases.sort_by_key(|c| (c.stdin.len(), c.name.clone()));
            cases.truncate(2);
        }
        let mut costed = 0;
        for c in &cases {
            let t = Instant::now();
            let v5 = i5.eval(&t5, &c.inputs).unwrap();
            let t6s = Instant::now();
            let v6 = i6.eval(&t6, &c.inputs).unwrap_or_else(|e| panic!("{key} on {}: {e}", c.name));
            let six = t6s.elapsed().as_secs_f64();
            assert_eq!(v6, v5, "{key} on {}: master 6's value differs from master 5's", c.name);
            let sizes = sig.sizes(&c.inputs).unwrap();
            assert_eq!(i6.work_parts(&t6, &sizes).unwrap(), i5.work_parts(&t5, &sizes).unwrap(), "{key} on {}: the work differs", c.name);
            let s6 = i6.static_cost(&t6, &sizes).unwrap().expect("master 6 costs every view");
            if let Some(s5) = i5.static_cost(&t5, &sizes).unwrap() {
                assert_eq!(s6, s5, "{key} on {}: the static cost differs", c.name);
                costed += 1;
            }
            eprintln!("{key} on {}: {:.2} s, master 6's interpreter {six:.2} s", c.name, t.elapsed().as_secs_f64());
        }
        report.push(format!("{key}: {} cases equal (value, work){}", cases.len(), if costed > 0 { format!(", static cost equal on {costed}") } else { String::new() }));
    }
    eprintln!("{}", report.join("\n"));
}

/// PROOF 1, THE INTERPRETER (plan 7.3 item 1): 1,000 random genes of every master 6 view run on the interpreter on
/// one stored case (the smallest whose sizes all differ; a gene over the per-submit cap on it, which phylu never runs,
/// on it [`cropped`]: k-means's d is 1536) with no run-time error: every index in range, every size agreeing, every
/// value's shape its properties'. And THE PER-SUBMIT CAP of every one is computed from every stored
/// case's sizes with its adapters counted ([`Work::nodes`], what phylu's `submit_work_of` reads): a gene's nodes are
/// at least its adapters' charge, so no adapter escapes the cap.
#[test]
fn a_thousand_random_genes_a_view_run_on_the_interpreter_and_have_a_work_bound() {
    let n = 1000;
    let mut report = String::new();
    for (v, view6) in VIEWS6.iter().enumerate() {
        let started = Instant::now();
        let (k, i6) = interp6(view6);
        let renderer = k.renderer6(crate::futhark_table::Header6::of_spec(&crate::futhark_view::master6::committed(view6).unwrap().spec)).unwrap();
        let case = run_cases_and_distinct(&k, view6).1;
        let case_sizes_of = signature(&problem_of(view6).instance).unwrap().sizes(&case.inputs).unwrap();
        let small = cropped(view6, &case);
        let mut cropped_runs = 0usize;
        let all: Vec<BTreeMap<String, usize>> = load_cases(&problem_of(view6)).unwrap().iter().map(|c| signature(&problem_of(view6).instance).unwrap().sizes(&c.inputs).unwrap()).collect();
        let (mut adapted, mut most, mut charged) = (0usize, 0u128, 0u128);
        for (g, tree) in random_trees(&k, view6, n, 0x5eed_0000 + v as u64).iter().enumerate() {
            let program = renderer.program(tree).unwrap();
            let over = i6.work_parts(tree, &case_sizes_of).unwrap().nodes > CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT;
            let inputs = if over { &small.0 } else { &case.inputs };
            cropped_runs += usize::from(over);
            i6.eval(tree, inputs).unwrap_or_else(|e| panic!("{view6} gene {g} on {}{}: {e}\n{}", case.name, if over { " cropped" } else { "" }, program.text));
            let d = renderer.decode(tree).unwrap();
            for s in &all {
                let w = i6.work_parts(tree, s).unwrap_or_else(|e| panic!("{view6} gene {g}: the work: {e}"));
                most = most.max(w.nodes);
                if !program.adapter_free() {
                    let charge = adapters_charge(&k, &d, s);
                    assert!(w.nodes >= charge, "{view6} gene {g}: nodes {} do not count its adapters' {charge}\n{}", w.nodes, program.text);
                    charged += charge;
                }
            }
            adapted += usize::from(!program.adapter_free());
        }
        assert!(adapted == 0 || charged > 0, "{view6}: {adapted} genes render adapters and none is charged");
        let line = format!("{view6}: {n} of {n} ran on the interpreter on {} with no error ({cropped_runs} over the per-submit cap on it ran on it cropped), {adapted} with adapters; the per-submit work computed on {} cases, the most {most} node executions, adapters charged {charged} in all ({:.1} s)\n", case.name, all.len(), started.elapsed().as_secs_f64());
        eprint!("{line}");
        report.push_str(&line);
    }
    eprint!("{report}");
}

/// The adapters' charge of a decoded tree at `sizes` (the node adapters at their node's trips, the body adapters at
/// their body's, `fit.root` once), recomputed apart from `work_parts` from the same rendering decision
/// (`futhark_cost::adapters6`).
fn adapters_charge(k: &Kingdom6, d: &crate::typed_environment::Decoding, sizes: &BTreeMap<String, usize>) -> u128 {
    use crate::futhark_cost::{adapters6, rendered6, requirements6, trips6, CostTable6};
    let table = CostTable6::load(k).unwrap();
    let reqs = requirements6(k);
    fn go(k: &Kingdom6, table: &CostTable6, reqs: &[&crate::geneframe::Requirement], n: &crate::typed_environment::Node, mult: u128, sizes: &BTreeMap<String, usize>) -> u128 {
        let n = rendered6(n);
        if n.substituted.is_some() {
            return 0;
        }
        let trips = trips6(k, n, sizes).unwrap();
        let mut ops: Vec<_> = n.kids.iter().map(|c| rendered6(c).properties.clone()).collect();
        let mut total = 0;
        for (body, by) in [(false, mult), (true, mult * trips.unwrap_or(1))] {
            for c in adapters6(k, reqs, n, body, &mut ops).unwrap() {
                total += table.charge(k, &c, sizes).unwrap() * by;
            }
        }
        let trip = k.symbols[n.symbol as usize].trip;
        for (i, c) in n.kids.iter().enumerate() {
            total += go(k, table, reqs, c, if trip == Some(i as u32) { mult * trips.unwrap_or(1) } else { mult }, sizes);
        }
        total
    }
    let mut total = go(k, &table, &reqs, &d.root, 1, sizes);
    let mut root = rendered6(&d.root).properties.clone();
    for o in d.root_outcomes.iter().filter(|o| !o.holds) {
        root.insert(k.root.requires[o.requirement].property.clone(), o.against.clone());
        total += table.charge(k, &crate::futhark_cost::Charge6 { adapter: o.adapter.clone(), operand: None, fitted: Some((k.root.data_type, root.clone())) }, sizes).unwrap();
    }
    total
}

/// A program as a module of a batch (`module m<i> = { .. }`, its entry a `def`) and the entry `e<i>` of the program's
/// own header calling it.
fn with_entry(i: usize, text: &str) -> String {
    let start = text.find("entry main ").expect("a program has its entry");
    let line = &text[start..start + text[start..].find(" =\n").expect("the entry's header ends")];
    let names: Vec<&str> = line.split('(').skip(1).filter_map(|p| p.split_once(':').map(|(n, _)| n.trim())).collect();
    format!("module m{i} = {{\n{}}}\n\n{} = m{i}.main {}\n\n", text.replacen("entry main ", "def main ", 1), line.replacen("entry main ", &format!("entry e{i} "), 1), names.join(" "))
}

/// One program's run on a case's stdin: its output, or the failure.
fn run_entry(exe: &std::path::Path, entry: &str, stdin: &[u8]) -> Result<Array, String> {
    let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(exe).arg("-e").arg(entry).arg("-b"), Duration::from_secs(120), Some(stdin))?;
    if !out.status.success() {
        return Err(format!("exit {}: {}", out.status, String::from_utf8_lossy(&out.stderr)));
    }
    read_one(&out.stdout)
}

/// FUTHARK'S `0 / x` FOLD: Futhark 0.28.1's simplifier folds a division whose dividend it knows to be zero to zero
/// (measured 2026-11-05 through `futhark c`: `0f32 / x`, `map (/ s) (replicate n 0f32)`, `(replicate n 0f32)[0] / s`
/// and `map (/ s) (map (\_ -> 0.0) xs)` are 0 for a zero or NaN `x` or `s`), where IEEE, and the interpreter, give
/// NaN; whether it knows depends on its fusion (`map2 (\r s -> map (/ s) r) (map (\xs -> map (\_ -> 0.0) xs) A) S`
/// alone is NaN, under a further `map (*2.0)` 0), so the interpreter does not guess it. An output is equal but for the
/// fold when the program divides and every cell is the interpreter's or a zero where the interpreter's is NaN.
fn zero_fold(got: &Array, want: &Array, text: &str) -> bool {
    (text.contains(" / ") || text.contains("(/ "))
        && got.dims == want.dims
        && got.elem == Elem::F32
        && want.elem == Elem::F32
        && got.data.iter().zip(&want.data).all(|(g, w)| same_value(Elem::F32, *g, *w) || (f32::from_bits(*w).is_nan() && f32::from_bits(*g) == 0.0))
}

/// PROOF 1, THE `c` BACKEND AGAINST THE INTERPRETER (plan 7.3 item 1): 100 random genes of every master 6 view
/// compile with Futhark's `c` backend and run on the view's run cases ([`run_cases`]: up to three stored cases, one of
/// distinct sizes, none a test case; a gene over the per-submit cap on a case on it [`cropped`]), and every output
/// equals the interpreter's, N of N: BIT FOR BIT (two NaNs equal),
/// the rule of master 5's interpreter against the `c` backend; where an `f32` output differs in its bits it must be
/// within the A7 f32 bound (`fitness::Exactness::F32SumBound`, two products a cell), or equal but for Futhark's
/// `0 / x` fold ([`zero_fold`]), and such outputs are counted apart.
#[test]
fn a_hundred_random_genes_a_view_compile_on_c_and_equal_the_interpreter() {
    let (n, chunk) = (100, 25);
    let mut report = String::new();
    for (v, view6) in VIEWS6.iter().enumerate() {
        let (k, i6) = interp6(view6);
        let renderer = k.renderer6(crate::futhark_table::Header6::of_spec(&crate::futhark_view::master6::committed(view6).unwrap().spec)).unwrap();
        let trees = random_trees(&k, view6, n, 0xc0de_0000 + v as u64);
        let texts: Vec<String> = trees.iter().map(|t| renderer.program(t).unwrap().text).collect();
        let cases = run_cases(&k, view6);
        let dir = std::env::temp_dir().join(format!("fuller_f3_{view6}_{}", std::process::id()));
        let (mut exact, mut bounded, mut folded, mut adapted, mut cropped_runs) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut bad: Vec<String> = Vec::new();
        for (c, part) in texts.chunks(chunk).enumerate() {
            let base = c * chunk;
            let src: String = part.iter().enumerate().map(|(i, t)| with_entry(base + i, t)).collect();
            let exe = compile_c(&src, &dir.join(format!("c{c}"))).unwrap_or_else(|e| panic!("{view6} genes {base}..: futhark c: {e}"));
            for (i, text) in part.iter().enumerate() {
                let g = base + i;
                adapted += usize::from(text.contains("def fit") || text.contains("def clampix"));
                for case in &cases {
                    let over = i6.work_parts(&trees[g], &signature(&problem_of(view6).instance).unwrap().sizes(&case.inputs).unwrap()).unwrap().nodes > CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT;
                    let small;
                    let (inputs, stdin) = if over {
                        small = cropped(view6, case);
                        cropped_runs += 1;
                        (&small.0, &small.1)
                    } else {
                        (&case.inputs, &case.stdin)
                    };
                    let want = match i6.eval(&trees[g], inputs) {
                        Ok(a) => a,
                        Err(e) => {
                            bad.push(format!("{view6} gene {g} on {}: the interpreter: {e}\n{text}", case.name));
                            continue;
                        }
                    };
                    match run_entry(&exe, &format!("e{g}"), stdin) {
                        Ok(got) if got == want => exact += 1,
                        Ok(got) if zero_fold(&got, &want, text) => folded += 1,
                        Ok(got) if got.elem == Elem::F32 && Exactness::F32SumBound { products: 2 }.agrees(&got, &want, inputs) => bounded += 1,
                        Ok(got) => bad.push(format!("{view6} gene {g} on {}: c gives {:?} {:?}, the interpreter {:?} {:?}\n{text}", case.name, got.dims, got.values().take(8).collect::<Vec<_>>(), want.dims, want.values().take(8).collect::<Vec<_>>())),
                        Err(e) => bad.push(format!("{view6} gene {g} on {}: {e}\n{text}", case.name)),
                    }
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
        let names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
        let line = format!("{view6}: {n} compiled (c), {adapted} with adapters; on {names:?}: {exact} of {} runs equal the interpreter bit for bit, {bounded} within the A7 f32 bound, {folded} equal but for Futhark's 0 / x fold ({cropped_runs} runs over the per-submit cap on their case, on it cropped)\n", n * cases.len());
        eprint!("{line}");
        report.push_str(&line);
        assert!(bad.is_empty(), "{report}{}", bad.iter().take(5).cloned().collect::<Vec<_>>().join("\n"));
    }
}

/// AN ADAPTER COSTS MORE THAN ITS CLEAN FORM (the Fable review's guard, plan 7.5 risk 4): each gene with a failing
/// requirement, against the same gene whose variable names a size or an array that satisfies it, at sizes where every
/// size name has one value (so the adapter is needless and the two compute the same): its static cost's traffic and
/// its work's node executions are both strictly more, by the adapter's charge.
#[test]
fn a_gene_with_an_adapter_costs_more_than_the_same_gene_without() {
    let pairs: [(&str, &str, &str, &str); 5] = [
        ("kmeans", "fit.pair", "(let.a.f32@m.f32[w] (map2.add@a.f32 zeros_n zeros_k) centres)", "(let.a.f32@m.f32[w] (map2.add@a.f32 zeros_n zeros_n) centres)"),
        ("kmeans", "fit.body", "(tabrows@m.f32[r] k d zeros_n)", "(tabrows@m.f32[r] k d zeros_d)"),
        ("kmeans", "clampix", "(let.a.f32@m.f32[w] (replicate@a.f32 n (get@f32 zeros_n (last@i64 k))) centres)", "(let.a.f32@m.f32[w] (replicate@a.f32 n (get@f32 zeros_n (last@i64 n))) centres)"),
        ("kmeans", "fit.state", "(let.a.f32@m.f32[w] (loop@a.f32[s,i] n zeros_k (map2.add@a.f32 zeros_n zeros_n)) centres)", "(let.a.f32@m.f32[w] (loop@a.f32[s,i] n zeros_n (map2.add@a.f32 zeros_n zeros_n)) centres)"),
        ("histogram", "fit.pair", "(hist.add@a.i32 k zeros_n zeros_k)", "(hist.add@a.i32 k zeros_n zeros_n)"),
    ];
    for (view6, adapter, with, without) in pairs {
        let (k, i6) = interp6(view6);
        let renderer = k.renderer6(crate::futhark_table::Header6::of_spec(&crate::futhark_view::master6::committed(view6).unwrap().spec)).unwrap();
        let (a, b) = (build6(&k, with).unwrap(), build6(&k, without).unwrap());
        let (pa, pb) = (renderer.program(&a).unwrap(), renderer.program(&b).unwrap());
        assert!(pa.adapters.iter().any(|(_, x)| x == adapter) && pb.adapter_free(), "{with}: {:?}; {without}: {:?}", pa.adapters, pb.adapters);
        let sizes: BTreeMap<String, usize> = k.variables.iter().filter(|v| v.kind == "size").map(|v| (v.name.clone(), 4)).collect();
        let (ca, cb) = (i6.static_cost(&a, &sizes).unwrap().unwrap(), i6.static_cost(&b, &sizes).unwrap().unwrap());
        let (wa, wb) = (i6.work_parts(&a, &sizes).unwrap(), i6.work_parts(&b, &sizes).unwrap());
        assert!(ca.traffic > cb.traffic && wa.nodes > wb.nodes, "{adapter}: with {ca:?} {wa:?}, without {cb:?} {wb:?}");
        eprintln!("{view6} {adapter}: traffic {} against {}, node executions {} against {}", ca.traffic, cb.traffic, wa.nodes, wb.nodes);
    }
}

/// The fold's allowance is narrow: a zero where the interpreter has NaN, in a program that divides; never another
/// value, never in a program with no division.
#[test]
fn the_zero_fold_allowance_takes_only_a_zero_for_a_nan_where_the_program_divides() {
    let want = Array::f32s(vec![2], vec![f32::NAN, 1.0]);
    let zero = Array::f32s(vec![2], vec![0.0, 1.0]);
    let other = Array::f32s(vec![2], vec![3.0, 1.0]);
    assert!(zero_fold(&zero, &want, "map (/ s) r"));
    assert!(!zero_fold(&zero, &want, "map (*2.0) r"));
    assert!(!zero_fold(&other, &want, "map (/ s) r"));
    assert!(!zero_fold(&want, &zero, "map (/ s) r"), "a NaN where the interpreter has a number is never the fold");
}
