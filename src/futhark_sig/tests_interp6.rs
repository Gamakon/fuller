//! Master 6's interpreter, work and static cost (plan section 7, task F3; proof 1's second half, plan 7.3 item 1):
//! the reference programs keep master 5's values, work and static cost; every random gene of every view runs on the
//! interpreter with no run-time error and its per-submit cap is computed, adapters counted; 100 random genes a view
//! compile with Futhark's `c` backend and equal the interpreter, N of N; an adapter costs more than its clean form.

use super::*;
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
    adapter_charges(k, d, sizes).iter().map(|(_, own, by)| own * by).sum()
}

/// Every adapter a decoded tree renders at `sizes`: what it adapts, its own charge, and how many times it runs.
fn adapter_charges(k: &Kingdom6, d: &crate::typed_environment::Decoding, sizes: &BTreeMap<String, usize>) -> Vec<(crate::futhark_cost::Charge6, u128, u128)> {
    use crate::futhark_cost::{adapters6, rendered6, requirements6, trips6, Charge6, CostTable6};
    let table = CostTable6::load(k).unwrap();
    let reqs = requirements6(k);
    type Found = Vec<(Charge6, u128, u128)>;
    fn go(k: &Kingdom6, table: &CostTable6, reqs: &[&crate::futhark_table::Requirement6], n: &crate::typed_environment::Node, mult: u128, sizes: &BTreeMap<String, usize>, out: &mut Found) {
        let n = rendered6(n);
        if n.substituted.is_some() {
            return;
        }
        let trips = trips6(k, n, sizes).unwrap();
        let mut ops: Vec<_> = n.kids.iter().map(|c| rendered6(c).properties.clone()).collect();
        for (body, by) in [(false, mult), (true, mult * trips.unwrap_or(1))] {
            for c in adapters6(k, reqs, n, body, &mut ops).unwrap() {
                let own = table.charge(k, &c, sizes).unwrap();
                out.push((c, own, by));
            }
        }
        let trip = k.symbols[n.symbol as usize].trip;
        for (i, c) in n.kids.iter().enumerate() {
            go(k, table, reqs, c, if trip == Some(i as u32) { mult * trips.unwrap_or(1) } else { mult }, sizes, out);
        }
    }
    let mut out = Vec::new();
    go(k, &table, &reqs, &d.root, 1, sizes, &mut out);
    let mut root = rendered6(&d.root).properties.clone();
    for o in d.root_outcomes.iter().filter(|o| !o.holds) {
        let source = Some((k.root.data_type, root.clone()));
        root.insert(k.root.requires[o.requirement].property.clone(), o.against().clone());
        let c = Charge6 { adapter: o.adapter_name().to_string(), operand: None, fitted: Some((k.root.data_type, root.clone())), source };
        let own = table.charge(k, &c, sizes).unwrap();
        out.push((c, own, 1));
    }
    out
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

/// FUTHARK'S ZERO-DIVIDEND FOLD (`docs/issues/2026-11-05_futhark_zero_division_fold.md`): Futhark 0.28.1's simplifier
/// folds a division whose dividend it sees to be a zero to that zero (measured 2026-11-05 through `futhark c`:
/// `0f32 / x` is `0` for `x` = 0, -3, NaN and inf, and `map (/ s) (replicate n 0f32)` is zeros for `s` = 0), where
/// IEEE, `futhark run` and the interpreter give NaN for `0 / 0` and `-0.0` for `0 / -3`. Whether it sees the zero
/// depends on the program (a zero carried through a loop is not folded), so the interpreter does not guess it: it runs
/// the tree twice, with IEEE's division and with every zero dividend folded
/// ([`interp6::Interp6::eval_zero_dividends_folded`]). An output is equal but for the fold when every cell is the
/// interpreter's, or is the folded run's at a cell where the two runs differ: a cell a zero dividend reached. Never a
/// NaN of another origin (a `sqrt` of a negative number, `inf - inf`), never a cell no zero dividend reached.
fn zero_fold(got: &Array, want: &Array, folded: &Array) -> bool {
    let e = want.elem;
    got.dims == want.dims
        && folded.dims == want.dims
        && got.elem == e
        && folded.elem == e
        && got.data.len() == want.data.len()
        && folded.data.len() == want.data.len()
        && got.data.iter().zip(&want.data).zip(&folded.data).all(|((g, w), f)| same_value(e, *g, *w) || (!same_value(e, *w, *f) && same_value(e, *g, *f)))
}

/// How the `c` backend's output agrees with the interpreter's.
#[derive(Debug, PartialEq, Eq)]
enum Agreement {
    /// Bit for bit (two NaNs equal).
    Exact,
    /// Equal but for Futhark's zero-dividend fold ([`zero_fold`]).
    Folded,
    /// Any other difference, however small: the exact paths have no tolerance.
    Differs,
}

/// The agreement of `got` (the `c` backend's) with `want` (the interpreter's), the folded run computed only when the
/// two differ.
fn agreement(got: &Array, want: &Array, folded: impl FnOnce() -> Result<Array, String>) -> Result<Agreement, String> {
    if got == want {
        return Ok(Agreement::Exact);
    }
    Ok(if zero_fold(got, want, &folded()?) { Agreement::Folded } else { Agreement::Differs })
}

/// The interpreter's value of a tree with Futhark's zero-dividend fold.
fn eval_folded(i: &Interp, tree: &Tree, inputs: &[Array]) -> Result<Array, String> {
    match &i.0 {
        Masters::Six(m) => m.eval_zero_dividends_folded(tree, inputs),
        Masters::Five(_) => Err("the zero-dividend fold is master 6's".into()),
    }
}

/// PROOF 1, THE `c` BACKEND AGAINST THE INTERPRETER (plan 7.3 item 1): 100 random genes of every master 6 view
/// compile with Futhark's `c` backend and run on the view's run cases ([`run_cases`]: up to three stored cases, one of
/// distinct sizes, none a test case; a gene over the per-submit cap on a case on it [`cropped`]), and every output
/// equals the interpreter's, N of N: BIT FOR BIT (two NaNs equal), with no tolerance, the rule of master 5's
/// interpreter against the `c` backend; the one allowance is Futhark's zero-dividend fold at the cells a zero dividend
/// reached ([`zero_fold`]), and such outputs are counted apart.
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
        let (mut exact, mut folded, mut adapted, mut cropped_runs) = (0usize, 0usize, 0usize, 0usize);
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
                    match run_entry(&exe, &format!("e{g}"), stdin).and_then(|got| agreement(&got, &want, || eval_folded(&i6, &trees[g], inputs)).map(|a| (got, a))) {
                        Ok((_, Agreement::Exact)) => exact += 1,
                        Ok((_, Agreement::Folded)) => folded += 1,
                        Ok((got, Agreement::Differs)) => bad.push(format!("{view6} gene {g} on {}: c gives {:?} {:?}, the interpreter {:?} {:?}\n{text}", case.name, got.dims, got.values().take(8).collect::<Vec<_>>(), want.dims, want.values().take(8).collect::<Vec<_>>())),
                        Err(e) => bad.push(format!("{view6} gene {g} on {}: {e}\n{text}", case.name)),
                    }
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
        let names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
        let line = format!("{view6}: {n} compiled (c), {adapted} with adapters; on {names:?}: {exact} of {} runs equal the interpreter bit for bit, {folded} equal but for Futhark's zero-dividend fold ({cropped_runs} runs over the per-submit cap on their case, on it cropped)\n", n * cases.len());
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

/// Genes that render adapters (every `fit` kind the costs reach, `clampix`), by view.
const ADAPTED_GENES: [(&str, &str); 5] = [
    ("kmeans", "(let.a.f32@m.f32[w] (map2.add@a.f32 zeros_n zeros_k) centres)"),
    ("kmeans", "(tabrows@m.f32[r] k d zeros_n)"),
    ("kmeans", "(let.a.f32@m.f32[w] (replicate@a.f32 n (get@f32 zeros_n (last@i64 k))) centres)"),
    ("kmeans", "(let.a.f32@m.f32[w] (loop@a.f32[s,i] n zeros_k (map2.add@a.f32 zeros_n zeros_n)) centres)"),
    ("histogram", "(hist.add@a.i32 k zeros_n zeros_k)"),
];

/// Four size settings, the view's size variables in order: each `fit` of [`ADAPTED_GENES`] shorter, equal and longer.
const SIZE_SETTINGS: [[usize; 3]; 4] = [[1000, 10, 3], [4, 4, 4], [2, 7, 5], [5, 3, 9]];

/// AN ADAPTER'S OWN CHARGE COVERS ITS FULL TRAFFIC (F3 review finding 3): every adapter of the adapted genes above,
/// at several size settings that make each `fit` shorter, equal and longer than its operand, is charged more than 0
/// and at least the cells it reads of its operand plus the cells it writes, so the cost never rewards an adapter for
/// its own sake. What a shorter array saves DOWNSTREAM is real work saved, and is pinned as a known property: at
/// n = 1000, k = 10, `map2.add zeros_n zeros_k` (both fitted to 10) costs less than `map2.add zeros_n zeros_n`.
#[test]
fn an_adapters_own_charge_covers_its_full_traffic_at_every_size() {
    let (genes, settings) = (ADAPTED_GENES, SIZE_SETTINGS);
    let (mut shorter, mut equal, mut longer, mut other) = (0usize, 0usize, 0usize, 0usize);
    for (view6, text) in genes {
        let (k, _) = interp6(view6);
        let renderer = k.renderer6(crate::futhark_table::Header6::of_spec(&crate::futhark_view::master6::committed(view6).unwrap().spec)).unwrap();
        let d = renderer.decode(&build6(&k, text).unwrap()).unwrap();
        let names: Vec<String> = k.variables.iter().filter(|v| v.kind == "size").map(|v| v.name.clone()).collect();
        for values in settings {
            let sizes: BTreeMap<String, usize> = names.iter().cloned().zip(values).collect();
            let charges = adapter_charges(&k, &d, &sizes);
            assert!(!charges.is_empty(), "{text}: no adapter renders");
            for (c, own, _) in charges {
                assert!(own > 0, "{text} at {sizes:?}: {} is charged nothing", c.adapter);
                let Some(((ts, ps), (tf, pf))) = c.source.as_ref().zip(c.fitted.as_ref()) else {
                    other += 1;
                    continue;
                };
                let (read, write) = (crate::futhark_cost::cells6(&k, *ts, ps, &sizes).unwrap(), crate::futhark_cost::cells6(&k, *tf, pf, &sizes).unwrap());
                assert!(own >= read + write, "{text} at {sizes:?}: {} is charged {own}, reads {read} and writes {write}", c.adapter);
                match write.cmp(&read) {
                    std::cmp::Ordering::Less => shorter += 1,
                    std::cmp::Ordering::Equal => equal += 1,
                    std::cmp::Ordering::Greater => longer += 1,
                }
            }
        }
    }
    assert!(shorter > 0 && equal > 0 && longer > 0 && other > 0, "fits shorter {shorter}, equal {equal}, longer {longer}; other adapters {other}");
    eprintln!("adapters charged their full traffic: fits to a shorter length {shorter}, an equal {equal}, a longer {longer}; other adapters {other}");
    let (k, i6) = interp6("kmeans");
    let sizes: BTreeMap<String, usize> = [("n", 1000), ("k", 10), ("d", 3)].into_iter().map(|(a, b)| (a.to_string(), b)).collect();
    let cost = |t: &str| i6.static_cost(&build6(&k, t).unwrap(), &sizes).unwrap().unwrap().traffic;
    let (fitted, clean) = (cost(genes[0].1), cost("(let.a.f32@m.f32[w] (map2.add@a.f32 zeros_n zeros_n) centres)"));
    assert!(fitted < clean, "a shortening fit costs {fitted}, its clean form {clean}");
    eprintln!("a shortening fit's program costs {fitted} against its clean form's {clean} (n = 1000, k = 10): the downstream saving is real");
}

/// THE `c` COMPARISON HAS NO TOLERANCE (F3 review finding 1): an output one unit in the last place from the
/// interpreter's, which the A7 f32 bound would accept, differs.
#[test]
fn an_output_one_unit_in_the_last_place_off_differs() {
    let want = Array::f32s(vec![2], vec![1.0, 2.0]);
    let got = Array::f32s(vec![2], vec![1.0, f32::from_bits(2.0f32.to_bits() + 1)]);
    assert_eq!(agreement(&got, &want, || Ok(want.clone())).unwrap(), Agreement::Differs);
    assert_eq!(agreement(&want, &want, || Err("the folded run is not needed".into())).unwrap(), Agreement::Exact);
}

/// THE ZERO-DIVIDEND FOLD IS ALLOWED ONLY WHERE A ZERO DIVIDEND REACHED THE CELL (F3 review finding 2). Rows of
/// `0, 1, ..` divided by `-0, -1, ..`: the interpreter gives NaN for `0 / -0` and `-0.0` for `0 / -1`, the folded run
/// `0` for both, and Futhark's folded output is allowed (`+0.0` for `-0.0` included). A value at such a cell other
/// than the fold's, a zero for a NaN that a `sqrt` of a negative number gave in a program that divides, and a zero
/// where the interpreter has a number all differ.
#[test]
fn the_zero_fold_is_allowed_only_where_a_zero_dividend_reached_the_cell() {
    let (k, i6) = interp6("kmeans");
    let (n, kk, d) = (2usize, 3usize, 2usize);
    let inputs = [Array::f32s(vec![n, d], vec![0.5; n * d]), Array::f32s(vec![kk, d], vec![0.25; kk * d])];
    let divides = build6(&k, "(rowdiv@m.f32 (tabrows@m.f32[r] k d (tab@a.f32[j] d (ival@f32 j))) (tab@a.f32[i] k (neg@f32 (ival@f32 i))))").unwrap();
    let want = i6.eval(&divides, &inputs).unwrap();
    let folded = eval_folded(&i6, &divides, &inputs).unwrap();
    let cell = |a: &Array, r: usize, c: usize| f32::from_bits(a.data[r * d + c]);
    assert!(cell(&want, 0, 0).is_nan() && cell(&folded, 0, 0).to_bits() == 0, "0 / -0: {} and {}", cell(&want, 0, 0), cell(&folded, 0, 0));
    assert!(cell(&want, 1, 0).to_bits() == (-0.0f32).to_bits() && cell(&folded, 1, 0).to_bits() == 0, "0 / -1: {} and {}", cell(&want, 1, 0), cell(&folded, 1, 0));
    assert_eq!(cell(&want, 1, 1), cell(&folded, 1, 1), "1 / -1 has no zero dividend");
    assert_eq!(agreement(&folded, &want, || eval_folded(&i6, &divides, &inputs)).unwrap(), Agreement::Folded, "Futhark's fold, +0.0 for -0.0 included");
    let mut off = folded.clone();
    off.data[0] = 1.0f32.to_bits();
    assert_eq!(agreement(&off, &want, || Ok(folded.clone())).unwrap(), Agreement::Differs, "a value other than the fold's at a zero-dividend cell");
    let mut zeroed = want.clone();
    zeroed.data[d + 1] = 0;
    assert_eq!(agreement(&zeroed, &want, || Ok(folded.clone())).unwrap(), Agreement::Differs, "a zero where the interpreter has a number");
    let nan_of_sqrt = build6(&k, "(rowdiv@m.f32 (tabrows@m.f32[r] k d (tab@a.f32[j] d (sqrt@f32 (neg@f32 (ival@f32 j))))) (tab@a.f32[i] k (add@f32 (ival@f32 i) 1)))").unwrap();
    let want = i6.eval(&nan_of_sqrt, &inputs).unwrap();
    assert!(cell(&want, 0, 1).is_nan(), "sqrt (-1) / 1 is NaN");
    let mut zeroed = want.clone();
    zeroed.data[1] = 0;
    assert_eq!(agreement(&zeroed, &want, || eval_folded(&i6, &nan_of_sqrt, &inputs)).unwrap(), Agreement::Differs, "a zero for a NaN of a sqrt, in a program that divides");
}

/// A `?` LENGTH IS CHARGED AT THE LARGEST SIZE IN SCOPE (F3 review finding 5): a view with a row of `?` length (a
/// scan whose result's length the table leaves unknown, as a `filter`'s is) gives F2's `?` programs, a `map2` over
/// two `?`-length operands, a loop of `?` state length and a body of `?` length; each has a work, a static cost and a
/// value, and none is cheaper than the same program with every `?` length the largest size, known (a `?` is charged
/// that size, plus the adapters that fit it).
#[test]
fn a_question_mark_length_is_charged_at_the_largest_size_in_scope() {
    let v = crate::futhark_view::master6::committed("kmeans").unwrap();
    let rows = v.symbols_tsv.lines().count() - 1;
    let spelling = v.symbols_tsv.lines().find(|l| l.split('\t').nth(1) == Some("M0030")).and_then(|l| l.split('\t').nth(10)).unwrap().to_string();
    let symbols = format!("{}{rows}\tM0030\tqscan@a.f32\tfunction\t[]f32\t[]f32\tlen=?\t\t\t\t{spelling}\n", v.symbols_tsv);
    let k = Arc::new(Kingdom6::from_tables("kmeans_unknown", &v.datatypes_tsv, &symbols, &v.environment_tsv, crate::futhark_view::master6::ADAPTERS6, &v.spec.literals).unwrap());
    let i6 = Interp::new6(&k, &signature(&problem_of("kmeans").instance).unwrap()).unwrap();
    let (n, kk, d) = (5usize, 3usize, 2usize);
    let inputs = [Array::f32s(vec![n, d], vec![0.5; n * d]), Array::f32s(vec![kk, d], vec![0.25; kk * d])];
    let sizes: BTreeMap<String, usize> = [("n", n), ("k", kk), ("d", d)].into_iter().map(|(a, b)| (a.to_string(), b)).collect();
    let programs = [
        ("a map2 over two ? lengths", "(let.a.f32@m.f32[w] (map2.add@a.f32 (qscan@a.f32 zeros_n) (qscan@a.f32 zeros_n)) centres)", "(let.a.f32@m.f32[w] (map2.add@a.f32 (scan.add@a.f32 zeros_n) (scan.add@a.f32 zeros_n)) centres)"),
        ("a loop of ? state length", "(let.a.f32@m.f32[w] (loop@a.f32[s,i] n (qscan@a.f32 zeros_n) (map2.add@a.f32 s s)) centres)", "(let.a.f32@m.f32[w] (loop@a.f32[s,i] n (scan.add@a.f32 zeros_n) (map2.add@a.f32 s s)) centres)"),
        ("a body of ? length", "(tabrows@m.f32[r] k d (qscan@a.f32 zeros_n))", "(tabrows@m.f32[r] k d (scan.add@a.f32 zeros_n))"),
    ];
    for (name, unknown, known) in programs {
        let t = build6(&k, unknown).unwrap_or_else(|e| panic!("{name}: {e}"));
        let w = i6.work_parts(&t, &sizes).unwrap_or_else(|e| panic!("{name}: the work: {e}"));
        let c = i6.static_cost(&t, &sizes).unwrap_or_else(|e| panic!("{name}: the static cost: {e}")).unwrap();
        i6.eval(&t, &inputs).unwrap_or_else(|e| panic!("{name}: the value: {e}"));
        // Every `?` is a scan of zeros_n here, of the largest size n: the program with `scan.add` (its length n) reads and
        // writes as much, but renders no adapter where the `?` one fits to its own length; so the `?` one is no cheaper.
        let tk = build6(&k, known).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (wk, ck) = (i6.work_parts(&tk, &sizes).unwrap(), i6.static_cost(&tk, &sizes).unwrap().unwrap());
        assert!(w.nodes >= wk.nodes && c.traffic >= ck.traffic, "{name}: with ? {w:?} {c:?}, at the largest size {wk:?} {ck:?}");
        eprintln!("{name}: node executions {} (at the largest size, no adapter {}), traffic {} ({})", w.nodes, wk.nodes, c.traffic, ck.traffic);
    }
}

/// THE TEXT FORM BUILDS BACK (phylu P3, d73ca669): 1,000 random genes of every master 6 view print in the text form
/// ([`sexp6`], the ledger's, the hall of fame's and the checkpoint's) to a text that [`build6`] reads back to a tree
/// that renders the same Futhark text, the genes with a `var.T` reading `T`'s constant (`var@i64`) among them.
#[test]
fn a_thousand_random_genes_a_view_print_to_a_text_that_builds_back_to_the_same_program() {
    let n = 1000;
    for (v, view6) in VIEWS6.iter().enumerate() {
        let k = kingdom6(view6).unwrap_or_else(|e| panic!("{view6}: {e}"));
        let renderer = k.renderer6(crate::futhark_table::Header6::of_spec(&crate::futhark_view::master6::committed(view6).unwrap().spec)).unwrap();
        let mut constants = 0usize;
        for (g, tree) in random_trees(&k, view6, n, 0x7e47_0000 + v as u64).iter().enumerate() {
            let text = sexp6(&k, tree).unwrap_or_else(|e| panic!("{view6} gene {g}: {e}"));
            constants += usize::from(text.split([' ', '(', ')']).any(|w| w.starts_with("var@")));
            let back = build6(&k, &text).unwrap_or_else(|e| panic!("{view6} gene {g}: {text}: {e}"));
            assert_eq!(renderer.program(&back).unwrap().text, renderer.program(tree).unwrap().text, "{view6} gene {g}: {text}");
            assert_eq!(sexp6(&k, &back).unwrap(), text, "{view6} gene {g}");
        }
        eprintln!("{view6}: {n} of {n} built back to the same program, {constants} with a var@T reading its constant");
    }
}

/// A `var@T` leaf where a variable of `T` is in scope is not built: the decoder would read that variable, not `T`'s
/// constant.
#[test]
fn a_var_constant_leaf_where_a_variable_of_its_type_is_in_scope_is_not_built() {
    let k = kingdom6("histogram").unwrap();
    let good = "(hist.add@a.i32 k (tab@a.i32[i] n (mod@i32 k (get@i32 is i))) vs)";
    build6(&k, good).unwrap();
    let e = build6(&k, &good.replace("is i)", "is var@i64)")).unwrap_err();
    assert!(e.contains("var@i64") && e.contains("in scope"), "{e}");
}

/// THE COST TABLE IS THE SINGLE SOURCE OF AN ADAPTER'S CHARGE (F3 fix 57de2f98's note): every `fit` line of
/// `master/6/costs.tsv` charges `fitted.cells+source.cells` (the cells it writes and the cells it reads of its
/// operand), `clampix` 2 and a substitution 0, and every adapter of [`ADAPTED_GENES`] at the four
/// [`SIZE_SETTINGS`] is charged exactly what 57de2f98 charged in code: a `fit` its fitted cells plus its source's,
/// `clampix` 2.
#[test]
fn every_adapters_charge_is_the_cost_tables_and_unchanged() {
    let table = include_str!("../../kingdoms/futhark/master/6/costs.tsv");
    let lines: Vec<Vec<&str>> = table.lines().map(|l| l.split('\t').collect()).filter(|f: &Vec<&str>| f[0] == "adapter").collect();
    assert!(!lines.is_empty());
    for f in &lines {
        let want = if f[2].starts_with("fit") { "fitted.cells+source.cells" } else if f[2] == "clampix" { "2" } else { "0" };
        assert_eq!(f[8], want, "the adapter {}'s line in master/6/costs.tsv", f[2]);
    }
    let (mut fits, mut clamps) = (0usize, 0usize);
    for (view6, text) in ADAPTED_GENES {
        let k = kingdom6(view6).unwrap();
        let renderer = k.renderer6(crate::futhark_table::Header6::of_spec(&crate::futhark_view::master6::committed(view6).unwrap().spec)).unwrap();
        let d = renderer.decode(&build6(&k, text).unwrap()).unwrap();
        let names: Vec<String> = k.variables.iter().filter(|v| v.kind == "size").map(|v| v.name.clone()).collect();
        for values in SIZE_SETTINGS {
            let sizes: BTreeMap<String, usize> = names.iter().cloned().zip(values).collect();
            for (c, own, _) in adapter_charges(&k, &d, &sizes) {
                let was = match (&c.source, &c.fitted) {
                    (Some((ts, ps)), Some((tf, pf))) if c.adapter.starts_with("fit") => {
                        fits += 1;
                        crate::futhark_cost::cells6(&k, *tf, pf, &sizes).unwrap() + crate::futhark_cost::cells6(&k, *ts, ps, &sizes).unwrap()
                    }
                    (None, None) if c.adapter == "clampix" => {
                        clamps += 1;
                        2
                    }
                    _ => panic!("{text}: an adapter the charge does not cover: {c:?}"),
                };
                assert_eq!(own, was, "{text} at {sizes:?}: {}", c.adapter);
            }
        }
    }
    assert!(fits > 0 && clamps > 0, "fits {fits}, clampix {clamps}");
    eprintln!("adapters charged as before, from the table: {fits} fits, {clamps} clampix");
}
