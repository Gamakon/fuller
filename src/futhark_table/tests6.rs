//! Master 6's renderer (pass 5, plan section 7 task F2): the reference programs render their master 5 texts byte for
//! byte, every adapter of the design's section 5 table renders a program Futhark accepts, and random genes of every
//! view render programs Futhark accepts and runs.

use super::*;
use crate::futhark_sig::{build6, expected_text6, kingdom6, REFERENCES6};

/// A problem's kingdom and renderer.
fn view(problem: &str) -> (Kingdom6, Header6) {
    let k = kingdom6(problem).unwrap_or_else(|e| panic!("{problem}: {e}"));
    let spec = crate::futhark_view::master6::committed(problem).unwrap().spec;
    (k, Header6::of_spec(&spec))
}

/// EVERY REFERENCE RENDERS ITS MASTER 5 TEXT: each F2a tree, decoded and rendered, is byte for byte the stored master 5
/// rendering (`tests/fixtures/futhark_m6_references.txt`), `0i64` indices included, and needs no adapter.
#[test]
fn every_reference_renders_its_master_5_text_byte_for_byte() {
    for r in &REFERENCES6 {
        let key = r.key();
        let (k, header) = view(r.problem);
        let renderer = k.renderer6(header).unwrap();
        let tree = build6(&k, &(r.master6)()).unwrap_or_else(|e| panic!("{key}: {e}"));
        let p = renderer.program(&tree).unwrap_or_else(|e| panic!("{key}: {e}"));
        let want = expected_text6(&key).unwrap_or_else(|| panic!("{key}: no stored text"));
        assert!(p.adapter_free() && p.substitution_free(), "{key}: renders adapters {:?} and substitutions {:?}", p.adapters, p.substitutions);
        if p.text != want {
            let at = p.text.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(p.text.len().min(want.len()));
            panic!("{key}: differs at byte {at}:\n got: {}\nwant: {}", &p.text[at.saturating_sub(80)..(at + 80).min(p.text.len())], &want[at.saturating_sub(80)..(at + 80).min(want.len())]);
        }
    }
}

use crate::futhark_sig::{const_len6, problem, random_gene};
use crate::typed_gep::{express, GeneLayout};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// Every master 6 view proof 1 covers (plan 7.3): the problems' views and the `i32` + `f32` k-means view.
const VIEWS6: [&str; 7] = ["reduce", "scan", "histogram", "pathfinder", "a7", "kmeans", "kmeans_i32"];

/// The program size of a view's genes: its problem's (the `i32` + `f32` k-means view's is k-means's).
fn program_size(view: &str) -> usize {
    problem(if view == "kmeans_i32" { "kmeans" } else { view }).unwrap().program_size
}

/// `n` random genes of a view, expressed and rendered, from the random seed `seed`.
fn random_programs(view6: &str, n: usize, seed: u64) -> Vec<(Tree, Program6)> {
    let (k, header) = view(view6);
    let renderer = k.renderer6(header).unwrap();
    let layout = GeneLayout::from_program_size(&renderer.contract, k.root.data_type, program_size(view6)).unwrap();
    let consts = const_len6(&k);
    let mut rng = GeneRng::new(seed);
    (0..n)
        .map(|i| {
            let gene = random_gene(&renderer.contract, &layout, &consts, &mut rng).unwrap();
            let tree = express(&renderer.contract, &layout, &gene, &consts).unwrap_or_else(|e| panic!("{view6} gene {i}: {e:?}")).tree();
            let p = renderer.program(&tree).unwrap_or_else(|e| panic!("{view6} gene {i}: our software has a bug: {e}"));
            (tree, p)
        })
        .collect()
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("fuller_f2_{name}_{}", std::process::id()))
}

/// A program as a module of a batch (`module m<i> = { .. }`, its entry a `def`), so programs with their own preludes
/// share one file and `futhark check` checks each prelude.
fn as_module(i: usize, text: &str) -> String {
    format!("module m{i} = {{\n{}}}\n\n", text.replacen("entry main ", "def main ", 1))
}

/// A program in a batch with its entry kept: its module ([`as_module`]) and the entry `e<i>` of the program's own
/// header calling it ([`entry_calling`]), so the rules Futhark applies only to an entry point are checked for every
/// program of the batch (F2 review finding 5).
fn with_entry(i: usize, text: &str) -> String {
    format!("{}{}\n", as_module(i, text), entry_calling(i, text))
}

/// `futhark check` on a file, bounded at `limit`: `Ok` when Futhark accepts it, else its error text.
fn futhark_check(src: &str, dir: &Path, limit: Duration) -> Result<(), String> {
    std::fs::create_dir_all(dir).unwrap();
    let fut = dir.join("check.fut");
    std::fs::write(&fut, src).unwrap();
    let out = crate::futhark_exec::output_with_timeout(Command::new("futhark").arg("check").arg(&fut), limit, None)?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).to_string())
    }
}

/// `futhark check` on a batch of programs; on a rejection, each program alone, so the error names its program.
fn check_programs(view6: &str, programs: &[&str], dir: &Path) -> Vec<(usize, String)> {
    let limit = Duration::from_secs(600);
    let batch: String = programs.iter().enumerate().map(|(i, t)| with_entry(i, t)).collect();
    match futhark_check(&batch, dir, limit) {
        Ok(()) => Vec::new(),
        Err(e) if e.contains("no answer within") => vec![(usize::MAX, format!("{view6}: the batch's futhark check ran past {limit:?}: {e}"))],
        Err(_) => programs.iter().enumerate().filter_map(|(i, t)| futhark_check(t, &dir.join(format!("p{i}")), limit).err().map(|e| (i, e))).collect(),
    }
}

/// THE EARLY GUARD (plan 7.5 risk 6) AND PROOF 1's CHECK (plan 7.3 item 1): 1,000 random genes of every master 6 view
/// express, decode and render with no defect, and `futhark check` accepts every program with its entry point kept,
/// N of N. Printed per view, apart: the share of genes with no adapter and of rendered nodes with one, and the share
/// of genes with no work-bound substitution and the substituted nodes (master 5: 100 % of both).
#[test]
fn a_thousand_random_genes_a_view_render_programs_futhark_accepts() {
    let n = 1000;
    let mut report = String::new();
    for (v, view6) in VIEWS6.iter().enumerate() {
        let programs = random_programs(view6, n, 0x5eed_0000 + v as u64);
        let free = programs.iter().filter(|(_, p)| p.adapter_free()).count();
        let unsubstituted = programs.iter().filter(|(_, p)| p.substitution_free()).count();
        let nodes: usize = programs.iter().map(|(_, p)| p.nodes).sum();
        let adapted: usize = programs.iter().map(|(_, p)| p.adapter_nodes()).sum();
        let substituted: usize = programs.iter().map(|(_, p)| p.substituted_nodes()).sum();
        let (mut by_adapter, mut by_cause): (BTreeMap<&str, usize>, BTreeMap<&str, usize>) = (BTreeMap::new(), BTreeMap::new());
        for (_, p) in &programs {
            for (_, a) in &p.adapters {
                *by_adapter.entry(a.as_str()).or_default() += 1;
            }
            for (_, c) in &p.substitutions {
                *by_cause.entry(c.as_str()).or_default() += 1;
            }
        }
        let texts: Vec<&str> = programs.iter().map(|(_, p)| p.text.as_str()).collect();
        let dir = scratch(&format!("check_{view6}"));
        let mut bad = Vec::new();
        for (c, chunk) in texts.chunks(100).enumerate() {
            bad.extend(check_programs(view6, chunk, &dir.join(format!("c{c}"))).into_iter().map(|(i, e)| (c * 100 + i, e)));
        }
        std::fs::remove_dir_all(&dir).ok();
        let line = format!(
            "{view6}: {} of {n} accepted by futhark check; adapter-free genes {free} of {n} ({:.1} %); nodes with an adapter {adapted} of {nodes} ({:.2} %); adapters {by_adapter:?}; substitution-free genes {unsubstituted} of {n} ({:.1} %); substituted nodes {substituted} of {nodes} ({:.2} %); substitutions {by_cause:?}\n",
            n - bad.len(),
            100.0 * free as f64 / n as f64,
            100.0 * adapted as f64 / nodes.max(1) as f64,
            100.0 * unsubstituted as f64 / n as f64,
            100.0 * substituted as f64 / nodes.max(1) as f64
        );
        eprint!("{line}");
        report.push_str(&line);
        if let Some((i, e)) = bad.first() {
            panic!("{report}{view6} gene {i} is rejected:\n{}\n{e}", texts.get(*i).unwrap_or(&""));
        }
    }
}

/// The batch check finds a rejected program and names it: the guard above cannot pass by checking nothing.
#[test]
fn the_batch_check_names_a_program_futhark_rejects() {
    let good = "entry main [n] (xs: [n]i32) : i32 =\n  i32.sum (xs)\n";
    let bad = "entry main [n] (xs: [n]i32) : i32 =\n  map2 (+) (xs) (iota 3)\n";
    let dir = scratch("negative");
    let found = check_programs("negative", &[good, bad, good], &dir);
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(found.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![1], "{found:?}");
}

/// The batch check keeps each program's entry point: a program Futhark accepts as a `def` but not as an entry (a
/// function-valued parameter) is named, so the entry's own rules are checked in the batch too.
#[test]
fn the_batch_check_keeps_the_entry_rules() {
    let good = "entry main [n] (xs: [n]i32) : i32 =\n  i32.sum (xs)\n";
    let not_an_entry = "entry main (f: i32 -> i32) : i32 =\n  f 1\n";
    let dir = scratch("entry_rules");
    let found = check_programs("entry_rules", &[good, not_an_entry, good], &dir);
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(found.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![1], "{found:?}");
}

/// The entry `entry e<i> .. = m<i>.main ..` that calls a batch module's program with the program's own signature.
fn entry_calling(i: usize, text: &str) -> String {
    let start = text.find("entry main ").expect("a program has its entry");
    let line = &text[start..start + text[start..].find(" =\n").expect("the entry's header ends")];
    let names: Vec<&str> = line.split('(').skip(1).filter_map(|p| p.split_once(':').map(|(n, _)| n.trim())).collect();
    format!("{} = m{i}.main {}\n", line.replacen("entry main ", &format!("entry e{i} "), 1), names.join(" "))
}

/// The sizes of a case: every dimension of its array inputs and the value of each integer scalar input (the
/// histogram's `k`).
fn case_sizes(case: &crate::futhark_sig::Case) -> BTreeSet<i64> {
    let mut sizes = BTreeSet::new();
    for a in &case.inputs {
        if a.dims.is_empty() {
            if a.elem == crate::futhark_sig::Elem::I32 {
                sizes.insert(i64::from(a.data[0] as i32));
            }
        } else {
            sizes.extend(a.dims.iter().map(|d| *d as i64));
        }
    }
    sizes
}

/// The number of size names a view's entry has (`n`, `m`, `p`): its size variables and the sizes its inputs name.
fn roles(view6: &str) -> usize {
    let k = view(view6).0;
    let mut names: BTreeSet<String> = k.variables.iter().filter(|v| v.kind == "size").map(|v| v.name.clone()).collect();
    for v in k.variables.iter().filter(|v| v.kind == "input") {
        names.extend(v.properties.iter().filter_map(|(_, e)| if let PropertyExpr::Size(n) = e { Some(n.clone()) } else { None }));
    }
    names.len()
}

/// A case whose sizes are as many different values as the view has size names, each at least 2: a `fit` or `clampix`
/// to the wrong one of two sizes moves data or reads out of bounds on it.
fn distinct_sizes(roles: usize, case: &crate::futhark_sig::Case) -> bool {
    let sizes = case_sizes(case);
    sizes.len() == roles && sizes.iter().all(|s| *s >= 2)
}

/// The cases a view's random programs run on, never a test case (the test split is kept for the exactness claim, F2
/// review finding 6), each the smallest of its kind by its input's bytes: the smallest case (often a size of 1), the
/// smallest whose sizes are distinct and at least 2 (A7's `held_n7_m5_p3_s4`, the histogram's `k = 2`; F2 review
/// finding 2), and the smallest training case.
fn run_cases(view6: &str) -> Vec<crate::futhark_sig::Case> {
    let p = problem(if view6 == "kmeans_i32" { "kmeans" } else { view6 }).unwrap();
    let mut cases: Vec<_> = crate::futhark_sig::load_cases(&p).unwrap().into_iter().filter(|c| c.split != crate::futhark_sig::Split::Test).collect();
    cases.sort_by_key(|c| (c.stdin.len(), c.name.clone()));
    let r = roles(view6);
    let picks = [cases.first(), cases.iter().find(|c| distinct_sizes(r, c)), cases.iter().find(|c| c.split == crate::futhark_sig::Split::Train)];
    let mut out: Vec<crate::futhark_sig::Case> = Vec::new();
    for c in picks.into_iter().map(|c| c.expect("a view has a case of each kind")) {
        if !out.iter().any(|o| o.name == c.name) {
            out.push(c.clone());
        }
    }
    out
}

/// THE RUN CASES ARE RECTANGULAR AND NEVER TEST CASES: A7 runs on a case of `n`, `m`, `p` all different, the histogram
/// on one with `k > 1`, pathfinder on one of two different sizes, and no view on a test case.
#[test]
fn the_run_cases_have_distinct_sizes_and_no_test_case() {
    for view6 in VIEWS6 {
        let cases = run_cases(view6);
        let names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
        assert!(cases.iter().all(|c| c.split != crate::futhark_sig::Split::Test), "{view6} runs on a test case: {names:?}");
        assert!(cases.iter().any(|c| distinct_sizes(roles(view6), c)), "{view6}: no case of distinct sizes in {names:?}");
        eprintln!("{view6}: run cases {names:?}");
    }
    assert!(run_cases("a7").iter().any(|c| case_sizes(c).len() == 3), "a7 has no case with n, m, p all different");
    assert!(run_cases("histogram").iter().any(|c| c.inputs.iter().any(|a| a.dims.is_empty() && a.data[0] as i32 > 1)), "histogram has no case with k > 1");
    assert!(run_cases("pathfinder").iter().any(|c| case_sizes(c).len() == 2), "pathfinder has no rectangular case");
}

/// The dims of a case's output: its stored answer's, or (k-means, scored on its objective) the centres'.
fn output_dims(case: &crate::futhark_sig::Case) -> Vec<usize> {
    match &case.output {
        Some(a) => a.dims.clone(),
        None => case.inputs[1].dims.clone(),
    }
}

/// A row that can make a NaN from numbers that are not NaN (a division, a root, a logarithm, a power, an extreme
/// constant whose sum overflows): a program that renders none has no NaN in its output.
fn may_make_nan(text: &str) -> bool {
    ["/", "sqrt", "log", "**", "exp", "highest", "lowest", "inf", ".nan"].iter().any(|s| text.contains(s))
}

/// THE LEGAL OUTPUT of a run: the entry's declared shape, and no NaN where the program renders nothing that can make
/// one. The values are not compared with an interpreter here: master 6's interpreter is task F3 (proof 1's second
/// half), and master 5's does not read master 6 trees.
fn legal_output(case: &crate::futhark_sig::Case, out: &crate::futhark_sig::Array, text: &str) -> Result<(), String> {
    let want = output_dims(case);
    if out.dims != want {
        return Err(format!("the output is {:?}, the entry declares {want:?}", out.dims));
    }
    if out.elem == crate::futhark_sig::Elem::F32 && !may_make_nan(text) {
        if let Some(i) = out.data.iter().position(|w| f32::from_bits(*w).is_nan()) {
            return Err(format!("element {i} of the output is NaN, and the program renders nothing that makes one"));
        }
    }
    Ok(())
}

/// The legality check rejects a wrong shape and an unexpected NaN, and lets a NaN pass where a row can make one.
#[test]
fn the_legal_output_check_rejects_a_wrong_shape_and_an_unexpected_nan() {
    let case = run_cases("kmeans").remove(0);
    let dims = output_dims(&case);
    let words = dims.iter().product::<usize>();
    let fine = crate::futhark_sig::Array { dims: dims.clone(), elem: crate::futhark_sig::Elem::F32, data: vec![0f32.to_bits(); words] };
    assert_eq!(legal_output(&case, &fine, "map2 (+) xs ys"), Ok(()));
    let mut nan = fine.clone();
    nan.data[words - 1] = f32::NAN.to_bits();
    assert!(legal_output(&case, &nan, "map2 (+) xs ys").is_err());
    assert_eq!(legal_output(&case, &nan, "map (\\x -> f32.sqrt x) xs"), Ok(()));
    let short = crate::futhark_sig::Array { dims: vec![dims[0]], elem: crate::futhark_sig::Elem::F32, data: vec![0; dims[0]] };
    assert!(legal_output(&case, &short, "xs").is_err());
}

/// What one program's run gives: its output's dims and element, or the failure.
fn run_entry(exe: &Path, entry: &str, stdin: &[u8]) -> Result<crate::futhark_sig::Array, String> {
    let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(exe).arg("-e").arg(entry).arg("-b"), Duration::from_secs(120), Some(stdin))?;
    if !out.status.success() {
        return Err(format!("exit {}: {}", out.status, String::from_utf8_lossy(&out.stderr)));
    }
    crate::futhark_sig::read_one(&out.stdout)
}

/// PROOF 1, RUN (plan 7.3 item 1, the C half): 100 random genes of every master 6 view compile with Futhark's `c`
/// backend and run legally on the view's run cases ([`run_cases`]: up to three stored cases, none a test case, one of
/// distinct sizes): exit 0 (no index out of bounds, no size mismatch), the entry's declared output shape, and no NaN
/// where the program renders nothing that makes one ([`legal_output`]); values are not compared with an interpreter
/// (F3). N of N. Every compile and run is bounded; one that runs past its bound fails the test with its program named.
#[test]
fn a_hundred_random_genes_a_view_compile_and_run_on_the_c_backend() {
    let (n, chunk) = (100, 25);
    let mut report = String::new();
    for (v, view6) in VIEWS6.iter().enumerate() {
        let programs = random_programs(view6, n, 0xc0de_0000 + v as u64);
        let cases = run_cases(view6);
        let dir = scratch(&format!("run_{view6}"));
        let mut bad: Vec<String> = Vec::new();
        for (c, part) in programs.chunks(chunk).enumerate() {
            let base = c * chunk;
            let src: String = part.iter().enumerate().map(|(i, (_, p))| format!("{}{}\n", as_module(base + i, &p.text), entry_calling(base + i, &p.text))).collect();
            let exe = match crate::futhark_sig::compile_c(&src, &dir.join(format!("c{c}"))) {
                Ok(exe) => exe,
                Err(e) => {
                    bad.push(format!("{view6} genes {base}..{}: futhark c: {e}", base + part.len()));
                    continue;
                }
            };
            for (i, (_, p)) in part.iter().enumerate() {
                for case in &cases {
                    if let Err(e) = run_entry(&exe, &format!("e{}", base + i), &case.stdin).and_then(|a| legal_output(case, &a, &p.text)) {
                        bad.push(format!("{view6} gene {} on {}: {e}\n{}", base + i, case.name, p.text));
                    }
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
        let names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
        let line = format!("{view6}: {n} compiled (c); {} of {} runs legal on {names:?}\n", n * cases.len() - bad.len(), n * cases.len());
        eprint!("{line}");
        report.push_str(&line);
        assert!(bad.is_empty(), "{report}{}", bad.join("\n"));
    }
}

/// Render a master 6 s-expression of a view; the program and the adapters it renders.
fn render(k: &Kingdom6, header: &Header6, text: &str) -> Program6 {
    let tree = build6(k, text).unwrap_or_else(|e| panic!("{text}: {e}"));
    k.renderer6(header.clone()).unwrap().program(&tree).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// Check every program with `futhark check` (one batch), and panic naming the first one Futhark rejects.
fn assert_accepted(what: &str, programs: &[(String, Program6)]) {
    let texts: Vec<&str> = programs.iter().map(|(_, p)| p.text.as_str()).collect();
    let dir = scratch(what);
    let bad = check_programs(what, &texts, &dir);
    std::fs::remove_dir_all(&dir).ok();
    if let Some((i, e)) = bad.first() {
        let (name, p) = programs.get(*i).map_or(("the batch", ""), |(n, p)| (n.as_str(), p.text.as_str()));
        panic!("{what}: {name} is rejected:\n{p}\n{e}");
    }
}

/// The k-means view's expressions that fail one requirement each, with the adapter each renders (design section 5's
/// table; the `fit` of a slot is `fit.body`'s, a lambda body against its size argument), as (name, data type tag,
/// s-expression). `[]f32` of lengths `n`, `k`, `d` are the zero arrays; the case's sizes differ.
const ADAPTER_ROWS: [(&str, &str, &str, &str); 6] = [
    ("map2 of two lengths", "fit.pair", "a.f32", "(map2.add@a.f32 zeros_n zeros_k)"),
    ("a slot of another length", "fit.body", "m.f32", "(tabrows@m.f32[r] k d zeros_n)"),
    ("an index of another bound", "clampix", "f32", "(get@f32 zeros_n (last@i64 k))"),
    ("a loop body of another length", "fit.state", "a.f32", "(loop@a.f32[s,i] n zeros_k zeros_n)"),
    ("matmul's inner sizes", "fit.inner", "m.f32", "(matmul@m.f32 points centres)"),
    ("map2_rows' rows and columns", "fit.pair", "m.f32", "(map2_rows@m.f32 add@fn.arr2_f32_arr_f32_arr_f32 points (transpose@m.f32 centres))"),
];

/// An expression of the tag's data type as a `[]f32` (a scalar replicated over `n`, a matrix's first row).
fn as_array(tag: &str, e: &str) -> String {
    match tag {
        "f32" => format!("(replicate@a.f32 n {e})"),
        "a.f32" => e.to_string(),
        _ => format!("(row@a.f32 {e} zero@i64)"),
    }
}

/// An expression of the tag's data type as an `f32`.
fn as_scalar(tag: &str, e: &str) -> String {
    match tag {
        "f32" => e.to_string(),
        "a.f32" => format!("(sum@f32 {e})"),
        _ => format!("(get@f32 (row@a.f32 {e} zero@i64) zero@i64)"),
    }
}

/// The expression nested three deep in its own adapted operand: each adapter's operand is the adapter before (a
/// loop's through its state, not its body, so no level adds a trip; `tabrows` has no other operand than its body, so
/// its third level sits at the work bound and is substituted).
fn nested3(name: &str, e: &str) -> String {
    let mut x = e.to_string();
    for _ in 0..2 {
        x = match name {
            "map2 of two lengths" => format!("(map2.add@a.f32 {x} zeros_d)"),
            "a slot of another length" => format!("(tabrows@m.f32[r] k d (row@a.f32 {x} zero@i64))"),
            "an index of another bound" => format!("(get@f32 zeros_d (clamp@i64 k {x}))"),
            "a loop body of another length" => format!("(loop@a.f32[s,i] n {x} zeros_d)"),
            "matmul's inner sizes" => format!("(matmul@m.f32 {x} centres)"),
            _ => format!("(map2_rows@m.f32 add@fn.arr2_f32_arr_f32_arr_f32 {x} (transpose@m.f32 centres))"),
        };
    }
    x
}

/// THE DESIGN'S ADAPTER TABLE AS FIXTURES (design section 5; master 6 README 5.5): each requirement that fails renders
/// its adapter, and `futhark check` accepts the program at the root, inside a `tabulate` lambda, inside a loop body,
/// and with the adapter nested three deep in its own operand. The entry's result (`fit.root`) is the root's: a
/// program of `[n][d]` and one of `[d][n]` where the entry returns `[k][d]`.
#[test]
fn every_adapter_of_the_design_table_renders_a_program_futhark_accepts() {
    let (k, header) = view("kmeans");
    let root = |arr: &str| format!("(let.a.f32@m.f32[w] {arr} centres)");
    let mut programs = Vec::new();
    for (name, adapter, tag, e) in ADAPTER_ROWS {
        let p = render(&k, &header, &root(&as_array(tag, e)));
        assert!(p.adapters.iter().any(|(_, a)| a == adapter), "{name}: renders {:?}, not {adapter}", p.adapters);
        programs.push((format!("{name} at the root"), p));
        let contexts = [
            ("inside a tabulate lambda", root(&format!("(tab@a.f32[t] n {})", as_scalar(tag, e)))),
            ("inside a loop body", root(&format!("(loop@a.f32[s,i] n zeros_n (map2.add@a.f32 s {}))", as_array(tag, e)))),
            ("nested three deep", root(&as_array(tag, &nested3(name, e)))),
        ];
        for (place, text) in contexts {
            let p = render(&k, &header, &text);
            assert!(p.adapters.iter().any(|(_, a)| a == adapter), "{name} {place}: renders {:?}, not {adapter}", p.adapters);
            if place == "nested three deep" {
                assert!(p.adapters.iter().filter(|(_, a)| a == adapter).count() >= 3 || adapter == "fit.body", "{name} {place}: {:?}", p.adapters);
            }
            programs.push((format!("{name} {place}"), p));
        }
    }
    for (name, text) in [("the entry's rows", "points"), ("the entry's rows and columns", "(transpose@m.f32 points)")] {
        let p = render(&k, &header, text);
        assert!(p.adapters.iter().all(|(path, a)| path.is_empty() && a == "fit.root") && !p.adapters.is_empty(), "{name}: {:?}", p.adapters);
        programs.push((name.to_string(), p));
    }
    let (h, hh) = view("histogram");
    let p = render(&h, &hh, "(hist.add@a.i32 k zeros_n zeros_k)");
    assert_eq!(p.adapters, vec![(Vec::new(), "fit.pair".to_string())]);
    programs.push(("hist of two lengths".to_string(), p));
    assert_accepted("adapter_table", &programs);
}

/// A k-means view with one more row whose length is known only at run time (`filter`: `len=?`), so `?` lengths reach
/// the adapters. No row of the committed master gives `?`; the renderer's `let`-bound form must still be legal.
fn kmeans_with_a_filter() -> (Kingdom6, Header6) {
    kmeans_with(&[])
}

/// The k-means view with `filter@a.f32` and the synthetic `rows` (each a symbols.tsv row from its `ref` column on).
fn kmeans_with(rows: &[&str]) -> (Kingdom6, Header6) {
    let v = crate::futhark_view::master6::committed("kmeans").unwrap();
    let n = v.symbols_tsv.lines().count() - 1;
    let mut symbols = format!("{}{n}\tM9999\tfilter@a.f32\tfunction\t[]f32\t[]f32\tlen=?\t\t\t\tfilter (\\x -> x > 0f32) {{0}}\n", v.symbols_tsv);
    for (i, row) in rows.iter().enumerate() {
        symbols.push_str(&format!("{}\t{row}\n", n + 1 + i));
    }
    let k = Kingdom6::from_tables("kmeans_filter", &v.datatypes_tsv, &symbols, &v.environment_tsv, crate::futhark_view::master6::ADAPTERS6, &v.spec.literals).unwrap();
    (k, Header6::of_spec(&v.spec))
}

/// Compile programs on `c` with their entries `e<i>` and run each on the k-means run cases: the outputs, per case.
fn run_on_kmeans(what: &str, programs: &[&Program6]) -> Vec<Vec<crate::futhark_sig::Array>> {
    let src: String = programs.iter().enumerate().map(|(i, p)| with_entry(i, &p.text)).collect();
    let dir = scratch(what);
    let exe = crate::futhark_sig::compile_c(&src, &dir).unwrap_or_else(|e| panic!("{what}: {e}"));
    let out = run_cases("kmeans")
        .iter()
        .map(|case| {
            (0..programs.len())
                .map(|i| {
                    let a = run_entry(&exe, &format!("e{i}"), &case.stdin).unwrap_or_else(|e| panic!("{what} program {i} on {}: {e}\n{}", case.name, programs[i].text));
                    legal_output(case, &a, &programs[i].text).unwrap_or_else(|e| panic!("{what} program {i} on {}: {e}\n{}", case.name, programs[i].text));
                    a
                })
                .collect()
        })
        .collect();
    std::fs::remove_dir_all(&dir).ok();
    out
}

/// TWO `?` ADAPTERS ON ONE NODE BIND DIFFERENT NAMES (F2 review finding 7): a node with two `fit.pair` requirements,
/// each between two `?`-length operands, binds `ad_a`, `ad_b`, `ad_l`, then `ad_a1`, `ad_b1`, `ad_l1`; Futhark
/// accepts the program and it runs.
#[test]
fn two_unknown_length_adapters_on_one_node_bind_different_names() {
    let (k, header) = kmeans_with(&["M9998\tdot2@f32\tfunction\tf32\t[]f32;[]f32;[]f32;[]f32\t\tin1.len = in0.len -> fit.pair; in3.len = in2.len -> fit.pair; env.trips < 3 -> subst\t\t\tf32.sum (map2 (*) {0} {1}) + f32.sum (map2 (*) {2} {3})"]);
    let text = "(let.a.f32@m.f32[w] (replicate@a.f32 n (dot2@f32 (filter@a.f32 zeros_n) (filter@a.f32 zeros_d) (filter@a.f32 zeros_k) (filter@a.f32 zeros_n))) centres)";
    let p = render(&k, &header, text);
    assert_eq!(p.adapters.iter().filter(|(_, a)| a == "fit.pair").count(), 2, "{:?}", p.adapters);
    for name in ["let ad_l = ", "let ad_l1 = ", "let ad_a = ", "let ad_a1 = "] {
        assert!(p.text.contains(name), "no {name:?} in\n{}", p.text);
    }
    assert_accepted("two_unknown", &[("dot2".to_string(), p.clone())]);
    run_on_kmeans("two_unknown_run", &[&p]);
}

/// A CLAMP READS THE INDEXED ARRAY AS ITS NODE'S OTHER ADAPTERS LEFT IT (F2 review finding 7): a node that clamps an
/// index into an array (`clampix`) and fits that array to a shorter length (`fit`) clamps to the fitted length, so
/// the program reads in bounds (a clamp to the old length reads past the fitted array on a case where `k < n`).
#[test]
fn a_clamp_reads_the_length_its_array_was_fitted_to() {
    let (k, header) = kmeans_with(&["M9997\tgetk@f32\tfunction\tf32\t[]f32;i64;size\t\tin1.bound <= in0.len -> clampix; in0.len = in2.value -> fit; env.trips < 3 -> subst\t\t\t{0}[{1}]"]);
    let text = "(let.a.f32@m.f32[w] (replicate@a.f32 n (getk@f32 zeros_n (last@i64 d) k)) centres)";
    let p = render(&k, &header, text);
    let names: Vec<&str> = p.adapters.iter().map(|(_, a)| a.as_str()).collect();
    assert_eq!(names, ["fit", "clampix"], "{}", p.text);
    assert!(p.text.contains("clampix k "), "the clamp is not to the fitted length k:\n{}", p.text);
    run_on_kmeans("clamp_fitted", &[&p]);
}

/// `?` LENGTHS ARE BOUND TO A NAME BEFORE FUTHARK SEES THEM (plan 7.1): a loop whose state's length is `?` declares it
/// `[ad_m]` after the size binding `let [ad_m] (ad_s: [ad_m]f32) = ..`; a `map2` over two `?`-length operands binds both and fits
/// them to `ad_l`; an index into a `?`-length array clamps to `length ad_a`; a body of `?` length fits to its size.
/// Futhark accepts each, and each compiles and runs on the c backend.
#[test]
fn unknown_lengths_are_bound_to_a_name_and_futhark_accepts_them() {
    let (k, header) = kmeans_with_a_filter();
    let root = |arr: &str| format!("(let.a.f32@m.f32[w] {arr} centres)");
    let cases = [
        ("a loop of ? state length", root("(loop@a.f32[s,i] n (filter@a.f32 zeros_n) (map2.add@a.f32 s s))"), "let [ad_m] (ad_s: [ad_m]f32) = "),
        ("a map2 over two ? lengths", root("(map2.add@a.f32 (filter@a.f32 zeros_n) (filter@a.f32 zeros_d))"), "let ad_l = i64.min (length ad_a) (length ad_b)"),
        ("an index into a ? length", root("(replicate@a.f32 n (get@f32 (filter@a.f32 zeros_n) (last@i64 n)))"), "clampix ad_m"),
        ("a body of ? length", "(tabrows@m.f32[r] k d (filter@a.f32 zeros_d))".to_string(), "fit d 0f32"),
        ("map2 of a ? length nested three deep", root("(map2.add@a.f32 (map2.add@a.f32 (map2.add@a.f32 (filter@a.f32 zeros_n) zeros_k) zeros_d) zeros_n)"), "let ad_l"),
    ];
    let mut programs = Vec::new();
    for (name, text, shows) in cases {
        let p = render(&k, &header, &text);
        assert!(p.text.contains(shows), "{name}: no {shows:?} in\n{}", p.text);
        programs.push((name.to_string(), p));
    }
    assert!(programs[0].1.text.contains("loop (s_a_q: [ad_m]f32) = (ad_s)"), "{}", programs[0].1.text);
    assert_accepted("unknown_lengths", &programs);
    let src: String = programs.iter().enumerate().map(|(i, (_, p))| format!("{}{}\n", as_module(i, &p.text), entry_calling(i, &p.text))).collect();
    let dir = scratch("unknown_run");
    let exe = crate::futhark_sig::compile_c(&src, &dir).unwrap();
    for case in run_cases("kmeans") {
        for (i, (name, _)) in programs.iter().enumerate() {
            let a = run_entry(&exe, &format!("e{i}"), &case.stdin).unwrap_or_else(|e| panic!("{name} on {}: {e}", case.name));
            assert_eq!(a.dims, output_dims(&case), "{name} on {}", case.name);
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// TEXT LINEAR IN THE DEPTH (adversarial H1): ten `map2` adapters, each the operand of the next, grow the text by a
/// bounded amount a level (never doubling), and Futhark accepts the deepest.
#[test]
fn ten_nested_map2_adapters_give_text_linear_in_the_depth() {
    let (k, header) = view("kmeans");
    let zeros = ["zeros_k", "zeros_d", "zeros_n"];
    let mut e = "zeros_n".to_string();
    let mut lengths = Vec::new();
    let mut last = None;
    for level in 0..10 {
        e = format!("(map2.add@a.f32 {e} {})", zeros[level % 3]);
        let p = render(&k, &header, &format!("(let.a.f32@m.f32[w] {e} centres)"));
        assert_eq!(p.adapters.iter().filter(|(_, a)| a == "fit.pair").count(), level + 1, "level {level}: {:?}", p.adapters);
        lengths.push(p.text.len());
        last = Some(p);
    }
    let steps: Vec<usize> = lengths.windows(2).map(|w| w[1] - w[0]).collect();
    let most = *steps.iter().max().unwrap();
    assert!(most < 120 && lengths[9] < lengths[0] + 9 * most, "the text grows by {steps:?} a level ({lengths:?})");
    eprintln!("ten nested map2 adapters: {lengths:?} bytes, a level adds {steps:?}");
    assert_accepted("linear_depth", &[("depth 10".to_string(), last.unwrap())]);
}

/// NESTED BINDERS OF ONE DATA TYPE AND PROPERTIES HAVE DISTINCT NAMES (`i_n`, `i2_n`, as master 5's `var_name`), the
/// text is master 5's rendering of the same program, and the interpreter (master 5's, on that program) and the `c`
/// backend give the same value bit for bit on every stored case.
#[test]
fn nested_binders_render_distinct_names_and_the_interpreter_and_c_agree() {
    let (k, header) = view("reduce");
    let m6 = "(sum@i32 (tab@a.i32[i] n (sum@i32 (tab@a.i32[j] n (sub@i32 (mul@i32 (get@i32 xs i) 2) (get@i32 xs j))))))";
    let p = render(&k, &header, m6);
    assert!(p.adapter_free(), "{:?}", p.adapters);
    assert!(p.text.contains("(\\i_n -> ") && p.text.contains("(\\i2_n -> "), "{}", p.text);
    let inst = problem("reduce").unwrap().instance;
    let k5 = crate::futhark_sig::load(&inst).unwrap();
    let sig = crate::futhark_sig::signature(&inst).unwrap();
    let tree5 = crate::futhark_sig::build(&k5, "(sum.n (tab (sum.n (tab (sub (mul (get.n xs i_n) 2) (get.n xs i2_n))))))").unwrap();
    assert_eq!(p.text, crate::futhark_sig::program(&k5, &sig, &tree5).unwrap(), "master 6 and master 5 render one program differently");
    let interp = crate::futhark_sig::Interp::new(&k5, &sig).unwrap();
    let dir = scratch("nested_binders");
    let exe = crate::futhark_sig::compile_c(&p.text, &dir).unwrap();
    let cases = crate::futhark_sig::load_cases(&problem("reduce").unwrap()).unwrap();
    for case in &cases {
        let want = interp.eval(&tree5, &case.inputs).unwrap();
        let got = run_entry(&exe, "main", &case.stdin).unwrap();
        assert_eq!(got, want, "{}", case.name);
    }
    std::fs::remove_dir_all(&dir).ok();
    eprintln!("nested binders: the interpreter and c agree on {} cases", cases.len());
}

/// A digest of the texts of 200 random genes of every view.
fn renderings_digest() -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (v, view6) in VIEWS6.iter().enumerate() {
        for (_, p) in random_programs(view6, 200, 0xd1ce_0000 + v as u64) {
            for b in p.text.bytes().chain([0u8]) {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    format!("{h:016x}")
}

/// THE RENDERED TEXT IS THE SAME ACROSS TWO PROCESSES: this test binary run again renders 200 random genes of every view
/// to the same bytes (no `HashMap` order, no address, no clock in the text).
#[test]
fn the_rendered_text_is_the_same_across_two_processes() {
    let mine = renderings_digest();
    if std::env::var("FULLER_F2_RENDER_CHILD").is_ok_and(|v| v == "1") {
        println!("RENDER6_DIGEST={mine}");
        return;
    }
    let out = crate::futhark_exec::output_with_timeout(
        Command::new(std::env::current_exe().unwrap()).args(["--exact", "futhark_table::tests6::the_rendered_text_is_the_same_across_two_processes", "--nocapture"]).env("FULLER_F2_RENDER_CHILD", "1"),
        Duration::from_secs(600),
        None,
    )
    .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let theirs = text.lines().find_map(|l| l.strip_prefix("RENDER6_DIGEST=")).unwrap_or_else(|| panic!("the child printed no digest:\n{text}"));
    assert_eq!(theirs, mine);
}

/// THE ADF RENDERING: a reference program with subtrees hoisted into `def adf_<i>` (sizes and free variables as
/// parameters, a nested ADF defined before the one that calls it) is accepted by Futhark and computes, on the `c`
/// backend, the same output bit for bit as the program without ADFs.
#[test]
fn a_program_with_adfs_computes_what_it_computes_without_them() {
    let picks: [(&str, &str, &[&[u32]]); 4] = [
        ("histogram", "reference", &[&[1], &[1, 1]]),
        ("pathfinder", "reference", &[&[2, 1], &[2, 1, 1, 0]]),
        ("kmeans", "reference", &[&[0, 1, 0], &[1, 2], &[0, 1, 0, 1, 0]]),
        ("a7", "reference", &[&[1], &[2, 0]]),
    ];
    for (problem6, form, paths) in picks {
        let r = REFERENCES6.iter().find(|r| r.problem == problem6 && r.form == form).unwrap();
        let (k, header) = view(problem6);
        let renderer = k.renderer6(header).unwrap();
        let tree = build6(&k, &(r.master6)()).unwrap();
        let paths: Vec<Vec<u32>> = paths.iter().map(|p| p.to_vec()).collect();
        let plain = renderer.program(&tree).unwrap();
        let folded = renderer.program_with_adfs(&tree, &paths).unwrap_or_else(|e| panic!("{problem6}: {e}"));
        assert_eq!(folded.text.matches("\ndef adf_").count() + usize::from(folded.text.starts_with("def adf_")), paths.len(), "{problem6}:\n{}", folded.text);
        assert!(folded.text.contains("adf_0") && plain.text != folded.text);
        let dir = scratch(&format!("adf_{problem6}"));
        let src = format!("{}{}\n{}{}\n", as_module(0, &plain.text), entry_calling(0, &plain.text), as_module(1, &folded.text), entry_calling(1, &folded.text));
        let exe = crate::futhark_sig::compile_c(&src, &dir).unwrap_or_else(|e| panic!("{problem6}:\n{}\n{e}", folded.text));
        for case in run_cases(problem6) {
            let a = run_entry(&exe, "e0", &case.stdin).unwrap();
            let b = run_entry(&exe, "e1", &case.stdin).unwrap();
            assert_eq!(a, b, "{problem6} on {}", case.name);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// THE ADF RENDERING OF ADAPTED PROGRAMS (F2 review finding 8): random k-means genes with adapters, each with the
/// subtree at an adapted node (and, where it has one, its parent) hoisted into an ADF, compute on the `c` backend the
/// same output bit for bit as without ADFs.
#[test]
fn an_adapted_program_with_adfs_computes_what_it_computes_without_them() {
    let (k, header) = view("kmeans");
    let renderer = k.renderer6(header).unwrap();
    let mut pairs: Vec<(Program6, Program6)> = Vec::new();
    for (tree, plain) in random_programs("kmeans", 200, 0xadf0_0006) {
        if pairs.len() == 6 {
            break;
        }
        let decoding = renderer.decode(&tree).unwrap();
        let hoistable = |path: &[u32]| -> bool {
            !path.is_empty() && decoding.node(path).and_then(|n| k.data_type(n.data_type)).is_some_and(|d| d.name != "size" && !matches!(d.kind, DataKind::Function { .. }))
        };
        let Some(at) = plain.adapters.iter().map(|(p, _)| p.clone()).find(|p| hoistable(p)) else { continue };
        let mut paths = vec![at.clone()];
        if hoistable(&at[..at.len() - 1]) {
            paths.push(at[..at.len() - 1].to_vec());
        }
        let folded = renderer.program_with_adfs(&tree, &paths).unwrap_or_else(|e| panic!("ADFs at {paths:?}: {e}\n{}", plain.text));
        assert!(folded.text.contains("def adf_0") && !folded.adapters.is_empty(), "{}", folded.text);
        pairs.push((plain, folded));
    }
    assert_eq!(pairs.len(), 6, "fewer than 6 of 200 random k-means genes have an adapter to hoist");
    let programs: Vec<&Program6> = pairs.iter().flat_map(|(a, b)| [a, b]).collect();
    for (c, outputs) in run_on_kmeans("adf_adapted", &programs).iter().enumerate() {
        for (i, pair) in outputs.chunks(2).enumerate() {
            assert_eq!(pair[0], pair[1], "pair {i} on case {c}:\n{}\n{}", pairs[i].0.text, pairs[i].1.text);
        }
    }
}

/// The k-means view's tables, with `edit` applied to one of them (by file name).
fn kmeans_tables_with(file: &str, edit: impl Fn(&str) -> String) -> Result<Kingdom6, String> {
    let v = crate::futhark_view::master6::committed("kmeans").unwrap();
    let (mut d, mut s, mut e, mut a) = (v.datatypes_tsv.clone(), v.symbols_tsv.clone(), v.environment_tsv.clone(), crate::futhark_view::master6::ADAPTERS6.to_string());
    match file {
        "datatypes" => d = edit(&d),
        "symbols" => s = edit(&s),
        "environment" => e = edit(&e),
        _ => a = edit(&a),
    }
    let k = Kingdom6::from_tables("kmeans_edited", &d, &s, &e, &a, &v.spec.literals)?;
    k.contract()?;
    Ok(k)
}

/// THE LOADER CHECKS WHAT IT USED TO SKIP (F1b review finding 5): every table's header, a non-function row with
/// inputs, a constant's properties with no constant, and the root and its variables against their data types; and
/// every view's entry, the `i32` + `f32` k-means view's among them, passes the entry check at load.
#[test]
fn the_loader_checks_headers_row_kinds_and_the_root() {
    let swap_header = |t: &str| {
        let (head, rest) = t.split_once('\n').unwrap();
        let mut cols: Vec<&str> = head.split('\t').collect();
        let n = cols.len();
        cols.swap(n - 2, n - 3);
        format!("{}\n{rest}", cols.join("\t"))
    };
    for file in ["datatypes", "symbols", "environment", "adapters"] {
        let e = kmeans_tables_with(file, swap_header).err().unwrap_or_else(|| panic!("{file}.tsv with two columns swapped loads"));
        assert!(e.contains("header"), "{file}: {e}");
    }
    let e = kmeans_tables_with("symbols", |t| t.replacen("\tterminal\ti64\t\t", "\tterminal\ti64\ti64\t", 1)).expect_err("a terminal with an input loads");
    assert!(e.contains("only a function takes inputs"), "{e}");
    let e = kmeans_tables_with("datatypes", |t| t.replacen("\tsize\t\t\t", "\tsize\t\tvalue=1\t", 1)).expect_err("a constant's properties with no constant load");
    assert!(e.contains("has no constant"), "{e}");
    let e = kmeans_tables_with("environment", |t| t.replacen("rows=k & cols=d\tout.rows", "rows=k\tout.rows", 1)).expect_err("a root without its cols loads");
    assert!(e.contains("the root"), "{e}");
    let e = kmeans_tables_with("environment", |t| t.replacen("zeros\tzeros_n\t[]f32\tlen=n", "zeros\tzeros_n\t[]f32\tlen=n & rows=n", 1)).expect_err("a zero array with a rows loads");
    assert!(e.contains("zeros_n"), "{e}");
    let e = kmeans_tables_with("environment", |t| t.replacen("size\tk\tsize\tvalue=k", "size\tk\tsize\tvalue=n", 1)).expect_err("a size of another value loads");
    assert!(e.contains("size k"), "{e}");
    let e = kmeans_tables_with("environment", |t| t.replacen("out.rows = k -> fit.root", "out.width = k -> fit.root", 1)).expect_err("a root requirement on a property it does not carry loads");
    assert!(e.contains("does not carry"), "{e}");
    for view6 in VIEWS6 {
        view(view6).0.contract().unwrap_or_else(|e| panic!("{view6}: {e}"));
    }
}
