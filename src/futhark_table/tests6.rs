//! Master 6's rendering (pass 5 by the decoder's template engine, plan section 7 task F2): the reference programs render
//! their published texts byte for byte, every repair of the design's section 5 table renders a program Futhark accepts,
//! and random genes of every view render programs Futhark accepts and runs.

use super::*;
use crate::futhark_sig::{build6, expected_text6, kingdom6, REFERENCES6};
use std::collections::{BTreeMap, BTreeSet};

/// A problem's kingdom.
fn view(problem: &str) -> Kingdom6 {
    kingdom6(problem).unwrap_or_else(|e| panic!("{problem}: {e}"))
}

/// EVERY REFERENCE RENDERS ITS PUBLISHED TEXT: each reference tree, built from its text form, decoded and rendered, is
/// byte for byte its stored program text (`tests/fixtures/futhark_m6_references.txt`), `0i64` indices included, and
/// needs no adapter.
#[test]
fn every_reference_renders_its_published_text_byte_for_byte() {
    for r in &REFERENCES6 {
        let key = r.key();
        let k = view(r.problem);
        let tree = build6(&k, &(r.master6)()).unwrap_or_else(|e| panic!("{key}: {e}"));
        let p = k.program(&tree).unwrap_or_else(|e| panic!("{key}: {e}"));
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
    let k = view(view6);
    let layout = GeneLayout::from_program_size(&k.contract, k.entry.output, program_size(view6)).unwrap();
    let consts = const_len6(&k);
    let mut rng = GeneRng::new(seed);
    (0..n)
        .map(|i| {
            let gene = random_gene(&k.contract, &layout, &consts, &mut rng).unwrap();
            let tree = express(&k.contract, &layout, &gene, &consts).unwrap_or_else(|e| panic!("{view6} gene {i}: {e:?}")).tree();
            let p = k.program(&tree).unwrap_or_else(|e| panic!("{view6} gene {i}: our software has a bug: {e}"));
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
        let called: usize = programs.iter().map(|(_, p)| p.repairs()).sum();
        let substituted: usize = programs.iter().map(|(_, p)| p.substitutions.len()).sum();
        let (mut by_adapter, mut by_cause): (BTreeMap<String, usize>, BTreeMap<&str, usize>) = (BTreeMap::new(), BTreeMap::new());
        for (_, p) in &programs {
            for (_, a, placed) in &p.adapters {
                *by_adapter.entry(format!("{a} {}", if *placed { "placed" } else { "called" })).or_default() += 1;
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
            "{view6}: {} of {n} accepted by futhark check; adapter-free genes {free} of {n} ({:.1} %); adapters a repair called {called} in {nodes} nodes; adapters {by_adapter:?}; substitution-free genes {unsubstituted} of {n} ({:.1} %); substituted nodes {substituted} of {nodes} ({:.2} %); substitutions {by_cause:?}\n",
            n - bad.len(),
            100.0 * free as f64 / n as f64,
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
    let k = view(view6);
    let mut names: BTreeSet<String> = k.spec.env.iter().filter(|l| l.kind == "size").map(|l| l.name.clone()).collect();
    for l in k.spec.env.iter().filter(|l| l.kind == "input") {
        names.extend(l.dims.iter().cloned());
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
/// one. The values are compared with the interpreter elsewhere (`futhark_sig::tests_interp6`).
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
fn render(k: &Kingdom6, text: &str) -> Program6 {
    let tree = build6(k, text).unwrap_or_else(|e| panic!("{text}: {e}"));
    k.program(&tree).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// The adapters a repair called in a program, by name.
fn called(p: &Program6) -> Vec<&str> {
    p.adapters.iter().filter(|a| !a.2).map(|a| a.1.as_str()).collect()
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

/// The k-means view's expressions that fail one guard each, with the adapter its repair calls (design section 5's
/// table: a slot of another length is `tabrows`' body, fitted to its size argument), as (name, adapter, data type tag,
/// s-expression). `[]f32` of lengths `n`, `k`, `d` are the zero arrays; the case's sizes differ.
const ADAPTER_ROWS: [(&str, &str, &str, &str); 6] = [
    ("map2 of two lengths", "fit@a.f32", "a.f32", "(map2.add@a.f32 zeros_n zeros_k)"),
    ("a slot of another length", "fit@a.f32", "m.f32", "(tabrows@m.f32[r] k d zeros_n)"),
    ("an index of another bound", "clampix@i64", "f32", "(get@f32 zeros_n (last@i64 k))"),
    ("a loop body of another length", "fit@a.f32", "a.f32", "(loop@a.f32[s,i] n zeros_k zeros_n)"),
    ("matmul's inner sizes", "fitcols@m.f32", "m.f32", "(matmul@m.f32 points centres)"),
    ("map2_rows' rows and columns", "fitrows@m.f32", "m.f32", "(map2_rows@m.f32 add@fn.arr2_f32_arr_f32_arr_f32 points (transpose@m.f32 centres))"),
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

/// The expression nested three deep in its own repaired operand: each repair's operand is the one before (a loop's
/// through its state, not its body, so no level adds a trip; `tabrows` has no other operand than its body, so its third
/// level sits at the work bound and is substituted).
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

/// THE DESIGN'S ADAPTER TABLE AS FIXTURES (design section 5): each guard that fails is repaired by a call of its
/// adapter, and `futhark check` accepts the program at the root, inside a `tabulate` lambda, inside a loop body, and
/// with the repair nested three deep in its own operand. The entry's result is repaired at the root: a program of
/// `[n][d]` (its rows fitted) and one of `[d][n]` (its rows and columns) where the entry returns `[k][d]`.
#[test]
fn every_adapter_of_the_design_table_renders_a_program_futhark_accepts() {
    let k = view("kmeans");
    let root = |arr: &str| format!("(let.a.f32@m.f32[w] {arr} centres)");
    let mut programs = Vec::new();
    for (name, adapter, tag, e) in ADAPTER_ROWS {
        let p = render(&k, &root(&as_array(tag, e)));
        assert!(called(&p).contains(&adapter), "{name}: calls {:?}, not {adapter}", p.adapters);
        programs.push((format!("{name} at the root"), p));
        let contexts = [
            ("inside a tabulate lambda", root(&format!("(tab@a.f32[t] n {})", as_scalar(tag, e)))),
            ("inside a loop body", root(&format!("(loop@a.f32[s,i] n zeros_n (map2.add@a.f32 s {}))", as_array(tag, e)))),
            ("nested three deep", root(&as_array(tag, &nested3(name, e)))),
        ];
        for (place, text) in contexts {
            let p = render(&k, &text);
            assert!(called(&p).contains(&adapter), "{name} {place}: calls {:?}, not {adapter}", p.adapters);
            programs.push((format!("{name} {place}"), p));
        }
    }
    for (name, text, want) in [("the entry's rows", "points", vec!["fitrows@m.f32"]), ("the entry's rows and columns", "(transpose@m.f32 points)", vec!["fitrows@m.f32", "fitcols@m.f32"])] {
        let p = render(&k, text);
        assert!(p.adapters.iter().all(|(path, _, placed)| path.is_none() && !placed), "{name}: {:?}", p.adapters);
        assert_eq!(called(&p), want, "{name}");
        programs.push((name.to_string(), p));
    }
    let h = view("histogram");
    let p = render(&h, "(hist.add@a.i32 k zeros_n zeros_k)");
    assert_eq!(p.adapters, vec![(Some(Vec::new()), "fit@a.i32".to_string(), false); 2], "both operands fitted to min(k, n), which neither length is as written");
    programs.push(("hist of two lengths".to_string(), p));
    // An adapter evolution placed renders as its symbol's spelling, with its prelude.
    let p = render(&k, "(let.a.f32@m.f32[w] (fit@a.f32 d zeros_n) centres)");
    assert_eq!(p.adapters, vec![(Some(vec![0]), "fit@a.f32".to_string(), true)]);
    assert!(p.text.contains("fit d 0f32 (zeros_n)") && p.text.starts_with("def fit "), "{}", p.text);
    programs.push(("a placed fit".to_string(), p));
    assert_accepted("adapter_table", &programs);
}

/// TEXT LINEAR IN THE DEPTH (adversarial H1): ten `map2` adapters, each the operand of the next, grow the text by a
/// bounded amount a level (never doubling), and Futhark accepts the deepest.
#[test]
fn ten_nested_map2_adapters_give_text_linear_in_the_depth() {
    let k = view("kmeans");
    let zeros = ["zeros_k", "zeros_d", "zeros_n"];
    let mut e = "zeros_n".to_string();
    let mut lengths = Vec::new();
    let mut last = None;
    for level in 0..10 {
        e = format!("(map2.add@a.f32 {e} {})", zeros[level % 3]);
        let p = render(&k, &format!("(let.a.f32@m.f32[w] {e} centres)"));
        assert!(p.repairs() > level, "level {level}: {:?}", p.adapters);
        lengths.push(p.text.len());
        last = Some(p);
    }
    let steps: Vec<usize> = lengths.windows(2).map(|w| w[1] - w[0]).collect();
    let most = *steps.iter().max().unwrap();
    assert!(most < 120 && lengths[9] < lengths[0] + 9 * most, "the text grows by {steps:?} a level ({lengths:?})");
    eprintln!("ten nested map2 adapters: {lengths:?} bytes, a level adds {steps:?}");
    assert_accepted("linear_depth", &[("depth 10".to_string(), last.unwrap())]);
}

/// NESTED BINDERS OF ONE DATA TYPE AND PROPERTIES HAVE DISTINCT NAMES (`i_n`, `i2_n`: the engine's `{nth}` name
/// hygiene), and the interpreter and the `c` backend give the same value bit for bit on every stored case.
#[test]
fn nested_binders_render_distinct_names_and_the_interpreter_and_c_agree() {
    let k = view("reduce");
    let m6 = "(sum@i32 (tab@a.i32[i] n (sum@i32 (tab@a.i32[j] n (sub@i32 (mul@i32 (get@i32 xs i) 2) (get@i32 xs j))))))";
    let p = render(&k, m6);
    assert!(p.adapter_free(), "{:?}", p.adapters);
    assert!(p.text.contains("(\\i_n -> ") && p.text.contains("(\\i2_n -> "), "{}", p.text);
    let tree = build6(&k, m6).unwrap();
    let interp = crate::futhark_sig::Interp::new6(&std::sync::Arc::new(k.clone()), &crate::futhark_sig::signature(problem("reduce").unwrap().name).unwrap()).unwrap();
    let dir = scratch("nested_binders");
    let exe = crate::futhark_sig::compile_c(&p.text, &dir).unwrap();
    let cases = crate::futhark_sig::load_cases(&problem("reduce").unwrap()).unwrap();
    for case in &cases {
        let want = interp.eval(&tree, &case.inputs).unwrap();
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

