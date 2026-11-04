//! THE VIEW REGRESSION (`docs/PLAN_futhark_kingdom.md`, design D6 and 3g): each problem's kingdom, computed at
//! runtime from the master table and its view specification, is the kingdom its written-out table (`View::write`,
//! read back from the files; the text is pinned by its fingerprint in `futhark_view::tests`) loads: the same duals and rows, so the same
//! contract, constant lengths and alphabet at every cell of a layout, so the same genes express the same trees and the
//! same Futhark programs; and its signature is the one the old `interface.tsv` declared.

use super::*;
use crate::typed_gep::express;

/// The problems whose view is written out and loaded back.
const WRITTEN: [&str; 4] = ["pathfinder", "reduce", "scan", "histogram"];

/// A view's tables as `View::write` writes them, read back from the files: the written-out table (the option the plan
/// keeps), which the generator's witnesses check (`gen_tables.py --views DIR`).
fn written_tables(name: &str, view: &crate::futhark_view::View) -> (String, String) {
    let dir = std::env::temp_dir().join(format!("futhark_sig_written_{name}_{}", std::process::id()));
    view.write(&dir).unwrap();
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
    let out = (read("types.tsv"), read("functions.tsv"));
    std::fs::remove_dir_all(&dir).unwrap();
    out
}

fn p(name: &str, elem: &str, dims: &[&str]) -> Param {
    Param { name: name.into(), elem: elem.into(), dims: dims.iter().map(|d| d.to_string()).collect() }
}

/// The signatures the hand-written `interface.tsv` files declared (commit d26a018a), restated.
fn declared(name: &str) -> Signature {
    let strs = |v: &[&str]| v.iter().map(|d| d.to_string()).collect::<Vec<_>>();
    match name {
        "pathfinder" => Signature { inputs: vec![p("walls", "i32", &["rows", "cols"])], output: p("main", "i32", &["cols"]), depth: 2, dims: strs(&["rows", "cols"]), size_inputs: vec![], elements: strs(&["i32"]) },
        "reduce" => Signature { inputs: vec![p("xs", "i32", &["n"])], output: p("main", "i32", &[]), depth: 2, dims: strs(&["n"]), size_inputs: vec![], elements: strs(&["i32"]) },
        "scan" => Signature { inputs: vec![p("xs", "i32", &["n"])], output: p("main", "i32", &["n"]), depth: 2, dims: strs(&["n"]), size_inputs: vec![], elements: strs(&["i32"]) },
        "histogram" => Signature {
            inputs: vec![p("k", "i32", &[]), p("is", "i32", &["n"]), p("vs", "i32", &["n"])],
            output: p("main", "i32", &["k"]),
            depth: 2,
            dims: strs(&["k", "n"]),
            size_inputs: strs(&["k"]),
            elements: strs(&["i32"]),
        },
        other => panic!("no declared signature for {other}"),
    }
}

#[test]
fn every_view_is_the_written_kingdom_and_the_same_genes_express_the_same_programs() {
    for name in WRITTEN {
        let prob = problem(name).unwrap();
        let k = load(&prob.instance).unwrap();
        let (types, functions) = written_tables(name, prob.instance.view().unwrap());
        let written = FutharkKingdom::from_tables(crate::futhark_table::FutharkFamily::Shape, &k.kingdom, &types, &functions).unwrap();
        let sig = signature(&prob.instance).unwrap();
        assert_eq!(*sig, declared(name), "{name}: the view's signature is not the declared one");
        assert_eq!(k.duals, written.duals, "{name}: the duals differ");
        assert_eq!(k.rows, written.rows, "{name}: the rows differ");
        assert_eq!(format!("{:?}", k.relations()), format!("{:?}", written.relations()), "{name}: the contract relations differ");
        let const_len = k.const_len();
        assert_eq!(const_len, written.const_len(), "{name}: the constant lengths differ");
        let (ca, cb) = (k.relations().check().unwrap(), written.relations().check().unwrap());
        let root = k.roots()[0];
        assert_eq!(vec![root], written.roots(), "{name}: the roots differ");
        let layout = GeneLayout::from_program_size(&ca, root, 13).unwrap();
        let layout_b = GeneLayout::from_program_size(&cb, root, 13).unwrap();
        assert_eq!(format!("{layout:?}"), format!("{layout_b:?}"), "{name}: the layouts differ");
        for pos in 0..layout.width {
            assert_eq!(layout.alphabet(&ca, pos, &const_len), layout_b.alphabet(&cb, pos, &const_len), "{name}: the alphabet of cell {pos} differs");
        }
        // Every row renders the same text: a leaf as itself, a constant at every pool index, a function over leaf
        // children of its argument types.
        let leaf_of = |kk: &FutharkKingdom, ty: Ty| kk.rows.iter().position(|r| r.output == ty && r.inputs.is_empty() && r.kind == FutharkKind::Terminal).map(|i| Tree { symbol: i as u32, const_index: None, kids: vec![] });
        for (i, r) in k.rows.iter().enumerate() {
            let trees: Vec<Tree> = match r.kind {
                FutharkKind::Const => (0..const_len[&r.output]).map(|c| Tree { symbol: i as u32, const_index: Some(c), kids: vec![] }).collect(),
                _ => vec![Tree { symbol: i as u32, const_index: None, kids: r.inputs.iter().map(|t| leaf_of(&k, *t).unwrap_or_else(|| panic!("{name}: no terminal of {t:?}"))).collect() }],
            };
            for t in trees {
                assert_eq!(k.render(&t).unwrap(), written.render(&t).unwrap(), "{name}: the row {} renders differently", r.name);
            }
        }
        // The reference and 200 random genes: the same trees, the same programs.
        let reference = build(&k, prob.reference().unwrap()).unwrap();
        assert_eq!(program(&k, &sig, &reference).unwrap(), program(&written, &sig, &build(&written, prob.reference().unwrap()).unwrap()).unwrap(), "{name}: the reference's program differs");
        let mut rng = GeneRng::new(0x71E3_0001);
        for g in 0..200 {
            let gene = random_gene(&ca, &layout, &const_len, &mut rng).unwrap();
            let ta = express(&ca, &layout, &gene, &const_len).unwrap().tree();
            let tb = express(&cb, &layout_b, &gene, &const_len).unwrap().tree();
            assert_eq!(ta, tb, "{name}: gene {g} expresses differently");
            assert_eq!(program(&k, &sig, &ta).unwrap(), program(&written, &sig, &tb).unwrap(), "{name}: gene {g}'s program differs");
        }
        // The CPU interpreter on the view's kingdom: the reference is exact on every stored case.
        let interp = Interp::new(&k, &sig).unwrap();
        for c in load_cases(&prob).unwrap() {
            assert_eq!(interp.eval(&reference, &c.inputs).unwrap(), *c.answer().unwrap(), "{name}: the reference on {}", c.name);
        }
    }
}

/// THE PROGRAMS BEFORE BIDIRECTIONAL REACHABILITY (`docs/PLAN_futhark_kingdom.md`, 3e): each problem's reference and
/// planted slowdown, and A7's reference (the product twice, at n = 2, m = 3, p = 4), written as Futhark programs by
/// the views of commit 97dbbb80, when a signature view still admitted only the families its source used. Since the
/// contexts became sets (3g) a bound variable is named by its class, so a signature program's text differs from the
/// record in its binder names ONLY (checked token by token, one new name an old name, one old name a new name), and
/// the program's values are exact on every stored case (host interpreter). A7's matrix view has no binders: its
/// program is the record byte for byte.
const RECORDED: [(&str, &str, &str); 4] = [
    ("pathfinder", include_str!("../../tests/fixtures/futhark_view_programs/pathfinder.reference.fut"), include_str!("../../tests/fixtures/futhark_view_programs/pathfinder.planted.fut")),
    ("reduce", include_str!("../../tests/fixtures/futhark_view_programs/reduce.reference.fut"), include_str!("../../tests/fixtures/futhark_view_programs/reduce.planted.fut")),
    ("scan", include_str!("../../tests/fixtures/futhark_view_programs/scan.reference.fut"), include_str!("../../tests/fixtures/futhark_view_programs/scan.planted.fut")),
    ("histogram", include_str!("../../tests/fixtures/futhark_view_programs/histogram.reference.fut"), include_str!("../../tests/fixtures/futhark_view_programs/histogram.planted.fut")),
];

/// Whether two Futhark texts differ only in binder names: equal token for token, except where the old token is a
/// stack-context binder name (`j1`, `i2`, `s1`, `t1`) and the new one a count-context name (`i_D`, `i2_D`, `ti_D`, `s_C`), with
/// one consistent renaming both ways. Returns the renaming.
fn binder_renaming(old: &str, new: &str) -> Option<BTreeMap<String, String>> {
    let toks = |t: &str| {
        let mut v = Vec::new();
        let mut cur = String::new();
        for c in t.chars() {
            if c.is_ascii_alphanumeric() || c == '_' {
                cur.push(c);
            } else {
                if !cur.is_empty() {
                    v.push(std::mem::take(&mut cur));
                }
                if !c.is_whitespace() {
                    v.push(c.to_string());
                }
            }
        }
        if !cur.is_empty() {
            v.push(cur);
        }
        v
    };
    let (a, b) = (toks(old), toks(new));
    if a.len() != b.len() {
        return None;
    }
    let is_old = |x: &str| x.strip_prefix(['i', 'j', 's', 't']).is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
    let is_new = |x: &str| x.strip_prefix('t').unwrap_or(x).strip_prefix(['i', 's']).is_some_and(|r| r.trim_start_matches(|c: char| c.is_ascii_digit()).starts_with('_'));
    let (mut fwd, mut back): (BTreeMap<String, String>, BTreeMap<String, String>) = (BTreeMap::new(), BTreeMap::new());
    for (x, y) in a.iter().zip(&b) {
        if x == y {
            continue;
        }
        if !(is_old(x) && is_new(y)) || fwd.insert(x.clone(), y.clone()).is_some_and(|p| p != *y) || back.insert(y.clone(), x.clone()).is_some_and(|p| p != *x) {
            return None;
        }
    }
    Some(fwd)
}

#[test]
fn the_reference_programs_differ_from_the_recorded_ones_in_binder_names_only_and_are_exact() {
    for (name, reference, planted) in RECORDED {
        let prob = problem(name).unwrap();
        let k = load(&prob.instance).unwrap();
        let sig = signature(&prob.instance).unwrap();
        let interp = Interp::new(&k, &sig).unwrap();
        let cases = load_cases(&prob).unwrap();
        for (what, text, recorded) in [("reference", prob.reference().unwrap(), reference), ("planted slowdown", prob.planted().unwrap(), planted)] {
            let t = build(&k, text).unwrap();
            let got = program(&k, &sig, &t).unwrap();
            let renaming = binder_renaming(recorded, &got).unwrap_or_else(|| panic!("{name}: the {what}'s program differs from the record in more than binder names:\n{recorded}\n{got}"));
            eprintln!("{name} {what}: renamed {renaming:?}");
            for c in &cases {
                assert_eq!(interp.eval(&t, &c.inputs).unwrap(), *c.answer().unwrap(), "{name}: the {what} on {}", c.name);
            }
        }
    }
    let k = FutharkKingdom::load().unwrap();
    let sizes: BTreeMap<String, usize> = [("n", 2), ("m", 3), ("p", 4)].into_iter().map(|(a, b)| (a.to_string(), b)).collect();
    let got = crate::futhark::program(&k, k.roots()[0], &crate::futhark_search::product_twice(&k), &sizes).unwrap();
    assert_eq!(got, include_str!("../../tests/fixtures/futhark_view_programs/a7.reference.fut"), "A7: the reference's program changed");
}

#[test]
fn the_binder_renaming_check_finds_a_difference_that_is_not_a_name() {
    assert!(binder_renaming("tabulate n (\\j1 -> xs[j1])", "tabulate n (\\i_n -> xs[i_n])").is_some());
    assert!(binder_renaming("tabulate n (\\j1 -> xs[j1])", "tabulate n (\\i_n -> ys[i_n])").is_none(), "a changed input");
    assert!(binder_renaming("f j1 j2", "f i_n i_n").is_none(), "two old names to one new name");
    assert!(binder_renaming("f j1 j1", "f i_n i_k").is_none(), "one old name to two new names");
}

/// The gene layouts of `docs/PLAN_futhark_kingdom.md` 3e, at the runner's default program size (the reference's
/// function count): the head length (the sum of the per-type heads) and the gene width in words. Before 3e (commit
/// 97dbbb80): pathfinder 14065 and 82573, reduce 1 and 13, scan 1 and 9, histogram 2506 and 13945; after 3e (stack
/// contexts) pathfinder 14065 and 84015, histogram 2506 and 13945. Count contexts (3g): these.
#[test]
fn the_gene_layouts_are_the_measured_ones() {
    let got: Vec<(&str, usize, usize, usize)> = ["pathfinder", "reduce", "scan", "histogram"]
        .into_iter()
        .map(|name| {
            let prob = problem(name).unwrap();
            let k = load(&prob.instance).unwrap();
            let reference = build(&k, prob.reference().unwrap()).unwrap();
            let n = reference.level_order().iter().filter(|(t, _)| !k.rows[t.symbol as usize].inputs.is_empty()).count();
            let layout = GeneLayout::from_program_size(&k.relations().check().unwrap(), k.roots()[0], n).unwrap();
            (name, n, layout.head_lens().values().sum::<usize>(), layout.width)
        })
        .collect();
    assert_eq!(got, [("pathfinder", 13, 3289, 22261), ("reduce", 1, 1, 11), ("scan", 1, 1, 8), ("histogram", 4, 721, 4655)]);
}

/// THE PROGRAM SIZE IS AN EXPERIMENT PARAMETER (`docs/reviews/2026-11-01_futhark_kingdom_design_fable.md`, F3): the
/// runner's default `DEFAULT_PROGRAM_SIZE` is above every problem's reference and planted slowdown, and both fit its
/// layout, so a run that starts from either (regime P, the slope check) needs no `--program-size`.
#[test]
fn the_default_program_size_is_above_every_reference_and_both_programs_fit() {
    let mut got = Vec::new();
    for name in ["pathfinder", "reduce", "scan", "histogram"] {
        let prob = problem(name).unwrap();
        let k = load(&prob.instance).unwrap();
        let contract = k.relations().check().unwrap();
        let layout = GeneLayout::from_program_size(&contract, k.roots()[0], DEFAULT_PROGRAM_SIZE).unwrap();
        for text in [prob.reference().unwrap(), prob.planted().unwrap()] {
            let t = build(&k, text).unwrap();
            assert!(function_count(&k, &t) < DEFAULT_PROGRAM_SIZE, "{name}: {} functions, the default program size is {DEFAULT_PROGRAM_SIZE}", function_count(&k, &t));
            crate::typed_gep::fits(&contract, &layout, &t).unwrap_or_else(|e| panic!("{name}: {text} does not fit N = {DEFAULT_PROGRAM_SIZE}: {e}"));
        }
        assert!(layout.guaranteed_program_size(&contract) >= DEFAULT_PROGRAM_SIZE, "{name}: the layout guarantees fewer functions than asked");
        got.push((name, layout.head_lens().values().sum::<usize>(), layout.width));
    }
    assert_eq!(got, [("pathfinder", 8495, 57487), ("reduce", 1551, 10025), ("scan", 1551, 10021), ("histogram", 7441, 47999)], "the head lengths and gene widths at N = {DEFAULT_PROGRAM_SIZE}");
}

// ---- THE PAIR CONSTRUCTS (docs/PLAN_futhark_kingdom.md, step 4c (b)) ---------------------------------------------
//
// Two test views over two elements (f32 and i32), sniffed from `kingdoms/futhark/sig/kahan/kahan.fut`: the kingdom
// must express UltraDim's k-means constructs construct for construct (`update_centroids`: a sequential loop whose state
// is the pair (sum, compensation), each step guarded by the membership; `assign_points`: the pair (best distance, best
// index) from the first distance over the rest). The programs below are those constructs as genes; `futhark c`
// computes the same bits as the yardstick entries of `kahan.fut` and as the host interpreter.

/// The Kahan view: `(xs: [n]f32) (ms: [n]i32) -> f32`.
const KAHAN: Instance = Instance { name: "kahan", spec: include_str!("../../kingdoms/futhark/sig/kahan/view.tsv") };
/// The argmin view: `(ds: [n]f32) -> i32`.
const ARGMIN: Instance = Instance { name: "argmin", spec: include_str!("../../kingdoms/futhark/sig/argmin/view.tsv") };
/// The cell view (master version 5, the triple): `(xs: [n]f32) (ms: [n]i32) (olds: [n]f32) -> f32`.
const CELL: Instance = Instance { name: "cell", spec: include_str!("../../kingdoms/futhark/sig/cell/view.tsv") };
/// The kernels view (master version 5, the kernel boundary): `(xs: [n]f32) -> f32`, `lets a.i32.n`.
const KERNELS: Instance = Instance { name: "kernels", spec: include_str!("../../kingdoms/futhark/sig/kernels/view.tsv") };

/// UltraDim's `update_centroids` for one cell, as a gene: the Kahan-compensated sum of the `xs` whose membership is 0
/// (`|m| - 1 < 0`, that is `m == 0`), the state the pair (sum, compensation) destructured into `s_f32` and `s2_f32`,
/// each step `y = x - c; t = s + y; (t, (t - s) - y)` (the kingdom has no `let`: `t` and `y` are written out), over the
/// count of those members (a second loop, UltraDim's third state), the count made an f32 by the conversion row.
const KAHAN_MEAN: &str = "(div (fst.f32.f32 (loop.n zero (psel.i32 (sub (abs (get.n ms i_n)) 1)
    (pair (add s_f32 (sub (get.n xs i_n) s2_f32))
          (sub (sub (add s_f32 (sub (get.n xs i_n) s2_f32)) s_f32) (sub (get.n xs i_n) s2_f32)))
    (pair s_f32 s2_f32))))
  (cvt.i32 (loop.n 0 (sel (sub (abs (get.n ms i_n)) 1) (add s_i32 1) s_i32))))";

/// UltraDim's `assign_points` for one point, as a gene: the pair (best distance, best index) starts at the first
/// distance and index 0, and over the indices 1 .. n-1 a distance replaces it only when strictly smaller (`d - best <
/// 0`, which is `d < best` for every pair of f32 values, the infinities and a NaN included); the index is the result.
const ARGMIN_FIRST: &str = "(snd.f32.i32 (loop1.n (pair (get.n ds zero) 0)
  (psel.f32 (sub (get.n ds i_n) s_f32) (pair (get.n ds i_n) (ival.n i_n)) (pair s_f32 s_i32))))";

/// UltraDim's `update_centroids` for one cell, as a gene, in ONE loop as the shader has it (step 4c (c)): the state
/// is the TRIPLE (sum, compensation, count), destructured into `s_f32`, `s2_f32` and `s_i32`; a point whose
/// membership is 0 (`|m| - 1 < 0`) steps `y = x - c; t = s + y; (t, (t - s) - y, count + 1)`, another leaves the
/// triple; then `if (count > 0u)` (as `-(f32 count) < 0`) the sum over the count, else the previous centre. The loop is
/// written three times (the kingdom has no `let` below the top; the compiler's CSE computes it once).
const CELL_UPDATE: &str = "(sel (neg (cvt.i32 (thd.f32.f32.i32 LOOP)))
  (div (fst.f32.f32.i32 LOOP) (cvt.i32 (thd.f32.f32.i32 LOOP)))
  (get.n olds zero))";
const CELL_LOOP: &str = "(loop.n zero (psel.i32 (sub (abs (get.n ms i_n)) 1)
    (triple (add s_f32 (sub (get.n xs i_n) s2_f32))
            (sub (sub (add s_f32 (sub (get.n xs i_n) s2_f32)) s_f32) (sub (get.n xs i_n) s2_f32))
            (add s_i32 1))
    (triple s_f32 s2_f32 s_i32)))";

/// THE KERNEL BOUNDARY as a gene: one kernel writes a buffer the next one reads. The first kernel flags each point
/// (`x - 0.5 < 0` is 0, else 1, an i32 by the exact conversion of 0 or 1); the `let` names it at the top; the second
/// kernel is a loop over the points reading the flag (`flag == 1` as `-(f32 flag) < 0`).
const KERNELS_PROGRAM: &str = "(let (tab (cvt.f32 (sel (sub (get.n xs i_n) 0.5) 0 1)))
  (loop.n 0 (sel (neg (cvt.i32 (get.n s_a_i32_n i_n))) (add s_f32 (get.n xs i_n)) s_f32)))";

/// The cell program with its loop written in.
fn cell_update() -> String {
    CELL_UPDATE.replace("LOOP", CELL_LOOP)
}

fn have_futhark() -> bool {
    std::process::Command::new("futhark").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn scratch(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("futhark_sig_pairs_{name}_{}", std::process::id()))
}

/// `futhark check` on a program: Futhark's type checker accepts it.
fn futhark_check(src: &str, dir: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let fut = dir.join("prog.fut");
    std::fs::write(&fut, src).map_err(|e| format!("write {}: {e}", fut.display()))?;
    let out = crate::futhark_exec::output_with_timeout(std::process::Command::new("futhark").arg("check").arg(&fut), std::time::Duration::from_secs(120), None)?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!("futhark check:\n{}", String::from_utf8_lossy(&out.stderr)))
    }
}

/// A compiled executable's one output for an entry point, on binary input.
fn run_entry(exe: &std::path::Path, entry: &str, stdin: &[u8]) -> Result<Array, String> {
    let out = crate::futhark_exec::output_with_timeout_bytes(std::process::Command::new(exe).arg("-b").arg("-e").arg(entry), std::time::Duration::from_secs(120), Some(stdin))?;
    if !out.status.success() {
        return Err(format!("{} -e {entry} failed:\n{}", exe.display(), String::from_utf8_lossy(&out.stderr)));
    }
    read_one(&out.stdout)
}

/// Random f32 values spanning magnitudes (10^-3 to 10^5, a few large among many small, both signs), and memberships in
/// {-1, 0, 1, 2}.
fn pair_inputs(n: usize, rng: &mut GeneRng) -> (Array, Array) {
    let scales = [1e-3f32, 1.0, 1e3, 1e5];
    let xs: Vec<f32> = (0..n).map(|_| ((rng.below(1 << 24) as f32) / 16_777_216.0 - 0.5) * scales[if rng.below(8) == 0 { 3 } else { rng.below(3) }]).collect();
    let ms: Vec<i32> = (0..n).map(|_| rng.below(4) as i32 - 1).collect();
    (Array::f32s(vec![n], xs), Array::i32s(vec![n], ms))
}

#[test]
fn the_pair_views_load_and_every_row_has_a_meaning() {
    for (inst, root, functions) in [(&KAHAN, "f32:c", 2196), (&ARGMIN, "i32:c", 2196), (&CELL, "f32:c", 6749), (&KERNELS, "f32:c", 12371)] {
        let k = load(inst).unwrap();
        let sig = signature(inst).unwrap();
        assert_eq!(sig.elements, vec!["f32".to_string(), "i32".to_string()]);
        assert_eq!(k.roots(), vec![k.ty(root).unwrap()], "{}", inst.name);
        let contract = k.relations().check().unwrap_or_else(|d| panic!("{}: the kingdom check: {:?}", inst.name, &d[..d.len().min(3)]));
        // Every row of the view has a meaning (`Interp::new` refuses one that has none).
        Interp::new(&k, &sig).unwrap_or_else(|e| panic!("{}: {e}", inst.name));
        assert_eq!(k.rows.iter().filter(|r| r.kind == FutharkKind::Function).count(), functions, "{}", inst.name);
        let layout = GeneLayout::from_program_size(&contract, k.roots()[0], DEFAULT_PROGRAM_SIZE).unwrap();
        eprintln!("{}: {} types, {} rows; N = {DEFAULT_PROGRAM_SIZE}: {} words a gene, head {}", inst.name, k.duals.len(), k.rows.len(), layout.width, layout.head_lens().values().sum::<usize>());
    }
}

/// The two constructs, as genes, compute what `futhark c` computes of the yardstick entries (`kahan.fut`, spelled
/// with Futhark's own tuple loop, a triple state and a range loop) and of the kingdom's own rendering, bit for bit,
/// on inputs spanning magnitudes, n = 1 included; the host interpreter agrees with both.
#[test]
fn the_kahan_mean_and_the_argmin_compute_what_futhark_c_computes() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let base = scratch("constructs");
    let yard = compile_c(include_str!("../../kingdoms/futhark/sig/kahan/kahan.fut"), &base.join("yardstick")).unwrap();
    let mut rng = GeneRng::new(0x4A1A_0001);
    let mut checked = 0;
    // The Kahan view.
    let k = load(&KAHAN).unwrap();
    let sig = signature(&KAHAN).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let tree = build(&k, KAHAN_MEAN).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(sexp(&k, &tree).split_whitespace().collect::<String>(), KAHAN_MEAN.split_whitespace().collect::<String>(), "the Kahan mean reads back");
    let src = program(&k, &sig, &tree).unwrap();
    assert!(src.contains("loop ((s_f32, s2_f32): (f32, f32)) = ((0f32, 0f32)) for i_n < n do"), "{src}");
    assert!(src.contains("let zeros_f32_n = replicate n 0f32") == k.rows.iter().any(|r| r.spelling == "zeros_f32_n"));
    let exe = compile_c(&src, &base.join("kahan")).unwrap_or_else(|e| panic!("{e}\n{src}"));
    for n in [1usize, 2, 7, 64, 1000] {
        for round in 0..3 {
            let (xs, mut ms) = pair_inputs(n, &mut rng);
            // Round 1: every value a member; round 2: none (a count of 0: 0 / 0 is a NaN, which agrees with a NaN).
            if round == 1 {
                ms = Array::i32s(vec![n], vec![0; n]);
            } else if round == 2 {
                ms = Array::i32s(vec![n], vec![1; n]);
            }
            let stdin = write_binary(&[&xs, &ms]);
            let want = run_entry(&yard, "kahan", &stdin).unwrap();
            let got = run_exe(&exe, &stdin).unwrap();
            let host = interp.eval(&tree, &[xs.clone(), ms.clone()]).unwrap();
            assert_eq!(got, want, "the Kahan mean compiled, n = {n}, round {round}: {:?} vs the yardstick's {:?}", got.f32_values(), want.f32_values());
            assert_eq!(host, want, "the Kahan mean interpreted, n = {n}, round {round}");
            checked += 1;
        }
    }
    // The argmin view.
    let k = load(&ARGMIN).unwrap();
    let sig = signature(&ARGMIN).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let tree = build(&k, ARGMIN_FIRST).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(sexp(&k, &tree).split_whitespace().collect::<String>(), ARGMIN_FIRST.split_whitespace().collect::<String>(), "the argmin reads back");
    let src = program(&k, &sig, &tree).unwrap();
    assert!(src.contains("loop ((s_f32, s_i32): (f32, i32)) = ") && src.contains(" for ti_n < n - 1 do let i_n = ti_n + 1 in "), "{src}");
    let exe = compile_c(&src, &base.join("argmin")).unwrap_or_else(|e| panic!("{e}\n{src}"));
    for n in [1usize, 2, 7, 64, 1000] {
        for round in 0..3 {
            let (mut ds, _) = pair_inputs(n, &mut rng);
            // Round 1: distances from a set of four, so the least ties and the first wins; round 2: a NaN among them.
            if round == 1 {
                ds = Array::f32s(vec![n], (0..n).map(|_| [0.5f32, 1.0, 0.5, 2.0][rng.below(4)]).collect());
            } else if round == 2 && n > 1 {
                let mut v = ds.f32_values().unwrap();
                v[rng.below(n)] = f32::NAN;
                ds = Array::f32s(vec![n], v);
            }
            let stdin = write_binary(&[&ds]);
            let want = run_entry(&yard, "argmin", &stdin).unwrap();
            let got = run_exe(&exe, &stdin).unwrap();
            let host = interp.eval(&tree, std::slice::from_ref(&ds)).unwrap();
            assert_eq!((got.elem, got.dims.clone()), (Elem::I32, vec![]));
            assert_eq!(got, want, "the argmin compiled, n = {n}, round {round}: {:?} vs the yardstick's {:?}", got.i32_values(), want.i32_values());
            assert_eq!(host, want, "the argmin interpreted, n = {n}, round {round}");
            checked += 1;
        }
    }
    eprintln!("pair constructs: {checked} runs of the Kahan mean and the argmin equal the yardstick and the interpreter bit for bit");
    assert_eq!(checked, 30);
    let _ = std::fs::remove_dir_all(&base);
}

/// The two conversions equal `futhark c` on the edges: an f32 to an i32 on a NaN, the infinities, values past i32's
/// range, 2147483520 and 2147483584 (the last f32 below 2^31 and the first at it), fractions of both signs; an i32 to
/// an f32 on i32's bounds and on 16777217 (the first integer an f32 cannot hold).
#[test]
fn the_conversions_equal_futhark_c_on_the_edges() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let base = scratch("cvt");
    let k = load(&ARGMIN).unwrap();
    let sig = signature(&ARGMIN).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let tree = build(&k, "(cvt.f32 (get.n ds zero))").unwrap();
    let exe = compile_c(&program(&k, &sig, &tree).unwrap(), &base.join("to_i32")).unwrap();
    let edges = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3e9, -3e9, 2.5, -2.5, 2_147_483_520.0, 2_147_483_584.0, -2_147_483_648.0, -2_147_483_904.0, 0.0, -0.0, 1e-45, 16_777_217.0];
    for v in edges {
        let ds = Array::f32s(vec![1], vec![v]);
        let got = run_exe(&exe, &write_binary(&[&ds])).unwrap();
        assert_eq!(interp.eval(&tree, std::slice::from_ref(&ds)).unwrap(), got, "i32.f32 of {v}");
    }
    let k = load(&KAHAN).unwrap();
    let sig = signature(&KAHAN).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let tree = build(&k, "(cvt.i32 (get.n ms zero))").unwrap();
    let exe = compile_c(&program(&k, &sig, &tree).unwrap(), &base.join("to_f32")).unwrap();
    for v in [0i32, 1, -1, i32::MAX, i32::MIN, 16_777_216, 16_777_217, -16_777_217, 123_456_789] {
        let xs = Array::f32s(vec![1], vec![0.0]);
        let ms = Array::i32s(vec![1], vec![v]);
        let got = run_exe(&exe, &write_binary(&[&xs, &ms])).unwrap();
        assert_eq!(interp.eval(&tree, &[xs, ms]).unwrap(), got, "f32.i32 of {v}");
    }
    let _ = std::fs::remove_dir_all(&base);
}

/// The triple state and the kernel boundary (master version 5), as genes, compute what `futhark c` computes of the
/// yardstick entries `cell` and `kernels` (Futhark's own triple loop and `let`), bit for bit, on inputs spanning
/// magnitudes, n = 1 included, every point a member, no member (the previous centre is kept); the host interpreter
/// agrees with both.
#[test]
fn the_cell_update_and_the_kernel_boundary_compute_what_futhark_c_computes() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let base = scratch("version5");
    let yard = compile_c(include_str!("../../kingdoms/futhark/sig/kahan/kahan.fut"), &base.join("yardstick")).unwrap();
    let mut rng = GeneRng::new(0x4A1A_0005);
    let mut checked = 0;
    let k = load(&CELL).unwrap();
    let sig = signature(&CELL).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let tree = build(&k, &cell_update()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(sexp(&k, &tree).split_whitespace().collect::<String>(), cell_update().split_whitespace().collect::<String>(), "the cell update reads back");
    let src = program(&k, &sig, &tree).unwrap();
    assert!(src.contains("loop ((s_f32, s2_f32, s_i32): (f32, f32, i32)) = ((0f32, 0f32, 0i32)) for i_n < n do"), "{src}");
    let exe = compile_c(&src, &base.join("cell")).unwrap_or_else(|e| panic!("{e}\n{src}"));
    for n in [1usize, 2, 7, 64, 1000] {
        for round in 0..3 {
            let (xs, mut ms) = pair_inputs(n, &mut rng);
            let (olds, _) = pair_inputs(n, &mut rng);
            if round == 1 {
                ms = Array::i32s(vec![n], vec![0; n]);
            } else if round == 2 {
                ms = Array::i32s(vec![n], vec![1; n]);
            }
            let stdin = write_binary(&[&xs, &ms, &olds]);
            let want = run_entry(&yard, "cell", &stdin).unwrap();
            let got = run_exe(&exe, &stdin).unwrap();
            let host = interp.eval(&tree, &[xs.clone(), ms.clone(), olds.clone()]).unwrap();
            if round == 2 {
                assert_eq!(want, Array::f32s(vec![], vec![olds.f32_values().unwrap()[0]]), "no member: the previous centre is kept");
            }
            assert_eq!(got, want, "the cell update compiled, n = {n}, round {round}: {:?} vs the yardstick's {:?}", got.f32_values(), want.f32_values());
            assert_eq!(host, want, "the cell update interpreted, n = {n}, round {round}");
            checked += 1;
        }
    }
    let k = load(&KERNELS).unwrap();
    let sig = signature(&KERNELS).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let tree = build(&k, KERNELS_PROGRAM).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(sexp(&k, &tree).split_whitespace().collect::<String>(), KERNELS_PROGRAM.split_whitespace().collect::<String>(), "the kernels program reads back");
    let src = program(&k, &sig, &tree).unwrap();
    assert!(src.contains("let s_a_i32_n = (tabulate n (\\i_n -> "), "{src}");
    let exe = compile_c(&src, &base.join("kernels")).unwrap_or_else(|e| panic!("{e}\n{src}"));
    for n in [1usize, 2, 7, 64, 1000] {
        for round in 0..3 {
            let (xs, _) = pair_inputs(n, &mut rng);
            let xs = if round == 1 { Array::f32s(vec![n], vec![0.75; n]) } else if round == 2 { Array::f32s(vec![n], vec![0.25; n]) } else { xs };
            let stdin = write_binary(&[&xs]);
            let want = run_entry(&yard, "kernels", &stdin).unwrap();
            let got = run_exe(&exe, &stdin).unwrap();
            let host = interp.eval(&tree, std::slice::from_ref(&xs)).unwrap();
            assert_eq!(got, want, "the kernels program compiled, n = {n}, round {round}: {:?} vs the yardstick's {:?}", got.f32_values(), want.f32_values());
            assert_eq!(host, want, "the kernels program interpreted, n = {n}, round {round}");
            checked += 1;
        }
    }
    eprintln!("version 5 constructs: {checked} runs of the cell update and the kernel boundary equal the yardstick and the interpreter bit for bit");
    assert_eq!(checked, 30);
    let _ = std::fs::remove_dir_all(&base);
}

/// Legality, measured on the four views: 50 random genes at the runner's default program size express to programs
/// `futhark check` accepts (the pair and triple rows, their loops, the kernel boundary, the conversions and both
/// elements' rows among them), and the first 8 of each compile with `futhark c` and equal the interpreter on three
/// small cases, n = 1 included.
/// One outcome is the COMPILER'S: Futhark 0.28.1's "Fuse SOACs" pass stops with an internal error ("Name ... bound
/// twice") on a legal program that reads a scan's result in two places (measured 2026-11-03 on
/// `f32.sum (map2 (+) (hist (+) 0f32 n (map i64.f32 s) xs) s)` with `s` the scan written twice; `opaque` around the
/// scan's result avoids it, which would change the scan rows' spelling in every pinned view, a decision for the lead);
/// such a gene is counted and printed, never taken as a fault of the kingdom, and the rest must compile and agree.
#[test]
fn random_pair_view_genes_are_accepted_by_futhark_check_and_equal_the_interpreter() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let base = scratch("random");
    let mut rng = GeneRng::new(0x51C_0004);
    let mut used = BTreeMap::<String, usize>::new();
    for inst in [&KAHAN, &ARGMIN, &CELL, &KERNELS] {
        let k = load(inst).unwrap();
        let sig = signature(inst).unwrap();
        let interp = Interp::new(&k, &sig).unwrap();
        let contract = k.relations().check().unwrap();
        let layout = GeneLayout::from_program_size(&contract, k.roots()[0], DEFAULT_PROGRAM_SIZE).unwrap();
        let const_len = k.const_len();
        let small: Vec<Vec<Array>> = [5usize, 1, 3]
            .iter()
            .map(|&n| {
                let (xs, ms) = pair_inputs(n, &mut rng);
                let (olds, _) = pair_inputs(n, &mut rng);
                match inst.name {
                    "kahan" => vec![xs, ms],
                    "cell" => vec![xs, ms, olds],
                    _ => vec![xs],
                }
            })
            .collect();
        let (mut checked, mut compiled, mut compiler_bug) = (0, 0, 0);
        for g in 0..50 {
            let gene = random_gene(&contract, &layout, &const_len, &mut rng).unwrap();
            let tree = express(&contract, &layout, &gene, &const_len).unwrap().tree();
            for (t, _) in tree.level_order() {
                let op = k.rows[t.symbol as usize].name.split('@').next().unwrap_or("").split('.').next().unwrap_or("").to_string();
                *used.entry(op).or_default() += 1;
            }
            let src = program(&k, &sig, &tree).unwrap();
            futhark_check(&src, &base.join(format!("{}_c{g}", inst.name))).unwrap_or_else(|e| panic!("{}: gene {g} is not accepted: {e}\n{src}", inst.name));
            checked += 1;
            if g < 8 {
                let batch = program_cases(&k, &sig, &tree, small.len()).unwrap();
                let bexe = match compile_c(&batch, &base.join(format!("{}_b{g}", inst.name))) {
                    Ok(exe) => exe,
                    Err(e) if e.contains("Internal compiler error") && e.contains("Fuse SOACs") => {
                        eprintln!("{}: gene {g} trips Futhark 0.28.1's fusion pass (the compiler's bug, not the kingdom's): {}", inst.name, sexp(&k, &tree));
                        compiler_bug += 1;
                        continue;
                    }
                    Err(e) => panic!("{}: gene {g}'s quality program does not compile: {e}\n{batch}", inst.name),
                };
                let refs: Vec<&Array> = small.iter().flatten().collect();
                let out = crate::futhark_exec::output_with_timeout_bytes(std::process::Command::new(&bexe).arg("-b"), std::time::Duration::from_secs(120), Some(&write_binary(&refs))).unwrap();
                assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
                let wants: Vec<Array> = small.iter().map(|w| interp.eval(&tree, w).unwrap()).collect();
                assert_eq!(read_binary(&out.stdout).unwrap(), wants, "{}: gene {g}: compiled and interpreted differ\n{src}", inst.name);
                compiled += 1;
            }
        }
        eprintln!("{}: {checked} random genes accepted by futhark check, {compiled} compiled and equal to the interpreter, {compiler_bug} stopped by the compiler's fusion bug", inst.name);
        assert_eq!((checked, compiled + compiler_bug), (50, 8));
        assert!(compiled >= 6, "{}: only {compiled} of 8 genes compiled", inst.name);
    }
    eprintln!("rows used by the random genes: {used:?}");
    for op in ["pair", "fst", "snd", "psel", "loop", "cvt", "zero", "triple", "thd", "let"] {
        assert!(used.contains_key(op), "no random gene used {op}: {used:?}");
    }
    let _ = std::fs::remove_dir_all(&base);
}

/// The pair rules of the e-graph: the first of a built pair is its first component (`fst (pair a b) = a`), so the
/// e-graph's variants of a program that builds a pair and takes it apart hold the component alone; and every variant
/// of the Kahan mean computes the same bits (the rules are only commutativity of `+`/`*`, `-(-a) = a` and the two
/// projections: no real-number identity touches `(t - s) - y`).
#[test]
fn the_pair_rules_are_exact_and_the_kahan_form_survives_saturation() {
    let k = load(&KAHAN).unwrap();
    let sig = signature(&KAHAN).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let bridge = Bridge::load(&KAHAN).unwrap();
    let wrapped = build(&k, "(fst.f32.f32 (pair (sum.n xs) (cvt.i32 (sum.n ms))))").unwrap();
    let plain = build(&k, "(sum.n xs)").unwrap();
    let vs = variants(&k, &bridge, &wrapped, 2, 16, 16).unwrap();
    assert!(vs.contains(&plain), "the variants of fst (pair a b) do not hold a: {:?}", vs.iter().map(|v| sexp(&k, v)).collect::<Vec<_>>());
    let kahan = build(&k, KAHAN_MEAN).unwrap();
    let vs = variants(&k, &bridge, &kahan, 3, 64, 64).unwrap();
    assert!(!vs.is_empty(), "no variants of the Kahan mean");
    let mut rng = GeneRng::new(0x4A1A_0002);
    let cases: Vec<(Array, Array)> = [1usize, 9, 300].into_iter().map(|n| pair_inputs(n, &mut rng)).collect();
    for v in &vs {
        for (xs, ms) in &cases {
            assert_eq!(interp.eval(v, &[xs.clone(), ms.clone()]).unwrap(), interp.eval(&kahan, &[xs.clone(), ms.clone()]).unwrap(), "the variant {} differs from the Kahan mean", sexp(&k, v));
        }
        // Every variant keeps the compensation: three subtractions remain.
        assert!(sexp(&k, v).matches("(sub ").count() >= 3, "the variant {} lost the compensation", sexp(&k, v));
    }
    eprintln!("kahan: {} variants, every one the same bits on {} cases", vs.len(), cases.len());
}

/// THE KAHAN FORM SURVIVES SATURATION IN THE TRIPLE (step 4c (c)): the e-graph's rules for f32 are the bit-exact
/// ones only (commutativity of `+` and `*`, `-(-a) = a`, the tuple projections), so no variant of the cell update
/// simplifies `(t - s) - y` (0 in real arithmetic, not in f32): every variant computes the same bits on inputs
/// spanning magnitudes and keeps its three subtractions; and the triple's projections are exact (`thd (triple a b c)
/// = c`, so the variants of a projected triple hold the component alone).
#[test]
fn the_triple_rules_are_exact_and_the_kahan_form_survives_saturation_in_the_triple() {
    let k = load(&CELL).unwrap();
    let sig = signature(&CELL).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let bridge = Bridge::load(&CELL).unwrap();
    let wrapped = build(&k, "(cvt.i32 (thd.f32.f32.i32 (triple (sum.n xs) (sum.n olds) (sum.n ms))))").unwrap();
    let plain = build(&k, "(cvt.i32 (sum.n ms))").unwrap();
    let vs = variants(&k, &bridge, &wrapped, 2, 16, 16).unwrap();
    assert!(vs.contains(&plain), "the variants of thd (triple a b c) do not hold c: {:?}", vs.iter().map(|v| sexp(&k, v)).collect::<Vec<_>>());
    let cell = build(&k, &cell_update()).unwrap();
    let vs = variants(&k, &bridge, &cell, 3, 64, 64).unwrap();
    assert!(!vs.is_empty(), "no variants of the cell update");
    let mut rng = GeneRng::new(0x4A1A_0006);
    let cases: Vec<Vec<Array>> = [1usize, 9, 300].into_iter().map(|n| {
        let (xs, ms) = pair_inputs(n, &mut rng);
        let (olds, _) = pair_inputs(n, &mut rng);
        vec![xs, ms, olds]
    }).collect();
    for v in &vs {
        for c in &cases {
            assert_eq!(interp.eval(v, c).unwrap(), interp.eval(&cell, c).unwrap(), "the variant {} differs from the cell update", sexp(&k, v));
        }
        assert!(sexp(&k, v).matches("(sub ").count() >= 3, "the variant {} lost the compensation", sexp(&k, v));
    }
    eprintln!("cell: {} variants, every one the same bits on {} cases", vs.len(), cases.len());
}
