//! Master 6's reader (plan section 7, task F4): every reference tree reads back from its rendering to itself; random
//! genes of every view read back to their canonical trees and render the same text again, adapters included; the
//! pool's 842 programs read from their master 5 renderings, render the same text and compute the recorded values on
//! every stored case; a text with an adapter the gene does not render is not read; the k-means pool's programs read.

use super::*;
use crate::futhark_sig::{build6, build_stack_named, const_len6, kmeans_reference6, load, load_cases, problem, program, random_gene, signature, values_fingerprint, Interp, REFERENCES6};
use crate::futhark_table::{GeneRng, Header6, Kingdom6};
use crate::typed_gep::{express, GeneLayout};

/// A view's kingdom and header.
fn view(name: &str) -> (Kingdom6, Header6) {
    let k = crate::futhark_sig::kingdom6(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let spec = crate::futhark_view::master6::committed(name).unwrap_or_else(|e| panic!("{name}: {e}")).spec;
    (k, Header6::of_spec(&spec))
}

/// The first node where two trees differ: its path and both nodes' symbol and word.
fn tree_diff(a: &Tree, b: &Tree) -> String {
    fn go(a: &Tree, b: &Tree, path: &mut Vec<usize>) -> Option<String> {
        if a.symbol != b.symbol || a.const_index != b.const_index || a.kids.len() != b.kids.len() {
            return Some(format!("at {path:?}: {} {:?} against {} {:?}", a.symbol, a.const_index, b.symbol, b.const_index));
        }
        for (i, (x, y)) in a.kids.iter().zip(&b.kids).enumerate() {
            path.push(i);
            let r = go(x, y, path);
            path.pop();
            if r.is_some() {
                return r;
            }
        }
        None
    }
    go(a, b, &mut Vec::new()).unwrap_or_else(|| "equal".into())
}

/// The first byte where two texts differ, with some context.
fn diff(got: &str, want: &str) -> String {
    let at = got.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
    format!("differ at byte {at}:\n got: {}\nwant: {}", &got[at.saturating_sub(100)..(at + 100).min(got.len())], &want[at.saturating_sub(100)..(at + 100).min(want.len())])
}

/// `read(render(t)) = t` FOR EVERY F2a TREE: each of the 15 reference programs, rendered by pass 5, reads back to its
/// own tree, which is already canonical (no substitution, every index word the least that names its variable).
#[test]
fn every_reference_tree_reads_back_from_its_rendering_to_itself() {
    for r in &REFERENCES6 {
        let key = r.key();
        let (k, header) = view(r.problem);
        let renderer = k.renderer6(header.clone()).unwrap();
        let reader = Reader6::new(&k, header).unwrap_or_else(|e| panic!("{key}: {e}"));
        let tree = build6(&k, &(r.master6)()).unwrap_or_else(|e| panic!("{key}: {e}"));
        assert_eq!(reader.canonical(&tree).unwrap(), tree, "{key}: the F2a tree is not canonical");
        let text = renderer.program(&tree).unwrap().text;
        let read = reader.read_program(&text).unwrap_or_else(|g| panic!("{key}: does not read: {g}"));
        assert!(read == tree, "{key}: reads back to another tree: {}", tree_diff(&read, &tree));
    }
}

/// Every master 6 view.
const VIEWS6: [&str; 7] = ["reduce", "scan", "histogram", "pathfinder", "a7", "kmeans", "kmeans_i32"];

/// RANDOM GENES, ADAPTERS INCLUDED: 200 random genes a view, expressed and rendered, read back; the tree read renders
/// the same text again (`render(read(x)) = x`: an adapter is read as its plain row and the requirement that rendered it
/// fails again), and it is the gene's canonical tree wherever one is defined (no constant substitution, no variable
/// read as its data type's constant: a terminal of the same spelling reads first there).
#[test]
fn random_genes_read_back_to_their_canonical_trees_and_render_the_same_text() {
    let mut report = Vec::new();
    for (vi, name) in VIEWS6.iter().enumerate() {
        let (k, header) = view(name);
        let renderer = k.renderer6(header.clone()).unwrap();
        let reader = Reader6::new(&k, header).unwrap();
        let size = problem(if *name == "kmeans_i32" { "kmeans" } else { name }).unwrap().program_size;
        let layout = GeneLayout::from_program_size(&renderer.contract, k.root.data_type, size).unwrap();
        let consts = const_len6(&k);
        let mut rng = GeneRng::new(0xF4_0000 + vi as u64);
        let (mut with_adapters, mut compared) = (0, 0);
        for i in 0..200 {
            let gene = random_gene(&renderer.contract, &layout, &consts, &mut rng).unwrap();
            let tree = express(&renderer.contract, &layout, &gene, &consts).unwrap().tree();
            let p = renderer.program(&tree).unwrap();
            let read = reader.read_program(&p.text).unwrap_or_else(|g| panic!("{name} gene {i}: does not read: {g}\n{}", p.text));
            let again = renderer.program(&read).unwrap();
            assert!(again.text == p.text, "{name} gene {i}: renders another text: {}", diff(&again.text, &p.text));
            // The same adapters by name: a substitution by an input renders that input, which the reader reads in the
            // substituted node's place, so an adapter below it has a shorter path.
            let names = |q: &crate::futhark_table::Program6| q.adapters.iter().map(|a| a.1.clone()).filter(|a| !a.starts_with("subst")).collect::<Vec<_>>();
            assert_eq!(names(&again), names(&p), "{name} gene {i}: renders other adapters");
            if !p.adapter_free() {
                with_adapters += 1;
            }
            // Only the two constant cases are skipped (a terminal of the same spelling reads first there); any other
            // reason is a bug.
            match reader.canonical(&tree) {
                Ok(c) => {
                    assert!(read == c, "{name} gene {i}: reads back to another tree than its canonical one: {}\n{}", tree_diff(&read, &c), p.text);
                    compared += 1;
                }
                Err(NotCanonical::SubstitutedByConstant | NotCanonical::VariableAsConstant) => {}
                Err(NotCanonical::Bug(e)) => panic!("{name} gene {i}: no canonical tree: {e}"),
            }
        }
        eprintln!("{name}: 200 random genes read back and render the same text; {with_adapters} with an adapter; {compared} equal to their canonical tree, the rest render a constant");
        report.push((*name, with_adapters, compared));
    }
    // The genes are drawn from fixed seeds: the counts of this run, as floors.
    let floors = [("reduce", 0, 104), ("scan", 2, 165), ("histogram", 166, 143), ("pathfinder", 143, 147), ("a7", 189, 200), ("kmeans", 180, 150), ("kmeans_i32", 183, 156)];
    for ((name, adapters, compared), (fname, fa, fc)) in report.iter().zip(floors) {
        assert!(*name == fname && *adapters >= fa && *compared >= fc, "{name}: {adapters} with an adapter (floor {fa}), {compared} compared (floor {fc}): {report:?}");
    }
}

/// The tree read by [`Reader6::read_expression`] rendered as pass 5 renders it: a whole program whose entry returns
/// its data type.
fn render_expression(k: &Kingdom6, renderer: &crate::futhark_table::Renderer6, tree: &Tree, ty: Ty) -> String {
    let entry = crate::typed_environment::Entry { output: ty, variables: renderer.entry.variables.clone(), requirements: Vec::new() };
    let decoding = crate::typed_environment::Decoder::new(&renderer.tables, &renderer.contract).unwrap().decode(&entry, tree).unwrap();
    k.program6(&renderer.header, &decoding).unwrap().text
}

/// THE POOL'S 842 PROGRAMS (`tests/fixtures/futhark_pool_values/`: histogram's and pathfinder's bench, llm and
/// alternatives, recorded with their values on the stored cases): each, built in its master 5 view from its recorded
/// text and checked against its recorded values (host interpreter), is rendered by master 5's renderer; master 6's
/// reader reads that text, and master 6's renderer renders the tree read as the same text, byte for byte (an `i32`
/// expression: its whole body). Master 6 has no interpreter yet, so the VALUES are checked through master 5's: the text
/// master 6 renders is read by master 5's reader and evaluated on every stored case, against the recorded values.
#[test]
fn the_pools_842_programs_read_back_with_the_same_values() {
    let (mut report, mut all_gaps) = (Vec::new(), Vec::new());
    for pname in ["histogram", "pathfinder"] {
        let p = problem(pname).unwrap();
        let k5 = load(&p.instance).unwrap();
        let sig = signature(&p.instance).unwrap();
        let interp = Interp::new(&k5, &sig).unwrap();
        let cases = load_cases(&p).unwrap();
        let root5 = k5.roots()[0];
        let (k6, header) = view(pname);
        let renderer = k6.renderer6(header.clone()).unwrap();
        let reader = Reader6::new(&k6, header).unwrap();
        let reader5 = Reader::new(&k5, &sig).unwrap();
        let values_of = |text6: &str, what: &str| {
            let back = reader5.read_program(text6).unwrap_or_else(|g| panic!("{what}: master 5 does not read master 6's text: {g}\n{text6}"));
            format!("{:016x}", values_fingerprint(&interp, &back, &cases))
        };
        for file in ["bench", "llm", "alternatives"] {
            let path = std::path::Path::new(crate::CHECKOUT).join("tests/fixtures/futhark_pool_values").join(format!("{pname}.{file}.tsv"));
            let fixture = std::fs::read_to_string(&path).unwrap();
            let (mut n, mut same, mut gaps) = (0, 0, Vec::new());
            for line in fixture.lines().skip(1) {
                let f: Vec<&str> = line.split('\t').collect();
                let (seq, dual, text, values) = (f[0], f[1], f[2], f[3]);
                n += 1;
                let t5 = build_stack_named(&k5, text, dual).unwrap_or_else(|e| panic!("{pname} {file} {seq}: {e}"));
                assert_eq!(format!("{:016x}", values_fingerprint(&interp, &t5, &cases)), values, "{pname} {file} {seq}: the master 5 program's values are not the recorded ones");
                let ty5 = k5.ty(dual).unwrap();
                if ty5 == root5 {
                    let text5 = program(&k5, &sig, &t5).unwrap();
                    match reader.read_program(&text5) {
                        Ok(t6) => {
                            let text6 = renderer.program(&t6).unwrap().text;
                            assert!(text6 == text5, "{pname} {file} {seq}: {}", diff(&text6, &text5));
                            assert_eq!(values_of(&text6, &format!("{pname} {file} {seq}")), values, "{pname} {file} {seq}: master 6's text computes other values");
                        }
                        Err(g) => {
                            gaps.push(format!("{pname} {file} {seq}: {}", g.construct));
                            continue;
                        }
                    }
                } else {
                    let text5 = k5.render(&t5).unwrap();
                    let ty6 = k6.ty(dual.split(':').next().unwrap()).unwrap_or_else(|| panic!("{pname} {file} {seq}: no master 6 data type of {dual}"));
                    let t6 = reader.read_expression(&text5, ty6).unwrap_or_else(|g| panic!("{pname} {file} {seq}: does not read: {g}\n{text5}"));
                    let text6 = render_expression(&k6, &renderer, &t6, ty6);
                    let body6 = reader.program_body(&text6).unwrap_or_else(|g| panic!("{pname} {file} {seq}: {g}\n{text6}")).1;
                    assert_eq!(unparen(&body6), unparen(&parse_expression(&text5, false, true).unwrap()), "{pname} {file} {seq}: renders another expression\n{text6}");
                    // Master 5's reader reads a whole program at its root only: the body of master 6's text, parsed in
                    // master 5's syntax without the header's bindings, is read at the dual's type.
                    let defs5 = parse_file(&text6).unwrap_or_else(|g| panic!("{pname} {file} {seq}: {g}\n{text6}"));
                    let mut body5 = defs5.iter().find(|d| d.entry).unwrap().body.clone().unwrap();
                    while let E::Let(x, _, rest) = &body5 {
                        assert!(x == "k" || x.starts_with("zeros_"), "{pname} {file} {seq}: a binding {x} past the header\n{text6}");
                        body5 = *rest.clone();
                    }
                    reader5.steps.set(0);
                    let back = reader5.read(&body5, ty5).unwrap_or_else(|g| panic!("{pname} {file} {seq}: master 5 does not read master 6's body: {g}\n{text6}"));
                    assert_eq!(format!("{:016x}", values_fingerprint(&interp, &back, &cases)), values, "{pname} {file} {seq}: master 6's text computes other values");
                }
                same += 1;
            }
            eprintln!("{pname} {file}: {n} programs, {same} read back and render the same text; not read {}: {gaps:#?}", gaps.len());
            report.push((pname, file, n, same));
            all_gaps.extend(gaps);
        }
    }
    assert_eq!(report, [("histogram", "bench", 10, 10), ("histogram", "llm", 48, 48), ("histogram", "alternatives", 577, 577), ("pathfinder", "bench", 9, 9), ("pathfinder", "llm", 45, 43), ("pathfinder", "alternatives", 153, 153)]);
    // THE TWO NOT READ ARE NO MASTER 6 PROGRAM: each indexes `walls[i64.min (rows - 1) ((0i64) + 1)]`, master 5's `next`
    // of the literal index 0, whose class gave it the bound `rows`. Master 6 gives the literal `0i64` the bound 1 (plan
    // 7.1 "Literal indices"), so its `next` renders `i64.min (1 - 1) ((0i64) + 1)` (row 0, not row 1), and no tree of the
    // view renders this text: the property placeholder `{0.bound}` is checked and fails.
    assert_eq!(all_gaps, ["pathfinder llm 12: {0.bound}", "pathfinder llm 26: {0.bound}"]);
}

/// THE K-MEANS POOL'S PROGRAMS (`kingdoms/futhark/sig/kmeans/pool/*.fut`, what `pool.fut_read_by_reader` counts):
/// `seed.fut`, the seed's rendering, reads to the k-means reference tree.
#[test]
fn the_kmeans_pool_programs_read() {
    let (k, header) = view("kmeans");
    let reader = Reader6::new(&k, header).unwrap();
    let dir = std::path::Path::new(crate::CHECKOUT).join("kingdoms/futhark/sig/kmeans/pool");
    let mut names: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "fut")).collect();
    names.sort();
    let mut report = Vec::new();
    for path in &names {
        let src = std::fs::read_to_string(path).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let r = reader.read_program(&src);
        eprintln!("{name}: {:?}", r.as_ref().map(|t| t.level_order().len()));
        if name == "seed.fut" {
            let read = r.as_ref().unwrap_or_else(|g| panic!("seed.fut does not read: {g}"));
            let want = build6(&k, &kmeans_reference6()).unwrap();
            assert!(*read == want, "{}", tree_diff(read, &want));
        }
        report.push((name, r.map(|_| ()).map_err(|g| g.construct)));
    }
    eprintln!("{report:?}");
    // textbook.fut is the published rodinia step (its gene is textbook.tsv): its helpers fold over `zip` with
    // tuple-patterned lambdas into an `i32` membership, which no row of the `f32` view is.
    assert_eq!(report, [("seed.fut".to_string(), Ok(())), ("textbook.fut".to_string(), Err("call of the program's function centroids_of".to_string()))]);
}

/// Float literals are read by their value, in one spelling.
#[test]
fn float_literals_have_one_spelling() {
    assert_eq!(float_literal("0f32"), Some("0.0f32".into()));
    assert_eq!(float_literal("0.0f32"), Some("0.0f32".into()));
    assert_eq!(float_literal("-1f32"), Some("-1.0f32".into()));
    assert_eq!(float_literal("1.0"), Some("1.0".into()));
    assert_eq!(float_literal("1_000.5f64"), Some("1000.5f64".into()));
    assert_eq!(float_literal("integer of type u8"), None);
}

/// An expression with every fit's mark taken away.
fn unmarked(e: &E) -> E {
    match unmark(e) {
        (inner, _) if !std::ptr::eq(inner, e) => unmarked(inner),
        _ => e.map_kids(&mut unmarked),
    }
}

/// An adapter is read as its plain row: `fit`, `map (fit ..)`, `clampix` and the adapters' `let ad_*` are gone, a
/// fit leaving the length it fitted to as its mark.
#[test]
fn adapters_are_read_as_the_plain_row() {
    let e = parse_expression("let ad_a = (xs) let ad_b = (ys) let ad_l = i64.min (length ad_a) (length ad_b) in map2 (+) (fit ad_l 0f32 ad_a) (map (fit d 0f32) (fit ad_l 0f32 ad_b))", false, true).unwrap();
    let stripped = strip_adapters(&e);
    assert_eq!(unparen(&unmarked(&stripped)), parse_expression("map2 (+) xs ys", false, true).unwrap());
    let E::App(_, args) = &stripped else { panic!("{stripped:?}") };
    assert_eq!(unmark(&args[1]).1.map(unparen), Some(unparen(&parse_expression("i64.min (length xs) (length ys)", false, true).unwrap())));
    let e = parse_expression("let [ad_m] (ad_s: [ad_m]f32) = (xs) in (ad_s)[(clampix ad_m (i))]", false, true).unwrap();
    assert_eq!(unparen(&unmarked(&strip_adapters(&e))), parse_expression("xs[i]", false, true).unwrap());
}

/// AN ADAPTER THE GENE DOES NOT RENDER IS NOT READ (F4 review finding 1): a text with a `fit` or a `clampix` where the
/// tree read renders none (its requirement holds) is a program no gene expresses, with other values (the `fit`
/// truncates to 5 elements), so it is a gap naming the adapter, never read as the plain row. The same program as pass 5
/// renders it reads, and so does a random gene's text whose adapters pass 5 rendered.
#[test]
fn an_adapter_the_gene_does_not_render_is_not_read() {
    let (k, header) = view("kmeans");
    let reader = Reader6::new(&k, header).unwrap();
    let seed = std::fs::read_to_string(std::path::Path::new(crate::CHECKOUT).join("kingdoms/futhark/sig/kmeans/pool/seed.fut")).unwrap();
    reader.read_program(&seed).unwrap_or_else(|g| panic!("seed.fut does not read: {g}"));
    for (from, to) in [("(points)[(i_n)]", "(fit 5 0f32 ((points)[(i_n)]))"), ("((centres)[(0i64)])", "((centres)[(clampix k (0i64))])")] {
        assert!(seed.contains(from), "{from}");
        let text = seed.replacen(from, to, 1);
        let g = reader.read_program(&text).expect_err(to);
        assert_eq!(g.construct, "adapter the gene does not render", "{to}: {g}");
    }
    let (k, header) = view("histogram");
    let renderer = k.renderer6(header.clone()).unwrap();
    let reader = Reader6::new(&k, header).unwrap();
    let layout = GeneLayout::from_program_size(&renderer.contract, k.root.data_type, problem("histogram").unwrap().program_size).unwrap();
    let consts = const_len6(&k);
    let mut rng = GeneRng::new(0xF4_0001);
    let fitted = (0..500)
        .map(|_| renderer.program(&express(&renderer.contract, &layout, &random_gene(&renderer.contract, &layout, &consts, &mut rng).unwrap(), &consts).unwrap().tree()).unwrap())
        .find(|p| p.adapters.iter().any(|a| a.1.starts_with("fit")))
        .expect("no random histogram gene of 500 renders a fit");
    reader.read_program(&fitted.text).unwrap_or_else(|g| panic!("a fit pass 5 renders does not read: {g}\n{}", fitted.text));
}
