//! Master 6's reader (plan section 7, task F4): every reference tree reads back from its rendering to itself; random
//! genes of every view read back to their canonical trees and render the same text again, adapters included; the
//! pool's 842 programs, kept as program text (`tests/fixtures/futhark_pool_texts/`), read, render the same text and
//! compute the recorded values on every stored case; a text with an adapter the gene does not render is not read; the
//! k-means pool's programs read.

use super::*;
use crate::futhark_sig::{build6, const_len6, kmeans_reference6, load_cases, problem, random_gene, signature, values_fingerprint, Interp, REFERENCES6};
use crate::futhark_table::{GeneRng, Kingdom6};
use crate::typed_gep::{express, GeneLayout};

/// A view's kingdom.
fn view(name: &str) -> Kingdom6 {
    crate::futhark_sig::kingdom6(name).unwrap_or_else(|e| panic!("{name}: {e}"))
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
        let k = view(r.problem);
        let reader = Reader6::new(&k).unwrap_or_else(|e| panic!("{key}: {e}"));
        let tree = build6(&k, &(r.master6)()).unwrap_or_else(|e| panic!("{key}: {e}"));
        assert_eq!(reader.canonical(&tree).unwrap(), tree, "{key}: the F2a tree is not canonical");
        let text = k.program(&tree).unwrap().text;
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
        let k = view(name);
        let reader = Reader6::new(&k).unwrap();
        let size = problem(if *name == "kmeans_i32" { "kmeans" } else { name }).unwrap().program_size;
        let layout = GeneLayout::from_program_size(&k.contract, k.entry.output, size).unwrap();
        let consts = const_len6(&k);
        let mut rng = GeneRng::new(0xF4_0000 + vi as u64);
        let (mut with_adapters, mut compared, mut ambiguous, mut unread) = (0, 0, 0, 0);
        for i in 0..200 {
            let gene = random_gene(&k.contract, &layout, &consts, &mut rng).unwrap();
            let tree = express(&k.contract, &layout, &gene, &consts).unwrap().tree();
            let p = k.program(&tree).unwrap();
            // A text mixing placed adapters and repairs the reader may not untangle: counted apart (a gap, never a wrong
            // read); a gene with no placed adapter always reads.
            let read = match reader.read_program(&p.text) {
                Ok(t) => t,
                Err(g) if p.adapters.iter().any(|a| a.2) => {
                    unread += 1;
                    eprintln!("{name} gene {i}: a placed adapter not read: {}", g.construct);
                    continue;
                }
                Err(g) => panic!("{name} gene {i}: does not read: {g}\n{}", p.text),
            };
            let again = k.program(&read).unwrap();
            assert!(again.text == p.text, "{name} gene {i}: renders another text: {}", diff(&again.text, &p.text));
            // The same adapters by name: a substitution by an input renders that input, which the reader reads in the
            // substituted node's place, so an adapter below it has a shorter path.
            let names = |q: &crate::futhark_table::Program6| {
                let mut v = q.adapters.iter().map(|a| a.1.clone()).collect::<Vec<_>>();
                v.sort();
                v
            };
            assert_eq!(names(&again), names(&p), "{name} gene {i}: renders other adapters");
            if !p.adapter_free() {
                with_adapters += 1;
            }
            // Only the two constant cases are skipped (a terminal of the same spelling reads first there); any other
            // reason is a bug.
            match reader.canonical(&tree) {
                Ok(c) if read == c => compared += 1,
                // An adapter evolution placed and a repair's call spell one text: the reader reads the repair (or the
                // innermost placed adapter that makes the text), a tree of the same program.
                Ok(c) => {
                    assert_eq!(k.program(&c).unwrap().text, again.text, "{name} gene {i}: reads back to another program than its canonical tree's: {}", tree_diff(&read, &c));
                    ambiguous += 1;
                }
                Err(NotCanonical::SubstitutedByConstant | NotCanonical::VariableAsConstant) => {}
                Err(NotCanonical::Bug(e)) => panic!("{name} gene {i}: no canonical tree: {e}"),
            }
        }
        eprintln!("{name}: 200 random genes read back and render the same text; {with_adapters} with an adapter; {compared} equal to their canonical tree, {ambiguous} another tree of the same text (a placed adapter and a repair), {unread} with a placed adapter not read, the rest render a constant");
        report.push((*name, with_adapters, compared));
    }
    // The genes are drawn from fixed seeds: the counts of this run, as floors (2026-11-05, the adapters in the alphabet:
    // the same seeds draw other genes than before).
    let floors = FLOORS;
    for ((name, adapters, compared), (fname, fa, fc)) in report.iter().zip(floors) {
        assert!(*name == fname && *adapters >= fa && *compared >= fc, "{name}: {adapters} with an adapter (floor {fa}), {compared} compared (floor {fc}): {report:?}");
    }
}

/// The floors of [`random_genes_read_back_to_their_canonical_trees_and_render_the_same_text`]: (view, genes with an
/// adapter, genes equal to their canonical tree), measured on its fixed seeds.
const FLOORS: [(&str, usize, usize); 7] = [("reduce", 13, 107), ("scan", 42, 158), ("histogram", 173, 151), ("pathfinder", 148, 141), ("a7", 190, 181), ("kmeans", 183, 164), ("kmeans_i32", 180, 164)];

/// THE POOL'S 842 PROGRAMS, kept as PROGRAM TEXT (`tests/fixtures/futhark_pool_texts/`: histogram's and pathfinder's
/// bench, llm and alternatives, each the Futhark text master 5 rendered, with the values recorded on the stored cases):
/// master 6's reader reads each text (a whole program, or an `i32` expression at its data type), master 6 renders the
/// tree read as the same text byte for byte (an expression: its whole body), and a whole program's tree computes the
/// recorded values on every stored case (master 6's interpreter).
#[test]
fn the_pools_842_programs_read_back_with_the_same_values() {
    let (mut report, mut all_gaps) = (Vec::new(), Vec::new());
    for pname in ["histogram", "pathfinder"] {
        let p = problem(pname).unwrap();
        let k = std::sync::Arc::new(view(pname));
        let interp = Interp::new6(&k, &signature(&p.instance).unwrap()).unwrap();
        let cases = load_cases(&p).unwrap();
        let reader = Reader6::new(&k).unwrap();
        for file in ["bench", "llm", "alternatives"] {
            let path = std::path::Path::new(crate::CHECKOUT).join("tests/fixtures/futhark_pool_texts").join(format!("{pname}.{file}.tsv"));
            let fixture = std::fs::read_to_string(&path).unwrap();
            let (mut n, mut same, mut gaps) = (0, 0, Vec::new());
            for line in fixture.lines().skip(1) {
                let f: Vec<&str> = line.split('\t').collect();
                let (seq, dt, whole, text, values) = (f[0], f[1], f[2], crate::geneframe::unescape(f[3]), f[4]);
                n += 1;
                if whole == "program" {
                    match reader.read_program(&text) {
                        Ok(t6) => {
                            let text6 = k.program(&t6).unwrap().text;
                            assert!(text6 == text, "{pname} {file} {seq}: {}", diff(&text6, &text));
                            assert_eq!(format!("{:016x}", values_fingerprint(&interp, &t6, &cases)), values, "{pname} {file} {seq}: the program read computes other values");
                        }
                        Err(g) => {
                            gaps.push(format!("{pname} {file} {seq}: {}", g.construct));
                            continue;
                        }
                    }
                } else {
                    let ty = k.ty(dt).unwrap_or_else(|| panic!("{pname} {file} {seq}: no data type {dt}"));
                    let t6 = reader.read_expression(&text, ty).unwrap_or_else(|g| panic!("{pname} {file} {seq}: does not read: {g}\n{text}"));
                    let entry = crate::typed_environment::Entry { output: ty, guards: Vec::new(), ..k.entry.clone() };
                    let d = k.decoder.decode(&entry, &t6).unwrap();
                    let body = k.decoder.render(&entry, &d, &|s, w| k.literals(s).get(w as usize).cloned().ok_or_else(|| format!("no literal {w}"))).unwrap().body;
                    assert_eq!(unparen(&parse_expression(&body, false, true).unwrap()), unparen(&parse_expression(&text, false, true).unwrap()), "{pname} {file} {seq}: renders another expression\n{body}");
                }
                same += 1;
            }
            eprintln!("{pname} {file}: {n} programs, {same} read back and render the same text; not read {}: {gaps:#?}", gaps.len());
            report.push((pname, file, n, same));
            all_gaps.extend(gaps);
        }
    }
    assert_eq!(report, [("histogram", "bench", 10, 10), ("histogram", "llm", 48, 48), ("histogram", "alternatives", 577, 577), ("pathfinder", "bench", 9, 9), ("pathfinder", "llm", 45, 45), ("pathfinder", "alternatives", 153, 153)]);
    assert!(all_gaps.is_empty(), "{all_gaps:#?}");
}

/// THE K-MEANS POOL'S PROGRAMS (`kingdoms/futhark/sig/kmeans/pool/*.fut`, what `pool.fut_read_by_reader` counts):
/// `seed.fut`, the seed's rendering, reads to the k-means reference tree.
#[test]
fn the_kmeans_pool_programs_read() {
    let k = view("kmeans");
    let reader = Reader6::new(&k).unwrap();
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
    let stripped = strip_adapters(&e, &BTreeSet::new());
    assert_eq!(unparen(&unmarked(&stripped)), parse_expression("map2 (+) xs ys", false, true).unwrap());
    let E::App(_, args) = &stripped else { panic!("{stripped:?}") };
    assert_eq!(unmark(&args[1]).1.map(unparen), Some(unparen(&parse_expression("i64.min (length xs) (length ys)", false, true).unwrap())));
    let e = parse_expression("let [ad_m] (ad_s: [ad_m]f32) = (xs) in (ad_s)[(clampix ad_m (i))]", false, true).unwrap();
    assert_eq!(unparen(&unmarked(&strip_adapters(&e, &BTreeSet::new()))), parse_expression("xs[i]", false, true).unwrap());
}

/// AN ADAPTER THE GENE DOES NOT RENDER IS NOT READ (F4 review finding 1): a text with a `fit` or a `clampix` where the
/// tree read renders none (its requirement holds) is a program no gene expresses, with other values (the `fit`
/// truncates to 5 elements), so it is a gap naming the adapter, never read as the plain row. The same program as pass 5
/// renders it reads, and so does a random gene's text whose adapters pass 5 rendered.
#[test]
fn an_adapter_the_gene_does_not_render_is_not_read() {
    let k = view("kmeans");
    let reader = Reader6::new(&k).unwrap();
    let seed = std::fs::read_to_string(std::path::Path::new(crate::CHECKOUT).join("kingdoms/futhark/sig/kmeans/pool/seed.fut")).unwrap();
    reader.read_program(&seed).unwrap_or_else(|g| panic!("seed.fut does not read: {g}"));
    // A fit to a length no size spells is no program a gene expresses: a gap naming the adapter.
    let (from, to) = ("(points)[(i_n)]", "(fit 5 0f32 ((points)[(i_n)]))");
    assert!(seed.contains(from), "{from}");
    let g = reader.read_program(&seed.replacen(from, to, 1)).expect_err(to);
    assert_eq!(g.construct, "adapter the gene does not render", "{to}: {g}");
    // A clamp the gene's guard does not need is the adapter SYMBOL placed there (design 3.3.1): read as it, and it
    // renders the same text.
    let (from, to) = ("((centres)[(0i64)])", "((centres)[(clampix k (0i64))])");
    let text = seed.replacen(from, to, 1).replacen("entry main", "def clampix (m: i64) (i: i64) : i64 = i64.max 0 (i64.min (m - 1) i)\n\nentry main", 1);
    let read = reader.read_program(&text).unwrap_or_else(|g| panic!("{to}: {g}"));
    assert!(k.program(&read).unwrap().adapters.iter().any(|a| a.1 == "clampix@i64" && a.2), "the clamp is read as placed");
    let k = view("histogram");
    let reader = Reader6::new(&k).unwrap();
    let layout = GeneLayout::from_program_size(&k.contract, k.entry.output, problem("histogram").unwrap().program_size).unwrap();
    let consts = const_len6(&k);
    let mut rng = GeneRng::new(0xF4_0001);
    let fitted = (0..500)
        .map(|_| k.program(&express(&k.contract, &layout, &random_gene(&k.contract, &layout, &consts, &mut rng).unwrap(), &consts).unwrap().tree()).unwrap())
        .find(|p| p.adapters.iter().any(|a| a.1.starts_with("fit") && !a.2))
        .expect("no random histogram gene of 500 renders a fit");
    reader.read_program(&fitted.text).unwrap_or_else(|g| panic!("a fit pass 5 renders does not read: {g}\n{}", fitted.text));
}
