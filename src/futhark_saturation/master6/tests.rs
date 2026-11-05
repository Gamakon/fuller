//! Master 6's e-graph (plan 7, task F5): every extracted candidate writes back and renders its own text; the
//! references' variants are master 5's texts; the extracted variable is in scope where written and is the same
//! resolved variable; a shared part used under two environments evaluates bit for bit as the unshared program; a tree
//! with a failing requirement is not saturated.

use super::*;
use crate::futhark_sig::{build6, const_len6, problem, random_gene, REFERENCES6};
use crate::futhark_table::GeneRng;
use crate::geneframe::Ty;
use crate::saturated_fold::{fold_saturated, Budget};
use crate::typed_gep::{express, write_back, GeneLayout};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// The phylu start population's settings for the variants (`start.rs`: `variants(.., 3, 32, 2 * rows)`, 24 rows).
const ITERS: u32 = 3;
const MEMBERS: usize = 32;
const N: usize = 48;

/// A problem's gene frame: its layout at the problem's program size, its constant pools and a padding gene.
fn frame(e: &Egraph6) -> (GeneLayout, BTreeMap<Ty, u32>, Vec<u32>) {
    let size = problem(&e.problem).unwrap().program_size;
    let layout = GeneLayout::from_program_size(&e.contract, e.kingdom.root.data_type, size).unwrap();
    let consts = const_len6(&e.kingdom);
    let pad = random_gene(&e.contract, &layout, &consts, &mut GeneRng::new(0xF5)).unwrap();
    (layout, consts, pad)
}

/// A tree written onto `pad` and expressed again.
fn round_trip(e: &Egraph6, layout: &GeneLayout, consts: &BTreeMap<Ty, u32>, pad: &[u32], t: &Tree) -> Tree {
    let gene = write_back(&e.contract, layout, pad, t).unwrap_or_else(|d| panic!("{}: a candidate does not write back: {d:?}", e.problem));
    express(&e.contract, layout, &gene, consts).unwrap_or_else(|d| panic!("{}: a written candidate does not express: {d:?}", e.problem)).tree()
}

fn egraphs() -> BTreeMap<&'static str, Egraph6> {
    REFERENCES6.iter().map(|r| r.problem).collect::<BTreeSet<_>>().into_iter().map(|p| (p, Egraph6::load(p).unwrap_or_else(|e| panic!("{p}: {e}")))).collect()
}

/// The generated datatype and ruleset load into egglog for every view, and every symbol has its constructors.
#[test]
fn every_view_egraph_loads() {
    for (problem, ..) in crate::futhark_view::master6::COMMITTED6 {
        let e = Egraph6::load(problem).unwrap_or_else(|e| panic!("{problem}: {e}"));
        let mut g = EGraph::default();
        g.parse_and_run_program(None, &e.texts.datatype).unwrap_or_else(|x| panic!("{problem}: {x}"));
        g.parse_and_run_program(None, &e.texts.rules).unwrap_or_else(|x| panic!("{problem}: {x}"));
    }
}

/// EVERY EXTRACTED CANDIDATE WRITES BACK AND RENDERS ITS OWN TEXT, AND ITS VARIABLES ARE IN SCOPE: for every reference
/// program (15), its e-graph variants at the start population's settings are all placed (no extracted term is dropped:
/// a variable not in scope where it is written, or not the resolved variable it was extracted as, is a bug), each
/// writes back onto a gene of the problem's size, expresses to a tree that renders the candidate's own text and
/// resolves to the candidate's resolved tree, and none is the reference's own text.
#[test]
fn every_extracted_candidate_writes_back_and_renders_its_own_text() {
    let es = egraphs();
    let mut report = Vec::new();
    for r in &REFERENCES6 {
        let key = r.key();
        let e = &es[r.problem];
        let (layout, consts, pad) = frame(e);
        let tree = build6(&e.kingdom, &(r.master6)()).unwrap_or_else(|x| panic!("{key}: {x}"));
        let own = e.program(&tree).unwrap().text;
        let vs = e.variants(&tree, ITERS, MEMBERS, N).unwrap_or_else(|x| panic!("{key}: {x}"));
        assert!(vs.saturated, "{key}: a reference has no failing requirement and is saturated");
        assert_eq!(vs.dropped, 0, "{key}: {} extracted terms are no program where they stand", vs.dropped);
        for v in &vs.trees {
            let text = e.program(v).unwrap().text;
            assert_ne!(text, own, "{key}: a variant is the reference itself");
            let back = round_trip(e, &layout, &consts, &pad, v);
            assert_eq!(e.program(&back).unwrap().text, text, "{key}: a candidate written back renders another text");
            assert_eq!(e.resolved(&back).unwrap(), e.resolved(v).unwrap(), "{key}: a candidate written back resolves to another tree");
        }
        report.push(format!("{key} {}", vs.trees.len()));
    }
    eprintln!("variants a reference: {}", report.join(", "));
}

/// The master 5 variants' texts of a reference: master 5's own e-graph and renderer (`futhark_sig::variants`,
/// `program`), at the start population's settings.
fn master5_variant_texts(r: &crate::futhark_sig::Reference6) -> Vec<String> {
    let inst = problem(r.problem).unwrap().instance;
    let k = crate::futhark_sig::load(&inst).unwrap();
    let sig = crate::futhark_sig::signature(&inst).unwrap();
    let bridge = crate::futhark_sig::Bridge::load(&inst).unwrap();
    let tree = crate::futhark_sig::build(&k, &(r.master5)()).unwrap();
    crate::futhark_sig::variants(&k, &bridge, &tree, ITERS, MEMBERS, N).unwrap().iter().map(|v| crate::futhark_sig::program(&k, &sig, v).unwrap()).collect()
}

fn master6_variant_texts(e: &Egraph6, r: &crate::futhark_sig::Reference6) -> Vec<String> {
    let tree = build6(&e.kingdom, &(r.master6)()).unwrap();
    e.variants(&tree, ITERS, MEMBERS, N).unwrap().trees.iter().map(|v| e.program(v).unwrap().text).collect()
}

/// THE VARIANTS ARE TODAY'S TEXTS: every reference of reduce, scan, histogram, pathfinder and A7 gives, through master
/// 6's e-graph, the variants master 5's e-graph gives, text for text and in the same order (master 5 live; k-means
/// below, from its fixture, since master 5's k-means view takes minutes to load).
#[test]
fn the_variants_are_todays_texts() {
    let es = egraphs();
    for r in REFERENCES6.iter().filter(|r| r.problem != "kmeans") {
        let want = master5_variant_texts(r);
        let got = master6_variant_texts(&es[r.problem], r);
        assert!(got == want, "{}: master 6 gives {} variants, master 5 {}; the first that differ:\n{:?}", r.key(), got.len(), want.len(), got.iter().zip(&want).find(|(a, b)| a != b));
    }
}

/// Today's (master 5's) variants of the k-means reference, one after a line `==== <i> ====`.
const KMEANS_VARIANTS5: &str = include_str!("../../../tests/fixtures/futhark_m6_kmeans_variants.txt");
const KMEANS_VARIANTS5_PATH: &str = "tests/fixtures/futhark_m6_kmeans_variants.txt";

fn fixture_texts(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in text.split_inclusive('\n') {
        if line.starts_with("==== ") {
            out.push(String::new());
        } else if let Some(last) = out.last_mut() {
            last.push_str(line);
        }
    }
    out
}

fn kmeans_reference() -> &'static crate::futhark_sig::Reference6 {
    REFERENCES6.iter().find(|r| r.problem == "kmeans" && r.form == "reference").unwrap()
}

/// THE K-MEANS REFERENCE'S VARIANTS ARE TODAY'S TEXTS: master 6's e-graph gives, text for text and in order, the
/// variants master 5's gives (stored in the fixture; [`the_kmeans_fixture_is_master_5s_variants`] checks the fixture
/// against master 5 live).
#[test]
fn the_kmeans_reference_variants_are_todays_texts() {
    let e = Egraph6::load("kmeans").unwrap();
    let want = fixture_texts(KMEANS_VARIANTS5);
    assert!(!want.is_empty(), "the fixture holds no variant");
    let got = master6_variant_texts(&e, kmeans_reference());
    assert!(got == want, "k-means: master 6 gives {} variants, today {}; the first that differ:\n{:?}", got.len(), want.len(), got.iter().zip(&want).find(|(a, b)| a != b));
    eprintln!("k-means reference: {} variants, today's texts", got.len());
}

/// The fixture is master 5's k-means variants, live (minutes: master 5's k-means view). With `FULLER_F5_WRITE=1` it
/// writes the fixture and FAILS, so a rewrite never passes as a check.
#[test]
#[ignore = "minutes: master 5's k-means view; run with --ignored"]
fn the_kmeans_fixture_is_master_5s_variants() {
    let texts = master5_variant_texts(kmeans_reference());
    if std::env::var("FULLER_F5_WRITE").is_ok_and(|v| v == "1") {
        let body: String = texts.iter().enumerate().map(|(i, t)| format!("==== {i} ====\n{t}")).collect();
        std::fs::write(Path::new(env!("CARGO_MANIFEST_DIR")).join(KMEANS_VARIANTS5_PATH), body).unwrap();
        panic!("wrote {} variants to {KMEANS_VARIANTS5_PATH}; run again without FULLER_F5_WRITE", texts.len());
    }
    assert_eq!(texts, fixture_texts(KMEANS_VARIANTS5), "the fixture is not master 5's k-means variants");
}

/// A7's e-graph finds the winner: the reference's variants hold the winning program (the transposed product
/// computed directly).
#[test]
fn the_a7_reference_variants_hold_the_winner() {
    let e = Egraph6::load("a7").unwrap();
    let r = |form: &str| REFERENCES6.iter().find(|r| r.problem == "a7" && r.form == form).unwrap();
    let winner = e.program(&build6(&e.kingdom, &(r("winner").master6)()).unwrap()).unwrap().text;
    let texts = master6_variant_texts(&e, r("reference"));
    assert!(texts.contains(&winner), "the winner is not among {} variants", texts.len());
}

/// THE EXTRACTED VARIABLE IS IN SCOPE AND THE SAME RESOLVED VARIABLE: a tree imported and extracted with no rule run
/// comes back as itself, every variable by the word that names its variable where it stands, even when the tree's
/// words were other words naming the same variables; a resolved variable carried where it is not in scope is no
/// program there (`None`), never a variable chosen by accident.
#[test]
fn the_extracted_variable_is_in_scope_and_the_same_resolved_variable() {
    let es = egraphs();
    for r in &REFERENCES6 {
        let e = &es[r.problem];
        let tree = build6(&e.kingdom, &(r.master6)()).unwrap();
        // The same tree with every variable's word moved by its count: the same variables, other words.
        let mut shifted = tree.clone();
        let sites = e.variables(&tree).unwrap();
        for (path, site) in &sites {
            let mut t = &mut shifted;
            for k in path {
                t = &mut t.kids[*k as usize];
            }
            t.const_index = Some(site.place + site.count);
        }
        assert_eq!(e.import(&shifted).unwrap(), e.import(&tree).unwrap(), "{}: the import reads words, not variables", r.key());
        let mut g = e.saturated(&e.import(&shifted).unwrap(), 0).unwrap();
        let (sort, value) = g.eval_expr(&exprs::var("__root")).unwrap();
        let mut td = TermDag::default();
        let (_, id) = hff_extract(&g, &mut td, value, sort, &|_: &TermDag, _: TermId| 1.0, 1)[0];
        let back = e.to_tree(&td, id).unwrap().unwrap_or_else(|| panic!("{}: the tree itself is no program", r.key()));
        assert_eq!(back, tree, "{}: the extracted tree is not the tree, word for word", r.key());
    }
    // An index of depth 1 at the root, where nothing is bound: not in scope, so no program.
    let e = &es["pathfinder"];
    let var = e.kingdom.symbols.iter().position(|s| s.name == "var@i64").unwrap() as SymbolId;
    let walls = build6(&e.kingdom, "(row@a.i32 walls zero@i64)").unwrap();
    let mut stray = walls.clone();
    stray.kids[1] = Tree { symbol: var, const_index: pack(Binder { depth: 1, ordinal: 0 }), kids: Vec::new() };
    assert_eq!(e.place(&walls).unwrap().as_ref(), Some(&walls));
    assert_eq!(e.place(&stray).unwrap(), None, "a variable not in scope was written");
}

/// A TREE WITH A FAILING REQUIREMENT IS NOT SATURATED: the histogram's planted slowdown with its tabulate over `k`
/// (so its indices and values differ in length from the input's `n`: `fit.pair`, `clampix`) holds `map2.min vs vs`,
/// which a rule would reduce to `vs`; it gives no variant, and the fold keeps it as submitted while it saturates the
/// planted slowdown beside it.
#[test]
fn a_tree_with_a_failing_requirement_is_not_saturated() {
    let e = Egraph6::load("histogram").unwrap();
    let failing = build6(&e.kingdom, "(hist.add@a.i32 k (tab@a.i32[i] k (mod@i32 k (get@i32 is i))) (map2.min@a.i32 vs vs))").unwrap();
    let planted = build6(&e.kingdom, crate::futhark_sig::HISTOGRAM_PLANTED6).unwrap();
    assert!(!e.decode(&failing).unwrap().failures().is_empty(), "the tree fails a requirement");
    assert!(!e.saturable(&failing).unwrap() && e.saturable(&planted).unwrap());
    let vs = e.variants(&failing, ITERS, MEMBERS, N).unwrap();
    assert!(!vs.saturated && vs.trees.is_empty(), "a tree with a failing requirement was saturated");
    let (layout, consts, pad) = frame(&e);
    let trees = [failing.clone(), planted.clone()];
    let genes: Vec<Vec<u32>> = trees.iter().map(|t| write_back(&e.contract, &layout, &pad, t).unwrap()).collect();
    let trees: Vec<Tree> = genes.iter().map(|g| express(&e.contract, &layout, g, &consts).unwrap().tree()).collect();
    let fold = fold_saturated(&mut crate::population_fold::AdfTable::default(), &FoldEgraph6 { egraph: &e }, &e.contract, &layout, &consts, &genes, &trees, &Budget::default()).unwrap();
    assert_eq!(fold.kept, 1);
    assert_eq!((&fold.genes[0], &fold.forms[0]), (&genes[0], &trees[0]), "the failing tree was rewritten");
    let reference = e.program(&build6(&e.kingdom, crate::futhark_sig::HISTOGRAM_REFERENCE6).unwrap()).unwrap().text;
    assert_eq!(e.program(&fold.forms[1]).unwrap().text, reference, "the planted slowdown beside it was not saturated to the reference");
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("fuller_f5_{name}_{}", std::process::id()))
}

/// A program as a module of a batch with an entry `e<i>` calling it.
fn as_entry(i: usize, text: &str) -> String {
    let start = text.find("entry main ").expect("a program has its entry");
    let line = &text[start..start + text[start..].find(" =\n").expect("the entry's header ends")];
    let names: Vec<&str> = line.split('(').skip(1).filter_map(|p| p.split_once(':').map(|(n, _)| n.trim())).collect();
    format!("module m{i} = {{\n{}}}\n\n{} = m{i}.main {}\n\n", text.replacen("entry main ", "def main ", 1), line.replacen("entry main ", &format!("entry e{i} "), 1), names.join(" "))
}

fn run(exe: &Path, entry: &str, stdin: &[u8]) -> crate::futhark_sig::Array {
    let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(exe).arg("-e").arg(entry).arg("-b"), Duration::from_secs(120), Some(stdin)).unwrap();
    assert!(out.status.success(), "{entry}: {}", String::from_utf8_lossy(&out.stderr));
    crate::futhark_sig::read_one(&out.stdout).unwrap()
}

/// A SHARED PART USED UNDER TWO DIFFERENT ENVIRONMENTS EVALUATES BIT FOR BIT AS THE UNSHARED PROGRAM. In k-means, the
/// part `tab[c] k (get zeros_k c)` stands under `tabrows[i]` in one individual (the `i64`s in scope `i`, `c`: `c` is
/// the 2nd of 2) and under a `let` of a `[]f32` in the other (`c` the 1st of 1): one resolved variable, two index
/// words. The fold shares the part as an ADF whose `c` takes the one word that names it at both (1), every folded gene
/// expresses a tree that resolves to its form, and each individual's ADF program (the part a `def`), its folded tree
/// and its unshared tree give the same output bit for bit on k-means' stored cases (`c` backend).
#[test]
fn a_shared_part_under_two_environments_evaluates_as_the_unshared_program() {
    let e = Egraph6::load("kmeans").unwrap();
    let part = "(tab@a.f32[c] k (get@f32 zeros_k c))";
    let a = build6(&e.kingdom, &format!("(transpose@m.f32 (tabrows@m.f32[i] d k {part}))")).unwrap();
    let b = build6(&e.kingdom, &format!("(let.a.f32@m.f32[z] zeros_n (rowdiv@m.f32 centres {part}))")).unwrap();
    for t in [&a, &b] {
        assert!(e.saturable(t).unwrap(), "{:?}", e.decode(t).unwrap().failures());
    }
    let (pa, pb) = (vec![0, 2], vec![1, 1]);
    let word = |t: &Tree, p: &[u32]| e.variables(t).unwrap().into_iter().find(|(q, _)| q.as_slice() == p).map(|(_, s)| s).unwrap();
    assert_eq!((word(&a, &[0, 2, 1, 1]), word(&b, &[1, 1, 1, 1])), (VariableSite { place: 1, count: 2 }, VariableSite { place: 0, count: 1 }), "the part's `c` has two index words");
    let (layout, consts, pad) = frame(&e);
    let trees = vec![a.clone(), b.clone()];
    let genes: Vec<Vec<u32>> = trees.iter().map(|t| write_back(&e.contract, &layout, &pad, t).unwrap()).collect();
    let fold = fold_saturated(&mut crate::population_fold::AdfTable::default(), &FoldEgraph6 { egraph: &e }, &e.contract, &layout, &consts, &genes, &trees, &Budget::default()).unwrap();
    assert_eq!((fold.kept, fold.unshared_by_scope), (0, 0));
    assert!(!fold.used.is_empty(), "the part is not shared");
    let bodies = fold.adfs.expanded(&fold.epoch);
    let body = bodies.values().find(|b| e.kingdom.symbols[b.symbol as usize].name == "tab@a.f32").expect("the part is an ADF");
    assert_eq!(body.kids[1].kids[1].const_index, Some(1), "the part's `c` takes the word that names it in both environments");
    let cl = crate::typed_chromosome::ChromosomeLayout { normal: crate::typed_gep::Layout::new(0, vec![layout.clone()]), mains: Vec::new() };
    let mut sources = String::new();
    for (i, (t, g)) in trees.iter().zip(&fold.genes).enumerate() {
        let c = crate::typed_chromosome::Chromosome { chromosome_id: i as u64, genes: vec![g.clone()], mains: Vec::new() };
        let inlined = crate::typed_chromosome::express_chromosome(&fold.epoch, &fold.adfs, &cl, &c, &consts).unwrap().inlined(&fold.epoch, &fold.adfs).remove(0).1;
        assert_eq!(e.resolved(&inlined).unwrap(), e.resolved(&fold.forms[i]).unwrap(), "individual {i}: the folded gene does not express its form");
        let path = if i == 0 { &pa } else { &pb };
        let adf = e.kingdom.program6_with_adfs(&e.header, &e.decode(&inlined).unwrap(), std::slice::from_ref(path)).unwrap().text;
        assert!(adf.contains("def adf_0"), "{adf}");
        for (j, text) in [e.program(t).unwrap().text, e.program(&inlined).unwrap().text, adf].iter().enumerate() {
            sources.push_str(&as_entry(3 * i + j, text));
        }
    }
    let dir = scratch("shared_part");
    let exe = crate::futhark_sig::compile_c(&sources, &dir).unwrap_or_else(|x| panic!("{sources}\n{x}"));
    let cases = crate::futhark_sig::load_cases(&problem("kmeans").unwrap()).unwrap();
    for case in cases.iter().take(4) {
        for i in 0..2 {
            let want = run(&exe, &format!("e{}", 3 * i), &case.stdin);
            for j in 1..3 {
                assert_eq!(run(&exe, &format!("e{}", 3 * i + j), &case.stdin), want, "individual {i}, form {j}, on {}", case.name);
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The same tree gives the same variants twice (no hash order in the ranking).
#[test]
fn the_same_tree_gives_the_same_variants() {
    let e = Egraph6::load("pathfinder").unwrap();
    let r = REFERENCES6.iter().find(|r| r.problem == "pathfinder" && r.form == "planted").unwrap();
    assert_eq!(master6_variant_texts(&e, r), master6_variant_texts(&e, r));
}
