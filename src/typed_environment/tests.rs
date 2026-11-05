//! Tests of the decoder's passes 2 to 4 (plan section 7, task G3), on a histogram kingdom built here from G1's tables
//! as master 6 spells its rows (`kingdoms/futhark/master/6/rows.tsv`, the histogram view at depth 2).

use std::collections::BTreeMap;

use super::*;
use crate::geneframe::{Argument, Binding, DataKind, DataType, Direction, PropertyKind, PropertyRule, PropertyType, Requirement, SymbolRow, TripArgument};
use crate::kingdom_contract::relations_from_rows;
use crate::typed_gep::{express, GeneLayout};

const I32: Ty = Ty::Futhark(0);
const I64: Ty = Ty::Futhark(1);
const SIZE: Ty = Ty::Futhark(2);
const ARR: Ty = Ty::Futhark(3);

// The symbols, by their contract (and table) id.
const HIST: SymbolId = 0;
const TAB: SymbolId = 1;
const GET: SymbolId = 2;
const MOD: SymbolId = 3;
const MAP2: SymbolId = 4;
const SUM: SymbolId = 5;
const ZERO: SymbolId = 6;
const LAST: SymbolId = 7;
const LOOP: SymbolId = 8;
const VAR_I64: SymbolId = 11;
const VAR_SIZE: SymbolId = 12;
const VAR_ARR: SymbolId = 13;
const LET: SymbolId = 14;

/// `(name, inputs, output)` of every symbol, in id order.
fn rows() -> Vec<(String, Vec<Ty>, Ty)> {
    [
        ("hist.add", vec![SIZE, ARR, ARR], ARR),
        ("tab", vec![SIZE, I32], ARR),
        ("get", vec![ARR, I64], I32),
        ("mod", vec![SIZE, I32], I32),
        ("map2.add", vec![ARR, ARR], ARR),
        ("sum", vec![ARR], I32),
        ("zero", vec![], I64),
        ("last", vec![SIZE], I64),
        ("loop", vec![SIZE, ARR, ARR], ARR),
        ("const.i32", vec![], I32),
        ("var.i32", vec![], I32),
        ("var.i64", vec![], I64),
        ("var.size", vec![], SIZE),
        ("var.a.i32", vec![], ARR),
        ("let.a.i32", vec![ARR, ARR], ARR),
    ]
    .into_iter()
    .map(|(n, i, o)| (n.to_string(), i, o))
    .collect()
}

fn contract() -> KingdomContract {
    relations_from_rows("histogram G3", vec![ARR], &rows(), &["const.i32"])
        .with_variables(&["var.i32", "var.i64", "var.size", "var.a.i32"])
        .expect("variables")
        .check()
        .expect("contract")
}

fn p(text: &str) -> PropertyExpr {
    PropertyExpr::parse(text).expect(text)
}

/// The histogram view's tables at depth 2, as master 6 writes `hist.add`, `tab`, `get`, `mod`, `map2.add`, `sum`,
/// `zero`, `last`, `loop` and `let` (a binding with no trip).
fn tables() -> KingdomTables {
    let mut t = KingdomTables { kingdom: "histogram G3".to_string(), ..KingdomTables::default() };
    for (i, (name, ins, out)) in rows().into_iter().enumerate() {
        let symbol = i as i64;
        t.symbols.push(SymbolRow { symbol, symbol_name: name.clone(), alias: name.clone(), semantic_id: name, spelling: String::new() });
        for (pos, ty) in ins.iter().enumerate() {
            t.arguments.push(Argument { symbol, position: pos as u32, direction: Direction::In, data_type: *ty });
        }
        t.arguments.push(Argument { symbol, position: 0, direction: Direction::Out, data_type: out });
    }
    t.data_types = vec![
        DataType { id: I32, name: "i32".into(), kind: DataKind::Scalar, constant: Some(vec![]) },
        DataType { id: I64, name: "i64".into(), kind: DataKind::Scalar, constant: Some(vec![("bound".into(), p("1"))]) },
        DataType { id: SIZE, name: "size".into(), kind: DataKind::Scalar, constant: None },
        DataType { id: ARR, name: "[]i32".into(), kind: DataKind::Array { element: I32 }, constant: Some(vec![("len".into(), p("1"))]) },
    ];
    t.property_types = vec![
        PropertyType { id: 0, data_type: ARR, name: "len".into(), kind: PropertyKind::Size },
        PropertyType { id: 1, data_type: I64, name: "bound".into(), kind: PropertyKind::Bound },
        PropertyType { id: 2, data_type: SIZE, name: "value".into(), kind: PropertyKind::Size },
    ];
    let rule = |symbol: SymbolId, property: &str, expr: &str| PropertyRule { symbol: symbol as i64, property: property.into(), expr: p(expr) };
    t.property_rules = vec![
        rule(HIST, "len", "in0.value"),
        rule(TAB, "len", "in0.value"),
        rule(MAP2, "len", "min(in0.len, in1.len)"),
        rule(ZERO, "bound", "1"),
        rule(LAST, "bound", "in0.value"),
        rule(LOOP, "len", "in1.len"),
        rule(LET, "len", "in1.len"),
    ];
    let on = |position: u32, property: &str| Subject::Input { position, property: property.into() };
    let req = |symbol: SymbolId, subject: Subject, relation: Relation, against: &str, adapter: &str| Requirement {
        symbol: symbol as i64,
        subject,
        relation,
        against: p(against),
        adapter: adapter.into(),
    };
    let work = |symbol: SymbolId| req(symbol, Subject::Trips, Relation::Lt, "2", "subst");
    t.requirements = vec![
        req(HIST, on(2, "len"), Relation::Eq, "in1.len", "fit.pair"),
        req(HIST, Subject::Trips, Relation::Lt, "2", "subst.const"),
        work(TAB),
        req(GET, on(1, "bound"), Relation::Le, "in0.len", "clampix"),
        req(MAP2, on(1, "len"), Relation::Eq, "in0.len", "fit.pair"),
        work(MAP2),
        work(SUM),
        req(LOOP, on(2, "len"), Relation::Eq, "in1.len", "fit.state"),
        work(LOOP),
    ];
    t.bindings = vec![
        Binding { symbol: TAB as i64, position: 1, data_type: I64, properties: vec![("bound".into(), p("in0.value"))] },
        Binding { symbol: LOOP as i64, position: 2, data_type: ARR, properties: vec![("len".into(), p("in1.len"))] },
        Binding { symbol: LOOP as i64, position: 2, data_type: I64, properties: vec![("bound".into(), p("in0.value"))] },
        Binding { symbol: LET as i64, position: 1, data_type: ARR, properties: vec![("len".into(), p("in0.len"))] },
    ];
    t.trip_arguments = vec![TripArgument { symbol: TAB as i64, position: 1 }, TripArgument { symbol: LOOP as i64, position: 2 }];
    t.adapters = vec![
        Adapter { name: "clampix".into(), heavy: false, runs_at: AdapterSite::Node },
        Adapter { name: "fit.pair".into(), heavy: true, runs_at: AdapterSite::Node },
        Adapter { name: "fit.state".into(), heavy: true, runs_at: AdapterSite::Body },
        Adapter { name: "fit.root".into(), heavy: true, runs_at: AdapterSite::Root },
        Adapter { name: "subst".into(), heavy: false, runs_at: AdapterSite::Node },
        Adapter { name: "subst.const".into(), heavy: false, runs_at: AdapterSite::Node },
    ];
    t
}

/// The histogram entry `main [n] (k: i32) (is: [n]i32) (vs: [n]i32) : [k]i32`, its root environment as the view
/// writes it (`kingdoms/futhark/sig/histogram/m6/environment.tsv`): the inputs, the zero arrays, the sizes.
fn entry() -> Entry {
    let arr = |name: &str, len: &str| EntryVariable { name: name.into(), data_type: ARR, properties: vec![("len".into(), p(len))] };
    let size = |name: &str| EntryVariable { name: name.into(), data_type: SIZE, properties: vec![("value".into(), p(name))] };
    Entry {
        output: ARR,
        variables: vec![arr("is", "n"), arr("vs", "n"), arr("zeros_k", "k"), arr("zeros_n", "n"), size("k"), size("n")],
        requirements: vec![RootRequirement { property: "len".into(), relation: Relation::Eq, against: p("k"), adapter: "fit.root".into() }],
    }
}

fn f(symbol: SymbolId, kids: Vec<Tree>) -> Tree {
    Tree { symbol, const_index: None, kids }
}

fn v(symbol: SymbolId, word: u32) -> Tree {
    Tree { symbol, const_index: Some(word), kids: vec![] }
}

// The root environment's words: arrays is 0, vs 1, zeros_k 2, zeros_n 3; sizes k 0, n 1.
const W_IS: u32 = 0;
const W_VS: u32 = 1;
const W_ZEROS_K: u32 = 2;
const W_K: u32 = 0;
const W_N: u32 = 1;

/// Design section 6's reference `hist (+) 0 k (tabulate n (\i -> is[i] % k)) vs`, the `tabulate` size's word given.
fn reference(tab_size: u32) -> Tree {
    let body = f(MOD, vec![v(VAR_SIZE, W_K), f(GET, vec![v(VAR_ARR, W_IS), v(VAR_I64, 0)])]);
    f(HIST, vec![v(VAR_SIZE, W_K), f(TAB, vec![v(VAR_SIZE, tab_size), body]), v(VAR_ARR, W_VS)])
}

fn decode(tree: &Tree) -> Decoding {
    let (t, k) = (tables(), contract());
    let d = Decoder::new(&t, &k).unwrap_or_else(|e| panic!("{e:#?}"));
    d.decode(&entry(), tree).unwrap_or_else(|e| panic!("{e}"))
}

fn binder(depth: u32, ordinal: u32) -> Binder {
    Binder { depth, ordinal }
}

fn props(pairs: &[(&str, &str)]) -> BTreeMap<String, PropertyExpr> {
    pairs.iter().map(|(k, e)| (k.to_string(), p(e))).collect()
}

/// Design section 6, the reference: every requirement holds, and each annotation is the design's table's.
#[test]
fn the_histogram_reference_holds_every_requirement() {
    let d = decode(&reference(W_N));
    assert_eq!(d.failures(), vec![], "{d:#?}");
    assert!(d.root_outcomes.iter().all(|o| o.holds) && d.root_outcomes.len() == 1);
    let node = |path: &[u32]| d.node(path).expect("node");
    // hist.add: len = k; indices.len = values.len: n = n.
    assert_eq!(node(&[]).properties, props(&[("len", "k")]));
    let hist_pair = &node(&[]).outcomes[0];
    assert_eq!((&hist_pair.subject, &hist_pair.against, hist_pair.holds), (&p("n"), &p("n"), true));
    // tabulate n: len = n, binds i of bound n, one trip.
    assert_eq!(node(&[1]).properties, props(&[("len", "n")]));
    let body_env = &node(&[1, 1]).environment;
    assert_eq!((body_env.depth, body_env.trips), (1, 1));
    assert_eq!(in_scope(body_env, I64).iter().map(|v| (v.binder, v.properties.clone())).collect::<Vec<_>>(), vec![(binder(1, 0), props(&[("bound", "n")]))]);
    // the variables: k, n, is, i, vs.
    assert_eq!(node(&[0]).choice, Some(Choice::Variable(binder(0, 4))));
    assert_eq!(node(&[1, 0]).choice, Some(Choice::Variable(binder(0, 5))));
    assert_eq!(node(&[1, 1, 1, 0]).choice, Some(Choice::Variable(binder(0, 0))));
    assert_eq!(node(&[1, 1, 1, 1]).choice, Some(Choice::Variable(binder(1, 0))));
    assert_eq!(node(&[2]).choice, Some(Choice::Variable(binder(0, 1))));
    assert_eq!(node(&[2]).properties, props(&[("len", "n")]));
    // get: i.bound <= is.len, n <= n.
    let get = &node(&[1, 1, 1]).outcomes[0];
    assert_eq!((&get.subject, &get.against, get.holds), (&p("n"), &p("n"), true));
    assert!(d.root.substituted.is_none() && node(&[1]).substituted.is_none());
}

/// Design section 6, the mutation: the `tabulate` size's index word chooses `k`, which fails exactly two requirements,
/// `hist`'s lengths (`fit.pair`) and `get`'s bound (`clampix`); the result's length is still `k`.
#[test]
fn the_tabulate_size_choosing_k_fails_exactly_two() {
    let d = decode(&reference(W_K));
    let t = tables();
    let failures = d.failures();
    assert_eq!(
        failures,
        vec![
            Failure::Adapter { path: vec![], requirement: 0, adapter: "fit.pair".into() },
            Failure::Adapter { path: vec![1, 1, 1], requirement: 3, adapter: "clampix".into() },
        ],
        "{d:#?}"
    );
    assert_eq!(t.requirements[0].symbol, HIST as i64);
    assert_eq!(t.requirements[3].symbol, GET as i64);
    let get = &d.node(&[1, 1, 1]).expect("get").outcomes[0];
    assert_eq!((&get.subject, &get.against), (&p("k"), &p("n")), "i's bound is k, is's length n");
    assert_eq!(d.root.properties, props(&[("len", "k")]));
    assert!(d.root_outcomes[0].holds);
    // Any word congruent to k's modulo the two sizes in scope chooses k too.
    assert_eq!(decode(&reference(W_K + 2)).failures(), failures);
    assert_eq!(decode(&reference(W_N + 4)).failures(), vec![]);
}

/// Master 6 README 5.1 and plan 7.1 "Literal indices": `0i64` (bound 1) holds `bound <= len` for every length, and
/// `{size} - 1` (bound the size) for every array of that length; another size fails.
#[test]
fn literal_indices_hold_bound_at_most_len_for_every_length() {
    let (t, k) = (tables(), contract());
    let d = Decoder::new(&t, &k).expect("decoder");
    let arr = |name: &str, len: &str| EntryVariable { name: name.into(), data_type: ARR, properties: vec![("len".into(), p(len))] };
    let size = |name: &str| EntryVariable { name: name.into(), data_type: SIZE, properties: vec![("value".into(), p(name))] };
    let sizes = ["d", "k", "n"];
    let entry = Entry { output: ARR, variables: vec![arr("a_d", "d"), arr("a_k", "k"), arr("a_n", "n"), size("d"), size("k"), size("n")], requirements: vec![] };
    // An array of each length: a root input, a tabulate and a histogram of each size (the histogram's two operands
    // the same array, so its own requirement holds).
    let mut arrays: Vec<(Tree, PropertyExpr)> = Vec::new();
    for (w, s) in sizes.iter().enumerate() {
        let w = w as u32;
        arrays.push((v(VAR_ARR, w), p(s)));
        arrays.push((f(TAB, vec![v(VAR_SIZE, w), v(10, 0)]), p(s)));
        arrays.push((f(HIST, vec![v(VAR_SIZE, w), v(VAR_ARR, 2), v(VAR_ARR, 2)]), p(s)));
    }
    for (array, len) in &arrays {
        // tab(n, get(array, index)) is legal at the root.
        let at = |index: Tree| d.decode(&entry, &f(TAB, vec![v(VAR_SIZE, 2), f(GET, vec![array.clone(), index])])).expect("decodes");
        let zero = at(f(ZERO, vec![]));
        assert_eq!(zero.failures(), vec![], "0i64 into an array of length {len}");
        assert_eq!(zero.node(&[1, 1]).expect("zero").properties, props(&[("bound", "1")]));
        for (w, s) in sizes.iter().enumerate() {
            let last = at(f(LAST, vec![v(VAR_SIZE, w as u32)]));
            assert_eq!(last.node(&[1, 1]).expect("last").properties, props(&[("bound", s)]));
            assert_eq!(last.failures().is_empty(), p(s) == *len, "{s} - 1 into an array of length {len}: {:#?}", last.failures());
        }
    }
    // A literal index of bound 1 into a length that is a min of sizes, or the constant array's length 1, also holds.
    for len in ["min(k, n)", "min(d, k, n)", "1"] {
        assert!(Relation::Le.holds(&p("1"), &p(len)), "{len}");
    }
}

/// Plan 7.5 risk 3: the resolved tree names variables by binder, so equal words under different environments
/// resolve to different trees, and different words naming the same variable resolve equal.
#[test]
fn the_resolved_tree_names_variables_by_binder() {
    let (t, k) = (tables(), contract());
    let d = Decoder::new(&t, &k).expect("decoder");
    let e = entry();
    let get_i = f(GET, vec![v(VAR_ARR, W_IS), v(VAR_I64, 1)]);
    // Under one tabulate, word 1 of the one i64 in scope is the outer i; under two, it is the inner i.
    let one = f(TAB, vec![v(VAR_SIZE, W_N), get_i.clone()]);
    let two = f(TAB, vec![v(VAR_SIZE, W_N), f(SUM, vec![f(TAB, vec![v(VAR_SIZE, W_K), get_i.clone()])])]);
    let (r1, r2) = (d.resolve(&e, &one).expect("one"), d.resolve(&e, &two).expect("two"));
    let (g1, g2) = (&r1.kids[1], &r2.kids[1].kids[0].kids[1]);
    assert_eq!((&one.kids[1], &two.kids[1].kids[0].kids[1]), (&get_i, &get_i), "the same words");
    assert_ne!(g1, g2, "different variables");
    assert_eq!(g1.kids[1].choice, Some(Choice::Variable(binder(1, 0))));
    assert_eq!(g2.kids[1].choice, Some(Choice::Variable(binder(2, 0))));
    // Words 0 and 4 of the four arrays in scope both name `is`; 1 names `vs`.
    let with = |w: u32| f(TAB, vec![v(VAR_SIZE, W_N), f(GET, vec![v(VAR_ARR, w), f(ZERO, vec![])])]);
    assert_ne!(with(0), with(4));
    assert_eq!(d.resolve(&e, &with(0)).expect("0"), d.resolve(&e, &with(4)).expect("4"));
    assert_ne!(d.resolve(&e, &with(0)).expect("0"), d.resolve(&e, &with(1)).expect("1"));
    // The inverse: the least word naming a variable, in the one order.
    let dec = d.decode(&e, &two).expect("two");
    let env = &dec.node(&[1, 0, 1]).expect("inner body").environment;
    assert_eq!(env.index_of(I64, binder(2, 0)), Some(1));
    assert_eq!(env.index_of(I64, binder(1, 0)), Some(0));
    assert_eq!(env.choose(I64, 7).map(|v| v.binder), Some(binder(2, 0)));
    assert_eq!(env.index_of(ARR, binder(0, 3)), Some(3));
    assert_eq!(env.index_of(SIZE, binder(0, 5)), Some(1));
}

/// Master 6 README 5.3: the work bound `env.trips < 2`, its substitution by the first input of the node's own data
/// type else by the type's constant, and a heavy body adapter that would run at trips 2 substituting its node.
#[test]
fn the_work_bound_substitutes_as_an_annotation() {
    // The outer tabulate is of k, so the result meets the entry's `len = k`.
    let inner = |body: Tree| f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![f(TAB, vec![v(VAR_SIZE, W_N), body])])]);
    // map2 at trips 2 renders as its input 0 (`is`), so get sees is's length.
    let map2 = decode(&inner(f(GET, vec![f(MAP2, vec![v(VAR_ARR, W_IS), v(VAR_ARR, W_VS)]), v(VAR_I64, 1)])));
    assert_eq!(map2.failures(), vec![Failure::Substituted { path: vec![1, 0, 1, 0], by: Substitution::Input(0), requirement: 5 }], "{map2:#?}");
    let m = map2.node(&[1, 0, 1, 0]).expect("map2");
    assert_eq!((m.environment.trips, &m.properties), (2, &props(&[("len", "n")])));
    // sum at trips 2 has no input of its own type: its constant.
    let sum = decode(&inner(f(SUM, vec![v(VAR_ARR, W_IS)])));
    assert_eq!(sum.failures(), vec![Failure::Substituted { path: vec![1, 0, 1], by: Substitution::Constant, requirement: 6 }]);
    // A loop at trips 1 whose body (at trips 2) is of another length: fit.state would run at trips 2, so the loop is
    // substituted by its state (`zeros_k`, length k) instead.
    let mismatched = f(LOOP, vec![v(VAR_SIZE, W_K), v(VAR_ARR, W_ZEROS_K), v(VAR_ARR, W_IS)]);
    let deep = decode(&f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![mismatched.clone()])]));
    assert_eq!(deep.failures(), vec![Failure::Substituted { path: vec![1, 0], by: Substitution::Input(1), requirement: 7 }], "{deep:#?}");
    assert_eq!(deep.node(&[1, 0]).expect("loop").properties, props(&[("len", "k")]));
    let body_env = &deep.node(&[1, 0, 2]).expect("body").environment;
    assert_eq!((body_env.trips, body_env.depth), (2, 2));
    assert_eq!(in_scope(body_env, ARR).last().map(|v| (v.binder, v.properties.clone())), Some((binder(2, 0), props(&[("len", "k")]))));
    // The same loop at the root: fit.state runs at trips 1, so it is an adapter, not a substitution.
    let shallow = decode(&mismatched);
    assert_eq!(shallow.failures(), vec![Failure::Adapter { path: vec![], requirement: 7, adapter: "fit.state".into() }]);
    assert!(shallow.root_outcomes[0].holds, "the loop's length is its state's, k");
}

/// A gene with no `var.T` in scope reads its type's constant; `size` has none, so a size word with no size in scope is
/// a bug, not a guess.
#[test]
fn a_variable_with_none_in_scope_reads_the_constant() {
    let (t, k) = (tables(), contract());
    let d = Decoder::new(&t, &k).expect("decoder");
    let bare = Entry { output: ARR, variables: vec![EntryVariable { name: "n".into(), data_type: SIZE, properties: vec![("value".into(), p("n"))] }], requirements: vec![] };
    let x = d.decode(&bare, &f(MAP2, vec![v(VAR_ARR, 5), f(TAB, vec![v(VAR_SIZE, 9), v(10, 3)])])).expect("decodes");
    assert_eq!(x.node(&[0]).expect("var").choice, Some(Choice::Constant));
    assert_eq!(x.node(&[0]).expect("var").properties, props(&[("len", "1")]));
    assert_eq!(x.root.properties, props(&[("len", "min(1, n)")]));
    assert_eq!(x.failures(), vec![Failure::Adapter { path: vec![], requirement: 4, adapter: "fit.pair".into() }]);
    // An entry with no size, where `size` has no constant, is a bug of the entry, found before any tree is read.
    let none = Entry { output: ARR, variables: vec![], requirements: vec![] };
    let e = d.decode(&none, &f(TAB, vec![v(VAR_SIZE, 0), v(10, 0)])).expect_err("no size in scope");
    assert_eq!(e, "our software has a bug: kingdom histogram G3: the entry has no parameter of Futhark(2), which has no constant, so var.size has nothing to read at the root");
    assert_eq!(d.check_entry(&none), Err(vec![e]));
    let e = d.decode(&bare, &f(TAB, vec![Tree { symbol: VAR_SIZE, const_index: None, kids: vec![] }, v(10, 0)])).expect_err("no word");
    assert!(e.contains("carries no word"), "{e}");
}

/// A small deterministic generator for random genes.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// A gene valid by layout: each head cell any symbol of its type, each tail cell a terminal, each Dc and Dv cell a word.
fn random_gene(k: &KingdomContract, layout: &GeneLayout, consts: &BTreeMap<Ty, u32>, rng: &mut Rng) -> Vec<u32> {
    use crate::typed_gep::Domain;
    let mut gene = vec![0u32; layout.width];
    for s in layout.spans() {
        let (fs, ts) = (k.functions_of(s.ty), k.terminals_of(s.ty));
        for cell in &mut gene[s.off..s.off + s.len] {
            *cell = match s.domain {
                Domain::Head => {
                    let i = rng.below(fs.len() + ts.len());
                    if i < fs.len() {
                        fs[i]
                    } else {
                        ts[i - fs.len()]
                    }
                }
                Domain::Tail => ts[rng.below(ts.len())],
                Domain::Dc => rng.below(consts.get(&s.ty).copied().unwrap_or(1).max(1) as usize) as u32,
                Domain::Dv => rng.below(crate::typed_gep::DV_WORDS as usize) as u32,
            };
        }
    }
    gene
}

/// Random histogram genes, expressed (pass 1) and decoded: every one decodes, the same gene twice to the same
/// annotations, and every failure names an adapter of the tables or a substitution.
#[test]
fn every_random_histogram_gene_decodes_the_same_twice() {
    let (t, k) = (tables(), contract());
    let d = Decoder::new(&t, &k).expect("decoder");
    let layout = GeneLayout::from_program_size(&k, ARR, 12).expect("layout");
    let consts = BTreeMap::from([(I32, 5u32)]);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut with_failures, mut substituted) = (0, 0);
    for g in 0..300 {
        let gene = random_gene(&k, &layout, &consts, &mut rng);
        let tree = express(&k, &layout, &gene, &consts).unwrap_or_else(|e| panic!("gene {g}: {e:?}")).tree();
        let a = d.decode(&entry(), &tree).unwrap_or_else(|e| panic!("gene {g}: {e}"));
        let b = d.decode(&entry(), &tree).expect("again");
        assert_eq!(a, b, "gene {g}");
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "gene {g}");
        let fs = a.failures();
        with_failures += usize::from(!fs.is_empty());
        substituted += usize::from(fs.iter().any(|f| matches!(f, Failure::Substituted { .. })));
        for f in &fs {
            if let Failure::Adapter { adapter, .. } | Failure::Root { adapter, .. } = f {
                assert!(t.adapters.iter().any(|a| &a.name == adapter), "gene {g}: {adapter}");
            }
        }
    }
    eprintln!("300 random histogram genes: {with_failures} with a failing requirement, {substituted} with a substitution");
    assert!(with_failures > 0 && substituted > 0, "the genes exercise failures ({with_failures}) and substitutions ({substituted})");
}

const CHILD: &str = "FULLER_G3_DECODE_CHILD";

/// The annotations of a fixed set of trees (the reference, the mutation, the work-bound cases), as text.
fn annotations_text() -> String {
    let mismatched = f(LOOP, vec![v(VAR_SIZE, W_K), v(VAR_ARR, W_ZEROS_K), v(VAR_ARR, W_IS)]);
    let trees = [reference(W_N), reference(W_K), f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![mismatched.clone()])]), mismatched];
    trees.iter().map(|t| format!("{:?}\n{:?}\n", decode(t), decode(t).failures())).collect()
}

/// The environment and the annotations are the same across two processes (no `HashMap` order anywhere): this test
/// runs itself again as a child process and compares the child's text with its own.
#[test]
fn the_annotations_are_the_same_in_two_processes() {
    let mine = annotations_text();
    if std::env::var_os(CHILD).is_some() {
        println!("<<<{mine}>>>");
        return;
    }
    let exe = std::env::current_exe().expect("the test binary");
    let out = std::process::Command::new(exe)
        .args(["--exact", "typed_environment::tests::the_annotations_are_the_same_in_two_processes", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .output()
        .expect("the child runs");
    assert!(out.status.success(), "the child failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    let theirs = text.split_once("<<<").and_then(|(_, r)| r.split_once(">>>")).map(|(t, _)| t).expect("the child's annotations");
    assert!(mine.contains("Binder") && mine.contains("fit.pair"));
    assert_eq!(theirs, mine);
}

/// A kingdom with no property rows (symbolic regression, regex) decodes every gene with no failure, no substitution
/// and an empty environment: today's kingdom.
#[test]
fn a_kingdom_without_property_rows_has_no_failures() {
    use crate::kingdom_contract::{regex_relations, sr_relations};
    for relations in [sr_relations(), regex_relations()] {
        let k = relations.check().expect("contract");
        let rows: Vec<(SymbolRow, Vec<Ty>, Ty)> = (0..k.n_symbols() as SymbolId)
            .map(|s| {
                let name = k.name(s).to_string();
                (SymbolRow { symbol: s as i64, symbol_name: name.clone(), alias: name.clone(), semantic_id: name, spelling: String::new() }, k.signature(s).to_vec(), k.output(s))
            })
            .collect();
        let t = KingdomTables::from_signatures(&relations.kingdom, rows);
        assert!(!t.has_properties());
        let d = Decoder::new(&t, &k).unwrap_or_else(|e| panic!("{}: {e:#?}", relations.kingdom));
        let root = k.roots()[0];
        let layout = GeneLayout::from_program_size(&k, root, 16).expect("layout");
        let consts: BTreeMap<Ty, u32> = k.types().iter().filter(|t| k.constant_capable(**t)).map(|t| (*t, 5)).collect();
        let entry = Entry { output: root, variables: vec![], requirements: vec![] };
        let mut rng = Rng(7);
        for g in 0..200 {
            let gene = random_gene(&k, &layout, &consts, &mut rng);
            let tree = express(&k, &layout, &gene, &consts).expect("expresses").tree();
            let dec = d.decode(&entry, &tree).unwrap_or_else(|e| panic!("{} gene {g}: {e}", relations.kingdom));
            assert_eq!(dec.failures(), vec![], "{} gene {g}", relations.kingdom);
            fn empty(n: &Node) -> bool {
                n.environment == Environment::default() && n.properties.is_empty() && n.outcomes.is_empty() && n.kids.iter().all(empty)
            }
            assert!(empty(&dec.root), "{} gene {g}", relations.kingdom);
            let mut words = Vec::new();
            fn dc(n: &Node, out: &mut Vec<u32>) {
                if let Some(Choice::Dc(w)) = n.choice {
                    out.push(w);
                }
                n.kids.iter().for_each(|k| dc(k, out));
            }
            dc(&dec.root, &mut words);
            assert!(words.iter().all(|w| *w < 5));
        }
    }
}

/// The decoder reports tables that disagree with the contract as a bug.
#[test]
fn the_decoder_checks_the_contract_against_the_tables() {
    let k = contract();
    let mut t = tables();
    t.arguments.iter_mut().find(|a| a.symbol == GET as i64 && a.position == 1 && a.direction == Direction::In).expect("get in1").data_type = SIZE;
    t.requirements.retain(|q| q.symbol != GET as i64);
    t.check().expect("the edited tables check on their own");
    let e = Decoder::new(&t, &k).expect_err("a mismatch");
    assert!(e.iter().any(|m| m.contains("contract symbol 2 (get)")), "{e:#?}");
    let mut t = tables();
    t.property_rules.push(PropertyRule { symbol: VAR_ARR as i64, property: "len".into(), expr: p("1") });
    let e = Decoder::new(&t, &k).expect_err("a rule on a variable");
    assert!(e.iter().any(|m| m.contains("the variable terminal var.a.i32 has property rules")), "{e:#?}");
}

/// G3 review finding 1 (the lead's decision): a `hist` the work bound substitutes renders its CONSTANT (`subst.const`),
/// the histogram with nothing counted, never its indices or its values, though both are of its own data type.
#[test]
fn a_substituted_histogram_renders_its_constant() {
    // tab k (sum (tab n (get (hist k is vs) i))): the hist is at trips 2.
    let body = f(GET, vec![f(HIST, vec![v(VAR_SIZE, W_K), v(VAR_ARR, W_IS), v(VAR_ARR, W_VS)]), v(VAR_I64, 1)]);
    let d = decode(&f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![f(TAB, vec![v(VAR_SIZE, W_N), body])])]));
    let t = tables();
    assert_eq!((t.requirements[1].symbol, &t.requirements[1].adapter), (HIST as i64, &"subst.const".to_string()));
    assert_eq!(
        d.failures(),
        vec![
            Failure::Adapter { path: vec![1, 0, 1], requirement: 3, adapter: "clampix".into() },
            Failure::Substituted { path: vec![1, 0, 1, 0], by: Substitution::Constant, requirement: 1 },
        ],
        "{d:#?}"
    );
    let hist = d.node(&[1, 0, 1, 0]).expect("hist");
    assert_eq!((hist.environment.trips, &hist.properties), (2, &props(&[("len", "1")])), "the constant's length, not the indices' n");
    // The same tables with `subst` would have rendered the indices (in1, of length n): the substitute is the table's.
    let mut own = tables();
    own.requirements[1].adapter = "subst".into();
    let k = contract();
    let dec = Decoder::new(&own, &k).expect("decoder");
    let tree = f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![f(TAB, vec![v(VAR_SIZE, W_N), f(GET, vec![f(HIST, vec![v(VAR_SIZE, W_K), v(VAR_ARR, W_IS), v(VAR_ARR, W_VS)]), v(VAR_I64, 1)])])])]);
    assert_eq!(dec.decode(&entry(), &tree).expect("decodes").node(&[1, 0, 1, 0]).expect("hist").substituted.map(|s| s.by), Some(Substitution::Input(1)));
    // At the root the bound holds and the hist renders itself.
    assert!(decode(&reference(W_N)).root.substituted.is_none());
}

/// G3 review finding 4: a subtree's key carries its free variables' properties and its trips, so `get(is, i)` under
/// `tabulate n` and under `tabulate k` (one renders `is[i]`, the other `is[clampix n i]`) have different keys, though
/// their resolved subtrees are equal; a variable bound inside the subtree is keyed by its depth below the subtree's
/// root, so the same subtree under one more binding has the same key.
#[test]
fn the_subtree_key_carries_the_free_variables_properties() {
    let get_i = f(GET, vec![v(VAR_ARR, W_IS), v(VAR_I64, 0)]);
    let (under_n, under_k) = (decode(&f(TAB, vec![v(VAR_SIZE, W_N), get_i.clone()])), decode(&f(TAB, vec![v(VAR_SIZE, W_K), get_i.clone()])));
    assert_eq!(under_n.resolved().kids[1], under_k.resolved().kids[1], "the resolved subtrees are equal (finding 4's fault)");
    assert_eq!(under_n.failures(), vec![Failure::Root { requirement: 0, adapter: "fit.root".into() }], "its length is n, the entry's k");
    assert_eq!(under_k.failures(), vec![Failure::Adapter { path: vec![1], requirement: 3, adapter: "clampix".into() }]);
    let (a, b) = (under_n.subtree_key(&[1]).expect("key"), under_k.subtree_key(&[1]).expect("key"));
    assert_eq!(a.tree, b.tree);
    assert_ne!(a, b, "the keys differ by i's bound");
    assert_eq!(a.tree.kids.iter().map(|k| k.choice).collect::<Vec<_>>(), vec![Some(Choice::Free(0)), Some(Choice::Free(1))]);
    let free = |key: &SubtreeKey| key.free.iter().map(|f| (f.binder, f.data_type, f.properties.clone())).collect::<Vec<_>>();
    assert_eq!(free(&a), vec![(binder(0, 0), ARR, props(&[("len", "n")])), (binder(1, 0), I64, props(&[("bound", "n")]))]);
    assert_eq!(free(&b)[1], (binder(1, 0), I64, props(&[("bound", "k")])));
    assert_eq!((a.trips, b.trips), (1, 1));
    // The whole tree's key is its resolved tree with nothing free but the entry's parameters.
    let whole = under_n.subtree_key(&[]).expect("key");
    assert_eq!(whole.tree.kids[1].kids[1].choice, Some(Choice::Variable(binder(1, 0))));
    assert_eq!(whole.free.iter().map(|f| f.binder).collect::<Vec<_>>(), vec![binder(0, 5), binder(0, 0)]);
    // tab n (get is i) under tab k (sum .) and under tab k (sum (let zeros_n in .)): its own i is at depth 2 and 3, its
    // key the same (the same trips, the same free `is`, its i one binding below its root).
    let inner = f(TAB, vec![v(VAR_SIZE, W_N), f(GET, vec![v(VAR_ARR, W_IS), v(VAR_I64, 1)])]);
    let plain = decode(&f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![inner.clone()])]));
    let with_let = decode(&f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![f(LET, vec![v(VAR_ARR, 3), inner.clone()])])]));
    assert_eq!(with_let.failures(), plain.failures());
    let (p_node, l_node) = (plain.node(&[1, 0, 1, 1]).expect("plain i"), with_let.node(&[1, 0, 1, 1, 1]).expect("let i"));
    assert_eq!((p_node.choice, l_node.choice), (Some(Choice::Variable(binder(2, 0))), Some(Choice::Variable(binder(3, 0)))));
    assert_eq!(plain.subtree_key(&[1, 0]), with_let.subtree_key(&[1, 0, 1]));
    // A binding ABOVE a free variable's binder moves the binder, so the key differs: that costs sharing only.
    let lifted = decode(&f(LET, vec![v(VAR_ARR, 3), f(TAB, vec![v(VAR_SIZE, W_N), get_i.clone()])]));
    assert_ne!(lifted.subtree_key(&[1, 1]), under_n.subtree_key(&[1]));
    assert_eq!(under_n.subtree_key(&[7]), None);
}

/// G3 review finding 5: the entry is checked on every decode, before any tree is read.
#[test]
fn the_entry_is_checked_on_every_decode() {
    let (t, k) = (tables(), contract());
    let d = Decoder::new(&t, &k).expect("decoder");
    d.check_entry(&entry()).expect("the histogram entry");
    let mut typo = entry();
    typo.requirements[0].property = "lenn".into();
    let e = d.decode(&typo, &reference(W_N)).expect_err("out.lenn");
    assert!(e.contains("the entry's requirement out.lenn = k names out.lenn, which Futhark(3) does not carry"), "{e}");
    let mut twice = entry();
    twice.variables.push(twice.variables[0].clone());
    assert!(d.check_entry(&twice).expect_err("twice").iter().any(|m| m.contains("the entry has two parameters named is")));
    let mut wrong = entry();
    wrong.variables[0].properties = vec![("bound".into(), p("n"))];
    assert!(d.check_entry(&wrong).expect_err("bound").iter().any(|m| m.contains("the entry's parameter is sets bound, which Futhark(3) does not carry")));
}

/// A loaded master 6 view: its kingdom, tables, contract and entry.
fn view6(problem: &str) -> (crate::futhark_table::Kingdom6, KingdomTables, KingdomContract, Entry) {
    let k = crate::futhark_view::master6::committed(problem).and_then(|v| v.kingdom(problem)).unwrap_or_else(|e| panic!("{problem}: {e}"));
    let tables = k.tables().unwrap_or_else(|e| panic!("{problem}: {e}"));
    let contract = k.contract().unwrap_or_else(|e| panic!("{problem}: {e}"));
    let entry = Entry::of_kingdom6(&k);
    (k, tables, contract, entry)
}

/// G3 review finding 7: design section 6's reference and mutation on the REAL histogram view (loaded through
/// `Kingdom6::from_tables`), not the tables written here: the reference holds every requirement, the mutation fails
/// exactly `hist`'s `fit.pair` and `get`'s `clampix`, and a `hist` at trips 2 renders its constant.
#[test]
fn the_histogram_view_decodes_the_reference_and_the_mutation() {
    let (k, t, c, e) = view6("histogram");
    let d = Decoder::new(&t, &c).unwrap_or_else(|e| panic!("{e:#?}"));
    let id = |name: &str| k.symbols.iter().position(|s| s.name == name).unwrap_or_else(|| panic!("{name}")) as SymbolId;
    let (hist, tab, get, md, sum) = (id("hist.add@a.i32"), id("tab@a.i32"), id("get@i32"), id("mod@i32"), id("sum@i32"));
    let (vsize, varr, vi64) = (id("var@size"), id("var@a.i32"), id("var@i64"));
    let names: Vec<&str> = e.variables.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, ["is", "vs", "zeros_k", "zeros_n", "k", "n"], "the words below are the hand tables'");
    let reference = |size: u32| {
        let body = f(md, vec![v(vsize, W_K), f(get, vec![v(varr, W_IS), v(vi64, 0)])]);
        f(hist, vec![v(vsize, W_K), f(tab, vec![v(vsize, size), body]), v(varr, W_VS)])
    };
    let req = |symbol: SymbolId, adapter: &str| t.requirements.iter().position(|q| q.symbol == symbol as i64 && q.adapter == adapter).unwrap_or_else(|| panic!("{adapter}"));
    assert_eq!(d.decode(&e, &reference(W_N)).expect("reference").failures(), vec![]);
    let m = d.decode(&e, &reference(W_K)).expect("mutation");
    assert_eq!(
        m.failures(),
        vec![Failure::Adapter { path: vec![], requirement: req(hist, "fit.pair"), adapter: "fit.pair".into() }, Failure::Adapter { path: vec![1, 1, 1], requirement: req(get, "clampix"), adapter: "clampix".into() }]
    );
    assert!(m.root_outcomes.iter().all(|o| o.holds));
    let deep = f(tab, vec![v(vsize, W_K), f(sum, vec![f(tab, vec![v(vsize, W_N), f(get, vec![f(hist, vec![v(vsize, W_K), v(varr, W_IS), v(varr, W_VS)]), v(vi64, 1)])])])]);
    let s = d.decode(&e, &deep).expect("deep").node(&[1, 0, 1, 0]).expect("hist").substituted.expect("substituted");
    assert_eq!((s.by, s.requirement), (Substitution::Constant, req(hist, "subst.const")));
}

/// G3 review finding 7: random genes over EVERY committed master 6 view, loaded through `Kingdom6::from_tables`: each
/// decodes, the same twice; every failure names an adapter of the tables or a substitution by a work bound of a
/// substitution adapter; and in a view with a work bound, failures and substitutions both occur.
#[test]
fn random_genes_over_the_real_master6_views_fail_only_by_named_adapters() {
    for (problem, ..) in crate::futhark_view::master6::COMMITTED6 {
        let (k, t, c, e) = view6(problem);
        let d = Decoder::new(&t, &c).unwrap_or_else(|e| panic!("{problem}: {e:#?}"));
        let layout = GeneLayout::from_program_size(&c, e.output, 24).unwrap_or_else(|e| panic!("{problem}: {e:?}"));
        let consts: BTreeMap<Ty, u32> = k.data_types.iter().enumerate().filter(|(_, dt)| !dt.literals.is_empty()).map(|(i, dt)| (Ty::Futhark(i as u16), dt.literals.len() as u32)).collect();
        let bounded = t.requirements.iter().any(|q| q.subject == Subject::Trips);
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let (mut with_failures, mut substituted) = (0, 0);
        for g in 0..200 {
            let gene = random_gene(&c, &layout, &consts, &mut rng);
            let tree = express(&c, &layout, &gene, &consts).unwrap_or_else(|err| panic!("{problem} gene {g}: {err:?}")).tree();
            let a = d.decode(&e, &tree).unwrap_or_else(|err| panic!("{problem} gene {g}: {err}"));
            assert_eq!(a, d.decode(&e, &tree).expect("again"), "{problem} gene {g}");
            let fs = a.failures();
            with_failures += usize::from(!fs.is_empty());
            substituted += usize::from(fs.iter().any(|f| matches!(f, Failure::Substituted { .. })));
            for fl in &fs {
                match fl {
                    Failure::Adapter { adapter, requirement, .. } => {
                        assert!(t.adapters.iter().any(|x| &x.name == adapter) && Substitute::of_adapter(adapter).is_none(), "{problem} gene {g}: {adapter}");
                        assert_eq!(&t.requirements[*requirement].adapter, adapter);
                    }
                    Failure::Root { adapter, .. } => assert_eq!(t.adapters.iter().find(|x| &x.name == adapter).map(|x| x.runs_at), Some(AdapterSite::Root), "{problem} gene {g}: {adapter}"),
                    Failure::Substituted { path, by, requirement } => {
                        let q = &t.requirements[*requirement];
                        let node = a.node(path).expect("node");
                        let work = t.requirements.iter().find(|w| w.symbol == q.symbol && w.subject == Subject::Trips).expect("a work bound");
                        let how = Substitute::of_adapter(&work.adapter).expect("a substitution adapter");
                        let own = c.signature(node.symbol).iter().position(|ty| *ty == node.data_type);
                        let want = match (how, own) {
                            (Substitute::OwnInput, Some(p)) => Substitution::Input(p as u32),
                            _ => Substitution::Constant,
                        };
                        assert_eq!((q.symbol, *by), (node.symbol as i64, want), "{problem} gene {g}");
                    }
                }
            }
        }
        eprintln!("{problem}: 200 random genes, {with_failures} with a failing requirement, {substituted} with a substitution");
        assert!(with_failures > 0, "{problem}: no gene fails a requirement");
        assert_eq!(substituted > 0, bounded, "{problem}: substitutions occur exactly in a view with a work bound");
    }
}

/// A decoder built from `Arc`s owns its tables and contract: the caller's handles dropped, it still decodes, and to
/// the same decoding as one built by `Decoder::new`.
#[test]
fn a_decoder_from_arcs_outlives_the_callers_handles() {
    let (t, k) = (std::sync::Arc::new(tables()), std::sync::Arc::new(contract()));
    let d = Decoder::from_arcs(std::sync::Arc::clone(&t), std::sync::Arc::clone(&k)).unwrap_or_else(|e| panic!("{e:#?}"));
    drop((t, k));
    let tree = reference(W_N);
    assert_eq!(d.decode(&entry(), &tree).unwrap_or_else(|e| panic!("{e}")), decode(&tree));
}
