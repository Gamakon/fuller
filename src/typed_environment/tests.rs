//! Tests of the decoder's passes 2 to 5 and the cost's entry points (design section 4), on a histogram kingdom built
//! here in the four kinds of row (Argument, Flow, Guard, Entry; design 3.3), as master 6 spells its rows at depth 2,
//! and on small kingdoms of other shapes (TSR's stream order, WGSL's uniformity, SQL's columns, measured properties).

use std::collections::BTreeMap;

use super::*;
use crate::geneframe::{Argument, Constant, DataKind, DataType, Direction, EntryParameter, Flow, Guard, Owner, Relation, Repair, Source, SymbolRow};
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

fn c(text: &str) -> Condition {
    Condition::parse(text).expect(text)
}

/// The repairs `subst.own` (design 8.0): the node's first input of its own data type, else its constant.
fn subst_own() -> Vec<Repair> {
    vec![Repair::new(Site::Node, "{in.own}", ""), Repair::new(Site::Node, "{const}", "")]
}

/// A heavy adapter's repair at `site`, guarded by the work bound's depth, then `subst.own`.
fn heavy(site: Site, adapter: &str) -> Vec<Repair> {
    let template = match site {
        Site::In(k) => format!("{{{k}}}"),
        _ => "{self}".to_string(),
    };
    let mut r = Repair::new(site, &template, adapter);
    r.guard = c("env.trips < 2");
    let mut v = vec![r];
    v.extend(subst_own());
    v
}

fn must(symbol: SymbolId, order: u32, condition: &str, repairs: Vec<Repair>) -> Guard {
    Guard { owner: Owner::Symbol(symbol as i64), order, condition: c(condition), strength: Strength::Must, repairs }
}

/// The histogram view's tables at depth 2, as master 6 writes `hist.add`, `tab`, `get`, `mod`, `map2.add`, `sum`,
/// `zero`, `last`, `loop` and `let` (a binding with no trip), in the four kinds of row.
fn tables() -> KingdomTables {
    let mut t = KingdomTables { kingdom: "histogram G3".to_string(), ..KingdomTables::default() };
    for (i, (name, ins, out)) in rows().into_iter().enumerate() {
        let symbol = i as i64;
        let role = match name.as_str() {
            "const.i32" => Role::Constant,
            n if n.starts_with("var.") => Role::Variable,
            _ => Role::Function,
        };
        t.symbols.push(SymbolRow { role, ..SymbolRow::named(symbol, &name) });
        for (pos, ty) in ins.iter().enumerate() {
            t.arguments.push(Argument { symbol, position: pos as u32, direction: Direction::In, data_type: *ty, head: false });
        }
        t.arguments.push(Argument { symbol, position: 0, direction: Direction::Out, data_type: out, head: false });
    }
    let constant = |spelling: &str, props: &[(&str, &str)]| Some(Constant { spelling: spelling.into(), properties: props.iter().map(|(k, v)| (k.to_string(), p(v))).collect() });
    t.data_types = vec![
        DataType { constant: constant("0", &[]), ..DataType::scalar(I32, "i32") },
        DataType { properties: vec!["bound".into()], constant: constant("0i64", &[("bound", "1")]), ..DataType::scalar(I64, "i64") },
        DataType { properties: vec!["value".into()], size_names: vec![("d".into(), 1), ("k".into(), 1), ("n".into(), 1)], ..DataType::scalar(SIZE, "size") },
        DataType { kind: DataKind::Array { element: I32 }, properties: vec!["len".into()], constant: constant("[0]", &[("len", "1")]), ..DataType::scalar(ARR, "[]i32") },
    ];
    let up = |symbol: SymbolId, property: &str, expr: &str| Flow::up(symbol as i64, property, p(expr));
    let var = |symbol: SymbolId, k: u32, ty: Ty, group: u32| Flow::down(symbol as i64, k, FlowName::Var { ty, group }, p("1"));
    let prop = |symbol: SymbolId, k: u32, ty: Ty, group: u32, property: &str, expr: &str| Flow::down(symbol as i64, k, FlowName::VarProperty { ty, group, property: property.into() }, p(expr));
    let trip = |symbol: SymbolId, k: u32| Flow::down(symbol as i64, k, FlowName::Env("trips".into()), p("env.trips + 1"));
    t.flows = vec![
        up(HIST, "len", "in0.value"),
        up(TAB, "len", "in0.value"),
        up(MAP2, "len", "min(in0.len, in1.len)"),
        up(ZERO, "bound", "1"),
        up(LAST, "bound", "in0.value"),
        up(LOOP, "len", "in1.len"),
        up(LET, "len", "in1.len"),
        var(TAB, 1, I64, 0),
        prop(TAB, 1, I64, 0, "bound", "in0.value"),
        trip(TAB, 1),
        var(LOOP, 2, ARR, 0),
        prop(LOOP, 2, ARR, 0, "len", "in1.len"),
        var(LOOP, 2, I64, 0),
        prop(LOOP, 2, I64, 0, "bound", "in0.value"),
        trip(LOOP, 2),
        var(LET, 1, ARR, 0),
        prop(LET, 1, ARR, 0, "len", "in0.len"),
    ];
    // In master 6's row order, so the guards' indices are the requirements' (0 to 8).
    t.guards = vec![
        must(HIST, 1, "in2.len = in1.len", heavy(Site::Node, "fit.pair").into_iter().take(1).chain([Repair::new(Site::Node, "{const}", "")]).collect()),
        must(HIST, 0, "env.trips < 2", vec![Repair::new(Site::Node, "{const}", "")]),
        must(TAB, 0, "env.trips < 2", subst_own()),
        must(GET, 0, "in1.bound <= in0.len", vec![Repair::new(Site::In(1), "clampix ({=in0.len}) ({1})", "clampix")]),
        must(MAP2, 1, "in1.len = in0.len", heavy(Site::Node, "fit.pair")),
        must(MAP2, 0, "env.trips < 2", subst_own()),
        must(SUM, 0, "env.trips < 2", subst_own()),
        must(LOOP, 1, "in2.len = in1.len", heavy(Site::In(2), "fit.state")),
        must(LOOP, 0, "env.trips < 2", subst_own()),
    ];
    t
}

/// The histogram entry `main [n] (k: i32) (is: [n]i32) (vs: [n]i32) : [k]i32`, its root environment as the view
/// writes it: the inputs, the zero arrays, the sizes; its guard `out.len = k` repaired at the root by `fit.root`.
fn entry() -> Entry {
    let mut e = Entry::new("histogram", ARR);
    for (name, len) in [("is", "n"), ("vs", "n"), ("zeros_k", "k"), ("zeros_n", "n")] {
        e.parameters.push(EntryParameter::declared(name, ARR, &[("len", p(len))]));
    }
    for name in ["k", "n"] {
        e.parameters.push(EntryParameter::declared(name, SIZE, &[("value", p(name))]));
    }
    e.guards.push(Guard { owner: Owner::Entry, order: 0, condition: c("out.len = k"), strength: Strength::Must, repairs: vec![Repair::new(Site::Root, "{self}", "fit.root")] });
    e
}

fn f(symbol: SymbolId, kids: Vec<Tree>) -> Tree {
    Tree { symbol, const_index: None, kids }
}

fn v(symbol: SymbolId, word: u32) -> Tree {
    Tree { symbol, const_index: Some(word), kids: vec![] }
}

// The root environment's words, counted from the innermost (design 3.6; the root parameters in the entry's order are
// outermost, the last one innermost): arrays zeros_n 0, zeros_k 1, vs 2, is 3; sizes n 0, k 1.
const W_IS: u32 = 3;
const W_VS: u32 = 2;
const W_ZEROS_K: u32 = 1;
const W_K: u32 = 1;
const W_N: u32 = 0;

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
    assert_eq!((hist_pair.subject(), hist_pair.against(), hist_pair.holds), (&p("n"), &p("n"), true));
    // tabulate n: len = n, binds i of bound n, one trip.
    assert_eq!(node(&[1]).properties, props(&[("len", "n")]));
    let body_env = &node(&[1, 1]).environment;
    assert_eq!((body_env.depth, body_env.attribute("trips")), (1, Some(&p("1"))));
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
    assert_eq!((get.subject(), get.against(), get.holds), (&p("n"), &p("n"), true));
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
    assert_eq!(t.guards[0].owner, Owner::Symbol(HIST as i64));
    assert_eq!(t.guards[3].owner, Owner::Symbol(GET as i64));
    let get = &d.node(&[1, 1, 1]).expect("get").outcomes[0];
    assert_eq!((get.subject(), get.against()), (&p("k"), &p("n")), "i's bound is k, is's length n");
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
    let sizes = ["d", "k", "n"];
    let mut entry = Entry::new("literal indices", ARR);
    for s in sizes {
        entry.parameters.push(EntryParameter::declared(&format!("a_{s}"), ARR, &[("len", p(s))]));
    }
    for s in sizes {
        entry.parameters.push(EntryParameter::declared(s, SIZE, &[("value", p(s))]));
    }
    // Counted from the innermost: the w-th of d, k, n (outermost first) is word 2 - w.
    let word = |w: usize| (2 - w) as u32;
    // An array of each length: a root input, a tabulate and a histogram of each size (the histogram's two operands
    // the same array, so its own requirement holds).
    let mut arrays: Vec<(Tree, PropertyExpr)> = Vec::new();
    for (w, s) in sizes.iter().enumerate() {
        let w = word(w);
        arrays.push((v(VAR_ARR, w), p(s)));
        arrays.push((f(TAB, vec![v(VAR_SIZE, w), v(10, 0)]), p(s)));
        arrays.push((f(HIST, vec![v(VAR_SIZE, w), v(VAR_ARR, 0), v(VAR_ARR, 0)]), p(s)));
    }
    for (array, len) in &arrays {
        // tab(n, get(array, index)) is legal at the root.
        let at = |index: Tree| d.decode(&entry, &f(TAB, vec![v(VAR_SIZE, 0), f(GET, vec![array.clone(), index])])).expect("decodes");
        let zero = at(f(ZERO, vec![]));
        assert_eq!(zero.failures(), vec![], "0i64 into an array of length {len}");
        assert_eq!(zero.node(&[1, 1]).expect("zero").properties, props(&[("bound", "1")]));
        for (w, s) in sizes.iter().enumerate() {
            let last = at(f(LAST, vec![v(VAR_SIZE, word(w))]));
            assert_eq!(last.node(&[1, 1]).expect("last").properties, props(&[("bound", s)]));
            assert_eq!(last.failures().is_empty(), p(s) == *len, "{s} - 1 into an array of length {len}: {:#?}", last.failures());
        }
    }
    // A literal index of bound 1 into a length that is a min of sizes, or the constant array's length 1, also holds.
    for len in ["min(k, n)", "min(d, k, n)", "1"] {
        assert_eq!(Relation::Le.truth(&p("1"), &p(len), &tables().bounds()), Truth::Holds, "{len}");
    }
}

/// Plan 7.5 risk 3: the resolved tree names variables by binder, so equal words under different environments
/// resolve to different trees, and different words naming the same variable resolve equal.
#[test]
fn the_resolved_tree_names_variables_by_binder() {
    let (t, k) = (tables(), contract());
    let d = Decoder::new(&t, &k).expect("decoder");
    let e = entry();
    let get_i = f(GET, vec![v(VAR_ARR, W_IS), v(VAR_I64, 0)]);
    // Word 0 is the innermost i64: under one tabulate the outer i, under two the inner i.
    let one = f(TAB, vec![v(VAR_SIZE, W_N), get_i.clone()]);
    let two = f(TAB, vec![v(VAR_SIZE, W_N), f(SUM, vec![f(TAB, vec![v(VAR_SIZE, W_K), get_i.clone()])])]);
    let (r1, r2) = (d.resolve(&e, &one).expect("one"), d.resolve(&e, &two).expect("two"));
    let (g1, g2) = (&r1.kids[1], &r2.kids[1].kids[0].kids[1]);
    assert_eq!((&one.kids[1], &two.kids[1].kids[0].kids[1]), (&get_i, &get_i), "the same words");
    assert_ne!(g1, g2, "different variables");
    assert_eq!(g1.kids[1].choice, Some(Choice::Variable(binder(1, 0))));
    assert_eq!(g2.kids[1].choice, Some(Choice::Variable(binder(2, 0))));
    // Words 3 and 7 of the four arrays in scope both name `is`; 2 names `vs`.
    let with = |w: u32| f(TAB, vec![v(VAR_SIZE, W_N), f(GET, vec![v(VAR_ARR, w), f(ZERO, vec![])])]);
    assert_ne!(with(3), with(7));
    assert_eq!(d.resolve(&e, &with(3)).expect("3"), d.resolve(&e, &with(7)).expect("7"));
    assert_ne!(d.resolve(&e, &with(3)).expect("3"), d.resolve(&e, &with(2)).expect("2"));
    // The inverse: the least word naming a variable, in the one order (the innermost first).
    let dec = d.decode(&e, &two).expect("two");
    let env = &dec.node(&[1, 0, 1]).expect("inner body").environment;
    assert_eq!(env.index_of(I64, binder(2, 0)), Some(0));
    assert_eq!(env.index_of(I64, binder(1, 0)), Some(1));
    assert_eq!(env.choose(I64, 7).map(|v| v.binder), Some(binder(1, 0)));
    assert_eq!(env.index_of(ARR, binder(0, 3)), Some(0));
    assert_eq!(env.index_of(SIZE, binder(0, 5)), Some(0));
}

/// Master 6 README 5.3: the work bound `env.trips < 2`, its substitution by the first input of the node's own data
/// type else by the type's constant, and a heavy body adapter that would run at trips 2 substituting its node.
#[test]
fn the_work_bound_substitutes_as_an_annotation() {
    // The outer tabulate is of k, so the result meets the entry's `len = k`.
    let inner = |body: Tree| f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![f(TAB, vec![v(VAR_SIZE, W_N), body])])]);
    // map2 at trips 2 renders as its input 0 (`is`), so get sees is's length.
    let map2 = decode(&inner(f(GET, vec![f(MAP2, vec![v(VAR_ARR, W_IS), v(VAR_ARR, W_VS)]), v(VAR_I64, 0)])));
    assert_eq!(map2.failures(), vec![Failure::Substituted { path: vec![1, 0, 1, 0], by: Substitution::Input(0), requirement: 5 }], "{map2:#?}");
    let m = map2.node(&[1, 0, 1, 0]).expect("map2");
    assert_eq!((m.environment.attribute("trips"), &m.properties), (Some(&p("2")), &props(&[("len", "n")])));
    // sum at trips 2 has no input of its own type: its constant.
    let sum = decode(&inner(f(SUM, vec![v(VAR_ARR, W_IS)])));
    assert_eq!(sum.failures(), vec![Failure::Substituted { path: vec![1, 0, 1], by: Substitution::Constant, requirement: 6 }]);
    // A loop at trips 1 whose body (at trips 2) is of another length: fit.state would run at trips 2, so the loop is
    // substituted by its state (`zeros_k`, length k) instead.
    let mismatched = f(LOOP, vec![v(VAR_SIZE, W_K), v(VAR_ARR, W_ZEROS_K), v(VAR_ARR, W_IS + 1)]);
    let deep = decode(&f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![mismatched.clone()])]));
    assert_eq!(deep.failures(), vec![Failure::Substituted { path: vec![1, 0], by: Substitution::Input(1), requirement: 7 }], "{deep:#?}");
    assert_eq!(deep.node(&[1, 0]).expect("loop").properties, props(&[("len", "k")]));
    let body_env = &deep.node(&[1, 0, 2]).expect("body").environment;
    assert_eq!((body_env.attribute("trips"), body_env.depth), (Some(&p("2")), 2));
    assert_eq!(in_scope(body_env, ARR).first().map(|v| (v.binder, v.properties.clone())), Some((binder(2, 0), props(&[("len", "k")]))), "the loop's state is the innermost array");
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
    let mut bare = Entry::new("bare", ARR);
    bare.parameters.push(EntryParameter::declared("n", SIZE, &[("value", p("n"))]));
    let x = d.decode(&bare, &f(MAP2, vec![v(VAR_ARR, 5), f(TAB, vec![v(VAR_SIZE, 9), v(10, 3)])])).expect("decodes");
    assert_eq!(x.node(&[0]).expect("var").choice, Some(Choice::Constant));
    assert_eq!(x.node(&[0]).expect("var").properties, props(&[("len", "1")]));
    assert_eq!(x.root.properties, props(&[("len", "min(1, n)")]));
    assert_eq!(x.failures(), vec![Failure::Adapter { path: vec![], requirement: 4, adapter: "fit.pair".into() }]);
    // An entry with no size, where `size` has no constant, is a bug of the entry, found before any tree is read.
    let none = Entry::new("none", ARR);
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
                assert!(["fit.pair", "fit.state", "clampix", "fit.root"].contains(&adapter.as_str()), "gene {g}: {adapter}");
            }
        }
    }
    eprintln!("300 random histogram genes: {with_failures} with a failing requirement, {substituted} with a substitution");
    assert!(with_failures > 0 && substituted > 0, "the genes exercise failures ({with_failures}) and substitutions ({substituted})");
}

const CHILD: &str = "FULLER_G3_DECODE_CHILD";

/// The annotations of a fixed set of trees (the reference, the mutation, the work-bound cases), as text.
fn annotations_text() -> String {
    let mismatched = f(LOOP, vec![v(VAR_SIZE, W_K), v(VAR_ARR, W_ZEROS_K), v(VAR_ARR, W_IS + 1)]);
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
                (SymbolRow { role: k.role(s), ..SymbolRow::named(s as i64, &name) }, k.signature(s).to_vec(), k.output(s))
            })
            .collect();
        let t = KingdomTables::from_signatures(&relations.kingdom, rows);
        assert!(!t.has_properties());
        let d = Decoder::new(&t, &k).unwrap_or_else(|e| panic!("{}: {e:#?}", relations.kingdom));
        let root = k.roots()[0];
        let layout = GeneLayout::from_program_size(&k, root, 16).expect("layout");
        let consts: BTreeMap<Ty, u32> = k.types().iter().filter(|t| k.constant_capable(**t)).map(|t| (*t, 5)).collect();
        let entry = Entry::new("no properties", root);
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
    t.guards.retain(|g| g.owner != Owner::Symbol(GET as i64));
    t.check().expect("the edited tables check on their own");
    let e = Decoder::new(&t, &k).expect_err("a mismatch");
    assert!(e.iter().any(|m| m.contains("contract symbol 2 (get)")), "{e:#?}");
    let mut t = tables();
    t.flows.push(Flow::up(VAR_ARR as i64, "len", p("1")));
    let e = Decoder::new(&t, &k).expect_err("a flow on a variable");
    assert!(e.iter().any(|m| m.contains("the variable terminal var.a.i32 has flows")), "{e:#?}");
    let mut t = tables();
    t.symbols[GET as usize].role = Role::Constant;
    t.arguments.retain(|a| !(a.symbol == GET as i64 && a.direction == Direction::In));
    t.guards.retain(|g| g.owner != Owner::Symbol(GET as i64));
    assert!(Decoder::new(&t, &k).expect_err("a role mismatch").iter().any(|m| m.contains("contract symbol 2 (get)")));
}

/// G3 review finding 1 (the lead's decision): a `hist` the work bound substitutes renders its CONSTANT (`subst.const`),
/// the histogram with nothing counted, never its indices or its values, though both are of its own data type.
#[test]
fn a_substituted_histogram_renders_its_constant() {
    // tab k (sum (tab n (get (hist k is vs) i))): the hist is at trips 2.
    let body = f(GET, vec![f(HIST, vec![v(VAR_SIZE, W_K), v(VAR_ARR, W_IS), v(VAR_ARR, W_VS)]), v(VAR_I64, 0)]);
    let d = decode(&f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![f(TAB, vec![v(VAR_SIZE, W_N), body])])]));
    let t = tables();
    assert_eq!((t.guards[1].owner, t.guards[1].repairs.iter().map(|r| r.template.as_str()).collect::<Vec<_>>()), (Owner::Symbol(HIST as i64), vec!["{const}"]));
    assert_eq!(
        d.failures(),
        vec![
            Failure::Adapter { path: vec![1, 0, 1], requirement: 3, adapter: "clampix".into() },
            Failure::Substituted { path: vec![1, 0, 1, 0], by: Substitution::Constant, requirement: 1 },
        ],
        "{d:#?}"
    );
    let hist = d.node(&[1, 0, 1, 0]).expect("hist");
    assert_eq!((hist.environment.attribute("trips"), &hist.properties), (Some(&p("2")), &props(&[("len", "1")])), "the constant's length, not the indices' n");
    // The same tables with `subst.own` would have rendered the indices (in1, of length n): the substitute is the
    // table's.
    let mut own = tables();
    own.guards[1].repairs = subst_own();
    let k = contract();
    let dec = Decoder::new(&own, &k).expect("decoder");
    let tree = f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![f(TAB, vec![v(VAR_SIZE, W_N), f(GET, vec![f(HIST, vec![v(VAR_SIZE, W_K), v(VAR_ARR, W_IS), v(VAR_ARR, W_VS)]), v(VAR_I64, 0)])])])]);
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
    let trips = |key: &SubtreeKey| key.attributes.get("trips").cloned();
    assert_eq!(free(&a), vec![(binder(0, 0), ARR, props(&[("len", "n")])), (binder(1, 0), I64, props(&[("bound", "n")]))]);
    assert_eq!(free(&b)[1], (binder(1, 0), I64, props(&[("bound", "k")])));
    assert_eq!((trips(&a), trips(&b)), (Some(p("1")), Some(p("1"))));
    // The whole tree's key is its resolved tree with nothing free but the entry's parameters.
    let whole = under_n.subtree_key(&[]).expect("key");
    assert_eq!(whole.tree.kids[1].kids[1].choice, Some(Choice::Variable(binder(1, 0))));
    assert_eq!(whole.free.iter().map(|f| f.binder).collect::<Vec<_>>(), vec![binder(0, 5), binder(0, 0)]);
    // tab n (get is i) under tab k (sum .) and under tab k (sum (let zeros_n in .)): its own i is at depth 2 and 3, its
    // key the same (the same trips, the same free `is`, its i one binding below its root).
    // Under the let, `is` is one array further from the innermost: its word is one more (an outer reference's word
    // drifts by the binders of its type between, design 3.6); the subtree's own i keeps word 0.
    let inner = |w_is: u32| f(TAB, vec![v(VAR_SIZE, W_N), f(GET, vec![v(VAR_ARR, w_is), v(VAR_I64, 0)])]);
    let plain = decode(&f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![inner(W_IS)])]));
    let with_let = decode(&f(TAB, vec![v(VAR_SIZE, W_K), f(SUM, vec![f(LET, vec![v(VAR_ARR, 0), inner(W_IS + 1)])])]));
    assert_eq!(with_let.failures(), plain.failures());
    let (p_node, l_node) = (plain.node(&[1, 0, 1, 1]).expect("plain i"), with_let.node(&[1, 0, 1, 1, 1]).expect("let i"));
    assert_eq!((p_node.choice, l_node.choice), (Some(Choice::Variable(binder(2, 0))), Some(Choice::Variable(binder(3, 0)))));
    assert_eq!(plain.subtree_key(&[1, 0]), with_let.subtree_key(&[1, 0, 1]));
    // A binding ABOVE a free variable's binder moves the binder, so the key differs: that costs sharing only.
    let lifted = decode(&f(LET, vec![v(VAR_ARR, 0), f(TAB, vec![v(VAR_SIZE, W_N), f(GET, vec![v(VAR_ARR, W_IS + 1), v(VAR_I64, 0)])])]));
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
    typo.guards[0].condition = c("out.lenn = k");
    let e = d.decode(&typo, &reference(W_N)).expect_err("out.lenn");
    assert!(e.contains("the entry's guard 0 (out.lenn = k) reads out.lenn, which Futhark(3) does not carry"), "{e}");
    let mut twice = entry();
    twice.parameters.push(twice.parameters[0].clone());
    assert!(d.check_entry(&twice).expect_err("twice").iter().any(|m| m.contains("the entry has two parameters named is")));
    let mut sized = entry();
    sized.parameters.push(EntryParameter::declared("zeros_m", ARR, &[("len", p("m"))]));
    assert!(d.check_entry(&sized).expect_err("m").iter().any(|m| m.contains("reads the size name m, which no data type's size_names declares")));
    let mut wrong = entry();
    wrong.parameters[0].properties = vec![("bound".into(), p("n"), Source::Declared)];
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
    let names: Vec<&str> = e.parameters.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, ["is", "vs", "zeros_k", "zeros_n", "k", "n"], "the words below are the hand tables'");
    let reference = |size: u32| {
        let body = f(md, vec![v(vsize, W_K), f(get, vec![v(varr, W_IS), v(vi64, 0)])]);
        f(hist, vec![v(vsize, W_K), f(tab, vec![v(vsize, size), body]), v(varr, W_VS)])
    };
    let reqs = crate::futhark_cost::requirements6(&k);
    let req = |symbol: SymbolId, adapter: &str| reqs.iter().position(|q| q.symbol == symbol as i64 && q.adapter == adapter).unwrap_or_else(|| panic!("{adapter}"));
    assert_eq!(t.guards.len(), reqs.len(), "one guard a requirement, in its order");
    assert_eq!(d.decode(&e, &reference(W_N)).expect("reference").failures(), vec![]);
    let m = d.decode(&e, &reference(W_K)).expect("mutation");
    assert_eq!(
        m.failures(),
        vec![Failure::Adapter { path: vec![], requirement: req(hist, "fit.pair"), adapter: "fit.pair".into() }, Failure::Adapter { path: vec![1, 1, 1], requirement: req(get, "clampix"), adapter: "clampix".into() }]
    );
    assert!(m.root_outcomes.iter().all(|o| o.holds));
    let deep = f(tab, vec![v(vsize, W_K), f(sum, vec![f(tab, vec![v(vsize, W_N), f(get, vec![f(hist, vec![v(vsize, W_K), v(varr, W_IS), v(varr, W_VS)]), v(vi64, 0)])])])]);
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
        let reqs = crate::futhark_cost::requirements6(&k);
        let bounded = reqs.iter().any(|q| q.subject == crate::futhark_table::Subject6::Trips);
        assert_eq!(t.guards.len(), reqs.len());
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
                        assert!(k.adapters.iter().any(|x| &x.name == adapter) && crate::futhark_table::Substitute6::of_adapter(adapter).is_none(), "{problem} gene {g}: {adapter}");
                        assert_eq!(&reqs[*requirement].adapter, adapter);
                    }
                    Failure::Root { adapter, .. } => assert_eq!(k.adapters.iter().find(|x| &x.name == adapter).map(|x| x.runs_at), Some(crate::futhark_table::Site6::Root), "{problem} gene {g}: {adapter}"),
                    Failure::Substituted { path, by, requirement } => {
                        let q = &reqs[*requirement];
                        let node = a.node(path).expect("node");
                        let work = reqs.iter().find(|w| w.symbol == q.symbol && w.subject == crate::futhark_table::Subject6::Trips).expect("a work bound");
                        let how = crate::futhark_table::Substitute6::of_adapter(&work.adapter).expect("a substitution adapter");
                        let own = c.signature(node.symbol).iter().position(|ty| *ty == node.data_type);
                        let want = match (how, own) {
                            (crate::futhark_table::Substitute6::OwnInput, Some(p)) => Substitution::Input(p as u32),
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

// ---- the revised design's rows (external review 2026-11-05): strength, measured values, out.p, attributes,
// ---- variable groups, rendering, explicit adapters

/// The contract of a kingdom's own tables, rooted at `root`.
fn contract_of(t: &KingdomTables, root: Ty) -> KingdomContract {
    crate::kingdom_contract::relations_from_tables(t, vec![root]).check().unwrap_or_else(|d| panic!("{d:#?}"))
}

/// The histogram kingdom with an entry whose index `j` (bound n) is declared or measured, and with `get`'s guard also
/// written as an advise copy and an advise guard that always fails.
fn measured_case(source: Source) -> (Decoder, Entry) {
    let mut t = tables();
    t.guards.push(Guard { owner: Owner::Symbol(GET as i64), order: 1, condition: c("in1.bound <= in0.len"), strength: Strength::Advise, repairs: vec![] });
    t.guards.push(Guard { owner: Owner::Symbol(GET as i64), order: 2, condition: c("in1.bound <= 1"), strength: Strength::Advise, repairs: vec![] });
    let mut e = entry();
    e.parameters.push(EntryParameter { name: "j".into(), data_type: I64, properties: vec![("bound".into(), p("n"), source)], form_of: None });
    (Decoder::new(&t, &contract()).unwrap_or_else(|d| panic!("{d:#?}")), e)
}

/// Strength (design 3.3.3, two engines): a `must` guard that fails is repaired by its adapter; an `advise` guard that
/// fails applies no repair and is reported by the cost; and a MEASURED value never makes a `must` guard hold, though the
/// cost reads it as it is.
#[test]
fn must_repairs_advise_reports_and_a_measured_value_never_satisfies_a_must() {
    // tab k (get is j): is[j], j of bound n, is of length n.
    let tree = f(TAB, vec![v(VAR_SIZE, W_K), f(GET, vec![v(VAR_ARR, W_IS), v(VAR_I64, 1)])]);
    let (d, e) = measured_case(Source::Declared);
    let dec = d.decode(&e, &tree).expect("decodes");
    let get = dec.node(&[1]).expect("get");
    assert_eq!(get.kids[1].choice, Some(Choice::Variable(binder(0, 6))), "word 1 under tab: j, below tab's own i");
    assert!(get.outcomes.iter().all(|o| o.holds), "declared n <= n holds");
    assert_eq!(get.outcomes.len(), 1, "the decoder reads the must guard alone");
    let advice = d.advice(&e, &dec);
    assert_eq!(advice.iter().map(|a| (a.path.clone(), a.guard)).collect::<Vec<_>>(), vec![(Some(vec![1]), 10)], "only the advise guard that fails, reported, not repaired");
    assert!(dec.failures().is_empty(), "an advise failure repairs nothing");
    // The same index measured on the training cases: the must guard reads `?`, does not hold, and its clampix runs;
    // the advise copy reads the measured n and holds.
    let (d, e) = measured_case(Source::Measured("train".into()));
    let dec = d.decode(&e, &tree).expect("decodes");
    let get = dec.node(&[1]).expect("get");
    assert!(get.kids[1].measured.contains("bound"));
    assert_eq!((get.outcomes[0].holds, get.outcomes[0].subject()), (false, &PropertyExpr::Unknown), "a measured bound reads as ? in a must guard");
    assert_eq!(dec.failures(), vec![Failure::Adapter { path: vec![1], requirement: 3, adapter: "clampix".into() }]);
    assert_eq!(d.advice(&e, &dec).iter().map(|a| a.guard).collect::<Vec<_>>(), vec![10], "the cost reads the measured value as it is: the copy holds");
    let rescues = d.rescues(&dec);
    assert_eq!(rescues, vec![Rescue { path: Some(vec![1]), adapter: "clampix".into(), placed: false }]);
}

/// TSR's stream order (design 8.3) as rows: up flows reading `out.arm` and `out.ties` (no cycle), and the guard
/// `out.order <= 3` repaired by `{in0}`.
#[test]
fn tsr_stream_order_reads_out_p_and_replaces_by_in0() {
    let f_ = Ty::F;
    let mut t = KingdomTables { kingdom: "TSR stream order".into(), ..KingdomTables::default() };
    t.data_types = vec![DataType { properties: ["depth", "order", "fam", "arm", "ties"].map(String::from).to_vec(), ..DataType::scalar(f_, "F") }];
    let names = ["add", "mul", "a", "b", "c", "d"];
    for (i, name) in names.iter().enumerate() {
        let id = i as i64;
        t.symbols.push(SymbolRow::named(id, name));
        let n_in = if i < 2 { 2 } else { 0 };
        for k in 0..n_in {
            t.arguments.push(Argument { symbol: id, position: k, direction: Direction::In, data_type: f_, head: false });
        }
        t.arguments.push(Argument { symbol: id, position: 0, direction: Direction::Out, data_type: f_, head: false });
        if i >= 2 {
            for (prop, val) in [("depth", "0"), ("order", "1"), ("fam", "0"), ("arm", "1"), ("ties", "1")] {
                t.flows.push(Flow::up(id, prop, p(val)));
            }
        }
    }
    for (id, fam) in [(0i64, 1), (1, 2)] {
        let b = |k: u32| format!("if({fam} = in{k}.fam, in{k}.arm, in{k}.order)");
        let tk = |k: u32| format!("if({fam} = in{k}.fam, in{k}.ties, 1)");
        t.flows.push(Flow::up(id, "order", p("if(2 <= out.ties, out.arm + 1, out.arm)")));
        t.flows.push(Flow::up(id, "ties", p(&format!("if({} = out.arm, {}, 0) + if({} = out.arm, {}, 0)", b(0), tk(0), b(1), tk(1)))));
        t.flows.push(Flow::up(id, "arm", p(&format!("max({}, {})", b(0), b(1)))));
        t.flows.push(Flow::up(id, "fam", p(&fam.to_string())));
        t.flows.push(Flow::up(id, "depth", p("max(in0.depth, in1.depth)")));
        t.guards.push(Guard { owner: Owner::Symbol(id), order: 0, condition: c("out.order <= 3"), strength: Strength::Must, repairs: vec![Repair::new(Site::Node, "{in0}", "")] });
    }
    t.check().unwrap_or_else(|d| panic!("{d:#?}"));
    let names: Vec<FlowName> = t.up_order(0).expect("order").iter().map(|i| t.flows[*i].name.clone()).collect();
    let at = |q: &str| names.iter().position(|n| *n == FlowName::Property(q.into())).expect(q);
    assert!(at("arm") < at("ties") && at("ties") < at("order"), "order after the ties and arm it reads, ties after arm: {names:?}");
    let k = contract_of(&t, f_);
    let d = Decoder::new(&t, &k).unwrap_or_else(|e| panic!("{e:#?}"));
    let e = Entry::new("tsr", f_);
    let leaf = |i: SymbolId| f(i, vec![]);
    let (add, mul) = (|a: Tree, b: Tree| f(0, vec![a, b]), |a: Tree, b: Tree| f(1, vec![a, b]));
    let order = |tree: &Tree| d.decode(&e, tree).expect("decodes").root.properties["order"].clone();
    assert_eq!(order(&add(leaf(2), leaf(3))), p("2"), "a + b");
    let abc = add(add(leaf(2), leaf(3)), leaf(4));
    let dec = d.decode(&e, &abc).expect("decodes");
    assert_eq!((dec.root.properties["ties"].clone(), dec.root.properties["order"].clone()), (p("3"), p("2")), "(a + b) + c: Strahler order over the flattened chain");
    let big = mul(add(mul(leaf(2), leaf(3)), mul(leaf(4), leaf(5))), add(mul(leaf(2), leaf(4)), mul(leaf(3), leaf(5))));
    let dec = d.decode(&e, &big).expect("decodes");
    assert_eq!(dec.failures(), vec![Failure::Substituted { path: vec![], by: Substitution::Input(0), requirement: 1 }], "order 4 fails out.order <= 3: replaced by its first input");
    assert_eq!(dec.root.properties["order"], p("3"), "it takes its substitute's properties");
}

/// A flag attribute (WGSL's uniform control flow, design 8.4): a down flow sets it false in a branch, and a guard
/// `env.uniform` replaces a barrier there by its input.
#[test]
fn a_flag_attribute_flows_down_and_guards_a_node() {
    let mut t = KingdomTables { kingdom: "uniformity".into(), ..KingdomTables::default() };
    t.data_types = vec![DataType::scalar(Ty::S, "Stmt")];
    for (i, (name, n_in)) in [("barrier", 1u32), ("branch", 1), ("noop", 0)].into_iter().enumerate() {
        t.symbols.push(SymbolRow::named(i as i64, name));
        for k in 0..n_in {
            t.arguments.push(Argument { symbol: i as i64, position: k, direction: Direction::In, data_type: Ty::S, head: false });
        }
        t.arguments.push(Argument { symbol: i as i64, position: 0, direction: Direction::Out, data_type: Ty::S, head: false });
    }
    t.flows.push(Flow::down(1, 0, FlowName::Env("uniform".into()), p("false")));
    t.guards.push(Guard { owner: Owner::Symbol(0), order: 0, condition: c("env.uniform"), strength: Strength::Must, repairs: vec![Repair::new(Site::Node, "{in0}", "")] });
    let k = contract_of(&t, Ty::S);
    let d = Decoder::new(&t, &k).unwrap_or_else(|e| panic!("{e:#?}"));
    let e = Entry::new("uniformity", Ty::S);
    let at_root = d.decode(&e, &f(0, vec![f(2, vec![])])).expect("decodes");
    assert!(at_root.failures().is_empty(), "the root's control flow is uniform (the catalogue's root value)");
    let in_branch = d.decode(&e, &f(1, vec![f(0, vec![f(2, vec![])])])).expect("decodes");
    assert_eq!(in_branch.failures(), vec![Failure::Substituted { path: vec![0], by: Substitution::Input(0), requirement: 0 }]);
    assert_eq!(in_branch.node(&[0]).expect("barrier").environment.attribute("uniform"), Some(&PropertyExpr::Flag(false)));
    // The entry may override the root value.
    let mut off = Entry::new("uniformity", Ty::S);
    off.flows.push(Flow { owner: Owner::Entry, at: crate::geneframe::Place::Out, name: FlowName::Env("uniform".into()), formula: p("false"), template: String::new(), way: crate::geneframe::Way::Down });
    assert_eq!(d.decode(&off, &f(0, vec![f(2, vec![])])).expect("decodes").failures().len(), 1);
}

/// A down flow that introduces several variables (SQL's columns, design 8.7): a count, properties by `each`, names by
/// a spelling template; the innermost is the highest ordinal (design 3.6).
#[test]
fn a_group_of_variables_with_each_and_a_spelling() {
    let mut t = tables();
    t.flows.retain(|fl| fl.owner != Owner::Symbol(LET as i64) || fl.way == crate::geneframe::Way::Up);
    t.flows.push(Flow::down(LET as i64, 1, FlowName::Var { ty: I64, group: 0 }, p("in0.len")));
    t.flows.push(Flow::down(LET as i64, 1, FlowName::VarProperty { ty: I64, group: 0, property: "bound".into() }, p("each + 1")));
    t.flows.push(crate::geneframe::spell(LET as i64, 1, I64, 0, "c{each}_{depth}"));
    let d = Decoder::new(&t, &contract()).unwrap_or_else(|e| panic!("{e:#?}"));
    let mut e = entry();
    e.parameters.push(EntryParameter::declared("three", ARR, &[("len", p("3"))]));
    // let (three) in (tab n (get is var.i64)): three columns c0, c1, c2 in scope.
    let tree = |w: u32| f(LET, vec![v(VAR_ARR, 0), f(TAB, vec![v(VAR_SIZE, W_N), f(GET, vec![v(VAR_ARR, W_IS + 1), v(VAR_I64, w)])])]);
    let dec = d.decode(&e, &tree(0)).expect("decodes");
    let env = &dec.node(&[1, 1]).expect("tab body").environment;
    let cols: Vec<(Option<String>, PropertyExpr)> = in_scope(env, I64).iter().map(|v| (v.name.clone(), v.properties["bound"].clone())).collect();
    assert_eq!(cols[1..], [(Some("c2_1".into()), p("3")), (Some("c1_1".into()), p("2")), (Some("c0_1".into()), p("1"))], "innermost first: tab's i, then c2, c1, c0");
    assert_eq!(dec.node(&[1, 1, 1]).expect("index").choice, Some(Choice::Variable(binder(2, 0))), "word 0: the innermost, tab's own i");
    assert_eq!(d.decode(&e, &tree(1)).expect("decodes").node(&[1, 1, 1]).expect("index").choice, Some(Choice::Variable(binder(1, 2))), "word 1: c2");
}

/// Pass 5 by the template engine: spellings, variable names, a clampix repair at an operand, a fit.pair repair that
/// rewrites both operands in the node's own spelling, and the explicit adapter symbol `clampix` that evolution placed,
/// whose prelude is written once and which the cost charges as placed.
#[test]
fn pass_five_renders_spellings_and_repairs_and_the_cost_charges_every_adapter() {
    let mut t = tables();
    let spell = |t: &mut KingdomTables, s: SymbolId, text: &str| t.symbols[s as usize].spellings = vec![(String::new(), text.to_string())];
    spell(&mut t, HIST, "hist (+) 0 {0} ({1}) ({2})");
    spell(&mut t, TAB, "tabulate {0} (\\i -> {1})");
    spell(&mut t, GET, "{0}[{1}]");
    spell(&mut t, MOD, "{1} % {0}");
    spell(&mut t, SUM, "i32.sum {0}");
    t.flows.push(crate::geneframe::spell(TAB as i64, 1, I64, 0, "i"));
    let clampix = "def clampix (m: i64) (i: i64) : i64 = i64.max 0 (i64.min (m - 1) i)";
    let fit = "def fit [m] 't (l: i64) (z: t) (xs: [m]t) : [l]t = tabulate l (\\i -> if i < m then xs[i] else z)";
    t.symbols.push(SymbolRow { adapter: true, prelude: clampix.into(), spellings: vec![(String::new(), "clampix {0} {1}".into())], ..SymbolRow::named(15, "clampix") });
    t.symbols.push(SymbolRow { adapter: true, prelude: fit.into(), spellings: vec![(String::new(), "fit {0} 0 {1}".into())], ..SymbolRow::named(16, "fit") });
    for (s, ins, out) in [(15i64, vec![SIZE, I64], I64), (16, vec![SIZE, ARR], ARR)] {
        for (k, ty) in ins.iter().enumerate() {
            t.arguments.push(Argument { symbol: s, position: k as u32, direction: Direction::In, data_type: *ty, head: false });
        }
        t.arguments.push(Argument { symbol: s, position: 0, direction: Direction::Out, data_type: out, head: false });
    }
    t.flows.push(Flow::up(15, "bound", p("in0.value")));
    t.flows.push(Flow::up(16, "len", p("in0.value")));
    let pair = "{self(1 := {if in1.len = min(in1.len, in2.len) : {1} else fit ({=min(in1.len, in2.len)}) {zero} ({1})}, 2 := {if in2.len = min(in1.len, in2.len) : {2} else fit ({=min(in1.len, in2.len)}) {zero} ({2})})}";
    t.guards[0].repairs[0] = Repair { template: pair.into(), calls: "fit".into(), ..t.guards[0].repairs[0].clone() };
    t.guards[3].repairs[0].template = "clampix ({=in0.len}) ({1})".into();
    for g in &mut t.guards {
        for r in &mut g.repairs {
            if r.calls == "fit.pair" || r.calls == "fit.state" {
                r.calls = "fit".into();
            }
        }
    }
    let k = contract_of(&t, ARR);
    let d = Decoder::new(&t, &k).unwrap_or_else(|e| panic!("{e:#?}"));
    let mut e = entry();
    e.guards[0].repairs[0] = Repair::new(Site::Root, "fit k 0 ({self})", "fit");
    let dc = |_: SymbolId, w: u32| w.to_string();
    let plain = d.render(&e, &d.decode(&e, &reference(W_N)).expect("decodes"), &dc).expect("renders");
    assert_eq!(plain, Rendered { preludes: vec![], forms: vec![], body: "hist (+) 0 k (tabulate n (\\i -> is[i] % k)) (vs)".into() });
    let mutation = d.decode(&e, &reference(W_K)).expect("decodes");
    let text = d.render(&e, &mutation, &dc).expect("renders");
    assert_eq!(text.body, "hist (+) 0 k (fit (min(k, n)) 0 (tabulate k (\\i -> is[clampix (n) (i)] % k))) (fit (min(k, n)) 0 (vs))");
    assert_eq!(text.preludes, vec![clampix.to_string(), fit.to_string()], "each adapter's prelude once, in first use");
    // Evolution placed clampix itself: tab n (get is (clampix n i)). The guard holds (its bound is n), no repair, and the
    // cost still charges the adapter, as placed.
    let placed = f(TAB, vec![v(VAR_SIZE, W_N), f(GET, vec![v(VAR_ARR, W_IS), f(15, vec![v(VAR_SIZE, W_N), v(VAR_I64, 0)])])]);
    let dec = d.decode(&e, &placed).expect("decodes");
    assert_eq!(dec.failures(), vec![Failure::Root { requirement: 0, adapter: "fit".into() }], "only the result's length n differs from k");
    assert_eq!(d.rescues(&dec), vec![Rescue { path: None, adapter: "fit".into(), placed: false }, Rescue { path: Some(vec![1, 1]), adapter: "clampix".into(), placed: true }]);
    let text = d.render(&e, &dec, &dc).expect("renders");
    assert_eq!(text.body, "fit k 0 (tabulate n (\\i -> is[clampix n i]))");
    assert_eq!(text.preludes, vec![clampix.to_string(), fit.to_string()]);
}
