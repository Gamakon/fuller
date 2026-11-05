//! Tests of the configuration tables (design 3.3): the property catalogue, the text form of the four kinds of row, and
//! the table check's branches.

use super::*;

fn p(t: &str) -> PropertyExpr {
    PropertyExpr::parse(t).expect(t)
}

fn c(t: &str) -> Condition {
    Condition::parse(t).expect(t)
}

const F: Ty = Ty::Futhark(0);
const ARR: Ty = Ty::Futhark(1);
const IX: Ty = Ty::Futhark(2);
const SIZE: Ty = Ty::Futhark(3);

/// A small kingdom with every kind of row: spellings, an adapter symbol with a prelude, data types with properties, a
/// constant and size names, a head argument, up flows reading `out.p`, down flows (an attribute, two variables with
/// their properties and spelling), must and advise guards with inline repairs.
fn tables() -> KingdomTables {
    let mut t = KingdomTables { kingdom: "schema test".into(), ..KingdomTables::default() };
    let sym = |id: i64, name: &str, spelling: &str| SymbolRow { spellings: vec![(String::new(), spelling.to_string())], ..SymbolRow::named(id, name) };
    t.symbols = vec![
        sym(0, "map2", "map2 (+) {0} {1}"),
        sym(1, "tab", "tabulate {0} (\\i -> {1})"),
        sym(2, "get", "{0}[{1}]"),
        SymbolRow { role: Role::Variable, ..SymbolRow::named(3, "var.arr") },
        SymbolRow { role: Role::Variable, ..SymbolRow::named(4, "var.size") },
        SymbolRow { adapter: true, prelude: "def clampix (m: i64) (i: i64) : i64 = i64.max 0 (i64.min (m - 1) i)".into(), ..sym(5, "clampix", "clampix {0} {1}") },
        SymbolRow { role: Role::Constant, ..SymbolRow::named(6, "?f") },
        SymbolRow { spellings: vec![(String::new(), "{0}".into()), ("pl".into(), "{0}s".into())], ..SymbolRow::named(7, "noun") },
    ];
    t.data_types = vec![
        DataType::scalar(F, "f32"),
        DataType { kind: DataKind::Array { element: F }, properties: vec!["len".into(), "sorted".into()], constant: Some(Constant { spelling: "[0f32]".into(), properties: vec![("len".into(), p("1"))] }), ..DataType::scalar(ARR, "[]f32") },
        DataType { properties: vec!["bound".into()], constant: Some(Constant { spelling: "0i64".into(), properties: vec![("bound".into(), p("1"))] }), ..DataType::scalar(IX, "i64") },
        DataType { properties: vec!["value".into()], size_names: vec![("k".into(), 1), ("n".into(), 0)], ..DataType::scalar(SIZE, "size") },
    ];
    let mut arg = |symbol: i64, ins: &[Ty], out: Ty| {
        for (i, ty) in ins.iter().enumerate() {
            t.arguments.push(Argument { symbol, position: i as u32, direction: Direction::In, data_type: *ty, head: symbol == 7 && i == 0 });
        }
        t.arguments.push(Argument { symbol, position: 0, direction: Direction::Out, data_type: out, head: false });
    };
    arg(0, &[ARR, ARR], ARR);
    arg(1, &[SIZE, F], ARR);
    arg(2, &[ARR, IX], F);
    arg(3, &[], ARR);
    arg(4, &[], SIZE);
    arg(5, &[SIZE, IX], IX);
    arg(6, &[], F);
    arg(7, &[F], F);
    t.flows = vec![
        Flow::up(0, "len", p("min(in0.len, in1.len)")),
        Flow::up(0, "sorted", p("if(out.len <= 1, true, false)")),
        Flow::up(1, "len", p("in0.value")),
        Flow::up(5, "bound", p("in0.value")),
        Flow::down(1, 1, FlowName::Env("trips".into()), p("env.trips + 1")),
        Flow::down(1, 1, FlowName::Var { ty: IX, group: 0 }, p("2")),
        Flow::down(1, 1, FlowName::VarProperty { ty: IX, group: 0, property: "bound".into() }, p("in0.value + each")),
        spell(1, 1, IX, 0, "i{each}"),
    ];
    let mut fit = Repair::new(Site::Node, "{self(0 := fit ({=min(in0.len, in1.len)}) {zero} ({0}))}", "clampix");
    fit.guard = c("env.trips < 3");
    fit.when = c("known(in0.len)");
    t.guards = vec![
        Guard { owner: Owner::Symbol(0), order: 1, condition: c("in1.len = in0.len"), strength: Strength::Must, repairs: vec![fit, Repair::new(Site::Node, "{in.own}", ""), Repair::new(Site::Node, "{const}", "")] },
        Guard { owner: Owner::Symbol(0), order: 0, condition: c("env.trips < 3"), strength: Strength::Must, repairs: vec![Repair::new(Site::Node, "{in.own}", "")] },
        Guard { owner: Owner::Symbol(2), order: 0, condition: c("in1.bound <= in0.len"), strength: Strength::Must, repairs: vec![Repair::new(Site::In(1), "clampix ({=in0.len}) ({1})", "clampix")] },
        Guard { owner: Owner::Symbol(0), order: 2, condition: c("out.sorted"), strength: Strength::Advise, repairs: vec![] },
    ];
    t
}

#[test]
fn the_catalogue_reads_and_names_each_property_once() {
    let cat = parse_catalogue(CATALOGUE_TSV).expect("the catalogue reads");
    assert_eq!(cat.len(), catalogue().len());
    assert_eq!(property("trips").and_then(|x| x.root.clone()), Some(p("0")));
    assert_eq!(property("uniform").and_then(|x| x.root.clone()), Some(p("true")));
    assert_eq!(property("access").map(|x| x.values.clone()), Some(vec!["read".to_string(), "read_write".to_string()]));
    assert_eq!(property("shape").map(|x| (x.kind, x.element)), Some((PropertyKind::List, Some(PropertyKind::Size))));
    let header = "name\tkind\telement\tvalues\troot\tmeaning\n";
    for bad in [
        "name\tkind\n".to_string(),
        format!("{header}len\tsize\t\t\t\tone\nlen\tsize\t\t\t\ttwo\n"),
        format!("{header}x\tlist\t\t\t\tno element\n"),
        format!("{header}x\tsize\t\ta,b\t\tvalues on a size\n"),
        format!("{header}x\tcolour\t\t\t\tno such kind\n"),
    ] {
        assert!(parse_catalogue(&bad).expect_err(&bad).starts_with("our software has a bug: the property catalogue:"), "{bad}");
    }
}

#[test]
fn the_four_kinds_of_row_round_trip_through_text() {
    let t = tables();
    t.check().unwrap_or_else(|d| panic!("{d:#?}"));
    assert!(t.has_properties());
    let text = t.to_text().expect("text");
    let back = KingdomTables::parse(&text).expect("parse");
    assert_eq!(back, t);
    assert_eq!(back.to_text().expect("text"), text, "a fixed point");
    assert_eq!(t.bounds(), SizeBounds::from([("k".to_string(), 1), ("n".to_string(), 0)]));
    assert_eq!(t.attributes(), ["trips"]);
    assert_eq!(t.head(7), Some(0));
    assert_eq!(t.inputs(5), vec![SIZE, IX]);
    assert_eq!(t.arity(0).total_in(), 2);
    assert_eq!(t.decode_order(1).expect("order"), vec![0, 1], "tab's in1 reads in0.value");
    assert_eq!(t.up_order(0).expect("order"), vec![0, 1], "sorted reads out.len, so after it");
    // A kingdom with no property rows is today's.
    let plain = KingdomTables::from_signatures("plain", vec![(SymbolRow::named(0, "x"), vec![], Ty::F)]);
    plain.check().expect("plain");
    assert!(!plain.has_properties());
    let mut tab = t.clone();
    tab.symbols[0].spellings[0].1 = "a\tb".into();
    assert!(tab.to_text().expect_err("a tab").starts_with("our software has a bug:"));
    assert!(KingdomTables::parse("symbol\t1\tx").expect_err("short row").starts_with("our software has a bug:"));
    assert!(KingdomTables::parse(&format!("{text}kingdom\tother\n")).expect_err("two kingdoms").contains("names a second kingdom"));
}

/// Every branch of the check worth naming, each provoked alone and found by its message.
#[test]
fn every_branch_of_the_table_check() {
    type Edit = Box<dyn Fn(&mut KingdomTables)>;
    let cases: Vec<(&str, Edit)> = vec![
        ("symbol 0 (map2) is listed twice", Box::new(|t| t.symbols.push(t.symbols[0].clone()))),
        ("carries color, which the property catalogue does not define", Box::new(|t| t.data_types[1].properties.push("color".into()))),
        ("constant sets bound, which it does not carry", Box::new(|t| t.data_types[1].constant.as_mut().unwrap().properties.push(("bound".into(), p("1"))))),
        ("the size name \"n\" of size is not a name or is listed twice", Box::new(|t| t.data_types[3].size_names.push(("n".into(), 1)))),
        ("is a constant and takes 2 inputs", Box::new(|t| t.symbols[0].role = Role::Constant)),
        ("the variable terminal var.arr has flows", Box::new(|t| t.flows.push(Flow::up(3, "len", p("1"))))),
        ("sets out.bound, which Futhark(1) does not carry", Box::new(|t| t.flows.push(Flow::up(0, "bound", p("1"))))),
        ("reads in2.len, but the symbol has 2 inputs", Box::new(|t| t.flows[0].formula = p("in2.len"))),
        ("reads out.len, which cannot be read here", Box::new(|t| t.flows[6].formula = p("out.len"))),
        ("reads env.depth, which is not an environment attribute", Box::new(|t| t.flows[0].formula = p("env.depth"))),
        ("its own argument", Box::new(|t| t.flows[6].formula = p("in1.value"))),
        ("up flows read each other's out.p in a cycle", Box::new(|t| t.flows[0].formula = p("if(out.sorted, 1, 2)"))),
        ("which no up flow of the symbol sets", Box::new(|t| t.guards.push(Guard { owner: Owner::Symbol(1), order: 0, condition: c("out.sorted"), strength: Strength::Advise, repairs: vec![] }))),
        ("reads out.bound, which Futhark(0) does not carry", Box::new(|t| t.guards[2].condition = c("out.bound = 1"))),
        ("compares in1.len with 'read', which mixes kinds", Box::new(|t| t.guards[0].condition = c("in1.len = 'read'"))),
        ("a bound compares by <= only", Box::new(|t| t.guards[2].condition = c("in1.bound = in0.len"))),
        ("a tag, a flag or a list compares by = only", Box::new(|t| t.guards[3].condition = c("out.sorted <= true"))),
        ("is must and has no repair", Box::new(|t| t.guards[2].repairs.clear())),
        ("is advise and has a repair", Box::new(|t| t.guards[3].repairs.push(Repair::new(Site::Node, "{in.own}", "")))),
        ("the order is used twice", Box::new(|t| t.guards[1].order = 1)),
        ("runs at input 7, which the symbol has not", Box::new(|t| t.guards[2].repairs[0].site = Site::In(7))),
        ("runs at the root, which only the entry's guards do", Box::new(|t| t.guards[2].repairs[0].site = Site::Root)),
        ("is a substitute at in1", Box::new(|t| t.guards[2].repairs[0].template = "{const}".into())),
        ("calls fit, which is not an adapter symbol", Box::new(|t| t.guards[2].repairs[0].calls = "fit".into())),
        ("a repair's guard reads only the site's environment", Box::new(|t| t.guards[0].repairs[0].guard = c("in0.len = 1"))),
        ("every repair is a substitute the node never has", Box::new(|t| t.guards.push(Guard { owner: Owner::Symbol(2), order: 1, condition: c("in1.bound <= 1"), strength: Strength::Must, repairs: vec![Repair::new(Site::Node, "{in.own}", "")] }))),
        ("has a { that is not closed", Box::new(|t| t.symbols[0].spellings[0].1 = "map2 {0".into())),
        ("is not in its one spelling", Box::new(|t| t.flows[2].formula = PropertyExpr::Sum(vec![p("in0.value")]))),
        ("sets a property of variables no var flow introduces", Box::new(|t| t.flows.push(Flow::down(1, 1, FlowName::VarProperty { ty: ARR, group: 0, property: "len".into() }, p("1"))))),
    ];
    for (want, edit) in cases {
        let mut t = tables();
        edit(&mut t);
        let d = t.check().expect_err(want);
        assert!(d.iter().any(|e| e.contains(want)), "missing {want:?} in {d:#?}");
        assert!(d.iter().all(|e| e.starts_with("our software has a bug:")), "{d:#?}");
    }
}

/// The entry check: parameters, their properties and sources, size names, alternative forms, the entry's guards and
/// flows.
#[test]
fn every_branch_of_the_entry_check() {
    let t = tables();
    let good = || {
        let mut e = Entry::new("test", ARR);
        e.parameters.push(EntryParameter::declared("xs", ARR, &[("len", p("n"))]));
        e.parameters.push(EntryParameter { name: "ys".into(), data_type: ARR, properties: vec![("len".into(), p("k"), Source::Measured("train".into()))], form_of: None });
        e.parameters.push(EntryParameter { name: "xs_sorted".into(), data_type: ARR, properties: vec![], form_of: Some(("xs".into(), Transform { template: "sort {0}".into(), properties: vec![("len".into(), p("in0.len")), ("sorted".into(), p("true"))], cost: p("in0.len * 3") })) });
        e.guards.push(Guard { owner: Owner::Entry, order: 0, condition: c("out.len = k"), strength: Strength::Must, repairs: vec![Repair::new(Site::Root, "fit k 0 ({self})", "clampix")] });
        e.guards.push(Guard { owner: Owner::Entry, order: 1, condition: c("out.sorted"), strength: Strength::Advise, repairs: vec![] });
        e.flows.push(Flow { owner: Owner::Entry, at: Place::Out, name: FlowName::Env("trips".into()), formula: p("1"), template: String::new(), way: Way::Down });
        e
    };
    assert_eq!(t.entry_defects(&good()), Vec::<String>::new());
    type Edit = Box<dyn Fn(&mut Entry)>;
    let cases: Vec<(&str, Edit)> = vec![
        ("the entry returns Futhark(9)", Box::new(|e| e.output = Ty::Futhark(9))),
        ("the entry has two parameters named xs", Box::new(|e| e.parameters.push(e.parameters[0].clone()))),
        ("sets bound, which Futhark(1) does not carry", Box::new(|e| e.parameters[0].properties.push(("bound".into(), p("1"), Source::Declared)))),
        ("reads the size name m, which no data type's size_names declares", Box::new(|e| e.parameters[0].properties[0].1 = p("m"))),
        ("which an entry value cannot read", Box::new(|e| e.parameters[0].properties[0].1 = p("in0.len"))),
        ("is a form of zs, which is no earlier parameter", Box::new(|e| e.parameters[2].form_of.as_mut().unwrap().0 = "zs".into())),
        ("transform sets bound", Box::new(|e| e.parameters[2].form_of.as_mut().unwrap().1.properties.push(("bound".into(), p("1"))))),
        ("reads out.bound, which Futhark(1) does not carry", Box::new(|e| e.guards[0].condition = c("out.bound <= k"))),
        ("the entry's guard reads only out.p and env.a", Box::new(|e| e.guards[0].condition = c("in0.len = k"))),
        ("runs at node, and the entry's guards repair at the root", Box::new(|e| e.guards[0].repairs[0].site = Site::Node)),
        ("is must and has no repair", Box::new(|e| e.guards[0].repairs.clear())),
        ("the entry sets env.trips twice", Box::new(|e| e.flows.push(e.flows[0].clone()))),
        ("the entry sets env.depth, which is not an environment attribute", Box::new(|e| e.flows[0].name = FlowName::Env("depth".into()))),
    ];
    for (want, edit) in cases {
        let mut e = good();
        edit(&mut e);
        let d = t.entry_defects(&e);
        assert!(d.iter().any(|x| x.contains(want)), "missing {want:?} in {d:#?}");
    }
}
