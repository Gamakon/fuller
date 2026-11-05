//! The master 6 view: `view.py` and this module write the same bytes; every view loads its table of four kinds of row
//! into the gene frame's tables and the contract with its variable terminals, and lays out.

use super::*;
use crate::geneframe::{FlowName, Owner, Place, PropertyExpr, Role, Strength, Ty};
use crate::typed_gep::GeneLayout;

fn first_difference(a: &str, b: &str) -> String {
    match a.lines().zip(b.lines()).enumerate().find(|(_, (x, y))| x != y) {
        Some((i, (x, y))) => format!("line {}: {x:?} vs {y:?}", i + 1),
        None => format!("lengths {} and {}", a.len(), b.len()),
    }
}

/// THE BYTE-FOR-BYTE TEST: every committed master 6 view (the file `gen_tables.py --check` compares with what
/// `sniff.py --format 6` and `view.py` write) is what this module computes from the same specification.
#[test]
fn every_master6_view_is_what_view_py_writes_byte_for_byte() {
    let master = Master6::committed().unwrap();
    for (problem, spec, tables) in COMMITTED6 {
        let v = View6::load(&master, spec).unwrap_or_else(|e| panic!("{problem}: {e}"));
        for (file, got) in v.artefacts() {
            assert!(got == tables, "{problem}/m6/{file} differs from view.py's: {}", first_difference(got, tables));
        }
    }
}

/// `COMMITTED6` IS `gen_tables.py`'s LIST: its problems are exactly `SIG6_PROBLEMS`'s keys, in order, and exactly the
/// directories `kingdoms/futhark/sig/*/m6/`, so no view `view.py` writes goes uncomputed here.
#[test]
fn committed6_is_the_list_of_problems_gen_tables_writes() {
    let text = include_str!("../../../kingdoms/futhark/gen_tables.py");
    let start = text.find("\nSIG6_PROBLEMS = {\n").expect("gen_tables.py has SIG6_PROBLEMS") + 1;
    let block = &text[start..start + text[start..].find("\n}\n").expect("SIG6_PROBLEMS closes")];
    let in_python: Vec<&str> = block.lines().filter_map(|l| l.strip_prefix("    \"")).filter_map(|l| l.split_once("\": (")).map(|(name, _)| name).collect();
    let in_rust: Vec<&str> = COMMITTED6.iter().map(|c| c.0).collect();
    assert_eq!(in_rust, in_python, "COMMITTED6 and gen_tables.py's SIG6_PROBLEMS");
    let sig = concat!(env!("CARGO_MANIFEST_DIR"), "/kingdoms/futhark/sig");
    let mut on_disk: Vec<String> = std::fs::read_dir(sig).unwrap().filter_map(|e| e.ok()).filter(|e| e.path().join("m6").is_dir()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    on_disk.sort();
    assert_eq!(in_rust, on_disk, "COMMITTED6 and the directories sig/*/m6/");
}

/// The words of a laid-out gene, by kind: (head, tail, Dc, Dv, width).
fn words(layout: &GeneLayout) -> (usize, usize, usize, usize, usize) {
    let f = &layout.frames;
    (f.iter().map(|x| x.head_len).sum(), f.iter().map(|x| x.tail_len).sum(), f.iter().map(|x| x.dc_len).sum(), f.iter().map(|x| x.dv_len).sum(), layout.width)
}

/// EVERY VIEW LOADS AND LAYS OUT: its tables check, its contract checks with a `var.T` for each variable row, its
/// formula rows are no symbol of the alphabet, and its gene at program size 192 has a Dv lane of `h_T + t_T` words
/// exactly for the data types with a variable row, and a Dc lane only for those with a constant row.
#[test]
fn every_master6_view_loads_its_tables_and_contract_and_lays_out() {
    for (problem, ..) in COMMITTED6 {
        let k = committed_kingdom(problem).unwrap_or_else(|e| panic!("{problem}: {e}"));
        let formulas = k.tables.symbols.iter().filter(|s| s.role == Role::Formula).count();
        assert!(formulas > 0, "{problem}: the formula rows that spell a size");
        assert_eq!(k.contract.n_symbols() + formulas, k.tables.symbols.len(), "{problem}: the formula rows are not in the alphabet");
        for s in k.symbols() {
            let id = s.symbol as u32;
            assert_eq!(k.contract.is_variable(id), s.role == Role::Variable, "{problem}: {}", s.symbol_name);
            assert_eq!(k.contract.is_constant(id), s.role == Role::Constant, "{problem}: {}", s.symbol_name);
        }
        let layout = GeneLayout::from_program_size(&k.contract, k.entry.output, 192).unwrap_or_else(|e| panic!("{problem}: {e:?}"));
        for f in &layout.frames {
            let has = |role: Role| k.symbols().any(|s| s.role == role && k.output(s.symbol as u32) == f.ty);
            assert_eq!(f.dv_len, if has(Role::Variable) { f.head_len + f.tail_len } else { 0 }, "{problem}: {}", k.data_type(f.ty).unwrap().name);
            assert!(f.dc_len == 0 || has(Role::Constant), "{problem}: {}", k.data_type(f.ty).unwrap().name);
        }
        let (h, t, dc, dv, w) = words(&layout);
        eprintln!("{problem}: {} data types, {} symbols; N = 192: head {h}, tail {t}, Dc {dc}, Dv {dv}, {w} words", k.tables.data_types.len(), k.contract.n_symbols());
        assert_eq!(h + t + dc + dv, w);
    }
}

/// THE K-MEANS GENES at N = 192, measured 2026-11-05 with the adapters in the alphabet (four symbols more than before:
/// `fit`, `fitrows`, `fitcols`, `clampix`), pinned so a change to the master or the view is seen.
#[test]
fn the_kmeans_views_load_and_their_genes_are_the_measured_size() {
    for problem in ["kmeans", "kmeans_i32"] {
        let k = committed_kingdom(problem).unwrap();
        let layout = GeneLayout::from_program_size(&k.contract, k.entry.output, 192).unwrap();
        eprintln!("{problem}, N = 192, {} frames:", layout.frames.len());
        for f in &layout.frames {
            eprintln!("  {:<28} head {:>6} tail {:>6} Dc {:>6} Dv {:>6}", k.data_type(f.ty).unwrap().name, f.head_len, f.tail_len, f.dc_len, f.dv_len);
        }
        let got = words(&layout);
        eprintln!("  total: head {}, tail {}, Dc {}, Dv {}, {} words; {} data types, {} symbols", got.0, got.1, got.2, got.3, got.4, k.tables.data_types.len(), k.contract.n_symbols());
        assert!(k.symbol_named("fitrows@m.f32").is_some(), "{problem}: the adapters are symbols");
        assert!(got.4 > 30_000, "{problem}: {got:?}");
    }
}

/// The k-means view's rows: an up flow, a bound guard repaired by a call of `clampix`, the work bound with its
/// substitutes, a pair guard repaired by calls of `fit` guarded by the work bound's depth, a binder's variables and its
/// trip attribute, the constants' pools, and the entry with its guards repaired by calls of `fitrows` and `fitcols`.
#[test]
fn the_kmeans_view_loads_its_flows_guards_and_entry() {
    let k = committed_kingdom("kmeans").unwrap();
    let by = |name: &str| k.symbol_named(name).unwrap_or_else(|| panic!("{name}")) as i64;
    let guards = |s: i64| -> Vec<String> {
        k.tables
            .guards
            .iter()
            .filter(|g| g.owner == Owner::Symbol(s))
            .map(|g| format!("{} {} {}: {}", g.order, g.strength.name(), g.condition, g.repairs.iter().map(|r| format!("[{}] {} if {} when {}", r.site, r.template, r.guard, r.when)).collect::<Vec<_>>().join("; ")))
            .collect()
    };
    assert_eq!(guards(by("get@f32")), ["1 must in1.bound <= in0.len: [in1] {call clampix@i64({=in0.len}, {1})} if true when known(in0.len); [node] {in.own} if true when true; [node] {const} if true when true"]);
    assert_eq!(guards(by("sum@f32")), ["0 must env.trips < 3: [node] {in.own} if true when true; [node] {const} if true when true"]);
    let add = by("map2.add@a.f32");
    assert_eq!(k.tables.flows.iter().find(|f| f.owner == Owner::Symbol(add) && f.at == Place::Out).map(|f| f.formula.to_string()), Some("min(in0.len, in1.len)".into()));
    let pair = guards(add);
    assert!(pair[1].starts_with("1 must in1.len = in0.len: [node] {self(0 := {if in0.len = min(in1.len, in0.len) : {0} else {call fit@a.f32({=min(in1.len, in0.len)}, {0})}}"), "{pair:?}");
    assert!(pair[1].contains("if env.trips < 3 when known(in1.len) and known(in0.len)"), "{pair:?}");
    assert!(k.tables.guards.iter().all(|g| g.strength == Strength::Must));
    let tab = by("tab@a.f32");
    assert!(k.tables.flows.iter().any(|f| f.owner == Owner::Symbol(tab) && f.at == Place::In(1) && f.name == FlowName::Env("trips".into()) && f.formula == PropertyExpr::parse("env.trips + 1").unwrap()));
    let lp = by("loop@a.f32") as u32;
    assert_eq!((k.trip(lp), k.binds(lp).unwrap()), (Some(2), Some((2, 2))));
    let names: Vec<&str> = k.entry.parameters.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, ["points", "centres", "zeros_n", "zeros_d", "zeros_k", "n", "d", "k"]);
    assert_eq!(k.data_type(k.entry.output).unwrap().name, "[][]f32");
    let root: Vec<String> = k.entry.guards.iter().map(|g| format!("{}: {}", g.condition, g.repairs.iter().map(|r| r.template.clone()).collect::<Vec<_>>().join("; "))).collect();
    assert_eq!(root, ["out.rows = k: {call fitrows@m.f32({=k}, {self})}", "out.cols = d: {call fitcols@m.f32({=d}, {self})}"]);
    assert!(k.tables.data_types.iter().find(|d| d.name == "size").unwrap().constant.is_none());
    assert_eq!(k.tables.data_types.iter().find(|d| d.name == "size").unwrap().size_names, [("d".to_string(), 1), ("k".to_string(), 1), ("n".to_string(), 1)]);
    let lit = by("const@f32") as u32;
    assert_eq!(k.literals(lit), ["0f32", "1f32", "2f32", "-1f32", "0.5f32", "10f32"]);
    for adapter in ["fit@a.f32", "fitrows@m.f32", "fitcols@m.f32", "clampix@i64"] {
        let row = k.symbol(by(adapter) as u32).unwrap();
        assert!(row.adapter && !row.prelude.is_empty(), "{adapter} is an adapter symbol with its prelude");
    }
}

/// A view whose demanded data type can never have a variable in scope gives it its constant (`zero@T`, M0088), a
/// terminal with the constant's properties, and no variable row: the `i32` + `f32` view's `[][]i32`.
#[test]
fn a_demanded_data_type_with_no_variable_has_its_constant_and_no_var_row() {
    let k = committed_kingdom("kmeans_i32").unwrap();
    let m = k.ty("[][]i32").unwrap();
    assert!(!k.entry.parameters.iter().any(|v| v.data_type == m), "no root variable is a [][]i32");
    assert!(!k.tables.flows.iter().any(|f| matches!(f.name, FlowName::Var { ty, .. } if ty == m)), "no binder binds a [][]i32");
    assert!(!k.symbols().any(|s| s.role == Role::Variable && k.output(s.symbol as u32) == m), "a [][]i32 can never be in scope, so it has no var row");
    let zero: Vec<_> = k.symbols().filter(|s| k.output(s.symbol as u32) == m && k.inputs(s.symbol as u32).is_empty()).collect();
    assert_eq!(zero.len(), 1);
    assert_eq!((zero[0].symbol_name.as_str(), zero[0].role, zero[0].alias.as_str(), zero[0].spelling("")), ("zero@m.i32", Role::Function, "M0088", Some("[[0i32]]")));
    let ups: Vec<String> = k.tables.flows.iter().filter(|f| f.owner == Owner::Symbol(zero[0].symbol)).map(|f| format!("{:?}={}", f.name, f.formula)).collect();
    assert_eq!(ups, ["Property(\"rows\")=1", "Property(\"cols\")=1"]);
}

/// NO DEAD Dv LANE: in every view, a frame has a Dv lane only when its data type can have a variable in scope (a root
/// variable's or a binding's data type).
#[test]
fn a_dv_lane_only_for_a_data_type_that_can_have_a_variable() {
    for (problem, ..) in COMMITTED6 {
        let k = committed_kingdom(problem).unwrap();
        let layout = GeneLayout::from_program_size(&k.contract, k.entry.output, 192).unwrap();
        let can_have: std::collections::BTreeSet<Ty> = k.entry.parameters.iter().map(|v| v.data_type).chain(k.tables.flows.iter().filter_map(|f| if let FlowName::Var { ty, .. } = f.name { Some(ty) } else { None })).collect();
        let dead: usize = layout.frames.iter().filter(|f| f.dv_len > 0 && !can_have.contains(&f.ty)).map(|f| f.dv_len).sum();
        assert_eq!(dead, 0, "{problem}: Dv words of data types that never have a variable in scope");
        for s in k.symbols().filter(|s| s.role == Role::Variable) {
            assert!(can_have.contains(&k.output(s.symbol as u32)), "{problem}: {} names a data type no variable is of", s.symbol_name);
        }
    }
}

#[test]
fn a_guard_or_specification_that_does_not_parse_is_a_bug() {
    assert!(parse_guard("must in1.len = in0.len -> fit.pair").is_ok());
    assert!(parse_guard("must env.trips < 3 -> subst").is_ok());
    for bad in ["must in1.len == in0.len -> fit", "must in1.len = in0.len", "must x.len = 1 -> fit", "must in1.len = -> fit", "maybe in1.len = in0.len -> fit"] {
        assert!(parse_guard(bad).unwrap_err().starts_with("our software has a bug: "), "{bad}");
    }
    assert!(ViewSpec6::parse("format\t5\t\t\nmaster\t5\t\t\nelement\tf32\t\t\noutput\tmain\tf32\tn\n").unwrap_err().starts_with("our software has a bug: "));
    assert!(ViewSpec6::parse("nonsense\t1\t\t\n").is_err());
}

#[test]
fn the_template_readers_match_view_py() {
    let env = Env::from([("E".to_string(), "f32".to_string()), ("S.tag".to_string(), "a.f32".to_string())]);
    assert_eq!(subst("loop@{S.tag} {0} {1.v0} {=in0.len} {E}", &env).unwrap(), "loop@a.f32 {0} {1.v0} {=in0.len} f32");
    assert!(subst("{Y}", &env).unwrap_err().contains("nothing binds"));
    assert_eq!(property_terms("min(in0.len, in1.rows) + win2.len + in3.x"), [(0, "len".to_string()), (1, "rows".to_string()), (3, "x".to_string())]);
    assert_eq!(replace_depth("depth + depths - depth", 3), "3 + depths - 3");
    assert_eq!(parse_flow("in2: var i64 with bound := in0.value").unwrap(), FlowItem::Var(2, "i64".into(), vec![("bound".into(), "in0.value".into())]));
    assert_eq!(parse_flow("in2: var (f32, f32)").unwrap(), FlowItem::Var(2, "(f32, f32)".into(), vec![]));
    assert_eq!(parse_flow("out.len := min(in0.len, in1.len)").unwrap(), FlowItem::Up("len".into(), "min(in0.len, in1.len)".into()));
    assert_eq!(parse_flow("in1: env.trips := env.trips + 1").unwrap(), FlowItem::Env(1, "trips".into(), "env.trips + 1".into()));
}
