//! The master 6 view (task F1b): `view.py` and this module write the same bytes; every view loads through
//! `Kingdom6::from_tables` into the gene frame's tables and the contract with its variable terminals, and lays out.

use super::*;
use crate::futhark_table::Kind6;
use crate::geneframe::{Relation, Subject};
use crate::geneframe::Ty;
use crate::typed_gep::GeneLayout;

fn first_difference(a: &str, b: &str) -> String {
    match a.lines().zip(b.lines()).enumerate().find(|(_, (x, y))| x != y) {
        Some((i, (x, y))) => format!("line {}: {x:?} vs {y:?}", i + 1),
        None => format!("lengths {} and {}", a.len(), b.len()),
    }
}

/// THE BYTE-FOR-BYTE TEST: every committed master 6 view (the files `gen_tables.py --check` compares with what
/// `sniff.py --format 6` and `view.py` write) is what this module computes from the same specification.
#[test]
fn every_master6_view_is_what_view_py_writes_byte_for_byte() {
    let master = Master6::committed().unwrap();
    for (problem, spec, datatypes, symbols, environment) in COMMITTED6 {
        let v = View6::load(&master, spec).unwrap_or_else(|e| panic!("{problem}: {e}"));
        for ((file, got), want) in v.artefacts().into_iter().zip([datatypes, symbols, environment]) {
            assert!(got == want, "{problem}/m6/{file} differs from view.py's: {}", first_difference(got, want));
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

/// EVERY VIEW LOADS AND LAYS OUT: its tables check (G1), its contract checks with a `var.T` for each `var` row (G2),
/// and its gene at program size 192 has a Dv lane of `h_T + t_T` words exactly for the data types with a `var` row,
/// and a Dc lane exactly for those with a `const` row.
#[test]
fn every_master6_view_loads_its_tables_and_contract_and_lays_out() {
    for (problem, ..) in COMMITTED6 {
        let k = committed(problem).unwrap().kingdom(problem).unwrap_or_else(|e| panic!("{problem}: {e}"));
        let tables = k.tables().unwrap_or_else(|e| panic!("{problem}: {e}"));
        assert_eq!(tables.symbols.len(), k.symbols.len());
        assert_eq!(tables.data_types.len(), k.data_types.len());
        let contract = k.contract().unwrap_or_else(|e| panic!("{problem}: {e}"));
        for (i, s) in k.symbols.iter().enumerate() {
            assert_eq!(contract.is_variable(i as u32), s.kind == Kind6::Var, "{problem}: {}", s.name);
            assert_eq!(contract.is_constant(i as u32), s.kind == Kind6::Const, "{problem}: {}", s.name);
            assert_eq!(tables.arity(i as i64).total_in() as usize, s.inputs.len(), "{problem}: {}", s.name);
        }
        let layout = GeneLayout::from_program_size(&contract, k.root.data_type, 192).unwrap_or_else(|e| panic!("{problem}: {e:?}"));
        for f in &layout.frames {
            let has = |kind: Kind6| k.symbols.iter().any(|s| s.kind == kind && s.output == f.ty);
            assert_eq!(f.dv_len, if has(Kind6::Var) { f.head_len + f.tail_len } else { 0 }, "{problem}: {}", k.data_type(f.ty).unwrap().name);
            assert_eq!(f.dc_len > 0, has(Kind6::Const) && f.dc_len > 0, "{problem}: {}", k.data_type(f.ty).unwrap().name);
        }
        let (h, t, dc, dv, w) = words(&layout);
        eprintln!("{problem}: {} data types, {} symbols; N = 192: head {h}, tail {t}, Dc {dc}, Dv {dv}, {w} words", k.data_types.len(), k.symbols.len());
        assert_eq!(h + t + dc + dv, w);
    }
}

/// THE K-MEANS GENES (design 7.1: about 8 types and 31,000 words at N = 192, against master 5's 14,359 types and
/// 27.6 million words). Measured 2026-11-05 at N = 192: the `f32` view, 9 data types, head 3,057, tail 12,995, Dc
/// 4,202, Dv 12,230, 32,484 words; the faithful `i32` + `f32` view (master 5: 237,325 types, which did not load), 22
/// data types, 122,447 words (123,020 until its `[][]i32`, which no variable can be of, lost its var row and 573-word
/// Dv lane for its constant, M0088). The sizes are pinned, so a change to the master or the view is seen.
#[test]
fn the_kmeans_views_load_and_their_genes_are_the_measured_size() {
    for (problem, types, symbols, want) in [("kmeans", 9, 84, (3_057, 12_995, 4_202, 12_230, 32_484)), ("kmeans_i32", 22, 228, (8_023, 54_634, 19_482, 40_308, 122_447))] {
        let k = committed(problem).unwrap().kingdom(problem).unwrap();
        let contract = k.contract().unwrap();
        let layout = GeneLayout::from_program_size(&contract, k.root.data_type, 192).unwrap();
        eprintln!("{problem}, N = 192, {} frames:", layout.frames.len());
        for f in &layout.frames {
            eprintln!("  {:<28} head {:>6} tail {:>6} Dc {:>6} Dv {:>6}", k.data_type(f.ty).unwrap().name, f.head_len, f.tail_len, f.dc_len, f.dv_len);
        }
        let got = words(&layout);
        eprintln!("  total: head {}, tail {}, Dc {}, Dv {}, {} words", got.0, got.1, got.2, got.3, got.4);
        assert_eq!((k.data_types.len(), k.symbols.len()), (types, symbols), "{problem}");
        assert_eq!(got, want, "{problem}");
    }
}

/// The columns master 6 adds, as loaded into the gene frame's tables: a property rule, an index's `<=` requirement, a
/// length requirement and the work bound with their adapters, a binder's bindings and trip argument, the constants,
/// and the root with its requirements and environment in the specification's order.
#[test]
fn the_kmeans_view_loads_its_rules_requirements_bindings_and_environment() {
    let k = committed("kmeans").unwrap().kingdom("kmeans").unwrap();
    let tables = k.tables().unwrap();
    let by = |name: &str| k.symbols.iter().position(|s| s.name == name).unwrap_or_else(|| panic!("{name}"));
    let get = &k.symbols[by("get@f32")];
    assert_eq!(get.requires.len(), 1);
    assert_eq!((&get.requires[0].subject, get.requires[0].relation, get.requires[0].adapter.as_str()), (&Subject::Input { position: 1, property: "bound".into() }, Relation::Le, "clampix"));
    let sum = &k.symbols[by("sum@f32")];
    assert_eq!((&sum.requires[0].subject, sum.requires[0].relation, sum.requires[0].against.to_string()), (&Subject::Trips, Relation::Lt, "3".to_string()));
    let add = by("map2.add@a.f32");
    assert_eq!(k.symbols[add].rules[0].1.to_string(), "min(in0.len, in1.len)");
    let in_tables: Vec<String> = tables.requirements.iter().filter(|q| q.symbol == add as i64).map(|q| format!("{} {} {} -> {}", q.subject, q.relation.symbol(), q.against, q.adapter)).collect();
    assert_eq!(in_tables, ["in1.len = in0.len -> fit.pair", "env.trips < 3 -> subst"]);
    assert_eq!(tables.requirements.len(), k.symbols.iter().map(|s| s.requires.len()).sum::<usize>());
    assert!(tables.trip_arguments.iter().any(|t| t.symbol == by("tab@a.f32") as i64 && t.position == 1));
    assert_eq!(tables.adapters.len(), 9);
    let lp = &k.symbols[by("loop@a.f32")];
    assert_eq!((lp.trip, lp.binds.len()), (Some(2), 2));
    assert_eq!(k.data_type(lp.binds[0].data_type).unwrap().name, "[]f32");
    assert_eq!(lp.binds[0].properties[0].1.to_string(), "in1.len");
    assert_eq!(tables.bindings.iter().filter(|b| b.symbol == by("loop@a.f32") as i64).count(), 2);
    let names: Vec<&str> = k.variables.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, ["points", "centres", "zeros_n", "zeros_d", "zeros_k", "n", "d", "k"]);
    assert_eq!(k.data_type(k.root.data_type).unwrap().name, "[][]f32");
    assert_eq!(k.root.requires.iter().map(|q| (q.property.as_str(), q.relation, q.against.to_string(), q.adapter.as_str())).collect::<Vec<_>>(), [("rows", Relation::Eq, "k".to_string(), "fit.root"), ("cols", Relation::Eq, "d".to_string(), "fit.root")]);
    assert!(tables.data_types.iter().find(|d| d.name == "size").unwrap().constant.is_none());
    assert_eq!(k.data_type(Ty::Futhark(0)).unwrap().literals, ["0", "1", "2", "-1", "0.5", "10"]);
    assert_eq!(k.data_type(k.ty("[][]f32").unwrap()).unwrap().constant_props.len(), 2);
}

/// A view whose demanded data type can never have a variable in scope gives it its constant (`zero@T`, M0088), a
/// terminal with the constant's properties, and no `var` row: the `i32` + `f32` view's `[][]i32`, built only by
/// `tabrows`.
#[test]
fn a_demanded_data_type_with_no_variable_has_its_constant_and_no_var_row() {
    let k = committed("kmeans_i32").unwrap().kingdom("kmeans_i32").unwrap();
    let m = k.ty("[][]i32").unwrap();
    assert!(!k.variables.iter().any(|v| v.data_type == m), "no root variable is a [][]i32");
    assert!(!k.symbols.iter().any(|s| s.binds.iter().any(|b| b.data_type == m)), "no binder binds a [][]i32");
    assert!(!k.symbols.iter().any(|s| s.kind == Kind6::Var && s.output == m), "a [][]i32 can never be in scope, so it has no var row");
    let zero: Vec<_> = k.symbols.iter().filter(|s| s.output == m && s.inputs.is_empty()).collect();
    assert_eq!(zero.len(), 1);
    assert_eq!((zero[0].name.as_str(), zero[0].kind, zero[0].reference.as_str(), zero[0].spelling.as_str()), ("zero@m.i32", Kind6::Terminal, "M0088", "[[0i32]]"));
    assert_eq!(zero[0].rules.iter().map(|(p, e)| format!("{p}={e}")).collect::<Vec<_>>(), ["rows=1", "cols=1"]);
    assert_eq!(k.data_type(m).unwrap().constant, "[[0i32]]");
}

/// NO DEAD Dv LANE: in every view, a frame has a Dv lane only when its data type can have a variable in scope (a root
/// variable's or a binding's data type); every other demanded data type has a leaf that is not a variable.
#[test]
fn a_dv_lane_only_for_a_data_type_that_can_have_a_variable() {
    for (problem, ..) in COMMITTED6 {
        let k = committed(problem).unwrap().kingdom(problem).unwrap();
        let contract = k.contract().unwrap();
        let layout = GeneLayout::from_program_size(&contract, k.root.data_type, 192).unwrap();
        let can_have: std::collections::BTreeSet<Ty> = k.variables.iter().map(|v| v.data_type).chain(k.symbols.iter().flat_map(|s| s.binds.iter().map(|b| b.data_type))).collect();
        let dead: usize = layout.frames.iter().filter(|f| f.dv_len > 0 && !can_have.contains(&f.ty)).map(|f| f.dv_len).sum();
        assert_eq!(dead, 0, "{problem}: Dv words of data types that never have a variable in scope");
        for s in k.symbols.iter().filter(|s| s.kind == Kind6::Var) {
            assert!(can_have.contains(&s.output), "{problem}: {} names a data type no variable is of", s.name);
        }
    }
}

#[test]
fn a_requirement_or_specification_that_does_not_parse_is_a_bug() {
    assert!(parse_requirement("in1.len = in0.len -> fit.pair").is_ok());
    assert!(parse_requirement("env.trips < 3 -> subst").is_ok());
    for bad in ["in1.len == in0.len -> fit", "in1.len = in0.len", "x.len = 1 -> fit", "in1.len = -> fit"] {
        assert!(parse_requirement(bad).unwrap_err().starts_with("our software has a bug: "), "{bad}");
    }
    assert!(ViewSpec6::parse("format\t5\t\t\nmaster\t5\t\t\nelement\tf32\t\t\noutput\tmain\tf32\tn\n").unwrap_err().starts_with("our software has a bug: "));
    assert!(ViewSpec6::parse("nonsense\t1\t\t\n").is_err());
}

#[test]
fn the_template_readers_match_view_py() {
    let env = Env::from([("E".to_string(), "f32".to_string()), ("S.tag".to_string(), "a.f32".to_string())]);
    assert_eq!(subst("loop@{S.tag} {0} {1.v0} {v} {E}", &env).unwrap(), "loop@a.f32 {0} {1.v0} {v} f32");
    assert!(subst("{Y}", &env).unwrap_err().contains("nothing binds"));
    assert_eq!(property_terms("min(in0.len, in1.rows) + win2.len + in3.x"), [(0, "len".to_string()), (1, "rows".to_string()), (3, "x".to_string())]);
    assert_eq!(replace_depth("depth + depths - depth", 3), "3 + depths - 3");
    assert_eq!(parse_binding("2:i64 bound=in0.value").unwrap(), (2, "i64".to_string(), "bound=in0.value".to_string()));
    assert_eq!(parse_binding("2:f32").unwrap(), (2, "f32".to_string(), String::new()));
}
