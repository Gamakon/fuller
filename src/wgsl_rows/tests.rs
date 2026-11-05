//! Tests of the WGSL kingdom as rows: the table is the generator's and loads; it has nine data types; 200 random genes
//! render kernels that naga validates N of N; the host twin of the device rules decodes equal to the decoder; every
//! guard is exercised by the random genes (so each rule is live, not decorative).

use std::collections::BTreeMap;
use std::process::Command;

use super::*;
use crate::futhark_table::GeneRng;

const GENES: usize = 200;
const PROGRAM_SIZE: usize = 32;

fn kingdom() -> WgslRows {
    WgslRows::load().expect("the WGSL rows load")
}

/// The `var.T` node that names the entry's parameter `name` at the root (its index word in the environment's order:
/// the innermost first, so the parameters count back from the last).
fn param(k: &WgslRows, name: &str) -> Tree {
    let env = k.decoder.root_environment(&k.entry);
    let v = env.variables.iter().find(|v| v.name.as_deref() == Some(name)).unwrap_or_else(|| panic!("{name} is a parameter"));
    let word = env.index_of(v.data_type, v.binder).expect("in scope");
    Tree { symbol: k.symbol_named(&format!("var@{}", k.type_name(v.data_type))).expect("the variable terminal"), const_index: Some(word), kids: vec![] }
}

#[test]
fn the_table_is_the_generators() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kingdoms/wgsl_true");
    let out = Command::new("python3").args(["-I", "gen_rows.py", "--check"]).current_dir(&dir).output().expect("python3 runs");
    assert!(out.status.success(), "gen_rows.py --check: {}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
}

#[test]
fn nine_data_types_and_the_root_is_the_kernel() {
    let k = kingdom();
    assert_eq!(k.tables.data_types.len(), 9, "{:?}", k.tables.data_types.iter().map(|d| d.name.as_str()).collect::<Vec<_>>());
    assert_eq!(k.type_name(k.root()), "kernel");
    assert_eq!(k.const_len().len(), 3, "u32, f32 and bool have pools: {:?}", k.const_len());
    let layout = k.layout(PROGRAM_SIZE).expect("the layout");
    assert!(layout.width > 0);
}

/// The 200 random kernels, with each one's program.
fn kernels() -> (WgslRows, Vec<(Vec<u32>, Tree, WgslProgram)>) {
    let k = kingdom();
    let layout = k.layout(PROGRAM_SIZE).expect("the layout");
    let mut rng = GeneRng::new(0x0057_4753_4c5f_4b33);
    let mut out = Vec::new();
    for i in 0..GENES {
        let gene = k.random_gene(&layout, &mut rng);
        let tree = k.express(&layout, &gene).unwrap_or_else(|e| panic!("gene {i}: {e}"));
        let program = k.program(&tree).unwrap_or_else(|e| panic!("gene {i}: {e}"));
        out.push((gene, tree, program));
    }
    (k, out)
}

#[cfg(feature = "wgsl")]
#[test]
fn two_hundred_random_genes_pass_naga_n_of_n() {
    let (_, kernels) = kernels();
    let mut failures = Vec::new();
    for (i, (_, _, p)) in kernels.iter().enumerate() {
        let module = match naga::front::wgsl::parse_str(&p.text) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("gene {i}: parse: {}\n{}", e.emit_to_string(&p.text), p.text));
                continue;
            }
        };
        let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
        if let Err(e) = v.validate(&module) {
            failures.push(format!("gene {i}: validate: {}\n{}", e.emit_to_string(&p.text), p.text));
        }
    }
    assert!(failures.is_empty(), "{} of {GENES} fail naga:\n{}", failures.len(), failures.join("\n\n"));
}

#[test]
fn the_host_twin_of_the_device_rules_decodes_equal_to_the_decoder() {
    let (k, kernels) = kernels();
    let rules = k.device_rules().expect("every row is in the device's closed forms");
    assert_eq!(rules.len(), k.contract.n_symbols());
    for (i, (_, tree, _)) in kernels.iter().enumerate() {
        let d = k.decode(tree).expect("decodes");
        let twin = k.twin_decode(&rules, tree).unwrap_or_else(|e| panic!("gene {i}: {e}"));
        assert_eq!(twin, k.twin_of(&d), "gene {i}");
    }
}

#[test]
fn every_guard_is_exercised_and_the_work_bound_holds() {
    let (k, kernels) = kernels();
    let mut fired: BTreeMap<String, usize> = BTreeMap::new();
    for (_, _, p) in &kernels {
        for (_, c) in p.substitutions.iter().chain(&p.rewrites) {
            *fired.entry(c.clone()).or_default() += 1;
        }
    }
    let conditions: Vec<String> = k.tables.guards.iter().map(|g| g.condition.to_string()).collect();
    let silent: Vec<&String> = conditions.iter().filter(|c| !fired.contains_key(*c)).collect();
    assert!(silent.is_empty(), "guards no random gene fired: {silent:?}; fired {fired:?}");
    // the work bound: no kernel nests loops past the budget (read back from the rendered text's `for` counts)
    for (_, tree, _) in &kernels {
        let d = k.decode(tree).expect("decodes");
        fn deepest(n: &Node, work: i64, k: &WgslRows) -> i64 {
            let mut w = work;
            if k.name(n.symbol) == "loop" && n.substituted.is_none() {
                if let Some(PropertyExpr::Literal(v)) = n.kids[0].properties.get("value") {
                    w *= v;
                }
            }
            n.kids.iter().map(|c| deepest(c, w, k)).max().unwrap_or(w)
        }
        assert!(deepest(&d.root, 1, &k) <= BUDGET);
    }
}

#[test]
fn a_store_through_a_shared_index_is_dropped_and_through_an_owned_one_stands() {
    let k = kingdom();
    let s = |n: &str| k.symbol_named(n).unwrap_or_else(|| panic!("{n}"));
    let var = |name: &str| param(&k, name);
    let leaf = |n: &str| Tree { symbol: s(n), const_index: None, kids: vec![] };
    let node = |n: &str, kids: Vec<Tree>| Tree { symbol: s(n), const_index: None, kids };
    let store = |ix: Tree| node("main", vec![leaf("g8x8"), node("store", vec![var("h"), ix, leaf("rf0")])]);
    let owned = k.program(&store(var("lin"))).expect("renders");
    assert!(owned.substitutions.is_empty(), "{owned:?}");
    assert_eq!(owned.rewrites.len(), 1, "the index is clamped, its bound unknown: {owned:?}");
    assert!(owned.text.contains("h[min(lin, arrayLength(&h) - 1u)] = rf0;"), "{}", owned.text);
    let shared = k.program(&store(var("gid_x"))).expect("renders");
    assert_eq!(shared.substitutions.len(), 1, "{shared:?}");
    assert!(!shared.text.contains("gid_x]"), "{}", shared.text);
    // a store into phi (read) is dropped before the index is looked at
    let read_only = k.program(&node("main", vec![leaf("g8x8"), node("store", vec![var("phi"), var("lin"), leaf("rf0")])])).expect("renders");
    assert_eq!(read_only.substitutions[0].1, "in0.access = 'read_write'");
    // an owned index shifted by a uniform stays owned; by a non-uniform one does not
    let shifted = k.program(&store(node("add_u", vec![var("lin"), var("cfg_d")]))).expect("renders");
    assert!(shifted.substitutions.is_empty(), "lin + cfg_d is owned: {shifted:?}");
    let mixed = k.program(&store(node("add_u", vec![var("lin"), var("gid_x")]))).expect("renders");
    assert_eq!(mixed.substitutions.len(), 1, "lin + gid_x is not owned: {mixed:?}");
}

#[test]
fn a_barrier_under_a_non_uniform_branch_is_dropped_and_a_deep_loop_is_dropped() {
    let k = kingdom();
    let s = |n: &str| k.symbol_named(n).unwrap_or_else(|| panic!("{n}"));
    let var = |name: &str| param(&k, name);
    let leaf = |n: &str| Tree { symbol: s(n), const_index: None, kids: vec![] };
    let node = |n: &str, kids: Vec<Tree>| Tree { symbol: s(n), const_index: None, kids };
    let branch = |cond: Tree| node("main", vec![leaf("g8x8"), node("if_", vec![cond, leaf("barrier"), leaf("barrier")])]);
    let non_uniform = k.program(&branch(node("lt_u", vec![var("gid_x"), var("cfg_d")]))).expect("renders");
    assert_eq!(non_uniform.substitutions.len(), 2, "{non_uniform:?}");
    let uniform = k.program(&branch(node("lt_u", vec![var("cfg_d"), var("cfg_m")]))).expect("renders");
    assert!(uniform.substitutions.is_empty(), "{uniform:?}");
    assert_eq!(uniform.text.matches("workgroupBarrier();").count(), 2);
    // 8 * 8 = 64 trips stand; a third level of 2 is over the budget and dropped, and so is one loop of 128
    let nest = |inner: Tree| node("main", vec![leaf("g8x8"), node("loop", vec![leaf("c8"), node("loop", vec![leaf("c8"), inner])])]);
    let two = k.program(&nest(leaf("barrier"))).expect("renders");
    assert!(two.substitutions.is_empty(), "{two:?}");
    let three = k.program(&nest(node("loop", vec![leaf("c2"), leaf("barrier")]))).expect("renders");
    assert_eq!(three.substitutions.len(), 1, "{three:?}");
    assert_eq!(three.substitutions[0].0, vec![1, 1, 1]);
    assert!(three.substitutions[0].1.contains(&BUDGET.to_string()), "{three:?}");
    let wide = k.program(&node("main", vec![leaf("g8x8"), node("loop", vec![leaf("c128"), leaf("barrier")])])).expect("renders");
    assert_eq!(wide.substitutions.len(), 1, "{wide:?}");
    assert!(!wide.text.contains("for ("), "{}", wide.text);
}

#[test]
fn vector_widths_are_fitted() {
    let k = kingdom();
    let s = |n: &str| k.symbol_named(n).unwrap_or_else(|| panic!("{n}"));
    let leaf = |n: &str| Tree { symbol: s(n), const_index: None, kids: vec![] };
    let node = |n: &str, kids: Vec<Tree>| Tree { symbol: s(n), const_index: None, kids };
    let v2 = node("v2", vec![leaf("rf0"), leaf("rf1")]);
    let v4 = node("v4", vec![leaf("rf0"), leaf("rf1"), leaf("rf2"), leaf("rf3")]);
    let kernel = |a: Tree, b: Tree| node("main", vec![leaf("g8x8"), node("set_rf0", vec![node("dot", vec![a, b])])]);
    let widen = k.program(&kernel(v4.clone(), v2.clone())).expect("renders");
    assert!(widen.text.contains("dot(vec4<f32>(rf0, rf1, rf2, rf3), vec4<f32>(vec2<f32>(rf0, rf1), 0.0, 0.0))"), "{}", widen.text);
    let narrow = k.program(&kernel(v2, v4)).expect("renders");
    assert!(narrow.text.contains("dot(vec2<f32>(rf0, rf1), (vec4<f32>(rf0, rf1, rf2, rf3)).xy)"), "{}", narrow.text);
}

#[cfg(feature = "wgsl")]
#[test]
fn the_true_kingdom_built_from_the_rows_renders_through_the_decoder() {
    use crate::wgsl::true_kingdom::TrueKingdom;
    let rows = kingdom();
    let k = TrueKingdom::from_rows(rows.clone()).expect("the bridge");
    let contract = k.relations().check().expect("the contract checks");
    assert_eq!(contract.n_symbols(), rows.contract.n_symbols());
    assert_eq!(k.types.len(), 9);
    assert_eq!(k.const_lens(), rows.const_len());
    let layout = rows.layout(PROGRAM_SIZE).expect("the layout");
    let mut rng = GeneRng::new(7);
    for _ in 0..20 {
        let gene = rows.random_gene(&layout, &mut rng);
        let tree = rows.express(&layout, &gene).expect("expresses");
        assert_eq!(k.render(&tree).expect("renders"), rows.program(&tree).expect("renders").text);
        let bound = k.tree_work_bound(&tree).expect("the work bound");
        assert!(bound >= 1 && bound <= (tree.level_order().len() as u128) * BUDGET as u128, "{bound}");
    }
}
