//! Tests of the regex kingdom as rows: the table is the generator's and passes the check; every random gene expresses
//! and emits a level-order tree that agrees with the rows node for node; the one guard is compiled into the device's
//! form and a row outside that form is a bug; the host decode through the rows equals the device's rule, gene for gene.

use super::*;
use crate::typed_gep::{express, Domain, GeneLayout};

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n as u64) as usize
    }
}

fn kingdom(lit: bool) -> KingdomContract {
    typed_regex_relations(lit).check().expect("the typed regex kingdom passes the check")
}

fn random_gene(k: &KingdomContract, layout: &GeneLayout, rng: &mut Rng) -> Vec<u32> {
    let cl = const_len(k);
    let mut gene = vec![0u32; layout.width];
    for s in layout.spans() {
        for cell in &mut gene[s.off..s.off + s.len] {
            *cell = match s.domain {
                Domain::Head => {
                    let a: Vec<SymbolId> = (0..k.n_symbols() as SymbolId).filter(|x| k.output(*x) == s.ty).collect();
                    a[rng.below(a.len())]
                }
                Domain::Tail => {
                    let ts = k.terminals_of(s.ty);
                    ts[rng.below(ts.len())]
                }
                Domain::Dc => rng.below(cl[&s.ty] as usize) as u32,
                Domain::Dv => rng.below(crate::typed_gep::DV_WORDS as usize) as u32,
            };
        }
    }
    gene
}

fn typed(e: &crate::typed_gep::Expressed) -> Vec<TypedNode> {
    e.nodes.iter().map(|x| TypedNode { symbol: x.symbol, parent_step: x.parent_step, arg_index: x.arg_index, const_index: x.const_index }).collect()
}

/// The kingdom's committed master rows are what the generator writes.
#[test]
fn the_table_is_the_generators() {
    let out = std::process::Command::new("python3").args(["-I", "kingdoms/regex/gen_tables.py", "--check"]).current_dir(env!("CARGO_MANIFEST_DIR")).output().expect("python3 runs");
    assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
}

#[test]
fn both_variants_load_and_pass_the_kingdom_check() {
    for lit in [false, true] {
        let rows = RegexRows::load(lit).expect("the rows load");
        let k = &rows.contract;
        assert_eq!(k.roots(), [Ty::Pattern]);
        assert_eq!(k.constant_capable(Ty::Pattern), lit);
        assert_eq!(rows.symbol(PATTERN_LITERAL).is_some(), lit);
        for t in [Ty::Char, Ty::I] {
            assert!(k.constant_capable(t), "{t:?}");
        }
        let cl = const_len(k);
        assert_eq!(cl.get(&Ty::Char), Some(&CHAR_POOL_LEN));
        assert_eq!(cl.get(&Ty::I), Some(&COUNT_POOL_LEN));
        assert_eq!(cl.contains_key(&Ty::Pattern), lit);
        assert!(rows.tables.flows.is_empty(), "a regex carries no property: no Flow row");
        assert_eq!(rows.tables.guards.len(), 1);
        EmitTables::new(&rows).expect("every symbol has an opcode");
        // The alphabet: every output Pattern or CharClass, every input one of the four data types, and the three
        // repetitions take a class and counts.
        for s in 0..k.n_symbols() as SymbolId {
            assert!(matches!(k.output(s), Ty::Pattern | Ty::CharClass | Ty::Char | Ty::I), "{}", k.name(s));
            assert!(k.is_constant(s) || matches!(k.output(s), Ty::Pattern | Ty::CharClass), "{} is a function of {:?}", k.name(s), k.output(s));
        }
        for (name, sig) in [("regex_rep_n", &[Ty::CharClass, Ty::I][..]), ("regex_rep_upto", &[Ty::CharClass, Ty::I]), (REP_RANGE, &[Ty::CharClass, Ty::I, Ty::I])] {
            assert_eq!(k.signature(rows.symbol(name).expect(name)), sig, "{name}");
        }
    }
}

/// The compiler's buffers are sized from the layout's bound: the bound grows with N and no program size is refused.
#[test]
fn the_layout_bound_sizes_the_compiler() {
    for lit in [false, true] {
        let k = kingdom(lit);
        let cap = |n: usize| capacity(&k, &k.heads_from_program_size(Ty::Pattern, n).expect("heads")).expect("capacity");
        assert_eq!(cap(12), Capacity { nodes: 158, insts: 393 });
        assert_eq!(cap(28), Capacity { nodes: 382, insts: 937 });
        assert!(cap(100).nodes > cap(50).nodes && cap(100).insts > cap(50).insts);
    }
}

#[test]
fn regex_bound_table() {
    for lit in [false, true] {
        let k = kingdom(lit);
        for n in [8usize, 16, 28, 40] {
            let h = k.heads_from_program_size(Ty::Pattern, n).expect("heads");
            let b = k.bounds(Ty::Pattern, &h).expect("bounds");
            // a_Pattern = 2 (concat), a_CharClass = 2 (union), a_Char = 2 (range), a_I = 2 (the range's bounds).
            assert_eq!((h[&Ty::Pattern], h[&Ty::CharClass], h[&Ty::Char], h[&Ty::I]), (2 * n - 1, 2 * (n - 1), 2 * (n - 1), 2 * (n - 1)));
            assert_eq!(k.guaranteed_program_size(Ty::Pattern, &h), n);
            let mut line = format!("pattern_literal={lit} N={n} L={} width={}", b.l, b.width());
            for t in &b.types {
                line.push_str(&format!(" | {:?} h{} P{} t{} d{}{}", t.ty, t.h, t.p, t.t, t.d, if t.constant { "" } else { "-" }));
            }
            println!("{line}");
        }
    }
}

/// Every random gene expresses, and emission gives a level-order tree whose ops, arities, types and constants agree
/// with the kingdom, node for node; every emitted range has its bounds in order.
#[test]
fn emission_matches_the_kingdom_on_random_genes() {
    let mut rng = Rng(7013);
    for lit in [false, true] {
        let rows = RegexRows::load(lit).expect("rows");
        let k = &rows.contract;
        let t = EmitTables::new(&rows).expect("tables");
        let cl = const_len(k);
        for n in [1usize, 3, 8, 12] {
            let layout = rows.layout(n).expect("layout");
            for _ in 0..2000 {
                let gene = random_gene(k, &layout, &mut rng);
                let e = express(k, &layout, &gene, &cl).expect("a valid-by-construction gene expresses");
                let cap = capacity(k, &k.heads_from_program_size(Ty::Pattern, n).expect("heads")).expect("capacity");
                assert!(e.nodes.len() <= cap.nodes, "a layout of size {n} has at most {} nodes", cap.nodes);
                let g = emit_nodes(&t, &typed(&e)).expect("emits");
                assert_eq!(g.len(), e.nodes.len());
                assert_eq!(g[0].ty_code, Ty::Pattern.code());
                for (i, (node, x)) in g.iter().zip(&e.nodes).enumerate() {
                    let arity = k.signature(x.symbol).len();
                    assert_eq!(node.op, op_of_name(k.name(x.symbol)).map(|o| o as u32).expect("op"));
                    assert_eq!(node.ty_code, x.ty.code());
                    for a in 0..arity {
                        let c = node.child(a);
                        assert!(c > i && c < g.len(), "child {a} of node {i} is forward and in range");
                        assert_eq!(e.nodes[c].parent_step, Some(i + 1));
                        assert_eq!(e.nodes[c].arg_index, Some(a as u32));
                        assert_eq!(e.nodes[c].ty, k.signature(x.symbol)[a], "the child fills the slot's type");
                    }
                    if node.op == Op::RegexRepRange as u32 {
                        assert!(g[node.child(1)].konst <= g[node.child(2)].konst, "a range's bounds are in order");
                    } else if k.is_constant(x.symbol) {
                        let own = pool_value(x.ty, x.const_index.expect("index")).expect("in pool");
                        // A bound of a range may have been exchanged with the other bound.
                        let parent = x.parent_step.map(|p| &e.nodes[p - 1]);
                        let bound_of_range = parent.is_some_and(|p| k.name(p.symbol) == REP_RANGE) && x.arg_index != Some(0);
                        if bound_of_range {
                            let p = parent.expect("parent");
                            let bounds: Vec<u32> = e.nodes.iter().filter(|y| y.parent_step == x.parent_step && y.arg_index != Some(0)).map(|y| pool_value(y.ty, y.const_index.expect("index")).expect("in pool")).collect();
                            assert!(bounds.contains(&(node.konst as u32)), "{}: a bound is one of the range's two", k.name(p.symbol));
                        } else {
                            assert_eq!(node.konst as u32, own);
                        }
                    }
                }
            }
        }
    }
}

/// THE SMOKE TEST (plan 7.7, K2): a thousand random genes, the host decode through the rows (the formula engine deciding
/// the guard over the leaves' pool values, the repair as the rows say) equals the device's rule, node for node; the
/// guard fires on some of them.
#[test]
fn a_thousand_random_genes_decode_through_the_rows_equal_to_the_device_rule() {
    let mut rng = Rng(2026);
    let (mut genes, mut fired) = (0usize, 0usize);
    for lit in [false, true] {
        let rows = RegexRows::load(lit).expect("rows");
        let t = EmitTables::new(&rows).expect("tables");
        let cl = const_len(&rows.contract);
        for n in [3usize, 8] {
            let layout = rows.layout(n).expect("layout");
            for _ in 0..250 {
                let gene = random_gene(&rows.contract, &layout, &mut rng);
                let e = express(&rows.contract, &layout, &gene, &cl).expect("expresses");
                let device = emit_nodes(&t, &typed(&e)).expect("the device's rule");
                let host = rows.decode_gene(&layout, &gene, &cl).expect("the host decode");
                let words = |g: &[GpuNode]| g.iter().map(|n| (n.op, n.arg0, n.ty_code, n.konst.to_bits())).collect::<Vec<_>>();
                assert_eq!(words(&host), words(&device), "lit {lit} N {n}: the host decode differs from the device's rule");
                let plain = emit(&t, &typed(&e), false).expect("unruled");
                fired += usize::from(words(&plain) != words(&device));
                genes += 1;
            }
        }
    }
    assert_eq!(genes, 1000);
    assert!(fired > 20, "the guard must fire on some genes: {fired} of {genes}");
    println!("regex rows: {genes} genes, host decode = device rule on all; the range guard fired on {fired}");
}

#[test]
fn the_device_rules_are_the_rows_guard() {
    let rows = RegexRows::load(true).expect("rows");
    let rules = rows.device_rules().expect("rules");
    let range = rows.symbol(REP_RANGE).expect("the range");
    for (s, r) in rules.iter().enumerate() {
        assert_eq!(r.swap, (s as SymbolId == range).then_some((1, 2)), "{}", rows.contract.name(s as SymbolId));
        assert_eq!(DeviceRule::from_word(r.word()), *r);
    }
    let t = EmitTables::new(&rows).expect("tables");
    assert_eq!(t.sym_rule.iter().filter(|w| **w != 0).count(), 1);
}

/// A Guard row outside the device's form is a bug at load, never a silent approximation.
#[test]
fn a_guard_outside_the_device_form_is_a_bug() {
    let text = table_text(true).expect("the master has the regex kingdom");
    let guard = "guard\tregex_rep_range\t0\tmust\tin1.value <= in2.value";
    let repair = "repair\tregex_rep_range\t0\tnode\ttrue\ttrue\t(?:{0}){{{=in2.value},{=in1.value}}}";
    assert!(text.contains(guard) && text.contains(repair));
    for (from, to, why) in [
        (guard, "guard\tregex_rep_range\t0\tmust\tin1.value < in2.value", "a strict comparison"),
        (guard, "guard\tregex_rep_range\t0\tadvise\tin1.value <= in2.value", "an advise guard"),
        (repair, "repair\tregex_rep_range\t0\tnode\ttrue\ttrue\t(?:{0}){{{=in1.value},{=in2.value}}}", "a repair that does not exchange the bounds"),
        (repair, "repair\tregex_rep_range\t0\tnode\ttrue\ttrue\t{in0}", "a substitute"),
    ] {
        let planted = text.replace(from, to);
        let e = RegexRows::from_text(true, &planted).err().unwrap_or_else(|| panic!("{why} must not load"));
        assert!(e.contains("our software has a bug"), "{why}: {e}");
    }
    // A guard on two bounds of a data type whose symbols are not all constants: the device cannot read a value.
    let planted = text.replace(guard, "guard\tregex_rep_range\t0\tmust\tin0.value <= in2.value").replace("carries\tI\tvalue", "carries\tI\tvalue\ncarries\tCharClass\tvalue");
    let Err(e) = RegexRows::from_text(true, &planted) else { panic!("a class bound must not load") };
    assert!(e.contains("our software has a bug"), "{e}");
}

#[test]
fn a_broken_walk_is_a_defect_not_a_repair() {
    let rows = RegexRows::load(true).expect("rows");
    let t = EmitTables::new(&rows).expect("tables");
    let concat = rows.symbol("regex_concat").expect("concat");
    let dot = rows.symbol("regex_dot").expect("dot");
    let pat = rows.symbol(PATTERN_LITERAL).expect("?_Pat");
    let n = |symbol, parent_step, arg_index, const_index| TypedNode { symbol, parent_step, arg_index, const_index };
    assert!(matches!(emit_nodes(&t, &[]), Err(EmitDefect::Empty)));
    // A placeholder with no pool index.
    assert!(matches!(emit_nodes(&t, &[n(pat, None, None, None)]), Err(EmitDefect::BadConstant { .. })));
    // A placeholder index past the pool.
    assert!(matches!(emit_nodes(&t, &[n(pat, None, None, Some(CHAR_POOL_LEN))]), Err(EmitDefect::BadConstant { .. })));
    // concat(dot, dot) with the second child skipped to argument 2.
    let bad = [n(concat, None, None, None), n(dot, Some(1), Some(0), None), n(dot, Some(1), Some(2), None)];
    assert!(matches!(emit_nodes(&t, &bad), Err(EmitDefect::ChildrenNotContiguous { .. })));
    // A second root.
    assert!(matches!(emit_nodes(&t, &[n(dot, None, None, None), n(dot, None, None, None)]), Err(EmitDefect::NotARootedTree { .. })));
    let ok = [n(concat, None, None, None), n(dot, Some(1), Some(0), None), n(pat, Some(1), Some(1), Some(char_index(b'A').expect("A")))];
    let g = emit_nodes(&t, &ok).expect("emits");
    assert_eq!((g[0].op, g[0].arg0, g[1].op, g[2].op, g[2].konst as u32), (Op::RegexConcat as u32, 1, Op::RegexDot as u32, Op::Num as u32, 65));
    // A range whose bounds are in the wrong order comes out in order; one in order is untouched.
    let range = rows.symbol(REP_RANGE).expect("range");
    let digit = rows.symbol("regex_cc_digit").expect("digit");
    let count = rows.symbol("?_I").expect("?_I");
    let r = |lo, hi| [n(range, None, None, None), n(digit, Some(1), Some(0), None), n(count, Some(1), Some(1), Some(lo)), n(count, Some(1), Some(2), Some(hi))];
    let g = emit_nodes(&t, &r(5, 3)).expect("emits");
    assert_eq!((g[0].op, g[2].konst, g[3].konst), (Op::RegexRepRange as u32, 3.0, 5.0));
    let g = emit_nodes(&t, &r(2, 7)).expect("emits");
    assert_eq!((g[2].konst, g[3].konst), (2.0, 7.0));
}
