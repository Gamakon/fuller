//! Tests of the TSR kingdom as rows: the Flow rows give phylu's stream order and today's depth; each variant expresses
//! exactly the programs today's typed tables make legal; nothing illegal is expressed; the layout is the single
//! kingdom's; the tables are the generator's.

use std::collections::{BTreeMap, HashMap};

use super::*;
use crate::geneframe::PropertyExpr;
use crate::kingdom_contract::tsr_relations;
use crate::tsr_kingdom::{TsrKind, TsrKingdom};
use crate::typed_environment::Failure;
use crate::typed_gep::{express, Domain};

const N_INPUTS: usize = 4;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, p: f64) -> bool {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64) < p
    }
}

// ---- THE GOLDEN: phylu's definitions, copied (fuller must not depend on phylu) ----------------------------------
//
// `stream_order` is phylu `src/evolve/order.rs` `stream_order` (phylu bf41f2f4) and `t_depth` is phylu
// `src/evolve/engine.rs` `t_depth`, line for line, over a level-order node array whose children are contiguous from
// `arg0` (`GpuNode::child`). `family` is order.rs's; `RAISERS` is engine.rs `DEPTH_RAISING_OPS` within `wide()`'s
// alphabet (no `Pow`, so t_depth's whole-exponent clause never applies).

/// One node of the golden's array: its `Math` constructor (`Var`, `Num` for a leaf) and its first child's index.
#[derive(Debug, Clone, Copy)]
struct GoldNode {
    op: &'static str,
    arg0: usize,
}

impl GoldNode {
    fn child(&self, j: usize) -> usize {
        self.arg0 + j
    }
}

const BINARY: [&str; 4] = ["Add", "Sub", "Mul", "ProtectedDiv"];
const RAISERS: [&str; 10] = ["Abs", "ProtectedSqrt", "ProtectedLog", "ProtectedExp", "Sin", "Cos", "Tan", "Tanh", "ProtectedAsin", "ProtectedAcos"];
/// order.rs `RULES` upper bounds (argument order at most 2), `Inv` read as `ProtectedInv`.
const WINDOWED: [&str; 7] = ["Pow2", "Pow3", "ProtectedInv", "Sin", "Cos", "Tan", "Tanh"];

fn arity_of(op: &str) -> usize {
    if op == "Var" || op == "Num" {
        0
    } else if BINARY.contains(&op) {
        2
    } else {
        1
    }
}

fn family(op: &str) -> Option<u8> {
    if op == "Add" || op == "Sub" {
        Some(0)
    } else if op == "Mul" || op == "Div" || op == "ProtectedDiv" {
        Some(1)
    } else {
        None
    }
}

fn stream_order(nodes: &[GoldNode]) -> Vec<u32> {
    let n = nodes.len();
    let mut best = vec![1u32; n];
    let mut ties = vec![0u32; n];
    let mut order = vec![1u32; n];
    for i in (0..n).rev() {
        let op = nodes[i].op;
        let a = arity_of(op);
        if a == 0 {
            order[i] = 1;
            continue;
        }
        let fam = family(op);
        let (mut m, mut c) = (0u32, 0u32);
        let mut arm = |o: u32, k: u32| {
            if o > m {
                m = o;
                c = k;
            } else if o == m {
                c += k;
            }
        };
        let kids: Vec<usize> = (0..a).map(|j| nodes[i].child(j)).collect();
        for &k in &kids {
            if k >= n {
                continue;
            }
            if fam.is_some() && family(nodes[k].op) == fam {
                arm(best[k], ties[k]);
            } else {
                arm(order[k], 1);
            }
        }
        best[i] = m;
        ties[i] = c;
        order[i] = m + u32::from(c >= 2);
    }
    order
}

fn t_depth(nodes: &[GoldNode]) -> u32 {
    let counts = |i: usize| -> u32 { u32::from(RAISERS.contains(&nodes[i].op)) };
    if nodes.is_empty() {
        return 0;
    }
    let mut depth = vec![0u32; nodes.len()];
    depth[0] = counts(0);
    for i in 0..nodes.len() {
        let n = &nodes[i];
        for k in (0..arity_of(n.op)).map(|c| n.child(c)).filter(|&k| k < nodes.len()) {
            depth[k] = depth[i] + counts(k);
        }
    }
    depth.into_iter().max().unwrap_or(0)
}

/// The windows hold: every windowed op's argument has order at most 2.
fn windows_hold(nodes: &[GoldNode]) -> bool {
    let order = stream_order(nodes);
    nodes.iter().all(|n| !WINDOWED.contains(&n.op) || order[n.arg0] <= 2)
}

// ---- trees -------------------------------------------------------------------------------------------------------

/// The `Math` constructor of a symbol.
fn ctor(k: &TsrRows, s: SymbolId) -> &'static str {
    match k.symbol(s).expect("a symbol of the alphabet") {
        TsrSymbol::Operator(c) => {
            let all = BINARY.iter().chain(RAISERS.iter()).chain(["Neg", "Pow2", "Pow3", "ProtectedInv"].iter());
            all.copied().find(|x| *x == c).unwrap_or_else(|| panic!("{c} is not an operator of wide()"))
        }
        TsrSymbol::Input(_) => "Var",
        TsrSymbol::Constant => "Num",
    }
}

/// A tree as the golden's level-order array (math_to_nodes: breadth first, children contiguous), with each node's
/// path from the root in the same order.
fn flatten(k: &TsrRows, t: &Tree) -> (Vec<GoldNode>, Vec<Vec<u32>>) {
    let mut nodes: Vec<GoldNode> = Vec::new();
    let mut paths: Vec<Vec<u32>> = Vec::new();
    let mut queue: std::collections::VecDeque<(&Tree, Vec<u32>, usize)> = std::collections::VecDeque::from([(t, Vec::new(), 0)]);
    nodes.push(GoldNode { op: "Var", arg0: 0 });
    paths.push(Vec::new());
    while let Some((t, path, slot)) = queue.pop_front() {
        let first = nodes.len();
        nodes[slot] = GoldNode { op: ctor(k, t.symbol), arg0: if t.kids.is_empty() { 0 } else { first } };
        paths[slot] = path.clone();
        for (j, kid) in t.kids.iter().enumerate() {
            nodes.push(GoldNode { op: "Var", arg0: 0 });
            paths.push(Vec::new());
            let mut p = path.clone();
            p.push(j as u32);
            queue.push_back((kid, p, first + j));
        }
    }
    (nodes, paths)
}

fn subtree<'a>(t: &'a Tree, path: &[u32]) -> &'a Tree {
    path.iter().fold(t, |t, j| &t.kids[*j as usize])
}

fn named(k: &TsrRows, name: &str) -> SymbolId {
    k.decoder.symbol_named(name).unwrap_or_else(|| panic!("no symbol {name}"))
}

/// A random tree of `n` nodes over the whole alphabet, split often near the middle so order 4 and 5 are reached
/// (check_types.py `random_tree`); a leaf is an input column or `?`.
fn random_tree(k: &TsrRows, rng: &mut Rng, n: usize) -> Tree {
    let leaf = |rng: &mut Rng| {
        let i = rng.below(k.n_inputs + 1);
        if i < k.n_inputs {
            Tree { symbol: named(k, &format!("x{i}")), const_index: None, kids: Vec::new() }
        } else {
            Tree { symbol: named(k, "?"), const_index: Some(0), kids: Vec::new() }
        }
    };
    let unary = ["sin", "cos", "tanh", "psqrt", "plog", "pexp", "neg", "abs", "pow2", "pow3", "pinv", "tan", "pasin", "pacos"];
    let binary = ["add", "sub", "mul", "pdiv"];
    if n == 1 {
        return leaf(rng);
    }
    if n == 2 || rng.chance(0.3) {
        let op = named(k, unary[rng.below(unary.len())]);
        return Tree { symbol: op, const_index: None, kids: vec![random_tree(k, rng, n - 1)] };
    }
    let rest = n - 1;
    let left = if rng.chance(0.5) { (rest / 2 + rng.below(3)).saturating_sub(1).clamp(1, rest - 1) } else { 1 + rng.below(rest - 1) };
    let op = named(k, binary[rng.below(binary.len())]);
    Tree { symbol: op, const_index: None, kids: vec![random_tree(k, rng, left), random_tree(k, rng, rest - left)] }
}

/// Every tree of exactly `n` nodes over `unary` and `binary`, its one leaf `x0`.
fn all_trees(k: &TsrRows, n: usize, unary: &[&str], binary: &[&str], memo: &mut BTreeMap<usize, Vec<Tree>>) -> Vec<Tree> {
    if let Some(v) = memo.get(&n) {
        return v.clone();
    }
    let mut out = Vec::new();
    if n == 1 {
        out.push(Tree { symbol: named(k, "x0"), const_index: None, kids: Vec::new() });
    } else {
        for c in all_trees(k, n - 1, unary, binary, memo) {
            out.extend(unary.iter().map(|u| Tree { symbol: named(k, u), const_index: None, kids: vec![c.clone()] }));
        }
        for i in 1..n - 1 {
            let (ls, rs) = (all_trees(k, i, unary, binary, memo), all_trees(k, n - 1 - i, unary, binary, memo));
            for b in binary {
                for l in &ls {
                    out.extend(rs.iter().map(|r| Tree { symbol: named(k, b), const_index: None, kids: vec![l.clone(), r.clone()] }));
                }
            }
        }
    }
    memo.insert(n, out.clone());
    out
}

fn literal(e: Option<&PropertyExpr>) -> u32 {
    match e {
        Some(PropertyExpr::Literal(v)) if *v >= 0 => *v as u32,
        other => panic!("a property is {other:?}, not a literal"),
    }
}

/// Every node of the decoded tree has the golden's stream order and depth, in a kingdom with no guard.
fn assert_golden(k: &TsrRows, t: &Tree) {
    let d = k.decode(t).expect("decodes");
    let (nodes, paths) = flatten(k, t);
    let order = stream_order(&nodes);
    for (i, path) in paths.iter().enumerate() {
        let n = d.node(path).expect("a decoded node at every path");
        assert_eq!(literal(n.properties.get("order")), order[i], "order at {path:?} of {t:?}");
        let (sub, _) = flatten(k, subtree(t, path));
        assert_eq!(literal(n.properties.get("depth")), t_depth(&sub), "depth at {path:?} of {t:?}");
    }
}

// ---- the flows ---------------------------------------------------------------------------------------------------

/// The Flow rows give EXACTLY phylu's `stream_order` (at every node) and `t_depth` (of every subtree): exhaustively over
/// every tree of up to 5 nodes on all 18 operators (46,279), and over 100,000 random trees of up to 64 nodes (root
/// orders 1 to 5).
#[test]
fn the_flows_give_phylus_stream_order_and_todays_depth() {
    let k = TsrRows::load(TsrVariant::Single, N_INPUTS).unwrap();
    let unary = ["sin", "cos", "tanh", "psqrt", "plog", "pexp", "neg", "abs", "pow2", "pow3", "pinv", "tan", "pasin", "pacos"];
    let mut memo = BTreeMap::new();
    let mut exhaustive = 0usize;
    for n in 1..=5 {
        for t in all_trees(&k, n, &unary, &["add", "sub", "mul", "pdiv"], &mut memo) {
            assert_golden(&k, &t);
            exhaustive += 1;
        }
    }
    assert_eq!(exhaustive, 1 + 14 + 200 + 2912 + 43_152, "every tree of up to 5 nodes");
    let mut rng = Rng(20261105);
    let mut orders: BTreeMap<u32, usize> = BTreeMap::new();
    for _ in 0..100_000 {
        let n = 1 + rng.below(64);
        let t = random_tree(&k, &mut rng, n);
        assert_golden(&k, &t);
        *orders.entry(stream_order(&flatten(&k, &t).0)[0]).or_default() += 1;
    }
    assert!(orders.get(&4).copied().unwrap_or(0) > 1_000 && orders.contains_key(&5), "the random trees reach orders 4 and 5: {orders:?}");
    eprintln!("stream order of the 100,000 random trees' roots: {orders:?}");
}

// ---- today's typed tables ----------------------------------------------------------------------------------------

/// Today's typed tables of a variant (`types.tsv`, `functions.tsv`), as a bottom-up typer: a subtree's value is the
/// set of duals it can be given (a bitmask; a leaf fits every dual but ROOT), and a whole tree is legal when a row (a
/// root twin, or in `single` the one dual) takes it at the root.
struct Typed {
    rows: HashMap<String, Vec<(Vec<u32>, u32)>>,
    twins: HashMap<String, Vec<Vec<u32>>>,
    leaf: u64,
    root: u32,
    single: bool,
    memo: HashMap<(String, Vec<u64>), u64>,
}

impl Typed {
    fn new(variant: TsrVariant) -> Typed {
        let old = TsrKingdom::load(variant, N_INPUTS).unwrap();
        assert!(old.duals.len() <= 64);
        let ix = |ty: Ty| match ty {
            Ty::Tsr(i) => u32::from(i),
            other => panic!("{other:?}"),
        };
        let root = ix(old.root().unwrap());
        let mut rows: HashMap<String, Vec<(Vec<u32>, u32)>> = HashMap::new();
        let mut twins: HashMap<String, Vec<Vec<u32>>> = HashMap::new();
        for r in &old.rows {
            let ins: Vec<u32> = r.inputs.iter().map(|t| ix(*t)).collect();
            match r.kind {
                TsrKind::Function => rows.entry(r.semantic_id.clone()).or_default().push((ins, ix(r.output))),
                TsrKind::RootTwin => twins.entry(r.semantic_id.clone()).or_default().push(ins),
                TsrKind::Var(_) | TsrKind::Const => {}
            }
        }
        let single = variant == TsrVariant::Single;
        let leaf = (0..old.duals.len() as u32).filter(|i| single || *i != root).fold(0u64, |m, i| m | (1 << i));
        Typed { rows, twins, leaf, root, single, memo: HashMap::new() }
    }

    fn set(&mut self, k: &TsrRows, t: &Tree) -> u64 {
        if t.kids.is_empty() {
            return self.leaf;
        }
        let kids: Vec<u64> = t.kids.iter().map(|c| self.set(k, c)).collect();
        let name = k.contract.name(t.symbol).to_string();
        if let Some(v) = self.memo.get(&(name.clone(), kids.clone())) {
            return *v;
        }
        let fits = |ins: &[u32]| ins.iter().zip(&kids).all(|(a, s)| s & (1 << a) != 0);
        let v = self.rows.get(&name).into_iter().flatten().filter(|(ins, _)| fits(ins)).fold(0u64, |m, (_, out)| m | (1 << out));
        self.memo.insert((name, kids), v);
        v
    }

    fn legal(&mut self, k: &TsrRows, t: &Tree) -> bool {
        if t.kids.is_empty() {
            return true;
        }
        if self.single {
            return self.set(k, t) & (1 << self.root) != 0;
        }
        let kids: Vec<u64> = t.kids.iter().map(|c| self.set(k, c)).collect();
        let name = k.contract.name(t.symbol);
        self.twins.get(name).into_iter().flatten().any(|ins| ins.iter().zip(&kids).all(|(a, s)| s & (1 << a) != 0))
    }
}

/// A gene valid by layout: each head cell any symbol of its type, each tail cell a terminal, each Dc word below its
/// pool's length.
fn random_gene(c: &KingdomContract, layout: &GeneLayout, consts: &BTreeMap<Ty, u32>, rng: &mut Rng) -> Vec<u32> {
    let mut gene = vec![0u32; layout.width];
    for s in layout.spans() {
        let (fs, ts) = (c.functions_of(s.ty), c.terminals_of(s.ty));
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

/// The variant's limits by the golden: (order ceiling, depth limit, windows).
fn limits(v: TsrVariant) -> (Option<u32>, Option<u32>, bool) {
    match v {
        TsrVariant::Single => (None, None, false),
        TsrVariant::Order => (Some(3), None, false),
        TsrVariant::Depth => (None, Some(2), false),
        TsrVariant::OrderDepth => (Some(3), Some(2), false),
        TsrVariant::OrderDepthWindows => (Some(3), Some(2), true),
    }
}

fn golden_legal(k: &TsrRows, t: &Tree) -> bool {
    let (nodes, _) = flatten(k, t);
    let (ceiling, depth, windows) = limits(k.variant);
    ceiling.is_none_or(|c| stream_order(&nodes)[0] <= c) && depth.is_none_or(|d| t_depth(&nodes) <= d) && (!windows || windows_hold(&nodes))
}

const GENES: usize = 10_000;
const TYPED_GENES: usize = 2_000;

/// EVERY VARIANT'S LEGAL PROGRAMS ARE TODAY'S TYPED TABLES' (plan 8.6 tests 1 and 2), both ways, on random genes:
///
/// - on 10,000 random 104-word genes a variant (head 34, 4 inputs): an expressed tree decodes with NO substitution
///   exactly when today's typed tables give it a typing, and exactly when the golden says it is within the limits;
///   and the program it expresses (substitutions applied) has a typing and is within the limits: NOTHING ILLEGAL;
/// - on 2,000 random genes of today's typed variant (program size 17): every expressed tree decodes with no
///   substitution.
#[test]
fn every_variant_expresses_exactly_todays_legal_programs_and_nothing_illegal() {
    for variant in TsrVariant::ALL {
        let d = variant.dir();
        let k = TsrRows::load(variant, N_INPUTS).unwrap();
        let mut typed = Typed::new(variant);
        let layout = k.layout(HEAD).unwrap();
        let consts = BTreeMap::from([(Ty::F, 1u32)]);
        let mut rng = Rng(7011 + variant as u64);
        let (mut clean, mut substituted, mut nodes_dropped) = (0usize, 0usize, 0usize);
        for _ in 0..GENES {
            let gene = random_gene(&k.contract, &layout, &consts, &mut rng);
            let tree = express(&k.contract, &layout, &gene, &consts).unwrap().tree();
            let dec = k.decode(&tree).unwrap();
            let failures = dec.failures();
            assert!(failures.iter().all(|f| matches!(f, Failure::Substituted { by: Substitution::Input(0), .. })), "{d}: every repair is {{in0}}: {failures:?}");
            let no_sub = failures.is_empty();
            assert_eq!(no_sub, typed.legal(&k, &tree), "{d}: substitution-free exactly when today's tables type it: {tree:?}");
            assert_eq!(no_sub, golden_legal(&k, &tree), "{d}: substitution-free exactly when within the golden's limits: {tree:?}");
            let p = program(&dec);
            assert!(typed.legal(&k, &p) && golden_legal(&k, &p), "{d}: the expressed program is illegal: {p:?}");
            assert!(k.decode(&p).unwrap().failures().is_empty(), "{d}: the program decodes again with no substitution");
            if no_sub {
                clean += 1;
                assert_eq!(p, tree, "{d}: no substitution, the program is the tree");
            } else {
                substituted += 1;
                nodes_dropped += dropped(&tree, &dec);
            }
        }
        if variant == TsrVariant::Single {
            assert_eq!(substituted, 0, "single has no guard");
        } else {
            assert!(clean > 0 && substituted > 0, "{d}: both legal and illegal trees drawn ({clean} clean, {substituted} substituted)");
        }
        eprintln!("{d}: {GENES} genes, {clean} clean, {substituted} substituted, {nodes_dropped} nodes dropped");

        // The other way: genes of today's typed variant.
        let old = tsr_relations(variant, N_INPUTS).unwrap().check().unwrap();
        let root = old.roots()[0];
        let old_layout = GeneLayout::from_program_size(&old, root, 17).unwrap();
        let old_consts: BTreeMap<Ty, u32> = old.types().iter().filter(|t| old.constant_capable(**t)).map(|t| (*t, 1)).collect();
        fn to_rows(old: &KingdomContract, k: &TsrRows, t: &Tree) -> Tree {
            let name = old.name(t.symbol);
            let base = name.split_once('@').map_or(name, |(a, _)| a);
            Tree { symbol: named(k, base), const_index: t.const_index, kids: t.kids.iter().map(|c| to_rows(old, k, c)).collect() }
        }
        for _ in 0..TYPED_GENES {
            let gene = random_gene(&old, &old_layout, &old_consts, &mut rng);
            let tree = to_rows(&old, &k, &express(&old, &old_layout, &gene, &old_consts).unwrap().tree());
            assert!(k.decode(&tree).unwrap().failures().is_empty(), "{d}: a typed gene's tree needs a substitution: {tree:?}");
            assert!(golden_legal(&k, &tree), "{d}: a typed gene's tree is outside the golden's limits");
        }
    }
}

// ---- the rows, the layout, the generator -------------------------------------------------------------------------

/// One data type, the alphabet in `wide()`'s order, and each variant's Guard rows: none in single; the four binaries
/// (order); the ten raising unaries (depth); and the seven windowed unaries (windows), every one `must` with the one
/// repair `{in0}`.
#[test]
fn each_variant_is_the_same_rows_but_its_guards() {
    use crate::geneframe::Strength;
    let single = TsrRows::load(TsrVariant::Single, N_INPUTS).unwrap();
    for (variant, guards) in [(TsrVariant::Single, 0), (TsrVariant::Order, 4), (TsrVariant::Depth, 10), (TsrVariant::OrderDepth, 14), (TsrVariant::OrderDepthWindows, 21)] {
        let k = TsrRows::load(variant, N_INPUTS).unwrap();
        assert_eq!(k.tables.data_types.len(), 1);
        assert_eq!(k.contract.types(), &[Ty::F]);
        assert_eq!((k.tables.symbols.clone(), k.tables.arguments.clone(), k.tables.flows.clone()), (single.tables.symbols.clone(), single.tables.arguments.clone(), single.tables.flows.clone()), "{}: only the guards differ", variant.dir());
        assert_eq!(k.tables.guards.len(), guards, "{}", variant.dir());
        for g in k.tables.guards.iter() {
            assert_eq!(g.strength, Strength::Must);
            assert_eq!(g.repairs.iter().map(|r| r.template.as_str()).collect::<Vec<_>>(), vec!["{in0}"]);
        }
        let names: Vec<&str> = (0..k.contract.n_symbols() as SymbolId).map(|s| k.contract.name(s)).collect();
        let wide = ["add", "sub", "mul", "pdiv", "sin", "cos", "tanh", "psqrt", "plog", "pexp", "neg", "abs", "pow2", "pow3", "pinv", "tan", "pasin", "pacos", "x0", "x1", "x2", "x3", "?"];
        assert_eq!(names, wide, "wide(4)'s order");
    }
    for n in 1..=MAX_INPUTS {
        let k = TsrRows::load(TsrVariant::OrderDepthWindows, n).unwrap();
        assert_eq!(k.contract.n_symbols(), 18 + n + 1);
        assert_eq!(k.symbol(18 + n as SymbolId), Some(TsrSymbol::Constant));
        assert_eq!(k.symbol(18 + n as SymbolId - 1), Some(TsrSymbol::Input(n - 1)));
    }
    assert!(TsrRows::load(TsrVariant::Single, 0).is_err() && TsrRows::load(TsrVariant::Single, MAX_INPUTS + 1).is_err());
}

/// The layout is the single kingdom's: one frame `F`, at head 34 a tail of 35, a Dc of 35 and no Dv lane: 104 words
/// a gene; and at program size 17 the width of today's typed `single` variant (101).
#[test]
fn the_gene_is_one_type_and_104_words_at_todays_head() {
    for variant in TsrVariant::ALL {
        let k = TsrRows::load(variant, N_INPUTS).unwrap();
        let l = k.layout(HEAD).unwrap();
        assert_eq!(l.frames.len(), 1);
        let f = &l.frames[0];
        assert_eq!((f.ty, f.head_len, f.tail_len, f.dc_len, f.dv_off), (Ty::F, 34, 35, 35, None));
        assert_eq!(l.width, 104);
        let at17 = GeneLayout::from_program_size(&k.contract, Ty::F, 17).unwrap();
        let old = tsr_relations(TsrVariant::Single, N_INPUTS).unwrap().check().unwrap();
        assert_eq!(at17.width, GeneLayout::from_program_size(&old, old.roots()[0], 17).unwrap().width);
        assert_eq!(at17.width, 101);
    }
}

/// A substituted node renders as its first input (pass 5), and its parent sees its input's properties: an order-4
/// product of two order-3 sums renders as its first sum, of order 3.
#[test]
fn a_substituted_node_renders_and_reads_as_its_first_input() {
    let k = TsrRows::load(TsrVariant::Order, N_INPUTS).unwrap();
    let leaf = |i: usize| Tree { symbol: named(&k, &format!("x{i}")), const_index: None, kids: Vec::new() };
    let node = |op: &str, kids: Vec<Tree>| Tree { symbol: named(&k, op), const_index: None, kids };
    let sum = |a, b, c, d| node("add", vec![node("mul", vec![leaf(a), leaf(b)]), node("mul", vec![leaf(c), leaf(d)])]);
    let tree = node("mul", vec![sum(0, 1, 2, 3), sum(0, 2, 1, 3)]);
    let dec = k.decode(&tree).unwrap();
    assert!(matches!(dec.failures().as_slice(), [Failure::Substituted { path, by: Substitution::Input(0), .. }] if path.is_empty()));
    assert_eq!(program(&dec), sum(0, 1, 2, 3));
    assert_eq!(literal(dec.root.properties.get("order")), 3);
    assert_eq!(literal(dec.root.properties.get("fam")), 1, "the substituted node takes its input's family");
    let text = k.decoder.render(&k.entry, &dec, &|_, w| Ok(format!("c{w}"))).unwrap();
    assert_eq!(text.body, "((x0 * x1) + (x2 * x3))");
    // Under single the same tree is expressed as written, order 4.
    let single = TsrRows::load(TsrVariant::Single, N_INPUTS).unwrap();
    assert_eq!(literal(single.decode(&tree).unwrap().root.properties.get("order")), 4);
    // The same gene decodes the same twice.
    assert_eq!(k.decode(&tree).unwrap(), dec);
}

/// The committed tables are what `gen_tables.py` writes (its `--check`). A missing `python3` is a failure, not a skip.
#[test]
fn the_committed_tables_are_what_the_generator_writes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = std::process::Command::new("python3").arg("-I").arg(root.join("kingdoms/tsr/gen_tables.py")).arg("--check").output().expect("python3 runs the generator's check");
    assert!(out.status.success(), "the generator's check failed:\n{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
}
