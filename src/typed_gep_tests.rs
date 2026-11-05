//! The design's host tests of multi-typed GEP (`MultiTypedGEP.md` section 7,
//! revision 4): closure over 100,000 random genes per kingdom, the per-producer
//! tail bound with its measured maximum and the constructions that attain it,
//! locality and homologous crossover, one-type equality with classical Karva,
//! and determinism.
//!
//! The genes are drawn valid by layout: a head cell is any symbol outputting
//! its domain's type, a tail cell a terminal of that type, a Dc cell an index
//! into `C_T`. Nothing here repairs a gene; a defect fails the test with its text.

use std::collections::BTreeMap;

use crate::geneframe::Ty;
use crate::kingdom_contract::{regex_relations, sr_relations, KingdomContract, KingdomRelations, SymbolId};
use crate::typed_gep::{express, Domain, Expressed, GeneLayout, Span};

const GENES: usize = 100_000;
/// Length of every `C_T` in these tests.
const CONSTS: u32 = 5;
/// Program sizes `N` the layouts are built for.
const SIZES: [usize; 4] = [1, 3, 8, 32];

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
}

fn const_lens(k: &KingdomContract) -> BTreeMap<Ty, u32> {
    k.types().iter().filter(|t| k.constant_capable(**t)).map(|t| (*t, CONSTS)).collect()
}

/// A gene valid by layout: each head cell drawn from the symbols of its
/// domain's type, each tail cell a terminal of its type, each Dc cell an index.
fn random_gene(k: &KingdomContract, layout: &GeneLayout, rng: &mut Rng) -> Vec<u32> {
    let mut gene = vec![0u32; layout.width];
    for s in layout.spans() {
        draw_span(k, &s, &mut gene, rng);
    }
    gene
}

fn draw_span(k: &KingdomContract, s: &Span, gene: &mut [u32], rng: &mut Rng) {
    let (functions, terminals) = (k.functions_of(s.ty), k.terminals_of(s.ty));
    for cell in &mut gene[s.off..s.off + s.len] {
        *cell = match s.domain {
            Domain::Head => {
                let i = rng.below(functions.len() + terminals.len());
                if i < functions.len() { functions[i] } else { terminals[i - functions.len()] }
            }
            Domain::Tail => {
                let ts = k.terminals_of(s.ty);
                ts[rng.below(ts.len())]
            }
            Domain::Dc => rng.below(CONSTS as usize) as u32,
            Domain::Dv => rng.below(crate::typed_gep::DV_WORDS as usize) as u32,
        };
    }
}

/// The tree check, written without reading the decoder's own bookkeeping:
/// every expressed node has the type its parent's argument slot requires, the
/// root has the gene's root type, and every function has each of its arguments
/// exactly once.
fn assert_well_typed(k: &KingdomContract, root: Ty, e: &Expressed, what: &str) {
    let n = &e.nodes;
    assert_eq!(n[0].ty, root, "{what}: root type");
    assert_eq!(k.output(n[0].symbol), root, "{what}: root symbol");
    let mut children: Vec<Vec<Option<u32>>> = vec![Vec::new(); n.len() + 1];
    for (i, node) in n.iter().enumerate() {
        assert_eq!(node.step, i + 1, "{what}: level order");
        assert_eq!(k.output(node.symbol), node.ty, "{what}: node {} symbol type", node.step);
        if let (Some(p), Some(a)) = (node.parent_step, node.arg_index) {
            assert!(p < node.step, "{what}: parent before child");
            let parent = &n[p - 1];
            assert_eq!(k.signature(parent.symbol)[a as usize], node.ty, "{what}: node {} fills a slot of another type", node.step);
            children[p].push(Some(a));
        }
    }
    for node in n {
        let mut got: Vec<u32> = children[node.step].iter().flatten().copied().collect();
        got.sort_unstable();
        let want: Vec<u32> = (0..k.signature(node.symbol).len() as u32).collect();
        assert_eq!(got, want, "{what}: node {} arguments", node.step);
    }
}

/// Largest frame, tail and Dc cursor seen per type.
#[derive(Default)]
struct Maxima {
    frame: BTreeMap<Ty, usize>,
    tail: BTreeMap<Ty, usize>,
    q: BTreeMap<Ty, usize>,
}

impl Maxima {
    fn take(&mut self, e: &Expressed) {
        for (src, dst) in [(&e.reach.frame_used, &mut self.frame), (&e.reach.tail_used, &mut self.tail), (&e.reach.q_used, &mut self.q)] {
            for (t, v) in src {
                let m = dst.entry(*t).or_default();
                *m = (*m).max(*v);
            }
        }
    }
    /// Every maximum is within its frame, tail and Dc.
    fn assert_below(&self, layout: &GeneLayout) {
        for (t, v) in &self.frame {
            let f = layout.frame(*t).expect("frame");
            assert!(*v <= f.head_len + f.tail_len, "frame of {t:?} reached {v}, h + t = {}", f.head_len + f.tail_len);
        }
        for (t, v) in &self.tail {
            let f = layout.frame(*t).expect("frame");
            assert!(*v <= f.tail_len, "tail of {t:?} reached {v}, t_T = {}", f.tail_len);
        }
        for (t, v) in &self.q {
            let f = layout.frame(*t).expect("frame");
            assert!(*v <= f.dc_len, "q of {t:?} reached {v}, d_T = {}", f.dc_len);
        }
    }
}

/// 100,000 random genes through the decoder over every program size and root.
fn closure_n(rel: &KingdomRelations, label: &str, sizes: &[usize], genes: usize) -> usize {
    let k = rel.check().unwrap_or_else(|d| panic!("{label} fails its kingdom check: {d:?}"));
    let consts = const_lens(&k);
    let mut rng = Rng(0x5eed ^ label.len() as u64);
    let mut worst = 0usize;
    let mut layouts: BTreeMap<(usize, Ty), (GeneLayout, Maxima)> = BTreeMap::new();
    for g in 0..genes {
        let n = sizes[g % sizes.len()];
        let root = k.roots()[rng.below(k.roots().len())];
        let (layout, maxima) = layouts.entry((n, root)).or_insert_with(|| (GeneLayout::from_program_size(&k, root, n).expect("layout"), Maxima::default()));
        let gene = random_gene(&k, layout, &mut rng);
        let e = express(&k, layout, &gene, &consts).unwrap_or_else(|d| panic!("{label} gene {g}: {d}"));
        assert_well_typed(&k, root, &e, &format!("{label} gene {g}"));
        worst = worst.max(e.nodes.len());
        maxima.take(&e);
    }
    for (layout, m) in layouts.values() {
        m.assert_below(layout);
    }
    println!("{label}: {genes} random genes closed, largest tree {worst} nodes, {} (N, root) layouts", layouts.len());
    worst
}

/// A gene whose head is built to demand the most of `target`'s tail (most
/// arguments of any type when `target` is None). Slots are met in the walk's
/// order; the k-th slot of a type reads that type's k-th head row, which gets
/// the function of that output with the most arguments of `target`, ties drawn.
/// A head row with no function to put there gets a terminal.
fn greedy_gene(k: &KingdomContract, layout: &GeneLayout, target: Option<Ty>, rng: &mut Rng) -> Vec<u32> {
    let mut gene = random_gene(k, layout, rng);
    let mut cursor: BTreeMap<Ty, usize> = BTreeMap::new();
    let mut queue = std::collections::VecDeque::from([layout.root]);
    while let Some(u) = queue.pop_front() {
        let frame = layout.frame(u).expect("frame");
        let c = cursor.entry(u).or_default();
        let row = *c;
        *c += 1;
        if row >= frame.head_len {
            continue;
        }
        let score = |f: &SymbolId| match target {
            Some(t) => k.signature(*f).iter().filter(|a| **a == t).count(),
            None => k.signature(*f).len(),
        };
        let Some(best) = k.functions_of(u).iter().map(score).max() else {
            gene[frame.head_off + row] = k.terminals_of(u)[rng.below(k.terminals_of(u).len())];
            continue;
        };
        let tied: Vec<SymbolId> = k.functions_of(u).iter().copied().filter(|f| score(f) == best).collect();
        let f = tied[rng.below(tied.len())];
        gene[frame.head_off + row] = f;
        queue.extend(k.signature(f).iter().copied());
    }
    gene
}

/// How close the greedy genes came to each tail's bound.
#[derive(Default)]
struct Attained {
    /// (N, root, frame type) combinations tried with a non-empty tail.
    frames: usize,
    /// Of those, the tails read to exactly `t_T`.
    attained: usize,
    largest_ratio: f64,
}

/// Genes built to demand the most of each type, from every root. Records the
/// largest tail and Dc cursor and asserts the design bound holds.
fn adversarial(rel: &KingdomRelations, label: &str, sizes: &[usize]) -> Attained {
    let k = rel.check().expect("kingdom check");
    let consts = const_lens(&k);
    let mut rng = Rng(0xad7e);
    let mut out = Attained::default();
    for n in sizes {
        for root in k.roots() {
            let layout = GeneLayout::from_program_size(&k, *root, *n).expect("layout");
            let mut m = Maxima::default();
            let targets: Vec<Option<Ty>> = std::iter::once(None).chain(k.types().iter().filter(|t| k.a_t(**t) > 0).map(|t| Some(*t))).collect();
            for target in targets {
                let gene = greedy_gene(&k, &layout, target, &mut rng);
                let e = express(&k, &layout, &gene, &consts).unwrap_or_else(|d| panic!("{label} greedy gene for {target:?}: {d}"));
                assert_well_typed(&k, *root, &e, label);
                m.take(&e);
            }
            m.assert_below(&layout);
            for f in layout.frames.iter().filter(|f| f.tail_len > 0) {
                let reached = m.tail.get(&f.ty).copied().unwrap_or(0);
                out.frames += 1;
                out.attained += usize::from(reached == f.tail_len);
                out.largest_ratio = out.largest_ratio.max(reached as f64 / f.tail_len as f64);
            }
        }
    }
    out
}

#[test]
fn closure_sr() {
    closure_n(&sr_relations(), "SR", &SIZES, GENES);
}

#[test]
fn closure_regex() {
    closure_n(&regex_relations(), "regex", &SIZES, GENES);
}

/// The two WGSL kingdoms in full, every dual a symbol and the roots as the
/// kingdoms declare them (both pass the kingdom check). Widths are tens of
/// thousands of words, so fewer genes and smaller program sizes.
#[cfg(feature = "wgsl")]
#[test]
fn closure_wgsl_and_safewgsl() {
    closure_n(&crate::kingdom_contract::wgsl_relations(), "WGSL", &SIZES[..2], 3_000);
    closure_n(&crate::kingdom_contract::safewgsl_relations(), "safewgsl", &SIZES[..2], 2_000);
}

/// The bound, measured. For SR it is attained exactly (a head of binary
/// functions needs h + 1 leaves); for every kingdom no head built to demand the
/// most passes it (`Maxima::assert_below`).
#[test]
fn tail_bound_measured_maximum() {
    let sr = adversarial(&sr_relations(), "SR", &SIZES);
    assert_eq!(sr.attained, sr.frames, "SR: t_F is attained by an all-binary head");
    println!("SR: {}/{} tails read to exactly t_T", sr.attained, sr.frames);
    let re = adversarial(&regex_relations(), "regex", &SIZES);
    println!("regex: {}/{} tails read to exactly t_T, largest ratio {:.2}", re.attained, re.frames, re.largest_ratio);
    #[cfg(feature = "wgsl")]
    for (rel, label) in [(crate::kingdom_contract::wgsl_relations(), "WGSL"), (crate::kingdom_contract::safewgsl_relations(), "safewgsl")] {
        let w = adversarial(&rel, label, &SIZES[..2]);
        println!("{label}: {}/{} tails read to exactly t_T, largest ratio {:.2}", w.attained, w.frames, w.largest_ratio);
        assert!(w.largest_ratio <= 1.0);
    }
    assert!(re.largest_ratio <= 1.0);
}

/// (symbol, parent step, argument index, constant index) of a Karva node.
type KarvaNode = (u32, Option<usize>, Option<u32>, Option<u32>);

/// Classical Karva, written independently of `express`: the k-th node in level
/// order is the cell at position k, its children are the next cells.
fn classical_karva(k: &KingdomContract, gene: &[u32], dc_off: usize) -> Vec<KarvaNode> {
    let mut out: Vec<KarvaNode> = Vec::new();
    let mut next_cell = 1usize;
    let mut q = 0usize;
    let mut i = 0usize;
    let mut parent_arg: Vec<(Option<usize>, Option<u32>)> = vec![(None, None)];
    while i < parent_arg.len() {
        let sym = gene[i];
        let konst = k.is_constant(sym).then(|| {
            let v = gene[dc_off + q];
            q += 1;
            v
        });
        out.push((sym, parent_arg[i].0, parent_arg[i].1, konst));
        for a in 0..k.signature(sym).len() {
            parent_arg.push((Some(i + 1), Some(a as u32)));
            next_cell += 1;
        }
        i += 1;
    }
    assert_eq!(next_cell, out.len(), "karva closes");
    out
}

/// One type: typed decoding equals classical Karva, gene for gene, constants
/// included, on the physical layout head `h = 1 + (N - 1) n_max`, tail and Dc of
/// `h (n_max - 1) + 1` each.
#[test]
fn sr_one_type_output_equals_classical_karva() {
    let k = sr_relations().check().expect("SR");
    let consts = const_lens(&k);
    let mut rng = Rng(0x4a12);
    let mut hist = 0usize;
    for g in 0..GENES {
        let n = [1usize, 2, 3, 5, 8, 21, 32, 40][g % 8];
        let layout = GeneLayout::from_program_size(&k, Ty::F, n).expect("layout");
        let f = layout.frame(Ty::F).expect("F").clone();
        let h = 1 + (n - 1) * k.n_max() as usize;
        let t = h * (k.n_max() as usize - 1) + 1;
        assert_eq!((f.head_off, f.head_len, f.tail_off, f.tail_len, f.dc_off, f.dc_len, layout.width), (0, h, h, t, Some(h + t), t, h + 2 * t));
        let gene = random_gene(&k, &layout, &mut rng);
        let typed = express(&k, &layout, &gene, &consts).unwrap_or_else(|d| panic!("gene {g}: {d}"));
        let want = classical_karva(&k, &gene, h + t);
        let got: Vec<_> = typed.nodes.iter().map(|n| (n.symbol, n.parent_step, n.arg_index, n.const_index)).collect();
        assert_eq!(got, want, "gene {g} differs from classical Karva");
        for (i, n) in typed.nodes.iter().enumerate() {
            assert_eq!(n.pos, i, "gene {g}: a one-type walk reads cells in physical order");
        }
        hist += typed.nodes.len();
    }
    println!("SR: {GENES} genes equal classical Karva, {hist} nodes compared");
}

fn fingerprint(e: &Expressed, h: &mut u64) {
    for n in &e.nodes {
        for w in [n.symbol as u64, n.step as u64, n.parent_step.unwrap_or(0) as u64, n.arg_index.map_or(0, u64::from), n.const_index.map_or(u64::MAX, u64::from), n.pos as u64] {
            *h = (*h ^ w).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
}

/// Same seed, same genes, same trees, run twice over the kingdoms.
#[test]
fn expression_is_deterministic() {
    let rels = [sr_relations(), regex_relations()];
    let run = || -> Vec<u64> {
        rels.iter()
            .map(|rel| {
                let k = rel.check().expect("check");
                let consts = const_lens(&k);
                let mut rng = Rng(77);
                let mut h = 0xcbf2_9ce4_8422_2325u64;
                for g in 0..10_000 {
                    let layout = GeneLayout::from_program_size(&k, k.roots()[0], SIZES[g % SIZES.len()]).expect("layout");
                    let gene = random_gene(&k, &layout, &mut rng);
                    fingerprint(&express(&k, &layout, &gene, &consts).expect("expresses"), &mut h);
                }
                h
            })
            .collect()
    };
    let (a, b) = (run(), run());
    assert_eq!(a, b);
    assert_ne!(a[0], a[1], "the fingerprints are not constant");
}

/// THE WALK WITHOUT THE TREE (`walk_expressed`, the F3a code review's size guard): on random genes of SR and regex at
/// every program size, the walk visits exactly `express`'s nodes in its order (symbol and constant word), and with a
/// limit below the node count it stops after `limit + 1` nodes.
#[test]
fn the_walk_without_the_tree_counts_and_visits_the_expressed_nodes() {
    for rel in [sr_relations(), regex_relations()] {
        let k = rel.check().expect("check");
        let consts = const_lens(&k);
        let mut rng = Rng(0x5122);
        let mut cut = 0;
        for g in 0..4_000 {
            let layout = GeneLayout::from_program_size(&k, k.roots()[0], SIZES[g % SIZES.len()]).expect("layout");
            let gene = random_gene(&k, &layout, &mut rng);
            let e = express(&k, &layout, &gene, &consts).expect("expresses");
            let want: Vec<(SymbolId, Option<u32>)> = e.nodes.iter().map(|n| (n.symbol, n.const_index)).collect();
            let mut seen = Vec::new();
            let n = crate::typed_gep::walk_expressed(&k, &layout, &gene, &consts, usize::MAX - 1, &mut |s, c| seen.push((s, c))).expect("walks");
            assert_eq!((n, &seen), (want.len(), &want), "the walk is the expression's");
            if want.len() > 2 {
                let limit = want.len() / 2;
                let mut few = Vec::new();
                let m = crate::typed_gep::walk_expressed(&k, &layout, &gene, &consts, limit, &mut |s, c| few.push((s, c))).expect("walks");
                assert_eq!((m, few.as_slice()), (limit + 1, &want[..limit + 1]), "stopped after limit + 1 nodes");
                cut += 1;
            }
        }
        assert!(cut > 100, "the limit was exercised: {cut} genes");
    }
}

use crate::kingdom_contract::relations_from_rows;
use crate::typed_gep::{fits, required_heads, write_back, LayoutDefect, Tree};

/// The design's two-type kingdom with the polymorphic spelling.
fn two_type() -> KingdomContract {
    relations_from_rows(
        "two-type",
        vec![Ty::S],
        &[
            ("left".to_string(), vec![Ty::S, Ty::I], Ty::S),
            ("s".to_string(), vec![], Ty::S),
            ("i".to_string(), vec![], Ty::I),
            ("z".to_string(), vec![], Ty::S),
            ("z".to_string(), vec![], Ty::I),
        ],
        &[],
    )
    .check()
    .expect("fixture")
}

fn leaf(symbol: SymbolId) -> Tree {
    Tree { symbol, const_index: None, kids: vec![] }
}

/// The program size is attained: the layout built for `N` holds the chain of
/// `N` functions that gives each function its last argument of the type, the
/// chain needing exactly `h_T` head rows; and the layout built for `N` and
/// then expressed reads exactly `t_T` tail rows of that chain's type.
/// Two-type: `left^N`, S demand `N + 1 = h_S + t_S`, I demand `N = h_I + t_I`.
#[test]
fn bound_attained_by_the_chain_of_n_functions_two_type() {
    let k = two_type();
    let (left, s, i) = (0u32, 1u32, 2u32);
    for n in 1..=24usize {
        let layout = GeneLayout::from_program_size(&k, Ty::S, n).expect("layout");
        let mut tree = Tree { symbol: left, const_index: None, kids: vec![leaf(s), leaf(i)] };
        for _ in 1..n {
            tree = Tree { symbol: left, const_index: None, kids: vec![tree, leaf(i)] };
        }
        let j = required_heads(&k, &tree);
        assert_eq!(j.get(&Ty::S), Some(&n), "N = {n}: the chain needs h_S = N head rows");
        fits(&k, &layout, &tree).unwrap_or_else(|d| panic!("N = {n}: {d}"));
        let gene = write_back(&k, &layout, &vec![s; layout.width], &tree).expect("encodes");
        let e = express(&k, &layout, &gene, &BTreeMap::new()).expect("expresses");
        assert_eq!(e.tree(), tree, "N = {n}");
        let (fs, fi) = (layout.frame(Ty::S).expect("S"), layout.frame(Ty::I).expect("I"));
        assert_eq!((fs.head_len, fs.tail_len, fi.head_len, fi.tail_len), (n, 1, n - 1, 1));
        assert_eq!(e.reach.frame_used.get(&Ty::S), Some(&(fs.head_len + fs.tail_len)), "N = {n}: the last S row of the frame is read");
        assert_eq!(e.reach.frame_used.get(&Ty::I), Some(&(fi.head_len + fi.tail_len)), "N = {n}: the last I row of the frame is read");
        assert!(j.get(&Ty::S).copied().unwrap_or(0) <= fs.head_len);
    }
}

/// One type, any arity: the chain of `N` n-ary functions, each giving the
/// chain to its last argument, needs exactly `h = 1 + (N - 1) n_max` head rows,
/// so a layout of one row fewer cannot hold it.
#[test]
fn bound_attained_by_the_chain_of_n_functions_one_type() {
    let k = sr_relations().check().expect("SR");
    let binary = (0..k.n_symbols() as SymbolId).find(|s| k.signature(*s).len() == 2).expect("a binary function");
    let x = k.terminals_of(Ty::F).iter().copied().find(|s| !k.is_constant(*s)).expect("x");
    for n in 1..=40usize {
        let mut tree = Tree { symbol: binary, const_index: None, kids: vec![leaf(x), leaf(x)] };
        for _ in 1..n {
            tree = Tree { symbol: binary, const_index: None, kids: vec![leaf(x), tree] };
        }
        let h = 1 + (n - 1) * 2;
        assert_eq!(required_heads(&k, &tree).get(&Ty::F), Some(&h), "N = {n}");
        let layout = GeneLayout::from_program_size(&k, Ty::F, n).expect("layout");
        assert!(fits(&k, &layout, &tree).is_ok());
        let one_short = GeneLayout::from_heads(&k, Ty::F, &BTreeMap::from([(Ty::F, h - 1)])).expect("layout");
        assert_eq!(fits(&k, &one_short, &tree), Err(LayoutDefect::HeadTooShort { ty: Ty::F, need: h, h: h - 1 }), "N = {n}: one head row fewer does not hold it");
    }
}

/// The `P_T` and `L + h_T` bounds of the two-type kingdom are attained: a head
/// of `left` in `head_S` expresses every one (`h_S + 1 = h_S + t_S` S slots),
/// and for the root-only I slots `P_I = h_S`.
#[test]
fn tail_bound_attained_two_type() {
    let k = two_type();
    let mut rng = Rng(5);
    for n in 1..=24usize {
        let layout = GeneLayout::from_program_size(&k, Ty::S, n).expect("layout");
        let gene = greedy_gene(&k, &layout, None, &mut rng);
        let e = express(&k, &layout, &gene, &BTreeMap::new()).expect("expresses");
        for f in layout.frames.iter().filter(|f| f.tail_len > 0) {
            assert_eq!(e.reach.tail_used.get(&f.ty), Some(&f.tail_len), "N = {n}: tail {:?} read to exactly t_T", f.ty);
        }
    }
}

/// Locality: an edit in `head_T` changes only cells of `head_T`, draws only
/// symbols outputting `T`, and the child expresses (it may change which rows
/// of another frame are read, never their validity). 100,000 edits per kingdom.
fn locality(rel: &KingdomRelations, label: &str, sizes: &[usize]) {
    let k = rel.check().expect("check");
    let consts = const_lens(&k);
    let mut rng = Rng(0x10ca ^ label.len() as u64);
    let mut edits = 0usize;
    for g in 0..GENES {
        let n = sizes[g % sizes.len()];
        let root = k.roots()[rng.below(k.roots().len())];
        let layout = GeneLayout::from_program_size(&k, root, n).expect("layout");
        let gene = random_gene(&k, &layout, &mut rng);
        let heads: Vec<Span> = layout.spans().into_iter().filter(|s| s.domain == Domain::Head).collect();
        if heads.is_empty() {
            continue;
        }
        let span = heads[rng.below(heads.len())];
        let pos = span.off + rng.below(span.len);
        let alphabet = layout.alphabet(&k, pos, &consts);
        let value = alphabet[rng.below(alphabet.len())];
        let mut child = gene.clone();
        layout.mutate(&k, &mut child, pos, value, &consts).expect("a drawn value is in the alphabet");
        assert_eq!(k.output(value), span.ty, "{label}: head_T draws symbols of T");
        for p in 0..layout.width {
            assert!(p == pos || child[p] == gene[p], "{label}: cell {p} changed by an edit at {pos}");
        }
        let e = express(&k, &layout, &child, &consts).unwrap_or_else(|d| panic!("{label} edit {g}: {d}"));
        assert_well_typed(&k, root, &e, label);
        edits += 1;
    }
    println!("{label}: {edits} head edits stayed in their domain and expressed");
}

#[test]
fn locality_sr() {
    locality(&sr_relations(), "SR", &SIZES);
}

#[test]
fn locality_regex() {
    locality(&regex_relations(), "regex", &SIZES);
}

/// Homologous crossover: two genes of one layout exchange a range inside one
/// domain of one type; the cells outside the range are untouched, the range
/// in each child is the other parent's, and both children express. 100,000 per kingdom.
fn crossover(rel: &KingdomRelations, label: &str, sizes: &[usize]) {
    let k = rel.check().expect("check");
    let consts = const_lens(&k);
    let mut rng = Rng(0xc0de ^ label.len() as u64);
    let mut swaps = 0usize;
    for g in 0..GENES {
        let n = sizes[g % sizes.len()];
        let root = k.roots()[rng.below(k.roots().len())];
        let layout = GeneLayout::from_program_size(&k, root, n).expect("layout");
        let (a, b) = (random_gene(&k, &layout, &mut rng), random_gene(&k, &layout, &mut rng));
        let spans = layout.spans();
        let span = spans[rng.below(spans.len())];
        let lo = span.off + rng.below(span.len);
        let hi = lo + 1 + rng.below(span.off + span.len - lo);
        let (mut ca, mut cb) = (a.clone(), b.clone());
        layout.swap_range(&layout, &mut ca, &mut cb, lo, hi).expect("same layout, one domain");
        for p in 0..layout.width {
            let inside = (lo..hi).contains(&p);
            assert_eq!((ca[p], cb[p]), if inside { (b[p], a[p]) } else { (a[p], b[p]) }, "{label}: cell {p}, range [{lo}, {hi})");
        }
        for (child, what) in [(&ca, "first"), (&cb, "second")] {
            let e = express(&k, &layout, child, &consts).unwrap_or_else(|d| panic!("{label} {what} child {g}: {d}"));
            assert_well_typed(&k, root, &e, label);
        }
        swaps += 1;
    }
    // A layout of another program size is not homologous: nothing is exchanged.
    let (l1, l2) = (GeneLayout::from_program_size(&k, k.roots()[0], 3).expect("layout"), GeneLayout::from_program_size(&k, k.roots()[0], 4).expect("layout"));
    let (mut a, mut b) = (random_gene(&k, &l1, &mut rng), random_gene(&k, &l2, &mut rng));
    let (a0, b0) = (a.clone(), b.clone());
    assert_eq!(l1.swap_range(&l2, &mut a, &mut b, 0, 1), Err(LayoutDefect::NotHomologous));
    assert_eq!((a, b), (a0, b0));
    println!("{label}: {swaps} homologous exchanges closed");
}

#[test]
fn crossover_sr() {
    crossover(&sr_relations(), "SR", &SIZES);
}

#[test]
fn crossover_regex() {
    crossover(&regex_relations(), "regex", &SIZES);
}

/// Every invalid cell is named and none is repaired: in 100,000 random genes,
/// one cell, read or not, is replaced by a symbol outside its domain's
/// alphabet (a symbol of another type in `head_T` or `tail_T`, a function in a
/// tail, a symbol outside the kingdom, a Dc word past `C_T`). The decoder
/// returns exactly that defect at exactly that cell.
fn planted_defects(rel: &KingdomRelations, label: &str) {
    let k = rel.check().expect("check");
    let consts = const_lens(&k);
    let mut rng = Rng(0xdef ^ label.len() as u64);
    let mut seen = BTreeMap::<&str, usize>::new();
    for g in 0..GENES {
        let n = SIZES[g % SIZES.len()];
        let root = k.roots()[rng.below(k.roots().len())];
        let layout = GeneLayout::from_program_size(&k, root, n).expect("layout");
        let mut gene = random_gene(&k, &layout, &mut rng);
        let spans = layout.spans();
        let span = spans[rng.below(spans.len())];
        let pos = span.off + rng.below(span.len);
        let (want, kind) = match span.domain {
            Domain::Dc => {
                gene[pos] = CONSTS + rng.below(1000) as u32;
                (LayoutDefect::DcIndexPastConstants { ty: span.ty, index: gene[pos], len: CONSTS }, "dc past C_T")
            }
            Domain::Dv => {
                gene[pos] = crate::typed_gep::DV_WORDS + rng.below(1000) as u32;
                (LayoutDefect::DvWordPastRange { pos, word: gene[pos] }, "dv past its range")
            }
            Domain::Head | Domain::Tail => match rng.below(3) {
                0 => {
                    gene[pos] = (k.n_symbols() + rng.below(1000)) as u32;
                    (LayoutDefect::UnknownSymbol { pos, symbol: gene[pos] }, "unknown symbol")
                }
                1 => {
                    let foreign: Vec<SymbolId> = (0..k.n_symbols() as SymbolId).filter(|s| k.output(*s) != span.ty).collect();
                    if foreign.is_empty() {
                        continue;
                    }
                    gene[pos] = foreign[rng.below(foreign.len())];
                    (LayoutDefect::ForeignSymbol { ty: span.ty, symbol: gene[pos], pos }, if span.domain == Domain::Head { "wrong output in head_T" } else { "wrong output in tail_T" })
                }
                _ => {
                    let functions = k.functions_of(span.ty);
                    if span.domain == Domain::Head || functions.is_empty() {
                        continue;
                    }
                    gene[pos] = functions[rng.below(functions.len())];
                    (LayoutDefect::FunctionInTail { ty: span.ty, symbol: gene[pos], pos }, "function in tail_T")
                }
            },
        };
        let err = express(&k, &layout, &gene, &consts).expect_err("a planted defect must be reported");
        assert_eq!(err, want, "{label} gene {g}: {kind}");
        assert!(err.to_string().starts_with("our software has a bug: "), "{err}");
        *seen.entry(kind).or_default() += 1;
    }
    let genes = random_gene(&k, &GeneLayout::from_program_size(&k, k.roots()[0], 3).expect("layout"), &mut rng);
    let layout = GeneLayout::from_program_size(&k, k.roots()[0], 3).expect("layout");
    assert_eq!(express(&k, &layout, &genes[1..], &consts), Err(LayoutDefect::GeneLength { want: layout.width, got: layout.width - 1 }));
    println!("{label}: planted defects named at their cell: {seen:?}");
    assert!(seen.contains_key("wrong output in head_T") || k.types().len() == 1, "{label}: the head_T defect was exercised");
}

#[test]
fn planted_defects_sr() {
    planted_defects(&sr_relations(), "SR");
}

#[test]
fn planted_defects_regex() {
    planted_defects(&regex_relations(), "regex");
}

#[test]
fn planted_defects_two_type() {
    let rel = relations_from_rows(
        "two-type",
        vec![Ty::S],
        &[
            ("left".to_string(), vec![Ty::S, Ty::I], Ty::S),
            ("s".to_string(), vec![], Ty::S),
            ("i".to_string(), vec![], Ty::I),
            ("z".to_string(), vec![], Ty::S),
            ("z".to_string(), vec![], Ty::I),
        ],
        &[],
    );
    planted_defects(&rel, "two-type");
}

/// A frame the layout does not have and a Dc that is too short are defects the
/// walk reports when it meets them; nothing is padded or skipped.
#[test]
fn a_missing_frame_and_a_short_dc_are_defects() {
    let k = two_type();
    let (left, s, i) = (0u32, 1u32, 2u32);
    let mut layout = GeneLayout::from_heads(&k, Ty::S, &BTreeMap::from([(Ty::S, 2), (Ty::I, 1)])).expect("layout");
    layout.frames.retain(|f| f.ty != Ty::I);
    assert_eq!(express(&k, &layout, &[left, s, i, s, i], &BTreeMap::new()), Err(LayoutDefect::NoFrame { ty: Ty::I }));

    let f = relations_from_rows("c", vec![Ty::F], &[("add".to_string(), vec![Ty::F, Ty::F], Ty::F), ("x".to_string(), vec![], Ty::F), ("?".to_string(), vec![], Ty::F)], &["?"]).check().expect("valid");
    let mut layout = GeneLayout::from_heads(&f, Ty::F, &BTreeMap::from([(Ty::F, 2)])).expect("layout");
    layout.frames[0].dc_len = 0;
    assert_eq!(express(&f, &layout, &[0, 2, 1, 1, 1, 0, 0, 0], &BTreeMap::from([(Ty::F, 4)])), Err(LayoutDefect::DcRanOut { ty: Ty::F, q: 0 }));
}

use crate::typed_gep::Layout;

// ---------------------------------------------------------------------------
// The DuckDB bound and expression queries of the design (section 3), run on the
// fixtures, equal to the host
// ---------------------------------------------------------------------------

/// The design's views and recursive query, verbatim from `MultiTypedGEP.md`
/// section 3 (extracted, not retyped).
const WALK_SQL: &str = include_str!("../nucleotable/sql/typed_gep_expression.sql");

/// Loads the setup script, creates the views, derives `layout_bounds`, runs the
/// walk for each chromosome and prints `{bounds, walk: {chromosome_id: rows}}`
/// as JSON. Reads argv: the SQL file, the setup script, then chromosome ids.
const DRIVER: &str = r#"
import duckdb, json, sys
views, query = open(sys.argv[1]).read().split("\n\nWITH RECURSIVE", 1)
query = "WITH RECURSIVE" + query
con = duckdb.connect(":memory:")
con.execute(open(sys.argv[2]).read())
con.execute(views)
bounds = con.execute("SELECT layout_id, gene_id, type, h, p, l, t, d FROM layout_bounds").fetchall()
walk = {c: con.execute(query, [int(c)]).fetchall() for c in sys.argv[3:]}
print(json.dumps({"bounds": bounds, "walk": walk}))
"#;

fn sql_str(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// A kingdom to load: its name, contract, and which symbols are reference
/// terminals as `(symbol_id, kind, target_id)` rather than ordinary terminals.
struct SqlKingdom<'a> {
    name: &'a str,
    k: &'a KingdomContract,
    refs: &'a [(SymbolId, &'a str, i64)],
}

/// One gene to load: where it lives, its kingdom, layout and words.
struct Loaded<'a> {
    chromosome: u64,
    gene: u64,
    kingdom: &'a SqlKingdom<'a>,
    layout: &'a Layout,
    words: Vec<u32>,
}

fn setup_sql(kingdoms: &[&SqlKingdom], layouts: &[(&SqlKingdom, &Layout)], genes: &[Loaded]) -> String {
    let mut s = String::from(
        "CREATE TABLE functions(kingdom_id VARCHAR, symbol_id BIGINT, spelling VARCHAR, output_type VARCHAR);\n\
         CREATE TABLE terminals(kingdom_id VARCHAR, symbol_id BIGINT, spelling VARCHAR, output_type VARCHAR);\n\
         CREATE TABLE reference_terminals(kingdom_id VARCHAR, symbol_id BIGINT, spelling VARCHAR, output_type VARCHAR, kind VARCHAR, target_id BIGINT);\n\
         CREATE TABLE argument_signature(kingdom_id VARCHAR, symbol_id BIGINT, arg_index BIGINT, required_type VARCHAR);\n\
         CREATE TABLE layouts(layout_id BIGINT, kingdom_id VARCHAR, program_size BIGINT);\n\
         CREATE TABLE layout_genes(layout_id BIGINT, gene_id BIGINT, root_type VARCHAR);\n\
         CREATE TABLE layout_heads(layout_id BIGINT, gene_id BIGINT, type VARCHAR, h BIGINT);\n\
         CREATE TABLE genes(chromosome_id BIGINT, gene_id BIGINT, kingdom_id VARCHAR, layout_id BIGINT);\n\
         CREATE TABLE geneframe(chromosome_id BIGINT, gene_id BIGINT, kingdom_id VARCHAR, domain VARCHAR, domain_type VARCHAR, domain_seq_id BIGINT, symbol_id BIGINT, dc_index BIGINT);\n",
    );
    for kd in kingdoms {
        let (name, k) = (kd.name, kd.k);
        for sym in 0..k.n_symbols() as SymbolId {
            let out = sql_str(&format!("{:?}", k.output(sym)));
            if let Some((_, kind, target)) = kd.refs.iter().find(|(r, _, _)| *r == sym) {
                s.push_str(&format!("INSERT INTO reference_terminals VALUES ({}, {sym}, {}, {out}, {}, {target});\n", sql_str(name), sql_str(k.name(sym)), sql_str(kind)));
            } else {
                let table = if k.is_terminal(sym) { "terminals" } else { "functions" };
                s.push_str(&format!("INSERT INTO {table} VALUES ({}, {sym}, {}, {out});\n", sql_str(name), sql_str(k.name(sym))));
            }
            for (i, t) in k.signature(sym).iter().enumerate() {
                s.push_str(&format!("INSERT INTO argument_signature VALUES ({}, {sym}, {i}, {});\n", sql_str(name), sql_str(&format!("{t:?}"))));
            }
        }
    }
    for (kd, layout) in layouts {
        let program_size = layout.gene(0).and_then(|g| g.program_size).map_or("NULL".to_string(), |n| n.to_string());
        s.push_str(&format!("INSERT INTO layouts VALUES ({}, {}, {program_size});\n", layout.layout_id, sql_str(kd.name)));
        for (gid, g) in layout.genes().iter().enumerate() {
            s.push_str(&format!("INSERT INTO layout_genes VALUES ({}, {gid}, {});\n", layout.layout_id, sql_str(&format!("{:?}", g.root))));
            for (ty, h) in g.head_lens() {
                s.push_str(&format!("INSERT INTO layout_heads VALUES ({}, {gid}, {}, {h});\n", layout.layout_id, sql_str(&format!("{ty:?}"))));
            }
        }
    }
    for g in genes {
        s.push_str(&format!("INSERT INTO genes VALUES ({}, {}, {}, {});\n", g.chromosome, g.gene, sql_str(g.kingdom.name), g.layout.layout_id));
        let layout = g.layout.gene(g.gene).expect("gene id in layout");
        let rows: Vec<String> = layout
            .rows(g.chromosome, g.gene, &g.words)
            .iter()
            .map(|r| {
                format!(
                    "({}, {}, {}, {}, {}, {}, {}, {})",
                    r.chromosome_id,
                    r.gene_id,
                    sql_str(g.kingdom.name),
                    sql_str(r.key.domain.name()),
                    sql_str(&format!("{:?}", r.key.domain_type)),
                    r.key.domain_seq_id,
                    r.symbol.map_or("NULL".to_string(), |x| x.to_string()),
                    r.dc_index.map_or("NULL".to_string(), |x| x.to_string())
                )
            })
            .collect();
        s.push_str(&format!("INSERT INTO geneframe VALUES {};\n", rows.join(", ")));
    }
    s
}

/// A node as the query reports it: step, type, frame_k, spelling, depth, parent_step, arg_index.
fn host_row(k: &KingdomContract, n: &crate::typed_gep::Node) -> String {
    format!("{} {:?} {} {} {} {:?} {:?}", n.step, n.ty, n.frame_k, k.name(n.symbol), n.depth, n.parent_step, n.arg_index)
}

fn sql_row(r: &serde_json::Value) -> String {
    let opt = |v: &serde_json::Value| if v.is_null() { "None".to_string() } else { format!("Some({v})") };
    // The query's columns: gene_id, step, node_type, frame_k, spelling, depth, parent_step, arg_index.
    format!("{} {} {} {} {} {} {}", r[1], r[2].as_str().unwrap_or("?"), r[3], r[4].as_str().unwrap_or("?"), r[5], opt(&r[6]), opt(&r[7]))
}

/// The design's two queries, run in DuckDB on four fixtures: the two-type
/// kingdom with the polymorphic spelling, two genes of one chromosome sharing a
/// layout, an ADF reference terminal, and the same gene id in several
/// chromosomes (so a frame that leaked across a gene or chromosome would show);
/// the one-type SR layout of the design; and 100 random regex genes over 20
/// chromosomes. The bound query equals `KingdomContract::bounds` for every
/// (layout, gene, type) and every walk row equals the host decoder's node.
/// Skipped, with the reason printed, when no interpreter has the `duckdb`
/// module (`FULLER_DUCKDB_PYTHON` names one).
#[test]
fn duckdb_bound_and_expression_queries_equal_the_host() {
    use std::process::Command;
    let python = std::env::var("FULLER_DUCKDB_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let probe = Command::new(&python).args(["-I", "-c", "import duckdb"]).output();
    if !probe.is_ok_and(|o| o.status.success()) {
        eprintln!("SKIPPED duckdb_bound_and_expression_queries_equal_the_host: `{python} -I -c 'import duckdb'` fails; set FULLER_DUCKDB_PYTHON");
        return;
    }
    let (left, s, i, z_s, a7) = (0u32, 1u32, 2u32, 3u32, 5u32);
    // left:S(S,I); s:S; i:I; z:S and z:I (one symbol_id per output type); A7:I, an ADF reference.
    let two = relations_from_rows(
        "two-type",
        vec![Ty::S],
        &[
            ("left".to_string(), vec![Ty::S, Ty::I], Ty::S),
            ("s".to_string(), vec![], Ty::S),
            ("i".to_string(), vec![], Ty::I),
            ("z".to_string(), vec![], Ty::S),
            ("z".to_string(), vec![], Ty::I),
            ("A7".to_string(), vec![], Ty::I),
        ],
        &[],
    )
    .check()
    .expect("fixture");
    let z_i = 4u32;
    let sr = sr_relations().check().expect("SR");
    let re = regex_relations().check().expect("regex");
    let two_kd = SqlKingdom { name: "two-type", k: &two, refs: &[(a7, "adf", 7)] };
    let sr_kd = SqlKingdom { name: "sr", k: &sr, refs: &[] };
    let re_kd = SqlKingdom { name: "regex", k: &re, refs: &[] };

    let design = GeneLayout::from_heads(&two, Ty::S, &BTreeMap::from([(Ty::S, 2), (Ty::I, 1)])).expect("layout");
    let layout_two = Layout::new(1, vec![design.clone(), design]);
    let layout_sr = Layout::new(2, vec![GeneLayout::from_heads(&sr, Ty::F, &BTreeMap::from([(Ty::F, 7)])).expect("layout")]);
    let layout_re = Layout::new(3, (0..5).map(|_| GeneLayout::from_program_size(&re, Ty::Pattern, 6).expect("layout")).collect());

    let mut genes: Vec<Loaded> = Vec::new();
    let mut put = |chromosome: u64, gene: u64, kingdom, layout, words: Vec<u32>| genes.push(Loaded { chromosome, gene, kingdom, layout, words });
    put(1, 0, &two_kd, &layout_two, vec![left, s, i, s, i]); // left(s, i), the design's table
    put(1, 1, &two_kd, &layout_two, vec![left, left, i, s, z_i]); // left(left(s, z), i)
    put(2, 0, &two_kd, &layout_two, vec![left, s, a7, s, i]); // left(s, A7)
    put(3, 0, &two_kd, &layout_two, vec![left, left, i, z_s, a7]); // left(left(z, A7), i), z as an S
    let mut rng = Rng(0xd0c);
    for chromosome in 4..9u64 {
        let g = random_gene(&sr, layout_sr.gene(0).expect("gene"), &mut rng);
        put(chromosome, 0, &sr_kd, &layout_sr, g);
    }
    for chromosome in 9..29u64 {
        for gene in 0..5u64 {
            let words = random_gene(&re, layout_re.gene(gene).expect("gene"), &mut rng);
            put(chromosome, gene, &re_kd, &layout_re, words);
        }
    }
    let dir = std::env::temp_dir().join(format!("fuller_typed_gep_sql_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let (driver, walk, setup) = (dir.join("driver.py"), dir.join("walk.sql"), dir.join("setup.sql"));
    std::fs::write(&driver, DRIVER).expect("write driver");
    std::fs::write(&walk, WALK_SQL).expect("write walk");
    std::fs::write(&setup, setup_sql(&[&two_kd, &sr_kd, &re_kd], &[(&two_kd, &layout_two), (&sr_kd, &layout_sr), (&re_kd, &layout_re)], &genes)).expect("write setup");
    let mut chroms: Vec<u64> = genes.iter().map(|g| g.chromosome).collect();
    chroms.dedup();
    let out = Command::new(&python).arg("-I").arg(&driver).arg(&walk).arg(&setup).args(chroms.iter().map(u64::to_string)).output().expect("run the driver");
    assert!(out.status.success(), "duckdb driver failed: {}", String::from_utf8_lossy(&out.stderr));
    let result: serde_json::Value = serde_json::from_slice(&out.stdout).expect("driver prints JSON");

    // The bound query, row for row, against the kingdom contract.
    let mut got_bounds: Vec<String> = result["bounds"]
        .as_array()
        .expect("bounds rows")
        .iter()
        .map(|r| format!("{} {} {} h{} p{} l{} t{} d{}", r[0], r[1], r[2].as_str().unwrap_or("?"), r[3], r[4], r[5], r[6], r[7]))
        .collect();
    let mut want_bounds: Vec<String> = Vec::new();
    for (kd, layout) in [(&two_kd, &layout_two), (&sr_kd, &layout_sr), (&re_kd, &layout_re)] {
        for (gid, g) in layout.genes().iter().enumerate() {
            let b = kd.k.bounds(g.root, &g.head_lens()).expect("bounds");
            for t in &b.types {
                want_bounds.push(format!("{} {gid} {:?} h{} p{} l{} t{} d{}", layout.layout_id, t.ty, t.h, t.p, b.l, t.t, t.d));
            }
        }
    }
    got_bounds.sort();
    want_bounds.sort();
    assert_eq!(got_bounds, want_bounds, "the bound query differs from KingdomContract::bounds");
    // The design's figures: h_S = 2, h_I = 1, root S gives P_S 3, P_I 2, L 3, t 1/1, d 3/2; SR h = 7 gives t = d = 8.
    assert!(got_bounds.contains(&"1 0 S h2 p3 l3 t1 d3".to_string()) && got_bounds.contains(&"1 0 I h1 p2 l3 t1 d2".to_string()));
    assert!(got_bounds.contains(&"2 0 F h7 p15 l8 t8 d8".to_string()));

    // The walk, chromosome by chromosome, against the host decoder.
    let mut compared = 0usize;
    for c in &chroms {
        let rows = result["walk"][c.to_string()].as_array().unwrap_or_else(|| panic!("chromosome {c} has no result")).clone();
        let mut want: Vec<(u64, usize, String)> = Vec::new();
        for g in genes.iter().filter(|g| g.chromosome == *c) {
            let layout = g.layout.gene(g.gene).expect("gene");
            let e = express(g.kingdom.k, layout, &g.words, &const_lens(g.kingdom.k)).unwrap_or_else(|d| panic!("chromosome {c} gene {}: {d}", g.gene));
            want.extend(e.nodes.iter().map(|n| (g.gene, n.step, host_row(g.kingdom.k, n))));
        }
        want.sort_by_key(|(gene, step, _)| (*gene, *step));
        let got: Vec<String> = rows.iter().map(sql_row).collect();
        let want: Vec<String> = want.into_iter().map(|(_, _, r)| r).collect();
        assert_eq!(got, want, "chromosome {c}: SQL and host decoder differ");
        compared += got.len();
    }
    // The design's trees, as the SQL returned them.
    let walk_of = |c: &str| -> Vec<String> { result["walk"][c].as_array().expect("chromosome").iter().map(sql_row).collect() };
    assert_eq!(walk_of("1")[..3], ["1 S 0 left 0 None None", "2 S 1 s 1 Some(1) Some(0)", "3 I 0 i 1 Some(1) Some(1)"]);
    assert_eq!(walk_of("1")[3..], ["1 S 0 left 0 None None", "2 S 1 left 1 Some(1) Some(0)", "3 I 0 i 1 Some(1) Some(1)", "4 S 2 s 2 Some(2) Some(0)", "5 I 1 z 2 Some(2) Some(1)"]);
    assert_eq!(walk_of("2"), ["1 S 0 left 0 None None", "2 S 1 s 1 Some(1) Some(0)", "3 I 0 A7 1 Some(1) Some(1)"]);
    println!("DuckDB bound and expression queries equal the host on {} genes in {} chromosomes, {} bound rows, {compared} nodes", genes.len(), chroms.len(), got_bounds.len());
    for f in [&driver, &walk, &setup] {
        std::fs::remove_file(f).expect("clean");
    }
    std::fs::remove_dir(&dir).expect("clean");
}

/// THE ROW TABLE (Andrew, 2026-11-01): a domain of the gene is a table, row = position, column = type, and the
/// inversion, transposition and crossover act on rows across every column. Every symbol stays in its own column, so
/// every child is legal; with one type each is the one-type operator; ragged columns take part only where they have
/// the row; the same seed gives the same children.
mod rows {
    use super::*;
    use crate::typed_gep::{frac_of_row, frac_row, validate, FRAC_ONE};

    /// A fraction uniform in `[0, 1)`.
    fn frac(rng: &mut Rng) -> u64 {
        rng.next() >> 32
    }

    /// Two fractions `lo < hi <= 1`, `lo` uniform in `[0, 1)`.
    fn frac_pair(rng: &mut Rng) -> (u64, u64) {
        let lo = frac(rng);
        (lo, lo + 1 + rng.next() % (FRAC_ONE - lo))
    }

    /// How many random (layout, gene) pairs a kingdom's property test draws.
    const ROW_GENES: usize = 20_000;

    /// Every kingdom of the property tests: the one-type SR, the four-type regex, the design's two-type fixture.
    fn kingdoms() -> Vec<(KingdomContract, &'static str)> {
        vec![(sr_relations().check().expect("SR"), "SR"), (regex_relations().check().expect("regex"), "regex"), (two_type(), "two-type")]
    }

    fn legal(k: &KingdomContract, layout: &GeneLayout, gene: &[u32], consts: &BTreeMap<Ty, u32>, what: &str) {
        validate(k, layout, gene, consts).unwrap_or_else(|d| panic!("{what}: {d}"));
        let e = express(k, layout, gene, consts).unwrap_or_else(|d| panic!("{what}: {d}"));
        assert_well_typed(k, layout.root, &e, what);
    }

    /// The (column, row) of every position of `domain`'s table, `None` for a cell of another domain.
    fn cells_of(layout: &GeneLayout, domain: Domain) -> Vec<Option<(usize, usize)>> {
        let mut v = vec![None; layout.width];
        for (c, s) in layout.columns(domain).iter().enumerate() {
            for r in 0..s.len {
                v[s.off + r] = Some((c, r));
            }
        }
        v
    }

    /// The row in its whole column (`head_T`, `tail_T`, `Dc_T`) of every position.
    fn gene_rows_of(layout: &GeneLayout) -> Vec<usize> {
        let mut v = vec![usize::MAX; layout.width];
        for f in &layout.frames {
            let mut r = 0;
            while let Some(p) = GeneLayout::column_pos(f, r) {
                v[p] = r;
                r += 1;
            }
        }
        assert!(v.iter().all(|r| *r != usize::MAX), "every cell is in one column");
        v
    }

    /// The classic IS on one column, written independently: the run copied in, the column cut back to its length.
    fn classic_is(col: &[u32], src: usize, len: usize, dst: usize) -> Vec<u32> {
        let run: Vec<u32> = col[src..(src + len).min(col.len())].to_vec();
        let mut out: Vec<u32> = col[..dst].to_vec();
        out.extend(run);
        out.extend_from_slice(&col[dst..]);
        out.truncate(col.len());
        out
    }

    /// One random application of each row operator to random genes of random layouts; returns a fingerprint of
    /// every child, for the determinism test.
    fn row_operators(k: &KingdomContract, label: &str, seed: u64, genes: usize) -> u64 {
        let consts = const_lens(k);
        let mut rng = Rng(seed);
        let mut fp = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |g: &[u32]| {
            for w in g {
                fp = (fp ^ u64::from(*w)).wrapping_mul(0x100_0000_01b3);
            }
        };
        let (mut inverted, mut transposed, mut ris, mut crossed) = (0usize, 0usize, 0usize, 0usize);
        for g in 0..genes {
            let n = SIZES[g % SIZES.len()];
            let root = k.roots()[rng.below(k.roots().len())];
            let layout = GeneLayout::from_program_size(k, root, n).expect("layout");
            let (a, b) = (random_gene(k, &layout, &mut rng), random_gene(k, &layout, &mut rng));
            let what = |op: &str| format!("{label} gene {g} N = {n} {op}");

            // Inversion, in a domain drawn uniformly.
            let domain = [Domain::Head, Domain::Tail, Domain::Dc][rng.below(3)];
            let rows = layout.domain_rows(domain);
            if rows >= 2 {
                let lo = rng.below(rows - 1);
                let hi = lo + 2 + rng.below(rows - lo - 1);
                let mut c = a.clone();
                let changed = layout.invert_rows(&mut c, domain, lo, hi).expect("rows inside the table");
                let at = cells_of(&layout, domain);
                let cols = layout.columns(domain);
                for p in 0..layout.width {
                    let want = match at[p] {
                        Some((ci, r)) if r >= lo && r < hi.min(cols[ci].len) => a[cols[ci].off + lo + hi.min(cols[ci].len) - 1 - r],
                        _ => a[p],
                    };
                    assert_eq!(c[p], want, "{}: cell {p}", what("invert_rows"));
                }
                let differ = cols.iter().filter(|s| a[s.off..s.off + s.len] != c[s.off..s.off + s.len]).count();
                assert_eq!(changed, differ, "{}: the changed columns", what("invert_rows"));
                legal(k, &layout, &c, &consts, &what("invert_rows"));
                eat(&c);
                inverted += 1;
            }

            // IS, in a domain drawn uniformly (the runner's is the head; every domain keeps the child legal).
            let domain = [Domain::Head, Domain::Tail, Domain::Dc][rng.below(3)];
            let rows = layout.domain_rows(domain);
            if rows >= 1 {
                let (src, len, dst) = (rng.below(rows), 1 + rng.below(3), rng.below(rows));
                let mut c = a.clone();
                layout.transpose_rows(&mut c, domain, src, len, dst).expect("rows inside the table");
                for s in layout.columns(domain) {
                    let col = &a[s.off..s.off + s.len];
                    let want = if src < s.len && dst < s.len { classic_is(col, src, len, dst) } else { col.to_vec() };
                    assert_eq!(&c[s.off..s.off + s.len], &want[..], "{}: column {:?}", what("transpose_rows"), s.ty);
                }
                let at = cells_of(&layout, domain);
                assert!((0..layout.width).filter(|p| at[*p].is_none()).all(|p| a[p] == c[p]), "{}: a cell of another domain changed", what("transpose_rows"));
                legal(k, &layout, &c, &consts, &what("transpose_rows"));
                eat(&c);
                transposed += 1;
            }

            // RIS: the run starts where the root's head column holds a function, inserted at row 0.
            let rows = layout.domain_rows(Domain::Head);
            if rows >= 1 {
                if let Some(src) = layout.ris_row(k, &a, rng.below(rows)) {
                    let f = layout.frame(root).expect("root frame");
                    assert!(!k.signature(a[f.head_off + src]).is_empty(), "{}: RIS starts at a function", what("RIS"));
                    let mut c = a.clone();
                    layout.transpose_rows(&mut c, Domain::Head, src, 1 + rng.below(3), 0).expect("rows inside the table");
                    assert_eq!(c[f.head_off], a[f.head_off + src], "{}: the root's row 0 is the function", what("RIS"));
                    legal(k, &layout, &c, &consts, &what("RIS"));
                    eat(&c);
                    ris += 1;
                }
            }

            // One- and two-point crossover over the whole gene's rows.
            let rows = layout.gene_rows();
            let (cut1, cut2) = if rng.below(2) == 0 {
                (rng.below(rows), None)
            } else {
                let c1 = rng.below(rows);
                (c1, Some(c1 + 1 + rng.below(rows - c1)))
            };
            let (mut ca, mut cb) = (a.clone(), b.clone());
            let cells = layout.crossover_rows(&layout, &mut ca, &mut cb, cut1, cut2).expect("homologous, rows inside the table");
            let row = gene_rows_of(&layout);
            let hi = cut2.unwrap_or(rows);
            let mut swapped = 0;
            for p in 0..layout.width {
                let inside = row[p] >= cut1 && row[p] < hi;
                swapped += usize::from(inside);
                assert_eq!((ca[p], cb[p]), if inside { (b[p], a[p]) } else { (a[p], b[p]) }, "{}: cell {p}", what("crossover_rows"));
            }
            assert_eq!(cells, swapped, "{}: the cells exchanged", what("crossover_rows"));
            legal(k, &layout, &ca, &consts, &what("crossover_rows first child"));
            legal(k, &layout, &cb, &consts, &what("crossover_rows second child"));
            eat(&ca);
            eat(&cb);
            crossed += 1;
        }
        // THE RELATIVE ROWS: each column at its own row floor(f * len), so every column takes part.
        let mut relative = 0usize;
        for g in 0..genes {
            let n = SIZES[g % SIZES.len()];
            let root = k.roots()[rng.below(k.roots().len())];
            let layout = GeneLayout::from_program_size(k, root, n).expect("layout");
            let (a, b) = (random_gene(k, &layout, &mut rng), random_gene(k, &layout, &mut rng));
            let what = |op: &str| format!("{label} gene {g} N = {n} {op}");
            let domain = [Domain::Head, Domain::Tail, Domain::Dc][rng.below(3)];
            let cols = layout.columns(domain);

            let (lo, hi) = frac_pair(&mut rng);
            let mut c = a.clone();
            layout.invert_rel(&mut c, domain, lo, hi).expect("fractions in order");
            for s in &cols {
                let (l, h) = (frac_row(lo, s.len), frac_row(hi, s.len));
                let mut want = a[s.off..s.off + s.len].to_vec();
                want[l..h].reverse();
                assert_eq!(&c[s.off..s.off + s.len], &want[..], "{}: column {:?}", what("invert_rel"), s.ty);
            }
            legal(k, &layout, &c, &consts, &what("invert_rel"));
            eat(&c);

            let (src, dst) = (frac(&mut rng), frac(&mut rng));
            let end = src + rng.next() % (FRAC_ONE - src + 1);
            let mut c = a.clone();
            layout.transpose_rel(&mut c, domain, src, end, dst).expect("fractions in order");
            for s in &cols {
                let (r, d) = (frac_row(src, s.len), frac_row(dst, s.len));
                let e = frac_row(end, s.len).clamp(r + 1, s.len);
                let want = classic_is(&a[s.off..s.off + s.len], r, e - r, d);
                assert_eq!(&c[s.off..s.off + s.len], &want[..], "{}: column {:?}", what("transpose_rel"), s.ty);
            }
            legal(k, &layout, &c, &consts, &what("transpose_rel"));
            eat(&c);

            let (cut1, cut2) = if rng.below(2) == 0 { (frac(&mut rng), None) } else { let (x, y) = frac_pair(&mut rng); (x, Some(y)) };
            let (mut ca, mut cb) = (a.clone(), b.clone());
            layout.crossover_rel(&layout, &mut ca, &mut cb, cut1, cut2).expect("homologous, fractions in order");
            for f in &layout.frames {
                let len = f.head_len + f.tail_len + f.dc_len;
                let (l, h) = (frac_row(cut1, len), frac_row(cut2.unwrap_or(FRAC_ONE), len));
                for r in 0..len {
                    let p = GeneLayout::column_pos(f, r).expect("row");
                    let inside = r >= l && r < h;
                    assert_eq!((ca[p], cb[p]), if inside { (b[p], a[p]) } else { (a[p], b[p]) }, "{}: cell {p}", what("crossover_rel"));
                }
            }
            legal(k, &layout, &ca, &consts, &what("crossover_rel first child"));
            legal(k, &layout, &cb, &consts, &what("crossover_rel second child"));
            eat(&ca);
            eat(&cb);
            relative += 1;
        }
        println!("{label}: {inverted} row inversions, {transposed} row IS, {ris} row RIS, {crossed} row crossovers, {relative} genes through the three relative operators, every child legal");
        assert!(inverted > 0 && transposed > 0 && ris > 0 && crossed > 0, "{label}: every operator was applied");
        fp
    }

    #[test]
    fn row_operators_keep_every_symbol_in_its_column_and_every_child_legal() {
        for (k, label) in kingdoms() {
            row_operators(&k, label, 0x7ab1e ^ label.len() as u64, ROW_GENES);
        }
    }

    /// Several thousand genes of the pathfinder and histogram views: the kingdoms the row operators were made for,
    /// with ragged columns and a Dv lane.
    #[test]
    fn row_operators_keep_futhark_genes_legal() {
        use crate::futhark_table::GeneRng;
        for (name, n, genes) in [("pathfinder", 13, 60), ("histogram", 4, 300)] {
            let fk = crate::futhark_sig::kingdom6(name).expect("kingdom");
            let k = fk.contract.as_ref().clone();
            let consts = fk.const_len();
            let layout = GeneLayout::from_program_size(&k, fk.entry.output, n).expect("layout");
            let mut rng = GeneRng::new(0x5eed);
            for g in 0..genes {
                let a = crate::futhark_sig::random_gene(&k, &layout, &consts, &mut rng).expect("gene");
                let b = crate::futhark_sig::random_gene(&k, &layout, &consts, &mut rng).expect("gene");
                let what = |op: &str| format!("{name} gene {g} {op}");
                let rows = layout.domain_rows(Domain::Head);
                let mut c = a.clone();
                let lo = rng.below(rows - 1);
                layout.invert_rows(&mut c, Domain::Head, lo, lo + 2 + rng.below(rows - lo - 1)).expect("rows");
                legal(&k, &layout, &c, &consts, &what("invert_rows"));
                let mut c = a.clone();
                layout.transpose_rows(&mut c, Domain::Head, rng.below(rows), 1 + rng.below(3), rng.below(rows)).expect("rows");
                legal(&k, &layout, &c, &consts, &what("IS"));
                if let Some(src) = layout.ris_row(&k, &a, 0) {
                    let mut c = a.clone();
                    layout.transpose_rows(&mut c, Domain::Head, src, 1 + rng.below(3), 0).expect("rows");
                    legal(&k, &layout, &c, &consts, &what("RIS"));
                }
                let (mut ca, mut cb) = (a.clone(), b.clone());
                let c1 = rng.below(layout.gene_rows());
                let cut2 = if rng.below(2) == 0 { None } else { Some(c1 + 1 + rng.below(layout.gene_rows() - c1)) };
                layout.crossover_rows(&layout, &mut ca, &mut cb, c1, cut2).expect("rows");
                legal(&k, &layout, &ca, &consts, &what("crossover_rows"));
                legal(&k, &layout, &cb, &consts, &what("crossover_rows"));
                let f = |rng: &mut GeneRng| rng.next_u64() >> 32;
                let mut c = a.clone();
                let lo = f(&mut rng);
                layout.invert_rel(&mut c, Domain::Head, lo, lo + 1 + rng.next_u64() % (FRAC_ONE - lo)).expect("fractions");
                legal(&k, &layout, &c, &consts, &what("invert_rel"));
                let mut c = a.clone();
                let src = f(&mut rng);
                layout.transpose_rel(&mut c, Domain::Head, src, (src + rng.next_u64() % (FRAC_ONE / 8)).min(FRAC_ONE), f(&mut rng)).expect("fractions");
                legal(&k, &layout, &c, &consts, &what("transpose_rel"));
                let (mut ca, mut cb) = (a.clone(), b.clone());
                layout.crossover_rel(&layout, &mut ca, &mut cb, f(&mut rng), None).expect("fractions");
                legal(&k, &layout, &ca, &consts, &what("crossover_rel"));
                legal(&k, &layout, &cb, &consts, &what("crossover_rel"));
            }
            let types = layout.frames.len() as f64;
            let mean_head = layout.frames.iter().map(|f| f.head_len).sum::<usize>() as f64 / types;
            let mean_frame = layout.frames.iter().map(|f| f.head_len + f.tail_len).sum::<usize>() as f64 / types;
            let mean_column = layout.frames.iter().map(|f| f.head_len + f.tail_len + f.dc_len).sum::<usize>() as f64 / types;
            let max_frame = layout.frames.iter().map(|f| f.head_len + f.tail_len).max().unwrap_or(0);
            println!(
                "{name}: {} types, head table {} rows (mean column {mean_head:.1}), head+tail {max_frame} rows (mean {mean_frame:.1}), gene table {} rows (mean column {mean_column:.1}), {genes} genes, every child legal",
                layout.frames.len(),
                layout.domain_rows(Domain::Head),
                layout.gene_rows()
            );
        }
    }

    /// With ONE type the table has one column and each row operator IS the one-type operator: the inversion is
    /// `GeneLayout::invert`, the transposition `GeneLayout::transpose` (and the classic IS), the crossover the classic
    /// one- and two-point recombination over the linear gene (head, tail, Dc), which inside one domain is
    /// `GeneLayout::swap_range`.
    #[test]
    fn with_one_type_each_row_operator_is_the_one_type_operator() {
        let k = sr_relations().check().expect("SR");
        assert_eq!(k.types().len(), 1, "SR is single-typed");
        let mut rng = Rng(0x0e_7e);
        let mut checked = 0usize;
        for g in 0..ROW_GENES {
            let n = SIZES[g % SIZES.len()];
            let layout = GeneLayout::from_program_size(&k, k.roots()[0], n).expect("layout");
            assert_eq!(layout.frames.len(), 1);
            let f = layout.frames[0].clone();
            assert_eq!(layout.gene_rows(), layout.width, "one column is the linear gene");
            let (a, b) = (random_gene(&k, &layout, &mut rng), random_gene(&k, &layout, &mut rng));
            for s in layout.spans() {
                if s.len >= 2 {
                    let lo = rng.below(s.len - 1);
                    let hi = lo + 2 + rng.below(s.len - lo - 1);
                    let (mut x, mut y) = (a.clone(), a.clone());
                    layout.invert_rows(&mut x, s.domain, lo, hi).expect("rows");
                    layout.invert(&mut y, s.off + lo, s.off + hi).expect("span");
                    assert_eq!(x, y, "SR gene {g}: invert_rows is invert in the {}", s.domain.name());
                }
                let (src, len, dst) = (rng.below(s.len), 1 + rng.below(3), rng.below(s.len));
                let (mut x, mut y) = (a.clone(), a.clone());
                layout.transpose_rows(&mut x, s.domain, src, len, dst).expect("rows");
                layout.transpose(&mut y, s.off + src, len.min(s.len - src), s.off + dst).expect("span");
                assert_eq!(x, y, "SR gene {g}: transpose_rows is transpose in the {}", s.domain.name());
                assert_eq!(&x[s.off..s.off + s.len], &classic_is(&a[s.off..s.off + s.len], src, len, dst)[..], "SR gene {g}: the classic IS");
                // Two-point inside one domain is the homologous swap of that range.
                let lo = rng.below(s.len);
                let hi = lo + 1 + rng.below(s.len - lo);
                let (mut xa, mut xb, mut ya, mut yb) = (a.clone(), b.clone(), a.clone(), b.clone());
                layout.crossover_rows(&layout, &mut xa, &mut xb, s.off + lo, Some(s.off + hi)).expect("rows");
                layout.swap_range(&layout, &mut ya, &mut yb, s.off + lo, s.off + hi).expect("span");
                assert_eq!((xa, xb), (ya, yb), "SR gene {g}: two-point inside the {} is swap_range", s.domain.name());
            }
            // One-point at c: the classic recombination, everything from position c on exchanged.
            let c = rng.below(layout.width);
            let (mut xa, mut xb, mut ya, mut yb) = (a.clone(), b.clone(), a.clone(), b.clone());
            layout.crossover_rows(&layout, &mut xa, &mut xb, c, None).expect("rows");
            ya[c..].swap_with_slice(&mut yb[c..]);
            assert_eq!((xa, xb), (ya, yb), "SR gene {g}: one-point at {c} is the classic one-point (head {}, tail {}, Dc {})", f.head_len, f.tail_len, f.dc_len);
            // Two-point across domains: the classic two-point.
            let c1 = rng.below(layout.width);
            let c2 = c1 + 1 + rng.below(layout.width - c1);
            let (mut xa, mut xb, mut ya, mut yb) = (a.clone(), b.clone(), a.clone(), b.clone());
            layout.crossover_rows(&layout, &mut xa, &mut xb, c1, Some(c2)).expect("rows");
            ya[c1..c2].swap_with_slice(&mut yb[c1..c2]);
            assert_eq!((xa, xb), (ya, yb), "SR gene {g}: two-point [{c1}, {c2}) is the classic two-point");
            // THE RELATIVE ROWS with one type: the fraction of any row gives that row, so each relative operator is the
            // absolute one, and so the classic one.
            let w = layout.width;
            let (mut xa, mut xb, mut ya, mut yb) = (a.clone(), b.clone(), a.clone(), b.clone());
            layout.crossover_rel(&layout, &mut xa, &mut xb, frac_of_row(c, w), None).expect("fractions");
            ya[c..].swap_with_slice(&mut yb[c..]);
            assert_eq!((xa, xb), (ya, yb), "SR gene {g}: relative one-point at {c} is the classic one-point");
            let (mut xa, mut xb, mut ya, mut yb) = (a.clone(), b.clone(), a.clone(), b.clone());
            layout.crossover_rel(&layout, &mut xa, &mut xb, frac_of_row(c1, w), Some(frac_of_row(c2, w))).expect("fractions");
            ya[c1..c2].swap_with_slice(&mut yb[c1..c2]);
            assert_eq!((xa, xb), (ya, yb), "SR gene {g}: relative two-point [{c1}, {c2}) is the classic two-point");
            for s in layout.spans() {
                if s.len >= 2 {
                    let lo = rng.below(s.len - 1);
                    let hi = lo + 2 + rng.below(s.len - lo - 1);
                    let (mut x, mut y) = (a.clone(), a.clone());
                    layout.invert_rel(&mut x, s.domain, frac_of_row(lo, s.len), frac_of_row(hi, s.len)).expect("fractions");
                    layout.invert(&mut y, s.off + lo, s.off + hi).expect("span");
                    assert_eq!(x, y, "SR gene {g}: invert_rel is invert in the {}", s.domain.name());
                }
                let (src, len, dst) = (rng.below(s.len), 1 + rng.below(3), rng.below(s.len));
                let run = len.min(s.len - src);
                let (mut x, mut y) = (a.clone(), a.clone());
                layout.transpose_rel(&mut x, s.domain, frac_of_row(src, s.len), frac_of_row(src + run, s.len), frac_of_row(dst, s.len)).expect("fractions");
                layout.transpose(&mut y, s.off + src, run, s.off + dst).expect("span");
                assert_eq!(x, y, "SR gene {g}: transpose_rel is transpose in the {}", s.domain.name());
            }
            checked += 1;
        }
        println!("SR: {checked} genes, every row operator equal to its one-type operator");
    }

    /// Ragged columns by hand: the two-type fixture with `h_S = 4`, `h_I = 2`. The gene holds its own positions, so
    /// each result names where every cell came from.
    #[test]
    fn ragged_columns_take_part_only_where_they_have_the_row() {
        let k = two_type();
        let heads: BTreeMap<Ty, usize> = [(Ty::S, 4), (Ty::I, 2)].into_iter().collect();
        let layout = GeneLayout::from_heads(&k, Ty::S, &heads).expect("layout");
        let (s, i) = (layout.frame(Ty::S).expect("S").clone(), layout.frame(Ty::I).expect("I").clone());
        assert_eq!((s.head_off, s.head_len, i.head_off, i.head_len), (0, 4, 4, 2));
        assert_eq!(layout.domain_rows(Domain::Head), 4);
        let id: Vec<u32> = (0..layout.width as u32).collect();
        let head = |g: &[u32]| (g[0..4].to_vec(), g[4..6].to_vec());

        // Rows [1, 4): S reverses rows 1..4; I has only row 1 of them, one cell, unchanged.
        let mut g = id.clone();
        assert_eq!(layout.invert_rows(&mut g, Domain::Head, 1, 4), Ok(1));
        assert_eq!(head(&g), (vec![0, 3, 2, 1], vec![4, 5]));
        assert_eq!(g[6..], id[6..], "the tails are another table");
        // Rows [0, 3): S reverses rows 0..3, I its rows 0..2.
        let mut g = id.clone();
        assert_eq!(layout.invert_rows(&mut g, Domain::Head, 0, 3), Ok(2));
        assert_eq!(head(&g), (vec![2, 1, 0, 3], vec![5, 4]));

        // IS of rows [2, 4) to row 0: S takes it; I has no row 2, unchanged.
        let mut g = id.clone();
        assert_eq!(layout.transpose_rows(&mut g, Domain::Head, 2, 2, 0), Ok(1));
        assert_eq!(head(&g), (vec![2, 3, 0, 1], vec![4, 5]));
        // IS of rows [1, 4) to row 0: S's run is three rows, I's the one row 1 it has.
        let mut g = id.clone();
        assert_eq!(layout.transpose_rows(&mut g, Domain::Head, 1, 3, 0), Ok(2));
        assert_eq!(head(&g), (vec![1, 2, 3, 0], vec![5, 4]));
        // IS to row 3: I has no row 3, unchanged.
        let mut g = id.clone();
        assert_eq!(layout.transpose_rows(&mut g, Domain::Head, 0, 1, 3), Ok(1));
        assert_eq!(head(&g), (vec![0, 1, 2, 0], vec![4, 5]));

        // One-point crossover at row 3 of the whole gene's table: S's head row 3 and its whole tail; I's column is
        // head (2 rows) then tail, so its tail from tail row 1 on.
        let (mut a, mut b) = (id.clone(), id.iter().map(|w| w + 1000).collect::<Vec<u32>>());
        let cells = layout.crossover_rows(&layout, &mut a, &mut b, 3, None).expect("rows");
        let mut want: Vec<usize> = vec![s.head_off + 3];
        want.extend(s.tail_off..s.tail_off + s.tail_len);
        want.extend(i.tail_off + 1..i.tail_off + i.tail_len);
        assert_eq!(cells, want.len());
        for p in 0..layout.width {
            let swapped = want.contains(&p);
            assert_eq!(a[p], if swapped { id[p] + 1000 } else { id[p] }, "cell {p}");
        }

        // RELATIVE rows: every column at its own row. Inversion of [1/4, 1): S (4 rows) rows [1, 4), I (2 rows)
        // rows [0, 2) -- I takes part, where by absolute rows it had one row and did not change.
        let mut g = id.clone();
        assert_eq!(layout.invert_rel(&mut g, Domain::Head, FRAC_ONE / 4, FRAC_ONE), Ok(2));
        assert_eq!(head(&g), (vec![0, 3, 2, 1], vec![5, 4]));
        // IS from 1/2 (S row 2, I row 1), a run to 1 (S rows [2, 4), I row 1), to row 0 of each.
        let mut g = id.clone();
        assert_eq!(layout.transpose_rel(&mut g, Domain::Head, FRAC_ONE / 2, FRAC_ONE, 0), Ok(2));
        assert_eq!(head(&g), (vec![2, 3, 0, 1], vec![5, 4]));
        // One-point at 1/2 of every whole column: S from row (4 + t_S) / 2, I from row (2 + t_I) / 2.
        let (mut a2, mut b2) = (id.clone(), id.iter().map(|w| w + 1000).collect::<Vec<u32>>());
        layout.crossover_rel(&layout, &mut a2, &mut b2, FRAC_ONE / 2, None).expect("fractions");
        for f in [&s, &i] {
            let len = f.head_len + f.tail_len;
            for r in 0..len {
                let p = GeneLayout::column_pos(f, r).expect("row");
                assert_eq!(a2[p], if r >= len / 2 { id[p] + 1000 } else { id[p] }, "{:?} row {r}", f.ty);
            }
        }
        let mut g = id.clone();
        assert!(matches!(layout.invert_rel(&mut g, Domain::Head, FRAC_ONE / 2, FRAC_ONE / 2), Err(LayoutDefect::FractionOutsideUnit { .. })));
        assert!(matches!(layout.transpose_rel(&mut g, Domain::Head, FRAC_ONE, FRAC_ONE, 0), Err(LayoutDefect::FractionOutsideUnit { .. })));
        assert_eq!(g, id, "a defect changes nothing");

        // Rows outside the table are our bug, named.
        let mut g = id.clone();
        assert!(matches!(layout.invert_rows(&mut g, Domain::Head, 2, 5), Err(LayoutDefect::RowsOutsideTable { .. })));
        assert!(matches!(layout.transpose_rows(&mut g, Domain::Head, 4, 1, 0), Err(LayoutDefect::RowsOutsideTable { .. })));
        assert!(matches!(layout.crossover_rows(&layout, &mut a, &mut b, 2, Some(2)), Err(LayoutDefect::RowsOutsideTable { .. })));
        assert_eq!(g, id, "a defect changes nothing");
    }

    /// The same seed gives the same children; another seed others.
    #[test]
    fn the_same_seed_gives_the_same_children() {
        let k = regex_relations().check().expect("regex");
        let (x, y, z) = (row_operators(&k, "regex", 42, 2000), row_operators(&k, "regex", 42, 2000), row_operators(&k, "regex", 43, 2000));
        assert_eq!(x, y, "one seed, one result");
        assert_ne!(x, z, "another seed, other children");
    }
}
