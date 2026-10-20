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

// ---------------------------------------------------------------------------
// The TSR kingdoms (plan F3): the same generic tests over the generated tables
// ---------------------------------------------------------------------------

mod tsr {
    use super::*;
    use crate::kingdom_contract::{report, tsr_relations};
    use crate::tsr_kingdom::{TsrKind, TsrKingdom, TsrVariant};
    use crate::typed_gep::required_heads;

    /// The two kingdoms the plan's arms B and C use, each over 1 and 9 input variables.
    const ARMS: [TsrVariant; 2] = [TsrVariant::Order, TsrVariant::OrderDepth];
    const INPUTS: [usize; 2] = [1, 9];

    fn rel(variant: TsrVariant, n_inputs: usize) -> KingdomRelations {
        tsr_relations(variant, n_inputs).unwrap_or_else(|e| panic!("{}: {e}", variant.dir()))
    }

    fn label(variant: TsrVariant, n_inputs: usize) -> String {
        format!("TSR {} x{n_inputs}", variant.dir())
    }

    /// Plan test 1: 100,000 random genes, every expressed node typed as its slot.
    #[test]
    fn tsr_closure_over_random_genes() {
        for variant in ARMS {
            for n in INPUTS {
                closure_n(&rel(variant, n), &label(variant, n), &[8, 17], GENES);
            }
        }
    }

    /// Plan test 2, mutation: 100,000 domain-local edits each.
    #[test]
    fn tsr_locality_of_mutation() {
        for variant in ARMS {
            for n in INPUTS {
                locality(&rel(variant, n), &label(variant, n), &[17]);
            }
        }
    }

    /// Plan test 2, crossover: 100,000 homologous exchanges each.
    #[test]
    fn tsr_homologous_crossover() {
        for variant in ARMS {
            for n in INPUTS {
                crossover(&rel(variant, n), &label(variant, n), &[17]);
            }
        }
    }

    /// Plan test 6: the widths the generator and the plan pin.
    #[test]
    fn tsr_widths_are_the_generators() {
        let pinned = [
            (TsrVariant::Single, [(8, 47), (17, 101), (32, 191)]),
            (TsrVariant::Order, [(8, 1_622), (17, 3_656), (32, 7_046)]),
            (TsrVariant::Depth, [(8, 252), (17, 558), (32, 1_068)]),
            (TsrVariant::OrderDepth, [(8, 9_244), (17, 20_980), (32, 40_540)]),
            (TsrVariant::OrderDepthWindows, [(8, 9_244), (17, 20_980), (32, 40_540)]),
        ];
        for (variant, widths) in pinned {
            for n_inputs in INPUTS {
                let k = rel(variant, n_inputs).check().unwrap_or_else(|d| panic!("{}: {d:?}", variant.dir()));
                let root = k.roots()[0];
                for (n, width) in widths {
                    let heads = k.heads_from_program_size(root, n).expect("heads");
                    assert_eq!(k.bounds(root, &heads).expect("bounds").width(), width, "{} N = {n}", variant.dir());
                    assert_eq!(GeneLayout::from_program_size(&k, root, n).expect("layout").width, width, "{} N = {n} layout", variant.dir());
                }
            }
        }
    }

    /// Plan test 4, host side: the single kingdom's rows are `SymbolTable::wide`'s 18 operators over one
    /// dual, and its contract at N = 17 gives h = 33, t = d = 34, width 101.
    #[test]
    fn tsr_single_is_the_classical_wide_table() {
        let wide = [
            ("add", "Add", 2), ("sub", "Sub", 2), ("mul", "Mul", 2), ("pdiv", "ProtectedDiv", 2), ("sin", "Sin", 1), ("cos", "Cos", 1),
            ("tanh", "Tanh", 1), ("psqrt", "ProtectedSqrt", 1), ("plog", "ProtectedLog", 1), ("pexp", "ProtectedExp", 1), ("neg", "Neg", 1),
            ("abs", "Abs", 1), ("pow2", "Pow2", 1), ("pow3", "Pow3", 1), ("pinv", "ProtectedInv", 1), ("tan", "Tan", 1),
            ("pasin", "ProtectedAsin", 1), ("pacos", "ProtectedAcos", 1),
        ];
        let tsr = TsrKingdom::load(TsrVariant::Single, 9).expect("loads");
        assert_eq!(tsr.duals.len(), 1);
        let one = Ty::Tsr(0);
        let functions: Vec<_> = tsr.rows.iter().filter(|r| !r.terminal()).collect();
        assert_eq!(functions.len(), wide.len());
        for (r, (sem, ctor, arity)) in functions.iter().zip(wide) {
            assert_eq!((r.semantic_id.as_str(), r.constructor.as_str(), r.inputs.len(), r.output), (sem, ctor, arity, one));
            assert!(r.inputs.iter().all(|t| *t == one), "{sem}: every argument is the one dual");
            assert_eq!(r.kind, TsrKind::Function);
        }
        let terminals: Vec<_> = tsr.rows.iter().filter(|r| r.terminal()).collect();
        assert_eq!(terminals.iter().filter(|r| matches!(r.kind, TsrKind::Var(_))).count(), 9);
        assert_eq!(terminals.iter().filter(|r| r.kind == TsrKind::Const).count(), 1);
        let k = rel(TsrVariant::Single, 9).check().expect("check");
        assert_eq!(k.n_max(), 2);
        let b = k.bounds(one, &k.heads_from_program_size(one, 17).expect("heads")).expect("bounds");
        let f = b.get(one).expect("the one dual");
        assert_eq!((f.h, f.t, f.d, b.width()), (33, 34, 34, 101));
        assert!(f.constant);
    }

    /// Plan test 6: the check names the defect when a leaf exists only at the bottom dual. The generator
    /// does not write that variant, so the rows are cut here from the loaded kingdom: every terminal off
    /// `duals[0]` is dropped, leaving each other dual (33 of 34 in `order_depth`, 11 of 12 in `order`)
    /// demanded and without a terminal.
    #[test]
    fn tsr_exact_leaf_variant_fails_the_check_by_name() {
        for (variant, defects) in [(TsrVariant::Order, 11usize), (TsrVariant::OrderDepth, 33)] {
            let k = TsrKingdom::load(variant, 2).expect("loads");
            let bottom = Ty::Tsr(0);
            let rows: Vec<(String, Vec<Ty>, Ty)> = k.rows.iter().filter(|r| !r.terminal() || r.output == bottom).map(|r| (r.name.clone(), r.inputs.clone(), r.output)).collect();
            let constants: Vec<&str> = k.rows.iter().filter(|r| r.kind == TsrKind::Const && r.output == bottom).map(|r| r.name.as_str()).collect();
            let r = relations_from_rows("TSR exact-leaf", vec![k.root().expect("root")], &rows, &constants);
            let d = r.check().expect_err("a dual without a terminal must fail the check");
            assert_eq!(d.len(), defects, "{}: {d:?}", variant.dir());
            assert!(d.iter().all(|m| m.starts_with("our software has a bug: ") && m.contains("no terminal")), "{}: {d:?}", variant.dir());
            assert!(report(&r, 17).contains(&format!("CHECK FAILED, {defects} defects")), "{}", variant.dir());
            // the real kingdom has a terminal at every dual
            assert!(rel(variant, 2).check().is_ok());
        }
    }

    /// Plan test 6, bound exactness. For each type a layout holds, a gene built to demand the most of that
    /// type (`greedy_gene`) is expressed; the frame, tail and Dc reads never pass `min(P_T, L + h_T)`, `t_T`, `d_T`,
    /// and the head rows the tree needs (`required_heads`) never pass `h_T`.
    fn exactness(variant: TsrVariant, n_inputs: usize, sizes: &[usize]) -> (usize, usize, usize, usize) {
        let k = rel(variant, n_inputs).check().expect("check");
        let consts = const_lens(&k);
        let root = k.roots()[0];
        let mut rng = Rng(0x7512 ^ variant as u64);
        // (types checked, frames read to P_T, tails read to t_T, head demands equal to h_T)
        let mut out = (0usize, 0usize, 0usize, 0usize);
        for n in sizes {
            let layout = GeneLayout::from_program_size(&k, root, *n).expect("layout");
            let bounds = k.bounds(root, &layout.head_lens()).expect("bounds");
            for t in k.reachable(root) {
                let gene = greedy_gene(&k, &layout, Some(t), &mut rng);
                let e = express(&k, &layout, &gene, &consts).unwrap_or_else(|d| panic!("{} N = {n} target {t:?}: {d}", variant.dir()));
                assert_well_typed(&k, root, &e, &label(variant, n_inputs));
                let (f, b) = (layout.frame(t).expect("frame"), bounds.get(t).expect("bound"));
                assert_eq!((f.head_len, f.tail_len, f.dc_len), (b.h, b.t, if b.constant { b.d } else { 0 }), "{} N = {n} {t:?}: layout is the bound table", variant.dir());
                assert_eq!(b.h + b.t, b.p.min(bounds.l + b.h), "{} N = {n} {t:?}: the frame is min(P_T, L + h_T) rows", variant.dir());
                let frame = e.reach.frame_used.get(&t).copied().unwrap_or(0);
                let tail = e.reach.tail_used.get(&t).copied().unwrap_or(0);
                assert!(frame <= b.h + b.t && tail <= b.t, "{} N = {n} {t:?}: frame {frame} of {}, tail {tail} of t {}", variant.dir(), b.h + b.t, b.t);
                let need = required_heads(&k, &e.tree()).get(&t).copied().unwrap_or(0);
                assert!(need <= b.h, "{} N = {n} {t:?}: head demand {need} past h {}", variant.dir(), b.h);
                out.0 += 1;
                out.1 += usize::from(frame == b.h + b.t);
                out.2 += usize::from(tail == b.t);
                out.3 += usize::from(need == b.h);
            }
        }
        out
    }

    /// The bounds hold in every case. The head bound `h_T` is attained for every (N, type): the chain of N
    /// functions that gives each its last argument of the type needs exactly `h_T` head rows. `P_T` and
    /// `L + h_T` are upper bounds taken per producer and over the whole program, and a greedy gene (most
    /// arguments of the type in each head row) reads them exactly for only some types; the single kingdom,
    /// one type, reads every tail to exactly `t_T`.
    #[test]
    fn tsr_bounds_are_never_passed_and_head_chain_attains_h() {
        for variant in ARMS {
            for n_inputs in INPUTS {
                let (types, p, t, h) = exactness(variant, n_inputs, &[1, 2, 3, 8, 17]);
                println!("{}: {types} (N, type) cases; frame read to its bound in {p}, tail read to t_T in {t}, head demand equal to h_T in {h}", label(variant, n_inputs));
                assert_eq!(h, types, "{}: the head chain attains h_T for every type", label(variant, n_inputs));
            }
        }
        let single = adversarial(&rel(TsrVariant::Single, 9), "TSR single", &[1, 2, 3, 8, 17, 32]);
        assert_eq!(single.attained, single.frames, "TSR single: t_F is attained by an all-binary head");
    }
}
