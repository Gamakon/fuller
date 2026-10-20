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

/// Every symbol outputting `ty`: the alphabet of `head_ty`.
fn head_symbols(k: &KingdomContract, ty: Ty) -> Vec<SymbolId> {
    (0..k.n_symbols() as SymbolId).filter(|s| k.output(*s) == ty).collect()
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
    let alpha = if s.domain == Domain::Head { head_symbols(k, s.ty) } else { Vec::new() };
    for cell in &mut gene[s.off..s.off + s.len] {
        *cell = match s.domain {
            Domain::Head => alpha[rng.below(alpha.len())],
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

/// Drop what a kingdom cannot host today: functions with an input type that has
/// no terminal, and roots with none. This is a test-only view of the older WGSL
/// kingdom (`kingdoms/wgsl`), which fails its own check on the store and
/// `array<T>` duals and is reported as failing.
#[cfg(feature = "wgsl")]
fn covered(rel: &KingdomRelations) -> KingdomRelations {
    use crate::kingdom_contract::relations_from_rows;
    let n = rel.names.len();
    let mut out = vec![None; n];
    for (s, t) in &rel.symbol_outputs {
        out[*s as usize] = Some(*t);
    }
    let mut ins: Vec<Vec<(u32, Ty)>> = vec![Vec::new(); n];
    for (s, i, t) in &rel.argument_signature {
        ins[*s as usize].push((*i, *t));
    }
    let with_terminal: std::collections::BTreeSet<Ty> = (0..n).filter(|s| ins[*s].is_empty()).filter_map(|s| out[s]).collect();
    let mut rows = Vec::new();
    let mut constants = Vec::new();
    for s in 0..n {
        ins[s].sort_by_key(|(i, _)| *i);
        let args: Vec<Ty> = ins[s].iter().map(|(_, t)| *t).collect();
        if args.iter().all(|t| with_terminal.contains(t)) {
            if args.is_empty() && rel.constants.contains(&(s as SymbolId)) {
                constants.push(rel.names[s].clone());
            }
            rows.push((rel.names[s].clone(), args, out[s].expect("one output")));
        }
    }
    let roots: Vec<Ty> = rel.roots.iter().copied().filter(|t| with_terminal.contains(t)).collect();
    let c: Vec<&str> = constants.iter().map(String::as_str).collect();
    relations_from_rows(&rel.kingdom, roots, &rows, &c)
}

#[test]
fn closure_sr() {
    closure_n(&sr_relations(), "SR", &SIZES, GENES);
}

#[test]
fn closure_regex() {
    closure_n(&regex_relations(), "regex", &SIZES, GENES);
}

#[cfg(feature = "wgsl")]
#[test]
fn closure_wgsl_covered_duals() {
    let full = crate::kingdom_contract::wgsl_relations();
    let sub = covered(&full);
    let k = sub.check().expect("the covered subset passes the check");
    println!("WGSL covered subset: {} of {} symbols, {} of {} roots, {} types", sub.names.len(), full.names.len(), sub.roots.len(), full.roots.len(), k.types().len());
    closure_n(&sub, "WGSL (covered)", &SIZES[..3], 20_000);
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
    {
        let w = adversarial(&covered(&crate::kingdom_contract::wgsl_relations()), "WGSL (covered)", &SIZES[..3]);
        println!("WGSL (covered): {}/{} tails read to exactly t_T, largest ratio {:.2}", w.attained, w.frames, w.largest_ratio);
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
fn locality(rel: &KingdomRelations, label: &str) {
    let k = rel.check().expect("check");
    let consts = const_lens(&k);
    let mut rng = Rng(0x10ca ^ label.len() as u64);
    let mut edits = 0usize;
    for g in 0..GENES {
        let n = SIZES[g % SIZES.len()];
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
    locality(&sr_relations(), "SR");
}

#[test]
fn locality_regex() {
    locality(&regex_relations(), "regex");
}

/// Homologous crossover: two genes of one layout exchange a range inside one
/// domain of one type; the cells outside the range are untouched, the range
/// in each child is the other parent's, and both children express. 100,000 per kingdom.
fn crossover(rel: &KingdomRelations, label: &str) {
    let k = rel.check().expect("check");
    let consts = const_lens(&k);
    let mut rng = Rng(0xc0de ^ label.len() as u64);
    let mut swaps = 0usize;
    for g in 0..GENES {
        let n = SIZES[g % SIZES.len()];
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
    crossover(&sr_relations(), "SR");
}

#[test]
fn crossover_regex() {
    crossover(&regex_relations(), "regex");
}

