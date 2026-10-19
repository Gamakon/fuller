//! The design's host tests of multi-typed GEP (`MultiTypedGEP.md` section 7):
//! closure over 100,000 random genes per kingdom, the tail bound with its
//! measured maximum, one-type equality with classical Karva, and determinism.
//!
//! The genes are drawn valid by layout: a head cell is any kingdom symbol, a
//! tail cell of type `T` a terminal of `T`, a Dc cell an index into `C_T`.
//! Nothing here repairs a gene; a defect fails the test with its text.

use std::collections::BTreeMap;

use crate::geneframe::Ty;
use crate::kingdom_contract::{regex_relations, relations_from_rows, sr_relations, KingdomContract, KingdomRelations, SymbolId};
use crate::typed_gep::{express, Domain, Expressed, GeneLayout};

const GENES: usize = 100_000;
/// Length of every `C_T` in these tests.
const CONSTS: u32 = 5;
const HEADS: [usize; 4] = [1, 3, 8, 32];

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

/// A gene valid by layout, head cells drawn by `head`.
fn random_gene(k: &KingdomContract, layout: &GeneLayout, rng: &mut Rng, head: &mut dyn FnMut(&mut Rng) -> SymbolId) -> Vec<u32> {
    (0..layout.width)
        .map(|pos| {
            let key = layout.key(pos);
            match (key.domain, key.domain_type) {
                (Domain::Head, _) => head(rng),
                (Domain::Tail, Some(t)) => {
                    let ts = k.terminals_of(t);
                    ts[rng.below(ts.len())]
                }
                (Domain::Dc, _) => rng.below(CONSTS as usize) as u32,
                (Domain::Tail, None) => unreachable!("a tail has a type"),
            }
        })
        .collect()
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

/// Largest tail and Dc cursor seen per type.
#[derive(Default)]
struct Maxima {
    tail: BTreeMap<Ty, usize>,
    q: BTreeMap<Ty, usize>,
}

impl Maxima {
    fn take(&mut self, e: &Expressed) {
        for (t, v) in &e.reach.tail_used {
            let m = self.tail.entry(*t).or_default();
            *m = (*m).max(*v);
        }
        for (t, v) in &e.reach.q_used {
            let m = self.q.entry(*t).or_default();
            *m = (*m).max(*v);
        }
    }
    /// Every maximum is within its tail, and the worst ratio is returned.
    fn assert_below(&self, k: &KingdomContract, h: usize, root: Ty) {
        for (t, v) in &self.tail {
            assert!(*v <= k.tail_len(h, *t, root), "tail of {t:?} reached {v}, t_T = {}", k.tail_len(h, *t, root));
        }
        for (t, v) in &self.q {
            assert!(*v <= k.tail_len(h, *t, root), "q of {t:?} reached {v}, t_T = {}", k.tail_len(h, *t, root));
        }
    }
}

/// 100,000 random genes through the decoder; returns the per-(h, root) maxima.
fn closure(rel: &KingdomRelations, label: &str) -> usize {
    let k = rel.check().unwrap_or_else(|d| panic!("{label} fails its kingdom check: {d:?}"));
    let consts = const_lens(&k);
    let mut rng = Rng(0x5eed ^ label.len() as u64);
    let n_sym = k.n_symbols();
    let mut worst = 0usize;
    let mut layouts: BTreeMap<(usize, Ty), (GeneLayout, Maxima)> = BTreeMap::new();
    for g in 0..GENES {
        let h = HEADS[g % HEADS.len()];
        let root = k.roots()[rng.below(k.roots().len())];
        let (layout, maxima) = layouts.entry((h, root)).or_insert_with(|| (GeneLayout::new(&k, h, root).expect("layout"), Maxima::default()));
        let gene = random_gene(&k, layout, &mut rng, &mut |r| r.below(n_sym) as u32);
        let e = express(&k, layout, &gene, &consts).unwrap_or_else(|d| panic!("{label} gene {g}: {d}"));
        assert_well_typed(&k, root, &e, &format!("{label} gene {g}"));
        worst = worst.max(e.nodes.len());
        maxima.take(&e);
    }
    for ((h, root), (_, m)) in &layouts {
        m.assert_below(&k, *h, *root);
    }
    println!("{label}: {GENES} random genes closed, largest tree {worst} nodes, {} (h, root) layouts", layouts.len());
    worst
}

/// A head built to demand the most of `target`'s tail. The head is chosen in
/// the order the slots are met: for each slot of type `U` the function of
/// output `U` with the most arguments of type `target` (most arguments when
/// `target` is None), ties drawn. The walk then reads exactly these rows, since
/// the n-th slot of a type reads that type's n-th head row.
fn greedy_head(k: &KingdomContract, h: usize, root: Ty, target: Option<Ty>, rng: &mut Rng) -> Vec<SymbolId> {
    let mut head: Vec<SymbolId> = Vec::new();
    let mut queue = std::collections::VecDeque::from([root]);
    while let Some(u) = queue.pop_front() {
        if head.len() == h {
            break;
        }
        let score = |f: &SymbolId| match target {
            Some(t) => k.signature(*f).iter().filter(|a| **a == t).count(),
            None => k.signature(*f).len(),
        };
        let best = k.functions_of(u).iter().map(score).max();
        let Some(best) = best else { continue };
        let tied: Vec<SymbolId> = k.functions_of(u).iter().copied().filter(|f| score(f) == best).collect();
        let f = tied[rng.below(tied.len())];
        head.push(f);
        queue.extend(k.signature(f).iter().copied());
    }
    let n_fn: Vec<SymbolId> = (0..k.n_symbols() as SymbolId).filter(|s| !k.is_terminal(*s)).collect();
    while head.len() < h {
        head.push(n_fn[rng.below(n_fn.len())]);
    }
    head
}

/// Heads built to demand the most of each type, from every root. Records the
/// largest tail and Dc cursor and asserts the design bound holds.
fn adversarial(rel: &KingdomRelations, label: &str) -> Attained {
    let k = rel.check().expect("kingdom check");
    let consts = const_lens(&k);
    let mut rng = Rng(0xad7e);
    let mut out = Attained::default();
    for h in [3usize, 32] {
        for root in k.roots() {
            let layout = GeneLayout::new(&k, h, *root).expect("layout");
            let mut m = Maxima::default();
            let targets: Vec<Option<Ty>> = std::iter::once(None).chain(k.types().iter().filter(|t| k.a_t(**t) > 0).map(|t| Some(*t))).collect();
            for target in targets {
                let head = greedy_head(&k, h, *root, target, &mut rng);
                let mut it = head.into_iter();
                let gene = random_gene(&k, &layout, &mut rng, &mut |_| it.next().expect("h head cells"));
                let e = express(&k, &layout, &gene, &consts).unwrap_or_else(|d| panic!("{label} greedy head for {target:?}: {d}"));
                assert_well_typed(&k, *root, &e, label);
                m.take(&e);
            }
            m.assert_below(&k, h, *root);
            for t in k.frame_types(*root) {
                let (reached, bound) = (m.tail.get(&t).copied().unwrap_or(0), k.tail_len(h, t, *root));
                out.frames += 1;
                out.attained += usize::from(reached == bound);
                out.largest_ratio = out.largest_ratio.max(reached as f64 / bound as f64);
            }
        }
    }
    out
}

/// How close the greedy heads came to each tail's bound.
#[derive(Default)]
struct Attained {
    /// (h, root, frame type) combinations tried.
    frames: usize,
    /// Of those, the tails read to exactly `t_T`.
    attained: usize,
    largest_ratio: f64,
}

/// Drop what a kingdom cannot host today: functions with an input type that has
/// no terminal, and roots with none. This is a test-only view of the WGSL
/// kingdom until fuller supplies terminals for the store and `array<T>`
/// duals (CR section 9 item 3); the full kingdom fails its check and is
/// reported as failing.
#[cfg(feature = "wgsl")]
fn covered(rel: &KingdomRelations) -> KingdomRelations {
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
    closure(&sr_relations(), "SR");
}

#[test]
fn closure_regex() {
    closure(&regex_relations(), "regex");
}

#[cfg(feature = "wgsl")]
#[test]
fn closure_wgsl_covered_duals() {
    let full = crate::kingdom_contract::wgsl_relations();
    let sub = covered(&full);
    let k = sub.check().expect("the covered subset passes the check");
    println!("WGSL covered subset: {} of {} symbols, {} of {} roots, {} types", sub.names.len(), full.names.len(), sub.roots.len(), full.roots.len(), k.types().len());
    closure(&sub, "WGSL (covered)");
}

/// The bound, measured. For SR it is attained exactly (a head of binary
/// functions needs h + 1 leaves); for every kingdom no head built to demand the
/// most passes it (`Maxima::assert_below`).
#[test]
fn tail_bound_measured_maximum() {
    let sr = adversarial(&sr_relations(), "SR");
    assert_eq!(sr.attained, sr.frames, "SR: t_F is attained by an all-binary head");
    println!("SR: {}/{} tails read to exactly t_T", sr.attained, sr.frames);
    let re = adversarial(&regex_relations(), "regex");
    println!("regex: {}/{} tails read to exactly t_T, largest ratio {:.2}", re.attained, re.frames, re.largest_ratio);
    #[cfg(feature = "wgsl")]
    {
        let w = adversarial(&covered(&crate::kingdom_contract::wgsl_relations()), "WGSL (covered)");
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
/// included.
#[test]
fn sr_one_type_output_equals_classical_karva() {
    let k = sr_relations().check().expect("SR");
    let consts = const_lens(&k);
    let mut rng = Rng(0x4a12);
    let n_sym = k.n_symbols();
    let mut hist = 0usize;
    for g in 0..GENES {
        let h = HEADS[g % HEADS.len()];
        let layout = GeneLayout::new(&k, h, Ty::F).expect("layout");
        let gene = random_gene(&k, &layout, &mut rng, &mut |r| r.below(n_sym) as u32);
        let typed = express(&k, &layout, &gene, &consts).unwrap_or_else(|d| panic!("gene {g}: {d}"));
        let dc_off = layout.frame(Ty::F).and_then(|f| f.dc_off).expect("F has a Dc");
        let want = classical_karva(&k, &gene, dc_off);
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

/// Same seed, same genes, same trees, run twice over the three kingdoms.
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
                    let layout = GeneLayout::new(&k, HEADS[g % HEADS.len()], k.roots()[0]).expect("layout");
                    let n_sym = k.n_symbols();
                    let gene = random_gene(&k, &layout, &mut rng, &mut |r| r.below(n_sym) as u32);
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

// ---------------------------------------------------------------------------
// The DuckDB expression query of the design (section 3), run on the fixture
// ---------------------------------------------------------------------------

/// The design's query, verbatim from `MultiTypedGEP.md` section 3: a view and a
/// recursive query that binds the chromosome id.
const WALK_SQL: &str = include_str!("../nucleotable/sql/typed_gep_expression.sql");

/// Runs the setup script, the view and the query for each chromosome and prints
/// `{chromosome_id: rows}` as JSON. Reads argv: setup path, then chromosome ids.
const DRIVER: &str = r#"
import duckdb, json, sys
view, query = open(sys.argv[1]).read().split("\n\nWITH RECURSIVE", 1)
query = "WITH RECURSIVE" + query
con = duckdb.connect(":memory:")
con.execute(open(sys.argv[2]).read())
con.execute(view)
print(json.dumps({c: con.execute(query, [int(c)]).fetchall() for c in sys.argv[3:]}))
"#;

fn sql_str(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// One gene to load: where it lives, its kingdom and its words.
struct Loaded<'a> {
    chromosome: u64,
    gene: u64,
    kingdom: &'a str,
    k: &'a KingdomContract,
    layout: GeneLayout,
    words: Vec<u32>,
}

fn setup_sql(kingdoms: &[(&str, &KingdomContract)], genes: &[Loaded]) -> String {
    let mut s = String::from(
        "CREATE TABLE functions(kingdom_id VARCHAR, symbol VARCHAR, output_type VARCHAR);\n\
         CREATE TABLE terminals(kingdom_id VARCHAR, symbol VARCHAR, output_type VARCHAR);\n\
         CREATE TABLE argument_signature(kingdom_id VARCHAR, symbol VARCHAR, arg_index BIGINT, required_type VARCHAR);\n\
         CREATE TABLE genes(chromosome_id BIGINT, gene_id BIGINT, kingdom_id VARCHAR, root_type VARCHAR);\n\
         CREATE TABLE geneframe(chromosome_id BIGINT, gene_id BIGINT, kingdom_id VARCHAR, domain VARCHAR, domain_type VARCHAR, domain_seq_id BIGINT, symbol VARCHAR, dc_index BIGINT);\n",
    );
    for (name, k) in kingdoms {
        for sym in 0..k.n_symbols() as SymbolId {
            let table = if k.is_terminal(sym) { "terminals" } else { "functions" };
            s.push_str(&format!("INSERT INTO {table} VALUES ({}, {}, {});\n", sql_str(name), sql_str(k.name(sym)), sql_str(&format!("{:?}", k.output(sym)))));
            for (i, t) in k.signature(sym).iter().enumerate() {
                s.push_str(&format!("INSERT INTO argument_signature VALUES ({}, {}, {i}, {});\n", sql_str(name), sql_str(k.name(sym)), sql_str(&format!("{t:?}"))));
            }
        }
    }
    for g in genes {
        s.push_str(&format!("INSERT INTO genes VALUES ({}, {}, {}, {});\n", g.chromosome, g.gene, sql_str(g.kingdom), sql_str(&format!("{:?}", g.layout.root))));
        let rows: Vec<String> = g
            .layout
            .rows(g.chromosome, g.gene, &g.words)
            .iter()
            .map(|r| {
                format!(
                    "({}, {}, {}, {}, {}, {}, {}, {})",
                    r.chromosome_id,
                    r.gene_id,
                    sql_str(g.kingdom),
                    sql_str(r.key.domain.name()),
                    r.key.domain_type.map_or("NULL".to_string(), |t| sql_str(&format!("{t:?}"))),
                    r.key.domain_seq_id,
                    r.symbol.map_or("NULL".to_string(), |x| sql_str(g.k.name(x))),
                    r.dc_index.map_or("NULL".to_string(), |x| x.to_string())
                )
            })
            .collect();
        s.push_str(&format!("INSERT INTO geneframe VALUES {};\n", rows.join(", ")));
    }
    s
}

/// A node as the query reports it: step, type, frame_k, symbol, depth, parent_step, arg_index.
fn host_row(k: &KingdomContract, n: &crate::typed_gep::Node) -> String {
    format!("{} {:?} {} {} {} {:?} {:?}", n.step, n.ty, n.frame_k, k.name(n.symbol), n.depth, n.parent_step, n.arg_index)
}

fn sql_row(r: &serde_json::Value) -> String {
    let opt = |v: &serde_json::Value| if v.is_null() { "None".to_string() } else { format!("Some({v})") };
    format!("{} {} {} {} {} {} {}", r[0], r[1].as_str().unwrap_or("?"), r[2], r[3].as_str().unwrap_or("?"), r[4], opt(&r[5]), opt(&r[6]))
}

/// The design's query, run in DuckDB on the two-type fixture (two genes of one
/// chromosome and a gene of another, the same gene id in both, so a frame that
/// leaked across a gene or chromosome would show) and on 100 random regex genes
/// over 20 chromosomes: every row equals the host decoder's node. Skipped, with
/// the reason printed, when no interpreter has the `duckdb` module
/// (`FULLER_DUCKDB_PYTHON` names one).
#[test]
fn duckdb_expression_query_equals_the_host_decoder() {
    use std::process::Command;
    let python = std::env::var("FULLER_DUCKDB_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let probe = Command::new(&python).args(["-I", "-c", "import duckdb"]).output();
    if !probe.is_ok_and(|o| o.status.success()) {
        eprintln!("SKIPPED duckdb_expression_query_equals_the_host_decoder: `{python} -I -c 'import duckdb'` fails; set FULLER_DUCKDB_PYTHON");
        return;
    }
    let (left, s, i) = (0u32, 1u32, 2u32);
    let two = relations_from_rows(
        "two-type",
        vec![Ty::S],
        &[("left".to_string(), vec![Ty::S, Ty::I], Ty::S), ("s".to_string(), vec![], Ty::S), ("i".to_string(), vec![], Ty::I)],
        &[],
    )
    .check()
    .expect("fixture");
    let re = regex_relations().check().expect("regex");
    let mut genes: Vec<Loaded> = Vec::new();
    let mut fixture = |chromosome: u64, gene: u64, head: [u32; 3]| {
        let layout = GeneLayout::new(&two, 3, Ty::S).expect("layout");
        let mut words = head.to_vec();
        words.extend([s; 4]);
        words.extend([i; 3]);
        genes.push(Loaded { chromosome, gene, kingdom: "two-type", k: &two, layout, words });
    };
    fixture(1, 0, [left, i, s]); // the design's table
    fixture(1, 1, [left, left, s]);
    fixture(2, 0, [left, s, i]);
    let mut rng = Rng(0xd0c);
    let consts = const_lens(&re);
    for chromosome in 3..23u64 {
        for gene in 0..5u64 {
            let layout = GeneLayout::new(&re, 8, Ty::Pattern).expect("layout");
            let n_sym = re.n_symbols();
            let words = random_gene(&re, &layout, &mut rng, &mut |r| r.below(n_sym) as u32);
            genes.push(Loaded { chromosome, gene, kingdom: "regex", k: &re, layout, words });
        }
    }
    let dir = std::env::temp_dir().join(format!("fuller_typed_gep_sql_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let (driver, walk, setup) = (dir.join("driver.py"), dir.join("walk.sql"), dir.join("setup.sql"));
    std::fs::write(&driver, DRIVER).expect("write driver");
    std::fs::write(&walk, WALK_SQL).expect("write walk");
    std::fs::write(&setup, setup_sql(&[("two-type", &two), ("regex", &re)], &genes)).expect("write setup");
    let mut chroms: Vec<u64> = genes.iter().map(|g| g.chromosome).collect();
    chroms.dedup();
    let out = Command::new(&python)
        .arg("-I")
        .arg(&driver)
        .arg(&walk)
        .arg(&setup)
        .args(chroms.iter().map(u64::to_string))
        .output()
        .expect("run the driver");
    assert!(out.status.success(), "duckdb driver failed: {}", String::from_utf8_lossy(&out.stderr));
    let result: serde_json::Value = serde_json::from_slice(&out.stdout).expect("driver prints JSON");
    let mut compared = 0usize;
    for c in &chroms {
        let rows = result[c.to_string()].as_array().unwrap_or_else(|| panic!("chromosome {c} has no result")).clone();
        // The query orders by gene_id then step and does not name the gene, so
        // compare chromosome by chromosome: the host's nodes in the same order.
        let mut want: Vec<(u64, String)> = Vec::new();
        for g in genes.iter().filter(|g| g.chromosome == *c) {
            let e = express(g.k, &g.layout, &g.words, &consts).unwrap_or_else(|d| panic!("chromosome {c} gene {}: {d}", g.gene));
            want.extend(e.nodes.iter().map(|n| (g.gene, host_row(g.k, n))));
        }
        want.sort_by_key(|(gene, row)| (*gene, row.split(' ').next().and_then(|x| x.parse::<usize>().ok()).unwrap_or(0)));
        let got: Vec<String> = rows.iter().map(sql_row).collect();
        let want: Vec<String> = want.into_iter().map(|(_, r)| r).collect();
        assert_eq!(got, want, "chromosome {c}: SQL and host decoder differ");
        compared += got.len();
    }
    // The design's table, as the SQL returned it.
    let first = result["1"].as_array().expect("chromosome 1");
    let table: Vec<String> = first.iter().take(3).map(sql_row).collect();
    assert_eq!(table, vec!["1 S 0 left 0 None None", "2 S 1 s 1 Some(1) Some(0)", "3 I 0 i 1 Some(1) Some(1)"]);
    println!("DuckDB query equals the host decoder on {} genes in {} chromosomes, {compared} nodes", genes.len(), chroms.len());
    std::fs::remove_file(&driver).expect("clean");
    std::fs::remove_file(&walk).expect("clean");
    std::fs::remove_file(&setup).expect("clean");
    std::fs::remove_dir(&dir).expect("clean");
}
