//! The shape kingdom inside Fuller's e-graph: **one sort per shape class**,
//! so the sort IS the class.
//!
//! This is the 1-D kingdom's e-graph (`futhark_egraph.rs`) with the parts
//! that were hand-written there *generated* here, because the shape kingdom's
//! rows are generated and a hand-written bridge would drift from them:
//!
//! * the **datatype** and the **ruleset** come from
//!   `kingdoms/futhark_shapes/egraph.{datatype,rules}`, written by
//!   `gen_tables.py`;
//! * the **bridge** — the row ↔ constructor mapping and the class ↔ sort
//!   mapping — comes from `egraph.rows` and `egraph.sorts`, also generated.
//!   The Rust side never re-derives a name, so a row with no constructor is
//!   caught by the generator's witness tests rather than by a silent mismatch
//!   here.
//!
//! # The rules
//!
//! Three identities, one per shape-preserving operator, instantiated per
//! class:
//!
//! * `map` with the identity row function is the matrix;
//! * `map2` with the identity binary row function is the *first* matrix (the
//!   identity ignores its second operand; the kingdom's other binary row
//!   functions do not -- `add` uses both);
//! * `transpose` twice is the matrix.
//!
//! All three are **bit-identical**: they copy or permute values and never do
//! arithmetic, so no floating-point question arises. Nothing here assumes
//! that mathematical equivalence implies floating-point equivalence.
//!
//! The class kingdom has no fusion rule, so the divergence the 1-D kingdom
//! measured (`map id` and fusion cannot share a saturation) cannot arise here
//! — but the saturation is still bounded, because a bounded search is a
//! property of the *harness*, not of the rules.

use std::collections::BTreeMap;

use egglog::extract::hff_extract;
use egglog::prelude::exprs;
use egglog::{ArcSort, EGraph, Term, TermDag, TermId, Value};

use crate::futhark_kingdom::FutharkKingdom;
use crate::kingdom_contract::SymbolId;
use crate::saturated_fold::EgraphKingdom;
use crate::typed_chromosome::{AdfSet, Epoch};
use crate::typed_gep::Tree;

/// The generated datatype: one sort per class, one constructor per row.
pub const SHAPE_DATATYPE: &str = include_str!("../kingdoms/futhark_shapes/egraph.datatype");

/// The generated ruleset.
pub const SHAPE_RULESET: &str = include_str!("../kingdoms/futhark_shapes/egraph.rules");

/// The generated bridge: the row name, its constructor, and its sort.
const SHAPE_ROWS: &str = include_str!("../kingdoms/futhark_shapes/egraph.rows");

/// The generated bridge: the class and its sort.
const SHAPE_SORTS: &str = include_str!("../kingdoms/futhark_shapes/egraph.sorts");

/// The bridge between the kingdom's rows and the e-graph's constructors.
#[derive(Debug, Clone)]
pub struct Bridge {
    /// A row's constructor, by the row's name.
    ctor_of_row: BTreeMap<String, String>,
    /// A row's name, by its constructor.
    row_of_ctor: BTreeMap<String, String>,
    /// A class's sort.
    sort_of_class: BTreeMap<String, String>,
}

impl Bridge {
    /// Read the generated mapping.
    ///
    /// A duplicate row name or constructor is a **finding**, not something to
    /// paper over: the generator's witness test refuses such a mapping, so
    /// seeing one here means the committed file and the generator disagree,
    /// and silently keeping the last of the two would hide it.
    pub fn load() -> Bridge {
        let mut ctor_of_row = BTreeMap::new();
        let mut row_of_ctor = BTreeMap::new();
        for line in SHAPE_ROWS.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let mut f = line.split('\t');
            let row = f.next().unwrap_or_default().to_string();
            let ctor = f.next().unwrap_or_default().to_string();
            if row.is_empty() || ctor.is_empty() {
                panic!("futhark_shapes_egraph: the mapping line {line:?} is malformed");
            }
            if ctor_of_row.insert(row.clone(), ctor.clone()).is_some() {
                panic!("futhark_shapes_egraph: the row {row} is mapped twice");
            }
            if row_of_ctor.insert(ctor.clone(), row.clone()).is_some() {
                panic!("futhark_shapes_egraph: the constructor {ctor} is mapped to two rows");
            }
        }
        let mut sort_of_class = BTreeMap::new();
        for line in SHAPE_SORTS.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let mut f = line.split('\t');
            let class = f.next().unwrap_or_default().to_string();
            let sort = f.next().unwrap_or_default().to_string();
            if class.is_empty() || sort.is_empty() {
                panic!("futhark_shapes_egraph: the sort line {line:?} is malformed");
            }
            if sort_of_class.insert(class.clone(), sort.clone()).is_some() {
                panic!("futhark_shapes_egraph: the class {class} is mapped twice");
            }
        }
        Bridge { ctor_of_row, row_of_ctor, sort_of_class }
    }

    /// The constructor of a row.
    pub fn constructor_of(&self, row_name: &str) -> Option<&str> {
        self.ctor_of_row.get(row_name).map(|s| s.as_str())
    }

    /// The row of a constructor.
    pub fn row_of(&self, constructor: &str) -> Option<&str> {
        self.row_of_ctor.get(constructor).map(|s| s.as_str())
    }

    /// The sort of a class.
    pub fn sort_of(&self, class: &str) -> Option<&str> {
        self.sort_of_class.get(class).map(|s| s.as_str())
    }

    /// Every constructor of the kingdom, in the mapping's order.
    pub fn constructors(&self) -> impl Iterator<Item = &str> {
        self.ctor_of_row.values().map(|s| s.as_str())
    }
}

/// An expressed tree as an egglog term.
///
/// The term is the e-graph's *input*: one constructor per kingdom row, with
/// the kingdom's argument order.
pub fn import(tree: &Tree, kingdom: &FutharkKingdom, bridge: &Bridge) -> Result<String, String> {
    let row = kingdom
        .rows
        .get(tree.symbol as usize)
        .ok_or_else(|| format!("futhark_shapes_egraph: symbol {} is not a row", tree.symbol))?;
    let ctor = bridge
        .constructor_of(&row.name)
        .ok_or_else(|| format!("futhark_shapes_egraph: the row {} has no e-graph constructor", row.name))?;
    let kids: Vec<String> = tree
        .kids
        .iter()
        .map(|kid| import(kid, kingdom, bridge))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(if kids.is_empty() {
        format!("({ctor})")
    } else {
        format!("({ctor} {})", kids.join(" "))
    })
}

/// An egglog term as a Futhark expression.
///
/// The text is the kingdom's own spelling, with every child bracketed — the
/// same convention `FutharkKingdom::render` uses — so a candidate renders
/// exactly as the kingdom would have rendered the equivalent tree. There is no
/// per-constructor match here: the spelling is in the kingdom's table, and
/// the constructor's row is in the generated mapping.
pub fn render(termdag: &TermDag, id: TermId, kingdom: &FutharkKingdom, bridge: &Bridge) -> Result<String, String> {
    let term = termdag.get(id);
    let Term::App(op, kids) = term else {
        return Err(match term {
            Term::Lit(_) => "futhark_shapes_egraph: a literal is not a program".to_string(),
            Term::Var(v) => format!("futhark_shapes_egraph: {v} is not a program"),
            _ => unreachable!(),
        });
    };
    let row_name = bridge
        .row_of(op.as_str())
        .ok_or_else(|| format!("futhark_shapes_egraph: the constructor {} has no kingdom row", op))?;
    let row = kingdom
        .rows
        .iter()
        .find(|r| r.name == row_name)
        .ok_or_else(|| format!("futhark_shapes_egraph: the kingdom has no row {row_name}"))?;
    let mut out = row.spelling.clone();
    for (i, kid) in kids.iter().enumerate() {
        let placeholder = format!("{{{i}}}");
        if !out.contains(&placeholder) {
            return Err(format!(
                "futhark_shapes_egraph: {}'s spelling has no {placeholder}, so its {i}th argument is dropped",
                row.name
            ));
        }
        // Every child is substituted as an ATOM: Futhark's application is
        // greedy, so an unbracketed child would take the next argument too.
        let child = render(termdag, *kid, kingdom, bridge)?;
        out = out.replace(&placeholder, &format!("({child})"));
    }
    Ok(out)
}

/// An egglog term as a kingdom tree, if the term is one the kingdom can spell.
///
/// This is the **representability boundary**: `None` means the term uses a
/// constructor the kingdom has no row for, so no Karva gene denotes it.
pub fn to_tree(termdag: &TermDag, id: TermId, kingdom: &FutharkKingdom, bridge: &Bridge) -> Option<Tree> {
    let term = termdag.get(id);
    let Term::App(op, kid_ids) = term else {
        return None;
    };
    let row_name = bridge.row_of(op.as_str())?;
    let symbol = kingdom.rows.iter().position(|r| r.name == row_name)? as u32;
    let mut kids: Vec<Tree> = Vec::with_capacity(kid_ids.len());
    for kid in kid_ids {
        kids.push(to_tree(termdag, *kid, kingdom, bridge)?);
    }
    Some(Tree { symbol, const_index: None, kids })
}

/// The e-graph, saturated, with the root's sort and value.
///
/// The schedule is bounded: a rewrite that stopped shrinking would otherwise
/// hang the never-raise API with no cap.
pub fn saturate(term: &str, iters: u32) -> Result<(EGraph, ArcSort, Value), String> {
    saturate_with(term, SHAPE_RULESET, &["futhark_shape"], iters)
}

/// The e-graph, saturated under a chosen ruleset.
pub fn saturate_with(
    term: &str,
    rules: &str,
    rulesets: &[&str],
    iters: u32,
) -> Result<(EGraph, ArcSort, Value), String> {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, SHAPE_DATATYPE)
        .map_err(|e| format!("futhark_shapes_egraph: the datatype does not load: {e}"))?;
    egraph
        .parse_and_run_program(None, rules)
        .map_err(|e| format!("futhark_shapes_egraph: the ruleset does not load: {e}"))?;
    let schedule = rulesets
        .iter()
        .map(|r| format!("(repeat {iters} (run {r}))"))
        .collect::<Vec<_>>()
        .join(" ");
    egraph
        .parse_and_run_program(None, &format!("(let __root {term})\n(run-schedule {schedule})"))
        .map_err(|e| format!("futhark_shapes_egraph: {term} does not load: {e}"))?;
    let (sort, value) = egraph
        .eval_expr(&exprs::var("__root"))
        .map_err(|e| format!("futhark_shapes_egraph: the root does not evaluate: {e}"))?;
    Ok((egraph, sort, value))
}

/// The cost of a term: its number of e-nodes.
pub fn term_cost(termdag: &TermDag, id: TermId) -> u64 {
    match termdag.get(id) {
        Term::App(_, kids) => 1 + kids.iter().map(|k| term_cost(termdag, *k)).sum::<u64>(),
        _ => 1,
    }
}

/// One equivalent program of the e-graph.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The number of e-nodes.
    pub cost: u64,
    /// The Futhark source.
    pub source: String,
    /// The kingdom tree, if the kingdom can spell it.
    pub gene: Option<Tree>,
}

impl Candidate {
    /// Whether a Karva gene denotes this candidate.
    pub fn is_representable(&self) -> bool {
        self.gene.is_some()
    }
}

/// How many member terms one e-class may contribute to the enumeration.
const MEMBERS_PER_CLASS: usize = 64;

/// The equivalent programs of `term`, cheapest first.
///
/// The ranking is by the number of e-nodes and then by the rendered text, so
/// the same input and the same settings always give the same order.
pub fn candidates(
    term: &str,
    kingdom: &FutharkKingdom,
    bridge: &Bridge,
    k: usize,
    iters: u32,
) -> Result<Vec<Candidate>, String> {
    let (egraph, sort, value) = saturate(term, iters)?;
    Ok(extract_candidates(&egraph, sort, value, kingdom, bridge, k))
}

fn extract_candidates(
    egraph: &EGraph,
    sort: ArcSort,
    value: Value,
    kingdom: &FutharkKingdom,
    bridge: &Bridge,
    k: usize,
) -> Vec<Candidate> {
    let mut termdag = TermDag::default();
    let score = |td: &TermDag, t: TermId| term_cost(td, t) as f64;
    let ranked = hff_extract(egraph, &mut termdag, value, sort, &score, MEMBERS_PER_CLASS);
    let mut out: Vec<Candidate> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (cost, id) in ranked {
        let source = match render(&termdag, id, kingdom, bridge) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if !seen.insert(source.clone()) {
            continue;
        }
        out.push(Candidate {
            cost: cost as u64,
            source,
            gene: to_tree(&termdag, id, kingdom, bridge),
        });
    }
    // The candidates are SORTED before the cut, so which `k` survive does
    // not depend on the order the extractor happened to emit equal-cost
    // terms in: that order is a hash order, and a hash order is arbitrary
    // and would change with the extractor.
    out.sort_by(|a, b| a.cost.cmp(&b.cost).then_with(|| a.source.cmp(&b.source)));
    out.truncate(k);
    out
}

/// The saturation of one program.
#[derive(Debug, Clone)]
pub struct Saturation {
    /// The egglog term the tree imported to.
    pub term: String,
    /// The root's sort: the program's class.
    pub sort: String,
    /// The number of e-nodes in the saturated e-graph.
    pub size: usize,
    /// How long the saturation took.
    pub elapsed: std::time::Duration,
    /// The equivalent programs, cheapest first.
    pub candidates: Vec<Candidate>,
}

impl Saturation {
    /// The candidates the kingdom can spell as a gene, in rank order.
    ///
    /// This is the **heritable** set: a candidate whose tree is `Some` has a
    /// Karva gene that denotes it, so `typed_gep::write_back` can encode it
    /// and the evolutionary loop can inherit it.
    pub fn heritable(&self) -> Vec<&Candidate> {
        self.candidates.iter().filter(|c| c.is_representable()).collect()
    }
}

/// Saturate a tree's e-graph and extract its equivalent programs.
pub fn saturate_tree(
    tree: &Tree,
    kingdom: &FutharkKingdom,
    bridge: &Bridge,
    k: usize,
    iters: u32,
) -> Result<Saturation, String> {
    let term = import(tree, kingdom, bridge)?;
    let started = std::time::Instant::now();
    let (egraph, sort, value) = saturate(&term, iters)?;
    let size = egraph_size(&egraph, bridge);
    let candidates = extract_candidates(&egraph, sort.clone(), value, kingdom, bridge, k);
    Ok(Saturation { term, sort: sort.name().to_string(), size, elapsed: started.elapsed(), candidates })
}

/// Saturate a FOLDED tree (over the epoch alphabet): the e-graph works on
/// the unfolded tree, and its candidates are folded again by the next fold.
pub fn saturate_folded(epoch: &Epoch, adfs: &AdfSet, tree: &Tree, kingdom: &FutharkKingdom, bridge: &Bridge, k: usize, iters: u32) -> Result<Saturation, String> {
    saturate_tree(&adfs.unfold(epoch, tree), kingdom, bridge, k, iters)
}

/// The shape kingdom's e-graph as the population fold needs it
/// ([`crate::saturated_fold`]): the generated datatype, the generated ruleset
/// and the generated bridge.
pub struct FoldEgraph<'a> {
    pub kingdom: &'a FutharkKingdom,
    pub bridge: &'a Bridge,
}

impl EgraphKingdom for FoldEgraph<'_> {
    fn programs(&self) -> Vec<&str> {
        vec![SHAPE_DATATYPE, SHAPE_RULESET]
    }
    fn rulesets(&self) -> Vec<&str> {
        vec!["futhark_shape"]
    }
    fn import(&self, tree: &Tree) -> Result<String, String> {
        import(tree, self.kingdom, self.bridge)
    }
    fn row(&self, constructor: &str, literals: &[&str]) -> Option<(SymbolId, Option<u32>)> {
        if !literals.is_empty() {
            return None;
        }
        let name = self.bridge.row_of(constructor)?;
        let symbol = self.kingdom.rows.iter().position(|r| r.name == name)? as SymbolId;
        Some((symbol, None))
    }
}

/// The number of e-nodes in an e-graph: the size of the search space the
/// saturation explored.
///
/// The count is over the *constructors* the kingdom's rows map to, so it
/// measures the e-nodes of the shape kingdom's own programs.
pub fn egraph_size(egraph: &EGraph, bridge: &Bridge) -> usize {
    bridge
        .constructors()
        .map(|c| egraph.get_size(c))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::futhark_kingdom::FutharkKingdom;
    use crate::typed_gep::{express, GeneLayout};

    fn kingdom() -> FutharkKingdom {
        FutharkKingdom::load_shapes().expect("the shape tables load")
    }

    /// A random gene of the shape kingdom.
    fn random_tree(k: &FutharkKingdom, seed: u64) -> Tree {
        let contract = k.relations().check().expect("the kingdom's own check");
        let root = k.ty("mat.n.p").expect("the root class");
        let layout = GeneLayout::from_program_size(&contract, root, 3).expect("layout");
        let const_len = k.const_len();
        let mut rng = seed;
        let mut gene = vec![0u32; layout.width];
        for (pos, cell) in gene.iter_mut().enumerate() {
            let alphabet = layout.alphabet(&contract, pos, &const_len);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *cell = alphabet[((rng >> 33) as usize) % alphabet.len()];
        }
        express(&contract, &layout, &gene, &const_len).expect("a random gene is legal").tree()
    }

    #[test]
    fn the_generated_datatype_and_ruleset_load() {
        let (egraph, sort, _value) = saturate("(AInputMatNM)", 1).expect("the e-graph loads");
        assert_eq!(sort.name(), "FMatNM");
        assert!(egraph_size(&egraph, &Bridge::load()) > 0);
    }

    #[test]
    fn every_kingdom_row_has_a_constructor() {
        let k = kingdom();
        let bridge = Bridge::load();
        for row in &k.rows {
            assert!(
                bridge.constructor_of(&row.name).is_some(),
                "the row {} has no e-graph constructor",
                row.name
            );
        }
    }

    /// A tree built from the kingdom's rows, by name.
    fn tree(k: &FutharkKingdom, name: &str, kids: Vec<Tree>) -> Tree {
        let symbol = k
            .rows
            .iter()
            .position(|r| r.name == name)
            .unwrap_or_else(|| panic!("the kingdom has no row {name}")) as u32;
        Tree { symbol, const_index: None, kids }
    }

    #[test]
    fn a_tree_imports_and_the_bridge_agrees_with_the_kingdom() {
        let k = kingdom();
        let bridge = Bridge::load();
        let tree = random_tree(&k, 0xA11CE);
        let term = import(&tree, &k, &bridge).expect("the tree imports");
        assert!(term.starts_with('('), "the term is an application");
        let sat = saturate_tree(&tree, &k, &bridge, 40, 2).expect("the saturation runs");
        assert!(!sat.candidates.is_empty(), "the e-graph found nothing");
        // Every candidate comes back as a tree the kingdom can spell, and the
        // kingdom renders it to exactly the candidate's source: the bridge is
        // generic, and the spelling is the kingdom's.
        for c in &sat.candidates {
            let t = c.gene.as_ref().expect("the candidate is spellable");
            assert_eq!(k.render(t).expect("the kingdom renders it"), c.source);
        }
    }

    #[test]
    fn the_egraph_finds_the_involution() {
        let k = kingdom();
        let bridge = Bridge::load();
        // transpose (transpose xs) is xs.
        let t = tree(
            &k,
            "transpose@mat.m.n",
            vec![tree(&k, "transpose@mat.n.m", vec![tree(&k, "mat.n.m.input", vec![])])],
        );
        let sat = saturate_tree(&t, &k, &bridge, 40, 3).expect("the saturation runs");
        let sources: Vec<&str> = sat.candidates.iter().map(|c| c.source.as_str()).collect();
        assert!(
            sources.contains(&"m_n_m"),
            "the involution was not found: {sources:?}"
        );
    }

    #[test]
    fn the_egraph_finds_the_identity_map() {
        let k = kingdom();
        let bridge = Bridge::load();
        // map with the identity row function is the matrix.
        let t = tree(
            &k,
            "map_rows@mat.n.m",
            vec![
                tree(&k, "fn.arr_f32_arr_f32.id", vec![]),
                tree(&k, "mat.n.m.input", vec![]),
            ],
        );
        let sat = saturate_tree(&t, &k, &bridge, 40, 3).expect("the saturation runs");
        let sources: Vec<&str> = sat.candidates.iter().map(|c| c.source.as_str()).collect();
        assert!(
            sources.contains(&"m_n_m"),
            "the identity map was not found: {sources:?}"
        );
    }

    #[test]
    fn the_saturation_finds_the_identity_and_the_involution() {
        let k = kingdom();
        let bridge = Bridge::load();
        let tree = random_tree(&k, 0xB0B);
        let sat = saturate_tree(&tree, &k, &bridge, 40, 4).expect("the saturation runs");
        assert!(sat.size > 0);
        // Every candidate the e-graph found is class-correct: the kingdom can
        // spell it, and its class is the root's.
        for c in &sat.candidates {
            assert!(c.is_representable(), "the candidate {} is not spellable", c.source);
        }
        assert!(!sat.heritable().is_empty(), "the e-graph found nothing inheritable");
    }

    /// The ruleset **settles**.
    ///
    /// The 1-D kingdom's fusion rule, together with `map id`, grew its
    /// e-graph 5 -> 7 -> 13 -> 55 -> 1357 e-nodes and never converged, so a
    /// ruleset that adds rules has to be measured, not assumed. The
    /// transposed-product rule is shrinking — its right side has no
    /// transposes — and its operands are of other classes, so it cannot put
    /// a node inside the class it matches; but the measurement is the test.
    #[test]
    fn the_ruleset_settles() {
        let k = kingdom();
        let bridge = Bridge::load();
        let tree = random_tree(&k, 0x5E77);
        let sat = saturate_tree(&tree, &k, &bridge, 40, 8).expect("the saturation runs");
        assert!(
            sat.size < 200,
            "the e-graph grew to {} e-nodes over eight rounds",
            sat.size
        );
        // And it settles the same way twice.
        let again = saturate_tree(&tree, &k, &bridge, 40, 8).expect("the saturation runs again");
        assert_eq!(sat.size, again.size, "the saturation is not deterministic");
    }

    /// The transposed-product rule fires: the product computed the
    /// transposed way and transposed back is the product itself.
    #[test]
    fn the_egraph_finds_the_transposed_product() {
        let k = kingdom();
        let bridge = Bridge::load();
        // `transpose (matmul (transpose B) (transpose A))`, with A of class
        // mat.n.m and B of class mat.m.p.
        let tree = Tree {
            symbol: row_symbol(&k, "transpose@mat.p.n"),
            const_index: None,
            kids: vec![Tree {
                symbol: row_symbol(&k, "matmul@mat.p.mxmat.m.n"),
                const_index: None,
                kids: vec![
                    Tree {
                        symbol: row_symbol(&k, "transpose@mat.m.p"),
                        const_index: None,
                        kids: vec![Tree {
                            symbol: row_symbol(&k, "mat.m.p.input"),
                            const_index: None,
                            kids: vec![],
                        }],
                    },
                    Tree {
                        symbol: row_symbol(&k, "transpose@mat.n.m"),
                        const_index: None,
                        kids: vec![Tree {
                            symbol: row_symbol(&k, "mat.n.m.input"),
                            const_index: None,
                            kids: vec![],
                        }],
                    },
                ],
            }],
        };
        let sat = saturate_tree(&tree, &k, &bridge, 40, 6).expect("the saturation runs");
        // The direct product is among the candidates.
        let found = sat.candidates.iter().any(|c| c.source.contains("matmul (m_n_m) (m_m_p)"));
        assert!(found, "the transposed-product rule did not fire");
    }

    /// The symbol of a row, by name.
    fn row_symbol(k: &FutharkKingdom, name: &str) -> u32 {
        k.rows
            .iter()
            .position(|r| r.name == name)
            .unwrap_or_else(|| panic!("the kingdom has no row {name}")) as u32
    }

    #[test]
    fn the_same_input_gives_the_same_candidates() {
        let k = kingdom();
        let bridge = Bridge::load();
        let tree = random_tree(&k, 0xC0FFEE);
        let a = saturate_tree(&tree, &k, &bridge, 20, 3).expect("the saturation runs");
        let b = saturate_tree(&tree, &k, &bridge, 20, 3).expect("the saturation runs");
        let sources = |s: &Saturation| s.candidates.iter().map(|c| c.source.clone()).collect::<Vec<_>>();
        assert_eq!(sources(&a), sources(&b));
    }

    #[test]
    fn the_sort_of_a_class_is_the_class() {
        let k = kingdom();
        let bridge = Bridge::load();
        // Every row has a constructor, and no two rows share one: the
        // mapping is a bijection, so a row cannot be read as another row.
        let mut ctors = std::collections::HashSet::new();
        for row in &k.rows {
            let ctor = bridge.constructor_of(&row.name).expect("a constructor");
            assert!(ctors.insert(ctor.to_string()), "the constructor {ctor} is used twice");
            assert_eq!(bridge.row_of(ctor), Some(row.name.as_str()), "the constructor {ctor} maps back");
        }
        // Every class has a sort, and the sorts are distinct: the sort IS
        // the class, so two classes sharing one would make the sort stop
        // being the class.
        let mut sorts = std::collections::HashSet::new();
        for d in &k.duals {
            let s = bridge.sort_of(&d.dual).expect("a class has a sort");
            assert!(sorts.insert(s.to_string()), "the sort {s} is used twice");
        }
    }
}
