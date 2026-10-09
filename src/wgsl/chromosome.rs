//! A read function as a chromosome: its roots are the head genes, the
//! subtrees repeated across them become homeotic tail genes
//! (`homeotic::fold`), and every gene is encoded as Karva through the
//! generic pair at a chosen head length. The Karva round trip (encode →
//! decode → unfold → the roots' canonical text) is checked here; the naga
//! round trip (rebuild the module from the genes) is the scaffold's.
//!
//! **Sharing.** Exact repeats only, found by hash-consing the roots' trees in
//! Rust (the e-graph finder speaks the `Math` datatype, which has no
//! `class.instance` constructors). The load-sharing rule is applied in its
//! CONSERVATIVE form: a load of a target that is stored anywhere in the
//! function is unique at every site, so no two roots may share it; a load of
//! a target the function never stores (a read-only buffer, a uniform, an
//! argument) may. The plan's finer rule (no intervening store in statement
//! order) would admit more; what this form refuses is counted and reported.

use std::collections::{BTreeMap, BTreeSet};

use crate::extract::Match;
use crate::geneframe::Ty;
use crate::homeotic::{self, FoldOptions, FoldedChromosome, KarvaGene};
use crate::karva::{karva_to_terms_generic, parse_math, FunctionSpec, MathNode, PsetSpec};

use super::infer::infer_function;
use super::loader::WgslKingdom;
use super::reader::{KernelFunction, RootKind};
use super::versions::split_version;

/// What one function became.
#[derive(Debug, Clone)]
pub struct WgslChromosome {
    pub function: String,
    /// The roots' s-expressions, canonical, in root order (the head genes
    /// before folding).
    pub roots: Vec<String>,
    pub folded: FoldedChromosome,
    /// Matches the finder reported.
    pub matches: usize,
    /// Repeated subtrees the conservative load rule refused to share.
    pub refused_loads: usize,
    /// The pset the genes are encoded over: the function's terminals, the
    /// `class.instance` names it uses with their arities, and the hrefs.
    pub pset: PsetSpec,
    /// Nodes of the largest gene after folding: the head length needed.
    pub head_needed: usize,
    /// `(head_length, genes that do not fit)` for each length tried.
    pub oversized_at: Vec<(usize, usize)>,
    /// The Karva genes at the first head length every gene fits, if any.
    pub genes: Option<(usize, Vec<KarvaGene>)>,
    /// Per gene (head genes then tail genes, as `genes`), the dual CODE of
    /// each live head token in level order (`Ty::code()`; `None` where the
    /// slot has no dual or a form conflict left the node opaque). A tail
    /// gene's root dual is what an `href<t>` leaf carries. Empty when the
    /// chromosome was built untyped.
    pub ty_codes: Vec<Vec<Option<u32>>>,
    /// Repeated subtrees refused for sharing because their occurrences
    /// were read under two forms (the plan's form-conflict rule).
    pub refused_forms: usize,
    /// Repeats the history forbids: subtree texts equal once the load
    /// versions are stripped, but read under more than one version
    /// (`docs/PLAN_wgsl_lineage.md` §3). Counted, never folded.
    pub refused_lineage: usize,
}

/// Options for [`chromosome`].
#[derive(Debug, Clone)]
pub struct ChromosomeOptions {
    pub tail_slots: usize,
    /// Minimum internal operators of a shared definition.
    pub min_ops: usize,
    /// Head lengths to try, ascending.
    pub head_lengths: Vec<usize>,
    pub rng_seed: u64,
}

impl Default for ChromosomeOptions {
    fn default() -> Self {
        ChromosomeOptions { tail_slots: 16, min_ops: 2, head_lengths: vec![8, 12, 16, 24, 32, 48, 64], rng_seed: 7013 }
    }
}

fn render(n: &MathNode) -> String {
    match n {
        MathNode::Num(v) => format!("(Num {v:?})"),
        MathNode::Var(name) => format!("(Var {name:?})"),
        MathNode::App(ctor, children) => {
            let parts: Vec<String> = children.iter().map(render).collect();
            if parts.is_empty() {
                format!("({ctor})")
            } else {
                format!("({ctor} {})", parts.join(" "))
            }
        }
    }
}

fn op_count(n: &MathNode) -> usize {
    match n {
        MathNode::Num(_) | MathNode::Var(_) => 0,
        MathNode::App(_, children) => 1 + children.iter().map(op_count).sum::<usize>(),
    }
}

fn node_count(n: &MathNode) -> usize {
    match n {
        MathNode::Num(_) | MathNode::Var(_) => 1,
        MathNode::App(_, children) => 1 + children.iter().map(node_count).sum::<usize>(),
    }
}

/// Whether a subtree reads a target in `stored` (a `load.<target>` node or
/// leaf, or a local read as `(Var "load.<target>")`).
fn reads_stored(n: &MathNode, stored: &BTreeSet<String>) -> bool {
    match n {
        MathNode::Var(name) => split_version(name).0.strip_prefix("load.").is_some_and(|t| stored.contains(t)),
        MathNode::App(ctor, children) => split_version(ctor).0.strip_prefix("load.").is_some_and(|t| stored.contains(t)) || children.iter().any(|c| reads_stored(c, stored)),
        MathNode::Num(_) => false,
    }
}

/// `n` with every load's version stripped: the text two occurrences would
/// share if memory had no history.
fn strip_versions(n: &MathNode) -> MathNode {
    match n {
        MathNode::Num(v) => MathNode::Num(*v),
        MathNode::Var(name) => MathNode::Var(split_version(name).0.to_string()),
        MathNode::App(ctor, children) => MathNode::App(split_version(ctor).0.to_string(), children.iter().map(strip_versions).collect()),
    }
}

/// Repeats the history forbids: distinct version-stripped texts with at
/// least `min_ops` operators that occur at two or more sites under more
/// than one versioned text.
pub(crate) fn refused_by_lineage(roots: &[MathNode], min_ops: usize) -> usize {
    fn walk(n: &MathNode, out: &mut BTreeMap<String, (usize, BTreeSet<String>, usize)>) {
        if op_count(n) >= 1 {
            let entry = out.entry(render(&strip_versions(n))).or_insert_with(|| (op_count(n), BTreeSet::new(), 0));
            entry.1.insert(render(n));
            entry.2 += 1;
        }
        if let MathNode::App(_, children) = n {
            for c in children {
                walk(c, out);
            }
        }
    }
    let mut groups = BTreeMap::new();
    for r in roots {
        walk(r, &mut groups);
    }
    groups.values().filter(|(ops, texts, sites)| *ops >= min_ops && *sites >= 2 && texts.len() > 1).count()
}

/// Exact repeats across (and within) the roots, largest first, with the
/// containment rule of `extract::shared_sites_in` and the conservative load
/// rule. Returns the matches and how many candidate repeats the load rule
/// refused.
pub(crate) fn exact_shared(roots: &[MathNode], stored: &BTreeSet<String>, min_ops: usize) -> (Vec<Match>, usize) {
    #[derive(Clone)]
    struct Site {
        gene: usize,
        path: Vec<u8>,
    }
    fn walk<'a>(n: &'a MathNode, gene: usize, path: Vec<u8>, out: &mut BTreeMap<String, (usize, Vec<Site>, &'a MathNode)>) {
        let key = render(n);
        let entry = out.entry(key).or_insert_with(|| (op_count(n), Vec::new(), n));
        entry.1.push(Site { gene, path: path.clone() });
        if let MathNode::App(_, children) = n {
            for (i, c) in children.iter().enumerate() {
                let mut p = path.clone();
                p.push(i as u8);
                walk(c, gene, p, out);
            }
        }
    }
    let mut sites_of: BTreeMap<String, (usize, Vec<Site>, &MathNode)> = BTreeMap::new();
    for (g, root) in roots.iter().enumerate() {
        walk(root, g, Vec::new(), &mut sites_of);
    }
    let mut candidates: Vec<(&String, usize, &Vec<Site>, &MathNode)> =
        sites_of.iter().filter(|(_, (ops, sites, _))| sites.len() >= 2 && *ops >= min_ops).map(|(k, (ops, sites, n))| (k, *ops, sites, *n)).collect();
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let inside = |s: &Site, a: &Site| s.gene == a.gene && s.path.len() > a.path.len() && s.path[..a.path.len()] == a.path[..];
    let mut accepted: Vec<Site> = Vec::new();
    let mut matches = Vec::new();
    let mut refused = 0usize;
    for (_, ops, sites, node) in candidates {
        if reads_stored(node, stored) {
            refused += 1;
            continue;
        }
        let independent: Vec<&Site> = sites.iter().filter(|s| !accepted.iter().any(|a| inside(s, a))).collect();
        if independent.len() < 2 {
            continue;
        }
        matches.push(Match { sites: independent.iter().map(|s| (s.gene, s.path.clone())).collect(), internal_op_count: ops });
        accepted.extend(independent.into_iter().cloned());
    }
    (matches, refused)
}

/// The pset a function's genes are encoded over.
fn pset_of(roots: &[MathNode]) -> Result<PsetSpec, String> {
    let mut variables: BTreeSet<String> = BTreeSet::new();
    let mut functions: BTreeMap<String, usize> = BTreeMap::new();
    fn collect(n: &MathNode, vars: &mut BTreeSet<String>, fns: &mut BTreeMap<String, usize>) -> Result<(), String> {
        match n {
            MathNode::Var(name) => {
                vars.insert(name.clone());
            }
            MathNode::Num(_) => {}
            MathNode::App(ctor, children) => {
                let arity = children.len();
                if let Some(&seen) = fns.get(ctor) {
                    if seen != arity {
                        return Err(format!("{ctor} used with {seen} and {arity} children; one token cannot carry both"));
                    }
                } else {
                    fns.insert(ctor.clone(), arity);
                }
                for c in children {
                    collect(c, vars, fns)?;
                }
            }
        }
        Ok(())
    }
    for r in roots {
        collect(r, &mut variables, &mut functions)?;
    }
    Ok(PsetSpec {
        variables: variables.into_iter().collect(),
        functions: functions.into_iter().map(|(name, arity)| (name.clone(), FunctionSpec { semantic_id: name, arity })).collect(),
        rnc_values: vec![0.0],
    })
}

/// Fold and encode one read function, untyped (no duals carried).
pub fn chromosome(f: &KernelFunction, opts: &ChromosomeOptions) -> Result<WgslChromosome, String> {
    build(f, opts, None)
}

/// [`chromosome`] with the duals of [`infer_function`] carried: a text read
/// under two forms is never shared, and every gene carries a `ty_code` per
/// live head token.
pub fn chromosome_typed(f: &KernelFunction, kingdom: &WgslKingdom, opts: &ChromosomeOptions) -> Result<WgslChromosome, String> {
    build(f, opts, Some(kingdom))
}

fn build(f: &KernelFunction, opts: &ChromosomeOptions, kingdom: Option<&WgslKingdom>) -> Result<WgslChromosome, String> {
    let roots: Vec<MathNode> = f.roots.iter().map(|r| parse_math(&r.tree.to_sexpr())).collect::<Result<_, _>>()?;
    // The dual of every subtree TEXT (canonical rendering), and the texts
    // whose occurrences took more than one: those must not be shared.
    let mut dual_of_text: BTreeMap<String, Option<Ty>> = BTreeMap::new();
    let mut two_forms: BTreeSet<String> = BTreeSet::new();
    if let Some(kingdom) = kingdom {
        let types = infer_function(f, kingdom);
        for (text, duals) in &types.by_text {
            let key = render(&parse_math(text)?);
            if duals.len() > 1 {
                two_forms.insert(key);
            } else if let Some(d) = duals.iter().next() {
                dual_of_text.insert(key, *d);
            }
        }
    }
    let stored: BTreeSet<String> = f
        .roots
        .iter()
        .filter_map(|r| match &r.kind {
            RootKind::Store { target } => Some(target.clone()),
            RootKind::Init { local } => Some(local.clone()),
            RootKind::Let { .. } | RootKind::Condition { .. } | RootKind::Return | RootKind::Argument { .. } => None,
        })
        .collect();
    let canonical: Vec<String> = roots.iter().map(render).collect();
    let (matches, refused_loads) = exact_shared(&roots, &stored, opts.min_ops);
    let refused_lineage = refused_by_lineage(&roots, opts.min_ops);
    // The form-conflict rule: a repeated text whose occurrences were read
    // under two forms keeps its own dual at each site and is not folded.
    let (matches, refused_forms): (Vec<Match>, usize) = {
        let mut kept = Vec::new();
        let mut refused = 0usize;
        for m in matches {
            let Some((gene, path)) = m.sites.first() else { continue };
            let text = node_at(&roots[*gene], path).map(render).unwrap_or_default();
            if two_forms.contains(&text) {
                refused += 1;
            } else {
                kept.push(m);
            }
        }
        (kept, refused)
    };
    let max_head = opts.head_lengths.iter().copied().max().unwrap_or(0);
    let folded = homeotic::fold(&canonical, &matches, FoldOptions { tail_slots: opts.tail_slots, max_definition_ops: Some(max_head) })?;
    // The fold must unfold to what it was given.
    let back = homeotic::unfold(&folded)?;
    if back != canonical {
        return Err(format!("{}: the fold does not unfold to its input", f.name));
    }
    let all_genes: Vec<MathNode> = folded.head.iter().chain(folded.tail.iter()).map(|g| parse_math(g)).collect::<Result<_, _>>()?;
    let mut pset = pset_of(&all_genes)?;
    pset = homeotic::pset_with_hrefs(&pset, folded.tail.len());
    let head_needed = all_genes.iter().map(node_count).max().unwrap_or(0);
    let mut oversized_at = Vec::new();
    let mut genes = None;
    for &h in &opts.head_lengths {
        let over = all_genes.iter().filter(|g| node_count(g) > h).count();
        oversized_at.push((h, over));
        if over == 0 && genes.is_none() {
            let encoded = homeotic::encode_generic_k(&folded, &pset, opts.rng_seed, h, crate::gpu_eval::K_MAX)?;
            // The Karva round trip: every gene decodes to its own text.
            for (i, ((head, tail), text)) in encoded.iter().zip(folded.head.iter().chain(folded.tail.iter())).enumerate() {
                // Compared as trees through one renderer: the decoder spells a
                // float with `{}` and the fold with `{:?}`, equal values either way.
                let decoded = render(&parse_math(&karva_to_terms_generic(head, tail, &pset)?)?);
                if decoded != *text {
                    return Err(format!("{}: gene {i} does not round-trip through Karva at head {h}:\n  {text}\n  {decoded}", f.name));
                }
            }
            genes = Some((h, encoded));
        }
    }
    // Duals per live head token, level order, for every gene: a leaf
    // `href<t>` carries tail gene t's root dual; a literal's `Num` child
    // carries the literal's dual.
    let ty_codes = if kingdom.is_some() {
        let tail_root_dual: Vec<Option<Ty>> = folded.tail.iter().map(|g| parse_math(g).ok().and_then(|n| dual_of_text.get(&render(&n)).copied().flatten())).collect();
        all_genes.iter().map(|g| level_order_duals(g, &dual_of_text, &tail_root_dual)).collect()
    } else {
        Vec::new()
    };
    Ok(WgslChromosome {
        function: f.name.clone(),
        roots: canonical,
        folded,
        matches: matches.len(),
        refused_loads,
        pset,
        head_needed,
        oversized_at,
        genes,
        ty_codes,
        refused_forms,
        refused_lineage,
    })
}

fn node_at<'a>(root: &'a MathNode, path: &[u8]) -> Option<&'a MathNode> {
    let mut cur = root;
    for &i in path {
        match cur {
            MathNode::App(_, children) => cur = children.get(i as usize)?,
            _ => return None,
        }
    }
    Some(cur)
}

/// The dual code of every node of `g` in the encoder's level order (the
/// order of the live head tokens), `None` where unknown.
fn level_order_duals(g: &MathNode, dual_of_text: &BTreeMap<String, Option<Ty>>, tail_root_dual: &[Option<Ty>]) -> Vec<Option<u32>> {
    let mut out = Vec::new();
    let mut queue: std::collections::VecDeque<(&MathNode, Option<Ty>)> = std::collections::VecDeque::new();
    queue.push_back((g, None));
    while let Some((n, inherited)) = queue.pop_front() {
        let own = match n {
            MathNode::Var(name) => homeotic::href_slot(name).and_then(|t| tail_root_dual.get(t).copied().flatten()).or_else(|| dual_of_text.get(&render(n)).copied().flatten()),
            MathNode::Num(_) => inherited,
            MathNode::App(..) => dual_of_text.get(&render(n)).copied().flatten(),
        };
        out.push(own.map(Ty::code));
        if let MathNode::App(ctor, children) = n {
            let pass = if ctor.starts_with("literal.") { own } else { None };
            for c in children {
                queue.push_back((c, pass));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::reader::read;

    const TWO_STORES: &str = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x * 2u + 1u;
    let v = xs[i] * xs[i];
    a[i] = v + 1.0;
    b[i] = v - 1.0;
}
"#;

    #[test]
    fn a_repeated_subtree_across_two_stores_becomes_one_tail_gene_and_round_trips() {
        let k = read(TWO_STORES).unwrap();
        let f = &k.functions[0];
        let c = chromosome(f, &ChromosomeOptions { tail_slots: 3, min_ops: 2, head_lengths: vec![4, 8, 16], rng_seed: 1 }).unwrap();
        assert_eq!(c.roots.len(), 4, "two lets and two stores");
        // With `let i` and `let v` bound once, the stores read them by name:
        // nothing repeats with two or more operators, so nothing folds.
        assert_eq!(c.refused_loads, 0, "xs is never stored");
        let (h, genes) = c.genes.as_ref().expect("fits at some head length");
        assert_eq!(genes.len(), 4 + 3);
        assert!(*h <= 16);
        // The kingdom's K = 4 sets every gene's layout: tail 3·head + 1.
        assert!(genes.iter().all(|(head, tail)| head.len() == *h && tail.len() == 3 * h + 1));
        assert!(c.pset.variables.iter().any(|v| v.starts_with("let.v@")), "{:?}", c.pset.variables);
    }

    #[test]
    fn a_load_of_a_stored_local_is_never_shared() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var s = 1.0;
    out[0] = s * s + 2.0;
    s = 3.0;
    out[1] = s * s + 2.0;
}
"#;
        let k = read(src).unwrap();
        let c = chromosome(&k.functions[0], &ChromosomeOptions::default()).unwrap();
        // `s * s + 2.0` repeats only with the versions stripped: the reads
        // are `load.local.s@0@v0` and `@v1`, so the finder sees no repeat at
        // all and the lineage count says why.
        assert_eq!(c.matches, 0);
        assert_eq!(c.refused_loads, 0, "no repeat of one version reaches the load rule");
        assert_eq!(c.refused_lineage, 1, "s * s + 2.0 under two versions");
        assert_eq!(c.folded.filled, 0);
        assert!(c.genes.is_some());
        // The same two reads with no store between them share, and the rule
        // that refuses a stored target's load still holds (step 1 keeps it).
        let src = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var s = 1.0;
    out[0] = s * s + 2.0;
    out[1] = s * s + 2.0;
    s = 3.0;
}
"#;
        let k = read(src).unwrap();
        let c = chromosome(&k.functions[0], &ChromosomeOptions::default()).unwrap();
        assert_eq!((c.matches, c.refused_loads, c.refused_lineage), (0, 1, 0), "equal versions, refused by the conservative rule until legality lands");
    }

    #[test]
    fn the_typed_chromosome_carries_a_dual_per_live_token_and_refuses_two_form_texts() {
        let kingdom = WgslKingdom::load();
        // A float pipeline: every token has a dual.
        let k = read(TWO_STORES).unwrap();
        let c = chromosome_typed(&k.functions[0], &kingdom, &ChromosomeOptions::default()).unwrap();
        let (_, genes) = c.genes.as_ref().unwrap();
        assert_eq!(c.ty_codes.len(), genes.len());
        let texts: Vec<&String> = c.folded.head.iter().chain(c.folded.tail.iter()).collect();
        for ((codes, (head, _)), text) in c.ty_codes.iter().zip(genes).zip(texts) {
            assert!(codes.len() <= head.len(), "codes cover the live prefix of the head");
            if *text == homeotic::EMPTY_SLOT {
                assert_eq!(codes, &vec![None], "an empty tail slot carries no dual");
                continue;
            }
            assert!(codes.iter().all(|d| d.is_some()), "{text}: {codes:?}");
            assert!(codes.iter().flatten().all(|d| *d >= Ty::TABLE_CODE_BASE));
        }
        let store = &c.ty_codes[c.roots.len() - 1];
        assert_eq!(store[0], Some(kingdom.duals.iter().position(|d| d.dual == "store.store").unwrap() as u32 + Ty::TABLE_CODE_BASE), "a store root's root token is the store dual");
        // A text read as an index in one root and as bits in another is not shared.
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<u32>;
@group(0) @binding(1) var<storage, read_write> a: array<u32>;
@group(0) @binding(2) var<storage, read_write> b: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    a[gid.x] = xs[gid.x * 2u + 1u] + 1u;
    b[gid.x] = xs[gid.x * 2u + 1u] & 1u;
}
"#;
        let k = read(src).unwrap();
        let untyped = chromosome(&k.functions[0], &ChromosomeOptions::default()).unwrap();
        let typed = chromosome_typed(&k.functions[0], &kingdom, &ChromosomeOptions::default()).unwrap();
        assert!(untyped.matches >= 1, "untyped, the load of xs[...] repeats and folds");
        assert_eq!(typed.refused_forms, 1, "typed, that text was read as index and as bits");
        assert_eq!(typed.matches, untyped.matches - 1);
        assert!(typed.genes.is_some());
    }

    #[test]
    fn a_name_used_at_two_arities_is_refused_not_mis_encoded() {
        let roots = vec![parse_math("(store.x (Num 1.0))").unwrap(), parse_math("(store.x (Num 1.0) (Num 2.0))").unwrap()];
        assert!(pset_of(&roots).unwrap_err().contains("store.x used with 1 and 2 children"));
    }
}
