//! A safeWGSL kernel as Karva genes, and back.
//!
//! The genotype of an individual in the safeWGSL kingdom: ONE head gene,
//! the kernel's entry (`kernel.entry_safe grid body`), and homeotic tail
//! genes holding the subtrees too large for the head. A subtree is hoisted
//! into a tail slot when the node it hangs from would not fit the head
//! length; identical subtrees (by text) share one slot, so an exact repeat
//! inside a kernel is already a shared definition. A definition references
//! only lower slots (children are hoisted before their parents), which is
//! the order `homeotic::unfold` substitutes in.
//!
//! Every gene is encoded by the generic Karva encoder at the kingdom's
//! K = 4 (tail = 3·head + 1), and carries one dual code per live token in
//! level order, a literal's payload token inheriting the literal's dual and
//! an `href<t>` leaf carrying slot t's root dual: the form phylu's typed
//! loader reads.
//!
//! The way back is [`build`]: a decoded, unfolded gene text projected onto
//! the kingdom's rows under a demanded dual, so a phenotype is always a
//! typed tree the renderer accepts; a token the demand has no row for
//! becomes a fallback leaf, as the typed projection does on the device.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::geneframe::Ty;
use crate::homeotic::{self, href_name, href_slot, FoldedChromosome, KarvaGene, EMPTY_SLOT};
use crate::karva::{parse_math, FunctionSpec, MathNode, PsetSpec, Token};
use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::render::{cost, dual_ty, gram_role, gram_shipped_safe_tree, Cost, Rendered, Renderer, Role, Tree};
use crate::wgsl::table::FunctionTable;

/// The kingdom's K: every node has at most four children.
pub const K: usize = 4;

/// The terminal a projection substitutes for a token of the wrong dual.
pub const FALLBACK: &str = "fallback";

/// A kernel split into its head gene and homeotic tail genes; an `href<n>`
/// leaf carries slot n's root dual as its `ty`.
#[derive(Debug, Clone)]
pub struct SafeSplit {
    pub head: Tree,
    pub tail: Vec<Tree>,
}

/// The encoded individual.
#[derive(Debug, Clone)]
pub struct SafeChromosome {
    pub head_length: usize,
    pub tail_slots: usize,
    /// Gene texts in Karva form: `head` (one gene) then `tail` (every slot,
    /// the unfilled ones [`EMPTY_SLOT`]).
    pub folded: FoldedChromosome,
    pub pset: PsetSpec,
    /// Head gene first, then the tail slots in order.
    pub genes: Vec<KarvaGene>,
    /// Per gene, the dual code of each live token in level order; empty for
    /// an unfilled slot.
    pub ty_codes: Vec<Vec<u32>>,
    /// Per gene, its root dual (`None` for an unfilled slot).
    pub roots: Vec<Option<Ty>>,
}

/// A token with its dual, `name@T061`: the row's name and the dual of its
/// output by the dual's stable reference in types.tsv (`T001` is
/// `Ty::Wgsl(0)`). The gene text carries every token's dual: an untyped
/// name left the projection to break a tie among a row's signatures on the
/// first, so `select(vec4<i32>(0), ..)` came back as the vec4<f32> row (5
/// of 80 random rows of a phylu run) and a polymorphic row under a
/// polymorphic parent (`trig.acos` under `geom.length`) as its first
/// shape (237 of 1,000 random trees, 2026-10-13).
pub fn typed_token(name: &str, ty: Ty) -> String {
    match ty {
        Ty::Wgsl(i) => format!("{name}@T{:03}", u32::from(i) + 1),
        _ => name.to_string(),
    }
}

/// A fallback leaf's token: `fallback@T061`, its dual named.
pub fn fallback_token(ty: Ty) -> String {
    typed_token(FALLBACK, ty)
}

/// A token's name and the dual it names, if typed (`name@T061`).
pub fn split_token(token: &str) -> (&str, Option<Ty>) {
    if let Some((name, r)) = token.rsplit_once("@T") {
        if let Some(i) = r.parse::<u16>().ok().and_then(|n| n.checked_sub(1)) {
            return (name, Some(Ty::Wgsl(i)));
        }
    }
    (token, None)
}

/// A node's token in the Karva text: its name with its dual (a reference
/// to a shared gene stays as it is: the slot's definition carries its dual).
fn token(t: &Tree) -> String {
    if is_href(t) {
        t.name.clone()
    } else {
        typed_token(&t.name, t.ty)
    }
}

/// The tree in the Karva text form the encoder and the decoder share: a
/// terminal as `(Var "name@T…")`, a literal as `(literal.x@T… (Num v))`, an
/// application as `(name@T… kids…)`; each token typed ([`typed_token`]).
pub fn karva_text(t: &Tree) -> String {
    if t.kids.is_empty() {
        if t.name.starts_with("literal.") {
            return format!("({} (Num {:?}))", token(t), f64::from(t.konst));
        }
        return format!("(Var {:?})", token(t));
    }
    format!("({} {})", token(t), t.kids.iter().map(karva_text).collect::<Vec<_>>().join(" "))
}

/// The level-order token stream of the tree, as the encoder emits it: a
/// literal is a function token followed, in its level, by its payload.
fn level_order(t: &Tree) -> Vec<(&Tree, bool)> {
    // (node, is_payload): a payload is the Num token under a literal.
    let mut out = Vec::new();
    let mut queue: VecDeque<(&Tree, bool)> = VecDeque::new();
    queue.push_back((t, false));
    while let Some((n, payload)) = queue.pop_front() {
        out.push((n, payload));
        if payload {
            continue;
        }
        if n.kids.is_empty() && n.name.starts_with("literal.") {
            queue.push_back((n, true));
        }
        for k in &n.kids {
            queue.push_back((k, false));
        }
    }
    out
}

/// The head length the tree needs: the generic encoder writes EVERY token
/// of the level order into the head (the tail is padding only), so the
/// need is the token count, a literal counting its payload.
pub fn head_needed(t: &Tree) -> usize {
    level_order(t).len()
}

fn is_href(t: &Tree) -> bool {
    t.kids.is_empty() && href_slot(&t.name).is_some()
}

/// Split the kernel so every gene fits `head_length`: a node whose
/// subtree needs more is relieved of its largest children, hoisted into
/// tail slots (shared by text), until it fits.
pub fn split(tree: &Tree, head_length: usize) -> Result<SafeSplit, String> {
    if head_length < K + 1 {
        return Err(format!("head length {head_length} cannot hold a node with {K} children"));
    }
    let mut slots: Vec<Tree> = Vec::new();
    let mut by_text: HashMap<String, usize> = HashMap::new();
    fn fit(t: &Tree, h: usize, slots: &mut Vec<Tree>, by_text: &mut HashMap<String, usize>) -> Result<Tree, String> {
        let kids = t.kids.iter().map(|k| fit(k, h, slots, by_text)).collect::<Result<Vec<_>, _>>()?;
        let mut node = Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids };
        while head_needed(&node) > h {
            // The largest child that is not already a reference; a child
            // with children before a literal.
            let pick = node.kids.iter().enumerate().filter(|(_, k)| !is_href(k)).max_by_key(|(_, k)| (!k.kids.is_empty(), k.size()));
            let Some((i, _)) = pick else { return Err(format!("{} does not fit a head of {h} with every child hoisted", node.name)) };
            let sub = node.kids[i].clone();
            let text = karva_text(&sub);
            let slot = match by_text.get(&text) {
                Some(&s) => s,
                None => {
                    slots.push(sub.clone());
                    by_text.insert(text, slots.len() - 1);
                    slots.len() - 1
                }
            };
            node.kids[i] = Tree::new(&href_name(slot), sub.ty, vec![]);
        }
        Ok(node)
    }
    let head = fit(tree, head_length, &mut slots, &mut by_text)?;
    Ok(SafeSplit { head, tail: slots })
}

/// The dual code of every live token of the gene in level order.
fn codes_of(t: &Tree) -> Vec<u32> {
    level_order(t).iter().map(|(n, _)| n.ty.code()).collect()
}

/// Encode the kernel at `head_length` with `tail_slots` homeotic slots.
pub fn encode(tree: &Tree, head_length: usize, tail_slots: usize, rng_seed: u64) -> Result<SafeChromosome, String> {
    let s = split(tree, head_length)?;
    if s.tail.len() > tail_slots {
        return Err(format!("{} tail slots needed at head {head_length}, {tail_slots} in the layout", s.tail.len()));
    }
    let mut tail: Vec<Option<Tree>> = s.tail.into_iter().map(Some).collect();
    tail.resize(tail_slots, None);
    encode_folded(&s.head, &tail, head_length, rng_seed)
}

/// Encode a head gene and an explicit tail (a definition per filled slot,
/// `None` for an empty one; a definition references only lower slots).
pub fn encode_folded(head: &Tree, tail: &[Option<Tree>], head_length: usize, rng_seed: u64) -> Result<SafeChromosome, String> {
    let tail_slots = tail.len();
    // The pset: every terminal and every function with its arity.
    let mut variables: Vec<String> = Vec::new();
    let mut functions: BTreeMap<String, usize> = BTreeMap::new();
    fn collect(t: &Tree, variables: &mut Vec<String>, functions: &mut BTreeMap<String, usize>) -> Result<(), String> {
        if t.kids.is_empty() {
            if t.name.starts_with("literal.") {
                functions.insert(token(t), 1);
            } else if !variables.contains(&token(t)) && !is_href(t) {
                variables.push(token(t));
            }
            return Ok(());
        }
        if t.kids.len() > K {
            return Err(format!("{} has {} children, past K = {K}", t.name, t.kids.len()));
        }
        match functions.insert(token(t), t.kids.len()) {
            Some(a) if a != t.kids.len() => return Err(format!("{} appears with arity {a} and {}", t.name, t.kids.len())),
            _ => {}
        }
        t.kids.iter().try_for_each(|k| collect(k, variables, functions))
    }
    collect(head, &mut variables, &mut functions)?;
    tail.iter().flatten().try_for_each(|t| collect(t, &mut variables, &mut functions))?;
    if !variables.iter().any(|v| v == FALLBACK) {
        variables.push(FALLBACK.into());
    }
    let pset = PsetSpec { variables, functions: functions.iter().map(|(n, &a)| (n.clone(), FunctionSpec { semantic_id: n.clone(), arity: a })).collect(), rnc_values: vec![0.0] };
    let tail_text: Vec<String> = tail.iter().map(|t| t.as_ref().map_or_else(|| EMPTY_SLOT.to_string(), karva_text)).collect();
    let filled = tail.iter().filter(|t| t.is_some()).count();
    let folded = FoldedChromosome { head: vec![karva_text(head)], tail: tail_text, filled, skipped: 0 };
    let genes = homeotic::encode_generic_k(&folded, &pset, rng_seed, head_length, K)?;
    let mut ty_codes = vec![codes_of(head)];
    let mut roots = vec![Some(head.ty)];
    for t in tail {
        ty_codes.push(t.as_ref().map(codes_of).unwrap_or_default());
        roots.push(t.as_ref().map(|t| t.ty));
    }
    // Each gene's live tokens are as many as its codes.
    for (i, (g, codes)) in genes.iter().zip(&ty_codes).enumerate() {
        let live = live_tokens(&g.0, &g.1, &pset)?;
        if !codes.is_empty() && live != codes.len() {
            return Err(format!("gene {i}: {live} live tokens, {} codes", codes.len()));
        }
    }
    Ok(SafeChromosome { head_length, tail_slots, folded, pset, genes, ty_codes, roots })
}

/// A component shared across the population: its slot, text, how many
/// individuals carry it and its node count.
#[derive(Debug, Clone)]
pub struct SharedDefinition {
    pub slot: usize,
    pub text: String,
    pub referrers: usize,
    pub occurrences: usize,
    pub nodes: usize,
}

/// THE SHARED SECTIONS OF TWO KERNELS (Andrew, 2026-10-11: "find the largest
/// common section from the two individuals, then give that and the two
/// originals to the model and ask it to rewrite the kernel around that
/// largest share"): every sub-tree the two trees have in common with at
/// least `min_nodes` nodes, largest first, ties by text, matched as the
/// population fold matches (exact Karva text of the sub-tree); only maximal
/// sections, never a part of a larger shared one.
pub fn shared_sections(a: &Tree, b: &Tree, min_nodes: usize) -> Vec<SharedDefinition> {
    fn walk(t: &Tree, min_nodes: usize, out: &mut HashMap<String, (usize, usize)>) {
        if !t.kids.is_empty() && t.size() >= min_nodes {
            let e = out.entry(karva_text(t)).or_insert((t.size(), 0));
            e.1 += 1;
        }
        for k in &t.kids {
            walk(k, min_nodes, out);
        }
    }
    let (mut in_a, mut in_b) = (HashMap::new(), HashMap::new());
    walk(a, min_nodes, &mut in_a);
    walk(b, min_nodes, &mut in_b);
    // Maximal sections only: a shared sub-tree inside a larger shared one is
    // that one's part, not a section of its own.
    fn maximal(t: &Tree, inside_shared: bool, in_a: &HashMap<String, (usize, usize)>, in_b: &HashMap<String, (usize, usize)>, keep: &mut HashMap<String, usize>) {
        let text = if t.kids.is_empty() { None } else { Some(karva_text(t)) };
        let shared = text.as_ref().is_some_and(|k| in_a.contains_key(k) && in_b.contains_key(k));
        if shared && !inside_shared {
            let k = text.expect("internal");
            let n = in_a[&k].0;
            keep.insert(k, n);
        }
        for kid in &t.kids {
            maximal(kid, inside_shared || shared, in_a, in_b, keep);
        }
    }
    let mut keep = HashMap::new();
    maximal(a, false, &in_a, &in_b, &mut keep);
    let mut shared: Vec<SharedDefinition> = keep
        .into_iter()
        .map(|(text, nodes)| {
            let occurrences = in_a[&text].1 + in_b[&text].1;
            SharedDefinition { slot: 0, text, referrers: 2, occurrences, nodes }
        })
        .collect();
    shared.sort_by(|x, y| y.nodes.cmp(&x.nodes).then_with(|| x.text.cmp(&y.text)));
    shared
}

/// How many sub-trees of `tree` have exactly the Karva text `text` (the
/// check that a model's child kept the shared section it was given).
pub fn occurrences_of(tree: &Tree, text: &str) -> usize {
    let here = usize::from(karva_text(tree) == text);
    here + tree.kids.iter().map(|k| occurrences_of(k, text)).sum::<usize>()
}

/// What the population fold found and produced.
#[derive(Debug)]
pub struct PopulationFold {
    /// Per individual, its chromosome with the shared components at their
    /// common slots, or why it could not be folded (its own definitions do
    /// not fit below the reserved band).
    pub chromosomes: Vec<Result<SafeChromosome, String>>,
    pub shared: Vec<SharedDefinition>,
    /// Distinct subtree texts over the population (internal nodes), their
    /// occurrences, and the occurrences that are repeats of one seen before.
    pub distinct_subtrees: usize,
    pub occurrences: usize,
    pub repeated_occurrences: usize,
    /// The individuals' own definitions held by two or more of them: those
    /// placed at one common slot in every holder, and those left at slots of
    /// each holder's own (no common slot free above their references).
    pub own_aligned: usize,
    pub own_unaligned: usize,
}

/// SHARE-EQUIVALENCE OVER THE POPULATION (step 4 of the experiment): every
/// subtree of every individual keyed by its text; a subtree that occurs in
/// at least two individuals, fits a gene head and has at least `min_nodes`
/// nodes is a shared component. The components take the slots from
/// `reserved_from` up, the SAME slot in every individual, largest reach
/// first (referrers, then size), and every occurrence becomes a reference
/// to that slot; an individual's own hoisted definitions take the slots
/// below `reserved_from`. The kernels are unchanged (the unfold restores
/// every text); what changes is that the operators now move a component
/// between individuals as one gene.
pub fn fold_population(trees: &[Tree], head_length: usize, tail_slots: usize, reserved_from: usize, min_nodes: usize, rng_seed: u64) -> PopulationFold {
    // 1. Every internal subtree by text: which individuals, how often, an example.
    struct Seen {
        individuals: BTreeSet<usize>,
        occurrences: usize,
        nodes: usize,
        tokens: usize,
        example: Tree,
    }
    let mut seen: HashMap<String, Seen> = HashMap::new();
    let mut occurrences = 0usize;
    fn walk(t: &Tree, who: usize, root: bool, seen: &mut HashMap<String, Seen>, occurrences: &mut usize) {
        if !t.kids.is_empty() && !root {
            *occurrences += 1;
            let key = karva_text(t);
            match seen.get_mut(&key) {
                Some(s) => {
                    s.individuals.insert(who);
                    s.occurrences += 1;
                }
                None => {
                    seen.insert(key, Seen { individuals: BTreeSet::from([who]), occurrences: 1, nodes: t.size(), tokens: head_needed(t), example: t.clone() });
                }
            }
        }
        for k in &t.kids {
            walk(k, who, false, seen, occurrences);
        }
    }
    for (i, t) in trees.iter().enumerate() {
        walk(t, i, true, &mut seen, &mut occurrences);
    }
    let distinct_subtrees = seen.len();
    let repeated_occurrences = occurrences - distinct_subtrees;
    // 2. The components: shared by two or more, fitting a head, large enough.
    let mut candidates: Vec<(&String, &Seen)> = seen.iter().filter(|(_, s)| s.individuals.len() >= 2 && s.tokens <= head_length && s.nodes >= min_nodes).collect();
    candidates.sort_by(|a, b| b.1.individuals.len().cmp(&a.1.individuals.len()).then(b.1.nodes.cmp(&a.1.nodes)).then(a.0.cmp(b.0)));
    let room = tail_slots.saturating_sub(reserved_from);
    let mut shared: Vec<SharedDefinition> = Vec::new();
    let mut by_text: HashMap<String, usize> = HashMap::new();
    for (text, s) in candidates.into_iter().take(room) {
        let slot = reserved_from + shared.len();
        by_text.insert(text.clone(), slot);
        shared.push(SharedDefinition { slot, text: text.clone(), referrers: s.individuals.len(), occurrences: s.occurrences, nodes: s.nodes });
    }
    // 3. Every occurrence becomes a reference, top-down (a component inside a
    //    replaced component is not re-replaced), then each individual's own
    //    definitions are hoisted below the reserved band.
    fn replace(t: &Tree, by_text: &HashMap<String, usize>) -> Tree {
        if !t.kids.is_empty() {
            if let Some(&slot) = by_text.get(&karva_text(t)) {
                return Tree::new(&href_name(slot), t.ty, vec![]);
            }
        }
        Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| replace(k, by_text)).collect() }
    }
    let defs: Vec<Tree> = shared.iter().map(|d| seen[&d.text].example.clone()).collect();
    let splits: Vec<Result<SafeSplit, String>> = trees
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let s = split(&replace(t, &by_text), head_length)?;
            if s.tail.len() > reserved_from {
                return Err(format!("individual {i}: {} own definitions, {reserved_from} slots below the shared band", s.tail.len()));
            }
            Ok(s)
        })
        .collect();
    let (own_slots, own_aligned, own_unaligned) = align_own(&splits, reserved_from);
    let chromosomes = splits
        .into_iter()
        .zip(own_slots)
        .enumerate()
        .map(|(i, (s, slots))| {
            let s = s?;
            let slots = slots?;
            // Every own reference renumbered: head and definitions.
            let renamed = |t: &Tree| rename_own(t, &slots, reserved_from);
            let head = renamed(&s.head);
            let mut tail: Vec<Option<Tree>> = vec![None; tail_slots];
            for (j, def) in s.tail.iter().enumerate() {
                tail[slots[j]] = Some(renamed(def));
            }
            for (d, def) in shared.iter().zip(&defs) {
                // Only the individuals that reference the slot hold the definition.
                if karva_text(&head).contains(&format!("(Var \"{}\")", href_name(d.slot))) || tail.iter().flatten().any(|t| karva_text(t).contains(&format!("(Var \"{}\")", href_name(d.slot)))) {
                    tail[d.slot] = Some(def.clone());
                }
            }
            encode_folded(&head, &tail, head_length, rng_seed.wrapping_add(i as u64))
        })
        .collect();
    PopulationFold { chromosomes, shared, distinct_subtrees, occurrences, repeated_occurrences, own_aligned, own_unaligned }
}

/// `t` with every reference to an own slot (below `reserved_from`) moved
/// to its new slot.
fn rename_own(t: &Tree, slots: &[usize], reserved_from: usize) -> Tree {
    if let Some(j) = href_slot(&t.name).filter(|j| t.kids.is_empty() && *j < reserved_from) {
        if let Some(&n) = slots.get(j) {
            return Tree { name: href_name(n), ty: t.ty, konst: t.konst, kids: Vec::new() };
        }
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| rename_own(k, slots, reserved_from)).collect() }
}

/// Per individual, the new slot of each own definition, or why they no longer fit.
type OwnSlots = Result<Vec<usize>, String>;

/// SLOT ALIGNMENT of the individuals' own definitions (2026-10-13, the
/// engine's Karva crossovers swap genes by position, and each kernel
/// numbered its own definitions from 0: slot N of two kernels were
/// unrelated, and a crossover gave a head a reference to a definition of
/// another dual, a fallback). A definition is keyed by its content, its own
/// references expanded; the keys are placed in order of reference depth,
/// then by how many individuals hold them, each at the lowest slot above
/// its references that is free in every holder, the same slot in all of
/// them; a key with no such slot is placed in each holder at the lowest
/// free slot above its references there. Per individual, the new slot of
/// each old one (or why its definitions no longer fit), and the counts of
/// keys held by two or more individuals aligned and not.
fn align_own(splits: &[Result<SafeSplit, String>], reserved_from: usize) -> (Vec<OwnSlots>, usize, usize) {
    // Each individual's own definitions as content keys, and their references.
    fn expand(t: &Tree, tail: &[Tree], reserved_from: usize) -> Tree {
        if let Some(j) = href_slot(&t.name).filter(|j| t.kids.is_empty() && *j < reserved_from) {
            if let Some(def) = tail.get(j) {
                return Tree { name: "@def".into(), ty: t.ty, konst: 0.0, kids: vec![expand(def, tail, reserved_from)] };
            }
        }
        Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| expand(k, tail, reserved_from)).collect() }
    }
    fn refs(t: &Tree, reserved_from: usize, out: &mut BTreeSet<usize>) {
        if let Some(j) = href_slot(&t.name).filter(|j| t.kids.is_empty() && *j < reserved_from) {
            out.insert(j);
        }
        t.kids.iter().for_each(|k| refs(k, reserved_from, out));
    }
    let mut keys: Vec<Vec<String>> = Vec::new();
    let mut deps: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut holders: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for (i, s) in splits.iter().enumerate() {
        let Ok(s) = s else {
            keys.push(Vec::new());
            continue;
        };
        let k: Vec<String> = s.tail.iter().map(|d| karva_text(&expand(d, &s.tail, reserved_from))).collect();
        for (j, d) in s.tail.iter().enumerate() {
            let mut r = BTreeSet::new();
            refs(d, reserved_from, &mut r);
            deps.entry(k[j].clone()).or_default().extend(r.into_iter().filter_map(|x| k.get(x).cloned()));
            holders.entry(k[j].clone()).or_default().insert(i);
        }
        keys.push(k);
    }
    // Reference depth: a key above every key it references.
    fn depth(k: &str, deps: &BTreeMap<String, BTreeSet<String>>, memo: &mut BTreeMap<String, usize>) -> usize {
        if let Some(&d) = memo.get(k) {
            return d;
        }
        let d = deps.get(k).map_or(0, |ds| ds.iter().map(|x| 1 + depth(x, deps, memo)).max().unwrap_or(0));
        memo.insert(k.to_string(), d);
        d
    }
    let mut memo = BTreeMap::new();
    let mut order: Vec<(usize, usize, String)> = holders.iter().map(|(k, h)| (depth(k, &deps, &mut memo), h.len(), k.clone())).collect();
    order.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    let n = splits.len();
    let mut occupied: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    let mut slot_in: Vec<BTreeMap<String, usize>> = vec![BTreeMap::new(); n];
    let mut failed: Vec<Option<String>> = vec![None; n];
    let (mut aligned, mut unaligned) = (0usize, 0usize);
    for (_, _, k) in &order {
        let hs: Vec<usize> = holders[k].iter().copied().filter(|h| failed[*h].is_none()).collect();
        // Each holder's lowest slot above this key's references there.
        let los: Vec<usize> = hs.iter().map(|&h| deps.get(k).map_or(0, |ds| ds.iter().filter_map(|d| slot_in[h].get(d)).map(|s| s + 1).max().unwrap_or(0))).collect();
        let common_lo = los.iter().copied().max().unwrap_or(0);
        let common = (common_lo..reserved_from).find(|s| hs.iter().all(|&h| !occupied[h].contains(s)));
        match common {
            Some(s) => {
                for &h in &hs {
                    occupied[h].insert(s);
                    slot_in[h].insert(k.clone(), s);
                }
                aligned += usize::from(hs.len() >= 2);
            }
            None => {
                unaligned += usize::from(hs.len() >= 2);
                for (&h, &lo) in hs.iter().zip(&los) {
                    match (lo..reserved_from).find(|s| !occupied[h].contains(s)) {
                        Some(s) => {
                            occupied[h].insert(s);
                            slot_in[h].insert(k.clone(), s);
                        }
                        None => failed[h] = Some(format!("individual {h}: its own definitions do not fit below the shared band once aligned")),
                    }
                }
            }
        }
    }
    let out = (0..n)
        .map(|i| match (&splits[i], &failed[i]) {
            (Err(e), _) => Err(e.clone()),
            (_, Some(e)) => Err(e.clone()),
            (Ok(_), None) => keys[i].iter().map(|k| slot_in[i].get(k).copied().ok_or_else(|| format!("our software has a bug: individual {i}'s definition has no slot"))).collect(),
        })
        .collect();
    (out, aligned, unaligned)
}

/// The live token count of a gene: the GEP decode walk's reach.
fn live_tokens(head: &[Token], tail: &[Token], pset: &PsetSpec) -> Result<usize, String> {
    let stream: Vec<&Token> = head.iter().chain(tail.iter()).collect();
    let mut next = 1usize;
    let mut i = 0usize;
    while i < next && i < stream.len() {
        if let Token::Func(name) = stream[i] {
            next += pset.functions.get(name).map(|f| f.arity).ok_or_else(|| format!("function token {name:?} not in pset"))?;
        }
        i += 1;
    }
    Ok(next.min(stream.len()))
}

/// The dump of one kernel in the form `wgsl_read_kernel --dump` writes and
/// phylu's `read_sample` reads.
pub fn dump_json(file: &str, c: &SafeChromosome) -> serde_json::Value {
    let tok = |t: &Token| match t {
        Token::Func(n) => serde_json::json!({ "f": n }),
        Token::Var(n) => serde_json::json!({ "v": n }),
        Token::Num(v) => serde_json::json!({ "n": v }),
    };
    let mut functions: Vec<serde_json::Value> = c.pset.functions.iter().map(|(n, spec)| serde_json::json!({ "name": n, "arity": spec.arity })).collect();
    functions.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    let pset_vars = homeotic::pset_with_hrefs(&c.pset, c.tail_slots).variables;
    serde_json::json!({
        "file": file,
        "function": "main",
        "head_length": c.head_length,
        "tail_length": c.head_length * (K - 1) + 1,
        "k": K,
        "head_genes": 1,
        "tail_slots": c.tail_slots,
        "tail_filled": c.folded.filled,
        "pset": { "variables": pset_vars, "functions": functions, "rnc_values": c.pset.rnc_values },
        "genes": c.genes.iter().zip(&c.ty_codes).map(|((head, tail), codes)| serde_json::json!({ "head": head.iter().map(tok).collect::<Vec<_>>(), "tail": tail.iter().map(tok).collect::<Vec<_>>(), "ty_codes": codes })).collect::<Vec<_>>(),
        "refused_forms": 0,
        "refused_lineage": 0,
        "refused": {},
        "placements": c.roots.iter().map(|_| serde_json::Value::Null).collect::<Vec<_>>(),
        "roots": c.folded.head,
        "folded": { "head": c.folded.head, "tail": c.folded.tail },
    })
}

/// The whole dump file: the kingdom's header and one function.
pub fn dump_file(file: &str, kingdom: &WgslKingdom, c: &SafeChromosome) -> serde_json::Value {
    serde_json::json!({
        "kingdom": { "name": "safeWGSL", "duals": kingdom.duals.len(), "instances": kingdom.rows.len(), "k_max": K, "tables": "kingdoms/safewgsl/{types,functions}.tsv; fuller WgslKingdom::load_safe()" },
        "functions": [dump_json(file, c)],
    })
}

/// Decoded gene texts (head gene first, then every tail slot) unfolded into
/// the kernel's text.
pub fn unfold_decoded(decoded: &[String], filled: usize) -> Result<String, String> {
    if decoded.is_empty() {
        return Err("no genes".into());
    }
    let folded = FoldedChromosome { head: vec![decoded[0].clone()], tail: decoded[1..].to_vec(), filled, skipped: 0 };
    let roots = homeotic::unfold(&folded)?;
    roots.into_iter().next().ok_or_else(|| "unfold gave no head gene".into())
}

/// A per-kernel row with its duals resolved: `(name, inputs, output)`.
pub type PerKernelRow = (String, Vec<Ty>, Ty);

/// The role's per-kernel rows (the bindings' loads) with their duals
/// resolved; one the kingdom has no dual for is left out (it could never be
/// demanded).
pub fn per_kernel_rows(kingdom: &WgslKingdom, role: &Role) -> Vec<PerKernelRow> {
    role.per_kernel
        .iter()
        // `load.h` reads h at ANY index: in the safe kingdom h is read only at
        // the thread's own cell (`load.h_safe`), and a mutation of a phi load
        // into `load.h` read another thread's cell (8 of 1,000 single-symbol
        // mutants of the corpus, 2026-10-11). The reader never writes it.
        .filter(|p| p.name != "load.h")
        .filter_map(|p| {
            let inputs = p.inputs.iter().map(|d| dual_ty(kingdom, d).ok()).collect::<Option<Vec<_>>>()?;
            Some((p.name.clone(), inputs, dual_ty(kingdom, &p.output).ok()?))
        })
        .collect()
}

/// A gene text projected onto the kingdom's rows and the role's per-kernel
/// rows under the root demand `kernel.safe` (`kernel.module` under an
/// interface of several entry points): the typed tree the renderer takes.
pub fn build(text: &str, kingdom: &WgslKingdom, role: &Role) -> Result<Tree, String> {
    let node = parse_math(text)?;
    let root = dual_ty(kingdom, role.root_dual())?;
    Ok(project(&node, root, kingdom, &per_kernel_rows(kingdom, role)))
}

/// The tree made row-consistent under `demand`: every node's `ty` is the
/// dual its parent's row demands, and a child whose own row has another
/// output is wrapped in the one-input row that converts it (`index.of_count`
/// for a count where an index is demanded). A node with no row under its
/// demand, and no conversion to it, is an error naming both; this is what
/// makes a reader's tree a genotype whose dual codes a typed decoder can
/// trust.
/// The scalar lanes of a vector part: a compose's parts' lanes, a splat's
/// value repeated, a scalar itself, an opaque vector taken apart by `access_i`.
fn lanes(k: &Tree, kingdom: &WgslKingdom) -> Vec<Tree> {
    let d = kingdom.dual(k.ty);
    let l = d.map_or(1, |d| if d.slot.starts_with("vec") { d.lanes as usize } else { 1 });
    if l == 1 {
        return vec![k.clone()];
    }
    let stem = k.name.split('.').take(2).collect::<Vec<_>>().join(".");
    if stem.starts_with("shape.compose") && !k.kids.is_empty() {
        let parts: Vec<Tree> = k.kids.iter().flat_map(|p| lanes(p, kingdom)).collect();
        if parts.len() == l {
            return parts;
        }
    }
    if stem.starts_with("shape.splat") && k.kids.len() == 1 {
        return vec![k.kids[0].clone(); l];
    }
    let scalar = d.and_then(|d| d.scalar().map(|s| s.to_string())).unwrap_or_default();
    let form = d.map(|d| d.form.clone()).unwrap_or_default();
    let sty = crate::wgsl::render::dual_ty(kingdom, &format!("{scalar}.{form}")).unwrap_or(k.ty);
    (0..l).map(|i| Tree::new(&format!("shape.access_{i}"), sty, vec![k.clone()])).collect()
}

pub fn conform(t: &Tree, demand: Ty, kingdom: &WgslKingdom, per_kernel: &[PerKernelRow]) -> Result<Tree, String> {
    // Two conversions at a leaf only when the tree conforms no other way:
    // allowed everywhere, they took over three of 1,000 random trees that
    // one conversion higher up reads to a fixpoint (2026-10-16).
    conform_with(t, demand, kingdom, per_kernel, false).or_else(|e| conform_with(t, demand, kingdom, per_kernel, true).map_err(|_| e))
}

fn conform_with(t: &Tree, demand: Ty, kingdom: &WgslKingdom, per_kernel: &[PerKernelRow], chains: bool) -> Result<Tree, String> {
    conform_at(t, demand, kingdom, per_kernel, true, chains)
}

/// [`conform`]; `convert` false when the node is already being converted:
/// one conversion per node (the conversions between f32 and i32, or an
/// index and a bit field, go both ways, and a node with no row under either
/// dual bounced between them without end, 2026-10-12); `chains` lets a leaf
/// take two conversions, the inner one never converted further.
fn conform_at(t: &Tree, demand: Ty, kingdom: &WgslKingdom, per_kernel: &[PerKernelRow], convert: bool, chains: bool) -> Result<Tree, String> {
    let dual = |ty: Ty| kingdom.dual(ty).map(|d| d.dual.clone()).unwrap_or_else(|| format!("{ty:?}"));
    if t.name == FALLBACK {
        return Ok(Tree::fallback(demand));
    }
    // A splat of a literal where a vector dual with no splat row is demanded
    // (a 64-bit hash state, a wide fixed-point value): its fallback, which
    // the renderer writes as that splat of zero (2026-10-12).
    if let (true, [x]) = (t.name.starts_with("shape.splat"), t.kids.as_slice()) {
        if x.name.starts_with("literal.") && !kingdom.rows.iter().any(|r| r.output == demand && (r.name.starts_with("shape.splat") || r.name.starts_with("literal."))) && kingdom.dual(demand).is_some_and(|d| f64::from(d.fallback_bits) == f64::from(x.konst)) {
            return Ok(Tree::fallback(demand));
        }
    }
    // A count used as an index (index.of_count) where the count is demanded:
    // the count itself (the conversion writes nothing; 2026-10-12).
    if let ("index.of_count", [x]) = (t.name.as_str(), t.kids.as_slice()) {
        if dual(demand) == "u32.count" {
            return conform_at(x, demand, kingdom, per_kernel, convert, chains);
        }
    }
    // `select(-(x), x, s)` over a sign: select.negate_if, which the renderer
    // writes so (its condition is a sign, not a flag; 2026-10-12).
    if let ("select.scalar_cond" | "select.lane_cond", [x, n, c]) = (t.name.as_str(), t.kids.as_slice()) {
        // (naga folds the negation of a literal.)
        fn neg_of(n: &Tree, x: &Tree) -> bool {
            if n.name.starts_with("arith.neg") && n.kids.first() == Some(x) {
                return true;
            }
            if x.name.starts_with("literal.") && n.name.starts_with("literal.") {
                return n.konst == -x.konst;
            }
            // A compose of folded negations, part by part (a constant vector
            // negated: naga folds it into the composed negations, 2026-10-13).
            if x.name.starts_with("shape.compose") && n.name == x.name && n.kids.len() == x.kids.len() && !x.kids.is_empty() {
                return n.kids.iter().zip(&x.kids).all(|(a, b)| neg_of(a, b));
            }
            // A splat of a folded negation (naga writes it as the lanes composed).
            x.name.starts_with("shape.splat") && x.kids.len() == 1 && ((n.name == x.name && n.kids.len() == 1 && neg_of(&n.kids[0], &x.kids[0])) || (n.name.starts_with("shape.compose") && !n.kids.is_empty() && n.kids.iter().all(|l| neg_of(l, &x.kids[0]))))
        }
        let negated = neg_of(n, x);
        if negated {
            if let Ok(r) = conform_at(&Tree::new("select.negate_if", t.ty, vec![x.clone(), c.clone()]), demand, kingdom, per_kernel, convert, chains) {
                return Ok(r);
            }
        }
    }
    // A bit operation the reader named by naga's node, used on an index:
    // the index row of the same operation.
    if let Some(op) = t.name.strip_prefix("bits.").filter(|op| matches!(*op, "shl" | "shr" | "and" | "or")) {
        if signatures(&t.name, demand, t.kids.len(), kingdom, per_kernel).next().is_none() && signatures(&format!("index.{op}"), demand, t.kids.len(), kingdom, per_kernel).next().is_some() {
            return conform_at(&Tree { name: format!("index.{op}"), ty: t.ty, konst: t.konst, kids: t.kids.clone() }, demand, kingdom, per_kernel, convert, chains);
        }
    }
    // A vector composed from vector parts (`vec4<f32>(a: vec2, b: vec2)`,
    // k05): every part taken apart into its lanes first. A part that is
    // itself a compose or a splat gives its own lanes; only an opaque
    // vector part (a load, a register) is taken apart with `access_i`.
    // Taking a compose apart with `access_i` of the whole part copied the
    // part once per lane at every level (`vec2(vec2(a, b).x, vec2(a, b).y)`):
    // the kingdom's own compose chains grew at every reading, and run A's
    // r5 was costed at 3,023 per item for it.
    if t.name.starts_with("shape.compose") {
        let lanes_of = |k: &Tree| kingdom.dual(k.ty).map_or(1, |d| if d.slot.starts_with("vec") { d.lanes as usize } else { 1 });
        let want = kingdom.dual(demand).map_or(0, |d| if d.slot.starts_with("vec") { d.lanes as usize } else { 0 });
        let total: usize = t.kids.iter().map(lanes_of).sum();
        // The step form the rule below builds (`compose<n>(compose<n-1>, s)`)
        // is left as it is: flattening it again would undo that rule forever.
        let step_form = t.name == format!("shape.compose{want}") && t.kids.len() == 2 && lanes_of(&t.kids[1]) == 1 && t.kids[0].name.starts_with("shape.compose") && lanes_of(&t.kids[0]) + 1 == want;
        if want > 0 && total == want && !step_form && t.kids.iter().any(|k| lanes_of(k) > 1) {
            let flat = t.kids.iter().flat_map(|k| lanes(k, kingdom)).collect::<Vec<_>>();
            return conform_at(&Tree { name: format!("shape.compose{want}"), ty: t.ty, konst: t.konst, kids: flat }, demand, kingdom, per_kernel, convert, chains);
        }
    }
    // A vector written from all its lanes at once (naga's Compose): the
    // table composes in steps, a vector of n − 1 lanes and a scalar.
    if let Some(n) = t.name.strip_prefix("shape.compose").and_then(|n| n.parse::<usize>().ok()) {
        if t.kids.len() == n && n > 2 {
            // Each step typed by its own width (a vec2 typed as the vec3 it
            // builds was taken apart as three lanes, rebuilt, and grew
            // without end: a 3-lane swizzle hung the reader, 2026-10-12).
            let width = |w: usize| -> Ty {
                kingdom.dual(demand).or_else(|| kingdom.dual(t.ty)).and_then(|d| d.scalar().map(|s| format!("vec{w}<{s}>.{}", d.form))).and_then(|name| crate::wgsl::render::dual_ty(kingdom, &name).ok()).unwrap_or(t.ty)
            };
            let mut acc = Tree::new("shape.compose2", width(2), vec![t.kids[0].clone(), t.kids[1].clone()]);
            for (i, k) in t.kids.iter().enumerate().skip(2) {
                acc = Tree::new(&format!("shape.compose{}", i + 1), width(i + 1), vec![acc, k.clone()]);
            }
            return conform_at(&acc, demand, kingdom, per_kernel, convert, chains);
        }
    }
    if t.kids.is_empty() && t.name.starts_with("literal.") {
        if kingdom.rows.iter().any(|r| r.name == t.name && r.output == demand) {
            return Ok(Tree::literal(&t.name, demand, t.konst));
        }
        // A literal's form is the inference's guess (an index where a count
        // is demanded, `extractBits(n, 2u, 3u)`): the literal row of the
        // demanded dual, when its value is that form's (an integer form
        // takes a whole number, never a negative one for u32).
        let whole = t.konst.fract() == 0.0;
        if let Some(r) = kingdom.rows.iter().find(|r| r.name.starts_with("literal.") && r.inputs.is_empty() && r.output == demand && matches!(r.name.as_str(), "literal.index" | "literal.count" | "literal.bits") && whole && t.konst >= 0.0 && matches!(t.name.as_str(), "literal.index" | "literal.count" | "literal.bits")) {
            return Ok(Tree::literal(&r.name, demand, t.konst));
        }
        // A dual with no literal row (a hash state, a fixed-point value): its
        // fallback, the zero the renderer writes for it, when the literal is it
        // (2026-10-12: 48 of 1,000 random trees).
        if !kingdom.rows.iter().any(|r| r.name.starts_with("literal.") && r.output == demand) && kingdom.dual(demand).is_some_and(|d| d.lanes <= 1 && f64::from(d.fallback_bits) == f64::from(t.konst)) {
            return Ok(Tree::fallback(demand));
        }
    } else {
        // Every row of the name under the demand; the one that needs the
        // fewest conversions inserted below it wins (a comparison of two
        // indices takes the index row, not the real row with both sides
        // converted), the table's first on a tie.
        let mut errors = Vec::new();
        let mut best: Option<(usize, Vec<Tree>)> = None;
        for inputs in signatures(&t.name, demand, t.kids.len(), kingdom, per_kernel) {
            match t.kids.iter().zip(inputs).map(|(k, &d)| conform_with(k, d, kingdom, per_kernel, chains)).collect::<Result<Vec<_>, _>>() {
                Ok(kids) => {
                    let inserted = kids.iter().map(Tree::size).sum::<usize>() - t.kids.iter().map(Tree::size).sum::<usize>();
                    if best.as_ref().is_none_or(|(b, _)| inserted < *b) {
                        best = Some((inserted, kids));
                    }
                    // No conversion inserted: none can do better (the table's
                    // first such signature wins, as on a tie). Trying every
                    // signature of every node beneath was exponential in the
                    // overloads (a swizzle's composed lanes inside a call hung
                    // the reader, 2026-10-12).
                    if inserted == 0 {
                        break;
                    }
                }
                Err(e) => errors.push(format!("{} > {e}", t.name)),
            }
        }
        if let Some((_, kids)) = best {
            return Ok(Tree { name: t.name.clone(), ty: demand, konst: t.konst, kids });
        }
        if let Some(e) = errors.into_iter().next() {
            return Err(e);
        }
    }
    // The same operation in the demanded form's class (`max` read as
    // index.max where a count is demanded is count.max: the inference chose
    // the other form; 2026-10-12).
    if let (Some((class, op)), Some(d)) = (t.name.split_once('.'), kingdom.dual(demand)) {
        let twin = format!("{}.{op}", d.form);
        if class != d.form && !t.kids.is_empty() && kingdom.rows.iter().any(|r| r.name == twin && r.output == demand && r.inputs.len() == t.kids.len()) {
            if let Ok(x) = conform_at(&Tree { name: twin, ty: t.ty, konst: t.konst, kids: t.kids.clone() }, demand, kingdom, per_kernel, convert, chains) {
                return Ok(x);
            }
        }
    }
    // No row under the demand: a one-input row that converts the node's
    // own output to the demand.
    let own: Vec<Ty> = {
        let mut v: Vec<Ty> = kingdom.rows.iter().filter(|r| r.name == t.name && (r.inputs.len() == t.kids.len() || t.name.starts_with("literal."))).map(|r| r.output).collect();
        v.extend(per_kernel.iter().filter(|(n, i, _)| *n == t.name && i.len() == t.kids.len()).map(|(_, _, o)| *o));
        v.dedup();
        v
    };
    // Every output with a conversion is tried in turn: one output's failing
    // to conform says nothing about the next one's.
    let mut inner_errors = Vec::new();
    for o in own.iter().filter(|_| convert) {
        // A conversion between forms, or a splat of a scalar where a vector is demanded (WGSL's scalar-vector arithmetic).
        if let Some(conv) = kingdom.rows.iter().find(|r| r.inputs.len() == 1 && r.inputs[0] == *o && r.output == demand && (r.name.starts_with("index.of_") || r.name.starts_with("convert.") || r.name.starts_with("shape.splat"))) {
            match conform_at(t, *o, kingdom, per_kernel, false, chains) {
                Ok(inner) => return Ok(Tree::new(&conv.name, demand, vec![inner])),
                Err(e) => inner_errors.push(e),
            }
        }
    }
    // Two conversions, when no one converts: `bits.find_lsb(convert.index_to_bits(
    // index.of_gid(gid)))` renders `firstTrailingBit(gid)` (neither conversion
    // writes text) and reads back as find_lsb over the thread id, a gid where
    // bits are demanded (random seeds 31 and 44, 2026-10-16).
    let is_conv = |r: &&crate::wgsl::loader::WgslRow| r.inputs.len() == 1 && (r.name.starts_with("index.of_") || r.name.starts_with("convert.") || r.name.starts_with("shape.splat"));
    for o in own.iter().filter(|_| convert && chains && t.kids.is_empty()) {
        for first in kingdom.rows.iter().filter(is_conv).filter(|r| r.inputs[0] == *o && r.output != demand) {
            if let Some(second) = kingdom.rows.iter().filter(is_conv).find(|r| r.inputs[0] == first.output && r.output == demand) {
                match conform_at(t, *o, kingdom, per_kernel, false, chains) {
                    Ok(inner) => return Ok(Tree::new(&second.name, demand, vec![Tree::new(&first.name, first.output, vec![inner])])),
                    Err(e) => inner_errors.push(e),
                }
            }
        }
    }
    if let Some(e) = inner_errors.into_iter().next() {
        return Err(e);
    }
    Err(format!("{} (of {}) where {} is demanded: no row and no conversion", t.name, own.iter().map(|&o| dual(o)).collect::<Vec<_>>().join("/"), dual(demand)))
}

/// The number of fallback leaves a projection made.
pub fn fallbacks(t: &Tree) -> usize {
    usize::from(t.name == FALLBACK) + t.kids.iter().map(fallbacks).sum::<usize>()
}

/// The candidate signatures of a name under a demand: the kingdom's rows
/// and the per-kernel rows, each as its input duals.
fn signatures<'a>(name: &'a str, demand: Ty, arity: usize, kingdom: &'a WgslKingdom, per_kernel: &'a [(String, Vec<Ty>, Ty)]) -> impl Iterator<Item = &'a [Ty]> + 'a {
    // (No f16 input: naga 0.20 has no f16, so no such row renders; a
    // conversion to f16 chosen here failed to render, 2026-10-12.)
    let f16 = move |t: &Ty| kingdom.dual(*t).is_some_and(|d| d.scalar() == Some("f16"));
    let rows = kingdom.rows.iter().filter(move |r| r.name == name && r.output == demand && r.inputs.len() == arity && !r.inputs.iter().any(f16)).map(|r| r.inputs.as_slice());
    let pk = per_kernel.iter().filter(move |(n, i, o)| n == name && *o == demand && i.len() == arity).map(|(_, i, _)| i.as_slice());
    rows.chain(pk)
}

fn project(n: &MathNode, demand: Ty, kingdom: &WgslKingdom, per_kernel: &[(String, Vec<Ty>, Ty)]) -> Tree {
    project_scored(n, demand, kingdom, per_kernel).0
}

/// [`project`] with its cost: the nodes the gene did not ask to be
/// fallbacks that became one. A typed fallback under its own dual costs
/// nothing, under another dual one, so a row's signatures are told apart by
/// the duals its fallback leaves name.
fn project_scored(n: &MathNode, demand: Ty, kingdom: &WgslKingdom, per_kernel: &[(String, Vec<Ty>, Ty)]) -> (Tree, usize) {
    // A typed token under another dual than its position demands costs one.
    let off = |t: Option<Ty>| usize::from(t.is_some_and(|t| t != demand));
    match n {
        MathNode::Num(_) => (Tree::fallback(demand), 1),
        MathNode::Var(token) => {
            let (name, t) = split_token(token);
            if name == FALLBACK {
                // A fallback asked for: free under its own dual (untyped, as
                // before the duals were written, it costs one).
                (Tree::fallback(demand), t.map_or(1, |_| off(t)))
            } else if signatures(name, demand, 0, kingdom, per_kernel).next().is_some() {
                (Tree::new(name, demand, vec![]), off(t))
            } else {
                (Tree::fallback(demand), 1)
            }
        }
        MathNode::App(token, kids) => {
            let (ctor, t) = split_token(token);
            if ctor.starts_with("literal.") {
                let v = match kids.as_slice() {
                    [MathNode::Num(v)] => *v as f32,
                    _ => 0.0,
                };
                return if kingdom.rows.iter().any(|r| r.name == ctor && r.output == demand) { (Tree::literal(ctor, demand, v), off(t)) } else { (Tree::fallback(demand), 1) };
            }
            // Every row of the name under the demand with this arity; the
            // one whose inputs cost least wins (the first on a tie).
            let mut best: Option<(usize, Tree)> = None;
            for inputs in signatures(ctor, demand, kids.len(), kingdom, per_kernel) {
                let mut cost = off(t);
                let kids: Vec<Tree> = kids
                    .iter()
                    .zip(inputs)
                    .map(|(k, &d)| {
                        let (t, c) = project_scored(k, d, kingdom, per_kernel);
                        cost += c;
                        t
                    })
                    .collect();
                if best.as_ref().is_none_or(|(bc, _)| cost < *bc) {
                    best = Some((cost, Tree::new(ctor, demand, kids)));
                }
                if cost == 0 {
                    break;
                }
            }
            best.map_or_else(|| (Tree::fallback(demand), 1), |(c, t)| (t, c))
        }
    }
}

/// The kernel text of a safe tree, headed by its `// grid x y bx by` line:
/// the fitness evaluation dispatches from that header (ceil(d / (x·bx)) by
/// ceil(rows / (y·by)) workgroups); without it a register-blocked kernel is
/// dispatched at 1 × 1 blocks, sixteen times the workgroups for a 4 × 4
/// block, and runs at a fraction of its speed.
pub fn render_tree(tree: &Tree, kingdom: &WgslKingdom) -> Result<Rendered, String> {
    render_tree_with(tree, kingdom, &gram_role())
}

/// [`render_tree`] under an interface: the grid header, the interface's
/// header (none for the Gram's), the kernel.
pub fn render_tree_with(tree: &Tree, kingdom: &WgslKingdom, role: &crate::wgsl::render::Role) -> Result<Rendered, String> {
    let t = FunctionTable::safe();
    let mut r = Renderer::new(kingdom, &t, role).kernel(tree)?;
    let g = r.grid;
    // A kernel that writes a mirror says so (`// mode full`): the harness then
    // gives it the whole H in one row band (agreed 2026-10-14).
    let mode = if crate::wgsl::render::writes_mirror(tree) { "// mode full\n" } else { "" };
    // A module of several entry points: each entry point's grid on a line of
    // its own (`// entry search grid 64 1 1 1`), after the first's header.
    let entries: String = if r.entries.len() > 1 { r.entries.iter().map(|(name, grid)| format!("{}\n", grid.entry_header(name))).collect() } else { String::new() };
    r.text = format!("{}\n{entries}{mode}{}{}", g.header(), crate::wgsl::render::interface_header(role), r.text);
    Ok(r)
}

/// Decoded genes to kernel text: unfold, project, render; with the number of
/// fallback leaves the projection made.
pub fn rebuild(decoded: &[String], filled: usize, kingdom: &WgslKingdom) -> Result<(Rendered, usize), String> {
    rebuild_bounded(decoded, filled, kingdom, None).map(|(r, f, _)| (r, f))
}

/// [`rebuild`] under a role (an inferred interface's, nluca's).
pub fn rebuild_with(decoded: &[String], filled: usize, kingdom: &WgslKingdom, role: &crate::wgsl::render::Role) -> Result<(Rendered, usize), String> {
    rebuild_bounded_with(decoded, filled, kingdom, None, role).map(|(r, f, _)| (r, f))
}

/// The static cost bound a kernel must stay under BEFORE it is dispatched:
/// work the GPU has started cannot be cancelled, and a legal kernel of
/// bounded loops can still take hours (three nested tile loops of 64
/// inside the item loop). Per item: `SHIPPED_COST_FACTOR` times the shipped
/// kernel's per-thread cost; base: a fixed 20,000 operations.
///
/// Since 2026-10-11 this is only the LOOSE guard against a kernel too big
/// to run even at the correctness cases' sizes: the fitness measures each
/// kernel's time as it runs and stops it before a shape its measured rate
/// says is too long (`gram::Meter`, stage "slow"). The factor was 64
/// (2,560 per item) and it was binding: the largest estimate of any
/// feasible kernel in the fit_A2 ledgers of 2026-10-11 was 2,499, and the
/// eleven kernels it kept out in that day's `logs/HGP/001` ledgers
/// estimated 3,370 to 14,713 per item; four times that day the estimate
/// was an estimator or reader over-count that kept out a correct kernel
/// 8-9x faster than shipped. At 640
/// (25,600 per item) all eleven are dispatched and measured; three nested
/// tile loops of 64 (2.6e5 times shipped) are still kept out. The base was
/// never binding (no kernel of 2026-10-11 exceeded it) and is unchanged.
pub const SHIPPED_COST_FACTOR: f64 = 640.0;

pub fn default_bound(kingdom: &WgslKingdom) -> Result<Cost, String> {
    let shipped = cost(&gram_shipped_safe_tree(kingdom)?);
    Ok(Cost { per_item: SHIPPED_COST_FACTOR * shipped.per_item, base: 20_000.0 })
}

/// [`rebuild`] refusing, by name, a kernel whose static cost exceeds
/// `bound` (the per-item or the base figure); the kernel's cost is
/// returned with the text.
pub fn rebuild_bounded(decoded: &[String], filled: usize, kingdom: &WgslKingdom, bound: Option<&Cost>) -> Result<(Rendered, usize, Cost), String> {
    rebuild_bounded_with(decoded, filled, kingdom, bound, &gram_role())
}

/// [`rebuild_bounded`] under a role (an inferred interface's, nluca's).
pub fn rebuild_bounded_with(decoded: &[String], filled: usize, kingdom: &WgslKingdom, bound: Option<&Cost>, role: &crate::wgsl::render::Role) -> Result<(Rendered, usize, Cost), String> {
    let text = unfold_decoded(decoded, filled)?;
    let tree = build(&text, kingdom, role)?;
    // A grid the knob references cannot be resolved against is named before the cost.
    crate::wgsl::params::resolve(&tree)?;
    let f = fallbacks(&tree);
    let c = cost(&tree);
    if let Some(b) = bound {
        if c.per_item > b.per_item || c.base > b.base {
            return Err(format!("cost: {:.0} per item, {:.0} base exceeds the bound {:.0}, {:.0}; not dispatched", c.per_item, c.base, b.per_item, b.base));
        }
    }
    Ok((render_tree_with(&tree, kingdom, role)?, f, c))
}

#[cfg(test)]
mod tests {
    /// The two drafts k15 (split accumulators) and k07 (workgroup tile and a
    /// 4 x 4 register block) share their thread-to-cell geometry and store
    /// guard: the largest shared section has over thirty nodes and occurs in
    /// both; sections come largest first.
    #[test]
    fn two_drafts_share_their_geometry() {
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/logs/HGP/001/gen0"));
        let (Ok(a), Ok(b)) = (std::fs::read_to_string(dir.join("k15.wgsl")), std::fs::read_to_string(dir.join("k07.wgsl"))) else { return };
        let kingdom = crate::wgsl::loader::WgslKingdom::load_safe();
        let ta = crate::wgsl::safe_reader::read_safe(&a, &kingdom).expect("k15 reads").tree;
        let tb = crate::wgsl::safe_reader::read_safe(&b, &kingdom).expect("k07 reads").tree;
        let shared = super::shared_sections(&ta, &tb, 5);
        assert!(shared.len() >= 3, "{} shared sections", shared.len());
        assert!(shared[0].nodes >= 30, "largest has {} nodes", shared[0].nodes);
        assert!(shared.windows(2).all(|w| w[0].nodes >= w[1].nodes), "largest first");
        for d in shared.iter().take(3) {
            assert!(super::occurrences_of(&ta, &d.text) >= 1 && super::occurrences_of(&tb, &d.text) >= 1, "{} occurs in both", d.text);
        }
        assert_eq!(super::occurrences_of(&ta, "(Var \"no.such\")"), 0);
    }

    use super::*;
    use crate::karva::karva_to_terms_generic;
    use crate::wgsl::render::{gram_shipped_safe_tree, validate_text};
    use crate::wgsl::safe_reader::read_safe;

    const SHIPPED: &str = include_str!("../../kingdoms/wgsl/gram/shipped.wgsl");

    /// Encode at `h`, decode every gene, unfold, project, render: the text
    /// must be the tree's own rendering, with no fallback.
    fn round_trip(tree: &Tree, kingdom: &WgslKingdom, h: usize, slots: usize) -> SafeChromosome {
        let direct = render_tree(tree, kingdom).unwrap().text;
        let c = encode(tree, h, slots, 7).unwrap_or_else(|e| panic!("{e}"));
        let pset = homeotic::pset_with_hrefs(&c.pset, slots);
        let decoded: Vec<String> = c.genes.iter().map(|(hd, tl)| karva_to_terms_generic(hd, tl, &pset).unwrap()).collect();
        let bound = default_bound(kingdom).unwrap();
        let (r, f, _) = rebuild_bounded(&decoded, c.folded.filled, kingdom, Some(&bound)).unwrap_or_else(|e| panic!("{e}"));
        let text = unfold_decoded(&decoded, c.folded.filled).unwrap();
        assert_eq!(f, 0, "fallbacks in {}\nfrom {text}", build(&text, kingdom, &gram_role()).unwrap().to_sexpr());
        assert_eq!(r.text, direct);
        validate_text(&r.text).unwrap();
        for (g, codes) in c.genes.iter().zip(&c.ty_codes) {
            assert_eq!(g.1.len(), h * (K - 1) + 1);
            assert_eq!(g.0.len(), h);
            if !codes.is_empty() {
                assert!(codes.iter().all(|&code| Ty::from_code(code).is_some()));
            }
        }
        c
    }

    #[test]
    fn the_shipped_safe_tree_round_trips_through_karva() {
        let k = WgslKingdom::load_safe();
        let tree = gram_shipped_safe_tree(&k).unwrap();
        let c = round_trip(&tree, &k, 16, 8);
        assert!(c.folded.filled >= 1, "the body is hoisted");
        assert_eq!(c.roots[0], Some(dual_ty(&k, "kernel.safe").unwrap()));
        assert_eq!(c.ty_codes[0][0], dual_ty(&k, "kernel.safe").unwrap().code());
    }

    #[test]
    fn the_shipped_kernel_read_into_the_safe_grammar_round_trips() {
        let k = WgslKingdom::load_safe();
        let src = format!("// grid 16 16 1 1\n{SHIPPED}");
        let r = read_safe(&src, &k).unwrap();
        round_trip(&r.tree, &k, 24, 16);
    }

    #[test]
    fn a_definition_references_only_lower_slots() {
        let k = WgslKingdom::load_safe();
        let src = format!("// grid 16 16 1 1\n{SHIPPED}");
        let r = read_safe(&src, &k).unwrap();
        let s = split(&r.tree, 8).unwrap();
        for (i, t) in s.tail.iter().enumerate() {
            fn hrefs(t: &Tree, out: &mut Vec<usize>) {
                if let Some(s) = href_slot(&t.name) {
                    out.push(s);
                }
                t.kids.iter().for_each(|k| hrefs(k, out));
            }
            let mut refs = Vec::new();
            hrefs(t, &mut refs);
            assert!(refs.iter().all(|&r| r < i), "slot {i} references {refs:?}");
            assert!(head_needed(t) <= 8);
        }
        assert!(head_needed(&s.head) <= 8);
    }

    /// Every drafted kernel the safe reader converts round-trips at the
    /// population's layout (head 32, 48 slots): encode, decode, unfold,
    /// project, render, the text equal to the tree's own rendering.
    #[test]
    fn the_converted_population_round_trips_through_karva() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/HGP/001/gen0");
        let Ok(entries) = std::fs::read_dir(&dir) else { return };
        let k = WgslKingdom::load_safe();
        let mut files: Vec<_> = entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
        files.sort();
        let (mut converted, mut slots_max) = (0, 0);
        for f in &files {
            let src = std::fs::read_to_string(f).unwrap();
            let Ok(r) = read_safe(&src, &k) else { continue };
            let c = round_trip(&r.tree, &k, 32, 48);
            assert!(c.ty_codes.iter().take(c.folded.filled + 1).all(|codes| !codes.is_empty()), "{}: a gene without codes", f.display());
            slots_max = slots_max.max(c.folded.filled);
            converted += 1;
        }
        assert!(converted >= 26, "{converted} converted");
        assert!(slots_max <= 48, "{slots_max} slots");
    }

    /// A legal kernel of bounded loops that would still run for hours
    /// (three nested tile loops of 64 inside the item loop) is refused by
    /// the cost bound before any dispatch; the shipped kernel passes it.
    #[test]
    fn a_kernel_over_the_cost_bound_is_refused_before_dispatch() {
        let k = WgslKingdom::load_safe();
        let bound = default_bound(&k).unwrap();
        let src = format!("// grid 16 16 1 1\n{SHIPPED}");
        let r = read_safe(&src, &k).unwrap();
        let shipped_genes = vec![karva_text(&r.tree)];
        assert!(rebuild_bounded(&shipped_genes, 0, &k, Some(&bound)).is_ok());
        // The item loop's body wrapped in three tile loops of 64.
        let mut t = r.tree.clone();
        fn wrap(t: &mut Tree, k: &WgslKingdom) -> bool {
            if t.name == "loop.items_boundm" {
                let body = t.kids[1].clone();
                let lit = |v: f32| Tree::literal("literal.lit64", dual_ty(k, "u32.lit64").unwrap(), v);
                let l3 = Tree::new("loop.tile_L3", dual_ty(k, "stmt.L3").unwrap(), vec![lit(64.0), Tree::new("block.one_L4", dual_ty(k, "block.L4").unwrap(), vec![Tree::new("stmt.block_L4", dual_ty(k, "stmt.L4").unwrap(), vec![Tree { ty: dual_ty(k, "block.L4").unwrap(), ..body.clone() }])])]);
                let l2 = Tree::new("loop.tile_L2", dual_ty(k, "stmt.L2").unwrap(), vec![lit(64.0), Tree::new("block.one_L3", dual_ty(k, "block.L3").unwrap(), vec![l3])]);
                let l1 = Tree::new("loop.tile_L1", dual_ty(k, "stmt.L1").unwrap(), vec![lit(64.0), Tree::new("block.one_L2", dual_ty(k, "block.L2").unwrap(), vec![l2])]);
                t.kids[1] = Tree::new("block.one_L1", dual_ty(k, "block.L1").unwrap(), vec![l1]);
                return true;
            }
            t.kids.iter_mut().any(|c| wrap(c, k))
        }
        assert!(wrap(&mut t, &k));
        let heavy = vec![karva_text(&t)];
        let e = rebuild_bounded(&heavy, 0, &k, Some(&bound)).expect_err("refused");
        assert!(e.starts_with("cost:"), "{e}");
        assert!(cost(&t).per_item > bound.per_item);
    }

    /// The rendered text carries the grid header the fitness evaluation
    /// dispatches from, with the tree's own block factors.
    #[test]
    fn the_rendered_text_carries_its_grid_header() {
        let k = WgslKingdom::load_safe();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/HGP/001/gen0/k07.wgsl");
        let Ok(src) = std::fs::read_to_string(&dir) else { return };
        let r = read_safe(&src, &k).unwrap();
        let out = render_tree(&r.tree, &k).unwrap();
        assert!(out.text.starts_with("// grid 16 16 4 4\n"), "{}", out.text.lines().next().unwrap_or(""));
        assert_eq!(crate::wgsl::safe_reader::grid_header(&out.text), Some((16, 16, 4, 4)));
    }

    /// A store into a vec4 private array whose array leaf is of the wrong
    /// kind: the leaf becomes the vec4 array's fallback and the text names
    /// the vec4 array, so it validates (a vec4 into `pa` did not).
    #[test]
    fn an_array_fallback_names_the_array_of_its_dual() {
        let k = WgslKingdom::load_safe();
        let text = r#"(kernel.entry_safe (grid.of (literal.wgdim (Num 16.0)) (literal.wgdim (Num 16.0)) (literal.blk (Num 1.0)) (literal.blk (Num 1.0))) (block.seq2_L0 (parr.set_v_L0 (Var "parr.a") (safe.pidx16 (literal.index (Num 0.0))) (reg.get_v (Var "reg.v0"))) (wg.set_v_L0 (Var "wg.a") (safe.wgv_of (literal.index (Num 0.0))) (reg.get_v (Var "reg.v0")))) (block.one_O0 (store.h_O0 (safe.cell_of (safe.thread_row (Var "builtin.global_invocation_id") (literal.blk (Num 1.0))) (safe.thread_col (Var "builtin.global_invocation_id") (literal.blk (Num 1.0)))) (reg.get_f (Var "reg.f0")))))"#;
        let t = build(text, &k, &gram_role()).unwrap();
        assert_eq!(fallbacks(&t), 2, "{}", t.to_sexpr());
        let r = render_tree(&t, &k).unwrap();
        assert!(r.text.contains("pv[") && r.text.contains("wg_va["), "{}", r.text);
        validate_text(&r.text).unwrap();
    }

    /// The population fold: two kernels that share a subtree get it as ONE
    /// component at the same reserved slot, each references it, and both
    /// still rebuild to their own text.
    #[test]
    fn the_population_fold_shares_a_component_at_one_slot_and_rebuilds_every_kernel() {
        let k = WgslKingdom::load_safe();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/HGP/001/gen0");
        let Ok(entries) = std::fs::read_dir(&dir) else { return };
        let mut files: Vec<_> = entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
        files.sort();
        let trees: Vec<Tree> = files.iter().filter_map(|f| read_safe(&std::fs::read_to_string(f).ok()?, &k).ok()).map(|r| r.tree).take(12).collect();
        assert!(trees.len() >= 8);
        let direct: Vec<String> = trees.iter().map(|t| render_tree(t, &k).unwrap().text).collect();
        // The population's layout: 96 tail slots (k05, inlined, needs 43 own
        // definitions at head 32), the shared band the last eight.
        let f = fold_population(&trees, 32, 96, 88, 3, 7);
        assert!(!f.shared.is_empty(), "components shared by two or more kernels");
        assert!(f.shared.iter().all(|d| d.referrers >= 2 && (88..96).contains(&d.slot)));
        assert!(f.repeated_occurrences > 0);
        let pset_slots = 96;
        for (i, c) in f.chromosomes.iter().enumerate() {
            let c = c.as_ref().unwrap_or_else(|e| panic!("{e}"));
            let pset = homeotic::pset_with_hrefs(&c.pset, pset_slots);
            let decoded: Vec<String> = c.genes.iter().map(|(hd, tl)| karva_to_terms_generic(hd, tl, &pset).unwrap()).collect();
            let (r, fb) = rebuild(&decoded, c.folded.filled, &k).unwrap_or_else(|e| panic!("individual {i}: {e}"));
            assert_eq!(fb, 0);
            assert_eq!(r.text, direct[i], "individual {i} rebuilds to its own text after the fold");
        }
        // Own definitions several kernels hold sit at one slot in each (the
        // engine's crossovers then swap like with like).
        println!("own definitions held by two or more kernels: {} at one common slot, {} not", f.own_aligned, f.own_unaligned);
        assert!(f.own_aligned > 0 && f.own_aligned >= f.own_unaligned);
        // The first component sits at the same slot in every carrier.
        let d = &f.shared[0];
        let carriers = f.chromosomes.iter().filter(|c| c.as_ref().is_ok_and(|c| c.folded.tail[d.slot] == d.text)).count();
        assert_eq!(carriers, d.referrers, "{}", d.text);
    }

    /// THE POPULATION FOLD KEEPS EVERY KERNEL on fit_SF_r4's generation 24
    /// (2026-10-13: the fit stopped, "the fold rewrote row 24 into another
    /// kernel"): the 80 rows of that generation (tests/fold_g24), folded as
    /// the fit folds them (head 32, 128 slots, the shared band the last
    /// eight, seed 4 * 7 + 24), each rebuild to its own text.
    #[test]
    fn the_population_fold_keeps_every_kernel_of_fit_sf_r4_generation_24() {
        let k = WgslKingdom::load_safe();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fold_g24");
        let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
        files.sort();
        let trees: Vec<Tree> = files.iter().map(|f| read_safe(&std::fs::read_to_string(f).unwrap(), &k).unwrap_or_else(|e| panic!("{}: {e}", f.display())).tree).collect();
        let direct: Vec<String> = trees.iter().map(|t| render_tree(t, &k).unwrap().text).collect();
        let f = fold_population(&trees, 32, 128, 120, 3, 4 * 7 + 24);
        println!("shared {}, own aligned {}, unaligned {}", f.shared.len(), f.own_aligned, f.own_unaligned);
        let mut wrong = Vec::new();
        for (i, c) in f.chromosomes.iter().enumerate() {
            let Ok(c) = c else { continue };
            let pset = homeotic::pset_with_hrefs(&c.pset, 128);
            let decoded: Vec<String> = c.genes.iter().map(|(hd, tl)| karva_to_terms_generic(hd, tl, &pset).unwrap()).collect();
            let (r, _) = rebuild(&decoded, c.folded.filled, &k).unwrap_or_else(|e| panic!("{}: {e}", files[i].display()));
            if r.text != direct[i] {
                let first = r.text.lines().zip(direct[i].lines()).find(|(a, b)| a != b).map(|(a, b)| format!("\n  folded: {a}\n  kernel: {b}")).unwrap_or_default();
                wrong.push(format!("{}{first}", files[i].file_name().unwrap().to_string_lossy()));
            }
        }
        assert!(wrong.is_empty(), "{} kernels folded into others:\n{}", wrong.len(), wrong.join("\n"));
    }

    /// A gene text with a token of the wrong dual projects to a fallback
    /// leaf rather than failing.
    #[test]
    fn a_wrong_dual_becomes_a_fallback_leaf() {
        let k = WgslKingdom::load_safe();
        let text = r#"(kernel.entry_safe (grid.of (literal.wgdim (Num 16.0)) (literal.wgdim (Num 16.0)) (literal.blk (Num 1.0)) (literal.blk (Num 1.0))) (block.one_L0 (reg.set_f_L0 (Var "reg.f0") (Var "safe.counter"))) (block.one_O0 (store.h_O0 (safe.cell_of (safe.thread_row (Var "builtin.global_invocation_id") (literal.blk (Num 1.0))) (safe.thread_col (Var "builtin.global_invocation_id") (literal.blk (Num 1.0)))) (reg.get_f (Var "reg.f0")))))"#;
        let t = build(text, &k, &gram_role()).unwrap();
        assert_eq!(fallbacks(&t), 1, "{}", t.to_sexpr());
        assert_eq!(t.kids[1].name, "block.one_L0");
        render_tree(&t, &k).unwrap();
    }

    /// EVERY RANDOM GENE IS A KERNEL THAT STORES (Andrew's rule: a gene the
    /// kingdom admits that cannot be a correct kernel is a kingdom bug; 51 of
    /// 133 of fit_A2's first repair briefs carried an offspring that never
    /// writes h). N random chromosomes at the 128-slot layout (head 32, K 4,
    /// tail 97): gene 0's root is `kernel.entry_safe` (as the engine fixes
    /// it), every other head token is drawn over the whole symbol set (the
    /// kingdom's rows, the role's loads, the 128 hrefs, a literal payload),
    /// every tail token over the terminals; every gene is decoded, unfolded,
    /// projected and rendered, and the text must write h and validate.
    /// Measured 2026-10-11: 0 of 1,000 wrote h before the epilogue, 1,000 of
    /// 1,000 with it, all 1,000 valid.
    #[test]
    fn every_random_gene_renders_a_store_of_h() {
        const N: usize = 1000;
        const HEAD: usize = 32;
        const SLOTS: usize = 128;
        let k = WgslKingdom::load_safe();
        let role = gram_role();
        let tail_len = HEAD * (K - 1) + 1;
        let f16 = |t: &Ty| k.dual(*t).is_some_and(|d| d.scalar() == Some("f16"));
        let mut functions: BTreeMap<String, usize> = BTreeMap::new();
        let mut terminals: BTreeSet<String> = BTreeSet::new();
        for r in k.rows.iter().filter(|r| !f16(&r.output) && !r.inputs.iter().any(f16)) {
            if r.name.starts_with("literal.") {
                functions.insert(r.name.clone(), 1);
            } else if r.inputs.is_empty() {
                terminals.insert(r.name.clone());
            } else {
                functions.insert(r.name.clone(), r.inputs.len());
            }
        }
        let per_kernel = per_kernel_rows(&k, &role);
        for (name, inputs, _) in per_kernel.iter().cloned() {
            if inputs.is_empty() {
                terminals.insert(name);
            }
        }
        for s in 0..SLOTS {
            terminals.insert(href_name(s));
        }
        let functions: Vec<(String, usize)> = functions.into_iter().filter(|(_, a)| *a <= K).collect();
        let terminals: Vec<String> = terminals.into_iter().collect();
        let pset = PsetSpec { variables: terminals.clone(), functions: functions.iter().map(|(n, a)| (n.clone(), FunctionSpec { semantic_id: n.clone(), arity: *a })).collect(), rnc_values: vec![0.0] };
        let mut state = 0x5eed_u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let terminal = |next: &mut dyn FnMut() -> usize| -> Token {
            if next().is_multiple_of(8) {
                Token::Num((next() % 64) as f64)
            } else {
                Token::Var(terminals[next() % terminals.len()].clone())
            }
        };
        let (mut rendered, mut stored, mut failed, mut invalid) = (0usize, 0usize, Vec::new(), Vec::<String>::new());
        for i in 0..N {
            let mut genes: Vec<MathNode> = Vec::with_capacity(SLOTS + 1);
            for g in 0..=SLOTS {
                let head: Vec<Token> = (0..HEAD)
                    .map(|p| {
                        if g == 0 && p == 0 {
                            Token::Func("kernel.entry_safe".into())
                        } else if next().is_multiple_of(2) {
                            Token::Func(functions[next() % functions.len()].0.clone())
                        } else {
                            terminal(&mut next)
                        }
                    })
                    .collect();
                let tail: Vec<Token> = (0..tail_len).map(|_| terminal(&mut next)).collect();
                genes.push(parse_math(&karva_to_terms_generic(&head, &tail, &pset).unwrap_or_else(|e| panic!("chromosome {i} gene {g}: {e}"))).expect("a decoded gene parses"));
            }
            // The typed decode (the engine's): a slot's dual is its root's
            // row's output, and an href where its slot's dual is not the
            // demand is masked to the fallback, as a token of the wrong dual is.
            let root_dual = |n: &MathNode| -> Option<Ty> {
                match n {
                    MathNode::App(c, kids) => k.rows.iter().find(|r| r.name == *c && (r.inputs.len() == kids.len() || c.starts_with("literal."))).map(|r| r.output),
                    MathNode::Var(v) => k.rows.iter().find(|r| r.name == *v && r.inputs.is_empty()).map(|r| r.output).or_else(|| per_kernel.iter().find(|(n, i, _)| n == v && i.is_empty()).map(|(_, _, o)| *o)),
                    MathNode::Num(_) => None,
                }
            };
            let slot_duals: Vec<Option<Ty>> = genes[1..].iter().map(root_dual).collect();
            fn mask(n: &MathNode, demand: Ty, slot_duals: &[Option<Ty>], k: &WgslKingdom, per_kernel: &[PerKernelRow]) -> MathNode {
                match n {
                    MathNode::Var(v) => match href_slot(v) {
                        Some(s) if slot_duals.get(s).copied().flatten() == Some(demand) => n.clone(),
                        Some(_) => MathNode::Var(FALLBACK.into()),
                        None => n.clone(),
                    },
                    MathNode::Num(_) => n.clone(),
                    MathNode::App(c, kids) if c.starts_with("literal.") => MathNode::App(c.clone(), kids.clone()),
                    MathNode::App(c, kids) => match signatures(c, demand, kids.len(), k, per_kernel).next() {
                        Some(inputs) => MathNode::App(c.clone(), kids.iter().zip(inputs).map(|(x, &d)| mask(x, d, slot_duals, k, per_kernel)).collect()),
                        None => MathNode::Var(FALLBACK.into()),
                    },
                }
            }
            fn text(n: &MathNode) -> String {
                match n {
                    MathNode::Num(v) => format!("(Num {v:?})"),
                    MathNode::Var(v) => format!("(Var {v:?})"),
                    MathNode::App(c, kids) => format!("({c}{})", kids.iter().map(|x| format!(" {}", text(x))).collect::<String>()),
                }
            }
            let root = dual_ty(&k, "kernel.safe").expect("the root dual");
            let genes: Vec<String> = genes
                .iter()
                .enumerate()
                .map(|(g, n)| {
                    let demand = if g == 0 { Some(root) } else { slot_duals[g - 1] };
                    demand.map_or_else(|| EMPTY_SLOT.to_string(), |d| text(&mask(n, d, &slot_duals, &k, &per_kernel)))
                })
                .collect();
            let text = match unfold_decoded(&genes, SLOTS).and_then(|u| build(&u, &k, &role)).and_then(|t| render_tree(&t, &k)) {
                Ok(r) => r.text,
                Err(e) => {
                    failed.push(format!("chromosome {i}: {}", &e[..e.len().min(160)]));
                    continue;
                }
            };
            rendered += 1;
            if text.contains("h[") {
                stored += 1;
            }
            if let Err(e) = validate_text(&text) {
                invalid.push(format!("chromosome {i}: {}", e.lines().next().unwrap_or("")));
            }
        }
        println!("{N} random chromosomes: {rendered} rendered, {stored} write h, {} valid WGSL, {} did not render", rendered - invalid.len(), failed.len());
        for e in invalid.iter().take(5) {
            println!("  {e}");
        }
        assert!(failed.is_empty(), "{} did not render:\n{}", failed.len(), failed.iter().take(5).cloned().collect::<Vec<_>>().join("\n"));
        assert_eq!(stored, rendered, "{} of {rendered} rendered kernels never write h", rendered - stored);
        assert!(invalid.is_empty(), "{} rendered kernels naga refuses:\n{}", invalid.len(), invalid.iter().take(5).cloned().collect::<Vec<_>>().join("\n"));
    }

    /// SINGLE-SYMBOL MUTATIONS OF THE CORPUS (fit_A2, 2026-10-11: 69 of 92
    /// engine-made offspring failed correctness, typically one symbol of an
    /// index changed, lid.y for lid.x in a guard). N mutants: a corpus kernel
    /// that reads, one node that is not a literal, its row replaced by another
    /// row of exactly its signature (the engine's head mutation: a function of
    /// the same arity and duals, a terminal of the same dual). Each mutant is
    /// rendered, validated by naga and read back through the ownership proof;
    /// the mutants that change a thread id or its axis are counted apart.
    /// Measured 2026-10-11 with the same 1,000 draws: before the thread's
    /// axes, 1,000 validated, 990 kept the store at the thread's own cell and
    /// 273 changed a thread id or its axis; with them, 1,000, 1,000 and 0.
    #[test]
    fn single_symbol_mutations_of_the_corpus_keep_the_threads_cell() {
        const N: usize = 1000;
        let k = WgslKingdom::load_safe();
        let role = gram_role();
        let per_kernel = per_kernel_rows(&k, &role);
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus/safewgsl"));
        let manifest = std::fs::read_to_string(dir.join("manifest.tsv")).expect("manifest");
        let trees: Vec<Tree> = manifest
            .lines()
            .skip(1)
            .filter(|l| l.split('\t').nth(1) == Some("reads"))
            .filter_map(|l| read_safe(&std::fs::read_to_string(dir.join(format!("{}.wgsl", l.split('\t').next()?))).ok()?, &k).ok())
            .map(|r| r.tree)
            .collect();
        // Every row as (name, inputs, output).
        let mut rows: Vec<(String, Vec<Ty>, Ty)> = k.rows.iter().map(|r| (r.name.clone(), r.inputs.clone(), r.output)).collect();
        rows.extend(per_kernel.iter().cloned());
        fn paths(t: &Tree, at: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if !t.name.starts_with("literal.") && t.name != FALLBACK {
                out.push(at.clone());
            }
            if t.name.starts_with("literal.") {
                return;
            }
            for (i, c) in t.kids.iter().enumerate() {
                at.push(i);
                paths(c, at, out);
                at.pop();
            }
        }
        fn node_at<'t>(t: &'t mut Tree, p: &[usize]) -> &'t mut Tree {
            match p.split_first() {
                None => t,
                Some((i, rest)) => node_at(&mut t.kids[*i], rest),
            }
        }
        let mut state = 0x6d75_7461_u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        // A thread id: the three ids and the local index, a component of an
        // id (as naga writes it, or as the kingdom's axis row).
        let ids = ["builtin.global_invocation_id", "builtin.local_invocation_id", "builtin.workgroup_id", "builtin.local_invocation_index"];
        let thread_id = |t: &Tree| ids.contains(&t.name.as_str()) || t.name.starts_with("axis.") || (t.name.starts_with("shape.access_") && t.kids.first().is_some_and(|k| ids.contains(&k.name.as_str())));
        let (mut valid, mut owned, mut lost, mut other, mut unrendered, mut axis_swaps, mut draws) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut made = 0usize;
        // The first failures, named in the assertion with the mutant's text.
        let mut failures: Vec<String> = Vec::new();
        while made < N {
            draws += 1;
            assert!(draws < 100 * N, "only {made} mutable nodes found in {draws} draws");
            let tree = &trees[next() % trees.len()];
            let mut ps = Vec::new();
            paths(tree, &mut Vec::new(), &mut ps);
            let p = &ps[next() % ps.len()];
            let mut mutant = tree.clone();
            let node = node_at(&mut mutant, p);
            let ins: Vec<Ty> = node.kids.iter().map(|c| c.ty).collect();
            let choices: Vec<&String> = rows.iter().filter(|(n, i, o)| *n != node.name && *o == node.ty && *i == ins && !n.starts_with("literal.")).map(|(n, _, _)| n).collect();
            if choices.is_empty() {
                continue;
            }
            let to = choices[next() % choices.len()].clone();
            let was_id = thread_id(node);
            node.name = to;
            if was_id || thread_id(node) {
                axis_swaps += 1;
            }
            made += 1;
            let text = match render_tree(&mutant, &k) {
                Ok(r) => r.text,
                Err(e) => {
                    unrendered += 1;
                    failures.push(format!("does not render: {e}\n{}", mutant.to_sexpr()));
                    continue;
                }
            };
            if let Err(e) = validate_text(&text) {
                failures.push(format!("invalid: {e}\n{text}"));
                continue;
            }
            valid += 1;
            match read_safe(&text, &k) {
                Ok(_) => owned += 1,
                Err(e) if e.contains("not at this thread's cell") => lost += 1,
                Err(e) => {
                    other += 1;
                    failures.push(format!("does not read back: {e}\n{text}"));
                }
            }
        }
        println!("{N} single-symbol mutants of {} corpus kernels: {valid} validate, {owned} read back with the store at the thread's own cell, {lost} lose the thread's cell, {other} do not read back for another reason, {unrendered} do not render; {axis_swaps} change a thread id or its axis", trees.len());
        assert_eq!((valid, owned, axis_swaps), (N, N, 0), "every mutant validates, keeps the thread's cell and keeps its thread ids and axes; the first failures:\n{}", failures.iter().take(3).cloned().collect::<Vec<_>>().join("\n----\n"));
    }

    /// NLUCA'S SIX KERNELS, AS CUBECL EMITS THEM (2026-10-14): no `// interface`
    /// or `// grid` line; each parses into the kingdom with the interface its
    /// module gives (every binding read_write, the output the one it
    /// stores), encodes into a chromosome, decodes back to the same tree,
    /// renders to text naga validates, and passes every layer of the checker,
    /// layer 7 (write sets) under nluca's dispatch contract.
    #[test]
    fn nlucas_six_kernels_parse_survive_the_genome_render_and_pass_every_layer() {
        let k = WgslKingdom::load_safe();
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus/nluca"));
        let contract = crate::wgsl::safety::Contract::parse(&std::fs::read_to_string(dir.join("nluca.contract")).unwrap()).unwrap();
        for i in 1..=6 {
            let name = format!("k{i:02}");
            let src = std::fs::read_to_string(dir.join(format!("{name}.wgsl"))).unwrap();
            let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{name} does not read: {e}"));
            assert_eq!(r.role.name, crate::wgsl::render::INFERRED, "{name}");
            let tree = r.tree;
            let h = 128usize;
            let slots = split(&tree, h).unwrap().tail.len().div_ceil(8) * 8;
            let c = encode(&tree, h, slots, 7013).unwrap_or_else(|e| panic!("{name} does not encode: {e}"));
            let pset = homeotic::pset_with_hrefs(&c.pset, slots);
            let decoded: Vec<String> = c.genes.iter().map(|(hd, tl)| karva_to_terms_generic(hd, tl, &pset)).collect::<Result<_, _>>().unwrap();
            let back = build(&unfold_decoded(&decoded, c.folded.filled).unwrap(), &k, &r.role).unwrap();
            assert!(back == tree, "{name}: the genome decodes to another tree: {}", crate::wgsl::render::first_difference(&tree, &back).unwrap_or_default());
            let text = render_tree_with(&tree, &k, &r.role).unwrap_or_else(|e| panic!("{name} does not render: {e}")).text;
            validate_text(&text).unwrap_or_else(|e| panic!("{name}: {e}\n{text}"));
            // The entry point keeps the module's name (the harness binds it by name).
            assert!(r.role.entry.starts_with("generated_") && text.contains(&format!("fn {}(", r.role.entry)), "{name}: entry {}", r.role.entry);
            let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract.clone());
            checker.check(&tree).unwrap_or_else(|e| panic!("{name}: {e}"));
            println!("{name}: {} nodes, {} genes, rendered, every layer passed", tree.size(), c.genes.len());
        }
    }

    /// AST TO GENOME TO AST (Andrew, 2026-10-13): the 1,000 seeded sampler
    /// trees of the reader's completeness test, each encoded into a
    /// chromosome (head 128, 128 tail slots, the dumps' layout), every gene
    /// decoded, unfolded and projected: the tree drawn, node for node.
    #[test]
    fn random_trees_survive_the_genome() {
        let (k, role) = (WgslKingdom::load_safe(), gram_role());
        let root = dual_ty(&k, "kernel.safe").unwrap();
        let (h, slots) = (128usize, 128usize);
        let n = 1000u64;
        let mut equal = 0usize;
        let mut failed: Vec<String> = Vec::new();
        for seed in 0..n {
            let tree = crate::wgsl::render::sample(&k, &role, root, 7, seed).unwrap();
            let back: Result<Tree, String> = encode(&tree, h, slots, seed).and_then(|c| {
                let pset = homeotic::pset_with_hrefs(&c.pset, slots);
                let decoded: Vec<String> = c.genes.iter().map(|(hd, tl)| karva_to_terms_generic(hd, tl, &pset)).collect::<Result<_, _>>()?;
                build(&unfold_decoded(&decoded, c.folded.filled)?, &k, &role)
            });
            match back {
                Ok(t) if t == tree => equal += 1,
                Ok(t) => failed.push(format!("seed {seed}: {}", crate::wgsl::render::first_difference(&tree, &t).unwrap_or_default())),
                Err(e) => failed.push(format!("seed {seed}: {}", &e[..e.len().min(160)])),
            }
        }
        println!("random trees through the genome: {equal} of {n} the tree drawn");
        assert!(failed.is_empty(), "{} of {n}:\n{}", failed.len(), failed.iter().take(10).cloned().collect::<Vec<_>>().join("\n"));
    }

    /// THE SAVED GENOMES ARE TODAY'S READING OF THEIR TEXTS (2026-10-17): the
    /// dumps phylu launches from and tests against (`popA2_g006j/karva` and
    /// `library_ep`, head 32, 128 slots, the converter's seed 7013) are each
    /// the encoding of its text as the reader reads it now. They were written
    /// on 2026-10-13 and the reader changed after: g007_r02's `x - x % max(y,
    /// 1u)` was then a subtraction over `index.mod(x, index.max(y, 1))`, today
    /// `index.round_down(x, y)`, so the old genome decoded to `% max(max(y,
    /// 1u), 1u)` and phylu's launch-path test failed on 289 of 450 kernels
    /// with a faithful decode. A reader change that alters a saved tree fails
    /// here first: re-encode with `wgsl_safe_convert <texts> --karva <dir>
    /// --head 32 --slots 128`, keeping the old dumps in a dated directory.
    #[test]
    fn the_saved_genomes_are_todays_reading_of_their_texts() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/HGP/001");
        let k = WgslKingdom::load_safe();
        let (mut checked, mut stale) = (0usize, Vec::new());
        for (dumps, texts) in [("popA2_g006j/karva", "popA2_g006j/texts"), ("library_ep", "library_ep_texts")] {
            let Ok(entries) = std::fs::read_dir(root.join(dumps)) else { return };
            let mut files: Vec<_> = entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
            files.sort();
            for f in &files {
                let id = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                let saved: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap_or_else(|e| panic!("{dumps}/{id}: {e}"));
                let text = std::fs::read_to_string(root.join(texts).join(format!("{id}.wgsl"))).unwrap_or_else(|e| panic!("{dumps}/{id}: its text: {e}"));
                let tree = read_safe(&text, &k).unwrap_or_else(|e| panic!("{dumps}/{id}: its text no longer reads: {}", &e[..e.len().min(300)])).tree;
                let now = dump_file(&format!("{id}.wgsl"), &k, &encode(&tree, 32, 128, 7013).unwrap_or_else(|e| panic!("{dumps}/{id}: {e}")));
                if now != saved {
                    stale.push(format!("{dumps}/{id}"));
                }
                checked += 1;
            }
        }
        println!("saved genomes: {} of {checked} today's reading of their texts", checked - stale.len());
        assert!(stale.is_empty(), "{} of {checked} saved genomes are not today's reading of their texts (re-encode them): {}", stale.len(), stale.iter().take(10).cloned().collect::<Vec<_>>().join(", "));
    }
}
