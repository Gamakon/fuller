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

/// The tree in the Karva text form the encoder and the decoder share: a
/// terminal as `(Var "name")`, a literal as `(literal.x (Num v))`, an
/// application as `(name kids…)`.
pub fn karva_text(t: &Tree) -> String {
    if t.kids.is_empty() {
        if t.name.starts_with("literal.") {
            return format!("({} (Num {:?}))", t.name, f64::from(t.konst));
        }
        return format!("(Var {:?})", t.name);
    }
    format!("({} {})", t.name, t.kids.iter().map(karva_text).collect::<Vec<_>>().join(" "))
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
                functions.insert(t.name.clone(), 1);
            } else if !is_href(t) && !variables.contains(&t.name) {
                variables.push(t.name.clone());
            }
            return Ok(());
        }
        if t.kids.len() > K {
            return Err(format!("{} has {} children, past K = {K}", t.name, t.kids.len()));
        }
        match functions.insert(t.name.clone(), t.kids.len()) {
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
    let chromosomes = trees
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let folded = replace(t, &by_text);
            let s = split(&folded, head_length)?;
            if s.tail.len() > reserved_from {
                return Err(format!("individual {i}: {} own definitions, {reserved_from} slots below the shared band", s.tail.len()));
            }
            let mut tail: Vec<Option<Tree>> = s.tail.into_iter().map(Some).collect();
            tail.resize(tail_slots, None);
            for (d, def) in shared.iter().zip(&defs) {
                // Only the individuals that reference the slot hold the definition.
                if karva_text(&s.head).contains(&format!("(Var \"{}\")", href_name(d.slot))) || tail.iter().flatten().any(|t| karva_text(t).contains(&format!("(Var \"{}\")", href_name(d.slot)))) {
                    tail[d.slot] = Some(def.clone());
                }
            }
            encode_folded(&s.head, &tail, head_length, rng_seed.wrapping_add(i as u64))
        })
        .collect();
    PopulationFold { chromosomes, shared, distinct_subtrees, occurrences, repeated_occurrences }
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
/// rows under the root demand `kernel.safe`: the typed tree the renderer
/// takes.
pub fn build(text: &str, kingdom: &WgslKingdom, role: &Role) -> Result<Tree, String> {
    let node = parse_math(text)?;
    let root = dual_ty(kingdom, "kernel.safe")?;
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
    let dual = |ty: Ty| kingdom.dual(ty).map(|d| d.dual.clone()).unwrap_or_else(|| format!("{ty:?}"));
    if t.name == FALLBACK {
        return Ok(Tree::fallback(demand));
    }
    // A bit operation the reader named by naga's node, used on an index:
    // the index row of the same operation.
    if let Some(op) = t.name.strip_prefix("bits.").filter(|op| matches!(*op, "shl" | "shr" | "and" | "or")) {
        if signatures(&t.name, demand, t.kids.len(), kingdom, per_kernel).next().is_none() && signatures(&format!("index.{op}"), demand, t.kids.len(), kingdom, per_kernel).next().is_some() {
            return conform(&Tree { name: format!("index.{op}"), ty: t.ty, konst: t.konst, kids: t.kids.clone() }, demand, kingdom, per_kernel);
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
        if want > 0 && total == want && t.kids.iter().any(|k| lanes_of(k) > 1) {
            let flat = t.kids.iter().flat_map(|k| lanes(k, kingdom)).collect::<Vec<_>>();
            return conform(&Tree { name: format!("shape.compose{want}"), ty: t.ty, konst: t.konst, kids: flat }, demand, kingdom, per_kernel);
        }
    }
    // A vector written from all its lanes at once (naga's Compose): the
    // table composes in steps, a vector of n − 1 lanes and a scalar.
    if let Some(n) = t.name.strip_prefix("shape.compose").and_then(|n| n.parse::<usize>().ok()) {
        if t.kids.len() == n && n > 2 {
            let mut acc = Tree::new("shape.compose2", t.ty, vec![t.kids[0].clone(), t.kids[1].clone()]);
            for (i, k) in t.kids.iter().enumerate().skip(2) {
                acc = Tree::new(&format!("shape.compose{}", i + 1), t.ty, vec![acc, k.clone()]);
            }
            return conform(&acc, demand, kingdom, per_kernel);
        }
    }
    if t.kids.is_empty() && t.name.starts_with("literal.") {
        if kingdom.rows.iter().any(|r| r.name == t.name && r.output == demand) {
            return Ok(Tree::literal(&t.name, demand, t.konst));
        }
    } else {
        // Every row of the name under the demand; the one that needs the
        // fewest conversions inserted below it wins (a comparison of two
        // indices takes the index row, not the real row with both sides
        // converted), the table's first on a tie.
        let mut errors = Vec::new();
        let mut best: Option<(usize, Vec<Tree>)> = None;
        for inputs in signatures(&t.name, demand, t.kids.len(), kingdom, per_kernel) {
            match t.kids.iter().zip(inputs).map(|(k, &d)| conform(k, d, kingdom, per_kernel)).collect::<Result<Vec<_>, _>>() {
                Ok(kids) => {
                    let inserted = kids.iter().map(Tree::size).sum::<usize>() - t.kids.iter().map(Tree::size).sum::<usize>();
                    if best.as_ref().is_none_or(|(b, _)| inserted < *b) {
                        best = Some((inserted, kids));
                    }
                }
                Err(e) => errors.push(e),
            }
        }
        if let Some((_, kids)) = best {
            return Ok(Tree { name: t.name.clone(), ty: demand, konst: t.konst, kids });
        }
        if let Some(e) = errors.into_iter().next() {
            return Err(e);
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
    for o in &own {
        // A conversion between forms, or a splat of a scalar where a vector is demanded (WGSL's scalar-vector arithmetic).
        if let Some(conv) = kingdom.rows.iter().find(|r| r.inputs.len() == 1 && r.inputs[0] == *o && r.output == demand && (r.name.starts_with("index.of_") || r.name.starts_with("convert.") || r.name.starts_with("shape.splat"))) {
            let inner = conform(t, *o, kingdom, per_kernel)?;
            return Ok(Tree::new(&conv.name, demand, vec![inner]));
        }
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
    let rows = kingdom.rows.iter().filter(move |r| r.name == name && r.output == demand && r.inputs.len() == arity).map(|r| r.inputs.as_slice());
    let pk = per_kernel.iter().filter(move |(n, i, o)| n == name && *o == demand && i.len() == arity).map(|(_, i, _)| i.as_slice());
    rows.chain(pk)
}

fn project(n: &MathNode, demand: Ty, kingdom: &WgslKingdom, per_kernel: &[(String, Vec<Ty>, Ty)]) -> Tree {
    match n {
        MathNode::Num(_) => Tree::fallback(demand),
        MathNode::Var(name) => {
            if signatures(name, demand, 0, kingdom, per_kernel).next().is_some() {
                Tree::new(name, demand, vec![])
            } else {
                Tree::fallback(demand)
            }
        }
        MathNode::App(ctor, kids) => {
            if ctor.starts_with("literal.") {
                let v = match kids.as_slice() {
                    [MathNode::Num(v)] => *v as f32,
                    _ => 0.0,
                };
                return if kingdom.rows.iter().any(|r| r.name == *ctor && r.output == demand) { Tree::literal(ctor, demand, v) } else { Tree::fallback(demand) };
            }
            // Every row of the name under the demand with this arity; the
            // one whose inputs leave the fewest fallbacks wins.
            let mut best: Option<(usize, Tree)> = None;
            for inputs in signatures(ctor, demand, kids.len(), kingdom, per_kernel) {
                let kids: Vec<Tree> = kids.iter().zip(inputs).map(|(k, &d)| project(k, d, kingdom, per_kernel)).collect();
                let t = Tree::new(ctor, demand, kids);
                let f = fallbacks(&t);
                if best.as_ref().is_none_or(|(bf, _)| f < *bf) {
                    best = Some((f, t));
                }
                if f == 0 {
                    break;
                }
            }
            best.map_or_else(|| Tree::fallback(demand), |(_, t)| t)
        }
    }
}

/// The kernel text of a safe tree, headed by its `// grid x y bx by` line:
/// the fitness evaluation dispatches from that header (ceil(d / (x·bx)) by
/// ceil(rows / (y·by)) workgroups); without it a register-blocked kernel is
/// dispatched at 1 × 1 blocks, sixteen times the workgroups for a 4 × 4
/// block, and runs at a fraction of its speed.
pub fn render_tree(tree: &Tree, kingdom: &WgslKingdom) -> Result<Rendered, String> {
    let (t, role) = (FunctionTable::safe(), gram_role());
    let mut r = Renderer::new(kingdom, &t, &role).kernel(tree)?;
    let g = r.grid;
    r.text = format!("{}\n{}", g.header(), r.text);
    Ok(r)
}

/// Decoded genes to kernel text: unfold, project, render; with the number of
/// fallback leaves the projection made.
pub fn rebuild(decoded: &[String], filled: usize, kingdom: &WgslKingdom) -> Result<(Rendered, usize), String> {
    rebuild_bounded(decoded, filled, kingdom, None).map(|(r, f, _)| (r, f))
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
    let text = unfold_decoded(decoded, filled)?;
    let tree = build(&text, kingdom, &gram_role())?;
    let f = fallbacks(&tree);
    let c = cost(&tree);
    if let Some(b) = bound {
        if c.per_item > b.per_item || c.base > b.base {
            return Err(format!("cost: {:.0} per item, {:.0} base exceeds the bound {:.0}, {:.0}; not dispatched", c.per_item, c.base, b.per_item, b.base));
        }
    }
    Ok((render_tree(&tree, kingdom)?, f, c))
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
        // The first component sits at the same slot in every carrier.
        let d = &f.shared[0];
        let carriers = f.chromosomes.iter().filter(|c| c.as_ref().is_ok_and(|c| c.folded.tail[d.slot] == d.text)).count();
        assert_eq!(carriers, d.referrers, "{}", d.text);
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
                Err(_) => {
                    unrendered += 1;
                    continue;
                }
            };
            if validate_text(&text).is_err() {
                continue;
            }
            valid += 1;
            match read_safe(&text, &k) {
                Ok(_) => owned += 1,
                Err(e) if e.contains("not at this thread's cell") => lost += 1,
                Err(_) => other += 1,
            }
        }
        println!("{N} single-symbol mutants of {} corpus kernels: {valid} validate, {owned} read back with the store at the thread's own cell, {lost} lose the thread's cell, {other} do not read back for another reason, {unrendered} do not render; {axis_swaps} change a thread id or its axis", trees.len());
        assert_eq!((valid, owned, axis_swaps), (N, N, 0), "every mutant validates, keeps the thread's cell and keeps its thread ids and axes");
    }
}
