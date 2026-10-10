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

use std::collections::{BTreeMap, HashMap, VecDeque};

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
    collect(&s.head, &mut variables, &mut functions)?;
    s.tail.iter().try_for_each(|t| collect(t, &mut variables, &mut functions))?;
    if !variables.iter().any(|v| v == FALLBACK) {
        variables.push(FALLBACK.into());
    }
    let pset = PsetSpec { variables, functions: functions.iter().map(|(n, &a)| (n.clone(), FunctionSpec { semantic_id: n.clone(), arity: a })).collect(), rnc_values: vec![0.0] };
    let mut tail: Vec<String> = s.tail.iter().map(karva_text).collect();
    tail.resize(tail_slots, EMPTY_SLOT.to_string());
    let folded = FoldedChromosome { head: vec![karva_text(&s.head)], tail, filled: s.tail.len(), skipped: 0 };
    let genes = homeotic::encode_generic_k(&folded, &pset, rng_seed, head_length, K)?;
    let mut ty_codes = vec![codes_of(&s.head)];
    let mut roots = vec![Some(s.head.ty)];
    for t in &s.tail {
        ty_codes.push(codes_of(t));
        roots.push(Some(t.ty));
    }
    ty_codes.resize(tail_slots + 1, Vec::new());
    roots.resize(tail_slots + 1, None);
    // Each gene's live tokens are as many as its codes.
    for (i, (g, codes)) in genes.iter().zip(&ty_codes).enumerate() {
        let live = live_tokens(&g.0, &g.1, &pset)?;
        if !codes.is_empty() && live != codes.len() {
            return Err(format!("gene {i}: {live} live tokens, {} codes", codes.len()));
        }
    }
    Ok(SafeChromosome { head_length, tail_slots, folded, pset, genes, ty_codes, roots })
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

/// The kernel text of a safe tree.
pub fn render_tree(tree: &Tree, kingdom: &WgslKingdom) -> Result<Rendered, String> {
    let (t, role) = (FunctionTable::safe(), gram_role());
    Renderer::new(kingdom, &t, &role).kernel(tree)
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
/// kernel's per-thread cost (the converted drafts reach 23 times it, their
/// register blocks doing more per thread for fewer threads); base: a fixed
/// 20,000 operations.
pub const SHIPPED_COST_FACTOR: f64 = 64.0;

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
        assert!(converted >= 27, "{converted} converted");
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

    /// A gene text with a token of the wrong dual projects to a fallback
    /// leaf rather than failing.
    #[test]
    fn a_wrong_dual_becomes_a_fallback_leaf() {
        let k = WgslKingdom::load_safe();
        let text = r#"(kernel.entry_safe (grid.of (literal.wgdim (Num 16.0)) (literal.wgdim (Num 16.0)) (literal.blk (Num 1.0)) (literal.blk (Num 1.0))) (block.one_L0 (reg.set_f_L0 (Var "reg.f0") (Var "safe.counter"))))"#;
        let t = build(text, &k, &gram_role()).unwrap();
        assert_eq!(fallbacks(&t), 1, "{}", t.to_sexpr());
        assert_eq!(t.kids[1].name, "block.one_L0");
        render_tree(&t, &k).unwrap();
    }
}
