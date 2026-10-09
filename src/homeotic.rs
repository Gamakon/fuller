//! The homeotic tail: the converter from the share finder's output back to a
//! mutated chromosome (`docs/Geneframe_Homeotic_Genes_design.md`, §7).
//!
//! A chromosome is a HEAD of program genes followed by a TAIL of homeotic
//! genes. A tail gene holds a shared partial; a head gene, or an earlier
//! tail gene, reads it through a reference terminal. The share finder
//! (`extract::maximal_shared_saturated`) reports which subtrees repeat across
//! a chromosome's genes; [`fold`] moves each repeated subtree into an empty
//! tail slot and replaces every occurrence with a reference, and [`encode`]
//! turns the result back into Karva genes the engine can carry.
//!
//! **The invariant everything leans on.** `fold` takes the finder's CHOSEN
//! forms (the third value `maximal_shared_saturated` returns), never the
//! original gene text: after `extract_dag::materialize` every class has one
//! member, so two occurrences of one shared subtree render as byte-identical
//! text, and textual structural replacement is sound. On any other input two
//! occurrences may spell the same value differently and the fold would miss
//! one.
//!
//! **The reference's spelling is a contract with phylu's decoder.** A
//! reference to tail slot `t` is the variable `(Var "href<t>")`
//! ([`href_name`], [`href_slot`]). The device opcode is `gpu_eval::Op::GeneRef`
//! with `arg0 = t`; the decoder maps the name to it. Nothing in fuller
//! evaluates a reference numerically, so verification here is structural:
//! [`unfold`] substitutes every definition back and must reproduce the input.
//!
//! **Three rules, all held by construction.** Closure by type: a reference is
//! a terminal of the slot's root type and Design C's projection resolves it
//! like any other token. Acyclicity: a reference points only FORWARD, a tail
//! gene at slot `s` reads slots `u > s` only; `fold` processes matches largest
//! first, and a smaller subtree cannot contain a larger one, so a definition
//! can only reference slots allocated after it; `fold` asserts it anyway.
//! Unread slots are non-coding: an empty slot holds [`EMPTY_SLOT`], which
//! nothing references and the evaluator never runs.

use std::collections::BTreeMap;

use crate::extract::Match;
use crate::karva::{parse_math, terms_to_karva_sized, MathNode, PsetSpec, Token};

/// Prefix of a reference terminal's variable name: `href0`, `href1`, …
pub const HREF_PREFIX: &str = "href";

/// What an unfilled tail slot holds: a constant gene nothing reads.
pub const EMPTY_SLOT: &str = "(Num 0.0)";

/// The reference terminal for tail slot `slot`.
pub fn href_name(slot: usize) -> String {
    format!("{HREF_PREFIX}{slot}")
}

/// The tail slot a variable name refers to, if it is a reference.
pub fn href_slot(name: &str) -> Option<usize> {
    name.strip_prefix(HREF_PREFIX).and_then(|rest| rest.parse::<usize>().ok()).filter(|_| name.len() > HREF_PREFIX.len())
}

/// One Karva gene: its head tokens and its tail tokens.
pub type KarvaGene = (Vec<Token>, Vec<Token>);

/// The folded chromosome: `head` are the program genes with their repeated
/// subtrees replaced by references; `tail` holds exactly `tail_slots`
/// definitions, the unfilled ones [`EMPTY_SLOT`]. Gene order in the engine's
/// layout is `head` then `tail`.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldedChromosome {
    pub head: Vec<String>,
    pub tail: Vec<String>,
    /// Tail slots a definition was written into.
    pub filled: usize,
    /// Matches left inlined: no slot free, too large for the head length, or
    /// swallowed by a larger fold.
    pub skipped: usize,
}

/// What bounds a fold.
#[derive(Debug, Clone, Copy)]
pub struct FoldOptions {
    /// Tail slots in the layout; the fold never writes more definitions.
    pub tail_slots: usize,
    /// A definition's internal operator count must not exceed this, or the
    /// Karva head cannot hold it. `None` = unbounded.
    pub max_definition_ops: Option<usize>,
}

fn render(n: &MathNode) -> String {
    match n {
        // `{:?}` keeps a decimal point or exponent, so egglog reads it as f64.
        MathNode::Num(v) => format!("(Num {v:?})"),
        MathNode::Var(name) => format!("(Var {name:?})"),
        MathNode::App(ctor, children) => {
            let mut s = String::from("(");
            s.push_str(ctor);
            for c in children {
                s.push(' ');
                s.push_str(&render(c));
            }
            s.push(')');
            s
        }
    }
}

fn op_count(n: &MathNode) -> usize {
    match n {
        MathNode::Num(_) | MathNode::Var(_) => 0,
        MathNode::App(_, children) => 1 + children.iter().map(op_count).sum::<usize>(),
    }
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

fn count_occurrences(n: &MathNode, pattern: &str) -> usize {
    if render(n) == pattern {
        return 1;
    }
    match n {
        MathNode::App(_, children) => children.iter().map(|c| count_occurrences(c, pattern)).sum(),
        _ => 0,
    }
}

fn replace_occurrences(n: &MathNode, pattern: &str, with: &MathNode) -> MathNode {
    if render(n) == pattern {
        return with.clone();
    }
    match n {
        MathNode::App(ctor, children) => MathNode::App(ctor.clone(), children.iter().map(|c| replace_occurrences(c, pattern, with)).collect()),
        other => other.clone(),
    }
}

pub(crate) fn hrefs_in(n: &MathNode, out: &mut Vec<usize>) {
    match n {
        MathNode::Var(name) => {
            if let Some(slot) = href_slot(name) {
                out.push(slot);
            }
        }
        MathNode::App(_, children) => children.iter().for_each(|c| hrefs_in(c, out)),
        MathNode::Num(_) => {}
    }
}

/// Fold the finder's matches into a homeotic tail.
///
/// `chosen_forms` are the per-gene forms `maximal_shared_saturated` returned
/// (its third value), `matches` its first value; both are relative to each
/// other. Matches are taken largest first by internal operator count, ties by
/// text, so the result is deterministic and every reference points forward.
/// A match whose occurrences a larger fold already swallowed, or whose
/// definition would not fit the head length, or for which no slot is left, is
/// left inlined and counted in `skipped`.
pub fn fold(chosen_forms: &[String], matches: &[Match], opts: FoldOptions) -> Result<FoldedChromosome, String> {
    let mut head: Vec<MathNode> = chosen_forms.iter().map(|g| parse_math(g)).collect::<Result<_, _>>()?;
    // Candidate definitions, keyed for a deterministic order: largest first.
    let mut candidates: BTreeMap<(std::cmp::Reverse<usize>, String), ()> = BTreeMap::new();
    for m in matches {
        let Some((gene, path)) = m.sites.first() else { continue };
        let root = head.get(*gene).ok_or_else(|| format!("match site in gene {gene}, but only {} forms", head.len()))?;
        let node = node_at(root, path).ok_or_else(|| format!("match path {path:?} does not resolve in gene {gene}"))?;
        candidates.insert((std::cmp::Reverse(op_count(node)), render(node)), ());
    }
    let mut tail: Vec<MathNode> = Vec::new();
    let (mut filled, mut skipped) = (0usize, 0usize);
    for (std::cmp::Reverse(ops), pattern) in candidates.into_keys() {
        if opts.max_definition_ops.is_some_and(|max| ops > max) {
            skipped += 1;
            continue;
        }
        let occurrences: usize = head.iter().chain(tail.iter()).map(|g| count_occurrences(g, &pattern)).sum();
        if occurrences < 2 {
            skipped += 1;
            continue;
        }
        if filled >= opts.tail_slots {
            skipped += 1;
            continue;
        }
        let slot = filled;
        let reference = MathNode::Var(href_name(slot));
        for g in head.iter_mut().chain(tail.iter_mut()) {
            *g = replace_occurrences(g, &pattern, &reference);
        }
        tail.push(parse_math(&pattern)?);
        filled += 1;
    }
    // Forward-only, asserted: a definition at slot s reads only slots > s.
    for (s, def) in tail.iter().enumerate() {
        let mut refs = Vec::new();
        hrefs_in(def, &mut refs);
        if let Some(bad) = refs.iter().find(|&&u| u <= s) {
            return Err(format!("fold: tail slot {s} references slot {bad}, which is not forward"));
        }
    }
    while tail.len() < opts.tail_slots {
        tail.push(parse_math(EMPTY_SLOT)?);
    }
    Ok(FoldedChromosome { head: head.iter().map(render).collect(), tail: tail.iter().map(render).collect(), filled, skipped })
}

/// Substitute every tail definition back into the genes that read it, last
/// slot first so nested references resolve, and return the head genes as
/// text. A correct fold unfolds to its input's canonical rendering.
pub fn unfold(folded: &FoldedChromosome) -> Result<Vec<String>, String> {
    let mut tail: Vec<MathNode> = folded.tail.iter().map(|g| parse_math(g)).collect::<Result<_, _>>()?;
    let mut head: Vec<MathNode> = folded.head.iter().map(|g| parse_math(g)).collect::<Result<_, _>>()?;
    for s in (0..tail.len()).rev() {
        let pattern = render(&MathNode::Var(href_name(s)));
        let def = tail[s].clone();
        for g in head.iter_mut().chain(tail[..s].iter_mut()) {
            *g = replace_occurrences(g, &pattern, &def);
        }
    }
    Ok(head.iter().map(render).collect())
}

/// The canonical rendering `fold`/`unfold` compare against.
pub fn canonical(form: &str) -> Result<String, String> {
    Ok(render(&parse_math(form)?))
}

/// A pset that also admits the reference terminals of `tail_slots` slots.
pub fn pset_with_hrefs(pset: &PsetSpec, tail_slots: usize) -> PsetSpec {
    let mut out = pset.clone();
    for slot in 0..tail_slots {
        let name = href_name(slot);
        if !out.variables.contains(&name) {
            out.variables.push(name);
        }
    }
    out
}

/// Encode the folded chromosome as Karva genes, head genes then tail genes,
/// each at `head_length`. A gene that does not fit is an error: `fold` with
/// `max_definition_ops = Some(head_length)` prevents it for definitions, and a
/// head gene only shrinks under folding.
pub fn encode(folded: &FoldedChromosome, pset: &PsetSpec, rng_seed: u64, head_length: usize) -> Result<Vec<KarvaGene>, String> {
    encode_with(folded, pset, rng_seed, head_length, &terms_to_karva_sized)
}

/// [`encode`] through the generic Karva encoder (`karva::terms_to_karva_generic`):
/// a kingdom whose symbols are `class.instance` names rather than `Math`
/// constructors (the WGSL kingdom) folds and encodes with this one.
pub fn encode_generic(folded: &FoldedChromosome, pset: &PsetSpec, rng_seed: u64, head_length: usize) -> Result<Vec<KarvaGene>, String> {
    encode_with(folded, pset, rng_seed, head_length, &crate::karva::terms_to_karva_generic)
}

/// [`encode_generic`] at the kingdom's `K` (the tail is `head·(K−1)+1` for
/// every gene, whatever symbols the chromosome uses).
pub fn encode_generic_k(folded: &FoldedChromosome, pset: &PsetSpec, rng_seed: u64, head_length: usize, k: usize) -> Result<Vec<KarvaGene>, String> {
    encode_with(folded, pset, rng_seed, head_length, &move |t, p, s, h| crate::karva::terms_to_karva_generic_k(t, p, s, h, Some(k)))
}

type Encoder = dyn Fn(&str, &PsetSpec, u64, Option<usize>) -> Result<(Vec<Token>, Vec<Token>, bool), String>;

fn encode_with(folded: &FoldedChromosome, pset: &PsetSpec, rng_seed: u64, head_length: usize, encoder: &Encoder) -> Result<Vec<KarvaGene>, String> {
    let pset = pset_with_hrefs(pset, folded.tail.len());
    let mut genes = Vec::with_capacity(folded.head.len() + folded.tail.len());
    for (i, g) in folded.head.iter().chain(folded.tail.iter()).enumerate() {
        let (head, tail, oversized) = encoder(g, &pset, rng_seed.wrapping_add(i as u64), Some(head_length))?;
        if oversized {
            return Err(format!("gene {i} needs a head longer than {head_length}: {g}"));
        }
        genes.push((head, tail));
    }
    Ok(genes)
}

#[cfg(test)]
mod tests {
    use super::{canonical, encode, fold, href_name, href_slot, pset_with_hrefs, unfold, FoldOptions, EMPTY_SLOT};
    use crate::extract::{maximal_shared, maximal_shared_saturated, Match};
    use crate::karva::{karva_to_terms, FunctionSpec, PsetSpec};
    use std::collections::HashMap;

    fn opts(slots: usize) -> FoldOptions {
        FoldOptions { tail_slots: slots, max_definition_ops: None }
    }

    fn pset() -> PsetSpec {
        let mut functions = HashMap::new();
        for (name, arity) in [("add", 2), ("mul", 2), ("exp", 1), ("sqrt", 1)] {
            functions.insert(name.to_string(), FunctionSpec { semantic_id: name.into(), arity });
        }
        PsetSpec { variables: vec!["x0".into(), "x1".into(), "x2".into()], functions, rnc_values: vec![] }
    }

    #[test]
    fn href_names_round_trip_and_reject_others() {
        assert_eq!(href_name(3), "href3");
        assert_eq!(href_slot("href3"), Some(3));
        assert_eq!(href_slot("href"), None);
        assert_eq!(href_slot("x0"), None);
    }

    /// An exact cross-gene repeat moves into slot 0 and both genes read it.
    #[test]
    fn exact_cross_gene_repeat_fills_one_slot() {
        let genes = vec![
            r#"(Add (Mul (Var "x0") (Var "x1")) (Var "x2"))"#.to_string(),
            r#"(Sqrt (Mul (Var "x0") (Var "x1")))"#.to_string(),
        ];
        let matches = maximal_shared(&genes, 1).expect("finder");
        let folded = fold(&genes, &matches, opts(2)).expect("fold");
        assert_eq!(folded.filled, 1);
        assert_eq!(folded.tail[0], r#"(Mul (Var "x0") (Var "x1"))"#);
        assert_eq!(folded.tail[1], canonical(EMPTY_SLOT).unwrap());
        assert_eq!(folded.head[0], r#"(Add (Var "href0") (Var "x2"))"#);
        assert_eq!(folded.head[1], r#"(Sqrt (Var "href0"))"#);
        let back = unfold(&folded).expect("unfold");
        let want: Vec<String> = genes.iter().map(|g| canonical(g).unwrap()).collect();
        assert_eq!(back, want, "unfolding a fold reproduces the input");
    }

    /// A smaller repeat inside a larger one: the larger takes slot 0, and
    /// its definition reads the smaller at slot 1 — forward only.
    #[test]
    fn nested_repeat_references_a_later_slot() {
        let small = r#"(Mul (Var "x0") (Var "x1"))"#;
        let big = format!("(Add {small} (Var \"x2\"))");
        // The finder reports the small repeat only where it stands OUTSIDE
        // the larger one (its containment filter), so two such genes exist.
        let genes = vec![
            format!("(Sqrt {big})"),
            format!("(Exp {big})"),
            format!("(Add {small} (Var \"x0\"))"),
            format!("(Mul {small} (Var \"x2\"))"),
        ];
        let matches = maximal_shared(&genes, 1).expect("finder");
        assert_eq!(matches.len(), 2, "the finder reports both repeats");
        let folded = fold(&genes, &matches, opts(3)).expect("fold");
        assert_eq!(folded.filled, 2, "big and small both fold");
        assert_eq!(folded.tail[0], r#"(Add (Var "href1") (Var "x2"))"#, "the larger definition reads the smaller, forward");
        assert_eq!(folded.tail[1], small);
        assert_eq!(folded.head[2], r#"(Add (Var "href1") (Var "x0"))"#);
        assert_eq!(folded.head[3], r#"(Mul (Var "href1") (Var "x2"))"#);
        let want: Vec<String> = genes.iter().map(|g| canonical(g).unwrap()).collect();
        assert_eq!(unfold(&folded).unwrap(), want);
    }

    /// With one slot, the largest fold is taken and the rest stay inlined;
    /// the fold still unfolds exactly.
    #[test]
    fn capacity_keeps_the_largest_and_still_unfolds() {
        let small = r#"(Mul (Var "x0") (Var "x1"))"#;
        let big = format!("(Add {small} (Var \"x2\"))");
        let genes = vec![format!("(Sqrt {big})"), format!("(Exp {big})"), format!("(Add {small} (Var \"x0\"))"), format!("(Mul {small} (Var \"x2\"))")];
        let matches = maximal_shared(&genes, 1).expect("finder");
        let folded = fold(&genes, &matches, opts(1)).expect("fold");
        assert_eq!((folded.filled, folded.skipped), (1, 1));
        assert_eq!(folded.tail[0], format!("(Add {small} (Var \"x2\"))"));
        assert_eq!(folded.head[2], format!("(Add {small} (Var \"x0\"))"), "the small repeat stays inlined");
        let want: Vec<String> = genes.iter().map(|g| canonical(g).unwrap()).collect();
        assert_eq!(unfold(&folded).unwrap(), want);
    }

    /// A definition larger than the head length is skipped, not folded.
    #[test]
    fn a_definition_too_large_for_the_head_is_skipped() {
        let big = r#"(Add (Mul (Var "x0") (Var "x1")) (Mul (Var "x1") (Var "x2")))"#;
        let genes = vec![format!("(Sqrt {big})"), format!("(Exp {big})")];
        let matches = maximal_shared(&genes, 1).expect("finder");
        let folded = fold(&genes, &matches, FoldOptions { tail_slots: 2, max_definition_ops: Some(2) }).expect("fold");
        assert_eq!(folded.filled, 0);
        assert_eq!(folded.skipped, 1);
        assert_eq!(folded.head[0], canonical(&genes[0]).unwrap());
    }

    /// A match whose every occurrence a larger fold swallowed allocates no
    /// slot. Built by hand, since the finder's containment filter removes it.
    #[test]
    fn a_swallowed_match_allocates_no_slot() {
        let genes = vec![
            r#"(Sqrt (Add (Mul (Var "x0") (Var "x1")) (Var "x2")))"#.to_string(),
            r#"(Exp (Add (Mul (Var "x0") (Var "x1")) (Var "x2")))"#.to_string(),
        ];
        let big = Match { sites: vec![(0, vec![0]), (1, vec![0])], internal_op_count: 2 };
        let small = Match { sites: vec![(0, vec![0, 0]), (1, vec![0, 0])], internal_op_count: 1 };
        let folded = fold(&genes, &[small, big], opts(4)).expect("fold");
        assert_eq!((folded.filled, folded.skipped), (1, 1));
        let want: Vec<String> = genes.iter().map(|g| canonical(g).unwrap()).collect();
        assert_eq!(unfold(&folded).unwrap(), want);
    }

    /// The motivating case: equal only after rewriting, through the
    /// saturated finder's chosen forms. Both genes land in one class, the
    /// fold writes the chosen form to slot 0, both heads read it.
    #[test]
    fn exp_of_a_sum_and_product_of_exps_fold_through_the_saturated_finder() {
        let genes = vec![r#"(Exp (Add (Var "x") (Var "y")))"#.to_string(), r#"(Mul (Exp (Var "x")) (Exp (Var "y")))"#.to_string()];
        let (matches, _, chosen) = maximal_shared_saturated(&genes, 1).expect("finder");
        assert_eq!(matches.len(), 1);
        let folded = fold(&chosen, &matches, opts(1)).expect("fold");
        assert_eq!(folded.filled, 1);
        assert_eq!(folded.head, vec![r#"(Var "href0")"#.to_string(), r#"(Var "href0")"#.to_string()]);
        let want: Vec<String> = chosen.iter().map(|g| canonical(g).unwrap()).collect();
        assert_eq!(unfold(&folded).unwrap(), want);
    }

    /// Same input, same output.
    #[test]
    fn fold_is_deterministic() {
        let small = r#"(Mul (Var "x0") (Var "x1"))"#;
        let big = format!("(Add {small} (Var \"x2\"))");
        let genes = vec![format!("(Sqrt {big})"), format!("(Exp {big})"), format!("(Add {small} (Var \"x0\"))"), format!("(Mul {small} (Var \"x2\"))")];
        let matches = maximal_shared(&genes, 1).expect("finder");
        let a = fold(&genes, &matches, opts(3)).unwrap();
        let b = fold(&genes, &matches, opts(3)).unwrap();
        assert_eq!(a, b);
    }

    /// Encoded genes decode back to the folded text with the references in
    /// place, through the Karva round trip, with the references in the pset.
    #[test]
    fn encode_round_trips_through_karva_with_references() {
        let genes = vec![
            r#"(Add (Mul (Var "x0") (Var "x1")) (Var "x2"))"#.to_string(),
            r#"(Sqrt (Mul (Var "x0") (Var "x1")))"#.to_string(),
        ];
        let matches = maximal_shared(&genes, 1).expect("finder");
        let folded = fold(&genes, &matches, FoldOptions { tail_slots: 2, max_definition_ops: Some(8) }).expect("fold");
        let encoded = encode(&folded, &pset(), 7, 8).expect("encode");
        assert_eq!(encoded.len(), 4, "two head genes and two tail genes");
        let pset = pset_with_hrefs(&pset(), 2);
        let decoded: Vec<String> = encoded.iter().map(|(h, t)| karva_to_terms(h, t, &pset).expect("decode")).collect();
        let want: Vec<String> = folded.head.iter().chain(folded.tail.iter()).map(|g| canonical(g).unwrap()).collect();
        let got: Vec<String> = decoded.iter().map(|g| canonical(g).unwrap()).collect();
        assert_eq!(got, want);
    }
}
