//! Form inference by use: from the slot the reader knows for every node to
//! the dual `slot.form` the kingdom's rows are typed over.
//!
//! Two passes over each root's tree, with the instantiated rows as the
//! oracle. **Produce**, bottom-up: a literal carries its form; a leaf (a
//! load, an argument, a builtin, a constant, a call result) takes its slot's
//! default form (`real`, `int`, `index`, `flag`); an application takes the
//! out form of its row at the node's slot (`bits.and` → `bits`, `compare.lt`
//! → `flag`, `bits.count_ones` → `count`). **Demand**, top-down: a parent's
//! row at its produced dual demands a form of each child; where the child
//! produced something else, a leaf or a literal is retyped to the demand
//! (the kernel read a `u32` as bits, so it is bits), and any other node is
//! a FORM CONFLICT: the kernel mixes forms the kingdom keeps apart (`(a + i)
//! & 3u` adds an index and masks it as bits) with no `convert` written, and
//! the plan's rule is that no convert is ever inserted that the kernel did
//! not have. A conflicting node is typed `opaque` and counted; its parent
//! then has no typed row and is counted too.
//!
//! The result is what a typed chromosome needs: a dual for every node, in
//! the tree's pre-order, and the set of duals every subtree TEXT took across
//! its occurrences, which is what decides whether two textual repeats may
//! share one tail gene (the form-conflict rule of the plan).

use std::collections::{BTreeMap, BTreeSet};

use crate::geneframe::Ty;

use super::loader::{WgslDual, WgslKingdom, WgslRow};
use super::reader::{KernelFunction, Node};

/// One node's typing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeType {
    /// The dual, or `None` when the slot has no dual at all in the table
    /// (a struct, an array, a pointer, a `vec3<u32>.count`).
    pub dual: Option<Ty>,
    /// The typed row the node is an instance of, by index into the
    /// kingdom's rows; `None` for a leaf, a literal, an unrowed naga node,
    /// or a node whose children's duals match no instance.
    pub row: Option<usize>,
    /// Which rule decided the form.
    pub rule: &'static str,
}

/// A root's tree typed, nodes in pre-order (the order [`Node::walk`] gives).
#[derive(Debug, Clone)]
pub struct TypedTree {
    pub nodes: Vec<NodeType>,
}

/// A function typed.
#[derive(Debug, Clone)]
pub struct FunctionTypes {
    pub roots: Vec<TypedTree>,
    /// Every subtree text → the duals it took across its occurrences. A text
    /// with more than one dual must not be shared.
    pub by_text: BTreeMap<String, BTreeSet<Option<Ty>>>,
    pub stats: TypeStats,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TypeStats {
    pub nodes: usize,
    /// Nodes with a dual that is not `opaque`.
    pub typed: usize,
    /// Nodes typed `opaque` by a form conflict.
    pub conflicts: usize,
    /// Nodes whose slot has no dual in the table.
    pub no_dual: usize,
    /// Applications with a dual but no typed row (an unrowed naga node, or
    /// children whose duals match no instance of the row).
    pub no_row: usize,
    /// Subtree texts that took more than one dual across occurrences.
    pub conflicting_texts: usize,
    /// Nodes by the rule that typed them.
    pub by_rule: BTreeMap<&'static str, usize>,
    /// Opaque or dual-less nodes by slot.
    pub untyped_by_slot: BTreeMap<String, usize>,
}

/// Type every root of `f`.
pub fn infer_function(f: &KernelFunction, kingdom: &WgslKingdom) -> FunctionTypes {
    let ix = Index::new(kingdom);
    let mut roots = Vec::with_capacity(f.roots.len());
    let mut by_text: BTreeMap<String, BTreeSet<Option<Ty>>> = BTreeMap::new();
    let mut stats = TypeStats::default();
    for r in &f.roots {
        let mut produced = Vec::new();
        produce(&r.tree, &ix, &mut produced);
        let mut out = produced.clone();
        demand(&r.tree, &ix, &produced, &mut out, None, &mut 0);
        // Record by text and the counts, in the same pre-order.
        let mut nodes = Vec::new();
        r.tree.walk(&mut nodes);
        for (n, t) in nodes.iter().zip(&out) {
            by_text.entry(n.to_sexpr()).or_default().insert(t.dual);
            stats.nodes += 1;
            *stats.by_rule.entry(t.rule).or_default() += 1;
            match t.dual {
                None => {
                    stats.no_dual += 1;
                    *stats.untyped_by_slot.entry(n.slot().to_string()).or_default() += 1;
                }
                Some(d) if ix.is_opaque(d) => {
                    stats.conflicts += 1;
                    *stats.untyped_by_slot.entry(n.slot().to_string()).or_default() += 1;
                }
                Some(_) => stats.typed += 1,
            }
            if matches!(n, Node::App { known: true, name, .. } if !name.starts_with("store.") && !name.starts_with("load.")) && t.row.is_none() && t.dual.is_some() {
                stats.no_row += 1;
            }
        }
        roots.push(TypedTree { nodes: out });
    }
    stats.conflicting_texts = by_text.values().filter(|s| s.len() > 1).count();
    FunctionTypes { roots, by_text, stats }
}

/// The kingdom's rows and duals indexed for the passes.
struct Index<'a> {
    kingdom: &'a WgslKingdom,
    by_dual: BTreeMap<&'a str, usize>,
    /// Rows by name.
    by_name: BTreeMap<&'a str, Vec<usize>>,
}

impl<'a> Index<'a> {
    fn new(kingdom: &'a WgslKingdom) -> Self {
        let by_dual = kingdom.duals.iter().enumerate().map(|(i, d)| (d.dual.as_str(), i)).collect();
        let mut by_name: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (i, r) in kingdom.rows.iter().enumerate() {
            by_name.entry(r.name.as_str()).or_default().push(i);
        }
        Index { kingdom, by_dual, by_name }
    }

    fn dual(&self, slot: &str, form: &str) -> Option<Ty> {
        self.by_dual.get(format!("{slot}.{form}").as_str()).map(|&i| Ty::Wgsl(i as u16))
    }

    fn of(&self, ty: Ty) -> Option<&WgslDual> {
        self.kingdom.dual(ty)
    }

    fn form_of(&self, ty: Option<Ty>) -> Option<&str> {
        ty.and_then(|t| self.of(t)).map(|d| d.form.as_str())
    }

    fn is_opaque(&self, ty: Ty) -> bool {
        self.form_of(Some(ty)) == Some("opaque")
    }

    /// The rows named `name` whose output sits in `slot`.
    fn rows_at(&self, name: &str, slot: &str) -> Vec<(usize, &WgslRow)> {
        self.by_name
            .get(name)
            .map(|ix| ix.iter().map(|&i| (i, &self.kingdom.rows[i])).filter(|(_, r)| self.of(r.output).is_some_and(|d| d.slot == slot)).collect())
            .unwrap_or_default()
    }

    /// The default form of a slot before inference.
    fn default_form(slot: &str) -> &'static str {
        let scalar = match slot.find('<') {
            Some(i) if slot.starts_with("vec") || slot.starts_with("mat") => &slot[i + 1..slot.len() - 1],
            _ => slot,
        };
        match scalar {
            "f32" | "f16" => "real",
            "i32" => "int",
            "u32" => "index",
            "bool" => "flag",
            _ => "opaque",
        }
    }
}

/// Pass 1: what each node produces, bottom-up, pre-order output.
fn produce(n: &Node, ix: &Index, out: &mut Vec<NodeType>) {
    let at = out.len();
    out.push(NodeType { dual: None, row: None, rule: "" });
    let t = match n {
        Node::Literal { form, slot, .. } => NodeType { dual: ix.dual(slot, form), row: None, rule: "literal" },
        Node::Leaf { name, slot } => {
            let form = if name.starts_with("builtin.") && slot.contains("u32") { "index" } else { Index::default_form(slot) };
            NodeType { dual: ix.dual(slot, form), row: None, rule: "leaf default" }
        }
        Node::App { name, slot, known, kids } => {
            for k in kids {
                produce(k, ix, out);
            }
            if name.starts_with("store.") {
                NodeType { dual: ix.dual("store", "store"), row: None, rule: "store" }
            } else if !*known || name.starts_with("load.") {
                NodeType { dual: ix.dual(slot, Index::default_form(slot)), row: None, rule: if *known { "load default" } else { "unrowed default" } }
            } else {
                // The row's out form at this slot; the first instance in
                // table order when several share the slot (index before count).
                match ix.rows_at(name, slot).first() {
                    Some((i, r)) => NodeType { dual: Some(r.output), row: Some(*i), rule: "row out form" },
                    None => NodeType { dual: ix.dual(slot, Index::default_form(slot)), row: None, rule: "no row at slot" },
                }
            }
        }
    };
    out[at] = t;
}

/// Pass 2: what each parent demands of its children, top-down. `out` starts
/// as a copy of `produced` and is refined in place; `pos` walks pre-order.
fn demand(n: &Node, ix: &Index, produced: &[NodeType], out: &mut [NodeType], want: Option<Ty>, pos: &mut usize) {
    let me = *pos;
    *pos += 1;
    // Reconcile this node with what its parent wants.
    if let Some(w) = want {
        if out[me].dual != Some(w) {
            // A leaf, a literal, a load (the kingdom's per-kernel terminals)
            // or an unrowed naga node takes the form it is read under.
            let flexible = matches!(n, Node::Leaf { .. } | Node::Literal { .. })
                || matches!(n, Node::App { name, .. } if name.starts_with("load."))
                || out[me].rule == "unrowed default";
            let same_slot = ix.of(w).map(|d| d.slot.as_str()) == Some(n.slot());
            if flexible && same_slot {
                out[me] = NodeType { dual: Some(w), row: None, rule: "retyped by demand" };
            } else if same_slot {
                out[me] = NodeType { dual: ix.dual(n.slot(), "opaque"), row: None, rule: "form conflict" };
            }
        }
    }
    if let Node::App { name, slot, kids, .. } = n {
        // Choose the row instance at this node's (now settled) dual whose
        // inputs the children can satisfy, preferring the one whose input
        // forms equal what the children produced; demand its inputs.
        let wanted: Vec<Option<Ty>> = match out[me].dual {
            Some(d) if !name.starts_with("store.") => {
                let candidates: Vec<(usize, &WgslRow)> = ix.rows_at(name, slot).into_iter().filter(|(_, r)| r.output == d && r.inputs.len() == kids.len()).collect();
                let child_produced: Vec<Option<Ty>> = child_positions(n, me).into_iter().map(|p| produced[p].dual).collect();
                let best = candidates
                    .iter()
                    .max_by_key(|(_, r)| r.inputs.iter().zip(&child_produced).filter(|(a, b)| Some(**a) == **b).count())
                    .copied();
                match best {
                    Some((i, r)) => {
                        out[me].row = Some(i);
                        r.inputs.iter().map(|t| Some(*t)).collect()
                    }
                    None => {
                        out[me].row = None;
                        vec![None; kids.len()]
                    }
                }
            }
            _ => vec![None; kids.len()],
        };
        for (k, w) in kids.iter().zip(wanted) {
            demand(k, ix, produced, out, w, pos);
        }
    }
}

/// Pre-order positions of `n`'s children, given `n` sits at `me`.
fn child_positions(n: &Node, me: usize) -> Vec<usize> {
    let mut out = Vec::new();
    if let Node::App { kids, .. } = n {
        let mut p = me + 1;
        for k in kids {
            out.push(p);
            p += k.size_preorder();
        }
    }
    out
}

impl Node {
    /// Nodes in pre-order as [`Node::walk`] counts them (a literal is one
    /// node here; its `Num` is not walked).
    pub fn size_preorder(&self) -> usize {
        match self {
            Node::Leaf { .. } | Node::Literal { .. } => 1,
            Node::App { kids, .. } => 1 + kids.iter().map(Node::size_preorder).sum::<usize>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::reader::read;

    fn types(src: &str) -> (FunctionTypes, WgslKingdom) {
        let k = read(src).unwrap();
        let kingdom = WgslKingdom::load();
        (infer_function(&k.functions[0], &kingdom), kingdom)
    }

    #[test]
    fn a_float_pipeline_is_fully_typed_real_with_rows_for_every_application() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out[gid.x] = sin(xs[gid.x]) * 2.0 + 1.0;
}
"#;
        let (t, kingdom) = types(src);
        assert_eq!(t.stats.conflicts, 0);
        assert_eq!(t.stats.no_dual, 0);
        let root = &t.roots[0];
        // (store (shape.access_0 gid) (arith.add (arith.mul (trig.sin (load xs (access gid))) 2.0) 1.0))
        let forms: Vec<&str> = root.nodes.iter().map(|n| kingdom.dual(n.dual.unwrap()).unwrap().dual.as_str()).collect();
        assert_eq!(forms[0], "store.store");
        assert!(forms.contains(&"f32.real") && forms.contains(&"u32.index") && forms.contains(&"vec3<u32>.index"), "{forms:?}");
        // Every known application other than the store has a row.
        assert_eq!(t.stats.no_row, 0, "{:?}", root.nodes);
    }

    #[test]
    fn a_u32_masked_as_bits_is_a_form_conflict_and_a_literal_is_retyped() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out[0] = (gid.x + 1u) & 3u;
}
"#;
        let (t, kingdom) = types(src);
        let root = &t.roots[0];
        let names: Vec<String> = root.nodes.iter().map(|n| n.dual.map(|d| kingdom.dual(d).unwrap().dual.clone()).unwrap_or("-".into())).collect();
        // The `3u` literal is retyped to bits by the mask's demand; the
        // `gid.x + 1u` sum produced index and is demanded as bits: a conflict.
        assert!(names.contains(&"u32.bits".to_string()), "{names:?}");
        assert!(names.contains(&"u32.opaque".to_string()), "{names:?}");
        assert_eq!(t.stats.conflicts, 1, "{names:?}");
        assert!(t.stats.by_rule.get("retyped by demand").copied().unwrap_or(0) >= 1);
    }

    #[test]
    fn a_text_read_under_two_forms_is_recorded_as_conflicting() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<u32>;
@group(0) @binding(1) var<storage, read_write> a: array<u32>;
@group(0) @binding(2) var<storage, read_write> b: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    a[gid.x] = xs[gid.x] + 1u;
    b[gid.x] = xs[gid.x] & 1u;
}
"#;
        let (t, _) = types(src);
        // `xs[gid.x]` is an index in the first store and bits in the second.
        let loads: Vec<(&String, &BTreeSet<Option<Ty>>)> = t.by_text.iter().filter(|(k, _)| k.starts_with("(load.buffer.xs")).collect();
        assert_eq!(loads.len(), 1);
        assert_eq!(loads[0].1.len(), 2, "{:?}", loads[0].1);
        let others = t.by_text.iter().filter(|(k, s)| !k.starts_with("(load.buffer.xs") && s.len() > 1).count();
        assert_eq!(t.stats.conflicting_texts, loads.len() + others);
        assert_eq!(t.stats.conflicts, 0, "both reads are leaves and retype without conflict");
    }

    #[test]
    fn a_struct_slot_has_no_dual_and_is_counted_not_invented() {
        let src = r#"
struct P { n: u32, k: f32 }
@group(0) @binding(0) var<storage, read> ps: array<P>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = ps[gid.x];
    out[0] = p.k * f32(p.n);
}
"#;
        let (t, _) = types(src);
        assert!(t.stats.no_dual >= 1, "{:?}", t.stats);
        assert!(t.stats.untyped_by_slot.contains_key("struct"), "{:?}", t.stats.untyped_by_slot);
    }
}
