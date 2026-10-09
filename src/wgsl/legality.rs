//! The one place that decides whether occurrences of an expression may
//! share a definition, and where that definition goes
//! (`docs/PLAN_wgsl_lineage.md` §1, §5).
//!
//! `decide` takes a read function, its roots as trees, and the sites (root
//! index, path in the tree) of every occurrence of one text, and answers
//! `Accept { placement }` or `Refuse { reason }`. Sharing is legal when the
//! sites' texts are equal (with versions on their loads, so equal text is
//! equal lineage), the text is admissible (pure and deterministic), and a
//! point exists that dominates every site at which every versioned load
//! reads the version current there and every `let` and call result the
//! text reads is bound and in scope (a `call.<fn>@<idx>` leaf names the
//! value ONE `Call` statement bound; reading it repeats no call, so it is
//! a binding like a `let`, available after its statement). The placement
//! is the EARLIEST such point in program
//! order. Hoisting out of a loop needs no rule of its own: a loop that
//! bumps a location in the lineage gives it a header phi, so every point
//! before the loop fails the version clause, and a loop that bumps nothing
//! in the lineage passes it, which is the proof of invariance.
//!
//! A point is "before statement `path`", with `after_let` saying how many
//! of the `let` roots bound at that point the definition follows (a
//! definition reads lets bound before it and is read by lets bound after
//! it). For structured WGSL the points that dominate a set of sites are
//! totally ordered: every statement index of every block on the common
//! ancestor chain, up to the statement that contains the sites.

use std::collections::{BTreeMap, BTreeSet};

use crate::karva::MathNode;

use super::reader::{KernelFunction, RootKind};
use super::versions::split_version;

/// Where a shared definition is emitted: immediately before the statement
/// at `path`, after the first `after_let` `let` roots bound at that point.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Placement {
    pub path: Vec<usize>,
    pub after_let: usize,
}

/// Why sharing was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reason {
    /// A node of the text is not in the admissible (pure, deterministic)
    /// set: an atomic or image result, a builtin that is not an invocation
    /// id, an unrowed node of an effectful kind, or a `store.*` root itself
    /// (a hole in the scaffold, not a value: two call sites passing the
    /// literal `0u` repeat a ROOT, which the earlier finder folded as if it
    /// were a value and the device never saw).
    NotAdmissible(String),
    /// The sites' texts are not all equal.
    TextDiffers,
    /// The sites' texts are equal with the versions stripped but not with
    /// them: the same expression over different memory histories.
    LineageDiffers,
    /// No statement point dominates every site (a site in a local's
    /// initialiser, which runs before any statement; `score_main`'s
    /// `vec2(0.0, 0.0)` is one, repeated in 17 initialisers and the body).
    NoDominatingPoint,
    /// At every dominating point some `let` the text reads is not yet bound
    /// or not in lexical scope (bound in a branch arm).
    OperandUnavailable(String),
    /// At every dominating point some load's version is not the current one.
    VersionNotCurrent(String),
    /// A derivative or a subgroup operation: not movable across
    /// non-uniform control flow.
    UnsafeToMove(String),
}

impl Reason {
    /// The reason's name, for counting.
    pub fn name(&self) -> &'static str {
        match self {
            Reason::NotAdmissible(_) => "not_admissible",
            Reason::TextDiffers => "text_differs",
            Reason::LineageDiffers => "lineage_differs",
            Reason::NoDominatingPoint => "no_dominating_point",
            Reason::OperandUnavailable(_) => "operand_unavailable",
            Reason::VersionNotCurrent(_) => "version_not_current",
            Reason::UnsafeToMove(_) => "unsafe_to_move",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Accept { placement: Placement },
    Refuse { reason: Reason },
}

/// The invocation builtins: constant within one invocation's program.
pub const INVOCATION_BUILTINS: &[&str] = &["global_invocation_id", "local_invocation_id", "local_invocation_index", "workgroup_id", "num_workgroups"];

/// Unrowed naga nodes that are pure functions of their operands.
const PURE_NAGA: &[&str] = &["AccessIndex", "Access", "NestedAccess", "ArrayLength", "Compose", "As", "Math", "Unary", "Binary", "Select", "Relational", "Splat", "Swizzle", "ZeroValue", "Literal"];

/// Whether every node of `n` is admissible for sharing.
pub(crate) fn admissible(n: &MathNode) -> Result<(), Reason> {
    match n {
        MathNode::Num(_) => Ok(()),
        MathNode::Var(name) => {
            let (kind, rest) = name.split_once('.').unwrap_or((name.as_str(), ""));
            match kind {
                "load" | "let" | "call" | "arg" | "const" | "override" | "global" | "local" => Ok(()),
                "builtin" if INVOCATION_BUILTINS.contains(&rest) => Ok(()),
                "builtin" => Err(Reason::NotAdmissible(name.clone())),
                "naga" => Err(naga_reason(rest, name)),
                _ if name.starts_with("href") => Ok(()),
                _ => Err(Reason::NotAdmissible(name.clone())),
            }
        }
        MathNode::App(ctor, children) => {
            let (class, rest) = ctor.split_once('.').unwrap_or((ctor.as_str(), ""));
            match class {
                "arith" | "trig" | "geom" | "index" | "count" | "int" | "fixed" | "convert" | "quantise" | "bits" | "hash" | "compare" | "logic" | "select" | "shape" | "literal" | "load" => {}
                "naga" => {
                    let node = rest.split('.').next().unwrap_or("");
                    if !PURE_NAGA.contains(&node) {
                        return Err(naga_reason(rest, ctor));
                    }
                }
                _ => return Err(Reason::NotAdmissible(ctor.clone())),
            }
            children.iter().try_for_each(admissible)
        }
    }
}

fn naga_reason(rest: &str, name: &str) -> Reason {
    let node = rest.split('.').next().unwrap_or("");
    if node.starts_with("Derivative") || node.starts_with("Subgroup") {
        Reason::UnsafeToMove(name.to_string())
    } else {
        Reason::NotAdmissible(name.to_string())
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

fn strip_versions(n: &MathNode) -> MathNode {
    match n {
        MathNode::Num(v) => MathNode::Num(*v),
        MathNode::Var(name) => MathNode::Var(split_version(name).0.to_string()),
        MathNode::App(ctor, children) => MathNode::App(split_version(ctor).0.to_string(), children.iter().map(strip_versions).collect()),
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

/// The location a load target names: its first two steps (`buffer.q`,
/// `local.i@1`, `arg.p`), which is how the reader keys versions.
fn location_of_target(target: &str) -> String {
    target.splitn(3, '.').take(2).collect::<Vec<_>>().join(".")
}

/// Every versioned load of `n` as (location, version), every `let`
/// binding it reads, and every call result (by handle index).
fn reads(n: &MathNode, loads: &mut BTreeMap<String, u32>, lets: &mut BTreeSet<String>, calls: &mut BTreeSet<u32>) {
    let mut note = |name: &str| {
        let (base, version) = split_version(name);
        if let Some(target) = base.strip_prefix("load.") {
            if let Some(v) = version {
                loads.insert(location_of_target(target), v);
            }
        } else if let Some(l) = base.strip_prefix("let.") {
            lets.insert(l.to_string());
        } else if let Some(c) = base.strip_prefix("call.") {
            if let Some(idx) = c.rsplit_once('@').and_then(|(_, i)| i.parse().ok()) {
                calls.insert(idx);
            }
        }
    };
    match n {
        MathNode::Num(_) => {}
        MathNode::Var(name) => note(name),
        MathNode::App(ctor, children) => {
            note(ctor);
            for c in children {
                reads(c, loads, lets, calls);
            }
        }
    }
}

/// Whether a value bound before the statement at `bound` (a `let` or a
/// call's result) is in scope and already bound at `point`, excluding the
/// case of the same point (decided by the caller): bound earlier in the
/// same block, or in an ancestor block before (`strict`) or at the
/// statement that contains `point`. A `let` bound just before the
/// containing statement is visible inside it; a call AT that statement is
/// not bound yet, so a call needs `strict`.
fn bound_before(bound: &[usize], point: &[usize], strict: bool) -> bool {
    if bound.is_empty() || bound.len() > point.len() || bound.len().is_multiple_of(2) {
        return false;
    }
    let level = bound.len() - 1;
    if bound[..level] != point[..level] {
        return false;
    }
    if bound.len() == point.len() || strict {
        bound[level] < point[level]
    } else {
        bound[level] <= point[level]
    }
}

/// One site's use point: the root's evaluation point and, for a `let`
/// root, its index among the lets bound at that point.
struct Use {
    point: Vec<usize>,
    let_index: Option<usize>,
}

/// Decide whether the occurrences at `sites` of one text may share one
/// definition, and where it goes. `roots` are the function's root trees in
/// `f.roots` order (as the finder saw them).
pub(crate) fn decide(f: &KernelFunction, roots: &[MathNode], sites: &[(usize, Vec<u8>)]) -> Decision {
    let refuse = |reason: Reason| Decision::Refuse { reason };
    let nodes: Vec<&MathNode> = match sites.iter().map(|(g, p)| roots.get(*g).and_then(|r| node_at(r, p))).collect::<Option<Vec<_>>>() {
        Some(n) if !n.is_empty() => n,
        _ => return refuse(Reason::NoDominatingPoint),
    };
    let texts: BTreeSet<String> = nodes.iter().map(|n| render(n)).collect();
    if texts.len() > 1 {
        let stripped: BTreeSet<String> = nodes.iter().map(|n| render(&strip_versions(n))).collect();
        return refuse(if stripped.len() == 1 { Reason::LineageDiffers } else { Reason::TextDiffers });
    }
    let text = nodes[0];
    if let Err(r) = admissible(text) {
        return refuse(r);
    }
    let mut loads = BTreeMap::new();
    let mut lets = BTreeSet::new();
    let mut calls = BTreeSet::new();
    reads(text, &mut loads, &mut lets, &mut calls);
    // The let roots: binding name → (point, index among the lets at that point).
    let mut lets_at: BTreeMap<Vec<usize>, Vec<String>> = BTreeMap::new();
    for r in &f.roots {
        if let RootKind::Let { name } = &r.kind {
            lets_at.entry(r.point.clone()).or_default().push(name.clone());
        }
    }
    let binding_of = |name: &str| -> Option<(Vec<usize>, usize)> {
        lets_at.iter().find_map(|(point, names)| names.iter().position(|n| n == name).map(|i| (point.clone(), i)))
    };
    // The use points.
    let mut uses: Vec<Use> = Vec::new();
    for (g, _) in sites {
        let r = &f.roots[*g];
        if r.point.is_empty() {
            return refuse(Reason::NoDominatingPoint);
        }
        let let_index = match &r.kind {
            RootKind::Let { name } => lets_at.get(&r.point).and_then(|names| names.iter().position(|n| n == name)),
            _ => None,
        };
        uses.push(Use { point: r.point.clone(), let_index });
    }
    // The dominator chain: for every block that holds every use (a common
    // even-length prefix), the statement indices up to the one containing
    // the uses; outermost block first, earliest statement first.
    let mut candidates: Vec<Vec<usize>> = Vec::new();
    let shortest = uses.iter().map(|u| u.point.len()).min().unwrap_or(0);
    let mut depth = 0;
    while depth < shortest {
        // block prefix = point[..depth]; all uses must agree on it.
        let block = &uses[0].point[..depth];
        if !uses.iter().all(|u| &u.point[..depth] == block) {
            break;
        }
        let k_max = uses.iter().map(|u| u.point[depth]).min().expect("a use");
        let same_statement = uses.iter().all(|u| u.point[depth] == k_max);
        for k in 0..=k_max {
            let mut p = block.to_vec();
            p.push(k);
            candidates.push(p);
        }
        if !same_statement {
            break;
        }
        // Descend only if every use continues into the same child block of
        // statement k_max.
        if uses.iter().any(|u| u.point.len() < depth + 2) {
            break;
        }
        let child = uses[0].point[depth + 1];
        if !uses.iter().all(|u| u.point[depth + 1] == child) {
            break;
        }
        depth += 2;
    }
    let mut last_version = None;
    let mut last_operand = None;
    for point in candidates {
        // Versions current at the point.
        let Some(at) = f.points.get(&point) else { continue };
        if let Some((loc, v)) = loads.iter().find(|(loc, v)| at.get(*loc).copied().unwrap_or(0) != **v) {
            last_version = Some(format!("{loc}@v{v} at {point:?}"));
            continue;
        }
        // Every let read is bound before the point and in scope; the ones
        // bound at this very point set `after_let`.
        let mut after_let = 0usize;
        let mut unavailable = None;
        for l in &lets {
            match binding_of(l) {
                None => unavailable = Some(format!("let.{l} has no binding root")),
                Some((bp, i)) => {
                    if bp == point {
                        after_let = after_let.max(i + 1);
                    } else if !bound_before(&bp, &point, false) {
                        unavailable = Some(format!("let.{l} is bound at {bp:?}, not in scope at {point:?}"));
                    }
                }
            }
        }
        for c in &calls {
            match f.calls.get(c) {
                None => unavailable = Some(format!("call result {c} has no call statement")),
                Some(bp) if !bound_before(bp, &point, true) => unavailable = Some(format!("call result {c} is bound at {bp:?}, after {point:?}")),
                Some(_) => {}
            }
        }
        if let Some(u) = unavailable {
            last_operand = Some(u);
            continue;
        }
        // A let at this point that reads the text must come after the definition.
        if let Some(bad) = uses.iter().filter(|u| u.point == point).filter_map(|u| u.let_index).find(|&i| i < after_let) {
            last_operand = Some(format!("let {bad} at {point:?} reads the text but is bound before an operand of it"));
            continue;
        }
        return Decision::Accept { placement: Placement { path: point, after_let } };
    }
    refuse(match (last_version, last_operand) {
        (_, Some(o)) => Reason::OperandUnavailable(o),
        (Some(v), None) => Reason::VersionNotCurrent(v),
        (None, None) => Reason::NoDominatingPoint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::karva::parse_math;
    use crate::wgsl::reader::read;

    /// Decide for the text `needle` (a rendered subtree) over all its
    /// occurrences in `function`'s roots.
    fn decide_text(src: &str, function: &str, needle: &str) -> Decision {
        let k = read(src).unwrap_or_else(|e| panic!("{e}"));
        let f = k.functions.iter().find(|f| f.name == function).unwrap();
        let roots: Vec<MathNode> = f.roots.iter().map(|r| parse_math(&r.tree.to_sexpr()).unwrap()).collect();
        let want = render(&parse_math(needle).unwrap());
        let mut sites = Vec::new();
        fn walk(n: &MathNode, g: usize, path: Vec<u8>, want: &str, out: &mut Vec<(usize, Vec<u8>)>) {
            if render(n) == want {
                out.push((g, path.clone()));
            }
            if let MathNode::App(_, kids) = n {
                for (i, k) in kids.iter().enumerate() {
                    let mut p = path.clone();
                    p.push(i as u8);
                    walk(k, g, p, want, out);
                }
            }
        }
        for (g, r) in roots.iter().enumerate() {
            walk(r, g, Vec::new(), &want, &mut sites);
        }
        assert!(sites.len() >= 2, "{needle} occurs {} times: {:?}", sites.len(), f.roots.iter().map(|r| r.tree.to_sexpr()).collect::<Vec<_>>());
        decide(f, &roots, &sites)
    }

    fn accept(d: Decision) -> Placement {
        match d {
            Decision::Accept { placement } => placement,
            Decision::Refuse { reason } => panic!("refused: {reason:?}"),
        }
    }

    fn reason(d: Decision) -> Reason {
        match d {
            Decision::Refuse { reason } => reason,
            Decision::Accept { placement } => panic!("accepted at {placement:?}"),
        }
    }

    #[test]
    fn a_partial_current_at_the_function_start_is_placed_at_the_start() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var s = 1.0;
    a[0] = s * s + xs[0];
    b[0] = s * s + xs[1];
    s = 2.0;
}
"#;
        // s is bumped later, so its loads carry v0; both uses read v0 and
        // the earliest point (before statement 0) is current.
        let p = accept(decide_text(src, "main", "(arith.mul (Var \"load.local.s@0@v0\") (Var \"load.local.s@0@v0\"))"));
        assert_eq!(p, Placement { path: vec![0], after_let: 0 });
    }

    #[test]
    fn a_partial_whose_versions_differ_between_the_uses_is_refused_as_lineage() {
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
        let f = &k.functions[0];
        let roots: Vec<MathNode> = f.roots.iter().map(|r| parse_math(&r.tree.to_sexpr()).unwrap()).collect();
        // The two stores' value subtrees: index 1 of each store root.
        let sites = vec![(1usize, vec![1u8]), (3usize, vec![1u8])];
        assert_eq!(reason(decide(f, &roots, &sites)), Reason::LineageDiffers);
        // And two different texts altogether.
        let sites = vec![(1usize, vec![1u8]), (3usize, vec![0u8])];
        assert_eq!(reason(decide(f, &roots, &sites)), Reason::TextDiffers);
    }

    #[test]
    fn a_partial_after_a_store_is_placed_after_the_store_not_before_it() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var s = 1.0;
    out[0] = 0.0;
    s = 3.0;
    out[1] = s * s + 2.0;
    out[2] = s * s + 2.0;
}
"#;
        // The points before statements 0 and 1 hold s@v0; the definition
        // reads v1, current from statement 2 on.
        let p = accept(decide_text(src, "main", "(arith.mul (Var \"load.local.s@0@v1\") (Var \"load.local.s@0@v1\"))"));
        assert_eq!(p, Placement { path: vec![2], after_let: 0 });
    }

    #[test]
    fn a_partial_inside_a_loop_is_placed_inside_it_unless_the_loop_leaves_its_lineage_alone() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var s = 0.0;
    var i = 0u;
    loop {
        if (i >= 4u) { break; }
        out[0] = s * s + xs[0] * xs[0];
        out[1] = s * s + 1.0;
        s = s + 1.0;
        out[2] = xs[0] * xs[0] + 1.0;
        i = i + 1u;
    }
}
"#;
        // s * s reads the header version: current only inside the loop body
        // (the point before the loop holds v0), so the earliest legal point
        // is the body's first statement.
        let p = accept(decide_text(src, "main", "(arith.mul (Var \"load.local.s@0@v1\") (Var \"load.local.s@0@v1\"))"));
        assert_eq!(p, Placement { path: vec![0, 0, 0], after_let: 0 }, "before the first statement of the loop body");
        // xs[0] * xs[0] reads nothing the loop bumps: hoisted before the loop.
        let p = accept(decide_text(src, "main", "(arith.mul (load.buffer.xs.# (literal.index (Num 0.0))) (load.buffer.xs.# (literal.index (Num 0.0))))"));
        assert_eq!(p, Placement { path: vec![0], after_let: 0 });
    }

    #[test]
    fn a_site_in_a_locals_initialiser_has_no_dominating_point() {
        // A constant initialiser is an `Init` root evaluated before any
        // statement (naga turns a computed one into a store, which has a
        // point); its literal is a one-operator text the store repeats.
        let src = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var s = 1.0;
    out[0] = 1.0;
}
"#;
        let k = read(src).unwrap();
        let f = &k.functions[0];
        assert!(matches!(f.roots[0].kind, RootKind::Init { .. }));
        let roots: Vec<MathNode> = f.roots.iter().map(|r| parse_math(&r.tree.to_sexpr()).unwrap()).collect();
        let sites = vec![(0usize, vec![0u8]), (1usize, vec![1u8])];
        assert_eq!(reason(decide(f, &roots, &sites)), Reason::NoDominatingPoint);
    }

    #[test]
    fn a_let_bound_in_an_arm_is_not_available_after_the_join_but_is_inside_the_arm() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var r = 0.0;
    if (gid.x > 0u) {
        let v = xs[gid.x];
        out[0] = v * v + 1.0;
        r = v * v + 1.0;
    }
    out[1] = r;
}
"#;
        let k = read(src).unwrap();
        let v = k.functions[0].roots.iter().find_map(|r| match &r.kind {
            RootKind::Let { name } if name.starts_with("v@") => Some(name.clone()),
            _ => None,
        }).unwrap();
        let text = format!("(arith.mul (Var \"let.{v}\") (Var \"let.{v}\"))");
        // Both uses are in the arm, after the let: placed in the arm after it.
        let p = accept(decide_text(src, "main", &text));
        // `var r` is a constant init (no statement): the if is statement 0.
        assert_eq!(p, Placement { path: vec![0, 0, 0], after_let: 1 }, "before the arm's first statement, after its one let");
        // Hand it a use outside the arm as well: no point in scope.
        let f = &k.functions[0];
        let roots: Vec<MathNode> = f.roots.iter().map(|r| parse_math(&r.tree.to_sexpr()).unwrap()).collect();
        let in_arm = roots.iter().enumerate().find(|(_, r)| render(r).contains(&render(&parse_math(&text).unwrap()))).map(|(g, _)| g).unwrap();
        let after = roots.len() - 1;
        let sites = vec![(in_arm, vec![1u8]), (after, vec![1u8])];
        // The texts differ (r vs v*v+1), which is reported first; so compare
        // on a synthetic second site of the SAME text placed after the join.
        assert!(matches!(decide(f, &roots, &sites), Decision::Refuse { .. }));
        let mut roots2 = roots.clone();
        roots2[after] = parse_math(&format!("(store.buffer.out.# (literal.index (Num 1.0)) (arith.mul (Var \"let.{v}\") (Var \"let.{v}\")))")).unwrap();
        let sites = vec![(in_arm, vec![1u8, 0u8]), (after, vec![1u8])];
        let site_text: BTreeSet<String> = sites.iter().map(|(g, p)| render(node_at(&roots2[*g], p).unwrap())).collect();
        assert_eq!(site_text.len(), 1, "{site_text:?}");
        assert!(matches!(reason(decide(f, &roots2, &sites)), Reason::OperandUnavailable(_)));
    }

    #[test]
    fn a_definition_at_a_lets_point_follows_the_lets_it_reads() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let a = xs[gid.x];
    let b = a * a + 1.0;
    out[0] = (a * a) * b;
}
"#;
        let k = read(src).unwrap();
        let a = k.functions[0].roots.iter().find_map(|r| match &r.kind {
            RootKind::Let { name } if name.starts_with("a@") => Some(name.clone()),
            _ => None,
        }).unwrap();
        let p = accept(decide_text(src, "main", &format!("(arith.mul (Var \"let.{a}\") (Var \"let.{a}\"))")));
        assert_eq!(p, Placement { path: vec![0], after_let: 1 }, "after `let a`, before `let b`");
    }

    #[test]
    fn a_call_result_is_a_binding_available_after_its_call_and_effectful_nodes_are_refused() {
        let src = r#"
fn twice(x: f32) -> f32 { return x * 2.0; }
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    out[0] = 0.0;
    let t = twice(1.0);
    out[1] = t * t + 1.0;
    out[2] = t * t + 2.0;
}
"#;
        // `let t = twice(1.0)` is a Call statement whose result naga names
        // `t`; the reader reads its uses as `call.twice@<idx>` leaves. The
        // repeat `t * t` is placed after the call (statement 1), not at the
        // function start.
        let k = read(src).unwrap();
        let f = k.functions.iter().find(|f| f.name == "main").unwrap();
        let leaf = f.roots.iter().flat_map(|r| {
            let mut nodes = Vec::new();
            r.tree.walk(&mut nodes);
            nodes.into_iter().filter_map(|n| match n {
                super::super::reader::Node::Leaf { name, .. } if name.starts_with("call.twice@") => Some(name.clone()),
                _ => None,
            })
        }).next().expect("a call leaf");
        let p = accept(decide_text(src, "main", &format!("(arith.mul (Var \"{leaf}\") (Var \"{leaf}\"))")));
        assert_eq!(p, Placement { path: vec![2], after_let: 0 });
        assert!(matches!(admissible(&parse_math("(naga.DerivativeX (Var \"load.buffer.q.#@v1\"))").unwrap()), Err(Reason::UnsafeToMove(_))));
        assert!(matches!(admissible(&parse_math("(arith.add (Var \"builtin.subgroup_size\") (Num 1.0))").unwrap()), Err(Reason::NotAdmissible(_))));
        assert!(matches!(admissible(&parse_math("(arith.add (Var \"naga.AtomicResult\") (Num 1.0))").unwrap()), Err(Reason::NotAdmissible(_))));
        assert_eq!(admissible(&parse_math("(naga.AccessIndex.1 (naga.ArrayLength (Var \"global.xs\")))").unwrap()), Ok(()));
    }
}
