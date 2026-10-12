//! TREE DIFF AND RULE MINING over the safeWGSL typed trees
//! (`docs/PLAN_constrained_mutation.md` §6, task 9).
//!
//! Three steps, each on [`Tree`]s the reader (`safe_reader::read_safe`)
//! produced:
//!
//! 1. **Matching** ([`match_trees`]): a GumTree-style matcher (Falleri et
//!    al., 2014). Top-down, the largest isomorphic sub-trees (height at
//!    least two) are mapped, unique ones first, ambiguous ones by the
//!    similarity of their parents; bottom-up, a node not yet mapped is
//!    mapped to the node of the same class and type that holds most of its
//!    mapped descendants (Dice coefficient above one half).
//! 2. **Edits** ([`diff`]): from the two roots down, equal sub-trees are
//!    skipped, children are aligned (longest common subsequence of equal
//!    children, then the mapping, then position), and every change is
//!    reported at the smallest pair of sub-trees that differ with everything
//!    around them identical: an update (the node's row changes within its
//!    class, its children aligned and diffed further), a replacement (a
//!    sub-tree for another), an insertion or a deletion.
//! 3. **Generalisation** ([`rule_of`], [`generalise`], [`anti_unify`]): an
//!    edit's site (the edit itself, or its parent when the edit is a leaf)
//!    becomes a `find` / `replace` pair; sub-trees common to both sides become
//!    shared typed holes; the magic symbol table's step then replaces
//!    literals and variables (registers) by their types. Edits with the same
//!    generalised shape are one candidate rule; anti-unification of its
//!    concrete instances keeps whatever all of them agree on (the most
//!    specific common pattern).
//!
//! Holes are [`Tree`] leaves named `?n` with the hole's type in `ty`.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

use crate::wgsl::render::Tree;

/// The class of a row: `loop` of `loop.items_unrolled`.
pub fn class_of(name: &str) -> &str {
    name.split('.').next().unwrap_or(name)
}

/// A variable leaf: a register, a private or workgroup array, an argument.
/// The magic table generalises these to their types along with literals.
pub fn is_variable(t: &Tree) -> bool {
    t.kids.is_empty() && matches!(class_of(&t.name), "reg" | "parr" | "arg")
}

pub fn is_literal(t: &Tree) -> bool {
    t.name.starts_with("literal.")
}

pub fn is_hole(t: &Tree) -> bool {
    t.name.starts_with('?')
}

/// A tree flattened in preorder: the descendants of node `i` are
/// `i + 1 .. i + size[i]`.
struct Flat<'a> {
    nodes: Vec<&'a Tree>,
    parent: Vec<Option<usize>>,
    kids: Vec<Vec<usize>>,
    size: Vec<usize>,
    height: Vec<usize>,
    hash: Vec<u64>,
}

impl<'a> Flat<'a> {
    fn new(root: &'a Tree) -> Flat<'a> {
        let mut f = Flat { nodes: Vec::new(), parent: Vec::new(), kids: Vec::new(), size: Vec::new(), height: Vec::new(), hash: Vec::new() };
        f.push(root, None);
        f
    }

    fn push(&mut self, t: &'a Tree, parent: Option<usize>) -> usize {
        let i = self.nodes.len();
        self.nodes.push(t);
        self.parent.push(parent);
        self.kids.push(Vec::new());
        self.size.push(1);
        self.height.push(1);
        self.hash.push(0);
        let mut h = DefaultHasher::new();
        t.name.hash(&mut h);
        t.ty.code().hash(&mut h);
        t.konst.to_bits().hash(&mut h);
        let mut height = 0;
        for k in &t.kids {
            let c = self.push(k, Some(i));
            self.kids[i].push(c);
            self.size[i] += self.size[c];
            height = height.max(self.height[c]);
            self.hash[c].hash(&mut h);
        }
        self.height[i] = height + 1;
        self.hash[i] = h.finish();
        i
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn descendants(&self, i: usize) -> std::ops::Range<usize> {
        i + 1..i + self.size[i]
    }

    fn is_ancestor(&self, anc: usize, d: usize) -> bool {
        d > anc && d < anc + self.size[anc]
    }

    /// Sub-trees `i` here and `j` in `other` are equal.
    fn same(&self, i: usize, other: &Flat, j: usize) -> bool {
        self.hash[i] == other.hash[j] && self.nodes[i] == other.nodes[j]
    }
}

/// A mapping between the nodes of two trees (preorder indices).
pub struct Mapping {
    pub a_to_b: Vec<Option<usize>>,
    pub b_to_a: Vec<Option<usize>>,
}

impl Mapping {
    fn add(&mut self, i: usize, j: usize) {
        self.a_to_b[i] = Some(j);
        self.b_to_a[j] = Some(i);
    }

    /// How many nodes are mapped.
    pub fn len(&self) -> usize {
        self.a_to_b.iter().filter(|m| m.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Dice similarity of `i` in `fa` and `j` in `fb`: twice the descendants of
/// `i` mapped into descendants of `j`, over both descendant counts.
fn dice(fa: &Flat, fb: &Flat, m: &Mapping, i: usize, j: usize) -> f64 {
    let (na, nb) = (fa.size[i] - 1, fb.size[j] - 1);
    if na + nb == 0 {
        return 0.0;
    }
    let common = fa.descendants(i).filter(|&d| m.a_to_b[d].is_some_and(|e| fb.is_ancestor(j, e))).count();
    2.0 * common as f64 / (na + nb) as f64
}

/// THE GUMTREE MATCHER: top-down isomorphic sub-trees of height at least
/// `min_height`, then bottom-up containers with Dice above `min_dice`.
fn gumtree(fa: &Flat, fb: &Flat, min_height: usize, min_dice: f64) -> Mapping {
    let mut m = Mapping { a_to_b: vec![None; fa.len()], b_to_a: vec![None; fb.len()] };
    // Top-down: height-indexed lists of the nodes still open.
    let mut la: Vec<usize> = vec![0];
    let mut lb: Vec<usize> = vec![0];
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    let peek = |l: &[usize], f: &Flat| l.iter().map(|&i| f.height[i]).max().unwrap_or(0);
    let map_subtree = |m: &mut Mapping, i: usize, j: usize| {
        for k in 0..fa.size[i] {
            m.add(i + k, j + k);
        }
    };
    loop {
        let (ha, hb) = (peek(&la, fa), peek(&lb, fb));
        if ha.min(hb) < min_height {
            break;
        }
        if ha != hb {
            // Open the taller side's tallest nodes.
            let (l, f, h) = if ha > hb { (&mut la, fa, ha) } else { (&mut lb, fb, hb) };
            let (tall, rest): (Vec<usize>, Vec<usize>) = l.iter().partition(|&&i| f.height[i] == h);
            *l = rest;
            for i in tall {
                l.extend(f.kids[i].iter().copied());
            }
            continue;
        }
        let h = ha;
        let (ta, ra): (Vec<usize>, Vec<usize>) = la.iter().partition(|&&i| fa.height[i] == h);
        let (tb, rb): (Vec<usize>, Vec<usize>) = lb.iter().partition(|&&j| fb.height[j] == h);
        la = ra;
        lb = rb;
        let mut by_hash_a: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        let mut by_hash_b: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for &i in &ta {
            by_hash_a.entry(fa.hash[i]).or_default().push(i);
        }
        for &j in &tb {
            by_hash_b.entry(fb.hash[j]).or_default().push(j);
        }
        let mut used_a = vec![false; ta.len()];
        let mut used_b = vec![false; tb.len()];
        for (hsh, xs) in &by_hash_a {
            let Some(ys) = by_hash_b.get(hsh) else { continue };
            let xs: Vec<usize> = xs.iter().copied().filter(|&i| ys.iter().any(|&j| fa.same(i, fb, j))).collect();
            let ys: Vec<usize> = ys.iter().copied().filter(|&j| xs.iter().any(|&i| fa.same(i, fb, j))).collect();
            if xs.len() == 1 && ys.len() == 1 {
                map_subtree(&mut m, xs[0], ys[0]);
            } else {
                for &i in &xs {
                    for &j in &ys {
                        if fa.same(i, fb, j) {
                            candidates.push((i, j));
                        }
                    }
                }
            }
            for &i in &xs {
                used_a[ta.iter().position(|&t| t == i).expect("internal")] = true;
            }
            for &j in &ys {
                used_b[tb.iter().position(|&t| t == j).expect("internal")] = true;
            }
        }
        for (k, &i) in ta.iter().enumerate() {
            if !used_a[k] {
                la.extend(fa.kids[i].iter().copied());
            }
        }
        for (k, &j) in tb.iter().enumerate() {
            if !used_b[k] {
                lb.extend(fb.kids[j].iter().copied());
            }
        }
    }
    // Ambiguous isomorphic pairs: the pair whose parents are most alike
    // first, then the pair nearest in preorder position.
    let mut scored: Vec<(f64, usize, usize, usize)> = candidates
        .iter()
        .map(|&(i, j)| {
            let d = match (fa.parent[i], fb.parent[j]) {
                (Some(p), Some(q)) => dice(fa, fb, &m, p, q),
                _ => 0.0,
            };
            (d, i.abs_diff(j), i, j)
        })
        .collect();
    scored.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)).then(x.3.cmp(&y.3)));
    for (_, _, i, j) in scored {
        if m.a_to_b[i].is_none() && m.b_to_a[j].is_none() && fa.descendants(i).all(|d| m.a_to_b[d].is_none()) && fb.descendants(j).all(|e| m.b_to_a[e].is_none()) {
            map_subtree(&mut m, i, j);
        }
    }
    // Bottom-up, in postorder (a node after its descendants).
    let mut post: Vec<usize> = (0..fa.len()).collect();
    post.sort_by_key(|&i| (i + fa.size[i], std::cmp::Reverse(i)));
    for i in post {
        if m.a_to_b[i].is_some() || fa.kids[i].is_empty() {
            continue;
        }
        let ni = fa.nodes[i];
        // Candidates: unmapped ancestors (in B) of the images of i's
        // mapped descendants, of i's class and type.
        let mut seen: Vec<usize> = Vec::new();
        for d in fa.descendants(i) {
            let Some(mut e) = m.a_to_b[d] else { continue };
            while let Some(p) = fb.parent[e] {
                e = p;
                if seen.contains(&e) {
                    break;
                }
                seen.push(e);
            }
        }
        seen.sort_unstable();
        let mut best: Option<(f64, usize)> = None;
        for j in seen {
            let nj = fb.nodes[j];
            if m.b_to_a[j].is_some() || nj.ty != ni.ty || class_of(&nj.name) != class_of(&ni.name) {
                continue;
            }
            let d = dice(fa, fb, &m, i, j);
            if best.is_none_or(|(bd, _)| d > bd) {
                best = Some((d, j));
            }
        }
        if let Some((d, j)) = best {
            if d > min_dice {
                m.add(i, j);
            }
        }
    }
    if m.a_to_b[0].is_none() && m.b_to_a[0].is_none() && fa.nodes[0].ty == fb.nodes[0].ty && class_of(&fa.nodes[0].name) == class_of(&fb.nodes[0].name) {
        m.add(0, 0);
    }
    m
}

/// The GumTree mapping between two trees (preorder indices).
pub fn match_trees(a: &Tree, b: &Tree) -> Mapping {
    let (fa, fb) = (Flat::new(a), Flat::new(b));
    gumtree(&fa, &fb, 2, 0.5)
}

/// STATEMENT LISTS FLATTENED for diffing: the grammar spells a long
/// statement list as fixed-arity blocks (`block.seq2`..`block.seq4`,
/// `block.one`) nested through `stmt.block`, so one inserted statement
/// shifts every later one across the nesting and shows as a cascade of
/// replacements. Here every block becomes one `block.list[_level]` node
/// whose children are the statements in order, a `stmt.block` holding a
/// single block spliced in. A view for diffing only: the result is not a
/// tree of the kingdom.
pub fn flatten_blocks(t: &Tree) -> Tree {
    fn splice(t: &Tree, out: &mut Vec<Tree>) {
        for k in &t.kids {
            if k.name.starts_with("stmt.block") && k.kids.len() == 1 && class_of(&k.kids[0].name) == "block" {
                splice(&k.kids[0], out);
            } else {
                out.push(flatten_blocks(k));
            }
        }
    }
    if class_of(&t.name) == "block" {
        let level = t.name.split_once('_').map_or(String::new(), |(_, l)| format!("_{l}"));
        let mut kids = Vec::new();
        splice(t, &mut kids);
        return Tree { name: format!("block.list{level}"), ty: t.ty, konst: t.konst, kids };
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(flatten_blocks).collect() }
}

/// Twice the mapped nodes over both sizes: 1 for equal trees.
pub fn similarity(a: &Tree, b: &Tree) -> f64 {
    2.0 * match_trees(a, b).len() as f64 / (a.size() + b.size()) as f64
}

/// The tree with its registers renamed by order of first appearance within
/// each register kind (`reg.i7`, `reg.i2` → `reg.i#0`, `reg.i#1`): two
/// kernels that differ only in register allocation become equal.
pub fn canonical_registers(t: &Tree) -> Tree {
    fn kind(name: &str) -> Option<(&str, &str)> {
        let rest = name.strip_prefix("reg.")?;
        let digits = rest.find(|c: char| c.is_ascii_digit())?;
        let (k, n) = rest.split_at(digits);
        n.chars().all(|c| c.is_ascii_digit()).then_some((k, n))
    }
    fn go(t: &Tree, seen: &mut HashMap<String, String>, counts: &mut HashMap<String, usize>) -> Tree {
        if t.kids.is_empty() {
            if let Some((k, _)) = kind(&t.name) {
                let name = match seen.get(&t.name) {
                    Some(n) => n.clone(),
                    None => {
                        let c = counts.entry(k.to_string()).or_insert(0);
                        let n = format!("reg.{k}#{c}");
                        *c += 1;
                        seen.insert(t.name.clone(), n.clone());
                        n
                    }
                };
                return Tree { name, ty: t.ty, konst: t.konst, kids: Vec::new() };
            }
        }
        Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| go(k, seen, counts)).collect() }
    }
    go(t, &mut HashMap::new(), &mut HashMap::new())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EditKind {
    /// The node's row changes within its class; its children are aligned
    /// and diffed further.
    Update,
    /// One sub-tree for another.
    Replace,
    Insert,
    Delete,
}

/// One edit. `path_a` is the node's path in A (child indices from the
/// root); for an insertion, the path of the parent it goes into. `slot`
/// is the child position in the parent in A (for an insertion, the
/// position it goes before). `context` names A's ancestors, root first.
#[derive(Debug, Clone)]
pub struct Edit {
    pub kind: EditKind,
    pub path_a: Vec<usize>,
    pub slot: usize,
    pub before: Option<Tree>,
    pub after: Option<Tree>,
    pub context: Vec<String>,
}

/// THE DIFF: the edits turning `a` into `b`, each at the smallest pair of
/// differing sub-trees.
pub fn diff(a: &Tree, b: &Tree) -> Vec<Edit> {
    let (fa, fb) = (Flat::new(a), Flat::new(b));
    let m = gumtree(&fa, &fb, 2, 0.5);
    let mut d = Differ { fa: &fa, fb: &fb, m: &m, path: Vec::new(), ctx: Vec::new(), out: Vec::new() };
    d.walk(0, 0);
    d.out
}

/// The walk's state: both trees, their mapping, where it stands in A and
/// the edits so far.
struct Differ<'f, 't> {
    fa: &'f Flat<'t>,
    fb: &'f Flat<'t>,
    m: &'f Mapping,
    path: Vec<usize>,
    ctx: Vec<String>,
    out: Vec<Edit>,
}

impl Differ<'_, '_> {
    /// Whether `i` and `j` may be diffed as one node (an update at most).
    fn updatable(&self, i: usize, j: usize) -> bool {
        let (x, y) = (self.fa.nodes[i], self.fb.nodes[j]);
        x.ty == y.ty && class_of(&x.name) == class_of(&y.name) && !x.kids.is_empty() && !y.kids.is_empty() && (x.name == y.name || self.m.a_to_b[i] == Some(j) || dice(self.fa, self.fb, self.m, i, j) > 0.0)
    }

    fn edit(&mut self, kind: EditKind, path_a: Vec<usize>, slot: usize, before: Option<&Tree>, after: Option<&Tree>) {
        self.out.push(Edit { kind, path_a, slot, before: before.cloned(), after: after.cloned(), context: self.ctx.clone() });
    }

    fn walk(&mut self, i: usize, j: usize) {
        let (fa, fb) = (self.fa, self.fb);
        if fa.same(i, fb, j) {
            return;
        }
        let (x, y) = (fa.nodes[i], fb.nodes[j]);
        let slot = self.path.last().copied().unwrap_or(0);
        if x.name != y.name || x.kids.is_empty() || y.kids.is_empty() {
            let kind = if self.updatable(i, j) { EditKind::Update } else { EditKind::Replace };
            self.edit(kind, self.path.clone(), slot, Some(x), Some(y));
            if kind == EditKind::Replace {
                return;
            }
        }
        let (ka, kb) = (&fa.kids[i], &fb.kids[j]);
        let pairs = self.align(ka, kb);
        self.ctx.push(x.name.clone());
        let (mut pa, mut pb) = (0usize, 0usize);
        for (u, v) in pairs.into_iter().chain(std::iter::once((ka.len(), kb.len()))) {
            // Unpaired children before this pair: deletions and insertions.
            while pa < u {
                self.edit(EditKind::Delete, [self.path.as_slice(), &[pa]].concat(), pa, Some(fa.nodes[ka[pa]]), None);
                pa += 1;
            }
            while pb < v {
                self.edit(EditKind::Insert, self.path.clone(), u, None, Some(fb.nodes[kb[pb]]));
                pb += 1;
            }
            if u < ka.len() && v < kb.len() {
                self.path.push(u);
                self.walk(ka[u], kb[v]);
                self.path.pop();
            }
            pa = u + 1;
            pb = v + 1;
        }
        self.ctx.pop();
    }

    /// The children's pairing, monotone in both lists: equal children by
    /// longest common subsequence, then within each gap the mapped or
    /// similar pairs, or, in a gap of equal length on both sides, by
    /// position.
    fn align(&self, ka: &[usize], kb: &[usize]) -> Vec<(usize, usize)> {
        let (fa, fb) = (self.fa, self.fb);
        let (n, k) = (ka.len(), kb.len());
        let mut lcs = vec![vec![0usize; k + 1]; n + 1];
        for u in (0..n).rev() {
            for v in (0..k).rev() {
                lcs[u][v] = if fa.same(ka[u], fb, kb[v]) { lcs[u + 1][v + 1] + 1 } else { lcs[u + 1][v].max(lcs[u][v + 1]) };
            }
        }
        let mut anchors = Vec::new();
        let (mut u, mut v) = (0, 0);
        while u < n && v < k {
            if fa.same(ka[u], fb, kb[v]) && lcs[u][v] == lcs[u + 1][v + 1] + 1 {
                anchors.push((u, v));
                u += 1;
                v += 1;
            } else if lcs[u + 1][v] >= lcs[u][v + 1] {
                u += 1;
            } else {
                v += 1;
            }
        }
        let mut pairs = Vec::new();
        let (mut su, mut sv) = (0, 0);
        for &(au, av) in anchors.iter().chain(std::iter::once(&(n, k))) {
            self.gap_pairs(ka, kb, su..au, sv..av, &mut pairs);
            if au < n {
                pairs.push((au, av));
            }
            su = au + 1;
            sv = av + 1;
        }
        pairs
    }

    fn gap_pairs(&self, ka: &[usize], kb: &[usize], ga: std::ops::Range<usize>, gb: std::ops::Range<usize>, pairs: &mut Vec<(usize, usize)>) {
        let (fa, fb, m) = (self.fa, self.fb, self.m);
        if ga.is_empty() || gb.is_empty() {
            return;
        }
        // Similar pairs first (mapped, or same class and type with a share
        // of mapped descendants), monotone; then each stretch between them
        // of equal length on both sides pairs by position.
        let mut similar = Vec::new();
        let mut next_v = gb.start;
        for u in ga.clone() {
            let (x, nx) = (ka[u], fa.nodes[ka[u]]);
            let found = (next_v..gb.end).find(|&v| {
                let (y, ny) = (kb[v], fb.nodes[kb[v]]);
                m.a_to_b[x] == Some(y) || (nx.ty == ny.ty && class_of(&nx.name) == class_of(&ny.name) && !nx.kids.is_empty() && dice(fa, fb, m, x, y) >= 0.3)
            });
            if let Some(v) = found {
                similar.push((u, v));
                next_v = v + 1;
            }
        }
        let (mut su, mut sv) = (ga.start, gb.start);
        for &(u, v) in similar.iter().chain(std::iter::once(&(ga.end, gb.end))) {
            if u - su == v - sv {
                pairs.extend((su..u).zip(sv..v));
            }
            if u < ga.end {
                pairs.push((u, v));
            }
            su = u + 1;
            sv = v + 1;
        }
    }
}

/// The node at `path` under `t`.
pub fn at<'t>(t: &'t Tree, path: &[usize]) -> &'t Tree {
    path.iter().fold(t, |n, &k| &n.kids[k])
}

/// THE SITE OF AN EDIT as a concrete `find` / `replace` pair: the edit
/// itself when it is an internal replacement or update; its parent with
/// only this edit applied when it is a leaf replacement, an insertion or a
/// deletion (a changed literal or register means nothing without where it
/// stands).
pub fn site(a: &Tree, e: &Edit) -> Option<(Tree, Tree)> {
    let before = e.before.as_ref();
    let after = e.after.as_ref();
    match e.kind {
        EditKind::Update => Some((before?.clone(), after?.clone())),
        EditKind::Replace => {
            let (x, y) = (before?, after?);
            if !(x.kids.is_empty() && y.kids.is_empty()) || e.path_a.is_empty() {
                return Some((x.clone(), y.clone()));
            }
            let parent_path = &e.path_a[..e.path_a.len() - 1];
            let p = at(a, parent_path);
            let mut q = p.clone();
            q.kids[e.slot] = y.clone();
            Some((p.clone(), q))
        }
        EditKind::Insert => {
            let p = at(a, &e.path_a);
            let mut q = p.clone();
            q.kids.insert(e.slot.min(q.kids.len()), after?.clone());
            Some((p.clone(), q))
        }
        EditKind::Delete => {
            let parent_path = e.path_a.get(..e.path_a.len().checked_sub(1)?)?;
            let p = at(a, parent_path);
            let mut q = p.clone();
            if e.slot < q.kids.len() {
                q.kids.remove(e.slot);
            }
            Some((p.clone(), q))
        }
    }
}

fn hole(id: usize, t: &Tree) -> Tree {
    Tree::new(&format!("?{id}"), t.ty, Vec::new())
}

/// CONTEXT HOLES: every maximal sub-tree (two nodes or more, below the
/// root) of `find` that also occurs in `replace` becomes a shared typed hole
/// on both sides; what remains concrete is what the edit changed.
pub fn rule_of(find: &Tree, replace: &Tree) -> (Tree, Tree) {
    fn collect(t: &Tree, root: bool, out: &mut HashMap<String, usize>) {
        if !root && !t.kids.is_empty() {
            *out.entry(t.to_sexpr()).or_default() += 1;
        }
        for k in &t.kids {
            collect(k, false, out);
        }
    }
    let (mut in_f, mut in_r) = (HashMap::new(), HashMap::new());
    collect(find, true, &mut in_f);
    collect(replace, true, &mut in_r);
    let mut names: HashMap<String, usize> = HashMap::new();
    fn cut(t: &Tree, root: bool, in_f: &HashMap<String, usize>, in_r: &HashMap<String, usize>, names: &mut HashMap<String, usize>) -> Tree {
        if !root && !t.kids.is_empty() {
            let s = t.to_sexpr();
            if in_f.contains_key(&s) && in_r.contains_key(&s) {
                let n = names.len();
                let id = *names.entry(s).or_insert(n);
                return hole(id, t);
            }
        }
        Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| cut(k, false, in_f, in_r, names)).collect() }
    }
    let f = cut(find, true, &in_f, &in_r, &mut names);
    let r = cut(replace, true, &in_f, &in_r, &mut names);
    (f, r)
}

/// THE MAGIC TABLE'S GENERALISATION of a rule: literals and variables
/// replaced by typed holes, one hole per distinct value (shared between
/// `find` and `replace`), existing holes renumbered in order of
/// appearance. The result is the rule's shape, the key that groups edits.
pub fn generalise(find: &Tree, replace: &Tree) -> (Tree, Tree) {
    let mut names: HashMap<String, usize> = HashMap::new();
    fn go(t: &Tree, names: &mut HashMap<String, usize>) -> Tree {
        if is_literal(t) || is_variable(t) || is_hole(t) {
            let key = if is_hole(t) { format!("{}#hole", t.name) } else { t.to_sexpr() };
            let n = names.len();
            let id = *names.entry(key).or_insert(n);
            return hole(id, t);
        }
        Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| go(k, names)).collect() }
    }
    let f = go(find, &mut names);
    let r = go(replace, &mut names);
    (f, r)
}

/// ANTI-UNIFICATION (Plotkin): the most specific pattern of which every
/// one of `instances` (each a `find` / `replace` pair of the same shape) is
/// an instance; positions where the instances disagree become typed holes,
/// the same tuple of disagreeing sub-trees the same hole on both sides.
pub fn anti_unify(instances: &[(Tree, Tree)]) -> Option<(Tree, Tree)> {
    let first = instances.first()?;
    let mut store: BTreeMap<Vec<String>, usize> = BTreeMap::new();
    fn go(ts: &[&Tree], store: &mut BTreeMap<Vec<String>, usize>) -> Tree {
        let t0 = ts[0];
        let agree = ts.iter().all(|t| t.name == t0.name && t.ty == t0.ty && t.kids.len() == t0.kids.len() && t.konst.to_bits() == t0.konst.to_bits());
        if agree && !is_hole(t0) {
            let kids = (0..t0.kids.len()).map(|k| go(&ts.iter().map(|t| &t.kids[k]).collect::<Vec<_>>(), store)).collect();
            return Tree { name: t0.name.clone(), ty: t0.ty, konst: t0.konst, kids };
        }
        if ts.iter().all(|t| t.to_sexpr() == t0.to_sexpr()) {
            return t0.clone();
        }
        let key: Vec<String> = ts.iter().map(|t| t.to_sexpr()).collect();
        let n = store.len();
        let id = *store.entry(key).or_insert(n);
        hole(id, t0)
    }
    if instances.iter().any(|(f, r)| f.ty != first.0.ty || r.ty != first.1.ty) {
        return None;
    }
    let f = go(&instances.iter().map(|(f, _)| f).collect::<Vec<_>>(), &mut store);
    let r = go(&instances.iter().map(|(_, r)| r).collect::<Vec<_>>(), &mut store);
    Some((f, r))
}

/// A rule's text, `find  ==>  replace`, with holes as `?n:type` through
/// `type_name`.
pub fn rule_text(find: &Tree, replace: &Tree, type_name: &dyn Fn(&Tree) -> String) -> String {
    fn show(t: &Tree, type_name: &dyn Fn(&Tree) -> String) -> String {
        if is_hole(t) {
            return format!("{}:{}", t.name, type_name(t));
        }
        if t.kids.is_empty() {
            if is_literal(t) {
                return format!("{}", t.konst);
            }
            return t.name.clone();
        }
        format!("({} {})", t.name, t.kids.iter().map(|k| show(k, type_name)).collect::<Vec<_>>().join(" "))
    }
    format!("{}  ==>  {}", show(find, type_name), show(replace, type_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geneframe::Ty;

    fn lit(v: f32) -> Tree {
        Tree::literal("literal.index", Ty::Wgsl(1), v)
    }
    fn reg(n: &str) -> Tree {
        Tree::new(&format!("reg.{n}"), Ty::Wgsl(2), vec![])
    }
    fn add(a: Tree, b: Tree) -> Tree {
        Tree::new("index.add", Ty::Wgsl(1), vec![a, b])
    }
    fn mul(a: Tree, b: Tree) -> Tree {
        Tree::new("index.mul", Ty::Wgsl(1), vec![a, b])
    }
    fn stmt(n: &str, kids: Vec<Tree>) -> Tree {
        Tree::new(&format!("stmt.{n}"), Ty::Wgsl(3), kids)
    }
    fn block(kids: Vec<Tree>) -> Tree {
        Tree::new(&format!("block.seq{}", kids.len()), Ty::Wgsl(4), kids)
    }

    #[test]
    fn equal_trees_have_no_edits_and_map_every_node() {
        let t = block(vec![stmt("a", vec![add(reg("i0"), lit(4.0))]), stmt("b", vec![mul(reg("i1"), lit(2.0))])]);
        assert!(diff(&t, &t).is_empty());
        assert_eq!(match_trees(&t, &t).len(), t.size());
    }

    #[test]
    fn a_changed_literal_is_one_leaf_replacement_with_its_parent_as_site() {
        let a = block(vec![stmt("a", vec![add(reg("i0"), lit(4.0))]), stmt("b", vec![mul(reg("i1"), lit(2.0))])]);
        let b = block(vec![stmt("a", vec![add(reg("i0"), lit(8.0))]), stmt("b", vec![mul(reg("i1"), lit(2.0))])]);
        let e = diff(&a, &b);
        assert_eq!(e.len(), 1, "{e:?}");
        assert_eq!(e[0].kind, EditKind::Replace);
        assert_eq!(e[0].path_a, vec![0, 0, 1]);
        assert_eq!(e[0].context, vec!["block.seq2", "stmt.a", "index.add"]);
        let (f, r) = site(&a, &e[0]).unwrap();
        assert_eq!(f.to_sexpr(), "(index.add reg.i0 (literal.index 4))");
        assert_eq!(r.to_sexpr(), "(index.add reg.i0 (literal.index 8))");
    }

    #[test]
    fn an_inserted_statement_is_an_update_of_the_block_and_one_insertion() {
        let s1 = stmt("a", vec![add(reg("i0"), lit(4.0))]);
        let s2 = stmt("b", vec![mul(reg("i1"), lit(2.0))]);
        let s3 = stmt("c", vec![mul(reg("i2"), lit(3.0))]);
        let a = block(vec![s1.clone(), s2.clone()]);
        let b = block(vec![s1, s3.clone(), s2]);
        let e = diff(&a, &b);
        let kinds: Vec<EditKind> = e.iter().map(|x| x.kind).collect();
        assert_eq!(kinds, vec![EditKind::Update, EditKind::Insert], "{e:?}");
        assert_eq!(e[1].slot, 1);
        assert_eq!(e[1].after.as_ref().unwrap(), &s3);
        let (f, r) = site(&a, &e[1]).unwrap();
        assert_eq!(f.kids.len() + 1, r.kids.len());
    }

    #[test]
    fn a_moved_subtree_is_matched_top_down() {
        let big = mul(add(reg("i0"), lit(4.0)), add(reg("i1"), lit(8.0)));
        let a = block(vec![stmt("a", vec![big.clone()]), stmt("b", vec![reg("i3")])]);
        let b = block(vec![stmt("b", vec![reg("i3")]), stmt("a", vec![big.clone()])]);
        let m = match_trees(&a, &b);
        // big sits at preorder 2 in a, at 4 in b (block, stmt.b, reg.i3, stmt.a, big).
        assert_eq!(m.a_to_b[2], Some(4));
        assert_eq!(m.a_to_b[0], Some(0));
    }

    /// A statement moved past an inserted one is not paired by position
    /// with a statement of another kind.
    #[test]
    fn a_shifted_statement_pairs_with_its_like_not_its_position() {
        let lp = |v: f32| Tree::new("loop.tile", Ty::Wgsl(3), vec![lit(v), block(vec![stmt("a", vec![mul(add(reg("i0"), lit(1.0)), reg("i1"))])])]);
        let init = |r: &str| stmt("set", vec![reg(r), lit(0.0)]);
        let a = block(vec![init("i0"), lp(8.0)]);
        let b = block(vec![init("i0"), init("i2"), lp(4.0)]);
        let e = diff(&a, &b);
        // the loop pairs with the loop (its trip count 8 -> 4); the second
        // init is an insertion.
        assert!(e.iter().any(|x| x.kind == EditKind::Insert && x.after.as_ref().is_some_and(|t| t.name == "stmt.set")), "{e:?}");
        assert!(e.iter().any(|x| x.kind == EditKind::Replace && x.before.as_ref().is_some_and(|t| t.konst == 8.0) && x.after.as_ref().is_some_and(|t| t.konst == 4.0)), "{e:?}");
        assert!(!e.iter().any(|x| x.kind == EditKind::Replace && x.before.as_ref().is_some_and(|t| t.name == "loop.tile")), "{e:?}");
    }

    #[test]
    fn registers_canonicalise_by_first_appearance_and_similarity_is_one_for_equal_trees() {
        let a = block(vec![stmt("a", vec![add(reg("i7"), reg("i2"))]), stmt("b", vec![reg("i7")])]);
        let b = block(vec![stmt("a", vec![add(reg("i0"), reg("i1"))]), stmt("b", vec![reg("i0")])]);
        let c = block(vec![stmt("a", vec![add(reg("i0"), reg("i1"))]), stmt("b", vec![reg("i1")])]);
        assert_eq!(canonical_registers(&a), canonical_registers(&b));
        assert_ne!(canonical_registers(&a), canonical_registers(&c));
        assert_eq!(canonical_registers(&a).kids[0].kids[0].kids[0].name, "reg.i#0");
        assert!((similarity(&a, &a) - 1.0).abs() < 1e-12);
        assert!(similarity(&a, &c) < 1.0);
    }

    /// Nested fixed-arity blocks flatten to one list, so a statement
    /// inserted at the front is one insertion, not a cascade.
    #[test]
    fn nested_blocks_flatten_and_an_insertion_is_one_edit() {
        let nest = |kids: Vec<Tree>| Tree::new("stmt.block_L0", Ty::Wgsl(3), vec![Tree::new(&format!("block.seq{}_L0", kids.len()), Ty::Wgsl(4), kids)]);
        let s = |n: &str| stmt(n, vec![reg("i0")]);
        let a = Tree::new("block.seq2_L0", Ty::Wgsl(4), vec![s("a"), nest(vec![s("b"), s("c"), s("d")])]);
        let b = Tree::new("block.seq2_L0", Ty::Wgsl(4), vec![s("x"), nest(vec![s("a"), s("b"), Tree::new("block.seq2_L0", Ty::Wgsl(4), vec![s("c"), s("d")])])]);
        let (fa, fb) = (flatten_blocks(&a), flatten_blocks(&b));
        assert_eq!(fa.name, "block.list_L0");
        assert_eq!(fa.kids.iter().map(|k| k.name.as_str()).collect::<Vec<_>>(), vec!["stmt.a", "stmt.b", "stmt.c", "stmt.d"]);
        // b's inner seq2 is a block directly inside a block: kept as a
        // nested list (only stmt.block wrappers splice).
        assert_eq!(fb.kids.len(), 4);
        let e = diff(&fa, &flatten_blocks(&Tree::new("block.seq2_L0", Ty::Wgsl(4), vec![s("x"), nest(vec![s("a"), s("b"), s("c"), s("d")])])));
        assert_eq!(e.len(), 1, "{e:?}");
        assert_eq!(e[0].kind, EditKind::Insert);
        assert_eq!(e[0].slot, 0);
    }

    #[test]
    fn a_different_class_is_a_replacement_not_an_update() {
        let a = stmt("a", vec![add(reg("i0"), lit(4.0))]);
        let b = stmt("a", vec![Tree::new("bits.and", Ty::Wgsl(1), vec![reg("i0"), lit(15.0)])]);
        let e = diff(&a, &b);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].kind, EditKind::Replace);
        assert_eq!(e[0].before.as_ref().unwrap().name, "index.add");
    }

    #[test]
    fn rule_holes_the_common_context_and_generalise_types_the_rest() {
        let ctx = mul(reg("i1"), lit(2.0));
        let f = Tree::new("loop.items", Ty::Wgsl(5), vec![ctx.clone(), lit(4.0)]);
        let r = Tree::new("loop.items_unrolled", Ty::Wgsl(5), vec![ctx, lit(4.0)]);
        let (cf, cr) = rule_of(&f, &r);
        assert_eq!(cf.kids[0].name, "?0");
        assert_eq!(cr.kids[0].name, "?0");
        let (gf, gr) = generalise(&cf, &cr);
        assert_eq!(gf.to_sexpr(), "(loop.items ?0 ?1)");
        assert_eq!(gr.to_sexpr(), "(loop.items_unrolled ?0 ?1)");
    }

    #[test]
    fn anti_unification_keeps_what_instances_agree_on() {
        let i1 = (add(reg("i0"), lit(4.0)), add(reg("i0"), lit(8.0)));
        let i2 = (add(reg("i5"), lit(4.0)), add(reg("i5"), lit(8.0)));
        let (f, r) = anti_unify(&[i1, i2]).unwrap();
        // the register differs (one hole, shared both sides); 4 -> 8 is kept.
        assert_eq!(f.to_sexpr(), "(index.add ?0 (literal.index 4))");
        assert_eq!(r.to_sexpr(), "(index.add ?0 (literal.index 8))");
        let shown = rule_text(&f, &r, &|t| format!("{:?}", t.ty));
        assert_eq!(shown, "(index.add ?0:Wgsl(2) 4)  ==>  (index.add ?0:Wgsl(2) 8)");
    }
}
