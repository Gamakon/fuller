//! THE SHARE PASS ON THE ROWS (ST7's hash-consing, rows-aware): a value subtree a kernel repeats is computed once by a
//! `let` row of its data type (`let_u`, `let_f`, `let_v`: the rows that introduce a `var.T`) at the nearest statement
//! above every repeat, and each repeat becomes a `var.T` word naming that binding. Nothing here matches a row by name
//! beyond the one convention that a let row takes `(T, stmt) -> stmt` and binds one variable of `T`; the words are
//! written for the environment's order (`typed_environment::in_scope`: the innermost first), so every other `var.T`
//! inside the rewritten statement is re-targeted to keep naming the variable it named.
//!
//! What is shared: two occurrences are one value when their [`SubtreeKey`]s agree (the resolved tree and the same
//! binders free), the subtree is pure (no buffer or workgroup load, no register read: a hoisted load could read a value
//! a store between the two sites changes), has at least two nodes, its free variables are in scope at the let site, and
//! no `var.T` of the shared type inside the statement resolved to its constant (such a word would start naming the new
//! variable). The pass is a rendering of the tree for the scorer, as ST7's was: the gene is never written back.

use std::collections::BTreeMap;

use super::WgslRows;
use crate::geneframe::Ty;
use crate::kingdom_contract::SymbolId;
use crate::typed_environment::{Choice, Decoding, Node};
use crate::typed_gep::Tree;

/// A shared tree and how many `let`s the pass bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shared {
    pub tree: Tree,
    pub lets: usize,
}

/// The most lets one pass binds (each is one decode of the tree).
pub const MAX_LETS: usize = 64;

fn bug(what: String) -> String {
    format!("our software has a bug: {}: share: {what}", super::KINGDOM)
}

/// The `let` row of each data type: `(T, stmt) -> stmt` binding one `var.T` into its body.
fn let_rows(k: &WgslRows) -> BTreeMap<Ty, SymbolId> {
    let stmt = k.ty("stmt");
    (0..k.contract.n_symbols() as SymbolId)
        .filter(|s| k.contract.output(*s) == stmt.unwrap_or(k.root()) && k.name(*s).starts_with("let_"))
        .filter_map(|s| match k.contract.signature(s) {
            [t, body] if Some(*body) == stmt => Some((*t, s)),
            _ => None,
        })
        .collect()
}

/// Whether a subtree is pure: no row that reads memory or a register.
fn pure(k: &WgslRows, n: &Node) -> bool {
    let name = k.name(n.symbol);
    !(name == "load" || name == "wg_load" || name.starts_with("rf")) && n.kids.iter().all(|c| pure(k, c))
}

fn size(n: &Node) -> usize {
    1 + n.kids.iter().map(size).sum::<usize>()
}

fn at<'a>(t: &'a Tree, path: &[u32]) -> &'a Tree {
    path.iter().fold(t, |n, k| &n.kids[*k as usize])
}

fn at_mut<'a>(t: &'a mut Tree, path: &[u32]) -> &'a mut Tree {
    path.iter().fold(t, |n, k| &mut n.kids[*k as usize])
}

/// One sharing step: the largest repeated pure value subtree hoisted into a `let` at the nearest statement above its
/// occurrences. `None` when nothing repeats.
fn step(k: &WgslRows, tree: &Tree, d: &Decoding, lets: &BTreeMap<Ty, SymbolId>) -> Result<Option<Tree>, String> {
    let stmt = k.ty("stmt").ok_or_else(|| bug("no stmt type".into()))?;
    // the candidates: pure value subtrees of a let-able type, two nodes or more, by key
    let mut groups: BTreeMap<String, Vec<Vec<u32>>> = BTreeMap::new();
    let mut sizes: BTreeMap<String, usize> = BTreeMap::new();
    for (path, n) in d.rendered_nodes() {
        if !lets.contains_key(&n.data_type) || n.substituted.is_some() || size(n) < 2 || !pure(k, n) {
            continue;
        }
        let key = d.subtree_key(&path).ok_or_else(|| bug("a rendered node without a key".into()))?;
        let text = format!("{:?}|{:?}", key.tree, key.free.iter().map(|f| f.binder).collect::<Vec<_>>());
        sizes.insert(text.clone(), size(n));
        groups.entry(text).or_default().push(path);
    }
    let mut candidates: Vec<(usize, String)> = groups.iter().filter(|(_, v)| v.len() >= 2).map(|(key, _)| (sizes[key], key.clone())).collect();
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    for (_, key) in candidates {
        // non-nested occurrences, in pre-order
        let mut occ: Vec<Vec<u32>> = Vec::new();
        for p in &groups[&key] {
            if !occ.iter().any(|q| p.starts_with(q)) {
                occ.push(p.clone());
            }
        }
        if occ.len() < 2 {
            continue;
        }
        // the let site: the nearest statement at or above the occurrences' common ancestor
        let mut site: Vec<u32> = occ[0].clone();
        for p in &occ[1..] {
            let common = site.iter().zip(p).take_while(|(a, b)| a == b).count();
            site.truncate(common);
        }
        while d.node(&site).is_none_or(|n| n.data_type != stmt) {
            if site.pop().is_none() {
                break;
            }
        }
        let Some(site_node) = d.node(&site).filter(|n| n.data_type == stmt) else { continue };
        let first = d.node(&occ[0]).ok_or_else(|| bug("an occurrence without a node".into()))?;
        let ty = first.data_type;
        let let_symbol = lets[&ty];
        let site_env = &site_node.environment;
        // every free variable of the value is in scope at the site
        let free_ok = d.subtree_key(&occ[0]).is_some_and(|key| key.free.iter().all(|f| site_env.variables.iter().any(|v| v.binder == f.binder)));
        if !free_ok {
            continue;
        }
        // no var of this type inside the statement resolved to its constant (it would start naming the new variable)
        fn any_constant_var(k: &WgslRows, n: &Node, ty: Ty) -> bool {
            (n.data_type == ty && k.contract.is_variable(n.symbol) && matches!(n.choice, Some(Choice::Constant))) || n.kids.iter().any(|c| any_constant_var(k, c, ty))
        }
        if any_constant_var(k, site_node, ty) {
            continue;
        }
        // the hoisted value: its var words written for the site's environment
        fn retarget(k: &WgslRows, t: &Tree, n: &Node, word: &dyn Fn(&Node) -> Result<u32, String>) -> Result<Tree, String> {
            let const_index = if k.contract.is_variable(n.symbol) { Some(word(n)?) } else { t.const_index };
            let kids = t.kids.iter().zip(&n.kids).map(|(c, cn)| retarget(k, c, cn, word)).collect::<Result<_, _>>()?;
            Ok(Tree { symbol: t.symbol, const_index, kids })
        }
        let value = retarget(k, at(tree, &occ[0]), first, &|n: &Node| match n.choice {
            Some(Choice::Variable(b)) => site_env.index_of(n.data_type, b).ok_or_else(|| bug("a free variable not in scope at the let site".into())),
            _ => Err(bug("a variable of the value resolved to no binder".into())),
        })?;
        // the statement: each occurrence a var of the new binding; every other var of the type re-targeted past it
        let depth = site_env.depth;
        let body_tree = at(tree, &site).clone();
        let deeper = |n: &Node| n.environment.variables.iter().filter(|v| v.data_type == ty && v.binder.depth > depth).count() as u32;
        let body = retarget(k, &body_tree, site_node, &|n: &Node| {
            if n.data_type != ty {
                return n.const_index_word();
            }
            match n.choice {
                Some(Choice::Variable(b)) => {
                    let i = n.environment.index_of(ty, b).ok_or_else(|| bug("a bound variable not in its own scope".into()))?;
                    Ok(if i < deeper(n) { i } else { i + 1 })
                }
                _ => Err(bug("a variable of the shared type resolved to no binder".into())),
            }
        })?;
        let mut body = body;
        for p in &occ {
            let rel = &p[site.len()..];
            let use_node = d.node(p).ok_or_else(|| bug("an occurrence without a node".into()))?;
            let var_symbol = k.contract.terminals_of(ty).iter().copied().find(|s| k.contract.is_variable(*s)).ok_or_else(|| bug(format!("no var terminal of {ty:?}")))?;
            *at_mut(&mut body, rel) = Tree { symbol: var_symbol, const_index: Some(deeper(use_node)), kids: Vec::new() };
        }
        let mut out = tree.clone();
        *at_mut(&mut out, &site) = Tree { symbol: let_symbol, const_index: None, kids: vec![value, body] };
        return Ok(Some(out));
    }
    Ok(None)
}

trait Word {
    fn const_index_word(&self) -> Result<u32, String>;
}

impl Word for Node {
    /// A variable node's own word as it stands (the word the decoder read it by, recovered from its binder).
    fn const_index_word(&self) -> Result<u32, String> {
        match self.choice {
            Some(Choice::Variable(b)) => self.environment.index_of(self.data_type, b).ok_or_else(|| bug("a bound variable not in its own scope".into())),
            Some(Choice::Constant) => Ok(0),
            _ => Err(bug("a word asked of a node that is not a variable".into())),
        }
    }
}

impl WgslRows {
    /// THE SHARE PASS: repeated pure values bound once by `let` rows, until nothing repeats or [`MAX_LETS`] are bound.
    pub fn share(&self, tree: &Tree) -> Result<Shared, String> {
        let lets = let_rows(self);
        let mut cur = tree.clone();
        let mut n = 0;
        while n < MAX_LETS {
            let d = self.decode(&cur)?;
            match step(self, &cur, &d, &lets)? {
                Some(next) => {
                    cur = next;
                    n += 1;
                }
                None => break,
            }
        }
        Ok(Shared { tree: cur, lets: n })
    }
}
