//! THE PIXEL FAMILY AS ROWS of the gene frame's fixed schema (plan `docs/PLAN_futhark_kingdom.md` 7.7, task K4;
//! design `docs/DESIGN_geneframe.md` 3.3 and 8.5). One variant of the kingdom `futhark` per instance
//! (`futhark.pixel.volumepassage`, `futhark.pixel.cuneus`, `futhark.pixel.image`) in the one set of tables
//! `kingdoms/master/`, written by `kingdoms/futhark/pixel_rows.py` through `gen_tables.py` (never by hand; its
//! `--check` is a test).
//!
//! The types are shapes only: `image` (the root), `blk` (lets and loops ending in the colour), `body1`, `body3`,
//! `body5` (ending in the loop state's tuple), `f32`, `count`. Where the old tables carried a scope and a loop depth in
//! every type (435 of them), the rows carry them as context: `let`, `block` and `loop` bind f32 variables by
//! `var.f32` Flow rows, the one variable terminal `var@f32` names a variable in scope (the interface's inputs, the
//! pixel's position `px`, `py`, every binding above), and a loop is legal only below the deepest nest
//! (`guard loop: env.trips < 2 must -> {const}`).
//!
//! [`PixelRows::load`] reads the variant's rows, checks them, builds the contract and the decoder once and checks the
//! entry; [`PixelRows::program`] is the decoder's rendering inside the entry point that binds the interface from the
//! input vector, the same program text contract as [`super::program`].

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::geneframe::{Entry, KingdomTables, Role, Ty};
use crate::kingdom_contract::{relations_from_tables, KingdomContract, SymbolId};
use crate::typed_environment::{Decoder, Decoding};
use crate::typed_gep::{express, Domain, GeneLayout, Tree};

use super::{Input, HEADER};

/// The parent kingdom of the pixel family's variants.
pub const KINGDOM: &str = "futhark";

/// The variant of one instance: `futhark.pixel.<instance>`.
pub fn variant(instance: &str) -> String {
    format!("{KINGDOM}.pixel.{instance}")
}

fn bug(what: String) -> String {
    format!("our software has a bug: the pixel family: {what}")
}

/// One instance's kingdom: its rows, entry, contract and decoder, built once.
#[derive(Debug, Clone)]
pub struct PixelRows {
    pub variant: String,
    pub tables: Arc<KingdomTables>,
    pub entry: Entry,
    pub contract: Arc<KingdomContract>,
    pub decoder: Decoder,
}

impl PixelRows {
    /// Load the committed rows of an instance (`volumepassage`, `cuneus`, `image`) from the one set of tables.
    pub fn load(instance: &str) -> Result<PixelRows, String> {
        let v = variant(instance);
        PixelRows::from_text(&v, &crate::geneframe::master::select(&v)?.named_text()?)
    }

    /// Load a table's text (the named text form) as `variant`: parse, check, build the contract and the decoder, check
    /// the entry.
    pub fn from_text(variant: &str, text: &str) -> Result<PixelRows, String> {
        let ty = |i: usize| -> Result<Ty, String> {
            Ty::check_dual_count(variant, i + 1, Ty::MAX_FUTHARK_DUALS)?;
            Ok(Ty::Futhark(i as u16))
        };
        let (tables, entry) = KingdomTables::parse_named(variant, text, &ty)?;
        tables.check().map_err(|d| d.join("\n"))?;
        let contract = relations_from_tables(&tables, vec![entry.output]).check().map_err(|d| d.join("\n"))?;
        let (tables, contract) = (Arc::new(tables), Arc::new(contract));
        let decoder = Decoder::from_arcs(Arc::clone(&tables), Arc::clone(&contract)).map_err(|d| d.join("\n"))?;
        decoder.check_entry(&entry).map_err(|d| d.join("\n"))?;
        Ok(PixelRows { variant: variant.to_string(), tables, entry, contract, decoder })
    }

    /// The root data type (`image`).
    pub fn root(&self) -> Ty {
        self.entry.output
    }

    /// A data type by name.
    pub fn ty(&self, name: &str) -> Option<Ty> {
        self.tables.data_types.iter().find(|d| d.name == name).map(|d| d.id)
    }

    /// A symbol's id by name.
    pub fn symbol_named(&self, name: &str) -> Option<SymbolId> {
        self.decoder.symbol_named(name)
    }

    /// A symbol's name.
    pub fn name(&self, s: SymbolId) -> &str {
        self.contract.name(s)
    }

    /// The literals of a constant symbol's pool (its `pool` spelling).
    pub fn literals(&self, s: SymbolId) -> Vec<String> {
        self.tables.symbol(i64::from(s)).and_then(|r| r.spelling("pool")).map(|p| p.split(',').map(str::to_string).collect()).unwrap_or_default()
    }

    /// The Dc lengths of the gene: each constant symbol's data type and its pool's length.
    pub fn const_len(&self) -> BTreeMap<Ty, u32> {
        self.tables.symbols.iter().filter(|s| s.role == Role::Constant).filter_map(|s| self.tables.output(s.symbol).map(|ty| (ty, self.literals(s.symbol as SymbolId).len() as u32))).collect()
    }

    /// The gene's layout at program size `n`.
    pub fn layout(&self, n: usize) -> Result<GeneLayout, String> {
        let heads = self.contract.heads_from_program_size(self.root(), n)?;
        GeneLayout::from_heads(&self.contract, self.root(), &heads).map_err(|e| bug(format!("the layout at program size {n}: {e:?}")))
    }

    /// Express `gene` over `layout` (Karva's walk) to its tree.
    pub fn express(&self, layout: &GeneLayout, gene: &[u32]) -> Result<Tree, String> {
        Ok(express(&self.contract, layout, gene, &self.const_len()).map_err(|d| bug(format!("the gene does not express: {d}")))?.tree())
    }

    /// Passes 2 to 4 on an expressed tree: every node's environment and, where a `must` guard fails, its repair.
    pub fn decode(&self, tree: &Tree) -> Result<Decoding, String> {
        self.decoder.decode(&self.entry, tree)
    }

    /// The picture a tree spells (pass 5, the decoder's rendering), without the entry point.
    pub fn render(&self, d: &Decoding) -> Result<String, String> {
        let dc = |s: SymbolId, w: u32| -> Result<String, String> {
            let pool = self.literals(s);
            pool.get(w as usize).cloned().ok_or_else(|| bug(format!("{}'s literal {w} is past its pool of {}", self.name(s), pool.len())))
        };
        Ok(self.decoder.render(&self.entry, d, &dc)?.body)
    }

    /// The whole program: the entry point binding the tile and the interface from the input vector (the layout of
    /// [`super::inputs`]: `u_width`, `u_height`, `col0`, `row0`, `cols`, `rows`, then the interface), around the
    /// rendered picture.
    pub fn program(&self, iface: &[Input], tree: &Tree) -> Result<String, String> {
        let picture = self.render(&self.decode(tree)?)?;
        let mut s = String::from("entry main (ps: []f32) : [][][3]f32 =\n");
        s.push_str("  let col0 = ps[2]\n  let row0 = ps[3]\n  let cols = i64.f32 ps[4]\n  let rows = i64.f32 ps[5]\n");
        s.push_str("  let u_width = ps[0]\n  let u_height = ps[1]\n");
        for (i, input) in iface.iter().enumerate() {
            s.push_str(&format!("  let {} = ps[{}]\n", input.spelling, HEADER + i));
        }
        s.push_str(&format!("  in {picture}\n"));
        Ok(s)
    }

    /// A random gene over `layout` (a head cell a function or a terminal of its type at one half each, a tail cell a
    /// terminal, a Dc cell a word below its pool, a Dv cell a word below `DV_WORDS`; the root's first cell a function).
    pub fn random_gene(&self, layout: &GeneLayout, rng: &mut crate::futhark_table::GeneRng) -> Vec<u32> {
        let k = &self.contract;
        let const_len = self.const_len();
        let mut gene = vec![0u32; layout.width];
        for s in layout.spans() {
            let (functions, terminals) = (k.functions_of(s.ty), k.terminals_of(s.ty));
            for (i, cell) in gene[s.off..s.off + s.len].iter_mut().enumerate() {
                *cell = match s.domain {
                    Domain::Head => {
                        let force = s.ty == layout.root && i == 0 && !functions.is_empty();
                        if !functions.is_empty() && (terminals.is_empty() || force || rng.below(2) == 0) {
                            functions[rng.below(functions.len())]
                        } else {
                            terminals[rng.below(terminals.len())]
                        }
                    }
                    Domain::Tail => terminals[rng.below(terminals.len())],
                    Domain::Dc => rng.below(const_len.get(&s.ty).copied().unwrap_or(1).max(1) as usize) as u32,
                    Domain::Dv => rng.below(crate::typed_gep::DV_WORDS as usize) as u32,
                };
            }
        }
        gene
    }
}

/// What an s-expression position holds while it is read.
#[derive(Debug, Clone, Copy)]
enum Place {
    /// A block (`blk`, `None`) or a body of a state arity (`body<s>`).
    Block(Option<usize>),
    Scalar,
    Count,
}

impl PixelRows {
    /// READ A PROGRAM written as an s-expression of the family's forms (the shipped references:
    /// [`super::VOLUMEPASSAGE_REFERENCE`], [`super::CUNEUS_REFERENCE`]) into a tree of the rows. The text names its
    /// variables as the old tables did: an interface input (`u_time`), the position (`px`, `py`) or `v<k>`, the k-th
    /// variable bound on the way down (a `let` binds one, a block of state arity s binds s for its rest, a loop binds its
    /// s states and its counter in the body and its s results for its rest). Each becomes `var@f32` with the index word
    /// that names it in the decoder's environment order ([`crate::typed_environment::in_scope`]: innermost first). A
    /// literal must be in the f32 pool and a trip count in the count pool; a form with no row is an error that names it.
    pub fn read(&self, text: &str) -> Result<Tree, String> {
        let sexp = super::parse_sexp(text)?;
        let super::Sexp::List(items) = &sexp else { return Err("the program is not (image BLOCK)".into()) };
        if items.len() != 2 || items[0] != super::Sexp::Atom("image".into()) {
            return Err("the program is not (image BLOCK)".into());
        }
        // The environment, outermost first, as the decoder builds it: the entry's parameters, then the picture's px, py.
        let mut scope: Vec<String> = self.entry.parameters.iter().map(|p| p.name.clone()).collect();
        scope.extend(["px".to_string(), "py".to_string()]);
        let block = self.read_at(&items[1], Place::Block(None), &scope, 0)?;
        Ok(Tree { symbol: self.sym("image")?, const_index: None, kids: vec![block] })
    }

    fn sym(&self, name: &str) -> Result<SymbolId, String> {
        self.symbol_named(name).ok_or_else(|| format!("{}: the kingdom has no row {name}", self.variant))
    }

    /// `scope`: the reference's names in binding order (outermost first); `bound`: how many `v<k>` are bound.
    fn read_at(&self, e: &super::Sexp, at: Place, scope: &[String], bound: usize) -> Result<Tree, String> {
        use super::Sexp;
        let leaf = |symbol: SymbolId, index: u32| Tree { symbol, const_index: Some(index), kids: vec![] };
        let pool_index = |name: &str, a: &str| -> Result<u32, String> {
            let s = self.sym(name)?;
            let v: f64 = a.trim_end_matches("f32").parse().map_err(|_| format!("{a} is not a number"))?;
            self.literals(s)
                .iter()
                .position(|l| l.trim_end_matches("f32").parse::<f64>().map(|x| (x as f32).to_bits()) == Ok((v as f32).to_bits()))
                .map(|i| i as u32)
                .ok_or_else(|| format!("the literal {a} is not in the pool of {name}"))
        };
        let tag = |b: Option<usize>| b.map_or("blk".to_string(), |s| format!("body{s}"));
        let node = |name: &str, kids: Vec<Tree>| -> Result<Tree, String> { Ok(Tree { symbol: self.sym(name)?, const_index: None, kids }) };
        match (e, at) {
            (Sexp::Atom(a), Place::Count) => Ok(leaf(self.sym("?@count")?, pool_index("?@count", a)?)),
            (Sexp::Atom(a), Place::Scalar) => {
                if a.parse::<f64>().is_ok() {
                    return Ok(leaf(self.sym("?@f32")?, pool_index("?@f32", a)?));
                }
                if self.symbol_named(a).is_some() {
                    // A nullary function of the instance (an image target's nx, ny).
                    return node(a, vec![]);
                }
                // The innermost binding of the name, as the decoder's environment lists it.
                let place = scope.iter().rev().position(|n| n == a).ok_or_else(|| format!("{a} is not in scope"))?;
                Ok(leaf(self.sym("var@f32")?, place as u32))
            }
            (Sexp::Atom(a), Place::Block(b)) => {
                let name = match (a.as_str(), b) {
                    ("black" | "coord", None) => a.clone(),
                    ("zero", Some(s)) => format!("zero@body{s}"),
                    _ => return Err(format!("{a} is not a terminal of {}", tag(b))),
                };
                node(&name, vec![])
            }
            (Sexp::List(items), _) => {
                let Some(Sexp::Atom(head)) = items.first() else { return Err("a list with no head".into()) };
                let args = &items[1..];
                let scalars = |xs: &[Sexp]| xs.iter().map(|x| self.read_at(x, Place::Scalar, scope, bound)).collect::<Result<Vec<_>, _>>();
                let more = |n: usize| -> Vec<String> {
                    let mut s = scope.to_vec();
                    s.extend((bound..bound + n).map(|k| format!("v{k}")));
                    s
                };
                match at {
                    Place::Scalar => node(head, scalars(args)?),
                    Place::Count => Err("a count is a literal".into()),
                    Place::Block(b) => match head.as_str() {
                        "let" => {
                            let [value, rest] = args else { return Err("let takes VALUE REST".into()) };
                            let v = self.read_at(value, Place::Scalar, scope, bound)?;
                            let r = self.read_at(rest, at, &more(1), bound + 1)?;
                            node(&format!("let@{}", tag(b)), vec![v, r])
                        }
                        h if h.starts_with("loop") => {
                            let s: usize = h[4..].parse().map_err(|e| format!("{h}: {e}"))?;
                            let [Sexp::List(inits), count, body, rest] = args else { return Err(format!("{h} takes (INITS) COUNT BODY REST")) };
                            if inits.len() != s {
                                return Err(format!("{h} has {} initial values", inits.len()));
                            }
                            let mut kids = scalars(inits)?;
                            kids.push(self.read_at(count, Place::Count, scope, bound)?);
                            kids.push(self.read_at(body, Place::Block(Some(s)), &more(s + 1), bound + s + 1)?);
                            kids.push(self.read_at(rest, at, &more(s), bound + s)?);
                            node(&format!("{h}@{}", tag(b)), kids)
                        }
                        h if h.starts_with("block") => {
                            let s: usize = h[5..].parse().map_err(|e| format!("{h}: {e}"))?;
                            let [body, rest] = args else { return Err(format!("{h} takes BODY REST")) };
                            let inner = self.read_at(body, Place::Block(Some(s)), scope, bound)?;
                            let r = self.read_at(rest, at, &more(s), bound + s)?;
                            node(&format!("{h}@{}", tag(b)), vec![inner, r])
                        }
                        "rgb" if b.is_none() => node("rgb", scalars(args)?),
                        "tuple" if b.is_some() => node(&format!("tuple@{}", tag(b)), scalars(args)?),
                        other => Err(format!("{other} is not a form of {} (let, loopS, blockS, rgb, tuple)", tag(b))),
                    },
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
