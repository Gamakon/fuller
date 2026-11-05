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

use egglog::extract::hff_extract;
use egglog::prelude::exprs;
use egglog::{EGraph, Term, TermDag, TermId};

use crate::geneframe::{Entry, KingdomTables, Role, Ty};
use crate::kingdom_contract::{relations_from_tables, KingdomContract, SymbolId};
use crate::saturated_fold::EgraphKingdom;
use crate::typed_environment::{Choice, Decoder, Decoding, Node};
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

    /// A random gene over `layout`: head cells by `draw` ([`HeadDraw::Half`]: a function or a terminal of the cell's
    /// type at one half each, each uniform over its kind; [`HeadDraw::Uniform`]: uniform over both), tail cells a
    /// terminal, Dc cells a word below the pool, Dv cells a word below `DV_WORDS`; the root's first cell is the `image`
    /// row (a gene whose root is the black picture draws nothing to evolve from).
    pub fn random_gene(&self, layout: &GeneLayout, draw: HeadDraw, rng: &mut crate::futhark_table::GeneRng) -> Result<Vec<u32>, String> {
        let k = &self.contract;
        let const_len = self.const_len();
        let image = self.sym("image")?;
        let mut gene = vec![0u32; layout.width];
        for s in layout.spans() {
            let (functions, terminals) = (k.functions_of(s.ty), k.terminals_of(s.ty));
            for (i, cell) in gene[s.off..s.off + s.len].iter_mut().enumerate() {
                *cell = match s.domain {
                    Domain::Head if s.ty == layout.root && i == 0 => image,
                    Domain::Head => match draw {
                        HeadDraw::Half if !functions.is_empty() && (terminals.is_empty() || rng.below(2) == 0) => functions[rng.below(functions.len())],
                        HeadDraw::Half => terminals[rng.below(terminals.len())],
                        HeadDraw::Uniform => {
                            let n = rng.below(functions.len() + terminals.len());
                            if n < functions.len() {
                                functions[n]
                            } else {
                                terminals[n - functions.len()]
                            }
                        }
                    },
                    Domain::Tail => terminals[rng.below(terminals.len())],
                    Domain::Dc => rng.below(const_len.get(&s.ty).copied().unwrap_or(1).max(1) as usize) as u32,
                    Domain::Dv => rng.below(crate::typed_gep::DV_WORDS as usize) as u32,
                };
            }
        }
        Ok(gene)
    }

    /// The trips of a decoded `count` node: its pool literal.
    fn trips(&self, n: &Node) -> Result<u64, String> {
        match (n.substituted, &n.choice) {
            (Some(_), _) => Ok(1),
            (None, Some(Choice::Dc(w))) => self.literals(n.symbol).get(*w as usize).ok_or_else(|| bug(format!("a count past its pool: {w}")))?.parse().map_err(|e| bug(format!("a count that is not a number: {e}"))),
            _ => Err(bug("a count node with no pool word".into())),
        }
    }

    /// The node executions of one pixel: every rendered node once, times the trips of the loops it is inside (a loop
    /// row's body is its second-last argument and its trip count the `count` argument before it). The exact bound of
    /// the decoded tree.
    pub fn work_per_pixel(&self, d: &Decoding) -> Result<u128, String> {
        fn walk(k: &PixelRows, n: &Node, mult: u128) -> Result<u128, String> {
            let mut sum = mult;
            if n.substituted.is_some() {
                return Ok(sum);
            }
            let kids = n.kids.len();
            let is_loop = k.name(n.symbol).starts_with("loop");
            let trips = if is_loop { u128::from(k.trips(&n.kids[kids - 3])?) } else { 1 };
            for (i, kid) in n.kids.iter().enumerate() {
                sum += walk(k, kid, if is_loop && i == kids - 2 { mult * trips } else { mult })?;
            }
            Ok(sum)
        }
        walk(self, &d.root, 1)
    }

    /// THE VALUE OF A DECODED PICTURE AT ONE PIXEL, on the host: the family's meaning written out row by row (a test
    /// oracle and a host-side measurement, never the fitness path, which runs the compiled program on the GPU). `ps`
    /// is the input vector of [`super::inputs`]; `px`, `py` the pixel. A node a repair replaced is its type's constant.
    /// The arithmetic is f32 as Futhark's: `%` truncated, `**` `powf`, `f32.max`/`min` ignoring a NaN operand.
    pub fn eval_pixel(&self, d: &Decoding, iface: &[Input], ps: &[f32], px: f32, py: f32) -> Result<[f32; 3], String> {
        let mut values: BTreeMap<String, f32> = BTreeMap::from([("u_width".to_string(), ps[0]), ("u_height".to_string(), ps[1])]);
        for (i, input) in iface.iter().enumerate() {
            values.insert(input.spelling.clone(), *ps.get(HEADER + i).ok_or("the input vector is short")?);
        }
        let root = &d.root;
        if root.substituted.is_some() || self.name(root.symbol) == "image.black" {
            return Ok([0.0; 3]);
        }
        let block = root.kids.first().ok_or("an image row with no block")?;
        let mut e = Eval { k: self, values, px, py };
        e.bind(root, block, &[px, py])?;
        match e.block(block)?.as_slice() {
            [r, g, b] => Ok([*r, *g, *b]),
            other => Err(bug(format!("a pixel block gave {} values", other.len()))),
        }
    }
}

/// How a random gene draws its head cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadDraw {
    /// Uniform over the cell's whole alphabet (functions and terminals alike).
    Uniform,
    /// A function with probability one half, a terminal otherwise, each uniform over its kind (phylu's
    /// `FunctionShare::half`). A cell whose alphabet has only one kind draws from it.
    Half,
}

struct Eval<'a> {
    k: &'a PixelRows,
    values: BTreeMap<String, f32>,
    px: f32,
    py: f32,
}

impl Eval<'_> {
    /// Give the variables `parent` binds into `child` (those `child`'s environment adds) their values, in order.
    fn bind(&mut self, parent: &Node, child: &Node, vals: &[f32]) -> Result<(), String> {
        let new = &child.environment.variables[parent.environment.variables.len()..];
        if new.len() != vals.len() {
            return Err(bug(format!("{} binds {} variables, the interpreter has {} values", self.k.name(parent.symbol), new.len(), vals.len())));
        }
        for (v, x) in new.iter().zip(vals) {
            self.values.insert(v.name.clone().ok_or_else(|| bug("a bound variable with no name".into()))?, *x);
        }
        Ok(())
    }

    /// A substituted node's value: its type's constant (the black pixel, the zero tuple, 0).
    fn constant(&self, n: &Node) -> Vec<f32> {
        let name = self.k.tables.data_type(n.data_type).map_or("", |t| t.name.as_str());
        match name {
            "blk" => vec![0.0; 3],
            b if b.starts_with("body") => vec![0.0; b[4..].parse().unwrap_or(0)],
            _ => vec![0.0],
        }
    }

    fn block(&mut self, n: &Node) -> Result<Vec<f32>, String> {
        if n.substituted.is_some() {
            return Ok(self.constant(n));
        }
        let name = self.k.name(n.symbol).to_string();
        let op = name.split('@').next().unwrap_or("");
        match op {
            "let" => {
                let v = self.scalar(&n.kids[0])?;
                self.bind(n, &n.kids[1], &[v])?;
                self.block(&n.kids[1])
            }
            "rgb" | "tuple" => n.kids.iter().map(|c| self.scalar(c)).collect(),
            "black" => Ok(vec![0.0; 3]),
            "coord" => Ok(vec![self.px / self.values["u_width"], self.py / self.values["u_height"], 0.5]),
            "zero" => Ok(self.constant(n)),
            o if o.starts_with("block") => {
                let vals = self.block(&n.kids[0])?;
                self.bind(n, &n.kids[1], &vals)?;
                self.block(&n.kids[1])
            }
            o if o.starts_with("loop") => {
                let s: usize = o[4..].parse().map_err(|_| bug(format!("{o} has no state arity")))?;
                let mut state: Vec<f32> = n.kids[..s].iter().map(|c| self.scalar(c)).collect::<Result<_, _>>()?;
                let trips = self.k.trips(&n.kids[s])?;
                for i in 0..trips {
                    let mut vals = state.clone();
                    vals.push(i as f32);
                    self.bind(n, &n.kids[s + 1], &vals)?;
                    state = self.block(&n.kids[s + 1])?;
                    if state.len() != s {
                        return Err(bug(format!("{o}'s body gave {} values", state.len())));
                    }
                }
                self.bind(n, &n.kids[s + 2], &state)?;
                self.block(&n.kids[s + 2])
            }
            other => Err(bug(format!("{other} is not a block row the interpreter knows"))),
        }
    }

    fn scalar(&mut self, n: &Node) -> Result<f32, String> {
        if n.substituted.is_some() {
            return Ok(0.0);
        }
        match &n.choice {
            Some(Choice::Dc(w)) => {
                let text = self.k.literals(n.symbol).get(*w as usize).cloned().ok_or_else(|| bug(format!("a literal past its pool: {w}")))?;
                return text.trim_end_matches("f32").parse::<f32>().map_err(|e| bug(format!("the literal {text}: {e}")));
            }
            Some(Choice::Variable(b)) => {
                let v = n.environment.variables.iter().find(|v| v.binder == *b).ok_or_else(|| bug("a variable not in scope".into()))?;
                let name = v.name.as_deref().ok_or_else(|| bug("a variable with no name".into()))?;
                return self.values.get(name).copied().ok_or_else(|| bug(format!("the variable {name} has no value")));
            }
            _ => {}
        }
        let op = self.k.name(n.symbol).to_string();
        if n.kids.is_empty() {
            return match op.as_str() {
                "nx" => Ok(self.px / self.values["u_width"]),
                "ny" => Ok(self.py / self.values["u_height"]),
                other => Err(bug(format!("{other} is not a scalar terminal the interpreter knows"))),
            };
        }
        let a = self.scalar(&n.kids[0])?;
        if n.kids.len() == 3 {
            // sel: if a >= 0 then b else c (only the taken branch is evaluated, as Futhark's `if`).
            return if a >= 0.0 { self.scalar(&n.kids[1]) } else { self.scalar(&n.kids[2]) };
        }
        if n.kids.len() == 2 {
            let b = self.scalar(&n.kids[1])?;
            return Ok(match op.as_str() {
                "add" => a + b,
                "sub" => a - b,
                "mul" => a * b,
                "div" => a / b,
                "max" => a.max(b),
                "min" => a.min(b),
                "pow" => a.powf(b),
                "mod" => a % b,
                o => return Err(bug(format!("{o} is not a binary row"))),
            });
        }
        Ok(match op.as_str() {
            "neg" => -a,
            "abs" => a.abs(),
            "sqrt" => a.sqrt(),
            "sin" => a.sin(),
            "cos" => a.cos(),
            "tan" => a.tan(),
            "tanh" => a.tanh(),
            "exp" => a.exp(),
            "log" => a.ln(),
            "floor" => a.floor(),
            "trunc" => a.trunc(),
            o => return Err(bug(format!("{o} is not a unary row"))),
        })
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

// ---- the instance's e-graph, over the rows ---------------------------------------------------------------------------

/// The symbol <-> constructor mapping of an instance's e-graph (`egraph.rows`, generated with the datatype).
#[derive(Debug, Clone)]
pub struct Bridge {
    ctor_of_symbol: BTreeMap<String, String>,
    symbol_of_ctor: BTreeMap<String, String>,
}

impl Bridge {
    /// Read an instance's mapping; a repeated symbol or constructor is a bug.
    pub fn load(inst: &super::Instance) -> Result<Bridge, String> {
        let (mut ctor_of_symbol, mut symbol_of_ctor) = (BTreeMap::new(), BTreeMap::new());
        for line in inst.egraph_rows.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 3 {
                return Err(bug(format!("the e-graph mapping line {line:?} is malformed")));
            }
            if ctor_of_symbol.insert(f[0].to_string(), f[1].to_string()).is_some() || symbol_of_ctor.insert(f[1].to_string(), f[0].to_string()).is_some() {
                return Err(bug(format!("the e-graph mapping line {line:?} repeats a symbol or a constructor")));
            }
        }
        Ok(Bridge { ctor_of_symbol, symbol_of_ctor })
    }
}

/// The instance's e-graph over its rows: the kingdom, its datatype and rules, and the mapping.
pub struct PixelEgraph<'a> {
    pub kingdom: &'a PixelRows,
    pub inst: &'a super::Instance,
    pub bridge: Bridge,
}

impl<'a> PixelEgraph<'a> {
    pub fn new(kingdom: &'a PixelRows, inst: &'a super::Instance) -> Result<PixelEgraph<'a>, String> {
        let bridge = Bridge::load(inst)?;
        for s in 0..kingdom.contract.n_symbols() as SymbolId {
            if !bridge.ctor_of_symbol.contains_key(kingdom.name(s)) {
                return Err(bug(format!("{}: the symbol {} has no e-graph constructor; regenerate the e-graph", kingdom.variant, kingdom.name(s))));
            }
        }
        Ok(PixelEgraph { kingdom, inst, bridge })
    }

    /// A tree as an egglog term: a constant's or a variable's constructor applied to its word (the pool index, or the
    /// index word naming a variable in the node's environment; no rule moves a subterm between environments).
    pub fn import(&self, tree: &Tree) -> Result<String, String> {
        let name = self.kingdom.name(tree.symbol);
        let ctor = self.bridge.ctor_of_symbol.get(name).ok_or_else(|| bug(format!("the symbol {name} has no constructor")))?;
        if let Some(w) = tree.const_index {
            return Ok(format!("({ctor} {w})"));
        }
        let kids = tree.kids.iter().map(|t| self.import(t)).collect::<Result<Vec<_>, _>>()?;
        Ok(if kids.is_empty() { format!("({ctor})") } else { format!("({ctor} {})", kids.join(" ")) })
    }

    /// An egglog term as a tree of the rows; `None` when it is not one.
    pub fn to_tree(&self, termdag: &TermDag, id: TermId) -> Option<Tree> {
        let Term::App(op, kids) = termdag.get(id) else { return None };
        let symbol = self.kingdom.symbol_named(self.bridge.symbol_of_ctor.get(op.as_str())?)?;
        if self.kingdom.contract.is_terminal(symbol) && kids.len() == 1 {
            let Term::Lit(egglog::ast::Literal::Int(i)) = termdag.get(kids[0]) else { return None };
            return Some(Tree { symbol, const_index: Some(u32::try_from(*i).ok()?), kids: vec![] });
        }
        let kids = kids.iter().map(|t| self.to_tree(termdag, *t)).collect::<Option<Vec<_>>>()?;
        Some(Tree { symbol, const_index: None, kids })
    }

    /// The e-graph variants of a tree: saturate `iters` iterations of the instance's ruleset, extract the ranked
    /// members of the root's class (HFF extraction, `members` a class), and return up to `n` distinct trees other than
    /// the tree itself, smallest first and then by their rendered text, so the order is deterministic.
    pub fn variants(&self, tree: &Tree, iters: u32, members: usize, n: usize) -> Result<Vec<Tree>, String> {
        let term = self.import(tree)?;
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(None, self.inst.datatype).map_err(|e| bug(format!("the datatype does not load: {e}")))?;
        egraph.parse_and_run_program(None, self.inst.rules).map_err(|e| bug(format!("the ruleset does not load: {e}")))?;
        egraph
            .parse_and_run_program(None, &format!("(let __root {term})\n(run-schedule (repeat {iters} (run {})))", super::RULESET))
            .map_err(|e| bug(format!("the term does not load: {e}")))?;
        let (sort, value) = egraph.eval_expr(&exprs::var("__root")).map_err(|e| bug(format!("the root does not evaluate: {e}")))?;
        let mut termdag = TermDag::default();
        let score = |td: &TermDag, t: TermId| term_cost(td, t) as f64;
        let ranked = hff_extract(&egraph, &mut termdag, value, sort, &score, members);
        let mut out: Vec<(u64, String, Tree)> = Vec::new();
        for (cost, id) in ranked {
            let Some(t) = self.to_tree(&termdag, id) else { continue };
            if &t == tree {
                continue;
            }
            let text = self.kingdom.render(&self.kingdom.decode(&t)?)?;
            if out.iter().any(|(_, s, _)| s == &text) {
                continue;
            }
            out.push((cost as u64, text, t));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        out.truncate(n);
        Ok(out.into_iter().map(|(_, _, t)| t).collect())
    }
}

/// The cost of a term: its node count.
fn term_cost(termdag: &TermDag, id: TermId) -> u64 {
    match termdag.get(id) {
        Term::App(_, kids) => 1 + kids.iter().map(|k| term_cost(termdag, *k)).sum::<u64>(),
        _ => 1,
    }
}

impl EgraphKingdom for PixelEgraph<'_> {
    fn programs(&self) -> Vec<&str> {
        vec![self.inst.datatype, self.inst.rules]
    }
    fn rulesets(&self) -> Vec<&str> {
        vec![super::RULESET]
    }
    fn import(&self, tree: &Tree) -> Result<String, String> {
        PixelEgraph::import(self, tree)
    }
    fn row(&self, constructor: &str, literals: &[&str]) -> Option<(SymbolId, Option<u32>)> {
        let symbol = self.kingdom.symbol_named(self.bridge.symbol_of_ctor.get(constructor)?)?;
        match literals {
            [w] if self.kingdom.contract.is_terminal(symbol) => Some((symbol, Some(w.parse().ok()?))),
            [] => Some((symbol, None)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
