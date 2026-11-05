//! MASTER 6's INTERPRETER AND WORK (plan section 7, task F3). The meaning of a master 6 gene on the host, read from
//! its DECODING (`crate::typed_environment`, passes 2 to 4), never from the types: a variable is the binder its
//! environment names, a length or a bound is a property evaluated at the case's sizes, and every adapter the decoding
//! says renders is applied as the renderer spells it (`futhark_table::Render6`): `fit` takes the first elements of
//! its operand and pads with zeros, `clampix` clamps an index into its array, a substituted node is its substitute.
//! The arithmetic is master 5's ([`super::Num`]): Futhark's, bit for bit.
//!
//! THE INTERPRETER CHECKS WHAT FUTHARK CHECKS: an index in its array's range, the two arrays of a `map2` (a histogram,
//! a division of rows, a product's inner sizes) of one length, every row of a `tabulate` of rows of one length, and
//! every value's shape against its properties (an array's lengths, an index below its bound, a size its value), so a
//! renderer and an interpreter that disagree on a length are found as a run-time error, never as a silently
//! different value. Every such error is "our software has a bug".
//!
//! THE WORK ([`Interp6::work_parts`]) is master 5's measure (`Interp::work_parts`) with every size read from the
//! properties, and every adapter charged where it renders (`futhark_cost::adapters6`, the renderer's decision): a
//! `fit` the cells it writes, `clampix` its two compares, counted in [`Work::nodes`], the measure phylu's per-submit
//! cap reads (`submit_work_of`), a `fit` also the cells it reads of its operand. A size leaf costs nothing (it is a name, as master 5's row names hold their sizes),
//! so a reference program's work is master 5's.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::{Array, Bin, Elem, Num, Signature, Val, View, Work};
use crate::futhark_cost::{adapters6, program_cost6, rendered6, requirements6, size6, trips6, CostTable6, Score};
use crate::futhark_interpreter::{Row2Fn, RowFn};
use crate::futhark_table::{Kind6, Kingdom6};
use crate::geneframe::{AdapterSite, DataKind, PropertyExpr, Requirement, Subject, Ty};
use crate::typed_environment::{Binder, Choice, Decoder, Decoding, Entry, Node, Outcome, Substitution};
use crate::typed_gep::Tree;

/// What a master 6 row does, parsed once from its master row (`ref`) and its data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op6 {
    Bin(Bin),
    Neg,
    Abs,
    Div,
    Sqrt,
    Sel,
    Get,
    /// A reduction of an array: `sum`, `minimum`, `maximum`.
    Fold(Bin),
    Ival(Elem),
    Sval(Elem),
    Mod,
    Literal,
    Prev,
    Next,
    Clamp,
    ZeroIndex,
    Last,
    Tab(Elem),
    Replicate(Elem),
    Map2(Bin),
    Scan(Bin),
    Hist(Bin, Elem),
    Row,
    /// A loop carrying a state of `arity` components (a tuple's destructured), from 1 when `from1`.
    Loop { from1: bool, arity: usize },
    MapRows,
    Map2Rows,
    Transpose,
    Matmul,
    RowFn(RowFn),
    Row2Fn(Row2Fn),
    TabRows,
    HistRows(Elem),
    RowDiv,
    Tuple,
    Proj(usize),
    PSel,
    Cvt(Elem),
    Let,
    Var,
    /// The data type's constant (`zero` of a tuple, M0088's leafless constant).
    Constant,
}

/// What the shape check reads of a data type, gathered once: an array's element and property names in order, an
/// index, a size, a scalar's element, a tuple's components, a function's parameter count.
#[derive(Debug, Clone)]
enum Shape6 {
    Array(Elem, Vec<String>),
    Index,
    Size,
    Scalar(Elem),
    Tuple(Vec<Ty>),
    Function(usize),
}

/// Where a root variable's value comes from: an input of the case, a zero array, a size.
#[derive(Debug, Clone)]
enum Source6 {
    Input(usize),
    Zeros { len: PropertyExpr, elem: Elem },
    Size(String),
}

/// The interpreter of one master 6 view, prepared once: its decoder and entry, each symbol's meaning and literals, the
/// requirement rows, the root's sources, and the cost table.
pub(super) struct Interp6 {
    k: Arc<Kingdom6>,
    sig: Arc<Signature>,
    decoder: Decoder,
    entry: Entry,
    ops: Vec<Op6>,
    literals: Vec<Vec<Num>>,
    reqs: Vec<Requirement>,
    roots: Vec<Source6>,
    costs: CostTable6,
    shapes: Vec<Shape6>,
}

fn bug(k: &Kingdom6, what: String) -> String {
    format!("our software has a bug: {}: {what}", k.kingdom)
}

/// The element of a scalar data type, or of an array's innermost scalar.
fn elem6(k: &Kingdom6, ty: Ty) -> Result<Elem, String> {
    let mut t = ty;
    loop {
        let d = k.data_type(t).ok_or_else(|| bug(k, format!("{ty:?} is no data type")))?;
        match d.kind {
            DataKind::Array { element } => t = element,
            _ => return Elem::of(&d.name).map_err(|e| bug(k, format!("{} has no element: {e}", d.name))),
        }
    }
}

/// The meaning of one symbol, by its master row.
fn op6(k: &Kingdom6, s: &crate::futhark_table::Symbol6) -> Result<Op6, String> {
    let out = || elem6(k, s.output);
    let arity = || -> Result<usize, String> {
        let n = match &k.data_type(s.output).ok_or_else(|| bug(k, format!("{} has no data type", s.name)))?.kind {
            DataKind::Tuple { components } => components.len(),
            _ => 1,
        };
        if s.binds.len() != n + 1 {
            return Err(bug(k, format!("the loop {} binds {} variables, wants its state's {n} and its counter", s.name, s.binds.len())));
        }
        Ok(n)
    };
    Ok(match s.reference.as_str() {
        "M0001" => Op6::Bin(Bin::Add),
        "M0002" => Op6::Bin(Bin::Sub),
        "M0003" => Op6::Bin(Bin::Mul),
        "M0004" => Op6::Bin(Bin::Min),
        "M0005" => Op6::Bin(Bin::Max),
        "M0006" => Op6::Neg,
        "M0007" => Op6::Abs,
        "M0008" => Op6::Sel,
        "M0009" => Op6::Get,
        "M0010" => Op6::Fold(Bin::Add),
        "M0011" => Op6::Fold(Bin::Min),
        "M0012" => Op6::Fold(Bin::Max),
        "M0013" => Op6::Ival(out()?),
        "M0014" => Op6::Sval(out()?),
        "M0015" => Op6::Mod,
        "M0016" => Op6::Literal,
        "M0018" => Op6::Prev,
        "M0019" => Op6::Next,
        "M0020" | "M0056" => Op6::Clamp,
        "M0021" => Op6::ZeroIndex,
        "M0022" => Op6::Last,
        "M0024" => Op6::Tab(out()?),
        "M0025" => Op6::Replicate(out()?),
        "M0026" => Op6::Map2(Bin::Add),
        "M0027" => Op6::Map2(Bin::Sub),
        "M0028" => Op6::Map2(Bin::Min),
        "M0029" => Op6::Map2(Bin::Max),
        "M0062" => Op6::Map2(Bin::Mul),
        "M0030" => Op6::Scan(Bin::Add),
        "M0031" => Op6::Scan(Bin::Min),
        "M0032" => Op6::Scan(Bin::Max),
        "M0033" | "M0057" => Op6::Hist(Bin::Add, out()?),
        "M0034" | "M0058" => Op6::Hist(Bin::Min, out()?),
        "M0035" | "M0059" => Op6::Hist(Bin::Max, out()?),
        "M0036" => Op6::Row,
        "M0040" | "M0073" | "M0083" => Op6::Loop { from1: false, arity: arity()? },
        "M0041" | "M0074" | "M0084" => Op6::Loop { from1: true, arity: arity()? },
        "M0043" => Op6::MapRows,
        "M0044" => Op6::Map2Rows,
        "M0045" => Op6::Transpose,
        "M0046" => Op6::Matmul,
        "M0047" => Op6::RowFn(RowFn::Id),
        "M0048" => Op6::RowFn(RowFn::Add1),
        "M0049" => Op6::RowFn(RowFn::Scale2),
        "M0054" => Op6::RowFn(RowFn::Zero),
        "M0050" => Op6::Row2Fn(Row2Fn::Id2),
        "M0051" => Op6::Row2Fn(Row2Fn::Add1_2),
        "M0052" => Op6::Row2Fn(Row2Fn::Add2),
        "M0055" => Op6::Row2Fn(Row2Fn::Zero2),
        "M0060" => Op6::Div,
        "M0061" => Op6::Sqrt,
        "M0063" => Op6::TabRows,
        "M0064" | "M0065" => Op6::HistRows(out()?),
        "M0066" => Op6::RowDiv,
        "M0068" | "M0077" => Op6::Tuple,
        "M0069" | "M0078" => Op6::Proj(0),
        "M0070" | "M0079" => Op6::Proj(1),
        "M0080" => Op6::Proj(2),
        "M0071" | "M0081" | "M0088" => Op6::Constant,
        "M0072" | "M0082" => Op6::PSel,
        "M0075" | "M0076" => Op6::Cvt(out()?),
        "M0085" => Op6::Let,
        "M0087" => Op6::Var,
        other => return Err(bug(k, format!("the row {} ({other}) has no meaning in the interpreter", s.name))),
    })
}

/// The extent of an array value along a property (`len` and `rows` its outer length, `cols` its inner), and a size's
/// `value`.
fn extent(k: &Kingdom6, v: &Val, p: &str) -> Result<i64, String> {
    match (v, p) {
        (Val::A(a), "len") if a.dims.len() == 1 => Ok(a.dims[0] as i64),
        (Val::A(a), "rows") if a.dims.len() == 2 => Ok(a.dims[0] as i64),
        (Val::A(a), "cols") if a.dims.len() == 2 => Ok(a.dims[1] as i64),
        (Val::I(i), "value") => Ok(*i),
        _ => Err(bug(k, format!("the property {p} is read of {v:?}"))),
    }
}

/// THE ADAPTER `fit` on a value: exactly `l` along `p` (`len` or `rows`: the first `l` elements or rows, padded with
/// zeros or zero rows; `cols`: every row so), as `FIT_HELPER` computes.
fn fit(k: &Kingdom6, v: &Val, p: &str, l: i64) -> Result<Val, String> {
    let Val::A(a) = v else { return Err(bug(k, format!("fit on {v:?}"))) };
    let l = usize::try_from(l).map_err(|_| bug(k, format!("fit to the length {l}")))?;
    let src = a.slice();
    let (data, dims) = match (p, a.dims.as_slice()) {
        ("len", [m]) | ("rows", [m, _]) => {
            let stride = if a.dims.len() == 2 { a.dims[1] } else { 1 };
            let mut out = vec![0u32; l * stride];
            let keep = l.min(*m) * stride;
            out[..keep].copy_from_slice(&src[..keep]);
            let mut dims = a.dims.clone();
            dims[0] = l;
            (out, dims)
        }
        ("cols", [r, c]) => {
            let mut out = vec![0u32; r * l];
            let keep = l.min(*c);
            for i in 0..*r {
                out[i * l..i * l + keep].copy_from_slice(&src[i * c..i * c + keep]);
            }
            (out, vec![*r, l])
        }
        _ => return Err(bug(k, format!("fit along {p} of an array of {:?}", a.dims))),
    };
    Ok(Val::A(View::of(data, dims, a.elem)))
}

impl Interp6 {
    /// The interpreter of a master 6 kingdom whose case inputs are `sig`'s (its master 5 instance's signature).
    pub(super) fn new(k: &Arc<Kingdom6>, sig: &Arc<Signature>) -> Result<Interp6, String> {
        let tables = k.tables()?;
        let contract = k.contract()?;
        let decoder = Decoder::from_arcs(Arc::new(tables), Arc::new(contract)).map_err(|d| d.join("\n"))?;
        let entry = Entry::of_kingdom6(k);
        let ops = k.symbols.iter().map(|s| op6(k, s)).collect::<Result<Vec<_>, _>>()?;
        let mut literals = Vec::with_capacity(k.symbols.len());
        for s in &k.symbols {
            literals.push(if s.kind == Kind6::Const {
                let elem = elem6(k, s.output)?;
                let d = k.data_type(s.output).ok_or_else(|| bug(k, format!("{} has no data type", s.name)))?;
                d.literals.iter().map(|l| Num::parse(elem, l.trim_end_matches(elem.name()))).collect::<Result<Vec<_>, _>>()?
            } else {
                Vec::new()
            });
        }
        let mut roots = Vec::new();
        for v in &k.variables {
            roots.push(match v.kind.as_str() {
                "input" => Source6::Input(sig.inputs.iter().position(|p| p.name == v.name).ok_or_else(|| bug(k, format!("the input {} is not in the signature", v.name)))?),
                "zeros" => Source6::Zeros { len: v.properties.iter().find(|(p, _)| p == "len").map(|(_, e)| e.clone()).ok_or_else(|| bug(k, format!("the zero array {} has no length", v.name)))?, elem: elem6(k, v.data_type)? },
                "size" => Source6::Size(v.name.clone()),
                other => return Err(bug(k, format!("a root variable of the kind {other}"))),
            });
        }
        let reqs = requirements6(k).into_iter().cloned().collect();
        let costs = CostTable6::load(k)?;
        let mut shapes = Vec::with_capacity(k.data_types.len());
        for (i, d) in k.data_types.iter().enumerate() {
            let ty = Ty::Futhark(i as u16);
            shapes.push(match &d.kind {
                DataKind::Array { .. } => Shape6::Array(elem6(k, ty)?, d.properties.iter().map(|(p, _)| p.clone()).collect()),
                DataKind::Scalar if d.name == "i64" => Shape6::Index,
                DataKind::Scalar if d.name == "size" => Shape6::Size,
                DataKind::Scalar => Shape6::Scalar(elem6(k, ty)?),
                DataKind::Tuple { components } => Shape6::Tuple(components.clone()),
                DataKind::Function { parameters, .. } => Shape6::Function(parameters.len()),
            });
        }
        Ok(Interp6 { k: Arc::clone(k), sig: Arc::clone(sig), decoder, entry, ops, literals, reqs, roots, costs, shapes })
    }

    /// The decoding of an expressed tree (passes 2 to 4).
    pub(super) fn decode(&self, tree: &Tree) -> Result<Decoding, String> {
        self.decoder.decode(&self.entry, tree)
    }

    /// The value of an expressed tree on the inputs: the program's output, as Futhark computes it.
    pub(super) fn eval(&self, tree: &Tree, inputs: &[Array]) -> Result<Array, String> {
        self.eval_with(tree, inputs, false)
    }

    /// The value of an expressed tree with FUTHARK'S ZERO-DIVIDEND FOLD applied to every division: a division whose
    /// dividend is a zero (`0.0` or `-0.0`) is that dividend, whatever the divisor, where IEEE gives NaN for `0 / 0`
    /// and `0 / NaN` and `-0.0` for `0 / -3` (`docs/issues/2026-11-05_futhark_zero_division_fold.md`). A cell where this
    /// value differs from [`Interp6::eval`]'s is one a zero dividend reached; the `c` comparison allows the fold there
    /// and nowhere else.
    #[cfg(test)]
    pub(super) fn eval_zero_dividends_folded(&self, tree: &Tree, inputs: &[Array]) -> Result<Array, String> {
        self.eval_with(tree, inputs, true)
    }

    /// [`Interp6::eval`], with the zero-dividend fold when `fold`.
    fn eval_with(&self, tree: &Tree, inputs: &[Array], fold: bool) -> Result<Array, String> {
        let sizes = self.sig.sizes(inputs)?;
        let d = self.decode(tree)?;
        let mut root = Vec::with_capacity(self.roots.len());
        for r in &self.roots {
            root.push(Some(match r {
                Source6::Input(i) => Val::A(View::of(inputs[*i].data.clone(), inputs[*i].dims.clone(), inputs[*i].elem)),
                Source6::Zeros { len, elem } => {
                    let n = self.size(len, &sizes)?;
                    Val::A(View::of(vec![0; n as usize], vec![n as usize], *elem))
                }
                Source6::Size(name) => Val::I(*sizes.get(name).ok_or_else(|| bug(&self.k, format!("the size {name} has no value")))? as i64),
            }));
        }
        let mut run = Run { me: self, sizes: &sizes, vars: vec![root], expect: std::collections::HashMap::new(), fold };
        let mut v = run.at(&d.root)?;
        for o in d.root_outcomes.iter().filter(|o| !o.holds) {
            let q = self.k.root.requires.get(o.requirement).ok_or_else(|| bug(&self.k, format!("the entry has no requirement {}", o.requirement)))?;
            v = fit(&self.k, &v, &q.property, self.size(&o.against, &sizes)?)?;
        }
        let want: Vec<usize> = self.k.root.properties.iter().map(|(_, e)| self.size(e, &sizes).map(|x| x as usize)).collect::<Result<_, _>>()?;
        match v {
            Val::A(a) if a.dims == want => Ok(Array { dims: a.dims.clone(), elem: a.elem, data: a.slice().to_vec() }),
            Val::S(x) if want.is_empty() => Ok(Array { dims: vec![], elem: x.elem(), data: vec![x.word()] }),
            other => Err(bug(&self.k, format!("the program's value is {other:?}, the entry returns {want:?}"))),
        }
    }

    /// A property's value at `sizes`; `?` is a bug where a value needs it.
    fn size(&self, e: &PropertyExpr, sizes: &BTreeMap<String, usize>) -> Result<i64, String> {
        size6(e, sizes)?.ok_or_else(|| bug(&self.k, format!("the size {e} is unknown where it is read")))
    }

    /// The constant of a data type (what a `var` with no variable in scope, a substitution by the constant, and a
    /// constant row compute): a scalar's zero, a one-element array of zero, a tuple of zeros, the constant row
    /// function.
    fn constant(&self, ty: Ty) -> Result<Val, String> {
        let d = self.k.data_type(ty).ok_or_else(|| bug(&self.k, format!("{ty:?} is no data type")))?;
        Ok(match &d.kind {
            DataKind::Scalar if d.name == "i64" => Val::I(0),
            DataKind::Scalar if d.name == "size" => return Err(bug(&self.k, "the data type size has no constant".into())),
            DataKind::Scalar => Val::S(Num::zero(elem6(&self.k, ty)?)),
            DataKind::Array { element } => {
                let dims = match self.k.data_type(*element).map(|e| &e.kind) {
                    Some(DataKind::Scalar) => vec![1],
                    Some(DataKind::Array { .. }) => vec![1, 1],
                    _ => return Err(bug(&self.k, format!("the constant of {}", d.name))),
                };
                Val::A(View::of(vec![0], dims, elem6(&self.k, ty)?))
            }
            DataKind::Tuple { components } => Val::Tup(components.iter().map(|c| self.constant(*c)).collect::<Result<_, _>>()?),
            DataKind::Function { parameters, .. } => match parameters.len() {
                1 => Val::F(RowFn::Zero),
                2 => Val::F2(Row2Fn::Zero2),
                n => return Err(bug(&self.k, format!("a row function of {n} parameters"))),
            },
        })
    }

    /// THE SHAPE CHECK's expectation of a node at `sizes` (`Run::at` checks every value against it): an array's
    /// element and lengths, an index at least 0 and below its bound, a size its value.
    fn expect(&self, ty: Ty, props: &BTreeMap<String, PropertyExpr>, sizes: &BTreeMap<String, usize>) -> Result<Expect6, String> {
        let shape = match ty {
            Ty::Futhark(i) => self.shapes.get(usize::from(i)),
            _ => None,
        }
        .ok_or_else(|| bug(&self.k, format!("{ty:?} is no data type")))?;
        let known = |p: &str| -> Result<Option<i64>, String> { props.get(p).map_or(Ok(None), |e| size6(e, sizes)) };
        Ok(match shape {
            Shape6::Array(elem, names) => Expect6::Array(*elem, names.iter().map(|p| known(p)).collect::<Result<_, _>>()?),
            Shape6::Index => Expect6::Index(known("bound")?),
            Shape6::Size => Expect6::Size(known("value")?),
            Shape6::Scalar(e) => Expect6::Scalar(*e),
            Shape6::Tuple(c) => Expect6::Tuple(c.clone()),
            Shape6::Function(n) => Expect6::Function(*n),
        })
    }

    /// A value against its data type and properties (a tuple's components).
    fn check_of(&self, ty: Ty, props: &BTreeMap<String, PropertyExpr>, v: &Val, sizes: &BTreeMap<String, usize>) -> Result<(), String> {
        let shape = match ty {
            Ty::Futhark(i) => self.shapes.get(usize::from(i)),
            _ => None,
        }
        .ok_or_else(|| bug(&self.k, format!("{ty:?} is no data type")))?;
        let known = |p: &str| -> Result<Option<i64>, String> { props.get(p).map_or(Ok(None), |e| size6(e, sizes)) };
        let fine = match (shape, v) {
            (Shape6::Array(elem, names), Val::A(a)) => {
                let mut ok = a.dims.len() == names.len() && a.elem == *elem;
                for (i, p) in names.iter().enumerate() {
                    if ok && known(p)?.is_some_and(|want| want != a.dims[i] as i64) {
                        ok = false;
                    }
                }
                ok
            }
            (Shape6::Index, Val::I(i)) => *i >= 0 && known("bound")?.is_none_or(|b| *i < b),
            (Shape6::Size, Val::I(i)) => known("value")?.is_none_or(|x| x == *i),
            (Shape6::Scalar(e), Val::S(x)) => x.elem() == *e,
            (Shape6::Tuple(components), Val::Tup(parts)) => {
                if parts.len() == components.len() {
                    for (c, p) in components.iter().zip(parts) {
                        self.check_of(*c, &BTreeMap::new(), p, sizes)?;
                    }
                    true
                } else {
                    false
                }
            }
            (Shape6::Function(1), Val::F(_)) | (Shape6::Function(2), Val::F2(_)) => true,
            _ => false,
        };
        if fine {
            Ok(())
        } else {
            Err(bug(&self.k, format!("a value of {ty:?} with {props:?} is {v:?}: the program reads out of bounds or the sizes disagree")))
        }
    }

    /// The node executions of a tree on given sizes, the loops' state traffic included ([`Work::total`]).
    pub(super) fn work(&self, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<u128, String> {
        self.work_parts(tree, sizes).map(|w| w.total())
    }

    /// THE WORK of a tree on given sizes, in master 5's two parts, every size read from the properties (a `?` length
    /// at the largest size in scope) and every adapter charged in [`Work::nodes`] where it renders (a body adapter once
    /// a trip): a `fit` the cells it reads and writes, `clampix` 2. [`Work::nodes`]: every node once, times the trips of the binders it is inside; a reduction, a
    /// `map2`, a scan, a histogram and a division of rows also cost the cells of the array they read (as adapted), a
    /// `replicate` and a histogram their result's length, a histogram of rows and the matrix operations their result's
    /// cells (a product times its inner length); a size leaf nothing. [`Work::state`]: master 5's loop state traffic.
    pub(super) fn work_parts(&self, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<Work, String> {
        let d = self.decode(tree)?;
        let reqs: Vec<&Requirement> = self.reqs.iter().collect();
        let mut w = Work::default();
        self.walk(&d.root, 1, sizes, &reqs, &mut w)?;
        let mut root = rendered6(&d.root).properties.clone();
        for o in d.root_outcomes.iter().filter(|o| !o.holds) {
            let q = self.k.root.requires.get(o.requirement).ok_or_else(|| bug(&self.k, format!("the entry has no requirement {}", o.requirement)))?;
            let source = Some((self.k.root.data_type, root.clone()));
            root.insert(q.property.clone(), o.against.clone());
            w.nodes += self.costs.charge(&self.k, &crate::futhark_cost::Charge6 { adapter: o.adapter.clone(), operand: None, fitted: Some((self.k.root.data_type, root.clone())), source }, sizes)?;
        }
        Ok(w)
    }

    fn walk(&self, n: &Node, mult: u128, sizes: &BTreeMap<String, usize>, reqs: &[&Requirement], w: &mut Work) -> Result<(), String> {
        let n = rendered6(n);
        if n.substituted.is_some() {
            w.nodes += mult;
            return Ok(());
        }
        if self.k.data_type(n.data_type).is_some_and(|d| d.name == "size") {
            return Ok(());
        }
        let op = self.ops[n.symbol as usize];
        let cells = |ty: Ty, props: &BTreeMap<String, PropertyExpr>| crate::futhark_cost::cells6(&self.k, ty, props, sizes);
        let mut ops: Vec<BTreeMap<String, PropertyExpr>> = n.kids.iter().map(|c| rendered6(c).properties.clone()).collect();
        let trips = trips6(&self.k, n, sizes)?;
        let mut adapters = 0u128;
        for (body, by) in [(false, 1), (true, trips.unwrap_or(1))] {
            for c in adapters6(&self.k, reqs, n, body, &mut ops)? {
                adapters += self.costs.charge(&self.k, &c, sizes)? * by;
            }
        }
        let kid_ty = |i: usize| n.kids.get(i).map(|c| c.data_type).ok_or_else(|| bug(&self.k, format!("a node of {} has no argument {i}", self.k.symbols[n.symbol as usize].name)));
        let own = match op {
            Op6::Replicate(_) | Op6::Hist(..) | Op6::HistRows(_) | Op6::Transpose | Op6::MapRows | Op6::Map2Rows => 1 + cells(n.data_type, &n.properties)?,
            Op6::Matmul => {
                let inner = ops[0].get("cols").ok_or_else(|| bug(&self.k, "a product of no columns".into()))?;
                1 + cells(n.data_type, &n.properties)? * self.size(inner, sizes)? as u128
            }
            _ => 1,
        };
        w.nodes += mult * (own + adapters);
        let costed = match op {
            Op6::Fold(_) | Op6::Map2(_) | Op6::Scan(_) | Op6::RowDiv => Some(0),
            Op6::Hist(..) => Some(1),
            Op6::HistRows(_) => Some(3),
            _ => None,
        };
        if let Some(i) = costed {
            w.nodes += mult * cells(kid_ty(i)?, &ops[i])?;
        }
        if let (Op6::Loop { arity, .. }, Some(t)) = (op, trips) {
            w.state += mult * t * self.state_cells(n, arity, sizes)?;
        }
        let trip = self.k.symbols[n.symbol as usize].trip;
        for (i, c) in n.kids.iter().enumerate() {
            let m = if trip == Some(i as u32) { mult * trips.unwrap_or(1) } else { mult };
            self.walk(c, m, sizes, reqs, w)?;
        }
        Ok(())
    }

    /// The cells a loop's state is written a trip (master 5's rule): 0 when the body is the loop's own state, or the
    /// start reading none of the loop's variables (Futhark removes the loop); else its arrays' cells, a component
    /// that an operation writing a fresh array computes (or a fitted body) excepted.
    fn state_cells(&self, n: &Node, arity: usize, sizes: &BTreeMap<String, usize>) -> Result<u128, String> {
        let [_, start, body] = n.kids.as_slice() else { return Err(bug(&self.k, "a loop without a size, a start and a body".into())) };
        let depth = body.environment.depth;
        fn same(a: &Node, b: &Node) -> bool {
            a.symbol == b.symbol && a.choice == b.choice && a.substituted == b.substituted && a.kids.len() == b.kids.len() && a.kids.iter().zip(&b.kids).all(|(x, y)| same(x, y))
        }
        fn reads(t: &Node, depth: u32) -> bool {
            matches!(t.choice, Some(Choice::Variable(b)) if b.depth == depth) || t.kids.iter().any(|c| reads(c, depth))
        }
        let own_state = arity == 1 && self.ops[body.symbol as usize] == Op6::Var && body.substituted.is_none() && body.choice == Some(Choice::Variable(Binder { depth, ordinal: 0 }));
        if own_state || (same(body, start) && !reads(body, depth)) {
            return Ok(0);
        }
        let fitted = n.outcomes.iter().any(|o| !o.holds && o.adapter == "fit.state");
        let fresh = |b: &Node| -> bool {
            let b = rendered6(b);
            b.substituted.is_none()
                && matches!(
                    self.ops[b.symbol as usize],
                    Op6::Map2(_) | Op6::Scan(_) | Op6::Hist(..) | Op6::HistRows(_) | Op6::Replicate(_) | Op6::Tab(_) | Op6::TabRows | Op6::RowDiv | Op6::Transpose | Op6::Matmul | Op6::MapRows | Op6::Map2Rows
                )
        };
        if fitted || fresh(body) {
            return Ok(0);
        }
        let d = self.k.data_type(n.data_type).ok_or_else(|| bug(&self.k, "a loop of no data type".into()))?;
        match &d.kind {
            DataKind::Array { .. } => crate::futhark_cost::cells6(&self.k, n.data_type, &n.properties, sizes),
            _ => Ok(0),
        }
    }

    /// THE STATIC COST of a tree at `sizes` (`crate::futhark_cost::program_cost6`): traffic (adapters included),
    /// allocations and serial depth.
    pub(super) fn static_cost(&self, tree: &Tree, sizes: &BTreeMap<String, usize>) -> Result<Score, String> {
        program_cost6(&self.k, &self.costs, &self.decode(tree)?, sizes)
    }

    /// The signature's binder depth (the layout bound's exponent).
    pub(super) fn depth(&self) -> u32 {
        self.sig.depth
    }
}

/// One evaluation: the interpreter, the case's sizes and the variables in scope by binding depth and ordinal.
struct Run<'a> {
    me: &'a Interp6,
    sizes: &'a BTreeMap<String, usize>,
    vars: Vec<Vec<Option<Val>>>,
    /// Each node's expected shape at these sizes, by the node's address, computed at its first visit (a node inside a
    /// binder's body is visited once a trip).
    expect: std::collections::HashMap<usize, Expect6>,
    /// Whether a division by a zero dividend gives the dividend ([`Interp6::eval_zero_dividends_folded`]).
    fold: bool,
}

/// A node's expected shape at a case's sizes: an array's element and lengths (`None`: `?`), an index's bound, a size's
/// value, a scalar's element, a tuple's components, a function's parameter count.
#[derive(Debug, Clone)]
enum Expect6 {
    Array(Elem, Vec<Option<i64>>),
    Index(Option<i64>),
    Size(Option<i64>),
    Scalar(Elem),
    Tuple(Vec<Ty>),
    Function(usize),
}

impl Run<'_> {
    fn bug(&self, what: String) -> String {
        bug(&self.me.k, what)
    }

    fn size(&self, e: &PropertyExpr) -> Result<i64, String> {
        self.me.size(e, self.sizes)
    }

    /// Set a variable, giving back what its slot held (none at the top).
    fn set(&mut self, b: Binder, v: Option<Val>) -> Option<Val> {
        let (d, o) = (b.depth as usize, b.ordinal as usize);
        if self.vars.len() <= d {
            self.vars.resize(d + 1, Vec::new());
        }
        if self.vars[d].len() <= o {
            self.vars[d].resize(o + 1, None);
        }
        std::mem::replace(&mut self.vars[d][o], v)
    }

    fn var(&self, b: Binder) -> Result<Val, String> {
        self.vars.get(b.depth as usize).and_then(|d| d.get(b.ordinal as usize)).cloned().flatten().ok_or_else(|| self.bug(format!("the variable {b:?} is read unbound")))
    }

    fn at(&mut self, n: &Node) -> Result<Val, String> {
        let v = self.value(n)?;
        let key = n as *const Node as usize;
        if !self.expect.contains_key(&key) {
            let e = self.me.expect(n.data_type, &n.properties, self.sizes)?;
            self.expect.insert(key, e);
        }
        let fine = match (&self.expect[&key], &v) {
            (Expect6::Array(elem, dims), Val::A(a)) => a.elem == *elem && a.dims.len() == dims.len() && dims.iter().zip(&a.dims).all(|(want, got)| want.is_none_or(|w| w == *got as i64)),
            (Expect6::Index(bound), Val::I(i)) => *i >= 0 && bound.is_none_or(|b| *i < b),
            (Expect6::Size(value), Val::I(i)) => value.is_none_or(|x| x == *i),
            (Expect6::Scalar(e), Val::S(x)) => x.elem() == *e,
            (Expect6::Tuple(components), Val::Tup(parts)) => {
                if parts.len() != components.len() {
                    false
                } else {
                    for (c, p) in components.iter().zip(parts) {
                        self.me.check_of(*c, &BTreeMap::new(), p, self.sizes)?;
                    }
                    true
                }
            }
            (Expect6::Function(1), Val::F(_)) | (Expect6::Function(2), Val::F2(_)) => true,
            _ => false,
        };
        if fine {
            Ok(v)
        } else {
            Err(self.bug(format!("a value of {:?} with {:?} is {v:?}: the program reads out of bounds or the sizes disagree", n.data_type, n.properties)))
        }
    }

    /// One `f32` division: IEEE's, or with the zero-dividend fold the dividend when it is a zero.
    fn divide(&self, x: f32, y: f32) -> f32 {
        if self.fold && x == 0.0 {
            x
        } else {
            x / y
        }
    }

    fn index(&self, v: &Val) -> Result<i64, String> {
        match v {
            Val::I(i) => Ok(*i),
            other => Err(self.bug(format!("an index or a size holds {other:?}"))),
        }
    }

    fn scalar(&self, v: &Val) -> Result<Num, String> {
        match v {
            Val::S(x) => Ok(*x),
            other => Err(self.bug(format!("a scalar holds {other:?}"))),
        }
    }

    fn array<'v>(&self, v: &'v Val) -> Result<&'v View, String> {
        match v {
            Val::A(a) => Ok(a),
            other => Err(self.bug(format!("an array holds {other:?}"))),
        }
    }

    /// The failing requirements of a node whose adapter runs at the body (`body`) or at the node, applied to its
    /// operands in the renderer's order (the others first, `clampix` last).
    fn adapt(&self, n: &Node, body: bool, ops: &mut [Val]) -> Result<(), String> {
        let k = &self.me.k;
        let at_body = |name: &str| k.adapters.iter().find(|a| a.name == name).map(|a| a.runs_at == AdapterSite::Body);
        let failing: Vec<&Outcome> = n.outcomes.iter().filter(|o| !o.holds && o.adapter != "clampix").chain(n.outcomes.iter().filter(|o| !o.holds && o.adapter == "clampix")).collect();
        for o in failing {
            if at_body(&o.adapter).ok_or_else(|| self.bug(format!("the adapter {} is not in the master", o.adapter)))? == body {
                self.adapt_one(o, ops)?;
            }
        }
        Ok(())
    }

    fn adapt_one(&self, o: &Outcome, ops: &mut [Val]) -> Result<(), String> {
        let k = &self.me.k;
        let q = self.me.reqs.get(o.requirement).ok_or_else(|| self.bug(format!("the outcome names requirement {}", o.requirement)))?;
        let Subject::Input { position, property } = &q.subject else { return Err(self.bug(format!("the work bound fails at an unsubstituted node ({})", o.adapter))) };
        let (s, ps) = (*position as usize, property.as_str());
        let against = match &q.against {
            PropertyExpr::Input { position, property } => Some((*position as usize, property.as_str())),
            _ => None,
        };
        if s >= ops.len() || against.is_some_and(|(a, _)| a >= ops.len()) {
            return Err(self.bug(format!("the requirement {} reads an input past the node's {}", q.subject, ops.len())));
        }
        let known = |e: &PropertyExpr| size6(e, self.sizes);
        match o.adapter.as_str() {
            "clampix" => {
                let (a, pa) = against.ok_or_else(|| self.bug("clampix's requirement names no array".into()))?;
                let len = extent(k, &ops[a], pa)?;
                let i = self.index(&ops[s])?;
                ops[s] = Val::I(i.min(len - 1).max(0));
            }
            "fit" | "fit.body" | "fit.state" => {
                let target = match (known(&o.against)?, against) {
                    (Some(t), _) => t,
                    (None, Some((j, pj))) => extent(k, &ops[j], pj)?,
                    (None, None) => return Err(self.bug(format!("{} fits to {}, which has no value", o.adapter, o.against))),
                };
                ops[s] = fit(k, &ops[s], ps, target)?;
            }
            "fit.pair" => {
                let (j, pj) = against.ok_or_else(|| self.bug("fit.pair's requirement names no second input".into()))?;
                let t = extent(k, &ops[j], pj)?.min(extent(k, &ops[s], ps)?);
                ops[j] = fit(k, &ops[j], pj, t)?;
                ops[s] = fit(k, &ops[s], ps, t)?;
            }
            "fit.inner" => {
                let mut sides = vec![(s, ps)];
                if let Some(a) = against {
                    sides.push(a);
                }
                let subject = extent(k, &ops[s], ps)?;
                let other = match against {
                    Some((a, pa)) => extent(k, &ops[a], pa)?,
                    None => known(&o.against)?.ok_or_else(|| self.bug(format!("fit.inner fits to {}, which has no value", o.against)))?,
                };
                let cols: Vec<usize> = (0..sides.len()).filter(|i| sides[*i].1 == "cols").collect();
                let t = match cols.as_slice() {
                    [_, _] => subject.min(other),
                    [0] => other,
                    [1] => subject,
                    _ => return Err(self.bug(format!("fit.inner's requirement {} names no inner dimension", q.subject))),
                };
                for i in cols {
                    let x = sides[i].0;
                    ops[x] = fit(k, &ops[x], "cols", t)?;
                }
            }
            other => return Err(self.bug(format!("the adapter {other} renders at no node"))),
        }
        Ok(())
    }

    fn value(&mut self, n: &Node) -> Result<Val, String> {
        if let Some(s) = n.substituted {
            return match s.by {
                Substitution::Input(p) => {
                    let kid = n.kids.get(p as usize).ok_or_else(|| self.bug(format!("the substitution reads input {p}")))?;
                    self.at(kid)
                }
                Substitution::Constant => self.me.constant(n.data_type),
            };
        }
        let me = self.me;
        let op = me.ops[n.symbol as usize];
        let kid = |i: usize| n.kids.get(i).ok_or_else(|| bug(&me.k, format!("{} lacks argument {i}", me.k.symbols[n.symbol as usize].name)));
        match op {
            Op6::Var => {
                return match n.choice {
                    Some(Choice::Variable(b)) => self.var(b),
                    Some(Choice::Constant) => self.me.constant(n.data_type),
                    other => Err(self.bug(format!("a variable reads as {other:?}"))),
                }
            }
            Op6::Literal => {
                let Some(Choice::Dc(i)) = n.choice else { return Err(self.bug(format!("a literal reads as {:?}", n.choice))) };
                let pool = &self.me.literals[n.symbol as usize];
                return pool.get(i as usize).map(|x| Val::S(*x)).ok_or_else(|| self.bug(format!("the literal {i} is past its pool of {}", pool.len())));
            }
            Op6::Constant => return self.me.constant(n.data_type),
            _ => {}
        }
        let binder = matches!(op, Op6::Tab(_) | Op6::TabRows | Op6::Loop { .. } | Op6::Let | Op6::Sel | Op6::PSel);
        if binder {
            // A binder's node adapters would run before its body: none of master 6's rows has one.
            if n.outcomes.iter().any(|o| !o.holds && self.me.k.adapters.iter().any(|a| a.name == o.adapter && a.runs_at != AdapterSite::Body)) {
                return Err(self.bug(format!("the binder {} renders a node adapter", self.me.k.symbols[n.symbol as usize].name)));
            }
        }
        match op {
            Op6::Sel => {
                // Futhark evaluates only the branch taken.
                let c = self.at(kid(0)?)?;
                let c = self.scalar(&c)?;
                let v = self.at(if c.is_negative() { kid(1)? } else { kid(2)? })?;
                Ok(Val::S(self.scalar(&v)?))
            }
            Op6::PSel => {
                let c = self.at(kid(0)?)?;
                let c = self.scalar(&c)?;
                self.at(if c.is_negative() { kid(1)? } else { kid(2)? })
            }
            Op6::Tab(elem) => {
                let size = self.at(kid(0)?)?;
                let size = self.index(&size)?;
                let body = kid(1)?;
                let b = Binder { depth: body.environment.depth, ordinal: 0 };
                let outer = self.set(b, None);
                let mut out = Vec::with_capacity(size.max(0) as usize);
                for j in 0..size {
                    self.set(b, Some(Val::I(j)));
                    let v = self.at(body)?;
                    let mut ops = [Val::I(size), v];
                    self.adapt(n, true, &mut ops)?;
                    out.push(self.scalar(&ops[1])?.word());
                }
                self.set(b, outer);
                Ok(Val::A(View::of(out, vec![size as usize], elem)))
            }
            Op6::TabRows => {
                let rows = self.at(kid(0)?)?;
                let rows = self.index(&rows)?;
                let cols = self.at(kid(1)?)?;
                let cols = self.index(&cols)?;
                let body = kid(2)?;
                let b = Binder { depth: body.environment.depth, ordinal: 0 };
                let outer = self.set(b, None);
                let (mut out, mut shape) = (Vec::new(), None);
                for i in 0..rows {
                    self.set(b, Some(Val::I(i)));
                    let v = self.at(body)?;
                    let mut ops = [Val::I(rows), Val::I(cols), v];
                    self.adapt(n, true, &mut ops)?;
                    let row = self.array(&ops[2])?;
                    if row.dims.len() != 1 || shape.is_some_and(|(c, e)| c != row.dims[0] || e != row.elem) {
                        return Err(self.bug(format!("a tabulate of rows whose row {i} is {:?}", row.dims)));
                    }
                    shape = Some((row.dims[0], row.elem));
                    out.extend_from_slice(row.slice());
                }
                self.set(b, outer);
                let (c, elem) = shape.ok_or_else(|| self.bug("a tabulate of no rows".into()))?;
                Ok(Val::A(View::of(out, vec![rows as usize, c], elem)))
            }
            Op6::Loop { from1, arity } => {
                let size = self.at(kid(0)?)?;
                let size = self.index(&size)?;
                let start = self.at(kid(1)?)?;
                let body = kid(2)?;
                let depth = body.environment.depth;
                let slot = |o: usize| Binder { depth, ordinal: o as u32 };
                let outer: Vec<Option<Val>> = (0..=arity).map(|o| self.set(slot(o), None)).collect();
                let trips = if from1 { (size - 1).max(0) } else { size };
                let mut state = start.clone();
                for t in 0..trips {
                    match (arity, state) {
                        (1, s) => {
                            self.set(slot(0), Some(s));
                        }
                        (_, Val::Tup(parts)) if parts.len() == arity => {
                            for (o, p) in parts.into_iter().enumerate() {
                                self.set(slot(o), Some(p));
                            }
                        }
                        (_, other) => return Err(self.bug(format!("a loop's state of {arity} components is {other:?}"))),
                    }
                    self.set(slot(arity), Some(Val::I(t + i64::from(from1))));
                    let v = self.at(body)?;
                    let mut ops = [Val::I(size), start.clone(), v];
                    self.adapt(n, true, &mut ops)?;
                    let [_, _, next] = ops;
                    state = next;
                }
                for (o, v) in outer.into_iter().enumerate() {
                    self.set(slot(o), v);
                }
                Ok(state)
            }
            Op6::Let => {
                let v = self.at(kid(0)?)?;
                let body = kid(1)?;
                let b = Binder { depth: body.environment.depth, ordinal: 0 };
                let outer = self.set(b, Some(v));
                let out = self.at(body);
                self.set(b, outer);
                out
            }
            _ => {
                let mut ops = Vec::with_capacity(n.kids.len());
                for c in &n.kids {
                    ops.push(self.at(c)?);
                }
                self.adapt(n, false, &mut ops)?;
                self.apply(n, op, ops)
            }
        }
    }

    /// A node that is no binder, no leaf and no selection, on its adapted operands.
    fn apply(&self, n: &Node, op: Op6, ops: Vec<Val>) -> Result<Val, String> {
        let arg = |i: usize| ops.get(i).ok_or_else(|| self.bug(format!("{} lacks argument {i}", self.me.k.symbols[n.symbol as usize].name)));
        let s = |i: usize| -> Result<Num, String> { self.scalar(arg(i)?) };
        let ix = |i: usize| -> Result<i64, String> { self.index(arg(i)?) };
        let a = |i: usize| -> Result<&View, String> { self.array(arg(i)?) };
        Ok(match op {
            Op6::Bin(b) => Val::S(s(0)?.bin(b, s(1)?)?),
            Op6::Neg => Val::S(s(0)?.neg()),
            Op6::Abs => Val::S(s(0)?.abs()),
            Op6::Div => Val::S(Num::F(self.divide(s(0)?.f32_only("div")?, s(1)?.f32_only("div")?))),
            Op6::Sqrt => Val::S(Num::F(s(0)?.f32_only("sqrt")?.sqrt())),
            Op6::Get => {
                let (v, i) = (a(0)?, ix(1)?);
                if v.dims.len() != 1 || i < 0 || i as usize >= v.dims[0] {
                    return Err(self.bug(format!("the index {i} is out of range for {:?}", v.dims)));
                }
                Val::S(v.num(v.data[v.off + i as usize]))
            }
            Op6::Fold(b) => {
                let v = a(0)?;
                let mut acc = Num::neutral(v.elem, b);
                for w in v.slice() {
                    acc = acc.bin(b, v.num(*w))?;
                }
                Val::S(acc)
            }
            Op6::Ival(elem) | Op6::Sval(elem) => Val::S(Num::of_i64(elem, ix(0)?)),
            Op6::Mod => {
                let size = i32::try_from(ix(0)?).map_err(|_| self.bug("a modulus past i32".into()))?;
                match s(1)? {
                    Num::I(x) if size > 0 => Val::S(Num::I(x.rem_euclid(size))),
                    other => return Err(self.bug(format!("mod {size} of {other:?}"))),
                }
            }
            Op6::Prev => Val::I((ix(0)? - 1).max(0)),
            Op6::Next => {
                let bound = n.kids.first().and_then(|c| c.properties.get("bound")).ok_or_else(|| self.bug("next of an index with no bound".into()))?;
                Val::I((ix(0)? + 1).min(self.size(bound)? - 1))
            }
            Op6::Clamp => {
                let size = ix(0)?;
                if size < 1 {
                    return Err(self.bug(format!("a clamp below the size {size}")));
                }
                Val::I(s(1)?.clamp(size as usize))
            }
            Op6::ZeroIndex => Val::I(0),
            Op6::Last => Val::I(ix(0)? - 1),
            Op6::Replicate(elem) => {
                let (m, x) = (ix(0)?, s(1)?.word());
                Val::A(View::of(vec![x; m as usize], vec![m as usize], elem))
            }
            Op6::Map2(b) => {
                let (x, y) = (a(0)?, a(1)?);
                if x.dims != y.dims || x.dims.len() != 1 || x.elem != y.elem {
                    return Err(self.bug(format!("map2 over {:?} and {:?}", x.dims, y.dims)));
                }
                let out: Vec<u32> = x.slice().iter().zip(y.slice()).map(|(p, q)| x.num(*p).bin(b, y.num(*q)).map(Num::word)).collect::<Result<_, _>>()?;
                Val::A(View::of(out, x.dims.clone(), x.elem))
            }
            Op6::Scan(b) => {
                let x = a(0)?;
                let mut acc = Num::neutral(x.elem, b);
                let mut out = Vec::with_capacity(x.len());
                for w in x.slice() {
                    acc = acc.bin(b, x.num(*w))?;
                    out.push(acc.word());
                }
                Val::A(View::of(out, x.dims.clone(), x.elem))
            }
            Op6::Hist(b, elem) => {
                let (m, is, vs) = (ix(0)? as usize, a(1)?, a(2)?);
                if is.dims != vs.dims || is.dims.len() != 1 || vs.elem != elem {
                    return Err(self.bug(format!("a histogram of {:?} at {:?}", vs.dims, is.dims)));
                }
                let mut out = vec![Num::neutral(elem, b); m];
                for (i, v) in is.slice().iter().zip(vs.slice()) {
                    if let Some(i) = is.num(*i).hist_index(m) {
                        out[i] = out[i].bin(b, vs.num(*v))?;
                    }
                }
                Val::A(View::of(out.into_iter().map(Num::word).collect(), vec![m], elem))
            }
            Op6::HistRows(elem) => {
                let (kk, d, is, vs) = (ix(0)? as usize, ix(1)? as usize, a(2)?, a(3)?);
                if is.dims.len() != 1 || vs.dims.len() != 2 || vs.dims[0] != is.dims[0] || vs.dims[1] != d || vs.elem != elem {
                    return Err(self.bug(format!("a histogram of rows {:?} at {:?} into [{kk}][{d}]", vs.dims, is.dims)));
                }
                let mut out = vec![Num::zero(elem); kk * d];
                let rows = vs.slice();
                for (i, idx) in is.slice().iter().enumerate() {
                    if let Some(bucket) = is.num(*idx).hist_index(kk) {
                        for e in 0..d {
                            out[bucket * d + e] = out[bucket * d + e].bin(Bin::Add, vs.num(rows[i * d + e]))?;
                        }
                    }
                }
                Val::A(View::of(out.into_iter().map(Num::word).collect(), vec![kk, d], elem))
            }
            Op6::RowDiv => {
                let (m, sc) = (a(0)?, a(1)?);
                if m.dims.len() != 2 || sc.dims.len() != 1 || sc.dims[0] != m.dims[0] {
                    return Err(self.bug(format!("the rows of {:?} divided by {:?}", m.dims, sc.dims)));
                }
                let d = m.dims[1];
                let mut out = Vec::with_capacity(m.len());
                for (r, sw) in sc.slice().iter().enumerate() {
                    let div = sc.num(*sw).f32_only("rowdiv")?;
                    for w in &m.slice()[r * d..(r + 1) * d] {
                        out.push(self.divide(m.num(*w).f32_only("rowdiv")?, div).to_bits());
                    }
                }
                Val::A(View::of(out, m.dims.clone(), m.elem))
            }
            Op6::Row => {
                let (m, i) = (a(0)?, ix(1)?);
                if m.dims.len() != 2 || i < 0 || i as usize >= m.dims[0] {
                    return Err(self.bug(format!("the row {i} is out of range for {:?}", m.dims)));
                }
                let dims = m.dims[1..].to_vec();
                Val::A(View { data: m.data.clone(), off: m.off + i as usize * dims[0], dims, elem: m.elem })
            }
            Op6::RowFn(f) => Val::F(f),
            Op6::Row2Fn(f) => Val::F2(f),
            Op6::Transpose => {
                let m = a(0)?;
                let [rows, cols] = m.dims[..] else { return Err(self.bug(format!("a transpose of {:?}", m.dims))) };
                let src = m.slice();
                let mut out = vec![0u32; rows * cols];
                for i in 0..rows {
                    for j in 0..cols {
                        out[j * rows + i] = src[i * cols + j];
                    }
                }
                Val::A(View::of(out, vec![cols, rows], m.elem))
            }
            Op6::Matmul => {
                // The helper's arithmetic: `acc + a * b` in f32 from 0.0, in order (master 5's, measured bit-identical).
                let (x, y) = (a(0)?, a(1)?);
                let ([nn, mm], [m2, p]) = (&x.dims[..], &y.dims[..]) else { return Err(self.bug(format!("a matmul of {:?} and {:?}", x.dims, y.dims))) };
                if mm != m2 || x.elem != Elem::F32 || y.elem != Elem::F32 {
                    return Err(self.bug(format!("a matmul whose inner dimensions are {mm} and {m2}")));
                }
                let (nn, mm, p) = (*nn, *mm, *p);
                let (sa, sb) = (x.slice(), y.slice());
                let mut out = Vec::with_capacity(nn * p);
                for i in 0..nn {
                    for j in 0..p {
                        let mut acc = 0.0f32;
                        for kk in 0..mm {
                            acc += f32::from_bits(sa[i * mm + kk]) * f32::from_bits(sb[kk * p + j]);
                        }
                        out.push(acc.to_bits());
                    }
                }
                Val::A(View::of(out, vec![nn, p], Elem::F32))
            }
            Op6::MapRows => {
                let Val::F(f) = arg(0)? else { return Err(self.bug(format!("a row function slot holds {:?}", arg(0)?))) };
                let m = a(1)?;
                let [rows, cols] = m.dims[..] else { return Err(self.bug(format!("a map over the rows of {:?}", m.dims))) };
                let src = m.slice();
                let mut out = Vec::with_capacity(rows * cols);
                for i in 0..rows {
                    let row: Vec<f32> = src[i * cols..(i + 1) * cols].iter().map(|w| f32::from_bits(*w)).collect();
                    out.extend(f.apply(&row).into_iter().map(f32::to_bits));
                }
                Val::A(View::of(out, vec![rows, cols], Elem::F32))
            }
            Op6::Map2Rows => {
                let Val::F2(f) = arg(0)? else { return Err(self.bug(format!("a binary row function slot holds {:?}", arg(0)?))) };
                let (x, y) = (a(1)?, a(2)?);
                if x.dims != y.dims || x.dims.len() != 2 {
                    return Err(self.bug(format!("a map2 over the rows of {:?} and {:?}", x.dims, y.dims)));
                }
                let (rows, cols) = (x.dims[0], x.dims[1]);
                let (sa, sb) = (x.slice(), y.slice());
                let mut out = Vec::with_capacity(rows * cols);
                for i in 0..rows {
                    let u: Vec<f32> = sa[i * cols..(i + 1) * cols].iter().map(|w| f32::from_bits(*w)).collect();
                    let v: Vec<f32> = sb[i * cols..(i + 1) * cols].iter().map(|w| f32::from_bits(*w)).collect();
                    out.extend(f.apply(&u, &v)?.into_iter().map(f32::to_bits));
                }
                Val::A(View::of(out, vec![rows, cols], Elem::F32))
            }
            Op6::Tuple => Val::Tup(ops),
            Op6::Proj(i) => match ops.into_iter().next() {
                Some(Val::Tup(mut parts)) if i < parts.len() => parts.swap_remove(i),
                other => return Err(self.bug(format!("the projection {i} of {other:?}"))),
            },
            Op6::Cvt(into) => Val::S(s(0)?.convert(into)),
            Op6::Var | Op6::Literal | Op6::Constant | Op6::Sel | Op6::PSel | Op6::Tab(_) | Op6::TabRows | Op6::Loop { .. } | Op6::Let => {
                return Err(self.bug(format!("{op:?} applied as an operation")))
            }
        })
    }
}
