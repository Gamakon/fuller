//! THE FUTHARK KINGDOM ON MASTER 6, configured by rows alone (design `docs/DESIGN_futhark_types_and_properties.md`
//! 3.3; Andrew Morgan, 2026-11-05: the schema is fixed, a kingdom is rows). A view's table (`sig/<problem>/m6/
//! tables.tsv`, written by `view.py` and `futhark_view::master6`) is the gene frame's four kinds of row and nothing
//! else: Argument, Flow and Guard rows with their repairs inline, the entry, the symbols with their spellings and
//! preludes, the formula rows that spell a size in Futhark, and the ADAPTERS as ordinary symbols of the alphabet that a
//! repair's template calls.
//!
//! [`Kingdom6::load`] reads it ([`KingdomTables::parse_named`]), checks it, and builds the contract and the decoder
//! once. Pass 5 is the decoder's ([`Decoder::render`]): the program's body is the tables' rendering, and this module
//! adds only what the program's SIGNATURE gives, the entry point around it ([`Kingdom6::program`]): the preludes, then
//! `entry main [dims] (size inputs) (inputs) : result =`, the size inputs rebound as `i64`, the zero arrays bound, and
//! the body. Nothing here is matched on an adapter's name: the interpreter and the cost execute the repairs' templates
//! through the decoder ([`Decoder::exec`], [`Decoder::calls`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::futhark_view::master6::ViewSpec6;
use crate::geneframe::{DataKind, DataType, Entry, KingdomTables, Role, SymbolRow, Ty};
use crate::kingdom_contract::{relations_from_tables, KingdomContract, KingdomRelations, SymbolId};
use crate::typed_environment::{Decoder, Decoding, Failure};
use crate::typed_gep::Tree;

fn bug6(kingdom: &str, what: String) -> String {
    format!("our software has a bug: {kingdom}: {what}")
}

/// A loaded master 6 kingdom: its rows, its entry, its contract and decoder (built once), and the signature its program
/// is entered by.
#[derive(Debug, Clone)]
pub struct Kingdom6 {
    pub kingdom: String,
    pub tables: Arc<KingdomTables>,
    pub entry: Entry,
    pub contract: Arc<KingdomContract>,
    pub decoder: Decoder,
    pub spec: ViewSpec6,
}

/// A rendered master 6 program: its text, the nodes it renders (a replaced node's dropped inputs not counted), every
/// adapter in it with its node's path (`None`: the entry's result) and whether evolution placed it (else a repair
/// called it), and every replacement with its node's path and the guard whose repair made it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program6 {
    pub text: String,
    pub nodes: usize,
    pub adapters: Vec<(Option<Vec<u32>>, String, bool)>,
    pub substitutions: Vec<(Vec<u32>, String)>,
}

impl Program6 {
    /// No adapter in the program (placed or called).
    pub fn adapter_free(&self) -> bool {
        self.adapters.is_empty()
    }

    /// No replacement in the program.
    pub fn substitution_free(&self) -> bool {
        self.substitutions.is_empty()
    }

    /// The adapters a repair called (evolution placed none of them).
    pub fn repairs(&self) -> usize {
        self.adapters.iter().filter(|a| !a.2).count()
    }
}

impl Kingdom6 {
    /// Load a view's table: the four kinds of row and the entry (`tables.tsv`), and its specification (the signature
    /// the program is entered by). Data type `i` is `Ty::Futhark(i)`. The tables are checked, the contract built and
    /// checked, the decoder built, and the entry checked, once.
    pub fn load(kingdom: &str, tables_tsv: &str, spec: &ViewSpec6) -> Result<Kingdom6, String> {
        let ty = |i: usize| -> Result<Ty, String> {
            Ty::check_dual_count(kingdom, i + 1, Ty::MAX_FUTHARK_DUALS)?;
            Ok(Ty::Futhark(i as u16))
        };
        let (tables, entry) = KingdomTables::parse_named(kingdom, tables_tsv, &ty)?;
        tables.check().map_err(|d| d.join("\n"))?;
        let contract = relations_from_tables(&tables, vec![entry.output]).check().map_err(|d| d.join("\n"))?;
        let (tables, contract) = (Arc::new(tables), Arc::new(contract));
        let decoder = Decoder::from_arcs(Arc::clone(&tables), Arc::clone(&contract)).map_err(|d| d.join("\n"))?;
        decoder.check_entry(&entry).map_err(|d| d.join("\n"))?;
        Ok(Kingdom6 { kingdom: kingdom.to_string(), tables, entry, contract, decoder, spec: spec.clone() })
    }

    /// The kingdom's relations: its symbols' signatures from the Argument rows, rooted at the entry's result.
    pub fn relations(&self) -> KingdomRelations {
        relations_from_tables(&self.tables, vec![self.entry.output])
    }

    /// The type of a data type by name.
    pub fn ty(&self, name: &str) -> Option<Ty> {
        self.tables.data_types.iter().find(|d| d.name == name).map(|d| d.id)
    }

    /// The data type of a type.
    pub fn data_type(&self, ty: Ty) -> Option<&DataType> {
        self.tables.data_type(ty)
    }

    /// A symbol's row (its id is its contract id).
    pub fn symbol(&self, s: SymbolId) -> Option<&SymbolRow> {
        self.tables.symbol(i64::from(s))
    }

    /// The symbols of the alphabet (the contract's), in id order: every row but the formula rows.
    pub fn symbols(&self) -> impl Iterator<Item = &SymbolRow> {
        self.tables.symbols.iter().filter(|s| s.role != Role::Formula)
    }

    /// A symbol's id by name.
    pub fn symbol_named(&self, name: &str) -> Option<SymbolId> {
        self.decoder.symbol_named(name)
    }

    /// A symbol's name.
    pub fn name(&self, s: SymbolId) -> &str {
        self.contract.name(s)
    }

    /// A symbol's master row (its alias: `M0026`), which a kingdom's evaluator reads its meaning by.
    pub fn reference(&self, s: SymbolId) -> &str {
        self.symbol(s).map_or("", |r| r.alias.as_str())
    }

    /// The literals of a constant symbol's pool (its `pool` spelling), as Futhark writes them (`0f32`, `-1i32`).
    pub fn literals(&self, s: SymbolId) -> Vec<String> {
        self.symbol(s).and_then(|r| r.spelling("pool")).map(|p| p.split(',').map(str::to_string).collect()).unwrap_or_default()
    }

    /// The Dc lengths of the gene: each constant symbol's data type and its pool's length.
    pub fn const_len(&self) -> BTreeMap<Ty, u32> {
        self.symbols().filter(|s| s.role == Role::Constant).filter_map(|s| self.tables.output(s.symbol).map(|ty| (ty, self.literals(s.symbol as SymbolId).len() as u32))).collect()
    }

    /// A symbol's inputs' data types, in order.
    pub fn inputs(&self, s: SymbolId) -> &[Ty] {
        self.contract.signature(s)
    }

    /// A symbol's output data type.
    pub fn output(&self, s: SymbolId) -> Ty {
        self.contract.output(s)
    }

    /// A symbol's role.
    pub fn role(&self, s: SymbolId) -> Role {
        self.symbol(s).map_or(Role::Function, |r| r.role)
    }

    /// The argument a symbol introduces variables into and how many it introduces there (its down `var.T` flows, one
    /// variable each), `None` for a symbol that binds nothing. A symbol that binds into two arguments is a bug.
    pub fn binds(&self, s: SymbolId) -> Result<Option<(u32, usize)>, String> {
        let mut at: Option<(u32, usize)> = None;
        for f in self.tables.flows.iter().filter(|f| f.owner == crate::geneframe::Owner::Symbol(i64::from(s)) && matches!(f.name, crate::geneframe::FlowName::Var { .. })) {
            let crate::geneframe::Place::In(k) = f.at else { return Err(bug6(&self.kingdom, format!("{} introduces variables at its result", self.name(s)))) };
            let n = match f.formula {
                crate::geneframe::PropertyExpr::Literal(n) if n >= 0 => n as usize,
                ref other => return Err(bug6(&self.kingdom, format!("{} introduces {other} variables", self.name(s)))),
            };
            at = match at {
                None => Some((k, n)),
                Some((j, m)) if j == k => Some((j, m + n)),
                Some(_) => return Err(bug6(&self.kingdom, format!("{} introduces variables into two arguments", self.name(s)))),
            };
        }
        Ok(at)
    }

    /// The argument whose subtree runs once a trip (its down flow `env.trips := env.trips + 1`), if any.
    pub fn trip(&self, s: SymbolId) -> Option<u32> {
        self.tables.flows.iter().find(|f| f.owner == crate::geneframe::Owner::Symbol(i64::from(s)) && f.name == crate::geneframe::FlowName::Env("trips".to_string())).and_then(|f| match f.at {
            crate::geneframe::Place::In(k) => Some(k),
            crate::geneframe::Place::Out => None,
        })
    }

    /// A constant symbol's literal `i` without its type's suffix (`0`, `-1`, `0.5`: the text form's spelling).
    pub fn literal_text(&self, s: SymbolId, i: usize) -> Option<String> {
        let lit = self.literals(s).get(i)?.clone();
        let suffix = self.data_type(self.output(s)).map_or("", |d| d.name.as_str());
        Some(lit.strip_suffix(suffix).unwrap_or(&lit).to_string())
    }

    /// The scalar inside an array data type (itself for a scalar).
    pub fn innermost(&self, ty: Ty) -> Ty {
        let mut t = ty;
        while let Some(DataKind::Array { element }) = self.data_type(t).map(|d| &d.kind) {
            t = *element;
        }
        t
    }

    /// The decoding of a tree (passes 2 to 4).
    pub fn decode(&self, tree: &Tree) -> Result<Decoding, String> {
        self.decoder.decode(&self.entry, tree)
    }

    /// THE PROGRAM of a tree (passes 2 to 5 and the entry point): the preludes of the symbols it uses, each followed
    /// by a blank line, then the entry point around the rendered body.
    pub fn program(&self, tree: &Tree) -> Result<Program6, String> {
        self.program_of(&self.decode(tree)?)
    }

    /// [`Kingdom6::program`] of a decoding.
    pub fn program_of(&self, d: &Decoding) -> Result<Program6, String> {
        let dc = |s: SymbolId, w: u32| -> Result<String, String> {
            let pool = self.literals(s);
            pool.get(w as usize).cloned().ok_or_else(|| bug6(&self.kingdom, format!("{}'s literal {w} is past its pool of {}", self.name(s), pool.len())))
        };
        let r = self.decoder.render(&self.entry, d, &dc)?;
        let mut text = String::new();
        for p in &r.preludes {
            text.push_str(p);
            text.push('\n');
        }
        text.push_str(&self.frame(&r.body)?);
        let adapters = self.decoder.rescues(&self.entry, d)?.into_iter().map(|x| (x.path, x.adapter, x.placed)).collect();
        let substitutions = d
            .failures()
            .into_iter()
            .filter_map(|f| match f {
                Failure::Substituted { path, requirement, .. } => Some((path, self.tables.guards[requirement].condition.to_string())),
                _ => None,
            })
            .collect();
        Ok(Program6 { text, nodes: d.rendered_nodes().len(), adapters, substitutions })
    }

    /// The Futhark type of an entry parameter or the result: its dimensions as sizes (a size given as `i32` read
    /// `i64.i32 k`), then its element.
    fn entry_type(&self, elem: &str, dims: &[String]) -> String {
        let size_inputs = self.size_inputs();
        let mut s = String::new();
        for d in dims {
            s.push_str(&if size_inputs.contains(d) { format!("[i64.i32 {d}]") } else { format!("[{d}]") });
        }
        s.push_str(elem);
        s
    }

    /// The sizes the entry takes as an `i32` parameter (the histogram's `k`), rebound as `i64`.
    pub fn size_inputs(&self) -> Vec<String> {
        self.spec.env.iter().filter(|l| l.kind == "size" && !l.elem.is_empty()).map(|l| l.name.clone()).collect()
    }

    /// THE ENTRY POINT around a rendered body, from the signature: `entry main [dims] (k: i32) (xs: [n]i32) :
    /// <result> =`, the size inputs rebound, the zero arrays bound, then `in <body>` (the body alone when nothing is
    /// bound).
    pub fn frame(&self, body: &str) -> Result<String, String> {
        let size_inputs = self.size_inputs();
        let inputs: Vec<_> = self.spec.env.iter().filter(|l| l.kind == "input").collect();
        let mut dims: Vec<String> = Vec::new();
        for l in &inputs {
            for d in &l.dims {
                if !size_inputs.contains(d) && !dims.contains(d) {
                    dims.push(d.clone());
                }
            }
        }
        let (name, elem, out_dims) = &self.spec.output;
        let mut s = format!("entry {name} {}", dims.iter().map(|d| format!("[{d}]")).collect::<String>());
        for k in &size_inputs {
            s.push_str(&format!(" ({k}: i32)"));
        }
        for l in &inputs {
            s.push_str(&format!(" ({}: {})", l.name, self.entry_type(&l.elem, &l.dims)));
        }
        s.push_str(&format!(" : {} =\n", self.entry_type(elem, out_dims)));
        let mut lets = String::new();
        for k in &size_inputs {
            lets.push_str(&format!("  let {k} = i64.i32 {k}\n"));
        }
        for l in self.spec.env.iter().filter(|l| l.kind == "zeros" && self.entry.parameters.iter().any(|p| p.name == l.name)) {
            let len = l.dims.first().ok_or_else(|| bug6(&self.kingdom, format!("the zero array {} has no length", l.name)))?;
            let ty = self.ty(&l.elem).ok_or_else(|| bug6(&self.kingdom, format!("the zero array {} is of {}, which is no data type", l.name, l.elem)))?;
            let zero = self.tables.constant_of(ty).map(|c| c.spelling.clone()).ok_or_else(|| bug6(&self.kingdom, format!("{} has no constant", l.elem)))?;
            lets.push_str(&format!("  let {} = replicate {len} {zero}\n", l.name));
        }
        s.push_str(&lets);
        s.push_str(&format!("  {}{body}\n", if lets.is_empty() { "" } else { "in " }));
        Ok(s)
    }
}
