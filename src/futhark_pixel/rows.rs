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

#[cfg(test)]
mod tests;
