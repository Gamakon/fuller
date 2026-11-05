//! THE TSR KINGDOM AS ROWS of the gene frame's fixed schema (design `docs/DESIGN_futhark_types_and_properties.md`
//! 3.3 and 8.3; plan `docs/PLAN_futhark_kingdom.md` section 8). TSR is Typed Symbolic Regression.
//!
//! Each variant's `kingdoms/tsr/<variant>/tables.tsv`, written by `kingdoms/tsr/gen_tables.py` (never by hand; its
//! `--check` is a test), holds the four kinds of row and nothing else:
//!
//! - ONE data type `F`, carrying `depth`, `order`, `fam`, `arm` and `ties`;
//! - Argument rows: the 18 operators of `SymbolTable::wide()`, `F` in and out, then the input terminals `x0..x9` and
//!   the constant `?` (role constant: a Dc word), in `wide()`'s order;
//! - up Flow rows: depth is the greatest input depth, plus 1 for a raising unary; stream order is Horton-Strahler order
//!   over flattened sum and product chains, through `fam`, `arm` and `ties` (`out.p` read acyclically);
//! - `must` Guard rows, the variant's only difference: `out.order <= 3` on the four binaries (order), `in0.depth <= 1`
//!   on the ten raising unaries (depth), `in0.order <= 2` on the seven windowed unaries (windows); each repaired by
//!   `{in0}`, the node rendered as its first input. `single` has no guard.
//!
//! The gene is the single kingdom's word for word: one type, no Dv lane (the inputs are terminals, not variables), 104
//! words at head 34. A law of `n` columns keeps the first `n` input terminals ([`TsrRows::load`]), so its symbol ids
//! are `wide(n)`'s. An illegal shape stays representable in the gene and is INEXPRESSIBLE in the program: the decoder
//! ([`crate::typed_environment::Decoder`]) substitutes, bottom-up, and [`program`] is the tree that is scored.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::TsrVariant;
use crate::geneframe::{Entry, KingdomTables, Role, Ty};
use crate::kingdom_contract::{relations_from_tables, KingdomContract, SymbolId};
use crate::typed_environment::{Choice, Decoder, Decoding, Node, Substituted, Substitution};
use crate::typed_gep::{GeneLayout, Tree};

/// The input terminals the tables hold; a law keeps the first `n`.
pub const MAX_INPUTS: usize = 10;

/// The single kingdom's head (`Config::head`; program size 17), which gives the 104-word gene.
pub const HEAD: usize = 34;

impl TsrVariant {
    /// The variant's four kinds of row (`kingdoms/tsr/<variant>/tables.tsv`).
    pub fn rows_tsv(self) -> &'static str {
        match self {
            TsrVariant::Single => include_str!("../../kingdoms/tsr/single/tables.tsv"),
            TsrVariant::Order => include_str!("../../kingdoms/tsr/order/tables.tsv"),
            TsrVariant::Depth => include_str!("../../kingdoms/tsr/depth/tables.tsv"),
            TsrVariant::OrderDepth => include_str!("../../kingdoms/tsr/order_depth/tables.tsv"),
            TsrVariant::OrderDepthWindows => include_str!("../../kingdoms/tsr/order_depth_windows/tables.tsv"),
        }
    }
}

/// What a symbol of the alphabet is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsrSymbol<'a> {
    /// An operator, by its `Math` constructor (`Add`, `ProtectedDiv`, `Sin`).
    Operator(&'a str),
    /// Input column `k`.
    Input(usize),
    /// The constant `?`.
    Constant,
}

/// A loaded TSR variant: its rows, entry, contract and decoder, built once.
#[derive(Debug, Clone)]
pub struct TsrRows {
    pub variant: TsrVariant,
    pub n_inputs: usize,
    pub tables: Arc<KingdomTables>,
    pub entry: Entry,
    pub contract: Arc<KingdomContract>,
    pub decoder: Decoder,
}

fn bug(what: String) -> String {
    format!("our software has a bug: {what}")
}

/// The input column a symbol name `x<k>` names.
fn input_column(name: &str) -> Option<usize> {
    name.strip_prefix('x').filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())).and_then(|d| d.parse().ok())
}

impl TsrRows {
    /// Load `variant` for a law of `n_inputs` columns (1 to [`MAX_INPUTS`]): the rows naming an input terminal
    /// `x<k>`, `k >= n_inputs`, are left out (a view of the rows, as a Futhark problem's is of its master), then the
    /// tables are read and checked, the contract built and checked, the decoder built and the entry checked.
    pub fn load(variant: TsrVariant, n_inputs: usize) -> Result<TsrRows, String> {
        if !(1..=MAX_INPUTS).contains(&n_inputs) {
            return Err(bug(format!("TSR {}: a law of {n_inputs} input columns; the rows hold 1 to {MAX_INPUTS}", variant.dir())));
        }
        let kept: Vec<&str> = variant.rows_tsv().lines().filter(|l| l.split('\t').nth(1).and_then(input_column).is_none_or(|k| k < n_inputs)).collect();
        let kingdom = format!("{}/{}", super::TSR, variant.dir());
        let ty = |i: usize| if i == 0 { Ok(Ty::F) } else { Err(bug(format!("{kingdom}: data type {i}; TSR has one, F"))) };
        let (tables, entry) = KingdomTables::parse_named(&kingdom, &kept.join("\n"), &ty)?;
        tables.check().map_err(|d| d.join("\n"))?;
        let contract = relations_from_tables(&tables, vec![entry.output]).check().map_err(|d| d.join("\n"))?;
        let (tables, contract) = (Arc::new(tables), Arc::new(contract));
        let decoder = Decoder::from_arcs(Arc::clone(&tables), Arc::clone(&contract)).map_err(|d| d.join("\n"))?;
        decoder.check_entry(&entry).map_err(|d| d.join("\n"))?;
        Ok(TsrRows { variant, n_inputs, tables, entry, contract, decoder })
    }

    /// What symbol `s` is.
    pub fn symbol(&self, s: SymbolId) -> Option<TsrSymbol<'_>> {
        let row = self.tables.symbol(i64::from(s))?;
        Some(match (row.role, input_column(&row.symbol_name)) {
            (Role::Constant, _) => TsrSymbol::Constant,
            (Role::Function, Some(k)) => TsrSymbol::Input(k),
            (Role::Function, None) => TsrSymbol::Operator(row.alias.as_str()),
            (Role::Variable | Role::Formula, _) => return None,
        })
    }

    /// The gene's layout at head `head` (the single kingdom's: one frame, `F`; [`HEAD`] gives 104 words).
    pub fn layout(&self, head: usize) -> Result<GeneLayout, String> {
        GeneLayout::from_heads(&self.contract, Ty::F, &BTreeMap::from([(Ty::F, head)])).map_err(|e| bug(format!("TSR {}: the layout at head {head}: {e:?}", self.variant.dir())))
    }

    /// Passes 2 to 4 on an expressed tree: every node's properties, and where a `must` guard fails, its substitution.
    pub fn decode(&self, tree: &Tree) -> Result<Decoding, String> {
        self.decoder.decode(&self.entry, tree)
    }
}

/// THE PROGRAM a decoded gene expresses, the tree that is scored: every substituted node replaced by the input it
/// renders as (a TSR guard's only repair is `{in0}`, so never a constant). A node of the program is a node of the
/// expressed tree, with its symbol and its word.
pub fn program(d: &Decoding) -> Tree {
    fn go(n: &Node) -> Tree {
        if let Some(Substituted { by: Substitution::Input(k), .. }) = n.substituted {
            return go(&n.kids[k as usize]);
        }
        let const_index = match n.choice {
            Some(Choice::Dc(w)) => Some(w),
            _ => None,
        };
        Tree { symbol: n.symbol, const_index, kids: n.kids.iter().map(go).collect() }
    }
    go(&d.root)
}

/// How many nodes of the expressed tree a substitution drops (the program's size subtracted from the tree's).
pub fn dropped(tree: &Tree, d: &Decoding) -> usize {
    fn size(t: &Tree) -> usize {
        1 + t.kids.iter().map(size).sum::<usize>()
    }
    size(tree) - size(&program(d))
}

#[cfg(test)]
mod tests;
