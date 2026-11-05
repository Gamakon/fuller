//! The table-defined Futhark kingdoms' shared machinery: the rows, the loader
//! that reads a kingdom's `types.tsv` and `functions.tsv`, the renderer from an
//! expressed tree to Futhark source, the ADF rendering, and the gene RNG.
//!
//! Every Futhark kingdom is loaded through [`FutharkKingdom::from_tables`]: the
//! Futhark kingdom ([`crate::futhark`], `kingdoms/futhark/`) with its signature
//! and pixel families ([`crate::futhark_sig`], [`crate::futhark_pixel`]), and
//! the deprecated 1-D Futhark kingdom ([`crate::futhark_1d_kingdom`],
//! `kingdoms/futhark_1d/`). The [`FutharkFamily`] of a kingdom decides which
//! [`Ty`] family its duals belong to, so the kingdoms' duals cannot collide.
//!
//! The genetic representation is the Karva gene. The Futhark *source* is a
//! rendering of the expressed tree -- regenerated on every evaluation, never
//! read back into the genome.

use std::collections::{BTreeMap, BTreeSet};

use crate::geneframe::{Adapter, AdapterSite, Argument, Binding, DataKind, DataType, Direction, KingdomTables, PropertyExpr, PropertyKind, PropertyRule, PropertyType, Relation, Requirement, Subject, SymbolRow, TripArgument, Ty};
use crate::kingdom_contract::{KingdomContract, SymbolId};
use crate::kingdom_contract::{relations_from_rows, KingdomRelations};
use crate::typed_chromosome::{AdfId, AdfSet, Epoch, RefKind};
use crate::typed_gep::Tree;

/// Which table-defined type family a kingdom's duals belong to. The family
/// decides the [`Ty`] a dual index means, so the two kingdoms' duals can
/// never be confused for one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FutharkFamily {
    /// The 1-D Futhark kingdom: [`Ty::Futhark1d`].
    Plain,
    /// The Futhark kingdom: [`Ty::Futhark`].
    Shape,
}

impl FutharkFamily {
    /// The type a dual index means in this family.
    pub fn ty(self, index: u16) -> Ty {
        match self {
            FutharkFamily::Plain => Ty::Futhark1d(index),
            FutharkFamily::Shape => Ty::Futhark(index),
        }
    }

    /// The most duals a kingdom of this family may number.
    pub fn max_duals(self) -> usize {
        match self {
            FutharkFamily::Plain => Ty::MAX_FUTHARK_1D_DUALS,
            FutharkFamily::Shape => Ty::MAX_FUTHARK_DUALS,
        }
    }

    /// Whether a type belongs to this family.
    pub fn owns(self, ty: Ty) -> bool {
        match self {
            FutharkFamily::Plain => matches!(ty, Ty::Futhark1d(_)),
            FutharkFamily::Shape => matches!(ty, Ty::Futhark(_)),
        }
    }
}

/// One row of `types.tsv`: a dual of the kingdom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FutharkDual {
    /// The dual's name, `f32.real`.
    pub dual: String,
    /// The family, `f32` / `arr` / `scn` / `fn`.
    pub slot: String,
    /// The form within the family, `real` / `f32` / `f32_f32`.
    pub form: String,
    /// The values a `?_T` placeholder of this dual can take, in order.
    pub literals: Vec<String>,
    /// A program's root may be this dual.
    pub root: bool,
    pub meaning: String,
}

/// What a row of `functions.tsv` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FutharkKind {
    /// A combinator or an elementary function.
    Function,
    /// The constant placeholder `?_T` of its dual: its value is an index into
    /// the dual's literal list.
    Const,
    /// A leaf with a fixed meaning (the input array, the index array, the
    /// constant function).
    Terminal,
}

/// One row of `functions.tsv`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FutharkRow {
    /// The row's name: `map`, `fn.f32_f32.sqrt`, `?_f32`.
    pub name: String,
    /// The ordered argument duals.
    pub inputs: Vec<Ty>,
    pub output: Ty,
    /// The Futhark text, with `{0}`, `{1}`, ... the children.
    pub spelling: String,
    pub kind: FutharkKind,
}

/// The loaded kingdom.
#[derive(Debug, Clone)]
pub struct FutharkKingdom {
    /// The kingdom's name in the symbol table.
    pub kingdom: String,
    /// The type family its duals belong to.
    pub family: FutharkFamily,
    pub duals: Vec<FutharkDual>,
    /// Functions, then terminals, in table order. The symbol id of a row is
    /// its index here.
    pub rows: Vec<FutharkRow>,
}

fn data_lines(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines().skip(1).filter(|l| !l.trim().is_empty()).map(|l| l.split('\t').collect())
}

impl FutharkKingdom {
    /// THE TABLE'S FINGERPRINT: FNV-1a (64 bits) over the kingdom's name, its family and every dual and row as loaded
    /// (their `Debug` text, in table order). A program's text is rebuilt against the table, so a state saved under
    /// one table is read only under the same one (phylu's evaluator state); any change to a loaded row changes it.
    pub fn fingerprint(&self) -> String {
        let text = format!("{}\0{:?}\0{:?}\0{:?}", self.kingdom, self.family, self.duals, self.rows);
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in text.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{h:016x}")
    }

    /// Load a kingdom from its two tables.
    pub fn from_tables(family: FutharkFamily, kingdom: &str, types_tsv: &str, functions_tsv: &str) -> Result<FutharkKingdom, String> {
        let mut duals = Vec::new();
        for f in data_lines(types_tsv) {
            if f.len() < 7 {
                return Err(format!("{kingdom}: types.tsv row {f:?} has {} columns, wants 7", f.len()));
            }
            duals.push(FutharkDual {
                dual: f[1].to_string(),
                slot: f[2].to_string(),
                form: f[3].to_string(),
                literals: if f[4].is_empty() { Vec::new() } else { f[4].split(',').map(|s| s.to_string()).collect() },
                root: f[5] == "yes",
                meaning: f[6].to_string(),
            });
        }
        Ty::check_dual_count(kingdom, duals.len(), family.max_duals())?;
        let ty_of = |name: &str| -> Result<Ty, String> {
            duals
                .iter()
                .position(|d| d.dual == name)
                .map(|i| family.ty(i as u16))
                .ok_or_else(|| format!("{kingdom}: no dual {name}"))
        };

        let mut rows = Vec::new();
        for f in data_lines(functions_tsv) {
            if f.len() < 7 {
                return Err(format!("{kingdom}: functions.tsv row {f:?} has {} columns, wants 7", f.len()));
            }
            let (name, kind, out, args, spelling) = (f[1], f[2], f[3], f[4], f[5]);
            let inputs = if args.is_empty() {
                Vec::new()
            } else {
                args.split(';').map(&ty_of).collect::<Result<Vec<_>, _>>()?
            };
            let kind = match kind {
                "function" => FutharkKind::Function,
                "const" => FutharkKind::Const,
                "terminal" => FutharkKind::Terminal,
                other => return Err(format!("{kingdom}: unknown kind {other}")),
            };
            rows.push(FutharkRow {
                name: name.to_string(),
                inputs,
                output: ty_of(out)?,
                spelling: spelling.to_string(),
                kind,
            });
        }
        Ok(FutharkKingdom { kingdom: kingdom.to_string(), family, duals, rows })
    }

    /// The dual of a type, `None` for a type outside this kingdom.
    pub fn dual(&self, ty: Ty) -> Option<&FutharkDual> {
        match ty {
            Ty::Futhark1d(i) if self.family == FutharkFamily::Plain => self.duals.get(i as usize),
            Ty::Futhark(i) if self.family == FutharkFamily::Shape => self.duals.get(i as usize),
            _ => None,
        }
    }

    /// The type of a dual by name.
    pub fn ty(&self, dual: &str) -> Option<Ty> {
        self.duals.iter().position(|d| d.dual == dual).map(|i| self.family.ty(i as u16))
    }

    /// The roots, the duals the table marks: an ordinary array program, a
    /// scan-result program, and a reduction.
    pub fn roots(&self) -> Vec<Ty> {
        self.duals
            .iter()
            .enumerate()
            .filter(|(_, d)| d.root)
            .map(|(i, _)| self.family.ty(i as u16))
            .collect()
    }

    /// The length of each dual's constant array `C_T`: the number of literals
    /// it has, zero for a dual with none.
    pub fn const_len(&self) -> BTreeMap<Ty, u32> {
        self.duals
            .iter()
            .enumerate()
            .map(|(i, d)| (self.family.ty(i as u16), d.literals.len() as u32))
            .collect()
    }

    /// The kingdom's relations: every row with its ordered input duals, the
    /// `?_T` placeholders as the constant placeholders, and the two roots.
    pub fn relations(&self) -> KingdomRelations {
        let rows: Vec<(String, Vec<Ty>, Ty)> =
            self.rows.iter().map(|r| (r.name.clone(), r.inputs.clone(), r.output)).collect();
        let constants: Vec<&str> = self
            .rows
            .iter()
            .filter(|r| r.kind == FutharkKind::Const)
            .map(|r| r.name.as_str())
            .collect();
        relations_from_rows(&self.kingdom, self.roots(), &rows, &constants)
    }

    /// Render an expressed tree as a Futhark expression.
    ///
    /// A constant placeholder becomes the literal its constant-array index
    /// names; every other row's spelling has its `{0}`, `{1}`, ... replaced by
    /// the children's renderings, in order.
    pub fn render(&self, tree: &Tree) -> Result<String, String> {
        self.render_at(tree, &|_| None)
    }

    /// `render_at` with `call` deciding a node that is not a row of this kingdom
    /// (an ADF reference of the epoch alphabet): `Some` is its text.
    fn render_at(&self, tree: &Tree, call: &dyn Fn(&Tree) -> Option<String>) -> Result<String, String> {
        if let Some(text) = call(tree) {
            return Ok(text);
        }
        let row = self
            .rows
            .get(tree.symbol as usize)
            .ok_or_else(|| format!("{}: symbol {} is not a row of this kingdom", self.kingdom, tree.symbol))?;
        match row.kind {
            FutharkKind::Const => {
                let dual = self
                    .dual(row.output)
                    .ok_or_else(|| format!("{}: a constant placeholder with no dual", self.kingdom))?;
                let idx = tree
                    .const_index
                    .ok_or_else(|| format!("{}: {} has no constant index", self.kingdom, row.name))? as usize;
                dual.literals
                    .get(idx)
                    .cloned()
                    .ok_or_else(|| format!("{}: {} literal {idx} is past the list", self.kingdom, row.name))
            }
            _ => {
                let kids: Vec<String> =
                    tree.kids.iter().map(|k| self.render_at(k, call)).collect::<Result<Vec<_>, _>>()?;
                let mut out = row.spelling.clone();
                for (i, kid) in kids.iter().enumerate() {
                    let placeholder = format!("{{{i}}}");
                    if !out.contains(&placeholder) {
                        return Err(format!(
                            "{}: {}'s spelling has no {placeholder}, so its {i}th argument is dropped",
                            self.kingdom, row.name
                        ));
                    }
                    // Every child is substituted as an ATOM: Futhark's
                    // application is greedy, so an unbracketed child would
                    // swallow its siblings.
                    out = out.replace(&placeholder, &format!("({kid})"));
                }
                // A row whose spelling still holds a placeholder has fewer
                // children than placeholders: that is a table bug, not a gene.
                if out.contains('{') && out.contains('}') {
                    return Err(format!("{}: {}'s spelling still holds a placeholder", self.kingdom, row.name));
                }
                Ok(out)
            }
        }
    }
}

/// A parameter of a generated Futhark function: the variable, its type, and
/// the size variables the type names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FutharkParam {
    pub name: String,
    pub ty: String,
    pub sizes: Vec<String>,
}

/// A program's ADFs as Futhark: the `def adf_<id>` functions, lowest level
/// first, and each ADF's parameters (indices into
/// [`FutharkKingdom::adf_params`]).
#[derive(Debug, Clone)]
pub struct FutharkAdfs {
    pub definitions: String,
    pub params: BTreeMap<AdfId, Vec<usize>>,
}

fn identifiers(text: &str) -> BTreeSet<&str> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).filter(|w| !w.is_empty()).collect()
}

impl FutharkKingdom {
    /// The entry point's inputs, as an ADF may take them: the 1-D Futhark kingdom's
    /// array `xs: [n]f32`, the Futhark kingdom's one matrix per class
    /// (`m_n_m: [n][m]f32`), in the kingdom's own order. A terminal's spelling
    /// names the inputs it reads.
    pub fn adf_params(&self) -> Vec<FutharkParam> {
        match self.family {
            FutharkFamily::Plain => vec![FutharkParam { name: "xs".to_string(), ty: "[n]f32".to_string(), sizes: vec!["n".to_string()] }],
            FutharkFamily::Shape => self
                .duals
                .iter()
                .enumerate()
                .filter(|(_, d)| d.slot == "mat")
                .filter_map(|(i, d)| {
                    let ty = self.family.ty(i as u16);
                    let term = self.rows.iter().find(|r| r.kind == FutharkKind::Terminal && r.output == ty)?;
                    let dims: Vec<String> = d.form.split('.').map(str::to_string).collect();
                    Some(FutharkParam { name: term.spelling.clone(), ty: format!("[{}]f32", dims.join("][")), sizes: dims })
                })
                .collect(),
        }
    }

    /// The Futhark type of an ADF whose root is `ty`, and the size variables it
    /// names. A function-valued ADF is annotated, so its literals are `f32`.
    pub fn adf_type(&self, ty: Ty) -> Result<(String, Vec<String>), String> {
        let d = self.dual(ty).ok_or_else(|| format!("our software has a bug: {ty:?} is not a type of {}", self.kingdom))?;
        Ok(match (self.family, d.slot.as_str(), d.form.as_str()) {
            (FutharkFamily::Plain, "f32", _) => ("f32".to_string(), Vec::new()),
            (FutharkFamily::Plain, "arr" | "scn", "f32") => ("[n]f32".to_string(), vec!["n".to_string()]),
            (FutharkFamily::Plain, "fn", "f32_f32") => ("f32 -> f32".to_string(), Vec::new()),
            (FutharkFamily::Shape, "mat", form) => {
                let dims: Vec<String> = form.split('.').map(str::to_string).collect();
                (format!("[{}]f32", dims.join("][")), dims)
            }
            // The Futhark kingdom's row-function classes hold no function (every row of them is a leaf), so no fold
            // makes an ADF of one; meeting one here is a bug in the fold.
            _ => return Err(format!("our software has a bug: no ADF can have the type {} of {}", d.dual, self.kingdom)),
        })
    }

    /// The ADFs of `adfs` as Futhark functions, in level order (a definition
    /// after everything it calls): `def adf_<id> [sizes] (params) : type =
    /// body`. An ADF over the entry's inputs takes the inputs its body reads;
    /// a closed one takes none.
    pub fn adf_functions(&self, epoch: &Epoch, adfs: &AdfSet) -> Result<FutharkAdfs, String> {
        self.adf_functions_where(epoch, adfs, &|_| true)
    }

    /// [`Self::adf_functions`] for the ADFs `tree` calls, directly or through
    /// another ADF: the definitions one program needs.
    pub fn adf_functions_for(&self, epoch: &Epoch, adfs: &AdfSet, tree: &Tree) -> Result<FutharkAdfs, String> {
        let mut needed = BTreeSet::new();
        let mut stack = vec![tree.clone()];
        while let Some(t) = stack.pop() {
            if let Some(RefKind::Adf { adf_id }) = epoch.reference(t.symbol) {
                if needed.insert(adf_id) {
                    stack.push(adfs.get(adf_id).ok_or_else(|| format!("our software has a bug: A{adf_id} names no ADF of the set"))?.tree());
                }
            }
            stack.extend(t.kids);
        }
        self.adf_functions_where(epoch, adfs, &|id| needed.contains(&id))
    }

    fn adf_functions_where(&self, epoch: &Epoch, adfs: &AdfSet, keep: &dyn Fn(AdfId) -> bool) -> Result<FutharkAdfs, String> {
        let params = self.adf_params();
        let bodies = adfs.expanded(epoch);
        let mut uses: BTreeMap<AdfId, Vec<usize>> = BTreeMap::new();
        for (id, body) in &bodies {
            let mut names = BTreeSet::new();
            fn walk<'a>(k: &'a FutharkKingdom, t: &Tree, out: &mut BTreeSet<&'a str>) {
                if let Some(r) = k.rows.get(t.symbol as usize) {
                    out.extend(identifiers(&r.spelling));
                }
                for c in &t.kids {
                    walk(k, c, out);
                }
            }
            walk(self, body, &mut names);
            uses.insert(*id, params.iter().enumerate().filter(|(_, p)| names.contains(p.name.as_str())).map(|(i, _)| i).collect());
        }
        let mut definitions = String::new();
        for (id, _) in adfs.schedule().into_iter().filter(|(id, _)| keep(*id)) {
            let adf = adfs.get(id).ok_or_else(|| format!("our software has a bug: ADF {id} is scheduled but has no body"))?;
            let (ty, result_sizes) = self.adf_type(adf.root_type)?;
            let mine: Vec<&FutharkParam> = uses[&id].iter().map(|i| &params[*i]).collect();
            let mut sizes: Vec<&str> = Vec::new();
            for p in &mine {
                for s in &p.sizes {
                    if !sizes.contains(&s.as_str()) {
                        sizes.push(s);
                    }
                }
            }
            if let Some(s) = result_sizes.iter().find(|s| !sizes.contains(&s.as_str())) {
                return Err(format!("our software has a bug: ADF {id}'s result names the size {s}, which none of its inputs has"));
            }
            sizes.sort_unstable();
            let size_text: String = sizes.iter().map(|s| format!(" [{s}]")).collect();
            let param_text: String = mine.iter().map(|p| format!(" ({}: {})", p.name, p.ty)).collect();
            let body = self.render_called(epoch, &uses, &params, &adf.tree())?;
            definitions.push_str(&format!("def adf_{id}{size_text}{param_text} : {ty} =\n  {body}\n\n"));
        }
        Ok(FutharkAdfs { definitions, params: uses })
    }

    /// Render a tree of the epoch alphabet: a kingdom row as [`Self::render`]
    /// spells it, an ADF reference `A_d` as the call `adf_<d> <its inputs>`.
    pub fn render_folded(&self, epoch: &Epoch, adfs: &FutharkAdfs, tree: &Tree) -> Result<String, String> {
        self.render_called(epoch, &adfs.params, &self.adf_params(), tree)
    }

    fn render_called(&self, epoch: &Epoch, uses: &BTreeMap<AdfId, Vec<usize>>, params: &[FutharkParam], tree: &Tree) -> Result<String, String> {
        let bad = std::cell::RefCell::new(None);
        let call = |t: &Tree| -> Option<String> {
            match epoch.reference(t.symbol) {
                Some(RefKind::Adf { adf_id }) => match uses.get(&adf_id) {
                    Some(u) => Some(std::iter::once(format!("adf_{adf_id}")).chain(u.iter().map(|i| params[*i].name.clone())).collect::<Vec<_>>().join(" ")),
                    None => {
                        *bad.borrow_mut() = Some(format!("our software has a bug: A{adf_id} names no ADF of the set"));
                        Some(String::new())
                    }
                },
                Some(RefKind::Genic { gene_id }) => {
                    *bad.borrow_mut() = Some(format!("our software has a bug: a genic terminal G{gene_id} in a Futhark program"));
                    Some(String::new())
                }
                None => None,
            }
        };
        let text = self.render_at(tree, &call)?;
        match bad.into_inner() {
            Some(e) => Err(e),
            None => Ok(text),
        }
    }

    /// The 1-D Futhark kingdom's whole program for a folded tree: its ADFs, then
    /// `entry main (xs: []f32) : <result> = <body>`.
    pub fn plain_program_folded(&self, epoch: &Epoch, adfs: &AdfSet, root: Ty, tree: &Tree) -> Result<String, String> {
        if self.family != FutharkFamily::Plain {
            return Err(format!("our software has a bug: {} is not the 1-D Futhark kingdom", self.kingdom));
        }
        let f = self.adf_functions_for(epoch, adfs, tree)?;
        let result = match self.dual(root).map(|d| d.slot.as_str()) {
            Some("f32") => "f32",
            Some("arr" | "scn") => "[]f32",
            _ => return Err(format!("our software has a bug: {root:?} is not a program root")),
        };
        Ok(format!("{}entry main (xs: []f32) : {result} = {}\n", f.definitions, self.render_folded(epoch, &f, tree)?))
    }
}

/// A splitmix64: deterministic, no dependency, and the same sequence every
/// run (so a failure is reproducible).
///
/// It lives here, not in an example, because a *reported* failure has to be
/// reproducible: the gene that made the compiler hang must be regenerable
/// from its seed alone, by the rescue experiment as well as by the sweep that
/// found it.
#[derive(Debug, Clone)]
pub struct GeneRng(u64);

impl GeneRng {
    /// A generator with the given seed.
    pub fn new(seed: u64) -> GeneRng {
        GeneRng(seed)
    }

    /// The generator's state: `GeneRng::new(rng.state())` continues the sequence exactly (a checkpoint keeps it).
    pub fn state(&self) -> u64 {
        self.0
    }

    /// The next number in the sequence.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

// ---- master 6: data types, properties, requirements, bindings (plan section 7, task F1b) ---------------------------
//
// A master 6 view (`kingdoms/futhark/sig/<problem>/m6/`, written by `view.py` and by `futhark_view::master6`) is
// three tables: `datatypes.tsv`, `symbols.tsv` and `environment.tsv`, over the master's `adapters.tsv`.
// [`Kingdom6::from_tables`] reads them, and the kingdom gives the gene frame's tables ([`KingdomTables`], G1 and G3)
// and the contract with its variable terminals (G2). Master 5's loader above is unchanged and stays loadable until
// task D1.

/// One data type of a master 6 view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataType6 {
    /// As Futhark spells it without sizes: `[]f32`, `(f32, f32)`, `[]f32 -> []f32`.
    pub name: String,
    pub kind: DataKind,
    /// The data type inside symbol names: `a.f32`.
    pub tag: String,
    /// What a `var` cell renders with no variable in scope; empty for `size`, which has none.
    pub constant: String,
    pub constant_props: Vec<(String, PropertyExpr)>,
    /// The properties its values carry, in order.
    pub properties: Vec<(String, PropertyKind)>,
    /// A scalar element's literal pool (the view's `literals` line), for its `const` row.
    pub literals: Vec<String>,
}

/// What a master 6 symbol is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind6 {
    Function,
    /// A Dc literal of its data type's pool.
    Const,
    /// A leaf with a fixed meaning.
    Terminal,
    /// `var@T`: its Dv word chooses a variable of `T` in scope.
    Var,
}

/// One symbol of a master 6 view: its symbol id is its index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol6 {
    /// The master row it instantiates.
    pub reference: String,
    pub name: String,
    pub kind: Kind6,
    pub inputs: Vec<Ty>,
    pub output: Ty,
    pub rules: Vec<(String, PropertyExpr)>,
    /// Its requirements, each with its relation and adapter, in the `requires` order (the order adapters render in).
    pub requires: Vec<Requirement>,
    pub binds: Vec<Binding>,
    /// The argument whose subtree runs once a trip.
    pub trip: Option<u32>,
    pub spelling: String,
}

/// A requirement of the entry's result (`out.rows = k -> fit.root`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootRequirement6 {
    pub property: String,
    pub relation: Relation,
    pub against: PropertyExpr,
    pub adapter: String,
}

/// The entry: its name, its result's data type and properties, and the requirements on its result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Root6 {
    pub name: String,
    pub data_type: Ty,
    pub properties: Vec<(String, PropertyExpr)>,
    pub requires: Vec<RootRequirement6>,
}

/// A variable of the root environment: an input, a zero array or a size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootVariable6 {
    pub kind: String,
    pub name: String,
    pub data_type: Ty,
    pub properties: Vec<(String, PropertyExpr)>,
}

/// A loaded master 6 kingdom.
#[derive(Debug, Clone)]
pub struct Kingdom6 {
    pub kingdom: String,
    /// Data type `i` is `Ty::Futhark(i)`.
    pub data_types: Vec<DataType6>,
    pub symbols: Vec<Symbol6>,
    pub adapters: Vec<Adapter>,
    pub root: Root6,
    /// The root environment's variables, in the specification's order (G3's environment reads this).
    pub variables: Vec<RootVariable6>,
}

fn bug6(kingdom: &str, what: String) -> String {
    format!("our software has a bug: {kingdom}: {what}")
}

/// `p=expr` items separated by `sep`.
fn assignments(kingdom: &str, text: &str, sep: &str) -> Result<Vec<(String, PropertyExpr)>, String> {
    text.split(sep)
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(|item| {
            let (p, e) = item.split_once('=').ok_or_else(|| bug6(kingdom, format!("the property {item:?} is not <property>=<expr>")))?;
            Ok((p.trim().to_string(), PropertyExpr::parse(e.trim())?))
        })
        .collect()
}

/// A requirement's text as (subject, relation, against, adapter).
fn requirement6(kingdom: &str, text: &str) -> Result<(String, Relation, PropertyExpr, String), String> {
    let (subject, relation, expr, adapter) = crate::futhark_view::master6::parse_requirement(text)?;
    let relation = Relation::from_symbol(&relation).ok_or_else(|| bug6(kingdom, format!("the relation of {text:?}")))?;
    Ok((subject, relation, PropertyExpr::parse(&expr)?, adapter))
}

/// The table's rows (its header skipped), each with exactly `n` fields.
fn rows6<'a>(kingdom: &str, file: &str, text: &'a str, n: usize) -> Result<Vec<Vec<&'a str>>, String> {
    data_lines(text)
        .map(|f| if f.len() == n { Ok(f) } else { Err(bug6(kingdom, format!("{file} row {f:?} has {} columns, wants {n}", f.len()))) })
        .collect()
}

impl Kingdom6 {
    /// Load a master 6 kingdom from its three tables, the master's adapters and its elements' literal pools.
    pub fn from_tables(kingdom: &str, datatypes_tsv: &str, symbols_tsv: &str, environment_tsv: &str, adapters_tsv: &str, literals: &BTreeMap<String, Vec<String>>) -> Result<Kingdom6, String> {
        let dts = rows6(kingdom, "datatypes.tsv", datatypes_tsv, 8)?;
        Ty::check_dual_count(kingdom, dts.len(), Ty::MAX_FUTHARK_DUALS)?;
        let names: Vec<&str> = dts.iter().map(|f| f[1]).collect();
        let index_of = |name: &str| -> Result<usize, String> { names.iter().position(|n| *n == name).ok_or_else(|| bug6(kingdom, format!("no data type {name}"))) };
        let ty_of = |name: &str| -> Result<Ty, String> { index_of(name).map(|i| Ty::Futhark(i as u16)) };
        let tys = |text: &str| -> Result<Vec<Ty>, String> { text.split(';').filter(|a| !a.is_empty()).map(ty_of).collect() };
        let mut data_types = Vec::new();
        for (i, f) in dts.iter().enumerate() {
            if f[0] != i.to_string() {
                return Err(bug6(kingdom, format!("datatypes.tsv row {i} has the id {}", f[0])));
            }
            let kind = match f[2] {
                "scalar" => DataKind::Scalar,
                "array" => DataKind::Array { element: ty_of(f[3])? },
                "tuple" => DataKind::Tuple { components: tys(f[3])? },
                "function" => {
                    let (params, result) = f[3].split_once(" -> ").ok_or_else(|| bug6(kingdom, format!("the function type {} has the members {:?}", f[1], f[3])))?;
                    DataKind::Function { parameters: tys(params)?, result: ty_of(result)? }
                }
                other => return Err(bug6(kingdom, format!("the data type {} is of the kind {other}", f[1]))),
            };
            let properties = f[7]
                .split(';')
                .filter(|p| !p.is_empty())
                .map(|p| {
                    let (n, k) = p.split_once(':').ok_or_else(|| bug6(kingdom, format!("the property {p:?} of {} is not <name>:<kind>", f[1])))?;
                    Ok((n.to_string(), PropertyKind::from_name(k).ok_or_else(|| bug6(kingdom, format!("the property kind {k}")))?))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let literals = if kind == DataKind::Scalar { literals.get(f[1]).cloned().unwrap_or_default() } else { Vec::new() };
            data_types.push(DataType6 { name: f[1].to_string(), kind, tag: f[4].to_string(), constant: f[5].to_string(), constant_props: assignments(kingdom, f[6], "&")?, properties, literals });
        }
        let mut symbols = Vec::new();
        for (i, f) in rows6(kingdom, "symbols.tsv", symbols_tsv, 11)?.into_iter().enumerate() {
            if f[0] != i.to_string() {
                return Err(bug6(kingdom, format!("symbols.tsv row {i} has the id {}", f[0])));
            }
            let symbol = i as i64;
            let kind = match f[3] {
                "function" => Kind6::Function,
                "const" => Kind6::Const,
                "terminal" => Kind6::Terminal,
                "var" => Kind6::Var,
                other => return Err(bug6(kingdom, format!("the symbol {} is of the kind {other}", f[2]))),
            };
            if kind == Kind6::Const && data_types[index_of(f[4])?].literals.is_empty() {
                return Err(bug6(kingdom, format!("the constant {} has no literal pool", f[2])));
            }
            let requires = f[7]
                .split(';')
                .map(str::trim)
                .filter(|q| !q.is_empty())
                .map(|q| {
                    let (subject, relation, against, adapter) = requirement6(kingdom, q)?;
                    Ok(Requirement { symbol, subject: Subject::parse(&subject)?, relation, against, adapter })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let binds = f[8]
                .split(';')
                .map(str::trim)
                .filter(|b| !b.is_empty())
                .map(|b| {
                    let (pos, rest) = b.split_once(':').ok_or_else(|| bug6(kingdom, format!("the binding {b:?}")))?;
                    let (dt, props) = rest.split_once(' ').unwrap_or((rest, ""));
                    Ok(Binding { symbol, position: pos.parse().map_err(|_| bug6(kingdom, format!("the binding {b:?}")))?, data_type: ty_of(dt)?, properties: assignments(kingdom, props, "&")? })
                })
                .collect::<Result<Vec<_>, String>>()?;
            symbols.push(Symbol6 {
                reference: f[1].to_string(),
                name: f[2].to_string(),
                kind,
                inputs: tys(f[5])?,
                output: ty_of(f[4])?,
                rules: assignments(kingdom, f[6], ";")?,
                requires,
                binds,
                trip: if f[9].is_empty() { None } else { Some(f[9].parse().map_err(|_| bug6(kingdom, format!("the trip {:?} of {}", f[9], f[2])))?) },
                spelling: f[10].to_string(),
            });
        }
        let adapters = rows6(kingdom, "adapters.tsv", adapters_tsv, 5)?
            .into_iter()
            .map(|f| {
                let runs_at = AdapterSite::from_name(f[2]).ok_or_else(|| bug6(kingdom, format!("the adapter {} runs at {:?}", f[0], f[2])))?;
                let heavy = match f[3] {
                    "yes" => true,
                    "no" => false,
                    other => return Err(bug6(kingdom, format!("the adapter {} is heavy {other:?}", f[0]))),
                };
                Ok(Adapter { name: f[0].to_string(), heavy, runs_at })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let env = rows6(kingdom, "environment.tsv", environment_tsv, 5)?;
        let (first, rest) = env.split_first().ok_or_else(|| bug6(kingdom, "environment.tsv has no root".to_string()))?;
        if first[0] != "root" {
            return Err(bug6(kingdom, format!("environment.tsv's first row is {first:?}, not the root")));
        }
        let root_requires = first[4]
            .split(';')
            .map(str::trim)
            .filter(|q| !q.is_empty())
            .map(|q| {
                let (subject, relation, against, adapter) = requirement6(kingdom, q)?;
                let property = subject.strip_prefix("out.").ok_or_else(|| bug6(kingdom, format!("the root's requirement {q:?} is not on out.<property>")))?;
                Ok(RootRequirement6 { property: property.to_string(), relation, against, adapter })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let root = Root6 { name: first[1].to_string(), data_type: ty_of(first[2])?, properties: assignments(kingdom, first[3], "&")?, requires: root_requires };
        let variables = rest
            .iter()
            .map(|f| {
                if !["input", "zeros", "size"].contains(&f[0]) || !f[4].is_empty() {
                    return Err(bug6(kingdom, format!("environment.tsv row {f:?} is not a root variable")));
                }
                Ok(RootVariable6 { kind: f[0].to_string(), name: f[1].to_string(), data_type: ty_of(f[2])?, properties: assignments(kingdom, f[3], "&")? })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Kingdom6 { kingdom: kingdom.to_string(), data_types, symbols, adapters, root, variables })
    }

    /// The type of a data type by name.
    pub fn ty(&self, name: &str) -> Option<Ty> {
        self.data_types.iter().position(|d| d.name == name).map(|i| Ty::Futhark(i as u16))
    }

    /// The data type of a type.
    pub fn data_type(&self, ty: Ty) -> Option<&DataType6> {
        match ty {
            Ty::Futhark(i) => self.data_types.get(usize::from(i)),
            _ => None,
        }
    }

    /// THE GENE FRAME'S TABLES: one data type a row with its constant's properties (none for `size`) and its
    /// property types; one symbol a row with its Argument rows, property rules, requirements (relation and adapter),
    /// bindings and trip argument; the adapters. The tables are checked ([`KingdomTables::check`]).
    pub fn tables(&self) -> Result<KingdomTables, String> {
        let mut t = KingdomTables { kingdom: self.kingdom.clone(), adapters: self.adapters.clone(), ..KingdomTables::default() };
        for (i, d) in self.data_types.iter().enumerate() {
            let id = Ty::Futhark(i as u16);
            let constant = if d.constant.is_empty() { None } else { Some(d.constant_props.clone()) };
            t.data_types.push(DataType { id, name: d.name.clone(), kind: d.kind.clone(), constant });
            for (name, kind) in &d.properties {
                t.property_types.push(PropertyType { id: t.property_types.len() as u32, data_type: id, name: name.clone(), kind: *kind });
            }
        }
        for (i, s) in self.symbols.iter().enumerate() {
            let symbol = i as i64;
            t.symbols.push(SymbolRow { symbol, symbol_name: s.name.clone(), alias: s.reference.clone(), semantic_id: s.name.clone(), spelling: s.spelling.clone() });
            for (k, ty) in s.inputs.iter().enumerate() {
                t.arguments.push(Argument { symbol, position: k as u32, direction: Direction::In, data_type: *ty });
            }
            t.arguments.push(Argument { symbol, position: 0, direction: Direction::Out, data_type: s.output });
            for (property, expr) in &s.rules {
                t.property_rules.push(PropertyRule { symbol, property: property.clone(), expr: expr.clone() });
            }
            t.requirements.extend(s.requires.iter().cloned());
            t.bindings.extend(s.binds.iter().cloned());
            if let Some(position) = s.trip {
                t.trip_arguments.push(TripArgument { symbol, position });
            }
        }
        t.check().map_err(|d| d.join("\n"))?;
        Ok(t)
    }

    /// The kingdom's relations (G2): every symbol's signature from the gene frame's Argument rows, the root, the
    /// `const` rows as the constant placeholders and the `var` rows as the variable terminals.
    pub fn relations(&self) -> Result<KingdomRelations, String> {
        let t = self.tables()?;
        let rows: Vec<(String, Vec<Ty>, Ty)> = t.signatures().into_iter().map(|(s, ins, out)| (self.symbols[s as usize].name.clone(), ins, out)).collect();
        let constants: Vec<&str> = self.symbols.iter().filter(|s| s.kind == Kind6::Const).map(|s| s.name.as_str()).collect();
        let variables: Vec<&str> = self.symbols.iter().filter(|s| s.kind == Kind6::Var).map(|s| s.name.as_str()).collect();
        relations_from_rows(&self.kingdom, vec![self.root.data_type], &rows, &constants).with_variables(&variables)
    }

    /// The checked contract: what a gene of this kingdom is laid out by.
    pub fn contract(&self) -> Result<KingdomContract, String> {
        self.relations()?.check().map_err(|d| d.join("\n"))
    }
}

// ---- master 6: pass 5, the renderer (plan section 7, task F2) ---------------------------------------------------
//
// Passes 1 to 4 (`typed_gep::express`, `typed_environment::Decoder::decode`) give the decoded tree: every node's
// environment, variable, properties, requirement outcomes and work-bound substitution. Pass 5 writes the Futhark text:
// where every requirement holds, the row's plain spelling; where one fails, the spelling with the adapter its
// requirement names (master 6 README 5.5), in `requires` order; a substituted node as its substitute. The renderer
// decides nothing again: it reads the decoding.
//
// THE ONE INVARIANT the adapters rest on: the Futhark type of every rendered expression is its data type with its
// properties as sizes. A size name or a literal renders as itself and a `min` as `i64.min a b`, which Futhark 0.28.1
// compares as written (each `min` has one spelling, so two of the same value are the same text). An adapter fits an
// operand to a property it can spell (`fit (i64.min k n) 0f32 (xs)`), so the result's type is its rule's property and a
// parent's requirement that holds holds in Futhark too. Only a `?` property (no row of the committed master gives
// one) is read at run time: its operand is bound by `let` under a reserved name, as the README says (`let ad_a = ..
// let ad_b = .. let ad_l = i64.min (length ad_a) (length ad_b) in ..`), and where the length is a TYPE (a loop state's
// declaration) by a size binding (`let [ad_m] (ad_s: [ad_m]f32) = ..`, [`Render6::bind`]); the result is then an
// unknown size, `?` again, and every requirement on it fails, so it is adapted wherever a size must agree.
// (The README's `ad_l` form for two KNOWN lengths gives an existential size, and Futhark rejects a `map2` of two such
// results, "Sizes d12 and d25 do not match", measured 2026-11-05: so a known length is fitted to its spelling.)

/// What the entry point declares that the root environment does not: the sizes given as an `i32` parameter (the
/// histogram's `k`), which the program rebinds as `i64` (`let k = i64.i32 k`). Such a size is the entry's first
/// parameter, before the arrays, as every problem's source has it (the specification does not keep the order).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Header6 {
    pub size_inputs: Vec<String>,
}

impl Header6 {
    /// The header of a master 6 specification: its `size` lines that name an element.
    pub fn of_spec(spec: &crate::futhark_view::master6::ViewSpec6) -> Header6 {
        Header6 { size_inputs: spec.env.iter().filter(|l| l.kind == "size" && !l.elem.is_empty()).map(|l| l.name.clone()).collect() }
    }
}

/// A rendered master 6 program: its text, the nodes it renders (the inputs a substitution drops are not rendered;
/// an ADF's nodes once, in its definition), and every adapter it renders with its node's path (the entry's own
/// requirements at the root's path), in render order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program6 {
    pub text: String,
    pub nodes: usize,
    pub adapters: Vec<(Vec<u32>, String)>,
}

impl Program6 {
    /// No adapter and no substitution: the program renders as master 5 renders it.
    pub fn adapter_free(&self) -> bool {
        self.adapters.is_empty()
    }

    /// The number of nodes that render at least one adapter.
    pub fn adapter_nodes(&self) -> usize {
        self.adapters.iter().map(|(p, _)| p).collect::<BTreeSet<_>>().len()
    }
}

/// An operand while its node renders: its text, data type and properties (its Futhark type), and the name a `?`
/// property was bound to (`ad_m`), read in place of the property.
#[derive(Debug, Clone)]
struct Operand6 {
    text: String,
    ty: Ty,
    props: BTreeMap<String, PropertyExpr>,
    bound: BTreeMap<String, String>,
}

/// A property as Futhark text: a size name or a literal as itself, a `min` as nested `i64.min`, a sum with `+`; `None`
/// for `?`.
fn futhark_size(e: &PropertyExpr) -> Option<String> {
    match e {
        PropertyExpr::Size(n) => Some(n.clone()),
        PropertyExpr::Literal(v) => Some(v.to_string()),
        PropertyExpr::Min(args) => {
            let texts: Vec<String> = args.iter().map(|a| futhark_size(a).map(|t| atom(&t))).collect::<Option<_>>()?;
            let (last, rest) = texts.split_last()?;
            Some(rest.iter().rev().fold(last.clone(), |acc, a| format!("i64.min {a} {}", atom(&acc))))
        }
        PropertyExpr::Sum(terms) => Some(terms.iter().map(|t| futhark_size(t).map(|x| atom(&x))).collect::<Option<Vec<_>>>()?.join(" + ")),
        PropertyExpr::Unknown | PropertyExpr::Input { .. } => None,
    }
}

/// A text as an application's argument: a name or a non-negative literal as itself, anything else in parentheses.
fn atom(t: &str) -> String {
    if !t.is_empty() && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        t.to_string()
    } else {
        format!("({t})")
    }
}

/// A property in a bound variable's class text (master 5's class: `a.n`, `ix.cols`): a size name or a literal as
/// itself, a `min` as `min_k_n`, a sum as `sum_..`, `?` as `q`.
fn class_part(e: &PropertyExpr) -> String {
    match e {
        PropertyExpr::Size(n) => n.clone(),
        PropertyExpr::Literal(v) if *v < 0 => format!("m{}", -v),
        PropertyExpr::Literal(v) => v.to_string(),
        PropertyExpr::Min(args) => format!("min_{}", args.iter().map(class_part).collect::<Vec<_>>().join("_")),
        PropertyExpr::Sum(terms) => format!("sum_{}", terms.iter().map(class_part).collect::<Vec<_>>().join("_")),
        PropertyExpr::Unknown | PropertyExpr::Input { .. } => "q".to_string(),
    }
}

/// The renderer of one decoding.
struct Render6<'a> {
    k: &'a Kingdom6,
    decoding: &'a crate::typed_environment::Decoding,
    /// The tables' requirement rows, in the order G3's outcomes index them (each symbol's `requires`, symbol order).
    reqs: Vec<&'a Requirement>,
    /// The view has several elements: an array's class names its element (`a.f32.n`), as master 5's does.
    multi: bool,
    /// The subtrees rendered as ADFs, by path; the index is the ADF's number.
    adfs: &'a [Vec<u32>],
    uses: BTreeSet<&'static str>,
    adapters: Vec<(Vec<u32>, String)>,
    nodes: usize,
}

impl<'a> Render6<'a> {
    fn bug(&self, what: String) -> String {
        bug6(&self.k.kingdom, what)
    }

    fn data_type(&self, ty: Ty) -> Result<&'a DataType6, String> {
        self.k.data_type(ty).ok_or_else(|| self.bug(format!("{ty:?} is no data type of the view")))
    }

    /// The property names of a data type, in its order (`len`; `rows`, `cols`).
    fn property_names(&self, ty: Ty) -> Result<Vec<&'a str>, String> {
        Ok(self.data_type(ty)?.properties.iter().map(|(n, _)| n.as_str()).collect())
    }

    /// The scalar inside an array data type (itself for a scalar).
    fn innermost(&self, ty: Ty) -> Result<Ty, String> {
        let mut t = ty;
        while let DataKind::Array { element } = self.data_type(t)?.kind {
            t = element;
        }
        Ok(t)
    }

    /// The zero `fit` pads with, at an array's element: the scalar's constant for a vector, a zero row
    /// (`replicate c 0f32`) for a matrix.
    fn zero(&self, op: &Operand6) -> Result<String, String> {
        let d = self.data_type(op.ty)?;
        let DataKind::Array { element } = d.kind else { return Err(self.bug(format!("fit pads a {}, which is no array", d.name))) };
        let scalar = self.data_type(self.innermost(element)?)?.constant.clone();
        if matches!(self.data_type(element)?.kind, DataKind::Array { .. }) {
            Ok(format!("(replicate {} {scalar})", atom(&self.dim(op, "cols")?)))
        } else {
            Ok(scalar)
        }
    }

    /// An operand's property as Futhark text: the name it was bound to, else its spelling; a `?` not bound is a bug.
    fn dim(&self, op: &Operand6, p: &str) -> Result<String, String> {
        if let Some(name) = op.bound.get(p) {
            return Ok(name.clone());
        }
        let e = op.props.get(p).ok_or_else(|| self.bug(format!("a {} has no property {p}", self.data_type(op.ty).map_or("?", |d| d.name.as_str()))))?;
        futhark_size(e).ok_or_else(|| self.bug(format!("the property {p} = {e} is read where it has no spelling")))
    }

    /// The Futhark type of an operand: its data type with its properties as sizes (`[n]f32`, `[k][d]f32`, `(f32, f32)`).
    fn type_text(&self, op: &Operand6) -> Result<String, String> {
        let d = self.data_type(op.ty)?;
        if !matches!(d.kind, DataKind::Array { .. }) {
            return Ok(d.name.clone());
        }
        let mut s = String::new();
        for p in self.property_names(op.ty)? {
            s.push_str(&format!("[{}]", self.dim(op, p)?));
        }
        s.push_str(&self.data_type(self.innermost(op.ty)?)?.name);
        Ok(s)
    }

    /// A bound variable's class (master 5's, from its data type and properties): an index `ix.<bound>`, a scalar its
    /// element, an array `a.<dims>` (`a.<elem>.<dims>` in a view of several elements), any other its tag.
    fn class_of(&self, v: &crate::typed_environment::Variable) -> Result<String, String> {
        let d = self.data_type(v.data_type)?;
        let prop = |p: &str| v.properties.get(p).map_or_else(|| "q".to_string(), class_part);
        Ok(match &d.kind {
            DataKind::Scalar if d.properties.iter().any(|(n, _)| n == "bound") => format!("ix.{}", prop("bound")),
            DataKind::Array { .. } => {
                let dims: Vec<String> = d.properties.iter().map(|(n, _)| prop(n)).collect();
                let elem = &self.data_type(self.innermost(v.data_type)?)?.name;
                if self.multi {
                    format!("a.{elem}.{}", dims.join("."))
                } else {
                    format!("a.{}", dims.join("."))
                }
            }
            DataKind::Scalar => d.name.clone(),
            _ => d.tag.clone(),
        })
    }

    /// THE NAME OF A VARIABLE in an environment: an entry parameter's own name; a bound variable is master 5's
    /// `var_name` of its class and its ordinal, one past the bound variables of its class before it (`i_n`, `i2_n`,
    /// `s3_f32`, `s_a_n`), so two variables in scope never share a name.
    fn name_of(&self, env: &crate::typed_environment::Environment, v: &crate::typed_environment::Variable) -> Result<String, String> {
        if let Some(n) = &v.name {
            return Ok(n.clone());
        }
        let class = self.class_of(v)?;
        let mut ordinal = 1;
        for w in env.variables.iter().take_while(|w| w.binder != v.binder) {
            if w.name.is_none() && self.class_of(w)? == class {
                ordinal += 1;
            }
        }
        Ok(crate::futhark_view::var_name(&class, ordinal))
    }

    /// Fit an operand's property `p` to `target` (its value and its Futhark text): a vector's length by `fit`, a
    /// matrix's rows by `fit` with a zero row, its columns by `map (fit ..)`.
    fn fit_to(&mut self, op: &mut Operand6, p: &str, target: (PropertyExpr, String)) -> Result<(), String> {
        let (value, text) = target;
        let elem_zero = self.data_type(self.innermost(op.ty)?)?.constant.clone();
        op.text = match p {
            "len" | "rows" => format!("fit {} {} ({})", atom(&text), self.zero(op)?, op.text),
            "cols" => format!("map (fit {} {elem_zero}) ({})", atom(&text), op.text),
            other => return Err(self.bug(format!("fit has no rendering for the property {other}"))),
        };
        self.uses.insert("fit");
        op.props.insert(p.to_string(), value);
        if value_is_unknown(op.props.get(p)) {
            op.bound.insert(p.to_string(), text);
        } else {
            op.bound.remove(p);
        }
        Ok(())
    }

    /// Bind an operand by `let` under the reserved name `arr` (its text becomes the name) and, when `len` is given,
    /// its property `len.1` to the reserved name `len.0` as a SIZE of its type: `let [ad_m] (ad_s: [ad_m]f32) = (..)`.
    /// The README's `let ad_s = .. let ad_m = length ad_s` names the length as a value only, and Futhark 0.28.1 rejects
    /// a loop declared `[ad_m]` over it ("Sizes ad_m and d6 do not match", measured 2026-11-05); the size binding gives
    /// Futhark the name of the unknown size itself.
    fn bind(&self, op: &mut Operand6, arr: &str, len: Option<(&str, &str)>, lets: &mut Vec<String>, used: &mut BTreeSet<String>) -> Result<(), String> {
        for name in std::iter::once(arr).chain(len.map(|l| l.0)) {
            if !used.insert(name.to_string()) {
                return Err(self.bug(format!("one node binds the reserved name {name} twice")));
            }
        }
        match len {
            None => lets.push(format!("let {arr} = ({})", op.text)),
            Some((name, p)) => {
                if !self.property_names(op.ty)?.contains(&p) {
                    return Err(self.bug(format!("a {} has no size {p} to bind", self.data_type(op.ty)?.name)));
                }
                op.bound.insert(p.to_string(), name.to_string());
                lets.push(format!("let [{name}] ({arr}: {}) = ({})", self.type_text(op)?, op.text));
            }
        }
        op.text = arr.to_string();
        Ok(())
    }

    /// The input position and property a requirement side names (`in1.len`).
    fn side(&self, e: &PropertyExpr) -> Option<(usize, String)> {
        match e {
            PropertyExpr::Input { position, property } => Some((*position as usize, property.clone())),
            _ => None,
        }
    }

    /// One failing requirement's adapter on the node's operands.
    fn adapt(&mut self, o: &crate::typed_environment::Outcome, ops: &mut [Operand6], lets: &mut Vec<String>, used: &mut BTreeSet<String>) -> Result<(), String> {
        let q = *self.reqs.get(o.requirement).ok_or_else(|| self.bug(format!("the outcome names requirement {}, which the tables do not have", o.requirement)))?;
        let Subject::Input { position, property } = &q.subject else { return Err(self.bug(format!("the work bound's adapter {} renders as an adapter, not a substitution", o.adapter))) };
        let (s, ps) = (*position as usize, property.clone());
        if s >= ops.len() {
            return Err(self.bug(format!("the requirement reads input {s} of a node of {} inputs", ops.len())));
        }
        let against = self.side(&q.against);
        let known = |e: &PropertyExpr| futhark_size(e).map(|t| (e.clone(), t));
        match o.adapter.as_str() {
            "clampix" => {
                let (a, pa) = against.ok_or_else(|| self.bug("clampix's requirement names no input's length".to_string()))?;
                let len = match known(&o.against) {
                    Some((_, t)) => t,
                    None => {
                        let mut op = ops[a].clone();
                        self.bind(&mut op, "ad_a", Some(("ad_m", &pa)), lets, used)?;
                        ops[a] = op;
                        self.dim(&ops[a], &pa)?
                    }
                };
                ops[s].text = format!("clampix {} ({})", atom(&len), ops[s].text);
                self.uses.insert("clampix");
            }
            "fit" | "fit.body" | "fit.state" => {
                let target = match (known(&o.against), against) {
                    (Some(t), _) => t,
                    (None, Some((j, pj))) => {
                        let (arr, len) = if o.adapter == "fit.state" { ("ad_s", "ad_m") } else { ("ad_a", "ad_m") };
                        let mut op = ops[j].clone();
                        self.bind(&mut op, arr, Some((len, &pj)), lets, used)?;
                        ops[j] = op;
                        (PropertyExpr::Unknown, self.dim(&ops[j], &pj)?)
                    }
                    (None, None) => return Err(self.bug(format!("{} fits to {}, which has no spelling", o.adapter, o.against))),
                };
                let mut op = ops[s].clone();
                self.fit_to(&mut op, &ps, target)?;
                ops[s] = op;
            }
            "fit.pair" => {
                let (j, pj) = against.ok_or_else(|| self.bug("fit.pair's requirement names no second input".to_string()))?;
                if j >= ops.len() {
                    return Err(self.bug(format!("the requirement reads input {j} of a node of {} inputs", ops.len())));
                }
                let target = PropertyExpr::min(vec![o.subject.clone(), o.against.clone()]);
                match known(&target) {
                    Some(t) => {
                        for (x, px) in [(j, pj.as_str()), (s, ps.as_str())] {
                            if ops[x].props.get(px) != Some(&t.0) || ops[x].bound.contains_key(px) {
                                let mut op = ops[x].clone();
                                self.fit_to(&mut op, px, t.clone())?;
                                ops[x] = op;
                            }
                        }
                    }
                    None => {
                        let (mut a, mut b) = (ops[j].clone(), ops[s].clone());
                        self.bind(&mut a, "ad_a", None, lets, used)?;
                        self.bind(&mut b, "ad_b", None, lets, used)?;
                        let (la, lb) = (length_of("ad_a", &pj), length_of("ad_b", &ps));
                        let (Some(la), Some(lb)) = (la, lb) else { return Err(self.bug(format!("fit.pair pairs {pj} and {ps}, which no length reads"))) };
                        if !used.insert("ad_l".to_string()) {
                            return Err(self.bug("one node binds the reserved name ad_l twice".to_string()));
                        }
                        lets.push(format!("let ad_l = i64.min ({la}) ({lb})"));
                        self.fit_to(&mut a, &pj, (PropertyExpr::Unknown, "ad_l".to_string()))?;
                        self.fit_to(&mut b, &ps, (PropertyExpr::Unknown, "ad_l".to_string()))?;
                        ops[j] = a;
                        ops[s] = b;
                    }
                }
            }
            "fit.inner" => {
                let mut sides = vec![(s, ps.clone(), o.subject.clone())];
                if let Some((a, pa)) = against.clone() {
                    sides.push((a, pa, o.against.clone()));
                }
                let cols: Vec<usize> = (0..sides.len()).filter(|i| sides[*i].1 == "cols").collect();
                let target = match cols.as_slice() {
                    [_, _] => PropertyExpr::min(vec![o.subject.clone(), o.against.clone()]),
                    [0] => o.against.clone(),
                    [1] => o.subject.clone(),
                    _ => return Err(self.bug(format!("fit.inner's requirement {} {} {} names no inner dimension", q.subject, q.relation.symbol(), q.against))),
                };
                let t = known(&target).ok_or_else(|| self.bug(format!("fit.inner fits to {target}, which has no spelling")))?;
                for i in cols {
                    let x = sides[i].0;
                    if ops[x].props.get("cols") != Some(&t.0) {
                        let mut op = ops[x].clone();
                        self.fit_to(&mut op, "cols", t.clone())?;
                        ops[x] = op;
                    }
                }
            }
            other => return Err(self.bug(format!("the adapter {other} has no rendering at a node"))),
        }
        Ok(())
    }

    /// The node at `path` rendered (its children first), with its Futhark type. A node an ADF stands for renders as
    /// the ADF's call, except where the ADF itself is defined (`defining`).
    fn render(&mut self, n: &crate::typed_environment::Node, path: &mut Vec<u32>, defining: Option<&[u32]>) -> Result<Operand6, String> {
        use crate::typed_environment::{Choice, Substitution};
        if let Some(id) = self.adfs.iter().position(|p| p == path) {
            if defining != Some(path.as_slice()) {
                let (params, _) = self.adf_params(path)?;
                let call = std::iter::once(format!("adf_{id}")).chain(params.into_iter().map(|(name, _)| name)).collect::<Vec<_>>().join(" ");
                return Ok(Operand6 { text: call, ty: n.data_type, props: n.properties.clone(), bound: BTreeMap::new() });
            }
        }
        self.nodes += 1;
        let made = |text: String| Operand6 { text, ty: n.data_type, props: n.properties.clone(), bound: BTreeMap::new() };
        let constant = |r: &Self| -> Result<String, String> {
            let d = r.data_type(n.data_type)?;
            if d.constant.is_empty() {
                Err(r.bug(format!("a node of {} renders its constant, and it has none", d.name)))
            } else {
                Ok(d.constant.clone())
            }
        };
        if let Some(s) = n.substituted {
            let name = self.reqs.get(s.requirement).map(|q| q.adapter.clone()).ok_or_else(|| self.bug(format!("the substitution names requirement {}", s.requirement)))?;
            self.adapters.push((path.clone(), name));
            return match s.by {
                Substitution::Input(p) => {
                    let kid = n.kids.get(p as usize).ok_or_else(|| self.bug(format!("the substitution renders input {p}, which the node does not have")))?;
                    path.push(p);
                    let r = self.render(kid, path, defining);
                    path.pop();
                    r
                }
                Substitution::Constant => Ok(made(constant(self)?)),
            };
        }
        let sym = self.k.symbols.get(n.symbol as usize).ok_or_else(|| self.bug(format!("the tree names symbol {}", n.symbol)))?;
        match sym.kind {
            Kind6::Var => {
                return match n.choice {
                    Some(Choice::Variable(b)) => {
                        let v = n.environment.variables.iter().find(|v| v.binder == b).ok_or_else(|| self.bug(format!("{} names {b:?}, which is not in scope", sym.name)))?;
                        Ok(made(self.name_of(&n.environment, v)?))
                    }
                    Some(Choice::Constant) => Ok(made(constant(self)?)),
                    other => Err(self.bug(format!("the variable {} reads as {other:?}", sym.name))),
                };
            }
            Kind6::Const => {
                let Some(Choice::Dc(i)) = n.choice else { return Err(self.bug(format!("the constant {} reads as {:?}", sym.name, n.choice))) };
                let d = self.data_type(n.data_type)?;
                let lit = d.literals.get(i as usize).ok_or_else(|| self.bug(format!("{} literal {i} is past its pool of {}", sym.name, d.literals.len())))?;
                return Ok(made(format!("{lit}{}", d.name)));
            }
            Kind6::Terminal | Kind6::Function if n.kids.is_empty() => {
                if sym.spelling.contains('{') {
                    return Err(self.bug(format!("the leaf {}'s spelling {:?} holds a placeholder", sym.name, sym.spelling)));
                }
                return Ok(made(sym.spelling.clone()));
            }
            Kind6::Terminal | Kind6::Function => {}
        }
        let mut ops = Vec::with_capacity(n.kids.len());
        for (i, kid) in n.kids.iter().enumerate() {
            path.push(i as u32);
            let r = self.render(kid, path, defining);
            path.pop();
            ops.push(r?);
        }
        let (mut lets, mut used) = (Vec::new(), BTreeSet::new());
        for o in n.outcomes.iter().filter(|o| !o.holds) {
            self.adapt(o, &mut ops, &mut lets, &mut used)?;
            self.adapters.push((path.clone(), o.adapter.clone()));
        }
        if sym.spelling.starts_with("matmul ") {
            self.uses.insert("matmul");
        }
        let text = self.spell(sym, n, &ops)?;
        Ok(made(if lets.is_empty() { text } else { format!("{} in {text}", lets.join(" ")) }))
    }

    /// A row's spelling with its placeholders filled (master 6 README section 4): `{k}` operand k (a size bare, any
    /// other operand in parentheses: Futhark's application is greedy), `{k.vj}` the j-th variable argument k binds,
    /// `{k.type}` operand k's Futhark type, `{k.<property>}` its property.
    fn spell(&self, sym: &Symbol6, n: &crate::typed_environment::Node, ops: &[Operand6]) -> Result<String, String> {
        let mut out = String::with_capacity(sym.spelling.len() * 2);
        let mut rest = sym.spelling.as_str();
        while let Some(i) = rest.find('{') {
            out.push_str(&rest[..i]);
            let after = &rest[i + 1..];
            let j = after.find('}').ok_or_else(|| self.bug(format!("{}'s spelling has an unclosed placeholder", sym.name)))?;
            let ph = &after[..j];
            rest = &after[j + 1..];
            let (k, field) = ph.split_once('.').unwrap_or((ph, ""));
            let k: usize = k.parse().map_err(|_| self.bug(format!("{}'s spelling has the placeholder {{{ph}}}", sym.name)))?;
            let op = ops.get(k).ok_or_else(|| self.bug(format!("{}'s spelling reads operand {k} of {}", sym.name, ops.len())))?;
            match field {
                "" if self.data_type(op.ty)?.name == "size" => out.push_str(&op.text),
                "" => out.push_str(&format!("({})", op.text)),
                "type" => out.push_str(&self.type_text(op)?),
                f if f.starts_with('v') && f.len() > 1 && f[1..].bytes().all(|b| b.is_ascii_digit()) => {
                    let j: usize = f[1..].parse().map_err(|_| self.bug(format!("the placeholder {{{ph}}}")))?;
                    let kid = &n.kids[k];
                    let env = &kid.environment;
                    let bound: Vec<_> = env.variables.iter().filter(|v| env.depth > n.environment.depth && v.binder.depth == env.depth).collect();
                    let v = bound.get(j).ok_or_else(|| self.bug(format!("{}'s argument {k} binds {} variables, the spelling names the {j}th", sym.name, bound.len())))?;
                    out.push_str(&self.name_of(env, v)?);
                }
                p => out.push_str(&atom(&self.dim(op, p)?)),
            }
        }
        out.push_str(rest);
        Ok(out)
    }

    /// The parameters of the ADF at `path`, in order: (its name at the call, its Futhark type). First every size the
    /// subtree's properties or its free variables name (`(n: i64)`), in the root environment's order, so a later
    /// parameter's type may name it; then its free variables that are not sizes, in order of first use.
    fn adf_params(&self, path: &[u32]) -> Result<(Vec<(String, String)>, String), String> {
        let node = self.decoding.node(path).ok_or_else(|| self.bug(format!("no node at the ADF's path {path:?}")))?;
        let key = self.decoding.subtree_key(path).ok_or_else(|| self.bug(format!("no subtree at {path:?}")))?;
        let mut sizes: BTreeSet<String> = BTreeSet::new();
        fn names(e: &PropertyExpr, out: &mut BTreeSet<String>) {
            match e {
                PropertyExpr::Size(n) => {
                    out.insert(n.clone());
                }
                PropertyExpr::Sum(xs) | PropertyExpr::Min(xs) => xs.iter().for_each(|x| names(x, out)),
                _ => {}
            }
        }
        let mut stack = vec![node];
        while let Some(m) = stack.pop() {
            m.properties.values().for_each(|e| names(e, &mut sizes));
            m.outcomes.iter().for_each(|o| {
                names(&o.subject, &mut sizes);
                names(&o.against, &mut sizes);
            });
            stack.extend(m.kids.iter());
        }
        let size_ty = self.k.ty("size");
        let mut vars = Vec::new();
        for f in &key.free {
            f.properties.values().for_each(|e| names(e, &mut sizes));
            let v = node.environment.variables.iter().find(|v| v.binder == f.binder).ok_or_else(|| self.bug(format!("the ADF at {path:?} uses {:?}, which is not in its scope", f.binder)))?;
            let name = self.name_of(&node.environment, v)?;
            if Some(f.data_type) == size_ty {
                sizes.insert(name);
            } else {
                let op = Operand6 { text: name.clone(), ty: f.data_type, props: f.properties.clone(), bound: BTreeMap::new() };
                vars.push((name, self.type_text(&op)?));
            }
        }
        let order: Vec<&str> = self.k.variables.iter().map(|v| v.name.as_str()).collect();
        let mut sizes: Vec<String> = sizes.into_iter().collect();
        sizes.sort_by_key(|s| (order.iter().position(|o| o == s).unwrap_or(usize::MAX), s.clone()));
        let mut params: Vec<(String, String)> = sizes.into_iter().map(|s| (s, "i64".to_string())).collect();
        params.extend(vars);
        let d = self.data_type(node.data_type)?;
        if matches!(d.kind, DataKind::Function { .. }) || d.name == "size" {
            return Err(self.bug(format!("no ADF can have the data type {}", d.name)));
        }
        let result = self.type_text(&Operand6 { text: String::new(), ty: node.data_type, props: node.properties.clone(), bound: BTreeMap::new() })?;
        Ok((params, result))
    }
}

/// `length` of a bound name read as the property `p`: the outer length for `len` and `rows`, the inner for `cols`.
fn length_of(name: &str, p: &str) -> Option<String> {
    match p {
        "len" | "rows" => Some(format!("length {name}")),
        "cols" => Some(format!("length (transpose {name})")),
        _ => None,
    }
}

fn value_is_unknown(e: Option<&PropertyExpr>) -> bool {
    e.is_none_or(PropertyExpr::has_unknown)
}

impl Kingdom6 {
    /// THE PROGRAM (pass 5, plan 7 task F2) of a decoded tree: the prelude (the `matmul` helper, `fit`, `clampix`: each
    /// only if the program renders it), then the entry `entry main [dims] (size inputs: i32) (inputs) : result =`, the
    /// size inputs rebound as `i64`, the zero arrays of the root environment bound once, and the rendered tree with the
    /// entry's own requirements' adapters (`fit.root`). A program with no failing requirement renders exactly as master
    /// 5 renders it.
    pub fn program6(&self, header: &Header6, decoding: &crate::typed_environment::Decoding) -> Result<Program6, String> {
        self.program6_with_adfs(header, decoding, &[])
    }

    /// [`Kingdom6::program6`] with the subtrees at `adfs` (paths from the root) rendered as ADFs: each a top-level
    /// `def adf_<i>` (its sizes, then its free variables, as parameters; deepest first, so a definition follows every
    /// ADF it calls) and its node as the call `adf_<i> <arguments>`. The text computes what the program without ADFs
    /// computes.
    pub fn program6_with_adfs(&self, header: &Header6, decoding: &crate::typed_environment::Decoding, adfs: &[Vec<u32>]) -> Result<Program6, String> {
        let multi = self.data_types.iter().filter_map(|d| if let DataKind::Array { element } = d.kind { Some(element) } else { None }).filter(|e| self.data_type(*e).is_some_and(|d| d.kind == DataKind::Scalar)).collect::<BTreeSet<_>>().len() > 1;
        let reqs: Vec<&Requirement> = self.symbols.iter().flat_map(|s| s.requires.iter()).collect();
        let mut r = Render6 { k: self, decoding, reqs, multi, adfs, uses: BTreeSet::new(), adapters: Vec::new(), nodes: 0 };
        if let Some(p) = adfs.iter().find(|p| p.is_empty()) {
            return Err(r.bug(format!("the root ({p:?}) cannot be an ADF of its own program")));
        }
        let mut order: Vec<usize> = (0..adfs.len()).collect();
        order.sort_by_key(|i| (std::cmp::Reverse(adfs[*i].len()), adfs[*i].clone()));
        let mut definitions = String::new();
        for i in order {
            let path = &adfs[i];
            let node = decoding.node(path).ok_or_else(|| r.bug(format!("no node at the ADF's path {path:?}")))?;
            let body = r.render(node, &mut path.clone(), Some(path))?;
            let (params, result) = r.adf_params(path)?;
            let params: String = params.iter().map(|(n, t)| format!(" ({n}: {t})")).collect();
            definitions.push_str(&format!("def adf_{i}{params} : {result} =\n  {}\n\n", body.text));
        }
        let mut root = r.render(&decoding.root, &mut Vec::new(), None)?;
        for o in decoding.root_outcomes.iter().filter(|o| !o.holds) {
            if o.adapter != "fit.root" {
                return Err(r.bug(format!("the entry's requirement names the adapter {}, not fit.root", o.adapter)));
            }
            let q = self.root.requires.get(o.requirement).ok_or_else(|| r.bug(format!("the entry has no requirement {}", o.requirement)))?;
            let target = futhark_size(&o.against).ok_or_else(|| r.bug(format!("the entry's requirement fits to {}, which has no spelling", o.against)))?;
            r.fit_to(&mut root, &q.property, (o.against.clone(), target))?;
            r.adapters.push((Vec::new(), o.adapter.clone()));
        }
        let mut text = String::new();
        if r.uses.contains("matmul") {
            text.push_str(&format!("{}\n", crate::futhark::MATMUL_HELPER));
        }
        if r.uses.contains("fit") {
            text.push_str(&format!("{}\n", crate::futhark::FIT_HELPER));
        }
        if r.uses.contains("clampix") {
            text.push_str(&format!("{}\n", crate::futhark::CLAMPIX_HELPER));
        }
        text.push_str(&definitions);
        text.push_str(&self.entry6(header, &root.text)?);
        Ok(Program6 { text, nodes: r.nodes, adapters: r.adapters })
    }

    /// The entry point around a rendered expression: `entry <name> [dims] (k: i32) (xs: [n]i32) : <result> =`, the size
    /// inputs rebound, the zero arrays bound, then `in <expr>` (the expression alone when nothing is bound).
    fn entry6(&self, header: &Header6, expr: &str) -> Result<String, String> {
        let size_text = |e: &PropertyExpr| -> Result<String, String> {
            let t = futhark_size(e).ok_or_else(|| bug6(&self.kingdom, format!("the entry declares the size {e}, which has no spelling")))?;
            Ok(if header.size_inputs.contains(&t) { format!("i64.i32 {t}") } else { t })
        };
        let type_of = |ty: Ty, props: &[(String, PropertyExpr)]| -> Result<String, String> {
            let d = self.data_type(ty).ok_or_else(|| bug6(&self.kingdom, format!("{ty:?} is no data type")))?;
            let mut t = ty;
            let mut s = String::new();
            for (p, _) in &d.properties {
                let e = props.iter().find(|(q, _)| q == p).map(|(_, e)| e).ok_or_else(|| bug6(&self.kingdom, format!("the entry's {} has no property {p}", d.name)))?;
                s.push_str(&format!("[{}]", size_text(e)?));
            }
            while let Some(DataType6 { kind: DataKind::Array { element }, .. }) = self.data_type(t) {
                t = *element;
            }
            s.push_str(&self.data_type(t).map_or_else(|| d.name.clone(), |e| e.name.clone()));
            Ok(s)
        };
        let inputs: Vec<&RootVariable6> = self.variables.iter().filter(|v| v.kind == "input").collect();
        let mut dims: Vec<String> = Vec::new();
        for v in &inputs {
            for (_, e) in &v.properties {
                let t = futhark_size(e).ok_or_else(|| bug6(&self.kingdom, format!("the input {} has the size {e}", v.name)))?;
                if !header.size_inputs.contains(&t) && !dims.contains(&t) {
                    dims.push(t);
                }
            }
        }
        let mut s = format!("entry {} {}", self.root.name, dims.iter().map(|d| format!("[{d}]")).collect::<String>());
        for k in &header.size_inputs {
            s.push_str(&format!(" ({k}: i32)"));
        }
        for v in &inputs {
            s.push_str(&format!(" ({}: {})", v.name, type_of(v.data_type, &v.properties)?));
        }
        s.push_str(&format!(" : {} =\n", type_of(self.root.data_type, &self.root.properties)?));
        let mut lets = String::new();
        for k in &header.size_inputs {
            lets.push_str(&format!("  let {k} = i64.i32 {k}\n"));
        }
        for v in self.variables.iter().filter(|v| v.kind == "zeros") {
            let len = v.properties.iter().find(|(p, _)| p == "len").and_then(|(_, e)| futhark_size(e)).ok_or_else(|| bug6(&self.kingdom, format!("the zero array {} has no length", v.name)))?;
            let mut t = v.data_type;
            while let Some(DataType6 { kind: DataKind::Array { element }, .. }) = self.data_type(t) {
                t = *element;
            }
            let zero = self.data_type(t).map(|d| d.constant.clone()).ok_or_else(|| bug6(&self.kingdom, format!("the zero array {} has no element", v.name)))?;
            lets.push_str(&format!("  let {} = replicate {len} {zero}\n", v.name));
        }
        s.push_str(&lets);
        s.push_str(&format!("  {}{expr}\n", if lets.is_empty() { "" } else { "in " }));
        Ok(s)
    }

    /// THE RENDERER of this kingdom: its tables, contract and header, prepared once ([`Renderer6::program`]).
    pub fn renderer6(&self, header: Header6) -> Result<Renderer6<'_>, String> {
        Ok(Renderer6 { kingdom: self, tables: self.tables()?, contract: self.contract()?, header, entry: crate::typed_environment::Entry::of_kingdom6(self) })
    }
}

/// A kingdom's renderer, prepared once: passes 2 to 5 of an expressed tree.
#[derive(Debug, Clone)]
pub struct Renderer6<'a> {
    pub kingdom: &'a Kingdom6,
    pub tables: KingdomTables,
    pub contract: KingdomContract,
    pub header: Header6,
    pub entry: crate::typed_environment::Entry,
}

impl Renderer6<'_> {
    /// The decoding of a tree (passes 2 to 4).
    pub fn decode(&self, tree: &Tree) -> Result<crate::typed_environment::Decoding, String> {
        crate::typed_environment::Decoder::new(&self.tables, &self.contract).map_err(|d| d.join("\n"))?.decode(&self.entry, tree)
    }

    /// The program of a tree (passes 2 to 5).
    pub fn program(&self, tree: &Tree) -> Result<Program6, String> {
        self.kingdom.program6(&self.header, &self.decode(tree)?)
    }

    /// The program of a tree with the subtrees at `adfs` as ADFs ([`Kingdom6::program6_with_adfs`]).
    pub fn program_with_adfs(&self, tree: &Tree, adfs: &[Vec<u32>]) -> Result<Program6, String> {
        self.kingdom.program6_with_adfs(&self.header, &self.decode(tree)?, adfs)
    }
}

/// The symbol id of a row by name, for the loader's tests.
pub fn symbol_id(k: &FutharkKingdom, name: &str) -> Option<SymbolId> {
    k.rows.iter().position(|r| r.name == name).map(|i| i as SymbolId)
}

#[cfg(test)]
mod tests6;
