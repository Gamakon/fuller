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

/// The symbol id of a row by name, for the loader's tests.
pub fn symbol_id(k: &FutharkKingdom, name: &str) -> Option<SymbolId> {
    k.rows.iter().position(|r| r.name == name).map(|i| i as SymbolId)
}
