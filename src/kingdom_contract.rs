//! The kingdom contract of multi-typed GEP (`nucleotable/docs/MultiTypedGEP.md`
//! section 1): two ordered relations and the check that makes a kingdom fit
//! to host genes.
//!
//! * `symbol_outputs(symbol, output_type)` over functions AND terminals:
//!   exactly one row per symbol. A polymorphic spelling is one symbol per
//!   output type.
//! * `argument_signature(symbol, arg_index, required_type)`, primary key
//!   `(symbol, arg_index)`: the ORDER of a call's arguments is data. The
//!   many-hot [`crate::geneframe::Arity`] cannot hold it (it is keyed by type).
//!
//! [`KingdomRelations::check`] returns a [`KingdomContract`], the relations
//! plus the derived `a_{T,U}`, `a_T`, `n_U` and `n_max`, or every defect found,
//! each spelled "our software has a bug: ...". There is no repair here: a
//! kingdom that fails does not load.
//!
//! The lengths of a gene are not kingdom data: they belong to a layout
//! (`(layout_id, gene_id)`). [`KingdomContract::heads_from_program_size`] and
//! [`KingdomContract::bounds`] derive them from a root type and either a
//! declared program size `N` or explicit head lengths `h_T`, with the exact
//! per-producer tail bound of the design (section 2).

use std::collections::{BTreeMap, BTreeSet};

use crate::geneframe::{regex_rows, Symbol, SymbolTable, Ty, REGEX};

/// A symbol's dense index into a kingdom's `names`.
pub type SymbolId = u32;

/// The two relations as loaded, before any check. Anything can be wrong here
/// (a symbol with two outputs, a gap in the argument indices); that is what
/// [`KingdomRelations::check`] is for.
#[derive(Debug, Clone)]
pub struct KingdomRelations {
    pub kingdom: String,
    /// The types a gene of this kingdom may be rooted at.
    pub roots: Vec<Ty>,
    /// Symbol names; the index is the [`SymbolId`].
    pub names: Vec<String>,
    /// Terminals that are the constant placeholder `?_T` of their type.
    pub constants: BTreeSet<SymbolId>,
    pub symbol_outputs: Vec<(SymbolId, Ty)>,
    pub argument_signature: Vec<(SymbolId, u32, Ty)>,
    /// A root no function asks for has no tail cell (`P_R = 1 = h_R`, design section 2.2), so it needs no terminal
    /// when it has a function: WGSL's `kernel.safe` (section 9, hole 5). False for every other kingdom.
    pub tail_free_roots: bool,
}

/// A kingdom that passed the check, with what the layout needs derived from it.
#[derive(Debug, Clone)]
pub struct KingdomContract {
    kingdom: String,
    roots: Vec<Ty>,
    names: Vec<String>,
    output: Vec<Ty>,
    signature: Vec<Vec<Ty>>,
    constant: Vec<bool>,
    /// Every type that occurs, ascending [`Ty::code`].
    types: Vec<Ty>,
    /// `a_{T,U}`, keyed `(T, U)`: the most arguments of type `T` any one
    /// function outputting `U` takes. Absent means zero.
    a_tu: BTreeMap<(Ty, Ty), u32>,
    /// `a_T = max_U a_{T,U}`.
    a: BTreeMap<Ty, u32>,
    /// `n_U`: the largest arity among functions outputting `U`. Absent means
    /// there are none, and counts as one.
    n_u: BTreeMap<Ty, u32>,
    n_max: u32,
    terminals: BTreeMap<Ty, Vec<SymbolId>>,
    functions: BTreeMap<Ty, Vec<SymbolId>>,
    /// The relations' [`KingdomRelations::tail_free_roots`], kept so a contract derived from this one (an epoch
    /// alphabet) is checked by the same rule.
    tail_free_roots: bool,
}

fn bug(what: String) -> String {
    format!("our software has a bug: {what}")
}

impl KingdomRelations {
    /// The kingdom check of the design: one output per symbol, ordered
    /// contiguous signatures, a terminal for every root and every required
    /// type, then `a_T` and `n_max`. Returns every defect, not the first.
    pub fn check(&self) -> Result<KingdomContract, Vec<String>> {
        let mut defects = Vec::new();
        let n = self.names.len();
        let kd = &self.kingdom;

        let mut outputs: Vec<Vec<Ty>> = vec![Vec::new(); n];
        for (s, ty) in &self.symbol_outputs {
            match outputs.get_mut(*s as usize) {
                Some(v) => v.push(*ty),
                None => defects.push(bug(format!("{kd}: symbol_outputs names symbol {s}, which is not a symbol (only {n})"))),
            }
        }
        for (s, outs) in outputs.iter().enumerate() {
            if outs.len() != 1 {
                defects.push(bug(format!(
                    "{kd}: symbol {} has {} output rows {outs:?}, want exactly one",
                    self.names[s],
                    outs.len()
                )));
            }
        }

        let mut args: Vec<BTreeMap<u32, Vec<Ty>>> = vec![BTreeMap::new(); n];
        for (s, i, ty) in &self.argument_signature {
            match args.get_mut(*s as usize) {
                Some(m) => m.entry(*i).or_default().push(*ty),
                None => defects.push(bug(format!("{kd}: argument_signature names symbol {s}, which is not a symbol (only {n})"))),
            }
        }
        let mut signature: Vec<Vec<Ty>> = vec![Vec::new(); n];
        for (s, m) in args.iter().enumerate() {
            for (i, tys) in m {
                if tys.len() != 1 {
                    defects.push(bug(format!(
                        "{kd}: symbol {} argument {i} has {} signature rows, the key (symbol, arg_index) is unique",
                        self.names[s],
                        tys.len()
                    )));
                }
            }
            for (want, got) in m.keys().enumerate() {
                if want as u32 != *got {
                    defects.push(bug(format!(
                        "{kd}: symbol {} argument indices {:?} are not contiguous from zero",
                        self.names[s],
                        m.keys().collect::<Vec<_>>()
                    )));
                    break;
                }
            }
            signature[s] = m.values().filter_map(|t| t.first().copied()).collect();
        }

        let output: Vec<Ty> = outputs.iter().map(|o| o.first().copied().unwrap_or(Ty::S)).collect();
        let mut constant = vec![false; n];
        for c in &self.constants {
            match constant.get_mut(*c as usize) {
                Some(slot) => {
                    *slot = true;
                    if !signature[*c as usize].is_empty() {
                        defects.push(bug(format!("{kd}: constant placeholder {} takes arguments", self.names[*c as usize])));
                    }
                }
                None => defects.push(bug(format!("{kd}: constants names symbol {c}, which is not a symbol (only {n})"))),
            }
        }

        let mut terminals: BTreeMap<Ty, Vec<SymbolId>> = BTreeMap::new();
        let mut functions: BTreeMap<Ty, Vec<SymbolId>> = BTreeMap::new();
        let mut types: BTreeSet<Ty> = BTreeSet::new();
        for s in 0..n {
            if outputs[s].len() != 1 {
                continue;
            }
            types.insert(output[s]);
            let book = if signature[s].is_empty() { &mut terminals } else { &mut functions };
            book.entry(output[s]).or_default().push(s as SymbolId);
        }
        let mut a_tu: BTreeMap<(Ty, Ty), u32> = BTreeMap::new();
        let mut a: BTreeMap<Ty, u32> = BTreeMap::new();
        let mut n_u: BTreeMap<Ty, u32> = BTreeMap::new();
        let mut n_max = 0u32;
        for (s, sig) in signature.iter().enumerate() {
            n_max = n_max.max(sig.len() as u32);
            let mut count: BTreeMap<Ty, u32> = BTreeMap::new();
            for t in sig {
                *count.entry(*t).or_default() += 1;
                types.insert(*t);
            }
            if sig.is_empty() || outputs[s].len() != 1 {
                continue;
            }
            let u = output[s];
            let e = n_u.entry(u).or_default();
            *e = (*e).max(sig.len() as u32);
            for (t, c) in count {
                let e = a_tu.entry((t, u)).or_default();
                *e = (*e).max(c);
                let e = a.entry(t).or_default();
                *e = (*e).max(c);
            }
        }

        if self.roots.is_empty() {
            defects.push(bug(format!("{kd}: no root type")));
        }
        let mut demanded: BTreeSet<Ty> = self.roots.iter().copied().collect();
        demanded.extend(a.keys().copied());
        for t in demanded {
            // A root no function asks for has `P_R = 1 = h_R`, so no tail cell of its type exists (design section
            // 2.2); its one head cell holds a function, so it needs no terminal (WGSL's `kernel.safe`, section 9
            // hole 5). A root some function does ask for still needs one.
            let tail_free_root = self.tail_free_roots && self.roots.contains(&t) && !a.contains_key(&t) && functions.get(&t).is_some_and(|v| !v.is_empty());
            if tail_free_root {
                continue;
            }
            if terminals.get(&t).is_none_or(|v| v.is_empty()) {
                defects.push(bug(format!("{kd}: type {t:?} is demanded (a root or a required type) and has no terminal")));
            }
        }

        if defects.is_empty() {
            Ok(KingdomContract {
                kingdom: self.kingdom.clone(),
                roots: self.roots.clone(),
                names: self.names.clone(),
                output,
                signature,
                constant,
                types: types.into_iter().collect(),
                a_tu,
                a,
                n_u,
                n_max,
                terminals,
                functions,
                tail_free_roots: self.tail_free_roots,
            })
        } else {
            Err(defects)
        }
    }
}

impl KingdomContract {
    /// Whether the check exempted a root no function asks for from having a terminal.
    pub fn tail_free_roots(&self) -> bool {
        self.tail_free_roots
    }
    pub fn kingdom(&self) -> &str {
        &self.kingdom
    }
    pub fn roots(&self) -> &[Ty] {
        &self.roots
    }
    pub fn n_symbols(&self) -> usize {
        self.names.len()
    }
    pub fn name(&self, s: SymbolId) -> &str {
        &self.names[s as usize]
    }
    /// The symbol's one output type.
    pub fn output(&self, s: SymbolId) -> Ty {
        self.output[s as usize]
    }
    /// The ordered `required_type` of each argument; empty for a terminal.
    pub fn signature(&self, s: SymbolId) -> &[Ty] {
        &self.signature[s as usize]
    }
    pub fn is_terminal(&self, s: SymbolId) -> bool {
        self.signature[s as usize].is_empty()
    }
    /// A constant placeholder `?_T`.
    pub fn is_constant(&self, s: SymbolId) -> bool {
        self.constant[s as usize]
    }
    /// Every type that occurs, ascending [`Ty::code`].
    pub fn types(&self) -> &[Ty] {
        &self.types
    }
    /// `a_T`, zero for a type no function asks for.
    pub fn a_t(&self, ty: Ty) -> u32 {
        self.a.get(&ty).copied().unwrap_or(0)
    }
    /// The most arguments any function takes, zero if the kingdom has none.
    pub fn n_max(&self) -> u32 {
        self.n_max
    }
    /// The terminals that output `ty`.
    pub fn terminals_of(&self, ty: Ty) -> &[SymbolId] {
        self.terminals.get(&ty).map_or(&[], Vec::as_slice)
    }
    /// The functions that output `ty`.
    pub fn functions_of(&self, ty: Ty) -> &[SymbolId] {
        self.functions.get(&ty).map_or(&[], Vec::as_slice)
    }
    /// A type with a constant placeholder terminal.
    pub fn constant_capable(&self, ty: Ty) -> bool {
        self.terminals_of(ty).iter().any(|s| self.constant[*s as usize])
    }

    /// `a_{T,U}`: the most arguments of type `ty` any one function outputting
    /// `producer` takes; zero if none asks for it.
    pub fn a_tu(&self, ty: Ty, producer: Ty) -> u32 {
        self.a_tu.get(&(ty, producer)).copied().unwrap_or(0)
    }
    /// `n_U`: the largest arity among functions outputting `ty`; one if there
    /// are none (such a type contributes no leaves beyond its slot).
    pub fn n_u(&self, ty: Ty) -> u32 {
        self.n_u.get(&ty).copied().unwrap_or(1)
    }

    /// `reachable(R)`: `root` first, then every type reachable from it through
    /// the signatures of functions whose output is reachable, ascending code.
    pub fn reachable(&self, root: Ty) -> Vec<Ty> {
        let mut seen: BTreeSet<Ty> = BTreeSet::from([root]);
        let mut queue = vec![root];
        while let Some(u) = queue.pop() {
            for f in self.functions_of(u) {
                for t in self.signature(*f) {
                    if seen.insert(*t) {
                        queue.push(*t);
                    }
                }
            }
        }
        let mut v = vec![root];
        v.extend(seen.into_iter().filter(|t| *t != root));
        v
    }

    /// `a_T` over the producers reachable from `root`: what a gene rooted at
    /// `root` can actually be asked for. Producers no slot can reach are never
    /// expressed, so they do not widen the head.
    fn a_reachable(&self, ty: Ty, reach: &[Ty]) -> u32 {
        reach.iter().map(|u| self.a_tu(ty, *u)).max().unwrap_or(0)
    }

    /// The head lengths of a layout that holds every program of at most `n`
    /// functions rooted at `root`: `h_T = [T = R] + (N - 1) * a_T` over the
    /// types reachable from `root` (design section 2.1; `a_T` is taken over the
    /// reachable producers).
    pub fn heads_from_program_size(&self, root: Ty, n: usize) -> Result<BTreeMap<Ty, usize>, String> {
        self.check_root(root)?;
        if n == 0 {
            return Err(bug(format!("{}: a layout needs a program size of at least one function", self.kingdom)));
        }
        let reach = self.reachable(root);
        Ok(reach.iter().map(|t| (*t, usize::from(*t == root) + (n - 1) * self.a_reachable(*t, &reach) as usize)).collect())
    }

    /// The largest program size `N` the heads guarantee: the greatest `N` with
    /// `heads_from_program_size(root, N) <= heads` type by type; zero when the
    /// root head is empty.
    pub fn guaranteed_program_size(&self, root: Ty, heads: &BTreeMap<Ty, usize>) -> usize {
        let reach = self.reachable(root);
        let head = |t: Ty| heads.get(&t).copied().unwrap_or(0);
        if head(root) == 0 {
            return 0;
        }
        let mut n_minus_1 = usize::MAX;
        for t in &reach {
            let a = self.a_reachable(*t, &reach) as usize;
            if a > 0 {
                n_minus_1 = n_minus_1.min(head(*t).saturating_sub(usize::from(*t == root)) / a);
            }
        }
        if n_minus_1 == usize::MAX {
            1
        } else {
            n_minus_1 + 1
        }
    }

    fn check_root(&self, root: Ty) -> Result<(), String> {
        if self.roots.contains(&root) {
            Ok(())
        } else {
            Err(bug(format!("{}: type {root:?} is not a root of this kingdom", self.kingdom)))
        }
    }

    /// The bound table of a gene rooted at `root` with head lengths `heads`
    /// (design section 2.2): per reachable type `P_T = [T = R] + sum_U h_U *
    /// a_{T,U}`, `L = 1 + sum_U h_U * (n_U - 1)`, tail `t_T = max(0, min(P_T,
    /// L + h_T) - h_T)` and Dc `d_T = min(P_T, L)`. A head for a type the root
    /// cannot reach is a defect.
    pub fn bounds(&self, root: Ty, heads: &BTreeMap<Ty, usize>) -> Result<GeneBounds, String> {
        self.check_root(root)?;
        let reach = self.reachable(root);
        if let Some((t, h)) = heads.iter().find(|(t, h)| **h > 0 && !reach.contains(t)) {
            return Err(bug(format!("{}: a head of {h} for type {t:?}, which root {root:?} cannot reach", self.kingdom)));
        }
        let head = |t: Ty| heads.get(&t).copied().unwrap_or(0);
        let l = 1 + reach.iter().map(|u| head(*u) * (self.n_u(*u) as usize - 1)).sum::<usize>();
        let types = reach
            .iter()
            .map(|t| {
                let p = usize::from(*t == root) + reach.iter().map(|u| head(*u) * self.a_tu(*t, *u) as usize).sum::<usize>();
                let h = head(*t);
                TypeBound { ty: *t, h, p, t: p.min(l + h).saturating_sub(h), d: p.min(l), constant: self.constant_capable(*t) }
            })
            .collect();
        Ok(GeneBounds { root, l, types })
    }
}

/// One type's lengths in a gene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeBound {
    pub ty: Ty,
    /// `h_T`, the head length.
    pub h: usize,
    /// `P_T`, every `T` slot any expressed function can open (plus the root).
    pub p: usize,
    /// `t_T`, the tail length.
    pub t: usize,
    /// `d_T = min(P_T, L)`, the most `T` leaves, hence the length of `Dc_T`
    /// when `T` is constant-capable.
    pub d: usize,
    /// `T` has a constant placeholder `?_T`, so the gene holds a `Dc_T` of `d_T`.
    pub constant: bool,
}

/// The bound table of one gene: `L` and a [`TypeBound`] per reachable type,
/// the root first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneBounds {
    pub root: Ty,
    /// `L`, the most leaves any expressed program can have.
    pub l: usize,
    pub types: Vec<TypeBound>,
}

impl GeneBounds {
    pub fn get(&self, ty: Ty) -> Option<&TypeBound> {
        self.types.iter().find(|b| b.ty == ty)
    }
    /// The words of the gene: every head, tail and the Dc of each
    /// constant-capable type.
    pub fn width(&self) -> usize {
        self.types.iter().map(|b| b.h + b.t + if b.constant { b.d } else { 0 }).sum()
    }
}

// ---------------------------------------------------------------------------
// The shipped kingdoms
// ---------------------------------------------------------------------------

/// Relations over a `SymbolTable` kingdom whose every argument is one type
/// (`ty`): the order of such a signature is not in question.
fn single_typed_functions(table: &SymbolTable, kingdom: &str, ty: Ty) -> Vec<(String, Vec<Ty>, Ty)> {
    table
        .kingdom(kingdom)
        .into_iter()
        .map(|s: &Symbol| (s.semantic_id.clone(), vec![ty; s.arity.total_in() as usize], ty))
        .collect()
}

/// Build relations from `(name, ordered inputs, output)` rows plus the names
/// of the constant placeholders among them.
pub fn relations_from_rows(kingdom: &str, roots: Vec<Ty>, rows: &[(String, Vec<Ty>, Ty)], constants: &[&str]) -> KingdomRelations {
    let mut r = KingdomRelations {
        kingdom: kingdom.to_string(),
        roots,
        names: Vec::new(),
        constants: BTreeSet::new(),
        symbol_outputs: Vec::new(),
        argument_signature: Vec::new(),
        tail_free_roots: false,
    };
    for (name, ins, out) in rows {
        let id = r.names.len() as SymbolId;
        r.names.push(name.clone());
        r.symbol_outputs.push((id, *out));
        for (i, t) in ins.iter().enumerate() {
            r.argument_signature.push((id, i as u32, *t));
        }
        if ins.is_empty() && constants.contains(&name.as_str()) {
            r.constants.insert(id);
        }
    }
    r
}

/// The Symbolic Regression kingdom: the 26 `Math` ops over `F`, plus the two
/// terminals a gene needs, a variable `x` and the constant placeholder `?`.
pub fn sr_relations() -> KingdomRelations {
    let mut rows = single_typed_functions(&crate::geneframe::master_table(), "Symbolic Regression", Ty::F);
    rows.push(("x".to_string(), Vec::new(), Ty::F));
    rows.push(("?".to_string(), Vec::new(), Ty::F));
    relations_from_rows("Symbolic Regression", vec![Ty::F], &rows, &["?"])
}

/// The regex kingdom: the ordered rows of [`regex_rows`], plus the constant
/// placeholders `?_Char` and `?_I` that the `Char` and `I` slots of
/// `lit`, `a-z`, `{n}` and `{0,k}` need terminals for.
pub fn regex_relations() -> KingdomRelations {
    let mut rows: Vec<(String, Vec<Ty>, Ty)> = regex_rows().iter().map(|(sem, _, ins, out)| (sem.to_string(), ins.to_vec(), *out)).collect();
    rows.push(("?_Char".to_string(), Vec::new(), Ty::Char));
    rows.push(("?_I".to_string(), Vec::new(), Ty::I));
    relations_from_rows(REGEX, vec![Ty::Pattern], &rows, &["?_Char", "?_I"])
}

/// The WGSL kingdoms' rows with their ordered inputs, every dual a possible
/// root. A `literal.*` terminal is the constant placeholder of its dual.
#[cfg(feature = "wgsl")]
fn wgsl_family_relations(k: &crate::wgsl::loader::WgslKingdom, kingdom: &str, roots: Option<&[&str]>) -> KingdomRelations {
    let rows: Vec<(String, Vec<Ty>, Ty)> = {
        let mut per_name: BTreeMap<&str, usize> = BTreeMap::new();
        for r in &k.rows {
            *per_name.entry(r.name.as_str()).or_default() += 1;
        }
        k.rows.iter().map(|r| (r.semantic_id(&k.duals, per_name[r.name.as_str()]), r.inputs.clone(), r.output)).collect()
    };
    let roots: Vec<Ty> = match roots {
        Some(names) => k.duals.iter().filter(|d| names.contains(&d.dual.as_str())).map(crate::wgsl::loader::WgslDual::ty).collect(),
        None => k.duals.iter().map(crate::wgsl::loader::WgslDual::ty).collect(),
    };
    let constants: Vec<&str> = rows.iter().filter(|(n, ins, _)| ins.is_empty() && (n.starts_with("literal.") || n.starts_with("const."))).map(|(n, _, _)| n.as_str()).collect();
    relations_from_rows(kingdom, roots, &rows, &constants)
}

/// The older WGSL kingdom (`kingdoms/wgsl`, 83 duals): a program's root is `kernel.kernel`.
#[cfg(feature = "wgsl")]
pub fn wgsl_relations() -> KingdomRelations {
    wgsl_family_relations(&crate::wgsl::loader::WgslKingdom::load(), crate::wgsl::loader::WGSL, Some(&["kernel.kernel"]))
}

/// The live kingdom the v24 runs and the reader use (`kingdoms/safewgsl`). A
/// program's root is `kernel.safe`, or `kernel.module` for several entry
/// points; the other genes of a chromosome are homeotic definitions, referenced
/// through a terminal of their own dual.
#[cfg(feature = "wgsl")]
pub fn safewgsl_relations() -> KingdomRelations {
    wgsl_family_relations(&crate::wgsl::loader::WgslKingdom::load_safe(), crate::wgsl::loader::SAFEWGSL, Some(&["kernel.safe", "kernel.module"]))
}

/// The TSR kingdom of `variant` over `n_inputs` columns (`kingdoms/tsr`, plan F3): every row with its
/// ordered input duals, the constant placeholder `?@<dual>` at each dual, and the one ROOT dual (the only
/// dual of the single kingdom) as the program's root.
pub fn tsr_relations(variant: crate::tsr_kingdom::TsrVariant, n_inputs: usize) -> Result<KingdomRelations, String> {
    let k = crate::tsr_kingdom::TsrKingdom::load(variant, n_inputs)?;
    let root = k.root().ok_or_else(|| format!("our software has a bug: {} has no root dual", variant.dir()))?;
    let rows: Vec<(String, Vec<Ty>, Ty)> = k.rows.iter().map(|r| (r.name.clone(), r.inputs.clone(), r.output)).collect();
    let constants: Vec<&str> = k.rows.iter().filter(|r| r.kind == crate::tsr_kingdom::TsrKind::Const).map(|r| r.name.as_str()).collect();
    Ok(relations_from_rows(&format!("{}/{}", crate::tsr_kingdom::TSR, variant.dir()), vec![root], &rows, &constants))
}

/// What a kingdom check reports, for the log: counts, `n_max`, and for each
/// root the bound table at program size `n` (heads from the design's
/// `h_T = [T = R] + (N - 1) * a_T`): `L`, the gene width and the widest types.
pub fn report(r: &KingdomRelations, n: usize) -> String {
    let functions = r.argument_signature.iter().map(|(s, _, _)| *s).collect::<BTreeSet<_>>().len();
    let head = format!("{}: {} symbols ({} functions, {} terminals), {} roots", r.kingdom, r.names.len(), functions, r.names.len() - functions, r.roots.len());
    match r.check() {
        Err(d) => {
            let mut per_fn: BTreeMap<SymbolId, u32> = BTreeMap::new();
            for (s, _, _) in &r.argument_signature {
                *per_fn.entry(*s).or_default() += 1;
            }
            let n_max = per_fn.values().copied().max().unwrap_or(0);
            let mut s = format!("{head}\n  CHECK FAILED, {} defects (n_max {n_max} from the rows)", d.len());
            for line in d.iter().take(3) {
                s.push_str(&format!("\n    {line}"));
            }
            s
        }
        Ok(c) => {
            let mut s = format!("{head}\n  check passed; {} types, n_max {}, program size N = {n}", c.types().len(), c.n_max());
            for root in c.roots() {
                let table = c.heads_from_program_size(*root, n).and_then(|h| c.bounds(*root, &h).map(|b| (h, b)));
                match table {
                    Err(e) => s.push_str(&format!("\n  root {root:?}: {e}")),
                    Ok((heads, b)) => {
                        let n_back = c.guaranteed_program_size(*root, &heads);
                        s.push_str(&format!("\n  root {root:?}: {} types reachable, L = {}, gene width {} words, heads guarantee N = {n_back}", b.types.len(), b.l, b.width()));
                        let mut widest: Vec<&TypeBound> = b.types.iter().collect();
                        widest.sort_by_key(|t| std::cmp::Reverse((t.h + t.t + if t.constant { t.d } else { 0 }, t.ty.code())));
                        for t in widest.iter().take(6) {
                            s.push_str(&format!("\n    {:?}: h {} P {} t {} d {}{}", t.ty, t.h, t.p, t.t, t.d, if t.constant { "" } else { " (no Dc)" }));
                        }
                        if widest.len() > 6 {
                            s.push_str(&format!("\n    ... {} more types", widest.len() - 6));
                        }
                    }
                }
            }
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> KingdomRelations {
        // The design's two-type fixture: left:S(S,I), terminals s:S and i:I.
        relations_from_rows(
            "two-type",
            vec![Ty::S],
            &[
                ("left".to_string(), vec![Ty::S, Ty::I], Ty::S),
                ("s".to_string(), vec![], Ty::S),
                ("i".to_string(), vec![], Ty::I),
            ],
            &[],
        )
    }

    fn heads(pairs: &[(Ty, usize)]) -> BTreeMap<Ty, usize> {
        pairs.iter().copied().collect()
    }

    /// The design's worked figures (section 3): h_S = 2, h_I = 1, root S gives
    /// P_S = 3, P_I = 2, L = 3, t_S = t_I = 1, d_S = 3, d_I = 2.
    #[test]
    fn two_type_fixture_derives_the_designs_bound_table() {
        let c = fixture().check().expect("a valid kingdom");
        assert_eq!(c.n_max(), 2);
        assert_eq!((c.a_t(Ty::S), c.a_t(Ty::I)), (1, 1));
        assert_eq!((c.a_tu(Ty::S, Ty::S), c.a_tu(Ty::I, Ty::S), c.a_tu(Ty::S, Ty::I), c.n_u(Ty::S), c.n_u(Ty::I)), (1, 1, 0, 2, 1));
        assert_eq!(c.signature(0), [Ty::S, Ty::I], "the order is data");
        assert_eq!(c.reachable(Ty::S), vec![Ty::S, Ty::I]);
        assert_eq!(c.reachable(Ty::I), vec![Ty::I], "an I gene reaches only I");
        let h = c.heads_from_program_size(Ty::S, 2).expect("heads");
        assert_eq!(h, heads(&[(Ty::S, 2), (Ty::I, 1)]));
        let b = c.bounds(Ty::S, &h).expect("bounds");
        let row = |t: Ty| b.get(t).map(|x| (x.h, x.p, x.t, x.d)).expect("type");
        assert_eq!((b.l, row(Ty::S), row(Ty::I)), (3, (2, 3, 1, 3), (1, 2, 1, 2)));
        assert_eq!(b.width(), 2 + 1 + 1 + 1, "no constants, so no Dc");
        assert_eq!(c.guaranteed_program_size(Ty::S, &h), 2);
    }

    /// One type: t = d = h (n_max - 1) + 1, the physical Karva tail, and
    /// h = 1 + (N - 1) n_max.
    #[test]
    fn single_type_bound_is_the_karva_tail() {
        let c = sr_relations().check().expect("SR passes");
        for h in [1usize, 7, 32] {
            let b = c.bounds(Ty::F, &heads(&[(Ty::F, h)])).expect("bounds");
            let f = b.get(Ty::F).expect("F");
            assert_eq!((f.t, f.d, f.p, b.l), (h * (c.n_max() as usize - 1) + 1, h * (c.n_max() as usize - 1) + 1, h * c.n_max() as usize + 1, h * (c.n_max() as usize - 1) + 1));
            assert_eq!(b.width(), h + 2 * f.t);
        }
        for n in 1..40usize {
            let h = c.heads_from_program_size(Ty::F, n).expect("heads");
            assert_eq!(h[&Ty::F], 1 + (n - 1) * c.n_max() as usize);
            assert_eq!(c.guaranteed_program_size(Ty::F, &h), n);
        }
    }

    /// The explicit heads report the largest N they guarantee.
    #[test]
    fn explicit_heads_report_the_program_size_they_guarantee() {
        let c = fixture().check().expect("valid");
        let n = |s, i| c.guaranteed_program_size(Ty::S, &heads(&[(Ty::S, s), (Ty::I, i)]));
        assert_eq!((n(0, 9), n(1, 0), n(2, 1), n(2, 0), n(5, 4), n(5, 3), n(9, 9)), (0, 1, 2, 1, 5, 4, 9));
    }

    #[test]
    fn a_head_for_an_unreachable_type_or_a_root_outside_the_kingdom_is_a_defect() {
        let c = fixture().check().expect("valid");
        assert!(c.bounds(Ty::S, &heads(&[(Ty::S, 2), (Ty::B, 1)])).expect_err("unreachable").contains("cannot reach"));
        assert!(c.heads_from_program_size(Ty::I, 3).expect_err("I is not a root").starts_with("our software has a bug: "));
        assert!(c.heads_from_program_size(Ty::S, 0).is_err());
    }

    /// A type no function asks for has no frame in the table.
    #[test]
    fn a_type_with_no_demand_is_not_reachable() {
        let mut r = fixture();
        r.names.push("j".to_string());
        r.symbol_outputs.push((3, Ty::B));
        let c = r.check().expect("still valid");
        assert_eq!(c.a_t(Ty::B), 0);
        assert_eq!(c.reachable(Ty::S), vec![Ty::S, Ty::I]);
    }

    /// a_T is taken over the producers a root can reach: a producer no slot
    /// reaches never widens the head.
    #[test]
    fn an_unreachable_producer_does_not_widen_the_head() {
        // wide:B(I,I,I,I) is never reached from S; left:S(S,I) is.
        let r = relations_from_rows(
            "r",
            vec![Ty::S, Ty::B],
            &[
                ("left".to_string(), vec![Ty::S, Ty::I], Ty::S),
                ("wide".to_string(), vec![Ty::I, Ty::I, Ty::I, Ty::I], Ty::B),
                ("s".to_string(), vec![], Ty::S),
                ("i".to_string(), vec![], Ty::I),
                ("b".to_string(), vec![], Ty::B),
            ],
            &[],
        );
        let c = r.check().expect("valid");
        assert_eq!(c.a_t(Ty::I), 4);
        assert_eq!(c.heads_from_program_size(Ty::S, 3).expect("heads"), heads(&[(Ty::S, 3), (Ty::I, 2)]));
        assert_eq!(c.heads_from_program_size(Ty::B, 3).expect("heads")[&Ty::I], 8);
    }

    fn expect_defect(r: &KingdomRelations, needle: &str) {
        let d = r.check().expect_err("the check must fail");
        assert!(d.iter().all(|m| m.starts_with("our software has a bug: ")), "{d:?}");
        assert!(d.iter().any(|m| m.contains(needle)), "no defect mentions {needle:?}: {d:?}");
    }

    #[test]
    fn a_symbol_with_two_outputs_fails() {
        let mut r = fixture();
        r.symbol_outputs.push((1, Ty::I));
        expect_defect(&r, "2 output rows");
    }

    #[test]
    fn a_symbol_with_no_output_fails() {
        let mut r = fixture();
        r.symbol_outputs.retain(|(s, _)| *s != 2);
        expect_defect(&r, "0 output rows");
    }

    #[test]
    fn a_gap_in_the_argument_indices_fails() {
        let mut r = fixture();
        r.argument_signature = vec![(0, 0, Ty::S), (0, 2, Ty::I)];
        expect_defect(&r, "not contiguous from zero");
    }

    #[test]
    fn a_repeated_argument_index_fails() {
        let mut r = fixture();
        r.argument_signature.push((0, 1, Ty::S));
        expect_defect(&r, "unique");
    }

    #[test]
    fn a_demanded_type_without_a_terminal_fails() {
        let mut r = fixture();
        r.symbol_outputs.retain(|(s, _)| *s != 2);
        r.names.truncate(2);
        expect_defect(&r, "no terminal");
        let mut root_only = fixture();
        root_only.roots = vec![Ty::B];
        expect_defect(&root_only, "no terminal");
    }

    #[test]
    fn a_root_no_function_asks_for_needs_no_terminal() {
        // WGSL's kernel.safe: one function produces the root and nothing consumes it, so t_R = 0 and no terminal exists.
        let rows = vec![("top".to_string(), vec![Ty::S], Ty::B), ("s".to_string(), Vec::new(), Ty::S)];
        let mut r = relations_from_rows("tail-free root", vec![Ty::B], &rows, &[]);
        r.tail_free_roots = true;
        let k = r.check().expect("a root nothing asks for needs no terminal");
        let heads = k.heads_from_program_size(Ty::B, 8).unwrap();
        assert_eq!(k.bounds(Ty::B, &heads).unwrap().get(Ty::B).unwrap().t, 0, "the root has no tail cell");
        // the moment a function asks for the root type it needs one
        let mut asked = rows.clone();
        asked.push(("wrap".to_string(), vec![Ty::B], Ty::S));
        let mut a = relations_from_rows("asked root", vec![Ty::B], &asked, &[]);
        a.tail_free_roots = true;
        expect_defect(&a, "no terminal");
        expect_defect(&relations_from_rows("tail-free root off", vec![Ty::B], &rows, &[]), "no terminal");
    }

    #[test]
    fn a_constant_with_arguments_fails() {
        let mut r = fixture();
        r.constants.insert(0);
        expect_defect(&r, "takes arguments");
    }

    #[test]
    fn an_unknown_symbol_in_a_relation_fails() {
        let mut r = fixture();
        r.argument_signature.push((9, 0, Ty::S));
        expect_defect(&r, "not a symbol");
    }

    #[test]
    fn sr_passes() {
        let c = sr_relations().check().expect("SR passes");
        assert_eq!(c.n_max(), 2);
        assert_eq!(c.types(), [Ty::F]);
        assert!(c.constant_capable(Ty::F));
    }

    #[test]
    fn regex_keeps_the_argument_order_the_table_loses() {
        let c = regex_relations().check().expect("regex passes");
        let rep_n = (0..c.n_symbols() as SymbolId).find(|s| c.name(*s) == "regex_rep_n").expect("regex_rep_n");
        assert_eq!(c.signature(rep_n), [Ty::Pattern, Ty::I]);
        assert_eq!(c.a_t(Ty::Pattern), 2);
        assert_eq!(c.a_t(Ty::CharClass), 2);
        assert_eq!(c.a_t(Ty::Char), 2);
        assert_eq!(c.a_t(Ty::I), 1);
        assert_eq!(c.n_max(), 2);
        assert!(c.constant_capable(Ty::Char) && c.constant_capable(Ty::I));
        assert!(!c.constant_capable(Ty::Pattern));
    }

    /// The older WGSL kingdom (`kingdoms/wgsl`) passes the check, rooted at `kernel.kernel`: every demanded
    /// dual has a terminal (`docs/PLAN_wgsl_terminals.md`).
    #[cfg(feature = "wgsl")]
    #[test]
    fn wgsl_passes_the_kingdom_check() {
        let c = wgsl_relations().check().unwrap_or_else(|d| panic!("{} defects, first: {:?}", d.len(), d.first()));
        assert_eq!(c.roots().len(), 1);
    }

    /// The live kingdom (`kingdoms/safewgsl`) passes the check: every demanded
    /// dual has a terminal (`docs/PLAN_wgsl_terminals.md`).
    #[cfg(feature = "wgsl")]
    #[test]
    fn safewgsl_passes_the_kingdom_check() {
        let c = safewgsl_relations().check().unwrap_or_else(|d| panic!("{} defects, first: {:?}", d.len(), d.first()));
        assert_eq!(c.roots().len(), 2);
    }

    /// Prints the bound tables the design asks for (SR, regex, both WGSL
    /// kingdoms, TSR single, order and order_depth) at program sizes 8, 17 and 32: `cargo test ... the_kingdom_reports_print -- --nocapture`.
    #[test]
    fn the_kingdom_reports_print() {
        for n in [8usize, 17, 28, 32] {
            let mut out = vec![report(&sr_relations(), n), report(&regex_relations(), n)];
            for v in [crate::tsr_kingdom::TsrVariant::Single, crate::tsr_kingdom::TsrVariant::Order, crate::tsr_kingdom::TsrVariant::OrderDepth] {
                out.push(report(&tsr_relations(v, 8).expect("TSR tables load"), n));
            }
            #[cfg(feature = "wgsl")]
            out.push(report(&wgsl_relations(), n));
            #[cfg(feature = "wgsl")]
            out.push(report(&safewgsl_relations(), n));
            for o in out {
                println!("{o}");
            }
        }
    }
}
