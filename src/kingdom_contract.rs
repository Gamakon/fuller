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
//! plus the derived `a_T`, `n_max` and tail lengths, or every defect found, each
//! spelled "our software has a bug: ...". There is no repair here: a kingdom
//! that fails does not load.

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
    /// `a_T`: the most arguments of type `T` any one function takes.
    a: BTreeMap<Ty, u32>,
    n_max: u32,
    terminals: BTreeMap<Ty, Vec<SymbolId>>,
    functions: BTreeMap<Ty, Vec<SymbolId>>,
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
        let mut a: BTreeMap<Ty, u32> = BTreeMap::new();
        let mut n_max = 0u32;
        for sig in &signature {
            n_max = n_max.max(sig.len() as u32);
            let mut count: BTreeMap<Ty, u32> = BTreeMap::new();
            for t in sig {
                *count.entry(*t).or_default() += 1;
                types.insert(*t);
            }
            for (t, c) in count {
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
                a,
                n_max,
                terminals,
                functions,
            })
        } else {
            Err(defects)
        }
    }
}

impl KingdomContract {
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

    /// `t_T = min(h*a_T + [T = R], h*(n_max - 1) + 1)`, the tail length of a
    /// gene rooted at `root` with head length `h`; zero for a type with no
    /// frame (`a_T = 0` and not the root).
    pub fn tail_len(&self, h: usize, ty: Ty, root: Ty) -> usize {
        let is_root = usize::from(ty == root);
        let a = self.a_t(ty) as usize;
        if a == 0 && is_root == 0 {
            return 0;
        }
        (h * a + is_root).min(self.leaf_bound(h))
    }

    /// The tail length when `ty` may be the root: the envelope a layout shared
    /// by genes of every root is sized to.
    pub fn envelope_tail_len(&self, h: usize, ty: Ty) -> usize {
        (h * self.a_t(ty) as usize + 1).min(self.leaf_bound(h))
    }

    /// `h*(n_max - 1) + 1`: the most leaves a head of `h` functions opens.
    fn leaf_bound(&self, h: usize) -> usize {
        h * (self.n_max as usize).saturating_sub(1) + 1
    }

    /// The types that have a tail in a gene rooted at `root`: the root first,
    /// then the others with `a_T > 0` by ascending code. This is the order of
    /// the tails inside a gene.
    pub fn frame_types(&self, root: Ty) -> Vec<Ty> {
        let mut v = vec![root];
        v.extend(self.types.iter().copied().filter(|t| *t != root && self.a_t(*t) > 0));
        v
    }

    /// The words of one gene rooted at `root`: head, every tail, and a Dc of
    /// the same length as the tail of each constant-capable type.
    pub fn gene_width(&self, h: usize, root: Ty) -> usize {
        let mut w = h;
        for t in self.frame_types(root) {
            let tl = self.tail_len(h, t, root);
            w += tl;
            if self.constant_capable(t) {
                w += tl;
            }
        }
        w
    }

    /// The words of a gene in the all-type envelope (every type sized as if it
    /// were the root), the conservative capacity of the WGSL kingdom.
    pub fn envelope_gene_width(&self, h: usize) -> usize {
        let mut w = h;
        for t in &self.types {
            if self.a_t(*t) == 0 && !self.roots.contains(t) {
                continue;
            }
            let tl = self.envelope_tail_len(h, *t);
            w += tl;
            if self.constant_capable(*t) {
                w += tl;
            }
        }
        w
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
fn wgsl_family_relations(k: &crate::wgsl::loader::WgslKingdom, kingdom: &str) -> KingdomRelations {
    let rows: Vec<(String, Vec<Ty>, Ty)> = {
        let mut per_name: BTreeMap<&str, usize> = BTreeMap::new();
        for r in &k.rows {
            *per_name.entry(r.name.as_str()).or_default() += 1;
        }
        k.rows.iter().map(|r| (r.semantic_id(&k.duals, per_name[r.name.as_str()]), r.inputs.clone(), r.output)).collect()
    };
    let roots: Vec<Ty> = k.duals.iter().map(crate::wgsl::loader::WgslDual::ty).collect();
    let constants: Vec<&str> = rows.iter().filter(|(n, ins, _)| ins.is_empty() && n.starts_with("literal.")).map(|(n, _, _)| n.as_str()).collect();
    relations_from_rows(kingdom, roots, &rows, &constants)
}

/// The older WGSL kingdom (`kingdoms/wgsl`, 83 duals).
#[cfg(feature = "wgsl")]
pub fn wgsl_relations() -> KingdomRelations {
    wgsl_family_relations(&crate::wgsl::loader::WgslKingdom::load(), crate::wgsl::loader::WGSL)
}

/// The live kingdom the v24 runs and the reader use (`kingdoms/safewgsl`).
#[cfg(feature = "wgsl")]
pub fn safewgsl_relations() -> KingdomRelations {
    wgsl_family_relations(&crate::wgsl::loader::WgslKingdom::load_safe(), crate::wgsl::loader::SAFEWGSL)
}

/// What a kingdom check reports, for the log: counts, `n_max`, the tail
/// lengths and the gene width at head length `h`.
pub fn report(r: &KingdomRelations, h: usize) -> String {
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
            let mut tails: BTreeMap<usize, usize> = BTreeMap::new();
            for t in c.types() {
                *tails.entry(c.envelope_tail_len(h, *t)).or_default() += 1;
            }
            let widths: BTreeSet<usize> = c.roots().iter().map(|rt| c.gene_width(h, *rt)).collect();
            format!(
                "{head}\n  check passed; {} types, n_max {}\n  envelope t_T at h={h} (length: types) {tails:?}\n  gene width at h={h}: {} per root, envelope {}",
                c.types().len(),
                c.n_max(),
                widths.iter().map(usize::to_string).collect::<Vec<_>>().join("/"),
                c.envelope_gene_width(h)
            )
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

    #[test]
    fn two_type_fixture_derives_a_t_n_max_and_t_t() {
        let c = fixture().check().expect("a valid kingdom");
        assert_eq!(c.n_max(), 2);
        assert_eq!((c.a_t(Ty::S), c.a_t(Ty::I)), (1, 1));
        // h = 3, R = S: t_S = min(3*1+1, 3*1+1) = 4, t_I = min(3, 4) = 3.
        assert_eq!(c.tail_len(3, Ty::S, Ty::S), 4);
        assert_eq!(c.tail_len(3, Ty::I, Ty::S), 3);
        assert_eq!(c.signature(0), [Ty::S, Ty::I], "the order is data");
        assert_eq!(c.frame_types(Ty::S), vec![Ty::S, Ty::I]);
        assert_eq!(c.gene_width(3, Ty::S), 3 + 4 + 3);
    }

    #[test]
    fn single_type_tail_is_the_karva_tail() {
        let c = sr_relations().check().expect("SR passes");
        for h in [1usize, 7, 32] {
            assert_eq!(c.tail_len(h, Ty::F, Ty::F), h * (c.n_max() as usize - 1) + 1);
        }
        assert_eq!(c.frame_types(Ty::F), vec![Ty::F]);
    }

    #[test]
    fn a_type_with_no_demand_has_no_frame() {
        let mut r = fixture();
        r.names.push("j".to_string());
        r.symbol_outputs.push((3, Ty::B));
        let c = r.check().expect("still valid");
        assert_eq!(c.a_t(Ty::B), 0);
        assert_eq!(c.tail_len(5, Ty::B, Ty::S), 0);
        assert_eq!(c.frame_types(Ty::S), vec![Ty::S, Ty::I]);
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
        // Pattern root, h = 32: 3 tails of 33 + 32 for I, Dc for Char and I.
        assert_eq!(c.tail_len(32, Ty::Pattern, Ty::Pattern), 33);
        assert_eq!(c.tail_len(32, Ty::I, Ty::Pattern), 32);
        assert!(c.constant_capable(Ty::Char) && c.constant_capable(Ty::I));
        assert!(!c.constant_capable(Ty::Pattern));
    }

    /// The WGSL rows hold one output per symbol and contiguous ordered
    /// signatures; what the check finds today is only duals with no terminal
    /// (the store and `array<T>` duals, CR section 9 item 3), which is fuller
    /// data work and is not hidden here.
    #[cfg(feature = "wgsl")]
    #[test]
    fn wgsl_defects_are_only_missing_terminals() {
        let d = wgsl_relations().check().expect_err("store and array duals have no terminal yet");
        assert!(d.iter().all(|m| m.contains("has no terminal")), "{d:?}");
    }

    /// The live kingdom (`kingdoms/safewgsl`): the check finds only duals with no
    /// terminal, as the older kingdom does (`docs/PLAN_wgsl_terminals.md`).
    #[cfg(feature = "wgsl")]
    #[test]
    fn safewgsl_defects_are_only_missing_terminals() {
        let d = safewgsl_relations().check().expect_err("129 duals have no terminal yet");
        assert!(d.iter().all(|m| m.contains("has no terminal")), "{d:?}");
        assert_eq!(d.len(), 129, "the plan lists 129 duals; a change here changes the plan");
    }

    #[test]
    fn the_kingdom_reports_print() {
        let mut out = vec![report(&sr_relations(), 32), report(&regex_relations(), 32)];
        #[cfg(feature = "wgsl")]
        out.push(report(&wgsl_relations(), 32));
        #[cfg(feature = "wgsl")]
        out.push(report(&safewgsl_relations(), 32));
        for o in out {
            println!("{o}");
        }
    }
}
