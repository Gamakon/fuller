//! The WGSL kingdom as the runners and the optimisers read it: [`TrueKingdom::from_rows`] lays the rows of
//! `crate::wgsl_rows` (the gene frame's fixed schema, `kingdoms/master/`) out as `TrueRow`s, symbol `s` of the contract
//! row `s`, each data type a [`WgslDual`], each Dc pool a `C_T`. Rendering, the relations and the work bound read the
//! rows ([`TrueKingdom::render`], [`TrueKingdom::relations`], [`TrueKingdom::tree_work_bound`]).
//!
//! The parts the 305-type table carried (fuel strata read from type names, loop rows, solo phases, a counting-sort
//! scaffold, a prologue) are empty for the rows; the methods that read them return what an empty table gives. A
//! random typed gene ([`random_gene`]) and the substitution of argument texts into a spelling ([`substitute`]) are
//! here too: spelling templates `{N}` the text of argument N, `{N:F}` field F of that text split on whitespace,
//! `{v}` the value text a constant reads from its `C_T`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::geneframe::{PropertyExpr, Role, Ty};
use crate::kingdom_contract::{relations_from_rows, relations_from_tables, KingdomContract, KingdomRelations};
use crate::typed_gep::{GeneLayout, Tree};
use crate::wgsl::work_guard::Fuel;


/// The kingdom's name in its relations.
pub const TRUE_WGSL: &str = "TRUE_WGSL";

/// A data type of the kingdom: `Ty::Wgsl(index)`, its name (`dual`; `slot` the same for the rows).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgslDual {
    pub index: u16,
    pub dual: String,
    pub slot: String,
}

impl WgslDual {
    pub fn ty(&self) -> Ty {
        Ty::Wgsl(self.index)
    }
}

/// The plain workgroup pool's words of the old tables' `pool.py` (`wg_pool: array<u32, P + 4>`), for a table whose
/// prologue declares no pool.
pub const P_WORDS: u64 = 6080;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    Function,
    Terminal,
    /// The placeholder `?_T` of its output type.
    Constant,
}

#[derive(Debug, Clone)]
pub struct TrueRow {
    pub name: String,
    pub kind: RowKind,
    pub out: Ty,
    pub args: Vec<Ty>,
    pub spelling: String,
    pub note: String,
    /// The declarations of the kernel-scope variables the row's spelling names (a register, a private array, a loop's
    /// counter and bound): its prologue fragment, emitted when the row is in the expressed tree ([`TrueKingdom::declarations_of`]).
    pub declares: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TrueKingdom {
    pub dir: PathBuf,
    pub types: Vec<WgslDual>,
    pub rows: Vec<TrueRow>,
    /// `C_T`: the value texts of each constant-capable type.
    pub constants: BTreeMap<Ty, Vec<String>>,
    pub prologue: String,
    pub declarations: String,
    pub root: Ty,
    /// The interface's fuel budget `K`.
    pub fuel_budget: u32,
    /// For every row, how it repeats the nodes below it ([`LoopRow`]); `None` for a row whose spelling has no loop.
    pub loops: Vec<Option<LoopRow>>,
    /// For every row, the argument one invocation of the workgroup runs ([`solo_body`]): the body of a solo phase
    /// (`wg.solo_*`, `wg.solo_at_*`), read from its spelling; `None` for every other row.
    pub solo: Vec<Option<usize>>,
    /// A module interface (s9 batch 8): the roots of its normal genes, one entry point each, in gene order; empty for
    /// a one-entry interface, whose genes are rooted at [`TrueKingdom::root`].
    pub gene_roots: Vec<Ty>,
    /// A module interface's fixed homeotic linker (design section 6): the link function whose signature is the gene
    /// roots in order and whose output is [`TrueKingdom::root`].
    pub linker: Option<String>,
    /// The counting-sort scaffold's variants (`kingdoms/wgsl_true/countsort.py`, docs/DESIGN_inrange_index_types.md
    /// section 10): variant name -> piece -> text, from `scaffold.tsv`; empty for a table without a scaffold.
    pub scaffold: BTreeMap<String, BTreeMap<String, String>>,
    /// THE KINGDOM AS ROWS of the gene frame's fixed schema ([`crate::wgsl_rows`], plan 7.7 task K3), when this
    /// kingdom was built from them ([`TrueKingdom::from_rows`]): the contract is the rows' Argument rows, and
    /// [`TrueKingdom::render`] is the decoder's rendering (Flow and Guard rows applied), not the substitution.
    pub rows_kingdom: Option<std::sync::Arc<crate::wgsl_rows::WgslRows>>,
}

/// How a loop row repeats its arguments, read from its own spelling and types.
///
/// `inside[i]`: argument `i` is spelled inside the loop (its placeholder comes after the row's first `loop {` or
/// `for (`), so it runs once per trip; an argument spelled before the loop (a bound, a stride bound by a `let`) runs
/// once. `cap` is the trip cap the types give: `2^(6c)` for the stratum class `c` the row raises (the output's fuel
/// stratum less the body's, or less the row's own `_f<k>` for a row with no body, the workgroup reductions); a tile
/// row whose bound argument (`u32.lit64`) is a `?_T` constant uses that literal `n` when it is below the cap. A row whose
/// loop holds no argument (a reduction) executes `cap` trips of its own text: its own work is `cap`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopRow {
    pub inside: Vec<bool>,
    pub cap: u128,
    /// The argument that is the literal trip bound (`u32.lit64`), if the row has one.
    pub literal_arg: Option<usize>,
    /// No argument is inside the loop: the row's own text is what repeats.
    pub own_work: bool,
}

/// `H`: the sum of the head lengths `h_T` over the types `root` reaches, at program size `n`.
pub fn head_total_of(contract: &KingdomContract, root: Ty, n: usize) -> Result<u64, String> {
    let heads = contract.heads_from_program_size(root, n)?;
    let b = contract.bounds(root, &heads)?;
    Ok(b.types.iter().map(|t| t.h as u64).sum())
}

impl TrueKingdom {
    /// THE KINGDOM FROM ITS ROWS (plan 7.7 task K3): every consumer of a `TrueKingdom` (the typed device's draws, the
    /// runner, the share pass, the optimisers) reads the alphabet as `TrueRow`s, so the rows kingdom's alphabet is laid
    /// out as one: symbol `s` of the contract is row `s` (a Dc constant a `Constant` row, a `var.T` terminal and a
    /// function of no inputs a `Terminal`), each data type a dual named as the row names it, `C_T` each pool's
    /// literals. Rendering, the relations and the work bound read the rows ([`TrueKingdom::render`],
    /// [`TrueKingdom::relations`], [`TrueKingdom::tree_work_bound`]); nothing is matched on a type name, so `loops` and
    /// `solo` are empty, and the fuel budget is the rows' `BUDGET` read as the old K = 1.
    pub fn from_rows(rows: crate::wgsl_rows::WgslRows) -> Result<TrueKingdom, String> {
        let types: Vec<WgslDual> = rows
            .tables
            .data_types
            .iter()
            .enumerate()
            .map(|(i, d)| WgslDual { index: i as u16, dual: d.name.clone(), slot: d.name.clone() })
            .collect();
        let n = rows.contract.n_symbols();
        let mut table_rows = Vec::with_capacity(n);
        let mut constants: BTreeMap<Ty, Vec<String>> = BTreeMap::new();
        for s in 0..n as crate::kingdom_contract::SymbolId {
            let row = rows.tables.symbol(i64::from(s)).ok_or_else(|| format!("our software has a bug: {}: contract symbol {s} has no row", crate::wgsl_rows::KINGDOM))?;
            let args = rows.contract.signature(s).to_vec();
            let out = rows.contract.output(s);
            let kind = match row.role {
                Role::Constant => {
                    constants.insert(out, rows.literals(s));
                    RowKind::Constant
                }
                Role::Variable => RowKind::Terminal,
                Role::Function if args.is_empty() => RowKind::Terminal,
                Role::Function => RowKind::Function,
                Role::Formula => return Err(format!("our software has a bug: {}: a formula row ({}) in the contract", crate::wgsl_rows::KINGDOM, row.symbol_name)),
            };
            table_rows.push(TrueRow { name: row.symbol_name.clone(), kind, out, args, spelling: row.spelling("").unwrap_or("").to_string(), note: String::new(), declares: Vec::new() });
        }
        if constants.len() != rows.const_len().len() {
            return Err(format!("our software has a bug: {}: two Dc constants of one data type", crate::wgsl_rows::KINGDOM));
        }
        Ok(TrueKingdom {
            dir: PathBuf::new(),
            root: rows.root(),
            fuel_budget: 1,
            loops: vec![None; n],
            solo: vec![None; n],
            gene_roots: Vec::new(),
            linker: None,
            types,
            rows: table_rows,
            constants,
            prologue: String::new(),
            declarations: String::new(),
            scaffold: BTreeMap::new(),
            rows_kingdom: Some(std::sync::Arc::new(rows)),
        })
    }

    /// The rows kingdom's work bound of one invocation: over the nodes the program renders (a dropped statement is not
    /// run), the product of the trip counts of the enclosing loops (the environment's `work`), a loop node counted
    /// once per trip of its own. Exact, since every trip count is a literal of the `count` type.
    fn rows_work_bound(rows: &crate::wgsl_rows::WgslRows, tree: &Tree) -> Result<u128, String> {
        let d = rows.decode(tree)?;
        let mut total = 0u128;
        for (_, n) in d.rendered_nodes() {
            let work = match n.environment.attribute("work") {
                Some(PropertyExpr::Literal(w)) if *w >= 0 => *w as u128,
                other => return Err(format!("our software has a bug: {}: a node's env.work is {other:?}, not a literal", crate::wgsl_rows::KINGDOM)),
            };
            let own = match (rows.name(n.symbol) == "loop", n.kids.first().and_then(|k| k.properties.get("value"))) {
                (true, Some(PropertyExpr::Literal(v))) if *v >= 0 => *v as u128,
                _ => 1,
            };
            total = total.saturating_add(work.saturating_mul(own));
        }
        Ok(total)
    }

    /// The exact work bound of one invocation of the kernel `tree` spells, from its types: the sum over the expressed
    /// nodes of the product of the trip caps of the loops enclosing each ([`LoopRow`]); a loop whose text holds no
    /// argument counts its own `cap` trips. Every legal tree obeys it, whatever its uniforms; it is never more than `H * 2^(6K)`. It is a
    /// property of the expressed tree (the work guard's split of a dispatch into submits), never a verdict on the gene.
    pub fn tree_work_bound(&self, tree: &Tree) -> Result<u128, String> {
        if let Some(rows) = &self.rows_kingdom {
            return TrueKingdom::rows_work_bound(rows, tree);
        }
        let mut total = 0u128;
        let mut stack: Vec<(&Tree, u128)> = vec![(tree, 1)];
        while let Some((node, mult)) = stack.pop() {
            let row = self.rows.get(node.symbol as usize).ok_or_else(|| format!("our software has a bug: symbol {} is not a row", node.symbol))?;
            match self.loops.get(node.symbol as usize).and_then(|l| l.as_ref()) {
                None => {
                    total = total.saturating_add(mult);
                    stack.extend(node.kids.iter().map(|k| (k, mult)));
                }
                Some(l) => {
                    let mut cap = l.cap;
                    if let Some(i) = l.literal_arg {
                        let kid = node.kids.get(i).ok_or_else(|| format!("our software has a bug: {} has no argument {i}", row.name))?;
                        let kid_row = &self.rows[kid.symbol as usize];
                        if let (RowKind::Constant, Some(c)) = (kid_row.kind, kid.const_index) {
                            let text = self.constants.get(&kid_row.out).and_then(|v| v.get(c as usize)).ok_or_else(|| format!("our software has a bug: {}: constant {c} is past C_T", kid_row.name))?;
                            let n: u128 = text.trim_end_matches('u').parse().map_err(|e| format!("our software has a bug: {}: the literal {text:?} is not a count: {e}", kid_row.name))?;
                            cap = cap.min(n);
                        }
                    }
                    total = total.saturating_add(mult.saturating_mul(if l.own_work { cap.max(1) } else { 1 }));
                    for (i, k) in node.kids.iter().enumerate() {
                        let inside = l.inside.get(i).copied().unwrap_or(false);
                        stack.push((k, if inside { mult.saturating_mul(cap) } else { mult }));
                    }
                }
            }
        }
        Ok(total)
    }

    /// [`TrueKingdom::tree_work_bound`] for one run whose uniform fields are known (`fields`: each field's value, by
    /// name): a loop's trips are its bound's value in this run (interval arithmetic over the bound's own rows: the
    /// fields, literals, `+ * min max >> / & -`, the cap `min(.., 2^c)`), divided by its literal stride, never more than
    /// the cap its types give. A bound whose value the rows do not give (a register, a load) keeps the cap. It is an
    /// upper bound of the node executions of one invocation of that run, never more than the type bound.
    pub fn tree_work_bound_in(&self, tree: &Tree, fields: &BTreeMap<String, u64>) -> Result<u128, String> {
        self.tree_work_bound_workgroup_in(tree, fields, 1)
    }

    /// The bound of ONE WORKGROUP of `invocations` invocations of the run [`TrueKingdom::tree_work_bound_in`] bounds:
    /// every node counts once an invocation, except inside the body of a solo phase (`wg.solo_*`, spelled
    /// `if (lidx == 0u) { .. }`), which one invocation of the workgroup runs, so it counts once. With one invocation it
    /// is [`TrueKingdom::tree_work_bound_in`]; it is never more than `invocations` times it.
    pub fn tree_work_bound_workgroup_in(&self, tree: &Tree, fields: &BTreeMap<String, u64>, invocations: u64) -> Result<u128, String> {
        let mut total = 0u128;
        let all = u128::from(invocations.max(1));
        let mut stack: Vec<(&Tree, u128)> = vec![(tree, all)];
        while let Some((node, mult)) = stack.pop() {
            self.rows.get(node.symbol as usize).ok_or_else(|| format!("our software has a bug: symbol {} is not a row", node.symbol))?;
            match self.loops.get(node.symbol as usize).and_then(|l| l.as_ref()) {
                None => {
                    total = total.saturating_add(mult);
                    // a solo phase's body (read from its spelling): one invocation of the workgroup runs it
                    let body = self.solo.get(node.symbol as usize).copied().flatten();
                    stack.extend(node.kids.iter().enumerate().map(|(i, k)| (k, if body == Some(i) { (mult / all).max(1) } else { mult })));
                }
                Some(l) => {
                    let mut cap = l.cap;
                    // the bound: the first argument spelled before the loop; the stride: a literal second argument before it
                    let outside: Vec<usize> = (0..node.kids.len()).filter(|i| !l.inside.get(*i).copied().unwrap_or(false)).collect();
                    if let Some(b) = outside.first().and_then(|i| node.kids.get(*i)) {
                        if let Some(n) = self.value_max(b, fields) {
                            let stride = outside.get(1).and_then(|i| node.kids.get(*i)).and_then(|s| self.value_min(s)).filter(|s| *s > 0).unwrap_or(1);
                            cap = cap.min(u128::from(n.div_ceil(stride)));
                        }
                    }
                    total = total.saturating_add(mult.saturating_mul(if l.own_work { cap.max(1) } else { 1 }));
                    for (i, k) in node.kids.iter().enumerate() {
                        let inside = l.inside.get(i).copied().unwrap_or(false);
                        stack.push((k, if inside { mult.saturating_mul(cap) } else { mult }));
                    }
                }
            }
        }
        Ok(total)
    }

    /// The loops of `tree` above the literal class (more than 64 trips by their type) whose trip bound the rows do not
    /// give a value for in a run whose fields are `fields` (the work guard then counts the type's cap, `2^c`), each as
    /// `<loop row> bounded by <the bound's rows>`. In an interface whose trip counts are bounded by its size
    /// fields (kingdoms/wgsl_true/loops.py `size_bounds`) there is none, whatever the gene.
    pub fn unbounded_loops(&self, tree: &Tree, fields: &BTreeMap<String, u64>) -> Vec<String> {
        let mut out = Vec::new();
        let mut stack = vec![tree];
        while let Some(node) = stack.pop() {
            if let Some(l) = self.loops.get(node.symbol as usize).and_then(|l| l.as_ref()) {
                let outside: Vec<usize> = (0..node.kids.len()).filter(|i| !l.inside.get(*i).copied().unwrap_or(false)).collect();
                let b = outside.first().and_then(|i| node.kids.get(*i));
                // a literal-bounded loop (`u32.lit64`, at most 64 trips by its type) is bounded whatever its literal
                if l.cap > 64 && b.is_none_or(|b| self.value_max(b, fields).is_none()) {
                    let name = |t: &Tree| self.rows.get(t.symbol as usize).map_or("?".to_string(), |r| r.name.clone());
                    out.push(format!("{} bounded by {}", name(node), b.map_or("nothing".to_string(), |b| b.level_order().iter().map(|(t, _)| name(t)).collect::<Vec<_>>().join(" "))));
                }
            }
            stack.extend(node.kids.iter());
        }
        out
    }

    /// The plain workgroup pool's words (`pool.py` P_WORDS, an interface's `pool_words`): its prologue declares
    /// `wg_pool: array<u32, P + 4>` (four guard words); [`P_WORDS`] for a table with no pool.
    pub fn pool_words(&self) -> u64 {
        let decl = self.prologue.lines().find_map(|l| l.trim().strip_prefix("var<workgroup> wg_pool: array<u32, ").and_then(|r| r.strip_suffix(">;")).and_then(|n| n.trim().parse::<u64>().ok()));
        decl.map_or(P_WORDS, |n| n.saturating_sub(4))
    }

    /// A constant node's value, when its `C_T` text is an unsigned integer.
    fn const_value(&self, node: &Tree) -> Option<u64> {
        let row = self.rows.get(node.symbol as usize)?;
        let text = self.constants.get(&row.out)?.get(node.const_index? as usize)?;
        text.trim().trim_end_matches('u').parse().ok()
    }

    fn value_min(&self, node: &Tree) -> Option<u64> {
        let row = self.rows.get(node.symbol as usize)?;
        match row.kind {
            RowKind::Constant => self.const_value(node),
            RowKind::Terminal => row.spelling.trim().trim_end_matches('u').parse().ok(),
            RowKind::Function if row.spelling == "{0}" || row.spelling == "lit_u32({0})" => self.value_min(node.kids.first()?),
            _ => None,
        }
    }

    /// The largest value a u32 node can take in a run whose fields are `fields`, from its rows' spellings; `None` when
    /// the rows do not say.
    fn value_max(&self, node: &Tree, fields: &BTreeMap<String, u64>) -> Option<u64> {
        let row = self.rows.get(node.symbol as usize)?;
        let sp = row.spelling.as_str();
        let kid = |i: usize| node.kids.get(i).and_then(|k| self.value_max(k, fields));
        let r = match row.kind {
            RowKind::Constant => self.const_value(node)?,
            // a field of the uniform block, by the interface's own name for it (`cfg.n`, `params.x`)
            RowKind::Terminal => match sp.split_once('.').filter(|(u, f)| !u.is_empty() && u.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') && fields.contains_key(*f)) {
                Some((_, f)) => *fields.get(f)?,
                None => sp.trim().trim_start_matches("lit_u32(").trim_end_matches(')').trim_end_matches('u').parse().ok()?,
            },
            RowKind::Function => match sp {
                "{0}" | "lit_u32({0})" => kid(0)?,
                "({0} + {1})" => kid(0)?.checked_add(kid(1)?)?,
                "({0} * {1})" => kid(0)?.checked_mul(kid(1)?)?,
                "({0} - {1})" | "({0} / {1})" | "({0} >> {1})" | "sub_floor({0}, {1})" => kid(0)?,
                "({0} & {1})" | "min({0}, {1})" => match (kid(0), kid(1)) {
                    (Some(a), Some(b)) => a.min(b),
                    (Some(a), None) | (None, Some(a)) => a,
                    _ => return None,
                },
                "max({0}, {1})" => kid(0)?.max(kid(1)?),
                o => {
                    // the bound rows' cap: `min({0}, Nu)`
                    let n = o.strip_prefix("min({0}, ").and_then(|t| t.strip_suffix("u)")).and_then(|t| t.parse::<u64>().ok())?;
                    kid(0).map_or(n, |a| a.min(n))
                }
            },
        };
        (r <= u64::from(u32::MAX)).then_some(r)
    }

    /// The work guard's fuel for one expressed tree ([`TrueKingdom::tree_work_bound`]).
    pub fn tree_fuel(&self, tree: &Tree) -> Result<Fuel, String> {
        Ok(Fuel::Tree { per_invocation: self.tree_work_bound(tree)? })
    }

    pub fn type_name(&self, t: Ty) -> String {
        match t {
            Ty::Wgsl(i) => self.types[i as usize].dual.clone(),
            o => format!("{o:?}"),
        }
    }

    /// The two relations of the kingdom contract; the symbol id is the row's position.
    pub fn relations(&self) -> KingdomRelations {
        if let Some(rows) = &self.rows_kingdom {
            return relations_from_tables(&rows.tables, vec![self.root]);
        }
        let rows: Vec<(String, Vec<Ty>, Ty)> = self.rows.iter().map(|r| (r.name.clone(), r.args.clone(), r.out)).collect();
        let constants: Vec<&str> = self.rows.iter().filter(|r| r.kind == RowKind::Constant).map(|r| r.name.as_str()).collect();
        let mut roots = vec![self.root];
        roots.extend(self.gene_roots.iter().copied());
        let mut r = relations_from_rows(TRUE_WGSL, roots, &rows, &constants);
        r.tail_free_roots = true;
        r
    }

    /// `|C_T|` for every constant-capable type.
    pub fn const_lens(&self) -> BTreeMap<Ty, u32> {
        self.constants.iter().map(|(t, v)| (*t, v.len() as u32)).collect()
    }

    /// `H`, the sum of the head lengths over the types the root reaches, at program size `n`.
    pub fn head_total(&self, contract: &KingdomContract, n: usize) -> Result<u64, String> {
        head_total_of(contract, self.root, n)
    }

    /// The work guard's fuel for kernels of this kingdom at program size `n`: `H` and the interface's `K`. Every
    /// device run of a kernel this kingdom produced passes it (`work_guard`).
    pub fn fuel(&self, contract: &KingdomContract, n: usize) -> Result<Fuel, String> {
        Ok(Fuel::Kingdom { h: self.head_total(contract, n)?, k: self.fuel_budget })
    }

    /// The bound table at program size `n` (design section 2): for every type the root reaches its head `h_T`, the
    /// demand `P_T`, the tail `t_T` and the Dc length `d_T`, the leaf bound `L` and the gene width; then the types
    /// the root cannot reach.
    pub fn bound_table(&self, contract: &KingdomContract, n: usize) -> Result<String, String> {
        let heads = contract.heads_from_program_size(self.root, n)?;
        let b = contract.bounds(self.root, &heads)?;
        let mut s = format!(
            "bound table at program size N = {n}: root {}, {} types reachable of {}, L = {}, gene width {} words (heads guarantee N = {})\n  {:<34} {:>5} {:>7} {:>7} {:>7}\n",
            self.type_name(self.root),
            b.types.len(),
            self.types.len(),
            b.l,
            b.width(),
            contract.guaranteed_program_size(self.root, &heads),
            "type",
            "h_T",
            "P_T",
            "t_T",
            "d_T"
        );
        let mut tys: Vec<_> = b.types.iter().collect();
        tys.sort_by_key(|t| (std::cmp::Reverse(t.h + t.t + if t.constant { t.d } else { 0 }), t.ty.code()));
        for t in &tys {
            s.push_str(&format!("  {:<34} {:>5} {:>7} {:>7} {:>7}{}\n", self.type_name(t.ty), t.h, t.p, t.t, t.d, if t.constant { "" } else { "  (no Dc)" }));
        }
        let reach: std::collections::BTreeSet<Ty> = b.types.iter().map(|t| t.ty).collect();
        let dead: Vec<String> = self.types.iter().map(|d| d.ty()).filter(|t| !reach.contains(t)).map(|t| self.type_name(t)).collect();
        s.push_str(&format!("  unreachable from the root ({}): {}\n", dead.len(), dead.join(" ")));
        Ok(s)
    }

    /// Spell an expressed tree: substitute each node's argument texts into its
    /// row's spelling, then put the declarations of the rows the tree holds in the
    /// entry point behind the prologue. Nothing else is done to the text.
    pub fn render(&self, tree: &Tree) -> Result<String, String> {
        if let Some(rows) = &self.rows_kingdom {
            return Ok(rows.program(tree)?.text);
        }
        self.resolve_scaffold(tree, format!("{}{}", self.prologue_for(tree)?, self.spell(tree)?))
    }

    /// The scaffold variant a module's tree chooses (section 10): the one knob `cs.<variant>` it names, with the
    /// variant's pieces; `None` for a tree that names none.
    pub fn scaffold_variant(&self, tree: &Tree) -> Result<Option<Variant<'_>>, String> {
        let mut found: Option<Variant<'_>> = None;
        let mut stack = vec![tree];
        while let Some(n) = stack.pop() {
            stack.extend(n.kids.iter());
            let Some(row) = self.rows.get(n.symbol as usize) else { continue };
            let Some((name, v)) = row.name.strip_prefix("cs.").and_then(|v| self.scaffold.get_key_value(v)) else { continue };
            if found.is_some_and(|(f, _)| f != name.as_str()) {
                return Err(format!("our software has a bug: a tree names two scaffold variants, {} and {name}", found.map_or("", |f| f.0)));
            }
            found = Some((name.as_str(), v));
        }
        Ok(found)
    }

    /// One entry's text (section 10): `node` spelled, its declarations, and its scaffold markers spelled from `variant`'s
    /// pieces; the reader compares a source's scatter entry with the kingdom's own expression of the cluster store.
    pub fn render_entry(&self, node: &Tree, variant: &BTreeMap<String, String>) -> Result<String, String> {
        let mut out = self.spell(node)?;
        for (piece, t) in variant {
            out = out.replace(&format!("<cs:{piece}>"), t);
        }
        Ok(out)
    }

    /// The compiler's part of expression (section 10): every `<cs:piece>` marker of `text` spelled as the piece of the
    /// variant the module's knob chooses, as `<declarations>` is spelled from the tree's rows. A text with markers and a
    /// tree with no knob is a bug.
    pub fn resolve_scaffold(&self, tree: &Tree, text: String) -> Result<String, String> {
        if !text.contains("<cs:") {
            return Ok(text);
        }
        let (name, v) = self.scaffold_variant(tree)?.ok_or_else(|| "our software has a bug: the text spells a scaffold marker and the tree names no scaffold variant".to_string())?;
        let mut out = text;
        for (piece, t) in v {
            out = out.replace(&format!("<cs:{piece}>"), t);
        }
        if let Some(at) = out.find("<cs:") {
            return Err(format!("our software has a bug: variant {name} has no piece for {}", &out[at..out.len().min(at + 30)]));
        }
        Ok(out)
    }

    /// The words of the plain pool `tree` declares (`kingdoms/wgsl_true/pool.py`, the pool's extent): the largest
    /// `u32.wgext` its entries name; `None` for a tree that names none (a table with no pool, or a tree with no entry).
    pub fn pool_extent(&self, tree: &Tree) -> Result<Option<u64>, String> {
        let mut out: Option<u64> = None;
        let mut stack = vec![tree];
        while let Some(n) = stack.pop() {
            stack.extend(n.kids.iter());
            // a share row (a symbol past the base rows) is a name or a binding, never an extent
            let Some(row) = self.rows.get(n.symbol as usize) else { continue };
            if self.type_name(row.out) == "u32.wgext" {
                let v = match n.const_index {
                    Some(_) => self.const_value(n),
                    None => row.spelling.trim().parse::<u64>().ok(),
                }
                .ok_or_else(|| format!("our software has a bug: {} gives u32.wgext no number", row.name))?;
                out = Some(out.map_or(v, |o| o.max(v)));
            }
        }
        Ok(out)
    }

    /// The prologue of `tree`'s kernel: the table's, with the pool's two lines (`const WG_PW: u32 = Pu;` and
    /// `var<workgroup> wg_pool: array<u32, P + 4>;`, P the table's largest extent) spelled at the tree's extent E
    /// ([`TrueKingdom::pool_extent`]). Every row bounds the pool by `WG_PW`, so every subscript stays below E + 4.
    pub fn prologue_for(&self, tree: &Tree) -> Result<String, String> {
        let Some(e) = self.pool_extent(tree)? else { return Ok(self.prologue.clone()) };
        let p = self.pool_words();
        if e == 0 || e % 4 != 0 || e > p {
            return Err(format!("our software has a bug: a pool extent of {e} words, outside 4..={p} in steps of 4"));
        }
        let (c_old, a_old) = (format!("const WG_PW: u32 = {p}u;\n"), format!("var<workgroup> wg_pool: array<u32, {}>;\n", p + 4));
        if !self.prologue.contains(&c_old) || !self.prologue.contains(&a_old) {
            return Err(format!("our software has a bug: the table names u32.wgext but its prologue does not declare WG_PW = {p}u and wg_pool of {} words", p + 4));
        }
        Ok(self.prologue.replacen(&c_old, &format!("const WG_PW: u32 = {e}u;\n"), 1).replacen(&a_old, &format!("var<workgroup> wg_pool: array<u32, {}>;\n", e + 4), 1))
    }

    /// The declarations a tree's kernel needs: the `declares` fragments of the rows of its nodes, deduplicated, in the
    /// table's order (`declarations.wgsl`). Pure row data: a variable is declared exactly when a row in the tree names
    /// it, zero-initialised as the table writes it. A node that is not a base row (a share row) declares nothing.
    pub fn declarations_of(&self, tree: &Tree) -> Result<String, String> {
        let mut named: BTreeSet<&str> = BTreeSet::new();
        let mut stack = vec![tree];
        while let Some(n) = stack.pop() {
            if let Some(r) = self.rows.get(n.symbol as usize) {
                named.extend(r.declares.iter().map(String::as_str));
            }
            stack.extend(n.kids.iter());
        }
        let out: Vec<&str> = self.declarations.lines().filter(|l| named.contains(l)).collect();
        if out.len() != named.len() {
            let missing: Vec<&&str> = named.iter().filter(|d| !out.contains(d)).collect();
            return Err(format!("our software has a bug: rows declare {missing:?}, which declarations.wgsl does not hold"));
        }
        Ok(out.join("\n"))
    }

    fn spell(&self, node: &Tree) -> Result<String, String> {
        let row = self.rows.get(node.symbol as usize).ok_or_else(|| format!("our software has a bug: symbol {} is not a row", node.symbol))?;
        if node.kids.len() != row.args.len() {
            return Err(format!("our software has a bug: {} has {} children, its signature {}", row.name, node.kids.len(), row.args.len()));
        }
        // a node fills an argument only with the type the argument asks (the walk never does otherwise; a tree built by
        // hand that does is not a tree of the kingdom, and has no text)
        for (i, (kid, want)) in node.kids.iter().zip(&row.args).enumerate() {
            if let Some(kr) = self.rows.get(kid.symbol as usize).filter(|kr| kr.out != *want) {
                return Err(format!("the tree is not one the types allow: argument {i} of {} is {}, and {} gives {}", row.name, self.type_name(*want), kr.name, self.type_name(kr.out)));
            }
        }
        let kids: Vec<String> = node.kids.iter().map(|k| self.spell(k)).collect::<Result<_, _>>()?;
        let value = match (row.kind, node.const_index) {
            (RowKind::Constant, Some(i)) => Some(
                self.constants
                    .get(&row.out)
                    .and_then(|v| v.get(i as usize))
                    .ok_or_else(|| format!("our software has a bug: {}: constant index {i} is past C_T of {}", row.name, self.type_name(row.out)))?
                    .as_str(),
            ),
            (RowKind::Constant, None) => return Err(format!("our software has a bug: {}: a placeholder without a Dc index", row.name)),
            _ => None,
        };
        let text = substitute(&row.spelling, &kids, value).map_err(|e| format!("our software has a bug: {}: {e}", row.name))?;
        // An entry point's declarations are those of the rows of its own subtree: a one-entry kernel's root holds the
        // whole tree, and each entry of a module (s9 batch 8) declares only what its own gene names.
        if row.spelling.contains("<declarations>") {
            return Ok(text.replace("<declarations>", &self.declarations_of(node)?));
        }
        Ok(text)
    }
}

/// A scaffold variant (section 10): its name and its pieces (piece -> text).
pub type Variant<'k> = (&'k str, &'k BTreeMap<String, String>);

/// `{N}`, `{N:F}` and `{v}` of a spelling; every other brace is copied.
pub fn substitute(spelling: &str, kids: &[String], value: Option<&str>) -> Result<String, String> {
    let b = spelling.as_bytes();
    let mut out = String::with_capacity(spelling.len() + kids.iter().map(String::len).sum::<usize>());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'{' {
            if let Some(close) = spelling[i..].find('}') {
                let inner = &spelling[i + 1..i + close];
                if inner == "v" {
                    out.push_str(value.ok_or("{v} in a row with no constant")?);
                    i += close + 1;
                    continue;
                }
                let (a, f) = match inner.split_once(':') {
                    Some((a, f)) => (a, Some(f)),
                    None => (inner, None),
                };
                if !a.is_empty() && a.bytes().all(|c| c.is_ascii_digit()) && f.is_none_or(|f| !f.is_empty() && f.bytes().all(|c| c.is_ascii_digit())) {
                    let n: usize = a.parse().map_err(|_| "argument index")?;
                    let text = kids.get(n).ok_or_else(|| format!("{{{inner}}} but only {} arguments", kids.len()))?;
                    match f {
                        None => out.push_str(text),
                        Some(f) => {
                            let f: usize = f.parse().map_err(|_| "field index")?;
                            out.push_str(text.split_whitespace().nth(f).ok_or_else(|| format!("{{{inner}}}: the argument has no field {f}: {text:?}"))?);
                        }
                    }
                    i += close + 1;
                    continue;
                }
            }
        }
        let ch = spelling[i..].chars().next().ok_or("utf8")?;
        out.push(ch);
        i += ch.len_utf8();
    }
    Ok(out)
}

/// splitmix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// How often a head cell draws a function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Share {
    /// A fair coin per cell.
    Half,
    /// In proportion to the functions against the terminals of the cell's type.
    Proportional,
}

/// The host typed initialiser (phylu `typed_init`, the same draws): a head cell
/// is a function or a terminal of its type, a tail cell a terminal of its type
/// drawn uniformly, a Dc cell an index below `|C_T|`; `root_function` makes the
/// root type's first head cell a function. Nothing is masked or redrawn.
pub fn random_gene(k: &KingdomContract, layout: &GeneLayout, const_len: &BTreeMap<Ty, u32>, rng: &mut Rng, share: Share, root_function: bool) -> Vec<u32> {
    use crate::typed_gep::Domain;
    let mut gene = vec![0u32; layout.width];
    for s in layout.spans() {
        let (functions, terminals) = (k.functions_of(s.ty), k.terminals_of(s.ty));
        for (i, cell) in gene[s.off..s.off + s.len].iter_mut().enumerate() {
            *cell = match s.domain {
                Domain::Head => {
                    let force = root_function && s.ty == layout.root && i == 0 && !functions.is_empty();
                    let p = match share {
                        Share::Half => 0.5,
                        Share::Proportional => functions.len() as f64 / (functions.len() + terminals.len()).max(1) as f64,
                    };
                    let function = !functions.is_empty() && (terminals.is_empty() || force || rng.unit() < p);
                    if function {
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

/// The depth-biased typed initialiser, for the compiler test only: the same draws as [`random_gene`], but a row is
/// chosen in proportion to `weight[row]` (a function among the functions of the cell's type, a terminal among its
/// terminals) and a head cell of a type in `function_p` draws a function with that probability. Nothing is masked or
/// redrawn: every cell still holds a symbol of its domain's type.
pub fn random_gene_weighted(k: &KingdomContract, layout: &GeneLayout, const_len: &BTreeMap<Ty, u32>, rng: &mut Rng, weight: &[f64], function_p: &BTreeMap<Ty, f64>, root_function: bool) -> Vec<u32> {
    use crate::typed_gep::Domain;
    fn cumulative(ids: &[u32], weight: &[f64]) -> Vec<f64> {
        let mut acc = 0.0;
        ids.iter()
            .map(|i| {
                acc += weight[*i as usize].max(1e-9);
                acc
            })
            .collect()
    }
    fn pick(ids: &[u32], cum: &[f64], rng: &mut Rng) -> u32 {
        let x = rng.unit() * cum.last().copied().unwrap_or(1.0);
        ids[cum.partition_point(|c| *c <= x).min(ids.len() - 1)]
    }
    let mut tables: BTreeMap<Ty, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for f in &layout.frames {
        tables.insert(f.ty, (cumulative(k.functions_of(f.ty), weight), cumulative(k.terminals_of(f.ty), weight)));
    }
    let mut gene = vec![0u32; layout.width];
    for s in layout.spans() {
        let (functions, terminals) = (k.functions_of(s.ty), k.terminals_of(s.ty));
        let (fcum, tcum) = &tables[&s.ty];
        for (i, cell) in gene[s.off..s.off + s.len].iter_mut().enumerate() {
            *cell = match s.domain {
                Domain::Head => {
                    let force = root_function && s.ty == layout.root && i == 0 && !functions.is_empty();
                    let p = function_p.get(&s.ty).copied().unwrap_or(0.5);
                    let function = !functions.is_empty() && (terminals.is_empty() || force || rng.unit() < p);
                    if function {
                        pick(functions, fcum, rng)
                    } else {
                        pick(terminals, tcum, rng)
                    }
                }
                Domain::Tail => pick(terminals, tcum, rng),
                Domain::Dc => rng.below(const_len.get(&s.ty).copied().unwrap_or(1).max(1) as usize) as u32,
                Domain::Dv => rng.below(crate::typed_gep::DV_WORDS as usize) as u32,
            };
        }
    }
    gene
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitution_copies_wgsl_braces_and_fills_arguments() {
        let kids = vec!["16 16".to_string(), "x".to_string()];
        assert_eq!(substitute("{ let r = {1}; w({0:1}) }", &kids, None).unwrap(), "{ let r = x; w(16) }");
        assert_eq!(substitute("lit_u32({v}u)", &[], Some("7")).unwrap(), "lit_u32(7u)");
        assert!(substitute("{2}", &kids, None).is_err());
        assert!(substitute("{v}", &[], None).is_err());
    }

    /// The bridge from the rows: one `TrueRow` per contract symbol, the rows' relations pass the kingdom check, and a
    /// random gene renders through the decoder with a work bound.
    #[test]
    fn the_rows_kingdom_lays_out_one_row_per_symbol_and_renders() {
        for v in [crate::wgsl_rows::VARIANT, crate::wgsl_rows::IMAGE] {
            let k = TrueKingdom::from_rows(crate::wgsl_rows::WgslRows::load_variant(v).expect("the rows")).expect("the bridge");
            let c = k.relations().check().unwrap_or_else(|d| panic!("{v}: {} defects", d.len()));
            assert_eq!(k.rows.len(), c.n_symbols(), "{v}");
            let layout = GeneLayout::from_program_size(&c, k.root, 8).expect("layout");
            let gene = random_gene(&c, &layout, &k.const_lens(), &mut Rng(7), Share::Half, true);
            let tree = crate::typed_gep::express(&c, &layout, &gene, &k.const_lens()).expect("expresses").tree();
            assert!(k.render(&tree).expect("renders").contains("fn main("), "{v}");
            assert!(k.tree_work_bound(&tree).expect("the work bound") >= 1, "{v}");
        }
    }
}
