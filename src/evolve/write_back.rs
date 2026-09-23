//! Snap, stage 4: THE WRITE-BACK. A snapped form that passed the guard goes back
//! INTO THE KARVA GENE, so the population carries the named constant from then
//! on — `hff_sr_engine.py::_apply_snap_to_winners`: "breeding mixes
//! named-constant tokens instead of float approximations". Named constants are
//! never drawable terminals: this is the only way one enters a gene.
//!
//! WHAT SNAP SEARCHES, in this engine. A gene's numbers are its CONSTANT
//! SUBTREES: arithmetic over "?" values (and named constants already carried)
//! FOLDS to one number — `22/7`, `1/(2*3)`. [`gene_form`] folds each maximal
//! constant subtree to a single `Num` in f64 by the crate's evaluator (the
//! linter's fold, `lint::engine::fold`), and OFFERS it to the match when it is
//! not a whole number: a lone "?" is a whole number already, a lone named
//! constant needs no re-snapping, and neither is offered. The fitted scale `a`
//! and offset `b` sit OUTSIDE the genes and are re-fitted at every evaluation, so
//! writing them into a gene would change nothing the next least-squares fit does
//! not undo: they are left to the reporting snap.
//!
//! THE GUARD JUDGES THE MODEL, not the lone gene (`_snap_op.py` compares the
//! whole individual's R² with the snapped gene substituted): [`model_form`] is
//! `a * WRAPPER(LINKER(genes)) + b` as ONE flat expression, the examined gene's
//! folded constants offered and every other literal withheld
//! (`SnapGuard::run_resident_offered`). A model longer than the evaluator's SLOT
//! is skipped, and counted.
//!
//! THE WRITE-BACK ITSELF, [`write_back`]: each site's constant subtree is replaced
//! by its lattice entry's template (from the HOST's f64-confirmed hits — never
//! from the device's verdict block), in gene symbols, and the gene is
//! re-serialised by the cleanse's own re-serialiser (`vary::relevel`). The
//! template's operators in the gene's symbols: `Div -> ProtectedDiv`, `Sqrt ->
//! ProtectedSqrt`, `Inv -> ProtectedInv`, `Pow(x, 2) -> Pow2`, `Pow(x, 3) ->
//! Pow3`, `Add Sub Mul Neg` as they are; any other operator or exponent is
//! `UnmappableOp`. A protected form equals the raw one only where its guard does
//! not fire, so the mapped graft is EVALUATED in f64 and must give the entry's
//! value, or the write-back is `UnmappableOp` (a division by hbar, 1e-34, is
//! under ProtectedDiv's 1e-6 guard). A template `Num` becomes a "?" whose Dc
//! entry points at an rnc slot holding that value — one that holds it already,
//! else one no surviving "?" reads, else `NoRncSlot`; a name becomes its
//! withheld `Named` terminal. "Never raises into evolution": on any refusal the
//! gene and its constants are bit for bit as they were, and the reason is
//! COUNTED ([`SnapCounts`]).
//!
//! A WGSL TWIN would need: the gene tree scratch the cleanse kernel already has
//! (`child`, `ordinal`, queue, new_tok, new_dc at MAX_CLEANSE), the resident
//! template and info blocks (stage 1), a per-gene "site, entry, sign" block
//! uploaded from the host's f64 decisions, the op -> token map and the named ids
//! as a small table, and a graft scratch of `sites * (TEMPLATE_MAX + 1)` nodes;
//! the rnc claim is a linear scan over `n_rnc`. Nothing here allocates by a
//! pattern a kernel could not hold in fixed scratch.

use super::engine::{Symbol, SymbolTable};
use super::vary::{relevel, GeneTree, Graft, Unfit, GRAFT};
use super::{Layout, SymbolCodes};
use crate::chrom_score::Wrapper;
use crate::gpu_eval::Op;
use crate::lint::flat::{Flat, LNode};
use crate::lint::node::Tree;
use crate::lint::snap_table::{Hit, SnapTable, FAMILIES};

/// A gene, or a model, as snap sees it.
#[derive(Clone, Debug, PartialEq)]
pub enum Form {
    /// A number. `Some(p)`: the constant subtree at gene position `p`, folded, and
    /// OFFERED to the match.
    Num(f64, Option<usize>),
    /// A data column.
    Var(u32),
    App(Op, Vec<Form>),
}

impl Form {
    pub fn offered(&self) -> usize {
        match self {
            Form::Num(_, site) => usize::from(site.is_some()),
            Form::Var(_) => 0,
            Form::App(_, kids) => kids.iter().map(Form::offered).sum(),
        }
    }

    /// Every site withheld: how a gene enters a model that examines another.
    pub fn withheld(&self) -> Form {
        match self {
            Form::Num(v, _) => Form::Num(*v, None),
            Form::Var(c) => Form::Var(*c),
            Form::App(op, kids) => Form::App(*op, kids.iter().map(Form::withheld).collect()),
        }
    }

    /// Canonical level order, as `Flat::from_tree` lays a tree out, over the
    /// columns `cols`; and beside each node the site it offers, if any.
    pub fn flatten(&self, cols: &[String]) -> (Flat, Vec<Option<usize>>) {
        let blank = LNode { op: 0, arg0: 0, arg1: 0, lit: 0.0 };
        let mut nodes = vec![blank];
        let mut sites = vec![None];
        let mut queue: Vec<(&Form, usize)> = vec![(self, 0)];
        let mut next = 0usize;
        while next < queue.len() {
            let (f, at) = queue[next];
            next += 1;
            nodes[at] = match f {
                Form::Num(v, site) => {
                    sites[at] = *site;
                    LNode { op: Op::Num as u32, arg0: 0, arg1: 0, lit: *v }
                }
                Form::Var(col) => LNode { op: Op::Var as u32, arg0: *col, arg1: 0, lit: 0.0 },
                Form::App(op, kids) => {
                    let first = nodes.len();
                    for k in kids {
                        nodes.push(blank);
                        sites.push(None);
                        queue.push((k, nodes.len() - 1));
                    }
                    LNode { op: *op as u32, arg0: first as u32, arg1: if kids.len() > 1 { first as u32 + 1 } else { 0 }, lit: 0.0 }
                }
            };
        }
        (Flat { nodes, vars: cols.to_vec() }, sites)
    }
}

/// `op` on constants, in f64, by the crate's evaluator; `None` unless finite.
fn value_of(op: Op, args: &[f64]) -> Option<f64> {
    Tree::App(op, args.iter().map(|v| Tree::Num(*v)).collect()).eval(&[]).ok().filter(|v| v.is_finite())
}

fn constants(kids: &[Form]) -> Option<Vec<f64>> {
    kids.iter().map(|k| if let Form::Num(v, _) = k { Some(*v) } else { None }).collect()
}

/// A gene's expression with every maximal constant subtree FOLDED to one number
/// (only a finite value folds). With `offer`, a folded subtree whose value is not
/// a whole number carries its gene position as a site. `None`: the expression
/// does not close, or a "?" has no constant to read (as `decode_gene`).
pub fn gene_form(tokens: &[u32], rnc: &[f32], layout: Layout, table: &SymbolTable, codes: &SymbolCodes, offer: bool) -> Option<Form> {
    struct Gene<'a> {
        tokens: &'a [u32],
        rnc: &'a [f32],
        table: &'a SymbolTable,
        tree: GeneTree,
        ht: usize,
        dc_len: usize,
        offer: bool,
    }
    fn go(p: usize, c: &Gene) -> Option<Form> {
        let Gene { tokens, rnc, table, tree, ht, dc_len, offer } = c;
        let (ht, dc_len, offer) = (*ht, *dc_len, *offer);
        let kid = |j: usize| go(tree.child[p] + j, c);
        let folded = |op: Op, kids: Vec<Form>| match constants(&kids).and_then(|v| value_of(op, &v)) {
            Some(v) => Form::Num(v, None),
            None => Form::App(op, kids),
        };
        let form = match table.symbols[tokens[p] as usize] {
            Symbol::Input(col) => return Some(Form::Var(col)),
            Symbol::Constant(v) => return Some(Form::Num(f64::from(v), None)),
            Symbol::Named(k) => return Some(Form::Num(table.named.get(k as usize)?.1, None)),
            Symbol::Rnc => {
                let o = tree.ordinal[p];
                let slot = if o < dc_len { tokens[ht + o] as usize } else { return None };
                return Some(Form::Num(f64::from(*rnc.get(slot)?), None));
            }
            Symbol::Function(op) => folded(op, (0..op.arity()).map(kid).collect::<Option<Vec<Form>>>()?),
            Symbol::Compound(compound) => {
                let ops = compound.expansion();
                let mut form = folded(ops[0], vec![kid(0)?, kid(1)?]);
                for op in &ops[1..] {
                    form = folded(*op, vec![form]);
                }
                form
            }
        };
        Some(match form {
            Form::Num(v, _) if offer && v.fract() != 0.0 => Form::Num(v, Some(p)),
            other => other,
        })
    }
    let tree = GeneTree::of(tokens, layout, codes)?;
    go(0, &Gene { tokens, rnc, table, tree, ht: (layout.head + layout.tail) as usize, dc_len: layout.tail as usize, offer })
}

/// `a * WRAPPER(LINKER(genes)) + b`, as `Engine::math_of` writes it. `linker` is
/// the linker's name (avgval, mulval, addval).
pub fn model_form(genes: &[Form], linker: &str, wrapper: Wrapper, a: f64, b: f64) -> Form {
    let mut body = genes[0].clone();
    for g in &genes[1..] {
        body = Form::App(if linker == "mulval" { Op::Mul } else { Op::Add }, vec![body, g.clone()]);
    }
    if linker == "avgval" && genes.len() > 1 {
        body = Form::App(Op::Div, vec![body, Form::Num(genes.len() as f64, None)]);
    }
    body = match wrapper {
        Wrapper::LogAbs => Form::App(Op::Log, vec![Form::App(Op::Abs, vec![body])]),
        Wrapper::SqrtAbs => Form::App(Op::Sqrt, vec![Form::App(Op::Abs, vec![body])]),
        _ => body,
    };
    Form::App(Op::Add, vec![Form::App(Op::Mul, vec![Form::Num(a, None), body]), Form::Num(b, None)])
}

/// What became of one gene's write-back — the linter plan's status contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteBack {
    /// The gene now decodes to the snapped variant.
    Grafted,
    /// The rewritten gene is the gene: nothing to write.
    Unchanged,
    /// A function would sit at or past the virtual head.
    HeadOversize,
    /// A template number needs an rnc slot and every slot is read by a "?"; or
    /// the rewritten gene has more "?" than the Dc domain has entries.
    NoRncSlot,
    /// The table has no symbol for a template's operator, exponent or name, or
    /// the protected form does not compute the entry's value.
    UnmappableOp,
    /// The rewritten expression does not close within head + tail.
    NotClosed,
}

/// The gene symbol of a template operator; `Pow` is the caller's.
fn gene_op(op: u32) -> Option<Op> {
    [(Op::Div, Op::ProtectedDiv), (Op::Sqrt, Op::ProtectedSqrt), (Op::Inv, Op::ProtectedInv), (Op::Add, Op::Add), (Op::Sub, Op::Sub), (Op::Mul, Op::Mul), (Op::Neg, Op::Neg)]
        .iter()
        .find(|(from, _)| *from as u32 == op)
        .map(|(_, to)| *to)
}

/// One site's template as grafts appended to `grafts`, the rnc slots it claims
/// marked in `rnc` / `used`; returns the graft's tree entry and its f64 value
/// under the GENE's (protected) operators.
fn graft_template(
    hit: Hit,
    snap: &SnapTable,
    table: &SymbolTable,
    codes: &SymbolCodes,
    grafts: &mut Vec<Graft>,
    rnc: (&mut [f32], &mut [bool]),
) -> Result<(usize, f64), WriteBack> {
    let template = snap.template(hit.entry);
    let (rnc, used) = rnc;
    // (template node, the graft that is its parent and which child it is)
    let mut stack: Vec<(usize, Option<(usize, usize)>)> = vec![(0, None)];
    let first = grafts.len();
    let mut applied: Vec<(usize, Op)> = Vec::new();
    let mut leaf_values: Vec<(usize, f64)> = Vec::new();
    while let Some((t, parent)) = stack.pop() {
        let node = template.nodes[t];
        let at = grafts.len();
        if let Some((parent, which)) = parent {
            grafts[parent].kids[which] = GRAFT + at;
        }
        if node.op == Op::Var as u32 {
            let token = table.named_id(node.arg0).ok_or(WriteBack::UnmappableOp)?;
            leaf_values.push((at, table.named[node.arg0 as usize].1));
            grafts.push(Graft { token, kids: [0, 0], dc: 0 });
        } else if node.op == Op::Num as u32 {
            let token = codes.rnc_id.ok_or(WriteBack::UnmappableOp)?;
            let v = node.lit as f32;
            if f64::from(v) != node.lit {
                return Err(WriteBack::UnmappableOp);
            }
            let slot = match rnc.iter().position(|x| x.to_bits() == v.to_bits()) {
                Some(slot) => slot,
                None => used.iter().position(|u| !u).ok_or(WriteBack::NoRncSlot)?,
            };
            rnc[slot] = v;
            used[slot] = true;
            leaf_values.push((at, node.lit));
            grafts.push(Graft { token, kids: [0, 0], dc: slot as u32 });
        } else if node.op == Op::Pow as u32 {
            let exponent = template.nodes[node.arg1 as usize];
            let op = match (exponent.op == Op::Num as u32, exponent.lit) {
                (true, 2.0) => Op::Pow2,
                (true, 3.0) => Op::Pow3,
                _ => return Err(WriteBack::UnmappableOp),
            };
            grafts.push(Graft { token: table.function_id(op).ok_or(WriteBack::UnmappableOp)?, kids: [0, 0], dc: 0 });
            applied.push((at, op));
            stack.push((node.arg0 as usize, Some((at, 0))));
        } else {
            let op = gene_op(node.op).ok_or(WriteBack::UnmappableOp)?;
            grafts.push(Graft { token: table.function_id(op).ok_or(WriteBack::UnmappableOp)?, kids: [0, 0], dc: 0 });
            applied.push((at, op));
            // The second child is pushed first, so the first is laid out first.
            if op.arity() == 2 {
                stack.push((node.arg1 as usize, Some((at, 1))));
            }
            stack.push((node.arg0 as usize, Some((at, 0))));
        }
    }
    // The graft's value under the gene's operators: children were appended after
    // their parents, so a backward scan has every child's value ready.
    let mut value = vec![f64::NAN; grafts.len() - first];
    for (at, v) in leaf_values {
        value[at - first] = v;
    }
    for (at, op) in applied.iter().rev() {
        let kids = grafts[*at].kids;
        let args: Vec<f64> = kids.iter().take(op.arity()).map(|k| value[k - GRAFT - first]).collect();
        value[at - first] = value_of(*op, &args).ok_or(WriteBack::UnmappableOp)?;
    }
    let mut root = GRAFT + first;
    let mut v = value[0];
    if hit.negative {
        let token = table.function_id(Op::Neg).ok_or(WriteBack::UnmappableOp)?;
        grafts.push(Graft { token, kids: [root, 0], dc: 0 });
        root = GRAFT + grafts.len() - 1;
        v = -v;
    }
    Ok((root, v))
}

/// How far a protected graft's value may sit from its entry's: both are f64
/// evaluations of the same few operations.
const GRAFT_VALUE_AGREE: f64 = 1e-12;

/// THE WRITE-BACK on ONE gene (`tokens` = head, tail and Dc; `rnc` = its
/// constants): every `(gene position, hit)` of `sites` — a folded constant
/// subtree and the lattice entry the guard kept for it — is replaced by the
/// entry's form in gene symbols. A pure function of its arguments; on anything
/// but `Grafted` the gene and its constants are exactly as they were.
pub fn write_back(
    tokens: &mut [u32],
    rnc: &mut [f32],
    layout: Layout,
    vhead: u32,
    table: &SymbolTable,
    sites: &[(usize, Hit)],
    snap: &SnapTable,
) -> WriteBack {
    let codes = table.codes();
    let Some(tree) = GeneTree::of(tokens, layout, &codes) else { return WriteBack::NotClosed };
    if sites.is_empty() {
        return WriteBack::Unchanged;
    }
    let ht = (layout.head + layout.tail) as usize;
    // The rnc slots the SURVIVING "?"s read: a position under a site goes.
    let mut gone = vec![false; tree.n];
    let mut used = vec![false; rnc.len()];
    for p in 0..tree.n {
        gone[p] = gone[p] || sites.iter().any(|(site, _)| *site == p);
        for j in 0..codes.arity[tokens[p] as usize] as usize {
            gone[tree.child[p] + j] = gone[p];
        }
        let o = tree.ordinal[p];
        if !gone[p] && o < layout.tail as usize {
            if let Some(slot) = used.get_mut(tokens[ht + o] as usize) {
                *slot = true;
            }
        }
    }
    let mut new_rnc = rnc.to_vec();
    let mut grafts: Vec<Graft> = Vec::new();
    let mut swaps: Vec<(usize, usize)> = Vec::with_capacity(sites.len());
    for (site, hit) in sites {
        let (root, value) = match graft_template(*hit, snap, table, &codes, &mut grafts, (&mut new_rnc, &mut used)) {
            Ok(done) => done,
            Err(why) => return why,
        };
        let entry = snap.signed_value(*hit);
        if (value - entry).abs() > GRAFT_VALUE_AGREE * entry.abs() {
            return WriteBack::UnmappableOp;
        }
        swaps.push((*site, root));
    }
    let before = tokens.to_vec();
    match relevel(tokens, layout, vhead, &codes, &tree, &swaps, &grafts) {
        Err(Unfit::HeadOversize) => WriteBack::HeadOversize,
        Err(Unfit::NotClosed) => WriteBack::NotClosed,
        Err(Unfit::DcOversize) => WriteBack::NoRncSlot,
        Ok(()) if before == tokens && new_rnc.iter().zip(rnc.iter()).all(|(a, b)| a.to_bits() == b.to_bits()) => WriteBack::Unchanged,
        Ok(()) => {
            rnc.copy_from_slice(&new_rnc);
            WriteBack::Grafted
        }
    }
}

/// ONE SUBSTITUTION SNAP ACTUALLY MADE — `3.142857 -> pi`, and where.
///
/// The counts below say HOW MANY; they never say WHAT, and what a snap did is the
/// only part of it an operator can judge. A count of 4 in the `Algebraic` family
/// is a number; `3.142857 -> pi` at generation 240 is a finding.
///
/// It is written ONLY for a `Grafted` write-back — the gene really carries the
/// named constant from here on — so the formatting cost rides on the rarest
/// branch of the whole pipeline and never on the examined-gene path, which runs
/// thousands of times a beat.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapRecord {
    pub generation: u32,
    /// The population row and the gene within it — where to look in the genome.
    pub row: usize,
    pub gene: usize,
    /// The folded constant subtree's value, before.
    pub before: f64,
    /// The lattice entry's form as infix — `pi`, `2*pi`, `sqrt(2)` — signed as it
    /// was grafted.
    pub after: String,
    /// What that form computes, so a reader can see the snap's own error.
    pub after_value: f64,
}

/// How many substitutions are kept. A fit runs 20,000 generations and this rides
/// beside the hot loop, so the log is a RING and not a history: the last few are
/// what a screen shows and what an operator asks about, and an unbounded vector
/// of them would be the instrumentation growing with the run.
pub const SNAP_RING: usize = 32;

/// What snap did in a fit: the SNAP line of `examples/evolve_fit.rs`, and the
/// measurement of whether snap can bite with whole-number constants.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SnapCounts {
    pub beats: u64,
    /// Rows chosen (snap winners), and those skipped: not scored, or a gene
    /// that does not decode.
    pub rows: u64,
    pub rows_unscored: u64,
    /// Genes examined, and those with at least one folded constant that is not a
    /// whole number — the only ones snap can do anything for.
    pub genes_examined: u64,
    pub genes_with_literal: u64,
    /// Of those, the model did not fit the evaluator's SLOT: skipped.
    pub model_oversize: u64,
    pub literals_offered: u64,
    /// Offered literals the device matched within the tolerance, by the
    /// matched entry's family (`snap_table::Family`'s order).
    pub literals_matched: u64,
    pub matched_by_family: [u64; FAMILIES],
    pub band_overflows: u64,
    /// Models (one per examined gene) whose variant the device kept, and those
    /// of them the host's f64 word refused.
    pub device_kept: u64,
    pub refused_f64: u64,
    /// The write-back's statuses, one per kept model.
    pub grafted: u64,
    pub unchanged: u64,
    pub head_oversize: u64,
    pub no_rnc_slot: u64,
    pub unmappable_op: u64,
    pub not_closed: u64,
    /// Rows whose genome changed (re-scored by the next evaluation).
    pub rows_changed: u64,
    pub seconds: f64,
    /// THE LAST [`SNAP_RING`] SUBSTITUTIONS, oldest first. Bounded by
    /// construction, so a 20,000-generation fit carries the same 32 records a
    /// 40-generation one does.
    pub recent: std::collections::VecDeque<SnapRecord>,
    /// How many substitutions there have EVER been, which is not `recent.len()`
    /// once the ring has turned over. A reader that has written the first `n` as
    /// events knows from this how many it missed, so a dropped record is a
    /// countable gap rather than a silent one.
    pub substitutions: u64,
}

impl SnapCounts {
    pub fn count(&mut self, status: WriteBack) {
        *match status {
            WriteBack::Grafted => &mut self.grafted,
            WriteBack::Unchanged => &mut self.unchanged,
            WriteBack::HeadOversize => &mut self.head_oversize,
            WriteBack::NoRncSlot => &mut self.no_rnc_slot,
            WriteBack::UnmappableOp => &mut self.unmappable_op,
            WriteBack::NotClosed => &mut self.not_closed,
        } += 1;
    }

    /// Keep one substitution, dropping the oldest past [`SNAP_RING`].
    pub fn remember(&mut self, record: SnapRecord) {
        self.substitutions += 1;
        self.recent.push_back(record);
        while self.recent.len() > SNAP_RING {
            self.recent.pop_front();
        }
    }

    /// THE MEASUREMENT's line: can snap bite with whole-number constants?
    pub fn detail(&self) -> String {
        let families = crate::lint::snap_table::Family::ALL.iter().zip(self.matched_by_family).map(|(f, n)| format!("{f:?}={n}")).collect::<Vec<_>>().join("/");
        format!(
            "SNAP_DETAIL\trows={}\trows_skipped={}\tgenes_examined={}\tgenes_with_nonwhole_literal={}\tliterals_offered={}\tliterals_matched={}\tby_family={families}\tband_overflows={}\tkept_by_device={}\trefused_by_f64={}\twritten_back={}\trows_changed={}\tseconds_per_beat={:.4}",
            self.rows,
            self.rows_unscored,
            self.genes_examined,
            self.genes_with_literal,
            self.literals_offered,
            self.literals_matched,
            self.band_overflows,
            self.device_kept,
            self.refused_f64,
            self.grafted,
            self.rows_changed,
            self.seconds / self.beats.max(1) as f64
        )
    }

    /// One beat's counts joined to a fit's.
    pub fn add(&mut self, beat: &SnapCounts) {
        let pairs = [
            (&mut self.beats, beat.beats),
            (&mut self.rows, beat.rows),
            (&mut self.rows_unscored, beat.rows_unscored),
            (&mut self.genes_examined, beat.genes_examined),
            (&mut self.genes_with_literal, beat.genes_with_literal),
            (&mut self.model_oversize, beat.model_oversize),
            (&mut self.literals_offered, beat.literals_offered),
            (&mut self.literals_matched, beat.literals_matched),
            (&mut self.band_overflows, beat.band_overflows),
            (&mut self.device_kept, beat.device_kept),
            (&mut self.refused_f64, beat.refused_f64),
            (&mut self.grafted, beat.grafted),
            (&mut self.unchanged, beat.unchanged),
            (&mut self.head_oversize, beat.head_oversize),
            (&mut self.no_rnc_slot, beat.no_rnc_slot),
            (&mut self.unmappable_op, beat.unmappable_op),
            (&mut self.not_closed, beat.not_closed),
            (&mut self.rows_changed, beat.rows_changed),
        ];
        for (total, n) in pairs {
            *total += n;
        }
        for (total, n) in self.matched_by_family.iter_mut().zip(beat.matched_by_family) {
            *total += n;
        }
        self.seconds += beat.seconds;
        // The beat's substitutions join the fit's, newest kept: the ring is the
        // LAST few of the whole run, not the last few of one beat. The TOTAL is
        // the beat's own, not the number of records that survived its ring —
        // a beat that grafted more than the ring holds still says how many.
        for record in &beat.recent {
            self.remember(record.clone());
        }
        self.substitutions = self.substitutions - beat.recent.len() as u64 + beat.substitutions;
    }

    /// `SNAP\t<beats>\t<genes examined>\t<literals matched>\t<variants kept by
    /// device>\t<refused by f64>\t<written back>\t<aborted: reason=n/...>`.
    pub fn line(&self) -> String {
        format!(
            "SNAP\t{}\t{}\t{}\t{}\t{}\t{}\thead_oversize={}/no_rnc_slot={}/unmappable_op={}/not_closed={}/unchanged={}/model_oversize={}",
            self.beats,
            self.genes_examined,
            self.literals_matched,
            self.device_kept,
            self.refused_f64,
            self.grafted,
            self.head_oversize,
            self.no_rnc_slot,
            self.unmappable_op,
            self.not_closed,
            self.unchanged,
            self.model_oversize
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolve::engine::{decode_gene, evaluate_math, nodes_to_math_named, Config, Data, Engine};
    use crate::evolve::vary::{vary, GenParams, Generation, Island, Rates};
    use crate::evolve::{below, draw, init, InitParams};
    use crate::lint::engine::LitMode;
    use crate::lint::snap_graft::{match_literals, snap_graft, variant_sites};
    use crate::lint::snap_guard::{snap_guard, GuardData, Verdict, R2_DROP_TOL};
    use crate::lint::snap_table::Context;

    const TOL: f64 = 1e-3;

    fn lattice() -> SnapTable {
        SnapTable::standard().expect("the lattice builds")
    }

    fn table(n_inputs: u32, snap: &SnapTable) -> SymbolTable {
        SymbolTable::wide(n_inputs).with_named(&snap.names).expect("every name has a value")
    }

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("x_{i}")).collect()
    }

    fn bits(rnc: &[f32]) -> Vec<u32> {
        rnc.iter().map(|v| v.to_bits()).collect()
    }

    /// A gene that expresses `math` (gene symbols: `ProtectedDiv`, not `Div`), its
    /// numbers as "?"s reading rnc slots in order of first use; the rest of the
    /// head and tail is column 0, the rest of the constants `spare`.
    fn gene_of(math: &str, layout: Layout, table: &SymbolTable, spare: f32) -> (Vec<u32>, Vec<f32>) {
        let tree = Tree::parse(math).expect("parses");
        let codes = table.codes();
        let column = |c: u32| table.symbols.iter().position(|s| *s == Symbol::Input(c)).expect("a column") as u32;
        let ht = (layout.head + layout.tail) as usize;
        let mut tokens = vec![column(0); layout.gene_width() as usize];
        tokens[ht..].fill(0);
        let mut rnc: Vec<f32> = Vec::new();
        let (mut m, mut k) = (0usize, 0usize);
        let mut queue = std::collections::VecDeque::from([&tree]);
        while let Some(t) = queue.pop_front() {
            tokens[m] = match t {
                Tree::Var(name) => column(name.trim_start_matches("x_").parse().expect("x_<column>")),
                Tree::Num(v) => {
                    let slot = rnc.iter().position(|x| *x == *v as f32).unwrap_or_else(|| {
                        rnc.push(*v as f32);
                        rnc.len() - 1
                    });
                    tokens[ht + k] = slot as u32;
                    k += 1;
                    codes.rnc_id.expect("the wide table has a ?")
                }
                Tree::App(op, kids) => {
                    queue.extend(kids);
                    table.function_id(*op).unwrap_or_else(|| panic!("{op:?} is not a gene symbol"))
                }
            };
            m += 1;
        }
        assert!(m <= ht && rnc.len() <= layout.n_rnc as usize, "the test gene does not fit its layout");
        rnc.resize(layout.n_rnc as usize, spare);
        (tokens, rnc)
    }

    fn math_of(tokens: &[u32], rnc: &[f32], layout: Layout, table: &SymbolTable, cols: usize) -> String {
        let nodes = decode_gene(tokens, rnc, layout, table).expect("the gene decodes");
        nodes_to_math_named(&nodes, 0, &names(cols), &table.named_values())
    }

    fn rows(n: usize, cols: usize) -> Vec<Vec<(String, f64)>> {
        (0..n).map(|r| (0..cols).map(|c| (format!("x_{c}"), 0.5 + below(draw(41, 0, r as u32, c as u32, 7), 2000) as f64 * 1e-3)).collect()).collect()
    }

    /// The offered sites of a gene.
    fn offered_sites(tokens: &[u32], rnc: &[f32], layout: Layout, table: &SymbolTable) -> Vec<(usize, f64)> {
        let form = gene_form(tokens, rnc, layout, table, &table.codes(), true).expect("a form");
        let (flat, at) = form.flatten(&names(3));
        at.iter().zip(&flat.nodes).filter_map(|(site, n)| Some(((*site)?, n.lit))).collect()
    }

    #[test]
    fn named_constants_are_withheld_terminals_appended_after_every_symbol() {
        let snap = lattice();
        assert!(!snap.names.is_empty());
        let plain = SymbolTable::wide(3).with_compounds();
        let named = SymbolTable::wide(3).with_compounds().with_named(&snap.names).unwrap();
        // The wide set, then the compounds, then the named constants: no id moves.
        assert_eq!(named.symbols[..plain.symbols.len()], plain.symbols[..]);
        assert_eq!(named.symbols.len(), plain.symbols.len() + snap.names.len());
        let known = crate::snap_karva::constant_values();
        for (k, name) in snap.names.iter().enumerate() {
            let id = named.named_id(k as u32).expect("the constant has an id") as usize;
            assert_eq!(id, plain.symbols.len() + k);
            assert!(named.withheld[id] && named.arity(id as u32) == 0, "{name}");
            assert_eq!(named.named[k], (name.clone(), known[name]));
        }
        assert!(named.withheld[..plain.symbols.len()].iter().all(|w| !w));
        let codes = named.codes();
        codes.validate().unwrap();
        assert_eq!(codes.sample_terminals, plain.codes().sample_terminals);
        assert_eq!(codes.sample_functions, plain.codes().sample_functions);

        // Never drawn: not by init, not by 200 generations of every operator.
        let layout = Layout::for_arity(400, 3, 16, named.max_arity(), 10);
        let first_named = plain.symbols.len() as u32;
        let symbols = (layout.head + layout.tail) as usize;
        let never = |g: &Generation, when: &str| {
            let drawn = g.pop.genome.chunks(layout.gene_width() as usize).any(|gene| gene[..symbols].iter().any(|id| *id >= first_named));
            assert!(!drawn, "{when}: a withheld named constant was drawn");
        };
        let pop = init(layout, &codes, &InitParams { seed: 5, generation: 0, rnc_lo: -100, rnc_hi: 100, n_wrappers: 3, vhead: 0 }).unwrap();
        let mut now = Generation { fitness: vec![0.0; 400], pop };
        never(&now, "init");
        // every operator on, the cleanse included — on the islands that breed.
        let rates = Rates::with_cleanse(layout, 0.5);
        let islands = [Island { lo: 0, hi: 300, elites: 2, tournsize: 20, arrivals: 0, rates }, Island { lo: 300, hi: 400, elites: 2, tournsize: 7, arrivals: 0, rates }];
        for generation in 1..=200 {
            now = vary(&now, &islands, &codes, &GenParams { seed: 5, generation, rnc_lo: -100, rnc_hi: 100, cohort_merge: 0, vhead: 0 }).unwrap();
            now.pop.check(&codes).unwrap();
            never(&now, "vary");
            for (r, f) in now.fitness.iter_mut().enumerate() {
                *f = below(draw(5, generation, r as u32, 1, 99), 1_000_000) as f32;
            }
        }
    }

    #[test]
    fn a_named_constant_decodes_to_its_value_and_prints_its_exact_f64() {
        let snap = lattice();
        let t = table(2, &snap);
        let layout = Layout::for_arity(1, 1, 6, t.max_arity(), 4);
        let known = crate::snap_karva::constant_values();
        for (k, name) in snap.names.iter().enumerate() {
            let (mut tokens, rnc) = gene_of(r#"(Mul (Var "x_0") (Var "x_1"))"#, layout, &t, 0.0);
            tokens[2] = t.named_id(k as u32).unwrap();
            let nodes = decode_gene(&tokens, &rnc, layout, &t).unwrap();
            assert_eq!((nodes[2].op, nodes[2].arg0, nodes[2].konst), (Op::Num as u32, k as u32 + 1, known[name] as f32), "{name}");
            let math = nodes_to_math_named(&nodes, 0, &names(2), &t.named_values());
            assert_eq!(math, format!("(Mul (Var \"x_0\") (Num {:?}))", known[name]), "{name}");
            // The folded form reads the f64 too, and a lone constant is not offered.
            let form = gene_form(&tokens, &rnc, layout, &t, &t.codes(), true).unwrap();
            assert_eq!(form, Form::App(Op::Mul, vec![Form::Var(0), Form::Num(known[name], None)]));
        }
    }

    /// 22/7 is 4e-4 from pi: reachable from "?" values in -100..100.
    const X0_TIMES_22_OVER_7: &str = r#"(Add (Mul (Var "x_0") (ProtectedDiv (Num 22.0) (Num 7.0))) (Mul (Num 5.0) (Var "x_1")))"#;

    #[test]
    fn a_constant_subtree_near_pi_is_written_back_as_pi() {
        let snap = lattice();
        let t = table(3, &snap);
        let layout = Layout::for_arity(1, 1, 10, t.max_arity(), 6);
        let (mut tokens, mut rnc) = gene_of(X0_TIMES_22_OVER_7, layout, &t, -3.0);
        let offered = offered_sites(&tokens, &rnc, layout, &t);
        assert_eq!(offered.len(), 1, "one folded constant that is not a whole number: {offered:?}");
        assert_eq!(offered[0].1, 22.0 / 7.0);
        let pi = snap.nearest_in(22.0 / 7.0, Context::Algebraic, TOL, LitMode::F64).expect("22/7 snaps");
        assert_eq!(snap.maths[pi.entry as usize], r#"(Var "pi")"#, "the lattice's choice for 22/7");
        let sites = [(offered[0].0, pi)];

        let (before_tokens, before_rnc) = (tokens.clone(), rnc.clone());
        assert_eq!(write_back(&mut tokens, &mut rnc, layout, layout.head, &t, &sites, &snap), WriteBack::Grafted);
        assert_ne!(tokens, before_tokens);
        // The rewritten gene IS x0 * pi + 5 * x1, in f64, and the other "?" still
        // reads ITS constant (the Dc domain was rebuilt: it was the third "?").
        let math = math_of(&tokens, &rnc, layout, &t, 3);
        assert_eq!(math, format!("(Add (Mul (Var \"x_0\") (Num {:?})) (Mul (Num 5.0) (Var \"x_1\")))", std::f64::consts::PI));
        let data = rows(8, 3);
        let got = evaluate_math(&math, &data).unwrap();
        for (row, v) in data.iter().zip(got) {
            assert_eq!(v, row[0].1 * std::f64::consts::PI + 5.0 * row[1].1);
        }
        let pi_id = t.named_id(snap.names.iter().position(|n| n == "pi").expect("the lattice names pi") as u32).unwrap();
        assert!(tokens[..(layout.head + layout.tail) as usize].contains(&pi_id));

        // Deterministic: the same inputs, the same gene.
        let (mut again_tokens, mut again_rnc) = (before_tokens, before_rnc);
        assert_eq!(write_back(&mut again_tokens, &mut again_rnc, layout, layout.head, &t, &sites, &snap), WriteBack::Grafted);
        assert_eq!((again_tokens, bits(&again_rnc)), (tokens, bits(&rnc)));
    }

    /// Every entry of the lattice through the write-back on a roomy gene: what is
    /// grafted decodes to the entry's value (so `Div -> ProtectedDiv`, `Sqrt ->
    /// ProtectedSqrt`, `Pow -> Pow2 / Pow3` kept the value), and what is not leaves
    /// the gene as it was.
    #[test]
    fn a_grafted_template_computes_its_entrys_value_in_gene_symbols() {
        let snap = lattice();
        let t = table(3, &snap);
        let layout = Layout::for_arity(1, 1, 24, t.max_arity(), 10);
        let (base_tokens, base_rnc) = gene_of(r#"(Mul (Var "x_0") (ProtectedDiv (Num 22.0) (Num 7.0)))"#, layout, &t, 0.0);
        let div = t.function_id(Op::ProtectedDiv).unwrap();
        let mut census = std::collections::BTreeMap::new();
        let mut div_kept = 0;
        for entry in 0..snap.len() as u32 {
            for negative in [false, true] {
                let hit = Hit { entry, negative };
                let (mut tokens, mut rnc) = (base_tokens.clone(), base_rnc.clone());
                let status = write_back(&mut tokens, &mut rnc, layout, layout.head, &t, &[(2, hit)], &snap);
                *census.entry(format!("{status:?}")).or_insert(0usize) += 1;
                if status != WriteBack::Grafted {
                    assert!(tokens == base_tokens && bits(&rnc) == bits(&base_rnc), "{status:?} changed the gene");
                    continue;
                }
                let form = gene_form(&tokens, &rnc, layout, &t, &t.codes(), false).unwrap();
                let Form::App(Op::Mul, kids) = &form else { panic!("{form:?}") };
                let Form::Num(v, _) = kids[1] else { panic!("the graft is not a constant: {form:?}") };
                let want = snap.signed_value(hit);
                assert!((v - want).abs() <= 1e-12 * want.abs(), "{}: the gene computes {v}, the entry is {want}", snap.maths[entry as usize]);
                div_kept += usize::from(snap.maths[entry as usize].contains("(Div ") && tokens.contains(&div));
            }
        }
        eprintln!("write-back of every lattice entry (both signs) into a roomy gene: {census:?}");
        assert!(div_kept > 0, "no template with a Div was written as ProtectedDiv");
        assert!(census.get("Grafted").copied().unwrap_or(0) > 0);
    }

    #[test]
    fn every_abort_leaves_the_gene_and_its_constants_bit_identical() {
        let snap = lattice();
        let t = table(3, &snap);
        let refused = |math: &str, layout: Layout, vhead: u32, hit: Hit, want: WriteBack| {
            let (mut tokens, mut rnc) = gene_of(math, layout, &t, 0.0);
            let (before_tokens, before_rnc) = (tokens.clone(), bits(&rnc));
            let sites: Vec<(usize, Hit)> = offered_sites(&tokens, &rnc, layout, &t).into_iter().map(|(site, _)| (site, hit)).collect();
            assert_eq!(write_back(&mut tokens, &mut rnc, layout, vhead, &t, &sites, &snap), want, "{math}");
            assert_eq!((tokens, bits(&rnc)), (before_tokens, before_rnc), "{want:?} changed the gene");
        };
        // 2 pi: a Mul, a "?" and pi where 44/7 was.
        let two_pi = snap.nearest_in(44.0 / 7.0, Context::Algebraic, TOL, LitMode::F64).expect("44/7 snaps");
        assert_eq!(snap.maths[two_pi.entry as usize], r#"(Mul (Num 2.0) (Var "pi"))"#, "the lattice's choice for 44/7");
        let roomy = Layout::for_arity(1, 1, 10, t.max_arity(), 6);
        // head_oversize: the form's Mul would sit at position 6, past a virtual head of 4.
        let deep = r#"(Add (Var "x_0") (Add (Var "x_1") (Add (Var "x_2") (ProtectedDiv (Num 44.0) (Num 7.0)))))"#;
        refused(deep, roomy, 4, two_pi, WriteBack::HeadOversize);
        // no_rnc_slot: three constants, each read by a surviving "?", and the form needs a 2.
        let tight = Layout { n_rnc: 3, ..roomy };
        let reads_all = r#"(Add (Add (Mul (Num 9.0) (Var "x_0")) (Mul (Num 44.0) (Var "x_1"))) (Mul (Mul (Num 7.0) (Var "x_2")) (ProtectedDiv (Num 44.0) (Num 7.0))))"#;
        refused(reads_all, tight, tight.head, two_pi, WriteBack::NoRncSlot);
        // unmappable_op: a division by a constant under ProtectedDiv's guard, or a
        // power the table has no symbol for — the first entry refused so.
        let wide = Layout::for_arity(1, 1, 24, t.max_arity(), 10);
        let unmappable = (0..snap.len() as u32).map(|entry| Hit { entry, negative: false }).find(|hit| {
            let (mut tokens, mut rnc) = gene_of(X0_TIMES_22_OVER_7, wide, &t, 0.0);
            write_back(&mut tokens, &mut rnc, wide, wide.head, &t, &[(2, *hit)], &snap) == WriteBack::UnmappableOp
        });
        let hit = unmappable.expect("the lattice has an entry the gene symbols cannot write");
        eprintln!("unmappable_op: {}", snap.maths[hit.entry as usize]);
        refused(X0_TIMES_22_OVER_7, wide, wide.head, hit, WriteBack::UnmappableOp);
        // not_closed: a gene of three positions, and a form of three nodes under a Neg.
        let tiny = Layout::for_arity(1, 1, 1, t.max_arity(), 6);
        assert_eq!(tiny.head + tiny.tail, 3);
        refused(r#"(ProtectedDiv (Num 44.0) (Num 7.0))"#, tiny, tiny.head, Hit { negative: !two_pi.negative, ..two_pi }, WriteBack::NotClosed);
        // unchanged: no constant subtree to snap.
        refused(r#"(Mul (Var "x_0") (Add (Var "x_1") (Num 3.0)))"#, roomy, roomy.head, two_pi, WriteBack::Unchanged);
    }

    /// 600 rows, x in 1 .. 2.
    fn guard_data(y_of: impl Fn(&[f64]) -> f64) -> GuardData {
        let n = 600;
        let x: Vec<f64> = (0..n * 2).map(|i| 1.0 + below(draw(9, 0, i as u32, 0, 3), 100_000) as f64 * 1e-5).collect();
        let y = x.chunks(2).map(&y_of).collect();
        GuardData { cols: names(2), x, y, rows: 0..n }
    }

    /// The CPU guard on a model with only the examined gene's literals offered.
    fn guarded(model: &Form, data: &GuardData, snap: &SnapTable) -> Verdict {
        let (flat, sites) = model.flatten(&data.cols);
        let hits: Vec<Option<Hit>> = match_literals(&flat, snap, TOL, LitMode::F64).into_iter().zip(&sites).map(|(h, s)| h.filter(|_| s.is_some())).collect();
        assert!(hits.iter().any(Option::is_some), "nothing matched");
        let decision = snap_guard(&flat, &snap_graft(&flat, &hits, snap), &snap.names, data, R2_DROP_TOL).unwrap();
        if let Some(slot) = decision.slot {
            assert!(variant_sites(&flat, &hits, slot).iter().all(|i| sites[*i].is_some()), "a withheld literal was grafted");
        }
        decision.status
    }

    #[test]
    fn the_guard_judges_the_model_not_the_lone_gene() {
        let snap = lattice();
        let gene = Form::App(Op::Mul, vec![Form::Num(22.0 / 7.0, Some(1)), Form::Var(0)]);
        // y = pi x0, the model a * (22/7) x0 with its fitted a: the snap improves
        // nothing the scale had not already absorbed, costs (4e-4)^2 of R², is kept.
        let data = guard_data(|x| std::f64::consts::PI * x[0]);
        let a = std::f64::consts::PI / (22.0 / 7.0);
        let model = model_form(std::slice::from_ref(&gene), "addval", Wrapper::Identity, a, 0.0);
        assert_eq!(guarded(&model, &data, &snap), Verdict::Kept);
        // The scale, 0.9996 and within 1e-3 of 1, is not snap's: only the gene's is offered.
        assert_eq!(model.offered(), 1);

        // y = 1000 (22/7 - 3.13) x0: a second gene cancels most of the first and the
        // scale amplifies what is left. The LONE gene, scaled to y on its own, barely
        // moves (pi for 22/7 costs 1.6e-7 of R²); the MODEL with pi in it is 11.6 x0
        // for 12.9 x0 — refused.
        let data = guard_data(|x| 1000.0 * (22.0 / 7.0 - 3.13) * x[0]);
        let other = Form::App(Op::Mul, vec![Form::Num(-3.13, None), Form::Var(0)]);
        let model = model_form(&[gene.clone(), other], "addval", Wrapper::Identity, 1000.0, 0.0);
        assert_eq!(guarded(&model, &data, &snap), Verdict::R2Dropped);
        let lone = model_form(&[gene], "addval", Wrapper::Identity, 1000.0 * (22.0 / 7.0 - 3.13) / (22.0 / 7.0), 0.0);
        assert_eq!(guarded(&lone, &data, &snap), Verdict::Kept, "the lone gene's guard would have let it through");
    }

    /// y = sin(2 pi x0) x1 over 2,000 rows, x0 in [0, 1], x1 in [0.5, 1.5].
    fn sine_data() -> Data {
        let n = 2000usize;
        let x: Vec<f32> = (0..n * 2).map(|i| below(draw(77, 0, i as u32, 0, 5), 1_000_000) as f32 * 1e-6 + if i % 2 == 1 { 0.5 } else { 0.0 }).collect();
        let y = x.chunks(2).map(|r| (2.0 * std::f64::consts::PI * f64::from(r[0])).sin() * f64::from(r[1])).collect();
        Data { names: names(2), x, y, splits: crate::chrom_score::Splits { n_train: 1600, n_val: 400, n_extrap: 0 } }
    }

    #[test]
    fn snap_winners_runs_in_a_fit_and_every_written_gene_decodes() {
        let config = Config { pop_intake: 600, pop_champion: 200, rnc_lo: -100, rnc_hi: 100, snap_every: 5, max_generations: 40, max_seconds: 25.0, ..Config::srbench(7101) };
        let mut engine = Engine::new(config, sine_data()).expect("engine");
        let out = engine.fit().expect("the fit runs with snap on");
        let s = &out.snap;
        eprintln!("{}\n{}", s.line(), s.detail());
        assert_eq!(s.beats, u64::from(out.generations / 5));
        let confirmed = s.device_kept - s.refused_f64;
        assert!(s.grafted <= confirmed && s.device_kept <= s.literals_matched && s.literals_matched <= s.literals_offered, "{s:?}");
        assert_eq!(s.grafted + s.unchanged + s.head_oversize + s.no_rnc_slot + s.unmappable_op + s.not_closed, confirmed, "{s:?}");
        assert!(s.genes_with_literal <= s.genes_examined && s.rows_changed <= s.grafted, "{s:?}");
        assert!(s.grafted == 0 || s.rows_changed > 0, "{s:?}");
        let pop = engine.population().expect("the population reads back");
        pop.check(&engine.table.codes()).expect("the population keeps the rules");
        let l = engine.layout;
        let (width, nr, ht) = (l.gene_width() as usize, l.n_rnc as usize, (l.head + l.tail) as usize);
        let first_named = engine.table.named_id(0).expect("the table carries the named constants");
        let (mut carrying, mut expressed) = (0usize, 0usize);
        for (g, gene) in pop.genome.chunks(width).enumerate() {
            if gene[..ht].iter().any(|id| *id >= first_named) {
                carrying += 1;
                let nodes = decode_gene(gene, &pop.rnc[g * nr..(g + 1) * nr], l, &engine.table).expect("a gene that carries a named constant decodes");
                expressed += usize::from(nodes.iter().any(|n| n.op == Op::Num as u32 && n.arg0 != 0));
            }
        }
        let in_model: Vec<&str> = engine.table.named.iter().filter(|(_, v)| out.math.contains(&format!("(Num {v:?})"))).map(|(n, _)| n.as_str()).collect();
        eprintln!(
            "after {} generations ({}): {carrying} genes of {} carry a named constant, {expressed} express one; the final model holds {in_model:?}; validation 1 - R² {:.3e}",
            out.generations,
            out.stopped_by,
            pop.genome.len() / width,
            out.best.one_minus_r2[1]
        );
    }

    /// THE RING IS BOUNDED AND IT IS THE RUN'S, not one beat's. A fit that
    /// grafts thousands of times carries the same 32 records, they are the LAST
    /// 32, and the total says how many there really were — a dropped record is a
    /// countable gap and never a silent one.
    #[test]
    fn the_substitution_log_is_a_ring_and_says_what_it_dropped() {
        let record = |n: u32| SnapRecord {
            generation: n,
            row: n as usize,
            gene: 0,
            before: f64::from(n) + 0.5,
            after: "pi".to_string(),
            after_value: std::f64::consts::PI,
        };
        let mut beat = SnapCounts::default();
        for n in 0..100 {
            beat.remember(record(n));
        }
        assert_eq!(beat.recent.len(), SNAP_RING, "the ring grew past its bound");
        assert_eq!(beat.substitutions, 100, "the total is every substitution, not the ring's length");
        assert_eq!(beat.recent.front().map(|r| r.generation), Some(100 - SNAP_RING as u32));
        assert_eq!(beat.recent.back().map(|r| r.generation), Some(99), "the ring keeps the NEWEST");

        // Joined into a fit: the total is exact although the beat's own ring had
        // already dropped 68 of them.
        let mut fit = SnapCounts::default();
        fit.remember(record(1000));
        fit.add(&beat);
        assert_eq!(fit.substitutions, 101, "the fit's total lost the records the beat's ring dropped");
        assert_eq!(fit.recent.len(), SNAP_RING);
        assert_eq!(fit.recent.back().map(|r| r.generation), Some(99));

        // And the default is still the default: the off-by-default test compares
        // against it, and a ring with anything in it would break that.
        assert_eq!(SnapCounts::default().recent.len(), 0);
        assert_eq!(SnapCounts::default().substitutions, 0);
    }

    /// SNAP IS ON BY DEFAULT, and turning it OFF leaves nothing of it in the
    /// engine.
    ///
    /// The second half is the invariant worth having: the switch has to be a
    /// real switch, so that a fit run without snap is the engine it was, symbol
    /// table included. The first half only records what the default is — snap
    /// earned it on the fit that recovered strogatz_bacres1 (952 beats, 1,303
    /// forms written back, 6.25 s of 1,115).
    #[test]
    fn snap_is_on_by_default_and_off_leaves_the_symbol_table_it_was() {
        let c = Config::srbench(1);
        assert_eq!((c.snap_every, c.snap_top_k, c.snap_rel_tol, c.snap_r2_drop), (c.pump_every, 50, 1e-3, 1e-4));
        assert!(c.snap_every > 0, "snap is on by default");
        let off = Config { pop_intake: 60, pop_champion: 20, head: 8, snap_every: 0, ..c };
        let engine = Engine::new(off, sine_data()).expect("engine");
        assert_eq!(engine.table.symbols, SymbolTable::wide(2).symbols, "with snap off the symbol table is the plain one");
        assert_eq!(engine.snap_counts(), SnapCounts::default());
    }
}
