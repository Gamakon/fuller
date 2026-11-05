//! MASTER 6's E-GRAPH (plan `docs/PLAN_futhark_kingdom.md` section 7, task F5), for A7 and every view alike: one sort
//! a data type, one constructor a symbol, the rules of `master/6/rules.tsv`, all GENERATED
//! (`kingdoms/futhark/sig/<problem>/m6/egraph.*`, written by `gen_tables.py`, computed again by
//! [`crate::futhark_view::egraph6`]).
//!
//! * **The variable constructor on the resolved variable.** A `var@T` enters the e-graph as `(Xvar__T depth ordinal)`,
//!   the identity of the variable it names (G3's [`Binder`]), never its index word: an index word is relative to its
//!   environment, so the same word names different variables in different scopes and different words the same one
//!   (plan 7.5 risk 3). With no variable of `T` in scope it enters as `(Kvar__T)`, the constant it reads.
//! * **The bridge writes an extracted variable back** as the index word that names it in the DESTINATION's
//!   environment ([`Egraph6::place`]), checked by decoding again: the variable is in scope where it is written and is
//!   the same resolved variable.
//! * **Saturation only on trees with no failing requirement** ([`Egraph6::saturable`]): a rule is an identity under an
//!   adapter only where no adapter sits between its sides. A candidate with a failing requirement is no variant.
//! * **The fold's key is the resolved tree**: [`FoldEgraph6`] imports by the resolved variable, so two genes whose
//!   words name the same variables are one root, and a shared part's variables take one word that names their
//!   resolved variable wherever the part is used ([`crate::saturated_fold::common_word`]).

use std::collections::{BTreeMap, BTreeSet};

use egglog::extract::hff_extract;
use egglog::prelude::exprs;
use egglog::{EGraph, Term, TermDag, TermId};

use crate::futhark_table::{Header6, Kind6, Kingdom6, Program6};
use crate::futhark_view::egraph6::{egraph6, Egraph6Texts, RULESET6};
use crate::geneframe::KingdomTables;
use crate::kingdom_contract::{KingdomContract, SymbolId};
use crate::saturated_fold::{EgraphKingdom, VariableSite};
use crate::typed_environment::{in_scope, Binder, Choice, Decoder, Decoding, Entry, Node, ResolvedTree};
use crate::typed_gep::Tree;

fn bug(what: String) -> String {
    format!("our software has a bug: master 6's e-graph: {what}")
}

/// What a constructor reads: a node's inputs, a `const` row's pool index, a variable's resolved identity, or the
/// constant a `var` row reads with no variable of its data type in scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reads {
    Node,
    Dc,
    Binder,
    Constant,
}

impl Reads {
    fn parse(text: &str) -> Option<Reads> {
        Some(match text {
            "node" => Reads::Node,
            "dc" => Reads::Dc,
            "binder" => Reads::Binder,
            "constant" => Reads::Constant,
            _ => return None,
        })
    }
}

/// A var node's word while it is still a resolved identity (an extracted form before [`Egraph6::place`]): the high bit
/// set, the binding depth and the ordinal below it. An index word is below 2^16, so the two never meet.
const PACKED: u32 = 1 << 31;
/// The packed word of the constant reading (`K<name>`).
const PACKED_CONSTANT: u32 = u32::MAX;

/// The packed word of a resolved variable, `None` when its depth or ordinal does not fit.
fn pack(b: Binder) -> Option<u32> {
    (b.depth < 0x7FFF && b.ordinal <= 0xFFFF).then_some(PACKED | (b.depth << 16) | b.ordinal)
}

/// What a var node's word is: an index word, a resolved variable, or the constant reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Word {
    Index,
    Resolved(Binder),
    Constant,
}

fn unpack(w: u32) -> Word {
    if w == PACKED_CONSTANT {
        Word::Constant
    } else if w & PACKED != 0 {
        Word::Resolved(Binder { depth: (w & !PACKED) >> 16, ordinal: w & 0xFFFF })
    } else {
        Word::Index
    }
}

/// One problem's master 6 e-graph: its kingdom, its renderer's parts and its generated datatype, rules and bridge.
#[derive(Debug, Clone)]
pub struct Egraph6 {
    pub problem: String,
    pub kingdom: Kingdom6,
    pub header: Header6,
    pub tables: KingdomTables,
    pub contract: KingdomContract,
    pub entry: Entry,
    decoder: Decoder,
    /// The generated texts (`egraph.datatype`, `egraph.rules`, `egraph.rows`, `egraph.sorts`).
    pub texts: Egraph6Texts,
    ctor_of: BTreeMap<(SymbolId, Reads), String>,
    of_ctor: BTreeMap<String, (SymbolId, Reads)>,
}

/// The e-graph variants of one tree ([`Egraph6::variants`]).
#[derive(Debug, Clone, Default)]
pub struct Variants6 {
    /// Whether the tree was saturated: a tree with a failing requirement is not.
    pub saturated: bool,
    /// The variants, smallest first, then by their rendered text.
    pub trees: Vec<Tree>,
    /// Extracted terms that are no program where they stand (a variable not in scope, a failing requirement).
    pub dropped: usize,
}

impl Egraph6 {
    /// The e-graph of a committed master 6 view (`kingdoms/futhark/sig/<problem>/m6/`).
    pub fn load(problem: &str) -> Result<Egraph6, String> {
        let view = crate::futhark_view::master6::committed(problem)?;
        let kingdom = view.kingdom(problem)?;
        let texts = egraph6(&view)?;
        Egraph6::new(problem, kingdom, Header6::of_spec(&view.spec), texts)
    }

    /// The e-graph of a kingdom and its generated texts; the bridge is read from `egraph.rows`, never derived again.
    pub fn new(problem: &str, kingdom: Kingdom6, header: Header6, texts: Egraph6Texts) -> Result<Egraph6, String> {
        let (tables, contract) = (kingdom.tables()?, kingdom.contract()?);
        let entry = Entry::of_kingdom6(&kingdom);
        let mut ctor_of = BTreeMap::new();
        let mut of_ctor = BTreeMap::new();
        for line in texts.rows.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            let [name, ctor, _, reads] = f[..] else { return Err(bug(format!("{problem}: the bridge line {line:?} is malformed"))) };
            let symbol = kingdom.symbols.iter().position(|s| s.name == name).ok_or_else(|| bug(format!("{problem}: the bridge names {name}, which is no symbol")))? as SymbolId;
            let reads = Reads::parse(reads).ok_or_else(|| bug(format!("{problem}: the bridge line {line:?} reads {reads:?}")))?;
            if ctor_of.insert((symbol, reads), ctor.to_string()).is_some() || of_ctor.insert(ctor.to_string(), (symbol, reads)).is_some() {
                return Err(bug(format!("{problem}: the bridge line {line:?} repeats a symbol or a constructor")));
            }
        }
        for (s, sym) in kingdom.symbols.iter().enumerate() {
            let want: &[Reads] = match sym.kind {
                Kind6::Const => &[Reads::Dc],
                Kind6::Var => &[Reads::Binder, Reads::Constant],
                Kind6::Function | Kind6::Terminal => &[Reads::Node],
            };
            if want.iter().any(|r| !ctor_of.contains_key(&(s as SymbolId, *r))) {
                return Err(bug(format!("{problem}: the symbol {} has no constructor for every reading {want:?}", sym.name)));
            }
        }
        let decoder = Decoder::new(&tables, &contract).map_err(|d| d.join("\n"))?;
        decoder.check_entry(&entry).map_err(|d| d.join("\n"))?;
        Ok(Egraph6 { problem: problem.to_string(), kingdom, header, tables, contract, entry, decoder, texts, ctor_of, of_ctor })
    }

    fn ctor(&self, symbol: SymbolId, reads: Reads) -> Result<&str, String> {
        self.ctor_of.get(&(symbol, reads)).map(String::as_str).ok_or_else(|| bug(format!("{}: symbol {symbol} has no {reads:?} constructor", self.problem)))
    }

    /// The decoding of a tree (passes 2 to 4).
    pub fn decode(&self, tree: &Tree) -> Result<Decoding, String> {
        self.decoder.decode(&self.entry, tree)
    }

    /// The resolved tree.
    pub fn resolved(&self, tree: &Tree) -> Result<ResolvedTree, String> {
        Ok(self.decode(tree)?.resolved())
    }

    /// The program of a tree (passes 2 to 5).
    pub fn program(&self, tree: &Tree) -> Result<Program6, String> {
        self.kingdom.program6(&self.header, &self.decode(tree)?)
    }

    /// SATURATION ONLY ON TREES WITH NO FAILING REQUIREMENT: no adapter, no work-bound substitution, no adapter on the
    /// entry's result.
    pub fn saturable(&self, tree: &Tree) -> Result<bool, String> {
        Ok(self.decode(tree)?.failures().is_empty())
    }

    /// A tree as an egglog term: a variable by its resolved identity, a constant by its pool index.
    pub fn import(&self, tree: &Tree) -> Result<String, String> {
        let d = self.decode(tree)?;
        let mut out = String::new();
        self.term(&d.root, tree, &mut out)?;
        Ok(out)
    }

    fn term(&self, n: &Node, t: &Tree, out: &mut String) -> Result<(), String> {
        let sym = self.kingdom.symbols.get(t.symbol as usize).ok_or_else(|| bug(format!("{}: symbol {} is not in the kingdom", self.problem, t.symbol)))?;
        match (sym.kind, n.choice) {
            (Kind6::Var, Some(Choice::Variable(b))) => out.push_str(&format!("({} {} {})", self.ctor(t.symbol, Reads::Binder)?, b.depth, b.ordinal)),
            (Kind6::Var, Some(Choice::Constant)) => out.push_str(&format!("({})", self.ctor(t.symbol, Reads::Constant)?)),
            (Kind6::Const, Some(Choice::Dc(i))) => out.push_str(&format!("({} {i})", self.ctor(t.symbol, Reads::Dc)?)),
            (Kind6::Var | Kind6::Const, c) => return Err(bug(format!("{}: {} reads as {c:?}", self.problem, sym.name))),
            (Kind6::Function | Kind6::Terminal, _) => {
                out.push('(');
                out.push_str(self.ctor(t.symbol, Reads::Node)?);
                for (kn, kt) in n.kids.iter().zip(&t.kids) {
                    out.push(' ');
                    self.term(kn, kt, out)?;
                }
                out.push(')');
            }
        }
        Ok(())
    }

    /// An egglog term as a tree spelled by the bridge, every variable still its resolved identity (packed); `None`
    /// when a constructor is no symbol.
    fn spelled(&self, termdag: &TermDag, id: TermId) -> Option<Tree> {
        let Term::App(op, kids) = termdag.get(id) else { return None };
        let (symbol, reads) = *self.of_ctor.get(op.as_str())?;
        let int = |k: &TermId| match termdag.get(*k) {
            Term::Lit(egglog::ast::Literal::Int(i)) => u32::try_from(*i).ok(),
            _ => None,
        };
        let leaf = |w: u32| Some(Tree { symbol, const_index: Some(w), kids: Vec::new() });
        match (reads, kids.as_slice()) {
            (Reads::Dc, [i]) => leaf(int(i)?),
            (Reads::Binder, [d, o]) => leaf(pack(Binder { depth: int(d)?, ordinal: int(o)? })?),
            (Reads::Constant, []) => leaf(PACKED_CONSTANT),
            (Reads::Node, _) => Some(Tree { symbol, const_index: None, kids: kids.iter().map(|k| self.spelled(termdag, *k)).collect::<Option<Vec<_>>>()? }),
            _ => None,
        }
    }

    /// THE BRIDGE BACK: an extracted form's variables written as the index words that name them in the destination's
    /// environment, and checked by decoding again (each reads as the resolved variable it was extracted as: a failure is
    /// a bug). A var node whose word is already an index word (a submitted tree's node) keeps its variable. `None` when
    /// the form is no program here: a variable not in scope where it stands (the e-graph shares a class across
    /// environments, so an extraction can carry a part to where its variable is not), or a failing requirement.
    pub fn place(&self, form: &Tree) -> Result<Option<Tree>, String> {
        let d = self.decode(form)?;
        let mut out = form.clone();
        let mut wanted: Vec<(Vec<u32>, Option<Binder>)> = Vec::new();
        fn walk(e: &Egraph6, n: &Node, t: &mut Tree, path: &mut Vec<u32>, wanted: &mut Vec<(Vec<u32>, Option<Binder>)>) -> Result<bool, String> {
            if e.contract.is_variable(t.symbol) {
                let word = t.const_index.ok_or_else(|| bug(format!("{}: a variable at {path:?} has no word", e.problem)))?;
                let want = match unpack(word) {
                    Word::Resolved(b) => Some(b),
                    Word::Constant => None,
                    Word::Index => match n.choice {
                        Some(Choice::Variable(b)) => Some(b),
                        _ => None,
                    },
                };
                match want {
                    Some(b) => match n.environment.index_of(n.data_type, b) {
                        Some(w) => t.const_index = Some(w),
                        None => return Ok(false),
                    },
                    None if in_scope(&n.environment, n.data_type).is_empty() => t.const_index = Some(0),
                    None => return Ok(false),
                }
                wanted.push((path.clone(), want));
                return Ok(true);
            }
            for (i, (kn, kt)) in n.kids.iter().zip(t.kids.iter_mut()).enumerate() {
                path.push(i as u32);
                let ok = walk(e, kn, kt, path, wanted)?;
                path.pop();
                if !ok {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        if !walk(self, &d.root, &mut out, &mut Vec::new(), &mut wanted)? {
            return Ok(None);
        }
        let again = self.decode(&out)?;
        for (path, want) in wanted {
            let got = again.node(&path).and_then(|n| n.choice);
            let expected = Some(want.map_or(Choice::Constant, Choice::Variable));
            if got != expected {
                return Err(bug(format!("{}: the variable written at {path:?} reads as {got:?}, not {expected:?}", self.problem)));
            }
        }
        Ok(again.failures().is_empty().then_some(out))
    }

    /// The variable nodes of a placed form: each one's path, its place among the variables of its data type in scope
    /// and their number.
    pub fn variables(&self, form: &Tree) -> Result<Vec<(Vec<u32>, VariableSite)>, String> {
        let d = self.decode(form)?;
        let mut out = Vec::new();
        fn walk(e: &Egraph6, n: &Node, path: &mut Vec<u32>, out: &mut Vec<(Vec<u32>, VariableSite)>) -> Result<(), String> {
            if e.contract.is_variable(n.symbol) {
                let vs = in_scope(&n.environment, n.data_type);
                let place = match n.choice {
                    Some(Choice::Variable(b)) => vs.iter().position(|v| v.binder == b).ok_or_else(|| bug(format!("{}: the variable at {path:?} is not in scope", e.problem)))?,
                    _ => 0,
                };
                out.push((path.clone(), VariableSite { place: place as u32, count: vs.len() as u32 }));
            }
            for (i, k) in n.kids.iter().enumerate() {
                path.push(i as u32);
                walk(e, k, path, out)?;
                path.pop();
            }
            Ok(())
        }
        walk(self, &d.root, &mut Vec::new(), &mut out)?;
        Ok(out)
    }

    /// An egglog term as a tree of the kingdom, its variables placed where they stand ([`Egraph6::place`]); `None`
    /// when it is no program here.
    pub fn to_tree(&self, termdag: &TermDag, id: TermId) -> Result<Option<Tree>, String> {
        match self.spelled(termdag, id) {
            Some(t) => self.place(&t),
            None => Err(bug(format!("{}: an extracted term has a constructor the bridge does not spell", self.problem))),
        }
    }

    /// A term's cost: its e-nodes, a constant's pool index counted as one more, a variable one node (master 5 counted
    /// a constant two and a variable one, so the variants rank as master 5's do).
    fn cost(&self, termdag: &TermDag, id: TermId) -> u64 {
        match termdag.get(id) {
            Term::App(op, kids) => {
                if matches!(self.of_ctor.get(op.as_str()), Some((_, Reads::Binder | Reads::Constant))) {
                    return 1;
                }
                1 + kids.iter().map(|k| self.cost(termdag, *k)).sum::<u64>()
            }
            _ => 1,
        }
    }

    /// The e-graph, loaded, with the tree asserted as `__root` and `iters` iterations of the ruleset run.
    fn saturated(&self, term: &str, iters: u32) -> Result<EGraph, String> {
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(None, &self.texts.datatype).map_err(|e| bug(format!("{}: the datatype does not load: {e}", self.problem)))?;
        egraph.parse_and_run_program(None, &self.texts.rules).map_err(|e| bug(format!("{}: the ruleset does not load: {e}", self.problem)))?;
        egraph.parse_and_run_program(None, &format!("(let __root {term})\n(run-schedule (repeat {iters} (run {RULESET6})))")).map_err(|e| bug(format!("{}: the term does not load: {e}", self.problem)))?;
        Ok(egraph)
    }

    /// THE E-GRAPH VARIANTS of a tree, as master 5's `futhark_sig::variants`: `iters` iterations of the view's ruleset
    /// (bounded), the ranked members of the root's class (HFF extraction, `members` a class), up to `n` distinct
    /// programs other than the tree itself, smallest first and then by their rendered text. Every variant is placed
    /// where it stands and has no failing requirement. A tree with a failing requirement is not saturated.
    pub fn variants(&self, tree: &Tree, iters: u32, members: usize, n: usize) -> Result<Variants6, String> {
        if !self.saturable(tree)? {
            return Ok(Variants6::default());
        }
        let term = self.import(tree)?;
        let mut egraph = self.saturated(&term, iters)?;
        let (sort, value) = egraph.eval_expr(&exprs::var("__root")).map_err(|e| bug(format!("{}: the root does not evaluate: {e}", self.problem)))?;
        let mut termdag = TermDag::default();
        let score = |td: &TermDag, t: TermId| self.cost(td, t) as f64;
        let ranked = hff_extract(&egraph, &mut termdag, value, sort, &score, members);
        let own = self.program(tree)?.text;
        let mut out: Vec<(u64, String, Tree)> = Vec::new();
        let mut texts: BTreeSet<String> = BTreeSet::from([own]);
        let mut dropped = 0;
        for (_, id) in ranked {
            let Some(t) = self.to_tree(&termdag, id)? else {
                dropped += 1;
                continue;
            };
            let text = self.program(&t)?.text;
            if texts.insert(text.clone()) {
                out.push((self.cost(&termdag, id), text, t));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        out.truncate(n);
        Ok(Variants6 { saturated: true, trees: out.into_iter().map(|(_, _, t)| t).collect(), dropped })
    }
}

/// A master 6 e-graph as the saturated fold needs it ([`crate::saturated_fold::fold_saturated`]).
pub struct FoldEgraph6<'a> {
    pub egraph: &'a Egraph6,
}

impl EgraphKingdom for FoldEgraph6<'_> {
    fn programs(&self) -> Vec<&str> {
        vec![&self.egraph.texts.datatype, &self.egraph.texts.rules]
    }
    fn rulesets(&self) -> Vec<&str> {
        vec![RULESET6]
    }
    fn import(&self, tree: &Tree) -> Result<String, String> {
        self.egraph.import(tree)
    }
    fn row(&self, constructor: &str, literals: &[&str]) -> Option<(SymbolId, Option<u32>)> {
        let (symbol, reads) = *self.egraph.of_ctor.get(constructor)?;
        let int = |t: &str| t.parse::<u32>().ok();
        match (reads, literals) {
            (Reads::Node, []) => Some((symbol, None)),
            (Reads::Dc, [i]) => Some((symbol, Some(int(i)?))),
            (Reads::Binder, [d, o]) => Some((symbol, Some(pack(Binder { depth: int(d)?, ordinal: int(o)? })?))),
            (Reads::Constant, []) => Some((symbol, Some(PACKED_CONSTANT))),
            _ => None,
        }
    }
    fn saturable(&self, tree: &Tree) -> Result<bool, String> {
        self.egraph.saturable(tree)
    }
    fn place(&self, form: &Tree) -> Result<Option<Tree>, String> {
        self.egraph.place(form)
    }
    fn variables(&self, form: &Tree) -> Result<Vec<(Vec<u32>, VariableSite)>, String> {
        self.egraph.variables(form)
    }
}

#[cfg(test)]
mod tests;
