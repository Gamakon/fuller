//! Memory versions for the reader (`docs/PLAN_wgsl_lineage.md` §2): every
//! location a function can store to gets a version number that changes at
//! every store, at every join of control flow where the incoming versions
//! differ (a phi), at a loop header and a loop exit for every location the
//! loop stores, at a barrier for every location the barrier makes visible,
//! and at a call for everything the callee stores. A load is named with
//! the version current at its program point, so two loads with equal text
//! read the same value by construction.
//!
//! A *location* is the base of a pointer chain as `reader::pointer_target`
//! names it: `buffer.xs`, `workgroup.w`, `local.i@1`, `arg.p`. Whole-array
//! granularity: `a[i] = x` bumps `a` for every index.
//!
//! The walk is over naga's structured statement tree, so the joins are
//! known without a CFG: an `if` joins its two arms, a `switch` its cases
//! (a fall-through case flows into the next), a loop header joins the
//! entry with the back edge (every `continue` and the end of `continuing`)
//! and a loop exit joins every `break` and the `break_if`. Dead code after
//! a `break`, `continue`, `return` or `discard` is still walked (the
//! rebuild needs its roots) but does not join.

use std::collections::{BTreeMap, BTreeSet};

use naga::{Expression, Function, Handle, Module, Statement};

/// What a function stores, in its own terms: locations (globals, and its
/// own locals, which matter only inside it) and the pointer parameters it
/// stores through (substituted per call site by the caller).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Effects {
    pub locations: BTreeSet<String>,
    pub params: BTreeSet<u32>,
}

/// The effects of every function and entry point of a module, in the
/// reader's function order (module functions, then entry points), closed
/// over calls by iteration to a fixpoint (WGSL has no recursion, so the
/// iteration ends).
pub fn module_effects(module: &Module) -> Vec<Effects> {
    let n = module.functions.len() + module.entry_points.len();
    let mut effects = vec![Effects::default(); n];
    loop {
        let mut changed = false;
        for (i, f) in functions(module).enumerate() {
            let mut e = Effects::default();
            block_effects(module, f, &effects, &f.body, &mut e);
            if e != effects[i] {
                effects[i] = e;
                changed = true;
            }
        }
        if !changed {
            return effects;
        }
    }
}

/// Module functions then entry points: the reader's numbering.
pub fn functions(module: &Module) -> impl Iterator<Item = &Function> {
    module.functions.iter().map(|(_, f)| f).chain(module.entry_points.iter().map(|ep| &ep.function))
}

/// Every location `block` bumps (stores, atomics, barriers, image stores,
/// calls with the callee's effects substituted), recursively.
pub fn block_effects(module: &Module, f: &Function, callees: &[Effects], block: &naga::Block, out: &mut Effects) {
    for stmt in block.iter() {
        match stmt {
            Statement::Store { pointer, .. } | Statement::Atomic { pointer, .. } => out.add(location_of(module, f, *pointer)),
            Statement::ImageStore { image, .. } => out.add(location_of(module, f, *image)),
            Statement::Barrier(flags) => barrier_locations(module, *flags, &mut out.locations),
            // `workgroupUniformLoad` has barrier semantics (naga's doc).
            Statement::WorkGroupUniformLoad { .. } => barrier_locations(module, naga::Barrier::WORK_GROUP, &mut out.locations),
            Statement::Call { function, arguments, .. } => {
                let callee = &callees[function.index()];
                for loc in &callee.locations {
                    if !loc.starts_with("local.") {
                        out.add(loc.clone());
                    }
                }
                for &p in &callee.params {
                    if let Some(&a) = arguments.get(p as usize) {
                        out.add(location_of(module, f, a));
                    }
                }
            }
            Statement::Block(b) => block_effects(module, f, callees, b, out),
            Statement::If { accept, reject, .. } => {
                block_effects(module, f, callees, accept, out);
                block_effects(module, f, callees, reject, out);
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    block_effects(module, f, callees, &c.body, out);
                }
            }
            Statement::Loop { body, continuing, .. } => {
                block_effects(module, f, callees, body, out);
                block_effects(module, f, callees, continuing, out);
            }
            Statement::Emit(_)
            | Statement::Break
            | Statement::Continue
            | Statement::Return { .. }
            | Statement::Kill
            | Statement::RayQuery { .. }
            | Statement::SubgroupBallot { .. }
            | Statement::SubgroupGather { .. }
            | Statement::SubgroupCollectiveOperation { .. } => {}
        }
    }
}

impl Effects {
    /// Record a bumped location; a pointer parameter is recorded by index so
    /// the caller substitutes its own argument.
    fn add(&mut self, loc: String) {
        match loc.strip_prefix("arg#") {
            Some(i) => {
                self.params.insert(i.parse().expect("arg# carries the index"));
            }
            None => {
                self.locations.insert(loc);
            }
        }
    }
}

/// The locations a barrier makes visible: every workgroup global for a
/// workgroup barrier, every storage global for a storage barrier.
pub fn barrier_locations(module: &Module, flags: naga::Barrier, out: &mut BTreeSet<String>) {
    for (h, g) in module.global_variables.iter() {
        let hit = match g.space {
            naga::AddressSpace::WorkGroup => flags.contains(naga::Barrier::WORK_GROUP),
            naga::AddressSpace::Storage { .. } => flags.contains(naga::Barrier::STORAGE),
            _ => false,
        };
        if hit {
            out.insert(global_location(module, h));
        }
    }
}

pub fn global_location(module: &Module, g: Handle<naga::GlobalVariable>) -> String {
    let gv = &module.global_variables[g];
    let name = gv.name.clone().unwrap_or_else(|| format!("global{}", g.index()));
    let kind = match gv.space {
        naga::AddressSpace::Storage { .. } => "buffer",
        naga::AddressSpace::Uniform => "uniform",
        naga::AddressSpace::WorkGroup => "workgroup",
        naga::AddressSpace::Private => "private",
        naga::AddressSpace::PushConstant => "push",
        naga::AddressSpace::Handle | naga::AddressSpace::Function => "global",
    };
    format!("{kind}.{name}")
}

/// The location at the base of a pointer chain, as `pointer_target` names
/// it (`buffer.xs`, `local.i@1`); a pointer parameter is `arg#<i>` here so
/// [`Effects::add`] can file it by index, and `arg.<name>` as a location
/// once substituted. An unresolvable base (a pointer computed some other
/// way) is `naga.<Variant>`, its own location.
pub fn location_of(module: &Module, f: &Function, pointer: Handle<Expression>) -> String {
    match &f.expressions[pointer] {
        Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => location_of(module, f, *base),
        Expression::GlobalVariable(g) => global_location(module, *g),
        Expression::LocalVariable(l) => format!("local.{}@{}", f.local_variables[*l].name.clone().unwrap_or_else(|| "local".into()), l.index()),
        Expression::FunctionArgument(i) => format!("arg#{i}"),
        other => {
            let dbg = format!("{other:?}");
            format!("naga.{}", dbg.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("?"))
        }
    }
}

/// The location name a pointer parameter has inside its function, as the
/// reader's `pointer_target` spells it.
pub fn param_location(f: &Function, i: u32) -> String {
    format!("arg.{}", f.arguments[i as usize].name.clone().unwrap_or_else(|| format!("{i}")))
}

/// The versions current at one program point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    /// Location → version; absent means 0, the version at function entry.
    pub current: BTreeMap<String, u32>,
    /// Whether this point is reachable in program order (false after a
    /// `break`, `continue`, `return` or `discard`).
    pub live: bool,
}

impl State {
    pub fn entry() -> Self {
        State { current: BTreeMap::new(), live: true }
    }

    pub fn version(&self, loc: &str) -> u32 {
        self.current.get(loc).copied().unwrap_or(0)
    }
}

/// Fresh version numbers, one counter per location, function-wide.
#[derive(Debug, Default)]
pub struct Counters {
    next: BTreeMap<String, u32>,
}

impl Counters {
    pub fn fresh(&mut self, loc: &str) -> u32 {
        let n = self.next.entry(loc.to_string()).or_insert(0);
        *n += 1;
        *n
    }

    /// Versions created per location (the count of bumps and phis).
    pub fn created(&self) -> &BTreeMap<String, u32> {
        &self.next
    }
}

/// A store, atomic, barrier or call effect: a new version of `loc`.
pub fn bump(state: &mut State, counters: &mut Counters, loc: &str) {
    let v = counters.fresh(loc);
    state.current.insert(loc.to_string(), v);
}

/// The join of several incoming states: where every live predecessor agrees
/// on a location's version it is kept, otherwise the location takes a fresh
/// version (a phi). With no live predecessor the result is dead and carries
/// the first state's versions, so dead code still reads deterministically.
pub fn merge(states: &[State], counters: &mut Counters) -> State {
    let live: Vec<&State> = states.iter().filter(|s| s.live).collect();
    if live.is_empty() {
        return State { current: states.first().map(|s| s.current.clone()).unwrap_or_default(), live: false };
    }
    let mut keys: BTreeSet<&String> = BTreeSet::new();
    for s in &live {
        keys.extend(s.current.keys());
    }
    let mut out = State { current: BTreeMap::new(), live: true };
    for loc in keys {
        let first = live[0].version(loc);
        if live.iter().all(|s| s.version(loc) == first) {
            if first != 0 {
                out.current.insert(loc.clone(), first);
            }
        } else {
            bump(&mut out, counters, loc);
        }
    }
    out
}

/// Locations a loop stores (its header and exit phis).
pub fn loop_stores(module: &Module, f: &Function, callees: &[Effects], body: &naga::Block, continuing: &naga::Block) -> BTreeSet<String> {
    let mut e = Effects::default();
    block_effects(module, f, callees, body, &mut e);
    block_effects(module, f, callees, continuing, &mut e);
    let mut locs = e.locations;
    for p in e.params {
        locs.insert(param_location(f, p));
    }
    locs
}

/// Everything a function bumps anywhere (its own effects with its pointer
/// parameters named): the locations whose loads carry a version.
pub fn bumped_locations(f: &Function, effects: &Effects) -> BTreeSet<String> {
    let mut locs = effects.locations.clone();
    for &p in &effects.params {
        locs.insert(param_location(f, p));
    }
    locs
}

/// `name` without its trailing `@v<n>`, and the version if it had one.
pub fn split_version(name: &str) -> (&str, Option<u32>) {
    if let Some((head, tail)) = name.rsplit_once("@v") {
        if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(v) = tail.parse() {
                return (head, Some(v));
            }
        }
    }
    (name, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_version_strips_only_a_trailing_version() {
        assert_eq!(split_version("load.local.i@1@v3"), ("load.local.i@1", Some(3)));
        assert_eq!(split_version("load.buffer.xs.#@v12"), ("load.buffer.xs.#", Some(12)));
        assert_eq!(split_version("load.uniform.hp.rows"), ("load.uniform.hp.rows", None));
        assert_eq!(split_version("load.local.i@1"), ("load.local.i@1", None));
        assert_eq!(split_version("let.v@7"), ("let.v@7", None));
    }

    #[test]
    fn merge_keeps_agreed_versions_and_phis_the_rest() {
        let mut c = Counters::default();
        let mut a = State::entry();
        let mut b = State::entry();
        bump(&mut a, &mut c, "local.x@0");
        bump(&mut a, &mut c, "local.y@1");
        bump(&mut b, &mut c, "local.y@1");
        // x: 1 in a, 0 in b → phi (fresh 2); y: 1 in a, 2 in b → phi (fresh 3).
        let m = merge(&[a.clone(), b.clone()], &mut c);
        assert_eq!(m.version("local.x@0"), 2);
        assert_eq!(m.version("local.y@1"), 3);
        // A dead arm does not join.
        b.live = false;
        let m = merge(&[a.clone(), b], &mut c);
        assert_eq!((m.version("local.x@0"), m.version("local.y@1"), m.live), (1, 1, true));
        // Equal arms: nothing fresh.
        let m = merge(&[a.clone(), a.clone()], &mut c);
        assert_eq!(m, a);
    }
}
