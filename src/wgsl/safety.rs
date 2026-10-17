//! THE GENERAL SAFETY LAYER (`docs/PLAN_general_safety_layer.md`, Revision 1,
//! orders (b) and (c)): a proof over the naga module of ANY compute kernel,
//! with no code or declaration written for its problem, that
//!
//! 1. every store index of an output is affine over the thread's coordinates
//!    (`wid`, `lid`; `gid = W * wid + lid`), the loop counters and bounded
//!    digits (a `min(e, c)` or `e & c` of an invariant c, a vector's lane),
//!    with every coefficient a polynomial in the dispatch's invariants
//!    (uniform fields, read-only storage read at an invariant index, the
//!    workgroup count), and that its arithmetic does not wrap (proved on
//!    intervals from the [`Contract`] and the guards on the store's path);
//! 2. the write sets are disjoint: no two invocations write one cell, over
//!    every pair of store sites and every iteration of their loops, per bound
//!    buffer, and two store sites of DIFFERENT index forms never write one
//!    cell, not even in one invocation (a mirror pair whose guards both admit
//!    the diagonal writes it twice);
//! 3. no invocation reads an output cell another invocation writes;
//! 4. in workgroup memory, no two invocations of a workgroup touch one
//!    element (one of them writing) between two workgroup barriers;
//! 5. a barrier and a subgroup operation stand in uniform control flow.
//!
//! THE DISJOINTNESS PROOF. Two store sites A and B, one copy of the variables
//! for each invocation. Their index forms are split together, mixed-radix
//! fashion: by a coefficient c both forms are `c * H + L` with `0 <= L <= c - 1`
//! (L's bound from the guards on the site's own path, or the variables'
//! ranges), so `A = B` gives `H_A = H_B` and `L_A = L_B`; the split recurses
//! and the leaves are linear equations between the two copies. With the
//! guards of each site (linear inequalities between coordinates: `i < j`,
//! `i <= j`, `col < cfg.d`) and the variables' ranges, Fourier–Motzkin
//! elimination shows the system has no solution in which the two invocations
//! differ (or, for two different forms, no solution at all). The Gram's own
//! cell `row * d + col`, its 4 x 4 blocks, nluca's `out[gid + offset]` and a
//! mirror pair `h[i*d + j]`, `h[j*d + i]` under `i < j` are instances; a stride
//! that may be zero, `wid * W' + lid` with the wrong W', a wrapping offset and
//! a non-affine index are not proved.
//!
//! The proof is conservative: what it cannot show is illegal, by name. An
//! exhaustive small-domain test enumerating every invocation's writes on the
//! CPU interpreter cross-checks it (a bug-finder, not the proof).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use naga::{AddressSpace, BinaryOperator, Block, Expression, Function, Handle, MathFunction, Module, ScalarKind, Statement, TypeInner};

use crate::wgsl::interface::{compute_entries, infer_entry, ModuleInterface, Use};

pub const U32_MAX: i128 = u32::MAX as i128;
const BIG: i128 = 1 << 100;

// ------------------------------------------------------------------ contract

/// The dispatch contract (Revision 1 §1): what the problem's data says about
/// a dispatch. The workgroup size is the module's own; the workgroup counts,
/// the admissible values of the invariants (`cfg.d`, `buffer_6_global[0]`,
/// `arrayLength(h)`) and the binding of each binding to a buffer range are
/// data. An invariant the contract does not name ranges over every `u32`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contract {
    /// Workgroups per axis: (least, most).
    pub workgroups: [(i128, i128); 3],
    pub params: BTreeMap<String, (i128, i128)>,
    /// A binding bound to a range of a buffer another binding also sees:
    /// binding → (buffer, first element). Bindings not named are each their
    /// own buffer.
    pub aliases: BTreeMap<String, (String, i128)>,
    /// The ranges a kernel that writes the whole output (`// mode full`)
    /// runs under instead (the Gram's: one band, r_off 0).
    pub full: BTreeMap<String, (i128, i128)>,
    /// The bindings that are the design's own state (`arena index_u32
    /// index_f32`: v24's index arenas). Read by layer 7 in arena mode only.
    pub arena: BTreeSet<String>,
    /// THE SWITCH: layer 7's mode. False (strict, the default): every
    /// binding under the ownership rules. True (arena): on the `arena`
    /// bindings, atomic accesses at any index and plain loads and stores at
    /// data-dependent indexes, each index proved in bounds by a mask or a
    /// clamp; every other binding, and workgroup memory, as strict.
    pub arena_mode: bool,
}

impl Default for Contract {
    fn default() -> Self {
        Contract { workgroups: [(1, 65535); 3], params: BTreeMap::new(), aliases: BTreeMap::new(), full: BTreeMap::new(), arena: BTreeSet::new(), arena_mode: false }
    }
}

impl Contract {
    /// The contract's text form, one fact a line (`#` comments):
    /// `workgroups x 1 65535`, `param cfg.d 1 8192`, `alias b buffer 128`.
    pub fn parse(text: &str) -> Result<Contract, String> {
        let mut c = Contract::default();
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let w: Vec<&str> = line.split_whitespace().collect();
            let num = |s: &str| s.replace('_', "").parse::<i128>().map_err(|e| format!("contract line {}: {s}: {e}", n + 1));
            match w.as_slice() {
                ["workgroups", axis, lo, hi] => {
                    let i = "xyz".find(axis).ok_or_else(|| format!("contract line {}: axis {axis}", n + 1))?;
                    c.workgroups[i] = (num(lo)?, num(hi)?);
                }
                ["param", name, lo, hi] => {
                    c.params.insert(name.to_string(), (num(lo)?, num(hi)?));
                }
                ["alias", b, buffer, off] => {
                    c.aliases.insert(b.to_string(), (buffer.to_string(), num(off)?));
                }
                ["full", name, lo, hi] => {
                    c.full.insert(name.to_string(), (num(lo)?, num(hi)?));
                }
                ["arena", names @ ..] if !names.is_empty() => {
                    c.arena.extend(names.iter().map(|b| b.to_string()));
                }
                _ => return Err(format!("contract line {}: {line}", n + 1)),
            }
        }
        Ok(c)
    }

    /// The same contract with layer 7 in arena mode (`on`) or strict.
    pub fn with_arena_mode(&self, on: bool) -> Contract {
        Contract { arena_mode: on, ..self.clone() }
    }

    /// Whether layer 7 treats `binding` as an arena (arena mode, named).
    fn is_arena(&self, binding: &str) -> bool {
        self.arena_mode && self.arena.contains(binding)
    }

    /// The contract of a kernel that writes the whole output (`// mode full`).
    pub fn full(&self) -> Contract {
        let mut c = self.clone();
        for (k, v) in &self.full {
            c.params.insert(k.clone(), *v);
        }
        c
    }

    /// The Gram problem's contract with the role's own names for its fields
    /// (`cfg.d` is `p.nc` under the cosine interface); for a role without the
    /// Gram's fields, the contract that names nothing.
    pub fn for_role(role: &crate::wgsl::render::Role) -> Contract {
        if !role.fields.iter().any(|f| f.role == "d") {
            return Contract::default();
        }
        let g = gram_contract();
        let rename = |m: &BTreeMap<String, (i128, i128)>| -> BTreeMap<String, (i128, i128)> { m.iter().map(|(k, v)| (k.strip_prefix("cfg.").map_or_else(|| k.clone(), |r| role.field(r)), *v)).collect() };
        Contract { params: rename(&g.params), full: rename(&g.full), ..g }
    }

    pub fn param(&self, name: &str) -> (i128, i128) {
        self.params.get(name).copied().unwrap_or((0, U32_MAX))
    }

    fn buffer_of(&self, binding: &str) -> (String, i128) {
        self.aliases.get(binding).cloned().unwrap_or_else(|| (binding.to_string(), 0))
    }
}

/// What the layer refused, by rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsafe {
    pub rule: &'static str,
    pub detail: String,
}

impl fmt::Display for Unsafe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.rule, self.detail)
    }
}

fn unsafe_(rule: &'static str, detail: String) -> Unsafe {
    Unsafe { rule, detail }
}

/// What a passing module was proved over.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    pub write_sites: usize,
    pub read_sites: usize,
    pub workgroup_sites: usize,
    pub pairs: usize,
}

// --------------------------------------------------------------- polynomials

#[derive(Debug, Clone)]
struct SymInfo {
    name: String,
    lo: i128,
    hi: i128,
}

/// A polynomial with integer coefficients in the invariant symbols:
/// monomial (sorted symbol ids, with repetition) → coefficient.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Poly(BTreeMap<Vec<u32>, i128>);

fn sat(v: i128) -> i128 {
    v.clamp(-BIG, BIG)
}

impl Poly {
    fn konst(c: i128) -> Poly {
        let mut m = BTreeMap::new();
        if c != 0 {
            m.insert(Vec::new(), c);
        }
        Poly(m)
    }

    fn sym(id: u32) -> Poly {
        Poly(BTreeMap::from([(vec![id], 1)]))
    }

    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    fn as_const(&self) -> Option<i128> {
        match self.0.len() {
            0 => Some(0),
            1 => self.0.get(&Vec::new()).copied(),
            _ => None,
        }
    }

    fn add(&self, o: &Poly) -> Poly {
        let mut m = self.0.clone();
        for (k, v) in &o.0 {
            let e = m.entry(k.clone()).or_insert(0);
            *e = sat(*e + v);
            if *e == 0 {
                m.remove(k);
            }
        }
        Poly(m)
    }

    fn scale(&self, c: i128) -> Poly {
        if c == 0 {
            return Poly::default();
        }
        Poly(self.0.iter().map(|(k, v)| (k.clone(), sat(v.saturating_mul(c)))).collect())
    }

    fn sub(&self, o: &Poly) -> Poly {
        self.add(&o.scale(-1))
    }

    fn mul(&self, o: &Poly) -> Poly {
        let mut out = Poly::default();
        for (ka, va) in &self.0 {
            for (kb, vb) in &o.0 {
                let mut k = ka.clone();
                k.extend(kb);
                k.sort_unstable();
                out = out.add(&Poly(BTreeMap::from([(k, sat(va.saturating_mul(*vb)))])));
            }
        }
        out
    }

    /// The single monomial of a one-term polynomial.
    fn monomial(&self) -> Option<(Vec<u32>, i128)> {
        (self.0.len() == 1).then(|| self.0.iter().next().map(|(k, v)| (k.clone(), *v))).flatten()
    }

    /// Exact division by a monomial `c * m`.
    fn div_mono(&self, m: &[u32], c: i128) -> Option<Poly> {
        if c == 0 {
            return None;
        }
        let mut out = BTreeMap::new();
        for (k, v) in &self.0 {
            if v % c != 0 {
                return None;
            }
            let mut rest = k.clone();
            for s in m {
                let p = rest.iter().position(|x| x == s)?;
                rest.remove(p);
            }
            out.insert(rest, v / c);
        }
        Some(Poly(out))
    }

    /// The range over the symbols' ranges (interval arithmetic).
    fn range(&self, syms: &[SymInfo]) -> (i128, i128) {
        let (mut lo, mut hi) = (0i128, 0i128);
        for (k, v) in &self.0 {
            let (mut a, mut b) = (*v, *v);
            for s in k {
                let (sl, sh) = (syms[*s as usize].lo, syms[*s as usize].hi);
                let c = [sat(a.saturating_mul(sl)), sat(a.saturating_mul(sh)), sat(b.saturating_mul(sl)), sat(b.saturating_mul(sh))];
                a = *c.iter().min().expect("four");
                b = *c.iter().max().expect("four");
            }
            lo = sat(lo + a);
            hi = sat(hi + b);
        }
        (lo, hi)
    }

    /// Whether the polynomial is >= 0 wherever every symbol is in its range:
    /// each symbol written as its least value plus a non-negative part, every
    /// coefficient of the expansion non-negative; else its interval's least.
    fn nonneg(&self, syms: &[SymInfo]) -> bool {
        if self.range(syms).0 >= 0 {
            return true;
        }
        let mut expanded = Poly::default();
        for (k, v) in &self.0 {
            let mut term = Poly::konst(*v);
            for s in k {
                term = term.mul(&Poly::konst(syms[*s as usize].lo).add(&Poly::sym(*s)));
            }
            expanded = expanded.add(&term);
        }
        expanded.0.values().all(|v| *v >= 0)
    }

    fn show(&self, syms: &[SymInfo]) -> String {
        if self.0.is_empty() {
            return "0".into();
        }
        self.0
            .iter()
            .map(|(k, v)| {
                let names: Vec<&str> = k.iter().map(|s| syms[*s as usize].name.as_str()).collect();
                match (names.is_empty(), *v) {
                    (true, v) => v.to_string(),
                    (false, 1) => names.join("*"),
                    (false, v) => format!("{v}*{}", names.join("*")),
                }
            })
            .collect::<Vec<_>>()
            .join(" + ")
    }
}

// ------------------------------------------------------------- linear forms

/// A variable of an index: a workgroup's coordinate, a thread's coordinate
/// within its workgroup, or a digit (a loop counter, a clamped value, a lane).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Var {
    Wid(u8),
    Lid(u8),
    Dig(u32),
}

impl Var {
    fn thread(self) -> bool {
        matches!(self, Var::Wid(_) | Var::Lid(_))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Lin {
    t: BTreeMap<Var, Poly>,
    k: Poly,
}

impl Lin {
    fn konst(p: Poly) -> Lin {
        Lin { t: BTreeMap::new(), k: p }
    }

    fn var(v: Var, c: Poly) -> Lin {
        let mut t = BTreeMap::new();
        if !c.is_zero() {
            t.insert(v, c);
        }
        Lin { t, k: Poly::default() }
    }

    fn add(&self, o: &Lin) -> Lin {
        let mut t = self.t.clone();
        for (v, c) in &o.t {
            let n = t.get(v).map_or_else(|| c.clone(), |x| x.add(c));
            if n.is_zero() {
                t.remove(v);
            } else {
                t.insert(*v, n);
            }
        }
        Lin { t, k: self.k.add(&o.k) }
    }

    fn scale(&self, p: &Poly) -> Lin {
        let t: BTreeMap<Var, Poly> = self.t.iter().map(|(v, c)| (*v, c.mul(p))).filter(|(_, c)| !c.is_zero()).collect();
        Lin { t, k: self.k.mul(p) }
    }

    fn neg(&self) -> Lin {
        self.scale(&Poly::konst(-1))
    }

    fn sub(&self, o: &Lin) -> Lin {
        self.add(&o.neg())
    }

    fn invariant(&self) -> bool {
        self.t.is_empty()
    }
}

// ------------------------------------------------------------ abstract values

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Info {
    /// The same in every invocation of the dispatch.
    inv: bool,
    /// The same in every invocation of a workgroup.
    uni: bool,
    /// Arithmetic that may have wrapped made it.
    wrapped: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum Cond {
    Lt(Lin, Lin),
    Le(Lin, Lin),
    Eq(Lin, Lin),
    Ne(Lin, Lin),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
    Not(Box<Cond>),
    Const(bool),
    Opaque(Info),
}

#[derive(Debug, Clone, PartialEq)]
enum Base {
    Global(Handle<naga::GlobalVariable>),
    Local(usize, Handle<naga::LocalVariable>),
}

#[derive(Debug, Clone, PartialEq)]
struct Ptr {
    base: Base,
    path: Vec<Abs>,
}

#[derive(Debug, Clone, PartialEq)]
enum Abs {
    /// A u32 equal to the form (no wrap on the way).
    Lin(Lin),
    Cond(Cond),
    Vec(Vec<Abs>),
    Ptr(Ptr),
    /// Anything else, with an identity (copies of one value share it).
    Other(Info, u64),
}

#[derive(Debug, Clone)]
struct DigitInfo {
    name: String,
    hi: Option<Poly>,
    uni: bool,
    /// A quotient or remainder of a form by a constant c: (x, c, quotient):
    /// x = c * (x / c) + x % c, so equal quotients and equal remainders of
    /// one form in two invocations make the forms equal.
    def: Option<(Lin, i128, bool)>,
}

// ------------------------------------------------------------------- sites

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Write,
    Read,
    Atomic,
}

#[derive(Debug, Clone)]
struct Site {
    alloc: String,
    kind: Kind,
    form: Result<Lin, (&'static str, String)>,
    facts: Vec<Lin>,
    what: String,
}

// ------------------------------------------------------------------- frames

struct Frame<'m> {
    func: &'m Function,
    fkey: Option<Handle<Function>>,
    id: usize,
    ctx: String,
    args: Vec<Abs>,
    vals: HashMap<Handle<Expression>, Abs>,
    ret: Option<Abs>,
}

type LocalKey = (Option<Handle<Function>>, Handle<naga::LocalVariable>);

struct An<'m> {
    module: &'m Module,
    /// The compute entry point proved (one dispatch).
    entry: String,
    contract: &'m Contract,
    iface: ModuleInterface,
    wg: [u32; 3],
    syms: Vec<SymInfo>,
    sym_ix: HashMap<String, u32>,
    digits: Vec<DigitInfo>,
    digit_ix: HashMap<String, u32>,
    env: HashMap<(usize, Handle<naga::LocalVariable>), Abs>,
    privates: HashMap<Handle<naga::GlobalVariable>, Abs>,
    facts: Vec<Lin>,
    ctrl_uni: bool,
    ctrl_inv: bool,
    record: bool,
    /// Per enclosing loop (innermost last): the locals stored in it with a
    /// value that is not workgroup-uniform (or under such control), and not
    /// dispatch-invariant.
    loop_stores: Vec<(BTreeSet<LocalKey>, BTreeSet<LocalKey>)>,
    /// Per enclosing loop or switch: whether a thread may leave it (a break
    /// or a return under non-uniform control) while others go on.
    loop_exits: Vec<bool>,
    /// A continue under non-uniform control in this pass of the innermost
    /// loop: the rest of its body runs in some threads only (as after a
    /// return, `ret_taint`); its continuing block and its exit in all.
    skip_taint: bool,
    /// Whether each function called is pure (no global, no pointer
    /// argument, no barrier, atomic or subgroup statement, its calls pure),
    /// and the opaque value of each pure call by its arguments.
    pure: HashMap<Handle<Function>, bool>,
    call_ids: Vec<(Handle<Function>, Vec<Abs>, u64)>,
    /// The opaque value of each pure operation by its operator and its
    /// operands' values (`An::canonical`).
    value_ids: Vec<(String, Vec<Abs>, u64)>,
    ret_taint: bool,
    sites: Vec<Site>,
    site_ix: HashMap<String, usize>,
    wg_sites: Vec<Site>,
    wg_ix: HashMap<String, usize>,
    wg_open: BTreeSet<usize>,
    wg_pairs: BTreeSet<(usize, usize)>,
    /// Per workgroup pair: the loop counters equal in both invocations (the
    /// pair met in one pass of a uniform loop every pass of which crosses a
    /// workgroup barrier: both invocations are in that pass).
    wg_equal: BTreeMap<(usize, usize), BTreeSet<u32>>,
    /// The enclosing loops (innermost last): their uniform counters, whether
    /// every pass crosses a workgroup barrier, and, in the loop-carried walk
    /// of the body before its first barrier, the accesses still open from the
    /// pass before (a pair with one of them is two passes: counters one step
    /// apart).
    loops: Vec<(Vec<u32>, bool, BTreeSet<usize>)>,
    errors: Vec<Unsafe>,
    frames: Vec<Frame<'m>>,
    next_frame: usize,
    next_id: u64,
    steps: usize,
}

/// The cap on statements walked (a kernel the layer would take too long over is not proved).
const STEP_CAP: usize = 4_000_000;

impl<'m> An<'m> {
    fn other(&mut self, inv: bool, uni: bool, wrapped: bool) -> Abs {
        self.next_id += 1;
        Abs::Other(Info { inv, uni, wrapped }, self.next_id)
    }

    fn sym(&mut self, name: &str, lo: i128, hi: i128) -> u32 {
        if let Some(&i) = self.sym_ix.get(name) {
            return i;
        }
        let i = self.syms.len() as u32;
        self.syms.push(SymInfo { name: name.into(), lo, hi });
        self.sym_ix.insert(name.into(), i);
        i
    }

    fn param(&mut self, name: &str) -> Abs {
        let (lo, hi) = self.contract.param(name);
        let s = self.sym(name, lo, hi);
        Abs::Lin(Lin::konst(Poly::sym(s)))
    }

    fn digit(&mut self, key: &str, hi: Option<Poly>, uni: bool) -> Var {
        if let Some(&i) = self.digit_ix.get(key) {
            self.digits[i as usize].hi = hi;
            self.digits[i as usize].uni = uni;
            return Var::Dig(i);
        }
        let i = self.digits.len() as u32;
        self.digits.push(DigitInfo { name: key.into(), hi, uni, def: None });
        self.digit_ix.insert(key.into(), i);
        Var::Dig(i)
    }

    // ------------------------------------------------------- value queries

    fn lin_uni(&self, l: &Lin) -> bool {
        l.t.keys().all(|v| match v {
            Var::Wid(_) => true,
            Var::Lid(_) => false,
            Var::Dig(d) => self.digits[*d as usize].uni,
        })
    }

    fn cond_info(&self, c: &Cond) -> (bool, bool) {
        match c {
            Cond::Lt(a, b) | Cond::Le(a, b) | Cond::Eq(a, b) | Cond::Ne(a, b) => (a.invariant() && b.invariant(), self.lin_uni(a) && self.lin_uni(b)),
            Cond::And(a, b) | Cond::Or(a, b) => {
                let (x, y) = (self.cond_info(a), self.cond_info(b));
                (x.0 && y.0, x.1 && y.1)
            }
            Cond::Not(a) => self.cond_info(a),
            Cond::Const(_) => (true, true),
            Cond::Opaque(i) => (i.inv, i.uni),
        }
    }

    /// (invariant, uniform) of a value.
    fn info(&self, a: &Abs) -> (bool, bool) {
        match a {
            Abs::Lin(l) => (l.invariant(), self.lin_uni(l)),
            Abs::Cond(c) => self.cond_info(c),
            Abs::Vec(v) => v.iter().fold((true, true), |(i, u), x| {
                let (a, b) = self.info(x);
                (i && a, u && b)
            }),
            Abs::Ptr(p) => p.path.iter().fold((true, true), |(i, u), x| {
                let (a, b) = self.info(x);
                (i && a, u && b)
            }),
            Abs::Other(i, _) => (i.inv, i.uni),
        }
    }

    fn wrapped(a: &Abs) -> bool {
        match a {
            Abs::Other(i, _) => i.wrapped,
            Abs::Vec(v) => v.iter().any(Self::wrapped),
            _ => false,
        }
    }

    // ----------------------------------------------------------- bounds

    /// The concrete range of a variable, narrowed by single-variable facts.
    fn static_range(&self, v: Var) -> (i128, i128) {
        match v {
            Var::Wid(c) => (0, self.contract.workgroups[c as usize].1 - 1),
            Var::Lid(c) => (0, i128::from(self.wg[c as usize]) - 1),
            Var::Dig(d) => (0, self.digits[d as usize].hi.as_ref().map_or(U32_MAX, |p| p.range(&self.syms).1.min(U32_MAX))),
        }
    }

    /// The concrete range of a variable, narrowed by the facts it has a
    /// constant coefficient in (the fact's other terms at their static ranges:
    /// `k < min(e, 32)` bounds k by 31).
    fn var_range(&self, v: Var, facts: &[Lin]) -> (i128, i128) {
        let (mut lo, mut hi) = self.static_range(v);
        for f in facts {
            let Some(a) = f.t.get(&v).and_then(Poly::as_const) else { continue };
            // a v + rest + k <= 0 for the actual rest + k, which is at least
            // its least: a > 0 gives a v <= -(rest + k) <= -min(rest + k);
            // a < 0 gives |a| v >= rest + k >= min(rest + k).
            let mut rest_lo = f.k.range(&self.syms).0;
            for (w, c) in &f.t {
                if *w == v {
                    continue;
                }
                let (x, _) = Self::iv_mul(c.range(&self.syms), self.static_range(*w));
                rest_lo = sat(rest_lo + x);
            }
            if a > 0 {
                hi = hi.min((-rest_lo).div_euclid(a));
            } else if a < 0 {
                let b = -a;
                lo = lo.max((rest_lo + b - 1).div_euclid(b));
            }
        }
        (lo, hi)
    }

    /// The concrete range of a form: interval arithmetic over the variables'
    /// ranges, narrowed by any fact over the same variable part.
    fn range(&self, f: &Lin, facts: &[Lin]) -> (i128, i128) {
        let (mut lo, mut hi) = f.k.range(&self.syms);
        for (v, c) in &f.t {
            let (cl, ch) = c.range(&self.syms);
            let (vl, vh) = self.var_range(*v, facts);
            let p = [sat(cl.saturating_mul(vl)), sat(cl.saturating_mul(vh)), sat(ch.saturating_mul(vl)), sat(ch.saturating_mul(vh))];
            lo = sat(lo + p.iter().min().expect("four"));
            hi = sat(hi + p.iter().max().expect("four"));
        }
        for g in facts {
            if g.t.is_empty() {
                continue;
            }
            // g.t + g.k <= 0 with g.t == f.t: f <= f.k - g.k.
            if g.t == f.t {
                hi = hi.min(f.k.sub(&g.k).range(&self.syms).1);
            }
            // g.t == -f.t: -f.t <= -g.k, f >= f.k + g.k.
            if g.t.len() == f.t.len() && g.t.iter().all(|(v, c)| f.t.get(v).is_some_and(|x| *x == c.scale(-1))) {
                lo = lo.max(f.k.add(&g.k).range(&self.syms).0);
            }
        }
        // Each fact's terms found scaled in the form (`row * d` under
        // `row < r_rows`): that part bounded by the fact times its factor,
        // the rest by the variables' ranges.
        let mut rest = f.t.clone();
        let (mut glo, mut ghi) = f.k.range(&self.syms);
        let naive = |an: &Self, t: &BTreeMap<Var, Poly>| -> (i128, i128) {
            t.iter().fold((0, 0), |(lo, hi), (v, c)| {
                let (a, b) = Self::iv_mul(c.range(&an.syms), an.var_range(*v, facts));
                (sat(lo + a), sat(hi + b))
            })
        };
        for g in facts {
            let Some((v0, c0)) = g.t.iter().next() else { continue };
            let Some(c0) = c0.as_const().filter(|c| *c != 0) else { continue };
            // g.t + g.k <= 0: with c0 > 0 the part P = g.t is at most -g.k;
            // with c0 < 0 the part P = -g.t is at least g.k.
            let sign = if c0 > 0 { 1 } else { -1 };
            let Some(p) = rest.get(v0).and_then(|cf| cf.div_mono(&[], c0 * sign)) else { continue };
            if !g.t.iter().all(|(v, c)| rest.get(v).is_some_and(|x| *x == c.mul(&p).scale(sign))) || !p.nonneg(&self.syms) {
                continue;
            }
            let part_t: BTreeMap<Var, Poly> = g.t.iter().map(|(v, c)| (*v, c.scale(sign))).collect();
            let (nlo, nhi) = naive(self, &part_t);
            let part = if sign > 0 { (nlo, nhi.min(g.k.scale(-1).range(&self.syms).1)) } else { (nlo.max(g.k.range(&self.syms).0), nhi) };
            let (a, b) = Self::iv_mul(p.range(&self.syms), part);
            glo = sat(glo + a);
            ghi = sat(ghi + b);
            for v in g.t.keys() {
                rest.remove(v);
            }
        }
        for (v, c) in &rest {
            let (a, b) = Self::iv_mul(c.range(&self.syms), self.var_range(*v, facts));
            glo = sat(glo + a);
            ghi = sat(ghi + b);
        }
        (lo.max(glo), hi.min(ghi))
    }

    /// Symbolic upper bounds of a form whose coefficients are non-negative:
    /// a fact over part of its terms plus each other term at its variable's
    /// bound (the least of its static bound and its single-variable facts).
    fn upper_polys(&self, f: &Lin, facts: &[Lin]) -> Vec<Poly> {
        let var_hi = |v: Var| -> Poly {
            let mut cands: Vec<Poly> = Vec::new();
            match v {
                Var::Wid(c) => {
                    let name = format!("nwg.{}", ["x", "y", "z"][c as usize]);
                    match self.sym_ix.get(&name) {
                        Some(&s) => cands.push(Poly::sym(s).add(&Poly::konst(-1))),
                        None => cands.push(Poly::konst(self.contract.workgroups[c as usize].1 - 1)),
                    }
                }
                Var::Lid(c) => cands.push(Poly::konst(i128::from(self.wg[c as usize]) - 1)),
                Var::Dig(d) => cands.push(self.digits[d as usize].hi.clone().unwrap_or_else(|| Poly::konst(U32_MAX))),
            }
            for g in facts {
                if g.t.len() == 1 {
                    if let Some((gv, c)) = g.t.iter().next() {
                        if *gv == v && c.as_const() == Some(1) {
                            cands.push(g.k.scale(-1));
                        } else if *gv == v {
                            if let (Some(a), Some(k)) = (c.as_const(), g.k.as_const()) {
                                if a > 0 {
                                    cands.push(Poly::konst((-k).div_euclid(a)));
                                }
                            }
                        }
                    }
                }
            }
            cands.into_iter().min_by_key(|p| p.range(&self.syms).1).expect("one bound")
        };
        let rest_bound = |terms: &BTreeMap<Var, Poly>| -> Poly { terms.iter().fold(Poly::default(), |acc, (v, c)| acc.add(&c.mul(&var_hi(*v)))) };
        let mut out = vec![rest_bound(&f.t).add(&f.k)];
        for g in facts {
            if g.t.is_empty() || !g.t.iter().all(|(v, c)| f.t.get(v) == Some(c)) {
                continue;
            }
            let rest: BTreeMap<Var, Poly> = f.t.iter().filter(|(v, _)| !g.t.contains_key(v)).map(|(v, c)| (*v, c.clone())).collect();
            out.push(g.k.scale(-1).add(&rest_bound(&rest)).add(&f.k));
        }
        out
    }

    // ------------------------------------------------------- arithmetic

    fn lin_or_wrap(&mut self, f: Lin, uni: bool) -> Abs {
        self.lin_within(f, uni, None)
    }

    /// The form when its value is proved within u32, by its own range or by
    /// `operands`, the interval its operands' ranges give (either is sound;
    /// the operands' carries the guards on a factor, `row * d` under
    /// `row < r_rows`).
    fn lin_within(&mut self, f: Lin, uni: bool, operands: Option<(i128, i128)>) -> Abs {
        let (lo, hi) = self.range(&f, &self.facts.clone());
        let (lo, hi) = match operands {
            Some((a, b)) => (lo.max(a), hi.min(b)),
            None => (lo, hi),
        };
        if lo < 0 || hi > U32_MAX {
            return self.other(f.invariant(), uni, true);
        }
        Abs::Lin(f)
    }

    fn iv_mul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
        let p = [sat(a.0.saturating_mul(b.0)), sat(a.0.saturating_mul(b.1)), sat(a.1.saturating_mul(b.0)), sat(a.1.saturating_mul(b.1))];
        (*p.iter().min().expect("four"), *p.iter().max().expect("four"))
    }

    /// An invariant symbol for a non-affine function of invariants.
    fn opaque_inv(&mut self, key: String, lo: i128, hi: i128) -> Abs {
        let s = self.sym(&key, lo.max(0), hi.min(U32_MAX));
        Abs::Lin(Lin::konst(Poly::sym(s)))
    }

    fn binary(&mut self, op: BinaryOperator, a: Abs, b: Abs) -> Abs {
        use BinaryOperator as B;
        if let (Abs::Vec(x), Abs::Vec(y)) = (&a, &b) {
            if x.len() == y.len() {
                let v = x.clone().into_iter().zip(y.clone()).map(|(p, q)| self.binary(op, p, q)).collect();
                return Abs::Vec(v);
            }
        }
        if let Abs::Vec(x) = &a {
            if !matches!(b, Abs::Vec(_)) {
                let v = x.clone().into_iter().map(|p| self.binary(op, p, b.clone())).collect();
                return Abs::Vec(v);
            }
        }
        if let Abs::Vec(y) = &b {
            let v = y.clone().into_iter().map(|q| self.binary(op, a.clone(), q)).collect();
            return Abs::Vec(v);
        }
        let (ia, ua) = self.info(&a);
        let (ib, ub) = self.info(&b);
        let (inv, uni) = (ia && ib, ua && ub);
        let wrapped = Self::wrapped(&a) || Self::wrapped(&b);
        match (op, &a, &b) {
            (B::LogicalAnd, Abs::Cond(x), Abs::Cond(y)) => return Abs::Cond(Cond::And(Box::new(x.clone()), Box::new(y.clone()))),
            (B::LogicalOr, Abs::Cond(x), Abs::Cond(y)) => return Abs::Cond(Cond::Or(Box::new(x.clone()), Box::new(y.clone()))),
            // `&` and `|` of booleans are their logical forms (WGSL's non-short-circuit).
            (B::And, Abs::Cond(x), Abs::Cond(y)) => return Abs::Cond(Cond::And(Box::new(x.clone()), Box::new(y.clone()))),
            (B::InclusiveOr, Abs::Cond(x), Abs::Cond(y)) => return Abs::Cond(Cond::Or(Box::new(x.clone()), Box::new(y.clone()))),
            _ => {}
        }
        // A mask of any value by an invariant: a digit in [0, m], whatever the
        // value (a register, a load); the value's identity keys it.
        if op == B::And {
            for (v, m) in [(&a, &b), (&b, &a)] {
                if let (Abs::Other(_, id), Abs::Lin(m)) = (v, m) {
                    if let Some(c) = m.k.as_const().filter(|c| m.invariant() && *c >= 0) {
                        let d = self.digit(&format!("and(#{id},{c})"), Some(Poly::konst(c)), uni);
                        return Abs::Lin(Lin::var(d, Poly::konst(1)));
                    }
                }
            }
        }
        let (Abs::Lin(x), Abs::Lin(y)) = (&a, &b) else {
            if matches!(op, B::Less | B::LessEqual | B::Greater | B::GreaterEqual | B::Equal | B::NotEqual | B::LogicalAnd | B::LogicalOr) {
                return Abs::Cond(Cond::Opaque(Info { inv, uni, wrapped: false }));
            }
            if let (Abs::Cond(_), Abs::Cond(_)) = (&a, &b) {
                return Abs::Cond(Cond::Opaque(Info { inv, uni, wrapped: false }));
            }
            return self.other(inv, uni, wrapped);
        };
        let (x, y) = (x.clone(), y.clone());
        match op {
            B::Add | B::Subtract | B::Multiply => {
                let facts = self.facts.clone();
                let (ra, rb) = (self.range(&x, &facts), self.range(&y, &facts));
                match op {
                    B::Add => self.lin_within(x.add(&y), uni, Some((sat(ra.0 + rb.0), sat(ra.1 + rb.1)))),
                    B::Subtract => self.lin_within(x.sub(&y), uni, Some((sat(ra.0 - rb.1), sat(ra.1 - rb.0)))),
                    _ if y.invariant() => self.lin_within(x.scale(&y.k), uni, Some(Self::iv_mul(ra, rb))),
                    _ if x.invariant() => self.lin_within(y.scale(&x.k), uni, Some(Self::iv_mul(ra, rb))),
                    _ => self.other(inv, uni, false),
                }
            }
            B::ShiftLeft => match y.k.as_const().filter(|_| y.invariant()) {
                Some(s) if (0..32).contains(&s) => self.lin_or_wrap(x.scale(&Poly::konst(1 << s)), uni),
                _ => self.other(inv, uni, true),
            },
            B::Less => Abs::Cond(Cond::Lt(x, y)),
            B::LessEqual => Abs::Cond(Cond::Le(x, y)),
            B::Greater => Abs::Cond(Cond::Lt(y, x)),
            B::GreaterEqual => Abs::Cond(Cond::Le(y, x)),
            B::Equal => Abs::Cond(Cond::Eq(x, y)),
            B::NotEqual => Abs::Cond(Cond::Ne(x, y)),
            // A quotient or remainder by a constant c that the form splits
            // exactly: x = c * H + L with 0 <= L < c gives x / c = H and
            // x % c = L (a shift by k and a mask by 2^k - 1 alike). k02's
            // `idx >> 6u`, `idx & 63u` of idx = tid + 64 j are j and tid.
            B::Divide | B::Modulo | B::ShiftRight | B::And
                if !x.invariant()
                    && y.invariant()
                    && y.k.as_const().is_some_and(|c| match op {
                        B::ShiftRight => (1..32).contains(&c),
                        B::And => c >= 1 && (c + 1) & c == 0,
                        _ => c >= 2,
                    }) =>
            {
                let k = y.k.as_const().unwrap_or(0);
                let c = match op {
                    B::ShiftRight => 1i128 << k,
                    B::And => k + 1,
                    _ => k,
                };
                let facts = self.facts.clone();
                // (A mask or a remainder that cannot change its operand, a
                // negative coefficient among its terms: `(255u - lid.x) & 1023u`.)
                let (xl, xh) = self.range(&x, &facts);
                if matches!(op, B::And | B::Modulo) && xl >= 0 && xh < c {
                    return Abs::Lin(x);
                }
                if x.t.values().all(|p| p.nonneg(&self.syms)) {
                    if let Some((h, l)) = self.split(&x, &[], c, &facts) {
                        return Abs::Lin(if matches!(op, B::Divide | B::ShiftRight) { h } else { l });
                    }
                }
                // Else a digit of its own, with its definition: the
                // quotient in [0, max x / c], the remainder in [0, c - 1].
                // (Both made, the quotient and the remainder: the proof
                // relates them, x = c * q + r, wherever either is read.)
                let quotient = matches!(op, B::Divide | B::ShiftRight);
                let (_, xh) = self.range(&x, &facts);
                let q = self.digit(&format!("div({x:?},{c})"), Some(Poly::konst(xh.max(0) / c)), uni);
                let r = self.digit(&format!("mod({x:?},{c})"), Some(Poly::konst(c - 1)), uni);
                for (d, is_q) in [(q, true), (r, false)] {
                    if let Var::Dig(i) = d {
                        self.digits[i as usize].def = Some((x.clone(), c, is_q));
                    }
                }
                Abs::Lin(Lin::var(if quotient { q } else { r }, Poly::konst(1)))
            }
            B::And | B::Modulo => {
                let facts = self.facts.clone();
                // A mask or a remainder that cannot change its operand.
                if let Some(m) = y.k.as_const().filter(|_| y.invariant()) {
                    let (xl, xh) = self.range(&x, &facts);
                    let keep = if op == B::And { m >= 0 && (m + 1) & m == 0 && xl >= 0 && xh <= m } else { m > 0 && xl >= 0 && xh < m };
                    if keep {
                        return Abs::Lin(x);
                    }
                    if op == B::And && m >= 0 {
                        // A mask by an invariant: a digit in [0, m].
                        let key = format!("and({x:?},{m})");
                        let d = self.digit(&key, Some(Poly::konst(m)), uni);
                        return Abs::Lin(Lin::var(d, Poly::konst(1)));
                    }
                }
                if inv {
                    let (_, ah) = self.range(&x, &facts);
                    let (_, bh) = self.range(&y, &facts);
                    let hi = if op == B::And { ah.min(bh) } else { ah.min(bh - 1).max(0) };
                    return self.opaque_inv(format!("{op:?}({x:?},{y:?})"), 0, hi);
                }
                self.other(inv, uni, false)
            }
            B::Divide | B::ShiftRight | B::ExclusiveOr | B::InclusiveOr => {
                if inv {
                    let facts = self.facts.clone();
                    let (al, ah) = self.range(&x, &facts);
                    let (bl, bh) = self.range(&y, &facts);
                    let (lo, hi) = match op {
                        B::Divide => (al / bh.max(1), ah / bl.max(1)),
                        B::ShiftRight => (al >> bh.clamp(0, 31), ah >> bl.clamp(0, 31)),
                        _ => (0, ((ah.max(bh).max(0) + 1) as u128).next_power_of_two() as i128 - 1),
                    };
                    return self.opaque_inv(format!("{op:?}({x:?},{y:?})"), lo, hi);
                }
                self.other(inv, uni, false)
            }
            _ => self.other(inv, uni, wrapped),
        }
    }

    fn math(&mut self, fun: MathFunction, args: Vec<Abs>) -> Abs {
        let (mut inv, mut uni) = (true, true);
        for a in &args {
            let (i, u) = self.info(a);
            inv &= i;
            uni &= u;
        }
        if let (MathFunction::Min | MathFunction::Max, [Abs::Lin(a), Abs::Lin(b)]) = (fun, args.as_slice()) {
            let facts = self.facts.clone();
            let (al, ah) = self.range(a, &facts);
            let (bl, bh) = self.range(b, &facts);
            let min = fun == MathFunction::Min;
            if (min && ah <= bl) || (!min && al >= bh) {
                return Abs::Lin(a.clone());
            }
            if (min && bh <= al) || (!min && bl >= ah) {
                return Abs::Lin(b.clone());
            }
            if inv {
                let (lo, hi) = if min { (al.min(bl), ah.min(bh)) } else { (al.max(bl), ah.max(bh)) };
                return self.opaque_inv(format!("{fun:?}({a:?},{b:?})"), lo, hi);
            }
            if min {
                // The clamp of a value by an invariant: a digit in [0, c].
                let (x, c) = if b.invariant() { (a, b) } else if a.invariant() { (b, a) } else { return self.other(inv, uni, false) };
                let key = format!("min({x:?},{c:?})");
                let d = self.digit(&key, Some(c.k.clone()), uni);
                return Abs::Lin(Lin::var(d, Poly::konst(1)));
            }
        }
        // The clamp of any value by an invariant: a digit in [0, c] (a u32
        // clamped by 0 is 0: a grouped block of one group writes min(g, 0u)).
        if let (MathFunction::Min, [x, y]) = (fun, args.as_slice()) {
            for (v, c) in [(x, y), (y, x)] {
                if let (Abs::Other(_, id), Abs::Lin(c)) = (v, c) {
                    if c.invariant() && c.k.is_zero() {
                        return Abs::Lin(Lin::default());
                    }
                    if c.invariant() {
                        let d = self.digit(&format!("min(#{id},{:?})", c.k), Some(c.k.clone()), uni);
                        return Abs::Lin(Lin::var(d, Poly::konst(1)));
                    }
                }
            }
        }
        if let (MathFunction::Min | MathFunction::Max, [Abs::Vec(a), Abs::Vec(b)]) = (fun, args.as_slice()) {
            if a.len() == b.len() {
                let v = a.clone().into_iter().zip(b.clone()).map(|(x, y)| self.math(fun, vec![x, y])).collect();
                return Abs::Vec(v);
            }
        }
        self.other(inv, uni, false)
    }

    // ------------------------------------------------------- facts

    /// Whether the facts on the path prove `c` is `truth`: the facts with
    /// what `c` is otherwise have no solution.
    fn implied(&self, c: &Cond, truth: bool) -> bool {
        if let Cond::Const(b) = c {
            return *b == truth;
        }
        let neg = Self::facts_of(c, !truth);
        if neg.is_empty() {
            return false;
        }
        let mut sys: Vec<Ineq> = self.facts.iter().chain(neg.iter()).filter_map(|f| Self::ineq(f, 0)).collect();
        let vars: BTreeSet<(Var, u8)> = sys.iter().flat_map(|i| i.a.keys().copied()).collect();
        for (v, copy) in vars {
            sys.push(Ineq { a: BTreeMap::from([((v, copy), -1)]), k: Poly::default() });
            if let Some(hi) = self.var_hi_poly(v) {
                sys.push(Ineq { a: BTreeMap::from([((v, copy), 1)]), k: hi.scale(-1) });
            }
        }
        self.infeasible(sys)
    }

    /// The linear facts a condition proves when it is `truth`: `f <= 0` each.
    fn facts_of(c: &Cond, truth: bool) -> Vec<Lin> {
        let one = Lin::konst(Poly::konst(1));
        match (c, truth) {
            (Cond::Lt(a, b), true) => vec![a.sub(b).add(&one)],
            (Cond::Lt(a, b), false) => vec![b.sub(a)],
            (Cond::Le(a, b), true) => vec![a.sub(b)],
            (Cond::Le(a, b), false) => vec![b.sub(a).add(&one)],
            (Cond::Eq(a, b), true) | (Cond::Ne(a, b), false) => vec![a.sub(b), b.sub(a)],
            (Cond::And(a, b), true) | (Cond::Or(a, b), false) => [Self::facts_of(a, truth), Self::facts_of(b, truth)].concat(),
            (Cond::Not(a), t) => Self::facts_of(a, !t),
            _ => Vec::new(),
        }
    }

    // ------------------------------------------------------- expressions

    fn zero_of(&mut self, ty: Handle<naga::Type>) -> Abs {
        match &self.module.types[ty].inner {
            TypeInner::Scalar(s) if s.kind == ScalarKind::Uint => Abs::Lin(Lin::default()),
            TypeInner::Scalar(s) if s.kind == ScalarKind::Bool => Abs::Cond(Cond::Const(false)),
            TypeInner::Vector { size, scalar } => {
                let e = if scalar.kind == ScalarKind::Uint { Abs::Lin(Lin::default()) } else if scalar.kind == ScalarKind::Bool { Abs::Cond(Cond::Const(false)) } else { self.other(true, true, false) };
                Abs::Vec(vec![e; *size as usize])
            }
            _ => self.other(true, true, false),
        }
    }

    fn literal(&mut self, l: &naga::Literal) -> Abs {
        match l {
            naga::Literal::U32(v) => Abs::Lin(Lin::konst(Poly::konst(i128::from(*v)))),
            naga::Literal::Bool(b) => Abs::Cond(Cond::Const(*b)),
            _ => self.other(true, true, false),
        }
    }

    fn global_expr(&mut self, h: Handle<Expression>) -> Abs {
        match &self.module.global_expressions[h] {
            Expression::Literal(l) => self.literal(&l.clone()),
            Expression::ZeroValue(t) => self.zero_of(*t),
            Expression::Constant(c) => self.global_expr(self.module.constants[*c].init),
            Expression::Compose { components, .. } => {
                let parts: Vec<Abs> = components.clone().into_iter().map(|c| self.global_expr(c)).collect();
                Abs::Vec(parts)
            }
            Expression::Splat { size, value } => {
                let v = self.global_expr(*value);
                Abs::Vec(vec![v; *size as usize])
            }
            _ => self.other(true, true, false),
        }
    }

    /// The locals a condition bounds where it holds: `x < c` (or `c > x`),
    /// x a local whose value has no affine form and c an invariant constant,
    /// through `&&`: each such local as a digit in [0, c - 1] (2026-10-17:
    /// v24's searches store `cands[q * 64u + rank]` under `rank < 64u`).
    fn bounded_by(&mut self, fr: usize, cond: Handle<Expression>) -> Vec<((usize, Handle<naga::LocalVariable>), Abs)> {
        let f = self.frames[fr].func;
        match &f.expressions[cond] {
            Expression::Binary { op: BinaryOperator::LogicalAnd, left, right } => {
                let (l, r) = (*left, *right);
                let mut out = self.bounded_by(fr, l);
                out.extend(self.bounded_by(fr, r));
                out
            }
            Expression::Binary { op: op @ (BinaryOperator::Less | BinaryOperator::Greater), left, right } => {
                let (x, c) = if *op == BinaryOperator::Less { (*left, *right) } else { (*right, *left) };
                let Expression::Load { pointer } = &f.expressions[x] else { return Vec::new() };
                let Expression::LocalVariable(l) = &f.expressions[*pointer] else { return Vec::new() };
                let key = (self.frames[fr].id, *l);
                let Some(Abs::Other(info, id)) = self.env.get(&key).cloned() else { return Vec::new() };
                let k = match self.eval(fr, c) {
                    Abs::Lin(cl) if cl.invariant() => cl.k.as_const(),
                    _ => None,
                };
                match k {
                    Some(k) if k >= 1 => {
                        let d = self.digit(&format!("lt(#{id},{k})"), Some(Poly::konst(k - 1)), info.uni);
                        vec![(key, Abs::Lin(Lin::var(d, Poly::konst(1))))]
                    }
                    _ => Vec::new(),
                }
            }
            _ => Vec::new(),
        }
    }

    fn eval(&mut self, fr: usize, h: Handle<Expression>) -> Abs {
        if let Some(v) = self.frames[fr].vals.get(&h) {
            return v.clone();
        }
        let before = self.next_id;
        let v = self.compute(fr, h);
        let v = self.canonical(fr, h, v, before);
        self.frames[fr].vals.insert(h, v.clone());
        v
    }

    /// An operation's operator and operands when its value is a function of
    /// its operands' values alone (no memory, no call, no invocation state).
    fn pure_parts(e: &Expression) -> Option<(String, Vec<Handle<Expression>>)> {
        Some(match e {
            Expression::Literal(l) => (format!("{l:?}"), vec![]),
            Expression::ZeroValue(t) => (format!("zero {t:?}"), vec![]),
            Expression::Binary { op, left, right } => (format!("{op:?}"), vec![*left, *right]),
            Expression::Unary { op, expr } => (format!("{op:?}"), vec![*expr]),
            Expression::Math { fun, arg, arg1, arg2, arg3 } => (format!("{fun:?}"), [Some(*arg), *arg1, *arg2, *arg3].into_iter().flatten().collect()),
            Expression::As { expr, kind, convert } => (format!("as {kind:?} {convert:?}"), vec![*expr]),
            Expression::Select { condition, accept, reject } => ("select".into(), vec![*condition, *accept, *reject]),
            Expression::Relational { fun, argument } => (format!("{fun:?}"), vec![*argument]),
            Expression::Splat { size, value } => (format!("splat {size:?}"), vec![*value]),
            Expression::Swizzle { size, vector, pattern } => (format!("swizzle {size:?} {pattern:?}"), vec![*vector]),
            Expression::Compose { ty, components } => (format!("compose {ty:?}"), components.clone()),
            Expression::AccessIndex { base, index } => (format!("field {index}"), vec![*base]),
            Expression::Access { base, index } => ("index".into(), vec![*base, *index]),
            _ => return None,
        })
    }

    /// A pure operation's new opaque values (ids above `before`) given the
    /// identity of the first equal operation on equal operands: the same
    /// expression written twice is one value (seed 99, 2026-10-14: a cell's
    /// offset through u32(..) written in its guard and in its index, two
    /// unknowns, so the guard bounded neither of the store's).
    fn canonical(&mut self, fr: usize, h: Handle<Expression>, v: Abs, before: u64) -> Abs {
        let fresh = |a: &Abs| matches!(a, Abs::Other(_, id) if *id > before);
        let any = match &v {
            Abs::Vec(parts) => parts.iter().any(fresh),
            other => fresh(other),
        };
        if !any {
            return v;
        }
        let Some((op, operands)) = Self::pure_parts(&self.frames[fr].func.expressions[h]) else { return v };
        let Some(args) = operands.iter().map(|o| self.frames[fr].vals.get(o).cloned()).collect::<Option<Vec<Abs>>>() else { return v };
        let id_of = |an: &mut Self, key: String| -> u64 {
            if let Some((_, _, id)) = an.value_ids.iter().find(|(k, a, _)| *k == key && *a == args) {
                return *id;
            }
            an.next_id += 1;
            an.value_ids.push((key, args.clone(), an.next_id));
            an.next_id
        };
        match v {
            Abs::Other(info, _) => Abs::Other(info, id_of(self, op)),
            Abs::Vec(parts) => Abs::Vec(parts.into_iter().enumerate().map(|(k, p)| if fresh(&p) { if let Abs::Other(info, _) = p { Abs::Other(info, id_of(self, format!("{op}#{k}"))) } else { p } } else { p }).collect()),
            other => other,
        }
    }

    fn element(&mut self, v: Abs, i: &Abs) -> Abs {
        match (v, i) {
            (Abs::Vec(parts), Abs::Lin(l)) if l.invariant() => match l.k.as_const() {
                Some(c) if c >= 0 && (c as usize) < parts.len() => parts[c as usize].clone(),
                _ => {
                    let (inv, uni) = self.info(&Abs::Vec(parts.clone()));
                    self.other(inv, uni, false)
                }
            },
            (v, i) => {
                let (a, b) = self.info(&v);
                let (c, d) = self.info(i);
                self.other(a && c, b && d, false)
            }
        }
    }

    fn compute(&mut self, fr: usize, h: Handle<Expression>) -> Abs {
        let f = self.frames[fr].func;
        let u32c = |v: u32| Abs::Lin(Lin::konst(Poly::konst(i128::from(v))));
        match &f.expressions[h] {
            Expression::Literal(l) => self.literal(l),
            Expression::Constant(c) => self.global_expr(self.module.constants[*c].init),
            Expression::Override(_) => self.other(true, true, false),
            Expression::ZeroValue(t) => self.zero_of(*t),
            Expression::Compose { components, .. } => {
                let mut parts = Vec::new();
                for c in components.clone() {
                    match self.eval(fr, c) {
                        Abs::Vec(v) => parts.extend(v),
                        x => parts.push(x),
                    }
                }
                Abs::Vec(parts)
            }
            Expression::Splat { size, value } => {
                let v = self.eval(fr, *value);
                Abs::Vec(vec![v; *size as usize])
            }
            Expression::Swizzle { size, vector, pattern } => {
                let v = self.eval(fr, *vector);
                let pick: Vec<Abs> = pattern[..*size as usize].iter().map(|c| *c as u32).collect::<Vec<_>>().into_iter().map(|c| self.element(v.clone(), &u32c(c))).collect();
                Abs::Vec(pick)
            }
            Expression::Access { base, index } => {
                let (b, i) = (self.eval(fr, *base), self.eval(fr, *index));
                match b {
                    Abs::Ptr(mut p) => {
                        p.path.push(i);
                        Abs::Ptr(p)
                    }
                    v => self.element(v, &i),
                }
            }
            Expression::AccessIndex { base, index } => {
                let b = self.eval(fr, *base);
                match b {
                    Abs::Ptr(mut p) => {
                        p.path.push(u32c(*index));
                        Abs::Ptr(p)
                    }
                    v => self.element(v, &u32c(*index)),
                }
            }
            Expression::FunctionArgument(i) => self.frames[fr].args.get(*i as usize).cloned().unwrap_or(Abs::Other(Info { inv: false, uni: false, wrapped: false }, 0)),
            Expression::GlobalVariable(g) => Abs::Ptr(Ptr { base: Base::Global(*g), path: Vec::new() }),
            Expression::LocalVariable(l) => Abs::Ptr(Ptr { base: Base::Local(self.frames[fr].id, *l), path: Vec::new() }),
            Expression::Load { pointer } => {
                let p = self.eval(fr, *pointer);
                self.load(fr, h, p)
            }
            Expression::Unary { op, expr } => {
                let v = self.eval(fr, *expr);
                match (op, v) {
                    (naga::UnaryOperator::LogicalNot, Abs::Cond(c)) => Abs::Cond(Cond::Not(Box::new(c))),
                    (naga::UnaryOperator::LogicalNot, Abs::Vec(v)) => {
                        let (i, u) = self.info(&Abs::Vec(v));
                        Abs::Cond(Cond::Opaque(Info { inv: i, uni: u, wrapped: false }))
                    }
                    (_, v) => {
                        let (i, u) = self.info(&v);
                        if let (naga::UnaryOperator::BitwiseNot, Abs::Lin(l), true) = (op, &v, i) {
                            return self.opaque_inv(format!("not({l:?})"), 0, U32_MAX);
                        }
                        self.other(i, u, false)
                    }
                }
            }
            Expression::Binary { op, left, right } => {
                let (a, b) = (self.eval(fr, *left), self.eval(fr, *right));
                self.binary(*op, a, b)
            }
            Expression::Select { condition, accept, reject } => {
                let (c, a, r) = (self.eval(fr, *condition), self.eval(fr, *accept), self.eval(fr, *reject));
                if a == r {
                    return a;
                }
                // A condition the guards on the path decide: its arm (the
                // kingdom's floored difference select(a - b, 0u, b > a) under
                // a >= b is a - b).
                if let Abs::Cond(cond) = &c {
                    if self.implied(cond, true) {
                        return a;
                    }
                    if self.implied(cond, false) {
                        return r;
                    }
                }
                let (i1, u1) = self.info(&c);
                let (i2, u2) = self.info(&a);
                let (i3, u3) = self.info(&r);
                self.other(i1 && i2 && i3, u1 && u2 && u3, false)
            }
            Expression::Math { fun, arg, arg1, arg2, arg3 } => {
                let args: Vec<Abs> = [Some(*arg), *arg1, *arg2, *arg3].into_iter().flatten().map(|a| self.eval(fr, a)).collect();
                self.math(*fun, args)
            }
            Expression::As { expr, kind, .. } => {
                let v = self.eval(fr, *expr);
                match (kind, &v) {
                    (ScalarKind::Uint, Abs::Lin(_)) => v,
                    (ScalarKind::Uint, Abs::Vec(parts)) if parts.iter().all(|p| matches!(p, Abs::Lin(_))) => v,
                    _ => {
                        let (i, u) = self.info(&v);
                        self.other(i, u, false)
                    }
                }
            }
            Expression::ArrayLength(p) => {
                let name = match self.eval(fr, *p) {
                    Abs::Ptr(Ptr { base: Base::Global(g), .. }) => self.module.global_variables[g].name.clone().unwrap_or_default(),
                    _ => "?".into(),
                };
                self.param(&format!("arrayLength({name})"))
            }
            Expression::Relational { argument, .. } => {
                let v = self.eval(fr, *argument);
                let (i, u) = self.info(&v);
                Abs::Cond(Cond::Opaque(Info { inv: i, uni: u, wrapped: false }))
            }
            // Results a statement sets (a call's, an atomic's, a subgroup operation's)
            // are in the memo; one read before its statement is unknown.
            _ => self.other(false, false, false),
        }
    }

    /// Whether a global holds atomics (an `atomic<T>`, an array of them):
    /// its loads and stores are atomic accesses, no data race among them.
    fn atomic_global(&self, g: Handle<naga::GlobalVariable>) -> bool {
        match &self.module.types[self.module.global_variables[g].ty].inner {
            TypeInner::Atomic(_) => true,
            TypeInner::Array { base, .. } => matches!(self.module.types[*base].inner, TypeInner::Atomic(_)),
            _ => false,
        }
    }

    /// The flat element index of a pointer path into a global (scalar units):
    /// `[i]` on an array, `[i][lane]` on an array of vectors; a whole vector
    /// element is its lanes as a digit.
    fn flat_index(&mut self, g: Handle<naga::GlobalVariable>, path: &[Abs]) -> (String, Abs) {
        let gv = &self.module.global_variables[g];
        let name = gv.name.clone().unwrap_or_default();
        let mut ty = gv.ty;
        let mut member = String::new();
        let mut idx: Option<Abs> = None;
        let mut rest = path.iter();
        // A struct member first (a buffer of a struct holding an array).
        if let TypeInner::Struct { members, .. } = &self.module.types[ty].inner {
            if let Some(Abs::Lin(m)) = rest.next() {
                let i = m.k.as_const().unwrap_or(0) as usize;
                if let Some(mm) = members.get(i) {
                    member = format!(".m{i}");
                    ty = mm.ty;
                }
            }
        }
        let alloc = format!("{name}{member}");
        let lanes_of = |module: &Module, t: Handle<naga::Type>| -> u32 {
            match &module.types[t].inner {
                TypeInner::Vector { size, .. } => *size as u32,
                _ => 1,
            }
        };
        match self.module.types[ty].inner.clone() {
            TypeInner::Array { base, .. } => {
                let lanes = lanes_of(self.module, base);
                let i = rest.next().cloned().unwrap_or(Abs::Lin(Lin::default()));
                let lane = rest.next().cloned();
                let flat = match (i, lane, lanes) {
                    (i, None, 1) => i,
                    (Abs::Lin(i), Some(Abs::Lin(l)), n) => self.lin_or_wrap(i.scale(&Poly::konst(i128::from(n))).add(&l), true),
                    (Abs::Lin(i), None, n) => {
                        let d = self.digit(&format!("lane{n}"), Some(Poly::konst(i128::from(n) - 1)), true);
                        self.lin_or_wrap(i.scale(&Poly::konst(i128::from(n))).add(&Lin::var(d, Poly::konst(1))), true)
                    }
                    (i, _, _) => {
                        let (a, b) = self.info(&i);
                        self.other(a, b, Self::wrapped(&i))
                    }
                };
                idx = Some(flat);
            }
            TypeInner::Vector { size, .. } => {
                let n = size as u32;
                idx = Some(match rest.next() {
                    Some(l) => l.clone(),
                    None => {
                        let d = self.digit(&format!("lane{n}"), Some(Poly::konst(i128::from(n) - 1)), true);
                        Abs::Lin(Lin::var(d, Poly::konst(1)))
                    }
                });
            }
            _ => {}
        }
        (alloc, idx.unwrap_or(Abs::Lin(Lin::default())))
    }

    fn binding_use(&self, g: Handle<naga::GlobalVariable>) -> Option<Use> {
        let name = self.module.global_variables[g].name.clone().unwrap_or_default();
        self.iface.binding(&name).map(|b| b.role)
    }

    /// Whether a binding shares its buffer with an output.
    fn aliased_with_output(&self, name: &str) -> bool {
        let (buf, _) = self.contract.buffer_of(name);
        self.iface.outputs().any(|o| o.name != name && self.contract.buffer_of(&o.name).0 == buf)
    }

    fn load(&mut self, fr: usize, h: Handle<Expression>, p: Abs) -> Abs {
        let Abs::Ptr(p) = p else { return self.other(false, false, false) };
        match p.base.clone() {
            Base::Local(id, l) => {
                let v = self.env.get(&(id, l)).cloned();
                let v = match v {
                    Some(v) => v,
                    None => self.other(false, false, false),
                };
                self.project(v, &p.path)
            }
            Base::Global(g) => {
                let gv = &self.module.global_variables[g];
                let name = gv.name.clone().unwrap_or_default();
                match gv.space {
                    AddressSpace::Private => {
                        let v = match self.privates.get(&g).cloned() {
                            Some(v) => v,
                            None => {
                                let t = gv.ty;
                                self.zero_of(t)
                            }
                        };
                        self.project(v, &p.path)
                    }
                    AddressSpace::Uniform => {
                        // A field of the uniform block: its symbol (a u32 field).
                        let ty = gv.ty;
                        if let (TypeInner::Struct { members, .. }, Some(Abs::Lin(m))) = (&self.module.types[ty].inner, p.path.first()) {
                            let i = m.k.as_const().unwrap_or(0) as usize;
                            if let Some(mm) = members.get(i) {
                                let fname = mm.name.clone().unwrap_or_else(|| format!("f{i}"));
                                if matches!(self.module.types[mm.ty].inner, TypeInner::Scalar(s) if s.kind == ScalarKind::Uint) && p.path.len() == 1 {
                                    return self.param(&format!("{name}.{fname}"));
                                }
                            }
                        }
                        // A vector block of u32 (`var<uniform> params: vec4<u32>`):
                        // each lane its symbol, `params.x` .. `params.w`; the
                        // whole vector the vector of them. A u32 block: itself.
                        const LANES: [&str; 4] = ["x", "y", "z", "w"];
                        match (&self.module.types[ty].inner, p.path.as_slice()) {
                            (TypeInner::Vector { scalar, .. }, [Abs::Lin(m)]) if scalar.kind == ScalarKind::Uint => {
                                if let Some(lane) = m.k.as_const().and_then(|i| LANES.get(i as usize)) {
                                    return self.param(&format!("{name}.{lane}"));
                                }
                            }
                            (TypeInner::Vector { size, scalar }, []) if scalar.kind == ScalarKind::Uint => {
                                return Abs::Vec(LANES[..*size as usize].iter().map(|lane| self.param(&format!("{name}.{lane}"))).collect());
                            }
                            (TypeInner::Scalar(s), []) if s.kind == ScalarKind::Uint => return self.param(&name),
                            _ => {}
                        }
                        self.other(true, true, false)
                    }
                    AddressSpace::Storage { .. } => {
                        let role = self.binding_use(g);
                        let (alloc, idx) = self.flat_index(g, &p.path);
                        if role == Some(Use::Input) && !self.aliased_with_output(&name) {
                            let (iinv, iuni) = self.info(&idx);
                            let u32_elem = self.elem_is_u32(g);
                            if let (Abs::Lin(l), true) = (&idx, iinv) {
                                let key = format!("{alloc}[{}]", l.k.show(&self.syms));
                                if u32_elem {
                                    return self.param(&key);
                                }
                            }
                            return self.other(iinv, iuni, false);
                        }
                        let key = format!("{}:{}", self.frames[fr].ctx, h.index());
                        let kind = if self.atomic_global(g) { Kind::Atomic } else { Kind::Read };
                        self.site(key, &alloc, kind, idx, false);
                        self.other(false, false, false)
                    }
                    AddressSpace::WorkGroup => {
                        // (atomicLoad is a Load of an atomic: an atomic access.)
                        let kind = if self.atomic_global(g) { Kind::Atomic } else { Kind::Read };
                        let (alloc, idx) = self.flat_index(g, &p.path);
                        let key = format!("{}:{}", self.frames[fr].ctx, h.index());
                        self.site(key, &alloc, kind, idx, true);
                        self.other(false, false, false)
                    }
                    _ => self.other(false, false, false),
                }
            }
        }
    }

    fn elem_is_u32(&self, g: Handle<naga::GlobalVariable>) -> bool {
        let t = self.module.global_variables[g].ty;
        let base = match &self.module.types[t].inner {
            TypeInner::Array { base, .. } => *base,
            _ => t,
        };
        matches!(self.module.types[base].inner, TypeInner::Scalar(s) if s.kind == ScalarKind::Uint)
    }

    fn project(&mut self, mut v: Abs, path: &[Abs]) -> Abs {
        for i in path {
            v = self.element(v, i);
        }
        v
    }

    fn site(&mut self, key: String, alloc: &str, kind: Kind, idx: Abs, workgroup: bool) {
        if !self.record || self.dead(&self.facts) {
            return;
        }
        let (buf, off) = if workgroup { (alloc.to_string(), 0) } else { self.contract.buffer_of(alloc) };
        let form = match idx {
            Abs::Lin(l) => Ok(l.add(&Lin::konst(Poly::konst(off)))),
            Abs::Other(i, _) if i.wrapped => Err(("store_index_may_wrap", "its arithmetic may wrap at the contract's ranges".to_string())),
            _ => Err(("store_index_not_affine", "not an affine form of the thread's coordinates, the loop counters and clamped digits".to_string())),
        };
        let what = match (&form, kind) {
            (Ok(l), _) => format!("{kind:?} {alloc}[{}]", self.show_lin(l)),
            (Err(_), _) => format!("{kind:?} {alloc}[?]"),
        };
        let s = Site { alloc: buf, kind, form, facts: self.facts.clone(), what };
        if workgroup {
            let id = match self.wg_ix.get(&key) {
                Some(&i) => i,
                None => {
                    self.wg_sites.push(s);
                    self.wg_ix.insert(key, self.wg_sites.len() - 1);
                    self.wg_sites.len() - 1
                }
            };
            let open: Vec<usize> = self.wg_open.iter().copied().collect();
            let pair = |an: &mut Self, key: (usize, usize)| {
                let equal: BTreeSet<u32> = an.loops.iter().filter(|(_, barrier, before)| *barrier && !before.contains(&key.0) && !before.contains(&key.1)).flat_map(|(c, _, _)| c.iter().copied()).collect();
                an.wg_pairs.insert(key);
                // (A pair met more than once keeps the counters equal every time.)
                match an.wg_equal.get_mut(&key) {
                    Some(e) => e.retain(|d| equal.contains(d)),
                    None => {
                        an.wg_equal.insert(key, equal.clone());
                    }
                }
            };
            for o in open {
                let (a, b) = (self.wg_sites[o].kind, self.wg_sites[id].kind);
                let conflict = !(a == b && a != Kind::Write) && self.wg_sites[o].alloc == self.wg_sites[id].alloc;
                if conflict {
                    pair(self, (o.min(id), o.max(id)));
                }
            }
            if self.wg_sites[id].kind == Kind::Write {
                pair(self, (id, id));
            }
            self.wg_open.insert(id);
        } else if !self.site_ix.contains_key(&key) {
            self.sites.push(s);
            self.site_ix.insert(key, self.sites.len() - 1);
        }
    }

    fn show_lin(&self, l: &Lin) -> String {
        let mut parts: Vec<String> = l
            .t
            .iter()
            .map(|(v, c)| {
                let vn = match v {
                    Var::Wid(c) => format!("wid.{}", ["x", "y", "z"][*c as usize]),
                    Var::Lid(c) => format!("lid.{}", ["x", "y", "z"][*c as usize]),
                    Var::Dig(d) => self.digits[*d as usize].name.chars().take(40).collect(),
                };
                format!("({})*{vn}", c.show(&self.syms))
            })
            .collect();
        if !l.k.is_zero() {
            parts.push(l.k.show(&self.syms));
        }
        if parts.is_empty() {
            "0".into()
        } else {
            parts.join(" + ")
        }
    }

    // ------------------------------------------------------- statements

    fn local_key(&self, fr: usize, l: Handle<naga::LocalVariable>) -> LocalKey {
        (self.frames[fr].fkey, l)
    }

    fn frame_of_id(&self, id: usize) -> Option<usize> {
        self.frames.iter().position(|f| f.id == id)
    }

    fn store_local(&mut self, id: usize, l: Handle<naga::LocalVariable>, path: &[Abs], v: Abs) {
        let (inv, uni) = self.info(&v);
        if let Some(fr) = self.frame_of_id(id) {
            let key = self.local_key(fr, l);
            let (nu, ni) = (!uni || !self.ctrl_uni, !inv || !self.ctrl_inv);
            for (su, si) in self.loop_stores.iter_mut() {
                if nu {
                    su.insert(key);
                }
                if ni {
                    si.insert(key);
                }
            }
        }
        let new = if path.is_empty() {
            v
        } else {
            let old = self.env.get(&(id, l)).cloned().unwrap_or(Abs::Other(Info { inv: false, uni: false, wrapped: false }, 0));
            match (old, path) {
                (Abs::Vec(mut parts), [Abs::Lin(i)]) if i.invariant() && i.k.as_const().is_some_and(|c| c >= 0 && (c as usize) < parts.len()) => {
                    let c = i.k.as_const().unwrap_or(0) as usize;
                    parts[c] = v;
                    Abs::Vec(parts)
                }
                (old, _) => {
                    let (a, b) = self.info(&old);
                    self.other(a && inv, b && uni, false)
                }
            }
        };
        self.env.insert((id, l), new);
    }

    fn stmt(&mut self, fr: usize, s: &'m Statement) -> bool {
        self.steps += 1;
        if self.steps > STEP_CAP {
            return false;
        }
        let ctx = format!("{}:{:p}", self.frames[fr].ctx, s as *const Statement);
        match s {
            Statement::Emit(range) => {
                for h in range.clone() {
                    let before = self.next_id;
                    let v = self.compute(fr, h);
                    let v = self.canonical(fr, h, v, before);
                    self.frames[fr].vals.insert(h, v);
                }
                true
            }
            Statement::Block(b) => self.block(fr, b),
            Statement::If { condition, accept, reject } => {
                let c = match self.eval(fr, *condition) {
                    Abs::Cond(c) => c,
                    other => {
                        let (i, u) = self.info(&other);
                        Cond::Opaque(Info { inv: i, uni: u, wrapped: false })
                    }
                };
                let (ci, cu) = self.cond_info(&c);
                let (saved_env, saved_priv) = (self.env.clone(), self.privates.clone());
                let (su, si) = (self.ctrl_uni, self.ctrl_inv);
                let base = self.facts.len();
                self.ctrl_uni = su && cu;
                self.ctrl_inv = si && ci;
                self.facts.extend(Self::facts_of(&c, true));
                // A local of no affine form tested below an invariant
                // (`if (rank < 64u && ..)`): in the branch, a digit below it,
                // so `cands[q * 64u + rank]` is an affine form there.
                for (key, v) in self.bounded_by(fr, *condition) {
                    self.env.insert(key, v);
                }
                let ft_a = self.block(fr, accept);
                let (env_a, priv_a) = (std::mem::replace(&mut self.env, saved_env), std::mem::replace(&mut self.privates, saved_priv));
                self.facts.truncate(base);
                self.ctrl_uni = su && cu;
                self.ctrl_inv = si && ci;
                self.facts.extend(Self::facts_of(&c, false));
                let ft_r = self.block(fr, reject);
                self.facts.truncate(base);
                self.ctrl_uni = su;
                self.ctrl_inv = si;
                match (ft_a, ft_r) {
                    (true, true) => {
                        let (env_r, priv_r) = (std::mem::take(&mut self.env), std::mem::take(&mut self.privates));
                        self.env = self.merge_env(env_a, env_r, ci, cu);
                        self.privates = self.merge_map(priv_a, priv_r, ci, cu);
                    }
                    (true, false) => {
                        self.env = env_a;
                        self.privates = priv_a;
                        self.facts.extend(Self::facts_of(&c, true));
                    }
                    (false, true) => {
                        self.facts.extend(Self::facts_of(&c, false));
                    }
                    (false, false) => {
                        self.taint_after();
                        return false;
                    }
                }
                self.taint_after();
                true
            }
            Statement::Switch { selector, cases } => {
                let sel = self.eval(fr, *selector);
                let (ci, cu) = self.info(&sel);
                let (saved_env, saved_priv) = (self.env.clone(), self.privates.clone());
                let (su, si) = (self.ctrl_uni, self.ctrl_inv);
                let mut outs: Vec<(HashMap<_, _>, HashMap<_, _>)> = Vec::new();
                for case in cases {
                    self.env = saved_env.clone();
                    self.privates = saved_priv.clone();
                    self.ctrl_uni = su && cu;
                    self.ctrl_inv = si && ci;
                    // A switch's break leaves the switch: the case's end.
                    self.loop_exits.push(false);
                    let ft = self.block(fr, &case.body);
                    self.loop_exits.pop();
                    if ft || self.env != saved_env {
                        outs.push((std::mem::take(&mut self.env), std::mem::take(&mut self.privates)));
                    }
                }
                self.ctrl_uni = su;
                self.ctrl_inv = si;
                let mut it = outs.into_iter();
                match it.next() {
                    Some((mut e, mut p)) => {
                        for (e2, p2) in it {
                            e = self.merge_env(e, e2, ci, cu);
                            p = self.merge_map(p, p2, ci, cu);
                        }
                        self.env = e;
                        self.privates = p;
                    }
                    None => {
                        self.env = saved_env;
                        self.privates = saved_priv;
                    }
                }
                self.taint_after();
                true
            }
            Statement::Loop { body, continuing, break_if } => {
                self.walk_loop(fr, s, body, continuing, *break_if);
                true
            }
            Statement::Break => {
                if !self.ctrl_uni {
                    if let Some(e) = self.loop_exits.last_mut() {
                        *e = true;
                    }
                }
                false
            }
            // (A continue in a switch continues the loop around it.)
            Statement::Continue => {
                if !self.ctrl_uni {
                    self.skip_taint = true;
                }
                false
            }
            Statement::Return { value } => {
                if let Some(v) = value {
                    let v = self.eval(fr, *v);
                    let merged = match self.frames[fr].ret.take() {
                        None => v,
                        Some(old) if old == v => old,
                        Some(old) => {
                            let (a, b) = self.info(&old);
                            let (c, d) = self.info(&v);
                            self.other(a && c, b && d, false)
                        }
                    };
                    self.frames[fr].ret = Some(merged);
                }
                if !self.ctrl_uni {
                    self.ret_taint = true;
                    self.loop_exits.iter_mut().for_each(|e| *e = true);
                }
                false
            }
            Statement::Kill => false,
            Statement::Barrier(flags) => {
                if !self.ctrl_uni && self.record {
                    self.errors.push(unsafe_("barrier_in_nonuniform_control", format!("a barrier ({flags:?}) where the invocations of a workgroup may disagree on reaching it")));
                }
                if flags.contains(naga::Barrier::WORK_GROUP) {
                    self.wg_open.clear();
                    self.loops.iter_mut().for_each(|l| l.2.clear());
                }
                true
            }
            Statement::Store { pointer, value } => {
                let p = self.eval(fr, *pointer);
                let v = self.eval(fr, *value);
                self.store(fr, &ctx, p, v, Kind::Write);
                true
            }
            Statement::Atomic { pointer, value, result, .. } => {
                let p = self.eval(fr, *pointer);
                let v = self.eval(fr, *value);
                self.store(fr, &ctx, p, v, Kind::Atomic);
                let r = self.other(false, false, false);
                self.frames[fr].vals.insert(*result, r);
                true
            }
            Statement::WorkGroupUniformLoad { pointer, result } => {
                if !self.ctrl_uni && self.record {
                    self.errors.push(unsafe_("barrier_in_nonuniform_control", "a workgroupUniformLoad (a barrier) in non-uniform control flow".into()));
                }
                self.wg_open.clear();
                let p = self.eval(fr, *pointer);
                let _ = self.load(fr, *result, p);
                self.wg_open.clear();
                self.loops.iter_mut().for_each(|l| l.2.clear());
                let r = self.other(false, true, false);
                self.frames[fr].vals.insert(*result, r);
                true
            }
            Statement::SubgroupBallot { result, .. } | Statement::SubgroupGather { result, .. } | Statement::SubgroupCollectiveOperation { result, .. } => {
                if let Statement::SubgroupGather { argument, .. } | Statement::SubgroupCollectiveOperation { argument, .. } = s {
                    let _ = self.eval(fr, *argument);
                }
                if !self.ctrl_uni && self.record {
                    self.errors.push(unsafe_("subgroup_operation_in_nonuniform_control", "a subgroup operation where the invocations of a workgroup may disagree on reaching it".into()));
                }
                let r = self.other(false, false, false);
                self.frames[fr].vals.insert(*result, r);
                true
            }
            Statement::Call { function, arguments, result } => {
                let args: Vec<Abs> = arguments.iter().map(|a| self.eval(fr, *a)).collect();
                let ret = self.call(*function, args.clone(), &ctx);
                // A pure function's opaque result is one value for one set of
                // arguments: a cell's guard and its store then bound one value
                // (seed 28, 2026-10-14: at_vec3u(..) in a cell's offset, each
                // of its three calls its own unknown, so the guard c < d
                // bounded none of the store's).
                let ret = match ret {
                    Some(Abs::Other(info, _)) if self.pure_fn(*function) => Some(Abs::Other(info, self.call_id(*function, &args))),
                    other => other,
                };
                if let Some(r) = result {
                    let v = match ret {
                        Some(v) => v,
                        None => self.other(false, false, false),
                    };
                    self.frames[fr].vals.insert(*r, v);
                }
                true
            }
            Statement::ImageStore { .. } | Statement::RayQuery { .. } => {
                if self.record {
                    self.errors.push(unsafe_("statement_not_in_kingdom", "an image store or a ray query".into()));
                }
                true
            }
        }
    }

    fn taint_after(&mut self) {
        if self.ret_taint || self.skip_taint {
            self.ctrl_uni = false;
            self.ctrl_inv = false;
        }
    }

    fn merge_val(&mut self, a: Abs, b: Abs, ci: bool, cu: bool) -> Abs {
        if a == b {
            return a;
        }
        match (a, b) {
            (Abs::Vec(x), Abs::Vec(y)) if x.len() == y.len() => Abs::Vec(x.into_iter().zip(y).map(|(p, q)| self.merge_val(p, q, ci, cu)).collect()),
            (a, b) => {
                let (i1, u1) = self.info(&a);
                let (i2, u2) = self.info(&b);
                self.other(i1 && i2 && ci, u1 && u2 && cu, false)
            }
        }
    }

    fn merge_env(&mut self, a: HashMap<(usize, Handle<naga::LocalVariable>), Abs>, mut b: HashMap<(usize, Handle<naga::LocalVariable>), Abs>, ci: bool, cu: bool) -> HashMap<(usize, Handle<naga::LocalVariable>), Abs> {
        let mut out = HashMap::new();
        for (k, va) in a {
            let v = match b.remove(&k) {
                Some(vb) => self.merge_val(va, vb, ci, cu),
                None => self.other(false, false, false),
            };
            out.insert(k, v);
        }
        for (k, _) in b {
            let v = self.other(false, false, false);
            out.insert(k, v);
        }
        out
    }

    fn merge_map(&mut self, a: HashMap<Handle<naga::GlobalVariable>, Abs>, mut b: HashMap<Handle<naga::GlobalVariable>, Abs>, ci: bool, cu: bool) -> HashMap<Handle<naga::GlobalVariable>, Abs> {
        let mut out = HashMap::new();
        for (k, va) in a {
            let v = match b.remove(&k) {
                Some(vb) => self.merge_val(va, vb, ci, cu),
                None => self.other(false, false, false),
            };
            out.insert(k, v);
        }
        for (k, _) in b {
            let v = self.other(false, false, false);
            out.insert(k, v);
        }
        out
    }

    fn store(&mut self, fr: usize, ctx: &str, p: Abs, v: Abs, kind: Kind) {
        let Abs::Ptr(p) = p else { return };
        match p.base.clone() {
            Base::Local(id, l) => self.store_local(id, l, &p.path, v),
            Base::Global(g) => {
                let gv = &self.module.global_variables[g];
                match gv.space {
                    AddressSpace::Private => {
                        let new = if p.path.is_empty() {
                            v
                        } else {
                            let (a, b) = self.info(&v);
                            self.other(a && self.ctrl_inv, b && self.ctrl_uni, false)
                        };
                        self.privates.insert(g, new);
                    }
                    AddressSpace::Storage { .. } => {
                        let kind = if self.atomic_global(g) { Kind::Atomic } else { kind };
                        let (alloc, idx) = self.flat_index(g, &p.path);
                        self.site(ctx.to_string(), &alloc, kind, idx, false);
                    }
                    AddressSpace::WorkGroup => {
                        // (atomicStore is a Store to an atomic: an atomic access.)
                        let kind = if self.atomic_global(g) { Kind::Atomic } else { kind };
                        let (alloc, idx) = self.flat_index(g, &p.path);
                        self.site(ctx.to_string(), &alloc, kind, idx, true);
                    }
                    _ => {}
                }
                let _ = fr;
            }
        }
    }

    /// The locals a block stores (directly, through calls not followed).
    fn stored_locals(f: &Function, b: &Block, out: &mut Vec<(Handle<naga::LocalVariable>, bool)>, in_continuing: bool) {
        for s in b.iter() {
            match s {
                Statement::Store { pointer, .. } => {
                    let mut e = *pointer;
                    loop {
                        match &f.expressions[e] {
                            Expression::LocalVariable(l) => {
                                out.push((*l, in_continuing));
                                break;
                            }
                            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => e = *base,
                            _ => break,
                        }
                    }
                }
                Statement::Call { arguments, .. } => {
                    // A local passed by pointer may be stored by the callee.
                    for a in arguments {
                        if let Expression::LocalVariable(l) = &f.expressions[*a] {
                            out.push((*l, in_continuing));
                        }
                    }
                }
                Statement::Block(x) => Self::stored_locals(f, x, out, in_continuing),
                Statement::If { accept, reject, .. } => {
                    Self::stored_locals(f, accept, out, in_continuing);
                    Self::stored_locals(f, reject, out, in_continuing);
                }
                Statement::Switch { cases, .. } => cases.iter().for_each(|c| Self::stored_locals(f, &c.body, out, in_continuing)),
                Statement::Loop { body, continuing, .. } => {
                    Self::stored_locals(f, body, out, in_continuing);
                    Self::stored_locals(f, continuing, out, in_continuing);
                }
                _ => {}
            }
        }
    }

    /// Whether every pass of a loop with this body crosses a workgroup
    /// barrier: one at the body's top level (or in a plain block there), and
    /// no continue that could skip it (a break leaves the loop; a
    /// non-uniform one is a non-uniform exit).
    fn barrier_every_pass(body: &Block) -> bool {
        fn barrier(b: &Block) -> bool {
            b.iter().any(|s| match s {
                Statement::Barrier(f) => f.contains(naga::Barrier::WORK_GROUP),
                Statement::WorkGroupUniformLoad { .. } => true,
                Statement::Block(x) => barrier(x),
                _ => false,
            })
        }
        fn continues(b: &Block) -> bool {
            b.iter().any(|s| match s {
                Statement::Continue => true,
                Statement::Block(x) => continues(x),
                Statement::If { accept, reject, .. } => continues(accept) || continues(reject),
                Statement::Switch { cases, .. } => cases.iter().any(|c| continues(&c.body)),
                _ => false,
            })
        }
        barrier(body) && !continues(body)
    }

    /// A counter: stored once, in the continuing block, as `l = l + k` (k a
    /// literal of at least 1), and nowhere in the body; its step k.
    fn counter_step(f: &Function, continuing: &Block, l: Handle<naga::LocalVariable>) -> Option<u32> {
        let (mut found, mut step) = (0, 0u32);
        for s in continuing.iter() {
            if let Statement::Store { pointer, value } = s {
                if !matches!(f.expressions[*pointer], Expression::LocalVariable(x) if x == l) {
                    continue;
                }
                found += 1;
                let k = match &f.expressions[*value] {
                    Expression::Binary { op: BinaryOperator::Add, left, right } => {
                        let is_load = |e: Handle<Expression>| matches!(&f.expressions[e], Expression::Load { pointer } if matches!(f.expressions[*pointer], Expression::LocalVariable(x) if x == l));
                        let step_of = |e: Handle<Expression>| match &f.expressions[e] {
                            Expression::Literal(naga::Literal::U32(k)) if *k >= 1 => Some(*k),
                            _ => None,
                        };
                        if is_load(*left) {
                            step_of(*right)
                        } else if is_load(*right) {
                            step_of(*left)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                step = k?;
            }
        }
        (found == 1).then_some(step)
    }

    /// The exit test a loop's body opens with: its leading emits and calls
    /// and the condition of an `if` one of whose branches is exactly `break`
    /// and the other empty, with whether the loop goes on when the condition
    /// holds. None when the body opens otherwise, or when an emitted
    /// expression reads memory other than a local's (evaluating it early
    /// must record nothing; a call must be pure, checked where it runs).
    fn head_exit(f: &Function, body: &'m Block) -> Option<(Vec<&'m Statement>, Handle<Expression>, bool)> {
        let mut emits = Vec::new();
        for st in body.iter() {
            match st {
                Statement::Emit(range) => {
                    for h in range.clone() {
                        if let Expression::Load { pointer } = &f.expressions[h] {
                            if !matches!(f.expressions[*pointer], Expression::LocalVariable(_)) {
                                return None;
                            }
                        }
                    }
                    emits.push(st);
                }
                Statement::Call { .. } => emits.push(st),
                Statement::If { condition, accept, reject } => {
                    let is_break = |b: &Block| b.len() == 1 && matches!(b.iter().next(), Some(Statement::Break));
                    return match (is_break(accept), accept.is_empty(), is_break(reject), reject.is_empty()) {
                        (true, _, _, true) => Some((emits, *condition, false)),
                        (_, true, true, _) => Some((emits, *condition, true)),
                        _ => None,
                    };
                }
                _ => return None,
            }
        }
        None
    }

    fn walk_loop(&mut self, fr: usize, s: &'m Statement, body: &'m Block, continuing: &'m Block, break_if: Option<Handle<Expression>>) {
        let f = self.frames[fr].func;
        let id = self.frames[fr].id;
        let mut stored = Vec::new();
        Self::stored_locals(f, body, &mut stored, false);
        Self::stored_locals(f, continuing, &mut stored, true);
        let modified: BTreeSet<Handle<naga::LocalVariable>> = stored.iter().map(|(l, _)| *l).collect();
        let counters: BTreeMap<Handle<naga::LocalVariable>, u32> = modified.iter().copied().filter(|l| !stored.iter().any(|(x, cont)| x == l && !cont)).filter_map(|l| Self::counter_step(f, continuing, l).map(|k| (l, k))).collect();
        let head = Self::head_exit(f, body);
        let loop_key = format!("{}:{:p}", self.frames[fr].ctx, s as *const Statement);
        let entry_uni = self.ctrl_uni;
        let entry_env = self.env.clone();
        // One pass: the counters as digits, every other local the loop
        // stores as an unknown, uniform (invariant) when it enters so and no
        // store of the loop's (`assume`, from the passes before) makes it
        // otherwise. `head_uni`: every thread starts each pass together (no
        // thread has left the loop); a continue under non-uniform control
        // makes the rest of its pass's body non-uniform (`skip_taint`), not
        // the continuing block, the next pass or the exit.
        type Sets = (BTreeSet<LocalKey>, BTreeSet<LocalKey>);
        let every_pass = Self::barrier_every_pass(body) && break_if.is_none();
        let run = |an: &mut An<'m>, head_uni: bool, assume: &Sets, carried: bool| -> (bool, Sets) {
            an.loop_stores.push((BTreeSet::new(), BTreeSet::new()));
            let mut uniform_counters = Vec::new();
            let mut strided: Vec<(Handle<naga::LocalVariable>, Var, u32, Lin)> = Vec::new();
            for l in &modified {
                let key = an.local_key(fr, *l);
                let (ei, eu) = entry_env.get(&(id, *l)).map_or((false, false), |e| an.info(e));
                let (inv, uni) = (ei && !assume.1.contains(&key), eu && head_uni && !assume.0.contains(&key));
                if let Some(&k) = counters.get(l) {
                    let entry = entry_env.get(&(id, *l)).cloned();
                    let entry_ok = entry.as_ref().is_some_and(|e| an.info(e).1);
                    // (The counter is uniform when it enters uniform and every
                    // thread steps it together, in the continuing block: its
                    // value is its own, not the local's elsewhere.)
                    let d = an.digit(&format!("{loop_key}:ctr{}", l.index()), None, head_uni && entry_ok);
                    if let (Var::Dig(i), true) = (d, head_uni && entry_ok) {
                        uniform_counters.push(i);
                    }
                    // THE STRIDE: a counter that enters as a form e and steps
                    // by k is e + k * d, d the pass, when the loop's exit test
                    // bounds it so that no step wraps (checked below, once
                    // every counter is in place); else d, any u32.
                    let stride = match &entry {
                        Some(Abs::Lin(e)) if head.is_some() => Some(e.add(&Lin::var(d, Poly::konst(i128::from(k))))),
                        _ => None,
                    };
                    match stride {
                        Some(form) => {
                            strided.push((*l, d, k, form.clone()));
                            an.env.insert((id, *l), Abs::Lin(form));
                        }
                        None => {
                            an.env.insert((id, *l), Abs::Lin(Lin::var(d, Poly::konst(1))));
                        }
                    }
                } else {
                    let v = an.other(inv, uni, false);
                    an.env.insert((id, *l), v);
                }
            }
            // No step wraps: under the exit test's facts (the pass goes on),
            // each strided counter plus its step stays within u32. The
            // leading emits run first (they record nothing: `head_exit`);
            // the walk of the body recomputes them.
            if !strided.is_empty() {
                let mut held = false;
                let pure = head.as_ref().is_some_and(|(emits, _, _)| emits.iter().all(|st| match st {
                    Statement::Call { function, .. } => an.pure_fn(*function),
                    _ => true,
                }));
                if let (Some((emits, cond, goes_on)), true) = (&head, pure) {
                    for st in emits {
                        an.stmt(fr, st);
                    }
                    let facts = match an.eval(fr, *cond) {
                        Abs::Cond(c) => Self::facts_of(&c, *goes_on),
                        _ => Vec::new(),
                    };
                    let all = [an.facts.clone(), facts].concat();
                    held = strided.iter().all(|(_, _, k, form)| {
                        let (lo, hi) = an.range(&form.add(&Lin::konst(Poly::konst(i128::from(*k)))), &all);
                        lo >= 0 && hi <= U32_MAX
                    });
                    // The pass is bounded too: every reached pass has
                    // e + k * d <= the counter's bound, so d <= (that - e's
                    // least) / k (else every form of d would look as if it
                    // might wrap).
                    if held {
                        for (l, d, k, form) in &strided {
                            let hi_t = an.range(form, &all).1;
                            let entry = match entry_env.get(&(id, *l)) {
                                Some(Abs::Lin(e)) => an.range(e, &all).0,
                                _ => 0,
                            };
                            if let Var::Dig(i) = d {
                                an.digits[*i as usize].hi = Some(Poly::konst((hi_t - entry).max(0) / i128::from(*k)));
                            }
                        }
                    }
                }
                if !held {
                    for (l, d, _, _) in &strided {
                        an.env.insert((id, *l), Abs::Lin(Lin::var(*d, Poly::konst(1))));
                    }
                }
            }
            let (su, si) = (an.ctrl_uni, an.ctrl_inv);
            let skip = std::mem::replace(&mut an.skip_taint, false);
            an.ctrl_uni = su && head_uni;
            an.loop_exits.push(false);
            let before = if carried { an.wg_open.clone() } else { BTreeSet::new() };
            an.loops.push((uniform_counters, every_pass && head_uni, before));
            let base = an.facts.len();
            an.block(fr, body);
            an.facts.truncate(base);
            an.skip_taint = false;
            // (Walked whether or not the body falls through: a continue
            // reaches the continuing block too.)
            an.ctrl_uni = su && head_uni;
            an.block(fr, continuing);
            // (A `break if` on a value the threads may disagree on: they leave
            // the loop at different passes.)
            if let Some(b) = break_if {
                let c = an.eval(fr, b);
                if !(an.ctrl_uni && an.info(&c).1) {
                    if let Some(e) = an.loop_exits.last_mut() {
                        *e = true;
                    }
                }
            }
            an.facts.truncate(base);
            let exits = an.loop_exits.pop().unwrap_or(false);
            an.loops.pop();
            an.skip_taint = skip;
            an.ctrl_uni = su;
            an.ctrl_inv = si;
            let sets = an.loop_stores.pop().unwrap_or_default();
            // (A store in this loop is one in every loop around it too.)
            if let Some(outer) = an.loop_stores.last_mut() {
                outer.0.extend(sets.0.iter().copied());
                outer.1.extend(sets.1.iter().copied());
            }
            (exits, sets)
        };
        // Whether a thread may leave the loop while others go on (then every
        // statement of the body is in non-uniform control), and which locals
        // the loop makes non-uniform: a fixpoint of passes
        // that record nothing (each pass only adds, to finite sets, so it
        // ends; a pass cut short would leave a local uniform that is not).
        let record = self.record;
        self.record = false;
        let wg_open = self.wg_open.clone();
        let (mut nonuniform_exit, mut assume): (bool, Sets) = (false, (BTreeSet::new(), BTreeSet::new()));
        loop {
            let (exit, sets) = run(self, entry_uni && !nonuniform_exit, &assume, false);
            self.env = entry_env.clone();
            let grown = exit != nonuniform_exit || !sets.0.is_subset(&assume.0) || !sets.1.is_subset(&assume.1);
            nonuniform_exit |= exit;
            assume.0.extend(sets.0);
            assume.1.extend(sets.1);
            if !grown {
                break;
            }
        }
        self.record = record;
        self.wg_open = wg_open;
        let head_uni = entry_uni && !nonuniform_exit;
        let wg_before = self.wg_sites.len();
        run(self, head_uni, &assume, false);
        // Loop-carried workgroup conflicts: the accesses open at the end of a
        // pass meet the next pass's.
        if record && self.wg_sites.len() > wg_before {
            self.env = entry_env.clone();
            run(self, head_uni, &assume, true);
        }
        // After the loop every local it stores is unknown: uniform when it
        // entered so, no store of the loop made it otherwise and every thread
        // left the loop together.
        for l in &modified {
            let key = self.local_key(fr, *l);
            let (ei, eu) = entry_env.get(&(id, *l)).map_or((false, false), |e| self.info(e));
            let (inv, uni) = (ei && !assume.1.contains(&key) && entry_uni, eu && !assume.0.contains(&key) && head_uni);
            let v = self.other(inv, uni, false);
            self.env.insert((id, *l), v);
        }
    }

    fn block(&mut self, fr: usize, b: &'m Block) -> bool {
        let base = self.facts.len();
        for s in b.iter() {
            if !self.stmt(fr, s) {
                self.facts.truncate(base);
                return false;
            }
        }
        self.facts.truncate(base);
        true
    }

    fn call(&mut self, function: Handle<Function>, args: Vec<Abs>, ctx: &str) -> Option<Abs> {
        if self.frames.len() > 32 {
            return None;
        }
        let func = &self.module.functions[function];
        self.next_frame += 1;
        let id = self.next_frame;
        self.frames.push(Frame { func, fkey: Some(function), id, ctx: ctx.to_string(), args, vals: HashMap::new(), ret: None });
        let fr = self.frames.len() - 1;
        self.init_locals(fr);
        let saved_taint = self.ret_taint;
        // A return in the callee ends the callee, not the caller's loops.
        let saved_exits = std::mem::take(&mut self.loop_exits);
        let saved_skip = std::mem::replace(&mut self.skip_taint, false);
        let (su, si) = (self.ctrl_uni, self.ctrl_inv);
        self.ret_taint = false;
        let facts = self.facts.len();
        self.block(fr, &func.body);
        self.facts.truncate(facts);
        self.ret_taint = saved_taint;
        self.loop_exits = saved_exits;
        self.skip_taint = saved_skip;
        self.ctrl_uni = su;
        self.ctrl_inv = si;
        let frame = self.frames.pop().expect("the frame");
        self.env.retain(|(i, _), _| *i != id);
        frame.ret
    }

    /// [`An::pure`] of `f`, computed once.
    fn pure_fn(&mut self, f: Handle<Function>) -> bool {
        if let Some(&p) = self.pure.get(&f) {
            return p;
        }
        // (A recursive call is not WGSL; seen first, it is impure.)
        self.pure.insert(f, false);
        let func = &self.module.functions[f];
        let types = &self.module.types;
        let p = !func.arguments.iter().any(|a| matches!(types[a.ty].inner, TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. })) && !func.expressions.iter().any(|(_, e)| matches!(e, Expression::GlobalVariable(_))) && self.pure_block(&func.body);
        self.pure.insert(f, p);
        p
    }

    fn pure_block(&mut self, b: &'m Block) -> bool {
        b.iter().all(|s| match s {
            Statement::Emit(_) | Statement::Break | Statement::Continue | Statement::Return { .. } | Statement::Kill | Statement::Store { .. } => true,
            Statement::Block(x) => self.pure_block(x),
            Statement::If { accept, reject, .. } => self.pure_block(accept) && self.pure_block(reject),
            Statement::Switch { cases, .. } => cases.iter().all(|c| self.pure_block(&c.body)),
            Statement::Loop { body, continuing, .. } => self.pure_block(body) && self.pure_block(continuing),
            Statement::Call { function, .. } => self.pure_fn(*function),
            _ => false,
        })
    }

    /// The identity of a pure call's opaque value by its arguments.
    fn call_id(&mut self, f: Handle<Function>, args: &[Abs]) -> u64 {
        if let Some((_, _, id)) = self.call_ids.iter().find(|(g, a, _)| *g == f && a.as_slice() == args) {
            return *id;
        }
        self.next_id += 1;
        self.call_ids.push((f, args.to_vec(), self.next_id));
        self.next_id
    }

    fn init_locals(&mut self, fr: usize) {
        let f = self.frames[fr].func;
        let id = self.frames[fr].id;
        for (h, l) in f.local_variables.iter() {
            let v = match l.init {
                Some(e) => self.eval(fr, e),
                None => self.zero_of(l.ty),
            };
            self.env.insert((id, h), v);
        }
    }

    /// The entry point's arguments: the builtins as forms over the coordinates.
    fn entry_args(&mut self, f: &Function) -> Vec<Abs> {
        let lin = |v: Var, c: i128| Lin::var(v, Poly::konst(c));
        let mut out = Vec::new();
        let wid = |an: &An, c: usize| -> Lin { if an.contract.workgroups[c].1 > 1 { lin(Var::Wid(c as u8), 1) } else { Lin::default() } };
        let lid = |an: &An, c: usize| -> Lin { if an.wg[c] > 1 { lin(Var::Lid(c as u8), 1) } else { Lin::default() } };
        for a in &f.arguments {
            let v = match &a.binding {
                Some(naga::Binding::BuiltIn(b)) => match b {
                    naga::BuiltIn::GlobalInvocationId => Abs::Vec((0..3).map(|c| Abs::Lin(wid(self, c).scale(&Poly::konst(i128::from(self.wg[c]))).add(&lid(self, c)))).collect()),
                    naga::BuiltIn::LocalInvocationId => Abs::Vec((0..3).map(|c| Abs::Lin(lid(self, c))).collect()),
                    naga::BuiltIn::WorkGroupId => Abs::Vec((0..3).map(|c| Abs::Lin(wid(self, c))).collect()),
                    naga::BuiltIn::NumWorkGroups => Abs::Vec(
                        (0..3)
                            .map(|c| {
                                let (lo, hi) = self.contract.workgroups[c];
                                let s = self.sym(&format!("nwg.{}", ["x", "y", "z"][c]), lo, hi);
                                Abs::Lin(Lin::konst(Poly::sym(s)))
                            })
                            .collect(),
                    ),
                    naga::BuiltIn::WorkGroupSize => Abs::Vec((0..3).map(|c| Abs::Lin(Lin::konst(Poly::konst(i128::from(self.wg[c]))))).collect()),
                    naga::BuiltIn::LocalInvocationIndex => {
                        let (wx, wy) = (i128::from(self.wg[0]), i128::from(self.wg[1]));
                        Abs::Lin(lid(self, 0).add(&lid(self, 1).scale(&Poly::konst(wx))).add(&lid(self, 2).scale(&Poly::konst(wx * wy))))
                    }
                    naga::BuiltIn::SubgroupSize => {
                        let s = self.sym("subgroup_size", 4, 128);
                        Abs::Lin(Lin::konst(Poly::sym(s)))
                    }
                    naga::BuiltIn::NumSubgroups => {
                        let n = i128::from(self.wg[0] * self.wg[1] * self.wg[2]);
                        let s = self.sym("num_subgroups", 1, n);
                        Abs::Lin(Lin::konst(Poly::sym(s)))
                    }
                    naga::BuiltIn::SubgroupInvocationId => {
                        let d = self.digit("subgroup_invocation_id", Some(Poly::konst(127)), false);
                        Abs::Lin(lin(d, 1))
                    }
                    naga::BuiltIn::SubgroupId => {
                        let n = i128::from(self.wg[0] * self.wg[1] * self.wg[2]);
                        let d = self.digit("subgroup_id", Some(Poly::konst(n - 1)), false);
                        Abs::Lin(lin(d, 1))
                    }
                    _ => self.other(false, false, false),
                },
                _ => self.other(false, false, false),
            };
            out.push(v);
        }
        out
    }

    fn walk_entry(&mut self) {
        let ep = self.module.entry_points.iter().find(|e| e.stage == naga::ShaderStage::Compute && e.name == self.entry).expect("the entry point (checked)");
        let f = &ep.function;
        self.env.clear();
        self.privates.clear();
        self.facts.clear();
        self.wg_open.clear();
        self.frames.clear();
        self.ctrl_uni = true;
        self.ctrl_inv = true;
        self.ret_taint = false;
        self.skip_taint = false;
        self.steps = 0;
        let args = self.entry_args(f);
        self.next_frame += 1;
        self.frames.push(Frame { func: f, fkey: None, id: self.next_frame, ctx: "main".into(), args, vals: HashMap::new(), ret: None });
        self.init_locals(0);
        self.block(0, &f.body);
    }

    // ------------------------------------------------------- the proof

    fn var_hi_poly(&self, v: Var) -> Option<Poly> {
        match v {
            Var::Wid(c) => {
                let name = format!("nwg.{}", ["x", "y", "z"][c as usize]);
                Some(match self.sym_ix.get(&name) {
                    Some(&s) => Poly::sym(s).add(&Poly::konst(-1)),
                    None => Poly::konst(self.contract.workgroups[c as usize].1 - 1),
                })
            }
            Var::Lid(c) => Some(Poly::konst(i128::from(self.wg[c as usize]) - 1)),
            Var::Dig(d) => self.digits[d as usize].hi.clone(),
        }
    }

    /// `f = c * H + L` with `0 <= L <= c - 1` under `facts`, c = `coef * m`.
    fn split(&self, f: &Lin, m: &[u32], coef: i128, facts: &[Lin]) -> Option<(Lin, Lin)> {
        let c = Poly(BTreeMap::from([(m.to_vec(), coef)]));
        if !c.add(&Poly::konst(-1)).nonneg(&self.syms) {
            return None;
        }
        let mut h = Lin::default();
        let mut l = Lin::default();
        for (v, k) in &f.t {
            match k.div_mono(m, coef) {
                Some(q) => {
                    h.t.insert(*v, q);
                }
                None => {
                    l.t.insert(*v, k.clone());
                }
            }
        }
        if l.t.values().any(|k| !k.nonneg(&self.syms)) {
            return None;
        }
        let fits = |l: &Lin| -> bool { (l.k.nonneg(&self.syms) || self.range(l, facts).0 >= 0) && self.upper_polys(l, facts).iter().any(|u| c.add(&Poly::konst(-1)).sub(u).nonneg(&self.syms)) };
        l.k = f.k.clone();
        if fits(&l) {
            return Some((h, l));
        }
        // The constant's multiples of c to H, the rest to L (`16 + 16 d` by d:
        // 16 to each).
        let (mut kh, mut kl) = (Poly::default(), Poly::default());
        for (mono, v) in &f.k.0 {
            let one = Poly(BTreeMap::from([(mono.clone(), *v)]));
            match one.div_mono(m, coef) {
                Some(q) => kh = kh.add(&q),
                None => kl = kl.add(&one),
            }
        }
        if !kh.is_zero() {
            l.k = kl;
            h.k = kh;
            if fits(&l) {
                return Some((h, l));
            }
        }
        None
    }

    /// Split A and B together; the leaves are equations `A_part = B_part`.
    fn joint(&self, a: &Lin, b: &Lin, fa: &[Lin], fb: &[Lin], eqs: &mut Vec<(Lin, Lin)>, depth: usize) {
        // A = B is also A - k = B - k: the pair as written, then with its
        // constant parts as their difference on the side where it is
        // non-negative (a common band offset cancels).
        let delta = b.k.sub(&a.k);
        let mut variants = vec![(a.clone(), b.clone())];
        if delta.nonneg(&self.syms) {
            variants.push((Lin { t: a.t.clone(), k: Poly::default() }, Lin { t: b.t.clone(), k: delta }));
        } else if delta.scale(-1).nonneg(&self.syms) {
            variants.push((Lin { t: a.t.clone(), k: delta.scale(-1) }, Lin { t: b.t.clone(), k: Poly::default() }));
        }
        if depth < 40 && !(a.t.is_empty() && b.t.is_empty()) {
            let mut cands: Vec<(Vec<u32>, i128)> = Vec::new();
            for k in a.t.values().chain(b.t.values()) {
                if let Some((m, c)) = k.monomial() {
                    let c = c.abs();
                    if (m.is_empty() && c <= 1) || cands.contains(&(m.clone(), c)) {
                        continue;
                    }
                    cands.push((m, c));
                }
            }
            // And every symbol monomial alone, as in a coefficient or the
            // constant part (a row's offset beyond the thread's row stride:
            // a grouped block's second group, `51 * nc` beside `4 * nc * lid.y`,
            // splits only at `nc`: one of 1,000 random kernels, 2026-10-14).
            for p in a.t.values().chain(b.t.values()).chain([&a.k, &b.k]) {
                for m in p.0.keys().filter(|m| !m.is_empty()) {
                    if !cands.contains(&(m.clone(), 1)) {
                        cands.push((m.clone(), 1));
                    }
                }
            }
            // The largest strides first.
            cands.sort_by_key(|(m, c)| std::cmp::Reverse(Poly(BTreeMap::from([(m.clone(), *c)])).range(&self.syms).1));
            for (a, b) in variants.iter() {
                for (m, c) in &cands {
                    let (Some((ha, la)), Some((hb, lb))) = (self.split(a, m, *c, fa), self.split(b, m, *c, fb)) else { continue };
                    if ha.t.is_empty() && hb.t.is_empty() {
                        continue;
                    }
                    self.joint(&ha, &hb, fa, fb, eqs, depth + 1);
                    self.joint(&la, &lb, fa, fb, eqs, depth + 1);
                    return;
                }
            }
        }
        let (a, b) = variants.pop().expect("a variant");
        eqs.push((a, b));
    }

    fn ineq(f: &Lin, copy: u8) -> Option<Ineq> {
        let mut a = BTreeMap::new();
        for (v, c) in &f.t {
            a.insert((*v, copy), c.as_const()?);
        }
        Some(Ineq { a, k: f.k.clone() })
    }

    /// A form that holds a whole x beside x's quotient by c, written
    /// through the quotient and the remainder (x = c * q + r): the padded
    /// index x + x / 32 is 33 * q + r, which splits at 33 (pad-shared,
    /// 2026-10-14: 89 children refused).
    fn by_quotients(&self, f: &Lin) -> Lin {
        let mut f = f.clone();
        let qs: Vec<u32> = f.t.keys().filter_map(|v| if let Var::Dig(d) = v { Some(*d) } else { None }).collect();
        for q in qs {
            let Some((x, c, true)) = self.digits[q as usize].def.clone() else { continue };
            let Some(&r) = self.digit_ix.get(&format!("mod({x:?},{c})")) else { continue };
            let Some((v, xc)) = x.t.iter().next().and_then(|(v, p)| Some((*v, p.as_const()?))) else { continue };
            let Some(a) = f.t.get(&v).and_then(|p| p.as_const()).filter(|a| xc != 0 && a % xc == 0).map(|a| a / xc) else { continue };
            // (x's every term in f at a times its coefficient.)
            if a == 0 || !x.t.iter().all(|(w, p)| f.t.get(w).is_some_and(|fp| *fp == p.scale(a))) {
                continue;
            }
            let qr = Lin::var(Var::Dig(q), Poly::konst(c)).add(&Lin::var(Var::Dig(r), Poly::konst(1)));
            f = f.sub(&x.scale(&Poly::konst(a))).add(&qr.scale(&Poly::konst(a)));
        }
        f
    }

    /// Whether the guards on a path contradict each other over every
    /// variable's range (Fourier–Motzkin): the code under them never runs,
    /// and an access there is none (seed 18, 2026-10-14: a store of
    /// wg_v2d[lidx + 2160u] under `lidx + 2160u < 1024u`).
    fn dead(&self, facts: &[Lin]) -> bool {
        let mut sys: Vec<Ineq> = facts.iter().filter_map(|f| Self::ineq(f, 0)).collect();
        if sys.is_empty() {
            return false;
        }
        let vars: BTreeSet<(Var, u8)> = sys.iter().flat_map(|i| i.a.keys().copied()).collect();
        for (v, copy) in vars {
            sys.push(Ineq { a: BTreeMap::from([((v, copy), -1)]), k: Poly::default() });
            if let Some(hi) = self.var_hi_poly(v) {
                sys.push(Ineq { a: BTreeMap::from([((v, copy), 1)]), k: hi.scale(-1) });
            }
        }
        self.infeasible(sys)
    }

    /// Whether A (in one invocation) and B (in another, or any when
    /// `distinct` is false) never write one cell.
    fn disjoint(&self, a: &Site, b: &Site, distinct: bool, same_wg: bool) -> bool {
        self.disjoint_with(a, b, distinct, same_wg, &BTreeSet::new())
    }

    /// [`An::disjoint`] with the loop counters `equal` in both invocations.
    fn disjoint_with(&self, a: &Site, b: &Site, distinct: bool, same_wg: bool, equal: &BTreeSet<u32>) -> bool {
        let (Ok(fa), Ok(fb)) = (&a.form, &b.form) else { return false };
        let (fa, fb) = (&self.by_quotients(fa), &self.by_quotients(fb));
        let mut eqs = Vec::new();
        self.joint(fa, fb, &a.facts, &b.facts, &mut eqs, 0);
        // Equal quotients and remainders of one form: the forms are equal,
        // split again (`(x >> 5) * 32 + (x & 31)` is x).
        let mut done: BTreeSet<(Lin, i128)> = BTreeSet::new();
        loop {
            let single = |l: &Lin| -> Option<u32> {
                match (l.t.len(), l.t.iter().next(), l.k.is_zero()) {
                    (1, Some((Var::Dig(d), c)), true) if c.as_const() == Some(1) => Some(*d),
                    _ => None,
                }
            };
            let pairs: Vec<(Lin, i128, bool)> = eqs
                .iter()
                .filter_map(|(x, y)| {
                    let (dx, dy) = (single(x)?, single(y)?);
                    let (a, b) = (self.digits[dx as usize].def.clone()?, self.digits[dy as usize].def.clone()?);
                    (a == b).then_some(a)
                })
                .collect();
            let found = pairs.iter().find(|(x, c, q)| *q && pairs.iter().any(|(x2, c2, q2)| !q2 && x2 == x && c2 == c) && !done.contains(&(x.clone(), *c))).cloned();
            let Some((x, c, _)) = found else { break };
            done.insert((x.clone(), c));
            self.joint(&x, &x, &a.facts, &b.facts, &mut eqs, 0);
        }
        let mut sys: Vec<Ineq> = Vec::new();
        let both = |sys: &mut Vec<Ineq>, x: &Lin, y: &Lin| {
            // x(copy 0) - y(copy 1) = 0
            if let (Some(p), Some(q)) = (Self::ineq(x, 0), Self::ineq(&y.neg(), 1)) {
                let d = p.add(&q);
                sys.push(d.neg());
                sys.push(d);
            }
        };
        for (x, y) in &eqs {
            both(&mut sys, x, y);
        }
        for f in &a.facts {
            if let Some(i) = Self::ineq(f, 0) {
                sys.push(i);
            }
        }
        for f in &b.facts {
            if let Some(i) = Self::ineq(f, 1) {
                sys.push(i);
            }
        }
        // Every quotient and remainder digit read, in each copy, by its
        // definition x = c * q + r (and the digits x reads, in turn): two
        // remainders of x and x + 1 by 2 differ, the halves of a double
        // buffer (2026-10-14: 104 double-buffer children refused).
        let mut defined: BTreeSet<(u32, u8)> = BTreeSet::new();
        loop {
            let digits: Vec<(u32, u8)> = sys.iter().flat_map(|i| i.a.keys()).filter_map(|(v, c)| if let Var::Dig(d) = v { Some((*d, *c)) } else { None }).filter(|k| !defined.contains(k)).collect::<BTreeSet<_>>().into_iter().collect();
            if digits.is_empty() {
                break;
            }
            for (d, copy) in digits {
                defined.insert((d, copy));
                let Some((x, c, _)) = self.digits[d as usize].def.clone() else { continue };
                let (Some(&q), Some(&r)) = (self.digit_ix.get(&format!("div({x:?},{c})")), self.digit_ix.get(&format!("mod({x:?},{c})"))) else { continue };
                let Some(mut e) = Self::ineq(&x, copy) else { continue };
                *e.a.entry((Var::Dig(q), copy)).or_default() -= c;
                *e.a.entry((Var::Dig(r), copy)).or_default() -= 1;
                e.a.retain(|_, v| *v != 0);
                sys.push(e.neg());
                sys.push(e);
                defined.insert((q, copy));
                defined.insert((r, copy));
            }
        }
        let mut coords: Vec<Var> = Vec::new();
        for c in 0..3u8 {
            if self.contract.workgroups[c as usize].1 > 1 {
                coords.push(Var::Wid(c));
            }
            if self.wg[c as usize] > 1 {
                coords.push(Var::Lid(c));
            }
        }
        if same_wg {
            for c in 0..3u8 {
                if self.contract.workgroups[c as usize].1 > 1 {
                    let v = Lin::var(Var::Wid(c), Poly::konst(1));
                    both(&mut sys, &v, &v);
                }
            }
            coords.retain(|v| matches!(v, Var::Lid(_)));
            for d in equal {
                let v = Lin::var(Var::Dig(*d), Poly::konst(1));
                both(&mut sys, &v, &v);
            }
        }
        // Every variable's range, in both copies.
        let mut vars: BTreeSet<(Var, u8)> = sys.iter().flat_map(|i| i.a.keys().copied()).collect();
        if distinct {
            for v in &coords {
                vars.insert((*v, 0));
                vars.insert((*v, 1));
            }
        }
        for (v, copy) in vars {
            sys.push(Ineq { a: BTreeMap::from([((v, copy), -1)]), k: Poly::default() });
            if let Some(hi) = self.var_hi_poly(v) {
                sys.push(Ineq { a: BTreeMap::from([((v, copy), 1)]), k: hi.scale(-1) });
            }
        }
        if !distinct {
            return self.infeasible(sys);
        }
        coords.iter().all(|v| {
            [(0u8, 1u8), (1, 0)].iter().all(|(p, q)| {
                // v_p - v_q + 1 <= 0
                let mut s = sys.clone();
                s.push(Ineq { a: BTreeMap::from([((*v, *p), 1), ((*v, *q), -1)]), k: Poly::konst(1) });
                self.infeasible(s)
            })
        })
    }

    /// Fourier–Motzkin: whether `sys` (each `a . x + k <= 0`) has no rational solution.
    fn infeasible(&self, sys: Vec<Ineq>) -> bool {
        let mut set: BTreeSet<Ineq> = sys.into_iter().map(Ineq::tightened).collect();
        // Equations first (an inequality and its negation both present): one
        // with a variable of coefficient +-1 eliminates that variable from
        // every other constraint exactly, so Fourier-Motzkin starts small.
        loop {
            let eq = set.iter().find_map(|i| {
                if !set.contains(&i.neg()) {
                    return None;
                }
                i.a.iter().find(|(_, c)| c.abs() == 1).map(|(v, c)| (i.clone(), *v, *c))
            });
            let Some((e, v, c)) = eq else { break };
            let mut next = BTreeSet::new();
            for i in &set {
                let Some(&ci) = i.a.get(&v) else {
                    next.insert(i.clone());
                    continue;
                };
                // i - (ci / c) e: v's coefficient cancels (c is +-1).
                let mut r = i.add(&e.scale(-ci * c));
                r.a.remove(&v);
                next.insert(r.tightened());
            }
            set = next;
        }
        loop {
            let mut rest = BTreeSet::new();
            for i in set {
                if i.a.is_empty() {
                    if i.k.add(&Poly::konst(-1)).nonneg(&self.syms) {
                        return true;
                    }
                } else {
                    rest.insert(i);
                }
            }
            set = rest;
            let vars: BTreeSet<(Var, u8)> = set.iter().flat_map(|i| i.a.keys().copied()).collect();
            let Some(v) = vars.into_iter().min_by_key(|v| {
                let p = set.iter().filter(|i| i.a.get(v).is_some_and(|c| *c > 0)).count();
                let n = set.iter().filter(|i| i.a.get(v).is_some_and(|c| *c < 0)).count();
                p * n
            }) else {
                return false;
            };
            let (with, without): (Vec<Ineq>, Vec<Ineq>) = set.into_iter().partition(|i| i.a.contains_key(&v));
            let mut next: BTreeSet<Ineq> = without.into_iter().collect();
            let pos: Vec<&Ineq> = with.iter().filter(|i| i.a[&v] > 0).collect();
            let neg: Vec<&Ineq> = with.iter().filter(|i| i.a[&v] < 0).collect();
            for p in &pos {
                for n in &neg {
                    let (cp, cn) = (p.a[&v], -n.a[&v]);
                    let mut c = p.scale(cn).add(&n.scale(cp));
                    c.a.remove(&v);
                    next.insert(c.tightened());
                }
            }
            // Of the constraints on one combination of variables, only the
            // strongest (the largest constant: a . x <= -k) is kept.
            let mut by_a: BTreeMap<BTreeMap<(Var, u8), i128>, Vec<Poly>> = BTreeMap::new();
            for i in next {
                let ks = by_a.entry(i.a).or_default();
                if ks.iter().any(|k| k.sub(&i.k).nonneg(&self.syms)) {
                    continue;
                }
                ks.retain(|k| !i.k.sub(k).nonneg(&self.syms));
                ks.push(i.k);
            }
            let next: BTreeSet<Ineq> = by_a.into_iter().flat_map(|(a, ks)| ks.into_iter().map(move |k| Ineq { a: a.clone(), k })).collect();
            if next.len() > 20_000 {
                return false;
            }
            set = next;
        }
    }
}

/// `a . x + k <= 0` over (variable, copy).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Ineq {
    a: BTreeMap<(Var, u8), i128>,
    k: Poly,
}

impl Ineq {
    fn add(&self, o: &Ineq) -> Ineq {
        let mut a = self.a.clone();
        for (v, c) in &o.a {
            let e = a.entry(*v).or_insert(0);
            *e += c;
            if *e == 0 {
                a.remove(v);
            }
        }
        Ineq { a, k: self.k.add(&o.k) }
    }

    fn scale(&self, c: i128) -> Ineq {
        Ineq { a: self.a.iter().map(|(v, x)| (*v, x * c)).collect(), k: self.k.scale(c) }
    }

    fn neg(&self) -> Ineq {
        self.scale(-1)
    }

    /// Divided by the gcd of its coefficients, the constant rounded (an
    /// integer solution satisfies the tightened inequality).
    fn tightened(self) -> Ineq {
        fn gcd(a: i128, b: i128) -> i128 {
            if b == 0 { a.abs() } else { gcd(b, a % b) }
        }
        let g = self.a.values().fold(0, |g, c| gcd(g, *c));
        if g <= 1 {
            return self;
        }
        match self.k.as_const() {
            // g y + k <= 0  →  y <= floor(-k / g)  →  y + ceil(k / g) <= 0
            Some(k) => Ineq { a: self.a.iter().map(|(v, c)| (*v, c / g)).collect(), k: Poly::konst(k.div_euclid(g) + i128::from(k.rem_euclid(g) != 0)) },
            None => self,
        }
    }
}

// ------------------------------------------------------------------ entry

/// Prove `module` safe under `contract`, or name the first rule it breaks.
pub fn check_module(module: &Module, contract: &Contract) -> Result<Report, Unsafe> {
    // Every compute entry point of the module, each its own dispatch under
    // its own interface (v24's `ingest` and `search`, 2026-10-17); the first
    // that fails, named; the reports summed.
    let names = compute_entries(module);
    if names.is_empty() {
        return Err(unsafe_("interface", "no compute entry point".into()));
    }
    let several = names.len() > 1;
    let mut total = Report { write_sites: 0, read_sites: 0, workgroup_sites: 0, pairs: 0 };
    for name in &names {
        let r = check_entry(module, contract, name).map_err(|e| if several { Unsafe { detail: format!("entry point {name}: {}", e.detail), ..e } } else { e })?;
        total.write_sites += r.write_sites;
        total.read_sites += r.read_sites;
        total.workgroup_sites += r.workgroup_sites;
        total.pairs += r.pairs;
    }
    Ok(total)
}

/// [`check_module`] for the compute entry point `entry` alone.
fn check_entry(module: &Module, contract: &Contract, entry: &str) -> Result<Report, Unsafe> {
    let iface = infer_entry(module, entry).map_err(|e| unsafe_("interface", e))?;
    let wg = iface.workgroup_size;
    let mut an = An {
        module,
        entry: entry.to_string(),
        contract,
        iface,
        wg,
        syms: Vec::new(),
        sym_ix: HashMap::new(),
        digits: Vec::new(),
        digit_ix: HashMap::new(),
        env: HashMap::new(),
        privates: HashMap::new(),
        facts: Vec::new(),
        ctrl_uni: true,
        ctrl_inv: true,
        record: false,
        loop_stores: Vec::new(),
        loop_exits: Vec::new(),
        skip_taint: false,
        pure: HashMap::new(),
        call_ids: Vec::new(),
        value_ids: Vec::new(),
        ret_taint: false,
        sites: Vec::new(),
        site_ix: HashMap::new(),
        wg_sites: Vec::new(),
        wg_ix: HashMap::new(),
        wg_open: BTreeSet::new(),
        wg_pairs: BTreeSet::new(),
        wg_equal: BTreeMap::new(),
        loops: Vec::new(),
        errors: Vec::new(),
        frames: Vec::new(),
        next_frame: 0,
        next_id: 0,
        steps: 0,
    };
    an.record = true;
    an.walk_entry();
    if an.steps > STEP_CAP {
        return Err(unsafe_("too_large_to_prove", format!("more than {STEP_CAP} statements walked")));
    }
    if let Some(e) = an.errors.first() {
        return Err(e.clone());
    }
    let mut report = Report { write_sites: an.sites.iter().filter(|s| s.kind != Kind::Read).count(), read_sites: an.sites.iter().filter(|s| s.kind == Kind::Read).count(), workgroup_sites: an.wg_sites.len(), pairs: 0 };
    // ARENA MODE: an arena binding every access of which is in bounds, by a
    // mask or a clamp of its index, is under no ownership rule; one with an
    // access not so proved keeps the strict rules (arena mode proves all
    // strict proves), and a strict refusal on it names that access.
    let unbounded = if contract.arena_mode { arena_bounds(module, contract, entry)? } else { BTreeMap::new() };
    let arena = |s: &Site| contract.is_arena(&s.alloc) && !unbounded.contains_key(&s.alloc);
    let named = |e: Unsafe, sites: &[&Site]| -> Unsafe {
        match sites.iter().find_map(|s| unbounded.get(&s.alloc)) {
            Some(why) => unsafe_("arena_index_not_in_bounds", format!("{why} (so it keeps the strict rules: {}: {})", e.rule, e.detail)),
            None => e,
        }
    };
    // Every plain write's index affine, every thread stride non-zero. An
    // atomic site needs neither: every access to an atomic global is atomic
    // (Kind::Atomic, loads and stores included), and atomic accesses do not
    // race one another (the pairs below), whatever their index (v24's
    // `atomicAdd(&index_u32[hash & mask], 1u)`, 2026-10-17).
    for s in an.sites.iter().filter(|s| s.kind == Kind::Write && !arena(s)) {
        match &s.form {
            Err((rule, why)) => return Err(named(unsafe_(rule, format!("{}: {why}", s.what)), &[s])),
            Ok(f) => {
                for (v, c) in &f.t {
                    if v.thread() && !c.add(&Poly::konst(-1)).nonneg(&an.syms) {
                        return Err(named(unsafe_("store_stride_may_be_zero", format!("{}: the stride {} of a thread coordinate is not proved at least 1", s.what, c.show(&an.syms))), &[s]));
                    }
                }
            }
        }
    }
    let writes: Vec<usize> = (0..an.sites.len()).filter(|i| an.sites[*i].kind != Kind::Read && !arena(&an.sites[*i])).collect();
    for (x, &i) in writes.iter().enumerate() {
        for &j in &writes[x..] {
            let (a, b) = (&an.sites[i], &an.sites[j]);
            if a.alloc != b.alloc || (a.kind == Kind::Atomic && b.kind == Kind::Atomic) {
                continue;
            }
            report.pairs += 1;
            let same_form = a.form == b.form;
            let distinct = i == j || same_form || a.kind == Kind::Atomic || b.kind == Kind::Atomic;
            if !an.disjoint(a, b, distinct, false) {
                let how = if distinct { "two invocations may write one cell" } else { "two store sites of different index forms may write one cell" };
                return Err(named(unsafe_("write_sets_overlap", format!("{how}: {} and {}", a.what, b.what)), &[a, b]));
            }
        }
    }
    for r in (0..an.sites.len()).filter(|i| an.sites[*i].kind == Kind::Read && !arena(&an.sites[*i])) {
        for &w in &writes {
            let (a, b) = (&an.sites[r], &an.sites[w]);
            if a.alloc != b.alloc {
                continue;
            }
            report.pairs += 1;
            if !an.disjoint(a, b, true, false) {
                return Err(named(unsafe_("output_read_of_another_invocations_cell", format!("{} may read a cell another invocation's {} writes", a.what, b.what)), &[a, b]));
            }
        }
    }
    let pairs: Vec<(usize, usize)> = an.wg_pairs.iter().copied().collect();
    for (i, j) in pairs {
        let (a, b) = (&an.wg_sites[i], &an.wg_sites[j]);
        report.pairs += 1;
        let equal = an.wg_equal.get(&(i, j)).cloned().unwrap_or_default();
        if !an.disjoint_with(a, b, true, true, &equal) {
            return Err(unsafe_("workgroup_memory_race", format!("{} and {} may touch one element in two invocations of a workgroup with no barrier between", a.what, b.what)));
        }
    }
    Ok(report)
}

/// The largest value an index expression can take, proved from its form
/// alone: a literal or constant, a mask `x & c`, a clamp `min(x, c)` or
/// `clamp(x, lo, c)`, a remainder `x % c`, a shift right of a bounded value,
/// sums, products and selects of bounded values, a call whose every return
/// is bounded; `min(x, arrayLength(&b) - 1u)` is in bounds of b by its own
/// length (`Bound::Len`). None when the form proves no bound.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Bound {
    Max(u128),
    Len,
}

fn bound_of(module: &Module, f: &Function, e: Handle<Expression>, of: Handle<naga::GlobalVariable>, depth: usize) -> Option<Bound> {
    use naga::Literal;
    if depth > 64 {
        return None;
    }
    let max = |b: Option<Bound>| match b {
        Some(Bound::Max(m)) => Some(m),
        _ => None,
    };
    let lit = |l: &Literal| -> Option<u128> {
        match *l {
            Literal::U32(v) => Some(u128::from(v)),
            Literal::I32(v) => u128::try_from(v).ok(),
            Literal::AbstractInt(v) => u128::try_from(v).ok(),
            _ => None,
        }
    };
    let global = |h: Handle<Expression>| -> Option<u128> {
        match &module.global_expressions[h] {
            Expression::Literal(l) => lit(l),
            _ => None,
        }
    };
    let rec = |x: Handle<Expression>| bound_of(module, f, x, of, depth + 1);
    match &f.expressions[e] {
        Expression::Literal(l) => lit(l).map(Bound::Max),
        Expression::Constant(c) => global(module.constants[*c].init).map(Bound::Max),
        Expression::ZeroValue(_) => Some(Bound::Max(0)),
        Expression::Binary { op: BinaryOperator::And, left, right } => {
            let (a, b) = (max(rec(*left)), max(rec(*right)));
            a.into_iter().chain(b).min().map(Bound::Max)
        }
        Expression::Binary { op: BinaryOperator::Modulo, right, .. } => max(rec(*right)).filter(|c| *c >= 1).map(|c| Bound::Max(c - 1)),
        Expression::Binary { op: BinaryOperator::ShiftRight, left, .. } => max(rec(*left)).map(Bound::Max),
        Expression::Binary { op: BinaryOperator::Divide, left, .. } => max(rec(*left)).map(Bound::Max),
        Expression::Binary { op: BinaryOperator::Add, left, right } => Some(Bound::Max(max(rec(*left))?.checked_add(max(rec(*right))?)?)),
        Expression::Binary { op: BinaryOperator::Multiply, left, right } => Some(Bound::Max(max(rec(*left))?.checked_mul(max(rec(*right))?)?)),
        Expression::Math { fun: MathFunction::Min, arg, arg1: Some(b), .. } => {
            // A clamp to the binding's own last element.
            let len_minus_one = |x: Handle<Expression>| -> bool {
                let Expression::Binary { op: BinaryOperator::Subtract, left, right } = &f.expressions[x] else { return false };
                let one = matches!(&f.expressions[*right], Expression::Literal(l) if lit(l) == Some(1));
                let Expression::ArrayLength(p) = &f.expressions[*left] else { return false };
                one && matches!(&f.expressions[*p], Expression::GlobalVariable(g) if *g == of)
            };
            if len_minus_one(*arg) || len_minus_one(*b) {
                return Some(Bound::Len);
            }
            let (x, y) = (rec(*arg), rec(*b));
            if x == Some(Bound::Len) || y == Some(Bound::Len) {
                return Some(Bound::Len);
            }
            max(x).into_iter().chain(max(y)).min().map(Bound::Max)
        }
        Expression::Math { fun: MathFunction::Clamp, arg2: Some(hi), .. } => rec(*hi),
        Expression::Math { fun: MathFunction::Max, arg, arg1: Some(b), .. } => Some(Bound::Max(max(rec(*arg))?.max(max(rec(*b))?))),
        Expression::Select { accept, reject, .. } => Some(Bound::Max(max(rec(*accept))?.max(max(rec(*reject))?))),
        Expression::CallResult(callee) => {
            // Every return of the callee bounded (its arguments unknown).
            let g = &module.functions[*callee];
            let mut returns = Vec::new();
            fn walk(b: &Block, out: &mut Vec<Handle<Expression>>) {
                for s in b.iter() {
                    match s {
                        Statement::Return { value: Some(v) } => out.push(*v),
                        Statement::Block(x) => walk(x, out),
                        Statement::If { accept, reject, .. } => {
                            walk(accept, out);
                            walk(reject, out);
                        }
                        Statement::Loop { body, continuing, .. } => {
                            walk(body, out);
                            walk(continuing, out);
                        }
                        Statement::Switch { cases, .. } => cases.iter().for_each(|c| walk(&c.body, out)),
                        _ => {}
                    }
                }
            }
            walk(&g.body, &mut returns);
            let each: Option<Vec<u128>> = returns.iter().map(|r| max(bound_of(module, g, *r, of, depth + 1))).collect();
            each.and_then(|v| v.into_iter().max()).map(Bound::Max)
        }
        _ => None,
    }
}

/// Arena mode's bounds proof: every access (`Access` / `AccessIndex`) into
/// an arena binding, in any function the entry point reaches, at an index
/// proved below the binding's length (the contract's least
/// `arrayLength(<binding>)`, or the binding's own length by a clamp to it).
/// The arena bindings with an access not so proved, each with its first
/// such access named (those bindings keep the strict rules).
fn arena_bounds(module: &Module, contract: &Contract, entry: &str) -> Result<BTreeMap<String, String>, Unsafe> {
    let mut unbounded: BTreeMap<String, String> = BTreeMap::new();
    let ep = module.entry_points.iter().find(|e| e.name == entry).ok_or_else(|| unsafe_("interface", format!("no entry point {entry}")))?;
    // The functions the entry point reaches, through its calls.
    fn calls(b: &Block, out: &mut Vec<Handle<Function>>) {
        for s in b.iter() {
            match s {
                Statement::Call { function, .. } => out.push(*function),
                Statement::Block(x) => calls(x, out),
                Statement::If { accept, reject, .. } => {
                    calls(accept, out);
                    calls(reject, out);
                }
                Statement::Loop { body, continuing, .. } => {
                    calls(body, out);
                    calls(continuing, out);
                }
                Statement::Switch { cases, .. } => cases.iter().for_each(|c| calls(&c.body, out)),
                _ => {}
            }
        }
    }
    let mut reached: Vec<Handle<Function>> = Vec::new();
    let mut todo = Vec::new();
    calls(&ep.function.body, &mut todo);
    while let Some(h) = todo.pop() {
        if !reached.contains(&h) {
            reached.push(h);
            calls(&module.functions[h].body, &mut todo);
        }
    }
    let mut fns: Vec<&Function> = vec![&ep.function];
    fns.extend(reached.iter().map(|h| &module.functions[*h]));
    for f in fns {
        for (_, e) in f.expressions.iter() {
            let (base, index) = match e {
                Expression::Access { base, index } => (*base, Some(*index)),
                Expression::AccessIndex { base, .. } => (*base, None),
                _ => continue,
            };
            let Expression::GlobalVariable(g) = &f.expressions[base] else { continue };
            let name = module.global_variables[*g].name.clone().unwrap_or_default();
            if !contract.is_arena(&name) {
                continue;
            }
            let least = contract.params.get(&format!("arrayLength({name})")).map(|(lo, _)| *lo);
            let at = match (e, index) {
                (Expression::AccessIndex { index, .. }, _) => Some(Bound::Max(u128::from(*index))),
                (_, Some(i)) => bound_of(module, f, i, *g, 0),
                (_, None) => None,
            };
            let why = match (at, least) {
                (Some(Bound::Len), _) => None,
                (Some(Bound::Max(m)), Some(n)) if i128::try_from(m).is_ok_and(|m| m < n) => None,
                (Some(Bound::Max(m)), Some(n)) => Some(format!("its index may reach {m}, the arena holds {n}")),
                (Some(Bound::Max(m)), None) => Some(format!("its index may reach {m}, and the contract gives no arrayLength({name})")),
                (None, _) => Some("its index is proved below no bound (mask it or clamp it)".to_string()),
            };
            if let Some(why) = why {
                unbounded.entry(name.clone()).or_insert_with(|| format!("an access to the arena {name} in {}: {why}", f.name.clone().unwrap_or_else(|| entry.to_string())));
            }
        }
    }
    Ok(unbounded)
}

/// [`check_module`] over WGSL text.
pub fn check_text(source: &str, contract: &Contract) -> Result<Report, Unsafe> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| unsafe_("parse", e.emit_to_string(source)))?;
    check_module(&module, contract)
}

/// The Gram problem's contract (`kingdoms/safewgsl/contracts/gram.contract`).
pub fn gram_contract() -> Contract {
    Contract::parse(include_str!("../../kingdoms/safewgsl/contracts/gram.contract")).expect("the Gram contract parses")
}

/// The Gram contract of a kernel that writes the whole H (`// mode full`).
pub fn gram_full_contract() -> Contract {
    gram_contract().full()
}

/// The contract a kernel text runs under in the Gram problem: the full-mode
/// one when it carries `// mode full`.
pub fn gram_contract_for(source: &str) -> Contract {
    if source.lines().any(|l| l.trim() == "// mode full") { gram_full_contract() } else { gram_contract() }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRE: &str = "struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
";

    fn gram(body: &str, wg: &str) -> String {
        format!("{PRE}@compute @workgroup_size({wg})\nfn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n{body}\n}}\n")
    }

    fn rule(src: &str, c: &Contract) -> &'static str {
        match check_text(src, c) {
            Ok(r) => panic!("proved safe ({r:?}):\n{src}"),
            Err(e) => {
                println!("refused as expected: {e}");
                e.rule
            }
        }
    }

    fn proved(src: &str, c: &Contract) -> Report {
        check_text(src, c).unwrap_or_else(|e| panic!("{e}\n{src}"))
    }

    fn one_d() -> Contract {
        let mut c = gram_contract();
        c.workgroups = [(1, 4), (1, 1), (1, 1)];
        c
    }

    #[test]
    fn the_shipped_blocked_and_tiled_gram_kernels_are_proved() {
        for (name, src) in [("shipped", include_str!("../../kingdoms/wgsl/gram/shipped.wgsl")), ("blocked", include_str!("../../kingdoms/wgsl/gram/blocked.wgsl")), ("tiled", include_str!("../../kingdoms/wgsl/gram/tiled.wgsl"))] {
            let r = check_text(src, &gram_contract()).unwrap_or_else(|e| panic!("{name}: {e}"));
            println!("{name}: {r:?}");
            assert!(r.write_sites >= 1);
        }
    }

    /// The planted races, each refused by its rule (and its safe twin proved).
    #[test]
    fn planted_races_are_refused() {
        let c = one_d();
        // Two branches whose indices overlap: thread 0 and thread 8 write cell 0.
        assert_eq!(rule(&gram("    if (gid.x < 8u) { h[gid.x] = 1.0; } else { h[gid.x - 8u] = 2.0; }", "16, 1, 1"), &c), "write_sets_overlap");
        proved(&gram("    if (gid.x < 8u) { h[2u * gid.x] = 1.0; } else { h[2u * gid.x + 1u] = 2.0; }", "16, 1, 1"), &c);
        // A uniform stride that may be zero, and the same stride proved at least 1.
        let mut cz = c.clone();
        cz.params.insert("cfg.m".into(), (0, 64));
        assert_eq!(rule(&gram("    h[cfg.m * gid.x] = 1.0;", "16, 1, 1"), &cz), "store_stride_may_be_zero");
        cz.params.insert("cfg.m".into(), (1, 64));
        proved(&gram("    h[cfg.m * gid.x] = 1.0;", "16, 1, 1"), &cz);
        // wid * W + lid with the wrong W (8; the workgroup is 16 wide).
        assert_eq!(rule(&gram("    h[wid.x * 8u + lid.x] = 1.0;", "16, 1, 1"), &c), "write_sets_overlap");
        proved(&gram("    h[wid.x * 16u + lid.x] = 1.0;", "16, 1, 1"), &c);
        // A wrapping offset: gid.x + 2^32 - 1 is gid.x - 1 for gid.x >= 1.
        assert_eq!(rule(&gram("    h[gid.x + 4294967295u] = 1.0;", "16, 1, 1"), &c), "store_index_may_wrap");
        // Non-affine indices: two threads share gid.x / 2; a masked square.
        assert_eq!(rule(&gram("    h[gid.x / 2u] = 1.0;", "16, 1, 1"), &c), "write_sets_overlap");
        assert_eq!(rule(&gram("    h[(gid.x * gid.x) & 1023u] = 1.0;", "16, 1, 1"), &c), "write_sets_overlap");
        assert_eq!(rule(&gram("    h[select(gid.x, 0u, gid.x > 3u)] = 1.0;", "16, 1, 1"), &c), "store_index_not_affine");
        // A dimension the index does not tell apart: rows y of one column x.
        let mut c2 = c.clone();
        c2.workgroups = [(1, 4), (1, 4), (1, 1)];
        assert_eq!(rule(&gram("    h[gid.x] = 1.0;", "16, 1, 1"), &c2), "write_sets_overlap");
    }

    /// The symmetric half (2026-10-14): the thread at (i, j) writes h[i*d + j]
    /// and its mirror h[j*d + i]. Without a triangle guard two threads write
    /// one cell; under i < j, the diagonal written once by its own store,
    /// every cell has one writer; under i <= j for both stores one thread
    /// writes the diagonal twice (an accumulation doubled).
    #[test]
    fn the_mirror_pair_is_proved_only_under_its_triangle_guard() {
        let c = gram_full_contract();
        let head = "    let i = gid.y; let j = gid.x; let d = cfg.d;\n    if (i >= d || j >= d) { return; }\n";
        let unguarded = gram(&format!("{head}    h[i * d + j] = h[i * d + j] + 1.0;\n    h[j * d + i] = h[j * d + i] + 1.0;"), "16, 16, 1");
        assert_eq!(rule(&unguarded, &c), "write_sets_overlap");
        let half = gram(&format!("{head}    if (i < j) {{\n        h[i * d + j] = h[i * d + j] + 1.0;\n        h[j * d + i] = h[j * d + i] + 1.0;\n    }}\n    if (i == j) {{ h[i * d + j] = h[i * d + j] + 1.0; }}"), "16, 16, 1");
        println!("guarded half: {:?}", proved(&half, &c));
        let diag_twice = gram(&format!("{head}    if (i <= j) {{\n        h[i * d + j] = h[i * d + j] + 1.0;\n        h[j * d + i] = h[j * d + i] + 1.0;\n    }}"), "16, 16, 1");
        assert_eq!(rule(&diag_twice, &c), "write_sets_overlap");
        // The mirror under i < j inside an i <= j guard (a diagonal block's
        // lower half written once): proved.
        let inner = gram(&format!("{head}    if (i <= j) {{\n        h[i * d + j] = h[i * d + j] + 1.0;\n        if (i < j) {{ h[j * d + i] = h[j * d + i] + 1.0; }}\n    }}"), "16, 16, 1");
        proved(&inner, &c);
    }

    /// Reads of output cells another invocation writes, workgroup memory
    /// touched by two invocations with no barrier between, a subgroup
    /// operation under a thread's condition.
    #[test]
    fn reads_of_others_cells_and_unordered_workgroup_memory_are_refused() {
        let c = one_d();
        assert_eq!(rule(&gram("    h[gid.x] = h[gid.x + 1u] + 1.0;", "16, 1, 1"), &c), "output_read_of_another_invocations_cell");
        proved(&gram("    h[gid.x] = h[gid.x] + 1.0;", "16, 1, 1"), &c);
        let wg = |body: &str| format!("{PRE}var<workgroup> t: array<f32, 64>;\n@compute @workgroup_size(16, 1, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{\n{body}\n}}\n");
        assert_eq!(rule(&wg("    t[lid.x] = phi[gid.x];\n    h[gid.x] = t[(lid.x + 1u) & 15u];"), &c), "workgroup_memory_race");
        proved(&wg("    t[lid.x] = phi[gid.x];\n    workgroupBarrier();\n    h[gid.x] = t[(lid.x + 1u) & 15u];"), &c);
        // Loop-carried: the next pass overwrites what this pass's neighbours read.
        assert_eq!(rule(&wg("    for (var k = 0u; k < 4u; k = k + 1u) {\n        t[lid.x] = phi[gid.x + k];\n        workgroupBarrier();\n        h[gid.x] = h[gid.x] + t[15u - lid.x];\n    }"), &c), "workgroup_memory_race");
        proved(&wg("    for (var k = 0u; k < 4u; k = k + 1u) {\n        t[lid.x] = phi[gid.x + k];\n        workgroupBarrier();\n        h[gid.x] = h[gid.x] + t[15u - lid.x];\n        workgroupBarrier();\n    }"), &c);
        let sg = format!("{PRE}@compute @workgroup_size(16, 1, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    var v = phi[gid.x];\n    if (gid.x < 3u) {{ v = subgroupAdd(v); }}\n    h[gid.x] = v;\n}}\n");
        assert_eq!(rule(&sg, &c), "subgroup_operation_in_nonuniform_control");
    }

    /// A strided preload (nluca's Sonnet kernels, 2026-10-15): `for (t = 0;
    /// t < 64; t += 16) { w[t + lid.x] = ...; }` with 16 threads writes each
    /// element once; the counter is its start plus 16 per pass, so the two
    /// writes of two threads meet only at one thread and one pass. Proved,
    /// with the bound behind a pure call as the kingdom renders it
    /// (`min(lit_u32(64u), ...)`). Still races: a step of 1, a step of 8
    /// (thread 8's pass 0 is thread 0's pass 1), and a step of 16 whose
    /// bound lets the counter wrap (then it is any u32).
    #[test]
    fn a_strided_preload_is_proved_and_its_overlapping_twins_are_refused() {
        let c = one_d();
        let wg = |head: &str| format!("{PRE}var<workgroup> t: array<f32, 64>;\nfn lit_u32(x: u32) -> u32 {{ return x; }}\n@compute @workgroup_size(16, 1, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{\n    for (var k = 0u; {head}) {{\n        if ((k + lid.x) < 64u) {{ t[(k + lid.x) & 63u] = phi[k + lid.x]; }}\n    }}\n    workgroupBarrier();\n    h[gid.x] = t[(lid.x * 3u) & 63u];\n}}\n");
        proved(&wg("k < 64u; k = k + 16u"), &c);
        proved(&wg("k < min(lit_u32(64u), 16777216u); k = k + 16u"), &c);
        assert_eq!(rule(&wg("k < 64u; k = k + 1u"), &c), "workgroup_memory_race");
        assert_eq!(rule(&wg("k < 64u; k = k + 8u"), &c), "workgroup_memory_race");
        assert_eq!(rule(&wg("k < 4294967295u; k = k + 16u"), &c), "workgroup_memory_race");
    }

    /// Uniformity through loops (2026-10-14): a local is non-uniform at a
    /// loop's head when a pass of that loop stores a thread's value in it,
    /// not because a later statement does; a continue under a thread's
    /// condition makes the rest of its pass non-uniform, not the loop's
    /// counter, its exit or a later loop. Atomic accesses do not race one
    /// another; an access under guards that contradict each other never runs.
    #[test]
    fn loops_atomics_and_dead_accesses() {
        let c = one_d();
        let k = |body: &str, decls: &str| format!("{PRE}{decls}@compute @workgroup_size(16, 1, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{\n    var t: u32 = 0u;\n    var x: u32 = 0u;\n{body}\n    h[gid.x] = h[gid.x] + phi[0];\n}}\n");
        // The item loop's remainder after a main loop with a thread's
        // continue (seed 141); a barrier after that continue in its pass.
        proved(&k("    for (t = 0u; t < cfg.m; t = t + 4u) {\n        if (lid.x < 3u) { continue; }\n    }\n    for (; t < cfg.m; t = t + 1u) {\n        storageBarrier();\n    }", ""), &c);
        assert_eq!(rule(&k("    for (t = 0u; t < cfg.m; t = t + 4u) {\n        if (lid.x < 3u) { continue; }\n        storageBarrier();\n    }", ""), &c), "barrier_in_nonuniform_control");
        // A thread's value stored after the loop, then in the loop's next pass.
        proved(&k("    for (t = 0u; t < x + 4u; t = t + 1u) {\n        workgroupBarrier();\n    }\n    x = lid.x;", ""), &c);
        assert_eq!(rule(&k("    for (t = 0u; t < 4u; t = t + 1u) {\n        if (x < 2u) { workgroupBarrier(); }\n        x = lid.x;\n    }", ""), &c), "barrier_in_nonuniform_control");
        // A workgroup atomic stored and added to by every thread.
        proved(&k("    atomicStore(&a, 0u);\n    _ = atomicAdd(&a, 1u);\n    _ = atomicLoad(&a);", "var<workgroup> a: atomic<u32>;\n"), &c);
        // A put no thread executes (seed 18: lidx + 2160 into 1,024 elements), and one some do.
        proved(&k("    if (lid.x + 32u < 16u) { w[lid.x + 32u] = vec2<f32>(1.0); }", "var<workgroup> w: array<vec2<f32>, 16>;\n"), &c);
        proved(&k("    if (lid.x + 8u < 16u) { w[lid.x + 8u] = vec2<f32>(1.0); }", "var<workgroup> w: array<vec2<f32>, 16>;\n"), &c);
        assert_eq!(rule(&k("    if (lid.x + 8u < 16u) { w[lid.x + 8u] = w[(lid.x + 9u) & 15u]; }", "var<workgroup> w: array<vec2<f32>, 16>;\n"), &c), "workgroup_memory_race");
    }

    /// A STORAGE ARRAY OF ATOMICS AT A HASHED INDEX (2026-10-17, v24's index
    /// arena): an atomicAdd, atomicStore and atomicLoad at `hash & mask` are
    /// atomic accesses, proved whatever their index; a PLAIN store at a
    /// non-affine index still fails (not affine), one at the hash fails (two
    /// invocations may share it), and so does a plain store beside the
    /// atomics that two invocations may make to one cell.
    #[test]
    fn atomics_at_a_hashed_index_are_atomic_and_plain_writes_there_are_not() {
        let c = one_d();
        let k = |body: &str| format!("@group(0) @binding(0) var<storage, read> keys: array<f32>;\n@group(0) @binding(1) var<storage, read_write> idx: array<atomic<u32>>;\n@group(0) @binding(2) var<storage, read_write> plain: array<u32>;\n@compute @workgroup_size(16, 1, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let b = bitcast<u32>(keys[gid.x]) & 1023u;\n{body}\n}}\n");
        proved(&k("    let slot = atomicAdd(&idx[min(b, arrayLength(&idx) - 1u)], 1u);\n    atomicStore(&idx[min(1024u + slot, arrayLength(&idx) - 1u)], gid.x);\n    _ = atomicMax(&idx[b], atomicLoad(&idx[b + 1u]));"), &c);
        assert_eq!(rule(&k("    plain[b * b] = gid.x;"), &c), "store_index_not_affine");
        // (A masked hash is a bounded digit: affine, and shared by invocations.)
        assert_eq!(rule(&k("    plain[b] = gid.x;"), &c), "write_sets_overlap");
        assert_eq!(rule(&k("    _ = atomicAdd(&idx[b], 1u);\n    plain[0u] = gid.x;"), &c), "write_sets_overlap");
    }

    /// ONE WORKGROUP A QUERY (2026-10-17, every v24 search design):
    /// `cands[64u * wid.x + lid.x]` in a 16-thread workgroup is disjoint
    /// across workgroups and lanes (lid.x < 16 <= 64) under a 1-D dispatch;
    /// a stride smaller than the workgroup's width (8) overlaps; and with y
    /// and z dispatch free (the default contract) two workgroups differing in
    /// wid.y write one cell, as they would.
    #[test]
    fn a_stride_at_least_the_workgroup_width_is_disjoint_across_workgroups() {
        let one_d = Contract::parse("workgroups x 1 65535\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        let k = |stride: u32| format!("@group(0) @binding(0) var<storage, read_write> cands: array<u32>;\n@compute @workgroup_size(16, 1, 1)\nfn search(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n    cands[{stride}u * wid.x + lid.x] = lid.x;\n}}\n");
        proved(&k(64), &one_d);
        proved(&k(16), &one_d);
        assert_eq!(rule(&k(8), &one_d), "write_sets_overlap");
        assert_eq!(rule(&k(64), &Contract::default()), "write_sets_overlap");
    }

    /// TWO ENTRY POINTS (2026-10-17): each is proved as its own dispatch,
    /// under its own interface; a race in the second is named with it.
    #[test]
    fn every_entry_point_of_a_module_is_proved() {
        let c = one_d();
        let m = |search: &str| format!("@group(0) @binding(0) var<storage, read> keys: array<f32>;\n@group(0) @binding(1) var<storage, read_write> idx: array<atomic<u32>>;\n@group(0) @binding(2) var<storage, read_write> out: array<u32>;\n@compute @workgroup_size(16, 1, 1)\nfn ingest(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    _ = atomicAdd(&idx[bitcast<u32>(keys[gid.x]) & 63u], 1u);\n}}\n@compute @workgroup_size(16, 1, 1)\nfn search(@builtin(global_invocation_id) gid: vec3<u32>) {{\n{search}\n}}\n");
        let both = proved(&m("    out[gid.x] = atomicLoad(&idx[gid.x & 63u]);"), &c);
        assert_eq!(both.write_sites, 2, "the atomicAdd of ingest and the store of search: {both:?}");
        let e = check_text(&m("    out[0u] = atomicLoad(&idx[gid.x & 63u]);"), &c).expect_err("every invocation of search writes out[0]");
        assert_eq!(e.rule, "write_sets_overlap");
        assert!(e.detail.starts_with("entry point search: "), "{e}");
    }

    /// One value written twice is one value (2026-10-14): a cell's offset
    /// through a pure helper, a conversion of a literal, or a clamp by 0,
    /// written in its guard and in its index, is bounded by its guard; read
    /// from an atomic twice it is two values, and the guard bounds neither.
    /// A row offset past the thread's row stride (a grouped block's second
    /// group) splits at the column count alone.
    #[test]
    fn equal_values_and_strides() {
        let c = gram_contract();
        let k = |off: &str, decls: &str| format!("{PRE}{decls}@compute @workgroup_size(16, 16, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n    if (gid.y < cfg.r_rows && gid.x * 8u + {off} < cfg.d) {{ h[gid.y * cfg.d + gid.x * 8u + {off}] = h[gid.y * cfg.d + gid.x * 8u + {off}] + phi[0]; }}\n}}\n");
        let at = "fn at3(v: vec3<u32>, i: u32) -> u32 { var t = v; return t[min(i, 2u)]; }\n";
        proved(&k("min(at3(vec3<u32>(1u, 2u, 3u), cfg.r_off), 7u)", at), &c);
        proved(&k("min(u32(77i), 7u)", ""), &c);
        proved(&k("min(u32(phi[1]), 0u)", ""), &c);
        assert_eq!(rule(&k("min(atomicLoad(&a), 7u)", "var<workgroup> a: atomic<u32>;\n"), &c), "write_sets_overlap");
        let grouped = |a: &str, b: &str| gram(&format!("    let r = wid.y * 96u + lid.y * 4u;\n    let q = wid.x * 256u + lid.x;\n    if (r + {a} < cfg.r_rows && q + 64u < cfg.d) {{ h[(r + {a}) * cfg.d + q + 64u] = 1.0; }}\n    if (r < cfg.r_rows && q + {b} < cfg.d) {{ h[r * cfg.d + q + {b}] = 2.0; }}"), "32, 12, 1");
        proved(&grouped("51u", "128u"), &c);
        // (Row r + 4 is the next thread's first row.)
        assert_eq!(rule(&grouped("4u", "64u"), &c), "write_sets_overlap");
    }

    /// The halves of a double buffer and a padded index (2026-10-14): a pass
    /// writes half ((t >> 5) + 1) & 1 and reads half (t >> 5) & 1 (each
    /// thread reading another's element, the transpose), then a barrier;
    /// both invocations of a pair met in one pass of that uniform, barriered
    /// loop have one t, and a quotient's remainder relates x and x plus one.
    /// Refused: one half for both; the loop without its barrier (two
    /// invocations a pass apart); a continue that may skip the barrier. The
    /// padded index x + x / 32 of a thread's own x is proved, of a shared x
    /// refused.
    #[test]
    fn double_buffer_halves_and_padded_indices() {
        let c = one_d();
        let k = |body: &str| format!("{PRE}var<workgroup> w: array<f32, 1100>;\n@compute @workgroup_size(16, 16, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {{\n    var acc: f32 = 0.0;\n    var t: u32 = 0u;\n{body}\n    h[gid.x * 16u + lid.y] = h[gid.x * 16u + lid.y] + acc;\n}}\n");
        let pass = |write_half: &str, tail: &str| k(&format!("    for (t = 0u; t < cfg.m; t = t + 32u) {{\n        w[(lidx + 512u * {write_half}) & 1023u] = phi[t + lidx];\n        acc = acc + w[(lid.y + 16u * lid.x + 512u * ((t >> 5u) & 1u)) & 1023u];\n{tail}    }}"));
        let next = "(((t >> 5u) + 1u) & 1u)";
        proved(&pass(next, "        workgroupBarrier();\n"), &c);
        assert_eq!(rule(&pass("((t >> 5u) & 1u)", "        workgroupBarrier();\n"), &c), "workgroup_memory_race");
        assert_eq!(rule(&pass(next, ""), &c), "workgroup_memory_race");
        assert_eq!(rule(&pass(next, "        if (t == 64u) { continue; }\n        workgroupBarrier();\n"), &c), "workgroup_memory_race");
        proved(&k("    let x = lidx;\n    w[x + (x >> 5u)] = 1.0;"), &c);
        proved(&k("    for (var j = 0u; j < 2u; j = j + 1u) {\n        let x = lidx + 256u * j;\n        w[x + (x >> 5u)] = 1.0;\n    }"), &c);
        assert_eq!(rule(&k("    let x = lidx >> 1u;\n    w[x + (x >> 5u)] = 1.0;"), &c), "workgroup_memory_race");
    }

    /// nluca's six kernels as CubeCL emits them, under their contract.
    #[test]
    fn nlucas_six_kernels_are_proved() {
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus/nluca"));
        let c = Contract::parse(&std::fs::read_to_string(dir.join("nluca.contract")).unwrap()).unwrap();
        for k in 1..=6 {
            let src = std::fs::read_to_string(dir.join(format!("k{k:02}.wgsl"))).unwrap();
            let r = check_text(&src, &c).unwrap_or_else(|e| panic!("k{k:02}: {e}"));
            println!("k{k:02}: {r:?}");
            assert_eq!(r.write_sites, 1);
        }
        // A band offset the contract does not bound: gid + offset may wrap.
        let mut open = c.clone();
        open.params.remove("buffer_6_global[0]");
        let src = std::fs::read_to_string(dir.join("k01.wgsl")).unwrap();
        assert_eq!(rule(&src, &open), "store_index_may_wrap");
    }

    /// LAYER 7 IN ARENA MODE (2026-10-17, v24's index arenas): a kernel that
    /// keeps its own state in `index_u32` / `index_f32` at data-dependent
    /// indexes (a hashed bucket's count by atomicAdd, a plain store of a
    /// centre at the slot it got, a plain load of it), every index masked
    /// or clamped to the arena: strict turns it away, arena mode proves it.
    /// The same store unmasked, a mask past the arena's length, and a
    /// data-dependent store to `cands` (not an arena) are refused in both
    /// modes; a workgroup race stays a race; the arena line changes nothing
    /// in strict mode.
    #[test]
    fn arena_mode_proves_masked_state_and_keeps_every_other_binding_strict() {
        let pre = "struct Cfg { n_queries: u32, };
@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
";
        let k = |body: &str| format!("{pre}fn slot_of(h: u32, j: u32) -> u32 {{ return (h * 64u + j) & 1048575u; }}\n@compute @workgroup_size(64, 1, 1)\nfn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{\n    let h = bitcast<u32>(keys[gid.x]) & 1023u;\n{body}\n}}\n");
        let strict = Contract::parse("workgroups x 1 5000\nworkgroups y 1 1\nworkgroups z 1 1\narena index_u32 index_f32\nparam arrayLength(index_u32) 1048576 1048576\nparam arrayLength(index_f32) 1048576 1048576\n").unwrap();
        let arena = strict.with_arena_mode(true);
        let state = "    let slot = atomicAdd(&index_u32[h], 1u);\n    index_f32[slot_of(h, slot & 63u)] = keys[gid.x];\n    let back = index_f32[min(h * 64u + slot, arrayLength(&index_f32) - 1u)];\n    atomicStore(&index_u32[(1024u + h) & 1048575u], bitcast<u32>(back));\n    cands[gid.x] = slot;";
        assert_eq!(rule(&k(state), &strict), "write_sets_overlap");
        proved(&k(state), &arena);
        // The arena line alone changes nothing: strict without it, the same rule.
        let mut no_line = strict.clone();
        no_line.arena.clear();
        assert_eq!(rule(&k(state), &no_line), "write_sets_overlap");
        // Unmasked: no bound. Masked past the arena's length: out of bounds.
        assert_eq!(rule(&k("    index_f32[bitcast<u32>(keys[gid.x])] = 1.0;"), &arena), "arena_index_not_in_bounds");
        assert_eq!(rule(&k("    index_f32[bitcast<u32>(keys[gid.x])] = 1.0;"), &strict), "store_index_not_affine");
        assert_eq!(rule(&k("    index_f32[bitcast<u32>(keys[gid.x]) & 2097151u] = 1.0;"), &arena), "arena_index_not_in_bounds");
        // (An atomic at an unbounded index: strict proves it, so arena does
        // too, the binding under the strict rules.)
        proved(&k("    _ = atomicAdd(&index_u32[bitcast<u32>(keys[gid.x])], 1u);"), &strict);
        proved(&k("    _ = atomicAdd(&index_u32[bitcast<u32>(keys[gid.x])], 1u);"), &arena);
        // A binding the contract does not name stays strict.
        assert_eq!(rule(&k("    cands[h] = gid.x;"), &arena), "write_sets_overlap");
        // Workgroup memory stays strict.
        let wg = format!("{pre}var<workgroup> t: array<u32, 64>;\n@compute @workgroup_size(64, 1, 1)\nfn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{\n    t[lid.x] = gid.x;\n    cands[gid.x] = t[(lid.x + 1u) & 63u];\n}}\n");
        assert_eq!(rule(&wg, &arena), "workgroup_memory_race");
        // With no arrayLength in the contract a mask proves nothing; a clamp
        // to the binding's own length still does.
        let mut unsized_ = arena.clone();
        unsized_.params.retain(|k, _| !k.starts_with("arrayLength"));
        assert_eq!(rule(&k("    index_f32[slot_of(h, 0u)] = 1.0;"), &unsized_), "arena_index_not_in_bounds");
        proved(&k("    index_f32[min(h * 4096u, arrayLength(&index_f32) - 1u)] = 1.0;"), &unsized_);
    }

    /// STRICT UNCHANGED ON THE EXISTING SEEDS (v24 run 1's 73 raw kernels
    /// under the problem's contract with the arena line added): strict
    /// gives every seed the verdict it gives without the line; arena mode
    /// proves at least those strict proves. The counts are printed.
    #[test]
    fn strict_is_unchanged_by_the_arena_line_on_run_ones_seeds() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let base = std::fs::read_to_string(root.join("problems/v24_ingest/safety.contract")).unwrap();
        let plain = Contract::parse(&base).unwrap();
        let lined = Contract::parse(&format!("{base}\narena index_u32 index_f32\nparam arrayLength(index_u32) 8388608 8388608\nparam arrayLength(index_f32) 8388608 8388608\n")).unwrap();
        let arena = lined.with_arena_mode(true);
        let dir = root.join("logs/V24/run1/raw");
        let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())).flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
        files.sort();
        let (mut strict_ok, mut arena_ok) = (0usize, 0usize);
        let mut why: BTreeMap<&'static str, usize> = BTreeMap::new();
        for f in &files {
            let src = std::fs::read_to_string(f).unwrap();
            let (a, b) = (check_text(&src, &plain).map_err(|e| e.rule), check_text(&src, &lined).map_err(|e| e.rule));
            assert_eq!(a.as_ref().err(), b.as_ref().err(), "{}: strict changed by the arena line", f.display());
            let c = check_text(&src, &arena);
            if a.is_ok() {
                strict_ok += 1;
                assert!(c.is_ok(), "{}: strict proves it, arena does not: {:?}", f.display(), c.err());
            }
            match c {
                Ok(_) => arena_ok += 1,
                Err(e) => *why.entry(e.rule).or_default() += 1,
            }
        }
        println!("run 1's {} raw seeds at layer 7: strict proves {strict_ok}, arena {arena_ok}; arena's refusals by rule {why:?}", files.len());
    }

    /// A SEARCH THREAD'S OWN ROW OF cands (2026-10-17, the arena gate's 139
    /// store_index_not_affine, nearly all `cands[q * 64u + rank]`): a rank
    /// counted per candidate and stored under `rank < 64u` is a digit in
    /// [0, 63] there, so the cell is in the row of q; with q the thread's
    /// own (a thread per query) the row is proved its own in both modes, and
    /// with one thread of a workgroup per query too. Two threads of one
    /// workgroup writing their query's row at ranks the form cannot tell
    /// apart overlap (a limit of the proof, named), and a rank not tested
    /// below 64 has no form.
    #[test]
    fn a_rank_tested_below_the_row_width_stores_in_the_threads_own_row() {
        let pre = "struct Cfg { n_queries: u32, };
@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
";
        let k = |q: &str, test: &str| format!("{pre}@compute @workgroup_size(64, 1, 1)\nfn search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n    let q = {q};\n    var rank = 0u;\n    for (var j = 0u; j < 256u; j = j + 1u) {{\n        if (keys[j] > keys[q]) {{ rank = rank + 1u; }}\n    }}\n    if ({test}) {{\n        cands[q * 64u + rank] = j0(q);\n    }}\n}}\nfn j0(q: u32) -> u32 {{ return q + 1u; }}\n");
        let c = Contract::parse("workgroups x 1 2500\nworkgroups y 1 1\nworkgroups z 1 1\nparam cfg.n_queries 1 2500\n").unwrap();
        let arena = c.with_arena_mode(true);
        for m in [&c, &arena] {
            proved(&k("gid.x", "rank < 64u && q < cfg.n_queries"), m);
            proved(&k("wid.x", "lid.x == 0u && rank < 64u"), m);
            assert_eq!(rule(&k("wid.x", "rank < 64u"), m), "write_sets_overlap");
            assert_eq!(rule(&k("gid.x", "rank < 65u"), m), "write_sets_overlap");
            assert_eq!(rule(&k("gid.x", "q < cfg.n_queries"), m), "store_index_not_affine");
        }
    }
}
