//! THE GENERIC DISPATCH-PROGRAM HARNESS (`docs/PLAN_general_safety_layer.md`,
//! Revision 1 point 4, order step (e)): a fitness that is data, not code.
//! A problem is a directory; one harness runs any kernel against any
//! problem directory and returns the same [`Fitness`] the evolution
//! consumes (`gram::Fitness`), so a fit needs only a `--problem <dir>` flag.
//!
//! # The problem directory
//!
//! ```text
//! <problem>/problem.toml      the problem: schema, program, suites, budget
//! <problem>/cases/<suite>.toml one manifest per suite: its cases
//! <problem>/cases/<suite>/... the cases' raw little-endian .bin files
//!                             (or any directory a manifest's `root` names)
//! ```
//!
//! # `problem.toml`
//!
//! Every expression below is a string over the case's shape parameters (see
//! *Expressions*). Top-level keys:
//!
//! | key | meaning |
//! |---|---|
//! | `name`, `description` | the problem's name and one line |
//! | `work` | the work of one case (the cost model's unit, `gram::Meter`), e.g. `"d * d * m"` |
//! | `alias` | `"distinct"`: no buffer at two bindings of one dispatch (the only policy) |
//! | `speed_column` | how a speed-up becomes a fitness column: `"reciprocal"` (default, `gram.rs`'s `1 / s` clamped to 1: kernels at or below the reference's speed tie) or `"ranked"` (`1 / (1 + s)`: the reference 0.5, every speed-up ranks) |
//! | `derive` | `["name = expr", ...]`, evaluated in order after the parameters |
//! | `[params]` | `name = "expr"`: a parameter's default when a case does not give it (`limit = "binding_limit"`) |
//! | `[kernel]` | `entry` (the entry point), `grid_header` (read `// grid x y bx by [rows-x]` as `gram.rs` does; default false), `reference` (the shipped kernel's path, relative to the problem: the speed-ups' denominator and the accuracy baseline), `modes` (the modes a kernel may declare with a `// mode <name>` line; the program reads `mode_<name>`, 1 or 0) |
//! | `derive_run` | `["name = expr", ...]`, evaluated after the kernel is known, before the run steps |
//! | `[[binding]]` | the typed binding schema, one per `@group(0) @binding(n)`: `binding`, `name`, `kind` (`storage_read`, `storage_read_write`, `uniform`), `element` (`f32`, `u32`, `i32`: the storage array's scalar) or `fields` (a uniform's scalars, in order) |
//! | `[[input]]` | a host tensor of every case: `name`, `element`, `shape` (expressions), and `generate` (below) for a case that gives no file for it |
//! | `[[output]]` | a host tensor the program reads back and the suites compare: `name`, `element` (`f32`, `u32`, `i32`), `shape` (one or two expressions: the closeness columns read it as rows x cols), `expected_element` (an expected file's element, `f32` or `f64`; default the output's), `points` (comparison points: the tensor split into this many equal slices along its first axis, each judged by the norm; default `"1"`), `kind` and the tie rule (*Outputs* below) |
//! | `[oracle]` | where expected outputs come from (*The shipped kernel is the oracle* below): `mode` (`shipped`, the default, or `reference`), `on` (`executor`, the default, or `cpu`), `cache` (a directory relative to the problem, default `oracle`) |
//! | `setup` | steps run once per case when the context is built: the resident buffers (inputs no kernel may write) |
//! | `run` | steps run per evaluation of a kernel |
//! | `[[suite]]` | the cases and how they are judged (below) |
//! | `[budget]` | `base_ms`, `round_ms`, `timing_rounds` (`gram::BASE_BUDGET_MS`, `gram::ROUND_BUDGET_MS`, the paired timing rounds); optional `min_sample_ms` (100: a timing sample repeats the run until the reference kernel's GPU time reaches it); optional `deadline_factor` and `deadline_floor_ms` together: the fit watchdog's deadline, the factor times the reference kernel's measured evaluation, at least the floor and the gates' own maximum (`Harness::deadline_ms`) |
//!
//! ## Steps (the dispatch program)
//!
//! A step is a table with exactly one action; `let = ["name = expr"]`
//! binds values for it (per iteration in a loop) and `if = "expr"` skips it
//! when the expression is 0. Buffer names are templates: `"phi{ib}"`,
//! `"s{(t + 1) % 2}"` (a braced expression is replaced by its integer).
//!
//! | action | keys | meaning |
//! |---|---|---|
//! | loop | `for`, `count`, `body` | run `body` with `for` = 0..count |
//! | buffer | `buffer`, `element`, `elements`, and `fill = "zero"` or `from` (an input) with `offset` and `set = ["index = expr"]` | create a device buffer: cleared to zero (`clear_buffer`, initialisation and clearing), or a slice of an input with elements overwritten (a per-step uniform update in a storage word) |
//! | uniform | `uniform`, `of` (a uniform binding), `words` | create a uniform buffer, each word typed by the binding's `fields` |
//! | dispatch | `dispatch = { binding = "buffer", ... }`, `workgroups` (three expressions) | one compute pass of the kernel, timed by GPU timestamps; every binding of the schema bound; the dispatch geometry may read the kernel's `wg_x`, `wg_y`, `wg_z`, `bx`, `by`, `rows_x` |
//! | copy | `copy`, `to`, `elements`, `src_offset`, `dst_offset` | a copy edge between buffers (a ping-pong is two buffers named by the step: `"s{t % 2}"`; persistent state is a buffer made before the loop) |
//! | read | `read`, `into` (an output), `elements`, `src_offset`, `offset` | an output slice: the buffer's elements copied into the output tensor |
//! | sync | `sync = true` | end of a submission: the compute is submitted and waited for, then the segment's reads are copied back |
//! | require | `require` (expression), `because` | the case cannot run unless the expression holds: a plan error naming it (a full-matrix kernel at a d whose H exceeds the binding limit) |
//!
//! A read is made after its segment's compute submission completes, so a
//! buffer written again in the same segment after its read is a plan
//! error. A batch of cases (the constraint's chunks) runs as one segment,
//! its syncs dropped; a plan whose reads are not safe that way cannot be
//! batched. Setup buffers may be bound only at `storage_read` and
//! `uniform` bindings (no kernel can change a resident input).
//!
//! ## Suites
//!
//! `[[suite]]`: `name`, `role`, `cases` (the manifest), and by role:
//!
//! | role | judged | keys |
//! |---|---|---|
//! | `constraint` | every case must pass: the norm at every comparison point within `pass`; run in chunks of `chunk` cases, one submission a chunk; `staged = true`: cheapest work first, each later chunk predicted as the kernel's time so far over the shipped kernel's on the same chunks times the shipped kernel's on it, not dispatched past `base_ms`, stopped at the first failing chunk, a timing build measured over the slow limit scored slow | `norm`, `pass` (expression), `chunk`, `staged` |
//! | `rung` | timed, gated, not judged (a step of the cost model between the cases and the pairs) | |
//! | `accuracy` | the norm against the reference kernel's own value: fails over `max(slack * reference, 1)`; the split's column is `min(1, worst / reference / slack)` | `split` (0 train, 1 validation, 2 extrapolation), `norm`, `tau`, `slack` |
//! | `timing` | three cases in all: the speed-up against the reference kernel timed beside the kernel, the median of `timing_rounds` rounds, each the two kernels' samples in ABBA order (`Harness::paired_speedup`) | |
//!
//! Norms (the tolerance and its norm): `mean_abs_over_max` (the mean over
//! cells of the error over the reference's largest finite magnitude, a
//! non-finite cell 0 when the classes agree and 1 when not:
//! `probes::distance`), `cauchy_schwarz` (the largest
//! `|out - ref| / (tau sqrt(ref[r][r] ref[c][c]))` of a square output, a
//! non-finite mismatch counted: `gram::check_max`), `max_abs_over_rms` (the
//! largest `|out - ref|` over the RMS of the reference, infinite with a
//! non-finite output where the reference is finite).
//!
//! # The shipped kernel is the oracle
//!
//! Differential testing against the program as shipped, so a new problem
//! needs no hand-made reference: a case of a judged suite (constraint,
//! accuracy) that gives no expected file for an output gets the reference
//! (shipped) kernel's own output on the case's inputs. The harness computes
//! it when it starts ([`Harness::new`]), on its own executor (the device the
//! evaluation runs on) or, with `on = "cpu"`, on the CPU executor
//! ([`CpuExecutor`], `cpu.rs`: an error naming why when it cannot run the
//! kernel), and caches it as raw bytes in `cache/<key>.<output>.bin`, the key
//! a hash over the shipped text, the executor ([`Executor::oracle_tag`]), the
//! case's name, parameters and input bytes, and the binding limit. With
//! `mode = "reference"` every judged case must carry its files (the Gram,
//! nluca: their owners' f64 references); a case that does is never re-run.
//!
//! ## Generated inputs
//!
//! A case that gives no file for an input draws it from the input's
//! `generate = { dist = ..., ... }`, deterministic in the case (its `seed`
//! parameter, else its name) and the input's name; every value an
//! expression over the case's parameters:
//!
//! | `dist` | keys | values |
//! |---|---|---|
//! | `uniform` | `lo`, `hi` | uniform in [lo, hi) |
//! | `normal` | `mean`, `std` | normal |
//! | `unit_rows` | `row` | rows of `row` normal draws scaled to unit length (f32) |
//! | `index` | `lo`, `hi` | integers in [lo, hi) (an index buffer within its bounds) |
//! | `constant` | `value` | one value |
//!
//! ## Outputs
//!
//! An output is compared by its `kind`: `float` (the default for f32) by the
//! suite's norm and `pass`; `exact` (the default for u32 and i32) element
//! for element, the suite's value the fraction that differ (pass 0 unless
//! the suite says otherwise); `argmax` (an index of the best of `classes`
//! scores) by its `tie` rule: `exact`, or `margin`: an index that differs is
//! right when it lies in [0, classes) and its score is within `margin` of the
//! expected index's, the scores the `score` expression over the case's
//! inputs in f64 with `i` the element and `j` the index (a near tie: either
//! index is the argmax under rounding). Norms and `pass` are for float
//! outputs; an exact or argmax suite needs neither ([`judge`]).
//!
//! # Case manifests
//!
//! ```toml
//! root = "constraint"            # relative to the manifest, or absolute
//! [[case]]
//! name = "A t0 c0 v1"
//! label = "d 32 m 16"            # optional, names the case in a detail
//! dir = "w4096_k1"               # optional, under root
//! params = { d = 32, m = 16 }
//! inputs = { phi = "0000.phi.bin" }
//! expected = { h = ["0000.h.bin"] }   # several files are concatenated
//! ```
//!
//! # Expressions
//!
//! Numbers, names, `+ - * / %`, comparisons (`< <= > >= == !=`, 1 or 0),
//! parentheses and `min`, `max`, `floor`, `ceil`, `sqrt`, `abs`,
//! `floor_div`, `ceil_div` (exact on integers), `select(c, a, b)` (`a` when
//! `c` is not 0). `eps_f32` is `f32::EPSILON`; `binding_limit` the
//! device's per-binding byte limit (`gram::effective_buffer_limit`). An
//! argmax output's score also reads the case's inputs: `t[e]` is element e
//! of input t, and `sum(k, n, e)` the sum of e with k = 0..n.
//!
//! # What runs it
//!
//! [`Harness`] over an [`Executor`]: [`DeviceExecutor`] (feature `gpu`;
//! wgpu on the device paths of `gram.rs`, `create_shader_module` with
//! runtime bounds checks on, every wait `gram::wait_gpu`), or
//! [`HostExecutor`], which runs the same plans on the CPU with a Rust model
//! of a kernel (the tests: the program, the scoring and the timing code
//! paths without a device), or [`CpuExecutor`], which runs the kernel's own
//! WGSL on the CPU executor (`cpu.rs`: no model, the CPU's wall clock).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::wgsl::closeness;
use crate::wgsl::gram::{fitness_columns, gate, median, shipped_min, slow_limit_ms, speedup, truenorth, Build, Fitness, Meter, Split};

// ------------------------------------------------------------------
// Expressions
// ------------------------------------------------------------------

/// A parsed expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
    /// An element of a host tensor, `points[i * dp + p]` (an output's tie
    /// rule reads the case's inputs; see *Expressions*).
    Index(String, Box<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// The values an expression reads.
pub type Env = BTreeMap<String, f64>;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Name(String),
    Sym(&'static str),
}

fn tokens(s: &str) -> Result<Vec<Tok>, String> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && i + 1 < b.len() && (b[i + 1] as char).is_ascii_digit()) {
            let start = i;
            while i < b.len() && ((b[i] as char).is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                i += 1;
                if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                    i += 1;
                }
                while i < b.len() && (b[i] as char).is_ascii_digit() {
                    i += 1;
                }
            }
            out.push(Tok::Num(s[start..i].parse().map_err(|e| format!("expression {s:?}: {e}"))?));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(Tok::Name(s[start..i].to_string()));
        } else {
            let two = if i + 1 < b.len() { &s[i..i + 2] } else { "" };
            let sym = match two {
                "<=" => Some("<="),
                ">=" => Some(">="),
                "==" => Some("=="),
                "!=" => Some("!="),
                _ => None,
            };
            if let Some(sym) = sym {
                out.push(Tok::Sym(sym));
                i += 2;
                continue;
            }
            let one = match c {
                '+' => "+",
                '-' => "-",
                '*' => "*",
                '/' => "/",
                '%' => "%",
                '(' => "(",
                ')' => ")",
                '[' => "[",
                ']' => "]",
                ',' => ",",
                '<' => "<",
                '>' => ">",
                other => return Err(format!("expression {s:?}: unexpected {other:?}")),
            };
            out.push(Tok::Sym(one));
            i += 1;
        }
    }
    Ok(out)
}

struct Parser<'a> {
    toks: Vec<Tok>,
    at: usize,
    text: &'a str,
}

impl Parser<'_> {
    fn peek_sym(&self) -> Option<&'static str> {
        match self.toks.get(self.at) {
            Some(Tok::Sym(s)) => Some(s),
            _ => None,
        }
    }

    fn expect(&mut self, sym: &str) -> Result<(), String> {
        if self.peek_sym() == Some(sym) {
            self.at += 1;
            Ok(())
        } else {
            Err(format!("expression {:?}: expected {sym:?} at token {}", self.text, self.at))
        }
    }

    fn cmp(&mut self) -> Result<Expr, String> {
        let a = self.add()?;
        let op = match self.peek_sym() {
            Some("<") => Op::Lt,
            Some("<=") => Op::Le,
            Some(">") => Op::Gt,
            Some(">=") => Op::Ge,
            Some("==") => Op::Eq,
            Some("!=") => Op::Ne,
            _ => return Ok(a),
        };
        self.at += 1;
        let b = self.add()?;
        Ok(Expr::Bin(op, Box::new(a), Box::new(b)))
    }

    fn add(&mut self) -> Result<Expr, String> {
        let mut a = self.mul()?;
        loop {
            let op = match self.peek_sym() {
                Some("+") => Op::Add,
                Some("-") => Op::Sub,
                _ => return Ok(a),
            };
            self.at += 1;
            let b = self.mul()?;
            a = Expr::Bin(op, Box::new(a), Box::new(b));
        }
    }

    fn mul(&mut self) -> Result<Expr, String> {
        let mut a = self.unary()?;
        loop {
            let op = match self.peek_sym() {
                Some("*") => Op::Mul,
                Some("/") => Op::Div,
                Some("%") => Op::Rem,
                _ => return Ok(a),
            };
            self.at += 1;
            let b = self.unary()?;
            a = Expr::Bin(op, Box::new(a), Box::new(b));
        }
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if self.peek_sym() == Some("-") {
            self.at += 1;
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<Expr, String> {
        match self.toks.get(self.at).cloned() {
            Some(Tok::Num(v)) => {
                self.at += 1;
                Ok(Expr::Num(v))
            }
            Some(Tok::Name(n)) => {
                self.at += 1;
                if self.peek_sym() == Some("(") {
                    self.at += 1;
                    let mut args = Vec::new();
                    if self.peek_sym() != Some(")") {
                        loop {
                            args.push(self.cmp()?);
                            if self.peek_sym() == Some(",") {
                                self.at += 1;
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(")")?;
                    Ok(Expr::Call(n, args))
                } else if self.peek_sym() == Some("[") {
                    self.at += 1;
                    let i = self.cmp()?;
                    self.expect("]")?;
                    Ok(Expr::Index(n, Box::new(i)))
                } else {
                    Ok(Expr::Var(n))
                }
            }
            Some(Tok::Sym("(")) => {
                self.at += 1;
                let e = self.cmp()?;
                self.expect(")")?;
                Ok(e)
            }
            other => Err(format!("expression {:?}: unexpected {other:?}", self.text)),
        }
    }
}

/// Parse an expression.
pub fn parse_expr(s: &str) -> Result<Expr, String> {
    let mut p = Parser { toks: tokens(s)?, at: 0, text: s };
    let e = p.cmp()?;
    if p.at != p.toks.len() {
        return Err(format!("expression {s:?}: trailing input at token {}", p.at));
    }
    Ok(e)
}

/// An integral, non-negative value as an exact integer (at most 2^64).
fn integral(x: f64, what: &str) -> Result<u128, String> {
    if x.is_finite() && x >= 0.0 && x.fract() == 0.0 && x <= 18_446_744_073_709_551_616.0 {
        Ok(x as u128)
    } else {
        Err(format!("{what}: {x} is not a non-negative integer"))
    }
}

/// Host tensors an expression may index, by name, as f64 values.
pub type Tensors = BTreeMap<String, Vec<f64>>;

impl Expr {
    pub fn eval(&self, env: &Env) -> Result<f64, String> {
        self.eval_in(env, &Tensors::new())
    }

    /// [`Expr::eval`] with host tensors: `t[i]` reads element i of tensor t
    /// (an error past its end), `sum(k, n, body)` is the sum of `body` with
    /// `k` = 0..n.
    pub fn eval_in(&self, env: &Env, tensors: &Tensors) -> Result<f64, String> {
        Ok(match self {
            Expr::Num(v) => *v,
            Expr::Var(n) => match n.as_str() {
                "eps_f32" => f64::from(f32::EPSILON),
                _ => *env.get(n).ok_or_else(|| format!("unknown name {n:?}"))?,
            },
            Expr::Neg(a) => -a.eval_in(env, tensors)?,
            Expr::Index(t, i) => {
                let v = tensors.get(t).ok_or_else(|| format!("no tensor {t:?}"))?;
                let i = integral(i.eval_in(env, tensors)?, t)?;
                *usize::try_from(i).ok().and_then(|i| v.get(i)).ok_or_else(|| format!("{t}[{i}]: past its {} elements", v.len()))?
            }
            Expr::Call(f, args) if f == "sum" => {
                let [Expr::Var(k), n, body] = args.as_slice() else { return Err("sum takes (name, count, body)".into()) };
                let n = integral(n.eval_in(env, tensors)?, "sum")?;
                let mut e = env.clone();
                let mut acc = 0.0f64;
                for j in 0..n {
                    e.insert(k.clone(), j as f64);
                    acc += body.eval_in(&e, tensors)?;
                }
                acc
            }
            Expr::Bin(op, a, b) => {
                let (x, y) = (a.eval_in(env, tensors)?, b.eval_in(env, tensors)?);
                let truth = |c: bool| if c { 1.0 } else { 0.0 };
                match op {
                    Op::Add => x + y,
                    Op::Sub => x - y,
                    Op::Mul => x * y,
                    Op::Div => x / y,
                    Op::Rem => {
                        let (p, q) = (integral(x, "%")?, integral(y, "%")?);
                        if q == 0 {
                            return Err("% by zero".into());
                        }
                        (p % q) as f64
                    }
                    Op::Lt => truth(x < y),
                    Op::Le => truth(x <= y),
                    Op::Gt => truth(x > y),
                    Op::Ge => truth(x >= y),
                    Op::Eq => truth(x == y),
                    Op::Ne => truth(x != y),
                }
            }
            Expr::Call(f, args) => {
                let v: Vec<f64> = args.iter().map(|a| a.eval_in(env, tensors)).collect::<Result<_, _>>()?;
                let arity = |n: usize| if v.len() == n { Ok(()) } else { Err(format!("{f} takes {n} arguments, not {}", v.len())) };
                match f.as_str() {
                    "min" => {
                        arity(2)?;
                        v[0].min(v[1])
                    }
                    "max" => {
                        arity(2)?;
                        v[0].max(v[1])
                    }
                    "floor" => {
                        arity(1)?;
                        v[0].floor()
                    }
                    "ceil" => {
                        arity(1)?;
                        v[0].ceil()
                    }
                    "sqrt" => {
                        arity(1)?;
                        v[0].sqrt()
                    }
                    "abs" => {
                        arity(1)?;
                        v[0].abs()
                    }
                    "floor_div" | "ceil_div" => {
                        arity(2)?;
                        let (p, q) = (integral(v[0], f)?, integral(v[1], f)?);
                        if q == 0 {
                            return Err(format!("{f} by zero"));
                        }
                        (if f == "floor_div" { p / q } else { p.div_ceil(q) }) as f64
                    }
                    "select" => {
                        arity(3)?;
                        if v[0] != 0.0 { v[1] } else { v[2] }
                    }
                    other => return Err(format!("unknown function {other:?}")),
                }
            }
        })
    }
}

/// Parse and evaluate.
pub fn eval(s: &str, env: &Env) -> Result<f64, String> {
    parse_expr(s)?.eval(env).map_err(|e| format!("{s:?}: {e}"))
}

/// Evaluate to an exact unsigned integer.
pub fn eval_int(s: &str, env: &Env) -> Result<u64, String> {
    let v = integral(eval(s, env)?, s)?;
    u64::try_from(v).map_err(|_| format!("{s:?}: {v} exceeds 64 bits"))
}

/// A name template: every `{expr}` replaced by its integer value.
pub fn template(s: &str, env: &Env) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = s;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let close = rest[open..].find('}').ok_or_else(|| format!("template {s:?}: unclosed brace"))? + open;
        out.push_str(&eval_int(&rest[open + 1..close], env)?.to_string());
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Bind `["name = expr", ...]` into `env`, in order.
pub fn bind_lets(lets: &[String], env: &mut Env) -> Result<(), String> {
    for l in lets {
        let (name, expr) = l.split_once('=').ok_or_else(|| format!("let {l:?}: no '='"))?;
        let name = name.trim();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || name.starts_with(|c: char| c.is_ascii_digit()) {
            return Err(format!("let {l:?}: {name:?} is not a name"));
        }
        let v = eval(expr.trim(), env)?;
        env.insert(name.to_string(), v);
    }
    Ok(())
}

// ------------------------------------------------------------------
// The schema
// ------------------------------------------------------------------

/// A scalar element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Elem {
    F32,
    U32,
    I32,
    F64,
}

impl Elem {
    pub fn size(self) -> u64 {
        if self == Elem::F64 { 8 } else { 4 }
    }

    /// One value as little-endian bytes.
    fn encode(self, v: f64) -> Result<Vec<u8>, String> {
        Ok(match self {
            Elem::F32 => (v as f32).to_le_bytes().to_vec(),
            Elem::F64 => v.to_le_bytes().to_vec(),
            Elem::U32 => {
                let i = integral(v, "u32")?;
                u32::try_from(i).map_err(|_| format!("{v} exceeds u32"))?.to_le_bytes().to_vec()
            }
            Elem::I32 => {
                if v.fract() != 0.0 || v < f64::from(i32::MIN) || v > f64::from(i32::MAX) {
                    return Err(format!("{v} is not an i32"));
                }
                (v as i32).to_le_bytes().to_vec()
            }
        })
    }

    /// Little-endian bytes as f64 values.
    pub fn decode(self, bytes: &[u8]) -> Vec<f64> {
        match self {
            Elem::F64 => bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect(),
            Elem::F32 => bytes.chunks_exact(4).map(|c| f64::from(f32::from_le_bytes(c.try_into().expect("4 bytes")))).collect(),
            Elem::U32 => bytes.chunks_exact(4).map(|c| f64::from(u32::from_le_bytes(c.try_into().expect("4 bytes")))).collect(),
            Elem::I32 => bytes.chunks_exact(4).map(|c| f64::from(i32::from_le_bytes(c.try_into().expect("4 bytes")))).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingKind {
    StorageRead,
    StorageReadWrite,
    Uniform,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BindingSpec {
    #[serde(default)]
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub kind: BindingKind,
    pub element: Option<Elem>,
    pub fields: Option<Vec<Elem>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelSpec {
    pub entry: String,
    #[serde(default)]
    pub grid_header: bool,
    pub reference: String,
    /// The modes a kernel may declare by a `// mode <name>` line; the
    /// program reads `mode_<name>` (1 when declared, else 0).
    #[serde(default)]
    pub modes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSpec {
    pub name: String,
    pub element: Elem,
    pub shape: Vec<String>,
    /// How a case that gives no file for this input generates it.
    pub generate: Option<Generate>,
}

/// A distribution an input is drawn from (see *Generated inputs*).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dist {
    Uniform,
    Normal,
    UnitRows,
    Index,
    Constant,
}

/// `generate = { dist = ..., ... }` of an input: every value an expression
/// over the case's parameters.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Generate {
    pub dist: Dist,
    pub lo: Option<String>,
    pub hi: Option<String>,
    pub mean: Option<String>,
    pub std: Option<String>,
    /// `unit_rows`: the length of a row.
    pub row: Option<String>,
    /// `constant`: the value.
    pub value: Option<String>,
}

/// How an output is compared with its expected value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    /// f32 values under the suite's norm and tolerance.
    Float,
    /// Integers or indices: equal, element for element.
    Exact,
    /// The index of the best of `classes` scores, judged by the tie rule.
    Argmax,
}

/// An argmax output's tie rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tie {
    /// The index must equal the expected one.
    Exact,
    /// Another index is a correct answer when its score is within `margin`
    /// of the expected index's (a near tie, either one the argmax under
    /// rounding); both scores from the `score` expression in f64.
    Margin,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSpec {
    pub name: String,
    pub element: Elem,
    pub shape: Vec<String>,
    /// The element of an expected file (default: the output's own).
    pub expected_element: Option<Elem>,
    #[serde(default = "one")]
    pub points: String,
    /// `float` (default for f32), `exact` (default for u32 and i32) or `argmax`.
    pub kind: Option<OutputKind>,
    /// An argmax output's tie rule, and for `margin` its score of index `j`
    /// for element `i`, its margin and its number of classes (expressions).
    pub tie: Option<Tie>,
    pub score: Option<String>,
    pub margin: Option<String>,
    pub classes: Option<String>,
}

impl OutputSpec {
    pub fn kind(&self) -> OutputKind {
        self.kind.unwrap_or(if self.element == Elem::F32 { OutputKind::Float } else { OutputKind::Exact })
    }

    pub fn expected_elem(&self) -> Elem {
        self.expected_element.unwrap_or(self.element)
    }
}

/// Where a case's expected outputs come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OracleMode {
    /// THE SHIPPED KERNEL IS THE ORACLE (the default): a case that gives no
    /// expected file gets the shipped kernel's own outputs on its inputs.
    #[default]
    Shipped,
    /// Every judged case gives its expected files (a problem owner's reference).
    Reference,
}

/// What runs the shipped kernel for the oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OracleOn {
    /// The harness's own executor (the device, at evaluation start).
    #[default]
    Executor,
    /// The CPU executor (`cpu.rs`, [`CpuExecutor`]); a kernel it cannot run
    /// is an error naming why.
    Cpu,
}

/// `[oracle]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Oracle {
    #[serde(default)]
    pub mode: OracleMode,
    #[serde(default)]
    pub on: OracleOn,
    /// The cache directory, relative to the problem (default `oracle`).
    #[serde(default = "oracle_dir")]
    pub cache: String,
}

impl Default for Oracle {
    fn default() -> Oracle {
        Oracle { mode: OracleMode::Shipped, on: OracleOn::Executor, cache: oracle_dir() }
    }
}

fn oracle_dir() -> String {
    "oracle".into()
}

fn one() -> String {
    "1".into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Alias {
    Distinct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Fill {
    Zero,
}

/// One step of the dispatch program (see the module header).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    #[serde(rename = "let", default)]
    pub lets: Vec<String>,
    #[serde(rename = "if")]
    pub when: Option<String>,
    #[serde(rename = "for")]
    pub var: Option<String>,
    pub count: Option<String>,
    #[serde(default)]
    pub body: Vec<Step>,
    pub buffer: Option<String>,
    pub element: Option<Elem>,
    pub elements: Option<String>,
    pub fill: Option<Fill>,
    pub from: Option<String>,
    pub offset: Option<String>,
    #[serde(default)]
    pub set: Vec<String>,
    pub uniform: Option<String>,
    pub of: Option<String>,
    pub words: Option<Vec<String>>,
    pub dispatch: Option<BTreeMap<String, String>>,
    pub workgroups: Option<Vec<String>>,
    pub copy: Option<String>,
    pub to: Option<String>,
    pub src_offset: Option<String>,
    pub dst_offset: Option<String>,
    pub read: Option<String>,
    pub into: Option<String>,
    pub sync: Option<bool>,
    pub require: Option<String>,
    pub because: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Constraint,
    Rung,
    Accuracy,
    Timing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Norm {
    MeanAbsOverMax,
    CauchySchwarz,
    MaxAbsOverRms,
}

impl Norm {
    fn unit(self) -> &'static str {
        match self {
            Norm::CauchySchwarz => " tau",
            Norm::MeanAbsOverMax | Norm::MaxAbsOverRms => "",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteSpec {
    pub name: String,
    pub role: Role,
    pub cases: String,
    pub split: Option<usize>,
    pub norm: Option<Norm>,
    pub pass: Option<String>,
    pub tau: Option<String>,
    pub slack: Option<f64>,
    pub chunk: Option<usize>,
    pub output: Option<String>,
    /// A constraint suite run STAGED: its cases cheapest work first, every
    /// chunk after the first predicted from the rate measured so far and
    /// not dispatched when it would carry the cases past the base budget
    /// (a "slow" outcome, `gram::gate`), and the suite stopped at the first
    /// chunk with a failing case. Absent: the manifest's order, every chunk
    /// run (the Gram's constraint as `gram.rs` runs it).
    #[serde(default)]
    pub staged: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub base_ms: f64,
    pub round_ms: f64,
    #[serde(default = "one_round")]
    pub timing_rounds: usize,
    /// The least GPU ms of one timing sample: a sample repeats the run
    /// until the reference kernel's time reaches it (a 64-step run of
    /// 1,000 rows is about 1 ms, mostly per-dispatch overhead). Measured,
    /// nluca at 100,000 rows, the shipped kernel scored against itself:
    /// HFF 0.474 to 0.542 with 20 ms samples, 0.497 to 0.508 with 100.
    #[serde(default = "hundred_ms")]
    pub min_sample_ms: f64,
    /// THE EVALUATION'S DEADLINE (the fit's watchdog, for a true runaway
    /// only): `deadline_factor` times the reference kernel's measured
    /// evaluation, at least `deadline_floor_ms`. Both or neither; absent,
    /// the caller's own rule (`Harness::deadline_ms` is None).
    pub deadline_factor: Option<f64>,
    pub deadline_floor_ms: Option<f64>,
}

fn one_round() -> usize {
    1
}

fn hundred_ms() -> f64 {
    100.0
}

/// THE THREE SPEED BANDS of a problem's timing cases (by index): with three
/// or fewer the cases in their manifest order (a band with none takes the
/// nearest: one case fills all three, two give the first, the first, the
/// second); with more, the cases sorted by work (a stable sort) and cut into
/// three runs of nearly equal size, smallest work first.
pub fn timing_bands(works: &[f64]) -> [Vec<usize>; 3] {
    let n = works.len();
    let order: Vec<usize> = if n > 3 {
        let mut o: Vec<usize> = (0..n).collect();
        o.sort_by(|a, b| works[*a].total_cmp(&works[*b]));
        o
    } else {
        (0..n).collect()
    };
    std::array::from_fn(|b| {
        let (lo, hi) = (b * n / 3, (b + 1) * n / 3);
        if lo < hi {
            order[lo..hi].to_vec()
        } else {
            vec![order[lo.min(n.saturating_sub(1))]]
        }
    })
}

/// Each band's largest value of a per-case list.
fn band_max(bands: &[Vec<usize>; 3], per: &[f64]) -> [f64; 3] {
    std::array::from_fn(|b| bands[b].iter().filter_map(|&i| per.get(i)).copied().fold(0.0, f64::max))
}

/// Each band's geometric mean of a per-case list of speed-ups.
fn band_geomean(bands: &[Vec<usize>; 3], per: &[f64]) -> [f64; 3] {
    std::array::from_fn(|b| {
        let v: Vec<f64> = bands[b].iter().filter_map(|&i| per.get(i)).copied().collect();
        match v.as_slice() {
            [] => 0.0,
            // One case: its own value, bit for bit (three cases score as before).
            [x] => *x,
            _ => (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp(),
        }
    })
}

/// HOW A SPEED-UP BECOMES A FITNESS COLUMN (in [0, 1], 0 best).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeedColumn {
    /// `1 / speed-up` clamped to [0, 1], `gram.rs`'s: every kernel at or
    /// below the reference's speed scores 1, so slower kernels tie (the
    /// Gram's candidates were faster than shipped; nluca_s2's 97 feasible
    /// kernels, all slower, all scored HFF π/3).
    #[default]
    Reciprocal,
    /// `1 / (1 + speed-up)`: the reference's speed 0.5, faster toward 0,
    /// slower toward 1, smooth and strictly decreasing, so every speed-up
    /// ranks.
    Ranked,
}

impl SpeedColumn {
    pub fn column(self, speedup: f64) -> f64 {
        match self {
            SpeedColumn::Reciprocal => (1.0 / speedup).clamp(0.0, 1.0),
            SpeedColumn::Ranked => (1.0 / (1.0 + speedup.max(0.0))).clamp(0.0, 1.0),
        }
    }
}

/// `problem.toml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProblemSpec {
    pub name: String,
    pub description: Option<String>,
    pub work: String,
    pub alias: Alias,
    #[serde(default)]
    pub derive: Vec<String>,
    /// Derived after the kernel is known (its `wg_x`.., `bx`, `by`,
    /// `rows_x`, `mode_<name>`), before the run steps.
    #[serde(default)]
    pub derive_run: Vec<String>,
    #[serde(default)]
    pub params: BTreeMap<String, String>,
    pub kernel: KernelSpec,
    pub binding: Vec<BindingSpec>,
    pub input: Vec<TensorSpec>,
    pub output: Vec<OutputSpec>,
    #[serde(default)]
    pub setup: Vec<Step>,
    pub run: Vec<Step>,
    pub suite: Vec<SuiteSpec>,
    pub budget: Budget,
    /// How a speed-up becomes a fitness column (`speed_column`).
    #[serde(default)]
    pub speed_column: SpeedColumn,
    /// Where the expected outputs come from (`[oracle]`; default: the shipped kernel).
    #[serde(default)]
    pub oracle: Oracle,
}

/// One case of a manifest.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaseSpec {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dir: Option<String>,
    pub params: BTreeMap<String, f64>,
    #[serde(default)]
    pub inputs: BTreeMap<String, String>,
    #[serde(default)]
    pub expected: BTreeMap<String, Vec<String>>,
}

/// A suite's manifest, `cases/<suite>.toml`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
    pub case: Vec<CaseSpec>,
}

/// A case with its files resolved and its base environment (the
/// parameters, their defaults, `binding_limit` and the derived values).
#[derive(Debug, Clone)]
pub struct Case {
    pub name: String,
    /// The case in a detail ("d 512 m 2048"); the name when not given.
    pub label: String,
    pub params: BTreeMap<String, f64>,
    pub inputs: BTreeMap<String, PathBuf>,
    pub expected: BTreeMap<String, Vec<PathBuf>>,
}

/// A suite with its cases.
#[derive(Debug, Clone)]
pub struct Suite {
    pub spec: SuiteSpec,
    pub cases: Vec<Case>,
}

/// A problem directory, read and checked.
#[derive(Debug, Clone)]
pub struct Problem {
    pub dir: PathBuf,
    pub spec: ProblemSpec,
    pub suites: Vec<Suite>,
    /// The reference (shipped) kernel's text.
    pub reference: String,
}

/// A case's host data: the inputs' bytes, the expected outputs as f64.
#[derive(Debug, Clone, Default)]
pub struct CaseData {
    pub inputs: BTreeMap<String, Vec<u8>>,
    pub expected: BTreeMap<String, Vec<f64>>,
}

/// FNV-1a over bytes: the stable hash of the oracle's cache keys and seeds.
pub fn fnv(bytes: &[u8], mut h: u64) -> u64 {
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

/// A generated input's seed: the case's `seed` parameter, else its name,
/// mixed with the input's name (two inputs of a case differ).
pub fn case_seed(case: &Case, input: &str) -> u64 {
    let base = match case.params.get("seed") {
        Some(v) => fnv(&v.to_le_bytes(), FNV0),
        None => fnv(case.name.as_bytes(), FNV0),
    };
    fnv(input.as_bytes(), base ^ 0x9e37_79b9_7f4a_7c15)
}

/// splitmix64.
fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A uniform f64 in [0, 1).
fn unit(state: &mut u64) -> f64 {
    (mix(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// A standard normal (Box-Muller).
fn normal(state: &mut u64) -> f64 {
    let u = 1.0 - unit(state);
    let v = unit(state);
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// GENERATED INPUTS: `n` values of the input's element drawn from its
/// declared distribution, deterministic in `seed`. `uniform` in [lo, hi);
/// `normal` (mean, std); `unit_rows` normal rows of `row` values each scaled
/// to unit length in f64 (a row of zeros stays zero); `index` integers in
/// [lo, hi), within the bound an index input declares; `constant` one value.
pub fn generate(i: &TensorSpec, g: &Generate, n: u64, seed: u64, env: &Env) -> Result<Vec<u8>, String> {
    let ex = |e: &Option<String>| -> Result<f64, String> { eval(e.as_deref().ok_or("missing")?, env) };
    let mut state = seed;
    let n = usize::try_from(n).map_err(|_| format!("{n} values"))?;
    // The values are encoded as they are drawn (a production-sized input is
    // gigabytes: no f64 copy of it is held).
    let mut out = Vec::with_capacity(n * i.element.size() as usize);
    let put = |v: f64, out: &mut Vec<u8>| -> Result<(), String> {
        out.extend(i.element.encode(v)?);
        Ok(())
    };
    match g.dist {
        Dist::Uniform => {
            let (lo, hi) = (ex(&g.lo)?, ex(&g.hi)?);
            if lo >= hi || !lo.is_finite() || !hi.is_finite() {
                return Err(format!("uniform [{lo}, {hi}) is empty"));
            }
            for _ in 0..n {
                put(lo + (hi - lo) * unit(&mut state), &mut out)?;
            }
        }
        Dist::Normal => {
            let (m, sd) = (ex(&g.mean)?, ex(&g.std)?);
            for _ in 0..n {
                put(m + sd * normal(&mut state), &mut out)?;
            }
        }
        Dist::UnitRows => {
            let row = usize::try_from(integral(ex(&g.row)?, "row")?).map_err(|e| e.to_string())?;
            if row == 0 || !n.is_multiple_of(row) {
                return Err(format!("{n} values are not whole rows of {row}"));
            }
            let mut r = vec![0.0f64; row];
            for _ in 0..n / row {
                r.iter_mut().for_each(|x| *x = normal(&mut state));
                let norm = r.iter().map(|x| x * x).sum::<f64>().sqrt();
                for x in &r {
                    put(if norm > 0.0 { x / norm } else { *x }, &mut out)?;
                }
            }
        }
        Dist::Index => {
            let (lo, hi) = (ex(&g.lo)?, ex(&g.hi)?);
            if lo.fract() != 0.0 || hi.fract() != 0.0 || lo >= hi {
                return Err(format!("index [{lo}, {hi}) is not a non-empty integer range"));
            }
            let span = (hi - lo) as u64;
            for _ in 0..n {
                put(lo + (mix(&mut state) % span) as f64, &mut out)?;
            }
        }
        Dist::Constant => {
            let v = ex(&g.value)?;
            for _ in 0..n {
                put(v, &mut out)?;
            }
        }
    }
    Ok(out)
}

fn read_toml<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

impl Problem {
    /// Read `dir/problem.toml` and every suite's manifest, and check the
    /// schema (no case data is read: [`Problem::case_data`] does).
    pub fn load(dir: &Path) -> Result<Problem, String> {
        let spec: ProblemSpec = read_toml(&dir.join("problem.toml"))?;
        let reference_path = dir.join(&spec.kernel.reference);
        let reference = std::fs::read_to_string(&reference_path).map_err(|e| format!("{}: {e}", reference_path.display()))?;
        let mut suites = Vec::new();
        for s in &spec.suite {
            let mpath = dir.join(&s.cases);
            let manifest: Manifest = read_toml(&mpath)?;
            let mdir = mpath.parent().unwrap_or(dir).to_path_buf();
            let root = match &manifest.root {
                Some(r) if Path::new(r).is_absolute() => PathBuf::from(r),
                Some(r) => mdir.join(r),
                None => mdir,
            };
            let cases = manifest
                .case
                .into_iter()
                .map(|c| {
                    let base = match &c.dir {
                        Some(d) => root.join(d),
                        None => root.clone(),
                    };
                    Case {
                        label: c.label.clone().unwrap_or_else(|| c.name.clone()),
                        name: c.name,
                        params: c.params,
                        inputs: c.inputs.into_iter().map(|(k, v)| (k, base.join(v))).collect(),
                        expected: c.expected.into_iter().map(|(k, v)| (k, v.into_iter().map(|f| base.join(f)).collect())).collect(),
                    }
                })
                .collect();
            suites.push(Suite { spec: s.clone(), cases });
        }
        let p = Problem { dir: dir.to_path_buf(), spec, suites, reference };
        p.check()?;
        Ok(p)
    }

    fn check(&self) -> Result<(), String> {
        let s = &self.spec;
        let name = &s.name;
        let mut seen = std::collections::BTreeSet::new();
        for b in &s.binding {
            if b.group != 0 {
                return Err(format!("{name}: binding {} is in group {}; the harness binds group 0", b.name, b.group));
            }
            if !seen.insert(b.binding) {
                return Err(format!("{name}: binding {} declared twice", b.binding));
            }
            match b.kind {
                BindingKind::Uniform if b.fields.as_ref().is_none_or(|f| f.is_empty()) => return Err(format!("{name}: uniform binding {} has no fields", b.name)),
                BindingKind::StorageRead | BindingKind::StorageReadWrite if b.element.is_none() => return Err(format!("{name}: storage binding {} has no element", b.name)),
                _ => {}
            }
        }
        for o in &s.output {
            let what = |e: &str| Err(format!("{name}: output {}: {e}", o.name));
            if o.element == Elem::F64 || o.shape.is_empty() || o.shape.len() > 2 {
                return what("an output is f32, u32 or i32 of one or two dimensions");
            }
            match (o.kind(), o.tie) {
                (OutputKind::Float, _) if o.element != Elem::F32 => return what("a float output is f32"),
                (OutputKind::Exact | OutputKind::Argmax, _) if o.element == Elem::F32 => return what("an exact or argmax output is u32 or i32"),
                (OutputKind::Argmax, None) => return what("an argmax output declares its tie rule (tie = \"exact\" or \"margin\")"),
                (OutputKind::Argmax, Some(Tie::Margin)) if o.score.is_none() || o.margin.is_none() || o.classes.is_none() => return what("tie = \"margin\" needs score, margin and classes"),
                (OutputKind::Float | OutputKind::Exact, Some(_)) => return what("a tie rule is an argmax output's"),
                _ => {}
            }
            for e in [&o.score, &o.margin, &o.classes].into_iter().flatten() {
                parse_expr(e).map_err(|e| format!("{name}: output {}: {e}", o.name))?;
            }
        }
        for i in &s.input {
            if let Some(g) = &i.generate {
                let need = |ok: bool, what: &str| if ok { Ok(()) } else { Err(format!("{name}: input {}: dist {:?} needs {what}", i.name, g.dist)) };
                match g.dist {
                    Dist::Uniform | Dist::Index => need(g.lo.is_some() && g.hi.is_some(), "lo and hi")?,
                    Dist::Normal => need(g.mean.is_some() && g.std.is_some(), "mean and std")?,
                    Dist::UnitRows => need(g.row.is_some() && i.element == Elem::F32, "row, of an f32 input")?,
                    Dist::Constant => need(g.value.is_some(), "value")?,
                }
                if g.dist == Dist::Index && i.element == Elem::F32 {
                    return Err(format!("{name}: input {}: an index input is u32 or i32", i.name));
                }
            }
        }
        let count = |r: Role| self.suites.iter().filter(|x| x.spec.role == r).map(|x| x.cases.len()).sum::<usize>();
        if self.suites.iter().filter(|x| x.spec.role == Role::Constraint).count() != 1 {
            return Err(format!("{name}: exactly one constraint suite"));
        }
        if count(Role::Timing) == 0 {
            return Err(format!("{name}: no timing case; the fitness's three speed columns are bands of the timing cases (`timing_bands`)"));
        }
        match (s.budget.deadline_factor, s.budget.deadline_floor_ms) {
            (None, None) => {}
            (Some(f), Some(m)) if f >= 1.0 && m > 0.0 => {}
            (f, m) => return Err(format!("{name}: [budget] deadline_factor ({f:?}, at least 1) and deadline_floor_ms ({m:?}, above 0) go together")),
        }
        if self.suites.iter().any(|x| x.spec.staged && x.spec.role != Role::Constraint) {
            return Err(format!("{name}: only the constraint suite can be staged"));
        }
        for su in &self.suites {
            let need = |what: &str, ok: bool| if ok { Ok(()) } else { Err(format!("{name}: suite {} needs {what}", su.spec.name)) };
            match su.spec.role {
                Role::Constraint => {
                    need("a norm", su.spec.norm.is_some() || self.output_of(su)?.kind() != OutputKind::Float)?;
                    need("pass", su.spec.pass.is_some() || self.output_of(su)?.kind() != OutputKind::Float)?;
                }
                Role::Accuracy => {
                    need("a norm", su.spec.norm.is_some() || self.output_of(su)?.kind() != OutputKind::Float)?;
                    need("a split of 0, 1 or 2", su.spec.split.is_some_and(|k| k < 3))?;
                    need("slack", su.spec.slack.is_some())?;
                }
                Role::Rung | Role::Timing => {}
            }
            if su.spec.norm == Some(Norm::CauchySchwarz) {
                need("tau", su.spec.tau.is_some())?;
            }
            self.output_of(su)?;
            for c in &su.cases {
                for i in &s.input {
                    if !c.inputs.contains_key(&i.name) && i.generate.is_none() {
                        return Err(format!("{name}: case {} of suite {} has no input {}, and the input declares no generate", c.name, su.spec.name, i.name));
                    }
                }
                if matches!(su.spec.role, Role::Constraint | Role::Accuracy) && s.oracle.mode == OracleMode::Reference && !c.expected.contains_key(&self.output_of(su)?.name) {
                    return Err(format!("{name}: case {} of suite {} has no expected {} (oracle mode reference)", c.name, su.spec.name, self.output_of(su)?.name));
                }
            }
        }
        Ok(())
    }

    /// The output a suite compares: its `output`, or the problem's only one.
    pub fn output_of(&self, su: &Suite) -> Result<&OutputSpec, String> {
        match &su.spec.output {
            Some(n) => self.spec.output.iter().find(|o| &o.name == n).ok_or_else(|| format!("suite {}: no output {n}", su.spec.name)),
            None if self.spec.output.len() == 1 => Ok(&self.spec.output[0]),
            None => Err(format!("suite {}: name the output it compares", su.spec.name)),
        }
    }

    /// The case's environment: its parameters, the defaults it lacks,
    /// `binding_limit`, then the derived values.
    pub fn env(&self, case: &Case, binding_limit: u64) -> Result<Env, String> {
        let mut env: Env = case.params.clone();
        env.insert("binding_limit".into(), binding_limit as f64);
        for (k, e) in &self.spec.params {
            if !env.contains_key(k) {
                let v = eval(e, &env)?;
                env.insert(k.clone(), v);
            }
        }
        bind_lets(&self.spec.derive, &mut env).map_err(|e| format!("case {}: {e}", case.name))?;
        Ok(env)
    }

    /// Read a case's files, checking each against its declared shape.
    pub fn case_data(&self, case: &Case, binding_limit: u64) -> Result<CaseData, String> {
        let env = self.env(case, binding_limit)?;
        let mut out = CaseData::default();
        for i in &self.spec.input {
            let n = i.shape.iter().map(|s| eval_int(s, &env)).collect::<Result<Vec<_>, _>>()?.iter().product::<u64>();
            let Some(path) = case.inputs.get(&i.name) else {
                let g = i.generate.as_ref().ok_or_else(|| format!("case {}: no input {}", case.name, i.name))?;
                out.inputs.insert(i.name.clone(), generate(i, g, n, case_seed(case, &i.name), &env).map_err(|e| format!("case {}: input {}: {e}", case.name, i.name))?);
                continue;
            };
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            if bytes.len() as u64 != n * i.element.size() {
                return Err(format!("{}: {} bytes, the shape says {} {:?}", path.display(), bytes.len(), n, i.element));
            }
            out.inputs.insert(i.name.clone(), bytes);
        }
        for o in &self.spec.output {
            let Some(files) = case.expected.get(&o.name) else { continue };
            let mut v = Vec::new();
            for f in files {
                let bytes = std::fs::read(f).map_err(|e| format!("{}: {e}", f.display()))?;
                if !(bytes.len() as u64).is_multiple_of(o.expected_elem().size()) {
                    return Err(format!("{}: {} bytes is not a whole number of {:?}", f.display(), bytes.len(), o.expected_elem()));
                }
                v.extend(o.expected_elem().decode(&bytes));
            }
            let n = self.output_elems(o, &env)?;
            if v.len() as u64 != n {
                return Err(format!("case {}: expected {} holds {} values, the output has {n}", case.name, o.name, v.len()));
            }
            out.expected.insert(o.name.clone(), v);
        }
        Ok(out)
    }

    pub fn output_dims(&self, o: &OutputSpec, env: &Env) -> Result<(usize, usize), String> {
        let d: Vec<u64> = o.shape.iter().map(|s| eval_int(s, env)).collect::<Result<_, _>>()?;
        Ok(if d.len() == 1 { (1, d[0] as usize) } else { (d[0] as usize, d[1] as usize) })
    }

    pub fn output_elems(&self, o: &OutputSpec, env: &Env) -> Result<u64, String> {
        let (r, c) = self.output_dims(o, env)?;
        Ok((r * c) as u64)
    }
}

// ------------------------------------------------------------------
// The kernel's contract
// ------------------------------------------------------------------

/// What the program reads from a compiled kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelInfo {
    pub workgroup: [u32; 3],
    pub bx: u32,
    pub by: u32,
    pub rows_x: bool,
    /// The modes its `// mode <name>` lines declare, sorted.
    pub modes: Vec<String>,
}

fn scalar_of(module: &naga::Module, ty: naga::Handle<naga::Type>, out: &mut Vec<Elem>) -> Result<(), String> {
    use naga::TypeInner;
    let elem = |s: naga::Scalar| -> Result<Elem, String> {
        match (s.kind, s.width) {
            (naga::ScalarKind::Float, 4) => Ok(Elem::F32),
            (naga::ScalarKind::Uint, 4) => Ok(Elem::U32),
            (naga::ScalarKind::Sint, 4) => Ok(Elem::I32),
            other => Err(format!("a scalar {other:?} the schema cannot name")),
        }
    };
    match &module.types[ty].inner {
        TypeInner::Scalar(s) => out.push(elem(*s)?),
        TypeInner::Vector { size, scalar } => out.extend(std::iter::repeat_n(elem(*scalar)?, *size as usize)),
        TypeInner::Struct { members, .. } => {
            for m in members {
                scalar_of(module, m.ty, out)?;
            }
        }
        TypeInner::Array { base, size: naga::ArraySize::Constant(n), .. } => {
            for _ in 0..n.get() {
                scalar_of(module, *base, out)?;
            }
        }
        other => return Err(format!("a type {other:?} the schema cannot name")),
    }
    Ok(())
}

/// The storage element of a binding's type: `array<T>` or `array<vecN<T>>`.
fn storage_elem(module: &naga::Module, ty: naga::Handle<naga::Type>) -> Result<Elem, String> {
    let base = match &module.types[ty].inner {
        naga::TypeInner::Array { base, .. } => *base,
        _ => ty,
    };
    let mut v = Vec::new();
    scalar_of(module, base, &mut v)?;
    let first = *v.first().ok_or("an empty element")?;
    if v.iter().any(|e| *e != first) {
        return Err("an element of mixed scalars".into());
    }
    Ok(first)
}

/// Parse and validate `source` (naga, the current target), find the
/// entry point, read the grid as `gram::grid_of` does when the problem
/// asks for the header, and check every `@group(0)` binding of the module
/// against the schema: declared, of the declared kind (a `storage_read`
/// binding may be any storage access; `storage_read_write` must be
/// writable), of the declared scalar (a uniform's scalars a prefix of its
/// `fields`). A kernel may leave a schema binding undeclared.
pub fn kernel_info(problem: &ProblemSpec, source: &str) -> Result<KernelInfo, String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    super::target::Target::current().validate(&module, source).map_err(|e| format!("validation: {e}"))?;
    let entry = &problem.kernel.entry;
    let ep = module.entry_points.iter().find(|e| &e.name == entry).ok_or_else(|| format!("no entry point {entry}"))?;
    let [x, y, z] = ep.workgroup_size;
    let (mut bx, mut by, mut rows_x) = (1, 1, false);
    if problem.kernel.grid_header {
        rows_x = super::safe_reader::grid_rows_x(source);
        if let Some((hx, hy, hbx, hby)) = super::safe_reader::grid_header(source) {
            if (hx, hy) != (x, y) {
                return Err(format!("the grid header says workgroup {hx} x {hy}, the shader says {x} x {y}"));
            }
            if !(1..=16).contains(&hbx) || !(1..=16).contains(&hby) {
                return Err(format!("the grid header's block {hbx} x {hby} is outside 1..=16"));
            }
            (bx, by) = (hbx, hby);
        }
    }
    let mut modes: Vec<String> = Vec::new();
    for l in source.lines() {
        if let Some(m) = l.trim().strip_prefix("// mode ") {
            let m = m.trim().to_string();
            if !problem.kernel.modes.contains(&m) {
                return Err(format!("mode: the kernel declares mode {m:?}, which the problem does not list"));
            }
            if !modes.contains(&m) {
                modes.push(m);
            }
        }
    }
    modes.sort();
    for (_, g) in module.global_variables.iter() {
        let Some(rb) = &g.binding else { continue };
        let what = format!("@group({}) @binding({})", rb.group, rb.binding);
        let spec = problem.binding.iter().find(|b| b.group == rb.group && b.binding == rb.binding).ok_or_else(|| format!("binding: the kernel declares {what}, which the problem's schema does not"))?;
        match (g.space, spec.kind) {
            (naga::AddressSpace::Uniform, BindingKind::Uniform) => {
                let mut v = Vec::new();
                scalar_of(&module, g.ty, &mut v).map_err(|e| format!("binding: {what} ({}): {e}", spec.name))?;
                let fields = spec.fields.as_deref().unwrap_or(&[]);
                if v.len() > fields.len() || v.iter().zip(fields).any(|(a, b)| a != b) {
                    return Err(format!("binding: {what} ({}) is {v:?}, the schema's fields are {fields:?}", spec.name));
                }
            }
            (naga::AddressSpace::Storage { access }, BindingKind::StorageRead | BindingKind::StorageReadWrite) => {
                if spec.kind == BindingKind::StorageReadWrite && !access.contains(naga::StorageAccess::STORE) {
                    return Err(format!("binding: {what} ({}) is read-only, the schema's is read_write", spec.name));
                }
                if spec.kind == BindingKind::StorageRead && access.contains(naga::StorageAccess::STORE) {
                    return Err(format!("binding: {what} ({}) is read_write, the schema's is read-only", spec.name));
                }
                let e = storage_elem(&module, g.ty).map_err(|e| format!("binding: {what} ({}): {e}", spec.name))?;
                if Some(e) != spec.element {
                    return Err(format!("binding: {what} ({}) holds {e:?}, the schema's {:?}", spec.name, spec.element));
                }
            }
            (space, kind) => return Err(format!("binding: {what} ({}) is in {space:?}, the schema's is {kind:?}", spec.name)),
        }
    }
    Ok(KernelInfo { workgroup: [x, y, z], bx, by, rows_x, modes })
}

// ------------------------------------------------------------------
// The plan
// ------------------------------------------------------------------

/// A buffer of the plan, by position.
pub type BufId = usize;

/// How a buffer begins.
#[derive(Debug, Clone, PartialEq)]
pub enum Init {
    /// Zero (a `Clear` follows it in a run).
    Zero,
    /// A slice of an input tensor from `byte_offset`, with words overwritten.
    Input { tensor: String, byte_offset: u64, set: Vec<(u64, Vec<u8>)> },
    /// A uniform's words.
    Words(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BufferDecl {
    pub name: String,
    pub elem: Elem,
    pub bytes: u64,
    pub uniform: bool,
    /// Made at staging (setup) and kept for every evaluation.
    pub resident: bool,
    pub init: Init,
}

/// One command of a run.
#[derive(Debug, Clone, PartialEq)]
pub enum Cmd {
    Create(BufId),
    Clear(BufId),
    Dispatch { bind: Vec<(u32, BufId)>, workgroups: [u32; 3] },
    Copy { src: BufId, src_offset: u64, dst: BufId, dst_offset: u64, bytes: u64 },
    Read { src: BufId, src_offset: u64, output: usize, dst_offset: u64, bytes: u64 },
    Sync,
}

/// A case's program, expanded.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub buffers: Vec<BufferDecl>,
    /// The resident buffers, in creation order.
    pub setup: Vec<BufId>,
    pub run: Vec<Cmd>,
    /// Each output's bytes, in `[[output]]` order.
    pub outputs: Vec<u64>,
    /// Its reads are made right with the syncs dropped (it may be batched).
    pub batch_safe: bool,
}

/// The first read that a later write in its segment would spoil, if any.
pub fn hazard(plan: &Plan, binding_kinds: &BTreeMap<u32, BindingKind>, keep_syncs: bool) -> Option<String> {
    let mut pending: Vec<BufId> = Vec::new();
    let name = |id: BufId| plan.buffers[id].name.clone();
    for c in &plan.run {
        let writes: Vec<BufId> = match c {
            Cmd::Sync if keep_syncs => {
                pending.clear();
                Vec::new()
            }
            Cmd::Read { src, .. } => {
                pending.push(*src);
                Vec::new()
            }
            Cmd::Clear(id) => vec![*id],
            Cmd::Copy { dst, .. } => vec![*dst],
            Cmd::Dispatch { bind, .. } => bind.iter().filter(|(b, _)| binding_kinds.get(b) == Some(&BindingKind::StorageReadWrite)).map(|(_, id)| *id).collect(),
            _ => Vec::new(),
        };
        if let Some(id) = writes.iter().find(|w| pending.contains(w)) {
            return Some(format!("{} is read and then written in the same segment; put a sync after the read", name(*id)));
        }
    }
    None
}

struct Builder<'a> {
    problem: &'a ProblemSpec,
    buffers: Vec<BufferDecl>,
    names: BTreeMap<String, BufId>,
    setup: Vec<BufId>,
    run: Vec<Cmd>,
    out_bytes: Vec<u64>,
}

impl Builder<'_> {
    fn buffer(&self, name: &str) -> Result<BufId, String> {
        self.names.get(name).copied().ok_or_else(|| format!("no buffer {name}"))
    }

    fn steps(&mut self, steps: &[Step], env: &Env, resident: bool) -> Result<(), String> {
        for s in steps {
            if let Some(w) = &s.when {
                if eval(w, env)? == 0.0 {
                    continue;
                }
            }
            if let Some(var) = &s.var {
                let n = eval_int(s.count.as_deref().ok_or("a loop needs count")?, env)?;
                for i in 0..n {
                    let mut e = env.clone();
                    e.insert(var.clone(), i as f64);
                    bind_lets(&s.lets, &mut e)?;
                    self.steps(&s.body, &e, resident)?;
                }
                continue;
            }
            let mut e = env.clone();
            bind_lets(&s.lets, &mut e)?;
            self.action(s, &e, resident)?;
        }
        Ok(())
    }

    fn declare(&mut self, d: BufferDecl) -> BufId {
        let id = self.buffers.len();
        self.names.insert(d.name.clone(), id);
        self.buffers.push(d);
        id
    }

    fn action(&mut self, s: &Step, env: &Env, resident: bool) -> Result<(), String> {
        let actions = [s.buffer.is_some(), s.uniform.is_some(), s.dispatch.is_some(), s.copy.is_some(), s.read.is_some(), s.sync.is_some(), s.require.is_some()].iter().filter(|x| **x).count();
        if actions != 1 {
            return Err(format!("a step has {actions} actions, not one: {s:?}"));
        }
        let int = |e: &Option<String>, default: u64| -> Result<u64, String> { e.as_deref().map_or(Ok(default), |x| eval_int(x, env)) };
        if let Some(r) = &s.require {
            if eval(r, env)? == 0.0 {
                return Err(format!("the problem requires {r}: {}", s.because.as_deref().unwrap_or("")));
            }
            return Ok(());
        }
        if let Some(b) = &s.buffer {
            let name = template(b, env)?;
            let elem = s.element.ok_or_else(|| format!("buffer {name} needs element"))?;
            let n = int(&s.elements, 0)?;
            if n == 0 {
                return Err(format!("buffer {name} has no elements"));
            }
            let init = match (&s.fill, &s.from) {
                (Some(Fill::Zero), None) => Init::Zero,
                (None, Some(t)) => {
                    let spec = self.problem.input.iter().find(|i| &i.name == t).ok_or_else(|| format!("buffer {name}: no input {t}"))?;
                    if spec.element != elem {
                        return Err(format!("buffer {name} is {elem:?}, input {t} is {:?}", spec.element));
                    }
                    let total = spec.shape.iter().map(|x| eval_int(x, env)).collect::<Result<Vec<_>, _>>()?.iter().product::<u64>();
                    let off = int(&s.offset, 0)?;
                    if off + n > total {
                        return Err(format!("buffer {name}: elements {off}..{} of input {t}, which has {total}", off + n));
                    }
                    let mut set = Vec::new();
                    for l in &s.set {
                        let (i, v) = l.split_once('=').ok_or_else(|| format!("set {l:?}: no '='"))?;
                        let i = eval_int(i.trim(), env)?;
                        if i >= n {
                            return Err(format!("buffer {name}: set index {i} of {n}"));
                        }
                        set.push((i * elem.size(), elem.encode(eval(v.trim(), env)?)?));
                    }
                    Init::Input { tensor: t.clone(), byte_offset: off * elem.size(), set }
                }
                _ => return Err(format!("buffer {name}: give fill = \"zero\" or from")),
            };
            let id = self.declare(BufferDecl { name, elem, bytes: n * elem.size(), uniform: false, resident, init: init.clone() });
            if resident {
                self.setup.push(id);
            } else {
                self.run.push(Cmd::Create(id));
                if init == Init::Zero {
                    self.run.push(Cmd::Clear(id));
                }
            }
        } else if let Some(u) = &s.uniform {
            let name = template(u, env)?;
            let of = s.of.as_deref().ok_or_else(|| format!("uniform {name} needs of"))?;
            let spec = self.problem.binding.iter().find(|b| b.name == of && b.kind == BindingKind::Uniform).ok_or_else(|| format!("uniform {name}: no uniform binding {of}"))?;
            let fields = spec.fields.as_deref().unwrap_or(&[]);
            let words = s.words.as_deref().ok_or_else(|| format!("uniform {name} needs words"))?;
            if words.len() != fields.len() {
                return Err(format!("uniform {name}: {} words for {} fields", words.len(), fields.len()));
            }
            let mut bytes = Vec::new();
            for (w, f) in words.iter().zip(fields) {
                bytes.extend(f.encode(eval(w, env)?)?);
            }
            let id = self.declare(BufferDecl { name, elem: Elem::U32, bytes: bytes.len() as u64, uniform: true, resident, init: Init::Words(bytes) });
            if resident {
                self.setup.push(id);
            } else {
                self.run.push(Cmd::Create(id));
            }
        } else if let Some(map) = &s.dispatch {
            if resident {
                return Err("a dispatch in setup".into());
            }
            let mut bind = Vec::new();
            for b in &self.problem.binding {
                let t = map.get(&b.name).ok_or_else(|| format!("a dispatch leaves binding {} unbound", b.name))?;
                let id = self.buffer(&template(t, env)?)?;
                let buf = &self.buffers[id];
                match b.kind {
                    BindingKind::Uniform => {
                        let want = 4 * b.fields.as_ref().map_or(0, |f| f.len()) as u64;
                        if !buf.uniform || buf.bytes != want {
                            return Err(format!("binding {} wants a uniform of {want} bytes, {} is not one", b.name, buf.name));
                        }
                    }
                    BindingKind::StorageRead | BindingKind::StorageReadWrite => {
                        if buf.uniform {
                            return Err(format!("binding {} is storage, {} is a uniform", b.name, buf.name));
                        }
                        if b.kind == BindingKind::StorageReadWrite && buf.resident {
                            return Err(format!("binding {} is writable, {} is resident (a setup buffer no kernel may change)", b.name, buf.name));
                        }
                    }
                }
                if bind.iter().any(|(_, x)| *x == id) {
                    return Err(format!("alias: {} is bound twice in one dispatch", buf.name));
                }
                bind.push((b.binding, id));
            }
            if let Some(extra) = map.keys().find(|k| !self.problem.binding.iter().any(|b| &b.name == *k)) {
                return Err(format!("a dispatch binds {extra}, which is not in the schema"));
            }
            let wg = s.workgroups.as_deref().ok_or("a dispatch needs workgroups")?;
            if wg.len() != 3 {
                return Err("workgroups takes three expressions".into());
            }
            let mut workgroups = [0u32; 3];
            for (w, e) in workgroups.iter_mut().zip(wg) {
                let v = eval_int(e, env)?;
                *w = u32::try_from(v).ok().filter(|v| *v <= 65_535).ok_or_else(|| format!("workgroups {e:?} = {v}, over 65,535"))?;
            }
            self.run.push(Cmd::Dispatch { bind, workgroups });
        } else if let Some(src) = &s.copy {
            let (src, dst) = (self.buffer(&template(src, env)?)?, self.buffer(&template(s.to.as_deref().ok_or("a copy needs to")?, env)?)?);
            let es = self.buffers[src].elem.size();
            let n = int(&s.elements, 0)? * es;
            let (so, d_o) = (int(&s.src_offset, 0)? * es, int(&s.dst_offset, 0)? * es);
            if so + n > self.buffers[src].bytes || d_o + n > self.buffers[dst].bytes || self.buffers[dst].resident || resident {
                return Err(format!("copy {} -> {}: out of range, or into a resident buffer", self.buffers[src].name, self.buffers[dst].name));
            }
            self.run.push(Cmd::Copy { src, src_offset: so, dst, dst_offset: d_o, bytes: n });
        } else if let Some(src) = &s.read {
            let src = self.buffer(&template(src, env)?)?;
            let into = s.into.as_deref().ok_or("a read needs into")?;
            let output = self.problem.output.iter().position(|o| o.name == into).ok_or_else(|| format!("no output {into}"))?;
            let es = self.problem.output[output].element.size();
            let n = int(&s.elements, 0)? * es;
            let (so, d_o) = (int(&s.src_offset, 0)? * es, int(&s.offset, 0)? * es);
            if resident || so + n > self.buffers[src].bytes || d_o + n > self.out_bytes[output] {
                return Err(format!("read {} into {into}: out of range", self.buffers[src].name));
            }
            self.run.push(Cmd::Read { src, src_offset: so, output, dst_offset: d_o, bytes: n });
        } else if s.sync == Some(true) {
            if !resident {
                self.run.push(Cmd::Sync);
            }
        } else {
            return Err("sync = false is not a step".into());
        }
        Ok(())
    }
}

impl Problem {
    /// The binding kinds by binding number.
    pub fn binding_kinds(&self) -> BTreeMap<u32, BindingKind> {
        self.spec.binding.iter().map(|b| (b.binding, b.kind)).collect()
    }

    /// Expand a case's program: the setup always, the run for a kernel
    /// (`None` builds the setup alone, for staging; its buffers are the
    /// same either way).
    pub fn plan(&self, case: &Case, binding_limit: u64, kernel: Option<&KernelInfo>) -> Result<Plan, String> {
        let env = self.env(case, binding_limit)?;
        let mut out_bytes = Vec::new();
        for o in &self.spec.output {
            out_bytes.push(self.output_elems(o, &env)? * o.element.size());
        }
        let mut b = Builder { problem: &self.spec, buffers: Vec::new(), names: BTreeMap::new(), setup: Vec::new(), run: Vec::new(), out_bytes: out_bytes.clone() };
        b.steps(&self.spec.setup, &env, true).map_err(|e| format!("case {}: setup: {e}", case.name))?;
        if let Some(k) = kernel {
            let mut env = env;
            for (n, v) in [("wg_x", k.workgroup[0]), ("wg_y", k.workgroup[1]), ("wg_z", k.workgroup[2]), ("bx", k.bx), ("by", k.by), ("rows_x", u32::from(k.rows_x))] {
                env.insert(n.into(), f64::from(v));
            }
            for m in &self.spec.kernel.modes {
                env.insert(format!("mode_{m}"), if k.modes.contains(m) { 1.0 } else { 0.0 });
            }
            bind_lets(&self.spec.derive_run, &mut env).map_err(|e| format!("case {}: {e}", case.name))?;
            b.steps(&self.spec.run, &env, false).map_err(|e| format!("case {}: run: {e}", case.name))?;
        }
        let mut plan = Plan { buffers: b.buffers, setup: b.setup, run: b.run, outputs: out_bytes, batch_safe: false };
        let kinds = self.binding_kinds();
        if let Some(h) = hazard(&plan, &kinds, true) {
            return Err(format!("case {}: {h}", case.name));
        }
        plan.batch_safe = hazard(&plan, &kinds, false).is_none();
        Ok(plan)
    }

    /// A case's work, the cost model's unit.
    pub fn work(&self, case: &Case, binding_limit: u64) -> Result<f64, String> {
        eval(&self.spec.work, &self.env(case, binding_limit)?)
    }
}

/// The initial bytes of a buffer from the case's data.
pub fn initial_bytes(b: &BufferDecl, data: &CaseData) -> Result<Vec<u8>, String> {
    Ok(match &b.init {
        Init::Zero => vec![0u8; b.bytes as usize],
        Init::Words(w) => w.clone(),
        Init::Input { tensor, byte_offset, set } => {
            let src = data.inputs.get(tensor).ok_or_else(|| format!("buffer {}: no data for input {tensor}", b.name))?;
            let (a, z) = (*byte_offset as usize, (*byte_offset + b.bytes) as usize);
            let mut v = src.get(a..z).ok_or_else(|| format!("buffer {}: input {tensor} holds {} bytes, not {z}", b.name, src.len()))?.to_vec();
            for (at, bytes) in set {
                v[*at as usize..*at as usize + bytes.len()].copy_from_slice(bytes);
            }
            v
        }
    })
}

// ------------------------------------------------------------------
// Norms
// ------------------------------------------------------------------

/// One comparison point's error and its non-finite count.
pub fn norm_value(norm: Norm, out: &[f32], want: &[f64], rows: usize, cols: usize, tau: f64) -> (f64, usize) {
    match norm {
        Norm::MeanAbsOverMax => (crate::wgsl::probes::distance_to(out, want), 0),
        Norm::CauchySchwarz => {
            if rows != cols {
                return (f64::INFINITY, out.len());
            }
            crate::wgsl::gram::check_max_tau(out, want, rows, tau)
        }
        Norm::MaxAbsOverRms => {
            let rms = (want.iter().filter(|w| w.is_finite()).map(|w| w * w).sum::<f64>() / want.len().max(1) as f64).sqrt();
            let (mut worst, mut nf) = (0.0f64, 0usize);
            for (g, w) in out.iter().zip(want) {
                let g = f64::from(*g);
                if !w.is_finite() {
                    if !((g.is_nan() && w.is_nan()) || g == *w) {
                        nf += 1;
                    }
                    continue;
                }
                if !g.is_finite() {
                    nf += 1;
                    continue;
                }
                worst = worst.max((g - w).abs());
            }
            let v = if nf > 0 { f64::INFINITY } else if rms > 0.0 { worst / rms } else if worst == 0.0 { 0.0 } else { f64::INFINITY };
            (v, nf)
        }
    }
}

// ------------------------------------------------------------------
// Executors
// ------------------------------------------------------------------

/// One run of a case's program.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RunOut {
    /// Each output's bytes (empty when not read).
    pub outputs: Vec<Vec<u8>>,
    /// GPU milliseconds: the sum of every dispatch's timestamps.
    pub gpu_ms: f64,
    /// Wall milliseconds of the compute submissions alone.
    pub compute_ms: f64,
    /// Wall milliseconds of the submissions and read-backs.
    pub wall_ms: f64,
}

impl RunOut {
    /// As `gram::Build` (for `gram::Split::build`).
    pub fn build(&self) -> Build {
        Build { h: Vec::new(), gpu_ms: self.gpu_ms, compute_ms: self.compute_ms, wall_ms: self.wall_ms, upload_ms: 0.0, passes: 0, item_bands: 0, row_bands: 0 }
    }
}

/// What runs a plan.
pub trait Executor {
    type Kernel;
    type Staged;
    /// The per-binding byte limit (`binding_limit`).
    fn binding_limit(&self) -> u64;
    /// What runs the kernels, as the oracle's cache key names it (two
    /// executors may compute different bits).
    fn oracle_tag(&self) -> String;
    /// Compile a kernel for the problem: [`kernel_info`], then the device's own.
    fn compile(&self, problem: &Problem, source: &str) -> Result<(Self::Kernel, KernelInfo), String>;
    /// Make a case's resident buffers; the milliseconds it took.
    fn stage(&self, plan: &Plan, data: &CaseData) -> Result<(Self::Staged, f64), String>;
    /// Run one case; `read` false skips the reads.
    fn run(&self, k: &Self::Kernel, plan: &Plan, data: &CaseData, staged: &Self::Staged, read: bool) -> Result<RunOut, String>;
    /// Run several cases in one submission and one read-back, every plan
    /// batch-safe; each case's run, and the submission's compute ms.
    fn run_batch(&self, k: &Self::Kernel, items: &[(&Plan, &CaseData, &Self::Staged)]) -> Result<(Vec<RunOut>, f64), String>;
    /// One case run `n` times for timing, nothing read: the summed GPU ms
    /// and the longest compute ms a run (the staged slow gate's measure).
    /// One run after another unless the executor batches them.
    fn run_repeat(&self, k: &Self::Kernel, plan: &Plan, data: &CaseData, staged: &Self::Staged, n: usize) -> Result<(f64, f64), String> {
        let (mut gpu, mut longest) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let r = self.run(k, plan, data, staged, false)?;
            gpu += r.gpu_ms;
            longest = longest.max(r.compute_ms);
        }
        Ok((gpu, longest))
    }
}

/// The runs of a plan one timing submission holds: every dispatch takes two
/// of a query set's timestamps (at most 4,096 a submission) and every run its
/// own buffers (at most 32 MiB a submission). Measured on nluca, 2026-10-15:
/// 34 batched runs at 100,000 rows (8,192 timestamps, 512 MiB) failed with
/// ResourceCreationFailed; 28 (256 MiB) left the wait for the GPU blocked at
/// 0 % CPU (killed after 180 s); 4 (32 MiB) ran; 32 runs at 10,000 rows
/// (about 22 MB) ran. Why the larger submissions block is not yet known.
pub fn runs_a_submission(plan: &Plan) -> usize {
    let dispatches = plan.run.iter().filter(|c| matches!(c, Cmd::Dispatch { .. })).count().max(1);
    let bytes: u64 = plan.buffers.iter().filter(|b| !b.resident).map(|b| b.bytes).sum::<u64>() + plan.outputs.iter().sum::<u64>();
    (2048 / dispatches).min((32u64 << 20).checked_div(bytes).unwrap_or(u64::MAX).min(u64::from(u32::MAX)) as usize).max(1)
}

/// A host model of a kernel: one invocation over the bound buffers.
pub trait HostKernel {
    /// Run one invocation; return its operation count (the host
    /// executor's synthetic time).
    fn invoke(&self, ids: &HostIds, mem: &mut HostMemory) -> u64;
}

/// The ids of one invocation.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostIds {
    pub gid: [u32; 3],
    pub lid: [u32; 3],
    pub wid: [u32; 3],
    pub nwg: [u32; 3],
}

/// The buffers bound to one dispatch, by binding number, under wgpu's
/// runtime bounds checks (`ReadZeroSkipWrite`): a load out of range reads
/// zero, a store out of range or to a read-only binding is skipped.
pub struct HostMemory {
    slots: Vec<(u32, Vec<u8>, bool)>,
}

impl HostMemory {
    fn slot(&self, binding: u32) -> Option<&(u32, Vec<u8>, bool)> {
        self.slots.iter().find(|s| s.0 == binding)
    }

    pub fn load_u32(&self, binding: u32, index: u32) -> u32 {
        let Some((_, b, _)) = self.slot(binding) else { return 0 };
        let at = index as usize * 4;
        b.get(at..at + 4).map_or(0, |w| u32::from_le_bytes(w.try_into().expect("4 bytes")))
    }

    pub fn load_f32(&self, binding: u32, index: u32) -> f32 {
        f32::from_bits(self.load_u32(binding, index))
    }

    pub fn store_u32(&mut self, binding: u32, index: u32, v: u32) {
        let Some((_, b, writable)) = self.slots.iter_mut().find(|s| s.0 == binding) else { return };
        let at = index as usize * 4;
        if *writable && at + 4 <= b.len() {
            b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
    }

    pub fn store_f32(&mut self, binding: u32, index: u32, v: f32) {
        self.store_u32(binding, index, v.to_bits());
    }
}

/// The CPU executor: plans run as the device runs them (segments, deferred
/// reads, batches), each dispatch invoking a registered [`HostKernel`] for
/// every invocation of the grid in order. Times are synthetic and
/// deterministic: a dispatch costs `dispatch_ms` plus `ns_per_op` per
/// operation its invocations report, a compute submission `submit_ms`.
pub struct HostExecutor {
    pub binding_limit: u64,
    pub ns_per_op: f64,
    pub dispatch_ms: f64,
    pub submit_ms: f64,
    models: Vec<(String, Box<dyn HostKernel>)>,
}

impl HostExecutor {
    pub fn new(binding_limit: u64) -> HostExecutor {
        HostExecutor { binding_limit, ns_per_op: 1.0, dispatch_ms: 0.01, submit_ms: 0.5, models: Vec::new() }
    }

    /// The model `compile` returns for exactly this kernel text.
    pub fn register(&mut self, source: &str, model: Box<dyn HostKernel>) {
        self.models.push((source.to_string(), model));
    }

    /// Run `cmds` over `bufs`; reads deferred to each segment's end. The
    /// gpu ms per item and the compute ms of every submission.
    fn exec(&self, k: usize, info: &KernelInfo, items: &mut [HostItem], batch: bool, read: bool) -> Result<f64, String> {
        let model = &self.models[k].1;
        let mut compute = 0.0;
        // Each item's command cursor; a segment ends at a sync (or, batched,
        // at the end of every item).
        let mut segment_ms = 0.0;
        let mut pending: Vec<(usize, BufId, u64, usize, u64, u64)> = Vec::new();
        let mut in_segment = false;
        let flush = |items: &mut [HostItem], pending: &mut Vec<(usize, BufId, u64, usize, u64, u64)>, compute: &mut f64, seg: &mut f64, any: &mut bool| -> Result<(), String> {
            if *any {
                *compute += self.submit_ms + *seg;
            }
            for (item, src, so, out, d_o, n) in pending.drain(..) {
                let bytes = items[item].bufs[src].as_ref().ok_or("a read of a buffer never made")?[so as usize..(so + n) as usize].to_vec();
                items[item].outputs[out][d_o as usize..(d_o + n) as usize].copy_from_slice(&bytes);
            }
            *seg = 0.0;
            *any = false;
            Ok(())
        };
        for i in 0..items.len() {
            let run = items[i].plan.run.clone();
            for c in &run {
                match c {
                    Cmd::Create(id) => {
                        let b = &items[i].plan.buffers[*id];
                        items[i].bufs[*id] = Some(initial_bytes(b, items[i].data)?);
                    }
                    Cmd::Clear(id) => {
                        if let Some(b) = items[i].bufs[*id].as_mut() {
                            b.iter_mut().for_each(|x| *x = 0);
                        }
                        in_segment = true;
                    }
                    Cmd::Dispatch { bind, workgroups } => {
                        let mut mem = HostMemory { slots: Vec::new() };
                        let kinds: Vec<bool> = bind.iter().map(|(b, _)| items[i].writable.get(b).copied().unwrap_or(false)).collect();
                        for ((b, id), w) in bind.iter().zip(&kinds) {
                            let data = items[i].bufs[*id].clone().ok_or("a dispatch of a buffer never made")?;
                            mem.slots.push((*b, data, *w));
                        }
                        let mut ops = 0u64;
                        let wg = info.workgroup;
                        for wz in 0..workgroups[2] {
                            for wy in 0..workgroups[1] {
                                for wx in 0..workgroups[0] {
                                    for lz in 0..wg[2] {
                                        for ly in 0..wg[1] {
                                            for lx in 0..wg[0] {
                                                let ids = HostIds { gid: [wx * wg[0] + lx, wy * wg[1] + ly, wz * wg[2] + lz], lid: [lx, ly, lz], wid: [wx, wy, wz], nwg: *workgroups };
                                                ops += model.invoke(&ids, &mut mem);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        for ((_, id), (_, data, w)) in bind.iter().zip(mem.slots) {
                            if w {
                                items[i].bufs[*id] = Some(data);
                            }
                        }
                        let ms = self.dispatch_ms + ops as f64 * self.ns_per_op / 1e6;
                        items[i].gpu_ms += ms;
                        segment_ms += ms;
                        in_segment = true;
                    }
                    Cmd::Copy { src, src_offset, dst, dst_offset, bytes } => {
                        let v = items[i].bufs[*src].as_ref().ok_or("a copy of a buffer never made")?[*src_offset as usize..(*src_offset + *bytes) as usize].to_vec();
                        items[i].bufs[*dst].as_mut().ok_or("a copy into a buffer never made")?[*dst_offset as usize..(*dst_offset + *bytes) as usize].copy_from_slice(&v);
                        in_segment = true;
                    }
                    Cmd::Read { src, src_offset, output, dst_offset, bytes } => {
                        if read {
                            pending.push((i, *src, *src_offset, *output, *dst_offset, *bytes));
                        }
                    }
                    Cmd::Sync => {
                        if !batch {
                            flush(items, &mut pending, &mut compute, &mut segment_ms, &mut in_segment)?;
                        }
                    }
                }
            }
            if !batch {
                flush(items, &mut pending, &mut compute, &mut segment_ms, &mut in_segment)?;
            }
        }
        flush(items, &mut pending, &mut compute, &mut segment_ms, &mut in_segment)?;
        Ok(compute)
    }
}

struct HostItem<'a> {
    plan: &'a Plan,
    data: &'a CaseData,
    bufs: Vec<Option<Vec<u8>>>,
    writable: BTreeMap<u32, bool>,
    outputs: Vec<Vec<u8>>,
    gpu_ms: f64,
}

/// A case's resident buffers on the host.
pub struct HostStaged {
    bufs: Vec<(BufId, Vec<u8>)>,
}

impl HostExecutor {
    fn item<'a>(&self, plan: &'a Plan, data: &'a CaseData, staged: &HostStaged, writable: &BTreeMap<u32, bool>) -> HostItem<'a> {
        let mut bufs = vec![None; plan.buffers.len()];
        for (id, b) in &staged.bufs {
            bufs[*id] = Some(b.clone());
        }
        HostItem { plan, data, bufs, writable: writable.clone(), outputs: plan.outputs.iter().map(|n| vec![0u8; *n as usize]).collect(), gpu_ms: 0.0 }
    }
}

/// The host kernel: its model's index, its info and which bindings it may write.
pub struct HostCompiled {
    model: usize,
    info: KernelInfo,
    writable: BTreeMap<u32, bool>,
}

impl Executor for HostExecutor {
    type Kernel = HostCompiled;
    type Staged = HostStaged;

    fn binding_limit(&self) -> u64 {
        self.binding_limit
    }

    fn oracle_tag(&self) -> String {
        "host".into()
    }

    fn compile(&self, problem: &Problem, source: &str) -> Result<(HostCompiled, KernelInfo), String> {
        let info = kernel_info(&problem.spec, source)?;
        let model = self.models.iter().position(|(s, _)| s == source).ok_or("host executor: no model is registered for this kernel text")?;
        let writable = problem.spec.binding.iter().map(|b| (b.binding, b.kind == BindingKind::StorageReadWrite)).collect();
        Ok((HostCompiled { model, info: info.clone(), writable }, info))
    }

    fn stage(&self, plan: &Plan, data: &CaseData) -> Result<(HostStaged, f64), String> {
        let bufs = plan.setup.iter().map(|id| initial_bytes(&plan.buffers[*id], data).map(|b| (*id, b))).collect::<Result<_, _>>()?;
        Ok((HostStaged { bufs }, 0.0))
    }

    fn run(&self, k: &HostCompiled, plan: &Plan, data: &CaseData, staged: &HostStaged, read: bool) -> Result<RunOut, String> {
        let mut items = vec![self.item(plan, data, staged, &k.writable)];
        let compute = self.exec(k.model, &k.info, &mut items, false, read)?;
        let it = items.pop().expect("one item");
        Ok(RunOut { outputs: if read { it.outputs } else { Vec::new() }, gpu_ms: it.gpu_ms, compute_ms: compute, wall_ms: compute })
    }

    fn run_batch(&self, k: &HostCompiled, items: &[(&Plan, &CaseData, &HostStaged)]) -> Result<(Vec<RunOut>, f64), String> {
        if let Some((p, _, _)) = items.iter().find(|(p, _, _)| !p.batch_safe) {
            return Err(format!("a plan of {} commands is not batch-safe", p.run.len()));
        }
        let mut its: Vec<HostItem> = items.iter().map(|(p, d, s)| self.item(p, d, s, &k.writable)).collect();
        let compute = self.exec(k.model, &k.info, &mut its, true, true)?;
        Ok((its.into_iter().map(|it| RunOut { outputs: it.outputs, gpu_ms: it.gpu_ms, compute_ms: compute, wall_ms: compute }).collect(), compute))
    }
}

/// THE CPU EXECUTOR OF A PROBLEM: the plans run as the device runs them,
/// every dispatch the kernel's own WGSL on `cpu.rs` (the whole dispatch,
/// workgroup memory shared, WGSL's value semantics, wgpu's bounds policy),
/// so no model of the kernel is written. Its times are the CPU's wall
/// milliseconds (the speed-ups of a CPU run say nothing about a device).
/// The oracle's executor when a problem says `[oracle] on = "cpu"`, and the
/// harness's executor in the tests.
pub struct CpuExecutor {
    pub binding_limit: u64,
    /// The cap on statements a dispatch executes (`cpu::run`).
    pub max_steps: u64,
}

impl CpuExecutor {
    pub fn new(binding_limit: u64) -> CpuExecutor {
        CpuExecutor { binding_limit, max_steps: 1 << 40 }
    }
}

/// How a binding's bytes become a `cpu::Val`.
#[derive(Clone)]
enum CpuShape {
    /// A runtime-sized array of this element type.
    Elements(naga::Handle<naga::Type>),
    /// A value of this whole type (a fixed array, a uniform).
    Whole(naga::Handle<naga::Type>),
}

/// A kernel on the CPU executor: its module and, per binding number, its
/// global's name, its shape, and whether the schema lets it be written.
pub struct CpuKernel {
    module: naga::Module,
    bindings: BTreeMap<u32, (String, CpuShape, bool)>,
}

/// A case's resident buffers on the CPU.
pub struct CpuStaged {
    bufs: Vec<(BufId, Vec<u8>)>,
}

fn words_of(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| {
        let mut w = [0u8; 4];
        w[..c.len()].copy_from_slice(c);
        u32::from_le_bytes(w)
    }).collect()
}

impl CpuExecutor {
    fn value(k: &CpuKernel, shape: &CpuShape, bytes: &[u8]) -> Result<crate::wgsl::cpu::Val, String> {
        let words = words_of(bytes);
        match shape {
            CpuShape::Elements(e) => crate::wgsl::cpu::from_words(&k.module, *e, &words),
            CpuShape::Whole(t) => match crate::wgsl::cpu::from_words(&k.module, *t, &words)? {
                crate::wgsl::cpu::Val::V(mut v) if !v.is_empty() => Ok(v.remove(0)),
                _ => crate::wgsl::cpu::zero(&k.module, *t),
            },
        }
    }

    /// Run a plan's commands; reads are made when met (a plan's hazard rule
    /// guarantees no later write in its segment spoils one). The milliseconds
    /// of its dispatches.
    fn exec(&self, k: &CpuKernel, plan: &Plan, data: &CaseData, staged: &CpuStaged, read: bool, outputs: &mut [Vec<u8>]) -> Result<f64, String> {
        let mut bufs: Vec<Option<Vec<u8>>> = vec![None; plan.buffers.len()];
        for (id, b) in &staged.bufs {
            bufs[*id] = Some(b.clone());
        }
        let mut ms = 0.0f64;
        for c in &plan.run {
            match c {
                Cmd::Create(id) => bufs[*id] = Some(initial_bytes(&plan.buffers[*id], data)?),
                Cmd::Clear(id) => bufs[*id].as_mut().ok_or("a clear of a buffer never made")?.iter_mut().for_each(|x| *x = 0),
                Cmd::Dispatch { bind, workgroups } => {
                    let mut vals = std::collections::HashMap::new();
                    for (b, id) in bind {
                        let Some((name, shape, _)) = k.bindings.get(b) else { continue };
                        let bytes = bufs[*id].as_ref().ok_or("a dispatch of a buffer never made")?;
                        vals.insert(name.clone(), Self::value(k, shape, bytes)?);
                    }
                    let t = Instant::now();
                    let d = crate::wgsl::cpu::run(&k.module, vals, *workgroups, self.max_steps)?;
                    ms += t.elapsed().as_secs_f64() * 1e3;
                    for (b, id) in bind {
                        let Some((name, _, true)) = k.bindings.get(b) else { continue };
                        let mut w = Vec::new();
                        d.buffers.get(name).ok_or_else(|| format!("the dispatch returned no {name}"))?.words(&mut w);
                        let buf = bufs[*id].as_mut().expect("bound above");
                        if w.len() * 4 != buf.len() {
                            return Err(format!("{name}: the dispatch returned {} words for {} bytes", w.len(), buf.len()));
                        }
                        buf.copy_from_slice(&w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
                    }
                }
                Cmd::Copy { src, src_offset, dst, dst_offset, bytes } => {
                    let v = bufs[*src].as_ref().ok_or("a copy of a buffer never made")?[*src_offset as usize..(*src_offset + *bytes) as usize].to_vec();
                    bufs[*dst].as_mut().ok_or("a copy into a buffer never made")?[*dst_offset as usize..(*dst_offset + *bytes) as usize].copy_from_slice(&v);
                }
                Cmd::Read { src, src_offset, output, dst_offset, bytes } => {
                    if read {
                        let v = &bufs[*src].as_ref().ok_or("a read of a buffer never made")?[*src_offset as usize..(*src_offset + *bytes) as usize];
                        outputs[*output][*dst_offset as usize..(*dst_offset + *bytes) as usize].copy_from_slice(v);
                    }
                }
                Cmd::Sync => {}
            }
        }
        // A plan of no dispatch still takes time (a speed-up's denominator is never 0).
        Ok(ms.max(1e-6))
    }
}

impl Executor for CpuExecutor {
    type Kernel = CpuKernel;
    type Staged = CpuStaged;

    fn binding_limit(&self) -> u64 {
        self.binding_limit
    }

    fn oracle_tag(&self) -> String {
        "cpu".into()
    }

    fn compile(&self, problem: &Problem, source: &str) -> Result<(CpuKernel, KernelInfo), String> {
        let info = kernel_info(&problem.spec, source)?;
        let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
        let mut bindings = BTreeMap::new();
        for (_, g) in module.global_variables.iter() {
            let Some(rb) = &g.binding else { continue };
            let shape = match &module.types[g.ty].inner {
                naga::TypeInner::Array { base, size: naga::ArraySize::Dynamic, .. } => CpuShape::Elements(*base),
                _ => CpuShape::Whole(g.ty),
            };
            let writable = problem.spec.binding.iter().any(|b| b.binding == rb.binding && b.kind == BindingKind::StorageReadWrite);
            bindings.insert(rb.binding, (g.name.clone().unwrap_or_default(), shape, writable));
        }
        Ok((CpuKernel { module, bindings }, info))
    }

    fn stage(&self, plan: &Plan, data: &CaseData) -> Result<(CpuStaged, f64), String> {
        let bufs = plan.setup.iter().map(|id| initial_bytes(&plan.buffers[*id], data).map(|b| (*id, b))).collect::<Result<_, _>>()?;
        Ok((CpuStaged { bufs }, 0.0))
    }

    fn run(&self, k: &CpuKernel, plan: &Plan, data: &CaseData, staged: &CpuStaged, read: bool) -> Result<RunOut, String> {
        let mut outputs: Vec<Vec<u8>> = plan.outputs.iter().map(|n| vec![0u8; *n as usize]).collect();
        let ms = self.exec(k, plan, data, staged, read, &mut outputs)?;
        Ok(RunOut { outputs: if read { outputs } else { Vec::new() }, gpu_ms: ms, compute_ms: ms, wall_ms: ms })
    }

    fn run_batch(&self, k: &CpuKernel, items: &[(&Plan, &CaseData, &CpuStaged)]) -> Result<(Vec<RunOut>, f64), String> {
        let mut runs = Vec::with_capacity(items.len());
        let mut total = 0.0;
        for (p, d, s) in items {
            if !p.batch_safe {
                return Err(format!("a plan of {} commands is not batch-safe", p.run.len()));
            }
            let r = self.run(k, p, d, s, true)?;
            total += r.compute_ms;
            runs.push(r);
        }
        Ok((runs, total))
    }
}

#[cfg(feature = "gpu")]
pub use device::DeviceExecutor;

#[cfg(feature = "gpu")]
mod device {
    //! The device executor: the paths of `gram.rs` (`Gram::compile_with`,
    //! `Gram::run`, `Gram::run_batch`) driven by a plan.

    use std::time::Instant;

    use wgpu::util::DeviceExt;

    use super::{initial_bytes, kernel_info, BindingKind, BufId, CaseData, Cmd, Executor, KernelInfo, Plan, Problem, RunOut};
    use crate::wgsl::gram::{effective_buffer_limit, wait_gpu};
    use crate::wgsl::oracle::device::Device;

    pub struct DeviceExecutor<'a> {
        pub dev: &'a Device,
        limit: u64,
    }

    impl<'a> DeviceExecutor<'a> {
        pub fn new(dev: &'a Device) -> DeviceExecutor<'a> {
            DeviceExecutor { dev, limit: effective_buffer_limit(&dev.device) }
        }
    }

    pub struct DeviceKernel {
        pipeline: wgpu::ComputePipeline,
        period_ns: f32,
    }

    pub struct DeviceStaged {
        bufs: Vec<(BufId, wgpu::Buffer)>,
    }

    impl Drop for DeviceStaged {
        fn drop(&mut self) {
            for (_, b) in &self.bufs {
                b.destroy();
            }
        }
    }

    /// One case's buffers during a run.
    struct Live<'p> {
        plan: &'p Plan,
        data: &'p CaseData,
        bufs: Vec<Option<Slot<'p>>>,
        outputs: Vec<Vec<u8>>,
        gpu_ms: f64,
    }

    /// A run's buffer, or a resident one borrowed from its stage.
    enum Slot<'p> {
        Own(wgpu::Buffer),
        Resident(&'p wgpu::Buffer),
    }

    impl Live<'_> {
        fn buf(&self, id: BufId, what: &str) -> Result<&wgpu::Buffer, String> {
            match self.bufs[id].as_ref() {
                Some(Slot::Own(b)) => Ok(b),
                Some(Slot::Resident(b)) => Ok(b),
                None => Err(format!("{what} of a buffer never made")),
            }
        }
    }

    impl DeviceExecutor<'_> {
        fn make(&self, plan: &Plan, id: BufId, data: &CaseData) -> Result<wgpu::Buffer, String> {
            let b = &plan.buffers[id];
            let usage = if b.uniform { wgpu::BufferUsages::UNIFORM } else { wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST };
            Ok(match &b.init {
                super::Init::Zero => self.dev.device.create_buffer(&wgpu::BufferDescriptor { label: Some(&b.name), size: b.bytes, usage, mapped_at_creation: false }),
                _ => self.dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(&b.name), contents: &initial_bytes(b, data)?, usage }),
            })
        }

        /// Run the items' commands: segments end at a sync (dropped when
        /// batched) and at the end; each segment is one timed compute
        /// submission, then one read-back submission of its timestamps
        /// (resolved after the passes completed, as `Gram::run` does) and
        /// its reads. Returns the compute ms.
        fn exec<'p>(&self, k: &DeviceKernel, live: &mut [Live<'p>], staged: &[&'p DeviceStaged], batch: bool, read: bool) -> Result<f64, String> {
            let device = &self.dev.device;
            let queue = &self.dev.queue;
            let layout = k.pipeline.get_bind_group_layout(0);
            for (l, s) in live.iter_mut().zip(staged) {
                for (id, b) in &s.bufs {
                    l.bufs[*id] = Some(Slot::Resident(b));
                }
            }
            let mut compute_ms = 0.0;
            // A segment: its encoder, its dispatches' (item, query) and its reads.
            struct Seg {
                enc: wgpu::CommandEncoder,
                queries: Vec<usize>,
                reads: Vec<(usize, BufId, u64, usize, u64, u64)>,
                keep: Vec<wgpu::Buffer>,
                any: bool,
            }
            let new_seg = || Seg { enc: device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("harness segment") }), queries: Vec::new(), reads: Vec::new(), keep: Vec::new(), any: false };
            // The query set is sized for the largest segment: count the
            // dispatches between syncs first.
            let mut most = 1usize;
            {
                let mut n = 0usize;
                for l in live.iter() {
                    for c in &l.plan.run {
                        match c {
                            Cmd::Dispatch { .. } => n += 1,
                            Cmd::Sync if !batch => {
                                most = most.max(n);
                                n = 0;
                            }
                            _ => {}
                        }
                    }
                    if !batch {
                        most = most.max(n);
                        n = 0;
                    }
                }
                most = most.max(n);
            }
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("harness timestamps"), ty: wgpu::QueryType::Timestamp, count: 2 * most as u32 });
            let stamps_bytes = 16 * most as u64;
            let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("harness resolve"), size: stamps_bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let stamps_read = device.create_buffer(&wgpu::BufferDescriptor { label: Some("harness stamps"), size: stamps_bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut seg = new_seg();
            let flush = |seg: Seg, live: &mut [Live], compute_ms: &mut f64| -> Result<(), String> {
                if !seg.any && seg.reads.is_empty() {
                    return Ok(());
                }
                let t = Instant::now();
                queue.submit(Some(seg.enc.finish()));
                wait_gpu(device, queue)?;
                if seg.any {
                    *compute_ms += t.elapsed().as_secs_f64() * 1e3;
                }
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("harness read-back") });
                let nq = 2 * seg.queries.len() as u32;
                if nq > 0 {
                    enc.resolve_query_set(&query_set, 0..nq, &resolve_buf, 0);
                    enc.copy_buffer_to_buffer(&resolve_buf, 0, &stamps_read, 0, 8 * u64::from(nq));
                }
                let total: u64 = seg.reads.iter().map(|r| r.5).sum();
                let staging = (total > 0).then(|| device.create_buffer(&wgpu::BufferDescriptor { label: Some("harness staging"), size: total, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
                let mut at = 0u64;
                for (item, src, so, _, _, n) in &seg.reads {
                    let b = live[*item].buf(*src, "a read")?;
                    enc.copy_buffer_to_buffer(b, *so, staging.as_ref().expect("staging for reads"), at, *n);
                    at += n;
                }
                queue.submit(Some(enc.finish()));
                let (tx, rx) = std::sync::mpsc::channel();
                let mut waits = 0;
                if let Some(st) = &staging {
                    st.slice(..).map_async(wgpu::MapMode::Read, { let tx = tx.clone(); move |r| { tx.send(r).expect("the receiver waits for the mapping"); } });
                    waits += 1;
                }
                if nq > 0 {
                    stamps_read.slice(..8 * u64::from(nq)).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).expect("the receiver waits for the mapping"); });
                    waits += 1;
                }
                wait_gpu(device, queue)?;
                for _ in 0..waits {
                    rx.recv().map_err(|e| format!("map: {e}"))?.map_err(|e| format!("map: {e}"))?;
                }
                if let Some(st) = &staging {
                    {
                        let mapped = st.slice(..).get_mapped_range();
                        let mut at = 0usize;
                        for (item, _, _, out, d_o, n) in &seg.reads {
                            live[*item].outputs[*out][*d_o as usize..(*d_o + *n) as usize].copy_from_slice(&mapped[at..at + *n as usize]);
                            at += *n as usize;
                        }
                    }
                    st.unmap();
                    st.destroy();
                }
                if nq > 0 {
                    {
                        let mapped = stamps_read.slice(..8 * u64::from(nq)).get_mapped_range();
                        let stamps: &[u64] = bytemuck::cast_slice(&mapped);
                        for (q, item) in seg.queries.iter().enumerate() {
                            live[*item].gpu_ms += stamps[2 * q + 1].wrapping_sub(stamps[2 * q]) as f64 * f64::from(k.period_ns) / 1e6;
                        }
                    }
                    stamps_read.unmap();
                }
                drop(seg.keep);
                Ok(())
            };
            for i in 0..live.len() {
                let run = live[i].plan.run.clone();
                for c in &run {
                    match c {
                        Cmd::Create(id) => {
                            let b = self.make(live[i].plan, *id, live[i].data)?;
                            live[i].bufs[*id] = Some(Slot::Own(b));
                        }
                        Cmd::Clear(id) => {
                            seg.enc.clear_buffer(live[i].buf(*id, "a clear")?, 0, None);
                            seg.any = true;
                        }
                        Cmd::Dispatch { bind, workgroups } => {
                            let entries: Vec<wgpu::BindGroupEntry> = bind.iter().map(|(b, id)| live[i].buf(*id, "a dispatch").map(|buf| wgpu::BindGroupEntry { binding: *b, resource: buf.as_entire_binding() })).collect::<Result<_, _>>()?;
                            let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("harness bind"), layout: &layout, entries: &entries });
                            let q = seg.queries.len() as u32;
                            seg.queries.push(i);
                            let timestamp_writes = Some(wgpu::ComputePassTimestampWrites { query_set: &query_set, beginning_of_pass_write_index: Some(2 * q), end_of_pass_write_index: Some(2 * q + 1) });
                            let mut pass = seg.enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("harness dispatch"), timestamp_writes });
                            pass.set_pipeline(&k.pipeline);
                            pass.set_bind_group(0, &group, &[]);
                            pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
                            seg.any = true;
                        }
                        Cmd::Copy { src, src_offset, dst, dst_offset, bytes } => {
                            let (s, d) = (live[i].buf(*src, "a copy")?, live[i].buf(*dst, "a copy")?);
                            seg.enc.copy_buffer_to_buffer(s, *src_offset, d, *dst_offset, *bytes);
                            seg.any = true;
                        }
                        Cmd::Read { src, src_offset, output, dst_offset, bytes } => {
                            if read {
                                seg.reads.push((i, *src, *src_offset, *output, *dst_offset, *bytes));
                            }
                        }
                        Cmd::Sync => {
                            if !batch {
                                let done = std::mem::replace(&mut seg, new_seg());
                                flush(done, live, &mut compute_ms)?;
                            }
                        }
                    }
                }
                if !batch {
                    let done = std::mem::replace(&mut seg, new_seg());
                    flush(done, live, &mut compute_ms)?;
                }
            }
            flush(seg, live, &mut compute_ms)?;
            // Run buffers go now; the residents stay with their stage.
            for l in live.iter_mut() {
                for b in l.bufs.iter_mut() {
                    if let Some(Slot::Own(b)) = b.take() {
                        b.destroy();
                    }
                }
            }
            device.poll(wgpu::Maintain::Poll);
            Ok(compute_ms)
        }
    }

    impl<'a> Executor for DeviceExecutor<'a> {
        type Kernel = DeviceKernel;
        type Staged = DeviceStaged;

        fn binding_limit(&self) -> u64 {
            self.limit
        }

        fn oracle_tag(&self) -> String {
            format!("device {}", self.dev.name)
        }

        fn compile(&self, problem: &Problem, source: &str) -> Result<(DeviceKernel, KernelInfo), String> {
            let info = kernel_info(&problem.spec, source)?;
            let dev = self.dev;
            if !dev.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
                return Err("this device has no timestamp queries".into());
            }
            dev.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let shader = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&problem.spec.name), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)) });
            // An explicit layout over the schema's bindings: a kernel that
            // reads none of them still binds them all (as `Gram::compile_with`).
            let entries: Vec<wgpu::BindGroupLayoutEntry> = problem
                .spec
                .binding
                .iter()
                .map(|b| {
                    let ty = match b.kind {
                        BindingKind::StorageRead => wgpu::BufferBindingType::Storage { read_only: true },
                        BindingKind::StorageReadWrite => wgpu::BufferBindingType::Storage { read_only: false },
                        BindingKind::Uniform => wgpu::BufferBindingType::Uniform,
                    };
                    wgpu::BindGroupLayoutEntry { binding: b.binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None }
                })
                .collect();
            let bgl = dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("harness"), entries: &entries });
            let layout = dev.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("harness"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
            let pipeline = dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("harness"), layout: Some(&layout), module: &shader, entry_point: &problem.spec.kernel.entry, compilation_options: Default::default() });
            if let Some(e) = pollster::block_on(dev.device.pop_error_scope()) {
                return Err(format!("pipeline: {e}"));
            }
            Ok((DeviceKernel { pipeline, period_ns: dev.queue.get_timestamp_period() }, info))
        }

        fn stage(&self, plan: &Plan, data: &CaseData) -> Result<(DeviceStaged, f64), String> {
            let t = Instant::now();
            let bufs = plan.setup.iter().map(|id| self.make(plan, *id, data).map(|b| (*id, b))).collect::<Result<Vec<_>, _>>()?;
            self.dev.queue.submit(None);
            wait_gpu(&self.dev.device, &self.dev.queue)?;
            Ok((DeviceStaged { bufs }, t.elapsed().as_secs_f64() * 1e3))
        }

        fn run(&self, k: &DeviceKernel, plan: &Plan, data: &CaseData, staged: &DeviceStaged, read: bool) -> Result<RunOut, String> {
            let t = Instant::now();
            let mut live = vec![Live { plan, data, bufs: (0..plan.buffers.len()).map(|_| None).collect(), outputs: plan.outputs.iter().map(|n| vec![0u8; *n as usize]).collect(), gpu_ms: 0.0 }];
            let compute_ms = self.exec(k, &mut live, &[staged], false, read)?;
            let l = live.pop().expect("one item");
            Ok(RunOut { outputs: if read { l.outputs } else { Vec::new() }, gpu_ms: l.gpu_ms, compute_ms, wall_ms: t.elapsed().as_secs_f64() * 1e3 })
        }

        /// THE TIMING REPEAT IN ONE SUBMISSION: up to `runs_a_submission`
        /// runs of a batch-safe plan encoded together and submitted once,
        /// nothing read; each run's own dispatches timed as `run` times them.
        /// A run alone pays its submission, wait and timestamp read-back
        /// (nluca at 10,000 rows: 0.5 ms of GPU in about 7 ms of wall).
        fn run_repeat(&self, k: &DeviceKernel, plan: &Plan, data: &CaseData, staged: &DeviceStaged, n: usize) -> Result<(f64, f64), String> {
            let (mut gpu, mut longest) = (0.0f64, 0.0f64);
            // A plan with a sync before its end splits its work into
            // submissions on purpose (kmeans's bands keep each under the
            // runaway guard); batching drops syncs, so it runs alone.
            let inner_sync = plan.run.iter().rev().skip_while(|c| !matches!(c, Cmd::Sync)).skip(1).any(|c| matches!(c, Cmd::Sync));
            if !plan.batch_safe || inner_sync || n <= 1 {
                for _ in 0..n {
                    let r = self.run(k, plan, data, staged, false)?;
                    gpu += r.gpu_ms;
                    longest = longest.max(r.compute_ms);
                }
                return Ok((gpu, longest));
            }
            // The first run alone measures the run, so a submission holds
            // about a second of work at most.
            let first = self.run(k, plan, data, staged, false)?;
            gpu += first.gpu_ms;
            longest = longest.max(first.compute_ms);
            let per = super::runs_a_submission(plan).min(((1000.0 / first.compute_ms.max(1e-3)) as usize).max(1));
            let mut left = n - 1;
            while left > 0 {
                let m = left.min(per);
                let mut live: Vec<Live> = (0..m).map(|_| Live { plan, data, bufs: (0..plan.buffers.len()).map(|_| None).collect(), outputs: plan.outputs.iter().map(|n| vec![0u8; *n as usize]).collect(), gpu_ms: 0.0 }).collect();
                let staged_all: Vec<&DeviceStaged> = vec![staged; m];
                let compute_ms = self.exec(k, &mut live, &staged_all, true, false)?;
                gpu += live.iter().map(|l| l.gpu_ms).sum::<f64>();
                longest = longest.max(compute_ms / m as f64);
                left -= m;
            }
            Ok((gpu, longest))
        }

        fn run_batch(&self, k: &DeviceKernel, items: &[(&Plan, &CaseData, &DeviceStaged)]) -> Result<(Vec<RunOut>, f64), String> {
            if items.iter().any(|(p, _, _)| !p.batch_safe) {
                return Err("a plan in the batch is not batch-safe".into());
            }
            let t = Instant::now();
            let mut live: Vec<Live> = items.iter().map(|(p, d, _)| Live { plan: p, data: d, bufs: (0..p.buffers.len()).map(|_| None).collect(), outputs: p.outputs.iter().map(|n| vec![0u8; *n as usize]).collect(), gpu_ms: 0.0 }).collect();
            let staged: Vec<&DeviceStaged> = items.iter().map(|(_, _, s)| *s).collect();
            let compute_ms = self.exec(k, &mut live, &staged, true, true)?;
            let wall = t.elapsed().as_secs_f64() * 1e3;
            Ok((live.into_iter().map(|l| RunOut { outputs: l.outputs, gpu_ms: l.gpu_ms, compute_ms, wall_ms: wall }).collect(), compute_ms))
        }
    }
}

// ------------------------------------------------------------------
// The context and the fitness
// ------------------------------------------------------------------

/// A staged case.
pub struct Staged<S> {
    pub case: Case,
    pub data: CaseData,
    pub staged: S,
    pub work: f64,
    /// The compared output's rows and columns, its comparison points, and
    /// the norm's tau for the case.
    pub dims: (usize, usize),
    pub points: usize,
    pub tau: f64,
    /// The reference kernel's norm value (accuracy cases).
    pub reference: f64,
    /// The constraint's pass bound (constraint cases).
    pub pass: f64,
    /// The case's environment (its parameters and derived values).
    pub env: Env,
}

/// An accuracy case with its norm and slack.
pub type AccuracyCase<S> = (Staged<S>, Norm, f64);

/// A problem staged on an executor: the fitness context.
pub struct Harness<'e, E: Executor> {
    pub problem: Problem,
    pub exec: &'e E,
    pub constraint: Vec<Staged<E::Staged>>,
    pub constraint_norm: Norm,
    pub chunk: usize,
    pub rungs: Vec<Staged<E::Staged>>,
    /// Accuracy cases by split, each with its norm and slack.
    pub accuracy: [Vec<AccuracyCase<E::Staged>>; 3],
    pub timing: Vec<Staged<E::Staged>>,
    pub timing_rounds: usize,
    pub base_budget_ms: f64,
    pub round_budget_ms: f64,
    /// The reference kernel's compute ms at the timing shapes, measured
    /// once (the slow limit).
    pub shipped_ms: [f64; 3],
    /// THE THREE SPEED COLUMNS AS BANDS of the timing cases (`timing_bands`):
    /// each column the geometric mean of its band's paired speed-ups; the
    /// band arrays (`shipped_ms`, the shipped GPU ms) its largest shape's.
    pub bands: [Vec<usize>; 3],
    /// The reference kernel's compute ms at each timing case.
    pub shape_ms: Vec<f64>,
    pub shipped_gpu_ms: std::cell::Cell<[f64; 3]>,
    pub shipped_min_gpu_ms: std::cell::Cell<[f64; 3]>,
    /// The reference kernel, compiled once: every timing round runs it
    /// beside the candidate.
    pub reference_kernel: (E::Kernel, KernelInfo),
    /// Runs a timing sample a shape: the least count whose reference GPU
    /// time reaches `min_sample_ms` (`time_shipped`).
    pub sample_runs: std::cell::RefCell<Vec<usize>>,
    /// The constraint suite is staged (`SuiteSpec::staged`).
    pub staged: bool,
    /// One whole evaluation of the reference kernel, measured when the
    /// problem's budget names a deadline.
    pub reference_eval_ms: Option<f64>,
    /// The reference kernel's compute ms on each chunk of a staged
    /// constraint, in the order the chunks run.
    pub reference_chunk_ms: Vec<f64>,
}

impl<'e, E: Executor> Harness<'e, E> {
    /// Stage every case, measure the reference kernel on the accuracy cases
    /// and at the timing shapes, as `gram::FitnessContext::new` does.
    pub fn new(problem: Problem, exec: &'e E) -> Result<Harness<'e, E>, String> {
        let limit = exec.binding_limit();
        let (shipped, info) = exec.compile(&problem, &problem.reference).map_err(|e| format!("the reference kernel: {e}"))?;
        let mut constraint = Vec::new();
        let (mut rungs, mut timing) = (Vec::new(), Vec::new());
        let mut accuracy: [Vec<AccuracyCase<E::Staged>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let (mut constraint_norm, mut chunk) = (Norm::MeanAbsOverMax, 16usize);
        for su in &problem.suites {
            let out = problem.output_of(su)?.clone();
            for c in &su.cases {
                let env = problem.env(c, limit)?;
                let mut data = problem.case_data(c, limit)?;
                let plan = problem.plan(c, limit, None)?;
                let (staged, _) = exec.stage(&plan, &data)?;
                if matches!(su.spec.role, Role::Constraint | Role::Accuracy) {
                    oracle_fill(&problem, exec, &shipped, &info, c, &mut data, &staged).map_err(|e| format!("the oracle, case {}: {e}", c.name))?;
                }
                let dims = problem.output_dims(&out, &env)?;
                let points = eval_int(&out.points, &env)? as usize;
                if points == 0 || dims.0 % points != 0 {
                    return Err(format!("case {}: {points} comparison points do not divide {} rows", c.name, dims.0));
                }
                let tau = match &su.spec.tau {
                    Some(t) => eval(t, &env)?,
                    None => 0.0,
                };
                let pass = match &su.spec.pass {
                    Some(p) => eval(p, &env)?,
                    None => 0.0,
                };
                let s = Staged { case: c.clone(), data, staged, work: problem.work(c, limit)?, dims, points, tau, reference: 0.0, pass, env };
                // (An exact or argmax output has no norm: its kind judges it.)
                let norm_of = |su: &Suite| su.spec.norm.unwrap_or(Norm::MeanAbsOverMax);
                match su.spec.role {
                    Role::Constraint => {
                        constraint_norm = norm_of(su);
                        chunk = su.spec.chunk.unwrap_or(16).max(1);
                        constraint.push(s);
                    }
                    Role::Rung => rungs.push(s),
                    Role::Timing => timing.push(s),
                    Role::Accuracy => {
                        let norm = norm_of(su);
                        let mut s = s;
                        let plan = problem.plan(&s.case, limit, Some(&info))?;
                        let r = exec.run(&shipped, &plan, &s.data, &s.staged, true)?;
                        s.reference = compare(&problem, &out, norm, &s, &r)?.0;
                        accuracy[su.spec.split.expect("checked at load")].push((s, norm, su.spec.slack.expect("checked at load")));
                    }
                }
            }
        }
        // The reference kernel's compute time at each timing shape, after
        // one warm-up run at the first, so the limit comes from this device.
        let run_at = |s: &Staged<E::Staged>, read: bool| -> Result<RunOut, String> { exec.run(&shipped, &problem.plan(&s.case, limit, Some(&info))?, &s.data, &s.staged, read) };
        run_at(&timing[0], true)?;
        let mut shape_ms = Vec::with_capacity(timing.len());
        for s in &timing {
            shape_ms.push(run_at(s, true)?.compute_ms);
        }
        let bands = timing_bands(&timing.iter().map(|s| s.work).collect::<Vec<_>>());
        let shipped_ms = band_max(&bands, &shape_ms);
        let round_budget_ms = problem.spec.budget.round_ms;
        let worst = shipped_ms.iter().copied().fold(0.0, f64::max);
        if slow_limit_ms(round_budget_ms, &shipped_ms) < worst {
            return Err(format!("the reference kernel takes {shipped_ms:?} ms at the timing shapes: a timing round of {round_budget_ms} ms cannot hold it and a kernel at its speed"));
        }
        // A staged constraint runs its cheapest cases first (a stable sort:
        // equal work keeps the manifest's order).
        let staged = problem.suites.iter().any(|s| s.spec.role == Role::Constraint && s.spec.staged);
        if staged {
            constraint.sort_by(|a, b| a.work.total_cmp(&b.work));
        }
        let timing_n = timing.len();
        let mut h = Harness {
            timing_rounds: problem.spec.budget.timing_rounds,
            base_budget_ms: problem.spec.budget.base_ms,
            round_budget_ms,
            problem,
            exec,
            constraint,
            constraint_norm,
            chunk,
            staged,
            rungs,
            accuracy,
            timing,
            shipped_ms,
            bands,
            shape_ms,
            shipped_gpu_ms: std::cell::Cell::new([0.0; 3]),
            shipped_min_gpu_ms: std::cell::Cell::new([f64::INFINITY; 3]),
            reference_kernel: (shipped, info),
            sample_runs: std::cell::RefCell::new(vec![1; timing_n]),
            reference_eval_ms: None,
            reference_chunk_ms: Vec::new(),
        };
        // STAGED: the shipped kernel's own time on every chunk of the
        // constraint, as the evaluation runs them (the staged gate's scale).
        if staged {
            let mut v = Vec::new();
            for chunk in h.constraint.chunks(h.chunk) {
                let plans = chunk.iter().map(|s| h.problem.plan(&s.case, limit, Some(&h.reference_kernel.1))).collect::<Result<Vec<_>, _>>()?;
                let items: Vec<(&Plan, &CaseData, &E::Staged)> = chunk.iter().zip(&plans).map(|(s, p)| (p, &s.data, &s.staged)).collect();
                v.push(exec.run_batch(&h.reference_kernel.0, &items)?.1);
            }
            h.reference_chunk_ms = v;
        }
        h.time_shipped()?;
        // The deadline's base: one whole evaluation of the reference kernel,
        // which must be feasible on its own problem.
        if h.problem.spec.budget.deadline_factor.is_some() {
            let reference = h.problem.reference.clone();
            let t = Instant::now();
            let f = h.evaluate(&reference);
            if !f.feasible {
                return Err(format!("the reference kernel is not feasible on its own problem: {}", f.detail));
            }
            h.reference_eval_ms = Some(t.elapsed().as_secs_f64() * 1e3);
        }
        Ok(h)
    }

    /// THE EVALUATION'S DEADLINE, the fit's watchdog for a true runaway:
    /// `deadline_factor` times the reference kernel's measured evaluation,
    /// at least `deadline_floor_ms` and never under what the evaluation's
    /// own budgets allow a kernel (the cases' base budget, then every timing
    /// round of the three shapes at the slow limit), so a slow kernel is
    /// scored "slow" by the gates before the watchdog could fire. None when
    /// the problem's budget names no deadline.
    pub fn deadline_ms(&self) -> Option<f64> {
        let (factor, floor) = (self.problem.spec.budget.deadline_factor?, self.problem.spec.budget.deadline_floor_ms?);
        Some((factor * self.reference_eval_ms?).max(floor).max(self.gated_ms()))
    }

    /// The six columns: the three speed-ups under the problem's
    /// `speed_column`, then the three accuracy columns as given.
    pub fn columns(&self, speedups: [f64; 3], accuracy: [f64; 3]) -> [f64; 6] {
        match self.problem.spec.speed_column {
            SpeedColumn::Reciprocal => fitness_columns(speedups, accuracy),
            s => [s.column(speedups[0]), s.column(speedups[1]), s.column(speedups[2]), accuracy[0], accuracy[1], accuracy[2]],
        }
    }

    /// The longest an evaluation's gates let it run: the correctness cases
    /// (and rungs and pairs) within the base budget, then the timing rounds,
    /// each shape at most the slow limit.
    pub fn gated_ms(&self) -> f64 {
        // A round is two samples of each kernel, a sample `sample_runs` runs.
        let runs = self.sample_runs.borrow().iter().copied().max().unwrap_or(1) as f64;
        let shipped = self.shape_ms.iter().copied().fold(0.0, f64::max);
        self.base_budget_ms + self.timing.len() as f64 * self.timing_rounds.max(1) as f64 * 2.0 * runs * (self.slow_limit_ms() + shipped)
    }

    /// The reference kernel at the timing shapes: one warm-up, then the
    /// median of `timing_rounds` GPU times a shape; the least so far per
    /// shape is reported (`gram::FitnessContext::time_shipped`), and the
    /// median sets the runs a timing sample takes (`sample_runs`).
    pub fn time_shipped(&self) -> Result<[f64; 3], String> {
        let limit = self.exec.binding_limit();
        let (k, info) = &self.reference_kernel;
        let run = |s: &Staged<E::Staged>| -> Result<RunOut, String> { self.exec.run(k, &self.problem.plan(&s.case, limit, Some(info))?, &s.data, &s.staged, false) };
        run(&self.timing[0])?;
        let mut per = Vec::with_capacity(self.timing.len());
        for (i, s) in self.timing.iter().enumerate() {
            let mut v = Vec::with_capacity(self.timing_rounds.max(1));
            for _ in 0..self.timing_rounds.max(1) {
                v.push(run(s)?.gpu_ms);
            }
            let m = median(&mut v);
            if m <= 0.0 {
                return Err(format!("the reference kernel timed at zero at shape {i}"));
            }
            per.push(m);
        }
        let out = band_max(&self.bands, &per);
        self.shipped_gpu_ms.set(out);
        self.shipped_min_gpu_ms.set(shipped_min(self.shipped_min_gpu_ms.get(), out));
        let least = self.problem.spec.budget.min_sample_ms;
        *self.sample_runs.borrow_mut() = per.iter().map(|ms| ((least / ms).ceil() as usize).max(1)).collect();
        Ok(out)
    }

    /// One timing sample of a kernel at timing shape `i`: `sample_runs`
    /// runs, their GPU ms summed, and the longest run's compute ms (the
    /// staged slow gate's measure).
    fn sample(&self, k: &E::Kernel, plan: &Plan, i: usize) -> Result<(f64, f64), String> {
        let s = &self.timing[i];
        let n = self.sample_runs.borrow()[i];
        self.exec.run_repeat(k, plan, &s.data, &s.staged, n)
    }

    /// THE PAIRED SPEED-UP of a kernel at timing shape `i`: `timing_rounds`
    /// rounds, each the reference and the kernel twice in ABBA order (R K K R
    /// on even rounds, K R R K on odd), the round's ratio the reference's two
    /// samples over the kernel's; the median ratio, and the kernel's median
    /// sample compute ms. The GPU's speed drifts up to 2x between minutes
    /// (nluca, 2026-10-14: the shipped kernel 0.65 then 1.45 ms at 1,000
    /// rows), so a ratio against a time taken earlier ranks noise; timed
    /// side by side, the drift divides out. `slow` is the staged gate: a
    /// kernel run whose compute ms passes it ends the timing as slow.
    fn paired_speedup(&self, k: &E::Kernel, plan: &Plan, ref_plan: &Plan, i: usize, slow: Option<f64>) -> Result<(f64, f64, f64), String> {
        let rounds = self.timing_rounds.max(1);
        let (mut ratios, mut walls) = (Vec::with_capacity(rounds), Vec::with_capacity(rounds));
        let mut cand_ms = 0.0f64;
        for round in 0..rounds {
            let (mut r_ms, mut k_ms) = (0.0f64, 0.0f64);
            for slot in 0..4 {
                let reference = (slot == 0 || slot == 3) == (round % 2 == 0);
                if reference {
                    r_ms += self.sample(&self.reference_kernel.0, ref_plan, i)?.0;
                } else {
                    let t = Instant::now();
                    let (gpu, longest) = self.sample(k, plan, i)?;
                    cand_ms += t.elapsed().as_secs_f64() * 1e3;
                    if let Some(limit) = slow {
                        if longest > limit {
                            return Err(format!("slow: measured {:.1} s at the timing shape {}, over the limit of {:.1} s a build", longest / 1e3, self.timing[i].case.label, limit / 1e3));
                        }
                    }
                    k_ms += gpu;
                    walls.push(longest);
                }
            }
            ratios.push(speedup(r_ms, k_ms)?);
        }
        Ok((median(&mut ratios), median(&mut walls), cand_ms))
    }

    pub fn shipped_reference(&self) -> [f64; 3] {
        self.shipped_min_gpu_ms.get()
    }

    pub fn slow_limit_ms(&self) -> f64 {
        slow_limit_ms(self.round_budget_ms, &self.shipped_ms)
    }

    /// Re-time a kernel scored feasible: its timing runs only, against the
    /// least reference times; its accuracy columns kept (`gram::FitnessContext::retime`).
    pub fn retime(&self, source: &str, columns: &[f64; 6]) -> Result<([f64; 3], [f64; 6], f64), String> {
        let limit = self.exec.binding_limit();
        let (k, info) = self.exec.compile(&self.problem, source)?;
        let mut per = Vec::with_capacity(self.timing.len());
        for s in &self.timing {
            let plan = self.problem.plan(&s.case, limit, Some(&info))?;
            let ref_plan = self.problem.plan(&s.case, limit, Some(&self.reference_kernel.1))?;
            per.push(self.paired_speedup(&k, &plan, &ref_plan, per.len(), None)?.0);
        }
        let speedups = band_geomean(&self.bands, &per);
        let c = self.columns(speedups, [columns[3], columns[4], columns[5]]);
        Ok((speedups, c, truenorth(&c)))
    }

    /// THE FITNESS of one kernel against the problem: compile, the
    /// constraint in chunks, the rungs, the accuracy cases smallest work
    /// first, the timing shapes predicted then timed, the six columns and
    /// the HFF angle; an infeasible kernel's graded distance and closeness
    /// columns. The procedure of `gram::fitness_with`, read from the data.
    pub fn evaluate(&self, source: &str) -> Fitness {
        let t = Instant::now();
        let mut meter = Meter::default();
        let mut split = Split::default();
        let infeasible = |passed: usize, distance: f64, closeness: [f64; closeness::N], detail: String, t: Instant, meter: &Meter, split: &Split| Fitness { feasible: false, passed, distance, closeness, closeness_hff: closeness::angle(&closeness), columns: [1.0; 6], speedups: [0.0; 3], hff: f64::INFINITY, detail, seconds: t.elapsed().as_secs_f64(), predicted_ms: [0.0; 3], rate_per_g: meter.rate_per_g(), floor_ms: meter.floor_ms, measured_ms: [0.0; 3], split: Split { total: t.elapsed().as_secs_f64() * 1e3, ..*split } };
        let elapsed_ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
        let limit = self.exec.binding_limit();
        let compiled = self.exec.compile(&self.problem, source);
        split.compile = elapsed_ms(t);
        let (k, info) = match compiled {
            Ok(x) => x,
            Err(e) => return infeasible(0, f64::INFINITY, closeness::WORST, format!("compile: {}", e.lines().find(|l| l.contains("error") || l.contains("header")).unwrap_or(e.lines().next().unwrap_or(""))), t, &meter, &split),
        };
        let plan_of = |s: &Staged<E::Staged>| self.problem.plan(&s.case, limit, Some(&info));
        let out_spec = match self.problem.suites.iter().find(|s| s.spec.role == Role::Constraint).map(|s| self.problem.output_of(s)) {
            Some(Ok(o)) => o.clone(),
            _ => return infeasible(0, f64::INFINITY, closeness::WORST, "our software has a bug: the problem has no constraint output".into(), t, &meter, &split),
        };
        let out_index = self.problem.spec.output.iter().position(|o| o.name == out_spec.name).expect("the output is the problem's");
        let mut passed = 0;
        let mut first_fail = String::new();
        let (mut sum_distance, mut ran) = (0.0f64, 0usize);
        let mean_distance = |sum: f64, ran: usize| if ran == 0 { f64::INFINITY } else { sum / ran as f64 };
        let cases = self.constraint.len();
        let mut per_case: Vec<Option<[f64; closeness::N]>> = vec![None; cases];
        let mut outs: Vec<Option<(u64, u64, usize)>> = vec![None; cases];
        let t_cases = Instant::now();
        let (mut done, mut cases_ms) = (0usize, 0.0f64);
        for chunk in self.constraint.chunks(self.chunk) {
            // STAGED: a failing chunk ends the cases; a chunk the rate measured
            // so far predicts past the base budget is not dispatched.
            if self.staged && done > 0 {
                if passed < done {
                    return infeasible(passed, mean_distance(sum_distance, ran), closeness::columns(&per_case, passed, &outs), format!("constraint {passed}/{cases} cases: {first_fail}; the {} costlier cases not dispatched", cases - done), t, &meter, &split);
                }
                let what = format!("the correctness cases {}..{} ({})", done + 1, done + chunk.len(), chunk.iter().map(|s| s.case.label.as_str()).collect::<Vec<_>>().join(", "));
                // The prediction is the kernel's time so far relative to the
                // shipped kernel's on the same chunks, times the shipped
                // kernel's time on this one: the shipped kernel's own cost
                // curve, fixed overheads included (a rate per unit of work
                // through zero, fitted on a tiny first case, put the whole
                // dispatch overhead on its work: kmeans_assign's n 1 case
                // predicted 481 s for a 0.5 s case). The time spent is the
                // executor's own measure of the chunks run.
                let k = done / self.chunk;
                let (ref_done, ref_next) = (self.reference_chunk_ms[..k].iter().sum::<f64>(), self.reference_chunk_ms[k]);
                let predicted = if ref_done > 0.0 { cases_ms / ref_done * ref_next } else { cases_ms };
                if cases_ms + predicted > self.base_budget_ms {
                    return infeasible(passed, mean_distance(sum_distance, ran), closeness::columns(&per_case, passed, &outs), format!("slow: predicted {:.1} s at {what}, {:.2}x the shipped kernel's {:.1} s there by the cases run so far ({:.1} s against its {:.1} s), over the cases' budget of {:.1} s; not dispatched", predicted / 1e3, if ref_done > 0.0 { cases_ms / ref_done } else { 1.0 }, ref_next / 1e3, cases_ms / 1e3, ref_done / 1e3, self.base_budget_ms / 1e3), t, &meter, &split);
                }
            }
            let plans = match chunk.iter().map(plan_of).collect::<Result<Vec<_>, _>>() {
                Ok(p) => p,
                Err(e) => return infeasible(passed, mean_distance(sum_distance, ran), closeness::columns(&per_case, passed, &outs), format!("run: {e}"), t, &meter, &split),
            };
            let items: Vec<(&Plan, &CaseData, &E::Staged)> = chunk.iter().zip(&plans).map(|(s, p)| (p, &s.data, &s.staged)).collect();
            let (runs, compute_ms) = match self.exec.run_batch(&k, &items) {
                Ok(x) => x,
                Err(e) => return infeasible(passed, mean_distance(sum_distance, ran), closeness::columns(&per_case, passed, &outs), format!("run: {e}"), t, &meter, &split),
            };
            cases_ms += compute_ms;
            for (s, r) in chunk.iter().zip(&runs) {
                meter.push(s.work, compute_ms / chunk.len() as f64);
                let want = &s.data.expected[&out_spec.name];
                let (d, nf, h) = match judge(&self.problem, &out_spec, self.constraint_norm, s, &r.outputs[out_index]) {
                    Ok(x) => x,
                    Err(e) => return infeasible(passed, mean_distance(sum_distance, ran), closeness::columns(&per_case, passed, &outs), format!("our software has a bug: {e}"), t, &meter, &split),
                };
                sum_distance += if d.is_nan() { f64::INFINITY } else { d };
                per_case[ran] = Some(closeness::tensor_columns(&h, want, s.dims.0, s.dims.1));
                outs[ran] = Some((closeness::hash_bits(&h, |v: f32| u64::from(v.to_bits())), closeness::hash_bits(want, f64::to_bits), s.dims.0 * s.dims.1));
                ran += 1;
                if nf == 0 && d <= s.pass {
                    passed += 1;
                } else if first_fail.is_empty() {
                    first_fail = format!("{} ({d:.2e})", s.case.name);
                }
            }
            done += chunk.len();
            let projected = elapsed_ms(t_cases) * cases as f64 / done as f64;
            // (A staged suite is gated by work before each chunk instead: its
            // cases grow, so a projection by count would undercount the rest.)
            if !self.staged && done < cases && projected > self.base_budget_ms / 3.0 {
                return infeasible(passed, mean_distance(sum_distance, ran), closeness::columns(&per_case, passed, &outs), format!("slow: the correctness cases are projected at {:.1} s ({:.1} s for the first {done}), over a third of the {:.1} s budget; the rest not dispatched", projected / 1e3, elapsed_ms(t_cases) / 1e3, self.base_budget_ms / 1e3), t, &meter, &split);
            }
        }
        split.cases = elapsed_ms(t_cases);
        if passed < cases {
            return infeasible(passed, mean_distance(sum_distance, ran), closeness::columns(&per_case, passed, &outs), format!("constraint {passed}/{cases} cases: {first_fail}"), t, &meter, &split);
        }
        meter.fix_floor();
        let limit_ms = self.slow_limit_ms();
        let t_rungs = Instant::now();
        for s in &self.rungs {
            if let Err(e) = gate(&meter, &format!("the timing rung {}", s.case.label), s.work, elapsed_ms(t), limit_ms, self.base_budget_ms) {
                return infeasible(passed, f64::INFINITY, closeness::columns(&per_case, passed, &outs), e, t, &meter, &split);
            }
            match plan_of(s).and_then(|p| self.exec.run(&k, &p, &s.data, &s.staged, false)) {
                Ok(r) => {
                    meter.push(s.work, r.compute_ms);
                    split.build(&r.build());
                }
                Err(e) => return infeasible(passed, f64::INFINITY, closeness::columns(&per_case, passed, &outs), format!("run: {e}"), t, &meter, &split),
            }
        }
        split.rungs = elapsed_ms(t_rungs);
        let mut order: Vec<(usize, &AccuracyCase<E::Staged>)> = self.accuracy.iter().enumerate().flat_map(|(k, set)| set.iter().map(move |x| (k, x))).collect();
        order.sort_by(|a, b| a.1 .0.work.total_cmp(&b.1 .0.work));
        let mut worst = [0.0f64; 3];
        for (sp, (s, norm, slack)) in order {
            if let Err(e) = gate(&meter, &format!("pair {} ({})", s.case.name, s.case.label), s.work, elapsed_ms(t), limit_ms, self.base_budget_ms) {
                return infeasible(passed, f64::INFINITY, closeness::columns(&per_case, passed, &outs), e, t, &meter, &split);
            }
            let t_pb = Instant::now();
            let r = match plan_of(s).and_then(|p| self.exec.run(&k, &p, &s.data, &s.staged, true)) {
                Ok(r) => r,
                Err(e) => return infeasible(passed, f64::INFINITY, closeness::columns(&per_case, passed, &outs), format!("run: {e}"), t, &meter, &split),
            };
            split.pairs_build += elapsed_ms(t_pb);
            split.build(&r.build());
            meter.push(s.work, r.compute_ms);
            let t_pc = Instant::now();
            let suite_out = &self.problem.spec.output[out_index];
            let (v, non_finite, h) = match compare(&self.problem, suite_out, *norm, s, &r) {
                Ok(x) => x,
                Err(e) => return infeasible(passed, f64::INFINITY, closeness::columns(&per_case, passed, &outs), format!("our software has a bug: {e}"), t, &meter, &split),
            };
            split.pairs_check += elapsed_ms(t_pc);
            let smax = s.reference;
            if non_finite > 0 || v > (slack * smax).max(1.0) {
                let over = if non_finite > 0 { f64::INFINITY } else { v / (slack * smax).max(1.0) };
                let mut pair_closeness = closeness::columns(&per_case, passed, &outs);
                let pair = closeness::tensor_columns(&h, &s.data.expected[&suite_out.name], s.dims.0, s.dims.1);
                for (c, x) in pair_closeness.iter_mut().zip(pair).take(9).skip(1) {
                    *c = (*c + x) / 2.0;
                }
                return infeasible(passed, over, pair_closeness, format!("pair {}: max {v:.3}{} (shipped {smax:.3}), non-finite {non_finite}", s.case.name, norm.unit()), t, &meter, &split);
            }
            worst[sp] = worst[sp].max(if smax > 0.0 { v / smax } else { v });
        }
        let slacks: [f64; 3] = std::array::from_fn(|k| self.accuracy[k].first().map_or(2.0, |x| x.2));
        let mut predicted_shape = Vec::with_capacity(self.timing.len());
        for s in &self.timing {
            match gate(&meter, &format!("the timing shape {}", s.case.label), s.work, 0.0, limit_ms, f64::INFINITY) {
                Ok(p) => predicted_shape.push(p),
                Err(e) => return Fitness { predicted_ms: band_max(&self.bands, &predicted_shape), ..infeasible(passed, f64::INFINITY, closeness::columns(&per_case, passed, &outs), e, t, &meter, &split) },
            }
        }
        let predicted_ms = band_max(&self.bands, &predicted_shape);
        let (mut per, mut walls) = (Vec::with_capacity(self.timing.len()), Vec::with_capacity(self.timing.len()));
        let slow = if self.staged { Some(limit_ms) } else { None };
        for (i, s) in self.timing.iter().enumerate() {
            let fail = |e: String| Fitness { predicted_ms, ..infeasible(passed, f64::INFINITY, closeness::columns(&per_case, passed, &outs), e, t, &meter, &split) };
            let plan = match plan_of(s) {
                Ok(p) => p,
                Err(e) => return fail(e),
            };
            let ref_plan = match self.problem.plan(&s.case, limit, Some(&self.reference_kernel.1)) {
                Ok(p) => p,
                Err(e) => return fail(e),
            };
            match self.paired_speedup(&k, &plan, &ref_plan, i, slow) {
                Ok((x, wall, cand_ms)) => {
                    per.push(x);
                    walls.push(wall);
                    split.timing_cand += cand_ms;
                }
                Err(e) if e.starts_with("slow:") => return fail(format!("{e} (predicted {:.1} s)", predicted_shape[i] / 1e3)),
                Err(e) => return fail(format!("{e} at {}", s.case.label)),
            }
        }
        let (speedups, measured_ms) = (band_geomean(&self.bands, &per), band_max(&self.bands, &walls));
        let [et, ev, ex] = worst;
        let columns = self.columns(speedups, [(et / slacks[0]).clamp(0.0, 1.0), (ev / slacks[1]).clamp(0.0, 1.0), (ex / slacks[2]).clamp(0.0, 1.0)]);
        Fitness { feasible: true, passed, distance: 0.0, closeness: [0.0; closeness::N], closeness_hff: 0.0, columns, speedups, hff: truenorth(&columns), detail: String::new(), seconds: t.elapsed().as_secs_f64(), predicted_ms, rate_per_g: meter.rate_per_g(), floor_ms: meter.floor_ms, measured_ms, split: Split { total: t.elapsed().as_secs_f64() * 1e3, ..split } }
    }
}

/// Little-endian bytes as f32.
pub fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect()
}

/// The norm over every comparison point: the worst point's value, the
/// non-finite counts summed. One point is the whole tensor.
pub fn points_value(norm: Norm, h: &[f32], want: &[f64], dims: (usize, usize), points: usize, tau: f64) -> (f64, usize) {
    if points <= 1 {
        return norm_value(norm, h, want, dims.0, dims.1, tau);
    }
    let rows = dims.0 / points;
    let n = rows * dims.1;
    let (mut worst, mut nf) = (0.0f64, 0usize);
    for p in 0..points {
        let (v, k) = norm_value(norm, &h[p * n..(p + 1) * n], &want[p * n..(p + 1) * n], rows, dims.1, tau);
        worst = if v.is_nan() || worst.is_nan() { f64::NAN } else { worst.max(v) };
        nf += k;
    }
    (worst, nf)
}

fn compare<S>(problem: &Problem, out: &OutputSpec, norm: Norm, s: &Staged<S>, r: &RunOut) -> Result<(f64, usize, Vec<f32>), String> {
    let i = problem.spec.output.iter().position(|o| o.name == out.name).ok_or("no such output")?;
    judge(problem, out, norm, s, r.outputs.get(i).ok_or("the run read no output")?)
}

/// THE COMPARISON BY THE OUTPUT'S KIND of a run's output bytes against the
/// case's expected values: the value the suite's bound judges, the
/// non-finite count, and the output as f32 (the closeness columns' input).
/// `float`: the norm over the comparison points. `exact`: the fraction of
/// elements that differ. `argmax`: the fraction of elements that differ and
/// that the tie rule does not accept (`margin`: an index in [0, classes)
/// whose score is within the margin of the expected index's, both scores
/// the output's `score` expression in f64 over the case's inputs, with `i`
/// the element and `j` the index).
pub fn judge<S>(problem: &Problem, out: &OutputSpec, norm: Norm, s: &Staged<S>, bytes: &[u8]) -> Result<(f64, usize, Vec<f32>), String> {
    let want = s.data.expected.get(&out.name).ok_or_else(|| format!("case {}: no expected {}", s.case.name, out.name))?;
    if out.kind() == OutputKind::Float {
        let h = bytemuck_f32(bytes);
        let (v, nf) = points_value(norm, &h, want, s.dims, s.points, s.tau);
        return Ok((v, nf, h));
    }
    let got = out.element.decode(bytes);
    if got.len() != want.len() {
        return Err(format!("case {}: {} holds {} values, expected {}", s.case.name, out.name, got.len(), want.len()));
    }
    let h: Vec<f32> = got.iter().map(|v| *v as f32).collect();
    let n = got.len().max(1) as f64;
    let mismatched: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if mismatched.is_empty() || out.kind() == OutputKind::Exact || out.tie == Some(Tie::Exact) {
        return Ok((mismatched.len() as f64 / n, 0, h));
    }
    // The margin rule: the case's inputs as f64 tensors, the scores of both indices.
    let mut tensors = Tensors::new();
    for i in &problem.spec.input {
        if let Some(b) = s.data.inputs.get(&i.name) {
            tensors.insert(i.name.clone(), i.element.decode(b));
        }
    }
    let ex = |e: &Option<String>| parse_expr(e.as_deref().ok_or("checked at load")?);
    let (score, margin, classes) = (ex(&out.score)?, ex(&out.margin)?, ex(&out.classes)?);
    let (margin, classes) = (margin.eval(&s.env)?, classes.eval(&s.env)?);
    let mut env = s.env.clone();
    let mut wrong = 0usize;
    for i in mismatched {
        let g = got[i];
        if g.fract() != 0.0 || g < 0.0 || g >= classes {
            wrong += 1;
            continue;
        }
        env.insert("i".into(), i as f64);
        env.insert("j".into(), g);
        let sg = score.eval_in(&env, &tensors)?;
        env.insert("j".into(), want[i]);
        let sw = score.eval_in(&env, &tensors)?;
        if sw - sg > margin {
            wrong += 1;
        }
    }
    Ok((wrong as f64 / n, 0, h))
}

/// The expected outputs of a judged case its manifest gives no file for:
/// THE SHIPPED KERNEL'S OWN OUTPUTS on the case's inputs, run on the
/// harness's executor (the stage already made) or on the CPU executor as
/// `[oracle] on` says, and cached under `[oracle] cache` by a key over the
/// shipped text, the executor, the case's name and parameters, its inputs'
/// bytes and the binding limit.
fn oracle_fill<E: Executor>(problem: &Problem, exec: &E, shipped: &E::Kernel, info: &KernelInfo, case: &Case, data: &mut CaseData, staged: &E::Staged) -> Result<(), String> {
    let missing: Vec<usize> = problem.spec.output.iter().enumerate().filter(|(_, o)| !data.expected.contains_key(&o.name)).map(|(k, _)| k).collect();
    if missing.is_empty() {
        return Ok(());
    }
    let oracle = &problem.spec.oracle;
    if oracle.mode == OracleMode::Reference {
        return Err(format!("no expected {} and the oracle mode is reference", problem.spec.output[missing[0]].name));
    }
    let limit = exec.binding_limit();
    let tag = match oracle.on {
        OracleOn::Executor => exec.oracle_tag(),
        OracleOn::Cpu => "cpu".to_string(),
    };
    let mut key = fnv(problem.reference.as_bytes(), FNV0);
    key = fnv(tag.as_bytes(), key);
    key = fnv(case.name.as_bytes(), key);
    for (k, v) in &case.params {
        key = fnv(&v.to_le_bytes(), fnv(k.as_bytes(), key));
    }
    for (k, b) in &data.inputs {
        key = fnv(b, fnv(k.as_bytes(), key));
    }
    key = fnv(&limit.to_le_bytes(), key);
    let dir = problem.dir.join(&oracle.cache);
    let path = |o: &OutputSpec| dir.join(format!("{key:016x}.{}.bin", o.name));
    let env = problem.env(case, limit)?;
    let want_bytes = |o: &OutputSpec| -> Result<u64, String> { Ok(problem.output_elems(o, &env)? * o.element.size()) };
    let cached: Option<Vec<Vec<u8>>> = missing.iter().map(|k| std::fs::read(path(&problem.spec.output[*k])).ok()).collect();
    let outputs: Vec<Vec<u8>> = match cached {
        Some(c) => c,
        None => {
            let r = match oracle.on {
                OracleOn::Executor => exec.run(shipped, &problem.plan(case, limit, Some(info))?, data, staged, true)?,
                OracleOn::Cpu => {
                    let cpu = CpuExecutor::new(limit);
                    let (k, i) = cpu.compile(problem, &problem.reference).map_err(|e| format!("the CPU executor cannot compile the shipped kernel: {e}"))?;
                    let plan = problem.plan(case, limit, Some(&i))?;
                    let (st, _) = cpu.stage(&plan, data)?;
                    cpu.run(&k, &plan, data, &st, true).map_err(|e| format!("the CPU executor cannot run the shipped kernel: {e}"))?
                }
            };
            let outs: Vec<Vec<u8>> = missing.iter().map(|k| r.outputs.get(*k).cloned().ok_or("the shipped run read no output")).collect::<Result<_, _>>()?;
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            for (k, b) in missing.iter().zip(&outs) {
                let f = path(&problem.spec.output[*k]);
                std::fs::write(&f, b).map_err(|e| format!("{}: {e}", f.display()))?;
            }
            outs
        }
    };
    for (k, b) in missing.iter().zip(outputs) {
        let o = &problem.spec.output[*k];
        if b.len() as u64 != want_bytes(o)? {
            return Err(format!("the oracle's {} holds {} bytes, the output {}", o.name, b.len(), want_bytes(o)?));
        }
        data.expected.insert(o.name.clone(), o.element.decode(&b));
    }
    Ok(())
}

// ------------------------------------------------------------------
// Writing cases
// ------------------------------------------------------------------

/// One case to write: its name, label, parameters, and per tensor the
/// little-endian bytes of its input and of its expected output's files.
pub struct CaseOut {
    pub name: String,
    pub label: Option<String>,
    pub params: Vec<(String, f64)>,
    pub inputs: Vec<(String, Vec<u8>)>,
    pub expected: Vec<(String, Vec<Vec<u8>>)>,
}

/// Write a suite: `cases/<suite>.toml` and its files under `cases/<suite>/`
/// (`NNNN.<tensor>.bin`, `NNNN.<tensor>.<k>.bin` for an expected output's
/// k-th file). Returns the bytes written.
pub fn write_suite(problem_dir: &Path, suite: &str, cases: &[CaseOut]) -> Result<u64, String> {
    let cdir = problem_dir.join("cases");
    let fdir = cdir.join(suite);
    std::fs::create_dir_all(&fdir).map_err(|e| format!("{}: {e}", fdir.display()))?;
    let mut total = 0u64;
    let mut manifest = Manifest { root: Some(suite.to_string()), case: Vec::new() };
    for (i, c) in cases.iter().enumerate() {
        let mut inputs = BTreeMap::new();
        for (t, bytes) in &c.inputs {
            let f = format!("{i:04}.{t}.bin");
            std::fs::write(fdir.join(&f), bytes).map_err(|e| format!("{f}: {e}"))?;
            total += bytes.len() as u64;
            inputs.insert(t.clone(), f);
        }
        let mut expected = BTreeMap::new();
        for (t, files) in &c.expected {
            let mut names = Vec::new();
            for (k, bytes) in files.iter().enumerate() {
                let f = if files.len() == 1 { format!("{i:04}.{t}.expected.bin") } else { format!("{i:04}.{t}.expected.{k}.bin") };
                std::fs::write(fdir.join(&f), bytes).map_err(|e| format!("{f}: {e}"))?;
                total += bytes.len() as u64;
                names.push(f);
            }
            expected.insert(t.clone(), names);
        }
        manifest.case.push(CaseSpec { name: c.name.clone(), label: c.label.clone(), dir: None, params: c.params.iter().cloned().collect(), inputs, expected });
    }
    let text = toml::to_string(&manifest).map_err(|e| format!("{suite}: {e}"))?;
    let path = cdir.join(format!("{suite}.toml"));
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(total)
}

/// f32 values as little-endian bytes.
pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// f64 values as little-endian bytes.
pub fn f64_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The speed bands: three or fewer cases in manifest order (the nearest
    /// filling an empty band), more sorted by work and cut into thirds; a
    /// band's speed-up the geometric mean of its cases', one case exactly.
    #[test]
    fn timing_cases_fill_the_three_speed_bands() {
        assert_eq!(timing_bands(&[5.0]), [vec![0], vec![0], vec![0]]);
        assert_eq!(timing_bands(&[5.0, 1.0]), [vec![0], vec![0], vec![1]]);
        assert_eq!(timing_bands(&[9.0, 1.0, 5.0]), [vec![0], vec![1], vec![2]]);
        assert_eq!(timing_bands(&[9.0, 1.0, 5.0, 3.0, 7.0]), [vec![1], vec![3, 2], vec![4, 0]]);
        assert_eq!(timing_bands(&[1.0; 7]), [vec![0, 1], vec![2, 3], vec![4, 5, 6]]);
        let bands = timing_bands(&[9.0, 1.0, 5.0, 3.0, 7.0]);
        let g = band_geomean(&bands, &[2.0, 1.1, 4.0, 1.0, 8.0]);
        assert_eq!(g[0].to_bits(), 1.1f64.to_bits());
        assert!((g[1] - 2.0).abs() < 1e-12 && (g[2] - 4.0).abs() < 1e-12, "{g:?}");
        assert_eq!(band_max(&bands, &[2.0, 1.1, 4.0, 1.0, 8.0]), [1.1, 4.0, 8.0]);
    }
    use crate::wgsl::gram::{self, AccuracyPair, Batch, DriverStep, Kind, Which, SHIPPED};
    use crate::wgsl::render::Grid;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    /// Blue World's per-binding ceiling on the M3 Max (`gram::effective_buffer_limit`: 256 MiB / 2).
    const LIMIT: u64 = 128 * 1024 * 1024;

    #[test]
    fn expressions_parse_evaluate_and_divide_exactly() {
        let mut env = Env::new();
        env.insert("d".into(), 40.0);
        env.insert("m".into(), 50.0);
        assert_eq!(eval("d * d * m", &env).unwrap(), 80_000.0);
        assert_eq!(eval("ceil_div(m, 16) + floor_div(m, 16) * 10", &env).unwrap(), 34.0);
        assert_eq!(eval("min(max(floor_div(18446744073709551616, d * 4), 1), m)", &env).unwrap(), 50.0);
        assert_eq!(eval("select(d > m, 1, 2) + (d <= m) + (d == 40) * 10 + -m", &env).unwrap(), 2.0 + 1.0 + 10.0 - 50.0);
        assert_eq!(eval("m % 7", &env).unwrap(), 1.0);
        assert_eq!(eval("4 * eps_f32 * sqrt(m)", &env).unwrap(), 4.0 * f64::from(f32::EPSILON) * 50f64.sqrt());
        assert_eq!(template("phi{d - 1}_s{(m + 1) % 2}", &env).unwrap(), "phi39_s1");
        assert!(eval("floor_div(d, 0)", &env).unwrap_err().contains("by zero"));
        assert!(eval("q + 1", &env).unwrap_err().contains("unknown name"));
        assert!(eval("ceil_div(1.5, 2)", &env).unwrap_err().contains("not a non-negative integer"));
        assert!(parse_expr("(d").is_err() && parse_expr("d d").is_err());
        let mut e = env.clone();
        bind_lets(&["a = d + 1".into(), "b = a * 2".into()], &mut e).unwrap();
        assert_eq!(e["b"], 82.0);
    }

    fn gram_problem() -> Problem {
        Problem::load(&root().join("problems/gram")).expect("problems/gram")
    }

    /// The harness's plan of a Gram case, read back as the driver's steps.
    fn as_driver_steps(p: &Plan, d: usize) -> Vec<DriverStep> {
        let row = d as u64 * 4;
        let mut out = Vec::new();
        for (band, id) in p.setup.iter().enumerate() {
            let b = &p.buffers[*id];
            assert_eq!(b.name, format!("phi{band}"));
            let Init::Input { tensor, byte_offset, set } = &b.init else { panic!("{} is not an input slice", b.name) };
            assert!(tensor == "phi" && set.is_empty() && byte_offset.is_multiple_of(row) && b.bytes.is_multiple_of(row));
            out.push(DriverStep::Upload { band, start: (byte_offset / row) as usize, end: ((byte_offset + b.bytes) / row) as usize });
        }
        let mut acc = None;
        for c in &p.run {
            match c {
                Cmd::Create(_) => {}
                Cmd::Clear(id) => {
                    acc = Some(*id);
                    out.push(DriverStep::Clear { rows: (p.buffers[*id].bytes / row) as usize });
                }
                Cmd::Dispatch { bind, workgroups } => {
                    assert_eq!(bind.len(), 3);
                    assert_eq!(bind[1], (1, acc.expect("a dispatch before its accumulator")), "the dispatch binds the band's accumulator");
                    let band = p.setup.iter().position(|id| *id == bind[0].1).expect("phi is resident");
                    let Init::Words(w) = &p.buffers[bind[2].1].init else { panic!("cfg is not a uniform") };
                    let cfg: Vec<u32> = w.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                    assert_eq!(workgroups[2], 1);
                    out.push(DriverStep::Dispatch { band, cfg: [cfg[0], cfg[1], cfg[2], cfg[3]], workgroups: (workgroups[0], workgroups[1]) });
                }
                Cmd::Sync => out.push(DriverStep::Sync),
                Cmd::Read { src, src_offset, output, dst_offset, bytes } => {
                    assert!(Some(*src) == acc && *src_offset == 0 && *output == 0 && *bytes == p.buffers[*src].bytes);
                    out.push(DriverStep::Read { r_off: (dst_offset / row) as usize, rows: (bytes / row) as usize });
                }
                Cmd::Copy { .. } => panic!("the Gram program has no copy"),
            }
        }
        out
    }

    fn info_of(g: Grid) -> KernelInfo {
        KernelInfo { workgroup: [g.x, g.y, 1], bx: g.bx, by: g.by, rows_x: g.rows_x, modes: Vec::new() }
    }

    /// Every case's `limit` as the driver reads it (u64; the manifest's float).
    fn limit_of(c: &Case) -> u64 {
        c.params.get("limit").map_or(LIMIT, |l| if *l >= 18_446_744_073_709_551_615.0 { u64::MAX } else { *l as u64 })
    }

    fn assert_plans_match(p: &Problem, grid: Grid, what: &str) -> usize {
        let mut n = 0;
        for su in &p.suites {
            for c in &su.cases {
                let (d, m) = (c.params["d"] as usize, c.params["m"] as usize);
                let plan = p.plan(c, LIMIT, Some(&info_of(grid))).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(plan.batch_safe, "{what} {}", c.name);
                assert_eq!(as_driver_steps(&plan, d), gram::driver_steps(d, m, limit_of(c), grid), "{what}: case {} of {}", c.name, su.spec.name);
                n += 1;
            }
        }
        n
    }

    /// THE GRAM PROGRAM IS GRAM.RS'S DRIVER: for every case of
    /// `problems/gram` (the 128 constraint probes with their forced bands,
    /// the rungs, the five pairs, the timing shapes) and the grids of the
    /// shipped, tiled and blocked kernels and a rows-on-x layout, the
    /// harness's plan is the command sequence `gram::driver_steps` gives:
    /// the same uploads, clears, dispatches (cfg words and workgroups),
    /// syncs and reads.
    #[test]
    fn the_gram_program_is_the_drivers_command_sequence() {
        let p = gram_problem();
        let grids = [gram::grid_of(SHIPPED).unwrap(), gram::grid_of(gram::TILED).unwrap(), gram::grid_of(gram::BLOCKED).unwrap(), Grid { x: 8, y: 32, bx: 2, by: 4, rows_x: true }];
        let mut n = 0;
        for g in grids {
            n += assert_plans_match(&p, g, &format!("{g:?}"));
        }
        let cases: usize = p.suites.iter().map(|s| s.cases.len()).sum();
        // Five pairs: validation keeps nonfinite only (its lowrank pair is 1.7e10 item-cells, over the 4.5e9 of `pairs::fitness_pick`).
        assert_eq!(cases, 128 + 2 + 5 + 3);
        println!("{n} plans ({cases} cases x 4 grids) equal gram.rs's driver steps");
    }

    /// THE SAVED SETS ON THE CPU: every kernel of the saved populations
    /// and the library gets the same compile-stage verdict from the harness
    /// (`kernel_info` on problems/gram) as from gram.rs (`gram::grid_of`):
    /// both accept with the same grid or both refuse; and for every one
    /// accepted, the harness's plan of every case equals gram.rs's driver
    /// steps under its grid.
    #[test]
    fn the_saved_sets_get_gram_rss_verdicts_and_plans() {
        let p = gram_problem();
        let dirs = ["logs/HGP/001/popA_g001/texts", "logs/HGP/001/popA2_g001b/texts", "logs/HGP/001/library_texts", "logs/HGP/001/popB_g080/texts", "logs/HGP/001/reserve2/texts", "logs/HGP/001/popC_rand20/texts", "logs/HGP/001/popR_rules/texts"];
        let (mut kernels, mut accepted, mut plans) = (0usize, 0usize, 0usize);
        let mut differ = Vec::new();
        for dir in dirs {
            let dir = root().join(dir);
            let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())).flatten().map(|e| e.path()).filter(|f| f.extension().is_some_and(|x| x == "wgsl")).collect();
            files.sort();
            for f in files {
                let text = std::fs::read_to_string(&f).unwrap();
                kernels += 1;
                match (gram::grid_of(&text), kernel_info(&p.spec, &text)) {
                    (Ok(g), Ok(k)) => {
                        if info_of(g) != k {
                            differ.push(format!("{}: gram {g:?}, harness {k:?}", f.display()));
                            continue;
                        }
                        accepted += 1;
                        plans += assert_plans_match(&p, g, &f.display().to_string());
                    }
                    (Err(_), Err(_)) => {}
                    (a, b) => differ.push(format!("{}: gram {:?}, harness {:?}", f.display(), a.map(|_| ()), b.map(|_| ()))),
                }
            }
        }
        println!("{kernels} saved kernels: {accepted} accepted by both, {} differ; {plans} plans equal the driver's", differ.len());
        assert!(differ.is_empty(), "{}", differ.join("\n"));
        assert!(kernels > 100, "{kernels}");
    }

    /// A host model of the shipped kernel and of kernels broken one way each.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Model {
        Shipped,
        NoProduct,
        ShortLoop,
        NoRowOffset,
        Transposed,
        Blind,
        Slow,
        WideColumns,
        Mirror,
    }

    impl HostKernel for Model {
        fn invoke(&self, ids: &HostIds, mem: &mut HostMemory) -> u64 {
            let (d, mb, r_rows, r_off) = (mem.load_u32(2, 0), mem.load_u32(2, 1), mem.load_u32(2, 2), mem.load_u32(2, 3));
            let (c, lr) = (ids.gid[0], ids.gid[1]);
            if lr >= r_rows || c >= d {
                return 0;
            }
            let mirror = *self == Model::Mirror;
            if mirror && lr > c {
                return 0;
            }
            let gr = if *self == Model::NoRowOffset { lr } else { r_off.wrapping_add(lr) };
            let items = if *self == Model::ShortLoop { mb.saturating_sub(1) } else { mb };
            let mut acc = 0.0f32;
            for t in 0..items {
                let base = t.wrapping_mul(d);
                acc = match self {
                    Model::NoProduct => acc + mem.load_f32(0, base.wrapping_add(gr)),
                    Model::Blind => acc + 1.0,
                    _ => acc + mem.load_f32(0, base.wrapping_add(gr)) * mem.load_f32(0, base.wrapping_add(c)),
                };
            }
            if *self == Model::WideColumns && c >= 41 {
                acc *= 1.001;
            }
            if mirror {
                let (a, b) = (gr.wrapping_mul(d).wrapping_add(c), c.wrapping_mul(d).wrapping_add(gr));
                mem.store_f32(1, a, mem.load_f32(1, a) + acc);
                if gr != c {
                    mem.store_f32(1, b, mem.load_f32(1, b) + acc);
                }
            } else {
                let idx = if *self == Model::Transposed { c.wrapping_mul(d).wrapping_add(lr) } else { lr.wrapping_mul(d).wrapping_add(c) };
                mem.store_f32(1, idx, mem.load_f32(1, idx) + acc);
            }
            u64::from(items) * if *self == Model::Slow { 64 } else { 1 }
        }
    }

    /// The symmetric-half kernel: it computes the cells with row <= col and
    /// writes each to h[row][col] and h[col][row], so its mirror writes
    /// leave a row band; it declares `// mode full`.
    const MIRROR: &str = "// mode full
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lr = gid.y;
    let c = gid.x;
    if (lr >= cfg.r_rows || c >= cfg.d || lr > c) { return; }
    let gr = cfg.r_off + lr;
    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        acc = acc + phi[base + gr] * phi[base + c];
    }
    h[gr * cfg.d + c] = h[gr * cfg.d + c] + acc;
    if (gr != c) { h[c * cfg.d + gr] = h[c * cfg.d + gr] + acc; }
}
";

    fn model_text(m: Model) -> String {
        let r = |a: &str, b: &str| {
            assert!(SHIPPED.contains(a), "{a}");
            SHIPPED.replace(a, b)
        };
        match m {
            Model::Shipped => SHIPPED.to_string(),
            Model::NoProduct => r("phi[base + gr] * phi[base + c]", "phi[base + gr]"),
            Model::ShortLoop => r("t < cfg.m;", "t + 1u < cfg.m;"),
            Model::NoRowOffset => r("let gr = cfg.r_off + lr;", "let gr = lr;"),
            Model::Transposed => r("let idx = lr * cfg.d + c;", "let idx = c * cfg.d + lr;"),
            Model::Blind => r("acc = acc + phi[base + gr] * phi[base + c];", "acc = acc + 1.0;"),
            Model::Slow => format!("{SHIPPED}// the host model of this text runs 64 times the shipped kernel's operations\n"),
            Model::WideColumns => r("    let idx = lr * cfg.d + c;", "    if (c >= 41u) { acc = acc * 1.001; }\n    let idx = lr * cfg.d + c;"),
            Model::Mirror => MIRROR.to_string(),
        }
    }

    const MODELS: [Model; 9] = [Model::Shipped, Model::NoProduct, Model::ShortLoop, Model::NoRowOffset, Model::Transposed, Model::Blind, Model::Slow, Model::WideColumns, Model::Mirror];

    fn host() -> HostExecutor {
        let mut h = HostExecutor::new(LIMIT);
        for m in MODELS {
            h.register(&model_text(m), Box::new(m));
        }
        h
    }

    const SMALL_RUNGS: [(usize, usize); 2] = [(48, 40), (64, 64)];
    const SMALL_TIMING: [(usize, usize); 3] = [(64, 128), (80, 64), (96, 48)];
    /// The timing round's budget for the small problem: a third of what the
    /// shipped kernel leaves is about 12 ms a build, so the slow model
    /// (64 times the operations) is stopped by the gate and the rest pass.
    const SMALL_ROUND_MS: f64 = 40.0;

    fn small_pairs() -> [Vec<gram::PairShape>; 3] {
        [vec![(Kind::Gaussian, 48, 64, 100), (Kind::Weighted, 40, 50, 101), (Kind::Ragged, 33, 77, 102)], vec![(Kind::LowRank, 48, 64, 1000), (Kind::NonFinite, 40, 64, 1001)], vec![(Kind::Gaussian, 64, 96, 2000)]]
    }

    /// `problems/gram` at small pair and timing shapes, in a temporary
    /// directory: its problem.toml with the reference path made absolute and
    /// the small round budget, its cases from `gram::write_problem_cases`.
    fn small_gram_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fuller_harness_{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        let text = std::fs::read_to_string(root().join("problems/gram/problem.toml")).unwrap();
        let shipped = root().join("kingdoms/wgsl/gram/shipped.wgsl");
        let text = text.replace("reference = \"../../kingdoms/wgsl/gram/shipped.wgsl\"", &format!("reference = {:?}", shipped.display().to_string())).replace("round_ms = 20000.0", &format!("round_ms = {SMALL_ROUND_MS:?}"));
        assert!(text.contains(&format!("round_ms = {SMALL_ROUND_MS:?}")) && text.contains(&shipped.display().to_string()));
        std::fs::write(dir.join("problem.toml"), text).unwrap();
        gram::write_problem_cases(&dir, &SMALL_RUNGS, &small_pairs(), &SMALL_TIMING, None).unwrap();
        dir
    }

    /// gram.rs's own driver steps (`gram::driver_steps`) as a plan, for
    /// gram.rs's fitness procedure on the host executor: built from gram's
    /// formulas, not from problem.toml.
    fn driver_plan(d: usize, m: usize, limit: u64, grid: Grid) -> Plan {
        let mut buffers: Vec<BufferDecl> = Vec::new();
        let (mut setup, mut run) = (Vec::new(), Vec::new());
        let mut phis = Vec::new();
        let mut acc = 0;
        for s in gram::driver_steps(d, m, limit, grid) {
            let id = buffers.len();
            match s {
                DriverStep::Upload { band, start, end } => {
                    buffers.push(BufferDecl { name: format!("phi{band}"), elem: Elem::F32, bytes: ((end - start) * d * 4) as u64, uniform: false, resident: true, init: Init::Input { tensor: "phi".into(), byte_offset: (start * d * 4) as u64, set: Vec::new() } });
                    setup.push(id);
                    phis.push(id);
                }
                DriverStep::Clear { rows } => {
                    buffers.push(BufferDecl { name: format!("acc{id}"), elem: Elem::F32, bytes: (rows * d * 4) as u64, uniform: false, resident: false, init: Init::Zero });
                    run.push(Cmd::Create(id));
                    run.push(Cmd::Clear(id));
                    acc = id;
                }
                DriverStep::Dispatch { band, cfg, workgroups } => {
                    buffers.push(BufferDecl { name: format!("cfg{id}"), elem: Elem::U32, bytes: 16, uniform: true, resident: false, init: Init::Words(cfg.iter().flat_map(|w| w.to_le_bytes()).collect()) });
                    run.push(Cmd::Create(id));
                    run.push(Cmd::Dispatch { bind: vec![(0, phis[band]), (1, acc), (2, id)], workgroups: [workgroups.0, workgroups.1, 1] });
                }
                DriverStep::Sync => run.push(Cmd::Sync),
                DriverStep::Read { r_off, rows } => run.push(Cmd::Read { src: acc, src_offset: 0, output: 0, dst_offset: (r_off * d * 4) as u64, bytes: (rows * d * 4) as u64 }),
            }
        }
        Plan { buffers, setup, run, outputs: vec![(d * d * 4) as u64], batch_safe: true }
    }

    /// gram.rs's fitness procedure (`gram::fitness_with`) on the host
    /// executor: its cases, rungs, pairs and timing shapes as `FitnessContext`
    /// holds them, each run through gram's own driver steps.
    struct GramHost<'a> {
        exec: &'a HostExecutor,
        problem: &'a Problem,
        cases: Vec<HostCase>,
        rungs: Vec<HostCase>,
        pairs: [Vec<HostCase>; 3],
        timing: Vec<HostCase>,
    }

    /// A Phi staged for gram's driver: d, m, limit, its data, its residents.
    type HostCase = (usize, usize, u64, CaseData, HostStaged);

    /// A pair: name, d, m, Phi, the f64 reference.
    type PairData = (String, usize, usize, Vec<f32>, Vec<f64>);

    impl GramHost<'_> {
        fn staged(exec: &HostExecutor, d: usize, m: usize, limit: u64, phi: &[f32]) -> HostCase {
            let mut data = CaseData::default();
            data.inputs.insert("phi".into(), f32_bytes(phi));
            let plan = driver_plan(d, m, limit, Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false });
            let (st, _) = exec.stage(&plan, &data).unwrap();
            (d, m, limit, data, st)
        }
    }

    impl gram::Driver for GramHost<'_> {
        type Kernel = (HostCompiled, Grid);

        fn compile(&self, source: &str) -> Result<(HostCompiled, Grid), String> {
            let grid = gram::grid_of(source)?;
            Ok((self.exec.compile(self.problem, source)?.0, grid))
        }

        fn batch(&self, k: &(HostCompiled, Grid), cases: std::ops::Range<usize>) -> Result<Batch, String> {
            let plans: Vec<Plan> = self.cases[cases.clone()].iter().map(|(d, m, l, _, _)| driver_plan(*d, *m, *l, k.1)).collect();
            let items: Vec<(&Plan, &CaseData, &HostStaged)> = self.cases[cases].iter().zip(&plans).map(|((_, _, _, data, st), p)| (p, data, st)).collect();
            let (runs, compute_ms) = self.exec.run_batch(&k.0, &items)?;
            Ok(Batch { h: runs.iter().map(|r| bytemuck_f32(&r.outputs[0])).collect(), gpu_ms: runs.iter().map(|r| r.gpu_ms).collect(), compute_ms })
        }

        fn run(&self, k: &(HostCompiled, Grid), which: Which, read_h: bool) -> Result<Build, String> {
            let (d, m, l, data, st) = match which {
                Which::Rung(i) => &self.rungs[i],
                Which::Pair(s, i) => &self.pairs[s][i],
                Which::Timing(i) => &self.timing[i],
            };
            let r = self.exec.run(&k.0, &driver_plan(*d, *m, *l, k.1), data, st, read_h)?;
            Ok(Build { h: if read_h { bytemuck_f32(&r.outputs[0]) } else { Vec::new() }, gpu_ms: r.gpu_ms, compute_ms: r.compute_ms, wall_ms: r.wall_ms, upload_ms: 0.0, passes: 0, item_bands: 0, row_bands: 0 })
        }
    }

    /// The fields of a fitness that are not wall-clock: everything but
    /// `seconds` and `split`.
    fn same_fitness(a: &Fitness, b: &Fitness) -> Result<(), String> {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let pairs: [(&str, Vec<u64>, Vec<u64>); 8] = [
            ("distance", bits(&[a.distance]), bits(&[b.distance])),
            ("closeness", bits(&a.closeness), bits(&b.closeness)),
            ("closeness_hff", bits(&[a.closeness_hff]), bits(&[b.closeness_hff])),
            ("columns", bits(&a.columns), bits(&b.columns)),
            ("speedups", bits(&a.speedups), bits(&b.speedups)),
            ("hff", bits(&[a.hff]), bits(&[b.hff])),
            ("predicted_ms", bits(&a.predicted_ms), bits(&b.predicted_ms)),
            ("meter", bits(&[a.rate_per_g, a.floor_ms]), bits(&[b.rate_per_g, b.floor_ms])),
        ];
        if a.feasible != b.feasible || a.passed != b.passed || a.detail != b.detail || bits(&a.measured_ms) != bits(&b.measured_ms) {
            return Err(format!("verdict: {} {} {:?} / {} {} {:?}", a.feasible, a.passed, a.detail, b.feasible, b.passed, b.detail));
        }
        for (name, x, y) in pairs {
            if x != y {
                return Err(format!("{name}: {x:?} / {y:?}"));
            }
        }
        Ok(())
    }

    /// THE HARNESS REPRODUCES GRAM.RS'S FITNESS (on the host executor, the
    /// small problem): for the shipped kernel and eight models broken or
    /// changed one way each, `Harness::evaluate` over problem.toml and
    /// gram.rs's own procedure (`gram::fitness_with` over gram's driver
    /// steps and its in-memory probes and pairs) give the same fitness:
    /// feasibility, cases passed, the detail, the distance, every closeness
    /// column and its angle, the six columns, the speed-ups, the HFF angle,
    /// the meter and the predicted and measured times, bit for bit. The
    /// host executor's times are synthetic and deterministic, so the cost
    /// model, the gates (the slow model is stopped) and the timing columns
    /// are compared as well.
    #[test]
    fn the_harness_reproduces_gram_rss_fitness_on_the_host() {
        let dir = small_gram_dir("reproduce");
        let exec = host();
        let problem = Problem::load(&dir).unwrap();
        let harness = Harness::new(problem.clone(), &exec).unwrap();
        // gram.rs's side, from its own data and procedure.
        let probes = crate::wgsl::probes::probes(1);
        let admission = crate::wgsl::probes::admission(&probes, 1);
        let cases = admission.iter().map(|&i| GramHost::staged(&exec, probes[i].d, probes[i].m, probes[i].limit, &probes[i].phi)).collect();
        let at = |set: &[(usize, usize)], seed: u64| set.iter().map(|&(d, m)| GramHost::staged(&exec, d, m, LIMIT, &gram::phi(d, m, seed))).collect::<Vec<_>>();
        let pair_data: [Vec<PairData>; 3] = small_pairs().map(|set| set.into_iter().map(|(k, d, m, s)| { let p = gram::phi_of(k, d, m, s); let h = gram::reference(&p, d, m); (format!("{}_d{d}_m{m}_s{s}", k.name()), d, m, p, h) }).collect());
        let mut drv = GramHost { exec: &exec, problem: &problem, cases, rungs: at(&SMALL_RUNGS, gram::RUNG_SEED), pairs: [Vec::new(), Vec::new(), Vec::new()], timing: at(&SMALL_TIMING, gram::TIMING_SEED) };
        drv.pairs = std::array::from_fn(|k| pair_data[k].iter().map(|(_, d, m, p, _)| GramHost::staged(&exec, *d, *m, LIMIT, p)).collect());
        // FitnessContext::new: the shipped kernel's error on each pair, its
        // compute ms at the timing shapes after a warm-up, then time_shipped.
        let shipped = gram::Driver::compile(&drv, SHIPPED).unwrap();
        let accuracy: [Vec<AccuracyPair>; 3] = std::array::from_fn(|k| {
            pair_data[k].iter().enumerate().map(|(i, (n, d, m, p, h))| {
                let b = gram::Driver::run(&drv, &shipped, Which::Pair(k, i), true).unwrap();
                (n.clone(), *d, *m, p.clone(), h.clone(), gram::check(&b.h, h, *d, *m).max_ratio)
            }).collect()
        });
        gram::Driver::run(&drv, &shipped, Which::Timing(0), true).unwrap();
        let shipped_ms: [f64; 3] = std::array::from_fn(|k| gram::Driver::run(&drv, &shipped, Which::Timing(k), true).unwrap().compute_ms);
        gram::Driver::run(&drv, &shipped, Which::Timing(0), false).unwrap();
        let reference: [f64; 3] = std::array::from_fn(|k| gram::Driver::run(&drv, &shipped, Which::Timing(k), false).unwrap().gpu_ms);
        assert_eq!(shipped_ms, harness.shipped_ms);
        assert_eq!(reference, harness.shipped_reference());
        for (k, set) in accuracy.iter().enumerate() {
            let theirs: Vec<u64> = harness.accuracy[k].iter().map(|x| x.0.reference.to_bits()).collect();
            assert_eq!(set.iter().map(|p| p.5.to_bits()).collect::<Vec<_>>(), theirs, "split {k}: the shipped kernel's error on the pairs");
        }
        let spec = gram::Spec { probes: &probes, admission: &admission, pairs: [&accuracy[0], &accuracy[1], &accuracy[2]], rungs: SMALL_RUNGS.to_vec(), timing: SMALL_TIMING.to_vec(), limit_ms: gram::slow_limit_ms(SMALL_ROUND_MS, &shipped_ms), base_budget_ms: gram::BASE_BUDGET_MS, timing_rounds: 1, shipped_reference: reference };
        let mut matched = 0;
        let mut lines = Vec::new();
        for m in MODELS {
            let text = model_text(m);
            let g = gram::fitness_with(&spec, &drv, &text);
            let h = harness.evaluate(&text);
            lines.push(format!("{m:?}: feasible {} passed {} hff {:.4} closeness_hff {:.4} {}", h.feasible, h.passed, h.hff, h.closeness_hff, h.detail));
            match (m, same_fitness(&g, &h)) {
                (Model::Mirror, _) => {
                    // gram.rs's driver bands every kernel; a mirrored kernel's
                    // mirror writes leave its band there, so gram.rs's fitness
                    // of it is wrong and only the harness's is compared (below).
                    assert!(!g.feasible, "gram.rs's banded driver cannot hold a mirrored kernel: {g:?}");
                }
                (_, Ok(())) => matched += 1,
                (_, Err(e)) => panic!("{m:?}: {e}\ngram: {g:?}\nharness: {h:?}"),
            }
        }
        println!("{}", lines.join("\n"));
        println!("{matched} of {} kernels: the harness's fitness equals gram.rs's", MODELS.len() - 1);
        // The verdicts the models are built to give.
        let f = |m: Model| harness.evaluate(&model_text(m));
        let s = f(Model::Shipped);
        assert!(s.feasible && s.passed == 128 && s.speedups.iter().all(|x| (*x - 1.0).abs() < 1e-12), "{s:?}");
        for m in [Model::NoProduct, Model::ShortLoop, Model::NoRowOffset, Model::Transposed, Model::Blind] {
            let x = f(m);
            assert!(!x.feasible && x.detail.starts_with("constraint") && x.passed < 128 && x.closeness_hff > 0.0, "{m:?}: {x:?}");
        }
        let slow = f(Model::Slow);
        assert!(!slow.feasible && slow.passed == 128 && slow.detail.starts_with("slow: predicted"), "{slow:?}");
        let wide = f(Model::WideColumns);
        assert!(!wide.feasible && wide.passed == 128 && wide.detail.starts_with("pair "), "{wide:?}");
        assert_eq!(matched, MODELS.len() - 1);
    }

    /// THE SYMMETRIC-HALF KERNEL: a kernel whose mirror writes leave its row
    /// band, declaring `// mode full`, gets one row band over all of H; its
    /// H equals the shipped kernel's bit for bit on every constraint case
    /// and pair (the product commutes and each cell's sum has the shipped
    /// order) and it is feasible. Without the mode line it gets row bands,
    /// its mirror writes fall outside the band's accumulator and the
    /// constraint fails it. A full-mode kernel at a d whose H exceeds the
    /// binding limit is a plan error naming the requirement.
    #[test]
    fn a_symmetric_half_kernel_runs_over_the_full_matrix() {
        let dir = small_gram_dir("mirror");
        let exec = host();
        let problem = Problem::load(&dir).unwrap();
        let (mirror, mi) = exec.compile(&problem, MIRROR).unwrap();
        let (shipped, si) = exec.compile(&problem, SHIPPED).unwrap();
        assert_eq!(mi.modes, vec!["full".to_string()]);
        let mut compared = 0;
        for su in problem.suites.iter().filter(|s| matches!(s.spec.role, Role::Constraint | Role::Accuracy)) {
            for c in &su.cases {
                let data = problem.case_data(c, LIMIT).unwrap();
                let pm = problem.plan(c, LIMIT, Some(&mi)).unwrap();
                let ps = problem.plan(c, LIMIT, Some(&si)).unwrap();
                assert_eq!(pm.run.iter().filter(|x| matches!(x, Cmd::Sync)).count(), 1, "one row band");
                let (stm, _) = exec.stage(&pm, &data).unwrap();
                let (sts, _) = exec.stage(&ps, &data).unwrap();
                let hm = exec.run(&mirror, &pm, &data, &stm, true).unwrap();
                let hs = exec.run(&shipped, &ps, &data, &sts, true).unwrap();
                assert_eq!(hm.outputs, hs.outputs, "{}: the mirrored H is not the shipped H", c.name);
                let (d, m) = (c.params["d"] as usize, c.params["m"] as usize);
                if su.spec.role == Role::Accuracy {
                    let v = gram::check(&bytemuck_f32(&hm.outputs[0]), &data.expected["h"], d, m);
                    assert!(v.passes() || c.name.starts_with("nonfinite"), "{}: {v:?}", c.name);
                }
                compared += 1;
            }
        }
        let harness = Harness::new(problem.clone(), &exec).unwrap();
        let f = harness.evaluate(MIRROR);
        assert!(f.feasible && f.passed == 128, "{f:?}");
        let banded = MIRROR.replace("// mode full\n", "");
        let mut exec2 = host();
        exec2.register(&banded, Box::new(Model::Mirror));
        let h2 = Harness::new(problem.clone(), &exec2).unwrap();
        let b = h2.evaluate(&banded);
        assert!(!b.feasible && b.detail.starts_with("constraint"), "{b:?}");
        // At d 8192 one H is 256 MiB: over the limit.
        let big = Case { name: "big".into(), label: "big".into(), params: [("d".to_string(), 8192.0), ("m".to_string(), 1024.0)].into_iter().collect(), inputs: BTreeMap::new(), expected: BTreeMap::new() };
        let e = problem.plan(&big, LIMIT, Some(&mi)).unwrap_err();
        assert!(e.contains("the problem requires") && e.contains("one buffer"), "{e}");
        assert!(problem.plan(&big, LIMIT, Some(&si)).is_ok());
        println!("{compared} cases: the mirrored kernel's H is the shipped kernel's bit for bit; banded it is {}", b.detail);
    }

    /// The timing code paths on the host: re-timing a feasible kernel keeps
    /// its accuracy columns and gives the speed-ups the evaluation gave; a
    /// second `time_shipped` leaves the least reference times where they
    /// were; the gate's limit is a third of what the reference leaves.
    #[test]
    fn retiming_and_the_shipped_reference_follow_gram_rss_rules() {
        let dir = small_gram_dir("timing");
        let exec = host();
        let harness = Harness::new(Problem::load(&dir).unwrap(), &exec).unwrap();
        let before = harness.shipped_reference();
        let again = harness.time_shipped().unwrap();
        assert_eq!(before, again);
        assert_eq!(harness.shipped_reference(), before);
        assert_eq!(harness.slow_limit_ms(), (SMALL_ROUND_MS - harness.shipped_ms.iter().sum::<f64>()) / 3.0);
        let text = model_text(Model::Mirror);
        let f = harness.evaluate(&text);
        assert!(f.feasible, "{f:?}");
        let (speedups, columns, hff) = harness.retime(&text, &f.columns).unwrap();
        assert_eq!(speedups, f.speedups);
        assert_eq!(&columns[3..], &f.columns[3..]);
        assert_eq!(hff, f.hff);
        // The mirrored kernel does about half the operations: faster than 1.
        assert!(speedups.iter().all(|s| *s > 1.0), "{speedups:?}");
    }

    /// A toy problem: a resident input `res`, bindings a (read) and b
    /// (read-write), and the run steps given.
    fn toy(run: &str) -> Problem {
        let text = format!(
            "name = \"toy\"\nwork = \"n\"\nalias = \"distinct\"\n[kernel]\nentry = \"main\"\nreference = \"none\"\n\
             [[binding]]\nbinding = 0\nname = \"a\"\nkind = \"storage_read\"\nelement = \"f32\"\n\
             [[binding]]\nbinding = 1\nname = \"b\"\nkind = \"storage_read_write\"\nelement = \"f32\"\n\
             [[input]]\nname = \"x\"\nelement = \"f32\"\nshape = [\"n\"]\n\
             [[output]]\nname = \"y\"\nelement = \"f32\"\nshape = [\"n\"]\nexpected_element = \"f32\"\n\
             [[setup]]\nbuffer = \"res\"\nelement = \"f32\"\nelements = \"n\"\nfrom = \"x\"\n\
             {run}\n\
             [[suite]]\nname = \"c\"\nrole = \"constraint\"\ncases = \"none\"\nnorm = \"mean_abs_over_max\"\npass = \"0\"\n\
             [budget]\nbase_ms = 1.0\nround_ms = 1.0\n"
        );
        let spec: ProblemSpec = toml::from_str(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        Problem { dir: PathBuf::new(), spec, suites: Vec::new(), reference: String::new() }
    }

    fn toy_plan(run: &str) -> Result<Plan, String> {
        let c = Case { name: "n8".into(), label: "n8".into(), params: [("n".to_string(), 8.0)].into_iter().collect(), inputs: BTreeMap::new(), expected: BTreeMap::new() };
        toy(run).plan(&c, LIMIT, Some(&KernelInfo { workgroup: [64, 1, 1], bx: 1, by: 1, rows_x: false, modes: Vec::new() }))
    }

    const O: &str = "[[run]]\nbuffer = \"o\"\nelement = \"f32\"\nelements = \"n\"\nfill = \"zero\"\n";
    const DISPATCH: &str = "[[run]]\ndispatch = { a = \"res\", b = \"o\" }\nworkgroups = [\"1\", \"1\", \"1\"]\n";
    const READ: &str = "[[run]]\nread = \"o\"\ninto = \"y\"\nelements = \"n\"\n";
    const SYNC: &str = "[[run]]\nsync = true\n";

    /// The plan's rules: a read spoiled by a later write in its segment is
    /// an error; one spoiled only when the syncs are dropped is not
    /// batch-safe; a resident buffer bound writable, a buffer bound twice, a
    /// binding left unbound, and a kernel's unknown mode or a binding of
    /// the wrong access are each named.
    #[test]
    fn the_plans_rules_are_enforced_and_named() {
        let ok = toy_plan(&[O, DISPATCH, READ, SYNC].concat()).unwrap();
        assert!(ok.batch_safe);
        let e = toy_plan(&[O, DISPATCH, READ, DISPATCH, SYNC].concat()).unwrap_err();
        assert!(e.contains("o is read and then written in the same segment"), "{e}");
        let unsafe_batch = toy_plan(&[O, DISPATCH, READ, SYNC, DISPATCH, SYNC].concat()).unwrap();
        assert!(!unsafe_batch.batch_safe);
        let e = toy_plan(&[O, "[[run]]\ndispatch = { a = \"res\", b = \"res\" }\nworkgroups = [\"1\", \"1\", \"1\"]\n"].concat()).unwrap_err();
        assert!(e.contains("res is resident"), "{e}");
        let e = toy_plan(&[O, "[[run]]\ndispatch = { a = \"o\", b = \"o\" }\nworkgroups = [\"1\", \"1\", \"1\"]\n"].concat()).unwrap_err();
        assert!(e.contains("alias: o is bound twice"), "{e}");
        let e = toy_plan(&[O, "[[run]]\ndispatch = { a = \"res\" }\nworkgroups = [\"1\", \"1\", \"1\"]\n"].concat()).unwrap_err();
        assert!(e.contains("leaves binding b unbound"), "{e}");
        let e = toy_plan(&[O, "[[run]]\ndispatch = { a = \"res\", b = \"o\" }\nworkgroups = [\"70000\", \"1\", \"1\"]\n"].concat()).unwrap_err();
        assert!(e.contains("over 65,535"), "{e}");
        let p = gram_problem();
        let e = kernel_info(&p.spec, &MIRROR.replace("// mode full", "// mode half")).unwrap_err();
        assert!(e.contains("mode \"half\""), "{e}");
        let e = kernel_info(&p.spec, &SHIPPED.replace("var<storage, read> phi", "var<storage, read_write> phi")).unwrap_err();
        assert!(e.contains("is read_write, the schema's is read-only"), "{e}");
        let e = kernel_info(&p.spec, &SHIPPED.replace("@binding(2) var<uniform> cfg", "@binding(3) var<uniform> cfg")).unwrap_err();
        assert!(e.contains("@binding(3), which the problem's schema does not"), "{e}");
    }

    fn nluca() -> Problem {
        Problem::load(&root().join("problems/nluca_step")).expect("problems/nluca_step")
    }

    /// NLUCA'S CASES WIRED IN: the problem loads Noctiluca's 16 cases where
    /// Noctiluca wrote them, every file has the size its shape says, the
    /// shipped kernel meets the binding schema, and every case's plan is the
    /// ping-pong of the contract: 64 dispatches, step t reading s{t % 2} and
    /// writing s{(t + 1) % 2} with offset[3] = t, the state copied into the
    /// checkpoints after steps 1, 8 and 64, read once, batch-safe.
    #[test]
    fn nlucas_cases_load_and_plan_as_the_contract_says() {
        let p = nluca();
        let info = kernel_info(&p.spec, &p.reference).unwrap();
        assert_eq!(info.workgroup, [256, 1, 1]);
        let mut n = 0;
        for su in &p.suites {
            for c in &su.cases {
                let data = p.case_data(c, LIMIT).unwrap_or_else(|e| panic!("{e}"));
                let plan = p.plan(c, LIMIT, Some(&info)).unwrap();
                assert!(plan.batch_safe);
                let width = c.params["width"] as u64;
                let dispatches: Vec<&Vec<(u32, BufId)>> = plan.run.iter().filter_map(|x| if let Cmd::Dispatch { bind, .. } = x { Some(bind) } else { None }).collect();
                assert_eq!(dispatches.len(), 64);
                for (t, bind) in dispatches.iter().enumerate() {
                    let name = |b: u32| plan.buffers[bind.iter().find(|x| x.0 == b).unwrap().1].name.clone();
                    assert_eq!((name(1), name(4), name(6)), (format!("s{}", t % 2), format!("s{}", (t + 1) % 2), format!("offset{t}")));
                    let off = &plan.buffers[bind[6].1];
                    let bytes = initial_bytes(off, &data).unwrap();
                    assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), t as u32);
                    assert_eq!(bytes[..12], data.inputs["offset"][..12]);
                }
                let copies: Vec<u64> = plan.run.iter().filter_map(|x| if let Cmd::Copy { dst_offset, .. } = x { Some(*dst_offset / 4 / width) } else { None }).collect();
                assert_eq!(copies, vec![0, 1, 2]);
                assert_eq!(plan.run.iter().filter(|x| matches!(x, Cmd::Read { .. })).count(), 1);
                if su.spec.role == Role::Constraint {
                    assert_eq!(data.expected["trajectory"].len() as u64, 3 * width);
                }
                n += 1;
            }
        }
        assert_eq!(n, 16 + 3);
    }

    /// EVERY SPEED-UP RANKS under `speed_column = "ranked"` (nluca's): the
    /// reference's speed is 0.5, the column strictly decreasing in the
    /// speed-up on both sides of 1; nluca_s2's r0 (0.192 / 0.706 / 0.864)
    /// and r3 (0.164 / 0.701 / 0.865), both HFF π/3 under the Gram's clamp,
    /// now differ and r0 ranks better. The Gram's rule is unchanged.
    #[test]
    fn a_ranked_speed_column_orders_slower_kernels() {
        assert_eq!(nluca().spec.speed_column, SpeedColumn::Ranked);
        assert_eq!(gram_problem().spec.speed_column, SpeedColumn::Reciprocal);
        let r = SpeedColumn::Ranked;
        assert_eq!(r.column(1.0), 0.5);
        let xs = [0.01, 0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 8.0, 100.0];
        assert!(xs.windows(2).all(|w| r.column(w[0]) > r.column(w[1])), "not strictly decreasing");
        let hff = |s: [f64; 3], rule: SpeedColumn| truenorth(&[rule.column(s[0]), rule.column(s[1]), rule.column(s[2]), 0.0, 0.0, 0.0]);
        let (r0, r3) = ([0.192, 0.706, 0.864], [0.164, 0.701, 0.865]);
        assert_eq!(hff(r0, SpeedColumn::Reciprocal), hff(r3, SpeedColumn::Reciprocal), "the clamp ties them");
        assert!((hff(r0, SpeedColumn::Reciprocal) - std::f64::consts::FRAC_PI_3).abs() < 1e-9);
        assert!(hff(r0, r) < hff(r3, r), "r0 is faster on the first shape and ranks better: {} {}", hff(r0, r), hff(r3, r));
        assert!(hff([1.0; 3], r) < hff(r0, r) && hff([2.0; 3], r) < hff([1.0; 3], r));
        assert_eq!(fitness_columns([8.0, 0.5, 1.0], [0.1, 0.2, 0.3]), [0.125, 1.0, 1.0, 0.1, 0.2, 0.3]);
    }

    /// F16 STATE MUST FAIL THIS ORACLE (Noctiluca's design): the expected
    /// checkpoints rounded to f16, the least error any kernel holding its
    /// state in f16 makes, measured by the constraint's norm against every
    /// case's tolerance. Printed per case; every case must fail.
    #[test]
    fn f16_state_fails_every_nluca_case() {
        let p = nluca();
        let su = p.suites.iter().find(|s| s.spec.role == Role::Constraint).unwrap();
        let mut failing = 0;
        for c in &su.cases {
            let data = p.case_data(c, LIMIT).unwrap();
            let want = &data.expected["trajectory"];
            let width = c.params["width"] as usize;
            let f16 = |x: f64| -> f32 {
                // Round to 11 significant bits (f16's), exponent range kept.
                if x == 0.0 || !x.is_finite() {
                    return x as f32;
                }
                let e = x.abs().log2().floor();
                let q = 2f64.powf(e - 10.0);
                ((x / q).round() * q) as f32
            };
            let got: Vec<f32> = want.iter().map(|x| f16(*x)).collect();
            let (v, _) = points_value(Norm::MaxAbsOverRms, &got, want, (3, width), 3, 0.0);
            let tol = c.params["tol"];
            println!("{}: f16 rounding {v:.3e}, tolerance {tol:.3e}", c.name);
            if v > tol {
                failing += 1;
            }
        }
        assert_eq!(failing, su.cases.len(), "f16 state passes {} of {} cases", su.cases.len() - failing, su.cases.len());
    }

    /// Tensor indexing and `sum` (an argmax output's score), out-of-range named.
    #[test]
    fn expressions_index_host_tensors_and_sum() {
        let mut env = Env::new();
        env.insert("i".into(), 1.0);
        env.insert("dp".into(), 3.0);
        let mut t = Tensors::new();
        t.insert("a".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        t.insert("b".into(), vec![1.0, 0.5, 0.25]);
        assert_eq!(parse_expr("sum(p, dp, a[i * dp + p] * b[p])").unwrap().eval_in(&env, &t).unwrap(), 4.0 + 2.5 + 1.5);
        assert!(parse_expr("a[i * 9]").unwrap().eval_in(&env, &t).unwrap_err().contains("past its 6 elements"));
        assert!(parse_expr("c[0]").unwrap().eval_in(&env, &t).unwrap_err().contains("no tensor"));
        assert!(parse_expr("a[0").is_err());
    }

    fn kmeans() -> Problem {
        Problem::load(&root().join("problems/kmeans_assign")).expect("problems/kmeans_assign")
    }

    /// GENERATED INPUTS are deterministic in the case, differ between inputs
    /// and cases, hold their declared shape, and unit rows are unit length;
    /// an index input stays in its bounds.
    #[test]
    fn generated_inputs_follow_the_schema() {
        let p = kmeans();
        let su = p.suites.iter().find(|s| s.spec.role == Role::Constraint).unwrap();
        let c = su.cases.iter().find(|c| c.name == "n300_c33_d256_band128").unwrap();
        let (a, b) = (p.case_data(c, LIMIT).unwrap(), p.case_data(c, LIMIT).unwrap());
        assert_eq!(a.inputs, b.inputs, "the same case generates the same inputs");
        let pts = Elem::F32.decode(&a.inputs["points"]);
        assert_eq!(pts.len(), 300 * 256);
        for r in pts.chunks(256) {
            let n: f64 = r.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-5, "a row of norm {n}");
        }
        assert_ne!(a.inputs["points"][..1024], a.inputs["centroids"][..1024]);
        let other = su.cases.iter().find(|c| c.name == "n64_c300_d64").unwrap();
        assert_ne!(p.case_data(other, LIMIT).unwrap().inputs["points"][..256], a.inputs["points"][..256]);
        let idx = TensorSpec { name: "ix".into(), element: Elem::U32, shape: vec!["1000".into()], generate: Some(Generate { dist: Dist::Index, lo: Some("3".into()), hi: Some("n".into()), mean: None, std: None, row: None, value: None }) };
        let mut env = Env::new();
        env.insert("n".into(), 10.0);
        let v = Elem::U32.decode(&generate(&idx, idx.generate.as_ref().unwrap(), 1000, 7, &env).unwrap());
        assert!(v.iter().all(|x| (3.0..10.0).contains(x)) && v.contains(&3.0) && v.contains(&9.0), "index values within [3, 10)");
    }

    /// THE ARGMAX TIE RULE: an index that differs from the expected one is
    /// accepted when its score is within the margin of the expected index's,
    /// refused past it or outside [0, classes); an exact output refuses any
    /// difference.
    #[test]
    fn an_argmax_output_accepts_a_near_tie_and_nothing_else() {
        let p = kmeans();
        let out = p.spec.output[0].clone();
        assert_eq!((out.kind(), out.tie), (OutputKind::Argmax, Some(Tie::Margin)));
        let mut env = Env::new();
        for (k, v) in [("n", 3.0), ("c", 3.0), ("dp", 1.0)] {
            env.insert(k.into(), v);
        }
        // Point scores: c[j] * x. Centroids 0.5, 0.5 + 1e-7 (a near tie under
        // the margin 2 * dp * eps = 2.4e-7) and 0.9.
        let f = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let mut data = CaseData::default();
        data.inputs.insert("points".into(), f(&[1.0, 1.0, 1.0]));
        data.inputs.insert("centroids".into(), f(&[0.5, 0.5 + 1e-7, 0.9]));
        data.expected.insert("assignments".into(), vec![1.0, 2.0, 2.0]);
        let case = Case { name: "tie".into(), label: "tie".into(), params: BTreeMap::new(), inputs: BTreeMap::new(), expected: BTreeMap::new() };
        let s = Staged { case, data, staged: (), work: 0.0, dims: (1, 3), points: 1, tau: 0.0, reference: 0.0, pass: 0.0, env };
        let u = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        assert_eq!(judge(&p, &out, Norm::MeanAbsOverMax, &s, &u(&[0, 2, 2])).unwrap().0, 0.0, "a near tie is accepted");
        assert_eq!(judge(&p, &out, Norm::MeanAbsOverMax, &s, &u(&[1, 0, 2])).unwrap().0, 1.0 / 3.0, "0.5 against 0.9 is wrong");
        assert_eq!(judge(&p, &out, Norm::MeanAbsOverMax, &s, &u(&[1, 2, 7])).unwrap().0, 1.0 / 3.0, "an index past the classes is wrong");
        let exact = OutputSpec { kind: Some(OutputKind::Exact), tie: None, ..out.clone() };
        assert_eq!(judge(&p, &exact, Norm::MeanAbsOverMax, &s, &u(&[0, 2, 2])).unwrap().0, 1.0 / 3.0, "an exact output refuses the tie");
    }
}
