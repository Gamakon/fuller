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
//! | `[kernel]` | `entry` (the entry point a dispatch runs when it names none), `entries` (further entry points of the one module a dispatch may name; each compiled into its own pipeline over one layout; the program reads `wg_x_<entry>`, `wg_y_<entry>`, `wg_z_<entry>`), `passes_header` (read `// passes N`, 1 <= N <= 16, default 1: the program reads `passes`), `grid_header` (read `// grid x y bx by [rows-x]` as `gram.rs` does; default false), `reference` (the shipped kernel's path, relative to the problem: the speed-ups' denominator and the accuracy baseline), `modes` (the modes a kernel may declare with a `// mode <name>` line; the program reads `mode_<name>`, 1 or 0) |
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
//! | buffer | `buffer`, `element`, `elements`, and `fill = "zero"`, `fill = "pad"` or `from` (an input, a template) with `offset` and `set = ["index = expr"]` | create a device buffer: cleared to zero (`clear_buffer`, initialisation and clearing), every bit set (`0xFFFFFFFF` words: a candidate list's pads), or a slice of an input with elements overwritten (a per-step uniform update in a storage word). A run buffer is made fresh by every run and lives until the run ends: state that persists across a run's loop iterations (an index arena) and never across runs |
//! | uniform | `uniform`, `of` (a uniform binding), `words` | create a uniform buffer, each word typed by the binding's `fields` |
//! | dispatch | `dispatch = { entry = "name", binding = "buffer", ... }`, `workgroups` (three expressions), `sample` (a label template) | one compute pass of the kernel's `entry` (default `[kernel] entry`), timed by GPU timestamps, its GPU ms its own value (`RunOut::dispatch_ms`, summed by `sample` label in `samples_by_label`); every binding of the schema bound; the dispatch geometry may read the kernel's `wg_x`, `wg_y`, `wg_z` (and per entry `wg_x_<entry>`..), `bx`, `by`, `rows_x`, `passes` |
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
//! `pooled_recall` (v24, `[output.pooled]`, [`PooledSpec`]): per stage and
//! query the candidate lists of several independent indexes pooled and
//! re-ranked exactly against CSR records, counted per window (stage x
//! half) as misses, pool size and broken entries beside today's index's
//! counts ([`pooled_windows`]); its fitness is the hff library's TrueNorth
//! over a vector of inputs ([`Harness::evaluate_pooled`], `Fitness::inputs`).
//! A pooled problem has its constraint suite only: its timing samples are
//! fresh runs of its cases, not paired timing shapes.
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
    /// The entry point a dispatch runs when it names none.
    pub entry: String,
    /// MULTI-ENTRY: further entry points of the one module a dispatch may
    /// name (`dispatch = { entry = "search", ... }`), each compiled into
    /// its own pipeline over the same layout; `entry` is always one.
    #[serde(default)]
    pub entries: Vec<String>,
    #[serde(default)]
    pub grid_header: bool,
    /// Read `// passes N` (1 <= N <= 16, default 1): the program reads
    /// `passes` (a pass loop's count).
    #[serde(default)]
    pub passes_header: bool,
    pub reference: String,
    /// The modes a kernel may declare by a `// mode <name>` line; the
    /// program reads `mode_<name>` (1 when declared, else 0).
    #[serde(default)]
    pub modes: Vec<String>,
}

impl KernelSpec {
    /// Every entry point a dispatch may name, `entry` first: a plan's
    /// dispatch holds its index here, an executor compiles one pipeline each.
    pub fn entry_names(&self) -> Vec<String> {
        let mut v = vec![self.entry.clone()];
        for e in &self.entries {
            if !v.contains(e) {
                v.push(e.clone());
            }
        }
        v
    }
}

/// The most passes a kernel may declare (`// passes N`).
pub const MAX_PASSES: u32 = 16;

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
    /// Candidate lists of several independent indexes per stage, pooled,
    /// re-ranked exactly and counted per window ([`PooledSpec`]).
    PooledRecall,
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
    /// A `pooled_recall` output's judging (`[output.pooled]`).
    pub pooled: Option<PooledSpec>,
}

/// HOW A `pooled_recall` OUTPUT IS JUDGED (`[output.pooled]`). The output
/// is u32 `[stages * seeds * queries, c_max]`: stage-major, then seed, then
/// query, each query's `c_max` candidate stream-row ids, `0xFFFFFFFF` a pad.
/// Per stage and query the seeds' lists are pooled (pads dropped,
/// deduplicated); an entry at or past the stage's `arrived` is broken
/// (a future row, an oracle row or out of range); every other entry is
/// re-ranked by its exact cosine with the query's record (the sparse dot in
/// f64 of the CSR records, unit vectors: the exporter's measure), the top
/// `top` counted against the query's accept set. Per window (stage x half):
/// `M = top * |half| - sum found` (found capped at `top`), `P` the pooled
/// entries summed (broken ones included: they were sent), `B` the broken
/// entries summed, each beside today's `M` and `P` summed over the same
/// queries. Expressions read the case's environment, `stage` (1-based) and
/// `q` (the query); names with braces are templates over `stage`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PooledSpec {
    pub seeds: String,
    pub stages: String,
    pub queries: String,
    /// The re-ranked list counted (10).
    pub top: String,
    /// The rows arrived by `stage` (the arrived rows are 0..arrived).
    pub arrived: String,
    /// Query `q`'s record row in the CSR.
    pub query_row: String,
    /// The CSR records' inputs: offsets (rows + 1), indices, values.
    pub rec_offsets: String,
    pub rec_indices: String,
    pub rec_values: String,
    /// The input holding each query's half, 0 or 1.
    pub half: String,
    /// Each stage's accept sets: per query offsets and the rows.
    pub accept_offsets: String,
    pub accept_rows: String,
    /// Today's index per query and stage: misses, candidates pooled.
    pub today_misses: String,
    pub today_pooled: String,
    /// The timing samples, per stage.
    #[serde(default)]
    pub timing: Vec<PooledTiming>,
}

/// One timing sample of a stage: the GPU ms of the run's dispatches whose
/// `sample` label is this template at the stage, summed (one stage's work,
/// every pass and seed); today's ms from `today` (an expression, its braces
/// a template over `stage`: `"today_settle_ms_{stage}"`) or `today_input`
/// (a per-query input, summed over the queries judged).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PooledTiming {
    pub sample: String,
    /// The kernel's ms through the stage: the label's ms of stages 1..k
    /// summed (an ingest against today's whole build of sections 1..k).
    #[serde(default)]
    pub cumulative: bool,
    pub today: Option<String>,
    pub today_input: Option<String>,
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
    /// Every bit set (`0xFFFFFFFF` a u32 word): a list's pads before any write.
    Pad,
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
    /// A dispatch's timing-sample label (a template): its GPU ms is a
    /// separate value of the run (`RunOut::dispatch_ms`), summed by label.
    pub sample: Option<String>,
    /// A setup dispatch of the machinery kernel of this name (`[[machine]]`).
    pub machine: Option<String>,
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
    /// The longest one submission of the problem's work may run before it is
    /// named a runaway (default the wgpu wait's own 4.9 s): a problem whose
    /// single dispatch is legitimately longer declares it.
    #[serde(default = "gave_up_ms")]
    pub gpu_wait_ms: f64,
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

fn gave_up_ms() -> f64 {
    4_900.0
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
    /// The per-binding byte limit the problem plans by (`binding_limit`):
    /// `"effective"` (the default, `gram::effective_buffer_limit`: 128 MiB
    /// on the M3 Max) or `"adapter"` (the adapter's own,
    /// `gram::adapter_buffer_limit`); `DeviceExecutor::for_problem` applies it.
    #[serde(default)]
    pub binding_limit: LimitRule,
    /// MACHINERY KERNELS (`[[machine]]`): hand-written kernels the setup
    /// runs on the device to fill resident buffers (v24_ingest5's key
    /// projection), each with its own binding schema; never the kernel
    /// evaluated, never evolved.
    #[serde(default)]
    pub machine: Vec<MachineSpec>,
}

/// `[[machine]]`: a machinery kernel (its WGSL file relative to the problem,
/// its entry point, its `[[machine.binding]]` schema). A setup step runs it
/// with `dispatch = { ... }` and `machine = "<name>"`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineSpec {
    pub name: String,
    pub source: String,
    pub entry: String,
    pub binding: Vec<BindingSpec>,
}

/// Which per-binding limit a problem plans by (`binding_limit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitRule {
    /// The device's storage-binding limit capped by Metal's 256 MiB queue
    /// write, halved: the Gram's, nluca's and kmeans's plans.
    #[default]
    Effective,
    /// The adapter's own storage-binding limit and largest buffer (the v24
    /// full set's 205 MB keys a seed at stage 5).
    Adapter,
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

/// A manifest's cases, its files resolved (`<problem>/<manifest>`).
fn read_cases(dir: &Path, manifest_path: &str) -> Result<Vec<Case>, String> {
    let mpath = dir.join(manifest_path);
    let manifest: Manifest = read_toml(&mpath)?;
    let mdir = mpath.parent().unwrap_or(dir).to_path_buf();
    let root = match &manifest.root {
        Some(r) if Path::new(r).is_absolute() => PathBuf::from(r),
        Some(r) => mdir.join(r),
        None => mdir,
    };
    Ok(manifest
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
        .collect())
}

/// ONE RUN OF A KERNEL'S TEXT ON ONE CASE ON THE CPU EXECUTOR (its own
/// WGSL interpreted, `cpu.rs`): compile, the case's data, its plan, its
/// resident buffers, one run with every read. A differential check (a
/// design's text against its rendering) compares the outputs' bytes.
pub fn cpu_run(problem: &Problem, case: &Case, source: &str, binding_limit: u64) -> Result<RunOut, String> {
    let exec = CpuExecutor::new(binding_limit);
    let (k, info) = exec.compile(problem, source)?;
    let data = problem.case_data(case, binding_limit)?;
    let plan = problem.plan(case, binding_limit, Some(&info))?;
    let (staged, _) = exec.stage(&plan, &data)?;
    exec.run(&k, &plan, &data, &staged, true)
}

impl Problem {
    /// The cases of a manifest that is no suite of the problem (a v24
    /// problem's `cases/admission.toml`: a small unjudged case for a
    /// differential check), its path relative to the problem.
    pub fn cases_of(&self, manifest_path: &str) -> Result<Vec<Case>, String> {
        read_cases(&self.dir, manifest_path)
    }

    /// Read `dir/problem.toml` and every suite's manifest, and check the
    /// schema (no case data is read: [`Problem::case_data`] does).
    pub fn load(dir: &Path) -> Result<Problem, String> {
        let spec: ProblemSpec = read_toml(&dir.join("problem.toml"))?;
        let reference_path = dir.join(&spec.kernel.reference);
        let reference = std::fs::read_to_string(&reference_path).map_err(|e| format!("{}: {e}", reference_path.display()))?;
        let mut suites = Vec::new();
        for s in &spec.suite {
            suites.push(Suite { spec: s.clone(), cases: read_cases(dir, &s.cases)? });
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
            if b.name == "entry" {
                return Err(format!("{name}: binding {} is named entry, the dispatch's entry-point key", b.binding));
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
                (OutputKind::Float | OutputKind::Exact | OutputKind::PooledRecall, Some(_)) => return what("a tie rule is an argmax output's"),
                (OutputKind::PooledRecall, None) if o.element != Elem::U32 || o.shape.len() != 2 || o.pooled.is_none() => return what("a pooled_recall output is u32 [stages * seeds * queries, c_max] with [output.pooled]"),
                _ => {}
            }
            if o.pooled.is_some() && o.kind() != OutputKind::PooledRecall {
                return what("[output.pooled] is a pooled_recall output's");
            }
            if let Some(ps) = &o.pooled {
                for e in [&ps.seeds, &ps.stages, &ps.queries, &ps.top, &ps.arrived, &ps.query_row] {
                    parse_expr(e).map_err(|e| format!("{name}: output {}: {e}", o.name))?;
                }
                for t in &ps.timing {
                    if t.today.is_some() == t.today_input.is_some() {
                        return what("a pooled timing sample names today's ms by exactly one of today and today_input");
                    }
                }
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
        if self.pooled_output().is_some() {
            // A pooled problem's timing samples are its constraint cases'
            // own runs (`Harness::evaluate_pooled`): no other suite.
            if let Some(x) = self.suites.iter().find(|x| x.spec.role != Role::Constraint) {
                return Err(format!("{name}: suite {} is a {:?} suite; a pooled_recall problem has its constraint suite only", x.spec.name, x.spec.role));
            }
        } else if count(Role::Timing) == 0 {
            return Err(format!("{name}: no timing case; the fitness's three speed columns are bands of the timing cases (`timing_bands`)"));
        }
        match (s.budget.deadline_factor, s.budget.deadline_floor_ms) {
            (None, None) => {}
            (Some(f), Some(m)) if f >= 1.0 && m > 0.0 => {}
            (f, m) => return Err(format!("{name}: [budget] deadline_factor ({f:?}, at least 1) and deadline_floor_ms ({m:?}, above 0) go together")),
        }
        if self.suites.iter().any(|x| x.spec.staged && (x.spec.role != Role::Constraint || self.pooled_output().is_some())) {
            return Err(format!("{name}: only the constraint suite of a problem that is not pooled_recall can be staged"));
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

    /// The constraint suite's output when it is `pooled_recall`.
    pub fn pooled_output(&self) -> Option<&OutputSpec> {
        let su = self.suites.iter().find(|s| s.spec.role == Role::Constraint)?;
        self.output_of(su).ok().filter(|o| o.kind() == OutputKind::PooledRecall)
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
    /// Each entry point's workgroup size, in `KernelSpec::entry_names` order
    /// (the program reads `wg_x_<entry>`, `wg_y_<entry>`, `wg_z_<entry>`).
    pub entry_workgroups: Vec<[u32; 3]>,
    /// Its `// passes N` (1 when the problem reads no passes header).
    pub passes: u32,
}

/// A kernel's `// passes N` header: N in 1..=[`MAX_PASSES`], 1 when absent;
/// a malformed or out-of-range header, or two that differ, is an error.
pub fn passes_header(source: &str) -> Result<u32, String> {
    let mut found: Option<u32> = None;
    for l in source.lines() {
        let Some(n) = l.trim().strip_prefix("// passes ") else { continue };
        let n: u32 = n.trim().parse().map_err(|_| format!("the passes header {:?} is not a count", l.trim()))?;
        if !(1..=MAX_PASSES).contains(&n) {
            return Err(format!("the passes header says {n}, outside 1..={MAX_PASSES}"));
        }
        if found.is_some_and(|f| f != n) {
            return Err(format!("two passes headers, {} and {n}", found.unwrap_or(n)));
        }
        found = Some(n);
    }
    Ok(found.unwrap_or(1))
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
        // A storage atomic's word is its scalar (`array<atomic<u32>>` is a u32 binding).
        TypeInner::Scalar(s) | TypeInner::Atomic(s) => out.push(elem(*s)?),
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
    let module = crate::wgsl::naga_shape::parse(source)?;
    super::target::Target::current().validate(&module, source).map_err(|e| format!("validation: {e}"))?;
    let mut entry_workgroups = Vec::new();
    for entry in problem.kernel.entry_names() {
        let ep = module.entry_points.iter().find(|e| e.name == entry && e.stage == naga::ShaderStage::Compute).ok_or_else(|| format!("no compute entry point {entry}"))?;
        entry_workgroups.push(ep.workgroup_size);
    }
    let passes = if problem.kernel.passes_header { passes_header(source)? } else { 1 };
    let [x, y, z] = entry_workgroups[0];
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
    Ok(KernelInfo { workgroup: [x, y, z], bx, by, rows_x, modes, entry_workgroups, passes })
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
    /// Every byte 0xFF (`fill = "pad"`).
    Pad,
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
    /// One compute pass: `entry` indexes `KernelSpec::entry_names`; its GPU
    /// ms is one value of `RunOut::dispatch_ms`, under its `sample` label.
    Dispatch { bind: Vec<(u32, BufId)>, workgroups: [u32; 3], entry: usize, sample: Option<String> },
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
    /// The longest one of its submissions may run (`Budget::gpu_wait_ms`).
    pub gpu_wait_ms: f64,
    /// The machinery kernels the setup runs (`[[machine]]`, their text read),
    /// and the setup's dispatches of them, in order, run on the device when
    /// the case is staged.
    pub machines: Vec<MachineCode>,
    pub setup_dispatch: Vec<SetupDispatch>,
}

/// A machinery kernel as a plan carries it.
#[derive(Debug, Clone, PartialEq)]
pub struct MachineCode {
    pub name: String,
    pub source: String,
    pub entry: String,
    pub bindings: Vec<(u32, BindingKind)>,
}

/// One setup dispatch of a machinery kernel: its index in `Plan::machines`,
/// its buffers by binding, its workgroups.
#[derive(Debug, Clone, PartialEq)]
pub struct SetupDispatch {
    pub machine: usize,
    pub bind: Vec<(u32, BufId)>,
    pub workgroups: [u32; 3],
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
    setup_dispatch: Vec<SetupDispatch>,
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

    /// A setup dispatch of a machinery kernel: every binding of its schema
    /// bound to a buffer made in the setup (a resident it may write: that is
    /// what it is for), its workgroups.
    fn machine_dispatch(&mut self, s: &Step, name: &str, env: &Env, resident: bool) -> Result<(), String> {
        if !resident {
            return Err(format!("machine {name}: a machinery kernel runs in the setup only"));
        }
        let machine = self.problem.machine.iter().position(|m| m.name == name).ok_or_else(|| format!("no machine {name}"))?;
        let spec = &self.problem.machine[machine];
        let map = s.dispatch.as_ref().ok_or_else(|| format!("machine {name}: give its bindings as dispatch = {{ ... }}"))?;
        let mut bind = Vec::new();
        for b in &spec.binding {
            let t = map.get(&b.name).ok_or_else(|| format!("machine {name} leaves binding {} unbound", b.name))?;
            let id = self.buffer(&template(t, env)?)?;
            let buf = &self.buffers[id];
            match b.kind {
                BindingKind::Uniform => {
                    let want = 4 * b.fields.as_ref().map_or(0, |f| f.len()) as u64;
                    if !buf.uniform || buf.bytes != want {
                        return Err(format!("machine {name}: binding {} wants a uniform of {want} bytes, {} is not one", b.name, buf.name));
                    }
                }
                BindingKind::StorageRead | BindingKind::StorageReadWrite if buf.uniform => return Err(format!("machine {name}: binding {} is storage, {} is a uniform", b.name, buf.name)),
                _ => {}
            }
            if bind.iter().any(|(_, x)| *x == id) {
                return Err(format!("machine {name}: alias: {} is bound twice", buf.name));
            }
            bind.push((b.binding, id));
        }
        if let Some(extra) = map.keys().find(|k| !spec.binding.iter().any(|b| &b.name == *k)) {
            return Err(format!("machine {name} binds {extra}, which is not in its schema"));
        }
        let wg = s.workgroups.as_deref().ok_or("a machine dispatch needs workgroups")?;
        if wg.len() != 3 {
            return Err("workgroups takes three expressions".into());
        }
        let mut workgroups = [0u32; 3];
        for (w, e) in workgroups.iter_mut().zip(wg) {
            let v = eval_int(e, env)?;
            *w = u32::try_from(v).ok().filter(|v| *v <= 65_535).ok_or_else(|| format!("workgroups {e:?} = {v}, over 65,535"))?;
        }
        self.setup_dispatch.push(SetupDispatch { machine, bind, workgroups });
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
        if s.sample.is_some() && s.dispatch.is_none() {
            return Err(format!("sample labels a dispatch, not this step: {s:?}"));
        }
        if let Some(name) = &s.machine {
            return self.machine_dispatch(s, name, env, resident);
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
                (Some(Fill::Pad), None) => Init::Pad,
                (None, Some(t)) => {
                    let t = &template(t, env)?;
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
                _ => return Err(format!("buffer {name}: give fill = \"zero\", fill = \"pad\" or from")),
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
            let machine_binding = of.split_once('.').and_then(|(m, b)| self.problem.machine.iter().find(|x| x.name == m).and_then(|x| x.binding.iter().find(|y| y.name == b)));
            let spec = self.problem.binding.iter().find(|b| b.name == of).or(machine_binding).filter(|b| b.kind == BindingKind::Uniform).ok_or_else(|| format!("uniform {name}: no uniform binding {of}"))?;
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
            let names = self.problem.kernel.entry_names();
            let entry = match map.get("entry") {
                Some(e) => names.iter().position(|n| n == e).ok_or_else(|| format!("a dispatch names entry {e}, which [kernel] does not list"))?,
                None => 0,
            };
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
            if let Some(extra) = map.keys().find(|k| k.as_str() != "entry" && !self.problem.binding.iter().any(|b| &b.name == *k)) {
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
            let sample = s.sample.as_deref().map(|x| template(x, env)).transpose()?;
            self.run.push(Cmd::Dispatch { bind, workgroups, entry, sample });
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
        let mut b = Builder { problem: &self.spec, buffers: Vec::new(), names: BTreeMap::new(), setup: Vec::new(), run: Vec::new(), out_bytes: out_bytes.clone(), setup_dispatch: Vec::new() };
        b.steps(&self.spec.setup, &env, true).map_err(|e| format!("case {}: setup: {e}", case.name))?;
        if let Some(k) = kernel {
            let mut env = env;
            for (n, v) in [("wg_x", k.workgroup[0]), ("wg_y", k.workgroup[1]), ("wg_z", k.workgroup[2]), ("bx", k.bx), ("by", k.by), ("rows_x", u32::from(k.rows_x))] {
                env.insert(n.into(), f64::from(v));
            }
            for m in &self.spec.kernel.modes {
                env.insert(format!("mode_{m}"), if k.modes.contains(m) { 1.0 } else { 0.0 });
            }
            for (e, wg) in self.spec.kernel.entry_names().iter().zip(&k.entry_workgroups) {
                for (axis, v) in ["x", "y", "z"].iter().zip(wg) {
                    env.insert(format!("wg_{axis}_{e}"), f64::from(*v));
                }
            }
            env.insert("passes".into(), f64::from(k.passes));
            bind_lets(&self.spec.derive_run, &mut env).map_err(|e| format!("case {}: {e}", case.name))?;
            b.steps(&self.spec.run, &env, false).map_err(|e| format!("case {}: run: {e}", case.name))?;
        }
        let mut machines = Vec::new();
        for m in &self.spec.machine {
            let path = self.dir.join(&m.source);
            let source = std::fs::read_to_string(&path).map_err(|e| format!("machine {}: {}: {e}", m.name, path.display()))?;
            machines.push(MachineCode { name: m.name.clone(), source, entry: m.entry.clone(), bindings: m.binding.iter().map(|b| (b.binding, b.kind)).collect() });
        }
        let mut plan = Plan { buffers: b.buffers, setup: b.setup, run: b.run, outputs: out_bytes, batch_safe: false, gpu_wait_ms: self.spec.budget.gpu_wait_ms, machines, setup_dispatch: b.setup_dispatch };
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
        Init::Pad => vec![0xFFu8; b.bytes as usize],
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
    /// Every dispatch's own GPU ms, in the plan's dispatch order (each a
    /// separate timing sample: `pooled_samples` sums them by label).
    pub dispatch_ms: Vec<f64>,
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
    /// THE POOLED JUDGE of one run's `pooled_recall` output (`bytes`): the
    /// windows' counts (`PooledWindow`). The device re-ranks on the device
    /// (`pooled_judge.wgsl`, the exact dot in fixed point); the host and CPU
    /// test executors by [`pooled_windows`], the reference the device's is
    /// tested against.
    fn judge_pooled(&self, problem: &Problem, out: &OutputSpec, env: &Env, data: &CaseData, bytes: &[u8]) -> Result<Vec<PooledWindow>, String>;
    /// One case run `n` times for timing, nothing read: the summed GPU ms
    /// and the longest compute ms a run (the staged slow gate's measure).
    /// One run after another. (Batching the runs into one submission,
    /// fuller 6c6179a-b2f119c, left the wait for the GPU blocked at 0 % CPU on
    /// nluca at 10,000 and 100,000 rows, 2026-10-15, cause not found: taken
    /// out until it is.)
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
    /// The dispatch's entry point, an index of `KernelSpec::entry_names`.
    pub entry: usize,
}

/// The byte offset of word `min(index, words - 1)` of a binding of `len` bytes; None when it holds no whole word.
fn restricted_word(len: usize, index: u32) -> Option<usize> {
    (len / 4).checked_sub(1).map(|last| (index as usize).min(last) * 4)
}

/// The buffers bound to one dispatch, by binding number, under wgpu's
/// runtime bounds checks on the device (wgpu 30 on Metal: naga's `Restrict`,
/// the index is `min(index, len - 1)`; `tests/runtime_bounds_checks.rs`): a load
/// out of range reads the LAST element of the binding and a store out of range
/// writes it; a store to a read-only binding is skipped (WGSL forbids one). An
/// empty binding has no element: a load is zero, a store is skipped.
pub struct HostMemory {
    slots: Vec<(u32, Vec<u8>, bool)>,
}

impl HostMemory {
    fn slot(&self, binding: u32) -> Option<&(u32, Vec<u8>, bool)> {
        self.slots.iter().find(|s| s.0 == binding)
    }

    pub fn load_u32(&self, binding: u32, index: u32) -> u32 {
        let Some((_, b, _)) = self.slot(binding) else { return 0 };
        let Some(at) = restricted_word(b.len(), index) else { return 0 };
        u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
    }

    pub fn load_f32(&self, binding: u32, index: u32) -> f32 {
        f32::from_bits(self.load_u32(binding, index))
    }

    pub fn store_u32(&mut self, binding: u32, index: u32, v: u32) {
        let Some((_, b, writable)) = self.slots.iter_mut().find(|s| s.0 == binding) else { return };
        if !*writable {
            return;
        }
        if let Some(at) = restricted_word(b.len(), index) {
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
                    Cmd::Dispatch { bind, workgroups, entry, .. } => {
                        let mut mem = HostMemory { slots: Vec::new() };
                        let kinds: Vec<bool> = bind.iter().map(|(b, _)| items[i].writable.get(b).copied().unwrap_or(false)).collect();
                        for ((b, id), w) in bind.iter().zip(&kinds) {
                            let data = items[i].bufs[*id].clone().ok_or("a dispatch of a buffer never made")?;
                            mem.slots.push((*b, data, *w));
                        }
                        let mut ops = 0u64;
                        let wg = info.entry_workgroups[*entry];
                        for wz in 0..workgroups[2] {
                            for wy in 0..workgroups[1] {
                                for wx in 0..workgroups[0] {
                                    for lz in 0..wg[2] {
                                        for ly in 0..wg[1] {
                                            for lx in 0..wg[0] {
                                                let ids = HostIds { gid: [wx * wg[0] + lx, wy * wg[1] + ly, wz * wg[2] + lz], lid: [lx, ly, lz], wid: [wx, wy, wz], nwg: *workgroups, entry: *entry };
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
                        items[i].dispatch_ms.push(ms);
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
    dispatch_ms: Vec<f64>,
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
        HostItem { plan, data, bufs, writable: writable.clone(), outputs: plan.outputs.iter().map(|n| vec![0u8; *n as usize]).collect(), gpu_ms: 0.0, dispatch_ms: Vec::new() }
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

    fn judge_pooled(&self, problem: &Problem, out: &OutputSpec, env: &Env, data: &CaseData, bytes: &[u8]) -> Result<Vec<PooledWindow>, String> {
        pooled_windows(problem, out, env, data, bytes)
    }

    fn compile(&self, problem: &Problem, source: &str) -> Result<(HostCompiled, KernelInfo), String> {
        let info = kernel_info(&problem.spec, source)?;
        let model = self.models.iter().position(|(s, _)| s == source).ok_or("host executor: no model is registered for this kernel text")?;
        let writable = problem.spec.binding.iter().map(|b| (b.binding, b.kind == BindingKind::StorageReadWrite)).collect();
        Ok((HostCompiled { model, info: info.clone(), writable }, info))
    }

    fn stage(&self, plan: &Plan, data: &CaseData) -> Result<(HostStaged, f64), String> {
        if !plan.setup_dispatch.is_empty() {
            return Err("our software has a bug: this problem's setup runs a machinery kernel ([[machine]]) on the device; the host executor does not run one".into());
        }
        let bufs = plan.setup.iter().map(|id| initial_bytes(&plan.buffers[*id], data).map(|b| (*id, b))).collect::<Result<_, _>>()?;
        Ok((HostStaged { bufs }, 0.0))
    }

    fn run(&self, k: &HostCompiled, plan: &Plan, data: &CaseData, staged: &HostStaged, read: bool) -> Result<RunOut, String> {
        let mut items = vec![self.item(plan, data, staged, &k.writable)];
        let compute = self.exec(k.model, &k.info, &mut items, false, read)?;
        let it = items.pop().expect("one item");
        Ok(RunOut { outputs: if read { it.outputs } else { Vec::new() }, gpu_ms: it.gpu_ms, compute_ms: compute, wall_ms: compute, dispatch_ms: it.dispatch_ms })
    }

    fn run_batch(&self, k: &HostCompiled, items: &[(&Plan, &CaseData, &HostStaged)]) -> Result<(Vec<RunOut>, f64), String> {
        if let Some((p, _, _)) = items.iter().find(|(p, _, _)| !p.batch_safe) {
            return Err(format!("a plan of {} commands is not batch-safe", p.run.len()));
        }
        let mut its: Vec<HostItem> = items.iter().map(|(p, d, s)| self.item(p, d, s, &k.writable)).collect();
        let compute = self.exec(k.model, &k.info, &mut its, true, true)?;
        Ok((its.into_iter().map(|it| RunOut { outputs: it.outputs, gpu_ms: it.gpu_ms, compute_ms: compute, wall_ms: compute, dispatch_ms: it.dispatch_ms }).collect(), compute))
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
    /// `KernelSpec::entry_names`, a dispatch's entry its index here.
    entries: Vec<String>,
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
    /// of its dispatches, and each dispatch's own.
    fn exec(&self, k: &CpuKernel, plan: &Plan, data: &CaseData, staged: &CpuStaged, read: bool, outputs: &mut [Vec<u8>]) -> Result<(f64, Vec<f64>), String> {
        let mut bufs: Vec<Option<Vec<u8>>> = vec![None; plan.buffers.len()];
        for (id, b) in &staged.bufs {
            bufs[*id] = Some(b.clone());
        }
        let mut ms = 0.0f64;
        let mut each = Vec::new();
        for c in &plan.run {
            match c {
                Cmd::Create(id) => bufs[*id] = Some(initial_bytes(&plan.buffers[*id], data)?),
                Cmd::Clear(id) => bufs[*id].as_mut().ok_or("a clear of a buffer never made")?.iter_mut().for_each(|x| *x = 0),
                Cmd::Dispatch { bind, workgroups, entry, .. } => {
                    let mut vals = std::collections::HashMap::new();
                    for (b, id) in bind {
                        let Some((name, shape, _)) = k.bindings.get(b) else { continue };
                        let bytes = bufs[*id].as_ref().ok_or("a dispatch of a buffer never made")?;
                        vals.insert(name.clone(), Self::value(k, shape, bytes)?);
                    }
                    let t = Instant::now();
                    let d = crate::wgsl::cpu::run_entry(&k.module, &k.entries[*entry], vals, *workgroups, self.max_steps)?;
                    let one = t.elapsed().as_secs_f64() * 1e3;
                    ms += one;
                    each.push(one);
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
        Ok((ms.max(1e-6), each))
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

    fn judge_pooled(&self, problem: &Problem, out: &OutputSpec, env: &Env, data: &CaseData, bytes: &[u8]) -> Result<Vec<PooledWindow>, String> {
        pooled_windows(problem, out, env, data, bytes)
    }

    fn compile(&self, problem: &Problem, source: &str) -> Result<(CpuKernel, KernelInfo), String> {
        let info = kernel_info(&problem.spec, source)?;
        let module = crate::wgsl::naga_shape::parse(source)?;
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
        Ok((CpuKernel { module, bindings, entries: problem.spec.kernel.entry_names() }, info))
    }

    fn stage(&self, plan: &Plan, data: &CaseData) -> Result<(CpuStaged, f64), String> {
        if !plan.setup_dispatch.is_empty() {
            return Err("our software has a bug: this problem's setup runs a machinery kernel ([[machine]]) on the device; the CPU executor does not run one".into());
        }
        let bufs = plan.setup.iter().map(|id| initial_bytes(&plan.buffers[*id], data).map(|b| (*id, b))).collect::<Result<_, _>>()?;
        Ok((CpuStaged { bufs }, 0.0))
    }

    fn run(&self, k: &CpuKernel, plan: &Plan, data: &CaseData, staged: &CpuStaged, read: bool) -> Result<RunOut, String> {
        let mut outputs: Vec<Vec<u8>> = plan.outputs.iter().map(|n| vec![0u8; *n as usize]).collect();
        let (ms, dispatch_ms) = self.exec(k, plan, data, staged, read, &mut outputs)?;
        Ok(RunOut { outputs: if read { outputs } else { Vec::new() }, gpu_ms: ms, compute_ms: ms, wall_ms: ms, dispatch_ms })
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

    use super::{initial_bytes, judge_inputs, kernel_info, windows_of_judged, BindingKind, BufId, CaseData, Cmd, Env, Executor, KernelInfo, OutputSpec, Plan, PooledWindow, Problem, RunOut};
    use crate::wgsl::gram::{adapter_buffer_limit, effective_buffer_limit, wait_gpu, wait_gpu_within};
    use crate::wgsl::oracle::device::{Device, Fuel};

    /// A bound of one dispatch's node executions an invocation, from its entry (index into
    /// `KernelSpec::entry_names`) and its uniform's words (the run's own field values): the work guard of a kernel of
    /// the true kingdom read per dispatch (`TrueKingdom::tree_work_bound_in`).
    pub type DispatchBound = Box<dyn Fn(usize, &[u8]) -> Result<u128, String>>;

    pub struct DeviceExecutor<'a> {
        pub dev: &'a Device,
        limit: u64,
        /// With a dispatch bound, every dispatch is checked against the per-submit cap by its own bound, and a
        /// segment is submitted early (at a dispatch boundary) rather than let its bound pass the cap: the values
        /// are those of one submission a segment, the timings are not. `None`: the segment's bound by [`Fuel`].
        dispatch_bound: Option<DispatchBound>,
        /// The work guard (`work_guard`): `Fuel::Proven` for the shipped and hand kernels; a candidate of a kingdom
        /// with a fuel budget gets [`DeviceExecutor::with_fuel`], and every submit is checked against the cap
        /// before it is made.
        fuel: Fuel,
        /// The pooled judge's pipeline, made on first use, and each judged
        /// case's resident tables (its records, accept sets, query rows),
        /// uploaded once a case (keyed by the case data's address, which lives
        /// as long as its stage) and reused by every individual.
        judge: std::cell::RefCell<Option<Judge>>,
    }

    struct Judge {
        bgl: wgpu::BindGroupLayout,
        pipeline: wgpu::ComputePipeline,
        resident: Vec<(usize, JudgeResident)>,
    }

    struct JudgeResident {
        bufs: [wgpu::Buffer; 7],
        params: [u32; 8],
    }

    impl<'a> DeviceExecutor<'a> {
        pub fn new(dev: &'a Device, fuel: Fuel) -> DeviceExecutor<'a> {
            DeviceExecutor { dev, limit: effective_buffer_limit(&dev.device), dispatch_bound: None, fuel, judge: std::cell::RefCell::new(None) }
        }

        /// The work guard per dispatch ([`DispatchBound`]), for a value check (one submission a segment is not kept).
        pub fn with_dispatch_bound(mut self, f: DispatchBound) -> DeviceExecutor<'a> {
            self.dispatch_bound = Some(f);
            self
        }


        /// The device executor at the binding limit `problem` plans by
        /// (`ProblemSpec::binding_limit`: the effective limit, or the
        /// adapter's own).
        pub fn for_problem(dev: &'a Device, problem: &super::Problem, fuel: Fuel) -> DeviceExecutor<'a> {
            let limit = match problem.spec.binding_limit {
                super::LimitRule::Effective => effective_buffer_limit(&dev.device),
                super::LimitRule::Adapter => adapter_buffer_limit(&dev.device),
            };
            DeviceExecutor { dev, limit, dispatch_bound: None, fuel, judge: std::cell::RefCell::new(None) }
        }
    }

    /// A kernel's pipelines, one per entry point (`KernelSpec::entry_names`
    /// order), over one explicit layout.
    pub struct DeviceKernel {
        pipelines: Vec<wgpu::ComputePipeline>,
        /// Invocations per workgroup of each entry point (the work guard's workgroup size).
        workgroup_invocations: Vec<u64>,
        bgl: wgpu::BindGroupLayout,
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
        dispatch_ms: Vec<f64>,
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
        /// The invocations per workgroup of compute entry `entry` of `source` (the work guard's workgroup size).
        fn entry_invocations(source: &str, entry: &str) -> Result<u64, String> {
            let module = crate::wgsl::naga_shape::parse(source)?;
            let ep = module.entry_points.iter().find(|e| e.name == entry && e.stage == naga::ShaderStage::Compute).ok_or_else(|| format!("no compute entry point {entry}"))?;
            Ok(ep.workgroup_size.iter().map(|x| u64::from(*x)).product())
        }

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
            let layout = &k.bgl;
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
                /// The invocations of the dispatches encoded so far: the work guard's input.
                invocations: u64,
                /// With a dispatch bound: the node executions the encoded dispatches are bounded by.
                bound: u128,
            }
            let new_seg = || Seg { enc: device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("harness segment") }), queries: Vec::new(), reads: Vec::new(), keep: Vec::new(), any: false, invocations: 0, bound: 0 };
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
            // The longest a submission may run before it is a runaway: the
            // largest any item's problem declares (`Budget::gpu_wait_ms`).
            let wait_ms = live.iter().map(|l| l.plan.gpu_wait_ms).fold(0.0, f64::max);
            let flush = |seg: Seg, live: &mut [Live], compute_ms: &mut f64| -> Result<(), String> {
                if !seg.any && seg.reads.is_empty() {
                    return Ok(());
                }
                let t = Instant::now();
                queue.submit(Some(seg.enc.finish()));
                wait_gpu_within(device, queue, wait_ms)?;
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
                wait_gpu_within(device, queue, wait_ms)?;
                for _ in 0..waits {
                    rx.recv().map_err(|e| format!("map: {e}"))?.map_err(|e| format!("map: {e}"))?;
                }
                if let Some(st) = &staging {
                    {
                        let mapped = crate::gpu_io::mapped(&st.slice(..))?;
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
                        let mapped = crate::gpu_io::mapped(&stamps_read.slice(..8 * u64::from(nq)))?;
                        let stamps: &[u64] = bytemuck::cast_slice(&mapped);
                        for (q, item) in seg.queries.iter().enumerate() {
                            let ms = stamps[2 * q + 1].wrapping_sub(stamps[2 * q]) as f64 * f64::from(k.period_ns) / 1e6;
                            live[*item].gpu_ms += ms;
                            live[*item].dispatch_ms.push(ms);
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
                        Cmd::Dispatch { bind, workgroups, entry, .. } => {
                            // The work guard, before the dispatch is encoded: the segment is one submit, and
                            // nothing of it has been submitted when the bound is over the cap.
                            let invocations = workgroups.iter().fold(1u64, |a, w| a.saturating_mul(u64::from(*w))).saturating_mul(k.workgroup_invocations[*entry]);
                            match &self.dispatch_bound {
                                None => {
                                    seg.invocations = seg.invocations.saturating_add(invocations);
                                    self.fuel.check("a harness segment", seg.invocations)?;
                                }
                                Some(bound) => {
                                    let words: &[u8] = bind
                                        .iter()
                                        .find_map(|(_, id)| match &live[i].plan.buffers[*id] {
                                            super::BufferDecl { uniform: true, init: super::Init::Words(w), .. } => Some(w.as_slice()),
                                            _ => None,
                                        })
                                        .unwrap_or(&[]);
                                    let per = bound(*entry, words)?;
                                    let this = per.saturating_mul(u128::from(invocations));
                                    let cap = crate::wgsl::work_guard::CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT;
                                    if this > cap {
                                        return Err(format!("the work guard: a dispatch of entry {} is bounded by {this} node executions ({per} an invocation, {invocations} invocations), over the per-submit cap {cap}; nothing of it was submitted", *entry));
                                    }
                                    if seg.bound.saturating_add(this) > cap {
                                        let done = std::mem::replace(&mut seg, new_seg());
                                        flush(done, live, &mut compute_ms)?;
                                    }
                                    seg.bound = seg.bound.saturating_add(this);
                                }
                            }
                            let entries: Vec<wgpu::BindGroupEntry> = bind.iter().map(|(b, id)| live[i].buf(*id, "a dispatch").map(|buf| wgpu::BindGroupEntry { binding: *b, resource: buf.as_entire_binding() })).collect::<Result<_, _>>()?;
                            let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("harness bind"), layout, entries: &entries });
                            let q = seg.queries.len() as u32;
                            seg.queries.push(i);
                            let timestamp_writes = Some(wgpu::ComputePassTimestampWrites { query_set: &query_set, beginning_of_pass_write_index: Some(2 * q), end_of_pass_write_index: Some(2 * q + 1) });
                            let mut pass = seg.enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("harness dispatch"), timestamp_writes });
                            pass.set_pipeline(&k.pipelines[*entry]);
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
            crate::gpu_io::drain(device)?;
            Ok(compute_ms)
        }
    }

    impl DeviceExecutor<'_> {
        /// THE POOLED JUDGE ON THE DEVICE (`pooled_judge.wgsl`): the case's
        /// tables resident from its first judge, the run's candidates
        /// uploaded, one invocation a (stage, query), its (found, pooled,
        /// broken) read back and summed into windows.
        fn judge_on_device(&self, problem: &Problem, out: &OutputSpec, env: &Env, data: &CaseData, bytes: &[u8]) -> Result<Vec<PooledWindow>, String> {
            let device = &self.dev.device;
            let mut judge = self.judge.borrow_mut();
            if judge.is_none() {
                let scope = crate::wgsl::oracle::device::ErrorScopes::push(device);
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("pooled judge"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("pooled_judge.wgsl"))) });
                let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..10u32)
                    .map(|b| wgpu::BindGroupLayoutEntry {
                        binding: b,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { ty: if b == 0 { wgpu::BufferBindingType::Uniform } else { wgpu::BufferBindingType::Storage { read_only: b != 9 } }, has_dynamic_offset: false, min_binding_size: None },
                        count: None,
                    })
                    .collect();
                let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("pooled judge"), entries: &entries });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("pooled judge"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("judge_main"), layout: Some(&layout), module: &shader, entry_point: Some("judge_main"), compilation_options: Default::default(), cache: None });
                if let Some(e) = scope.pop() {
                    return Err(format!("our software has a bug: the pooled judge does not compile: {e}"));
                }
                *judge = Some(Judge { bgl, pipeline, resident: Vec::new() });
            }
            let judge = judge.as_mut().expect("made above");
            let key = data as *const CaseData as usize;
            let inp = judge_inputs(problem, out, env, data, bytes.len())?;
            if !judge.resident.iter().any(|(k, _)| *k == key) {
                let buf = |label: &str, words: &[u32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytemuck::cast_slice(if words.is_empty() { &[0u32][..] } else { words }), usage: wgpu::BufferUsages::STORAGE });
                let bufs = [buf("rec_offsets", &inp.rec_offsets), buf("rec_indices", &inp.rec_indices), buf("rec_values", &inp.rec_values), buf("qrow", &inp.qrow), buf("arrived", &inp.arrived), buf("accept_offsets", &inp.accept_offsets), buf("accept_rows", &inp.accept_rows)];
                judge.resident.push((key, JudgeResident { bufs, params: [inp.stages as u32, inp.seeds as u32, inp.queries as u32, inp.c_max as u32, inp.top as u32, 0, 0, 0] }));
            }
            let r = &judge.resident.iter().find(|(k, _)| *k == key).expect("resident above").1;
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("judge params"), contents: bytemuck::cast_slice(&r.params), usage: wgpu::BufferUsages::UNIFORM });
            let cands = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("judge cands"), contents: bytes, usage: wgpu::BufferUsages::STORAGE });
            let n = inp.stages * inp.queries;
            let size = (n * 3 * 4) as u64;
            let judged = device.create_buffer(&wgpu::BufferDescriptor { label: Some("judged"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("judged read"), size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let order: [&wgpu::Buffer; 10] = [&params, &cands, &r.bufs[0], &r.bufs[1], &r.bufs[2], &r.bufs[3], &r.bufs[4], &r.bufs[5], &r.bufs[6], &judged];
            let entries: Vec<wgpu::BindGroupEntry> = order.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("pooled judge"), layout: &judge.bgl, entries: &entries });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pooled judge") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pooled judge"), timestamp_writes: None });
                pass.set_pipeline(&judge.pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
            }
            enc.copy_buffer_to_buffer(&judged, 0, &staging, 0, size);
            self.dev.queue.submit(Some(enc.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).expect("the receiver waits for the mapping");
            });
            wait_gpu(device, &self.dev.queue)?;
            rx.recv().map_err(|e| format!("map: {e}"))?.map_err(|e| format!("map: {e}"))?;
            let words: Vec<u32> = bytemuck::cast_slice(&crate::gpu_io::mapped(&staging.slice(..))?).to_vec();
            staging.unmap();
            for b in [params, cands, judged, staging] {
                b.destroy();
            }
            windows_of_judged(&inp, &words)
        }
    }

    impl<'a> Executor for DeviceExecutor<'a> {
        type Kernel = DeviceKernel;
        type Staged = DeviceStaged;

        fn judge_pooled(&self, problem: &Problem, out: &OutputSpec, env: &Env, data: &CaseData, bytes: &[u8]) -> Result<Vec<PooledWindow>, String> {
            self.judge_on_device(problem, out, env, data, bytes)
        }

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
            let scope = crate::wgsl::oracle::device::ErrorScopes::push(&dev.device);
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
            let layout = dev.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("harness"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
            // MULTI-ENTRY: one pipeline per entry point, every one over the layout.
            let pipelines = problem.spec.kernel.entry_names().iter().map(|e| dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(e), layout: Some(&layout), module: &shader, entry_point: Some(e), compilation_options: Default::default(), cache: None })).collect();
            if let Some(e) = scope.pop() {
                return Err(format!("pipeline: {e}"));
            }
            let workgroup_invocations = problem.spec.kernel.entry_names().iter().map(|e| Self::entry_invocations(source, e)).collect::<Result<Vec<_>, _>>()?;
            Ok((DeviceKernel { pipelines, workgroup_invocations, bgl, period_ns: dev.queue.get_timestamp_period() }, info))
        }

        fn stage(&self, plan: &Plan, data: &CaseData) -> Result<(DeviceStaged, f64), String> {
            let t = Instant::now();
            let bufs = plan.setup.iter().map(|id| self.make(plan, *id, data).map(|b| (*id, b))).collect::<Result<Vec<_>, _>>()?;
            self.dev.queue.submit(None);
            wait_gpu(&self.dev.device, &self.dev.queue)?;
            // MACHINERY: the setup's dispatches of its machinery kernels fill
            // resident buffers on the device (one submission each, waited for
            // within the problem's gpu_wait_ms), nothing read back.
            if !plan.setup_dispatch.is_empty() {
                let device = &self.dev.device;
                let mut compiled = Vec::with_capacity(plan.machines.len());
                for m in &plan.machines {
                    let scope = crate::wgsl::oracle::device::ErrorScopes::push(device);
                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&m.name), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&m.source)) });
                    let entries: Vec<wgpu::BindGroupLayoutEntry> = m
                        .bindings
                        .iter()
                        .map(|(b, kind)| {
                            let ty = match kind {
                                BindingKind::StorageRead => wgpu::BufferBindingType::Storage { read_only: true },
                                BindingKind::StorageReadWrite => wgpu::BufferBindingType::Storage { read_only: false },
                                BindingKind::Uniform => wgpu::BufferBindingType::Uniform,
                            };
                            wgpu::BindGroupLayoutEntry { binding: *b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None }
                        })
                        .collect();
                    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(&m.name), entries: &entries });
                    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(&m.name), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
                    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(&m.name), layout: Some(&layout), module: &shader, entry_point: Some(&m.entry), compilation_options: Default::default(), cache: None });
                    if let Some(e) = scope.pop() {
                        return Err(format!("our software has a bug: the machinery kernel {} does not compile: {e}", m.name));
                    }
                    compiled.push((pipeline, bgl));
                }
                let buf = |id: BufId| bufs.iter().find(|(b, _)| *b == id).map(|(_, b)| b).ok_or_else(|| format!("a machine dispatch of {}, not a setup buffer", plan.buffers[id].name));
                for d in &plan.setup_dispatch {
                    let (pipeline, bgl) = &compiled[d.machine];
                    let entries: Vec<wgpu::BindGroupEntry> = d.bind.iter().map(|(b, id)| buf(*id).map(|x| wgpu::BindGroupEntry { binding: *b, resource: x.as_entire_binding() })).collect::<Result<_, _>>()?;
                    let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("harness machine"), layout: bgl, entries: &entries });
                    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("harness machine") });
                    {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("harness machine"), timestamp_writes: None });
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, &group, &[]);
                        pass.dispatch_workgroups(d.workgroups[0], d.workgroups[1], d.workgroups[2]);
                    }
                    self.dev.queue.submit(Some(enc.finish()));
                    wait_gpu_within(device, &self.dev.queue, plan.gpu_wait_ms)?;
                }
            }
            Ok((DeviceStaged { bufs }, t.elapsed().as_secs_f64() * 1e3))
        }

        fn run(&self, k: &DeviceKernel, plan: &Plan, data: &CaseData, staged: &DeviceStaged, read: bool) -> Result<RunOut, String> {
            let t = Instant::now();
            let mut live = vec![Live { plan, data, bufs: (0..plan.buffers.len()).map(|_| None).collect(), outputs: plan.outputs.iter().map(|n| vec![0u8; *n as usize]).collect(), gpu_ms: 0.0, dispatch_ms: Vec::new() }];
            let compute_ms = self.exec(k, &mut live, &[staged], false, read)?;
            let l = live.pop().expect("one item");
            Ok(RunOut { outputs: if read { l.outputs } else { Vec::new() }, gpu_ms: l.gpu_ms, compute_ms, wall_ms: t.elapsed().as_secs_f64() * 1e3, dispatch_ms: l.dispatch_ms })
        }

        fn run_batch(&self, k: &DeviceKernel, items: &[(&Plan, &CaseData, &DeviceStaged)]) -> Result<(Vec<RunOut>, f64), String> {
            if items.iter().any(|(p, _, _)| !p.batch_safe) {
                return Err("a plan in the batch is not batch-safe".into());
            }
            let t = Instant::now();
            let mut live: Vec<Live> = items.iter().map(|(p, d, _)| Live { plan: p, data: d, bufs: (0..p.buffers.len()).map(|_| None).collect(), outputs: p.outputs.iter().map(|n| vec![0u8; *n as usize]).collect(), gpu_ms: 0.0, dispatch_ms: Vec::new() }).collect();
            let staged: Vec<&DeviceStaged> = items.iter().map(|(_, _, s)| *s).collect();
            let compute_ms = self.exec(k, &mut live, &staged, true, true)?;
            let wall = t.elapsed().as_secs_f64() * 1e3;
            Ok((live.into_iter().map(|l| RunOut { outputs: l.outputs, gpu_ms: l.gpu_ms, compute_ms, wall_ms: wall, dispatch_ms: l.dispatch_ms }).collect(), compute_ms))
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
        // A pooled_recall problem's expected values are its accept sets and
        // today's counts (case inputs), not the reference kernel's outputs.
        let pooled = problem.pooled_output().is_some();
        for su in &problem.suites {
            let out = problem.output_of(su)?.clone();
            for c in &su.cases {
                let env = problem.env(c, limit)?;
                let mut data = problem.case_data(c, limit)?;
                let plan = problem.plan(c, limit, None)?;
                let (staged, _) = exec.stage(&plan, &data)?;
                if matches!(su.spec.role, Role::Constraint | Role::Accuracy) && !pooled {
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
        let round_budget_ms = problem.spec.budget.round_ms;
        // A pooled problem has no paired timing shapes: its speed anchor is
        // today's index (`Harness::evaluate_pooled`).
        let (shape_ms, bands, shipped_ms) = if pooled {
            (Vec::new(), [Vec::new(), Vec::new(), Vec::new()], [0.0; 3])
        } else {
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
            let worst = shipped_ms.iter().copied().fold(0.0, f64::max);
            if slow_limit_ms(round_budget_ms, &shipped_ms) < worst {
                return Err(format!("the reference kernel takes {shipped_ms:?} ms at the timing shapes: a timing round of {round_budget_ms} ms cannot hold it and a kernel at its speed"));
            }
            (shape_ms, bands, shipped_ms)
        };
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
        if !pooled {
            h.time_shipped()?;
        }
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
        if self.timing.is_empty() {
            return Err(format!("our software has a bug: time_shipped on {}, a pooled_recall problem: it has no paired timing shapes (its speed anchor is today's index, inside Fitness::inputs)", self.problem.spec.name));
        }
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
        if self.timing.is_empty() {
            return Err(format!("our software has a bug: retime on {}, a pooled_recall problem: its timing samples are inside Fitness::inputs (evaluate again)", self.problem.spec.name));
        }
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
        if self.problem.pooled_output().is_some() {
            return self.evaluate_pooled(source);
        }
        let t = Instant::now();
        let mut meter = Meter::default();
        let mut split = Split::default();
        let infeasible = |passed: usize, distance: f64, closeness: [f64; closeness::N], detail: String, t: Instant, meter: &Meter, split: &Split| Fitness { feasible: false, passed, distance, closeness, closeness_hff: closeness::angle(&closeness), columns: [1.0; 6], speedups: [0.0; 3], hff: f64::INFINITY, inputs: Vec::new(), detail, seconds: t.elapsed().as_secs_f64(), predicted_ms: [0.0; 3], rate_per_g: meter.rate_per_g(), floor_ms: meter.floor_ms, measured_ms: [0.0; 3], split: Split { total: t.elapsed().as_secs_f64() * 1e3, ..*split } };
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
        Fitness { feasible: true, passed, distance: 0.0, closeness: [0.0; closeness::N], closeness_hff: 0.0, columns, speedups, hff: truenorth(&columns), inputs: Vec::new(), detail: String::new(), seconds: t.elapsed().as_secs_f64(), predicted_ms, rate_per_g: meter.rate_per_g(), floor_ms: meter.floor_ms, measured_ms, split: Split { total: t.elapsed().as_secs_f64() * 1e3, ..split } }
    }

    /// THE FITNESS OF A POOLED_RECALL PROBLEM (v24, fuller
    /// `docs/PLAN_v24_ingest_evolution.md`): compile (every entry point),
    /// then per constraint case one run with its reads, judged per window on
    /// the executor (`Executor::judge_pooled`: the device's exact re-rank;
    /// x_M, x_P, x_B) and its per-stage samples timing run 1, then
    /// `timing_rounds - 1` fresh runs a case without reads; every timing
    /// run's per-stage samples (`samples_by_label`, `pooled_timing`) one input
    /// each against today's ms: `timing_rounds` runs a case in all, the
    /// inputs as when the read run was not timed. Every run is a
    /// whole fresh run: its index arenas created and zeroed at its start, so
    /// no sample reuses a built index. θ is the hff library's TrueNorth over
    /// the inputs (`score::truenorth_angle`); nothing is averaged. A kernel
    /// that does not compile or run is infeasible. The six columns are not
    /// measured (1) and the speed-ups 0.
    pub fn evaluate_pooled(&self, source: &str) -> Fitness {
        let t = Instant::now();
        let mut split = Split::default();
        let infeasible = |passed: usize, detail: String, split: &Split| Fitness { feasible: false, passed, distance: f64::INFINITY, closeness: closeness::WORST, closeness_hff: closeness::angle(&closeness::WORST), columns: [1.0; 6], speedups: [0.0; 3], hff: f64::INFINITY, inputs: Vec::new(), detail, seconds: t.elapsed().as_secs_f64(), predicted_ms: [0.0; 3], rate_per_g: 0.0, floor_ms: 0.0, measured_ms: [0.0; 3], split: Split { total: t.elapsed().as_secs_f64() * 1e3, ..*split } };
        let limit = self.exec.binding_limit();
        let compiled = self.exec.compile(&self.problem, source);
        split.compile = t.elapsed().as_secs_f64() * 1e3;
        let (k, info) = match compiled {
            Ok(x) => x,
            Err(e) => return infeasible(0, format!("compile: {}", e.lines().find(|l| l.contains("error") || l.contains("header")).unwrap_or(e.lines().next().unwrap_or(""))), &split),
        };
        let Some(out) = self.problem.pooled_output().cloned() else { return infeasible(0, "our software has a bug: evaluate_pooled on a problem whose constraint output is not pooled_recall".into(), &split) };
        let oi = self.problem.spec.output.iter().position(|o| o.name == out.name).expect("the output is the problem's");
        let mut inputs = Vec::new();
        let mut details = Vec::new();
        let mut first_samples = Vec::new();
        let t_cases = Instant::now();
        let mut passed = 0;
        for s in &self.constraint {
            let plan = match self.problem.plan(&s.case, limit, Some(&info)) {
                Ok(p) => p,
                Err(e) => return infeasible(passed, format!("run: {e}"), &split),
            };
            let r = match self.exec.run(&k, &plan, &s.data, &s.staged, true) {
                Ok(r) => r,
                Err(e) => return infeasible(passed, format!("run: {e} at {}", s.case.label), &split),
            };
            let w = match r.outputs.get(oi).ok_or_else(|| "the run read no output".to_string()).and_then(|b| self.exec.judge_pooled(&self.problem, &out, &s.env, &s.data, b)) {
                Ok(w) => w,
                Err(e) => return infeasible(passed, format!("our software has a bug: {e}"), &split),
            };
            // THE READ RUN IS TIMING RUN 1: its dispatches' timestamps are
            // taken in its compute submissions, before and apart from its
            // read-back, so it is a fresh run's samples as a timing run's are.
            let samples = match samples_by_label(&plan, &r).and_then(|m| pooled_timing(&self.problem, &out, &s.env, &s.data, &m)) {
                Ok(x) => x,
                Err(e) => return infeasible(passed, format!("our software has a bug: {e}"), &split),
            };
            inputs.extend(window_inputs(&w));
            details.push(window_detail(&s.case.label, &w));
            first_samples.push(samples);
            passed += 1;
        }
        split.cases = t_cases.elapsed().as_secs_f64() * 1e3;
        // Timing run 1's inputs, in the order the timing runs give theirs.
        for samples in &first_samples {
            details.push(samples.iter().map(|(l, ms, today)| format!("{l} {ms:.2} ms (today {today:.1})")).collect::<Vec<_>>().join(", "));
            inputs.extend(samples.iter().map(|(_, ms, today)| time_input(*today, *ms)));
        }
        let t_timing = Instant::now();
        for rep in 1..self.timing_rounds.max(1) {
            for s in &self.constraint {
                let plan = match self.problem.plan(&s.case, limit, Some(&info)) {
                    Ok(p) => p,
                    Err(e) => return infeasible(passed, format!("run: {e}"), &split),
                };
                let r = match self.exec.run(&k, &plan, &s.data, &s.staged, false) {
                    Ok(r) => r,
                    Err(e) => return infeasible(passed, format!("run: {e} at {} (timing run {})", s.case.label, rep + 1), &split),
                };
                let samples = match samples_by_label(&plan, &r).and_then(|m| pooled_timing(&self.problem, &out, &s.env, &s.data, &m)) {
                    Ok(x) => x,
                    Err(e) => return infeasible(passed, format!("our software has a bug: {e}"), &split),
                };
                inputs.extend(samples.iter().map(|(_, ms, today)| time_input(*today, *ms)));
            }
        }
        split.timing_cand = t_timing.elapsed().as_secs_f64() * 1e3;
        if let Some(bad) = inputs.iter().find(|x| !(0.0..=1.0).contains(*x)) {
            return infeasible(passed, format!("our software has a bug: an HFF input {bad} outside [0, 1]"), &split);
        }
        let hff = crate::score::truenorth_angle(&inputs);
        Fitness { feasible: true, passed, distance: 0.0, closeness: [0.0; closeness::N], closeness_hff: 0.0, columns: [1.0; 6], speedups: [0.0; 3], hff, inputs, detail: details.join("; "), seconds: t.elapsed().as_secs_f64(), predicted_ms: [0.0; 3], rate_per_g: 0.0, floor_ms: 0.0, measured_ms: [0.0; 3], split: Split { total: t.elapsed().as_secs_f64() * 1e3, ..split } }
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

// ------------------------------------------------------------------
// The pooled-recall judge (v24)
// ------------------------------------------------------------------

/// A candidate list's pad.
pub const PAD: u32 = 0xFFFF_FFFF;

/// ONE WINDOW (a stage x an oracle half) of a pooled_recall output: its
/// counts, each a sum over the window's queries, beside today's index's
/// on the same queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PooledWindow {
    /// 1-based.
    pub stage: usize,
    pub half: usize,
    pub queries: u64,
    /// Accept-set rows in the re-ranked top, each query's capped at `top`.
    pub found: u64,
    /// M = top * queries - found.
    pub misses: u64,
    /// P: the pools' entries (deduplicated, pads dropped, broken included).
    pub pooled: u64,
    /// B: pool entries that are not arrived stream rows.
    pub broken: u64,
    pub misses_today: u64,
    pub pooled_today: u64,
}

/// An input tensor of the case as f64 values (by its declared element).
fn input_f64(problem: &Problem, data: &CaseData, name: &str) -> Result<Vec<f64>, String> {
    let spec = problem.spec.input.iter().find(|i| i.name == name).ok_or_else(|| format!("no input {name}"))?;
    let b = data.inputs.get(name).ok_or_else(|| format!("the case holds no input {name}"))?;
    Ok(spec.element.decode(b))
}

/// An input tensor of non-negative integers as usize.
fn input_usize(problem: &Problem, data: &CaseData, name: &str) -> Result<Vec<usize>, String> {
    input_f64(problem, data, name)?.into_iter().map(|v| integral(v, name).map(|x| x as usize)).collect()
}

/// THE POOLED JUDGE'S TABLES for the device (`pooled_judge.wgsl`): the
/// output's layout, the CSR records (values as f32 bits), each query's record
/// row, each stage's arrived rows, the accept sets of every stage
/// concatenated (offsets absolute), the halves and today's counts per stage
/// and query (for the windows' sums).
#[derive(Debug)]
pub struct JudgeInputs {
    pub stages: usize,
    pub seeds: usize,
    pub queries: usize,
    pub c_max: usize,
    pub top: usize,
    pub rec_offsets: Vec<u32>,
    pub rec_indices: Vec<u32>,
    pub rec_values: Vec<u32>,
    pub qrow: Vec<u32>,
    pub arrived: Vec<u32>,
    pub accept_offsets: Vec<u32>,
    pub accept_rows: Vec<u32>,
    pub half: Vec<usize>,
    pub today_misses: Vec<Vec<usize>>,
    pub today_pooled: Vec<Vec<usize>>,
}

/// [`JudgeInputs`] of a case for an output of `bytes` bytes, every check
/// [`pooled_windows`] makes.
pub fn judge_inputs(problem: &Problem, out: &OutputSpec, env: &Env, data: &CaseData, bytes: usize) -> Result<JudgeInputs, String> {
    let ps = out.pooled.as_ref().ok_or_else(|| format!("output {} has no [output.pooled]", out.name))?;
    let int = |e: &str, env: &Env| eval_int(e, env).map(|v| v as usize);
    let (seeds, stages, queries, top) = (int(&ps.seeds, env)?, int(&ps.stages, env)?, int(&ps.queries, env)?, int(&ps.top, env)?);
    let c_max = int(&out.shape[1], env)?;
    if !bytes.is_multiple_of(4) || bytes / 4 != stages * seeds * queries * c_max {
        return Err(format!("{}: {bytes} bytes, the layout is {stages} stages x {seeds} seeds x {queries} queries x {c_max} u32", out.name));
    }
    if seeds * c_max > 256 || top == 0 || top > 16 {
        return Err(format!("the pooled judge holds pools of at most 256 and a top of 1..=16: {seeds} seeds x {c_max}, top {top}"));
    }
    let u32s = |v: Vec<usize>, what: &str| v.into_iter().map(|x| u32::try_from(x).map_err(|_| format!("{what}: {x} past u32"))).collect::<Result<Vec<u32>, String>>();
    let (offsets, indices) = (input_usize(problem, data, &ps.rec_offsets)?, input_usize(problem, data, &ps.rec_indices)?);
    let values = input_f64(problem, data, &ps.rec_values)?;
    let records = offsets.len().saturating_sub(1);
    if offsets.last().copied() != Some(indices.len()) || indices.len() != values.len() || offsets.windows(2).any(|w| w[0] > w[1]) {
        return Err(format!("the CSR records: {} offsets, {} indices, {} values, not a CSR", offsets.len(), indices.len(), values.len()));
    }
    // The values as the device reads them: f32 bits, each exactly the f64 the host reads.
    let rec_values: Vec<u32> = values.iter().map(|&v| (v as f32).to_bits()).collect();
    if values.iter().zip(&rec_values).any(|(&v, &b)| f64::from(f32::from_bits(b)) != v) {
        return Err(format!("{}: a value that is not an f32: the pooled judge reads f32 records", ps.rec_values));
    }
    let half = input_usize(problem, data, &ps.half)?;
    if half.len() < queries || half[..queries].iter().any(|&h| h > 1) {
        return Err(format!("{}: {} values for {queries} queries, each a half (0 or 1)", ps.half, half.len()));
    }
    let mut qrow = Vec::with_capacity(queries);
    let mut e = env.clone();
    for q in 0..queries {
        e.insert("q".into(), q as f64);
        let r = int(&ps.query_row, &e)?;
        if r >= records {
            return Err(format!("query {q}: record row {r} of {records}"));
        }
        qrow.push(r as u32);
    }
    let (mut arrived, mut accept_offsets, mut accept_rows, mut today_misses, mut today_pooled) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for stage in 1..=stages {
        let mut e = env.clone();
        e.insert("stage".into(), stage as f64);
        let a = int(&ps.arrived, &e)?;
        if a > records {
            return Err(format!("stage {stage}: {a} arrived rows, the CSR holds {records} records"));
        }
        arrived.push(a as u32);
        let (ao, ar) = (input_usize(problem, data, &template(&ps.accept_offsets, &e)?)?, input_usize(problem, data, &template(&ps.accept_rows, &e)?)?);
        let (tm, tp) = (input_usize(problem, data, &template(&ps.today_misses, &e)?)?, input_usize(problem, data, &template(&ps.today_pooled, &e)?)?);
        if ao.len() < queries + 1 || ao[queries] > ar.len() || tm.len() < queries || tp.len() < queries {
            return Err(format!("stage {stage}: accept offsets {}, rows {}, today's lists {} and {}, for {queries} queries", ao.len(), ar.len(), tm.len(), tp.len()));
        }
        let base = accept_rows.len();
        accept_offsets.extend(u32s(ao[..=queries].iter().map(|o| o + base).collect(), "an accept offset")?);
        accept_rows.extend(u32s(ar, "an accept row")?);
        today_misses.push(tm[..queries].to_vec());
        today_pooled.push(tp[..queries].to_vec());
    }
    Ok(JudgeInputs { stages, seeds, queries, c_max, top, rec_offsets: u32s(offsets, "a record offset")?, rec_indices: u32s(indices, "a record index")?, rec_values, qrow, arrived, accept_offsets, accept_rows, half, today_misses, today_pooled })
}

/// The windows of the device's judged words (found, pooled, broken per
/// stage and query), summed per stage and half as [`pooled_windows`] sums.
pub fn windows_of_judged(inp: &JudgeInputs, words: &[u32]) -> Result<Vec<PooledWindow>, String> {
    if words.len() != 3 * inp.stages * inp.queries {
        return Err(format!("the pooled judge wrote {} words for {} stages x {} queries", words.len(), inp.stages, inp.queries));
    }
    let mut windows = Vec::with_capacity(2 * inp.stages);
    for stage in 1..=inp.stages {
        let mut w = [PooledWindow { stage, half: 0, ..Default::default() }, PooledWindow { stage, half: 1, ..Default::default() }];
        for q in 0..inp.queries {
            let at = 3 * ((stage - 1) * inp.queries + q);
            let x = &mut w[inp.half[q]];
            x.queries += 1;
            x.found += u64::from(words[at]);
            x.pooled += u64::from(words[at + 1]);
            x.broken += u64::from(words[at + 2]);
            x.misses_today += inp.today_misses[stage - 1][q] as u64;
            x.pooled_today += inp.today_pooled[stage - 1][q] as u64;
        }
        for x in &mut w {
            x.misses = inp.top as u64 * x.queries - x.found;
        }
        windows.extend(w);
    }
    Ok(windows)
}

/// THE POOLED-RECALL COUNTS of one run's output (`PooledSpec`): per stage
/// and half, in stage order, half 0 first. Per query: the seeds' lists
/// pooled (pads dropped, deduplicated); entries at or past the stage's
/// arrived rows counted broken (B) and not ranked; the rest scored by their
/// exact cosine with the query's record, the sparse dot in f64 of the CSR
/// records (unit vectors), ranked by score then row id; found = the top
/// `top`'s rows in the query's accept set, at most `top`. M and P, and
/// today's, are sums over the window's queries.
pub fn pooled_windows(problem: &Problem, out: &OutputSpec, env: &Env, data: &CaseData, bytes: &[u8]) -> Result<Vec<PooledWindow>, String> {
    let ps = out.pooled.as_ref().ok_or_else(|| format!("output {} has no [output.pooled]", out.name))?;
    let int = |e: &str, env: &Env| eval_int(e, env).map(|v| v as usize);
    let (seeds, stages, queries, top) = (int(&ps.seeds, env)?, int(&ps.stages, env)?, int(&ps.queries, env)?, int(&ps.top, env)?);
    let c_max = int(&out.shape[1], env)?;
    if !bytes.len().is_multiple_of(4) || bytes.len() / 4 != stages * seeds * queries * c_max {
        return Err(format!("{}: {} bytes, the layout is {stages} stages x {seeds} seeds x {queries} queries x {c_max} u32", out.name, bytes.len()));
    }
    let cands: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes"))).collect();
    let (offsets, indices, values) = (input_usize(problem, data, &ps.rec_offsets)?, input_usize(problem, data, &ps.rec_indices)?, input_f64(problem, data, &ps.rec_values)?);
    let records = offsets.len().saturating_sub(1);
    if offsets.last().copied() != Some(indices.len()) || indices.len() != values.len() || offsets.windows(2).any(|w| w[0] > w[1]) {
        return Err(format!("the CSR records: {} offsets, {} indices, {} values, not a CSR", offsets.len(), indices.len(), values.len()));
    }
    let record = |r: usize| (&indices[offsets[r]..offsets[r + 1]], &values[offsets[r]..offsets[r + 1]]);
    let half = input_usize(problem, data, &ps.half)?;
    if half.len() < queries {
        return Err(format!("{}: {} values for {queries} queries", ps.half, half.len()));
    }
    let mut windows = Vec::with_capacity(2 * stages);
    for stage in 1..=stages {
        let mut e = env.clone();
        e.insert("stage".into(), stage as f64);
        let arrived = int(&ps.arrived, &e)?;
        if arrived > records {
            return Err(format!("stage {stage}: {arrived} arrived rows, the CSR holds {records} records"));
        }
        let (ao, ar) = (input_usize(problem, data, &template(&ps.accept_offsets, &e)?)?, input_usize(problem, data, &template(&ps.accept_rows, &e)?)?);
        let (tm, tp) = (input_usize(problem, data, &template(&ps.today_misses, &e)?)?, input_usize(problem, data, &template(&ps.today_pooled, &e)?)?);
        if ao.len() < queries + 1 || ao[queries] > ar.len() || tm.len() < queries || tp.len() < queries {
            return Err(format!("stage {stage}: accept offsets {}, rows {}, today's lists {} and {}, for {queries} queries", ao.len(), ar.len(), tm.len(), tp.len()));
        }
        let mut w = [PooledWindow { stage, half: 0, ..Default::default() }, PooledWindow { stage, half: 1, ..Default::default() }];
        for q in 0..queries {
            let h = half[q];
            if h > 1 {
                return Err(format!("{}[{q}] is {h}, not a half (0 or 1)", ps.half));
            }
            e.insert("q".into(), q as f64);
            let qrow = int(&ps.query_row, &e)?;
            if qrow >= records {
                return Err(format!("query {q}: record row {qrow} of {records}"));
            }
            let (qi, qv) = record(qrow);
            let mut query: Vec<(usize, f64)> = qi.iter().copied().zip(qv.iter().copied()).collect();
            query.sort_by_key(|x| x.0);
            let mut pool: Vec<u32> = (0..seeds).flat_map(|s| cands[((stage - 1) * seeds + s) * queries * c_max + q * c_max..][..c_max].iter().copied()).filter(|&c| c != PAD).collect();
            pool.sort_unstable();
            pool.dedup();
            let broken = pool.iter().filter(|&&c| c as usize >= arrived).count();
            let mut scored: Vec<(f64, u32)> = pool
                .iter()
                .filter(|&&c| (c as usize) < arrived)
                .map(|&c| {
                    let (ci, cv) = record(c as usize);
                    let dot = ci.iter().zip(cv).map(|(i, v)| query.binary_search_by_key(i, |x| x.0).map_or(0.0, |k| query[k].1 * v)).sum::<f64>();
                    (dot, c)
                })
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let accept = &ar[ao[q]..ao[q + 1]];
            let found = scored.iter().take(top).filter(|(_, c)| accept.binary_search(&(*c as usize)).is_ok()).count().min(top);
            let x = &mut w[h];
            x.queries += 1;
            x.found += found as u64;
            x.pooled += pool.len() as u64;
            x.broken += broken as u64;
            x.misses_today += tm[q] as u64;
            x.pooled_today += tp[q] as u64;
        }
        for x in &mut w {
            x.misses = top as u64 * x.queries - x.found;
        }
        windows.extend(w);
    }
    Ok(windows)
}

/// A count's HFF input against today's on the same window:
/// `min(1, v / (2 today))`, today's count 0.5. A window where today's count
/// is 0 uses a floor of 1 (`v / 2`: none 0, one 0.5, two or more 1), so a
/// window today's index does perfectly still ranks a kernel's misses.
pub fn count_input(v: u64, today: u64) -> f64 {
    (v as f64 / (2.0 * today.max(1) as f64)).min(1.0)
}

/// The broken entries' HFF input: `min(1, B / 10)`.
pub fn broken_input(b: u64) -> f64 {
    (b as f64 / 10.0).min(1.0)
}

/// A timing sample's HFF input against today's ms:
/// `clamp(0.5 - log10(today / kernel) / 4, 0, 1)`: today's speed 0.5, ten
/// times faster 0.25, ten thousand times faster (or a 0 ms sample) 0, ten
/// thousand times slower 1.
pub fn time_input(today_ms: f64, kernel_ms: f64) -> f64 {
    (0.5 - (today_ms / kernel_ms).log10() / 4.0).clamp(0.0, 1.0)
}

/// The windows' HFF inputs, three a window in window order: x_M, x_P, x_B.
pub fn window_inputs(windows: &[PooledWindow]) -> Vec<f64> {
    windows.iter().flat_map(|w| [count_input(w.misses, w.misses_today), count_input(w.pooled, w.pooled_today), broken_input(w.broken)]).collect()
}

/// The windows in one line: per stage, each half's M, P and B with today's.
fn window_detail(label: &str, windows: &[PooledWindow]) -> String {
    let stages: Vec<String> = windows
        .chunks(2)
        .map(|h| {
            let f = |g: &dyn Fn(&PooledWindow) -> u64| h.iter().map(|w| g(w).to_string()).collect::<Vec<_>>().join("|");
            format!("stage {}: M {} (today {}) P {} (today {}) B {}", h[0].stage, f(&|w| w.misses), f(&|w| w.misses_today), f(&|w| w.pooled), f(&|w| w.pooled_today), f(&|w| w.broken))
        })
        .collect();
    format!("{label}: {}", stages.join(", "))
}

/// A run's timing samples by label: each dispatch's own GPU ms
/// (`RunOut::dispatch_ms`, the plan's dispatch order) summed under its
/// plan `sample` label; unlabelled dispatches are in no sample.
pub fn samples_by_label(plan: &Plan, run: &RunOut) -> Result<BTreeMap<String, f64>, String> {
    let labels: Vec<&Option<String>> = plan.run.iter().filter_map(|c| if let Cmd::Dispatch { sample, .. } = c { Some(sample) } else { None }).collect();
    if labels.len() != run.dispatch_ms.len() {
        return Err(format!("the run timed {} dispatches, the plan has {}", run.dispatch_ms.len(), labels.len()));
    }
    let mut out = BTreeMap::new();
    for (l, ms) in labels.iter().zip(&run.dispatch_ms) {
        if let Some(l) = l {
            *out.entry(l.clone()).or_insert(0.0) += ms;
        }
    }
    Ok(out)
}

/// THE TIMING SAMPLES OF ONE RUN against today's: per stage, per
/// `[[output.pooled.timing]]` in order, (label, the kernel's GPU ms, today's
/// ms); a `cumulative` sample's ms is its label's through the stage (stages
/// 1..k summed: a measured total, not an average). Today's from `today` (an expression) or `today_input` (summed over
/// the queries judged); a label no dispatch carries, or today's ms not a
/// positive finite value, is an error.
pub fn pooled_timing(problem: &Problem, out: &OutputSpec, env: &Env, data: &CaseData, samples: &BTreeMap<String, f64>) -> Result<Vec<(String, f64, f64)>, String> {
    let ps = out.pooled.as_ref().ok_or_else(|| format!("output {} has no [output.pooled]", out.name))?;
    let stages = eval_int(&ps.stages, env)? as usize;
    let queries = eval_int(&ps.queries, env)? as usize;
    let mut v = Vec::new();
    for stage in 1..=stages {
        let mut e = env.clone();
        e.insert("stage".into(), stage as f64);
        for t in &ps.timing {
            let label = template(&t.sample, &e)?;
            let mut ms = 0.0;
            for k in if t.cumulative { 1..=stage } else { stage..=stage } {
                let mut ek = env.clone();
                ek.insert("stage".into(), k as f64);
                let l = template(&t.sample, &ek)?;
                ms += *samples.get(&l).ok_or_else(|| format!("no dispatch is labelled {l}"))?;
            }
            let today = match (&t.today, &t.today_input) {
                (Some(x), None) => eval(&template(x, &e)?, &e)?,
                (None, Some(i)) => {
                    let name = template(i, &e)?;
                    let per = input_f64(problem, data, &name)?;
                    per.get(..queries).ok_or_else(|| format!("{name}: {} values for {queries} queries", per.len()))?.iter().sum()
                }
                _ => return Err("a pooled timing sample names today's ms by exactly one of today and today_input".into()),
            };
            if !(today.is_finite() && today > 0.0 && ms.is_finite() && ms >= 0.0) {
                return Err(format!("{label}: today's {today} ms, the kernel's {ms} ms"));
            }
            v.push((label, ms, today));
        }
    }
    Ok(v)
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
                Cmd::Dispatch { bind, workgroups, .. } => {
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
        KernelInfo { workgroup: [g.x, g.y, 1], bx: g.bx, by: g.by, rows_x: g.rows_x, modes: Vec::new(), entry_workgroups: vec![[g.x, g.y, 1]], passes: 1 }
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
        let dirs = ["popA_g001", "popA2_g001b", "library_texts", "popB_g080", "reserve2", "popC_rand20", "popR_rules"];
        let (mut kernels, mut accepted, mut plans) = (0usize, 0usize, 0usize);
        let mut differ = Vec::new();
        for dir in dirs {
            for (name, text) in crate::wgsl::fixture_texts::set(dir) {
                let f = std::path::PathBuf::from(format!("{dir}/{name}"));
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
                    run.push(Cmd::Dispatch { bind: vec![(0, phis[band]), (1, acc), (2, id)], workgroups: [workgroups.0, workgroups.1, 1], entry: 0, sample: None });
                }
                DriverStep::Sync => run.push(Cmd::Sync),
                DriverStep::Read { r_off, rows } => run.push(Cmd::Read { src: acc, src_offset: 0, output: 0, dst_offset: (r_off * d * 4) as u64, bytes: (rows * d * 4) as u64 }),
            }
        }
        Plan { buffers, setup, run, outputs: vec![(d * d * 4) as u64], batch_safe: true, gpu_wait_ms: gram::WAIT_GAVE_UP_MS, machines: Vec::new(), setup_dispatch: Vec::new() }
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
        toy(run).plan(&c, LIMIT, Some(&KernelInfo { workgroup: [64, 1, 1], bx: 1, by: 1, rows_x: false, modes: Vec::new(), entry_workgroups: vec![[64, 1, 1]], passes: 1 }))
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

    // ------------------------------------------------------------------
    // v24: multi-entry dispatch, passes, run-lifetime buffers, stage
    // reads, per-dispatch samples, the pooled_recall judge
    // ------------------------------------------------------------------

    /// A two-entry kernel: `ingest` adds 1 to every arena word (a pass
    /// each), `search` writes `arena * 100 + stage` to the even slots of the
    /// stage's output; odd slots stay pads.
    const MULTI: &str = "// passes 3
struct Cfg { k: u32, p: u32, n: u32, };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> arena: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> cfg: Cfg;
@compute @workgroup_size(4)
fn ingest(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x < cfg.n) { arena[g.x] = arena[g.x] + 1u; }
}
@compute @workgroup_size(2)
fn search(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x < cfg.n && g.x % 2u == 0u) { out[g.x] = arena[g.x] * 100u + cfg.k; }
}
";

    /// The host model of `MULTI`, by entry.
    struct MultiModel;

    impl HostKernel for MultiModel {
        fn invoke(&self, ids: &HostIds, mem: &mut HostMemory) -> u64 {
            let (k, n, i) = (mem.load_u32(3, 0), mem.load_u32(3, 2), ids.gid[0]);
            if i >= n {
                return 0;
            }
            match ids.entry {
                0 => mem.store_u32(1, i, mem.load_u32(1, i) + 1),
                _ if i % 2 == 0 => mem.store_u32(2, i, mem.load_u32(1, i) * 100 + k),
                _ => {}
            }
            1
        }
    }

    fn multi_problem() -> Problem {
        let text = "name = \"multi\"\nwork = \"n\"\nalias = \"distinct\"\n\
             [kernel]\nentry = \"ingest\"\nentries = [\"search\"]\npasses_header = true\nreference = \"none\"\n\
             [[binding]]\nbinding = 0\nname = \"a\"\nkind = \"storage_read\"\nelement = \"f32\"\n\
             [[binding]]\nbinding = 1\nname = \"arena\"\nkind = \"storage_read_write\"\nelement = \"u32\"\n\
             [[binding]]\nbinding = 2\nname = \"out\"\nkind = \"storage_read_write\"\nelement = \"u32\"\n\
             [[binding]]\nbinding = 3\nname = \"cfg\"\nkind = \"uniform\"\nfields = [\"u32\", \"u32\", \"u32\"]\n\
             [[input]]\nname = \"x\"\nelement = \"f32\"\nshape = [\"n\"]\ngenerate = { dist = \"constant\", value = \"1\" }\n\
             [[output]]\nname = \"y\"\nelement = \"u32\"\nshape = [\"stages\", \"n\"]\n\
             [[setup]]\nbuffer = \"res\"\nelement = \"f32\"\nelements = \"n\"\nfrom = \"x\"\n\
             [[run]]\nbuffer = \"arena\"\nelement = \"u32\"\nelements = \"n\"\nfill = \"zero\"\n\
             [[run]]\nfor = \"k0\"\ncount = \"stages\"\n\
             [[run.body]]\nbuffer = \"o{k0}\"\nelement = \"u32\"\nelements = \"n\"\nfill = \"pad\"\n\
             [[run.body]]\nfor = \"p\"\ncount = \"passes\"\n\
             [[run.body.body]]\nuniform = \"c{k0}_{p}\"\nof = \"cfg\"\nwords = [\"k0\", \"p\", \"n\"]\n\
             [[run.body.body]]\ndispatch = { entry = \"ingest\", a = \"res\", arena = \"arena\", out = \"o{k0}\", cfg = \"c{k0}_{p}\" }\nworkgroups = [\"ceil_div(n, wg_x_ingest)\", \"1\", \"1\"]\nsample = \"ingest{k0}\"\n\
             [[run.body]]\nuniform = \"cs{k0}\"\nof = \"cfg\"\nwords = [\"k0\", \"passes\", \"n\"]\n\
             [[run.body]]\ndispatch = { entry = \"search\", a = \"res\", arena = \"arena\", out = \"o{k0}\", cfg = \"cs{k0}\" }\nworkgroups = [\"ceil_div(n, wg_x_search)\", \"1\", \"1\"]\nsample = \"search{k0}\"\n\
             [[run.body]]\nread = \"o{k0}\"\ninto = \"y\"\nelements = \"n\"\noffset = \"k0 * n\"\n\
             [[run]]\nsync = true\n\
             [[suite]]\nname = \"c\"\nrole = \"constraint\"\ncases = \"none\"\n\
             [budget]\nbase_ms = 1.0\nround_ms = 1.0\n";
        let spec: ProblemSpec = toml::from_str(text).unwrap_or_else(|e| panic!("{e}"));
        Problem { dir: PathBuf::new(), spec, suites: Vec::new(), reference: String::new() }
    }

    fn multi_case() -> Case {
        Case { name: "n6".into(), label: "n6".into(), params: [("n".to_string(), 6.0), ("stages".to_string(), 2.0)].into_iter().collect(), inputs: BTreeMap::new(), expected: BTreeMap::new() }
    }

    fn u32s(b: &[u8]) -> Vec<u32> {
        b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    /// MULTI-ENTRY DISPATCH, PASSES, RUN-LIFETIME BUFFERS, STAGE READS AND
    /// PER-DISPATCH SAMPLES, on the host executor (a model by entry) and
    /// the CPU executor (the kernel's own WGSL): each dispatch runs the
    /// entry it names with that entry's workgroup size; `// passes 3` gives
    /// three ingest dispatches a stage; the arena is made and zeroed at the
    /// run's start and persists across its stages (3 then 6), so a second
    /// run gives the same output (no state reused); each stage's read lands
    /// on its own stage row; unwritten slots are pads; every dispatch's ms
    /// is its own value, summed by label.
    #[test]
    fn a_multi_entry_kernel_runs_each_entry_with_passes_and_fresh_arenas() {
        let p = multi_problem();
        let c = multi_case();
        let data = p.case_data(&c, LIMIT).unwrap();
        let mut host = HostExecutor::new(LIMIT);
        host.register(MULTI, Box::new(MultiModel));
        let cpu = CpuExecutor::new(LIMIT);
        let (hk, info) = host.compile(&p, MULTI).unwrap();
        assert_eq!((info.passes, info.entry_workgroups.clone()), (3, vec![[4, 1, 1], [2, 1, 1]]));
        let plan = p.plan(&c, LIMIT, Some(&info)).unwrap();
        let dispatches: Vec<(usize, [u32; 3], Option<String>)> = plan.run.iter().filter_map(|x| if let Cmd::Dispatch { entry, workgroups, sample, .. } = x { Some((*entry, *workgroups, sample.clone())) } else { None }).collect();
        let want: Vec<(usize, [u32; 3], Option<String>)> = (0..2).flat_map(|k| { let mut v = vec![(0, [2, 1, 1], Some(format!("ingest{k}"))); 3]; v.push((1, [3, 1, 1], Some(format!("search{k}")))); v }).collect();
        assert_eq!(dispatches, want);
        let expect: Vec<u32> = (0..2u32).flat_map(|k| (0..6u32).map(move |i| if i % 2 == 0 { 300 * (k + 1) + k } else { PAD })).collect();
        let (hs, _) = host.stage(&plan, &data).unwrap();
        let (ck, _) = cpu.compile(&p, MULTI).unwrap();
        let (cs, _) = cpu.stage(&plan, &data).unwrap();
        for run in 0..2 {
            let h = host.run(&hk, &plan, &data, &hs, true).unwrap();
            let g = cpu.run(&ck, &plan, &data, &cs, true).unwrap();
            assert_eq!(u32s(&h.outputs[0]), expect, "host run {run}");
            assert_eq!(u32s(&g.outputs[0]), expect, "cpu run {run}");
            for r in [&h, &g] {
                assert_eq!(r.dispatch_ms.len(), 8);
                let s = samples_by_label(&plan, r).unwrap();
                assert_eq!(s.keys().cloned().collect::<Vec<_>>(), ["ingest0", "ingest1", "search0", "search1"]);
                let ingest0: f64 = r.dispatch_ms[..3].iter().sum();
                assert_eq!(s["ingest0"].to_bits(), ingest0.to_bits());
                assert_eq!(s["search1"].to_bits(), r.dispatch_ms[7].to_bits());
            }
        }
        // The rules named: an entry [kernel] does not list, a kernel without
        // an entry the problem lists, passes headers out of range.
        let bad = multi_problem();
        let mut spec = bad.spec.clone();
        spec.run[1].body[3].dispatch.as_mut().unwrap().insert("entry".into(), "settle".into());
        let e = Problem { spec, ..bad }.plan(&c, LIMIT, Some(&info)).unwrap_err();
        assert!(e.contains("names entry settle"), "{e}");
        let e = kernel_info(&p.spec, &MULTI.replace("fn search", "fn find")).unwrap_err();
        assert!(e.contains("no compute entry point search"), "{e}");
        assert_eq!(passes_header("@compute fn f() {}"), Ok(1));
        assert_eq!(passes_header("// passes 16\n"), Ok(16));
        for h in ["// passes 0", "// passes 17", "// passes two", "// passes 2\n// passes 3"] {
            assert!(passes_header(h).is_err(), "{h}");
        }
        assert!(kernel_info(&p.spec, &MULTI.replace("// passes 3", "// passes 40")).unwrap_err().contains("outside 1..=16"));
        // STORAGE ATOMICS: an `array<atomic<u32>>` arena is a u32 binding, and
        // the CPU executor runs atomicAdd and atomicLoad to the same output.
        let atomic = MULTI
            .replace("var<storage, read_write> arena: array<u32>", "var<storage, read_write> arena: array<atomic<u32>>")
            .replace("arena[g.x] = arena[g.x] + 1u;", "atomicAdd(&arena[g.x], 1u);")
            .replace("out[g.x] = arena[g.x] * 100u + cfg.k;", "out[g.x] = atomicLoad(&arena[g.x]) * 100u + cfg.k;");
        let (ak, ai) = cpu.compile(&p, &atomic).unwrap();
        assert_eq!(ai, info);
        let g = cpu.run(&ak, &plan, &data, &cs, true).unwrap();
        assert_eq!(u32s(&g.outputs[0]), expect, "atomic arena on the cpu");
    }

    /// The hand-made pooled problem (see `pooled_judge_counts_each_window`):
    /// 6 stream rows in two stages of 3, 3 queries (records 6, 7, 8), 2
    /// seeds, 4 candidates a seed, top 2.
    fn pooled_problem_text(reference: &str) -> String {
        format!(
            "name = \"pooled\"\nwork = \"n_stream\"\nalias = \"distinct\"\n\
             [kernel]\nentry = \"ingest\"\nentries = [\"search\"]\nreference = {reference:?}\n\
             [[binding]]\nbinding = 0\nname = \"cands\"\nkind = \"storage_read_write\"\nelement = \"u32\"\n\
             [[binding]]\nbinding = 1\nname = \"cfg\"\nkind = \"uniform\"\nfields = [\"u32\", \"u32\"]\n\
             [[input]]\nname = \"rec_offsets\"\nelement = \"u32\"\nshape = [\"10\"]\n\
             [[input]]\nname = \"rec_indices\"\nelement = \"u32\"\nshape = [\"nnz\"]\n\
             [[input]]\nname = \"rec_values\"\nelement = \"f32\"\nshape = [\"nnz\"]\n\
             [[input]]\nname = \"half\"\nelement = \"u32\"\nshape = [\"3\"]\n\
             [[input]]\nname = \"ao1\"\nelement = \"u32\"\nshape = [\"4\"]\n\
             [[input]]\nname = \"ar1\"\nelement = \"u32\"\nshape = [\"7\"]\n\
             [[input]]\nname = \"ao2\"\nelement = \"u32\"\nshape = [\"4\"]\n\
             [[input]]\nname = \"ar2\"\nelement = \"u32\"\nshape = [\"7\"]\n\
             [[input]]\nname = \"tm1\"\nelement = \"u32\"\nshape = [\"3\"]\n\
             [[input]]\nname = \"tp1\"\nelement = \"u32\"\nshape = [\"3\"]\n\
             [[input]]\nname = \"tm2\"\nelement = \"u32\"\nshape = [\"3\"]\n\
             [[input]]\nname = \"tp2\"\nelement = \"u32\"\nshape = [\"3\"]\n\
             [[input]]\nname = \"ts1\"\nelement = \"f32\"\nshape = [\"3\"]\n\
             [[input]]\nname = \"ts2\"\nelement = \"f32\"\nshape = [\"3\"]\n\
             [[output]]\nname = \"cands\"\nelement = \"u32\"\nshape = [\"2 * 2 * 3\", \"4\"]\nkind = \"pooled_recall\"\n\
             [output.pooled]\nseeds = \"2\"\nstages = \"2\"\nqueries = \"3\"\ntop = \"2\"\narrived = \"stage * 3\"\nquery_row = \"6 + q\"\n\
             rec_offsets = \"rec_offsets\"\nrec_indices = \"rec_indices\"\nrec_values = \"rec_values\"\nhalf = \"half\"\n\
             accept_offsets = \"ao{{stage}}\"\naccept_rows = \"ar{{stage}}\"\ntoday_misses = \"tm{{stage}}\"\ntoday_pooled = \"tp{{stage}}\"\n\
             [[output.pooled.timing]]\nsample = \"ingest{{stage}}\"\ncumulative = true\ntoday = \"settle_{{stage}}\"\n\
             [[output.pooled.timing]]\nsample = \"search{{stage}}\"\ntoday_input = \"ts{{stage}}\"\n\
             [[run]]\nfor = \"k0\"\ncount = \"2\"\nlet = [\"stage = k0 + 1\"]\n\
             [[run.body]]\nfor = \"s\"\ncount = \"2\"\n\
             [[run.body.body]]\nbuffer = \"c{{k0}}_{{s}}\"\nelement = \"u32\"\nelements = \"12\"\nfill = \"pad\"\n\
             [[run.body.body]]\nuniform = \"u{{k0}}_{{s}}\"\nof = \"cfg\"\nwords = [\"stage\", \"s\"]\n\
             [[run.body.body]]\ndispatch = {{ entry = \"ingest\", cands = \"c{{k0}}_{{s}}\", cfg = \"u{{k0}}_{{s}}\" }}\nworkgroups = [\"1\", \"1\", \"1\"]\nsample = \"ingest{{stage}}\"\n\
             [[run.body.body]]\ndispatch = {{ entry = \"search\", cands = \"c{{k0}}_{{s}}\", cfg = \"u{{k0}}_{{s}}\" }}\nworkgroups = [\"1\", \"1\", \"1\"]\nsample = \"search{{stage}}\"\n\
             [[run.body.body]]\nread = \"c{{k0}}_{{s}}\"\ninto = \"cands\"\nelements = \"12\"\noffset = \"(k0 * 2 + s) * 12\"\n\
             [[run.body]]\nsync = true\n\
             [[suite]]\nname = \"stream\"\nrole = \"constraint\"\ncases = \"cases/stream.toml\"\nchunk = 1\n\
             [budget]\nbase_ms = 1000.0\nround_ms = 1000.0\ntiming_rounds = 2\n"
        )
    }

    /// The records: r0 {0: 1}, r1 {0: .8, 1: .6}, r2 {1: 1}, r3 {0: .8,
    /// 2: .6}, r4 {2: 1}, r5 {0: .6, 1: .8}; the queries r6 {0: 1}, r7
    /// {1: 1}, r8 {2: 1}; halves 0, 1, 0. Accept sets (top 2 with ties):
    /// stage 1 (rows 0..3) {0,1}, {1,2}, {0,1,2} (q2 scores 0 everywhere);
    /// stage 2 (rows 0..6) {0,1,3} (r1, r3 tie at .8), {2,5}, {3,4}.
    fn pooled_inputs_data() -> Vec<(&'static str, Vec<u8>)> {
        let u = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let f = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        vec![
            ("rec_offsets", u(&[0, 1, 3, 4, 6, 7, 9, 10, 11, 12])),
            ("rec_indices", u(&[0, 0, 1, 1, 0, 2, 2, 0, 1, 0, 1, 2])),
            ("rec_values", f(&[1.0, 0.8, 0.6, 1.0, 0.8, 0.6, 1.0, 0.6, 0.8, 1.0, 1.0, 1.0])),
            ("half", u(&[0, 1, 0])),
            ("ao1", u(&[0, 2, 4, 7])),
            ("ar1", u(&[0, 1, 1, 2, 0, 1, 2])),
            ("ao2", u(&[0, 3, 5, 7])),
            ("ar2", u(&[0, 1, 3, 2, 5, 3, 4])),
            ("tm1", u(&[0, 1, 0])),
            ("tp1", u(&[10, 20, 30])),
            ("tm2", u(&[0, 0, 0])),
            ("tp2", u(&[5, 5, 5])),
            ("ts1", f(&[1.0, 2.0, 3.0])),
            ("ts2", f(&[4.0, 4.0, 4.0])),
        ]
    }

    const NNZ: f64 = 12.0;

    /// The candidates, [stage][seed][query][4]: duplicates across seeds,
    /// pads, an oracle row (7), a future row (4 at stage 1), an id past every
    /// record (99), an all-tie pool, an empty pool.
    fn pooled_cands() -> Vec<u32> {
        let x = PAD;
        [
            [[1, 0, x, x], [2, x, x, x], [4, 99, 0, x]],
            [[0, 7, x, x], [x, x, x, x], [0, 1, 2, x]],
            [[5, 3, x, x], [5, 2, 1, 0], [x, x, x, x]],
            [[3, x, x, x], [6, 8, x, x], [x, x, x, x]],
        ]
        .iter()
        .flatten()
        .flatten()
        .copied()
        .collect()
    }

    fn pooled_staged(p: &Problem) -> (CaseData, Env) {
        let mut data = CaseData::default();
        for (n, b) in pooled_inputs_data() {
            data.inputs.insert(n.into(), b);
        }
        let mut env = Env::new();
        env.insert("nnz".into(), NNZ);
        for (k, v) in [("n_stream", 6.0), ("settle_1", 10.0), ("settle_2", 20.0)] {
            env.insert(k.into(), v);
        }
        assert!(p.spec.output[0].pooled.is_some());
        (data, env)
    }

    /// THE POOLED-RECALL JUDGE on the hand-made case, every count worked by
    /// hand. Stage 1: q0 pools {0,1,7} (0 twice), 7 an oracle row: P 3, B 1,
    /// top {0,1} found 2; q1 pools {2}: found 1; q2 pools {0,1,2,4,99}, 4 a
    /// row not yet arrived and 99 past every record: P 5, B 2, the three
    /// valid rows tie at 0, the top by id {0,1}, found 2. Stage 2: q0 pools
    /// {3,5}: top {3,5}, found 1 (5 is not in {0,1,3}); q1 pools
    /// {0,1,2,5,6,8}, 6 and 8 oracle rows: P 6, B 2, top {2,5} found 2; q2
    /// pools nothing: found 0. Windows (half 0 = q0, q2; half 1 = q1).
    #[test]
    fn pooled_judge_counts_each_window() {
        let spec: ProblemSpec = toml::from_str(&pooled_problem_text("none")).unwrap();
        let p = Problem { dir: PathBuf::new(), spec, suites: Vec::new(), reference: String::new() };
        let (data, env) = pooled_staged(&p);
        let out = p.spec.output[0].clone();
        let bytes: Vec<u8> = pooled_cands().iter().flat_map(|x| x.to_le_bytes()).collect();
        let w = pooled_windows(&p, &out, &env, &data, &bytes).unwrap();
        let row = |stage, half, queries, found, misses, pooled, broken, misses_today, pooled_today| PooledWindow { stage, half, queries, found, misses, pooled, broken, misses_today, pooled_today };
        assert_eq!(w, vec![row(1, 0, 2, 4, 0, 8, 3, 0, 40), row(1, 1, 1, 1, 1, 1, 0, 1, 20), row(2, 0, 2, 1, 3, 2, 0, 0, 10), row(2, 1, 1, 2, 0, 6, 2, 0, 5)]);
        // x_M, x_P, x_B a window: today's 0 misses floored to 1.
        assert_eq!(window_inputs(&w), vec![0.0, 0.1, 0.3, 0.5, 0.025, 0.0, 1.0, 0.1, 0.0, 0.0, 0.6, 0.2]);
        // A wrong layout is named.
        assert!(pooled_windows(&p, &out, &env, &data, &bytes[4..]).unwrap_err().contains("the layout is 2 stages x 2 seeds x 3 queries x 4 u32"));
    }

    /// THE INPUT MAPPING: today's count maps to 0.5 (a floor of 1 when
    /// today's is 0), twice today's or more to 1; B over 10; a timing sample
    /// at today's ms 0.5, ten times faster 0.25, ten times slower 0.75,
    /// 10,000 times faster or 0 ms 0, clamped.
    #[test]
    fn the_hff_inputs_map_todays_counts_to_one_half() {
        assert_eq!([count_input(5, 5), count_input(0, 0), count_input(1, 0), count_input(2, 0), count_input(20, 5), count_input(0, 7)], [0.5, 0.0, 0.5, 1.0, 1.0, 0.0]);
        assert_eq!([broken_input(0), broken_input(5), broken_input(11)], [0.0, 0.5, 1.0]);
        assert_eq!([time_input(100.0, 100.0), time_input(100.0, 10.0), time_input(100.0, 1000.0), time_input(100.0, 0.01), time_input(100.0, 0.0), time_input(1.0, 1e9)], [0.5, 0.25, 0.75, 0.0, 0.0, 1.0]);
    }

    /// The model behind the pooled harness test: `search` writes the
    /// hand-made lists of its stage and seed (cfg words 0 and 1), `ingest`
    /// nothing; ingest 5 operations, search 50 (the host's synthetic time).
    struct PooledModel;

    impl HostKernel for PooledModel {
        fn invoke(&self, ids: &HostIds, mem: &mut HostMemory) -> u64 {
            if ids.entry == 0 {
                return 5;
            }
            let (stage, seed) = (mem.load_u32(1, 0) as usize, mem.load_u32(1, 1) as usize);
            let c = pooled_cands();
            let at = ((stage - 1) * 2 + seed) * 12;
            for (i, v) in c[at..at + 12].iter().enumerate() {
                mem.store_u32(0, i as u32, *v);
            }
            50
        }
    }

    const POOLED_WGSL: &str = "struct Cfg { stage: u32, seed: u32, };
@group(0) @binding(0) var<storage, read_write> cands: array<u32>;
@group(0) @binding(1) var<uniform> cfg: Cfg;
@compute @workgroup_size(1)
fn ingest() {}
@compute @workgroup_size(1)
fn search() { cands[0] = cfg.stage; }
";

    /// THE POOLED FITNESS through `Harness::new` and `evaluate` on the host
    /// executor: a problem directory with the hand-made files; the inputs
    /// are the twelve window inputs above, then per timing run (two, each a
    /// fresh run) and stage the ingest through the stage (two seeds' ingest
    /// a stage, each 0.01 ms and 5 ns; stage 2's sample stages 1 and 2
    /// summed) against `settle_<k>` and the search sample against today's
    /// per-query ms summed; θ is the hff library's TrueNorth over them,
    /// bit for bit. A kernel that does not compile is infeasible; the
    /// paired-timing paths name that a pooled problem has none.
    /// THE DEVICE JUDGE'S TABLES AND SUMS on the hand-made case: the device's
    /// per-query words as worked by hand above (found, pooled, broken: stage 1
    /// q0 2 3 1, q1 1 1 0, q2 2 5 2; stage 2 q0 1 2 0, q1 2 6 2, q2 0 0 0)
    /// sum to exactly the host's windows; the tables hold each stage's accept
    /// offsets absolute into the concatenated rows, the query rows, the
    /// arrived rows; a wrong layout, a pool past 256 or a top past 16 is named.
    #[test]
    fn the_device_judges_tables_and_sums_match_the_host() {
        let spec: ProblemSpec = toml::from_str(&pooled_problem_text("none")).unwrap();
        let p = Problem { dir: PathBuf::new(), spec, suites: Vec::new(), reference: String::new() };
        let (data, env) = pooled_staged(&p);
        let out = p.spec.output[0].clone();
        let bytes: Vec<u8> = pooled_cands().iter().flat_map(|x| x.to_le_bytes()).collect();
        let inp = judge_inputs(&p, &out, &env, &data, bytes.len()).unwrap();
        assert_eq!((inp.stages, inp.seeds, inp.queries, inp.c_max, inp.top), (2, 2, 3, 4, 2));
        assert_eq!(inp.accept_offsets.len(), 2 * (inp.queries + 1));
        assert_eq!(inp.accept_offsets[inp.queries + 1], inp.accept_offsets[inp.queries], "stage 2's offsets start where stage 1's rows end");
        assert_eq!(*inp.accept_offsets.last().unwrap() as usize, inp.accept_rows.len());
        assert_eq!(inp.arrived.len(), 2);
        let words: Vec<u32> = vec![2, 3, 1, 1, 1, 0, 2, 5, 2, 1, 2, 0, 2, 6, 2, 0, 0, 0];
        assert_eq!(windows_of_judged(&inp, &words).unwrap(), pooled_windows(&p, &out, &env, &data, &bytes).unwrap());
        assert!(windows_of_judged(&inp, &words[3..]).is_err(), "a short judge");
        assert!(judge_inputs(&p, &out, &env, &data, bytes.len() - 4).unwrap_err().contains("the layout is 2 stages x 2 seeds x 3 queries x 4 u32"));
    }

    /// The pooled judge's WGSL validates on the host (naga, fuller's validator).
    #[test]
    fn the_pooled_judge_kernel_validates() {
        crate::wgsl::render::validate_text(include_str!("pooled_judge.wgsl")).unwrap();
    }

    /// THE DEVICE JUDGE ON THE HAND-MADE CASE (GPU, built, not run): the
    /// device's windows are the host's, every count worked by hand.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_device_judge_counts_the_hand_case_on_the_gpu() {
        let dev = crate::wgsl::oracle::device::Device::new().expect("device");
        let exec = DeviceExecutor::new(&dev, crate::wgsl::oracle::device::Fuel::Proven("a hand kernel of this test"));
        let spec: ProblemSpec = toml::from_str(&pooled_problem_text("none")).unwrap();
        let p = Problem { dir: PathBuf::new(), spec, suites: Vec::new(), reference: String::new() };
        let (data, env) = pooled_staged(&p);
        let out = p.spec.output[0].clone();
        let bytes: Vec<u8> = pooled_cands().iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(exec.judge_pooled(&p, &out, &env, &data, &bytes).unwrap(), pooled_windows(&p, &out, &env, &data, &bytes).unwrap());
        // Again, the case's tables resident: the same windows.
        assert_eq!(exec.judge_pooled(&p, &out, &env, &data, &bytes).unwrap(), pooled_windows(&p, &out, &env, &data, &bytes).unwrap());
    }

    /// THE DEVICE JUDGE AGAINST THE HOST ON RUN 3'S KERNELS (GPU, built, not
    /// run): problems/v24_ingest's stream case (25,000 + 2,500, stages 1..3),
    /// eight feasible kernels of run 3's strict arm (tests/fixtures/v24_judge,
    /// from logs/V24/run3_strict/kernels), each run once with its reads; the
    /// device's exact re-rank gives every window's found, pooled and broken
    /// counts exactly as the host's f64 re-rank (`pooled_windows`). The
    /// seconds each judge takes are printed.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_device_judge_counts_as_the_host_on_run_3s_kernels_on_the_gpu() {
        let dev = crate::wgsl::oracle::device::Device::new().expect("device");
        let p = Problem::load(&root().join("problems/v24_ingest")).unwrap();
        let exec = DeviceExecutor::for_problem(&dev, &p, crate::wgsl::oracle::device::Fuel::Proven("a hand kernel of this test"));
        let limit = exec.binding_limit();
        let c = &p.suites[0].cases[0];
        let data = p.case_data(c, limit).unwrap();
        let env = p.env(c, limit).unwrap();
        let out = p.pooled_output().cloned().expect("pooled_recall");
        let oi = p.spec.output.iter().position(|o| o.name == out.name).unwrap();
        let (_, info0) = exec.compile(&p, &p.reference).unwrap();
        let (staged, _) = exec.stage(&p.plan(c, limit, Some(&info0)).unwrap(), &data).unwrap();
        let dir = root().join("tests/fixtures/v24_judge");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).filter(|f| f.extension().is_some_and(|x| x == "wgsl")).collect();
        files.sort();
        assert_eq!(files.len(), 8, "the fixtures");
        for f in &files {
            let text = std::fs::read_to_string(f).unwrap();
            let (k, info) = exec.compile(&p, &text).unwrap();
            let plan = p.plan(c, limit, Some(&info)).unwrap();
            let r = exec.run(&k, &plan, &data, &staged, true).unwrap();
            let bytes = &r.outputs[oi];
            let t = Instant::now();
            let device = exec.judge_pooled(&p, &out, &env, &data, bytes).unwrap();
            let device_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let host = pooled_windows(&p, &out, &env, &data, bytes).unwrap();
            let host_ms = t.elapsed().as_secs_f64() * 1e3;
            println!("{}: device judge {device_ms:.1} ms, host {host_ms:.1} ms; windows {:?}", f.display(), device.iter().map(|w| (w.found, w.pooled, w.broken)).collect::<Vec<_>>());
            assert_eq!(device, host, "{}: the device's counts are not the host's", f.display());
        }
    }

    #[test]
    fn the_pooled_fitness_is_truenorth_over_the_inputs() {
        let dir = std::env::temp_dir().join("fuller_harness_pooled");
        std::fs::create_dir_all(dir.join("cases/stream")).unwrap();
        std::fs::write(dir.join("ref.wgsl"), POOLED_WGSL).unwrap();
        std::fs::write(dir.join("problem.toml"), pooled_problem_text("ref.wgsl")).unwrap();
        let mut inputs = String::new();
        for (n, b) in pooled_inputs_data() {
            std::fs::write(dir.join(format!("cases/stream/{n}.bin")), b).unwrap();
            inputs.push_str(&format!("{n} = \"{n}.bin\", "));
        }
        let manifest = format!("root = \"stream\"\n[[case]]\nname = \"hand\"\nparams = {{ nnz = {NNZ}, n_stream = 6, settle_1 = 10, settle_2 = 20 }}\ninputs = {{ {} }}\n", inputs.trim_end_matches(", "));
        std::fs::write(dir.join("cases/stream.toml"), manifest).unwrap();
        let mut exec = HostExecutor::new(LIMIT);
        exec.register(POOLED_WGSL, Box::new(PooledModel));
        let h = Harness::new(Problem::load(&dir).unwrap(), &exec).unwrap();
        let f = h.evaluate(POOLED_WGSL);
        assert!(f.feasible, "{f:?}");
        let ingest_ms = |_stage: usize| 2.0 * (exec.dispatch_ms + 5.0 * exec.ns_per_op / 1e6);
        let search_ms = 2.0 * (exec.dispatch_ms + 50.0 * exec.ns_per_op / 1e6);
        let mut want = vec![0.0, 0.1, 0.3, 0.5, 0.025, 0.0, 1.0, 0.1, 0.0, 0.0, 0.6, 0.2];
        for _ in 0..2 {
            // The ingest through stage 2 is stages 1 and 2's ingest summed.
            want.extend([time_input(10.0, ingest_ms(1)), time_input(6.0, search_ms), time_input(20.0, ingest_ms(1) + ingest_ms(2)), time_input(12.0, search_ms)]);
        }
        assert_eq!(f.inputs, want);
        let library = hff_core::core_functions::calculate_single_hyperspherical_fitness_f64_with_method(&ndarray::Array1::from(want.clone()), want.len(), false, None, "truenorth");
        assert_eq!(f.hff.to_bits(), library.to_bits());
        assert_eq!(f.hff.to_bits(), crate::score::truenorth_angle(&want).to_bits());
        println!("pooled fitness: θ {:.6} over {} inputs; {}", f.hff, f.inputs.len(), f.detail);
        let bad = h.evaluate("not wgsl");
        assert!(!bad.feasible && bad.detail.starts_with("compile:") && bad.inputs.is_empty() && bad.hff.is_infinite(), "{bad:?}");
        assert!(h.time_shipped().unwrap_err().contains("pooled_recall problem"));
        assert!(h.retime(POOLED_WGSL, &[0.0; 6]).unwrap_err().contains("pooled_recall problem"));
    }

    fn v24_small() -> Problem {
        Problem::load(&root().join("problems/v24_ingest_small")).expect("problems/v24_ingest_small")
    }

    /// THE v24 PROGRAM AS THE CONTRACT SAYS (plan only, no data read): per
    /// stage k and seed, `passes` ingest dispatches of `per_section`
    /// workgroups and one search of `n_queries`, every one binding that
    /// seed's own arenas and a keys buffer of exactly the arrived rows
    /// (k * per_section rows: no future key exists), cfg words (seed, stage,
    /// arrived_begin, arrived_end, n_queries, pass_index, passes, 64); each
    /// stage's cands read at its own stage-axis offset; batch-safe; the
    /// reference kernel compiles with both entries.
    #[test]
    fn the_v24_program_binds_only_arrived_keys() {
        let p = v24_small();
        let info = kernel_info(&p.spec, &p.reference).unwrap();
        assert_eq!((info.entry_workgroups.clone(), info.passes), (vec![[1, 1, 1], [16, 1, 1]], 1));
        let c = &p.suites[0].cases[0];
        for passes in [1u32, 4] {
            let info = KernelInfo { passes, ..info.clone() };
            let plan = p.plan(c, LIMIT, Some(&info)).unwrap();
            assert!(plan.batch_safe);
            let (n, per, d) = (c.params["n_oracle"] as u64, c.params["per_section"] as u64, c.params["d"] as u64);
            let mut at = 0;
            let ds: Vec<&Cmd> = plan.run.iter().filter(|x| matches!(x, Cmd::Dispatch { .. })).collect();
            assert_eq!(ds.len() as u32, 5 * 4 * (passes + 1));
            for stage in 1..=5u64 {
                for seed in 0..4u64 {
                    for pass in 0..=passes {
                        let Cmd::Dispatch { bind, workgroups, entry, sample } = ds[at] else { unreachable!() };
                        at += 1;
                        let search = pass == passes;
                        assert_eq!((*entry, *workgroups), (usize::from(search), [if search { n as u32 } else { per as u32 }, 1, 1]));
                        assert_eq!(sample.as_deref(), Some(format!("{}{stage}", if search { "search" } else { "ingest" }).as_str()));
                        let buf = |b: u32| &plan.buffers[bind.iter().find(|x| x.0 == b).unwrap().1];
                        assert_eq!(buf(0).bytes, stage * per * d * 4, "stage {stage}: the keys binding holds only the arrived rows");
                        assert_eq!((buf(2).name.clone(), buf(3).name.clone()), (format!("iu{seed}"), format!("if{seed}")));
                        let Init::Words(w) = &buf(5).init else { panic!("cfg") };
                        assert_eq!(u32s(w), vec![seed as u32, stage as u32, ((stage - 1) * per) as u32, (stage * per) as u32, n as u32, if search { 0 } else { pass }, passes, 64]);
                    }
                }
            }
            let reads: Vec<u64> = plan.run.iter().filter_map(|x| if let Cmd::Read { dst_offset, .. } = x { Some(*dst_offset) } else { None }).collect();
            assert_eq!(reads, (0..20).map(|i| i * n * 64 * 4).collect::<Vec<_>>());
        }
        // The arenas are made and zeroed by every run, before any stage.
        let plan = p.plan(c, LIMIT, Some(&info)).unwrap();
        let first_dispatch = plan.run.iter().position(|x| matches!(x, Cmd::Dispatch { .. })).unwrap();
        for s in 0..4 {
            for kind in ["iu", "if"] {
                let id = plan.buffers.iter().position(|b| b.name == format!("{kind}{s}")).unwrap();
                assert!(!plan.buffers[id].resident && plan.buffers[id].bytes == 32 << 20);
                assert!(plan.run[..first_dispatch].contains(&Cmd::Create(id)) && plan.run[..first_dispatch].contains(&Cmd::Clear(id)));
            }
        }
    }

    /// THE FULL SET, STAGES 1..3 (run 1): problems/v24_ingest judges three
    /// stages; its largest keys binding is stage 3's 15,000 rows (122.9 MB,
    /// under the 128 MiB binding) and its output 3 stages x 4 seeds x 2,500
    /// queries x 64; four stages (163.8 MB a seed) are a plan error naming
    /// the requirement.
    #[test]
    fn the_full_v24_problem_judges_three_stages_under_the_binding_limit() {
        let p = Problem::load(&root().join("problems/v24_ingest")).expect("problems/v24_ingest");
        let c = &p.suites[0].cases[0];
        assert_eq!(c.params["stages"], 3.0);
        let info = kernel_info(&p.spec, &p.reference).unwrap();
        let plan = p.plan(c, LIMIT, Some(&info)).unwrap();
        let most = plan.buffers.iter().filter(|b| b.resident).map(|b| b.bytes).max().unwrap();
        assert_eq!(most, 15_000 * 2048 * 4);
        assert_eq!(plan.outputs, vec![3 * 4 * 2500 * 64 * 4]);
        assert_eq!(plan.run.iter().filter(|x| matches!(x, Cmd::Dispatch { .. })).count(), 3 * 4 * 2);
        let mut four = c.clone();
        four.params.insert("stages".into(), 4.0);
        let e = p.plan(&four, LIMIT, None).unwrap_err();
        assert!(e.contains("the problem requires stages * per_section * d * 4 <= binding_limit"), "{e}");
    }

    /// THE REFERENCE KERNEL END TO END ON THE CPU EXECUTOR (its own WGSL,
    /// interpreted) on the exporter's small case, cut to two stages and the
    /// first two oracles, arenas of 1,024 words, one timing run: Harness::new
    /// then evaluate. Feasible; brute force returns only arrived rows (B 0),
    /// 64 a seed (the seeds' lists overlap: P 64..256 a query); 2 stages x 2 halves x 3 + 1 run x 2
    /// stages x 2 inputs; θ the library's. Each returned row's key cosine is
    /// within 1e-5 of the seed's 64th best (an f64 brute force here).
    #[test]
    fn the_v24_reference_runs_end_to_end_on_the_cpu_executor() {
        let case_dir = Path::new("/Users/andrewmorgan/data/arm/v24_case_small");
        if !case_dir.join("case.json").exists() {
            println!("{} is not present: the reference's end-to-end CPU run is not made", case_dir.display());
            return;
        }
        let t0 = Instant::now();
        let dir = std::env::temp_dir().join("fuller_harness_v24_cpu");
        std::fs::create_dir_all(dir.join("cases")).unwrap();
        let src = root().join("problems/v24_ingest_small");
        let text = std::fs::read_to_string(src.join("problem.toml")).unwrap();
        let text = text.replace("reference = \"reference.wgsl\"", &format!("reference = {:?}", src.join("reference.wgsl").display().to_string())).replace("timing_rounds = 3", "timing_rounds = 1").replace("deadline_factor = 20.0\ndeadline_floor_ms = 120000.0\n", "");
        assert!(text.contains("timing_rounds = 1") && !text.contains("deadline_factor"));
        std::fs::write(dir.join("problem.toml"), text).unwrap();
        let manifest = std::fs::read_to_string(src.join("cases/stream.toml")).unwrap();
        let manifest = manifest.replacen("params = { ", "params = { stages = 2, n_queries = 2, arena_u32 = 1024, arena_f32 = 1024, ", 1);
        std::fs::write(dir.join("cases/stream.toml"), manifest).unwrap();
        let exec = CpuExecutor::new(LIMIT);
        let h = Harness::new(Problem::load(&dir).unwrap(), &exec).unwrap();
        let f = h.evaluate(&h.problem.reference);
        println!("reference on the CPU executor: {:.1} s; θ {:.6}; inputs {:?}; {}", t0.elapsed().as_secs_f64(), f.hff, f.inputs, f.detail);
        assert!(f.feasible, "{f:?}");
        assert_eq!(f.inputs.len(), 2 * 2 * 3 + 2 * 2);
        let library = hff_core::core_functions::calculate_single_hyperspherical_fitness_f64_with_method(&ndarray::Array1::from(f.inputs.clone()), f.inputs.len(), false, None, "truenorth");
        assert_eq!(f.hff.to_bits(), library.to_bits());
        // The windows and the candidates, from one more run.
        let s = &h.constraint[0];
        let (k, info) = exec.compile(&h.problem, &h.problem.reference).unwrap();
        let plan = h.problem.plan(&s.case, LIMIT, Some(&info)).unwrap();
        let r = exec.run(&k, &plan, &s.data, &s.staged, true).unwrap();
        let out = h.problem.pooled_output().unwrap().clone();
        let w = pooled_windows(&h.problem, &out, &s.env, &s.data, &r.outputs[0]).unwrap();
        for x in &w {
            assert_eq!(x.broken, 0, "{x:?}");
            assert!(x.pooled >= 64 * x.queries && x.pooled <= 256 * x.queries, "{x:?}");
        }
        let cands = u32s(&r.outputs[0]);
        let f32s = |name: &str| Elem::F32.decode(&s.data.inputs[name]);
        for seed in 0..4 {
            let (keys, queries) = (f32s(&format!("keys{seed}")), f32s(&format!("queries{seed}")));
            for stage in 1..=2usize {
                for q in 0..2usize {
                    let qv = &queries[q * 2048..(q + 1) * 2048];
                    let cos = |r: usize| {
                        let kv = &keys[r * 2048..(r + 1) * 2048];
                        let (dot, kn, qn) = kv.iter().zip(qv).fold((0.0, 0.0, 0.0), |(a, b, c), (x, y)| (a + x * y, b + x * x, c + y * y));
                        dot / (kn * qn).sqrt()
                    };
                    let mut all: Vec<f64> = (0..stage * 500).map(cos).collect();
                    all.sort_by(|a, b| b.total_cmp(a));
                    let got = &cands[((stage - 1) * 4 + seed) * 2 * 64 + q * 64..][..64];
                    let mut seen = std::collections::BTreeSet::new();
                    for &c in got {
                        assert!((c as usize) < stage * 500 && seen.insert(c), "seed {seed} stage {stage} query {q}: row {c}");
                        assert!(cos(c as usize) >= all[63] - 1e-5, "seed {seed} stage {stage} query {q}: row {c} cosine {} under the 64th {}", cos(c as usize), all[63]);
                    }
                }
            }
        }
        println!("{w:?}");
    }

    /// THE v24 REFERENCE ON THE DEVICE (GPU; run by the session that owns
    /// the GPU): problems/v24_ingest_small whole (five stages, 250 queries,
    /// 32 MiB arenas, three timing runs) through Harness::new and evaluate
    /// on the device executor: every entry its own pipeline, every dispatch
    /// timed. Feasible, B 0 in every window, 30 + 3 x 10 inputs, θ the
    /// library's; the windows and the per-stage ms printed.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_v24_reference_runs_on_the_device() {
        let dev = crate::wgsl::oracle::device::Device::new().expect("device");
        let exec = DeviceExecutor::new(&dev, crate::wgsl::oracle::device::Fuel::Proven("a hand kernel of this test"));
        let t0 = Instant::now();
        let h = Harness::new(v24_small(), &exec).unwrap();
        let f = h.evaluate(&h.problem.reference);
        println!("v24 reference on the device: {:.1} s; θ {:.6}; {} inputs {:?}; {}", t0.elapsed().as_secs_f64(), f.hff, f.inputs.len(), f.inputs, f.detail);
        assert!(f.feasible, "{f:?}");
        assert_eq!(f.inputs.len(), 5 * 2 * 3 + 3 * 5 * 2);
        for w in f.inputs.chunks(3).take(10) {
            assert_eq!(w[2], 0.0, "brute force returns only arrived rows");
        }
        let library = hff_core::core_functions::calculate_single_hyperspherical_fitness_f64_with_method(&ndarray::Array1::from(f.inputs.clone()), f.inputs.len(), false, None, "truenorth");
        assert_eq!(f.hff.to_bits(), library.to_bits());
    }

    /// THE FULL SET'S REFERENCE RETURNS CANDIDATES AT STAGE 1 (run 1's
    /// gen-0 tree rows r0 and r33 judged P ~2 a query and x_M 1 on
    /// problems/v24_ingest, 03:10Z): problems/v24_ingest cut to stage 1 and
    /// its first 2 oracles, on the CPU executor (8 oracles took 443 s:
    /// the text gave about 100 candidates a query and missed nothing, today
    /// 0; the rendering 2 a query, 40 of 50 and 24 of 30 missed). Both the reference's own
    /// text and the safe kingdom's rendering of it (what a `tree` row runs:
    /// flow::shipped_gates) must give every query 64 arrived rows a seed (P
    /// 64..256 a query, B 0) and miss no more than today's index on every
    /// window, and the two must give the same candidates.
    #[test]
    fn the_full_v24_reference_and_its_rendering_return_candidates_at_stage_1() {
        let case_dir = Path::new("/Users/andrewmorgan/data/arm/v24_case");
        if !case_dir.join("case.json").exists() {
            println!("{} is not present: not run", case_dir.display());
            return;
        }
        let src = root().join("problems/v24_ingest");
        let dir = std::env::temp_dir().join("fuller_harness_v24_full_s1");
        std::fs::create_dir_all(dir.join("cases")).unwrap();
        let reference = root().join("problems/v24_ingest_small/reference.wgsl");
        let text = std::fs::read_to_string(src.join("problem.toml")).unwrap().replace("reference = \"../v24_ingest_small/reference.wgsl\"", &format!("reference = {:?}", reference.display().to_string())).replace("timing_rounds = 3", "timing_rounds = 1").replace("deadline_factor = 20.0\ndeadline_floor_ms = 120000.0\n", "");
        assert!(text.contains("timing_rounds = 1") && !text.contains("deadline_factor"));
        std::fs::write(dir.join("problem.toml"), text).unwrap();
        let manifest = std::fs::read_to_string(src.join("cases/stream.toml")).unwrap().replace("stages = 3,", "stages = 1,").replacen("params = { ", "params = { n_queries = 2, arena_u32 = 1024, arena_f32 = 1024, ", 1);
        assert!(manifest.contains("stages = 1,"));
        std::fs::write(dir.join("cases/stream.toml"), manifest).unwrap();
        let exec = CpuExecutor::new(LIMIT);
        let h = Harness::new(Problem::load(&dir).unwrap(), &exec).unwrap();
        let kingdom = crate::wgsl::loader::WgslKingdom::load_safe();
        let rendered = crate::wgsl::flow::shipped_gates(&src, &kingdom, &crate::wgsl::roundtrip::Trial::default()).unwrap().rendered;
        let s = &h.constraint[0];
        let out = h.problem.pooled_output().unwrap().clone();
        let mut lists = Vec::new();
        let mut bad = Vec::new();
        for (form, kernel) in [("text", h.problem.reference.clone()), ("rendered", rendered)] {
            let (k, info) = exec.compile(&h.problem, &kernel).unwrap_or_else(|e| panic!("{form}: {e}"));
            let plan = h.problem.plan(&s.case, LIMIT, Some(&info)).unwrap();
            let r = exec.run(&k, &plan, &s.data, &s.staged, true).unwrap_or_else(|e| panic!("{form}: {e}"));
            let w = pooled_windows(&h.problem, &out, &s.env, &s.data, &r.outputs[0]).unwrap();
            println!("{form}: {w:?}");
            for x in &w {
                if x.broken != 0 || x.pooled < 64 * x.queries || x.pooled > 256 * x.queries || x.misses > x.misses_today {
                    bad.push(format!("{form}: {x:?}"));
                }
            }
            lists.push(u32s(&r.outputs[0]));
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
        assert_eq!(lists[0], lists[1], "the rendering's candidates differ from the text's");
    }

    /// THE ADMISSION CASE (an unjudged cut of the small set for phylu's
    /// differential admission: 2 stages of 32 rows, 16 queries, the
    /// problem's own 32 MiB arenas): problems/v24_ingest_small/cases/admission.toml loads, its
    /// plan binds 32 then 64 arrived rows, and the reference's text run on it
    /// by `cpu_run` gives 64 arrived rows a query a seed (stage 1 holds 32
    /// rows: 32 and 32 pads). Timed, printed.
    #[test]
    fn the_v24_admission_case_runs_on_the_cpu_executor() {
        let p = v24_small();
        let cases = p.cases_of("cases/admission.toml").unwrap();
        assert_eq!(cases.len(), 1);
        let c = &cases[0];
        if !c.inputs.values().all(|f| f.exists()) {
            println!("the small case files are not present: not run");
            return;
        }
        let t = Instant::now();
        let r = cpu_run(&p, c, &p.reference, LIMIT).unwrap();
        println!("the reference on the admission case: {:.1} s", t.elapsed().as_secs_f64());
        let got = u32s(&r.outputs[0]);
        assert_eq!(got.len(), 2 * 4 * 16 * 64);
        for (i, list) in got.chunks(64).enumerate() {
            let stage = i / (4 * 16) + 1;
            let rows: Vec<u32> = list.iter().copied().filter(|&x| x != PAD).collect();
            assert_eq!(rows.len(), (32 * stage).min(64), "list {i}");
            assert!(rows.iter().all(|&x| (x as usize) < 32 * stage), "list {i}: {rows:?}");
        }
    }

    /// THE ADMISSION CASE BINDS THE PROBLEM'S ARENAS: in both v24 problems
    /// the admission case gives no arena size, and its plan's eight arenas
    /// (iu0..iu3, if0..if3) are each `arena_u32` / `arena_f32` words of
    /// problem.toml (8,388,608, 32 MiB), the stream case's, the size the
    /// designs mask for. Plans only: nothing runs.
    #[test]
    fn the_admission_case_binds_the_problems_arenas() {
        for name in ["v24_ingest_small", "v24_ingest"] {
            let p = Problem::load(&root().join("problems").join(name)).unwrap();
            let info = kernel_info(&p.spec, &p.reference).unwrap();
            let admission = p.cases_of("cases/admission.toml").unwrap().remove(0);
            assert!(!admission.params.contains_key("arena_u32") && !admission.params.contains_key("arena_f32"), "{name}: the admission case gives its own arena size");
            let declared = |k: &str| eval_int(&p.spec.params[k], &Env::new()).unwrap();
            assert_eq!((declared("arena_u32"), declared("arena_f32")), (8_388_608, 8_388_608), "{name}");
            let arenas = |c: &Case| {
                let plan = p.plan(c, LIMIT, Some(&info)).unwrap();
                let mut v: Vec<(String, u64)> = plan.buffers.iter().filter(|b| (b.name.starts_with("iu") || b.name.starts_with("if")) && !b.resident).map(|b| (b.name.clone(), b.bytes)).collect();
                v.sort();
                v
            };
            let a = arenas(&admission);
            let want: Vec<(String, u64)> = (0..4).map(|s| (format!("if{s}"), 4 * declared("arena_f32"))).chain((0..4).map(|s| (format!("iu{s}"), 4 * declared("arena_u32")))).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
            assert_eq!(a, want, "{name}: the admission case's arenas");
            assert_eq!(a, arenas(&p.suites[0].cases[0]), "{name}: the admission and stream cases' arenas");
        }
    }

    /// v24_ingest5 PLANS ALL FIVE STAGES BY THE ADAPTER'S LIMIT, ITS KEYS
    /// PROJECTED ON THE DEVICE (plans only): it declares binding_limit =
    /// "adapter" and judges sections 1..5; no key or query is an input (the
    /// CSR records are the only resident inputs); its setup runs the
    /// machinery kernel `project` 24 times (per seed its queries, then per
    /// stage k its keys of rows 0..k * 5,000, one workgroup a row, cfg (seed,
    /// 0, first row, rows)); every design dispatch binds the stage's
    /// projected keys (stage 5: 204.8 MB). At a 256 MiB limit it plans; at the
    /// 128 MiB effective limit the require names the binding limit. v24_ingest
    /// (run 3's) keeps the effective limit, three stages and its key inputs.
    #[test]
    fn v24_ingest5_plans_five_stages_by_the_adapter_limit() {
        let p = Problem::load(&root().join("problems/v24_ingest5")).expect("problems/v24_ingest5");
        assert_eq!(p.spec.binding_limit, LimitRule::Adapter);
        let old = Problem::load(&root().join("problems/v24_ingest")).unwrap();
        assert!(old.spec.binding_limit == LimitRule::Effective && old.spec.machine.is_empty() && old.spec.input.iter().any(|i| i.name == "keys0"));
        assert!(!p.spec.input.iter().any(|i| i.name.starts_with("keys") || i.name.starts_with("queries")), "a key is a host input");
        let c = &p.suites[0].cases[0];
        assert!(!c.params.contains_key("stages") && !c.inputs.keys().any(|k| k.starts_with("keys") || k.starts_with("queries")));
        let info = kernel_info(&p.spec, &p.reference).unwrap();
        let plan = p.plan(c, 256 << 20, Some(&info)).unwrap();
        assert!(plan.buffers.iter().filter(|b| b.resident).all(|b| match &b.init { Init::Input { tensor, .. } => tensor.starts_with("rec_"), _ => true }), "a resident input that is not a record");
        assert_eq!((plan.machines.len(), plan.setup_dispatch.len()), (1, 4 * 6));
        let m = &plan.machines[0];
        let module = crate::wgsl::naga_shape::parse(&m.source).unwrap_or_else(|e| panic!("project.wgsl: {e}"));
        crate::wgsl::target::Target::current().validate(&module, &m.source).unwrap_or_else(|e| panic!("project.wgsl: {e}"));
        assert!(module.entry_points.iter().any(|e| e.name == m.entry && e.workgroup_size == [256, 1, 1]));
        let seeds = [42u32, 137, 911, 4242];
        for (n, d) in plan.setup_dispatch.iter().enumerate() {
            let (s, k) = (n / 6, n % 6);
            let buf = |b: u32| &plan.buffers[d.bind.iter().find(|x| x.0 == b).unwrap().1];
            let Init::Words(w) = &buf(4).init else { panic!("pcfg") };
            let (name, first, rows) = if k == 0 { (format!("queries{s}"), 25_000, 2_500) } else { (format!("keys{s}_{}", k - 1), 0, 5_000 * k as u32) };
            assert_eq!((buf(3).name.clone(), buf(3).bytes, d.workgroups), (name, u64::from(rows) * 2048 * 4, [rows, 1, 1]));
            assert_eq!(u32s(w), vec![seeds[s], 0, first, rows]);
        }
        let most = plan.buffers.iter().filter(|b| b.resident).map(|b| b.bytes).max().unwrap();
        assert_eq!(most, 25_000 * 2048 * 4);
        let ds: Vec<&Cmd> = plan.run.iter().filter(|x| matches!(x, Cmd::Dispatch { .. })).collect();
        assert_eq!(ds.len(), 5 * 4 * 2);
        for (n, c) in ds.iter().enumerate() {
            let Cmd::Dispatch { bind, .. } = c else { unreachable!() };
            let (stage, seed) = (n / 8, (n % 8) / 2);
            assert_eq!(plan.buffers[bind.iter().find(|x| x.0 == 0).unwrap().1].name, format!("keys{seed}_{stage}"));
        }
        assert_eq!(plan.outputs, vec![5 * 4 * 2500 * 64 * 4]);
        let e = p.plan(c, LIMIT, None).unwrap_err();
        assert!(e.contains("the problem requires stages * per_section * d * 4 <= binding_limit"), "{e}");
    }

    /// THE KEYS PROJECTED ON THE DEVICE ARE QDRANT'S (GPU; run by the session
    /// that owns the GPU): v24_ingest5's stream case staged at the adapter's
    /// limit (`for_problem`), its setup projecting every seed's keys from the
    /// CSR records; each seed's stage-5 keys (rows 0..25,000, one 204.8 MB
    /// buffer) and queries (rows 25,000..27,500) read back and compared with
    /// qdrant's TRLKEYS1 keys of v24_main (the same rows): printed, the bits
    /// that match exactly and the largest difference; every component within
    /// 1e-6 (qdrant's norm is f64, this kernel's f32: see project.wgsl). Then
    /// the reference run once on the device: every window has candidates and
    /// no broken entry.
    #[cfg(feature = "gpu")]
    #[test]
    fn v24_ingest5_projects_its_keys_on_the_device() {
        let dev = crate::wgsl::oracle::device::Device::new().expect("device");
        let p = Problem::load(&root().join("problems/v24_ingest5")).unwrap();
        let exec = DeviceExecutor::for_problem(&dev, &p, crate::wgsl::oracle::device::Fuel::Proven("a hand kernel of this test"));
        let limit = exec.binding_limit();
        println!("v24_ingest5: binding limit {limit} bytes (the effective one {})", crate::wgsl::gram::effective_buffer_limit(&dev.device));
        assert!(limit >= 25_000 * 2048 * 4, "the adapter's limit {limit} is under stage 5's keys");
        let c = &p.suites[0].cases[0];
        let t = Instant::now();
        let (k, info) = exec.compile(&p, &p.reference).unwrap();
        let data = p.case_data(c, limit).unwrap();
        let plan = p.plan(c, limit, Some(&info)).unwrap();
        let (staged, stage_ms) = exec.stage(&plan, &data).unwrap();
        println!("staged, the keys projected, in {stage_ms:.0} ms");
        let dir = Path::new("/Users/andrewmorgan/Data/arm/v24_test/db/udv23_keys/v24_main");
        let (mut exact, mut total, mut worst) = (0u64, 0u64, 0.0f64);
        for (s, seed) in [42u64, 137, 911, 4242].iter().enumerate() {
            let bytes = std::fs::read(dir.join(format!("s{seed}.trlkeys"))).unwrap();
            assert_eq!(&bytes[..8], b"TRLKEYS1");
            let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
            let d = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
            assert_eq!((n, d), (27_500, 2048));
            let want: Vec<f32> = bytemuck_f32(&bytes[24..]);
            for (name, first) in [(format!("keys{s}_4"), 0usize), (format!("queries{s}"), 25_000)] {
                let id = plan.buffers.iter().position(|b| b.name == name).unwrap();
                let n_bytes = plan.buffers[id].bytes;
                let read = Plan { run: vec![Cmd::Read { src: id, src_offset: 0, output: 0, dst_offset: 0, bytes: n_bytes }], outputs: vec![n_bytes], ..plan.clone() };
                let got = bytemuck_f32(&exec.run(&k, &read, &data, &staged, true).unwrap().outputs[0]);
                let w = &want[first * 2048..first * 2048 + got.len()];
                for (g, x) in got.iter().zip(w) {
                    total += 1;
                    if g.to_bits() == x.to_bits() {
                        exact += 1;
                    }
                    worst = worst.max((f64::from(*g) - f64::from(*x)).abs());
                }
            }
        }
        println!("projected keys against qdrant's TRLKEYS1: {exact} of {total} components bit-exact ({:.4} %), largest difference {worst:.3e}", 100.0 * exact as f64 / total as f64);
        assert!(worst <= 1e-6, "a projected component differs from qdrant's by {worst:e}");
        let r = exec.run(&k, &plan, &data, &staged, true).unwrap();
        let env = p.env(c, limit).unwrap();
        let out = p.pooled_output().unwrap().clone();
        let w = pooled_windows(&p, &out, &env, &data, &r.outputs[0]).unwrap();
        println!("the reference run: {:.0} ms GPU, {:.1} s in all; {w:?}", r.gpu_ms, t.elapsed().as_secs_f64());
        assert_eq!(w.len(), 10);
        for x in &w {
            assert!(x.pooled > 0 && x.broken == 0, "{x:?}");
        }
    }

    /// v24_ingest5b: INGEST SEES ONLY ITS MICROBATCH'S KEYS, SEARCH NONE
    /// (plans only, at the 128 MiB effective limit): no binding over
    /// 128 MiB; the setup projects per seed its queries and, per stage k, the
    /// keys of microbatch k only (rows (k - 1) * 5,000.., 5,000 of them,
    /// cfg (seed, 0, first row, 5,000)); every ingest dispatch binds that
    /// microbatch's 41 MB of keys, every search dispatch the 64 zero floats
    /// (nokeys); the cfg words and entries are v24_ingest5's.
    #[test]
    fn v24_ingest5b_binds_only_the_microbatchs_keys_at_ingest_and_none_at_search() {
        let p = Problem::load(&root().join("problems/v24_ingest5b")).expect("problems/v24_ingest5b");
        assert_eq!(p.spec.binding_limit, LimitRule::Effective);
        let five = Problem::load(&root().join("problems/v24_ingest5")).unwrap();
        assert_eq!((p.spec.kernel.entry_names(), p.spec.binding.iter().map(|b| (b.binding, b.name.clone(), b.kind, b.fields.clone())).collect::<Vec<_>>()), (five.spec.kernel.entry_names(), five.spec.binding.iter().map(|b| (b.binding, b.name.clone(), b.kind, b.fields.clone())).collect::<Vec<_>>()));
        let c = &p.suites[0].cases[0];
        let info = kernel_info(&p.spec, &p.reference).unwrap();
        let plan = p.plan(c, LIMIT, Some(&info)).unwrap();
        assert!(plan.buffers.iter().all(|b| b.bytes <= LIMIT), "a buffer over the 128 MiB binding");
        assert_eq!(plan.setup_dispatch.len(), 4 * 6);
        for (n, d) in plan.setup_dispatch.iter().enumerate() {
            let (s, k) = (n / 6, n % 6);
            let buf = |b: u32| &plan.buffers[d.bind.iter().find(|x| x.0 == b).unwrap().1];
            let Init::Words(w) = &buf(4).init else { panic!("pcfg") };
            let (name, first, rows) = if k == 0 { (format!("queries{s}"), 25_000, 2_500) } else { (format!("mb{s}_{}", k - 1), 5_000 * (k as u32 - 1), 5_000) };
            assert_eq!((buf(3).name.clone(), buf(3).bytes, d.workgroups), (name, u64::from(rows) * 2048 * 4, [rows, 1, 1]));
            assert_eq!(u32s(w)[2..], [first, rows]);
        }
        let ds: Vec<&Cmd> = plan.run.iter().filter(|x| matches!(x, Cmd::Dispatch { .. })).collect();
        assert_eq!(ds.len(), 5 * 4 * 2);
        for (n, c) in ds.iter().enumerate() {
            let Cmd::Dispatch { bind, entry, .. } = c else { unreachable!() };
            let (stage, seed) = (n / 8, (n % 8) / 2);
            let keys = &plan.buffers[bind.iter().find(|x| x.0 == 0).unwrap().1];
            let Init::Words(w) = &plan.buffers[bind.iter().find(|x| x.0 == 5).unwrap().1].init else { panic!("cfg") };
            assert_eq!(u32s(w)[1..4], [stage as u32 + 1, 5_000 * stage as u32, 5_000 * (stage as u32 + 1)]);
            if *entry == 0 {
                assert_eq!((keys.name.clone(), keys.bytes), (format!("mb{seed}_{stage}"), 5_000 * 2048 * 4));
            } else {
                assert_eq!((keys.name.as_str(), keys.bytes), ("nokeys", 256));
            }
        }
    }

    /// v24_ingest5b ON THE DEVICE (GPU; run by the session that owns the
    /// GPU): its stream case staged at the effective limit (`for_problem`),
    /// every seed's microbatch keys projected on the device; microbatch 5's
    /// keys (stream rows 20,000..25,000) and the queries read back and
    /// compared with qdrant's TRLKEYS1 keys of the same rows (bits exact and
    /// the largest difference printed, within 1e-6); then the reference run
    /// once: it completes, no broken entry (its search sees no keys, so it
    /// returns pads: P 0 is expected and printed).
    #[cfg(feature = "gpu")]
    #[test]
    fn v24_ingest5b_projects_microbatch_keys_on_the_device() {
        let dev = crate::wgsl::oracle::device::Device::new().expect("device");
        let p = Problem::load(&root().join("problems/v24_ingest5b")).unwrap();
        let exec = DeviceExecutor::for_problem(&dev, &p, crate::wgsl::oracle::device::Fuel::Proven("a hand kernel of this test"));
        let limit = exec.binding_limit();
        let c = &p.suites[0].cases[0];
        let t = Instant::now();
        let (k, info) = exec.compile(&p, &p.reference).unwrap();
        let data = p.case_data(c, limit).unwrap();
        let plan = p.plan(c, limit, Some(&info)).unwrap();
        let (staged, stage_ms) = exec.stage(&plan, &data).unwrap();
        println!("v24_ingest5b at {limit} bytes a binding: staged, the microbatch keys projected, in {stage_ms:.0} ms");
        let dir = Path::new("/Users/andrewmorgan/Data/arm/v24_test/db/udv23_keys/v24_main");
        let (mut exact, mut total, mut worst) = (0u64, 0u64, 0.0f64);
        for (s, seed) in [42u64, 137, 911, 4242].iter().enumerate() {
            let bytes = std::fs::read(dir.join(format!("s{seed}.trlkeys"))).unwrap();
            assert_eq!(&bytes[..8], b"TRLKEYS1");
            let want: Vec<f32> = bytemuck_f32(&bytes[24..]);
            for (name, first) in [(format!("mb{s}_4"), 20_000usize), (format!("queries{s}"), 25_000)] {
                let id = plan.buffers.iter().position(|b| b.name == name).unwrap();
                let n_bytes = plan.buffers[id].bytes;
                let read = Plan { run: vec![Cmd::Read { src: id, src_offset: 0, output: 0, dst_offset: 0, bytes: n_bytes }], outputs: vec![n_bytes], ..plan.clone() };
                let got = bytemuck_f32(&exec.run(&k, &read, &data, &staged, true).unwrap().outputs[0]);
                for (g, x) in got.iter().zip(&want[first * 2048..first * 2048 + got.len()]) {
                    total += 1;
                    if g.to_bits() == x.to_bits() {
                        exact += 1;
                    }
                    worst = worst.max((f64::from(*g) - f64::from(*x)).abs());
                }
            }
        }
        println!("microbatch keys against qdrant's TRLKEYS1: {exact} of {total} components bit-exact ({:.4} %), largest difference {worst:.3e}", 100.0 * exact as f64 / total as f64);
        assert!(worst <= 1e-6, "a projected component differs from qdrant's by {worst:e}");
        let r = exec.run(&k, &plan, &data, &staged, true).unwrap();
        let env = p.env(c, limit).unwrap();
        let out = p.pooled_output().unwrap().clone();
        let w = pooled_windows(&p, &out, &env, &data, &r.outputs[0]).unwrap();
        println!("the reference run: {:.0} ms GPU, {:.1} s in all; {w:?}", r.gpu_ms, t.elapsed().as_secs_f64());
        assert_eq!(w.len(), 10);
        assert!(w.iter().all(|x| x.broken == 0), "{w:?}");
    }

    /// v24_ingest5b's MODEL CONTRACT (what the seed calls are given,
    /// `llm::contract`): 32 strategies, the first Andrew's bottom-up
    /// hypothesis whole (count hits, splay into right-sized buckets, promote
    /// at random, fixed ef per layer, scope growing level by level); the
    /// system text carries the section "v24_ingest5b: what binding 0 holds";
    /// the output rule names both entry points. Text only: nothing is called.
    #[cfg(feature = "gpu")]
    #[test]
    fn v24_ingest5b_contract_gives_the_models_32_strategies() {
        let p = Problem::load(&root().join("problems/v24_ingest5b")).unwrap();
        let c = crate::wgsl::llm::contract(Some(&p)).unwrap();
        assert_eq!(c.strategies.len(), 32);
        let first = &c.strategies[0];
        for part in ["bottom-up hypothesis", "Count hits", "splay", "right-sized buckets", "promote", "fixed ef per layer", "scope growing level by level"] {
            assert!(first.contains(part), "the first strategy lacks {part:?}: {first}");
        }
        assert!(c.system.contains("v24_ingest5b: what binding 0 holds"));
        assert!(c.output_rule.contains("`fn ingest`") && c.output_rule.contains("`fn search`"), "{}", c.output_rule);
    }
}
