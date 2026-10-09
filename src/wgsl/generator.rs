//! The oracle's generator (`docs/PLAN_wgsl_lineage.md` §4): small valid
//! WGSL compute kernels over the kingdom's rows, produced from a seed, that
//! reach the decision's refusal classes ON PURPOSE. Each program records
//! which classes it meant to reach, so a run reports intended against
//! achieved: a thousand clean kernels prove little unless the generator
//! reached the programs where sharing would be wrong.
//!
//! Shape: one invocation (`@workgroup_size(1)`, dispatched once), eight
//! lanes per buffer, `xs`/`ks` read, `out`/`cnt`/`ys` written, a uniform
//! `n`; locals `s`, `t` (`f32`), `i`, `acc` (`u32`); a pure helper, a
//! helper that stores to `ys`, and a helper with a `ptr<function>`
//! parameter. Statements: stores, local stores, `let`s, `if`/`else`,
//! bounded `for` loops, calls, and the repeat patterns below. Two
//! operations are kept out of undefined territory because this platform
//! does not follow the spec there (`oracle.rs`): integer division and
//! remainder always divide by `(e | 1u)`, and a float is clamped to
//! `[0, 100]` before `u32()` (an out-of-range conversion saturates at run
//! time on Metal but is folded to an arbitrary value when the compiler can
//! see the operand, found by seed 5 of the first run). Indices are masked
//! `& 7u` except in the programs that mean to go out of range.
//!
//! Repeat patterns, one class each: a repeat under one version (`share`);
//! across a store to its lineage (`lineage_differs`); inside and after a
//! loop that stores its lineage (`lineage_differs`) or leaves it alone
//! (`hoist`); in a branch arm and after the join reading the arm's `let`
//! (`operand_unavailable`); the step-5 `let` after a store to its operand's
//! target (`let_after_store`); a repeat of a local's initialiser
//! (`no_dominating_point`).

/// One generated program.
#[derive(Debug, Clone)]
pub struct Generated {
    pub seed: u64,
    pub source: String,
    /// The classes the program meant to reach (names as the report counts
    /// them: `share`, `hoist`, `lineage_differs`, `operand_unavailable`,
    /// `no_dominating_point`, `let_after_store`, `out_of_range`).
    pub intended: Vec<&'static str>,
    /// The inputs: eight lanes each.
    pub xs: [f32; 8],
    pub ks: [u32; 8],
    pub ys: [f32; 8],
    pub n: u32,
}

impl Generated {
    /// Whether an input lane is NaN or infinite: the device's fast math
    /// assumes neither, so the interpreter (which follows the spec) is
    /// compared on these programs but not held to agree.
    pub fn non_finite_inputs(&self) -> bool {
        self.xs.iter().chain(self.ys.iter()).any(|x| !x.is_finite())
    }
}

/// xorshift64*, enough for a generator that must only be deterministic.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }

    pub fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }

    pub fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        let t = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * t
    }
}

struct Gen {
    rng: Rng,
    lines: Vec<String>,
    /// `let` names in scope, innermost scope last.
    scopes: Vec<Vec<String>>,
    /// Whether `j` (a loop counter) is in scope.
    in_loop: usize,
    next_let: usize,
    intended: Vec<&'static str>,
    out_of_range: bool,
    indent: usize,
}

impl Gen {
    fn line(&mut self, s: impl Into<String>) {
        let pad = "    ".repeat(self.indent);
        self.lines.push(format!("{pad}{}", s.into()));
    }

    fn lets(&self) -> Vec<String> {
        self.scopes.iter().flatten().cloned().collect()
    }

    fn fresh_let(&mut self) -> String {
        self.next_let += 1;
        let name = format!("l{}", self.next_let);
        if let Some(scope) = self.scopes.last_mut() {
            scope.push(name.clone());
        }
        name
    }

    /// A `let` must not be bound to a constant expression: naga folds such a
    /// let into its uses, the reader then sees literal-only operator
    /// applications (`5.0 * 5.0 + 1.0`), and naga folds THOSE when it reads
    /// the rebuilt text back (`26.0`), so the structural gate refuses a
    /// kernel the device runs identically. A literal initialiser gets `t`.
    fn nonconst(&self, e: String) -> String {
        if is_const(&e) {
            format!("({e} + t)")
        } else {
            e
        }
    }

    fn uidx(&mut self) -> String {
        if self.out_of_range && self.rng.chance(30) {
            format!("({} + 9u)", self.uexpr(1))
        } else {
            format!("({} & 7u)", self.uexpr(1))
        }
    }

    fn uexpr(&mut self, depth: usize) -> String {
        if depth == 0 || self.rng.chance(35) {
            let mut leaves = vec!["acc".to_string(), "i".to_string(), "n".to_string(), format!("{}u", self.rng.below(16))];
            if self.in_loop > 0 {
                leaves.push("j".into());
            }
            if self.rng.chance(40) {
                let k = self.rng.below(8);
                leaves.push(format!("ks[{k}u]"));
            }
            return self.rng.pick(&leaves).clone();
        }
        let a = self.uexpr(depth - 1);
        let b = self.uexpr(depth - 1);
        match self.rng.below(7) {
            0 => format!("({a} + {b})"),
            1 => format!("({a} * {b})"),
            2 => format!("({a} & {b})"),
            3 => format!("({a} << {}u)", self.rng.below(5)),
            4 => format!("({a} % ({b} | 1u))"),
            5 => format!("({a} / ({b} | 1u))"),
            _ => {
                // naga refuses converting a float LITERAL to an integer; and
                // an out-of-range float→integer conversion is undefined in
                // MSL (Metal saturates at run time but its compiler folds a
                // constant one to anything), so the operand is clamped first.
                let e = self.fexpr(1);
                let e = if e.parse::<f32>().is_ok() { "t".to_string() } else { e };
                format!("u32(clamp({e}, 0.0, 100.0))")
            }
        }
    }

    fn fexpr(&mut self, depth: usize) -> String {
        if depth == 0 || self.rng.chance(30) {
            let mut leaves = vec!["s".to_string(), "t".to_string(), format!("{:.2}", self.rng.f32_in(-3.0, 3.0))];
            if self.rng.chance(60) {
                let ix = self.uidx();
                leaves.push(format!("xs[{ix}]"));
            }
            if self.rng.chance(30) {
                let ix = self.uidx();
                leaves.push(format!("ys[{ix}]"));
            }
            let lets = self.lets();
            if !lets.is_empty() && self.rng.chance(50) {
                leaves.push(self.rng.pick(&lets).clone());
            }
            return self.rng.pick(&leaves).clone();
        }
        let a = self.fexpr(depth - 1);
        let mut b = self.fexpr(depth - 1);
        // naga folds constant expressions and refuses one that folds to NaN
        // or infinity (`sqrt(-1.08)`, `1.0 / 0.0`): never two literals under
        // one operator, never a literal under sqrt.
        let is_literal = |e: &str| e.parse::<f32>().is_ok();
        if is_literal(&a) && is_literal(&b) {
            b = "s".to_string();
        }
        let a = if is_literal(&a) && self.rng.chance(50) { "t".to_string() } else { a };
        match self.rng.below(10) {
            0 => format!("({a} + {b})"),
            1 => format!("({a} - {b})"),
            2 => format!("({a} * {b})"),
            // No NaN of the program's own making (`x / 0.0`, `sqrt(-x)`):
            // Metal's fast math folds `s / s` and `s - s` to constants whatever
            // s holds, so a NaN that the spec produces and the device does not
            // is a known class, not a representation fault; the generator
            // keeps its own expressions NaN-free and leaves NaN to the inputs.
            3 => format!("({a} / (abs({b}) + 0.5))"),
            4 if !is_literal(&a) => format!("sqrt(abs({a}))"),
            4 => format!("sqrt(abs({a}) + t)"),
            5 => format!("abs({a})"),
            6 => format!("min({a}, {b})"),
            7 => format!("max({a}, {b})"),
            8 => {
                let c = self.bexpr();
                format!("select({a}, {b}, {c})")
            }
            _ => format!("f32({})", self.uexpr(1)),
        }
    }

    fn bexpr(&mut self) -> String {
        match self.rng.below(3) {
            0 => format!("({} < {})", self.fexpr(1), self.fexpr(1)),
            1 => format!("({} == {})", self.uexpr(1), self.uexpr(1)),
            _ => format!("(({} > {}) && ({} != 3u))", self.fexpr(1), self.fexpr(1), self.uexpr(1)),
        }
    }

    /// A repeat worth sharing: at least two operators, no literal-only text.
    fn repeat(&mut self) -> String {
        let a = self.fexpr(1);
        let b = self.fexpr(1);
        match self.rng.below(3) {
            0 => format!("({a} * {b} + xs[({} & 7u)])", self.rng.below(8)),
            1 => format!("(sqrt(abs({a})) + {b})"),
            _ => format!("(min({a}, {b}) * 2.0)"),
        }
    }

    fn store_out(&mut self, value: &str) {
        let ix = self.uidx();
        self.line(format!("out[{ix}] = {value};"));
    }

    fn statement(&mut self, depth: usize) {
        match self.rng.below(12) {
            0 | 1 => {
                let v = self.fexpr(2);
                self.store_out(&v);
            }
            2 => {
                let v = self.uexpr(2);
                let ix = self.uidx();
                self.line(format!("cnt[{ix}] = {v};"));
            }
            3 => {
                let v = self.fexpr(2);
                let ix = self.uidx();
                self.line(format!("ys[{ix}] = {v};"));
            }
            4 => {
                let v = self.fexpr(2);
                self.line(format!("s = {v};"));
            }
            5 => {
                let v = self.uexpr(2);
                self.line(format!("acc = {v};"));
            }
            6 => {
                let v = self.fexpr(2);
                let v = self.nonconst(v);
                let l = self.fresh_let();
                self.line(format!("let {l} = {v};"));
            }
            7 if depth > 0 => {
                let c = self.bexpr();
                self.line(format!("if {c} {{"));
                self.indent += 1;
                self.scopes.push(Vec::new());
                let n = 1 + self.rng.below(2);
                for _ in 0..n {
                    self.statement(depth - 1);
                }
                self.scopes.pop();
                self.indent -= 1;
                self.line("} else {");
                self.indent += 1;
                self.scopes.push(Vec::new());
                self.statement(depth - 1);
                self.scopes.pop();
                self.indent -= 1;
                self.line("}");
            }
            8 if depth > 0 && self.in_loop == 0 => {
                let k = 1 + self.rng.below(4);
                self.line(format!("for (var j = 0u; j < {k}u; j = j + 1u) {{"));
                self.indent += 1;
                self.in_loop += 1;
                self.scopes.push(Vec::new());
                let n = 1 + self.rng.below(3);
                for _ in 0..n {
                    self.statement(depth - 1);
                }
                self.scopes.pop();
                self.in_loop -= 1;
                self.indent -= 1;
                self.line("}");
            }
            9 => {
                let v = self.fexpr(1);
                let k = self.uexpr(1);
                self.line(format!("stash({v}, {k});"));
            }
            10 => {
                let v = self.fexpr(1);
                self.line(format!("poke(&s, {v});"));
            }
            _ => {
                let a = self.fexpr(1);
                let b = self.fexpr(1);
                let l = self.fresh_let();
                self.line(format!("let {l} = pure2({a}, {b});"));
            }
        }
    }

    /// One repeat pattern, aimed at a class.
    fn pattern(&mut self) {
        match self.rng.below(7) {
            0 => {
                // Two sites, nothing stored between: a share.
                let e = self.repeat();
                self.store_out(&format!("{e} + 1.0"));
                self.store_out(&format!("{e} * 2.0"));
                self.intended.push("share");
            }
            1 => {
                // A store to s between two sites of a text that reads s.
                let e = format!("({} * s + 1.0)", self.fexpr(1));
                self.store_out(&e);
                self.line("s = s + 1.0;");
                self.store_out(&e);
                self.intended.push("lineage_differs");
            }
            2 => {
                // Inside a loop that stores s, and after it.
                let e = format!("(s * s + {})", self.fexpr(1));
                let k = 1 + self.rng.below(3);
                self.line(format!("for (var j = 0u; j < {k}u; j = j + 1u) {{"));
                self.indent += 1;
                self.in_loop += 1;
                self.store_out(&e);
                self.line("s = s + xs[(j & 7u)];");
                self.in_loop -= 1;
                self.indent -= 1;
                self.line("}");
                self.store_out(&e);
                self.intended.push("lineage_differs");
            }
            3 => {
                // Inside a loop that leaves its lineage alone: hoisted.
                let a = self.rng.below(8);
                let b = self.rng.below(8);
                let e = format!("(xs[{a}u] * xs[{b}u] + t)");
                let k = 1 + self.rng.below(3);
                self.line(format!("for (var j = 0u; j < {k}u; j = j + 1u) {{"));
                self.indent += 1;
                self.in_loop += 1;
                self.line(format!("out[(j & 7u)] = {e} + f32(j);"));
                self.line(format!("cnt[(j & 7u)] = u32({e});"));
                self.in_loop -= 1;
                self.indent -= 1;
                self.line("}");
                self.intended.push("hoist");
            }
            4 => {
                // A let bound in an arm, read there and (through a local)
                // after the join; the arm's repeat shares, the join's not.
                let c = self.bexpr();
                let v = self.fexpr(1);
                let v = self.nonconst(v);
                self.line(format!("if {c} {{"));
                self.indent += 1;
                self.line(format!("let w = {v};"));
                self.line("t = w * w + 1.0;");
                self.store_out("w * w + 1.0");
                self.indent -= 1;
                self.line("} else {");
                self.indent += 1;
                self.line("t = -2.0;");
                self.indent -= 1;
                self.line("}");
                self.store_out("t");
                self.intended.push("share");
            }
            5 => {
                // The step-5 let: bound from ys[acc], acc advanced, used after.
                let p = self.fresh_let();
                self.line(format!("let {p} = ys[(acc & 7u)];"));
                self.line("acc = acc + 1u;");
                self.store_out(&format!("{p} + ys[(acc & 7u)]"));
                self.intended.push("let_after_store");
            }
            _ => {
                // A text that is also a local's constant initialiser cannot
                // be placed: the initialiser runs before any statement.
                self.store_out("xs[1u] * 0.5 + 1.0");
                self.store_out("xs[1u] * 0.5 + 1.0");
                self.intended.push("share");
            }
        }
    }
}

/// Whether an expression text reads no variable (so naga folds it): every
/// identifier in it is a function name or a type.
fn is_const(e: &str) -> bool {
    // A call to the helper is never folded, so it counts as a variable.
    const NOT_VARIABLES: &[&str] = &["sqrt", "abs", "min", "max", "select", "clamp", "f32", "u32", "u", "e"];
    !e.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|tok| !tok.is_empty() && tok.chars().next().is_some_and(|c| c.is_ascii_alphabetic()))
        .any(|tok| !NOT_VARIABLES.contains(&tok))
}

/// Generate the program for `seed`.
pub fn generate(seed: u64) -> Generated {
    let mut rng = Rng::new(seed);
    let out_of_range = rng.chance(20);
    let mut xs = [0f32; 8];
    for x in &mut xs {
        *x = rng.f32_in(-4.0, 4.0);
    }
    if rng.chance(30) {
        xs[rng.below(8) as usize] = *rng.pick(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0]);
    }
    let mut ks = [0u32; 8];
    for k in &mut ks {
        *k = if rng.chance(15) { rng.next_u64() as u32 } else { rng.below(64) as u32 };
    }
    let mut ys = [0f32; 8];
    for y in &mut ys {
        *y = rng.f32_in(-2.0, 2.0);
    }
    let n = rng.below(10) as u32;
    let mut g = Gen { rng, lines: Vec::new(), scopes: vec![Vec::new()], in_loop: 0, next_let: 0, intended: Vec::new(), out_of_range, indent: 1 };
    if out_of_range {
        g.intended.push("out_of_range");
    }
    let s0 = g.rng.f32_in(-2.0, 2.0);
    let t0 = g.rng.f32_in(-2.0, 2.0);
    let acc0 = g.rng.below(8);
    g.line(format!("var s: f32 = {s0:.2};"));
    g.line(format!("var t: f32 = {t0:.2};"));
    g.line("var i: u32 = 0u;");
    g.line(format!("var acc: u32 = {acc0}u;"));
    let n_stmts = 4 + g.rng.below(6);
    let patterns = 1 + g.rng.below(3);
    let mut done_patterns = 0;
    for k in 0..n_stmts {
        if done_patterns < patterns && (k % 3 == 1) {
            g.pattern();
            done_patterns += 1;
        } else {
            g.statement(1);
        }
    }
    while done_patterns < patterns {
        g.pattern();
        done_patterns += 1;
    }
    g.line("cnt[7u] = acc + i;");
    let body = g.lines.join("\n");
    let source = format!(
        "@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read> ks: array<u32>;
@group(0) @binding(2) var<uniform> n: u32;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> cnt: array<u32>;
@group(0) @binding(5) var<storage, read_write> ys: array<f32>;
fn pure2(a: f32, b: f32) -> f32 {{ return a * b + 1.0; }}
fn stash(v: f32, k: u32) {{ ys[(k & 7u)] = v; }}
fn poke(p: ptr<function, f32>, by: f32) {{ *p = *p + by; }}
@compute @workgroup_size(1)
fn main() {{
{body}
}}
"
    );
    Generated { seed, source, intended: g.intended, xs, ks, ys, n }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::chromosome::{chromosome, ChromosomeOptions};
    use crate::wgsl::reader::read;

    #[test]
    fn a_let_initialiser_is_never_a_constant_expression() {
        assert!(is_const("f32(5u)"));
        assert!(is_const("abs(1.42)"));
        assert!(is_const("-0.45"));
        assert!(!is_const("(5.0 + t)"));
        assert!(!is_const("xs[(acc & 7u)]"));
        assert!(!is_const("pure2(l1, 2.0)"));
        assert!(!is_const("pure2(2.08, 1.33)"), "a call is not folded");
        for seed in 1..=200 {
            let g = generate(seed);
            for line in g.source.lines() {
                if let Some(rest) = line.trim().strip_prefix("let ") {
                    let init = rest.split_once('=').map(|(_, v)| v.trim().trim_end_matches(';')).unwrap_or("");
                    assert!(!is_const(init), "seed {seed}: {line}");
                }
            }
        }
    }

    #[test]
    fn generated_programs_parse_validate_read_and_fold_and_the_classes_are_reached() {
        let mut reached: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
        let mut intended: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
        for seed in 1..=40 {
            let g = generate(seed);
            assert_eq!(generate(seed).source, g.source, "deterministic");
            let k = read(&g.source).unwrap_or_else(|e| panic!("seed {seed}: {e}\n{}", g.source));
            for i in &g.intended {
                *intended.entry(i).or_default() += 1;
            }
            for f in &k.functions {
                let c = chromosome(f, &ChromosomeOptions::default()).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
                if f.entry_point {
                    if c.folded.filled > 0 {
                        *reached.entry("share").or_default() += 1;
                    }
                    for (r, n) in &c.refused {
                        *reached.entry(match r.as_str() {
                            "lineage_differs" => "lineage_differs",
                            "operand_unavailable" => "operand_unavailable",
                            "no_dominating_point" => "no_dominating_point",
                            "version_not_current" => "version_not_current",
                            _ => "other",
                        }).or_default() += n;
                    }
                }
            }
        }
        assert!(reached.get("share").copied().unwrap_or(0) >= 10, "{reached:?}");
        assert!(reached.get("lineage_differs").copied().unwrap_or(0) >= 5, "{reached:?}");
        assert!(intended.len() >= 4, "{intended:?}");
    }
}
