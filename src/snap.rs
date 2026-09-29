//! Snap — clean a regression-fitted expression back toward math form.
//!
//! Two sound, problem-agnostic operations (both soundness-dial HIGH — they only
//! correct/rename within tolerance, never change behaviour, never need an
//! extrapolation gate):
//!
//!   1. CONSTANT SNAPPING. A numeric atom within `rel_tol` of a known constant
//!      (pi, e, sqrt2, 1/2, G, c, ...) is corrected to that constant's exact
//!      value, and recorded in a `{value -> symbol}` annotation map so the
//!      caller can render the symbol. The `Math` expression stays pure-numeric
//!      — fuller does not add a symbol node; rendering is the caller's job.
//!
//!   2. (Obvious algebraic rearrangement is handled by the existing denoise
//!      algebra ruleset — `x*1->x`, `sqrt(x^2)->|x|`, fold — not duplicated
//!      here. Run denoise before/after snap for the structural cleanup.)
//!
//! The library is CALLER-SUPPLIED (a list of `(symbol, value)`) so the crate
//! stays problem-agnostic; `default_constants()` provides the universally
//! agreed set. We deliberately do NOT ship hand-seeded combinations like
//! `4*pi^2/(G*M)` — snapping a fitted coefficient to a problem-specific
//! combination would be fitting to a known answer.

/// The universally-agreed constant library: mathematical constants and basic
/// rationals plus a few fundamental physical constants. Caller may extend or
/// replace it, but keep entries problem-AGNOSTIC.
pub fn default_constants() -> Vec<(&'static str, f64)> {
    vec![
        ("pi", std::f64::consts::PI),
        ("e", std::f64::consts::E),
        ("sqrt2", std::f64::consts::SQRT_2),
        ("tau", std::f64::consts::TAU),
        ("1/2", 0.5),
        ("1/3", 1.0 / 3.0),
        ("1/4", 0.25),
        ("2pi", std::f64::consts::TAU),
        ("G", 6.674_30e-11),       // gravitational constant
        ("c", 2.997_924_58e8),     // speed of light (m/s)
        ("k_e", 8.987_551_792_3e9),// Coulomb constant
    ]
}

/// Result of snapping an expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapped {
    /// The expression with numeric atoms corrected to exact constant values
    /// where they snapped (still a pure-numeric `Math` s-expression).
    pub expr: String,
    /// `value -> symbol` for each snapped constant, so the caller can render
    /// e.g. `3.141592653589793` as `pi`. Keyed by the exact corrected value's
    /// string form (matching what appears in `expr`).
    pub snapped: Vec<(String, String)>,
}

/// Snap the numeric atoms of `expr` to the nearest constant in `library` within
/// `rel_tol` (relative). Returns the corrected expression and the symbol map.
///
/// A `Num c` snaps to `(symbol, value)` iff `|c - value| <= rel_tol * |value|`
/// (and, for value 0, `|c| <= rel_tol`). The closest qualifying constant wins.
/// Signs are handled: `-3.1416` snaps to `-pi` (recorded as the negated value).
///
/// Tolerance note for callers: at the suggested `rel_tol = 1e-3` the capture
/// window around pi is ±0.0031 — e.g. a fitted `3.14` DOES snap to pi
/// (rel err ~5.1e-4). If your fitted coefficients legitimately live near
/// famous constants, tighten `rel_tol` accordingly.
pub fn snap(expr: &str, library: &[(&str, f64)], rel_tol: f64) -> Result<Snapped, String> {
    let mut tree = parse(expr).ok_or_else(|| format!("could not parse {expr:?}"))?;
    let mut snapped: Vec<(String, String)> = Vec::new();
    // Tree depth is bounded by the parse's depth cap, so this walk is safe.
    snap_node(&mut tree, library, rel_tol, &mut snapped);
    // De-duplicate the symbol map (same value may snap at several sites).
    snapped.sort();
    snapped.dedup();
    Ok(Snapped { expr: tree.to_math(), snapped })
}

/// Rational snap: rewrite every near-rational numeric atom `Num(v)` into the
/// exact ratio `(Div (Num p) (Num q))` for the smallest denominator `q` (2..=`max_q`)
/// with `|v - p/q| <= rel_tol * |v|`. Integers and atoms with no small rational
/// are left untouched.
///
/// WHY THIS IS ITS OWN SNAP, not a library entry for [`snap`]: constant snap
/// corrects a `Num`'s VALUE and records a symbol name, but the atom stays a
/// single float `Num` — and `to_infix` prints the decimal, the symbol being
/// metadata that never reaches the benchmark. SRBench's `round_floats` then
/// rounds that float to three decimals. A fitted `-0.0111111` that is really
/// `-1/90` therefore dies: its distributed coefficients round to `0.891/0.099`
/// instead of `0.9/0.1`. Emitting the RATIO STRUCTURE `(-1)/90` makes SymPy
/// carry it as an exact `Rational`, which `round_floats` leaves alone — so the
/// numerically-correct law is finally certified (proved on strogatz_shearflow2).
///
/// `snapped` records `value -> "p/q"` for each rewritten atom.
pub fn snap_rational(expr: &str, rel_tol: f64, max_q: i64) -> Result<Snapped, String> {
    let mut tree = parse(expr).ok_or_else(|| format!("could not parse {expr:?}"))?;
    let mut snapped: Vec<(String, String)> = Vec::new();
    snap_rational_node(&mut tree, rel_tol, max_q, &mut snapped);
    snapped.sort();
    snapped.dedup();
    Ok(Snapped { expr: tree.to_math(), snapped })
}

/// The smallest-denominator exact rational `(p, q)` (q in 2..=`max_q`) within
/// `rel_tol` of `v`, sign carried on `p`. None for integers, non-finite, or no
/// small rational.
fn best_rational(v: f64, rel_tol: f64, max_q: i64) -> Option<(i64, i64)> {
    if !v.is_finite() || v == 0.0 || v == v.trunc() {
        return None;
    }
    let a = v.abs();
    let mut best: Option<(i64, i64)> = None;
    for q in 2i64..=max_q {
        let p = (a * q as f64).round() as i64;
        if p == 0 {
            continue;
        }
        let rel = (p as f64 / q as f64 - a).abs() / a;
        if rel <= rel_tol && best.map(|(_, bq)| q < bq).unwrap_or(true) {
            best = Some((p, q));
        }
    }
    best.map(|(p, q)| (if v < 0.0 { -p } else { p }, q))
}

fn snap_rational_node(node: &mut Node, rel_tol: f64, max_q: i64, out: &mut Vec<(String, String)>) {
    match node {
        Node::Num(v) => {
            if let Some((p, q)) = best_rational(*v, rel_tol, max_q) {
                out.push((fmt_f64(*v), format!("{p}/{q}")));
                *node = Node::App("Div".to_string(), vec![Node::Num(p as f64), Node::Num(q as f64)]);
            }
        }
        Node::App(_, ch) => {
            for c in ch.iter_mut() {
                snap_rational_node(c, rel_tol, max_q, out);
            }
        }
        Node::Var(_) => {}
    }
}

fn snap_node(
    node: &mut Node,
    library: &[(&str, f64)],
    rel_tol: f64,
    out: &mut Vec<(String, String)>,
) {
    match node {
        Node::Num(v) => {
            if let Some((sym, exact)) = best_match(*v, library, rel_tol) {
                *v = exact; // correct the value to the constant's exact value
                out.push((fmt_f64(exact), sym));
            }
        }
        Node::App(_, ch) => {
            for c in ch.iter_mut() {
                snap_node(c, library, rel_tol, out);
            }
        }
        Node::Var(_) => {}
    }
}

/// Find the closest library constant to `v` within `rel_tol`, considering both
/// the constant and its negation. Returns `(symbol, exact_signed_value)`.
fn best_match(v: f64, library: &[(&str, f64)], rel_tol: f64) -> Option<(String, f64)> {
    if !v.is_finite() {
        return None;
    }
    let mut best: Option<(String, f64, f64)> = None; // (symbol, signed_value, rel_err)
    for &(sym, val) in library {
        for (signed, label) in [(val, sym.to_string()), (-val, format!("-{sym}"))] {
            let rel_err = if signed == 0.0 {
                v.abs()
            } else {
                (v - signed).abs() / signed.abs()
            };
            if rel_err <= rel_tol && best.as_ref().map(|b| rel_err < b.2).unwrap_or(true) {
                best = Some((label, signed, rel_err));
            }
        }
    }
    best.map(|(sym, val, _)| (sym, val))
}

// ---------------------------------------------------------------------------
// Math tree (parse / serialise) — local copy, kept independent of other modules
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Node {
    Num(f64),
    Var(String),
    App(String, Vec<Node>),
}

impl Node {
    fn to_math(&self) -> String {
        match self {
            Node::Num(v) => format!("(Num {})", fmt_f64(*v)),
            Node::Var(n) => format!("(Var \"{n}\")"),
            Node::App(op, ch) => {
                let parts: Vec<String> = ch.iter().map(Node::to_math).collect();
                format!("({op} {})", parts.join(" "))
            }
        }
    }
}

fn fmt_f64(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

fn parse(s: &str) -> Option<Node> {
    let toks = tok(s);
    let mut pos = 0;
    let n = pparse(&toks, &mut pos, 0)?;
    if pos == toks.len() { Some(n) } else { None }
}

fn tok(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '(' | ')' => { out.push(c.to_string()); chars.next(); }
            '"' => {
                let mut t = String::from("\"");
                chars.next();
                for c2 in chars.by_ref() { if c2 == '"' { break; } t.push(c2); }
                t.push('"');
                out.push(t);
            }
            c if c.is_whitespace() => { chars.next(); }
            _ => {
                let mut t = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2 == '(' || c2 == ')' || c2.is_whitespace() { break; }
                    t.push(c2);
                    chars.next();
                }
                out.push(t);
            }
        }
    }
    out
}

fn pparse(toks: &[String], pos: &mut usize, depth: usize) -> Option<Node> {
    // Depth cap: a pathologically nested input must fail the parse (the caller
    // returns an Err), not overflow the stack.
    if depth > crate::MAX_EXPR_DEPTH {
        return None;
    }
    if toks.get(*pos)? != "(" { return None; }
    *pos += 1;
    let head = toks.get(*pos)?.clone();
    *pos += 1;
    let node = match head.as_str() {
        "Num" => {
            let v: f64 = toks.get(*pos)?.parse().ok()?;
            // "inf"/"NaN" parse as f64 but render back to literals egglog
            // cannot read — refuse them.
            if !v.is_finite() { return None; }
            *pos += 1;
            Node::Num(v)
        }
        "Var" => { let n = toks.get(*pos)?.trim_matches('"').to_string(); *pos += 1; Node::Var(n) }
        ctor => {
            let mut ch = Vec::new();
            while *pos < toks.len() && toks[*pos] != ")" { ch.push(pparse(toks, pos, depth + 1)?); }
            Node::App(ctor.to_string(), ch)
        }
    };
    if toks.get(*pos)? != ")" { return None; }
    *pos += 1;
    Some(node)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snaps_pi_within_tolerance() {
        let lib = default_constants();
        // 3.1399 is within 1e-3 of pi -> corrected to exact pi, recorded.
        let r = snap(r#"(Mul (Num 3.1399) (Var "r"))"#, &lib, 1e-3).unwrap();
        assert!(r.expr.contains(&format!("{}", std::f64::consts::PI)), "{}", r.expr);
        assert!(r.snapped.iter().any(|(_, s)| s == "pi"));
    }

    #[test]
    fn does_not_snap_outside_tolerance() {
        let lib = default_constants();
        // 3.5 is not within 1e-3 of any constant -> unchanged, nothing recorded.
        let r = snap(r#"(Mul (Num 3.5) (Var "r"))"#, &lib, 1e-3).unwrap();
        assert_eq!(r.expr, r#"(Mul (Num 3.5) (Var "r"))"#);
        assert!(r.snapped.is_empty());
    }

    #[test]
    fn rational_snap_recovers_the_shearflow2_constant() {
        // The fitted leading constant of strogatz_shearflow2's reported form is
        // -0.011111110940650105, which IS -1/90. Rational snap must rewrite it to
        // the exact ratio (Div (Num -1) (Num 90)) so SymPy carries it symbolically
        // and SRBench's round_floats cannot degrade the distributed coefficients.
        let r = snap_rational(r#"(Mul (Num -0.011111110940650105) (Var "s"))"#, 1e-3, 128).unwrap();
        assert_eq!(r.expr, r#"(Mul (Div (Num -1.0) (Num 90.0)) (Var "s"))"#, "{}", r.expr);
        assert!(r.snapped.iter().any(|(_, s)| s == "-1/90"), "{:?}", r.snapped);
        // to_infix of the rewritten form prints the ratio, not a decimal — this
        // is what reaches SRBench and survives round_floats (int-emit: -1 not -1.0).
        let infix = crate::lint::node::Tree::parse(&r.expr).unwrap().to_infix();
        assert!(infix.contains("(-1)/90"), "{infix}");
    }

    #[test]
    fn rational_snap_leaves_integers_and_irrationals_alone() {
        // Integers are untouched; a value with no small rational is untouched.
        let r = snap_rational(r#"(Add (Num 3.0) (Num 0.31830988618))"#, 1e-3, 12).unwrap();
        // 1/pi = 0.3183... has no p/q with q<=12 within 1e-3, and 3 is an integer.
        assert_eq!(r.expr, r#"(Add (Num 3.0) (Num 0.31830988618))"#, "{}", r.expr);
        assert!(r.snapped.is_empty());
    }

    #[test]
    fn snaps_negative_constant() {
        let lib = default_constants();
        let r = snap(r#"(Num -2.71828)"#, &lib, 1e-3).unwrap();
        assert!(r.snapped.iter().any(|(_, s)| s == "-e"), "{:?}", r.snapped);
    }

    #[test]
    fn snaps_half() {
        let lib = default_constants();
        let r = snap(r#"(Mul (Num 0.4999) (Var "x"))"#, &lib, 1e-3).unwrap();
        assert!(r.snapped.iter().any(|(_, s)| s == "1/2"), "{:?}", r.snapped);
    }

    #[test]
    fn leaves_genuine_fitted_constant_alone() {
        let lib = default_constants();
        // -0.71 (the Study 1 coefficient) matches no universal constant -> kept.
        let r = snap(r#"(Mul (Num -0.71) (Var "x"))"#, &lib, 1e-3).unwrap();
        assert_eq!(r.expr, r#"(Mul (Num -0.71) (Var "x"))"#);
        assert!(r.snapped.is_empty(), "must not invent a constant for a real fitted value");
    }

    #[test]
    fn closest_constant_wins() {
        let lib = vec![("a", 1.0), ("b", 1.0005)];
        // 1.0004 is closer to b than a.
        let r = snap(r#"(Num 1.0004)"#, &lib, 1e-2).unwrap();
        assert!(r.snapped.iter().any(|(_, s)| s == "b"), "{:?}", r.snapped);
    }
}
