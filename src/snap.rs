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
