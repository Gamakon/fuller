//! Tests of the closed formula language (design section 3.4): each operator, its one spelling, the text round trip,
//! and the three-valued conditions.

use super::*;

fn p(t: &str) -> PropertyExpr {
    PropertyExpr::parse(t).unwrap_or_else(|e| panic!("{t}: {e}"))
}

fn c(t: &str) -> Condition {
    Condition::parse(t).unwrap_or_else(|e| panic!("{t}: {e}"))
}

fn ones(names: &[&str]) -> SizeBounds {
    names.iter().map(|n| (n.to_string(), 1)).collect()
}

/// Texts already in their one spelling read back to themselves, and each is its own normal form.
#[test]
fn every_operator_round_trips_through_its_text() {
    for text in [
        "n", "k_2", "3", "-1", "?", "in0.len", "in12.bound", "out.arm", "env.trips", "'read_write'", "true", "false",
        "in0.len + in1.len", "n + 1 + ?", "min(k, n)", "1 + min(k, n)", "min(?, in1.len, 1 + in0.len)", "max(in0.depth, in1.depth)",
        "2 * n", "n - k", "3 - k", "k * n", "-1 * j - k", "if(1 = in0.fam, in0.arm, in0.order)", "list(1, 2, 3)", "list()",
        "len(in0.cols)", "at(in0.shape, 0, 1)", "concat(in0.cols, in1.cols)", "find(in0.fields, in1.value)",
        "for(i < out.ndim : at(in0.shape, i, 1))", "in0.uniform and env.uniform", "min",
    ] {
        let e = p(text);
        assert_eq!(e.to_string(), text, "{text}");
        assert_eq!(e, e.normal(), "{text}");
        assert!(e.shape_defects().is_empty(), "{text}: {:?}", e.shape_defects());
    }
    for text in ["in1.len = in0.len", "in1.bound <= in0.len", "env.trips < 3", "known(in0.len) and known(in1.len)", "not in0.uniform", "x in list('a', 'b')", "all(i < n : at(in0.shape, i, 1) = 1)", "a = b or c <= d"] {
        assert_eq!(c(text).to_string(), text, "{text}");
    }
    for bad in ["", "in.len", "inx.len", "in0.", "a b", "n + ", "1x", "min(n)", "min(n, k", "min(n, k))", "'open", "and", "if(n, 1)", "len(a, b)", "at(a, 1)"] {
        assert!(PropertyExpr::parse(bad).expect_err(bad).starts_with("our software has a bug:"), "{bad}");
    }
}

/// The polynomial normal form: like monomials add, literals fold, products distribute, a zero term drops.
#[test]
fn sums_and_products_are_one_polynomial() {
    for (text, one) in [
        ("in0.len+n", "n + in0.len"),
        ("in1.len + in0.len", "in0.len + in1.len"),
        ("n + n", "2 * n"),
        ("n + 1 + 1", "n + 2"),
        ("n + 0", "n"),
        ("n - n", "0"),
        ("(n + 1) * 2", "2 + 2 * n"),
        ("(n + k) * (n - k)", "n * n - k * k"),
        ("k * n * 1", "k * n"),
        ("0 * n", "0"),
        ("n - 1 - 1", "n - 2"),
        ("? + ?", "? + ?"),
        ("min(n, k) + 1", "1 + min(k, n)"),
    ] {
        assert_eq!(p(text).to_string(), one, "{text}");
    }
    assert_eq!(p("a + b"), p("b + a"));
}

/// `min` and `max`: flattened, sorted, without repeats, one literal kept.
#[test]
fn min_and_max_keep_one_literal() {
    for (text, one) in [
        ("min(n, k)", "min(k, n)"),
        ("min(n, n)", "n"),
        ("min(n, min(k, n))", "min(k, n)"),
        ("min(1, 2, n)", "min(n, 1)"),
        ("max(1, 2, n)", "max(n, 2)"),
        ("max(max(a, b), c)", "max(a, b, c)"),
        ("max(3, 1)", "3"),
        ("min( n ,k )", "min(k, n)"),
    ] {
        assert_eq!(p(text).to_string(), one, "{text}");
    }
}

/// `if` takes a decided branch, keeps an undecided one, and is its branch when both are equal.
#[test]
fn if_takes_decided_branches() {
    assert_eq!(p("if(1 = 1, a, b)"), p("a"));
    assert_eq!(p("if(1 = 2, a, b)"), p("b"));
    assert_eq!(p("if('x' = 'y', a, b)"), p("b"));
    assert_eq!(p("if(n = k, a, a)"), p("a"));
    assert!(matches!(p("if(n = k, a, b)"), PropertyExpr::If { .. }));
    // `1 <= n` is decided only with n's lower bound.
    let e = p("if(1 <= n, a, b)");
    assert!(matches!(e, PropertyExpr::If { .. }));
    assert_eq!(e.normal_in(&ones(&["n"])), p("a"));
    // TSR's stream order (design 8.3): arm and ties of `a + b` read as rows, the order from out.arm and out.ties.
    let leaf = |r: &Reference| match r {
        Reference::Input { property, .. } => Some(PropertyExpr::Literal(match *property {
            "fam" => 0,
            _ => 1,
        })),
        Reference::Output(q) => Some(PropertyExpr::Literal(if *q == "arm" { 1 } else { 2 })),
        _ => None,
    };
    let order = p("if(2 <= out.ties, out.arm + 1, out.arm)");
    assert_eq!(order.substitute(&leaf, &SizeBounds::new()), PropertyExpr::Literal(2));
}

/// The list operators (host only): list, len, at with its default, concat, find, for.
#[test]
fn list_operators_evaluate_when_their_parts_are_literal() {
    assert_eq!(p("len(list(1, 2, 3))"), p("3"));
    assert_eq!(p("at(list(4, 5), 1, 0)"), p("5"));
    assert_eq!(p("at(list(4, 5), 2, 0)"), p("0"), "out of range: the default");
    assert_eq!(p("at(list(4, 5), -1, 9)"), p("9"));
    assert_eq!(p("concat(list(1), list(2, 3))"), p("list(1, 2, 3)"));
    assert_eq!(p("concat(list(1), ?)"), p("?"));
    assert_eq!(p("find(list('a', 'b'), 'b')"), p("1"));
    assert_eq!(p("find(list('a', 'b'), 'c')"), p("-1"));
    assert_eq!(p("find(list(n, 'b'), 'b')"), p("?"), "undecided on a ground list: ?");
    assert!(matches!(p("find(in0.fields, 'b')"), PropertyExpr::Find(..)), "a reference left: symbolic");
    assert_eq!(p("for(i < 3 : i * 2)"), p("list(0, 2, 4)"));
    assert!(matches!(p("for(i < n : i)"), PropertyExpr::For { .. }));
    // numpy's broadcast of shapes (design 8.10), with the shapes substituted.
    let shapes = |r: &Reference| match r {
        Reference::Input { position: 0, .. } => Some(p("list(3, 1)")),
        Reference::Input { position: 1, .. } => Some(p("list(4)")),
        Reference::Output(_) => Some(p("2")),
        _ => None,
    };
    let a = "at(in0.shape, i - out.ndim + len(in0.shape), 1)";
    let b = "at(in1.shape, i - out.ndim + len(in1.shape), 1)";
    let rule = p(&format!("for(i < out.ndim : if({a} = 1, {b}, {a}))"));
    assert_eq!(rule.substitute(&shapes, &SizeBounds::new()), p("list(3, 4)"));
    let legal = c(&format!("all(i < out.ndim : {a} = {b} or {a} = 1 or {b} = 1)"));
    assert_eq!(legal.substitute(&shapes, &SizeBounds::new()).truth(&SizeBounds::new()), Truth::Holds);
}

/// The three values (3.4.4): `?` makes a comparison undecided; `=` fails on two different literals; `<=` reads the
/// sizes' lower bounds; `and`, `or`, `not` are Kleene's.
#[test]
fn conditions_are_three_valued() {
    let b = ones(&["n", "k", "d"]);
    let t = |text: &str| c(text).truth(&b);
    use Truth::{Fails, Holds, Undecided};
    assert_eq!(t("n = n"), Holds);
    assert_eq!(t("n = k"), Undecided, "two size names: equal for some inputs");
    assert_eq!(t("1 = 2"), Fails);
    assert_eq!(t("'pl' = 'sg'"), Fails);
    assert_eq!(t("true = false"), Fails);
    assert_eq!(t("list(1, 2) = list(1, 2)"), Holds);
    assert_eq!(t("list(1, 2) = list(1, 3)"), Fails);
    assert_eq!(t("list(1) = list(1, 2)"), Fails);
    assert_eq!(t("? = ?"), Undecided);
    assert_eq!(t("min(n, k) = min(k, n)"), Holds);
    assert_eq!(t("n + k = k + n"), Holds);
    for len in ["n", "k", "1", "7", "min(k, n)", "min(d, k, n)", "n + k", "1 + min(k, n)", "max(n, 0)"] {
        assert_eq!(t(&format!("1 <= {len}")), Holds, "1 <= {len}");
        assert_eq!(t(&format!("{len} <= {len}")), Holds, "{len} <= {len}");
    }
    assert_eq!(t("1 <= 0"), Fails);
    assert_eq!(t("2 <= n"), Undecided);
    assert_eq!(t("1 <= n - 1"), Undecided);
    assert_eq!(t("min(k, n) <= n"), Holds);
    assert_eq!(t("n <= min(k, n)"), Undecided);
    assert_eq!(t("n <= max(k, n)"), Holds);
    assert_eq!(t("max(k, n) <= n"), Undecided);
    assert_eq!(t("1 < 2"), Holds);
    assert_eq!(t("2 < 2"), Fails);
    assert_eq!(t("n < n"), Fails);
    assert_eq!(t("0 < n"), Holds, "0 + 1 <= n by n's lower bound");
    assert_eq!(t("n < 3"), Undecided);
    assert_eq!(t("?  <= n"), Undecided);
    // The lower bounds are the table's, not a built-in n >= 1.
    assert_eq!(c("1 <= n").truth(&SizeBounds::new()), Undecided, "no SizeName row: no lower bound");
    assert_eq!(c("1 <= n").truth(&BTreeMap::from([("n".to_string(), 0)])), Undecided, "a size that may be 0");
    assert_eq!(c("1 <= n").truth(&BTreeMap::from([("n".to_string(), 2)])), Holds);
    // Kleene's connectives.
    assert_eq!(t("1 = 1 and n = k"), Undecided);
    assert_eq!(t("1 = 2 and n = k"), Fails);
    assert_eq!(t("1 = 1 or n = k"), Holds);
    assert_eq!(t("1 = 2 or n = k"), Undecided);
    assert_eq!(t("not n = k"), Undecided);
    assert_eq!(t("not 1 = 2"), Holds);
    // known, in, all, flags.
    assert_eq!(t("known(n + 1)"), Holds);
    assert_eq!(t("known(min(n, ?))"), Fails);
    assert_eq!(t("known(in0.len)"), Undecided, "a reference left");
    assert_eq!(t("'b' in list('a', 'b')"), Holds);
    assert_eq!(t("'c' in list('a', 'b')"), Fails);
    assert_eq!(t("n in list(k)"), Undecided);
    assert_eq!(t("'c' in ?"), Undecided);
    assert_eq!(t("all(i < 3 : i < 3)"), Holds);
    assert_eq!(t("all(i < 3 : i < 2)"), Fails);
    assert_eq!(t("all(i < n : i < 2)"), Undecided);
    assert_eq!(t("true"), Holds);
    assert_eq!(t("false"), Fails);
    assert_eq!(t("env.uniform"), Undecided, "a reference left");
    // A reference compared with itself is decided without its value.
    assert_eq!(t("in0.len = in0.len"), Holds);
    assert_eq!(t("in1.len = in0.len"), Undecided);
}

/// A condition substituted keeps its structure (a guard's outcome shows both sides), and its normal form decides it.
#[test]
fn substitution_keeps_a_conditions_structure() {
    let q = c("in1.len = in0.len");
    let vals = |r: &Reference| match r {
        Reference::Input { position: 0, .. } => Some(p("n")),
        Reference::Input { position: 1, .. } => Some(p("k")),
        _ => None,
    };
    let s = q.substitute(&vals, &SizeBounds::new());
    assert_eq!(s.comparison().map(|(r, a, b)| (r, a.clone(), b.clone())), Some((Relation::Eq, p("k"), p("n"))));
    assert_eq!(s.truth(&SizeBounds::new()), Truth::Undecided);
    assert_eq!(c("1 = 1 and x").normal_in(&SizeBounds::new()), c("x"));
    assert_eq!(c("1 = 2 and x").normal_in(&SizeBounds::new()).truth(&SizeBounds::new()), Truth::Fails);
    assert_eq!(c("b = a").normal_in(&SizeBounds::new()), c("a = b"));
    // A flag formula's condition: `uniform := env.uniform and in0.uniform`.
    let flag = p("env.uniform and in0.uniform");
    let env = |r: &Reference| match r {
        Reference::Env(_) => Some(PropertyExpr::Flag(true)),
        Reference::Input { .. } => Some(PropertyExpr::Flag(false)),
        _ => None,
    };
    assert_eq!(flag.substitute(&env, &SizeBounds::new()), PropertyExpr::Flag(false));
}

/// References: every one a formula reads, a comprehension's own index left out, an inner index shadowing.
#[test]
fn references_and_indices() {
    let e = p("for(i < len(in0.shape) : at(in0.shape, i, each) + out.n + env.trips)");
    let refs: Vec<String> = e.references().iter().map(|r| r.to_string()).collect();
    assert_eq!(refs, ["in0.shape", "out.n", "env.trips", "in0.shape", "each"], "in the order of the one spelling");
    assert_eq!(p("in0.len + in1.len").inputs(), vec![(0, "len"), (1, "len")]);
    assert!(p("n + 1").is_ground() && !p("in0.len").is_ground());
    let each = p("at(in0.cols, each, ?)");
    let at2 = each.substitute(&|r| if let Reference::Index("each") = r { Some(PropertyExpr::Literal(1)) } else { None }, &SizeBounds::new());
    assert_eq!(at2, p("at(in0.cols, 1, ?)"));
    // Hand-built values not in their one spelling are reported.
    let nested = PropertyExpr::Sum(vec![PropertyExpr::Sum(vec![p("k"), p("n")]), p("1")]);
    assert!(!nested.shape_defects().is_empty());
    assert!(PropertyExpr::Size("and".into()).shape_defects().iter().any(|d| d.contains("not a name")));
    assert_eq!(p("n").lower_bound(&ones(&["n"])), Some(1));
    assert_eq!(p("n * k + 2").lower_bound(&ones(&["n", "k"])), Some(3));
    assert_eq!(p("in0.len").lower_bound(&ones(&["n"])), None);
}
