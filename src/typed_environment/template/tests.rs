//! Tests of the template engine's closed placeholder list (design 8.14.2).

use super::*;

fn t(text: &str) -> Template {
    parse_template(text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn p(text: &str) -> PropertyExpr {
    PropertyExpr::parse(text).expect(text)
}

/// A kingdom's lexicon as Futhark's rows spell it: `min` as `i64.min a b`, a compound operand grouped in parentheses,
/// and in the `name` form `min_a_b` with no grouping; one adapter `fit` of a size and an array.
fn lexicon() -> Lexicon {
    let mut lex = Lexicon::default();
    for (op, form, text) in [("min", "", "i64.min {0} {1}"), ("+", "", "{0} + {1}"), ("group", "", "({0})"), ("min", "name", "min_{0}_{1}"), ("group", "name", "{0}"), ("negative", "name", "m{0}")] {
        lex.formulas.insert((op.to_string(), form.to_string()), t(text));
    }
    lex.adapters.insert("fit".to_string(), (t("fit {0} 0f32 ({1})"), "def fit".to_string()));
    lex
}

/// The node `map2 (+) {0} {1}` with operands `xs` (length k) and `ys` (length n), and a parameter.
fn render(text: &str) -> String {
    let spelling = t("map2 (+) {0} {1}");
    let values = |r: &Reference| match r {
        Reference::Input { position: 0, property: "len" } => Some(p("k")),
        Reference::Input { position: 1, property: "len" } => Some(p("n")),
        Reference::Input { position: 2, property: "len" } => Some(PropertyExpr::Unknown),
        Reference::Input { position: 0, property: "access" } => Some(PropertyExpr::Tag("read".into())),
        _ => None,
    };
    let bounds = SizeBounds::from([("k".to_string(), 1), ("n".to_string(), 1)]);
    let params = [p("3")];
    let lex = lexicon();
    let mut scope = Scope::new(&values, &bounds, &lex);
    scope.operands = vec!["xs".into(), "ys".into(), "zs".into()];
    scope.own = Some(1);
    scope.forms = BTreeMap::from([((1, "sg3".to_string()), "<1@sg3>".to_string())]);
    scope.bound = vec![vec![], vec!["i_n".to_string()]];
    scope.spelling = Some(&spelling);
    scope.params = &params;
    scope.zero = Some("0f32".into());
    scope.constant = Some("[0f32]".into());
    scope.render(&t(text), &mut Fresh::default()).unwrap_or_else(|e| panic!("{text}: {e}"))
}

#[test]
fn every_placeholder_renders() {
    assert_eq!(render("{0} and {1}"), "xs and ys");
    assert_eq!(render("{1@sg3}"), "<1@sg3>");
    assert_eq!(render("{1.v0}"), "i_n");
    assert_eq!(render("{=min(in0.len, in1.len)}"), "i64.min k n", "a formula is spelled by the kingdom's rows");
    assert_eq!(render("{=min(in0.len, in1.len)@name}"), "min_k_n", "in the name form");
    assert_eq!(render("{=in0.access}"), "read", "a tag renders as its name");
    assert_eq!(render("{p0} {=p0 + 1}"), "3 4");
    assert_eq!(render("{zero} {const}"), "0f32 [0f32]");
    assert_eq!(render("{in0} {in.own}"), "xs ys");
    assert_eq!(render("{self}"), "map2 (+) xs ys");
    assert_eq!(render("{self(0 := {call fit({=min(in0.len, in1.len)}, {0})}, 1 := {call fit({=k}, {1})})}"), "map2 (+) fit (i64.min k n) 0f32 (xs) fit k 0f32 (ys)", "a call's formula argument is an operand: grouped when compound");
    assert_eq!(render("{if known(in0.len) and known(in1.len) : static else run-time}"), "static");
    assert_eq!(render("{if known(in2.len) : static else run-time}"), "run-time");
    assert_eq!(render("{if 1 <= in0.len : yes}"), "yes", "1 <= k by k's lower bound");
    assert_eq!(render("vec{p0}<f32>({0}{for i < p0 - 1 : , 0.0})"), "vec3<f32>(xs, 0.0, 0.0)");
    assert_eq!(render("[{for i < 3 sep , : {=i * 2}}]"), "[0, 2, 4]");
    assert_eq!(render("{{literal}} {0}"), "{literal} xs");
    // A fresh name is one name within its template, and a new one in the next template.
    let spelling = t("{0}");
    let values = |_: &Reference| None;
    let bounds = SizeBounds::new();
    let lex = Lexicon::default();
    let mut scope = Scope::new(&values, &bounds, &lex);
    scope.operands = vec!["a".into()];
    scope.spelling = Some(&spelling);
    let mut fresh = Fresh::default();
    let ad = t("let {fresh ad_a} = {0} in {fresh ad_a}");
    assert_eq!(scope.render(&ad, &mut fresh).expect("one"), "let ad_a1 = a in ad_a1");
    assert_eq!(scope.render(&ad, &mut fresh).expect("two"), "let ad_a2 = a in ad_a2");
    assert!(scope.render(&t("{zero}"), &mut fresh).expect_err("no constant").starts_with("our software has a bug:"));
    assert!(scope.render(&t("{in.own}"), &mut fresh).expect_err("no own input").starts_with("our software has a bug:"));
}

/// FINDINGS 3 AND 4 of the code review: a formula operator with no formula row, and a `?` in a rendered value, are
/// bugs, never text the target language would reject.
#[test]
fn an_operator_with_no_row_and_a_question_mark_are_bugs() {
    let values = |r: &Reference| match r {
        Reference::Input { position: 0, property: "len" } => Some(PropertyExpr::Unknown),
        _ => None,
    };
    let bounds = SizeBounds::from([("k".to_string(), 1), ("n".to_string(), 1)]);
    let empty = Lexicon::default();
    let scope = Scope::new(&values, &bounds, &empty);
    let e = scope.render(&t("{=min(k, n)}"), &mut Fresh::default()).expect_err("no min row");
    assert!(e.contains("no formula row =min"), "{e}");
    let lex = lexicon();
    let scope = Scope::new(&values, &bounds, &lex);
    let e = scope.render(&t("{=in0.len}"), &mut Fresh::default()).expect_err("a ?");
    assert!(e.contains("the value ?"), "{e}");
    assert_eq!(scope.render(&t("{=min(k, n, 2)}"), &mut Fresh::default()).expect("min of three"), "i64.min k (i64.min n 2)", "an n-ary operator folds from the right, each compound operand grouped");
}

/// A repair's template EXECUTES over values as it renders: a call, an operand, a formula, and an `if` decided as the
/// renderer decides it; text is not executable.
#[test]
fn a_template_executes_its_calls() {
    struct Lens(Vec<i64>);
    impl Exec for Lens {
        type V = i64;
        fn operand(&mut self, k: u32) -> Result<i64, String> {
            Ok(self.0[k as usize])
        }
        fn value(&mut self, e: &PropertyExpr) -> Result<i64, String> {
            match e {
                PropertyExpr::Size(n) if n == "k" => Ok(3),
                PropertyExpr::Size(n) if n == "n" => Ok(5),
                PropertyExpr::Min(_) => Ok(3),
                other => Err(format!("{other}")),
            }
        }
        fn zero(&mut self) -> Result<i64, String> {
            Ok(0)
        }
        fn constant(&mut self) -> Result<i64, String> {
            Ok(1)
        }
        fn whole(&mut self) -> Result<i64, String> {
            Ok(-1)
        }
        fn call(&mut self, symbol: &str, args: Vec<i64>) -> Result<i64, String> {
            assert_eq!(symbol, "fit");
            Ok(args[0])
        }
    }
    let values = |r: &Reference| match r {
        Reference::Input { position: 0, property: "len" } => Some(p("k")),
        Reference::Input { position: 1, property: "len" } => Some(p("n")),
        _ => None,
    };
    let bounds = SizeBounds::from([("k".to_string(), 1), ("n".to_string(), 1)]);
    let lex = lexicon();
    let scope = Scope::new(&values, &bounds, &lex);
    let mut x = Lens(vec![3, 5]);
    let pair = t("{if in1.len = min(in0.len, in1.len) : {1} else {call fit({=min(in0.len, in1.len)}, {1})}}");
    assert_eq!(scope.exec(&pair, &mut x), Ok(3), "n is not the minimum: the call runs");
    let keep = t("{if in0.len = min(in0.len, in1.len) : {0} else {call fit({=min(in0.len, in1.len)}, {0})}}");
    assert_eq!(scope.exec(&keep, &mut x), Ok(3), "k is the minimum: the operand as it is");
    assert!(scope.exec(&t("fit {0}"), &mut x).is_err(), "text is not executable");
    assert_eq!(calls_of(&pair), vec!["fit".to_string()]);
}

#[test]
fn a_template_of_one_substitute_replaces() {
    assert_eq!(substitute_of(&t("{in0}")), Some(Substitute::Input(0)));
    assert_eq!(substitute_of(&t("{in.own}")), Some(Substitute::Own));
    assert_eq!(substitute_of(&t("{const}")), Some(Substitute::Const));
    assert_eq!(substitute_of(&t("({const})")), None, "text around it: a wrap");
    assert_eq!(substitute_of(&t("{self}")), None);
}

#[test]
fn malformed_templates_are_bugs() {
    for bad in ["{", "}", "{}", "{nope here}", "{fresh 1x}", "{self(0 fit)}", "{if n = k}", "{for i n : x}", "{=min(n)}", "{call fit}", "{1.x0}", "{=n@}"] {
        assert!(parse_template(bad).expect_err(bad).starts_with("our software has a bug:"), "{bad}");
    }
    assert_eq!(t("plain text"), vec![Piece::Text("plain text".into())]);
    assert_eq!(t(""), Vec::<Piece>::new());
}
