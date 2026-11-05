//! Tests of the template engine's closed placeholder list (design 8.14.2).

use super::*;

fn t(text: &str) -> Template {
    parse_template(text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn p(text: &str) -> PropertyExpr {
    PropertyExpr::parse(text).expect(text)
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
    let in_form = |k: u32, f: &str| Ok(format!("<{k}@{f}>"));
    let bounds = SizeBounds::from([("k".to_string(), 1), ("n".to_string(), 1)]);
    let params = [p("3")];
    let scope = Scope {
        operands: vec!["xs".into(), "ys".into(), "zs".into()],
        own: Some(1),
        in_form: &in_form,
        spelling: Some(&spelling),
        values: &values,
        params: &params,
        zero: Some("0f32".into()),
        constant: Some("[0f32]".into()),
        bounds: &bounds,
    };
    scope.render(&t(text), &mut Fresh::default()).unwrap_or_else(|e| panic!("{text}: {e}"))
}

#[test]
fn every_placeholder_renders() {
    assert_eq!(render("{0} and {1}"), "xs and ys");
    assert_eq!(render("{1@sg3}"), "<1@sg3>");
    assert_eq!(render("{=min(in0.len, in1.len)}"), "min(k, n)");
    assert_eq!(render("{=in0.access}"), "read", "a tag renders as its name");
    assert_eq!(render("{p0} {=p0 + 1}"), "3 4");
    assert_eq!(render("{zero} {const}"), "0f32 [0f32]");
    assert_eq!(render("{in0} {in.own}"), "xs ys");
    assert_eq!(render("{self}"), "map2 (+) xs ys");
    assert_eq!(render("{self(0 := fit ({=min(in0.len, in1.len)}) {zero} ({0}), 1 := fit k {zero} ({1}))}"), "map2 (+) fit (min(k, n)) 0f32 (xs) fit k 0f32 (ys)");
    assert_eq!(render("{if known(in0.len) and known(in1.len) : static else run-time}"), "static");
    assert_eq!(render("{if known(in2.len) : static else run-time}"), "run-time");
    assert_eq!(render("{if 1 <= in0.len : yes}"), "yes", "1 <= k by k's lower bound");
    assert_eq!(render("vec{p0}<f32>({0}{for i < p0 - 1 : , 0.0})"), "vec3<f32>(xs, 0.0, 0.0)");
    assert_eq!(render("[{for i < 3 sep , : {=i * 2}}]"), "[0, 2, 4]");
    assert_eq!(render("{{literal}} {0}"), "{literal} xs");
    // A fresh name is one name within its template, and a new one in the next template.
    let spelling = t("{0}");
    let values = |_: &Reference| None;
    let in_form = |_: u32, _: &str| Ok(String::new());
    let bounds = SizeBounds::new();
    let scope = Scope { operands: vec!["a".into()], own: None, in_form: &in_form, spelling: Some(&spelling), values: &values, params: &[], zero: None, constant: None, bounds: &bounds };
    let mut fresh = Fresh::default();
    let ad = t("let {fresh ad_a} = {0} in {fresh ad_a}");
    assert_eq!(scope.render(&ad, &mut fresh).expect("one"), "let ad_a1 = a in ad_a1");
    assert_eq!(scope.render(&ad, &mut fresh).expect("two"), "let ad_a2 = a in ad_a2");
    assert!(scope.render(&t("{zero}"), &mut fresh).expect_err("no constant").starts_with("our software has a bug:"));
    assert!(scope.render(&t("{in.own}"), &mut fresh).expect_err("no own input").starts_with("our software has a bug:"));
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
    for bad in ["{", "}", "{}", "{nope here}", "{fresh 1x}", "{self(0 fit)}", "{if n = k}", "{for i n : x}", "{=min(n)}"] {
        assert!(parse_template(bad).expect_err(bad).starts_with("our software has a bug:"), "{bad}");
    }
    assert_eq!(t("plain text"), vec![Piece::Text("plain text".into())]);
    assert_eq!(t(""), Vec::<Piece>::new());
}
