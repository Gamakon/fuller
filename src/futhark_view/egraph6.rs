//! MASTER 6's E-GRAPH (plan `docs/PLAN_futhark_kingdom.md` section 7, task F5): every master 6 view's e-graph, A7's
//! included, computed from the view's own tables and `kingdoms/futhark/master/6/rules.tsv`, byte for byte what
//! `gen_tables.py` (`egraph6_texts`) writes beside the view under `kingdoms/futhark/sig/<problem>/m6/`.
//!
//! * ONE SORT A DATA TYPE, `S<tag>`: the sort IS the data type.
//! * ONE CONSTRUCTOR A SYMBOL, `X<name>`, of its arguments' sorts; a `const` row takes its pool index (`i64`); a `var`
//!   row takes its variable's RESOLVED identity, the binding depth and the ordinal (`i64 i64`), never its index word
//!   (plan 7.5 risk 3), and has a second constructor `K<name>` of no argument for the reading with no variable of its
//!   data type in scope (the data type's constant).
//! * THE RULES: each rule of `rules.tsv` at every data type it is on (`elem`: the view's element scalars; `arr`: their
//!   rank-1 arrays; `mat`: their rank-2 arrays; `tuple`/`tuple3`: the pairs and triples), for the elements it is exact
//!   for, where the view has the row it `needs` and every row it names. `{op}` names `op@<tag>`; on a tuple `{fst}`,
//!   `{snd}`, `{thd}` name the projection `fst.<tag>@<component's tag>`; `{op@tag}` names that symbol exactly.
//! * `egraph.rows`: each symbol's constructors, their sort and what each reads (`node`, `dc`, `binder`, `constant`);
//!   `egraph.sorts`: each data type's sort.

use std::collections::{BTreeMap, BTreeSet};

use super::master6::View6;

/// Master 6's e-graph rules (`on`, `elems`, `needs`, `rule`).
pub const RULES6: &str = include_str!("../../kingdoms/futhark/master/6/rules.tsv");

/// The name of every master 6 view's ruleset.
pub const RULESET6: &str = "futhark_sig";

/// A view's four e-graph texts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Egraph6Texts {
    pub datatype: String,
    pub rules: String,
    pub rows: String,
    pub sorts: String,
}

impl Egraph6Texts {
    /// The texts by file name, in the order `gen_tables.py` writes them.
    pub fn artefacts(&self) -> [(&'static str, &str); 4] {
        [("egraph.datatype", self.datatype.as_str()), ("egraph.rules", self.rules.as_str()), ("egraph.rows", self.rows.as_str()), ("egraph.sorts", self.sorts.as_str())]
    }
}

fn bug(what: String) -> String {
    format!("our software has a bug: master 6's e-graph: {what}")
}

/// A symbol name or tag as an egglog identifier: letters and digits kept, `@` as `__`, anything else `_`.
pub fn egraph_name(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if c == '@' {
            out.push_str("__");
        } else {
            out.push('_');
        }
    }
    out
}

/// A symbol's constructor (`X<name>`) and, for a `var` row, the constructor of its constant reading (`K<name>`).
pub fn constructor(symbol: &str) -> String {
    format!("X{}", egraph_name(symbol))
}

/// The constructor of a `var` row's reading with no variable of its data type in scope.
pub fn constant_constructor(symbol: &str) -> String {
    format!("K{}", egraph_name(symbol))
}

/// A table's rows as maps from its header's columns.
fn table<'a>(text: &'a str, file: &str, columns: &[&str]) -> Result<Vec<BTreeMap<&'a str, &'a str>>, String> {
    let mut lines = text.lines().filter(|l| !l.is_empty());
    let head: Vec<&str> = lines.next().ok_or_else(|| bug(format!("{file} is empty")))?.split('\t').collect();
    if head != columns {
        return Err(bug(format!("{file}'s header is {head:?}, not {columns:?}")));
    }
    lines
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            if f.len() != head.len() {
                return Err(bug(format!("{file} row {l:?} has {} fields", f.len())));
            }
            Ok(head.iter().copied().zip(f).collect())
        })
        .collect()
}

/// The `{op}` placeholders of `rule`, in order.
fn placeholders(rule: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = rule;
    while let Some(i) = rest.find('{') {
        let after = &rest[i + 1..];
        match after.find(['{', '}']) {
            Some(j) if after.as_bytes()[j] == b'}' && after[..j].starts_with(|c: char| c.is_ascii_lowercase()) => {
                out.push(&after[..j]);
                rest = &after[j + 1..];
            }
            _ => rest = after,
        }
    }
    out
}

/// `rule` with its placeholders replaced, in order, by `ctors`.
fn instantiate(rule: &str, ctors: &[String]) -> String {
    let mut out = String::new();
    let mut rest = rule;
    let mut next = ctors.iter();
    while let Some(i) = rest.find('{') {
        let after = &rest[i + 1..];
        match after.find(['{', '}']) {
            Some(j) if after.as_bytes()[j] == b'}' && after[..j].starts_with(|c: char| c.is_ascii_lowercase()) => {
                out.push_str(&rest[..i]);
                out.push_str(next.next().map_or("", String::as_str));
                rest = &after[j + 1..];
            }
            _ => {
                out.push_str(&rest[..=i]);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// THE E-GRAPH OF A MASTER 6 VIEW: its four texts, as `gen_tables.py` writes them.
pub fn egraph6(view: &View6) -> Result<Egraph6Texts, String> {
    let dts: Vec<BTreeMap<&str, &str>> = view.types.iter().map(|[n, k, m, t]| BTreeMap::from([("name", n.as_str()), ("kind", k.as_str()), ("members", m.as_str()), ("tag", t.as_str())])).collect();
    let args_text: Vec<String> = view.symbols.iter().map(|s| s.3.join(";")).collect();
    let syms: Vec<BTreeMap<&str, &str>> = view.symbols.iter().zip(&args_text).map(|((n, k, o, _), a)| BTreeMap::from([("name", n.as_str()), ("kind", k.as_str()), ("out", o.as_str()), ("args", a.as_str())])).collect();
    let rules = table(RULES6, "rules.tsv", &["on", "elems", "needs", "rule"])?;
    let by_name: BTreeMap<&str, &BTreeMap<&str, &str>> = dts.iter().map(|t| (t["name"], t)).collect();
    let names: BTreeSet<&str> = syms.iter().map(|s| s["name"]).collect();
    let elements = &view.spec.elements;
    let is_element = |n: &str| elements.iter().any(|e| e == n);
    let sort = |dt: &str| -> Result<String, String> { Ok(format!("S{}", egraph_name(by_name.get(dt).ok_or_else(|| bug(format!("no data type {dt}")))?["tag"]))) };

    let mut data = String::from("; A master 6 view's e-graph datatype, generated by gen_tables.py.\n; One sort per data type: the sort IS the data type.\n\n");
    for t in &dts {
        data.push_str(&format!("(sort {})\n", sort(t["name"])?));
    }
    data.push('\n');
    let mut rows = String::from("# symbol\tconstructor\tsort\treads\n");
    for s in &syms {
        let (x, out) = (constructor(s["name"]), sort(s["out"])?);
        match s["kind"] {
            "const" => {
                data.push_str(&format!("(constructor {x} (i64) {out})\n"));
                rows.push_str(&format!("{}\t{x}\t{out}\tdc\n", s["name"]));
            }
            "var" => {
                let k = constant_constructor(s["name"]);
                data.push_str(&format!("(constructor {x} (i64 i64) {out})\n(constructor {k} () {out})\n"));
                rows.push_str(&format!("{}\t{x}\t{out}\tbinder\n{}\t{k}\t{out}\tconstant\n", s["name"], s["name"]));
            }
            _ => {
                let args = s["args"].split(';').filter(|a| !a.is_empty()).map(sort).collect::<Result<Vec<_>, _>>()?;
                data.push_str(&format!("(constructor {x} ({}) {out})\n", args.join(" ")));
                rows.push_str(&format!("{}\t{x}\t{out}\tnode\n", s["name"]));
            }
        }
    }
    let mut text = format!("; A master 6 view's e-graph ruleset, generated by gen_tables.py from master/6/rules.tsv.\n(ruleset {RULESET6})\n\n");
    for on in ["elem", "arr", "mat", "tuple", "tuple3"] {
        for t in &dts {
            let members: Vec<&str> = t["members"].split(';').filter(|m| !m.is_empty()).collect();
            let inner = |m: &str| by_name.get(m).filter(|d| d["kind"] == "array").map(|d| d["members"]);
            let exact: Vec<&str> = match (on, t["kind"]) {
                ("elem", "scalar") if is_element(t["name"]) => vec![t["name"]],
                ("arr", "array") if members.first().is_some_and(|m| is_element(m)) => members.clone(),
                ("mat", "array") if members.first().and_then(|m| inner(m)).is_some_and(is_element) => vec![members.first().and_then(|m| inner(m)).unwrap_or_default()],
                ("tuple", "tuple") if members.len() == 2 => members.clone(),
                ("tuple3", "tuple") if members.len() == 3 => members.clone(),
                _ => continue,
            };
            let mut mine = Vec::new();
            for r in rules.iter().filter(|r| r["on"] == on) {
                if !exact.iter().all(|e| r["elems"] == "*" || r["elems"].split(',').any(|x| x == *e)) {
                    continue;
                }
                if !r["needs"].is_empty() && !names.contains(format!("{}@{}", r["needs"], t["tag"]).as_str()) {
                    continue;
                }
                let named: Vec<String> = placeholders(r["rule"])
                    .into_iter()
                    .map(|op| {
                        if op.contains('@') {
                            op.to_string()
                        } else if t["kind"] == "tuple" && ["fst", "snd", "thd"].contains(&op) {
                            let i = ["fst", "snd", "thd"].iter().position(|p| *p == op).unwrap_or(0);
                            let comp = members.get(i).and_then(|c| by_name.get(c)).map_or("?", |c| c["tag"]);
                            format!("{op}.{}@{comp}", t["tag"])
                        } else {
                            format!("{op}@{}", t["tag"])
                        }
                    })
                    .collect();
                if named.iter().any(|n| !names.contains(n.as_str())) {
                    continue;
                }
                mine.push(instantiate(r["rule"], &named.iter().map(|n| constructor(n)).collect::<Vec<_>>()));
            }
            if !mine.is_empty() {
                text.push_str(&format!("; {}\n", t["name"]));
                for m in mine {
                    text.push_str(&m);
                    text.push('\n');
                }
                text.push('\n');
            }
        }
    }
    let mut sorts = String::from("# data type\tsort\n");
    for t in &dts {
        sorts.push_str(&format!("{}\t{}\n", t["name"], sort(t["name"])?));
    }
    Ok(Egraph6Texts { datatype: data, rules: text, rows, sorts })
}

macro_rules! egraph6 {
    ($p:literal) => {
        (
            $p,
            [
                include_str!(concat!("../../kingdoms/futhark/sig/", $p, "/m6/egraph.datatype")),
                include_str!(concat!("../../kingdoms/futhark/sig/", $p, "/m6/egraph.rules")),
                include_str!(concat!("../../kingdoms/futhark/sig/", $p, "/m6/egraph.rows")),
                include_str!(concat!("../../kingdoms/futhark/sig/", $p, "/m6/egraph.sorts")),
            ],
        )
    };
}

/// Every committed master 6 view's e-graph texts as `gen_tables.py` wrote them (datatype, rules, rows, sorts), in
/// [`super::master6::COMMITTED6`]'s order.
pub const COMMITTED_EGRAPH6: [(&str, [&str; 4]); 7] = [egraph6!("a7"), egraph6!("histogram"), egraph6!("kmeans"), egraph6!("kmeans_i32"), egraph6!("pathfinder"), egraph6!("reduce"), egraph6!("scan")];

#[cfg(test)]
mod tests {
    use super::*;

    /// EVERY VIEW'S E-GRAPH IS WHAT `gen_tables.py` WRITES, byte for byte: the Rust computation over the view's tables
    /// and the committed texts the generator's `--check` holds.
    #[test]
    fn every_master6_egraph_is_what_gen_tables_writes_byte_for_byte() {
        for (problem, committed) in COMMITTED_EGRAPH6 {
            let view = super::super::master6::committed(problem).unwrap();
            let texts = egraph6(&view).unwrap_or_else(|e| panic!("{problem}: {e}"));
            for ((name, got), want) in texts.artefacts().iter().zip(committed) {
                assert!(*got == want, "{problem}/m6/{name}: the Rust e-graph differs from gen_tables.py's:\n--- rust\n{got}\n--- committed\n{want}");
            }
        }
        let problems: Vec<&str> = COMMITTED_EGRAPH6.iter().map(|c| c.0).collect();
        let views: Vec<&str> = super::super::master6::COMMITTED6.iter().map(|c| c.0).collect();
        assert_eq!(problems, views, "every committed master 6 view has its e-graph");
    }

    #[test]
    fn a_placeholder_names_its_row_and_text_is_kept() {
        assert_eq!(placeholders("(rewrite ({add} a ({id@fn.x}) {B}) a)"), vec!["add", "id@fn.x"]);
        assert_eq!(instantiate("(rewrite ({add} a {B}) ({add} b a))", &["Xa".to_string(), "Xb".to_string()]), "(rewrite (Xa a {B}) (Xb b a))");
        assert_eq!(egraph_name("map2.add@a.f32"), "map2_add__a_f32");
    }
}
