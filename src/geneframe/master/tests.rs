//! The master tables: every kingdom selects, the Rust rendering of the named text form is byte for byte the Python
//! helper's, and the helper's own self-test (export all, import into empty tables, byte-identical) passes.

use super::*;

fn python(args: &[&str]) -> std::process::Output {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    std::process::Command::new("python3").arg("-I").arg(root.join("kingdoms/master/write_rows.py")).args(args).current_dir(root).output().expect("python3 runs write_rows.py")
}

/// Every kingdom of `kingdoms.tsv` selects; a parent has no rows, a variant has one result row and at least one data
/// type and symbol; every variant's named text parses as a kingdom table (with a family-free type numbering).
#[test]
fn every_kingdom_selects_and_every_variant_parses() {
    let m = Master::committed();
    assert!(m.kingdom("futhark").is_some() && m.kingdom("tsr").is_some());
    let mut variants = 0;
    for k in &m.kingdoms {
        let s = m.select(&k.name).unwrap();
        if k.parent.is_empty() {
            assert!(s.is_empty(), "{}: a parent has no rows of its own", k.name);
            assert!(!m.variants(&k.name).is_empty(), "{}: a parent has variants", k.name);
            continue;
        }
        variants += 1;
        assert!(!s.rows("data_types.tsv").is_empty() && !s.rows("symbols.tsv").is_empty(), "{}", k.name);
        let (tables, entry) = s.load(&|i| Ok(Ty::Futhark(i as u16))).unwrap_or_else(|e| panic!("{}: {e}", k.name));
        assert_eq!(tables.data_type(entry.output).map(|d| d.name.as_str()), Some(k.root.as_str()), "{}: its root is the entry's result", k.name);
    }
    assert_eq!(variants, 12, "7 Futhark views and 5 TSR variants");
    assert!(select("no_such_kingdom").is_err());
    assert_eq!(view("tsr", "order_depth").unwrap().kingdom.name, "tsr.order_depth");
}

/// THE TWO RENDERINGS AGREE: for every variant, `Selection::named_text` is byte for byte what `write_rows.py --named`
/// prints, so a generator's rows (written through the helper) load as the generator meant them.
#[test]
fn the_named_text_is_byte_for_byte_the_python_helpers() {
    for k in Master::committed().kingdoms.iter().filter(|k| !k.parent.is_empty()) {
        let out = python(&["--named", &k.name]);
        assert!(out.status.success(), "{}: {}", k.name, String::from_utf8_lossy(&out.stderr));
        let want = String::from_utf8(out.stdout).unwrap();
        let got = select(&k.name).unwrap().named_text().unwrap();
        if got != want {
            let (i, (a, b)) = got.lines().zip(want.lines()).enumerate().find(|(_, (a, b))| a != b).unwrap_or((0, ("", "")));
            panic!("{}: line {}: {a:?} (rust) vs {b:?} (python); lengths {} and {}", k.name, i + 1, got.len(), want.len());
        }
    }
}

/// THE HELPER'S SELF-TEST (plan 7.7 K0, export and import): export all, import into empty tables, byte-identical;
/// every kingdom's rows round trip through the named text form. A missing `python3` is a failure, not a skip.
#[test]
fn write_rows_selftest_passes() {
    let out = python(&["--selftest"]);
    assert!(out.status.success(), "{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
}

/// A table whose row names an unlisted kingdom, or a missing table, is the bug it is.
#[test]
fn an_unlisted_kingdom_or_a_missing_table_is_a_bug() {
    let kingdoms = "name\tparent\tdescription\troot\tnotes\nk\t\td\tF\t\n";
    let mut tables: Vec<(&str, &[&str], &str)> = TABLES.iter().map(|(f, c, _)| (*f, *c, "")).collect();
    let empty: Vec<String> = TABLES.iter().map(|(_, c, _)| c.join("\t") + "\n").collect();
    for (t, e) in tables.iter_mut().zip(&empty) {
        t.2 = e;
    }
    assert!(Master::parse(kingdoms, &tables).is_ok());
    let stray = format!("{}other\tF\tscalar\t\t\t\t\t\t\n", empty[0]);
    tables[0].2 = &stray;
    assert!(Master::parse(kingdoms, &tables).unwrap_err().contains("does not list"));
    assert!(Master::parse(kingdoms, &tables[1..]).unwrap_err().contains("was not given"));
}
