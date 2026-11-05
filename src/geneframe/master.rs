//! THE ONE SET OF TABLES EVERY KINGDOM LIVES IN (`kingdoms/master/`; design `docs/DESIGN_geneframe.md` 3.3, plan
//! `docs/PLAN_futhark_kingdom.md` 7.7 task K0; Andrew Morgan, 2026-11-05: "all kingdoms exist in a single set of
//! tables, with a master table having a list of kingdoms that we select to filter the rest of the data").
//!
//! `kingdoms.tsv` lists the kingdoms and their variants (`tsr.order_depth` under `tsr`, `futhark.kmeans` under
//! `futhark`); every other table is one schema entity with a `kingdom` column ([`TABLES`]), and the Property catalogue
//! (`properties.tsv`) is shared by every kingdom by design (3.3.1), so it has none. [`select`] filters every table to one
//! kingdom, renders the rows in the NAMED TEXT FORM ([`Selection::named_text`], byte for byte what
//! `kingdoms/master/write_rows.py` renders) and [`Selection::load`] hands that text to
//! [`KingdomTables::parse_named`], so the schema, the check and the decoder are untouched: only where the rows are
//! stored and how they are selected changed. Every kingdom's rows are written by its generator through
//! `write_rows.py`, never by hand; a kingdom's test compares the committed rows with a fresh generation.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use super::schema::KingdomTables;
use super::{Entry, Ty};

fn bug(what: String) -> String {
    format!("our software has a bug: kingdoms/master: {what}")
}

/// One row of `kingdoms.tsv`: a kingdom, or a variant of its parent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KingdomRow {
    pub name: String,
    /// Empty for a kingdom; the kingdom a variant belongs to.
    pub parent: String,
    pub description: String,
    /// The root data type's name (the entry's result); empty for a parent.
    pub root: String,
    pub notes: String,
}

/// The entity tables, in the order the named text form lists their rows: file, columns (the first is `kingdom`), text.
pub const TABLES: [(&str, &[&str], &str); 8] = [
    ("data_types.tsv", &["kingdom", "name", "kind", "members", "result", "properties", "constant", "constant_properties", "size_names"], include_str!("../../kingdoms/master/data_types.tsv")),
    ("symbols.tsv", &["kingdom", "name", "alias", "role", "adapter", "prelude"], include_str!("../../kingdoms/master/symbols.tsv")),
    ("spellings.tsv", &["kingdom", "symbol", "form", "template"], include_str!("../../kingdoms/master/spellings.tsv")),
    ("arguments.tsv", &["kingdom", "symbol", "position", "direction", "data_type", "head"], include_str!("../../kingdoms/master/arguments.tsv")),
    ("flows.tsv", &["kingdom", "symbol", "position", "direction", "name", "formula"], include_str!("../../kingdoms/master/flows.tsv")),
    ("guards.tsv", &["kingdom", "symbol", "order", "strength", "condition"], include_str!("../../kingdoms/master/guards.tsv")),
    ("repairs.tsv", &["kingdom", "symbol", "order", "site", "when", "guard", "template"], include_str!("../../kingdoms/master/repairs.tsv")),
    ("entries.tsv", &["kingdom", "problem", "parameter", "data_type", "properties", "form_of", "transform", "transform_properties", "transform_cost"], include_str!("../../kingdoms/master/entries.tsv")),
];

/// `kingdoms.tsv`'s text and columns.
pub const KINGDOMS_TSV: &str = include_str!("../../kingdoms/master/kingdoms.tsv");
const KINGDOMS_COLUMNS: [&str; 5] = ["name", "parent", "description", "root", "notes"];

/// The owner of a Flow or Guard row that is the entry's, and the entries row that is the result.
const ENTRY: &str = "entry";
const RESULT: &str = "out";

/// Every table, read: the kingdom list and each entity table's rows (the kingdom column kept).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Master {
    pub kingdoms: Vec<KingdomRow>,
    tables: BTreeMap<&'static str, Vec<Vec<String>>>,
}

/// One row's fields after a header check: as many fields as columns, or the bug.
fn read_table(file: &str, columns: &[&str], text: &str) -> Result<Vec<Vec<String>>, String> {
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
    if header != columns {
        return Err(bug(format!("{file}: the header is {header:?}, expected {columns:?}")));
    }
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        let f: Vec<String> = line.split('\t').map(str::to_string).collect();
        if f.len() != columns.len() {
            return Err(bug(format!("{file}:{}: {} fields, expected {}", i + 2, f.len(), columns.len())));
        }
        rows.push(f);
    }
    Ok(rows)
}

impl Master {
    /// Read every table from its text: `kingdoms.tsv`, then each of [`TABLES`] (file, columns, text). Every row's
    /// kingdom must be listed and every variant's parent too.
    pub fn parse(kingdoms_tsv: &str, tables: &[(&str, &[&str], &str)]) -> Result<Master, String> {
        let kingdoms: Vec<KingdomRow> = read_table("kingdoms.tsv", &KINGDOMS_COLUMNS, kingdoms_tsv)?
            .into_iter()
            .map(|f| KingdomRow { name: f[0].clone(), parent: f[1].clone(), description: f[2].clone(), root: f[3].clone(), notes: f[4].clone() })
            .collect();
        for (i, k) in kingdoms.iter().enumerate() {
            if k.name.is_empty() || kingdoms[..i].iter().any(|o| o.name == k.name) {
                return Err(bug(format!("kingdoms.tsv lists {:?} twice, or an empty name", k.name)));
            }
            if !k.parent.is_empty() && !kingdoms.iter().any(|o| o.name == k.parent) {
                return Err(bug(format!("kingdoms.tsv: {} names the parent {:?}, which is not listed", k.name, k.parent)));
            }
        }
        let mut out = BTreeMap::new();
        for (file, columns, text) in tables {
            let known = TABLES.iter().find(|t| t.0 == *file).ok_or_else(|| bug(format!("{file} is not one of the entity tables")))?;
            if known.1 != *columns {
                return Err(bug(format!("{file}: the columns {columns:?} are not the table's {:?}", known.1)));
            }
            let rows = read_table(file, columns, text)?;
            for r in &rows {
                if !kingdoms.iter().any(|k| k.name == r[0]) {
                    return Err(bug(format!("{file}: a row of the kingdom {:?}, which kingdoms.tsv does not list", r[0])));
                }
            }
            out.insert(known.0, rows);
        }
        for (file, ..) in TABLES {
            if !out.contains_key(file) {
                return Err(bug(format!("{file} was not given")));
            }
        }
        Ok(Master { kingdoms, tables: out })
    }

    /// THE COMMITTED TABLES, read once (a malformed table is our software's bug and stops the process, as the property
    /// catalogue does).
    pub fn committed() -> &'static Master {
        static MASTER: OnceLock<Master> = OnceLock::new();
        MASTER.get_or_init(|| Master::parse(KINGDOMS_TSV, &TABLES).unwrap_or_else(|e| panic!("{e}")))
    }

    /// The row of a kingdom or variant.
    pub fn kingdom(&self, name: &str) -> Option<&KingdomRow> {
        self.kingdoms.iter().find(|k| k.name == name)
    }

    /// The variants of a kingdom (the rows whose parent it is), in `kingdoms.tsv`'s order.
    pub fn variants(&self, parent: &str) -> Vec<&KingdomRow> {
        self.kingdoms.iter().filter(|k| k.parent == parent).collect()
    }

    /// A table's rows (the kingdom column kept), by file name.
    pub fn table(&self, file: &str) -> &[Vec<String>] {
        self.tables.get(file).map_or(&[], Vec::as_slice)
    }

    /// SELECT ONE KINGDOM: every table filtered to its rows, in the order written, the kingdom column dropped.
    pub fn select(&self, kingdom: &str) -> Result<Selection, String> {
        let row = self.kingdom(kingdom).ok_or_else(|| bug(format!("{kingdom:?} is not a kingdom of kingdoms.tsv ({})", self.kingdoms.iter().map(|k| k.name.as_str()).collect::<Vec<_>>().join(", "))))?;
        let rows = self.tables.iter().map(|(file, rows)| (*file, rows.iter().filter(|r| r[0] == kingdom).map(|r| r[1..].to_vec()).collect())).collect();
        Ok(Selection { kingdom: row.clone(), rows })
    }
}

/// One kingdom's rows, selected from the master.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub kingdom: KingdomRow,
    rows: BTreeMap<&'static str, Vec<Vec<String>>>,
}

impl Selection {
    /// The kingdom's rows of one table (without the kingdom column), by file name.
    pub fn rows(&self, file: &str) -> &[Vec<String>] {
        self.rows.get(file).map_or(&[], Vec::as_slice)
    }

    /// Whether the kingdom has rows of its own (a parent has none; its variants do).
    pub fn is_empty(&self) -> bool {
        self.rows.values().all(Vec::is_empty)
    }

    /// THE NAMED TEXT FORM of the rows ([`KingdomTables::parse_named`] reads it): data types each followed by its
    /// carries, constant and size names; symbols each followed by its spellings; arguments; the symbols' flows; the
    /// symbols' guards each followed by its repairs; the entry and its parameters; the entry's flows; the entry's guards
    /// with their repairs. Byte for byte `write_rows.py`'s `named_of_rows`.
    pub fn named_text(&self) -> Result<String, String> {
        let k = &self.kingdom.name;
        let mut out: Vec<String> = Vec::new();
        for d in self.rows("data_types.tsv") {
            let (name, kind, members, result, props, constant, cprops, sizes) = (&d[0], &d[1], &d[2], &d[3], &d[4], &d[5], &d[6], &d[7]);
            out.push(match kind.as_str() {
                "scalar" => format!("data_type\t{name}\tscalar"),
                "array" | "tuple" => format!("data_type\t{name}\t{kind}\t{members}"),
                "function" => format!("data_type\t{name}\tfunction\t{members}\t{result}"),
                other => return Err(bug(format!("{k}: the data type {name} is of the kind {other:?}"))),
            });
            out.extend(props.split(';').filter(|p| !p.is_empty()).map(|p| format!("carries\t{name}\t{p}")));
            if !constant.is_empty() {
                out.push(format!("constant\t{name}\t{constant}\t{cprops}"));
            }
            for item in sizes.split(';').filter(|s| !s.is_empty()) {
                let (n, lb) = item.rsplit_once(':').ok_or_else(|| bug(format!("{k}: the size name {item:?} of {name} is not name:bound")))?;
                out.push(format!("size_name\t{name}\t{n}\t{lb}"));
            }
        }
        let joined = |kind: &str, f: &[String]| format!("{kind}\t{}", f.join("\t"));
        for s in self.rows("symbols.tsv") {
            out.push(joined("symbol", s));
            out.extend(self.rows("spellings.tsv").iter().filter(|sp| sp[0] == s[0]).map(|sp| joined("spelling", sp)));
        }
        out.extend(self.rows("arguments.tsv").iter().map(|a| joined("argument", a)));
        let guards = |entry: bool, out: &mut Vec<String>| {
            for g in self.rows("guards.tsv").iter().filter(|g| (g[0] == ENTRY) == entry) {
                out.push(joined("guard", g));
                out.extend(self.rows("repairs.tsv").iter().filter(|r| r[0] == g[0] && r[1] == g[1]).map(|r| joined("repair", r)));
            }
        };
        out.extend(self.rows("flows.tsv").iter().filter(|f| f[0] != ENTRY).map(|f| joined("flow", f)));
        guards(false, &mut out);
        let entries = self.rows("entries.tsv");
        let result: Vec<&Vec<String>> = entries.iter().filter(|e| e[1] == RESULT).collect();
        let [result] = result.as_slice() else { return Err(bug(format!("{k}: {} result rows ({RESULT}) in its entries, not one", result.len()))) };
        out.push(format!("entry\t{}\t{}", result[0], result[2]));
        out.extend(entries.iter().filter(|e| e[1] != RESULT).map(|e| joined("parameter", &e[1..])));
        out.extend(self.rows("flows.tsv").iter().filter(|f| f[0] == ENTRY).map(|f| joined("flow", f)));
        guards(true, &mut out);
        let mut text = out.join("\n");
        text.push('\n');
        Ok(text)
    }

    /// LOAD THE KINGDOM: its tables and entry, read from the named text by [`KingdomTables::parse_named`] (data type `i`
    /// is `ty(i)`). The caller checks the tables and builds the contract and the decoder as before.
    pub fn load(&self, ty: &dyn Fn(usize) -> Result<Ty, String>) -> Result<(KingdomTables, Entry), String> {
        KingdomTables::parse_named(&self.kingdom.name, &self.named_text()?, ty)
    }
}

/// Select a kingdom (or a variant, `parent.variant`) from the committed tables.
pub fn select(kingdom: &str) -> Result<Selection, String> {
    Master::committed().select(kingdom)
}

/// Select a variant of a kingdom: Futhark's problem view (`view("futhark", "kmeans")`), TSR's variant.
pub fn view(kingdom: &str, variant: &str) -> Result<Selection, String> {
    select(&format!("{kingdom}.{variant}"))
}

/// Load a kingdom from the committed tables: [`select`] then [`Selection::load`].
pub fn load(kingdom: &str, ty: &dyn Fn(usize) -> Result<Ty, String>) -> Result<(KingdomTables, Entry), String> {
    select(kingdom)?.load(ty)
}

#[cfg(test)]
mod tests;
