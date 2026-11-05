#!/usr/bin/env python3
"""THE ONE SET OF TABLES every kingdom lives in (`kingdoms/master/`; design `docs/DESIGN_geneframe.md` 3.3, plan
`docs/PLAN_futhark_kingdom.md` 7.7 task K0; Andrew Morgan, 2026-11-05: "all kingdoms exist in a single set of tables,
with a master table having a list of kingdoms that we select to filter the rest of the data").

`kingdoms.tsv` lists the kingdoms and their variants (a variant names its parent: `tsr.order_depth` under `tsr`,
`futhark.kmeans` under `futhark`). Every other table is ONE schema entity with a `kingdom` column, the physical model
of the gene frame's fixed schema (three lists that name things, four kinds of row):

| file | columns | entity |
|---|---|---|
| `data_types.tsv` | kingdom, name, kind, members, result, properties, constant, constant_properties, size_names | DataType (its carried properties, constant and size names are its columns) |
| `symbols.tsv` | kingdom, name, alias, role, adapter, prelude | Symbol |
| `spellings.tsv` | kingdom, symbol, form, template | Symbol's spellings (one a row: a template holds `;` and `=`) |
| `arguments.tsv` | kingdom, symbol, position, direction, data_type, head | Argument |
| `flows.tsv` | kingdom, symbol, position, direction, name, formula | Flow (`symbol` may be `entry`) |
| `guards.tsv` | kingdom, symbol, order, strength, condition | Guard (`symbol` may be `entry`) |
| `repairs.tsv` | kingdom, symbol, order, site, when, guard, template | Guard's repairs, in order |
| `entries.tsv` | kingdom, problem, parameter, data_type, properties, form_of, transform, transform_properties, transform_cost | Entry (the row `parameter = out` is the result) |
| `properties.tsv` | name, kind, element, values, root, meaning | the Property catalogue, SHARED by every kingdom by design (3.3.1), so it has no kingdom column and no generator writes it |

A row shared by two kingdoms is written once per kingdom (a variant holds every row it needs): the simpler rule, and
the only one under which a kingdom's symbol ids are the order of its own rows. Within a kingdom the rows keep the order
its generator wrote them in (data types and symbols are numbered by it); across kingdoms the blocks are sorted by
kingdom name, so two generators writing different kingdoms never clobber each other and the files are stable.

A generator produces the NAMED TEXT FORM (`KingdomTables::parse_named`: one row a line, its kind first) as before and
calls `write_kingdom` (replace one kingdom's rows, keep the rest) or `check_kingdom` (compare). `named_of_rows` is the
inverse, byte for byte what `src/geneframe/master.rs` renders for the loader, so the two sides are tested equal.

    python3 -I write_rows.py --named KINGDOM            # print a kingdom's named text form
    python3 -I write_rows.py --write KINGDOM [--parent P --description D --root R] < named.tsv
    python3 -I write_rows.py --check KINGDOM < named.tsv
    python3 -I write_rows.py --export [KINGDOM ...] > kingdoms.json   # all kingdoms when none is named
    python3 -I write_rows.py --import kingdoms.json [--into DIR]
    python3 -I write_rows.py --selftest                 # export all, import into empty tables: byte-identical
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

KINGDOMS_FILE = "kingdoms.tsv"
KINGDOMS_COLUMNS = ["name", "parent", "description", "root", "notes"]
PROPERTIES_FILE = "properties.tsv"

# The entity tables, in the order the named text form lists their rows. Every one's first column is `kingdom`.
TABLES = {
    "data_types.tsv": ["kingdom", "name", "kind", "members", "result", "properties", "constant", "constant_properties",
                       "size_names"],
    "symbols.tsv": ["kingdom", "name", "alias", "role", "adapter", "prelude"],
    "spellings.tsv": ["kingdom", "symbol", "form", "template"],
    "arguments.tsv": ["kingdom", "symbol", "position", "direction", "data_type", "head"],
    "flows.tsv": ["kingdom", "symbol", "position", "direction", "name", "formula"],
    "guards.tsv": ["kingdom", "symbol", "order", "strength", "condition"],
    "repairs.tsv": ["kingdom", "symbol", "order", "site", "when", "guard", "template"],
    "entries.tsv": ["kingdom", "problem", "parameter", "data_type", "properties", "form_of", "transform",
                    "transform_properties", "transform_cost"],
}
ENTRY = "entry"
RESULT = "out"


class RowsError(ValueError):
    """A table or a named text our software wrote wrongly: a bug, never a user error."""


def bug(what):
    return RowsError(f"our software has a bug: kingdoms/master: {what}")


# ------------------------------------------------------------------------------------------------------ reading ----

def read_table(path, columns):
    """A TSV with the given header: its rows, each a list of as many fields as columns. A missing file has no rows."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or lines[0].split("\t") != columns:
        raise bug(f"{path}: the header is {lines[0].split(chr(9)) if lines else None!r}, expected {columns}")
    rows = []
    for i, line in enumerate(lines[1:], start=2):
        f = line.split("\t")
        if len(f) != len(columns):
            raise bug(f"{path}:{i}: {len(f)} fields, expected {len(columns)}")
        rows.append(f)
    return rows


def read_master(master_dir=HERE):
    """Every table: the kingdoms list and, by file name, each entity table's rows (the kingdom column included)."""
    out = {KINGDOMS_FILE: read_table(os.path.join(master_dir, KINGDOMS_FILE), KINGDOMS_COLUMNS)}
    for fname, columns in TABLES.items():
        out[fname] = read_table(os.path.join(master_dir, fname), columns)
    return out


def kingdom_names(master):
    return [r[0] for r in master[KINGDOMS_FILE]]


def select(master, kingdom):
    """One kingdom's rows, by file name, WITHOUT the kingdom column, in the order written."""
    if kingdom not in kingdom_names(master):
        raise bug(f"{kingdom!r} is not a kingdom of {KINGDOMS_FILE} ({', '.join(kingdom_names(master))})")
    return {fname: [r[1:] for r in master[fname] if r[0] == kingdom] for fname in TABLES}


# ------------------------------------------------------------------------------------------------------ writing ----

def write_table(path, columns, rows):
    text = "\n".join(["\t".join(columns)] + ["\t".join(r) for r in rows]) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def merged(master, kingdom, rows, kingdom_row):
    """The master with `kingdom`'s rows replaced by `rows` (by file name, without the kingdom column) and its
    `kingdoms.tsv` row by `kingdom_row` (name, parent, description, root, notes), blocks sorted by kingdom name."""
    if kingdom_row[0] != kingdom:
        raise bug(f"the kingdom row names {kingdom_row[0]!r}, not {kingdom!r}")
    if len(kingdom_row) != len(KINGDOMS_COLUMNS):
        raise bug(f"the kingdom row has {len(kingdom_row)} fields, not {len(KINGDOMS_COLUMNS)}")
    for fname, columns in TABLES.items():
        for r in rows.get(fname, []):
            if len(r) != len(columns) - 1:
                raise bug(f"{fname}: a row of {kingdom} has {len(r)} fields, expected {len(columns) - 1}: {r!r}")
            if any("\t" in x or "\n" in x for x in r):
                raise bug(f"{fname}: a row of {kingdom} holds a tab or a line break: {r!r}")
    out = {}
    kept = [r for r in master[KINGDOMS_FILE] if r[0] != kingdom] + [list(kingdom_row)]
    out[KINGDOMS_FILE] = sorted(kept, key=lambda r: r[0])
    for fname in TABLES:
        kept = [r for r in master[fname] if r[0] != kingdom] + [[kingdom] + list(r) for r in rows.get(fname, [])]
        out[fname] = sorted(kept, key=lambda r: r[0])  # a stable sort: each kingdom's rows keep their order
    parents = {r[0] for r in out[KINGDOMS_FILE]}
    for r in out[KINGDOMS_FILE]:
        if r[1] and r[1] not in parents:
            raise bug(f"{r[0]} names the parent {r[1]!r}, which {KINGDOMS_FILE} does not list; write the parent first")
    for fname in TABLES:
        for r in out[fname]:
            if r[0] not in parents:
                raise bug(f"{fname}: a row of {r[0]!r}, which {KINGDOMS_FILE} does not list")
    return out


def write_master(master, master_dir=HERE):
    os.makedirs(master_dir, exist_ok=True)
    write_table(os.path.join(master_dir, KINGDOMS_FILE), KINGDOMS_COLUMNS, master[KINGDOMS_FILE])
    for fname, columns in TABLES.items():
        write_table(os.path.join(master_dir, fname), columns, master[fname])


def write_kingdom(kingdom, rows, kingdom_row, master_dir=HERE):
    """REPLACE one kingdom's rows (and its kingdoms.tsv row) and keep every other kingdom's, then write every table."""
    write_master(merged(read_master(master_dir), kingdom, rows, kingdom_row), master_dir)


def write_parent(name, description, master_dir=HERE, notes=""):
    """A parent kingdom's row (`futhark`, `tsr`): it has no rows of its own, its variants do."""
    write_kingdom(name, {}, (name, "", description, "", notes), master_dir)


def check_kingdom(kingdom, rows, kingdom_row, master_dir=HERE):
    """The differences between the committed tables and what `write_kingdom` would write: none when they are equal."""
    master = read_master(master_dir)
    want = merged(master, kingdom, rows, kingdom_row)
    bad = []
    for fname in [KINGDOMS_FILE] + list(TABLES):
        if master[fname] != want[fname]:
            got = [r for r in master[fname] if r[0] == kingdom]
            new = [r for r in want[fname] if r[0] == kingdom]
            where = next((f"row {i + 1} of {kingdom}: {a!r} vs {b!r}" for i, (a, b) in enumerate(zip(got, new)) if a != b),
                         f"{len(got)} rows committed, {len(new)} generated")
            bad.append(f"kingdoms/master/{fname} is not what the generator writes for {kingdom} ({where})")
    return bad


# --------------------------------------------------------------------------------- the named text form, both ways ----

def rows_of_named(text):
    """The named text form (one row a line, its kind first, data types and symbols by NAME) as entity rows by file
    name, without the kingdom column. Each line's fields are kept as written (`-` for none, as the form spells it)."""
    rows = {fname: [] for fname in TABLES}
    types = {}  # name -> index into rows["data_types.tsv"]
    problem = None
    for n, line in enumerate(text.split("\n"), start=1):
        if not line:
            continue
        f = line.split("\t")
        kind, rest = f[0], f[1:]
        where = f"line {n} of the named text ({line!r})"
        if kind == "data_type":
            if len(rest) < 2 or rest[0] in types:
                raise bug(f"{where} is not a data type declared once")
            name, dkind = rest[0], rest[1]
            if dkind == "scalar" and len(rest) == 2:
                members, result = "", ""
            elif dkind in ("array", "tuple") and len(rest) == 3:
                members, result = rest[2], ""
            elif dkind == "function" and len(rest) == 4:
                members, result = rest[2], rest[3]
            else:
                raise bug(f"{where} is not a data type of one of the four kinds")
            types[name] = len(rows["data_types.tsv"])
            rows["data_types.tsv"].append([name, dkind, members, result, "", "", "", ""])
        elif kind in ("carries", "constant", "size_name"):
            if not rest or rest[0] not in types:
                raise bug(f"{where} names no data type declared before it")
            row = rows["data_types.tsv"][types[rest[0]]]
            if kind == "carries" and len(rest) == 2:
                row[4] = ";".join([row[4], rest[1]] if row[4] else [rest[1]])
            elif kind == "constant" and len(rest) == 3 and not row[5]:
                row[5], row[6] = rest[1], rest[2]
            elif kind == "size_name" and len(rest) == 3:
                item = f"{rest[1]}:{rest[2]}"
                row[7] = ";".join([row[7], item] if row[7] else [item])
            else:
                raise bug(f"{where} is not a carries, constant (one a data type) or size_name line")
        elif kind == "entry":
            if len(rest) != 2 or problem is not None:
                raise bug(f"{where} is not the one entry line")
            problem = rest[0]
            rows["entries.tsv"].append([problem, RESULT, rest[1], "-", "-", "-", "-", "-"])
        elif kind == "parameter":
            if len(rest) != 7 or problem is None:
                raise bug(f"{where} is not a parameter line after the entry line")
            if rest[0] == RESULT:
                raise bug(f"{where} names a parameter {RESULT!r}, the result's name")
            rows["entries.tsv"].append([problem] + rest)
        else:
            fname = {"symbol": "symbols.tsv", "spelling": "spellings.tsv", "argument": "arguments.tsv",
                     "flow": "flows.tsv", "guard": "guards.tsv", "repair": "repairs.tsv"}.get(kind)
            if fname is None or len(rest) != len(TABLES[fname]) - 1:
                raise bug(f"{where} is not a row of the named text form")
            rows[fname].append(rest)
    if problem is None:
        raise bug("the named text has no entry line")
    return rows


def root_of(rows):
    """The root data type of a kingdom's rows: its entry's result (the `kingdoms.tsv` root column)."""
    return next((e[2] for e in rows["entries.tsv"] if e[1] == RESULT), "")


def named_of_rows(rows):
    """The named text form of one kingdom's rows (`select`'s result): the inverse of `rows_of_named`, and byte for
    byte what the Rust loader renders (`geneframe::master::Selection::named_text`)."""
    out = []
    for name, dkind, members, result, props, constant, cprops, sizes in rows["data_types.tsv"]:
        if dkind == "scalar":
            out.append(f"data_type\t{name}\tscalar")
        elif dkind in ("array", "tuple"):
            out.append(f"data_type\t{name}\t{dkind}\t{members}")
        elif dkind == "function":
            out.append(f"data_type\t{name}\tfunction\t{members}\t{result}")
        else:
            raise bug(f"the data type {name} is of the kind {dkind!r}")
        out += [f"carries\t{name}\t{p}" for p in props.split(";") if p]
        if constant:
            out.append(f"constant\t{name}\t{constant}\t{cprops}")
        for item in [s for s in sizes.split(";") if s]:
            n, lb = item.rsplit(":", 1)
            out.append(f"size_name\t{name}\t{n}\t{lb}")
    spellings = {}
    for s in rows["spellings.tsv"]:
        spellings.setdefault(s[0], []).append(s)
    for s in rows["symbols.tsv"]:
        out.append("symbol\t" + "\t".join(s))
        out += ["spelling\t" + "\t".join(sp) for sp in spellings.get(s[0], [])]
    out += ["argument\t" + "\t".join(a) for a in rows["arguments.tsv"]]
    repairs = {}
    for r in rows["repairs.tsv"]:
        repairs.setdefault((r[0], r[1]), []).append(r)

    def guards(owner_is_entry):
        for g in rows["guards.tsv"]:
            if (g[0] == ENTRY) == owner_is_entry:
                out.append("guard\t" + "\t".join(g))
                out.extend("repair\t" + "\t".join(r) for r in repairs.get((g[0], g[1]), []))

    out += ["flow\t" + "\t".join(f) for f in rows["flows.tsv"] if f[0] != ENTRY]
    guards(False)
    entries = rows["entries.tsv"]
    result = [e for e in entries if e[1] == RESULT]
    if len(result) != 1:
        raise bug(f"{len(result)} result rows ({RESULT}) in the entries, not one")
    out.append(f"entry\t{result[0][0]}\t{result[0][2]}")
    out += ["parameter\t" + "\t".join(e[1:]) for e in entries if e[1] != RESULT]
    out += ["flow\t" + "\t".join(f) for f in rows["flows.tsv"] if f[0] == ENTRY]
    guards(True)
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------- export and import ----

def export_kingdoms(master, names=None):
    """One or more kingdoms (every one when `names` is None) as a self-contained JSON document: their kingdoms.tsv rows
    and, by file name, their entity rows (kingdom column included). A parent named alone exports its variants too."""
    listed = kingdom_names(master)
    if names is None:
        chosen = listed
    else:
        chosen = []
        for n in names:
            if n not in listed:
                raise bug(f"{n!r} is not a kingdom of {KINGDOMS_FILE}")
            chosen += [k for k in listed if (k == n or any(r[0] == k and r[1] == n for r in master[KINGDOMS_FILE])) and k not in chosen]
    doc = {"format": "kingdoms/master 1", "columns": {KINGDOMS_FILE: KINGDOMS_COLUMNS, **TABLES},
           KINGDOMS_FILE: [r for r in master[KINGDOMS_FILE] if r[0] in chosen]}
    for fname in TABLES:
        doc[fname] = [r for r in master[fname] if r[0] in chosen]
    return doc


def import_kingdoms(doc, master):
    """The master with every kingdom of `doc` written in (each replaced as `write_kingdom` would, parents first)."""
    if doc.get("format") != "kingdoms/master 1" or doc.get("columns") != {KINGDOMS_FILE: KINGDOMS_COLUMNS, **TABLES}:
        raise bug("the JSON document is not an export of these tables")
    order = sorted(doc[KINGDOMS_FILE], key=lambda r: (r[1] != "", r[0]))  # parents before variants
    for kr in order:
        rows = {fname: [r[1:] for r in doc[fname] if r[0] == kr[0]] for fname in TABLES}
        master = merged(master, kr[0], rows, kr)
    return master


def empty_master():
    return {fname: [] for fname in [KINGDOMS_FILE] + list(TABLES)}


def selftest(master_dir=HERE):
    """Export every kingdom, import into empty tables: byte-identical files; and every kingdom's rows round trip
    through the named text form."""
    import tempfile
    master = read_master(master_dir)
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        doc = json.loads(json.dumps(export_kingdoms(master)))
        write_master(import_kingdoms(doc, empty_master()), tmp)
        for fname in [KINGDOMS_FILE] + list(TABLES):
            with open(os.path.join(master_dir, fname), "rb") as a, open(os.path.join(tmp, fname), "rb") as b:
                if a.read() != b.read():
                    bad.append(f"{fname}: export-all then import into empty tables is not byte-identical")
    for k in kingdom_names(master):
        rows = select(master, k)
        if not rows["entries.tsv"]:
            continue  # a parent: no rows of its own
        if rows_of_named(named_of_rows(rows)) != rows:
            bad.append(f"{k}: its rows do not round trip through the named text form")
    return bad


# ---------------------------------------------------------------------------------------------------------- CLI ----

def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--named", metavar="KINGDOM", help="print a kingdom's named text form")
    mode.add_argument("--write", metavar="KINGDOM", help="replace a kingdom's rows with the named text on stdin")
    mode.add_argument("--check", metavar="KINGDOM", help="compare a kingdom's rows with the named text on stdin")
    mode.add_argument("--write-parent", metavar="KINGDOM", help="write a parent kingdom's row (no rows of its own)")
    mode.add_argument("--export", nargs="*", metavar="KINGDOM", help="JSON of the named kingdoms (all when none)")
    mode.add_argument("--import", dest="import_", metavar="FILE.json", help="write every kingdom of the JSON in")
    mode.add_argument("--selftest", action="store_true", help="export all, import into empty tables: byte-identical")
    ap.add_argument("--parent", default="", help="the variant's parent kingdom (--write, --check)")
    ap.add_argument("--description", default="", help="the kingdom's one-line description")
    ap.add_argument("--root", default="", help="the root data type (default: the entry's result)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--into", default=HERE, metavar="DIR", help="the tables' directory (default: this one)")
    a = ap.parse_args(argv[1:])
    master_dir = a.into
    if a.named:
        sys.stdout.write(named_of_rows(select(read_master(master_dir), a.named)))
        return 0
    if a.write_parent:
        write_parent(a.write_parent, a.description, master_dir, a.notes)
        print(f"{a.write_parent}: parent row written into {master_dir}")
        return 0
    if a.write or a.check:
        kingdom = a.write or a.check
        rows = rows_of_named(sys.stdin.read())
        root = a.root or root_of(rows)
        kingdom_row = (kingdom, a.parent, a.description, root, a.notes)
        if a.write:
            write_kingdom(kingdom, rows, kingdom_row, master_dir)
            print(f"{kingdom}: {sum(len(v) for v in rows.values())} rows written into {master_dir}")
            return 0
        bad = check_kingdom(kingdom, rows, kingdom_row, master_dir)
        for b in bad:
            print(b)
        print(f"{kingdom}: {'MATCHES' if not bad else 'DIFFERS FROM'} the committed tables")
        return 1 if bad else 0
    if a.export is not None:
        json.dump(export_kingdoms(read_master(master_dir), a.export or None), sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    if a.import_:
        with open(a.import_, encoding="utf-8") as f:
            doc = json.load(f)
        write_master(import_kingdoms(doc, read_master(master_dir)), master_dir)
        print(f"{len(doc[KINGDOMS_FILE])} kingdoms imported into {master_dir}")
        return 0
    bad = selftest(master_dir)
    for b in bad:
        print(b)
    print(f"selftest: {len(bad)} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except RowsError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
