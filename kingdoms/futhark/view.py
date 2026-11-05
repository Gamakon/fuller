"""THE MASTER AND SPECIFICATION READERS of the Futhark kingdom.

The view itself -- filter the master's rows by the specification, bind them,
keep the types reachable both ways, write the exact table -- has ONE
implementation, in Rust (`src/futhark_view.rs`, docs/PLAN_futhark_kingdom.md
3g). Its tables are written by `cargo run --release --example futhark_views --
DIR`, and `gen_tables.py --views DIR` runs the witnesses on them. This module
keeps only what the generator's master witnesses and the sniffer read: the
master's files (`master()`), a specification (`read_spec`), and the template
grammar (`VAR_RE`, `QUANT_RE`). The Python expansion that wrote the same
tables byte for byte was retired with it.
"""

import itertools
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "master")

VAR_RE = re.compile(r"\{([A-Z][A-Z0-9]*(?:\.[a-z0-9]+)?)\}")
QUANT_RE = re.compile(r"^([A-Z][A-Z0-9]*):([a-z0-9]+)(?:\((.*)\))?$")
ROW_COLUMNS = ["ref", "since", "family", "kind", "elems", "scope", "when", "name", "out", "args", "spelling", "note"]
FRAME_COLUMNS = ["family", "kind", "since", "code", "scope", "binds"]


def read_tsv(path, columns):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    head = lines[0].split("\t")
    if head != columns:
        raise ValueError(f"{path}: the header is {head}, expected {columns}")
    out = []
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        f = line.split("\t")
        if len(f) != len(columns):
            raise ValueError(f"{path}:{i}: {len(f)} columns, expected {len(columns)}")
        out.append(dict(zip(columns, f)))
    return out


def master(path=MASTER):
    """The master: its version, rows, frames and literal pools."""
    with open(os.path.join(path, "VERSION")) as f:
        version = int(f.read().strip())
    pools = {}
    for p in read_tsv(os.path.join(path, "pools.tsv"), ["element", "literals"]):
        pools[p["element"]] = p["literals"].split(",")
    return {
        "version": version,
        "rows": read_tsv(os.path.join(path, "rows.tsv"), ROW_COLUMNS),
        "frames": read_tsv(os.path.join(path, "frames.tsv"), FRAME_COLUMNS),
        "rules": read_tsv(os.path.join(path, "rules.tsv"), ["on", "elems", "needs", "rule"]),
        "pools": pools,
    }


# ---- the specification --------------------------------------------------------------------------------------------

def read_spec(text):
    """A view specification: four-column lines `kind name element dims`; `#` lines are comments. A `families` line
    belongs to the matrix view only: a signature view admits every family of the master. The `element` line lists
    the view's elements (`elements`; `element` is the first); a `literals` line is one element's pool (its element
    in the third column, empty in a view of one element): `literals` maps each element to its pool. A `lets` line
    names the array classes a `let` at the top of a program may bind, each once (master version 5, the kernel boundary)."""
    spec = {"inputs": [], "sizes": [], "families": [], "shape": "signature", "elements": [], "literals": {}, "lets": []}
    pools = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) != 4:
            raise ValueError(f"the specification line {line!r} has {len(f)} fields, wants 4")
        kind, name, elem, dims = f
        lst = [x for x in dims.split(",") if x]
        if kind == "input":
            spec["inputs"].append((name, elem, tuple(lst)))
        elif kind == "size":
            spec["inputs"].append((name, "size", ()))
            spec["sizes"].append(name)
        elif kind == "output":
            spec["output"] = (elem, tuple(lst))
        elif kind in ("master", "depth"):
            spec[kind] = int(name)
        elif kind == "shape":
            spec[kind] = name
        elif kind == "element":
            spec["elements"] = [x for x in name.split(",") if x]
            spec["element"] = spec["elements"][0] if spec["elements"] else ""
        elif kind == "literals":
            pools.append((elem, [x for x in name.split(",") if x]))
        elif kind in ("families", "dims", "roles", "root", "lets"):
            spec[kind] = [x for x in name.split(",") if x]
        else:
            raise ValueError(f"a specification line of kind {kind}")
    if not spec["elements"]:
        raise ValueError("the specification names no element")
    for elem, pool in pools:
        if not elem:
            if len(spec["elements"]) != 1:
                raise ValueError(f"a literals line names no element and the view has {len(spec['elements'])}")
            elem = spec["elements"][0]
        if elem not in spec["elements"] or elem in spec["literals"]:
            raise ValueError(f"the literals line of {elem} is not one of the view's elements, once")
        spec["literals"][elem] = pool
    if spec["shape"] == "signature" and any(l.split("\t")[0] == "families" for l in text.splitlines()):
        raise ValueError("a signature specification has a families line: a signature view admits every family")
    return spec


# ---- the class ------------------------------------------------------------------------------------------------------

# The element types a class may name; a dimension is never named as one (the class text would read two ways).
ELEMENT_NAMES = ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "f32", "f64", "bool")
DIM_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def split_top(text):
    """The pieces of `text` between its top-level commas (a comma inside parentheses separates a tuple's components)."""
    out, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == "," and depth == 0:
            out.append(text[start:i])
            start = i + 1
    out.append(text[start:])
    return out


def parse_class(text, elements):
    """A class's text as its AST (`src/futhark_view.rs`, `Class`; the grammar is in kingdoms/futhark/README.md):
    ("scalar", elem) | ("ix", dim) | ("arr", elem, (dims)) | ("tuple", (classes)). An array written without its
    element (a view of one element) is of that element. Raises on a text that does not parse."""
    if text.startswith("("):
        if not text.endswith(")"):
            raise ValueError(f"the class {text} opens a tuple it does not close")
        parts = split_top(text[1:-1])
        if len(parts) < 2:
            raise ValueError(f"the tuple class {text} has {len(parts)} component")
        return ("tuple", tuple(parse_class(p, elements) for p in parts))
    if text.startswith("ix."):
        d = text[3:]
        if not DIM_RE.match(d) or d in ELEMENT_NAMES:
            raise ValueError(f"the index class {text} is not over a dimension")
        return ("ix", d)
    if text.startswith("a."):
        pieces = text[2:].split(".")
        if pieces and pieces[0] in ELEMENT_NAMES:
            elem = pieces.pop(0)
            if elem not in elements:
                raise ValueError(f"the class {text} is of the element {elem}, which the view does not have")
        elif len(elements) == 1:
            elem = elements[0]
        else:
            raise ValueError(f"the array class {text} names no element and the view has {len(elements)}")
        if not pieces or not all(DIM_RE.match(d) and d not in ELEMENT_NAMES for d in pieces):
            raise ValueError(f"the array class {text} is not over dimensions")
        return ("arr", elem, tuple(pieces))
    if text in elements:
        return ("scalar", text)
    raise ValueError(f"the class {text!r} does not parse")


def class_text(cls, elements):
    """The text of a parsed class in a view of `elements` (the element written only when there are several)."""
    kind = cls[0]
    if kind == "scalar":
        return cls[1]
    if kind == "ix":
        return f"ix.{cls[1]}"
    if kind == "arr":
        return "a." + ".".join(((cls[1],) if len(elements) > 1 else ()) + cls[2])
    return "(" + ",".join(class_text(p, elements) for p in cls[1]) + ")"


def class_futhark_type(cls):
    """A parsed class's Futhark type: `f32`, `[n][d]f32`, `(f32, i32)`."""
    kind = cls[0]
    if kind == "scalar":
        return cls[1]
    if kind == "ix":
        return "i64"
    if kind == "arr":
        return "".join(f"[{d}]" for d in cls[2]) + cls[1]
    return "(" + ", ".join(class_futhark_type(p) for p in cls[1]) + ")"


def class_tag(cls, elements):
    """A parsed class as a piece of an identifier: its text without the `a.`/`ix.` prefix, `.` as `_` (`n`, `f32_n`)."""
    t = class_text(cls, elements)
    for p in ("a.", "ix."):
        if t.startswith(p):
            t = t[len(p):]
            break
    return "".join(c if c.isalnum() else "_" for c in t)


# ==== MASTER 6 =======================================================================================================
#
# Master 6 (`master/6/`, the format in `master/README.md`, plan section 7 task F1a): a type is a DATA TYPE and nothing
# else; an array's length and an index's bound are PROPERTIES, carried by rules and checked by requirements; the
# variables a node may use are its ENVIRONMENT, built by the decoder from the binding rows. The view below is the one
# Python implementation of a master 6 view; `src/futhark_view.rs` computes the same tables byte for byte (task F1b).
#
# A master 6 view is
#
# 1. the CANDIDATE data types: `datatypes.tsv`'s templates, in order, each over the view's elements (`E`, or the
#    ordered pairs and triples of elements), a function type only where its members are candidates;
# 2. the CANDIDATE rows: each admitted row (version at most the pin, its family admitted -- every family when the
#    specification names none, and the `var` row in every view -- one of its elements in the view) instantiated once for each of its elements when its templates read `{E}`, else once, then over its `over`
#    quantifiers in order; an instance whose output or an argument is not a candidate data type does not exist;
#    `depth` is replaced by the view's depth (the work-bound requirement is dropped in a view with no depth), and a
#    rule, requirement or binding property that names a property its data type does not carry is dropped;
# 3. kept when REACHABLE both ways, as master 5's view keeps its rows (forward from the leaves, backward from the
#    root), with `var@{T}` for every data type that has a variable: a root variable, or a binding of a kept row (the
#    two are iterated until they settle); and for every data type the root or a kept row demands that no kept row
#    of no arguments gives (its constant when no variable is in scope);
# 4. written: `datatypes.tsv`, `symbols.tsv` and `environment.tsv` (columns below).

MASTER6 = os.path.join(MASTER, "6")
ROW6_COLUMNS = ["ref", "since", "family", "kind", "elems", "over", "name", "out", "args", "rules", "requires", "binds",
                "trip", "spelling", "note"]
DATATYPE6_COLUMNS = ["name", "kind", "members", "over", "tag", "constant", "constant_props", "note"]
PROPERTY6_COLUMNS = ["data_type", "name", "kind", "meaning"]
ADAPTER6_COLUMNS = ["name", "repairs", "runs_at", "heavy", "rendering"]
# The written view's columns.
VIEW6_DATATYPE_COLUMNS = ["id", "name", "kind", "members", "tag", "constant", "constant_props", "properties"]
VIEW6_SYMBOL_COLUMNS = ["id", "ref", "name", "kind", "out", "args", "rules", "requires", "binds", "trip", "spelling"]
VIEW6_ENV_COLUMNS = ["kind", "name", "data_type", "properties", "requires"]
# The template columns a row's `{E}` and quantifier variables are substituted in.
ROW6_TEMPLATED = ["name", "out", "args", "rules", "requires", "binds", "spelling", "note"]
PROPERTY_TERM_RE = re.compile(r"\bin(\d+)\.([a-z_][a-z0-9_]*)\b")
REQUIRE6_RE = re.compile(r"^(in\d+\.[a-z_][a-z0-9_]*|env\.trips|out\.[a-z_][a-z0-9_]*) (=|<=|<) (.+) -> ([a-z][a-z0-9_.]*)$")
BIND6_RE = re.compile(r"^(\d+):(\S+)(?: (.*))?$")


def master6(path=MASTER6):
    """Master 6: its version, data types, properties, rows, adapters, rules and literal pools."""
    with open(os.path.join(path, "VERSION")) as f:
        version = int(f.read().strip())
    pools = {}
    for p in read_tsv(os.path.join(path, "pools.tsv"), ["element", "literals"]):
        pools[p["element"]] = p["literals"].split(",")
    return {
        "version": version,
        "datatypes": read_tsv(os.path.join(path, "datatypes.tsv"), DATATYPE6_COLUMNS),
        "properties": read_tsv(os.path.join(path, "properties.tsv"), PROPERTY6_COLUMNS),
        "rows": read_tsv(os.path.join(path, "rows.tsv"), ROW6_COLUMNS),
        "adapters": read_tsv(os.path.join(path, "adapters.tsv"), ADAPTER6_COLUMNS),
        "rules": read_tsv(os.path.join(path, "rules.tsv"), ["on", "elems", "needs", "rule"]),
        "pools": pools,
    }


def items(field, sep=";"):
    """A field's `;`-separated items, stripped; an empty field has none."""
    return [x.strip() for x in field.split(sep) if x.strip()]


def read_spec6(text):
    """A master 6 view specification (written by `sniff.py --format 6`): four-column lines `kind name element dims`.
    `format 6`; `master N`, the row-version pin; `element`, the elements; `families`, when the view admits only those;
    `input`, `zeros` and `size` lines, which in their order are the ROOT ENVIRONMENT (inputs, zero arrays, sizes);
    `output`; `depth`, absent in a view with no work bound; `lets`, data types; `literals`, one element's pool."""
    spec = {"format": None, "master": None, "elements": [], "families": [], "env": [], "output": None, "depth": None,
            "lets": [], "literals": {}}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) != 4:
            raise ValueError(f"our software has a bug: the master 6 specification line {line!r} has {len(f)} fields, wants 4")
        kind, name, elem, dims = f
        lst = [x for x in dims.split(",") if x]
        if kind in ("format", "master", "depth"):
            spec[kind] = int(name)
        elif kind == "element":
            spec["elements"] = [x for x in name.split(",") if x]
        elif kind in ("families", "lets"):
            spec[kind] = [x for x in name.split(",") if x]
        elif kind in ("input", "zeros", "size"):
            spec["env"].append((kind, name, elem, tuple(lst)))
        elif kind == "output":
            spec["output"] = (name, elem, tuple(lst))
        elif kind == "literals":
            spec["literals"][elem or (spec["elements"][0] if len(spec["elements"]) == 1 else "")] = [x for x in name.split(",") if x]
        else:
            raise ValueError(f"our software has a bug: a master 6 specification line of kind {kind!r}")
    if spec["format"] != 6 or spec["master"] is None or not spec["elements"] or spec["output"] is None:
        raise ValueError("our software has a bug: a master 6 specification needs its format 6, master, element and output lines")
    return spec


def array_type(elem, rank):
    """The data type of an array of `rank` dimensions of `elem` (a scalar at rank 0)."""
    return "[]" * rank + elem


def env_variable(entry):
    """A root environment line as (name, data type, [(property, value)])."""
    kind, name, elem, dims = entry
    if kind == "size":
        return name, "size", [("value", name)]
    dt = array_type(elem, len(dims))
    props = {0: [], 1: ["len"], 2: ["rows", "cols"]}.get(len(dims))
    if props is None:
        raise ValueError(f"our software has a bug: the root variable {name} has {len(dims)} dimensions; master 6 has arrays of rank 1 and 2")
    return name, dt, list(zip(props, dims))


def _subst(text, env):
    """Replace every `{V}` / `{V.x}` quantifier variable of `env` (an unbound one is a bug)."""
    def one(m):
        if m.group(1) not in env:
            raise ValueError(f"our software has a bug: the template {text!r} reads {{{m.group(1)}}}, which nothing binds")
        return env[m.group(1)]
    return VAR_RE.sub(one, text)


class View6:
    """A master 6 view: the specification over the master, its tables computed at construction."""

    def __init__(self, spec, mst=None):
        self.spec = spec
        self.mst = mst or master6()
        self.elements = spec["elements"]
        self.multi = len(self.elements) > 1
        self.props_of_template = {}
        for p in self.mst["properties"]:
            self.props_of_template.setdefault(p["data_type"], []).append((p["name"], p["kind"]))
        self.candidates = self._candidate_types()
        self.by_name = {t["name"]: t for t in self.candidates}
        self.root_vars = [env_variable(e) for e in spec["env"]]
        out_name, out_elem, out_dims = spec["output"]
        self.root = array_type(out_elem, len(out_dims))
        for _, dt, _ in self.root_vars + [(None, self.root, None)]:
            if dt not in self.by_name:
                raise ValueError(f"our software has a bug: the specification names the data type {dt}, which the master does not have for {self.elements}")
        self.symbols, self.types = self._compute()

    # ---- 1. the candidate data types ----------------------------------------------------------------------------

    def _candidate_types(self):
        out = []
        for t in self.mst["datatypes"]:
            over = t["over"]
            if over == "E":
                envs = [{"E": e} for e in self.elements]
            elif over == "A,B:elements":
                envs = [{"A": a, "B": b} for a in self.elements for b in self.elements]
            elif over == "A,B,C:elements":
                envs = [{"A": a, "B": b, "C": c} for a in self.elements for b in self.elements for c in self.elements]
            elif over == "":
                envs = [{}]
            else:
                raise ValueError(f"our software has a bug: the data type {t['name']} is over {over!r}")
            for env in envs:
                name = _subst(t["name"], env)
                members_text = _subst(t["members"], env)
                if t["kind"] == "function":
                    params, result = members_text.split(" -> ")
                    members = items(params) + [result]
                else:
                    members = items(members_text)
                if any(m not in [c["name"] for c in out] for m in members):
                    continue  # a function type of an element the view does not have
                props = [(p, k) for p, k in self.props_of_template.get(t["name"], [])]
                out.append({"name": name, "kind": t["kind"], "members": members_text, "member_list": members,
                            "tag": _subst(t["tag"], env), "constant": _subst(t["constant"], env),
                            "constant_props": t["constant_props"], "properties": props})
        return out

    def carries(self, dt, prop):
        return any(p == prop for p, _ in self.by_name[dt]["properties"])

    # ---- 2. the candidate rows ----------------------------------------------------------------------------------

    def _admitted(self, r):
        return (int(r["since"]) <= self.spec["master"]
                and (not self.spec["families"] or r["family"] in self.spec["families"] or r["kind"] == "var")
                and (r["elems"] == "*" or any(e in self.elements for e in r["elems"].split(","))))

    def _quantifier(self, q, env, row_elems, var_types):
        var, _, kind = q.partition(":")
        def tagged(dt):
            return {var: dt, f"{var}.tag": self.by_name[dt]["tag"]}
        out = []
        if kind == "states":
            elems = [env["E"]] if "E" in env else row_elems
            for dt in [e for e in elems] + ["[]" + e for e in elems]:
                if dt in self.by_name:
                    out.append(dict(env, **tagged(dt)))
        elif kind in ("tuples", "tuples3"):
            n = 2 if kind == "tuples" else 3
            for parts in itertools.product(self.elements, repeat=n):
                dt = "(" + ", ".join(parts) + ")"
                if dt in self.by_name and all(p in row_elems for p in parts):
                    b = dict(env, **tagged(dt))
                    b.update({f"{var}.{x}": p for x, p in zip("abc", parts)})
                    b[f"{var}.zero"] = "(" + ", ".join("0" + p for p in parts) + ")"
                    out.append(b)
        elif kind == "lets":
            out = [dict(env, **tagged(dt)) for dt in self.spec["lets"]]
        elif kind == "all":
            out = [dict(env, **tagged(t["name"])) for t in self.candidates if t["name"] != "size"]
        elif kind == "vars":
            out = [dict(env, **tagged(t["name"])) for t in self.candidates if t["name"] in var_types]
        else:
            raise ValueError(f"our software has a bug: the quantifier {q!r} is not one of master 6's")
        return out

    def _instance(self, r, env):
        """One instance of row `r` under `env`, or None when one of its data types is not a candidate."""
        t = {c: _subst(r[c], env) for c in ROW6_TEMPLATED}
        args = items(t["args"])
        if t["out"] not in self.by_name or any(a not in self.by_name for a in args):
            return None
        depth = self.spec["depth"]

        def held(expr):
            """Whether every `in<k>.<p>` of `expr` names a property its argument carries."""
            return all(int(k) < len(args) and self.carries(args[int(k)], p) for k, p in PROPERTY_TERM_RE.findall(expr))

        rules = []
        for rule in items(t["rules"]):
            prop, _, expr = rule.partition("=")
            if self.carries(t["out"], prop) and held(expr):
                rules.append(rule)
        requires = []
        for q in items(t["requires"]):
            m = REQUIRE6_RE.match(q)
            if not m:
                raise ValueError(f"our software has a bug: the requirement {q!r} of {r['ref']} does not parse")
            subject, rel, expr, adapter = m.groups()
            if subject == "env.trips":
                if depth is None:
                    continue
                bound = re.sub(r"\bdepth\b", str(depth), expr)
                requires.append(f"{subject} {rel} {bound} -> {adapter}")
                continue
            k, p = subject[2:].split(".")
            if int(k) < len(args) and self.carries(args[int(k)], p) and held(expr):
                requires.append(q)
        binds = []
        for b in items(t["binds"]):
            m = BIND6_RE.match(b)
            if not m:
                raise ValueError(f"our software has a bug: the binding {b!r} of {r['ref']} does not parse")
            arg, dt, props = m.groups()
            if dt not in self.by_name:
                return None
            kept = [pe for pe in items(props or "", "&") if self.carries(dt, pe.partition("=")[0]) and held(pe.partition("=")[2])]
            binds.append((int(arg), dt, kept))
        return {"ref": r["ref"], "name": t["name"], "kind": r["kind"], "family": r["family"], "out": t["out"],
                "args": args, "rules": rules, "requires": requires, "binds": binds, "trip": r["trip"],
                "spelling": t["spelling"]}

    def _rows(self, var_types):
        out = []
        for r in self.mst["rows"]:
            if not self._admitted(r):
                continue
            row_elems = list(self.elements) if r["elems"] == "*" else [e for e in self.elements if e in r["elems"].split(",")]
            if any("{E}" in r[c] or "{E.tag}" in r[c] for c in ROW6_TEMPLATED):
                envs = [{"E": e, "E.tag": f".{e}" if self.multi else ""} for e in row_elems]
            else:
                envs = [{}]
            for q in items(r["over"]):
                envs = [b for env in envs for b in self._quantifier(q, env, row_elems, var_types)]
            for env in envs:
                inst = self._instance(r, env)
                if inst is not None:
                    out.append(inst)
        return out

    # ---- 3. reachable both ways ---------------------------------------------------------------------------------

    def _reachable(self, rows):
        built = {r["out"] for r in rows if r["kind"] != "function"}
        while True:
            before = len(built)
            for r in rows:
                if r["out"] not in built and all(a in built for a in r["args"]):
                    built.add(r["out"])
            if len(built) == before:
                break
        if self.root not in built:
            raise ValueError(f"our software has a bug: the root {self.root} cannot be built from the view's leaves")
        usable = [all(a in built for a in r["args"]) for r in rows]
        keep = {self.root}
        while True:
            before = len(keep)
            for r, u in zip(rows, usable):
                if u and r["out"] in keep:
                    keep.update(r["args"])
            if len(keep) == before:
                break
        return [r for r, u in zip(rows, usable) if u and r["out"] in keep]

    def _leafless(self, rows):
        """The data types the root and `rows` demand that no row of no arguments in `rows` gives. Each must have a
        constant (every data type but `size` has one; a view whose `size` is leafless names no size, a bug)."""
        leaves = {r["out"] for r in rows if not r["args"]}
        demanded = {self.root} | {a for r in rows for a in r["args"]}
        out = {t for t in demanded if t not in leaves}
        for t in sorted(out):
            if not self.by_name[t]["constant"]:
                raise ValueError(f"our software has a bug: the data type {t} is demanded, has no leaf and no constant")
        return out

    def _compute(self):
        root_types = {dt for _, dt, _ in self.root_vars}
        var_types = root_types | {dt for r in self._rows(set()) for _, dt, _ in r["binds"]}
        for _ in range(len(self.candidates) + 2):
            rows = self._reachable(self._rows(var_types))
            settled = root_types | {dt for r in rows for _, dt, _ in r["binds"]}
            settled = {t for t in var_types if t in settled}
            if settled == var_types:
                break
            var_types = settled
        else:
            raise ValueError("our software has a bug: the data types with a variable do not settle")
        # A data type the root or a kept row demands that no kept row of no arguments gives (the `i32` + `f32` k-means
        # view's `[][]i32`: built by `tabrows`, with no variable) takes its var row too, which with no variable in scope
        # renders the data type's constant: Karva's tail needs a leaf of every data type it can demand (the contract's
        # check), and the design's `var.T` is T's constant when c = 0. A var row has no argument, so the reachable rows
        # are otherwise unchanged.
        leafless = self._leafless(rows)
        if leafless:
            var_types = var_types | leafless
            rows = self._reachable(self._rows(var_types))
        used = {self.root}
        for r in rows:
            used.add(r["out"])
            used.update(r["args"])
            used.update(dt for _, dt, _ in r["binds"])
        while True:
            before = len(used)
            for t in self.candidates:
                if t["name"] in used:
                    used.update(t["member_list"])
            if len(used) == before:
                break
        types = [t for t in self.candidates if t["name"] in used]
        self.var_types = [t["name"] for t in types if any(r["kind"] == "var" and r["out"] == t["name"] for r in rows)]
        return rows, types

    # ---- 4. the written tables ----------------------------------------------------------------------------------

    def environment(self):
        """The root environment: the root (the entry's result, its requirements against the signature's sizes) and the
        variables whose data type the view can name (has a `var` row), in the specification's order."""
        name, _, dims = self.spec["output"]
        props = {0: [], 1: ["len"], 2: ["rows", "cols"]}[len(dims)]
        root_props = " & ".join(f"{p}={d}" for p, d in zip(props, dims))
        root_req = "; ".join(f"out.{p} = {d} -> fit.root" for p, d in zip(props, dims))
        lines = [("root", name, self.root, root_props, root_req)]
        for (kind, _, _, _), (vname, dt, ps) in zip(self.spec["env"], self.root_vars):
            if dt in self.var_types:
                lines.append((kind, vname, dt, " & ".join(f"{p}={v}" for p, v in ps), ""))
        return lines

    def texts(self):
        """The view's three tables as text, by file name."""
        dt = ["\t".join(VIEW6_DATATYPE_COLUMNS)]
        for i, t in enumerate(self.types):
            dt.append("\t".join([str(i), t["name"], t["kind"], t["members"], t["tag"], t["constant"], t["constant_props"],
                                 ";".join(f"{p}:{k}" for p, k in t["properties"])]))
        sy = ["\t".join(VIEW6_SYMBOL_COLUMNS)]
        for i, r in enumerate(self.symbols):
            binds = "; ".join(f"{a}:{d}" + (" " + " & ".join(ps) if ps else "") for a, d, ps in r["binds"])
            sy.append("\t".join([str(i), r["ref"], r["name"], r["kind"], r["out"], ";".join(r["args"]), "; ".join(r["rules"]),
                                 "; ".join(r["requires"]), binds, r["trip"], r["spelling"]]))
        en = ["\t".join(VIEW6_ENV_COLUMNS)] + ["\t".join(line) for line in self.environment()]
        return {"datatypes.tsv": "\n".join(dt) + "\n", "symbols.tsv": "\n".join(sy) + "\n",
                "environment.tsv": "\n".join(en) + "\n"}
