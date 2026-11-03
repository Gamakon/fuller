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
