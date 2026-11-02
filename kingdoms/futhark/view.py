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
FRAME_COLUMNS = ["family", "kind", "code", "scope", "binds"]


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
    belongs to the matrix view only: a signature view admits every family of the master."""
    spec = {"inputs": [], "sizes": [], "families": [], "shape": "signature"}
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
        elif kind in ("element", "shape"):
            spec[kind] = name
        elif kind in ("families", "literals", "dims", "roles", "root"):
            spec[kind] = [x for x in name.split(",") if x]
        else:
            raise ValueError(f"a specification line of kind {kind}")
    if spec["shape"] == "signature" and any(l.split("\t")[0] == "families" for l in text.splitlines()):
        raise ValueError("a signature specification has a families line: a signature view admits every family")
    return spec
