#!/usr/bin/env python3
"""THE SNIFFER: a Futhark problem's source to its view specification.

The problem is a Futhark program and its entry point. The Futhark compiler
types it: `futhark dev -t` prints the type-checked program with every size
inferred (an anonymous result such as histogram's `: []i32` prints as
`: [i64.i32 k]i32`), and the sniffer reads the entry's line of that print:

* the size parameters and the parameters `((name: T): T)`;
* a scalar `i32` parameter that names a dimension, as `i64.i32 p`, is a SIZE
  INPUT (a dimension given as a value); any other parameter is an input array
  over named dimensions;
* the result type is the output; its dimensions must all be dimensions of an
  input (a size the inputs do not fix would be unknown);
* the element types are those of the parameters and the result (i32, f32),
  in order of first appearance; a view of several elements writes an array
  class with its element (`a.f32.n`, docs/PLAN_futhark_kingdom.md D7), a view
  of one keeps the names of before (`a.n`). `--elements E,..` adds elements
  the signature does not name, for values a program keeps inside (k-means's
  assignments are i32 indices beside its f32 points); `--lets C,..` names the
  array classes a `let` at the top of a program may bind, each once (master
  version 5, the KERNEL BOUNDARY: one of UltraDim's kernels writes a buffer
  the next one reads; k-means's assignments are `a.i32.n`);
* no FAMILY is chosen here: a view admits every family of the master and
  keeps the types reachable both ways, forward from the leaves and backward
  from the root (`src/futhark_view.rs`; docs/PLAN_futhark_kingdom.md, 3e), so what the
  problem's source happens to use limits nothing;
* the literal pool is the master's for the element; the depth is the
  master's default (2) unless `--depth N` says otherwise (k-means needs 3: a
  reduction over `d` inside a loop over `k` inside a tabulate over `n`); the
  master version is the current one unless `--master N` PINS the view to an
  older one (the four i32 benchmarks are pinned to version 2, so the rows
  added since do not change their tables; moving a pin is a decision, with
  its regression re-measured).

It writes the specification `view.tsv` (the WHERE clause of the view, see
`src/futhark_view.rs`), replacing the hand-written `interface.tsv`.

    python3 kingdoms/futhark/sniff.py <source.fut> [--entry main] [--depth N] [--master N] [--elements E,..] [--lets CLASS,..] [--out view.tsv]
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import view  # noqa: E402

DEFAULT_DEPTH = 2
ELEMENTS = ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "f32", "f64", "bool")
TYPE_RE = re.compile(r"^((?:\[[^\]]*\])*)([a-z][a-z0-9]*)$")


def typecheck(path):
    """The type-checked program as `futhark dev -t` prints it."""
    out = subprocess.run(["futhark", "dev", "-t", path], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise ValueError(f"futhark dev -t {path} failed:\n{out.stderr}")
    return out.stdout


def parse_type(text):
    """A printed type `[n][i64.i32 k]i32` -> (["n", "i64.i32 k"], "i32")."""
    m = TYPE_RE.match(text.strip())
    if not m:
        raise ValueError(f"the type {text!r} is not an array of an element (a tuple or a record is step 4)")
    dims = re.findall(r"\[([^\]]*)\]", m.group(1))
    return dims, m.group(2)


def check_compiler():
    """The sniffer parses the print of Futhark 0.28.1; another version is stopped, not guessed at."""
    out = subprocess.run(["futhark", "--version"], capture_output=True, text=True, timeout=60).stdout
    if not out.startswith("Futhark 0.28.1"):
        raise ValueError(f"the sniffer reads the print of Futhark 0.28.1, this is {out.splitlines()[:1]}")


def entry_line(printed, entry):
    for line in printed.splitlines():
        if line.startswith(f"entry {entry} "):
            if not line.rstrip().endswith("="):
                raise ValueError(f"the entry line of {entry} does not end its signature on one line: {line!r}")
            return line
    raise ValueError(f"the program has no entry {entry}")


def split_params(text):
    """`[n] ((k: i32): i32) ((is: [n]i32): [n]i32)` -> (["n"], [("k", "i32"), ("is", "[n]i32")]); the parameter's
    pattern type, which is the declared one."""
    sizes, params, i = [], [], 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
        elif c == "[":
            j = text.index("]", i)
            sizes.append(text[i + 1:j].strip())
            i = j + 1
        elif c == "(":
            depth, j = 0, i
            while True:
                depth += {"(": 1, ")": -1}.get(text[j], 0)
                if depth == 0:
                    break
                j += 1
            m = re.fullmatch(r"\(\((\w+)\s*:\s*([^()]+)\)\s*:\s*([^()]+)\)", text[i:j + 1])
            if not m:
                raise ValueError(f"the parameter {text[i:j + 1]!r} is not a typed name")
            params.append((m.group(1), m.group(2).strip()))
            i = j + 1
        else:
            raise ValueError(f"the entry's parameters hold {text[i:]!r}")
    return sizes, params


def sniff(path, entry="main", depth=DEFAULT_DEPTH, master=None, elements=(), lets=()):
    """The view specification of a problem, as text; `master` pins the view to a master version (the current one
    when None); `elements` are added to the ones the signature names; `lets` are the kernel boundaries' classes."""
    check_compiler()
    if master is None:
        master = view.master()["version"]
    elif not 1 <= master <= view.master()["version"]:
        raise ValueError(f"the master version {master} is not one of 1 .. {view.master()['version']}")
    for e in elements:
        if e not in ELEMENTS:
            raise ValueError(f"--elements names {e!r}, which is not a Futhark primitive type")
    printed = typecheck(path)
    line = entry_line(printed, entry)
    head = line[len(f"entry {entry} "):].rstrip()[:-1].rstrip()
    cut = head.rindex(") : ") if ") : " in head else head.index(": ")
    params_text, result = (head[:cut + 1], head[cut + 4:]) if ") : " in head else ("", head[cut + 2:])
    sizes, params = split_params(params_text)
    typed = [(n, *parse_type(t)) for n, t in params]
    scalars = {n for n, dims, e in typed if not dims and e == "i32"}

    def dim(d):
        m = re.fullmatch(r"i64\.i32 (\w+)", d)
        if m and m.group(1) in scalars:
            return m.group(1), True
        if re.fullmatch(r"[a-z][a-z0-9_]*", d) and d in sizes:
            return d, False
        raise ValueError(f"the dimension {d!r} is neither a size parameter nor a size input")

    size_inputs = set()
    for _, dims, _ in typed:
        for d in dims:
            n, is_size = dim(d)
            if is_size:
                size_inputs.add(n)
    out_dims, out_elem = parse_type(result)
    out_dims = [dim(d) for d in out_dims]
    size_inputs |= {n for n, s in out_dims if s}
    # The elements in order of first appearance (the parameters, then the result), then the ones added.
    elems = []
    for e in [e for n, dims, e in typed if n not in size_inputs] + [out_elem] + list(elements):
        if e not in elems:
            elems.append(e)
    for n, dims, e in typed:
        if n not in size_inputs and not dims:
            raise ValueError(f"the parameter {n} is a scalar that is not a size: the master has no row for a scalar input")
    if any(e not in ELEMENTS for e in elems):
        raise ValueError(f"the problem has the element types {elems}: not all are primitive types")
    for e in elems:
        if e not in view.master()["pools"]:
            raise ValueError(f"the master has no literal pool for the element {e} (its rows have no {e} yet)")
    for d in set(sizes) | {n for n, _, _ in typed if n in size_inputs}:
        if d in ELEMENTS:
            raise ValueError(f"the dimension {d} is named as an element type: the class text would read two ways")
    lines = ["# kind\tname\telement\tdims",
             f"# written by kingdoms/futhark/sniff.py from {os.path.relpath(os.path.abspath(path), REPO)}, entry {entry}",
             f"master\t{master}\t\t",
             f"element\t{','.join(elems)}\t\t"]
    dims_seen = []
    for n, dims, e in typed:
        if n in size_inputs:
            lines.append(f"size\t{n}\ti32\t")
            dims_seen.append(n)
            continue
        names = [dim(d)[0] for d in dims]
        lines.append(f"input\t{n}\t{e}\t{','.join(names)}")
        dims_seen += [d for d in names if d not in dims_seen]
    lines.append(f"output\t{entry}\t{out_elem}\t{','.join(n for n, _ in out_dims)}")
    lines.append(f"depth\t{depth}\t\t")
    if lets:
        for c in lets:
            parsed = view.parse_class(c, elems)
            if parsed[0] != "arr":
                raise ValueError(f"--lets names {c!r}, which is not an array class")
            if view.class_text(parsed, elems) != c:
                raise ValueError(f"--lets names {c!r}, which is not written as this view writes it ({view.class_text(parsed, elems)})")
        lines.append(f"lets\t{','.join(lets)}\t\t")
    # One element's pool a line; a view of one element leaves the element column empty (the specifications of before).
    for e in elems:
        lines.append(f"literals\t{','.join(view.master()['pools'][e])}\t{e if len(elems) > 1 else ''}\t")
    lines.append(f"dims\t{','.join(dims_seen)}\t\t")
    return "\n".join(lines) + "\n"


def main(argv):
    args = argv[1:]
    if not args or args[0].startswith("-"):
        print(__doc__.strip().splitlines()[-1])
        return 2
    entry = args[args.index("--entry") + 1] if "--entry" in args else "main"
    depth = int(args[args.index("--depth") + 1]) if "--depth" in args else DEFAULT_DEPTH
    master = int(args[args.index("--master") + 1]) if "--master" in args else None
    elements = [e for e in args[args.index("--elements") + 1].split(",") if e] if "--elements" in args else []
    lets = [c for c in args[args.index("--lets") + 1].split(",") if c] if "--lets" in args else []
    text = sniff(args[0], entry, depth, master, elements, lets)
    if "--out" in args:
        with open(args[args.index("--out") + 1], "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
