#!/usr/bin/env python3
"""THE SNIFFER: a Futhark problem's source to its view specification (master/README.md section 8).

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
  in order of first appearance; `--elements E,..` adds elements the signature
  does not name, for values a program keeps inside (k-means's assignments are
  i32 indices beside its f32 points); `--lets T,..` names the array data types
  a `let` at the top of a program may bind (the KERNEL BOUNDARY: one of
  UltraDim's kernels writes a buffer the next one reads);
* the literal pool is the master's for the element; the depth is 2 unless
  `--depth N` says otherwise (k-means needs 3: a reduction over `d` inside a
  loop over `k` inside a tabulate over `n`) or `--depth none` asks for no work
  bound (A7); the master version is the current one unless `--master N` PINS
  the view to an older one (moving a pin is a decision, with its regression
  re-measured); `--families F,..` admits only those families (A7's matrix
  family).

It writes the specification `view.tsv`: the root environment in its order
(inputs, zero arrays, sizes), `lets` as data types. `--format 6` is the one
format, the master's version of the format.

    python3 kingdoms/futhark/sniff.py <source.fut> [--format 6] [--entry main] [--depth N|none] [--master N] [--elements E,..] [--lets TYPE,..] [--families F,..] [--out view.tsv]
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


def read_signature(path, entry, elements, pools):
    """The entry's signature, read from the type-checked program: (elements, typed parameters, the size inputs, the
    result's element, the result's dimensions as (name, is a size input), the dimensions in order of first
    appearance). `pools` are the master's literal pools (an element without one is stopped)."""
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
        if e not in pools:
            raise ValueError(f"the master has no literal pool for the element {e} (its rows have no {e} yet)")
    for d in set(sizes) | {n for n, _, _ in typed if n in size_inputs}:
        if d in ELEMENTS:
            raise ValueError(f"the dimension {d} is named as an element type: the class text would read two ways")
    dims_seen = []
    for n, dims, e in typed:
        if n in size_inputs:
            dims_seen.append(n)
            continue
        for d in dims:
            if dim(d)[0] not in dims_seen:
                dims_seen.append(dim(d)[0])
    return elems, typed, size_inputs, out_elem, out_dims, dims_seen, dim


# Names the adapters of master 6 reserve (master/README.md 5.5): an entry parameter may not take one.
RESERVED_FUNCTIONS = ("fit", "clampix")
RESERVED_PREFIX = "ad_"


def sniff6(path, entry="main", depth=DEFAULT_DEPTH, master=None, elements=(), lets=(), families=()):
    """The MASTER 6 view specification of a problem, as text (master/README.md section 8; plan section 7, F1a).

    The lines, in order: `format 6`; `master`, the row-version pin (the rows keep the versions they were added at, so
    a pin means what it meant); `element`; `families` when the view admits only those (A7's matrix family); the ROOT
    ENVIRONMENT in its order, which is the input of the decoder's one environment function: the `input` lines in
    signature order (a size input is a size, not an input), then a `zeros` line for each element and each size (the
    zero array `zeros_<size>`, `zeros_<element>_<size>` in a view of several elements, bound once by the entry), then
    the `size` lines in order of first appearance (the element column is `i32` for a size given as an `i32` value);
    `output`; `depth` (none: no work bound, A7); `lets`, data types; `literals`, one element's pool a line.
    `depth` None writes no depth line."""
    check_compiler()
    m6 = view.master6()
    if master is None:
        master = m6["version"]
    elif not 1 <= master <= m6["version"]:
        raise ValueError(f"the master version {master} is not one of 1 .. {m6['version']}")
    elems, typed, size_inputs, out_elem, out_dims, dims_seen, dim = read_signature(path, entry, elements, m6["pools"])
    names = [n for n, _, _ in typed]
    for n in names + [entry]:
        if n in RESERVED_FUNCTIONS or n.startswith(RESERVED_PREFIX):
            raise ValueError(f"our software has a bug: the entry {entry} names {n}, a name master 6's adapters reserve")
    zeros = [(f"zeros_{d}" if len(elems) == 1 else f"zeros_{e}_{d}", e, d) for e in elems for d in dims_seen]
    for z, _, _ in zeros:
        if z in names or z in dims_seen:
            raise ValueError(f"our software has a bug: the entry {entry} names {z}, the name of a zero array")
    lines = ["# kind\tname\telement\tdims",
             f"# written by kingdoms/futhark/sniff.py --format 6 from {os.path.relpath(os.path.abspath(path), REPO)}, entry {entry}",
             "format\t6\t\t",
             f"master\t{master}\t\t",
             f"element\t{','.join(elems)}\t\t"]
    for f in families:
        if f not in {r["family"] for r in m6["rows"]}:
            raise ValueError(f"--families names {f!r}, which is not a family of master 6")
    if families:
        lines.append(f"families\t{','.join(families)}\t\t")
    for n, dims, e in typed:
        if n not in size_inputs:
            lines.append(f"input\t{n}\t{e}\t{','.join(dim(d)[0] for d in dims)}")
    lines.append(f"output\t{entry}\t{out_elem}\t{','.join(n for n, _ in out_dims)}")
    for z, e, d in zeros:
        lines.append(f"zeros\t{z}\t{e}\t{d}")
    for d in dims_seen:
        lines.append(f"size\t{d}\t{'i32' if d in size_inputs else ''}\t")
    if depth is not None:
        lines.append(f"depth\t{depth}\t\t")
    if lets:
        for c in lets:
            m = re.fullmatch(r"((?:\[\])+)([a-z][a-z0-9]*)", c)
            if not m or m.group(2) not in elems or len(m.group(1)) > 4:
                raise ValueError(f"--lets names {c!r}, which is not an array data type of rank 1 or 2 of the view's elements")
        lines.append(f"lets\t{','.join(lets)}\t\t")
    for e in elems:
        lines.append(f"literals\t{','.join(m6['pools'][e])}\t{e if len(elems) > 1 else ''}\t")
    return "\n".join(lines) + "\n"


def main(argv):
    args = argv[1:]
    if not args or args[0].startswith("-"):
        print(__doc__.strip().splitlines()[-1])
        return 2
    entry = args[args.index("--entry") + 1] if "--entry" in args else "main"
    depth_text = args[args.index("--depth") + 1] if "--depth" in args else str(DEFAULT_DEPTH)
    depth = DEFAULT_DEPTH if depth_text == "none" else int(depth_text)
    master = int(args[args.index("--master") + 1]) if "--master" in args else None
    elements = [e for e in args[args.index("--elements") + 1].split(",") if e] if "--elements" in args else []
    lets = [c for c in args[args.index("--lets") + 1].split(",") if c] if "--lets" in args else []
    if "--format" in args and args[args.index("--format") + 1] != "6":
        print(f"--format {args[args.index('--format') + 1]}: the one format is 6")
        return 2
    families = [f for f in args[args.index("--families") + 1].split(",") if f] if "--families" in args else []
    if depth_text == "none":
        depth = None
    text = sniff6(args[0], entry, depth, master, elements, lets, families)
    if "--out" in args:
        with open(args[args.index("--out") + 1], "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
