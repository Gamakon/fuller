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
* the element type is the one element of the parameters and the result (one
  element a view until step 4 of docs/PLAN_futhark_kingdom.md);
* an optional FAMILY of the master is admitted when its construct appears in
  the type-checked program's own text: `scan`; `hist` or `reduce_by_index`;
  the modulus `%`;
* the depth, the literal pool and the master version are the master's
  defaults.

It writes the specification `view.tsv` (the WHERE clause of the view, see
`view.py`), replacing the hand-written `interface.tsv`.

    python3 kingdoms/futhark/sniff.py <source.fut> [--entry main] [--out view.tsv]
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
FAMILY_CONSTRUCTS = {"scan": [r"\bscan\b"], "hist": [r"\bhist\b", r"\breduce_by_index\b"], "mod": [r"%"]}
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


def reachable_source(path, entry):
    """The problem's own text the entry reaches: the entry's definition and every top-level definition it names,
    transitively, comments stripped. The prelude is not the problem's text, so it is never read."""
    with open(path) as f:
        src = "\n".join(re.sub(r"--.*", "", l) for l in f.read().splitlines())
    defs, cur = {}, None
    for line in src.splitlines():
        m = re.match(r"^(?:entry|def|let)\s+([A-Za-z_][A-Za-z0-9_']*)", line)
        if m:
            cur = m.group(1)
            defs[cur] = ""
        if cur is not None:
            defs[cur] += line + "\n"
    if entry not in defs:
        raise ValueError(f"{path} has no definition {entry}")
    seen, todo = set(), [entry]
    while todo:
        d = todo.pop()
        if d in seen:
            continue
        seen.add(d)
        todo += [n for n in re.findall(r"[A-Za-z_][A-Za-z0-9_']*", defs[d]) if n in defs and n not in seen]
    return "".join(defs[d] for d in defs if d in seen)


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


def sniff(path, entry="main"):
    """The view specification of a problem, as text."""
    check_compiler()
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
    elems = {e for n, dims, e in typed if n not in size_inputs} | {out_elem}
    if len(elems) != 1 or next(iter(elems)) not in ELEMENTS:
        raise ValueError(f"the problem has the element types {sorted(elems)}: a view holds one (step 4 adds more)")
    elem = next(iter(elems))
    lines = ["# kind\tname\telement\tdims",
             f"# written by kingdoms/futhark/sniff.py from {os.path.relpath(os.path.abspath(path), REPO)}, entry {entry}",
             f"master\t{view.master()['version']}\t\t",
             f"element\t{elem}\t\t"]
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
    lines.append(f"depth\t{DEFAULT_DEPTH}\t\t")
    user = reachable_source(path, entry)
    fams = [f for f in view.SIGNATURE_FAMILIES if any(re.search(p, user) for p in FAMILY_CONSTRUCTS[f])]
    lines.append(f"families\t{','.join(fams)}\t\t")
    lines.append(f"literals\t{','.join(view.master()['pools'][elem])}\t\t")
    lines.append(f"dims\t{','.join(dims_seen)}\t\t")
    return "\n".join(lines) + "\n"


def main(argv):
    args = argv[1:]
    if not args or args[0].startswith("-"):
        print(__doc__.strip().splitlines()[-1])
        return 2
    entry = args[args.index("--entry") + 1] if "--entry" in args else "main"
    text = sniff(args[0], entry)
    if "--out" in args:
        with open(args[args.index("--out") + 1], "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
