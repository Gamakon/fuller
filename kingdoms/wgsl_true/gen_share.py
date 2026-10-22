#!/usr/bin/env python3
"""ST7 share rows of the true WGSL kingdom (docs/PLAN_true_gep_wgsl.md, ST7).

A separate generator section from gen_tables.py: it reads the generated table
(types.tsv, functions.tsv) and writes the row family the share pass of
src/wgsl/true_share.rs uses to spell a subtree a kernel repeats once, as a WGSL
`let`, and to reference it by name:

  share.name                 constant   out share.name      the binding's name, {v} from its C_T
  share.let.<dual>.<body>    function   out <body>          let {0}: <slot> = {1};\\n{2}
                                         args share.name; <dual>; <body>
  share.ref.<dual>           function   out <dual>          {0}
                                         args share.name

<dual> is every value dual (slot a WGSL scalar, vector or matrix type); <body>
is the type of the root row's first argument spelled after <declarations>, the
start of the entry point's body, so every let is at function scope ahead of
every in-body use. The rows are spellings only: the renderer substitutes
argument texts and does nothing else. They are not in the evolution alphabet
(the kingdom contract is functions.tsv alone); the share pass writes them into
an expressed tree and the unshare pass removes them, and the tests assert the
round trip is the identity.

Writes share.tsv (the functions.tsv header) and share_constants.tsv (the
constants.tsv header; share.name's C_T, append-only).

  python3 kingdoms/wgsl_true/gen_share.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = 4096
NAME_PREFIX = "st7_s"
SCALARS = ("bool", "i32", "u32", "f32", "f16")


def value_slot(slot):
    if slot in SCALARS:
        return True
    m = re.fullmatch(r"vec([234])<(\w+)>", slot)
    if m:
        return m.group(2) in SCALARS
    return re.fullmatch(r"mat[234]x[234]<f(32|16)>", slot) is not None


def read_tsv(name):
    with open(os.path.join(HERE, name)) as f:
        lines = f.read().split("\n")
    header = lines[0].split("\t")
    rows = [dict(zip(header, l.split("\t"))) for l in lines[1:] if l]
    return header, rows


def body_types(functions, root):
    """For each row producing the root: the type of its first argument spelled after <declarations>."""
    out = []
    for r in functions:
        if r["out"] != root:
            continue
        spelling = r["spelling"].replace("\\n", "\n")
        at = spelling.find("<declarations>")
        if at < 0:
            sys.exit(f"our software has a bug: root row {r['name']} has no <declarations>, so no function body to put a let in")
        args = r["args"].split(";") if r["args"] else []
        placed = sorted((m.start(), int(m.group(1))) for m in re.finditer(r"\{(\d+)(?::\d+)?\}", spelling) if m.start() > at)
        if not placed:
            sys.exit(f"our software has a bug: root row {r['name']} spells no argument in its body")
        out.append(args[placed[0][1]])
    return sorted(set(out))


def main():
    _, types = read_tsv("types.tsv")
    fheader, functions = read_tsv("functions.tsv")
    _, meta = read_tsv("interface.tsv")
    root = next(m["value"] for m in meta if m["key"] == "root")
    bodies = body_types(functions, root)
    duals = [t for t in types if value_slot(t["slot"])]
    rows = [("share.name", "constant", "share.name", "", "{v}", "the name a let binds: st7_s<k>, never a value")]
    for t in duals:
        for b in bodies:
            rows.append((f"share.let.{t['dual']}.{b}", "function", b, f"share.name;{t['dual']};{b}", f"let {{0}}: {t['slot']} = {{1}};\\n{{2}}", f"binds a {t['dual']} subtree the kernel repeats, at function scope ahead of every in-body use"))
        rows.append((f"share.ref.{t['dual']}", "function", t["dual"], "share.name", "{0}", f"the value of a {t['dual']} let, by its name"))
    with open(os.path.join(HERE, "share.tsv"), "w") as f:
        # the share rows declare no variable, so share.tsv keeps the seven columns before functions.tsv's `declares`
        f.write("\t".join(fheader[:7]) + "\n")
        for i, r in enumerate(rows):
            f.write("\t".join((f"S{i + 1:05d}",) + r) + "\n")
    with open(os.path.join(HERE, "share_constants.tsv"), "w") as f:
        f.write("type\tindex\ttext\n")
        for k in range(NAMES):
            f.write(f"share.name\t{k}\t{NAME_PREFIX}{k}\n")
    print(f"share.tsv: {len(rows)} rows ({len(duals)} value duals x {len(bodies)} body types + refs + name); share_constants.tsv: {NAMES} names")


if __name__ == "__main__":
    main()
