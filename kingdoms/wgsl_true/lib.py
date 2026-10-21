"""Registries and writers of the true multi-typed WGSL symbol table.

A type is `slot.form` (the design's `dual`). A row is (name, output type,
ordered argument types, spelling). A row with no arguments is a terminal; a
terminal with `values` is the constant placeholder `?_T` of its output type
and the values are that type's append-only constants `C_T`.

Spelling templates (MultiTypedGEP_s9_WGSL_draft.md, the renderer only
substitutes, it never rewrites):
    {N}    the text of argument N
    {N:F}  field F (whitespace-separated) of argument N's text (a literal
           such as the grid shape `16 16`)
    {v}    the value text a constant terminal reads from its C_T entry
Every other brace is WGSL. Every argument is spelled exactly once (hole 8:
a row that spells an argument twice nests exponentially), except where a
row declares `fields=True` (it spells fields of a literal argument).
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

TYPES = {}      # name -> dict(meaning, lanes, literal)
ROWS = {}       # name -> dict(out, args, spell, note, values, twin, fields)
HELPERS = {}    # name -> WGSL text (prologue)
DECLS = []      # function-scope declarations, in order
PROLOGUE_TOP = []   # module-scope lines before the helpers
TWIN = {}       # base type -> w twin type
SAFE_ARGS = set()   # argument types a W twin may keep as they are

PH = re.compile(r"\{(\d+)(?::(\d+))?\}")


def T(name, meaning, lanes=1, literal="-"):
    if name in TYPES:
        return name
    assert "." in name, name
    TYPES[name] = dict(meaning=meaning, lanes=lanes, literal=literal)
    return name


def R(name, out, args, spell, note="", twin=False, fields=False):
    assert name not in ROWS, f"duplicate row {name}"
    assert out in TYPES, f"{name}: output type {out} is not declared"
    for a in args:
        assert a in TYPES, f"{name}: argument type {a} is not declared"
    used = [int(m.group(1)) for m in PH.finditer(spell)]
    if not fields:
        assert sorted(used) == list(range(len(args))), f"{name}: spelling {spell!r} uses arguments {used}, signature has {len(args)}"
    else:
        assert set(used) <= set(range(len(args))), f"{name}: spelling {spell!r}"
    assert "{v}" not in spell, f"{name}: a function spelling has no {{v}}"
    ROWS[name] = dict(out=out, args=list(args), spell=spell, note=note, values=None, twin=twin, fields=fields)
    return name


def C(name, out, spell, values, note="", twin=False):
    """The constant placeholder `?_T` of `out`: a terminal whose spelling reads {v}."""
    assert name not in ROWS, f"duplicate row {name}"
    assert out in TYPES, f"{name}: output type {out} is not declared"
    assert "{v}" in spell, f"{name}: a constant spelling has {{v}}"
    assert all(i == "v" for i in re.findall(r"\{(\w+)\}", spell)), spell
    ROWS[name] = dict(out=out, args=[], spell=spell, note=note, values=list(values), twin=twin, fields=False)
    return name


def helper(name, text):
    if name in HELPERS:
        assert HELPERS[name] == text, f"helper {name} defined twice differently"
    HELPERS[name] = text
    return name


def decl(text):
    DECLS.append(text)


# ---------------------------------------------------------------- statements

K = 2           # the interface's fuel budget: strata 0..=K (the root takes K)
DEPTH = {"L": range(0, 6), "C": range(0, 6), "O": range(0, 5), "W": range(0, 6)}
STRATA = range(0, K + 1)


def S(x, d, k):
    return f"stmt.{x}{d}_f{k}"


def B(x, d, k):
    return f"block.{x}{d}_f{k}"


def BB(d, k):
    return f"block.B{d}_f{k}"


def WBB(d, k):
    return f"block.WB{d}_f{k}"


def rungs(xs):
    """(X, d, k) for every statement rung of the kinds `xs`."""
    for x in xs:
        for d in DEPTH[x]:
            for k in STRATA:
                yield x, d, k


# ---------------------------------------------------------------- value types

def vt(s, form, lanes=1):
    return f"{s}.{form}" if lanes == 1 else f"vec{lanes}<{s}>.{form}"


def register_twin(base, meaning=None):
    """Declare the `w` twin of a value or helper type."""
    w = base + "w"
    T(w, meaning or f"the write-phase twin of {base}: a value no foreign workgroup read reached", TYPES[base]["lanes"], "-")
    TWIN[base] = w
    return w


def w_of(t):
    return TWIN.get(t, t)


def finish_twins():
    """Generate the twin of every row marked twin=True: same spelling over the w family.
    A twin exists only when its output has a twin type and every argument is a twin type
    or a type that can carry no foreign read (SAFE_ARGS)."""
    made = 0
    for name, r in list(ROWS.items()):
        if not r["twin"] or r["out"] not in TWIN:
            continue
        if not all(a in TWIN or a in SAFE_ARGS or a.endswith("w") and a in TYPES and a[:-1] in TWIN for a in r["args"]):
            continue
        args = [w_of(a) for a in r["args"]]
        tn = name + "_w"
        if tn in ROWS:
            continue
        if r["values"] is not None:
            C(tn, TWIN[r["out"]], r["spell"], r["values"], note="the w twin of " + name)
        else:
            R(tn, TWIN[r["out"]], args, r["spell"], note="the w twin of " + name, fields=r["fields"])
        made += 1
    return made


# ---------------------------------------------------------------- literals

def f32(x):
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]


def f16(x):
    import struct
    return struct.unpack("e", struct.pack("e", x))[0]


def ftext(x):
    """A finite float as WGSL text an f32 holds exactly (abstract float)."""
    s = repr(float(x))
    if "e" not in s and "." not in s:
        s += ".0"
    return s


# ---------------------------------------------------------------- checks

def check():
    problems = []
    produced = {}
    consumed = set()
    for n, r in ROWS.items():
        produced.setdefault(r["out"], []).append(n)
        consumed.update(r["args"])
    for t in sorted(consumed):
        terms = [n for n in produced.get(t, []) if not ROWS[n]["args"]]
        if not terms:
            problems.append(f"type {t} is an argument type and has no terminal")
    return problems, produced, consumed


def esc(s):
    return str(s).replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def write_all(outdir, root, iface):
    problems, produced, consumed = check()
    names = [re.match(r"var (\w+):", d).group(1) for d in DECLS]
    for n in set(names):
        if names.count(n) > 1:
            problems.append(f"declaration {n} is declared {names.count(n)} times")
    tnames = list(TYPES)
    with open(os.path.join(outdir, "types.tsv"), "w") as f:
        f.write("ref\tdual\tslot\tform\tlanes\tmeaning\tliteral\n")
        for i, t in enumerate(tnames):
            slot, form = t.rsplit(".", 1)
            m = TYPES[t]
            f.write(f"T{i + 1:04d}\t{t}\t{slot}\t{form}\t{m['lanes']}\t{esc(m['meaning'])}\t{m['literal']}\n")
    with open(os.path.join(outdir, "functions.tsv"), "w") as f:
        f.write("ref\tname\tkind\tout\targs\tspelling\tnote\n")
        for i, (n, r) in enumerate(ROWS.items()):
            kind = "constant" if r["values"] is not None else ("terminal" if not r["args"] else "function")
            f.write(f"F{i + 1:05d}\t{n}\t{kind}\t{r['out']}\t{';'.join(r['args'])}\t{esc(r['spell'])}\t{esc(r['note'])}\n")
    with open(os.path.join(outdir, "constants.tsv"), "w") as f:
        f.write("type\tindex\ttext\n")
        for n, r in ROWS.items():
            if r["values"] is not None:
                for j, v in enumerate(r["values"]):
                    f.write(f"{r['out']}\t{j}\t{esc(v)}\n")
    with open(os.path.join(outdir, "prologue.wgsl"), "w") as f:
        f.write(iface["prologue_top"])
        f.write("\n".join(PROLOGUE_TOP) + "\n")
        for n in sorted(HELPERS):
            f.write(HELPERS[n].rstrip("\n") + "\n")
    with open(os.path.join(outdir, "declarations.wgsl"), "w") as f:
        f.write("\n".join(DECLS) + "\n")
    with open(os.path.join(outdir, "interface.tsv"), "w") as f:
        f.write("key\tvalue\n")
        f.write(f"root\t{root}\n")
        f.write(f"fuel_budget\t{K}\n")
        f.write(f"interface\t{iface['name']}\n")
    return problems, produced
