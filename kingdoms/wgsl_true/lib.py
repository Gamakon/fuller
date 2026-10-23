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
BARE = {}       # scalar value type (and its w twin) -> its bare literal type (values.declare)
TWIN_A = {}     # value type (and its w twin) -> its a twin (slot phase), pool interfaces only
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

K = 1


def configure(k, epilogue=True, pools=False):
    """Set the fuel budget K of the interface about to be generated. The submodules read K and STRATA when they
    are imported (`from lib import *`), so this runs before they are imported. An interface with no primary output
    (a module of several entries, s9 batch 8) has no epilogue: its O rungs do not exist."""
    global K, STRATA
    K = k
    STRATA = range(0, K + 1)
    if not epilogue:
        DEPTH["O"] = range(0)
    if pools:
        # s9 batch 9: the pair phase's body (P, W plus the partner load) and the solo phase's body (S, one invocation)
        DEPTH["P"] = range(0, 6)
        DEPTH["S"] = range(0, 6)
        # the split phase's body (X): W, writing below the split and reading above it
        DEPTH["X"] = range(0, 6)
        # the swap-pair phase's body (E, UI1 batch 10): W's families with the pair rows in place of own writes
        DEPTH["E"] = range(0, 6)
        # the block-write phase's body (Y, UI1 batch 10): W's families with the block rows in place of own writes
        DEPTH["Y"] = range(0, 6)
        # the slot phase's body (A, UI1 batch 10): writes at slots one counter hands out, reads above the split
        DEPTH["A"] = range(0, 6)


DEPTH = {"L": range(0, 6), "C": range(0, 6), "O": range(0, 5), "W": range(0, 6), "P": range(0), "S": range(0), "X": range(0), "E": range(0), "Y": range(0), "A": range(0)}
STRATA = range(0, K + 1)


def S(x, d, k):
    return f"stmt.{x}{d}_f{k}"


def B(x, d, k):
    return f"block.{x}{d}_f{k}"


def BB(d, k):
    return f"block.B{d}_f{k}"


def WBB(d, k):
    return f"block.WB{d}_f{k}"


def XBB(d, k):
    return f"block.XB{d}_f{k}"


def ABB(d, k):
    return f"block.AB{d}_f{k}"


def SBB(d, k):
    return f"block.SB{d}_f{k}"


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
    """Declare the `w` twin of a value or helper type (and, in a pool interface, its `a` twin: a value that read no
    workgroup element at all, neither a foreign nor an own one, the values of the slot phase)."""
    w = base + "w"
    T(w, meaning or f"the write-phase twin of {base}: a value no foreign workgroup read reached", TYPES[base]["lanes"], "-")
    TWIN[base] = w
    if DEPTH["A"]:
        a = base + "a"
        T(a, f"the slot-phase twin of {base}: a value no workgroup read reached", TYPES[base]["lanes"], "-")
        TWIN_A[base] = a
        TWIN_A[w] = a
    return w


def w_of(t):
    return TWIN.get(t, t)


def a_of(t):
    return TWIN_A.get(t, t)


def fam(x, t):
    """The value type a statement of rung `x` takes for `t`: the `w` twin at the write-phase rungs, the `a` twin at the
    slot phase's, `t` itself elsewhere."""
    if x in "WPXEY":
        return w_of(t)
    if x == "A":
        return a_of(t)
    return t


def extra(iface, fam, base):
    """`base` followed by the interface's own constants of this family (append-only: the base list is unchanged, so an
    interface with none generates exactly the table it always did)."""
    out = list(base)
    for v in iface.get("extra_constants", {}).get(fam, []):
        if v not in out:
            out.append(v)
    return out


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
    # the a twins (slot phase): the same rows over the a family; no own read has one (an own read is not a twin row)
    for name, r in list(ROWS.items()):
        if not TWIN_A or not r["twin"] or r["out"] not in TWIN_A or name.endswith("_w"):
            continue
        if not all(a in TWIN_A or a in SAFE_ARGS for a in r["args"]):
            continue
        tn = name + "_a"
        if tn in ROWS:
            continue
        if r["values"] is not None:
            C(tn, TWIN_A[r["out"]], r["spell"], r["values"], note="the a twin of " + name)
        else:
            R(tn, TWIN_A[r["out"]], [a_of(a) for a in r["args"]], r["spell"], note="the a twin of " + name, fields=r["fields"])
        made += 1
    return made


# ---------------------------------------------------------------- bare literals

VALUE_FORMS = {"real", "index", "count", "int", "flag"}
VALUE_SLOT = re.compile(r"(f32|f16|u32|i32|bool)|vec[234]<(f32|f16|u32|i32|bool)>|mat[234]x[234]<f32>")


def is_value(t):
    """A value type of the families (scalar, vector, matrix; the w twins included): never a const-expression, because
    every literal that produces one is a runtime literal (`lit_*`) and every other producer reads a runtime name."""
    slot, form = t.rsplit(".", 1)
    return (form.removesuffix("w") in VALUE_FORMS or form.removesuffix("a") in VALUE_FORMS) and VALUE_SLOT.fullmatch(slot) is not None


# WGSL 8.8 and 8.10: an integer `/` or `%` whose divisor is a const-expression equal to zero, and a shift whose count is
# a const-expression at or past the bit width, are shader-creation errors whatever the other operand is.
INT_BARE = {"u32.bare_index", "u32.bare_count", "i32.bare"}


def finish_bare():
    """Every function row taking a scalar value type in slot i gets a twin `<row>_bare<i>` taking that family's bare
    literal there, when the row is never constant-evaluated or cannot fail if it is:

    * a statement row (an assignment, a store, a condition): a statement is not a const-expression, and the bare
      literal standing alone as its argument expression cannot fail;
    * an expression row with another argument of a value type: that argument is never a const-expression, so the row's
      expression is not one either and WGSL evaluates it at run time (8.1.1);

    except an integer divisor or shift count (INT_BARE): WGSL makes a constant zero divisor and a constant count at or
    past the width shader-creation errors even beside a runtime operand. One slot a twin: the others keep their types."""
    made = 0
    for name, r in list(ROWS.items()):
        if r["values"] is not None or not r["args"]:
            continue
        for i, a in enumerate(r["args"]):
            b = BARE.get(a)
            if b is None:
                continue
            stmt = r["out"].startswith("stmt.")
            other = any(is_value(x) for j, x in enumerate(r["args"]) if j != i)
            if not (stmt or other):
                continue
            if b in INT_BARE and re.search(r"(/|%|<<|>>)\s*\(*\{" + str(i) + r"\}", r["spell"]):
                continue
            # naga 30 reads a constant component of an integer vector constructor as a constant shift count or divisor
            # lane (measured 2026-10-22, the 1,000-gene loop: "Shift amount exceeds the bit width" on
            # `x >> vec4<u32>(v, 2097153u)`), so an integer vector is never built from a bare literal
            if b in INT_BARE and VALUE_SLOT.fullmatch(r["out"].rsplit(".", 1)[0]) and r["out"].startswith("vec"):
                continue
            args = list(r["args"])
            args[i] = b
            R(f"{name}_bare{i}", r["out"], args, r["spell"], note=f"{name} with argument {i} a bare literal", fields=r["fields"])
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
    problems += never_constant_problems()
    return problems, produced, consumed


# The types that may be const-expressions: the bare literals and the literal-only helper types spelled bare.
MAYBE_CONST = re.compile(r"u32\.(lit64|lit4096|litpow2_\d+|boundx\d+w?|boundu\d+|ktile|blk_row|blk_col|lanelit|lane|lanew|sgoff|pidx_\d+w?|wgidx_\d+w?|aidx_\w+)|f32\.nonneg|i32\.nonneg|grid\.shape|bounds<\w+>\.ordered")
RUNTIME_NAMES = {"gid", "lid", "wid", "lidx", "nwg", "sg_size", "sg_num", "sg_lane", "sg_id"}


def never_constant_problems():
    """The rule bare literals rest on (s9 batch 7): no value type is ever a const-expression. A row giving a value type
    whose arguments may all be constant must name something that is not: a runtime literal or other helper call, a
    declared variable, a builtin, a binding."""
    bare = set(BARE.values())
    runtime = set(HELPERS) | {re.match(r"var (\w+):", d).group(1) for d in DECLS} | RUNTIME_NAMES | BINDINGS
    out = []
    for n, r in ROWS.items():
        if not is_value(r["out"]) or any(not (a in bare or MAYBE_CONST.fullmatch(a)) for a in r["args"]):
            continue
        if not set(IDENT.findall(PH.sub(" ", r["spell"]).replace("{v}", " "))) & runtime:
            out.append(f"row {n} gives the value type {r['out']} and may be a const-expression: {r['spell']!r}")
    return out


def esc(s):
    return str(s).replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


IDENT = re.compile(r"\b[A-Za-z_]\w*\b")


def declares_of(spell):
    """The declarations (DECLS, in table order) of the kernel-scope variables a row's spelling names: the row-attached
    prologue fragment the renderer emits for each row in the expressed tree, deduplicated, in table order."""
    named = set(IDENT.findall(PH.sub(" ", spell).replace("{v}", " ")))
    return [d for d in DECLS if re.match(r"var (\w+):", d).group(1) in named]


BINDINGS = set()   # the interface's module-scope variable names (write_all)


def write_all(outdir, root, iface):
    BINDINGS.update(re.findall(r"var(?:<[^>]*>)?\s+(\w+)\s*:", iface["prologue_top"]))
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
        f.write("ref\tname\tkind\tout\targs\tspelling\tnote\tdeclares\n")
        for i, (n, r) in enumerate(ROWS.items()):
            kind = "constant" if r["values"] is not None else ("terminal" if not r["args"] else "function")
            f.write(f"F{i + 1:05d}\t{n}\t{kind}\t{r['out']}\t{';'.join(r['args'])}\t{esc(r['spell'])}\t{esc(r['note'])}\t{esc(chr(10).join(declares_of(r['spell'])))}\n")
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
        if iface.get("module"):
            # the normal genes' roots, in gene order: the fixed homeotic linker (design section 6) takes them
            f.write("gene_roots\t" + ";".join(f"kernel.{e}" for e, _ in iface["module"]) + "\n")
            f.write(f"linker\tkernel.module_{iface['name']}\n")
    return problems, produced
