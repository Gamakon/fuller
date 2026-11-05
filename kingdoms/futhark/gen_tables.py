#!/usr/bin/env python3
"""The Futhark kingdom's generator: the tables written as code, and the witnesses that check them.

A gene's types are DATA TYPES (`f32`, `[]f32`, `[][]f32`, `i64`, `size`, tuples, function types). An array's length
and an index's bound are PROPERTIES, scope is the ENVIRONMENT the decoder builds, and where a requirement fails the
rendered program carries an ADAPTER (`docs/DESIGN_futhark_types_and_properties.md`; the master's format in
`master/README.md`).

The master (`master/6/`) is checked by `master6_main`: its witnesses, its static cost table (`master/6/costs.tsv`,
generated here), and each problem's view, whose specification the sniffer writes (`sniff.py --format 6`) and whose
tables `view.py` writes (`View6`), committed under `sig/<problem>/m6/` and compared byte for byte. The pixel family
(`pixel_rows.py`: one master kingdom `futhark.pixel.<shader>` a shader, its interface and e-graph under
`pixel/<shader>/`) is generated and checked by `pixel_rows_main`.

The witness tests are the reason to trust the tables: they re-read the emitted text and re-check it against models
written apart from the expansion code, so a mis-declared row cannot pass silently.

    python3 kingdoms/futhark/gen_tables.py [--check | --write]
"""

import os
import re
import sys

import view as master_view  # noqa: E402
import sniff  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "master"))
import write_rows  # noqa: E402  (kingdoms/master/write_rows.py: the one writer of every kingdom's rows)

# The parent kingdom's row in `kingdoms/master/kingdoms.tsv`; each problem's view is a variant `futhark.<problem>`.
FUTHARK_DESCRIPTION = ("the Futhark kingdom: shape-polymorphic GPU kernels; each problem is a variant, its rows computed by "
                       "view.py from the master templates (kingdoms/futhark/master/6) and its sniffed view.tsv")

PLACEHOLDER_RE = re.compile(r"\{(\d+)\}")

HERE = os.path.dirname(os.path.abspath(__file__))


def first_difference(got, want):
    """Where two byte strings first differ, for the report."""
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            line = got.count(b"\n", 0, i) + 1
            return f"line {line} differs"
    if len(got) != len(want):
        return f"the lengths differ ({len(got)} and {len(want)})"
    return "they are equal"


# --------------------------------------------------------------------------
# THE PIXEL FAMILY: a picture is a map over the pixels of a scalar program. Its rows are `pixel_rows.py` (shape types,
# scope by var.f32, the loop bound by env.trips; plan 7.7 K4); this section reads each instance's shader (its uniform
# interface and its float literals) and writes the instance's rows into the master tables, its interface and its
# e-graph under pixel/<instance>/.
# --------------------------------------------------------------------------

# The generic part of the constant pool, before the instance's own literals.
PIXEL_GENERIC_LITERALS = ("0.0", "1.0", "-1.0", "0.5", "2.0", "3.0", "0.25", "10.0", "0.1")

# An image target's own literals (it has no shader to read them from): the tenths of the unit interval, so a colour
# channel is a constant of the pool.
PIXEL_IMAGE_LITERALS = ("0.2", "0.3", "0.4", "0.6", "0.7", "0.8", "0.9")

# The instances: the WGSL source of each, relative to the repository root.
PIXEL_INSTANCES = {
    "cuneus": "tests/corpus/cuneus_visual/cuneus.wgsl",
    # An IMAGE TARGET: no shader, no uniforms, no time -- a picture is a function of the pixel's position and the
    # picture's size. Any picture file is then a target with no hand work (the runner loads it).
    "image": None,
    "volumepassage": "tests/corpus/cuneus_visual/volumepassage.wgsl",
}

REPO = os.path.dirname(os.path.dirname(HERE))

STRUCT_RE = re.compile(r"struct\s+(\w+)\s*\{([^}]*)\}", re.S)
FIELD_RE = re.compile(r"(\w+)\s*:\s*([\w<>]+)")
UNIFORM_RE = re.compile(r"@group\((\d+)\)\s*@binding\((\d+)\)\s*var<uniform>\s+(\w+)\s*:\s*(\w+)\s*;")
FLOAT_RE = re.compile(r"(?<![\w.])-?\d+\.\d+(?:[eE][-+]?\d+)?")


def strip_comments(src):
    return re.sub(r"//[^\n]*", "", src)


def pixel_interface(shader_path):
    """The interface of an instance, read from its WGSL source: every f32
    field of every uniform block, in declaration order, with the block's
    group and binding and the field's word offset under WGSL's uniform layout
    (a scalar is 4 bytes aligned to 4, a vec2 8 aligned to 8, a vec3 12 and a
    vec4 16 aligned to 16). A `vecN<f32>` field is N inputs, `name_x`,
    `name_y`, `name_z`, `name_w`. A field whose name starts with `_` is padding
    and is not an input. Any other field type is a finding: the offsets would
    be wrong."""
    src = strip_comments(open(shader_path).read())
    structs = {m.group(1): FIELD_RE.findall(m.group(2)) for m in STRUCT_RE.finditer(src)}
    layout = {"f32": (4, 4, 1), "u32": (4, 4, 0), "i32": (4, 4, 0),
              "vec2<f32>": (8, 8, 2), "vec3<f32>": (12, 16, 3), "vec4<f32>": (16, 16, 4)}
    out = []
    for g, b, _var, ty in UNIFORM_RE.findall(src):
        if ty not in structs:
            raise ValueError(f"the uniform block type {ty} has no struct in {shader_path}")
        offset = 0
        for name, fty in structs[ty]:
            if fty not in layout:
                raise ValueError(f"the uniform field {ty}.{name} is a {fty}, which the interface does not lay out")
            size, align, lanes = layout[fty]
            offset = (offset + align - 1) // align * align
            if not name.startswith("_"):
                if fty == "f32":
                    out.append({"name": name, "group": int(g), "binding": int(b), "word": offset // 4, "spelling": f"u_{name}"})
                for i in range(lanes if fty.startswith("vec") else 0):
                    lane = "xyzw"[i]
                    out.append({"name": f"{name}.{lane}", "group": int(g), "binding": int(b), "word": offset // 4 + i,
                                "spelling": f"u_{name}_{lane}"})
            offset += size
    names = [f["spelling"] for f in out]
    if len(set(names)) != len(names):
        raise ValueError(f"two uniform fields share a name in {shader_path}: {names}")
    return out


def f32_bits(text):
    import struct
    return struct.unpack("<I", struct.pack("<f", float(text)))[0]


def pixel_literals(shader_path):
    """The constant pool: the generic literals, then every float literal of
    the shader's source in order of appearance, each value once (by its f32
    bits), each spelled as an f32 literal."""
    src = strip_comments(open(shader_path).read())
    seen, out = set(), []
    for text in list(PIXEL_GENERIC_LITERALS) + FLOAT_RE.findall(src):
        bits = f32_bits(text)
        if bits in seen:
            continue
        seen.add(bits)
        out.append(f"{text}f32")
    return out


def pixel_dir(instance):
    return os.path.join(HERE, "pixel", instance)


# --------------------------------------------------------------------------
# THE E-GRAPH RULES' WITNESS: both sides of every rule of `master/6/rules.tsv` agree on random values.
# --------------------------------------------------------------------------

def _wrap(x):
    return (x + 2**31) % 2**32 - 2**31


def _f32(x):
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]


def _fmin(x, y):
    if x != x:
        return y
    if y != y or x < y:
        return x
    if y < x:
        return y
    import math
    return x if math.copysign(1.0, x) < 0 else y


def _fmax(x, y):
    if x != x:
        return y
    if y != y or x > y:
        return x
    if y > x:
        return y
    import math
    return x if math.copysign(1.0, x) > 0 else y


RULE_SEMANTICS = {
    "i32": {
        "add": lambda a, b: _wrap(a + b), "sub": lambda a, b: _wrap(a - b), "mul": lambda a, b: _wrap(a * b),
        "min": min, "max": max, "neg": lambda a: _wrap(-a),
        "map2.add": lambda a, b: [_wrap(x + y) for x, y in zip(a, b)], "map2.min": lambda a, b: [min(x, y) for x, y in zip(a, b)],
        "map2.max": lambda a, b: [max(x, y) for x, y in zip(a, b)], "map2.sub": lambda a, b: [_wrap(x - y) for x, y in zip(a, b)],
        "map2.mul": lambda a, b: [_wrap(x * y) for x, y in zip(a, b)],
    },
    "f32": {
        "add": lambda a, b: _f32(a + b), "sub": lambda a, b: _f32(a - b), "mul": lambda a, b: _f32(a * b),
        "min": _fmin, "max": _fmax, "neg": lambda a: -a,
        "map2.add": lambda a, b: [_f32(x + y) for x, y in zip(a, b)], "map2.min": lambda a, b: [_fmin(x, y) for x, y in zip(a, b)],
        "map2.max": lambda a, b: [_fmax(x, y) for x, y in zip(a, b)], "map2.sub": lambda a, b: [_f32(x - y) for x, y in zip(a, b)],
        "map2.mul": lambda a, b: [_f32(x * y) for x, y in zip(a, b)],
    },
}


def _matmul_f32(a, b):
    """`matmul` of the prelude (`src/futhark.rs`, MATMUL_HELPER): each cell the f32 sum, in order, of the products."""
    out = []
    for row in a:
        cells = []
        for j in range(len(b[0])):
            acc = 0.0
            for k, x in enumerate(row):
                acc = _f32(acc + _f32(x * b[k][j]))
            cells.append(acc)
        out.append(cells)
    return out


# The matrix family's rows (master 6's `mat` rules, A7's e-graph): the row functions are values, so a function row of
# no argument evaluates to the function it spells.
RULE_SEMANTICS["f32"].update({
    "map_rows": lambda f, m: [f(r) for r in m], "map2_rows": lambda f, a, b: [f(x, y) for x, y in zip(a, b)],
    "transpose": lambda m: [list(c) for c in zip(*m)], "matmul": _matmul_f32,
    "id@fn.arr_f32_arr_f32": lambda: (lambda xs: xs), "id@fn.arr2_f32_arr_f32_arr_f32": lambda: (lambda xs, _ys: xs),
})
# The tuple rows' meanings (any element): a pair or a triple is built, and its parts are its components.
for _sem in RULE_SEMANTICS.values():
    _sem.update({"pair": lambda a, b: (a, b), "triple": lambda a, b, c: (a, b, c), "fst": lambda t: t[0], "snd": lambda t: t[1], "thd": lambda t: t[2]})


def _rule_eval(text, env, semantics):
    """Evaluate one side of a rule template: `({op} x y)` with `{op}` an operator of `semantics`."""
    toks = text.replace("(", " ( ").replace(")", " ) ").split()

    def at(i):
        if toks[i] != "(":
            return env[toks[i]], i + 1
        op = toks[i + 1]
        m = re.fullmatch(r"\{(.+)\}", op)
        if not m or m.group(1) not in semantics:
            raise ValueError(f"the operator {op}")
        args, j = [], i + 2
        while toks[j] != ")":
            v, j = at(j)
            args.append(v)
        return semantics[m.group(1)](*args), j + 1

    v, end = at(0)
    if end != len(toks):
        raise ValueError(f"text after the term: {toks[end:]}")
    return v


def _same_value(elem, x, y):
    """The comparison policy of the kingdom's values (src/futhark_sig.rs, `Array`): the same bits, or two NaNs
    (an f32 NaN's payload is not a value); -0.0 and +0.0 differ."""
    if isinstance(x, (list, tuple)):
        return type(x) is type(y) and len(x) == len(y) and all(_same_value(elem, a, b) for a, b in zip(x, y))
    if elem == "i32":
        return x == y
    import struct
    if x != x and y != y:
        return True
    return struct.pack("f", x) == struct.pack("f", y)


def witness_rules(mst):
    """Both sides of every rule agree on random values of its element, with the operators' own meaning and under the
    kingdom's comparison policy (bit for bit; two NaNs agree): a rule that changed a value, a sign of a zero included,
    would let the e-graph justify a candidate that computes something else."""
    import random
    bad = []
    rnd = random.Random(20261101)
    edges = {"i32": [0, 1, -1, 2**31 - 1, -2**31, 7, -40],
             "f32": [0.0, -0.0, float("nan"), float("inf"), float("-inf"), 1.0, -1.0, 3e38, 1e-45, 2.5, 7.0, -40.0, 16777217.0]}
    for r in mst["rules"]:
        elems = r["elems"].split(",")
        if any(e not in RULE_SEMANTICS for e in elems):
            bad.append(f"M9 the rule {r['rule']!r} is admitted for {r['elems']}: the witness knows {sorted(RULE_SEMANTICS)}")
            continue
        m = re.fullmatch(r"\(rewrite (.+) :ruleset futhark_sig\)", r["rule"])
        if not m or r["on"] not in ("elem", "arr", "mat", "tuple", "tuple3"):
            bad.append(f"M9 the rule {r['rule']!r} is not a rewrite of the futhark_sig ruleset on elem, arr, mat, tuple or tuple3")
            continue
        body = m.group(1)
        depth, cut = 0, None
        for i, c in enumerate(body):
            depth += {"(": 1, ")": -1}.get(c, 0)
            if depth == 0 and c == ")":
                cut = i + 1
                break
        lhs, rhs = body[:cut], body[cut:].strip()
        for elem in elems:
            sem, ed = RULE_SEMANTICS[elem], edges[elem]

            def draw(trial):
                if trial < 3 * len(ed):
                    return ed[trial % len(ed)]
                if elem == "i32":
                    return rnd.randint(-2**31, 2**31 - 1)
                return _f32(rnd.uniform(-1e3, 1e3)) if rnd.random() < 0.8 else rnd.choice(ed)

            for trial in range(300):
                if r["on"] in ("elem", "tuple", "tuple3"):
                    # Every edge pairs with every edge once (the signed zeros and the NaNs together are the cases).
                    if trial < len(ed) ** 2:
                        env = {"a": ed[trial // len(ed)], "b": ed[trial % len(ed)], "d": ed[(trial * 7) % len(ed)]}
                    else:
                        env = {v: draw(trial) for v in "abd"}
                elif r["on"] == "mat":
                    # A matrix rule's operands: `a` and `d` of one shape, `b` the shape `a` multiplies (3 x 4, 4 x 2).
                    env = {v: [[draw(rnd.randrange(300)) for _ in range(cols)] for _ in range(rows)]
                           for v, rows, cols in (("a", 3, 4), ("d", 3, 4), ("b", 4, 2))}
                else:
                    env = {v: [draw(rnd.randrange(300)) for _ in range(5)] for v in "abd"}
                try:
                    if not _same_value(elem, _rule_eval(lhs, env, sem), _rule_eval(rhs, env, sem)):
                        bad.append(f"M9 the rule {r['rule']!r} changes a value on {elem} at {env}")
                        break
                except (ValueError, KeyError, IndexError) as e:
                    bad.append(f"M9 the rule {r['rule']!r} does not evaluate on {elem}: {e}")
                    break
    return bad


# --------------------------------------------------------------------------
# MASTER 6 (`master/6/`, the format in `master/README.md`; plan section 7, task F1a). A type is a data type; lengths
# and bounds are properties; scope is the environment; a failing requirement renders an adapter.
#
# The witnesses below read `master/6/` and each view's WRITTEN text, apart from the expansion in `view.py`, and check
# them against models restated here: every data type but `size` has a constant; every requirement names an adapter
# that repairs its relation; every rule, requirement and binding names properties that exist; no row returns `size`;
# the work bound; the binders and their bindings; the index templates.
# --------------------------------------------------------------------------

# THE MASTER 6 PROBLEMS: (source, entry, depth or None, master version pin, elements added, lets, families). Every
# view's specification is written by `sniff.py --format 6` from the problem's Futhark source, and its tables by
# `view.py`; both are committed under `sig/<problem>/m6/` and checked byte for byte. A pin is a row version (the rows
# keep the versions they were added at). A7 is the matrix family alone, with no work bound (no `depth` line).
SIG6_PROBLEMS = {
    "a7": ("tests/corpus/futhark_problems/a7/yardstick.fut", "main", None, 1, (), (), ("matrix",)),
    "histogram": ("tests/corpus/futhark_problems/histogram/yardstick.fut", "main", 2, 2, (), (), ()),
    "kmeans": ("tests/corpus/futhark_problems/kmeans/yardstick.fut", "main", 3, 5, (), ("[]f32",), ()),
    # The faithful k-means view (plan 7, F1b): the assignments an `i32` array beside the `f32` points, its kernel
    # boundary `[]i32` (with sizes and scope in the type, as before master 6, it was 237,325 types and did not load).
    "kmeans_i32": ("tests/corpus/futhark_problems/kmeans/yardstick.fut", "main", 3, 5, ("i32",), ("[]i32",), ()),
    "pathfinder": ("tests/corpus/futhark_problems/pathfinder/yardstick.fut", "main", 2, 2, (), (), ()),
    "reduce": ("tests/corpus/futhark_problems/reduce/yardstick.fut", "main", 2, 2, (), (), ()),
    "scan": ("tests/corpus/futhark_problems/scan/yardstick.fut", "main", 2, 2, (), (), ()),
}
SIG6_FILES = ("view.tsv", "egraph.datatype", "egraph.rules", "egraph.rows", "egraph.sorts")  # the rows are in kingdoms/master/
# The operations whose rows are heavy (a work bound `env.trips < depth` substitutes them): the binders, the reductions,
# the scans and histograms, the matrix family's and the fitting adapters.
HEAVY6_OPS = {"tab", "tabrows", "replicate", "sum", "minimum", "maximum", "map2", "scan", "hist", "histrows", "rowdiv",
              "loop", "loop1", "map_rows", "map2_rows", "matmul", "fit", "fitrows", "fitcols"}
# `subst.const` substitutes its constant, for the rows whose own-type inputs are their indices and values, not a result
# carried through (`hist.*`, `histrows.*`: the constant fitted to the slot is the histogram with nothing counted).
SUBST_CONST6_OPS = {"hist", "histrows"}


def subst6(op):
    """The substitution list of a heavy row's work bound, by its operation (`hist`, `map2`, ...)."""
    return "subst.const" if op in SUBST_CONST6_OPS else "subst"


def work_bound6(op):
    """A heavy row's work-bound guard, as the master writes it."""
    return f"must env.trips < depth -> {subst6(op)}"


# The data type templates and their properties (README 2 and 3).
DATATYPES6 = {"{E}": ("scalar", "", "E", "{E}", "0{E}", ""), "i64": ("scalar", "", "", "i64", "0i64", "bound=1"),
              "size": ("scalar", "", "", "size", "", ""), "[]{E}": ("array", "{E}", "E", "a.{E}", "[0{E}]", "len=1"),
              "[][]{E}": ("array", "[]{E}", "E", "m.{E}", "[[0{E}]]", "rows=1 & cols=1"),
              "({A}, {B})": ("tuple", "{A};{B}", "A,B:elements", "{A}.{B}", "(0{A}, 0{B})", ""),
              "({A}, {B}, {C})": ("tuple", "{A};{B};{C}", "A,B,C:elements", "{A}.{B}.{C}", "(0{A}, 0{B}, 0{C})", ""),
              "[]f32 -> []f32": ("function", "[]f32 -> []f32", "", "fn.arr_f32_arr_f32", "\\xs -> map (\\_ -> 0.0) xs", ""),
              "[]f32 -> []f32 -> []f32": ("function", "[]f32;[]f32 -> []f32", "", "fn.arr2_f32_arr_f32_arr_f32",
                                          "\\xs _ys -> map (\\_ -> 0.0) xs", "")}
PROPERTIES6 = {("[]{E}", "len"): "size", ("[][]{E}", "rows"): "size", ("[][]{E}", "cols"): "size",
               ("i64", "bound"): "bound", ("size", "value"): "size"}
# The binders restated: op -> (args, flows, spelling): each binder's variables, their properties and the trip.
BINDERS6 = {
    ("tab", ""): ("size;{E}", "out.len := in0.value; in1: var i64 with bound := in0.value; in1: env.trips := env.trips + 1", "tabulate {0} (\\{1.v0} -> ({1}))"),
    ("tabrows", ""): ("size;size;[]{E}", "out.rows := in0.value; out.cols := in1.value; in2: var i64 with bound := in0.value; in2: env.trips := env.trips + 1", "tabulate {0} (\\{2.v0} -> ({2}))"),
    ("loop", "S:states"): ("size;{S};{S}", "out.len := in1.len; in2: var {S} with len := in1.len; in2: var i64 with bound := in0.value; in2: env.trips := env.trips + 1",
                           "loop ({2.v0}: {TYPE1}) = ({1}) for {2.v1} < {0} do ({2})"),
    ("loop1", "S:states"): ("size;{S};{S}", "out.len := in1.len; in2: var {S} with len := in1.len; in2: var i64 with bound := in0.value; in2: env.trips := env.trips + 1",
                            "loop ({2.v0}: {TYPE1}) = ({1}) for t{2.v1} < {0} - 1 do let {2.v1} = t{2.v1} + 1 in ({2})"),
    ("loop", "S:tuples"): ("size;{S};{S}", "in2: var {S.a}; in2: var {S.b}; in2: var i64 with bound := in0.value; in2: env.trips := env.trips + 1",
                           "loop (({2.v0}, {2.v1}): {TYPE1}) = ({1}) for {2.v2} < {0} do ({2})"),
    ("loop1", "S:tuples"): ("size;{S};{S}", "in2: var {S.a}; in2: var {S.b}; in2: var i64 with bound := in0.value; in2: env.trips := env.trips + 1",
                            "loop (({2.v0}, {2.v1}): {TYPE1}) = ({1}) for t{2.v2} < {0} - 1 do let {2.v2} = t{2.v2} + 1 in ({2})"),
    ("loop", "S:tuples3"): ("size;{S};{S}", "in2: var {S.a}; in2: var {S.b}; in2: var {S.c}; in2: var i64 with bound := in0.value; in2: env.trips := env.trips + 1",
                            "loop (({2.v0}, {2.v1}, {2.v2}): {TYPE1}) = ({1}) for {2.v3} < {0} do ({2})"),
    ("loop1", "S:tuples3"): ("size;{S};{S}", "in2: var {S.a}; in2: var {S.b}; in2: var {S.c}; in2: var i64 with bound := in0.value; in2: env.trips := env.trips + 1",
                             "loop (({2.v0}, {2.v1}, {2.v2}): {TYPE1}) = ({1}) for t{2.v3} < {0} - 1 do let {2.v3} = t{2.v3} + 1 in ({2})"),
    # The kernel boundary: a `let` binds its argument 1 with no trip and no work bound, anywhere.
    ("let", "S:lets;Y:all"): ("{S};{Y}", "out.len := in1.len; out.rows := in1.rows; out.cols := in1.cols; out.bound := in1.bound; in1: var {S} with len := in0.len & rows := in0.rows & cols := in0.cols", "let {1.v0} = ({0}) in ({1})"),
}
# The adapters restated (design section 5): name -> (args, flows, spelling, prelude).
ADAPTERS6 = {
    "fit@a.{E}": ("size;[]{E}", "out.len := in0.value", "fit {0} 0{E} ({1})", "fit"),
    "fitrows@m.{E}": ("size;[][]{E}", "out.rows := in0.value; out.cols := in1.cols", "fitrows {0} 0{E} ({1})", "fitrows"),
    "fitcols@m.{E}": ("size;[][]{E}", "out.rows := in1.rows; out.cols := in0.value", "map (fit {0} 0{E}) ({1})", "fit"),
    "clampix@i64": ("size;i64", "out.bound := in0.value", "clampix {0} ({1})", "clampix"),
}
ROW6_KINDS = {"function", "terminal", "const", "var", "adapter"}
# The repair lists restated (design 3.3.3 and 5): each list's sites and the adapters its templates may call.
REPAIRS6 = {"clampix": ["in{S}"], "fit": ["in{S}"], "fit.against": ["in{J}"], "fit.pair": ["node"], "fit.root": ["root"],
            "subst": ["node", "node"], "subst.const": ["node"]}
FORMULA6_OPERATORS = {"min", "max", "+", "*", "negative", "group"}
# An engine placeholder a master spelling may hold (template.rs's closed list, as master 6 uses it), and a quantifier
# variable or an argument's type spelling `{TYPE<k>}`, which the view binds.
SPELLING6_PLACEHOLDER_RE = re.compile(r"\{(\d+|\d+\.v\d+|=in\d+\.[a-z]+|TYPE\d+|[A-Z][A-Z0-9]*(?:\.[a-z0-9]+)?)\}")


def _expr6_terms(expr):
    """The `in<k>.<p>` terms of a property expression."""
    return [(int(k), p) for k, p in master_view.PROPERTY_TERM_RE.findall(expr)]


def witness_master6(mst=None):
    """Check master 6 itself, in the four kinds of row: its data types and properties as restated; every row's kind,
    reference, flows (each parses: an up flow, an environment attribute, a variable with its properties), guards (each
    parses and names a repair list of `repairs.tsv`; a heavy row's work bound by its operation's list), and spelling
    (only the engine's placeholders); the binders, the adapters and the repair lists as restated; the formula rows a
    closed list of operators spelled by their operands. Returns the failures (empty = pass)."""
    bad = []
    mst = mst or master_view.master6()
    if mst["version"] != 6:
        bad.append(f"N0 master 6's VERSION is {mst['version']}")
    seen = []
    for t in mst["datatypes"]:
        want = DATATYPES6.get(t["name"])
        got = (t["kind"], t["members"], t["over"], t["tag"], t["constant"], t["constant_props"])
        if want != got:
            bad.append(f"N1 the data type {t['name']} is {got}, the model says {want}")
        seen.append(t["name"])
    if sorted(seen) != sorted(DATATYPES6):
        bad.append(f"N1 the data types are {seen}, the model has {sorted(DATATYPES6)}")
    got = {(p["data_type"], p["name"]): p["kind"] for p in mst["properties"]}
    if got != PROPERTIES6 or len(mst["properties"]) != len(PROPERTIES6):
        bad.append(f"N2 the properties are {got}, the model has {PROPERTIES6}")
    bad += witness_rules(mst)
    lists = {}
    for r in mst["repairs"]:
        lists.setdefault(r["list"], []).append(r)
    if {n: [r["site"] for r in rs] for n, rs in lists.items()} != REPAIRS6:
        bad.append(f"N3 the repair lists are {sorted(lists)}, the model has {sorted(REPAIRS6)}")
    for n, rs in lists.items():
        for r in rs:
            for c in re.findall(r"\{call ([^({}]+)\(", r["template"]):
                if c not in ("{FITS}", "{FITJ}", "clampix@i64"):
                    bad.append(f"N3 the repair list {n} calls {c}, which is no adapter of the master")
    for s in ("subst", "subst.const"):
        last = lists.get(s, [{}])[-1]
        if (last.get("template"), last.get("when"), last.get("guard")) != ("{const}", "true", "true"):
            bad.append(f"N3 the substitution list {s} does not end with an unconditional {{const}}, so a guard could be left with no repair")
    refs = set()
    for r in mst["rows"]:
        ref = r["ref"]
        if not re.fullmatch(r"M\d{4}", ref) or ref in refs:
            bad.append(f"N4 the reference {ref} is malformed or repeated")
        refs.add(ref)
        if r["kind"] not in ROW6_KINDS:
            bad.append(f"N4 the row {ref} has the kind {r['kind']}")
        args = master_view.items(r["args"])
        for item in master_view.items(r["flows"]):
            m_up, m_env, m_var = master_view.UP6_RE.match(item), master_view.ENV6_RE.match(item), master_view.VAR6_RE.match(item)
            if not (m_up or m_env or m_var):
                bad.append(f"N5 the flow {item!r} of {ref} does not parse")
                continue
            expr = m_up.group(2) if m_up else m_env.group(3) if m_env else (m_var.group(3) or "")
            if any(k >= len(args) for k, _ in _expr6_terms(expr)):
                bad.append(f"N5 the flow {item!r} of {ref} reads an argument {ref} has not")
            at = int((m_env or m_var).group(1)) if (m_env or m_var) else None
            if at is not None and at >= len(args):
                bad.append(f"N5 the flow {item!r} of {ref} acts at an argument {ref} has not")
        op = re.match(r"[a-z0-9_]+", r["name"]).group(0)
        guards = master_view.items(r["guards"])
        for q in guards:
            m = master_view.GUARD6_RE.match(q)
            if not m:
                bad.append(f"N6 the guard {q!r} of {ref} does not parse")
                continue
            if m.group(5) not in lists:
                bad.append(f"N6 the guard {q!r} of {ref} names the repair list {m.group(5)}, which repairs.tsv does not have")
        work = [q for q in guards if "env.trips" in q]
        if r["kind"] in ("function", "adapter") and op in HEAVY6_OPS and work != [work_bound6(op)]:
            bad.append(f"N6 the heavy row {ref} {r['name']} has the work bound {work}, wants [{work_bound6(op)!r}]")
        if op not in HEAVY6_OPS and work:
            bad.append(f"N6 the row {ref} {r['name']} is not heavy and has a work bound")
        for ph in re.findall(r"\{([^{}]*)\}", r["spelling"]):
            if not SPELLING6_PLACEHOLDER_RE.fullmatch("{" + ph + "}"):
                bad.append(f"N7 the spelling of {ref} holds {{{ph}}}, which is no placeholder of the template engine")
        for (bop, over), (bargs, bflows, bspell) in BINDERS6.items():
            if op == bop and r["over"] == over:
                if (r["args"], r["flows"], r["spelling"]) != (bargs, bflows, bspell):
                    bad.append(f"N8 the binder {ref} {r['name']} is {(r['args'], r['flows'], r['spelling'])}, restated {(bargs, bflows, bspell)}")
        if r["kind"] == "adapter":
            want = ADAPTERS6.get(r["name"])
            if want is None or (r["args"], r["flows"], r["spelling"], r["prelude"]) != want:
                bad.append(f"N9 the adapter {ref} {r['name']} is {(r['args'], r['flows'], r['spelling'], r['prelude'])}, restated {want}")
            if r["prelude"] not in mst["preludes"]:
                bad.append(f"N9 the adapter {r['name']}'s prelude {r['prelude']} is not in master/6/preludes/")
    if sorted(r["name"] for r in mst["rows"] if r["kind"] == "adapter") != sorted(ADAPTERS6):
        bad.append("N9 the adapter rows are not the restated adapters")
    for f in mst["formulas"]:
        if f["operator"] not in FORMULA6_OPERATORS:
            bad.append(f"N10 the formula row {f['operator']} is not one of the formula language's operators")
        if re.sub(r"\{[01]\}", "", f["spelling"]).count("{"):
            bad.append(f"N10 the formula row {f['operator']} ({f['form']!r}) holds a placeholder other than its operands")
    return bad


def read_tables6(text):
    """A view's rows in the named text form, by kind of line: each a list of its fields."""
    out = {}
    for line in text.splitlines():
        if line:
            fs = line.split("\t")
            out.setdefault(fs[0], []).append(fs[1:])
    return out


def master_kingdom6(name):
    """The kingdom a problem's view is in the master tables (`kingdoms/master/`): `futhark.<problem>`."""
    return f"futhark.{name}"


def committed_rows6(name):
    """A problem's view as the master tables hold it (apart from the generator): the named text of `futhark.<problem>`."""
    return write_rows.named_of_rows(write_rows.select(write_rows.read_master(), master_kingdom6(name)))


def kingdom_row6(name, rows):
    """The view's row of `kingdoms.tsv`: name, parent, description, root (the entry's result), notes."""
    src = SIG6_PROBLEMS[name][0]
    return (master_kingdom6(name), "futhark", f"the Futhark kingdom's view of the problem {name} ({src})", write_rows.root_of(rows), "")


def witness_view6(tables, name, spec, mst=None):
    """Check one master 6 view from its WRITTEN rows (the four kinds of row, as the master tables hold them), apart from
    the generator: every data type
    and symbol a line names is declared before; every symbol but a formula row has one output; every up flow sets a
    property its result carries and reads only properties its arguments carry; every variable's data type is declared
    and its properties carried; every guard reads only carried properties, and every must guard's last repair always
    applies (guard and when `true`); every repair's call names an adapter symbol of the view; the entry's parameters
    are the specification's, and every size it names is a size name at least 1."""
    bad = []
    t = read_tables6(tables)
    types = [d[0] for d in t.get("data_type", [])]
    carries = {}
    for dt, p in t.get("carries", []):
        carries.setdefault(dt, set()).add(p)
    sizes = {n: int(lb) for _, n, lb in t.get("size_name", [])}
    symbols = {s[0]: s for s in t.get("symbol", [])}
    adapters = {s[0] for s in t.get("symbol", []) if s[3] == "adapter"}
    for d in t.get("data_type", []):
        for m in [x for x in ";".join(d[2:]).split(";") if x] if d[1] != "scalar" else []:
            if m not in types:
                bad.append(f"V1 {name}: the data type {d[0]} names {m}, which is not declared")
    args, outs = {}, {}
    for s, pos, way, dt, _ in t.get("argument", []):
        if s not in symbols or dt not in types:
            bad.append(f"V2 {name}: the argument {s} {pos} names an undeclared symbol or data type")
        if way == "in":
            args.setdefault(s, []).append(dt)
        else:
            outs.setdefault(s, []).append(dt)
    for s, row in symbols.items():
        if row[2] != "formula" and len(outs.get(s, [])) != 1:
            bad.append(f"V2 {name}: the symbol {s} has {len(outs.get(s, []))} outputs")
        if row[2] == "formula" and (s in outs or s in args):
            bad.append(f"V2 {name}: the formula row {s} has an argument")
    for s, at, way, fname, value in t.get("flow", []):
        a = args.get(s, [])
        if way == "up":
            if fname not in carries.get(outs.get(s, [""])[0], set()):
                bad.append(f"V3 {name}: {s}'s up flow sets {fname}, which its result does not carry")
        elif fname.startswith("var."):
            dt = re.split(r"[./]", fname[4:], maxsplit=1)[0]
            if dt not in types:
                bad.append(f"V3 {name}: {s}'s flow {fname} introduces variables of an undeclared data type")
            prop = fname[4 + len(dt):].lstrip("/0123456789").lstrip(".")
            if prop and prop != "spell" and prop not in carries.get(dt, set()):
                bad.append(f"V3 {name}: {s}'s flow {fname} sets a property {dt} does not carry")
        if not fname.endswith(".spell"):
            for k, p in _expr6_terms(value):
                if k >= len(a) or p not in carries.get(a[k], set()):
                    bad.append(f"V3 {name}: {s}'s flow {fname} reads in{k}.{p}, which its argument does not carry")
    guards = {}
    for g in t.get("guard", []):
        guards[(g[0], g[1])] = (g, [])
    for r in t.get("repair", []):
        guards[(r[0], r[1])][1].append(r)
    for (s, order), (g, repairs) in guards.items():
        a = args.get(s, [])
        for k, p in _expr6_terms(g[3]):
            if k >= len(a) or p not in carries.get(a[k], set()):
                bad.append(f"V4 {name}: {s}'s guard {g[3]} reads in{k}.{p}, which its argument does not carry")
        if g[2] == "must" and (not repairs or (repairs[-1][3], repairs[-1][4]) != ("true", "true")):
            bad.append(f"V4 {name}: {s}'s must guard {g[3]} has no repair that always applies last")
        for r in repairs:
            for c in re.findall(r"\{call ([^({}]+)\(", r[5]):
                if c not in adapters:
                    bad.append(f"V4 {name}: {s}'s repair calls {c}, which is no adapter symbol of the view")
    root_vars = [master_view.env_variable(e) for e in spec["env"]]
    params = [p[0] for p in t.get("parameter", [])]
    var_types = {outs[s][0] for s, row in symbols.items() if row[2] == "variable" and outs.get(s)}
    if params != [n for n, dt, _ in root_vars if dt in var_types]:
        bad.append(f"V5 {name}: the entry's parameters {params} are not the specification's, in order")
    for p in t.get("parameter", []):
        for item in [x for x in p[2].split(";") if x and x != "-"]:
            for n in re.findall(r"\b[a-z_][a-z0-9_]*\b", item.partition("=")[2]):
                if n not in sizes or sizes[n] < 1:
                    bad.append(f"V5 {name}: the parameter {p[0]} names the size {n}, which is no size name at least 1")
    return bad


def sig6_dir(name):
    return os.path.join(HERE, "sig", name, "m6")


def sig6_texts(name):
    """A master 6 problem's specification (the sniffer's), its view's tables (`view.py`) and its e-graph (below), by
    file name."""
    src, entry, depth, master, elements, lets, families = SIG6_PROBLEMS[name]
    spec_text = sniff.sniff6(os.path.join(REPO, src), entry, depth, master, elements, lets, families)
    out = {"view.tsv": spec_text}
    spec = master_view.read_spec6(spec_text)
    view = master_view.View6(spec)
    out.update(view.texts())
    out.update(egraph6_texts(view, spec, master_view.master6()))
    return out


# --------------------------------------------------------------------------
# MASTER 6's E-GRAPH (plan section 7, task F5): every view's, A7's included, written from the view's own tables and
# `master/6/rules.tsv`, beside them under `sig/<problem>/m6/` (`egraph.datatype`, `egraph.rules`, `egraph.rows`,
# `egraph.sorts`). `src/futhark_view/egraph6.rs` computes the same four texts byte for byte.
#
# * ONE SORT A DATA TYPE: `S<tag>` (the sort IS the data type).
# * ONE CONSTRUCTOR A SYMBOL, `X<name>`, its arguments' sorts in order; a `const` row takes its pool index (`i64`); a
#   `var` row takes its variable's RESOLVED identity, the binding depth and the ordinal (`i64 i64`), never its index
#   word (plan 7.5 risk 3), and has a second constructor `K<name>` of no argument for the reading with no variable of
#   its data type in scope (its constant).
# * THE RULES: each rule of `rules.tsv` at every data type it is on (`elem`: the view's element scalars; `arr`: their
#   rank-1 arrays; `mat`: their rank-2 arrays; `tuple`/`tuple3`: the pairs and triples of elements), for the elements it
#   is exact for, where the view has the row it `needs` and every row it names. A placeholder `{op}` names `op@<tag>`;
#   on a tuple, `{fst}`, `{snd}`, `{thd}` name the projection `fst.<tag>@<component's tag>`; `{op@tag}` names that
#   symbol exactly (a row function of another data type).
# * `egraph.rows`: each symbol's constructors with their sort and what the constructor reads (`node`, `dc`, `binder`,
#   `constant`); `egraph.sorts`: each data type's sort.
# --------------------------------------------------------------------------

EGRAPH6_RULESET = "futhark_sig"


def egraph6_name(text):
    """A symbol name or tag as an egglog identifier: letters and digits kept, `@` as `__`, anything else `_`."""
    return "".join(c if c.isalnum() else ("__" if c == "@" else "_") for c in text)


def egraph6_rule_symbols(rule, dt, by_name):
    """The rule template's `{op}` placeholders at the data type `dt` (a datatypes.tsv row), as symbol names, in order."""
    out = []
    for op in re.findall(r"\{([a-z][^{}]*)\}", rule):
        if "@" in op:
            out.append(op)
        elif dt["kind"] == "tuple" and op in ("fst", "snd", "thd"):
            comp = dt["members"].split(";")[("fst", "snd", "thd").index(op)]
            out.append(f"{op}.{dt['tag']}@{by_name[comp]['tag']}" if comp in by_name else f"{op}.{dt['tag']}@?")
        else:
            out.append(f"{op}@{dt['tag']}")
    return out


def egraph6_texts(view, spec, mst):
    """The four e-graph texts of a master 6 view (a `View6`): its data types and symbols in its table's order, its
    specification and the master's rules."""
    dts = [{"name": t["name"], "kind": t["kind"], "members": t["members"], "tag": t["tag"]} for t in view.types]
    syms = [{"name": r["name"], "kind": r["kind"], "out": r["out"], "args": ";".join(r["args"])} for r in view.symbols]
    by_name = {t["name"]: t for t in dts}
    names = {s["name"] for s in syms}
    elements = spec["elements"]

    def sort(dt_name):
        return "S" + egraph6_name(by_name[dt_name]["tag"])

    data = ["; A master 6 view's e-graph datatype, generated by gen_tables.py.",
            "; One sort per data type: the sort IS the data type.", ""]
    data += [f"(sort {sort(t['name'])})" for t in dts]
    data.append("")
    rows = ["# symbol\tconstructor\tsort\treads"]
    for s in syms:
        args = [a for a in s["args"].split(";") if a]
        x, out = "X" + egraph6_name(s["name"]), sort(s["out"])
        if s["kind"] == "const":
            data.append(f"(constructor {x} (i64) {out})")
            rows.append(f"{s['name']}\t{x}\t{out}\tdc")
        elif s["kind"] == "var":
            k = "K" + egraph6_name(s["name"])
            data.append(f"(constructor {x} (i64 i64) {out})")
            data.append(f"(constructor {k} () {out})")
            rows.append(f"{s['name']}\t{x}\t{out}\tbinder")
            rows.append(f"{s['name']}\t{k}\t{out}\tconstant")
        else:
            data.append(f"(constructor {x} ({' '.join(sort(a) for a in args)}) {out})")
            rows.append(f"{s['name']}\t{x}\t{out}\tnode")
    rules = ["; A master 6 view's e-graph ruleset, generated by gen_tables.py from master/6/rules.tsv.",
             f"(ruleset {EGRAPH6_RULESET})", ""]
    for on in ("elem", "arr", "mat", "tuple", "tuple3"):
        for t in dts:
            members = [m for m in t["members"].split(";") if m]
            if on == "elem" and t["kind"] == "scalar" and t["name"] in elements:
                exact = [t["name"]]
            elif on == "arr" and t["kind"] == "array" and members[0] in elements:
                exact = members
            elif (on == "mat" and t["kind"] == "array" and by_name[members[0]]["kind"] == "array"
                  and by_name[members[0]]["members"] in elements):
                exact = [by_name[members[0]]["members"]]
            elif on in ("tuple", "tuple3") and t["kind"] == "tuple" and len(members) == (2 if on == "tuple" else 3):
                exact = members
            else:
                continue
            mine = []
            for r in mst["rules"]:
                if r["on"] != on or not all(r["elems"] == "*" or e in r["elems"].split(",") for e in exact):
                    continue
                if r["needs"] and f"{r['needs']}@{t['tag']}" not in names:
                    continue
                named = egraph6_rule_symbols(r["rule"], t, by_name)
                missing = sorted({n for n in named if n not in names})
                if missing:
                    # A view without the operation has none of the rule's rows (A7 has no `add@f32`). A rule whose
                    # `needs` row is present, or with some of its rows present and others not, names a row that was
                    # renamed or dropped: dropping the rule in silence would lose it from this view and the Rust copy
                    # would agree byte for byte, so the generator stops.
                    if (r["needs"] and f"{r['needs']}@{t['tag']}" in names) or len(missing) < len(set(named)):
                        raise ValueError(f"our software has a bug: the master 6 rule {r['rule']!r} on {t['name']} "
                                         f"names {', '.join(missing)}, which the view does not hold")
                    continue
                text = r["rule"]
                for n in named:
                    text = re.sub(r"\{[a-z][^{}]*\}", "X" + egraph6_name(n), text, count=1)
                mine.append(text)
            if mine:
                rules.append(f"; {t['name']}")
                rules += mine
                rules.append("")
    sorts = ["# data type\tsort"] + [f"{t['name']}\t{sort(t['name'])}" for t in dts]
    return {"egraph.datatype": "\n".join(data) + "\n", "egraph.rules": "\n".join(rules) + "\n",
            "egraph.rows": "\n".join(rows) + "\n", "egraph.sorts": "\n".join(sorts) + "\n"}


def witness_egraph6(dir_path, tables, name, mst=None):
    """The written e-graph of a master 6 view (its files under `dir_path`, its rows `tables` as the master tables hold
    them), read back apart from the generator: every symbol has its constructor
    (a `var` row two), no constructor or sort is used twice, every data type has its sort, every constructor of a rule
    is declared, every rule's operator rows have their sorts' arities, and NO RULE NAMES A BINDER (a row with a
    binding), so no rule moves a subtree under another binder (plan 7.5 risk 3) and a variable keeps its environment."""
    mst = mst or master_view.master6()
    bad = []

    def read(fname):
        with open(os.path.join(dir_path, fname)) as f:
            return f.read()
    t = read_tables6(tables)
    dts = [{"name": d[0]} for d in t.get("data_type", [])]
    ins = {}
    for s_, _, way, dt, _ in t.get("argument", []):
        if way == "in":
            ins.setdefault(s_, []).append(dt)
    binding = {f[0] for f in t.get("flow", []) if f[2] == "down" and f[3].startswith("var.")}
    kind = {"constant": "const", "variable": "var"}
    syms = [{"name": s_[0], "kind": kind.get(s_[2], "function"), "args": ";".join(ins.get(s_[0], [])), "binds": s_[0] in binding} for s_ in t.get("symbol", []) if s_[2] != "formula"]
    data, rules = read("egraph.datatype"), read("egraph.rules")
    declared = {}
    for m in re.finditer(r"^\(constructor (\S+) \(([^)]*)\) (\S+)\)$", data, re.M):
        if m.group(1) in declared:
            bad.append(f"E1 {name}: the constructor {m.group(1)} is declared twice")
        declared[m.group(1)] = (m.group(2).split(), m.group(3))
    sorts = re.findall(r"^\(sort (\S+)\)$", data, re.M)
    if len(set(sorts)) != len(sorts) or len(sorts) != len(dts):
        bad.append(f"E2 {name}: {len(sorts)} sorts ({len(set(sorts))} distinct) for {len(dts)} data types")
    sort_lines = [l.split("\t") for l in read("egraph.sorts").splitlines()[1:] if l]
    if [l[0] for l in sort_lines] != [t["name"] for t in dts] or [l[1] for l in sort_lines] != sorts:
        bad.append(f"E2 {name}: egraph.sorts does not give each data type, in order, its declared sort")
    sort_of = {l[0]: l[1] for l in sort_lines}
    row_lines = [l.split("\t") for l in read("egraph.rows").splitlines()[1:] if l]
    of_symbol = {}
    for sym, ctor, srt, reads in row_lines:
        of_symbol.setdefault(sym, []).append(reads)
        if ctor not in declared or declared[ctor][1] != srt:
            bad.append(f"E3 {name}: {sym}'s constructor {ctor} of {srt} is not declared so")
    if len({l[1] for l in row_lines}) != len(row_lines):
        bad.append(f"E3 {name}: a constructor stands for two symbols")
    binders = set()
    for s in syms:
        want = {"const": ["dc"], "var": ["binder", "constant"]}.get(s["kind"], ["node"])
        if of_symbol.get(s["name"]) != want:
            bad.append(f"E4 {name}: the symbol {s['name']} has the constructors {of_symbol.get(s['name'])}, wants {want}")
        args = [a for a in s["args"].split(";") if a]
        ctor = "X" + egraph6_name(s["name"])
        if s["kind"] not in ("const", "var") and ctor in declared and declared[ctor][0] != [sort_of.get(a) for a in args]:
            bad.append(f"E5 {name}: {ctor} takes {declared[ctor][0]}, {s['name']} takes {args}")
        if s["binds"]:
            binders.add(ctor)
    for line in rules.splitlines():
        if not line.startswith("(rewrite "):
            continue
        if not line.endswith(f":ruleset {EGRAPH6_RULESET})"):
            bad.append(f"E6 {name}: the rule {line!r} is not in the ruleset {EGRAPH6_RULESET}")
        for op in re.findall(r"\((\w+)", line)[1:]:
            if op not in declared:
                bad.append(f"E6 {name}: the rule {line!r} names {op}, which is not declared")
            if op in binders:
                bad.append(f"E7 {name}: the rule {line!r} names the binder {op}: a rule may not move a subtree under another binder")
    return bad


# --------------------------------------------------------------------------
# MASTER 6's STATIC COST (plan section 7, task F3): `master/6/costs.tsv`, GENERATED from `master/6/rows.tsv` and
# `master/6/adapters.tsv`. One line a master row (`kind` = `row`): what the row reads from each argument when that
# argument is materialised (`reads`, `;`-separated in argument order), what it writes and allocates when it is
# materialised itself, its serial depth, and whether it is elementwise and so ABSORBS its arguments (master 5's
# `costs.tsv` rules, restated over argument properties instead of class roles). One line an adapter (`kind` =
# `adapter`): the traffic it adds where it renders (`traffic`): every `fit` the cells of the operand it fits, at its
# target length (`fitted.cells`), plus the cells it reads of that operand at its source length (`source.cells`), so
# its own charge covers its full traffic (F3 review finding 3; this table is the charge's single source); `clampix`
# its two compares; a substitution nothing (its node renders as its substitute, which is costed in its place). So an
# elite with a needless adapter loses to its own clean form (plan 7.5 risk 4). `src/futhark_cost.rs` (`CostTable6`,
# `program_cost6`) reads it.
#
# A formula is `0` or a sum of products of factors: an integer, `out.<p>` (the row's result) or `in<k>.<p>` (argument
# k as the node's adapters leave it), `<p>` a property or `cells` (an array's elements, a scalar 1, a tuple its
# components, a function 0); `fitted.cells` and `source.cells` in an adapter's line. The cost is a MODEL and a
# pre-optimisation one, as master 5's is: a node's costs count once a trip of every binder above it, its depth once a
# trip of every loop above it (a `tabulate` is parallel).
# --------------------------------------------------------------------------

COSTS6_COLUMNS = ["kind", "ref", "name", "reads", "write", "alloc", "depth", "absorb", "traffic"]
COSTS6_PATH = os.path.join(HERE, "master", "6", "costs.tsv")

# The rows that are not elementwise scalar operations, by operation: (reads, write, alloc, depth, absorb).
COST6_RULES = {
    "get": ("1;in1.cells", "out.cells", "0", "0", "0"),
    "sum": ("in0.len", "out.cells", "0", "in0.len", "0"),
    "minimum": ("in0.len", "out.cells", "0", "in0.len", "0"),
    "maximum": ("in0.len", "out.cells", "0", "in0.len", "0"),
    "tab": ("0;in0.value", "out.cells", "out.cells", "0", "0"),
    "tabrows": ("0;0;out.cells", "out.cells", "out.cells", "0", "0"),
    "replicate": ("0;1", "out.cells", "out.cells", "0", "0"),
    "map2": ("in0.len;in1.len", "out.cells", "out.cells", "0", "1"),
    "scan": ("in0.len", "out.cells", "out.cells", "in0.len", "0"),
    "hist": ("0;in1.len;in2.len", "out.cells", "out.cells", "in2.len", "0"),
    "histrows": ("0;0;in2.len;in3.cells", "out.cells", "out.cells", "in3.rows", "0"),
    "rowdiv": ("in0.cells;in1.len", "out.cells", "out.cells", "0", "1"),
    "row": ("in0.cols;in1.cells", "0", "0", "0", "0"),
    "loop": ("0;in1.cells;in2.cells", "out.cells", "out.cells", "in0.value", "0"),
    "loop1": ("0;in1.cells;in2.cells", "out.cells", "out.cells", "in0.value", "0"),
    "let": ("0;0", "0", "0", "0", "0"),
    "map_rows": ("0;in1.cells", "out.cells", "out.cells", "0", "1"),
    "map2_rows": ("0;in1.cells;in2.cells", "out.cells", "out.cells", "0", "1"),
    "transpose": ("in0.cells", "out.cells", "out.cells", "0", "0"),
    "matmul": ("in0.cells;in1.cells", "out.cells", "out.cells", "in0.cols", "0"),
}


def cost6_of_row(r):
    """The cost of one master 6 row template: (reads, write, alloc, depth, absorb).

    * a leaf (a variable, a literal, a terminal) and a row function: nothing read, written or allocated -- a variable
      is the interface's input or a binder's, already in memory, as master 5's terminal is;
    * the rows of `COST6_RULES` as written there;
    * every other row is an elementwise SCALAR operation: it reads each argument's cells (a size nothing: it is a
      name), writes its result's cells, allocates nothing, and absorbs its arguments (a chain of scalar operations is
      one expression).
    """
    op = re.match(r"[a-z0-9_]+", r["name"]).group(0)
    args = master_view.items(r["args"])
    if r["kind"] == "adapter":
        return (";".join("0" for _ in args), "0", "0", "0", "0")
    if r["kind"] != "function" or "->" in r["out"]:
        if args:
            raise ValueError(f"our software has a bug: the leaf or row function {r['name']} takes arguments")
        return ("", "0", "0", "0", "0")
    if op in COST6_RULES:
        rule = COST6_RULES[op]
        if len(rule[0].split(";")) != len(args):
            raise ValueError(f"our software has a bug: the cost rule of {op} reads {rule[0]!r}, and {r['name']} takes {len(args)} arguments")
        return rule
    if r["out"].startswith("["):
        raise ValueError(f"our software has a bug: the array row {r['name']} has no cost rule")
    reads = ";".join("0" if a == "size" else f"in{k}.cells" for k, a in enumerate(args))
    return (reads, "out.cells", "0", "0", "1")


def adapter6_traffic(r):
    """The traffic one run of an adapter row adds, placed by evolution or called by a repair: a `fit` the cells it
    writes and the cells it reads of its operand (`out.cells+in1.cells`), `clampix` its two compares; any other row
    nothing."""
    if r["kind"] != "adapter":
        return "0"
    if r["name"].startswith("fit"):
        return "out.cells+in1.cells"
    if r["name"].startswith("clampix"):
        return "2"
    raise ValueError(f"our software has a bug: the adapter {r['name']} has no cost rule")


def costs6_text(mst):
    """The text of `master/6/costs.tsv`: one line a master row, an adapter's with its traffic."""
    out = ["\t".join(COSTS6_COLUMNS)]
    for r in mst["rows"]:
        reads, write, alloc, depth, absorb = cost6_of_row(r)
        out.append("\t".join(["row", r["ref"], r["name"], reads, write, alloc, depth, absorb, adapter6_traffic(r)]))
    return "\n".join(out) + "\n"


# The witness's restatement, by master row: what each row reads, writes, allocates, its depth and whether it absorbs,
# written out from the rows' meanings (README 4, the spellings), not from the generator's rules above.
COST6_LEAVES = {"M0016", "M0021", "M0047", "M0048", "M0049", "M0050", "M0051", "M0052", "M0054", "M0055", "M0071",
                "M0081", "M0087", "M0088"}
COST6_BY_REF = {}
for _refs, _cost in [
    (("M0009",), ("1;in1.cells", "out.cells", "0", "0", "0")),
    (("M0010", "M0011", "M0012"), ("in0.len", "out.cells", "0", "in0.len", "0")),
    (("M0024",), ("0;in0.value", "out.cells", "out.cells", "0", "0")),
    (("M0063",), ("0;0;out.cells", "out.cells", "out.cells", "0", "0")),
    (("M0025",), ("0;1", "out.cells", "out.cells", "0", "0")),
    (("M0026", "M0027", "M0028", "M0029", "M0062"), ("in0.len;in1.len", "out.cells", "out.cells", "0", "1")),
    (("M0030", "M0031", "M0032"), ("in0.len", "out.cells", "out.cells", "in0.len", "0")),
    (("M0033", "M0034", "M0035", "M0057", "M0058", "M0059"), ("0;in1.len;in2.len", "out.cells", "out.cells", "in2.len", "0")),
    (("M0064", "M0065"), ("0;0;in2.len;in3.cells", "out.cells", "out.cells", "in3.rows", "0")),
    (("M0066",), ("in0.cells;in1.len", "out.cells", "out.cells", "0", "1")),
    (("M0036",), ("in0.cols;in1.cells", "0", "0", "0", "0")),
    (("M0040", "M0041", "M0073", "M0074", "M0083", "M0084"), ("0;in1.cells;in2.cells", "out.cells", "out.cells", "in0.value", "0")),
    (("M0085",), ("0;0", "0", "0", "0", "0")),
    (("M0043",), ("0;in1.cells", "out.cells", "out.cells", "0", "1")),
    (("M0044",), ("0;in1.cells;in2.cells", "out.cells", "out.cells", "0", "1")),
    (("M0045",), ("in0.cells", "out.cells", "out.cells", "0", "0")),
    (("M0046",), ("in0.cells;in1.cells", "out.cells", "out.cells", "in0.cols", "0")),
]:
    for _ref in _refs:
        COST6_BY_REF[_ref] = _cost
COST6_FACTOR = r"(?:[1-9][0-9]*|out\.[a-z]+|in[0-9]+\.[a-z]+)"
COST6_FORMULA_RE = re.compile(rf"^(?:0|{COST6_FACTOR}(?:\*{COST6_FACTOR})*(?:\+{COST6_FACTOR}(?:\*{COST6_FACTOR})*)*)$")


def cost6_properties(template, mst):
    """The properties a value of the master data type `template` (a row's `out` or one of its `args`) carries: its
    data type's in `master/6/properties.tsv` (an array of a concrete element, `[]f32`, is `[]{E}`'s), and `cells`."""
    t = re.sub(r"^((?:\[\])+)(?:f32|i32)$", r"\1{E}", template)
    return {p["name"] for p in mst["properties"] if p["data_type"] == t} | {"cells"}


def witness_costs6(path, mst):
    """Check the committed `master/6/costs.tsv` against the restatement above (C1-C6): its header; one row line a
    master row, in order, with the row's own ref and name; each row's cost as restated (a leaf costs nothing; an
    elementwise scalar operation, every row not restated, reads each non-size argument's cells, writes its result's,
    allocates nothing and absorbs); every formula well formed and naming only arguments the row has and properties
    the data type of that argument (or of the result) carries; an adapter row reads, writes and allocates nothing and
    adds its traffic, every `fit` the cells it writes and reads (`out.cells+in1.cells`), `clampix` 2; no other row adds
    traffic."""
    bad = []
    with open(path) as f:
        lines = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    if not lines or lines[0] != COSTS6_COLUMNS:
        return [f"C1 master/6/costs.tsv's header is {lines[0] if lines else None}, wants {COSTS6_COLUMNS}"]
    body = [l for l in lines[1:]]
    for l in body:
        if len(l) != len(COSTS6_COLUMNS):
            bad.append(f"C1 the cost line {l} has {len(l)} columns, wants {len(COSTS6_COLUMNS)}")
    if bad:
        return bad
    rows = [dict(zip(COSTS6_COLUMNS, l)) for l in body if l[0] == "row"]
    if len(rows) != len(body):
        bad.append("C1 a cost line is not a row's")
    if [(c["ref"], c["name"]) for c in rows] != [(r["ref"], r["name"]) for r in mst["rows"]]:
        bad.append("C2 the cost rows are not the master's rows, one a row, in order")
        return bad
    for c, r in zip(rows, mst["rows"]):
        args = master_view.items(r["args"])
        got = (c["reads"], c["write"], c["alloc"], c["depth"], c["absorb"])
        if r["kind"] == "adapter":
            want = (";".join("0" for _ in args), "0", "0", "0", "0")
        elif r["ref"] in COST6_LEAVES:
            want = ("", "0", "0", "0", "0")
        elif r["ref"] in COST6_BY_REF:
            want = COST6_BY_REF[r["ref"]]
        else:
            want = (";".join(f"in{k}.cells" if a != "size" else "0" for k, a in enumerate(args)), "out.cells", "0", "0", "1")
        if got != want:
            bad.append(f"C3 the cost of {r['ref']} {r['name']} is {got}, restated {want}")
        want_traffic = ("out.cells+in1.cells" if r["name"].startswith("fit") else "2") if r["kind"] == "adapter" else "0"
        if c["traffic"] != want_traffic:
            bad.append(f"C3 the row {r['name']} adds traffic {c['traffic']}, restated {want_traffic} (only an adapter adds any)")
        reads = c["reads"].split(";") if c["reads"] else []
        if len(reads) != len(args):
            bad.append(f"C4 {r['name']} reads {len(reads)} arguments and takes {len(args)}")
        for f in reads + [c["write"], c["alloc"], c["depth"], c["traffic"]]:
            if not COST6_FORMULA_RE.match(f):
                bad.append(f"C4 {r['name']}'s formula {f!r} is not a sum of products of integers, out.<p> and in<k>.<p>")
                continue
            for k, p in re.findall(r"in([0-9]+)\.([a-z]+)", f):
                if int(k) >= len(args):
                    bad.append(f"C4 {r['name']}'s formula {f!r} reads in{k}.{p}, which it does not have")
                elif p not in cost6_properties(args[int(k)], mst):
                    bad.append(f"C4 {r['name']}'s formula {f!r} reads in{k}.{p}, and its argument {k}'s data type {args[int(k)]} has no {p}")
            for p in re.findall(r"out\.([a-z]+)", f):
                if p not in cost6_properties(r["out"], mst):
                    bad.append(f"C4 {r['name']}'s formula {f!r} reads out.{p}, and its result's data type {r['out']} has no {p}")
        if c["absorb"] not in ("0", "1"):
            bad.append(f"C4 {r['name']}'s absorb is {c['absorb']!r}")
    return bad


def witness_costs6_catches(mst):
    """THE C4 WITNESS CHECKS EACH ARGUMENT'S OWN DATA TYPE (F3 review finding 6): the committed table with one
    property read off an argument whose data type does not carry it (`sum`'s vector read by `cols`, `transpose`'s
    matrix by `len`, `next`'s index by `value`) must be found by C4 once each (C3, the restatement, finds them too)."""
    import tempfile
    with open(COSTS6_PATH) as f:
        lines = f.read().split("\n")
    corrupt = {"M0010": ("in0.len", "in0.cols"), "M0045": ("in0.cells", "in0.len")}
    nxt = next((r["ref"] for r in mst["rows"] if r["name"] == "next@i64"), None)
    if nxt is not None:
        corrupt[nxt] = ("in0.cells", "in0.value")
    out = []
    for l in lines:
        f = l.split("\t")
        if len(f) == len(COSTS6_COLUMNS) and f[0] == "row" and f[1] in corrupt:
            old, new = corrupt[f[1]]
            if f[3].split(";")[0] != old:
                return [f"C4 self-check: the row {f[1]} reads {f[3]!r}, the corruption expects {old!r} first"]
            f[3] = ";".join([new] + f[3].split(";")[1:])
        out.append("\t".join(f))
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as t:
        t.write("\n".join(out))
    try:
        got = witness_costs6(t.name, mst)
    finally:
        os.remove(t.name)
    c4 = [g for g in got if g.startswith("C4 ")]
    if len(c4) != len(corrupt) or not all("has no" in g for g in c4):
        return [f"C4 self-check: {len(corrupt)} arguments read by a property their data type lacks gave C4 {c4}"]
    return []


def costs6_main(mode, mst):
    """`master/6/costs.tsv`: checked byte for byte against the generator (the default), or written to a staging file,
    witnessed there and only then moved into place; then witnessed."""
    text = costs6_text(mst)
    if mode == "--write":
        staged = COSTS6_PATH + ".staged"
        with open(staged, "w") as f:
            f.write(text)
        bad = witness_costs6(staged, mst)
        if bad:
            os.remove(staged)
            print("THE MASTER 6 COST WITNESS TESTS FAILED (the committed file is untouched):")
            for b in bad:
                print("  " + b)
            return 1
        os.replace(staged, COSTS6_PATH)
    else:
        try:
            with open(COSTS6_PATH, "rb") as f:
                got = f.read()
        except OSError as e:
            print(f"THE MASTER 6 COST TABLE master/6/costs.tsv cannot be read: {e}")
            return 1
        if got != text.encode("utf-8"):
            print(f"THE COMMITTED master/6/costs.tsv is not what the generator produces ({first_difference(got, text.encode('utf-8'))}); "
                  "run with --write to regenerate")
            return 1
    bad = witness_costs6(COSTS6_PATH, mst) + witness_costs6_catches(mst)
    if bad:
        print("THE MASTER 6 COST WITNESS TESTS FAILED:")
        for b in bad:
            print("  " + b)
        return 1
    print(f"master/6/costs.tsv: {len(mst['rows'])} rows, {sum(1 for r in mst['rows'] if r['kind'] == 'adapter')} of them adapters; "
          f"{'committed table MATCHES the generator, ' if mode == '--check' else 'written, '}cost witness tests C1-C6 PASS")
    return 0


def witness_egraph6_admission(mst):
    """E8: the generator stops on a rule that names a row the view does not hold while the view holds the rule's other
    rows (a renamed row). A7's view with `id@fn.arr_f32_arr_f32` taken out of its symbols must fail, naming that row;
    A7's own view must not."""
    src, entry, depth, master, elements, lets, families = SIG6_PROBLEMS["a7"]
    spec_text = sniff.sniff6(os.path.join(REPO, src), entry, depth, master, elements, lets, families)
    spec = master_view.read_spec6(spec_text)
    view = master_view.View6(spec)
    try:
        egraph6_texts(view, spec, mst)
    except ValueError as e:
        return [f"E8 a7: the view's own e-graph does not generate: {e}"]
    gone = "id@fn.arr_f32_arr_f32"
    cut = [r for r in view.symbols if r["name"] != gone]
    if len(cut) != len(view.symbols) - 1:
        return [f"E8 a7: the view holds no single row {gone} to take out"]
    view.symbols = cut
    try:
        egraph6_texts(view, spec, mst)
    except ValueError as e:
        return [] if gone in str(e) else [f"E8 a7: without {gone} the generator stops for another reason: {e}"]
    return [f"E8 a7: without {gone} the generator drops A7's rule on it in silence"]


def master6_main(mode):
    """Master 6: the master witnesses, its static cost table, then each problem's specification and view, checked byte
    for byte (the default) or written (to a staging directory, witnessed there, and only then moved into place)."""
    mst = master_view.master6()
    bad = witness_master6() + witness_egraph6_admission(mst)
    if bad:
        print("THE MASTER 6 WITNESS TESTS FAILED:")
        for b in bad:
            print("  " + b)
        return 1
    mst = master_view.master6()
    if costs6_main(mode, mst):
        return 1
    print(f"master 6: {len(mst['rows'])} row templates ({sum(1 for r in mst['rows'] if r['kind'] == 'adapter')} adapters), "
          f"{len(mst['datatypes'])} data type templates, {len(mst['repairs'])} repairs in {len({r['list'] for r in mst['repairs']})} lists, "
          f"version {mst['version']}; master 6 witness tests PASS")
    if mode == "--write":
        write_rows.write_parent("futhark", FUTHARK_DESCRIPTION)
    elif [k for k in write_rows.read_master()[write_rows.KINGDOMS_FILE] if k[0] == "futhark"] != [["futhark", "", FUTHARK_DESCRIPTION, "", ""]]:
        print("THE MASTER TABLES' row for the kingdom futhark is not the generator's; run with --write to regenerate")
        return 1
    for name in sorted(SIG6_PROBLEMS):
        out_dir = sig6_dir(name)
        try:
            wanted = sig6_texts(name)
        except ValueError as e:
            print(f"THE MASTER 6 VIEW {name} FAILED: {e}")
            return 1
        # The view's rows go into the master tables (`kingdoms/master/`, the kingdom `futhark.<problem>`), its
        # specification and e-graph beside the problem under `sig/<problem>/m6/`.
        tables = wanted.pop("tables.tsv")
        rows = write_rows.rows_of_named(tables)
        if mode == "--write":
            staged = os.path.join(out_dir, ".staged")
            os.makedirs(staged, exist_ok=True)
            for fname, text in wanted.items():
                with open(os.path.join(staged, fname), "w") as f:
                    f.write(text)
            bad = witness_view6(tables, name, master_view.read_spec6(wanted["view.tsv"]), mst) + witness_egraph6(staged, tables, name, mst)
            if bad:
                print(f"THE MASTER 6 VIEW {name} FAILED (the committed files are untouched):")
                for b in bad[:40]:
                    print("  " + b)
                for fname in wanted:
                    os.remove(os.path.join(staged, fname))
                os.rmdir(staged)
                return 1
            for fname in wanted:
                os.replace(os.path.join(staged, fname), os.path.join(out_dir, fname))
            os.rmdir(staged)
            write_rows.write_kingdom(master_kingdom6(name), rows, kingdom_row6(name, rows))
        else:
            for fname, text in wanted.items():
                path = os.path.join(out_dir, fname)
                try:
                    with open(path, "rb") as f:
                        got = f.read()
                except OSError as e:
                    bad.append(f"the master 6 artefact sig/{name}/m6/{fname} cannot be read: {e}")
                    continue
                if got != text.encode("utf-8"):
                    bad.append(f"the committed sig/{name}/m6/{fname} is not what the sniffer and view.py produce "
                               f"({first_difference(got, text.encode('utf-8'))}); run with --write to regenerate")
            bad += write_rows.check_kingdom(master_kingdom6(name), rows, kingdom_row6(name, rows))
            if not bad:
                committed = committed_rows6(name)
                bad = witness_view6(committed, name, master_view.read_spec6(wanted["view.tsv"]), mst) + witness_egraph6(out_dir, committed, name, mst)
            if bad:
                print(f"THE MASTER 6 VIEW {name} FAILED:")
                for b in bad[:40]:
                    print("  " + b)
                return 1
        t6 = read_tables6(tables)
        kinds = {}
        for s_ in t6.get("symbol", []):
            kind = "adapter" if s_[3] == "adapter" else s_[2]
            kinds[kind] = kinds.get(kind, 0) + 1
        print(f"sig/{name}/m6: {len(t6.get('data_type', []))} data types ({', '.join(d[0] for d in t6.get('data_type', []))}), "
              f"{len(t6.get('symbol', []))} symbols ({', '.join(f'{n} {k}' for k, n in sorted(kinds.items()))}), "
              f"{len(t6.get('flow', []))} flows, {len(t6.get('guard', []))} guards; "
              f"{'committed view MATCHES the sniffer and view.py, ' if mode == '--check' else 'written, '}witness tests PASS")
    bad = write_rows.selftest()
    if bad:
        print("THE MASTER TABLES' SELF-TEST FAILED:")
        for b in bad:
            print("  " + b)
        return 1
    return 0


def main(argv):
    """Check the committed artefacts, or write them.

    The DEFAULT is to CHECK. The trust boundary this kingdom rests on is the
    committed table, so the generator's job is to say whether the committed
    table is still what the generator would produce. Writing first and
    checking afterwards would launder any hand edit: the witness tests would
    only ever see the generator's own output, and the corruption would be gone
    by the time they ran.
    """
    args = argv[1:]
    mode = args[0] if args else "--check"
    if mode not in ("--check", "--write") or len(args) > 1:
        print(f"usage: {os.path.basename(__file__)} [--check | --write]")
        print("  --check  verify the committed artefacts (the default)")
        print("  --write  regenerate them, then verify")
        return 2
    rc = master6_main(mode)
    return rc if rc else pixel_rows_main(mode)


def pixel_kingdom(instance):
    """The pixel family's kingdom of one instance in the master tables (plan 7.7 K4)."""
    return f"futhark.pixel.{instance}"


def pixel_rows_main(mode):
    """The pixel family on the four row kinds (`pixel_rows.py`): each instance's rows written into, or checked against,
    the master tables as the kingdom `futhark.pixel.<instance>`, a variant of `futhark`."""
    import pixel_rows
    bad = []
    for instance in sorted(PIXEL_INSTANCES):
        if PIXEL_INSTANCES[instance] is None:
            interface, literals = [], [f"{t}f32" for t in PIXEL_GENERIC_LITERALS + PIXEL_IMAGE_LITERALS]
        else:
            shader = os.path.join(REPO, PIXEL_INSTANCES[instance])
            interface, literals = pixel_interface(shader), pixel_literals(shader)
        inputs = [f["spelling"] for f in interface]
        name = pixel_kingdom(instance)
        text = pixel_rows.named(instance, inputs, literals, image=PIXEL_INSTANCES[instance] is None)
        rows = write_rows.rows_of_named(text)
        source = PIXEL_INSTANCES[instance] or "an image target: no shader"
        kingdom_row = (name, "futhark", f"the Futhark pixel family's instance {instance} ({source})", pixel_rows.IMAGE, "")
        files = {"interface.tsv": "\n".join(["# input\tgroup\tbinding\tword\tspelling"] + [f"{f['name']}\t{f['group']}\t{f['binding']}\t{f['word']}\t{f['spelling']}" for f in interface]) + "\n"}
        files.update(pixel_rows.egraph_texts(text, literals))
        if mode == "--write":
            write_rows.write_kingdom(name, rows, kingdom_row)
            for f, t in files.items():
                with open(os.path.join(pixel_dir(instance), f), "w") as out:
                    out.write(t)
        else:
            bad += write_rows.check_kingdom(name, rows, kingdom_row)
            for f, t in files.items():
                path = os.path.join(pixel_dir(instance), f)
                got = open(path, "rb").read() if os.path.exists(path) else b""
                if got != t.encode("utf-8"):
                    bad.append(f"the committed pixel/{instance}/{f} is not what the generator produces ({first_difference(got, t.encode('utf-8'))}); run with --write")
    for b in bad:
        print("  " + b)
    print(f"pixel rows: {len(PIXEL_INSTANCES)} instances {'DIFFER FROM' if bad else 'equal'} the master tables")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
