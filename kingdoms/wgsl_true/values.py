"""Batch 2: builtins and axes, u32 / i32 / f32 / f16 arithmetic, comparisons,
logic, selects, conversions, input loads, uniform fields, literals. Every
scalar row whose operands are one scalar type exists per lane count
(batch 5, "Lane counts"); every value row has a w twin (batch 4)."""
import random
from lib import *

FAMS = {
    "real": ("f32", "real", "a number"),
    "h": ("f16", "real", "a half-precision number"),
    "index": ("u32", "index", "a u32 used as a position"),
    "count": ("u32", "count", "a u32 used as a size"),
    "int": ("i32", "int", "a signed integer"),
    "flag": ("bool", "flag", "a predicate"),
}
LANES = (1, 2, 3, 4)
F32MAX = 3.4028234663852886e38


def ft(fam, lanes=1):
    s, f, _ = FAMS[fam]
    return vt(s, f, lanes)


def sfx(fam, lanes=1):
    return ("_f16" if fam == "h" else "") + ("" if lanes == 1 else f"_v{lanes}")


def rng_of(name):
    return random.Random(sum(ord(c) * (i + 1) for i, c in enumerate(name)))


def int_values(lo, hi, n, name):
    r = rng_of(name)
    vals = {lo, hi}
    for p in range(0, 33):
        for v in ((1 << p) - 1, 1 << p, (1 << p) + 1):
            if lo <= v <= hi:
                vals.add(v)
    for v in (0, 1, 2, 3, 5, 7, 10, 31, 32, 33, 63, 64, 100, 255, 1000, 4095):
        if lo <= v <= hi:
            vals.add(v)
    if lo < 0:
        for v in (-1, -2, -3, -32, -100, -1000):
            if lo <= v <= hi:
                vals.add(v)
    vals = sorted(vals)
    r.shuffle(vals)
    keep = vals[: max(n // 2, 1)]
    while len(keep) < n:
        keep.append(r.randint(lo, hi))
    return keep[:n]


def f32_values(lo, hi, n, name, half=False):
    r = rng_of(name)
    rnd = f16 if half else f32
    interesting = [0.0, 1.0, 0.5, 2.0, 3.0, 10.0, 100.0, 0.1, 1e-3, 1e-6, 1e3, 1e6, 3.14159265, 2.71828183, 0.70710678]
    if not half:
        interesting += [1e10, 1e20, 1e30, F32MAX, 1.17549435e-38, 1e-40]
    else:
        interesting += [65504.0, 6.1035e-5, 5.96e-8]
    if lo < 0:
        interesting += [-0.0, -1.0, -0.5, -2.0, -100.0, -1e-3, lo]
    vals = []
    for v in interesting + [hi]:
        if abs(v) > (65504.0 if half else F32MAX):
            continue
        v = rnd(v)
        if lo <= v <= hi:
            vals.append(v)
    while len(vals) < n:
        mag = 10 ** r.uniform(-8, 8 if not half else 4)
        v = rnd(mag * (r.choice((-1, 1)) if lo < 0 else 1))
        if lo <= v <= hi:
            vals.append(v)
    return [ftext(v) for v in vals[:n]]


def declare(iface):
    for fam, (s, f, mean) in FAMS.items():
        for L in LANES:
            t = vt(s, f, L)
            lit = {
                "real": f"-{F32MAX}..={F32MAX}", "h": "-65504.0..=65504.0", "index": "0..=4294967295", "count": "0..=4294967295",
                "int": "-2147483648..=2147483647", "flag": "0,1",
            }[fam] if L == 1 else "-"
            T(t, mean if L == 1 else f"a vector of {L} lanes: {mean}", L, lit)
            register_twin(t)
    T("f32.nonneg", "a non-negative width", 1, f"0.0..={F32MAX}")
    T("i32.nonneg", "a non-negative width", 1, "0..=2147483647")
    T("bounds<f32>.ordered", "a pair (lo, hi) with lo <= hi by construction")
    T("bounds<i32>.ordered", "a pair (lo, hi) with lo <= hi by construction")
    T("u32.lit4096", "a literal 1..=4096", 1, "1..=4096")
    T("u32.uniform", "a u32 the same in every invocation of the workgroup")
    T("bool.uflag", "a predicate the same in every invocation")
    for t in ("f32.nonneg", "i32.nonneg", "bounds<f32>.ordered", "bounds<i32>.ordered", "u32.lit4096", "u32.uniform", "bool.uflag"):
        SAFE_ARGS.add(t)


CMP = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}
BIN = {"add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})", "div": "({0} / {1})", "min": "min({0}, {1})", "max": "max({0}, {1})",
       "pow": "pow({0}, {1})", "atan2": "atan2({0}, {1})", "step": "step({0}, {1})"}
UN = ["neg", "abs", "sign", "floor", "ceil", "round", "fract", "trunc", "saturate", "sqrt", "inverse_sqrt", "exp", "exp2", "log", "log2",
      "sin", "cos", "tan", "tanh", "asin", "acos", "atan"]
WGSL_NAME = {"inverse_sqrt": "inverseSqrt"}
TRI = {"mix": "mix({0}, {1}, {2})", "fma": "fma({0}, {1}, {2})", "smoothstep": "smoothstep({0}, {1}, {2})"}
IDX_BIN = {"add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})", "div": "({0} / {1})", "mod": "({0} % {1})", "min": "min({0}, {1})",
           "max": "max({0}, {1})", "shl": "({0} << {1})", "shr": "({0} >> {1})", "and": "({0} & {1})", "or": "({0} | {1})"}
INT_BIN = {"add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})", "div": "({0} / {1})", "mod": "({0} % {1})", "min": "min({0}, {1})",
           "max": "max({0}, {1})", "and": "({0} & {1})", "or": "({0} | {1})", "xor": "({0} ^ {1})"}
INT_UN = {"neg": "(-{0})", "abs": "abs({0})", "sign": "sign({0})", "not": "(~{0})"}


def build(iface):
    # ---- builtins and axes
    R("builtin.local_invocation_id", "vec3<u32>.lid", [], "lid", "its own type, so a mutation cannot swap lid for gid inside an ownership row")
    R("builtin.workgroup_id", "vec3<u32>.wid", [], "wid")
    R("builtin.local_invocation_index", "u32.lidx", [], "lidx")
    R("builtin.num_workgroups", "vec3<u32>.nwg", [], "nwg")
    for i in ("gid", "lid", "wid"):
        R(f"axis.x_{i}", "u32.colid", [f"vec3<u32>.{i}"], "col_axis({0})", "the column axis the grid binds")
        R(f"axis.y_{i}", "u32.rowid", [f"vec3<u32>.{i}"], "row_axis({0})", "the row axis the grid binds")
        R(f"axis.z_{i}", "u32.index", [f"vec3<u32>.{i}"], "{0}.z", "z is a plain index", twin=True)
    for c in "xyz":
        R(f"count.nwg_{c}", "u32.count", ["vec3<u32>.nwg"], "{0}." + c, "a workgroup count is a size", twin=True)
    R("index.of_row", "u32.index", ["u32.rowid"], "{0}", "an axis component as a plain position; one way only", twin=True)
    R("index.of_col", "u32.index", ["u32.colid"], "{0}", twin=True)
    R("index.of_lidx", "u32.index", ["u32.lidx"], "{0}", twin=True)
    # ---- loads
    for f in iface["fields"]:
        R(f"load.{f}", "u32.count", [], f"cfg.{f}", "a field of the uniform block is a size", twin=True)
    for i in iface["inputs"]:
        n = i["name"]
        if i.get("kind") == "atomic_u32":
            # a device buffer of atomic<u32> the host writes; this kernel only reads it, through atomicLoad
            R(f"load.{n}", "u32.count", ["u32.index"], f"atomicLoad(&{n}[min({{0}}, arrayLength(&{n}) - 1u)])", "an atomic read of any position, clamped to the last element by the row's own text", twin=True)
            R(f"load.{n}_len", "u32.count", [], f"arrayLength(&{n})", twin=True)
            continue
        R(f"load.{n}", i["ty"], ["u32.index"], f"{n}[min({{0}}, arrayLength(&{n}) - 1u)]", "a read at any position, clamped to the last element by the row's own text", twin=True)
        R(f"load.{n}_len", "u32.count", [], f"arrayLength(&{n})", twin=True)
    for o in iface["outputs"]:
        n = o["name"]
        if o.get("kind") == "texture":
            continue   # a write-only storage texture has no load row
        helper(f"own_load_{n}", f"fn own_load_{n}(r: u32, c: u32) -> f32 {{ return {n}[min(r * {o['cols']} + c, arrayLength(&{n}) - 1u)]; }}")
        R(f"load.{n}_own", o["ty"], [f"u32.threadrow_{n}", f"u32.threadcol_{n}"], f"own_load_{n}({{0}}, {{1}})", "a read of the invocation's own cell only", twin=True)
        R(f"load.{n}_len", "u32.count", [], f"arrayLength(&{n})", twin=True)
    # ---- literals
    C("literal.index", "u32.index", "lit_u32({v}u)", extra(iface, "index", int_values(0, 2**32 - 1, 48, "index")), "?_index", twin=True)
    C("literal.count", "u32.count", "lit_u32({v}u)", extra(iface, "count", int_values(0, 2**32 - 1, 48, "count")), "?_count", twin=True)
    C("literal.real", "f32.real", "lit_f32({v})", extra(iface, "real", f32_values(-F32MAX, F32MAX, 48, "real")), "?_real: a finite f32, so no NaN or infinity literal exists", twin=True)
    C("literal.nonneg", "f32.nonneg", "lit_f32({v})", f32_values(0.0, F32MAX, 32, "nonneg"), "?_nonneg: a non-negative width")
    C("literal.f16", "f16.real", "lit_f16({v}h)", f32_values(-65504.0, 65504.0, 32, "f16", True), "?_f16: the finite f16 range", twin=True)
    C("literal.int", "i32.int", "lit_i32({v})", extra(iface, "int", [("(-2147483647i - 1i)" if v == -2147483648 else f"{v}i") for v in int_values(-2**31, 2**31 - 1, 48, "int")]), "?_int", twin=True)
    C("literal.nonneg_int", "i32.nonneg", "lit_i32({v}i)", [str(v) for v in int_values(0, 2**31 - 1, 32, "nonneg_int")], "?_nonneg_int")
    C("literal.lit4096", "u32.lit4096", "lit_u32({v}u)", [str(v) for v in int_values(1, 4096, 32, "lit4096")], "?_lit4096")
    R("own.u32_lit4096", "u32.lit4096", [], "lit_u32(1u)")
    R("own.u32_lit64", "u32.lit64", [], "lit_u32(1u)")
    R("own.f32_nonneg", "f32.nonneg", [], "lit_f32(0.0)")
    R("own.i32_nonneg", "i32.nonneg", [], "lit_i32(0i)")
    R("own.bounds_f32", "bounds<f32>.ordered", [], "vec2<f32>(0.0, 0.0)", "an ordered pair")
    R("own.bounds_i32", "bounds<i32>.ordered", [], "vec2<i32>(0i, 0i)", "an ordered pair")
    # ---- index and count
    R("index.of_count", "u32.index", ["u32.count"], "{0}", "a size used as a position", twin=True)
    for L in LANES:
        t = vt("u32", "index", L)
        for op, sp in IDX_BIN.items():
            R(f"index.{op}" + sfx("index", L), t, [t, t], sp, "wraps modulo 2^32 as WGSL defines; a wrapped value is a wrong position, never an unsafe one" if op in ("add", "sub", "mul") else "", twin=True)
        c = vt("u32", "count", L)
        for op in ("add", "min", "max"):
            R(f"count.{op}" + sfx("count", L), c, [c, c], IDX_BIN[op], twin=True)
    helper("sub_floor", "fn sub_floor(a: u32, b: u32) -> u32 { return a - min(a, b); }")
    R("index.sub_floor", "u32.index", ["u32.index", "u32.index"], "sub_floor({0}, {1})", "the floored difference (a prologue helper spells each argument once)", twin=True)
    R("count.sub_floor", "u32.count", ["u32.count", "u32.count"], "sub_floor({0}, {1})", twin=True)
    R("index.clamp_to", "u32.index", ["u32.index", "u32.count"], "min({0}, max({1}, 1u) - 1u)", "i < n, and 0 for n = 0", twin=True)
    R("index.shl_m", "u32.index", ["u32.index", "u32.index"], "({0} << (({1}) & 31u))", "the explicitly masked spelling", twin=True)
    R("index.shr_m", "u32.index", ["u32.index", "u32.index"], "({0} >> (({1}) & 31u))", twin=True)
    # ---- float arithmetic
    for fam in ("real", "h"):
        for L in LANES:
            t = ft(fam, L)
            for op, sp in BIN.items():
                R(f"arith.{op}" + sfx(fam, L), t, [t, t], sp, "IEEE arithmetic: a special value is scored, never a fault", twin=True)
            for u in UN:
                sp = "(-{0})" if u == "neg" else WGSL_NAME.get(u, u) + "({0})"
                R(f"arith.{u}" + sfx(fam, L), t, [t], sp, twin=True)
            for op, sp in TRI.items():
                R(f"arith.{op}" + sfx(fam, L), t, [t, t, t], sp, twin=True)
    # ---- bounds, clamp
    helper("lo_width", "fn lo_width(lo: f32, w: f32) -> vec2<f32> { return vec2<f32>(lo, lo + w); }")
    helper("sorted", "fn sorted(a: f32, b: f32) -> vec2<f32> { return vec2<f32>(min(a, b), max(a, b)); }")
    helper("clamp_p", "fn clamp_p(x: f32, p: vec2<f32>) -> f32 { return clamp(x, p.x, p.y); }")
    R("bounds.of_lo_width", "bounds<f32>.ordered", ["f32.real", "f32.nonneg"], "lo_width({0}, {1})", "an ordered pair by its construction")
    R("bounds.sorted", "bounds<f32>.ordered", ["f32.real", "f32.real"], "sorted({0}, {1})", "the reordering pair")
    R("arith.clamp", "f32.real", ["f32.real", "bounds<f32>.ordered"], "clamp_p({0}, {1})", "the pair type guarantees low <= high", twin=True)
    helper("lo_width_i", "fn lo_width_i(lo: i32, w: i32) -> vec2<i32> { return vec2<i32>(lo, max(lo, lo + w)); }")
    helper("sorted_i", "fn sorted_i(a: i32, b: i32) -> vec2<i32> { return vec2<i32>(min(a, b), max(a, b)); }")
    helper("clamp_pi", "fn clamp_pi(x: i32, p: vec2<i32>) -> i32 { return clamp(x, p.x, p.y); }")
    R("bounds.of_lo_width_int", "bounds<i32>.ordered", ["i32.int", "i32.nonneg"], "lo_width_i({0}, {1})")
    R("bounds.sorted_int", "bounds<i32>.ordered", ["i32.int", "i32.int"], "sorted_i({0}, {1})")
    R("int.clamp", "i32.int", ["i32.int", "bounds<i32>.ordered"], "clamp_pi({0}, {1})", twin=True)
    # ---- integer arithmetic
    for L in LANES:
        t = vt("i32", "int", L)
        for op, sp in INT_BIN.items():
            R(f"int.{op}" + sfx("int", L), t, [t, t], sp, "+ - * wrap; / and % by zero and MIN / -1 are defined (WGSL 8.8)" if op == "div" else "", twin=True)
        for u, sp in INT_UN.items():
            R(f"int.{u}" + sfx("int", L), t, [t], sp, twin=True)
    R("int.shl", "i32.int", ["i32.int", "u32.index"], "({0} << {1})", "WGSL 8.10: a runtime count is taken modulo 32", twin=True)
    R("int.shr", "i32.int", ["i32.int", "u32.index"], "({0} >> {1})", twin=True)
    # ---- conversions
    helper("nan0_u32", "fn nan0_u32(x: f32) -> u32 { return u32(select(x, 0.0, x != x)); }")
    helper("nan0_i32", "fn nan0_i32(x: f32) -> i32 { return i32(select(x, 0.0, x != x)); }")
    R("convert.index_to_f32", "f32.real", ["u32.index"], "f32({0})", twin=True)
    R("convert.count_to_f32", "f32.real", ["u32.count"], "f32({0})", twin=True)
    R("convert.f32_to_index", "u32.index", ["f32.real"], "u32({0})", "WGSL 15.7.6: a non-NaN value is clamped and rounded toward zero; a NaN gives an indeterminate u32 that fitness scores", twin=True)
    R("convert.f32_to_index_nan0", "u32.index", ["f32.real"], "nan0_u32({0})", "the deterministic variant", twin=True)
    R("convert.f32_to_count", "u32.count", ["f32.real"], "u32({0})", twin=True)
    R("convert.index_to_int", "i32.int", ["u32.index"], "i32({0})", "a reinterpretation of bits, defined", twin=True)
    R("convert.int_to_index", "u32.index", ["i32.int"], "u32({0})", twin=True)
    R("index.of_int_floor", "u32.index", ["i32.int"], "u32(max({0}, 0i))", twin=True)
    R("convert.f32_to_int", "i32.int", ["f32.real"], "i32({0})", twin=True)
    R("convert.f32_to_int_nan0", "i32.int", ["f32.real"], "nan0_i32({0})", twin=True)
    R("convert.int_to_f32", "f32.real", ["i32.int"], "f32({0})", twin=True)
    R("convert.f32_to_f16", "f16.real", ["f32.real"], "f16({0})", "numeric conversion: out of range gives an infinity", twin=True)
    R("convert.f16_to_f32", "f32.real", ["f16.real"], "f32({0})", "exact", twin=True)
    # ---- comparisons, logic, select
    for fam in ("real", "h", "index", "count", "int"):
        for L in LANES:
            t = ft(fam, L)
            o = vt("bool", "flag", L)
            for c, op in CMP.items():
                R(f"compare.{c}_{fam}" + ("" if L == 1 else f"_v{L}"), o, [t, t], "({0} " + op + " {1})", twin=True)
    for L in LANES:
        t = vt("bool", "flag", L)
        if L == 1:
            R("logic.and", t, [t, t], "({0} && {1})", twin=True)
            R("logic.or", t, [t, t], "({0} || {1})", twin=True)
            R("logic.not", t, [t], "(!{0})", twin=True)
        else:
            R(f"logic.and_v{L}", t, [t, t], "({0} & {1})", twin=True)
            R(f"logic.or_v{L}", t, [t, t], "({0} | {1})", twin=True)
            R(f"logic.not_v{L}", t, [t], "(!{0})", twin=True)
            R(f"logic.all_v{L}", "bool.flag", [t], "all({0})", twin=True)
            R(f"logic.any_v{L}", "bool.flag", [t], "any({0})", twin=True)
    for fam in FAMS:
        t = ft(fam)
        R(f"select.{fam}", t, [t, t, "bool.flag"], "select({0}, {1}, {2})", twin=True)
        if fam != "flag":
            for L in (2, 3, 4):
                v = ft(fam, L)
                R(f"select.lane_{fam}_v{L}", v, [v, v, vt("bool", "flag", L)], "select({0}, {1}, {2})", "per lane", twin=True)
