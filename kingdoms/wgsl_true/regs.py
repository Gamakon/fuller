"""Batch 3 and 5: registers and private arrays (names, never values), the
index types that bound a private array's access, and their get and set rows at
every L and C rung of every stratum (the W rungs over the w family)."""
from lib import *
from values import ft, FAMS

# (key, register type, value type, name prefix, count, WGSL type, WGSL zero)
REGS = []
PARRS = []   # (key, array type, value type, index type, prefix, count, element WGSL type, length)
PIDX = {4: 15, 6: 63, 7: 127}   # u32.pidx_<j> masks to the array length - 1


def vecname(s, L):
    return f"vec{L}<{s}>"


def declare(iface):
    # an interface may hold larger kernels (v24: register_scale 4, private arrays of 256): every count is row data, so a
    # register no tree names is never declared
    scale = iface.get("register_scale", 1)
    if iface.get("big_private_arrays"):
        PIDX[8] = 255

    def reg(key, scalar, form, prefix, count, wtype, zero):
        count *= scale
        t = vt(scalar, form) if isinstance(scalar, str) and "<" not in scalar else scalar
        REGS.append((key, f"reg<{wtype}>.{form}", t, prefix, count, wtype, zero))
    reg("f", "f32", "real", "rf", 64, "f32", "0.0")
    reg("i", "u32", "index", "ru", 64, "u32", "0u")
    reg("b", "bool", "flag", "rb", 16, "bool", "false")
    reg("n", "i32", "int", "rn", 16, "i32", "0i")
    reg("h", "f16", "real", "rh", 16, "f16", "0.0h")
    for L, p, c in ((4, "v", 32), (2, "w", 16), (3, "C", 16)):
        reg(f"v{L}", vt("f32", "real", L), "real", p, c, vecname("f32", L), f"{vecname('f32', L)}()")
    for s, form, key in (("u32", "index", "u"), ("i32", "int", "i"), ("bool", "flag", "b"), ("f16", "real", "h")):
        for L in (2, 3, 4):
            reg(f"{key}v{L}", vt(s, form, L), form, f"rv{key}{L}_", 8, vecname(s, L), f"{vecname(s, L)}()")
    for n, p in ((2, "x"), (3, "y"), (4, "z")):
        reg(f"m{n}", f"mat{n}x{n}<f32>.real", "real", p, 4, f"mat{n}x{n}<f32>", f"mat{n}x{n}<f32>()")
    # the uniform register is separate (loops.py)
    for key, rt, vtyp, prefix, count, wtype, zero in REGS:
        T(rt, f"a named {wtype} register {prefix}0..{prefix}{count - 1}, initialised to zero: a name, never a value")
        SAFE_ARGS.add(rt)
        for n in range(count):
            decl(f"var {prefix}{n}: {wtype} = {zero};")
    def parr(key, elem_t, value_t, j, prefix, count, welem):
        count *= scale
        length = {4: 16, 6: 64, 7: 128, 8: 256}[j]
        at = f"parr<{welem},{length}>.{value_t.split('.')[1]}"
        PARRS.append((key, at, value_t, f"u32.pidx_{j}", prefix, count, welem, length))
    parr("f", "f32", "f32.real", 6, "pf", 8, "f32")
    parr("i", "u32", "u32.index", 4, "pi", 5, "u32")
    parr("j", "u32", "u32.index", 6, "pj", 2, "u32")
    parr("k", "u32", "u32.index", 7, "pk", 2, "u32")
    parr("b", "bool", "bool.flag", 4, "pb", 3, "bool")
    parr("v", "vec4<f32>", "vec4<f32>.real", 4, "pv", 6, "vec4<f32>")
    parr("q", "vec2<f32>", "vec2<f32>.real", 4, "pq", 4, "vec2<f32>")
    parr("vq", "vec4<f32>", "vec4<f32>.real", 6, "pvq", 4, "vec4<f32>")
    if iface.get("big_private_arrays"):
        parr("fk", "f32", "f32.real", 7, "pfk", 2, "f32")
        parr("fh", "f32", "f32.real", 8, "pfh", 2, "f32")
        parr("jh", "u32", "u32.index", 8, "pjh", 2, "u32")
    for j, mask in PIDX.items():
        T(f"u32.pidx_{j}", f"an index below {mask + 1}, by the mask its row writes")
        register_twin(f"u32.pidx_{j}")
    for key, at, vtyp, it, prefix, count, welem, length in PARRS:
        T(at, f"a private array of {length} {welem}: {prefix}0..{prefix}{count - 1}")
        SAFE_ARGS.add(at)
        for n in range(count):
            decl(f"var {prefix}{n}: array<{welem}, {length}>;")


def set_rungs():
    return list(rungs("LCS")) + list(rungs("WPXEYAG"))


def build(iface):
    for j, mask in PIDX.items():
        R(f"safe.pidx_{j}", f"u32.pidx_{j}", ["u32.index"], f"(({{0}}) & {mask}u)", "the mask is the row's text; one index type per array length", twin=True)
        if iface.get("module"):
            # the device's restriction of a private array's index to its last element (a seed's array of L elements
            # read past its end reads element L - 1): min(i, last) with last below the kingdom array's length
            T(f"u32.plast_{j}", f"the last index of a private array held in a {mask + 1}-element array: a bare literal", 1, f"0..={mask}")
            SAFE_ARGS.add(f"u32.plast_{j}")
            C(f"literal.plast_{j}", f"u32.plast_{j}", "{v}u", [str(i) for i in range(mask + 1)], f"?_plast_{j}")
            R(f"own.u32_plast_{j}", f"u32.plast_{j}", [], "0u")
            R(f"safe.pidx_c_{j}", f"u32.pidx_{j}", ["u32.index", f"u32.plast_{j}"], "min({0}, {1})", "the index restricted to the array's last element, as the device restricts it", twin=True)
        R(f"own.u32_pidx_{j}", f"u32.pidx_{j}", [], "0u")
        R(f"own.u32_pidx_{j}w", f"u32.pidx_{j}w", [], "0u")
        if DEPTH["A"]:
            R(f"own.u32_pidx_{j}a", f"u32.pidx_{j}a", [], "0u")
    # ---- registers
    for key, rt, vtyp, prefix, count, wtype, zero in REGS:
        for n in range(count):
            R(f"reg.{prefix}{n}", rt, [], f"{prefix}{n}", "a register is a name, never a value: only a get reads it")
        R(f"reg.get_{key}", vtyp, [rt], "{0}", "a register holds any invocation's value: no row makes a uniform or an ownership type from it", twin=True)
        for x, d, k in set_rungs():
            if x in "WPXEYAG":
                R(f"reg.set_{key}_{x}{d}_f{k}", S(x, d, k), [rt, fam(x, vtyp)], "{0} = {1};")
            else:
                R(f"reg.set_{key}_{x}{d}_f{k}", S(x, d, k), [rt, vtyp], "{0} = {1};")
        lanes = TYPES[vtyp]["lanes"]
        if vtyp.startswith("vec"):
            mask = {2: "& 1u", 3: "% 3u", 4: "& 3u"}[lanes]
            # the scalar of a vector value type
            sname = vtyp.split("<")[1].split(">")[0]
            form = vtyp.split(".")[1]
            scalar_t = vt(sname, form)
            if scalar_t in TYPES:
                R(f"reg.get_{key}_lane", scalar_t, [rt, "u32.index"], "{0}[({1}) " + mask + "]", "a register's lane at a run-time index, reduced by the row's text", twin=True)
                for x, d, k in set_rungs():
                    if x in "WPXEYAG":
                        R(f"reg.set_{key}_lane_{x}{d}_f{k}", S(x, d, k), [rt, fam(x, "u32.index"), fam(x, scalar_t)], "{0}[({1}) " + mask + "] = {2};")
                    else:
                        R(f"reg.set_{key}_lane_{x}{d}_f{k}", S(x, d, k), [rt, "u32.index", scalar_t], "{0}[({1}) " + mask + "] = {2};")
    # ---- private arrays
    for key, at, vtyp, it, prefix, count, welem, length in PARRS:
        for n in range(count):
            R(f"parr.{prefix}{n}", at, [], f"{prefix}{n}")
        R(f"parr.get_{key}", vtyp, [at, it], "{0}[{1}]", "in range by the index type", twin=True)
        for x, d, k in set_rungs():
            if x in "WPXEYAG":
                R(f"parr.set_{key}_{x}{d}_f{k}", S(x, d, k), [at, fam(x, it), fam(x, vtyp)], "{0}[{1}] = {2};")
            else:
                R(f"parr.set_{key}_{x}{d}_f{k}", S(x, d, k), [at, it, vtyp], "{0}[{1}] = {2};")
        lanes = TYPES[vtyp]["lanes"]
        if vtyp.startswith("vec"):
            mask = {2: "& 1u", 3: "% 3u", 4: "& 3u"}[lanes]
            sname = vtyp.split("<")[1].split(">")[0]
            scalar_t = vt(sname, vtyp.split(".")[1])
            R(f"parr.get_{key}_lane", scalar_t, [at, it, "u32.index"], "{0}[{1}][({2}) " + mask + "]", twin=True)
            for x, d, k in set_rungs():
                if x in "WPXEYAG":
                    R(f"parr.set_{key}_lane_{x}{d}_f{k}", S(x, d, k), [at, fam(x, it), fam(x, "u32.index"), fam(x, scalar_t)], "{0}[{1}][({2}) " + mask + "] = {3};")
                else:
                    R(f"parr.set_{key}_lane_{x}{d}_f{k}", S(x, d, k), [at, it, "u32.index", scalar_t], "{0}[{1}][({2}) " + mask + "] = {3};")
