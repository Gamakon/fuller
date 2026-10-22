"""Batch 4 and 6: the W phase and workgroup memory, reductions, atomics,
subgroups. The workgroup-memory map of this interface (the design's stratum
map): f32 at 2^11, u32 at 2^10, vec4<f32> at 2^9, i32 at 2^9, vec2<f32> at
2^8, vec3<f32> at 2^7, atomic<u32> at 2^8, atomic<i32> at 2^7; 28,160 bytes,
inside the target's 32,768."""
from lib import *

# key, array type, WGSL element, j, value type, array name
WG = [
    ("f", "wgmem<f32,2048>.real", "f32", 11, "f32.real", "wg_f_11"),
    ("u", "wgmem<u32,1024>.index", "u32", 10, "u32.index", "wg_u_10"),
    ("v4", "wgmem<vec4<f32>,512>.real", "vec4<f32>", 9, "vec4<f32>.real", "wg_v4_9"),
    ("v2", "wgmem<vec2<f32>,256>.real", "vec2<f32>", 8, "vec2<f32>.real", "wg_v2_8"),
    ("v3", "wgmem<vec3<f32>,128>.real", "vec3<f32>", 7, "vec3<f32>.real", "wg_v3_7"),
    ("i", "wgmem<i32,512>.int", "i32", 9, "i32.int", "wg_i_9"),
]
ATOMS = [
    ("u", "wgmem<atomic<u32>,256>.index", "u32", 8, "u32.index", "wg_au_8", "u32.atomicold", "u32.indexw"),
    ("i", "wgmem<atomic<i32>,128>.int", "i32", 7, "i32.int", "wg_ai_7", "i32.atomicold", "i32.intw"),
]
JS = (7, 8, 9, 10, 11)
ATOMIC_FUNS = {"add": "Add", "sub": "Sub", "min": "Min", "max": "Max", "and": "And", "or": "Or", "xor": "Xor"}
T_MIN = 32   # the smallest thread count of the admitted shape table


def reduce_class(j):
    trips = j * -(-(1 << j) // T_MIN)
    i = 0
    while (1 << (6 * i)) < trips:
        i += 1
    return i


def declare(iface):
    for key, at, welem, j, vtyp, name in WG:
        T(at, f"the workgroup array {name}: {1 << j} {welem}")
        SAFE_ARGS.add(at)
        PROLOGUE_TOP.append(f"var<workgroup> {name}: array<{welem}, {1 << j}>;")
    for key, at, welem, j, vtyp, name, old, wval in ATOMS:
        T(at, f"the workgroup array {name} of atomics")
        SAFE_ARGS.add(at)
        PROLOGUE_TOP.append(f"var<workgroup> {name}: array<atomic<{welem}>, {1 << j}>;")
        T(old, "the value an atomic read-modify-write returned: order-dependent across invocations; consumed only by discard")
    for j in JS:
        T(f"u32.wgidx_{j}", f"an index below 2^{j}, by the mask its row writes")
        T(f"u32.wgown_{j}", f"the candidate lidx + k * KNOB_THREADS of this invocation's own element of a 2^{j} array; the access rows guard it")
        SAFE_ARGS.add(f"u32.wgown_{j}")
    for j in sorted({w[3] for w in WG if w[0] in ("f", "v4", "v2", "v3", "i")}):
        T(f"u32.litpow2_{j}", f"a literal power of two from 1 to 2^{j}: a reduction's element count", 1, f"1..={1 << j}")
        SAFE_ARGS.add(f"u32.litpow2_{j}")
    T("u32.lane", "a subgroup lane below the subgroup size, by the mask its row writes")
    T("u32.sgoff", "a subgroup-uniform shuffle offset or mask")
    T("u32.lanelit", "a literal lane 0..=min_subgroup_size - 1 of the target")
    SAFE_ARGS.update(["u32.lane", "u32.sgoff", "u32.lanelit"])


def build(iface):
    for j in JS:
        R(f"safe.wg_of_{j}", f"u32.wgidx_{j}", ["u32.index"], f"(({{0}}) & {(1 << j) - 1}u)", "the mask is the row's text, one index type per stratum")
        R(f"safe.wg_own_{j}", f"u32.wgown_{j}", ["u32.lit64"], "(lidx + {0} * KNOB_THREADS)", "unclamped: for one k the map lidx -> lidx + k * KNOB_THREADS is injective over the workgroup")
        R(f"safe.wg_own_x_{j}", f"u32.wgown_{j}", ["u32.indexw"], "(lidx + {0} * KNOB_THREADS)", "any per-invocation k: injective across invocations whatever each chooses")
        R(f"own.u32_wgidx_{j}", f"u32.wgidx_{j}", [], "0u")
        R(f"own.u32_wgown_{j}", f"u32.wgown_{j}", [], "lidx", "each invocation's own element, injective")
    for j in sorted({w[3] for w in WG if w[0] != "u"}):
        C(f"literal.litpow2_{j}", f"u32.litpow2_{j}", "{v}u", [str(1 << p) for p in range(0, j + 1)], f"?_litpow2_{j}")
        R(f"own.u32_litpow2_{j}", f"u32.litpow2_{j}", [], "1u")
    # the phase
    for x, d, k in rungs("L"):
        R(f"wg.phase_L{d}_f{k}", S("L", d, k), [B("W", d, k), B("L", d, k)], "workgroupBarrier(); {0} workgroupBarrier(); {1}", "the phase: a W block holds no foreign read and writes own elements only")
    for key, at, welem, j, vtyp, name in WG:
        size = 1 << j
        R(f"wg.{key}_{j}", at, [], name, "terminal; the array is a name, read and written only by the access rows")
        zero = "0u" if welem == "u32" else ("0i" if welem == "i32" else ("0.0" if welem == "f32" else f"{welem}()"))
        helper(f"own_get_{name}", f"fn own_get_{name}(i: u32) -> {welem} {{ return select({zero}, {name}[min(i, {size - 1}u)], i < {size}u); }}")
        R(f"wg.get_{key}_{j}", vtyp, [at, f"u32.wgidx_{j}"], "{0}[{1}]", "a foreign read: an ordinary value outside W, in range by its index type; it has no w twin")
        R(f"wg.get_own_{key}_{j}", w_of(vtyp), [at, f"u32.wgown_{j}"], f"own_get_{name}({{1}})", "an own read: a w value", fields=True)
        R(f"wg.get_own_plain_{key}_{j}", vtyp, [at, f"u32.wgown_{j}"], f"own_get_{name}({{1}})", "its plain twin", fields=True)
        for d in range(0, 6):
            for k in STRATA:
                R(f"wg.set_{key}_{j}_W{d}_f{k}", S("W", d, k), [at, f"u32.wgown_{j}", w_of(vtyp)], f"{{ let o = {{1}}; if (o < {size}u) {{ {{0}}[o] = {{2}}; }} }}", "the only write; guarded by the row's text; the index is spelled once (hole 8)")
                R(f"wg.put_{key}_{j}_L{d}_f{k}", S("L", d, k), [at, f"u32.wgown_{j}", vtyp], f"workgroupBarrier(); {{ let o = {{1}}; if (o < {size}u) {{ {{0}}[o] = {{2}}; }} }} workgroupBarrier();", "a one-write phase")
        if key != "u":
            cls = reduce_class(j)
            for op in ("add", "max", "min"):
                body = f"{{0}}[i] + {{0}}[i + s]" if op == "add" else f"{op}({{0}}[i], {{0}}[i + s])"
                text = "{ var s = {1} >> 1u; loop { if (s == 0u) { break; } workgroupBarrier(); var i = lidx; loop { if (i >= s) { break; } {0}[i] = " + body + "; i = i + KNOB_THREADS; } s = s >> 1u; } workgroupBarrier(); }"
                for d in range(0, 6):
                    for k in STRATA:
                        if k + cls <= K:
                            R(f"wg.reduce_{op}_{key}_{j}_L{d}_f{k}", S("L", d, k + cls), [at, f"u32.litpow2_{j}"], text, "the exact reduction in a[0]; barriers separate steps", fields=True)
    # the stratum map's u32 array also reads under uload below (uniform loads)
    for d in range(0, 6):
        for k in STRATA:
            R(f"wg.uload_L{d}_f{k}", S("L", d, k), ["ureg<u32>.uniform", "wgmem<u32,1024>.index", "u32.uniform"], "{0} = workgroupUniformLoad(&{1}[({2}) & 1023u]);", "the pointer is uniform and the call is a control barrier: L rungs only")
    # ---- atomics
    for key, at, welem, j, vtyp, name, old, wval in ATOMS:
        size = 1 << j
        rt = "reg<u32>.index" if welem == "u32" else "reg<i32>.int"
        R(f"wg.a{key}_{j}", at, [], name)
        for fn, F in ATOMIC_FUNS.items():
            R(f"atomic.{fn}_{key}_{j}", old, [at, f"u32.wgidx_{j}", vtyp], f"atomic{F}(&{{0}}[{{1}}], {{2}})", "race-free at any index; the old value is order-dependent so it is atomicold")
        R(f"own.{old.replace('.', '_')}", old, [], "lit_u32(0u)" if welem == "u32" else "lit_i32(0i)", "a value whose only consumer discards it")
        for x, d, k in rungs("LCO"):
            R(f"stmt.discard_{key}_{x}{d}_f{k}", S(x, d, k), [old], "_ = {0};", "the effect without the value")
        for d in range(0, 6):
            for k in STRATA:
                R(f"atomic.uload_{key}_{j}_L{d}_f{k}", S("L", d, k), [rt, at, f"u32.wgidx_{j}"], "workgroupBarrier(); {0} = atomicLoad(&{1}[{2}]); workgroupBarrier();", "the strict load: the barriers order it after every read-modify-write before it")
                R(f"atomic.store_{key}_{j}_W{d}_f{k}", S("W", d, k), [at, f"u32.wgown_{j}", wval], f"{{ let o = {{1}}; if (o < {size}u) {{ atomicStore(&{{0}}[o], {{2}}); }} }}", "a store at the invocation's own element inside a phase")
    # ---- subgroups
    if iface["has_subgroups"]:
        sg = min(iface["subgroup_sizes"])
        R("builtin.subgroup_invocation_id", "u32.index", [], "sg_lane", twin=True)
        R("uniform.subgroup_size", "u32.uniform", [], "sg_size")
        R("uniform.num_subgroups", "u32.uniform", [], "sg_num")
        R("builtin.subgroup_id", "u32.index", [], "sg_id", twin=True)
        R("lane.of_index", "u32.lane", ["u32.index"], "(({0}) & (sg_size - 1u))", "below the subgroup size by the row's text")
        R("lane.lit", "u32.lane", ["u32.lanelit"], "{0}")
        R("own.u32_lane", "u32.lane", [], "0u", "lane 0 is active in every full subgroup")
        C("literal.lanelit", "u32.lanelit", "{v}u", [str(i) for i in range(sg)], "?_lanelit: spelled bare, subgroupBroadcast needs a const-expression lane")
        R("own.u32_lanelit", "u32.lanelit", [], "0u")
        R("sgoff.lit", "u32.sgoff", ["u32.lanelit"], "{0}")
        R("sgoff.of_uniform", "u32.sgoff", ["u32.uniform"], "{0}", "a workgroup-uniform value is subgroup-uniform")
        R("own.u32_sgoff", "u32.sgoff", [], "0u")
        helper("shuffle_down_f", "fn shuffle_down_f(x: f32, d: u32, lane: u32, size: u32) -> f32 { let dd = min(d, size); return select(subgroupShuffleDown(x, dd), x, lane + dd >= size); }")
        helper("shuffle_up_f", "fn shuffle_up_f(x: f32, d: u32, lane: u32, size: u32) -> f32 { let dd = min(d, size); return select(subgroupShuffleUp(x, dd), x, lane < dd); }")
        helper("shuffle_down_i", "fn shuffle_down_i(x: u32, d: u32, lane: u32, size: u32) -> u32 { let dd = min(d, size); return select(subgroupShuffleDown(x, dd), x, lane + dd >= size); }")
        helper("shuffle_up_i", "fn shuffle_up_i(x: u32, d: u32, lane: u32, size: u32) -> u32 { let dd = min(d, size); return select(subgroupShuffleUp(x, dd), x, lane < dd); }")
        OPS = {"add": "Add", "mul": "Mul", "min": "Min", "max": "Max", "inclusive_add": "InclusiveAdd", "inclusive_mul": "InclusiveMul", "exclusive_add": "ExclusiveAdd", "exclusive_mul": "ExclusiveMul"}
        for x, d, k in rungs("L"):
            s = S("L", d, k)
            sfx = f"_L{d}_f{k}"
            for suffix, rt, vtyp in (("f", "reg<f32>.real", "f32.real"), ("i", "reg<u32>.index", "u32.index")):
                for op, F in OPS.items():
                    R(f"subgroup.{op}_{suffix}" + sfx, s, [rt, vtyp], "{0} = subgroup" + F + "({1});", "a collective is a statement at L rungs only")
                R(f"subgroup.broadcast_first_{suffix}" + sfx, s, [rt, vtyp], "{0} = subgroupBroadcastFirst({1});")
                R(f"subgroup.broadcast_{suffix}" + sfx, s, [rt, vtyp, "u32.lanelit"], "{0} = subgroupBroadcast({1}, {2});", "WGSL requires a constant lane")
                R(f"subgroup.shuffle_{suffix}" + sfx, s, [rt, vtyp, "u32.lane"], "{0} = subgroupShuffle({1}, {2});")
                R(f"subgroup.shuffle_xor_{suffix}" + sfx, s, [rt, vtyp, "u32.sgoff"], "{0} = subgroupShuffleXor({1}, ({2}) & (sg_size - 1u));")
                R(f"subgroup.shuffle_down_{suffix}" + sfx, s, [rt, vtyp, "u32.sgoff"], "{0} = shuffle_down_" + suffix + "({1}, {2}, sg_lane, sg_size);")
                R(f"subgroup.shuffle_up_{suffix}" + sfx, s, [rt, vtyp, "u32.sgoff"], "{0} = shuffle_up_" + suffix + "({1}, {2}, sg_lane, sg_size);")
            R("subgroup.all" + sfx, s, ["reg<bool>.flag", "bool.flag"], "{0} = subgroupAll({1});")
            R("subgroup.any" + sfx, s, ["reg<bool>.flag", "bool.flag"], "{0} = subgroupAny({1});")
