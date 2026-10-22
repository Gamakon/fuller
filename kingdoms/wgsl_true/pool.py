"""Batch 9 (s9, 2026-10-22): workgroup memory as two pools, for an interface that declares `wg_pool` (v24).

A seed declares any number of workgroup arrays of any length and element type; the kingdom declares two:

* `wg_pool: array<u32, P>`, every non-atomic workgroup array of a kernel (u32, f32 and i32 bit-cast, vec4<f32> as four
  lane regions), each a **region** of the pool at a base the reader allocates;
* `wg_apool: array<atomic<u32>, A>`, every atomic workgroup array.

P + A words are the target's workgroup memory (32,768 bytes), so no gene exceeds it whatever it touches.

**The rule that keeps a write phase race-free: region bases are multiples of the workgroup size T.** A base is spelled
`(m * KNOB_THREADS)` (`u32.wgbase` from the literal block count `u32.wgblk`). An own element is `lidx + k * T`
(`u32.wgown`, any per-invocation `k`). Two own writes `b1 + t1 + k1 T` and `b2 + t2 + k2 T` of one phase meet only if
`t1 = t2 (mod T)`, that is `t1 = t2`: one invocation. A write is guarded by the pool's length, never clamped (a clamp
would collapse lanes onto the last word), and the sum wraps modulo 2^32, which T (a power of two) divides, so the rule
holds through the wrap. A vec4 array is four lane regions (structure of arrays), each its own T-aligned region. The
reader stores a seed's array written block by block (`t * C + j`) transposed (`j * T + t`), a bijection it applies to
every access of that array, so every write of a phase is an own write.

The phases (L rungs only, so in uniform control):

* `wg.phase`: `barrier; W; barrier; rest` (batch 4) over the pool: W writes own elements, reads own elements, never a
  foreign one;
* `wg.pair` (the tree step): `barrier; { let ps = min(s, T / 2); if (lidx < ps) { P } } barrier; rest`. P is W plus
  the partner load `r = A[lidx + ps]`: the partner elements are at `ps..2ps` modulo T and every own write of an
  invocation below `ps` is below `ps` modulo T, so no partner element is written in the phase;
* `wg.solo`: `barrier; if (lidx == 0u) { S } barrier; rest`: S is one invocation's code (C statements plus a write
  at any element): the phase has no other writer and no other reader.

Outside a phase every element is an ordinary value (`wg.get_*`, foreign reads). The atomic pool is relaxed (the
interface's determinism, batch 8): a read-modify-write, load or store at any element is one atomic operation, never a
race, its old value order-dependent (`u32.atomicold`). `workgroupUniformLoad` of a pool word is a uniform value.
"""
from lib import *

P_WORDS = 6080
A_WORDS = 2112
# element families of the plain pool: key, value type, read cast, write cast
ELEMS = [
    ("u", "u32.index", "{}", "{}"),
    ("f", "f32.real", "bitcast<f32>({})", "bitcast<u32>({})"),
    ("n", "i32.int", "bitcast<i32>({})", "bitcast<u32>({})"),
]
ATOMIC_FUNS = {"add": "Add", "sub": "Sub", "min": "Min", "max": "Max", "and": "And", "or": "Or", "xor": "Xor", "exchange": "Exchange"}
REGS = {"u": "reg<u32>.index", "f": "reg<f32>.real", "n": "reg<i32>.int", "v4": "reg<vec4<f32>>.real"}


def declare(iface):
    T("u32.atomicold", "the value an atomic returned: order-dependent across invocations; consumed by discard, or (relaxed) as a position")
    T("u32.wgblk", "a region's base in blocks of the workgroup size: a bare literal", 1, "0..=511")
    T("u32.wgbase", "a region's base: a multiple of the workgroup size T, so own writes of two regions never meet")
    T("u32.wgown", "the candidate lidx + k * T of this invocation's own element of a region; the write rows guard it")
    T("u32.wgabase", "a region's first word in the atomic pool: a bare literal (an atomic never races, so no alignment)", 1, f"0..={A_WORDS - 1}")
    for t in ("u32.wgblk", "u32.wgbase", "u32.wgown", "u32.wgabase"):
        SAFE_ARGS.add(t)
    PROLOGUE_TOP.append(f"var<workgroup> wg_pool: array<u32, {P_WORDS}>;")
    PROLOGUE_TOP.append(f"var<workgroup> wg_apool: array<atomic<u32>, {A_WORDS}>;")


def build(iface):
    C("literal.wgblk", "u32.wgblk", "{v}u", [str(i) for i in range(0, 512)], "?_wgblk: a block count, bare (its product with KNOB_THREADS is at most 511 * 256)")
    R("own.u32_wgblk", "u32.wgblk", [], "0u")
    R("wg.base", "u32.wgbase", ["u32.wgblk"], "({0} * KNOB_THREADS)", "a region's base: m blocks of the workgroup size")
    R("own.u32_wgbase", "u32.wgbase", [], "0u", "the pool's first region")
    R("wg.own_x", "u32.wgown", ["u32.indexw"], "(lidx + {0} * KNOB_THREADS)", "any per-invocation k: lidx + k * T is injective across invocations")
    R("own.u32_wgown", "u32.wgown", [], "lidx", "each invocation's own element of the region's first T")
    last = P_WORDS - 1
    for key, vt_, rd, wr in ELEMS:
        cell = f"wg_pool[min({{0}} + {{1}}, {last}u)]"
        R(f"wg.get_{key}", vt_, ["u32.wgbase", "u32.index"], rd.format(cell), "a foreign read of a region's element: an ordinary value outside a phase; no w twin")
        R(f"wg.get_own_{key}", w_of(vt_), ["u32.wgbase", "u32.wgown"], rd.format(cell), "an own read: a w value")
        R(f"wg.get_own_plain_{key}", vt_, ["u32.wgbase", "u32.wgown"], rd.format(cell), "its plain twin")
    helper("wg_v4", f"fn wg_v4(b: u32, ls: u32, i: u32) -> vec4<f32> {{ return vec4<f32>(bitcast<f32>(wg_pool[min(b + i, {last}u)]), bitcast<f32>(wg_pool[min(b + ls + i, {last}u)]), bitcast<f32>(wg_pool[min(b + 2u * ls + i, {last}u)]), bitcast<f32>(wg_pool[min(b + 3u * ls + i, {last}u)])); }}")
    helper("wg_set_v4", f"fn wg_set_v4(o: u32, ls: u32, v: vec4<f32>) {{ if (o < {P_WORDS}u) {{ wg_pool[o] = bitcast<u32>(v.x); }} if (o + ls < {P_WORDS}u) {{ wg_pool[o + ls] = bitcast<u32>(v.y); }} if (o + 2u * ls < {P_WORDS}u) {{ wg_pool[o + 2u * ls] = bitcast<u32>(v.z); }} if (o + 3u * ls < {P_WORDS}u) {{ wg_pool[o + 3u * ls] = bitcast<u32>(v.w); }} }}")
    R("wg.get_v4", "vec4<f32>.real", ["u32.wgbase", "u32.wgbase", "u32.index"], "wg_v4({0}, {1}, {2})", "element i of a vec4 array kept as four lane regions (lane stride the second base)")
    R("wg.get_own_v4", "vec4<f32>.realw", ["u32.wgbase", "u32.wgbase", "u32.wgown"], "wg_v4({0}, {1}, {2})", "an own read: a w value")
    R("wg.get_own_plain_v4", "vec4<f32>.real", ["u32.wgbase", "u32.wgbase", "u32.wgown"], "wg_v4({0}, {1}, {2})", "its plain twin")
    for x, d, k in rungs("WP"):
        s = S(x, d, k)
        for key, vt_, rd, wr in ELEMS:
            R(f"wg.set_{key}_{x}{d}_f{k}", s, ["u32.wgbase", "u32.wgown", w_of(vt_)], "{ let o = {0} + {1}; if (o < " + str(P_WORDS) + "u) { wg_pool[o] = " + wr.format("{2}") + "; } }", "an own write, guarded by the pool's length (never clamped)")
        R(f"wg.set_v4_{x}{d}_f{k}", s, ["u32.wgbase", "u32.wgbase", "u32.wgown", "vec4<f32>.realw"], "wg_set_v4({0} + {2}, {1}, {3});", "an own write of a vec4 element: its four lanes, each an own write of its lane region")
    for x, d, k in rungs("S"):
        s = S(x, d, k)
        for key, vt_, rd, wr in ELEMS:
            R(f"wg.put_{key}_S{d}_f{k}", s, ["u32.wgbase", "u32.index", vt_], "{ let o = {0} + {1}; if (o < " + str(P_WORDS) + "u) { wg_pool[o] = " + wr.format("{2}") + "; } }", "the solo invocation writes any element: no other invocation touches the pool in a solo phase")
        R(f"wg.put_v4_S{d}_f{k}", s, ["u32.wgbase", "u32.wgbase", "u32.index", "vec4<f32>.real"], "wg_set_v4({0} + {2}, {1}, {3});")
    for x, d, k in rungs("P"):
        s = S(x, d, k)
        for key, vt_, rd, wr in ELEMS:
            R(f"wg.pload_{key}_P{d}_f{k}", s, [REGS[key], "u32.wgbase"], "{0} = " + rd.format(f"wg_pool[min({{1}} + lidx + ps, {last}u)]") + ";", "the partner element, lidx + ps: no invocation writes it in a pair phase")
        R(f"wg.pload_v4_P{d}_f{k}", s, [REGS["v4"], "u32.wgbase", "u32.wgbase"], "{0} = wg_v4({1}, {2}, lidx + ps);")
    for x, d, k in rungs("L"):
        s = S("L", d, k)
        R(f"wg.phase_L{d}_f{k}", s, [B("W", d, k), B("L", d, k)], "workgroupBarrier(); {0} workgroupBarrier(); {1}", "the write phase: own writes and own reads only")
        R(f"wg.pair_L{d}_f{k}", s, ["u32.uniform", B("P", d, k), B("L", d, k)], "workgroupBarrier(); { let ps = min({0}, KNOB_THREADS / 2u); if (lidx < ps) { {1} } } workgroupBarrier(); {2}", "the tree step: invocations below ps write own elements and read their partners")
        R(f"wg.solo_L{d}_f{k}", s, [B("S", d, k), B("L", d, k)], "workgroupBarrier(); if (lidx == 0u) { {0} } workgroupBarrier(); {1}", "one invocation's phase")
        R(f"wg.uload_L{d}_f{k}", s, ["ureg<u32>.uniform", "u32.wgbase", "u32.uniform"], "{0} = workgroupUniformLoad(&wg_pool[min({1} + {2}, " + str(last) + "u)]);", "the pointer is uniform and the call is a control barrier: L rungs only; the value is uniform")
    R("own.u32_atomicold", "u32.atomicold", [], "lit_u32(0u)", "a value whose consumer discards it")
    C("literal.wgabase", "u32.wgabase", "{v}u", [str(i) for i in range(0, A_WORDS)], "?_wgabase: a word of the atomic pool")
    R("own.u32_wgabase", "u32.wgabase", [], "0u")
    for x, d, k in rungs("LCS"):
        R(f"stmt.discard_u_{x}{d}_f{k}", S(x, d, k), ["u32.atomicold"], "_ = {0};", "the effect without the value")
    alast = A_WORDS - 1
    acell = f"&wg_apool[min({{0}} + {{1}}, {alast}u)]"
    for fn, F in ATOMIC_FUNS.items():
        R(f"atomic.wg_{fn}", "u32.atomicold", ["u32.wgabase", "u32.index", "u32.index"], f"atomic{F}({acell}, {{2}})", "an atomic of the atomic pool: never a race, its old value order-dependent")
    R("atomic.wg_load", "u32.atomicold", ["u32.wgabase", "u32.index"], f"atomicLoad({acell})", "a relaxed load of the atomic pool")
    for x, d, k in rungs("LCSWP"):
        it, vt_ = ("u32.indexw", "u32.indexw") if x in "WP" else ("u32.index", "u32.index")
        R(f"atomic.wg_store_{x}{d}_f{k}", S(x, d, k), ["u32.wgabase", it, vt_], "atomicStore(&wg_apool[min({0} + {1}, " + str(alast) + "u)], {2});", "an atomic store at any element: one atomic operation, never a race")
