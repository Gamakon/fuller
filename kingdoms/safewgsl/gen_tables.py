#!/usr/bin/env python3
"""The safeWGSL kingdom's tables: the WGSL kingdom's tables (kingdoms/wgsl,
copied 2026-10-10, every dual and row kept with its ref) AUGMENTED with the
types that make a crashing or hanging kernel unrepresentable (Andrew:
"solve all these kernel crash issues by being inventive with types", the
Horton-order move of the TSR kingdom): bounded loop counts, loop rungs,
conditional contexts without barriers, index provenance, a cost per row.

Types are duals `slot.form` (the logical descriptor); each gets a unique ref
`T<n>`. Functions are `class.instance`; each gets a unique ref `F<n>` and a
signature over duals. The loader reads the TSVs; the Markdown is rendered from
the same rows so the spec and the data cannot drift.

    python3 gen_tables.py        # writes types.tsv, functions.tsv, TYPES.md, symbols.md
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- types ----
SCALARS = ["f32", "f16", "i32", "u32", "bool"]
LANES = [1, 2, 3, 4]

FORMS = {
    # form: (meaning, legal scalar slots)
    "real": ("a plain number", ["f32", "f16"]),
    "int": ("a signed integer used as a number (an RNC integer, a count that may go negative)", ["i32"]),
    "q15_16": ("fixed point, 16 fractional bits (faxl's Q15.16); saturating arithmetic only", ["i32"]),
    "q15_16w": ("the two-limb wide accumulator of q15_16 (vec2<i32> only)", []),
    "index": ("a position into an array, row, gene or node", ["u32"]),
    "count": ("a cardinality or a length", ["u32"]),
    "code4": ("a 4-level codebook entry", ["u32"]),
    "code8": ("an 8-level codebook entry", ["u32"]),
    "code16": ("a 16-level codebook entry", ["u32"]),
    "code1024": ("a 1,024-level codebook entry", ["u32"]),
    "bits": ("a bit field, mask, packed lanes or a hash at rest", ["u32"]),
    "hash32": ("a 32-bit counter-based hash state (fmix32)", ["u32"]),
    "hash64": ("a 64-bit hash state as two limbs (mix64); vec2<u32> only", []),
    "flag": ("a predicate", ["bool"]),
    "sign": ("±1 carried as one bit", ["bool"]),
    "opaque": ("a value carried through unchanged; any slot", SCALARS),
    # Statement-level forms (the multityped kingdom, PLAN_wgsl_gram_superopt §2):
    # a random typed gene whose root is `kernel.kernel` is a whole kernel.
    "kernel": ("the entry point: a grid and a body; a gene's root", []),
    "grid": ("workgroup size (x, y) and the block of cells each thread owns (bx, by)", []),
    "block": ("a sequence of statements", []),
    "stmt": ("one statement", []),
    "wgdim": ("a workgroup dimension, a literal from {4, 8, 16, 32} (64 x 64 would pass 1,024 threads)", ["u32"]),
    "blk": ("a block factor, a literal from {1, 2, 4, 8}", ["u32"]),
    "unroll": ("an unroll factor, a literal from {2, 4, 8}", ["u32"]),
    # ---- safeWGSL: the types below exist so that a bad program has no spelling ----
    # Bounded counts: the only values a loop may count to.
    "boundm": ("a loop bound that is the item count cfg.m", ["u32"]),
    "boundd": ("a loop bound that is the width cfg.d", ["u32"]),
    "boundrows": ("a loop bound that is the band's rows cfg.r_rows", ["u32"]),
    "lit64": ("a loop bound that is a literal at most 64", ["u32"]),
    "lit4096": ("a loop bound that is a literal at most 4,096", ["u32"]),
    # The loop's own counter, and the only condition a loop may test.
    "counter": ("the item loop's counter, stepped by the loop row itself", ["u32"]),
    "tcounter": ("a tile loop's counter k, stepped by the loop row itself", ["u32"]),
    "tcounter2": ("an inner tile loop's counter k2, stepped by the loop row itself", ["u32"]),
    "bound": ("a loop condition: counter < bound, produced by nothing else", ["bool"]),
    # Index provenance: where an index came from decides what it may address.
    "threadrow": ("a row index derived from the thread's own ids and block factor", ["u32"]),
    "threadcol": ("a column index derived from the thread's own ids and block factor", ["u32"]),
    "cell": ("an output cell index row*d + col owned by this thread; the only index a store to h accepts", ["u32"]),
    "phi": ("unused: a load of phi takes any index, since a read past the buffer is zero by WGSL's rule; kept so earlier refs stand", ["u32"]),
    "wgindex": ("a workgroup-array index, masked to the array by its row", ["u32"]),
    "wgvindex": ("an index into a vec4 workgroup array, masked to the array by its row", ["u32"]),
    # Statement rungs: L0 outside any loop, L1 inside the item loop, L2 inside a tile loop;
    # C marks a block under a thread-dependent condition, where no barrier may stand.
    "L0": ("a statement or block at the top level", []),
    "L1": ("a statement or block inside the item loop", []),
    "L2": ("a statement or block inside a tile loop", []),
    "C0": ("a statement or block under a condition at the top level: no barrier", []),
    "C1": ("a statement or block under a condition inside the item loop: no barrier", []),
    "C2": ("a statement or block under a condition inside a tile loop: no barrier", []),
    "L3": ("a statement or block inside an inner tile loop", []),
    "C3": ("a statement or block under a condition inside an inner tile loop: no barrier", []),
    "tcounter3": ("a third nested tile loop's counter k3, stepped by the loop row itself", ["u32"]),
    "L4": ("a statement or block inside a third nested tile loop", []),
    "C4": ("a statement or block under a condition inside a third nested tile loop: no barrier", []),
    "safe": ("the safeWGSL kernel: the root dual of this kingdom, produced only by kernel.entry_safe", []),
    "pindex": ("a private-array index, masked to the array by its row", ["u32"]),
    "boundx": ("a loop bound that is any index expression (cfg.m & ~3u, (cfg.d + 63u) / 64u, a register): rendered as written, never capped; costed at the worst shape", ["u32"]),
    # A fourth nested tile loop (r2x_union_g9 and _g10 of run A: item loop, then g, j, r, c).
    "tcounter4": ("a fourth nested tile loop's counter k4, stepped by the loop row itself", ["u32"]),
    "L5": ("a statement or block inside a fourth nested tile loop", []),
    "C5": ("a statement or block under a condition inside a fourth nested tile loop: no barrier", []),
    # THE EPILOGUE (2026-10-11, Andrew's rule: a gene that cannot be a correct kernel is a kingdom bug; 51 of
    # 133 of fit_A2's repaired offspring never wrote h): a statement or block that ENDS IN THE STORE of this
    # thread's cell, at tile depth d after the item loop. Its leading statements are conditional statements of
    # the same depth (C<d>: no barrier, no return); its last is a store, or an if, if-else, block or tile loop
    # whose body is again an epilogue. The kernel's root takes one, so every gene writes h.
    "O0": ("a statement or block of the epilogue, ending in the store of this thread's cell, outside any tile loop", []),
    "O1": ("a statement or block of the epilogue inside one tile loop", []),
    "O2": ("a statement or block of the epilogue inside two nested tile loops", []),
    "O3": ("a statement or block of the epilogue inside three nested tile loops", []),
    "O4": ("a statement or block of the epilogue inside four nested tile loops", []),
    # THE THREAD'S AXES (2026-10-11: 69 of 92 engine-made offspring of fit_A2 failed correctness, typically
    # one symbol of an index changed, lid.y for lid.x; 266 of 1,000 single-symbol mutants of the corpus swapped a
    # thread-id axis or builtin): each thread id is its own dual, and its row and column components are of
    # their own forms, so a mutation keeps the builtin and the axis a kernel wrote.
    "gid": ("the thread's global id (global_invocation_id), its own dual", []),
    "lid": ("the thread's id within its workgroup (local_invocation_id), its own dual", []),
    "wid": ("the workgroup's id (workgroup_id), its own dual", []),
    "rowid": ("the y component of a thread id: an index on the output's ROW axis", []),
    "colid": ("the x component of a thread id: an index on the output's COLUMN axis", []),
    "lidx": ("the thread's index within its workgroup (local_invocation_index), its own dual", []),
    # A thread's row and column by the kind of its block: an offset of a strided or a grouped block applied to
    # a contiguous block's row (or the reverse) leaves the thread's cells, so each kind is its own form.
    "threadrows": ("a row of a thread's STRIDED block (rows Y apart)", ["u32"]),
    "threadcols": ("a column of a thread's STRIDED block (columns X apart)", ["u32"]),
    "threadrowg": ("a row of a thread's GROUPED block", ["u32"]),
    "threadcolg": ("a column of a thread's GROUPED block", ["u32"]),
    # THE UNIFORMS, each its own dual (2026-10-11: a native offspring of fit_A2 compared its row with cfg.m
    # instead of cfg.r_rows; 78 of 1,000 single-symbol mutants of the corpus read another uniform).
    "cfgm": ("cfg.m, the item count", ["u32"]),
    "cfgd": ("cfg.d, the width (columns of h, values per item)", ["u32"]),
    "cfgrows": ("cfg.r_rows, the band's rows", ["u32"]),
    "cfgroff": ("cfg.r_off, the band's first row", ["u32"]),
}

SAFE_FORMS = ["cfgm", "cfgd", "cfgrows", "cfgroff", "lidx", "threadrows", "threadcols", "threadrowg", "threadcolg", "gid", "lid", "wid", "rowid", "colid", "O0", "O1", "O2", "O3", "O4", "tcounter4", "L5", "C5", "safe", "pindex", "boundx", "boundm", "boundd", "boundrows", "lit64", "lit4096", "counter", "tcounter", "tcounter2", "L3", "C3", "tcounter3", "L4", "C4", "wgvindex", "bound", "threadrow", "threadcol", "cell", "phi", "wgindex", "L0", "L1", "L2", "C0", "C1", "C2"]
RUNGS = ["L0", "L1", "L2", "L3", "L4", "C0", "C1", "C2", "C3", "C4"]
# The fifth rung came after every other row (run A, 2026-10-11); its rows are appended at the end so
# every earlier ref stands, and it is kept out of RUNGS for that reason.
RUNGS5 = ["L5", "C5"]
# The epilogue's rungs by tile depth; O<d>'s leading statements are C<d> statements.
EPILOGUE = ["O0", "O1", "O2", "O3", "O4"]

FALLBACK = {
    "real": "0.0", "int": "0", "q15_16": "0", "q15_16w": "vec2(0, 0)", "index": "0u", "count": "0u",
    "code4": "0u", "code8": "0u", "code16": "0u", "code1024": "0u", "bits": "0u", "hash32": "0u",
    "hash64": "vec2(0u, 0u)", "flag": "false", "sign": "true (+1)", "opaque": "the slot's zero", "store": "no-op, codon masked",
    "<form of T>": "the element's fallback",
    # The renderer gives a statement-level fallback leaf its meaning: an empty
    # kernel, the 16 x 16 x 1 x 1 grid, an empty block, an empty statement, the
    # smallest literal, the first register, the first workgroup array.
    "kernel": "an empty kernel", "grid": "16 x 16, 1 x 1", "block": "empty", "stmt": "empty",
    "wgdim": "4u", "blk": "1u", "unroll": "2u",
    "boundm": "cfg.m", "boundd": "cfg.d", "boundrows": "cfg.r_rows", "lit64": "1u", "lit4096": "1u",
    "counter": "0u", "tcounter": "0u", "tcounter2": "0u", "bound": "false", "threadrow": "0u", "threadcol": "0u", "cell": "0u", "phi": "0u", "wgindex": "0u",
    "L0": "empty", "L1": "empty", "L2": "empty", "L3": "empty", "C0": "empty", "C1": "empty", "C2": "empty", "C3": "empty",
    "safe": "an empty kernel", "pindex": "0u",
    "tcounter3": "0u", "L4": "empty", "C4": "empty", "wgvindex": "0u",
    "boundx": "1u",
    "tcounter4": "0u", "L5": "empty", "C5": "empty",
    # The epilogue's fallback is a store, never nothing: h[cell] = h[cell] + f0 at the thread's first cell
    # (gid.y * by, gid.x * bx), under the store's own bounds check.
    "O0": "the thread's own store", "O1": "the thread's own store", "O2": "the thread's own store", "O3": "the thread's own store", "O4": "the thread's own store",
    "gid": "0u", "lid": "0u", "wid": "0u", "rowid": "0u", "colid": "0u",
    "lidx": "0u", "threadrows": "0u", "threadcols": "0u", "threadrowg": "0u", "threadcolg": "0u",
    "cfgm": "0u", "cfgd": "0u", "cfgrows": "0u", "cfgroff": "0u",
}

# The fallback by machine: the lane's 32-bit pattern as an unsigned integer. Vectors and matrices
# repeat the lane, so one number suffices. Only `sign` (true = +1) is non-zero. `store` substitutes
# nothing (fallback_noop = 1; every substitution counts the codon as masked, so that word is not
# reused here) and `array<T>` defers to its element (fallback_elem = 1); their bits are placeholders
# the loader must not read before testing those two columns.
FALLBACK_BITS = {
    "real": 0, "int": 0, "q15_16": 0, "q15_16w": 0, "index": 0, "count": 0,
    "code4": 0, "code8": 0, "code16": 0, "code1024": 0, "bits": 0, "hash32": 0,
    "hash64": 0, "flag": 0, "sign": 1, "opaque": 0, "store": 0, "<form of T>": 0,
    "kernel": 0, "grid": 0, "block": 0, "stmt": 0, "wgdim": 0, "blk": 0, "unroll": 0,
    "boundm": 0, "boundd": 0, "boundrows": 0, "lit64": 0, "lit4096": 0,
    "counter": 0, "tcounter": 0, "tcounter2": 0, "bound": 0, "threadrow": 0, "threadcol": 0, "cell": 0, "phi": 0, "wgindex": 0,
    "L0": 0, "L1": 0, "L2": 0, "L3": 0, "C0": 0, "C1": 0, "C2": 0, "C3": 0, "safe": 0, "pindex": 0,
    "tcounter3": 0, "L4": 0, "C4": 0, "wgvindex": 0,
    "boundx": 0,
    "tcounter4": 0, "L5": 0, "C5": 0,
    "O0": 0, "O1": 0, "O2": 0, "O3": 0, "O4": 0,
    "gid": 0, "lid": 0, "wid": 0, "rowid": 0, "colid": 0,
    "lidx": 0, "threadrows": 0, "threadcols": 0, "threadrowg": 0, "threadcolg": 0,
    "cfgm": 0, "cfgd": 0, "cfgrows": 0, "cfgroff": 0,
}
assert set(FALLBACK_BITS) == set(FALLBACK), "every form with a fallback needs its bits, and no more"


def slot_name(scalar: str, lanes: int) -> str:
    return scalar if lanes == 1 else f"vec{lanes}<{scalar}>"


def build_types():
    rows = []  # (ref, dual, slot, form, lanes, fallback, fallback_bits, fallback_noop, fallback_elem, meaning)

    def add(slot, form, lanes, meaning):
        noop = int(form == "store")
        elem = int(slot == "array<T>")
        rows.append((f"T{len(rows) + 1:03d}", f"{slot}.{form}", slot, form, lanes, FALLBACK[form], FALLBACK_BITS[form], noop, elem, meaning))

    for form, (meaning, scalars) in FORMS.items():
        if form in ("wgdim", "blk", "unroll") or form in SAFE_FORMS:
            continue  # added after the specials, below, so T070/T071 keep their refs
        for s in scalars:
            for l in LANES:
                # vector forms that make no sense: codes, hashes and signs are scalar-only;
                # a count has vector slots because the builtin num_workgroups is a vec3<u32>.count
                if l > 1 and form in ("code4", "code8", "code16", "code1024", "hash32", "sign", "wgdim", "blk", "unroll"):
                    continue
                add(slot_name(s, l), form, l, meaning)
    add("vec2<i32>", "q15_16w", 2, FORMS["q15_16w"][0])
    add("vec2<u32>", "hash64", 2, FORMS["hash64"][0])
    for c in (2, 3, 4):
        for r in (2, 3, 4):
            add(f"mat{c}x{r}<f32>", "real", c * r, "a matrix of plain numbers")
    add("array<T>", "<form of T>", 0, "a storage or uniform buffer of T; indexed by u32.index; one row per buffer at read-in")
    add("store", "store", 0, "the statement a head gene's root produces")
    # Statement-level duals, appended after the value duals so every earlier
    # ref and index is unchanged.
    add("kernel", "kernel", 0, FORMS["kernel"][0])
    add("grid", "grid", 0, FORMS["grid"][0])
    add("block", "block", 0, FORMS["block"][0])
    add("stmt", "stmt", 0, FORMS["stmt"][0])
    add("reg<f32>", "real", 1, "a named f32 register: f0, f1, f2, f3")
    add("reg<vec4<f32>>", "real", 4, "a named vec4<f32> register: v0, v1")
    add("reg<u32>", "index", 1, "a named u32 register: i0, i1, i2, i3")
    add("reg<bool>", "flag", 1, "a named bool register: b0")
    add("wgmem<f32>", "real", 0, "a workgroup-memory array of 2,048 f32: a, b")
    for form in ("wgdim", "blk", "unroll"):
        add("u32", form, 1, FORMS[form][0])
    # ---- safeWGSL duals, after every WGSL dual (T084 on) ----
    for form in ("boundm", "boundd", "boundrows", "lit64", "lit4096", "counter", "threadrow", "threadcol", "cell", "phi", "wgindex"):
        add("u32", form, 1, FORMS[form][0])
    add("bool", "bound", 1, FORMS["bound"][0])
    for r in ("L0", "L1", "L2", "C0", "C1", "C2"):
        add("stmt", r, 0, "a statement " + FORMS[r][0][len("a statement or block "):])
        add("block", r, 0, "a block " + FORMS[r][0][len("a statement or block "):])
    add("kernel", "safe", 0, FORMS["safe"][0])
    add("u32", "tcounter", 1, FORMS["tcounter"][0])
    add("u32", "tcounter2", 1, FORMS["tcounter2"][0])
    add("u32", "pindex", 1, FORMS["pindex"][0])
    # Private arrays: a thread's own, fixed size, every index masked by its row.
    add("parr<f32>", "real", 0, "a private array of 64 f32: pa, pb")
    add("parr<vec4<f32>>", "real", 0, "a private array of 16 vec4<f32>: pv")
    add("parr<u32>", "index", 0, "a private array of 16 u32: pi")
    for r in ("L3", "C3"):
        add("stmt", r, 0, "a statement " + FORMS[r][0][len("a statement or block "):])
        add("block", r, 0, "a block " + FORMS[r][0][len("a statement or block "):])
    # A third nested tile loop (register-blocked kernels loop over rows, columns and the tile).
    add("u32", "tcounter3", 1, FORMS["tcounter3"][0])
    for r in ("L4", "C4"):
        add("stmt", r, 0, "a statement " + FORMS[r][0][len("a statement or block "):])
        add("block", r, 0, "a block " + FORMS[r][0][len("a statement or block "):])
    add("parr<bool>", "flag", 0, "a private array of 16 bool: pm, pn")
    add("wgmem<vec4<f32>>", "real", 0, "a workgroup-memory array of 512 vec4<f32>: va, vb (256 until 2026-10-11; two f32 arrays of 2,048 and two vec4 arrays of 512 are Apple's 32 KB)")
    add("u32", "wgvindex", 1, FORMS["wgvindex"][0])
    # ---- the complete kingdom (step 6, 2026-10-11) ----
    add("u32", "boundx", 1, FORMS["boundx"][0])
    # ---- run A's declines (2026-10-11): a fourth nested tile loop ----
    add("u32", "tcounter4", 1, FORMS["tcounter4"][0])
    for r in RUNGS5:
        add("stmt", r, 0, "a statement " + FORMS[r][0][len("a statement or block "):])
        add("block", r, 0, "a block " + FORMS[r][0][len("a statement or block "):])
    # ---- the epilogue (2026-10-11): the kernel ends in the store of its thread's cell ----
    for r in EPILOGUE:
        add("stmt", r, 0, "a statement " + FORMS[r][0][len("a statement or block "):])
        add("block", r, 0, "a block " + FORMS[r][0][len("a statement or block "):])
    # ---- the thread's axes (2026-10-11) ----
    for b in ("gid", "lid", "wid"):
        add("vec3<u32>", b, 3, FORMS[b][0])
    add("u32", "rowid", 1, FORMS["rowid"][0])
    add("u32", "colid", 1, FORMS["colid"][0])
    for form in ("lidx", "threadrows", "threadcols", "threadrowg", "threadcolg"):
        add("u32", form, 1, FORMS[form][0])
    for form in ("cfgm", "cfgd", "cfgrows", "cfgroff"):
        add("u32", form, 1, FORMS[form][0])
    return rows


# ------------------------------------------------------------ functions ----
# Each entry: (class, instance, naga semantic, [in duals], out dual, instantiation note)
# "<S.real>" stands for every real dual over scalar/vector lanes (f32, f16, vecL<..>).
FUNCTIONS = []


def f(cls, inst, naga, ins, out, note=""):
    FUNCTIONS.append((cls, inst, naga, ins, out, note))


# literal / load / builtin: terminals
f("literal", "real", "Literal", [], "<S.real>", "an RNC-style constant in the kingdom's range")
f("literal", "int", "Literal", [], "i32.int", "")
f("literal", "index", "Literal", [], "u32.index", "")
f("literal", "count", "Literal", [], "u32.count", "")
f("literal", "bits", "Literal", [], "u32.bits", "")
f("literal", "flag", "Literal", [], "bool.flag", "")
f("load", "buffer", "Load(Access)", ["u32.index"], "<T.form of the buffer>", "one instance per buffer the kernel binds; the buffer is the instance's identity")
f("load", "uniform", "Load(AccessIndex)", [], "<T.form of the field>", "one instance per uniform field")
f("load", "local", "Load", [], "<T.form of the local>", "one instance per local at that scaffold point")
f("arg", "<name>", "FunctionArgument", [], "<T.form of the argument>", "one instance per argument")
# The three thread ids each their own dual (2026-10-11): one cannot be mutated into another.
for b, out in [("global_invocation_id", "vec3<u32>.gid"), ("local_invocation_id", "vec3<u32>.lid"), ("workgroup_id", "vec3<u32>.wid"), ("local_invocation_index", "u32.lidx"), ("num_workgroups", "vec3<u32>.count")]:
    f("builtin", b, "FunctionArgument(builtin)", [], out, "")

# arith: real
for op, naga in [("add", "Binary::Add"), ("sub", "Binary::Subtract"), ("mul", "Binary::Multiply"), ("div", "Binary::Divide")]:
    f("arith", op, naga, ["<S.real>", "<S.real>"], "<S.real>", "per real dual, lanes equal")
f("arith", "mul_scalar_vec", "Binary::Multiply", ["f32.real", "<vecL<f32>.real>"], "<vecL<f32>.real>", "")
f("arith", "mul_mat_vec", "Binary::Multiply", ["<matCxR<f32>.real>", "<vecC<f32>.real>"], "<vecR<f32>.real>", "")
f("arith", "mul_mat_mat", "Binary::Multiply", ["<matKxR<f32>.real>", "<matCxK<f32>.real>"], "<matCxR<f32>.real>", "WGSL matCxR is C columns of R rows: the left's columns equal the right's rows")
f("arith", "neg", "Unary::Negate", ["<S.real>"], "<S.real>", "")
for op in ["abs", "sign", "floor", "ceil", "round", "fract", "trunc", "saturate", "sqrt", "inverse_sqrt", "exp", "exp2", "log", "log2"]:
    f("arith", op, f"Math::{op.title().replace('_', '')}", ["<S.real>"], "<S.real>", "")
for op in ["min", "max", "step", "pow", "atan2", "ldexp"]:
    f("arith", op, f"Math::{op.title()}", ["<S.real>", "<S.real>"], "<S.real>", "")
# naga's own names: SmoothStep (title-casing gave Smoothstep, which no naga function is, so a kernel's
# smoothstep, the kingdom's own rendering included, did not read back; 2026-10-11).
for op, naga in [("clamp", "Clamp"), ("mix", "Mix"), ("fma", "Fma"), ("smoothstep", "SmoothStep")]:
    f("arith", op, f"Math::{naga}", ["<S.real>", "<S.real>", "<S.real>"], "<S.real>", "arity 3")
f("arith", "mix_scalar_weight", "Math::Mix", ["<vecL<f32>.real>", "<vecL<f32>.real>", "f32.real"], "<vecL<f32>.real>", "arity 3")
# trig
for op in ["sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "asinh", "acosh", "atanh", "radians", "degrees"]:
    f("trig", op, f"Math::{op.title()}", ["<S.real>"], "<S.real>", "")
# geom
f("geom", "dot", "Math::Dot", ["<vecL<f32>.real>", "<vecL<f32>.real>"], "f32.real", "")
f("geom", "cross", "Math::Cross", ["vec3<f32>.real", "vec3<f32>.real"], "vec3<f32>.real", "")
f("geom", "length", "Math::Length", ["<vecL<f32>.real>"], "f32.real", "")
f("geom", "distance", "Math::Distance", ["<vecL<f32>.real>", "<vecL<f32>.real>"], "f32.real", "")
f("geom", "normalize", "Math::Normalize", ["<vecL<f32>.real>"], "<vecL<f32>.real>", "")
f("geom", "outer", "Math::Outer", ["<vecR<f32>.real>", "<vecC<f32>.real>"], "<matCxR<f32>.real>", "")
f("geom", "transpose", "Math::Transpose", ["<matCxR<f32>.real>"], "<matRxC<f32>.real>", "")
f("geom", "determinant", "Math::Determinant", ["<matNxN<f32>.real>"], "f32.real", "")
f("geom", "inverse", "Math::Inverse", ["<matNxN<f32>.real>"], "<matNxN<f32>.real>", "")
f("geom", "reflect", "Math::Reflect", ["<vecL<f32>.real>", "<vecL<f32>.real>"], "<vecL<f32>.real>", "")
f("geom", "refract", "Math::Refract", ["<vecL<f32>.real>", "<vecL<f32>.real>", "f32.real"], "<vecL<f32>.real>", "arity 3")
f("geom", "face_forward", "Math::FaceForward", ["<vecL<f32>.real>", "<vecL<f32>.real>", "<vecL<f32>.real>"], "<vecL<f32>.real>", "arity 3")
# intarith: index, count, int
for op, naga in [("add", "Binary::Add"), ("sub", "Binary::Subtract"), ("mul", "Binary::Multiply"), ("div", "Binary::Divide"), ("mod", "Binary::Modulo"), ("min", "Math::Min"), ("max", "Math::Max")]:
    f("index", op, naga, ["<u32.index>", "<u32.index>"], "<u32.index>", "lanes equal")
f("index", "clamp", "Math::Clamp", ["<u32.index>", "<u32.index>", "<u32.index>"], "<u32.index>", "arity 3")
f("index", "of_int", "composite", ["i32.int"], "u32.index", "floor at zero")
f("index", "of_count", "composite", ["u32.count"], "u32.index", "")
f("index", "clamp_to", "composite", ["u32.index", "u32.count"], "u32.index", "i < n")
f("index", "band_of", "composite", ["u32.index", "u32.count"], "u32.index", "i / band")
for op, naga in [("add", "Binary::Add"), ("sub_floor", "composite"), ("min", "Math::Min"), ("max", "Math::Max")]:
    f("count", op, naga, ["u32.count", "u32.count"], "u32.count", "sub floors at zero")
for op, naga in [("add", "Binary::Add"), ("sub", "Binary::Subtract"), ("mul", "Binary::Multiply"), ("div", "Binary::Divide"), ("mod", "Binary::Modulo"), ("neg", "Unary::Negate"), ("abs", "Math::Abs"), ("min", "Math::Min"), ("max", "Math::Max")]:
    ins = ["<i32.int>"] if op in ("neg", "abs") else ["<i32.int>", "<i32.int>"]
    f("int", op, naga, ins, "<i32.int>", "lanes equal")
# fixed
f("fixed", "add_sat", "composite", ["i32.q15_16", "i32.q15_16"], "i32.q15_16", "saturating")
f("fixed", "sub_sat", "composite", ["i32.q15_16", "i32.q15_16"], "i32.q15_16", "saturating")
f("fixed", "mul", "composite", ["i32.q15_16", "i32.q15_16"], "i32.q15_16", "(x*y) >> 16 with rounding; the only multiply on q15_16")
f("fixed", "shl", "composite", ["i32.q15_16", "u32.count"], "i32.q15_16", "")
f("fixed", "shr", "composite", ["i32.q15_16", "u32.count"], "i32.q15_16", "")
f("fixed", "wide_mul", "composite", ["i32.q15_16", "i32.q15_16"], "vec2<i32>.q15_16w", "two limbs, no rescale")
f("fixed", "wide_add_sat", "composite", ["vec2<i32>.q15_16w", "vec2<i32>.q15_16w"], "vec2<i32>.q15_16w", "")
f("fixed", "narrow", "composite", ["vec2<i32>.q15_16w"], "i32.q15_16", "")
# convert: the only slot and form crossings
f("convert", "f32_to_f16", "As", ["f32.real"], "f16.real", "")
f("convert", "f16_to_f32", "As", ["f16.real"], "f32.real", "")
f("convert", "f32_to_q15_16", "composite", ["f32.real"], "i32.q15_16", "round to nearest, saturate")
f("convert", "q15_16_to_f32", "composite", ["i32.q15_16"], "f32.real", "")
f("convert", "int_to_f32", "As", ["i32.int"], "f32.real", "")
f("convert", "f32_to_int", "As", ["f32.real"], "i32.int", "truncating")
f("convert", "count_to_f32", "As", ["u32.count"], "f32.real", "")
f("convert", "index_to_f32", "As", ["u32.index"], "f32.real", "")
f("convert", "f32_to_count", "As", ["f32.real"], "u32.count", "truncating, floor at 0")
f("convert", "bitcast_f32_bits", "As(bitcast)", ["f32.real"], "u32.bits", "")
f("convert", "bitcast_bits_f32", "As(bitcast)", ["u32.bits"], "f32.real", "")
f("convert", "sign_to_real", "composite", ["bool.sign"], "f32.real", "±1.0")
f("convert", "hash32_to_bits", "composite", ["u32.hash32"], "u32.bits", "")
f("convert", "hash32_to_unit", "composite", ["u32.hash32"], "f32.real", "scale to [0, 1)")
f("convert", "hash32_to_index", "composite", ["u32.hash32", "u32.count"], "u32.index", "masked rejection, never modulo")
# quantise
for k in (4, 8, 16, 1024):
    f("quantise", f"real_to_code{k}", "composite", ["f32.real"], f"u32.code{k}", f"nearest of the {k}-level codebook")
    f("quantise", f"code{k}_to_real", "composite", [f"u32.code{k}"], "f32.real", "the level's value")
f("quantise", "pack4_code8", "composite", ["u32.code8", "u32.code8", "u32.code8"], "u32.bits", "arity 3: three lanes; pack4_code8_hi adds the fourth")
f("quantise", "pack4_code8_hi", "composite", ["u32.bits", "u32.code8"], "u32.bits", "")
f("quantise", "unpack_code8", "composite", ["u32.bits", "u32.index"], "u32.code8", "lane i")
# bits
for op, naga in [("and", "Binary::And"), ("or", "Binary::InclusiveOr"), ("xor", "Binary::ExclusiveOr")]:
    f("bits", op, naga, ["<u32.bits>", "<u32.bits>"], "<u32.bits>", "lanes equal")
f("bits", "not", "Unary::BitwiseNot", ["<u32.bits>"], "<u32.bits>", "")
f("bits", "shl", "Binary::ShiftLeft", ["<u32.bits>", "u32.count"], "<u32.bits>", "")
f("bits", "shr", "Binary::ShiftRight", ["<u32.bits>", "u32.count"], "<u32.bits>", "")
for op, naga in [("count_ones", "Math::CountOneBits"), ("leading_zeros", "Math::CountLeadingZeros"), ("trailing_zeros", "Math::CountTrailingZeros"), ("find_lsb", "Math::FindLsb"), ("find_msb", "Math::FindMsb")]:
    f("bits", op, naga, ["u32.bits"], "u32.count", "")
f("bits", "reverse", "Math::ReverseBits", ["u32.bits"], "u32.bits", "")
f("bits", "extract", "Math::ExtractBits", ["u32.bits", "u32.index", "u32.count"], "u32.bits", "arity 3")
f("bits", "insert_lo", "composite", ["u32.bits", "u32.bits", "u32.index"], "u32.bits", "arity 3; with insert_hi composes naga InsertBits (arity 4)")
f("bits", "insert_hi", "composite", ["u32.bits", "u32.count"], "u32.bits", "")
f("bits", "pack_sign", "composite", ["bool.sign"], "u32.bits", "")
f("bits", "unpack_sign", "composite", ["u32.bits", "u32.index"], "bool.sign", "")
for op in ["pack4x8snorm", "pack4x8unorm", "pack2x16snorm", "pack2x16unorm", "pack2x16float"]:
    lanes = "vec4<f32>.real" if "4x8" in op else "vec2<f32>.real"
    f("bits", op, f"Math::{op[0].upper() + op[1:]}", [lanes], "u32.bits", "")
    f("bits", "un" + op, f"Math::Un{op[0].upper() + op[1:]}", ["u32.bits"], lanes, "")
# hash
f("hash", "mix64", "composite", ["vec2<u32>.hash64"], "vec2<u32>.hash64", "SplitMix64 finaliser, two limbs")
f("hash", "fmix32", "composite", ["u32.hash32"], "u32.hash32", "MurmurHash3 finaliser")
f("hash", "combine", "composite", ["u32.hash32", "u32.bits"], "u32.hash32", "xor-in a key or an index")
f("hash", "split_lo", "composite", ["vec2<u32>.hash64"], "u32.hash32", "")
f("hash", "split_hi", "composite", ["vec2<u32>.hash64"], "u32.hash32", "")
# compare / logic / select
for op, naga in [("eq", "Binary::Equal"), ("ne", "Binary::NotEqual"), ("lt", "Binary::Less"), ("le", "Binary::LessEqual"), ("gt", "Binary::Greater"), ("ge", "Binary::GreaterEqual")]:
    f("compare", op, naga, ["<S.c>", "<S.c>"], "<bool.flag per lane>", "any form c that permits comparison (not bits, hash, opaque; codes eq/ne only)")
f("logic", "and", "Binary::LogicalAnd", ["bool.flag", "bool.flag"], "bool.flag", "")
f("logic", "or", "Binary::LogicalOr", ["bool.flag", "bool.flag"], "bool.flag", "")
f("logic", "not", "Unary::LogicalNot", ["bool.flag"], "bool.flag", "")
f("logic", "all", "Relational::All", ["<vecL<bool>.flag>"], "bool.flag", "")
f("logic", "any", "Relational::Any", ["<vecL<bool>.flag>"], "bool.flag", "")
f("logic", "is_nan", "Relational::IsNan", ["<S.real>"], "<bool.flag per lane>", "")
f("logic", "is_inf", "Relational::IsInf", ["<S.real>"], "<bool.flag per lane>", "")
f("select", "scalar_cond", "Select", ["<S.c>", "<S.c>", "bool.flag"], "<S.c>", "arity 3; both branches one dual")
f("select", "lane_cond", "Select", ["<vecL.c>", "<vecL.c>", "<vecL<bool>.flag>"], "<vecL.c>", "arity 3")
f("select", "negate_if", "composite", ["<S.real>", "bool.sign"], "<S.real>", "")
# shape
for i in range(4):
    f("shape", f"access_{i}", "AccessIndex", ["<vecL.c>"], "<S.c>", f"lane {i}; row exists only for L > {i}")
f("shape", "access_dyn", "Access", ["<array<T>.c>", "u32.index"], "<T.c>", "")
f("shape", "swizzle", "Swizzle", ["<vecL.c>"], "<vecK.c>", "one instance per pattern met at read-in (xy, xyz, zyx, …)")
f("shape", "compose2", "Compose", ["<S.c>", "<S.c>"], "<vec2.c>", "")
f("shape", "compose3", "Compose", ["<vec2.c>", "<S.c>"], "<vec3.c>", "composed from compose2")
f("shape", "compose4", "Compose", ["<vec3.c>", "<S.c>"], "<vec4.c>", "")
f("shape", "compose_mat", "Compose", ["<vecR<f32>.real>"] * 2, "<mat2xR<f32>.real>", "wider matrices by further composes")
for l in (2, 3, 4):
    f("shape", f"splat{l}", "Splat", ["<S.c>"], f"<vec{l}.c>", "")
# store
f("store", "buffer", "Store(Access)", ["u32.index", "<T.form of the buffer>"], "store", "one instance per buffer; a head gene's root")
f("store", "local", "Store", ["<T.form of the local>"], "store", "")
f("store", "branch", "If/Loop condition", ["bool.flag"], "store", "a branch-condition root in the scaffold")

# ---- statements: the multityped kingdom (PLAN_wgsl_gram_superopt §2). ----
# Every row is rendered to text by src/wgsl/render.rs; naga validates the text.
f("literal", "wgdim", "Literal", [], "u32.wgdim", "a workgroup dimension from {4, 8, 16, 32}; the renderer snaps the constant to the set")
f("literal", "blk", "Literal", [], "u32.blk", "a block factor from {1, 2, 4, 8} (8 since 2026-10-11: a model child's 4 x 8 block)")
f("literal", "unroll", "Literal", [], "u32.unroll", "an unroll factor from {2, 4, 8}")
f("kernel", "entry", "composite", ["grid.grid", "block.block"], "kernel.kernel", "the module: bindings, registers, workgroup arrays, @workgroup_size(x, y, 1) fn main")
f("grid", "of", "composite", ["u32.wgdim", "u32.wgdim", "u32.blk", "u32.blk"], "grid.grid", "arity 4: workgroup x, y and the cells per thread bx, by; the driver dispatches ceil(d / (x bx)) x ceil(rows / (y by)); x y > 1,024 threads is refused")
f("block", "one", "composite", ["stmt.stmt"], "block.block", "")
f("block", "seq2", "composite", ["stmt.stmt", "stmt.stmt"], "block.block", "the statements in order")
f("block", "seq3", "composite", ["stmt.stmt"] * 3, "block.block", "arity 3")
f("block", "seq4", "composite", ["stmt.stmt"] * 4, "block.block", "arity 4")
f("stmt", "block", "composite", ["block.block"], "stmt.stmt", "{ ... }")
f("stmt", "if", "composite", ["bool.flag", "block.block"], "stmt.stmt", "if (c) { ... }")
f("stmt", "if_else", "composite", ["bool.flag", "block.block", "block.block"], "stmt.stmt", "arity 3")
f("stmt", "return", "composite", [], "stmt.stmt", "return; the thread guard's exit")
f("stmt", "barrier", "composite", [], "stmt.stmt", "workgroupBarrier()")
f("loop", "items", "composite", ["block.block"], "stmt.stmt", "for (t = 0u; t < cfg.m; t++) { ... }: the one long loop; nested in another loop.items the kernel is refused")
f("loop", "items_unrolled", "composite", ["u32.unroll", "block.block"], "stmt.stmt", "the body k times with t + j, then the remainder loop")
f("loop", "tile", "composite", ["u32.wgdim", "block.block"], "stmt.stmt", "for (k = 0u; k < n; k++) { ... }: a short loop")
f("builtin", "item", "composite", [], "u32.index", "t, the item loop's index; 0u outside it")
f("builtin", "tile", "composite", [], "u32.index", "k, the tile loop's index; 0u outside it")
for name, dual in [("f0", "reg<f32>.real"), ("f1", "reg<f32>.real"), ("f2", "reg<f32>.real"), ("f3", "reg<f32>.real"), ("v0", "reg<vec4<f32>>.real"), ("v1", "reg<vec4<f32>>.real"), ("i0", "reg<u32>.index"), ("i1", "reg<u32>.index"), ("i2", "reg<u32>.index"), ("i3", "reg<u32>.index"), ("b0", "reg<bool>.flag")]:
    f("reg", name, "composite", [], dual, "a register, declared at the top of the kernel and initialised to its form's fallback")
for suffix, reg, val in [("f", "reg<f32>.real", "f32.real"), ("v", "reg<vec4<f32>>.real", "vec4<f32>.real"), ("i", "reg<u32>.index", "u32.index"), ("b", "reg<bool>.flag", "bool.flag")]:
    f("reg", f"get_{suffix}", "composite", [reg], val, "the register's value")
    f("reg", f"set_{suffix}", "composite", [reg, val], "stmt.stmt", "name = value;")
f("wg", "a", "composite", [], "wgmem<f32>.real", "workgroup array a")
f("wg", "b", "composite", [], "wgmem<f32>.real", "workgroup array b")
f("wg", "get", "composite", ["wgmem<f32>.real", "u32.index"], "f32.real", "a[i & 2047u]")
f("wg", "set", "composite", ["wgmem<f32>.real", "u32.index", "f32.real"], "stmt.stmt", "arity 3: a[i & 2047u] = v;")

# More registers for drafted kernels (sixteen f32, eight vec4, eight u32, four bool).
for i in range(4, 16):
    f("reg", f"f{i}", "composite", [], "reg<f32>.real", "a register")
for i in range(2, 8):
    f("reg", f"v{i}", "composite", [], "reg<vec4<f32>>.real", "a register")
for i in range(4, 16):
    f("reg", f"i{i}", "composite", [], "reg<u32>.index", "a register")
for i in range(1, 4):
    f("reg", f"b{i}", "composite", [], "reg<bool>.flag", "a register")

# Private arrays (a thread's own) with masked indices.
f("parr", "a", "composite", [], "parr<f32>.real", "private array pa of 64 f32")
f("parr", "b", "composite", [], "parr<f32>.real", "private array pb of 64 f32")
f("parr", "c", "composite", [], "parr<f32>.real", "private array pc of 64 f32")
f("parr", "d", "composite", [], "parr<f32>.real", "private array pd of 64 f32")
f("parr", "v", "composite", [], "parr<vec4<f32>>.real", "private array pv of 16 vec4<f32>")
f("parr", "i", "composite", [], "parr<u32>.index", "private array pi of 16 u32")
f("safe", "pidx64", "composite", ["u32.index"], "u32.pindex", "i & 63u")
f("safe", "pidx16", "composite", ["u32.index"], "u32.pindex", "i & 15u")
f("parr", "get_f", "composite", ["parr<f32>.real", "u32.pindex"], "f32.real", "pa[i]")
f("parr", "get_v", "composite", ["parr<vec4<f32>>.real", "u32.pindex"], "vec4<f32>.real", "pv[i]")
f("parr", "get_i", "composite", ["parr<u32>.index", "u32.pindex"], "u32.index", "pi[i]")
f("parr", "m", "composite", [], "parr<bool>.flag", "private array pm of 16 bool")
f("parr", "n", "composite", [], "parr<bool>.flag", "private array pn of 16 bool")
f("parr", "e", "composite", [], "parr<f32>.real", "private array pe of 64 f32")
f("parr", "f", "composite", [], "parr<f32>.real", "private array pf of 64 f32")
f("parr", "j", "composite", [], "parr<u32>.index", "private array pj of 16 u32")
f("parr", "k", "composite", [], "parr<u32>.index", "private array pk of 16 u32")
f("wg", "va", "composite", [], "wgmem<vec4<f32>>.real", "workgroup array va of 512 vec4<f32>")
f("wg", "vb", "composite", [], "wgmem<vec4<f32>>.real", "workgroup array vb of 512 vec4<f32>")
f("safe", "wgv_of", "composite", ["u32.index"], "u32.wgvindex", "i & 511u, in range by construction")
f("wg", "get_v_safe", "composite", ["wgmem<vec4<f32>>.real", "u32.wgvindex"], "vec4<f32>.real", "va[i], in range by its type")
# Index arithmetic the drafted kernels use on addresses: shifts (the count masked to 0..31) and masks.
f("index", "shl", "composite", ["u32.index", "u32.index"], "u32.index", "i << (s & 31u)")
f("index", "shr", "composite", ["u32.index", "u32.index"], "u32.index", "i >> (s & 31u)")
f("index", "and", "composite", ["u32.index", "u32.index"], "u32.index", "i & j")
f("index", "or", "composite", ["u32.index", "u32.index"], "u32.index", "i | j")
f("parr", "get_b", "composite", ["parr<bool>.flag", "u32.pindex"], "bool.flag", "pm[i]")

# ---- safeWGSL rows: the only spellings of loops, conditions, stores and loads ----
# Bounds: the bindings' counts and small literals are the only loop bounds.
f("bound", "m", "composite", [], "u32.boundm", "cfg.m")
f("bound", "d", "composite", [], "u32.boundd", "cfg.d")
f("bound", "rows", "composite", [], "u32.boundrows", "cfg.r_rows")
f("literal", "lit64", "Literal", [], "u32.lit64", "a literal snapped into 1..=64")
f("literal", "lit4096", "Literal", [], "u32.lit4096", "a literal snapped into 1..=4096")
# The loop condition: counter < bound, and nothing else produces a flag.bound.
for b in ("boundm", "boundd", "boundrows", "lit4096"):
    f("safe", "lt_" + b, "Binary::Less", ["u32.counter", "u32." + b], "bool.bound", "the only producer of a loop condition")
f("safe", "lt_lit64", "Binary::Less", ["u32.tcounter", "u32.lit64"], "bool.bound", "a tile loop's condition")
f("safe", "lt_lit64_2", "Binary::Less", ["u32.tcounter2", "u32.lit64"], "bool.bound", "an inner tile loop's condition")
f("safe", "lt_lit64_3", "Binary::Less", ["u32.tcounter3", "u32.lit64"], "bool.bound", "a third nested tile loop's condition")
f("safe", "tile_counter3", "composite", [], "u32.tcounter3", "the third nested tile loop's counter k3 (0u outside it)")
f("safe", "tile_index3", "composite", ["u32.tcounter3"], "u32.index", "the third tile counter as a plain index")
f("safe", "row_plus_tile3", "composite", ["u32.threadrow", "u32.tcounter3"], "u32.threadrow", "row + min(k3, block rows - 1)")
f("safe", "col_plus_tile3", "composite", ["u32.threadcol", "u32.tcounter3"], "u32.threadcol", "col + min(k3, block columns - 1)")
f("safe", "counter", "composite", [], "u32.counter", "the item loop's counter t (0u outside it)")
f("safe", "tile_counter", "composite", [], "u32.tcounter", "the tile loop's counter k (0u outside it)")
f("safe", "tile_counter2", "composite", [], "u32.tcounter2", "the inner tile loop's counter k2 (0u outside it)")
f("safe", "counter_index", "composite", ["u32.counter"], "u32.index", "the item counter as a plain index")
f("safe", "tile_index", "composite", ["u32.tcounter"], "u32.index", "the tile counter as a plain index")
f("safe", "tile_index2", "composite", ["u32.tcounter2"], "u32.index", "the inner tile counter as a plain index")
f("safe", "row_plus_tile", "composite", ["u32.threadrow", "u32.tcounter"], "u32.threadrow", "row + min(k, block rows - 1): still a row this thread owns")
f("safe", "col_plus_tile", "composite", ["u32.threadcol", "u32.tcounter"], "u32.threadcol", "col + min(k, block columns - 1)")
f("safe", "row_plus_tile2", "composite", ["u32.threadrow", "u32.tcounter2"], "u32.threadrow", "row + min(k2, block rows - 1)")
f("safe", "col_plus_tile2", "composite", ["u32.threadcol", "u32.tcounter2"], "u32.threadcol", "col + min(k2, block columns - 1)")
# Index provenance.
f("safe", "thread_row", "composite", ["vec3<u32>.gid", "u32.blk"], "u32.threadrow", "gid.y * blk (+ i by the block row): a row this thread owns")
f("safe", "thread_col", "composite", ["vec3<u32>.gid", "u32.blk"], "u32.threadcol", "gid.x * blk (+ j by the block column): a column this thread owns")
f("safe", "thread_row_s", "composite", ["vec3<u32>.wid", "vec3<u32>.lid", "u32.blk"], "u32.threadrows", "arity 3: wid.y * (Y * blk) + lid.y: the first row of a thread's STRIDED block (its rows Y apart)")
f("safe", "thread_col_s", "composite", ["vec3<u32>.wid", "vec3<u32>.lid", "u32.blk"], "u32.threadcols", "arity 3: wid.x * (X * blk) + lid.x: the first column of a thread's strided block")
f("safe", "row_plus_s", "composite", ["u32.threadrows", "u32.lit64"], "u32.threadrows", "row + min(i, blk - 1) * Y: the i-th row of a strided block")
f("safe", "col_plus_s", "composite", ["u32.threadcols", "u32.lit64"], "u32.threadcols", "col + min(j, blk - 1) * X")
f("safe", "row_plus_tile_s", "composite", ["u32.threadrows", "u32.tcounter"], "u32.threadrows", "row + min(k, blk - 1) * Y: the k-th row of a strided block")
f("safe", "col_plus_tile_s", "composite", ["u32.threadcols", "u32.tcounter"], "u32.threadcols", "col + min(k, blk - 1) * X")
f("safe", "row_plus_tile2_s", "composite", ["u32.threadrows", "u32.tcounter2"], "u32.threadrows", "row + min(k2, blk - 1) * Y")
f("safe", "col_plus_tile2_s", "composite", ["u32.threadcols", "u32.tcounter2"], "u32.threadcols", "col + min(k2, blk - 1) * X")
f("safe", "row_plus_tile3_s", "composite", ["u32.threadrows", "u32.tcounter3"], "u32.threadrows", "row + min(k3, blk - 1) * Y")
f("safe", "col_plus_tile3_s", "composite", ["u32.threadcols", "u32.tcounter3"], "u32.threadcols", "col + min(k3, blk - 1) * X")
f("safe", "cell_of", "composite", ["u32.threadrow", "u32.threadcol"], "u32.cell", "row * cfg.d + col, the cell this thread owns; the only index store.h accepts")
f("safe", "phi_of", "composite", ["u32.counter", "u32.index"], "u32.phi", "unused (a load of phi takes any index); kept so earlier refs stand")
f("safe", "wg_of", "composite", ["u32.index"], "u32.wgindex", "i & 2047u, in range by construction")
f("safe", "row_plus", "composite", ["u32.threadrow", "u32.lit64"], "u32.threadrow", "row + min(i, block rows - 1)")
f("safe", "col_plus", "composite", ["u32.threadcol", "u32.lit64"], "u32.threadcol", "col + min(j, block columns - 1)")
f("safe", "row_as_index", "composite", ["u32.threadrow"], "u32.index", "")
f("safe", "col_as_index", "composite", ["u32.threadcol"], "u32.index", "")
# Statements by rung. A top-level or loop-body statement may hold a barrier; a conditional one may not.
CTX = {"L0": "C0", "L1": "C1", "L2": "C2", "L3": "C3", "L4": "C4"}
for r in RUNGS:
    f("block", "one_" + r, "composite", ["stmt." + r], "block." + r, "")
    f("block", "seq2_" + r, "composite", ["stmt." + r] * 2, "block." + r, "")
    f("block", "seq3_" + r, "composite", ["stmt." + r] * 3, "block." + r, "arity 3")
    f("block", "seq4_" + r, "composite", ["stmt." + r] * 4, "block." + r, "arity 4")
    f("stmt", "block_" + r, "composite", ["block." + r], "stmt." + r, "{ ... }")
    if r in ("L0", "L1", "L2"):
        # No return under a condition: a thread-dependent return makes every
        # later barrier non-uniform (WGSL's rule); the guard is spelled as
        # `if (in range) { store }` instead.
        f("stmt", "return_" + r, "composite", [], "stmt." + r, "return; never in a conditional context")
    f("reg", "set_f_" + r, "composite", ["reg<f32>.real", "f32.real"], "stmt." + r, "")
    f("reg", "set_v_" + r, "composite", ["reg<vec4<f32>>.real", "vec4<f32>.real"], "stmt." + r, "")
    f("reg", "set_i_" + r, "composite", ["reg<u32>.index", "u32.index"], "stmt." + r, "")
    f("reg", "set_b_" + r, "composite", ["reg<bool>.flag", "bool.flag"], "stmt." + r, "")
    f("wg", "set_" + r, "composite", ["wgmem<f32>.real", "u32.wgindex", "f32.real"], "stmt." + r, "arity 3; the index is in range by its type")
    f("wg", "set_v_" + r, "composite", ["wgmem<vec4<f32>>.real", "u32.wgvindex", "vec4<f32>.real"], "stmt." + r, "arity 3: va[i] = v")
    f("parr", "set_f_" + r, "composite", ["parr<f32>.real", "u32.pindex", "f32.real"], "stmt." + r, "arity 3: pa[i] = v")
    f("parr", "set_v_" + r, "composite", ["parr<vec4<f32>>.real", "u32.pindex", "vec4<f32>.real"], "stmt." + r, "arity 3: pv[i] = v")
    f("parr", "set_i_" + r, "composite", ["parr<u32>.index", "u32.pindex", "u32.index"], "stmt." + r, "arity 3: pi[i] = v")
    f("parr", "set_b_" + r, "composite", ["parr<bool>.flag", "u32.pindex", "bool.flag"], "stmt." + r, "arity 3: pm[i] = v")
    f("store", "h_" + r, "composite", ["u32.cell", "f32.real"], "stmt." + r, "h[cell] = h[cell] + v: read-add-write of a cell this thread owns")
    # A condition puts its block in the C context of the same level: no barrier under it.
    c = CTX.get(r, r)
    f("stmt", "if_" + r, "composite", ["bool.flag", "block." + c], "stmt." + r, "if (c) { ... }: the body is a conditional context")
    f("stmt", "if_else_" + r, "composite", ["bool.flag", "block." + c, "block." + c], "stmt." + r, "arity 3")
# Barriers only where control flow is uniform: top level and loop bodies, never under a condition.
for r in ("L0", "L1", "L2", "L3", "L4"):
    f("stmt", "barrier_" + r, "composite", [], "stmt." + r, "workgroupBarrier(); never in a conditional context")
# Loops: the item loop only at the top level, bounded by m, d or rows; the tile loop at L0 or L1, bounded by a literal <= 64; no loop produces an L2 statement, so a tile loop never nests.
for b in ("boundm", "boundd", "boundrows"):
    f("loop", "items_" + b, "composite", ["u32." + b, "block.L1"], "stmt.L0", "for (t = 0u; t < bound; t = t + 1u) { ... }: the one long loop")
    f("loop", "items_unrolled_" + b, "composite", ["u32.unroll", "u32." + b, "block.L1"], "stmt.L0", "arity 3: the body k times, then the remainder")
    f("loop", "items_strided_" + b, "composite", ["u32." + b, "u32.lit4096", "block.L1"], "stmt.L0", "arity 3: for (t = 0u; t < bound; t = t + stride): a tile of items per iteration, the stride a literal in 1..=4096 (64 until 2026-10-11)")
f("loop", "tile_L0", "composite", ["u32.lit64", "block.L2"], "stmt.L0", "a short loop (counter k) at the top level")
f("loop", "tile_L1", "composite", ["u32.lit64", "block.L2"], "stmt.L1", "a short loop (counter k) inside the item loop")
f("loop", "tile_L2", "composite", ["u32.lit64", "block.L3"], "stmt.L2", "an inner short loop (counter k2) inside a tile loop; nothing produces a loop at L3")
f("loop", "tile_C0", "composite", ["u32.lit64", "block.C1"], "stmt.C0", "a short loop (counter k) under a condition; its body stays conditional (no barrier)")
f("loop", "tile_C1", "composite", ["u32.lit64", "block.C2"], "stmt.C1", "a short loop (counter k) under a condition inside the item loop")
f("loop", "tile_C2", "composite", ["u32.lit64", "block.C3"], "stmt.C2", "an inner short loop (counter k2) under a condition inside a tile loop")
f("loop", "tile_L3", "composite", ["u32.lit64", "block.L4"], "stmt.L3", "a third nested short loop (counter k3); nothing produces a loop at L4")
f("loop", "tile_C3", "composite", ["u32.lit64", "block.C4"], "stmt.C3", "a third nested short loop (counter k3) under a condition")
# A stepped tile loop: for (k = 0u; k < n; k = k + s), the step a literal in 1..=64.
for r, body in (("L0", "L2"), ("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("C0", "C1"), ("C1", "C2"), ("C2", "C3"), ("C3", "C4")):
    f("loop", "tile_strided_" + r, "composite", ["u32.lit64", "u32.lit64", "block." + body], "stmt." + r, "arity 3: a short loop stepped by a literal (the counter of its depth)")
# The kernel: a grid and a top-level block.
f("kernel", "entry_safe", "composite", ["grid.grid", "block.L0", "block.O0"], "kernel.safe", "arity 3: the safeWGSL kernel, the only producer of the root dual kernel.safe: a body (an L0 block) and an epilogue (an O0 block) that ends in the store of the thread's cell; arity 2 until 2026-10-11, when a body alone let 51 of 133 repaired offspring never write h")
# Loads of the kernel's bindings by provenance.
f("load", "phi_safe", "composite", ["u32.index"], "f32.real", "phi[i]: any index, a read past the buffer is zero (WGSL's rule)")
f("load", "phi4_safe", "composite", ["u32.index"], "vec4<f32>.real", "four consecutive values from an index")
f("load", "h_safe", "composite", ["u32.cell"], "f32.real", "h[cell], the thread's own cell")
f("wg", "get_safe", "composite", ["wgmem<f32>.real", "u32.wgindex"], "f32.real", "a[i], in range by its type")

# ---- the complete kingdom (step 6, 2026-10-11): rows the drafts and model kernels needed ----
# More registers still: 32 f32, 16 vec4, 32 u32, 8 bool (a model kernel asked for f7..f10, the reader refused "more than 16 i registers").
for i in range(16, 32):
    f("reg", f"f{i}", "composite", [], "reg<f32>.real", "a register")
for i in range(8, 16):
    f("reg", f"v{i}", "composite", [], "reg<vec4<f32>>.real", "a register")
for i in range(16, 32):
    f("reg", f"i{i}", "composite", [], "reg<u32>.index", "a register")
for i in range(4, 8):
    f("reg", f"b{i}", "composite", [], "reg<bool>.flag", "a register")
f("parr", "w", "composite", [], "parr<vec4<f32>>.real", "private array pw of 16 vec4<f32> (a second vec4 array)")
# Expression-bounded loops (k16, k26: `cfg.m & ~3u`; the pump's `cfg.m / 4u`; k06: a test `t + 4u > m`,
# normalised to t < m - 4). The bound is any index expression; the loop is finite because every u32 is;
# the cost estimator evaluates the bound at the worst shape, so a slow kernel is refused before dispatch.
f("bound", "of_index", "composite", ["u32.index"], "u32.boundx", "any index expression as a loop bound")
# A vector component chosen at run time (k08's `v[j]`, a model child's): read through a helper, since naga
# demands a constant index on a vector VALUE; the index is masked to the lanes.
f("shape", "at", "composite", ["<vecL.c>", "u32.index"], "<S.c>", "lane (i masked to the lanes) of a vector, through a helper")
# `m & ~3u` (k16, k26): the mask 0xFFFFFFFC is not exact in an f32 literal, so the reader spells it as a rounding
# down to a multiple: i - i % k. An index literal above 2^24 is refused by name, never rounded silently.
f("index", "round_down", "composite", ["u32.index", "u32.index"], "u32.index", "i rounded down to a multiple of k (k16's `m & ~3u` is round_down(m, 4))")
# A vector register's lane by index (k04, k14: `bv.x = phi[...]`; k18: `v[j]`): a read or a write of one
# component of a vec4 register, the index masked to the lanes. Spelled as a private array before, which
# left the register itself unchanged: the sweep of 2026-10-11 found three drafts wrong for it.
f("reg", "get_v_lane", "composite", ["reg<vec4<f32>>.real", "u32.index"], "f32.real", "v[i & 3u]")
for r in RUNGS:
    f("reg", "set_v_lane_" + r, "composite", ["reg<vec4<f32>>.real", "u32.index", "f32.real"], "stmt." + r, "arity 3: v[i & 3u] = x;")
# A CONTINUING item loop (k05, k26: a strided main loop then a remainder loop over the SAME counter): the counter
# is not reset; `for (; t < bound; t = t + s)`. Spelled only for a counter an earlier item loop stepped.
for b in ("boundm", "boundd", "boundrows", "boundx"):
    f("loop", "items_cont_" + b, "composite", ["u32." + b, "block.L1"], "stmt.L0", "for (; t < bound; t = t + 1u): the counter continues from the earlier item loop")
    f("loop", "items_cont_strided_" + b, "composite", ["u32." + b, "u32.lit4096", "block.L1"], "stmt.L0", "arity 3: for (; t < bound; t = t + stride), the counter continuing")
f("index", "sub_floor", "composite", ["u32.index", "u32.index"], "u32.index", "i - j, floored at zero (a loop test `t + k > m` becomes t < m - k)")
f("loop", "items_boundx", "composite", ["u32.boundx", "block.L1"], "stmt.L0", "for (t = 0u; t < expr; t = t + 1u)")
f("loop", "items_strided_boundx", "composite", ["u32.boundx", "u32.lit4096", "block.L1"], "stmt.L0", "arity 3: for (t = 0u; t < expr; t = t + stride)")
f("loop", "items_unrolled_boundx", "composite", ["u32.unroll", "u32.boundx", "block.L1"], "stmt.L0", "arity 3: the body k times, then the remainder")
for r, body in (("L0", "L2"), ("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("C0", "C1"), ("C1", "C2"), ("C2", "C3"), ("C3", "C4")):
    f("loop", "tilex_" + r, "composite", ["u32.boundx", "block." + body], "stmt." + r, "a short loop (the counter of its depth) bounded by an expression")
    f("loop", "tilex_strided_" + r, "composite", ["u32.boundx", "u32.lit64", "block." + body], "stmt." + r, "arity 3: the same, stepped by a literal")

# ---- run A's declines (2026-10-11): rows the model kernels of fit_A needed ----
# Registers again: 64 f32, 32 vec4, 64 u32, 16 bool. Five model kernels copied the kingdom's own
# rendered declarations (32 u32 registers and the four counters) and asked for a 33rd u32 register.
# The reader also stopped allocating a register for a local whose only initialiser is its form's
# fallback (the declaration already holds it), so a copy of a rendered kernel costs no registers.
for i in range(32, 64):
    f("reg", f"f{i}", "composite", [], "reg<f32>.real", "a register (f32 is spelled f32r: a reserved word)")
for i in range(16, 32):
    f("reg", f"v{i}", "composite", [], "reg<vec4<f32>>.real", "a register")
for i in range(32, 64):
    f("reg", f"i{i}", "composite", [], "reg<u32>.index", "a register (i32 is spelled i32r: a reserved word)")
for i in range(8, 16):
    f("reg", f"b{i}", "composite", [], "reg<bool>.flag", "a register")
# A lane of a vec4 element of a workgroup or private array (seed08: `sa[si][comp] = va` on a
# workgroup array of vec4): the element by its masked index, the lane masked to the four.
f("wg", "get_v_lane", "composite", ["wgmem<vec4<f32>>.real", "u32.wgvindex", "u32.index"], "f32.real", "arity 3: va[i][j & 3u]")
f("parr", "get_v_lane", "composite", ["parr<vec4<f32>>.real", "u32.pindex", "u32.index"], "f32.real", "arity 3: pv[i][j & 3u]")
for r in RUNGS:
    f("wg", "set_v_lane_" + r, "composite", ["wgmem<vec4<f32>>.real", "u32.wgvindex", "u32.index", "f32.real"], "stmt." + r, "arity 4: va[i][j & 3u] = x;")
    f("parr", "set_v_lane_" + r, "composite", ["parr<vec4<f32>>.real", "u32.pindex", "u32.index", "f32.real"], "stmt." + r, "arity 4: pv[i][j & 3u] = x;")
# A fourth nested tile loop (r2x_union_g9 and _g10 of run A: the item loop, then loops over g, j, r and c):
# the counter k4, the rungs L5 and C5 with every statement row the other rungs have, and the loops at L4/C4.
f("safe", "lt_lit64_4", "Binary::Less", ["u32.tcounter4", "u32.lit64"], "bool.bound", "a fourth nested tile loop's condition")
f("safe", "tile_counter4", "composite", [], "u32.tcounter4", "the fourth nested tile loop's counter k4 (0u outside it)")
f("safe", "tile_index4", "composite", ["u32.tcounter4"], "u32.index", "the fourth tile counter as a plain index")
f("safe", "row_plus_tile4", "composite", ["u32.threadrow", "u32.tcounter4"], "u32.threadrow", "row + min(k4, block rows - 1)")
f("safe", "col_plus_tile4", "composite", ["u32.threadcol", "u32.tcounter4"], "u32.threadcol", "col + min(k4, block columns - 1)")
f("safe", "row_plus_tile4_s", "composite", ["u32.threadrows", "u32.tcounter4"], "u32.threadrows", "row + min(k4, blk - 1) * Y")
f("safe", "col_plus_tile4_s", "composite", ["u32.threadcols", "u32.tcounter4"], "u32.threadcols", "col + min(k4, blk - 1) * X")
CTX["L5"] = "C5"
for r in RUNGS5:
    f("block", "one_" + r, "composite", ["stmt." + r], "block." + r, "")
    f("block", "seq2_" + r, "composite", ["stmt." + r] * 2, "block." + r, "")
    f("block", "seq3_" + r, "composite", ["stmt." + r] * 3, "block." + r, "arity 3")
    f("block", "seq4_" + r, "composite", ["stmt." + r] * 4, "block." + r, "arity 4")
    f("stmt", "block_" + r, "composite", ["block." + r], "stmt." + r, "{ ... }")
    f("reg", "set_f_" + r, "composite", ["reg<f32>.real", "f32.real"], "stmt." + r, "")
    f("reg", "set_v_" + r, "composite", ["reg<vec4<f32>>.real", "vec4<f32>.real"], "stmt." + r, "")
    f("reg", "set_i_" + r, "composite", ["reg<u32>.index", "u32.index"], "stmt." + r, "")
    f("reg", "set_b_" + r, "composite", ["reg<bool>.flag", "bool.flag"], "stmt." + r, "")
    f("wg", "set_" + r, "composite", ["wgmem<f32>.real", "u32.wgindex", "f32.real"], "stmt." + r, "arity 3; the index is in range by its type")
    f("wg", "set_v_" + r, "composite", ["wgmem<vec4<f32>>.real", "u32.wgvindex", "vec4<f32>.real"], "stmt." + r, "arity 3: va[i] = v")
    f("parr", "set_f_" + r, "composite", ["parr<f32>.real", "u32.pindex", "f32.real"], "stmt." + r, "arity 3: pa[i] = v")
    f("parr", "set_v_" + r, "composite", ["parr<vec4<f32>>.real", "u32.pindex", "vec4<f32>.real"], "stmt." + r, "arity 3: pv[i] = v")
    f("parr", "set_i_" + r, "composite", ["parr<u32>.index", "u32.pindex", "u32.index"], "stmt." + r, "arity 3: pi[i] = v")
    f("parr", "set_b_" + r, "composite", ["parr<bool>.flag", "u32.pindex", "bool.flag"], "stmt." + r, "arity 3: pm[i] = v")
    f("store", "h_" + r, "composite", ["u32.cell", "f32.real"], "stmt." + r, "h[cell] = h[cell] + v: read-add-write of a cell this thread owns")
    c = CTX.get(r, r)
    f("stmt", "if_" + r, "composite", ["bool.flag", "block." + c], "stmt." + r, "if (c) { ... }: the body is a conditional context")
    f("stmt", "if_else_" + r, "composite", ["bool.flag", "block." + c, "block." + c], "stmt." + r, "arity 3")
    f("reg", "set_v_lane_" + r, "composite", ["reg<vec4<f32>>.real", "u32.index", "f32.real"], "stmt." + r, "arity 3: v[i & 3u] = x;")
    f("wg", "set_v_lane_" + r, "composite", ["wgmem<vec4<f32>>.real", "u32.wgvindex", "u32.index", "f32.real"], "stmt." + r, "arity 4: va[i][j & 3u] = x;")
    f("parr", "set_v_lane_" + r, "composite", ["parr<vec4<f32>>.real", "u32.pindex", "u32.index", "f32.real"], "stmt." + r, "arity 4: pv[i][j & 3u] = x;")
f("stmt", "barrier_L5", "composite", [], "stmt.L5", "workgroupBarrier(); never in a conditional context")
f("loop", "tile_L4", "composite", ["u32.lit64", "block.L5"], "stmt.L4", "a fourth nested short loop (counter k4); nothing produces a loop at L5")
f("loop", "tile_C4", "composite", ["u32.lit64", "block.C5"], "stmt.C4", "a fourth nested short loop (counter k4) under a condition")
for r, body in (("L4", "L5"), ("C4", "C5")):
    f("loop", "tile_strided_" + r, "composite", ["u32.lit64", "u32.lit64", "block." + body], "stmt." + r, "arity 3: a short loop stepped by a literal (the counter of its depth)")
    f("loop", "tilex_" + r, "composite", ["u32.boundx", "block." + body], "stmt." + r, "a short loop (the counter of its depth) bounded by an expression")
    f("loop", "tilex_strided_" + r, "composite", ["u32.boundx", "u32.lit64", "block." + body], "stmt." + r, "arity 3: the same, stepped by a literal")
# A block offset that is any index expression, clamped into the block (fit_A2's r2x_reinvent: a row offset
# 2 * q + s over two tile counters, which the reader proves below the block factor from the counters' bounds):
# the clamp keeps the row this thread's whatever the expression.
f("safe", "row_plus_x", "composite", ["u32.threadrow", "u32.index"], "u32.threadrow", "row + min(off, block rows - 1): any offset expression, clamped into the block")
f("safe", "col_plus_x", "composite", ["u32.threadcol", "u32.index"], "u32.threadcol", "col + min(off, block columns - 1)")
f("safe", "row_plus_x_s", "composite", ["u32.threadrows", "u32.index"], "u32.threadrows", "row + min(off, blk - 1) * Y: the off-th row of a strided block")
f("safe", "col_plus_x_s", "composite", ["u32.threadcols", "u32.index"], "u32.threadcols", "col + min(off, blk - 1) * X")
# u32(flag) (fit_A2's r2x_reinvent_g2: an index multiplied by u32(ok)).
f("convert", "flag_to_index", "As", ["bool.flag"], "u32.index", "u32(b): 1 or 0")
f("convert", "flag_to_f32", "As", ["bool.flag"], "f32.real", "f32(b): 1.0 or 0.0")
# A GROUPED block (fit_A2's r6x_reinvent: a thread's 8 columns as 2 groups of 4 contiguous columns 32 apart,
# col = wid.x * 64u + hf * 32u + lid.x * 4u + e): the thread's first cell of the block, and a cell of it by group
# and lane, both clamped (G groups of B / G cells, the groups W * B / G apart; G = 1 is the contiguous block).
f("safe", "thread_row_g", "composite", ["vec3<u32>.wid", "vec3<u32>.lid", "u32.blk", "u32.lit64"], "u32.threadrowg", "arity 4: wid.y * (Y * blk) + lid.y * (blk / G): the first row of a thread's grouped block")
f("safe", "thread_col_g", "composite", ["vec3<u32>.wid", "vec3<u32>.lid", "u32.blk", "u32.lit64"], "u32.threadcolg", "arity 4: wid.x * (X * blk) + lid.x * (blk / G)")
f("safe", "row_plus_g", "composite", ["u32.threadrowg", "u32.lit64", "u32.index", "u32.index"], "u32.threadrowg", "arity 4: row + min(g, G - 1) * (Y * blk / G) + min(e, blk / G - 1)")
f("safe", "col_plus_g", "composite", ["u32.threadcolg", "u32.lit64", "u32.index", "u32.index"], "u32.threadcolg", "arity 4: col + min(g, G - 1) * (X * blk / G) + min(e, blk / G - 1)")
# Four vec4 private arrays (fit_A2 gen 2's r2x_reinvent uses three: pe, po, p).
f("parr", "x", "composite", [], "parr<vec4<f32>>.real", "private array px of 16 vec4<f32> (a third vec4 array)")
f("parr", "y", "composite", [], "parr<vec4<f32>>.real", "private array py of 16 vec4<f32> (a fourth vec4 array)")

# ---- the epilogue (2026-10-11): rows of the O rungs, after every other row so every ref stands ----
# O<d> at tile depth d: a block of up to three conditional statements (C<d>) then one epilogue statement;
# an epilogue statement is the store itself, an if or if-else whose blocks are epilogues, a block, or (below
# depth 4) a tile loop whose body is the epilogue of the next depth.
for d, r in enumerate(EPILOGUE):
    c = f"C{d}"
    f("block", "one_" + r, "composite", ["stmt." + r], "block." + r, "the epilogue's last statement")
    f("block", "seq2_" + r, "composite", ["stmt." + c, "stmt." + r], "block." + r, "a conditional statement, then the epilogue's last")
    f("block", "seq3_" + r, "composite", ["stmt." + c, "stmt." + c, "stmt." + r], "block." + r, "arity 3")
    f("block", "seq4_" + r, "composite", ["stmt." + c, "stmt." + c, "stmt." + c, "stmt." + r], "block." + r, "arity 4")
    f("stmt", "block_" + r, "composite", ["block." + r], "stmt." + r, "{ ... }")
    f("store", "h_" + r, "composite", ["u32.cell", "f32.real"], "stmt." + r, "h[cell] = h[cell] + v: the store that ends the kernel")
    f("stmt", "if_" + r, "composite", ["bool.flag", "block." + r], "stmt." + r, "if (c) { epilogue }")
    f("stmt", "if_else_" + r, "composite", ["bool.flag", "block." + r, "block." + r], "stmt." + r, "arity 3: both branches end in a store")
    if d + 1 < len(EPILOGUE):
        body = "block." + EPILOGUE[d + 1]
        f("loop", "tile_" + r, "composite", ["u32.lit64", body], "stmt." + r, "a short loop whose body ends in the store")
        f("loop", "tile_strided_" + r, "composite", ["u32.lit64", "u32.lit64", body], "stmt." + r, "arity 3: stepped by a literal")
        f("loop", "tilex_" + r, "composite", ["u32.boundx", body], "stmt." + r, "bounded by an expression")
        f("loop", "tilex_strided_" + r, "composite", ["u32.boundx", "u32.lit64", body], "stmt." + r, "arity 3: bounded by an expression, stepped by a literal")

# ---- the thread's axes (2026-10-11): a thread id's components by axis, after every other row ----
# x is the column axis, y the row axis (the grid dispatches columns on x and rows on y); z is a plain index.
for b in ("gid", "lid", "wid"):
    f("axis", "x_" + b, "AccessIndex", ["vec3<u32>." + b], "u32.colid", f"{b}.x: an index on the column axis")
    f("axis", "y_" + b, "AccessIndex", ["vec3<u32>." + b], "u32.rowid", f"{b}.y: an index on the row axis")
    f("axis", "z_" + b, "AccessIndex", ["vec3<u32>." + b], "u32.index", f"{b}.z")
f("index", "of_row", "composite", ["u32.rowid"], "u32.index", "a row-axis index used as a plain index")
f("index", "of_col", "composite", ["u32.colid"], "u32.index", "a column-axis index used as a plain index")
f("index", "of_lidx", "composite", ["u32.lidx"], "u32.index", "the thread's index within its workgroup used as a plain index")
# The cell and the plain index of a row or column of each block kind.
for r in ("threadrow", "threadrows", "threadrowg"):
    for c in ("threadcol", "threadcols", "threadcolg"):
        if (r, c) != ("threadrow", "threadcol"):
            f("safe", "cell_of", "composite", ["u32." + r, "u32." + c], "u32.cell", "row * cfg.d + col, the cell this thread owns")
for r in ("threadrows", "threadrowg"):
    f("safe", "row_as_index", "composite", ["u32." + r], "u32.index", "")
for c in ("threadcols", "threadcolg"):
    f("safe", "col_as_index", "composite", ["u32." + c], "u32.index", "")

# ---- continue (2026-10-11), in the epilogue only ----
# `if (c) { continue; }` skips the rest of a pass. Anywhere a later barrier of the same loop body could stand,
# a thread-dependent continue would make that barrier non-uniform, and a rung cannot say which loop it sits in.
# The epilogue holds no barrier, and an epilogue block at depth d >= 1 is the body of an epilogue tile loop, so
# there the continue is spelled: the guard, then the rest of the block (blocked.wgsl's `if (lr >= r_rows) { continue; }`).
for d in range(1, len(EPILOGUE)):
    r = EPILOGUE[d]
    f("block", "cont_" + r, "composite", ["bool.flag", "block." + r], "block." + r, "if (c) { continue; } then the rest of the epilogue block: skip this pass of the epilogue's tile loop")

# ---- the uniforms (2026-10-11), each its own dual: the role's loads are typed so in the safe kingdom ----
# (safe_karva::per_kernel_rows), and these rows carry them into index and count arithmetic.
for u in ("cfgm", "cfgd", "cfgrows", "cfgroff"):
    f("index", "of_" + u, "composite", ["u32." + u], "u32.index", f"the uniform ({FORMS[u][0]}) as a plain index")
for u in ("cfgm", "cfgd", "cfgrows"):
    f("convert", u + "_count", "composite", ["u32." + u], "u32.count", f"the uniform ({FORMS[u][0]}) as a count")

# ---- a register set to zero (2026-10-11), a row with no literal: a native offspring of fit_A2 started an
# accumulator at vec4(8.0) for vec4(0.0) (a literal mutation); the corpus spells 703 register zero-inits ----
for r in RUNGS + RUNGS5:
    for kind, reg in (("f", "reg<f32>.real"), ("v", "reg<vec4<f32>>.real"), ("i", "reg<u32>.index"), ("b", "reg<bool>.flag")):
        f("reg", f"clear_{kind}_{r}", "composite", [reg], "stmt." + r, "r = 0 (its form's zero): a reset with no literal to mutate")


def write_tsv(path, header, rows):
    with open(path, "w") as out:
        out.write("\t".join(header) + "\n")
        for r in rows:
            out.write("\t".join(str(x) for x in r) + "\n")


def main():
    types = build_types()
    write_tsv(os.path.join(HERE, "types.tsv"), ["ref", "dual", "slot", "form", "lanes", "fallback", "fallback_bits", "fallback_noop", "fallback_elem", "meaning"], types)
    frows = [(f"F{i + 1:03d}", c, inst, naga, " ".join(ins) if ins else "-", out, len(ins), note) for i, (c, inst, naga, ins, out, note) in enumerate(FUNCTIONS)]
    write_tsv(os.path.join(HERE, "functions.tsv"), ["ref", "class", "instance", "naga", "in", "out", "arity", "note"], frows)

    with open(os.path.join(HERE, "TYPES.md"), "w") as md:
        md.write("# safeWGSL kingdom — the type table\n\n")
        md.write("Generated by `gen_tables.py` from `types.tsv`; edit the script, not this file.\n\n")
        md.write("A type is a dual `slot.form`: the slot is what the hardware stores, the form is what the value means and which functions are legal on it. Every dual has a unique ref. naga admits no implicit conversion, so a function's signature over duals is exact; the `convert` class holds the only crossings.\n\n")
        md.write("## Forms\n\n| form | meaning | legal scalar slots |\n|---|---|---|\n")
        for form, (meaning, scalars) in FORMS.items():
            md.write(f"| `{form}` | {meaning} | {', '.join(scalars) if scalars else 'see duals'} |\n")
        md.write("\n## Duals\n\n| ref | dual | slot | form | lanes | fallback | bits | noop | elem |\n|---|---|---|---|---|---|---|---|---|\n")
        for ref, dual, slot, form, lanes, fb, bits, noop, elem, _ in types:
            md.write(f"| {ref} | `{dual}` | `{slot}` | `{form}` | {lanes} | `{fb}` | {bits} | {noop} | {elem} |\n")
        md.write(f"\n{len(types)} duals. The fallback is what Design C's projection substitutes for a codon whose dual does not match the demand; the codon is counted as masked.\n")
        md.write("\nThe three machine columns (`fallback_bits`, `fallback_noop`, `fallback_elem` in `types.tsv`) are what the loader builds the fallback leaf from, never the prose: `bits` is the lane's 32-bit pattern as an unsigned integer, repeated across the lanes of a vector or matrix (only `bool.sign` is non-zero, true = +1); `noop` is 1 for `store.store`, which substitutes no value at all; `elem` is 1 for `array<T>`, whose fallback is its element's. The loader tests `noop` and `elem` before reading `bits`; for those two rows `bits` is a placeholder 0. Every substitution, whichever column decides it, counts the codon as masked.\n")
        md.write("\n## Not in the table\n\nAtomics, images, samplers, ray queries, binding arrays, pointers as values, non-square `f16` matrices.\n")

    with open(os.path.join(HERE, "symbols.md"), "w") as md:
        md.write("# safeWGSL kingdom — the function table\n\n")
        md.write("Generated by `gen_tables.py` from `functions.tsv`; edit the script, not this file.\n\n")
        md.write("A function is `class.instance` with a signature over duals. `<S.real>` stands for every real dual over scalar and vector lanes, `<S.c>` for any dual of form c, `<vecL.c>` for a vector of L lanes; the loader instantiates one row per legal dual, lanes equal across inputs unless the note says otherwise. The `naga` column is the IR node the row reads from and writes to; `composite` is a row the kingdom defines as a small WGSL function.\n\n")
        md.write("**Arity.** naga's select, fma, clamp, mix, smoothstep and extractBits take three operands. The decoders' limit is four (`K_MAX` = 4, `docs/PLAN_wgsl_kernel_reader.md` §2a, delivered in both crates 2026-10-08): the node is `(fn_id, first_child, ty_code, konst)` with children contiguous in level order and the arity from the kingdom's table; templates above four operands stay composed from two rows.\n\n")
        cur = None
        for ref, c, inst, naga, ins, out, arity, note in frows:
            if c != cur:
                cur = c
                md.write(f"\n## class `{c}`\n\n| ref | instance | naga | in | out | n | note |\n|---|---|---|---|---|---|---|\n")
            md.write(f"| {ref} | `{c}.{inst}` | `{naga}` | `{ins}` | `{out}` | {arity} | {note} |\n")
        md.write(f"\n{len(frows)} function templates. Statement rows (kernel, grid, block, stmt, loop, reg, wg, the three literal codes and builtin.item/tile) make the kingdom multityped over whole kernels (`docs/PLAN_wgsl_gram_superopt.md` §2); `src/wgsl/render.rs` writes their text. Deliberately absent: atomics, images, derivatives, subgroup operations, pointers as values, implicit conversions, raw multiply on `q15_16`, modulo on a hash, arithmetic on `bits`.\n")
    print(f"{len(types)} duals, {len(frows)} functions (safeWGSL)")


if __name__ == "__main__":
    main()
