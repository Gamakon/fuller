#!/usr/bin/env python3
"""The WGSL kingdom's tables, generated from one source of truth.

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
    "wgdim": ("a workgroup dimension, a literal from {4, 8, 16, 32, 64}", ["u32"]),
    "blk": ("a block factor, a literal from {1, 2, 4}", ["u32"]),
    "unroll": ("an unroll factor, a literal from {2, 4, 8}", ["u32"]),
}

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
        if form in ("wgdim", "blk", "unroll"):
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
    add("wgmem<f32>", "real", 0, "a workgroup-memory array of 2,048 f32: a, b (WGMEM_LEN in src/wgsl/render.rs)")
    for form in ("wgdim", "blk", "unroll"):
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
for b, out in [("global_invocation_id", "vec3<u32>.index"), ("local_invocation_id", "vec3<u32>.index"), ("workgroup_id", "vec3<u32>.index"), ("local_invocation_index", "u32.index"), ("num_workgroups", "vec3<u32>.count")]:
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
# naga's own names: SmoothStep (title-casing gave Smoothstep, which no naga function is; 2026-10-11).
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
    f("bits", "un" + op, f"Math::Un{op}", ["u32.bits"], lanes, "")
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
f("literal", "wgdim", "Literal", [], "u32.wgdim", "a workgroup dimension from {4, 8, 16, 32, 64}; the renderer snaps the constant to the set")
f("literal", "blk", "Literal", [], "u32.blk", "a block factor from {1, 2, 4}")
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
        md.write("# WGSL kingdom — the type table\n\n")
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
        md.write("# WGSL kingdom — the function table\n\n")
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
    print(f"{len(types)} duals, {len(frows)} functions")


if __name__ == "__main__":
    main()
