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
}

FALLBACK = {
    "real": "0.0", "int": "0", "q15_16": "0", "q15_16w": "vec2(0, 0)", "index": "0u", "count": "0u",
    "code4": "0u", "code8": "0u", "code16": "0u", "code1024": "0u", "bits": "0u", "hash32": "0u",
    "hash64": "vec2(0u, 0u)", "flag": "false", "sign": "true (+1)", "opaque": "the slot's zero", "store": "no-op, codon masked",
}

# The fallback by machine: the lane's 32-bit pattern as an unsigned integer. Vectors and matrices
# repeat the lane, so one number suffices. Only `sign` (true = +1) is non-zero. `store` substitutes
# nothing (fallback_masked = 1) and `array<T>` defers to its element (fallback_elem = 1); their bits
# are placeholders the loader must not read before testing those two columns.
FALLBACK_BITS = {
    "real": 0, "int": 0, "q15_16": 0, "q15_16w": 0, "index": 0, "count": 0,
    "code4": 0, "code8": 0, "code16": 0, "code1024": 0, "bits": 0, "hash32": 0,
    "hash64": 0, "flag": 0, "sign": 1, "opaque": 0, "store": 0, "<form of T>": 0,
}
assert set(FALLBACK_BITS) >= set(FALLBACK) | {"<form of T>"}, "every form with a fallback needs its bits"


def slot_name(scalar: str, lanes: int) -> str:
    return scalar if lanes == 1 else f"vec{lanes}<{scalar}>"


def build_types():
    rows = []  # (ref, dual, slot, form, lanes, fallback, fallback_bits, fallback_masked, fallback_elem, meaning)

    def add(slot, form, lanes, meaning):
        masked = int(form == "store")
        elem = int(slot == "array<T>")
        rows.append((f"T{len(rows) + 1:03d}", f"{slot}.{form}", slot, form, lanes, FALLBACK.get(form, "the element's fallback"), FALLBACK_BITS[form], masked, elem, meaning))

    for form, (meaning, scalars) in FORMS.items():
        for s in scalars:
            for l in LANES:
                # vector forms that make no sense: codes, counts and hashes are scalar-only
                if l > 1 and form in ("count", "code4", "code8", "code16", "code1024", "hash32", "sign"):
                    continue
                add(slot_name(s, l), form, l, meaning)
    add("vec2<i32>", "q15_16w", 2, FORMS["q15_16w"][0])
    add("vec2<u32>", "hash64", 2, FORMS["hash64"][0])
    for c in (2, 3, 4):
        for r in (2, 3, 4):
            add(f"mat{c}x{r}<f32>", "real", c * r, "a matrix of plain numbers")
    add("array<T>", "<form of T>", 0, "a storage or uniform buffer of T; indexed by u32.index; one row per buffer at read-in")
    add("store", "store", 0, "the statement a head gene's root produces")
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
f("arith", "mul_mat_mat", "Binary::Multiply", ["<matAxB<f32>.real>", "<matBxC<f32>.real>"], "<matAxC<f32>.real>", "")
f("arith", "neg", "Unary::Negate", ["<S.real>"], "<S.real>", "")
for op in ["abs", "sign", "floor", "ceil", "round", "fract", "trunc", "saturate", "sqrt", "inverse_sqrt", "exp", "exp2", "log", "log2"]:
    f("arith", op, f"Math::{op.title().replace('_', '')}", ["<S.real>"], "<S.real>", "")
for op in ["min", "max", "step", "pow", "atan2", "ldexp"]:
    f("arith", op, f"Math::{op.title()}", ["<S.real>", "<S.real>"], "<S.real>", "")
for op in ["clamp", "mix", "fma", "smoothstep"]:
    f("arith", op, f"Math::{op.title()}", ["<S.real>", "<S.real>", "<S.real>"], "<S.real>", "arity 3")
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


def write_tsv(path, header, rows):
    with open(path, "w") as out:
        out.write("\t".join(header) + "\n")
        for r in rows:
            out.write("\t".join(str(x) for x in r) + "\n")


def main():
    types = build_types()
    write_tsv(os.path.join(HERE, "types.tsv"), ["ref", "dual", "slot", "form", "lanes", "fallback", "fallback_bits", "fallback_masked", "fallback_elem", "meaning"], types)
    frows = [(f"F{i + 1:03d}", c, inst, naga, " ".join(ins) if ins else "-", out, len(ins), note) for i, (c, inst, naga, ins, out, note) in enumerate(FUNCTIONS)]
    write_tsv(os.path.join(HERE, "functions.tsv"), ["ref", "class", "instance", "naga", "in", "out", "arity", "note"], frows)

    with open(os.path.join(HERE, "TYPES.md"), "w") as md:
        md.write("# WGSL kingdom — the type table\n\n")
        md.write("Generated by `gen_tables.py` from `types.tsv`; edit the script, not this file.\n\n")
        md.write("A type is a dual `slot.form`: the slot is what the hardware stores, the form is what the value means and which functions are legal on it. Every dual has a unique ref. naga admits no implicit conversion, so a function's signature over duals is exact; the `convert` class holds the only crossings.\n\n")
        md.write("## Forms\n\n| form | meaning | legal scalar slots |\n|---|---|---|\n")
        for form, (meaning, scalars) in FORMS.items():
            md.write(f"| `{form}` | {meaning} | {', '.join(scalars) if scalars else 'see duals'} |\n")
        md.write("\n## Duals\n\n| ref | dual | slot | form | lanes | fallback | bits | masked | elem |\n|---|---|---|---|---|---|---|---|---|\n")
        for ref, dual, slot, form, lanes, fb, bits, masked, elem, _ in types:
            md.write(f"| {ref} | `{dual}` | `{slot}` | `{form}` | {lanes} | `{fb}` | {bits} | {masked} | {elem} |\n")
        md.write(f"\n{len(types)} duals. The fallback is what Design C's projection substitutes for a codon whose dual does not match the demand; the codon is counted as masked.\n")
        md.write("\nThe three machine columns (`fallback_bits`, `fallback_masked`, `fallback_elem` in `types.tsv`) are what the loader builds the fallback leaf from, never the prose: `bits` is the lane's 32-bit pattern as an unsigned integer, repeated across the lanes of a vector or matrix (only `bool.sign` is non-zero, true = +1); `masked` is 1 for `store.store`, whose fallback is a no-op; `elem` is 1 for `array<T>`, whose fallback is its element's. The loader tests `masked` and `elem` before reading `bits`; for those two rows `bits` is a placeholder 0.\n")
        md.write("\n## Not in the table\n\nAtomics, images, samplers, ray queries, binding arrays, pointers as values, non-square `f16` matrices.\n")

    with open(os.path.join(HERE, "symbols.md"), "w") as md:
        md.write("# WGSL kingdom — the function table\n\n")
        md.write("Generated by `gen_tables.py` from `functions.tsv`; edit the script, not this file.\n\n")
        md.write("A function is `class.instance` with a signature over duals. `<S.real>` stands for every real dual over scalar and vector lanes, `<S.c>` for any dual of form c, `<vecL.c>` for a vector of L lanes; the loader instantiates one row per legal dual, lanes equal across inputs unless the note says otherwise. The `naga` column is the IR node the row reads from and writes to; `composite` is a row the kingdom defines as a small WGSL function.\n\n")
        md.write("**Arity.** naga's select, fma, clamp, mix, smoothstep and extractBits take three operands; the decoders' structural limit is two today. This kingdom needs the limit at three; four-operand nodes are composed from two rows.\n\n")
        cur = None
        for ref, c, inst, naga, ins, out, arity, note in frows:
            if c != cur:
                cur = c
                md.write(f"\n## class `{c}`\n\n| ref | instance | naga | in | out | n | note |\n|---|---|---|---|---|---|---|\n")
            md.write(f"| {ref} | `{c}.{inst}` | `{naga}` | `{ins}` | `{out}` | {arity} | {note} |\n")
        md.write(f"\n{len(frows)} function templates. Deliberately absent: atomics, barriers, images, derivatives, subgroup operations, pointers as values, implicit conversions, raw multiply on `q15_16`, modulo on a hash, arithmetic on `bits`.\n")
    print(f"{len(types)} duals, {len(frows)} functions")


if __name__ == "__main__":
    main()
