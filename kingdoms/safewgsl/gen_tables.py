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
    "wgdim": ("a workgroup dimension, any integer 1..=1,024 (the M3 Max's per-axis limit); a grid's y is cut so x * y <= 1,024 invocations (2026-10-12; {4, 8, 16, 32} before)", ["u32"]),
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
    "phi": ("unused: a load of phi takes any index, since a read past the buffer is the buffer's LAST element (the device clamps the index, naga Restrict); kept so earlier refs stand", ["u32"]),
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
    "pindex16": ("an index into a private array of 16 elements (i & 15u)", ["u32"]),
    "pindex128": ("an index into a private array of 128 elements (i & 127u)", ["u32"]),
    "uniform": ("a u32 the same for every thread of the workgroup: a binding's field, a literal, the workgroup's id, the grid's size, the subgroup size, and arithmetic of them", ["u32"]),
    "boundu": ("a loop bound the same for every thread of the workgroup (a loop that may hold a barrier)", ["u32"]),
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
    # WORKGROUP ARRAYS AS KNOBS (2026-10-12): which arrays a kernel declares, their element type (f32, vec2,
    # vec4) and their length, within the device's 32 KB; four fixed arrays of 2,048 f32 / 512 vec4 before.
    "wgv2index": ("an index into a vec2 workgroup array, masked to the array by its row", ["u32"]),
    "wglen": ("a workgroup array's length, 1..=8,192 elements (the renderer halves the largest array until all fit 32 KB)", ["u32"]),
    "decl": ("one workgroup array declaration: the array and its length", []),
    "decls": ("the kernel's workgroup array declarations", []),
    # INTEGER ATOMICS (2026-10-12) in workgroup memory: arrays of atomic<u32> and atomic<i32>, an index masked
    # into them, and the atomic built-ins as rows.
    "wgaindex": ("an index into an atomic workgroup array, masked to the array by its row", ["u32"]),
    # A loop body that breaks (2026-10-12, WGSL_COVERAGE.md gap 7): conditional statements, the guard
    # `if (c) { break; }`, conditional statements. No barrier anywhere in it, so a thread that breaks
    # never skips a barrier the others reach. The digit is the body's rung (its tile depth).
    "B1": ("a loop body that breaks, one tile deep: no barrier", []),
    "B2": ("a loop body that breaks, two tile loops deep (or a tile loop in the item loop): no barrier", []),
    "B3": ("a loop body that breaks, three deep: no barrier", []),
    "B4": ("a loop body that breaks, four deep: no barrier", []),
    "B5": ("a loop body that breaks, five deep: no barrier", []),
    # NO WRAP IN A LOOP BOUND (2026-10-12, Andrew: a u32 wrap is illegal in a loop bound, which it runs to its
    # cap, and in a store's cell index; a diagnostic elsewhere). The uniform values that may wrap (a difference,
    # a product, a shift) are their own dual, an index only; a loop bound's value is built from registers,
    # uniform values and arithmetic that does not grow past its operands, or clamped by a min.
    "uniformw": ("a uniform u32 whose arithmetic may wrap (a difference, a product, a shift of uniform values): no row takes it (a wrap is illegal everywhere)", ["u32"]),
    # THE ORIGINS (2026-10-12): an offset row (safe.row_plus*) takes the thread's origin only, never another
    # offset row: nested, each offset clamped to the block, the sum left it (the reader's ownership proof
    # refused 50 of 1,000 random trees; the cell was another thread's).
    "rowbase": ("the ORIGIN of a thread's row: the only row an offset row takes (an offset of an offset leaves the block, another thread's cell)", ["u32"]),
    "colbase": ("the ORIGIN of a thread's column: the only row an offset row takes (an offset of an offset leaves the block, another thread's cell)", ["u32"]),
    "rowbases": ("the ORIGIN of a thread's row of a strided block: the only row an offset row takes (an offset of an offset leaves the block, another thread's cell)", ["u32"]),
    "colbases": ("the ORIGIN of a thread's column of a strided block: the only row an offset row takes (an offset of an offset leaves the block, another thread's cell)", ["u32"]),
    "rowbaseg": ("the ORIGIN of a thread's row of a grouped block: the only row an offset row takes (an offset of an offset leaves the block, another thread's cell)", ["u32"]),
    "colbaseg": ("the ORIGIN of a thread's column of a grouped block: the only row an offset row takes (an offset of an offset leaves the block, another thread's cell)", ["u32"]),
    "indexw": ("an index that may wrap (a raw difference index.sub, a left shift index.shl): no row takes it; the kingdom's difference is index.sub_floor, its shift by k a product by 2^k", ["u32"]),
    "wgown": ("a workgroup-array element this thread owns, lidx + k * the workgroup's threads (k a literal or a tile counter; 2026-10-14): a put writes it only below the array's length, so no two threads of a workgroup write one element", ["u32"]),
    "mirror": ("the MIRROR of a cell this thread owns, col * cfg.d + row (the symmetric half's second write, 2026-10-14): a kernel that writes one runs with the whole H (`// mode full`); no sampler draws it, the symmetric-half rule writes it under its triangle guard and the checker's layer 7 proves it", ["u32"]),
    "payload": ("a knob reference's payload (fuller params.rs): the constant of a literal.p_* row, the knob times payload / 256; its own dual so a point mutation cannot swap a knob reference for an index literal (seed01's mutant 616, 2026-10-17); no row outputs it", ["u32"]),
    "xbound": ("the value of an expression loop bound: a register, a uniform value, the arithmetic of them that cannot wrap (no difference but the floored one, no product, no shift), or a min with any index", ["u32"]),
    # SEVERAL ENTRY POINTS (2026-10-17, B6 of PLAN_v24_ingest_evolution: an index's `ingest` and `search` in one
    # module over shared bindings): the root of a gene whose interface names two to four entry points, one safe
    # kernel each, in the interface's order. Nothing takes it, so a module never nests and a single-entry gene
    # (kernel.safe) is never mutated into one.
    "uflag": ("a predicate the same for every thread of the workgroup: comparisons of uniform values and their logic (a test whose body keeps its rung)", ["bool"]),
    "module": ("a module of two to four entry points over the interface's shared bindings, one safe kernel each, in the interface's order: a gene's root", []),
}

SAFE_FORMS = ["pindex128", "payload", "uflag", "module", "wgown", "mirror", "rowbase", "colbase", "rowbases", "colbases", "rowbaseg", "colbaseg", "indexw", "uniformw", "xbound", "uniform", "boundu", "pindex16", "wgaindex", "wgv2index", "wglen", "decl", "decls", "lidx", "threadrows", "threadcols", "threadrowg", "threadcolg", "gid", "lid", "wid", "rowid", "colid", "O0", "O1", "O2", "O3", "O4", "tcounter4", "L5", "C5", "safe", "pindex", "boundx", "boundm", "boundd", "boundrows", "lit64", "lit4096", "counter", "tcounter", "tcounter2", "L3", "C3", "tcounter3", "L4", "C4", "wgvindex", "bound", "threadrow", "threadcol", "cell", "phi", "wgindex", "L0", "L1", "L2", "C0", "C1", "C2"]
RUNGS = ["L0", "L1", "L2", "L3", "L4", "C0", "C1", "C2", "C3", "C4"]
# The fifth rung came after every other row (run A, 2026-10-11); its rows are appended at the end so
# every earlier ref stands, and it is kept out of RUNGS for that reason.
RUNGS5 = ["L5", "C5"]
# The epilogue's rungs by tile depth; O<d>'s leading statements are C<d> statements.
EPILOGUE = ["O0", "O1", "O2", "O3", "O4"]

FALLBACK = {
    "payload": "256 (the knob itself)",
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
    "counter": "0u", "tcounter": "0u", "tcounter2": "0u", "bound": "false", "threadrow": "the thread's own first row", "threadcol": "the thread's own first column", "cell": "the thread's own first cell", "phi": "0u", "wgindex": "0u",
    "L0": "empty", "L1": "empty", "L2": "empty", "L3": "empty", "C0": "empty", "C1": "empty", "C2": "empty", "C3": "empty",
    "safe": "an empty kernel", "pindex": "0u", "pindex16": "0u", "pindex128": "0u", "uniform": "0u", "boundu": "1u",
    "tcounter3": "0u", "L4": "empty", "C4": "empty", "wgvindex": "0u",
    "boundx": "1u",
    "tcounter4": "0u", "L5": "empty", "C5": "empty",
    # The epilogue's fallback is a store, never nothing: h[cell] = h[cell] + f0 at the thread's first cell
    # (gid.y * by, gid.x * bx), under the store's own bounds check.
    "O0": "the thread's own store", "O1": "the thread's own store", "O2": "the thread's own store", "O3": "the thread's own store", "O4": "the thread's own store",
    "gid": "0u", "lid": "0u", "wid": "0u", "rowid": "0u", "colid": "0u",
    "lidx": "0u", "threadrows": "the thread's own first row", "threadcols": "the thread's own first column", "threadrowg": "the thread's own first row", "threadcolg": "the thread's own first column",
    "wgaindex": "0u", "wgv2index": "0u", "wglen": "the array's default length", "decl": "none", "decls": "none (the arrays the kernel uses, at their default lengths)",
    "B1": "empty", "B2": "empty", "B3": "empty", "B4": "empty", "B5": "empty",
    "uniformw": "0u", "xbound": "1u", "indexw": "0u", "mirror": "the mirror of the thread's own first cell", "wgown": "the thread's own first element (lidx)",
    "rowbase": "the thread's own first row", "colbase": "the thread's own first column", "rowbases": "the thread's own first row", "colbases": "the thread's own first column", "rowbaseg": "the thread's own first row", "colbaseg": "the thread's own first column",
    "module": "an empty kernel at each of the interface's entry points",
    "uflag": "false",
}

# The fallback by machine: the lane's 32-bit pattern as an unsigned integer. Vectors and matrices
# repeat the lane, so one number suffices. Only `sign` (true = +1) is non-zero. `store` substitutes
# nothing (fallback_noop = 1; every substitution counts the codon as masked, so that word is not
# reused here) and `array<T>` defers to its element (fallback_elem = 1); their bits are placeholders
# the loader must not read before testing those two columns.
FALLBACK_BITS = {
    "payload": 256,
    "real": 0, "int": 0, "q15_16": 0, "q15_16w": 0, "index": 0, "count": 0,
    "code4": 0, "code8": 0, "code16": 0, "code1024": 0, "bits": 0, "hash32": 0,
    "hash64": 0, "flag": 0, "sign": 1, "opaque": 0, "store": 0, "<form of T>": 0,
    "kernel": 0, "grid": 0, "block": 0, "stmt": 0, "wgdim": 0, "blk": 0, "unroll": 0,
    "boundm": 0, "boundd": 0, "boundrows": 0, "lit64": 0, "lit4096": 0,
    "counter": 0, "tcounter": 0, "tcounter2": 0, "bound": 0, "threadrow": 0, "threadcol": 0, "cell": 0, "phi": 0, "wgindex": 0,
    "L0": 0, "L1": 0, "L2": 0, "L3": 0, "C0": 0, "C1": 0, "C2": 0, "C3": 0, "safe": 0, "pindex": 0, "pindex16": 0, "pindex128": 0, "uniform": 0, "boundu": 1,
    "tcounter3": 0, "L4": 0, "C4": 0, "wgvindex": 0,
    # A loop bound's fallback is one pass, as its text says (1u; 0 before, 2026-10-12: a loop that never
    # ran, and an epilogue whose store sat in it never stored).
    "boundx": 1,
    "tcounter4": 0, "L5": 0, "C5": 0,
    "O0": 0, "O1": 0, "O2": 0, "O3": 0, "O4": 0,
    "gid": 0, "lid": 0, "wid": 0, "rowid": 0, "colid": 0,
    "lidx": 0, "threadrows": 0, "threadcols": 0, "threadrowg": 0, "threadcolg": 0,
    "wgaindex": 0, "wgv2index": 0, "wglen": 0, "decl": 0, "decls": 0,
    "B1": 0, "B2": 0, "B3": 0, "B4": 0, "B5": 0,
    "uniformw": 0, "xbound": 1, "indexw": 0, "mirror": 0, "wgown": 0,
    "rowbase": 0, "colbase": 0, "rowbases": 0, "colbases": 0, "rowbaseg": 0, "colbaseg": 0,
    "module": 0,
    "uflag": 0,
}
assert set(FALLBACK_BITS) == set(FALLBACK), "every form with a fallback needs its bits, and no more"


def slot_name(scalar: str, lanes: int) -> str:
    return scalar if lanes == 1 else f"vec{lanes}<{scalar}>"


# THE LITERAL DOMAINS (phylu docs/PLAN_wgsl_literals_as_constants.md, 2.3): the values
# a literal of a scalar dual may take, by form; a literal row's value is drawn and
# mutated inside its dual's domain, and every value of a domain renders and reads
# back as a valid literal. An integer form takes its scalar's whole range: a tree's
# konst is an f64, exact for every u32 and i32, and the reader reads a literal of any value as that value. A range "lo..=hi"
# (a real range for an f32 slot: every finite f32, as a kernel's own constants are), a set "a,b,c", "-" for a dual no literal takes.
LITERAL_DOMAIN = {
    "real": "-3.4028234663852886e38..=3.4028234663852886e38",
    "int": "-2147483648..=2147483647",
    "index": "0..=4294967295",
    "count": "0..=4294967295",
    "bits": "0..=4294967295",
    "flag": "0,1",
    "wgdim": "1..=1024",
    "blk": "1,2,4,8",
    "unroll": "2,4,8",
    "lit64": "1..=64",
    "lit4096": "1..=4096",
    "wglen": "1..=8192",
    "xbound": "0..=4294967295",
    "payload": "1..=1048576",
    # The forms that had no literal (2026-10-19, the typed-GEP terminal plan): a constant of the dual is a real
    # value, and a tail of the dual needs a terminal. Each is its form's full range, to what the f32 konst of a
    # tree represents exactly (a u32 or i32 beyond 2^24 is carried as the nearest f32 here; the gene's C_T holds the
    # exact bit pattern).
    "q15_16": "-2147483648..=2147483647",
    "code4": "0..=3",
    "code8": "0..=7",
    "code16": "0..=15",
    "code1024": "0..=1023",
    "sign": "0,1",
    "hash32": "0..=4294967295",
    # A hash64 or wide fixed-point value is two lanes of its limb's range.
    "hash64": "0..=4294967295",
    "q15_16w": "-2147483648..=2147483647",
}


def literal_domain(slot, form):
    """A value slot's literal domain, a vector's or matrix's its scalar's (the renderer
    splats a literal over the lanes); f16's is its finite range."""
    scalar = slot
    if slot.startswith("vec") or slot.startswith("mat"):
        scalar = slot[slot.index("<") + 1:-1]
    # f16: its finite range (the module enables f16 when its text uses it).
    if scalar == "f16":
        return "-65504.0..=65504.0" if form in ("real", "opaque") else "-"
    if scalar not in ("f32", "i32", "u32", "bool"):
        return "-"
    if form == "opaque":
        return {"f32": LITERAL_DOMAIN["real"], "i32": "-2147483648..=2147483647", "u32": "0..=4294967295", "bool": "0,1"}[scalar]
    return LITERAL_DOMAIN.get(form, "-")


# The register kinds beyond the first eight (f v i b n u w and the square matrices x y z): a letter, the vector
# type, the form of its dual, and how many the kernel may name. The letter is the register's name (C0..C7) and the
# `reg.set_<letter>` / `reg.get_<letter>` rows' suffix. src/wgsl/render.rs holds the same list (EXTRA_REGISTERS);
# a test there checks every entry against the rows this file writes.
REG_KINDS = [
    ("C", "vec3<f32>", "real"),
    ("D", "vec2<i32>", "int"),
    ("E", "vec3<i32>", "int"),
    ("F", "vec4<i32>", "int"),
    ("G", "vec2<u32>", "index"),
    ("H", "vec3<u32>", "index"),
    ("J", "vec2<bool>", "flag"),
    ("K", "vec3<bool>", "flag"),
    ("L", "vec4<bool>", "flag"),
]
REG_KIND_COUNT = 16

# The registers of the earlier kinds beyond the counts they were written with: letter, dual, count written above, count now.
REG_EXTEND = [
    ("u", "reg<vec4<u32>>.bits", 8, 16),
    ("w", "reg<vec2<f32>>.real", 8, 16),
    ("x", "reg<mat2x2<f32>>.real", 2, 4),
    ("y", "reg<mat3x3<f32>>.real", 2, 4),
    ("z", "reg<mat4x4<f32>>.real", 2, 4),
]

# The arrays of one dual (the slot counts): class, dual, the number written above, the stem the further names are
# built from (stem + number), the number now. src/wgsl/render.rs holds the same list (WORKGROUP_ARRAYS and
# PRIVATE_ARRAYS); a test checks every name against the rows.
ARRAY_FAMILIES = [
    ("wg", "wgmem<f32>.real", 8, "f", 12),
    ("wg", "wgmem<vec4<f32>>.real", 4, "vq", 8),
    ("wg", "wgmem<vec2<f32>>.real", 4, "v2_", 8),
    ("wg", "wgmem<u32>.index", 12, "u_", 16),
    ("wg", "wgmem<atomic<u32>>.index", 4, "au_", 8),
    ("wg", "wgmem<atomic<i32>>.int", 2, "ai_", 4),
    ("wg", "wgmem<i32>.int", 2, "n_", 6),
    ("parr", "parr<f32>.real", 6, "f", 8),
    ("parr", "parr<vec4<f32>>.real", 4, "w", 6),
    ("parr", "parr<u32>.index", 3, "i", 5),
    ("parr", "parr<bool>.flag", 2, "m", 4),
    ("parr", "parr<vec2<f32>>.real", 2, "q", 4),
    ("parr", "parr64<u32>.index", 2, "ua", 4),
    ("parr", "parr64<vec4<f32>>.real", 2, "vq", 4),
    ("parr", "parr128<u32>.index", 2, "ux", 4),
]


def build_types():
    rows = []  # (ref, dual, slot, form, lanes, fallback, fallback_bits, fallback_noop, fallback_elem, meaning, literal)

    def add(slot, form, lanes, meaning):
        noop = int(form == "store")
        elem = int(slot == "array<T>")
        rows.append((f"T{len(rows) + 1:03d}", f"{slot}.{form}", slot, form, lanes, FALLBACK[form], FALLBACK_BITS[form], noop, elem, meaning, literal_domain(slot, form)))

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
    # ---- workgroup arrays as knobs (2026-10-12) ----
    add("wgmem<vec2<f32>>", "real", 0, "a workgroup-memory array of vec2<f32>: v2a..v2d")
    add("wgmem<u32>", "index", 0, "a workgroup-memory array of u32 indices: ua..ud (2026-10-16: 73 of 1,798 saved model kernels kept index lists in workgroup memory)")
    add("u32", "wgv2index", 1, FORMS["wgv2index"][0])
    add("u32", "wglen", 1, FORMS["wglen"][0])
    add("wg", "decl", 0, FORMS["decl"][0])
    add("wg", "decls", 0, FORMS["decls"][0])
    # ---- integer atomics (2026-10-12) ----
    add("wgmem<atomic<u32>>", "index", 0, "a workgroup-memory array of atomic<u32>: au_a, au_b (au_c, au_d since 2026-10-19)")
    add("wgmem<atomic<i32>>", "int", 0, "a workgroup-memory array of atomic<i32>: ai_a, ai_b")
    add("u32", "wgaindex", 1, FORMS["wgaindex"][0])
    # ---- i32 registers (2026-10-12, WGSL_COVERAGE.md gap 6) ----
    add("reg<i32>", "int", 1, "a named i32 register: n0..n15")
    # ---- loop bodies that break (2026-10-12, gap 7) ----
    for d in (1, 2, 3, 4, 5):
        add("block", f"B{d}", 0, FORMS[f"B{d}"][0])
    # ---- vec4<u32> registers and vec2 private arrays (2026-10-12, gap 8) ----
    add("reg<vec4<u32>>", "bits", 4, "a named vec4<u32> register: u0..u7 (subgroupBallot's result)")
    add("parr<vec2<f32>>", "real", 0, "a private array of 16 vec2<f32>: pq, pr")
    # ---- matrix registers (2026-10-12, gap 9) ----
    for n in (2, 3, 4):
        add(f"reg<mat{n}x{n}<f32>>", "real", n * n, f"a named mat{n}x{n}<f32> register")
    # ---- the index of a 16-element private array (2026-10-12, legality item 4) ----
    add("u32", "pindex16", 1, FORMS["pindex16"][0])
    # ---- uniform values and the bound of a loop that may hold a barrier (2026-10-12, legality item 5) ----
    add("u32", "uniform", 1, FORMS["uniform"][0])
    add("u32", "boundu", 1, FORMS["boundu"][0])
    # ---- no wrap in a loop bound (2026-10-12) ----
    add("u32", "uniformw", 1, FORMS["uniformw"][0])
    add("u32", "xbound", 1, FORMS["xbound"][0])
    # ---- no wrap anywhere (2026-10-12, Codex's review) ----
    add("u32", "indexw", 1, FORMS["indexw"][0])
    # ---- the origins an offset row takes (2026-10-12) ----
    for form in ("rowbase", "colbase", "rowbases", "colbases", "rowbaseg", "colbaseg"):
        add("u32", form, 1, FORMS[form][0])
    # ---- the symmetric half (2026-10-14) ----
    add("u32", "mirror", 1, FORMS["mirror"][0])
    # ---- the workgroup put (2026-10-14) ----
    add("u32", "wgown", 1, FORMS["wgown"][0])
    # ---- several entry points in one module (2026-10-17) ----
    add("kernel", "module", 0, FORMS["module"][0])
    # ---- a workgroup-uniform predicate (2026-10-17) ----
    add("bool", "uflag", 1, FORMS["uflag"][0])
    # ---- 64-element u32 private arrays (2026-10-17) ----
    add("parr64<u32>", "index", 0, "a private array of 64 u32: pua, puc (a list of ids beside 64 scores)")
    # ---- vec2<f32> registers (2026-10-17) ----
    add("reg<vec2<f32>>", "real", 2, "a named vec2<f32> register: w0..w7 (a helper's two sums returned together)")
    # ---- 64-element vec4<f32> private arrays (2026-10-17) ----
    add("parr64<vec4<f32>>", "real", 0, "a private array of 64 vec4<f32>: pvq, pvr (a query's 32 vec4 lanes held in registers)")
    # A knob reference's payload (2026-10-17): the dual of a literal.p_* row's constant.
    add("u32", "payload", 1, FORMS["payload"][0])
    # Workgroup arrays of i32 (2026-10-19, run 4's seeds 14 and 29: quantised int8 values staged in i32).
    add("wgmem<i32>", "int", 0, "a workgroup-memory array of i32: na, nb")
    # 128-element u32 private arrays and their index (2026-10-19, run 4's seed10 caches 100 values a thread).
    add("parr128<u32>", "index", 0, "a private array of 128 u32: pux, puy")
    add("u32", "pindex128", 1, FORMS["pindex128"][0])
    # Registers of the other vector types (2026-10-19, the harvested corpus: 2,700 of 4,100 reads that failed held a
    # vec2/vec3/vec4 of i32, u32, f32 or bool in a local or returned one): see REG_KINDS.
    for letter, slot, form in REG_KINDS:
        add(f"reg<{slot}>", form, int(slot[3]), f"a named {slot} register: {letter}0..{letter}{REG_KIND_COUNT - 1}")
    return rows


# ------------------------------------------------------------ functions ----
# Each entry: (class, instance, naga semantic, [in duals], out dual, instantiation note)
# "<S.real>" stands for every real dual over scalar/vector lanes (f32, f16, vecL<..>).
FUNCTIONS = []


# The private arrays of 16 elements (vec4, u32, bool, vec2) take an index of their own dual,
# u32.pindex16 (i & 15u); the f32 arrays of 64 take u32.pindex (i & 63u). One shared dual let
# safe.pidx64 index a 16-element array (the legality checker: 369 of 1,000 random genes, 2026-10-12).
SIXTEEN = ("parr<vec4<f32>>.real", "parr<u32>.index", "parr<bool>.flag", "parr<vec2<f32>>.real")


ORIGIN_OF = {"u32.threadrow": "u32.rowbase", "u32.threadcol": "u32.colbase", "u32.threadrows": "u32.rowbases", "u32.threadcols": "u32.colbases", "u32.threadrowg": "u32.rowbaseg", "u32.threadcolg": "u32.colbaseg"}


def f(cls, inst, naga, ins, out, note=""):
    if ins and ins[0] in SIXTEEN:
        ins = ["u32.pindex16" if x == "u32.pindex" else x for x in ins]
    if (cls, inst) == ("safe", "pidx16"):
        out = "u32.pindex16"
    # A loop whose body may hold a barrier (an expression-bounded tile loop at an L rung, an item loop
    # bounded by an expression) takes a uniform bound: every thread runs it the same number of times, so
    # every thread reaches its barriers (the legality checker: 87 of 1,000 random genes had a barrier in a
    # loop bounded by a thread's or a subgroup's value, 2026-10-12). Loops that break and loops at C and O
    # rungs hold no barrier and keep any bound.
    # An offset row takes an origin (2026-10-12): u32.threadrow* as its first input is the origin's dual.
    if cls == "safe" and (inst.startswith("row_plus") or inst.startswith("col_plus")) and ins:
        ins = [ORIGIN_OF.get(ins[0], ins[0])] + ins[1:]
    if cls == "loop" and out.startswith("stmt.L") and "u32.boundx" in ins and "_brk_" not in inst:
        ins = ["u32.boundu" if x == "u32.boundx" else x for x in ins]
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
    if op == "sub":
        # NO WRAP ANYWHERE (2026-10-12, Codex's review): a raw difference may wrap below zero, so its value is
        # u32.indexw, which no row takes; a difference is index.sub_floor (the reader reads x - y as it).
        f("index", op, naga, ["u32.index", "u32.index"], "u32.indexw", "a raw difference: it may wrap, no row takes it (index.sub_floor is the kingdom's difference)")
        continue
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
f("convert", "f32_to_f16", "As", ["<f32.real>"], "<f16.real>", "per lane")
f("convert", "f16_to_f32", "As", ["<f16.real>"], "<f32.real>", "per lane")
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
for op, naga in [("count_ones", "Math::CountOneBits"), ("leading_zeros", "Math::CountLeadingZeros"), ("trailing_zeros", "Math::CountTrailingZeros"), ("find_lsb", "Math::FirstTrailingBit"), ("find_msb", "Math::FirstLeadingBit")]:
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
f("literal", "wgdim", "Literal", [], "u32.wgdim", "a workgroup dimension, any integer 1..=1,024 (from {4, 8, 16, 32} until 2026-10-12)")
f("literal", "blk", "Literal", [], "u32.blk", "a block factor from {1, 2, 4, 8} (8 since 2026-10-11: a model child's 4 x 8 block)")
f("literal", "unroll", "Literal", [], "u32.unroll", "an unroll factor from {2, 4, 8}")
f("kernel", "entry", "composite", ["grid.grid", "block.block"], "kernel.kernel", "the module: bindings, registers, workgroup arrays, @workgroup_size(x, y, 1) fn main")
f("grid", "of", "composite", ["u32.wgdim", "u32.wgdim", "u32.blk", "u32.blk"], "grid.grid", "arity 4: workgroup x, y and the cells per thread bx, by; the driver dispatches ceil(d / (x bx)) x ceil(rows / (y by)); y is cut so x y <= 1,024 threads")
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
# A left shift may wrap (2026-10-12, Codex's review: no wrap anywhere): u32.indexw, which no row takes; the
# reader reads x << k (a literal k) as x * 2^k.
f("index", "shl", "composite", ["u32.index", "u32.index"], "u32.indexw", "i << (s & 31u): it may wrap, no row takes it")
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
f("load", "phi_safe", "composite", ["u32.index"], "f32.real", "phi[i]: any index, a read past the buffer is the buffer's LAST element (the device clamps the index, naga Restrict)")
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
f("bound", "of_index", "composite", ["u32.xbound"], "u32.boundx", "an expression loop bound (no wrapping arithmetic, 2026-10-12; any index expression before), capped at MAX_LOOP_TRIP = 2^24 = 16,777,216 (the kingdom's declared maximum trip count; legality, 2026-10-12): min(e, 16777216u)")
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
    f("loop", "tilex_strided_" + r, "composite", ["u32.boundx", "u32.lit4096", "block." + body], "stmt." + r, "arity 3: the same, stepped by a literal in 1..=4096 (64 until 2026-10-16: 62 of the 1,798 saved model kernels stepped wider)")

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
    f("loop", "tilex_strided_" + r, "composite", ["u32.boundx", "u32.lit4096", "block." + body], "stmt." + r, "arity 3: the same, stepped by a literal in 1..=4096 (64 until 2026-10-16: 62 of the 1,798 saved model kernels stepped wider)")
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
        f("loop", "tilex_strided_" + r, "composite", ["u32.boundx", "u32.lit4096", body], "stmt." + r, "arity 3: bounded by an expression, stepped by a literal in 1..=4096 (64 until 2026-10-16)")

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

# ---- workgroup arrays as knobs (2026-10-12) ----
# Four arrays of each element type (f32 a..d, vec2 v2a..v2d, vec4 va..vd); a kernel's declarations name the
# arrays it has and their lengths (a cons list on the grid); an array used but not declared gets its default
# length (f32 2,048, vec2 1,024, vec4 512); the renderer halves the largest array until the total fits 32 KB.
for n in ("c", "d"):
    f("wg", n, "composite", [], "wgmem<f32>.real", f"workgroup array wg_{n} of f32")
for n in ("vc", "vd"):
    f("wg", n, "composite", [], "wgmem<vec4<f32>>.real", f"workgroup array wg_{n} of vec4<f32>")
for n in ("v2a", "v2b", "v2c", "v2d"):
    f("wg", n, "composite", [], "wgmem<vec2<f32>>.real", f"workgroup array wg_{n} of vec2<f32>")
f("safe", "wgv2_of", "composite", ["u32.index"], "u32.wgv2index", "i masked into its vec2 array, in range by construction")
f("wg", "get_v2_safe", "composite", ["wgmem<vec2<f32>>.real", "u32.wgv2index"], "vec2<f32>.real", "a[i], in range by its type")
for r in RUNGS + RUNGS5:
    f("wg", "set_v2_" + r, "composite", ["wgmem<vec2<f32>>.real", "u32.wgv2index", "vec2<f32>.real"], "stmt." + r, "arity 3: a[i] = v")
f("literal", "wglen", "Literal", [], "u32.wglen", "a workgroup array's length, 1..=8,192 elements")
f("wg", "decl_f", "composite", ["wgmem<f32>.real", "u32.wglen"], "wg.decl", "var<workgroup> a: array<f32, n>")
f("wg", "decl_v2", "composite", ["wgmem<vec2<f32>>.real", "u32.wglen"], "wg.decl", "var<workgroup> a: array<vec2<f32>, n>")
f("wg", "decl_v4", "composite", ["wgmem<vec4<f32>>.real", "u32.wglen"], "wg.decl", "var<workgroup> a: array<vec4<f32>, n>")
# ---- workgroup arrays of indices (2026-10-16): u32 lists in workgroup memory (candidate lists,
# bucket offsets, staged neighbour ids), four arrays ua..ud of 2,048 by default, indexed as the f32 arrays are ----
for n in ("ua", "ub", "uc", "ud"):
    f("wg", n, "composite", [], "wgmem<u32>.index", f"workgroup array wg_{n} of u32")
f("wg", "get_u_safe", "composite", ["wgmem<u32>.index", "u32.wgindex"], "u32.index", "a[i], in range by its type")
for r in RUNGS + RUNGS5:
    f("wg", "set_u_" + r, "composite", ["wgmem<u32>.index", "u32.wgindex", "u32.index"], "stmt." + r, "arity 3: a[i] = v")
f("wg", "decl_u", "composite", ["wgmem<u32>.index", "u32.wglen"], "wg.decl", "var<workgroup> a: array<u32, n>")
f("wg", "decls_one", "composite", ["wg.decl"], "wg.decls", "one declaration")
f("wg", "decls_cons", "composite", ["wg.decl", "wg.decls"], "wg.decls", "a declaration, then the rest")
f("grid", "with_wg", "composite", ["grid.grid", "wg.decls"], "grid.grid", "a grid and the kernel's workgroup arrays (the arrays it declares and their lengths)")

# ---- full unrolling (2026-10-12) ----
# A literal-count loop written out as straight-line code: n copies of the body, the loop's counter a
# literal in each (the matmul article's fully unrolled 8 x 8); the same signature as the tile loop it
# unrolls, so tile or unroll is one symbol of the gene.
for r, body in (("L0", "L2"), ("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("L4", "L5"), ("C0", "C1"), ("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5")):
    f("loop", "unroll_" + r, "composite", ["u32.lit64", "block." + body], "stmt." + r, "the body n times, straight-line, its counter a literal in each copy")
for d in range(len(EPILOGUE) - 1):
    f("loop", "unroll_" + EPILOGUE[d], "composite", ["u32.lit64", "block." + EPILOGUE[d + 1]], "stmt." + EPILOGUE[d], "the epilogue's loop body n times, straight-line")

# ---- tree reduction in workgroup memory (2026-10-12) ----
# One block: a[0..n) reduced into a[0] by stride halving, a barrier before and after every step, in uniform
# control flow (so only at the L rungs, where a barrier stands); knobs: the array (its element type), n and
# the operation. Written as its log2(n) steps (`if (lidx < s) { a[lidx] = a[lidx] op a[lidx + s]; }`), each
# index masked into the array.
for op in ("add", "max", "min"):
    for r in ("L0", "L1", "L2", "L3", "L4", "L5"):
        for arr in ("wgmem<f32>.real", "wgmem<vec2<f32>>.real", "wgmem<vec4<f32>>.real"):
            f("wg", f"reduce_{op}_{r}", "composite", [arr, "u32.lit4096"], "stmt." + r, f"a[0..n) reduced into a[0] by {op}: stride halving, barriers in uniform control flow")

# ---- integer atomics in workgroup memory (2026-10-12) ----
# Arrays of atomic<u32> (au_a, au_b) and atomic<i32> (ai_a, ai_b), declared on the grid like the others; the
# read-modify-write built-ins return the old value, atomicLoad reads, atomicStore writes, `_ = atomicX(...)`
# keeps the effect only. NOT spelled: atomicCompareExchangeWeak, and the float add built on it (the
# compare-exchange loop on the value's bits): naga 0.20's Metal backend refuses it ("MSL:
# FeatureNotImplemented(atomic CompareExchange)", measured on the M3 Max 2026-10-12), so no gene with it would
# compile (verified again 2026-10-17 in naga 0.20.0's back/msl/mod.rs, AtomicFunction::to_msl). Storage
# atomics are not rows of this table: they are the per-kernel rows of an interface that declares an atomic
# binding (render.rs Role::interface): a scalar `atomic<u32>` its read-modify-writes `atomic.<name>_<fun>(v)`;
# an array of atomics (`array<atomic<u32>>`, v24's index arena, 2026-10-17) `load.<name>(i)` (atomicLoad),
# `store.<name>_<rung>(i, v)` (atomicStore) and `atomic.<name>_<fun>(i, v)`, every index as written (the bounds
# policy decides an access past the binding; a clamp to its last element changed what kernels computed).
for n in ("au_a", "au_b"):
    f("wg", n, "composite", [], "wgmem<atomic<u32>>.index", f"workgroup array wg_{n} of atomic<u32>")
for n in ("ai_a", "ai_b"):
    f("wg", n, "composite", [], "wgmem<atomic<i32>>.int", f"workgroup array wg_{n} of atomic<i32>")
f("safe", "wga_of", "composite", ["u32.index"], "u32.wgaindex", "i masked into its atomic array, in range by construction")
f("wg", "decl_au", "composite", ["wgmem<atomic<u32>>.index", "u32.wglen"], "wg.decl", "var<workgroup> a: array<atomic<u32>, n>")
f("wg", "decl_ai", "composite", ["wgmem<atomic<i32>>.int", "u32.wglen"], "wg.decl", "var<workgroup> a: array<atomic<i32>, n>")
for op, wgsl in (("add", "atomicAdd"), ("sub", "atomicSub"), ("max", "atomicMax"), ("min", "atomicMin"), ("and", "atomicAnd"), ("or", "atomicOr"), ("xor", "atomicXor"), ("exchange", "atomicExchange")):
    f("atomic", op + "_u", "composite", ["wgmem<atomic<u32>>.index", "u32.wgaindex", "u32.index"], "u32.index", f"{wgsl}(&a[i], v): the old value")
    f("atomic", op + "_i", "composite", ["wgmem<atomic<i32>>.int", "u32.wgaindex", "i32.int"], "i32.int", f"{wgsl}(&a[i], v): the old value")
f("atomic", "load_u", "composite", ["wgmem<atomic<u32>>.index", "u32.wgaindex"], "u32.index", "atomicLoad(&a[i])")
f("atomic", "load_i", "composite", ["wgmem<atomic<i32>>.int", "u32.wgaindex"], "i32.int", "atomicLoad(&a[i])")
for r in RUNGS + RUNGS5:
    f("atomic", "store_u_" + r, "composite", ["wgmem<atomic<u32>>.index", "u32.wgaindex", "u32.index"], "stmt." + r, "arity 3: atomicStore(&a[i], v)")
    f("atomic", "store_i_" + r, "composite", ["wgmem<atomic<i32>>.int", "u32.wgaindex", "i32.int"], "stmt." + r, "arity 3: atomicStore(&a[i], v)")
    f("stmt", "discard_u_" + r, "composite", ["u32.index"], "stmt." + r, "_ = e: an atomic's effect, its old value unused")
    f("stmt", "discard_i_" + r, "composite", ["i32.int"], "stmt." + r, "_ = e: an atomic's effect, its old value unused")

# ---- subgroups (2026-10-12; the M3 Max grants SUBGROUP and SUBGROUP_BARRIER, sizes 4..64) ----
# The built-in values, the collective operations (reduce, inclusive and exclusive scan), the gathers
# (broadcast with a literal lane, broadcast-first, shuffle, shuffle up / down / xor) and subgroupBarrier (at
# the L rungs). naga 0.20 reads them WITHOUT `enable subgroups;` (it cannot parse an enable directive); the
# device must ask for the feature (fuller's oracle device does). subgroupBallot is not spelled: its
# vec4<u32> result has no register.
f("builtin", "subgroup_invocation_id", "FunctionArgument(builtin)", [], "u32.index", "the thread's lane in its subgroup")
f("builtin", "subgroup_size", "FunctionArgument(builtin)", [], "u32.count", "the subgroup's width (4..64 on the M3 Max)")
f("builtin", "num_subgroups", "FunctionArgument(builtin)", [], "u32.count", "the subgroups in the workgroup")
f("builtin", "subgroup_id", "FunctionArgument(builtin)", [], "u32.index", "the thread's subgroup in its workgroup")
for op, wgsl in (("add", "subgroupAdd"), ("mul", "subgroupMul"), ("min", "subgroupMin"), ("max", "subgroupMax"), ("inclusive_add", "subgroupInclusiveAdd"), ("inclusive_mul", "subgroupInclusiveMul"), ("exclusive_add", "subgroupExclusiveAdd"), ("exclusive_mul", "subgroupExclusiveMul")):
    f("subgroup", op, "SubgroupCollectiveOperation", ["<S.real>"], "<S.real>", f"{wgsl}(x) over the subgroup's active lanes")
    f("subgroup", op, "SubgroupCollectiveOperation", ["u32.index"], "u32.index", f"{wgsl}(x) over the subgroup's active lanes")
for op, wgsl in (("and", "subgroupAnd"), ("or", "subgroupOr"), ("xor", "subgroupXor")):
    f("subgroup", op, "SubgroupCollectiveOperation", ["u32.bits"], "u32.bits", f"{wgsl}(x)")
for op, wgsl in (("all", "subgroupAll"), ("any", "subgroupAny")):
    f("subgroup", op, "SubgroupCollectiveOperation", ["bool.flag"], "bool.flag", f"{wgsl}(b)")
for value in ("<S.real>", "u32.index"):
    f("subgroup", "broadcast_first", "SubgroupGather", [value], value, "subgroupBroadcastFirst(x)")
    f("subgroup", "broadcast", "SubgroupGather", [value, "u32.lit64"], value, "subgroupBroadcast(x, lane): the lane a literal")
    for op, wgsl in (("shuffle", "subgroupShuffle"), ("shuffle_down", "subgroupShuffleDown"), ("shuffle_up", "subgroupShuffleUp"), ("shuffle_xor", "subgroupShuffleXor")):
        f("subgroup", op, "SubgroupGather", [value, "u32.index"], value, f"{wgsl}(x, i)")
for r in ("L0", "L1", "L2", "L3", "L4", "L5"):
    f("stmt", "subgroup_barrier_" + r, "composite", [], "stmt." + r, "subgroupBarrier()")

# ---- the layout knob (2026-10-12): which output axis runs on the fast thread index ----
# grid.of puts the output's columns on thread x (the fast index: neighbouring lanes read and write
# neighbouring columns) and its rows on y. grid.of_rows_x puts the rows on x and the columns on y: the
# same workgroup and block, the driver dispatches ceil(rows / (x by)) x ceil(d / (y bx)), and every row
# and column of the gene renders on the other thread axis. One symbol of the gene chooses the coalescing.
f("grid", "of_rows_x", "composite", ["u32.wgdim", "u32.wgdim", "u32.blk", "u32.blk"], "grid.grid", "arity 4: grid.of with the output's rows on thread x (the fast index) and its columns on y; the header says `rows-x`")


# ---- the u32 forms and the bit, dot and modulo overloads (2026-10-12, WGSL_COVERAGE.md gaps 2 and 3) ----
# A u32 is an index, a count or a bit field by its meaning; WGSL has one u32. The crossings are explicit
# rows (the convert class holds the only crossings): an index or a count read as a bit field and a bit
# field read as an index, per lane. The bit-count functions, reverseBits and extractBits over vectors;
# insertBits as naga's one call (arity 4); dot over integer vectors; % over reals (x - y * trunc(x / y)).
f("convert", "index_to_bits", "As", ["<u32.index>"], "<u32.bits>", "the same u32 lanes read as a bit field")
f("convert", "bits_to_index", "As", ["<u32.bits>"], "<u32.index>", "the same u32 lanes read as an index")
f("convert", "count_to_bits", "As", ["<u32.count>"], "<u32.bits>", "the same u32 lanes read as a bit field")
for L in (2, 3, 4):
    f("index", "of_count", "composite", [f"vec{L}<u32>.count"], f"vec{L}<u32>.index", "per lane")
    for op, naga in (("count_ones", "CountOneBits"), ("leading_zeros", "CountLeadingZeros"), ("trailing_zeros", "CountTrailingZeros"), ("find_lsb", "FirstTrailingBit"), ("find_msb", "FirstLeadingBit")):
        f("bits", op, f"Math::{naga}", [f"vec{L}<u32>.bits"], f"vec{L}<u32>.count", "per lane")
    f("bits", "reverse", "Math::ReverseBits", [f"vec{L}<u32>.bits"], f"vec{L}<u32>.bits", "per lane")
    f("bits", "extract", "Math::ExtractBits", [f"vec{L}<u32>.bits", "u32.index", "u32.count"], f"vec{L}<u32>.bits", "arity 3, per lane")
f("bits", "insert", "Math::InsertBits", ["<u32.bits>", "<u32.bits>", "u32.index", "u32.count"], "<u32.bits>", "arity 4: insertBits(e, newbits, offset, count)")
f("geom", "dot", "Math::Dot", ["<vecL<u32>.index>", "<vecL<u32>.index>"], "u32.index", "integer dot product (wraps)")
f("geom", "dot", "Math::Dot", ["<vecL<i32>.int>", "<vecL<i32>.int>"], "i32.int", "integer dot product (wraps)")
f("arith", "mod", "Binary::Modulo", ["<S.real>", "<S.real>"], "<S.real>", "WGSL's % on reals: x - y * trunc(x / y)")


# ---- storageBarrier (2026-10-12, WGSL_COVERAGE.md gap 4): its own statement, read as workgroupBarrier before ----
for r in ("L0", "L1", "L2", "L3", "L4", "L5"):
    f("stmt", "storage_barrier_" + r, "composite", [], "stmt." + r, "storageBarrier(): storage-buffer writes visible to the workgroup")


# ---- i32 (2026-10-12, WGSL_COVERAGE.md gap 6): registers, the crossings to u32 and f32, and the integer
# rows WGSL has on i32 (clamp, sign, the bit operations and counts, shifts, subgroup operations) ----
for i in range(16):
    f("reg", f"n{i}", "composite", [], "reg<i32>.int", "an i32 register")
f("reg", "get_n", "composite", ["reg<i32>.int"], "i32.int", "the register's value")
for r in RUNGS + RUNGS5:
    f("reg", "set_n_" + r, "composite", ["reg<i32>.int", "i32.int"], "stmt." + r, "")
f("convert", "index_to_int", "As", ["<u32.index>"], "<i32.int>", "i32(u): the same bits read as two's complement")
f("convert", "int_to_index", "As", ["<i32.int>"], "<u32.index>", "u32(i): the same bits read unsigned (index.of_int floors at zero instead)")
f("convert", "bitcast_int_bits", "As(bitcast)", ["<i32.int>"], "<u32.bits>", "")
f("convert", "bitcast_bits_int", "As(bitcast)", ["<u32.bits>"], "<i32.int>", "")
f("convert", "bitcast_real_int", "As(bitcast)", ["<f32.real>"], "<i32.int>", "")
f("convert", "bitcast_int_real", "As(bitcast)", ["<i32.int>"], "<f32.real>", "")
for L in (2, 3, 4):
    f("convert", "f32_to_int", "As", [f"vec{L}<f32>.real"], f"vec{L}<i32>.int", "truncating, per lane")
    f("convert", "int_to_f32", "As", [f"vec{L}<i32>.int"], f"vec{L}<f32>.real", "per lane")
f("int", "clamp", "Math::Clamp", ["<i32.int>", "<i32.int>", "<i32.int>"], "<i32.int>", "arity 3")
f("int", "sign", "Math::Sign", ["<i32.int>"], "<i32.int>", "")
for op, naga in (("and", "Binary::And"), ("or", "Binary::InclusiveOr"), ("xor", "Binary::ExclusiveOr")):
    f("int", op, naga, ["<i32.int>", "<i32.int>"], "<i32.int>", "lanes equal")
f("int", "not", "Unary::BitwiseNot", ["<i32.int>"], "<i32.int>", "")
f("int", "shl", "Binary::ShiftLeft", ["i32.int", "u32.count"], "i32.int", "")
f("int", "shr", "Binary::ShiftRight", ["i32.int", "u32.count"], "i32.int", "arithmetic: the sign bit fills")
for op, naga in (("count_ones", "CountOneBits"), ("leading_zeros", "CountLeadingZeros"), ("trailing_zeros", "CountTrailingZeros"), ("find_lsb", "FirstTrailingBit"), ("find_msb", "FirstLeadingBit")):
    f("int", op, f"Math::{naga}", ["<i32.int>"], "<i32.int>", "WGSL returns i32 for i32")
f("int", "reverse", "Math::ReverseBits", ["<i32.int>"], "<i32.int>", "")
f("int", "extract", "Math::ExtractBits", ["<i32.int>", "u32.index", "u32.count"], "<i32.int>", "arity 3, sign-extending")
f("int", "insert", "Math::InsertBits", ["<i32.int>", "<i32.int>", "u32.index", "u32.count"], "<i32.int>", "arity 4")
for op, wgsl in (("add", "subgroupAdd"), ("mul", "subgroupMul"), ("min", "subgroupMin"), ("max", "subgroupMax"), ("inclusive_add", "subgroupInclusiveAdd"), ("inclusive_mul", "subgroupInclusiveMul"), ("exclusive_add", "subgroupExclusiveAdd"), ("exclusive_mul", "subgroupExclusiveMul"), ("and", "subgroupAnd"), ("or", "subgroupOr"), ("xor", "subgroupXor")):
    f("subgroup", op, "SubgroupCollectiveOperation", ["i32.int"], "i32.int", f"{wgsl}(x) on i32")
f("subgroup", "broadcast_first", "SubgroupGather", ["i32.int"], "i32.int", "subgroupBroadcastFirst(x)")
f("subgroup", "broadcast", "SubgroupGather", ["i32.int", "u32.lit64"], "i32.int", "subgroupBroadcast(x, lane): the lane a literal")
for op, wgsl in (("shuffle", "subgroupShuffle"), ("shuffle_down", "subgroupShuffleDown"), ("shuffle_up", "subgroupShuffleUp"), ("shuffle_xor", "subgroupShuffleXor")):
    f("subgroup", op, "SubgroupGather", ["i32.int", "u32.index"], "i32.int", f"{wgsl}(x, i)")


# ---- break and continue in loops (2026-10-12, WGSL_COVERAGE.md gap 7) ----
# A loop that breaks: a tile loop whose body is a B block (conditional statements, the guard, conditional
# statements), at every rung a tile loop has; its body holds no barrier, so the loop is safe at an L rung.
# A continue at an L or C rung inside a loop: the guard, then the rest of the block at the conditional
# rung of its depth (no barrier after it; every thread still reaches the next iteration).
for d in (1, 2, 3, 4, 5):
    f("block", f"brk_B{d}", "composite", [f"block.C{d}", "bool.flag", f"block.C{d}"], f"block.B{d}", "arity 3: before; if (c) { break; }; after")
for r, b in (("L0", 2), ("L1", 2), ("L2", 3), ("L3", 4), ("L4", 5), ("C0", 1), ("C1", 2), ("C2", 3), ("C3", 4), ("C4", 5)):
    f("loop", "tile_brk_" + r, "composite", ["u32.lit64", f"block.B{b}"], "stmt." + r, "for (k = 0u; k < n; k++) { body that breaks }")
    f("loop", "tilex_brk_" + r, "composite", ["u32.boundx", f"block.B{b}"], "stmt." + r, "for (k = 0u; k < (expr); k++) { body that breaks }")
for r in ("L1", "L2", "L3", "L4", "L5", "C1", "C2", "C3", "C4", "C5"):
    f("block", "cont_" + r, "composite", ["bool.flag", "block.C" + r[1]], "block." + r, "if (c) { continue; } then the rest, conditional")


# ---- vectors (2026-10-12, WGSL_COVERAGE.md gap 8): a thread id as a vector operand, and bitcasts of whole
# vectors ----
for b in ("gid", "lid", "wid"):
    f("index", "of_" + b, "composite", [f"vec3<u32>.{b}"], "vec3<u32>.index", f"the thread id {b} used as a plain vector of indices")
for L in (2, 3, 4):
    f("convert", "bitcast_f32_bits", "As(bitcast)", [f"vec{L}<f32>.real"], f"vec{L}<u32>.bits", "per lane")
    f("convert", "bitcast_bits_f32", "As(bitcast)", [f"vec{L}<u32>.bits"], f"vec{L}<f32>.real", "per lane")


# ---- vec4<u32> registers (subgroupBallot) and vec2 private arrays (2026-10-12, gap 8) ----
for i in range(8):
    f("reg", f"u{i}", "composite", [], "reg<vec4<u32>>.bits", "a vec4<u32> register")
f("reg", "get_u", "composite", ["reg<vec4<u32>>.bits"], "vec4<u32>.bits", "the register's value")
for r in RUNGS + RUNGS5:
    f("reg", "set_u_" + r, "composite", ["reg<vec4<u32>>.bits", "vec4<u32>.bits"], "stmt." + r, "")
f("subgroup", "ballot", "SubgroupBallot", ["bool.flag"], "vec4<u32>.bits", "subgroupBallot(b): a bit per active lane where b holds")
f("parr", "q", "composite", [], "parr<vec2<f32>>.real", "private array pq of 16 vec2<f32>")
f("parr", "r", "composite", [], "parr<vec2<f32>>.real", "private array pr of 16 vec2<f32>")
f("parr", "get_t", "composite", ["parr<vec2<f32>>.real", "u32.pindex"], "vec2<f32>.real", "pq[i]")
for r in RUNGS + RUNGS5:
    f("parr", "set_t_" + r, "composite", ["parr<vec2<f32>>.real", "u32.pindex", "vec2<f32>.real"], "stmt." + r, "arity 3: pq[i] = v;")


# ---- matrices (2026-10-12, WGSL_COVERAGE.md gap 9): composition from 3 and 4 columns, column access, the
# products and sums WGSL has on matrices, and square matrix registers ----
f("shape", "compose_mat3", "Compose", ["<vecR<f32>.real>"] * 3, "<mat3xR<f32>.real>", "arity 3: three columns")
f("shape", "compose_mat4", "Compose", ["<vecR<f32>.real>"] * 4, "<mat4xR<f32>.real>", "arity 4: four columns")
for C in (2, 3, 4):
    for R in (2, 3, 4):
        for i in range(C):
            f("shape", f"col_{i}", "AccessIndex", [f"mat{C}x{R}<f32>.real"], f"vec{R}<f32>.real", f"m[{i}]: a column")
f("shape", "col_at", "composite", ["<matCxR<f32>.real>", "u32.index"], "<vecR<f32>.real>", "m[min(i, C - 1)]: a column chosen at run time")
f("arith", "mul_vec_mat", "Binary::Multiply", ["<vecR<f32>.real>", "<matCxR<f32>.real>"], "<vecC<f32>.real>", "v * m (a row vector)")
f("arith", "mul_mat_scalar", "Binary::Multiply", ["<matCxR<f32>.real>", "f32.real"], "<matCxR<f32>.real>", "")
f("arith", "mul_scalar_mat", "Binary::Multiply", ["f32.real", "<matCxR<f32>.real>"], "<matCxR<f32>.real>", "")
f("arith", "add_mat", "Binary::Add", ["<matCxR<f32>.real>", "<matCxR<f32>.real>"], "<matCxR<f32>.real>", "")
f("arith", "sub_mat", "Binary::Subtract", ["<matCxR<f32>.real>", "<matCxR<f32>.real>"], "<matCxR<f32>.real>", "")
f("arith", "neg_mat", "Unary::Negate", ["<matCxR<f32>.real>"], "<matCxR<f32>.real>", "")
for n, k in ((2, "x"), (3, "y"), (4, "z")):
    for i in range(2):
        f("reg", f"{k}{i}", "composite", [], f"reg<mat{n}x{n}<f32>>.real", f"a mat{n}x{n}<f32> register")
    f("reg", f"get_{k}", "composite", [f"reg<mat{n}x{n}<f32>>.real"], f"mat{n}x{n}<f32>.real", "the register's value")
    for r in RUNGS + RUNGS5:
        f("reg", f"set_{k}_" + r, "composite", [f"reg<mat{n}x{n}<f32>>.real", f"mat{n}x{n}<f32>.real"], "stmt." + r, "")


# ---- modf / frexp results and the bindings' lengths (2026-10-12, WGSL_COVERAGE.md gap 10) ----
# modf and frexp return a struct; the kingdom spells each field as its own row. arrayLength of the Gram
# role's runtime-sized bindings, phi and h.
f("arith", "modf_fract", "composite", ["<S.real>"], "<S.real>", "modf(x).fract")
f("arith", "modf_whole", "composite", ["<S.real>"], "<S.real>", "modf(x).whole")
f("arith", "frexp_fract", "composite", ["<S.real>"], "<S.real>", "frexp(x).fract")
f("arith", "frexp_exp", "composite", ["<f32.real>"], "<i32.int>", "frexp(x).exp")
f("load", "phi_len", "ArrayLength", [], "u32.count", "arrayLength(&phi)")
f("load", "h_len", "ArrayLength", [], "u32.count", "arrayLength(&h)")

# ---- kernel parameters (2026-10-12, docs/PLAN_constrained_mutation.md task 0, sections 2.3 and 3.1) ----
# A knob is a first-class, closed parameter of the kernel: the grid's x, y, bx, by, the tile depth T
# (declared by grid.with_ktile) and each workgroup array's declared length. A position that depends on a
# knob is a REFERENCE leaf, `literal.p_<knob>`, whose payload is an integer multiplier of the knob in
# units of 1/256 (Q8: 256 is the knob itself, 16 is a sixteenth of it); the renderer writes the value the
# kernel's own declarations give (src/wgsl/params.rs resolves it), so one change of a knob rewrites every
# reference coherently, and a reference grafted into another kernel reads that kernel's knob. `_log2` is the
# base-2 logarithm of the scaled knob (a shift), `_m1` the scaled knob minus one (a mask). The knobs by the
# output's axes (the layout knob swaps x and y): wcol, wrow the workgroup's extent along the columns and
# rows; bcol, brow the block factors; colspan = wcol * bcol and rowspan = wrow * brow the cells a workgroup
# covers; threads = wcol * wrow; ktile the tile depth. A reference never names a counter, register, binding
# or local.
KNOBS = ("wcol", "wrow", "bcol", "brow", "colspan", "rowspan", "threads", "ktile")
for k in KNOBS:
    for out in ("u32.index", "u32.lit64", "u32.lit4096", "u32.wglen"):
        f("literal", "p_" + k, "Literal", [], out, f"the knob {k} times the payload / 256")
    f("literal", "p_" + k + "_log2", "Literal", [], "u32.index", f"log2 of the knob {k} times the payload / 256 (a shift)")
    f("literal", "p_" + k + "_m1", "Literal", [], "u32.index", f"the knob {k} times the payload / 256, minus one (a mask)")
f("grid", "with_ktile", "composite", ["grid.grid", "u32.lit4096"], "grid.grid", "a grid and the kernel's tile depth T (the item loop's step), the knob literal.p_ktile references")


# ---- uniform values (2026-10-12, legality item 5) ----
# The values every thread of a workgroup holds alike, and the arithmetic of them: the bound of a loop that
# may hold a barrier. index.of_uniform uses one anywhere an index is wanted. A loop with any other bound at
# an L rung is written as a conditional statement in place (stmt.cblock_L<d>): its body holds no barrier.
for fld, wgsl in (("cfg_m", "cfg.m"), ("cfg_d", "cfg.d"), ("cfg_r_rows", "cfg.r_rows"), ("cfg_r_off", "cfg.r_off")):
    f("uniform", fld, "composite", [], "u32.uniform", wgsl)
f("uniform", "of_lit", "composite", ["u32.lit4096"], "u32.uniform", "a literal")
for c in ("x", "y", "z"):
    f("uniform", "wid_" + c, "composite", ["vec3<u32>.wid"], "u32.uniform", f"wid.{c}: the workgroup's id")
    # A leaf (2026-10-12): its input was any vec3<u32>.count, which a splat of a thread's value also is, while
    # the text is nwg.<c> whatever it holds (seed 523 of the random genes: a barrier the checker saw in a loop
    # bounded by a thread's value).
    f("uniform", "nwg_" + c, "composite", [], "u32.uniform", f"nwg.{c}: the grid's size")
f("uniform", "subgroup_size", "composite", [], "u32.uniform", "the subgroup size")
f("uniform", "num_subgroups", "composite", [], "u32.uniform", "the subgroups in the workgroup")
# A difference, a product and a shift may wrap: their value is u32.uniformw, an index only (2026-10-12).
WRAPS = ("sub", "mul", "shl")
for op in ("add", "sub_floor", "mul", "div", "min", "max", "shl", "shr", "and", "or"):
    f("uniform", op, "composite", ["u32.uniform", "u32.uniform"], "u32.uniformw" if op in WRAPS else "u32.uniform", f"as index.{op}, of uniform values")
for op in ("sub", "mod", "round_down", "band_of", "clamp_to"):
    f("uniform", op, "composite", ["u32.uniform", "u32.uniform"], "u32.uniformw" if op in WRAPS else "u32.uniform", f"as index.{op}, of uniform values")
f("uniform", "clamp", "composite", ["u32.uniform", "u32.uniform", "u32.uniform"], "u32.uniform", "as index.clamp, of uniform values (arity 3)")
# A loop counter is uniform wherever a loop that may hold a barrier can read it: every loop at an L rung has
# a uniform bound (a literal, a binding, a uniform value), and a loop with any other bound is at a C rung.
f("uniform", "of_counter", "composite", ["u32.counter"], "u32.uniform", "the item loop's counter t")
f("uniform", "of_tcounter", "composite", ["u32.tcounter"], "u32.uniform", "the tile counter k")
f("uniform", "of_tcounter2", "composite", ["u32.tcounter2"], "u32.uniform", "the tile counter k2")
f("uniform", "of_tcounter3", "composite", ["u32.tcounter3"], "u32.uniform", "the tile counter k3")
f("uniform", "of_tcounter4", "composite", ["u32.tcounter4"], "u32.uniform", "the tile counter k4")
f("index", "of_uniform", "composite", ["u32.uniform"], "u32.index", "a uniform value used as an index")
f("bound", "of_uniform", "composite", ["u32.uniform"], "u32.boundu", "a uniform loop bound")
for d in (0, 1, 2, 3, 4, 5):
    f("stmt", f"cblock_L{d}", "composite", [f"block.C{d}"], f"stmt.L{d}", "conditional statements (no barrier) written in place")
# A uniform bound where any bound is taken (a loop at a C or O rung, a loop that breaks): uncapped, the
# uniform values are known (2026-10-12, the checker's bounded loops: bound.of_index carries its cap).
f("bound", "x_of_uniform", "composite", ["u32.uniform"], "u32.boundx", "a uniform loop bound, uncapped")
# NO WRAP IN A LOOP BOUND (2026-10-12): a uniform value that may wrap is an index; a min clamps it to a value
# that cannot (min(1024u, cfg.m - t), reserve2 g003 r12); an expression bound's value is u32.xbound.
# Retired the same day (Codex's review: a wrap is illegal everywhere, a clamped one too): these took a
# u32.uniformw; their output is u32.uniformw now, so no demand reaches them. Kept so later refs stand.
f("index", "of_uniform", "composite", ["u32.uniformw"], "u32.uniformw", "retired: no row takes u32.uniformw")
f("uniform", "min", "composite", ["u32.uniform", "u32.uniformw"], "u32.uniformw", "retired: no row takes u32.uniformw")
f("uniform", "min", "composite", ["u32.uniformw", "u32.uniform"], "u32.uniformw", "retired: no row takes u32.uniformw")
# An expression bound's value (u32.xbound): the rows of u32.index that cannot pass their operands, at
# signatures of their own (the same rows, so the renderer and the checker read them as the index rows).
f("reg", "get_i", "composite", ["reg<u32>.index"], "u32.xbound", "a u32 register's value as a loop bound")
f("index", "of_uniform", "composite", ["u32.uniform"], "u32.xbound", "a uniform value as a loop bound")
for op in ("add", "max", "or"):
    f("index", op, "composite", ["u32.xbound", "u32.xbound"], "u32.xbound", f"index.{op} of bound values")
for op in ("sub_floor", "div", "mod", "shr", "and", "round_down", "min"):
    f("index", op, "composite", ["u32.xbound", "u32.index"], "u32.xbound", f"index.{op}: at most its first operand, whatever the second")
f("index", "min", "composite", ["u32.index", "u32.xbound"], "u32.xbound", "min(a, b): at most b, whatever a")
f("index", "clamp_to", "composite", ["u32.xbound", "u32.count"], "u32.xbound", "i < n: at most i")
f("literal", "index", "Literal", [], "u32.xbound", "a literal bound value, any u32 (2026-10-12: a uniform literal is 1..=4096)")
for op in ("min", "max"):
    f("subgroup", op, "SubgroupCollectiveOperation", ["u32.xbound"], "u32.xbound", f"subgroup{op.capitalize()}(x) of a bound value")
f("subgroup", "broadcast_first", "SubgroupGather", ["u32.xbound"], "u32.xbound", "subgroupBroadcastFirst(x) of a bound value")
f("subgroup", "broadcast", "SubgroupGather", ["u32.xbound", "u32.lit64"], "u32.xbound", "subgroupBroadcast(x, lane) of a bound value")
for op in ("shuffle", "shuffle_down", "shuffle_up", "shuffle_xor"):
    f("subgroup", op, "SubgroupGather", ["u32.xbound", "u32.index"], "u32.xbound", f"{op}(x, i) of a bound value")
# THE ORIGINS (2026-10-12): the thread's origin rows at the origins' duals, the only first input an offset row takes.
f("safe", "thread_row", "composite", ["vec3<u32>.gid", "u32.blk"], "u32.rowbase", "gid.y * blk: the thread's first row, as an offset row's origin")
f("safe", "thread_col", "composite", ["vec3<u32>.gid", "u32.blk"], "u32.colbase", "gid.x * blk: the thread's first column, as an offset column's origin")
f("safe", "thread_row_s", "composite", ["vec3<u32>.wid", "vec3<u32>.lid", "u32.blk"], "u32.rowbases", "arity 3: the first row of a strided block, as an offset row's origin")
f("safe", "thread_col_s", "composite", ["vec3<u32>.wid", "vec3<u32>.lid", "u32.blk"], "u32.colbases", "arity 3: the first column of a strided block, as an offset column's origin")
f("safe", "thread_row_g", "composite", ["vec3<u32>.wid", "vec3<u32>.lid", "u32.blk", "u32.lit64"], "u32.rowbaseg", "arity 4: the first row of a grouped block, as an offset row's origin")
f("safe", "thread_col_g", "composite", ["vec3<u32>.wid", "vec3<u32>.lid", "u32.blk", "u32.lit64"], "u32.colbaseg", "arity 4: the first column of a grouped block, as an offset column's origin")
# THE SYMMETRIC HALF (2026-10-14): the mirror of the thread's own cell and its store, after every other row
# so every ref stands. No sampler draws them (render::sample leaves them out): the symmetric-half rule
# (rules_more.rs) writes the mirror only under its triangle guard (row < col inside row <= col), and the
# checker's layer 7 proves the write sets under the whole-H contract.
for r in ("threadrow", "threadrows", "threadrowg"):
    for c in ("threadcol", "threadcols", "threadcolg"):
        f("safe", "mirror_of", "composite", ["u32." + r, "u32." + c], "u32.mirror", "col * cfg.d + row: the mirror of the cell this thread owns")
# A test of two workgroup-uniform values whose body keeps its rung (it may hold a barrier): the symmetric
# half's `if (wid_row <= wid_col)`, which skips the workgroups strictly below the diagonal. Uniform by its
# operands' dual; no sampler draws it either.
for r in ("L0", "L1", "L2", "L3", "L4", "L5"):
    f("stmt", "if_ule_" + r, "composite", ["u32.uniform", "u32.uniform", "block." + r], "stmt." + r, "arity 3: if (a <= b) { ... } over workgroup-uniform values: the body keeps the rung")
# (Not an epilogue statement: the epilogue still ends in the store of the thread's own cell.)
for r in RUNGS + RUNGS5:
    f("store", "h_mirror_" + r, "composite", ["u32.mirror", "f32.real"], "stmt." + r, "h[mirror] = h[mirror] + v under the bounds check of the thread's own cell: the symmetric half's second write")
# THE WORKGROUP PUT (2026-10-14, the checker's layer 7: 458 of 1,000 random kernels had two threads of a
# workgroup write one element, the old rows taking any masked index): a thread writes its own element,
# lidx + k * threads, below the array's length, between two workgroup barriers, so every read of the
# array is ordered against every write and no two threads write one element. At the barrier rungs (L)
# only. The sampler draws these and no longer the old writes (wg.set*, which the reader keeps).
f("safe", "wg_own", "composite", ["u32.lit64"], "u32.wgown", "lidx + k * threads: this thread's k-th element of a workgroup array")
f("safe", "wg_own_tile", "composite", ["u32.tcounter"], "u32.wgown", "lidx + k * threads over the tile loop's counter k")
f("safe", "wg_own_tile2", "composite", ["u32.tcounter2"], "u32.wgown", "lidx + k2 * threads over the inner tile loop's counter k2")
for r in ("L0", "L1", "L2", "L3", "L4", "L5"):
    f("wg", "put_" + r, "composite", ["wgmem<f32>.real", "u32.wgown", "f32.real"], "stmt." + r, "arity 3: barrier; if (i < len) { a[i] = v; } barrier")
    f("wg", "put_v_" + r, "composite", ["wgmem<vec4<f32>>.real", "u32.wgown", "vec4<f32>.real"], "stmt." + r, "arity 3: barrier; if (i < len) { va[i] = v; } barrier")
    f("wg", "put_v2_" + r, "composite", ["wgmem<vec2<f32>>.real", "u32.wgown", "vec2<f32>.real"], "stmt." + r, "arity 3: barrier; if (i < len) { v2a[i] = v; } barrier")
    f("wg", "put_u_" + r, "composite", ["wgmem<u32>.index", "u32.wgown", "u32.index"], "stmt." + r, "arity 3: barrier; if (i < len) { ua[i] = v; } barrier")

# ---- several entry points in one module (2026-10-17, B6 of PLAN_v24_ingest_evolution) ----
# A module of N entry points (the interface names them, in order: v24's `ingest` then `search`): one safe
# kernel each, each with its own grid, body and epilogue; the bindings, the uniform struct and the helpers
# are the module's, written once. Arity up to the decoders' K_MAX of four.
for n in (2, 3, 4):
    f("kernel", f"module_{n}", "composite", ["kernel.safe"] * n, "kernel.module", f"arity {n}: a module of {n} entry points, the interface's names in order, the shared declarations once")

# ---- uniform tests whose body keeps its rung, beyond <= (2026-10-17, run 1 of v24: 13 of 21 model seeds
# hold a barrier under `if (cfg.pass_index == 0u)`, one pass's work in one kernel): ==, != and < of two
# workgroup-uniform values, as stmt.if_ule; > and >= are these with their operands swapped. After every
# other row so every ref stands; no sampler draws them (as stmt.if_ule). ----
for op, wgsl in (("ueq", "=="), ("une", "!="), ("ult", "<")):
    for r in ("L0", "L1", "L2", "L3", "L4", "L5"):
        f("stmt", f"if_{op}_" + r, "composite", ["u32.uniform", "u32.uniform", "block." + r], "stmt." + r, f"arity 3: if (a {wgsl} b) {{ ... }} over workgroup-uniform values: the body keeps the rung")
# The same tests with an else (`if (cfg.pass_index == 0u) { ... } else { ... }`, 12 of the 29 seeds of run 1):
# both bodies keep the rung; `else if` is an else whose block holds the next test.
for op, wgsl in (("ule", "<="), ("ueq", "=="), ("une", "!="), ("ult", "<")):
    for r in ("L0", "L1", "L2", "L3", "L4", "L5"):
        f("stmt", f"if_else_{op}_" + r, "composite", ["u32.uniform", "u32.uniform", "block." + r, "block." + r], "stmt." + r, f"arity 4: if (a {wgsl} b) {{ ... }} else {{ ... }} over workgroup-uniform values: both bodies keep the rung")


# ---- an index read as a count (2026-10-17, 10 of run 1's 73 seeds: a shift by index arithmetic, `h >> (k * 4u)`,
# whose count operand is u32.count): the same u32 lanes, the conversion conform inserts. ----
f("convert", "index_to_count", "As", ["<u32.index>"], "<u32.count>", "the same u32 lanes read as a count (a shift's count computed as an index)")
# ---- four more workgroup arrays of indices (2026-10-17, 3 of run 1's seeds keep six to eight u32 lists and
# counters in workgroup memory: a search's selected ids, counts, a slot) ----
for n in ("ue", "uf", "ug", "uh"):
    f("wg", n, "composite", [], "wgmem<u32>.index", f"workgroup array wg_{n} of u32")
# (and four more f32 arrays, wg.e..wg.h, for a search's score lists beside its staged rows; a u32 read from
# workgroup memory as a loop bound's value: a loop over the candidates a thread staged)
for n in ("e", "f", "g", "h"):
    f("wg", n, "composite", [], "wgmem<f32>.real", f"workgroup array wg_{n} of f32")
f("wg", "get_u_safe", "composite", ["wgmem<u32>.index", "u32.wgindex"], "u32.xbound", "a[i] as a loop bound's value")
# ---- a loop that breaks, stepped by a literal (2026-10-17, 4 of run 1's seeds: a search scanning its candidates
# 256 at a time until it has enough): the tile loop's stride beside the break block ----
for r, b in (("L0", 2), ("L1", 2), ("L2", 3), ("L3", 4), ("L4", 5), ("C0", 1), ("C1", 2), ("C2", 3), ("C3", 4), ("C4", 5)):
    f("loop", "tilex_strided_brk_" + r, "composite", ["u32.boundx", "u32.lit4096", f"block.B{b}"], "stmt." + r, "arity 3: for (k = 0u; k < (expr); k = k + s) { body that breaks }")


# ---- a workgroup-uniform predicate and its tests (2026-10-17, run 1's seeds: `if (cfg.stage == 1u && wid.x <
# NC) { ..barrier.. }`, `else if (p == 1u || p == 3u)`): comparisons of uniform values, their and, or and not,
# and the if and if-else whose bodies keep their rung over one. No sampler draws them (as stmt.if_ule). ----
for op, wgsl in (("eq", "=="), ("ne", "!="), ("lt", "<"), ("le", "<="), ("gt", ">"), ("ge", ">=")):
    f("uflag", op, "composite", ["u32.uniform", "u32.uniform"], "bool.uflag", f"a {wgsl} b of workgroup-uniform values")
f("uflag", "and", "composite", ["bool.uflag", "bool.uflag"], "bool.uflag", "a && b")
f("uflag", "or", "composite", ["bool.uflag", "bool.uflag"], "bool.uflag", "a || b")
f("uflag", "not", "composite", ["bool.uflag"], "bool.uflag", "!a")
for r in ("L0", "L1", "L2", "L3", "L4", "L5"):
    f("stmt", "if_uflag_" + r, "composite", ["bool.uflag", "block." + r], "stmt." + r, "if (c) { ... } over a workgroup-uniform predicate: the body keeps the rung")
    f("stmt", "if_else_uflag_" + r, "composite", ["bool.uflag", "block." + r, "block." + r], "stmt." + r, "arity 3: if (c) { ... } else { ... } over a workgroup-uniform predicate: both bodies keep the rung")

# ---- 64-element u32 private arrays (2026-10-17, the v24 reference's `var rows: array<u32, 64>` beside its 64 scores:
# read into a 16-element array before, masked & 15u, which changed what it computed): pua, puc, indexed as the
# 64-element f32 arrays are (u32.pindex, i & 63u), read and written by the u32 array rows' names ----
for n in ("ua", "uc"):
    f("parr", n, "composite", [], "parr64<u32>.index", f"private array p{n} of 64 u32")
f("parr", "get_i", "composite", ["parr64<u32>.index", "u32.pindex"], "u32.index", "pua[i]")
for r in RUNGS + RUNGS5:
    f("parr", "set_i_" + r, "composite", ["parr64<u32>.index", "u32.pindex", "u32.index"], "stmt." + r, "arity 3: pua[i] = v")

# ---- four more workgroup arrays of indices (2026-10-17, v24 run 1's seed20 keeps nine u32 lists and counters in
# workgroup memory: two top lists, a cell's count, order, selection and starts, two scalar counters, the rows) ----
for n in ("ui", "uj", "uk", "ul"):
    f("wg", n, "composite", [], "wgmem<u32>.index", f"workgroup array wg_{n} of u32")
# ---- vec2<f32> registers (2026-10-17, v24 run 1's seed11: a statement helper returning `vec2<f32>(s, n)`, a dot
# product and a norm, read into a register at its call) ----
for i in range(8):
    f("reg", f"w{i}", "composite", [], "reg<vec2<f32>>.real", "a vec2<f32> register")
f("reg", "get_w", "composite", ["reg<vec2<f32>>.real"], "vec2<f32>.real", "the register's value")
for r in RUNGS + RUNGS5:
    f("reg", "set_w_" + r, "composite", ["reg<vec2<f32>>.real", "vec2<f32>.real"], "stmt." + r, "")
# ---- any flag as a workgroup-uniform predicate, proved so by the checker (2026-10-17, v24 run 1's seeds 06, 21,
# 28, 29, 31: `if (n > 0u)` over `n = workgroupUniformLoad(&wu1)`, `if (dc)` over a flag set under a uniform test,
# each over a barrier): the row writes the flag; layer 4 proves it uniform or names the barrier in non-uniform
# control ----
f("uflag", "of_flag", "composite", ["bool.flag"], "bool.uflag", "the flag, workgroup-uniform when the checker's layer 4 proves it")
# (and an index as a uniform value, a loop's bound over a barrier: `for (b = 0u; b < c; b += 8u)` with c a helper's
# argument from a workgroupUniformLoad, seeds 21 and 29)
f("uniform", "of_checked", "composite", ["u32.index"], "u32.uniform", "the index, workgroup-uniform when the checker's layer 4 proves it")
# ---- workgroupUniformLoad of a workgroup array's element into a register (2026-10-17, v24 run 1's seed21: a count
# written by one thread, read by all as `workgroupUniformLoad(&wu1)` and tested over a barrier): one statement, a
# barrier for the checker (uniform control only), its register uniform when the index is; L rungs only ----
for r in ("L0", "L1", "L2", "L3", "L4", "L5"):
    f("wg", "uload_i_" + r, "composite", ["reg<u32>.index", "wgmem<u32>.index", "u32.wgindex"], "stmt." + r, "arity 3: r = workgroupUniformLoad(&a[i])")
    f("wg", "uload_f_" + r, "composite", ["reg<f32>.real", "wgmem<f32>.real", "u32.wgindex"], "stmt." + r, "arity 3: r = workgroupUniformLoad(&a[i])")
# ---- 64-element vec4<f32> private arrays (2026-10-17, v24 run 1's seed28: `var qreg: array<vec4<f32>, 32>`, a query
# held in registers, read 16-wide before; indexed as the 64-element arrays are, u32.pindex, i & 63u) ----
for n in ("vq", "vr"):
    f("parr", n, "composite", [], "parr64<vec4<f32>>.real", f"private array p{n} of 64 vec4<f32>")
f("parr", "get_v", "composite", ["parr64<vec4<f32>>.real", "u32.pindex"], "vec4<f32>.real", "pvq[i]")
for r in RUNGS + RUNGS5:
    f("parr", "set_v_" + r, "composite", ["parr64<vec4<f32>>.real", "u32.pindex", "vec4<f32>.real"], "stmt." + r, "arity 3: pvq[i] = v")
# ---- an argmax or argmin tree reduction (2026-10-17, v24 run 1's seed04: the best centre's score and its index in two
# workgroup arrays, `if (t < s) { if (v[t + s] > v[t]) { v[t] = v[t + s]; i[t] = i[t + s]; } }` halving s): the value
# array and the index array reduced into element 0, a barrier before and after every step, L rungs only ----
for op in ("argmax", "argmin"):
    for r in ("L0", "L1", "L2", "L3", "L4", "L5"):
        f("wg", f"reduce_{op}_{r}", "composite", ["wgmem<f32>.real", "wgmem<u32>.index", "u32.lit4096"], "stmt." + r, f"v[0..n) and i[0..n) reduced into element 0 by {op} of v, i moving with v: stride halving, barriers in uniform control flow")
# ---- two more workgroup arrays of atomic<u32> (2026-10-19, 5 of run 4's 32 seeds keep three or four atomic
# counters in workgroup memory: a search's candidate count, its ties, a slot) ----
for n in ("au_c", "au_d"):
    f("wg", n, "composite", [], "wgmem<atomic<u32>>.index", f"workgroup array wg_{n} of atomic<u32>")
# ---- workgroup arrays of i32 (2026-10-19, run 4's seeds 14 and 29 stage int8-quantised values in i32 arrays
# in workgroup memory): two arrays na, nb of 2,048 by default, indexed as the f32 and u32 arrays are ----
for n in ("na", "nb"):
    f("wg", n, "composite", [], "wgmem<i32>.int", f"workgroup array wg_{n} of i32")
f("wg", "get_n_safe", "composite", ["wgmem<i32>.int", "u32.wgindex"], "i32.int", "a[i], in range by its type")
for r in RUNGS + RUNGS5:
    f("wg", "set_n_" + r, "composite", ["wgmem<i32>.int", "u32.wgindex", "i32.int"], "stmt." + r, "arity 3: a[i] = v")
f("wg", "decl_n", "composite", ["wgmem<i32>.int", "u32.wglen"], "wg.decl", "var<workgroup> a: array<i32, n>")
# ---- 128-element u32 private arrays (2026-10-19, run 4's seed10 caches its 100 Hamming distances a thread in
# `var cache: array<u32, 100>` between two barriers): pux, puy, indexed i & 127u (u32.pindex128) ----
f("safe", "pidx128", "composite", ["u32.index"], "u32.pindex128", "i & 127u")
for n in ("ux", "uy"):
    f("parr", n, "composite", [], "parr128<u32>.index", f"private array p{n} of 128 u32")
f("parr", "get_i", "composite", ["parr128<u32>.index", "u32.pindex128"], "u32.index", "pux[i]")
for r in RUNGS + RUNGS5:
    f("parr", "set_i_" + r, "composite", ["parr128<u32>.index", "u32.pindex128", "u32.index"], "stmt." + r, "arity 3: pux[i] = v")
for r in ("L0", "L1", "L2", "L3", "L4", "L5"):
    f("wg", "put_n_" + r, "composite", ["wgmem<i32>.int", "u32.wgown", "i32.int"], "stmt." + r, "arity 3: barrier; if (i < len) { na[i] = v; } barrier")

# ---- vector conversions between floats and unsigned integers (2026-10-19, the harvested corpus: `vec2<f32>(gid.xy)`,
# `vec2<u32>(uv * size)`): per lane, as the signed ones are ----
for L in (2, 3, 4):
    f("convert", "vindex_to_f32", "As", [f"vec{L}<u32>.index"], f"vec{L}<f32>.real", "per lane")
    f("convert", "vcount_to_f32", "As", [f"vec{L}<u32>.count"], f"vec{L}<f32>.real", "per lane")
    f("convert", "vf32_to_count", "As", [f"vec{L}<f32>.real"], f"vec{L}<u32>.count", "truncating, floor at 0, per lane")

# ---- registers of the other vector types (2026-10-19): the registers, the read, a set at every rung ----
for letter, slot, form in REG_KINDS:
    for i in range(REG_KIND_COUNT):
        f("reg", f"{letter}{i}", "composite", [], f"reg<{slot}>.{form}", f"a {slot} register")
    f("reg", f"get_{letter}", "composite", [f"reg<{slot}>.{form}"], f"{slot}.{form}", "the register's value")
    for r in RUNGS + RUNGS5:
        f("reg", f"set_{letter}_" + r, "composite", [f"reg<{slot}>.{form}", f"{slot}.{form}"], "stmt." + r, "")

# ---- a module of more than four entry points (2026-10-19, the harvested corpus: 59 of 344 non-dawn reads that failed;
# cuneus's blackhole has eight): a node has at most four children (K), so a module of five or more is a module of
# two to four with entry points appended in threes: module_ext<k>(module, k kernels) is the module and k more ----
for k in (1, 2, 3):
    f("kernel", f"module_ext{k}", "composite", ["kernel.module"] + ["kernel.safe"] * k, "kernel.module", f"arity {k + 1}: the module and {k} more entry points, in order after it")

# ---- the slot counts (2026-10-19): registers of the earlier kinds past the counts they were written with, and the
# arrays of every family past the names it grew with. Only terminal rows: the reads, writes and declarations are rows
# of the array's dual ----
for letter, dual, old, new in REG_EXTEND:
    for i in range(old, new):
        f("reg", f"{letter}{i}", "composite", [], dual, "a register")
for cls, dual, old, stem, new in ARRAY_FAMILIES:
    for i in range(old, new):
        f(cls, f"{stem}{i}", "composite", [], dual, f"array {stem}{i} of {dual}")


# ---- a real terminal for every demanded dual (2026-10-19, docs/PLAN_wgsl_terminals.md): multi-typed gene
# expression programming fills a tail of type T with a terminal of T, so a dual with no terminal is an unfillable
# tail. A constant of a value dual is a `const` row over its domain (class const, not literal: the reader reads a
# kernel's text into `literal` rows and never into these, and the sampler does not draw them); a dual with provenance (a thread's own row,
# cell, an empty block, a kernel that stores) takes an `own` row, whose text the renderer writes as the thread's own
# first row, column or cell, an empty block, a store of the thread's own cell, a kernel of the default grid. ----
for form, out, note in [
    ("vint", "<vecL.int>", "a constant vector of whole numbers, one value on every lane"),
    ("vindex", "<vecL.index>", "a constant vector of indices, one value on every lane"),
    ("vcount", "<vecL.count>", "a constant vector of counts, one value on every lane"),
    ("vbits", "<vecL.bits>", "a constant vector of bit fields, one value on every lane"),
    ("vflag", "<vecL.flag>", "a constant vector of predicates, one value on every lane"),
    ("q15_16", "<i32.q15_16>", "a fixed-point constant"),
    ("q15_16w", "vec2<i32>.q15_16w", "a wide fixed-point constant, one limb value on both lanes"),
    ("code4", "u32.code4", "a codebook entry"),
    ("code8", "u32.code8", "a codebook entry"),
    ("code16", "u32.code16", "a codebook entry"),
    ("code1024", "u32.code1024", "a codebook entry"),
    ("hash32", "u32.hash32", "a hash state constant"),
    ("hash64", "vec2<u32>.hash64", "a 64-bit hash state constant, one limb value on both lanes"),
    ("sign", "bool.sign", "+1 or -1 as a bit"),
    ("opaque", "<S.opaque>", "a constant of an opaque dual: any value of its scalar"),
    ("mat", "<matCxR<f32>.real>", "a matrix constant, one number on every element"),
]:
    f("const", form, "Literal", [], out, note)
OWN_NOTE = "the real terminal of the dual: "
for dual, text in [
    ("u32.threadrow", "the thread's own first row"), ("u32.threadcol", "the thread's own first column"),
    ("u32.threadrows", "the thread's own first row of a strided block"), ("u32.threadcols", "the thread's own first column of a strided block"),
    ("u32.threadrowg", "the thread's own first row of a grouped block"), ("u32.threadcolg", "the thread's own first column of a grouped block"),
    ("u32.rowbase", "the origin of the thread's row"), ("u32.colbase", "the origin of the thread's column"),
    ("u32.rowbases", "the origin of the thread's strided row"), ("u32.colbases", "the origin of the thread's strided column"),
    ("u32.rowbaseg", "the origin of the thread's grouped row"), ("u32.colbaseg", "the origin of the thread's grouped column"),
    ("u32.cell", "the thread's own first output cell"), ("u32.mirror", "the mirror of the thread's own first cell"),
    ("u32.wgown", "the thread's own workgroup element, lidx"), ("u32.rowid", "the y of the thread's global id"), ("u32.colid", "the x of the thread's global id"),
    ("u32.wgindex", "workgroup-array index 0"), ("u32.wgvindex", "vec4 workgroup-array index 0"), ("u32.wgv2index", "vec2 workgroup-array index 0"),
    ("u32.wgaindex", "atomic workgroup-array index 0"), ("u32.pindex", "private-array index 0"), ("u32.pindex16", "16-element private-array index 0"),
    ("u32.pindex128", "128-element private-array index 0"), ("u32.boundx", "a loop bound of one pass, 1u"), ("u32.boundu", "a workgroup-uniform loop bound of one pass, 1u"),
    ("u32.uniform", "a workgroup-uniform value, 0u"), ("u32.uniformw", "a workgroup-uniform value, 0u"), ("bool.uflag", "a workgroup-uniform predicate, false"),
    ("block.block", "an empty block {}"),
    ("kernel.safe", "the default-grid kernel whose epilogue stores the thread's own cell"),
    ("kernel.module", "a module of one such kernel at each of the interface's entry points"),
    ("grid.grid", "the default grid: 16 x 16 threads, one cell each"),
    ("wg.decl", "no workgroup array declaration"), ("wg.decls", "the arrays the kernel uses, at their default lengths"),
]:
    slot, form = dual.split(".")
    f("own", f"{slot}_{form}", "composite", [], dual, OWN_NOTE + text)
for r in ("L0", "L1", "L2", "L3", "L4", "L5", "C0", "C1", "C2", "C3", "C4", "C5", "O0", "O1", "O2", "O3", "O4", "B1", "B2", "B3", "B4", "B5"):
    store = r.startswith("O")
    if r[0] != "L" and r[0] != "B":
        f("own", f"stmt_{r}", "composite", [], f"stmt.{r}", OWN_NOTE + ("a store to the thread's own cell" if store else "an empty statement {}"))
    f("own", f"block_{r}", "composite", [], f"block.{r}", OWN_NOTE + ("a block ending in a store to the thread's own cell" if store else "an empty block {}"))


def write_tsv(path, header, rows):
    with open(path, "w") as out:
        out.write("\t".join(header) + "\n")
        for r in rows:
            out.write("\t".join(str(x) for x in r) + "\n")


def main():
    types = build_types()
    write_tsv(os.path.join(HERE, "types.tsv"), ["ref", "dual", "slot", "form", "lanes", "fallback", "fallback_bits", "fallback_noop", "fallback_elem", "meaning", "literal"], types)
    frows = [(f"F{i + 1:03d}", c, inst, naga, " ".join(ins) if ins else "-", out, len(ins), note) for i, (c, inst, naga, ins, out, note) in enumerate(FUNCTIONS)]
    write_tsv(os.path.join(HERE, "functions.tsv"), ["ref", "class", "instance", "naga", "in", "out", "arity", "note"], frows)

    with open(os.path.join(HERE, "TYPES.md"), "w") as md:
        md.write("# safeWGSL kingdom — the type table\n\n")
        md.write("Generated by `gen_tables.py` from `types.tsv`; edit the script, not this file.\n\n")
        md.write("A type is a dual `slot.form`: the slot is what the hardware stores, the form is what the value means and which functions are legal on it. Every dual has a unique ref. naga admits no implicit conversion, so a function's signature over duals is exact; the `convert` class holds the only crossings.\n\n")
        md.write("## Forms\n\n| form | meaning | legal scalar slots |\n|---|---|---|\n")
        for form, (meaning, scalars) in FORMS.items():
            md.write(f"| `{form}` | {meaning} | {', '.join(scalars) if scalars else 'see duals'} |\n")
        md.write("\n## Duals\n\n| ref | dual | slot | form | lanes | fallback | bits | noop | elem | literal |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for ref, dual, slot, form, lanes, fb, bits, noop, elem, _, lit in types:
            md.write(f"| {ref} | `{dual}` | `{slot}` | `{form}` | {lanes} | `{fb}` | {bits} | {noop} | {elem} | `{lit}` |\n")
        md.write(f"\n{len(types)} duals. The fallback is what Design C's projection substitutes for a codon whose dual does not match the demand; the codon is counted as masked.\n")
        md.write("\nThe three machine columns (`fallback_bits`, `fallback_noop`, `fallback_elem` in `types.tsv`) are what the loader builds the fallback leaf from, never the prose: `bits` is the lane's 32-bit pattern as an unsigned integer, repeated across the lanes of a vector or matrix (only `bool.sign` is non-zero, true = +1); `noop` is 1 for `store.store`, which substitutes no value at all; `elem` is 1 for `array<T>`, whose fallback is its element's. The loader tests `noop` and `elem` before reading `bits`; for those two rows `bits` is a placeholder 0. Every substitution, whichever column decides it, counts the codon as masked.\n")
        md.write("\nThe `literal` column is a scalar dual's literal domain (`LITERAL_DOMAIN` in this script): the values a literal of the dual takes, a range `lo..=hi` (real for an f32 slot) or a set `a,b,c`; `-` where no literal is. A gene holds a literal's value in its Dc tail, never as a symbol (phylu `docs/PLAN_wgsl_literals_as_constants.md`); mutation draws inside the domain and every value is read into it before it becomes a tree. An integer form takes its scalar's whole range: a tree's constant is exact for every u32 and i32.\n")
        md.write("\n## Loop trip counts\n\nMAX_LOOP_TRIP = 2^24 = 16,777,216 is the kingdom's declared maximum trip count of a loop (`render::MAX_LOOP_TRIP`): no real workload exceeds it (production builds run m ~ 420,000 items, d up to 8,192). A loop at an L rung, which may hold a barrier, takes a uniform bound (`u32.boundu`); a loop with no barrier takes any expression bound. Every bound a gene computes, uniform or not (`bound.of_index`, `bound.of_uniform`, `bound.x_of_uniform`), renders `min(e, 16777216u)`, so a bound in range computes the same and no loop runs past it. The checker's range layer ranges such a bound up to it, and proves index arithmetic over it.\n")
        md.write("\n## No wrap, no zero divisor, every computed bound capped\n\nA u32 wrap in index arithmetic at the worst shape is illegal everywhere (Codex's review of Andrew's decision, 2026-10-12): the checker's range layer fails it. The kingdom cannot spell a difference that wraps: a raw difference is `u32.indexw` and a uniform difference, product or shift `u32.uniformw`, which no row takes; the kingdom's difference is the floored one (`index.sub_floor`, `uniform.sub_floor`), and the reader reads `x - y` as it (the same value wherever it does not wrap). A sum, product or left shift of index values is not typed by range: the checker fails one that wraps at the worst shape. An expression bound takes `u32.xbound` (registers, uniform values, the rows that cannot pass their operands). Every loop bound a gene computes renders `min(e, 16777216u)` (`bound.of_index`, `bound.of_uniform`, `bound.x_of_uniform`); a binding's bound (`cfg.m`, `cfg.d`, `cfg.r_rows`) is the host's data size and a literal is at most 64. A divisor that is not a literal of at least 1 renders `max(b, 1u)` (WGSL's `x / 0u` is x, as is `x / max(0u, 1u)`).\n")
        md.write("\n## Not in the table\n\nImages, samplers, ray queries, binding arrays, pointers as values, non-square `f16` matrices. Storage-buffer atomics are the per-kernel rows of an interface that declares an atomic binding (`render::Role::interface`: a scalar `atomic<u32>`, or an array of atomics at any index, the bounds policy deciding an access past it), not rows of this table.\n")

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
        md.write(f"\n{len(frows)} function templates. Statement rows (kernel, grid, block, stmt, loop, reg, wg, the three literal codes and builtin.item/tile) make the kingdom multityped over whole kernels (`docs/PLAN_wgsl_gram_superopt.md` §2); `src/wgsl/render.rs` writes their text. Storage-buffer atomics are an interface's per-kernel rows, not rows of this table. Deliberately absent: atomic compare-exchange (naga 0.20's Metal backend lacks it), subgroupBallot, images, derivatives, pointers as values, implicit conversions, raw multiply on `q15_16`, modulo on a hash, arithmetic on `bits`.\n")
    print(f"{len(types)} duals, {len(frows)} functions (safeWGSL)")


if __name__ == "__main__":
    main()
