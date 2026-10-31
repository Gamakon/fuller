#!/usr/bin/env python3
"""The true multi-typed WGSL symbol table (ST2), generated exactly from
nucleotable/docs/MultiTypedGEP_s9_WGSL_draft.md for one interface instance
(the Gram worked instance, interface.py).

Why a new generator beside kingdoms/safewgsl/gen_tables.py: that one writes
the older dual/class/instance tables (templates over naga operations, an
`in` column that is a set of duals with no argument order) that the live
reader, renderer and loader (src/wgsl/loader.rs, safe_reader.rs, render.rs)
read through include_str!; rewriting it in place would break all of them.
The true kingdom needs ordered signatures and a spelling per row, a table
shape the old format cannot hold, so it is written beside it.

    python3 gen_tables.py        # writes types.tsv functions.tsv constants.tsv
                                 # prologue.wgsl declarations.wgsl interface.tsv
    python3 gen_tables.py img_volumepassage   # the image-output instance (ST9), into img/volumepassage/

The Gram instance is the default and writes beside this file. An interface with an output of kind "texture" (a
write-only storage texture, the cuneus visual shaders) adds the families of image.py and takes its own fuel budget K
from the interface; the directory it writes is `img/<name without img_>/`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib

import lib
from interface import INTERFACES


def main(argv):
    iface = INTERFACES[argv[1] if len(argv) > 1 else "gram"]
    # K and the strata are read by the submodules when they are imported
    # `phase_depth`: how deep the pool's phase rungs nest (6, every depth, unless the interface says less)
    lib.configure(iface.get("fuel_budget", 1), epilogue=not iface.get("module"), pools=bool(iface.get("wg_pool")), phase_depth=iface.get("phase_depth", 6))
    skeleton, values, regs, loops, wg, vectors, start = (importlib.import_module(m) for m in ("skeleton", "values", "regs", "loops", "wg", "vectors", "start"))
    image = importlib.import_module("image") if iface.get("image") else None
    arena = importlib.import_module("arena") if iface.get("arenas") or iface.get("row_outputs") else None
    pool = importlib.import_module("pool") if iface.get("wg_pool") else None
    outdir = lib.HERE if iface["name"] == "gram" else (os.path.join(lib.HERE, "mod", iface["name"]) if iface.get("module") else os.path.join(lib.HERE, "img", iface["name"].removeprefix("img_")))
    os.makedirs(outdir, exist_ok=True)
    # types first (every row names declared types), in dependency order
    skeleton.declare_types(iface)
    start.declare(iface)
    values.declare(iface)
    vectors.declare(iface)
    regs.declare(iface)
    loops.declare(iface)
    inrange = importlib.import_module("inrange")
    inrange.declare(iface)
    inrange.declare_runs(iface)
    countsort = importlib.import_module("countsort")
    countsort.declare(iface)
    if not pool:
        # a pool interface (s9 batch 9) keeps all its workgroup memory in the two pools of pool.py, not the stratum map
        wg.declare(iface)
    skeleton.build(iface)
    start.build(iface)
    values.build(iface)
    vectors.build(iface)
    regs.build(iface)
    loops.build(iface)
    if not pool:
        wg.build(iface)
    if image:
        image.declare(iface)
        image.build(iface)
    if pool:
        pool.declare(iface)
        pool.build(iface)
    if arena:
        arena.declare(iface)
        arena.build(iface)
    inrange.build(iface)
    inrange.build_runs(iface)
    countsort.build(iface)
    twins = lib.finish_twins()
    bare = lib.finish_bare()
    # docs/DESIGN_inrange_index_types.md: every in-range rule proved from the emitted spellings before anything is written
    for p in inrange.witness(iface):
        print("PROBLEM:", p)
        return 1
    # the entry-start type (start.py): a first-phase row stands only where nothing of the dispatch ran before it
    for p in start.witness(iface):
        print("PROBLEM:", p)
        return 1
    # the pool's extent (pool.py): every subscript of the plain pool in range of WG_PW + 4 words, whatever WG_PW is
    # compiler-assigned slots (countsort.py, section 10): every durable write is an own-index store or the cluster store,
    # every scaffold a compiler variant byte for byte
    for p in countsort.witness(iface) + countsort.modes(iface):
        print("PROBLEM:", p)
        return 1
    pool_problems = pool.witness(iface) if pool else []
    for p in pool_problems:
        print("PROBLEM:", p)
    if pool_problems:
        return 1
    problems, produced = lib.write_all(outdir, "kernel.module" if iface.get("module") else "kernel.safe", iface)
    countsort.write(outdir, iface)
    problems += countsort.check_dir(outdir, iface) if countsort.enabled(iface) else []
    print(f"{len(lib.TYPES)} types, {len(lib.ROWS)} rows ({twins} w twins, {bare} bare-literal twins), {sum(1 for r in lib.ROWS.values() if r['values'] is not None)} constant placeholders")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
