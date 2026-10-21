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
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from interface import GRAM
import skeleton, values, regs, loops, wg, vectors


def main():
    iface = GRAM
    # types first (every row names declared types), in dependency order
    skeleton.declare_types(iface)
    values.declare(iface)
    vectors.declare(iface)
    regs.declare(iface)
    loops.declare(iface)
    wg.declare(iface)
    skeleton.build(iface)
    values.build(iface)
    vectors.build(iface)
    regs.build(iface)
    loops.build(iface)
    wg.build(iface)
    twins = lib.finish_twins()
    problems, produced = lib.write_all(lib.HERE, "kernel.safe", iface)
    print(f"{len(lib.TYPES)} types, {len(lib.ROWS)} rows ({twins} w twins), {sum(1 for r in lib.ROWS.values() if r['values'] is not None)} constant placeholders")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
