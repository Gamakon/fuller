"""Batch 1: the kernel skeleton, the statement and block rungs, the epilogue's
store, the ownership rows (design section 9). Every statement and block type
carries a depth `d` (the loop ladder) and a fuel stratum `k` (batch 3)."""
from lib import *

SHAPES = []   # target shapes "X Y": x * y a multiple of the largest subgroup size, at most max invocations


def shape_table(iface):
    big = max(iface["subgroup_sizes"]) if iface["has_subgroups"] else 1
    out = []
    for a in range(0, 11):
        for b in range(0, 11):
            n = (1 << a) * (1 << b)
            if n <= iface["max_invocations"] and n % big == 0:
                out.append(f"{1 << a} {1 << b}")
    return out


def declare_types(iface):
    T("kernel.safe", "the entry point; the gene's root")
    T("grid.shape", "one of the target's legal workgroup shapes (x, y, 1): a literal from the target's table")
    T("u32.blk_row", "a block factor on the row axis: cells per invocation along it", 1, "1,2,4,8")
    T("u32.blk_col", "a block factor on the column axis", 1, "1,2,4,8")
    T("grid.grid", "the entry's workgroup shape and block")
    T("u32.ktile", "the tile-depth knob: a bare literal 1..=4096 (a const initialiser, so no knob reference or call may produce it)", 1, "1..=4096")
    for x, d, k in rungs("LCOW"):
        T(S(x, d, k), f"one {x}-rung statement at depth {d}, fuel stratum {k}")
        T(B(x, d, k), f"a {x}-rung statement sequence at depth {d}, fuel stratum {k}")
    for d in range(1, 6):
        for k in STRATA:
            T(BB(d, k), f"a loop body that breaks, depth {d}, stratum {k}")
            T(WBB(d, k), f"a write-phase loop body that breaks, depth {d}, stratum {k}")
    T("vec3<u32>.gid", "global_invocation_id", 3)
    T("vec3<u32>.lid", "local_invocation_id", 3)
    T("vec3<u32>.wid", "workgroup_id", 3)
    T("vec3<u32>.nwg", "num_workgroups", 3)
    T("u32.lidx", "local_invocation_index")
    T("u32.rowid", "an id's component on the output's row axis (.y)")
    T("u32.colid", "an id's component on the output's column axis (.x)")
    T("u32.lit64", "a literal 1..=64", 1, "1..=64")
    T("u32.index", "any u32 used as a position (no ownership; wraps as WGSL defines)", 1, "0..=4294967295")
    T("u32.count", "any u32 used as a size or a field's value (no ownership)", 1, "0..=4294967295")
    T("f32.real", "a number", 1, "-3.4028234663852886e38..=3.4028234663852886e38")
    T("bool.flag", "a predicate that may differ between invocations", 1, "0,1")
    for o in iface["outputs"]:
        b = o["name"]
        T(f"u32.rowbase_{b}", f"the origin of this invocation's block of output {b} on the row axis")
        T(f"u32.colbase_{b}", f"the origin on the column axis of output {b}")
        T(f"u32.threadrow_{b}", f"a row of {b} this invocation owns")
        T(f"u32.threadcol_{b}", f"a column of {b} this invocation owns")
    for t in ("vec3<u32>.gid", "vec3<u32>.lid", "vec3<u32>.wid", "vec3<u32>.nwg", "u32.lidx", "u32.rowid", "u32.colid", "u32.lit64"):
        SAFE_ARGS.add(t)
    for o in iface["outputs"]:
        b = o["name"]
        for t in (f"u32.rowbase_{b}", f"u32.colbase_{b}", f"u32.threadrow_{b}", f"u32.threadcol_{b}"):
            SAFE_ARGS.add(t)


def store_text(iface, o, val="{2}"):
    b = o["name"]
    return "{ let r = {0}; let c = {1}; if (r < " + o["rows"] + " && c < " + o["cols"] + ") { " + b + "[r * " + o["cols"] + " + c] = " + val + "; } }"


def own_store_text(iface):
    o = next(o for o in iface["outputs"] if o["name"] == iface["primary"])
    b = o["name"]
    idx = f"{b}[r * {o['cols']} + c]"
    return "{ let r = gid.y * KNOB_BROW; let c = gid.x * KNOB_BCOL; if (r < " + o["rows"] + " && c < " + o["cols"] + ") { " + idx + " = " + idx + "; } }"


def build(iface):
    helper("lit_u32", "fn lit_u32(x: u32) -> u32 { return x; }")
    helper("lit_i32", "fn lit_i32(x: i32) -> i32 { return x; }")
    helper("lit_f32", "fn lit_f32(x: f32) -> f32 { return x; }")
    helper("lit_bool", "fn lit_bool(x: bool) -> bool { return x; }")
    helper("lit_f16", "fn lit_f16(x: f16) -> f16 { return x; }")

    entry = (
        "{0}\n@compute @workgroup_size(WG_X, WG_Y, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>, "
        "@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, "
        "@builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>"
        + (", @builtin(subgroup_size) sg_size: u32, @builtin(num_subgroups) sg_num: u32" if iface["has_subgroups"] else "")
        + ") {\n"
        + ("let sg_lane: u32 = lidx & (sg_size - 1u);\nlet sg_id: u32 = lidx / sg_size;\n" if iface["has_subgroups"] else "")
        + "<declarations>\n{1}\n{2}\n}\n"
    )
    R("kernel.entry_safe", "kernel.safe", ["grid.grid", B("L", 0, K), B("O", 0, K)], entry,
      "the only producer of the root; a kernel is a grid, a body and an epilogue that ends in the store of its own cell")

    # ---- grids
    kn = lambda wcol, wrow, bcol, brow: (
        "const WG_X: u32 = {0:0}u;\nconst WG_Y: u32 = {0:1}u;\nconst KNOB_WCOL: u32 = " + wcol + ";\nconst KNOB_WROW: u32 = " + wrow + ";\n"
        "const KNOB_BCOL: u32 = " + bcol + ";\nconst KNOB_BROW: u32 = " + brow + ";\nconst KNOB_KTILE: u32 = {3}u;\n"
        "const KNOB_THREADS: u32 = {0:0}u * {0:1}u;\n")
    R("grid.of", "grid.grid", ["grid.shape", "u32.blk_col", "u32.blk_row", "u32.ktile"],
      "// grid {0:0} {0:1} {1} {2}\n" + kn("{0:0}u", "{0:1}u", "{1}u", "{2}u"),
      "the shape is in the target's limits by the table; the header and the attribute come from one node", fields=True)
    R("grid.of_rows_x", "grid.grid", ["grid.shape", "u32.blk_row", "u32.blk_col", "u32.ktile"],
      "// grid {0:0} {0:1} {1} {2} rows-x\n" + kn("{0:1}u", "{0:0}u", "{2}u", "{1}u"),
      "the output's rows on thread x; the arguments in the order the spelling binds them", fields=True)
    C("literal.shape", "grid.shape", "{v}", shape_table(iface), "?_shape; its Dc domain is the table")
    C("literal.blk_row", "u32.blk_row", "{v}", ["1", "2", "4", "8"], "?_blk_row")
    C("literal.blk_col", "u32.blk_col", "{v}", ["1", "2", "4", "8"], "?_blk_col")
    gx, gy, bx, by = iface["default_grid"]
    R("own.grid_grid", "grid.grid", [], f"// grid {gx} {gy} {bx} {by}\nconst WG_X: u32 = {gx}u;\nconst WG_Y: u32 = {gy}u;\nconst KNOB_WCOL: u32 = {gx}u;\nconst KNOB_WROW: u32 = {gy}u;\nconst KNOB_BCOL: u32 = {bx}u;\nconst KNOB_BROW: u32 = {by}u;\nconst KNOB_KTILE: u32 = 1u;\nconst KNOB_THREADS: u32 = {gx * gy}u;\n",
      "the interface's default grid: its reference kernel's shape and blocks, ktile 1")
    C("literal.ktile", "u32.ktile", "{v}", [str(v) for v in (1, 2, 3, 4, 8, 16, 32, 64, 100, 256, 1000, 4096)], "?_ktile: a bare literal")
    R("own.u32_ktile", "u32.ktile", [], "1")
    R("own.u32_blk_row", "u32.blk_row", [], "1", "the domain's smallest value")
    R("own.u32_blk_col", "u32.blk_col", [], "1", "the domain's smallest value")

    # ---- the statement rungs
    for x, d, k in rungs("LCOW"):
        s, b = S(x, d, k), B(x, d, k)
        sfx = f"_{x}{d}_f{k}"
        R("block.one" + sfx, b, [s if x != "O" else S("O", d, k)], "{0}")
        if x == "O":
            lead = S("C", d, k)
            R("block.seq2" + sfx, b, [lead, s], "{0} {1}")
            R("block.seq3" + sfx, b, [lead, lead, s], "{0} {1} {2}")
            R("block.seq4" + sfx, b, [lead, lead, lead, s], "{0} {1} {2} {3}")
        else:
            R("block.seq2" + sfx, b, [s, s], "{0} {1}")
            R("block.seq3" + sfx, b, [s, s, s], "{0} {1} {2}")
            R("block.seq4" + sfx, b, [s, s, s, s], "{0} {1} {2} {3}")
        R("stmt.block" + sfx, s, [b], "{ {0} }")
        if x == "O":
            R("own.block" + sfx, b, [], own_store_text(iface), "the value-neutral store of the primary output's first cell")
            R("own.stmt" + sfx, s, [], own_store_text(iface), "the value-neutral store")
        else:
            R("own.block" + sfx, b, [], "{}", "an empty sequence")
            R("own.stmt" + sfx, s, [], "{}", "an empty statement")
        if k >= 1:
            R("stmt.lift" + sfx, s, [S(x, d, k - 1)], "{0}", "a statement of a lower stratum stands in a higher one")
        if x == "L":
            R("stmt.if" + sfx, s, ["bool.flag", B("C", d, k)], "if ({0}) { {1} }", "the body is C: a barrier under a divergent condition is unspellable")
        if x == "C":
            R("stmt.if" + sfx, s, ["bool.flag", B("C", d, k)], "if ({0}) { {1} }", "nesting stays conditional")
        if x == "O":
            R("stmt.if_else" + sfx, s, ["bool.flag", b, b], "if ({0}) { {1} } else { {2} }", "both branches are epilogues so a store happens whichever is taken")
        if x == "W":
            R("stmt.if" + sfx, s, ["bool.flagw", b], "if ({0}) { {1} }", "the body stays W")
            R("stmt.if_else" + sfx, s, ["bool.flagw", b, b], "if ({0}) { {1} } else { {2} }")
        if x in "LCO":
            for o in iface["outputs"]:
                R(f"store.{o['name']}{sfx}", s, [f"u32.threadrow_{o['name']}", f"u32.threadcol_{o['name']}", o["ty"]], store_text(iface, o),
                  "the store takes a row and a column of this invocation's block of the output")
    for d in range(1, 6):
        for k in STRATA:
            R(f"own.block_B{d}_f{k}", BB(d, k), [], "{}", "an empty loop body")
            R(f"own.block_WB{d}_f{k}", WBB(d, k), [], "{}", "an empty write-phase loop body")
            R(f"block.brk_B{d}_f{k}", BB(d, k), [B("C", d, k), "bool.flag", B("C", d, k)], "{0} if ({1}) { break; } {2}", "both parts are C blocks: no barrier before or after the break")
            R(f"block.cont_B{d}_f{k}", BB(d, k), ["bool.flag", B("C", d, k)], "if ({0}) { continue; } {1}", "continue lives only in the divergent barrier-free loop bodies")
            R(f"block.brk_WB{d}_f{k}", WBB(d, k), [B("W", d, k), "bool.flagw", B("W", d, k)], "{0} if ({1}) { break; } {2}")
            R(f"block.cont_WB{d}_f{k}", WBB(d, k), ["bool.flagw", B("W", d, k)], "if ({0}) { continue; } {1}")

    # ---- ownership rows, per output binding
    for o in iface["outputs"]:
        b = o["name"]
        R(f"safe.thread_row_{b}", f"u32.rowbase_{b}", ["vec3<u32>.gid"], "({0}.y * KNOB_BROW)", "the origin of the invocation's block on the row axis")
        R(f"safe.thread_col_{b}", f"u32.colbase_{b}", ["vec3<u32>.gid"], "({0}.x * KNOB_BCOL)")
        R(f"safe.row_plus_{b}", f"u32.threadrow_{b}", [f"u32.rowbase_{b}", "u32.lit64"], "({0} + min({1}, KNOB_BROW - 1u))", "an origin plus an offset clamped into the block")
        R(f"safe.col_plus_{b}", f"u32.threadcol_{b}", [f"u32.colbase_{b}", "u32.lit64"], "({0} + min({1}, KNOB_BCOL - 1u))")
        R(f"safe.rowbase_as_row_{b}", f"u32.threadrow_{b}", [f"u32.rowbase_{b}"], "{0}", "the origin is the block's first row")
        R(f"safe.colbase_as_col_{b}", f"u32.threadcol_{b}", [f"u32.colbase_{b}"], "{0}")
        R(f"safe.row_as_index_{b}", "u32.index", [f"u32.threadrow_{b}"], "{0}", "an owned row used as a plain position; nothing goes the other way", twin=True)
        R(f"safe.col_as_index_{b}", "u32.index", [f"u32.threadcol_{b}"], "{0}", twin=True)
        R(f"own.u32_rowbase_{b}", f"u32.rowbase_{b}", [], "(gid.y * KNOB_BROW)", "the invocation's own origin")
        R(f"own.u32_colbase_{b}", f"u32.colbase_{b}", [], "(gid.x * KNOB_BCOL)")
        R(f"own.u32_threadrow_{b}", f"u32.threadrow_{b}", [], "(gid.y * KNOB_BROW)", "the block's first row")
        R(f"own.u32_threadcol_{b}", f"u32.threadcol_{b}", [], "(gid.x * KNOB_BCOL)")
    C("literal.lit64", "u32.lit64", "lit_u32({v}u)", [str(i) for i in range(1, 65)], "?_lit64")
    R("builtin.global_invocation_id", "vec3<u32>.gid", [], "gid", "the only producer of vec3<u32>.gid")
    C("literal.flag", "bool.flag", "lit_bool({v})", ["true", "false"], "?_flag", twin=True)
    R("own.u32_rowid", "u32.rowid", [], "gid.y")
    R("own.u32_colid", "u32.colid", [], "gid.x")
