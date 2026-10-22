"""Batch 1: the kernel skeleton, the statement and block rungs, the epilogue's
store, the ownership rows (design section 9). Every statement and block type
carries a depth `d` (the loop ladder) and a fuel stratum `k` (batch 3)."""
from lib import *

AXES_COLS_X = "fn row_axis(g: vec3<u32>) -> u32 { return g.y; }\nfn col_axis(g: vec3<u32>) -> u32 { return g.x; }\n"
AXES_ROWS_X = "fn row_axis(g: vec3<u32>) -> u32 { return g.x; }\nfn col_axis(g: vec3<u32>) -> u32 { return g.y; }\n"
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
    if iface.get("module"):
        # s9 batch 8: a module of several entry points, each a normal gene rooted at its own entry type, joined by the
        # fixed homeotic linker of design section 6 (the module type is the linker's root)
        T("kernel.module", "a module of entry points: the fixed linker's root")
        for e, _ in iface["module"]:
            T(f"kernel.{e}", f"the entry point {e}: the root of its normal gene")
        T("u32.passes", "the // passes N count, a literal in the interface's declared range", 1, "1..=16")
        SAFE_ARGS.add("u32.passes")
    else:
        T("kernel.safe", "the entry point; the gene's root")
    T("grid.shape", "one of the target's legal workgroup shapes (x, y, 1): a literal from the target's table")
    T("u32.blk_row", "a block factor on the row axis: cells per invocation along it", 1, "1,2,4,8")
    T("u32.blk_col", "a block factor on the column axis", 1, "1,2,4,8")
    T("grid.grid", "the entry's workgroup shape and block")
    T("u32.ktile", "the tile-depth knob: a bare literal 1..=4096 (a const initialiser, so no knob reference or call may produce it)", 1, "1..=4096")
    for x, d, k in rungs("LCOWPSX"):
        T(S(x, d, k), f"one {x}-rung statement at depth {d}, fuel stratum {k}")
        T(B(x, d, k), f"a {x}-rung statement sequence at depth {d}, fuel stratum {k}")
    for d in range(1, 6):
        for k in STRATA:
            T(BB(d, k), f"a loop body that breaks, depth {d}, stratum {k}")
            T(WBB(d, k), f"a write-phase loop body that breaks, depth {d}, stratum {k}")
            if d in DEPTH["S"]:
                T(SBB(d, k), f"a solo-phase loop body that breaks, depth {d}, stratum {k}")
                T(XBB(d, k), f"a split-phase loop body that breaks, depth {d}, stratum {k}")
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
    if o.get("kind") == "texture":
        return "{ let r = {0}; let c = {1}; if (r < " + o["rows"] + " && c < " + o["cols"] + ") { textureStore(" + b + ", vec2<u32>(c, r), " + val + "); } }"
    return "{ let r = {0}; let c = {1}; if (r < " + o["rows"] + " && c < " + o["cols"] + ") { " + b + "[r * " + o["cols"] + " + c] = " + val + "; } }"


def own_store_text(iface):
    o = next(o for o in iface["outputs"] if o["name"] == iface["primary"])
    b = o["name"]
    if o.get("kind") == "texture":
        return "{ let r = row_axis(gid) * KNOB_BROW; let c = col_axis(gid) * KNOB_BCOL; if (r < " + o["rows"] + " && c < " + o["cols"] + ") { textureStore(" + b + ", vec2<u32>(c, r), vec4<f32>()); } }"
    idx = f"{b}[r * {o['cols']} + c]"
    return "{ let r = row_axis(gid) * KNOB_BROW; let c = col_axis(gid) * KNOB_BCOL; if (r < " + o["rows"] + " && c < " + o["cols"] + ") { " + idx + " = " + idx + "; } }"


def build(iface):
    helper("lit_u32", "fn lit_u32(x: u32) -> u32 { return x; }")
    helper("lit_i32", "fn lit_i32(x: i32) -> i32 { return x; }")
    helper("lit_f32", "fn lit_f32(x: f32) -> f32 { return x; }")
    helper("lit_bool", "fn lit_bool(x: bool) -> bool { return x; }")
    helper("lit_f16", "fn lit_f16(x: f16) -> f16 { return x; }")

    # A tiled dispatch (ST9, the image interfaces): the host splits the dispatch into submits of whole workgroups (the
    # work guard's bound of each submit at most its cap), and the interface's `dispatch_tile` uniform gives each submit's
    # origin in workgroups and the whole grid. `gid`, `wid` and `nwg` are then the ids of the whole dispatch, so every
    # row that reads them means what it means in one dispatch; the builtins of the submit are `gid_tile`, `wid_tile`,
    # `nwg_tile`. The prologue's pipeline-overridable constant `DISPATCH_TILED` (TILE_DECL, interface.py) is set by the
    # host at pipeline creation: false for a pass of one submit (origin 0, grid the submit's own), so the ids are the
    # builtins and Apple's compiler folds the tile uniform away (measured 2.9 % of volumepassage at 64 x 64, 2026-10-22).
    tiled = iface.get("tiled_dispatch", False)
    ids = (
        "@builtin(global_invocation_id) gid_tile: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid_tile: vec3<u32>, "
        "@builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg_tile: vec3<u32>"
        if tiled
        else "@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, "
        "@builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>"
    )
    entry = (
        "{0}\n@compute @workgroup_size(WG_X, WG_Y, 1)\nfn main(" + ids
        + (", @builtin(subgroup_size) sg_size: u32, @builtin(num_subgroups) sg_num: u32" if iface["has_subgroups"] else "")
        + ") {\n"
        + ("let wid: vec3<u32> = select(wid_tile, wid_tile + dispatch_tile.origin.xyz, DISPATCH_TILED);\n"
           "let gid: vec3<u32> = select(gid_tile, gid_tile + dispatch_tile.origin.xyz * vec3<u32>(WG_X, WG_Y, 1u), DISPATCH_TILED);\n"
           "let nwg: vec3<u32> = select(nwg_tile, dispatch_tile.grid.xyz, DISPATCH_TILED);\n" if tiled else "")
        + ("let sg_lane: u32 = lidx & (sg_size - 1u);\nlet sg_id: u32 = lidx / sg_size;\n" if iface["has_subgroups"] else "")
        + "<declarations>\n{1}\n{2}\n}\n"
    )
    if iface.get("module"):
        build_module(iface)
        return build_rest(iface)
    R("kernel.entry_safe", "kernel.safe", ["grid.grid", B("L", 0, K), B("O", 0, K)], entry,
      "the only producer of the root; a kernel is a grid, a body and an epilogue that ends in the store of its own cell")

    # ---- grids
    kn = lambda wcol, wrow, bcol, brow: (
        "const WG_X: u32 = {0:0}u;\nconst WG_Y: u32 = {0:1}u;\nconst KNOB_WCOL: u32 = " + wcol + ";\nconst KNOB_WROW: u32 = " + wrow + ";\n"
        "const KNOB_BCOL: u32 = " + bcol + ";\nconst KNOB_BROW: u32 = " + brow + ";\nconst KNOB_KTILE: u32 = {3}u;\n"
        "const KNOB_THREADS: u32 = {0:0}u * {0:1}u;\n")
    R("grid.of", "grid.grid", ["grid.shape", "u32.blk_col", "u32.blk_row", "u32.ktile"],
      "// grid {0:0} {0:1} {1} {2}\n" + AXES_COLS_X + kn("{0:0}u", "{0:1}u", "{1}u", "{2}u"),
      "the shape is in the target's limits by the table; the header and the attribute come from one node", fields=True)
    R("grid.of_rows_x", "grid.grid", ["grid.shape", "u32.blk_row", "u32.blk_col", "u32.ktile"],
      "// grid {0:0} {0:1} {1} {2} rows-x\n" + AXES_ROWS_X + kn("{0:1}u", "{0:0}u", "{2}u", "{1}u"),
      "the output's rows on thread x; the arguments in the order the spelling binds them", fields=True)
    C("literal.shape", "grid.shape", "{v}", shape_table(iface), "?_shape; its Dc domain is the table")
    C("literal.blk_row", "u32.blk_row", "{v}", ["1", "2", "4", "8"], "?_blk_row")
    C("literal.blk_col", "u32.blk_col", "{v}", ["1", "2", "4", "8"], "?_blk_col")
    gx, gy, bx, by = iface["default_grid"]
    R("own.grid_grid", "grid.grid", [], f"// grid {gx} {gy} {bx} {by}\n" + AXES_COLS_X + f"const WG_X: u32 = {gx}u;\nconst WG_Y: u32 = {gy}u;\nconst KNOB_WCOL: u32 = {gx}u;\nconst KNOB_WROW: u32 = {gy}u;\nconst KNOB_BCOL: u32 = {bx}u;\nconst KNOB_BROW: u32 = {by}u;\nconst KNOB_KTILE: u32 = 1u;\nconst KNOB_THREADS: u32 = {gx * gy}u;\n",
      "the interface's default grid: its reference kernel's shape and blocks, ktile 1")
    grid_literals(iface)
    build_rest(iface)


def grid_literals(iface):
    C("literal.ktile", "u32.ktile", "{v}", [str(v) for v in (1, 2, 3, 4, 8, 16, 32, 64, 100, 256, 1000, 4096)], "?_ktile: a bare literal")
    R("own.u32_ktile", "u32.ktile", [], "1")
    R("own.u32_blk_row", "u32.blk_row", [], "1", "the domain's smallest value")
    R("own.u32_blk_col", "u32.blk_col", [], "1", "the domain's smallest value")


MODULE_IDS = ("@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, "
              "@builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>")


def build_module(iface):
    """s9 batch 8: the entry points of a module. Each entry is the root of its own normal gene, `kernel.entry_<e>: (grid,
    [passes,] block.L0) -> kernel.<e>`; the fixed homeotic linker `kernel.module_<iface>: (kernel.<e1>, kernel.<e2>, ...)`
    joins them (design section 6: head `(link)`, tails the genic terminals in gene order, immutable). Two entries of one
    module cannot share module-scope knobs, so the grid's knobs are function-scope constants of the entry, spelled from the
    grid node's fields (`{0:F}`): the attribute and the knobs still come from one node. There is no epilogue: the
    interface has no primary output whose own cell every invocation must store (the v24 contract: every slot of `cands`
    starts as a pad, a slot not written stays a pad)."""
    helper("row_axis", "fn row_axis(g: vec3<u32>) -> u32 { return g.y; }")
    helper("col_axis", "fn col_axis(g: vec3<u32>) -> u32 { return g.x; }")
    knobs = ("const KNOB_WCOL: u32 = {0:0}u; const KNOB_WROW: u32 = {0:1}u; const KNOB_BCOL: u32 = {0:2}u; const KNOB_BROW: u32 = {0:3}u; "
             "const KNOB_KTILE: u32 = {0:4}u; const KNOB_THREADS: u32 = {0:0}u * {0:1}u;\n")
    entries = []
    for e, passes in iface["module"]:
        body = "{2}" if passes else "{1}"
        head = "// passes {1}\n" if passes else ""
        text = head + "@compute @workgroup_size({0:0}, {0:1}, 1)\nfn " + e + "(" + MODULE_IDS + ") {\n" + knobs + "<declarations>\n" + body + "\n}\n"
        args = ["grid.grid", "u32.passes", B("L", 0, K)] if passes else ["grid.grid", B("L", 0, K)]
        R(f"kernel.entry_{e}", f"kernel.{e}", args, text, f"the entry point {e}: a grid and a body (no epilogue: the module has no primary output)", fields=True)
        empty = ("// passes 1\n" if passes else "") + f"@compute @workgroup_size(1, 1, 1)\nfn {e}() {{\n}}\n"
        R(f"own.kernel_{e}", f"kernel.{e}", [], empty, "the empty entry: the terminal the linker's argument type needs (the kingdom check); a fixed linker never expresses it")
        entries.append(f"kernel.{e}")
    R(f"kernel.module_{iface['name']}", "kernel.module", entries, "\n".join("{%d}" % i for i in range(len(entries))) + "\n",
      "the fixed homeotic linker's link function: the module is its entries in gene order")
    R("grid.of", "grid.grid", ["grid.shape", "u32.blk_col", "u32.blk_row", "u32.ktile"], "{0:0} {0:1} {1} {2} {3}",
      "the entry's workgroup shape, blocks and tile depth as fields the entry spells", fields=True)
    C("literal.shape", "grid.shape", "{v}", shape_table(iface), "?_shape; its Dc domain is the table")
    C("literal.blk_row", "u32.blk_row", "{v}", ["1", "2", "4", "8"], "?_blk_row")
    C("literal.blk_col", "u32.blk_col", "{v}", ["1", "2", "4", "8"], "?_blk_col")
    gx, gy, bx, by = iface["default_grid"]
    R("own.grid_grid", "grid.grid", [], f"{gx} {gy} {bx} {by} 1", "the interface's default grid")
    lo, hi = iface.get("passes", (1, 16))
    C("literal.passes", "u32.passes", "{v}", [str(n) for n in range(lo, hi + 1)], "?_passes: the harness reads the count from the header")
    R("own.u32_passes", "u32.passes", [], str(lo), "the domain's smallest value")
    grid_literals(iface)


def build_rest(iface):
    # ---- the statement rungs
    for x, d, k in rungs("LCOWPSX"):
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
        if x in "WPX":
            R("stmt.if" + sfx, s, ["bool.flagw", b], "if ({0}) { {1} }", "the body stays " + x)
            R("stmt.if_else" + sfx, s, ["bool.flagw", b, b], "if ({0}) { {1} } else { {2} }")
        if x == "S":
            R("stmt.if" + sfx, s, ["bool.flag", b], "if ({0}) { {1} }", "the body stays S: one invocation")
            R("stmt.if_else" + sfx, s, ["bool.flag", b, b], "if ({0}) { {1} } else { {2} }")
        if x in "LC" and iface.get("module"):
            R("stmt.if_else" + sfx, s, ["bool.flag", B("C", d, k), B("C", d, k)], "if ({0}) { {1} } else { {2} }", "both branches conditional (s9 batch 8)")
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
            if d in DEPTH["S"]:
                R(f"own.block_SB{d}_f{k}", SBB(d, k), [], "{}", "an empty solo-phase loop body")
                R(f"block.brk_SB{d}_f{k}", SBB(d, k), [B("S", d, k), "bool.flag", B("S", d, k)], "{0} if ({1}) { break; } {2}")
                R(f"own.block_XB{d}_f{k}", XBB(d, k), [], "{}", "an empty split-phase loop body")
                R(f"block.brk_XB{d}_f{k}", XBB(d, k), [B("X", d, k), "bool.flagw", B("X", d, k)], "{0} if ({1}) { break; } {2}")

    # ---- ownership rows, per output binding
    for o in iface["outputs"]:
        b = o["name"]
        R(f"safe.thread_row_{b}", f"u32.rowbase_{b}", ["vec3<u32>.gid"], "(row_axis({0}) * KNOB_BROW)", "the origin of the invocation's block on the row axis")
        R(f"safe.thread_col_{b}", f"u32.colbase_{b}", ["vec3<u32>.gid"], "(col_axis({0}) * KNOB_BCOL)")
        R(f"safe.row_plus_{b}", f"u32.threadrow_{b}", [f"u32.rowbase_{b}", "u32.lit64"], "({0} + min({1}, KNOB_BROW - 1u))", "an origin plus an offset clamped into the block")
        R(f"safe.col_plus_{b}", f"u32.threadcol_{b}", [f"u32.colbase_{b}", "u32.lit64"], "({0} + min({1}, KNOB_BCOL - 1u))")
        R(f"safe.rowbase_as_row_{b}", f"u32.threadrow_{b}", [f"u32.rowbase_{b}"], "{0}", "the origin is the block's first row")
        R(f"safe.colbase_as_col_{b}", f"u32.threadcol_{b}", [f"u32.colbase_{b}"], "{0}")
        R(f"safe.row_as_index_{b}", "u32.index", [f"u32.threadrow_{b}"], "{0}", "an owned row used as a plain position; nothing goes the other way", twin=True)
        R(f"safe.col_as_index_{b}", "u32.index", [f"u32.threadcol_{b}"], "{0}", twin=True)
        R(f"own.u32_rowbase_{b}", f"u32.rowbase_{b}", [], "(row_axis(gid) * KNOB_BROW)", "the invocation's own origin")
        R(f"own.u32_colbase_{b}", f"u32.colbase_{b}", [], "(col_axis(gid) * KNOB_BCOL)")
        R(f"own.u32_threadrow_{b}", f"u32.threadrow_{b}", [], "(row_axis(gid) * KNOB_BROW)", "the block's first row")
        R(f"own.u32_threadcol_{b}", f"u32.threadcol_{b}", [], "(col_axis(gid) * KNOB_BCOL)")
    C("literal.lit64", "u32.lit64", "{v}u", [str(i) for i in range(1, 65)], "?_lit64: a bare literal; its consumers (a loop bound's assignment, an offset's min against a knob, the own-element's product with KNOB_THREADS, at most 64 * 1024) cannot fail if constant-evaluated")
    R("builtin.global_invocation_id", "vec3<u32>.gid", [], "gid", "the only producer of vec3<u32>.gid")
    C("literal.flag", "bool.flag", "lit_bool({v})", ["true", "false"], "?_flag", twin=True)
    R("own.u32_rowid", "u32.rowid", [], "row_axis(gid)")
    R("own.u32_colid", "u32.colid", [], "col_axis(gid)")
