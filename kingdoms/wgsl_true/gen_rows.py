#!/usr/bin/env python3
"""THE WGSL KINGDOM AS ROWS of the gene frame's fixed schema (docs/DESIGN_geneframe.md 3.3 and 8.4;
docs/PLAN_futhark_kingdom.md 7.7, task K3): the Gram interface (`phi` in, `h` out, `cfg { d, m, r_rows, r_off }`).

    python3 -I gen_rows.py            # writes kingdom `wgsl` (variant `wgsl.gram`) into kingdoms/master/ through write_rows.py
    python3 -I gen_rows.py --check    # the committed warehouse rows equal a fresh generation (a Rust test runs this)

Every legality rule of WGSL that the old table carried in a TYPE (305 duals: ownership types, statement rungs, fuel
strata, masked index types) is a PROPERTY here, moved by Flow rows and enforced by `must` Guard rows with their repairs
inline, so the kingdom has nine data types:

| data type | properties | what carries the rule |
|---|---|---|
| `kernel` | | the root; its spelling ends in the primary output's store (the old `O` rung) |
| `stmt` | | a statement; its constant `{}` is the empty block, the repair that drops a statement |
| `u32` | `bound`, `uniform`, `owned` | `owned`: injective in the invocation, so a store through it is one thread's (the old ownership types); `uniform`: the same in every invocation (barriers); `bound`: the exclusive upper bound (index clamps) |
| `f32`, `bool` | `uniform` | |
| `vecf` | `width`, `uniform` | the vector width, fitted by swizzle or extension where two operands differ |
| `count` | `value` | a literal trip count or stride: terminals `c1` .. `c64`, so the work bound is decided at decode |
| `grid` | | the workgroup shape, a terminal |
| `buf` | `len`, `access` | a storage buffer: `phi` (`read`) and `h` (`read_write`), the entry's variables |

Scope is context (design 8.5): `let` bindings, the loop counter and the entry's builtins are variables chosen by
`var.T`'s index word in the node's environment, never folded into a type. The work bound is the environment attribute
`work` (the product of the trip counts of the enclosing loops, root 1): `guard loop: env.work * in0.value <= BUDGET must
-> {const}`, a safety rule (a runaway random kernel starved the display server on 2026-10-21). Uniform control flow is
the attribute `uniform` (root true), lowered by `if` on a non-uniform condition: `guard barrier: env.uniform must ->
{const}`.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "master"))
import write_rows  # noqa: E402  (kingdoms/master/write_rows.py: the one set of tables every kingdom lives in)

KINGDOM = "wgsl"
VARIANT = "wgsl.gram"
DESCRIPTION = "the true-GEP WGSL kingdom on the four-row schema: nine data types, legality by Flow and Guard rows (owned, uniform, bound, width, work), scope by var.T"
VARIANT_DESCRIPTION = "the WGSL kingdom's Gram problem: phi in, h out, cfg { d, m, r_rows, r_off }"

# The per-invocation work bound: the product of the trip counts of the loops around a node. The Gram interface's old
# fuel budget was K = 1, one stratum of 2^6 = 64 trips, and this is the same bound: one loop of 64, or two nested of 8;
# a loop of 128 is over it and dropped.
BUDGET = 64
# The workgroup array's length: one f32 slot per invocation of the largest workgroup shape (64), indexed by the local
# invocation index, so a store through a `local_owned` index is one invocation's.
WG_WORDS = 64

# The Gram interface's bindings (interface.py GRAM) and its workgroup memory, the kernel's prelude: text added once to a
# program that uses `main`.
PRELUDE = (
    "struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };\\n"
    "@group(0) @binding(0) var<storage, read> phi: array<f32>;\\n"
    "@group(0) @binding(1) var<storage, read_write> h: array<f32>;\\n"
    "@group(0) @binding(2) var<uniform> cfg: Cfg;\\n"
    f"var<workgroup> wg_f: array<f32, {WG_WORDS}>;\\n"
)

# The kernel row's spelling: the grid's constants, the entry point, the builtins the entry's parameters name, the
# registers, the body, and the epilogue that always stores the primary output (one owned element per invocation).
KERNEL = (
    "{0} @compute @workgroup_size(WG_X, WG_Y, 1) "
    "fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, "
    "@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {{ "
    "let gid_x: u32 = gid.x; let gid_y: u32 = gid.y; let lid_x: u32 = lid.x; let lid_y: u32 = lid.y; "
    "let wid_x: u32 = wid.x; let wid_y: u32 = wid.y; "
    "let lin: u32 = gid.y * (nwg.x * WG_X) + gid.x; "
    "let cfg_d: u32 = cfg.d; let cfg_m: u32 = cfg.m; let cfg_r_rows: u32 = cfg.r_rows; let cfg_r_off: u32 = cfg.r_off; "
    "var rf0: f32 = 0.0; var rf1: f32 = 0.0; var rf2: f32 = 0.0; var rf3: f32 = 0.0; "
    "{1} "
    "if (gid.x < cfg.d && gid.y < cfg.r_rows) {{ h[gid.y * cfg.d + gid.x] = rf0; }} }}"
)

GRIDS = [("g8x8", 8, 8), ("g16x16", 16, 16), ("g32x2", 32, 2), ("g64x1", 64, 1)]
COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
REGISTERS = 4

BOTH_UNIFORM = "in0.uniform and in1.uniform"
# The one way a sum stays injective in the invocation: an owned term plus a term every invocation shares.
INJECTIVE_SUM = "(in0.owned and in1.uniform) or (in1.owned and in0.uniform)"
LOCAL_INJECTIVE_SUM = "(in0.local_owned and in1.uniform) or (in1.local_owned and in0.uniform)"
CLAMP = "min({1}, arrayLength(&{0}) - 1u)"
# The width fit of vector operand 1 to operand 0's width: a swizzle where it is wider, an extension by zeros where
# narrower (design 8.4, the list `vfit`).
VFIT = ("{if in1.width < in0.width : vec{=in0.width}<f32>({1}, {for i < in0.width - in1.width sep , : 0.0}) "
        "else ({1}).{=at(list('x', 'xy', 'xyz', 'xyzw'), in0.width - 1, 'x')}}")


class Table:
    def __init__(self):
        self.lines: list[str] = []
        self.symbols: set[str] = set()

    def row(self, *fields):
        self.lines.append("\t".join(str(f) for f in fields))

    # -- the lists that name things --
    def data_type(self, name, carries=(), constant=None):
        self.row("data_type", name, "scalar")
        for p in carries:
            self.row("carries", name, p)
        if constant is not None:
            spelling, props = constant
            self.row("constant", name, spelling, props or "-")

    def symbol(self, name, role="function", adapter=False, prelude=None, spelling=None, forms=()):
        if name in self.symbols:
            raise SystemExit(f"our software has a bug: the symbol {name} is declared twice")
        self.symbols.add(name)
        self.row("symbol", name, "-", role, "adapter" if adapter else "-", prelude or "-")
        if spelling is not None:
            self.row("spelling", name, "", spelling)
        for form, text in forms:
            self.row("spelling", name, form, text)

    # -- the four kinds of row --
    def function(self, name, ins, out, spelling, ups=(), downs=(), guards=(), prelude=None):
        """One function with its Argument rows, up flows `(property, formula)`, down flows `(k, name, formula)` and
        guards `(condition, [repairs])`, each repair `(site, when, guard, template)`."""
        self.symbol(name, spelling=spelling, prelude=prelude)
        for k, t in enumerate(ins):
            self.row("argument", name, k, "in", t, "-")
        self.row("argument", name, 0, "out", out, "-")
        for p, f in ups:
            self.row("flow", name, "out", "up", p, f)
        for k, n, f in downs:
            self.row("flow", name, f"in{k}", "down", n, f)
        for order, (cond, repairs) in enumerate(guards):
            self.row("guard", name, order, "must", cond)
            for site, when, guard, template in repairs:
                self.row("repair", name, order, site, when, guard, template)

    def terminal(self, name, out, spelling, ups=(), prelude=None):
        self.function(name, [], out, spelling, ups, prelude=prelude)

    def constant(self, ty, pool, ups=()):
        """The Dc constant of `ty` with its pool of literals and the properties every literal has (its up flows: a Dc
        symbol's properties are its flows', not the data type's constant's, which is `{const}`'s)."""
        self.symbol(f"?@{ty}", role="constant", spelling="{v}", forms=[("pool", ",".join(pool))])
        self.row("argument", f"?@{ty}", 0, "out", ty, "-")
        for p, f in ups:
            self.row("flow", f"?@{ty}", "out", "up", p, f)

    def variable(self, ty):
        self.symbol(f"var@{ty}", role="variable")
        self.row("argument", f"var@{ty}", 0, "out", ty, "-")


DROP = [("node", "true", "true", "{const}")]


def build() -> str:
    t = Table()
    # ---- the data types and what they carry ----
    t.data_type("kernel")
    t.data_type("stmt", constant=("{}", None))
    t.data_type("u32", ["bound", "uniform", "owned", "local_owned"], constant=("0u", "uniform=true;owned=false;local_owned=false"))
    t.data_type("f32", ["uniform"], constant=("0.0", "uniform=true"))
    t.data_type("bool", ["uniform"], constant=("false", "uniform=true"))
    t.data_type("vecf", ["width", "uniform"], constant=("vec4<f32>(0.0)", "width=4;uniform=true"))
    t.data_type("count", ["value"])
    t.data_type("grid")
    t.data_type("buf", ["len", "access"])

    # ---- the kernel ----
    # the prelude is the kernel's: added once to a program that uses `main` (every program does)
    t.function("main", ["grid", "stmt"], "kernel", KERNEL, prelude=PRELUDE)
    # every type the root reaches needs a terminal (the frame's tail): the empty kernel, 8 x 8, the epilogue alone
    t.terminal("empty", "kernel", KERNEL.replace("{0}", "const WG_X: u32 = 8u; const WG_Y: u32 = 8u;").replace("{1}", ""), prelude=PRELUDE)
    for name, x, y in GRIDS:
        t.terminal(name, "grid", f"const WG_X: u32 = {x}u; const WG_Y: u32 = {y}u;")

    # ---- statements ----
    t.function("seq", ["stmt", "stmt"], "stmt", "{0} {1}")
    t.function("let_u", ["u32", "stmt"], "stmt", "{{ let {1.v0}: u32 = {0}; {1} }}",
               downs=[(1, "var.u32", "1"), (1, "var.u32.bound", "in0.bound"), (1, "var.u32.uniform", "in0.uniform"), (1, "var.u32.owned", "in0.owned"), (1, "var.u32.local_owned", "in0.local_owned"), (1, "var.u32.spell", "u{depth}{nth}")])
    t.function("let_f", ["f32", "stmt"], "stmt", "{{ let {1.v0}: f32 = {0}; {1} }}",
               downs=[(1, "var.f32", "1"), (1, "var.f32.uniform", "in0.uniform"), (1, "var.f32.spell", "f{depth}{nth}")])
    t.function("let_v", ["vecf", "stmt"], "stmt", "{{ let {1.v0}: vec{=in0.width}<f32> = {0}; {1} }}",
               downs=[(1, "var.vecf", "1"), (1, "var.vecf.width", "in0.width"), (1, "var.vecf.uniform", "in0.uniform"), (1, "var.vecf.spell", "v{depth}{nth}")])
    for r in range(REGISTERS):
        t.function(f"set_rf{r}", ["f32"], "stmt", f"rf{r} = {{0}};")
    t.function("store", ["buf", "u32", "f32"], "stmt", "{0}[{1}] = {2};",
               guards=[("in0.access = 'read_write'", DROP),
                       ("in1.owned", DROP),
                       ("in1.bound <= in0.len", [("in1", "true", "true", CLAMP)] + DROP)])
    t.function("if_", ["bool", "stmt", "stmt"], "stmt", "if ({0}) {{ {1} }} else {{ {2} }}",
               downs=[(1, "env.uniform", "env.uniform and in0.uniform"), (2, "env.uniform", "env.uniform and in0.uniform")])
    t.function("loop", ["count", "stmt"], "stmt", "for (var {1.v0}: u32 = 0u; {1.v0} < {0}u; {1.v0}++) {{ {1} }}",
               downs=[(1, "var.u32", "1"), (1, "var.u32.bound", "in0.value"), (1, "var.u32.uniform", "true"), (1, "var.u32.owned", "false"), (1, "var.u32.local_owned", "false"), (1, "var.u32.spell", "i{depth}{nth}"),
                      (1, "env.work", "env.work * in0.value")],
               guards=[(f"env.work * in0.value <= {BUDGET}", DROP)])
    t.function("barrier", [], "stmt", "workgroupBarrier();", guards=[("env.uniform", DROP)])
    # workgroup memory: a store needs an index that is one invocation's within the workgroup (`local_owned`, injective in
    # the local invocation index) and within the array (`bound <= WG_WORDS`, else clamped); a load needs the index in range
    t.function("wg_store", ["u32", "f32"], "stmt", "wg_f[{0}] = {1};",
               guards=[("in0.local_owned", DROP),
                       (f"in0.bound <= {WG_WORDS}", [("in0", "true", "true", f"min({{0}}, {WG_WORDS - 1}u)")] + DROP)])

    # ---- u32 ----
    t.function("add_u", ["u32", "u32"], "u32", "({0} + {1})", ups=[("bound", "in0.bound + in1.bound"), ("uniform", BOTH_UNIFORM), ("owned", INJECTIVE_SUM), ("local_owned", LOCAL_INJECTIVE_SUM)])
    t.function("mul_uc", ["u32", "count"], "u32", "({0} * {1}u)", ups=[("bound", "in0.bound * in1.value"), ("uniform", "in0.uniform"), ("owned", "in0.owned"), ("local_owned", "in0.local_owned")])
    t.function("mod_uc", ["u32", "count"], "u32", "({0} % {1}u)", ups=[("bound", "in1.value"), ("uniform", "in0.uniform"), ("owned", "false"), ("local_owned", "false")])
    t.function("min_u", ["u32", "u32"], "u32", "min({0}, {1})", ups=[("bound", "min(in0.bound, in1.bound)"), ("uniform", BOTH_UNIFORM), ("owned", "false"), ("local_owned", "false")])
    t.constant("u32", ["0u", "1u", "2u", "3u", "4u", "8u", "16u", "32u"], ups=[("uniform", "true"), ("owned", "false"), ("local_owned", "false")])
    t.variable("u32")

    # ---- f32 ----
    for name, text in [("add_f", "({0} + {1})"), ("sub_f", "({0} - {1})"), ("mul_f", "({0} * {1})"), ("max_f", "max({0}, {1})")]:
        t.function(name, ["f32", "f32"], "f32", text, ups=[("uniform", BOTH_UNIFORM)])
    t.function("fma_f", ["f32", "f32", "f32"], "f32", "fma({0}, {1}, {2})", ups=[("uniform", "in0.uniform and in1.uniform and in2.uniform")])
    for name, text in [("sqrt_f", "sqrt({0})"), ("exp_f", "exp({0})")]:
        t.function(name, ["f32"], "f32", text, ups=[("uniform", "in0.uniform")])
    t.function("f32_u", ["u32"], "f32", "f32({0})", ups=[("uniform", "in0.uniform")])
    t.function("load", ["buf", "u32"], "f32", "{0}[{1}]", ups=[("uniform", "false")],
               guards=[("in1.bound <= in0.len", [("in1", "true", "true", CLAMP)] + DROP)])
    t.function("wg_load", ["u32"], "f32", "wg_f[{0}]", ups=[("uniform", "false")],
               guards=[(f"in0.bound <= {WG_WORDS}", [("in0", "true", "true", f"min({{0}}, {WG_WORDS - 1}u)")] + DROP)])
    t.function("dot", ["vecf", "vecf"], "f32", "dot({0}, {1})", ups=[("uniform", BOTH_UNIFORM)],
               guards=[("in1.width = in0.width", [("in1", "true", "true", VFIT)] + DROP)])
    for r in range(REGISTERS):
        t.terminal(f"rf{r}", "f32", f"rf{r}", ups=[("uniform", "false")])
    t.constant("f32", ["0.0", "1.0", "0.5", "2.0", "-1.0", "0.001"], ups=[("uniform", "true")])
    t.variable("f32")

    # ---- bool ----
    t.function("lt_u", ["u32", "u32"], "bool", "({0} < {1})", ups=[("uniform", BOTH_UNIFORM)])
    t.function("lt_f", ["f32", "f32"], "bool", "({0} < {1})", ups=[("uniform", BOTH_UNIFORM)])
    t.function("and_b", ["bool", "bool"], "bool", "({0} && {1})", ups=[("uniform", BOTH_UNIFORM)])
    t.function("not_b", ["bool"], "bool", "(!{0})", ups=[("uniform", "in0.uniform")])
    t.constant("bool", ["true", "false"], ups=[("uniform", "true")])

    # ---- vecf ----
    t.function("v2", ["f32", "f32"], "vecf", "vec2<f32>({0}, {1})", ups=[("width", "2"), ("uniform", BOTH_UNIFORM)])
    t.function("v4", ["f32", "f32", "f32", "f32"], "vecf", "vec4<f32>({0}, {1}, {2}, {3})", ups=[("width", "4"), ("uniform", "in0.uniform and in1.uniform and in2.uniform and in3.uniform")])
    t.function("vadd", ["vecf", "vecf"], "vecf", "({0} + {1})", ups=[("width", "in0.width"), ("uniform", BOTH_UNIFORM)],
               guards=[("in1.width = in0.width", [("in1", "true", "true", VFIT), ("node", "true", "true", "{in.own}")])])
    t.function("vscale", ["vecf", "f32"], "vecf", "({0} * {1})", ups=[("width", "in0.width"), ("uniform", BOTH_UNIFORM)])
    t.variable("vecf")

    # ---- counts and buffers ----
    for c in COUNTS:
        t.terminal(f"c{c}", "count", str(c), ups=[("value", str(c))])
    t.variable("buf")

    # ---- the formula rows, last (the alphabet's ids are the contract's): how a formula's value is spelled in WGSL ----
    for name, text in [("=min", "min({0}, {1})"), ("=max", "max({0}, {1})"), ("=+", "{0} + {1}"), ("=*", "{0} * {1}"), ("=group", "({0})"), ("=negative", "-{0}")]:
        t.symbol(name, role="formula", spelling=text, forms=[("name", "{0}_{1}" if name in ("=min", "=max", "=+", "=*") else "{0}")])

    # ---- the entry: the Gram problem's root environment ----
    t.row("entry", "gram", "kernel")
    t.row("parameter", "phi", "buf", "access='read'", "-", "-", "-", "-")
    t.row("parameter", "h", "buf", "access='read_write'", "-", "-", "-", "-")
    t.row("parameter", "lin", "u32", "owned=true;local_owned=true;uniform=false", "-", "-", "-", "-")
    t.row("parameter", "lidx", "u32", f"owned=false;local_owned=true;uniform=false;bound={WG_WORDS}", "-", "-", "-", "-")
    for name in ["gid_x", "gid_y", "lid_x", "lid_y"]:
        t.row("parameter", name, "u32", "owned=false;local_owned=false;uniform=false", "-", "-", "-", "-")
    for name in ["wid_x", "wid_y", "cfg_d", "cfg_m", "cfg_r_rows", "cfg_r_off"]:
        t.row("parameter", name, "u32", "owned=false;local_owned=false;uniform=true", "-", "-", "-", "-")
    return "\n".join(t.lines) + "\n"


def main(argv):
    text = build()
    rows = write_rows.rows_of_named(text)
    kingdom_row = (VARIANT, KINGDOM, VARIANT_DESCRIPTION, write_rows.root_of(rows), "")
    parent_row = (KINGDOM, "", DESCRIPTION, "", "")
    if "--check" in argv:
        bad = write_rows.check_kingdom(KINGDOM, {}, parent_row) + write_rows.check_kingdom(VARIANT, rows, kingdom_row)
        for b in bad:
            print(b, file=sys.stderr)
        if bad:
            return 1
        print(f"kingdoms/master: {VARIANT} up to date ({len(text.splitlines())} rows)")
        return 0
    write_rows.write_kingdom(KINGDOM, {}, parent_row)
    write_rows.write_kingdom(VARIANT, rows, kingdom_row)
    print(f"wrote {VARIANT} into kingdoms/master ({len(text.splitlines())} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
