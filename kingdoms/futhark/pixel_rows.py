"""THE PIXEL FAMILY ON THE GENE FRAME'S FOUR ROW KINDS (fuller docs/PLAN_futhark_kingdom.md 7.7 K4).

A picture is a map over the pixels of a scalar program: one compute invocation per pixel reads the pixel's position
and a block of uniforms, runs bounded loops over f32 values, and stores one colour. In Futhark

    map (\\yi -> map (\\xi -> <the pixel's program>) (iota cols)) (iota rows)

where the pixel's program is a chain of `let`s and bounded `loop`s ending in a colour `[r, g, b]`.

The types are SHAPES only (design: types are shapes, scope is context):

    image   the picture, [rows][cols][3]f32 (the root)
    blk     a pixel block: lets and loops ending in the colour
    body1, body3, body5   a block or loop body ending in a 1-, 3- or 5-tuple (the loop's state arity)
    f32     a scalar
    count   a loop's trip count

Scope is the environment: `let`, `block` and `loop` bind f32 variables into their argument's environment by `var.f32`
Flow rows, and the one variable terminal `var@f32` names an in-scope variable (the entry's parameters, the pixel's
position `px`, `py` and every binding above the node), so no gene names an unbound variable. A loop's depth is the
environment attribute `trips`: a loop row binds `env.trips + 1` into its body and is legal only below the deepest
nest (`guard ... must env.trips < PIXEL_DEPTH`, repaired by the type's constant), so the work of a pixel is at most
`max(count) ** PIXEL_DEPTH` trips of a body whose size the gene bounds.

The generic part (the operators, the counts, the depth) names no shader; an INSTANCE adds its interface (the WGSL
source's uniform fields, each an entry parameter) and its literals (the f32 pool). Written by `gen_tables.py` (the
pixel family's section) into the master tables as the kingdom `futhark.pixel.<instance>`.
"""
from __future__ import annotations

# The loop state arities: a scalar, a vec3, five lanes.
PIXEL_ARITIES = (1, 3, 5)
# The deepest loop nest.
PIXEL_DEPTH = 2
# The trip counts a loop may take.
PIXEL_COUNTS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64)

# The scalar operators: (name, arity, spelling). Every argument is parenthesised, so a spelling never depends on the
# precedence of its arguments' spellings.
PIXEL_OPERATORS = [
    ("add", 2, "({0}) + ({1})"),
    ("sub", 2, "({0}) - ({1})"),
    ("mul", 2, "({0}) * ({1})"),
    ("div", 2, "({0}) / ({1})"),
    ("max", 2, "f32.max ({0}) ({1})"),
    ("min", 2, "f32.min ({0}) ({1})"),
    ("pow", 2, "({0}) ** ({1})"),
    ("mod", 2, "({0}) % ({1})"),
    ("neg", 1, "-({0})"),
    ("abs", 1, "f32.abs ({0})"),
    ("sqrt", 1, "f32.sqrt ({0})"),
    ("sin", 1, "f32.sin ({0})"),
    ("cos", 1, "f32.cos ({0})"),
    ("tan", 1, "f32.tan ({0})"),
    ("tanh", 1, "f32.tanh ({0})"),
    ("exp", 1, "f32.exp ({0})"),
    ("log", 1, "f32.log ({0})"),
    ("floor", 1, "f32.floor ({0})"),
    ("trunc", 1, "f32.trunc ({0})"),
    ("sel", 3, "if ({0}) >= 0f32 then ({1}) else ({2})"),
]

IMAGE, BLK, F32, COUNT = "image", "blk", "f32", "count"


def body(s):
    return f"body{s}"


# The program header the renderer wraps the picture in reads the tile from the input vector: `col0`, `row0` (f32),
# `cols`, `rows` (i64), `u_width`, `u_height`, then one `u_<field>` per interface input. The picture binds the pixel's
# position `px`, `py` (f32: the branch compiler's WGSL has no f32-to-i64 conversion inside a kernel).
IMAGE_SPELLING = ("map (\\yi -> map (\\xi -> let {0.v0} = col0 + f32.i64 xi in let {0.v1} = row0 + f32.i64 yi in {0}) "
                  "(iota cols)) (iota rows)")
BLACK_IMAGE = "map (\\_ -> map (\\_ -> [0f32, 0f32, 0f32]) (iota cols)) (iota rows)"


def named(problem, inputs, literals, image=False):
    """The named text form of one instance (`write_rows.rows_of_named` reads it): `inputs` are the interface's
    spellings (`u_time`, ...), `literals` the f32 pool spelled as Futhark literals."""
    out = []
    add = out.append
    # ---- data types (the root first) and their constants (a repair's `{const}` and a demanded type's last resort) --
    add(f"data_type\t{IMAGE}\tscalar")
    add(f"constant\t{IMAGE}\t{BLACK_IMAGE}\t-")
    add(f"data_type\t{BLK}\tscalar")
    add(f"constant\t{BLK}\t[0f32, 0f32, 0f32]\t-")
    for s in PIXEL_ARITIES:
        add(f"data_type\t{body(s)}\tscalar")
        add(f"constant\t{body(s)}\t({', '.join(['0f32'] * s)})\t-")
    add(f"data_type\t{F32}\tscalar")
    add(f"constant\t{F32}\t0f32\t-")
    add(f"data_type\t{COUNT}\tscalar")
    add(f"constant\t{COUNT}\t1\t-")

    symbols, spellings, arguments, flows, guards, repairs = [], [], [], [], [], []

    def symbol(name, role, spelling, out_ty, args=(), pool=None):
        symbols.append(f"symbol\t{name}\t-\t{role}\t-\t-")
        spellings.append(f"spelling\t{name}\t\t{spelling}")
        if pool is not None:
            spellings.append(f"spelling\t{name}\tpool\t{','.join(pool)}")
        for i, a in enumerate(args):
            arguments.append(f"argument\t{name}\t{i}\tin\t{a}\t-")
        arguments.append(f"argument\t{name}\t0\tout\t{out_ty}\t-")

    def binds(name, at, n, stem):
        """`n` f32 variables bound into argument `at`'s environment, spelled `<stem><depth>_<each>`."""
        flows.append(f"flow\t{name}\tin{at}\tdown\tvar.f32\t{n}")
        flows.append(f"flow\t{name}\tin{at}\tdown\tvar.f32.spell\t{stem}{{depth}}_{{each}}{{nth}}")

    def loop_guard(name):
        guards.append(f"guard\t{name}\t0\tmust\tenv.trips < {PIXEL_DEPTH}")
        repairs.append(f"repair\t{name}\t0\tnode\ttrue\ttrue\t{{const}}")

    # ---- the picture ----
    symbol("image", "function", IMAGE_SPELLING, IMAGE, [BLK])
    # The pixel's position: two groups of one variable each, so each has its own fixed name.
    flows.append("flow\timage\tin0\tdown\tvar.f32\t1")
    flows.append("flow\timage\tin0\tdown\tvar.f32.spell\tpx")
    flows.append("flow\timage\tin0\tdown\tvar.f32/1\t1")
    flows.append("flow\timage\tin0\tdown\tvar.f32/1.spell\tpy")
    symbol("image.black", "function", BLACK_IMAGE, IMAGE)

    # ---- the binders of a block (ending in the colour) and of a body (ending in an s-tuple) ----
    def binders(out_ty, tag):
        let = f"let@{tag}"
        symbol(let, "function", "let {1.v0} = {0} in {1}", out_ty, [F32, out_ty])
        binds(let, 1, 1, "v")
        for s in PIXEL_ARITIES:
            # A block: a let-chain of its own returning s values; its own bindings end with it.
            blk = f"block{s}@{tag}"
            state = ", ".join(f"{{1.v{i}}}" for i in range(s))
            symbol(blk, "function", f"let ({state}) = {{0}} in {{1}}", out_ty, [body(s), out_ty])
            binds(blk, 1, s, "b")
        for s in PIXEL_ARITIES:
            # A loop of state arity s: inits 0..s-1, the count s, the body s+1 (the state and its counter bound), the
            # rest s+2 (the final state bound).
            lp = f"loop{s}@{tag}"
            rest = ", ".join(f"{{{s + 2}.v{i}}}" for i in range(s))
            state = ", ".join(f"{{{s + 1}.v{i}}}" for i in range(s))
            inits = ", ".join(f"{{{i}}}" for i in range(s))
            spelling = (f"let ({rest}) = loop ({state}) = ({inits}) for i < {{{s}}} do "
                        f"let {{{s + 1}.v{s}}} = f32.i64 i in {{{s + 1}}} in {{{s + 2}}}")
            symbol(lp, "function", spelling, out_ty, [F32] * s + [COUNT, body(s), out_ty])
            binds(lp, s + 1, s + 1, "s")
            flows.append(f"flow\t{lp}\tin{s + 1}\tdown\tenv.trips\tenv.trips + 1")
            binds(lp, s + 2, s, "r")
            loop_guard(lp)

    binders(BLK, BLK)
    symbol("rgb", "function", "[{0}, {1}, {2}]", BLK, [F32, F32, F32])
    if image:
        # An image target's block terminal is the position's colour, not black: a random program whose root block is
        # a terminal then draws a gradient.
        symbol("coord", "function", "[px / u_width, py / u_height, 0.5f32]", BLK)
    else:
        symbol("black", "function", "[0f32, 0f32, 0f32]", BLK)
    for s in PIXEL_ARITIES:
        binders(body(s), body(s))
        symbol(f"tuple@{body(s)}", "function", "(" + ", ".join(f"{{{i}}}" for i in range(s)) + ")", body(s), [F32] * s)
        symbol(f"zero@{body(s)}", "function", "(" + ", ".join(["0f32"] * s) + ")", body(s))

    # ---- the scalars ----
    for name, arity, spelling in PIXEL_OPERATORS:
        symbol(name, "function", spelling, F32, [F32] * arity)
    if image:
        # An image target also reads the position as a fraction of the picture: nx, ny in [0, 1).
        symbol("nx", "function", "(px / u_width)", F32)
        symbol("ny", "function", "(py / u_height)", F32)
    symbol("var@f32", "variable", "{v}", F32)
    symbol("?@f32", "constant", "{v}", F32, pool=literals)
    symbol("?@count", "constant", "{v}", COUNT, pool=[str(c) for c in PIXEL_COUNTS])

    out += symbols + spellings + arguments + flows + guards + repairs
    # ---- the entry: the picture of the interface's inputs ----
    out.append(f"entry\t{problem}\t{IMAGE}")
    for p in ["u_width", "u_height"] + list(inputs):
        out.append(f"parameter\t{p}\t{F32}\t-\t-\t-\t-\t-")
    return "\n".join(out) + "\n"
