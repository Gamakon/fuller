"""Batch 8 (s9, 2026-10-22): the design's own state and the row outputs of a module interface (v24).

Generated only for an interface that declares `arenas` or `row_outputs` (the Gram and image tables are unchanged):

* an **arena** (`policy arena`, batch 5): a binding the design keeps its own state in, read and written by any
  invocation. Every access is spelled with the device's own restriction of an index to the last element,
  `a[min(i, arrayLength(&a) - 1u)]`, so it is in range by the row's text and reads and writes exactly the cell the
  device would for the same index. Two invocations may store one cell: WGSL gives one of the values, and the arena
  policy (the contract's index state) accepts it. An atomic arena is declared `array<atomic<u32>>`: its plain load and
  store are `atomicLoad` / `atomicStore` (a value one invocation stored and another loads is the same either way), its
  read-modify-write rows give `u32.atomicold`;
* **relaxed determinism** (`determinism relaxed`, batch 5): the contract allows `atomicExchange` and `atomicLoad` on
  the arena, whose only use is an order-dependent value, so `index.of_atomicold` consumes the old value of a
  read-modify-write as a position (slot allocation by `atomicAdd`);
* a **row output** (`rows <b> <W>`): workgroup `w` owns row `w` of `W` cells; a store at any index is guarded by the
  row's text to the workgroup's own row (`i / W == wid.x`) and the binding's length; the cells within the row are the
  design's (the race check reads them, as the gate's rows mode did);
* the **vector view** of an f32 binding (`array<vec4<f32>>` in a seed): element `j` of the view, restricted to the
  view's last element as the device restricts it, from four f32 reads;
* `literal.uniform`: a runtime literal is the same in every invocation.
"""
from lib import *
from values import int_values

ATOMIC_FUNS = {"add": "Add", "sub": "Sub", "min": "Min", "max": "Max", "and": "And", "or": "Or", "xor": "Xor", "exchange": "Exchange"}


def views(iface):
    return [b for b in iface.get("inputs", []) + iface.get("arenas", []) if b.get("vec4_view")]


def declare(iface):
    pass


def build(iface):
    for b in views(iface):
        n = b["name"]
        helper(f"load4v_{n}", f"fn load4v_{n}(j: u32) -> vec4<f32> {{ let m = min(j, arrayLength(&{n}) / 4u - 1u) * 4u; return vec4<f32>({n}[m], {n}[m + 1u], {n}[m + 2u], {n}[m + 3u]); }}")
        R(f"load.{n}_v4", "vec4<f32>.real", ["u32.index"], f"load4v_{n}({{0}})", "element j of the binding's vec4 view, restricted to the view's last element as the device restricts it", twin=True)
        R(f"load.{n}_len_v4", "u32.count", [], f"(arrayLength(&{n}) / 4u)", "the vec4 view's length", twin=True)
    for a in iface.get("arenas", []):
        n, ty, atomic = a["name"], a["ty"], a.get("atomic", False)
        at = f"&{n}[min({{0}}, arrayLength(&{n}) - 1u)]"
        if atomic:
            R(f"load.{n}", ty, ["u32.index"], f"atomicLoad({at})", "a read of any cell of the atomic arena (relaxed: another invocation may be writing it)", twin=True)
        else:
            R(f"load.{n}", ty, ["u32.index"], f"{n}[min({{0}}, arrayLength(&{n}) - 1u)]", "a read of any cell of the arena (another invocation may be writing it)", twin=True)
        R(f"load.{n}_len", "u32.count", [], f"arrayLength(&{n})", twin=True)
        for x, d, k in rungs("LCWSPX"):
            it, vt_ = ("u32.indexw", w_of(ty)) if x in "WPX" else ("u32.index", ty)
            cell = f"{n}[min(i, arrayLength(&{n}) - 1u)]"
            text = f"{{ let i = {{0}}; atomicStore(&{cell}, {{1}}); }}" if atomic else f"{{ let i = {{0}}; {cell} = {{1}}; }}"
            R(f"store.{n}_{x}{d}_f{k}", S(x, d, k), [it, vt_], text, "a store at any cell of the arena (the arena policy accepts two invocations storing one cell)")
        if atomic:
            for fn, F in ATOMIC_FUNS.items():
                R(f"atomic.{n}_{fn}", "u32.atomicold", ["u32.index", "u32.index"], f"atomic{F}({at}, {{1}})", "a read-modify-write of any cell: the old value is order-dependent", twin=True)
    if iface.get("determinism") == "relaxed":
        R("index.of_atomicold", "u32.index", ["u32.atomicold"], "{0}", "relaxed kingdoms only: the design consumes an order-dependent value (slot allocation by atomicAdd)", twin=True)
    for o in iface.get("row_outputs", []):
        n, W = o["name"], o["row"]
        for x, d, k in rungs("LCWSPX"):
            it, vt_ = ("u32.indexw", w_of(o["ty"])) if x in "WPX" else ("u32.index", o["ty"])
            R(f"store.{n}_{x}{d}_f{k}", S(x, d, k), [it, vt_], f"{{ let i = {{0}}; if (i / {W}u == wid.x && i < arrayLength(&{n})) {{ {n}[i] = {{1}}; }} }}",
              f"a store into the workgroup's own row of {W} cells only, by the row's text")
    if iface.get("module"):
        # u32 exclusive or as a position and as a uniform (a seed's hash `x ^ y`), and the numeric conversions between
        # vectors of f32, u32 and i32 (lane by lane, as the scalar conversions)
        for L in (1, 2, 3, 4):
            t = "u32.index" if L == 1 else f"vec{L}<u32>.index"
            R("index.xor" + ("" if L == 1 else f"_v{L}"), t, [t, t], "({0} ^ {1})", "exclusive or, every bit defined", twin=True)
        R("uniform.xor", "u32.uniform", ["u32.uniform", "u32.uniform"], "({0} ^ {1})", "exclusive or of uniforms is uniform")
        forms = {"f32": "real", "u32": "index", "i32": "int"}
        for L in (2, 3, 4):
            for a, fa in forms.items():
                for b, fb in forms.items():
                    if a != b:
                        R(f"convert.{b}_to_{a}_v{L}", f"vec{L}<{a}>.{fa}", [f"vec{L}<{b}>.{fb}"], f"vec{L}<{a}>({{0}})", "a numeric conversion lane by lane", twin=True)
        C("literal.uniform", "u32.uniform", "lit_u32({v}u)", extra(iface, "index", int_values(0, 2**32 - 1, 48, "index")), "?_uniform: a runtime literal is the same in every invocation")
