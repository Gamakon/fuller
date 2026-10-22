"""The image-output families (ST9): a kernel that draws a picture into a write-only storage texture from uniforms.

Generic over the interface, generated only when the interface has an output of kind "texture" (so the Gram tables are
byte-identical to what they were). What the families are:

* the output binding of kind storage texture: `skeleton` writes its store row over the pixel the invocation owns
  (`threadrow_<b>`, `threadcol_<b>` from `gid`, exactly as a buffer output's cell), guarded by the texture's dimensions;
  `textureStore` is the only way a pixel is written;
* `load.image_w` / `load.image_h`: `textureDimensions(<b>)`, a size;
* one load row per field of each uniform block (`load.<block>.<field>`), typed by the field (f32 a number, u32 a size,
  i32 a signed integer); a field whose name starts with `_` is padding and has no row;
* the f32 remainder `%` (arith.rem), which the Gram instance has no use for;
* `uniform.<block>.<field>` for an i32 field, the field as a loop bound: `u32(max(field, 0))`, the trip count of a
  `for (i = 0; i < field; i++)` exactly.
"""
from lib import *
from values import LANES, ft, sfx

SCALAR_TYPE = {"f32": "f32.real", "u32": "u32.count", "i32": "i32.int"}


def textures(iface):
    return [o for o in iface["outputs"] if o.get("kind") == "texture"]


def declare(iface):
    pass


def build(iface):
    for o in textures(iface):
        n = o["name"]
        R(f"load.image_w_{n}", "u32.count", [], f"textureDimensions({n}).x", "the output texture's width", twin=True)
        R(f"load.image_h_{n}", "u32.count", [], f"textureDimensions({n}).y", "the output texture's height", twin=True)
    for u in iface.get("uniforms", []):
        b = u["name"]
        for name, ty in u["fields"]:
            if name.startswith("_"):
                continue
            R(f"load.{b}.{name}", SCALAR_TYPE[ty], [], f"{b}.{name}", f"a field of the uniform block {b}", twin=True)
            if ty == "i32":
                R(f"uniform.{b}.{name}", "u32.uniform", [], f"u32(max({b}.{name}, 0i))", "an i32 field as a loop bound: the trip count of a counted loop, a negative field gives none")
    for L in LANES:
        t = ft("real", L)
        R("arith.rem" + sfx("real", L), t, [t, t], "({0} % {1})", "WGSL f32 remainder: the sign of the dividend, a zero divisor gives NaN (scored)", twin=True)
