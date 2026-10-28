"""Batch 5 and 6: vectors and their shape rows, matrices, bit fields, hashes,
codebook entries, fixed point, signs."""
from lib import *
from values import FAMS, ft, sfx, int_values, rng_of

SWIZZLES = ["xy", "yx", "xx", "zw", "xz", "yzw", "xyz", "zyx", "xxx", "xyxy", "xxxx", "wzyx"]
CODES = (4, 8, 16, 1024)
GEOM1 = ["length"]
GEOM2 = ["distance", "reflect"]



INVERSE_HELPERS = {
    2: "fn inverse_2(m: mat2x2<f32>) -> mat2x2<f32> { let id = 1.0 / determinant(m); return mat2x2<f32>(vec2<f32>(m[1][1], -m[0][1]) * id, vec2<f32>(-m[1][0], m[0][0]) * id); }",
    3: "fn inverse_3(m: mat3x3<f32>) -> mat3x3<f32> { let r0 = cross(m[1], m[2]); let r1 = cross(m[2], m[0]); let r2 = cross(m[0], m[1]); let id = 1.0 / dot(m[0], r0); return transpose(mat3x3<f32>(r0, r1, r2)) * id; }",
    4: """fn inverse_4(m: mat4x4<f32>) -> mat4x4<f32> {
  let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
  let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
  let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
  let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];
  let b00 = a00 * a11 - a01 * a10; let b01 = a00 * a12 - a02 * a10; let b02 = a00 * a13 - a03 * a10;
  let b03 = a01 * a12 - a02 * a11; let b04 = a01 * a13 - a03 * a11; let b05 = a02 * a13 - a03 * a12;
  let b06 = a20 * a31 - a21 * a30; let b07 = a20 * a32 - a22 * a30; let b08 = a20 * a33 - a23 * a30;
  let b09 = a21 * a32 - a22 * a31; let b10 = a21 * a33 - a23 * a31; let b11 = a22 * a33 - a23 * a32;
  let id = 1.0 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
  return mat4x4<f32>(
    vec4<f32>(a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09, a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03) * id,
    vec4<f32>(a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07, a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01) * id,
    vec4<f32>(a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06, a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00) * id,
    vec4<f32>(a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06, a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00) * id);
}""",
}


def mat(c, r):
    return f"mat{c}x{r}<f32>.real"


def declare(iface):
    for c in (2, 3, 4):
        for r in (2, 3, 4):
            T(mat(c, r), f"a matrix of {c} columns and {r} rows", c * r)
            register_twin(mat(c, r))
    T("u32.bits", "a bit field: masks, packed lanes; no position, no size")
    if iface.get("module"):
        register_twin("u32.bits")
    T("u32.hash32", "a 32-bit hash state")
    for n in CODES:
        T(f"u32.code{n}", f"a codebook entry below {n}", 1, f"0..={n - 1}")
    T("i32.q15_16", "fixed point, 16 fractional bits; saturating arithmetic only", 1, "-2147483648..=2147483647")
    T("bool.sign", "+1/-1 as a bit", 1, "0,1")


def swz_name(p):
    return p


def build(iface):
    scal = {"real": "f32", "h": "f16", "index": "u32", "count": "u32", "int": "i32", "flag": "bool"}
    fs = {"real": "", "h": "_f16", "index": "_index", "count": "_count", "int": "_int", "flag": "_flag"}
    # ---- shape rows
    for fam, s in scal.items():
        form = FAMS[fam][1]
        t1 = ft(fam)
        for L in (2, 3, 4):
            v = ft(fam, L)
            helper(f"zero_vec{L}_{s}", f"fn zero_vec{L}_{s}() -> vec{L}<{s}> {{ return vec{L}<{s}>(); }}")
            R(f"own.vec{L}" + fs[fam], v, [], f"zero_vec{L}_{s}()", "the zero vector: the second terminal every demanded type has; a call, so no operation on it is evaluated at shader creation", twin=True)
            R(f"shape.splat{L}" + fs[fam], v, [t1], f"vec{L}<{s}>({{0}})", twin=True)
            for i in range(L):
                R(f"shape.access_{i}_v{L}" + fs[fam], t1, [v], "{0}[" + str(i) + "]", "a row exists only for i < L, so a constant lane is never past the vector", twin=True)
            names = "xyzw"
            sw = ft(fam, L)
            hn = f"at{L}_{s}"
            body = "".join(f"case {i}u: {{ return v.{names[i]}; }} " for i in range(L - 1))
            helper(hn, f"fn {hn}(v: vec{L}<{s}>, i: u32) -> {s} {{ switch i {{ {body}default: {{ return v.{names[L - 1]}; }} }} }}")
            mask = f"({{1}}) & {L - 1}u" if L in (2, 4) else "({1}) % 3u"
            R(f"shape.at_v{L}" + fs[fam], t1, [v, "u32.index"], f"{hn}({{0}}, {mask})", "a run-time lane reduced by the row's text", twin=True)
        R("shape.compose2" + fs[fam], ft(fam, 2), [t1, t1], f"vec2<{s}>({{0}}, {{1}})", twin=True)
        R("shape.compose3" + fs[fam], ft(fam, 3), [ft(fam, 2), t1], f"vec3<{s}>({{0}}, {{1}})", twin=True)
        R("shape.compose4" + fs[fam], ft(fam, 4), [ft(fam, 3), t1], f"vec4<{s}>({{0}}, {{1}})", twin=True)
        for L in (2, 3, 4):
            for p in SWIZZLES:
                if max("xyzw".index(ch) for ch in p) < L and len(p) in (2, 3, 4):
                    R(f"shape.swizzle_{p}_v{L}" + fs[fam], ft(fam, len(p)), [ft(fam, L)], "{0}." + p, twin=True)
    for fam in ("real", "h"):
        for L in (2, 3, 4):
            R("arith.mul_scalar_vec" + sfx(fam, L), ft(fam, L), [ft(fam), ft(fam, L)], "({0} * {1})", twin=True)
    # ---- geometry (f32)
    for L in (2, 3, 4):
        v = ft("real", L)
        R(f"geom.dot_v{L}", "f32.real", [v, v], "dot({0}, {1})", twin=True)
        R(f"geom.dot_index_v{L}", "u32.index", [ft("index", L), ft("index", L)], "dot({0}, {1})", "wraps, defined", twin=True)
        R(f"geom.dot_int_v{L}", "i32.int", [ft("int", L), ft("int", L)], "dot({0}, {1})", "wraps, defined", twin=True)
        R(f"geom.length_v{L}", "f32.real", [v], "length({0})", twin=True)
        R(f"geom.distance_v{L}", "f32.real", [v, v], "distance({0}, {1})", twin=True)
        R(f"geom.normalize_v{L}", v, [v], "normalize({0})", "normalize(0) gives NaN lanes, scored", twin=True)
        R(f"geom.reflect_v{L}", v, [v, v], "reflect({0}, {1})", twin=True)
        R(f"geom.refract_v{L}", v, [v, v, "f32.real"], "refract({0}, {1}, {2})", twin=True)
        R(f"geom.face_forward_v{L}", v, [v, v, v], "faceForward({0}, {1}, {2})", twin=True)
    R("geom.cross", ft("real", 3), [ft("real", 3), ft("real", 3)], "cross({0}, {1})", twin=True)
    # ---- matrices
    for c in (2, 3, 4):
        for r in (2, 3, 4):
            m = mat(c, r)
            colv = ft("real", r)
            col = ", ".join([f"vec{r}<f32>(lit_f32({{v}}))"] * c)
            C(f"literal.mat{c}x{r}", m, f"mat{c}x{r}<f32>({col})", [x for x in __import__("values").f32_values(-3.4028234663852886e38, 3.4028234663852886e38, 16, f"mat{c}x{r}")], f"?_mat{c}x{r}: one number on every element", twin=True)
            helper(f"zero_mat{c}x{r}", f"fn zero_mat{c}x{r}() -> mat{c}x{r}<f32> {{ return mat{c}x{r}<f32>(); }}")
            R(f"own.mat{c}x{r}", m, [], f"zero_mat{c}x{r}()", "a call, so no operation on it is evaluated at shader creation")
            R(f"shape.compose_mat{c}x{r}", m, [colv] * c, f"mat{c}x{r}<f32>(" + ", ".join("{%d}" % i for i in range(c)) + ")", twin=True)
            for i in range(c):
                # a helper call, not `{0}[i]`: indexing a matrix constructor (`mat4x4<f32>(..)[3]`) makes naga's MSL an
                # indirection of a value, which Apple's compiler rejects (ST9, ST3 loop genes 173 and 489)
                R(f"shape.col_{i}_{c}x{r}", colv, [m], f"col_at{c}x{r}({{0}}, {i}u)", twin=True)
            names = "xyzw"
            hn = f"col_at{c}x{r}"
            body = "".join(f"case {i}u: {{ return m[{i}]; }} " for i in range(c - 1))
            helper(hn, f"fn {hn}(m: mat{c}x{r}<f32>, i: u32) -> vec{r}<f32> {{ switch i {{ {body}default: {{ return m[{c - 1}]; }} }} }}")
            R(f"shape.col_at_{c}x{r}", colv, [m, "u32.index"], f"{hn}({{0}}, ({{1}}) % {c}u)", "a run-time column reduced by the row's text", twin=True)
            R(f"arith.mul_mat_vec_{c}x{r}", colv, [m, ft("real", c)], "({0} * {1})", twin=True)
            R(f"arith.mul_vec_mat_{c}x{r}", ft("real", c), [colv, m], "({0} * {1})", twin=True)
            R(f"arith.mul_mat_scalar_{c}x{r}", m, [m, "f32.real"], "({0} * {1})", twin=True)
            R(f"arith.mul_scalar_mat_{c}x{r}", m, ["f32.real", m], "({0} * {1})", twin=True)
            R(f"arith.add_mat_{c}x{r}", m, [m, m], "({0} + {1})", twin=True)
            R(f"arith.sub_mat_{c}x{r}", m, [m, m], "({0} - {1})", twin=True)
            R(f"arith.neg_mat_{c}x{r}", m, [m], "(-1.0 * {0})", "WGSL has no unary minus on a matrix: the product with the scalar -1", twin=True)
            R(f"geom.transpose_{c}x{r}", mat(r, c), [m], "transpose({0})", twin=True)
            helper(f"outer_{c}x{r}", f"fn outer_{c}x{r}(a: vec{r}<f32>, b: vec{c}<f32>) -> mat{c}x{r}<f32> {{ return mat{c}x{r}<f32>(" + ", ".join(f"a * b.{'xyzw'[i]}" for i in range(c)) + "); }")
            R(f"geom.outer_{c}x{r}", m, [colv, ft("real", c)], f"outer_{c}x{r}({{0}}, {{1}})", twin=True)
            for k in (2, 3, 4):
                R(f"arith.mul_mat_mat_{c}x{r}_k{k}", m, [mat(k, r), mat(c, k)], "({0} * {1})", "the left's columns equal the right's rows (K), by signature", twin=True)
        R(f"geom.determinant_{c}", "f32.real", [mat(c, c)], "determinant({0})", twin=True)
        helper(f"inverse_{c}", INVERSE_HELPERS[c])
        R(f"geom.inverse_{c}", mat(c, c), [mat(c, c)], f"inverse_{c}({{0}})", "square only by signature; a singular matrix's inverse is inf/NaN lanes, scored", twin=True)
    # ---- bits
    C("literal.bits", "u32.bits", "lit_u32({v}u)", extra(iface, "bits", [str(v) for v in int_values(0, 2**32 - 1, 32, "bits")]), "?_bits", twin=bool(iface.get("module")))
    R("own.u32_bits", "u32.bits", [], "lit_u32(0u)")
    tw = bool(iface.get("module"))   # a module interface (s9 batch 8) gives the bit-field rows their w twins
    for op, sp in (("and", "({0} & {1})"), ("or", "({0} | {1})"), ("xor", "({0} ^ {1})")):
        R(f"bits.{op}", "u32.bits", ["u32.bits", "u32.bits"], sp, twin=tw)
    R("bits.not", "u32.bits", ["u32.bits"], "(~{0})", twin=tw)
    R("bits.shl", "u32.bits", ["u32.bits", "u32.index"], "({0} << {1})", "WGSL 8.10: modulo 32", twin=tw)
    R("bits.shr", "u32.bits", ["u32.bits", "u32.index"], "({0} >> {1})", twin=tw)
    for n, f in (("count_ones", "countOneBits"), ("leading_zeros", "countLeadingZeros"), ("trailing_zeros", "countTrailingZeros"), ("find_lsb", "firstTrailingBit"), ("find_msb", "firstLeadingBit")):
        R(f"bits.{n}", "u32.count", ["u32.bits"], f + "({0})", twin=tw)
    R("bits.reverse", "u32.bits", ["u32.bits"], "reverseBits({0})", twin=tw)
    R("bits.extract", "u32.bits", ["u32.bits", "u32.index", "u32.count"], "extractBits({0}, {1}, {2})", "for runtime operands the spec clamps: defined", twin=tw)
    R("bits.insert", "u32.bits", ["u32.bits", "u32.bits", "u32.index", "u32.count"], "insertBits({0}, {1}, {2}, {3})", twin=tw)
    R("convert.index_to_bits", "u32.bits", ["u32.index"], "{0}", twin=tw)
    R("convert.bits_to_index", "u32.index", ["u32.bits"], "{0}", twin=tw)
    R("convert.count_to_bits", "u32.bits", ["u32.count"], "{0}", twin=tw)
    R("convert.bitcast_f32_bits", "u32.bits", ["f32.real"], "bitcast<u32>({0})", twin=tw)
    R("convert.bitcast_bits_f32", "f32.real", ["u32.bits"], "bitcast<f32>({0})", "a bitcast to f32 can make a NaN: a value", twin=tw)
    for n, src in (("pack4x8snorm", 4), ("pack4x8unorm", 4), ("pack2x16snorm", 2), ("pack2x16unorm", 2), ("pack2x16float", 2)):
        R(f"bits.{n}", "u32.bits", [ft("real", src)], n + "({0})", twin=tw)
        R(f"bits.un{n}", ft("real", src), ["u32.bits"], "un" + n + "({0})", twin=tw)
    # ---- hash
    helper("fmix32", "fn fmix32(h0: u32) -> u32 { var h = h0; h = h ^ (h >> 16u); h = h * 0x85ebca6bu; h = h ^ (h >> 13u); h = h * 0xc2b2ae35u; h = h ^ (h >> 16u); return h; }")
    helper("combine_hash", "fn combine_hash(a: u32, b: u32) -> u32 { return fmix32(a ^ (b * 0x9e3779b9u + (a << 6u) + (a >> 2u))); }")
    R("hash.fmix32", "u32.hash32", ["u32.hash32"], "fmix32({0})")
    R("hash.combine", "u32.hash32", ["u32.hash32", "u32.bits"], "combine_hash({0}, {1})")
    R("convert.hash32_to_bits", "u32.bits", ["u32.hash32"], "{0}")
    R("convert.hash32_to_unit", "f32.real", ["u32.hash32"], "(f32({0} >> 8u) * 5.9604645e-8)")
    R("convert.hash32_to_index", "u32.index", ["u32.hash32", "u32.count"], "({0} % max({1}, 1u))")
    C("literal.hash32", "u32.hash32", "lit_u32({v}u)", [str(v) for v in int_values(0, 2**32 - 1, 32, "hash32")], "?_hash32")
    R("own.u32_hash32", "u32.hash32", [], "lit_u32(0u)")
    # ---- codebooks
    for n in CODES:
        R(f"quantise.real_to_code{n}", f"u32.code{n}", ["f32.real"], f"min(u32(clamp({{0}}, 0.0, {n - 1}.0) + 0.5), {n - 1}u)", "the row's text keeps the code below N for every input")
        R(f"quantise.code{n}_to_real", "f32.real", [f"u32.code{n}"], "f32({0})", twin=True)
        vals = list(range(n)) if n <= 16 else [str(v) for v in int_values(0, n - 1, 32, f"code{n}")]
        C(f"literal.code{n}", f"u32.code{n}", "lit_u32({v}u)", [str(v) for v in vals], f"?_code{n}")
        R(f"own.u32_code{n}", f"u32.code{n}", [], "lit_u32(0u)")
    # ---- fixed point
    helper("add_sat", "fn add_sat(a: i32, b: i32) -> i32 { let s = a + b; let ov = ((a ^ s) & (b ^ s)) < 0i; return select(s, select(2147483647i, (-2147483647i - 1i), a < 0i), ov); }")
    helper("sub_sat", "fn sub_sat(a: i32, b: i32) -> i32 { let s = a - b; let ov = ((a ^ b) & (a ^ s)) < 0i; return select(s, select(2147483647i, (-2147483647i - 1i), a < 0i), ov); }")
    helper("sat_add_u32", "fn sat_add_u32(x: u32, y: u32) -> u32 { let s = x + y; return select(s, 0xffffffffu, s < x); }")
    helper("mul_q", """fn mul_q(a: i32, b: i32) -> i32 {
  let neg = (a < 0i) != (b < 0i);
  let ua = select(u32(a), u32(-a), a < 0i);
  let ub = select(u32(b), u32(-b), b < 0i);
  let ah = ua >> 16u; let al = ua & 0xffffu; let bh = ub >> 16u; let bl = ub & 0xffffu;
  let hh = ah * bh;
  var m = 0xffffffffu;
  if (hh < 32768u) {
    m = sat_add_u32(sat_add_u32(hh << 16u, ah * bl), sat_add_u32(al * bh, (al * bl) >> 16u));
  }
  if (neg) { return select(-i32(m), (-2147483647i - 1i), m > 2147483648u); }
  return select(i32(m), 2147483647i, m > 2147483647u);
}""")
    R("fixed.add", "i32.q15_16", ["i32.q15_16", "i32.q15_16"], "add_sat({0}, {1})", "saturating")
    R("fixed.sub", "i32.q15_16", ["i32.q15_16", "i32.q15_16"], "sub_sat({0}, {1})", "saturating")
    R("fixed.mul", "i32.q15_16", ["i32.q15_16", "i32.q15_16"], "mul_q({0}, {1})", "saturating")
    R("fixed.shl", "i32.q15_16", ["i32.q15_16", "u32.count"], "({0} << {1})")
    R("fixed.shr", "i32.q15_16", ["i32.q15_16", "u32.count"], "({0} >> {1})")
    R("convert.f32_to_q15_16", "i32.q15_16", ["f32.real"], "i32(clamp({0} * 65536.0, -2147483648.0, 2147483520.0))", "saturating")
    R("convert.q15_16_to_f32", "f32.real", ["i32.q15_16"], "(f32({0}) / 65536.0)", twin=True)
    C("literal.q15_16", "i32.q15_16", "lit_i32({v})", [("(-2147483647i - 1i)" if v == -2147483648 else f"{v}i") for v in int_values(-2**31, 2**31 - 1, 32, "q15_16")], "?_q15_16")
    R("own.i32_q15_16", "i32.q15_16", [], "lit_i32(0i)")
    # ---- signs
    helper("negate_if", "fn negate_if(x: f32, s: bool) -> f32 { return select(-x, x, s); }")
    R("convert.sign_to_real", "f32.real", ["bool.sign"], "select(-1.0, 1.0, {0})", twin=True)
    C("literal.sign", "bool.sign", "lit_bool({v})", ["true", "false"], "?_sign")
    R("own.bool_sign", "bool.sign", [], "lit_bool(true)")
    R("select.negate_if", "f32.real", ["f32.real", "bool.sign"], "negate_if({0}, {1})", twin=True)
    R("bits.pack_sign", "u32.bits", ["bool.sign"], "select(0u, 1u, {0})")
    R("bits.unpack_sign", "bool.sign", ["u32.bits"], "(({0} & 1u) != 0u)")
    # ---- input vector loads
    for i in iface["inputs"]:
        n = i["name"]
        if i.get("kind") == "atomic_u32":
            continue   # no vector read of atomics
        # the vector of the input's own element: an f32 input's four reals, a u32 input's (k-means labels) four positions
        e = i.get("e", "f32")
        vty = {"f32": "vec4<f32>.real", "u32": "vec4<u32>.index"}[e]
        helper(f"load4_{n}", f"fn load4_{n}(k: u32) -> vec4<{e}> {{ let m = arrayLength(&{n}) - 1u; return vec4<{e}>({n}[min(k, m)], {n}[min(k + 1u, m)], {n}[min(k + 2u, m)], {n}[min(k + 3u, m)]); }}")
        R(f"load.{n}4", vty, ["u32.index"], f"load4_{n}({{0}})", "four consecutive elements, each read clamped in the helper", twin=True)
    # ---- the knob references (batch 6)
    T("u32.payload", "a knob reference's multiplier, a literal 1..=2^20 in units of 1/256", 1, "1..=1048576")
    SAFE_ARGS.add("u32.payload")
    C("literal.payload", "u32.payload", "lit_u32({v}u)", [str(v) for v in int_values(1, 1048576, 24, "payload")], "?_payload")
    R("own.u32_payload", "u32.payload", [], "lit_u32(256u)")
    for knob in ("wcol", "wrow", "bcol", "brow", "colspan", "rowspan", "threads", "ktile"):
        K_ = {"colspan": "(KNOB_WCOL * KNOB_BCOL)", "rowspan": "(KNOB_WROW * KNOB_BROW)"}.get(knob, "KNOB_" + knob.upper())
        R(f"literal.p_{knob}", "u32.index", ["u32.payload"], f"clamp(({K_} * {{0}}) / 256u, 0u, 4294967295u)", "a reference to a kernel knob over the prologue's constants", twin=True)
        R(f"literal.p_{knob}_lit64", "u32.lit64", ["u32.payload"], f"clamp(({K_} * {{0}}) / 256u, 1u, 64u)")
        R(f"literal.p_{knob}_lit4096", "u32.lit4096", ["u32.payload"], f"clamp(({K_} * {{0}}) / 256u, 1u, 4096u)")
        R(f"literal.p_{knob}_log2", "u32.index", ["u32.payload"], f"firstLeadingBit(max(({K_} * {{0}}) / 256u, 1u))", twin=True)
        R(f"literal.p_{knob}_m1", "u32.index", ["u32.payload"], f"(({K_} * {{0}}) / 256u - 1u)", twin=True)
