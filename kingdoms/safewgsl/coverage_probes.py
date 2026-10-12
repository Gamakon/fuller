# The WGSL coverage probes (docs/kingdom/WGSL_COVERAGE.md): one Gram-role kernel per WGSL compute
# construct. MOD is module scope, BODY runs after the item loop and leaves g (f32) finite; the store
# adds g * 0.0, so every probe is the shipped Gram kernel plus its construct.
#
#   python3 kingdoms/safewgsl/coverage_probes.py <dir>/in
#   ./target/release/examples/wgsl_safe_convert <dir>/in --out <dir>/out --karva <dir>/k --slots 128
#
# A probe that does not convert is a construct our software does not yet read (or one naga 0.20
# cannot parse, which the converter's message shows as a parse error).
import os, sys, re
OUT = sys.argv[1]
T = """// grid 16 16 1 1
// probe: {name}
struct Cfg {{ d: u32, m: u32, r_rows: u32, r_off: u32, }};
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
{mod}
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {{
    var f0: f32 = 0.0;
    var g: f32 = 0.0;
    var n: u32 = gid.x;
    var v: vec4<f32> = vec4<f32>(0.5);
    let gr = cfg.r_off + gid.y;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {{
        f0 = f0 + phi[t * cfg.d + gr] * phi[t * cfg.d + gid.x];
    }}
{body}
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {{
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0 + g * 0.0;
    }}
}}
"""
P = []
def p(name, body, mod=""):
    P.append((name, ("    var s: i32 = -3;\n" if re.search(r"\bs\b", body) else "") + body, mod))
# ---- statements
p("stmt_while", "    var w: u32 = 0u;\n    while (w < 4u) { g = g + 1.0; w = w + 1u; }")
p("stmt_loop_continuing", "    var w: u32 = 0u;\n    loop { if (w >= 4u) { break; } g = g + 1.0; continuing { w = w + 1u; } }")
p("stmt_break_if", "    var w: u32 = 0u;\n    loop { g = g + 1.0; continuing { w = w + 1u; break if w >= 4u; } }")
p("stmt_switch", "    switch (n & 3u) { case 0u: { g = 1.0; } case 1u, 2u: { g = 2.0; } default: { g = 3.0; } }")
p("stmt_switch_i32", "    switch (s) { case -1: { g = 1.0; } default { g = 3.0; } }")
p("stmt_if_else_if", "    if (f0 > 1.0) { g = 1.0; } else if (f0 > 0.5) { g = 2.0; } else { g = 3.0; }")
p("stmt_compound_f32", "    g += f0; g *= 0.5; g -= 1.0; g /= 2.0;")
p("stmt_compound_u32", "    n %= 3u; n <<= 1u; n |= 1u; n &= 7u; n ^= 2u; n >>= 1u; n += 1u; n -= 1u; n *= 2u; n /= 2u;\n    g = f32(n);")
p("stmt_increment", "    n++; n--; s++;\n    g = f32(n) + f32(s);")
p("stmt_phony", "    _ = f0;\n    g = 1.0;")
p("stmt_const_local", "    const K: f32 = 2.0;\n    g = g + K;")
p("stmt_block", "    { g = 1.0; }")
p("stmt_break_in_for", "    for (var w: u32 = 0u; w < 8u; w = w + 1u) { if (f32(w) > f0) { break; } g = g + 1.0; }")
p("stmt_continue_in_for", "    for (var w: u32 = 0u; w < 8u; w = w + 1u) { if ((w & 1u) == 1u) { continue; } g = g + 1.0; }")
p("stmt_return_early", "    if (gid.x >= cfg.d) { return; }\n    g = 1.0;")
p("stmt_call_ptr_helper", "    bump(&g);", "fn bump(p: ptr<function, f32>) { *p = *p + 1.0; }")
p("stmt_call_value_helper", "    g = twice(f0);", "fn twice(x: f32) -> f32 { return 2.0 * x; }")
p("stmt_helper_with_loop", "    g = sum4(f0);", "fn sum4(x: f32) -> f32 { var a: f32 = 0.0; for (var i: u32 = 0u; i < 4u; i = i + 1u) { a = a + x; } return a; }")
p("stmt_helper_early_return", "    g = clip(f0);", "fn clip(x: f32) -> f32 { if (x > 1.0) { return 1.0; } return x; }")
p("stmt_storage_barrier", "    storageBarrier();\n    g = 1.0;")
p("stmt_workgroup_uniform_load", "    if (lidx == 0u) { wv = f0; }\n    g = workgroupUniformLoad(&wv);", "var<workgroup> wv: f32;")
p("stmt_workgroup_barrier", "    workgroupBarrier();\n    g = 1.0;")
# ---- declarations
p("decl_const_module", "    g = SCALE * f0;", "const SCALE: f32 = 0.5;")
p("decl_const_module_abstract", "    g = f32(N) * f0;", "const N = 4;")
p("decl_override", "    g = SCALE * f0;", "override SCALE: f32 = 0.5;")
p("decl_alias", "    var q: real = f0;\n    g = q;", "alias real = f32;")
p("decl_struct_local", "    var q: P;\n    q.a = f0;\n    q.b = n;\n    g = q.a + f32(q.b);", "struct P { a: f32, b: u32, }")
p("decl_struct_align_size", "    var q: Q;\n    q.a = f0;\n    g = q.a;", "struct Q { @align(16) a: f32, @size(16) b: u32, }")
p("decl_private_var", "    pg = f0;\n    g = pg;", "var<private> pg: f32;")
p("decl_private_array", "    pa[n & 7u] = f0;\n    g = pa[n & 7u];", "var<private> pa: array<f32, 8>;")
p("decl_function_array_const_index", "    var a: array<f32, 4>;\n    a[2] = f0;\n    g = a[2];")
p("decl_let_typed", "    let q: f32 = f0 * 2.0;\n    g = q;")
p("decl_ptr_local", "    let pp = &g;\n    *pp = f0;")
p("decl_ptr_array_elem", "    var a: array<f32, 4>;\n    let pp = &a[1];\n    *pp = f0;\n    g = a[1];")
p("decl_must_use", "    g = twice(f0);", "@must_use fn twice(x: f32) -> f32 { return 2.0 * x; }")
p("decl_workgroup_size_override", "    g = 1.0;", "override WG: u32 = 16u;")
# ---- types and expressions
p("ty_i32_from_u32", "    let q: i32 = i32(n) - 3;\n    g = f32(q);")
p("ty_i32_scalar", "    g = f32(abs(s)) + f32(max(s, -2)) + f32(min(s, 2)) + f32(clamp(s, -1, 1)) + f32(sign(s));")
p("ty_i32_arith", "    let q: i32 = (s * 3 - 1) / 2 % 5;\n    g = f32(q) + f32(-s);")
p("ty_i32_bits", "    g = f32(countOneBits(s)) + f32(firstLeadingBit(s)) + f32(extractBits(s, 1u, 3u)) + f32(reverseBits(s) & 1) + f32(s >> 1u) + f32(s << 1u);")
p("ty_u32_minmax", "    g = f32(min(n, 3u)) + f32(max(n, 1u)) + f32(clamp(n, 1u, 9u));")
p("ty_vec2_f32", "    let q = vec2<f32>(f0, 1.0);\n    g = q.x + q.y + length(q) + dot(q, q);")
p("ty_vec3_f32", "    let q = vec3<f32>(f0, 1.0, 2.0);\n    g = length(cross(q, q.zyx)) + q.z;")
p("ty_vec4_f32_ops", "    let q = v * f0 + vec4<f32>(1.0);\n    g = q.w + dot(q, v);")
p("ty_vec2_u32", "    let q = vec2<u32>(n, 2u) + vec2<u32>(1u);\n    g = f32(q.x + q.y);")
p("ty_vec3_u32", "    let q = gid + vec3<u32>(1u);\n    g = f32(q.x * q.z);")
p("ty_vec4_u32", "    let q = vec4<u32>(n, 1u, 2u, 3u) & vec4<u32>(7u);\n    g = f32(q.x + q.w);")
p("ty_vec2_i32", "    let q = vec2<i32>(s, -1);\n    g = f32(q.x * q.y);")
p("ty_vec4_i32", "    let q = vec4<i32>(s) - vec4<i32>(1, 2, 3, 4);\n    g = f32(q.w);")
p("ty_vec4_bool", "    let q = vec4<f32>(f0) > vec4<f32>(0.5, 1.0, 1.5, 2.0);\n    g = select(0.0, 1.0, any(q)) + select(0.0, 1.0, all(q));")
p("ty_vec2_bool_select", "    let q = select(vec2<f32>(1.0), vec2<f32>(2.0), vec2<bool>(f0 > 1.0, true));\n    g = q.x + q.y;")
p("ty_swizzle_multi", "    let q = v.zyx;\n    let r = v.xy;\n    g = q.x + r.y;")
p("ty_swizzle_assign", "    var q = v;\n    q.x = f0;\n    q.w = 1.0;\n    g = q.x + q.w;")
p("ty_mat2x2", "    let m = mat2x2<f32>(f0, 1.0, 0.0, f0);\n    g = determinant(m) + (m * vec2<f32>(1.0)).x;")
p("ty_mat3x3", "    let m = mat3x3<f32>(vec3<f32>(f0), vec3<f32>(1.0), vec3<f32>(0.0, 0.0, 1.0));\n    g = determinant(m) + (m * vec3<f32>(1.0)).y;")
p("ty_mat4x4", "    let m = mat4x4<f32>(v, v, v, vec4<f32>(f0));\n    g = (m * v).x + determinant(m);")
p("ty_mat3x4_nonsquare", "    let m = mat3x4<f32>(v, v, vec4<f32>(f0));\n    g = (m * vec3<f32>(1.0)).w;")
p("ty_mat_transpose_mul", "    let m = mat2x2<f32>(f0, 1.0, 0.0, f0);\n    g = (transpose(m) * m)[1][0];")
p("ty_mat_column_access", "    var m = mat2x2<f32>(f0, 1.0, 0.0, f0);\n    m[1] = vec2<f32>(2.0);\n    g = m[1].y + m[0][0];")
p("ty_array_of_vec2", "    var a: array<vec2<f32>, 4>;\n    a[n & 3u] = vec2<f32>(f0);\n    g = a[n & 3u].y;")
p("ty_array_of_struct", "    var a: array<P, 2>;\n    a[1].a = f0;\n    g = a[1].a;", "struct P { a: f32, b: u32, }")
p("ty_array_constructor", "    let a = array<f32, 3>(f0, 1.0, 2.0);\n    g = a[n % 3u];")
p("ty_runtime_array_length", "    g = f32(arrayLength(&phi));")
p("ty_bitcast", "    g = bitcast<f32>(bitcast<u32>(f0)) + f32(bitcast<i32>(n));")
p("ty_bitcast_vec", "    g = bitcast<vec4<f32>>(bitcast<vec4<u32>>(v)).x;")
p("ty_conversion_f32_u32_i32", "    g = f32(u32(f0)) + f32(i32(f0)) + f32(u32(s)) + f32(i32(n));")
p("ty_bool_conversion", "    g = f32(f0 > 0.0) + f32(u32(n > 2u)) + f32(i32(true));")
p("ty_abstract_literals", "    g = 1 + 2.5 + f0;")
p("ty_hex_and_suffix_literals", "    n = 0xFFu & n;\n    g = 1e-3f + f32(n) + 0x1p-2;")
# ---- operators
p("op_logical_short_circuit", "    if (f0 > 0.5 && f32(n) > 1.0 || f0 < -1.0) { g = 1.0; }")
p("op_bitwise_not", "    g = f32(~n & 15u);")
p("op_modulo_f32", "    g = f0 % 0.75;")
p("op_negate_vec", "    g = (-v).x;")
p("op_compare_vec_all", "    g = select(0.0, 1.0, all(v == vec4<f32>(0.5)));")
# ---- built-in functions (f32 scalar and vec4)
FL = ["abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil", "cos", "cosh", "degrees", "exp", "exp2", "floor", "fract", "inverseSqrt", "length", "log", "log2", "normalize", "radians", "round", "saturate", "sign", "sin", "sinh", "sqrt", "tan", "tanh", "trunc"]
for fn in FL:
    arg = "(0.25 + abs(f0) * 0.001)"
    if fn in ("acosh",): arg = "(1.25 + abs(f0) * 0.001)"
    if fn in ("length", "normalize"):
        p(f"fn_{fn}_vec4", f"    g = f32({fn}(v + vec4<f32>({arg})){'.x' if fn == 'normalize' else ''});")
        continue
    p(f"fn_{fn}_f32", f"    g = {fn}({arg});")
    p(f"fn_{fn}_vec4", f"    g = {fn}(v * {arg}).y;")
for fn, args in [("atan2", "f0, 2.0"), ("clamp", "f0, 0.0, 1.0"), ("distance", "f0, 1.0"), ("fma", "f0, 2.0, 1.0"), ("ldexp", "f0, 2"), ("max", "f0, 1.0"), ("min", "f0, 1.0"), ("mix", "f0, 1.0, 0.25"), ("pow", "abs(f0), 2.0"), ("smoothstep", "0.0, 1.0, f0"), ("step", "0.5, f0"), ("dot", "v, v"), ("reflect", "v, v"), ("faceForward", "v, v, v")]:
    p(f"fn_{fn}", f"    g = f32({fn}({args}){'.x' if fn in ('reflect', 'faceForward') else ''});")
p("fn_refract", "    g = refract(normalize(v), vec4<f32>(0.0, 0.0, 1.0, 0.0), 0.5).z;")
p("fn_cross", "    g = cross(v.xyz, vec3<f32>(1.0, 0.0, 0.0)).y;")
p("fn_modf", "    let r = modf(f0 + 0.5);\n    g = r.fract + r.whole;")
p("fn_frexp", "    let r = frexp(f0 + 0.5);\n    g = r.fract + f32(r.exp);")
p("fn_mix_vec_scalar", "    g = mix(v, vec4<f32>(1.0), 0.25).x;")
p("fn_select_scalar", "    g = select(1.0, 2.0, f0 > 1.0);")
# ---- integer built-ins
for fn in ["countOneBits", "countLeadingZeros", "countTrailingZeros", "reverseBits", "firstLeadingBit", "firstTrailingBit"]:
    p(f"fn_{fn}_u32", f"    g = f32({fn}(n));")
    p(f"fn_{fn}_vec2u", f"    g = f32({fn}(vec2<u32>(n, 3u)).y);")
p("fn_extractBits_u32", "    g = f32(extractBits(n, 2u, 3u));")
p("fn_insertBits_u32", "    g = f32(insertBits(n, 5u, 2u, 3u));")
p("fn_abs_vec_i32", "    g = f32(abs(vec2<i32>(s, -2)).x);")
p("fn_dot_vec_u32", "    g = f32(dot(vec2<u32>(n, 1u), vec2<u32>(2u, 3u)));")
# ---- packing
for fn, arg in [("pack4x8snorm", "v"), ("pack4x8unorm", "v"), ("pack2x16snorm", "v.xy"), ("pack2x16unorm", "v.xy"), ("pack2x16float", "v.xy")]:
    p(f"fn_{fn}", f"    g = f32({fn}({arg}) & 255u);")
for fn in ["unpack4x8snorm", "unpack4x8unorm", "unpack2x16snorm", "unpack2x16unorm", "unpack2x16float"]:
    p(f"fn_{fn}", f"    g = {fn}(n).x;")
# ---- built-in values
p("bv_num_workgroups", "    g = f32(nwg.x);", "")
p("bv_all_builtins", "    g = 1.0;", "")
# ---- atomics and subgroups beyond the kingdom
p("atomic_storage", "    g = 1.0;", "")
p("subgroup_ballot", "    let b = subgroupBallot(f0 > 0.5);\n    g = f32(countOneBits(b.x));")
p("subgroup_vec4", "    g = subgroupAdd(v).x;")
p("subgroup_i32", "    g = f32(subgroupMax(s));")
os.makedirs(OUT, exist_ok=True)
for name, body, mod in P:
    text = T.format(name=name, mod=mod, body=body)
    if name == "bv_num_workgroups":
        text = text.replace("@builtin(local_invocation_index) lidx: u32)", "@builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>)")
    if name == "bv_all_builtins":
        text = text.replace("@builtin(local_invocation_index) lidx: u32)", "@builtin(local_invocation_index) lidx: u32, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>)").replace("    g = 1.0;", "    g = f32(lid.x + wid.y + nwg.z) * 0.0 + 1.0;")
    if name == "decl_workgroup_size_override":
        text = text.replace("@workgroup_size(16, 16, 1)", "@workgroup_size(WG, 16, 1)")
    if name == "atomic_storage":
        text = text.replace("var<uniform> cfg: Cfg;", "var<uniform> cfg: Cfg;\n@group(0) @binding(3) var<storage, read_write> counter: atomic<u32>;").replace("    g = 1.0;", "    g = f32(atomicAdd(&counter, 1u)) * 0.0;")
    open(os.path.join(OUT, name + ".wgsl"), "w").write(text)
print(len(P), "probes")
