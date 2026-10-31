@group(0) @binding(0) var<storage, read> vectors: array<f32>;
@group(0) @binding(1) var<storage, read> assignments: array<u32>;
@group(0) @binding(2) var<storage, read_write> centroids: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
var<workgroup> wg_pool: array<u32, 6084>;
var<workgroup> wg_apool: array<atomic<u32>, 2068>;
var<workgroup> wg_scnt: array<atomic<u32>, 4>;
var<workgroup> wg_seg: array<u32, 36>;
fn add_sat(a: i32, b: i32) -> i32 { let s = a + b; let ov = ((a ^ s) & (b ^ s)) < 0i; return select(s, select(2147483647i, (-2147483647i - 1i), a < 0i), ov); }
fn at2_bool(v: vec2<bool>, i: u32) -> bool { switch i { case 0u: { return v.x; } default: { return v.y; } } }
fn at2_f32(v: vec2<f32>, i: u32) -> f32 { switch i { case 0u: { return v.x; } default: { return v.y; } } }
fn at2_i32(v: vec2<i32>, i: u32) -> i32 { switch i { case 0u: { return v.x; } default: { return v.y; } } }
fn at2_u32(v: vec2<u32>, i: u32) -> u32 { switch i { case 0u: { return v.x; } default: { return v.y; } } }
fn at3_bool(v: vec3<bool>, i: u32) -> bool { switch i { case 0u: { return v.x; } case 1u: { return v.y; } default: { return v.z; } } }
fn at3_f32(v: vec3<f32>, i: u32) -> f32 { switch i { case 0u: { return v.x; } case 1u: { return v.y; } default: { return v.z; } } }
fn at3_i32(v: vec3<i32>, i: u32) -> i32 { switch i { case 0u: { return v.x; } case 1u: { return v.y; } default: { return v.z; } } }
fn at3_u32(v: vec3<u32>, i: u32) -> u32 { switch i { case 0u: { return v.x; } case 1u: { return v.y; } default: { return v.z; } } }
fn at4_bool(v: vec4<bool>, i: u32) -> bool { switch i { case 0u: { return v.x; } case 1u: { return v.y; } case 2u: { return v.z; } default: { return v.w; } } }
fn at4_f32(v: vec4<f32>, i: u32) -> f32 { switch i { case 0u: { return v.x; } case 1u: { return v.y; } case 2u: { return v.z; } default: { return v.w; } } }
fn at4_i32(v: vec4<i32>, i: u32) -> i32 { switch i { case 0u: { return v.x; } case 1u: { return v.y; } case 2u: { return v.z; } default: { return v.w; } } }
fn at4_u32(v: vec4<u32>, i: u32) -> u32 { switch i { case 0u: { return v.x; } case 1u: { return v.y; } case 2u: { return v.z; } default: { return v.w; } } }
fn clamp_p(x: f32, p: vec2<f32>) -> f32 { return clamp(x, p.x, p.y); }
fn clamp_pi(x: i32, p: vec2<i32>) -> i32 { return clamp(x, p.x, p.y); }
fn col_at2x2(m: mat2x2<f32>, i: u32) -> vec2<f32> { switch i { case 0u: { return m[0]; } default: { return m[1]; } } }
fn col_at2x3(m: mat2x3<f32>, i: u32) -> vec3<f32> { switch i { case 0u: { return m[0]; } default: { return m[1]; } } }
fn col_at2x4(m: mat2x4<f32>, i: u32) -> vec4<f32> { switch i { case 0u: { return m[0]; } default: { return m[1]; } } }
fn col_at3x2(m: mat3x2<f32>, i: u32) -> vec2<f32> { switch i { case 0u: { return m[0]; } case 1u: { return m[1]; } default: { return m[2]; } } }
fn col_at3x3(m: mat3x3<f32>, i: u32) -> vec3<f32> { switch i { case 0u: { return m[0]; } case 1u: { return m[1]; } default: { return m[2]; } } }
fn col_at3x4(m: mat3x4<f32>, i: u32) -> vec4<f32> { switch i { case 0u: { return m[0]; } case 1u: { return m[1]; } default: { return m[2]; } } }
fn col_at4x2(m: mat4x2<f32>, i: u32) -> vec2<f32> { switch i { case 0u: { return m[0]; } case 1u: { return m[1]; } case 2u: { return m[2]; } default: { return m[3]; } } }
fn col_at4x3(m: mat4x3<f32>, i: u32) -> vec3<f32> { switch i { case 0u: { return m[0]; } case 1u: { return m[1]; } case 2u: { return m[2]; } default: { return m[3]; } } }
fn col_at4x4(m: mat4x4<f32>, i: u32) -> vec4<f32> { switch i { case 0u: { return m[0]; } case 1u: { return m[1]; } case 2u: { return m[2]; } default: { return m[3]; } } }
fn col_axis(g: vec3<u32>) -> u32 { return g.x; }
fn combine_hash(a: u32, b: u32) -> u32 { return fmix32(a ^ (b * 0x9e3779b9u + (a << 6u) + (a >> 2u))); }
fn fmix32(h0: u32) -> u32 { var h = h0; h = h ^ (h >> 16u); h = h * 0x85ebca6bu; h = h ^ (h >> 13u); h = h * 0xc2b2ae35u; h = h ^ (h >> 16u); return h; }
fn inverse_2(m: mat2x2<f32>) -> mat2x2<f32> { let id = 1.0 / determinant(m); return mat2x2<f32>(vec2<f32>(m[1][1], -m[0][1]) * id, vec2<f32>(-m[1][0], m[0][0]) * id); }
fn inverse_3(m: mat3x3<f32>) -> mat3x3<f32> { let r0 = cross(m[1], m[2]); let r1 = cross(m[2], m[0]); let r2 = cross(m[0], m[1]); let id = 1.0 / dot(m[0], r0); return transpose(mat3x3<f32>(r0, r1, r2)) * id; }
fn inverse_4(m: mat4x4<f32>) -> mat4x4<f32> {
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
}
fn lit_bool(x: bool) -> bool { return x; }
fn lit_f32(x: f32) -> f32 { return x; }
fn lit_i32(x: i32) -> i32 { return x; }
fn lit_u32(x: u32) -> u32 { return x; }
fn lo_width(lo: f32, w: f32) -> vec2<f32> { return vec2<f32>(lo, lo + w); }
fn lo_width_i(lo: i32, w: i32) -> vec2<i32> { return vec2<i32>(lo, max(lo, lo + w)); }
fn load4_assignments(k: u32) -> vec4<u32> { let m = arrayLength(&assignments) - 1u; return vec4<u32>(assignments[min(k, m)], assignments[min(k + 1u, m)], assignments[min(k + 2u, m)], assignments[min(k + 3u, m)]); }
fn load4_vectors(k: u32) -> vec4<f32> { let m = arrayLength(&vectors) - 1u; return vec4<f32>(vectors[min(k, m)], vectors[min(k + 1u, m)], vectors[min(k + 2u, m)], vectors[min(k + 3u, m)]); }
fn mul_q(a: i32, b: i32) -> i32 {
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
}
fn nan0_i32(x: f32) -> i32 { return i32(select(x, 0.0, x != x)); }
fn nan0_u32(x: f32) -> u32 { return u32(select(x, 0.0, x != x)); }
fn negate_if(x: f32, s: bool) -> f32 { return select(-x, x, s); }
fn outer_2x2(a: vec2<f32>, b: vec2<f32>) -> mat2x2<f32> { return mat2x2<f32>(a * b.x, a * b.y); }
fn outer_2x3(a: vec3<f32>, b: vec2<f32>) -> mat2x3<f32> { return mat2x3<f32>(a * b.x, a * b.y); }
fn outer_2x4(a: vec4<f32>, b: vec2<f32>) -> mat2x4<f32> { return mat2x4<f32>(a * b.x, a * b.y); }
fn outer_3x2(a: vec2<f32>, b: vec3<f32>) -> mat3x2<f32> { return mat3x2<f32>(a * b.x, a * b.y, a * b.z); }
fn outer_3x3(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> { return mat3x3<f32>(a * b.x, a * b.y, a * b.z); }
fn outer_3x4(a: vec4<f32>, b: vec3<f32>) -> mat3x4<f32> { return mat3x4<f32>(a * b.x, a * b.y, a * b.z); }
fn outer_4x2(a: vec2<f32>, b: vec4<f32>) -> mat4x2<f32> { return mat4x2<f32>(a * b.x, a * b.y, a * b.z, a * b.w); }
fn outer_4x3(a: vec3<f32>, b: vec4<f32>) -> mat4x3<f32> { return mat4x3<f32>(a * b.x, a * b.y, a * b.z, a * b.w); }
fn outer_4x4(a: vec4<f32>, b: vec4<f32>) -> mat4x4<f32> { return mat4x4<f32>(a * b.x, a * b.y, a * b.z, a * b.w); }
fn own_load_centroids(r: u32, c: u32) -> f32 { return centroids[min(r * (params.y * params.z) + c, arrayLength(&centroids) - 1u)]; }
fn row_axis(g: vec3<u32>) -> u32 { return g.y; }
fn sat_add_u32(x: u32, y: u32) -> u32 { let s = x + y; return select(s, 0xffffffffu, s < x); }
fn seg_at(i: u32) -> u32 { return wg_seg[min(i, 32u)]; }
fn sorted(a: f32, b: f32) -> vec2<f32> { return vec2<f32>(min(a, b), max(a, b)); }
fn sorted_i(a: i32, b: i32) -> vec2<i32> { return vec2<i32>(min(a, b), max(a, b)); }
fn sub_floor(a: u32, b: u32) -> u32 { return a - min(a, b); }
fn sub_sat(a: i32, b: i32) -> i32 { let s = a - b; let ov = ((a ^ b) & (a ^ s)) < 0i; return select(s, select(2147483647i, (-2147483647i - 1i), a < 0i), ov); }
fn wg_set_v4(o: u32, ls: u32, v: vec4<f32>) { if (o < 6080u) { wg_pool[o] = bitcast<u32>(v.x); } if (o + ls < 6080u) { wg_pool[o + ls] = bitcast<u32>(v.y); } if (o + 2u * ls < 6080u) { wg_pool[o + 2u * ls] = bitcast<u32>(v.z); } if (o + 3u * ls < 6080u) { wg_pool[o + 3u * ls] = bitcast<u32>(v.w); } }
fn wg_set_v4_below(o: u32, ls: u32, v: vec4<f32>, sl: u32, sp: u32) { if (o >= sl && o < sp) { wg_pool[o] = bitcast<u32>(v.x); } if (o + ls >= sl && o + ls < sp) { wg_pool[o + ls] = bitcast<u32>(v.y); } if (o + 2u * ls >= sl && o + 2u * ls < sp) { wg_pool[o + 2u * ls] = bitcast<u32>(v.z); } if (o + 3u * ls >= sl && o + 3u * ls < sp) { wg_pool[o + 3u * ls] = bitcast<u32>(v.w); } }
fn wg_v4(b: u32, ls: u32, i: u32) -> vec4<f32> { return vec4<f32>(bitcast<f32>(wg_pool[min(b + i, 6080u)]), bitcast<f32>(wg_pool[min(b + ls + i, 6080u)]), bitcast<f32>(wg_pool[min(b + 2u * ls + i, 6080u)]), bitcast<f32>(wg_pool[min(b + 3u * ls + i, 6080u)])); }
fn wg_v4_above(b: u32, ls: u32, i: u32, sl: u32, sp: u32) -> vec4<f32> { return vec4<f32>(bitcast<f32>(wg_pool[xw(b + i, sl, sp)]), bitcast<f32>(wg_pool[xw(b + ls + i, sl, sp)]), bitcast<f32>(wg_pool[xw(b + 2u * ls + i, sl, sp)]), bitcast<f32>(wg_pool[xw(b + 3u * ls + i, sl, sp)])); }
fn xidx(p: u32, h: u32, m: u32) -> u32 { if (m == 0u) { return p; } let lo = p & (h - 1u); return ((p & ~(h - 1u)) << 1u) | lo; }
fn xok(p: u32, h: u32, m: u32) -> bool { if (m == 0u) { return (p & h) == 0u; } return p < 2147483648u; }
fn xw(a: u32, sl: u32, sp: u32) -> u32 { return select(min(a, 6080u), 6080u, a >= sl && a < sp); }
fn zero_mat2x2() -> mat2x2<f32> { return mat2x2<f32>(); }
fn zero_mat2x3() -> mat2x3<f32> { return mat2x3<f32>(); }
fn zero_mat2x4() -> mat2x4<f32> { return mat2x4<f32>(); }
fn zero_mat3x2() -> mat3x2<f32> { return mat3x2<f32>(); }
fn zero_mat3x3() -> mat3x3<f32> { return mat3x3<f32>(); }
fn zero_mat3x4() -> mat3x4<f32> { return mat3x4<f32>(); }
fn zero_mat4x2() -> mat4x2<f32> { return mat4x2<f32>(); }
fn zero_mat4x3() -> mat4x3<f32> { return mat4x3<f32>(); }
fn zero_mat4x4() -> mat4x4<f32> { return mat4x4<f32>(); }
fn zero_vec2_bool() -> vec2<bool> { return vec2<bool>(); }
fn zero_vec2_f32() -> vec2<f32> { return vec2<f32>(); }
fn zero_vec2_i32() -> vec2<i32> { return vec2<i32>(); }
fn zero_vec2_u32() -> vec2<u32> { return vec2<u32>(); }
fn zero_vec3_bool() -> vec3<bool> { return vec3<bool>(); }
fn zero_vec3_f32() -> vec3<f32> { return vec3<f32>(); }
fn zero_vec3_i32() -> vec3<i32> { return vec3<i32>(); }
fn zero_vec3_u32() -> vec3<u32> { return vec3<u32>(); }
fn zero_vec4_bool() -> vec4<bool> { return vec4<bool>(); }
fn zero_vec4_f32() -> vec4<f32> { return vec4<f32>(); }
fn zero_vec4_i32() -> vec4<i32> { return vec4<i32>(); }
fn zero_vec4_u32() -> vec4<u32> { return vec4<u32>(); }
