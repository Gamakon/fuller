enable f16;
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;
struct VolumeParams {
    speed: f32, intensity: f32,
    color1_r: f32, color1_g: f32, color1_b: f32,
    color2_r: f32, color2_g: f32, color2_b: f32,
    color3_r: f32, color3_g: f32, color3_b: f32,
    gamma: f32, zoom: f32, _padding1: f32, _padding2: f32, _padding3: f32,
};
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: VolumeParams;
struct DispatchTile {
    origin: vec4<u32>,
    grid: vec4<u32>,
};
@group(3) @binding(0) var<uniform> dispatch_tile: DispatchTile;
var<workgroup> wg_f_11: array<f32, 2048>;
var<workgroup> wg_u_10: array<u32, 1024>;
var<workgroup> wg_v4_9: array<vec4<f32>, 512>;
var<workgroup> wg_v2_8: array<vec2<f32>, 256>;
var<workgroup> wg_v3_7: array<vec3<f32>, 128>;
var<workgroup> wg_i_9: array<i32, 512>;
var<workgroup> wg_au_8: array<atomic<u32>, 256>;
var<workgroup> wg_ai_7: array<atomic<i32>, 128>;
fn add_sat(a: i32, b: i32) -> i32 { let s = a + b; let ov = ((a ^ s) & (b ^ s)) < 0i; return select(s, select(2147483647i, (-2147483647i - 1i), a < 0i), ov); }
fn at2_bool(v: vec2<bool>, i: u32) -> bool { switch i { case 0u: { return v.x; } default: { return v.y; } } }
fn at2_f16(v: vec2<f16>, i: u32) -> f16 { switch i { case 0u: { return v.x; } default: { return v.y; } } }
fn at2_f32(v: vec2<f32>, i: u32) -> f32 { switch i { case 0u: { return v.x; } default: { return v.y; } } }
fn at2_i32(v: vec2<i32>, i: u32) -> i32 { switch i { case 0u: { return v.x; } default: { return v.y; } } }
fn at2_u32(v: vec2<u32>, i: u32) -> u32 { switch i { case 0u: { return v.x; } default: { return v.y; } } }
fn at3_bool(v: vec3<bool>, i: u32) -> bool { switch i { case 0u: { return v.x; } case 1u: { return v.y; } default: { return v.z; } } }
fn at3_f16(v: vec3<f16>, i: u32) -> f16 { switch i { case 0u: { return v.x; } case 1u: { return v.y; } default: { return v.z; } } }
fn at3_f32(v: vec3<f32>, i: u32) -> f32 { switch i { case 0u: { return v.x; } case 1u: { return v.y; } default: { return v.z; } } }
fn at3_i32(v: vec3<i32>, i: u32) -> i32 { switch i { case 0u: { return v.x; } case 1u: { return v.y; } default: { return v.z; } } }
fn at3_u32(v: vec3<u32>, i: u32) -> u32 { switch i { case 0u: { return v.x; } case 1u: { return v.y; } default: { return v.z; } } }
fn at4_bool(v: vec4<bool>, i: u32) -> bool { switch i { case 0u: { return v.x; } case 1u: { return v.y; } case 2u: { return v.z; } default: { return v.w; } } }
fn at4_f16(v: vec4<f16>, i: u32) -> f16 { switch i { case 0u: { return v.x; } case 1u: { return v.y; } case 2u: { return v.z; } default: { return v.w; } } }
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
fn lit_f16(x: f16) -> f16 { return x; }
fn lit_f32(x: f32) -> f32 { return x; }
fn lit_i32(x: i32) -> i32 { return x; }
fn lit_u32(x: u32) -> u32 { return x; }
fn lo_width(lo: f32, w: f32) -> vec2<f32> { return vec2<f32>(lo, lo + w); }
fn lo_width_i(lo: i32, w: i32) -> vec2<i32> { return vec2<i32>(lo, max(lo, lo + w)); }
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
fn own_get_wg_f_11(i: u32) -> f32 { return select(0.0, wg_f_11[min(i, 2047u)], i < 2048u); }
fn own_get_wg_i_9(i: u32) -> i32 { return select(0i, wg_i_9[min(i, 511u)], i < 512u); }
fn own_get_wg_u_10(i: u32) -> u32 { return select(0u, wg_u_10[min(i, 1023u)], i < 1024u); }
fn own_get_wg_v2_8(i: u32) -> vec2<f32> { return select(vec2<f32>(), wg_v2_8[min(i, 255u)], i < 256u); }
fn own_get_wg_v3_7(i: u32) -> vec3<f32> { return select(vec3<f32>(), wg_v3_7[min(i, 127u)], i < 128u); }
fn own_get_wg_v4_9(i: u32) -> vec4<f32> { return select(vec4<f32>(), wg_v4_9[min(i, 511u)], i < 512u); }
fn sat_add_u32(x: u32, y: u32) -> u32 { let s = x + y; return select(s, 0xffffffffu, s < x); }
fn sorted(a: f32, b: f32) -> vec2<f32> { return vec2<f32>(min(a, b), max(a, b)); }
fn sorted_i(a: i32, b: i32) -> vec2<i32> { return vec2<i32>(min(a, b), max(a, b)); }
fn sub_floor(a: u32, b: u32) -> u32 { return a - min(a, b); }
fn sub_sat(a: i32, b: i32) -> i32 { let s = a - b; let ov = ((a ^ b) & (a ^ s)) < 0i; return select(s, select(2147483647i, (-2147483647i - 1i), a < 0i), ov); }
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
fn zero_vec2_f16() -> vec2<f16> { return vec2<f16>(); }
fn zero_vec2_f32() -> vec2<f32> { return vec2<f32>(); }
fn zero_vec2_i32() -> vec2<i32> { return vec2<i32>(); }
fn zero_vec2_u32() -> vec2<u32> { return vec2<u32>(); }
fn zero_vec3_bool() -> vec3<bool> { return vec3<bool>(); }
fn zero_vec3_f16() -> vec3<f16> { return vec3<f16>(); }
fn zero_vec3_f32() -> vec3<f32> { return vec3<f32>(); }
fn zero_vec3_i32() -> vec3<i32> { return vec3<i32>(); }
fn zero_vec3_u32() -> vec3<u32> { return vec3<u32>(); }
fn zero_vec4_bool() -> vec4<bool> { return vec4<bool>(); }
fn zero_vec4_f16() -> vec4<f16> { return vec4<f16>(); }
fn zero_vec4_f32() -> vec4<f32> { return vec4<f32>(); }
fn zero_vec4_i32() -> vec4<i32> { return vec4<i32>(); }
fn zero_vec4_u32() -> vec4<u32> { return vec4<u32>(); }
// grid 8 8 1 1
fn row_axis(g: vec3<u32>) -> u32 { return g.y; }
fn col_axis(g: vec3<u32>) -> u32 { return g.x; }
const WG_X: u32 = 8u;
const WG_Y: u32 = 8u;
const KNOB_WCOL: u32 = 8u;
const KNOB_WROW: u32 = 8u;
const KNOB_BCOL: u32 = 1u;
const KNOB_BROW: u32 = 1u;
const KNOB_KTILE: u32 = 1u;
const KNOB_THREADS: u32 = 8u * 8u;

@compute @workgroup_size(WG_X, WG_Y, 1)
fn main(@builtin(global_invocation_id) gid_tile: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid_tile: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
let wid: vec3<u32> = wid_tile + dispatch_tile.origin.xyz;
let gid: vec3<u32> = gid_tile + dispatch_tile.origin.xyz * vec3<u32>(WG_X, WG_Y, 1u);
let nwg: vec3<u32> = dispatch_tile.grid.xyz;
var rf0: f32 = 0.0;
var rf1: f32 = 0.0;
var rf2: f32 = 0.0;
var rf3: f32 = 0.0;
var rf4: f32 = 0.0;
var rf5: f32 = 0.0;
var rf6: f32 = 0.0;
var rf7: f32 = 0.0;
var rf8: f32 = 0.0;
var rf9: f32 = 0.0;
var rf10: f32 = 0.0;
var rf11: f32 = 0.0;
var rf12: f32 = 0.0;
var rf13: f32 = 0.0;
var rf14: f32 = 0.0;
var rf15: f32 = 0.0;
var rf16: f32 = 0.0;
var rf17: f32 = 0.0;
var rf18: f32 = 0.0;
var rf19: f32 = 0.0;
var rf20: f32 = 0.0;
var rf21: f32 = 0.0;
var rf22: f32 = 0.0;
var rf23: f32 = 0.0;
var rf24: f32 = 0.0;
var rf25: f32 = 0.0;
var rf26: f32 = 0.0;
var rf27: f32 = 0.0;
var rf28: f32 = 0.0;
var rf29: f32 = 0.0;
var rf30: f32 = 0.0;
var rf31: f32 = 0.0;
var rf32: f32 = 0.0;
var rf33: f32 = 0.0;
var rf34: f32 = 0.0;
var rf35: f32 = 0.0;
var rf36: f32 = 0.0;
var rf37: f32 = 0.0;
var rf38: f32 = 0.0;
var rf39: f32 = 0.0;
var rf40: f32 = 0.0;
var rf41: f32 = 0.0;
var rf42: f32 = 0.0;
var rf43: f32 = 0.0;
var rf44: f32 = 0.0;
var rf45: f32 = 0.0;
var rf46: f32 = 0.0;
var rf47: f32 = 0.0;
var rf48: f32 = 0.0;
var rf49: f32 = 0.0;
var rf50: f32 = 0.0;
var rf51: f32 = 0.0;
var rf52: f32 = 0.0;
var rf53: f32 = 0.0;
var rf54: f32 = 0.0;
var rf55: f32 = 0.0;
var rf56: f32 = 0.0;
var rf57: f32 = 0.0;
var rf58: f32 = 0.0;
var rf59: f32 = 0.0;
var rf60: f32 = 0.0;
var rf61: f32 = 0.0;
var rf62: f32 = 0.0;
var rf63: f32 = 0.0;
var ru0: u32 = 0u;
var ru1: u32 = 0u;
var ru2: u32 = 0u;
var ru3: u32 = 0u;
var ru4: u32 = 0u;
var ru5: u32 = 0u;
var ru6: u32 = 0u;
var ru7: u32 = 0u;
var ru8: u32 = 0u;
var ru9: u32 = 0u;
var ru10: u32 = 0u;
var ru11: u32 = 0u;
var ru12: u32 = 0u;
var ru13: u32 = 0u;
var ru14: u32 = 0u;
var ru15: u32 = 0u;
var ru16: u32 = 0u;
var ru17: u32 = 0u;
var ru18: u32 = 0u;
var ru19: u32 = 0u;
var ru20: u32 = 0u;
var ru21: u32 = 0u;
var ru22: u32 = 0u;
var ru23: u32 = 0u;
var ru24: u32 = 0u;
var ru25: u32 = 0u;
var ru26: u32 = 0u;
var ru27: u32 = 0u;
var ru28: u32 = 0u;
var ru29: u32 = 0u;
var ru30: u32 = 0u;
var ru31: u32 = 0u;
var ru32: u32 = 0u;
var ru33: u32 = 0u;
var ru34: u32 = 0u;
var ru35: u32 = 0u;
var ru36: u32 = 0u;
var ru37: u32 = 0u;
var ru38: u32 = 0u;
var ru39: u32 = 0u;
var ru40: u32 = 0u;
var ru41: u32 = 0u;
var ru42: u32 = 0u;
var ru43: u32 = 0u;
var ru44: u32 = 0u;
var ru45: u32 = 0u;
var ru46: u32 = 0u;
var ru47: u32 = 0u;
var ru48: u32 = 0u;
var ru49: u32 = 0u;
var ru50: u32 = 0u;
var ru51: u32 = 0u;
var ru52: u32 = 0u;
var ru53: u32 = 0u;
var ru54: u32 = 0u;
var ru55: u32 = 0u;
var ru56: u32 = 0u;
var ru57: u32 = 0u;
var ru58: u32 = 0u;
var ru59: u32 = 0u;
var ru60: u32 = 0u;
var ru61: u32 = 0u;
var ru62: u32 = 0u;
var ru63: u32 = 0u;
var rb0: bool = false;
var rb1: bool = false;
var rb2: bool = false;
var rb3: bool = false;
var rb4: bool = false;
var rb5: bool = false;
var rb6: bool = false;
var rb7: bool = false;
var rb8: bool = false;
var rb9: bool = false;
var rb10: bool = false;
var rb11: bool = false;
var rb12: bool = false;
var rb13: bool = false;
var rb14: bool = false;
var rb15: bool = false;
var rn0: i32 = 0i;
var rn1: i32 = 0i;
var rn2: i32 = 0i;
var rn3: i32 = 0i;
var rn4: i32 = 0i;
var rn5: i32 = 0i;
var rn6: i32 = 0i;
var rn7: i32 = 0i;
var rn8: i32 = 0i;
var rn9: i32 = 0i;
var rn10: i32 = 0i;
var rn11: i32 = 0i;
var rn12: i32 = 0i;
var rn13: i32 = 0i;
var rn14: i32 = 0i;
var rn15: i32 = 0i;
var rh0: f16 = 0.0h;
var rh1: f16 = 0.0h;
var rh2: f16 = 0.0h;
var rh3: f16 = 0.0h;
var rh4: f16 = 0.0h;
var rh5: f16 = 0.0h;
var rh6: f16 = 0.0h;
var rh7: f16 = 0.0h;
var rh8: f16 = 0.0h;
var rh9: f16 = 0.0h;
var rh10: f16 = 0.0h;
var rh11: f16 = 0.0h;
var rh12: f16 = 0.0h;
var rh13: f16 = 0.0h;
var rh14: f16 = 0.0h;
var rh15: f16 = 0.0h;
var v0: vec4<f32> = vec4<f32>();
var v1: vec4<f32> = vec4<f32>();
var v2: vec4<f32> = vec4<f32>();
var v3: vec4<f32> = vec4<f32>();
var v4: vec4<f32> = vec4<f32>();
var v5: vec4<f32> = vec4<f32>();
var v6: vec4<f32> = vec4<f32>();
var v7: vec4<f32> = vec4<f32>();
var v8: vec4<f32> = vec4<f32>();
var v9: vec4<f32> = vec4<f32>();
var v10: vec4<f32> = vec4<f32>();
var v11: vec4<f32> = vec4<f32>();
var v12: vec4<f32> = vec4<f32>();
var v13: vec4<f32> = vec4<f32>();
var v14: vec4<f32> = vec4<f32>();
var v15: vec4<f32> = vec4<f32>();
var v16: vec4<f32> = vec4<f32>();
var v17: vec4<f32> = vec4<f32>();
var v18: vec4<f32> = vec4<f32>();
var v19: vec4<f32> = vec4<f32>();
var v20: vec4<f32> = vec4<f32>();
var v21: vec4<f32> = vec4<f32>();
var v22: vec4<f32> = vec4<f32>();
var v23: vec4<f32> = vec4<f32>();
var v24: vec4<f32> = vec4<f32>();
var v25: vec4<f32> = vec4<f32>();
var v26: vec4<f32> = vec4<f32>();
var v27: vec4<f32> = vec4<f32>();
var v28: vec4<f32> = vec4<f32>();
var v29: vec4<f32> = vec4<f32>();
var v30: vec4<f32> = vec4<f32>();
var v31: vec4<f32> = vec4<f32>();
var w0: vec2<f32> = vec2<f32>();
var w1: vec2<f32> = vec2<f32>();
var w2: vec2<f32> = vec2<f32>();
var w3: vec2<f32> = vec2<f32>();
var w4: vec2<f32> = vec2<f32>();
var w5: vec2<f32> = vec2<f32>();
var w6: vec2<f32> = vec2<f32>();
var w7: vec2<f32> = vec2<f32>();
var w8: vec2<f32> = vec2<f32>();
var w9: vec2<f32> = vec2<f32>();
var w10: vec2<f32> = vec2<f32>();
var w11: vec2<f32> = vec2<f32>();
var w12: vec2<f32> = vec2<f32>();
var w13: vec2<f32> = vec2<f32>();
var w14: vec2<f32> = vec2<f32>();
var w15: vec2<f32> = vec2<f32>();
var C0: vec3<f32> = vec3<f32>();
var C1: vec3<f32> = vec3<f32>();
var C2: vec3<f32> = vec3<f32>();
var C3: vec3<f32> = vec3<f32>();
var C4: vec3<f32> = vec3<f32>();
var C5: vec3<f32> = vec3<f32>();
var C6: vec3<f32> = vec3<f32>();
var C7: vec3<f32> = vec3<f32>();
var C8: vec3<f32> = vec3<f32>();
var C9: vec3<f32> = vec3<f32>();
var C10: vec3<f32> = vec3<f32>();
var C11: vec3<f32> = vec3<f32>();
var C12: vec3<f32> = vec3<f32>();
var C13: vec3<f32> = vec3<f32>();
var C14: vec3<f32> = vec3<f32>();
var C15: vec3<f32> = vec3<f32>();
var rvu2_0: vec2<u32> = vec2<u32>();
var rvu2_1: vec2<u32> = vec2<u32>();
var rvu2_2: vec2<u32> = vec2<u32>();
var rvu2_3: vec2<u32> = vec2<u32>();
var rvu2_4: vec2<u32> = vec2<u32>();
var rvu2_5: vec2<u32> = vec2<u32>();
var rvu2_6: vec2<u32> = vec2<u32>();
var rvu2_7: vec2<u32> = vec2<u32>();
var rvu3_0: vec3<u32> = vec3<u32>();
var rvu3_1: vec3<u32> = vec3<u32>();
var rvu3_2: vec3<u32> = vec3<u32>();
var rvu3_3: vec3<u32> = vec3<u32>();
var rvu3_4: vec3<u32> = vec3<u32>();
var rvu3_5: vec3<u32> = vec3<u32>();
var rvu3_6: vec3<u32> = vec3<u32>();
var rvu3_7: vec3<u32> = vec3<u32>();
var rvu4_0: vec4<u32> = vec4<u32>();
var rvu4_1: vec4<u32> = vec4<u32>();
var rvu4_2: vec4<u32> = vec4<u32>();
var rvu4_3: vec4<u32> = vec4<u32>();
var rvu4_4: vec4<u32> = vec4<u32>();
var rvu4_5: vec4<u32> = vec4<u32>();
var rvu4_6: vec4<u32> = vec4<u32>();
var rvu4_7: vec4<u32> = vec4<u32>();
var rvi2_0: vec2<i32> = vec2<i32>();
var rvi2_1: vec2<i32> = vec2<i32>();
var rvi2_2: vec2<i32> = vec2<i32>();
var rvi2_3: vec2<i32> = vec2<i32>();
var rvi2_4: vec2<i32> = vec2<i32>();
var rvi2_5: vec2<i32> = vec2<i32>();
var rvi2_6: vec2<i32> = vec2<i32>();
var rvi2_7: vec2<i32> = vec2<i32>();
var rvi3_0: vec3<i32> = vec3<i32>();
var rvi3_1: vec3<i32> = vec3<i32>();
var rvi3_2: vec3<i32> = vec3<i32>();
var rvi3_3: vec3<i32> = vec3<i32>();
var rvi3_4: vec3<i32> = vec3<i32>();
var rvi3_5: vec3<i32> = vec3<i32>();
var rvi3_6: vec3<i32> = vec3<i32>();
var rvi3_7: vec3<i32> = vec3<i32>();
var rvi4_0: vec4<i32> = vec4<i32>();
var rvi4_1: vec4<i32> = vec4<i32>();
var rvi4_2: vec4<i32> = vec4<i32>();
var rvi4_3: vec4<i32> = vec4<i32>();
var rvi4_4: vec4<i32> = vec4<i32>();
var rvi4_5: vec4<i32> = vec4<i32>();
var rvi4_6: vec4<i32> = vec4<i32>();
var rvi4_7: vec4<i32> = vec4<i32>();
var rvb2_0: vec2<bool> = vec2<bool>();
var rvb2_1: vec2<bool> = vec2<bool>();
var rvb2_2: vec2<bool> = vec2<bool>();
var rvb2_3: vec2<bool> = vec2<bool>();
var rvb2_4: vec2<bool> = vec2<bool>();
var rvb2_5: vec2<bool> = vec2<bool>();
var rvb2_6: vec2<bool> = vec2<bool>();
var rvb2_7: vec2<bool> = vec2<bool>();
var rvb3_0: vec3<bool> = vec3<bool>();
var rvb3_1: vec3<bool> = vec3<bool>();
var rvb3_2: vec3<bool> = vec3<bool>();
var rvb3_3: vec3<bool> = vec3<bool>();
var rvb3_4: vec3<bool> = vec3<bool>();
var rvb3_5: vec3<bool> = vec3<bool>();
var rvb3_6: vec3<bool> = vec3<bool>();
var rvb3_7: vec3<bool> = vec3<bool>();
var rvb4_0: vec4<bool> = vec4<bool>();
var rvb4_1: vec4<bool> = vec4<bool>();
var rvb4_2: vec4<bool> = vec4<bool>();
var rvb4_3: vec4<bool> = vec4<bool>();
var rvb4_4: vec4<bool> = vec4<bool>();
var rvb4_5: vec4<bool> = vec4<bool>();
var rvb4_6: vec4<bool> = vec4<bool>();
var rvb4_7: vec4<bool> = vec4<bool>();
var rvh2_0: vec2<f16> = vec2<f16>();
var rvh2_1: vec2<f16> = vec2<f16>();
var rvh2_2: vec2<f16> = vec2<f16>();
var rvh2_3: vec2<f16> = vec2<f16>();
var rvh2_4: vec2<f16> = vec2<f16>();
var rvh2_5: vec2<f16> = vec2<f16>();
var rvh2_6: vec2<f16> = vec2<f16>();
var rvh2_7: vec2<f16> = vec2<f16>();
var rvh3_0: vec3<f16> = vec3<f16>();
var rvh3_1: vec3<f16> = vec3<f16>();
var rvh3_2: vec3<f16> = vec3<f16>();
var rvh3_3: vec3<f16> = vec3<f16>();
var rvh3_4: vec3<f16> = vec3<f16>();
var rvh3_5: vec3<f16> = vec3<f16>();
var rvh3_6: vec3<f16> = vec3<f16>();
var rvh3_7: vec3<f16> = vec3<f16>();
var rvh4_0: vec4<f16> = vec4<f16>();
var rvh4_1: vec4<f16> = vec4<f16>();
var rvh4_2: vec4<f16> = vec4<f16>();
var rvh4_3: vec4<f16> = vec4<f16>();
var rvh4_4: vec4<f16> = vec4<f16>();
var rvh4_5: vec4<f16> = vec4<f16>();
var rvh4_6: vec4<f16> = vec4<f16>();
var rvh4_7: vec4<f16> = vec4<f16>();
var x0: mat2x2<f32> = mat2x2<f32>();
var x1: mat2x2<f32> = mat2x2<f32>();
var x2: mat2x2<f32> = mat2x2<f32>();
var x3: mat2x2<f32> = mat2x2<f32>();
var y0: mat3x3<f32> = mat3x3<f32>();
var y1: mat3x3<f32> = mat3x3<f32>();
var y2: mat3x3<f32> = mat3x3<f32>();
var y3: mat3x3<f32> = mat3x3<f32>();
var z0: mat4x4<f32> = mat4x4<f32>();
var z1: mat4x4<f32> = mat4x4<f32>();
var z2: mat4x4<f32> = mat4x4<f32>();
var z3: mat4x4<f32> = mat4x4<f32>();
var pf0: array<f32, 64>;
var pf1: array<f32, 64>;
var pf2: array<f32, 64>;
var pf3: array<f32, 64>;
var pf4: array<f32, 64>;
var pf5: array<f32, 64>;
var pf6: array<f32, 64>;
var pf7: array<f32, 64>;
var pi0: array<u32, 16>;
var pi1: array<u32, 16>;
var pi2: array<u32, 16>;
var pi3: array<u32, 16>;
var pi4: array<u32, 16>;
var pj0: array<u32, 64>;
var pj1: array<u32, 64>;
var pk0: array<u32, 128>;
var pk1: array<u32, 128>;
var pb0: array<bool, 16>;
var pb1: array<bool, 16>;
var pb2: array<bool, 16>;
var pv0: array<vec4<f32>, 16>;
var pv1: array<vec4<f32>, 16>;
var pv2: array<vec4<f32>, 16>;
var pv3: array<vec4<f32>, 16>;
var pv4: array<vec4<f32>, 16>;
var pv5: array<vec4<f32>, 16>;
var pq0: array<vec2<f32>, 16>;
var pq1: array<vec2<f32>, 16>;
var pq2: array<vec2<f32>, 16>;
var pq3: array<vec2<f32>, 16>;
var pvq0: array<vec4<f32>, 64>;
var pvq1: array<vec4<f32>, 64>;
var pvq2: array<vec4<f32>, 64>;
var pvq3: array<vec4<f32>, 64>;
var n1: u32 = 0u;
var k1: u32 = 0u;
var nx1: u32 = 0u;
var kx1: u32 = 0u;
var n2: u32 = 0u;
var k2: u32 = 0u;
var nx2: u32 = 0u;
var kx2: u32 = 0u;
var n3: u32 = 0u;
var k3: u32 = 0u;
var nx3: u32 = 0u;
var kx3: u32 = 0u;
var n4: u32 = 0u;
var k4: u32 = 0u;
var nx4: u32 = 0u;
var kx4: u32 = 0u;
var n5: u32 = 0u;
var k5: u32 = 0u;
var nx5: u32 = 0u;
var kx5: u32 = 0u;
var U0: u32 = 0u;
var U1: u32 = 0u;
var U2: u32 = 0u;
var U3: u32 = 0u;
var U4: u32 = 0u;
var U5: u32 = 0u;
var U6: u32 = 0u;
var U7: u32 = 0u;
w1 = vec2<f32>(f32(textureDimensions(output).x), f32(textureDimensions(output).y)); w0 = vec2<f32>(f32(col_axis(gid)), f32(row_axis(gid))); C0 = zero_vec3_f32(); { rf3 = lit_f32(0.0); rf1 = lit_f32(0.0); rf0 = lit_f32(0.0); { rf4 = (time_data.time * params.speed); rf2 = lit_f32(0.0); n1 = lit_u32(50u); k1 = 4294967295u; loop { k1 = k1 + 1u; if (k1 >= n1) { break; } rf25 = (rf3 + lit_f32(1.0)); rf0 = (rf0 + (rf1 * lit_f32(0.6))); C1 = (rf0 * normalize(vec3<f32>(((w0 - (lit_f32(0.5) * w1)) / vec2<f32>(params.zoom)), w1[1]))); { C1 = vec3<f32>(C1.xy, (C1[2] + rf4)); rf10 = (C1[2] * lit_f32(1.1)); rf8 = cos(rf10); { rf9 = sin(rf10); w2 = (mat2x2<f32>(vec2<f32>(rf8, (-rf9)), vec2<f32>(rf9, rf8)) * C1.xy); C1 = vec3<f32>(w2, C1[2]); { C2 = (C1 + vec3<f32>((rf4 * lit_f32(0.1)))); rf6 = lit_f32(0.0); rf7 = lit_f32(1.0); { n2 = lit_u32(7u); k2 = 4294967295u; loop { k2 = k2 + 1u; if (k2 >= n2) { break; } w3 = (C2.xy + vec2<f32>((C2[2] * lit_f32(0.5)))); rf11 = (sin(((w3[0] * lit_f32(3.0)) + sin((w3[1] * lit_f32(2.7))))) * cos(((w3[1] * lit_f32(1.1)) + cos((lit_f32(2.3) * w3[0]))))); rf6 = (rf6 + (rf11 * rf7)); { C2 = (lit_f32(2.0) * C2); rf7 = (rf7 * lit_f32(0.5)); } } rf2 = rf6; rf1 = ((length(C1.xy) - lit_f32(1.0)) + (rf2 * lit_f32(0.3))); { rf12 = (C1[2] % lit_f32(4.0)); C1 = vec3<f32>(C1.xy, (select((rf12 + lit_f32(4.0)), rf12, (rf12 >= lit_f32(0.0))) - (lit_f32(4.0) * lit_f32(0.5)))); rf1 = (abs(rf1) + lit_f32(0.01)); { rf5 = max((length((C1.xy + vec2<f32>(rf2))) * lit_f32(0.8)), lit_f32(0.005)); rf13 = max((length((C1.xz + vec2<f32>(rf2))) * lit_f32(0.8)), lit_f32(0.005)); C3 = (vec3<f32>(vec2<f32>(params.color1_r, params.color1_g), params.color1_b) / vec3<f32>(rf5)); { C3 = (C3 + (vec3<f32>(vec2<f32>(params.color2_r, params.color2_g), params.color2_b) / vec3<f32>(rf13))); rf14 = (lit_f32(0.5) + (lit_f32(0.5) * sin(((rf2 * lit_f32(1.0)) + rf4)))); C3 = (C3 + (rf14 * vec3<f32>(vec2<f32>(params.color3_r, params.color3_g), params.color3_b))); C0 = (C0 + (lit_f32(0.12) * (C3 / vec3<f32>(rf1)))); } } } } } } } } } }
{ C4 = tanh((params.intensity * C0)); C5 = (mat3x3<f32>(vec3<f32>(vec2<f32>(lit_f32(0.59719), lit_f32(0.07600)), lit_f32(0.02840)), vec3<f32>(vec2<f32>(lit_f32(0.35458), lit_f32(0.90834)), lit_f32(0.13383)), vec3<f32>(vec2<f32>(lit_f32(0.04823), lit_f32(0.01566)), lit_f32(0.83777))) * C4); C6 = ((C5 * (C5 + vec3<f32>(lit_f32(0.0245786)))) - vec3<f32>(lit_f32(0.000090537))); { C7 = ((C5 * ((C5 * vec3<f32>(lit_f32(0.983729))) + vec3<f32>(lit_f32(0.4329510)))) + vec3<f32>(lit_f32(0.238081))); C4 = (mat3x3<f32>(vec3<f32>(vec2<f32>(lit_f32(1.60475), lit_f32(-0.10208)), lit_f32(-0.00327)), vec3<f32>(vec2<f32>(lit_f32(-0.53108), lit_f32(1.10813)), lit_f32(-0.07276)), vec3<f32>(vec2<f32>(lit_f32(-0.07367), lit_f32(-0.00605)), lit_f32(1.07602))) * (C6 / C7)); } } C4 = max(C4, zero_vec3_f32()); C4 = pow(C4, vec3<f32>((lit_f32(1.0) / params.gamma))); { let r = (row_axis(gid) * KNOB_BROW); let c = (col_axis(gid) * KNOB_BCOL); if (r < textureDimensions(output).y && c < textureDimensions(output).x) { textureStore(output, vec2<u32>(c, r), vec4<f32>(C4, lit_f32(1.0))); } }
}
