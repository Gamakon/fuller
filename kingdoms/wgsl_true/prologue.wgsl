enable f16;
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
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
fn lit_bool(x: bool) -> bool { return x; }
fn lit_f16(x: f16) -> f16 { return x; }
fn lit_f32(x: f32) -> f32 { return x; }
fn lit_i32(x: i32) -> i32 { return x; }
fn lit_u32(x: u32) -> u32 { return x; }
fn lo_width(lo: f32, w: f32) -> vec2<f32> { return vec2<f32>(lo, lo + w); }
fn lo_width_i(lo: i32, w: i32) -> vec2<i32> { return vec2<i32>(lo, max(lo, lo + w)); }
fn load4_phi(k: u32) -> vec4<f32> { let m = arrayLength(&phi) - 1u; return vec4<f32>(phi[min(k, m)], phi[min(k + 1u, m)], phi[min(k + 2u, m)], phi[min(k + 3u, m)]); }
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
fn own_load_h(r: u32, c: u32) -> f32 { return h[min(r * cfg.d + c, arrayLength(&h) - 1u)]; }
fn sat_add_u32(x: u32, y: u32) -> u32 { let s = x + y; return select(s, 0xffffffffu, s < x); }
fn shuffle_down_f(x: f32, d: u32) -> f32 { let dd = min(d, sg_size); return select(subgroupShuffleDown(x, dd), x, sg_lane + dd >= sg_size); }
fn shuffle_down_i(x: u32, d: u32) -> u32 { let dd = min(d, sg_size); return select(subgroupShuffleDown(x, dd), x, sg_lane + dd >= sg_size); }
fn shuffle_up_f(x: f32, d: u32) -> f32 { let dd = min(d, sg_size); return select(subgroupShuffleUp(x, dd), x, sg_lane < dd); }
fn shuffle_up_i(x: u32, d: u32) -> u32 { let dd = min(d, sg_size); return select(subgroupShuffleUp(x, dd), x, sg_lane < dd); }
fn sorted(a: f32, b: f32) -> vec2<f32> { return vec2<f32>(min(a, b), max(a, b)); }
fn sorted_i(a: i32, b: i32) -> vec2<i32> { return vec2<i32>(min(a, b), max(a, b)); }
fn sub_floor(a: u32, b: u32) -> u32 { return a - min(a, b); }
fn sub_sat(a: i32, b: i32) -> i32 { let s = a - b; let ov = ((a ^ b) & (a ^ s)) < 0i; return select(s, select(2147483647i, (-2147483647i - 1i), a < 0i), ov); }
