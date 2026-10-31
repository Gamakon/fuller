@group(0) @binding(0) var<storage, read> vectors: array<f32>;
@group(0) @binding(1) var<storage, read> assignments: array<u32>;
@group(0) @binding(2) var<storage, read_write> centroids: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
var<workgroup> wg_pool: array<u32, 1028>;
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
fn col_axis(g: vec3<u32>) -> u32 { return g.x; }
fn combine_hash(a: u32, b: u32) -> u32 { return fmix32(a ^ (b * 0x9e3779b9u + (a << 6u) + (a >> 2u))); }
fn fmix32(h0: u32) -> u32 { var h = h0; h = h ^ (h >> 16u); h = h * 0x85ebca6bu; h = h ^ (h >> 13u); h = h * 0xc2b2ae35u; h = h ^ (h >> 16u); return h; }
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
fn own_load_centroids(r: u32, c: u32) -> f32 { return centroids[min(r * (params.y * params.z) + c, arrayLength(&centroids) - 1u)]; }
fn row_axis(g: vec3<u32>) -> u32 { return g.y; }
fn sat_add_u32(x: u32, y: u32) -> u32 { let s = x + y; return select(s, 0xffffffffu, s < x); }
fn seg_at(i: u32) -> u32 { return wg_seg[min(i, 32u)]; }
fn sorted(a: f32, b: f32) -> vec2<f32> { return vec2<f32>(min(a, b), max(a, b)); }
fn sorted_i(a: i32, b: i32) -> vec2<i32> { return vec2<i32>(min(a, b), max(a, b)); }
fn sub_floor(a: u32, b: u32) -> u32 { return a - min(a, b); }
fn sub_sat(a: i32, b: i32) -> i32 { let s = a - b; let ov = ((a ^ b) & (a ^ s)) < 0i; return select(s, select(2147483647i, (-2147483647i - 1i), a < 0i), ov); }
fn wg_set_v4(o: u32, ls: u32, v: vec4<f32>) { if (o < 1024u) { wg_pool[o] = bitcast<u32>(v.x); } if (o + ls < 1024u) { wg_pool[o + ls] = bitcast<u32>(v.y); } if (o + 2u * ls < 1024u) { wg_pool[o + 2u * ls] = bitcast<u32>(v.z); } if (o + 3u * ls < 1024u) { wg_pool[o + 3u * ls] = bitcast<u32>(v.w); } }
fn wg_set_v4_below(o: u32, ls: u32, v: vec4<f32>, sl: u32, sp: u32) { if (o >= sl && o < sp) { wg_pool[o] = bitcast<u32>(v.x); } if (o + ls >= sl && o + ls < sp) { wg_pool[o + ls] = bitcast<u32>(v.y); } if (o + 2u * ls >= sl && o + 2u * ls < sp) { wg_pool[o + 2u * ls] = bitcast<u32>(v.z); } if (o + 3u * ls >= sl && o + 3u * ls < sp) { wg_pool[o + 3u * ls] = bitcast<u32>(v.w); } }
fn wg_v4(b: u32, ls: u32, i: u32) -> vec4<f32> { return vec4<f32>(bitcast<f32>(wg_pool[min(b + i, 1024u)]), bitcast<f32>(wg_pool[min(b + ls + i, 1024u)]), bitcast<f32>(wg_pool[min(b + 2u * ls + i, 1024u)]), bitcast<f32>(wg_pool[min(b + 3u * ls + i, 1024u)])); }
fn wg_v4_above(b: u32, ls: u32, i: u32, sl: u32, sp: u32) -> vec4<f32> { return vec4<f32>(bitcast<f32>(wg_pool[xw(b + i, sl, sp)]), bitcast<f32>(wg_pool[xw(b + ls + i, sl, sp)]), bitcast<f32>(wg_pool[xw(b + 2u * ls + i, sl, sp)]), bitcast<f32>(wg_pool[xw(b + 3u * ls + i, sl, sp)])); }
fn xidx(p: u32, h: u32, m: u32) -> u32 { if (m == 0u) { return p; } let lo = p & (h - 1u); return ((p & ~(h - 1u)) << 1u) | lo; }
fn xok(p: u32, h: u32, m: u32) -> bool { if (m == 0u) { return (p & h) == 0u; } return p < 2147483648u; }
fn xw(a: u32, sl: u32, sp: u32) -> u32 { return select(min(a, 1024u), 1024u, a >= sl && a < sp); }
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
