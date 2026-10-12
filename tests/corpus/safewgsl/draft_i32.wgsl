// grid 16 16 1 1
// i32 (2026-10-12): an i32 register, i32(u32), u32(i32), bitcast, ++, clamp, sign, the bit counts, shifts, an atomic<i32>'s old value read, subgroupMax on i32.
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> tally: array<atomic<i32>, 64>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0;
    var g: f32 = 0.0;
    var n: u32 = gid.x;
    var v: vec4<f32> = vec4<f32>(0.5);
    let gr = cfg.r_off + gid.y;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        f0 = f0 + phi[t * cfg.d + gr] * phi[t * cfg.d + gid.x];
    }
    var s: i32 = i32(n) - 3;
    s++;
    s = clamp(s * 3 - 1, -50, 50) / 2 % 7;
    let b = countOneBits(s) + firstLeadingBit(s) + extractBits(s, 1u, 3u) + (reverseBits(s) & 1) + (s >> 1u) + (~s & 3);
    let old = atomicAdd(&tally[lidx & 63u], -1);
    let w = bitcast<u32>(s) & 255u;
    g = f32(abs(s)) + f32(sign(s)) + f32(b) + f32(old) * 0.0 + f32(w) + f32(u32(s & 7)) + f32(subgroupMax(s)) + f32(i32(f0));
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0 + g * 0.0;
    }
}
