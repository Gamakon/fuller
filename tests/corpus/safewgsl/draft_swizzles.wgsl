// grid 16 16 1 1
// Swizzles (2026-10-12): 2, 3 and 4 lanes, cross on a 3-lane swizzle, pack2x16snorm of a 2-lane swizzle.
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

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
    let q = v.zyx * f0;
    let r = v.xy + vec2<f32>(f0, 1.0);
    let c = cross(v.xyz, vec3<f32>(1.0, 0.0, 0.0));
    g = q.x + r.y + length(v.zyx) + c.y + f32(pack2x16snorm(v.xy * 0.5) & 255u) + v.wzyx.w;
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0 + g * 0.0;
    }
}
