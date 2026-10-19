// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pa: array<f32, 64>; var pb: array<f32, 64>; var pc: array<f32, 64>;
    for (t = 0u; t < cfg.m; t = t + 1u) {
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    f0 = 0.0f;
    if (((((wid.x * 64u) + (lid.x * 4u)) + k) < cfg.d)) {
    f0 = phi[(((t * cfg.d) + ((wid.x * 64u) + (lid.x * 4u))) + k)];
    }
    pa[((k) & 63u)] = f0;
    }
    i1 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    f1 = 0.0f;
    if ((((((wid.y * 64u) + (lid.y * 8u)) + k) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + (lid.y * 8u))) + k) < cfg.d))) {
    f1 = phi[((((t * cfg.d) + cfg.r_off) + ((wid.y * 64u) + (lid.y * 8u))) + k)];
    }
    pb[((k) & 63u)] = f1;
    }
    i2 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    i3 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    pc[((((k * 4u) + k2)) & 63u)] = (pc[((((k * 4u) + k2)) & 63u)] + (pb[((k) & 63u)] * pa[((k2) & 63u)]));
    }
    }
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    i4 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((((wid.y * 64u) + (lid.y * 8u)) + k) < cfg.r_rows) && ((((wid.x * 64u) + (lid.x * 4u)) + k2) < cfg.d))) {
    if ((((gid.y * 8u) + min(k, 7u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 8u) + min(k, 7u)))) < cfg.d) { h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pc[((((k * 4u) + k2)) & 63u)]; }
    }
    }
    }
}
