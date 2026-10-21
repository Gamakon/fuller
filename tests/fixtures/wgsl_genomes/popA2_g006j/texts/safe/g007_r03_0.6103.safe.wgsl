// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f21: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    var pm: array<bool, 16>;
    var pn: array<bool, 16>;
    i0 = 0u;
    for (k = 0u; k < 32u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    pm[((k) & 15u)] = ((((gid.y * 8u) + min(k, 7u)) < cfg.r_rows) & (((cfg.r_off + (gid.y * 8u)) + k) < cfg.d));
    }
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    pn[((k) & 15u)] = (((gid.x * 4u) + min(k, 3u)) < cfg.d);
    }
    i1 = 0u;
    for (t = 0u; t < cfg.m; t = t + 1u) {
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    f0 = 0.0f;
    if (pm[((k) & 15u)]) {
    f0 = phi[((((t * cfg.d) + cfg.r_off) + (gid.y * 8u)) + k)];
    }
    pb[((k) & 63u)] = f0;
    }
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    f21 = 0.0f;
    if (pn[((k) & 15u)]) {
    f1 = phi[(((t * cfg.d) + (gid.x * 4u)) + k)];
    }
    pc[((k) & 63u)] = f1;
    }
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    i2 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    pa[((((k * 4u) + k2)) & 63u)] = (pa[((((k * 4u) + k2)) & 63u)] + (pb[((k) & 63u)] * pc[((k2) & 63u)]));
    }
    }
    }
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    i2 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((pm[((k) & 15u)] & pn[((k2) & 15u)])) {
    if (((((gid.y * 8u) + min(k, 7u))) < cfg.r_rows) & ((((gid.x * 4u) + min(k2, 3u))) < cfg.d) & ((cfg.r_off + (((gid.y * 8u) + min(k, 7u)))) < cfg.d)) { h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pa[((((k * 4u) + k2)) & 63u)]; }
    }
    }
    }
}
