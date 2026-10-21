// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    i0 = 0u;
    for (k = 0u; k < 32u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    i1 = 0u;
    for (t = 0u; t < cfg.m; t = t + 1u) {
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    pb[((k) & 63u)] = 0.0f;
    if (((((gid.y * 8u) + min(k, 7u)) < cfg.r_rows) & ((cfg.r_off + ((gid.y * 8u) + min(k, 7u))) < cfg.d))) {
    pb[((k) & 63u)] = phi[((t * cfg.d) + (cfg.r_off + ((gid.y * 8u) + min(k, 7u))))];
    }
    }
    i2 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    pc[((k) & 63u)] = 0.0f;
    if ((((gid.x * 4u) + min(k, 3u)) < cfg.d)) {
    pc[((k) & 63u)] = phi[((t * cfg.d) + ((gid.x * 4u) + min(k, 3u)))];
    }
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
    if (((((gid.y * 8u) + min(k, 7u)) < cfg.r_rows) & (((gid.x * 4u) + min(k2, 3u)) < cfg.d))) {
    if (((((gid.y * 8u) + min(k, 7u))) < cfg.r_rows) & ((((gid.x * 4u) + min(k2, 3u))) < cfg.d) & ((cfg.r_off + (((gid.y * 8u) + min(k, 7u)))) < cfg.d)) { h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pa[((((k * 4u) + k2)) & 63u)]; }
    }
    }
    }
}
