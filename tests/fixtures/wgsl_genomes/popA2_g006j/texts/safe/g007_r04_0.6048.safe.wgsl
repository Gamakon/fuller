// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    var f4: f32 = 0.0;
    var f5: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var i4: u32 = 0u;
    var i5: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    var pd: array<f32, 64>;
    var pe: array<f32, 64>;
    for (k = 0u; k < 32u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    for (t = 0u; t < min(((cfg.m / 2u)), 16777216u); t = t + 1u) {
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((((gid.y * 8u) + min(k, 7u)) < cfg.r_rows) & (((cfg.r_off + (gid.y * 8u)) + k) < cfg.d))) {
    f0 = phi[(((((t * 2u) * cfg.d) + cfg.r_off) + (gid.y * 8u)) + k)];
    f1 = phi[((((((t * 2u) * cfg.d) + cfg.d) + cfg.r_off) + (gid.y * 8u)) + k)];
    }
    pb[((k) & 63u)] = f0;
    pc[((k) & 63u)] = f1;
    }
    i1 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    f2 = 0.0f;
    f3 = 0.0f;
    if ((((gid.x * 4u) + min(k, 3u)) < cfg.d)) {
    f2 = phi[((((t * 2u) * cfg.d) + (gid.x * 4u)) + k)];
    f3 = phi[(((((t * 2u) * cfg.d) + cfg.d) + (gid.x * 4u)) + k)];
    }
    pd[((k) & 63u)] = f2;
    pe[((k) & 63u)] = f3;
    }
    i2 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    i3 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    pa[((((k * 4u) + k2)) & 63u)] = (pa[((((k * 4u) + k2)) & 63u)] + ((pb[((k) & 63u)] * pd[((k2) & 63u)]) + (pc[((k) & 63u)] * pe[((k2) & 63u)])));
    }
    }
    }
    if ((((cfg.m) & (1u)) == 1u)) {
    for (k = 0u; k < 8u; k = k + 1u) {
    f4 = 0.0f;
    if (((((gid.y * 8u) + min(k, 7u)) < cfg.r_rows) & (((cfg.r_off + (gid.y * 8u)) + k) < cfg.d))) {
    f4 = phi[((((select((cfg.m) - (1u), 0u, (1u) > (cfg.m)) * cfg.d) + cfg.r_off) + (gid.y * 8u)) + k)];
    }
    pb[((k) & 63u)] = f4;
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    f5 = 0.0f;
    if ((((gid.x * 4u) + min(k, 3u)) < cfg.d)) {
    f5 = phi[(((select((cfg.m) - (1u), 0u, (1u) > (cfg.m)) * cfg.d) + (gid.x * 4u)) + k)];
    }
    pd[((k) & 63u)] = f5;
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    i4 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    pa[((((k * 4u) + k2)) & 63u)] = (pa[((((k * 4u) + k2)) & 63u)] + (pb[((k) & 63u)] * pd[((k2) & 63u)]));
    }
    }
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    i5 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((((gid.y * 8u) + min(k, 7u)) < cfg.r_rows) & (((cfg.r_off + (gid.y * 8u)) + k) < cfg.d)) & (((gid.x * 4u) + min(k2, 3u)) < cfg.d))) {
    if (((((gid.y * 8u) + min(k, 7u))) < cfg.r_rows) & ((((gid.x * 4u) + min(k2, 3u))) < cfg.d) & ((cfg.r_off + (((gid.y * 8u) + min(k, 7u)))) < cfg.d)) { h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pa[((((k * 4u) + k2)) & 63u)]; }
    }
    }
    }
}
