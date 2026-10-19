// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var t0: u32 = 0u; var t1: u32 = 0u; var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u;
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u;
    var pa: array<f32, 64>; var pb: array<f32, 64>; var pc: array<f32, 64>; var pd: array<f32, 64>; var pe: array<f32, 64>;
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    for (t0 = 0u; t0 < cfg.m; t0 = t0 + 1024u) {
    for (k = 0u; k < 16u; k = k + 1u) {
    pb[((k) & 63u)] = 0.0f;
    }
    for (t1 = 0u; t1 < min(1024u, cfg.m - t0); t1 = t1 + 32u) {
    t = t0 + t1;
    for (k = 0u; k < 8u; k = k + 1u) {
    i0 = lidx + (256u * k);
    i1 = i0 >> 6u;
    i2 = i0 & 63u;
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + i1) < cfg.m)) {
    if ((((wid.x * 64u) + i2) < cfg.d)) {
    f0 = phi[((((t + i1) * cfg.d) + (wid.x * 64u)) + i2)];
    }
    if (((((wid.y * 64u) + i2) < cfg.r_rows) && (((cfg.r_off + (wid.y * 64u)) + i2) < cfg.d))) {
    f1 = phi[(((((t + i1) * cfg.d) + cfg.r_off) + (wid.y * 64u)) + i2)];
    }
    }
    wg_a[((i0) & 2047u)] = f0;
    wg_b[((i0) & 2047u)] = f1;
    }
    workgroupBarrier();
    for (k = 0u; k < 16u; k = k + 1u) {
    pc[((k) & 63u)] = 0.0f;
    }
    for (k = 0u; k < 32u; k = k + 1u) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    pd[((k2) & 63u)] = wg_a[(((((k * 64u) + (lid.x * 4u)) + k2)) & 2047u)];
    pe[((k2) & 63u)] = wg_b[(((((k * 64u) + (lid.y * 4u)) + k2)) & 2047u)];
    }
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
    pc[((((k2 * 4u) + k3)) & 63u)] = (pc[((((k2 * 4u) + k3)) & 63u)] + (pe[((k2) & 63u)] * pd[((k3) & 63u)]));
    }
    }
    }
    for (k = 0u; k < 16u; k = k + 1u) {
    pb[((k) & 63u)] = (pb[((k) & 63u)] + pc[((k) & 63u)]);
    }
    workgroupBarrier();
    }
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[((k) & 63u)] = (pa[((k) & 63u)] + pb[((k) & 63u)]);
    }
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pa[((((k * 4u) + k2)) & 63u)]; }
    }
    }
}
