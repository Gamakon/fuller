// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    i0 = 0u;
    for (k = 0u; k < 32u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    i1 = 0u;
    for (t = 0u; t < min((((cfg.m + 31u) / 32u)), 16777216u); t = t + 1u) {
    i0 = 0u;
    for (k = 0u; k < 32u; k = k + 1u) {
    pb[((k) & 63u)] = 0.0f;
    pc[((k) & 63u)] = 0.0f;
    }
    i0 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    f0 = 0.0f;
    if (((((t * 32u) + ((lidx + (k * 128u)) / 64u)) < cfg.m) & (((wid.x * 64u) + ((lidx + (k * 128u)) % 64u)) < cfg.d))) {
    f0 = phi[(((((t * 32u) + ((lidx + (k * 128u)) / 64u)) * cfg.d) + (wid.x * 64u)) + ((lidx + (k * 128u)) % 64u))];
    }
    wg_a[(((lidx + (k * 128u))) & 2047u)] = f0;
    f1 = 0.0f;
    if ((((((t * 32u) + ((lidx + (k * 128u)) / 64u)) < cfg.m) & (((wid.y * 64u) + ((lidx + (k * 128u)) % 64u)) < cfg.r_rows)) & (((cfg.r_off + (wid.y * 64u)) + ((lidx + (k * 128u)) % 64u)) < cfg.d))) {
    f1 = phi[((((((t * 32u) + ((lidx + (k * 128u)) / 64u)) * cfg.d) + cfg.r_off) + (wid.y * 64u)) + ((lidx + (k * 128u)) % 64u))];
    }
    wg_b[(((lidx + (k * 128u))) & 2047u)] = f1;
    }
    workgroupBarrier();
    i0 = 0u;
    for (k = 0u; k < 32u; k = k + 2u) {
    i2 = 0u;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    i3 = 0u;
    for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
    pb[((((k2 * 4u) + k3)) & 63u)] = (pb[((((k2 * 4u) + k3)) & 63u)] + (wg_b[(((((k * 64u) + (lid.y * 8u)) + k2)) & 2047u)] * wg_a[(((((k * 64u) + (lid.x * 4u)) + k3)) & 2047u)]));
    pc[((((k2 * 4u) + k3)) & 63u)] = (pc[((((k2 * 4u) + k3)) & 63u)] + (wg_b[((((((k + 1u) * 64u) + (lid.y * 8u)) + k2)) & 2047u)] * wg_a[((((((k + 1u) * 64u) + (lid.x * 4u)) + k3)) & 2047u)]));
    }
    }
    }
    i0 = 0u;
    for (k = 0u; k < 32u; k = k + 1u) {
    pa[((k) & 63u)] = (pa[((k) & 63u)] + (pb[((k) & 63u)] + pc[((k) & 63u)]));
    }
    workgroupBarrier();
    }
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    i2 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((((gid.y * 8u) + min(k, 7u)) < cfg.r_rows) & (((gid.x * 4u) + min(k2, 3u)) < cfg.d)) & (((cfg.r_off + (gid.y * 8u)) + k) < cfg.d))) {
    if (((((gid.y * 8u) + min(k, 7u))) < cfg.r_rows) & ((((gid.x * 4u) + min(k2, 3u))) < cfg.d) & ((cfg.r_off + (((gid.y * 8u) + min(k, 7u)))) < cfg.d)) { h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pa[((((k * 4u) + k2)) & 63u)]; }
    }
    }
    }
}
