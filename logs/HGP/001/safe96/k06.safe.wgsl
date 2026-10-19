// grid 8 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var i4: u32 = 0u;
    var i5: u32 = 0u;
    var i6: u32 = 0u;
    var i7: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    var pd: array<f32, 64>;
    var pe: array<f32, 64>;
    var pi: array<u32, 16>;
    var pm: array<bool, 16>;
    var pn: array<bool, 16>;
    for (k = 0u; k < 4u; k = k + 1u) {
    pm[((k) & 15u)] = ((((gid.y * 4u) + min(k, 3u)) < cfg.r_rows) & ((cfg.r_off + ((gid.y * 4u) + min(k, 3u))) < cfg.d));
    pi[((k) & 15u)] = (cfg.r_off + ((gid.y * 4u) + min(k, 3u)));
    pn[((k) & 15u)] = (((gid.x * 4u) + min(k, 3u)) < cfg.d);
    }
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    for (t = 0u; t < min((select(((cfg.m + 1u)) - (4u), 0u, (4u) > ((cfg.m + 1u)))), 16777216u); t = t + 4u) {
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    pb[0u] = 0.0f;
    pb[1u] = 0.0f;
    pb[2u] = 0.0f;
    pb[3u] = 0.0f;
    pc[0u] = 0.0f;
    pc[1u] = 0.0f;
    pc[2u] = 0.0f;
    pc[3u] = 0.0f;
    i1 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    pb[((k2) & 63u)] = select(0.0f, phi[(((t + k) * cfg.d) + pi[((k2) & 15u)])], pm[((k2) & 15u)]);
    pc[((k2) & 63u)] = select(0.0f, phi[((((t + k) * cfg.d) + (gid.x * 4u)) + k2)], pn[((k2) & 15u)]);
    }
    i2 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    i3 = 0u;
    for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
    pa[((((k2 * 4u) + k3)) & 63u)] = (pa[((((k2 * 4u) + k3)) & 63u)] + (pb[((k2) & 63u)] * pc[((k3) & 63u)]));
    }
    }
    }
    }
    for (; t < cfg.m; t = t + 1u) {
    pd[0u] = 0.0f;
    pd[1u] = 0.0f;
    pd[2u] = 0.0f;
    pd[3u] = 0.0f;
    pe[0u] = 0.0f;
    pe[1u] = 0.0f;
    pe[2u] = 0.0f;
    pe[3u] = 0.0f;
    i4 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    pd[((k) & 63u)] = select(0.0f, phi[((t * cfg.d) + pi[((k) & 15u)])], pm[((k) & 15u)]);
    pe[((k) & 63u)] = select(0.0f, phi[(((t * cfg.d) + (gid.x * 4u)) + k)], pn[((k) & 15u)]);
    }
    i5 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    i6 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    pa[((((k * 4u) + k2)) & 63u)] = (pa[((((k * 4u) + k2)) & 63u)] + (pd[((k) & 63u)] * pe[((k2) & 63u)]));
    }
    }
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((((gid.y * 4u) + min(k, 3u)) < cfg.r_rows)) {
    i7 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * 4u) + min(k2, 3u)) < cfg.d)) {
    if (((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows) & ((((gid.x * 4u) + min(k2, 3u))) < cfg.d) & ((cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d)) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pa[((((k * 4u) + k2)) & 63u)]; }
    }
    }
    }
    }
}
