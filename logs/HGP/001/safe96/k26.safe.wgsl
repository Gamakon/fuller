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
    var pj: array<u32, 16>;
    var pk: array<u32, 16>;
    var pm: array<bool, 16>;
    var pn: array<bool, 16>;
    for (k = 0u; k < 4u; k = k + 1u) {
    pm[((k) & 15u)] = ((((wid.y * 32u + lid.y) + min(k, 3u) * 8u) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 32u + lid.y) + min(k, 3u) * 8u)) < cfg.d));
    pi[((k) & 15u)] = ((wid.y * 32u + lid.y) + min(k, 3u) * 8u);
    pj[((k) & 15u)] = select(0u, (cfg.r_off + ((wid.y * 32u + lid.y) + min(k, 3u) * 8u)), ((((wid.y * 32u + lid.y) + min(k, 3u) * 8u) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 32u + lid.y) + min(k, 3u) * 8u)) < cfg.d)));
    pn[((k) & 15u)] = (((wid.x * 32u + lid.x) + min(k, 3u) * 8u) < cfg.d);
    pk[((k) & 15u)] = select(0u, ((wid.x * 32u + lid.x) + min(k, 3u) * 8u), (((wid.x * 32u + lid.x) + min(k, 3u) * 8u) < cfg.d));
    }
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    for (t = 0u; t < min((((cfg.m) - ((cfg.m) % max(4u, 1u)))), 16777216u); t = t + 4u) {
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    i1 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    pb[((k2) & 63u)] = phi[(((t + k) * cfg.d) + pj[((k2) & 15u)])];
    pc[((k2) & 63u)] = phi[(((t + k) * cfg.d) + pk[((k2) & 15u)])];
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
    i4 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    pd[((k) & 63u)] = phi[((t * cfg.d) + pj[((k) & 15u)])];
    pe[((k) & 63u)] = phi[((t * cfg.d) + pk[((k) & 15u)])];
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
    if (pm[((k) & 15u)]) {
    i7 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if (pn[((k2) & 15u)]) {
    if ((((wid.y * 32u + lid.y) + min(k, 3u) * 8u)) < cfg.r_rows && (((wid.x * 32u + lid.x) + min(k2, 3u) * 8u)) < cfg.d && (cfg.r_off + (((wid.y * 32u + lid.y) + min(k, 3u) * 8u))) < cfg.d) { h[((((wid.y * 32u + lid.y) + min(k, 3u) * 8u)) * cfg.d + (((wid.x * 32u + lid.x) + min(k2, 3u) * 8u)))] = h[((((wid.y * 32u + lid.y) + min(k, 3u) * 8u)) * cfg.d + (((wid.x * 32u + lid.x) + min(k2, 3u) * 8u)))] + pa[((((k * 4u) + k2)) & 63u)]; }
    }
    }
    }
    }
}
