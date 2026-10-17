// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var i4: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + (((((lid.y * 16u) + lid.x) + (256u * k))) >> ((6u) & 31u))) < cfg.m)) {
    if (((((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (256u * k))) & (63u))) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (256u * k))) & (63u)))) < cfg.d))) {
    f0 = phi[(((t + (((((lid.y * 16u) + lid.x) + (256u * k))) >> ((6u) & 31u))) * cfg.d) + (cfg.r_off + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (256u * k))) & (63u)))))];
    }
    if ((((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (256u * k))) & (63u))) < cfg.d)) {
    f1 = phi[(((t + (((((lid.y * 16u) + lid.x) + (256u * k))) >> ((6u) & 31u))) * cfg.d) + ((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (256u * k))) & (63u))))];
    }
    }
    wg_a[(((((lid.y * 16u) + lid.x) + (256u * k))) & 1023u)] = f0;
    wg_b[(((((lid.y * 16u) + lid.x) + (256u * k))) & 1023u)] = f1;
    }
    workgroupBarrier();
    for (k = 0u; k < 16u; k = k + 1u) {
    i1 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    pb[((k2) & 63u)] = wg_a[(((((k * 64u) + lid.y) + (16u * k2))) & 1023u)];
    pc[((k2) & 63u)] = wg_b[(((((k * 64u) + lid.x) + (16u * k2))) & 1023u)];
    }
    i2 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    i3 = 0u;
    for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
    pa[((((k2 * 4u) + k3)) & 63u)] = (pa[((((k2 * 4u) + k3)) & 63u)] + (pb[((k2) & 63u)] * pc[((k3) & 63u)]));
    }
    }
    }
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if (((((wid.y * 64u + lid.y) + min(k, 3u) * 16u) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) < cfg.d))) {
    i4 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((wid.x * 64u + lid.x) + min(k2, 3u) * 16u) < cfg.d)) {
    if ((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) < cfg.r_rows && (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)) < cfg.d && (cfg.r_off + (((wid.y * 64u + lid.y) + min(k, 3u) * 16u))) < cfg.d) { h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] = h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] + pa[((((k * 4u) + k2)) & 63u)]; }
    }
    }
    }
    }
}
