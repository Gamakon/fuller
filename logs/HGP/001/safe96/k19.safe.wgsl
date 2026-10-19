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
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    var f4: f32 = 0.0;
    var f5: f32 = 0.0;
    var f6: f32 = 0.0;
    var f7: f32 = 0.0;
    var f8: f32 = 0.0;
    var f9: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) < cfg.m)) {
    if (((((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) < cfg.r_rows) & ((cfg.r_off + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u)))) < cfg.d))) {
    f0 = phi[(((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) * cfg.d) + (cfg.r_off + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u)))))];
    }
    if ((((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) < cfg.d)) {
    f1 = phi[(((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) * cfg.d) + ((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))))];
    }
    }
    wg_a[(((((lid.y * 16u) + lid.x) + (k * 256u))) & 1023u)] = f0;
    wg_b[(((((lid.y * 16u) + lid.x) + (k * 256u))) & 1023u)] = f1;
    }
    workgroupBarrier();
    i1 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    f2 = wg_a[((((k * 64u) + lid.y)) & 1023u)];
    f3 = wg_a[(((((k * 64u) + lid.y) + 16u)) & 1023u)];
    f4 = wg_a[(((((k * 64u) + lid.y) + 32u)) & 1023u)];
    f5 = wg_a[(((((k * 64u) + lid.y) + 48u)) & 1023u)];
    f6 = wg_b[((((k * 64u) + lid.x)) & 1023u)];
    f7 = wg_b[(((((k * 64u) + lid.x) + 16u)) & 1023u)];
    f8 = wg_b[(((((k * 64u) + lid.x) + 32u)) & 1023u)];
    f9 = wg_b[(((((k * 64u) + lid.x) + 48u)) & 1023u)];
    pa[0u] = (pa[0u] + (f2 * f6));
    pa[1u] = (pa[1u] + (f2 * f7));
    pa[2u] = (pa[2u] + (f2 * f8));
    pa[3u] = (pa[3u] + (f2 * f9));
    pa[4u] = (pa[4u] + (f3 * f6));
    pa[5u] = (pa[5u] + (f3 * f7));
    pa[6u] = (pa[6u] + (f3 * f8));
    pa[7u] = (pa[7u] + (f3 * f9));
    pa[8u] = (pa[8u] + (f4 * f6));
    pa[9u] = (pa[9u] + (f4 * f7));
    pa[10u] = (pa[10u] + (f4 * f8));
    pa[11u] = (pa[11u] + (f4 * f9));
    pa[12u] = (pa[12u] + (f5 * f6));
    pa[13u] = (pa[13u] + (f5 * f7));
    pa[14u] = (pa[14u] + (f5 * f8));
    pa[15u] = (pa[15u] + (f5 * f9));
    }
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if (((((wid.y * 64u + lid.y) + min(k, 3u) * 16u) < cfg.r_rows) & ((cfg.r_off + ((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) < cfg.d))) {
    i2 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((wid.x * 64u + lid.x) + min(k2, 3u) * 16u) < cfg.d)) {
    if (((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) < cfg.r_rows) & ((((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)) < cfg.d) & ((cfg.r_off + (((wid.y * 64u + lid.y) + min(k, 3u) * 16u))) < cfg.d)) { h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] = h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] + pa[((((k * 4u) + k2)) & 63u)]; }
    }
    }
    }
    }
}
