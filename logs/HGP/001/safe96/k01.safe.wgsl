// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
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
    var t: u32 = 0u;
    var k: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 16u) {
    for (k = 0u; k < 2u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + ((((lid.y * 16u) + lid.x) + (256u * k)) / 32u)) < cfg.m)) {
    if (((((wid.y * 32u) + ((((lid.y * 16u) + lid.x) + (256u * k)) % 32u)) < cfg.r_rows) & ((cfg.r_off + ((wid.y * 32u) + ((((lid.y * 16u) + lid.x) + (256u * k)) % 32u))) < cfg.d))) {
    f0 = phi[(((t + ((((lid.y * 16u) + lid.x) + (256u * k)) / 32u)) * cfg.d) + (cfg.r_off + ((wid.y * 32u) + ((((lid.y * 16u) + lid.x) + (256u * k)) % 32u))))];
    }
    if ((((wid.x * 32u) + ((((lid.y * 16u) + lid.x) + (256u * k)) % 32u)) < cfg.d)) {
    f1 = phi[(((t + ((((lid.y * 16u) + lid.x) + (256u * k)) / 32u)) * cfg.d) + ((wid.x * 32u) + ((((lid.y * 16u) + lid.x) + (256u * k)) % 32u)))];
    }
    }
    wg_a[(((((lid.y * 16u) + lid.x) + (256u * k))) & 511u)] = f0;
    wg_b[(((((lid.y * 16u) + lid.x) + (256u * k))) & 511u)] = f1;
    }
    workgroupBarrier();
    i0 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    f2 = wg_a[((((k * 32u) + lid.y)) & 511u)];
    f3 = wg_a[(((((k * 32u) + lid.y) + 16u)) & 511u)];
    f4 = wg_b[((((k * 32u) + lid.x)) & 511u)];
    f5 = wg_b[(((((k * 32u) + lid.x) + 16u)) & 511u)];
    f6 = (f6 + (f2 * f4));
    f7 = (f7 + (f2 * f5));
    f8 = (f8 + (f3 * f4));
    f9 = (f9 + (f3 * f5));
    }
    workgroupBarrier();
    }
    if (((((wid.y * 32u + lid.y) < cfg.r_rows) & ((cfg.r_off + (wid.y * 32u + lid.y)) < cfg.d)) & ((wid.x * 32u + lid.x) < cfg.d))) {
    if ((((wid.y * 32u + lid.y)) < cfg.r_rows) & (((wid.x * 32u + lid.x)) < cfg.d) & ((cfg.r_off + ((wid.y * 32u + lid.y))) < cfg.d)) { h[(((wid.y * 32u + lid.y)) * cfg.d + ((wid.x * 32u + lid.x)))] = h[(((wid.y * 32u + lid.y)) * cfg.d + ((wid.x * 32u + lid.x)))] + f6; }
    }
    if (((((wid.y * 32u + lid.y) < cfg.r_rows) & ((cfg.r_off + (wid.y * 32u + lid.y)) < cfg.d)) & (((wid.x * 32u + lid.x) + 16u) < cfg.d))) {
    if ((((wid.y * 32u + lid.y)) < cfg.r_rows) & ((((wid.x * 32u + lid.x) + 16u)) < cfg.d) & ((cfg.r_off + ((wid.y * 32u + lid.y))) < cfg.d)) { h[(((wid.y * 32u + lid.y)) * cfg.d + (((wid.x * 32u + lid.x) + 16u)))] = h[(((wid.y * 32u + lid.y)) * cfg.d + (((wid.x * 32u + lid.x) + 16u)))] + f7; }
    }
    if ((((((wid.y * 32u + lid.y) + 16u) < cfg.r_rows) & ((cfg.r_off + ((wid.y * 32u + lid.y) + 16u)) < cfg.d)) & ((wid.x * 32u + lid.x) < cfg.d))) {
    if (((((wid.y * 32u + lid.y) + 16u)) < cfg.r_rows) & (((wid.x * 32u + lid.x)) < cfg.d) & ((cfg.r_off + (((wid.y * 32u + lid.y) + 16u))) < cfg.d)) { h[((((wid.y * 32u + lid.y) + 16u)) * cfg.d + ((wid.x * 32u + lid.x)))] = h[((((wid.y * 32u + lid.y) + 16u)) * cfg.d + ((wid.x * 32u + lid.x)))] + f8; }
    }
    if ((((((wid.y * 32u + lid.y) + 16u) < cfg.r_rows) & ((cfg.r_off + ((wid.y * 32u + lid.y) + 16u)) < cfg.d)) & (((wid.x * 32u + lid.x) + 16u) < cfg.d))) {
    if (((((wid.y * 32u + lid.y) + 16u)) < cfg.r_rows) & ((((wid.x * 32u + lid.x) + 16u)) < cfg.d) & ((cfg.r_off + (((wid.y * 32u + lid.y) + 16u))) < cfg.d)) { h[((((wid.y * 32u + lid.y) + 16u)) * cfg.d + (((wid.x * 32u + lid.x) + 16u)))] = h[((((wid.y * 32u + lid.y) + 16u)) * cfg.d + (((wid.x * 32u + lid.x) + 16u)))] + f9; }
    }
}
