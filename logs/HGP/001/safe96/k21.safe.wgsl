// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 256>;
var<workgroup> wg_b: array<f32, 256>;
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
    var t: u32 = 0u;
    var k: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 8u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + ((((lid.y * 16u) + lid.x)) >> ((5u) & 31u))) < cfg.m)) {
    if ((((wid.x * 32u) + ((((lid.y * 16u) + lid.x)) & (31u))) < cfg.d)) {
    f0 = phi[(((t + ((((lid.y * 16u) + lid.x)) >> ((5u) & 31u))) * cfg.d) + ((wid.x * 32u) + ((((lid.y * 16u) + lid.x)) & (31u))))];
    }
    if (((((wid.y * 32u) + ((((lid.y * 16u) + lid.x)) & (31u))) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 32u) + ((((lid.y * 16u) + lid.x)) & (31u)))) < cfg.d))) {
    f1 = phi[(((t + ((((lid.y * 16u) + lid.x)) >> ((5u) & 31u))) * cfg.d) + (cfg.r_off + ((wid.y * 32u) + ((((lid.y * 16u) + lid.x)) & (31u)))))];
    }
    }
    wg_a[((((((((lid.y * 16u) + lid.x)) >> ((5u) & 31u)) * 32u) + ((((lid.y * 16u) + lid.x)) & (31u)))) & 255u)] = f0;
    wg_b[((((((((lid.y * 16u) + lid.x)) >> ((5u) & 31u)) * 32u) + ((((lid.y * 16u) + lid.x)) & (31u)))) & 255u)] = f1;
    workgroupBarrier();
    for (k = 0u; k < 8u; k = k + 1u) {
    f2 = wg_b[((((k * 32u) + (lid.y * 2u))) & 255u)];
    f3 = wg_b[(((((k * 32u) + (lid.y * 2u)) + 1u)) & 255u)];
    f4 = wg_a[((((k * 32u) + (lid.x * 2u))) & 255u)];
    f5 = wg_a[(((((k * 32u) + (lid.x * 2u)) + 1u)) & 255u)];
    f6 = (f6 + (f2 * f4));
    f7 = (f7 + (f2 * f5));
    f8 = (f8 + (f3 * f4));
    f9 = (f9 + (f3 * f5));
    }
    workgroupBarrier();
    }
    if (((gid.y * 2u) < cfg.r_rows)) {
    if (((gid.x * 2u) < cfg.d)) {
    if (((gid.y * 2u)) < cfg.r_rows && ((gid.x * 2u)) < cfg.d && (cfg.r_off + ((gid.y * 2u))) < cfg.d) { h[(((gid.y * 2u)) * cfg.d + ((gid.x * 2u)))] = h[(((gid.y * 2u)) * cfg.d + ((gid.x * 2u)))] + f6; }
    }
    if ((((gid.x * 2u) + 1u) < cfg.d)) {
    if (((gid.y * 2u)) < cfg.r_rows && (((gid.x * 2u) + 1u)) < cfg.d && (cfg.r_off + ((gid.y * 2u))) < cfg.d) { h[(((gid.y * 2u)) * cfg.d + (((gid.x * 2u) + 1u)))] = h[(((gid.y * 2u)) * cfg.d + (((gid.x * 2u) + 1u)))] + f7; }
    }
    }
    if ((((gid.y * 2u) + 1u) < cfg.r_rows)) {
    if (((gid.x * 2u) < cfg.d)) {
    if ((((gid.y * 2u) + 1u)) < cfg.r_rows && ((gid.x * 2u)) < cfg.d && (cfg.r_off + (((gid.y * 2u) + 1u))) < cfg.d) { h[((((gid.y * 2u) + 1u)) * cfg.d + ((gid.x * 2u)))] = h[((((gid.y * 2u) + 1u)) * cfg.d + ((gid.x * 2u)))] + f8; }
    }
    if ((((gid.x * 2u) + 1u) < cfg.d)) {
    if ((((gid.y * 2u) + 1u)) < cfg.r_rows && (((gid.x * 2u) + 1u)) < cfg.d && (cfg.r_off + (((gid.y * 2u) + 1u))) < cfg.d) { h[((((gid.y * 2u) + 1u)) * cfg.d + (((gid.x * 2u) + 1u)))] = h[((((gid.y * 2u) + 1u)) * cfg.d + (((gid.x * 2u) + 1u)))] + f9; }
    }
    }
}
