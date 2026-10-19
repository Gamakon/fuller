// grid 8 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (k = 0u; k < 4u; k = k + 1u) {
    pv[((k) & 15u)] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + ((((lid.y * 8u) + lid.x) + (k * 64u)) / 32u)) < cfg.m)) {
    if ((((wid.x * 32u) + ((((lid.y * 8u) + lid.x) + (k * 64u)) % 32u)) < cfg.d)) {
    f0 = phi[(((t + ((((lid.y * 8u) + lid.x) + (k * 64u)) / 32u)) * cfg.d) + ((wid.x * 32u) + ((((lid.y * 8u) + lid.x) + (k * 64u)) % 32u)))];
    }
    if (((((wid.y * 32u) + ((((lid.y * 8u) + lid.x) + (k * 64u)) % 32u)) < cfg.r_rows) & ((cfg.r_off + ((wid.y * 32u) + ((((lid.y * 8u) + lid.x) + (k * 64u)) % 32u))) < cfg.d))) {
    f1 = phi[(((t + ((((lid.y * 8u) + lid.x) + (k * 64u)) / 32u)) * cfg.d) + (cfg.r_off + ((wid.y * 32u) + ((((lid.y * 8u) + lid.x) + (k * 64u)) % 32u))))];
    }
    }
    wg_a[(((((lid.y * 8u) + lid.x) + (k * 64u))) & 511u)] = f0;
    wg_b[(((((lid.y * 8u) + lid.x) + (k * 64u))) & 511u)] = f1;
    }
    workgroupBarrier();
    for (k = 0u; k < 16u; k = k + 1u) {
    v0 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 32u) + (lid.x * 4u))) & 511u)], wg_a[(((((k * 32u) + (lid.x * 4u)) + 1u)) & 511u)]), wg_a[(((((k * 32u) + (lid.x * 4u)) + 2u)) & 511u)]), wg_a[(((((k * 32u) + (lid.x * 4u)) + 3u)) & 511u)]);
    v1 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[((((k * 32u) + (lid.y * 4u))) & 511u)], wg_b[(((((k * 32u) + (lid.y * 4u)) + 1u)) & 511u)]), wg_b[(((((k * 32u) + (lid.y * 4u)) + 2u)) & 511u)]), wg_b[(((((k * 32u) + (lid.y * 4u)) + 3u)) & 511u)]);
    pv[0u] = (pv[0u] + (vec4<f32>(v1.x) * v0));
    pv[1u] = (pv[1u] + (vec4<f32>(v1.y) * v0));
    pv[2u] = (pv[2u] + (vec4<f32>(v1.z) * v0));
    pv[3u] = (pv[3u] + (vec4<f32>(v1.w) * v0));
    }
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if (((((gid.y * 4u) + min(k, 3u)) < cfg.r_rows) & ((cfg.r_off + ((gid.y * 4u) + min(k, 3u))) < cfg.d))) {
    v2 = pv[((k) & 15u)];
    if (((gid.x * 4u) < cfg.d)) {
    if (((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows) & (((gid.x * 4u)) < cfg.d) & ((cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d)) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + ((gid.x * 4u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + ((gid.x * 4u)))] + v2.x; }
    }
    if ((((gid.x * 4u) + 1u) < cfg.d)) {
    if (((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows) & ((((gid.x * 4u) + 1u)) < cfg.d) & ((cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d)) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 1u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 1u)))] + v2.y; }
    }
    if ((((gid.x * 4u) + 2u) < cfg.d)) {
    if (((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows) & ((((gid.x * 4u) + 2u)) < cfg.d) & ((cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d)) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 2u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 2u)))] + v2.z; }
    }
    if ((((gid.x * 4u) + 3u) < cfg.d)) {
    if (((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows) & ((((gid.x * 4u) + 3u)) < cfg.d) & ((cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d)) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 3u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 3u)))] + v2.w; }
    }
    }
    }
}
